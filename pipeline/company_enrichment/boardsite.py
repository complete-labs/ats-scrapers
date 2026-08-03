"""Recover a corporate domain by reading the tenant's ATS board page.

The last resort in :mod:`.resolve` for tenants whose name matches
nothing. Almost every ATS board renders the employer's logo or a "visit
our website" link pointing at the company's own site, so one cheap GET
often yields the domain that unlocks the whole PDL record.

Uses ``ats_scrapers.fetch.Fetcher`` for retries, backoff, ``Retry-After``
handling, and proxy support rather than reimplementing them. Requests
are bounded by a semaphore and the whole stage is skippable — it is
strictly additive, and a failed fetch just leaves the tenant unresolved.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin

import polars as pl

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import (
    corporate_domain,
    is_ats_host,
    name_key,
    normalize_domain,
)

logger = logging.getLogger(__name__)

CONCURRENCY = 12
PER_REQUEST_TIMEOUT = 20.0

_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
# Canonical / OG tags are the most reliable single signal when present.
_META_URL_RE = re.compile(
    r"""<(?:link[^>]+rel=["']canonical["'][^>]+href|"""
    r"""meta[^>]+property=["']og:url["'][^>]+content)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)

# Hosts that appear on every board and never identify the employer.
_NOISE_DOMAINS = frozenset(
    {
        "google.com", "www.google.com", "gstatic.com", "googleapis.com",
        "gravatar.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
        "linkedin.com", "youtube.com", "github.com", "medium.com", "glassdoor.com",
        "indeed.com", "apple.com", "microsoft.com", "adobe.com", "cloudflare.com",
        "amazonaws.com", "cloudfront.net", "jsdelivr.net", "unpkg.com",
        "fontawesome.com", "bootstrapcdn.com", "typekit.net", "hotjar.com",
        "segment.com", "sentry.io", "intercom.com", "hubspot.com", "vimeo.com",
        "w3.org", "schema.org", "mozilla.org", "wikipedia.org", "tiktok.com",
        "threads.net", "bsky.app", "mailchimp.com", "calendly.com", "notion.so",
        "eeoc.gov", "dol.gov", "sec.gov", "ada.gov", "uscis.gov",
    }
)


def _board_url(row: dict[str, object]) -> str:
    url = str(row.get("url") or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    if url:
        return f"https://{url}"
    return ""


def _score_candidate(domain: str, key: str) -> int:
    """Prefer a domain whose label overlaps the company name."""
    if not domain:
        return -1
    label = domain.split(".")[0].replace("-", "")
    if not key:
        return 0
    if label == key:
        return 100
    if label and (label in key or key in label):
        return 60
    return 10


def _extract_domain(html: str, base_url: str, company_name: str) -> str:
    key = name_key(company_name)
    best_domain, best_score = "", -1

    for match in _META_URL_RE.finditer(html):
        domain = corporate_domain(match.group(1))
        if domain and domain not in _NOISE_DOMAINS:
            score = _score_candidate(domain, key) + 25  # canonical outranks body links
            if score > best_score:
                best_domain, best_score = domain, score

    for match in _HREF_RE.finditer(html):
        href = match.group(1).strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        host = normalize_domain(absolute)
        if not host or is_ats_host(host) or host in _NOISE_DOMAINS:
            continue
        if any(host.endswith("." + noise) for noise in _NOISE_DOMAINS):
            continue
        domain = corporate_domain(absolute)
        if not domain:
            continue
        score = _score_candidate(domain, key)
        if score > best_score:
            best_domain, best_score = domain, score

    # A bare "some outbound link exists" is not evidence of identity.
    return best_domain if best_score >= 60 else ""


async def _fetch_one(
    fetcher: object, semaphore: asyncio.Semaphore, row: dict[str, object]
) -> dict[str, object] | None:
    url = _board_url(row)
    if not url:
        return None
    async with semaphore:
        try:
            html = await fetcher.get_text(url)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("board fetch failed for %s/%s: %s", row["ats"], row["slug"], exc)
            return None
    domain = _extract_domain(html, url, str(row.get("name") or ""))
    if not domain:
        return None
    return {"ats": row["ats"], "slug": row["slug"], "domain_hint": domain}


async def _recover_async(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    from ats_scrapers.fetch import Fetcher

    semaphore = asyncio.Semaphore(CONCURRENCY)
    # One attempt only: this is a best-effort enrichment over thousands
    # of boards, many of them dead, and the default backoff would turn a
    # long tail of 404s into a multi-hour stage.
    async with Fetcher(
        label="company-enrichment board page",
        timeout=PER_REQUEST_TIMEOUT,
        retries=1,
    ) as fetcher:
        results = await asyncio.gather(
            *(_fetch_one(fetcher, semaphore, row) for row in rows),
            return_exceptions=True,
        )
    return [r for r in results if isinstance(r, dict)]


def recover_domains(cohort_subset: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    """Fetch board pages for unresolved tenants and extract their domains.

    Results are cached to disk; a re-run only fetches tenants not seen
    before, so this expensive stage is paid once.
    """
    empty = pl.DataFrame(
        schema={"ats": pl.String, "slug": pl.String, "domain_hint": pl.String}
    )
    if cohort_subset.is_empty():
        return empty

    cache_path = config.CACHE_DIR / "board_domains.parquet"
    cached = pl.read_parquet(cache_path) if cache_path.exists() else empty
    todo = cohort_subset.join(cached.select("ats", "slug"), on=["ats", "slug"], how="anti")
    if limit is not None:
        todo = todo.head(limit)

    if todo.is_empty():
        logger.info("board fallback: all %d tenants already cached", cached.height)
        return cached.filter(pl.col("domain_hint") != "")

    rows = todo.select("ats", "slug", "name", "url").to_dicts()
    logger.info("board fallback: fetching %d board pages", len(rows))
    found = asyncio.run(_recover_async(rows))
    logger.info("board fallback: recovered %d domains", len(found))

    # Negative results are cached too, so failures are not retried.
    found_keys = {(r["ats"], r["slug"]) for r in found}
    misses = [
        {"ats": r["ats"], "slug": r["slug"], "domain_hint": ""}
        for r in rows
        if (r["ats"], r["slug"]) not in found_keys
    ]
    fresh = pl.DataFrame(found + misses, schema=empty.schema)
    combined = pl.concat([cached, fresh], how="vertical").unique(
        subset=["ats", "slug"], keep="last"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(cache_path)
    return combined.filter(pl.col("domain_hint") != "")

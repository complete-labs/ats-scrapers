"""Read a company's own homepage for its description and careers link.

The primary description source. A company's ``og:description`` is the
one sentence it wrote to introduce itself to strangers, which is exactly
the datapoint wanted, and it comes with the careers link sitting in the
same navigation bar — so one GET answers both questions at once.

Yield on a hand-checked stratified sample of 25 tenants: 25 homepages
reachable, 16 with a usable description and 16 with a careers link. That
is about two thirds of the tenants that have a resolved ``domain``.

The mechanics are :mod:`.boardsite`'s: ``ats_scrapers.fetch.Fetcher`` for
retries and proxying, a semaphore to bound concurrency, one attempt per
host because the long tail of dead domains would otherwise dominate the
runtime, and a cache that stores misses so a re-run never pays for the
same failure twice. The stage is strictly additive and skippable.

The text belongs to the company, so :mod:`.profile` records the page it
was taken from in ``company_description_url`` rather than presenting it
as the pipeline's own words.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit

import polars as pl

from pipeline.company_enrichment import blurb, config
from pipeline.company_enrichment.normalize import corporate_domain, normalize_domain

logger = logging.getLogger(__name__)

CONCURRENCY = 12
PER_REQUEST_TIMEOUT = 20.0

# Homepages of large companies routinely ship a megabyte of inlined
# state. Everything wanted here is in the head or the nav, so the scan
# is bounded rather than letting one page dominate the stage's CPU.
MAX_HTML_CHARS = 600_000

SCHEMA: dict[str, pl.DataType] = {
    "ats": pl.String,
    "slug": pl.String,
    "domain": pl.String,
    "site_url": pl.String,
    "site_title": pl.String,
    "site_description": pl.String,
    "site_careers_url": pl.String,
    "site_fetched_at": pl.String,
}

_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(
    r"""([a-zA-Z_:][-\w:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"""
)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_LD_JSON = re.compile(
    r"""<script[^>]+type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
    re.IGNORECASE | re.DOTALL,
)
_ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_ANY_TAG = re.compile(r"<[^>]+>")

# Meta names carrying a self-description, best first. `og:description`
# is written for sharing and is the most consistently a real sentence;
# the bare `description` tag is more often keyword-stuffed for search.
_DESCRIPTION_KEYS = ("og:description", "description", "twitter:description")

# Path segments that mark a careers destination.
_CAREERS_PATH = re.compile(
    r"(?i)(?:^|/)(?:"
    r"careers?|jobs?|job-openings|open-roles|open-positions|opportunities"
    r"|join-us|joinus|join-our-team|work-with-us|work-for-us|work-here"
    r"|employment|life-at-[\w-]+|why-[\w-]+"
    r")(?:/|$|[?#.])"
)
# Anchor labels for the same. Corroboration only — on its own, link text
# picked out `quantummetric.com/digital-analytics/experience-analytics`
# from an "opportunities" anchor, which is a product page.
_CAREERS_LABEL = re.compile(
    r"(?i)^\W*(?:careers?|jobs?|join\s+us|join\s+our\s+team|work\s+with\s+us"
    r"|we(?:'|\u2019)?re\s+hiring|open\s+roles|open\s+positions"
    r"|life\s+at\s+.{0,30}|view\s+(?:all\s+)?(?:jobs|openings))\W*$"
)

_SCHEMA_ORG_TYPES = frozenset(
    {
        "organization",
        "corporation",
        "localbusiness",
        "ngo",
        "educationalorganization",
        "governmentorganization",
        "medicalorganization",
        "onlinebusiness",
        "performinggroup",
        "sportsorganization",
    }
)


def _attrs(tag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _ATTR.finditer(tag):
        value = match.group(2) or match.group(3) or match.group(4) or ""
        out[match.group(1).lower()] = value
    return out


def extract_title(html: str) -> str:
    match = _TITLE.search(html)
    return blurb.tidy(match.group(1)) if match else ""


def _jsonld_description(html: str) -> str:
    """``description`` from a schema.org Organization node, if present.

    Sites that publish structured data usually give a fuller and less
    marketing-shaped sentence here than in their meta tags, but only a
    minority publish it, so it sits behind them as a fallback.
    """

    def walk(node: object) -> str:
        if isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
            return ""
        if not isinstance(node, dict):
            return ""
        if "@graph" in node:
            found = walk(node["@graph"])
            if found:
                return found
        types = node.get("@type")
        types = types if isinstance(types, list) else [types]
        named = {str(t).lower() for t in types if t}
        if named & _SCHEMA_ORG_TYPES:
            text = blurb.tidy(node.get("description"))
            if len(text) >= config.DESCRIPTION_MIN_CHARS:
                return text
        return ""

    for match in _LD_JSON.finditer(html):
        try:
            payload = json.loads(match.group(1).strip())
        except ValueError:
            continue
        found = walk(payload)
        if found:
            return found
    return ""


def extract_description(html: str) -> str:
    """The page's self-description, or ``""``.

    Meta content arrives HTML-escaped often enough that skipping
    :func:`blurb.tidy` here would publish ``America&#39;s leading
    health solutions company`` verbatim.
    """
    found: dict[str, str] = {}
    for tag in _META_TAG.finditer(html):
        attributes = _attrs(tag.group(0))
        key = (attributes.get("property") or attributes.get("name") or "").lower()
        if key in _DESCRIPTION_KEYS and key not in found:
            text = blurb.tidy(attributes.get("content"))
            if len(text) >= config.DESCRIPTION_MIN_CHARS:
                found[key] = text
    for key in _DESCRIPTION_KEYS:
        if key in found:
            return found[key]
    return _jsonld_description(html)


def _same_company(host: str, domain: str) -> tuple[bool, bool]:
    """``(belongs to this company, is a careers subdomain)`` for ``host``.

    ``corporate_domain`` does the work: it peels ``careers.``/``jobs.``
    back to the registrable domain and returns ``""`` for ATS hosts, so
    a board link is rejected here without a separate check. The board
    URL is already known from the directory; what this stage is after is
    the company's own page.
    """
    if not host:
        return True, False
    normalized = normalize_domain(host)
    if not normalized:
        return False, False
    peeled = corporate_domain(normalized)
    if not peeled or peeled != domain:
        return False, False
    return True, peeled != normalized


def extract_careers_url(html: str, base_url: str, domain: str) -> str:
    """The company's own careers page linked from ``html``, or ``""``.

    Requires the link to point somewhere that is recognisably about
    careers *and* to stay on the company's own domain. Either signal
    alone is too weak: a path check on its own follows a partner's job
    board, and a label check on its own follows a product page that
    happens to be called "Opportunities".
    """
    best_url, best_score = "", 0
    for match in _ANCHOR.finditer(html):
        href = (_attrs(match.group(1)).get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        absolute = urljoin(base_url, href)
        if not absolute.startswith(("http://", "https://")):
            continue
        parts = urlsplit(absolute)
        ours, careers_host = _same_company(parts.hostname or "", domain)
        if not ours:
            continue

        path_hit = bool(_CAREERS_PATH.search(parts.path or "/"))
        if not (path_hit or careers_host):
            continue

        label = blurb.tidy(_ANY_TAG.sub(" ", match.group(2)))
        score = 40 * path_hit + 40 * careers_host + 20 * bool(_CAREERS_LABEL.match(label))
        # Prefer the shallowest match: "/careers" is the careers page,
        # "/careers/engineering/openings/123" is one listing inside it.
        score -= 5 * max(0, len([p for p in (parts.path or "").split("/") if p]) - 1)
        if score > best_score:
            best_url, best_score = absolute, score
    return best_url


def _candidate_urls(domain: str) -> list[str]:
    # Plenty of hosts serve the apex as a bare redirect stub with no
    # meta tags, or refuse it outright, so `www` is tried first.
    return [f"https://www.{domain}", f"https://{domain}"]


async def _fetch_one(
    fetcher: object, semaphore: asyncio.Semaphore, row: dict[str, object]
) -> dict[str, object]:
    domain = str(row.get("domain") or "")
    record: dict[str, object] = {
        "ats": row["ats"],
        "slug": row["slug"],
        "domain": domain,
        "site_url": "",
        "site_title": "",
        "site_description": "",
        "site_careers_url": "",
        "site_fetched_at": datetime.now(tz=UTC).date().isoformat(),
    }
    if not domain:
        return record

    async with semaphore:
        html, used = "", ""
        for candidate in _candidate_urls(domain):
            try:
                html = await fetcher.get_text(candidate)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("homepage fetch failed for %s: %s", candidate, exc)
                continue
            used = candidate
            break
    if not html:
        return record

    html = html[:MAX_HTML_CHARS]
    record["site_url"] = used
    record["site_title"] = extract_title(html)
    record["site_description"] = extract_description(html)
    record["site_careers_url"] = extract_careers_url(html, used, domain)
    return record


async def _crawl_async(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    from ats_scrapers.fetch import Fetcher

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with Fetcher(
        label="company-enrichment homepage",
        timeout=PER_REQUEST_TIMEOUT,
        retries=1,
    ) as fetcher:
        results = await asyncio.gather(
            *(_fetch_one(fetcher, semaphore, row) for row in rows),
            return_exceptions=True,
        )
    return [r for r in results if isinstance(r, dict)]


def load() -> pl.DataFrame:
    """Everything the crawl has ever found, hits and misses alike."""
    if not config.COMPANY_SITE_CACHE.exists():
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(config.COMPANY_SITE_CACHE)


def crawl(resolved: pl.DataFrame, *, limit: int | None = None) -> pl.DataFrame:
    """Fetch the homepage of every tenant with a domain, and cache it.

    The cache key includes the domain, so a tenant whose resolution has
    since changed is re-fetched against its new domain rather than
    keeping a description written by the previous match.
    """
    if resolved.is_empty():
        return pl.DataFrame(schema=SCHEMA)

    targets = resolved.filter(
        pl.col("domain").is_not_null() & (pl.col("domain") != "")
    ).select("ats", "slug", "domain")
    if targets.is_empty():
        logger.warning("no tenants carry a domain — run the resolve stage first")
        return pl.DataFrame(schema=SCHEMA)

    cached = load()
    todo = targets.join(
        cached.select("ats", "slug", "domain"), on=["ats", "slug", "domain"], how="anti"
    )
    if limit is not None:
        todo = todo.head(limit)

    if todo.is_empty():
        logger.info("homepage crawl: all %d tenants already cached", targets.height)
        return cached

    rows = todo.to_dicts()
    logger.info("homepage crawl: fetching %d company sites", len(rows))
    found = asyncio.run(_crawl_async(rows))
    with_text = sum(1 for r in found if r["site_description"])
    with_link = sum(1 for r in found if r["site_careers_url"])
    logger.info(
        "homepage crawl: %d descriptions, %d careers links from %d fetches",
        with_text,
        with_link,
        len(found),
    )

    fresh = pl.DataFrame(found, schema=SCHEMA)
    combined = pl.concat([cached, fresh], how="vertical").unique(
        subset=["ats", "slug"], keep="last"
    )
    config.COMPANY_SITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(config.COMPANY_SITE_CACHE)
    return combined

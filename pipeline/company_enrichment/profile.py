"""What the company does, and where to apply.

Three things are assembled here, all keyed on ``(ats, slug)``.

**A description.** Four sources, in descending order of how directly
they speak for the company:

``company_site``
    The ``og:description`` from the company's own homepage. Its own
    words, one sentence, written to introduce itself. See
    :mod:`.companysite`.
``posting_boilerplate``
    The copy it repeats at the top of every job posting. Also its own
    words, and free — the text is already in the jobs snapshot — but
    longer and shaped for candidates. See :mod:`.boilerplate`.
``sec_10k``
    The opening of Item 1, Business. Filed under penalty, so it is the
    most accountable of the four, but it exists only for registrants
    and reads like a filing.
``wikidata``
    A third-party one-liner, usually of the form "American aerospace
    and defence company". Short, neutral, and it names a location.

**A guard against describing the wrong company.** Three of the four
sources are reached through a fuzzy match — the homepage through a
domain, the 10-K through a CIK, Wikidata through a name — so a
description can be confidently worded and about someone else. Every
description is checked for the tenant's own name, and a source reached
through an already-suspect match is *skipped* rather than emitted, which
lets the waterfall fall through to a source that is tied to the tenant
by construction. Only ``posting_boilerplate`` is immune: those are
literally this tenant's postings.

**Two careers URLs**, which are different things and both wanted. The
ATS board is where the postings live and the directory already knows it;
the company's own careers page is what a human would call the careers
page. The board URL is corroborated by re-resolving it, and where the
directory has nothing usable it is recovered from a posting URL instead.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from urllib.parse import urlsplit

import polars as pl

from ats_scrapers.enrichment.uslocation import STATE_NAME_TO_ABBR
from pipeline.company_enrichment import (
    blurb,
    boilerplate,
    companysite,
    config,
    registrant,
    sechttp,
)
from pipeline.company_enrichment.normalize import name_key, normalize_domain

logger = logging.getLogger(__name__)

# Highest wins. See the module docstring for why they rank this way.
SOURCE_RANK: dict[str, int] = {
    "company_site": 40,
    "posting_boilerplate": 30,
    "sec_10k": 20,
    "wikidata": 10,
}

# The generic length floor exists to reject scraped page furniture
# ("Home", "Welcome"). A Wikidata description is a curated one-liner, so
# a short one is the expected shape rather than a sign of junk —
# "American aerospace and defence company" answers the question in 38
# characters and dropping it would be pedantry.
WIKIDATA_MIN_CHARS = 20
MIN_CHARS_BY_SOURCE: dict[str, int] = {"wikidata": WIKIDATA_MIN_CHARS}

DESCRIPTION_COLUMNS = (
    "company_description",
    "company_description_source",
    "company_description_url",
    "company_description_as_of",
    "company_description_name_corroborated",
)

SCHEMA: dict[str, pl.DataType] = {
    "ats": pl.String,
    "slug": pl.String,
    "company_description": pl.String,
    "company_description_source": pl.String,
    "company_description_url": pl.String,
    "company_description_as_of": pl.String,
    "company_description_name_corroborated": pl.Boolean,
    "careers_url": pl.String,
    "careers_url_source": pl.String,
    "careers_url_verified": pl.Boolean,
    "company_careers_url": pl.String,
    "company_careers_url_source": pl.String,
    "headquarters": pl.String,
    "headquarters_source": pl.String,
}

# A name match below this is weak enough that anything reached through
# it needs the description to name the company before it is published.
# Mirrors the `weak_name_match` threshold in `assemble._quality_flags`.
SUSPECT_MATCH_SCORE = 95.0

_ABBR_TO_STATE_NAME = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}


# --- descriptions from SEC 10-K filings --------------------------------

_SCRIPT = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAG = re.compile(r"<[^>]+>")
# The heading, in the several ways filers punctuate it.
_ITEM_1 = re.compile(r"(?i)\bitem\s*1\s*[.:\-\u2013\u2014]?\s*business\b")
_ITEM_1A = re.compile(r"(?i)\bitem\s*1a\b")
# A self-describing opening sentence. Filers overwhelmingly use one of
# these two voices to start Item 1.
_SELF_INTRO = re.compile(
    r"(?i)^(?:we\s+(?:are|were|provide|design|develop|operate|offer|manufacture)"
    r"|our\s+company\b|the\s+company\s+(?:is|was|operates|provides)\b)"
)
_ITEM1_WINDOW = 8000

# Sub-headings filers put between "Item 1. Business" and the first real
# sentence. They carry no punctuation, so the sentence splitter glues
# them onto the text that follows: "General Headquartered in Louisville,
# Kentucky, Humana Inc. …".
_SECTION_HEADING = re.compile(
    r"(?i)^(?:general|overview|introduction|background|our\s+business"
    r"|the\s+business|business\s+overview|company\s+overview"
    r"|organi[sz]ational\s+history|corporate\s+history|our\s+company"
    r"|purpose\s*(?:&|and)\s*strategy|who\s+we\s+are)\b[\s:.\-\u2013\u2014]*"
)


def _visible_text(html: str) -> str:
    text = _SCRIPT.sub(" ", html)
    text = _TAG.sub(" ", text)
    return blurb.tidy(text)


def description_from_10k(html: str, names: tuple[str, ...]) -> str:
    """The opening of Item 1, Business, or ``""``.

    Every 10-K contains the phrase "Item 1. Business" at least twice —
    once in the table of contents and once at the section itself — so
    each occurrence is checked for a "Item 1A" following immediately
    behind it, which is the shape of a contents listing rather than of
    the section body.
    """
    text = _visible_text(html)
    if not text:
        return ""
    for anchor in _ITEM_1.finditer(text):
        window = text[anchor.end() : anchor.end() + _ITEM1_WINDOW].lstrip(" .:-\u2014")
        window = _SECTION_HEADING.sub("", window, count=1)
        following = _ITEM_1A.search(window)
        if following and following.start() < 400:
            continue
        picked: list[str] = []
        for sentence in blurb.sentences(window):
            if not picked:
                if len(sentence) < 60:
                    continue
                names_it = blurb.mentions_name(sentence, *names)
                if not (names_it or _SELF_INTRO.match(sentence)):
                    continue
                if blurb.is_legalese(sentence):
                    continue
            picked.append(sentence)
            if len(" ".join(picked)) >= config.DESCRIPTION_MAX_CHARS:
                break
        candidate = blurb.trim(blurb.tidy(" ".join(picked)))
        if blurb.acceptable(candidate):
            return candidate
    return ""


def descriptions_from_10k(ciks: list[int], names: dict[int, tuple[str, ...]]) -> pl.DataFrame:
    """Item 1 openings for each CIK, from the filings already on disk."""
    from pipeline.company_enrichment.teamsize import _latest_10k

    schema = {
        "cik": pl.Int64,
        "text": pl.String,
        "url": pl.String,
        "as_of": pl.String,
    }
    records: list[dict[str, object]] = []
    for index, cik in enumerate(ciks, start=1):
        if index % 100 == 0:
            logger.info("10-K descriptions: %d/%d", index, len(ciks))
        found = _latest_10k(cik)
        if not found:
            continue
        url, filed = found
        try:
            html = sechttp.get(url, suffix=".htm").decode("utf-8", errors="replace")
        except Exception:
            continue
        text = description_from_10k(html, names.get(cik, ()))
        if text:
            records.append(
                {"cik": cik, "text": text, "url": url, "as_of": filed[:10] or None}
            )
    return pl.DataFrame(records, schema=schema) if records else pl.DataFrame(schema=schema)


# --- descriptions from Wikidata ----------------------------------------

# Restricted to US business enterprises that publish a website. The
# unrestricted form ("every US entity with a description") is millions
# of rows and times the endpoint out; this returns ~52k in about 20s.
WIKIDATA_DESCRIPTION_QUERY = """
SELECT ?company ?companyLabel ?description ?cik ?website WHERE {
  ?company wdt:P31/wdt:P279* wd:Q4830453 ;
           wdt:P17 wd:Q30 ;
           wdt:P856 ?website ;
           schema:description ?description .
  FILTER(LANG(?description) = "en")
  OPTIONAL { ?company wdt:P5531 ?cik . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def fetch_wikidata_descriptions(*, force: bool = False) -> pl.DataFrame:
    """US companies with an English one-line description. CC0."""
    schema = {
        "wikidata_id": pl.String,
        "wikidata_description": pl.String,
        "cik": pl.Int64,
        "domain": pl.String,
        "name_key_core": pl.String,
        "wikidata_as_of": pl.String,
    }
    cache = config.CACHE_DIR / "wikidata_descriptions.json"
    if cache.exists() and not force:
        payload = json.loads(cache.read_text())
    else:
        import httpx

        logger.info("querying Wikidata for company descriptions")
        try:
            response = httpx.get(
                config.WIKIDATA_SPARQL_URL,
                params={"query": WIKIDATA_DESCRIPTION_QUERY},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": config.SEC_USER_AGENT,
                },
                timeout=300.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            # A tail source; the endpoint being busy must not fail a run.
            logger.warning("skipping Wikidata descriptions: %s", exc)
            return pl.DataFrame(schema=schema)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    as_of = date.fromtimestamp(cache.stat().st_mtime).isoformat()
    rows = payload["results"]["bindings"]

    def value(row: dict, key: str) -> str | None:
        return (row.get(key) or {}).get("value")

    def as_cik(raw: str | None) -> int | None:
        try:
            return int(str(raw)) if raw else None
        except ValueError:
            return None

    frame = pl.DataFrame(
        {
            "wikidata_id": [r["company"]["value"].rsplit("/", 1)[-1] for r in rows],
            "wikidata_description": [blurb.tidy(value(r, "description")) for r in rows],
            "cik": [as_cik(value(r, "cik")) for r in rows],
            "domain": [normalize_domain(value(r, "website")) for r in rows],
            "name_key_core": [name_key(value(r, "companyLabel") or "") for r in rows],
            "wikidata_as_of": [as_of] * len(rows),
        },
        schema=schema,
    ).filter(pl.col("wikidata_description").str.len_chars() >= WIKIDATA_MIN_CHARS)
    logger.info("wikidata: %d companies with a description", frame.height)
    # Both join keys collide, so a stable rule decides: the longest
    # description, then the lowest Q-id, which never depends on the
    # order rows happened to arrive in.
    return frame.sort(
        [pl.col("wikidata_description").str.len_chars(), pl.col("wikidata_id")],
        descending=[True, False],
    )


def _unique_on(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    """One row per non-empty ``key``, keeping the incoming sort order."""
    present = pl.col(key).is_not_null()
    if frame.schema[key] == pl.String:
        present = present & (pl.col(key) != "")
    return frame.filter(present).unique(
        subset=[key], keep="first", maintain_order=True
    )


# --- careers URLs -------------------------------------------------------


def _slug_tokens(slug: str) -> list[str]:
    """Identifying words in a tenant slug, scheme and TLD noise removed."""
    text = re.sub(r"^https?://", "", str(slug or "").lower())
    ignore = {"com", "www", "http", "https", "net", "org", "io", "co"}
    return [t for t in re.split(r"[^a-z0-9]+", text) if t and t not in ignore]


def _belongs_to(url: str, ats: str, slug: str) -> bool:
    """True when ``url`` is a board URL for this exact tenant.

    Both halves matter. ``resolve_careers_url`` confirms the URL is a
    board of the right ATS but not *whose* — on Workable it reads the
    short-link path ``/j/2B084BEA7B`` as a tenant called "j" — so the
    tenant's own slug has to be present in the URL as well.
    """
    from ats_scrapers.resolve import resolve_careers_url

    try:
        resolved = resolve_careers_url(url)
    except Exception:
        return False
    if resolved is None or str(resolved.ats.value) != ats:
        return False
    lowered = url.lower()
    tokens = _slug_tokens(slug)
    return bool(tokens) and all(token in lowered for token in tokens)


def board_url_from_posting(posting_url: str, ats: str, slug: str) -> str:
    """The board root a posting URL sits under, or ``""``.

    Every multi-tenant ATS puts the board at either the host root
    (``acme.breezy.hr``) or the first path segment
    (``jobs.lever.co/acme``), so both are tried. The host root is tried
    first because a subdomain tenant's first path segment is part of the
    posting, not the board — Pinpoint's ``/en/postings/<id>`` would
    otherwise leave the locale behind as ``…pinpointhq.com/en``. A
    path-based host with no segment does not resolve at all, so it falls
    through to the second candidate on its own.
    """
    if not posting_url:
        return ""
    parsed = urlsplit(posting_url if "//" in posting_url else f"https://{posting_url}")
    host = parsed.hostname or ""
    if not host:
        return ""
    segments = [s for s in parsed.path.split("/") if s]
    candidates = [f"https://{host}"]
    if segments:
        candidates.append(f"https://{host}/{segments[0]}")
    for candidate in candidates:
        if _belongs_to(candidate, ats, slug):
            return candidate
    return ""


def _careers_columns(row: dict[str, object]) -> tuple[str | None, str | None, bool]:
    """``(careers_url, source, verified)`` for one tenant."""
    ats = str(row.get("ats") or "")
    slug = str(row.get("slug") or "")
    directory = str(row.get("url") or "").strip()
    if directory and not directory.startswith(("http://", "https://")):
        directory = f"https://{directory}"

    if directory and _belongs_to(directory, ats, slug):
        return directory, "directory", True

    inferred = board_url_from_posting(
        str(row.get("sample_posting_url") or ""), ats, slug
    )
    if inferred:
        return inferred, "posting_url", True

    # A custom-domain careers site (Phenom, Avature) is a perfectly good
    # careers URL that no URL-shape rule can confirm. Keep it, say so.
    if directory:
        return directory, "directory", False
    return None, None, False


# --- headquarters -------------------------------------------------------


def _titlecase_place(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # PDL stores places lowercased; anything already mixed-case is left
    # alone so "LaGrange" and "DeKalb" survive.
    return text.title() if text.islower() else text


def _headquarters(row: dict[str, object]) -> tuple[str | None, str | None]:
    locality = _titlecase_place(row.get("locality"))
    region = _titlecase_place(row.get("region"))
    if locality or region:
        joined = ", ".join(p for p in (locality, region) if p)
        return joined, "pdl"

    city = _titlecase_place(row.get("registrant_city"))
    state = str(row.get("registrant_state") or "").strip().upper()
    # EDGAR files a postal code; PDL files a full state name. Expanding
    # the code keeps one shape in the column whichever source won.
    state_name = _titlecase_place(_ABBR_TO_STATE_NAME.get(state, "")) or state
    if city or state_name:
        joined = ", ".join(p for p in (city, state_name) if p)
        return joined, "sec_edgar"
    return None, None


# --- assembly -----------------------------------------------------------


def _candidate(
    frame: pl.DataFrame,
    *,
    text: str,
    source: str,
    url: pl.Expr,
    as_of: pl.Expr,
) -> pl.DataFrame:
    return frame.select(
        "ats",
        "slug",
        pl.col(text).alias("company_description"),
        pl.lit(source).alias("company_description_source"),
        url.alias("company_description_url"),
        as_of.alias("company_description_as_of"),
    )


def _tenant_names(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(n)
        for n in (
            row.get("resolved_name"),
            row.get("display_name"),
            row.get("name"),
            row.get("jobs_company"),
        )
        if n
    )


def build(*, use_site_crawl: bool = True, use_10k: bool = True) -> pl.DataFrame:
    resolved = pl.read_parquet(config.RESOLVED_PARQUET)
    if resolved.is_empty():
        raise RuntimeError("resolved.parquet is required; run the resolve stage first")

    cohort = pl.read_parquet(config.COHORT_PARQUET).select(
        "ats", "slug", "sample_posting_url"
    )
    base = resolved.join(cohort, on=["ats", "slug"], how="left")

    profiles = registrant.load()
    if "registrant_city" in profiles.columns:
        base = base.join(
            profiles.select("cik", "registrant_city", "registrant_state"),
            on="cik",
            how="left",
        )
    else:
        base = base.with_columns(
            pl.lit(None, dtype=pl.String).alias("registrant_city"),
            pl.lit(None, dtype=pl.String).alias("registrant_state"),
        )

    base = base.with_columns(
        pl.coalesce(pl.col("pdl_score"), pl.col("edgar_score")).alias("_match_score"),
        pl.col("name").map_elements(name_key, return_dtype=pl.String).alias("name_key_core"),
    )
    # A match this weak cannot carry a description on its own authority;
    # the description has to name the company for it to be used at all.
    suspect = (
        pl.col("matched_on_slug_only").fill_null(False)
        | (pl.col("_match_score").fill_null(0) < SUSPECT_MATCH_SCORE)
    )
    base = base.with_columns(suspect.alias("_suspect_match"))

    names = {
        (row["ats"], row["slug"]): _tenant_names(row)
        for row in base.iter_rows(named=True)
    }

    candidates: list[pl.DataFrame] = []

    # --- the company's own homepage ------------------------------------
    site = companysite.crawl(base) if use_site_crawl else companysite.load()
    if not site.is_empty():
        usable = site.filter(pl.col("site_description") != "").select(
            "ats", "slug", "site_description", "site_url", "site_fetched_at"
        )
        candidates.append(
            _candidate(
                usable,
                text="site_description",
                source="company_site",
                url=pl.col("site_url"),
                as_of=pl.col("site_fetched_at"),
            )
        )

    # --- copy repeated across the tenant's own postings ----------------
    mined = boilerplate.load()
    if not mined.is_empty():
        candidates.append(
            _candidate(
                mined,
                text="boilerplate_description",
                source="posting_boilerplate",
                url=pl.lit(None, dtype=pl.String),
                as_of=pl.col("boilerplate_as_of"),
            )
        )

    # --- Item 1, Business ----------------------------------------------
    if use_10k:
        with_cik = base.filter(pl.col("cik").is_not_null())
        by_cik: dict[int, tuple[str, ...]] = {}
        for row in with_cik.iter_rows(named=True):
            by_cik.setdefault(int(row["cik"]), _tenant_names(row))
        filings = descriptions_from_10k(sorted(by_cik), by_cik)
        if not filings.is_empty():
            candidates.append(
                _candidate(
                    with_cik.join(filings, on="cik", how="inner"),
                    text="text",
                    source="sec_10k",
                    url=pl.col("url"),
                    as_of=pl.col("as_of"),
                )
            )

    # --- Wikidata --------------------------------------------------------
    wikidata = fetch_wikidata_descriptions()
    if not wikidata.is_empty():
        columns = ["wikidata_description", "wikidata_as_of"]
        # CIK and domain only. :mod:`.teamsize` also matches Wikidata on
        # a name key, and can afford to: a collision there produces a
        # number that the PDL band and the filed floors immediately
        # contradict. A description has no such second opinion — nothing
        # downstream can tell that "UK-based artificial intelligence and
        # business automation company" belongs to a different firm — and
        # the usual fallback of checking whether the text names the
        # company does not apply, because Wikidata descriptions never do.
        # The name key was worth about 50 rows and all of the risk.
        joins = [
            base.filter(pl.col("cik").is_not_null()).join(
                _unique_on(wikidata.filter(pl.col("cik").is_not_null()), "cik").select(
                    "cik", *columns
                ),
                on="cik",
                how="inner",
            ),
            base.filter(pl.col("domain").fill_null("") != "").join(
                _unique_on(wikidata, "domain").select("domain", *columns),
                on="domain",
                how="inner",
            ),
        ]
        matched = pl.concat(joins, how="diagonal").unique(
            subset=["ats", "slug"], keep="first", maintain_order=True
        )
        candidates.append(
            _candidate(
                matched,
                text="wikidata_description",
                source="wikidata",
                url=pl.lit(None, dtype=pl.String),
                as_of=pl.col("wikidata_as_of"),
            )
        )

    description = _pick_description(base, candidates, names)

    careers_rows: list[dict[str, object]] = []
    for row in base.select(
        "ats", "slug", "url", "sample_posting_url"
    ).iter_rows(named=True):
        url, source, verified = _careers_columns(row)
        careers_rows.append(
            {
                "ats": row["ats"],
                "slug": row["slug"],
                "careers_url": url,
                "careers_url_source": source,
                "careers_url_verified": verified,
            }
        )
    careers = pl.DataFrame(
        careers_rows,
        schema={
            "ats": pl.String,
            "slug": pl.String,
            "careers_url": pl.String,
            "careers_url_source": pl.String,
            "careers_url_verified": pl.Boolean,
        },
    )

    own_careers = (
        site.filter(pl.col("site_careers_url") != "").select(
            "ats",
            "slug",
            pl.col("site_careers_url").alias("company_careers_url"),
            pl.lit("company_site").alias("company_careers_url_source"),
        )
        if not site.is_empty()
        else pl.DataFrame(
            schema={
                "ats": pl.String,
                "slug": pl.String,
                "company_careers_url": pl.String,
                "company_careers_url_source": pl.String,
            }
        )
    )

    hq_rows: list[dict[str, object]] = []
    for row in base.select(
        "ats", "slug", "locality", "region", "registrant_city", "registrant_state"
    ).iter_rows(named=True):
        place, source = _headquarters(row)
        hq_rows.append(
            {
                "ats": row["ats"],
                "slug": row["slug"],
                "headquarters": place,
                "headquarters_source": source,
            }
        )
    headquarters = pl.DataFrame(
        hq_rows,
        schema={
            "ats": pl.String,
            "slug": pl.String,
            "headquarters": pl.String,
            "headquarters_source": pl.String,
        },
    )

    out = (
        base.select("ats", "slug")
        .join(description, on=["ats", "slug"], how="left")
        .join(careers, on=["ats", "slug"], how="left")
        .join(own_careers, on=["ats", "slug"], how="left")
        .join(headquarters, on=["ats", "slug"], how="left")
    )
    for column, dtype in SCHEMA.items():
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return out.select(list(SCHEMA))


def _pick_description(
    base: pl.DataFrame,
    candidates: list[pl.DataFrame],
    names: dict[tuple[str, str], tuple[str, ...]],
) -> pl.DataFrame:
    """One description per tenant: the best source that survives its checks."""
    empty = pl.DataFrame(
        schema={
            "ats": pl.String,
            "slug": pl.String,
            "company_description": pl.String,
            "company_description_source": pl.String,
            "company_description_url": pl.String,
            "company_description_as_of": pl.String,
            "company_description_name_corroborated": pl.Boolean,
        }
    )
    usable = [c for c in candidates if not c.is_empty()]
    if not usable:
        return empty

    pool = pl.concat(usable, how="diagonal")
    suspect = {
        (row["ats"], row["slug"])
        for row in base.filter(pl.col("_suspect_match")).iter_rows(named=True)
    }

    kept: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = (row["ats"], row["slug"])
        source = str(row["company_description_source"])
        text = blurb.trim(blurb.tidy(row["company_description"]))
        if not blurb.acceptable(text, min_chars=MIN_CHARS_BY_SOURCE.get(source)):
            continue
        corroborated = blurb.mentions_name(text, *names.get(key, ()))
        # A source reached by a fuzzy match, on a match already known to
        # be shaky, that never names the company: three doubts stacked.
        # Drop it here rather than downstream, so a weaker source that
        # is tied to this tenant by construction can still be used.
        if source != "posting_boilerplate" and not corroborated and key in suspect:
            continue
        kept.append(
            {
                **row,
                "company_description": text,
                "company_description_name_corroborated": corroborated,
                "_rank": SOURCE_RANK.get(source, 0),
            }
        )
    if not kept:
        return empty

    return (
        pl.DataFrame(kept)
        # Corroboration outranks the source ordering: a homepage blurb
        # that never names the company is a weaker claim than posting
        # copy that does, whatever their provenance.
        .sort(
            ["company_description_name_corroborated", "_rank"],
            descending=[True, True],
            nulls_last=True,
        )
        .unique(subset=["ats", "slug"], keep="first", maintain_order=True)
        .select(list(empty.columns))
    )


def load() -> pl.DataFrame:
    if not config.PROFILE_PARQUET.exists():
        logger.warning("%s missing — run the `profile` stage", config.PROFILE_PARQUET)
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(config.PROFILE_PARQUET)


def run(*, use_site_crawl: bool = True, use_10k: bool = True) -> pl.DataFrame:
    config.ensure_dirs()
    out = build(use_site_crawl=use_site_crawl, use_10k=use_10k)
    out.write_parquet(config.PROFILE_PARQUET)

    total = out.height
    described = out.filter(pl.col("company_description").is_not_null())
    print("\n=== Company profile ===")
    print(f"Tenants                       : {total:>6,}")
    if not total:
        return out

    def pct(count: int) -> str:
        return f"{count:>6,}  ({count / total:5.1%})"

    print(f"With a description            : {pct(described.height)}")
    if not described.is_empty():
        print("  by source:")
        for source, n in (
            described.group_by("company_description_source")
            .len()
            .sort("len", descending=True)
            .iter_rows()
        ):
            print(f"    {source:<20}: {n:>6,}")
        corroborated = described.filter(
            pl.col("company_description_name_corroborated")
        ).height
        print(
            f"  names the company           : {corroborated:>6,}  "
            f"({corroborated / described.height:5.1%} of descriptions)"
        )
        median = described["company_description"].str.len_chars().median()
        print(f"  median length               : {median:>6,.0f} characters")

    print(
        f"With an ATS board URL         : "
        f"{pct(out.filter(pl.col('careers_url').is_not_null()).height)}"
    )
    for source, n in (
        out.filter(pl.col("careers_url_source").is_not_null())
        .group_by("careers_url_source")
        .len()
        .sort("len", descending=True)
        .iter_rows()
    ):
        print(f"    {source:<20}: {n:>6,}")
    unverified = out.filter(
        pl.col("careers_url").is_not_null()
        & ~pl.col("careers_url_verified").fill_null(False)
    ).height
    print(f"    unconfirmed shape   : {unverified:>6,}")
    print(
        f"With their own careers page   : "
        f"{pct(out.filter(pl.col('company_careers_url').is_not_null()).height)}"
    )
    print(
        f"With a headquarters           : "
        f"{pct(out.filter(pl.col('headquarters').is_not_null()).height)}"
    )

    print("\nSample:")
    print(
        described.head(6)
        .select(
            "ats",
            "slug",
            pl.col("company_description_source").alias("source"),
            pl.col("company_description").str.slice(0, 76).alias("description"),
        )
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.PROFILE_PARQUET}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run()

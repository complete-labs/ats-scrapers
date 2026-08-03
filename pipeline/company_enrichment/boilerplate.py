"""Recover a company blurb from the copy it repeats across its postings.

The fallback for the two thirds of the cohort whose homepage yields
nothing. It costs no new crawling: the text is already in the published
jobs snapshot, which :mod:`.cohort` reads anyway.

**The blurb is the shared prefix.** A tenant that introduces itself in
its postings does so in the same words at the top of every one, so the
longest common prefix across several of its postings *is* that
introduction, with no parsing of headings required. Measured on real
tenants: Crusoe 1,181 characters, Shield AI 759, Snowflake 707,
Mineralys 495, SpaceX 284, KARE 280 — all of it clean company copy.

**The shared suffix is not.** The same measurement found the common
suffix to be, without exception, employment law: SpaceX's 1,163-character
ITAR block, Crusoe's EEO statement, Fetch's recruiting-fraud notice. It
is discarded rather than used as a second candidate.

**Headings are a trap.** ``About the Role`` and ``About This Role``
introduce the vacancy, not the employer, and a bare ``About the
Company`` on a staffing board describes the agency's *client* — the
tenant Kimmel & Associates advertises a construction firm it is not.
Only headings that name the tenant or say "us" are accepted, and only
when the text under them repeats across postings.

**A shared display name is not a shared company.** Postings are keyed on
``(ats_type, company)``, and 152 of those keys cover more than one tenant
in the cohort. Some are one employer running several boards (Turner
Construction has three), but most are a parent's name stretched over
distinct businesses: five FOX entities, ten Stagwell agencies, four VF
Corporation brands. Pooling their postings hands FOX Factory's blurb to
FOX Rehabilitation. Where a key is shared, the blurb is kept only for the
tenants it actually names.

Two artefacts, both cached: the sampled posting heads, so the remote
scan is paid once, and the derived blurbs, so re-deriving is instant.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import UTC, datetime

import polars as pl

from pipeline.company_enrichment import blurb, config

logger = logging.getLogger(__name__)

# Postings sampled per tenant. Three is the minimum for a common prefix
# to mean "boilerplate" rather than "these two openings are in the same
# department"; beyond about six, extra samples only shorten the prefix.
SAMPLES_PER_TENANT = 6
MIN_SAMPLES = 3

# The blurb is a prefix, so only the head of each posting is fetched.
# This is what keeps the sample table to tens of megabytes instead of
# streaming whole descriptions for the entire cohort.
SAMPLE_HEAD_CHARS = 3000

# Shorter than this and a "shared prefix" is a shared greeting rather
# than a description of the company.
MIN_BLURB_CHARS = 120

SCHEMA: dict[str, pl.DataType] = {
    "ats": pl.String,
    "slug": pl.String,
    "boilerplate_description": pl.String,
    "boilerplate_method": pl.String,
    "boilerplate_samples": pl.Int32,
    "boilerplate_as_of": pl.String,
}

_SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "ats_type": pl.String,
    "company": pl.String,
    "title": pl.String,
    "head": pl.String,
}

# Markdown and bullet decoration a heading may be wrapped in.
_HEADING_LEAD = r"[\s#*>_\-\u2022]*"
_HEADING_TRAIL = r"[\s:*#_\-\u2013\u2014]*"
# Headings that introduce the employer. "About the Company" is
# deliberately absent; see the module docstring.
_ABOUT_ALTERNATIVES = (
    r"about\s+(?:us|our\s+company|our\s+team|our\s+organi[sz]ation)\b"
    r"|who\s+we\s+are\b"
    r"|our\s+(?:mission|story|purpose)\b"
    r"|company\s+overview\b"
)
_ABOUT_HEADING = re.compile(
    rf"(?im)^{_HEADING_LEAD}(?:{_ABOUT_ALTERNATIVES}){_HEADING_TRAIL}"
)
# "About <Tenant>" is built per tenant, since it needs the name.
_HEADING_TAIL_CHARS = 1200
# A blank line, or a new heading, ends the section.
_SECTION_BREAK = re.compile(
    r"\n\s*\n|^\s*#{1,6}\s+\S|^\s*\*\*[^*\n]{3,60}\*\*\s*$", re.MULTILINE
)


def _sample_sql(url: str) -> str:
    """Up to ``SAMPLES_PER_TENANT`` posting heads for each cohort tenant.

    The join to the cohort runs before the window function so the scan
    only materialises rows for employers that are actually in the
    cohort, and only the head of each description crosses the wire.
    """
    return f"""
    WITH matched AS (
        SELECT
            j.ats_type,
            j.company,
            j.title,
            substr(j.description, 1, {SAMPLE_HEAD_CHARS}) AS head,
            row_number() OVER (
                PARTITION BY j.ats_type, j.company ORDER BY j.url
            ) AS rn
        FROM read_parquet('{url}') j
        SEMI JOIN cohort_keys c
            ON c.ats_type = j.ats_type AND c.jobs_company = j.company
        WHERE j.description IS NOT NULL
          AND length(j.description) BETWEEN 400 AND 20000
    )
    SELECT ats_type, company, title, head
    FROM matched
    WHERE rn <= {SAMPLES_PER_TENANT}
    """


def sample_postings(cohort: pl.DataFrame, *, force: bool = False) -> pl.DataFrame:
    """Download posting heads for the cohort, or reuse the cached copy."""
    cache = config.POSTING_SAMPLES_CACHE
    if cache.exists() and not force:
        logger.info("reusing cached posting samples at %s", cache)
        return pl.read_parquet(cache)

    import duckdb

    from pipeline.company_enrichment.cohort import _jobs_parquet_url

    keys = (
        cohort.select(
            pl.col("ats").alias("ats_type"),
            pl.col("jobs_company"),
        )
        .filter(pl.col("jobs_company").is_not_null() & (pl.col("jobs_company") != ""))
        .unique()
    )
    logger.info(
        "sampling up to %d postings for each of %d employers — reads "
        "`description`, so expect to stream most of the jobs parquet on a "
        "cold cache",
        SAMPLES_PER_TENANT,
        keys.height,
    )

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar = true;")
    con.register("cohort_keys", keys.to_arrow())
    frame = pl.from_arrow(con.execute(_sample_sql(_jobs_parquet_url())).arrow())
    assert isinstance(frame, pl.DataFrame)
    frame = frame.cast(_SAMPLE_SCHEMA)  # type: ignore[arg-type]
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache)
    logger.info("cached %d posting samples at %s", frame.height, cache)
    return frame


def common_prefix(texts: list[str]) -> str:
    """The longest string every entry of ``texts`` starts with."""
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        limit = min(len(prefix), len(text))
        index = 0
        while index < limit and prefix[index] == text[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _heading_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    """``_ABOUT_HEADING`` plus "About <Tenant>" for this tenant's names.

    A lookahead rather than ``\\b`` terminates the name, because a
    display name can legitimately end in punctuation ("Kimmel &
    Associates") where a word boundary would not hold.
    """
    escaped = sorted(
        {re.escape(n.strip()) for n in names if n and len(n.strip()) >= 3},
        key=len,
        reverse=True,
    )
    if not escaped:
        return _ABOUT_HEADING
    joined = "|".join(escaped)
    return re.compile(
        rf"(?im)^{_HEADING_LEAD}(?:{_ABOUT_ALTERNATIVES}"
        rf"|(?:about|why)\s+(?:{joined})(?!\w)"
        rf"){_HEADING_TRAIL}"
    )


def _heading_candidate(text: str, pattern: re.Pattern[str]) -> str:
    """Text introduced by an accepted "about the employer" heading."""
    match = pattern.search(text)
    if not match:
        return ""
    tail = text[match.end() : match.end() + _HEADING_TAIL_CHARS].lstrip(" \t\r\n:*#-")
    stop = _SECTION_BREAK.search(tail)
    if stop and stop.start() >= MIN_BLURB_CHARS:
        tail = tail[: stop.start()]
    return blurb.tidy(tail)


def _contains_a_title(text: str, titles: list[str]) -> bool:
    """True when the candidate quotes one of the sampled job titles.

    A blurb that names a specific vacancy is that vacancy's copy, not
    the company's. Short titles are ignored because "Nurse" or "Driver"
    appear legitimately in a description of what a company does.
    """
    folded = text.lower()
    return any(len(t) >= 12 and t.lower() in folded for t in titles if t)


def _normalised(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def derive(samples: list[str], titles: list[str], names: tuple[str, ...]) -> tuple[str, str]:
    """``(blurb, method)`` for one tenant's sampled postings.

    Tries the shared prefix first and the guarded heading second. Both
    have to survive the same acceptance checks, so a tenant whose
    postings open on an EEO statement yields nothing rather than
    yielding the statement.
    """
    usable = [s for s in samples if s]
    if len(usable) < MIN_SAMPLES:
        return "", ""

    prefix = blurb.tidy(common_prefix(usable))
    if len(prefix) >= MIN_BLURB_CHARS and not _contains_a_title(prefix, titles):
        trimmed = blurb.trim(prefix)
        if blurb.acceptable(trimmed, min_chars=MIN_BLURB_CHARS):
            return trimmed, "shared_prefix"

    pattern = _heading_pattern(names)
    candidates = [_heading_candidate(sample, pattern) for sample in usable]
    counts = Counter(_normalised(c) for c in candidates if len(c) >= MIN_BLURB_CHARS)
    if not counts:
        return "", ""
    key, repeats = counts.most_common(1)[0]
    # Text under an "about us" heading that appears in only one posting
    # is that posting's copy; boilerplate repeats by definition.
    if repeats < 2:
        return "", ""
    best = max(
        (c for c in candidates if _normalised(c) == key), key=len, default=""
    )
    if _contains_a_title(best, titles):
        return "", ""
    trimmed = blurb.trim(best)
    if not blurb.acceptable(trimmed, min_chars=MIN_BLURB_CHARS):
        return "", ""
    return trimmed, "about_heading"


def build(cohort: pl.DataFrame, samples: pl.DataFrame) -> pl.DataFrame:
    """Derive a blurb per ``(ats, slug)`` from the sampled postings."""
    if samples.is_empty() or cohort.is_empty():
        return pl.DataFrame(schema=SCHEMA)

    grouped = samples.group_by("ats_type", "company").agg(
        pl.col("head").alias("heads"), pl.col("title").alias("titles")
    )
    keyed = cohort.select(
        "ats", "slug", "jobs_company", "display_name", "name"
    ).join(
        grouped,
        left_on=["ats", "jobs_company"],
        right_on=["ats_type", "company"],
        how="inner",
    )

    # Keys covering more than one tenant pool several companies' postings
    # into one sample; see the module docstring.
    shared_keys = {
        (row["ats"], row["jobs_company"])
        for row in keyed.group_by("ats", "jobs_company")
        .len()
        .filter(pl.col("len") > 1)
        .iter_rows(named=True)
    }

    # The jobs snapshot has no per-posting capture date the sample keeps,
    # so the blurb is dated to when it was derived from that snapshot.
    as_of = datetime.now(tz=UTC).date().isoformat()
    unattributable = 0
    records: list[dict[str, object]] = []
    for row in keyed.iter_rows(named=True):
        heads = [h for h in (row["heads"] or []) if h]
        names = tuple(
            n for n in (row["display_name"], row["name"], row["jobs_company"]) if n
        )
        text, method = derive(heads, list(row["titles"] or []), names)
        if not text:
            continue
        if (row["ats"], row["jobs_company"]) in shared_keys and not blurb.mentions_name(
            text, *names
        ):
            unattributable += 1
            continue
        records.append(
            {
                "ats": row["ats"],
                "slug": row["slug"],
                "boilerplate_description": text,
                "boilerplate_method": method,
                "boilerplate_samples": len(heads),
                "boilerplate_as_of": as_of,
            }
        )
    if unattributable:
        logger.info(
            "withdrew %d blurbs that could not be attributed to one tenant "
            "under a shared display name",
            unattributable,
        )
    return pl.DataFrame(records, schema=SCHEMA) if records else pl.DataFrame(schema=SCHEMA)


def load() -> pl.DataFrame:
    """Read the derived blurbs, or an empty frame if the stage has not run."""
    if not config.BOILERPLATE_CACHE.exists():
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(config.BOILERPLATE_CACHE)


def run(*, force: bool = False) -> pl.DataFrame:
    config.ensure_dirs()
    cohort = pl.read_parquet(config.COHORT_PARQUET)
    samples = sample_postings(cohort, force=force)
    derived = build(cohort, samples)
    derived.write_parquet(config.BOILERPLATE_CACHE)

    total = cohort.height
    sampled = samples.select("ats_type", "company").unique().height
    print("\n=== Posting boilerplate ===")
    print(f"Cohort tenants                : {total:>6,}")
    print(f"Employers with sampled posts  : {sampled:>6,}")
    if total:
        found = derived.height
        print(f"Blurb recovered               : {found:>6,}  ({found / total:5.1%})")
    if not derived.is_empty():
        print("\nBy method:")
        print(
            derived.group_by("boilerplate_method")
            .agg(pl.len().alias("tenants"), pl.col("boilerplate_description").str.len_chars().median().alias("median_chars"))
            .sort("tenants", descending=True)
            .to_pandas()
            .to_string(index=False)
        )
        print("\nSample:")
        print(
            derived.head(5)
            .with_columns(pl.col("boilerplate_description").str.slice(0, 90))
            .to_pandas()
            .to_string(index=False)
        )
    print(f"\nwrote {config.BOILERPLATE_CACHE}")
    return derived


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

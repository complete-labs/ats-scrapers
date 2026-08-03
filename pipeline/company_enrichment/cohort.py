"""Stage 0 — select the enrichment cohort.

The target is *US companies with at least one pay-transparent posting*.
Both halves of that filter matter:

- US-only keeps us inside the jurisdiction where the free sources
  actually have coverage (SEC filings are US-only).
- A disclosed salary is a strong proxy for a real US-operating
  employer, because US pay-transparency statutes (CO, NY, CA, WA, IL,
  MD, …) are what force the disclosure in the first place.

Neither half can be read straight off the published columns. 37% of
rows have an empty ``country_iso``, and the six largest US ATSes have no
structured salary at all — see :mod:`ats_scrapers.enrichment.uslocation`
for the measurements and the recovery patterns. This module applies those
patterns inside the DuckDB scan.

The published ``all.parquet`` is ~2 GB / 4.2M rows and the pay check has
to read ``description``, so the first run streams most of the file. The
aggregate is cached, so subsequent stages are instant.

Output: ``cohort.parquet`` — one row per ``(ats, slug)`` tenant that
matched, carrying posting counts used later to prioritise review.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ats_scrapers.enrichment.uslocation import pay_sql_expr, us_sql_expr
from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import display_name, name_key

logger = logging.getLogger(__name__)


def _cohort_sql(url: str) -> str:
    """Aggregate US pay-transparent postings down to one row per employer.

    All salary columns are BYTE_ARRAY in the published parquet, so
    emptiness means both NULL and ''.
    """
    is_us = us_sql_expr()
    has_pay = pay_sql_expr()
    return f"""
    SELECT
        ats_type,
        company,
        count(*)                                        AS postings,
        count(*) FILTER (
            WHERE salary_min IS NOT NULL AND salary_min <> ''
        )                                               AS postings_structured_salary,
        count(*) FILTER (WHERE country_iso = 'US')      AS postings_country_us,
        min(TRY_CAST(salary_min AS DOUBLE))             AS salary_min_observed,
        max(TRY_CAST(salary_max AS DOUBLE))             AS salary_max_observed,
        any_value(location)                             AS sample_location,
        any_value(url)                                  AS sample_posting_url
    FROM read_parquet('{url}')
    WHERE company IS NOT NULL
      AND company <> ''
      AND {is_us}
      AND {has_pay}
    GROUP BY ats_type, company
    """


def _jobs_parquet_url() -> str:
    import json
    import urllib.request

    req = urllib.request.Request(
        config.JOBS_MANIFEST_URL, headers={"User-Agent": config.SEC_USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        manifest = json.load(resp)
    url = manifest.get("all", {}).get("parquet")
    if not url:
        raise RuntimeError("manifest has no all.parquet entry")
    logger.info(
        "jobs snapshot: %s rows, %.2f GB parquet",
        f"{manifest['all'].get('rows', 0):,}",
        manifest["all"].get("parquet_size_bytes", 0) / 1e9,
    )
    return str(url)


def scan_us_salary_companies(*, cache: Path | None = None) -> pl.DataFrame:
    """Aggregate the jobs snapshot down to US pay-transparent employers.

    Result is cached to ``cache`` because the remote scan takes minutes;
    delete that file to force a refresh.
    """
    cache = cache or (config.CACHE_DIR / "us_salary_companies.parquet")
    if cache.exists():
        logger.info("reusing cached jobs aggregate at %s", cache)
        return pl.read_parquet(cache)

    import duckdb

    url = _jobs_parquet_url()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar = true;")
    logger.info(
        "scanning remote jobs parquet — reads `description`, so expect to "
        "stream most of the ~2 GB file on a cold cache"
    )
    arrow_table = con.execute(_cohort_sql(url)).arrow()
    df = pl.from_arrow(arrow_table)
    assert isinstance(df, pl.DataFrame)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    logger.info("cached jobs aggregate (%d employer rows) at %s", df.height, cache)
    return df


def load_companies_directory() -> pl.DataFrame:
    """Load ``ats-companies/*.csv`` into one frame of ``ats,name,slug,url``.

    Mirrors ``build_aggregated`` in ``.github/scripts/publish_companies.py``
    so the cohort keys line up with the published directory, including
    the empty-``slug`` fallback for un-migrated per-ATS files.
    """
    frames: list[pl.DataFrame] = []
    for path in sorted(config.ATS_COMPANIES_DIR.glob("*.csv")):
        df = pl.read_csv(path, infer_schema_length=0)  # all-string, like pandas dtype=str
        cols = set(df.columns)
        if not {"name", "url"}.issubset(cols):
            logger.warning("%s missing name/url columns: %s", path.name, df.columns)
            continue
        if "slug" not in cols:
            df = df.with_columns(pl.lit("").alias("slug"))
        df = df.select(
            pl.lit(path.stem).alias("ats"),
            pl.col("name").fill_null(""),
            pl.col("slug").fill_null(""),
            pl.col("url").fill_null(""),
            # eightfold is the only source carrying a real corporate
            # domain; keep it, the resolver treats it as a free hint.
            (
                pl.col("domain").fill_null("")
                if "domain" in cols
                else pl.lit("").alias("domain")
            ).alias("source_domain"),
        )
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no CSVs under {config.ATS_COMPANIES_DIR}")
    return pl.concat(frames, how="vertical")


def _with_keys(df: pl.DataFrame, name_col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(name_col)
        .map_elements(name_key, return_dtype=pl.String)
        .alias("_key"),
        pl.col(name_col)
        .map_elements(lambda s: name_key(s, keep_suffix=True), return_dtype=pl.String)
        .alias("_key_suffixed"),
    )


def _slug_from_posting_url(url: str | None) -> str:
    """Recover the tenant slug from a posting URL, or ``""``.

    ``Job`` carries no slug, so the display name is normally the only
    bridge back to the directory. For the ATSes whose posting URLs embed
    the tenant (``job-boards.greenhouse.io/<slug>/jobs/123``) this is an
    exact key instead of a fuzzy one.
    """
    if not url:
        return ""
    try:
        from ats_scrapers.resolve import resolve_careers_url

        resolved = resolve_careers_url(str(url))
    except Exception:
        return ""
    return resolved.slug.lower() if resolved else ""


def build_cohort() -> pl.DataFrame:
    """Join US pay-transparent employers to the companies directory.

    Three keys are tried in descending order of trust: the slug parsed
    out of a posting URL, the raw employer string used literally as a
    slug (several ATSes report the tenant id in ``company``), and finally
    the normalised name. The winning strategy is recorded per row in
    ``join_method`` so downstream stages can discount the weak ones.
    """
    jobs = scan_us_salary_companies()
    companies = load_companies_directory()
    logger.info(
        "inputs: %d US salary-bearing employers, %d directory tenants",
        jobs.height,
        companies.height,
    )

    jobs = _with_keys(jobs, "company").filter(pl.col("_key") != "")
    directory = _with_keys(companies, "name").filter(pl.col("_key") != "")

    # Collapse employer rows that normalise to the same key within an ATS
    # ("Acme, Inc." and "Acme Inc" are one tenant).
    jobs_agg = (
        jobs.sort("postings", descending=True)
        .group_by(["ats_type", "_key"])
        .agg(
            pl.col("company").first().alias("jobs_company"),
            pl.col("postings").sum(),
            pl.col("postings_structured_salary").sum(),
            pl.col("postings_country_us").sum(),
            pl.col("salary_min_observed").min(),
            pl.col("salary_max_observed").max(),
            pl.col("sample_location").first(),
            pl.col("sample_posting_url").first(),
        )
        .with_columns(
            pl.col("sample_posting_url")
            .map_elements(_slug_from_posting_url, return_dtype=pl.String)
            .alias("_url_slug"),
            pl.col("jobs_company").str.to_lowercase().str.strip_chars().alias("_literal_slug"),
        )
    )

    directory = directory.with_columns(
        pl.col("slug").str.to_lowercase().alias("_slug_lower")
    )
    # Payload carried over from the jobs side. The three candidate join
    # keys are excluded so each strategy can promote its own to the join
    # column without colliding with the directory's `_key`.
    key_cols = {"ats_type", "_key", "_url_slug", "_literal_slug"}
    payload = [c for c in jobs_agg.columns if c not in key_cols]

    def _try(left_key: str, right_key: str, method: str) -> pl.DataFrame:
        right = jobs_agg.select(["ats_type", right_key, *payload]).filter(
            pl.col(right_key).is_not_null() & (pl.col(right_key) != "")
        )
        return directory.join(
            right,
            left_on=["ats", left_key],
            right_on=["ats_type", right_key],
            how="inner",
        ).with_columns(pl.lit(method).alias("join_method"))

    by_url = _try("_slug_lower", "_url_slug", "url_slug")
    by_literal = _try("_slug_lower", "_literal_slug", "literal_slug")
    by_name = _try("_key", "_key", "name")

    # Later frames only contribute tenants the earlier ones missed.
    joined = pl.concat([by_url, by_literal, by_name], how="diagonal")

    # One directory row per (ats, slug); ordering by join_method rank
    # keeps the most trustworthy match when several strategies fire.
    method_rank = pl.col("join_method").replace_strict(
        {"url_slug": 0, "literal_slug": 1, "name": 2}, default=3, return_dtype=pl.Int32
    )
    cohort = (
        joined.with_columns(method_rank.alias("_rank"))
        .sort(["_rank", "postings"], descending=[False, True])
        .unique(subset=["ats", "slug"], keep="first")
        .with_columns(
            pl.col("name").map_elements(display_name, return_dtype=pl.String).alias("display_name"),
        )
        .select(
            "join_method",
            "ats",
            "slug",
            "name",
            "display_name",
            "jobs_company",
            "url",
            "source_domain",
            "_key",
            "_key_suffixed",
            "postings",
            "postings_structured_salary",
            "postings_country_us",
            "salary_min_observed",
            "salary_max_observed",
            "sample_location",
            "sample_posting_url",
        )
        .sort("postings", descending=True)
    )
    return cohort


def run() -> pl.DataFrame:
    config.ensure_dirs()
    jobs = scan_us_salary_companies()
    cohort = build_cohort()

    matched_postings = cohort["postings"].sum()
    total_postings = jobs["postings"].sum()
    directory_size = load_companies_directory().height

    print("\n=== Stage 0: cohort ===")
    print(f"US pay-transparent employers in jobs : {jobs.height:>7,}")
    print(f"Companies directory tenants          : {directory_size:>7,}")
    print(f"Matched cohort (ats, slug)           : {cohort.height:>7,}")
    print(
        f"Employer join hit rate               : "
        f"{cohort.height / max(jobs.height, 1):>7.1%}"
    )
    print(
        f"Postings covered                     : "
        f"{matched_postings:,} / {total_postings:,} "
        f"({matched_postings / max(total_postings, 1):.1%})"
    )
    print("\nBy join method:")
    print(
        cohort.group_by("join_method")
        .agg(pl.len().alias("companies"), pl.col("postings").sum())
        .sort("companies", descending=True)
        .to_pandas()
        .to_string(index=False)
    )

    # ATSes with no directory CSV (aggregators like welcometothejungle,
    # ycombinator; single-company scrapers like tesla) can never join,
    # so surface them separately instead of hiding them in the miss rate.
    known_ats = {p.stem for p in config.ATS_COMPANIES_DIR.glob("*.csv")}
    unjoinable = jobs.filter(~pl.col("ats_type").is_in(list(known_ats)))
    if unjoinable.height:
        print(
            f"\nEmployers on ATSes with no directory CSV (structurally "
            f"unjoinable): {unjoinable.height:,}"
        )
        print(
            unjoinable.group_by("ats_type")
            .agg(pl.len().alias("employers"))
            .sort("employers", descending=True)
            .head(8)
            .to_pandas()
            .to_string(index=False)
        )
        joinable = jobs.height - unjoinable.height
        print(
            f"Hit rate among joinable employers    : "
            f"{cohort.height / max(joinable, 1):.1%} "
            f"({cohort.height:,} / {joinable:,})"
        )

    print("\nBy ATS (top 15):")
    by_ats = (
        cohort.group_by("ats")
        .agg(pl.len().alias("companies"), pl.col("postings").sum())
        .sort("companies", descending=True)
        .head(15)
    )
    print(by_ats.to_pandas().to_string(index=False))
    print("\nTop 15 tenants by postings:")
    print(
        cohort.select("ats", "slug", "display_name", "postings")
        .head(15)
        .to_pandas()
        .to_string(index=False)
    )

    cohort.write_parquet(config.COHORT_PARQUET)
    print(f"\nwrote {config.COHORT_PARQUET} ({cohort.height:,} rows)")
    return cohort


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

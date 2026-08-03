"""OSHA Injury Tracking Application — a verified floor under headcount.

Why this source exists here
---------------------------
The PDL size band is a self-reported LinkedIn-style bucket from a free
extract that is a *snapshot*, and it goes stale in the two ways that
matter most: a company grows out of its band (Phasor Engineering is
"11-50" here and 201-500 on LinkedIn today) or it renames and leaves a
dormant shell behind (PDL still carries "anthem, inc." at "1-10" while
the live record is "elevance health" at 10001+). Nothing else in the
pipeline could see either mistake, because :mod:`.teamsize`'s exact
sources only cover public registrants and small SEC issuers — 7.6% of
the cohort.

Employers in high-hazard industries must file OSHA Form 300A
electronically each year (29 CFR 1904.41), and the summary file states
``annual_average_employees`` per establishment. That is a *filed*
number, current to the previous calendar year, and it covers exactly the
kind of employer — construction, manufacturing, healthcare, retail,
warehousing — that PDL is worst at and that has no SEC presence.

A floor, never a count
----------------------
Only establishments meeting OSHA's industry and size criteria have to
report, so summing a company's establishments gives a **lower bound on
its US headcount, not its headcount**. A company with ten warehouses and
a head office contributes the warehouses only. So this is written as
``employee_count_floor`` and is never merged into ``employee_count``;
promoting it would invent the same false precision that turning "51-200"
into 125 would.

What it is good for is contradiction. A floor of 33,396 filed
establishment-by-establishment refutes a "1-10" band no matter how much
of the company is missing from the file.

Licence: US Government work, public domain. See
https://www.osha.gov/itadata.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import polars as pl

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import name_key

logger = logging.getLogger(__name__)

ATTRIBUTION = (
    "Establishment employment from the OSHA Injury Tracking Application "
    "Form 300A summary data (29 CFR 1904.41), a US Government work in the "
    "public domain."
)

# Filenames carry the calendar year the injuries occurred in:
# "ITA_300A_Summary_Data_2025_through_03-15-2026_v2.csv".
_SUMMARY_LINK = re.compile(
    r"href=\"(?P<href>[^\"]*ITA[_\- ]?300A[_\- ]?Summary[_\- ]?Data[^\"]*?"
    r"(?P<year>20\d{2})[^\"]*\.(?P<ext>csv|zip))\"",
    re.IGNORECASE,
)

_COLUMNS = (
    "company_name",
    "establishment_name",
    "annual_average_employees",
    "total_hours_worked",
    "state",
    "naics_code",
    "year_filing_for",
)

# OSHA's own upload validation rejects a count at or above 25,000, so
# anything larger is a data-entry error rather than a big site. The file
# contains such rows regardless: AECOM's Texas entry claims 32,841,346
# employees against 15,789 hours worked.
_MAX_ESTABLISHMENT_EMPLOYEES = 25_000

# Hours worked is the cross-check on that field. Even an all-seasonal
# workforce averages more than twenty hours per person per *year*;
# below that the two numbers cannot describe the same establishment.
_MIN_HOURS_PER_EMPLOYEE = 20.0

# Rollup keys shorter than this match too many unrelated employers to
# carry any weight.
_MIN_KEY_LEN = 6


def _discover_summary_url() -> tuple[str, int]:
    """(URL, year) of the most recent Form 300A summary file.

    The filename changes every year and the directory moves between
    ``files`` and ``largefiles``, so the index page is the only stable
    reference.
    """
    import httpx

    response = httpx.get(
        config.OSHA_ITA_INDEX_URL,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()

    found: list[tuple[int, str]] = []
    for match in _SUMMARY_LINK.finditer(response.text):
        href = match.group("href")
        if href.startswith("/"):
            href = "https://www.osha.gov" + href
        found.append((int(match.group("year")), href))
    if not found:
        raise RuntimeError(
            f"no Form 300A summary link found at {config.OSHA_ITA_INDEX_URL}"
        )
    year, url = max(found)
    logger.info("OSHA ITA: newest summary file is %s (CY %d)", url, year)
    return url, year


def _download(url: str, dest: Path) -> None:
    import httpx

    with httpx.stream(
        "GET",
        url,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=600.0,
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)


def _read_summary(path: Path) -> pl.DataFrame:
    """Read the summary CSV, whether it arrived bare or inside a ZIP."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise RuntimeError(f"no CSV inside {path}")
            # Older years ship the summary and the establishment file
            # together; the one with the 300A columns is the summary.
            for name in sorted(names):
                raw = archive.read(name)
                head = raw[:4096].decode("utf-8", errors="replace")
                if "annual_average_employees" in head:
                    return pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
            raise RuntimeError(f"no 300A summary CSV inside {path}")
    return pl.read_csv(path, infer_schema_length=0)


def _clean(raw: pl.DataFrame) -> pl.DataFrame:
    """Drop unusable rows and attach the rollup key."""
    for column in _COLUMNS:
        if column not in raw.columns:
            raw = raw.with_columns(pl.lit(None, dtype=pl.String).alias(column))

    df = raw.select(_COLUMNS).with_columns(
        pl.col("annual_average_employees").cast(pl.Int64, strict=False).alias("employees"),
        pl.col("total_hours_worked").cast(pl.Float64, strict=False).alias("hours"),
    )
    before = df.height

    df = df.filter(
        pl.col("employees").is_not_null()
        & (pl.col("employees") >= 1)
        & (pl.col("employees") <= _MAX_ESTABLISHMENT_EMPLOYEES)
    )
    implausible = (
        pl.col("hours").is_not_null()
        & (pl.col("hours") > 0)
        & ((pl.col("hours") / pl.col("employees")) < _MIN_HOURS_PER_EMPLOYEE)
    )
    df = df.filter(~implausible)

    # `company_name` is the parent an establishment reports under and is
    # what a tenant name should match; it is optional on the form, so
    # single-site filers fall back to the establishment name.
    df = (
        df.with_columns(
            pl.coalesce(pl.col("company_name"), pl.col("establishment_name")).alias(
                "reported_name"
            )
        )
        .filter(pl.col("reported_name").is_not_null())
        .with_columns(
            pl.col("reported_name")
            .map_elements(name_key, return_dtype=pl.String)
            .alias("name_key_core")
        )
        .filter(pl.col("name_key_core").str.len_chars() >= _MIN_KEY_LEN)
    )
    logger.info("OSHA ITA: kept %d of %d establishment rows", df.height, before)
    return df


def ingest(*, force: bool = False) -> pl.DataFrame:
    """Materialise the per-company employment floor."""
    config.ensure_dirs()
    if config.OSHA_PARQUET.exists() and not force:
        logger.info("reusing %s", config.OSHA_PARQUET)
        return pl.read_parquet(config.OSHA_PARQUET)

    url, year = _discover_summary_url()
    suffix = ".zip" if url.lower().endswith(".zip") else ".csv"
    cached = config.CACHE_DIR / f"osha_ita_300a_{year}{suffix}"
    if not cached.exists() or force:
        logger.info("downloading %s", url)
        _download(url, cached)

    df = _clean(_read_summary(cached))

    rolled = (
        df.group_by("name_key_core")
        .agg(
            pl.col("employees").sum().alias("employee_count_floor"),
            pl.len().alias("osha_establishments"),
            pl.col("state").drop_nulls().unique().alias("osha_states"),
            pl.col("reported_name").first().alias("osha_name"),
        )
        .with_columns(
            pl.lit(f"{year}-12-31").alias("employee_count_floor_as_of"),
            pl.lit("osha_ita").alias("employee_count_floor_source"),
        )
        .sort("employee_count_floor", descending=True)
    )
    rolled.write_parquet(config.OSHA_PARQUET)
    logger.info("wrote %s (%d companies)", config.OSHA_PARQUET, rolled.height)
    return rolled


def run(*, force: bool = False) -> pl.DataFrame:
    df = ingest(force=force)
    print("\n=== OSHA ITA employment floor ===")
    print(ATTRIBUTION)
    print(f"\nCompanies                 : {df.height:>8,}")
    print(f"Establishments            : {df['osha_establishments'].sum():>8,}")
    print(f"Multi-site companies      : {df.filter(pl.col('osha_establishments') > 1).height:>8,}")
    print(f"Reporting year            : {df['employee_count_floor_as_of'][0]:>8}")
    print("\nLargest floors:")
    print(
        df.head(10)
        .select(
            "osha_name",
            pl.col("employee_count_floor").alias("floor"),
            pl.col("osha_establishments").alias("sites"),
            pl.col("osha_states").list.len().alias("states"),
        )
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.OSHA_PARQUET}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

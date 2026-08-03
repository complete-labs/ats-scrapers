"""Reg CF (Form C) and Reg A (Form 1-A) offerings — the small-raise tail.

Form D covers accredited-investor rounds, which is most venture money
but not all of it. Two smaller exemptions fill in companies Form D
misses entirely:

**Regulation Crowdfunding / Form C.** Raises capped in the low millions,
used by very early-stage companies. Quarterly datasets from May 2016.

**Regulation A / Form 1-A.** "Mini-IPO" offerings up to $75M, sitting
between Reg D and a registered offering. Quarterly datasets from 2015.

Both are public domain and tiny — a few hundred KB per quarter against
Form D's 3.5 MB — so the whole history costs one short pass.

Beyond funding, these two forms carry fields Form D lacks and that are
genuinely hard to get for free elsewhere:

- ``FORM_C_ISSUER_INFORMATION.ISSUERWEBSITE`` — the issuer's own domain,
  self-reported to the SEC. A high-quality resolution signal.
- ``FORM_C_DISCLOSURE.CURRENTEMPLOYEES`` and
  ``REG_A_EMPLOYEES_INFO.FULLTIMEEMPLOYEES`` — **exact** headcounts filed
  under penalty of perjury, which :mod:`.teamsize` prefers over PDL's
  self-reported band.

The headcounts are as-of the filing date and small companies grow fast,
so the filing date travels with the number rather than being dropped.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile

import polars as pl

from pipeline.company_enrichment import config, sechttp
from pipeline.company_enrichment.normalize import name_key, normalize_domain

logger = logging.getLogger(__name__)

FORM_C_INDEX = "https://www.sec.gov/data-research/sec-markets-data/crowdfunding-offerings-data-sets"
REG_A_INDEX = "https://www.sec.gov/data-research/sec-markets-data/regulation-data-sets"

_ZIP_HREF_RE = re.compile(r'href="([^"]*\.zip)"', re.IGNORECASE)


def _discover(index_url: str, needle: str) -> list[str]:
    html = sechttp.get(index_url, suffix=".html").decode("utf-8", errors="replace")
    hrefs = sorted({h for h in _ZIP_HREF_RE.findall(html) if needle in h.lower()})
    urls = [h if h.startswith("http") else f"https://www.sec.gov{h}" for h in hrefs]
    logger.info("discovered %d datasets at %s", len(urls), index_url)
    return urls


def _read_table(archive: zipfile.ZipFile, stem: str) -> pl.DataFrame | None:
    names = [
        n
        for n in archive.namelist()
        if n.upper().endswith((f"{stem}.TSV", f"{stem}.TXT"))
    ]
    if not names:
        return None
    return pl.read_csv(
        io.BytesIO(archive.read(names[0])),
        separator="\t",
        infer_schema_length=0,
        quote_char=None,
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )


def _pick(df: pl.DataFrame, *candidates: str) -> str | None:
    """First column present, matched case-insensitively.

    Column names drift across quarters (and the SEC's own documentation
    truncates some), so callers pass a few spellings.
    """
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _num(df: pl.DataFrame, column: str | None) -> pl.Expr:
    if column is None:
        return pl.lit(None, dtype=pl.Float64)
    return pl.col(column).str.replace_all(r"[^0-9.]", "").cast(pl.Float64, strict=False)


def _date(column: str | None) -> pl.Expr:
    if column is None:
        return pl.lit(None, dtype=pl.Date)
    text = pl.col(column).str.strip_chars()
    return pl.coalesce(
        text.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        text.str.to_date("%d-%b-%Y", strict=False),
    )


def load_form_c() -> pl.DataFrame:
    """Reg CF offerings with issuer website and self-reported headcount."""
    frames: list[pl.DataFrame] = []
    for url in _discover(FORM_C_INDEX, "_cf.zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(sechttp.get(url, suffix=".zip")))
        except Exception as exc:
            logger.warning("skipping %s: %s", url, exc)
            continue
        submission = _read_table(archive, "FORM_C_SUBMISSION")
        issuer = _read_table(archive, "FORM_C_ISSUER_INFORMATION")
        disclosure = _read_table(archive, "FORM_C_DISCLOSURE")
        if submission is None or issuer is None:
            continue

        merged = submission.join(issuer, on="ACCESSION_NUMBER", how="left")
        if disclosure is not None:
            merged = merged.join(disclosure, on="ACCESSION_NUMBER", how="left")

        name_col = _pick(merged, "NAMEOFISSUER", "COMPANYNAME")
        if name_col is None:
            continue
        frames.append(
            merged.select(
                pl.col("CIK").cast(pl.Int64, strict=False).alias("cik"),
                pl.col(name_col).alias("issuer_name"),
                _date(_pick(merged, "FILING_DATE")).alias("filing_date"),
                _num(merged, _pick(merged, "OFFERINGAMOUNT")).alias("offering_amount_usd"),
                _num(merged, _pick(merged, "MAXIMUMOFFERINGAMOUNT")).alias("max_offering_usd"),
                _num(merged, _pick(merged, "CURRENTEMPLOYEES")).alias("employee_count"),
                pl.col(_pick(merged, "ISSUERWEBSITE") or name_col).alias("_website_raw"),
                pl.col(_pick(merged, "STATEORCOUNTRY") or name_col).alias("state"),
                pl.lit("sec_form_c").alias("source"),
            )
        )

    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal")
    website_col = pl.col("_website_raw").map_elements(
        normalize_domain, return_dtype=pl.String
    )
    return combined.with_columns(website_col.alias("domain")).drop("_website_raw")


def load_reg_a() -> pl.DataFrame:
    """Reg A offerings with full-time headcount."""
    frames: list[pl.DataFrame] = []
    for url in _discover(REG_A_INDEX, "_rega.zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(sechttp.get(url, suffix=".zip")))
        except Exception as exc:
            logger.warning("skipping %s: %s", url, exc)
            continue
        submission = _read_table(archive, "REG_A_SUBMISSION")
        employees = _read_table(archive, "REG_A_EMPLOYEES_INFO")
        summary = _read_table(archive, "REG_A_SUMMARY_INFO")
        if submission is None or employees is None:
            continue

        merged = submission.join(employees, on="ACCESSION_NUMBER", how="left")
        if summary is not None:
            merged = merged.join(summary, on="ACCESSION_NUMBER", how="left")

        cik_col = _pick(merged, "CIK", "ISSUERCIK")
        name_col = _pick(merged, "ISSUERNAME")
        if cik_col is None or name_col is None:
            continue

        full_time = _num(merged, _pick(merged, "FULLTIMEEMPLOYEES"))
        frames.append(
            merged.select(
                pl.col(cik_col).cast(pl.Int64, strict=False).alias("cik"),
                pl.col(name_col).alias("issuer_name"),
                _date(_pick(merged, "FILING_DATE")).alias("filing_date"),
                _num(
                    merged,
                    _pick(
                        merged,
                        "ISSUERAGGREGATEOFFERINGPRICE",
                        "AGGREGRATEOFFERINGPRICE",
                        "AGGREGATEOFFERINGPRICE",
                    ),
                ).alias("offering_amount_usd"),
                pl.lit(None, dtype=pl.Float64).alias("max_offering_usd"),
                full_time.alias("employee_count"),
                pl.lit(None, dtype=pl.String).alias("domain"),
                pl.lit(None, dtype=pl.String).alias("state"),
                pl.lit("sec_reg_a").alias("source"),
            )
        )

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


def ingest(*, force: bool = False) -> pl.DataFrame:
    """Combined Reg CF + Reg A offerings, one row per filing."""
    config.ensure_dirs()
    cache = config.CACHE_DIR / "exempt_offerings.parquet"
    if cache.exists() and not force:
        logger.info("reusing %s", cache)
        return pl.read_parquet(cache)

    parts = [df for df in (load_form_c(), load_reg_a()) if not df.is_empty()]
    if not parts:
        raise RuntimeError("no Reg CF or Reg A data loaded")
    combined = pl.concat(parts, how="diagonal").filter(
        pl.col("issuer_name").is_not_null() & (pl.col("issuer_name") != "")
    )
    combined = combined.with_columns(
        pl.col("issuer_name").map_elements(name_key, return_dtype=pl.String).alias("issuer_key_core"),
        # Zero is the default for an unanswered headcount field, not a
        # company with no staff.
        pl.when(pl.col("employee_count") > 0)
        .then(pl.col("employee_count"))
        .otherwise(None)
        .alias("employee_count"),
    )
    combined.write_parquet(cache)
    logger.info("wrote %s (%d filings)", cache, combined.height)
    return combined


def run() -> pl.DataFrame:
    df = ingest()
    print("\n=== Reg CF (Form C) + Reg A (Form 1-A) ===")
    print(
        df.group_by("source")
        .agg(
            pl.len().alias("filings"),
            pl.col("cik").n_unique().alias("issuers"),
            pl.col("employee_count").is_not_null().sum().alias("with_headcount"),
            pl.col("domain").is_not_null().sum().alias("with_domain"),
        )
        .to_pandas()
        .to_string(index=False)
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

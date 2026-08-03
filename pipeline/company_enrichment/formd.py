"""Private funding history from SEC Form D.

Form D is the notice a company must file within 15 days of first sale in
a Regulation D exempt offering — the route essentially every US venture
round takes. The SEC republishes the filings as quarterly structured
datasets (public domain, no key, ~3.5 MB per quarter back to 2008),
which makes this the strongest free substitute for a commercial funding
database.

Three tables matter. ``FORMDSUBMISSION`` carries the filing date and
whether the submission is an original ``D`` or an amendment ``D/A``.
``ISSUERS`` carries the filer's CIK, legal name, address, and
incorporation details. ``OFFERING`` carries the money:
``TOTALOFFERINGAMOUNT``, ``TOTALAMOUNTSOLD``, the first sale date, the
investor count, and a revenue range.

What Form D does not contain
----------------------------
- **No round labels.** There is no "Series A" field. A round here is one
  offering with an amount and a date, nothing more.
- **No valuation**, pre- or post-money.
- **No investor names.** ``RECIPIENTS`` lists placement agents and
  broker-dealers, not the venture funds that participated, so it is
  deliberately unused here.
- **No coverage of rounds raised outside Reg D**, or by companies that
  file late or never. Absence of a filing is not absence of funding.

Two cleanups are essential before the numbers mean anything. Amendments
restate the same offering, so filings are collapsed along their
``PREVIOUSACCESSIONNUMBER`` chain and only the latest filing per chain
counts. And Reg D is dominated by investment vehicles rather than
operating companies — funds and SPVs are excluded using the form's own
``ISPOOLEDINVESTMENTFUNDTYPE`` / ``IS40ACT`` / industry flags rather
than by guessing from names.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import UTC, datetime

import polars as pl

from pipeline.company_enrichment import config, sechttp
from pipeline.company_enrichment.normalize import name_key

logger = logging.getLogger(__name__)

_ZIP_HREF_RE = re.compile(r'href="([^"]*form-d-data-sets/[^"]*\.zip)"', re.IGNORECASE)

# Industry buckets that are investment vehicles rather than employers.
FUND_INDUSTRIES = frozenset(
    {
        "Pooled Investment Fund",
        "Other Investment Fund",
        "Investing",
        "Commercial Banking",
    }
)

SUBMISSION_COLUMNS = ["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE", "TESTORLIVE"]
ISSUER_COLUMNS = [
    "ACCESSIONNUMBER", "IS_PRIMARYISSUER_FLAG", "CIK", "ENTITYNAME",
    "CITY", "STATEORCOUNTRY", "JURISDICTIONOFINC", "ENTITYTYPE",
    "YEAROFINC_VALUE_ENTERED",
]
OFFERING_COLUMNS = [
    "ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "INVESTMENTFUNDTYPE", "IS40ACT",
    "REVENUERANGE", "ISAMENDMENT", "PREVIOUSACCESSIONNUMBER", "SALE_DATE",
    "TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "TOTALNUMBERALREADYINVESTED",
    "ISEQUITYTYPE", "ISDEBTTYPE", "ISPOOLEDINVESTMENTFUNDTYPE",
]


def discover_quarter_urls() -> list[str]:
    """Scrape the SEC index page for every published quarterly ZIP.

    The filenames are not templatable — the SEC has used
    ``2012q1_d.zip``, ``2012q2_d_0.zip``, and at least two different
    directory prefixes over the years — so the listing page is the only
    reliable source.
    """
    html = sechttp.get(config.SEC_FORMD_INDEX_URL, suffix=".html").decode(
        "utf-8", errors="replace"
    )
    hrefs = sorted(set(_ZIP_HREF_RE.findall(html)))
    urls = [
        href if href.startswith("http") else f"https://www.sec.gov{href}"
        for href in hrefs
    ]
    if not urls:
        raise RuntimeError("no Form D dataset links found on the SEC index page")
    logger.info("discovered %d Form D quarterly datasets", len(urls))
    return urls


def _read_member(archive: zipfile.ZipFile, stem: str, columns: list[str]) -> pl.DataFrame:
    """Read one TSV out of a quarterly archive, tolerating schema drift."""
    names = [n for n in archive.namelist() if n.upper().endswith(f"{stem}.TSV")]
    if not names:
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.String))
    raw = archive.read(names[0])
    df = pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        infer_schema_length=0,
        quote_char=None,
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )
    # Older quarters predate some columns; fill them so the concat aligns.
    for column in columns:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(column))
    return df.select(columns)


def load_filings(*, force: bool = False) -> pl.DataFrame:
    """Download every quarter and join submissions, issuers, and offerings."""
    cache = config.CACHE_DIR / "formd_filings.parquet"
    if cache.exists() and not force:
        logger.info("reusing %s", cache)
        return pl.read_parquet(cache)

    frames: list[pl.DataFrame] = []
    urls = discover_quarter_urls()
    for index, url in enumerate(urls, start=1):
        try:
            payload = sechttp.get(url, suffix=".zip")
        except Exception as exc:
            logger.warning("skipping %s: %s", url, exc)
            continue
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile:
            logger.warning("skipping %s: not a zip", url)
            continue

        submissions = _read_member(archive, "FORMDSUBMISSION", SUBMISSION_COLUMNS)
        issuers = _read_member(archive, "ISSUERS", ISSUER_COLUMNS)
        offerings = _read_member(archive, "OFFERING", OFFERING_COLUMNS)
        if submissions.is_empty() or issuers.is_empty():
            continue

        # Co-issuers share an accession; only the primary issuer is the
        # company doing the raising.
        primary = issuers.filter(
            pl.col("IS_PRIMARYISSUER_FLAG").str.to_lowercase().is_in(["true", "y", "1"])
        )
        if primary.is_empty():
            primary = issuers

        quarter = (
            primary.join(submissions, on="ACCESSIONNUMBER", how="inner")
            .join(offerings, on="ACCESSIONNUMBER", how="left")
            .with_columns(pl.lit(url.rsplit("/", 1)[-1]).alias("source_file"))
        )
        frames.append(quarter)
        if index % 10 == 0:
            logger.info("loaded %d/%d quarters", index, len(urls))

    if not frames:
        raise RuntimeError("no Form D quarters loaded")
    combined = pl.concat(frames, how="diagonal")
    cache.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(cache)
    logger.info("wrote %s (%d filings)", cache, combined.height)
    return combined


def _to_float(column: str) -> pl.Expr:
    return pl.col(column).str.replace_all(r"[^0-9.]", "").cast(pl.Float64, strict=False)


def _is_true(column: str) -> pl.Expr:
    return pl.col(column).str.to_lowercase().is_in(["true", "y", "1"]).fill_null(False)


def _parse_date(column: str) -> pl.Expr:
    """Parse the three date shapes the Form D datasets actually contain.

    ``FILING_DATE`` is mostly ``2012-06-04 07:29:40`` but older quarters
    use ``04-JUN-2012``; ``SALE_DATE`` is plain ``2007-08-03``. Slicing
    the first ten characters handles both ISO variants, and the
    DD-MON-YYYY form is tried against the untruncated string.
    """
    text = pl.col(column).str.strip_chars()
    return pl.coalesce(
        text.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        text.str.to_date("%d-%b-%Y", strict=False),
    )


def normalize_offerings(filings: pl.DataFrame) -> pl.DataFrame:
    """Clean filings into one row per distinct offering.

    Test submissions are dropped, funds and SPVs are excluded, and every
    amendment chain collapses to its most recent filing so a round that
    was amended five times is counted once at its final amount.
    """
    df = filings.filter(
        pl.col("TESTORLIVE").str.to_uppercase().fill_null("LIVE") != "TEST"
    ).with_columns(
        _to_float("TOTALOFFERINGAMOUNT").alias("offering_amount_usd"),
        _to_float("TOTALAMOUNTSOLD").alias("amount_sold_usd"),
        _to_float("TOTALNUMBERALREADYINVESTED").alias("investor_count"),
        _parse_date("FILING_DATE").alias("filing_date"),
        _parse_date("SALE_DATE").alias("first_sale_date"),
        pl.col("CIK").cast(pl.Int64, strict=False).alias("cik"),
        _is_true("ISPOOLEDINVESTMENTFUNDTYPE").alias("is_pooled_fund"),
        _is_true("IS40ACT").alias("is_40_act"),
        _is_true("ISEQUITYTYPE").alias("is_equity"),
        _is_true("ISDEBTTYPE").alias("is_debt"),
    )

    before = df.height
    df = df.filter(
        ~pl.col("is_pooled_fund")
        & ~pl.col("is_40_act")
        & ~pl.col("INDUSTRYGROUPTYPE").is_in(list(FUND_INDUSTRIES)).fill_null(False)
        & (pl.col("INVESTMENTFUNDTYPE").is_null() | (pl.col("INVESTMENTFUNDTYPE") == ""))
    )
    logger.info(
        "excluded %d fund/SPV filings, %d operating-company filings remain",
        before - df.height,
        df.height,
    )

    df = _collapse_amendments(df)
    return df.with_columns(
        pl.col("ENTITYNAME")
        .map_elements(name_key, return_dtype=pl.String)
        .alias("issuer_key_core"),
        pl.col("ENTITYNAME")
        .map_elements(lambda s: name_key(s, keep_suffix=True), return_dtype=pl.String)
        .alias("issuer_key_raw"),
    )


def _collapse_amendments(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce each amendment chain to its latest filing.

    ``PREVIOUSACCESSIONNUMBER`` points at the filing being amended.
    Following it to a root groups every restatement of one offering, and
    because amendments restate cumulative totals the newest filing in a
    chain is the authoritative one.
    """
    accession_to_previous = {
        accession: previous
        for accession, previous in zip(
            df["ACCESSIONNUMBER"], df["PREVIOUSACCESSIONNUMBER"], strict=True
        )
        if previous and previous.strip() and previous != accession
    }

    roots: list[str] = []
    for accession in df["ACCESSIONNUMBER"]:
        current = accession
        seen = {current}
        # Bounded walk: malformed self-referential chains do exist.
        for _ in range(10):
            parent = accession_to_previous.get(current)
            if not parent or parent in seen:
                break
            current = parent
            seen.add(current)
        roots.append(current)

    df = df.with_columns(pl.Series("offering_root", roots, dtype=pl.String))
    collapsed = (
        df.sort(["filing_date", "ACCESSIONNUMBER"], descending=[True, True], nulls_last=True)
        .unique(subset=["offering_root"], keep="first")
    )
    logger.info(
        "collapsed %d filings into %d distinct offerings",
        df.height,
        collapsed.height,
    )
    return collapsed


def _state_from_location(location: str | None) -> str:
    """Two-letter state guess from a posting location, for corroboration."""
    if not location:
        return ""
    match = re.search(r"[,/(]\s*([A-Z]{2})\b", str(location))
    return match.group(1) if match else ""


def build_funding(offerings: pl.DataFrame, resolved: pl.DataFrame) -> pl.DataFrame:
    """Attach an offering history to every tenant with a CIK."""
    with_cik = resolved.filter(pl.col("cik").is_not_null()).select(
        "ats", "slug", "name", "cik", "cik_confidence", "ticker"
    )
    if with_cik.is_empty():
        return pl.DataFrame()

    matched = with_cik.join(
        offerings.select(
            "cik", "ENTITYNAME", "STATEORCOUNTRY", "offering_amount_usd",
            "amount_sold_usd", "investor_count", "filing_date", "first_sale_date",
            "INDUSTRYGROUPTYPE", "REVENUERANGE", "is_equity", "is_debt",
            "offering_root",
        ),
        on="cik",
        how="inner",
    )
    if matched.is_empty():
        return pl.DataFrame()

    # Equity and debt are reported separately. A mature public company's
    # private bond placement is a Reg D offering too, and rolling it into
    # one total makes Cardinal Health look like it raised $79B of venture
    # money. Equity-only figures are the venture-comparable ones.
    equity_amount = (
        pl.when(pl.col("is_equity") & ~pl.col("is_debt"))
        .then(pl.col("amount_sold_usd"))
        .otherwise(None)
    )
    rounds = (
        matched.sort("filing_date", nulls_last=False)
        .group_by("ats", "slug")
        .agg(
            pl.col("amount_sold_usd").sum().alias("funding_total_usd"),
            equity_amount.sum().alias("funding_equity_total_usd"),
            pl.len().alias("funding_round_count"),
            equity_amount.is_not_null().sum().alias("funding_equity_round_count"),
            pl.col("amount_sold_usd").last().alias("funding_last_amount_usd"),
            pl.col("filing_date").max().alias("funding_last_date"),
            pl.col("filing_date").min().alias("funding_first_date"),
            pl.col("offering_amount_usd").sum().alias("funding_offered_total_usd"),
            pl.col("investor_count").max().alias("funding_max_investors"),
            pl.col("ENTITYNAME").last().alias("formd_issuer_name"),
            pl.col("STATEORCOUNTRY").last().alias("formd_state"),
            pl.col("INDUSTRYGROUPTYPE").last().alias("formd_industry"),
            pl.col("REVENUERANGE").last().alias("formd_revenue_range"),
            pl.struct(
                pl.col("filing_date").alias("filing_date"),
                pl.col("first_sale_date").alias("first_sale_date"),
                pl.col("amount_sold_usd").alias("amount_sold_usd"),
                pl.col("offering_amount_usd").alias("offering_amount_usd"),
                pl.col("investor_count").alias("investor_count"),
                pl.col("is_equity").alias("is_equity"),
                pl.col("is_debt").alias("is_debt"),
            )
            .alias("funding_rounds"),
        )
    )
    return rounds.with_columns(
        pl.lit("sec_form_d").alias("funding_source"),
        pl.lit(datetime.now(tz=UTC).date().isoformat()).alias("funding_as_of"),
    )


def add_exempt_offerings(funding: pl.DataFrame, resolved: pl.DataFrame) -> pl.DataFrame:
    """Fold Reg CF and Reg A raises in alongside the Form D history.

    Kept in separate columns rather than summed into
    ``funding_total_usd``: a $500k crowdfunding raise and a $50M Series C
    are not comparable, and a consumer that wants one total can add them
    knowingly. Tenants whose only filings are Reg CF or Reg A appear as
    new rows so they are not lost.
    """
    from pipeline.company_enrichment import exempt

    try:
        offerings = exempt.ingest()
    except Exception as exc:
        logger.warning("skipping Reg CF/Reg A: %s", exc)
        return funding
    if offerings.is_empty():
        return funding

    with_cik = resolved.filter(pl.col("cik").is_not_null()).select("ats", "slug", "cik")
    matched = with_cik.join(
        offerings.select("cik", "offering_amount_usd", "filing_date", "source"),
        on="cik",
        how="inner",
    )
    if matched.is_empty():
        return funding

    aggregated = matched.group_by("ats", "slug").agg(
        pl.col("offering_amount_usd").sum().alias("exempt_total_usd"),
        pl.len().alias("exempt_round_count"),
        pl.col("filing_date").max().alias("exempt_last_date"),
        pl.col("source").unique().sort().str.join(",").alias("exempt_sources"),
    )
    if funding.is_empty():
        return aggregated
    return funding.join(aggregated, on=["ats", "slug"], how="full", coalesce=True)


def corroborate(funding: pl.DataFrame, cohort: pl.DataFrame) -> pl.DataFrame:
    """Upgrade low-confidence CIK matches that agree on state.

    Entity resolution grades a single-token private name match as "low"
    because EDGAR is full of unrelated filers sharing a common word. If
    the Form D issuer's state matches the state the tenant is hiring in,
    that is independent evidence the CIK is the right one.
    """
    states = cohort.select(
        "ats",
        "slug",
        pl.col("sample_location")
        .map_elements(_state_from_location, return_dtype=pl.String)
        .alias("posting_state"),
    )
    return funding.join(states, on=["ats", "slug"], how="left").with_columns(
        (
            (pl.col("posting_state") != "")
            & (pl.col("posting_state") == pl.col("formd_state"))
        ).alias("funding_state_corroborated")
    )


def run() -> pl.DataFrame:
    config.ensure_dirs()
    resolved = pl.read_parquet(config.RESOLVED_PARQUET)
    cohort = pl.read_parquet(config.COHORT_PARQUET)

    filings = load_filings()
    offerings = normalize_offerings(filings)
    funding = build_funding(offerings, resolved)
    formd_tenants = funding.height
    funding = add_exempt_offerings(funding, resolved)
    if funding.is_empty():
        logger.warning("no funding matches found")
        funding.write_parquet(config.FUNDING_PARQUET)
        return funding
    funding = corroborate(funding, cohort)
    funding.write_parquet(config.FUNDING_PARQUET)

    total = resolved.height
    exempt_tenants = (
        funding.filter(pl.col("exempt_round_count").is_not_null()).height
        if "exempt_round_count" in funding.columns
        else 0
    )
    print("\n=== Funding history (SEC Form D + Reg CF + Reg A) ===")
    print(f"Form D filings loaded        : {filings.height:>8,}")
    print(f"Distinct operating offerings : {offerings.height:>8,}")
    print(f"Tenants with a Form D history: {formd_tenants:>8,}  ({formd_tenants / total:.1%} of cohort)")
    print(f"Tenants with Reg CF / Reg A  : {exempt_tenants:>8,}")
    print(f"Tenants with any funding     : {funding.height:>8,}  ({funding.height / total:.1%} of cohort)")
    print(
        f"  state-corroborated         : "
        f"{funding.filter(pl.col('funding_state_corroborated').fill_null(False)).height:>8,}"
    )
    print("\nRounds per company:")
    print(
        funding.group_by("funding_round_count")
        .agg(pl.len().alias("companies"))
        .sort("funding_round_count")
        .head(10)
        .to_pandas()
        .to_string(index=False)
    )
    print("\nTop 15 by equity raised (debt-only offerings excluded):")
    print(
        funding.unique(subset=["formd_issuer_name"], keep="first")
        .sort("funding_equity_total_usd", descending=True, nulls_last=True)
        .head(15)
        .select(
            "formd_issuer_name",
            (pl.col("funding_equity_total_usd") / 1e6).round(1).alias("equity_$M"),
            "funding_equity_round_count",
            "funding_last_date",
            "funding_state_corroborated",
        )
        .to_pandas()
        .to_string(index=False)
    )
    dated = funding.filter(pl.col("funding_last_date").is_not_null()).height
    print(f"\nWith a parsed last-round date: {dated:,} / {funding.height:,}")
    print(f"\nwrote {config.FUNDING_PARQUET}")
    return funding


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

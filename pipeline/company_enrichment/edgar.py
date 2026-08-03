"""SEC EDGAR identity — the CIK spine everything else joins to.

Two public-domain files give near-complete coverage of registrants:

``company_tickers_exchange.json``
    ~10.4k currently-listed companies as ``(cik, name, ticker,
    exchange)``. This is what makes a tenant *public* and supplies the
    ticker that :mod:`.marketcap` prices.

``cik-lookup-data.txt``
    Every name EDGAR has ever issued a CIK to, as ``NAME:CIK:`` lines —
    roughly a million entries including private Form D issuers and
    historical names. This is what lets a private startup resolve to a
    CIK so :mod:`.formd` can find its offerings.

Both are public domain (17 CFR 200.80(b)). Neither needs a key.

A company can hold several CIKs and a CIK several names, so the output
is deliberately one row per ``(cik, name)`` pair rather than per
company; the resolver picks a winner using the cohort name.
"""

from __future__ import annotations

import json
import logging

import polars as pl

from pipeline.company_enrichment import config, sechttp
from pipeline.company_enrichment.normalize import name_key

logger = logging.getLogger(__name__)


def _load_tickers() -> pl.DataFrame:
    payload = json.loads(sechttp.get(config.SEC_COMPANY_TICKERS_EXCHANGE_URL, suffix=".json"))
    fields = [str(f) for f in payload["fields"]]
    rows = payload["data"]
    df = pl.DataFrame(
        {
            field: [row[i] for row in rows]
            for i, field in enumerate(fields)
        }
    )
    logger.info("company_tickers_exchange: %d listed registrants", df.height)
    return df.select(
        pl.col("cik").cast(pl.Int64),
        pl.col("name").cast(pl.String),
        pl.col("ticker").cast(pl.String),
        pl.col("exchange").cast(pl.String),
    )


def _load_cik_lookup() -> pl.DataFrame:
    """Parse ``NAME:CIK:`` lines into a frame.

    The file is latin-1 and contains names with embedded colons, so the
    CIK is taken from the *last* populated field rather than by
    splitting on the first colon.
    """
    raw = sechttp.get(
        "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", suffix=".txt"
    )
    text = raw.decode("latin-1")
    names: list[str] = []
    ciks: list[int] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.rstrip(":").rsplit(":", 1)
        if len(parts) != 2:
            continue
        name, cik_text = parts
        if not cik_text.isdigit() or not name.strip():
            continue
        names.append(name.strip())
        ciks.append(int(cik_text))
    logger.info("cik-lookup-data: %d (name, cik) pairs", len(names))
    return pl.DataFrame({"name": names, "cik": ciks}, schema={"name": pl.String, "cik": pl.Int64})


def ingest(*, force: bool = False) -> pl.DataFrame:
    """Build the combined EDGAR identity table."""
    config.ensure_dirs()
    required = ("name_key_raw", "name_key_core", "is_public")
    if config.EDGAR_PARQUET.exists() and not force:
        existing = pl.read_parquet(config.EDGAR_PARQUET)
        if all(c in existing.columns for c in required):
            logger.info("reusing %s", config.EDGAR_PARQUET)
            return existing
        logger.info("schema of %s is stale — rebuilding", config.EDGAR_PARQUET)

    tickers = _load_tickers()
    lookup = _load_cik_lookup()

    # The ticker file is authoritative for listing status; the lookup
    # file contributes every other name EDGAR knows, including former
    # names that a tenant may still trade under.
    combined = (
        lookup.join(tickers.select("cik", "ticker", "exchange"), on="cik", how="left")
        .with_columns(pl.lit("cik_lookup").alias("name_source"))
        .vstack(
            tickers.select(
                "name",
                "cik",
                "ticker",
                "exchange",
                pl.lit("ticker_file").alias("name_source"),
            )
        )
        .unique(subset=["cik", "name"], keep="first")
    )

    # Both key forms are indexed: EDGAR names carry the full legal form
    # ("LEIDOS HOLDINGS, INC.") while tenants use the trading name
    # ("Leidos"), so only the suffix-stripped key bridges them.
    df = (
        combined.with_columns(
            pl.col("name")
            .map_elements(
                lambda s: name_key(s, keep_suffix=True), return_dtype=pl.String
            )
            .alias("name_key_raw"),
            pl.col("name")
            .map_elements(name_key, return_dtype=pl.String)
            .alias("name_key_core"),
            pl.col("ticker").is_not_null().alias("is_public"),
        )
        .filter(pl.col("name_key_raw") != "")
        .sort(["cik", "name"])
    )

    df.write_parquet(config.EDGAR_PARQUET)
    logger.info("wrote %s (%d rows)", config.EDGAR_PARQUET, df.height)
    return df


def run(*, force: bool = False) -> pl.DataFrame:
    df = ingest(force=force)
    print("\n=== SEC EDGAR identity ===")
    print(f"(name, cik) pairs        : {df.height:,}")
    print(f"Distinct CIKs            : {df['cik'].n_unique():,}")
    print(f"Listed (has ticker)      : {df.filter(pl.col('is_public')).height:,}")
    print(
        f"Distinct listed CIKs     : "
        f"{df.filter(pl.col('is_public'))['cik'].n_unique():,}"
    )
    print("\nBy exchange:")
    print(
        df.filter(pl.col("is_public"))
        .unique(subset=["cik"])
        .group_by("exchange")
        .agg(pl.len().alias("companies"))
        .sort("companies", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.EDGAR_PARQUET}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

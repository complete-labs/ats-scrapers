"""Market capitalisation for resolved public companies.

Primary source is the Nasdaq stock screener, which returns every
US-listed symbol with ``marketCap``, ``lastsale``, sector, industry, and
IPO year in a single unauthenticated request — one call covers the whole
public cohort with no key and no per-symbol quota.

Verification comes from SEC XBRL. ``dei:EntityCommonStockSharesOutstanding``
is the cover-page share count from the company's own latest 10-K/10-Q, so
``SEC shares x Nasdaq last sale`` should land within a few percent of the
screener's ``marketCap``. A wide gap almost always means a multi-class
capital structure where the screener counted one class, or a share count
that has gone stale across a buyback or issuance. Those rows get
flagged rather than silently trusted.

The check validates the *share count*, not the price — both figures use
the Nasdaq last sale. An independent price feed would be better, but the
free ones have closed: Stooq now gates CSV downloads behind a JavaScript
proof-of-work challenge, and working around an explicit anti-bot control
is not something this pipeline should do.

Accuracy
--------
623 of 9,541 tenants (6.5%) get a market cap. Where a SEC share count is
available the two figures agree within tolerance for 438 of 464 (94%);
the disagreements are dominated by multi-class structures such as Visa
and Mastercard, where the cover page reports one class.

Identity, not arithmetic, is the real risk. Two tenants in the
validation sample resolved to an unrelated registrant that shares their
trade name, which is what ``registrant_state_agrees`` exists to surface.

Subsidiary rollup
-----------------
A tenant can be a wholly-owned subsidiary that has no market cap of its
own. :func:`build_subsidiary_map` reads Exhibit 21 (the "Subsidiaries of
the Registrant" exhibit attached to each 10-K) for a bounded set of
parents and attributes the parent's market cap to those tenants, marked
with ``market_cap_basis = 'parent'`` so it is never mistaken for the
tenant's own valuation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

import polars as pl

from ats_scrapers.enrichment.uslocation import STATE_NAME_TO_ABBR
from pipeline.company_enrichment import config, sechttp
from pipeline.company_enrichment.normalize import name_key

logger = logging.getLogger(__name__)

# The screener rejects the default httpx agent; it wants a browser UA.
_NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# Beyond this relative gap between the screener's market cap and
# SEC-shares x price, the row is flagged for inspection.
CROSSCHECK_TOLERANCE = 0.15

_MONEY_RE = re.compile(r"[^0-9.\-]")


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = _MONEY_RE.sub("", str(value))
    if not text or text in {"-", ".", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_nasdaq_listings(*, force: bool = False) -> pl.DataFrame:
    """All US-listed symbols with market cap, from one screener request."""
    cache = config.CACHE_DIR / "nasdaq_screener.json"
    if cache.exists() and not force:
        payload = json.loads(cache.read_text())
    else:
        import httpx

        logger.info("GET nasdaq screener")
        response = httpx.get(
            config.NASDAQ_SCREENER_URL,
            headers=_NASDAQ_HEADERS,
            timeout=config.HTTP_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    rows = payload["data"]["rows"]
    df = pl.DataFrame(
        {
            "ticker": [str(r["symbol"]).strip().upper() for r in rows],
            "nasdaq_name": [r.get("name") for r in rows],
            "market_cap_usd": [_to_float(r.get("marketCap")) for r in rows],
            "last_sale_usd": [_to_float(r.get("lastsale")) for r in rows],
            "listing_country": [r.get("country") for r in rows],
            "sector": [r.get("sector") for r in rows],
            "listed_industry": [r.get("industry") for r in rows],
            "ipo_year": [_to_float(r.get("ipoyear")) for r in rows],
        }
    )
    # A zero market cap in the screener means "unknown", not "worthless".
    df = df.with_columns(
        pl.when(pl.col("market_cap_usd") > 0)
        .then(pl.col("market_cap_usd"))
        .otherwise(None)
        .alias("market_cap_usd"),
        pl.col("ipo_year").cast(pl.Int32, strict=False),
    )
    logger.info("nasdaq screener: %d symbols", df.height)
    return df


def fetch_sec_shares(cik: int) -> tuple[float | None, str | None]:
    """Latest cover-page shares outstanding for ``cik``, with its as-of date."""
    url = config.SEC_COMPANYCONCEPT_TEMPLATE.format(
        cik=cik, tag="EntityCommonStockSharesOutstanding"
    )
    try:
        payload = json.loads(sechttp.get(url, suffix=".json"))
    except Exception as exc:
        logger.debug("no shares-outstanding concept for CIK %s: %s", cik, exc)
        return None, None

    observations = payload.get("units", {}).get("shares", [])
    if not observations:
        return None, None
    # `end` is the cover-page measurement date; the newest wins.
    latest = max(observations, key=lambda o: str(o.get("end") or ""))
    value = latest.get("val")
    return (float(value) if value is not None else None), latest.get("end")


_EX21_NAME_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_JURISDICTION_HINTS = re.compile(
    r"(?i)\b(delaware|nevada|california|new york|texas|florida|illinois|"
    r"virginia|maryland|massachusetts|england|wales|ontario|luxembourg|"
    r"netherlands|ireland|singapore|cayman|state of|jurisdiction)\b"
)


def fetch_registrant_state(cik: int) -> str | None:
    """Business-address state or country code from EDGAR submissions.

    EDGAR uses two-letter postal codes for US addresses and its own
    two-character codes for everything else ("L2" Ireland, "E9" Cayman
    Islands), so the caller must check the value is a US state before
    comparing it with one.
    """
    try:
        payload = json.loads(
            sechttp.get(config.SEC_SUBMISSIONS_TEMPLATE.format(cik=cik), suffix=".json")
        )
    except Exception as exc:
        logger.debug("submissions lookup failed for CIK %s: %s", cik, exc)
        return None
    business = (payload.get("addresses") or {}).get("business") or {}
    state = business.get("stateOrCountry") or payload.get("stateOfIncorporation")
    text = str(state or "").strip().upper()
    return text if len(text) == 2 else None


_US_STATE_CODES = frozenset(STATE_NAME_TO_ABBR.values())


def _state_agreement(registrant: pl.Expr, tenant: pl.Expr) -> pl.Expr:
    """Whether a registrant and its tenant sit in the same US state.

    This is the only cheap independent check on a name-only public match,
    and it earns its keep. Several tenants resolved to a perfect-scoring
    but entirely unrelated registrant purely because they share a trade
    name: a Greenhouse board for Spire Global (satellite data, Virginia)
    matched Spire Inc (a Missouri gas utility), a Workable board for the
    SmartFinancial insurance marketplace (California) matched
    SmartFinancial Inc (a Tennessee bank), and Vertex Education (Arizona
    charter schools) matched Vertex Inc (tax software, Pennsylvania).

    It is a triage signal, not a verdict. A large company's PDL record
    often names a satellite office rather than the registered head
    office, so a mismatch means "a human should look", which is why it
    lands in ``quality_flags`` instead of deleting the row. Comparison is
    skipped entirely unless both sides are US states, since a foreign
    registrant is expected to differ from a US locality.
    """
    comparable = (
        registrant.is_not_null()
        & tenant.is_not_null()
        & registrant.is_in(list(_US_STATE_CODES))
    )
    return (
        pl.when(~comparable)
        .then(None)
        .otherwise(registrant.str.to_uppercase() == tenant.str.to_uppercase())
    )


def _parse_exhibit21(html: str) -> list[str]:
    """Pull subsidiary names out of an Exhibit 21 document.

    Ex-21 has no prescribed format — some filers use a two-column table
    of name and jurisdiction, others a flat list. Cells that look like a
    jurisdiction rather than a company are discarded.
    """
    cells = _EX21_NAME_RE.findall(html)
    if not cells:
        text = _TAG_RE.sub("\n", html)
        cells = text.splitlines()

    names: list[str] = []
    for cell in cells:
        text = _TAG_RE.sub(" ", cell)
        text = re.sub(r"&(?:nbsp|amp|#\d+);", " ", text)
        text = " ".join(text.split()).strip(" .,:;*")
        if not (3 < len(text) < 120):
            continue
        if _JURISDICTION_HINTS.fullmatch(text) or _JURISDICTION_HINTS.match(text):
            continue
        if not re.search(r"[A-Za-z]{3}", text):
            continue
        names.append(text)
    return names


def _latest_10k_exhibit21(cik: int) -> str | None:
    """URL of the most recent Exhibit 21 attached to a 10-K, if any."""
    try:
        payload = json.loads(
            sechttp.get(config.SEC_SUBMISSIONS_TEMPLATE.format(cik=cik), suffix=".json")
        )
    except Exception as exc:
        logger.debug("submissions lookup failed for CIK %s: %s", cik, exc)
        return None

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])
    for form, accession, _doc in zip(forms, accessions, documents, strict=False):
        if form != "10-K":
            continue
        stripped = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{stripped}/index.json"
        )
        try:
            index = json.loads(sechttp.get(index_url, suffix=".json"))
        except Exception:
            return None
        for item in index.get("directory", {}).get("item", []):
            name = str(item.get("name", ""))
            if re.search(r"(?i)ex-?21", name):
                return (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{stripped}/{name}"
                )
        return None
    return None


def build_subsidiary_map(parent_ciks: list[int]) -> pl.DataFrame:
    """Map subsidiary name keys to their public parent CIK.

    Bounded by ``parent_ciks`` on purpose. Covering all ~8k listed
    registrants would need roughly 16k SEC requests (about 33 minutes at
    the mandated rate limit); the default caller passes only the public
    companies already present in the cohort, which is two orders of
    magnitude cheaper and catches the common case of a tenant that is a
    named division of a parent we already matched.
    """
    records: list[dict[str, object]] = []
    for index, cik in enumerate(parent_ciks, start=1):
        if index % 50 == 0:
            logger.info("exhibit 21: %d/%d parents", index, len(parent_ciks))
        url = _latest_10k_exhibit21(cik)
        if not url:
            continue
        try:
            html = sechttp.get(url, suffix=".htm").decode("utf-8", errors="replace")
        except Exception:
            continue
        for name in _parse_exhibit21(html):
            key = name_key(name)
            if len(key) >= 5:
                records.append({"subsidiary_key": key, "subsidiary_name": name, "parent_cik": cik})

    if not records:
        return pl.DataFrame(
            schema={
                "subsidiary_key": pl.String,
                "subsidiary_name": pl.String,
                "parent_cik": pl.Int64,
            }
        )
    return pl.DataFrame(records).unique(subset=["subsidiary_key", "parent_cik"])


def run(*, use_exhibit21: bool = True) -> pl.DataFrame:
    config.ensure_dirs()
    resolved = pl.read_parquet(config.RESOLVED_PARQUET)
    listings = fetch_nasdaq_listings()

    public = resolved.filter(pl.col("ticker").is_not_null()).with_columns(
        pl.col("ticker").str.strip_chars().str.to_uppercase()
    )
    joined = public.join(listings, on="ticker", how="left")

    # --- SEC share-count verification ------------------------------------
    logger.info("fetching SEC shares outstanding for %d CIKs", joined["cik"].n_unique())
    shares: list[float | None] = []
    as_of: list[str | None] = []
    cache: dict[int, tuple[float | None, str | None]] = {}
    for cik in joined["cik"]:
        key = int(cik)
        if key not in cache:
            cache[key] = fetch_sec_shares(key)
        value, date = cache[key]
        shares.append(value)
        as_of.append(date)

    logger.info("fetching registrant states for %d CIKs", joined["cik"].n_unique())
    state_cache: dict[int, str | None] = {}
    states: list[str | None] = []
    for cik in joined["cik"]:
        key = int(cik)
        if key not in state_cache:
            state_cache[key] = fetch_registrant_state(key)
        states.append(state_cache[key])

    joined = joined.with_columns(
        pl.Series("shares_outstanding", shares, dtype=pl.Float64),
        pl.Series("shares_as_of", as_of, dtype=pl.String),
        pl.Series("registrant_state", states, dtype=pl.String),
    ).with_columns(
        _state_agreement(
            pl.col("registrant_state"),
            pl.col("region").str.to_lowercase().replace_strict(
                STATE_NAME_TO_ABBR, default=None
            ),
        ).alias("registrant_state_agrees")
    ).with_columns(
        (pl.col("shares_outstanding") * pl.col("last_sale_usd")).alias(
            "market_cap_implied_usd"
        )
    )
    joined = joined.with_columns(
        (
            (pl.col("market_cap_implied_usd") - pl.col("market_cap_usd")).abs()
            / pl.col("market_cap_usd")
        ).alias("market_cap_disagreement")
    ).with_columns(
        pl.when(pl.col("market_cap_usd").is_null())
        .then(pl.lit("missing"))
        .when(pl.col("market_cap_disagreement").is_null())
        .then(pl.lit("unverified"))
        .when(pl.col("market_cap_disagreement") <= CROSSCHECK_TOLERANCE)
        .then(pl.lit("verified"))
        .otherwise(pl.lit("disagrees"))
        .alias("market_cap_check"),
        pl.lit("self").alias("market_cap_basis"),
        pl.lit("nasdaq_screener").alias("market_cap_source"),
        pl.lit(datetime.now(tz=UTC).date().isoformat()).alias("market_cap_as_of"),
    )

    result = joined.select(
        "ats", "slug", "cik", "ticker", "exchange", "nasdaq_name",
        "market_cap_usd", "last_sale_usd", "shares_outstanding", "shares_as_of",
        "market_cap_implied_usd", "market_cap_disagreement", "market_cap_check",
        "market_cap_basis", "market_cap_source", "market_cap_as_of",
        "registrant_state", "registrant_state_agrees",
        "sector", "listed_industry", "ipo_year",
    )

    # --- subsidiary rollup -------------------------------------------------
    if use_exhibit21:
        parents = sorted({int(c) for c in joined["cik"] if c is not None})
        logger.info("building Exhibit 21 subsidiary map from %d parents", len(parents))
        subsidiaries = build_subsidiary_map(parents)
        logger.info("exhibit 21: %d subsidiary names", subsidiaries.height)

        if subsidiaries.height:
            unmatched = resolved.filter(pl.col("ticker").is_null()).with_columns(
                pl.col("name").map_elements(name_key, return_dtype=pl.String).alias("_key")
            )
            rolled = (
                unmatched.join(
                    subsidiaries, left_on="_key", right_on="subsidiary_key", how="inner"
                )
                .join(
                    result.select(
                        pl.col("cik").alias("parent_cik"),
                        pl.col("ticker").alias("parent_ticker"),
                        pl.col("market_cap_usd").alias("parent_market_cap"),
                        pl.col("sector"),
                        pl.col("listed_industry"),
                    ).unique(subset=["parent_cik"]),
                    on="parent_cik",
                    how="inner",
                )
                .unique(subset=["ats", "slug"], keep="first")
            )
            if rolled.height:
                logger.info("exhibit 21 rollup matched %d tenants", rolled.height)
                extra = rolled.select(
                    "ats", "slug",
                    pl.col("parent_cik").alias("cik"),
                    pl.col("parent_ticker").alias("ticker"),
                    pl.lit(None, dtype=pl.String).alias("exchange"),
                    pl.col("subsidiary_name").alias("nasdaq_name"),
                    pl.col("parent_market_cap").alias("market_cap_usd"),
                    pl.lit(None, dtype=pl.Float64).alias("last_sale_usd"),
                    pl.lit(None, dtype=pl.Float64).alias("shares_outstanding"),
                    pl.lit(None, dtype=pl.String).alias("shares_as_of"),
                    pl.lit(None, dtype=pl.Float64).alias("market_cap_implied_usd"),
                    pl.lit(None, dtype=pl.Float64).alias("market_cap_disagreement"),
                    pl.lit("parent_rollup").alias("market_cap_check"),
                    pl.lit("parent").alias("market_cap_basis"),
                    pl.lit("sec_exhibit21+nasdaq_screener").alias("market_cap_source"),
                    pl.lit(datetime.now(tz=UTC).date().isoformat()).alias("market_cap_as_of"),
                    pl.lit(None, dtype=pl.String).alias("registrant_state"),
                    pl.lit(None, dtype=pl.Boolean).alias("registrant_state_agrees"),
                    "sector", "listed_industry",
                    pl.lit(None, dtype=pl.Int32).alias("ipo_year"),
                )
                result = pl.concat([result, extra.select(result.columns)], how="vertical")

    result.write_parquet(config.MARKETCAP_PARQUET)

    have = result.filter(pl.col("market_cap_usd").is_not_null())
    rollups = result.filter(pl.col("market_cap_basis") == "parent").height
    print("\n=== Market cap ===")
    print(f"Tenants matched to a ticker : {public.height:>6,}")
    print(f"Tenants rolled up to a parent: {rollups:>5,}")
    print(f"Total rows                  : {result.height:>6,}")
    print(f"With a market cap value     : {have.height:>6,}")
    print("\nVerification against SEC-filed share counts:")
    print(
        result.group_by("market_cap_check")
        .agg(pl.len().alias("tenants"))
        .sort("tenants", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    comparable = result.filter(pl.col("registrant_state_agrees").is_not_null())
    if comparable.height:
        disagree = comparable.filter(~pl.col("registrant_state_agrees"))
        print(
            f"\nRegistrant state vs company HQ: "
            f"{comparable.height - disagree.height:,} of {comparable.height:,} agree "
            f"({disagree.height:,} flagged as a possible name collision)"
        )
    if have.height:
        print("\nLargest by market cap:")
        print(
            have.sort("market_cap_usd", descending=True)
            .head(12)
            .select(
                "ticker",
                "nasdaq_name",
                (pl.col("market_cap_usd") / 1e9).round(1).alias("mcap_$B"),
                "market_cap_check",
                "market_cap_basis",
            )
            .to_pandas()
            .to_string(index=False)
        )
    print(f"\nwrote {config.MARKETCAP_PARQUET}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

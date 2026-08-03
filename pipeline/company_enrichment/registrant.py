"""Registrant facts EDGAR states outright, for every matched CIK.

``data.sec.gov/submissions/CIK##########.json`` is already fetched three
times over in this package — for the latest 10-K, for Exhibit 21, and for
the business-address state — and every caller parses out the one field it
came for and drops the rest. Three of those fields are worth keeping.

``ein``
    The registrant's Employer Identification Number. This is the only
    **exact** identifier in the whole pipeline. Every other cross-source
    join here is a name match with a similarity score and a collision
    problem; an EIN either matches or it does not. It is what lets
    :mod:`.form5500` attach a filed employee count to a private company
    without guessing, and it is a far stronger corroboration of a CIK
    match than any name-shape heuristic.

``registrant_city``
    The business-address city. Read together with the state it is a
    *filed* head office, where the locality :mod:`.pdl` supplies is
    inferred, and it is the only location available for a tenant that
    matched EDGAR but not PDL. :mod:`.profile` composes the two into
    ``headquarters``.

``registrant_state``
    The business-address state. :mod:`.marketcap` already computes this,
    but only for tenants that reached it — the ones carrying a ticker.
    That leaves the state check, which is the most precise error detector
    in the pipeline, running on 519 of 2,621 CIK-bearing tenants and on
    *none* of the tenants with a funding history. Because this stage
    visits every CIK rather than every ticker, the state is captured for
    2,461 of them at no extra cost; wiring the check itself up to the
    wider set is a separate change and has not been made here.

``entity_type`` and ``sic`` come along for free and are worth carrying:
EDGAR marks investment vehicles and shell registrants distinctly from
operating companies, which is a cheap sanity check on a match to a
tenant that demonstrably employs people.

Cost is one request per CIK, throttled and cached by :mod:`.sechttp`, so
a re-run is free and a cold run is a few minutes for the whole cohort.

Licence: SEC EDGAR is a US Government work in the public domain
(17 CFR 200.80).
"""

from __future__ import annotations

import json
import logging

import polars as pl

from pipeline.company_enrichment import config, sechttp

logger = logging.getLogger(__name__)

SCHEMA: dict[str, pl.DataType] = {
    "cik": pl.Int64,
    "ein": pl.String,
    "registrant_name": pl.String,
    "registrant_city": pl.String,
    "registrant_state": pl.String,
    "registrant_entity_type": pl.String,
    "registrant_sic": pl.String,
}

# EDGAR writes an unknown EIN as a run of zeros rather than omitting the
# field, and a few registrants carry an obvious placeholder.
_NULL_EINS = frozenset({"", "000000000", "00-0000000", "999999999"})


def _clean_ein(value: object) -> str | None:
    """Nine digits, or ``None``.

    EDGAR is inconsistent about the hyphen ("52-1568099" and
    "521568099" both occur) while DOL always stores nine bare digits, so
    the punctuation has to go for the join to land.
    """
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 9 or digits in _NULL_EINS or digits == "0" * 9:
        return None
    return digits


def fetch_profile(cik: int) -> dict[str, object] | None:
    """Registrant facts for one CIK, or ``None`` if EDGAR has no record."""
    try:
        payload = json.loads(
            sechttp.get(config.SEC_SUBMISSIONS_TEMPLATE.format(cik=cik), suffix=".json")
        )
    except Exception as exc:
        logger.debug("submissions lookup failed for CIK %s: %s", cik, exc)
        return None

    business = (payload.get("addresses") or {}).get("business") or {}
    state = business.get("stateOrCountry") or payload.get("stateOfIncorporation")
    state_text = str(state or "").strip().upper()

    city = str(business.get("city") or "").strip()

    return {
        "cik": int(cik),
        "ein": _clean_ein(payload.get("ein")),
        "registrant_name": payload.get("name"),
        # Paired with the state below, this is the registrant's stated
        # head office — the only *filed* location in the pipeline, where
        # PDL's is inferred. `profile` uses it to place a tenant that PDL
        # never matched.
        "registrant_city": city or None,
        # EDGAR uses two-letter postal codes for US addresses and its own
        # two-character codes for everything else ("L2" Ireland, "E9"
        # Cayman), so a caller must confirm this is a US state before
        # comparing it with one. See `marketcap._state_agreement`.
        "registrant_state": state_text if len(state_text) == 2 else None,
        "registrant_entity_type": payload.get("entityType"),
        "registrant_sic": payload.get("sic"),
    }


def build(ciks: list[int]) -> pl.DataFrame:
    """Fetch a profile for each CIK, skipping the ones EDGAR does not know."""
    records: list[dict[str, object]] = []
    for index, cik in enumerate(sorted(set(ciks)), start=1):
        if index % 200 == 0:
            logger.info("registrant profiles: %d/%d", index, len(set(ciks)))
        profile = fetch_profile(cik)
        if profile is not None:
            records.append(profile)
    return pl.DataFrame(records, schema=SCHEMA) if records else pl.DataFrame(schema=SCHEMA)


def load() -> pl.DataFrame:
    """Read the built profiles, or an empty frame if the stage has not run."""
    if not config.REGISTRANT_PARQUET.exists():
        logger.warning(
            "%s missing — run the `registrant` stage to enable the EIN join",
            config.REGISTRANT_PARQUET,
        )
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(config.REGISTRANT_PARQUET)


def run(*, force: bool = False) -> pl.DataFrame:
    config.ensure_dirs()
    if config.REGISTRANT_PARQUET.exists() and not force:
        logger.info("reusing %s", config.REGISTRANT_PARQUET)
        return _report(pl.read_parquet(config.REGISTRANT_PARQUET))

    resolved = pl.read_parquet(config.RESOLVED_PARQUET)
    ciks = [int(c) for c in resolved["cik"].drop_nulls().unique()]
    logger.info("fetching EDGAR registrant profiles for %d CIKs", len(ciks))

    profiles = build(ciks)
    profiles.write_parquet(config.REGISTRANT_PARQUET)
    logger.info("wrote %s (%d registrants)", config.REGISTRANT_PARQUET, profiles.height)
    return _report(profiles)


def _report(df: pl.DataFrame) -> pl.DataFrame:
    total = df.height
    print("\n=== EDGAR registrant profiles ===")
    print(f"Registrants                : {total:>6,}")
    if not total:
        return df
    with_ein = df.filter(pl.col("ein").is_not_null()).height
    with_state = df.filter(pl.col("registrant_state").is_not_null()).height
    with_city = df.filter(pl.col("registrant_city").is_not_null()).height
    print(f"With an EIN                : {with_ein:>6,}  ({with_ein / total:.1%})")
    print(f"With a business-address state: {with_state:>4,}  ({with_state / total:.1%})")
    print(f"With a business-address city : {with_city:>4,}  ({with_city / total:.1%})")
    print("\nEntity type as EDGAR classifies it:")
    print(
        df.group_by("registrant_entity_type")
        .agg(pl.len().alias("registrants"))
        .sort("registrants", descending=True)
        .head(8)
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.REGISTRANT_PARQUET}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

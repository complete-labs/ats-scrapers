"""Assemble the final enrichment table keyed on ``(ats, slug)``.

Joins the stage outputs into one row per tenant. Every enriched value
travels with its ``*_source`` and ``*_as_of``, and the identity match
carries its own score and confidence, so a consumer can always tell a
verified figure from a fuzzy guess. Nothing is silently averaged or
back-filled across sources.

One value is deliberately dropped rather than joined: a public
company's funding history. See :func:`_withdraw_funding_from_public`.

The result is **private**. It is written to the gitignored output
directory and no part of it reaches
``.github/scripts/publish_companies.py``, which keeps publishing exactly
``ats,name,slug,url``. :func:`assert_public_schema_untouched` enforces
that so the separation cannot rot unnoticed.

``company_description`` is the one column here that is quoted rather than
computed — it is the company's own sentence about itself, taken from its
homepage, its postings, or its 10-K. It therefore travels with
``company_description_url`` as well as the usual source and date, and
with ``company_description_name_corroborated``, which records whether the
text names the tenant it is attached to. See :mod:`.profile`.

``jobs_company`` is carried through from the cohort because the published
jobs dataset has no ``slug``: a posting identifies its employer only by
``ats_type`` + ``company``, so ``(ats, jobs_company)`` is the sole key an
internal consumer can join a posting to this table on. It is not unique —
one employer can run several boards under one display name — so a
consumer must aggregate on it rather than assume one row per key.
"""

from __future__ import annotations

import logging

import polars as pl

from pipeline.company_enrichment import config, pdl

logger = logging.getLogger(__name__)

PUBLIC_COMPANY_COLUMNS = ("ats", "name", "slug", "url")

# Everything the Form D / Reg CF / Reg A stage contributes. Grouped so
# it can be withdrawn as a unit for public companies — see `build`.
FUNDING_COLUMNS = (
    "funding_total_usd",
    "funding_equity_total_usd",
    "funding_round_count",
    "funding_equity_round_count",
    "funding_last_amount_usd",
    "funding_last_date",
    "funding_first_date",
    "funding_offered_total_usd",
    "funding_max_investors",
    "funding_state_corroborated",
    "formd_issuer_name",
    "formd_state",
    "formd_industry",
    "formd_revenue_range",
    "funding_rounds",
    "exempt_total_usd",
    "exempt_round_count",
    "exempt_last_date",
    "exempt_sources",
    "funding_source",
    "funding_as_of",
)

# Column order of the delivered table, grouped by concern.
OUTPUT_COLUMNS = (
    # identity
    "ats",
    "slug",
    "name",
    # The jobs-side display name, i.e. the join key into the published
    # postings dataset. See the module docstring.
    "jobs_company",
    "display_name",
    "resolved_name",
    "url",
    "postings",
    "join_method",
    # what the company does, and where to apply. `url` above is the
    # directory's raw value; `careers_url` is the same board URL after
    # it has been confirmed to resolve back to this tenant, or repaired
    # from a posting URL when it did not.
    "company_description",
    "company_description_source",
    "company_description_url",
    "company_description_as_of",
    "company_description_name_corroborated",
    "careers_url",
    "careers_url_source",
    "careers_url_verified",
    "company_careers_url",
    "company_careers_url_source",
    "headquarters",
    "headquarters_source",
    # resolution provenance
    "domain",
    "linkedin_url",
    "cik",
    "ticker",
    "exchange",
    "is_public",
    "match_method",
    "match_score",
    "matched_variant",
    "matched_on_slug_only",
    "cik_confidence",
    # market cap
    "market_cap_usd",
    "shares_outstanding",
    "market_cap_implied_usd",
    "market_cap_disagreement",
    "market_cap_check",
    "market_cap_basis",
    "market_cap_source",
    "market_cap_as_of",
    "registrant_state",
    "registrant_state_agrees",
    "sector",
    "listed_industry",
    "ipo_year",
    # funding
    *FUNDING_COLUMNS,
    # team size
    "employee_count",
    "employee_count_as_of",
    "employee_count_source",
    "employee_count_floor",
    "employee_count_floor_source",
    "employee_count_floor_as_of",
    "employee_count_band",
    "employee_count_band_source",
    "employee_count_band_as_of",
    "employee_count_agrees_with_band",
    "employee_count_above_floor",
    "employee_count_band_conflict",
    # firmographics from PDL
    "founded",
    "locality",
    "region",
    "industry",
    # quality
    "quality_flags",
)


def _quality_flags() -> pl.Expr:
    """Detected internal contradictions, as a comma-separated list.

    Cross-source disagreement is the only automatic signal available that
    a match is wrong. The clearest case: a tenant resolves to a public
    company whose 10-K reports 300,000 staff while its PDL record says
    "51-200" — the two cannot describe the same company, so the PDL row
    (and the domain and LinkedIn URL that came with it) is suspect.
    """
    checks = [
        (
            pl.col("employee_count_agrees_with_band").eq(False),
            "headcount_contradicts_band",
        ),
        (
            pl.col("employee_count_band_conflict").fill_null("") != "",
            "band_suppressed",
        ),
        # A count below an independently filed floor is arithmetically
        # impossible, so one of the two is wrong. The clear-cut cases are
        # already withdrawn in `teamsize`; what reaches here is the
        # residue that is contradicted but not decisively.
        (
            pl.col("employee_count_above_floor").eq(False),
            "headcount_below_filed_floor",
        ),
        # A description that never names the company cannot be checked
        # against the tenant it is attached to. `profile` already drops
        # the combination of this and a shaky match; what reaches here
        # is a description hanging off a match that looked sound, which
        # is worth a second pair of eyes but not a withdrawal.
        #
        # Wikidata is excluded because its descriptions are written to
        # follow the name rather than repeat it — "American investment
        # bank and financial services corporation" is the house style,
        # so the flag would fire on every one of them and mean nothing.
        (
            pl.col("company_description").is_not_null()
            & (pl.col("company_description_source") != "wikidata")
            & pl.col("company_description_name_corroborated").eq(False),
            "description_name_unconfirmed",
        ),
        # A careers URL no ATS URL-shape rule recognises. Usually a
        # custom-domain careers site, which is genuine; occasionally a
        # directory entry that has gone stale.
        (
            pl.col("careers_url").is_not_null()
            & pl.col("careers_url_verified").eq(False),
            "careers_url_unverified",
        ),
        (pl.col("cik_confidence") == "low", "cik_low_confidence"),
        (pl.col("market_cap_check") == "disagrees", "market_cap_disagrees"),
        (pl.col("market_cap_basis") == "parent", "market_cap_is_parent"),
        (
            pl.col("registrant_state_agrees").eq(False),
            "registrant_state_mismatch",
        ),
        (
            pl.col("match_score").is_not_null() & (pl.col("match_score") < 95),
            "weak_name_match",
        ),
        (pl.col("matched_on_slug_only").eq(True), "matched_on_slug_only"),
    ]
    parts = [
        pl.when(condition).then(pl.lit(label)).otherwise(pl.lit(None))
        for condition, label in checks
    ]
    return pl.concat_list(parts).list.drop_nulls().list.join(",")


def assert_public_schema_untouched() -> None:
    """Fail loudly if the public companies schema has drifted.

    This package must stay invisible to the published dataset. If someone
    later teaches ``publish_companies.py`` to emit an enrichment column,
    private firmographics would start leaking into a public R2 object.
    """
    script = config.REPO_ROOT / ".github" / "scripts" / "publish_companies.py"
    if not script.exists():
        logger.warning("could not find %s to verify public schema", script)
        return
    text = script.read_text()
    for column in PUBLIC_COMPANY_COLUMNS:
        if f'"{column}"' not in text and f"'{column}'" not in text:
            raise AssertionError(
                f"publish_companies.py no longer references the public column "
                f"{column!r}; the public schema may have changed"
            )
    leaked = [
        name
        for name in (
            "market_cap",
            "funding_total",
            "employee_count",
            "linkedin_url",
            "cik",
            "company_description",
            "careers_url",
            "headquarters",
        )
        if name in text
    ]
    if leaked:
        raise AssertionError(
            f"publish_companies.py references private enrichment fields {leaked}; "
            "enrichment data must never reach the public dataset"
        )
    logger.info("verified public companies schema is still ats,name,slug,url")


def _withdraw_funding_from_public(out: pl.DataFrame) -> pl.DataFrame:
    """Drop the funding history of anything that trades publicly.

    Form D covers exempt offerings, so a listed company's rows are its
    pre-IPO raises plus the odd later private placement. Read next to a
    market capitalisation that is two orders of magnitude larger, that
    number answers a question nobody asked and invites the reader to
    treat it as the company's funding position. Funding is a private
    company's measure; for a public one the market cap is the answer.

    ``when/otherwise`` rather than ``lit(None)`` so each column keeps its
    dtype — ``funding_rounds`` is a list of structs and would not
    survive being overwritten with an untyped null.
    """
    public = pl.col("is_public").fill_null(False)
    columns = [c for c in FUNDING_COLUMNS if c in out.columns]
    withdrawn = out.filter(public & pl.col("funding_round_count").is_not_null()).height
    if withdrawn:
        logger.info("withdrew funding history from %d public tenants", withdrawn)
    return out.with_columns(
        pl.when(public).then(None).otherwise(pl.col(column)).alias(column)
        for column in columns
    )


def _read(path: object, label: str) -> pl.DataFrame:
    from pathlib import Path

    path = Path(str(path))
    if not path.exists():
        logger.warning("%s missing (%s) — continuing without it", label, path)
        return pl.DataFrame()
    df = pl.read_parquet(path)
    logger.info("%s: %d rows", label, df.height)
    return df


def build() -> pl.DataFrame:
    resolved = _read(config.RESOLVED_PARQUET, "resolved")
    if resolved.is_empty():
        raise RuntimeError("resolved.parquet is required; run the resolve stage first")

    marketcap = _read(config.MARKETCAP_PARQUET, "marketcap")
    funding = _read(config.FUNDING_PARQUET, "funding")
    teamsize = _read(config.TEAMSIZE_PARQUET, "teamsize")
    profile = _read(config.PROFILE_PARQUET, "profile")

    # `match_method` / `match_score` describe how the domain and firmographics
    # were reached; the CIK carries its own confidence separately.
    out = resolved.with_columns(
        pl.coalesce(pl.col("pdl_method"), pl.col("edgar_method")).alias("match_method"),
        pl.coalesce(pl.col("pdl_score"), pl.col("edgar_score")).alias("match_score"),
    )

    if not marketcap.is_empty():
        # The market-cap stage may attribute a parent's CIK/ticker to a
        # subsidiary tenant, so its identity columns are dropped in favour
        # of the resolved ones except where the tenant had none.
        mc = marketcap.drop("exchange").rename(
            {"cik": "mc_cik", "ticker": "mc_ticker"}
        )
        out = out.join(mc, on=["ats", "slug"], how="left").with_columns(
            pl.coalesce(pl.col("cik"), pl.col("mc_cik")).alias("cik"),
            pl.coalesce(pl.col("ticker"), pl.col("mc_ticker")).alias("ticker"),
        )

    if not funding.is_empty():
        out = out.join(funding, on=["ats", "slug"], how="left")
        out = _withdraw_funding_from_public(out)

    if not teamsize.is_empty():
        out = out.join(
            teamsize.drop("cik", "domain"), on=["ats", "slug"], how="left"
        )

    if not profile.is_empty():
        out = out.join(profile, on=["ats", "slug"], how="left")

    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None).alias(column))

    out = out.with_columns(_quality_flags().alias("quality_flags"))
    return out.select(OUTPUT_COLUMNS).sort("postings", descending=True)


def run() -> pl.DataFrame:
    config.ensure_dirs()
    assert_public_schema_untouched()
    out = build()
    out.write_parquet(config.ENRICHMENT_PARQUET)
    # The rounds list cannot round-trip through CSV; drop it for the flat
    # export and leave parquet as the complete artefact.
    out.drop("funding_rounds").write_csv(config.ENRICHMENT_CSV)

    n = out.height

    def pct(expr: pl.Expr) -> str:
        count = out.filter(expr).height
        return f"{count:>6,}  ({count / n:5.1%})"

    print("\n=== company_enrichment.parquet ===")
    print(f"Rows (one per ats+slug)       : {n:>6,}")
    print(f"Columns                       : {len(out.columns):>6,}")
    print("\nCoverage:")
    print(f"  domain                      : {pct(pl.col('domain').is_not_null() & (pl.col('domain') != ''))}")
    print(f"  linkedin_url                : {pct(pl.col('linkedin_url').is_not_null())}")
    print(f"  cik                         : {pct(pl.col('cik').is_not_null())}")
    print(f"  market_cap_usd              : {pct(pl.col('market_cap_usd').is_not_null())}")
    print(f"  employee_count (exact)      : {pct(pl.col('employee_count').is_not_null())}")
    print(f"  employee_count_floor (filed): {pct(pl.col('employee_count_floor').is_not_null())}")
    print(f"  employee_count_band         : {pct(pl.col('employee_count_band').is_not_null())}")
    print(f"  company_description         : {pct(pl.col('company_description').is_not_null())}")
    print(f"  careers_url                 : {pct(pl.col('careers_url').is_not_null())}")
    print(f"  company_careers_url         : {pct(pl.col('company_careers_url').is_not_null())}")
    print(f"  headquarters                : {pct(pl.col('headquarters').is_not_null())}")

    described = out.filter(pl.col("company_description").is_not_null())
    if not described.is_empty():
        print("\nDescription by source:")
        for source, count in (
            described.group_by("company_description_source")
            .len()
            .sort("len", descending=True)
            .iter_rows()
        ):
            print(f"  {source:<28}: {count:>6,}  ({count / n:5.1%})")

    # Funding is reported over private tenants only, since it is now
    # withdrawn from public ones by design rather than missing.
    private = out.filter(~pl.col("is_public").fill_null(False))
    funded = private.filter(pl.col("funding_round_count").is_not_null()).height
    print(
        f"  funding history (private)   : {funded:>6,}  "
        f"({funded / private.height:5.1%} of {private.height:,} private tenants)"
    )
    any_enrichment = (
        pl.col("market_cap_usd").is_not_null()
        | pl.col("funding_round_count").is_not_null()
        | pl.col("employee_count").is_not_null()
        | pl.col("employee_count_band").is_not_null()
    )
    print(f"  any of the three requested  : {pct(any_enrichment)}")

    flags = (
        out.select(pl.col("quality_flags").str.split(","))
        .explode("quality_flags")
        .filter(pl.col("quality_flags") != "")
        .group_by("quality_flags")
        .len()
        .sort("len", descending=True)
    )
    if not flags.is_empty():
        print("\nQuality flags (rows needing a human before you trust them):")
        for row in flags.iter_rows():
            print(f"  {row[0]:<28}: {row[1]:>6,}  ({row[1] / n:5.1%})")

    print("\nAttribution required by source licence:")
    print(f"  {pdl.ATTRIBUTION}")
    print("  SEC EDGAR / Form D / Reg CF / Reg A: public domain (17 CFR 200.80).")
    print("  OSHA Injury Tracking Application 300A: public domain (29 CFR 1904.41).")
    print("  DOL EFAST2 Form 5500 annual datasets: public domain.")
    print("  Wikidata: CC0.")
    print(
        "  company_description quotes the company's own website, postings, or\n"
        "  10-K verbatim; company_description_url records the page it came from."
    )
    print("\nSample (highest-traffic tenants):")
    print(
        out.head(10)
        .select(
            "ats",
            "slug",
            "resolved_name",
            "ticker",
            (pl.col("market_cap_usd") / 1e9).round(1).alias("mcap_$B"),
            (pl.col("funding_equity_total_usd") / 1e6).round(0).alias("equity_$M"),
            "employee_count",
            "employee_count_band",
        )
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.ENRICHMENT_PARQUET}")
    print(f"wrote {config.ENRICHMENT_CSV}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

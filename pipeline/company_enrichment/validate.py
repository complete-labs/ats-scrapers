"""Measure how much of the enrichment table is actually right.

Coverage is free to compute and says nothing about correctness: a
confident wrong domain counts the same as a right one. So this module
does two separate things.

**Coverage and internal consistency** are computed over the whole table —
per-field fill rates, and cross-source agreement where two independent
sources speak to the same fact (a 10-K headcount against a PDL band, a
Nasdaq market cap against SEC-filed share counts times a Stooq close).
Agreement is not proof, but disagreement is proof of a problem.

**Precision** cannot be computed automatically at all. It needs a human
deciding whether ``ngc/northrop_grumman_external_site`` really is
Northrop Grumman. :func:`sample` draws a stratified, seed-fixed sample
for that review, and the verdicts are stored in ``VALIDATION_LABELS`` so
the number is reproducible and re-auditable rather than a claim in a
commit message.

Strata matter because error rates are wildly uneven. A NYSE-listed
tenant matched on an exact legal name is nearly always right; a
four-employee agency whose tenant name is a slug is where the false
positives live. A uniform sample would be dominated by the long tail and
understate quality for the rows most consumers care about.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from pipeline.company_enrichment import config

logger = logging.getLogger(__name__)

VALIDATION_LABELS = Path(__file__).with_name("validation_labels.csv")

# Reviewer verdicts. Deliberately coarse — a reviewer can tell "this is
# the wrong company" from "this is the right company" reliably, and
# anything finer invites inconsistent labelling. ``no_claim`` is not a
# quality judgement: the row was left unresolved, so there is nothing to
# be right or wrong about, and it is excluded from precision.
VERDICTS = ("correct", "wrong", "ambiguous", "no_claim")

SAMPLE_SEED = 20260729

# Strata are evaluated in order and are mutually exclusive.
STRATA: tuple[tuple[str, pl.Expr, int], ...] = (
    (
        "public_listed",
        pl.col("ticker").is_not_null(),
        12,
    ),
    (
        "funded_private",
        pl.col("ticker").is_null() & (pl.col("funding_equity_round_count") >= 2),
        12,
    ),
    (
        "cik_only",
        pl.col("ticker").is_null()
        & pl.col("cik").is_not_null()
        & pl.col("funding_equity_round_count").is_null(),
        8,
    ),
    (
        "smb_band_only",
        pl.col("cik").is_null() & pl.col("employee_count_band").is_not_null(),
        12,
    ),
    (
        "unresolved",
        pl.col("cik").is_null() & pl.col("employee_count_band").is_null(),
        6,
    ),
)


REVIEW_COLUMNS = (
    "stratum",
    "ats",
    "slug",
    "name",
    "resolved_name",
    "domain",
    "linkedin_url",
    "cik",
    "ticker",
    "match_method",
    "match_score",
    "cik_confidence",
    "market_cap_usd",
    "funding_equity_total_usd",
    "funding_equity_round_count",
    "employee_count",
    "employee_count_source",
    "employee_count_band",
    # A reviewer deciding whether a row describes the right company gets
    # far more from one sentence of prose than from any identifier, so
    # the description sits in the review file next to the name it is
    # supposed to be describing.
    "company_description",
    "company_description_source",
    "company_description_name_corroborated",
    "headquarters",
    "careers_url",
    "company_careers_url",
    "postings",
    "url",
    "quality_flags",
)


def with_stratum(df: pl.DataFrame) -> pl.DataFrame:
    """Label every row with the first stratum whose predicate it meets."""
    stratum = pl.lit(None, dtype=pl.String)
    claimed = pl.lit(False)
    for name, predicate, _size in STRATA:
        # `fill_null` is essential: a comparison against a null column
        # yields null, not false, and a null in `claimed` would make
        # `~claimed` null and silently drop every later stratum.
        safe = predicate.fill_null(False)
        stratum = pl.when(safe & ~claimed).then(pl.lit(name)).otherwise(stratum)
        claimed = claimed | safe
    return df.with_columns(stratum.alias("stratum"))


def _draw_order() -> pl.Expr:
    """Deterministic per-row draw key derived only from the tenant identity.

    Positional sampling was a mistake here. ``DataFrame.sample`` picks by
    row position, so re-running the pipeline against a refreshed
    ``all.parquet`` that added two tenants reshuffled the draw and left
    only 5 of 50 hand labels pointing at sampled rows. Hashing
    ``(ats, slug)`` instead makes membership a property of the tenant, so
    the sample barely moves when unrelated rows come and go.
    """
    return (pl.col("ats") + pl.lit("\x00") + pl.col("slug") + pl.lit(f"\x00{SAMPLE_SEED}")).hash()


def sample(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Stratified review sample, stable under changes to unrelated rows."""
    if df is None:
        df = pl.read_parquet(config.ENRICHMENT_PARQUET)

    tagged = with_stratum(df).with_columns(_draw_order().alias("_draw"))
    taken: list[pl.DataFrame] = []
    for name, _predicate, size in STRATA:
        pool = tagged.filter(pl.col("stratum") == name)
        if pool.is_empty():
            logger.warning("stratum %s is empty", name)
            continue
        # A plain uniform draw within the stratum: weighting by posting
        # volume would over-pick megacaps and hide long-tail errors.
        taken.append(pool.sort("_draw").head(size))

    return pl.concat(taken, how="vertical").select(REVIEW_COLUMNS)


def write_sample_template() -> Path:
    """Write the sample out for a reviewer to fill in."""
    out = config.OUTPUT_DIR / "validation_sample.csv"
    sample().with_columns(
        pl.lit("").alias("verdict"), pl.lit("").alias("note")
    ).write_csv(out)
    logger.info("wrote review template %s", out)
    return out


def load_labels() -> pl.DataFrame:
    if not VALIDATION_LABELS.exists():
        return pl.DataFrame(
            schema={
                "ats": pl.String,
                "slug": pl.String,
                "verdict": pl.String,
                "note": pl.String,
            }
        )
    labels = pl.read_csv(VALIDATION_LABELS)
    bad = set(labels["verdict"].unique()) - set(VERDICTS)
    if bad:
        raise ValueError(f"unknown verdicts in {VALIDATION_LABELS.name}: {sorted(bad)}")
    return labels


def reviewed_rows(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every hand-labelled tenant, joined to its current enrichment row.

    Labels are keyed on ``(ats, slug)`` and joined to the live table
    rather than to a freshly drawn sample. The sample is only how rows
    were *chosen* for review; once a human has ruled on a tenant that
    verdict stands on its own, so precision must not silently shrink just
    because a redrawn sample no longer contains it. The stratum is
    recomputed from current data, so a tenant that has since gained a
    ticker is scored in the stratum it now belongs to.
    """
    if df is None:
        df = pl.read_parquet(config.ENRICHMENT_PARQUET)
    labels = load_labels()
    if labels.is_empty():
        return pl.DataFrame()
    return with_stratum(df).join(labels, on=["ats", "slug"], how="inner")


def precision(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-stratum resolution precision over the hand-labelled set.

    ``ambiguous`` rows are counted against precision. A tenant a careful
    reviewer cannot pin down is not a usable match, so scoring it as
    neither right nor wrong would flatter the number.
    """
    reviewed = reviewed_rows(df)
    if reviewed.is_empty():
        return pl.DataFrame()
    reviewed = reviewed.filter(pl.col("verdict") != "no_claim")
    if reviewed.is_empty():
        return pl.DataFrame()
    return (
        reviewed.group_by("stratum")
        .agg(
            pl.len().alias("reviewed"),
            (pl.col("verdict") == "correct").sum().alias("correct"),
            (pl.col("verdict") == "wrong").sum().alias("wrong"),
            (pl.col("verdict") == "ambiguous").sum().alias("ambiguous"),
        )
        .with_columns(
            (pl.col("correct") / pl.col("reviewed")).alias("precision")
        )
        .sort("stratum")
    )


def flag_recall(df: pl.DataFrame | None = None) -> tuple[int, int]:
    """How many hand-confirmed bad rows carry a ``quality_flags`` entry.

    This is the number that decides whether the flags are worth acting
    on. If most errors arrive unflagged, a consumer has to review
    everything and the flags are decoration.
    """
    reviewed = reviewed_rows(df)
    if reviewed.is_empty():
        return (0, 0)
    bad = reviewed.filter(pl.col("verdict").is_in(["wrong", "ambiguous"]))
    if bad.is_empty():
        return (0, 0)
    flagged = bad.filter(pl.col("quality_flags").fill_null("") != "").height
    return (flagged, bad.height)


def consistency(df: pl.DataFrame) -> dict[str, tuple[int, int]]:
    """Cross-source agreement counts as ``(agreeing, comparable)``."""
    band = df.filter(pl.col("employee_count_agrees_with_band").is_not_null())
    mcap = df.filter(pl.col("market_cap_check").is_in(["verified", "disagrees"]))
    described = df.filter(pl.col("company_description").is_not_null())
    careers = df.filter(pl.col("careers_url").is_not_null())
    return {
        "headcount vs PDL band": (
            band.filter(pl.col("employee_count_agrees_with_band")).height,
            band.height,
        ),
        "market cap vs SEC shares x price": (
            mcap.filter(pl.col("market_cap_check") == "verified").height,
            mcap.height,
        ),
        # Whether the description names the tenant it hangs off. The
        # only automatic check available on prose, and the one that
        # catches a description lifted from a mis-resolved domain.
        "description names its tenant": (
            described.filter(
                pl.col("company_description_name_corroborated").fill_null(False)
            ).height,
            described.height,
        ),
        "careers URL resolves to tenant": (
            careers.filter(pl.col("careers_url_verified").fill_null(False)).height,
            careers.height,
        ),
    }


def run() -> pl.DataFrame:
    config.ensure_dirs()
    df = pl.read_parquet(config.ENRICHMENT_PARQUET)
    n = df.height

    print("\n=== Validation ===")
    print(f"Tenants in cohort            : {n:>6,}")

    print("\nCoverage by field:")
    fields = {
        "domain": pl.col("domain").is_not_null() & (pl.col("domain") != ""),
        "linkedin_url": pl.col("linkedin_url").is_not_null(),
        "cik": pl.col("cik").is_not_null(),
        "cik (high confidence)": pl.col("cik_confidence") == "high",
        "ticker": pl.col("ticker").is_not_null(),
        "market_cap_usd": pl.col("market_cap_usd").is_not_null(),
        "funding (any round)": pl.col("funding_round_count").is_not_null(),
        "funding (equity round)": pl.col("funding_equity_round_count").is_not_null(),
        "employee_count (exact)": pl.col("employee_count").is_not_null(),
        "employee_count_band": pl.col("employee_count_band").is_not_null(),
        "company_description": pl.col("company_description").is_not_null(),
        "company_description (named)": pl.col(
            "company_description_name_corroborated"
        ).fill_null(False),
        "careers_url": pl.col("careers_url").is_not_null(),
        "careers_url (confirmed)": pl.col("careers_url_verified").fill_null(False),
        "company_careers_url": pl.col("company_careers_url").is_not_null(),
        "headquarters": pl.col("headquarters").is_not_null(),
    }
    for label, predicate in fields.items():
        count = df.filter(predicate).height
        print(f"  {label:<27}: {count:>6,}  ({count / n:5.1%})")

    print("\nCross-source consistency:")
    for label, (agree, total) in consistency(df).items():
        rate = f"{agree / total:5.1%}" if total else "   n/a"
        print(f"  {label:<34}: {agree:>5,} / {total:<5,}  ({rate})")

    labels = load_labels()
    matched = reviewed_rows(df)
    if not labels.is_empty():
        missing = labels.height - matched.height
        print(
            f"\nHand labels: {labels.height} recorded, {matched.height} still "
            f"present in the cohort"
            + (f", {missing} no longer present" if missing else "")
        )

    prec = precision(df)
    if prec.is_empty():
        print(
            "\nNo hand labels yet. Run `write_sample_template()` and fill in "
            f"{VALIDATION_LABELS.name}."
        )
    else:
        print("\nHand-checked resolution precision:")
        print(prec.to_pandas().to_string(index=False))
        total_reviewed = int(prec["reviewed"].sum())
        total_correct = int(prec["correct"].sum())
        print(
            f"\n  overall: {total_correct}/{total_reviewed} "
            f"({total_correct / total_reviewed:.1%})"
        )
        flagged, bad = flag_recall(df)
        if bad:
            print(
                f"\n  of the {bad} bad rows found by hand, {flagged} "
                f"({flagged / bad:.0%}) already carried a quality flag"
            )
    return prec


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

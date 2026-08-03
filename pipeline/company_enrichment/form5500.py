"""DOL Form 5500 — a filed headcount floor for *private* companies.

Why this source exists here
--------------------------
Every exact headcount in this pipeline comes from a securities filing or
from Wikidata, and both are effectively public-company sources. Measured
over the cohort, private tenants had an exact count 2.3% of the time and
**not one of them came from a 10-K** — the 10-K parse only runs for
registrants carrying a ticker. A third of private tenants had no headcount
signal of any kind. Form 5500 is the only free, bulk, public-domain source
that speaks to private-company employment at scale.

Any employer sponsoring an ERISA benefit plan — a 401(k), a pension, or a
health and welfare plan — files a Form 5500 annually, and DOL republishes
every filing as an annual dataset. The full form covers large plans; the
short form (``F_5500_SF``) covers small ones, and that is where the SMB
long tail lives: plan year 2024 held 222k full-form and 795k short-form
filings.

What the number means
---------------------
``TOT_ACTIVE_PARTCP_CNT`` counts *active* participants: people currently
employed and accruing benefits under the plan. Retirees and separated
vested participants are counted in their own columns and deliberately not
used here, because they are not staff.

Active participants are a **floor on headcount, never a headcount**. Not
every employee is eligible for a plan, and not every eligible employee
enrols. So this is written as ``employee_count_floor`` alongside
:mod:`.osha` and is never promoted into ``employee_count``.

What has to be excluded
-----------------------
Two filer categories would wreck the number and neither can be guessed
from the name, so the form's own ``TYPE_PLAN_ENTITY_CD`` decides. The
codes differ between the two forms, which is easy to get backwards, so
they were confirmed against the data rather than the instructions:

* Full form, code ``1`` is a **multiemployer** plan — 79.6% of those
  sponsor names are unions, joint boards, and trust funds ("RACINE
  PAINTERS & ALLIED TRADES UNION LOCAL 108"). One such plan covers the
  members of many unrelated employers, so its participant count says
  nothing about any one of them. Code ``2`` is the single-employer plan
  and is the only one kept.
* Code ``4`` is a **Direct Filing Entity** — a master trust or pooled
  account, not an employer. These report no active participants anyway.

On the short form the numbering is the other way round: code ``1`` is the
single-employer plan (0.5% union-looking names) and is the one kept.

The entity code is necessary but **not sufficient**. Several union
benefit trusts file as single-employer plans anyway — the American
Federation of Teachers Benefit Trust reports 1.28M active participants —
so sponsors whose own ``BUSINESS_CODE`` puts them in a benefit-fund or
labor-organisation industry are dropped as well. One residual is
knowingly left in: a professional employer organisation such as ADP
TotalSource sponsors a plan covering its clients' staff and looks like an
ordinary services company in every field on the form. That inflates the
floor for the PEO itself and cannot reach its clients, since the EIN is
the PEO's own, so it is recorded here rather than guessed at.

Two counts, not the larger of them
----------------------------------
Each filing states active participants at the start of the plan year and
at the end. The end-of-year figure is the answer and the earlier one only
fills a gap, because taking the larger maximises exposure to data-entry
errors: a dog day-care filed 327,659 participants at the start of the
year and 3 at the end. The two figures also cross-check each other, the
way :mod:`.osha` uses hours worked, so a filing whose counts move by an
impossible factor is dropped rather than resolved in either direction.

Joining without guessing
------------------------
The sponsor's EIN is the key, and :mod:`.registrant` recovers the same
number for every matched CIK straight out of EDGAR. That makes this the
one **exact** cross-source join in the package — every other one is a
name match with a similarity score and a collision problem.

For tenants with no CIK there is no EIN to join on, so a name path exists
too, guarded the way :mod:`.osha`'s floor is: a name key shared by more
than one sponsor EIN is ambiguous, and a floor claimed on it is only used
when an independent signal agrees. ``dol_ein_count`` carries that so the
caller can decide.

Licence: US Government work, public domain.
https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/public-disclosure/foia/form-5500-datasets
"""

from __future__ import annotations

import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import name_key

logger = logging.getLogger(__name__)

ATTRIBUTION = (
    "Plan sponsor employment from the US Department of Labor EFAST2 Form 5500 "
    "annual datasets, a US Government work in the public domain."
)

# Columns pulled from each form. Read as strings and cast afterwards, so a
# stray value in one of a million rows cannot abort the whole ingest.
_MAIN_COLUMNS = {
    "ein": "SPONS_DFE_EIN",
    "sponsor_name": "SPONSOR_DFE_NAME",
    "entity_code": "TYPE_PLAN_ENTITY_CD",
    "tax_period": "FORM_TAX_PRD",
    "active_current": "TOT_ACTIVE_PARTCP_CNT",
    "active_boy": "TOT_ACT_PARTCP_BOY_CNT",
    "state": "SPONS_DFE_MAIL_US_STATE",
    "state_alt": "SPONS_DFE_LOC_US_STATE",
    "business_code": "BUSINESS_CODE",
}
_SF_COLUMNS = {
    "ein": "SF_SPONS_EIN",
    "sponsor_name": "SF_SPONSOR_NAME",
    "entity_code": "SF_PLAN_ENTITY_CD",
    "tax_period": "SF_TAX_PRD",
    "active_current": "SF_TOT_ACT_PARTCP_EOY_CNT",
    "active_boy": "SF_TOT_ACT_PARTCP_BOY_CNT",
    "state": "SF_SPONS_US_STATE",
    "state_alt": "SF_SPONS_US_STATE",
    "business_code": "SF_BUSINESS_CODE",
}

# The single-employer plan code, per form. Confirmed against the data:
# see the module docstring.
_SINGLE_EMPLOYER_CODE = {"main": "2", "sf": "1"}

# NAICS codes DOL's own ``BUSINESS_CODE`` field gives these sponsors,
# where the filer is a benefit-plan vehicle rather than an employer and
# its participants are somebody else's staff. The entity code does not
# catch them: several union trusts file as single-employer plans.
# Deliberately narrow — only categories that cannot be an operating
# employer are listed, since anything broader would drop real companies.
_NON_EMPLOYER_NAICS = frozenset(
    {
        "813930",  # labor unions and similar labor organisations
        "813940",  # political organisations
        "525100",  # insurance and employee benefit funds
        "525110",  # pension funds
        "525120",  # health and welfare funds
        "525190",  # other insurance and employee benefit funds
        "525900",  # other investment pools and funds
        "524290",  # other insurance-related activities (plan administrators)
    }
)

# Sanity envelope. The largest US employer is around 2.3M; below one
# active participant there is nothing to report.
_MIN_PARTICIPANTS = 1
_MAX_PARTICIPANTS = 3_000_000

# The beginning-of-year and end-of-year counts describe the same
# workforce twelve months apart, so they are each other's cross-check —
# the same trick :mod:`.osha` plays with hours worked. A dog day-care
# filed 327,659 active participants at the start of the year and 3 at the
# end; no real employer moves by that factor, so a row whose two counts
# disagree this badly is a data-entry error and is dropped rather than
# resolved in either direction.
_MAX_YEAR_ON_YEAR_RATIO = 50.0
# Below this the ratio test is not applied: a genuine startup really can
# go from 2 staff to 60 in a year.
_RATIO_TEST_FLOOR = 1_000

# A name key shorter than this matches too many unrelated employers to be
# worth offering on the name path. Same bar as `osha`.
_MIN_KEY_LEN = 6

SCHEMA: dict[str, pl.DataType] = {
    "ein": pl.String,
    "name_key_core": pl.String,
    "dol_sponsor_name": pl.String,
    "dol_participants": pl.Int64,
    "dol_states": pl.List(pl.String),
    "dol_plans": pl.Int64,
    "dol_business_code": pl.String,
    "dol_as_of": pl.String,
}


def _years() -> list[int]:
    """Plan years to pull, newest first.

    The current calendar year is skipped: its plan year has not ended for
    most filers, so the dataset would be near-empty.
    """
    latest = datetime.now(tz=UTC).year - 1
    return [latest - offset for offset in range(config.DOL_5500_YEARS)]


def _download(url: str, dest: Path) -> bool:
    import httpx

    try:
        with httpx.stream(
            "GET",
            url,
            headers={"User-Agent": config.SEC_USER_AGENT},
            timeout=600.0,
            follow_redirects=True,
        ) as response:
            if response.status_code == 404:
                logger.info("not published yet: %s", url)
                return False
            response.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        logger.warning("skipping %s: %s", url, exc)
        return False
    return True


def _read_form(archive_path: Path, columns: dict[str, str], kind: str) -> pl.DataFrame:
    """Extract one dataset's CSV and keep only the rows and columns needed.

    The short-form CSV is 227 MB for a single year, so it is extracted to
    a scratch file and streamed with projection pushdown rather than
    parsed whole in memory, then deleted. Peak extra disk stays at one
    CSV instead of every year at once.
    """
    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            logger.warning("no CSV inside %s", archive_path)
            return pl.DataFrame()
        member = names[0]
        scratch = archive_path.with_suffix(".scratch.csv")
        with archive.open(member) as src, scratch.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)

    try:
        scan = pl.scan_csv(
            scratch,
            infer_schema_length=0,
            truncate_ragged_lines=True,
            encoding="utf8-lossy",
        )
        header = scan.collect_schema().names()
        wanted = {alias: col for alias, col in columns.items() if col in header}
        missing = set(columns) - set(wanted)
        if "ein" in missing or "active" in missing:
            logger.warning("%s lacks the EIN/participant columns; skipping", member)
            return pl.DataFrame()
        if missing:
            logger.info("%s has no %s column(s)", member, sorted(missing))

        frame = scan.select(
            [pl.col(col).alias(alias) for alias, col in wanted.items()]
        ).collect()
    finally:
        scratch.unlink(missing_ok=True)

    for alias in columns:
        if alias not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(alias))

    keep_code = _SINGLE_EMPLOYER_CODE[kind]
    before = frame.height
    frame = frame.with_columns(
        pl.col("entity_code").str.strip_chars().str.strip_chars('"')
    ).filter(pl.col("entity_code") == keep_code)
    logger.info(
        "%s: %d of %d filings are single-employer plans", member, frame.height, before
    )
    return frame


def _normalise(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast, clean, and drop rows that cannot contribute a floor."""
    if frame.is_empty():
        return frame

    digits = pl.col("ein").str.replace_all(r"[^0-9]", "")

    def positive(column: str) -> pl.Expr:
        """The column as an integer, treating zero as unanswered.

        A reported zero is indistinguishable from a skipped field and
        cannot bound a headcount either way.
        """
        value = pl.col(column).cast(pl.Int64, strict=False)
        return pl.when(value > 0).then(value).otherwise(None)

    frame = frame.with_columns(
        pl.when(digits.str.len_chars() == 9).then(digits).otherwise(None).alias("ein"),
        positive("active_current").alias("_current"),
        positive("active_boy").alias("_boy"),
        pl.coalesce(
            pl.col("state").str.strip_chars(), pl.col("state_alt").str.strip_chars()
        )
        .str.to_uppercase()
        .alias("sponsor_state"),
        # `FORM_TAX_PRD` is the plan year end as `YYYY-MM-DD`, which is
        # exactly the as-of date the count refers to.
        pl.col("tax_period").str.strip_chars().str.slice(0, 10).alias("as_of"),
        pl.col("business_code").str.strip_chars().alias("business_code"),
    )

    high = pl.max_horizontal("_current", "_boy")
    low = pl.min_horizontal("_current", "_boy")
    contradictory = (
        pl.col("_current").is_not_null()
        & pl.col("_boy").is_not_null()
        & (high > _RATIO_TEST_FLOOR)
        & (high > low * _MAX_YEAR_ON_YEAR_RATIO)
    )

    before = frame.height
    frame = frame.filter(~contradictory)
    if before - frame.height:
        logger.info(
            "dropped %d filings whose year-on-year counts contradict each other",
            before - frame.height,
        )

    return (
        frame.with_columns(
            # The current count is the answer; the prior year only fills a
            # gap. Never the larger of the two — that maximises exposure
            # to exactly the data-entry errors the ratio test cannot see.
            pl.coalesce(pl.col("_current"), pl.col("_boy")).alias("participants")
        )
        .filter(
            pl.col("ein").is_not_null()
            & pl.col("participants").is_not_null()
            & (pl.col("participants") >= _MIN_PARTICIPANTS)
            & (pl.col("participants") <= _MAX_PARTICIPANTS)
            & ~pl.col("business_code").is_in(list(_NON_EMPLOYER_NAICS)).fill_null(False)
        )
        .select(
            "ein", "sponsor_name", "participants", "sponsor_state", "business_code", "as_of"
        )
    )


def _rollup(filings: pl.DataFrame) -> pl.DataFrame:
    """One row per sponsor EIN, from its most recent plan year.

    A sponsor commonly files several plans — a 401(k), a health plan, a
    dental plan — so the counts are combined with ``max`` and never
    summed: each plan covers some subset of the same staff, and adding
    them would multiply one workforce by its number of benefit plans.

    Only the newest plan year for that sponsor contributes, so a company
    that shrank is not reported at its historical peak.
    """
    latest = pl.col("as_of").max()
    rolled = filings.group_by("ein").agg(
        pl.col("participants").filter(pl.col("as_of") == latest).max().alias("dol_participants"),
        latest.alias("dol_as_of"),
        pl.len().alias("dol_plans"),
        pl.col("sponsor_name").filter(pl.col("as_of") == latest).first().alias("dol_sponsor_name"),
        pl.col("sponsor_state").drop_nulls().unique().alias("dol_states"),
        pl.col("business_code").drop_nulls().first().alias("dol_business_code"),
    )
    return (
        rolled.filter(pl.col("dol_participants").is_not_null())
        .with_columns(
            pl.col("dol_sponsor_name")
            .map_elements(name_key, return_dtype=pl.String)
            .alias("name_key_core")
        )
        .select(list(SCHEMA))
        .sort("dol_participants", descending=True)
    )


def ingest(*, force: bool = False) -> pl.DataFrame:
    """Materialise the per-EIN employment floor."""
    config.ensure_dirs()
    if config.FORM5500_PARQUET.exists() and not force:
        logger.info("reusing %s", config.FORM5500_PARQUET)
        return pl.read_parquet(config.FORM5500_PARQUET)

    frames: list[pl.DataFrame] = []
    for year in _years():
        for kind, template, columns in (
            ("main", config.DOL_5500_URL_TEMPLATE, _MAIN_COLUMNS),
            ("sf", config.DOL_5500_SF_URL_TEMPLATE, _SF_COLUMNS),
        ):
            url = template.format(year=year)
            cached = config.CACHE_DIR / f"dol_5500_{kind}_{year}.zip"
            if not cached.exists() or force:
                logger.info("downloading %s", url)
                if not _download(url, cached):
                    cached.unlink(missing_ok=True)
                    continue
            raw = _read_form(cached, columns, kind)
            cleaned = _normalise(raw)
            if not cleaned.is_empty():
                logger.info("%s %d: %d usable sponsor filings", kind, year, cleaned.height)
                frames.append(cleaned)

    if not frames:
        raise RuntimeError("no Form 5500 datasets could be loaded")

    rolled = _rollup(pl.concat(frames, how="vertical"))
    rolled.write_parquet(config.FORM5500_PARQUET)
    logger.info("wrote %s (%d sponsors)", config.FORM5500_PARQUET, rolled.height)
    return rolled


def load() -> pl.DataFrame:
    """Read the built floor, or an empty frame if the stage has not run."""
    if not config.FORM5500_PARQUET.exists():
        logger.warning("%s missing — continuing without it", config.FORM5500_PARQUET)
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(config.FORM5500_PARQUET)


def by_name(floor: pl.DataFrame | None = None) -> pl.DataFrame:
    """The same floor keyed on sponsor name, for tenants with no CIK.

    ``dol_ein_count`` is the point of this view: a name key held by two
    different sponsor EINs cannot identify an employer on its own, and the
    caller is expected to require corroboration before believing it.
    """
    floor = load() if floor is None else floor
    if floor.is_empty():
        return pl.DataFrame(
            schema={
                "name_key_core": pl.String,
                "dol_participants": pl.Int64,
                "dol_as_of": pl.String,
                "dol_states": pl.List(pl.String),
                "dol_plans": pl.Int64,
                "dol_ein_count": pl.Int64,
                "dol_sponsor_name": pl.String,
            }
        )
    return (
        floor.filter(pl.col("name_key_core").str.len_chars() >= _MIN_KEY_LEN)
        .group_by("name_key_core")
        .agg(
            # `max`, not `sum`: distinct EINs under one name key are
            # distinct legal employers, and where the key is ambiguous the
            # caller drops the row anyway.
            pl.col("dol_participants").max().alias("dol_participants"),
            pl.col("dol_as_of").max().alias("dol_as_of"),
            pl.col("dol_states").explode().drop_nulls().unique().alias("dol_states"),
            pl.col("dol_plans").max().alias("dol_plans"),
            pl.col("ein").n_unique().alias("dol_ein_count"),
            pl.col("dol_sponsor_name").first().alias("dol_sponsor_name"),
        )
    )


def run(*, force: bool = False) -> pl.DataFrame:
    df = ingest(force=force)
    print("\n=== DOL Form 5500 employment floor ===")
    print(ATTRIBUTION)
    print(f"\nPlan sponsors (unique EIN)  : {df.height:>8,}")
    print(f"Plan years pulled           : {', '.join(str(y) for y in _years())}")
    named = by_name(df)
    print(f"Distinct sponsor name keys  : {named.height:>8,}")
    print(
        f"  unambiguous (one EIN)     : "
        f"{named.filter(pl.col('dol_ein_count') == 1).height:>8,}"
    )
    print("\nParticipant distribution:")
    for label, low, high in (
        ("1-10", 1, 10),
        ("11-50", 11, 50),
        ("51-200", 51, 200),
        ("201-1000", 201, 1000),
        ("1001-10000", 1001, 10000),
        ("10001+", 10001, _MAX_PARTICIPANTS),
    ):
        n = df.filter(pl.col("dol_participants").is_between(low, high)).height
        print(f"  {label:<12}: {n:>8,}")
    print("\nLargest sponsors:")
    print(
        df.head(10)
        .select(
            "dol_sponsor_name",
            pl.col("dol_participants").alias("active"),
            pl.col("dol_plans").alias("plans"),
            "dol_as_of",
        )
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.FORM5500_PARQUET}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

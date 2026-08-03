"""Tests for tenant → company resolution (`company_enrichment.resolve`).

This is the load-bearing stage: a wrong answer here does not produce one
wrong field, it produces a wholly plausible row about a different
company — domain, LinkedIn URL, headcount band, headquarters, industry
and founding year all internally consistent and all belonging to someone
else.

The fixtures are deliberately tiny and named after real failures. The
``harvey`` tenant is an Ashby board for a legal-AI company that resolved
to Harvey Holdings, a North Carolina logistics firm, because both reduce
to the blocking key ``harvey``. ``ngc`` is the Workday tenant whose only
legible identity lives in its board path.

Four tests are marked ``xfail(strict=True)``. They assert the behaviour
the module should have rather than the behaviour it has, so the defect
is executable and tracked, and so a fix announces itself by turning the
suite red on an unexpected pass instead of going unnoticed.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from pipeline.company_enrichment import config
from pipeline.company_enrichment import resolve as resolve_mod
from pipeline.company_enrichment.resolve import (
    _confidence_expr,
    _first_slug_segment,
    _mark_slug_only_matches,
    _pair_score,
    _pick_best,
    _slug_path_name,
    candidate_keys,
    declared_names,
    name_variants,
    resolve_edgar,
    resolve_pdl,
)

COHORT_COLUMNS = (
    "ats", "slug", "name", "display_name", "jobs_company", "url", "source_domain",
)


def cohort(*rows: dict[str, str]) -> pl.DataFrame:
    """A cohort frame with every column `resolve` reads."""
    filled = [{column: row.get(column, "") for column in COHORT_COLUMNS} for row in rows]
    return pl.DataFrame(filled, schema=dict.fromkeys(COHORT_COLUMNS, pl.String))


HARVEY = {
    "ats": "ashby",
    "slug": "harvey",
    "name": "Harvey",
    "display_name": "Harvey",
    "jobs_company": "harvey",
    "url": "https://jobs.ashbyhq.com/harvey",
}
ACME = {
    "ats": "ashby",
    "slug": "acme",
    "name": "Acme Robotics",
    "display_name": "Acme Robotics",
    "jobs_company": "Acme Robotics",
    "url": "https://jobs.ashbyhq.com/acme",
}
NGC = {
    "ats": "workday",
    "slug": "ngc/northrop_grumman_external_site",
    "name": "Ngc",
    "display_name": "Ngc",
    "jobs_company": "Ngc",
    "url": "https://ngc.wd1.myworkdayjobs.com/northrop_grumman_external_site",
}


def pdl_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "name": pl.String, "domain": pl.String, "linkedin_url": pl.String,
            "size": pl.String, "size_midpoint": pl.Int64, "founded": pl.Int64,
            "locality": pl.String, "region": pl.String, "industry": pl.String,
            "name_key_raw": pl.String, "name_key_core": pl.String,
        },
    )


def pdl_row(name: str, key_core: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name, "domain": f"{key_core}.com", "linkedin_url": f"li/{key_core}",
        "size": "51-200", "size_midpoint": 125, "founded": 2003,
        "locality": "somewhere", "region": "texas", "industry": "software",
        "name_key_raw": key_core, "name_key_core": key_core,
    }
    row.update(overrides)
    return row


def edgar_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "name": pl.String, "cik": pl.Int64, "ticker": pl.String,
            "exchange": pl.String, "is_public": pl.Boolean,
            "name_key_raw": pl.String, "name_key_core": pl.String,
        },
    )


# --- reading a name out of the record -----------------------------------


def test_board_path_recovers_the_only_legible_name() -> None:
    """"Ngc" is an acronym everywhere except inside the URL."""
    assert _first_slug_segment(NGC["slug"]) == "ngc"
    assert _slug_path_name(NGC["slug"]) == "northrop grumman"
    assert name_variants(NGC) == ["ngc", "northrop grumman"]


def test_declared_names_exclude_anything_guessed_from_the_slug() -> None:
    assert declared_names(NGC) == ["ngc"]


def test_a_short_slug_token_never_becomes_a_blocking_key() -> None:
    """"ngc" is three characters; blocking on it would match noise."""
    assert candidate_keys(NGC) == ["northropgrumman"]


def test_generic_words_are_never_blocking_keys() -> None:
    row = dict(ACME, name="Careers", display_name="Careers", jobs_company="jobs", slug="corporate")
    assert candidate_keys(row) == []


def test_both_key_forms_are_emitted() -> None:
    row = dict(ACME, name="Leidos Holdings Inc", display_name="", jobs_company="", slug="leidos")
    assert set(candidate_keys(row)) >= {"leidos", "leidosholdingsinc"}


# --- scoring ------------------------------------------------------------


def test_identical_names_score_full_marks() -> None:
    assert _pair_score("harvey", "harvey") == 100.0


def test_a_run_together_variant_reaches_its_spaced_form() -> None:
    """Without the desuffixed variant this pair scores below accept."""
    assert _pair_score("columbiasportswear", "columbia sportswear") > config.MATCH_ACCEPT_SCORE


def test_empty_side_scores_zero() -> None:
    assert _pair_score("", "harvey") == 0.0
    assert _pair_score("harvey", "") == 0.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "token_set_ratio is discounted by only 2%, which is not enough to "
        "pull a one-token subset below the accept threshold: 'acme' scores "
        "98 against 'acme bank of delaware'. The docstring on _pair_score "
        "claims this pairing is prevented; it is not."
    ),
)
def test_a_bare_token_should_not_match_a_longer_name_containing_it() -> None:
    assert _pair_score("acme", "acme bank of delaware") < config.MATCH_ACCEPT_SCORE


# --- picking one winner -------------------------------------------------


def test_a_complete_record_outranks_a_dormant_duplicate() -> None:
    """PDL carries shells under famous names, with no site and no size.

    Under Polars' default null ordering those sort first on a descending
    tiebreak, which is how Boeing once resolved to a 51-200 shell.
    """
    candidates = pl.DataFrame(
        {
            "ats": ["greenhouse", "greenhouse"],
            "slug": ["boeing", "boeing"],
            "match_score": [100.0, 100.0],
            "linkedin_url": [None, "li/boeing"],
            "size_midpoint": [None, 150_000],
        },
        schema_overrides={"linkedin_url": pl.String, "size_midpoint": pl.Int64},
    )
    best = _pick_best(
        candidates,
        prefer=[
            pl.col("linkedin_url").is_not_null().cast(pl.Int8),
            pl.col("size_midpoint").fill_null(0),
        ],
    )
    assert best.height == 1
    assert best["size_midpoint"][0] == 150_000


def test_pick_best_returns_one_row_per_tenant() -> None:
    candidates = pl.DataFrame(
        {
            "ats": ["a", "a", "b"],
            "slug": ["x", "x", "y"],
            "match_score": [90.0, 100.0, 95.0],
            "pdl_name": ["low", "high", "other"],
        }
    )
    best = _pick_best(candidates).sort("slug")
    assert best.height == 2
    assert best["pdl_name"].to_list() == ["high", "other"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_pick_best sorts only on the score and the caller's tiebreaks, so "
        "a genuine tie is broken by input row order. Roughly 200 tenants "
        "land on a different CIK and 270 on a different domain between "
        "runs. A deterministic final tiebreak (the candidate name, say) "
        "would pin it."
    ),
)
def test_a_tie_is_broken_the_same_way_every_run() -> None:
    rows = [
        {"ats": "g", "slug": "s", "match_score": 100.0, "pdl_name": name}
        for name in ("alpha", "beta", "gamma")
    ]
    winners = {
        _pick_best(pl.DataFrame(list(order)))["pdl_name"][0]
        for order in itertools.permutations(rows)
    }
    assert len(winners) == 1


# --- confidence grading -------------------------------------------------


def test_a_single_token_private_match_is_graded_low() -> None:
    """A perfect score on one word proves nothing. EDGAR has a "TEMPO,
    LLC" that matches any tenant called Tempo."""
    graded = pl.DataFrame(
        {
            "match_score": [100.0, 100.0, 93.0, 85.0],
            "matched_variant": ["harvey", "acme robotics", "acme robotics", "acme robotics"],
            "is_public": [False, False, False, True],
        }
    ).with_columns(
        _confidence_expr("match_score", "matched_variant", public_col="is_public").alias("c")
    )
    assert graded["c"].to_list() == ["low", "high", "medium", "low"]


def test_being_listed_substitutes_for_a_distinctive_name() -> None:
    graded = pl.DataFrame(
        {
            "match_score": [100.0],
            "matched_variant": ["tempo"],
            "is_public": [True],
        }
    ).with_columns(
        _confidence_expr("match_score", "matched_variant", public_col="is_public").alias("c")
    )
    assert graded["c"][0] == "high"


# --- the PDL track ------------------------------------------------------


def test_a_domain_match_outranks_a_name_match() -> None:
    """Domain equality is near-certain identity and needs no score."""
    tenants = cohort(dict(ACME, source_domain="acmerobotics.com"))
    pdl = pdl_frame(
        [
            pdl_row("acme robotics", "acmerobotics", domain="acmerobotics.com"),
            pdl_row("acme robotics of nebraska", "acmerobotics", domain="other.com"),
        ]
    )
    out = resolve_pdl(tenants, pdl)
    assert out.height == 1
    assert out["match_method"][0] == "pdl_domain"
    assert out["domain"][0] == "acmerobotics.com"


def test_websiteless_pdl_rows_are_dropped_from_the_name_track() -> None:
    """Every row keyed `columbia` is a shell; one of them wins on a
    perfect score and brings a "1-10" band with it."""
    tenants = cohort(HARVEY)
    pdl = pdl_frame(
        [
            pdl_row("harvey holdings, inc", "harvey", domain="harveyholdings.com"),
            pdl_row("harvey", "harvey", domain="", size="1-10", size_midpoint=5),
        ]
    )
    out = resolve_pdl(tenants, pdl)
    assert out.height == 1
    assert out["domain"][0] == "harveyholdings.com"


def test_a_tenant_with_no_candidate_gets_no_row() -> None:
    out = resolve_pdl(cohort(HARVEY), pdl_frame([pdl_row("acme robotics", "acmerobotics")]))
    assert out.is_empty()


def test_the_harvey_collision_is_accepted_at_a_perfect_score() -> None:
    """Characterisation of the live failure, so a fix has a witness.

    A legal-AI company and a logistics firm share one blocking key, and
    nothing in the name can separate them. What ships is the logistics
    firm's band, locality and industry under the AI company's tenant.
    """
    out = resolve_pdl(
        cohort(HARVEY),
        pdl_frame(
            [
                pdl_row(
                    "harvey holdings, inc", "harvey",
                    domain="harveyholdings.com", size="51-200",
                    locality="statesville", region="north carolina",
                    industry="logistics and supply chain",
                )
            ]
        ),
    )
    assert out["match_score"][0] == 100.0
    assert out["match_method"][0] == "pdl_name"
    assert out["matched_variant"][0] == "harvey"
    assert out["size"][0] == "51-200"
    assert out["industry"][0] == "logistics and supply chain"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_confidence_expr is applied to the EDGAR track only. The PDL "
        "track supplies the domain, LinkedIn URL, headcount band, "
        "headquarters, industry and founding year, and grades none of "
        "them — so a single-token collision scoring 100 is indistinguishable "
        "from a multi-token match and raises no quality flag."
    ),
)
def test_the_pdl_track_grades_its_matches_too() -> None:
    out = resolve_pdl(cohort(HARVEY), pdl_frame([pdl_row("harvey holdings, inc", "harvey")]))
    assert "match_confidence" in out.columns


# --- the EDGAR track ----------------------------------------------------


def test_a_listed_registrant_wins_a_tie() -> None:
    """A wrong CIK on a private company only costs a Form D history; a
    tenant matching both a filer and a shell is likelier the filer."""
    out = resolve_edgar(
        cohort(ACME),
        edgar_frame(
            [
                {
                    "name": "ACME ROBOTICS LLC", "cik": 111, "ticker": None,
                    "exchange": None, "is_public": False,
                    "name_key_raw": "acmeroboticsllc", "name_key_core": "acmerobotics",
                },
                {
                    "name": "ACME ROBOTICS, INC.", "cik": 222, "ticker": "ACME",
                    "exchange": "NASDAQ", "is_public": True,
                    "name_key_raw": "acmeroboticsinc", "name_key_core": "acmerobotics",
                },
            ]
        ),
    )
    assert out["cik"][0] == 222


def test_a_bare_single_token_cik_is_flagged_for_review() -> None:
    out = resolve_edgar(
        cohort(HARVEY),
        edgar_frame(
            [
                {
                    "name": "HARVEY HOLDINGS INC", "cik": 1_156_231, "ticker": None,
                    "exchange": None, "is_public": False,
                    "name_key_raw": "harveyholdingsinc", "name_key_core": "harvey",
                }
            ]
        ),
    )
    assert out["cik"][0] == 1_156_231
    assert out["match_confidence"][0] == "low"


def test_edgar_returns_an_empty_frame_rather_than_raising() -> None:
    out = resolve_edgar(cohort(HARVEY), edgar_frame([]))
    assert out.is_empty()
    assert "match_confidence" in out.columns


# --- provenance ---------------------------------------------------------


def test_a_match_resting_only_on_the_slug_is_marked() -> None:
    """An iCIMS board at `careers-vanguard` whose directory name is
    "Deerfield Management" picked up vanguard.com this way."""
    tenants = cohort(NGC, ACME)
    resolved = pl.DataFrame(
        {
            "ats": ["workday", "ashby"],
            "slug": [NGC["slug"], "acme"],
            "matched_variant": ["northrop grumman", "acme robotics"],
        }
    )
    marked = _mark_slug_only_matches(resolved, tenants).sort("slug")
    assert marked["matched_on_slug_only"].to_list() == [False, True]


def test_an_unmatched_row_is_not_marked_slug_only() -> None:
    tenants = cohort(ACME)
    resolved = pl.DataFrame({"ats": ["ashby"], "slug": ["acme"], "matched_variant": [""]})
    marked = _mark_slug_only_matches(resolved, tenants)
    assert marked["matched_on_slug_only"][0] is False


# --- thresholds are configuration, not magic ----------------------------


def test_accept_threshold_sits_above_review_threshold() -> None:
    assert config.MATCH_ACCEPT_SCORE > config.MATCH_REVIEW_SCORE
    assert resolve_mod.config is config

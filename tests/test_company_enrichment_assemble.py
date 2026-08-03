"""Tests for the final join (`company_enrichment.assemble`).

Assemble is where the stages stop being independent. It is the only
place that sees the resolution's view of a tenant and the market-cap
stage's view at the same time, which makes it the last chance to notice
that the two disagree — and the place where a contradiction, once
written, becomes a fact to every consumer downstream.

`quality_flags` is the module's own account of what it is unsure about,
so each condition is pinned individually: a flag that silently stops
firing removes a row from someone's review queue without removing the
problem.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pipeline.company_enrichment import config
from pipeline.company_enrichment.assemble import (
    OUTPUT_COLUMNS,
    PUBLIC_COMPANY_COLUMNS,
    _quality_flags,
    _withdraw_funding_from_public,
    assert_public_schema_untouched,
    build,
)

# Every column `_quality_flags` reads, in its no-flag state.
CLEAN: dict[str, object] = {
    "employee_count_agrees_with_band": None,
    "employee_count_band_conflict": "",
    "employee_count_above_floor": None,
    "company_description": None,
    "company_description_source": None,
    "company_description_name_corroborated": None,
    "careers_url": None,
    "careers_url_verified": None,
    "cik_confidence": None,
    "market_cap_check": None,
    "market_cap_basis": None,
    "registrant_state_agrees": None,
    "match_score": None,
    "matched_on_slug_only": None,
}

FLAG_SCHEMA = {
    "employee_count_agrees_with_band": pl.Boolean,
    "employee_count_band_conflict": pl.String,
    "employee_count_above_floor": pl.Boolean,
    "company_description": pl.String,
    "company_description_source": pl.String,
    "company_description_name_corroborated": pl.Boolean,
    "careers_url": pl.String,
    "careers_url_verified": pl.Boolean,
    "cik_confidence": pl.String,
    "market_cap_check": pl.String,
    "market_cap_basis": pl.String,
    "registrant_state_agrees": pl.Boolean,
    "match_score": pl.Float64,
    "matched_on_slug_only": pl.Boolean,
}


def flags_for(**overrides: object) -> list[str]:
    row = dict(CLEAN)
    row.update(overrides)
    frame = pl.DataFrame([row], schema=FLAG_SCHEMA).with_columns(
        _quality_flags().alias("quality_flags")
    )
    raised = frame["quality_flags"][0]
    return raised.split(",") if raised else []


# --- quality flags ------------------------------------------------------


def test_a_clean_row_raises_nothing() -> None:
    assert flags_for() == []


FLAG_CASES = [
    pytest.param(
        {"employee_count_agrees_with_band": False},
        "headcount_contradicts_band",
        id="headcount-vs-band",
    ),
    pytest.param(
        {"employee_count_band_conflict": "exact_count"}, "band_suppressed", id="band-withdrawn"
    ),
    pytest.param(
        {"employee_count_above_floor": False},
        "headcount_below_filed_floor",
        id="below-filed-floor",
    ),
    pytest.param({"cik_confidence": "low"}, "cik_low_confidence", id="weak-cik"),
    pytest.param({"market_cap_check": "disagrees"}, "market_cap_disagrees", id="mcap-disagrees"),
    pytest.param({"market_cap_basis": "parent"}, "market_cap_is_parent", id="rolled-up-mcap"),
    pytest.param(
        {"registrant_state_agrees": False}, "registrant_state_mismatch", id="state-mismatch"
    ),
    pytest.param({"match_score": 88.0}, "weak_name_match", id="weak-score"),
    pytest.param({"matched_on_slug_only": True}, "matched_on_slug_only", id="slug-only"),
    pytest.param(
        {"careers_url": "https://x.test", "careers_url_verified": False},
        "careers_url_unverified",
        id="unverified-careers-url",
    ),
    pytest.param(
        {
            "company_description": "We do things.",
            "company_description_source": "company_site",
            "company_description_name_corroborated": False,
        },
        "description_name_unconfirmed",
        id="description-never-names-tenant",
    ),
]


@pytest.mark.parametrize(("overrides", "expected"), FLAG_CASES)
def test_each_condition_raises_its_own_flag(
    overrides: dict[str, object], expected: str
) -> None:
    assert flags_for(**overrides) == [expected]


def test_wikidata_descriptions_are_exempt_from_the_naming_check() -> None:
    """Wikidata's house style follows the name rather than repeating it,
    so the flag would fire on every one of them and mean nothing."""
    assert (
        flags_for(
            company_description="American investment bank.",
            company_description_source="wikidata",
            company_description_name_corroborated=False,
        )
        == []
    )


def test_a_perfect_score_never_raises_a_weak_match_flag() -> None:
    """The gap this leaves is the whole Harvey class of error: a name
    collision scores 100, so no score-based flag can reach it."""
    assert flags_for(match_score=100.0) == []


def test_flags_accumulate_in_declaration_order() -> None:
    assert flags_for(cik_confidence="low", market_cap_basis="parent") == [
        "cik_low_confidence",
        "market_cap_is_parent",
    ]


# --- funding withdrawal -------------------------------------------------


def funding_frame(is_public: list[bool | None]) -> pl.DataFrame:
    height = len(is_public)
    return pl.DataFrame(
        {
            "is_public": is_public,
            "funding_round_count": [3] * height,
            "funding_total_usd": [1.0e8] * height,
            "funding_rounds": [[{"amount": 1.0}]] * height,
        },
        schema_overrides={
            "is_public": pl.Boolean,
            "funding_round_count": pl.UInt32,
            "funding_rounds": pl.List(pl.Struct({"amount": pl.Float64})),
        },
    )


def test_a_listed_company_loses_its_pre_ipo_raises() -> None:
    out = _withdraw_funding_from_public(funding_frame([True]))
    assert out["funding_round_count"][0] is None
    assert out["funding_total_usd"][0] is None


def test_a_private_company_keeps_its_funding() -> None:
    out = _withdraw_funding_from_public(funding_frame([False]))
    assert out["funding_round_count"][0] == 3


def test_an_unknown_listing_status_is_treated_as_private() -> None:
    out = _withdraw_funding_from_public(funding_frame([None]))
    assert out["funding_round_count"][0] == 3


def test_withdrawal_preserves_the_nested_rounds_dtype() -> None:
    """`when/otherwise` rather than `lit(None)`, or the list-of-structs
    column would not survive being overwritten with an untyped null."""
    out = _withdraw_funding_from_public(funding_frame([True, False]))
    assert out.schema["funding_rounds"] == pl.List(pl.Struct({"amount": pl.Float64}))


# --- the public schema must stay untouched ------------------------------


def test_the_public_companies_schema_guard_passes_today() -> None:
    assert_public_schema_untouched()


def test_the_guard_notices_a_leaked_enrichment_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / ".github" / "scripts" / "publish_companies.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'COLUMNS = ["ats", "name", "slug", "url", "market_cap_usd"]\n'
    )
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="market_cap"):
        assert_public_schema_untouched()


def test_the_guard_notices_a_dropped_public_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / ".github" / "scripts" / "publish_companies.py"
    script.parent.mkdir(parents=True)
    script.write_text('COLUMNS = ["ats", "name", "slug"]\n')
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError, match="url"):
        assert_public_schema_untouched()


def test_the_public_column_set_has_not_grown() -> None:
    assert PUBLIC_COMPANY_COLUMNS == ("ats", "name", "slug", "url")


# --- the join itself ----------------------------------------------------


def resolved_row(**overrides: object) -> pl.DataFrame:
    row: dict[str, object] = {
        "ats": "ashby", "slug": "harvey", "name": "Harvey", "display_name": "Harvey",
        "jobs_company": "harvey", "url": "https://jobs.ashbyhq.com/harvey", "postings": 99,
        "join_method": "url_slug", "pdl_name": "harvey holdings, inc",
        "domain": "harveyholdings.com", "linkedin_url": "li/harvey-holdings",
        "size": "51-200", "size_midpoint": 125, "founded": 2003,
        "locality": "statesville", "region": "north carolina", "industry": "logistics",
        "matched_variant": "harvey", "pdl_score": 100.0, "pdl_method": "pdl_name",
        "edgar_name": "HARVEY HOLDINGS INC", "cik": 1_156_231, "ticker": None,
        "exchange": None, "is_public": False, "edgar_score": 100.0,
        "edgar_method": "edgar_name", "cik_confidence": "low",
        "resolved_name": "HARVEY HOLDINGS INC", "matched_on_slug_only": False,
    }
    row.update(overrides)
    return pl.DataFrame(
        [row],
        schema_overrides={"ticker": pl.String, "exchange": pl.String, "is_public": pl.Boolean},
    )


def rollup_row() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ats": "ashby", "slug": "harvey", "cik": 1_318_220, "ticker": "WCN",
                "exchange": None, "nasdaq_name": "HARVEY HOLDINGS, LLC",
                "market_cap_usd": 4.4077e10, "last_sale_usd": None,
                "shares_outstanding": None, "shares_as_of": None,
                "market_cap_implied_usd": None, "market_cap_disagreement": None,
                "market_cap_check": "parent_rollup", "market_cap_basis": "parent",
                "market_cap_source": "sec_exhibit21+nasdaq_screener",
                "market_cap_as_of": "2026-07-30", "registrant_state": None,
                "registrant_state_agrees": None, "sector": "Utilities",
                "listed_industry": "Environmental Services", "ipo_year": None,
            }
        ],
        schema_overrides={
            "exchange": pl.String, "registrant_state": pl.String,
            "registrant_state_agrees": pl.Boolean, "ipo_year": pl.Int32,
            "last_sale_usd": pl.Float64, "shares_outstanding": pl.Float64,
            "shares_as_of": pl.String, "market_cap_implied_usd": pl.Float64,
            "market_cap_disagreement": pl.Float64,
        },
    )


@pytest.fixture
def staged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every stage path at ``tmp_path``; absent stages are optional."""
    missing = tmp_path / "absent.parquet"
    for attribute in ("FUNDING_PARQUET", "TEAMSIZE_PARQUET", "PROFILE_PARQUET"):
        monkeypatch.setattr(config, attribute, missing)
    monkeypatch.setattr(config, "RESOLVED_PARQUET", tmp_path / "resolved.parquet")
    monkeypatch.setattr(config, "MARKETCAP_PARQUET", tmp_path / "marketcap.parquet")
    return tmp_path


def test_build_requires_the_resolve_stage(staged: Path) -> None:
    with pytest.raises(RuntimeError, match=r"resolved\.parquet is required"):
        build()


def test_build_emits_the_declared_column_order(staged: Path) -> None:
    resolved_row().write_parquet(staged / "resolved.parquet")
    rollup_row().write_parquet(staged / "marketcap.parquet")
    assert tuple(build().columns) == OUTPUT_COLUMNS


def test_the_resolved_cik_survives_the_market_cap_join(staged: Path) -> None:
    """The rollup's CIK belongs to the parent, so the tenant's own wins."""
    resolved_row().write_parquet(staged / "resolved.parquet")
    rollup_row().write_parquet(staged / "marketcap.parquet")
    assert build()["cik"][0] == 1_156_231


def test_a_tenant_with_no_cik_of_its_own_inherits_the_parents(staged: Path) -> None:
    resolved_row(cik=None).write_parquet(staged / "resolved.parquet")
    rollup_row().write_parquet(staged / "marketcap.parquet")
    assert build()["cik"][0] == 1_318_220


def test_a_rolled_up_market_cap_is_flagged_as_the_parents(staged: Path) -> None:
    resolved_row().write_parquet(staged / "resolved.parquet")
    rollup_row().write_parquet(staged / "marketcap.parquet")
    assert "market_cap_is_parent" in build()["quality_flags"][0]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "build coalesces `ticker` from the market-cap stage but never "
        "revisits `is_public`, which stays False from resolve. The row "
        "then asserts private while carrying a ticker, a $44B market cap, "
        "a sector and a listed industry, and any consumer inferring "
        "listing status from those fields renders it public. The same gap "
        "means _withdraw_funding_from_public does not fire on rolled-up "
        "rows."
    ),
)
def test_a_row_carrying_a_ticker_is_not_reported_as_private(staged: Path) -> None:
    resolved_row().write_parquet(staged / "resolved.parquet")
    rollup_row().write_parquet(staged / "marketcap.parquet")
    row = build().row(0, named=True)
    assert row["ticker"] is not None
    assert row["market_cap_usd"] is not None
    assert row["is_public"] is True

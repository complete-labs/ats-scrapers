"""Tests for market capitalisation and the subsidiary rollup
(`company_enrichment.marketcap`).

Two very different risks live in this module. The arithmetic — parsing a
screener figure, comparing it against SEC-filed shares — is easy to get
right and easy to test. The rollup is not: it takes tenants that
*failed* to match a public registrant, the least-verified population in
the cohort, and hands them a ticker, a market cap, a sector and a listed
industry on the strength of a name appearing in some parent's Exhibit
21. Nothing about that population makes a bare name match safer there
than it is anywhere else, and the rollup applies fewer guards than the
resolution stage it bypasses.

`build_subsidiary_map` is exercised against a stubbed SEC transport
rather than the network, so the parsing and keying rules are pinned
without a live dependency.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from pipeline.company_enrichment import marketcap
from pipeline.company_enrichment.marketcap import (
    _parse_exhibit21,
    _state_agreement,
    _to_float,
    build_subsidiary_map,
)

# --- money --------------------------------------------------------------

AMOUNTS = [
    pytest.param("$1,234.5", 1234.5, id="dollar-and-thousands"),
    pytest.param("44077362336", 44_077_362_336.0, id="bare-integer"),
    pytest.param("$0.00", 0.0, id="zero"),
    pytest.param("-12.5", -12.5, id="negative"),
]


@pytest.mark.parametrize(("raw", "expected"), AMOUNTS)
def test_screener_money_parses(raw: str, expected: float) -> None:
    assert _to_float(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "--", "-", ".", "N/A"])
def test_a_missing_figure_is_none_not_zero(raw: object) -> None:
    """A blank market cap means "unknown", never "worthless"."""
    assert _to_float(raw) is None


# --- Exhibit 21 parsing -------------------------------------------------


def test_a_two_column_table_yields_names_without_jurisdictions() -> None:
    html = (
        "<table>"
        "<tr><td>Harvey Holdings, LLC</td><td>Delaware</td></tr>"
        "<tr><td>Acme Widgets Inc.</td><td>Nevada</td></tr>"
        "</table>"
    )
    assert _parse_exhibit21(html) == ["Harvey Holdings, LLC", "Acme Widgets Inc"]


def test_a_flat_list_falls_back_to_lines() -> None:
    """Ex-21 has no prescribed format; plenty of filers use prose."""
    html = "<p>Alpha Systems LLC</p><p>Delaware</p><p>Beta Corp</p>"
    assert _parse_exhibit21(html) == ["Alpha Systems LLC", "Beta Corp"]


def test_jurisdiction_only_cells_are_discarded() -> None:
    html = "<table><tr><td>State of Incorporation</td><td>England and Wales</td></tr></table>"
    assert _parse_exhibit21(html) == []


def test_cells_with_no_letters_are_discarded() -> None:
    html = "<table><tr><td>12345</td><td>Gamma Industries Ltd</td></tr></table>"
    assert _parse_exhibit21(html) == ["Gamma Industries Ltd"]


# --- subsidiary keying --------------------------------------------------


def stub_sec(monkeypatch: pytest.MonkeyPatch, documents: dict[int, str]) -> None:
    """Serve Exhibit 21 HTML per CIK without touching the network."""

    def fake_exhibit_url(cik: int) -> str | None:
        return f"https://example.invalid/{cik}.htm" if cik in documents else None

    def fake_get(url: str, **_kwargs: Any) -> bytes:
        cik = int(url.rsplit("/", 1)[1].removesuffix(".htm"))
        return documents[cik].encode()

    monkeypatch.setattr(marketcap, "_latest_10k_exhibit21", fake_exhibit_url)
    monkeypatch.setattr(marketcap.sechttp, "get", fake_get)


def test_a_subsidiary_maps_to_its_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_sec(
        monkeypatch,
        {1_318_220: "<table><tr><td>Harvey Holdings, LLC</td><td>Delaware</td></tr></table>"},
    )
    mapped = build_subsidiary_map([1_318_220])
    assert mapped.to_dicts() == [
        {
            "subsidiary_key": "harvey",
            "subsidiary_name": "Harvey Holdings, LLC",
            "parent_cik": 1_318_220,
        }
    ]


def test_a_parent_with_no_exhibit_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_sec(monkeypatch, {})
    mapped = build_subsidiary_map([999])
    assert mapped.is_empty()
    assert mapped.columns == ["subsidiary_key", "subsidiary_name", "parent_cik"]


def test_a_short_key_is_not_worth_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only guard the rollup applies is a five-character floor."""
    stub_sec(monkeypatch, {7: "<table><tr><td>ACME</td><td>Delaware</td></tr></table>"})
    assert build_subsidiary_map([7]).is_empty()


def test_the_soft_suffix_collapse_reaches_the_rollup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterisation of the live failure.

    Waste Connections lists "Harvey Holdings, LLC" among its
    subsidiaries. That reduces to the key ``harvey``, which is also the
    key of an unrelated Ashby tenant, and the rollup joins on the
    tenant's raw directory name with no score, no confidence grade and
    no stop-word filter — so the tenant inherits a $44B market cap and
    the sector "Utilities".
    """
    stub_sec(
        monkeypatch,
        {1_318_220: "<table><tr><td>Harvey Holdings, LLC</td><td>Delaware</td></tr></table>"},
    )
    assert build_subsidiary_map([1_318_220])["subsidiary_key"][0] == "harvey"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "build_subsidiary_map filters only on key length, not on how much "
        "the key discriminates. 'Solutions Holdings, LLC' becomes the key "
        "'solutions', which resolve._STOP_KEYS already refuses to block on "
        "in the name track. The rollup does not consult that list, so a "
        "tenant literally called Solutions inherits Ameresco's market cap."
    ),
)
def test_a_generic_subsidiary_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_sec(
        monkeypatch,
        {5: "<table><tr><td>Solutions Holdings, LLC</td><td>Delaware</td></tr></table>"},
    )
    assert build_subsidiary_map([5]).is_empty()


# --- the one independent check on a name-only public match --------------


def test_state_agreement_only_speaks_when_both_sides_are_us_states() -> None:
    """A foreign registrant is *expected* to differ from a US locality,
    so the comparison is skipped rather than scored as a mismatch."""
    frame = pl.DataFrame(
        {"registrant": ["CA", "CA", None, "L2"], "tenant": ["CA", "TX", "CA", "CA"]}
    ).with_columns(
        _state_agreement(pl.col("registrant"), pl.col("tenant")).alias("agrees")
    )
    assert frame["agrees"].to_list() == [True, False, None, None]


def test_the_tenant_side_may_be_any_case() -> None:
    frame = pl.DataFrame({"registrant": ["CA"], "tenant": ["ca"]}).with_columns(
        _state_agreement(pl.col("registrant"), pl.col("tenant")).alias("agrees")
    )
    assert frame["agrees"][0] is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The equality test uppercases both sides, but the gate deciding "
        "whether the pair is comparable at all checks the raw registrant "
        "against _US_STATE_CODES. A lowercase registrant state is therefore "
        "judged non-US and skipped rather than compared. Latent today — SEC "
        "submissions return stateOfIncorporation uppercased — so this "
        "records an inconsistency, not a live miss."
    ),
)
def test_the_registrant_side_may_be_any_case_too() -> None:
    frame = pl.DataFrame({"registrant": ["ca"], "tenant": ["CA"]}).with_columns(
        _state_agreement(pl.col("registrant"), pl.col("tenant")).alias("agrees")
    )
    assert frame["agrees"][0] is True


def test_crosscheck_tolerance_is_a_fraction_not_a_multiple() -> None:
    """Guards against someone reading it as the band slack's 3.0."""
    assert 0 < marketcap.CROSSCHECK_TOLERANCE < 1

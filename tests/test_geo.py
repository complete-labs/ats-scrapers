"""Tests for the shared country resolver.

The inputs here are real spellings taken from live ATS payloads: Lever
sends a lowercase alpha-2, Amazon an alpha-3, Workday the ISO long name,
Ashby whatever the employer typed.
"""

from __future__ import annotations

import pytest

from ats_scrapers.enrichment import country_to_iso, region_for, resolve_country


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # alpha-2, in the casings different ATSes use
        ("us", "US"),
        ("US", "US"),
        ("gb", "GB"),
        ("au", "AU"),
        # alpha-3 (Amazon's country_code)
        ("USA", "US"),
        ("GBR", "GB"),
        ("COL", "CO"),
        # ISO long names (Workday descriptors)
        ("United States of America", "US"),
        ("Korea, Republic of", "KR"),
        ("Bolivia (Plurinational State of)", "BO"),
        ("Viet Nam", "VN"),
        # everyday spellings and colloquial short forms
        ("United States", "US"),
        ("United Kingdom", "GB"),
        ("UK", "GB"),
        ("England", "GB"),
        ("South Korea", "KR"),
        ("Czech Republic", "CZ"),
        ("Turkey", "TR"),
        ("Vietnam", "VN"),
        ("Ivory Coast", "CI"),
        # accents and localised names
        ("Türkiye", "TR"),
        ("España", "ES"),
        ("Deutschland", "DE"),
        ("Côte d'Ivoire", "CI"),
        # Kosovo has no ISO assignment but appears in live payloads
        ("xk", "XK"),
        # whitespace and punctuation noise
        ("  united   states  ", "US"),
    ],
)
def test_resolves_country(value: str, expected: str) -> None:
    assert country_to_iso(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        # Names a market or "anywhere", not a country. Mapping these to an
        # arbitrary member state would be worse than returning nothing.
        "European Union",
        "Global",
        "Any Location",
        "Worldwide",
        "Remote",
        "EMEA",
        "APAC",
        "Europe",
        "unknown",
        "",
        "   ",
        "ZZ",
        "not a country at all",
        None,
        42,
    ],
)
def test_rejects_non_countries(value: object) -> None:
    assert country_to_iso(value) is None


@pytest.mark.parametrize(
    ("iso", "expected"),
    [
        ("US", "North America"),
        ("CA", "North America"),
        ("MX", "North America"),
        ("CR", "North America"),
        ("BR", "South America"),
        ("AR", "South America"),
        ("GB", "Europe"),
        ("DE", "Europe"),
        ("JP", "Asia"),
        ("IN", "Asia"),
        ("TW", "Asia"),
        ("ZA", "Africa"),
        ("AU", "Oceania"),
        ("NZ", "Oceania"),
        ("AQ", "Antarctica"),
    ],
)
def test_region_for(iso: str, expected: str) -> None:
    assert region_for(iso) == expected


def test_region_for_rejects_unknown() -> None:
    assert region_for("ZZ") is None
    assert region_for(None) is None


def test_resolve_country_returns_both() -> None:
    assert resolve_country("USA") == ("US", "North America")
    assert resolve_country("European Union") == (None, None)


def test_every_country_has_a_documented_region() -> None:
    """``Job.region`` documents a closed vocabulary — nothing may fall
    outside it."""
    from ats_scrapers.enrichment.geo import _COUNTRIES

    allowed = {
        "Europe", "North America", "South America",
        "Asia", "Africa", "Oceania", "Antarctica",
    }
    assert {region for _, region, _ in _COUNTRIES.values()} <= allowed
    assert all(len(code) == 2 and code.isupper() for code in _COUNTRIES)

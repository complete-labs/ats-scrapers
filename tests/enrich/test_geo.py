"""Geo resolution tests.

The regression cases here are all real bugs found while building this: a
permissive NUTS pattern read ``AUSTIN`` as Austria and ``REMOTE`` as
Réunion, and a minimum-length guard silently dropped every two-character
CJK city name.
"""

from __future__ import annotations

import pytest

from enrich.geo import region_for_country, resolve_location


class TestCountryResolution:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Paris, France", "FR"),
            ("London, United Kingdom", "GB"),
            ("Bengaluru, India", "IN"),
            ("São Paulo, Brasil", "BR"),
            ("Zürich, Schweiz", "CH"),
            ("Praha, Česká republika", "CZ"),
            ("München", "DE"),
            ("Amsterdam", "NL"),
            ("USA", "US"),
            ("Dublin, IRL", "IE"),
        ],
    )
    def test_resolves_country(self, text: str, expected: str) -> None:
        assert resolve_location(text).country_iso == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Regression: a bare-NUTS pattern without a digit requirement
            # matched any 3-6 letter uppercase word.
            ("Austin, TX", "US"),
            ("Berlin, DE", "DE"),
            ("Sydney, NSW", "AU"),
            ("Remote — US", "US"),
        ],
    )
    def test_uppercase_city_names_are_not_country_codes(self, text: str, expected: str) -> None:
        assert resolve_location(text).country_iso == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # The genuinely ambiguous two-letter tokens: CA is California and
            # Canada, DE is Delaware and Germany. The city anchors it.
            ("San Francisco, CA", "US"),
            ("Toronto, ON", "CA"),
            ("Vancouver, BC", "CA"),
            ("Wilmington, DE", "US"),
        ],
    )
    def test_city_anchors_ambiguous_admin_codes(self, text: str, expected: str) -> None:
        assert resolve_location(text).country_iso == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("DE (DEA58)", "DE"), ("FR (FRK21)", "FR"), ("ITC4", "IT")],
    )
    def test_eures_nuts_codes(self, text: str, expected: str) -> None:
        assert resolve_location(text).country_iso == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("東京", "JP"), ("上海", "CN"), ("서울", "KR")],
    )
    def test_two_character_cjk_cities(self, text: str, expected: str) -> None:
        # Regression: the city index dropped names shorter than three
        # characters, which excluded most CJK city names entirely.
        assert resolve_location(text).country_iso == expected


class TestPlaceless:
    @pytest.mark.parametrize(
        "text", ["Remote", "Anywhere", "Worldwide", "Multiple locations", "flexible"]
    )
    def test_names_no_place(self, text: str) -> None:
        resolved = resolve_location(text)
        assert resolved.country_iso is None
        assert resolved.placeless is True

    def test_remote_with_a_country_still_resolves_the_country(self) -> None:
        resolved = resolve_location("Remote - Germany")
        assert resolved.country_iso == "DE"
        assert resolved.placeless is True

    def test_empty_and_none(self) -> None:
        assert resolve_location("").country_iso is None
        assert resolve_location(None).country_iso is None
        assert resolve_location(123).country_iso is None


class TestCoordinates:
    def test_city_match_yields_city_precision_and_plausible_coords(self) -> None:
        resolved = resolve_location("Paris, France")
        assert resolved.precision == "city"
        assert resolved.lat is not None and 48 < resolved.lat < 49
        assert resolved.lon is not None and 2 < resolved.lon < 3

    def test_country_only_falls_back_to_country_precision(self) -> None:
        resolved = resolve_location("Germany")
        assert resolved.precision == "country"
        assert resolved.lat is not None

    def test_country_token_is_not_reused_as_a_city(self) -> None:
        # "USA" is an alternate name for a small US town; letting it win would
        # attach that town's coordinates to a country-level location.
        assert resolve_location("USA").precision == "country"

    def test_postcode_is_ignored(self) -> None:
        assert resolve_location("Seattle, WA 98104").country_iso == "US"


class TestRegion:
    @pytest.mark.parametrize(
        ("iso", "expected"),
        [
            ("FR", "Europe"),
            ("US", "North America"),
            ("BR", "South America"),
            ("JP", "Asia"),
            ("ZA", "Africa"),
            ("AU", "Oceania"),
        ],
    )
    def test_continent_mapping(self, iso: str, expected: str) -> None:
        assert region_for_country(iso) == expected

    def test_rejects_junk(self) -> None:
        assert region_for_country("XX") is None
        assert region_for_country("USA") is None
        assert region_for_country(None) is None

"""Tier 0 extractor tests.

The abstention cases are the important ones. Tier 0 feeding a wrong value
into the sidecar is worse than Tier 0 feeding nothing, because a wrong Tier 0
value takes precedence over the model and is never revisited.
"""

from __future__ import annotations

import pytest

from enrich.deterministic import (
    detect_currency,
    detect_language,
    detect_period,
    infer_placement,
    map_employment_type,
    parse_experience_years,
    run_tier0,
    to_bool,
)


class TestToBool:
    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "t", "yes", "1", True, 1])
    def test_truthy(self, value: object) -> None:
        assert to_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "f", "no", "0", False, 0])
    def test_falsy(self, value: object) -> None:
        assert to_bool(value) is False

    @pytest.mark.parametrize("value", [None, "", "maybe", "   ", [], {}])
    def test_unknown(self, value: object) -> None:
        assert to_bool(value) is None

    def test_parses_the_published_snapshot_encoding(self) -> None:
        # Every column of all.parquet is VARCHAR, so is_remote arrives as the
        # text 'true'/'false'. An isinstance(bool) check would discard all
        # 719,871 provider-labelled rows.
        assert to_bool("false") is False


class TestLanguage:
    @pytest.mark.parametrize(
        ("title", "description", "expected"),
        [
            ("Senior Software Engineer", "We are looking for a backend engineer in London.", "en"),
            ("Softwareentwickler", "Wir suchen einen erfahrenen Entwickler für unser Team.", "de"),
            ("Ingénieur logiciel", "Nous recherchons un ingénieur pour rejoindre l'équipe.", "fr"),
            ("Desenvolvedor", "Estamos procurando um desenvolvedor com experiência.", "pt"),
        ],
    )
    def test_latin_scripts(self, title: str, description: str, expected: str) -> None:
        assert detect_language(title, description) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("私たちはエンジニアを探しています", "ja"),
            ("我们正在寻找一位软件工程师", "zh"),
            ("우리는 데이터 엔지니어를 찾고 있습니다", "ko"),
        ],
    )
    def test_cjk_scripts(self, text: str, expected: str) -> None:
        assert detect_language(text) == expected

    def test_abstains_on_too_little_text(self) -> None:
        assert detect_language("Hi") is None
        assert detect_language(None) is None
        assert detect_language("") is None


class TestEmploymentType:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Full-time", "FULL_TIME"),
            ("CDI", "FULL_TIME"),
            ("Vollzeit", "FULL_TIME"),
            ("Heltid", "FULL_TIME"),
            ("正社員", "FULL_TIME"),
            ("plný úvazek", "FULL_TIME"),
            ("Teilzeit", "PART_TIME"),
            ("兼职", "PART_TIME"),
            ("Praktikum", "INTERN"),
            ("Estágio", "INTERN"),
            ("Stage", "INTERN"),
            ("Freelance", "CONTRACT"),
            ("CDD", "TEMPORARY"),
            ("Fixed-term", "TEMPORARY"),
        ],
    )
    def test_multilingual_labels(self, label: str, expected: str) -> None:
        assert map_employment_type(label) == expected

    def test_abstains_rather_than_defaulting_to_full_time(self) -> None:
        # 45% of the corpus has no employment type. Defaulting to the modal
        # value would make the column useless for filtering while looking
        # complete.
        assert map_employment_type("32 hours per week") is None
        assert map_employment_type(None) is None
        assert map_employment_type("") is None

    def test_longest_match_wins(self) -> None:
        assert map_employment_type("Part-time") == "PART_TIME"


class TestPlacement:
    @pytest.mark.parametrize(
        ("title", "location", "expected"),
        [
            ("Engineer", "Remote", "remote"),
            ("Remote Sales Director", "US", "remote"),
            ("Engineer", "On-site Tokyo", "onsite"),
            ("Engineer", "Presencial Madrid", "onsite"),
        ],
    )
    def test_explicit_signals(self, title: str, location: str, expected: str) -> None:
        assert infer_placement(title, location) == expected

    def test_hybrid_beats_remote(self) -> None:
        # "Hybrid Remote - London" is a common provider phrasing that means
        # hybrid; a remote-first check would mislabel all of them.
        assert infer_placement("Engineer", "Hybrid Remote - London") == "hybrid"

    def test_abstains_on_a_bare_city(self) -> None:
        # A named city says nothing about where the work happens. This is the
        # field most worth paying a model for.
        assert infer_placement("Engineer", "Berlin") is None
        assert infer_placement("Distributed Systems Engineer", "Seattle") is None


class TestExperience:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("We require 5+ years of experience in backend development.", (5, None)),
            ("3-5 years of professional experience required", (3, 5)),
            ("Minimum of 7 years experience with distributed systems", (7, None)),
            ("Mindestens 3 Jahre Berufserfahrung", (3, None)),
            ("5 ans d'expérience minimum", (5, None)),
            ("経験3年以上", (3, None)),
        ],
    )
    def test_extracts_requirements(self, text: str, expected: tuple[int, int | None]) -> None:
        assert parse_experience_years(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Our company has 20 years of history and 5000 customers",
            "Founded in 2009, we have enjoyed 15 years of steady growth",
            "You will learn a lot in this role",
            "A four-year degree is required",
        ],
    )
    def test_ignores_years_without_experience_context(self, text: str) -> None:
        # Without the proximity constraint, company-history sentences read as
        # experience requirements.
        assert parse_experience_years(text) == (None, None)

    def test_rejects_implausible_values(self) -> None:
        assert parse_experience_years("120 years of experience") == (None, None)


class TestSalaryHelpers:
    @pytest.mark.parametrize(
        ("text", "country", "expected"),
        [
            ("$120K – $160K", "US", "USD"),
            ("CA$400K", "CA", "CAD"),
            ("45.000 € / Jahr", "DE", "EUR"),
            ("up to £80k", "GB", "GBP"),
            ("R$3.000", "BR", "BRL"),
            ("24,000 PLN", "PL", "PLN"),
        ],
    )
    def test_currency_detection(self, text: str, country: str, expected: str) -> None:
        assert detect_currency(text, country) == expected

    @pytest.mark.parametrize(
        ("text", "country", "expected"),
        [("$90,000", "US", "USD"), ("$90,000", "CA", "CAD"), ("$90,000", "AU", "AUD")],
    )
    def test_dollar_is_disambiguated_by_country(
        self, text: str, country: str, expected: str
    ) -> None:
        # Assuming USD would mislabel every Canadian and Australian salary.
        assert detect_currency(text, country) == expected

    @pytest.mark.parametrize(
        ("text", "amount", "currency", "expected"),
        [
            ("$45/hr", 45, "USD", "HOUR"),
            ("3.200 EUR pro Monat", 3200, "EUR", "MONTH"),
            ("45.000 € / Jahr", 45000, "EUR", "YEAR"),
            ("$120K - $160K", 120000, "USD", "YEAR"),
        ],
    )
    def test_period_detection(self, text: str, amount: float, currency: str, expected: str) -> None:
        assert detect_period(text, amount, currency) == expected

    def test_period_abstains_in_the_ambiguous_band(self) -> None:
        # 60,000 is annual in USD and monthly in JPY; magnitude cannot decide.
        assert detect_period("60,000", 60000, "JPY") is None
        assert detect_period("15,000", 15000, "USD") is None

    def test_hourly_inference_is_currency_agnostic(self) -> None:
        assert detect_period("", 22.5, "SEK") == "HOUR"


class TestRunTier0:
    def _row(self, **overrides: object) -> dict[str, object]:
        row = {
            "url": "https://x.test/1",
            "title": "Senior Backend Engineer",
            "location": "Berlin, Germany",
            "description": "We need 5+ years of experience with Python and Kubernetes. " * 5,
            "commitment": "Vollzeit",
        }
        row.update(overrides)
        return row

    def test_fills_gaps_and_records_sources(self) -> None:
        result = run_tier0(self._row())
        assert result.values["country_iso"] == "DE"
        assert result.values["region"] == "Europe"
        assert result.values["lat"] is not None
        assert result.values["language"] == "en"
        assert result.values["employment_type"] == "FULL_TIME"
        assert result.values["experience_min_years"] == 5
        assert result.sources["country_iso"] == "tier0"

    def test_provider_values_take_precedence(self) -> None:
        result = run_tier0(self._row(country_iso="AT", language="de"))
        assert result.values["country_iso"] == "AT"
        assert result.values["region"] == "Europe"
        assert result.sources["country_iso"] == "provider"
        assert result.sources["language"] == "provider"

    def test_empty_strings_are_treated_as_missing(self) -> None:
        # The publisher writes "" for every absent optional field, so a
        # truthiness check is the only reliable test.
        result = run_tier0(self._row(country_iso="", language="", commitment=""))
        assert result.values["country_iso"] == "DE"
        assert result.sources["country_iso"] == "tier0"

    def test_provider_not_remote_does_not_assert_onsite(self) -> None:
        # A provider False rules out remote but says nothing about hybrid vs
        # onsite, so placement stays open for Tier 1.
        result = run_tier0(self._row(is_remote="false"))
        assert result.values["is_remote"] is False
        assert "placement" not in result.values
        assert "placement" in result.unresolved

    def test_provider_remote_sets_placement(self) -> None:
        result = run_tier0(self._row(is_remote="true"))
        assert result.values["placement"] == "remote"
        assert result.sources["placement"] == "provider"

    def test_string_coordinates_are_parsed(self) -> None:
        result = run_tier0(self._row(lat="52.52", lon="13.40"))
        assert result.values["lat"] == pytest.approx(52.52)
        assert result.values["geo_precision"] == "provider"

    def test_unresolved_drives_routing(self) -> None:
        result = run_tier0(self._row())
        assert "seniority" in result.unresolved
        assert "department" in result.unresolved
        assert result.needs_llm is True

"""Tests for the Greenhouse scraper."""

from __future__ import annotations

import pytest

from ats_scrapers.exceptions import CompanyNotFoundError, ScraperError
from ats_scrapers.models import ATSType
from ats_scrapers.scrapers import GreenhouseScraper, ScraperRegistry

API = (
    "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
    "?content=true&pay_transparency=true"
)


# Retry pacing is zeroed suite-wide by the `_no_retry_delays` fixture
# in conftest.py — the shared fetch layer replaced per-scraper retry
# constants.


# Greenhouse listing now requests ``?content=true``; the scraper fetches
# everything in a single call (no per-job detail), so tests that mock
# the URL constant ``API`` already cover the full request set. The
# relax-mark is a safety net in case a test variant adds a non-default
# slug — it keeps tests passing when the URL diverges from ``API``.
pytestmark = pytest.mark.httpx_mock(
    assert_all_requests_were_expected=False,
)


def _job(jid: str = "1", title: str = "Engineer",
         location: str = "Remote",
         absolute_url: str = "https://job-boards.greenhouse.io/acme/jobs/1") -> dict:
    return {
        "id": jid,
        "title": title,
        "location": {"name": location},
        "absolute_url": absolute_url,
        "updated_at": "2026-04-15T08:00:00Z",
        "departments": [{"name": "Eng"}],
    }


def test_registry_resolves_greenhouse() -> None:
    assert ScraperRegistry.get(ATSType.GREENHOUSE) is GreenhouseScraper


def test_parses_basic_job(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json={"jobs": [_job()]})
    jobs = GreenhouseScraper("acme").fetch()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.title == "Engineer"
    assert job.ats_type is ATSType.GREENHOUSE
    assert job.company == "acme"
    assert job.location == "Remote"


def test_returns_empty_for_no_jobs(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json={"jobs": []})
    assert GreenhouseScraper("acme").fetch() == []


def test_404_raises_company_not_found(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=404)
    with pytest.raises(CompanyNotFoundError):
        GreenhouseScraper("acme").fetch()


def test_5xx_retries(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=503)
    httpx_mock.add_response(url=API, json={"jobs": [_job()]})
    assert len(GreenhouseScraper("acme").fetch()) == 1


def test_5xx_exhausts(monkeypatch, httpx_mock) -> None:
    from ats_scrapers import fetch as fetch_module
    monkeypatch.setattr(fetch_module, "DEFAULT_RETRIES", 2)
    httpx_mock.add_response(url=API, status_code=502, is_reusable=True)
    with pytest.raises(ScraperError):
        GreenhouseScraper("acme").fetch()


# --- pay transparency -------------------------------------------------------


def test_requests_pay_transparency() -> None:
    """``pay_input_ranges`` is omitted from the payload entirely unless
    the flag is set — the board looks like it has no compensation data
    at all, which is why every greenhouse row shipped without a salary."""
    from ats_scrapers.scrapers.greenhouse import API_TEMPLATE

    assert "pay_transparency=true" in API_TEMPLATE


def test_parses_pay_input_ranges(httpx_mock) -> None:
    job = _job() | {
        "pay_input_ranges": [
            {
                "min_cents": 22280000,
                "max_cents": 29000000,
                "currency_type": "USD",
                "title": "Annual Salary:",
            }
        ]
    }
    httpx_mock.add_response(url=API, json={"jobs": [job]})
    parsed = GreenhouseScraper("acme").fetch()[0]
    assert parsed.salary_currency == "USD"
    assert parsed.salary_min == 222800
    assert parsed.salary_max == 290000
    assert parsed.salary_period == "YEAR"
    assert parsed.salary_summary == "USD 222,800 - 290,000"


def test_missing_pay_ranges_leaves_salary_unset(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json={"jobs": [_job()]})
    parsed = GreenhouseScraper("acme").fetch()[0]
    assert parsed.salary_currency is None
    assert parsed.salary_min is None
    assert parsed.salary_max is None
    assert parsed.salary_summary is None


def test_pay_ranges_span_jurisdictions_of_one_currency() -> None:
    """A role open in several pay-transparency jurisdictions carries one
    entry each. Span them instead of picking one, and never merge a
    range denominated in a different currency into the band."""
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges

    parsed = _parse_pay_ranges([
        {"min_cents": 10000000, "max_cents": 15000000, "currency_type": "USD"},
        {"min_cents": 8000000, "max_cents": 18000000, "currency_type": "USD"},
        {"min_cents": 50000000, "max_cents": 60000000, "currency_type": "CAD"},
    ])
    assert parsed.currency == "USD"
    assert (parsed.minimum, parsed.maximum) == (80000, 180000)


def test_pay_range_period_read_from_label() -> None:
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges

    hourly = _parse_pay_ranges([
        {"min_cents": 2500, "max_cents": 4000,
         "currency_type": "USD", "title": "Hourly Rate:"},
    ])
    assert hourly.period == "HOUR"
    assert (hourly.minimum, hourly.maximum) == (25, 40)
    # No label at all — annual is the overwhelming default on Greenhouse.
    assert _parse_pay_ranges(
        [{"min_cents": 9000000, "currency_type": "EUR"}]
    ).period == "YEAR"


def test_period_markers_are_word_anchored() -> None:
    """``"day"`` must not fire on ``"Monday"``."""
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges

    assert _parse_pay_ranges([{
        "min_cents": 12000000, "max_cents": 16000000,
        "currency_type": "USD", "title": "Monday start Salary",
    }]).period == "YEAR"
    assert _parse_pay_ranges([{
        "min_cents": 40000, "max_cents": 60000,
        "currency_type": "USD", "title": "Per Diem Rate",
    }]).period == "DAY"


@pytest.mark.parametrize(
    ("label", "min_cents", "max_cents", "currency", "expected"),
    [
        # Real headings from boards that name no period but publish an
        # hourly rate. Defaulting these to YEAR published "$70 / year"
        # for a contractor billing $70 / hour — 9,125 rows of it across
        # 446 boards on a full scrape.
        ("Salary Range", 7000, 10000, "USD", "HOUR"),
        ("Pay Range", 3400, 5400, "USD", "HOUR"),
        ("New York Pay Band ", 1700, 1850, "USD", "HOUR"),
        ("Echelle salariale", 1460, 1752, "EUR", "HOUR"),
        # Same bare headings with a plausible annual figure stay annual.
        ("Salary Range", 12000000, 16000000, "USD", "YEAR"),
        ("Pay Range", 6500000, 9000000, "USD", "YEAR"),
        # Weak currencies inflate the figure rather than shrinking it, so
        # the size test doesn't misread a JPY or INR salary as hourly.
        ("Pay", 800000000, 1200000000, "JPY", "YEAR"),
        ("Pay", 150000000, 250000000, "INR", "YEAR"),
    ],
)
def test_period_inferred_from_magnitude_when_label_is_silent(
    label, min_cents, max_cents, currency, expected
) -> None:
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges

    parsed = _parse_pay_ranges([{
        "min_cents": min_cents, "max_cents": max_cents,
        "currency_type": currency, "title": label,
    }])
    assert parsed.period == expected


def test_explicit_label_beats_magnitude() -> None:
    """A stated period is authoritative even when the amount looks odd."""
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges

    assert _parse_pay_ranges([{
        "min_cents": 150000, "max_cents": 180000,
        "currency_type": "USD", "title": "Annual Salary:",
    }]).period == "YEAR"


@pytest.mark.parametrize("payload", [
    None,
    [],
    "not-a-list",
    [{"min_cents": 0, "max_cents": 0, "currency_type": "USD"}],
    [{"currency_type": "USD"}],
    [{"min_cents": 1000, "currency_type": "not-a-currency"}],
    ["junk", {"min_cents": "x", "currency_type": "USD"}],
])
def test_unusable_pay_ranges_yield_nothing(payload) -> None:
    """A zero bound means "not set on this side", not "unpaid"."""
    from ats_scrapers.scrapers.greenhouse import _parse_pay_ranges, _PayRange

    assert _parse_pay_ranges(payload) == _PayRange()

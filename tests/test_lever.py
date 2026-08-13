"""Tests for the Lever scraper."""

from __future__ import annotations

import pytest

from ats_scrapers.exceptions import CompanyNotFoundError
from ats_scrapers.models import ATSType
from ats_scrapers.scrapers import LeverScraper, ScraperRegistry

API = "https://api.lever.co/v0/postings/acme?mode=json"


# Retry pacing is zeroed suite-wide by the `_no_retry_delays` fixture
# in conftest.py — the shared fetch layer replaced per-scraper retry
# constants.


def _job(jid: str = "x1", text: str = "SWE",
         location: str = "Remote") -> dict:
    return {
        "id": jid,
        "text": text,
        "hostedUrl": f"https://jobs.lever.co/acme/{jid}",
        "categories": {"location": location, "team": "Eng"},
        "createdAt": 1714521600000,  # ~2026-04-30
    }


def test_registry_resolves_lever() -> None:
    assert ScraperRegistry.get(ATSType.LEVER) is LeverScraper


def test_parses_basic_job(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json=[_job()])
    jobs = LeverScraper("acme").fetch()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.title == "SWE"
    assert job.company == "acme"
    assert job.location == "Remote"
    assert job.ats_id == "x1"


def test_returns_empty_list(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json=[])
    assert LeverScraper("acme").fetch() == []


def test_404_company_not_found(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=404)
    with pytest.raises(CompanyNotFoundError):
        LeverScraper("acme").fetch()


def test_5xx_retries(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=502)
    httpx_mock.add_response(url=API, json=[_job()])
    assert len(LeverScraper("acme").fetch()) == 1


@pytest.mark.parametrize(
    ("workplace_type", "expected"),
    [
        ("remote", True),
        ("onsite", False),
        ("on-site", False),
        # Hybrid requires office attendance, so the role is not remote.
        # Matches the convention used by the Ashby and Workday scrapers.
        ("hybrid", False),
        ("", None),
    ],
)
def test_workplace_type_decides_is_remote(httpx_mock, workplace_type, expected) -> None:
    httpx_mock.add_response(url=API, json=[_job() | {"workplaceType": workplace_type}])
    assert LeverScraper("acme").fetch()[0].is_remote is expected


def test_country_code_populates_country_iso(httpx_mock) -> None:
    """Lever ships an alpha-2 ``country`` that used to sit unused in
    ``raw``, leaving the publisher to guess from the location text."""
    httpx_mock.add_response(url=API, json=[_job(location="Berlin") | {"country": "DE"}])
    job = LeverScraper("acme").fetch()[0]
    assert job.country_iso == "DE"
    assert job.region == "Europe"


def test_missing_country_leaves_iso_unset(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json=[_job()])
    job = LeverScraper("acme").fetch()[0]
    assert job.country_iso is None
    assert job.region is None


def test_posted_at_is_utc_not_host_local(httpx_mock) -> None:
    """``createdAt`` is epoch milliseconds; parsing it without a timezone
    silently shifted every timestamp by the pipeline host's UTC offset."""
    httpx_mock.add_response(url=API, json=[_job() | {"createdAt": 1767225600000}])
    posted_at = LeverScraper("acme").fetch()[0].posted_at
    assert posted_at is not None
    assert posted_at.tzinfo is not None
    assert (posted_at.year, posted_at.month, posted_at.day) == (2026, 1, 1)

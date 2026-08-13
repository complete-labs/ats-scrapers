"""Structured location extraction across the high-volume sources.

Each of these ATSes ships a machine-readable country (and sometimes real
coordinates) that used to be dropped on the floor, leaving the publisher
to guess a country from free text like "SF Office". The payload shapes
below are taken verbatim from live responses.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ats_scrapers.scrapers.amazon import AmazonScraper
from ats_scrapers.scrapers.oracle import OracleScraper
from ats_scrapers.scrapers.smartrecruiters import SmartRecruitersScraper
from ats_scrapers.scrapers.workday import WorkdayScraper

# --- SmartRecruiters -------------------------------------------------------


def _sr_posting(**location_overrides) -> dict:
    location = {
        "city": "Austin",
        "region": "TX",
        "country": "us",
        "remote": False,
        "hybrid": False,
        "latitude": "30.267153",
        "longitude": "-97.7430608",
        "fullLocation": "Austin, TX, United States",
    }
    location.update(location_overrides)
    return {
        "id": "744000133907678",
        "name": "Senior Engineer",
        "releasedDate": "2026-06-24T10:00:11.853Z",
        "location": location,
    }


def test_smartrecruiters_country_and_coordinates() -> None:
    job = SmartRecruitersScraper("acme")._parse_job(_sr_posting())
    assert job.country_iso == "US"
    assert job.region == "North America"
    assert (job.lat, job.lon) == (30.267153, -97.7430608)


def test_smartrecruiters_uses_full_location_string() -> None:
    """The lowercase ISO code used to leak into the display string as
    ``"Austin, TX, us"``."""
    job = SmartRecruitersScraper("acme")._parse_job(_sr_posting())
    assert job.location == "Austin, TX, United States"


def test_smartrecruiters_drops_empty_location_segments() -> None:
    job = SmartRecruitersScraper("acme")._parse_job(
        _sr_posting(fullLocation="LILLEBONNE, , France")
    )
    assert job.location == "LILLEBONNE, France"


def test_smartrecruiters_upper_cases_country_without_full_location() -> None:
    job = SmartRecruitersScraper("acme")._parse_job(_sr_posting(fullLocation=None))
    assert job.location == "Austin, TX, US"


@pytest.mark.parametrize(
    ("remote", "hybrid", "expected"),
    [
        (True, False, True),
        (False, True, False),  # hybrid is not remote
        (False, False, False),
    ],
)
def test_smartrecruiters_hybrid_is_not_remote(remote, hybrid, expected) -> None:
    job = SmartRecruitersScraper("acme")._parse_job(
        _sr_posting(remote=remote, hybrid=hybrid)
    )
    assert job.is_remote is expected


def test_smartrecruiters_ignores_placeholder_coordinates() -> None:
    job = SmartRecruitersScraper("acme")._parse_job(
        _sr_posting(latitude="0", longitude="0")
    )
    assert job.lat is None and job.lon is None


def test_smartrecruiters_survives_missing_coordinates() -> None:
    job = SmartRecruitersScraper("acme")._parse_job(
        _sr_posting(latitude=None, longitude=None)
    )
    assert job.lat is None and job.lon is None
    assert job.country_iso == "US"


# --- Oracle ----------------------------------------------------------------


def test_oracle_country_from_primary_location_country() -> None:
    item = {
        "Id": "1005247",
        "Title": "Senior Engineer",
        "PrimaryLocation": "Sydney, Australia",
        "PrimaryLocationCountry": "AU",
        "ExternalURL": "https://example.oraclecloud.com/job/1005247",
    }
    job = OracleScraper("acme")._parse_job(item, "https://example.oraclecloud.com", "1")
    assert job.country_iso == "AU"
    assert job.region == "Oceania"


def test_oracle_without_country_stays_none() -> None:
    item = {
        "Id": "1",
        "Title": "Engineer",
        "PrimaryLocation": "Somewhere",
        "ExternalURL": "https://example.oraclecloud.com/job/1",
    }
    job = OracleScraper("acme")._parse_job(item, "https://example.oraclecloud.com", "1")
    assert job.country_iso is None


# --- Amazon ----------------------------------------------------------------


def _amazon_location(city: str, iso2: str, coords: str) -> str:
    return json.dumps(
        {
            "normalizedLocation": city,
            "countryIso2a": iso2,
            "coordinates": coords,
            "type": "ONSITE",
        }
    )


def test_amazon_per_location_country_and_coordinates() -> None:
    hit = {
        "id_icims": "3080000",
        "title": "Software Development Engineer",
        "job_path": "/en/jobs/3080000/sde",
        "country_code": "USA",
        "normalized_location": "Seattle, Washington, USA",
        "locations": [
            _amazon_location("Seattle, Washington, USA", "US", "47.60357,-122.32945"),
            _amazon_location("London, England, GBR", "GB", "51.50643,-0.12719"),
        ],
    }
    rows = AmazonScraper("amazon")._parse_hit(hit)
    by_location = {r.location: r for r in rows}
    seattle = by_location["Seattle, Washington, USA"]
    london = by_location["London, England, GBR"]

    assert (seattle.country_iso, seattle.region) == ("US", "North America")
    assert (seattle.lat, seattle.lon) == (47.60357, -122.32945)
    # A multi-country posting must not stamp the primary country on every row.
    assert (london.country_iso, london.region) == ("GB", "Europe")
    assert (london.lat, london.lon) == (51.50643, -0.12719)


def test_amazon_falls_back_to_posting_country_code() -> None:
    hit = {
        "id_icims": "1",
        "title": "SDE",
        "job_path": "/en/jobs/1/sde",
        "country_code": "COL",
        "normalized_location": "Bogota, D.C., COL",
    }
    rows = AmazonScraper("amazon")._parse_hit(hit)
    assert len(rows) == 1
    assert rows[0].country_iso == "CO"
    assert rows[0].region == "South America"


def test_amazon_emits_a_row_when_no_location_is_named() -> None:
    rows = AmazonScraper("amazon")._parse_hit(
        {"id_icims": "2", "title": "SDE", "job_path": "/en/jobs/2/sde"}
    )
    assert len(rows) == 1
    assert rows[0].location is None


# --- Workday ---------------------------------------------------------------


_WORKDAY_DETAIL = {
    "jobPostingInfo": {
        "jobDescription": "<p>Build things.</p>",
        "location": "California, USA - Remote",
        "startDate": "2026-08-13",
        "timeType": "Full time",
        "remoteType": "Remote",
        "country": {"descriptor": "United States of America"},
    }
}


def _hydrate(detail: dict, listing: dict | None = None):
    """Run one job through the detail-hydration pass against a stub API."""
    base = "https://acme.wd1.myworkdayjobs.com/ext"
    scraper = WorkdayScraper(base)
    job = scraper._parse_job(
        listing
        or {
            "title": "Principal Engineer",
            "externalPath": "/job/California/Principal-Engineer_1",
            "locationsText": "California, USA - Remote",
            "postedOn": "Posted Today",
            "bulletFields": ["26WD97363"],
        },
        base,
        "acme",
    )
    jobs = [job]

    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=detail))

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await scraper._enrich_details(
                client,
                asyncio.Semaphore(1),
                "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/ext",
                jobs,
            )

    asyncio.run(run())
    return jobs[0]


def test_workday_detail_fills_country_posted_and_type() -> None:
    """All of this rides along on the detail request the scraper already
    makes for descriptions."""
    job = _hydrate(_WORKDAY_DETAIL)
    assert job.country_iso == "US"
    assert job.region == "North America"
    assert job.posted_at is not None
    assert (job.posted_at.year, job.posted_at.month, job.posted_at.day) == (2026, 8, 13)
    assert job.posted_at.tzinfo is not None  # UTC, not the host's local zone
    assert job.employment_type == "FULL_TIME"
    assert job.is_remote is True
    assert job.description


def test_workday_detail_handles_missing_fields() -> None:
    job = _hydrate({"jobPostingInfo": {"jobDescription": "<p>Body.</p>"}})
    assert job.country_iso is None
    assert job.posted_at is None
    assert job.description


def test_workday_hybrid_is_not_remote() -> None:
    detail = {"jobPostingInfo": dict(_WORKDAY_DETAIL["jobPostingInfo"], remoteType="Hybrid")}
    assert _hydrate(detail).is_remote is False


def test_workday_vague_remote_label_stays_unknown() -> None:
    """"Flex" may mean hybrid or may mean flexible hours."""
    detail = {"jobPostingInfo": dict(_WORKDAY_DETAIL["jobPostingInfo"], remoteType="Flex")}
    assert _hydrate(detail).is_remote is None


# --- Amazon workplace type -------------------------------------------------


@pytest.mark.parametrize(
    ("location_type", "expected"),
    [
        ("VIRTUAL", True),  # Amazon's spelling for fully remote
        ("ONSITE", False),
        ("HYBRID", False),
        ("SOMETHING_ELSE", None),
    ],
)
def test_amazon_is_remote_from_location_type(location_type, expected) -> None:
    hit = {
        "id_icims": "5",
        "title": "SDE",
        "job_path": "/en/jobs/5/sde",
        "country_code": "USA",
        "normalized_location": "Seattle, Washington, USA",
        "locations": [
            json.dumps(
                {
                    "normalizedLocation": "Seattle, Washington, USA",
                    "countryIso2a": "US",
                    "coordinates": "47.60357,-122.32945",
                    "type": location_type,
                }
            )
        ],
    }
    assert AmazonScraper("amazon")._parse_hit(hit)[0].is_remote is expected


# --- Oracle workplace type -------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("ORA_REMOTE", True),
        ("ORA_ON_SITE", False),
        ("ORA_HYBRID", False),
        ("ORA_MYSTERY", None),
    ],
)
def test_oracle_is_remote_from_workplace_code(code, expected) -> None:
    item = {
        "Id": "1",
        "Title": "Engineer",
        "PrimaryLocation": "Sydney, Australia",
        "PrimaryLocationCountry": "AU",
        "WorkplaceTypeCode": code,
        "ExternalURL": "https://example.oraclecloud.com/job/1",
    }
    job = OracleScraper("acme")._parse_job(item, "https://example.oraclecloud.com", "1")
    assert job.is_remote is expected

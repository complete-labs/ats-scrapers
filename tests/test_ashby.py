"""Tests for the Ashby scraper.

The fixtures mirror shapes taken verbatim from live boards — in
particular the hybrid ``isRemote: true`` combination and the per-zone
compensation tiers, both of which produced wrong rows before.
"""

from __future__ import annotations

import pytest

from ats_scrapers.exceptions import CompanyNotFoundError
from ats_scrapers.models import ATSType
from ats_scrapers.scrapers import AshbyScraper, ScraperRegistry

API = "https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true"


# Retry pacing is zeroed suite-wide by the `_no_retry_delays` fixture
# in conftest.py — the shared fetch layer replaced per-scraper retry
# constants.


def _job(jid: str = "j1", title: str = "SWE", location: str = "Remote", **extra) -> dict:
    return {
        "id": jid,
        "title": title,
        "location": location,
        "jobUrl": f"https://jobs.ashbyhq.com/acme/{jid}",
        "publishedAt": "2026-04-15T08:00:00.000Z",
        **extra,
    }


def _address(country: str, locality: str | None = None) -> dict:
    postal: dict[str, str] = {"addressCountry": country}
    if locality:
        postal["addressLocality"] = locality
    return {"postalAddress": postal}


def _fetch_one(httpx_mock, job: dict):
    httpx_mock.add_response(url=API, json={"jobs": [job]})
    return AshbyScraper("acme").fetch()[0]


def test_registry_resolves_ashby() -> None:
    assert ScraperRegistry.get(ATSType.ASHBY) is AshbyScraper


def test_parses_basic_job(httpx_mock) -> None:
    job = _fetch_one(httpx_mock, _job())
    assert job.title == "SWE"
    assert job.company == "acme"
    assert job.ats_type is ATSType.ASHBY


def test_returns_empty_list(httpx_mock) -> None:
    httpx_mock.add_response(url=API, json={"jobs": []})
    assert AshbyScraper("acme").fetch() == []


def test_404_company_not_found(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=404)
    with pytest.raises(CompanyNotFoundError):
        AshbyScraper("acme").fetch()


def test_5xx_retries(httpx_mock) -> None:
    httpx_mock.add_response(url=API, status_code=503)
    httpx_mock.add_response(url=API, json={"jobs": [_job()]})
    assert len(AshbyScraper("acme").fetch()) == 1


# --- Placement -------------------------------------------------------------


@pytest.mark.parametrize(
    ("workplace_type", "is_remote_flag", "expected"),
    [
        ("Remote", True, True),
        # Ashby sets isRemote=true on hybrid roles too; workplaceType wins.
        ("Hybrid", True, False),
        ("OnSite", False, False),
        ("On-site", False, False),
        (None, None, None),
    ],
)
def test_workplace_type_decides_is_remote(
    httpx_mock, workplace_type, is_remote_flag, expected
) -> None:
    job = _fetch_one(
        httpx_mock, _job(workplaceType=workplace_type, isRemote=is_remote_flag)
    )
    assert job.is_remote is expected


def test_bare_is_remote_true_is_not_trusted(httpx_mock) -> None:
    """Without ``workplaceType`` a ``true`` flag only means "not fully
    on-site", which does not justify claiming the role is remote."""
    assert _fetch_one(httpx_mock, _job(isRemote=True)).is_remote is None


def test_bare_is_remote_false_is_trusted(httpx_mock) -> None:
    assert _fetch_one(httpx_mock, _job(isRemote=False)).is_remote is False


# --- Location / country ----------------------------------------------------


def test_country_and_region_from_postal_address(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock, _job(location="SF Office", address=_address("United States", "San Francisco"))
    )
    assert job.country_iso == "US"
    assert job.region == "North America"


def test_country_alias_is_normalized(httpx_mock) -> None:
    assert _fetch_one(httpx_mock, _job(address=_address("USA"))).country_iso == "US"


def test_supranational_country_is_not_mapped(httpx_mock) -> None:
    job = _fetch_one(httpx_mock, _job(address=_address("European Union")))
    assert job.country_iso is None
    assert job.region is None


def test_secondary_locations_join_into_location(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            location="New York",
            address=_address("United States", "New York"),
            secondaryLocations=[
                {"location": "Boston", "address": _address("United States", "Boston")},
                {"location": "Southeast", "address": _address("United States")},
            ],
        ),
    )
    assert job.location == "New York, Boston, Southeast"
    assert job.country_iso == "US"
    assert job.raw["secondary_locations"] == ["Boston", "Southeast"]


def test_cross_border_posting_has_no_single_country(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            location="London",
            address=_address("United Kingdom", "London"),
            secondaryLocations=[
                {"location": "New York", "address": _address("United States", "New York")}
            ],
        ),
    )
    assert job.location == "London, New York"
    assert job.country_iso is None
    assert job.region is None


# --- Compensation ----------------------------------------------------------


def _tier(minimum: int, maximum: int, currency: str = "USD", interval: str = "1 YEAR") -> dict:
    return {
        "components": [
            {
                "compensationType": "Salary",
                "interval": interval,
                "currencyCode": currency,
                "minValue": minimum,
                "maxValue": maximum,
            }
        ]
    }


def test_compensation_summary_passthrough(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock, _job(compensation={"compensationTierSummary": "$120k - $180k"})
    )
    assert job.salary_summary == "$120k - $180k"


def test_structured_salary_from_summary_components(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            compensation={
                "compensationTierSummary": "$257K – $335K • Offers Equity",
                "summaryComponents": [
                    # Non-salary components come first on real payloads.
                    {
                        "compensationType": "EquityCashValue",
                        "interval": "1 YEAR",
                        "currencyCode": "USD",
                        "minValue": None,
                        "maxValue": None,
                    },
                    {
                        "compensationType": "Salary",
                        "interval": "1 YEAR",
                        "currencyCode": "USD",
                        "minValue": 257000,
                        "maxValue": 335000,
                    },
                ],
            }
        ),
    )
    assert (job.salary_min, job.salary_max) == (257000, 335000)
    assert job.salary_currency == "USD"
    assert job.salary_period == "YEAR"


def test_multi_zone_tiers_span_the_full_range(httpx_mock) -> None:
    """A posting with per-zone bands must publish the widest band, not the
    first one — the tier summary already reports the span."""
    job = _fetch_one(
        httpx_mock,
        _job(
            compensation={
                "compensationTierSummary": "$165.4K – $285K • Multiple Ranges",
                "compensationTiers": [
                    _tier(207000, 285000),
                    _tier(186300, 256500),
                    _tier(165400, 228000),
                ],
            }
        ),
    )
    assert (job.salary_min, job.salary_max) == (165400, 285000)


def test_mixed_currency_tiers_do_not_merge(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            compensation={
                "compensationTiers": [
                    _tier(230000, 385000, currency="USD"),
                    _tier(131000, 245000, currency="GBP"),
                ]
            }
        ),
    )
    assert job.salary_currency == "USD"
    assert (job.salary_min, job.salary_max) == (230000, 385000)


def test_hourly_interval_is_preserved(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(compensation={"compensationTiers": [_tier(24, 28, interval="1 HOUR")]}),
    )
    assert job.salary_period == "HOUR"
    assert (job.salary_min, job.salary_max) == (24, 28)


def test_unknown_interval_is_not_guessed_as_annual(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(compensation={"compensationTiers": [_tier(4000, 5000, interval="2 WEEK")]}),
    )
    assert job.salary_period is None
    assert (job.salary_min, job.salary_max) == (4000, 5000)


def test_equity_only_compensation_has_no_range(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            compensation={
                "compensationTierSummary": "Offers Equity",
                "summaryComponents": [
                    {
                        "compensationType": "EquityCashValue",
                        "interval": "1 YEAR",
                        "currencyCode": "USD",
                        "minValue": None,
                        "maxValue": None,
                    }
                ],
            }
        ),
    )
    assert job.salary_summary == "Offers Equity"
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.salary_currency is None


# --- Misc fields -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("FullTime", "FULL_TIME"),
        ("PartTime", "PART_TIME"),
        ("Contract", "CONTRACT"),
        ("Intern", "INTERN"),
        ("Temporary", "TEMPORARY"),
        ("Mystery", None),
    ],
)
def test_employment_type_mapping(httpx_mock, raw_type, expected) -> None:
    job = _fetch_one(httpx_mock, _job(employmentType=raw_type))
    assert job.employment_type == expected


def test_derived_apply_url_is_dropped(httpx_mock) -> None:
    """Ashby mints ``{jobUrl}/application`` on every posting; it is not a
    separate apply destination."""
    job = _fetch_one(
        httpx_mock, _job(applyUrl="https://jobs.ashbyhq.com/acme/j1/application")
    )
    assert job.apply_url is None


def test_distinct_apply_url_is_kept(httpx_mock) -> None:
    job = _fetch_one(httpx_mock, _job(applyUrl="https://acme.example/apply/j1"))
    assert str(job.apply_url) == "https://acme.example/apply/j1"


def test_html_description_preferred_and_truncated(httpx_mock) -> None:
    job = _fetch_one(
        httpx_mock,
        _job(
            descriptionHtml="<p>" + "x" * 30_000,
            descriptionPlain="plain text",
        ),
    )
    assert job.description.startswith("<p>")
    assert len(job.description) == 25_000


def test_plain_description_fallback(httpx_mock) -> None:
    job = _fetch_one(httpx_mock, _job(descriptionPlain="plain text"))
    assert job.description == "plain text"

"""Tests for the per-ATS scrapers and the registry plumbing."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Several scrapers retry 3× on 429/5xx with a 1.5s base delay → up
    to 9s per failing test. Knock those down to 0 so tests stay fast."""
    for mod_name in ("greenhouse", "lever", "ashby"):
        try:
            mod = __import__(f"ats_scrapers.scrapers.{mod_name}", fromlist=[""])
        except ImportError:
            continue
        if hasattr(mod, "MAX_RETRIES"):
            monkeypatch.setattr(mod, "MAX_RETRIES", 1)
        if hasattr(mod, "RETRY_BASE_DELAY"):
            monkeypatch.setattr(mod, "RETRY_BASE_DELAY", 0.0)

from ats_scrapers.exceptions import CompanyNotFoundError, ScraperError  # noqa: E402
from ats_scrapers.models import ATSType  # noqa: E402
from ats_scrapers.scrapers import (  # noqa: E402
    AshbyScraper,
    BaseScraper,
    GreenhouseScraper,
    LeverScraper,
    ScraperRegistry,
    get_scraper,
)

# --- Registry ----------------------------------------------------------------

def test_registry_contains_known_scrapers() -> None:
    registered = ScraperRegistry.all()
    assert registered[ATSType.GREENHOUSE] is GreenhouseScraper
    assert registered[ATSType.LEVER] is LeverScraper
    assert registered[ATSType.ASHBY] is AshbyScraper


def test_registry_keys_are_valid_ats_types() -> None:
    """Every registered scraper must map to a real `ATSType`."""
    registered = ScraperRegistry.all()
    for ats in registered:
        assert isinstance(ats, ATSType)


def test_sources_without_scraper_adapters_are_explicit() -> None:
    registered = set(ScraperRegistry.all())
    assert set(ATSType) - registered == {ATSType.CUSTOM}


def test_registry_covers_core_atses() -> None:
    """Sanity check: the core production ATSes always have a scraper."""
    registered = ScraperRegistry.all()
    core = {
        ATSType.GREENHOUSE,
        ATSType.LEVER,
        ATSType.ASHBY,
        ATSType.SMARTRECRUITERS,
        ATSType.WORKABLE,
        ATSType.RIPPLING,
        ATSType.WORKDAY,
    }
    assert core.issubset(set(registered.keys()))


def test_get_scraper_returns_instance() -> None:
    scraper = get_scraper("greenhouse", "acme")
    assert isinstance(scraper, GreenhouseScraper)
    assert scraper.company_slug == "acme"


def test_get_scraper_accepts_enum_too() -> None:
    scraper = get_scraper(ATSType.LEVER, "acme")
    assert isinstance(scraper, LeverScraper)


def test_get_scraper_unknown_ats_raises() -> None:
    with pytest.raises(ScraperError):
        get_scraper("custom", "acme")


def test_registry_returns_copy_so_external_mutation_is_safe() -> None:
    snapshot = ScraperRegistry.all()
    snapshot.pop(ATSType.GREENHOUSE, None)
    assert ATSType.GREENHOUSE in ScraperRegistry.all()


def test_has_scraper() -> None:
    assert ScraperRegistry.has_scraper("greenhouse") is True
    assert ScraperRegistry.has_scraper(ATSType.GREENHOUSE) is True
    assert ScraperRegistry.has_scraper("beisen") is True
    # A source name not even in the enum must not raise.
    assert ScraperRegistry.has_scraper("futuristic_ats_9000") is False


def test_register_decorator_adds_new_scraper() -> None:
    @ScraperRegistry.register(ATSType.CUSTOM)
    class TempScraper(BaseScraper):
        ats = ATSType.CUSTOM

        def fetch(self):
            return []

    try:
        assert ScraperRegistry.get(ATSType.CUSTOM) is TempScraper
    finally:
        ScraperRegistry._scrapers.pop(ATSType.CUSTOM, None)


# --- BaseScraper -------------------------------------------------------------

def test_base_scraper_repr() -> None:
    scraper = GreenhouseScraper("acme")
    assert repr(scraper) == "GreenhouseScraper('acme')"


def test_base_scraper_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseScraper("x")  # type: ignore[abstract]


def test_base_scraper_default_timeout() -> None:
    scraper = GreenhouseScraper("acme")
    assert scraper.timeout == 30.0


def test_base_scraper_custom_timeout() -> None:
    scraper = GreenhouseScraper("acme", timeout=5.0)
    assert scraper.timeout == 5.0


def test_display_company_defaults_to_slug() -> None:
    assert GreenhouseScraper("acme").display_company == "acme"


def test_display_company_prefers_configured_name() -> None:
    scraper = GreenhouseScraper("acme", company_name="Acme Corp")
    assert scraper.display_company == "Acme Corp"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_display_company_ignores_blank_names(blank) -> None:
    scraper = GreenhouseScraper("acme", company_name=blank)
    assert scraper.company_name is None
    assert scraper.display_company == "acme"


@pytest.mark.parametrize(
    "ats",
    [
        ATSType.GREENHOUSE, ATSType.LEVER, ATSType.ASHBY, ATSType.WORKABLE,
        ATSType.TEAMTAILOR, ATSType.SMARTRECRUITERS, ATSType.RIPPLING,
        ATSType.PINPOINT, ATSType.JOIN_COM, ATSType.JAZZHR, ATSType.GEM,
        ATSType.BAMBOOHR,
    ],
)
def test_slug_only_scrapers_accept_a_company_name(ats: ATSType) -> None:
    """These once published their board slug as the employer name.

    A slug is a routing token: Greenhouse's ``anthropic`` and Welcome
    to the Jungle's ``Anthropic`` are one employer that split into two
    facets downstream. Each must now take the curated display name.
    """
    scraper = get_scraper(ats, "acme", company_name="Acme Corp")
    assert scraper.display_company == "Acme Corp"


@pytest.mark.parametrize(
    ("ats", "slug"),
    [
        (ATSType.ORACLE, "https://careers.acme.com"),
        (ATSType.PHENOM, "https://jobs.acme.com"),
        (ATSType.PERSONIO, "https://acme.jobs.personio.de"),
    ],
)
def test_hostname_fallback_yields_to_the_curated_name(
    ats: ATSType, slug: str
) -> None:
    """These three derive the employer from the careers hostname.

    Left alone that publishes ``jobs.acme.com`` as the employer name.
    Every tenant row for all three carries a curated ``name``, so the
    hostname is only ever a last resort.
    """
    scraper = get_scraper(ats, slug, company_name="Acme Corp")
    assert scraper.company_name == "Acme Corp"


# --- Greenhouse --------------------------------------------------------------

# Derived from the scraper's own template rather than hardcoded, so a
# change to the query string can't silently desync these mocks.
def _gh_url(slug: str) -> str:
    from ats_scrapers.scrapers.greenhouse import API_TEMPLATE

    return API_TEMPLATE.format(slug=slug)


GH_SAMPLE = {
    "jobs": [
        {
            "id": 4567,
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/4567",
            "title": "Software Engineer",
            "location": {"name": "San Francisco"},
            "updated_at": "2026-04-01T12:00:00Z",
        },
        {
            "id": 4568,
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/4568",
            "title": "Research Scientist",
            "location": None,
            "updated_at": None,
        },
    ]
}


def test_greenhouse_parses_jobs(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_gh_url("acme"),
        json=GH_SAMPLE,
    )
    jobs = GreenhouseScraper("acme").fetch()
    assert len(jobs) == 2
    assert jobs[0].title == "Software Engineer"
    assert jobs[0].location == "San Francisco"
    assert jobs[0].posted_at is not None
    assert jobs[1].location is None
    assert jobs[1].posted_at is None


def test_greenhouse_raises_company_not_found_on_404(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_gh_url("missing"),
        status_code=404,
    )
    with pytest.raises(CompanyNotFoundError):
        GreenhouseScraper("missing").fetch()


def test_greenhouse_raises_scraper_error_on_5xx(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_gh_url("x"),
        status_code=503,
        is_reusable=True,  # retry now fires; mock must satisfy all attempts
    )
    with pytest.raises(ScraperError):
        GreenhouseScraper("x").fetch()


def test_greenhouse_raises_on_network_failure(httpx_mock) -> None:
    import httpx

    httpx_mock.add_exception(
        httpx.ConnectError("boom"),
        url=_gh_url("x"),
        is_reusable=True,
    )
    with pytest.raises(ScraperError):
        GreenhouseScraper("x").fetch()


def test_greenhouse_handles_empty_jobs_list(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_gh_url("empty"),
        json={"jobs": []},
    )
    assert GreenhouseScraper("empty").fetch() == []


# --- Lever -------------------------------------------------------------------

LEVER_SAMPLE = [
    {
        "id": "abc-123",
        "hostedUrl": "https://jobs.lever.co/acme/abc-123",
        "text": "Backend Engineer",
        "categories": {"location": "Remote"},
        "createdAt": 1735689600000,  # 2025-01-01
    },
    {
        "id": "def-456",
        "hostedUrl": "https://jobs.lever.co/acme/def-456",
        "text": "Designer",
        "categories": None,
        "createdAt": None,
    },
]


def test_lever_parses_jobs(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.lever.co/v0/postings/acme?mode=json",
        json=LEVER_SAMPLE,
    )
    jobs = LeverScraper("acme").fetch()
    assert len(jobs) == 2
    assert jobs[0].title == "Backend Engineer"
    assert jobs[0].location == "Remote"
    assert jobs[0].posted_at is not None
    assert jobs[1].location is None


def test_lever_404(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.lever.co/v0/postings/missing?mode=json",
        status_code=404,
    )
    with pytest.raises(CompanyNotFoundError):
        LeverScraper("missing").fetch()


def test_lever_5xx(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.lever.co/v0/postings/x?mode=json",
        status_code=500,
        is_reusable=True,
    )
    with pytest.raises(ScraperError):
        LeverScraper("x").fetch()


# --- Ashby -------------------------------------------------------------------

ASHBY_SAMPLE = {
    "jobs": [
        {
            "id": "job-uuid-1",
            "title": "Founding Engineer",
            "location": "New York",
            "jobUrl": "https://jobs.ashbyhq.com/ramp/job-uuid-1",
            "publishedAt": "2026-03-15T10:00:00Z",
            "compensation": {
                "compensationTierSummary": "$200K - $300K",
                "scrapeableCompensationSalarySummary": "$200K - $300K",
                "compensationTiers": [
                    {
                        "components": [
                            {
                                "compensationType": "Salary",
                                "interval": "1 YEAR",
                                "minValue": 200000,
                                "maxValue": 300000,
                                "currencyCode": "USD",
                            },
                            {
                                "compensationType": "EquityPercentage",
                                "interval": "NONE",
                                "minValue": None,
                                "maxValue": None,
                                "currencyCode": None,
                            },
                        ]
                    }
                ],
            },
        },
        {
            "id": "job-uuid-2",
            "title": "Product Designer",
            "location": "Remote",
            "applyUrl": "https://jobs.ashbyhq.com/ramp/job-uuid-2/apply",
            "publishedAt": None,
            "compensation": None,
        },
    ]
}


def test_ashby_parses_jobs_with_compensation(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/ramp?includeCompensation=true",
        json=ASHBY_SAMPLE,
    )
    jobs = AshbyScraper("ramp").fetch()
    assert len(jobs) == 2

    eng = jobs[0]
    assert eng.title == "Founding Engineer"
    assert eng.salary_currency == "USD"
    assert eng.salary_min == 200000
    assert eng.salary_max == 300000
    assert eng.salary_period == "YEAR"
    assert eng.salary_summary == "$200K - $300K"

    designer = jobs[1]
    assert designer.salary_currency is None
    assert str(designer.url).endswith("/apply")


def _ashby_comp_payload(*component_types: str) -> dict:
    """One Ashby job whose tier lists the given component types.

    Mirrors the live shape: only Salary carries values; the rest are
    disclosure flags with null amounts.
    """
    components = []
    for kind in component_types:
        if kind == "Salary":
            components.append({
                "compensationType": "Salary", "interval": "1 YEAR",
                "minValue": 200000, "maxValue": 300000, "currencyCode": "USD",
            })
        else:
            components.append({
                "compensationType": kind, "interval": "1 YEAR",
                "minValue": None, "maxValue": None, "currencyCode": "USD",
                "summary": f"Offers {kind}",
            })
    return {
        "jobs": [{
            "id": "x", "title": "X", "location": "Remote",
            "jobUrl": "https://jobs.ashbyhq.com/x/x", "publishedAt": None,
            "compensation": {"compensationTiers": [{"components": components}]},
        }]
    }


@pytest.mark.parametrize(
    ("component", "expected_field"),
    [
        ("EquityPercentage", "offers_equity"),
        ("EquityCashValue", "offers_equity"),
        ("Bonus", "offers_bonus"),
        ("Commission", "offers_commission"),
    ],
)
def test_ashby_surfaces_non_salary_package_components(
    httpx_mock, component: str, expected_field: str
) -> None:
    """Audit finding 12: equity and total comp weren't represented.

    These components were parsed and then discarded — only ``Salary``
    reached the schema, leaving the rest reachable solely by digging
    through ``raw``.
    """
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/ramp?includeCompensation=true",
        json=_ashby_comp_payload("Salary", component),
    )
    job = AshbyScraper("ramp").fetch()[0]
    assert getattr(job, expected_field) is True
    assert job.salary_min == 200000


def test_ashby_leaves_absent_components_unknown(httpx_mock) -> None:
    """Not mentioning equity is not the same as excluding it."""
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/ramp?includeCompensation=true",
        json=_ashby_comp_payload("Salary"),
    )
    job = AshbyScraper("ramp").fetch()[0]
    assert job.offers_equity is None
    assert job.offers_bonus is None
    assert job.offers_commission is None


def test_ashby_404(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/missing?includeCompensation=true",
        status_code=404,
    )
    with pytest.raises(CompanyNotFoundError):
        AshbyScraper("missing").fetch()


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("HOURLY", "HOUR"),
        ("DAILY", "DAY"),
        ("WEEKLY", "WEEK"),
        ("MONTHLY", "MONTH"),
        ("ANNUALLY", "YEAR"),
        ("YEARLY", "YEAR"),
    ],
)
def test_ashby_interval_mapping(httpx_mock, interval: str, expected: str) -> None:
    payload = {
        "jobs": [
            {
                "id": "x",
                "title": "X",
                "location": "Remote",
                "jobUrl": "https://jobs.ashbyhq.com/x/x",
                "publishedAt": None,
                "compensation": {
                    "compensationTiers": [
                        {
                            "components": [
                                {
                                    "compensationType": "Salary",
                                    "interval": interval,
                                    "minValue": 1,
                                    "maxValue": 2,
                                    "currencyCode": "USD",
                                }
                            ]
                        }
                    ]
                },
            }
        ]
    }
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/co?includeCompensation=true",
        json=payload,
    )
    jobs = AshbyScraper("co").fetch()
    assert jobs[0].salary_period == expected


def test_ashby_handles_compensation_without_tiers(httpx_mock) -> None:
    """Summary string surfaces even when structured tiers are absent."""
    payload = {
        "jobs": [
            {
                "id": "x",
                "title": "X",
                "location": "Remote",
                "jobUrl": "https://jobs.ashbyhq.com/co/x",
                "publishedAt": None,
                "compensation": {"compensationTierSummary": "Competitive"},
            }
        ]
    }
    httpx_mock.add_response(
        url="https://api.ashbyhq.com/posting-api/job-board/co?includeCompensation=true",
        json=payload,
    )
    jobs = AshbyScraper("co").fetch()
    assert jobs[0].salary_currency is None
    assert jobs[0].salary_min is None
    assert jobs[0].salary_summary == "Competitive"

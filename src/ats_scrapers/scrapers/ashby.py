"""Ashby scraper.

Ashby exposes a public JSON board at:
    https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true

Every posting carries a structured ``address.postalAddress`` (present on
~97% of live rows) that we map to ``country_iso`` / ``region``, and a rich
``compensation`` block (range + currency + interval) on the ~19% of
postings whose employer opted into pay transparency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

from ats_scrapers.enrichment.geo import resolve_country
from ats_scrapers.models import ATSType, EmploymentType, Job, SalaryPeriod
from ats_scrapers.scrapers.base import BaseScraper, ScraperRegistry

if TYPE_CHECKING:
    from typing import Any

API_TEMPLATE = "https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true"

# Ashby's current payloads prefix the unit with a multiplier ("1 YEAR",
# "1 HOUR"); older ones use the bare adverb. Anything we don't recognise
# resolves to ``None`` rather than a guessed ``YEAR`` — publishing an
# hourly rate as an annual salary is the one error here that changes the
# number by three orders of magnitude.
_INTERVAL_MAP: dict[str, SalaryPeriod] = {
    "HOUR": "HOUR",
    "HOURLY": "HOUR",
    "DAY": "DAY",
    "DAILY": "DAY",
    "WEEK": "WEEK",
    "WEEKLY": "WEEK",
    "MONTH": "MONTH",
    "MONTHLY": "MONTH",
    "YEAR": "YEAR",
    "YEARLY": "YEAR",
    "ANNUALLY": "YEAR",
}

_EMPLOYMENT_TYPE_MAP: dict[str, EmploymentType] = {
    "FULLTIME": "FULL_TIME",
    "FULL_TIME": "FULL_TIME",
    "PARTTIME": "PART_TIME",
    "PART_TIME": "PART_TIME",
    "CONTRACT": "CONTRACT",
    "CONTRACTOR": "CONTRACT",
    "INTERNSHIP": "INTERN",
    "INTERN": "INTERN",
    "TEMPORARY": "TEMPORARY",
}

# ``workplaceType`` is the authoritative placement signal. ``isRemote`` is
# *not*: Ashby sets it to ``true`` for hybrid roles as well as fully remote
# ones, so trusting it publishes a quarter of the board as remote work that
# actually requires office attendance. Hybrid is ``False`` here — the role
# cannot be performed remotely, it can only be performed partly remotely.
_WORKPLACE_REMOTE = {
    "remote": True,
    "hybrid": False,
    "onsite": False,
    "inperson": False,
    "office": False,
}

_MAX_DESCRIPTION_CHARS = 25_000


@ScraperRegistry.register(ATSType.ASHBY)
class AshbyScraper(BaseScraper):
    ats = ATSType.ASHBY

    async def afetch(self) -> list[Job]:
        url = API_TEMPLATE.format(slug=self.company_slug)
        async with self.make_fetcher() as fetch:
            payload = await fetch.get_json(url)
        return [self._parse_job(item) for item in payload.get("jobs", [])]

    def _parse_job(self, item: dict[str, Any]) -> Job:
        comp = item.get("compensation") or {}
        summary = (
            comp.get("compensationTierSummary")
            or comp.get("scrapeableCompensationSalarySummary")
        )
        salary = _parse_comp(comp)
        package = _parse_package(comp)

        emp_type = (item.get("employmentType") or "").upper()
        employment_type = _EMPLOYMENT_TYPE_MAP.get(emp_type)

        place = _locations(item)

        # Description — prefer ``descriptionHtml`` over ``descriptionPlain``.
        # The HTML form retains paragraph breaks, bullet lists, and headings
        # (the plain text concatenates them into a single block); the
        # post-scrape markdownify step in scripts/normalize_descriptions.py
        # then converts the HTML into clean markdown. Plain stays as a
        # last-ditch fallback.
        description = _clean_description(
            item.get("descriptionHtml") or item.get("descriptionPlain")
        )

        job_url = item.get("jobUrl") or item.get("applyUrl")

        raw: dict[str, Any] = {}
        if item.get("department"):
            raw["department"] = item["department"]
        if item.get("team"):
            raw["team"] = item["team"]
        if place.secondary:
            raw["secondary_locations"] = list(place.secondary)
        if item.get("address"):
            raw["address"] = item["address"]
        if item.get("workplaceType"):
            raw["workplace_type"] = item["workplaceType"]
        if comp:
            # Keep the full compensation tier structure for downstream consumers
            # who want to surface bonus/equity/commission separately, or the
            # per-zone bands that ``salary_min``/``salary_max`` span over.
            raw["compensation_tiers"] = comp.get("compensationTiers")

        return Job(
            url=job_url,
            title=item["title"],
            company=self.display_company,
            ats_type=ATSType.ASHBY,
            ats_id=item["id"],
            location=place.display,
            country_iso=place.country_iso,
            region=place.region,
            is_remote=_is_remote(item),
            description=description,
            employment_type=employment_type,
            department=item.get("department") if isinstance(item.get("department"), str) else None,
            team=item.get("team") if isinstance(item.get("team"), str) else None,
            apply_url=_apply_url(item.get("applyUrl"), job_url),
            salary_currency=salary.currency,
            salary_period=salary.period,
            salary_summary=summary,
            salary_min=salary.minimum,
            salary_max=salary.maximum,
            **package,
            posted_at=_parse_iso(item.get("publishedAt")),
            fetched_at=datetime.now(UTC),
            raw=raw or None,
        )


def _is_remote(item: dict[str, Any]) -> bool | None:
    """Resolve the placement flag, preferring ``workplaceType``.

    ``isRemote`` conflates "remote" and "hybrid" into a single ``true``, so
    it is only consulted when ``workplaceType`` is absent — and even then
    only its ``false`` (definitively on-site) case is trusted. A bare
    ``isRemote: true`` stays ``None``: it means "not fully on-site", which
    is not the same claim as "remote".
    """
    workplace = item.get("workplaceType")
    if isinstance(workplace, str):
        normalized = workplace.strip().lower().replace("-", "").replace(" ", "")
        resolved = _WORKPLACE_REMOTE.get(normalized)
        if resolved is not None:
            return resolved
    if item.get("isRemote") is False:
        return False
    return None


class _Location(NamedTuple):
    display: str | None = None
    country_iso: str | None = None
    region: str | None = None
    secondary: tuple[str, ...] = ()


def _locations(item: dict[str, Any]) -> _Location:
    """Merge the primary and secondary locations into one row.

    Ashby splits a multi-location posting into ``location`` (the primary
    office) plus a ``secondaryLocations`` list, which leaves roles reading
    "SF Office" when they are equally open in "Remote (USA)". The canonical
    schema wants all of them comma-joined.

    ``country_iso`` / ``region`` are only set when every address that names
    a recognisable country names the *same* one — a posting split across
    London and New York has no single country.
    """
    entries: list[tuple[str | None, object]] = [
        (_clean(item.get("location")), item.get("address"))
    ]
    secondary: list[str] = []
    for loc in item.get("secondaryLocations") or []:
        if not isinstance(loc, dict):
            continue
        name = _clean(loc.get("location"))
        if name:
            secondary.append(name)
        entries.append((name, loc.get("address")))

    names = list(dict.fromkeys(name for name, _ in entries if name))
    countries = {
        meta for _, address in entries if (meta := _country_metadata(address)) is not None
    }
    country_iso, region = next(iter(countries)) if len(countries) == 1 else (None, None)
    return _Location(
        display=", ".join(names) or None,
        country_iso=country_iso,
        region=region,
        secondary=tuple(secondary),
    )


def _country_metadata(address: object) -> tuple[str, str] | None:
    """Resolve one Ashby address block to ``(country_iso, region)``.

    ``addressCountry`` is free text the employer typed, and a fair share of
    it ("Global", "European Union") names no single country — the shared
    resolver returns ``None`` for those.
    """
    if not isinstance(address, dict):
        return None
    postal = address.get("postalAddress")
    if not isinstance(postal, dict):
        return None
    country_iso, region = resolve_country(postal.get("addressCountry"))
    return (country_iso, region) if country_iso and region else None


class _SalaryRange(NamedTuple):
    currency: str | None = None
    period: SalaryPeriod | None = None
    minimum: float | None = None
    maximum: float | None = None


def _parse_comp(comp: dict[str, Any] | None) -> _SalaryRange:
    """Pull structured min/max/currency/period out of a compensation block.

    Ashby returns several component types (Salary, Bonus, Commission,
    Equity*); Salary is the one that becomes min/max. Field names live at
    the component level (``minValue``, ``maxValue``, ``currencyCode``) —
    not nested in ``compensationValue`` as some older docs suggest.

    ``summaryComponents`` is preferred over walking ``compensationTiers``
    because it is Ashby's own aggregate across every tier. A posting with
    per-zone bands ($207K–$285K for SF, down to $165.4K–$228K for Zone B)
    carries one tier each, and reading only the first publishes a floor
    $42K above the real one — while ``salary_summary``, taken from the tier
    summary, correctly reads "$165.4K – $285K". Spanning the tiers is kept
    as the fallback for payloads that omit the aggregate.
    """
    if not comp:
        return _SalaryRange()
    span = _salary_from_components(comp.get("summaryComponents"))
    if span.currency is not None:
        return span
    return _salary_from_components(
        [
            component
            for tier in comp.get("compensationTiers") or []
            if isinstance(tier, dict)
            for component in tier.get("components") or []
        ]
    )


def _salary_from_components(components: object) -> _SalaryRange:
    """Span the Salary components into a single range.

    Only components sharing the first one's currency are merged, so a USD
    band and a GBP band never combine into a nonsense range.
    """
    if not isinstance(components, list):
        return _SalaryRange()

    currency: str | None = None
    period: SalaryPeriod | None = None
    minimum: float | None = None
    maximum: float | None = None

    for component in components:
        if not isinstance(component, dict):
            continue
        if component.get("compensationType") != "Salary":
            continue
        entry_currency = component.get("currencyCode")
        if not isinstance(entry_currency, str) or len(entry_currency) != 3:
            continue
        entry_currency = entry_currency.upper()
        if currency is None:
            currency = entry_currency
            period = _interval(component.get("interval"))
        elif entry_currency != currency:
            continue
        low = _amount(component.get("minValue"))
        high = _amount(component.get("maxValue"))
        if low is not None and (minimum is None or low < minimum):
            minimum = low
        if high is not None and (maximum is None or high > maximum):
            maximum = high

    if currency is None or (minimum is None and maximum is None):
        return _SalaryRange()
    return _SalaryRange(currency=currency, period=period, minimum=minimum, maximum=maximum)


def _interval(value: object) -> SalaryPeriod | None:
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    # "1 YEAR" / "1 HOUR" — strip the multiplier when it is the identity.
    # A genuine multiple ("2 WEEK") is not expressible as a SalaryPeriod
    # and falls through to None.
    if token.startswith("1 "):
        token = token[2:].strip()
    return _INTERVAL_MAP.get(token)


def _amount(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # A zero bound means "not set on this side", not "unpaid".
    return float(value) if value > 0 else None


def _apply_url(apply_url: object, job_url: object) -> str | None:
    """Keep ``applyUrl`` only when it is a genuinely separate destination.

    Ashby derives it mechanically as ``{jobUrl}/application`` on every
    posting, which carries no information the posting URL doesn't.
    """
    if not isinstance(apply_url, str) or not apply_url:
        return None
    if not isinstance(job_url, str):
        return apply_url
    if apply_url in (job_url, f"{job_url}/application", f"{job_url.rstrip('/')}/application"):
        return None
    return apply_url


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _clean_description(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value[:_MAX_DESCRIPTION_CHARS]


def _parse_package(comp: dict[str, Any] | None) -> dict[str, bool | None]:
    """Which non-salary components the posting says it includes.

    ``_parse_comp`` deliberately reads only the Salary component, so
    everything else Ashby discloses — equity, bonus, commission — was
    reachable solely by digging through ``raw``. These are booleans, not
    amounts: the equity and bonus components carry null values without
    exception, and their summary is the bare string "Offers Equity".
    Absence is not a denial, so a missing component stays ``None``.
    """
    flags: dict[str, bool | None] = {
        "offers_equity": None,
        "offers_bonus": None,
        "offers_commission": None,
    }
    if not comp:
        return flags
    for tier in comp.get("compensationTiers") or []:
        for component in tier.get("components") or []:
            kind = str(component.get("compensationType") or "")
            if kind.startswith("Equity"):
                flags["offers_equity"] = True
            elif kind == "Bonus":
                flags["offers_bonus"] = True
            elif kind == "Commission":
                flags["offers_commission"] = True
    return flags


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

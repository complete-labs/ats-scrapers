"""Greenhouse scraper.

Greenhouse exposes a public JSON board at:
    https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true

The most permissive ATS API — no auth, no rate limits in practice. The
``content=true`` flag inflates the response with full job descriptions
(HTML-encoded entities — we decode + strip tags). First_published gives
the canonical posted-at; updated_at is the better choice for "when this
posting changed" but we surface first_published since it's stable.

The list response also carries ``departments`` (array of named groups)
and ``offices`` (locations the role is open in). Employment type is
NOT in the list response — Greenhouse doesn't expose it on the public
board API, only via the authenticated harvest API.

Compensation lives behind the ``pay_transparency=true`` flag as
``pay_input_ranges``: a list of ``{min_cents, max_cents, currency_type,
title, blurb}``, one entry per jurisdiction the posting is open in.
"""

from __future__ import annotations

import html as html_mod
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

from ats_scrapers.models import ATSType, Job, SalaryPeriod
from ats_scrapers.scrapers.base import BaseScraper, ScraperRegistry

if TYPE_CHECKING:
    from typing import Any

# ``content=true`` opts the API into returning the full HTML description
# in each job entry. The flag adds ~5x to the response size but saves
# us per-job detail fetches across ~3,000 boards.
#
# ``pay_transparency=true`` is what unlocks ``pay_input_ranges``. Without
# it the key is absent from every job — not null, absent — so the board
# looks like it has no compensation data at all. Roughly a quarter of
# postings board-wide (and 90%+ on boards subject to US pay-transparency
# law) carry a range once the flag is set.
API_TEMPLATE = (
    "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    "?content=true&pay_transparency=true"
)

_TAG_RE = re.compile(r"<[^>]+>")

# Greenhouse labels each range with a free-text heading ("Annual Salary:",
# "Hourly Rate:"). It's the only period signal on the payload, and plenty
# of employers use a heading that names no period at all ("Salary Range",
# "Pay Range", "New York Pay Band", "Echelle salariale").
#
# Word-anchored so ``"day"`` doesn't fire on ``"Monday"``.
_PERIOD_MARKERS: tuple[tuple[re.Pattern[str], SalaryPeriod], ...] = (
    (re.compile(r"\bhour|\bhr\b|/hr\b"), "HOUR"),
    (re.compile(r"\bdaily\b|\bday\b|\bper diem\b"), "DAY"),
    (re.compile(r"\bweek"), "WEEK"),
    (re.compile(r"\bmonth|\bmensuel"), "MONTH"),
    (re.compile(r"\bannual|\byear|\bannuel"), "YEAR"),
)

# When the heading names no period, the magnitude settles it. No currency
# on earth expresses an *annual* salary as a number below this — the
# weakest ones inflate the figure, they don't shrink it — so a bare "Pay
# Range" of 17.00–18.50 is an hourly rate, not someone's yearly income.
# Getting this wrong is not a rounding error: it publishes "$70 / year"
# for a telehealth contractor billing $70 / hour.
_MAX_PLAUSIBLE_HOURLY = 2_000.0


@ScraperRegistry.register(ATSType.GREENHOUSE)
class GreenhouseScraper(BaseScraper):
    ats = ATSType.GREENHOUSE

    async def afetch(self) -> list[Job]:
        url = API_TEMPLATE.format(slug=self.company_slug)
        async with self.make_fetcher() as fetch:
            payload = await fetch.get_json(url)
        return [self._parse_job(item) for item in payload.get("jobs", [])]

    def _parse_job(self, item: dict[str, Any]) -> Job:
        offices = item.get("offices") or []
        departments = item.get("departments") or []
        first_dept = next(
            (d.get("name") for d in departments if isinstance(d, dict) and d.get("name")),
            None,
        )
        metadata = item.get("metadata") or []
        # Greenhouse "metadata" is a list of {name, value, value_type} dicts —
        # custom fields the employer set. Capture verbatim in ``raw``.
        raw: dict[str, Any] = {}
        if metadata:
            raw["metadata"] = metadata
        if departments:
            raw["departments"] = [d.get("name") for d in departments if isinstance(d, dict)]
        if offices:
            raw["offices"] = [o.get("name") for o in offices if isinstance(o, dict)]
        if item.get("internal_job_id") is not None:
            raw["internal_job_id"] = item["internal_job_id"]

        # Greenhouse's ``content`` is HTML-encoded twice on the public
        # API (the entities are escaped, then the whole string wrapped):
        # ``&lt;h2&gt;`` etc. We unescape once, then strip tags to plain
        # text for storage.
        description = _clean_description(item.get("content"))

        # ``first_published`` is a stable creation timestamp (employer
        # set when the posting first went live). ``updated_at`` only
        # tells us when an internal field changed (often noise). Prefer
        # first_published for "posted_at" semantics.
        posted_at = _parse_iso(item.get("first_published")) or _parse_iso(
            item.get("updated_at")
        )

        # ``requisition_id`` is sometimes a placeholder ("See Opening
        # ID", "TBD"); only keep when it looks like a real identifier.
        req_raw = item.get("requisition_id")
        requisition_id: str | None = None
        if isinstance(req_raw, (str, int)):
            req_str = str(req_raw).strip()
            if req_str and req_str.lower() not in (
                "see opening id", "tbd", "n/a", "tba",
            ):
                requisition_id = req_str

        salary = _parse_pay_ranges(item.get("pay_input_ranges"))

        return Job(
            url=item["absolute_url"],
            title=item["title"],
            company=self.company_slug,
            ats_type=ATSType.GREENHOUSE,
            ats_id=str(item["id"]),
            location=(item.get("location") or {}).get("name"),
            department=first_dept,
            description=description,
            requisition_id=requisition_id,
            salary_currency=salary.currency,
            salary_period=salary.period,
            salary_min=salary.minimum,
            salary_max=salary.maximum,
            salary_summary=salary.summary,
            posted_at=posted_at,
            fetched_at=datetime.now(UTC),
            raw=raw or None,
        )


class _PayRange(NamedTuple):
    currency: str | None = None
    period: SalaryPeriod | None = None
    minimum: float | None = None
    maximum: float | None = None
    summary: str | None = None


def _parse_pay_ranges(value: object) -> _PayRange:
    """Collapse ``pay_input_ranges`` into a single range.

    A posting open in several pay-transparency jurisdictions carries one
    entry per jurisdiction. We span them — lowest min, highest max —
    rather than picking one arbitrarily, and only across entries sharing
    the first entry's currency so a USD range and a CAD range never get
    merged into a nonsense band.
    """
    if not isinstance(value, list) or not value:
        return _PayRange()

    currency: str | None = None
    period: SalaryPeriod | None = None
    minimum: float | None = None
    maximum: float | None = None

    for entry in value:
        if not isinstance(entry, dict):
            continue
        entry_currency = entry.get("currency_type")
        if not isinstance(entry_currency, str) or len(entry_currency) != 3:
            continue
        entry_currency = entry_currency.upper()
        if currency is None:
            currency = entry_currency
            period = _period_from_label(entry.get("title"))
        elif entry_currency != currency:
            continue
        low = _cents_to_major(entry.get("min_cents"))
        high = _cents_to_major(entry.get("max_cents"))
        if low is not None and (minimum is None or low < minimum):
            minimum = low
        if high is not None and (maximum is None or high > maximum):
            maximum = high

    if currency is None or (minimum is None and maximum is None):
        return _PayRange()
    return _PayRange(
        currency=currency,
        period=period or _period_from_magnitude(minimum, maximum),
        minimum=minimum,
        maximum=maximum,
        summary=_compose_summary(currency, minimum, maximum),
    )


def _period_from_label(label: object) -> SalaryPeriod | None:
    if not isinstance(label, str) or not label.strip():
        return None
    lowered = label.lower()
    for pattern, period in _PERIOD_MARKERS:
        if pattern.search(lowered):
            return period
    return None


def _period_from_magnitude(
    minimum: float | None, maximum: float | None
) -> SalaryPeriod:
    """Fall back to the amount when the label names no period.

    ``YEAR`` is the right default — it's what the overwhelming majority of
    unlabelled Greenhouse ranges mean — but only once the hourly rates
    masquerading as salaries have been separated out by size.
    """
    upper = maximum if maximum is not None else minimum
    if upper is not None and upper < _MAX_PLAUSIBLE_HOURLY:
        return "HOUR"
    return "YEAR"


def _cents_to_major(value: object) -> float | None:
    """Greenhouse reports whole cents as an int (or a numeric string)."""
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        cents = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # A zero bound means "not set on this side", not "unpaid".
    return cents / 100 if cents > 0 else None


def _compose_summary(
    currency: str, minimum: float | None, maximum: float | None
) -> str:
    if minimum is not None and maximum is not None:
        return f"{currency} {minimum:,.0f} - {maximum:,.0f}"
    if minimum is not None:
        return f"{currency} {minimum:,.0f}+"
    return f"{currency} up to {maximum:,.0f}"


def _clean_description(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    # Greenhouse double-encodes ``content``: ``<div>`` shows up as
    # ``&lt;div&gt;``. Unescape once to recover real HTML; then leave the
    # tags in place so the post-scrape markdownify step can preserve
    # paragraph breaks, bullet lists, and headings. The previous brutal
    # tag-strip collapsed the body into a single space-separated blob,
    # losing all visual structure.
    text = html_mod.unescape(value).strip()
    return text[:25_000] or None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

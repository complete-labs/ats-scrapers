"""US-location and pay-disclosure detection from free-text fields.

Shared by two callers that need the same judgements about a posting's
``location`` string.

**The publisher**, to decide ``country_iso``. Its own parser matches
country *names* ("united states", "usa") and NUTS prefixes but has no
notion of US states, which leaves two defects: "San Francisco, CA" and
"Austin, TX" resolve to nothing, and worse, a bare trailing state code
that happens to spell a country — ``CA`` (California/Canada), ``DE``
(Delaware/Germany) — resolves to the wrong country outright.
:func:`looks_us` and :data:`US_STATE_CODES` close both.

**The company-enrichment cohort**, to pick US pay-transparent employers.
The structured salary columns are empty for the largest US ATSes —
Greenhouse, Workday, SmartRecruiters, iCIMS, Oracle, and SuccessFactors
have *zero* rows with ``salary_min`` or ``salary_summary`` populated, yet
30% of Greenhouse descriptions carry an explicit pay range. Restricting
the cohort to structured salary would exclude nearly every large US tech
employer, so :data:`PAY_RANGE_PATTERN` recovers those postings from the
description body.

Patterns are written as RE2-compatible strings so the same source runs
in Python and inside DuckDB via :func:`us_sql_expr` / :func:`pay_sql_expr`.
"""

from __future__ import annotations

import re

# Postal abbreviations, matched only after a comma/slash so "OR"
# (Oregon) does not fire on the conjunction and "IN" (Indiana) does not
# fire on the preposition.
_STATE_ABBR = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|"
    "MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|"
    "WA|WV|WI|WY|DC|PR"
)

_STATE_NAMES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
    "florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|"
    "louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|"
    "missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|"
    "new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|"
    "rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|"
    "virginia|washington|west virginia|wisconsin|wyoming|"
    "district of columbia|puerto rico"
)

# The two lists above are maintained in the same order, so they zip
# straight into a lookup rather than needing a third copy of the states.
# ``strict`` makes any future drift a startup error instead of a silently
# shifted mapping.
STATE_NAME_TO_ABBR: dict[str, str] = dict(
    zip(_STATE_NAMES.split("|"), _STATE_ABBR.split("|"), strict=True)
)

#: Every US postal abbreviation, for callers that need to tell a state
#: code apart from an ISO 3166-1 country code of the same spelling.
US_STATE_CODES: frozenset[str] = frozenset(_STATE_ABBR.split("|"))

# "Georgia" is also a country and "Washington" collides with nothing
# harmful, but a bare state name with no other signal is weak evidence.
# These need a US-ish companion token to count.
_AMBIGUOUS_STATE_NAMES = frozenset({"georgia", "washington", "virginia"})

# Assembled as SQL-compatible RE2 strings so the same patterns run
# inside DuckDB (`regexp_matches`) and in Python.
US_COUNTRY_PATTERN = r"(?i)\b(?:united states(?: of america)?|u\.?s\.?a\.?)\b"
US_STATE_ABBR_PATTERN = rf"(?:^|[,/(\-]\s*)(?:{_STATE_ABBR})\b\s*(?:[,()\-]|$)"
US_STATE_NAME_PATTERN = rf"(?i)\b(?:{_STATE_NAMES})\b"
US_REMOTE_PATTERN = (
    r"(?i)\b(?:remote|anywhere|nationwide)\b[\s,\-–—]*\(?\b(?:us|usa|u\.s\.)\b"
    r"|\b(?:us|usa|u\.s\.)\b[\s,\-–—]*\b(?:remote|only|based|wide)\b"
)

# Other-country markers that veto a weak US signal. The publisher's
# parser already catches most of these, but it runs name-first and
# returns on the first hit, so a "Toronto, ON, Canada" string that also
# contains "ON" is safe while "Ontario, CA" (Ontario, California) is
# genuinely ambiguous and gets vetoed conservatively.
_NON_US_VETO = re.compile(
    r"(?i)\b(?:canada|mexico|united kingdom|england|scotland|wales|ireland|"
    r"australia|new zealand|india|singapore|japan|china|germany|france|spain|"
    r"italy|netherlands|belgium|sweden|norway|denmark|finland|poland|brazil|"
    r"argentina|colombia|chile|philippines|indonesia|vietnam|thailand|"
    r"south africa|nigeria|kenya|egypt|israel|turkey|portugal|greece|"
    r"switzerland|austria|czech|romania|hungary|ukraine|russia)\b"
)

# Explicit pay range: two currency amounts joined by a dash or "to".
# This is the shape US pay-transparency statutes produce
# ("$135,000 - $150,000", "$139,700—$188,900", "$55.00 to $65.00").
PAY_RANGE_PATTERN = (
    r"\$\s?[0-9][0-9,]{2,}(?:\.[0-9]{2})?\s*(?:-|–|—|to|and)\s*"
    r"\$?\s?[0-9][0-9,]{2,}(?:\.[0-9]{2})?"
)

# Single amount qualified by explicit compensation language, for
# postings that publish a point rather than a band.
PAY_PHRASE_PATTERN = (
    r"(?i)(?:base (?:pay|salary)|salary range|pay range|compensation range|"
    r"pay scale|hiring range|annual salary|hourly rate|base compensation)"
    r"[^.$]{0,80}\$\s?[0-9][0-9,]{2,}"
)

_US_COUNTRY_RE = re.compile(US_COUNTRY_PATTERN)
_US_STATE_ABBR_RE = re.compile(US_STATE_ABBR_PATTERN)
_US_STATE_NAME_RE = re.compile(US_STATE_NAME_PATTERN)
_US_REMOTE_RE = re.compile(US_REMOTE_PATTERN)
_PAY_RANGE_RE = re.compile(PAY_RANGE_PATTERN)
_PAY_PHRASE_RE = re.compile(PAY_PHRASE_PATTERN)


def looks_us(location: str | None, *, country_iso: str | None = None) -> bool:
    """True when a posting is plausibly located in the United States.

    ``country_iso`` short-circuits when the publisher already resolved a
    country: ``"US"`` accepts, any other non-empty code rejects. Only
    when it is empty do the location heuristics run.
    """
    if country_iso:
        return country_iso.strip().upper() == "US"
    if not location or not location.strip():
        return False
    text = location.strip()

    if _US_COUNTRY_RE.search(text):
        return True
    if _US_REMOTE_RE.search(text):
        return True
    if _NON_US_VETO.search(text):
        return False
    if _US_STATE_ABBR_RE.search(text):
        return True
    match = _US_STATE_NAME_RE.search(text)
    return bool(match) and match.group(0).lower() not in _AMBIGUOUS_STATE_NAMES


def has_pay(
    *,
    salary_min: str | None = None,
    salary_summary: str | None = None,
    description: str | None = None,
) -> bool:
    """True when a posting discloses compensation anywhere we can see it."""
    if salary_min and str(salary_min).strip():
        return True
    if salary_summary and str(salary_summary).strip():
        return True
    if description:
        text = str(description)
        if _PAY_RANGE_RE.search(text) or _PAY_PHRASE_RE.search(text):
            return True
    return False


def _sql_quote(pattern: str) -> str:
    return pattern.replace("'", "''")


def us_sql_expr(country_col: str = "country_iso", location_col: str = "location") -> str:
    """DuckDB boolean expression mirroring :func:`looks_us`."""
    loc = f"coalesce({location_col}, '')"
    return f"""(
        CASE
            WHEN {country_col} IS NOT NULL AND {country_col} <> ''
                THEN {country_col} = 'US'
            WHEN regexp_matches({loc}, '{_sql_quote(US_COUNTRY_PATTERN)}') THEN TRUE
            WHEN regexp_matches({loc}, '{_sql_quote(US_REMOTE_PATTERN)}') THEN TRUE
            WHEN regexp_matches({loc}, '{_sql_quote(_NON_US_VETO.pattern)}') THEN FALSE
            WHEN regexp_matches({loc}, '{_sql_quote(US_STATE_ABBR_PATTERN)}') THEN TRUE
            WHEN regexp_matches({loc}, '{_sql_quote(US_STATE_NAME_PATTERN)}')
                AND NOT regexp_matches({loc}, '(?i)\\b(?:georgia|washington|virginia)\\b')
                THEN TRUE
            ELSE FALSE
        END
    )"""


def pay_sql_expr(
    salary_min_col: str = "salary_min",
    salary_summary_col: str = "salary_summary",
    description_col: str = "description",
) -> str:
    """DuckDB boolean expression mirroring :func:`has_pay`."""
    return f"""(
        ({salary_min_col} IS NOT NULL AND {salary_min_col} <> '')
        OR ({salary_summary_col} IS NOT NULL AND {salary_summary_col} <> '')
        OR ({description_col} IS NOT NULL AND (
            regexp_matches({description_col}, '{_sql_quote(PAY_RANGE_PATTERN)}')
            OR regexp_matches({description_col}, '{_sql_quote(PAY_PHRASE_PATTERN)}')
        ))
    )"""

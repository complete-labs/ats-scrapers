"""Derive enrichment columns from existing fields.

Pure functions, cheap to run, no network. Deliberately narrow:

- We only return ``True`` for ``is_remote`` when the **title** carries
  an unambiguous remote marker. The absence of such a marker is *not*
  evidence the role is on-site, so we return ``None`` (unknown) rather
  than ``False``.
- We do not parse the description for remote signals — phrasing there
  is too ambiguous for a hardcoded rule. LLM-based enrichment
  downstream fills the rest.
- Salary range parsing on free text is preserved (see
  :func:`parse_salary_range`) — the regex is tight and the input is
  conventionally structured.
"""

from __future__ import annotations

import html
import re
from typing import NamedTuple

# Markers we treat as definitive when they appear in a title.
# Conservative on purpose — "Remote Engineer" is unambiguous; titles
# like "Remote Sales Director" also fire True and the role really is
# remote in those cases.
#
# ``distributed`` is intentionally excluded: titles like
# "Distributed Systems Engineer" / "Senior Engineer, Distributed
# Storage" use the word as a technical-domain qualifier (distributed
# computing) rather than a workforce-placement signal. The downstream
# LLM enrichment pipeline can still classify such roles as remote
# from the full posting context if they actually are.
REMOTE_KEYWORDS = (
    "remote",
    "anywhere",
    "work from home",
    "wfh",
    "telework",
)


def infer_is_remote(title: object) -> bool | None:
    """Return ``True`` when the title contains a remote marker, else
    ``None``.

    Never returns ``False`` — the absence of "remote" in the title is
    not evidence the role is on-site. Many remote roles have plain
    titles like "Senior Engineer". LLM-based enrichment downstream is
    expected to fill ``True`` / ``False`` from the full posting
    context.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    if any(kw in title.lower() for kw in REMOTE_KEYWORDS):
        return True
    return None


# --- Seniority --------------------------------------------------------------


class SeniorityRule(NamedTuple):
    """One level of the ladder, plus the phrases that fake it.

    ``exclude`` is a second pattern rather than a negative lookaround
    because these run in polars at publish time, and its Rust regex
    engine has no lookahead or lookbehind.
    """

    level: str
    pattern: str
    exclude: str = ""


# Ordered by precedence, first match wins. Management titles are
# checked before the IC ladder so "Senior Director" is DIRECTOR rather
# than SENIOR, and INTERN outranks everything because an internship in
# a given function is still an internship.
#
# Every rule is deliberately biased toward returning nothing. A title
# is a few words with no shared vocabulary across industries, and a
# wrong level is worse than a missing one: it silently moves a posting
# into the wrong pay band, where nobody can see it happened. The
# ``exclude`` patterns are the false friends found in the corpus —
# mostly words that name a *function* ("Product Manager") or an
# *audience* ("Senior Living Nurse") rather than a rank.
SENIORITY_RULES: tuple[SeniorityRule, ...] = (
    SeniorityRule(
        "INTERN",
        r"(?i)\b(intern|interns|internship|trainee|apprentice\w*|"
        r"working student|werkstudent|praktikum|praktikant\w*|stagiaire|"
        r"becari[oa]|co-?op student|placement student)\b",
    ),
    SeniorityRule(
        "EXECUTIVE",
        r"(?i)(\b(ceo|cto|cfo|coo|cio|ciso|cmo|cpo|chro|cro|cdo|cco)\b|"
        r"\bchief\b|\b(vp|svp|evp|avp)\b|\bvice[- ]president\b|"
        r"\bpresident\b|\bmanaging director\b)",
        # "Chief of Staff" is a coordinator role, not the C-suite.
        r"(?i)\bchief of staff\b",
    ),
    SeniorityRule(
        "DIRECTOR",
        r"(?i)(\bdirectors?\b|\bhead of\b)",
        # Craft titles where "director" names the discipline.
        r"(?i)(\b(art|artistic|creative|casting|music|photography|funeral|"
        r"athletic|choir|band|stage|film|video|technical) directors?\b|"
        r"\bdirector of photography\b)",
    ),
    SeniorityRule(
        "MANAGER",
        r"(?i)\bmanagers?\b",
        # "X Manager" where X is the thing managed, not people. These
        # are individual contributors at every level of the ladder.
        r"(?i)\b(product|program|programme|project|account|community|"
        r"brand|category|content|social media|customer success|partner|"
        r"partnerships|case|property|fund|portfolio|asset|wealth|"
        r"relationship|territory|traffic|inventory|contract|vendor|"
        r"campaign|channel|product marketing|technical program|"
        r"technical product) managers?\b",
    ),
    SeniorityRule(
        "PRINCIPAL",
        r"(?i)\bprincipals?\b",
        # School principals and the research-grant sense of the word.
        r"(?i)(\b(school|assistant|vice|deputy|associate) principal\b|"
        r"\bprincipal (investigator|of|at)\b|\bprincipal'?s\b)",
    ),
    SeniorityRule(
        "STAFF",
        # Only a rank in the tech IC ladder. "Staff Nurse" is an
        # entry-grade nurse and "Staff Accountant" is a junior
        # accountant, so the word alone proves nothing.
        r"(?i)\bstaff\b.*\b(engineer|developer|scientist|designer|"
        r"researcher|architect|programmer|technologist|sre)s?\b",
    ),
    SeniorityRule(
        "LEAD",
        r"(?i)\b(lead|leads)\b",
        # Sales pipeline vocabulary, not a rank.
        r"(?i)\blead[- ]?(generation|gen|qualification|nurturing)\b",
    ),
    SeniorityRule(
        "SENIOR",
        r"(?i)(\bsenior\b|\bsr\.?\b|\bsnr\.?\b)",
        # Elder-care roles describe their clients, not the hire.
        r"(?i)\bsenior (living|care|citizens?|center|centre|community|"
        r"housing|services|advisor|advocate|companion|meals)\b",
    ),
    SeniorityRule("MID", r"(?i)\b(mid[- ]level|intermediate)\b"),
    SeniorityRule(
        "JUNIOR",
        r"(?i)(\bjunior\b|\bjr\.?\b|\bjnr\.?\b|\bentry[- ]level\b|"
        r"\bnew grad(uate)?\b|\bgraduate\b)",
        # School levels, and graduate programmes that name a field.
        r"(?i)(\bjunior (high|college|varsity|school)\b|"
        r"\bgraduate (school|admissions|studies|program|programme)\b)",
    ),
)

SENIORITY_LEVELS: tuple[str, ...] = tuple(rule.level for rule in SENIORITY_RULES)

_SENIORITY_COMPILED = tuple(
    (rule.level, re.compile(rule.pattern), re.compile(rule.exclude) if rule.exclude else None)
    for rule in SENIORITY_RULES
)


def infer_seniority(title: object) -> str | None:
    """Return the seniority level named by the title, else ``None``.

    Reads only the title, and only its explicit rank words. A plain
    "Software Engineer" returns ``None`` rather than ``MID``: most
    employers omit the qualifier at their baseline level, but which
    level that is varies by employer, so inferring it would invent a
    fact. As with :func:`infer_is_remote`, unknown is a real answer and
    the LLM pass downstream is expected to narrow it.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    for level, pattern, exclude in _SENIORITY_COMPILED:
        if pattern.search(title) and not (exclude and exclude.search(title)):
            return level
    return None


# --- Salary parsing ---------------------------------------------------------

_SALARY_RANGE_RE = re.compile(
    r"""
    (?P<sym1>[$£€¥]|CA\$|US\$|A\$|NZ\$|HK\$|S\$|R\$)?\s*
    (?P<n1>\d[\d,. ]*)\s*
    (?P<u1>[KMkm]|thousand|million)?
    \s*(?:[-–—~]|to)\s*
    (?P<sym2>[$£€¥]|CA\$|US\$|A\$|NZ\$|HK\$|S\$|R\$)?\s*
    (?P<n2>\d[\d,. ]*)\s*
    (?P<u2>[KMkm]|thousand|million)?
    """,
    re.VERBOSE,
)
_SALARY_SINGLE_RE = re.compile(
    r"""
    (?P<sym>[$£€¥]|CA\$|US\$|A\$)?\s*
    (?P<n>\d[\d,. ]{2,})\s*
    (?P<u>[KMkm]|thousand|million)?
    """,
    re.VERBOSE,
)


def _parse_salary_token(num: str, unit: str | None) -> float | None:
    """Convert a number token + optional unit suffix to a float amount."""
    cleaned = num.replace(",", "").replace(" ", "").rstrip(".")
    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "", cleaned.count(".") - 1)
    try:
        value = float(cleaned)
    except ValueError:
        return None
    multiplier = 1.0
    if unit:
        u = unit.lower()
        if u.startswith("k") or u == "thousand":
            multiplier = 1_000
        elif u.startswith("m") or u == "million":
            multiplier = 1_000_000
    return value * multiplier


def parse_salary_range(text: object) -> tuple[float | None, float | None]:
    """Extract `(min, max)` from a salary summary string.

    Handles `$257K - $335K`, `CA$400K – CA$500K`, `60,000 - 80,000`,
    `€80k–€120k`, etc. Returns (None, None) when nothing parseable.
    """
    if not isinstance(text, str) or not text.strip():
        return (None, None)
    match = _SALARY_RANGE_RE.search(text)
    if match:
        lo = _parse_salary_token(match.group("n1"), match.group("u1"))
        hi = _parse_salary_token(match.group("n2"), match.group("u2"))
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    match = _SALARY_SINGLE_RE.search(text)
    if match:
        value = _parse_salary_token(match.group("n"), match.group("u"))
        return (value, value)
    return (None, None)


# --- Disclosed pay blocks in description text -------------------------------
#
# Pay-transparency statutes made employers publish a compensation range
# in the posting body, and the ATSes render it in a fixed shape:
#
#     Annual Salary: $320,000 — $405,000 USD
#     Pay Range $182,000 — $250,208 USD
#     Salary Range: €200.000 — €255.000 EUR
#
# Greenhouse alone carries this on thousands of boards while exposing no
# structured salary field at all, which is most of why only 2.31% of the
# published corpus showed any pay. The trailing ISO code is what makes
# the shape safe to match: it is the ATS's own render, not prose, so it
# does not collide with dollar amounts quoted in the body copy.


class SalaryBlock(NamedTuple):
    """A pay range recovered from description text."""

    min_amount: float
    max_amount: float
    currency: str
    period: str


_CURRENCY_BY_SYMBOL = {
    "$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR",
    "C$": "CAD", "CA$": "CAD", "A$": "AUD", "NZ$": "NZD",
    "S$": "SGD", "HK$": "HKD", "R$": "BRL", "US$": "USD",
}

# Currencies the boards we scrape actually quote pay in. Kept as an
# allow-list so a stray three-letter word after a range can't be
# mistaken for the unit.
_ISO_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "CAD", "AUD", "NZD", "CHF", "SEK", "NOK", "DKK",
    "PLN", "CZK", "HUF", "RON", "BGN", "ISK", "JPY", "CNY", "HKD", "SGD",
    "INR", "KRW", "TWD", "THB", "MYR", "IDR", "PHP", "VND", "ILS", "AED",
    "SAR", "TRY", "ZAR", "NGN", "KES", "EGP", "BRL", "MXN", "ARS", "CLP",
    "COP", "PEN", "UYU", "UAH", "RSD", "MAD",
})

_SYMBOLS = r"US\$|CA\$|NZ\$|HK\$|C\$|A\$|S\$|R\$|[$£€¥₹]"
_AMOUNT = r"\d[\d.,\s]*\d|\d"

_SALARY_BLOCK_RE = re.compile(
    rf"""
    (?P<sym1>{_SYMBOLS})\s?
    (?P<n1>{_AMOUNT})
    \s*(?:[-–—~]|to)\s*
    (?:{_SYMBOLS})?\s?
    (?P<n2>{_AMOUNT})
    \s*(?P<code>[A-Z]{{3}})\b
    """,
    re.VERBOSE,
)

# Employers that don't use the ATS's pay widget write the same
# disclosure as prose — "The salary range for this role is $160,000 -
# $240,000" — with no trailing currency code to anchor on. A label is
# required immediately before the figure, because without one every
# referral bonus and revenue number in the body would qualify. Only
# consulted when the coded form is absent.
_PAY_LABEL = (
    r"(?:base\s+(?:pay|salary)|salary\s+range|pay\s+range|compensation\s+range"
    r"|pay\s+scale|hiring\s+range|annual\s+salary|hourly\s+rate"
    r"|target\s+earnings|salary|compensation)"
)
_SALARY_LABELLED_RE = re.compile(
    rf"""
    {_PAY_LABEL}
    [^.!?]{{0,70}}?
    (?P<sym1>{_SYMBOLS})\s?
    (?P<n1>{_AMOUNT})
    (?:\s*(?:[-–—~]|to)\s*(?:{_SYMBOLS})?\s?(?P<n2>{_AMOUNT}))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Period wording, searched in the text immediately before a match.
_PERIOD_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HOUR", ("per hour", "hourly", "an hour", "/hour", "/hr", "per hr")),
    ("DAY", ("per day", "daily", "a day", "/day", "per diem")),
    ("WEEK", ("per week", "weekly", "a week", "/week")),
    ("MONTH", ("per month", "monthly", "a month", "/month", "per mo")),
    ("YEAR", ("per year", "annual", "annually", "yearly", "a year", "/year", "/yr")),
)

# How far back to read for period wording. Long enough to catch the
# label that introduces the range ("Annual Salary:"), short enough that
# an unrelated "hourly" earlier in the posting can't reach it.
_PERIOD_LOOKBEHIND = 80

# Below this, an unlabelled figure is not credibly an annual salary, so
# no period is assumed rather than guessed. This is the check whose
# absence let a "3,800 per hour" range annualize into millions.
_MIN_CREDIBLE_ANNUAL = 10_000

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _parse_amount(token: str) -> float | None:
    """Read a localized number, resolving separator ambiguity.

    ``,`` and ``.`` swap roles between locales, and Greenhouse renders
    the employer's own formatting: ``$320,000`` and ``€200.000`` are
    both 'two hundred thousand'-scale figures. Treating a lone dot as a
    decimal point turned ``€200.000`` into €200.
    """
    cleaned = token.replace(" ", "").replace("\u00a0", "")
    if not cleaned:
        return None
    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")
    if last_comma >= 0 and last_dot >= 0:
        # Both present: whichever comes last is the decimal separator.
        if last_dot > last_comma:
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif last_comma >= 0 or last_dot >= 0:
        sep = "," if last_comma >= 0 else "."
        tail = len(cleaned) - cleaned.rfind(sep) - 1
        # Exactly three trailing digits, or more than one separator, is
        # thousands grouping; anything else is a genuine decimal.
        if tail == 3 or cleaned.count(sep) > 1:
            cleaned = cleaned.replace(sep, "")
        else:
            cleaned = cleaned.replace(sep, ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _period_before(text: str, end: int) -> str | None:
    """Period implied by the wording just before a matched range."""
    window = text[max(0, end - _PERIOD_LOOKBEHIND):end].lower()
    best: tuple[int, str] | None = None
    for period, markers in _PERIOD_MARKERS:
        for marker in markers:
            at = window.rfind(marker)
            if at >= 0 and (best is None or at > best[0]):
                best = (at, period)
    return best[1] if best else None


def _to_plain_text(text: str) -> str:
    """Flatten markup so a range split across tags reads as one string.

    The block is rendered as separate elements —
    ``<span>$320,000</span><span class="divider">&mdash;</span>`` — so
    matching the raw body would never see the dash between the amounts.
    Entities are unescaped repeatedly because several ATSes double-encode
    their description payload.
    """
    for _ in range(2):
        if "&" not in text:
            break
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return _WHITESPACE_RE.sub(" ", _HTML_TAG_RE.sub(" ", text))


def _blocks_from(text: str, pattern: re.Pattern[str]) -> SalaryBlock | None:
    """Widest range ``pattern`` finds in ``text``, or ``None``."""
    best: SalaryBlock | None = None
    for match in pattern.finditer(text):
        low = _parse_amount(match.group("n1"))
        groups = match.groupdict()
        # The labelled form allows a single amount: a disclosed point
        # value is a range whose ends coincide.
        raw_high = groups.get("n2")
        high = _parse_amount(raw_high) if raw_high else low
        if low is None or high is None or low <= 0 or high <= 0:
            continue
        if low > high:
            low, high = high, low

        symbol_currency = _CURRENCY_BY_SYMBOL.get(match.group("sym1"))
        if symbol_currency is None:
            continue
        # The trailing code is the ATS's own render and outranks the
        # symbol (``CA$120,000 — CA$150,000 CAD``), but only when it is
        # actually a currency: three capitals also spell "OTE" and
        # "USA", which would otherwise be read as the unit.
        code = (groups.get("code") or "").upper()
        currency = code if code in _ISO_CURRENCY_CODES else symbol_currency

        period = _period_before(text, match.start("sym1"))
        if period is None:
            if low < _MIN_CREDIBLE_ANNUAL:
                continue
            period = "YEAR"

        if best is None:
            best = SalaryBlock(low, high, currency, period)
        elif best.currency == currency and best.period == period:
            best = SalaryBlock(
                min(best.min_amount, low), max(best.max_amount, high),
                currency, period,
            )
    return best


def parse_salary_block(text: object) -> SalaryBlock | None:
    """Recover a disclosed pay range from description text.

    Accepts HTML or plain text. Two shapes are recognised, in order of
    trust: the ATS's own rendered widget, which closes the range with an
    ISO currency code, and a prose disclosure introduced by an explicit
    pay label. When a posting lists several ranges (per-zone pay bands
    are common), the result spans them all, because the advertised
    minimum and maximum are what the employer committed to.

    The period comes from the wording introducing the range. When
    nothing says, an annual period is assumed only for figures large
    enough to be annual; a bare "$25.00 — $32.00" yields ``None`` rather
    than a guess, since guessing the period is what turns an hourly rate
    into a multimillion-dollar salary downstream.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    plain = _to_plain_text(text)
    return _blocks_from(plain, _SALARY_BLOCK_RE) or _blocks_from(
        plain, _SALARY_LABELLED_RE
    )

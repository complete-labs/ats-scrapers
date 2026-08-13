"""Tier 0: everything resolvable without a model.

Measured against the full snapshot (see ``data/reports/gap_profile.md``),
these are the fields and how much of the corpus is missing them:

    region     91.9%      lat/lon    99.8%      language   94.7%
    country    33.2%      is_remote  85.2%      salary     96.3%
    employment_type 45.5%            department 68.4%

The first four columns of that list are *entirely* solvable offline — they
are geography and script, not judgement — and they account for the largest
gaps in the corpus. Doing them here rather than in a prompt removes about
4.5 million rows' worth of model output from the bill and makes them
exactly reproducible.

What Tier 0 deliberately does **not** do is guess at prose. Upstream's
``infer_is_remote`` established the right instinct — return ``None``
rather than a coin flip — and this module follows it: every extractor here
either finds an unambiguous signal or reports the field unresolved, which
is what routes the row to Tier 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ats_scrapers.enrichment.derived import parse_salary_range
from enrich.geo import resolve_location
from enrich.schema import EmploymentType as EmploymentTypeT
from enrich.schema import Placement
from enrich.schema import SalaryPeriod as SalaryPeriodT

# --- language ---------------------------------------------------------------

# The corpus languages, taken from the provider mix: eures and
# bundesagentur (de), welcometothejungle and lever FR (fr), infojobs_es
# (es), gupy and programathor (pt), jobs_cz (cs), herp/hrmos (ja),
# beisen/moka (zh), plus the usual European set. Restricting the set is
# what keeps lingua fast and its memory bounded; an unrestricted detector
# loads every model it ships.
_LANGUAGE_NAMES = (
    "ENGLISH",
    "GERMAN",
    "FRENCH",
    "SPANISH",
    "PORTUGUESE",
    "ITALIAN",
    "DUTCH",
    "SWEDISH",
    "NORWEGIAN",
    "DANISH",
    "FINNISH",
    "POLISH",
    "CZECH",
    "SLOVAK",
    "HUNGARIAN",
    "ROMANIAN",
    "GREEK",
    "TURKISH",
    "RUSSIAN",
    "UKRAINIAN",
    "JAPANESE",
    "CHINESE",
    "KOREAN",
    "ARABIC",
    "HEBREW",
    "HINDI",
    "INDONESIAN",
    "VIETNAMESE",
    "THAI",
)

# Scripts that identify a language on sight. Checking these first skips the
# statistical detector for CJK/Cyrillic/etc. entirely, which is both faster
# and more reliable than n-gram models on short text.
_SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ja", re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")),  # kana is decisive
    ("ko", re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")),
    ("th", re.compile(r"[\u0e00-\u0e7f]")),
    ("he", re.compile(r"[\u0590-\u05ff]")),
    ("ar", re.compile(r"[\u0600-\u06ff]")),
    ("el", re.compile(r"[\u0370-\u03ff]")),
    ("hi", re.compile(r"[\u0900-\u097f]")),
)
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


@lru_cache(maxsize=1)
def _detector() -> Any:
    """Build the lingua detector once per process.

    Low-accuracy mode is the right trade here: it cuts memory roughly
    fivefold and is materially faster, and our inputs are titles plus a
    few hundred characters of description — long enough that the accuracy
    loss is small, and the field is advisory rather than load-bearing.
    """
    from lingua import Language, LanguageDetectorBuilder

    languages = [getattr(Language, name) for name in _LANGUAGE_NAMES if hasattr(Language, name)]
    return (
        LanguageDetectorBuilder.from_languages(*languages)
        .with_low_accuracy_mode()
        .with_preloaded_language_models()
        .build()
    )


#: How much text the language detector reads. Detection accuracy plateaus
#: well before this; reading the whole 6 kB p90 description would multiply
#: the cost of the pass for no gain.
LANGUAGE_SAMPLE_CHARS = 400


def detect_language(title: object, description: object = None) -> str | None:
    """Return an ISO 639-1 code, or ``None`` when there is too little text.

    Script detection runs first and short-circuits: kana settles Japanese,
    Hangul settles Korean. Han characters alone are ambiguous between
    Chinese and Japanese, so those fall through to the statistical
    detector, which distinguishes them from context.
    """
    parts = [part for part in (title, description) if isinstance(part, str) and part.strip()]
    if not parts:
        return None
    sample = " ".join(parts)[:LANGUAGE_SAMPLE_CHARS].strip()
    if len(sample) < 8:
        return None

    for code, pattern in _SCRIPT_RANGES:
        if pattern.search(sample):
            return code
    if _HAN_RE.search(sample):
        # Han without kana: Chinese unless the detector says Japanese.
        detected = _detector().detect_language_of(sample)
        if detected is not None and detected.iso_code_639_1.name.lower() == "ja":
            return "ja"
        return "zh"

    detected = _detector().detect_language_of(sample)
    if detected is None:
        return None
    return detected.iso_code_639_1.name.lower()


# --- employment type --------------------------------------------------------

# Ordered longest-key-first at match time so "part-time" is not shadowed by
# "time" style substrings, and so "not full-time" style phrases do not
# accidentally match the shorter token first.
_EMPLOYMENT_PATTERNS: tuple[tuple[str, EmploymentTypeT], ...] = (
    # English
    ("full-time", "FULL_TIME"),
    ("full time", "FULL_TIME"),
    ("fulltime", "FULL_TIME"),
    ("part-time", "PART_TIME"),
    ("part time", "PART_TIME"),
    ("parttime", "PART_TIME"),
    ("permanent", "FULL_TIME"),
    ("regular", "FULL_TIME"),
    ("internship", "INTERN"),
    ("intern", "INTERN"),
    ("apprentice", "INTERN"),
    ("trainee", "INTERN"),
    ("working student", "PART_TIME"),
    ("contractor", "CONTRACT"),
    ("contract", "CONTRACT"),
    ("freelance", "CONTRACT"),
    ("consultant", "CONTRACT"),
    ("temporary", "TEMPORARY"),
    ("temp", "TEMPORARY"),
    ("seasonal", "TEMPORARY"),
    ("casual", "PART_TIME"),
    ("fixed term", "TEMPORARY"),
    ("fixed-term", "TEMPORARY"),
    # French
    ("cdi", "FULL_TIME"),
    ("cdd", "TEMPORARY"),
    ("stage", "INTERN"),
    ("stagiaire", "INTERN"),
    ("alternance", "INTERN"),
    ("apprentissage", "INTERN"),
    ("temps plein", "FULL_TIME"),
    ("temps partiel", "PART_TIME"),
    ("interim", "TEMPORARY"),
    ("intérim", "TEMPORARY"),
    # German
    ("vollzeit", "FULL_TIME"),
    ("teilzeit", "PART_TIME"),
    ("praktikum", "INTERN"),
    ("praktikant", "INTERN"),
    ("ausbildung", "INTERN"),
    ("werkstudent", "PART_TIME"),
    ("befristet", "TEMPORARY"),
    ("unbefristet", "FULL_TIME"),
    ("aushilfe", "PART_TIME"),
    ("minijob", "PART_TIME"),
    ("freier mitarbeiter", "CONTRACT"),
    # Spanish / Portuguese
    ("jornada completa", "FULL_TIME"),
    ("tiempo completo", "FULL_TIME"),
    ("tempo integral", "FULL_TIME"),
    ("media jornada", "PART_TIME"),
    ("tiempo parcial", "PART_TIME"),
    ("meio periodo", "PART_TIME"),
    ("meio período", "PART_TIME"),
    ("efetivo", "FULL_TIME"),
    ("estagio", "INTERN"),
    ("estágio", "INTERN"),
    ("becario", "INTERN"),
    ("practicas", "INTERN"),
    ("prácticas", "INTERN"),
    ("aprendiz", "INTERN"),
    ("temporal", "TEMPORARY"),
    ("autonomo", "CONTRACT"),
    ("autônomo", "CONTRACT"),
    # Nordic / Dutch / Italian / Polish / Czech
    ("heltid", "FULL_TIME"),
    ("deltid", "PART_TIME"),
    ("praktik", "INTERN"),
    ("voltijd", "FULL_TIME"),
    ("deeltijd", "PART_TIME"),
    ("stagiair", "INTERN"),
    ("tempo pieno", "FULL_TIME"),
    ("tempo determinato", "TEMPORARY"),
    ("tempo indeterminato", "FULL_TIME"),
    ("tirocinio", "INTERN"),
    ("pelny etat", "FULL_TIME"),
    ("pełny etat", "FULL_TIME"),
    ("staz", "INTERN"),
    ("staż", "INTERN"),
    ("plny uvazek", "FULL_TIME"),
    ("plný úvazek", "FULL_TIME"),
    # Japanese / Chinese / Korean
    ("正社員", "FULL_TIME"),
    ("契約社員", "CONTRACT"),
    ("派遣", "TEMPORARY"),
    ("アルバイト", "PART_TIME"),
    ("パート", "PART_TIME"),
    ("インターン", "INTERN"),
    ("業務委託", "CONTRACT"),
    ("新卒", "FULL_TIME"),
    ("全职", "FULL_TIME"),
    ("兼职", "PART_TIME"),
    ("实习", "INTERN"),
    ("實習", "INTERN"),
    ("合同", "CONTRACT"),
    ("外包", "CONTRACT"),
    ("정규직", "FULL_TIME"),
    ("계약직", "CONTRACT"),
    ("인턴", "INTERN"),
)

_EMPLOYMENT_SORTED = tuple(
    sorted(_EMPLOYMENT_PATTERNS, key=lambda pair: len(pair[0]), reverse=True)
)


def map_employment_type(*values: object) -> EmploymentTypeT | None:
    """Map a provider's raw commitment label to the normalized enum.

    Reads ``commitment`` (the free-text label the upstream schema keeps
    verbatim precisely so this mapping is possible) and the title. Returns
    ``None`` rather than defaulting to ``FULL_TIME``: 45% of the corpus has
    no employment type, and inventing the modal value would make the
    column useless for filtering while looking complete.
    """
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        lowered = value.casefold()
        for needle, mapped in _EMPLOYMENT_SORTED:
            if needle in lowered:
                return mapped
    return None


# --- placement --------------------------------------------------------------

_REMOTE_TOKENS = (
    "remote",
    "anywhere",
    "work from home",
    "wfh",
    "telework",
    "telearbeit",
    "télétravail",
    "teletrabajo",
    "teletrabalho",
    "remoto",
    "フルリモート",
    "リモート",
    "在家办公",
    "远程",
)
_HYBRID_TOKENS = ("hybrid", "hybride", "hibrido", "híbrido", "ibrido", "ハイブリッド", "混合办公")
_ONSITE_TOKENS = (
    "on-site",
    "on site",
    "onsite",
    "in-office",
    "in office",
    "vor ort",
    "presencial",
    "sur site",
    "prasenz",
    "präsenz",
    "出社",
    "现场",
)


def infer_placement(
    title: object,
    location: object = None,
    commitment: object = None,
) -> Placement | None:
    """Three-way placement from short structured fields only.

    Hybrid is checked before remote because "Hybrid Remote - London" is a
    common provider phrasing that means hybrid, and a naive remote-first
    check would mislabel every one of those rows. The description is *not*
    read here — that is Tier 1's job, and the whole reason placement is
    the field most worth paying a model for.
    """
    haystack = " ".join(
        value.casefold() for value in (title, location, commitment) if isinstance(value, str)
    )
    if not haystack.strip():
        return None
    if any(token in haystack for token in _HYBRID_TOKENS):
        return "hybrid"
    if any(token in haystack for token in _REMOTE_TOKENS):
        return "remote"
    if any(token in haystack for token in _ONSITE_TOKENS):
        return "onsite"
    return None


# --- experience -------------------------------------------------------------

# Requires an experience word near the number. "5 years" alone appears in
# "5 years of company growth" and in tenure-of-product sentences; without
# the proximity constraint the extractor reads those as requirements.
_EXPERIENCE_WORDS = (
    r"experience|experiencia|experiência|expérience|erfahrung|berufserfahrung|"
    r"esperienza|ervaring|erfarenhet|doswiadczen\w*|zkusenost\w*|経験|经验|経験年数"
)
_YEAR_WORDS = r"years?|yrs?|ans?|années?|jahre?n?|anos?|años?|anni|jaar|år|lat|let|年"

# ``(?<!\d)``/``(?!\d)`` are load-bearing: without them ``\d{1,2}`` happily
# matches the "20" inside "120 years", turning an implausible number into a
# plausible-looking requirement instead of rejecting it.
_YEAR_NUM = r"(?<!\d)(\d{1,2})(?!\d)"

_RANGE_RE = re.compile(
    rf"{_YEAR_NUM}\s*(?:-|–|—|to|bis|a|à|até|do)\s*{_YEAR_NUM}\s*\+?\s*(?:{_YEAR_WORDS})",
    re.IGNORECASE,
)
_MIN_RE = re.compile(
    rf"(?:at\s+least|minimum\s+of|min\.?|minimum|mindestens|au\s+moins|al\s+menos|"
    rf"no\s+m[ií]nimo|pelo\s+menos|almeno)\s*{_YEAR_NUM}\s*\+?\s*(?:{_YEAR_WORDS})",
    re.IGNORECASE,
)
_PLUS_RE = re.compile(rf"{_YEAR_NUM}\s*\+\s*(?:{_YEAR_WORDS})", re.IGNORECASE)
_PLAIN_RE = re.compile(rf"{_YEAR_NUM}\s*(?:{_YEAR_WORDS})", re.IGNORECASE)
_EXPERIENCE_NEAR_RE = re.compile(_EXPERIENCE_WORDS, re.IGNORECASE)

#: Characters either side of a year match within which an experience word
#: must appear for the match to count.
_PROXIMITY = 60


def _has_experience_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - _PROXIMITY) : end + _PROXIMITY]
    return bool(_EXPERIENCE_NEAR_RE.search(window))


def parse_experience_years(text: object) -> tuple[int | None, int | None]:
    """Extract ``(min_years, max_years)`` from prose.

    Only accepts a match that sits near an experience word, and caps at 50
    years to reject phone numbers, founding dates and "10 000 customers"
    style noise that happens to precede a year word.
    """
    if not isinstance(text, str) or not text.strip():
        return (None, None)
    haystack = text[:8000]

    for match in _RANGE_RE.finditer(haystack):
        if not _has_experience_context(haystack, match.start(), match.end()):
            continue
        low, high = int(match.group(1)), int(match.group(2))
        if 0 <= low <= high <= 50:
            return (low, high)

    for pattern in (_MIN_RE, _PLUS_RE, _PLAIN_RE):
        for match in pattern.finditer(haystack):
            if not _has_experience_context(haystack, match.start(), match.end()):
                continue
            value = int(match.group(1))
            if 0 <= value <= 50:
                return (value, None)
    return (None, None)


# --- salary -----------------------------------------------------------------

_CURRENCY_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("CA$", "CAD"),
    ("C$", "CAD"),
    ("A$", "AUD"),
    ("AU$", "AUD"),
    ("NZ$", "NZD"),
    ("HK$", "HKD"),
    ("S$", "SGD"),
    ("R$", "BRL"),
    ("US$", "USD"),
    ("US ", "USD"),
    ("£", "GBP"),
    ("€", "EUR"),
    ("₹", "INR"),
    ("₩", "KRW"),
    ("₽", "RUB"),
    ("₺", "TRY"),
    ("zł", "PLN"),
    ("Kč", "CZK"),
    ("kr", "SEK"),
    ("CHF", "CHF"),
    ("¥", "JPY"),
)

_ISO_CURRENCY_RE = re.compile(
    r"\b(USD|EUR|GBP|CAD|AUD|NZD|CHF|SEK|NOK|DKK|PLN|CZK|HUF|RON|BGN|TRY|"
    r"BRL|MXN|ARS|CLP|COP|PEN|INR|JPY|CNY|KRW|SGD|HKD|TWD|MYR|THB|IDR|PHP|VND|"
    r"ZAR|AED|SAR|ILS|RUB|UAH)\b"
)

# "$" is genuinely ambiguous. Resolve it by the posting's country when we
# know it rather than assuming USD, which would mislabel every Canadian and
# Australian salary in the corpus.
_DOLLAR_BY_COUNTRY = {
    "US": "USD",
    "CA": "CAD",
    "AU": "AUD",
    "NZ": "NZD",
    "SG": "SGD",
    "HK": "HKD",
    "MX": "MXN",
    "AR": "ARS",
    "CL": "CLP",
    "CO": "COP",
    "TW": "TWD",
}
_YEN_BY_COUNTRY = {"JP": "JPY", "CN": "CNY"}
_KRONA_BY_COUNTRY = {"SE": "SEK", "NO": "NOK", "DK": "DKK", "IS": "ISK"}

_PERIOD_TOKENS: tuple[tuple[SalaryPeriodT, tuple[str, ...]], ...] = (
    (
        "HOUR",
        (
            "per hour",
            "/hour",
            "/hr",
            "hourly",
            "an hour",
            "pro stunde",
            "par heure",
            "por hora",
            "/h",
            "時給",
        ),
    ),
    ("DAY", ("per day", "/day", "daily", "per diem", "pro tag", "par jour", "por dia", "日給")),
    ("WEEK", ("per week", "/week", "weekly", "pro woche", "par semaine", "por semana", "週給")),
    (
        "MONTH",
        (
            "per month",
            "/month",
            "monthly",
            "/mo",
            "a month",
            "pro monat",
            "monatlich",
            "par mois",
            "mensuel",
            "por mes",
            "mensual",
            "por mês",
            "mensal",
            "月給",
            "月薪",
        ),
    ),
    (
        "YEAR",
        (
            "per year",
            "/year",
            "yearly",
            "/yr",
            "annually",
            "annual",
            "per annum",
            "p.a.",
            "pro jahr",
            "/jahr",
            " jahr",
            "jährlich",
            "jaehrlich",
            "par an",
            "/an",
            "par année",
            "annuel",
            "por año",
            "anual",
            "por ano",
            "年収",
            "年薪",
        ),
    ),
)


def detect_currency(text: object, country_iso: str | None = None) -> str | None:
    """ISO 4217 code from a salary string, disambiguated by country."""
    if not isinstance(text, str) or not text.strip():
        return None
    iso = _ISO_CURRENCY_RE.search(text.upper())
    if iso:
        return iso.group(1)
    for symbol, code in _CURRENCY_SYMBOLS:
        if symbol in text:
            if symbol == "$":
                break
            if code == "JPY" and symbol == "¥":
                return _YEN_BY_COUNTRY.get(country_iso or "", "JPY")
            if code == "SEK" and symbol == "kr":
                return _KRONA_BY_COUNTRY.get(country_iso or "", "SEK")
            return code
    if "$" in text:
        return _DOLLAR_BY_COUNTRY.get(country_iso or "", "USD")
    return None


#: Currencies where a five-figure amount is routinely a *monthly* salary,
#: so magnitude cannot distinguish MONTH from YEAR. 20,000 INR/month or
#: 20,000 HKD/month are ordinary wages; 20,000 USD/month is not something a
#: posting would write without saying "per month".
_MONTHLY_SCALE_CURRENCIES = frozenset(
    {
        "JPY",
        "KRW",
        "IDR",
        "VND",
        "HUF",
        "CLP",
        "COP",
        "INR",
        "RUB",
        "TRY",
        "PHP",
        "THB",
        "TWD",
        "MXN",
        "BRL",
        "ZAR",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "CZK",
        "HKD",
        "AED",
        "SAR",
        "ILS",
        "UAH",
        "ARS",
        "PEN",
        "MYR",
        "BGN",
        "RON",
    }
)

#: Above this, and in a currency not on the monthly-scale list, an
#: unlabelled figure is annual.
_ANNUAL_THRESHOLD = 20_000


def detect_period(
    text: object,
    amount: float | None = None,
    currency: str | None = None,
) -> SalaryPeriodT | None:
    """Salary period from explicit tokens, with two narrow fallbacks.

    Explicit tokens are always preferred. Failing that, only two magnitude
    inferences are safe enough to make:

    * under 500 in any currency is an hourly rate;
    * at or above 20,000 in a currency where that cannot be a monthly wage
      is annual — this is what resolves the very common ``"$120K - $160K"``
      shape, which carries no period token at all.

    Everything between those bounds stays ``None`` and routes to Tier 1,
    because a 60,000 figure really is annual in USD and monthly in JPY.
    """
    if isinstance(text, str) and text.strip():
        lowered = text.casefold()
        for period, tokens in _PERIOD_TOKENS:
            if any(token in lowered for token in tokens):
                return period
    if amount is not None:
        if 0 < amount < 500:
            return "HOUR"
        code = (currency or "").upper()
        if amount >= _ANNUAL_THRESHOLD and code and code not in _MONTHLY_SCALE_CURRENCIES:
            return "YEAR"
    return None


# --- the pass ---------------------------------------------------------------

#: Fields Tier 1 can answer. Used to decide whether a row needs a model at
#: all: if none of these is unresolved, the row is finished for free.
LLM_ANSWERABLE = (
    "placement",
    "employment_type",
    "experience_min_years",
    "seniority",
    "salary_min",
    "department",
    "education_level",
    "visa_sponsorship",
)


@dataclass
class Tier0Result:
    """Deterministic findings plus what is still open."""

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)

    @property
    def needs_llm(self) -> bool:
        return bool(self.unresolved & set(LLM_ANSWERABLE))


#: Truthy/falsy spellings that appear in the published snapshot. Needed
#: because **every column of ``all.parquet`` is VARCHAR** — the publisher's
#: ``diagonal_relaxed`` concat of per-ATS slices widens everything to
#: strings, so ``is_remote`` arrives as the text ``'true'``/``'false'``
#: even though the per-ATS parquet files type it BOOLEAN. An
#: ``isinstance(value, bool)`` check silently discards all 719,871
#: provider-labelled rows.
_TRUE_STRINGS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_STRINGS = frozenset({"false", "f", "no", "n", "0"})


def to_bool(value: object) -> bool | None:
    """Parse a provider boolean that may arrive as a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _clean(value: object) -> Any:
    """Treat the empty string as missing.

    Necessary because ``_job_to_row`` writes ``""`` for every absent
    optional field, so the published parquet has no NULLs in most string
    columns and a plain truthiness test is the only reliable check.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def run_tier0(row: dict[str, Any]) -> Tier0Result:
    """Apply every deterministic extractor to one snapshot row.

    Provider-supplied values always win. The point of the sidecar is to
    *fill* gaps, not to overwrite what an ATS stated structurally — and
    ``sources`` records which of the two happened, something the published
    dataset cannot express today.
    """
    result = Tier0Result()
    values, sources = result.values, result.sources

    title = _clean(row.get("title"))
    description = _clean(row.get("description"))
    location = _clean(row.get("location"))
    commitment = _clean(row.get("commitment"))

    # --- language
    provider_language = _clean(row.get("language"))
    if provider_language:
        values["language"] = str(provider_language)[:2].lower()
        sources["language"] = "provider"
    else:
        detected = detect_language(title, description)
        if detected:
            values["language"] = detected
            sources["language"] = "tier0"
        else:
            result.unresolved.add("language")

    # --- geography
    provider_country = _clean(row.get("country_iso"))
    resolved = resolve_location(location)
    country = str(provider_country).upper()[:2] if provider_country else resolved.country_iso
    if country:
        values["country_iso"] = country
        sources["country_iso"] = "provider" if provider_country else "tier0"
        region = _clean(row.get("region"))
        if region:
            values["region"] = str(region)
            sources["region"] = "provider"
        else:
            from enrich.geo import region_for_country

            derived_region = region_for_country(country)
            if derived_region:
                values["region"] = derived_region
                sources["region"] = "tier0"
    else:
        result.unresolved.update({"country_iso", "region"})

    provider_lat, provider_lon = _clean(row.get("lat")), _clean(row.get("lon"))
    try:
        lat_value = float(provider_lat) if provider_lat is not None else None
        lon_value = float(provider_lon) if provider_lon is not None else None
    except (TypeError, ValueError):
        lat_value = lon_value = None
    if lat_value is not None and lon_value is not None:
        values["lat"], values["lon"] = lat_value, lon_value
        values["geo_precision"] = "provider"
        sources["lat"] = sources["lon"] = "provider"
    elif resolved.lat is not None and resolved.lon is not None:
        # Only trust derived coordinates when the country agrees with the
        # provider's own country field; otherwise we would place a job in
        # the wrong hemisphere off an ambiguous city name.
        if not provider_country or resolved.country_iso == country:
            values["lat"], values["lon"] = resolved.lat, resolved.lon
            values["geo_precision"] = resolved.precision
            sources["lat"] = sources["lon"] = "tier0"
        else:
            result.unresolved.update({"lat", "lon"})
    else:
        result.unresolved.update({"lat", "lon"})

    # --- placement
    provider_remote = to_bool(row.get("is_remote"))
    if provider_remote is not None:
        values["is_remote"] = provider_remote
        # A provider ``False`` means "not remote", which does not
        # distinguish onsite from hybrid. Recording it as ``onsite`` would
        # assert something the provider never said, so only ``True`` maps
        # to a placement and ``False`` leaves placement open for Tier 1.
        if provider_remote:
            values["placement"] = "remote"
            sources["placement"] = "provider"
        else:
            result.unresolved.add("placement")
        sources["is_remote"] = "provider"
    else:
        placement = infer_placement(title, location, commitment)
        if placement is None and resolved.placeless:
            # "Remote", "Anywhere", "Worldwide" as the entire location is a
            # placement statement even though it names no geography.
            placement = "remote"
        if placement:
            values["placement"] = placement
            values["is_remote"] = placement == "remote"
            sources["placement"] = sources["is_remote"] = "tier0"
        else:
            result.unresolved.add("placement")

    # --- employment type
    provider_employment = _clean(row.get("employment_type"))
    if provider_employment:
        values["employment_type"] = str(provider_employment).upper()
        sources["employment_type"] = "provider"
    else:
        mapped = map_employment_type(commitment, title)
        if mapped:
            values["employment_type"] = mapped
            sources["employment_type"] = "tier0"
        else:
            result.unresolved.add("employment_type")

    # --- salary
    provider_min, provider_max = _clean(row.get("salary_min")), _clean(row.get("salary_max"))
    summary = _clean(row.get("salary_summary"))
    salary_min = salary_max = None
    if provider_min is not None:
        try:
            salary_min = float(provider_min)
            salary_max = float(provider_max) if provider_max is not None else None
            sources["salary_min"] = "provider"
        except (TypeError, ValueError):
            salary_min = salary_max = None
    if salary_min is None and summary:
        # Upstream's regex, reused rather than reimplemented: it is tight
        # and already handles the currency-symbol shapes the providers emit.
        salary_min, salary_max = parse_salary_range(summary)
        if salary_min is not None:
            sources["salary_min"] = "tier0"
    if salary_min is not None:
        values["salary_min"] = salary_min
        values["salary_max"] = salary_max
        currency = _clean(row.get("salary_currency")) or detect_currency(
            summary, values.get("country_iso")
        )
        if currency:
            values["salary_currency"] = str(currency).upper()[:3]
            sources["salary_currency"] = (
                "provider" if _clean(row.get("salary_currency")) else "tier0"
            )
        period = _clean(row.get("salary_period")) or detect_period(
            summary, salary_min, values.get("salary_currency")
        )
        if period:
            values["salary_period"] = str(period).upper()
            sources["salary_period"] = "provider" if _clean(row.get("salary_period")) else "tier0"
        else:
            result.unresolved.add("salary_period")
    else:
        result.unresolved.add("salary_min")

    # --- experience
    experience_min, experience_max = parse_experience_years(description)
    if experience_min is not None:
        values["experience_min_years"] = experience_min
        values["experience_max_years"] = experience_max
        sources["experience_min_years"] = "tier0"
    else:
        result.unresolved.add("experience_min_years")

    # Fields no deterministic rule can reach. Listed explicitly so routing
    # is driven by data rather than by a hardcoded "always call the model".
    for open_field in ("seniority", "department", "education_level", "visa_sponsorship"):
        result.unresolved.add(open_field)

    return result

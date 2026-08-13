"""Offline geo resolution: free-text location -> country, region, lat/lon.

This is Tier 0's largest win and it costs nothing per row. Upstream's
``_country_iso_from_location`` in ``pipeline/publisher.py`` matches country
names and NUTS prefixes only, so anything named by city ("Austin, TX",
"Bengaluru", "München") resolves to no country at all — and ``lat``/``lon``
are documented as "not derived from ``location`` text" and ship empty for
almost every provider.

Everything here is a dict lookup against ``geonamescache``'s bundled
GeoNames extract (252 countries, 34k cities with localized alternate
names). No network, no Nominatim rate limit, so the full 4.85M-row corpus
resolves in a single pass.

Resolution order is deliberate, cheapest and least ambiguous first:

1. Remote-only sentinels ("Remote", "Anywhere") — no place at all.
2. Explicit country name, in any of the corpus's major languages.
3. EURES-style NUTS codes (``"DE (DEA58)"``).
4. City match, which yields country *and* coordinates in one hit.
5. Trailing administrative code (``", TX"``, ``", ON"``), used to confirm
   a city's country or to stand alone as a country when no city matched.

The ambiguity that matters is two-letter tokens: ``CA`` is both California
and Canada, ``DE`` both Delaware and Germany. Cities are resolved *before*
those tokens are read precisely so the token becomes confirmation rather
than a coin flip — "San Francisco, CA" is anchored by the city, not by
``CA``.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Literal, NamedTuple

GeoPrecision = Literal["country", "admin1", "city", "provider"]

_CONTINENT_TO_REGION = {
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "AS": "Asia",
    "AF": "Africa",
    "OC": "Oceania",
    "AN": "Antarctica",
}

# Location strings that name no place. Kept separate from the country
# lookup so "Remote - Germany" still resolves DE while bare "Remote" does
# not invent one.
_PLACELESS = frozenset(
    {
        "remote",
        "remote work",
        "fully remote",
        "100% remote",
        "work from home",
        "wfh",
        "anywhere",
        "anywhere in the world",
        "worldwide",
        "global",
        "globally",
        "international",
        "multiple locations",
        "various locations",
        "various",
        "flexible",
        "n/a",
        "na",
        "none",
        "tbd",
        "unspecified",
        "home office",
        "telearbeit",
        "teletrabajo",
        "télétravail",
        "remoto",
        "en remoto",
    }
)

# Short country tokens worth honouring explicitly. A general "any ISO3
# code" rule is unsafe: ``AND`` is Andorra, ``ARE`` the UAE, ``CAN``
# Canada, ``PER`` Peru — all common English words that appear in location
# strings for unrelated reasons. This curated set covers the codes that
# actually show up in the corpus without those false positives.
_SHORT_COUNTRY_TOKENS = {
    "usa": "US",
    "uae": "AE",
    "gbr": "GB",
    "deu": "DE",
    "fra": "FR",
    "esp": "ES",
    "ita": "IT",
    "nld": "NL",
    "bel": "BE",
    "che": "CH",
    "aut": "AT",
    "swe": "SE",
    "nor": "NO",
    "dnk": "DK",
    "fin": "FI",
    "pol": "PL",
    "cze": "CZ",
    "prt": "PT",
    "irl": "IE",
    "ind": "IN",
    "jpn": "JP",
    "chn": "CN",
    "kor": "KR",
    "sgp": "SG",
    "hkg": "HK",
    "bra": "BR",
    "mex": "MX",
    "arg": "AR",
    "zaf": "ZA",
    "nzl": "NZ",
}

# Country aliases GeoNames' English-only ``name`` field misses. Scoped to
# the languages the corpus actually contains: German (bundesagentur),
# French (welcometothejungle, lever FR), Spanish (infojobs_es),
# Portuguese (gupy, programathor), Czech (jobs_cz), Japanese (herp,
# hrmos), Chinese (beisen, moka), Dutch, Italian, Swedish, Polish.
_COUNTRY_ALIASES: dict[str, str] = {
    # English short forms and historical usages
    "usa": "US",
    "u s a": "US",
    "us": "US",
    "u s": "US",
    "united states of america": "US",
    "america": "US",
    "uk": "GB",
    "u k": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "uae": "AE",
    "holland": "NL",
    "south korea": "KR",
    "korea": "KR",
    "north korea": "KP",
    "russia": "RU",
    "czechia": "CZ",
    "czech republic": "CZ",
    "ivory coast": "CI",
    "vietnam": "VN",
    "turkey": "TR",
    "turkiye": "TR",
    # German
    "deutschland": "DE",
    "osterreich": "AT",
    "schweiz": "CH",
    "frankreich": "FR",
    "italien": "IT",
    "spanien": "ES",
    "niederlande": "NL",
    "belgien": "BE",
    "polen": "PL",
    "tschechien": "CZ",
    "danemark": "DK",
    "schweden": "SE",
    "norwegen": "NO",
    "finnland": "FI",
    "grossbritannien": "GB",
    "vereinigtes konigreich": "GB",
    "vereinigte staaten": "US",
    "irland": "IE",
    "ungarn": "HU",
    "rumanien": "RO",
    "griechenland": "GR",
    "turkei": "TR",
    "kroatien": "HR",
    "slowakei": "SK",
    "slowenien": "SI",
    "bulgarien": "BG",
    "russland": "RU",
    "japan": "JP",
    "indien": "IN",
    "china": "CN",
    "brasilien": "BR",
    "mexiko": "MX",
    "kanada": "CA",
    "australien": "AU",
    "neuseeland": "NZ",
    "sudafrika": "ZA",
    # French
    "allemagne": "DE",
    "autriche": "AT",
    "suisse": "CH",
    "belgique": "BE",
    "pays-bas": "NL",
    "pays bas": "NL",
    "espagne": "ES",
    "italie": "IT",
    "royaume-uni": "GB",
    "royaume uni": "GB",
    "etats-unis": "US",
    "etats unis": "US",
    "irlande": "IE",
    "pologne": "PL",
    "suede": "SE",
    "norvege": "NO",
    "danemark ": "DK",
    "finlande": "FI",
    "grece": "GR",
    "hongrie": "HU",
    "roumanie": "RO",
    "tchequie": "CZ",
    "republique tcheque": "CZ",
    "bresil": "BR",
    "mexique": "MX",
    "japon": "JP",
    "inde": "IN",
    "chine": "CN",
    "maroc": "MA",
    "tunisie": "TN",
    "algerie": "DZ",
    "senegal": "SN",
    "cote d'ivoire": "CI",
    "cote divoire": "CI",
    "luxembourg": "LU",
    # Spanish / Portuguese
    "alemania": "DE",
    "alemanha": "DE",
    "espana": "ES",
    "espanha": "ES",
    "francia": "FR",
    "franca": "FR",
    "italia": "IT",
    "reino unido": "GB",
    "estados unidos": "US",
    "paises bajos": "NL",
    "paises baixos": "NL",
    "belgica": "BE",
    "suecia": "SE",
    "suiza": "CH",
    "suica": "CH",
    "brasil": "BR",
    "mexico": "MX",
    "argentina": "AR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
    "portugal": "PT",
    "japao": "JP",
    "china ": "CN",
    "india": "IN",
    # Dutch / Italian / Nordic / Polish / Czech
    "duitsland": "DE",
    "nederland": "NL",
    "belgie": "BE",
    "frankrijk": "FR",
    "verenigd koninkrijk": "GB",
    "verenigde staten": "US",
    "germania": "DE",
    "regno unito": "GB",
    "stati uniti": "US",
    "svizzera": "CH",
    "tyskland": "DE",
    "sverige": "SE",
    "norge": "NO",
    "danmark": "DK",
    "suomi": "FI",
    "niemcy": "DE",
    "polska": "PL",
    "wielka brytania": "GB",
    "stany zjednoczone": "US",
    "nemecko": "DE",
    "ceska republika": "CZ",
    "cesko": "CZ",
    "slovensko": "SK",
    # Japanese / Chinese / Korean
    "日本": "JP",
    "アメリカ": "US",
    "米国": "US",
    "ドイツ": "DE",
    "フランス": "FR",
    "イギリス": "GB",
    "中国": "CN",
    "中華人民共和国": "CN",
    "美国": "US",
    "德国": "DE",
    "法国": "FR",
    "英国": "GB",
    "日本国": "JP",
    "新加坡": "SG",
    "香港": "HK",
    "台湾": "TW",
    "한국": "KR",
    "대한민국": "KR",
    "미국": "US",
}

# Canadian provinces and territories: geonamescache ships US states but
# not these, and Canadian postings are a meaningful slice (jobbankca).
_CA_PROVINCES = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}

# Australian states (seek).
_AU_STATES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "WA": "Western Australia",
    "SA": "South Australia",
    "TAS": "Tasmania",
    "ACT": "Australian Capital Territory",
    "NT": "Northern Territory",
}

_SPLIT_RE = re.compile(r"[,/|;·•\n\t]+|\s+[-–—]\s+|\s+\bor\b\s+|\s+\bund\b\s+|\s+\bet\b\s+")
_NUTS_PAREN_RE = re.compile(r"^([A-Z]{2})\s*\(([A-Z0-9]{2,6})\)$")
# A bare NUTS code must carry a digit (DEA58, FRK21, ITC4). Without that
# requirement this pattern matches any 3-6 letter uppercase word, which
# silently turned "AUSTIN" into AU, "BERLIN" into BE and "REMOTE" into RE.
_NUTS_BARE_RE = re.compile(r"^([A-Z]{2})([A-Z0-9]{1,4})$")
_PAREN_STRIP_RE = re.compile(r"\s*\([^)]*\)\s*")
_ZIP_RE = re.compile(r"\b\d{4,6}(?:-\d{4})?\b")
# Han / Hiragana / Katakana / Hangul. City names in these scripts are
# routinely two characters ("東京", "上海", "서울"), so the city index's
# minimum-length guard has to be script-aware or they never index.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _fold(value: str) -> str:
    """Lowercase, strip accents, collapse whitespace.

    CJK is left intact — NFKD does not decompose it and casefolding is a
    no-op, so the alias table's Japanese/Chinese keys still match.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.strip().lower())


class _Indices(NamedTuple):
    countries: dict[str, str]
    country_region: dict[str, str]
    country_coords: dict[str, tuple[float, float]]
    cities: dict[str, tuple[str, float, float, int]]
    us_states: dict[str, str]


@lru_cache(maxsize=1)
def _indices() -> _Indices:
    """Build every lookup table once per process.

    Costs roughly a second and ~150 MB for the city index including
    localized alternate names. Amortized over millions of rows that is
    free; ``lru_cache`` keeps it from being rebuilt per worker call.
    """
    import geonamescache

    cache = geonamescache.GeonamesCache()

    countries: dict[str, str] = {}
    country_region: dict[str, str] = {}
    country_coords: dict[str, tuple[float, float]] = {}
    for iso, meta in cache.get_countries().items():
        countries[_fold(meta["name"])] = iso
        countries[iso.lower()] = iso
        iso3 = meta.get("iso3")
        if iso3:
            countries[iso3.lower()] = iso
        region = _CONTINENT_TO_REGION.get(str(meta.get("continentcode")))
        if region:
            country_region[iso] = region
    for alias, iso in _COUNTRY_ALIASES.items():
        countries.setdefault(_fold(alias), iso)

    # City index. On a name collision keep the larger city: "Paris" should
    # resolve to France, not to Paris, Texas. Population is the only
    # signal available offline and it gets the common case right.
    cities: dict[str, tuple[str, float, float, int]] = {}

    def _offer(name: str, iso: str, lat: float, lon: float, population: int) -> None:
        min_length = 2 if _CJK_RE.search(name) else 3
        if not name or len(name) < min_length:
            return
        existing = cities.get(name)
        if existing is None or population > existing[3]:
            cities[name] = (iso, lat, lon, population)

    for meta in cache.get_cities().values():
        iso = str(meta["countrycode"])
        lat = float(meta["latitude"])
        lon = float(meta["longitude"])
        population = int(meta.get("population") or 0)
        _offer(_fold(str(meta["name"])), iso, lat, lon, population)
        # Alternate names carry the localized forms ("München", "東京",
        # "Bengaluru"), which is what makes non-English postings resolve.
        for alternate in meta.get("alternatenames") or []:
            if isinstance(alternate, str) and alternate:
                _offer(_fold(alternate), iso, lat, lon, population)

    # Fallback coordinates per country: the largest known city, used when a
    # posting names a country but no city we recognize.
    best_pop: dict[str, int] = {}
    for meta in cache.get_cities().values():
        iso = str(meta["countrycode"])
        population = int(meta.get("population") or 0)
        if population > best_pop.get(iso, -1):
            best_pop[iso] = population
            country_coords[iso] = (float(meta["latitude"]), float(meta["longitude"]))

    us_states: dict[str, str] = {}
    for code, meta in cache.get_us_states().items():
        us_states[code.upper()] = str(meta["name"])

    return _Indices(countries, country_region, country_coords, cities, us_states)


def region_for_country(country_iso: object) -> str | None:
    """Continent name for an ISO-3166 alpha-2 code, matching the upstream
    ``region`` vocabulary in ``docs/JOB_SCHEMA.md``."""
    if not isinstance(country_iso, str):
        return None
    code = country_iso.strip().upper()
    if len(code) != 2:
        return None
    return _indices().country_region.get(code)


class ResolvedLocation(NamedTuple):
    country_iso: str | None
    region: str | None
    lat: float | None
    lon: float | None
    precision: GeoPrecision | None
    placeless: bool


_EMPTY = ResolvedLocation(None, None, None, None, None, False)


def _tokens(text: str) -> list[str]:
    """Split a location string into candidate place tokens.

    Parenthesised content is kept as its own token before being stripped,
    because EURES writes the NUTS code there ("DE (DEA58)") while other
    providers use it for noise ("Berlin (hybrid)").
    """
    parts: list[str] = []
    for chunk in _SPLIT_RE.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(chunk)
        without_parens = _PAREN_STRIP_RE.sub(" ", chunk).strip()
        if without_parens and without_parens != chunk:
            parts.append(without_parens)
    return parts


def resolve_location(text: object) -> ResolvedLocation:
    """Resolve a free-text location to country, region and coordinates."""
    if not isinstance(text, str) or not text.strip():
        return _EMPTY

    raw = _ZIP_RE.sub(" ", text).strip()
    folded_full = _fold(raw)
    if folded_full in _PLACELESS:
        return ResolvedLocation(None, None, None, None, None, True)

    idx = _indices()
    tokens = _tokens(raw)
    folded_tokens = [_fold(token) for token in tokens]
    placeless = any(token in _PLACELESS for token in folded_tokens)

    country: str | None = None
    lat: float | None = None
    lon: float | None = None
    precision: GeoPrecision | None = None

    # 1. Explicit country name anywhere in the string. Longest token first
    #    so "New Zealand" is not shadowed by a stray "Zealand".
    country_token: str | None = None
    for folded in sorted(folded_tokens, key=len, reverse=True):
        found = idx.countries.get(folded) if len(folded) >= 4 else None
        found = found or _SHORT_COUNTRY_TOKENS.get(folded)
        if found:
            country = found
            country_token = folded
            precision = "country"
            break

    # 2. EURES NUTS codes: the first two characters are the country.
    if country is None:
        for token in tokens:
            upper = token.strip().upper()
            candidate: str | None = None
            paren = _NUTS_PAREN_RE.match(upper)
            if paren:
                candidate = paren.group(1)
            else:
                bare = _NUTS_BARE_RE.match(upper)
                # Require a digit in the suffix, else every uppercase word
                # of the right length reads as a NUTS code.
                if bare and any(ch.isdigit() for ch in bare.group(2)):
                    candidate = bare.group(1)
            if candidate and candidate in idx.country_region:
                country = candidate
                precision = "country"
                break

    # 3. City match. Runs before two-letter admin codes so that an
    #    anchoring city decides "CA" = California vs Canada.
    city_country: str | None = None
    best_population = -1
    for folded in folded_tokens:
        # A token already spent naming the country must not double as a
        # city: "USA" is an alternate name for a small US town, and letting
        # it win would attach that town's coordinates to a country-level
        # location.
        if folded == country_token:
            continue
        hit = idx.cities.get(folded)
        if hit is None:
            continue
        hit_iso, hit_lat, hit_lon, population = hit
        # When a country is already known, only accept a city inside it.
        if country is not None and hit_iso != country:
            continue
        if population > best_population:
            best_population = population
            city_country, lat, lon = hit_iso, hit_lat, hit_lon
    if city_country is not None:
        country = country or city_country
        precision = "city"

    # 4. Administrative codes. Confirmation when a city already anchored
    #    the country; a standalone country guess otherwise.
    if precision != "city" or country is None:
        for token in tokens:
            upper = token.strip().upper()
            short = _SHORT_COUNTRY_TOKENS.get(upper.lower())
            if short and country is None:
                country, precision = short, "country"
                break
            if len(upper) == 2:
                if upper in idx.us_states and country in (None, "US"):
                    country, precision = "US", precision or "admin1"
                    break
                if upper in _CA_PROVINCES and country in (None, "CA"):
                    country, precision = "CA", precision or "admin1"
                    break
                mapped = idx.countries.get(upper.lower())
                if mapped and country is None:
                    country, precision = mapped, "country"
                    break
            elif upper in _AU_STATES and country in (None, "AU"):
                country, precision = "AU", precision or "admin1"
                break
            elif upper in idx.us_states.values() or _fold(upper) in {
                _fold(name) for name in idx.us_states.values()
            }:
                if country in (None, "US"):
                    country, precision = "US", precision or "admin1"
                    break

    if country is None:
        return ResolvedLocation(None, None, None, None, None, placeless)

    if lat is None or lon is None:
        fallback = idx.country_coords.get(country)
        if fallback is not None:
            lat, lon = fallback
            precision = precision or "country"

    return ResolvedLocation(
        country_iso=country,
        region=region_for_country(country),
        lat=lat,
        lon=lon,
        precision=precision or "country",
        placeless=placeless,
    )


def country_from_location(text: object) -> str | None:
    """Just the ISO country code. Convenience for profiling queries."""
    return resolve_location(text).country_iso

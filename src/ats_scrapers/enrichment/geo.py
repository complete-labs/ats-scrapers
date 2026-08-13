"""Country normalization shared across scrapers.

ATSes name countries every possible way: an alpha-2 code (Lever's
``country``), an alpha-3 (Amazon's ``country_code``), or free text an
employer typed (Workday's ``United States of America``, Ashby's
``USA``). This module turns any of those into the canonical alpha-2
``Job.country_iso`` plus the continent that ``Job.region`` wants.

Before this existed every scraper carried its own partial table, which
meant a country was recognised on one source and dropped on another.
Add new spellings to :data:`_ALIASES` rather than to a scraper.

The ``_COUNTRIES`` table is generated from the ISO 3166-1 standard
list; ``region`` collapses the ISO region/sub-region pair onto the
seven continent values ``Job.region`` documents (the ISO "Americas"
bucket splits into North/South America).
"""

from __future__ import annotations

import re
import unicodedata

# alpha-2 -> (alpha-3, continent, canonical English name)
_COUNTRIES: dict[str, tuple[str, str, str]] = {
    "AD": ("AND", "Europe", "Andorra"),
    "AE": ("ARE", "Asia", "United Arab Emirates"),
    "AF": ("AFG", "Asia", "Afghanistan"),
    "AG": ("ATG", "North America", "Antigua and Barbuda"),
    "AI": ("AIA", "North America", "Anguilla"),
    "AL": ("ALB", "Europe", "Albania"),
    "AM": ("ARM", "Asia", "Armenia"),
    "AO": ("AGO", "Africa", "Angola"),
    "AQ": ("ATA", "Antarctica", "Antarctica"),
    "AR": ("ARG", "South America", "Argentina"),
    "AS": ("ASM", "Oceania", "American Samoa"),
    "AT": ("AUT", "Europe", "Austria"),
    "AU": ("AUS", "Oceania", "Australia"),
    "AW": ("ABW", "North America", "Aruba"),
    "AX": ("ALA", "Europe", "Åland Islands"),
    "AZ": ("AZE", "Asia", "Azerbaijan"),
    "BA": ("BIH", "Europe", "Bosnia and Herzegovina"),
    "BB": ("BRB", "North America", "Barbados"),
    "BD": ("BGD", "Asia", "Bangladesh"),
    "BE": ("BEL", "Europe", "Belgium"),
    "BF": ("BFA", "Africa", "Burkina Faso"),
    "BG": ("BGR", "Europe", "Bulgaria"),
    "BH": ("BHR", "Asia", "Bahrain"),
    "BI": ("BDI", "Africa", "Burundi"),
    "BJ": ("BEN", "Africa", "Benin"),
    "BL": ("BLM", "North America", "Saint Barthélemy"),
    "BM": ("BMU", "North America", "Bermuda"),
    "BN": ("BRN", "Asia", "Brunei Darussalam"),
    "BO": ("BOL", "South America", "Bolivia, Plurinational State of"),
    "BQ": ("BES", "North America", "Bonaire, Sint Eustatius and Saba"),
    "BR": ("BRA", "South America", "Brazil"),
    "BS": ("BHS", "North America", "Bahamas"),
    "BT": ("BTN", "Asia", "Bhutan"),
    "BV": ("BVT", "Antarctica", "Bouvet Island"),
    "BW": ("BWA", "Africa", "Botswana"),
    "BY": ("BLR", "Europe", "Belarus"),
    "BZ": ("BLZ", "North America", "Belize"),
    "CA": ("CAN", "North America", "Canada"),
    "CC": ("CCK", "Oceania", "Cocos (Keeling) Islands"),
    "CD": ("COD", "Africa", "Congo, Democratic Republic of the"),
    "CF": ("CAF", "Africa", "Central African Republic"),
    "CG": ("COG", "Africa", "Congo"),
    "CH": ("CHE", "Europe", "Switzerland"),
    "CI": ("CIV", "Africa", "Côte d'Ivoire"),
    "CK": ("COK", "Oceania", "Cook Islands"),
    "CL": ("CHL", "South America", "Chile"),
    "CM": ("CMR", "Africa", "Cameroon"),
    "CN": ("CHN", "Asia", "China"),
    "CO": ("COL", "South America", "Colombia"),
    "CR": ("CRI", "North America", "Costa Rica"),
    "CU": ("CUB", "North America", "Cuba"),
    "CV": ("CPV", "Africa", "Cabo Verde"),
    "CW": ("CUW", "North America", "Curaçao"),
    "CX": ("CXR", "Oceania", "Christmas Island"),
    "CY": ("CYP", "Asia", "Cyprus"),
    "CZ": ("CZE", "Europe", "Czechia"),
    "DE": ("DEU", "Europe", "Germany"),
    "DJ": ("DJI", "Africa", "Djibouti"),
    "DK": ("DNK", "Europe", "Denmark"),
    "DM": ("DMA", "North America", "Dominica"),
    "DO": ("DOM", "North America", "Dominican Republic"),
    "DZ": ("DZA", "Africa", "Algeria"),
    "EC": ("ECU", "South America", "Ecuador"),
    "EE": ("EST", "Europe", "Estonia"),
    "EG": ("EGY", "Africa", "Egypt"),
    "EH": ("ESH", "Africa", "Western Sahara"),
    "ER": ("ERI", "Africa", "Eritrea"),
    "ES": ("ESP", "Europe", "Spain"),
    "ET": ("ETH", "Africa", "Ethiopia"),
    "FI": ("FIN", "Europe", "Finland"),
    "FJ": ("FJI", "Oceania", "Fiji"),
    "FK": ("FLK", "South America", "Falkland Islands (Malvinas)"),
    "FM": ("FSM", "Oceania", "Micronesia, Federated States of"),
    "FO": ("FRO", "Europe", "Faroe Islands"),
    "FR": ("FRA", "Europe", "France"),
    "GA": ("GAB", "Africa", "Gabon"),
    "GB": ("GBR", "Europe", "United Kingdom of Great Britain and Northern Ireland"),
    "GD": ("GRD", "North America", "Grenada"),
    "GE": ("GEO", "Asia", "Georgia"),
    "GF": ("GUF", "South America", "French Guiana"),
    "GG": ("GGY", "Europe", "Guernsey"),
    "GH": ("GHA", "Africa", "Ghana"),
    "GI": ("GIB", "Europe", "Gibraltar"),
    "GL": ("GRL", "North America", "Greenland"),
    "GM": ("GMB", "Africa", "Gambia"),
    "GN": ("GIN", "Africa", "Guinea"),
    "GP": ("GLP", "North America", "Guadeloupe"),
    "GQ": ("GNQ", "Africa", "Equatorial Guinea"),
    "GR": ("GRC", "Europe", "Greece"),
    "GS": ("SGS", "Antarctica", "South Georgia and the South Sandwich Islands"),
    "GT": ("GTM", "North America", "Guatemala"),
    "GU": ("GUM", "Oceania", "Guam"),
    "GW": ("GNB", "Africa", "Guinea-Bissau"),
    "GY": ("GUY", "South America", "Guyana"),
    "HK": ("HKG", "Asia", "Hong Kong"),
    "HM": ("HMD", "Antarctica", "Heard Island and McDonald Islands"),
    "HN": ("HND", "North America", "Honduras"),
    "HR": ("HRV", "Europe", "Croatia"),
    "HT": ("HTI", "North America", "Haiti"),
    "HU": ("HUN", "Europe", "Hungary"),
    "ID": ("IDN", "Asia", "Indonesia"),
    "IE": ("IRL", "Europe", "Ireland"),
    "IL": ("ISR", "Asia", "Israel"),
    "IM": ("IMN", "Europe", "Isle of Man"),
    "IN": ("IND", "Asia", "India"),
    "IO": ("IOT", "Africa", "British Indian Ocean Territory"),
    "IQ": ("IRQ", "Asia", "Iraq"),
    "IR": ("IRN", "Asia", "Iran, Islamic Republic of"),
    "IS": ("ISL", "Europe", "Iceland"),
    "IT": ("ITA", "Europe", "Italy"),
    "JE": ("JEY", "Europe", "Jersey"),
    "JM": ("JAM", "North America", "Jamaica"),
    "JO": ("JOR", "Asia", "Jordan"),
    "JP": ("JPN", "Asia", "Japan"),
    "KE": ("KEN", "Africa", "Kenya"),
    "KG": ("KGZ", "Asia", "Kyrgyzstan"),
    "KH": ("KHM", "Asia", "Cambodia"),
    "KI": ("KIR", "Oceania", "Kiribati"),
    "KM": ("COM", "Africa", "Comoros"),
    "KN": ("KNA", "North America", "Saint Kitts and Nevis"),
    "KP": ("PRK", "Asia", "Korea, Democratic People's Republic of"),
    "KR": ("KOR", "Asia", "Korea, Republic of"),
    "KW": ("KWT", "Asia", "Kuwait"),
    "KY": ("CYM", "North America", "Cayman Islands"),
    "KZ": ("KAZ", "Asia", "Kazakhstan"),
    "LA": ("LAO", "Asia", "Lao People's Democratic Republic"),
    "LB": ("LBN", "Asia", "Lebanon"),
    "LC": ("LCA", "North America", "Saint Lucia"),
    "LI": ("LIE", "Europe", "Liechtenstein"),
    "LK": ("LKA", "Asia", "Sri Lanka"),
    "LR": ("LBR", "Africa", "Liberia"),
    "LS": ("LSO", "Africa", "Lesotho"),
    "LT": ("LTU", "Europe", "Lithuania"),
    "LU": ("LUX", "Europe", "Luxembourg"),
    "LV": ("LVA", "Europe", "Latvia"),
    "LY": ("LBY", "Africa", "Libya"),
    "MA": ("MAR", "Africa", "Morocco"),
    "MC": ("MCO", "Europe", "Monaco"),
    "MD": ("MDA", "Europe", "Moldova, Republic of"),
    "ME": ("MNE", "Europe", "Montenegro"),
    "MF": ("MAF", "North America", "Saint Martin (French part)"),
    "MG": ("MDG", "Africa", "Madagascar"),
    "MH": ("MHL", "Oceania", "Marshall Islands"),
    "MK": ("MKD", "Europe", "North Macedonia"),
    "ML": ("MLI", "Africa", "Mali"),
    "MM": ("MMR", "Asia", "Myanmar"),
    "MN": ("MNG", "Asia", "Mongolia"),
    "MO": ("MAC", "Asia", "Macao"),
    "MP": ("MNP", "Oceania", "Northern Mariana Islands"),
    "MQ": ("MTQ", "North America", "Martinique"),
    "MR": ("MRT", "Africa", "Mauritania"),
    "MS": ("MSR", "North America", "Montserrat"),
    "MT": ("MLT", "Europe", "Malta"),
    "MU": ("MUS", "Africa", "Mauritius"),
    "MV": ("MDV", "Asia", "Maldives"),
    "MW": ("MWI", "Africa", "Malawi"),
    "MX": ("MEX", "North America", "Mexico"),
    "MY": ("MYS", "Asia", "Malaysia"),
    "MZ": ("MOZ", "Africa", "Mozambique"),
    "NA": ("NAM", "Africa", "Namibia"),
    "NC": ("NCL", "Oceania", "New Caledonia"),
    "NE": ("NER", "Africa", "Niger"),
    "NF": ("NFK", "Oceania", "Norfolk Island"),
    "NG": ("NGA", "Africa", "Nigeria"),
    "NI": ("NIC", "North America", "Nicaragua"),
    "NL": ("NLD", "Europe", "Netherlands, Kingdom of the"),
    "NO": ("NOR", "Europe", "Norway"),
    "NP": ("NPL", "Asia", "Nepal"),
    "NR": ("NRU", "Oceania", "Nauru"),
    "NU": ("NIU", "Oceania", "Niue"),
    "NZ": ("NZL", "Oceania", "New Zealand"),
    "OM": ("OMN", "Asia", "Oman"),
    "PA": ("PAN", "North America", "Panama"),
    "PE": ("PER", "South America", "Peru"),
    "PF": ("PYF", "Oceania", "French Polynesia"),
    "PG": ("PNG", "Oceania", "Papua New Guinea"),
    "PH": ("PHL", "Asia", "Philippines"),
    "PK": ("PAK", "Asia", "Pakistan"),
    "PL": ("POL", "Europe", "Poland"),
    "PM": ("SPM", "North America", "Saint Pierre and Miquelon"),
    "PN": ("PCN", "Oceania", "Pitcairn"),
    "PR": ("PRI", "North America", "Puerto Rico"),
    "PS": ("PSE", "Asia", "Palestine, State of"),
    "PT": ("PRT", "Europe", "Portugal"),
    "PW": ("PLW", "Oceania", "Palau"),
    "PY": ("PRY", "South America", "Paraguay"),
    "QA": ("QAT", "Asia", "Qatar"),
    "RE": ("REU", "Africa", "Réunion"),
    "RO": ("ROU", "Europe", "Romania"),
    "RS": ("SRB", "Europe", "Serbia"),
    "RU": ("RUS", "Europe", "Russian Federation"),
    "RW": ("RWA", "Africa", "Rwanda"),
    "SA": ("SAU", "Asia", "Saudi Arabia"),
    "SB": ("SLB", "Oceania", "Solomon Islands"),
    "SC": ("SYC", "Africa", "Seychelles"),
    "SD": ("SDN", "Africa", "Sudan"),
    "SE": ("SWE", "Europe", "Sweden"),
    "SG": ("SGP", "Asia", "Singapore"),
    "SH": ("SHN", "Africa", "Saint Helena, Ascension and Tristan da Cunha"),
    "SI": ("SVN", "Europe", "Slovenia"),
    "SJ": ("SJM", "Europe", "Svalbard and Jan Mayen"),
    "SK": ("SVK", "Europe", "Slovakia"),
    "SL": ("SLE", "Africa", "Sierra Leone"),
    "SM": ("SMR", "Europe", "San Marino"),
    "SN": ("SEN", "Africa", "Senegal"),
    "SO": ("SOM", "Africa", "Somalia"),
    "SR": ("SUR", "South America", "Suriname"),
    "SS": ("SSD", "Africa", "South Sudan"),
    "ST": ("STP", "Africa", "Sao Tome and Principe"),
    "SV": ("SLV", "North America", "El Salvador"),
    "SX": ("SXM", "North America", "Sint Maarten (Dutch part)"),
    "SY": ("SYR", "Asia", "Syrian Arab Republic"),
    "SZ": ("SWZ", "Africa", "Eswatini"),
    "TC": ("TCA", "North America", "Turks and Caicos Islands"),
    "TD": ("TCD", "Africa", "Chad"),
    "TF": ("ATF", "Antarctica", "French Southern Territories"),
    "TG": ("TGO", "Africa", "Togo"),
    "TH": ("THA", "Asia", "Thailand"),
    "TJ": ("TJK", "Asia", "Tajikistan"),
    "TK": ("TKL", "Oceania", "Tokelau"),
    "TL": ("TLS", "Asia", "Timor-Leste"),
    "TM": ("TKM", "Asia", "Turkmenistan"),
    "TN": ("TUN", "Africa", "Tunisia"),
    "TO": ("TON", "Oceania", "Tonga"),
    "TR": ("TUR", "Asia", "Türkiye"),
    "TT": ("TTO", "North America", "Trinidad and Tobago"),
    "TV": ("TUV", "Oceania", "Tuvalu"),
    "TW": ("TWN", "Asia", "Taiwan, Province of China"),
    "TZ": ("TZA", "Africa", "Tanzania, United Republic of"),
    "UA": ("UKR", "Europe", "Ukraine"),
    "UG": ("UGA", "Africa", "Uganda"),
    "UM": ("UMI", "Oceania", "United States Minor Outlying Islands"),
    "US": ("USA", "North America", "United States of America"),
    "UY": ("URY", "South America", "Uruguay"),
    "UZ": ("UZB", "Asia", "Uzbekistan"),
    "VA": ("VAT", "Europe", "Holy See"),
    "VC": ("VCT", "North America", "Saint Vincent and the Grenadines"),
    "VE": ("VEN", "South America", "Venezuela, Bolivarian Republic of"),
    "VG": ("VGB", "North America", "Virgin Islands (British)"),
    "VI": ("VIR", "North America", "Virgin Islands (U.S.)"),
    "VN": ("VNM", "Asia", "Viet Nam"),
    "VU": ("VUT", "Oceania", "Vanuatu"),
    "WF": ("WLF", "Oceania", "Wallis and Futuna"),
    "WS": ("WSM", "Oceania", "Samoa"),
    # Kosovo has no ISO 3166-1 assignment, but XK is the de-facto code and
    # is what live ATS payloads use, so dropping it would lose real rows.
    "XK": ("XKX", "Europe", "Kosovo"),
    "YE": ("YEM", "Asia", "Yemen"),
    "YT": ("MYT", "Africa", "Mayotte"),
    "ZA": ("ZAF", "Africa", "South Africa"),
    "ZM": ("ZMB", "Africa", "Zambia"),
    "ZW": ("ZWE", "Africa", "Zimbabwe"),
}

# Spellings the ISO list doesn't carry: colloquial short forms, common
# ATS variants, and a few localised names seen in live payloads.
_ALIASES: dict[str, str] = {
    "usa": "US",
    "u s a": "US",
    "u s": "US",
    "america": "US",
    "united states": "US",
    "united states of america": "US",
    "the united states": "US",
    "uk": "GB",
    "u k": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "south korea": "KR",
    "republic of korea": "KR",
    "north korea": "KP",
    "russia": "RU",
    "vietnam": "VN",
    "taiwan": "TW",
    "czech republic": "CZ",
    "turkey": "TR",
    "turkiye": "TR",
    "ivory coast": "CI",
    "cape verde": "CV",
    "east timor": "TL",
    "laos": "LA",
    "macau": "MO",
    "macedonia": "MK",
    "bosnia": "BA",
    "uae": "AE",
    "syria": "SY",
    "iran": "IR",
    "bolivia": "BO",
    "venezuela": "VE",
    "tanzania": "TZ",
    "moldova": "MD",
    "brunei": "BN",
    "the netherlands": "NL",
    "holland": "NL",
    "swaziland": "SZ",
    "burma": "MM",
    "congo kinshasa": "CD",
    "democratic republic of the congo": "CD",
    "congo brazzaville": "CG",
    "republic of the congo": "CG",
    "palestine": "PS",
    "vatican city": "VA",
    "deutschland": "DE",
    "espana": "ES",
    "france metropolitaine": "FR",
    "osterreich": "AT",
    "schweiz": "CH",
    "suisse": "CH",
    "sverige": "SE",
    "danmark": "DK",
    "norge": "NO",
    "suomi": "FI",
    "polska": "PL",
    "italia": "IT",
    "nippon": "JP",
    "bharat": "IN",
}

# Values that name a market or "anywhere" rather than a country. Ashby,
# Workday and friends all put these in the same field as real countries,
# and mapping them to an arbitrary member state would be worse than
# returning nothing.
_NOT_A_COUNTRY = frozenset(
    {
        "global",
        "worldwide",
        "anywhere",
        "any location",
        "multiple locations",
        "various",
        "remote",
        "european union",
        "eu",
        "emea",
        "apac",
        "latam",
        "north america",
        "south america",
        "europe",
        "asia",
        "africa",
        "oceania",
        "international",
        "n a",
        "none",
        "unknown",
        "tbd",
    }
)

_BY_ALPHA3: dict[str, str] = {a3: a2 for a2, (a3, _, _) in _COUNTRIES.items()}
_BY_NAME: dict[str, str] = {}
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _normalize(value: str) -> str:
    """Casefold, strip accents, and collapse punctuation to single spaces."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _PUNCT_RE.sub(" ", stripped.casefold()).strip()


def _index_names() -> None:
    for alpha2, (_, _, name) in _COUNTRIES.items():
        _BY_NAME.setdefault(_normalize(name), alpha2)
        # ISO renders several names inverted ("Korea, Republic of",
        # "Bolivia (Plurinational State of)"); index the leading segment
        # too so the everyday spelling resolves.
        head = _normalize(re.split(r"[,(]", name)[0])
        if head:
            _BY_NAME.setdefault(head, alpha2)
    for alias, alpha2 in _ALIASES.items():
        _BY_NAME[_normalize(alias)] = alpha2


_index_names()


def country_to_iso(value: object) -> str | None:
    """Resolve an alpha-2, alpha-3, or country name to an alpha-2 code.

    Returns ``None`` for anything that names no single country —
    including supranational or "anywhere" values such as ``Global`` or
    ``European Union``, which several ATSes place in the same field as
    real countries.

    >>> country_to_iso("us"), country_to_iso("USA")
    ('US', 'US')
    >>> country_to_iso("United States of America")
    'US'
    >>> country_to_iso("European Union") is None
    True
    """
    if not isinstance(value, str):
        return None
    normalized = _normalize(value)
    if not normalized or normalized in _NOT_A_COUNTRY:
        return None
    compact = normalized.replace(" ", "").upper()
    if len(compact) == 2 and compact in _COUNTRIES:
        return compact
    if len(compact) == 3 and compact in _BY_ALPHA3:
        return _BY_ALPHA3[compact]
    return _BY_NAME.get(normalized)


def region_for(country_iso: object) -> str | None:
    """Return the continent for an alpha-2 code, or ``None`` if unknown.

    Matches the vocabulary ``Job.region`` documents: ``Europe``,
    ``North America``, ``Asia``, ``South America``, ``Africa``,
    ``Oceania``, ``Antarctica``.
    """
    if not isinstance(country_iso, str):
        return None
    entry = _COUNTRIES.get(country_iso.strip().upper())
    return entry[1] if entry else None


def resolve_country(value: object) -> tuple[str | None, str | None]:
    """Convenience: ``(country_iso, region)`` in one call."""
    iso = country_to_iso(value)
    return iso, region_for(iso)

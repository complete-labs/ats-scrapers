"""Company-name and domain normalisation for entity resolution.

Matching an ATS tenant name ("10x Genomics", "Acme Robotics Inc.") to a
SEC legal name ("10X GENOMICS, INC.") or a PDL record ("10x genomics")
needs more than casefolding. Three normal forms are produced, in
decreasing strictness:

``display``
    Light cleanup only — whitespace collapsed, ATS decorations dropped.
    What a human should see in the review file.

``core``
    Casefolded, accent-stripped, punctuation-normalised, legal suffix
    removed. This is the string fed to rapidfuzz.

``key``
    ``core`` reduced to ``[a-z0-9]`` only. Used as an exact-match
    blocking key so fuzzy scoring runs against a handful of candidates
    instead of the whole 23M-row PDL table.

Note on ``pipeline.publisher``: its ``_company_norm`` is a bare
``[^a-z0-9]`` strip over an already-lowercased column. That is enough
for a self-consistent dedup blocking key, but it keeps no legal-suffix
or accent handling, so it cannot match "Acme, Inc." to "Acme" the way
cross-source resolution needs. This module does not reuse it.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

# Trailing corporate-form tokens. Order matters only in that multi-word
# forms must be tried before their single-word prefixes.
_LEGAL_SUFFIXES: tuple[str, ...] = (
    "kabushiki kaisha",
    "public limited company",
    "limited liability company",
    "limited partnership",
    "incorporated",
    "corporation",
    "companies",
    "company",
    "limited",
    "holdings",
    "holding",
    "pllc",
    "llc",
    "lllp",
    "llp",
    "ltda",
    "ltd",
    "inc",
    "corp",
    "plc",
    "pbc",
    "lp",
    "pc",
    "sarl",
    "gmbh",
    "mbh",
    "ag",
    "nv",
    "bv",
    "sa",
    "as",
    "ab",
    "oy",
    "oyj",
    "aps",
    "srl",
    "spa",
    "pty",
    "kk",
    "kg",
    "se",
    "co",
)

# Decorations ATS tenants bolt onto their display name. Stripped from
# every normal form including `display`.
_ATS_DECORATIONS = re.compile(
    r"""
    \s*(?:
        \b(?:careers?|jobs?|job\s+board|hiring|recruiting|talent)\b
        (?:\s+(?:page|site|portal))?
      | \(\s*(?:us|usa|united\s+states|global|corporate|hq|official)\s*\)
      | \b(?:corporate|external)\s+careers?\b
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# EDGAR appends the state of incorporation to a registrant's name as a
# slash-delimited code: "NORTHROP GRUMMAN CORP /DE/", "WELLS FARGO &
# COMPANY/MN". It is metadata, not part of the name, and leaving it in
# costs real matches — it drags "northrop grumman corp de" away from the
# tenant's "northrop grumman", handing the win to a defunct CIK that
# happens to carry the untagged historical name.
_EDGAR_STATE_TAG = re.compile(r"\s*/\s*[a-z]{2,3}\s*/?\s*$", re.IGNORECASE)

_PUNCT_TO_SPACE = re.compile(r"[\-\u2010-\u2015_/\\|,;:+*~\"'`\[\]{}()<>!?#@$%^]")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]")

# Suffix stripping must never reduce a name to nothing but articles.
# "The Limited Inc" -> "the limited" -> "the" would otherwise fuzzy-match
# every company whose core name starts with an article.
_STOPWORDS = frozenset({"the", "a", "an", "of", "and", "for", "at", "on", "in"})

# "&" and "and" are interchangeable across sources ("AT&T" / "AT and T",
# "Johnson & Johnson" / "Johnson and Johnson").
_AMPERSAND = re.compile(r"\s*&\s*")


def strip_accents(text: str) -> str:
    """Fold accented Latin characters to ASCII ("Nestlé" -> "Nestle")."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def display_name(raw: str) -> str:
    """Human-readable cleanup: collapse whitespace, drop ATS decorations."""
    if not raw:
        return ""
    text = _WS.sub(" ", str(raw)).strip()
    # Applied repeatedly: "Acme Careers Page Jobs" sheds one tail per pass.
    for _ in range(3):
        stripped = _ATS_DECORATIONS.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text or _WS.sub(" ", str(raw)).strip()


def strip_legal_suffix(core: str) -> str:
    """Remove trailing corporate-form tokens from an already-cored name.

    Iterative, so "Acme Robotics Inc Ltd" reduces to "acme robotics".
    Two guards keep it from over-stripping: the last remaining token is
    never removed ("Limited" and "Group" are real company names), and a
    strip that would leave only articles is rejected ("The Limited Inc"
    stays "the limited" rather than collapsing to "the").

    Removing a suffix can expose a dangling conjunction — "Wells Fargo &
    Company" becomes "wells fargo and" — so trailing joining words are
    cleaned up afterwards.
    """
    text = core
    for _ in range(4):
        tokens = text.split()
        if len(tokens) < 2:
            break
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            parts = suffix.split()
            if len(tokens) > len(parts) and tokens[-len(parts) :] == parts:
                remainder = tokens[: -len(parts)]
                if all(tok in _STOPWORDS for tok in remainder):
                    continue
                text = " ".join(remainder)
                changed = True
                break
        if not changed:
            break

    tokens = text.split()
    while len(tokens) > 1 and tokens[-1] in _STOPWORDS:
        tokens.pop()
    return " ".join(tokens).strip() or core


def core_name(raw: str, *, keep_suffix: bool = False) -> str:
    """Aggressive normal form used for fuzzy scoring.

    Set ``keep_suffix`` to retain the corporate form, which disambiguates
    the rare pair where the suffix is the only difference.
    """
    if not raw:
        return ""
    text = strip_accents(display_name(raw)).lower()
    text = _EDGAR_STATE_TAG.sub("", text).rstrip("/")
    text = _AMPERSAND.sub(" and ", text)
    text = text.replace(".", "")
    text = _PUNCT_TO_SPACE.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    if not keep_suffix:
        text = strip_legal_suffix(text)
    return text


def name_key(raw: str, *, keep_suffix: bool = False) -> str:
    """Exact-match blocking key: ``core_name`` reduced to ``[a-z0-9]``."""
    return _NON_ALNUM.sub("", core_name(raw, keep_suffix=keep_suffix))


# Suffix forms with their internal spaces removed, longest first, so
# "limitedliabilitycompany" is tried before "company".
_CONCATENATED_SUFFIXES: tuple[str, ...] = tuple(
    sorted({suffix.replace(" ", "") for suffix in _LEGAL_SUFFIXES}, key=len, reverse=True)
)

# What must survive a strip. Shorter than this and the remainder matches
# too much to be worth blocking on ("alstonco" -> "alston" is useful,
# "hexco" -> "hex" is not).
_MIN_DESUFFIXED_LEN = 5


def desuffix_concatenated(key: str) -> list[str]:
    """Legal-suffix-stripped forms of a run-together blocking key.

    :func:`strip_legal_suffix` splits on whitespace, so it cannot touch a
    key that arrived as one unspaced token. That is the normal shape of a
    Workday tenant id: ``columbiasportswearcompany`` keeps its "company"
    forever and never reaches EDGAR's ``columbiasportswear``, which is
    the only place that tenant's real identity exists.

    Two passes, because a name can carry two forms
    ("...holdingsinc"). The results are *additional* candidates, never
    replacements — stripping is a guess ("clydeco" is a real brand, not
    "clyde" plus a suffix) and the score threshold is what decides.
    """
    out: list[str] = []
    current = key
    for _ in range(2):
        for suffix in _CONCATENATED_SUFFIXES:
            if (
                current.endswith(suffix)
                and len(current) - len(suffix) >= _MIN_DESUFFIXED_LEN
            ):
                current = current[: -len(suffix)]
                out.append(current)
                break
        else:
            break
    return out


def normalize_domain(value: str | None) -> str:
    """Reduce a URL or hostname to a bare lowercase registrable domain.

    Drops scheme, credentials, port, path, and a leading ``www.``.
    Returns ``""`` for anything that is not plausibly a hostname.
    """
    if not value:
        return ""
    text = str(value).strip().lower()
    if not text or text in {"-", "n/a", "none", "null"}:
        return ""
    if "//" not in text:
        text = "//" + text
    host = urlsplit(text).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    if "." not in host or " " in host:
        return ""
    return host


# Hosts that serve ATS boards rather than a company's own site. A domain
# extracted from a careers URL is only useful if it is *not* one of
# these, so `resolve` uses this to decide whether the URL carries signal.
ATS_HOSTS: frozenset[str] = frozenset(
    {
        "job-boards.greenhouse.io",
        "boards.greenhouse.io",
        "jobs.ashbyhq.com",
        "jobs.lever.co",
        "apply.workable.com",
        "jobs.workable.com",
        "myworkdayjobs.com",
        "eightfold.ai",
        "jobs.personio.com",
        "jobs.personio.de",
        "recruitee.com",
        "teamtailor.com",
        "breezy.hr",
        "applytojob.com",
        "bamboohr.com",
        "smartrecruiters.com",
        "icims.com",
        "jobvite.com",
        "taleo.net",
        "successfactors.com",
        "oraclecloud.com",
        "avature.net",
        "phenompeople.com",
        "join.com",
        "pinpointhq.com",
        "rippling-ats.com",
        "gupy.io",
        "zhiye.com",
        "mokahr.com",
        "darwinbox.in",
        "dayforcehcm.com",
        "ultipro.com",
        "paylocity.com",
        "adp.com",
        "cornerstoneondemand.com",
        "pageuppeople.com",
        "jazz.co",
        "applicantpro.com",
        "gem.com",
        "csod.com",
        "recruiterbox.com",
        "hire.trakstar.com",
        "trakstar.com",
        "jobs.jobvite.com",
        "myworkdaysite.com",
        "wd1.myworkdayjobs.com",
        "eu.greenhouse.io",
        "jobs.eu.lever.co",
        "hrmdirect.com",
        "silkroad.com",
        "brassring.com",
        "peoplefluent.com",
        "clearcompany.com",
        "workforcenow.adp.com",
        "paycomonline.net",
        "paycor.com",
        "isolvedhire.com",
        "jobs.smartrecruiters.com",
        "ashbyhq.com",
        "greenhouse.io",
        "lever.co",
        "workable.com",
        "personio.com",
        "workday.com",
    }
)


def is_ats_host(host: str) -> bool:
    """True when ``host`` (or a parent domain) is a known ATS board host."""
    if not host:
        return False
    if host in ATS_HOSTS:
        return True
    parts = host.split(".")
    return any(".".join(parts[i:]) in ATS_HOSTS for i in range(1, len(parts) - 1))


# Subdomains companies put their careers site on. Stripping them turns
# "careers.cintas.com" into "cintas.com", which is what PDL indexes.
_CAREERS_SUBDOMAINS = frozenset(
    {
        "careers",
        "career",
        "jobs",
        "job",
        "apply",
        "applications",
        "opportunities",
        "opportunity",
        "recruiting",
        "recruitment",
        "hiring",
        "talent",
        "work",
        "join",
        "bewerben",
        "empleo",
        "emplois",
        "karriere",
        "candidate",
        "candidates",
        "search",
        "www",
    }
)

# Multi-part public suffixes where the registrable domain needs three
# labels, not two. Not exhaustive — just the ones the cohort hits.
_MULTIPART_TLDS = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "co.nz", "co.za",
        "com.au", "com.br", "com.mx", "com.sg", "co.in", "com.cn",
    }
)


def corporate_domain(value: str | None) -> str:
    """Best-effort corporate domain from a careers URL or hostname.

    Returns ``""`` for ATS-hosted URLs, which carry no company identity.
    Otherwise peels careers-style subdomains down to the registrable
    domain: ``https://opportunities.alnylam.com/x`` -> ``alnylam.com``.
    """
    host = normalize_domain(value)
    if not host or is_ats_host(host):
        return ""
    parts = host.split(".")
    # Keep at least the registrable domain plus its suffix.
    floor = 3 if ".".join(parts[-2:]) in _MULTIPART_TLDS else 2
    while len(parts) > floor and parts[0] in _CAREERS_SUBDOMAINS:
        parts = parts[1:]
    return ".".join(parts)

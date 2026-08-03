"""Team size, separating a broad band from a trustworthy exact count.

This is the weakest of the three enrichments and the schema is built to
say so out loud. Three columns, never merged:

``employee_count_band``
    Broad coverage, low trust. PDL's ``size`` is a self-reported
    LinkedIn-style bucket ("51-200"). It exists for essentially every
    matched company, and it is not ground truth.

``employee_count``
    Narrow coverage, high trust. An exact integer, only ever written
    from a source that filed or published a specific number, with the
    date it referred to. Never derived from the band — turning "51-200"
    into 125 invents precision that was never there.

``employee_count_floor``
    A lower bound, from :mod:`.osha` or :mod:`.form5500`, whichever files
    the larger number. Establishments in high-hazard industries report an
    annual average employee count to OSHA, and employers sponsoring a
    benefit plan report active participants to DOL; neither can exceed
    the company's real headcount. Both are incomplete in their own way —
    OSHA misses a company's unreported sites, DOL misses staff who are
    not plan members — so neither is ever promoted into
    ``employee_count``. Form 5500 is the only one of the two that reaches
    private companies at scale, which is most of the cohort. Because a
    floor is filed independently of the exact sources, it is also the only
    check here that can catch a bad ``employee_count`` — see
    :func:`_check_against_floor`.

Exact sources, in the order they win ties:

1. **10-K** — Item 1 "Human Capital" states headcount as of fiscal year
   end. Filed under securities law, so it is the best number available
   for a public company. The SEC does not tag it in XBRL
   (``dei:EntityNumberOfEmployees`` returns 404 for the companies here),
   so it is parsed from the document text.
2. **Wikidata P1128** (CC0) — curated, point-in-time, with the
   qualifier date attached. Good for notable companies of any listing
   status.
3. **Reg CF / Reg A filings** — ``CURRENTEMPLOYEES`` and
   ``FULLTIMEEMPLOYEES``, self-reported to the SEC by small issuers.

LinkedIn is deliberately not scraped. That was the exposure that ended
Proxycurl, and it is the same category of risk that ruled out
Crunchbase for this project in the first place.

When several exact sources disagree the most recent as-of date wins,
because headcount moves fast; source rank only breaks ties within the
same date.

The band is stale by construction
---------------------------------
PDL's free extract is a snapshot and publishes no per-record
observation date, so ``employee_count_band_as_of`` carries the date the
extract was ingested, not the date anyone counted. It is an upper bound
on freshness. The band goes wrong in two ways that follow from this:
a company outgrows it (Phasor Engineering reads "11-50" here and
201-500 on LinkedIn today) or it renames and leaves a shell behind
(PDL still carries "anthem, inc." at "1-10" beside a live "elevance
health" at 10001+).

Neither is detectable from the band alone, so a band is **withdrawn**
whenever an independent source contradicts it outright, with the reason
recorded in ``employee_count_band_conflict``: a filed exact count far
outside it, an OSHA floor above it, a registrant in the wrong state, or
a stock listing behind a band of fifty. Roughly 2% of bands go this
way. A missing band costs a consumer a filter; a wrong one — "1-10" for
Columbia Sportswear — reads as an answer and gets believed.

Accuracy
--------
Coverage is 67.1% for the band, 7.7% for an exact count and 4.7% for an
OSHA floor. Where an exact count and a band both survive they agree 87%
of the time; the rest are surfaced as
``employee_count_agrees_with_band = false`` and are usually a wrong
identity match rather than a wrong number, so the flag doubles as a
check on :mod:`.resolve`.

The 10-K parser is measured separately against filings whose answer is
known, in ``_headcount_check.py`` (13/13 at the time of writing,
including two filings that state no company-wide total, where returning
nothing is the only right answer). That harness exists because every
regression here has been a *confident wrong number* — customers served
read as staff, a table-of-contents page number, a street address, a
percentage of the workforce — and a silent wrong integer is much worse
than a null.
"""

from __future__ import annotations

import json
import logging
import re

import polars as pl

from pipeline.company_enrichment import config, sechttp
from pipeline.company_enrichment.normalize import name_key, normalize_domain

logger = logging.getLogger(__name__)

SOURCE_RANK = {"sec_10k": 3, "wikidata": 2, "sec_form_c": 1, "sec_reg_a": 1}

# Inclusive bounds of each PDL band, used only to sanity-check an exact
# count against the independent band. Generous on both sides: the band is
# self-reported and often stale, so this catches gross contradictions
# (a "1-10" company reporting 40,000 staff) rather than policing edges.
BAND_BOUNDS: dict[str, tuple[int, int]] = {
    "1-10": (1, 10),
    "11-50": (11, 50),
    "51-200": (51, 200),
    "201-500": (201, 500),
    "501-1000": (501, 1000),
    "1001-5000": (1001, 5000),
    "5001-10000": (5001, 10000),
    "10001+": (10001, 3_000_000),
}
# How far outside its band an exact count may sit before it is called a
# contradiction.
_BAND_SLACK = 3.0

WIKIDATA_QUERY = """
SELECT ?company ?companyLabel ?employees ?asOf ?cik ?website WHERE {
  ?company wdt:P17 wd:Q30 .
  ?company p:P1128 ?stmt .
  ?stmt ps:P1128 ?employees .
  OPTIONAL { ?stmt pq:P585 ?asOf . }
  OPTIONAL { ?company wdt:P5531 ?cik . }
  OPTIONAL { ?company wdt:P856 ?website . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# A 10-K states headcount in either word order, so both are matched:
#   "we had approximately 7,834 full-time employees"
#   "Boeing's total workforce was approximately 182,000"
_STAFF_NOUN = (
    r"(?:employee[\s-]?partners?|employees|team\s?members|teammates|colleagues|"
    r"crew\s?members?|associates|people|staff|workforce|headcount|personnel)"
)
_QUALIFIER = r"(?:approximately|about|nearly|over|more\s+than|roughly|some)?\s*"
_DIGITS = r"[0-9][0-9,]{1,9}"
_NUMBER = rf"({_DIGITS})"

# The same hedges as `_QUALIFIER`, but as a bare alternation for the
# leading-context checks, which need to allow one without consuming the
# whitespace around it. "around" is included here and not there because a
# lead ending "a team of around" must match, while widening the patterns
# that select candidates in the first place is a separate question.
_APPROX = (
    r"(?:approximately|approx\.?|about|around|nearly|over|more\s+than|roughly|some)"
)

# A footnote marker can land between the number and the noun: Intel's
# "workforce, which was comprised of 85,100 1 people" is a reference to
# footnote 1, not part of the count.
_FOOTNOTE = r"(?:\s+\d{1,2})?"

_STAFF_MODIFIER = (
    r"(?:full[\s-]?time|part[\s-]?time|regular|permanent|salaried|"
    r"global|worldwide|total|internal|active)"
)

_COUNT_THEN_NOUN = re.compile(
    rf"(?i)\b{_QUALIFIER}{_NUMBER}{_FOOTNOTE}\s+"
    rf"(?:{_STAFF_MODIFIER}\s+)*{_STAFF_NOUN}\b"
)

# A total is often split across two components sharing one noun:
# "approximately 63,600 full-time and 1,800 part-time employees". Only the
# second number can reach the noun through the pattern above, so Brink's
# was published at 1,800 against 8,339 filed with DOL and NiSource at its
# 70 part-timers against 7,982. This captures the first component so both
# become candidates and the larger wins.
#
# It has to be a separate pattern rather than an optional bridge inside
# `_COUNT_THEN_NOUN`: a bridge consumes the second number, and `finditer`
# resumes past it, so the component stated second would stop being a
# candidate at all. Dick's leads with the smaller of its two.
#
# Summing the components would be nearer the literal total, but "total" is
# itself an allowed modifier, so "5,000 total and 300 part-time employees"
# would double-count. The largest stated component is the safer claim.
_COMPOUND_FIRST = re.compile(
    rf"(?i)\b{_QUALIFIER}{_NUMBER}{_FOOTNOTE}\s+(?:{_STAFF_MODIFIER}\s+)+"
    rf"and\s+{_DIGITS}\s+(?:{_STAFF_MODIFIER}\s+)*{_STAFF_NOUN}\b"
)

# Connectives between the noun and its count. Kept as an explicit
# alternation rather than a pile of optional groups so that at least one
# is always required — otherwise "Employees 12" in a table of contents
# would match.
_CONNECTIVE = (
    r"(?:"
    r"(?:,?\s+(?:which|that))?\s+(?:was|were|is|are|has|had|have)"
    r"(?:\s+(?:comprised|composed|consisted|comprising|consisting))?(?:\s+of)?"
    r"|\s+(?:comprised|composed|consisted|comprising|consisting|totaling|"
    r"totalling|numbering)\s+of"
    r"|\s+(?:totaled|totalled|numbered|included|reached)"
    r"|\s+(?:of|at)"
    r")"
)
_NOUN_THEN_COUNT = re.compile(
    rf"(?i)\b{_STAFF_NOUN}\b{_CONNECTIVE}\s+{_QUALIFIER}{_NUMBER}\b"
)

# Three shapes of text look exactly like a headcount and never are. Each
# is vetoed outright rather than scored down, because no amount of
# surrounding context redeems them.
#
# A table of contents: "Our Service Offerings 7 Sales 11 Members 11
# Management Information Systems 11 Employees and Human Capital 12".
# Recognised by a run of "word <small number>" pairs before the match.
_TOC_PAIR = re.compile(r"[A-Za-z]{3,}\s+\d{1,3}(?=\s)")
# Customers rather than staff: "we serve nearly 500,000 people living
# with diabetes". Only "people" is ambiguous this way.
_SERVED_NOT_STAFFED = re.compile(
    r"(?i)^\s*(?:living|served|serviced|using|who\s+(?:use|used|rely)|"
    r"across\s+the\s+globe\s+living|worldwide\s+living)\b"
)
_SERVE_VERB = re.compile(r"(?i)\b(?:serve[sd]?|serving|support(?:s|ed|ing)?|reach(?:es|ed)?)\b")
# A share of the workforce rather than a count of it.
_PERCENT_TAIL = re.compile(r"^\s*(?:%|percent\b)")
# A street address: "located at 109 Associates Blvd.".
_ADDRESS_TAIL = re.compile(
    r"(?i)^\s*(?:blvd|boulevard|ave|avenue|st|street|rd|road|dr|drive|way|ln|lane|"
    r"ct|court|cir|circle|pkwy|parkway|plaza|place|pl|suite|ste|building|bldg)\b"
)

# Scoring bonus rather than a requirement: language about employing
# people corroborates a candidate, but plenty of legitimate totals go
# without it ("Headquartered in Reston, Virginia, with 47,000 global
# employees"). The bare noun "employees" is excluded because it is the
# thing being matched and would corroborate everything.
_EMPLOYMENT_CUE = re.compile(
    r"(?i)\b(?:employ(?:s|ed|ing|ment)?|workforce|headcount|payroll|"
    r"we\s+had|we\s+have|company\s+had|corporation\s+had|consisted\s+of|"
    r"staffed\s+by|team\s+(?:of|comprised)|hired)\b"
)

# "approximately 2.1 million associates" — Walmart states the biggest
# headcount in the cohort in words, so the magnitude is folded into the
# digits before matching.
_MAGNITUDE = re.compile(r"(?i)\b(\d{1,3}(?:\.\d{1,3})?)\s+(million|billion)\b")
_MAGNITUDES = {"million": 1_000_000, "billion": 1_000_000_000}


def _expand_magnitudes(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = float(match.group(1)) * _MAGNITUDES[match.group(2).lower()]
        return f"{int(value)}"

    return _MAGNITUDE.sub(replace, text)

# Phrases that mark a number as a *subset* of the workforce rather than
# the total. Boeing's 10-K puts "approximately 72,000 employees ... were
# union represented" a few sentences from its 182,000 total.
#
# The layoff family needed every inflection, not just the noun. Matching
# only "reduction" missed "reductions" on the plural, and missed
# "reduced" and "reducing" entirely, which is how three headline numbers
# came to be the size of a layoff rather than of a company: Nelnet's 220
# "associates were impacted" by "workforce reductions" against 7,432
# filed, Beyond Meat's "reduced our workforce by approximately 65", and
# ServiceTitan's "reducing the Company's workforce by 221".
_SUBSET_CONTEXT = re.compile(
    r"(?i)\b(?:union|represented|collective bargaining|outside of|located in|"
    r"located outside|completed|leveraged|participated|enrolled|hired|"
    r"terminated|severance|laid off|option|award|grant|share|stock|"
    r"plan|segment|subsidiary|joint venture|customers|suppliers|shareholders|"
    r"holders of record|square feet|hours"
    r"|reduc(?:e|es|ed|ing|tion|tions)"
    r"|restructur(?:e|es|ed|ing)"
    r"|impacted|affected|eliminated|furloughed"
    r")\b"
)

# "14,500 full-time internal staff, including approximately 7,100
# employees engaged directly in Protiviti operations" — an inclusion
# immediately before the count marks it as a part of a total just stated.
#
# This has to be anchored to the end of the leading window, unlike the
# subset phrases above. "Including" in open prose is the *opposite*
# signal: it normally introduces a breakdown of the total, as in "9,525
# employees ... including" or "we had 10,000 employees, including". A
# loose match cost Nasdaq, Organon, DXC, Moelis, Life360, and Take-Two
# their correct totals, so only an inclusion abutting the number counts.
_INCLUSION_LEAD = re.compile(
    rf"(?i)\b(?:includ(?:ing|es|ed|e)|of\s+which|among\s+(?:them|whom))\s+"
    rf"(?:a\s+total\s+of\s*)?(?:{_APPROX}\s*)?$"
)

# A departmental qualifier immediately after the noun means the number
# counts one org, not the company: "3,900 employees in our research and
# development organization" against "8,100 employees operating across 35
# countries". Nothing but the text abutting the match distinguishes these,
# so this is checked in its own narrow trailing window, and
# `_DEPARTMENT_LEAD` below does the same on the other side.
_DEPARTMENT_QUALIFIER = re.compile(
    r"(?i)^\s*(?:in|within|at|of)\s+(?:our|the|its)?\s*"
    r"(?:[a-z&,\s-]{0,40}?)"
    r"(?:organization|organisation|department|division|team|segment|group|"
    r"function|unit|subsidiary|region|office|facility|plant|store|"
    r"sales|marketing|engineering|research|manufacturing|operations|support)"
)

# The identical qualifier just as often sits *before* the count, and
# scoping it only from the trailing side was the parser's largest blind
# spot: 10x Genomics' "our commercial organization consisted of 473 full
# time employees" was published as the company total against 977 filed
# with DOL, and Groupon's "leading a team of around 350 people" against
# 2,948. Anchored to the end of the leading window so that only a
# qualifier abutting the count scores against it. Scored rather than
# vetoed, like its trailing twin, because "our global organization of
# 47,000 employees" uses the same words for the whole company.
_DEPARTMENT_LEAD = re.compile(
    r"(?i)\b(?:a|an|our|the|its|their|his|her)\s+"
    r"(?:[a-z&,'\u2019\s-]{0,30}?)"
    r"(?:organization|organisation|department|division|team|segment|group|"
    r"function|unit)\s+"
    r"(?:(?:consisted|comprised|composed|consisting|comprising|totaling|"
    r"totalling|numbering)\s+)?"
    rf"of\s*(?:{_APPROX}\s*)?$"
)

# A one-word functional qualifier can also sit directly against the staff
# noun, in which case the match itself starts mid-phrase and the
# qualifier is the last thing in the leading window: Swift's "we had a
# sales staff of approximately 200" is its sales team, not its 31,022
# employees. Kept to unmistakably departmental words — nouns like
# "support" or "service" appear too often as ordinary verbs.
_DEPARTMENT_MODIFIER = re.compile(
    r"(?i)\b(?:sales|marketing|engineering|research|development|manufacturing|"
    r"production|operations|commercial|administrative|corporate|logistics|"
    r"clinical|regulatory|editorial|warehouse)\s+$"
)

# "employees were 55 years old or older" is a demographic disclosure, and
# the age reads as a headcount to any pattern that accepts a number after
# a staff noun. Ameren was published at 55 employees against 9,197 filed.
_AGE_TAIL = re.compile(
    r"(?i)^\s*years?\s+(?:old|of\s+age|or\s+older|or\s+younger|and\s+(?:older|younger))"
)
# A partitive construction is always a subset: "of the approximately
# 1,700 employees serving in the U.S. in management roles". Requiring a
# determiner after "of" keeps it from firing on "consisted of
# approximately 48,000 employees", which is a total.
_PARTITIVE = re.compile(
    r"(?i)\b(?:of|among|amongst|within)\s+(?:the|our|its|these|those)\s*$"
)
# "Total" is only evidence of a company-wide count when it is totalling
# people. ServiceTitan's layoff of 221 employees is followed by "total
# pre-tax charges", and that stray bonus was enough to lift a layoff to a
# non-negative score and have it published as the headcount.
_TOTAL_CONTEXT = re.compile(
    r"(?i)\b(?:total|worldwide|globally|in total|overall)\b"
    r"(?!\s+(?:pre-?tax|charges?|costs?|expenses?|revenues?|assets|liabilities|"
    r"debt|borrowings|compensation|net|operating|purchase|contract|backlog|"
    r"amounts?|payments?|proceeds|shares?|equity|capitalization)\b)"
)
_ASOF_CONTEXT = re.compile(r"(?i)\bas of\b")
# Anchors for the section that actually discusses headcount.
_SECTION_ANCHOR = re.compile(
    r"(?i)\b(?:human capital|our (?:employees|people|workforce|team)\b|"
    r"employees and human capital|item\s+1\.?\s*business)\b"
)

_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"<[^>]+>")

# Sanity envelope: below this a "headcount" is usually a share count or a
# section number; above it, the largest US employers top out near 2.3M.
_MIN_PLAUSIBLE = 2
_MAX_PLAUSIBLE = 3_000_000


def _clean_text(raw_html: str) -> str:
    import html as html_module

    text = _SCRIPT_RE.sub(" ", raw_html)
    text = _ANY_TAG_RE.sub(" ", text)
    # Entities must be decoded, not stripped: "Boeing&#8217;s total
    # workforce" has to become readable prose for the context scoring.
    text = html_module.unescape(text)
    return _expand_magnitudes(" ".join(text.split()))


def _vetoed(matched: str, leading: str, trailing: str) -> bool:
    """True when the match cannot be a headcount whatever its context."""
    # A percentage of the workforce, not a count of it: "our U.S.
    # workforce consisted of approximately 48% individuals identifying
    # as White".
    if _PERCENT_TAIL.match(trailing):
        return True
    # An age, not a count of people.
    if _AGE_TAIL.match(trailing):
        return True
    if len(_TOC_PAIR.findall(leading)) >= 2:
        return True
    if re.search(r"(?i)\bpeople\b", matched) and (
        _SERVED_NOT_STAFFED.match(trailing) or _SERVE_VERB.search(leading)
    ):
        return True
    return bool(
        re.search(r"(?i)\bassociates\b", matched) and _ADDRESS_TAIL.match(trailing)
    )


def parse_headcount(raw_html: str) -> int | None:
    """Extract a total headcount from a 10-K, or ``None`` if none is safe.

    Every mention of a staff count is scored on its surroundings rather
    than taking the first or the largest. Being inside the human-capital
    discussion and being described as a "total" push a candidate up;
    sitting next to union, equity-award, or facility language pushes it
    down, since those numbers are subsets. Among equally-scored
    candidates the largest wins, because a total is never smaller than
    its parts.

    Both sides of the match are read. Scoring only the trailing side was
    the parser's largest source of wrong headcounts, because the phrase
    that scopes a number to one department, one layoff, or one line of a
    breakdown usually comes before it: "our commercial organization
    consisted of 473", "reducing the Company's workforce by 221",
    "including approximately 7,100". Across the 288 filings whose parse
    can be checked against a floor filed with DOL, reading the leading
    side cut counts contradicted by that floor from 22 to 14, and counts
    wrong by more than 2x from 13 to 4, while giving up 3 of 473 values.
    """
    text = _clean_text(raw_html)
    if not text:
        return None

    anchors = [m.start() for m in _SECTION_ANCHOR.finditer(text)]

    candidates: list[tuple[int, int]] = []  # (score, value)
    for pattern in (_COUNT_THEN_NOUN, _NOUN_THEN_COUNT, _COMPOUND_FIRST):
        for match in pattern.finditer(text):
            try:
                value = int(match.group(1).replace(",", ""))
            except ValueError:
                continue
            if not (_MIN_PLAUSIBLE <= value <= _MAX_PLAUSIBLE):
                continue

            leading = text[max(0, match.start() - 140) : match.start()]
            trailing = text[match.end() : match.end() + 90]
            if _vetoed(match.group(0), leading, trailing):
                continue

            score = 0
            if _EMPLOYMENT_CUE.search(leading) or _EMPLOYMENT_CUE.search(trailing):
                score += 50
            # Closest thing to a topic sentence: the first count stated
            # right after the "Human Capital" heading is the total.
            if any(anchor <= match.start() <= anchor + 400 for anchor in anchors):
                score += 120
            elif any(anchor <= match.start() <= anchor + 8000 for anchor in anchors):
                score += 60
            if _TOTAL_CONTEXT.search(leading) or _TOTAL_CONTEXT.search(trailing):
                score += 40
            if _ASOF_CONTEXT.search(leading):
                score += 30
            if _DEPARTMENT_QUALIFIER.match(trailing) or _DEPARTMENT_LEAD.search(
                leading
            ):
                score -= 150
            if _DEPARTMENT_MODIFIER.search(leading):
                score -= 150
            if _SUBSET_CONTEXT.search(leading) or _SUBSET_CONTEXT.search(trailing):
                score -= 90
            if _PARTITIVE.search(leading) or _INCLUSION_LEAD.search(leading):
                score -= 140
            candidates.append((score, value))

    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    # A candidate that only ever appears in subset context is not a total.
    if best_score < 0:
        return None
    # Take the largest value among the near-best candidates rather than
    # the single top-scored one. Scoring separates totals from subsets
    # reliably but ranks two legitimate phrasings of the same total
    # arbitrarily, and between two totals the larger is the company-wide
    # one.
    threshold = best_score - 40
    return max(value for score, value in candidates if score >= threshold)


def _latest_10k(cik: int) -> tuple[str, str] | None:
    """(document URL, filing date) for the newest 10-K, if any."""
    try:
        payload = json.loads(
            sechttp.get(config.SEC_SUBMISSIONS_TEMPLATE.format(cik=cik), suffix=".json")
        )
    except Exception as exc:
        logger.debug("submissions lookup failed for CIK %s: %s", cik, exc)
        return None
    recent = payload.get("filings", {}).get("recent", {})
    for form, accession, document, date in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("reportDate", recent.get("filingDate", [])),
        strict=False,
    ):
        if form == "10-K" and document:
            stripped = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{stripped}/{document}",
                str(date or ""),
            )
    return None


def headcounts_from_10k(ciks: list[int]) -> pl.DataFrame:
    """Parse headcount out of the latest 10-K for each CIK."""
    records: list[dict[str, object]] = []
    for index, cik in enumerate(ciks, start=1):
        if index % 50 == 0:
            logger.info("10-K headcount: %d/%d", index, len(ciks))
        found = _latest_10k(cik)
        if not found:
            continue
        url, date = found
        try:
            html = sechttp.get(url, suffix=".htm").decode("utf-8", errors="replace")
        except Exception:
            continue
        count = parse_headcount(html)
        if count is None:
            continue
        records.append(
            {
                "cik": cik,
                "employee_count": count,
                "employee_count_as_of": date[:10] if date else None,
                "employee_count_source": "sec_10k",
            }
        )
    schema = {
        "cik": pl.Int64,
        "employee_count": pl.Int64,
        "employee_count_as_of": pl.String,
        "employee_count_source": pl.String,
    }
    return pl.DataFrame(records, schema=schema) if records else pl.DataFrame(schema=schema)


def fetch_wikidata(*, force: bool = False) -> pl.DataFrame:
    """US companies with a P1128 headcount, latest statement per company."""
    cache = config.CACHE_DIR / "wikidata_employees.json"
    if cache.exists() and not force:
        payload = json.loads(cache.read_text())
    else:
        import httpx

        logger.info("querying Wikidata for P1128 employee counts")
        response = httpx.get(
            config.WIKIDATA_SPARQL_URL,
            params={"query": WIKIDATA_QUERY},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": config.SEC_USER_AGENT,
            },
            timeout=300.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    rows = payload["results"]["bindings"]
    df = pl.DataFrame(
        {
            "wikidata_id": [r["company"]["value"].rsplit("/", 1)[-1] for r in rows],
            "wikidata_label": [r.get("companyLabel", {}).get("value") for r in rows],
            "employee_count": [
                _safe_int(r.get("employees", {}).get("value")) for r in rows
            ],
            "employee_count_as_of": [
                (r.get("asOf", {}).get("value") or "")[:10] or None for r in rows
            ],
            "cik": [_safe_int(r.get("cik", {}).get("value")) for r in rows],
            "website": [r.get("website", {}).get("value") for r in rows],
        }
    ).filter(pl.col("employee_count").is_not_null())

    # P1128 is a time series; only the newest statement is current. Where
    # two statements carry the same date, the larger count wins.
    latest = (
        df.sort(
            ["employee_count_as_of", "employee_count"],
            descending=[True, True],
            nulls_last=True,
        )
        .unique(subset=["wikidata_id"], keep="first", maintain_order=True)
        .with_columns(
            pl.lit("wikidata").alias("employee_count_source"),
            pl.col("website")
            .map_elements(normalize_domain, return_dtype=pl.String)
            .alias("domain"),
            pl.col("wikidata_label")
            .map_elements(name_key, return_dtype=pl.String)
            .alias("name_key_core"),
        )
    )
    logger.info("wikidata: %d companies with a current headcount", latest.height)
    return latest


def _largest_per_key(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    """One row per ``key``, keeping the largest ``employee_count``.

    Neither of the keys Wikidata is matched on is unique. "Verizon" carries
    both 132,200 and 87,000, "7-Eleven" both 76,029 and 20,700, and 21 name
    keys collide this way. The largest wins: an ATS careers board belongs to
    the parent far more often than to a same-named subsidiary, and a total
    is never smaller than its parts. Date and Wikidata id settle exact ties
    so the answer never depends on where a row happened to sit.

    ``maintain_order`` is what makes the sort stick. Polars' ``unique`` may
    reorder, so ``keep="first"`` without it keeps an arbitrary row whatever
    the preceding sort said. That is how a Hampton Inn franchise carried
    Hilton Worldwide's 182,000 in one run and Hilton Grand Vacations' 15,000
    in the next, with nothing about either company having changed — only
    unrelated rows elsewhere in the frame.
    """
    return frame.sort(
        ["employee_count", "employee_count_as_of", "wikidata_id"],
        descending=[True, True, False],
        nulls_last=True,
    ).unique(subset=[key], keep="first", maintain_order=True)


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _exact_from_exempt() -> pl.DataFrame:
    """Latest self-reported headcount from Reg CF / Reg A filings."""
    from pipeline.company_enrichment import exempt

    try:
        offerings = exempt.ingest()
    except Exception as exc:
        logger.warning("skipping Reg CF/Reg A headcounts: %s", exc)
        return pl.DataFrame(
            schema={
                "cik": pl.Int64,
                "employee_count": pl.Int64,
                "employee_count_as_of": pl.String,
                "employee_count_source": pl.String,
            }
        )
    return (
        offerings.filter(
            pl.col("employee_count").is_not_null() & pl.col("cik").is_not_null()
        )
        .sort(
            ["filing_date", "employee_count"],
            descending=[True, True],
            nulls_last=True,
        )
        .unique(subset=["cik"], keep="first", maintain_order=True)
        .select(
            "cik",
            pl.col("employee_count").cast(pl.Int64),
            pl.col("filing_date").cast(pl.String).alias("employee_count_as_of"),
            "source",
        )
        .rename({"source": "employee_count_source"})
    )


_FLOOR_COLUMNS = (
    "employee_count_floor",
    "employee_count_floor_source",
    "employee_count_floor_as_of",
)

_EMPTY_FLOOR_SCHEMA: dict[str, pl.DataType] = {
    "ats": pl.String,
    "slug": pl.String,
    "employee_count_floor": pl.Int64,
    "employee_count_floor_source": pl.String,
    "employee_count_floor_as_of": pl.String,
}


def _posting_states(result: pl.DataFrame) -> pl.DataFrame:
    """The state a tenant's own postings sit in, for corroborating a match."""
    cohort = _read_optional(config.COHORT_PARQUET, "cohort")
    if cohort.is_empty():
        return result.select("ats", "slug", pl.lit("").alias("_posting_state"))

    from pipeline.company_enrichment.formd import _state_from_location

    return cohort.select(
        "ats",
        "slug",
        pl.col("sample_location")
        .map_elements(_state_from_location, return_dtype=pl.String)
        .alias("_posting_state"),
    )


def _osha_candidate(result: pl.DataFrame, posting_state: pl.DataFrame) -> pl.DataFrame:
    """OSHA establishment floors, keeping only corroborated name matches.

    The floor is matched on a name key, so it inherits the collision
    problem that dogs every name match here: an ``ashby/stronghold``
    board picks up an unrelated Stronghold that filed 3,399 employees.
    Two independent checks decide whether a match is worth believing,
    and a floor that fails them is dropped rather than published:

    * If the tenant's own postings name a state, that state must appear
      in the employer's OSHA footprint. A New York psychiatry practice
      is not the Athena that files in three other states.
    * Otherwise the employer must have filed for more than one
      establishment. A single generic-named site is the shape of every
      false positive in the cohort; a multi-site filer is a real
      employer whose name is doing more work than one word.
    """
    floor = _read_optional(config.OSHA_PARQUET, "osha floor")
    if floor.is_empty():
        return pl.DataFrame(schema=_EMPTY_FLOOR_SCHEMA)

    joined = result.join(
        floor.select(
            "name_key_core", *_FLOOR_COLUMNS, "osha_states", "osha_establishments"
        ),
        on="name_key_core",
        how="inner",
    ).join(posting_state, on=["ats", "slug"], how="left")

    known_state = pl.col("_posting_state").fill_null("") != ""
    corroborated = (
        pl.when(known_state)
        .then(pl.col("osha_states").list.contains(pl.col("_posting_state")))
        .otherwise(pl.col("osha_establishments") > 1)
    )
    return joined.filter(
        pl.col("employee_count_floor").is_not_null() & corroborated.fill_null(False)
    ).select("ats", "slug", *_FLOOR_COLUMNS)


def _form5500_candidate(result: pl.DataFrame, posting_state: pl.DataFrame) -> pl.DataFrame:
    """DOL Form 5500 participant floors, by EIN where possible.

    Two paths, and they are not equally trustworthy:

    The **EIN path** joins :mod:`.registrant`'s EDGAR-sourced EIN to the
    plan sponsor's EIN. That is an exact identifier match on both sides,
    the only one in this package, so it needs no corroboration — there is
    no name to collide.

    The **name path** exists because a tenant with no CIK has no EIN, and
    that is most of the cohort. It carries the usual collision risk, so it
    is guarded like the OSHA floor above: the name must belong to exactly
    one sponsor EIN across all 920k of them, and then either the tenant's
    posting state matches the sponsor's or the sponsor filed more than one
    plan. Note the state signal is weaker here than for OSHA, because DOL
    gives the sponsor's single mailing state rather than a multi-site
    footprint, so a multi-state employer can fail it and be dropped.
    """
    from pipeline.company_enrichment import form5500, registrant

    floor = form5500.load()
    if floor.is_empty():
        return pl.DataFrame(schema=_EMPTY_FLOOR_SCHEMA)

    frames: list[pl.DataFrame] = []

    profiles = registrant.load()
    if not profiles.is_empty():
        frames.append(
            result.filter(pl.col("cik").is_not_null())
            .join(
                profiles.filter(pl.col("ein").is_not_null()).select("cik", "ein"),
                on="cik",
                how="inner",
            )
            .join(floor.select("ein", "dol_participants", "dol_as_of"), on="ein", how="inner")
            .select(
                "ats",
                "slug",
                pl.col("dol_participants").alias("employee_count_floor"),
                pl.lit("dol_5500_ein").alias("employee_count_floor_source"),
                pl.col("dol_as_of").alias("employee_count_floor_as_of"),
            )
        )

    unambiguous = form5500.by_name(floor).filter(pl.col("dol_ein_count") == 1)
    if not unambiguous.is_empty():
        joined = result.join(unambiguous, on="name_key_core", how="inner").join(
            posting_state, on=["ats", "slug"], how="left"
        )
        known_state = pl.col("_posting_state").fill_null("") != ""
        corroborated = (
            pl.when(known_state)
            .then(pl.col("dol_states").list.contains(pl.col("_posting_state")))
            .otherwise(pl.col("dol_plans") > 1)
        )
        frames.append(
            joined.filter(
                pl.col("dol_participants").is_not_null() & corroborated.fill_null(False)
            ).select(
                "ats",
                "slug",
                pl.col("dol_participants").alias("employee_count_floor"),
                pl.lit("dol_5500_name").alias("employee_count_floor_source"),
                pl.col("dol_as_of").alias("employee_count_floor_as_of"),
            )
        )

    usable = [f for f in frames if not f.is_empty()]
    if not usable:
        return pl.DataFrame(schema=_EMPTY_FLOOR_SCHEMA)
    return pl.concat(usable, how="vertical")


def _attach_floor(result: pl.DataFrame) -> pl.DataFrame:
    """Attach the tightest corroborated lower bound on headcount.

    Two independent filing systems offer one: OSHA establishment reports
    and DOL Form 5500 plan filings. Both are floors, so where both exist
    the **larger** is the tightest true bound and wins; the surviving
    ``employee_count_floor_source`` says which system it came from. Taking
    the larger is only sound because each is separately a lower bound —
    the same reason neither may ever be promoted into ``employee_count``.
    """
    posting_state = _posting_states(result)
    candidates = [
        _osha_candidate(result, posting_state),
        _form5500_candidate(result, posting_state),
    ]
    usable = [c for c in candidates if not c.is_empty()]
    if not usable:
        return result.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("employee_count_floor"),
            pl.lit(None, dtype=pl.String).alias("employee_count_floor_source"),
            pl.lit(None, dtype=pl.String).alias("employee_count_floor_as_of"),
        )

    # The largest floor wins — every candidate is a filed lower bound, so
    # the highest one is the most informative and they cannot contradict
    # each other. Source name breaks ties so an EIN match outranks a name
    # match at equal size, and `maintain_order` keeps both decisions from
    # being undone by the dedup.
    best = (
        pl.concat(usable, how="vertical")
        .sort(
            ["employee_count_floor", "employee_count_floor_source"],
            descending=[True, False],
            nulls_last=True,
        )
        .unique(subset=["ats", "slug"], keep="first", maintain_order=True)
    )
    return result.join(best, on=["ats", "slug"], how="left")


# A floor may sit slightly above a published count without either being
# wrong: plan participants can include people who left partway through the
# year, and the two numbers rarely share an as-of date. Beyond this
# margin the two cannot both describe the same company.
_FLOOR_TOLERANCE = 1.15

# The margin at which a contradiction stops being arguable and the exact
# count is withdrawn rather than merely flagged. Every EIN-path case above
# it turned out to be a real error in the published count: Comcast at
# 24,000 against 93,051 filed, Figma's Wikidata entry at 324 against
# 1,305.
_FLOOR_REFUTES_COUNT = 2.0

# Only a floor matched on an exact identifier may withdraw a filed count.
# A name-matched floor is the wrong tool for overruling a 10-K.
_EXACT_IDENTITY_FLOOR_SOURCE = "dol_5500_ein"


def _check_against_floor(result: pl.DataFrame) -> pl.DataFrame:
    """Withdraw exact headcounts that a filed floor proves impossible.

    A floor and an exact count are independent filings, so the floor is
    the only check in this package that can catch a bad ``employee_count``.
    It withdraws 11: a 10-K stating one segment or one line of a breakdown
    (Comcast "24,000" against 93,051 filed, Werner "1,251" against 9,919),
    a stale Wikidata entry (Figma "324" against 1,305), and Reg CF issuers
    self-reporting at a seed raise (Caylent "3" against 296). None of that
    is visible from inside the source that produced it.

    It withdrew twice as many until :func:`parse_headcount` learned to read
    the context *leading* a number, which is the better place to fix a
    parse error — a withdrawal loses the row, a correct parse keeps it.
    Seven of the eleven left are staleness or self-report, which no parser
    can reach.

    This has to run *before* the band check, because a wrong exact count
    does double damage: it is published, and it then withdraws the correct
    PDL band for disagreeing with it. Nelnet's real band, 5001-10000, was
    being dropped in favour of 550 employees.

    ``employee_count_above_floor`` records the comparison for every row
    where both exist, so a contradiction too small to act on is still
    visible to a reviewer.
    """
    if "employee_count_floor" not in result.columns:
        return result.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("employee_count_above_floor")
        )

    comparable = pl.col("employee_count").is_not_null() & pl.col(
        "employee_count_floor"
    ).is_not_null()
    result = result.with_columns(
        pl.when(comparable)
        .then(
            pl.col("employee_count_floor")
            <= pl.col("employee_count") * _FLOOR_TOLERANCE
        )
        .otherwise(None)
        .alias("employee_count_above_floor")
    )

    refuted = (
        comparable
        & (pl.col("employee_count_floor_source") == _EXACT_IDENTITY_FLOOR_SOURCE)
        & (
            pl.col("employee_count_floor")
            > pl.col("employee_count") * _FLOOR_REFUTES_COUNT
        )
    )
    dropped = result.filter(refuted).height
    if dropped:
        logger.info("withdrew %d exact headcounts refuted by a filed floor", dropped)

    return result.with_columns(
        pl.when(refuted).then(None).otherwise(pl.col(column)).alias(column)
        for column in (
            "employee_count",
            "employee_count_source",
            "employee_count_as_of",
            "employee_count_agrees_with_band",
            # Cleared with the count it describes: a withdrawn count is
            # not a count sitting below a floor, and leaving the
            # comparison behind would flag rows that need no review.
            "employee_count_above_floor",
        )
    )


# Reasons a band is withdrawn, in the order they are reported. Each is a
# case of an *independent* source contradicting PDL, never a judgement
# about the band on its own.
_BAND_CONFLICTS: tuple[tuple[str, str], ...] = (
    ("exact_count", "a filed exact headcount sits far outside the band"),
    ("filed_floor", "filed establishment or plan-participant counts exceed the band"),
    ("registrant_state", "the matched registrant is in a different state"),
    ("listed_registrant", "a listed registrant cannot have fifty staff"),
)

# A band whose top is at or below this cannot describe a company with
# its own stock listing and market capitalisation.
_MAX_LISTED_BAND_TOP = 50


def _suppress_contradicted_bands(result: pl.DataFrame) -> pl.DataFrame:
    """Withdraw bands that an independent source contradicts.

    A wrong band is worse than a missing one. It travels with a company
    name into whatever reads this table, and unlike a wrong exact count
    it looks unremarkable — "1-10" for Anthem or Columbia Sportswear
    reads as a real answer. Where a second source disagrees outright,
    the band is dropped and the reason recorded in
    ``employee_count_band_conflict`` so the withdrawal is auditable.
    """
    conditions = {
        "exact_count": pl.col("employee_count_agrees_with_band").eq(False),
        "filed_floor": (
            pl.col("employee_count_floor").is_not_null()
            & pl.col("_band_high").is_not_null()
            & (pl.col("employee_count_floor") > pl.col("_band_high") * _BAND_SLACK)
        ),
        "registrant_state": pl.col("registrant_state_agrees").eq(False),
        "listed_registrant": (
            pl.col("ticker").is_not_null()
            & pl.col("market_cap_basis").eq("self")
            & (pl.col("_band_high") <= _MAX_LISTED_BAND_TOP)
        ),
    }
    has_band = pl.col("employee_count_band").is_not_null()
    conflict = (
        pl.concat_list(
            pl.when(has_band & conditions[label]).then(pl.lit(label)).otherwise(pl.lit(None))
            for label, _ in _BAND_CONFLICTS
        )
        .list.drop_nulls()
        .list.join(",")
    )

    result = result.with_columns(conflict.alias("employee_count_band_conflict"))
    contested = pl.col("employee_count_band_conflict") != ""
    return result.with_columns(
        pl.when(contested).then(None).otherwise(pl.col(column)).alias(column)
        for column in (
            "employee_count_band",
            "employee_count_band_source",
            "employee_count_band_as_of",
        )
    )


def _read_optional(path: object, label: str) -> pl.DataFrame:
    """Read a stage output, or an empty frame if it has not been built."""
    from pathlib import Path

    path = Path(str(path))
    if not path.exists():
        logger.warning("%s missing (%s) — continuing without it", label, path)
        return pl.DataFrame()
    return pl.read_parquet(path)


def _market_cap_signals(keys: pl.DataFrame) -> pl.DataFrame:
    """The two market-cap columns the band cross-check needs."""
    marketcap = _read_optional(config.MARKETCAP_PARQUET, "marketcap")
    if marketcap.is_empty():
        return keys.with_columns(
            pl.lit(None, dtype=pl.Boolean).alias("registrant_state_agrees"),
            pl.lit(None, dtype=pl.String).alias("market_cap_basis"),
        )
    return keys.join(
        marketcap.select("ats", "slug", "registrant_state_agrees", "market_cap_basis"),
        on=["ats", "slug"],
        how="left",
    )


def run(*, use_10k: bool = True) -> pl.DataFrame:
    config.ensure_dirs()
    resolved = pl.read_parquet(config.RESOLVED_PARQUET)

    # --- band baseline ----------------------------------------------------
    base = resolved.select(
        "ats",
        "slug",
        "cik",
        "domain",
        "ticker",
        pl.col("name").map_elements(name_key, return_dtype=pl.String).alias("name_key_core"),
        pl.col("size").alias("employee_count_band"),
        pl.when(pl.col("size").is_not_null())
        .then(pl.lit("pdl"))
        .otherwise(None)
        .alias("employee_count_band_source"),
    )

    # --- exact sources ----------------------------------------------------
    exact_frames: list[pl.DataFrame] = [_exact_from_exempt()]

    if use_10k:
        public_ciks = sorted(
            {
                int(c)
                for c in resolved.filter(
                    pl.col("cik").is_not_null() & pl.col("ticker").is_not_null()
                )["cik"]
            }
        )
        logger.info("parsing 10-K headcount for %d public companies", len(public_ciks))
        exact_frames.append(headcounts_from_10k(public_ciks))

    wikidata = fetch_wikidata()
    wd_by_cik = wikidata.filter(pl.col("cik").is_not_null()).select(
        "cik", "employee_count", "employee_count_as_of", "employee_count_source"
    )
    exact_frames.append(wd_by_cik)

    by_cik = pl.concat([f for f in exact_frames if not f.is_empty()], how="vertical")

    # Wikidata rows without a CIK still match on domain or name.
    wd_columns = ("employee_count", "employee_count_as_of", "employee_count_source")
    wd_by_domain = _largest_per_key(
        wikidata.filter(pl.col("domain") != ""), "domain"
    ).select("domain", *wd_columns)
    wd_by_name = _largest_per_key(
        wikidata.filter(pl.col("name_key_core") != ""), "name_key_core"
    ).select("name_key_core", *wd_columns)

    candidates = pl.concat(
        [
            base.join(by_cik, on="cik", how="inner"),
            base.filter(pl.col("domain").is_not_null() & (pl.col("domain") != "")).join(
                wd_by_domain, on="domain", how="inner"
            ),
            base.join(wd_by_name, on="name_key_core", how="inner"),
        ],
        how="diagonal",
    )

    if candidates.is_empty():
        result = base.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("employee_count"),
            pl.lit(None, dtype=pl.String).alias("employee_count_as_of"),
            pl.lit(None, dtype=pl.String).alias("employee_count_source"),
        )
    else:
        best = (
            candidates.with_columns(
                pl.col("employee_count_source")
                .replace_strict(SOURCE_RANK, default=0, return_dtype=pl.Int32)
                .alias("_rank")
            )
            # Freshness first, then the source ranking, and only then the
            # count. Size is the last word rather than the first on
            # purpose: between a current 10-K and a stale Wikidata entry
            # the filing wins even when it is the smaller number. Size
            # settles what is left, so two sources that are equally fresh
            # and equally trusted resolve the same way every run.
            .sort(
                ["employee_count_as_of", "_rank", "employee_count"],
                descending=[True, True, True],
                nulls_last=True,
            )
            .unique(subset=["ats", "slug"], keep="first", maintain_order=True)
            .select(
                "ats", "slug", "employee_count", "employee_count_as_of",
                "employee_count_source",
            )
        )
        result = base.join(best, on=["ats", "slug"], how="left")

    from pipeline.company_enrichment import pdl

    result = (
        result.with_columns(
            # The band is as fresh as the PDL extract, which is not
            # today. See `pdl.snapshot_date`.
            pl.lit(pdl.snapshot_date(), dtype=pl.String).alias(
                "employee_count_band_as_of"
            ),
            pl.col("employee_count_band")
            .replace_strict(
                {band: bounds[0] for band, bounds in BAND_BOUNDS.items()},
                default=None,
                return_dtype=pl.Int64,
            )
            .alias("_band_low"),
            pl.col("employee_count_band")
            .replace_strict(
                {band: bounds[1] for band, bounds in BAND_BOUNDS.items()},
                default=None,
                return_dtype=pl.Int64,
            )
            .alias("_band_high"),
        )
        .with_columns(
            pl.when(
                pl.col("employee_count").is_null() | pl.col("_band_low").is_null()
            )
            .then(None)
            .otherwise(
                (pl.col("employee_count") >= pl.col("_band_low") / _BAND_SLACK)
                & (pl.col("employee_count") <= pl.col("_band_high") * _BAND_SLACK)
            )
            .alias("employee_count_agrees_with_band")
        )
    )

    # --- cross-source checks on the band ----------------------------------
    result = _attach_floor(result).drop("name_key_core")
    result = _check_against_floor(result)
    result = _market_cap_signals(result)
    offered_band = result.filter(pl.col("employee_count_band").is_not_null()).height
    result = _suppress_contradicted_bands(result).drop(
        "_band_low", "_band_high", "registrant_state_agrees", "market_cap_basis", "ticker"
    )
    result.write_parquet(config.TEAMSIZE_PARQUET)

    total = result.height
    with_band = result.filter(pl.col("employee_count_band").is_not_null()).height
    with_exact = result.filter(pl.col("employee_count").is_not_null()).height
    with_floor = result.filter(pl.col("employee_count_floor").is_not_null()).height
    both = result.filter(pl.col("employee_count_agrees_with_band").is_not_null())
    print("\n=== Team size ===")
    print(f"Tenants                      : {total:>6,}")
    print(f"With a size band (PDL)       : {with_band:>6,}  ({with_band / total:.1%})")
    print(f"With an exact headcount      : {with_exact:>6,}  ({with_exact / total:.1%})")
    print(f"With a filed floor           : {with_floor:>6,}  ({with_floor / total:.1%})")
    if with_floor:
        print("  by source:")
        for source, n in (
            result.filter(pl.col("employee_count_floor").is_not_null())
            .group_by("employee_count_floor_source")
            .len()
            .sort("len", descending=True)
            .iter_rows()
        ):
            print(f"    {source:<16}: {n:>6,}")

    withdrawn = result.filter(pl.col("employee_count_band_conflict") != "")
    print(
        f"\nBands withdrawn as contradicted: {withdrawn.height:,} of "
        f"{offered_band:,} offered by PDL"
    )
    if withdrawn.height:
        counts = (
            withdrawn.select(pl.col("employee_count_band_conflict").str.split(","))
            .explode("employee_count_band_conflict")
            .group_by("employee_count_band_conflict")
            .len()
            .sort("len", descending=True)
        )
        reasons = dict(_BAND_CONFLICTS)
        for label, n in counts.iter_rows():
            print(f"  {label:<18}: {n:>5,}  ({reasons.get(label, '')})")
    print("\nExact count by source:")
    print(
        result.filter(pl.col("employee_count").is_not_null())
        .group_by("employee_count_source")
        .agg(pl.len().alias("tenants"), pl.col("employee_count").median().alias("median"))
        .sort("tenants", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    if both.height:
        agree = both.filter(pl.col("employee_count_agrees_with_band")).height
        print(
            f"\nExact vs PDL band (both present, n={both.height:,}): "
            f"{agree:,} consistent ({agree / both.height:.1%})"
        )
    print(f"\nwrote {config.TEAMSIZE_PARQUET}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

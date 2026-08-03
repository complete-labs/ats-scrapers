"""Tests for the derived enrichment helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ats_scrapers.enrichment.derived import (
    SENIORITY_LEVELS,
    infer_is_remote,
    infer_seniority,
    parse_salary_block,
    parse_salary_range,
)

# --- infer_is_remote --------------------------------------------------------
#
# Title-only inference; never returns False (absence of keyword in
# title is not evidence the role is on-site, LLM downstream fills
# that nuance).


@pytest.mark.parametrize(
    "title",
    [
        "Remote Software Engineer",
        "remote backend developer",
        "Anywhere — Senior Engineer",
        "Distributed Systems Engineer (Remote)",
        "Work from home — Customer Success",
        "WFH Sales Rep",
        "Telework Researcher",
    ],
)
def test_remote_keywords_in_title_detected(title: str) -> None:
    assert infer_is_remote(title) is True


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Customer Success Manager",
        "Backend Engineer, NYC",
        "Onsite Recruiter — SF",  # no remote keyword in title; we don't infer False from "onsite"
        "In-office Designer",  # ditto — never return False
    ],
)
def test_titles_without_remote_marker_return_none(title: str) -> None:
    """Never assert False from heuristic. The absence of a remote
    keyword is not evidence of on-site."""
    assert infer_is_remote(title) is None


@pytest.mark.parametrize("title", ["", "   ", "\t"])
def test_empty_title_returns_none(title: str) -> None:
    assert infer_is_remote(title) is None


# --- infer_seniority --------------------------------------------------------
#
# Audit finding 09: seniority was 74.88% Unknown. Title-only, and it
# stays silent whenever the title names no rank.


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Software Engineering Intern", "INTERN"),
        ("Werkstudent Marketing (m/w/d)", "INTERN"),
        ("Stagiaire Développeur", "INTERN"),
        ("Chief Technology Officer", "EXECUTIVE"),
        ("VP of Engineering", "EXECUTIVE"),
        ("SVP, Global Sales", "EXECUTIVE"),
        ("Director of Engineering", "DIRECTOR"),
        ("Head of Data", "DIRECTOR"),
        ("Engineering Manager", "MANAGER"),
        ("Principal Engineer", "PRINCIPAL"),
        ("Staff Software Engineer", "STAFF"),
        ("Tech Lead, Payments", "LEAD"),
        ("Senior Software Engineer", "SENIOR"),
        ("Sr. Data Scientist", "SENIOR"),
        ("Mid-Level Accountant", "MID"),
        ("Junior Developer", "JUNIOR"),
        ("Graduate Software Engineer", "JUNIOR"),
    ],
)
def test_seniority_reads_the_rank_from_the_title(title: str, level: str) -> None:
    assert infer_seniority(title) == level


@pytest.mark.parametrize(
    ("title", "level"),
    [
        # Management rank outranks the IC qualifier in front of it.
        ("Senior Director, Sales", "DIRECTOR"),
        ("Senior Engineering Manager", "MANAGER"),
        # ... but "Senior Product Manager" is a senior IC, because
        # "Product Manager" names a function rather than a rank.
        ("Senior Product Manager", "SENIOR"),
        # An internship is an internship whatever it shadows.
        ("Intern - Product Management", "INTERN"),
        # The tech IC ladder above "senior".
        ("Senior Staff Engineer", "STAFF"),
    ],
)
def test_seniority_resolves_competing_markers_by_precedence(
    title: str, level: str
) -> None:
    assert infer_seniority(title) == level


@pytest.mark.parametrize(
    "title",
    [
        # "Manager" naming the thing managed, not people.
        "Product Manager", "Program Manager", "Project Manager",
        "Account Manager", "Community Manager", "Customer Success Manager",
        # "Staff" as employment class — a staff nurse is entry grade.
        "Staff Nurse", "Staff Accountant", "Wait Staff",
        # "Principal" as a job in its own right.
        "School Principal", "Assistant Principal", "Principal Investigator",
        # "Director" naming a craft.
        "Art Director", "Creative Director", "Funeral Director",
        # "Senior" describing the clients.
        "Senior Living Nurse", "Senior Care Aide", "Senior Services Coordinator",
        # "Lead" as sales pipeline vocabulary.
        "Lead Generation Specialist", "Lead Gen Analyst",
        # "Junior"/"Graduate" naming a school.
        "Junior High Teacher", "Graduate School Advisor",
        # Coordinator role that borrows the word "chief".
        "Chief of Staff",
    ],
)
def test_seniority_ignores_words_that_only_look_like_ranks(title: str) -> None:
    """A wrong level is worse than no level.

    Each of these carries a rank word used in a non-rank sense; scoring
    them would move real postings into the wrong pay band invisibly.
    """
    assert infer_seniority(title) is None


@pytest.mark.parametrize(
    "title",
    ["Software Engineer", "Nurse", "Data Analyst", "Barista", "Accountant"],
)
def test_seniority_stays_silent_on_unqualified_titles(title: str) -> None:
    """Baseline titles omit the qualifier, and the baseline differs by
    employer — calling these MID would invent a fact."""
    assert infer_seniority(title) is None


@pytest.mark.parametrize("value", [None, "", "   ", math.nan, 0, [], object()])
def test_seniority_survives_junk_input(value: object) -> None:
    assert infer_seniority(value) is None


def test_seniority_only_returns_declared_levels() -> None:
    """The column is a closed vocabulary downstream faceting relies on."""
    titles = [
        "Software Engineering Intern", "Chief Technology Officer",
        "Director of Engineering", "Engineering Manager", "Principal Engineer",
        "Staff Software Engineer", "Tech Lead", "Senior Engineer",
        "Mid-Level Accountant", "Junior Developer",
    ]
    levels = {infer_seniority(t) for t in titles}
    assert levels <= set(SENIORITY_LEVELS)
    assert None not in levels


@pytest.mark.parametrize("value", [None, math.nan, 0, 12.5, [], {}, object()])
def test_non_string_values_return_none(value: object) -> None:
    """NaN and other non-string types must not crash the function."""
    assert infer_is_remote(value) is None


def test_handles_pandas_nan_in_series() -> None:
    """Regression: pandas .apply() passes NaN floats for empty cells."""
    series = pd.Series([
        "Remote Engineer",
        None,
        float("nan"),
        "Senior Manager",
    ])
    result = series.apply(infer_is_remote)
    assert result.tolist() == [True, None, None, None]


# --- parse_salary_range -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$257K - $335K", (257_000.0, 335_000.0)),
        ("CA$400K – CA$500K", (400_000.0, 500_000.0)),
        ("€80k–€120k", (80_000.0, 120_000.0)),
        ("$60K to $80K", (60_000.0, 80_000.0)),
        ("$200,000 - $300,000", (200_000.0, 300_000.0)),
        ("OTE $1.5M - $2M", (1_500_000.0, 2_000_000.0)),
        ("$100K", (100_000.0, 100_000.0)),
    ],
)
def test_parse_salary_range_known_formats(text: str, expected: tuple[float, float]) -> None:
    lo, hi = parse_salary_range(text)
    assert lo == pytest.approx(expected[0])
    assert hi == pytest.approx(expected[1])


@pytest.mark.parametrize(
    "text",
    [None, "", "Competitive", "Negotiable", "DOE", "Based on experience", float("nan")],
)
def test_parse_salary_range_returns_none_when_unparseable(text: object) -> None:
    assert parse_salary_range(text) == (None, None)


def test_parse_salary_range_swaps_inverted_bounds() -> None:
    """`max - min` should be normalized to `(min, max)`."""
    lo, hi = parse_salary_range("$300K - $200K")
    assert (lo, hi) == (200_000.0, 300_000.0)


# --- parse_salary_block -----------------------------------------------------
#
# Recovers the pay range that pay-transparency law put in the
# description body. Most ATSes expose no structured salary field, so
# this is the difference between a posting reporting no pay and
# reporting the range the employer actually advertised.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The ATS's own rendered widget, closed by an ISO code.
        ("Annual Salary: $265,000 — $365,000 USD", (265000.0, 365000.0, "USD", "YEAR")),
        ("Pay Range $182,000 — $250,208 USD", (182000.0, 250208.0, "USD", "YEAR")),
        ("Salary Range: £190,000 — £230,000 GBP", (190000.0, 230000.0, "GBP", "YEAR")),
        # Prose disclosure with no code, gated on an explicit label.
        (
            "The salary range for this role is $160,000 - $240,000.",
            (160000.0, 240000.0, "USD", "YEAR"),
        ),
        ("Estimated annual salary of $380,000 - $470,000", (380000.0, 470000.0, "USD", "YEAR")),
    ],
)
def test_parse_salary_block_reads_disclosed_ranges(text, expected) -> None:
    block = parse_salary_block(text)
    assert block is not None
    assert (block.min_amount, block.max_amount, block.currency, block.period) == expected


def test_parse_salary_block_reads_european_thousands_separators() -> None:
    """``€200.000`` is two hundred thousand, not two hundred.

    Greenhouse renders the employer's own number formatting, so a lone
    dot is a grouping separator as often as a decimal point.
    """
    block = parse_salary_block("Annual Salary: €200.000 — €255.000 EUR")
    assert block is not None
    assert (block.min_amount, block.max_amount, block.currency) == (
        200000.0, 255000.0, "EUR",
    )


def test_parse_salary_block_spans_every_zone_band() -> None:
    """Location-banded postings state several ranges; all of them count."""
    text = (
        "Zone 1 Pay Range $217,000 — $255,000 USD "
        "Zone 2 Pay Range $196,000 — $230,000 USD "
        "Zone 3 Pay Range $166,000 — $195,000 USD"
    )
    block = parse_salary_block(text)
    assert block is not None
    assert (block.min_amount, block.max_amount) == (166000.0, 255000.0)


def test_parse_salary_block_reads_the_range_across_html_tags() -> None:
    """The widget splits the range over elements and escapes the dash."""
    html = (
        "<div class='title'>Annual Salary:</div><div class='pay-range'>"
        "<span>$320,000</span><span class='divider'>&mdash;</span>"
        "<span>$405,000 USD</span></div>"
    )
    block = parse_salary_block(html)
    assert block is not None
    assert (block.min_amount, block.max_amount, block.period) == (
        320000.0, 405000.0, "YEAR",
    )


def test_parse_salary_block_keeps_hourly_rates_hourly() -> None:
    block = parse_salary_block("The hourly rate for this role is $25.00 — $32.00 USD")
    assert block is not None
    assert (block.min_amount, block.max_amount, block.period) == (25.0, 32.0, "HOUR")


def test_parse_salary_block_refuses_to_guess_an_unlabelled_period() -> None:
    """A bare small range could be hourly or annual, so make no claim.

    Assuming annual here is what turns an hourly rate into a
    multimillion-dollar salary once something annualizes it.
    """
    assert parse_salary_block("Pay Range $25.00 — $32.00 USD") is None


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "No compensation information in this posting.",
        # Money with no pay label: a referral bonus is not a salary.
        "Refer a friend and earn $500 - $1,000.",
    ],
)
def test_parse_salary_block_returns_none_without_a_disclosure(text) -> None:
    assert parse_salary_block(text) is None


def test_parse_salary_block_reads_a_point_value_as_a_degenerate_range() -> None:
    """Some employers publish a single figure rather than a band."""
    block = parse_salary_block("On Target Earnings $300,000 USA OTE")
    assert block is not None
    assert (block.min_amount, block.max_amount) == (300000.0, 300000.0)
    # "USA" and "OTE" are three capitals but not currencies, so the
    # symbol decides the unit.
    assert block.currency == "USD"


def test_parse_salary_block_prefers_the_iso_code_over_the_symbol() -> None:
    block = parse_salary_block("Annual Salary: $120,000 — $150,000 CAD")
    assert block is not None
    assert block.currency == "CAD"

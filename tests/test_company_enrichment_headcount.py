"""Tests for 10-K headcount extraction (`company_enrichment.teamsize`).

Every sentence here is real text from a filing the parser once read wrong,
or one it reads right and must keep reading right. The two groups matter
equally: each of these rules was first written broadly enough to fix its
own case while quietly breaking several others, and the only thing that
caught it was checking the parse against a headcount the same company
filed with the Department of Labor.

The snippets carry the "Human Capital" heading where the filing did,
because proximity to that heading is a large part of a candidate's score.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from pipeline.company_enrichment.teamsize import _largest_per_key, parse_headcount

# Context is read 140 characters back and 90 forward, so two figures in
# adjacent sentences see each other's context and pick up each other's
# penalties. A real 10-K puts its total and its layoff disclosure sections
# apart; this stands in for that distance, and without it these cases
# would be testing an overlap that never occurs.
_FILLER = (
    "We believe our culture is a competitive advantage, and we invest in "
    "training, development, and internal mobility for all colleagues."
)

# --- totals the parser must find --------------------------------------
#
# Guards against over-eager subset detection. Most of these are companies
# that lost a correct total to a rule that looked reasonable in isolation.
TOTALS = [
    pytest.param(
        "Human Capital As of December 31, 2025, we had approximately "
        "48,000 employees worldwide.",
        48_000,
        id="plain-total",
    ),
    pytest.param(
        "Human Capital Our headcount includes 12,909 full-time employees, "
        "including 4,000 in international locations.",
        12_909,
        id="total-followed-by-its-own-breakdown",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had 10,000 employees, "
        "including approximately 4,000 in the United States.",
        10_000,
        id="total-then-including",
    ),
    pytest.param(
        "DXC serves a global client base, including many Fortune 500 "
        "companies, supported by 115,000 employees in 70 countries.",
        115_000,
        id="including-modifies-something-else",
    ),
    pytest.param(
        "Human Capital Our workforce, which was comprised of 85,100 1 "
        "people, spans the globe.",
        85_100,
        id="footnote-marker-between-count-and-noun",
    ),
    pytest.param(
        "Human Capital We employed approximately 2.1 million associates "
        "as of January 31, 2026.",
        2_100_000,
        id="magnitude-stated-in-words",
    ),
    pytest.param(
        "Human Capital The Company had approximately 14,500 full-time "
        "internal staff, including approximately 7,100 employees engaged "
        "directly in Protiviti operations, as of December 31, 2025.",
        14_500,
        id="parent-beats-the-part-it-includes",
    ),
]


@pytest.mark.parametrize(("html", "expected"), TOTALS)
def test_finds_company_total(html: str, expected: int) -> None:
    assert parse_headcount(html) == expected


# --- split totals -----------------------------------------------------
#
# "N full-time and M part-time employees" shares one noun between two
# numbers. Only the second can reach the noun by the ordinary pattern, so
# the larger component is matched separately. Dick's states the smaller of
# its two first, which is why the fix cannot consume the second number.
SPLIT_TOTALS = [
    pytest.param(
        "Human Capital At December 31, 2025, our company had approximately "
        "63,600 full-time and 1,800 part-time employees.",
        63_600,
        id="larger-component-first",
    ),
    pytest.param(
        "Human Capital Workforce Composition. As of December 31, 2025, we "
        "had 7,668 full-time and 70 part-time active employees.",
        7_668,
        id="larger-component-first-extra-modifier",
    ),
    pytest.param(
        "Human Capital Management As of January 31, 2026, we employed "
        "approximately 31,600 full-time and 73,600 part-time employees.",
        73_600,
        id="larger-component-second",
    ),
]


@pytest.mark.parametrize(("html", "expected"), SPLIT_TOTALS)
def test_prefers_larger_half_of_a_split_total(html: str, expected: int) -> None:
    assert parse_headcount(html) == expected


# --- numbers that are not the company total ---------------------------
#
# Each shape below was published as a company headcount until the parser
# learned to read the words *leading* a number as well as those trailing
# it. Scoping is deliberately comparative rather than an outright veto,
# because the same words carry both meanings: "our team of 16,000" is
# Expedia's entire company, and vetoing that phrasing would throw away
# more good totals than it saves.
#
# So these cases give the *subset the larger number*. That is the only
# arrangement that tests the rule: with the total larger, "the largest of
# the near-best candidates wins" would pick it regardless of scoping, and
# the test would pass with the rule deleted. Ranking a scoped figure below
# a smaller unscoped one is also the behaviour worth pinning, since it is
# where a parse goes wrong in a filing that never states a plain total.
SUBSET_LOSES_TO_TOTAL = [
    pytest.param(
        "Human Capital As of December 31, 2025, we had 1,178 full time "
        "employees. Our commercial organization consisted of 2,400 full "
        "time employees, many with PhD degrees.",
        1_178,
        id="lead-departmental-organization",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had 2,948 employees. "
        "Drabek joined as CTO and is based in Munich, leading a "
        "distribution team of around 4,000 people.",
        2_948,
        id="lead-a-qualified-team-of",
    ),
    pytest.param(
        "Human Capital At December 31, 2025, we employed 31,022 people. We "
        "had a sales staff of approximately 40,000 individuals across the "
        "US and Mexico.",
        31_022,
        id="lead-functional-modifier-on-the-noun",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had 589 employees. "
        f"{_FILLER} In 2023, we reduced our workforce by approximately 800 "
        "employees.",
        589,
        id="lead-layoff-reduced",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had 2,442 employees. "
        f"{_FILLER} The Company aligned its investments with its strategic "
        "priorities by reducing the Company's workforce by 3,000 "
        "employees, incurring total pre-tax charges of $8 million.",
        2_442,
        id="lead-layoff-reducing",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had 7,432 associates. "
        f"{_FILLER} The Company announced workforce reductions. "
        "Approximately 9,000 associates were impacted during 2024.",
        7_432,
        id="lead-layoff-reductions-plural",
    ),
    pytest.param(
        "Human Capital As of December 31, 2025, we had approximately 8,100 "
        f"employees operating across 35 countries. {_FILLER} We employed "
        "9,300 employees in our research and development organization.",
        8_100,
        id="trailing-departmental-qualifier",
    ),
    pytest.param(
        "Human Capital The Company had approximately 14,500 full-time "
        "internal staff, including approximately 20,000 employees engaged "
        "directly in Protiviti operations.",
        14_500,
        id="lead-inclusion",
    ),
]


@pytest.mark.parametrize(("html", "expected"), SUBSET_LOSES_TO_TOTAL)
def test_ranks_a_scoped_figure_below_an_unscoped_total(html: str, expected: int) -> None:
    assert parse_headcount(html) == expected


def test_total_only_counts_when_it_is_totalling_people() -> None:
    """"Total" next to money must not corroborate a headcount.

    This is ServiceTitan's filing, whose layoff of 221 employees carried
    no total and sat outside the human-capital section, so the only thing
    keeping its score non-negative was a bonus awarded for the phrase
    "total pre-tax charges". It was published at 221 against 2,442 filed.
    """
    html = (
        "The Company aligned its investments more closely with its "
        "strategic priorities by reducing the Company's workforce by 221 "
        "employees. The Company incurred total pre-tax charges of "
        "approximately $8 million."
    )
    assert parse_headcount(html) is None


def test_a_bare_team_of_is_left_alone() -> None:
    """The counterpart to the cases above: not every "team" is a department.

    Of the four published parses that still rest on a "team of N" lead,
    all four are correct company totals — Expedia's 16,000 and Fortrea's
    14,300 among them. This is why the leading-context rules score rather
    than veto, and why the departmental word list stays narrow.
    """
    html = (
        "Human Capital Our team of 16,000 employees spans 30 countries as "
        "of December 31, 2025."
    )
    assert parse_headcount(html) == 16_000


def test_an_age_is_never_a_headcount() -> None:
    """Unlike scoping, this is an outright veto: no context redeems an age.

    Ameren was published at 55 employees against 9,197 filed with DOL,
    from "approximately 22% of Ameren's total employees were 55 years old
    or older" — a demographic disclosure in the human-capital section,
    which is the highest-scoring place a candidate can sit.
    """
    html = (
        "Human Capital As of December 31, 2025, approximately 22% of "
        "Ameren's total employees were 55 years old or older."
    )
    assert parse_headcount(html) is None


def test_returns_none_when_nothing_looks_like_a_headcount() -> None:
    assert parse_headcount("<html><body><p>No staff figures here.</p></body></html>") is None
    assert parse_headcount("") is None


# --- picking between same-named companies -----------------------------


def _wikidata(rows: list[tuple[str, str, int, str | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "wikidata_id": pl.String,
            "name_key_core": pl.String,
            "employee_count": pl.Int64,
            "employee_count_as_of": pl.String,
        },
        orient="row",
    )


def test_largest_of_two_same_named_companies_wins() -> None:
    """Wikidata keys are not unique, so a rule has to choose.

    "Verizon" is two entities carrying 132,200 and 87,000. A careers board
    under that name belongs to the parent far more often than to a
    same-named subsidiary, so the larger count is the better guess.
    """
    frame = _wikidata(
        [
            ("Q1", "verizon", 87_000, "2024-01-01"),
            ("Q2", "verizon", 132_200, "2024-01-01"),
        ]
    )
    picked = _largest_per_key(frame, "name_key_core")
    assert picked.height == 1
    assert picked["employee_count"][0] == 132_200


def test_equal_counts_fall_back_to_the_newer_statement() -> None:
    frame = _wikidata(
        [
            ("Q1", "acme", 500, "2021-01-01"),
            ("Q2", "acme", 500, "2025-06-01"),
        ]
    )
    assert _largest_per_key(frame, "name_key_core")["employee_count_as_of"][0] == "2025-06-01"


def test_a_missing_count_never_beats_a_present_one() -> None:
    frame = pl.DataFrame(
        {
            "wikidata_id": ["Q1", "Q2"],
            "name_key_core": ["acme", "acme"],
            "employee_count": [None, 40],
            "employee_count_as_of": ["2025-06-01", "2020-01-01"],
        },
        schema_overrides={"employee_count": pl.Int64},
    )
    assert _largest_per_key(frame, "name_key_core")["employee_count"][0] == 40


def test_the_winner_does_not_depend_on_input_row_order() -> None:
    """The property that was actually broken.

    Polars' ``unique`` may reorder, so ``keep="first"`` after a sort keeps an
    arbitrary row unless ``maintain_order`` is set. The visible symptom was a
    Hampton Inn franchise carrying Hilton Worldwide's 182,000 in one run and
    Hilton Grand Vacations' 15,000 in the next, decided by unrelated rows
    elsewhere in the frame.
    """
    rows = [
        ("Q1", "hilton", 15_000, "2025-01-01"),
        ("Q2", "hilton", 182_000, "2025-01-01"),
        ("Q3", "other", 7, "2025-01-01"),
        ("Q4", "other", 9, "2025-01-01"),
    ]
    for permutation in itertools.permutations(rows):
        picked = _largest_per_key(_wikidata(list(permutation)), "name_key_core").sort(
            "name_key_core"
        )
        assert picked["employee_count"].to_list() == [182_000, 9]

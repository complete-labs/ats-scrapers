"""Tests for name and domain normalisation (`company_enrichment.normalize`).

This module is the narrowest part of the funnel. Every other stage —
the PDL name track, the EDGAR name track, the Form 5500 floor join, the
Exhibit 21 subsidiary rollup — blocks on a key this module produces, so
a key that is too aggressive does not cause one bad match, it causes the
same bad match in four places at once.

The tests are therefore split by what a change to them would mean.
:func:`test_suffix_stripping_recovers_a_real_match` covers the reason
suffix stripping exists at all; ``COLLISIONS`` covers the price it
charges, and those cases are pinned deliberately rather than
incidentally — they are the mechanism behind a real mis-resolution and
must not change silently.
"""

from __future__ import annotations

import pytest

from pipeline.company_enrichment.normalize import (
    core_name,
    corporate_domain,
    desuffix_concatenated,
    display_name,
    is_ats_host,
    name_key,
    normalize_domain,
    strip_accents,
    strip_legal_suffix,
)

# --- what suffix stripping is for ---------------------------------------
#
# The reference tables store names literally. PDL has "leidos" where
# EDGAR has "LEIDOS HOLDINGS, INC.", and only a suffix-stripped key
# reaches both from one tenant.

RECOVERED = [
    pytest.param("Leidos Holdings Inc", "leidos", id="holdings-inc"),
    pytest.param("Acme Robotics Inc Ltd", "acmerobotics", id="two-stacked-suffixes"),
    pytest.param("Two Six Technologies Holdings Inc.", "twosixtechnologies", id="holdings-with-dot"),
    pytest.param("Nestlé S.A.", "nestle", id="accented-with-sa"),
    pytest.param("AT&T Inc.", "atandt", id="ampersand-becomes-and"),
    pytest.param("Wells Fargo & Company", "wellsfargo", id="dangling-conjunction-cleaned"),
    pytest.param("NORTHROP GRUMMAN CORP /DE/", "northropgrumman", id="edgar-state-tag"),
    pytest.param("PROJECT44 HOLDINGS CORP", "project44", id="digits-survive"),
]


@pytest.mark.parametrize(("raw", "expected"), RECOVERED)
def test_suffix_stripping_recovers_a_real_match(raw: str, expected: str) -> None:
    assert name_key(raw) == expected


# --- what suffix stripping costs ----------------------------------------
#
# "Holdings" and "Company" are stripped as legal suffixes, so a parent
# or subsidiary named after a common word reduces to that bare word and
# becomes indistinguishable from any tenant of the same name. This is
# the mechanism behind the Ashby tenant "Harvey" (a legal-AI company)
# resolving to Harvey Holdings, a North Carolina logistics firm, and to
# Harvey Holdings LLC in Waste Connections' Exhibit 21.
#
# These assertions pin current behaviour. They are not an endorsement of
# it: changing them is allowed, changing them by accident is not.

COLLISIONS = [
    pytest.param("HARVEY HOLDINGS, LLC", "harvey", id="harvey-holdings-llc"),
    pytest.param("HARVEY HOLDINGS INC", "harvey", id="harvey-holdings-inc"),
    pytest.param("Harvey", "harvey", id="harvey-bare"),
    pytest.param("Solutions Holdings, LLC", "solutions", id="solutions-holdings"),
    pytest.param("Rowan Companies, LLC", "rowan", id="rowan-companies"),
    pytest.param("Graphite Holdings LLC", "graphite", id="graphite-holdings"),
]


@pytest.mark.parametrize(("raw", "expected"), COLLISIONS)
def test_soft_suffix_collapses_to_a_bare_common_word(raw: str, expected: str) -> None:
    assert name_key(raw) == expected


def test_a_subsidiary_and_an_unrelated_tenant_share_one_key() -> None:
    """The collision itself, stated as one fact rather than two.

    Nothing downstream can tell these apart on the name alone, which is
    why the disambiguating signals have to be independent of it.
    """
    assert name_key("HARVEY HOLDINGS, LLC") == name_key("Harvey")


def test_the_suffixed_form_still_separates_them() -> None:
    """``keep_suffix`` is the escape hatch the collision cases need.

    Both key forms are emitted for every candidate, so a caller that
    wants to know whether a match survived only because of stripping can
    compare the two.
    """
    assert name_key("HARVEY HOLDINGS, LLC", keep_suffix=True) == "harveyholdingsllc"
    assert name_key("Harvey", keep_suffix=True) == "harvey"


# --- guards that stop stripping going too far ---------------------------


def test_the_last_token_is_never_stripped() -> None:
    """"Group" and "Holdings" are real company names on their own."""
    assert core_name("Group") == "group"
    assert core_name("Holdings") == "holdings"


def test_a_strip_leaving_only_articles_is_rejected() -> None:
    """"The Limited Inc" must not collapse to "the"."""
    assert strip_legal_suffix("the limited inc") == "the limited"


def test_stripping_is_iterative_but_bounded() -> None:
    assert strip_legal_suffix("acme robotics inc ltd") == "acme robotics"


def test_accents_fold_to_ascii() -> None:
    assert strip_accents("Nestlé") == "Nestle"


def test_display_name_sheds_ats_decorations() -> None:
    assert display_name("Acme Careers") == "Acme"
    assert display_name("Acme Careers Page") == "Acme"
    # A name that is *only* a decoration keeps its original text rather
    # than becoming empty.
    assert display_name("Careers") == "Careers"


# --- run-together keys --------------------------------------------------


def test_concatenated_suffix_is_stripped_when_enough_survives() -> None:
    """Workday tenant ids arrive as one token, which the token-based
    stripper cannot touch."""
    assert desuffix_concatenated("columbiasportswearcompany") == ["columbiasportswear"]
    assert desuffix_concatenated("alstonco") == ["alston"]


def test_a_short_remainder_is_not_worth_blocking_on() -> None:
    """"hexco" must not become "hex"."""
    assert desuffix_concatenated("hexco") == []


# --- domains ------------------------------------------------------------


def test_ats_hosts_carry_no_company_identity() -> None:
    assert corporate_domain("https://jobs.ashbyhq.com/harvey") == ""
    assert is_ats_host("job-boards.greenhouse.io")
    assert is_ats_host("acme.myworkdayjobs.com")
    assert not is_ats_host("alnylam.com")


def test_careers_subdomains_peel_to_the_registrable_domain() -> None:
    assert corporate_domain("https://opportunities.alnylam.com/x") == "alnylam.com"
    assert corporate_domain("https://careers.cintas.com") == "cintas.com"


def test_multipart_tlds_keep_three_labels() -> None:
    assert corporate_domain("https://careers.foo.co.uk/jobs") == "foo.co.uk"


@pytest.mark.parametrize("value", ["", "-", "n/a", "none", "null", "not a host"])
def test_junk_is_not_a_domain(value: str) -> None:
    assert normalize_domain(value) == ""

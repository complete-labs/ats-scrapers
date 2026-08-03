"""Tests for company description and careers URL extraction.

Every case here is a real observation. The accepted descriptions are
text these extractors actually returned from live pages and postings;
the rejected ones are the specific wrong answers each guard exists to
stop, found by reading the output of an unguarded run:

- the shared *suffix* of a tenant's postings is employment law, not a
  description (SpaceX's ITAR block, Crusoe's EEO paragraph)
- ``About the Role`` introduces the vacancy (Gem, Bohler)
- ``About the Company`` on a staffing board introduces the agency's
  *client* (Kimmel & Associates)
- link text alone picks a product page called "Opportunities"
  (Quantum Metric)
- ``resolve_careers_url`` reads Workable's short-link path ``/j/XXXX``
  as a tenant named "j"
"""

from __future__ import annotations

import pytest

from pipeline.company_enrichment import blurb, boilerplate, companysite, profile

# --- blurb: presentation ----------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            "America&#39;s leading health solutions company",
            "America's leading health solutions company",
            id="numeric-entity",
        ),
        pytest.param(
            "Optimize with Quantum Metric&#x27;s platform",
            "Optimize with Quantum Metric's platform",
            id="hex-entity",
        ),
        pytest.param(
            "Tanger shops brands &amp; outlets",
            "Tanger shops brands & outlets",
            id="named-entity",
        ),
        pytest.param("Acme&amp;#39;s platform", "Acme's platform", id="double-escaped"),
        pytest.param(
            "\u201cMineralys is a clinical-stage company.\u201d",
            "Mineralys is a clinical-stage company.",
            id="wrapping-smart-quotes",
        ),
        pytest.param(
            "**Bold** copy\n\nwith  hard\twraps", "Bold copy with hard wraps",
            id="markdown-and-whitespace",
        ),
        pytest.param(
            "zero\u200bwidth\ufeffmarks", "zerowidthmarks", id="invisible-characters"
        ),
    ],
)
def test_tidy_normalises_presentation(raw: str, expected: str) -> None:
    assert blurb.tidy(raw) == expected


def test_trim_cuts_on_a_sentence_boundary() -> None:
    text = "One sentence here. Two sentence here. Three here. Four here."
    assert blurb.trim(text, max_sentences=2) == "One sentence here. Two sentence here."


def test_trim_does_not_split_on_an_abbreviation() -> None:
    text = "Acme Inc. builds rockets for a living. And more."
    assert blurb.trim(text, max_sentences=1) == "Acme Inc. builds rockets for a living."


def test_trim_falls_back_to_a_word_boundary_when_one_sentence_is_over_budget() -> None:
    trimmed = blurb.trim("word " * 300, max_chars=100)
    assert len(trimmed) <= 101
    assert trimmed.endswith("\u2026")
    assert not trimmed.endswith("wor\u2026")


# --- blurb: rejection --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "Acme is an Equal Opportunity Employer and considers all applicants.",
            id="eeo-opening",
        ),
        pytest.param(
            "To conform to U.S. Government export regulations, applicant must be a "
            "U.S. citizen or national. ITAR restrictions apply to this role.",
            id="itar",
        ),
        pytest.param(
            "If reasonable accommodation is needed to participate in the job "
            "application process, please contact our Human Resources Team.",
            id="accommodation-notice",
        ),
    ],
)
def test_is_legalese_rejects_employment_law(text: str) -> None:
    assert blurb.is_legalese(text)


def test_is_legalese_tolerates_one_marker_far_from_the_start() -> None:
    # Genuine company copy that closes on a compliance line still counts
    # as a description, so a single late marker must not reject it.
    text = (
        "Meter is building the vertically integrated network company: hardware, "
        "software, services, ISP, data, and autonomous networks from the local "
        "network to the data center, for enterprises of every size everywhere. "
        "Meter is an Equal Opportunity Employer."
    )
    assert not blurb.is_legalese(text)


@pytest.mark.parametrize(
    "text",
    [
        "About the Role: we're hiring an Android Engineer for the mobile team.",
        "About This Role: a Senior Customer Success Manager to support customers.",
        "We are seeking a highly motivated Survey Technician to join our team.",
        "Position Summary: the Construction Security Manager is responsible for...",
        "Are you looking for a company you can grow your career with?",
        # An aggregator's disclaimer, mined from a Lever board.
        "This position is listed on behalf of a partner company, who manages "
        "all applications and next steps.",
        "This position will be posted for a minimum of 5 days.",
        "Pay Rate: $250/Hr, OT Rate: $265/Hr, Callback Rate: $265/Hr.",
        "Compensation: - Compensation is unique to each candidate.",
    ],
)
def test_is_role_copy_rejects_vacancy_prose(text: str) -> None:
    assert blurb.is_role_copy(text)


@pytest.mark.parametrize(
    "text",
    [
        "Crusoe is on a mission to accelerate the abundance of energy.",
        # The noun alone is too common to reject on: these are company
        # names, not sentences about a vacancy.
        "The Job Shop is a staffing company serving the Pacific Northwest.",
        "The Opportunity Network connects students with employers.",
    ],
)
def test_is_role_copy_accepts_company_prose(text: str) -> None:
    assert not blurb.is_role_copy(text)


# --- blurb: identity corroboration -------------------------------------


@pytest.mark.parametrize(
    ("text", "name"),
    [
        pytest.param(
            "Black Duck delivers True Scale Application Security to teams.",
            "BLACK DUCK SOFTWARE INC",
            id="leading-tokens-only",
        ),
        pytest.param(
            "America's leading health solutions company, CVS Health provides care.",
            "CVS HEALTH CORP",
            id="first-token-too-short-whole-name-present",
        ),
        pytest.param(
            "ER Meds partners with regional and rural hospitals.",
            "ER Meds",
            id="two-short-tokens",
        ),
        pytest.param("Nestle makes food.", "Nestl\u00e9 S.A.", id="accented-name"),
        pytest.param(
            "Leidos delivers solutions at the intersection of national security.",
            "Leidos Holdings, Inc.",
            id="legal-suffix-stripped",
        ),
        pytest.param(
            "Fetch's app helps millions of users browse rewards.",
            "Fetch",
            id="possessive",
        ),
    ],
)
def test_mentions_name_accepts(text: str, name: str) -> None:
    assert blurb.mentions_name(text, name)


def test_mentions_name_rejects_a_generic_shared_word() -> None:
    # "Health" is shared, "LifeStance" is not, and a reader could not
    # confirm the subject of this sentence either.
    assert not blurb.mentions_name(
        "We offer personalized mental health care through in-person therapy.",
        "LifeStance Health",
    )


# --- boilerplate: the shared prefix ------------------------------------

_SPACEX = (
    "SpaceX was founded under the belief that a future where humanity is out "
    "exploring the stars is fundamentally more exciting than one where we are "
    "not. Today SpaceX is actively developing the technologies to make this "
    "possible, with the ultimate goal of enabling human life on Mars.\n\n"
)
_ITAR = (
    "\n\nITAR REQUIREMENTS: To conform to U.S. Government export regulations, "
    "applicant must be a U.S. citizen or lawful permanent resident."
)


def _postings(prefix: str, bodies: list[str], suffix: str = "") -> list[str]:
    return [f"{prefix}{body}{suffix}" for body in bodies]


def test_shared_prefix_is_the_company_blurb() -> None:
    samples = _postings(
        _SPACEX,
        ["Propulsion Engineer duties.", "Avionics work.", "Launch operations."],
        _ITAR,
    )
    text, method = boilerplate.derive(samples, [], ("SpaceX",))
    assert method == "shared_prefix"
    assert text.startswith("SpaceX was founded")


def test_shared_suffix_is_not_used() -> None:
    # The postings share only the ITAR block, which is exactly the wrong
    # answer this rule exists to avoid.
    samples = _postings(
        "", ["Wildly different opening A.", "Opening B.", "Opening C."], _ITAR
    )
    assert boilerplate.derive(samples, [], ("SpaceX",)) == ("", "")


def test_shared_prefix_of_legalese_is_rejected() -> None:
    eeo = (
        "Acme Corp is an Equal Opportunity Employer. All qualified applicants "
        "will receive consideration without regard to race, color, or religion. "
    )
    samples = _postings(eeo, ["Role A body.", "Role B body.", "Role C body."])
    assert boilerplate.derive(samples, [], ("Acme Corp",)) == ("", "")


def test_two_postings_are_not_enough_for_a_shared_prefix() -> None:
    # Two openings in the same department share an intro that is not
    # boilerplate, so the minimum sample size is three.
    assert boilerplate.derive([_SPACEX, _SPACEX], [], ("SpaceX",)) == ("", "")


def test_a_blurb_quoting_a_sampled_job_title_is_rejected() -> None:
    shared = (
        "Senior Propulsion Engineer openings at Acme are described at length "
        "in this paragraph which repeats across every posting we publish. "
    )
    samples = _postings(shared, ["A.", "B.", "C."])
    assert boilerplate.derive(samples, ["Senior Propulsion Engineer"], ("Acme",)) == (
        "",
        "",
    )


# --- boilerplate: the guarded heading ----------------------------------

_VULCAN = (
    "About Us\nFounded in 2015 to develop the world's first industrially "
    "scalable laser metal additive manufacturing solution, VulcanForms is "
    "reshaping how the world manufactures critical products.\n\n"
)


def test_about_us_heading_is_used_when_it_repeats() -> None:
    samples = [f"Unique opening {i} for this posting.\n\n{_VULCAN}" for i in range(3)]
    text, method = boilerplate.derive(samples, [], ("VulcanForms",))
    assert method == "about_heading"
    assert text.startswith("Founded in 2015")


def test_about_us_heading_appearing_once_is_not_boilerplate() -> None:
    samples = [
        f"Unique opening {i} for this posting.\n\n"
        f"About Us\nA one-off paragraph number {i} that differs every time and "
        f"is therefore this posting's own copy rather than the company's.\n\n"
        for i in range(3)
    ]
    assert boilerplate.derive(samples, [], ("VulcanForms",)) == ("", "")


def test_about_the_role_heading_is_not_a_company_blurb() -> None:
    samples = [
        f"Unique opening {i} varying enough to leave no shared prefix here.\n\n"
        "**About the Role**\n\nWe're hiring an Android Engineer to join our "
        "mobile team and enhance the rewards experience for our users.\n\n"
        for i in range(3)
    ]
    assert boilerplate.derive(samples, [], ("Fetch",)) == ("", "")


def test_about_the_company_on_a_staffing_board_is_not_the_tenant() -> None:
    samples = [
        f"Unique opening {i} varying enough to leave no shared prefix here.\n\n"
        "About the Company\nOur client is a leading construction organization "
        "known for delivering complex projects with operational excellence.\n\n"
        for i in range(3)
    ]
    assert boilerplate.derive(samples, [], ("Kimmel & Associates",)) == ("", "")


def test_about_the_tenant_by_name_is_accepted() -> None:
    samples = [
        f"Unique opening {i} varying enough to leave no shared prefix here.\n\n"
        "**About Fetch**\n\nAt Fetch, we're dedicated to helping pets live "
        "their healthiest and happiest lives. Our comprehensive insurance "
        "coverage is designed with modern pet parents in mind, and we're "
        "proud to support the animal shelter community.\n\n"
        for i in range(3)
    ]
    text, method = boilerplate.derive(samples, [], ("Fetch",))
    assert method == "about_heading"
    assert text.startswith("At Fetch")


def test_a_shared_display_name_keeps_the_blurb_only_for_who_it_names() -> None:
    # Five distinct FOX entities post under the display name "FOX", so
    # their postings pool together and the shared prefix is whichever
    # one the sample happened to draw.
    import polars as pl

    prefix = (
        "What We Do FOX Factory designs, engineers, manufactures and markets "
        "performance-defining products and systems for customers worldwide. "
    )
    samples = pl.DataFrame(
        {
            "ats_type": ["workday"] * 3,
            "company": ["FOX"] * 3,
            "title": ["Engineer", "Analyst", "Technician"],
            "head": [f"{prefix}Role body {i}." for i in range(3)],
        }
    )
    cohort = pl.DataFrame(
        {
            "ats": ["workday", "workday"],
            "slug": ["foxfactory/fox", "foxrehab/external"],
            "jobs_company": ["FOX", "FOX"],
            "display_name": ["Fox Factory", "FOX Rehabilitation"],
            "name": ["Fox Factory", "FOX Rehabilitation"],
        }
    )
    out = boilerplate.build(cohort, samples)
    assert out["slug"].to_list() == ["foxfactory/fox"]


def test_an_unshared_display_name_needs_no_naming() -> None:
    import polars as pl

    prefix = (
        "We're building a world of health around every individual, shaping a "
        "more connected and convenient experience for the people we serve. "
    )
    samples = pl.DataFrame(
        {
            "ats_type": ["workday"] * 3,
            "company": ["CVS Health"] * 3,
            "title": ["Pharmacist", "Supervisor", "Technician"],
            "head": [f"{prefix}Role body {i}." for i in range(3)],
        }
    )
    cohort = pl.DataFrame(
        {
            "ats": ["workday"],
            "slug": ["cvshealth/cvs_health_careers"],
            "jobs_company": ["CVS Health"],
            "display_name": ["CVS Health"],
            "name": ["CVS Health"],
        }
    )
    out = boilerplate.build(cohort, samples)
    assert out.height == 1
    assert out["boilerplate_description"][0].startswith("We're building a world")


def test_common_prefix_stops_at_the_first_difference() -> None:
    assert boilerplate.common_prefix(["abcdef", "abcxyz", "abc"]) == "abc"
    assert boilerplate.common_prefix(["abc", "xyz"]) == ""
    assert boilerplate.common_prefix([]) == ""


# --- companysite: metadata extraction ----------------------------------


def _page(head: str, body: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def test_og_description_is_preferred_over_the_meta_tag() -> None:
    html = _page(
        '<meta name="description" content="Keyword stuffed search copy for a '
        'company that sells many different things to many people.">'
        '<meta property="og:description" content="Septerna is a biotechnology '
        'company discovering novel small molecule medicines.">'
    )
    assert companysite.extract_description(html).startswith("Septerna is a")


def test_description_attributes_may_appear_in_any_order() -> None:
    html = _page(
        "<meta content='Meter is building the vertically integrated network "
        "company for enterprises.' property='og:description'>"
    )
    assert companysite.extract_description(html).startswith("Meter is building")


def test_description_is_unescaped() -> None:
    html = _page(
        '<meta property="og:description" content="America&#39;s leading health '
        'solutions company, CVS Health provides advanced care.">'
    )
    assert companysite.extract_description(html).startswith("America's leading")


def test_short_meta_description_is_ignored() -> None:
    assert companysite.extract_description(_page('<meta name="description" content="Home">')) == ""


def test_jsonld_organization_description_is_the_last_resort() -> None:
    html = _page(
        '<script type="application/ld+json">'
        '{"@graph":[{"@type":"WebSite","description":"ignored, wrong type"},'
        '{"@type":"Organization","description":"Mozilla is the not-for-profit '
        'behind the Firefox browser."}]}</script>'
    )
    assert companysite.extract_description(html).startswith("Mozilla is the")


def test_malformed_jsonld_does_not_raise() -> None:
    html = _page('<script type="application/ld+json">{not json,,,</script>')
    assert companysite.extract_description(html) == ""


def test_extract_title() -> None:
    assert companysite.extract_title(_page("<title>Home | Northrop Grumman</title>")) == (
        "Home | Northrop Grumman"
    )


# --- companysite: careers links ----------------------------------------


@pytest.mark.parametrize(
    ("html", "domain", "expected"),
    [
        pytest.param(
            '<a href="/careers">Careers</a>',
            "northropgrumman.com",
            "https://www.northropgrumman.com/careers",
            id="relative-path",
        ),
        pytest.param(
            '<a href="https://careers.leidos.com/">Careers</a>',
            "leidos.com",
            "https://careers.leidos.com/",
            id="careers-subdomain-no-path",
        ),
        pytest.param(
            '<a href="https://jobs.cvshealth.com?cid=nav">Careers</a>',
            "cvshealth.com",
            "https://jobs.cvshealth.com?cid=nav",
            id="jobs-subdomain-with-query",
        ),
        pytest.param(
            '<a href="/company/careers.html">Careers</a>',
            "blackducksoftware.com",
            "https://www.blackducksoftware.com/company/careers.html",
            id="careers-with-file-extension",
        ),
        pytest.param(
            '<a href="/en-US/careers/">Careers</a>',
            "mozilla.org",
            "https://www.mozilla.org/en-US/careers/",
            id="locale-prefixed-path",
        ),
    ],
)
def test_extract_careers_url_accepts(html: str, domain: str, expected: str) -> None:
    base = f"https://www.{domain}"
    assert companysite.extract_careers_url(html, base, domain) == expected


def test_link_text_alone_does_not_qualify_a_product_page() -> None:
    # The observed false positive: an "Opportunities" anchor pointing at
    # a product page. The real careers link must win instead.
    html = (
        '<a href="/digital-analytics/experience-analytics">Opportunities</a>'
        '<a href="/careers">Careers</a>'
    )
    assert companysite.extract_careers_url(
        html, "https://www.quantummetric.com", "quantummetric.com"
    ) == "https://www.quantummetric.com/careers"


def test_a_product_page_alone_yields_nothing() -> None:
    html = '<a href="/digital-analytics/experience-analytics">Opportunities</a>'
    assert (
        companysite.extract_careers_url(
            html, "https://www.quantummetric.com", "quantummetric.com"
        )
        == ""
    )


def test_a_careers_link_on_another_companys_domain_is_rejected() -> None:
    # tangeroutlet.com links to tanger.inc; a different registrable
    # domain cannot be confirmed to be the same company.
    html = '<a href="https://www.tanger.inc/join-us/working-at-tanger">Careers</a>'
    assert (
        companysite.extract_careers_url(
            html, "https://www.tangeroutlet.com", "tangeroutlet.com"
        )
        == ""
    )


def test_an_ats_board_link_is_not_the_companys_own_careers_page() -> None:
    # The board URL is already known from the directory; this stage is
    # looking for the company's own page.
    html = '<a href="https://job-boards.greenhouse.io/anduril">Careers</a>'
    assert companysite.extract_careers_url(html, "https://www.anduril.com", "anduril.com") == ""


def test_the_shallowest_careers_path_wins() -> None:
    html = (
        '<a href="/careers/engineering/openings/12345">Software Engineer</a>'
        '<a href="/careers">Careers</a>'
    )
    assert (
        companysite.extract_careers_url(html, "https://www.acme.com", "acme.com")
        == "https://www.acme.com/careers"
    )


@pytest.mark.parametrize("href", ["#main", "mailto:jobs@acme.com", "javascript:void(0)"])
def test_non_navigational_hrefs_are_skipped(href: str) -> None:
    html = f'<a href="{href}">Careers</a>'
    assert companysite.extract_careers_url(html, "https://www.acme.com", "acme.com") == ""


# --- profile: careers URL from a posting URL ---------------------------


@pytest.mark.parametrize(
    ("posting", "ats", "slug", "expected"),
    [
        pytest.param(
            "https://boards.greenhouse.io/andurilindustries/jobs/4802172007?gh_jid=1",
            "greenhouse",
            "andurilindustries",
            "https://boards.greenhouse.io/andurilindustries",
            id="greenhouse-path",
        ),
        pytest.param(
            "https://jobs.lever.co/cgsfederal/0130d854-9bae-48d4-8ef0-e9da1ee778a5",
            "lever",
            "cgsfederal",
            "https://jobs.lever.co/cgsfederal",
            id="lever-path",
        ),
        pytest.param(
            "https://jobs.ashbyhq.com/crusoe/9f1c-abc",
            "ashby",
            "crusoe",
            "https://jobs.ashbyhq.com/crusoe",
            id="ashby-path",
        ),
        pytest.param(
            "https://american-logistics-authority.breezy.hr/p/f924d9d7cb61-cdl-driver",
            "breezy",
            "american-logistics-authority",
            "https://american-logistics-authority.breezy.hr",
            id="breezy-subdomain",
        ),
        pytest.param(
            "https://ngc.wd1.myworkdayjobs.com/northrop_grumman_external_site/job/"
            "United-States-Alaska/Operator_R10242598",
            "workday",
            "ngc/northrop_grumman_external_site",
            "https://ngc.wd1.myworkdayjobs.com/northrop_grumman_external_site",
            id="workday-host-plus-site",
        ),
        pytest.param(
            "https://jobs.smartrecruiters.com/alphabeinsightinc/744000140323669",
            "smartrecruiters",
            "alphabeinsightinc",
            "https://jobs.smartrecruiters.com/alphabeinsightinc",
            id="smartrecruiters-path",
        ),
        pytest.param(
            "https://trilongroup.pinpointhq.com/en/postings/09de5bee-feed",
            "pinpoint",
            "trilongroup",
            "https://trilongroup.pinpointhq.com",
            id="pinpoint-subdomain",
        ),
    ],
)
def test_board_url_from_posting(
    posting: str, ats: str, slug: str, expected: str
) -> None:
    assert profile.board_url_from_posting(posting, ats, slug) == expected


def test_a_short_link_posting_url_does_not_yield_a_board() -> None:
    # `resolve_careers_url` reads Workable's "/j/<id>" short link as a
    # tenant called "j", so the tenant's own slug has to be checked too.
    assert (
        profile.board_url_from_posting(
            "https://apply.workable.com/j/2B084BEA7B", "workable", "gotham-enterprises"
        )
        == ""
    )


def test_a_posting_url_from_another_tenant_is_rejected() -> None:
    assert (
        profile.board_url_from_posting(
            "https://boards.greenhouse.io/spacex/jobs/1", "greenhouse", "andurilindustries"
        )
        == ""
    )


def test_an_unrecognised_posting_url_yields_nothing() -> None:
    assert profile.board_url_from_posting("https://careers.adobe.com/us/en/job/1", "phenom", "adobe") == ""
    assert profile.board_url_from_posting("", "greenhouse", "acme") == ""


# --- profile: careers URL selection ------------------------------------


def test_directory_url_is_used_and_confirmed() -> None:
    url, source, verified = profile._careers_columns(
        {
            "ats": "greenhouse",
            "slug": "andurilindustries",
            "url": "https://job-boards.greenhouse.io/andurilindustries",
            "sample_posting_url": "",
        }
    )
    assert (url, source, verified) == (
        "https://job-boards.greenhouse.io/andurilindustries",
        "directory",
        True,
    )


def test_a_missing_directory_url_falls_back_to_the_posting_url() -> None:
    url, source, verified = profile._careers_columns(
        {
            "ats": "lever",
            "slug": "cgsfederal",
            "url": "",
            "sample_posting_url": "https://jobs.lever.co/cgsfederal/0130d854-9bae",
        }
    )
    assert (url, source, verified) == ("https://jobs.lever.co/cgsfederal", "posting_url", True)


def test_a_custom_domain_careers_site_is_kept_but_unconfirmed() -> None:
    url, source, verified = profile._careers_columns(
        {
            "ats": "phenom",
            "slug": "adobe",
            "url": "https://careers.adobe.com",
            "sample_posting_url": "https://careers.adobe.com/us/en/job/1",
        }
    )
    assert (url, source, verified) == ("https://careers.adobe.com", "directory", False)


def test_no_careers_url_at_all() -> None:
    assert profile._careers_columns(
        {"ats": "greenhouse", "slug": "acme", "url": "", "sample_posting_url": ""}
    ) == (None, None, False)


# --- profile: headquarters ---------------------------------------------


def test_headquarters_prefers_pdl_and_title_cases_it() -> None:
    assert profile._headquarters(
        {"locality": "falls church", "region": "virginia"}
    ) == ("Falls Church, Virginia", "pdl")


def test_headquarters_falls_back_to_the_filed_address() -> None:
    assert profile._headquarters(
        {"locality": None, "region": None, "registrant_city": "WOONSOCKET", "registrant_state": "RI"}
    ) == ("WOONSOCKET, Rhode Island", "sec_edgar")


def test_headquarters_keeps_a_non_us_edgar_code_as_filed() -> None:
    place, source = profile._headquarters(
        {"registrant_city": "dublin", "registrant_state": "L2"}
    )
    assert (place, source) == ("Dublin, L2", "sec_edgar")


def test_headquarters_is_absent_when_nothing_is_known() -> None:
    assert profile._headquarters({}) == (None, None)


# --- profile: Item 1, Business -----------------------------------------


def _filing(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def test_10k_description_skips_the_table_of_contents() -> None:
    html = _filing(
        "<p>Item 1. Business ..... 3</p><p>Item 1A. Risk Factors ..... 14</p>"
        "<p>Item 1. Business</p>"
        "<p>We are a leading provider of laser metal additive manufacturing "
        "solutions to the aerospace and medical industries. We operate two "
        "production facilities in the United States.</p>"
    )
    text = profile.description_from_10k(html, ("VulcanForms",))
    assert text.startswith("We are a leading provider")


def test_10k_description_drops_a_leading_section_heading() -> None:
    # These headings carry no punctuation, so leaving them in glues them
    # onto the sentence: "General Headquartered in Louisville, ...".
    html = _filing(
        "<p>Item 1. Business</p><p>General</p>"
        "<p>Headquartered in Louisville, Kentucky, Humana Inc. and its "
        "subsidiaries offer health and well-being services nationwide.</p>"
    )
    assert profile.description_from_10k(html, ("Humana Inc",)).startswith(
        "Headquartered in Louisville"
    )


def test_10k_description_ignores_boilerplate_openings() -> None:
    html = _filing(
        "<p>Item 1. Business</p>"
        "<p>All qualified applicants will receive consideration for employment "
        "without regard to race, color, religion, sex, or national origin.</p>"
    )
    assert profile.description_from_10k(html, ("Acme",)) == ""


def test_10k_description_is_empty_without_the_section() -> None:
    assert profile.description_from_10k(_filing("<p>Item 7. MD&A</p>"), ()) == ""


# --- profile: description selection ------------------------------------


def test_source_rank_orders_the_waterfall() -> None:
    ranks = profile.SOURCE_RANK
    assert (
        ranks["company_site"]
        > ranks["posting_boilerplate"]
        > ranks["sec_10k"]
        > ranks["wikidata"]
    )

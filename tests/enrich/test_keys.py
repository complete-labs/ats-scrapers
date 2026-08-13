"""Key stability tests.

These matter more than they look. ``job_key`` is the join key for the whole
sidecar, so a change in its behaviour silently orphans every enrichment row
and re-bills the corpus. ``content_hash`` is the cache key, so a change in
*its* behaviour re-buys every answer.
"""

from __future__ import annotations

from enrich.keys import content_hash, fallback_key, job_key, normalize_url


class TestNormalizeUrl:
    def test_lowercases_scheme_and_host_only(self) -> None:
        # Path case must survive: Lever tenant slugs and Workday requisition
        # ids are case-sensitive, and upstream preserves them deliberately.
        assert normalize_url("HTTPS://Jobs.Example.COM/MyTenant/Job-123") == (
            "https://jobs.example.com/MyTenant/Job-123"
        )

    def test_strips_ui_anchors_and_trailing_slash(self) -> None:
        assert normalize_url("https://x.test/jobs/1/#apply") == "https://x.test/jobs/1"
        assert normalize_url("https://x.test/jobs/1#top") == "https://x.test/jobs/1"

    def test_preserves_an_identifying_fragment(self) -> None:
        # Oracle Cloud Recruiting puts the requisition id in the fragment.
        # Dropping it collapses a whole tenant onto one key.
        base = "https://t.fa.us2.oraclecloud.com/?mode=jobs&site_number=CX_1"
        assert job_key(f"{base}#217175") != job_key(f"{base}#216716")
        assert normalize_url(f"{base}#217175").endswith("#217175")

    def test_drops_tracking_parameters(self) -> None:
        assert (
            normalize_url("https://x.test/j/1?utm_source=li&gh_src=abc&id=7&fbclid=z")
            == "https://x.test/j/1?id=7"
        )

    def test_sorts_remaining_query_parameters(self) -> None:
        # Oracle and Taleo emit the same posting with parameters in either
        # order depending on the referring page.
        assert normalize_url("https://x.test/j?b=2&a=1") == normalize_url(
            "https://x.test/j?a=1&b=2"
        )

    def test_drops_default_port_but_keeps_others(self) -> None:
        assert normalize_url("https://x.test:443/j") == "https://x.test/j"
        assert normalize_url("https://x.test:8443/j") == "https://x.test:8443/j"

    def test_unparseable_input_still_yields_something_stable(self) -> None:
        assert normalize_url("not a url") == "notaurl"
        assert normalize_url("") == ""


class TestJobKey:
    def test_is_stable_across_cosmetic_variation(self) -> None:
        variants = [
            "https://jobs.example.com/x/1",
            "https://jobs.example.com/x/1/",
            "HTTPS://JOBS.EXAMPLE.COM/x/1#top",
            "https://jobs.example.com/x/1?utm_campaign=spring",
        ]
        assert len({job_key(url) for url in variants}) == 1

    def test_distinguishes_different_postings(self) -> None:
        assert job_key("https://x.test/1") != job_key("https://x.test/2")

    def test_is_128_bits_of_hex(self) -> None:
        key = job_key("https://x.test/1")
        assert len(key) == 32
        assert int(key, 16) >= 0


class TestContentHash:
    def test_ignores_whitespace_reflow(self) -> None:
        # Providers re-emit identical postings with different wrapping; those
        # must share one paid call.
        a = content_hash(title="Engineer", description="Line one.\nLine two.")
        b = content_hash(title="Engineer", description="Line one.    Line two.")
        assert a == b

    def test_changes_when_body_changes(self) -> None:
        a = content_hash(title="Engineer", description="We need Python.")
        b = content_hash(title="Engineer", description="We need Rust.")
        assert a != b

    def test_ignores_changes_past_the_truncation_limit(self) -> None:
        # This is the property that keeps the cache from missing on rows whose
        # prompts are byte-identical after truncation.
        head = "x" * 100
        a = content_hash(title="T", description=head + "A" * 50, truncate=100)
        b = content_hash(title="T", description=head + "B" * 50, truncate=100)
        assert a == b

    def test_respects_changes_inside_the_truncation_limit(self) -> None:
        a = content_hash(title="T", description="A" + "x" * 200, truncate=100)
        b = content_hash(title="T", description="B" + "x" * 200, truncate=100)
        assert a != b

    def test_nul_joining_prevents_field_boundary_collisions(self) -> None:
        assert content_hash(title="ab", description="c") != content_hash(
            title="a", description="bc"
        )

    def test_salary_and_commitment_participate(self) -> None:
        base = {"title": "T", "description": "body text here"}
        assert content_hash(**base) != content_hash(**base, salary_summary="$100k")
        assert content_hash(**base) != content_hash(**base, commitment="Vollzeit")


class TestFallbackKey:
    def test_survives_url_change(self) -> None:
        a = fallback_key("Acme Corp", "Senior Engineer", "Berlin, DE")
        b = fallback_key("acme corp", "senior  engineer", "berlin, de")
        assert a == b and a is not None

    def test_requires_company_and_title(self) -> None:
        # A key built from location alone would merge unrelated postings.
        assert fallback_key(None, "Engineer", "Berlin") is None
        assert fallback_key("Acme", None, "Berlin") is None
        assert fallback_key("Acme", "Engineer", None) is not None

    def test_accent_insensitive(self) -> None:
        assert fallback_key("Zürich AG", "Entwickler", "Zürich") == fallback_key(
            "Zurich AG", "Entwickler", "Zurich"
        )

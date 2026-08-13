"""Tests for the eval harness, Tier 2 bounds, and the delta classifier.

The eval harness needs its own tests because a scoring bug is worse than no
scoring: it produces a number that looks like evidence. In particular, the
over-claim path (label says "must be null") and the leakage masking are the
two things that make these scores mean anything.
"""

from __future__ import annotations

from typing import Any

import pytest

from enrich._version import ENRICHMENT_VERSION
from enrich.agent import (
    MIN_RECOVERED_CHARS,
    AgentRun,
    RecoveryTools,
    ToolResult,
    recover_row,
    summarize_runs,
)
from enrich.delta import DeltaCounts, classify
from enrich.eval import EvalReport, FieldScore, evaluate
from enrich.gold import load_edge_cases
from enrich.gold.loader import GoldExample, build_holdout
from enrich.keys import content_hash, job_key
from enrich.llm import StubClient
from enrich.prompt import prompt_hash, render_user_prompt
from enrich.schema import JobEnrichment


class TestFieldScore:
    def test_recall_counts_only_positives(self) -> None:
        score = FieldScore(hit=3, wrong=1, missed=1, over_claimed=2, correct_abstain=4)
        assert score.recall == pytest.approx(3 / 5)

    def test_precision_includes_over_claims(self) -> None:
        # An over-claim is a false positive: we asserted a value where the
        # right answer was to abstain.
        score = FieldScore(hit=3, wrong=1, over_claimed=2)
        assert score.precision == pytest.approx(3 / 6)

    def test_over_claim_rate_uses_negatives(self) -> None:
        score = FieldScore(over_claimed=1, correct_abstain=3)
        assert score.over_claim_rate == pytest.approx(0.25)

    def test_undefined_metrics_are_none_not_zero(self) -> None:
        # Reporting 0% for "no labels" would look like a failure.
        blank = FieldScore()
        assert blank.recall is None
        assert blank.precision is None
        assert blank.over_claim_rate is None


class TestEvalScoring:
    def _example(self, labels: dict[str, Any], **row: Any) -> GoldExample:
        payload: dict[str, Any] = {
            "url": "https://x.test/1",
            "title": "Engineer",
            "location": "Berlin, Germany",
            "description": "body " * 60,
            "language": None,
            "country_iso": None,
        }
        payload.update(row)
        payload["job_key"] = job_key(str(payload["url"]))
        payload["content_hash"] = content_hash(
            title=payload.get("title"), description=payload.get("description"), truncate=2000
        )
        return GoldExample(id="ex", row=payload, labels=labels)

    def _report(self) -> EvalReport:
        return EvalReport(model_id="test", truncate=2000, prompt_hash="p", generated_at="now")

    def _row(self, **values: Any) -> JobEnrichment:
        return JobEnrichment(
            job_key="k",
            content_hash="c",
            url="https://x.test/1",
            enrichment_version=ENRICHMENT_VERSION,
            **values,
        )

    def test_hit_when_value_matches(self) -> None:
        report = self._report()
        report.observe(self._example({"country_iso": "DE"}), self._row(country_iso="DE"), None)
        assert report.fields["country_iso"].hit == 1

    def test_missed_when_we_abstain_on_a_real_value(self) -> None:
        report = self._report()
        report.observe(self._example({"country_iso": "DE"}), self._row(), None)
        assert report.fields["country_iso"].missed == 1

    def test_over_claim_when_label_demands_null(self) -> None:
        report = self._report()
        report.observe(self._example({"salary_min": None}), self._row(salary_min=100000.0), None)
        assert report.fields["salary_min"].over_claimed == 1
        assert report.failures[0]["field"] == "salary_min"

    def test_correct_abstain_is_credited(self) -> None:
        report = self._report()
        report.observe(self._example({"salary_min": None}), self._row(), None)
        assert report.fields["salary_min"].correct_abstain == 1
        assert report.failures == []

    def test_money_tolerance_absorbs_rounding_only(self) -> None:
        report = self._report()
        report.observe(self._example({"salary_min": 100000}), self._row(salary_min=100500.0), None)
        assert report.fields["salary_min"].hit == 1
        report.observe(self._example({"salary_min": 100000}), self._row(salary_min=140000.0), None)
        assert report.fields["salary_min"].wrong == 1

    def test_years_near_miss_is_tracked_separately(self) -> None:
        report = self._report()
        report.observe(
            self._example({"experience_min_years": 3}),
            self._row(experience_min_years=4),
            None,
        )
        score = report.fields["experience_min_years"]
        assert score.wrong == 1
        assert score.near == 1

    def test_placement_is_split_by_difficulty(self) -> None:
        report = self._report()
        # "Remote" in the location makes this answerable by keyword match.
        report.observe(
            self._example({"placement": "remote"}, location="Remote"),
            self._row(placement="remote"),
            None,
        )
        report.observe(
            self._example({"placement": "hybrid"}, location="Berlin"),
            self._row(placement="hybrid"),
            None,
        )
        assert report.slices["placement_trivial"]["placement"].labelled == 1
        assert report.slices["placement_hard"]["placement"].labelled == 1

    def test_skills_include_and_exclude(self) -> None:
        report = self._report()
        report.observe(
            self._example({"skills_include": ["python"], "skills_exclude": ["team player"]}),
            self._row(skills=["python", "sql"]),
            None,
        )
        assert report.skills_pass == 1
        report.observe(
            self._example({"skills_exclude": ["team player"]}),
            self._row(skills=["team player"]),
            None,
        )
        assert report.skills_pass == 1
        assert report.skills_total == 2


class TestGoldSet:
    def test_edge_cases_load_and_carry_negative_labels(self) -> None:
        examples = load_edge_cases()
        assert len(examples) >= 40
        negatives = sum(1 for e in examples for value in e.labels.values() if value is None)
        # Negative labels are the only way to measure over-claiming, and a
        # provider-labelled set cannot supply them.
        assert negatives >= 15

    def test_masking_removes_the_leaking_field(self) -> None:
        example = GoldExample(
            id="x",
            row={"title": "T", "commitment": "Vollzeit", "description": "d"},
            labels={"employment_type": "FULL_TIME"},
            mask=("commitment",),
        )
        assert example.row["commitment"] == "Vollzeit"
        assert example.masked_row["commitment"] is None

    def test_masked_field_is_absent_from_the_prompt(self) -> None:
        example = GoldExample(
            id="x",
            row={
                "title": "T",
                "commitment": "Vollzeit",
                "salary_summary": "90000 EUR",
                "description": "body",
            },
            labels={"salary_min": 90000},
            mask=("salary_summary",),
        )
        rendered = render_user_prompt(example.masked_row, truncate=2000)
        assert "90000 EUR" not in rendered

    def test_holdout_builder_skips_thin_bodies(self) -> None:
        rows = [
            {"job_key": "a" * 32, "description": "short", "is_remote": "true", "url": "u"},
        ]
        assert build_holdout(rows, per_target=10) == []

    def test_holdout_builder_assigns_one_target_per_row(self) -> None:
        rows = [
            {
                "job_key": f"{index:032d}",
                "url": f"https://x.test/{index}",
                "description": "x" * 500,
                "is_remote": "true",
                "employment_type": "FULL_TIME",
                "language": "en",
                "country_iso": "US",
                "salary_min": "100000",
            }
            for index in range(5)
        ]
        records = build_holdout(rows, per_target=10)
        # Each row serves exactly one target, so its mask cannot destroy the
        # evidence for another label.
        assert len(records) == 5
        for record in records:
            assert len(record["labels"]) <= 2

    def test_provider_false_is_not_a_three_way_placement_label(self) -> None:
        rows = [
            {
                "job_key": "b" * 32,
                "url": "https://x.test/1",
                "description": "x" * 500,
                "is_remote": "false",
            }
        ]
        labels = build_holdout(rows, per_target=10)[0]["labels"]
        assert labels["is_remote"] is False
        assert "placement" not in labels


class TestEvaluateEndToEnd:
    def test_runs_the_real_pipeline(self) -> None:
        examples = load_edge_cases()[:10]
        report = evaluate(examples, StubClient(), truncate=2000)
        assert report.examples == 10
        assert report.model_id == "stub-deterministic-v1"
        assert report.prompt_hash == prompt_hash(truncate=2000)
        assert report.fields

    def test_report_serializes(self) -> None:
        report = evaluate(load_edge_cases()[:5], StubClient(), truncate=2000)
        data = report.as_dict()
        assert "fields" in data and "slices" in data


class TestAgentBounds:
    class _Tools(RecoveryTools):
        def __init__(self, results: dict[str, ToolResult]) -> None:
            super().__init__()
            self.results = results
            self.calls: list[str] = []

        def scraper_description(self, url: str) -> ToolResult:
            self.calls.append("scraper_description")
            return self.results.get(
                "scraper_description", ToolResult("scraper_description", "", error="none")
            )

        def fetch_page(self, url: str) -> ToolResult:
            self.calls.append("fetch_page")
            return self.results.get("fetch_page", ToolResult("fetch_page", "", error="none"))

        def company_careers(self, company: str) -> ToolResult:
            self.calls.append("company_careers")
            return self.results.get(
                "company_careers", ToolResult("company_careers", "", error="none")
            )

    def _row(self) -> dict[str, Any]:
        return {
            "job_key": "k",
            "url": "https://x.test/1",
            "company": "Acme",
            "description": "",
        }

    def test_stops_at_the_first_usable_result(self) -> None:
        tools = self._Tools({"scraper_description": ToolResult("scraper_description", "x" * 500)})
        text, run = recover_row(self._row(), tools, max_steps=3)
        assert text is not None
        # One fetch, not three: the ordering is what keeps the tier affordable.
        assert tools.calls == ["scraper_description"]
        assert run.steps == 1
        assert run.resolved is False  # set later, by the model stage

    def test_falls_through_to_later_tools(self) -> None:
        tools = self._Tools({"fetch_page": ToolResult("fetch_page", "y" * 500)})
        text, _run = recover_row(self._row(), tools, max_steps=3)
        assert text is not None
        assert tools.calls == ["scraper_description", "fetch_page"]

    def test_respects_the_step_cap(self) -> None:
        tools = self._Tools({})
        text, run = recover_row(self._row(), tools, max_steps=1)
        assert text is None
        assert len(tools.calls) == 1
        assert run.steps == 1

    def test_rejects_text_below_the_useful_threshold(self) -> None:
        tools = self._Tools({"fetch_page": ToolResult("fetch_page", "too short")})
        text, _ = recover_row(self._row(), tools, max_steps=3)
        assert text is None

    def test_records_the_error_from_a_failed_tool(self) -> None:
        tools = self._Tools(
            {"scraper_description": ToolResult("scraper_description", "", error="HTTP 404")}
        )
        _, run = recover_row(self._row(), tools, max_steps=1)
        assert run.error == "HTTP 404"

    def test_summarize_runs(self) -> None:
        runs = [
            AgentRun(job_key="a", steps=1, recovered_chars=MIN_RECOVERED_CHARS + 1),
            AgentRun(job_key="b", steps=3, recovered_chars=0),
        ]
        summary = summarize_runs(runs)
        assert summary["runs"] == 2
        assert summary["recovered"] == 1
        assert summary["mean_steps"] == 2


class TestDeltaClassify:
    def _row(self, url: str, content: str) -> dict[str, Any]:
        return {"job_key": job_key(url), "content_hash": content, "url": url}

    def test_new_rows_are_selected(self) -> None:
        rows = [self._row("https://x.test/1", "c1")]
        todo, counts = classify(iter([rows]), {}, current_prompt_hash="p1")
        assert counts.new == 1
        assert len(todo) == 1

    def test_unchanged_rows_are_skipped(self) -> None:
        row = self._row("https://x.test/1", "c1")
        state = {row["job_key"]: ("c1", ENRICHMENT_VERSION, "p1")}
        todo, counts = classify(iter([[row]]), state, current_prompt_hash="p1")
        assert counts.unchanged == 1
        assert todo == []

    def test_changed_content_is_selected(self) -> None:
        row = self._row("https://x.test/1", "c2")
        state = {row["job_key"]: ("c1", ENRICHMENT_VERSION, "p1")}
        todo, counts = classify(iter([[row]]), state, current_prompt_hash="p1")
        assert counts.changed == 1
        assert len(todo) == 1

    def test_prompt_change_makes_rows_stale(self) -> None:
        # This is what makes prompt iteration affordable: only the answers are
        # invalidated, and shared bodies still hit the cache.
        row = self._row("https://x.test/1", "c1")
        state = {row["job_key"]: ("c1", ENRICHMENT_VERSION, "old-prompt")}
        todo, counts = classify(iter([[row]]), state, current_prompt_hash="new-prompt")
        assert counts.stale == 1
        assert len(todo) == 1

    def test_version_bump_makes_rows_stale(self) -> None:
        row = self._row("https://x.test/1", "c1")
        state = {row["job_key"]: ("c1", ENRICHMENT_VERSION - 1, None)}
        _todo, counts = classify(iter([[row]]), state, current_prompt_hash="p1")
        assert counts.stale == 1

    def test_counts_report_a_delta_percentage(self) -> None:
        counts = DeltaCounts(seen=100, new=2, changed=3, unchanged=95)
        assert counts.to_enrich == 5
        assert counts.as_dict()["delta_pct"] == 5.0

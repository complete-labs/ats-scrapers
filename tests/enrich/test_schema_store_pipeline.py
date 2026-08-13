"""Contract, store and merge-precedence tests.

Merge precedence is the highest-risk logic in the layer: it decides when a
model may overwrite something an employer stated. Getting it wrong is
invisible in aggregate coverage numbers and corrupts the output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from enrich._version import ENRICHMENT_VERSION
from enrich.deterministic import run_tier0
from enrich.keys import content_hash, job_key
from enrich.llm import Request, StubClient, parse_batch_output, write_batch_requests
from enrich.pipeline import merge
from enrich.schema import Evidence, JobEnrichment, LlmExtraction, llm_json_schema
from enrich.store import EnrichmentStore


def _extraction(**overrides: Any) -> LlmExtraction:
    base: dict[str, Any] = {
        "insufficient_text": False,
        "placement": None,
        "employment_type": None,
        "experience_min_years": None,
        "experience_max_years": None,
        "seniority": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
        "department": None,
        "skills": [],
        "education_level": None,
        "visa_sponsorship": None,
        "evidence": Evidence(salary_quote=None, placement_quote=None, experience_quote=None),
    }
    base.update(overrides)
    return LlmExtraction(**base)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "url": "https://x.test/1",
        "title": "Senior Backend Engineer",
        "location": "Berlin, Germany",
        "description": "We need 5+ years of experience with Python. " * 10,
        "commitment": None,
        "ats_type": "lever",
        "ats_id": "acme-123",
    }
    row.update(overrides)
    row["job_key"] = job_key(str(row["url"]))
    row["fallback_key"] = None
    row["content_hash"] = content_hash(
        title=row.get("title"), description=row.get("description"), truncate=2000
    )
    return row


class TestStrictSchema:
    def test_every_object_forbids_extra_keys_and_requires_all(self) -> None:
        schema = llm_json_schema()

        def check(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" or "properties" in node:
                    properties = node.get("properties") or {}
                    assert node.get("additionalProperties") is False
                    assert set(node.get("required") or []) == set(properties)
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(schema)

    def test_unsupported_validation_keywords_are_stripped(self) -> None:
        serialized = repr(llm_json_schema())
        for keyword in ("'default'", "'minimum'", "'maxLength'", "'pattern'"):
            assert keyword not in serialized

    def test_nullability_is_preserved(self) -> None:
        # Strict mode has no optional keys; "not stated" has to travel as a
        # nullable type or the contract cannot express abstention.
        placement = llm_json_schema()["properties"]["placement"]
        assert "anyOf" in placement or "null" in str(placement)


class TestIsRemoteProjection:
    @pytest.mark.parametrize(
        ("placement", "expected"),
        [("remote", True), ("hybrid", False), ("onsite", False), (None, None)],
    )
    def test_hybrid_is_not_remote(self, placement: str | None, expected: bool | None) -> None:
        row = JobEnrichment(
            job_key="k",
            content_hash="c",
            url="https://x.test/1",
            enrichment_version=ENRICHMENT_VERSION,
            placement=placement,  # type: ignore[arg-type]
        )
        assert row.as_is_remote() is expected


class TestMergePrecedence:
    def test_provider_value_is_never_overwritten(self) -> None:
        row = _row(country_iso="AT", employment_type="PART_TIME")
        tier0 = run_tier0(row)
        merged = merge(
            row, tier0, _extraction(employment_type="FULL_TIME"), tier="tier1", model_id="m"
        )
        assert merged.employment_type == "PART_TIME"
        assert merged.sources["employment_type"] == "provider"

    def test_model_may_override_a_tier0_placement_keyword(self) -> None:
        # Tier 0 only reads the title and location; a model that has read the
        # body is better evidence.
        row = _row(location="Remote")
        tier0 = run_tier0(row)
        assert tier0.values["placement"] == "remote"
        merged = merge(row, tier0, _extraction(placement="hybrid"), tier="tier1", model_id="m")
        assert merged.placement == "hybrid"
        assert merged.sources["placement"] == "tier1"

    def test_model_may_not_override_a_provider_remote_flag(self) -> None:
        row = _row(is_remote="false")
        tier0 = run_tier0(row)
        merged = merge(row, tier0, _extraction(placement="remote"), tier="tier1", model_id="m")
        assert merged.is_remote is False
        assert merged.placement == "hybrid"
        assert merged.needs_review is True

    def test_model_fills_fields_tier0_cannot_reach(self) -> None:
        row = _row()
        merged = merge(
            row,
            run_tier0(row),
            _extraction(seniority="staff", department="engineering", skills=["Python", " SQL "]),
            tier="tier1",
            model_id="m",
        )
        assert merged.seniority == "staff"
        assert merged.department == "engineering"
        assert merged.skills == ["python", "sql"]

    def test_tier0_experience_wins_over_the_model(self) -> None:
        # Tier 0's regex found it in the text; the model's number is not a
        # better source for the same evidence.
        row = _row()
        merged = merge(
            row, run_tier0(row), _extraction(experience_min_years=99), tier="tier1", model_id="m"
        )
        assert merged.experience_min_years == 5

    def test_unquoted_model_salary_is_flagged(self) -> None:
        row = _row()
        merged = merge(
            row,
            run_tier0(row),
            _extraction(salary_min=100000, salary_currency="USD"),
            tier="tier1",
            model_id="m",
        )
        assert merged.needs_review is True
        assert merged.review_reason is not None
        assert "quote" in merged.review_reason

    def test_quoted_model_salary_is_accepted(self) -> None:
        row = _row()
        merged = merge(
            row,
            run_tier0(row),
            _extraction(
                salary_min=100000,
                salary_currency="USD",
                evidence=Evidence(
                    salary_quote="$100,000 per year",
                    placement_quote=None,
                    experience_quote=None,
                ),
            ),
            tier="tier1",
            model_id="m",
        )
        assert merged.salary_min == 100000
        assert merged.evidence["salary"] == "$100,000 per year"
        assert merged.needs_review is False

    def test_insufficient_text_routes_to_review(self) -> None:
        row = _row(description="")
        merged = merge(row, run_tier0(row), _extraction(insufficient_text=True), tier="tier1")
        assert merged.needs_review is True

    def test_inverted_ranges_are_repaired(self) -> None:
        row = _row()
        merged = merge(
            row,
            run_tier0(row),
            _extraction(
                salary_min=200000,
                salary_max=100000,
                evidence=Evidence(salary_quote="q", placement_quote=None, experience_quote=None),
            ),
            tier="tier1",
        )
        assert merged.salary_min == 100000
        assert merged.salary_max == 200000

    def test_global_id_is_reconstructed(self) -> None:
        row = _row()
        merged = merge(row, run_tier0(row), None)
        assert merged.global_id == "lever:acme-123"


class TestStore:
    @pytest.fixture
    def store(self) -> Any:
        store = EnrichmentStore(":memory:")
        yield store
        store.close()

    def _enrichment(self, key: str = "k1", **overrides: Any) -> JobEnrichment:
        payload: dict[str, Any] = {
            "job_key": key,
            "content_hash": "c1",
            "url": f"https://x.test/{key}",
            "enrichment_version": ENRICHMENT_VERSION,
            "enriched_at": datetime.now(tz=UTC),
        }
        payload.update(overrides)
        return JobEnrichment(**payload)

    def test_upsert_is_idempotent(self, store: EnrichmentStore) -> None:
        row = self._enrichment()
        store.upsert([row])
        store.upsert([row])
        assert store.count() == 1

    def test_upsert_replaces_by_key(self, store: EnrichmentStore) -> None:
        store.upsert([self._enrichment(country_iso="FR")])
        store.upsert([self._enrichment(country_iso="DE")])
        assert store.count() == 1
        value = store.con.execute("SELECT country_iso FROM enrichment").fetchone()[0]
        assert value == "DE"

    def test_all_null_typed_columns_insert(self, store: EnrichmentStore) -> None:
        # A Tier 0-only batch has no salary at all; inferring Arrow types from
        # all-None columns yields a null type that will not insert.
        store.upsert([self._enrichment(key="k2")])
        assert store.count() == 1

    def test_lists_and_json_round_trip(self, store: EnrichmentStore) -> None:
        store.upsert(
            [
                self._enrichment(
                    skills=["python", "sql"],
                    sources={"country_iso": "tier0"},
                    evidence={"salary": "$1"},
                )
            ]
        )
        skills, sources = store.con.execute("SELECT skills, sources FROM enrichment").fetchone()
        assert list(skills) == ["python", "sql"]
        assert "tier0" in sources

    def test_cache_is_keyed_by_content_prompt_and_model(self, store: EnrichmentStore) -> None:
        extraction = _extraction(seniority="senior")
        store.cache_put_many(
            [("hash-a", extraction, 100, 20, 0.001)], prompt_hash="p1", model_id="m1"
        )
        assert store.cache_get_many(["hash-a"], prompt_hash="p1", model_id="m1")
        # A prompt change must miss, so only affected rows are re-bought.
        assert not store.cache_get_many(["hash-a"], prompt_hash="p2", model_id="m1")
        assert not store.cache_get_many(["hash-a"], prompt_hash="p1", model_id="m2")

    def test_cache_survives_an_undecodable_entry(self, store: EnrichmentStore) -> None:
        store.con.execute(
            """
            INSERT INTO llm_cache VALUES
            ('bad', 'p1', 'm1', '{"not":"valid"}', 0, 0, 0.0, now())
            """
        )
        assert store.cache_get_many(["bad"], prompt_hash="p1", model_id="m1") == {}

    def test_cost_ledger_accumulates(self, store: EnrichmentStore) -> None:
        store.record_cost(stage="a", calls=1, input_tokens=10, output_tokens=2, cost_usd=0.5)
        store.record_cost(stage="b", calls=1, input_tokens=10, output_tokens=2, cost_usd=0.25)
        assert store.total_cost() == pytest.approx(0.75)

    def test_batch_lifecycle(self, store: EnrichmentStore) -> None:
        store.record_batch(batch_id="b1", shard="s0", status="submitted", request_count=5)
        assert [b["batch_id"] for b in store.open_batches()] == ["b1"]
        store.record_batch(batch_id="b1", shard="s0", status="completed", request_count=5)
        assert store.open_batches() == []
        assert store.batch_shards_done() == {"s0"}


class TestBatchFormat:
    def test_written_shard_shape(self, tmp_path: Any) -> None:
        import json

        row = _row()
        path = tmp_path / "shard.jsonl"
        write_batch_requests(path, [Request.from_row(row, truncate=2000)], model_id="test-model")
        record = json.loads(path.read_text().strip())
        # custom_id must be the content hash so results reattach to the cache
        # without a side table.
        assert record["custom_id"] == row["content_hash"]
        assert record["url"] == "/v1/chat/completions"
        assert record["body"]["model"] == "test-model"
        assert record["body"]["response_format"]["json_schema"]["strict"] is True
        assert record["body"]["temperature"] == 0

    def test_request_line_round_trips(self) -> None:
        import json

        from enrich.llm import batch_request_line

        row = _row()
        record = json.loads(batch_request_line(Request.from_row(row, truncate=2000), model_id="m"))
        assert record["custom_id"] == row["content_hash"]
        assert record["body"]["model"] == "m"

    def test_shard_ceilings_respect_the_api_limits(self) -> None:
        from enrich.llm import MAX_SHARD_BYTES, MAX_SHARD_REQUESTS, batch_request_line

        # The provider caps input files at 200 MB and 50,000 requests.
        assert MAX_SHARD_BYTES < 200 * 1000 * 1000
        assert MAX_SHARD_REQUESTS <= 50_000

        # A single request costs roughly 10 kB because every line repeats the
        # system prompt and the full JSON schema. That means the byte ceiling
        # binds first, and sharding on request count alone would produce files
        # the API rejects.
        size = len(
            batch_request_line(
                Request.from_row(_row(), truncate=2000), model_id="gpt-4o-mini"
            ).encode()
        )
        assert size > 4000
        assert size * MAX_SHARD_REQUESTS > MAX_SHARD_BYTES

    def test_parse_output_handles_mixed_success_and_failure(self, tmp_path: Any) -> None:
        import json

        good = _extraction(seniority="senior").model_dump_json()
        path = tmp_path / "out.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "custom_id": "h1",
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "choices": [{"message": {"content": good}}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                },
                            },
                        }
                    ),
                    json.dumps({"custom_id": "h2", "error": {"message": "rate limited"}}),
                    "not json at all",
                    "",
                ]
            )
        )
        completions = parse_batch_output(path)
        by_hash = {c.content_hash: c for c in completions}
        # One malformed line must not discard the paid answers around it.
        assert by_hash["h1"].ok is True
        assert by_hash["h1"].input_tokens == 10
        assert by_hash["h2"].ok is False


class TestBatchRoundTrip:
    """The half of the backfill that runs after money is spent.

    ``batch-simulate`` writes a provider-shaped output file for a shard, so
    write -> collect -> apply is exercised offline. Without it, the first real
    test of the collect path would be a submitted batch, which is the worst
    possible time to discover a parsing bug.
    """

    def test_written_shard_survives_simulate_and_collect(self, tmp_path: Any) -> None:
        import json

        from enrich.llm import StubClient, batch_request_line, parse_batch_output

        rows = [_row(url=f"https://x.test/{i}", description="A" * 900) for i in range(3)]
        shard = tmp_path / "shard-00000.jsonl"
        shard.write_text(
            "\n".join(
                batch_request_line(Request.from_row(r, truncate=2000), model_id="m") for r in rows
            )
            + "\n"
        )

        client = StubClient()
        lines = []
        for line in shard.read_text().splitlines():
            record = json.loads(line)
            prompt = next(m["content"] for m in record["body"]["messages"] if m["role"] == "user")
            completion = client.complete(
                [Request(content_hash=record["custom_id"], user_prompt=prompt)]
            )[0]
            assert completion.extraction is not None
            lines.append(
                json.dumps(
                    {
                        "custom_id": record["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": completion.extraction.model_dump_json()
                                        }
                                    }
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                            },
                        },
                    }
                )
            )
        out = tmp_path / "out.jsonl"
        out.write_text("\n".join(lines) + "\n")

        completions = parse_batch_output(out)
        assert len(completions) == len(rows)
        assert all(c.ok for c in completions)
        # custom_id is the content hash, which is what lets results reattach to
        # the cache with no side table.
        assert {c.content_hash for c in completions} == {r["content_hash"] for r in rows}


class TestCostAttribution:
    """The ledger must only ever contain money that was actually spent."""

    def test_a_rehearsal_file_is_priced_at_zero(self, tmp_path: Any) -> None:
        import json

        from enrich.llm import STUB_MODEL_ID, ModelConfig, reported_model

        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "custom_id": "h1",
                    "response": {"status_code": 200, "body": {"model": STUB_MODEL_ID}},
                }
            )
            + "\n"
        )
        assert reported_model(path) == STUB_MODEL_ID
        # Collecting a batch-simulate file used to book it at the configured
        # model's rate: 18,669 stub responses showed up as $1.50 of spend that
        # never happened.
        assert ModelConfig.from_env(STUB_MODEL_ID).cost(10_000_000, 1_000_000) == 0.0
        assert ModelConfig.from_env("gpt-4o-mini").cost(10_000_000, 1_000_000) > 0

    def test_price_overrides_cannot_make_the_stub_cost_money(self, monkeypatch: Any) -> None:
        from enrich.llm import STUB_MODEL_ID, ModelConfig

        monkeypatch.setenv("ATS_ENRICH_PRICE_IN", "99")
        monkeypatch.setenv("ATS_ENRICH_PRICE_OUT", "99")
        assert ModelConfig.from_env(STUB_MODEL_ID).cost(1_000_000, 1_000_000) == 0.0

    def test_a_real_output_file_keeps_its_own_model(self, tmp_path: Any) -> None:
        import json

        from enrich.llm import reported_model

        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps(
                {
                    "custom_id": "h1",
                    "response": {"status_code": 200, "body": {"model": "gpt-4o-mini"}},
                }
            )
            + "\n"
        )
        assert reported_model(path) == "gpt-4o-mini"

    def test_missing_model_falls_back_to_the_requested_one(self, tmp_path: Any) -> None:
        import json

        from enrich.llm import reported_model

        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"custom_id": "h1", "response": {"body": {}}}) + "\n")
        assert reported_model(path) is None


class TestStubClient:
    def test_is_deterministic(self) -> None:
        row = _row()
        request = Request.from_row(row, truncate=2000)
        first = StubClient().complete([request])[0]
        second = StubClient().complete([request])[0]
        assert first.extraction == second.extraction

    def test_flags_short_bodies_as_insufficient(self) -> None:
        row = _row(description="too short")
        completion = StubClient().complete([Request.from_row(row, truncate=2000)])[0]
        assert completion.extraction is not None
        assert completion.extraction.insufficient_text is True

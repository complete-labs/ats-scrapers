"""Opt-in quality gates.

Deselected by default (marker ``enrich_eval``) because they need
``data/all.parquet`` and, for a meaningful score, a model key. Run with:

    pytest -m enrich_eval

The thresholds are the point of this file: they turn "the eval looks fine"
into a build failure. They are set against **Tier 0 only**, so they pass with
no API key and no spend — Tier 0's geography and language extraction is
deterministic, and if it regresses these fail immediately.

Model-dependent thresholds are deliberately *not* asserted here. A gate on a
stub's score would be meaningless, and a gate on a real model's score belongs
in a run that actually pays for one. Set ``ATS_ENRICH_EVAL_CLIENT=openai`` to
score the real pipeline; the assertions then cover it too.
"""

from __future__ import annotations

import os

import pytest

from enrich.eval import evaluate
from enrich.gold import load_edge_cases, load_gold
from enrich.llm import resolve_client
from enrich.paths import DEFAULT_SNAPSHOT

pytestmark = pytest.mark.enrich_eval


def _client_name() -> str:
    return os.environ.get("ATS_ENRICH_EVAL_CLIENT", "stub")


class TestTier0Gates:
    """Thresholds that hold with no model at all."""

    @pytest.fixture(scope="class")
    def report(self):
        examples = load_gold(truncate=2000)
        if len(examples) < 100:
            pytest.skip("gold holdout not built; run: python -m enrich.run gold-build")
        return evaluate(examples, resolve_client(_client_name()), truncate=2000)

    def test_country_recall_holds(self, report) -> None:
        score = report.fields.get("country_iso")
        assert score is not None and score.labelled >= 50
        # Measured at 95.5% against provider labels. A drop below 90 means the
        # gazetteer or the resolution ordering regressed.
        assert score.recall is not None and score.recall >= 0.90

    def test_language_recall_holds(self, report) -> None:
        score = report.fields.get("language")
        assert score is not None and score.labelled >= 50
        assert score.recall is not None and score.recall >= 0.85

    def test_no_over_claiming_on_experience(self, report) -> None:
        # Tier 0's experience regex must not invent requirements from company
        # history sentences. This is the check that would have caught the
        # "120 years" digit-boundary bug.
        score = report.fields.get("experience_min_years")
        assert score is not None
        if score.negatives:
            assert score.over_claim_rate == 0.0


class TestAbstentionGates:
    """The pipeline must not assert values the posting does not state."""

    @pytest.fixture(scope="class")
    def report(self):
        return evaluate(
            load_edge_cases(truncate=2000), resolve_client(_client_name()), truncate=2000
        )

    def test_salary_over_claim_stays_bounded(self, report) -> None:
        score = report.fields.get("salary_min")
        assert score is not None and score.negatives >= 5
        # Equity ranges, revenue figures and bonus percentages must not be
        # read as pay. A wrong salary is worse than a missing one because
        # nothing downstream can tell them apart.
        assert score.over_claim_rate is not None and score.over_claim_rate <= 0.25

    def test_placement_is_not_guessed_from_a_bare_city(self, report) -> None:
        score = report.fields.get("placement")
        assert score is not None
        if score.negatives:
            assert score.over_claim_rate is not None and score.over_claim_rate <= 0.25


@pytest.mark.skipif(not DEFAULT_SNAPSHOT.exists(), reason="snapshot not downloaded")
class TestSnapshotShape:
    """Assumptions about the published parquet that the layer depends on."""

    def test_join_key_assumptions_still_hold(self) -> None:
        from enrich.store import connect

        con = connect(database=":memory:")
        columns = {
            str(row[0]): str(row[1])
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{DEFAULT_SNAPSHOT}')"
            ).fetchall()
        }
        # The layer keys on url precisely because global_id is documented but
        # never published. If that changes, revisit enrich/keys.py.
        assert "global_id" not in columns
        assert "url" in columns
        # Every column is VARCHAR in the merged snapshot even where the
        # per-ATS slices are typed; the coercion in deterministic.py depends
        # on this.
        assert columns["is_remote"] == "VARCHAR"
        assert columns["lat"] == "VARCHAR"

    def test_limited_reads_are_reproducible(self) -> None:
        """``--limit N`` must return the same N rows every time.

        Regression: with ``preserve_insertion_order=false`` (needed to keep the
        full-corpus passes inside their memory budget) a parallel parquet scan
        returns whichever rows finish first. Two identical ``--limit 4000``
        reads shared only 1,952 rows, which made every limited run
        irreproducible: pilots could not be compared against each other, and
        ``delta`` reported hundreds of spurious "new" rows against an unchanged
        snapshot because each run sampled a different slice.
        """
        from enrich.snapshot import iter_rows

        def read(batch_size: int) -> list[str]:
            keys: list[str] = []
            for batch in iter_rows(
                DEFAULT_SNAPSHOT, truncate=2000, batch_size=batch_size, limit=2000
            ):
                keys.extend(str(row["job_key"]) for row in batch)
            return keys

        first = read(1000)
        assert len(first) == 2000
        assert first == read(1000)
        # Batch size is a memory knob, not part of the selection: delta and
        # pilot pass different ones and must still see the same rows.
        assert first == read(500)

    def test_urls_are_unique_enough_to_key_on(self) -> None:
        from enrich.store import connect

        con = connect(database=":memory:")
        rows, distinct = con.execute(
            f"""
            SELECT count(*), count(DISTINCT url)
            FROM (SELECT url FROM read_parquet('{DEFAULT_SNAPSHOT}')
                  USING SAMPLE 200000 ROWS (reservoir, 3))
            """
        ).fetchone()
        # Duplicate urls in a sample this size would mean job_key collides
        # across distinct postings and enrichment would overwrite itself.
        assert distinct / rows > 0.999

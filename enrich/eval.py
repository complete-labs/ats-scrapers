"""Scoring enrichment against the gold sets.

The metric that matters most is not accuracy. It is the pair
(**recall**, **over-claim rate**): how often the pipeline finds a value that
is really there, and how often it asserts one that is not. An enrichment
corpus is consumed as fact, so an over-claimed salary is a silent data
error, while a missing one is merely a gap. The harness therefore reports
them separately for every field and never blends them into one number.

Scoring rules per field type:

* **Categorical** (``placement``, ``employment_type``, ``seniority``,
  ``department``, ``education_level``, ``visa_sponsorship``, ``language``,
  ``country_iso``) — exact match. ``placement`` additionally splits by
  whether the answer was trivially visible in the title or location, since
  only the non-trivial slice says anything about comprehension.
* **Numeric money** (``salary_min``/``salary_max``) — correct within 2%,
  which absorbs "90k" versus 90,000 rounding without absorbing a real
  mistake.
* **Numeric years** (``experience_min_years``) — exact, and separately
  within one year, because "3+" versus "3-5" is a defensible disagreement.
* **Skills** — ``skills_include`` must all be present; ``skills_exclude``
  must all be absent. Soft-skill leakage is a real failure mode and this is
  what detects it.
* **``insufficient_text``** — a boolean gate scored on its own, because a
  false negative here is what sends unusable rows into the paid tier.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enrich.deterministic import run_tier0
from enrich.gold import GoldExample, load_gold
from enrich.llm import LlmClient, Request, resolve_client
from enrich.paths import REPORT_DIR
from enrich.pipeline import merge
from enrich.prompt import prompt_hash
from enrich.schema import JobEnrichment, LlmExtraction

_CATEGORICAL = (
    "placement",
    "employment_type",
    "seniority",
    "department",
    "education_level",
    "visa_sponsorship",
    "language",
    "country_iso",
    "salary_currency",
    "salary_period",
)
_MONEY = ("salary_min", "salary_max")
_YEARS = ("experience_min_years", "experience_max_years")

#: Relative tolerance for money comparisons.
MONEY_TOLERANCE = 0.02


@dataclass
class FieldScore:
    """Counts for one field. Kept as counts so slices can be summed."""

    labelled: int = 0
    #: Label is a real value and we produced it correctly.
    hit: int = 0
    #: Label is a real value and we produced a different one.
    wrong: int = 0
    #: Label is a real value and we abstained.
    missed: int = 0
    #: Label says "must be null" and we asserted something anyway.
    over_claimed: int = 0
    #: Label says "must be null" and we correctly abstained.
    correct_abstain: int = 0
    near: int = 0

    @property
    def positives(self) -> int:
        return self.hit + self.wrong + self.missed

    @property
    def negatives(self) -> int:
        return self.over_claimed + self.correct_abstain

    @property
    def recall(self) -> float | None:
        """Of values that exist, how many did we get right."""
        return self.hit / self.positives if self.positives else None

    @property
    def precision(self) -> float | None:
        """Of values we asserted, how many were right."""
        asserted = self.hit + self.wrong + self.over_claimed
        return self.hit / asserted if asserted else None

    @property
    def over_claim_rate(self) -> float | None:
        """Of cases that should have been null, how many did we fill."""
        return self.over_claimed / self.negatives if self.negatives else None

    @property
    def near_rate(self) -> float | None:
        return self.near / self.positives if self.positives else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "labelled": self.labelled,
            "hit": self.hit,
            "wrong": self.wrong,
            "missed": self.missed,
            "over_claimed": self.over_claimed,
            "correct_abstain": self.correct_abstain,
            "recall": self.recall,
            "precision": self.precision,
            "over_claim_rate": self.over_claim_rate,
            "near_rate": self.near_rate,
        }


def _money_equal(predicted: float, expected: float) -> bool:
    if expected == 0:
        return abs(predicted) < 1e-9
    return abs(predicted - expected) / abs(expected) <= MONEY_TOLERANCE


def _compare(field_name: str, predicted: Any, expected: Any) -> tuple[bool, bool]:
    """Return ``(correct, near)`` for one field."""
    if field_name in _MONEY:
        return (_money_equal(float(predicted), float(expected)), False)
    if field_name in _YEARS:
        exact = int(predicted) == int(expected)
        return (exact, abs(int(predicted) - int(expected)) <= 1)
    if field_name == "is_remote":
        return (bool(predicted) is bool(expected), False)
    if field_name in _CATEGORICAL:
        return (str(predicted).upper() == str(expected).upper(), False)
    return (predicted == expected, False)


@dataclass
class EvalReport:
    model_id: str
    truncate: int
    prompt_hash: str
    generated_at: str
    examples: int = 0
    fields: dict[str, FieldScore] = field(default_factory=dict)
    slices: dict[str, dict[str, FieldScore]] = field(default_factory=dict)
    skills_pass: int = 0
    skills_total: int = 0
    insufficient: FieldScore = field(default_factory=FieldScore)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def _bucket(self, slice_name: str, field_name: str) -> FieldScore:
        self.slices.setdefault(slice_name, {})
        self.slices[slice_name].setdefault(field_name, FieldScore())
        return self.slices[slice_name][field_name]

    def observe(
        self,
        example: GoldExample,
        row: JobEnrichment,
        extraction: LlmExtraction | None,
    ) -> None:
        self.examples += 1
        slice_names = [f"source:{example.source}"]
        if example.ats_type:
            slice_names.append(f"ats:{example.ats_type}")
        language = row.language or "unknown"
        slice_names.append(f"lang:{language}")

        for field_name, expected in example.labels.items():
            if field_name in ("skills_include", "skills_exclude"):
                continue
            if field_name == "insufficient_text":
                observed = bool(extraction.insufficient_text) if extraction else False
                self.insufficient.labelled += 1
                if bool(expected) == observed:
                    self.insufficient.hit += 1
                else:
                    self.insufficient.wrong += 1
                    self.failures.append(
                        {
                            "id": example.id,
                            "field": "insufficient_text",
                            "expected": expected,
                            "predicted": observed,
                        }
                    )
                continue

            predicted = getattr(row, field_name, None)
            buckets = [self.fields.setdefault(field_name, FieldScore())]
            if field_name == "placement":
                buckets.append(
                    self._bucket(
                        "placement_trivial" if example.placement_trivial else "placement_hard",
                        field_name,
                    )
                )
            for slice_name in slice_names:
                buckets.append(self._bucket(slice_name, field_name))
            for bucket in buckets:
                bucket.labelled += 1

            if expected is None:
                for bucket in buckets:
                    if predicted is None:
                        bucket.correct_abstain += 1
                    else:
                        bucket.over_claimed += 1
                if predicted is not None:
                    self.failures.append(
                        {
                            "id": example.id,
                            "field": field_name,
                            "expected": None,
                            "predicted": predicted,
                            "note": example.note,
                        }
                    )
                continue

            if predicted is None:
                for bucket in buckets:
                    bucket.missed += 1
                continue

            try:
                correct, near = _compare(field_name, predicted, expected)
            except (TypeError, ValueError):
                correct, near = False, False
            for bucket in buckets:
                if correct:
                    bucket.hit += 1
                    bucket.near += 1
                else:
                    bucket.wrong += 1
                    if near:
                        bucket.near += 1
            if not correct:
                self.failures.append(
                    {
                        "id": example.id,
                        "field": field_name,
                        "expected": expected,
                        "predicted": predicted,
                        "note": example.note,
                    }
                )

        include = example.labels.get("skills_include")
        exclude = example.labels.get("skills_exclude")
        if include is not None or exclude is not None:
            self.skills_total += 1
            skills = {skill.lower() for skill in row.skills}
            ok = True
            if include:
                ok = ok and all(term.lower() in skills for term in include)
            if exclude:
                ok = ok and not any(term.lower() in skills for term in exclude)
            if ok:
                self.skills_pass += 1
            else:
                self.failures.append(
                    {
                        "id": example.id,
                        "field": "skills",
                        "expected": {"include": include, "exclude": exclude},
                        "predicted": sorted(skills),
                    }
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "truncate": self.truncate,
            "prompt_hash": self.prompt_hash,
            "generated_at": self.generated_at,
            "examples": self.examples,
            "fields": {name: score.as_dict() for name, score in sorted(self.fields.items())},
            "slices": {
                slice_name: {name: score.as_dict() for name, score in sorted(scores.items())}
                for slice_name, scores in sorted(self.slices.items())
            },
            "skills": {
                "passed": self.skills_pass,
                "total": self.skills_total,
                "rate": (self.skills_pass / self.skills_total) if self.skills_total else None,
            },
            "insufficient_text": self.insufficient.as_dict(),
            "failures": self.failures,
        }


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def render_markdown(report: EvalReport) -> str:
    data = report.as_dict()
    lines = [
        "# Enrichment eval",
        "",
        f"- Model: `{data['model_id']}`",
        f"- Prompt: `{data['prompt_hash']}` (truncate={data['truncate']})",
        f"- Generated: {data['generated_at']}",
        f"- Examples scored: {data['examples']}",
        "",
        "## Per field",
        "",
        "`recall` = of values that exist, how many we got right. "
        "`over-claim` = of cases that should be null, how many we filled anyway.",
        "",
        "| field | labelled | recall | precision | over-claim | missed | wrong |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for name, score in data["fields"].items():
        lines.append(
            f"| `{name}` | {score['labelled']} | {_fmt(score['recall'])} | "
            f"{_fmt(score['precision'])} | {_fmt(score['over_claim_rate'])} | "
            f"{score['missed']} | {score['wrong']} |"
        )
    lines.append("")

    insufficient = data["insufficient_text"]
    if insufficient["labelled"]:
        lines.append(
            f"- `insufficient_text` gate: {insufficient['hit']}/{insufficient['labelled']} correct"
        )
    skills = data["skills"]
    if skills["total"]:
        lines.append(f"- skills include/exclude: {skills['passed']}/{skills['total']} passed")
    lines.append("")

    hard = data["slices"].get("placement_hard", {}).get("placement")
    trivial = data["slices"].get("placement_trivial", {}).get("placement")
    if hard or trivial:
        lines.append("## Placement, split by difficulty")
        lines.append("")
        lines.append(
            "Only the non-trivial row is informative: the trivial slice is "
            "answerable by keyword matching on the title or location."
        )
        lines.append("")
        if trivial:
            lines.append(
                f"- trivial (token visible): recall {_fmt(trivial['recall'])} "
                f"on {trivial['labelled']} labels"
            )
        if hard:
            lines.append(
                f"- non-trivial: recall {_fmt(hard['recall'])} on {hard['labelled']} labels"
            )
        lines.append("")

    language_slices = {
        name.split(":", 1)[1]: scores
        for name, scores in data["slices"].items()
        if name.startswith("lang:")
    }
    if len(language_slices) > 1:
        lines.append("## By language")
        lines.append("")
        lines.append("| language | field | labelled | recall |")
        lines.append("|---|---|--:|--:|")
        for language, scores in sorted(language_slices.items()):
            for name, score in sorted(scores.items()):
                if score["labelled"] >= 3:
                    lines.append(
                        f"| `{language}` | `{name}` | {score['labelled']} | "
                        f"{_fmt(score['recall'])} |"
                    )
        lines.append("")

    if data["failures"]:
        lines.append(f"## Failures ({len(data['failures'])})")
        lines.append("")
        for failure in data["failures"][:40]:
            lines.append(
                f"- `{failure['id']}` **{failure['field']}**: expected "
                f"`{failure['expected']}`, got `{failure['predicted']}`"
                + (f" — {failure['note']}" if failure.get("note") else "")
            )
        lines.append("")
    return "\n".join(lines)


def evaluate(
    examples: Sequence[GoldExample],
    client: LlmClient,
    *,
    truncate: int = 2000,
) -> EvalReport:
    """Run the real pipeline over gold examples and score the output."""
    digest = prompt_hash(truncate=truncate)
    report = EvalReport(
        model_id=client.model_id,
        truncate=truncate,
        prompt_hash=digest,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )
    if not examples:
        return report

    requests = [Request.from_row(example.masked_row, truncate=truncate) for example in examples]
    completions = client.complete(requests)
    by_hash: dict[str, LlmExtraction] = {}
    for completion in completions:
        if completion.extraction is not None:
            by_hash[completion.content_hash] = completion.extraction

    for example in examples:
        masked = example.masked_row
        tier0 = run_tier0(masked)
        extraction = by_hash.get(str(masked["content_hash"]))
        row = merge(
            masked,
            tier0,
            extraction,
            tier="tier1" if extraction is not None else "tier0",
            model_id=client.model_id,
            prompt_hash=digest,
        )
        report.observe(example, row, extraction)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default="auto", choices=("auto", "openai", "stub"))
    parser.add_argument("--truncate", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)

    examples = load_gold(truncate=args.truncate, include_holdout=not args.edge_only)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("no gold examples found", file=sys.stderr)
        return 1

    client = resolve_client(args.client)
    report = evaluate(examples, client, truncate=args.truncate)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval.json").write_text(json.dumps(report.as_dict(), indent=2, default=str))
    markdown = render_markdown(report)
    (args.out_dir / "eval.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Assembling tiers into an enrichment row, and running that over a corpus.

This module owns two things only: the merge rule that turns Tier 0 findings
plus an optional model extraction into a :class:`JobEnrichment`, and the
batch loop that applies it. Keeping the merge in one place matters because
precedence is where an enrichment pipeline quietly goes wrong — the whole
value of the sidecar is that a provider's structured value is never
overwritten by a guess.

Precedence, highest first:

1. **Provider.** The ATS said it in a structured field. Never overridden.
2. **Tier 0.** A deterministic rule with no judgement in it.
3. **Tier 1/2.** The model, for prose-only fields.

The one deliberate exception is ``placement``: Tier 0 only ever reads the
title and location, so a model that has read the body and found "this role
requires three days per week in our Berlin office" is better evidence than a
title keyword. A Tier 0 placement derived from a *keyword* therefore yields
to the model, while a provider-supplied one does not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from enrich._version import ENRICHMENT_VERSION
from enrich.deterministic import Tier0Result, run_tier0
from enrich.schema import JobEnrichment, LlmExtraction, Tier

#: Fields the model may fill, and whether it may override a Tier 0 value.
#: Everything not listed here is Tier 0's or the provider's alone.
_MODEL_FIELDS: tuple[tuple[str, bool], ...] = (
    ("placement", True),
    ("employment_type", False),
    ("experience_min_years", False),
    ("experience_max_years", False),
    ("seniority", False),
    ("salary_min", False),
    ("salary_max", False),
    ("salary_currency", False),
    ("salary_period", False),
    ("department", False),
    ("education_level", False),
    ("visa_sponsorship", False),
)


def _global_id(ats_type: object, ats_id: object) -> str | None:
    """Rebuild the ``{ats_type}:{ats_id}`` composite the dataset documents
    but does not ship."""
    if not ats_type or not ats_id:
        return None
    return f"{ats_type}:{ats_id}"


@dataclass
class MergeOutcome:
    row: JobEnrichment
    tier0: Tier0Result


def merge(
    row: dict[str, Any],
    tier0: Tier0Result,
    extraction: LlmExtraction | None = None,
    *,
    tier: Tier = "tier0",
    model_id: str | None = None,
    prompt_hash: str | None = None,
) -> JobEnrichment:
    """Build the stored row from a snapshot row, Tier 0, and the model."""
    values: dict[str, Any] = dict(tier0.values)
    sources: dict[str, str] = dict(tier0.sources)
    evidence: dict[str, str] = {}
    needs_review = False
    review_reason: str | None = None

    if extraction is not None:
        if extraction.insufficient_text:
            needs_review = True
            review_reason = "model reported insufficient text"
        else:
            for name, may_override in _MODEL_FIELDS:
                proposed = getattr(extraction, name, None)
                if proposed is None:
                    continue
                current_source = sources.get(name)
                if current_source == "provider":
                    continue
                if current_source is not None and not may_override:
                    continue
                values[name] = proposed
                sources[name] = tier

            if extraction.skills:
                values["skills"] = [
                    skill.strip().lower()
                    for skill in extraction.skills
                    if isinstance(skill, str) and skill.strip()
                ][:12]
                sources["skills"] = tier

            for name, quote in (
                ("salary", extraction.evidence.salary_quote),
                ("placement", extraction.evidence.placement_quote),
                ("experience", extraction.evidence.experience_quote),
            ):
                if quote and quote.strip():
                    evidence[name] = quote.strip()[:400]

            # A salary claim with no supporting quote is the highest-risk
            # output this pipeline can produce: it is a number a consumer
            # will act on, and nothing downstream can tell it apart from a
            # provider-stated one. Flag it rather than trusting it.
            if (
                values.get("salary_min") is not None
                and sources.get("salary_min") == tier
                and "salary" not in evidence
            ):
                needs_review = True
                review_reason = "model salary without supporting quote"

    # ``is_remote`` and ``placement`` must never contradict each other in the
    # output. Which one yields depends on who supplied ``is_remote``.
    placement = values.get("placement")
    if sources.get("is_remote") == "provider":
        provider_remote = values.get("is_remote")
        if provider_remote is False and placement == "remote":
            # The ATS shipped a structured "not remote" flag and the model
            # read the body as remote. The provider wins — it is the
            # employer's own field — but the disagreement is worth seeing,
            # because a systematic pattern here means a scraper is mapping
            # the provider's workplace type wrongly.
            values["placement"] = "hybrid"
            needs_review = True
            review_reason = "model said remote but provider flag is not remote"
        elif provider_remote is True and placement is not None and placement != "remote":
            values["placement"] = "remote"
            needs_review = True
            review_reason = "model contradicted provider remote flag"
    elif placement is not None:
        values["is_remote"] = placement == "remote"
        sources.setdefault("is_remote", sources.get("placement", tier))

    experience_min = values.get("experience_min_years")
    experience_max = values.get("experience_max_years")
    if (
        experience_min is not None
        and experience_max is not None
        and experience_max < experience_min
    ):
        values["experience_max_years"] = experience_min
        values["experience_min_years"] = experience_max

    salary_min = values.get("salary_min")
    salary_max = values.get("salary_max")
    if salary_min is not None and salary_max is not None and salary_max < salary_min:
        values["salary_min"], values["salary_max"] = salary_max, salary_min

    return JobEnrichment(
        job_key=str(row["job_key"]),
        content_hash=str(row["content_hash"]),
        fallback_key=row.get("fallback_key"),
        url=str(row.get("url") or ""),
        ats_type=row.get("ats_type"),
        ats_id=row.get("ats_id"),
        global_id=_global_id(row.get("ats_type"), row.get("ats_id")),
        sources=sources,  # type: ignore[arg-type]
        evidence=evidence,
        tier=tier,
        needs_review=needs_review,
        review_reason=review_reason,
        enrichment_version=ENRICHMENT_VERSION,
        model_id=model_id,
        prompt_hash=prompt_hash,
        enriched_at=datetime.now(tz=UTC),
        **values,
    )


def tier0_only(rows: Iterable[dict[str, Any]]) -> list[JobEnrichment]:
    """Run Tier 0 over rows and build stored rows with no model involved."""
    output: list[JobEnrichment] = []
    for row in rows:
        result = run_tier0(row)
        output.append(merge(row, result, None, tier="tier0"))
    return output


@dataclass
class CoverageCounter:
    """Field coverage accumulated across a run.

    Tracked per source as well as per field: "92% of rows have a region" is
    only meaningful alongside "and 91 points of that came from Tier 0", which
    is the number that justifies not having paid for it.
    """

    total: int = 0
    filled: dict[str, int] = None  # type: ignore[assignment]
    by_source: dict[str, dict[str, int]] = None  # type: ignore[assignment]
    needs_llm: int = 0

    def __post_init__(self) -> None:
        self.filled = {}
        self.by_source = {}

    def observe(self, row: JobEnrichment, *, needs_llm: bool = False) -> None:
        self.total += 1
        if needs_llm:
            self.needs_llm += 1
        for name in (
            "language",
            "country_iso",
            "region",
            "lat",
            "placement",
            "is_remote",
            "employment_type",
            "experience_min_years",
            "salary_min",
            "salary_currency",
            "salary_period",
            "department",
            "seniority",
            "skills",
        ):
            value = getattr(row, name, None)
            if value is None or (isinstance(value, list) and not value):
                continue
            self.filled[name] = self.filled.get(name, 0) + 1
            source = row.sources.get(name, "unknown")
            self.by_source.setdefault(name, {})
            self.by_source[name][source] = self.by_source[name].get(source, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.total,
            "needs_llm": self.needs_llm,
            "needs_llm_pct": (self.needs_llm / self.total) if self.total else 0.0,
            "coverage": {name: count / self.total for name, count in sorted(self.filled.items())}
            if self.total
            else {},
            "by_source": self.by_source,
        }


def run_tier0_over(
    batches: Iterable[Sequence[dict[str, Any]]],
    *,
    on_rows: Callable[[list[JobEnrichment]], None] | None = None,
    progress: Callable[[int], None] | None = None,
) -> CoverageCounter:
    """Stream Tier 0 over batches, handing finished rows to ``on_rows``."""
    counter = CoverageCounter()
    for batch in batches:
        built: list[JobEnrichment] = []
        for row in batch:
            result = run_tier0(row)
            enriched = merge(row, result, None, tier="tier0")
            counter.observe(enriched, needs_llm=result.needs_llm)
            built.append(enriched)
        if on_rows is not None:
            on_rows(built)
        if progress is not None:
            progress(len(built))
    return counter

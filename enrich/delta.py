"""Phase 5: enrich only what changed.

Steady state is a small fraction of the backfill, and the reason is the
content hash. A daily refresh of the snapshot mostly repeats yesterday's
postings verbatim; a row only needs paid work when it is new, or when its
title/description/location actually changed. Both conditions are one join
away:

    new      -> job_key absent from the sidecar
    changed  -> job_key present but content_hash differs
    stale    -> enrichment_version or prompt_hash no longer current

The third case is the one that makes prompt iteration affordable. Because
the cache is keyed by ``(content_hash, prompt_hash, model_id)``, editing the
prompt does not invalidate the *corpus* — it invalidates the answers, and
rows whose new prompt hash is already cached (because another row shared the
body) still cost nothing.

Tier 0 always re-runs for changed rows. It is free, so there is no reason to
be clever about it.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from enrich._version import ENRICHMENT_VERSION
from enrich.deterministic import run_tier0
from enrich.llm import ModelConfig, Request, resolve_client
from enrich.paths import REPORT_DIR
from enrich.pipeline import CoverageCounter, merge
from enrich.prompt import prompt_hash
from enrich.snapshot import iter_rows
from enrich.store import EnrichmentStore

log = logging.getLogger(__name__)

MIN_ENRICHABLE_CHARS = 200


@dataclass
class DeltaCounts:
    seen: int = 0
    new: int = 0
    changed: int = 0
    stale: int = 0
    unchanged: int = 0

    @property
    def to_enrich(self) -> int:
        return self.new + self.changed + self.stale

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "new": self.new,
            "changed": self.changed,
            "stale_prompt_or_version": self.stale,
            "unchanged": self.unchanged,
            "to_enrich": self.to_enrich,
            "delta_pct": round(self.to_enrich / self.seen * 100, 2) if self.seen else 0.0,
        }


def _existing_state(store: EnrichmentStore) -> dict[str, tuple[str, int, str | None]]:
    """Snapshot of what the sidecar already knows, keyed by ``job_key``."""
    rows = store.con.execute(
        "SELECT job_key, content_hash, enrichment_version, prompt_hash FROM enrichment"
    ).fetchall()
    return {
        str(job_key): (
            str(content_hash),
            int(version or 0),
            prompt if prompt is None else str(prompt),
        )
        for job_key, content_hash, version, prompt in rows
    }


def classify(
    batches: Iterator[list[dict[str, Any]]],
    state: dict[str, tuple[str, int, str | None]],
    *,
    current_prompt_hash: str,
) -> tuple[list[dict[str, Any]], DeltaCounts]:
    """Split incoming rows into those needing work and those that do not."""
    counts = DeltaCounts()
    todo: list[dict[str, Any]] = []
    for batch in batches:
        for row in batch:
            counts.seen += 1
            key = str(row["job_key"])
            known = state.get(key)
            if known is None:
                counts.new += 1
                todo.append(row)
                continue
            known_content, known_version, known_prompt = known
            if known_content != str(row["content_hash"]):
                counts.changed += 1
                todo.append(row)
            elif known_version < ENRICHMENT_VERSION or (
                known_prompt is not None and known_prompt != current_prompt_hash
            ):
                counts.stale += 1
                todo.append(row)
            else:
                counts.unchanged += 1
    return todo, counts


def run_delta(args: argparse.Namespace) -> int:
    """Classify the snapshot against the sidecar, then enrich only the delta."""
    store = EnrichmentStore(args.db)
    digest = prompt_hash(truncate=args.truncate)
    client = resolve_client(args.client, concurrency=args.concurrency, model_id=args.model)
    config: ModelConfig = getattr(client, "config", ModelConfig.from_env(args.model))

    log.info("loading sidecar state")
    state = _existing_state(store)
    log.info("sidecar knows %s rows", f"{len(state):,}")

    batches = iter_rows(
        args.snapshot,
        truncate=args.truncate,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    todo, counts = classify(batches, state, current_prompt_hash=digest)
    log.info("delta: %s", counts.as_dict())

    if args.max_rows and len(todo) > args.max_rows:
        log.warning(
            "delta of %s rows exceeds --max-rows %s; truncating. A delta this "
            "large usually means a prompt or version bump, not a daily change.",
            f"{len(todo):,}",
            f"{args.max_rows:,}",
        )
        todo = todo[: args.max_rows]

    if not todo:
        print(json.dumps({"delta": counts.as_dict(), "enriched": 0}, indent=2))
        store.close()
        return 0

    # Tier 0 for everything (free), Tier 1 only for rows with enough body.
    enrichable = [
        row for row in todo if len(str(row.get("description") or "")) >= MIN_ENRICHABLE_CHARS
    ]
    unique: dict[str, dict[str, Any]] = {}
    for row in enrichable:
        unique.setdefault(str(row["content_hash"]), row)

    cached = store.cache_get_many(unique, prompt_hash=digest, model_id=client.model_id)
    pending = [row for key, row in unique.items() if key not in cached]
    log.info(
        "tier1: %s unique bodies, %s cached, %s to fetch",
        f"{len(unique):,}",
        f"{len(cached):,}",
        f"{len(pending):,}",
    )

    completions = (
        client.complete([Request.from_row(row, truncate=args.truncate) for row in pending])
        if pending
        else []
    )
    input_tokens = sum(c.input_tokens for c in completions)
    output_tokens = sum(c.output_tokens for c in completions)
    cost = config.cost(input_tokens, output_tokens)
    store.cache_put_many(
        [
            (
                c.content_hash,
                c.extraction,  # type: ignore[arg-type]
                c.input_tokens,
                c.output_tokens,
                config.cost(c.input_tokens, c.output_tokens),
            )
            for c in completions
            if c.ok
        ],
        prompt_hash=digest,
        model_id=client.model_id,
    )
    if completions:
        store.record_cost(
            stage="delta",
            calls=len(completions),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_id=client.model_id,
            note=json.dumps(counts.as_dict()),
        )

    extractions = dict(cached)
    for completion in completions:
        if completion.ok:
            extractions[completion.content_hash] = completion.extraction  # type: ignore[assignment]

    counter = CoverageCounter()
    built = []
    for row in todo:
        extraction = extractions.get(str(row["content_hash"]))
        tier0 = run_tier0(row)
        enriched = merge(
            row,
            tier0,
            extraction,
            tier="tier1" if extraction is not None else "tier0",
            model_id=client.model_id if extraction is not None else None,
            prompt_hash=digest if extraction is not None else None,
        )
        counter.observe(enriched)
        built.append(enriched)
        if len(built) >= 20_000:
            store.upsert(built)
            built = []
    if built:
        store.upsert(built)

    summary = {
        "delta": counts.as_dict(),
        "enriched": len(todo),
        "unique_bodies": len(unique),
        "cache_hits": len(cached),
        "calls_made": len(completions),
        "cost_usd": round(cost, 4),
        "coverage": {k: round(v, 4) for k, v in counter.as_dict()["coverage"].items()},
        "model_id": client.model_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "delta.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    store.close()
    return 0

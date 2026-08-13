"""Command line entry point for the enrichment layer.

    uv run python -m enrich.run <command>

Commands, in the order you would run them:

    profile        quantify the gap in the published snapshot
    tier0          deterministic pass over the corpus (free)
    gold-build     assemble the provider-labelled holdout
    eval           score the pipeline against the gold sets
    dedup          report how many unique bodies Tier 1 must pay for
    pilot          Tier 1 on a stratified sample, with a cost projection
    batch-write    write Batch API shards for the full backfill
    batch-submit   submit written shards
    batch-status   poll submitted shards
    batch-collect  ingest finished shards into the cache and apply them
    agent          Tier 2 recovery for rows Tier 1 could not resolve
    join           write all_enriched.parquet
    delta          enrich only what changed since the last run
    stats          coverage and spend so far

Every command defaults to the offline stub client, so nothing spends money
until ``--client openai`` is passed with a key configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from enrich import profile as profile_module
from enrich._version import ENRICHMENT_VERSION
from enrich.deterministic import run_tier0
from enrich.gold.loader import HOLDOUT_PATH, GoldStats, build_holdout, load_gold, write_holdout
from enrich.llm import (
    MAX_SHARD_BYTES,
    MAX_SHARD_REQUESTS,
    BatchClient,
    ModelConfig,
    Request,
    batch_request_line,
    parse_batch_output,
    reported_model,
    require_api_key,
    resolve_client,
)
from enrich.paths import (
    BATCH_DIR,
    DEFAULT_DB,
    DEFAULT_OUTPUT,
    DEFAULT_SNAPSHOT,
    REPORT_DIR,
    ensure_dirs,
)
from enrich.pipeline import CoverageCounter, merge, tier0_only
from enrich.prompt import estimate_tokens, prompt_hash, render_user_prompt
from enrich.snapshot import count_rows, iter_rows, sample_rows
from enrich.store import EnrichmentStore

log = logging.getLogger("enrich")

#: Rows with less body than this cannot be enriched by a model at all; they
#: are Tier 2 candidates or permanent gaps. Matches the threshold the Phase 0
#: profile reports as "thin description".
MIN_ENRICHABLE_CHARS = 200


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


# --- tier 0 -----------------------------------------------------------------


def _tier0_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Worker body. Returns dumped rows because pydantic models pickle
    slowly and the parent only needs to insert them."""
    return [row.model_dump() for row in tier0_only(rows)]


def _bounded_map(
    pool: ProcessPoolExecutor,
    batches: Iterator[list[dict[str, Any]]],
    *,
    in_flight: int,
) -> Iterator[list[dict[str, Any]]]:
    """Map over batches with a bounded number of pending futures.

    A plain ``pool.map`` over the reader would pull the entire 4.85M-row
    snapshot into memory as pending work. This keeps at most ``in_flight``
    batches resident.
    """
    pending = set()
    try:
        for _ in range(in_flight):
            try:
                pending.add(pool.submit(_tier0_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    pending.add(pool.submit(_tier0_batch, next(batches)))
                except StopIteration:
                    continue
    finally:
        for future in pending:
            future.cancel()


def cmd_tier0(args: argparse.Namespace) -> int:
    """Run the deterministic pass and persist it."""
    from enrich.schema import JobEnrichment

    store = EnrichmentStore(args.db)
    total = args.limit or count_rows(args.snapshot)
    log.info("tier0 over %s rows with %d workers", f"{total:,}", args.workers)

    counter = CoverageCounter()
    written = 0
    batches = iter_rows(
        args.snapshot,
        truncate=args.truncate,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    def _persist(dumped: list[dict[str, Any]]) -> None:
        nonlocal written
        rows = [JobEnrichment.model_validate(record) for record in dumped]
        for row in rows:
            counter.observe(row)
        store.upsert(rows)
        written += len(rows)
        if written % (args.batch_size * 10) < args.batch_size:
            log.info("  %s / %s rows", f"{written:,}", f"{total:,}")

    if args.workers <= 1:
        for batch in batches:
            _persist(_tier0_batch(batch))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for dumped in _bounded_map(pool, batches, in_flight=args.workers * 2):
                _persist(dumped)

    report = counter.as_dict()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tier0_coverage.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("tier0 done: %s rows", f"{written:,}")
    print(json.dumps({k: round(v, 4) for k, v in report["coverage"].items()}, indent=1))
    store.close()
    return 0


# --- gold -------------------------------------------------------------------


def cmd_gold_build(args: argparse.Namespace) -> int:
    """Assemble the provider-labelled holdout from the snapshot."""
    # Oversample: only a fraction of rows carry a provider value for any
    # given target field, and each row is assigned to exactly one target.
    pool_size = max(args.per_target * 200, 20_000)
    log.info("sampling %s rows to find provider-labelled examples", f"{pool_size:,}")
    rows = sample_rows(
        args.snapshot,
        n=pool_size,
        truncate=args.truncate,
        seed=args.seed,
    )
    records = build_holdout(rows, per_target=args.per_target, truncate=args.truncate)
    path = write_holdout(records, HOLDOUT_PATH)
    counts: dict[str, int] = {}
    for record in records:
        for name in record["labels"]:
            counts[name] = counts.get(name, 0) + 1
    print(f"wrote {len(records)} holdout examples to {path}")
    print(json.dumps(counts, indent=1))

    stats = GoldStats.of(load_gold(truncate=args.truncate))
    print(
        f"\ngold total: {stats.total} "
        f"({stats.by_source.get('edge', 0)} hand-written, "
        f"{stats.by_source.get('holdout', 0)} provider-labelled)"
    )
    print(f"negative (must-abstain) labels: {sum(stats.negative_labels.values())}")
    return 0


# --- dedup ------------------------------------------------------------------


def cmd_dedup(args: argparse.Namespace) -> int:
    """Report how many unique bodies the paid tier must actually cover."""
    result = profile_module.dedup_savings(args.snapshot, truncate=args.truncate, sample=args.sample)
    print(json.dumps(result, indent=2))
    calls = result["distinct_enrichable"]
    rows = result["enrichable_rows"]
    print(
        f"\nTier 1 must pay for {calls:,} calls to cover {rows:,} rows "
        f"({result['enrichable_dedup_ratio'] * 100:.1f}% saved by dedup)."
    )
    return 0


# --- tier 1 -----------------------------------------------------------------


def _project_cost(
    rows: Sequence[dict[str, Any]],
    *,
    truncate: int,
    config: ModelConfig,
    unique_calls: int,
    total_calls: int,
) -> dict[str, Any]:
    """Extrapolate spend from a measured sample.

    Output tokens are estimated from the schema's typical filled size rather
    than measured, because a dry run produces none.
    """
    input_tokens = [estimate_tokens(render_user_prompt(row, truncate=truncate)) for row in rows]
    mean_input = sum(input_tokens) / len(input_tokens) if input_tokens else 0
    mean_output = 180
    per_call = config.cost(int(mean_input), mean_output, batch=True)
    return {
        "mean_input_tokens": round(mean_input, 1),
        "assumed_output_tokens": mean_output,
        "model_priced": config.model_id,
        "cost_per_call_batch_usd": round(per_call, 6),
        "sample_calls": unique_calls,
        "projected_calls": total_calls,
        "projected_cost_batch_usd": round(per_call * total_calls, 2),
        "projected_cost_sync_usd": round(
            config.cost(int(mean_input), mean_output) * total_calls, 2
        ),
        "pricing_note": (
            "Prices come from ModelConfig defaults or ATS_ENRICH_PRICE_IN/OUT. "
            "Verify against your contracted rate before trusting this figure."
        ),
    }


def cmd_pilot(args: argparse.Namespace) -> int:
    """Tier 1 over a stratified sample, then score and project cost."""
    store = EnrichmentStore(args.db)
    client = resolve_client(args.client, concurrency=args.concurrency, model_id=args.model)
    digest = prompt_hash(truncate=args.truncate)
    # Two configs, deliberately. ``config`` prices what this run actually
    # spent (the stub is free, and reporting otherwise would be a lie).
    # ``pricing`` is the real model's rate card, used for the backfill
    # projection — a projection built from the stub's zero cost is useless.
    config = getattr(client, "config", ModelConfig.from_env(args.model))
    pricing = ModelConfig.from_env(args.model)

    where = f"length(description) >= {MIN_ENRICHABLE_CHARS}"
    rows = sample_rows(args.snapshot, n=args.n, truncate=args.truncate, seed=args.seed, where=where)
    log.info("pilot sample: %d rows", len(rows))

    # Deduplicate by content hash before dispatch: this is the only place
    # the dedup ratio turns into money.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["content_hash"]), row)
    log.info("%d unique bodies (%.1f%% dedup)", len(unique), (1 - len(unique) / len(rows)) * 100)

    cached = store.cache_get_many(unique, prompt_hash=digest, model_id=client.model_id)
    todo = [row for key, row in unique.items() if key not in cached]
    log.info("%d cached, %d to fetch", len(cached), len(todo))

    projection = _project_cost(
        rows,
        truncate=args.truncate,
        config=pricing,
        unique_calls=len(unique),
        total_calls=args.projected_calls,
    )

    if args.dry_run:
        print(json.dumps(projection, indent=2))
        store.close()
        return 0

    requests = [Request.from_row(row, truncate=args.truncate) for row in todo]
    completions = client.complete(requests)

    ok = [c for c in completions if c.ok]
    failed = [c for c in completions if not c.ok]
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
            for c in ok
        ],
        prompt_hash=digest,
        model_id=client.model_id,
    )
    store.record_cost(
        stage="pilot",
        calls=len(completions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        model_id=client.model_id,
        note=f"n={args.n}",
    )

    extractions = dict(cached)
    for completion in ok:
        extractions[completion.content_hash] = completion.extraction  # type: ignore[assignment]

    built = []
    for row in rows:
        extraction = extractions.get(str(row["content_hash"]))
        tier0 = run_tier0(row)
        built.append(
            merge(
                row,
                tier0,
                extraction,
                tier="tier1" if extraction is not None else "tier0",
                model_id=client.model_id if extraction is not None else None,
                prompt_hash=digest if extraction is not None else None,
            )
        )
    store.upsert(built)

    counter = CoverageCounter()
    for row in built:
        counter.observe(row)

    summary = {
        "rows": len(rows),
        "unique_bodies": len(unique),
        "calls_made": len(completions),
        "calls_failed": len(failed),
        "cached_hits": len(cached),
        "model_id": client.model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "measured_cost_usd": round(cost, 4),
        "needs_review": sum(1 for row in built if row.needs_review),
        "coverage": {k: round(v, 4) for k, v in counter.as_dict()["coverage"].items()},
        "projection": projection,
    }
    if failed:
        summary["sample_errors"] = [c.error for c in failed[:5]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "pilot.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    store.close()
    return 0


# --- batch ------------------------------------------------------------------


def cmd_batch_write(args: argparse.Namespace) -> int:
    """Write Batch API shards for every unique body not already cached."""
    ensure_dirs()
    store = EnrichmentStore(args.db)
    digest = prompt_hash(truncate=args.truncate)
    model_id = args.model or ModelConfig.from_env().model_id
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    shard_index = 0
    buffer: list[str] = []
    buffer_bytes = 0
    written_shards = 0
    skipped_cached = 0
    # Roll over on whichever ceiling is hit first. Bytes bind long before
    # count: every line repeats the system prompt and the full JSON schema.
    max_requests = min(args.shard_size, MAX_SHARD_REQUESTS)

    def _flush() -> None:
        nonlocal shard_index, buffer, buffer_bytes, written_shards
        if not buffer:
            return
        path = shard_dir / f"shard-{shard_index:05d}.jsonl"
        # Shards are written the same way with or without a key: the JSONL is
        # exactly what the provider expects, so its shape is verifiable
        # offline and submission is a separate, deliberate step.
        path.write_text("\n".join(buffer) + "\n", encoding="utf-8")
        store.record_batch(
            batch_id=f"local:{path.name}",
            shard=path.name,
            status="written",
            request_count=len(buffer),
            model_id=model_id,
            prompt_hash=digest,
            input_path=str(path),
        )
        log.info(
            "wrote %s (%d requests, %.1f MB)",
            path.name,
            len(buffer),
            buffer_bytes / 1024 / 1024,
        )
        written_shards += 1
        shard_index += 1
        buffer = []
        buffer_bytes = 0

    where = f"length(description) >= {MIN_ENRICHABLE_CHARS}"
    for batch in iter_rows(
        args.snapshot,
        truncate=args.truncate,
        batch_size=args.batch_size,
        limit=args.limit,
        where=where,
    ):
        keys = [str(row["content_hash"]) for row in batch]
        cached = store.cache_get_many(keys, prompt_hash=digest, model_id=model_id)
        for row in batch:
            key = str(row["content_hash"])
            if key in seen:
                continue
            seen.add(key)
            if key in cached:
                skipped_cached += 1
                continue
            line = batch_request_line(
                Request.from_row(row, truncate=args.truncate), model_id=model_id
            )
            encoded = len(line.encode("utf-8")) + 1
            if buffer and (len(buffer) >= max_requests or buffer_bytes + encoded > MAX_SHARD_BYTES):
                _flush()
            buffer.append(line)
            buffer_bytes += encoded
    _flush()

    print(
        json.dumps(
            {
                "unique_bodies": len(seen),
                "already_cached": skipped_cached,
                "shards_written": written_shards,
                "shard_dir": str(shard_dir),
                "prompt_hash": digest,
                "model_id": model_id,
            },
            indent=2,
        )
    )
    store.close()
    return 0


def cmd_batch_simulate(args: argparse.Namespace) -> int:
    """Produce a provider-shaped output file for a shard, using the stub.

    Exists so the collect -> cache -> apply half of the backfill can be
    exercised and regression-tested without an account. Submission is the only
    step that genuinely requires one, and a pipeline whose second half is
    unverifiable until the moment you spend money is a pipeline that fails
    then.
    """
    from enrich.llm import StubClient

    shard = Path(args.shard)
    client = StubClient()
    lines: list[str] = []
    with shard.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            body = record.get("body") or {}
            messages = body.get("messages") or []
            user_prompt = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"), ""
            )
            request = Request(
                content_hash=str(record.get("custom_id") or ""), user_prompt=user_prompt
            )
            completion = client.complete([request])[0]
            payload = (
                completion.extraction.model_dump_json()
                if completion.extraction is not None
                else "{}"
            )
            lines.append(
                json.dumps(
                    {
                        "id": f"sim-{request.content_hash[:12]}",
                        "custom_id": request.content_hash,
                        "response": {
                            "status_code": 200,
                            "body": {
                                # Real responses report the model that served
                                # them, and collect prices against it. Stamping
                                # the stub's id here is what stops a rehearsal
                                # booking imaginary dollars into the ledger.
                                "model": client.model_id,
                                "choices": [{"message": {"content": payload, "role": "assistant"}}],
                                "usage": {
                                    "prompt_tokens": completion.input_tokens,
                                    "completion_tokens": completion.output_tokens,
                                },
                            },
                        },
                        "error": None,
                    },
                    ensure_ascii=False,
                )
            )
    out = Path(args.out or shard.with_suffix(".out.jsonl"))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"shard": str(shard), "responses": len(lines), "out": str(out)}, indent=2))
    return 0


def cmd_batch_submit(args: argparse.Namespace) -> int:
    require_api_key("batch-submit")
    store = EnrichmentStore(args.db)
    client = BatchClient(ModelConfig.from_env(args.model))
    pending = store.con.execute(
        "SELECT shard, input_path, request_count FROM batches WHERE status = 'written'"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    submitted = 0
    for shard, input_path, request_count in pending:
        try:
            provider_id = client.submit(input_path)
        except Exception as exc:
            log.error("submit failed for %s: %s", shard, exc)
            store.record_batch(
                batch_id=f"local:{shard}",
                shard=str(shard),
                status="written",
                error=str(exc)[:400],
                input_path=str(input_path),
            )
            continue
        store.record_batch(
            batch_id=provider_id,
            shard=str(shard),
            status="submitted",
            provider_id=provider_id,
            request_count=int(request_count or 0),
            model_id=client.model_id,
            input_path=str(input_path),
        )
        store.con.execute("DELETE FROM batches WHERE batch_id = ?", [f"local:{shard}"])
        submitted += 1
        log.info("submitted %s as %s", shard, provider_id)
    print(json.dumps({"submitted": submitted}, indent=2))
    store.close()
    return 0


def cmd_batch_status(args: argparse.Namespace) -> int:
    require_api_key("batch-status")
    store = EnrichmentStore(args.db)
    client = BatchClient(ModelConfig.from_env(args.model))
    rows = store.open_batches()
    out = []
    for row in rows:
        batch_id = str(row["batch_id"])
        if batch_id.startswith("local:"):
            out.append({"batch_id": batch_id, "status": row["status"]})
            continue
        try:
            status = client.status(batch_id)
        except Exception as exc:
            out.append({"batch_id": batch_id, "status": "error", "error": str(exc)[:200]})
            continue
        store.record_batch(
            batch_id=batch_id,
            shard=str(row["shard"]),
            status=status["status"],
            provider_id=batch_id,
            model_id=str(row.get("model_id") or ""),
            input_path=str(row.get("input_path") or ""),
        )
        out.append({"batch_id": batch_id, **{k: str(v) for k, v in status.items()}})
    print(json.dumps(out, indent=2))
    store.close()
    return 0


def cmd_batch_collect(args: argparse.Namespace) -> int:
    """Download finished shards, cache their answers, and apply them."""
    store = EnrichmentStore(args.db)
    digest = prompt_hash(truncate=args.truncate)
    model_id = args.model or ModelConfig.from_env().model_id
    config = ModelConfig.from_env(model_id)

    outputs: list[Path] = []
    if args.from_file:
        # Collecting from a downloaded file needs no account, which is what
        # makes the collect path testable offline.
        outputs = [Path(args.from_file)]
    else:
        require_api_key("batch-collect")
        client = BatchClient(config)
        for row in store.open_batches():
            batch_id = str(row["batch_id"])
            if batch_id.startswith("local:"):
                continue
            status = client.status(batch_id)
            if status["status"] != "completed" or not status["output_file_id"]:
                continue
            target = Path(args.shard_dir) / f"{row['shard']}.out.jsonl"
            client.download(str(status["output_file_id"]), target)
            store.record_batch(
                batch_id=batch_id,
                shard=str(row["shard"]),
                status="collected",
                provider_id=batch_id,
                output_path=str(target),
            )
            outputs.append(target)

    total_ok = total_failed = input_tokens = output_tokens = 0
    for path in outputs:
        completions = parse_batch_output(path)
        ok = [c for c in completions if c.ok]
        total_ok += len(ok)
        total_failed += len(completions) - len(ok)
        input_tokens += sum(c.input_tokens for c in completions)
        output_tokens += sum(c.output_tokens for c in completions)
        store.cache_put_many(
            [
                (
                    c.content_hash,
                    c.extraction,  # type: ignore[arg-type]
                    c.input_tokens,
                    c.output_tokens,
                    config.cost(c.input_tokens, c.output_tokens, batch=True),
                )
                for c in ok
            ],
            prompt_hash=digest,
            model_id=model_id,
        )
    if outputs:
        # Price against the model the file reports, not the one requested.
        # They differ whenever a rehearsal file from batch-simulate is
        # collected, and booking that at gpt-4o-mini rates would put spend in
        # the ledger that never happened.
        served_by = next((m for m in (reported_model(path) for path in outputs) if m), model_id)
        billed = config if served_by == model_id else ModelConfig.from_env(served_by)
        store.record_cost(
            stage="batch-collect",
            calls=total_ok + total_failed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=billed.cost(input_tokens, output_tokens, batch=True),
            model_id=served_by,
            note=f"{len(outputs)} shards",
        )
    print(
        json.dumps(
            {
                "shards": len(outputs),
                "cached": total_ok,
                "failed": total_failed,
                "cache_size": store.cache_size(),
            },
            indent=2,
        )
    )
    store.close()
    return cmd_apply(args) if args.apply else 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Re-merge stored rows against the current LLM cache.

    Separate from collection so a prompt or merge-rule change can be
    replayed over already-paid-for answers without touching the provider.
    """
    store = EnrichmentStore(args.db)
    digest = prompt_hash(truncate=args.truncate)
    model_id = args.model or ModelConfig.from_env().model_id
    applied = 0
    counter = CoverageCounter()

    for batch in iter_rows(
        args.snapshot,
        truncate=args.truncate,
        batch_size=args.batch_size,
        limit=args.limit,
    ):
        keys = [str(row["content_hash"]) for row in batch]
        cached = store.cache_get_many(keys, prompt_hash=digest, model_id=model_id)
        if not cached:
            continue
        built = []
        for row in batch:
            extraction = cached.get(str(row["content_hash"]))
            if extraction is None:
                continue
            tier0 = run_tier0(row)
            enriched = merge(
                row,
                tier0,
                extraction,
                tier="tier1",
                model_id=model_id,
                prompt_hash=digest,
            )
            counter.observe(enriched)
            built.append(enriched)
        if built:
            store.upsert(built)
            applied += len(built)
            log.info("applied %s rows", f"{applied:,}")

    print(json.dumps({"applied": applied, "coverage": counter.as_dict()["coverage"]}, indent=2))
    store.close()
    return 0


# --- join / stats -----------------------------------------------------------


def cmd_join(args: argparse.Namespace) -> int:
    """Write the joined output: snapshot columns plus enrichment.

    Joined in hash buckets rather than in one statement. Once the sidecar
    holds all 4.85M rows, a single join has to build a hash table over the
    whole enrichment side while streaming 4.85M wide description rows past
    it, and DuckDB dies with "failed to pin block" even at a 16 GB limit.

    Bucketing on ``hash(url) % parts`` divides the build side by ``parts``
    with no change in result: equal urls always land in the same bucket, so a
    row can only ever match within its own bucket, and unmatched left rows
    stay in theirs — the LEFT JOIN semantics are preserved exactly. The parts
    are then concatenated by a plain scan-and-write, which streams and needs
    no hash table, so the output is still one file.
    """
    store = EnrichmentStore(args.db)
    sidecar = Path(args.db).with_suffix(".sidecar.parquet")
    store.export_parquet(sidecar)
    store.close()

    from enrich.store import connect

    con = connect(database=":memory:")
    parts = max(1, int(args.parts))
    part_dir = Path(args.out).parent / f".{Path(args.out).stem}_parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    def _select(bucket: int | None) -> str:
        # Both sides get the same bucket predicate. Filtering only the probe
        # side would still build the hash table over the entire sidecar and
        # defeat the point.
        snapshot_where = "" if bucket is None else f"WHERE hash(j.url) % {parts} = {bucket}"
        sidecar_where = "" if bucket is None else f"WHERE hash(url) % {parts} = {bucket}"
        return f"""
            SELECT
                j.* EXCLUDE (country_iso, region, language, lat, lon, is_remote,
                             salary_min, salary_max, salary_currency, salary_period,
                             employment_type),
                coalesce(e.country_iso, j.country_iso)        AS country_iso,
                coalesce(e.region, j.region)                  AS region,
                coalesce(e.language, j.language)              AS language,
                e.lat                                         AS lat,
                e.lon                                         AS lon,
                e.geo_precision                               AS geo_precision,
                e.is_remote                                   AS is_remote,
                e.placement                                   AS placement,
                coalesce(e.employment_type, j.employment_type) AS employment_type,
                e.experience_min_years                        AS experience_min_years,
                e.experience_max_years                        AS experience_max_years,
                e.seniority                                   AS seniority,
                e.salary_min                                  AS salary_min,
                e.salary_max                                  AS salary_max,
                e.salary_currency                             AS salary_currency,
                e.salary_period                               AS salary_period,
                e.department                                  AS department_norm,
                e.skills                                      AS skills,
                e.education_level                             AS education_level,
                e.visa_sponsorship                            AS visa_sponsorship,
                e.global_id                                   AS global_id,
                e.sources                                     AS enrichment_sources,
                e.evidence                                    AS enrichment_evidence,
                e.tier                                        AS enrichment_tier,
                e.needs_review                                AS needs_review,
                e.enrichment_version                          AS enrichment_version,
                e.model_id                                    AS enrichment_model
            FROM read_parquet('{args.snapshot}') j
            LEFT JOIN (
                SELECT * FROM read_parquet('{sidecar}') {sidecar_where}
            ) e
              -- Joined on the raw URL rather than job_key: job_key is a
              -- SHA-256 of the *normalized* URL and cannot be recomputed in
              -- SQL, so the sidecar carries the original url for exactly
              -- this purpose.
              ON e.url = j.url
            {snapshot_where}
        """

    if parts == 1:
        con.execute(f"COPY ({_select(None)}) TO '{args.out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    else:
        for bucket in range(parts):
            target = part_dir / f"part-{bucket:03d}.parquet"
            con.execute(
                f"COPY ({_select(bucket)}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            log.info("joined bucket %d/%d", bucket + 1, parts)
        con.execute(
            f"""
            COPY (SELECT * FROM read_parquet('{part_dir}/part-*.parquet'))
            TO '{args.out}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        shutil.rmtree(part_dir)

    rows = con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()
    source = con.execute(f"SELECT count(*) FROM read_parquet('{args.snapshot}')").fetchone()
    written, expected = int(rows[0]), int(source[0])
    if written != expected:
        # A LEFT JOIN must not change the row count. If it did, the sidecar
        # has duplicate urls and rows have been fanned out.
        raise SystemExit(
            f"join produced {written:,} rows from a {expected:,}-row snapshot; "
            "the sidecar has duplicate urls"
        )
    print(json.dumps({"out": str(args.out), "rows": written, "parts": parts}, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = EnrichmentStore(args.db)
    stats = store.stats()
    stats["cache_entries"] = store.cache_size()
    stats["total_cost_usd"] = round(store.total_cost(), 4)
    stats["enrichment_version"] = ENRICHMENT_VERSION
    print(
        json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
            indent=2,
            default=str,
        )
    )
    store.close()
    return 0


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m enrich.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    common = {
        "snapshot": (("--snapshot",), {"type": Path, "default": DEFAULT_SNAPSHOT}),
        "db": (("--db",), {"type": Path, "default": DEFAULT_DB}),
        "truncate": (("--truncate",), {"type": int, "default": 2000}),
        "limit": (("--limit",), {"type": int, "default": None}),
        "batch_size": (("--batch-size",), {"type": int, "default": 20_000}),
        "model": (("--model",), {"default": None}),
        "client": (
            ("--client",),
            {"default": "stub", "choices": ("auto", "openai", "stub")},
        ),
        "shard_dir": (("--shard-dir",), {"type": Path, "default": BATCH_DIR}),
    }

    def add(sub: argparse.ArgumentParser, *names: str) -> None:
        for name in names:
            flags, kwargs = common[name]
            sub.add_argument(*flags, **kwargs)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("profile", help="quantify the enrichment gap")
    add(p, "snapshot", "truncate")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--skip-dedup", action="store_true")
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.set_defaults(func=lambda a: profile_module.main(_profile_argv(a)))

    p = subparsers.add_parser("tier0", help="deterministic pass")
    add(p, "snapshot", "db", "truncate", "limit", "batch_size")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_tier0)

    p = subparsers.add_parser("gold-build", help="build the provider-labelled holdout")
    add(p, "snapshot", "truncate")
    p.add_argument("--per-target", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_gold_build)

    p = subparsers.add_parser("eval", help="score against the gold sets")
    add(p, "truncate", "limit", "client")
    p.add_argument("--edge-only", action="store_true")
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.set_defaults(func=lambda a: _eval_main(a))

    p = subparsers.add_parser("dedup", help="report unique-body count")
    add(p, "snapshot", "truncate")
    p.add_argument("--sample", type=int, default=None)
    p.set_defaults(func=cmd_dedup)

    p = subparsers.add_parser("pilot", help="Tier 1 on a sample, with cost projection")
    add(p, "snapshot", "db", "truncate", "client", "model")
    p.add_argument("-n", type=int, default=5000, dest="n")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--projected-calls", type=int, default=4_229_953)
    p.set_defaults(func=cmd_pilot)

    p = subparsers.add_parser("batch-write", help="write Batch API shards")
    add(p, "snapshot", "db", "truncate", "limit", "batch_size", "model", "client", "shard_dir")
    p.add_argument("--shard-size", type=int, default=25_000)
    p.set_defaults(func=cmd_batch_write)

    p = subparsers.add_parser(
        "batch-simulate", help="fake a provider output file for a shard (offline)"
    )
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_batch_simulate)

    p = subparsers.add_parser("batch-submit", help="submit written shards")
    add(p, "db", "model", "limit")
    p.set_defaults(func=cmd_batch_submit)

    p = subparsers.add_parser("batch-status", help="poll submitted shards")
    add(p, "db", "model")
    p.set_defaults(func=cmd_batch_status)

    p = subparsers.add_parser("batch-collect", help="ingest finished shards")
    add(p, "snapshot", "db", "truncate", "limit", "batch_size", "model", "shard_dir")
    p.add_argument("--from-file", type=Path, default=None)
    p.add_argument("--apply", action="store_true")
    p.set_defaults(func=cmd_batch_collect)

    p = subparsers.add_parser("apply", help="re-merge rows from the cache")
    add(p, "snapshot", "db", "truncate", "limit", "batch_size", "model")
    p.set_defaults(func=cmd_apply)

    p = subparsers.add_parser("agent", help="Tier 2 recovery")
    add(p, "snapshot", "db", "truncate", "limit", "client", "model")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--ats", default=None, help="restrict to one ats_type")
    p.set_defaults(func=lambda a: _agent_main(a))

    p = subparsers.add_parser("join", help="write all_enriched.parquet")
    add(p, "snapshot", "db")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--parts",
        type=int,
        default=8,
        help="hash buckets to join in; raise if the join runs out of memory",
    )
    p.set_defaults(func=cmd_join)

    p = subparsers.add_parser("delta", help="enrich only what changed")
    add(p, "snapshot", "db", "truncate", "batch_size", "client", "model", "limit")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=200_000)
    p.set_defaults(func=lambda a: _delta_main(a))

    p = subparsers.add_parser("stats", help="coverage and spend")
    add(p, "db")
    p.set_defaults(func=cmd_stats)

    return parser


def _profile_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--snapshot", str(args.snapshot), "--truncate", str(args.truncate)]
    if args.sample:
        argv += ["--sample", str(args.sample)]
    if args.skip_dedup:
        argv.append("--skip-dedup")
    argv += ["--out-dir", str(args.out_dir)]
    return argv


def _eval_main(args: argparse.Namespace) -> int:
    from enrich import eval as eval_module

    argv = ["--client", args.client, "--truncate", str(args.truncate)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.edge_only:
        argv.append("--edge-only")
    argv += ["--out-dir", str(args.out_dir)]
    return eval_module.main(argv)


def _agent_main(args: argparse.Namespace) -> int:
    from enrich.agent import run_agent_recovery

    return run_agent_recovery(args)


def _delta_main(args: argparse.Namespace) -> int:
    from enrich.delta import run_delta

    return run_delta(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    ensure_dirs()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

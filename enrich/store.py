"""DuckDB-backed sidecar store.

Five tables, each with one job:

``enrichment``
    One row per ``job_key`` — the joinable output. Carries provenance
    (``enrichment_version``, ``model_id``, ``prompt_hash``) so invalidation
    is selective: bumping a prompt re-runs only rows whose prompt changed,
    not the corpus.

``llm_cache``
    Keyed by ``(content_hash, prompt_hash, model_id)`` — *not* by job. This
    is what makes duplicate postings free. A multi-location Workday
    requisition spread across 40 rows shares one description and therefore
    one paid call. It also makes the pipeline safely resumable: a crashed
    backfill re-reads its own answers instead of re-buying them.

``batches``
    Batch API checkpoints. A 4.85M-row backfill is thousands of shards
    across many hours; losing track of which shard was submitted is
    equivalent to paying twice.

``cost_ledger``
    Append-only token and cost record per call group. Cost is the main
    engineering constraint here, so it is measured rather than assumed.

``agent_runs``
    Tier 2 audit: what the agent did, how many steps and tokens it spent,
    and whether the escalation actually recovered anything. Without this
    there is no way to prove Tier 2 pays for itself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from enrich._version import ENRICHMENT_VERSION
from enrich.paths import DATA_DIR, DEFAULT_DB, ensure_dirs
from enrich.schema import JobEnrichment, LlmExtraction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS enrichment (
    job_key             VARCHAR PRIMARY KEY,
    content_hash        VARCHAR NOT NULL,
    fallback_key        VARCHAR,
    url                 VARCHAR NOT NULL,
    ats_type            VARCHAR,
    ats_id              VARCHAR,
    global_id           VARCHAR,
    language            VARCHAR,
    country_iso         VARCHAR,
    region              VARCHAR,
    lat                 DOUBLE,
    lon                 DOUBLE,
    geo_precision       VARCHAR,
    placement           VARCHAR,
    is_remote           BOOLEAN,
    employment_type     VARCHAR,
    experience_min_years INTEGER,
    experience_max_years INTEGER,
    seniority           VARCHAR,
    salary_min          DOUBLE,
    salary_max          DOUBLE,
    salary_currency     VARCHAR,
    salary_period       VARCHAR,
    department          VARCHAR,
    skills              VARCHAR[],
    education_level     VARCHAR,
    visa_sponsorship    VARCHAR,
    sources             VARCHAR,
    evidence            VARCHAR,
    tier                VARCHAR NOT NULL,
    needs_review        BOOLEAN NOT NULL,
    review_reason       VARCHAR,
    enrichment_version  INTEGER NOT NULL,
    model_id            VARCHAR,
    prompt_hash         VARCHAR,
    enriched_at         TIMESTAMP
);

CREATE INDEX IF NOT EXISTS enrichment_content ON enrichment (content_hash);
CREATE INDEX IF NOT EXISTS enrichment_review ON enrichment (needs_review);

CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash  VARCHAR NOT NULL,
    prompt_hash   VARCHAR NOT NULL,
    model_id      VARCHAR NOT NULL,
    payload       VARCHAR NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      DOUBLE,
    created_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (content_hash, prompt_hash, model_id)
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id     VARCHAR PRIMARY KEY,
    provider_id  VARCHAR,
    shard        VARCHAR NOT NULL,
    status       VARCHAR NOT NULL,
    request_count INTEGER,
    model_id     VARCHAR,
    prompt_hash  VARCHAR,
    input_path   VARCHAR,
    output_path  VARCHAR,
    error        VARCHAR,
    submitted_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cost_ledger (
    id            BIGINT,
    stage         VARCHAR NOT NULL,
    model_id      VARCHAR,
    calls         INTEGER NOT NULL,
    input_tokens  BIGINT NOT NULL,
    output_tokens BIGINT NOT NULL,
    cost_usd      DOUBLE NOT NULL,
    note          VARCHAR,
    recorded_at   TIMESTAMP NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS cost_ledger_id START 1;

CREATE TABLE IF NOT EXISTS agent_runs (
    job_key      VARCHAR NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    steps        INTEGER NOT NULL,
    tools_used   VARCHAR,
    recovered_chars INTEGER,
    resolved     BOOLEAN NOT NULL,
    cost_usd     DOUBLE,
    outcome      VARCHAR,
    error        VARCHAR
);
"""

#: Column order of ``enrichment``, used to build insert frames.
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "job_key",
    "content_hash",
    "fallback_key",
    "url",
    "ats_type",
    "ats_id",
    "global_id",
    "language",
    "country_iso",
    "region",
    "lat",
    "lon",
    "geo_precision",
    "placement",
    "is_remote",
    "employment_type",
    "experience_min_years",
    "experience_max_years",
    "seniority",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "department",
    "skills",
    "education_level",
    "visa_sponsorship",
    "sources",
    "evidence",
    "tier",
    "needs_review",
    "review_reason",
    "enrichment_version",
    "model_id",
    "prompt_hash",
    "enriched_at",
)


def _memory_limit() -> str:
    """Memory ceiling for a DuckDB connection.

    A fixed ceiling cannot work for both workloads this layer runs. The Tier 0
    pass streams and wants to leave RAM for 8 worker processes; the final join
    materializes a hash table over 4.85M rows and simply needs room. A limit
    tuned for the first turns the second into an "Out of Memory Error: failed
    to pin block" crash on a machine with 32 GB free.

    So scale with the machine and let the operator override. Half of RAM leaves
    room for the OS and for whatever else is running, and the 16 GB cap keeps a
    very large machine from letting a runaway query consume everything.
    """
    configured = os.environ.get("ATS_ENRICH_MEMORY_LIMIT")
    if configured:
        return configured
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    gigabytes = max(2, min(16, int(total / (1024**3) * 0.5)))
    return f"{gigabytes}GB"


def connect(
    *,
    database: str | Path = DEFAULT_DB,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the settings this layer needs."""
    if database != ":memory:":
        ensure_dirs()
    con = duckdb.connect(str(database), read_only=read_only)
    con.execute(f"SET memory_limit='{_memory_limit()}'")
    con.execute("SET preserve_insertion_order=false")
    # An in-memory database has *no* spill directory by default, so the
    # memory limit above becomes a hard failure rather than a throttle: the
    # 4.85M-row join in ``enrich.run join`` dies with "failed to pin block"
    # instead of spilling. File-backed databases default to "<db>.tmp", but
    # this layer's biggest query deliberately runs against ``:memory:``
    # because it only reads parquet, so the directory has to be explicit.
    if not read_only:
        ensure_dirs()
        con.execute(f"SET temp_directory='{DATA_DIR / 'duckdb-spill'}'")
    # The progress bar writes ANSI control sequences to stdout, which
    # corrupts piped report output and log files.
    con.execute("SET enable_progress_bar=false")
    return con


class EnrichmentStore:
    """Read/write access to the sidecar."""

    def __init__(self, database: str | Path = DEFAULT_DB) -> None:
        self.database = database
        self.con = connect(database=database)
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                self.con.execute(statement)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> EnrichmentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- enrichment rows ---------------------------------------------------

    def upsert(self, rows: Sequence[JobEnrichment]) -> int:
        """Replace-by-key insert of enrichment rows.

        Implemented as delete-then-insert inside one transaction rather
        than a 36-column ``ON CONFLICT DO UPDATE``: it is far less code to
        get wrong when a column is added, and DuckDB executes it as two
        vectorized operations over a registered frame.
        """
        if not rows:
            return 0
        records = [self._to_record(row) for row in rows]
        frame = _to_arrow(records, ENRICHMENT_COLUMNS)
        self.con.register("_incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                "DELETE FROM enrichment WHERE job_key IN (SELECT job_key FROM _incoming)"
            )
            columns = ", ".join(f'"{c}"' for c in ENRICHMENT_COLUMNS)
            # A single batch can legitimately contain two rows with the same
            # job_key: roughly 0.1% of the published snapshot repeats a url,
            # and normalization collapses urls that differ only by tracking
            # parameters. Deduplicate here rather than trusting the source —
            # otherwise one repeated posting aborts a 20,000-row insert.
            # Rows not flagged for review win, then the most recent, so the
            # choice is deterministic rather than dependent on scan order.
            self.con.execute(
                f"""
                INSERT INTO enrichment ({columns})
                SELECT {columns} FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY job_key
                        ORDER BY needs_review ASC, enriched_at DESC
                    ) AS _rank
                    FROM _incoming
                ) WHERE _rank = 1
                """
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("_incoming")
        return len(records)

    @staticmethod
    def _to_record(row: JobEnrichment) -> dict[str, Any]:
        data = row.model_dump()
        data["sources"] = json.dumps(data.get("sources") or {}, ensure_ascii=False)
        data["evidence"] = json.dumps(data.get("evidence") or {}, ensure_ascii=False)
        data["skills"] = list(data.get("skills") or [])
        return data

    def enriched_keys(self, *, version: int = ENRICHMENT_VERSION) -> set[str]:
        rows = self.con.execute(
            "SELECT job_key FROM enrichment WHERE enrichment_version >= ?", [version]
        ).fetchall()
        return {str(row[0]) for row in rows}

    def count(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM enrichment").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        row = self.con.execute(
            """
            SELECT count(*) AS rows,
                   sum(CASE WHEN tier = 'tier0' THEN 1 ELSE 0 END) AS tier0,
                   sum(CASE WHEN tier = 'tier1' THEN 1 ELSE 0 END) AS tier1,
                   sum(CASE WHEN tier = 'tier2' THEN 1 ELSE 0 END) AS tier2,
                   sum(CASE WHEN needs_review THEN 1 ELSE 0 END) AS needs_review,
                   avg(CASE WHEN country_iso IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cov_country,
                   avg(CASE WHEN language IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cov_language,
                   avg(CASE WHEN placement IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cov_placement,
                   avg(CASE WHEN salary_min IS NOT NULL THEN 1.0 ELSE 0.0 END) AS cov_salary,
                   avg(CASE WHEN experience_min_years IS NOT NULL THEN 1.0 ELSE 0.0 END)
                       AS cov_experience
            FROM enrichment
            """
        ).df()
        return row.to_dict(orient="records")[0]

    def export_parquet(self, path: str | Path) -> int:
        """Write the sidecar to parquet, with JSON columns kept as JSON text."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY enrichment TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        return self.count()

    # --- LLM cache ---------------------------------------------------------

    def cache_get_many(
        self, content_hashes: Iterable[str], *, prompt_hash: str, model_id: str
    ) -> dict[str, LlmExtraction]:
        """Look up many cache entries at once.

        Batched deliberately: the backfill checks the cache for every row
        before dispatching, and a per-row round trip through Python would
        dominate the runtime of an otherwise IO-bound stage.
        """
        keys = list(dict.fromkeys(content_hashes))
        if not keys:
            return {}
        found: dict[str, LlmExtraction] = {}
        chunk = 5000
        for start in range(0, len(keys), chunk):
            window = keys[start : start + chunk]
            placeholders = ", ".join("?" for _ in window)
            rows = self.con.execute(
                f"""
                SELECT content_hash, payload FROM llm_cache
                WHERE prompt_hash = ? AND model_id = ?
                  AND content_hash IN ({placeholders})
                """,
                [prompt_hash, model_id, *window],
            ).fetchall()
            for content_hash, payload in rows:
                try:
                    found[str(content_hash)] = LlmExtraction.model_validate_json(str(payload))
                except Exception:
                    # A cache entry written by an older, incompatible
                    # contract is a miss, not a crash. It will be
                    # overwritten by the next successful call.
                    continue
        return found

    def cache_put_many(
        self,
        entries: Sequence[tuple[str, LlmExtraction, int, int, float]],
        *,
        prompt_hash: str,
        model_id: str,
    ) -> int:
        """Insert cache entries. ``entries`` is
        ``(content_hash, extraction, input_tokens, output_tokens, cost_usd)``."""
        if not entries:
            return 0
        now = datetime.now(tz=UTC)
        rows = [
            (
                content_hash,
                prompt_hash,
                model_id,
                extraction.model_dump_json(),
                int(input_tokens),
                int(output_tokens),
                float(cost),
                now,
            )
            for content_hash, extraction, input_tokens, output_tokens, cost in entries
        ]
        self.con.executemany(
            """
            INSERT OR REPLACE INTO llm_cache
                (content_hash, prompt_hash, model_id, payload,
                 input_tokens, output_tokens, cost_usd, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def cache_size(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM llm_cache").fetchone()[0])

    # --- batches -----------------------------------------------------------

    def record_batch(
        self,
        *,
        batch_id: str,
        shard: str,
        status: str,
        provider_id: str | None = None,
        request_count: int | None = None,
        model_id: str | None = None,
        prompt_hash: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO batches
                (batch_id, provider_id, shard, status, request_count, model_id,
                 prompt_hash, input_path, output_path, error, submitted_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT submitted_at FROM batches WHERE batch_id = ?), ?),
                    CASE WHEN ? IN ('completed', 'failed', 'cancelled', 'expired')
                         THEN ? ELSE NULL END)
            """,
            [
                batch_id,
                provider_id,
                shard,
                status,
                request_count,
                model_id,
                prompt_hash,
                input_path,
                output_path,
                error,
                batch_id,
                datetime.now(tz=UTC),
                status,
                datetime.now(tz=UTC),
            ],
        )

    def open_batches(self) -> list[dict[str, Any]]:
        frame = self.con.execute(
            """
            SELECT * FROM batches
            WHERE status NOT IN ('completed', 'failed', 'cancelled', 'expired', 'collected')
            ORDER BY submitted_at
            """
        ).df()
        return list(frame.to_dict(orient="records"))

    def batch_shards_done(self) -> set[str]:
        rows = self.con.execute(
            "SELECT shard FROM batches WHERE status IN ('collected', 'completed')"
        ).fetchall()
        return {str(row[0]) for row in rows}

    # --- ledger ------------------------------------------------------------

    def record_cost(
        self,
        *,
        stage: str,
        calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        model_id: str | None = None,
        note: str | None = None,
    ) -> None:
        self.con.execute(
            """
            INSERT INTO cost_ledger
                (id, stage, model_id, calls, input_tokens, output_tokens,
                 cost_usd, note, recorded_at)
            VALUES (nextval('cost_ledger_id'), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                stage,
                model_id,
                int(calls),
                int(input_tokens),
                int(output_tokens),
                float(cost_usd),
                note,
                datetime.now(tz=UTC),
            ],
        )

    def total_cost(self) -> float:
        value = self.con.execute("SELECT coalesce(sum(cost_usd), 0.0) FROM cost_ledger").fetchone()
        return float(value[0]) if value else 0.0

    def record_agent_run(
        self,
        *,
        job_key: str,
        steps: int,
        tools_used: Sequence[str],
        recovered_chars: int,
        resolved: bool,
        cost_usd: float,
        outcome: str,
        error: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        self.con.execute(
            """
            INSERT INTO agent_runs
                (job_key, started_at, steps, tools_used, recovered_chars,
                 resolved, cost_usd, outcome, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                job_key,
                started_at or datetime.now(tz=UTC),
                int(steps),
                ",".join(tools_used),
                int(recovered_chars),
                bool(resolved),
                float(cost_usd),
                outcome,
                error,
            ],
        )


def _to_arrow(records: Sequence[dict[str, Any]], columns: Sequence[str]) -> Any:
    """Build an Arrow table with an explicit schema.

    Explicit typing matters: inferring from Python values makes an
    all-``None`` column come out as ``null`` type, which then fails to
    insert into a typed DuckDB column. That happens routinely — a Tier 0
    only batch has no ``salary_min`` at all.
    """
    import pyarrow as pa

    type_map: dict[str, pa.DataType] = {
        "lat": pa.float64(),
        "lon": pa.float64(),
        "is_remote": pa.bool_(),
        "experience_min_years": pa.int32(),
        "experience_max_years": pa.int32(),
        "salary_min": pa.float64(),
        "salary_max": pa.float64(),
        "skills": pa.list_(pa.string()),
        "needs_review": pa.bool_(),
        "enrichment_version": pa.int32(),
        "enriched_at": pa.timestamp("us", tz="UTC"),
    }
    fields = [pa.field(name, type_map.get(name, pa.string())) for name in columns]
    schema = pa.schema(fields)
    data = {name: [record.get(name) for record in records] for name in columns}
    return pa.table(data, schema=schema)

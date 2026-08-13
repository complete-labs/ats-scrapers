"""Reading the published snapshot in bounded memory.

The snapshot is 4.85M rows / 2.25 GB of parquet. Nothing in this layer ever
loads it whole. Every consumer here takes batches of plain dicts, which is
also what makes the same code path work for a 200-row test fixture, a 5000
row pilot and the full backfill.

One quirk drives the normalization below: ``_job_to_row`` in
``scripts/run_pipeline.py`` writes ``""`` for every absent optional field,
so the published parquet has almost no NULLs in its string columns and
``lat``/``lon`` are VARCHAR rather than DOUBLE. Treating ``""`` as a value
would make coverage look complete while every derived field silently
disagreed, so blanks are converted to ``None`` on the way in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from enrich.keys import content_hash, fallback_key, job_key
from enrich.store import connect

#: Columns the enrichment layer reads. Projecting explicitly keeps DuckDB
#: from materializing ``raw`` (a JSON blob up to 5 kB per row) for every
#: pass, which roughly halves the bytes read off disk.
SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "url",
    "title",
    "company",
    "ats_type",
    "ats_id",
    "location",
    "country_iso",
    "region",
    "language",
    "lat",
    "lon",
    "is_remote",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_summary",
    "employment_type",
    "department",
    "team",
    "description",
    "commitment",
    "posted_at",
    "requisition_id",
)


#: How much of ``description`` any pass ever reads off disk.
#:
#: Every consumer uses the *same* projection, and that is load-bearing:
#: :func:`enrich.keys.content_hash` collapses whitespace before truncating,
#: so hashing a 2000-character window of a differently-truncated body would
#: produce a different key for the same posting and silently miss the cache.
#: 8000 covers the p90 description (5,960 chars) and the 8000-character
#: window :func:`enrich.deterministic.parse_experience_years` scans.
DESCRIPTION_READ_CHARS = 8000


def projection(columns: Sequence[str] = SNAPSHOT_COLUMNS) -> str:
    """SQL projection list, truncating ``description`` at the read limit."""
    parts: list[str] = []
    for column in columns:
        if column == "description":
            parts.append(f'substr("description", 1, {DESCRIPTION_READ_CHARS}) AS "description"')
        else:
            parts.append(f'"{column}"')
    return ", ".join(parts)


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw snapshot row into the layer's internal shape."""
    clean = {key: _blank_to_none(value) for key, value in row.items()}
    return clean


def add_keys(row: dict[str, Any], *, truncate: int) -> dict[str, Any]:
    """Attach ``job_key`` / ``content_hash`` / ``fallback_key`` to a row."""
    row["job_key"] = job_key(str(row.get("url") or ""))
    row["fallback_key"] = fallback_key(row.get("company"), row.get("title"), row.get("location"))
    row["content_hash"] = content_hash(
        title=row.get("title"),
        description=row.get("description"),
        location=row.get("location"),
        salary_summary=row.get("salary_summary"),
        commitment=row.get("commitment"),
        employment_type=row.get("employment_type"),
        truncate=truncate,
    )
    return row


def iter_rows(
    snapshot: str | Path,
    *,
    truncate: int,
    batch_size: int = 20_000,
    limit: int | None = None,
    where: str | None = None,
    order_by: str | None = None,
    columns: Sequence[str] = SNAPSHOT_COLUMNS,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of normalized, keyed snapshot rows.

    Uses a DuckDB cursor with ``fetchmany`` rather than ``LIMIT``/``OFFSET``
    paging: OFFSET on a 4.85M-row parquet re-scans from the top for every
    page, which turns a linear pass into a quadratic one.
    """
    con = connect(database=":memory:")
    query = f"SELECT {projection(columns)} FROM read_parquet('{snapshot}')"
    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit:
        # A bare LIMIT over a parallel parquet scan returns whichever rows the
        # threads finish first, so it is *not* a stable prefix: two identical
        # ``--limit 4000`` reads were measured sharing only 1,952 rows. That
        # silently breaks everything built on repeat runs — a pilot cannot be
        # compared against the previous pilot, and ``delta`` reports hundreds
        # of "new" rows on an unchanged snapshot because it is looking at a
        # different sample. Restoring insertion order makes the limit mean
        # "the first N rows of the file"; it costs nothing here because the
        # scan stops after N rows anyway.
        con.execute("SET preserve_insertion_order=true")
        query += f" LIMIT {int(limit)}"

    cursor = con.execute(query)
    names = [description[0] for description in cursor.description or []]
    try:
        while True:
            chunk = cursor.fetchmany(batch_size)
            if not chunk:
                break
            yield [
                add_keys(normalize_row(dict(zip(names, record, strict=True))), truncate=truncate)
                for record in chunk
            ]
    finally:
        con.close()


def sample_rows(
    snapshot: str | Path,
    *,
    n: int,
    truncate: int,
    seed: int = 42,
    where: str | None = None,
    columns: Sequence[str] = SNAPSHOT_COLUMNS,
) -> list[dict[str, Any]]:
    """Reservoir-sample ``n`` rows. Used for pilots, evals and gold sets.

    A seeded reservoir sample rather than ``LIMIT``: the snapshot is
    physically grouped by ``ats_type``, so the first N rows are all one
    provider and any measurement taken from them is meaningless.
    """
    con = connect(database=":memory:")
    try:
        query = f"SELECT {projection(columns)} FROM read_parquet('{snapshot}')"
        if where:
            query += f" WHERE {where}"
        query += f" USING SAMPLE {int(n)} ROWS (reservoir, {int(seed)})"
        cursor = con.execute(query)
        names = [description[0] for description in cursor.description or []]
        return [
            add_keys(normalize_row(dict(zip(names, record, strict=True))), truncate=truncate)
            for record in cursor.fetchall()
        ]
    finally:
        con.close()


def count_rows(snapshot: str | Path, *, where: str | None = None) -> int:
    con = connect(database=":memory:")
    try:
        query = f"SELECT count(*) FROM read_parquet('{snapshot}')"
        if where:
            query += f" WHERE {where}"
        return int(con.execute(query).fetchone()[0])
    finally:
        con.close()

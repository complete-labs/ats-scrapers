"""Phase 0: quantify the enrichment gap before spending anything on it.

Answers three questions that decide the whole project:

1. **What is actually missing, per provider?** Null rates per field per
   ``ats_type``. Turns "the schema says the LLM fills these" into a ranked
   list weighted by row count.
2. **How much text is there to read?** A row with an empty or 200-character
   description cannot be enriched by any model; it needs Tier 2 or nothing.
   This sizes Tier 2 before Tier 2 exists.
3. **How much does deduplication save?** Distinct ``content_hash`` count
   versus row count is the single largest cost lever, because multi-location
   postings repeat one description across many rows and the LLM cache is
   keyed on content.

Reads the parquet through DuckDB, which projects only the columns each
query touches, so this runs in bounded memory over the 2.25 GB snapshot
rather than loading it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enrich.paths import DEFAULT_SNAPSHOT, REPORT_DIR
from enrich.store import connect

# Fields the schema doc explicitly defers to enrichment, plus the ones a
# consumer would filter on. ``experience`` is absent from the published
# parquet entirely, so it is reported as a schema gap rather than a null
# rate.
_GAP_FIELDS = (
    "country_iso",
    "region",
    "language",
    "lat",
    "lon",
    "is_remote",
    "salary_min",
    "salary_currency",
    "employment_type",
    "department",
)

# ``lat``/``lon`` arrive as VARCHAR carrying empty strings rather than
# NULL, because ``_job_to_row`` in scripts/run_pipeline.py writes "" for a
# missing float. A plain IS NULL test reports 0% missing and is wrong.
_BLANK = "(({col} IS NULL) OR (CAST({col} AS VARCHAR) = ''))"


def _blank(column: str) -> str:
    return _BLANK.format(col=f'"{column}"')


def snapshot_columns(con: Any, snapshot: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{snapshot}')").fetchall()
    return [str(row[0]) for row in rows]


def profile(snapshot: Path, *, sample: int | None = None) -> dict[str, Any]:
    """Build the Phase 0 report as a plain dict."""
    con = connect(read_only=False, database=":memory:")
    columns = snapshot_columns(con, snapshot)
    source = f"read_parquet('{snapshot}')"
    if sample:
        source = f"(SELECT * FROM {source} USING SAMPLE {int(sample)} ROWS)"
    con.execute(f"CREATE OR REPLACE VIEW jobs AS SELECT * FROM {source}")

    total = int(con.execute("SELECT count(*) FROM jobs").fetchone()[0])

    present = [field for field in _GAP_FIELDS if field in columns]
    missing_from_schema = [field for field in _GAP_FIELDS if field not in columns]
    # ``experience`` is in the Job model and in docs/JOB_SCHEMA.md but is
    # never written by _job_to_row; flag it explicitly.
    for field in ("global_id", "experience", "fetched_at"):
        if field not in columns:
            missing_from_schema.append(field)

    gap_exprs = ",\n           ".join(
        f"avg(CASE WHEN {_blank(field)} THEN 1.0 ELSE 0.0 END) AS null_{field}" for field in present
    )
    overall = con.execute(
        f"""
        SELECT count(*) AS rows,
           {gap_exprs},
           avg(CASE WHEN description IS NULL OR length(description) < 200
                    THEN 1.0 ELSE 0.0 END) AS thin_description,
           avg(CASE WHEN description IS NULL OR length(description) = 0
                    THEN 1.0 ELSE 0.0 END) AS empty_description,
           median(length(coalesce(description, ''))) AS median_desc_len,
           quantile_cont(length(coalesce(description, '')), 0.9) AS p90_desc_len
        FROM jobs
        """
    ).df()

    by_ats = con.execute(
        f"""
        SELECT ats_type, count(*) AS rows,
           {gap_exprs},
           avg(CASE WHEN description IS NULL OR length(description) < 200
                    THEN 1.0 ELSE 0.0 END) AS thin_description
        FROM jobs GROUP BY ats_type ORDER BY rows DESC
        """
    ).df()

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "snapshot": str(snapshot),
        "sampled_rows": sample,
        "total_rows": total,
        "columns": columns,
        "absent_columns": sorted(set(missing_from_schema)),
        "overall": overall.to_dict(orient="records")[0],
        "by_ats": by_ats.to_dict(orient="records"),
    }


def dedup_savings(snapshot: Path, *, truncate: int, sample: int | None = None) -> dict[str, Any]:
    """Measure how many LLM calls content-hash dedup removes.

    Hashes in SQL rather than in Python: pulling 4.85M descriptions into
    the interpreter to hash them costs minutes and gigabytes, while DuckDB
    does it in a streaming aggregate. The expression mirrors
    :func:`enrich.keys.content_hash` closely enough for an estimate — it
    normalizes whitespace and truncates to the same limit — but it is a
    *measurement*, not the key itself, so exact parity is not required.
    """
    con = connect(read_only=False, database=":memory:")
    source = f"read_parquet('{snapshot}')"
    if sample:
        source = f"(SELECT * FROM {source} USING SAMPLE {int(sample)} ROWS)"
    con.execute(f"CREATE OR REPLACE VIEW jobs AS SELECT * FROM {source}")

    body = (
        "substr(regexp_replace(trim(coalesce(description, '')), '\\s+', ' ', 'g'), 1, "
        f"{int(truncate)})"
    )
    key = (
        "md5(concat_ws('\\x00', regexp_replace(trim(coalesce(title, '')), '\\s+', ' ', 'g'), "
        f"{body}, regexp_replace(trim(coalesce(location, '')), '\\s+', ' ', 'g')))"
    )
    row = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT {key}) AS distinct_content,
               count(DISTINCT CASE WHEN description IS NOT NULL
                                    AND length(description) >= 200
                              THEN {key} END) AS distinct_enrichable,
               sum(CASE WHEN description IS NOT NULL AND length(description) >= 200
                        THEN 1 ELSE 0 END) AS enrichable_rows
        FROM jobs
        """
    ).fetchone()
    rows, distinct_content, distinct_enrichable, enrichable_rows = (int(v or 0) for v in row)
    return {
        "truncate": truncate,
        "rows": rows,
        "distinct_content": distinct_content,
        "dedup_ratio": (1 - distinct_content / rows) if rows else 0.0,
        "enrichable_rows": enrichable_rows,
        "distinct_enrichable": distinct_enrichable,
        "enrichable_dedup_ratio": (
            (1 - distinct_enrichable / enrichable_rows) if enrichable_rows else 0.0
        ),
    }


def render_markdown(report: dict[str, Any], dedup: dict[str, Any] | None) -> str:
    """Human-readable report. Written next to the JSON so the numbers are
    reviewable without a notebook."""
    lines: list[str] = []
    total = report["total_rows"]
    lines.append("# Enrichment gap profile")
    lines.append("")
    lines.append(f"- Snapshot: `{report['snapshot']}`")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Rows: {total:,}")
    if report["sampled_rows"]:
        lines.append(f"- Sampled: {report['sampled_rows']:,} rows")
    lines.append(f"- Columns present: {len(report['columns'])}")
    lines.append(
        "- Documented but absent from the published parquet: "
        + ", ".join(f"`{c}`" for c in report["absent_columns"])
    )
    lines.append("")

    overall = report["overall"]
    lines.append("## Overall missing rate")
    lines.append("")
    for key, value in overall.items():
        if key.startswith("null_"):
            lines.append(f"- `{key[5:]}`: {float(value) * 100:.1f}% missing")
    lines.append(
        f"- thin description (<200 chars): {float(overall['thin_description']) * 100:.1f}%"
    )
    lines.append(f"- empty description: {float(overall['empty_description']) * 100:.1f}%")
    lines.append(f"- median description length: {int(overall['median_desc_len']):,} chars")
    lines.append(f"- p90 description length: {int(overall['p90_desc_len']):,} chars")
    lines.append("")

    if dedup:
        lines.append("## Deduplication headroom")
        lines.append("")
        lines.append(f"- Truncation used for hashing: {dedup['truncate']:,} chars")
        lines.append(f"- Distinct content bodies: {dedup['distinct_content']:,}")
        lines.append(f"- Dedup ratio (all rows): {dedup['dedup_ratio'] * 100:.1f}%")
        lines.append(f"- Enrichable rows (description >= 200 chars): {dedup['enrichable_rows']:,}")
        lines.append(f"- Distinct enrichable bodies: {dedup['distinct_enrichable']:,}")
        lines.append(
            f"- Dedup ratio among enrichable rows: {dedup['enrichable_dedup_ratio'] * 100:.1f}%"
        )
        lines.append("")
        lines.append(
            "The last number is the one that sets Tier 1 cost: it is how many "
            "paid calls the backfill actually needs."
        )
        lines.append("")

    lines.append("## Top providers by row count")
    lines.append("")
    header_fields = (
        [k[5:] for k in report["by_ats"][0] if k.startswith("null_")] if report["by_ats"] else []
    )
    lines.append("| ats_type | rows | " + " | ".join(header_fields) + " | thin desc |")
    lines.append("|---|--:|" + "--:|" * (len(header_fields) + 1))
    for entry in report["by_ats"][:25]:
        cells = [f"{float(entry[f'null_{f}']) * 100:.0f}%" for f in header_fields]
        lines.append(
            f"| `{entry['ats_type']}` | {int(entry['rows']):,} | "
            + " | ".join(cells)
            + f" | {float(entry['thin_description']) * 100:.0f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Profile a random sample of N rows instead of the full snapshot.",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=2000,
        help="Description truncation limit to measure dedup against.",
    )
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)

    if not args.snapshot.exists():
        print(f"snapshot not found: {args.snapshot}", file=sys.stderr)
        return 1

    report = profile(args.snapshot, sample=args.sample)
    dedup = (
        None
        if args.skip_dedup
        else dedup_savings(args.snapshot, truncate=args.truncate, sample=args.sample)
    )
    if dedup:
        report["dedup"] = dedup

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gap_profile.json"
    md_path = args.out_dir / "gap_profile.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(render_markdown(report, dedup))
    print(render_markdown(report, dedup))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

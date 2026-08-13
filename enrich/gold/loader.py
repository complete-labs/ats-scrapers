"""Loading and building gold examples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from enrich.deterministic import to_bool
from enrich.keys import content_hash
from enrich.paths import DATA_DIR

EDGE_CASES_PATH = Path(__file__).resolve().parent / "edge_cases.jsonl"
HOLDOUT_PATH = DATA_DIR / "gold" / "holdout.jsonl"

#: Prompt inputs a masked example must not show the model. Keyed by the
#: label being tested — masking is what keeps the eval honest.
MASK_FOR_LABEL: dict[str, tuple[str, ...]] = {
    "salary_min": ("salary_summary",),
    "salary_max": ("salary_summary",),
    "salary_currency": ("salary_summary",),
    "salary_period": ("salary_summary",),
    "employment_type": ("commitment", "employment_type"),
    "placement": (),
    "language": (),
    "country_iso": (),
}

#: Tokens that make a placement label trivially guessable from the title or
#: location alone. Used to split placement scores, not to filter examples.
_TRIVIAL_PLACEMENT_TOKENS = ("remote", "hybrid", "anywhere", "work from home", "wfh", "on-site")


@dataclass
class GoldExample:
    """One labelled posting.

    ``row`` is a snapshot-shaped dict so the same code paths that run in
    production run here — Tier 0, prompt rendering and merging all take this
    exact shape.
    """

    id: str
    row: dict[str, Any]
    labels: dict[str, Any]
    mask: tuple[str, ...] = ()
    note: str | None = None
    source: str = "edge"
    ats_type: str | None = None

    @property
    def masked_row(self) -> dict[str, Any]:
        """The row as the model should see it, with leaking fields removed."""
        if not self.mask:
            return self.row
        return {key: (None if key in self.mask else value) for key, value in self.row.items()}

    @property
    def placement_trivial(self) -> bool:
        haystack = " ".join(
            str(self.row.get(key) or "").casefold() for key in ("title", "location")
        )
        return any(token in haystack for token in _TRIVIAL_PLACEMENT_TOKENS)


def _row_from_record(record: dict[str, Any], *, truncate: int) -> dict[str, Any]:
    row = {
        "url": record.get("url") or f"https://gold.invalid/{record['id']}",
        "title": record.get("title"),
        "company": record.get("company"),
        "ats_type": record.get("ats_type"),
        "ats_id": record.get("ats_id") or record["id"],
        "location": record.get("location"),
        "country_iso": record.get("country_iso"),
        "region": record.get("region"),
        "language": record.get("language"),
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "is_remote": record.get("is_remote"),
        "salary_min": record.get("salary_min_provider"),
        "salary_max": record.get("salary_max_provider"),
        "salary_currency": record.get("salary_currency_provider"),
        "salary_period": record.get("salary_period_provider"),
        "salary_summary": record.get("salary_summary"),
        "employment_type": record.get("employment_type_provider"),
        "department": record.get("department"),
        "team": record.get("team"),
        "description": record.get("description") or "",
        "commitment": record.get("commitment"),
    }
    row["job_key"] = f"gold:{record['id']}"
    row["fallback_key"] = None
    row["content_hash"] = content_hash(
        title=row["title"],
        description=row["description"],
        location=row["location"],
        salary_summary=row["salary_summary"],
        commitment=row["commitment"],
        employment_type=row["employment_type"],
        truncate=truncate,
    )
    return row


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_edge_cases(*, truncate: int = 2000, path: Path = EDGE_CASES_PATH) -> list[GoldExample]:
    examples: list[GoldExample] = []
    for record in _load_jsonl(path):
        examples.append(
            GoldExample(
                id=str(record["id"]),
                row=_row_from_record(record, truncate=truncate),
                labels=dict(record.get("labels") or {}),
                mask=tuple(record.get("mask") or ()),
                note=record.get("note"),
                source="edge",
                ats_type=record.get("ats_type"),
            )
        )
    return examples


def load_holdout(*, truncate: int = 2000, path: Path = HOLDOUT_PATH) -> list[GoldExample]:
    examples: list[GoldExample] = []
    for record in _load_jsonl(path):
        examples.append(
            GoldExample(
                id=str(record["id"]),
                row=_row_from_record(record, truncate=truncate),
                labels=dict(record.get("labels") or {}),
                mask=tuple(record.get("mask") or ()),
                note=record.get("note"),
                source="holdout",
                ats_type=record.get("ats_type"),
            )
        )
    return examples


def load_gold(*, truncate: int = 2000, include_holdout: bool = True) -> list[GoldExample]:
    examples = load_edge_cases(truncate=truncate)
    if include_holdout:
        examples.extend(load_holdout(truncate=truncate))
    return examples


# --- building the holdout ---------------------------------------------------

#: Which provider fields become labels, and what has to be masked when they
#: do. A row can serve as a label for several fields at once only if the
#: union of their masks does not remove the evidence for the others, so the
#: builder assigns each row exactly one target field.
_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("placement", "is_remote", ()),
    ("salary_min", "salary_min", ("salary_summary",)),
    ("employment_type", "employment_type", ("commitment", "employment_type")),
    ("language", "language", ()),
    ("country_iso", "country_iso", ()),
)


def build_holdout(
    rows: Iterable[dict[str, Any]],
    *,
    per_target: int = 200,
    truncate: int = 2000,
) -> list[dict[str, Any]]:
    """Turn snapshot rows carrying provider values into labelled records.

    Each row is assigned to exactly one target field so its mask cannot
    destroy the evidence for another label. Rows are only accepted when they
    have enough description to be answerable at all — a label on an empty
    body measures nothing about extraction quality.
    """
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in _TARGETS}
    for row in rows:
        description = str(row.get("description") or "")
        if len(description.strip()) < 400:
            continue
        for target, column, mask in _TARGETS:
            if len(buckets[target]) >= per_target:
                continue
            raw = row.get(column)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue

            label_value: Any
            if target == "placement":
                as_bool = to_bool(raw)
                if as_bool is None:
                    continue
                # A provider ``False`` only rules remote out; it does not
                # distinguish onsite from hybrid, so it cannot be a
                # three-way label. Score it as the binary question instead.
                label_value = "remote" if as_bool else None
                labels: dict[str, Any] = {"is_remote": as_bool}
                if label_value is not None:
                    labels["placement"] = label_value
            elif target == "salary_min":
                try:
                    labels = {"salary_min": float(str(raw))}
                except (TypeError, ValueError):
                    continue
                currency = row.get("salary_currency")
                if isinstance(currency, str) and currency.strip():
                    labels["salary_currency"] = currency.strip().upper()[:3]
            elif target == "employment_type":
                labels = {"employment_type": str(raw).strip().upper()}
            elif target == "language":
                labels = {"language": str(raw).strip().lower()[:2]}
            else:
                labels = {"country_iso": str(raw).strip().upper()[:2]}

            record = {
                "id": f"holdout-{target}-{row['job_key'][:12]}",
                "url": row.get("url"),
                "title": row.get("title"),
                "company": row.get("company"),
                "ats_type": row.get("ats_type"),
                "location": row.get("location"),
                "description": description[: truncate * 2],
                "commitment": row.get("commitment"),
                "salary_summary": row.get("salary_summary"),
                "employment_type_provider": row.get("employment_type"),
                "salary_min_provider": None,
                "labels": labels,
                "mask": list(mask),
                "note": f"provider-labelled holdout for {target}",
            }
            buckets[target].append(record)
            break
    return [record for bucket in buckets.values() for record in bucket]


def write_holdout(records: Sequence[dict[str, Any]], path: Path = HOLDOUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@dataclass
class GoldStats:
    total: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_label: dict[str, int] = field(default_factory=dict)
    negative_labels: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, examples: Sequence[GoldExample]) -> GoldStats:
        stats = cls(total=len(examples))
        for example in examples:
            stats.by_source[example.source] = stats.by_source.get(example.source, 0) + 1
            for name, value in example.labels.items():
                stats.by_label[name] = stats.by_label.get(name, 0) + 1
                if value is None:
                    stats.negative_labels[name] = stats.negative_labels.get(name, 0) + 1
        return stats

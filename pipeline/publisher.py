"""Publish a directory of per-ATS scraped CSVs to Cloudflare R2.

Layout produced under ``<prefix>`` (default ``jobhive/v1``):

    jobhive/v1/manifest.json
    jobhive/v1/all.{csv,parquet}     # full snapshot, both formats
    jobhive/v1/<ats>/jobs.csv        # per-ATS jobs slice
    jobhive/v1/<ats>/jobs.parquet    # idem in parquet

Tenant lists (``<ats>/companies.csv`` and the aggregated
``companies.{csv,parquet}``) are owned by the GitHub Actions workflow
``.github/workflows/publish-ats-companies.yml`` — the publisher only
touches the **jobs** side of the bucket. ``manifest.json`` is read,
patched (jobs entries updated, ``companies`` / ``by_ats_companies``
preserved), and re-uploaded so the two writers never clobber each
other.

Old layout (``jobs/all.parquet``, ``jobs/by-ats/*``, ``jobs/by-date/*``,
``companies/*``) is wiped on first run by :meth:`prune_legacy_paths`.

Memory: every pass is built on polars LazyFrames so no full-corpus
DataFrame is ever materialized.

  Pass 1 — per-ATS lazy: ``pl.scan_csv`` → vectorized enrichment
           expressions → ``sink_csv`` (streaming write to a temp
           file). The same temp CSV is re-scanned to ``sink_parquet``
           (streaming convert) and once more to harvest a thin keys
           frame (small ``collect``). Per-ATS peak is bounded by
           polars' streaming buffers, not the slice's row count.

  Pass 2 — cross-ATS dedup as window functions on the concatenated
           thin keys frame: a single ``sort + filter`` pass per stage
           with ``pl.col(...).first().over(group)`` instead of
           Python-set bookkeeping. The keys frame is the only memory
           peak in this pass. Its last two stages demote the gated
           sources (:data:`GATED_LINK_SOURCES`) against the employer's
           own board, so the published link is one a signed-out
           reader can open; the gated row's salary is carried onto
           the row that replaced it.

  Pass 3 — global ``all.parquet`` is built by lazy-scanning each
           per-ATS temp CSV, ``semi``-joining against its survivor
           index frame, ``diagonal_relaxed``-concatenating the parts
           and ``sink_parquet``-streaming the result. Nothing is
           materialized whole.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from ats_scrapers._version import __version__
from ats_scrapers.enrichment import infer_is_remote, parse_salary_range
from ats_scrapers.enrichment.derived import infer_seniority, parse_salary_block
from ats_scrapers.enrichment.uslocation import looks_us
from ats_scrapers.exceptions import StorageError
from ats_scrapers.models import ATSType

# Pull the keyword list used by ``infer_is_remote`` so the lazy
# enrichment path can express the rule as vectorized polars
# expressions. The list is optional — if a deploy ships a stripped
# variant of ``derived.py`` that doesn't export it, the publisher
# falls back to the Python callback via ``map_elements``.
try:
    from ats_scrapers.enrichment.derived import REMOTE_KEYWORDS as _REMOTE_KEYWORDS
except ImportError:
    _REMOTE_KEYWORDS = ()

try:
    from ats_scrapers.enrichment.derived import SENIORITY_RULES as _SENIORITY_RULES
except ImportError:
    _SENIORITY_RULES = ()

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pipeline.r2 import R2Client

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "jobhive/v1"
CACHE_CONTROL_LATEST = "public, max-age=300"  # manifest + latest data files

# ``all`` ships both formats — parquet for typed pandas / DuckDB
# consumers, CSV (~2.3 GB at the current ~4M-row corpus) for
# spreadsheet, ``grep``, and tools that don't speak parquet. The CSV
# is built by streaming the merged parquet back through polars, so
# the per-row data lives on disk in two places but never both in
# RAM.
FORMATS_ALL = ("csv", "parquet")
FORMATS_PER_ATS = ("csv", "parquet")

# Common pl.scan_csv options across every read path. ``ignore_errors``
# is what lets the scanner fall back to string for a column whose first
# 10k-row sniff says int but later rows hold an alphanumeric ID.
_SCAN_CSV_KWARGS: dict[str, object] = {
    "infer_schema_length": 10000,
    "ignore_errors": True,
}


@dataclass
class PublishResult:
    """Summary of what was uploaded in a single publish run."""

    manifest_key: str
    files: list[str] = field(default_factory=list)
    total_jobs: int = 0
    total_jobs_raw: int = 0
    ats_count: int = 0
    duration_seconds: float = 0.0


class DatasetPublisher:
    """Builds and publishes a versioned dataset to R2.

    The publisher is responsible for **jobs only**. Companies / tenant
    lists are written by the CI workflow. Both writers share
    ``manifest.json`` via read-modify-write, so the publisher must
    never touch the ``companies`` or ``by_ats_companies`` keys.
    """

    def __init__(
        self,
        r2_client: R2Client,
        *,
        prefix: str = DEFAULT_PREFIX,
        write_parquet: bool = True,
        write_all_csv: bool = True,
    ) -> None:
        self._r2 = r2_client
        self._prefix = prefix.strip("/")
        self._write_parquet = write_parquet
        self._write_all_csv = write_all_csv
        if write_parquet:
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise StorageError(
                    "pyarrow is required when write_parquet=True. "
                    "Install with `pip install ats-scrapers[publish]`."
                ) from exc

    def publish_from_directory(
        self,
        source_dir: Path,
        *,
        ats_csv_pattern: str = "{ats}/jobs.csv",
    ) -> PublishResult:
        """Publish jobs from a local directory.

        Reads ``<source_dir>/<ats>/jobs.csv`` for every supported ATS,
        produces:

        1. Per-ATS slice ``<prefix>/<ats>/jobs.{csv,parquet}`` (raw —
           no cross-ATS dedup, so single-ATS consumers see what that
           ATS exposes).
        2. Cross-ATS deduped global snapshot ``<prefix>/all.{csv,parquet}``.
        3. Patched ``<prefix>/manifest.json`` with refreshed
           ``all`` and ``by_ats`` jobs entries; ``companies`` and
           ``by_ats_companies`` (CI-owned) are preserved untouched.

        Then deletes the legacy paths
        (``<prefix>/jobs/*``, ``<prefix>/companies/*``).
        """
        started = datetime.now(tz=UTC)
        files_uploaded: list[str] = []
        manifest_key = f"{self._prefix}/manifest.json"
        existing_manifest = _load_existing_manifest(self._r2, manifest_key)
        _guard_suspicious_empty_job_slices(
            source_dir=source_dir,
            ats_csv_pattern=ats_csv_pattern,
            existing_manifest=existing_manifest,
        )

        # ExitStack owns every per-ATS CSV temp: Pass 1 streams each
        # enriched per-ATS slice into one of these, then Pass 3
        # ``scan_csv``s the same files (no re-enrichment) to build
        # the global all.parquet. Files are unlinked at function exit.
        with ExitStack() as stack:
            per_ats_csv_paths: dict[str, Path] = {}
            per_ats_entries: dict[ATSType, dict[str, object]] = {}
            quarantine_csv_paths: list[Path] = []
            schema_union: list[str] = []
            seen_cols: set[str] = set()

            any_csv_found = False
            for ats in ATSType:
                if ats is ATSType.CUSTOM:
                    continue
                source_path = source_dir / ats_csv_pattern.format(ats=ats.value)
                if not source_path.exists():
                    continue
                # Defense against the publisher firing while a scraper
                # is mid-write (cron + ad-hoc publish race): a 0-byte
                # CSV will blow up ``collect_schema`` with NoDataError.
                # Skip the slice for this run; the next cron publish
                # picks it up once the scraper has finished.
                if source_path.stat().st_size == 0:
                    logger.warning(
                        "%s: source CSV is empty (likely mid-write by a "
                        "concurrent scraper); skipping for this publish.",
                        ats.value,
                    )
                    continue
                any_csv_found = True

                # Build the lazy enriched chain for this ATS slice.
                lf = pl.scan_csv(source_path, **_SCAN_CSV_KWARGS)
                lf = lf.with_columns(pl.lit(ats.value).alias("ats_type"))
                lf = _enrich_lazy(lf)

                try:
                    schema_names = lf.collect_schema().names()
                except pl.exceptions.NoDataError:
                    # Header-only or otherwise empty CSV — same recovery
                    # as the size==0 branch above.
                    logger.warning(
                        "%s: source CSV has no rows; skipping.", ats.value,
                    )
                    continue
                for col in schema_names:
                    if col not in seen_cols:
                        seen_cols.add(col)
                        schema_union.append(col)

                # Rows failing a quality gate leave the dataset here, so
                # neither the per-ATS slice nor ``all`` can carry them.
                lf, rejected = _partition_quarantine(
                    lf, schema_names, now=started
                )
                if rejected is not None:
                    reject_path = stack.enter_context(_temp_file(".csv"))
                    rejected.sink_csv(reject_path)
                    if _csv_has_rows(reject_path):
                        quarantine_csv_paths.append(reject_path)

                csv_path = stack.enter_context(_temp_file(".csv"))
                # ``sink_csv`` runs the lazy chain through polars'
                # streaming engine — the per-ATS slice is never
                # materialized as one DataFrame in RAM.
                lf.sink_csv(csv_path)
                per_ats_csv_paths[ats.value] = csv_path

                entry, _ = self._upload_per_ats_streaming(
                    csv_path=csv_path,
                    base_key=f"{self._prefix}/{ats.value}/jobs",
                )
                per_ats_entries[ats] = entry
                files_uploaded.extend(_collect_uploaded_keys(entry))

            if not any_csv_found:
                raise StorageError(f"No ATS CSVs found in {source_dir}")

            # ---- Pass 2: cross-ATS dedup directly on the per-ATS temp CSVs ---
            # Build the keys frame as a single lazy scan-and-concat chain
            # rather than collecting per-ATS keys into separate eager
            # DataFrames during Pass 1. This avoids the cumulative
            # ~MB-per-ATS resident growth and lets polars' optimizer
            # decide when to materialize.
            outcome, n_raw, n_kept = _dedup_from_per_ats_csvs(
                per_ats_csv_paths
            )
            logger.info(
                "Cross-ATS dedup: %d → %d rows (%d duplicates removed)",
                n_raw,
                n_kept,
                n_raw - n_kept,
            )

            # ``salary_source`` only exists on rows that inherited pay
            # from a dropped gated posting, so it isn't in any per-ATS
            # slice's schema — add it to the union by hand or the
            # manifest would under-report the ``all`` columns.
            if any(
                not donation.is_empty() for donation in outcome.donations.values()
            ) and SALARY_SOURCE_COLUMN not in seen_cols:
                seen_cols.add(SALARY_SOURCE_COLUMN)
                schema_union.append(SALARY_SOURCE_COLUMN)

            # ---- Pass 3: stream all.parquet from the per-ATS temp CSVs ------
            all_entry = self._stream_write_all_polars(
                per_ats_csv_paths=per_ats_csv_paths,
                survivors=outcome.survivors,
                donations=outcome.donations,
                schema_union=schema_union,
                rows_total=n_kept,
                observed_at=started,
            )
            files_uploaded.extend(_collect_uploaded_keys(all_entry))

            quarantine_entry = self._upload_quarantine(quarantine_csv_paths)
            if quarantine_entry is not None:
                files_uploaded.extend(_collect_uploaded_keys(quarantine_entry))
            n_quarantined = (
                int(quarantine_entry["rows"])  # type: ignore[arg-type]
                if quarantine_entry is not None
                else 0
            )

            manifest_key = self._patch_and_upload_manifest(
                generated_at=started,
                stats_factory=lambda existing: {
                    "total_jobs": n_kept,
                    "total_jobs_raw": n_raw,
                    "total_jobs_quarantined": n_quarantined,
                    "total_companies": _sum_by_ats_companies_rows(existing),
                    "ats_count": len(per_ats_entries),
                    "schema_version": "2.0",
                    "schema_columns": schema_union,
                },
                all_entry=all_entry,
                by_ats=per_ats_entries,
                existing_manifest=existing_manifest,
            )
            files_uploaded.append(manifest_key)

            deleted = self.prune_legacy_paths()
            if deleted:
                logger.info("Deleted %d legacy keys", deleted)

            ended = datetime.now(tz=UTC)
            return PublishResult(
                manifest_key=manifest_key,
                files=files_uploaded,
                total_jobs=n_kept,
                total_jobs_raw=n_raw,
                ats_count=len(per_ats_entries),
                duration_seconds=(ended - started).total_seconds(),
            )

    def _upload_quarantine(
        self, quarantine_csv_paths: list[Path]
    ) -> dict[str, object] | None:
        """Publish the rows the quality gates rejected, or ``None``.

        Kept out of ``all`` and the per-ATS slices deliberately: these
        rows are wrong, not merely low-confidence. Writing them anyway
        makes the deletions auditable — a rule that starts eating good
        data shows up as a jump in ``total_jobs_quarantined`` and can be
        diffed row by row against the reason column.
        """
        if not quarantine_csv_paths:
            return None
        entry: dict[str, object] = {}
        with _temp_file(".parquet") as pq_path:
            pl.concat(
                [
                    pl.scan_csv(path, **_SCAN_CSV_KWARGS)
                    for path in quarantine_csv_paths
                ],
                how="diagonal_relaxed",
            ).sink_parquet(pq_path, compression="zstd")
            n_rows = pl.scan_parquet(pq_path).select(pl.len()).collect().item()
            sha, size = _file_sha_size(pq_path)
            key = f"{self._prefix}/quarantine.parquet"
            self._r2.upload(
                pq_path,
                key,
                content_type="application/vnd.apache.parquet",
                cache_control=CACHE_CONTROL_LATEST,
            )
        entry["parquet"] = self._public_or_key(key)
        entry["parquet_sha256"] = sha
        entry["parquet_size_bytes"] = size
        entry["rows"] = n_rows
        logger.info("Quarantined %d rows failing a quality gate", n_rows)
        return entry

    def prune_legacy_paths(self) -> int:
        """Delete every key under the pre-2.0 layout. Idempotent.

        Companies legacy paths are also deleted by the CI workflow's
        publisher script — calling them here as well makes the
        publisher correct in isolation when the CI hasn't run yet.
        """
        legacy_prefixes = [
            f"{self._prefix}/jobs/",
            f"{self._prefix}/companies/",
        ]
        keys: list[str] = []
        for prefix in legacy_prefixes:
            for obj in self._r2.list(prefix=prefix):
                key = obj.get("Key")
                if key:
                    keys.append(key)
        if not keys:
            return 0
        return self._r2.delete_many(keys)

    # --- internals ---------------------------------------------------------

    def _upload_per_ats_streaming(
        self,
        *,
        csv_path: Path,
        base_key: str,
    ) -> tuple[dict[str, object], int]:
        """Upload a per-ATS slice from a sunk temp CSV.

        Hashes + uploads the CSV, then ``scan_csv`` → ``sink_parquet``
        streams the parquet conversion through polars (no full Arrow
        table in RAM). Returns the manifest entry and the row count.
        """
        entry: dict[str, object] = {}

        csv_key = f"{base_key}.csv"
        csv_sha, csv_size = _file_sha_size(csv_path)
        self._r2.upload(
            csv_path,
            csv_key,
            content_type="text/csv",
            cache_control=CACHE_CONTROL_LATEST,
        )
        entry["csv"] = self._public_or_key(csv_key)
        entry["size_bytes"] = csv_size
        entry["sha256"] = csv_sha

        # Counting rows from the temp CSV is cheap and avoids carrying
        # the row count separately from the lazy chain.
        n_rows = (
            pl.scan_csv(csv_path, **_SCAN_CSV_KWARGS)
            .select(pl.len())
            .collect()
            .item()
        )
        entry["rows"] = n_rows

        if self._write_parquet:
            parquet_key = f"{base_key}.parquet"
            with _temp_file(".parquet") as pq_path:
                pl.scan_csv(csv_path, **_SCAN_CSV_KWARGS).sink_parquet(
                    pq_path, compression="zstd"
                )
                pq_sha, pq_size = _file_sha_size(pq_path)
                self._r2.upload(
                    pq_path,
                    parquet_key,
                    content_type="application/vnd.apache.parquet",
                    cache_control=CACHE_CONTROL_LATEST,
                )
            entry["parquet"] = self._public_or_key(parquet_key)
            entry["parquet_size_bytes"] = pq_size
            entry["parquet_sha256"] = pq_sha

        return entry, n_rows

    def _stamp_liveness(
        self, stage_stack: ExitStack, all_pq: Path, observed_at: datetime
    ) -> Path:
        """Add ``first_seen_at`` / ``last_seen_at`` to the global snapshot.

        The pipeline is a pure snapshot model: each run republishes
        whatever the boards currently serve, with no tombstoning, so
        "has this posting been up for three days or three months?" was
        not expressible at all. ``posted_at`` doesn't answer it — it is
        the employer's own claim, is missing on a large share of rows,
        and never changes when a posting is quietly relisted.

        Every row in this snapshot is live right now, so ``last_seen_at``
        is the publish timestamp. ``first_seen_at`` carries forward from
        the previous run through a two-column sidecar; a posting we have
        never seen starts today. A posting that disappears and returns
        restarts its clock, which is the honest reading — we cannot tell
        a relist from a fresh posting without the sweep below.

        This is the cheap half of audit finding 02. It makes staleness
        *measurable*; it does not detect a posting that 404s while still
        being served in the board's index. That needs a link-liveness
        sweep — an out-of-band job that walks ``url`` on a rotation and
        records a ``dead_at``, sized by the board's rate limits rather
        than by this publish. Landing these columns first gives that job
        somewhere to write and gives us the staleness baseline to
        justify it.
        """
        previous = self._load_first_seen()
        stamped = stage_stack.enter_context(_temp_file(".parquet"))
        now = pl.lit(observed_at).cast(pl.Datetime(time_unit="us", time_zone="UTC"))

        lf = pl.scan_parquet(all_pq).with_columns(_seen_key_expr())
        lf = lf.join(previous.lazy(), on=_SEEN_KEY_COLUMN, how="left").with_columns(
            pl.coalesce(pl.col(FIRST_SEEN_COLUMN), now).alias(FIRST_SEEN_COLUMN),
            now.alias(LAST_SEEN_COLUMN),
        )
        lf.drop(_SEEN_KEY_COLUMN).sink_parquet(stamped, compression="zstd")

        self._save_first_seen(stamped)
        return stamped

    def _load_first_seen(self) -> pl.DataFrame:
        """Previous run's first-sighting index, empty when unavailable.

        Never fatal: a missing or unreadable sidecar costs the age of
        the postings it covered, which self-heals as they are seen
        again, and is not worth failing a publish over.
        """
        empty = pl.DataFrame(
            schema={
                _SEEN_KEY_COLUMN: pl.String,
                FIRST_SEEN_COLUMN: pl.Datetime(time_unit="us", time_zone="UTC"),
            }
        )
        key = f"{self._prefix}/{FIRST_SEEN_SIDECAR}"
        try:
            body = self._r2.get_bytes(key)
        except StorageError:
            body = None
        if not body:
            logger.info("No %s yet; every posting starts its clock today.", key)
            return empty
        try:
            frame = pl.read_parquet(io.BytesIO(body))
        except Exception:
            logger.warning(
                "%s is unreadable; restarting first-seen tracking.", key, exc_info=True
            )
            return empty
        if set(frame.columns) != {_SEEN_KEY_COLUMN, FIRST_SEEN_COLUMN}:
            logger.warning("%s has an unexpected schema; ignoring it.", key)
            return empty
        return frame

    def _save_first_seen(self, stamped: Path) -> None:
        """Rewrite the sidecar from the snapshot we just built.

        Written from the published rows rather than merged with the old
        sidecar, so keys for postings that have gone away age out
        instead of growing without bound.
        """
        with _temp_file(".parquet") as sidecar:
            (
                pl.scan_parquet(stamped)
                .with_columns(_seen_key_expr())
                .select(_SEEN_KEY_COLUMN, FIRST_SEEN_COLUMN)
                .unique(subset=_SEEN_KEY_COLUMN, keep="first")
                .sink_parquet(sidecar, compression="zstd")
            )
            try:
                self._r2.upload(
                    sidecar,
                    f"{self._prefix}/{FIRST_SEEN_SIDECAR}",
                    content_type="application/vnd.apache.parquet",
                    cache_control=CACHE_CONTROL_LATEST,
                )
            except StorageError:
                logger.warning(
                    "Could not upload %s; ages restart next run.",
                    FIRST_SEEN_SIDECAR,
                )

    def _stream_write_all_polars(
        self,
        *,
        per_ats_csv_paths: dict[str, Path],
        survivors: dict[str, pl.DataFrame],
        schema_union: list[str],
        rows_total: int,
        observed_at: datetime,
        donations: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, object]:
        """Stream the global ``all.parquet`` from the per-ATS temp CSVs.

        Three stages, all streaming:

        1. Per-ATS — ``scan_csv`` + ``semi``-join against its
           survivor index frame, sunk to a per-ATS temp parquet via
           ``sink_parquet``. Polars's semi-join on a small RHS is
           hash-probe, so the LHS streams. Slices with a ``donations``
           frame also take a ``left``-join to inherit the pay of the
           gated rows they replaced.

        2. Merge parquet — the per-ATS temp parquets are concatenated
           into the global ``all.parquet`` via polars' lazy
           ``concat(diagonal_relaxed)`` + ``sink_parquet``, so
           heterogeneous per-ATS schemas are unified and peak memory
           is one Arrow batch.

        3. Convert to CSV — the merged parquet is re-scanned and
           ``sink_csv``'d for the ``all.csv`` artifact (~2.3 GB at
           current corpus size). Polars streams batches; nothing is
           materialized whole.
        """
        all_entry: dict[str, object] = {"rows": rows_total}

        with ExitStack() as stage_stack:
            per_ats_parquets: list[Path] = []
            for ats in ATSType:
                if ats is ATSType.CUSTOM:
                    continue
                survivor_frame = survivors.get(ats.value)
                if survivor_frame is None or survivor_frame.is_empty():
                    continue
                csv_path = per_ats_csv_paths.get(ats.value)
                if csv_path is None:
                    continue

                pq_temp = stage_stack.enter_context(_temp_file(".parquet"))
                lf = (
                    pl.scan_csv(csv_path, **_SCAN_CSV_KWARGS)
                    .with_row_index(name="_local_idx")
                    .join(survivor_frame.lazy(), on="_local_idx", how="semi")
                )
                donation = (donations or {}).get(ats.value)
                if donation is not None and not donation.is_empty():
                    lf = _apply_salary_donations(lf, donation)
                lf.drop("_local_idx").sink_parquet(pq_temp, compression="zstd")
                per_ats_parquets.append(pq_temp)

            # Build the merged parquet first — both ``all.parquet`` and
            # ``all.csv`` (when configured) source from this file so the
            # CSV path doesn't re-do the per-ATS semi-joins.
            all_pq = stage_stack.enter_context(_temp_file(".parquet"))
            if per_ats_parquets:
                _merge_parquets_streaming(per_ats_parquets, all_pq)
            else:
                pl.DataFrame(
                    schema=dict.fromkeys(schema_union, pl.String)
                ).write_parquet(all_pq, compression="zstd")

            all_pq = self._stamp_liveness(stage_stack, all_pq, observed_at)

            if "parquet" in FORMATS_ALL and self._write_parquet:
                pq_key = f"{self._prefix}/all.parquet"
                pq_sha, pq_size = _file_sha_size(all_pq)
                self._r2.upload(
                    all_pq,
                    pq_key,
                    content_type="application/vnd.apache.parquet",
                    cache_control=CACHE_CONTROL_LATEST,
                )
                all_entry["parquet"] = self._public_or_key(pq_key)
                all_entry["parquet_size_bytes"] = pq_size
                all_entry["parquet_sha256"] = pq_sha
                all_entry["size_bytes"] = pq_size
                all_entry["sha256"] = pq_sha

            if "csv" in FORMATS_ALL and self._write_all_csv:
                csv_key = f"{self._prefix}/all.csv"
                with _temp_file(".csv") as all_csv:
                    pl.scan_parquet(all_pq).sink_csv(all_csv)
                    csv_sha, csv_size = _file_sha_size(all_csv)
                    self._r2.upload(
                        all_csv,
                        csv_key,
                        content_type="text/csv",
                        cache_control=CACHE_CONTROL_LATEST,
                    )
                all_entry["csv"] = self._public_or_key(csv_key)
                # CSV's size + sha live in the canonical ``size_bytes``
                # / ``sha256`` slots (consumers default-fetching the
                # text format see the matching pair). Parquet keeps its
                # own ``parquet_*`` fields populated above.
                all_entry["size_bytes"] = csv_size
                all_entry["sha256"] = csv_sha

        return all_entry

    def _patch_and_upload_manifest(
        self,
        *,
        generated_at: datetime,
        stats_factory,
        all_entry: dict[str, object],
        by_ats: dict[ATSType, dict[str, object]],
        existing_manifest: dict[str, object] | None = None,
    ) -> str:
        """Read existing manifest, replace jobs-related fields, preserve
        the companies block written by the CI."""
        key = f"{self._prefix}/manifest.json"
        existing = (
            existing_manifest
            if existing_manifest is not None
            else _load_existing_manifest(self._r2, key)
        )

        manifest: dict[str, object] = {**existing}
        manifest["version"] = "2.0"
        manifest["generator"] = f"ats-scrapers/{__version__}"
        manifest["generated_at"] = generated_at.isoformat()
        # ``updated_at`` is the "manifest last touched" timestamp; both
        # writers (publisher + CI companies workflow) bump it so a
        # client like the homepage that reads only ``updated_at`` for
        # the freshness badge sees the latest write regardless of which
        # writer ran most recently. Format matches what the CI script
        # writes: UTC ``Z``-suffixed seconds.
        manifest["updated_at"] = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["stats"] = stats_factory(existing)
        manifest["all"] = all_entry
        manifest["by_ats"] = {ats.value: entry for ats, entry in by_ats.items()}

        # Drop fields from the pre-2.0 layout if they survived the
        # legacy-path prune. Their data is gone so the entries point
        # nowhere.
        for legacy in ("by_date", "companies_by_ats"):
            manifest.pop(legacy, None)

        body = json.dumps(manifest, indent=2, sort_keys=True, default=str).encode(
            "utf-8"
        )
        self._r2.upload_bytes(
            body,
            key,
            content_type="application/json",
            cache_control=CACHE_CONTROL_LATEST,
        )
        return key

    def _public_or_key(self, key: str) -> str:
        return self._r2.public_url(key) or key


# --- Cross-ATS dedup -------------------------------------------------------


# When the same (company, title, location) shows up under multiple ATSes,
# we keep the row from the highest-priority ATS (lowest number wins).
#
# Every source the pipeline publishes is listed. An unlisted one falls to
# ``_DEFAULT_DEDUP_PRIORITY``, which ties it with the public job boards —
# survivable, but the winner within a tie is decided by scan order rather
# than by anything meaningful, so a new source belongs in a tier here.
ATS_DEDUP_PRIORITY: dict[str, int] = {
    # Direct employer ATSes
    "adp": 1, "ashby": 1, "avature": 1, "bamboohr": 1, "beisen": 1,
    "beisen_legacy": 1,
    "breezy": 1, "cornerstone": 1,
    "darwinbox": 1, "dayforce": 1,
    "greenhouse": 1, "gupy": 1, "herp": 1, "hrmos": 1, "icims": 1, "jazzhr": 1, "join_com": 1, "jobvite": 1, "keka": 1,
    "lever": 1,
    "moka": 1, "oracle": 1, "pageup": 1, "paycom": 1, "paylocity": 1, "personio": 1, "phenom": 1, "pinpoint": 1,
    "recruitee": 1,
    "recruiterbox": 1, "rippling": 1, "smartrecruiters": 1, "softgarden": 1,
    "successfactors": 1, "taleo": 1, "teamtailor": 1, "ukg": 1, "workable": 1,
    "workday": 1,
    # Big-tech bespoke careers — also priority 1 (single-tenant, canonical)
    "amazon": 1, "apple": 1, "bytedance": 1, "google": 1, "meta": 1,
    "tesla": 1, "tiktok": 1, "uber": 1,
    # Public job boards that mirror employer postings. They rank below
    # the employer's own board but above the gated and sourcing tiers,
    # because their links open for a reader who isn't signed in.
    "builtin": 2, "getonbrd": 2, "jobsch": 2, "manfred": 2,
    "programathor": 2, "remoteok": 2, "thehub": 2, "wanted": 2,
    "wellfound": 2, "weworkremotely": 2, "ycombinator": 2,
    # Hybrid jobboards
    "welcometothejungle": 3, "mercor": 3, "gem": 3,
    "seek": 4,
    # Sourcing/matching layer that mirrors others
    "eightfold": 5,
    # National public-sector aggregators — government-curated but the
    # same role often appears here AND on the employer's direct ATS.
    "bundesagentur": 6,
    "arbetsformedlingen": 6,
    "eures": 6,
    "jobbankca": 6,
    "usajobs": 6,
}

# Where an unlisted source lands: the public-job-board tier, matching
# what the old bare ``.get(ats, 2)`` did for every source missing above.
_DEFAULT_DEDUP_PRIORITY = 2

# Sources whose postings sit behind a sign-in wall. A reader who isn't
# logged in can't open the link at all, so when the same posting also
# exists on the employer's own board we keep the employer's row — the
# ATS priority above already ranks it higher, but the gated source
# formats titles and locations differently enough that the exact and
# country-blocked passes never pair the two (see Pass 6 / Pass 7).
GATED_LINK_SOURCES: frozenset[str] = frozenset({"welcometothejungle"})

# The priority number that means "the employer's own board". Gated
# rows are only demoted against these, so a gated posting still beats
# the public aggregators (seek, eightfold, eures, bundesagentur, …)
# that sit below it in ``ATS_DEDUP_PRIORITY``.
DIRECT_EMPLOYER_PRIORITY = 1

# Pay columns carried from a dropped gated row onto the employer row
# that replaced it. Welcome to the Jungle is one of the few sources
# that publishes salary, so dropping its rows outright would take the
# pay figure with them. Currency and period travel with the numbers —
# a min/max without them is unusable.
_SALARY_CARRY_COLUMNS: tuple[tuple[str, pl.DataType], ...] = (
    ("salary_min", pl.Float64()),
    ("salary_max", pl.Float64()),
    ("salary_currency", pl.String()),
    ("salary_period", pl.String()),
    ("salary_summary", pl.String()),
)
_DONOR_PREFIX = "_donor_"
# Records which source the carried pay came from. Null on every row
# that kept its own salary, so a consumer can tell the two apart.
SALARY_SOURCE_COLUMN = "salary_source"

# Pass 7 fuzz-matches the punctuation-free slug rather than the raw
# title, because punctuation is exactly what the gated source changes:
# ``"Manager of Account Executives (Startups)"`` against ``"Manager,
# Account Executives, Startups"`` scores 70 on the raw titles.
#
# It also uses ``token_sort_ratio`` rather than Pass 5's
# ``token_set_ratio``, and a higher bar. ``token_set_ratio`` rewards
# the tokens two titles share and forgives the ones they don't, which
# is too loose once a wrong pair also hands over its salary: measured
# on real Anthropic postings it scored "Senior Staff Software Engineer
# (Node Infrastructure)" against "Staff Software Engineer, Data
# Infrastructure" at 93.8, and "Strategic Account Executive (GSI)"
# against "Manager of Account Executive (Strategic Sales)" at 93.1.
# Sorted-token matching puts those at 81.7 and 80.0 while the real
# pairs — plurals, ``Engineering``/``Engineer``, reordered qualifiers —
# stay at 95.8 and above.
_GATED_FUZZY_MARGIN = 5


def _company_norm_expr(column: str = "company") -> pl.Expr:
    """Punctuation-free employer key used to block the dedup passes.

    Lowercases inline rather than trusting the caller: the character
    class is ``[^a-z0-9]``, so on a column that still has capitals it
    silently eats them — ``"OpenAI"`` becomes ``"pen"`` — and every
    pass keyed on it would quietly stop matching instead of failing.
    The keys frame already lowercases ``company``, which makes this a
    no-op there, but the expression is now correct wherever it's used.
    """
    return (
        pl.col(column)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]", "")
        .alias("_company_norm")
    )


# Employer values that name no employer. Two families end up here.
#
# Localized "withheld" markers: EURES aggregates national job services
# that hide the employer until a candidate applies through the official
# portal, so ~86% of FR rows say "non renseigné" and ~60% of ES rows are
# blank. The scrapers pass these through verbatim on purpose — the
# locale of the placeholder identifies the source service, and the
# postings themselves (title, location, description, pay) are real.
#
# Bare careers hostnames: Oracle, Phenom and Personio derive the
# employer from the careers URL when no curated name is configured,
# which publishes "jobs.bell.ca" as if it were a company.
#
# Both are facet poison — they collect thousands of unrelated postings
# under one meaningless label. These rows are quarantined out of the
# published dataset: an employer is the one fact every consumer joins
# on, and a posting that names none is not usable as market data.
#
# This is the single largest quarantine rule by volume, and the drop is
# deliberate rather than incidental — check ``placeholder_employer`` in
# quarantine.parquet before reading any change in the FR/ES row counts
# as a scraper regression.
#
# Detection lives here rather than in the scrapers so the per-ATS
# slices still carry the source's own value, where the locale of the
# placeholder identifies which national service withheld it.
_PLACEHOLDER_EMPLOYERS = frozenset({
    "-", "--", ".", "...", "?", "n/a", "na", "n.a.", "none", "null",
    "not specified", "not disclosed", "not provided", "unspecified",
    "unknown", "undisclosed", "confidential", "confidentiel",
    "konfidentiell", "vertraulich", "anonymous", "anonyme", "anonym",
    "non renseigné", "non renseigne", "non précisé", "non precise",
    "no se especifica", "no especificado", "sin especificar",
    "siehe beschreibung", "see description", "see job description",
    "voir description", "ver descripción", "ver descripcion",
    "employer", "company", "empresa", "entreprise", "arbeitgeber",
    "private", "privat", "particulier",
})

# A value that is nothing but a domain: "jobs.bell.ca", "careers.acme.com".
# Anchored and TLD-gated so real names keep their dots — "Booz Allen
# Hamilton Inc." has spaces, and "Yahoo! Inc." has no dotted tail.
_HOSTNAME_EMPLOYER_RE = r"^(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+$"


FIRST_SEEN_COLUMN = "first_seen_at"
LAST_SEEN_COLUMN = "last_seen_at"
FIRST_SEEN_SIDECAR = "first_seen.parquet"
_SEEN_KEY_COLUMN = "_seen_key"


def _seen_key_expr() -> pl.Expr:
    """Identity used to recognize a posting across runs.

    Prefers the source's own id over the URL: boards rewrite URLs when a
    posting is edited or the employer renames its board, and a changed
    URL would look like a brand-new posting and reset its age. ``ats_id``
    is blank on a few sources, which fall back to the URL.
    """
    ats_id = pl.col("ats_id").cast(pl.String).fill_null("").str.strip_chars()
    return (
        pl.when(ats_id.str.len_bytes() > 0)
        .then(pl.col("ats_type").cast(pl.String).fill_null("") + pl.lit("|") + ats_id)
        .otherwise(pl.col("url").cast(pl.String).fill_null(""))
        .alias(_SEEN_KEY_COLUMN)
    )


def _placeholder_company_expr(column: str = "company") -> pl.Expr:
    """True where the employer value names no actual employer."""
    normalized = pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase()
    return normalized.is_in(sorted(_PLACEHOLDER_EMPLOYERS)) | normalized.str.contains(
        _HOSTNAME_EMPLOYER_RE
    )


def _title_slug_expr(column: str = "title") -> pl.Expr:
    """Punctuation-free title used to pair a gated posting with the
    employer's own listing of the same job.

    Welcome to the Jungle rewrites the employer's punctuation —
    ``"Staff Software Engineer, Environments Infrastructure"`` on
    Greenhouse becomes ``"Staff Software Engineer (Environments
    Infrastructure)"`` — so every exact pass keyed on the title text
    misses. Collapsing all non-alphanumerics to single spaces makes
    the two identical. Expects an already-lowercased column.
    """
    return (
        pl.col(column)
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
    )


def _key_col_or_empty(schema_names: list[str], name: str) -> pl.Expr:
    """Return ``pl.col(name)`` cast to String + filled, or an empty
    string literal if the column doesn't exist on this slice."""
    if name in schema_names:
        return (
            pl.col(name)
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.strip_chars()
        )
    return pl.lit("", dtype=pl.String)


def _donor_salary_exprs(
    schema_names: list[str], *, carry: bool
) -> list[pl.Expr]:
    """Pay columns for the keys frame, one ``_donor_``-prefixed column
    per entry in :data:`_SALARY_CARRY_COLUMNS`.

    Only slices that can donate (the gated sources) read the real
    column; everyone else gets a typed null literal. The keys frame
    spans the whole corpus, so this keeps the added width to a few
    all-null columns for the ~4M rows that will never donate.
    """
    exprs: list[pl.Expr] = []
    for name, dtype in _SALARY_CARRY_COLUMNS:
        alias = f"{_DONOR_PREFIX}{name}"
        if carry and name in schema_names:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(alias))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(alias))
    return exprs


# Country-code map for the locations the EU aggregators (eures /
# bundesagentur / France Travail / arbetsformedlingen) emit. We don't
# need to recognise every country in the world — only the ones that
# show up in the duplicate-prone pairs. Anything we don't match falls
# through to ``""`` and Phase 1 / 2 dedup just skip that row.
_COUNTRY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DE", ("deutschland", "germany", "allemagne")),
    ("FR", ("france", "frankreich")),
    ("BE", ("belgique", "belgium", "belgien", "belgië")),
    ("AT", ("österreich", "austria", "autriche")),
    ("NL", ("nederland", "netherlands", "pays-bas", "niederlande")),
    ("IT", ("italia", "italy", "italien")),
    ("ES", ("españa", "spain", "espagne", "spanien")),
    ("PT", ("portugal",)),
    ("PL", ("polska", "poland", "pologne", "polen")),
    ("CH", ("schweiz", "suisse", "switzerland", "svizzera")),
    ("LU", ("luxembourg", "luxemburg")),
    ("DK", ("danmark", "denmark", "dänemark")),
    ("SE", ("sverige", "sweden", "schweden")),
    ("NO", ("norge", "norway", "norwegen")),
    ("FI", ("suomi", "finland", "finnland")),
    ("IE", ("ireland", "irlande", "irland")),
    ("CZ", ("česko", "czech", "tschechien")),
    ("US", ("united states", "u.s.a", "usa")),
    # "UK" is not the ISO code (GB is), so the trailing-two-letter path
    # can never resolve it — postings written "London, UK" otherwise
    # come out with no country at all.
    ("GB", ("united kingdom", "england", "scotland", "wales", "uk", "u.k")),
    ("CA", ("canada",)),
)

# eures encodes the country as a 2-letter NUTS prefix followed by a
# parenthesised region code, e.g. ``"DE (DEA58)"`` / ``"FR (FRK21)"``.
# Matching just the two-letter prefix would false-positive on
# titles-as-locations like ``"Software Engineer, Remote"`` — we
# require the parens too. Case-insensitive: the pipeline lowercases
# ``location`` during harvest.
_NUTS_PREFIX_RE = re.compile(r"^\s*([a-z]{2})\s*\(", re.IGNORECASE)
_TRAILING_ISO_RE = re.compile(r"(?:^|[,\s(/])([a-z]{2})(?:[\s).]*)$", re.IGNORECASE)

# Word-boundary-anchored country needle patterns. Substring matching
# false-positived on common European place names — e.g. ``"usa"``
# inside ``"Lausanne"`` (CH) had Lausanne jobs ending up tagged as
# US. ``\b`` flanks every needle so the match requires non-word
# context on each side; the multi-character needles (``"u.s.a"``,
# ``"new zealand"``) still work because ``re.escape`` keeps the
# internal punctuation literal.
_COUNTRY_PATTERNS_RE: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        code,
        re.compile(
            "|".join(rf"\b{re.escape(n)}\b" for n in needles),
            re.IGNORECASE,
        ),
    )
    for code, needles in _COUNTRY_PATTERNS
)
_COUNTRY_CODES = {country_code for country_code, _ in _COUNTRY_PATTERNS}

# US states and Canadian provinces. Postings from North America almost
# never name the country — they write ``"Austin, TX"`` / ``"Toronto,
# ON"`` — so without these the largest slice of the corpus resolves to
# nothing at all. Neither set overlaps the other.
_US_STATE_CODES = frozenset({
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME",
    "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM",
    "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VA", "VT", "WA", "WI", "WV", "WY",
})
_CA_PROVINCE_CODES = frozenset({
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC",
    "SK", "YT",
})

# Spelled-out state names, for the ``"Boulder, Colorado"`` /
# ``"New York, New York"`` form that carries no two-letter token at all.
# Georgia is deliberately absent: it is also a country, and
# ``"Tbilisi, Georgia"`` must not resolve to the US.
_US_STATE_NAMES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
})

# Four two-letter codes mean both a US state and a country, so the code
# alone can't decide. Each entry pairs a set of anchor components with
# the verdict they force and the verdict to fall back on. The anchors and
# the defaults are picked from what the corpus actually contains:
# ``", CA"`` is California unless a province is spelled out beside it
# (Canadian rows write ``"Toronto, ON, CA"``), ``", DE"`` is Germany
# unless the city is one of Delaware's handful of towns, and ``", IN"`` /
# ``", ID"`` are Indiana / Idaho unless anchored by a major Indian or
# Indonesian city.
_DELAWARE_CITIES = frozenset({
    "wilmington", "newark", "dover", "middletown", "bear", "glasgow",
    "smyrna", "milford", "seaford", "georgetown", "lewes", "claymont",
    "hockessin", "christiana", "brookside", "elsmere", "milton", "camden",
    "harrington", "laurel", "selbyville", "townsend", "delmar", "clayton",
    "felton", "greenwood", "frankford", "millsboro", "edgemoor",
    "new castle", "pike creek", "rehoboth beach", "bethany beach",
    "dewey beach", "ocean view",
})
_INDIA_CITIES = frozenset({
    "mumbai", "navi mumbai", "delhi", "new delhi", "bangalore",
    "bengaluru", "hyderabad", "chennai", "kolkata", "pune", "ahmedabad",
    "surat", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane",
    "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad",
    "ludhiana", "agra", "nashik", "faridabad", "meerut", "rajkot",
    "gurgaon", "gurugram", "noida", "greater noida", "kochi", "cochin",
    "coimbatore", "chandigarh", "mysore", "mysuru", "trivandrum",
    "thiruvananthapuram", "bhubaneswar", "mohali", "vijayawada",
    "madurai", "guwahati", "dehradun", "jodhpur", "raipur", "ranchi",
    "amritsar",
})
_INDONESIA_CITIES = frozenset({
    "jakarta", "south jakarta", "west jakarta", "north jakarta",
    "east jakarta", "central jakarta", "surabaya", "bandung", "medan",
    "bekasi", "semarang", "makassar", "palembang", "tangerang",
    "south tangerang", "depok", "batam", "denpasar", "balikpapan",
    "yogyakarta", "malang", "samarinda", "pekanbaru", "banjarmasin",
    "bogor", "bali", "kalimantan", "kalimantan timur", "sumatera",
    "sulawesi", "jawa",
})

# code -> (anchor components, verdict when anchored, verdict otherwise)
_AMBIGUOUS_ADMIN_CODES: dict[str, tuple[frozenset[str], str, str]] = {
    "CA": (_CA_PROVINCE_CODES, "CA", "US"),
    "DE": (_DELAWARE_CITIES, "US", "DE"),
    "IN": (_INDIA_CITIES, "IN", "US"),
    "ID": (_INDONESIA_CITIES, "ID", "US"),
}

_COMPONENT_SPLIT_RE = re.compile(r"[,;|/]|\s+-\s+")


def _location_components(loc: str) -> list[str]:
    """Split a location on its separators.

    Matching anchors against whole components rather than as substrings
    is what keeps ``"Bear"`` (DE) from firing on ``"Bearsden"`` and
    ``"ON"`` from firing on the English word "on".
    """
    return [part.strip() for part in _COMPONENT_SPLIT_RE.split(loc) if part.strip()]


def _resolve_admin_code(code: str, components: list[str]) -> str:
    """Map a trailing two-letter token to an ISO country code."""
    ambiguous = _AMBIGUOUS_ADMIN_CODES.get(code)
    if ambiguous is not None:
        anchors, anchored, otherwise = ambiguous
        for part in components:
            if part.lower() in anchors or part.upper() in anchors:
                return anchored
        return otherwise
    if code in _COUNTRY_CODES:
        return code
    if code in _US_STATE_CODES:
        return "US"
    if code in _CA_PROVINCE_CODES:
        return "CA"
    return ""


def _country_iso_from_location(loc: object) -> str:
    """Heuristic ISO 3166-1 alpha-2 extraction from a free-form
    ``location`` string. Returns ``""`` when nothing matches.

    Covers the patterns observed across the duplicate-prone EU
    aggregators: full country names in DE/FR/EN, NUTS-region prefixes
    (``DE (DEA58)``), and ``"<City>, <Country>"`` suffixes, plus the
    ``"<City>, <ST>"`` form that North American postings use instead of
    ever naming their country.
    Word-boundary-anchored so ``"Lausanne"`` doesn't get tagged as US
    via a substring match on ``"usa"``.

    A spelled-out country always wins over a trailing two-letter token,
    so ``"Toronto, ON, Canada"`` is Canada and ``"Remote-Friendly,
    United States; San Francisco, CA"`` is the US. A trailing token that
    is simultaneously a country code and a US/Canadian admin code is
    settled by :func:`_resolve_admin_code` rather than being read as a
    country outright — see :data:`_AMBIGUOUS_ADMIN_CODES` for why ``CA``
    and ``DE`` disagree. Anything with no country token at all falls
    through to :func:`looks_us`, which reads the US phrasings this
    module's tables don't ("Remote - US").
    """
    if not isinstance(loc, str) or not loc.strip():
        return ""
    stripped = loc.strip()
    lowered = stripped.lower()
    for code, pat in _COUNTRY_PATTERNS_RE:
        if pat.search(lowered):
            return code
    m = _NUTS_PREFIX_RE.match(stripped)
    if m:
        return m.group(1).upper()
    components = _location_components(stripped)
    m = _TRAILING_ISO_RE.search(stripped)
    if m:
        resolved = _resolve_admin_code(m.group(1).upper(), components)
        if resolved:
            return resolved
    if any(part.lower() in _US_STATE_NAMES for part in components) or looks_us(
        stripped
    ):
        return "US"
    return ""


def _reconciled_country_expr(location: pl.Expr, stored: pl.Expr) -> pl.Expr:
    """A trustworthy ``country_iso``, reconciling the stored value with
    what the location text says.

    The stored value normally wins: a scraper reading a structured
    address field knows more than any heuristic over display text. The
    exception is the codes in :data:`_AMBIGUOUS_ADMIN_CODES`, which are
    simultaneously a country and a US or Canadian subdivision. A source
    that maps a trailing token straight to a country — as this module
    itself once did — files every ``"San Francisco, CA"`` under Canada,
    and the wrong code is worse than none: it is what splits a posting
    and its mirror into different dedup blocks so both survive. When the
    stored code is one of those four and the resolver reads the location
    differently, the resolver wins.

    The Python resolver only runs where the stored value cannot be
    trusted, so the common case stays vectorized.
    """
    normalized = stored.cast(pl.String).fill_null("").str.strip_chars().str.to_uppercase()
    blank = normalized.str.len_bytes() == 0
    ambiguous = normalized.is_in(sorted(_AMBIGUOUS_ADMIN_CODES))
    derived = (
        pl.when(blank | ambiguous)
        .then(location.cast(pl.String))
        .otherwise(pl.lit(None, dtype=pl.String))
        .map_elements(
            _country_iso_from_location, return_dtype=pl.String, skip_nulls=True
        )
        .fill_null("")
    )
    return (
        pl.when(blank)
        .then(derived)
        .when(ambiguous & (derived.str.len_bytes() > 0))
        .then(derived)
        .otherwise(normalized)
    )


# Trailing parenthesised occupational classification that eures
# appends to titles like ``"Anlagenmechaniker (m/w/d) ab 20€/Std.
# (Anlagenmechaniker/in)"``. Bundesagentur ships the same job without
# the trailing tag, so the exact-match dedup misses it. We strip the
# trailing ``(...)`` block **only** when at least one other parens
# block remains in the title — otherwise a clean
# ``"Backend Engineer (m/w/d)"`` would lose its qualifier and stop
# matching the eures-side cleaned version.
_PAREN_GROUP_RE = re.compile(r"\([^()]*\)")
_TRAILING_PARENS_RE = re.compile(r"\s*\([^()]*\)\s*$")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _title_core(title: object) -> str:
    """Normalised title for cross-source dedup.

    Lowercases, collapses whitespace, and (when the title carries
    *multiple* parenthesised blocks) strips the last one — the eures
    "Berufenet code" tag. Single-parens titles like
    ``"Backend Engineer (m/w/d)"`` are returned unchanged (lowercased)
    so they still match the eures version after its trailing tag is
    stripped.
    """
    if not isinstance(title, str) or not title.strip():
        return ""
    stripped = title
    if len(_PAREN_GROUP_RE.findall(title)) >= 2:
        stripped = _TRAILING_PARENS_RE.sub("", title)
    return _WHITESPACE_RUN_RE.sub(" ", stripped.lower()).strip()


@dataclass
class DedupOutcome:
    """What the cross-ATS dedup decided.

    ``survivors`` maps ``ats_value`` → a frame of ``_local_idx`` (the
    source-CSV row indices to keep). ``donations`` maps ``ats_value``
    → a frame of ``_local_idx`` plus ``_donor_``-prefixed pay columns
    to graft onto that row, produced when a gated posting was dropped
    in favour of it.
    """

    survivors: dict[str, pl.DataFrame]
    donations: dict[str, pl.DataFrame] = field(default_factory=dict)


def _dedup_from_per_ats_csvs(
    per_ats_csv_paths: dict[str, Path],
) -> tuple[DedupOutcome, int, int]:
    """Build keys, run cross-ATS dedup, and return per-ATS survivors.

    The keys frame is sunk to a temp parquet via ``sink_parquet``
    (polars streaming write — peak memory bounded by one Arrow batch,
    not the corpus) before we run the eager dedup on it. This is the
    key memory win vs an in-memory ``pl.concat([..]).collect()``: the
    per-ATS scans are pulled in one ATS at a time, and the keys
    parquet on disk is small (~80 MB / million rows for the nine
    thin string columns we project).

    Returns ``(outcome, n_raw, n_kept)``.
    """
    if not per_ats_csv_paths:
        return DedupOutcome(survivors={}), 0, 0

    key_lfs: list[pl.LazyFrame] = []
    for ats_value, csv_path in per_ats_csv_paths.items():
        scan = pl.scan_csv(csv_path, **_SCAN_CSV_KWARGS)
        schema_names = scan.collect_schema().names()
        # ``title_raw`` keeps the original (pre-strip-parens, pre-lower)
        # title so Phase 2 can compare titles with rapidfuzz over the
        # same text a human would compare.
        klf = scan.with_row_index(name="_local_idx").select(
            [
                pl.col("_local_idx").cast(pl.Int64),
                pl.lit(ats_value, dtype=pl.String).alias("ats_type"),
                pl.lit(
                    ATS_DEDUP_PRIORITY.get(ats_value, _DEFAULT_DEDUP_PRIORITY),
                    dtype=pl.Int32,
                ).alias("_priority"),
                _key_col_or_empty(schema_names, "url").alias("url"),
                _key_col_or_empty(schema_names, "title").alias("title_raw"),
                _key_col_or_empty(schema_names, "title")
                .str.to_lowercase()
                .alias("title"),
                _key_col_or_empty(schema_names, "company")
                .str.to_lowercase()
                .alias("company"),
                _key_col_or_empty(schema_names, "location")
                .str.to_lowercase()
                .alias("location"),
                _key_col_or_empty(schema_names, "country_iso")
                .str.to_uppercase()
                .alias("country_iso"),
                _key_col_or_empty(schema_names, "ats_id").alias("ats_id"),
                *_donor_salary_exprs(
                    schema_names, carry=ats_value in GATED_LINK_SOURCES
                ),
            ]
        )
        key_lfs.append(klf)

    keys_chain = (
        pl.concat(key_lfs, how="vertical_relaxed")
        .with_row_index(name="_orig_idx")
        .with_columns(pl.col("_orig_idx").cast(pl.Int64))
    )

    with _temp_file(".parquet") as keys_pq:
        keys_chain.sink_parquet(keys_pq, compression="zstd")
        keys = pl.read_parquet(keys_pq)

    n_raw = keys.height
    outcome = _decide_dedup_survivors_polars(keys)
    n_kept = sum(s.height for s in outcome.survivors.values())
    return outcome, n_raw, n_kept


def _decide_dedup_survivors_polars(
    keys: pl.DataFrame,
    *,
    fuzzy_threshold: int = 90,
    fuzzy_max_block_size: int = 5000,
) -> DedupOutcome:
    """Run the seven-pass cross-ATS dedup.

    Passes 1-3 are exact-match window-function passes (cheap). Pass 4
    is the normalisation pass that catches aggregator formatting
    variations (eures NUTS-code locations vs Bundesagentur full text,
    trailing Berufenet tags on titles). Pass 5 layers rapidfuzz over
    the remaining cross-ATS pairs within ``(company_norm, country_iso)``
    blocks to catch typo / minor-wording dups.

    Passes 6 and 7 handle the gated sources (:data:`GATED_LINK_SOURCES`)
    that passes 1-5 structurally cannot reach: they block on
    ``country_iso``, and the employer-side row often has none (a
    Greenhouse location like ``"San Francisco, CA | New York City,
    NY"`` extracts nothing) while the gated row has a clean ``US``.
    Both drop only gated rows, and only against the employer's own
    board, so no other source's survivorship changes.

    The fuzzy passes are bounded by ``fuzzy_max_block_size`` (default
    5 000 rows per block) so a pathological block — a recruiting agency
    with tens of thousands of postings — doesn't blow up the wall
    clock with n² fuzz calls. Blocks beyond the cap fall through to
    exact-match-only.

    Returns a :class:`DedupOutcome` whose ``survivors`` maps
    ``ats_value`` → polars frame with one column ``_local_idx`` (the
    source-CSV row indices to keep). The streaming Pass 3
    ``semi``-joins each per-ATS scan against that frame.
    """
    if keys.is_empty():
        return DedupOutcome(survivors={})

    work = keys.sort(["_priority", "_orig_idx"])

    # ---- Pass 1: URL exact-match dedup ------------------------------------
    url_keep = (
        (pl.col("url").str.len_bytes() == 0)
        | (pl.col("_orig_idx") == pl.col("_orig_idx").first().over("url"))
    )
    work = work.filter(url_keep)

    # ---- Pass 2: cross-ATS (company, title, location) dedup ---------------
    work = work.with_columns(
        (
            pl.col("company")
            + pl.lit("|")
            + pl.col("title")
            + pl.lit("|")
            + pl.col("location")
        ).alias("_dedup_key")
    )
    ctl_valid = (
        (pl.col("company").str.len_bytes() > 0)
        & (pl.col("title").str.len_bytes() > 0)
    )
    # Only count distinct ats_types AMONG VALID ROWS in each group —
    # invalid (empty c or t) rows must not push a group into "cross-ATS"
    # status.
    n_ats_in_valid_ctl = (
        pl.when(ctl_valid)
        .then(pl.col("ats_type"))
        .otherwise(None)
        .n_unique()
        .over("_dedup_key")
    )
    is_cross_ctl = ctl_valid & (n_ats_in_valid_ctl > 1)
    ctl_keep = ~is_cross_ctl | (
        pl.col("_orig_idx") == pl.col("_orig_idx").first().over("_dedup_key")
    )
    work = work.filter(ctl_keep).drop("_dedup_key")

    # ---- Pass 3: cross-ATS (company_norm, ats_id) dedup -------------------
    work = work.with_columns(_company_norm_expr())
    work = work.with_columns(
        (pl.col("_company_norm") + pl.lit("|") + pl.col("ats_id")).alias("_cid_key")
    )
    cid_valid = (
        (pl.col("_company_norm").str.len_bytes() > 0)
        & (pl.col("ats_id").str.len_bytes() > 0)
    )
    n_ats_in_valid_cid = (
        pl.when(cid_valid)
        .then(pl.col("ats_type"))
        .otherwise(None)
        .n_unique()
        .over("_cid_key")
    )
    is_cross_cid = cid_valid & (n_ats_in_valid_cid > 1)
    cid_keep = ~is_cross_cid | (
        pl.col("_orig_idx") == pl.col("_orig_idx").first().over("_cid_key")
    )
    work = work.filter(cid_keep).drop("_cid_key")

    # ---- Pass 4 (Phase 1): cross-ATS (company_norm, title_core, country) -
    # ``title_core`` strips the trailing parenthesised Berufenet tag
    # that eures appends but Bundesagentur doesn't. ``country_iso`` goes
    # through :func:`_reconciled_country_expr`, which prefers the
    # scraper's structured value but overrules it on the codes that also
    # name a US or Canadian subdivision, and otherwise extracts the
    # country from free-form ``location`` text (eures encodes it as the
    # leading ``DE``/``FR``/… token, Bundesagentur as a full
    # ``", Deutschland"`` suffix). The combination catches
    # formatting-only cross-source dups that Pass 2 misses entirely.
    stored_country = (
        pl.col("country_iso")
        if "country_iso" in work.columns
        else pl.lit("", dtype=pl.String)
    )
    work = work.with_columns(
        pl.col("title_raw")
        .map_elements(_title_core, return_dtype=pl.String)
        .alias("_title_core"),
        _reconciled_country_expr(pl.col("location"), stored_country).alias(
            "_country_iso"
        ),
    )
    work = work.with_columns(
        (
            pl.col("_company_norm")
            + pl.lit("|")
            + pl.col("_title_core")
            + pl.lit("|")
            + pl.col("_country_iso")
        ).alias("_p1_key")
    )
    p1_valid = (
        (pl.col("_company_norm").str.len_bytes() > 0)
        & (pl.col("_title_core").str.len_bytes() > 0)
        & (pl.col("_country_iso").str.len_bytes() > 0)
    )
    n_ats_in_valid_p1 = (
        pl.when(p1_valid)
        .then(pl.col("ats_type"))
        .otherwise(None)
        .n_unique()
        .over("_p1_key")
    )
    is_cross_p1 = p1_valid & (n_ats_in_valid_p1 > 1)
    p1_keep = ~is_cross_p1 | (
        pl.col("_orig_idx") == pl.col("_orig_idx").first().over("_p1_key")
    )
    work = work.filter(p1_keep).drop("_p1_key")

    # ---- Pass 4b: cross-ATS (company_norm, title_core), country-free -----
    # Pass 4 needs the two rows to agree on a country, and a third of the
    # corpus has no country to agree with: Ashby publishes a bare
    # ``"San Francisco"`` and no heuristic turns that into ``US``, while
    # the mirror on another board writes ``"San Francisco, CA, US"``. The
    # pair therefore lands in two different Pass 4 keys and two different
    # Pass 5 blocks, and both rows ship — even when the company matches
    # exactly and the titles are byte-identical.
    #
    # Dropping the country from the key is only safe with the conflict
    # guard: a group collapses when it carries at most one *distinct*
    # non-empty country, so a title that legitimately repeats across
    # countries (one company advertising "Software Engineer" in both
    # London and New York on two boards) is left alone here and handled
    # by Pass 4, which still separates it per country. Blank countries
    # are not a conflict — being unknown is the case this pass exists to
    # serve — so they are dropped before counting.
    work = work.with_columns(
        (pl.col("_company_norm") + pl.lit("|") + pl.col("_title_core")).alias("_ct_key")
    )
    ct_valid = (pl.col("_company_norm").str.len_bytes() > 0) & (
        pl.col("_title_core").str.len_bytes() > 0
    )
    n_ats_in_valid_ct = (
        pl.when(ct_valid)
        .then(pl.col("ats_type"))
        .otherwise(None)
        .n_unique()
        .over("_ct_key")
    )
    n_countries_in_ct = (
        pl.when(ct_valid & (pl.col("_country_iso").str.len_bytes() > 0))
        .then(pl.col("_country_iso"))
        .otherwise(None)
        .drop_nulls()
        .n_unique()
        .over("_ct_key")
    )
    is_cross_ct = ct_valid & (n_ats_in_valid_ct > 1) & (n_countries_in_ct <= 1)
    ct_keep = ~is_cross_ct | (
        pl.col("_orig_idx") == pl.col("_orig_idx").first().over("_ct_key")
    )
    work = work.filter(ct_keep).drop("_ct_key")

    # ---- Pass 5 (Phase 2): fuzzy within (company_norm, country) blocks ---
    drop_orig_idxs = _phase2_fuzzy_drops(
        work,
        threshold=fuzzy_threshold,
        max_block_size=fuzzy_max_block_size,
    )
    if drop_orig_idxs:
        work = work.filter(~pl.col("_orig_idx").is_in(list(drop_orig_idxs)))

    # ---- Pass 6: gated-source demotion, exact and location-free ----------
    # A gated posting loses to the employer's own board whenever the
    # two agree on company and on a punctuation-free title, with no
    # location or country agreement required — the gated source
    # reformats both. Only gated rows are droppable here, which is
    # what makes a location-free key safe: applied corpus-wide it
    # would collapse the aggregator blocks where one title legitimately
    # repeats across hundreds of cities.
    work = work.with_columns(
        _title_slug_expr().alias("_title_slug"),
        pl.col("ats_type").is_in(list(GATED_LINK_SOURCES)).alias("_is_gated"),
    )
    work = work.with_columns(
        (pl.col("_company_norm") + pl.lit("|") + pl.col("_title_slug")).alias(
            "_gated_key"
        )
    )
    gated_valid = (pl.col("_company_norm").str.len_bytes() > 0) & (
        pl.col("_title_slug").str.len_bytes() > 0
    )
    # ``min`` skips nulls, so this is the lowest ``_orig_idx`` among the
    # direct-employer rows of the group — the row that survives and
    # therefore the one that inherits the gated row's pay.
    work = work.with_columns(
        pl.when(
            gated_valid
            & ~pl.col("_is_gated")
            & (pl.col("_priority") == DIRECT_EMPLOYER_PRIORITY)
        )
        .then(pl.col("_orig_idx"))
        .otherwise(None)
        .min()
        .over("_gated_key")
        .alias("_direct_anchor")
    )
    gated_exact_drop = (
        gated_valid & pl.col("_is_gated") & pl.col("_direct_anchor").is_not_null()
    )
    donation_pairs = work.filter(gated_exact_drop).select(
        pl.col("_orig_idx").alias("_donor_orig_idx"),
        pl.col("_direct_anchor").alias("_recipient_orig_idx"),
    )
    work = work.filter(~gated_exact_drop).drop("_direct_anchor", "_gated_key")

    # ---- Pass 7: gated-source demotion, fuzzy within company blocks ------
    gated_fuzzy_drops, fuzzy_pairs = _gated_fuzzy_drops(
        work,
        threshold=fuzzy_threshold,
        max_block_size=fuzzy_max_block_size,
    )
    if gated_fuzzy_drops:
        work = work.filter(~pl.col("_orig_idx").is_in(list(gated_fuzzy_drops)))
        donation_pairs = pl.concat([donation_pairs, fuzzy_pairs])

    donations = _gated_salary_donations(keys, work, donation_pairs)

    work = work.drop(
        "_company_norm", "_title_core", "_country_iso", "_title_slug", "_is_gated"
    )

    # Materialize per-ATS survivor frames keyed on _local_idx for Pass 3
    # of the publish run.
    survivors: dict[str, pl.DataFrame] = {}
    parts = work.partition_by("ats_type", as_dict=True)
    for key_tuple, part in parts.items():
        ats_value = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
        survivors[str(ats_value)] = part.select("_local_idx")
    return DedupOutcome(survivors=survivors, donations=donations)


def _phase2_fuzzy_drops(
    work: pl.DataFrame,
    *,
    threshold: int,
    max_block_size: int,
) -> set[int]:
    """Within each ``company_norm`` block, greedily drop cross-ATS rows
    whose title fuzz-matches a higher-priority row's title at
    ``token_sort_ratio >= threshold``.

    Greedy, sorted by ``(_priority, _orig_idx)``: each new row is
    compared against every already-kept row from a *different* ATS.
    Same-ATS rows pass through (we never dedup within an ATS — that's
    the publisher's per-ATS-slice contract).

    Two rows are only comparable when their countries do not conflict:
    equal, or unknown on at least one side. Blocking on the company
    alone and testing the country per pair is what lets a row with no
    resolvable country reach its mirror — a third of the corpus has
    none, and keying the block on country exempted all of it from this
    pass. A company whose block exceeds ``max_block_size`` falls back to
    per-country sub-blocks rather than being skipped outright, so the
    recruiting agencies with tens of thousands of postings keep the
    dedup they had while staying clear of an n² sweep.

    Skips blocks with an empty company (no signal), blocks where every
    row shares one ATS (no cross-source dup possible), and sub-blocks
    that are still oversize after the fallback.

    Returns the set of ``_orig_idx`` to drop.
    """
    from rapidfuzz import fuzz, utils

    drop: set[int] = set()
    company_groups = work.filter(
        pl.col("_company_norm").str.len_bytes() > 0
    ).group_by(["_company_norm"], maintain_order=True)

    for _, company_block in company_groups:
        if company_block.height < 2:
            continue
        if company_block["ats_type"].n_unique() < 2:
            continue
        if company_block.height > max_block_size:
            logger.warning(
                "Phase-2 fuzzy dedup: company block oversize (%d rows, "
                "company=%s); falling back to per-country blocks.",
                company_block.height,
                company_block.row(0, named=True)["_company_norm"],
            )
            blocks = [
                part
                for part in company_block.partition_by("_country_iso")
                if part.height >= 2 and part["ats_type"].n_unique() >= 2
            ]
        else:
            blocks = [company_block]

        for block in blocks:
            if block.height > max_block_size:
                logger.warning(
                    "Phase-2 fuzzy dedup: skipping oversize block "
                    "(%d rows, company=%s country=%s).",
                    block.height,
                    block.row(0, named=True)["_company_norm"],
                    block.row(0, named=True)["_country_iso"],
                )
                continue
            _phase2_scan_block(
                block.sort(["_priority", "_orig_idx"]),
                drop=drop,
                threshold=threshold,
                fuzz=fuzz,
                processor=utils.default_process,
            )

    return drop


def _phase2_scan_block(
    block: pl.DataFrame,
    *,
    drop: set[int],
    threshold: int,
    fuzz: Any,
    processor: Any,
) -> None:
    """Greedy cross-ATS title sweep over one pre-sorted block."""
    # ``kept`` is a list of (title_raw, ats_type, country) tuples seen so
    # far; for each new row we check fuzz against every kept row from a
    # DIFFERENT ATS whose country does not contradict it.
    kept: list[tuple[str, str, str]] = []
    for row in block.iter_rows(named=True):
        title_raw = row["title_raw"]
        ats = row["ats_type"]
        country = row["_country_iso"]
        orig_idx = row["_orig_idx"]
        if not title_raw:
            continue
        matched = False
        for kept_title, kept_ats, kept_country in kept:
            if kept_ats == ats:
                continue
            if country and kept_country and country != kept_country:
                continue
            # ``default_process`` lowercases and drops punctuation before
            # tokenising. Without it the comparison is done on raw display
            # strings, where cosmetic differences alone sink the score
            # below the threshold: "Senior Cloud Engineer" vs "senior
            # cloud engineer" scores 86, and "Lead, Enterprise" vs
            # "Lead (Enterprise)" scores 69.
            #
            # ``token_sort_ratio`` and not ``token_set_ratio``: the set
            # variant scores the *intersection* of the two token sets, so
            # tokens present on only one side barely count. That is the
            # wrong reading for a job title, where the extra qualifier is
            # the whole difference — "Senior Staff Software Engineer (Node
            # Infrastructure)" against "Staff Software Engineer, Data
            # Infrastructure" scores 94 under the set variant and 82 once
            # the tokens must line up.
            if (
                fuzz.token_sort_ratio(title_raw, kept_title, processor=processor)
                >= threshold
            ):
                matched = True
                break
        if matched:
            drop.add(int(orig_idx))
        else:
            kept.append((title_raw, ats, country))


def _gated_fuzzy_drops(
    work: pl.DataFrame,
    *,
    threshold: int,
    max_block_size: int,
) -> tuple[set[int], pl.DataFrame]:
    """Drop gated rows whose title fuzz-matches an employer row's,
    blocking on ``company_norm`` alone.

    Pass 5 already does a fuzzy sweep, but it blocks on
    ``(company_norm, country_iso)`` and the employer-side row often
    has no extractable country, so the pair never lands in the same
    block. Dropping the country from the key is only safe because
    this pass is one-directional: gated rows are the only candidates
    for removal, and only ``DIRECT_EMPLOYER_PRIORITY`` rows count as
    anchors.

    Matching runs ``token_sort_ratio`` over ``_title_slug`` at
    ``threshold + _GATED_FUZZY_MARGIN`` — see that constant for why it
    is stricter than Pass 5. An abbreviation the employer spells out
    (``Applied AI`` vs ``Applied Artificial Intelligence``) falls under
    the bar and keeps both rows, which is the safe way to fail: a
    wrong pair would delete a posting and misattribute its pay.

    The scan is restricted to companies that actually have a gated
    row, which is what keeps it cheap — a few thousand companies
    rather than every employer in the corpus.

    Returns ``(orig_idxs_to_drop, pairs)`` where ``pairs`` maps each
    dropped row to the anchor that replaced it, so its pay can follow.
    """
    from rapidfuzz import fuzz

    cutoff = threshold + _GATED_FUZZY_MARGIN
    empty_pairs = pl.DataFrame(
        schema={"_donor_orig_idx": pl.Int64, "_recipient_orig_idx": pl.Int64}
    )
    named = pl.col("_company_norm").str.len_bytes() > 0
    gated_companies = (
        work.filter(pl.col("_is_gated") & named).select("_company_norm").unique()
    )
    if gated_companies.is_empty():
        return set(), empty_pairs

    scope = work.filter(
        named
        & (pl.col("_is_gated") | (pl.col("_priority") == DIRECT_EMPLOYER_PRIORITY))
    ).join(gated_companies, on="_company_norm", how="semi")

    drop: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, block in scope.group_by(["_company_norm"], maintain_order=True):
        gated_rows = block.filter(pl.col("_is_gated"))
        if gated_rows.is_empty():
            continue
        anchor_rows = block.filter(~pl.col("_is_gated")).sort("_orig_idx")
        if anchor_rows.is_empty():
            continue
        if block.height > max_block_size:
            logger.warning(
                "Gated-source fuzzy dedup: skipping oversize block "
                "(%d rows, company=%s).",
                block.height,
                block.row(0, named=True)["_company_norm"],
            )
            continue
        anchors = [
            (row["_title_slug"], int(row["_orig_idx"]))
            for row in anchor_rows.iter_rows(named=True)
            if row["_title_slug"]
        ]
        if not anchors:
            continue
        for row in gated_rows.iter_rows(named=True):
            title_slug = row["_title_slug"]
            if not title_slug:
                continue
            for anchor_title, anchor_idx in anchors:
                if fuzz.token_sort_ratio(title_slug, anchor_title) >= cutoff:
                    drop.add(int(row["_orig_idx"]))
                    pairs.append((int(row["_orig_idx"]), anchor_idx))
                    break

    if not pairs:
        return drop, empty_pairs
    return drop, pl.DataFrame(pairs, schema=empty_pairs.schema, orient="row")


def _gated_salary_donations(
    keys: pl.DataFrame,
    survivors: pl.DataFrame,
    pairs: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Route the pay of every dropped gated row to the row that
    replaced it.

    Welcome to the Jungle is one of the few sources that publishes
    salary, and the employer boards that outrank it usually publish
    none, so a plain drop would trade an openable link for a missing
    number. Recipients come from two places: the pairs recorded by
    passes 6 and 7, and — for gated rows that passes 1-5 dropped,
    which record no pair — an exact ``(company_norm, title_slug)``
    lookup against the survivors.

    Returns ``ats_value`` → frame of ``_local_idx`` plus the
    ``_donor_``-prefixed pay columns.
    """
    donor_columns = [f"{_DONOR_PREFIX}{name}" for name, _ in _SALARY_CARRY_COLUMNS]
    if any(column not in keys.columns for column in donor_columns):
        return {}

    priced = pl.col(f"{_DONOR_PREFIX}salary_min").is_not_null() | pl.col(
        f"{_DONOR_PREFIX}salary_max"
    ).is_not_null()
    donors = (
        keys.filter(pl.col("ats_type").is_in(list(GATED_LINK_SOURCES)) & priced)
        .join(survivors.select("_orig_idx"), on="_orig_idx", how="anti")
        .drop("_local_idx")
    )
    if donors.is_empty():
        return {}

    recipients = survivors.filter(
        ~pl.col("_is_gated") & (pl.col("_priority") == DIRECT_EMPLOYER_PRIORITY)
    )
    if recipients.is_empty():
        return {}

    donors = donors.with_columns(
        _company_norm_expr(),
        _title_slug_expr().alias("_title_slug"),
    )
    by_orig_idx = recipients.select(
        pl.col("_orig_idx").alias("_recipient_orig_idx"),
        pl.col("ats_type").alias("_pair_ats"),
        pl.col("_local_idx").alias("_pair_local_idx"),
    )
    # ``survivors`` is still ordered by ``(_priority, _orig_idx)``, so
    # the first row of a slug group is the best recipient in it.
    by_slug = recipients.group_by(
        ["_company_norm", "_title_slug"], maintain_order=True
    ).agg(
        pl.col("ats_type").first().alias("_slug_ats"),
        pl.col("_local_idx").first().alias("_slug_local_idx"),
    )

    donors = (
        donors.join(
            pairs.unique(subset=["_donor_orig_idx"], keep="first").rename(
                {"_donor_orig_idx": "_orig_idx"}
            ),
            on="_orig_idx",
            how="left",
        )
        .join(by_orig_idx, on="_recipient_orig_idx", how="left")
        .join(by_slug, on=["_company_norm", "_title_slug"], how="left")
        .with_columns(
            pl.coalesce("_pair_ats", "_slug_ats").alias("_recipient_ats"),
            pl.coalesce("_pair_local_idx", "_slug_local_idx").alias("_local_idx"),
        )
        .filter(pl.col("_recipient_ats").is_not_null())
    )
    if donors.is_empty():
        return {}

    # One donation per recipient — a row can only carry one salary, so
    # the earliest donor wins and the rest are discarded.
    donors = donors.sort("_orig_idx").unique(
        subset=["_recipient_ats", "_local_idx"], keep="first", maintain_order=True
    )

    selected = donors.select(
        "_recipient_ats",
        "_local_idx",
        *donor_columns,
        pl.col("ats_type").alias(f"{_DONOR_PREFIX}{SALARY_SOURCE_COLUMN}"),
    )
    donations: dict[str, pl.DataFrame] = {}
    for key_tuple, part in selected.partition_by(
        "_recipient_ats", as_dict=True
    ).items():
        ats_value = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
        donations[str(ats_value)] = part.drop("_recipient_ats")
    return donations


# --- helpers ---------------------------------------------------------------


def _apply_salary_donations(
    lf: pl.LazyFrame, donation: pl.DataFrame
) -> pl.LazyFrame:
    """Fill a row's pay from the gated posting that was dropped for it.

    Only rows that publish no salary of their own are touched, and all
    five pay columns move together so a min never ends up beside
    another posting's currency. :data:`SALARY_SOURCE_COLUMN` records
    where a filled figure came from and stays null everywhere else.

    The donation frame holds one row per recipient, so the ``left``
    join keeps the scan streaming and can't duplicate rows.
    """
    lf = lf.join(donation.lazy(), on="_local_idx", how="left")
    schema_names = lf.collect_schema().names()

    def own(name: str, dtype: pl.DataType) -> pl.Expr:
        if name in schema_names:
            return pl.col(name).cast(dtype, strict=False)
        return pl.lit(None, dtype=dtype)

    own_min = own("salary_min", pl.Float64())
    own_max = own("salary_max", pl.Float64())
    donor_min = pl.col(f"{_DONOR_PREFIX}salary_min")
    donor_max = pl.col(f"{_DONOR_PREFIX}salary_max")
    donated = (
        own_min.is_null()
        & own_max.is_null()
        & (donor_min.is_not_null() | donor_max.is_not_null())
    )

    filled = [
        pl.when(donated)
        .then(pl.col(f"{_DONOR_PREFIX}{name}"))
        .otherwise(own(name, dtype))
        .alias(name)
        for name, dtype in _SALARY_CARRY_COLUMNS
    ]
    filled.append(
        pl.when(donated)
        .then(pl.col(f"{_DONOR_PREFIX}{SALARY_SOURCE_COLUMN}"))
        .otherwise(own(SALARY_SOURCE_COLUMN, pl.String()))
        .alias(SALARY_SOURCE_COLUMN)
    )
    return lf.with_columns(filled).drop(
        [column for column in donation.columns if column.startswith(_DONOR_PREFIX)]
    )


# Column carrying the reason a row was rejected. Present only on the
# quarantine artifact — the published slices never ship it.
QUARANTINE_REASON_COLUMN = "quarantine_reason"

# How far past the publish time a ``posted_at`` may sit before the row
# is treated as corrupt rather than merely fresh. Publisher clocks,
# source timezones, and same-day scheduling all produce legitimately
# "future" stamps measured in hours; a genuine publication date is
# never days ahead. The audit found rows dated 2027 and 2028 leading a
# "past week" search on a July 2026 snapshot.
FUTURE_POSTED_AT_GRACE = timedelta(days=2)


def _future_posted_at_expr(now: datetime) -> pl.Expr:
    """True where ``posted_at`` is too far ahead to be a real date.

    ``posted_at`` arrives as text from the per-ATS CSV, so it is parsed
    here rather than compared directly. An unparseable value yields null
    and is left alone: this gate is for dates that are readable and
    wrong, not for malformed ones.
    """
    parsed = (
        pl.col("posted_at")
        .cast(pl.String)
        .str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
    )
    return parsed.is_not_null() & (parsed > pl.lit(now + FUTURE_POSTED_AT_GRACE))


# Largest credible pay at each period, in USD-equivalent units. These
# are outlier bounds, not market estimates: the point is to reject a
# figure no posting could carry, such as the $8.6M/year nurse the audit
# found, without touching a genuine executive package.
_MAX_CREDIBLE_USD_BY_PERIOD: dict[str, float] = {
    "HOUR": 2_000.0,
    "DAY": 10_000.0,
    "WEEK": 50_000.0,
    "MONTH": 200_000.0,
    "YEAR": 5_000_000.0,
}

# Rough units per USD, used to scale the bounds above. Without this a
# flat cap would delete real postings wholesale: ¥6,000,000/year is an
# ordinary Japanese salary, and ₩60,000,000 an ordinary Korean one.
# Order of magnitude is all that matters for an outlier test, so these
# do not need to track live exchange rates.
_UNITS_PER_USD: dict[str, float] = {
    "JPY": 150.0, "KRW": 1_300.0, "IDR": 15_000.0, "VND": 25_000.0,
    "INR": 85.0, "HUF": 360.0, "CLP": 950.0, "COP": 4_000.0,
    "NGN": 1_500.0, "KES": 130.0, "EGP": 50.0, "PHP": 58.0,
    "THB": 35.0, "TWD": 32.0, "RSD": 108.0, "UAH": 41.0,
    "ARS": 1_000.0, "MXN": 18.0, "ZAR": 18.0, "TRY": 40.0,
    "CZK": 23.0, "SEK": 11.0, "NOK": 11.0, "DKK": 7.0,
    "PLN": 4.0, "RON": 4.6, "BGN": 1.8, "ISK": 138.0,
    "BRL": 5.5, "PEN": 3.8, "UYU": 40.0, "MAD": 10.0,
    "CNY": 7.2, "HKD": 7.8, "MYR": 4.7, "ILS": 3.7,
    "SAR": 3.75, "AED": 3.67,
}

# Period wording that contradicts a declared ``salary_period``. Matched
# against ``salary_summary`` — the human-readable string the ATS itself
# rendered, which is the better witness when the two disagree.
_SUMMARY_PERIOD_PATTERNS: dict[str, str] = {
    "HOUR": r"(?i)(?:per\s+hour|hourly|/\s*hr\b|/\s*hour|\ban\s+hour\b)",
    "DAY": r"(?i)(?:per\s+day|daily|/\s*day|per\s+diem)",
    "WEEK": r"(?i)(?:per\s+week|weekly|/\s*week)",
    "MONTH": r"(?i)(?:per\s+month|monthly|/\s*month|\ba\s+month\b)",
    "YEAR": r"(?i)(?:per\s+year|per\s+annum|annually|annual|yearly|/\s*yr\b|/\s*year)",
}


def _implausible_salary_expr(schema_names: list[str]) -> pl.Expr:
    """True where the pay is too large to be real for its own period.

    Bounds are scaled by currency so the test means the same thing in
    every denomination; a row with no period can't be judged and is
    left alone. Only an upper bound is applied — a floor would delete
    genuine postings in low-wage economies far sooner than it would
    catch a bad one.
    """
    scale = (
        pl.col("salary_currency")
        .cast(pl.String)
        .str.to_uppercase()
        .replace_strict(_UNITS_PER_USD, default=1.0, return_dtype=pl.Float64)
    )
    ceiling = (
        pl.col("salary_period")
        .cast(pl.String)
        .str.to_uppercase()
        .replace_strict(
            _MAX_CREDIBLE_USD_BY_PERIOD, default=None, return_dtype=pl.Float64
        )
    ) * scale
    amounts = [
        pl.col(column).cast(pl.Float64)
        for column in ("salary_min", "salary_max")
        if column in schema_names
    ]
    amount = pl.max_horizontal(amounts) if len(amounts) > 1 else amounts[0]
    return ceiling.is_not_null() & amount.is_not_null() & (amount > ceiling)


def _contradicted_period_expr() -> pl.Expr:
    """True where ``salary_period`` disagrees with ``salary_summary``.

    A row claiming ``YEAR`` while its own summary reads "$65.00 per
    hour" has one of the two wrong, and the amount is being read on the
    strength of whichever it is. Rather than pick a winner we drop the
    row — this is the pairing behind the audit's $8.6M nurse.

    A summary naming both periods ("£50,000 per annum, £25 per hour")
    is not a contradiction: it agrees with the declared one and merely
    adds context.
    """
    declared = pl.col("salary_period").cast(pl.String).str.to_uppercase()
    summary = pl.col("salary_summary").cast(pl.String)
    mentions = {
        period: summary.str.contains(pattern)
        for period, pattern in _SUMMARY_PERIOD_PATTERNS.items()
    }
    contradiction = pl.lit(value=False)
    for period in _SUMMARY_PERIOD_PATTERNS:
        others = pl.any_horizontal(
            [expr for other, expr in mentions.items() if other != period]
        )
        contradiction = contradiction | (
            (declared == period) & ~mentions[period] & others
        )
    return summary.is_not_null() & declared.is_not_null() & contradiction


def _quarantine_reason_expr(schema_names: list[str], *, now: datetime) -> pl.Expr | None:
    """Reason this row must not be published, or null to keep it.

    Returns ``None`` when the slice has no column any check can read, so
    callers can skip the partition entirely.
    """
    checks: list[tuple[pl.Expr, str]] = []
    if "posted_at" in schema_names:
        checks.append((_future_posted_at_expr(now), "future_posted_at"))
    if {"salary_period", "salary_currency"} <= set(schema_names) and (
        {"salary_min", "salary_max"} & set(schema_names)
    ):
        checks.append((_implausible_salary_expr(schema_names), "implausible_salary"))
    if {"salary_period", "salary_summary"} <= set(schema_names):
        checks.append((_contradicted_period_expr(), "contradicted_salary_period"))
    if "company" in schema_names:
        checks.append((_placeholder_company_expr(), "placeholder_employer"))
    if not checks:
        return None
    expr = pl.lit(None, dtype=pl.String)
    for condition, reason in reversed(checks):
        expr = pl.when(condition).then(pl.lit(reason, dtype=pl.String)).otherwise(expr)
    return expr


def _partition_quarantine(
    lf: pl.LazyFrame, schema_names: list[str], *, now: datetime
) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    """Split a slice into rows to publish and rows to quarantine.

    Rejected rows are dropped from the dataset outright rather than
    flagged, but they are written to a sidecar so a bad rule shows up as
    a spike in the quarantine count instead of silently deleting data.
    """
    reason = _quarantine_reason_expr(schema_names, now=now)
    if reason is None:
        return lf, None
    tagged = lf.with_columns(reason.alias(QUARANTINE_REASON_COLUMN))
    kept = tagged.filter(pl.col(QUARANTINE_REASON_COLUMN).is_null()).drop(
        QUARANTINE_REASON_COLUMN
    )
    rejected = tagged.filter(pl.col(QUARANTINE_REASON_COLUMN).is_not_null())
    return kept, rejected


def _enrich_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``is_remote`` / ``salary_min`` / ``salary_max`` / ``country_iso``
    columns when they aren't already present on the input.

    Implemented as polars expressions whenever possible so the lazy
    chain stays streamable through ``sink_csv``. ``is_remote`` reads
    the ``REMOTE_KEYWORDS`` and ``ONSITE_KEYWORDS`` lists from the
    canonical :mod:`ats_scrapers.enrichment.derived` module — both are
    optional, so a deploy that has narrowed the heuristic to title-only
    (no ``ONSITE_KEYWORDS`` exported) still gets a usable column.

    ``salary_summary`` and ``country_iso`` parsing both go through a
    Python callback (``map_elements``); polars' streaming engine
    doesn't run user functions, so the lazy chain falls back to the
    eager engine for an ATS slice that needs either. The country
    extractor was already used internally for Pass 1 dedup
    (``_country_iso_from_location``); exposing it as a public column
    means downstream consumers (D1 sync, R2 parquet readers) can
    filter / facet by ISO code without re-parsing the location string.
    """
    schema_names = lf.collect_schema().names()

    if "title" in schema_names and "is_remote" not in schema_names:
        lf = lf.with_columns(_is_remote_expr().alias("is_remote"))

    if "title" in schema_names:
        if "seniority" not in schema_names:
            lf = lf.with_columns(_seniority_expr().alias("seniority"))
        else:
            # An ATS that ships its own level is more authoritative than
            # the title, so only fill the gaps it leaves.
            current = pl.col("seniority").cast(pl.String)
            lf = lf.with_columns(
                pl.when(
                    current.is_null() | (current.str.strip_chars().str.len_bytes() == 0)
                )
                .then(_seniority_expr())
                .otherwise(current)
                .alias("seniority")
            )

    if "salary_summary" in schema_names and "salary_min" not in schema_names:
        salary_struct = pl.Struct({"min": pl.Float64, "max": pl.Float64})
        lf = (
            lf.with_columns(
                pl.col("salary_summary")
                .map_elements(_safe_parse_salary, return_dtype=salary_struct)
                .alias("_salary_parsed")
            )
            .with_columns(
                pl.col("_salary_parsed").struct.field("min").alias("salary_min"),
                pl.col("_salary_parsed").struct.field("max").alias("salary_max"),
            )
            .drop("_salary_parsed")
        )

    if "description" in schema_names:
        lf = _backfill_salary_from_description(lf, schema_names)

    if "location" in schema_names:
        if "country_iso" not in schema_names:
            lf = lf.with_columns(
                pl.col("location")
                .map_elements(_country_iso_from_location, return_dtype=pl.String)
                .alias("country_iso")
            )
        else:
            # Blank values get filled, and a stored code that names a US
            # or Canadian subdivision as well as a country gets checked
            # against the location text — see
            # :func:`_reconciled_country_expr`. Publishing the reconciled
            # value (rather than only using it for dedup keys) is what
            # stops a consumer filtering on ``country_iso = 'US'`` from
            # losing every California posting a source filed under ``CA``.
            lf = lf.with_columns(
                _reconciled_country_expr(
                    pl.col("location"), pl.col("country_iso")
                ).alias("country_iso")
            )

    return lf


# Cheap pre-filter for the description salary parse. Running the full
# parser over every description would mean a Python callback across ~4M
# bodies of up to 25k chars; this vectorized test skips the ~95% that
# mention no money at all, leaving the callback to the plausible rows.
_PAY_HINT_PATTERN = (
    r"(?i)(?:salary|compensation|pay range|pay scale|hiring range|"
    r"hourly rate|base pay|target earnings)"
)

_SALARY_BLOCK_STRUCT = pl.Struct({
    "min": pl.Float64,
    "max": pl.Float64,
    "currency": pl.String,
    "period": pl.String,
})


def _safe_parse_salary_block(text: object) -> dict[str, object | None]:
    block = parse_salary_block(text)
    if block is None:
        return {"min": None, "max": None, "currency": None, "period": None}
    return {
        "min": block.min_amount,
        "max": block.max_amount,
        "currency": block.currency,
        "period": block.period,
    }


def _backfill_salary_from_description(
    lf: pl.LazyFrame, schema_names: list[str]
) -> pl.LazyFrame:
    """Fill missing pay columns from the disclosed range in the body.

    Most ATSes expose no structured salary field, which is why only
    2.31% of the published corpus showed any compensation — while
    pay-transparency law had already put the number in the description
    of a large share of those same postings. Existing values are never
    overwritten: a range the ATS stated outright always beats one
    recovered from prose.
    """
    existing = [c for c in ("salary_min", "salary_max") if c in schema_names]
    needs_pay = pl.lit(True)
    for column in existing:
        needs_pay = needs_pay & pl.col(column).cast(pl.Float64).is_null()

    candidate = (
        pl.when(needs_pay & pl.col("description").str.contains(_PAY_HINT_PATTERN))
        .then(pl.col("description"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    lf = lf.with_columns(
        candidate.map_elements(
            _safe_parse_salary_block,
            return_dtype=_SALARY_BLOCK_STRUCT,
            skip_nulls=True,
        ).alias("_pay_block")
    )
    for column, struct_field, dtype in (
        ("salary_min", "min", pl.Float64),
        ("salary_max", "max", pl.Float64),
        ("salary_currency", "currency", pl.String),
        ("salary_period", "period", pl.String),
    ):
        derived = pl.col("_pay_block").struct.field(struct_field).cast(dtype)
        if column in schema_names:
            lf = lf.with_columns(
                pl.coalesce(pl.col(column).cast(dtype), derived).alias(column)
            )
        else:
            lf = lf.with_columns(derived.alias(column))
    return lf.drop("_pay_block")


def _is_remote_expr() -> pl.Expr:
    """Vectorized polars version of :func:`infer_is_remote`.

    Reads ``title`` (not ``location``) — the canonical heuristic
    in :mod:`ats_scrapers.enrichment.derived` is intentionally narrow and
    only treats title-level remote markers as definitive. Free-form
    location text is left for the downstream LLM enrichment pipeline.

    Falls back to the eager ``map_elements`` callback when the deploy
    ships a stripped variant of ``derived.py`` that doesn't export
    ``REMOTE_KEYWORDS`` — the publisher stays usable, but that branch
    breaks lazy streaming for the slice that needs it.
    """
    if not _REMOTE_KEYWORDS:
        return (
            pl.col("title")
            .map_elements(infer_is_remote, return_dtype=pl.Boolean)
        )

    title_lower = (
        pl.col("title").cast(pl.String, strict=False).str.to_lowercase()
    )
    remote_match: pl.Expr = pl.lit(False)
    for kw in _REMOTE_KEYWORDS:
        remote_match = remote_match | title_lower.str.contains(kw, literal=True)
    # Narrow heuristic — never returns False; absence of a remote
    # marker in the title is not evidence the role is on-site.
    return pl.when(remote_match).then(pl.lit(True)).otherwise(None)


def _seniority_expr() -> pl.Expr:
    """Vectorized polars version of :func:`infer_seniority`.

    Builds one ``when``/``then`` branch per rule in precedence order, so
    the rules stay declared once in
    :mod:`ats_scrapers.enrichment.derived` and this only compiles them.
    Staying native matters here: a Python callback over ~4M titles would
    drop the whole slice off the streaming engine.

    Falls back to the callback when a stripped ``derived.py`` ships
    without the rules, matching :func:`_is_remote_expr`.
    """
    if not _SENIORITY_RULES:
        return pl.col("title").map_elements(infer_seniority, return_dtype=pl.String)

    title = pl.col("title").cast(pl.String, strict=False)
    chain = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for rule in _SENIORITY_RULES:
        matched = title.str.contains(rule.pattern)
        if rule.exclude:
            matched = matched & ~title.str.contains(rule.exclude)
        chain = chain.when(matched).then(pl.lit(rule.level, dtype=pl.String))
    return chain.otherwise(pl.lit(None, dtype=pl.String))


def _safe_parse_salary(value: object) -> dict[str, float | None]:
    if not isinstance(value, str):
        return {"min": None, "max": None}
    mn, mx = parse_salary_range(value)
    return {"min": mn, "max": mx}


def _collect_uploaded_keys(entry: dict[str, object]) -> list[str]:
    keys: list[str] = []
    for field_name in ("csv", "parquet"):
        value = entry.get(field_name)
        if isinstance(value, str):
            keys.append(value)
    return keys


def _sum_by_ats_companies_rows(manifest: dict[str, object]) -> int:
    """Sum ``rows`` across every ``by_ats_companies.<ats>`` entry.

    Companies are CI-owned, so the publisher derives ``total_companies``
    from whatever the CI most recently wrote — fallback to 0 when the
    CI hasn't run yet."""
    block = manifest.get("by_ats_companies")
    if not isinstance(block, dict):
        return 0
    total = 0
    for entry in block.values():
        if isinstance(entry, dict):
            rows = entry.get("rows")
            if isinstance(rows, int):
                total += rows
    return total


def _guard_suspicious_empty_job_slices(
    *,
    source_dir: Path,
    ats_csv_pattern: str,
    existing_manifest: dict[str, object],
) -> None:
    """Block publishes that would replace known-good provider data with empty.

    A header-only ``<ats>/jobs.csv`` is valid CSV, so the streaming publisher
    can otherwise upload it and patch the manifest to ``rows: 0``. Treat that
    as suspicious when the existing manifest proves the provider either had
    jobs before or still has tenants in ``by_ats_companies``.
    """
    if os.getenv("ATS_SCRAPERS_ALLOW_EMPTY_PUBLISH"):
        return

    by_ats = existing_manifest.get("by_ats")
    if not isinstance(by_ats, dict):
        by_ats = {}
    by_ats_companies = existing_manifest.get("by_ats_companies")
    if not isinstance(by_ats_companies, dict):
        by_ats_companies = {}

    suspicious: list[str] = []
    for ats in ATSType:
        if ats is ATSType.CUSTOM:
            continue
        source_path = source_dir / ats_csv_pattern.format(ats=ats.value)
        if not source_path.exists():
            continue

        previous_rows = _entry_rows(by_ats.get(ats.value))
        company_rows = _entry_rows(by_ats_companies.get(ats.value))
        has_prior_data = previous_rows > 0 or company_rows > 0
        if source_path.stat().st_size == 0:
            if has_prior_data:
                suspicious.append(
                    f"{ats.value}: local jobs.csv is 0 bytes; "
                    f"manifest previously had {previous_rows} jobs and "
                    f"{company_rows} companies. Suggested action: retry the "
                    "provider scrape or keep the previous published data."
                )
            continue
        if _csv_data_row_count(source_path) != 0:
            continue

        if has_prior_data:
            suspicious.append(
                f"{ats.value}: local jobs.csv has 0 rows; "
                f"manifest previously had {previous_rows} jobs and "
                f"{company_rows} companies. Suggested action: retry the "
                "provider scrape or keep the previous published data."
            )

    if suspicious:
        raise StorageError(
            "Refusing to publish suspicious empty provider slices. "
            "Set ATS_SCRAPERS_ALLOW_EMPTY_PUBLISH=1 only for intentional empty "
            "providers.\n- "
            + "\n- ".join(suspicious)
        )


def _entry_rows(entry: object) -> int:
    if not isinstance(entry, dict):
        return 0
    rows = entry.get("rows")
    return rows if isinstance(rows, int) and rows > 0 else 0


def _csv_data_row_count(path: Path) -> int:
    with path.open("rb") as f:
        lines = sum(1 for _ in f)
    return max(lines - 1, 0)


def _load_existing_manifest(r2_client: R2Client, key: str) -> dict[str, object]:
    """Best-effort fetch of an existing manifest. On any failure (missing
    object, malformed JSON) return an empty dict so the publisher
    proceeds with a fresh manifest rather than crashing the run."""
    try:
        body = r2_client.get_bytes(key)
    except StorageError as exc:
        logger.warning("Could not read existing manifest %s: %s", key, exc)
        return {}
    if not body:
        return {}
    try:
        loaded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Existing manifest %s did not parse as JSON (%s); starting fresh",
            key,
            exc,
        )
        return {}
    if not isinstance(loaded, dict):
        logger.warning(
            "Existing manifest %s root is not an object; starting fresh", key
        )
        return {}
    return loaded


@contextmanager
def _temp_file(suffix: str) -> Iterator[Path]:
    """Context manager yielding a temp file path that is unlinked on exit."""
    fd, path_str = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(path_str)
    try:
        yield path
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def _merge_parquets_streaming(input_paths: list[Path], out_path: Path) -> None:
    """Merge multiple parquet files into one via polars lazy concat.

    Different ATSes have heterogeneous schemas — same column name,
    different inferred dtype (an int64 ``ats_id`` on one ATS, a
    large_string on another) — and ``pyarrow.unify_schemas`` refuses
    to reconcile those. Polars' ``how="diagonal_relaxed"`` promotes
    conflicting dtypes to the wider one (string wins), then
    ``sink_parquet`` writes the unified result without buffering the
    full corpus.
    """
    if not input_paths:
        return
    lfs = [pl.scan_parquet(str(p)) for p in input_paths]
    pl.concat(lfs, how="diagonal_relaxed").sink_parquet(
        out_path, compression="zstd"
    )


def _csv_has_rows(path: Path) -> bool:
    """Whether a sunk CSV holds anything beyond its header.

    ``sink_csv`` always writes the header, so an empty quarantine slice
    is a non-zero file. Checking here keeps header-only temp files out
    of the concat, which would otherwise widen the schema for nothing.
    """
    try:
        return (
            pl.scan_csv(path, **_SCAN_CSV_KWARGS).select(pl.len()).collect().item() > 0
        )
    except pl.exceptions.NoDataError:
        return False


def _file_sha_size(path: Path) -> tuple[str, int]:
    """Stream-hash a file and return ``(sha256_hex, size_bytes)``."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size

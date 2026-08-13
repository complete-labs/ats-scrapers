"""Detail metadata stored beside cached descriptions.

Workday's country, posted date and time type only exist on the per-job
detail endpoint. The pipeline fetches that endpoint once and caches the
body, so without carrying the other fields through the same cache they
would only ever be set for newly-seen listings — on a board where most
rows are cache hits, that means almost never.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ats_scrapers.models import ATSType, Job

_SPEC = importlib.util.spec_from_file_location(
    "run_pipeline_for_tests",
    Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _job(ats_id: str = "1") -> Job:
    return Job(
        url=f"https://acme.wd1.myworkdayjobs.com/ext/job/{ats_id}",
        title="Engineer",
        company="Acme",
        ats_type=ATSType.WORKDAY,
        ats_id=ats_id,
    )


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "descriptions.sqlite3"


def test_metadata_round_trips(cache_path: Path) -> None:
    cache = runner.DescriptionCache(path=cache_path)
    try:
        cache.set(_job(), "the body", {"country_iso": "US", "region": "North America"})
        description, metadata = cache.get_with_metadata(_job())
    finally:
        cache.close()
    assert description == "the body"
    assert metadata == {"country_iso": "US", "region": "North America"}


def test_metadata_survives_a_description_only_rewrite(cache_path: Path) -> None:
    """Writing back a longer body must not wipe the fields beside it."""
    cache = runner.DescriptionCache(path=cache_path)
    try:
        cache.set(_job(), "short", {"country_iso": "DE"})
        cache.set(_job(), "a much longer body")
        description, metadata = cache.get_with_metadata(_job())
    finally:
        cache.close()
    assert description == "a much longer body"
    assert metadata == {"country_iso": "DE"}


def test_missing_metadata_reads_as_empty(cache_path: Path) -> None:
    cache = runner.DescriptionCache(path=cache_path)
    try:
        cache.set(_job(), "body only")
        assert cache.get_with_metadata(_job()) == ("body only", {})
        assert cache.get(_job()) == "body only"
    finally:
        cache.close()


def test_compressed_cache_round_trips_metadata(cache_path: Path) -> None:
    cache = runner.DescriptionCache(path=cache_path, compress=True)
    try:
        cache.set(_job(), "body", {"country_iso": "FR", "is_remote": True})
        assert cache.get_with_metadata(_job()) == (
            "body",
            {"country_iso": "FR", "is_remote": True},
        )
    finally:
        cache.close()


def test_upgrades_a_v2_cache_in_place(cache_path: Path) -> None:
    """A v2 file must be migrated, not rejected.

    The production Workday cache holds ~700k descriptions; rebuilding it
    would mean re-crawling the detail endpoint for every one.
    """
    conn = sqlite3.connect(cache_path)
    conn.execute(
        """
        CREATE TABLE descriptions (
            kind TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            description BLOB NOT NULL,
            PRIMARY KEY (kind, cache_key)
        )
        """
    )
    conn.execute("PRAGMA user_version = 2")
    for kind, key in runner._description_keys(_job("legacy")):
        conn.execute(
            "INSERT INTO descriptions (kind, cache_key, description) VALUES (?, ?, ?)",
            (kind, key, b"legacy body"),
        )
    conn.commit()
    conn.close()

    cache = runner.DescriptionCache(path=cache_path)
    try:
        # The pre-existing row survives and reads back normally.
        assert cache.get_with_metadata(_job("legacy")) == ("legacy body", {})
        # And the upgraded file accepts metadata going forward.
        cache.set(_job("fresh"), "new body", {"country_iso": "GB"})
        assert cache.get_with_metadata(_job("fresh"))[1] == {"country_iso": "GB"}
    finally:
        cache.close()

    conn = sqlite3.connect(cache_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    conn.close()


def test_apply_detail_fields_only_fills_blanks() -> None:
    job = _job()
    job.country_iso = "US"
    runner._apply_detail_fields(
        job, {"country_iso": "DE", "region": "Europe", "is_remote": True}
    )
    # A value the listing already supplied is more current than the cache.
    assert job.country_iso == "US"
    assert job.region == "Europe"
    assert job.is_remote is True


def test_apply_detail_fields_ignores_unknown_attributes() -> None:
    job = _job()
    runner._apply_detail_fields(job, {"title": "Overwritten", "nonsense": 1})
    assert job.title == "Engineer"


def test_apply_detail_fields_accepts_datetimes() -> None:
    job = _job()
    posted = datetime(2026, 8, 13, tzinfo=UTC)
    runner._apply_detail_fields(job, {"posted_at": posted})
    assert job.posted_at == posted


def test_posted_at_survives_the_json_round_trip(cache_path: Path) -> None:
    """``Job`` does not validate on assignment, so a datetime read back as
    an ISO string would otherwise be written through unconverted and blow
    up when the CSV writer calls ``.isoformat()``."""
    posted = datetime(2026, 8, 13, tzinfo=UTC)
    cache = runner.DescriptionCache(path=cache_path)
    try:
        cache.set(_job(), "body", {"posted_at": posted, "country_iso": "US"})
        _, metadata = cache.get_with_metadata(_job())
    finally:
        cache.close()
    assert isinstance(metadata["posted_at"], str)  # JSON has no datetime

    job = _job()
    runner._apply_detail_fields(job, metadata)
    assert job.posted_at == posted
    assert job.posted_at.isoformat().startswith("2026-08-13")

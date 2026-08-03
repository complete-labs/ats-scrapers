"""Tests for the dataset publisher.

Covers the v2.0 layout:

    jobhive/v1/manifest.json
    jobhive/v1/all.parquet
    jobhive/v1/<ats>/jobs.{csv,parquet}

The publisher owns jobs entries in ``manifest.json``. Companies (top-level
``companies`` block + per-ATS ``by_ats_companies``) are written by the CI
workflow — the publisher must read-modify-write so those entries survive
each run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json

import pandas as pd
import pytest

from ats_scrapers.exceptions import StorageError
from pipeline.publisher import (
    CACHE_CONTROL_LATEST,
    DEFAULT_PREFIX,
    DatasetPublisher,
)

# --- Layout -----------------------------------------------------------------


def test_publish_writes_per_ats_and_full_snapshot(ats_csv_dir, fake_r2) -> None:
    """``all.{csv,parquet}`` live at the top level; per-ATS slices
    ship CSV+parquet under ``<ats>/jobs.{csv,parquet}``."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir)

    assert result.total_jobs == 9
    assert result.ats_count == 3
    assert "jobhive/v1/manifest.json" in fake_r2.uploads
    assert "jobhive/v1/all.parquet" in fake_r2.uploads
    assert "jobhive/v1/all.csv" in fake_r2.uploads
    for ats in ("greenhouse", "lever", "ashby"):
        assert f"jobhive/v1/{ats}/jobs.csv" in fake_r2.uploads
        assert f"jobhive/v1/{ats}/jobs.parquet" in fake_r2.uploads


def test_publish_keeps_hosted_source_without_scraper_adapter(
    ats_csv_dir, fake_r2
) -> None:
    beisen_dir = ats_csv_dir / "beisen"
    beisen_dir.mkdir()
    pd.DataFrame(
        [
            {
                "url": "https://example.com/beisen/1",
                "title": "Engineer",
                "company": "Acme",
                "ats_id": "beisen-1",
            }
        ]
    ).to_csv(beisen_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])

    assert result.ats_count == 4
    assert "beisen" in manifest["by_ats"]


def test_publisher_does_not_write_companies_anywhere(ats_csv_dir, fake_r2) -> None:
    """Companies are CI-owned. The publisher must never write them."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    company_keys = [k for k in fake_r2.uploads if "companies" in k]
    assert company_keys == []


def test_publisher_does_not_write_by_date(ats_csv_dir, fake_r2) -> None:
    """The v1 by-date paths are gone in v2."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    bydate_keys = [k for k in fake_r2.uploads if "/by-date/" in k]
    assert bydate_keys == []


# --- Salary recovery from descriptions ---------------------------------------


def test_publish_recovers_pay_from_the_description_body(tmp_path, fake_r2) -> None:
    """Audit finding 01: only 2.31% of postings showed any pay.

    Most ATSes expose no salary field, but pay-transparency law already
    put the range in the description of a large share of them.
    """
    ats_dir = tmp_path / "greenhouse"
    ats_dir.mkdir()
    pd.DataFrame(
        [
            {
                "url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
                "ats_id": "1",
                "description": (
                    "<p>Build things.</p><div class='title'>Annual Salary:</div>"
                    "<div><span>$265,000</span><span>&mdash;</span>"
                    "<span>$365,000 USD</span></div>"
                ),
            },
            {
                "url": "https://gh.com/2", "title": "Designer", "company": "Acme",
                "ats_id": "2", "description": "<p>No money mentioned at all.</p>",
            },
        ]
    ).to_csv(ats_dir / "jobs.csv", index=False)

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(tmp_path)
    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    ).sort_values("ats_id")

    priced, unpriced = published.iloc[0], published.iloc[1]
    assert priced["salary_min"] == 265000.0
    assert priced["salary_max"] == 365000.0
    assert priced["salary_currency"] == "USD"
    assert priced["salary_period"] == "YEAR"
    assert pd.isna(unpriced["salary_min"])
    assert pd.isna(unpriced["salary_currency"])


def test_structured_salary_is_never_overwritten_by_the_body(
    tmp_path, fake_r2
) -> None:
    """A range the ATS stated outright outranks one parsed from prose."""
    ats_dir = tmp_path / "ashby"
    ats_dir.mkdir()
    pd.DataFrame(
        [
            {
                "url": "https://ashby.com/1", "title": "Engineer",
                "company": "Acme", "ats_id": "1",
                "salary_min": 100000.0, "salary_max": 120000.0,
                "description": "Annual Salary: $265,000 — $365,000 USD",
            }
        ]
    ).to_csv(ats_dir / "jobs.csv", index=False)

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(tmp_path)
    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )

    assert published["salary_min"].tolist() == [100000.0]
    assert published["salary_max"].tolist() == [120000.0]


# --- Quality gates / quarantine ---------------------------------------------


def _dir_with_posted_at(tmp_path, rows):
    ats_dir = tmp_path / "greenhouse"
    ats_dir.mkdir()
    pd.DataFrame(rows).to_csv(ats_dir / "jobs.csv", index=False)
    return tmp_path


def test_publish_drops_future_dated_postings(tmp_path, fake_r2) -> None:
    """Audit finding 05: 2027/2028 rows led a "past week" search.

    A publication date cannot be in the future, so the row is removed
    from every published artifact rather than merely flagged.
    """
    source = _dir_with_posted_at(
        tmp_path,
        [
            {"url": "https://gh.com/1", "title": "Real", "company": "Acme",
             "ats_id": "1", "posted_at": "2026-07-20T00:00:00Z"},
            {"url": "https://gh.com/2", "title": "Future", "company": "Acme",
             "ats_id": "2", "posted_at": "2028-07-15T00:00:00Z"},
            {"url": "https://gh.com/3", "title": "Future", "company": "Acme",
             "ats_id": "3", "posted_at": "2027-06-30T00:00:00Z"},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 1
    published = pd.read_csv(
        io.BytesIO(fake_r2.uploads["jobhive/v1/greenhouse/jobs.csv"]["data"])
    )
    assert published["ats_id"].tolist() == [1]


def test_quarantine_artifact_records_what_was_dropped(tmp_path, fake_r2) -> None:
    source = _dir_with_posted_at(
        tmp_path,
        [
            {"url": "https://gh.com/1", "title": "Real", "company": "Acme",
             "ats_id": "1", "posted_at": "2026-07-20T00:00:00Z"},
            {"url": "https://gh.com/2", "title": "Future", "company": "Acme",
             "ats_id": "2", "posted_at": "2028-07-15T00:00:00Z"},
        ],
    )
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(source)

    assert "jobhive/v1/quarantine.parquet" in fake_r2.uploads
    dropped = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/quarantine.parquet"]["data"])
    )
    assert dropped["ats_id"].tolist() == [2]
    assert dropped["quarantine_reason"].tolist() == ["future_posted_at"]

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["stats"]["total_jobs_quarantined"] == 1


def _dir_with_salary(tmp_path, rows):
    ats_dir = tmp_path / "workday"
    ats_dir.mkdir()
    base = {"salary_min": None, "salary_max": None, "salary_currency": None,
            "salary_period": None, "salary_summary": None}
    pd.DataFrame([base | r for r in rows]).to_csv(
        ats_dir / "jobs.csv", index=False
    )
    return tmp_path


def test_publish_drops_salaries_contradicted_by_their_own_summary(
    tmp_path, fake_r2
) -> None:
    """Audit finding 03: an hourly nurse listed at $8.6M/year.

    When ``salary_period`` and ``salary_summary`` disagree, the amount
    is being read on the strength of whichever one is wrong.
    """
    source = _dir_with_salary(
        tmp_path,
        [
            {"url": "https://wd.com/1", "title": "Nurse", "company": "Hosp",
             "ats_id": "1", "salary_min": 135200.0, "salary_max": 135200.0,
             "salary_currency": "USD", "salary_period": "YEAR",
             "salary_summary": "$65.00 per hour"},
            {"url": "https://wd.com/2", "title": "Engineer", "company": "Acme",
             "ats_id": "2", "salary_min": 250000.0, "salary_max": 400000.0,
             "salary_currency": "USD", "salary_period": "YEAR",
             "salary_summary": "$250,000 - $400,000 per year"},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 1
    dropped = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/quarantine.parquet"]["data"])
    )
    assert dropped["quarantine_reason"].tolist() == ["contradicted_salary_period"]


def test_publish_drops_amounts_too_large_for_their_period(tmp_path, fake_r2) -> None:
    source = _dir_with_salary(
        tmp_path,
        [
            {"url": "https://wd.com/1", "title": "Engineer", "company": "Acme",
             "ats_id": "1", "salary_min": 8_600_000.0, "salary_max": 8_600_000.0,
             "salary_currency": "USD", "salary_period": "YEAR"},
            {"url": "https://wd.com/2", "title": "Barista", "company": "Cafe",
             "ats_id": "2", "salary_min": 45_000.0, "salary_max": 45_000.0,
             "salary_currency": "USD", "salary_period": "HOUR"},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 0
    dropped = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/quarantine.parquet"]["data"])
    )
    assert set(dropped["quarantine_reason"]) == {"implausible_salary"}


def test_plausibility_bounds_scale_with_the_currency(tmp_path, fake_r2) -> None:
    """A flat cap would delete real postings wholesale.

    ¥6,000,000/year is an ordinary Japanese salary and ₩60,000,000 an
    ordinary Korean one; both exceed any USD-denominated bound.
    """
    source = _dir_with_salary(
        tmp_path,
        [
            {"url": "https://wd.com/1", "title": "Engineer", "company": "Acme JP",
             "ats_id": "1", "salary_min": 6_000_000.0, "salary_max": 9_000_000.0,
             "salary_currency": "JPY", "salary_period": "YEAR"},
            {"url": "https://wd.com/2", "title": "Engineer", "company": "Acme KR",
             "ats_id": "2", "salary_min": 60_000_000.0, "salary_max": 90_000_000.0,
             "salary_currency": "KRW", "salary_period": "YEAR"},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 2
    assert "jobhive/v1/quarantine.parquet" not in fake_r2.uploads


def test_a_summary_naming_two_periods_is_not_a_contradiction(
    tmp_path, fake_r2
) -> None:
    """"per annum ... per hour equivalent" agrees; it just adds context."""
    source = _dir_with_salary(
        tmp_path,
        [
            {"url": "https://wd.com/1", "title": "Nurse", "company": "Hosp",
             "ats_id": "1", "salary_min": 50_000.0, "salary_max": 50_000.0,
             "salary_currency": "GBP", "salary_period": "YEAR",
             "salary_summary": "£50,000 per annum (£25 per hour equivalent)"},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 1


def test_no_quarantine_artifact_when_everything_is_clean(
    ats_csv_dir, fake_r2
) -> None:
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(ats_csv_dir)

    assert "jobhive/v1/quarantine.parquet" not in fake_r2.uploads
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["stats"]["total_jobs_quarantined"] == 0


def test_published_slices_never_carry_the_reason_column(tmp_path, fake_r2) -> None:
    source = _dir_with_posted_at(
        tmp_path,
        [
            {"url": "https://gh.com/1", "title": "Real", "company": "Acme",
             "ats_id": "1", "posted_at": "2026-07-20T00:00:00Z"},
            {"url": "https://gh.com/2", "title": "Future", "company": "Acme",
             "ats_id": "2", "posted_at": "2028-07-15T00:00:00Z"},
        ],
    )
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(source)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert "quarantine_reason" not in manifest["stats"]["schema_columns"]
    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert "quarantine_reason" not in published.columns


def test_same_day_and_unparseable_dates_survive(tmp_path, fake_r2) -> None:
    """The gate targets corrupt dates, not merely fresh or messy ones.

    Source timezones and same-day scheduling put legitimate postings a
    few hours ahead of the publish clock.
    """
    from datetime import UTC, datetime, timedelta

    soon = (datetime.now(UTC) + timedelta(hours=6)).isoformat()
    source = _dir_with_posted_at(
        tmp_path,
        [
            {"url": "https://gh.com/1", "title": "Soon", "company": "Acme",
             "ats_id": "1", "posted_at": soon},
            {"url": "https://gh.com/2", "title": "Garbage", "company": "Acme",
             "ats_id": "2", "posted_at": "not-a-date"},
            {"url": "https://gh.com/3", "title": "Absent", "company": "Acme",
             "ats_id": "3", "posted_at": ""},
        ],
    )
    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 3
    assert "jobhive/v1/quarantine.parquet" not in fake_r2.uploads


# --- Manifest ---------------------------------------------------------------


def test_manifest_contains_expected_structure(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["version"] == "2.0"
    assert manifest["stats"]["total_jobs"] == 9
    assert manifest["stats"]["ats_count"] == 3
    assert "greenhouse" in manifest["by_ats"]
    assert manifest["by_ats"]["greenhouse"]["rows"] == 3
    # `all` lives at the top level now and ships both formats.
    assert manifest["all"]["parquet"].endswith("/all.parquet")
    assert manifest["all"]["csv"].endswith("/all.csv")


def test_manifest_includes_generator_string(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["generator"].startswith("ats-scrapers/")


def test_manifest_records_sha256_per_file(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    gh = manifest["by_ats"]["greenhouse"]
    assert "sha256" in gh
    assert len(gh["sha256"]) == 64
    assert "parquet_sha256" in gh
    assert len(gh["parquet_sha256"]) == 64


def test_manifest_uses_public_urls_when_base_set(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["all"]["parquet"].startswith("https://cdn.example.com/")
    assert manifest["by_ats"]["greenhouse"]["csv"].startswith(
        "https://cdn.example.com/"
    )


def test_manifest_falls_back_to_keys_when_no_public_url(
    ats_csv_dir, fake_r2_no_public
) -> None:
    publisher = DatasetPublisher(fake_r2_no_public, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(
        fake_r2_no_public.uploads["jobhive/v1/manifest.json"]["data"]
    )
    assert manifest["all"]["parquet"] == "jobhive/v1/all.parquet"
    assert manifest["by_ats"]["greenhouse"]["csv"] == "jobhive/v1/greenhouse/jobs.csv"


def test_manifest_includes_schema_version_and_columns(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["stats"]["schema_version"] == "2.0"
    assert "schema_columns" in manifest["stats"]


def test_publisher_derives_country_iso_column(ats_csv_dir, fake_r2) -> None:
    gh_csv = ats_csv_dir / "greenhouse" / "jobs.csv"
    df = pd.read_csv(gh_csv)
    df.loc[df["location"] == "Paris", "location"] = "Paris, France"
    df["country_iso"] = ""
    df.to_csv(gh_csv, index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert "country_iso" in manifest["stats"]["schema_columns"]

    csv_text = fake_r2.uploads["jobhive/v1/greenhouse/jobs.csv"]["data"].decode()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    france_rows = [row for row in rows if row["location"] == "Paris, France"]
    assert france_rows
    assert {row["country_iso"] for row in france_rows} == {"FR"}


def test_country_derivation_skips_callback_when_column_is_complete(monkeypatch) -> None:
    import pipeline.publisher as publisher_module

    frame = publisher_module.pl.DataFrame({
        "location": ["Paris, France"],
        "country_iso": ["FR"],
    }).lazy()

    def fail_if_called(_location):
        raise AssertionError("country callback should stay on the fast path")

    monkeypatch.setattr(
        publisher_module, "_country_iso_from_location", fail_if_called,
    )

    result = publisher_module._enrich_lazy(frame).collect()
    assert result["country_iso"].to_list() == ["FR"]


# --- Manifest patch (read-modify-write) -------------------------------------


def test_manifest_patch_preserves_companies_block(ats_csv_dir, fake_r2) -> None:
    """If the CI has previously uploaded a manifest with companies +
    by_ats_companies, the publisher must NOT clobber those keys."""
    pre_existing = {
        "version": "2.0",
        "companies": {
            "csv": "https://cdn.example.com/jobhive/v1/companies.csv",
            "parquet": "https://cdn.example.com/jobhive/v1/companies.parquet",
            "rows": 76627,
            "size_bytes": 4_990_244,
            "sha256": "a" * 64,
        },
        "by_ats_companies": {
            "greenhouse": {
                "csv": "https://cdn.example.com/jobhive/v1/greenhouse/companies.csv",
                "rows": 3076,
                "size_bytes": 176749,
                "sha256": "b" * 64,
            },
        },
        "updated_at": "2026-05-08T17:00:00Z",
    }
    fake_r2.upload_bytes(
        json.dumps(pre_existing).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["companies"] == pre_existing["companies"]
    assert manifest["by_ats_companies"] == pre_existing["by_ats_companies"]
    # And jobs entries got refreshed.
    assert manifest["by_ats"]["greenhouse"]["rows"] == 3
    assert manifest["all"]["parquet"].endswith("/all.parquet")


def test_manifest_patch_drops_legacy_fields(ats_csv_dir, fake_r2) -> None:
    """Pre-2.0 manifests carried `by_date` and `companies_by_ats`. Their
    underlying objects are deleted by `prune_legacy_paths`, so the
    manifest entries must be dropped too — leaving them would point
    consumers at 404s."""
    pre_existing = {
        "version": "1.0",
        "by_date": {"2026-05-03": {"parquet": "...", "rows": 50, "size_bytes": 60}},
        "companies_by_ats": {
            "greenhouse": {"csv": "...legacy...", "rows": 1, "size_bytes": 1}
        },
    }
    fake_r2.upload_bytes(
        json.dumps(pre_existing).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert "by_date" not in manifest
    assert "companies_by_ats" not in manifest


def test_manifest_patch_handles_missing_existing_manifest(ats_csv_dir, fake_r2) -> None:
    """First-ever publish has no manifest to read. Empty companies
    block in result is fine — the CI will fill it on next run."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["version"] == "2.0"
    assert "companies" not in manifest
    assert "by_ats_companies" not in manifest
    # Without a CI-written companies block, total_companies falls back
    # to 0 (instead of being absent — the published library 0.1.0
    # requires the field).
    assert manifest["stats"]["total_companies"] == 0


def test_total_companies_sums_by_ats_companies_rows(ats_csv_dir, fake_r2) -> None:
    """``stats.total_companies`` is derived from the CI's
    ``by_ats_companies`` block on every publish (not from a separate
    derivation), so the field stays fresh without re-uploading."""
    pre_existing = {
        "by_ats_companies": {
            "greenhouse": {"csv": "...", "rows": 3076, "size_bytes": 1, "sha256": "x" * 64},
            "lever": {"csv": "...", "rows": 1830, "size_bytes": 1, "sha256": "y" * 64},
            "ashby": {"csv": "...", "rows": 2058, "size_bytes": 1, "sha256": "z" * 64},
        },
    }
    fake_r2.upload_bytes(
        json.dumps(pre_existing).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["stats"]["total_companies"] == 3076 + 1830 + 2058


def test_manifest_patch_handles_corrupt_existing_manifest(
    ats_csv_dir, fake_r2, caplog
) -> None:
    """A non-JSON or non-object manifest must not crash the publish —
    we log a warning and proceed with a fresh manifest."""
    fake_r2.upload_bytes(
        b"<html>oops</html>",
        "jobhive/v1/manifest.json",
        content_type="text/html",
    )
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    with caplog.at_level("WARNING"):
        publisher.publish_from_directory(ats_csv_dir)
    assert "did not parse as JSON" in caplog.text
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["version"] == "2.0"


# --- Cache headers ----------------------------------------------------------


def test_cache_control_short_for_latest_files(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert (
        fake_r2.uploads["jobhive/v1/all.parquet"]["cache_control"]
        == CACHE_CONTROL_LATEST
    )
    assert (
        fake_r2.uploads["jobhive/v1/manifest.json"]["cache_control"]
        == CACHE_CONTROL_LATEST
    )
    assert (
        fake_r2.uploads["jobhive/v1/greenhouse/jobs.csv"]["cache_control"]
        == CACHE_CONTROL_LATEST
    )


def test_per_ats_csv_content_type(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert (
        fake_r2.uploads["jobhive/v1/greenhouse/jobs.csv"]["content_type"]
        == "text/csv"
    )


def test_parquet_content_type(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert (
        fake_r2.uploads["jobhive/v1/all.parquet"]["content_type"]
        == "application/vnd.apache.parquet"
    )


def test_manifest_content_type(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert (
        fake_r2.uploads["jobhive/v1/manifest.json"]["content_type"]
        == "application/json"
    )


# --- Ordering ---------------------------------------------------------------


def test_manifest_uploaded_after_data_files(ats_csv_dir, fake_r2) -> None:
    """The manifest must be uploaded last — a half-finished publish must
    never expose a manifest pointing at missing files."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    keys = [
        k
        for k in fake_r2.uploads
        # Ignore the pre-existing manifest seeded by other tests'
        # paths through this fixture.
        if not k.endswith("/manifest.json")
    ]
    assert "jobhive/v1/manifest.json" in fake_r2.uploads
    last_manifest_index = max(
        i
        for i, k in enumerate(fake_r2.uploads)
        if k == "jobhive/v1/manifest.json"
    )
    last_data_index = max(
        i for i, k in enumerate(fake_r2.uploads) if k in keys
    )
    assert last_manifest_index > last_data_index


# --- Legacy cleanup ---------------------------------------------------------


def test_legacy_paths_pruned(ats_csv_dir, fake_r2) -> None:
    """Legacy (v1) keys must be removed when the publisher runs."""
    legacy_keys = [
        "jobhive/v1/jobs/all.parquet",
        "jobhive/v1/jobs/by-ats/greenhouse.csv",
        "jobhive/v1/jobs/by-ats/greenhouse.parquet",
        "jobhive/v1/jobs/by-date/2026-05-03.parquet",
        "jobhive/v1/companies/all.csv",
        "jobhive/v1/companies/by-ats/greenhouse.csv",
    ]
    for k in legacy_keys:
        fake_r2.upload_bytes(b"legacy", k, content_type="text/plain")

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)

    for k in legacy_keys:
        assert k not in fake_r2.uploads, f"legacy key still present: {k}"
        assert k in fake_r2.deleted, f"legacy key never marked deleted: {k}"


def test_prune_legacy_paths_is_idempotent(ats_csv_dir, fake_r2) -> None:
    """Running prune twice must not error and must report 0 the second
    time (nothing to delete)."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    second_pass = publisher.prune_legacy_paths()
    assert second_pass == 0


# --- Custom prefix ----------------------------------------------------------


def test_custom_prefix_appears_in_all_keys(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, prefix="custom/v2", write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert all(k.startswith("custom/v2/") for k in fake_r2.uploads)


def test_default_prefix_is_jobhive_v1() -> None:
    assert DEFAULT_PREFIX == "jobhive/v1"


def test_prefix_strips_redundant_slashes(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, prefix="/foo/bar/", write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir)
    assert "foo/bar/manifest.json" in fake_r2.uploads


# --- Error paths ------------------------------------------------------------


def test_publish_raises_when_no_csvs_present(tmp_path, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    with pytest.raises(StorageError):
        publisher.publish_from_directory(tmp_path)


def test_publish_refuses_suspicious_empty_provider_slice(tmp_path, fake_r2) -> None:
    gh_dir = tmp_path / "greenhouse"
    gh_dir.mkdir()
    (gh_dir / "jobs.csv").write_text(
        "url,title,company,ats_type,ats_id,location,is_remote,salary_min,"
        "salary_max,salary_currency,salary_period,salary_summary,"
        "employment_type,department,team,description,posted_at,"
        "requisition_id,apply_url,commitment,raw\n",
        encoding="utf-8",
    )
    fake_r2.upload_bytes(
        json.dumps(
            {
                "version": "2.0",
                "by_ats": {"greenhouse": {"rows": 123, "size_bytes": 100}},
                "by_ats_companies": {"greenhouse": {"rows": 5, "size_bytes": 50}},
            }
        ).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    with pytest.raises(StorageError, match="Refusing to publish suspicious empty"):
        publisher.publish_from_directory(tmp_path)
    assert "jobhive/v1/greenhouse/jobs.csv" not in fake_r2.uploads


def test_publish_refuses_zero_byte_provider_slice_with_prior_manifest(
    tmp_path, fake_r2
) -> None:
    gh_dir = tmp_path / "greenhouse"
    gh_dir.mkdir()
    (gh_dir / "jobs.csv").write_bytes(b"")
    fake_r2.upload_bytes(
        json.dumps(
            {
                "version": "2.0",
                "by_ats": {"greenhouse": {"rows": 123, "size_bytes": 100}},
                "by_ats_companies": {"greenhouse": {"rows": 5, "size_bytes": 50}},
            }
        ).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    with pytest.raises(StorageError, match=r"local jobs\.csv is 0 bytes"):
        publisher.publish_from_directory(tmp_path)
    assert "jobhive/v1/greenhouse/jobs.csv" not in fake_r2.uploads


def test_publish_reuses_manifest_loaded_for_empty_slice_guard(
    ats_csv_dir, fake_r2
) -> None:
    fake_r2.upload_bytes(
        json.dumps(
            {
                "version": "2.0",
                "by_ats_companies": {"greenhouse": {"rows": 5}},
            }
        ).encode("utf-8"),
        "jobhive/v1/manifest.json",
        content_type="application/json",
    )
    fetched: list[str] = []
    real_get_bytes = fake_r2.get_bytes

    def counted_get_bytes(key: str):
        fetched.append(key)
        return real_get_bytes(key)

    fake_r2.get_bytes = counted_get_bytes

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(ats_csv_dir)

    assert fetched.count("jobhive/v1/manifest.json") == 1


def test_publish_without_pyarrow_raises(monkeypatch, fake_r2) -> None:
    """When write_parquet=True but pyarrow is missing, we fail fast."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(StorageError, match="pyarrow"):
        DatasetPublisher(fake_r2, write_parquet=True)


# --- Result object ----------------------------------------------------------


def test_result_reports_counts_and_duration(ats_csv_dir, fake_r2) -> None:
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir)
    assert result.total_jobs == 9
    assert result.total_jobs_raw == 9  # no cross-ATS dups in this fixture
    assert result.ats_count == 3
    assert result.duration_seconds >= 0.0
    assert result.manifest_key == "jobhive/v1/manifest.json"
    # 3 ATS slices × 2 formats + all.{csv,parquet} + manifest.json = 9 files
    assert len(result.files) == 9


# --- Cross-ATS deduplication ------------------------------------------------


def test_cross_ats_dedup_collapses_mirror_listings(
    ats_csv_dir_with_duplicates, fake_r2
) -> None:
    """When the same (company, title, location) appears under multiple ATSes,
    the global ``all`` snapshot keeps one row. Per-ATS slices stay raw so
    consumers querying a single ATS see what that ATS actually exposes."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir_with_duplicates)

    assert result.total_jobs == 3
    assert result.total_jobs_raw == 6
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["stats"]["total_jobs"] == 3
    assert manifest["stats"]["total_jobs_raw"] == 6
    # Per-ATS slices are NOT deduped — both still report 3 rows.
    assert manifest["by_ats"]["workday"]["rows"] == 3
    assert manifest["by_ats"]["eightfold"]["rows"] == 3


def test_cross_ats_dedup_keeps_higher_priority_ats(
    ats_csv_dir_with_duplicates, fake_r2
) -> None:
    """Workday is priority 1; Eightfold is priority 5. The deduped ``all``
    snapshot must keep the Workday rows."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir_with_duplicates)

    all_parquet = fake_r2.uploads["jobhive/v1/all.parquet"]["data"]
    df = pd.read_parquet(pd.io.common.BytesIO(all_parquet))
    assert len(df) == 3
    assert (df["ats_type"] == "workday").all()
    assert df["url"].str.contains("workday.com").all()


def test_cross_ats_dedup_prefers_structured_country_iso(tmp_path) -> None:
    import pipeline.publisher as publisher_module

    workday_path = tmp_path / "workday.csv"
    jobbank_path = tmp_path / "jobbankca.csv"
    pd.DataFrame([
        {
            "url": "https://workday.example/job/1",
            "title": "Backend Engineer",
            "company": "Acme",
            "location": "Toronto, Ontario",
            "country_iso": "CA",
            "ats_id": "wd-1",
        },
    ]).to_csv(workday_path, index=False)
    pd.DataFrame([
        {
            "url": "https://jobbank.example/job/1",
            "title": "Backend Engineer",
            "company": "Acme",
            "location": "Toronto (ON)",
            "country_iso": "CA",
            "ats_id": "jb-1",
        },
    ]).to_csv(jobbank_path, index=False)

    outcome, raw_count, kept_count = (
        publisher_module._dedup_from_per_ats_csvs(
            {"workday": workday_path, "jobbankca": jobbank_path}
        )
    )

    assert raw_count == 2
    assert kept_count == 1
    assert outcome.survivors["workday"].height == 1
    assert "jobbankca" not in outcome.survivors


# --- Gated-source demotion (Pass 6 / Pass 7) --------------------------------


@pytest.fixture
def ats_csv_dir_gated_link(tmp_path):
    """Welcome to the Jungle mirrors an employer's Greenhouse board.

    Nothing about the pair lines up for passes 1-5: wttj turns the
    employer's ``"Title, Qualifier"`` into ``"Title (Qualifier)"``
    (which fuzz-matches at only 80, under the 90 threshold), writes a
    single office where Greenhouse lists several, and the Greenhouse
    location yields no ``country_iso`` at all — so the country-blocked
    passes never compare the two. Pass 6's punctuation-free,
    location-free key is what pairs them.

    wttj also prices both of its rows; Greenhouse prices neither,
    which is the usual split and the reason the pay has to follow the
    dropped row.
    """
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([
        {
            "url": "https://www.welcometothejungle.com/en/companies/C5Jui5/jobs/sse",
            "title": "Staff Software Engineer (Environments Infrastructure)",
            "company": "Anthropic",
            "location": "New York, United States",
            "ats_id": "wttj-1",
            "salary_min": 320000.0,
            "salary_max": 485000.0,
            "salary_currency": "USD",
            "salary_period": "YEAR",
            "salary_summary": "USD 320,000 - 485,000",
        },
        {
            "url": "https://www.welcometothejungle.com/en/companies/C5Jui5/jobs/cm",
            "title": "Community Manager (EMEA)",
            "company": "Anthropic",
            "location": "London, United Kingdom",
            "ats_id": "wttj-2",
            "salary_min": 90000.0,
            "salary_max": 120000.0,
            "salary_currency": "GBP",
            "salary_period": "YEAR",
            "salary_summary": "GBP 90,000 - 120,000",
        },
    ]).to_csv(wttj_dir / "jobs.csv", index=False)

    greenhouse_dir = tmp_path / "greenhouse"
    greenhouse_dir.mkdir()
    pd.DataFrame([
        {
            "url": "https://job-boards.greenhouse.io/anthropic/jobs/5101378008",
            "title": "Staff Software Engineer, Environments Infrastructure",
            "company": "anthropic",
            "location": "San Francisco, CA | New York City, NY",
            "ats_id": "gh-1",
        },
    ]).to_csv(greenhouse_dir / "jobs.csv", index=False)

    return tmp_path


def test_gated_source_loses_the_link_to_the_employer_board(
    ats_csv_dir_gated_link, fake_r2
):
    """The published link must be one a signed-out reader can open, so
    the wttj row goes and the Greenhouse row stays. The wttj posting
    with no employer twin survives untouched."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir_gated_link)

    assert result.total_jobs_raw == 3
    assert result.total_jobs == 2

    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    staff = df[df["title"].str.startswith("Staff Software Engineer")]
    assert len(staff) == 1
    assert staff.iloc[0]["ats_type"] == "greenhouse"
    assert "job-boards.greenhouse.io" in staff.iloc[0]["url"]
    assert (df["ats_type"] == "welcometothejungle").sum() == 1

    # Per-ATS slices stay raw — the wttj slice still ships both rows.
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["by_ats"]["welcometothejungle"]["rows"] == 2


def test_gated_source_donates_its_salary_to_the_surviving_row(
    ats_csv_dir_gated_link, fake_r2
):
    """Greenhouse publishes no pay; wttj does. Dropping the gated row
    must not drop the number with it, and the whole pay block travels
    together so the figure keeps its currency and period."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(ats_csv_dir_gated_link)

    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    staff = df[df["ats_type"] == "greenhouse"].iloc[0]
    assert staff["salary_min"] == 320000.0
    assert staff["salary_max"] == 485000.0
    assert staff["salary_currency"] == "USD"
    assert staff["salary_period"] == "YEAR"
    assert staff["salary_source"] == "welcometothejungle"

    # A row that kept its own pay carries no attribution.
    survivor = df[df["ats_type"] == "welcometothejungle"].iloc[0]
    assert survivor["salary_min"] == 90000.0
    assert pd.isna(survivor["salary_source"])

    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert "salary_source" in manifest["stats"]["schema_columns"]


def test_employer_row_keeps_its_own_salary(tmp_path, fake_r2):
    """A donation only fills a gap. An employer row that prices the
    job keeps its own numbers and stays unattributed."""
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.welcometothejungle.com/en/companies/x/jobs/be",
        "title": "Backend Engineer (Payments)",
        "company": "Acme",
        "location": "Paris, France",
        "ats_id": "wttj-1",
        "salary_min": 50000.0,
        "salary_max": 60000.0,
        "salary_currency": "EUR",
        "salary_period": "YEAR",
        "salary_summary": "EUR 50,000 - 60,000",
    }]).to_csv(wttj_dir / "jobs.csv", index=False)

    lever_dir = tmp_path / "lever"
    lever_dir.mkdir()
    pd.DataFrame([{
        "url": "https://jobs.lever.co/acme/1",
        "title": "Backend Engineer, Payments",
        "company": "Acme",
        "location": "Paris",
        "ats_id": "lv-1",
        "salary_min": 90000.0,
        "salary_max": 110000.0,
        "salary_currency": "EUR",
        "salary_period": "YEAR",
        "salary_summary": "EUR 90,000 - 110,000",
    }]).to_csv(lever_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(tmp_path)

    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert len(df) == 1
    assert df.iloc[0]["ats_type"] == "lever"
    assert df.iloc[0]["salary_min"] == 90000.0
    assert pd.isna(df.iloc[0]["salary_source"])


def test_gated_fuzzy_pass_catches_reworded_titles(tmp_path, fake_r2):
    """``Research Engineer (Agent)`` vs ``Research Engineer, Agents``
    survives Pass 6 (the slugs differ by one letter) and has no
    country in common to block on, so Pass 7 is the only thing that
    can pair them."""
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.welcometothejungle.com/en/companies/x/jobs/re",
        "title": "Research Engineer (Agent)",
        "company": "Anthropic",
        "location": "New York, United States",
        "ats_id": "wttj-1",
    }]).to_csv(wttj_dir / "jobs.csv", index=False)

    greenhouse_dir = tmp_path / "greenhouse"
    greenhouse_dir.mkdir()
    pd.DataFrame([{
        "url": "https://job-boards.greenhouse.io/anthropic/jobs/1",
        "title": "Research Engineer, Agents",
        "company": "anthropic",
        "location": "San Francisco, CA | New York City, NY",
        "ats_id": "gh-1",
    }]).to_csv(greenhouse_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(tmp_path)

    assert result.total_jobs == 1
    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert df.iloc[0]["ats_type"] == "greenhouse"


def test_gated_fuzzy_pass_keeps_merely_similar_titles_apart(tmp_path, fake_r2):
    """Sharing most of a title is not being the same job. These two
    score 93.8 under Pass 5's ``token_set_ratio`` — loose enough to
    delete a real posting and hand its pay to a different one — and
    must both survive."""
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.welcometothejungle.com/en/companies/x/jobs/node",
        "title": "Senior Staff Software Engineer (Node Infrastructure)",
        "company": "Anthropic",
        "location": "New York, United States",
        "ats_id": "wttj-1",
        "salary_min": 300000.0,
        "salary_max": 405000.0,
        "salary_currency": "USD",
        "salary_period": "YEAR",
        "salary_summary": "USD 300,000 - 405,000",
    }]).to_csv(wttj_dir / "jobs.csv", index=False)

    greenhouse_dir = tmp_path / "greenhouse"
    greenhouse_dir.mkdir()
    pd.DataFrame([{
        "url": "https://job-boards.greenhouse.io/anthropic/jobs/1",
        "title": "Staff Software Engineer, Data Infrastructure",
        "company": "anthropic",
        "location": "San Francisco, CA",
        "ats_id": "gh-1",
    }]).to_csv(greenhouse_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(tmp_path)

    assert result.total_jobs == 2
    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert pd.isna(df[df["ats_type"] == "greenhouse"].iloc[0]["salary_min"])
    # Nothing was donated anywhere, so the column never materializes.
    assert "salary_source" not in df.columns


def test_salary_follows_a_row_dropped_by_the_older_passes(tmp_path, fake_r2):
    """Passes 1-5 already dropped gated rows before this change, and
    they record no donor/recipient pair. The exact
    ``(company, title_slug)`` fallback is what keeps their pay from
    disappearing — here Pass 2 collapses the pair on an identical
    ``(company, title, location)``."""
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.welcometothejungle.com/en/companies/x/jobs/pm",
        "title": "Product Manager",
        "company": "Acme",
        "location": "Berlin, Germany",
        "ats_id": "wttj-1",
        "salary_min": 80000.0,
        "salary_max": 95000.0,
        "salary_currency": "EUR",
        "salary_period": "YEAR",
        "salary_summary": "EUR 80,000 - 95,000",
    }]).to_csv(wttj_dir / "jobs.csv", index=False)

    ashby_dir = tmp_path / "ashby"
    ashby_dir.mkdir()
    pd.DataFrame([{
        "url": "https://jobs.ashbyhq.com/acme/1",
        "title": "Product Manager",
        "company": "Acme",
        "location": "Berlin, Germany",
        "ats_id": "ashby-1",
    }]).to_csv(ashby_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(tmp_path)

    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert len(df) == 1
    assert df.iloc[0]["ats_type"] == "ashby"
    assert df.iloc[0]["salary_max"] == 95000.0
    assert df.iloc[0]["salary_source"] == "welcometothejungle"


def test_gated_source_still_outranks_public_aggregators(tmp_path, fake_r2):
    """The demotion is scoped to the employer's own board. Against a
    public aggregator the existing ``ATS_DEDUP_PRIORITY`` order still
    decides, and wttj sits above it."""
    wttj_dir = tmp_path / "welcometothejungle"
    wttj_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.welcometothejungle.com/en/companies/x/jobs/de",
        "title": "Data Engineer",
        "company": "Acme",
        "location": "Sydney, Australia",
        "ats_id": "wttj-1",
    }]).to_csv(wttj_dir / "jobs.csv", index=False)

    seek_dir = tmp_path / "seek"
    seek_dir.mkdir()
    pd.DataFrame([{
        "url": "https://www.seek.com.au/job/1",
        "title": "Data Engineer",
        "company": "Acme",
        "location": "Sydney, Australia",
        "ats_id": "seek-1",
    }]).to_csv(seek_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(tmp_path)

    assert result.total_jobs == 1
    df = pd.read_parquet(
        pd.io.common.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert df.iloc[0]["ats_type"] == "welcometothejungle"


def test_gated_fuzzy_oversize_block_skipped(caplog):
    """Blocking on company alone makes the gated pass's blocks wider
    than Pass 5's, so it needs the same cap: past it the block is
    skipped with a warning rather than run as an n² fuzz sweep.

    Titles differ by one plural so Pass 6's exact slug key doesn't
    collapse them first — otherwise Pass 7 never sees the block and
    the guard isn't exercised — while staying close enough that the
    fuzzy pass would drop all 20 if it ran.
    """
    import polars as pl

    from pipeline.publisher import _decide_dedup_survivors_polars

    keys_rows = []
    for i in range(20):
        keys_rows.append({
            "_local_idx": i, "_orig_idx": i,
            "_priority": 1, "ats_type": "greenhouse",
            "url": f"https://job-boards.greenhouse.io/mega/{i}",
            "title_raw": f"Backend Engineer, Payments Role {i:03d}",
            "title": f"backend engineer, payments role {i:03d}",
            "company": "mega inc",
            "location": "san francisco, ca",
            "ats_id": f"gh{i}",
        })
    for i in range(20):
        keys_rows.append({
            "_local_idx": i, "_orig_idx": 20 + i,
            "_priority": 3, "ats_type": "welcometothejungle",
            "url": f"https://www.welcometothejungle.com/en/companies/mega/{i}",
            "title_raw": f"Backend Engineer (Payment) Role {i:03d}",
            "title": f"backend engineer (payment) role {i:03d}",
            "company": "mega inc",
            "location": "san francisco, united states",
            "ats_id": f"wttj{i}",
        })
    keys = pl.DataFrame(keys_rows)

    with caplog.at_level("WARNING"):
        survivors = _decide_dedup_survivors_polars(
            keys, fuzzy_threshold=90, fuzzy_max_block_size=30,
        ).survivors

    assert any(
        "Gated-source fuzzy" in r.getMessage() and "oversize" in r.getMessage()
        for r in caplog.records
    ), f"expected oversize warning in {[r.getMessage() for r in caplog.records]}"
    assert sum(s.height for s in survivors.values()) == 40


# --- Phase 1 / Phase 2 cross-source fuzzy dedup -----------------------------


@pytest.fixture
def ats_csv_dir_phase1(tmp_path):
    """Two aggregators emit the *same job* with formatting variations
    that defeat the exact-key (Pass 2) dedup:

      - eures: title with trailing Berufenet tag, location as NUTS
        prefix (``"DE (DEA58)"``).
      - bundesagentur: title without the tag, location as full text
        (``"Berlin, Berlin, Deutschland"``).

    Both rows share ``(company_norm, title_core, country_iso)`` so the
    new Phase 1 pass must collapse them; the global snapshot should
    keep only the higher-priority slice. Eures and Bundesagentur both
    sit at priority 6 — the earlier-emitted row (eures here, since the
    ATSType enum lists it first) wins on the tie-break.
    """
    # eures emits: title with trailing Berufenet code, NUTS-style location
    eures_dir = tmp_path / "eures"
    eures_dir.mkdir()
    pd.DataFrame([
        {
            "url": "https://eures.example/job/1",
            "title": "Backend Engineer (m/w/d) (Softwareentwickler/in)",
            "company": "ACME GmbH",
            "location": "DE (DE300)",
            "ats_id": "e1",
        },
        {
            "url": "https://eures.example/job/2",
            "title": "Marketing Manager (m/w/d) (Marketingfachkraft)",
            "company": "ACME GmbH",
            "location": "DE (DE712)",
            "ats_id": "e2",
        },
    ]).to_csv(eures_dir / "jobs.csv", index=False)

    # bundesagentur emits the same jobs without the Berufenet tag,
    # with full-text location.
    bundes_dir = tmp_path / "bundesagentur"
    bundes_dir.mkdir()
    pd.DataFrame([
        {
            "url": "https://arbeitsagentur.example/job/1",
            "title": "Backend Engineer (m/w/d)",
            "company": "ACME GmbH",
            "location": "Berlin, Berlin, Deutschland",
            "ats_id": "b1",
        },
        {
            "url": "https://arbeitsagentur.example/job/2",
            "title": "Marketing Manager (m/w/d)",
            "company": "ACME GmbH",
            "location": "München, Bayern, Deutschland",
            "ats_id": "b2",
        },
    ]).to_csv(bundes_dir / "jobs.csv", index=False)

    return tmp_path


def test_phase1_dedups_formatting_variations(ats_csv_dir_phase1, fake_r2):
    """Pass 4 (Phase 1) must collapse the eures / bundes mirror pair
    that differs only in trailing Berufenet tag + location format."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir_phase1)

    assert result.total_jobs_raw == 4  # two pairs of cross-source dups
    assert result.total_jobs == 2  # Phase 1 collapses each pair
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    # Per-ATS slices stay raw (2 rows each).
    assert manifest["by_ats"]["eures"]["rows"] == 2
    assert manifest["by_ats"]["bundesagentur"]["rows"] == 2


@pytest.fixture
def ats_csv_dir_phase2(tmp_path):
    """Two aggregators emit the same job with title typos / minor
    wording differences that defeat both the exact-key (Pass 2) and
    the formatting-normalised (Pass 4) dedups. Only Phase 2 fuzzy
    should collapse them.

    ``"Senior Backend Engineer (m/w/d)"`` vs
    ``"Sr. Backend Engineer (m/w/d)"`` — same role, ``token_set_ratio``
    sits around 86–95 depending on the rapidfuzz version. Phase 2's
    default threshold is 90 so this passes the bar.
    """
    eures_dir = tmp_path / "eures"
    eures_dir.mkdir()
    pd.DataFrame([{
        "url": "https://eures.example/fuzzy/1",
        "title": "Senior Backend Engineer (m/w/d)",
        "company": "Fuzzy GmbH",
        "location": "Berlin, Deutschland",
        "ats_id": "ef1",
    }]).to_csv(eures_dir / "jobs.csv", index=False)

    bundes_dir = tmp_path / "bundesagentur"
    bundes_dir.mkdir()
    pd.DataFrame([{
        "url": "https://arbeitsagentur.example/fuzzy/1",
        "title": "Senior Backend Engineer (m/w/d) - flexible",
        "company": "Fuzzy GmbH",
        "location": "München, Deutschland",
        "ats_id": "bf1",
    }]).to_csv(bundes_dir / "jobs.csv", index=False)

    return tmp_path


def test_phase2_fuzzy_dedups_title_variations(ats_csv_dir_phase2, fake_r2):
    """Pass 5 (Phase 2) must collapse cross-source rows whose titles
    differ by minor wording but share the ``(company_norm, country)``
    block."""
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(ats_csv_dir_phase2)

    assert result.total_jobs_raw == 2
    assert result.total_jobs == 1


def test_phase2_does_not_cross_dedup_within_ats(tmp_path, fake_r2):
    """Two rows from the SAME ATS with near-identical titles must
    both survive — the publisher's contract is that per-ATS slices
    stay raw, and that contract must hold through fuzzy dedup too."""
    eures_dir = tmp_path / "eures"
    eures_dir.mkdir()
    pd.DataFrame([
        {
            "url": "https://eures.example/a",
            "title": "Senior Backend Engineer",
            "company": "Same GmbH",
            "location": "Berlin, Deutschland",
            "ats_id": "e1",
        },
        {
            "url": "https://eures.example/b",
            "title": "Sr. Backend Engineer",
            "company": "Same GmbH",
            "location": "Berlin, Deutschland",
            "ats_id": "e2",
        },
    ]).to_csv(eures_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    result = publisher.publish_from_directory(tmp_path)

    # Both within-ATS rows kept (fuzzy is cross-ATS-only).
    assert result.total_jobs == 2
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])
    assert manifest["by_ats"]["eures"]["rows"] == 2


def test_phase2_respects_priority_when_dedupping(tmp_path, fake_r2):
    """When cross-ATS dups collide on fuzzy match, the higher-priority
    ATS's row wins. ``workday`` (priority 1) beats ``eightfold``
    (priority 5)."""
    workday_dir = tmp_path / "workday"
    workday_dir.mkdir()
    pd.DataFrame([{
        "url": "https://workday.example/job/1",
        "title": "Senior Backend Engineer (m/w/d)",
        "company": "Priority Co",
        "location": "Berlin, Deutschland",
        "ats_id": "w1",
    }]).to_csv(workday_dir / "jobs.csv", index=False)

    eightfold_dir = tmp_path / "eightfold"
    eightfold_dir.mkdir()
    pd.DataFrame([{
        "url": "https://eightfold.example/job/1",
        "title": "Sr. Backend Engineer (m/w/d)",
        "company": "Priority Co",
        "location": "Berlin, Deutschland",
        "ats_id": "ef1",
    }]).to_csv(eightfold_dir / "jobs.csv", index=False)

    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(tmp_path)

    all_parquet = fake_r2.uploads["jobhive/v1/all.parquet"]["data"]
    df = pd.read_parquet(pd.io.common.BytesIO(all_parquet))
    assert len(df) == 1
    assert df.iloc[0]["ats_type"] == "workday"
    assert "workday.example" in df.iloc[0]["url"]


def test_phase2_oversize_block_skipped(caplog):
    """A block with more rows than ``fuzzy_max_block_size`` must be
    skipped (rather than blowing up the wall clock with n² fuzz
    calls); a warning is logged so the operator can investigate.

    Titles are made *distinct* between the two ATS slices so Phase 1
    (exact ``(company_norm, title_core, country)`` collapse) doesn't
    eat the rows before Phase 2 sees them — otherwise the oversize
    code path is never exercised.
    """
    import polars as pl

    from pipeline.publisher import _decide_dedup_survivors_polars

    # 20 + 20 = 40-row block. Each ATS uses a different role family per
    # row so no two rows across slices share the same ``title_core``
    # (Phase 1 stays a no-op). The titles are still close enough that
    # rapidfuzz's ``token_set_ratio`` would fire if Phase 2 reached
    # them — which is exactly what the oversize guard prevents.
    keys_rows = []
    for i in range(20):
        keys_rows.append({
            "_local_idx": i, "_orig_idx": i,
            "_priority": 6, "ats_type": "eures",
            "url": f"https://eures.example/{i}",
            "title_raw": f"Backend Engineer Role {i:03d}",
            "title": f"backend engineer role {i:03d}",
            "company": "mega gmbh",
            "location": "berlin, deutschland",
            "ats_id": f"e{i}",
        })
    for i in range(20):
        keys_rows.append({
            "_local_idx": i, "_orig_idx": 20 + i,
            "_priority": 6, "ats_type": "bundesagentur",
            "url": f"https://arbeitsagentur.example/{i}",
            "title_raw": f"Senior Frontend Engineer Role {i:03d}",
            "title": f"senior frontend engineer role {i:03d}",
            "company": "mega gmbh",
            "location": "münchen, deutschland",
            "ats_id": f"b{i}",
        })
    keys = pl.DataFrame(keys_rows)

    with caplog.at_level("WARNING"):
        survivors = _decide_dedup_survivors_polars(
            keys, fuzzy_threshold=90, fuzzy_max_block_size=30,
        ).survivors

    # The warning is the contract: this block is skipped, not silently
    # dedupped past the cap.
    assert any(
        "Phase-2 fuzzy" in r.getMessage() and "oversize" in r.getMessage()
        for r in caplog.records
    ), f"expected oversize warning in {[r.getMessage() for r in caplog.records]}"

    # And the skip means nothing got dropped: all 40 rows survive
    # (eures 20 + bundesagentur 20).
    assert sum(s.height for s in survivors.values()) == 40


# --- helper-function unit tests ---------------------------------------------


def test_country_iso_extracts_common_eu_patterns():
    from pipeline.publisher import _country_iso_from_location as f

    # Full-text suffixes (Bundesagentur style)
    assert f("Berlin, Berlin, Deutschland") == "DE"
    assert f("Paris, France") == "FR"
    assert f("Wien, Österreich") == "AT"
    assert f("Brussels, Belgium") == "BE"
    # API-style alpha-2 suffixes (SmartRecruiters/Recruitee style)
    assert f("Berlin, DE") == "DE"
    assert f("Paris, FR") == "FR"


def _dir_with_postings(tmp_path, rows, name="run"):
    """A fresh source tree per run — publishes read the whole directory."""
    root = tmp_path / name
    ats_dir = root / "greenhouse"
    ats_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(ats_dir / "jobs.csv", index=False)
    return root


def _published(fake_r2):
    return pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    ).sort_values("ats_id")


def test_first_publish_starts_every_clock_today(tmp_path, fake_r2) -> None:
    """Audit finding 02: staleness was not expressible at all.

    The pipeline republishes a snapshot with no tombstoning, so nothing
    recorded how long a posting had been up.
    """
    source = _dir_with_postings(tmp_path, [
        {"url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
         "ats_id": "1"},
    ])
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(source)

    row = _published(fake_r2).iloc[0]
    assert row["first_seen_at"] == row["last_seen_at"]


def test_second_publish_keeps_the_original_sighting(tmp_path, fake_r2) -> None:
    """The whole point of the column: age survives a republish."""
    rows = [
        {"url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
         "ats_id": "1"},
    ]
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(_dir_with_postings(tmp_path, rows, "one"))
    first_run = _published(fake_r2).iloc[0]

    publisher.publish_from_directory(_dir_with_postings(tmp_path, rows, "two"))
    second_run = _published(fake_r2).iloc[0]

    assert second_run["first_seen_at"] == first_run["first_seen_at"]
    assert second_run["last_seen_at"] > first_run["last_seen_at"]


def test_a_posting_added_later_starts_its_own_clock(tmp_path, fake_r2) -> None:
    """Ages are per posting, not per publish."""
    old = {"url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
           "ats_id": "1"}
    new = {"url": "https://gh.com/2", "title": "Designer", "company": "Acme",
           "ats_id": "2"}
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(_dir_with_postings(tmp_path, [old], "one"))
    publisher.publish_from_directory(_dir_with_postings(tmp_path, [old, new], "two"))

    published = _published(fake_r2)
    assert published["first_seen_at"].iloc[0] < published["first_seen_at"].iloc[1]
    assert published["last_seen_at"].iloc[0] == published["last_seen_at"].iloc[1]


def test_a_relisted_url_keeps_its_age(tmp_path, fake_r2) -> None:
    """Boards rewrite URLs on edit; the source id is the stable identity.

    Keying on the URL would reset the age of every edited posting and
    make the corpus look permanently fresh.
    """
    rows = [{"url": "https://gh.com/jobs/1", "title": "Engineer",
             "company": "Acme", "ats_id": "1"}]
    moved = [{"url": "https://gh.com/careers/engineer-1", "title": "Engineer",
              "company": "Acme", "ats_id": "1"}]
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(_dir_with_postings(tmp_path, rows, "one"))
    before = _published(fake_r2).iloc[0]["first_seen_at"]

    publisher.publish_from_directory(_dir_with_postings(tmp_path, moved, "two"))

    assert _published(fake_r2).iloc[0]["first_seen_at"] == before


def test_sidecar_forgets_postings_that_went_away(tmp_path, fake_r2) -> None:
    """Otherwise the index grows forever against a fixed-size corpus."""
    import polars as pl

    from pipeline.publisher import FIRST_SEEN_SIDECAR

    gone = {"url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
            "ats_id": "1"}
    stays = {"url": "https://gh.com/2", "title": "Designer", "company": "Acme",
             "ats_id": "2"}
    publisher = DatasetPublisher(fake_r2, write_parquet=True)
    publisher.publish_from_directory(
        _dir_with_postings(tmp_path, [gone, stays], "one")
    )
    publisher.publish_from_directory(_dir_with_postings(tmp_path, [stays], "two"))

    sidecar = pl.read_parquet(
        io.BytesIO(fake_r2.uploads[f"jobhive/v1/{FIRST_SEEN_SIDECAR}"]["data"])
    )
    assert sidecar["_seen_key"].to_list() == ["greenhouse|2"]


def test_a_corrupt_sidecar_does_not_fail_the_publish(tmp_path, fake_r2) -> None:
    """Losing ages is recoverable; losing the publish is not."""
    from pipeline.publisher import FIRST_SEEN_SIDECAR

    fake_r2.upload_bytes(
        b"not a parquet file",
        f"jobhive/v1/{FIRST_SEEN_SIDECAR}",
        content_type="application/vnd.apache.parquet",
    )
    source = _dir_with_postings(tmp_path, [
        {"url": "https://gh.com/1", "title": "Engineer", "company": "Acme",
         "ats_id": "1"},
    ])

    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        source
    )

    assert result.total_jobs == 1
    assert _published(fake_r2)["first_seen_at"].notna().all()


SENIORITY_TITLES = [
    "Software Engineering Intern", "Werkstudent Marketing (m/w/d)",
    "Chief Technology Officer", "VP of Engineering", "Chief of Staff",
    "Director of Engineering", "Senior Director, Sales", "Art Director",
    "Engineering Manager", "Product Manager", "Senior Product Manager",
    "Principal Engineer", "School Principal", "Principal Investigator",
    "Staff Software Engineer", "Staff Nurse", "Staff Accountant",
    "Tech Lead, Payments", "Lead Generation Specialist",
    "Senior Software Engineer", "Sr. Data Scientist", "Senior Living Nurse",
    "Mid-Level Accountant", "Junior Developer", "Junior High Teacher",
    "Graduate Software Engineer", "Graduate School Advisor",
    "Software Engineer", "Nurse", "Barista",
]


def test_seniority_expression_matches_the_python_function():
    """The publisher recompiles the rules into polars for speed.

    polars runs Rust's regex engine, Python runs its own; the two accept
    different syntax. If they ever disagree the published column stops
    matching the documented function, silently and only for the titles
    that hit the difference.
    """
    import polars as pl

    from ats_scrapers.enrichment.derived import infer_seniority
    from pipeline.publisher import _seniority_expr

    vectorized = (
        pl.DataFrame({"title": SENIORITY_TITLES})
        .select(_seniority_expr().alias("seniority"))
        .to_series()
        .to_list()
    )
    assert vectorized == [infer_seniority(t) for t in SENIORITY_TITLES]


def test_publish_labels_seniority_from_the_title(tmp_path, fake_r2) -> None:
    """Audit finding 09: seniority was 74.88% Unknown."""
    ats_dir = tmp_path / "greenhouse"
    ats_dir.mkdir()
    pd.DataFrame([
        {"url": "https://gh.com/1", "title": "Senior Software Engineer",
         "company": "Acme", "ats_id": "1"},
        {"url": "https://gh.com/2", "title": "Software Engineering Intern",
         "company": "Acme", "ats_id": "2"},
        {"url": "https://gh.com/3", "title": "Staff Nurse",
         "company": "Hosp", "ats_id": "3"},
    ]).to_csv(ats_dir / "jobs.csv", index=False)

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(tmp_path)

    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    ).sort_values("ats_id")
    levels = published["seniority"].tolist()
    assert levels[:2] == ["SENIOR", "INTERN"]
    assert pd.isna(levels[2]), "'Staff Nurse' is an entry-grade nurse"


def test_publish_keeps_a_seniority_the_source_supplied(tmp_path, fake_r2) -> None:
    """An ATS that ships a structured level beats reading the title."""
    ats_dir = tmp_path / "greenhouse"
    ats_dir.mkdir()
    pd.DataFrame([
        {"url": "https://gh.com/1", "title": "Software Engineer",
         "company": "Acme", "ats_id": "1", "seniority": "STAFF"},
        {"url": "https://gh.com/2", "title": "Senior Software Engineer",
         "company": "Acme", "ats_id": "2", "seniority": ""},
    ]).to_csv(ats_dir / "jobs.csv", index=False)

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(tmp_path)

    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    ).sort_values("ats_id")
    assert published["seniority"].tolist() == ["STAFF", "SENIOR"]


def _placeholder(company: str) -> bool:
    import polars as pl

    from pipeline.publisher import _placeholder_company_expr

    return bool(
        pl.DataFrame({"company": [company]})
        .select(_placeholder_company_expr())
        .to_series()[0]
    )


def test_withheld_employers_are_recognized_across_locales():
    """Audit finding 09: one facet swallowing thousands of employers.

    EURES fronts national job services that hide the employer until a
    candidate applies, each in its own language.
    """
    for value in (
        "non renseigné", "non renseigne", "no se especifica",
        "siehe beschreibung", "see description", "Confidentiel",
        "konfidentiell", "anonymous", "N/A", "unknown", "  Unspecified  ",
    ):
        assert _placeholder(value), f"{value!r} should not name an employer"


def test_bare_careers_hostnames_are_recognized():
    """Oracle/Phenom/Personio publish the careers host absent a name."""
    for value in ("jobs.bell.ca", "careers.acme.com", "www.acme.co.uk"):
        assert _placeholder(value), f"{value!r} is a hostname, not an employer"


def test_real_employers_survive_the_placeholder_test():
    """The costly failure is a false positive silently deleting a name."""
    for value in (
        "Anthropic", "Booz Allen Hamilton Inc.", "Yahoo! Inc.",
        "Siemens", "Confidential Search Partners LLC", "N.A. Williams",
        "Privat Bank", "Company Three", "Acme S.A.",
    ):
        assert not _placeholder(value), f"{value!r} is a real employer"


def test_publish_drops_postings_that_name_no_employer(tmp_path, fake_r2) -> None:
    """Audit finding 09: one facet swallowing thousands of employers.

    The employer is the fact every consumer joins on, so a posting
    without one is dropped rather than published under a placeholder.
    """
    ats_dir = tmp_path / "eures"
    ats_dir.mkdir()
    pd.DataFrame([
        {"url": "https://eures.eu/1", "title": "Chef", "ats_id": "1",
         "company": "non renseigné", "location": "Paris, France"},
        {"url": "https://eures.eu/2", "title": "Nurse", "ats_id": "2",
         "company": "jobs.bell.ca", "location": "Toronto, ON, Canada"},
        {"url": "https://eures.eu/3", "title": "Engineer", "ats_id": "3",
         "company": "Siemens", "location": "Berlin, Germany"},
    ]).to_csv(ats_dir / "jobs.csv", index=False)

    result = DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(
        tmp_path
    )

    assert result.total_jobs == 1
    published = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/all.parquet"]["data"])
    )
    assert published["company"].tolist() == ["Siemens"]


def test_dropped_employers_are_auditable(tmp_path, fake_r2) -> None:
    """This rule drops the most rows of any of them.

    A regression in the pattern would quietly delete a large share of
    the FR/ES catalog, so the sidecar has to show what went and why.
    """
    ats_dir = tmp_path / "eures"
    ats_dir.mkdir()
    pd.DataFrame([
        {"url": "https://eures.eu/1", "title": "Chef", "ats_id": "1",
         "company": "non renseigné", "location": "Paris, France"},
        {"url": "https://eures.eu/2", "title": "Engineer", "ats_id": "2",
         "company": "Siemens", "location": "Berlin, Germany"},
    ]).to_csv(ats_dir / "jobs.csv", index=False)

    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(tmp_path)

    dropped = pd.read_parquet(
        io.BytesIO(fake_r2.uploads["jobhive/v1/quarantine.parquet"]["data"])
    )
    assert dropped["quarantine_reason"].tolist() == ["placeholder_employer"]
    assert dropped["company"].tolist() == ["non renseigné"]


def test_country_iso_reads_bare_us_state_suffixes():
    """"Austin, TX" is unambiguously US but names no country.

    The country-name list can't see states, so 37% of rows shipped with
    an empty ``country_iso`` and dropped out of every country filter.
    """
    from pipeline.publisher import _country_iso_from_location as f

    assert f("Austin, TX") == "US"
    assert f("Seattle, WA") == "US"
    assert f("New York, NY") == "US"
    assert f("Remote (US)") == "US"


def test_country_iso_reads_trailing_ca_as_california():
    """Audit finding 04: SF postings were filed under country Canada.

    ``CA`` is both California and Canada; a trailing bare ``CA`` used to
    resolve to Canada, so a US filter silently discarded San Francisco
    jobs with no way to see why.
    """
    from pipeline.publisher import _country_iso_from_location as f

    assert f("San Francisco, CA") == "US"
    assert f("Los Angeles, CA") == "US"
    assert f("San Diego, CA") == "US"


def test_country_iso_keeps_canada_when_a_province_is_present():
    """The California tie-break must not swallow real Canadian rows."""
    from pipeline.publisher import _country_iso_from_location as f

    assert f("Toronto, ON, CA") == "CA"
    assert f("Vancouver, BC, CA") == "CA"
    assert f("Montreal, QC, CA") == "CA"
    assert f("Toronto, ON, Canada") == "CA"


def test_country_iso_keeps_germany_for_trailing_de():
    """``DE`` resolves the other way from ``CA`` — see the resolver docs.

    ``"<City>, DE"`` is the alpha-2 suffix the EU ATSes emit, which is
    the high-volume meaning; Delaware in this bare form is the accepted
    residual.
    """
    from pipeline.publisher import _country_iso_from_location as f

    assert f("Berlin, DE") == "DE"
    assert f("Wilmington, DE") == "DE"
    # An explicit country marker still wins over the suffix.
    assert f("Wilmington, DE, USA") == "US"


def test_country_iso_uses_word_boundaries():
    """``"usa"`` is a substring of common European place names like
    ``"Lausanne"`` (CH). The earlier substring-match implementation
    tagged Lausanne jobs as US — cubic #1 on PR #33. The fix
    word-boundary-anchors every needle so the substring no longer
    matches."""
    from pipeline.publisher import _country_iso_from_location as f

    # Lausanne (CH) standalone — no country suffix. The bare city name
    # used to false-positive on US via the ``usa`` substring; now it
    # returns empty until something else identifies the country.
    assert f("Lausanne") == ""
    assert f("Lausanne (Vaud)") == ""
    assert f("Lausanne, Vaud") == ""
    # With the country suffix the CH match wins because CH appears
    # before US in the patterns list.
    assert f("Lausanne, Suisse") == "CH"
    assert f("Lausanne, Switzerland") == "CH"
    # Other word-fragment false positives that used to fire:
    assert f("Glausage") == ""
    assert f("usable") == ""
    # Real US strings still match.
    assert f("New York, USA") == "US"
    assert f("U.S.A. office") == "US"

    # NUTS-prefix style (eures)
    assert f("DE (DEA58)") == "DE"
    assert f("FR (FRK21)") == "FR"

    # Mixed-case full text
    assert f("Zurich, Switzerland") == "CH"

    # No signal → empty
    assert f("") == ""
    assert f(None) == ""
    assert f("Remote") == ""
    assert f("Remote, XX") == ""


def test_title_core_strips_trailing_parenthesised_tag():
    from pipeline.publisher import _title_core as f

    # The classic eures pattern: trailing Berufenet code in parens.
    assert (
        f("Anlagenmechaniker (m/w/d) ab 20€/Std. (Anlagenmechaniker/in)")
        == "anlagenmechaniker (m/w/d) ab 20€/std."
    )
    # Keep internal parens (m/w/d signals the same job).
    assert f("Marketing Manager (m/w/d)") == "marketing manager (m/w/d)"
    # Already clean title — no strip.
    assert f("Software Engineer") == "software engineer"
    # Empty / non-string
    assert f("") == ""
    assert f(None) == ""


# --- SHA256 stability -------------------------------------------------------


def test_sha256_is_stable_across_runs(ats_csv_dir, fake_r2) -> None:
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(ats_csv_dir)
    first = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])

    fake_r2.uploads.clear()
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(ats_csv_dir)
    second = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])

    assert (
        first["by_ats"]["greenhouse"]["sha256"]
        == second["by_ats"]["greenhouse"]["sha256"]
    )


def test_sha256_matches_uploaded_bytes(ats_csv_dir, fake_r2) -> None:
    DatasetPublisher(fake_r2, write_parquet=True).publish_from_directory(ats_csv_dir)
    manifest = json.loads(fake_r2.uploads["jobhive/v1/manifest.json"]["data"])

    csv_bytes = fake_r2.uploads["jobhive/v1/greenhouse/jobs.csv"]["data"]
    assert (
        hashlib.sha256(csv_bytes).hexdigest()
        == manifest["by_ats"]["greenhouse"]["sha256"]
    )

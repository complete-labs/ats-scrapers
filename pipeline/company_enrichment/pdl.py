"""People Data Labs free company dataset — domain, LinkedIn, size band.

Source and licence
------------------
PDL publishes a free extract of its company graph covering ~24M global
firms under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Any
downstream use must credit People Data Labs. Schema reference:
https://docs.peopledatalabs.com/docs/free-company-dataset

PDL's own download sits behind a click-through form, so it cannot be
fetched unattended. This module reads the Hugging Face mirror
``andreaaltomani/company-dataset`` (same data, tagged ``cc-by-4.0``,
32.3M rows) through Hugging Face's auto-converted parquet branch, which
supports column pruning and lets us keep only US rows. Operators who
prefer the first-party file can drop it in
``config.PDL_MANUAL_DIR`` and it will be used instead.

What it does and does not give us
---------------------------------
Available here: ``name``, ``website`` (bare domain), ``linkedin_url``,
``size`` (band), ``founded``, ``locality``/``region``/``country``,
``industry``.

**There is no exact headcount.** ``size`` is a self-reported band
("11-50", "1001-5000"); the integer ``employee_count`` field described
in PDL's schema docs is part of their paid product and is absent from
the free extract. Team size derived from this source is therefore a
band, and :mod:`.teamsize` treats it accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import name_key, normalize_domain

logger = logging.getLogger(__name__)

HF_REPO = "andreaaltomani/company-dataset"
HF_TREE_URL = (
    f"https://huggingface.co/api/datasets/{HF_REPO}"
    "/tree/refs%2Fconvert%2Fparquet/default/train"
)
HF_SHARD_URL = (
    f"https://huggingface.co/datasets/{HF_REPO}"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/{name}"
)

ATTRIBUTION = (
    "Company firmographics from the People Data Labs Free Company Dataset, "
    "used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
)

# PDL stores country as a lowercase display name, not an ISO code.
US_COUNTRY_VALUES = ("united states",)

# Midpoints for PDL's size bands, used only where a single number is
# unavoidable. The open-ended top band is deliberately conservative.
SIZE_BAND_MIDPOINT: dict[str, int] = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1001-5000": 3000,
    "5001-10000": 7500,
    "10001+": 15000,
}


def _shard_urls() -> list[str]:
    import json
    import urllib.request

    with urllib.request.urlopen(HF_TREE_URL, timeout=60) as resp:
        entries = json.load(resp)
    names = sorted(
        e["path"].rsplit("/", 1)[-1]
        for e in entries
        if str(e.get("path", "")).endswith(".parquet")
    )
    if not names:
        raise RuntimeError(f"no parquet shards found at {HF_TREE_URL}")
    total = sum(e.get("size", 0) for e in entries) / 1e9
    logger.info("PDL mirror: %d shards, %.2f GB total", len(names), total)
    return [HF_SHARD_URL.format(name=n) for n in names]


# Reading the shards in place over HTTP is pathologically slow: DuckDB
# issues a range request per column chunk, and against the HF CDN that
# latency dominates (measured ~99 MB in 20 minutes). Pulling each shard
# as a single sequential GET runs at ~37 MB/s instead. So we download,
# filter, and delete one shard at a time — peak extra disk stays at one
# shard (~280 MB) rather than the full 3.5 GB.
_SELECT_COLUMNS = [
    "name",
    "website",
    "linkedin_url",
    "size",
    "founded",
    "locality",
    "region",
    "country",
    "industry",
]


def _download(url: str, dest: Path) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)


def _manual_file() -> Path | None:
    """An operator-supplied PDL export, if one was dropped in the cache."""
    if not config.PDL_MANUAL_DIR.exists():
        return None
    for pattern in ("*.parquet", "*.json.gz", "*.jsonl.gz", "*.csv.gz", "*.csv"):
        matches = sorted(config.PDL_MANUAL_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_manual(path: Path) -> pl.DataFrame:
    logger.info("reading operator-supplied PDL export at %s", path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.name.endswith((".json.gz", ".jsonl.gz")):
        return pl.read_ndjson(path)
    return pl.read_csv(path, infer_schema_length=0)


def _scan_remote() -> pl.DataFrame:
    urls = _shard_urls()
    scratch = config.CACHE_DIR / "pdl_shards"
    scratch.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []

    for index, url in enumerate(urls, start=1):
        shard = scratch / f"shard_{index:04d}.parquet"
        try:
            logger.info("shard %d/%d: downloading", index, len(urls))
            _download(url, shard)
            part = (
                pl.scan_parquet(shard)
                .select(_SELECT_COLUMNS)
                .filter(
                    pl.col("country").is_in(US_COUNTRY_VALUES)
                    & pl.col("name").is_not_null()
                    & (pl.col("name") != "")
                )
                .collect()
            )
            frames.append(part)
            logger.info("shard %d/%d: kept %d US rows", index, len(urls), part.height)
        finally:
            shard.unlink(missing_ok=True)

    scratch.rmdir()
    return pl.concat(frames, how="vertical")


KEY_COLUMNS = ("name_key_raw", "name_key_core", "domain", "size_midpoint")

# PDL's free extract carries no per-record observation date, so the only
# freshness signal available is when we pulled it. Recorded beside the
# data because :mod:`.teamsize` must not stamp the size band with
# today's date — the band can be years old (Phasor Engineering is "11-50"
# here and 201-500 on LinkedIn now) and saying otherwise is a claim the
# source never made.
_SNAPSHOT_FILE = "pdl_snapshot.json"


def _record_snapshot() -> str:
    import json
    from datetime import UTC, datetime

    ingested = datetime.now(tz=UTC).date().isoformat()
    path = config.CACHE_DIR / _SNAPSHOT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ingested": ingested, "source": HF_REPO}))
    return ingested


def snapshot_date() -> str | None:
    """When the PDL extract was pulled, as an upper bound on freshness.

    Falls back to the extract's modification time for files written
    before the marker existed.
    """
    import json
    from datetime import UTC, datetime

    path = config.CACHE_DIR / _SNAPSHOT_FILE
    if path.exists():
        try:
            return str(json.loads(path.read_text())["ingested"])
        except (ValueError, KeyError):
            logger.warning("unreadable %s; falling back to file mtime", path)
    if config.PDL_PARQUET.exists():
        stamp = config.PDL_PARQUET.stat().st_mtime
        return datetime.fromtimestamp(stamp, tz=UTC).date().isoformat()
    return None


def _add_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the join keys the resolver blocks on.

    Both key forms are needed. ``name_key_raw`` keeps the legal suffix
    ("leidosholdingsinc") and ``name_key_core`` strips it ("leidos").
    Indexing only one side fails asymmetrically: a tenant called
    "Leidos" never reaches "LEIDOS HOLDINGS, INC." unless the reference
    row also carries the stripped form.
    """
    return df.with_columns(
        pl.col("name")
        .map_elements(lambda s: name_key(s, keep_suffix=True), return_dtype=pl.String)
        .alias("name_key_raw"),
        pl.col("name")
        .map_elements(name_key, return_dtype=pl.String)
        .alias("name_key_core"),
        pl.col("website")
        .map_elements(normalize_domain, return_dtype=pl.String)
        .alias("domain"),
        pl.col("size")
        .replace_strict(SIZE_BAND_MIDPOINT, default=None, return_dtype=pl.Int64)
        .alias("size_midpoint"),
    ).filter(pl.col("name_key_raw") != "")


def ingest(*, force: bool = False) -> pl.DataFrame:
    """Materialise the US slice of the PDL free dataset."""
    config.ensure_dirs()
    if config.PDL_PARQUET.exists() and not force:
        existing = pl.read_parquet(config.PDL_PARQUET)
        missing = [c for c in KEY_COLUMNS if c not in existing.columns]
        if not missing:
            logger.info("reusing %s", config.PDL_PARQUET)
            return existing
        # Backfill keys added after this file was written rather than
        # re-downloading 3.5 GB.
        logger.info("backfilling %s in %s", missing, config.PDL_PARQUET)
        rebuilt = _add_keys(existing.drop([c for c in KEY_COLUMNS if c in existing.columns]))
        rebuilt.write_parquet(config.PDL_PARQUET)
        return rebuilt

    manual = _manual_file()
    if manual is not None:
        raw = _read_manual(manual)
        if "country" in raw.columns:
            raw = raw.filter(pl.col("country").str.to_lowercase().is_in(US_COUNTRY_VALUES))
    else:
        raw = _scan_remote()

    df = _add_keys(raw)
    df.write_parquet(config.PDL_PARQUET)
    logger.info(
        "wrote %s (%d US rows, snapshot %s)",
        config.PDL_PARQUET,
        df.height,
        _record_snapshot(),
    )
    return df


def run(*, force: bool = False) -> pl.DataFrame:
    df = ingest(force=force)
    print("\n=== PDL free company dataset (US slice) ===")
    print(ATTRIBUTION)
    print(f"\nRows                    : {df.height:,}")
    print(f"With domain             : {df.filter(pl.col('domain') != '').height:,}")
    print(
        f"With LinkedIn URL       : "
        f"{df.filter(pl.col('linkedin_url').is_not_null()).height:,}"
    )
    print(
        f"With size band          : "
        f"{df.filter(pl.col('size').is_not_null()).height:,}"
    )
    print(
        f"With founded year       : "
        f"{df.filter(pl.col('founded').is_not_null()).height:,}"
    )
    print("\nSize band distribution:")
    print(
        df.group_by("size")
        .agg(pl.len().alias("companies"))
        .sort("companies", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.PDL_PARQUET}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

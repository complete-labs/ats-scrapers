"""Shared paths, HTTP settings, and source URLs for company enrichment.

Every artefact this package writes lands under :data:`OUTPUT_DIR`, which
is gitignored — the enrichment data is private and must never reach the
public R2 dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Private output root. Overridable so operators can point at a scratch
# volume; the default is gitignored via `/company-enrichment/`.
OUTPUT_DIR = Path(
    os.environ.get("ATS_ENRICHMENT_DIR") or REPO_ROOT / "company-enrichment"
)
# Raw third-party downloads (bulk ZIPs, TSVs, JSON). Kept separate from
# derived tables so a re-run can reuse multi-hundred-MB downloads.
CACHE_DIR = OUTPUT_DIR / "cache"

ATS_COMPANIES_DIR = REPO_ROOT / "ats-companies"

# --- Stage outputs ----------------------------------------------------
COHORT_PARQUET = OUTPUT_DIR / "cohort.parquet"
PDL_PARQUET = OUTPUT_DIR / "pdl_us.parquet"
EDGAR_PARQUET = OUTPUT_DIR / "edgar_identity.parquet"
RESOLVED_PARQUET = OUTPUT_DIR / "resolved.parquet"
RESOLVE_REVIEW_CSV = OUTPUT_DIR / "resolve_review.csv"
REGISTRANT_PARQUET = OUTPUT_DIR / "registrant_profile.parquet"
MARKETCAP_PARQUET = OUTPUT_DIR / "marketcap.parquet"
FUNDING_PARQUET = OUTPUT_DIR / "funding.parquet"
OSHA_PARQUET = OUTPUT_DIR / "osha_floor.parquet"
FORM5500_PARQUET = OUTPUT_DIR / "form5500_floor.parquet"
TEAMSIZE_PARQUET = OUTPUT_DIR / "teamsize.parquet"
PROFILE_PARQUET = OUTPUT_DIR / "profile.parquet"
ENRICHMENT_PARQUET = OUTPUT_DIR / "company_enrichment.parquet"
ENRICHMENT_CSV = OUTPUT_DIR / "company_enrichment.csv"

# --- Description caches -----------------------------------------------
# Both are keyed on (ats, slug) and store misses as well as hits, so a
# re-run never re-pays for a tenant that has already been tried.
COMPANY_SITE_CACHE = CACHE_DIR / "company_site.parquet"
BOILERPLATE_CACHE = CACHE_DIR / "posting_boilerplate.parquet"
# The raw posting heads the blurbs are derived from. Kept separately so
# the extraction rules can be changed and re-applied without paying for
# another pass over the remote jobs snapshot.
POSTING_SAMPLES_CACHE = CACHE_DIR / "posting_samples.parquet"

# A description is a one-line answer to "what does this company do", not
# an about page. Anything longer is a sign the extractor grabbed the
# surrounding prose rather than the blurb.
DESCRIPTION_MAX_CHARS = 600
DESCRIPTION_MIN_CHARS = 40

# --- Source URLs ------------------------------------------------------
JOBS_MANIFEST_URL = "https://storage.stapply.ai/jobhive/v1/manifest.json"

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)
SEC_SUBMISSIONS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
SEC_FORMD_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets"
SEC_FORMD_ZIP_TEMPLATE = "https://www.sec.gov/files/structureddata/data/form-d-data-sets/{year}q{quarter}_d.zip"
SEC_COMPANYCONCEPT_TEMPLATE = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/{tag}.json"
)
SEC_SUBMISSIONS_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
)
STOOQ_EOD_TEMPLATE = "https://stooq.com/q/l/?s={symbol}.us&f=sd2t2ohlcv&h&e=csv"

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# OSHA reposts the Form 300A summary under a new filename every year and
# moves it between `files/` and `largefiles/`, so the index page is the
# only durable entry point.
OSHA_ITA_INDEX_URL = "https://www.osha.gov/itadata"

# DOL EFAST2 Form 5500 annual datasets. Unlike OSHA these URLs *are*
# templatable and stable, and each year has a `Latest` directory holding
# the most recent revision of every filing for that plan year.
# `F_5500` is the full form; `F_5500_SF` is the short form small plans
# file, which is where the SMB long tail lives.
DOL_5500_URL_TEMPLATE = (
    "https://askebsa.dol.gov/FOIA%20Files/{year}/Latest/F_5500_{year}_Latest.zip"
)
DOL_5500_SF_URL_TEMPLATE = (
    "https://askebsa.dol.gov/FOIA%20Files/{year}/Latest/F_5500_SF_{year}_Latest.zip"
)
# Form 5500 is due seven months after the plan year ends and extensions
# push another two and a half months out, so the newest year is always
# partial: plan year 2025 held 33k full-form filings while 2024 held
# 222k. Several years are pulled and the newest filing per sponsor wins,
# which keeps coverage high without publishing a stale count when a
# fresher one exists.
DOL_5500_YEARS = 3

# PDL publishes the free dataset behind a click-through form, so it
# cannot be fetched unattended. The operator downloads it once and drops
# it here; `ingest-pdl` picks up whatever archive it finds.
PDL_DOWNLOAD_PAGE = "https://www.peopledatalabs.com/company-dataset"
PDL_MANUAL_DIR = CACHE_DIR / "pdl"

# --- HTTP -------------------------------------------------------------
# SEC requires a descriptive User-Agent with contact info and throttles
# above ~10 requests/second. https://www.sec.gov/os/webmaster-faq#developers
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "ats-scrapers company enrichment (contact: ops@stapply.ai)"
)
SEC_MAX_RPS = 8.0
HTTP_TIMEOUT = 120.0

# Names below this rapidfuzz score are written to the review file rather
# than accepted. Tuned in `resolve.py`; see that module for rationale.
MATCH_ACCEPT_SCORE = 92.0
MATCH_REVIEW_SCORE = 80.0


def ensure_dirs() -> None:
    """Create the private output tree. Safe to call repeatedly."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PDL_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

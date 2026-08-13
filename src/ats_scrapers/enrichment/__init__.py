"""Enrichment — salary parsing, geocoding, classification.

The legacy implementations live at the repo root in
`extract_salary_experience.py` and `classifier/`. This module will wrap them
progressively.
"""

from ats_scrapers.enrichment.derived import (
    infer_is_remote,
    parse_salary_range,
)
from ats_scrapers.enrichment.geo import (
    country_to_iso,
    region_for,
    resolve_country,
)

__all__ = [
    "country_to_iso",
    "infer_is_remote",
    "parse_salary_range",
    "region_for",
    "resolve_country",
]

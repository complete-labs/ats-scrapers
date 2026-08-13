"""Sidecar enrichment layer for the ats-scrapers jobs dataset.

The upstream package deliberately stops where prose-reading begins:
``ats_scrapers.enrichment.derived`` only infers ``is_remote`` from title
keywords (and never returns ``False``), and only parses salary out of the
``salary_summary`` field the ATS already structured. ``docs/JOB_SCHEMA.md``
defers ``country_iso``, ``region``, ``lat``/``lon``, ``is_remote``,
``experience`` and ``language`` to "the downstream LLM enrichment
pipeline" — which is not in the repo. This package is that pipeline.

Design constraints that shape every module here:

* **Sidecar, never in-place.** Nothing under ``src/``, ``scripts/`` or
  ``pipeline/`` is modified. Enrichment lands in its own DuckDB/parquet
  tables joined back on :func:`enrich.keys.job_key`. Upstream pulls stay
  conflict-free and a better model can re-enrich without re-scraping.
* **Tiered, cheapest-first.** Tier 0 is pure-Python and free; Tier 1 is a
  single batched LLM call per *unique description*; Tier 2 is a bounded
  tool-using agent that only runs on rows Tier 1 could not resolve.
* **Cache on content, not on row.** Multi-location postings repeat the
  same description across many rows, so the LLM cache is keyed by
  :func:`enrich.keys.content_hash`. A prompt change re-runs only the rows
  whose prompt actually changed.
"""

from enrich._version import ENRICHMENT_VERSION

__all__ = ["ENRICHMENT_VERSION"]

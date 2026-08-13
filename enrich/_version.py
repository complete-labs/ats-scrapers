"""Enrichment schema/pipeline version.

Bump when a change should invalidate previously written enrichment rows.
The store records this per row, so a bump makes ``pending`` re-select
already-enriched rows without a manual purge.

Do **not** bump for prompt-only edits — those are tracked separately by
``prompt_hash`` (see :mod:`enrich.prompt`), which invalidates at a finer
granularity.
"""

ENRICHMENT_VERSION = 1

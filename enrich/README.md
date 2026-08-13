# `enrich/` — sidecar enrichment for the jobs dataset

The upstream package stops where prose-reading begins. `docs/JOB_SCHEMA.md`
defers `country_iso`, `region`, `lat`/`lon`, `is_remote`, `experience` and
`language` to "the downstream LLM enrichment pipeline", and
`src/ats_scrapers/enrichment/__init__.py` points at an
`extract_salary_experience.py` and a `classifier/` that do not exist in the
repo. This package is that missing layer.

## Current state

| stage | status |
|---|---|
| Tier 0 over the full corpus | **run** — 4,854,656 rows, 31 min, $0 |
| `all_enriched.parquet` | **built** — 4,854,656 rows, 42 columns, 3.4 GB |
| Tier 1 plumbing | **verified** — 22,250 bodies through write → collect → apply |
| Tier 1 full backfill | **not run** — needs `OPENAI_API_KEY`, ~$385 |
| Tier 2 recovery | **built and tested**, not run at scale |
| delta loop | **run** — converges to zero work on an unchanged snapshot |

Everything that can be done without a provider account has been done. To spend
the money and finish Tier 1:

```bash
export OPENAI_API_KEY=...
python -m enrich.run batch-write     # ~42 GB of shards, resumable
python -m enrich.run batch-submit
python -m enrich.run batch-status    # poll; 24h completion window
python -m enrich.run batch-collect --apply
python -m enrich.run join
```

`batch-write` skips bodies already in the cache, so an interrupted backfill
never re-pays for work it already has. Before submitting, rehearse the
downstream half offline with `batch-simulate` — it writes a provider-shaped
response file so `batch-collect` is exercised before any money moves.

## Design constraints

**Sidecar, never in-place.** Nothing under `src/`, `scripts/` or `pipeline/`
is modified.<sup>1</sup> Enrichment lands in its own DuckDB tables and
parquet, joined back on the posting URL. Upstream pulls stay conflict-free,
and a better model can re-enrich without re-scraping.

**Tiered, cheapest-first.** Tier 0 is pure Python and free. Tier 1 is one
batched model call per *unique description*. Tier 2 is a bounded tool-using
agent that only runs where Tier 1 has nothing to read.

**Cache on content, not on row.** The model cache is keyed by
`(content_hash, prompt_hash, model_id)`, so editing the prompt re-runs only
the rows whose prompt actually changed, and a crashed backfill re-reads its
own answers instead of re-buying them.

**Abstention over plausibility.** Every extractor, and the prompt itself,
prefers `null` to a confident guess. Salary, placement and experience claims
require a verbatim quote from the posting; an unquoted model salary is
flagged for review rather than trusted.

<sup>1</sup> The single exception is two lines in `pyproject.toml`
registering the `enrich_eval` pytest marker, which `--strict-markers`
requires.

## Measured state of the corpus

From `data/reports/gap_profile.md` (full 4,854,656-row snapshot, 46s in
DuckDB):

| field | missing before | after Tier 0 | who fills it |
|---|--:|--:|---|
| `language` | 94.7% | 0.03% | Tier 0 |
| `lat` / `lon` | 99.8% | 13.9% | Tier 0 |
| `region` | 91.9% | 9.1% | Tier 0 |
| `country_iso` | 33.2% | 9.1% | Tier 0 |
| `employment_type` | 45.5% | 38.9% | Tier 0 + Tier 1 |
| `experience` | absent from schema | 76.8% | Tier 0 + Tier 1 |
| `is_remote` | 85.2% | 83.6% | Tier 1 |
| `salary_min` | 96.3% | 94.3% | Tier 1 |
| `department` | 68.4% | — | Tier 1 (closed taxonomy) |

Tier 0 coverage is measured on a seeded 30,000-row random sample; the
"missing before" column is the full corpus. Against provider-supplied ground
truth, Tier 0's `country_iso` agrees 95.5% of the time and `language` 95.0%
(`python -m enrich.run eval`).

**Tier 0 has since been run over the whole corpus** — 4,854,656 rows in 31
minutes on 8 workers, at zero cost. Final coverage, against the full-corpus
baseline rather than a sample:

| field | before | after | lift |
|---|--:|--:|--:|
| `language` | 5.3% | 100.0% | +94.6pp |
| `lat` / `lon` | 0.2% | 86.3% | +86.1pp |
| `region` | 8.1% | 91.1% | +83.0pp |
| `country_iso` | 66.8% | 91.1% | +24.3pp |
| `employment_type` | 54.5% | 61.0% | +6.5pp |
| `salary_min` | 3.7% | 5.8% | +2.1pp |
| `is_remote` | 14.8% | 16.6% | +1.8pp |

The shape of that table is the whole argument for tiering. The four fields
Tier 0 owns went from unusable to near-complete for free, and the two fields
it barely moves — `is_remote` and salary — are exactly the ones that need to
be read out of prose. Those are what Tier 1's ~$385 buys; paying a model to
infer `language` or a country centroid would have been waste.

In the joined output `country_iso` reaches 98.1%, because it coalesces the
provider's value with Tier 0's and the two disagree about which rows they
cover.

Three findings that changed the plan:

* **Dedup is worth ~6%, not 20–40%.** Distinct content bodies among
  enrichable rows: 4,229,953 of 4,499,647. Multi-location postings repeat
  less than expected, so dedup is a rounding error rather than the main cost
  lever. Truncation and Tier 0 short-circuiting do the real work. Measure it
  on the *full* corpus — a 200,000-row sample reports 0.9%, because
  duplicates are spread across the file.
* **`global_id` is not published.** It is documented, but `_job_to_row` never
  emits it and `DESCRIBE` confirms 26 columns without it. The sidecar keys on
  a normalized-URL hash and reconstructs `global_id` as `ats_type:ats_id`.
* **Every column of `all.parquet` is VARCHAR.** The publisher's
  `diagonal_relaxed` concat widens the per-ATS slices, so `is_remote` arrives
  as the text `'true'`/`'false'` and `lat`/`lon` as numeric strings. An
  `isinstance(value, bool)` check silently discards all 719,871
  provider-labelled remote rows.

## Cost

Projected from measured token counts (`pilot --dry-run`): **~$385** for the
full 4.23M-call backfill through the Batch API, ~$769 synchronous, at
gpt-4o-mini list rates (490 mean input tokens, 180 assumed output). Override
the rate card with `ATS_ENRICH_PRICE_IN` / `ATS_ENRICH_PRICE_OUT` before
trusting the figure.

## Usage

```bash
uv pip install -r enrich/requirements.txt

# One-time: fetch the published snapshot (2.25 GB, no auth).
curl -L -o data/all.parquet https://storage.stapply.ai/jobhive/v1/all.parquet

python -m enrich.run profile           # quantify the gap
python -m enrich.run tier0 --workers 8 # free deterministic pass
python -m enrich.run gold-build        # provider-labelled holdout
python -m enrich.run eval              # score against gold
python -m enrich.run dedup             # unique bodies Tier 1 must pay for

# Tier 1. Nothing spends money until --client openai is passed.
python -m enrich.run pilot -n 5000 --dry-run
OPENAI_API_KEY=... python -m enrich.run pilot -n 5000 --client openai
python -m enrich.run batch-write       # shards, written offline
python -m enrich.run batch-submit
python -m enrich.run batch-status
python -m enrich.run batch-collect --apply

# Submission is the only step that needs an account. batch-simulate writes a
# provider-shaped response file from a shard using the stub, so the
# collect -> cache -> apply half can be rehearsed offline first.
python -m enrich.run batch-simulate --shard data/batches/shard-00000.jsonl
python -m enrich.run batch-collect --from-file data/batches/shard-00000.out.jsonl --apply

python -m enrich.run agent --limit 500 # Tier 2 recovery
python -m enrich.run join              # data/all_enriched.parquet
python -m enrich.run delta             # steady state
python -m enrich.run stats
```

Every command defaults to `--client stub`, a deterministic offline extractor,
so the whole pipeline runs and is testable with no key and no spend. Stub
output is tagged `stub-deterministic-v1` and can never be mistaken for a real
model's.

## Modules

| module | responsibility |
|---|---|
| `keys.py` | `job_key`, `content_hash`, `fallback_key` and URL normalization |
| `schema.py` | `LlmExtraction` (the strict structured-output contract) and `JobEnrichment` (the stored row) |
| `geo.py` | offline gazetteer: location text → country, region, lat/lon |
| `deterministic.py` | Tier 0 extractors |
| `snapshot.py` | bounded-memory reads of the published parquet |
| `pipeline.py` | merge precedence and the Tier 0 batch loop |
| `prompt.py` | system prompt, user rendering, `prompt_hash` |
| `llm.py` | sync client, Batch API client, offline stub, cost model |
| `store.py` | DuckDB sidecar: enrichment, cache, batches, ledger, agent runs |
| `agent.py` | Tier 2 bounded recovery |
| `eval.py` | scoring: recall, precision, over-claim rate |
| `gold/` | 44 hand-written edge cases + 1000-row provider-labelled holdout |
| `delta.py` | steady-state loop |
| `profile.py` | Phase 0 gap and dedup measurement |
| `run.py` | CLI |

## Merge precedence

1. **Provider** — the ATS stated it structurally. Never overwritten.
2. **Tier 0** — a deterministic rule with no judgement.
3. **Tier 1 / Tier 2** — the model, for prose-only fields.

One deliberate exception: a Tier 0 `placement` derived from a *title keyword*
yields to the model, because Tier 0 never reads the body while the model
does. A provider-supplied `is_remote` still wins, and a disagreement is
recorded in `needs_review` — a systematic pattern there means a scraper is
mapping its provider's workplace type wrongly.

## Evaluation

`python -m enrich.run eval` scores 1,044 examples: 44 hand-written edge cases
(26 carrying *negative* labels — "the correct answer is to abstain") and 1,000
provider-labelled holdout rows.

Two things make these numbers mean something:

* **Masking.** If a label is visible in the prompt the eval measures nothing,
  so each example declares which inputs to withhold — `salary_summary` for a
  salary label, `commitment` and `employment_type` for an employment-type
  label.
* **The trivial/non-trivial split for `placement`.** When "Remote" appears in
  the title or location, any keyword matcher scores 100%. Only the
  non-trivial slice says anything about comprehension, so they are reported
  separately.

Reported per field: **recall** (of values that exist, how many we got right)
and **over-claim rate** (of cases that should be null, how many we filled
anyway). They are never blended — a missing salary is a gap, an invented one
is a silent data error.

Quality gates run as an opt-in suite: `pytest -m enrich_eval`.

### Known limitations

* Provider labels are imperfect: upstream's `is_remote` is partly derived
  from title keywords by the publisher, so some "ground truth" is a
  heuristic's output.
* The holdout is biased toward providers that ship structured fields.
* The stub client's scores measure keyword matching, not comprehension. Real
  quality numbers require `--client openai`.
* `department`, `seniority`, `skills`, `education_level` and
  `visa_sponsorship` have only hand-written labels (30 or fewer each), so
  their scores are directional rather than tight.
* `lat`/`lon` are only as precise as `geo_precision` says. A country-only
  location resolves to that country's largest city, not its centroid — jobs
  concentrate in cities, and a centroid puts "India" in rural Madhya Pradesh.
  Filter on `geo_precision = 'city'` before doing radius search, or every
  country-level posting will pile up on one point.

## Operational notes

Three things about running this at corpus scale that cost real debugging time:

* **Memory is scaled to the machine, not fixed.** `store.connect` sets
  `memory_limit` to half of RAM (capped at 16 GB), overridable with
  `ATS_ENRICH_MEMORY_LIMIT`. A fixed 4 GB ceiling suits the streaming Tier 0
  pass but makes `join` die with `failed to pin block` on a machine with 32 GB
  free, because the final join materializes a hash table over 4.85M rows.
* **In-memory DuckDB needs an explicit spill directory.** File-backed databases
  default to `<db>.tmp`; `:memory:` gets nothing, so the memory limit becomes a
  hard failure instead of a throttle. Set to `data/duckdb-spill/`.
* **The join runs in hash buckets.** `join --parts 8` splits on
  `hash(url) % parts`, applying the same predicate to both sides so the build
  side shrinks proportionally. A single-statement join over the full sidecar
  exhausts a 16 GB budget; bucketed, the whole 4.85M-row join takes 71
  seconds. Equal urls always hash into the same bucket, so the result is
  identical to the unbucketed join, and the command asserts the output row
  count still matches the snapshot — a LEFT JOIN that changes the row count
  means duplicate urls in the sidecar.
* **`--limit N` is a stable prefix.** Reads with a limit re-enable
  `preserve_insertion_order`, because a bare `LIMIT` over a parallel parquet
  scan returns whichever rows finish first — two identical `--limit 4000` reads
  were measured sharing only 1,952 rows. Without this, pilots cannot be
  compared against each other and `delta` reports spurious "new" rows against
  an unchanged snapshot. Note `--limit` is a *prefix*, and the snapshot is
  physically grouped by `ats_type`, so it is a debugging tool, not a sample:
  use `sample_rows` (seeded reservoir) for anything you intend to measure.

## Layout

Everything written lives under `data/` (gitignored upstream); override with
`ATS_ENRICH_DATA_DIR`.

```
data/
  all.parquet            # published snapshot
  enrichment.duckdb      # sidecar store
  all_enriched.parquet   # joined output
  gold/holdout.jsonl     # regenerable, seeded
  batches/               # Batch API shards
  duckdb-spill/          # scratch for spilling joins
  reports/               # gap_profile, eval, pilot, tier0_coverage, delta
```

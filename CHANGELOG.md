# Changelog

All notable changes to **ats-scrapers** are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Ashby accuracy

- `is_remote` no longer reports hybrid roles as remote. Ashby sets
  `isRemote: true` for hybrid *and* fully-remote postings, so the flag
  alone overstated remote work by roughly a third of the board.
  `workplaceType` is now the authoritative signal, and a bare
  `isRemote: true` with no `workplaceType` resolves to `None` rather
  than `True` — it only means "not fully on-site".
- `salary_min` / `salary_max` now span every pay band on postings that
  publish per-zone or per-level ranges, instead of reporting only the
  first band. Affected rows previously contradicted their own
  `salary_summary` (a `$165.4K – $285K` summary next to a `207000`
  minimum).
- `salary_period` is left `None` for compensation intervals we don't
  recognize instead of defaulting to `YEAR`, which would publish a
  short-period rate as an annual salary.

### Added — Ashby structured location

- `country_iso` and `region` are now populated from the posting's
  `address.postalAddress`, covering ~93% of rows. They are withheld
  when a posting spans several countries.
- Multi-location postings render every office in `location` rather
  than only the primary one.

### Added — shared country resolver

- `ats_scrapers.enrichment.geo` resolves an alpha-2 code, an alpha-3
  code, or a country name to `country_iso` + `region`. Every ATS spells
  countries differently — Lever sends `de`, Amazon sends `DEU`, Workday
  sends `Germany` — and each scraper previously carried its own partial
  table, so a country recognized on one source was dropped on another.
  Supranational and "anywhere" values (`European Union`, `Global`)
  resolve to nothing rather than to an arbitrary member state.

### Added — structured location on the highest-volume sources

Each of these ATSes already published a machine-readable country that
was being discarded, leaving the publisher to infer one from free text:

- **Workday**: `country_iso`, `region`, `posted_at`, `employment_type`
  and `is_remote` are now read from the per-job detail payload the
  scraper already fetches for descriptions, at no extra request cost.
  `posted_at` was previously always `None` — the search endpoint only
  reports a relative string ("Posted 30+ Days Ago") while the detail
  payload carries an absolute `startDate`. Measured on a live sample:
  country and posted date go from 0% to 99.7% of rows, employment type
  from 51% to 99.3%.
- **SmartRecruiters**: `country_iso`/`region` from `location.country`
  (63% → ~100% of rows), plus `lat`/`lon` from the coordinates it
  publishes for ~77% of postings — the first real geocodes in the
  dataset. The display location no longer ends in a lowercase ISO code
  ("Austin, TX, us").
- **Oracle**: `country_iso`/`region` from `PrimaryLocationCountry`
  (61% → ~100%).
- **Lever**: `country_iso`/`region` from `country` (22% → ~98%).
- **Amazon**: `country_iso`/`region` and `lat`/`lon` resolved per office,
  so a posting spanning several countries no longer stamps the primary
  country on every row. `is_remote` comes from each office's type —
  Amazon spells fully-remote roles `VIRTUAL`, not `REMOTE`.
- **Oracle**: `is_remote` now covers `ORA_HYBRID` as well as the remote
  and on-site codes. Hybrid is the second most common value, so leaving
  it unset stranded a third of the postings that state a workplace type.

### Added — detail fields cached beside descriptions

Providers whose extra fields only exist on a per-job detail endpoint can
return them from the new `BaseScraper.get_description_and_fields` hook.
The pipeline stores them in the description cache, so a cache hit is as
complete as a fresh fetch — otherwise Workday's country and posted date
would only ever be set for newly-seen listings, which on a board that is
almost entirely cache hits means almost never.

- `DescriptionCache` gains a nullable `metadata` column (schema v3).
  Existing v2 caches are migrated in place; the ~700k cached Workday
  descriptions are preserved rather than re-crawled.
- Writing back only a description no longer clears the metadata beside
  it.
- Set `ATS_SCRAPERS_DETAIL_FIELD_BACKFILL=1` to re-fetch cache entries
  that predate the metadata column. Off by default because it costs one
  extra pass over the board; after that the fields are cached normally.

### Fixed — cross-source consistency

- Hybrid roles report `is_remote = False` on every scraper. Lever and
  Workday left them `None` while Ashby reported `False`, so the field
  meant different things on different sources.
- `posted_at` is parsed as UTC in the seven scrapers that read epoch
  timestamps (lever, eightfold, eures, gem, tiktok, getonbrd,
  remoteok). They used a naive `datetime.fromtimestamp()`, which
  silently shifted every timestamp by the pipeline host's UTC offset.
- The publisher's location heuristic recognizes `UK`. It is not the ISO
  code for the United Kingdom (`GB` is), so postings written
  "London, UK" resolved to no country at all.

### Changed — the combined dataset prefers links you can open

A posting that exists both on Welcome to the Jungle and on the
employer's own ATS now resolves to the employer's row in
`all.{csv,parquet}`. WTTJ hides every posting behind a sign-in wall,
and it already ranked below the direct-employer sources — but it
reformats titles (`"Title, Qualifier"` → `"Title (Qualifier)"`) and
locations, so the dedup passes never paired the two and both rows
shipped. Two passes now match them on company plus a punctuation-free
title, with no location or country agreement required.

Only gated rows are dropped, and only against the employer's own
board: WTTJ still outranks the public aggregators. Per-source slices
(`<ats>/jobs.csv`) stay raw as before.

### Added — `salary_source`

New nullable column in `all.{csv,parquet}`. WTTJ is one of the few
sources that publishes pay, so a dropped row donates its salary block
to the employer row that replaced it when that row prices nothing
itself; `salary_source` names where the figure came from. Rows that
published their own salary keep it and leave the column null.

## [0.2.0] — 2026-07-23

### Added — company discovery without ATS knowledge

Nobody knows OpenAI runs on Ashby. Two new package-root entry points
remove the need to know the `(ats, slug)` pair:

- `get_scraper_for_url("https://jobs.ashbyhq.com/openai")` — builds
  the right scraper from a public careers URL. Recognizes 20+ ATS URL
  shapes (path-tenant, subdomain-tenant, and full-URL platforms like
  Workday/Taleo/iCIMS). `resolve_careers_url(url)` exposes the raw
  `(ats, slug)` mapping.
- `find_company("openai")` — case-insensitive name/slug lookup over
  the hosted companies directory (`Client.companies()`, cached
  in-process; exact matches rank first).

### Added — `ScraperRegistry.has_scraper(ats)`

Skip dataset sources this package can't scrape yet without catching
`ScraperError` (GH-185). The hosted dataset can list a source before a
matching scraper ships — `search()` already tolerates that; this makes
the scraper side symmetric.

### Fixed

- Search filters now treat user input as literal text instead of a regular
  expression, so values containing characters such as `+`, `(`, or `[` work
  correctly and cannot trigger regex errors (GH-182).
- Unknown hosted dataset sources remain usable even when the installed
  package does not yet define a matching enum member or scraper (GH-185).

### Security

- Multi-tenant scraper constructors now validate slugs before interpolating
  them into hostnames or URL paths, preventing malformed tenant input from
  escaping the intended ATS origin.

## [0.1.0] — 2026-07-22

Initial release of `ats-scrapers`:

- A Python client for querying the hosted job dataset (`search`,
  `Client`, `list_ats`, `Manifest`), including a compatibility backfill for
  `global_id` when reading legacy schema-v2 dataset artifacts.
- A shared `Job`/`Company` schema and 52 scraper adapters for ATS
  platforms and job sources.
- Async-first scrapers: every adapter implements `async def afetch()`;
  the sync `fetch()` wrapper is safe to call from inside a running
  event loop (Jupyter, FastAPI) — it runs the coroutine on a worker
  thread instead of crashing in `asyncio.run`.
- A shared HTTP layer, `ats_scrapers.fetch.Fetcher` (exported at the
  package root): retries/backoff with `Retry-After`, status→exception
  mapping, client lifecycle, default headers, proxy configuration
  (`ATS_SCRAPERS_PROXY`, legacy 4-colon `PROXY`), and two engines —
  plain httpx and httpcloak TLS impersonation — with per-scraper
  declared escalation for 403/406-blocking load balancers.
- `WorkdayScraper.from_url(...)` documents the full-careers-URL slug
  contract; `include_descriptions` and `proxy` are `BaseScraper`
  constructor parameters; `Job.fetched_at` is timezone-aware UTC.
- Optional extras: `[parquet]` (full-snapshot search), `[scrapers]`
  (BYO scraping), `[all]` (both).

The installable package is the library only — dataset publishing and
orchestration live in the repo-only `pipeline/` directory. The package
is installed as `ats-scrapers` and imported as `ats_scrapers`. It
replaces the retired `jobhive-py` distribution.

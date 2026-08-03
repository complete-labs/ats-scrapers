"""Stage 1 — resolve ATS tenants to real-world company identities.

This is the load-bearing stage. The companies directory carries only
``ats,name,slug,url`` and the ``url`` points at an ATS-hosted board
(``job-boards.greenhouse.io/<slug>``), so nothing in it joins to an
external dataset directly. Everything downstream needs either a **CIK**
(market cap, Form D funding) or a **domain** (team size), and this
module is what produces them.

Matching runs in two independent tracks that are merged at the end:

**Domain track.** Where a corporate domain can be recovered — from
eightfold's ``domain`` column, from a careers URL on the company's own
host, or from the board-page fallback — it is matched against PDL's
``website`` field. A domain equality is near-certain identity, so these
matches are accepted without a name score.

**Name track.** Otherwise the tenant name, the employer string seen in
the jobs feed, the slug, and the board path each generate two blocking
keys (with and without the legal suffix). Any exact key hit against PDL
or EDGAR becomes a candidate, and candidates are scored with rapidfuzz
against every one of the tenant's name variants, keeping the best.
Generating suffix variants on the query side keeps the reference tables
literal, so their keys never drift as the suffix list evolves.

The board path matters more than it sounds. Workday tenants are keyed by
an internal acronym, so ``ngc/northrop_grumman_external_site`` is called
"Ngc" everywhere in the directory and the only legible company name in
the record is inside the URL path.

Precision over recall: a match below
``config.MATCH_ACCEPT_SCORE`` is not written to the resolved table at
all. Anything scoring at least ``config.MATCH_REVIEW_SCORE`` is dropped
into ``resolve_review.csv`` for a human instead of being guessed at.

Measured precision is 84% overall — see the accuracy envelope in the
package docstring for the per-stratum breakdown. The residual error is
almost entirely **name collision**, not weak scoring: every wrong match
in the validation sample scored a perfect 100 against an exact name key,
because an unrelated company genuinely shares the tenant's trade name.
Raising the threshold would not catch any of them. What does catch them
is corroboration from something other than the name, which is why
:mod:`.marketcap` cross-checks the registrant's home state and
:mod:`.teamsize` cross-checks an exact headcount against the band.
"""

from __future__ import annotations

import logging
import re

import polars as pl
from rapidfuzz import fuzz

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import (
    core_name,
    corporate_domain,
    desuffix_concatenated,
    name_key,
)

logger = logging.getLogger(__name__)

# Generic single-token names that collide with hundreds of unrelated
# records ("careers", "corporate"). A key this weak is never used for
# blocking; it produces nothing but false positives.
_STOP_KEYS = frozenset(
    {
        "", "careers", "career", "jobs", "job", "corporate", "company",
        "inc", "llc", "ltd", "group", "holdings", "the", "test", "demo",
        "hr", "recruiting", "talent", "hiring", "external", "internal",
        "main", "home", "default", "new", "us", "usa", "global",
    }
)

# A key this short matches too much to be safe on its own.
_MIN_KEY_LEN = 4


def _first_slug_segment(slug: str) -> str:
    """Workday slugs are ``tenant/site``; the tenant comes first."""
    return (slug or "").split("/")[0].split("?")[0]


# Board-path boilerplate. Workday site paths are the company name plus
# some of these ("northrop_grumman_external_site", "abbottcareers"), so
# removing them recovers a usable name from an otherwise opaque slug.
_PATH_NOISE = frozenset(
    {
        "external", "internal", "site", "sites", "career", "careers",
        "job", "jobs", "search", "professional", "experienced", "campus",
        "student", "students", "global", "us", "usa", "corporate", "main",
        "portal", "candidate", "candidates", "apply", "hiring", "talent",
        "recruiting", "general", "central", "east", "west", "north",
        "south", "domestic", "subsidiary", "new", "prod", "live", "tap",
        "gateway", "home", "default", "opportunities", "erp", "cx", "ext",
    }
)


def _slug_path_name(slug: str) -> str:
    """Company name implied by a Workday-style board path, if any.

    ``ngc/northrop_grumman_external_site`` yields "northrop grumman" —
    the only place that tenant's real identity appears, since both its
    directory name and its tenant id are the acronym "Ngc".
    """
    parts = (slug or "").split("?")[0].split("/")[1:]
    words: list[str] = []
    for token in re.split(r"[^a-z0-9]+", " ".join(parts).lower()):
        if token and token not in _PATH_NOISE and not token.isdigit():
            words.append(token)
    return " ".join(words)


# A slug-derived single token this short is an acronym or a stub
# ("ngc", "bah", "cw"); scoring against it invents matches rather than
# finding them. The directory name is exempt — plenty of real companies
# are called Etsy or Zoom.
_MIN_WEAK_VARIANT_LEN = 6


def declared_names(row: dict[str, object]) -> list[str]:
    """Core names the directory and the jobs feed actually assert."""
    out: list[str] = []
    for key in ("name", "jobs_company", "display_name"):
        core = core_name(str(row.get(key) or ""))
        if core and core not in out:
            out.append(core)
    return out


def name_variants(row: dict[str, object]) -> list[str]:
    """Every string that plausibly names this tenant, in ``core`` form.

    A tenant describes itself in several places and any one of them can
    be the only usable one, so all are kept and a candidate scores
    against its best. The three declared names are trusted as-is; the two
    recovered from the slug are guesses and must clear a length bar.

    Run-together names get a suffix-stripped form as well, since
    ``core_name`` cannot tokenise them: without it
    ``columbiasportswearcompany`` only ever scores 82 against EDGAR's
    "COLUMBIA SPORTSWEAR CO", below the accept threshold, while the
    stripped ``columbiasportswear`` scores 97.
    """
    slug = str(row.get("slug") or "")
    guessed = (_first_slug_segment(slug), _slug_path_name(slug))

    out = declared_names(row)
    for text in guessed:
        core = core_name(text)
        if not core or core in out:
            continue
        if len(core) < _MIN_WEAK_VARIANT_LEN and " " not in core:
            continue
        out.append(core)

    for core in list(out):
        if " " in core:
            continue
        for stripped in desuffix_concatenated(core):
            if len(stripped) >= _MIN_WEAK_VARIANT_LEN and stripped not in out:
                out.append(stripped)
    return out


def candidate_keys(row: dict[str, object]) -> list[str]:
    """Blocking keys for one tenant, strongest first.

    Both suffixed and suffix-stripped forms are emitted because the
    reference tables store names literally — PDL has "acme robotics" and
    EDGAR has "ACME ROBOTICS, INC.", and only one of the two variants
    will hit each.
    """
    sources = [
        str(row.get("name") or ""),
        str(row.get("jobs_company") or ""),
        _first_slug_segment(str(row.get("slug") or "")),
        _slug_path_name(str(row.get("slug") or "")),
    ]
    keys: list[str] = []

    def add(key: str) -> None:
        if len(key) >= _MIN_KEY_LEN and key not in _STOP_KEYS and key not in keys:
            keys.append(key)

    for text in sources:
        if not text:
            continue
        for keep in (True, False):
            key = name_key(text, keep_suffix=keep)
            add(key)
            # The suffix stripper inside `name_key` is token-based, so a
            # run-together slug such as `columbiasportswearcompany` still
            # carries its corporate form and misses the reference row by
            # exactly one word.
            for stripped in desuffix_concatenated(key):
                add(stripped)
    return keys


_KEY_SOURCE_COLUMNS = ("ats", "slug", "name", "jobs_company", "display_name")


def _explode_keys(cohort: pl.DataFrame) -> pl.DataFrame:
    rows = cohort.select(_KEY_SOURCE_COLUMNS).to_dicts()
    pairs = [
        {"ats": r["ats"], "slug": r["slug"], "key": key}
        for r in rows
        for key in candidate_keys(r)
    ]
    return pl.DataFrame(pairs, schema={"ats": pl.String, "slug": pl.String, "key": pl.String})


def _cohort_variants(cohort: pl.DataFrame) -> dict[tuple[str, str], list[str]]:
    return {
        (r["ats"], r["slug"]): name_variants(r)
        for r in cohort.select(_KEY_SOURCE_COLUMNS).to_dicts()
    }


def _join_on_keys(
    keys: pl.DataFrame, reference: pl.DataFrame, payload: list[str]
) -> pl.DataFrame:
    """Join tenant keys against both key forms of a reference table.

    A tenant key can land on the reference row's suffixed form, its
    stripped form, or both; the union is deduplicated so a row matched
    twice is not scored twice.
    """
    frames = []
    for key_col in ("name_key_raw", "name_key_core"):
        if key_col not in reference.columns:
            continue
        right = reference.filter(pl.col(key_col) != "").select([key_col, *payload])
        frames.append(
            keys.join(right, left_on="key", right_on=key_col, how="inner")
        )
    if not frames:
        return keys.head(0)
    return pl.concat(frames, how="vertical").unique()


def _cohort_domains(cohort: pl.DataFrame) -> pl.DataFrame:
    """Domains known before any network call."""
    return (
        cohort.with_columns(
            pl.when(pl.col("source_domain") != "")
            .then(pl.col("source_domain"))
            .otherwise(pl.col("url"))
            .map_elements(corporate_domain, return_dtype=pl.String)
            .alias("domain_hint")
        )
        .filter(pl.col("domain_hint") != "")
        .select("ats", "slug", "domain_hint")
    )


def _pair_score(left: str, right: str) -> float:
    """Similarity of two core names.

    ``token_sort`` handles word-order differences ("Acme Robotics" vs
    "Robotics, Acme"). ``token_set`` is discounted because it ignores
    extra tokens entirely, which would rate "Acme" and "Acme Bank of
    Delaware" a perfect match.
    """
    if not left or not right:
        return 0.0
    return max(
        fuzz.token_sort_ratio(left, right),
        fuzz.token_set_ratio(left, right) * 0.98,
    )


def _best_by_score(
    candidates: pl.DataFrame,
    variants: dict[tuple[str, str], list[str]],
    name_col: str,
    *,
    method: str,
) -> pl.DataFrame:
    """Score every candidate against its tenant and keep the best one.

    A candidate is scored against each of the tenant's name variants and
    keeps its best. ``matched_variant`` records which one won, so a match
    that rests only on a slug-derived name is visible downstream rather
    than being indistinguishable from a directory-name match.
    """
    if candidates.is_empty():
        return candidates.with_columns(
            pl.lit(0.0).alias("match_score"),
            pl.lit(method).alias("match_method"),
            pl.lit("").alias("matched_variant"),
        )

    scores: list[float] = []
    winners: list[str] = []
    cache: dict[tuple[str, str], float] = {}
    for ats, slug, other in zip(
        candidates["ats"], candidates["slug"], candidates[name_col], strict=True
    ):
        right = core_name(other or "")
        best, best_variant = 0.0, ""
        for left in variants.get((ats, slug), ()):
            key = (left, right)
            score = cache.get(key)
            if score is None:
                score = _pair_score(left, right)
                cache[key] = score
            if score > best:
                best, best_variant = score, left
        scores.append(best)
        winners.append(best_variant)

    return candidates.with_columns(
        pl.Series("match_score", scores, dtype=pl.Float64),
        pl.lit(method).alias("match_method"),
        pl.Series("matched_variant", winners, dtype=pl.String),
    )


def _confidence_expr(score_col: str, name_col: str, *, public_col: str | None = None) -> pl.Expr:
    """Grade a match beyond its raw similarity score.

    A perfect string score means little for a single-token name: EDGAR
    has a "TEMPO, LLC" and a "GALVANIZE, LLC" that will match any tenant
    called Tempo or Galvanize at 100, and nothing in the name alone can
    separate the real company from an unrelated filer. Multi-token names
    and listed registrants carry far more information, so they grade
    high while bare single-token private matches are marked low and sent
    for review.
    """
    tokens = pl.col(name_col).str.strip_chars().str.split(" ").list.len()
    distinctive = tokens >= 2
    if public_col is not None:
        distinctive = distinctive | pl.col(public_col).fill_null(False)
    return (
        pl.when((pl.col(score_col) >= 95) & distinctive)
        .then(pl.lit("high"))
        .when((pl.col(score_col) >= config.MATCH_ACCEPT_SCORE) & distinctive)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
    )


def _pick_best(df: pl.DataFrame, *, prefer: list[pl.Expr] | None = None) -> pl.DataFrame:
    """One winning row per (ats, slug).

    ``nulls_last`` matters more than it looks: PDL carries several rows
    per famous name, typically one complete record plus dormant
    duplicates with no website and no headcount. Under Polars' default
    null ordering those empty rows sort ahead of the real one on a
    descending tiebreak, which silently resolved Boeing and CVS Health to
    a shell with no domain and a "51-200" band.
    """
    if df.is_empty():
        return df
    order = [pl.col("match_score"), *(prefer or [])]
    return (
        df.sort(by=order, descending=True, nulls_last=True)
        .unique(subset=["ats", "slug"], keep="first")
    )


def resolve_pdl(cohort: pl.DataFrame, pdl: pl.DataFrame) -> pl.DataFrame:
    """Match the cohort to PDL by domain, then by name."""
    variants = _cohort_variants(cohort)

    pdl_cols = [
        "name",
        "domain",
        "linkedin_url",
        "size",
        "size_midpoint",
        "founded",
        "locality",
        "region",
        "industry",
    ]

    # --- domain track --------------------------------------------------
    hints = _cohort_domains(cohort)
    by_domain = (
        hints.join(
            pdl.filter(pl.col("domain") != "").select(pdl_cols),
            left_on="domain_hint",
            right_on="domain",
            how="inner",
        )
        .rename({"name": "pdl_name"})
        .with_columns(
            pl.col("domain_hint").alias("domain"),
            pl.lit(100.0).alias("match_score"),
            pl.lit("pdl_domain").alias("match_method"),
            pl.lit("").alias("matched_variant"),
        )
    )
    if not by_domain.is_empty():
        # A domain can appear on several PDL rows (subsidiaries sharing a
        # site); prefer the one with the largest headcount band.
        by_domain = by_domain.sort(
            ["size_midpoint"], descending=True, nulls_last=True
        ).unique(subset=["ats", "slug"], keep="first")

    # --- name track ----------------------------------------------------
    keys = _explode_keys(cohort)
    by_name = _join_on_keys(keys, pdl, pdl_cols).rename({"name": "pdl_name"})

    # A PDL row with no website is a dormant duplicate, and the name
    # track is where they do damage: several sit under a famous name and
    # one of them wins on a perfect score, bringing a "1-10" band with
    # it. Every row keyed `columbia` is such a shell, which is how a
    # Columbia Sportswear board came to be a ten-person company. The
    # domain track is unaffected — it matched on the website itself.
    stubs = by_name.filter(pl.col("domain").fill_null("") == "").height
    if stubs:
        logger.info("dropped %d websiteless PDL name candidates", stubs)
    by_name = by_name.filter(pl.col("domain").fill_null("") != "")

    # Generic keys can fan out to thousands of PDL rows. Scoring those is
    # both slow and pointless — no single winner is defensible.
    fanout = by_name.group_by(["ats", "slug", "key"]).agg(pl.len().alias("n"))
    safe = fanout.filter(pl.col("n") <= 200).select("ats", "slug", "key")
    dropped = fanout.filter(pl.col("n") > 200)
    if dropped.height:
        logger.info("dropped %d over-broad PDL key blocks (>200 hits)", dropped.height)
    by_name = by_name.join(safe, on=["ats", "slug", "key"], how="semi")

    by_name = _best_by_score(by_name, variants, "pdl_name", method="pdl_name")
    by_name = by_name.filter(pl.col("match_score") >= config.MATCH_REVIEW_SCORE)
    # Every surviving candidate has a domain, so that is no longer a
    # useful tiebreak here.
    by_name = _pick_best(
        by_name,
        prefer=[
            pl.col("linkedin_url").is_not_null().cast(pl.Int8),
            pl.col("size_midpoint").fill_null(0),
        ],
    )

    # Domain matches outrank name matches for the same tenant.
    resolved_domain_keys = by_domain.select("ats", "slug")
    by_name = by_name.join(resolved_domain_keys, on=["ats", "slug"], how="anti")

    combined = pl.concat(
        [
            by_domain.select(
                "ats", "slug", "pdl_name", "domain", "linkedin_url", "size",
                "size_midpoint", "founded", "locality", "region", "industry",
                "match_score", "match_method", "matched_variant",
            ),
            by_name.select(
                "ats", "slug", "pdl_name", "domain", "linkedin_url", "size",
                "size_midpoint", "founded", "locality", "region", "industry",
                "match_score", "match_method", "matched_variant",
            ),
        ],
        how="vertical",
    )
    return combined


def resolve_edgar(cohort: pl.DataFrame, edgar: pl.DataFrame) -> pl.DataFrame:
    """Match the cohort to an EDGAR CIK by name.

    Listed registrants are preferred on ties: a tenant that matches both
    a public filer and some dormant shell of the same name is far more
    likely to be the public one, and a wrong CIK on a private company
    only costs a missing Form D history.
    """
    variants = _cohort_variants(cohort)
    keys = _explode_keys(cohort)
    candidates = _join_on_keys(
        keys, edgar, ["name", "cik", "ticker", "exchange", "is_public"]
    ).rename({"name": "edgar_name"})

    fanout = candidates.group_by(["ats", "slug", "key"]).agg(pl.len().alias("n"))
    safe = fanout.filter(pl.col("n") <= 200).select("ats", "slug", "key")
    candidates = candidates.join(safe, on=["ats", "slug", "key"], how="semi")

    scored = _best_by_score(candidates, variants, "edgar_name", method="edgar_name")
    scored = scored.filter(pl.col("match_score") >= config.MATCH_REVIEW_SCORE)
    best = _pick_best(scored, prefer=[pl.col("is_public").cast(pl.Int8)])
    if best.is_empty():
        return best.with_columns(pl.lit("low").alias("match_confidence"))
    return best.with_columns(
        _confidence_expr(
            "match_score", "matched_variant", public_col="is_public"
        ).alias("match_confidence")
    )


def _mark_slug_only_matches(
    resolved: pl.DataFrame, cohort: pl.DataFrame
) -> pl.DataFrame:
    """Flag matches that rest only on a name guessed from the slug.

    Reading a company name out of the board path is what recovers
    Northrop Grumman from the tenant "Ngc", but the same trick can latch
    onto something unrelated: an iCIMS board at ``careers-vanguard`` whose
    directory name is "Deerfield Management Companies" picked up
    ``vanguard.com`` because the slug matched The Vanguard Group. When
    the winning name is not one the directory or the jobs feed actually
    asserts, that is worth saying out loud.
    """
    declared = {
        (r["ats"], r["slug"]): set(declared_names(r))
        for r in cohort.select(_KEY_SOURCE_COLUMNS).to_dicts()
    }

    def slug_only(row: dict[str, object]) -> bool:
        variant = str(row.get("matched_variant") or "")
        if not variant:
            return False
        return variant not in declared.get(
            (str(row["ats"]), str(row["slug"])), set()
        )

    return resolved.with_columns(
        pl.struct("ats", "slug", "matched_variant")
        .map_elements(slug_only, return_dtype=pl.Boolean)
        .alias("matched_on_slug_only")
    )


def run(*, use_board_fallback: bool = True) -> pl.DataFrame:
    config.ensure_dirs()
    cohort = pl.read_parquet(config.COHORT_PARQUET)
    pdl = pl.read_parquet(config.PDL_PARQUET)
    edgar = pl.read_parquet(config.EDGAR_PARQUET)
    logger.info(
        "resolving %d tenants against %d PDL rows and %d EDGAR names",
        cohort.height,
        pdl.height,
        edgar.height,
    )

    pdl_matches = resolve_pdl(cohort, pdl)
    edgar_matches = resolve_edgar(cohort, edgar)

    if use_board_fallback:
        from pipeline.company_enrichment.boardsite import recover_domains

        missing = cohort.join(
            pdl_matches.filter(pl.col("match_score") >= config.MATCH_ACCEPT_SCORE)
            .select("ats", "slug"),
            on=["ats", "slug"],
            how="anti",
        )
        recovered = recover_domains(missing)
        if recovered.height:
            extra = (
                recovered.join(
                    pdl.filter(pl.col("domain") != ""),
                    left_on="domain_hint",
                    right_on="domain",
                    how="inner",
                )
                .rename({"name": "pdl_name"})
                .with_columns(
                    pl.col("domain_hint").alias("domain"),
                    pl.lit(99.0).alias("match_score"),
                    pl.lit("board_domain").alias("match_method"),
                    pl.lit("").alias("matched_variant"),
                )
                .sort("size_midpoint", descending=True, nulls_last=True)
                .unique(subset=["ats", "slug"], keep="first")
                .select(pdl_matches.columns)
            )
            # Only fill tenants the primary tracks left unresolved.
            extra = extra.join(
                pdl_matches.filter(
                    pl.col("match_score") >= config.MATCH_ACCEPT_SCORE
                ).select("ats", "slug"),
                on=["ats", "slug"],
                how="anti",
            )
            pdl_matches = pl.concat(
                [pdl_matches.join(extra.select("ats", "slug"), on=["ats", "slug"], how="anti"), extra],
                how="vertical",
            )
            logger.info("board-site fallback resolved %d extra domains", extra.height)

    accepted_pdl = pdl_matches.filter(pl.col("match_score") >= config.MATCH_ACCEPT_SCORE)
    accepted_edgar = edgar_matches.filter(
        pl.col("match_score") >= config.MATCH_ACCEPT_SCORE
    )

    resolved = (
        cohort.select(
            "ats", "slug", "name", "display_name", "jobs_company", "url",
            "postings", "join_method",
        )
        .join(
            accepted_pdl.select(
                "ats", "slug", "pdl_name", "domain", "linkedin_url", "size",
                "size_midpoint", "founded", "locality", "region", "industry",
                "matched_variant",
                pl.col("match_score").alias("pdl_score"),
                pl.col("match_method").alias("pdl_method"),
            ),
            on=["ats", "slug"],
            how="left",
        )
        .join(
            accepted_edgar.select(
                "ats", "slug", "edgar_name", "cik", "ticker", "exchange", "is_public",
                pl.col("match_score").alias("edgar_score"),
                pl.col("match_method").alias("edgar_method"),
                pl.col("match_confidence").alias("cik_confidence"),
            ),
            on=["ats", "slug"],
            how="left",
        )
        .with_columns(
            pl.col("is_public").fill_null(False),
            pl.coalesce(pl.col("edgar_name"), pl.col("pdl_name"), pl.col("display_name"))
            .alias("resolved_name"),
        )
    )
    resolved = _mark_slug_only_matches(resolved, cohort)

    # Review file: everything that scored in the grey band, plus tenants
    # with no match at all, ordered so the highest-traffic ones surface
    # first.
    grey_pdl = pdl_matches.filter(
        (pl.col("match_score") >= config.MATCH_REVIEW_SCORE)
        & (pl.col("match_score") < config.MATCH_ACCEPT_SCORE)
    ).select(
        "ats", "slug", pl.col("pdl_name").alias("candidate"),
        pl.col("match_score"), pl.lit("pdl").alias("source"), "matched_variant",
    )
    # Low-confidence CIKs are accepted into the output but still queued
    # for review — a wrong CIK silently attaches another company's
    # funding history, which is worse than a missing one.
    grey_edgar = edgar_matches.filter(
        (
            (pl.col("match_score") >= config.MATCH_REVIEW_SCORE)
            & (pl.col("match_score") < config.MATCH_ACCEPT_SCORE)
        )
        | (pl.col("match_confidence") == "low")
    ).select(
        "ats", "slug", pl.col("edgar_name").alias("candidate"),
        pl.col("match_score"), pl.lit("edgar").alias("source"), "matched_variant",
    )
    review = (
        pl.concat([grey_pdl, grey_edgar], how="vertical")
        .join(cohort.select("ats", "slug", "name", "url", "postings"), on=["ats", "slug"])
        .sort("postings", descending=True)
        .select(
            "ats", "slug", "name", "matched_variant", "candidate",
            "match_score", "source", "postings", "url",
        )
    )

    resolved.write_parquet(config.RESOLVED_PARQUET)
    review.write_csv(config.RESOLVE_REVIEW_CSV)

    n = resolved.height
    with_domain = resolved.filter(pl.col("domain").is_not_null() & (pl.col("domain") != "")).height
    with_cik = resolved.filter(pl.col("cik").is_not_null()).height
    with_ticker = resolved.filter(pl.col("ticker").is_not_null()).height
    with_size = resolved.filter(pl.col("size").is_not_null()).height

    print("\n=== Stage 1: entity resolution ===")
    print(f"Cohort tenants           : {n:>6,}")
    print(f"Resolved to a domain     : {with_domain:>6,}  ({with_domain / n:.1%})")
    print(f"Resolved to a CIK        : {with_cik:>6,}  ({with_cik / n:.1%})")
    print(f"  of which listed        : {with_ticker:>6,}  ({with_ticker / n:.1%})")
    print("\nCIK match confidence:")
    print(
        resolved.filter(pl.col("cik").is_not_null())
        .group_by("cik_confidence")
        .agg(pl.len().alias("tenants"))
        .sort("tenants", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    print(f"With PDL size band       : {with_size:>6,}  ({with_size / n:.1%})")
    slug_only = resolved.filter(pl.col("matched_on_slug_only")).height
    print(f"Matched on slug name only: {slug_only:>6,}  ({slug_only / n:.1%})")
    print(f"Grey-band review rows    : {review.height:>6,}")
    print("\nPDL match methods:")
    print(
        resolved.filter(pl.col("pdl_method").is_not_null())
        .group_by("pdl_method")
        .agg(pl.len().alias("tenants"))
        .sort("tenants", descending=True)
        .to_pandas()
        .to_string(index=False)
    )
    print(f"\nwrote {config.RESOLVED_PARQUET}")
    print(f"wrote {config.RESOLVE_REVIEW_CSV}")
    return resolved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()

"""CLI for the company enrichment pipeline.

Stages are ordered and mostly incremental — each reads the previous
stage's parquet from the private output directory, and every expensive
download is cached, so re-running a later stage is cheap.

    python -m pipeline.company_enrichment all
    python -m pipeline.company_enrichment resolve --no-board-fallback
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

STAGES: dict[str, str] = {
    "cohort": "Stage 0: select US pay-transparent tenants",
    "ingest-pdl": "Download the PDL free company dataset (US slice)",
    "ingest-edgar": "Download SEC CIK/ticker identity files",
    "ingest-exempt": "Download Reg CF (Form C) and Reg A datasets",
    "ingest-osha": "Download OSHA ITA Form 300A establishment employment",
    "ingest-5500": "Download DOL Form 5500 plan-sponsor employment",
    "ingest-boilerplate": "Mine company copy repeated across each tenant's postings",
    "resolve": "Stage 1: resolve tenants to CIK and domain",
    "registrant": "EIN, city, state, and SIC from EDGAR for every matched CIK",
    "marketcap": "Market cap for public tenants",
    "formd": "Funding history from Form D / Reg CF / Reg A",
    "teamsize": "Employee count and size band",
    "profile": "Company description, careers URLs, and headquarters",
    "assemble": "Build company_enrichment.parquet",
    "validate": "Report coverage, consistency, and hand-checked precision",
}

_ORDER = (
    "cohort",
    "ingest-pdl",
    "ingest-edgar",
    "ingest-exempt",
    "ingest-osha",
    "ingest-5500",
    "ingest-boilerplate",
    "resolve",
    "registrant",
    "marketcap",
    "formd",
    "teamsize",
    "profile",
    "assemble",
    "validate",
)


def _runner(stage: str, args: argparse.Namespace) -> Callable[[], object]:
    from pipeline.company_enrichment import (
        assemble,
        boilerplate,
        cohort,
        edgar,
        exempt,
        form5500,
        formd,
        marketcap,
        osha,
        pdl,
        profile,
        registrant,
        resolve,
        teamsize,
        validate,
    )

    match stage:
        case "cohort":
            return cohort.run
        case "ingest-pdl":
            return lambda: pdl.run(force=args.force)
        case "ingest-edgar":
            return lambda: edgar.run(force=args.force)
        case "ingest-exempt":
            return exempt.run
        case "ingest-osha":
            return lambda: osha.run(force=args.force)
        case "ingest-5500":
            return lambda: form5500.run(force=args.force)
        case "ingest-boilerplate":
            return lambda: boilerplate.run(force=args.force)
        case "resolve":
            return lambda: resolve.run(use_board_fallback=args.board_fallback)
        case "registrant":
            return lambda: registrant.run(force=args.force)
        case "marketcap":
            return lambda: marketcap.run(use_exhibit21=args.exhibit21)
        case "formd":
            return formd.run
        case "teamsize":
            return lambda: teamsize.run(use_10k=args.ten_k)
        case "profile":
            return lambda: profile.run(
                use_site_crawl=args.site_crawl, use_10k=args.ten_k
            )
        case "assemble":
            return assemble.run
        case "validate":
            return validate.run
        case _:
            raise SystemExit(f"unknown stage {stage!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.company_enrichment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=[*_ORDER, "all"],
        help="\n".join(f"{name}: {desc}" for name, desc in STAGES.items()),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "rebuild a cached ingest even if it looks current; required "
            "after changing normalize.py, since the stored blocking keys "
            "would otherwise be stale"
        ),
    )
    parser.add_argument(
        "--no-board-fallback",
        dest="board_fallback",
        action="store_false",
        help="skip fetching ATS board pages to recover domains (resolve)",
    )
    parser.add_argument(
        "--no-exhibit21",
        dest="exhibit21",
        action="store_false",
        help="skip Exhibit 21 subsidiary rollup (marketcap)",
    )
    parser.add_argument(
        "--no-10k",
        dest="ten_k",
        action="store_false",
        help="skip reading 10-K filings (teamsize headcount, profile description)",
    )
    parser.add_argument(
        "--no-site-crawl",
        dest="site_crawl",
        action="store_false",
        help=(
            "skip fetching company homepages for descriptions and careers "
            "links (profile); whatever is already cached is still used"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    # httpx logs a line per request; at ~1k requests that buries our own.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    stages = _ORDER if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n{'=' * 70}\n{stage}: {STAGES[stage]}\n{'=' * 70}")
        _runner(stage, args)()
    return 0


if __name__ == "__main__":
    sys.exit(main())

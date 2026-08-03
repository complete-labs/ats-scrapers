"""Ad-hoc harness: parse_headcount against known-answer 10-K filings.

Not a unit test — it needs the network and the answers move every year.
It exists so a change to the parser can be judged against reality
instead of against whether the pipeline still runs.
"""

from __future__ import annotations

import logging

from pipeline.company_enrichment import sechttp, teamsize

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# (label, CIK, expected headcount, tolerance as a fraction). ``None``
# means the filing states no company-wide total in parseable prose, so
# the only right answer is to return nothing. Copart gives only
# demographic subsets and SmartFinancial gives no figure at all; both
# used to yield a confident wrong number.
CASES: tuple[tuple[str, int, int | None, float], ...] = (
    ("Boeing", 12927, 172000, 0.25),
    ("Cintas", 723254, 48300, 0.15),
    ("Leidos", 1336920, 48000, 0.25),
    ("CVS Health", 64803, 300000, 0.20),
    ("Northrop Grumman", 1133421, 95000, 0.20),
    ("Tandem Diabetes", 1438133, 2500, 0.25),
    ("Walmart", 104169, 2100000, 0.20),
    ("Intel", 50863, 85100, 0.15),
    ("Bumble", 1830043, 580, 0.30),
    ("LendingTree", 1434621, 900, 0.40),
    ("ResMed", 943819, 10600, 0.20),
    ("Copart", 900075, None, 0.0),
    ("SmartFinancial", 1038773, None, 0.0),
)


def main() -> None:
    ok = 0
    checked = 0
    for label, cik, expected, tol in CASES:
        got = teamsize._latest_10k(cik)
        if not got:
            print(f"{label:20} no 10-K found")
            continue
        url, _date = got
        html = sechttp.get(url, suffix=".htm").decode("utf-8", "replace")
        value = teamsize.parse_headcount(html)
        checked += 1

        if expected is None:
            verdict = "ok  " if value is None else "WRONG"
            shown = "none" if value is None else f"{value:,}"
            print(f"{label:20} {verdict} got {shown:>9}  expected none")
        elif value is None:
            verdict = "MISS "
            print(f"{label:20} {verdict} expected ~{expected:,}")
        else:
            low, high = expected * (1 - tol), expected * (1 + tol)
            verdict = "ok  " if low <= value <= high else "WRONG"
            print(f"{label:20} {verdict} got {value:>9,}  expected ~{expected:,}")
        ok += verdict == "ok  "
    print(f"\n{ok}/{checked} correct")


if __name__ == "__main__":
    main()

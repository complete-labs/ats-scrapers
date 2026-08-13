"""Gold sets for evaluating enrichment quality.

Two complementary sources, because neither alone is sufficient:

**Hand-written edge cases** (``edge_cases.jsonl``, committed). Forty-five
cases written against the specific failure modes this pipeline can have:
equity ranges read as salary, a named city read as on-site, "15 years of
growth" read as an experience requirement, cookie banners read as job text,
non-USD currencies defaulted to USD. Crucially these carry *negative*
labels — an explicit ``null`` meaning "the correct answer is to abstain" —
which is the only way to measure over-claiming. A provider-labelled set
cannot measure that at all, because providers only label what they know.

**Provider-labelled holdout** (``data/gold/holdout.jsonl``, regenerable).
Rows where the ATS itself shipped a structured value. That value is real
ground truth, and there are millions of rows carrying one. The catch is
leakage: if the label is visible in the prompt, the eval measures nothing.
So each example records which prompt inputs must be *masked*, and the
harness withholds them. For a salary label that means hiding
``salary_summary``; for employment type, hiding ``commitment`` and the
provider's own ``employment_type``.

Known limitations, stated plainly because they bound what the scores mean:

* Provider labels are themselves imperfect. Upstream's ``is_remote`` is
  partly derived from title keywords by the publisher, so some of those
  labels are a heuristic's output rather than an employer's statement.
* The holdout is biased toward providers that ship structured fields, which
  are not a random sample of the corpus.
* ``placement`` labels split into "trivial" (a remote token appears in the
  title or location, so any keyword matcher wins) and "non-trivial". Only
  the second number says anything about the model, so the harness reports
  them separately.
"""

from enrich.gold.loader import (
    GoldExample,
    load_edge_cases,
    load_gold,
    load_holdout,
)

__all__ = ["GoldExample", "load_edge_cases", "load_gold", "load_holdout"]

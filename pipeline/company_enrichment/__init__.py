"""Company enrichment: what a company is, does, and is worth.

Augments the ats-scrapers companies directory with firmographic data
drawn exclusively from **free, non-Crunchbase** sources:

- SEC EDGAR (public domain) — CIK/ticker identity, Form D private
  offerings, XBRL company facts, 10-K human-capital headcount, the
  opening of Item 1 Business, and the registrant's EIN and address.
- People Data Labs free company dataset (CC BY 4.0) — domain,
  LinkedIn URL, employee-count band, locality.
- The companies' own homepages — the ``og:description`` a company
  wrote to introduce itself, and the careers link beside it.
- The companies' own job postings — the copy a tenant repeats at the
  top of every posting, which is already in the jobs snapshot.
- DOL EFAST2 Form 5500 (public domain) — benefit-plan active
  participants, a filed lower bound on headcount that reaches private
  companies. Joined on EIN, the only exact cross-source key here.
- OSHA Injury Tracking Application (public domain) — establishment
  employment, giving a filed lower bound on headcount.
- Wikidata (CC0) — point-in-time employee counts, and one-line
  descriptions, for notable firms.
- Nasdaq screener + Stooq — market capitalisation for US listings.

Funding and market cap are treated as alternatives, not companions: a
listed company's Form D history is withdrawn, because its pre-IPO
raises sitting beside a market capitalisation invite being read as its
funding position.

Scope is deliberately narrow: **US companies with at least one
pay-transparent posting**. That cohort is where the free sources have
real coverage (SEC filings are US-only; US pay-transparency laws make
salary-bearing postings a good proxy for a US-operating employer).

Outputs are **private**. Nothing here feeds the public R2 publish path
in ``.github/scripts/publish_companies.py`` — the published
``companies.{csv,parquet}`` schema stays ``ats,name,slug,url``. Results
land in a gitignored directory (see :mod:`.config`).

Run the stages in order via the CLI::

    python -m pipeline.company_enrichment cohort
    python -m pipeline.company_enrichment ingest-pdl
    python -m pipeline.company_enrichment ingest-edgar
    python -m pipeline.company_enrichment ingest-osha
    python -m pipeline.company_enrichment ingest-5500
    python -m pipeline.company_enrichment ingest-boilerplate
    python -m pipeline.company_enrichment resolve
    python -m pipeline.company_enrichment registrant
    python -m pipeline.company_enrichment marketcap
    python -m pipeline.company_enrichment formd
    python -m pipeline.company_enrichment teamsize
    python -m pipeline.company_enrichment profile
    python -m pipeline.company_enrichment assemble

Or end-to-end with ``python -m pipeline.company_enrichment all``.

Accuracy envelope
-----------------
Measured 2026-07-30 over a cohort of 9,541 tenants. Coverage and
cross-source consistency are computed over the whole table; precision
comes from a stratified 50-row sample checked by hand, recorded in
``validation_labels.csv`` and reproducible with
``python -m pipeline.company_enrichment.validate``.

The verdicts are keyed on ``(ats, slug)`` and scored against whatever the
table currently says, so they survive a refresh of the upstream jobs
dataset — the sample chose what to review, but each verdict then stands
on its own.

Coverage — what fraction of tenants carry a value at all:

===========================  =======  =======
field                          rows        %
===========================  =======  =======
domain                         6,556    68.7
linkedin_url                   6,556    68.7
cik                            2,621    27.5
cik, high confidence           1,133    11.9
market_cap_usd                   625     6.6
funding history                1,303    13.7
employee_count, exact            717     7.5
employee_count_floor           2,870    30.1
employee_count_band            6,346    66.5
company_description            5,857    61.4
careers_url                    9,541   100.0
company_careers_url            3,279    34.4
headquarters                   6,794    71.2
any of the three requested     6,684    70.1
===========================  =======  =======

Band, count, and funding coverage are all lower than a naive run would
give, on purpose. 210 bands (2.2%) are withdrawn as contradicted by an
independent source, 11 exact headcounts are withdrawn as refuted by a
filed floor, and every public company's funding history is withdrawn by
design; see :mod:`.teamsize` and :mod:`.assemble`.

The floor is the column that moved most, 4.7% to 30.1%, because Form 5500
reaches private employers that no securities filing does. It also pays
for itself as a check rather than only as coverage: an independently
filed floor is the only thing here that can catch a bad exact count, and
it caught 22 — Ameren published at 55 staff against 9,197 filed, Swift at
200 against 31,022.

Both of those were one 10-K parser bug. It scored the words following a
number but not the words leading it, so a figure scoped to a department,
a layoff, or one line of a breakdown read as the company total. Reading
the leading side too cut the counts a filed floor can prove wrong from
7.6% to 4.9% of those checkable, and the ones wrong by more than double
from 4.5% to 1.4%, at a cost of 3 of 473 values. Half the withdrawals
went away because the number is now parsed correctly instead of being
dropped, which is why exact coverage went up while errors went down.

``company_description`` is the one quoted column: the company's own
sentence about itself rather than a computed figure. It comes from four
sources, and which one won matters more than the total — a homepage
blurb (3,698) and posting copy (1,816) are the company speaking, a 10-K
opening (234) is the company filing, and a Wikidata line (109) is a
third party. 77% of them name the company they are attached to, which is
the only automatic check prose admits of, and the rest are flagged
``description_name_unconfirmed`` for review. Two guards cost coverage on
purpose: a description reached by an already-shaky match that never
names the company is dropped so a weaker but tenant-tied source can be
used instead, and Wikidata is matched on CIK and domain only, never on a
name key, because nothing downstream could catch the collision.

The two careers columns answer different questions. ``careers_url`` is
the ATS board, at 100%, and 98.5% of those re-resolve to the tenant they
are filed under; the 141 that do not are custom-domain careers sites no
URL-shape rule can confirm, flagged rather than dropped.
``company_careers_url`` is the company's own careers page, which only
exists where the homepage crawl found one.

Resolution precision — is the row about the right company:

======================  ========  =========
stratum                 reviewed  precision
======================  ========  =========
funded_private                12       100%
public_listed                 12        83%
smb_band_only                 12        83%
cik_only                       8        63%
**overall**               **44**    **84%**
======================  ========  =========

Cross-source consistency, over every row where two independent sources
speak to the same fact: an exact headcount agrees with the PDL band on
571 of 645 (88%), the Nasdaq market cap agrees with SEC-filed shares
times last sale on 440 of 466 (94%), and a description names the tenant
it hangs off on 4,493 of 5,857 (77%).

Read the counts above as approximate to within a few rows. `resolve` is
not yet reproducible: its winner-picking dedups do not pin the row order
they sort into, so roughly 200 tenants land on a different CIK and 270 on
a different domain from one run to the next, which moves every count
downstream of it. See `resolve._pick_best`.

Read those numbers with three things in mind:

1. **Precision is uneven and predictable.** A tenant that resolves to a
   registrant with a ticker or a Form D history is nearly always right,
   because two independent sources had to agree. A bare CIK match on a
   short trade name is the weak case, and ``cik_only`` at 63% is the
   honest cost of accepting those at all.
2. **The failures are name collisions, not fuzzy-match slop.** Every
   error in the sample scored a perfect 100 on an exact name key: Oath
   Animal Hospital matched Verizon's OATH, INC., a Greenhouse board for
   Spire Global matched a Missouri gas utility of the same name. No score
   threshold can fix this, which is why the disambiguating signals are
   independent facts — registrant state, headcount against band.
3. **The flags are worth acting on.** 5 of the 7 bad rows in the sample
   already carried a ``quality_flags`` entry, so filtering on that column
   removes most of the error at a small cost in volume. It is not a
   substitute for review of the rows it names.
"""

from __future__ import annotations

__all__ = ["config", "normalize"]

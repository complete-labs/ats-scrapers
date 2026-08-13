"""Tier 2: bounded tool-using recovery for rows Tier 1 cannot answer.

The Phase 0 profile sizes this tier precisely: 7.3% of the corpus (roughly
354,000 rows) has a description under 200 characters, and 2.7% (131,000) has
none at all. Those rows are not a prompting problem — there is nothing to
read — so no amount of Tier 1 spend fixes them. What fixes them is going
back to the source, which is exactly what the upstream scrapers already do:
``docs/provider-description-matrix.md`` documents ~25 providers that require
a second per-job detail fetch, and every one of them implements
``get_description``.

So this is an agent in the useful sense — it chooses among tools, observes
what came back, and decides whether to keep going — while staying strictly
bounded, because the failure mode of an agent loop over millions of rows is
an unbounded bill:

* at most ``max_steps`` tool calls per row (default 3);
* a hard per-run USD budget checked before every model call;
* one model call per row after recovery, not a conversation;
* every run written to ``agent_runs`` with its steps, recovered bytes and
  outcome, so the tier can be shown to pay for itself or be switched off.

The tools are the upstream scrapers, used as a library rather than
reimplemented. That is the whole reason to work inside this repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from enrich.deterministic import run_tier0
from enrich.llm import ModelConfig, Request, resolve_client
from enrich.pipeline import merge
from enrich.prompt import prompt_hash
from enrich.snapshot import SNAPSHOT_COLUMNS, iter_rows
from enrich.store import EnrichmentStore

log = logging.getLogger(__name__)

#: A recovered body must beat this to be worth a re-extraction.
MIN_RECOVERED_CHARS = 200

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = _HTML_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


@dataclass
class ToolResult:
    tool: str
    text: str
    error: str | None = None

    @property
    def usable(self) -> bool:
        return not self.error and len(self.text) >= MIN_RECOVERED_CHARS


class RecoveryTools:
    """The tools available to Tier 2, in the order they are worth trying.

    Ordered by cost and by likelihood of returning the actual posting body:
    the provider's own detail endpoint first (cheap, structured, exact), the
    posting URL as plain HTML second, and the company's careers listing last.
    """

    def __init__(self, *, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def scraper_description(self, row: dict[str, Any]) -> ToolResult:
        """Refetch the description through the provider's own scraper.

        This is the highest-value tool: for the ~25 providers that withhold
        descriptions from their list endpoint,
        ``BaseScraper.get_description`` already knows the per-job detail
        call, its auth quirks and its parsing.

        It takes a ``Job``, not a URL, and several providers read the detail
        endpoint out of the listing payload — so the snapshot's ``raw``
        column has to be rehydrated into the model or the scraper has
        nothing to work from. Providers whose listing already carries the
        description return it unchanged, which for a thin row means an empty
        string and a fall-through to the next tool. That is correct: there is
        no second endpoint to try.
        """
        try:
            from ats_scrapers import get_scraper_for_url
            from ats_scrapers.models import Job

            url = str(row.get("url") or "")
            scraper = get_scraper_for_url(url)
            if scraper is None:
                return ToolResult("scraper_description", "", error="no scraper matched url")

            raw = row.get("raw")
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    raw = parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    raw = None
            elif not isinstance(raw, dict):
                raw = None

            job = Job(
                url=url,
                title=str(row.get("title") or "(unknown)"),
                company=str(row.get("company") or "(unknown)"),
                ats_type=str(row.get("ats_type") or "unknown"),
                ats_id=row.get("ats_id"),
                location=row.get("location"),
                description=row.get("description") or None,
                raw=raw,
            )
            text = scraper.get_description(job)
            return ToolResult("scraper_description", _strip_html(str(text or "")))
        except Exception as exc:
            return ToolResult("scraper_description", "", error=f"{type(exc).__name__}: {exc}"[:300])

    def fetch_page(self, url: str) -> ToolResult:
        """Plain GET of the posting URL, HTML stripped.

        Deliberately unrendered: a headless browser costs orders of
        magnitude more per row, and any provider that truly needs one is
        already handled by its own scraper above.
        """
        try:
            import httpx

            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ats-enrich/1.0)"},
            )
            if response.status_code != 200:
                return ToolResult("fetch_page", "", error=f"HTTP {response.status_code}")
            return ToolResult("fetch_page", _strip_html(response.text))
        except Exception as exc:
            return ToolResult("fetch_page", "", error=f"{type(exc).__name__}: {exc}"[:300])

    def company_careers(self, company: str) -> ToolResult:
        """Resolve the company's careers page via the upstream registry."""
        try:
            from ats_scrapers import find_company

            matches = find_company(company)
            if not matches:
                return ToolResult("company_careers", "", error="company not in registry")
            first = matches[0]
            careers = getattr(first, "careers_url", None) or getattr(first, "url", None)
            if not careers:
                return ToolResult("company_careers", "", error="no careers url")
            return self.fetch_page(str(careers))
        except Exception as exc:
            return ToolResult("company_careers", "", error=f"{type(exc).__name__}: {exc}"[:300])


@dataclass
class AgentRun:
    job_key: str
    steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    recovered_chars: int = 0
    resolved: bool = False
    outcome: str = "not_attempted"
    error: str | None = None
    cost_usd: float = 0.0


def recover_row(
    row: dict[str, Any],
    tools: RecoveryTools,
    *,
    max_steps: int = 3,
) -> tuple[str | None, AgentRun]:
    """Try to obtain a usable description for one row.

    The loop is the decision: each tool's result is inspected, and the run
    stops as soon as something usable comes back rather than trying every
    tool. That ordering is why the average row costs one fetch, not three.
    """
    run = AgentRun(job_key=str(row.get("job_key") or ""))
    url = str(row.get("url") or "")
    company = str(row.get("company") or "")

    plan: list[tuple[str, Any]] = []
    if url:
        plan.append(("scraper_description", lambda: tools.scraper_description(row)))
        plan.append(("fetch_page", lambda: tools.fetch_page(url)))
    if company:
        plan.append(("company_careers", lambda: tools.company_careers(company)))

    best: str | None = None
    for name, action in plan[:max_steps]:
        run.steps += 1
        run.tools_used.append(name)
        result = action()
        if result.error:
            run.error = result.error
            continue
        if len(result.text) > len(best or ""):
            best = result.text
        if result.usable:
            run.recovered_chars = len(result.text)
            run.outcome = f"recovered via {name}"
            return result.text, run

    run.recovered_chars = len(best or "")
    run.outcome = "no usable text recovered"
    return (best if best and len(best) >= MIN_RECOVERED_CHARS else None), run


def run_agent_recovery(args: argparse.Namespace) -> int:
    """Tier 2 over rows that Tier 1 left unresolved."""
    store = EnrichmentStore(args.db)
    client = resolve_client(args.client, concurrency=args.concurrency, model_id=args.model)
    config: ModelConfig = getattr(client, "config", ModelConfig.from_env(args.model))
    digest = prompt_hash(truncate=args.truncate)
    tools = RecoveryTools()

    # Candidates: thin or empty descriptions. These are the rows for which no
    # amount of Tier 1 spend can help, which is exactly what makes them worth
    # a fetch.
    where = f"(description IS NULL OR length(description) < {MIN_RECOVERED_CHARS})"
    # Targeting one provider matters: recovery rate varies enormously by
    # provider, and 35 of them override ``get_description`` while the rest
    # cannot be recovered by the cheap tool at all. Running Tier 2 provider by
    # provider is how you find out where it pays before spending broadly.
    if getattr(args, "ats", None):
        escaped = str(args.ats).replace("'", "''")
        where += f" AND ats_type = '{escaped}'"
    limit = args.limit or 1000
    # ``raw`` is excluded from the default projection because it is a JSON
    # blob up to ~5 kB per row, but Tier 2 needs it: several providers locate
    # their detail endpoint inside the listing payload. It is affordable here
    # because Tier 2 only ever touches the ~7% of rows with a thin body.
    columns = (*SNAPSHOT_COLUMNS, "raw")
    candidates: list[dict[str, Any]] = []
    for batch in iter_rows(
        args.snapshot,
        truncate=args.truncate,
        batch_size=min(limit, 5000),
        limit=limit,
        where=where,
        columns=columns,
    ):
        candidates.extend(batch)
        if len(candidates) >= limit:
            break
    candidates = candidates[:limit]
    log.info("tier2 candidates: %d", len(candidates))

    spent = 0.0
    recovered_rows: list[dict[str, Any]] = []
    runs: list[AgentRun] = []

    for row in candidates:
        if spent >= args.budget_usd:
            log.warning("budget of $%.2f reached; stopping", args.budget_usd)
            break
        text, run = recover_row(row, tools, max_steps=args.max_steps)
        if text is None:
            runs.append(run)
            store.record_agent_run(
                job_key=run.job_key,
                steps=run.steps,
                tools_used=run.tools_used,
                recovered_chars=run.recovered_chars,
                resolved=False,
                cost_usd=0.0,
                outcome=run.outcome,
                error=run.error,
            )
            continue
        enriched_row = dict(row)
        enriched_row["description"] = text[: args.truncate * 4]
        recovered_rows.append(enriched_row)
        runs.append(run)

    if not recovered_rows:
        print(
            json.dumps(
                {
                    "candidates": len(candidates),
                    "recovered": 0,
                    "note": "no usable descriptions recovered",
                },
                indent=2,
            )
        )
        store.close()
        return 0

    # Re-run Tier 1 on the recovered text. Content hashes are recomputed from
    # the new body, so recovered rows get their own cache entries and a
    # re-run does not refetch them.
    from enrich.snapshot import add_keys

    rekeyed = [add_keys(row, truncate=args.truncate) for row in recovered_rows]
    requests = [Request.from_row(row, truncate=args.truncate) for row in rekeyed]
    completions = client.complete(requests)
    by_hash = {c.content_hash: c.extraction for c in completions if c.ok}
    input_tokens = sum(c.input_tokens for c in completions)
    output_tokens = sum(c.output_tokens for c in completions)
    cost = config.cost(input_tokens, output_tokens)
    spent += cost

    store.cache_put_many(
        [
            (
                c.content_hash,
                c.extraction,  # type: ignore[arg-type]
                c.input_tokens,
                c.output_tokens,
                config.cost(c.input_tokens, c.output_tokens),
            )
            for c in completions
            if c.ok
        ],
        prompt_hash=digest,
        model_id=client.model_id,
    )

    built = []
    resolved_keys: set[str] = set()
    for row in rekeyed:
        extraction = by_hash.get(str(row["content_hash"]))
        tier0 = run_tier0(row)
        enriched = merge(
            row,
            tier0,
            extraction,
            tier="tier2",
            model_id=client.model_id if extraction is not None else None,
            prompt_hash=digest if extraction is not None else None,
        )
        if extraction is not None and not extraction.insufficient_text:
            resolved_keys.add(str(row["job_key"]))
        built.append(enriched)
    store.upsert(built)

    per_row_cost = cost / len(built) if built else 0.0
    for run in runs:
        resolved = run.job_key in resolved_keys
        store.record_agent_run(
            job_key=run.job_key,
            steps=run.steps,
            tools_used=run.tools_used,
            recovered_chars=run.recovered_chars,
            resolved=resolved,
            cost_usd=per_row_cost if resolved else 0.0,
            outcome=run.outcome,
            error=run.error,
        )
    store.record_cost(
        stage="tier2",
        calls=len(completions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        model_id=client.model_id,
        note=f"{len(recovered_rows)} recovered of {len(candidates)} candidates",
    )

    tool_counts: dict[str, int] = {}
    for run in runs:
        for tool in run.tools_used:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    summary = {
        "candidates": len(candidates),
        "recovered_text": len(recovered_rows),
        "recovery_rate": round(len(recovered_rows) / len(candidates), 4) if candidates else 0.0,
        "resolved_by_model": len(resolved_keys),
        "tool_calls": tool_counts,
        "cost_usd": round(cost, 4),
        "cost_per_resolved_row_usd": (
            round(cost / len(resolved_keys), 6) if resolved_keys else None
        ),
        "model_id": client.model_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    from enrich.paths import REPORT_DIR

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tier2.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    store.close()
    return 0


def summarize_runs(runs: Sequence[AgentRun]) -> dict[str, Any]:
    """Aggregate agent runs. Used by tests and by the stats command."""
    if not runs:
        return {"runs": 0}
    return {
        "runs": len(runs),
        "mean_steps": sum(run.steps for run in runs) / len(runs),
        "recovered": sum(1 for run in runs if run.recovered_chars >= MIN_RECOVERED_CHARS),
        "mean_recovered_chars": sum(run.recovered_chars for run in runs) / len(runs),
    }

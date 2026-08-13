"""Model clients: synchronous, batch, and an offline stub.

Three implementations behind one protocol:

:class:`OpenAIClient`
    Structured outputs in strict mode against any OpenAI-compatible
    endpoint. Used for the pilot, for Tier 2, and for the delta loop.

:class:`BatchClient`
    The same requests written as JSONL and submitted to the Batch API,
    which is roughly half the price and the only sane way to move 4.2M
    calls. Shards are checkpointed in the store so an interrupted backfill
    resumes instead of re-buying answers.

:class:`StubClient`
    A deterministic, offline extractor. Its purpose is **not** to imitate a
    model's quality — it is to let the entire pipeline, its tests and its
    cost accounting run end-to-end with no API key and no spend. Anything
    it produces is tagged so it can never be mistaken for real output.

Cost is computed from a table rather than read back from the provider,
because the batch endpoint does not report per-request pricing. The table
is therefore an estimate and is labelled as such wherever it surfaces.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from enrich.prompt import SYSTEM_PROMPT, estimate_tokens, render_user_prompt
from enrich.schema import Evidence, LlmExtraction, llm_json_schema

log = logging.getLogger(__name__)

STUB_MODEL_ID = "stub-deterministic-v1"


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and its price per million tokens.

    Prices are **estimates for projection only** and default to a
    small-model tier. Override with ``ATS_ENRICH_PRICE_IN`` /
    ``ATS_ENRICH_PRICE_OUT`` (USD per million tokens) to match the actual
    contracted rate before trusting a backfill projection.
    """

    model_id: str
    input_per_mtok: float = 0.15
    output_per_mtok: float = 0.60
    batch_discount: float = 0.5

    @classmethod
    def from_env(cls, model_id: str | None = None) -> ModelConfig:
        resolved = model_id or os.environ.get("ATS_ENRICH_MODEL", "gpt-4o-mini")
        if resolved == STUB_MODEL_ID:
            # The stub is free, and the price override env vars must not make
            # it look otherwise: collecting a rehearsal file would then book
            # spend that never happened.
            return cls(model_id=resolved, input_per_mtok=0.0, output_per_mtok=0.0)
        return cls(
            model_id=resolved,
            input_per_mtok=float(os.environ.get("ATS_ENRICH_PRICE_IN", "0.15")),
            output_per_mtok=float(os.environ.get("ATS_ENRICH_PRICE_OUT", "0.60")),
        )

    def cost(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        raw = (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )
        return raw * (self.batch_discount if batch else 1.0)


@dataclass
class Completion:
    """One model answer, or a failure."""

    content_hash: str
    extraction: LlmExtraction | None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.extraction is not None


@dataclass
class Request:
    """One unit of work: a unique posting body plus its cache key."""

    content_hash: str
    user_prompt: str

    @classmethod
    def from_row(cls, row: dict[str, Any], *, truncate: int) -> Request:
        return cls(
            content_hash=str(row["content_hash"]),
            user_prompt=render_user_prompt(row, truncate=truncate),
        )


class LlmClient(Protocol):
    """What the pipeline needs from a model backend."""

    model_id: str

    def complete(self, requests: Sequence[Request]) -> list[Completion]: ...


# --- OpenAI-compatible ------------------------------------------------------


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "job_enrichment",
            "strict": True,
            "schema": llm_json_schema(),
        },
    }


def _parse_content(content_hash: str, content: str, usage: Any) -> Completion:
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    try:
        extraction = LlmExtraction.model_validate_json(content)
    except Exception as exc:
        return Completion(
            content_hash=content_hash,
            extraction=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=f"schema validation failed: {exc}"[:400],
        )
    return Completion(
        content_hash=content_hash,
        extraction=extraction,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OpenAIClient:
    """Synchronous structured-output client with bounded concurrency."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        concurrency: int = 8,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.config = config or ModelConfig.from_env()
        self.model_id = self.config.model_id
        self.concurrency = concurrency
        self.max_retries = max_retries
        base_url = os.environ.get("ATS_ENRICH_BASE_URL")
        self._client = OpenAI(
            timeout=timeout,
            **({"base_url": base_url} if base_url else {}),
        )

    def _one(self, request: Request) -> Completion:
        delay = 1.0
        last_error = "unknown"
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    response_format=_response_format(),
                    temperature=0,
                )
                choice = response.choices[0]
                refusal = getattr(choice.message, "refusal", None)
                if refusal:
                    return Completion(
                        content_hash=request.content_hash,
                        extraction=None,
                        error=f"refusal: {refusal}"[:400],
                    )
                return _parse_content(
                    request.content_hash,
                    choice.message.content or "",
                    response.usage,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:400]
                if attempt == self.max_retries - 1:
                    break
                # Jitter-free exponential backoff is adequate here: the
                # concurrency ceiling already spreads retries out.
                time.sleep(delay)
                delay *= 2
        return Completion(content_hash=request.content_hash, extraction=None, error=last_error)

    def complete(self, requests: Sequence[Request]) -> list[Completion]:
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, requests))


# --- Batch API --------------------------------------------------------------


#: Batch API input-file ceilings. Files are capped at 200 MB and 50,000
#: requests; the byte budget carries a safety margin because the limit is on
#: the uploaded file, not on our estimate of it.
#:
#: The byte limit binds long before the request limit here. Each line repeats
#: the system prompt (~1.7 kB) and the full JSON schema, so a request costs
#: roughly 10 kB and 20,000 of them already approach 200 MB. Sharding on
#: count alone silently produces files the API rejects.
MAX_SHARD_BYTES = 180 * 1024 * 1024
MAX_SHARD_REQUESTS = 50_000


def batch_request_line(request: Request, *, model_id: str) -> str:
    """Serialize one Batch API request.

    ``custom_id`` is the content hash, so results reattach to the cache
    without a side table mapping request ids back to rows.
    """
    return json.dumps(
        {
            "custom_id": request.content_hash,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.user_prompt},
                ],
                "response_format": _response_format(),
                "temperature": 0,
            },
        },
        ensure_ascii=False,
    )


def write_batch_requests(
    path: str | Path,
    requests: Sequence[Request],
    *,
    model_id: str,
) -> int:
    """Write one shard of Batch API JSONL.

    Module-level rather than a method so shards can be written (and their
    shape verified in tests) without constructing a client or holding a key.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(batch_request_line(request, model_id=model_id) + "\n")
    return len(requests)


class BatchClient:
    """Writes, submits, polls and parses Batch API shards."""

    def __init__(self, config: ModelConfig | None = None, *, timeout: float = 120.0) -> None:
        from openai import OpenAI

        self.config = config or ModelConfig.from_env()
        self.model_id = self.config.model_id
        base_url = os.environ.get("ATS_ENRICH_BASE_URL")
        self._client = OpenAI(timeout=timeout, **({"base_url": base_url} if base_url else {}))

    def write_requests(self, path: str | Path, requests: Sequence[Request]) -> int:
        return write_batch_requests(path, requests, model_id=self.model_id)

    def submit(self, path: str | Path) -> str:
        with Path(path).open("rb") as handle:
            uploaded = self._client.files.create(file=handle, purpose="batch")
        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return str(batch.id)

    def status(self, batch_id: str) -> dict[str, Any]:
        batch = self._client.batches.retrieve(batch_id)
        return {
            "status": str(batch.status),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "counts": getattr(batch, "request_counts", None),
        }

    def download(self, file_id: str, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self._client.files.content(file_id)
        target.write_bytes(content.read())
        return target


def reported_model(path: str | Path) -> str | None:
    """The model an output file says served it, if it says.

    Cost is booked against this rather than against whatever ``--model`` the
    operator happened to pass, so collecting a file produced by
    ``batch-simulate`` is priced at the stub's zero rate instead of quietly
    adding dollars that were never spent to the ledger.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = ((record.get("response") or {}).get("body")) or {}
            model = body.get("model")
            return str(model) if model else None
    return None


def parse_batch_output(path: str | Path) -> list[Completion]:
    """Parse a Batch API output file into completions.

    Tolerant by design: a single malformed line in a 50,000-request shard
    must not discard the other 49,999 paid answers.
    """
    completions: list[Completion] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content_hash = str(record.get("custom_id") or "")
            if not content_hash:
                continue
            response = record.get("response") or {}
            if record.get("error") or response.get("status_code") != 200:
                error = json.dumps(record.get("error") or response.get("body") or {})[:400]
                completions.append(
                    Completion(content_hash=content_hash, extraction=None, error=error)
                )
                continue
            body = response.get("body") or {}
            choices = body.get("choices") or []
            usage_raw = body.get("usage") or {}
            content = ""
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""

            class _Usage:
                prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
                completion_tokens = int(usage_raw.get("completion_tokens") or 0)

            completions.append(_parse_content(content_hash, content, _Usage()))
    return completions


# --- offline stub -----------------------------------------------------------

_STUB_SENIORITY = (
    ("intern", "intern"),
    ("principal", "principal"),
    ("staff", "staff"),
    ("director", "director"),
    ("vp ", "executive"),
    ("head of", "director"),
    ("manager", "manager"),
    ("senior", "senior"),
    ("junior", "entry"),
    ("graduate", "entry"),
)
_STUB_DEPARTMENT = (
    ("engineer", "engineering"),
    ("developer", "engineering"),
    ("software", "engineering"),
    ("data scientist", "data_science"),
    ("data engineer", "data_science"),
    ("analyst", "data_science"),
    ("security", "security"),
    ("designer", "design"),
    ("product manager", "product"),
    ("sales", "sales"),
    ("marketing", "marketing"),
    ("recruiter", "people"),
    ("nurse", "healthcare"),
    ("teacher", "education"),
    ("driver", "transport"),
)
_STUB_SKILL_TERMS = (
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "sql",
    "kubernetes",
    "docker",
    "aws",
    "azure",
    "gcp",
    "terraform",
    "go",
    "rust",
    "excel",
    "salesforce",
)
_STUB_SALARY_RE = re.compile(r"([$£€])\s?(\d[\d,.]*)\s*([kK])?")


@dataclass
class StubClient:
    """Deterministic offline extractor.

    Exists so the pipeline is testable and demonstrable without spend. It
    is keyword matching, not comprehension, and the eval harness reports
    its model id verbatim so a stub score is never mistaken for a real one.
    """

    model_id: str = STUB_MODEL_ID
    config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_id=STUB_MODEL_ID, input_per_mtok=0.0, output_per_mtok=0.0
        )
    )

    def complete(self, requests: Sequence[Request]) -> list[Completion]:
        return [self._one(request) for request in requests]

    def _one(self, request: Request) -> Completion:
        text = request.user_prompt
        lowered = text.casefold()
        body = lowered.split("description:", 1)[-1].strip()

        if len(body) < 120:
            extraction = LlmExtraction(
                insufficient_text=True,
                placement=None,
                employment_type=None,
                experience_min_years=None,
                experience_max_years=None,
                seniority=None,
                salary_min=None,
                salary_max=None,
                salary_currency=None,
                salary_period=None,
                department=None,
                skills=[],
                education_level=None,
                visa_sponsorship=None,
                evidence=Evidence(salary_quote=None, placement_quote=None, experience_quote=None),
            )
            return Completion(
                content_hash=request.content_hash,
                extraction=extraction,
                input_tokens=estimate_tokens(text),
                output_tokens=40,
            )

        placement = None
        placement_quote = None
        if "hybrid" in lowered:
            placement, placement_quote = "hybrid", "hybrid"
        elif "fully remote" in lowered or "remote" in lowered:
            placement, placement_quote = "remote", "remote"
        elif "on-site" in lowered or "onsite" in lowered or "in office" in lowered:
            placement, placement_quote = "onsite", "on-site"

        seniority = next((value for needle, value in _STUB_SENIORITY if needle in lowered), None)
        department = next((value for needle, value in _STUB_DEPARTMENT if needle in lowered), None)

        from enrich.deterministic import parse_experience_years

        experience_min, experience_max = parse_experience_years(body)

        salary_min = salary_max = None
        salary_currency = None
        salary_quote = None
        match = _STUB_SALARY_RE.search(text)
        if match:
            symbol, number, thousands = match.groups()
            try:
                value = float(number.replace(",", ""))
            except ValueError:
                value = 0.0
            if thousands:
                value *= 1000
            if value > 0:
                salary_min = value
                salary_currency = {"$": "USD", "£": "GBP", "€": "EUR"}[symbol]
                salary_quote = match.group(0)

        skills = [term for term in _STUB_SKILL_TERMS if term in lowered][:12]

        extraction = LlmExtraction(
            insufficient_text=False,
            placement=placement,  # type: ignore[arg-type]
            employment_type=None,
            experience_min_years=experience_min,
            experience_max_years=experience_max,
            seniority=seniority,  # type: ignore[arg-type]
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period="YEAR" if salary_min and salary_min >= 20000 else None,
            department=department,  # type: ignore[arg-type]
            skills=skills,
            education_level=None,
            visa_sponsorship=None,
            evidence=Evidence(
                salary_quote=salary_quote,
                placement_quote=placement_quote,
                experience_quote=None if experience_min is None else "years of experience",
            ),
        )
        return Completion(
            content_hash=request.content_hash,
            extraction=extraction,
            input_tokens=estimate_tokens(text),
            output_tokens=120,
        )


# --- selection --------------------------------------------------------------


def has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def require_api_key(action: str) -> None:
    """Fail with an actionable message instead of an SDK traceback.

    The batch commands talk to the provider unconditionally — there is no
    offline equivalent of "submit this shard" — so they check up front rather
    than surfacing a credentials error from three frames deep in the client.
    """
    if not has_api_key():
        raise SystemExit(
            f"{action} needs a provider account. Set OPENAI_API_KEY (and "
            "optionally ATS_ENRICH_BASE_URL / ATS_ENRICH_MODEL) first.\n"
            "Everything up to submission works offline: use "
            "'batch-write' to generate shards and 'pilot --dry-run' to "
            "project cost."
        )


def resolve_client(
    name: str = "auto",
    *,
    concurrency: int = 8,
    model_id: str | None = None,
) -> LlmClient:
    """Pick a backend.

    ``auto`` uses the real client when a key is configured and the stub
    otherwise, so every command in this layer runs offline by default and
    only spends money once a key is deliberately present.
    """
    if name == "stub":
        return StubClient()
    if name in ("auto", "openai"):
        if has_api_key():
            return OpenAIClient(ModelConfig.from_env(model_id), concurrency=concurrency)
        if name == "openai":
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or use --client stub to "
                "run the pipeline offline."
            )
        log.warning("OPENAI_API_KEY not set - falling back to the offline stub client.")
        return StubClient()
    raise ValueError(f"unknown client {name!r}")

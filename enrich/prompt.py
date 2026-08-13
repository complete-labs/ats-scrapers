"""The Tier 1 prompt, and its version identity.

Two ideas do most of the work here.

**Abstention is the default.** Every instruction pushes toward ``null``
rather than a plausible guess. An enrichment corpus is consumed as if it
were fact, and a wrong salary is worse than a missing one because nothing
downstream can tell them apart. The evidence quotes in
:class:`enrich.schema.Evidence` exist to enforce this: a model that must
quote the posting to claim a number invents far fewer numbers.

**The prompt is content-addressed.** :func:`prompt_hash` covers the system
text, the rendered user template and the JSON schema. That hash is stored on
every cache entry and every enrichment row, so editing a single instruction
invalidates exactly the rows whose prompt changed — not the corpus, and not
nothing.

``company`` is deliberately withheld from the model. It biases
classification toward whatever the company is known for (every posting at a
bank becomes ``finance``), and including it would fragment the content cache
across tenants that share boilerplate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from enrich.schema import llm_json_schema

SYSTEM_PROMPT = """\
You extract structured facts from job postings. You are a careful reader, \
not a recruiter: you report only what the posting states or unambiguously \
implies.

Rules, in priority order:

1. If the posting does not state something, return null for that field. \
Never infer from what is typical for the role, the industry, or the \
location. "Senior Engineer in Zurich" tells you nothing about salary.
2. For salary, placement and experience you must supply a verbatim quote \
from the posting in the `evidence` object. If you cannot quote it, the \
value must be null. Do not paraphrase the quote.
3. A named city is not evidence of on-site work. Many postings list an \
office for legal or tax reasons while the role is remote or hybrid. Only \
classify placement when the posting addresses where the work happens.
4. Salary means base pay for this role. Ignore equity ranges, bonus \
percentages, total-rewards language, revenue figures, funding amounts and \
customer counts.
5. Experience means required years of professional experience. A degree \
requirement, an age, a company's founding year, or "5 years of growth" are \
not experience requirements.
6. Report the currency the posting states. Do not convert. Do not assume \
USD because the text is English.
7. If the text is empty, truncated to boilerplate, a cookie or consent \
notice, a login wall, or otherwise not a readable job posting, set \
`insufficient_text` to true and every other field to null.
8. Skills must be concrete and checkable: named technologies, tools, \
certifications, languages, regulations. Never soft skills ("team player", \
"communication", "attention to detail").

Answer only with the JSON object matching the provided schema.\
"""


def render_user_prompt(
    row: dict[str, Any],
    *,
    truncate: int,
) -> str:
    """Build the user message for one posting.

    Only the fields the model needs to judge are included, and the
    structured fields the provider already supplied are shown as context so
    the model does not contradict them for no reason.
    """
    title = (row.get("title") or "").strip()
    location = (row.get("location") or "").strip()
    commitment = (row.get("commitment") or "").strip()
    salary_summary = (row.get("salary_summary") or "").strip()
    employment_type = (row.get("employment_type") or "").strip()
    description = (row.get("description") or "").strip()
    if truncate and len(description) > truncate:
        description = description[:truncate]

    lines = [f"TITLE: {title or '(none)'}"]
    if location:
        lines.append(f"LOCATION FIELD: {location}")
    if commitment:
        lines.append(f"COMMITMENT LABEL: {commitment}")
    if employment_type:
        lines.append(f"EMPLOYMENT TYPE (provider): {employment_type}")
    if salary_summary:
        lines.append(f"SALARY FIELD: {salary_summary}")
    lines.append("")
    lines.append("DESCRIPTION:")
    lines.append(description or "(empty)")
    return "\n".join(lines)


#: Bumped by hand when the *meaning* of the contract changes in a way the
#: hash would miss (for example, a vocabulary term that keeps its spelling
#: but changes definition).
PROMPT_REVISION = 1


def prompt_hash(*, truncate: int) -> str:
    """Stable identity of the prompt, schema and truncation together.

    ``truncate`` is part of the hash because showing the model 2000 versus
    4000 characters is a different question, and answers to the two are not
    interchangeable in a cache.
    """
    payload = json.dumps(
        {
            "revision": PROMPT_REVISION,
            "system": SYSTEM_PROMPT,
            "truncate": truncate,
            "schema": llm_json_schema(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer dependency.

    Used only for cost projection and budget guards, where being within
    ~10% is enough. The divisor of 3.6 characters per token is calibrated
    for the mixed-language, punctuation-heavy text of job postings rather
    than for English prose (which runs closer to 4).
    """
    return max(1, int(len(text) / 3.6))

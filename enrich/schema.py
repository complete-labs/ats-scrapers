"""The enrichment contract: what the model returns, and what we store.

Two models, deliberately separate:

:class:`LlmExtraction`
    Exactly what a model is asked to produce. Doubles as the JSON schema
    handed to the provider's structured-output mode, so it carries no
    numeric constraints (strict mode rejects ``minimum``/``maximum``) and
    every field is nullable rather than optional — "the posting does not
    say" is a first-class answer, not a missing key.

:class:`JobEnrichment`
    One row of the sidecar: Tier 0's deterministic findings merged with
    the model's, plus the provenance needed to invalidate selectively
    (``enrichment_version``, ``model_id``, ``prompt_hash``) and the
    routing state needed to resume (``tier``, ``needs_review``).

The ``Literal`` alphabets for ``employment_type`` and ``salary_period``
are imported from :mod:`ats_scrapers.models` rather than restated, so a
change upstream is a type error here instead of a silent divergence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ats_scrapers.models import EmploymentType, SalaryPeriod

__all__ = [
    "Department",
    "EducationLevel",
    "EmploymentType",
    "Evidence",
    "JobEnrichment",
    "LlmExtraction",
    "Placement",
    "Provenance",
    "SalaryPeriod",
    "Seniority",
    "Tier",
    "VisaSponsorship",
    "llm_json_schema",
]

# --- vocabularies -----------------------------------------------------------

#: Three-way workplace placement. This is the field ``is_remote`` in the
#: upstream schema cannot express: ``infer_is_remote`` returns ``True`` or
#: ``None`` and never ``False``, so "we checked and it is on-site" is
#: unrepresentable there. ``is_remote`` is derived from this for
#: compatibility (see :meth:`JobEnrichment.as_is_remote`).
Placement = Literal["onsite", "hybrid", "remote"]

Seniority = Literal[
    "intern",
    "entry",
    "mid",
    "senior",
    "staff",
    "principal",
    "manager",
    "director",
    "executive",
]

EducationLevel = Literal[
    "none",
    "high_school",
    "associate",
    "bachelor",
    "master",
    "doctorate",
]

VisaSponsorship = Literal["offered", "not_offered", "unclear"]

#: Closed department taxonomy. A closed set is the whole point — the raw
#: ``department`` column upstream contains tens of thousands of distinct
#: free-text values ("Eng", "R&D - Platform", "Technik"), which cannot be
#: faceted. Anything unmappable stays ``None`` and the raw value is kept
#: in the untouched upstream column.
Department = Literal[
    "engineering",
    "data_science",
    "it_operations",
    "security",
    "product",
    "design",
    "sales",
    "marketing",
    "customer_success",
    "support",
    "finance",
    "legal",
    "people",
    "operations",
    "supply_chain",
    "manufacturing",
    "healthcare",
    "education",
    "research",
    "hospitality",
    "retail",
    "construction",
    "transport",
    "public_sector",
    "other",
]

#: How the source of a field is recorded, so a consumer can tell a
#: provider-supplied value from a derived one — something the published
#: CSV explicitly cannot do today ("The CSV / parquet doesn't distinguish
#: derived from source-provided values", docs/JOB_SCHEMA.md).
Provenance = Literal["provider", "tier0", "tier1", "tier2"]

Tier = Literal["tier0", "tier1", "tier2"]


# --- the model-facing contract ---------------------------------------------


class Evidence(BaseModel):
    """Verbatim spans backing the three fields that carry real risk.

    Salary, placement and experience are the fields a consumer will act
    on, and the ones where a hallucination is expensive and invisible.
    Requiring a quote makes the claim auditable after the fact and, in
    practice, suppresses invention: a model that must quote will more
    often return ``null`` than fabricate.
    """

    model_config = ConfigDict(extra="forbid")

    salary_quote: str | None = Field(
        description=(
            "Verbatim substring of the posting stating pay. Null if the posting states no pay."
        )
    )
    placement_quote: str | None = Field(
        description=(
            "Verbatim substring stating where the work happens (remote, "
            "hybrid, on-site, office requirement). Null if unstated."
        )
    )
    experience_quote: str | None = Field(
        description=("Verbatim substring stating required years of experience. Null if unstated.")
    )


class LlmExtraction(BaseModel):
    """Structured output contract for Tier 1 / Tier 2.

    Every field is required-but-nullable. Provider strict mode requires
    all keys present; nullability carries "not stated". A model that omits
    a key fails schema validation loudly instead of silently defaulting.
    """

    model_config = ConfigDict(extra="forbid")

    insufficient_text: bool = Field(
        description=(
            "True when the supplied text is too short, boilerplate, or "
            "unrelated to judge this posting (e.g. a cookie banner, a "
            "login wall, or an empty description). When true, every other "
            "field should be null."
        )
    )

    placement: Placement | None = Field(
        description=(
            "onsite = presence at a specific location required. hybrid = "
            "split between remote and an office. remote = no regular "
            "on-site presence required. Null if the posting does not say. "
            "A named city alone is NOT evidence of onsite."
        )
    )
    employment_type: EmploymentType | None = Field(
        description="Contract shape. Null if the posting does not say."
    )
    experience_min_years: int | None = Field(
        description=(
            "Minimum years of professional experience required. '3+ years' "
            "-> 3. 'at least 5 years' -> 5. A degree requirement is not "
            "experience. Null if unstated."
        )
    )
    experience_max_years: int | None = Field(
        description="Upper bound when a range is given ('3-5 years' -> 5), else null."
    )
    seniority: Seniority | None = Field(
        description="Career level implied by the title and requirements. Null if unclear."
    )

    salary_min: float | None = Field(
        description=(
            "Lower bound of pay in the stated currency, as a number with "
            "no separators. '90k' -> 90000. Null unless the posting states pay."
        )
    )
    salary_max: float | None = Field(description="Upper bound of pay, or null.")
    salary_currency: str | None = Field(
        description="ISO 4217 code of the stated pay, uppercase (USD, EUR, GBP). Null if unstated."
    )
    salary_period: SalaryPeriod | None = Field(
        description=(
            "Period the stated pay covers. Infer from magnitude and "
            "context when not explicit (e.g. '$45/hr' -> HOUR)."
        )
    )

    department: Department | None = Field(
        description="Closed-taxonomy function of the role. Null if genuinely unclear."
    )
    skills: list[str] = Field(
        description=(
            "Up to 12 concrete, checkable skills or technologies required "
            "or preferred, lowercase (e.g. 'python', 'kubernetes', "
            "'ifrs'). Exclude soft skills and generic phrases. Empty list "
            "if none are stated."
        )
    )
    education_level: EducationLevel | None = Field(
        description=(
            "Minimum formal education required. 'none' means the posting "
            "explicitly requires no degree; null means it does not say."
        )
    )
    visa_sponsorship: VisaSponsorship | None = Field(
        description=(
            "'offered' / 'not_offered' only when the posting addresses "
            "sponsorship or work authorization explicitly; 'unclear' when "
            "it is mentioned ambiguously; null when never mentioned."
        )
    )

    evidence: Evidence = Field(description="Verbatim support for salary, placement and experience.")


# --- the stored row ---------------------------------------------------------


class JobEnrichment(BaseModel):
    """One sidecar row. Joined back to the snapshot on ``job_key``."""

    model_config = ConfigDict(extra="forbid")

    job_key: str
    content_hash: str
    fallback_key: str | None = None

    url: str
    ats_type: str | None = None
    ats_id: str | None = None
    #: ``{ats_type}:{ats_id}`` — the ``global_id`` the published dataset
    #: documents but does not ship. Recomputed here so consumers who
    #: expect it have it.
    global_id: str | None = None

    language: str | None = None
    country_iso: str | None = None
    region: str | None = None
    lat: float | None = None
    lon: float | None = None
    geo_precision: Literal["country", "admin1", "city", "provider"] | None = None

    placement: Placement | None = None
    is_remote: bool | None = None
    employment_type: EmploymentType | None = None
    experience_min_years: int | None = None
    experience_max_years: int | None = None
    seniority: Seniority | None = None

    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None

    department: Department | None = None
    skills: list[str] = Field(default_factory=list)
    education_level: EducationLevel | None = None
    visa_sponsorship: VisaSponsorship | None = None

    #: Per-field origin, e.g. ``{"placement": "tier1", "country_iso": "tier0"}``.
    sources: dict[str, Provenance] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)

    tier: Tier = "tier0"
    needs_review: bool = False
    review_reason: str | None = None

    enrichment_version: int
    model_id: str | None = None
    prompt_hash: str | None = None
    enriched_at: datetime | None = None

    def as_is_remote(self) -> bool | None:
        """Project :attr:`placement` onto the upstream ``is_remote`` bool.

        ``hybrid`` maps to ``False``: the upstream field asks whether the
        role "can be performed remotely", and a hybrid role requires
        regular office presence. Consumers who need the distinction should
        read ``placement`` instead.
        """
        if self.placement is None:
            return None
        return self.placement == "remote"


# --- provider strict-mode schema -------------------------------------------

# Keywords JSON-schema strict mode does not accept. Pydantic emits several
# of them from field metadata; leaving them in gets the request rejected
# with a 400 rather than degrading gracefully.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "default",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "examples",
        "$comment",
    }
)


def _harden(node: Any) -> Any:
    """Recursively make a JSON schema acceptable to strict structured output.

    Two transformations: every object gets ``additionalProperties: false``
    and *all* of its properties listed as ``required`` (strict mode has no
    notion of optional keys — nullability carries that meaning instead),
    and unsupported validation keywords are dropped.
    """
    if isinstance(node, list):
        return [_harden(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {
        key: _harden(value) for key, value in node.items() if key not in _UNSUPPORTED_SCHEMA_KEYS
    }
    if out.get("type") == "object" or "properties" in out:
        properties = out.get("properties") or {}
        out["additionalProperties"] = False
        out["required"] = list(properties)
    return out


def llm_json_schema() -> dict[str, Any]:
    """JSON schema for :class:`LlmExtraction`, hardened for strict mode."""
    return _harden(LlmExtraction.model_json_schema())

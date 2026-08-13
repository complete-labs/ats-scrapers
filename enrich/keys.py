"""Stable identity for a job row, and content identity for its prompt.

Why this module exists: the published dataset does **not** carry
``global_id``. ``docs/JOB_SCHEMA.md`` documents it, but
``_job_to_row`` in ``scripts/run_pipeline.py`` never emits it, and a
``DESCRIBE`` of ``jobhive/v1/<ats>/jobs.parquet`` confirms 26 columns with
no ``global_id`` and no ``experience``. So enrichment cannot key on it.

Three keys, each with a different job:

:func:`job_key`
    Primary key of the sidecar. A hash of the *normalized* posting URL.
    ``url`` is the one field ``docs/JOB_SCHEMA.md`` calls "the primary
    stable identifier consumers should use to deduplicate", and it is
    non-null by model constraint.

:func:`content_hash`
    Key of the LLM cache. Covers exactly the fields the prompt shows the
    model, so two rows with identical prompts share one paid call and an
    edited posting re-enriches. Multi-location Workday/Amazon postings
    repeat one description across many rows; this is what collapses them.

:func:`fallback_key`
    Secondary identity on ``(company, title, location)`` used to carry
    enrichment across a provider URL-format change. Without it, a
    provider re-slugging its URLs silently orphans every enrichment row
    for that tenant and re-bills the whole slice.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that never identify a posting. Dropping them keeps one
# posting from getting several job_keys because an aggregator appended a
# campaign tag. Prefix families (``utm_*``) are handled separately below.
_TRACKING_PARAMS = frozenset(
    {
        "gh_src",
        "gh_jid_src",
        "lever_source",
        "lever-source",
        "lever_via",
        "source",
        "src",
        "ref",
        "referrer",
        "referer",
        "fbclid",
        "gclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "trk",
        "trackingid",
        "tracking_id",
        "recruiter",
        "iis",
        "iisn",
        "campaign",
        "medium",
    }
)
_TRACKING_PREFIXES = ("utm_",)

# Fragments that are page anchors rather than identity.
#
# The fragment CANNOT be dropped unconditionally. Oracle Cloud Recruiting
# builds every posting URL as a single-page app route with the requisition id
# in the fragment:
#
#     https://<tenant>.fa.us2.oraclecloud.com/?...&site_number=CX_1#217175
#
# Discarding it collapses an entire tenant's postings — all 296,895 Oracle
# rows, the fourth-largest provider in the corpus — onto one key, silently
# overwriting enrichment for every one of them. So the default is to keep
# the fragment, and only this small allowlist of UI anchors is stripped.
#
# The asymmetry is deliberate: keeping a fragment that turns out to be noise
# costs one duplicate enrichment, while dropping one that turns out to be
# identity corrupts the data.
_NON_IDENTIFYING_FRAGMENTS = frozenset(
    {
        "",
        "apply",
        "apply-now",
        "applynow",
        "top",
        "main",
        "content",
        "header",
        "footer",
        "details",
        "job-details",
        "jobdetails",
        "description",
        "overview",
    }
)

# ``__proto__``-style junk and empty values also get dropped; see
# _clean_query.
_WS_RUN = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _is_tracking(name: str) -> bool:
    lowered = name.lower()
    if lowered in _TRACKING_PARAMS:
        return True
    return lowered.startswith(_TRACKING_PREFIXES)


def _clean_query(query: str) -> str:
    """Drop tracking params, keep the rest in a canonical order.

    Sorting matters: Taleo and Oracle build the same posting URL with
    ``org``/``cws`` in either order depending on which page linked it.
    """
    if not query:
        return ""
    pairs = [
        (name, value)
        for name, value in parse_qsl(query, keep_blank_values=False)
        if not _is_tracking(name)
    ]
    if not pairs:
        return ""
    pairs.sort()
    return urlencode(pairs)


def normalize_url(url: str) -> str:
    """Return a canonical form of a posting URL.

    Case-folds scheme and host, drops a default port, strips tracking
    parameters, removes a single trailing slash, and keeps the fragment
    unless it is a known UI anchor (see
    :data:`_NON_IDENTIFYING_FRAGMENTS`).

    Path and fragment case are **preserved**: Workday requisition ids and
    Lever tenant slugs are case-sensitive, and upstream went out of its way
    to keep them so (see "Preserve case-sensitive Lever tenant slugs", #245).
    """
    text = (url or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    scheme = parts.scheme.lower() or "https"
    host = (parts.hostname or "").lower()
    if not host:
        # Not a parseable absolute URL (relative path, or junk). Fall back
        # to whitespace-collapsed raw text so the row still gets a stable
        # key instead of colliding with every other unparseable row.
        return _WS_RUN.sub("", text)
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    fragment = parts.fragment
    if fragment.lower() in _NON_IDENTIFYING_FRAGMENTS:
        fragment = ""
    return urlunsplit((scheme, host, path, _clean_query(parts.query), fragment))


def _sha(*parts: str) -> str:
    """32 hex chars (128 bits) of SHA-256 over NUL-joined parts.

    Truncated because these keys live in every row of a ~5M-row table and
    get joined repeatedly; 128 bits keeps collision probability
    negligible (~1e-25 at this scale) at half the storage of full SHA-256.
    NUL-joining prevents ``("ab", "c")`` and ``("a", "bc")`` colliding.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8", "surrogatepass"))
    return digest.hexdigest()[:32]


def job_key(url: str) -> str:
    """Primary sidecar key: hash of the normalized posting URL."""
    return _sha("v1|url", normalize_url(url))


def _norm_text(value: object) -> str:
    """Lowercase, unaccent, collapse to single spaces. For fuzzy keys only."""
    if not isinstance(value, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WS_RUN.sub(" ", stripped.strip().lower())


def fallback_key(company: object, title: object, location: object) -> str | None:
    """Secondary identity that survives a provider URL-format change.

    Returns ``None`` when ``company`` or ``title`` is missing — a key
    built from a location alone would merge unrelated postings, which is
    far worse than having no fallback.
    """
    company_norm = _norm_text(company)
    title_norm = _norm_text(title)
    if not company_norm or not title_norm:
        return None
    return _sha("v1|ctl", company_norm, title_norm, _norm_text(location))


def _norm_body(value: object) -> str:
    """Whitespace-insensitive form of a description body.

    Providers re-emit the same posting with different line wrapping and
    indentation (the upstream ``normalize_descriptions.py`` markdown pass
    is not always applied before publish). Collapsing all whitespace runs
    means those variants share one cache entry.
    """
    if not isinstance(value, str):
        return ""
    return _WS_RUN.sub(" ", value).strip()


def content_hash(
    *,
    title: object,
    description: object,
    location: object = None,
    salary_summary: object = None,
    commitment: object = None,
    employment_type: object = None,
    truncate: int | None = None,
) -> str:
    """Key of the LLM cache: identity of everything the prompt shows.

    ``truncate`` must match the prompt's description truncation limit. If
    the prompt only shows the model the first 2000 characters, then two
    postings that differ only past character 2000 produce an identical
    prompt and must share a cache entry — otherwise the cache misses on
    rows whose paid answer would have been byte-identical.

    Deliberately excludes ``company``: it is not shown to the model (it
    biases classification toward whatever the company is famous for) and
    including it would fragment the cache across every tenant that
    reposts a boilerplate description.
    """
    body = _norm_body(description)
    if truncate is not None:
        body = body[:truncate]
    return _sha(
        "v1|content",
        _norm_body(title),
        body,
        _norm_body(location),
        _norm_body(salary_summary),
        _norm_body(commitment),
        _norm_body(employment_type),
    )


def slugify(value: object) -> str:
    """Lowercase alphanumeric-with-hyphens form. Used for report filenames."""
    normalized = _NON_ALNUM.sub("-", _norm_text(value))
    return normalized.strip("-")

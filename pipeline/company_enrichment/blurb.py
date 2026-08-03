"""Shared handling for company description text, whatever its source.

Four sources feed :mod:`.profile` — a company's own homepage, the copy it
repeats across its job postings, its 10-K, and Wikidata — and they fail
in the same three ways, so the guards live here rather than three times
over.

**Presentation.** Sources disagree about escaping and whitespace. A
``meta`` tag arrives HTML-escaped (``America&#39;s``), a 10-K arrives as
tag soup, posting copy arrives with markdown bullets and hard wraps.
:func:`tidy` puts all of them in the same shape and :func:`trim` cuts
them to the same length on a sentence boundary.

**Legalese.** Every source carries employment law somewhere near the
company blurb: EEO statements, ITAR restrictions, accommodation notices.
It is the single most common wrong answer, and :func:`is_legalese`
rejects it.

**Identity.** A description is only worth having if it is about the
right company, and the domain a description came from is itself a fuzzy
match. :func:`mentions_name` is the corroboration — on a hand-checked
sample of 16 homepage descriptions, 15 named the company outright and
the one that did not ("We offer personalized mental health care…") is
exactly the case where a reader cannot confirm the subject either.
"""

from __future__ import annotations

import html as html_module
import re

from pipeline.company_enrichment import config
from pipeline.company_enrichment.normalize import core_name, strip_accents

# Zero-width and bidi marks survive `unescape` and break both the
# character cap and any later equality check.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff\u00ad]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
# Markdown decoration that survives the scrapers' HTML-to-text pass.
_MD_DECORATION = re.compile(r"(?m)^\s{0,3}(?:[-*+>]\s+|#{1,6}\s+)|\*{1,3}|_{2,}|`")
_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb"

# Sentence end: terminator, closing quote/bracket, then whitespace. The
# lookbehind keeps common abbreviations from splitting a sentence.
_SENTENCE_END = re.compile(
    r"(?<!\b[A-Z])(?<!\bInc)(?<!\bCorp)(?<!\bLtd)(?<!\bSt)(?<!\bU\.S)"
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bNo)"
    r"[.!?][\"'\u201d\u2019)\]]*(?=\s|$)"
)

# Employment-law boilerplate. Every one of these is language a company
# is obliged to publish, not language describing what it does.
_LEGALESE = re.compile(
    r"(?i)\b(?:"
    r"equal\s+(?:opportunity|employment)\s+(?:employer|opportunity)"
    r"|without\s+regard\s+to\s+race"
    r"|regardless\s+of\s+race,"
    r"|protected\s+veterans?\s+status"
    r"|reasonable\s+accommodations?\b"
    r"|applicants?\s+(?:will|shall)\s+receive\s+consideration"
    r"|drug[\s-]free\s+workplace"
    r"|e-verify"
    r"|itar\b|export\s+(?:control|administration)\s+regulations"
    r"|u\.?s\.?\s+government\s+export\s+regulations"
    r"|lawful\s+permanent\s+resident"
    r"|recruit(?:ing|ment)\s+fraud"
    r"|background\s+(?:check|screening)\s+(?:is|will|may)"
    r"|not\s+intended\s+to\s+be\s+all[\s-]inclusive"
    r"|pursuant\s+to\s+(?:the\s+)?(?:fair\s+chance|san\s+francisco|los\s+angeles)"
    r"|\beeo\b|\bada\b\s+(?:compliance|accommodation)"
    r"|criminal\s+histor(?:y|ies)"
    r"|401\s?\(?k\)?\s+(?:retirement|plan|match)"
    r"|paid\s+time\s+off,|dental,?\s+vision"
    r")",
)

# Verbs that mark "The position…" as a sentence about the vacancy rather
# than the opening of a company name like "The Job Shop is a staffing
# firm". Without this the noun alone is too common to key off.
_PREDICATE = (
    r"(?:is|are|was|will|would|requires?|involves?|reports?|offers?|has|have"
    r"|may|can|entails?|includes?|focuses|sits|exists|of)\b"
)

# A blurb has to start by talking about the company. These openers say
# the text is about the vacancy, or the listing, instead.
_ROLE_OPENER = re.compile(
    r"(?i)^\s*(?:about\s+(?:the|this)\s+(?:role|position|job|opportunity|team)"
    rf"|(?:the|this)\s+(?:role|position|job|opportunity|posting|listing)\s+{_PREDICATE}"
    r"|position\s+(?:summary|overview|title)"
    r"|job\s+(?:summary|description|title|type)"
    r"|responsibilities\b|what\s+you(?:'|\u2019)?ll\s+(?:do|be)"
    r"|we\s+are\s+(?:seeking|hiring|looking\s+for)\b"
    # Copy that leads with the money is a pay disclosure, not a
    # description of the company: "Pay Rate: $250/Hr, OT Rate: …".
    r"|(?:compensation|salary|pay(?:\s+rate)?|hourly\s+rate|base\s+pay)\s*[:\-\u2013]"
    r"|are\s+you\b|do\s+you\b)"
)


def tidy(text: str | None) -> str:
    """One-line, unescaped, decoration-free version of ``text``."""
    if not text:
        return ""
    out = html_module.unescape(str(text))
    # A doubly-escaped source ("&amp;#39;") needs a second pass, but no
    # more than that: unescaping until fixpoint would corrupt copy that
    # legitimately contains "&amp;".
    if "&" in out:
        out = html_module.unescape(out)
    out = _INVISIBLE.sub("", out)
    out = _CONTROL.sub(" ", out)
    out = _MD_DECORATION.sub(" ", out)
    out = _WS.sub(" ", out).strip()
    out = out.strip(_QUOTES + " ")
    return _WS.sub(" ", out).strip()


def sentences(text: str) -> list[str]:
    """Split ``text`` on sentence boundaries, keeping the terminators."""
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        chunk = text[start : match.end()].strip()
        if chunk:
            out.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def trim(text: str, *, max_sentences: int = 3, max_chars: int | None = None) -> str:
    """Cut ``text`` to whole sentences within the length budget.

    Falls back to a word boundary with an ellipsis when the first
    sentence alone is over budget, which happens with the run-on
    single-sentence copy common in SEO meta tags.
    """
    limit = config.DESCRIPTION_MAX_CHARS if max_chars is None else max_chars
    text = text.strip()
    if not text:
        return ""

    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    # An unterminated tail is still a sentence for our purposes.
    if not ends or ends[-1] < len(text):
        ends.append(len(text))

    kept = 0
    for index, end in enumerate(ends):
        if index >= max_sentences or end > limit:
            break
        kept = end
    if kept:
        return text[:kept].strip()

    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-\u2013\u2014 ")
    return f"{cut}\u2026" if cut else text[:limit]


def is_legalese(text: str) -> bool:
    """True when ``text`` is employment-law boilerplate, not a description.

    A single marker near the start is decisive — the text opens on the
    EEO statement, so that is what it is. Further out one marker proves
    nothing, since plenty of genuine copy closes with a compliance line,
    so two are required.
    """
    if not text:
        return False
    matches = list(_LEGALESE.finditer(text))
    if not matches:
        return False
    return matches[0].start() < 200 or len(matches) >= 2


def is_role_copy(text: str) -> bool:
    """True when ``text`` opens by describing the vacancy, not the company."""
    return bool(text) and bool(_ROLE_OPENER.match(text))


def _match_haystack(text: str) -> str:
    folded = strip_accents(str(text)).lower()
    return f" {re.sub(r'[^a-z0-9]+', ' ', folded).strip()} "


def mentions_name(text: str, *names: str | None) -> bool:
    """True when ``text`` names one of ``names``.

    Two ways to qualify. The whole normalised name appearing verbatim is
    the strong form and covers short names whose first token is too
    generic to trust on its own ("CVS Health", "ER Meds"). Otherwise the
    *first* token must appear, because a company name leads with its
    distinctive part: "Black Duck delivers…" identifies Black Duck
    Software, while "…mental health care…" does not identify LifeStance
    Health.
    """
    if not text:
        return False
    hay = _match_haystack(text)
    for name in names:
        core = core_name(name or "")
        if not core:
            continue
        if len(core) >= 4 and f" {core} " in hay:
            return True
        first = core.split()[0]
        if len(first) >= 4 and f" {first}" in hay:
            return True
    return False


def acceptable(text: str, *, min_chars: int | None = None) -> bool:
    """True when ``text`` is long enough and is not obviously the wrong thing."""
    floor = config.DESCRIPTION_MIN_CHARS if min_chars is None else min_chars
    if len(text) < floor:
        return False
    # Copy with no letters at all is a stray navigation or price string.
    if not re.search(r"[a-zA-Z]{3}", text):
        return False
    return not is_legalese(text) and not is_role_copy(text)

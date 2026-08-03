"""Rate-limited SEC HTTP access with on-disk caching.

The SEC's access policy requires a descriptive ``User-Agent`` carrying
contact details and throttles clients above roughly 10 requests/second;
exceeding it earns a temporary IP block. Every SEC fetch in this package
goes through :func:`get` so the limit is enforced in one place.

Responses are cached under ``config.CACHE_DIR`` keyed by URL. The bulk
files (Form D quarterlies, ticker maps) are large and stable, so a
re-run should never re-download them.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import httpx

from pipeline.company_enrichment import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_request = 0.0
_MIN_INTERVAL = 1.0 / config.SEC_MAX_RPS


def _throttle() -> None:
    global _last_request
    with _lock:
        elapsed = time.monotonic() - _last_request
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request = time.monotonic()


def _cache_path(url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    stem = url.rstrip("/").rsplit("/", 1)[-1][:60] or "sec"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return config.CACHE_DIR / "sec" / f"{safe}.{digest}{suffix}"


def get(
    url: str,
    *,
    cache: bool = True,
    suffix: str = "",
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Fetch ``url`` from the SEC, honouring the rate limit and cache."""
    path = _cache_path(url, suffix)
    if cache and path.exists() and path.stat().st_size > 0:
        return path.read_bytes()

    _throttle()
    request_headers = {
        "User-Agent": config.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        **(headers or {}),
    }
    logger.info("GET %s", url)
    response = httpx.get(
        url,
        headers=request_headers,
        timeout=timeout or config.HTTP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    data = response.content
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def get_json(url: str, **kwargs: object) -> object:
    import json

    return json.loads(get(url, suffix=".json", **kwargs))  # type: ignore[arg-type]


def exists(url: str) -> bool:
    """HEAD probe used to discover which Form D quarters are published."""
    _throttle()
    try:
        response = httpx.head(
            url,
            headers={"User-Agent": config.SEC_USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200

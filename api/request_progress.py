"""
Request-progress tracking for long-running upload/recommendation jobs.

Provides a lightweight in-memory store (with TTL expiry) so the frontend can
poll for the status of an ongoing analysis without a persistent queue.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Dict, Optional

PROGRESS_TTL_SECONDS: int = 900  # 15 minutes

_PROGRESS_LOCK = Lock()
_REQUEST_PROGRESS: Dict[str, dict] = {}


def update_request_progress(
    request_id: str,
    *,
    phase: str = "",
    message: str = "",
    percent: Optional[float] = None,
    detail: Optional[str] = None,
    current_ingredient: Optional[str] = None,
    current: Optional[int] = None,
    total: Optional[int] = None,
    status: str = "running",
    extra: Optional[dict] = None,
) -> None:
    """Update (or create) the progress record for *request_id*."""
    request_id = str(request_id or "").strip()
    if not request_id:
        return
    now = time.time()
    with _PROGRESS_LOCK:
        # Evict stale entries first.
        stale_ids = [
            key
            for key, value in _REQUEST_PROGRESS.items()
            if now - float(value.get("updated_at_epoch", now) or now) > PROGRESS_TTL_SECONDS
        ]
        for key in stale_ids:
            _REQUEST_PROGRESS.pop(key, None)

        previous = dict(_REQUEST_PROGRESS.get(request_id, {}) or {})
        previous_phase = str(previous.get("phase", "") or "")
        next_phase = phase or previous_phase
        payload: dict = {
            **previous,
            "request_id": request_id,
            "phase": next_phase,
            "message": message or previous.get("message", ""),
            "detail": (
                detail
                if detail is not None
                else ("" if phase and phase != previous_phase else previous.get("detail", ""))
            ),
            "current_ingredient": (
                current_ingredient
                if current_ingredient is not None
                else ("" if phase and phase != previous_phase else previous.get("current_ingredient", ""))
            ),
            "status": status or previous.get("status", "running"),
            "updated_at_epoch": now,
        }
        if percent is not None:
            payload["percent"] = max(0, min(100, round(float(percent), 1)))
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)
        if extra:
            payload["extra"] = {**dict(previous.get("extra", {}) or {}), **extra}
        _REQUEST_PROGRESS[request_id] = payload


def get_request_progress(request_id: str) -> dict:
    """Return the current progress dict for *request_id*, or a sentinel if unknown."""
    request_id = str(request_id or "").strip()
    if not request_id:
        return {"request_id": "", "status": "missing", "percent": 0}
    with _PROGRESS_LOCK:
        payload = dict(_REQUEST_PROGRESS.get(request_id, {}) or {})
    if not payload:
        return {
            "request_id": request_id,
            "status": "unknown",
            "percent": 0,
            "message": "No progress record found.",
        }
    return payload

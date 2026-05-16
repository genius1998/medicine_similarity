from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from api.config import get_settings


def call_local_llm(message: str) -> str:
    settings = get_settings()
    if not settings.local_llm_enabled:
        raise RuntimeError("local LLM is disabled")

    payload = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(settings.local_llm_base_url, data=payload, headers=headers, method="POST")

    last_error = None
    for attempt in range(settings.local_llm_max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=settings.local_llm_timeout_sec) as response:
                body = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                return str(((parsed.get("message") or {}).get("content")) or "")
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= settings.local_llm_max_retries:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"local LLM request failed: {last_error}")


def extract_json_from_llm_content(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        return {}

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced if fenced else []

    if not candidates:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return {}

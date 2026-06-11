from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable

from api.config import ApiSettings, get_settings
from api.local_llm_client import call_local_llm


def _read_dotenv_value(path: Path, key: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{key}="
    export_prefix = f"export {key}="
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(export_prefix):
            value = line[len(export_prefix) :]
        elif line.startswith(prefix):
            value = line[len(prefix) :]
        else:
            continue
        value = value.strip().strip("'\"")
        return value
    return ""


def _candidate_env_paths(settings: ApiSettings) -> Iterable[Path]:
    configured = str(settings.ingredient_match_llm_env_path or "").strip()
    if configured and configured != ".":
        yield settings.ingredient_match_llm_env_path
    yield settings.root_dir / ".env"


def _load_api_key(settings: ApiSettings) -> str:
    key_name = settings.ingredient_match_llm_api_key_env or "OPENAI_API_KEY"
    configured = str(settings.ingredient_match_llm_env_path or "").strip()
    if configured and configured != ".":
        value = _read_dotenv_value(settings.ingredient_match_llm_env_path, key_name)
        if value:
            os.environ[key_name] = value
            return value

    value = os.getenv(key_name, "").strip()
    if value:
        return value

    for env_path in _candidate_env_paths(settings):
        value = _read_dotenv_value(env_path, key_name)
        if value:
            os.environ.setdefault(key_name, value)
            return value
    raise RuntimeError(f"{key_name} is not set")


def _extract_openai_response_text(payload: Dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    texts = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts).strip()


def _safe_openai_error(body: str) -> str:
    try:
        payload = json.loads(body)
        message = ((payload.get("error") or {}).get("message")) or ""
        if message:
            return _redact_secret_fragments(str(message))
    except Exception:
        pass
    text = str(body or "").strip()
    return _redact_secret_fragments(text[:500]) if text else "empty error body"


def _redact_secret_fragments(text: str) -> str:
    value = str(text or "")
    if "Incorrect API key provided:" in value:
        return "Incorrect API key provided."
    value = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "[REDACTED_OPENAI_KEY]", value)
    value = re.sub(r"AIza[A-Za-z0-9_-]{6,}", "[REDACTED_GOOGLE_KEY]", value)
    return value


def _openai_responses_payload(message: str, settings: ApiSettings, include_reasoning: bool = True) -> dict:
    payload = {
        "model": settings.ingredient_match_llm_model or "gpt-5-nano",
        "input": (
            "You classify health functional ingredient RAG candidates. "
            "Return one valid JSON object only, with no Markdown.\n\n"
            f"{message}"
        ),
        "max_output_tokens": max(100, int(settings.ingredient_match_llm_max_output_tokens or 700)),
    }
    effort = str(settings.ingredient_match_llm_reasoning_effort or "").strip()
    if include_reasoning and effort:
        payload["reasoning"] = {"effort": effort}
    return payload


def _call_openai_responses(message: str, settings: ApiSettings) -> str:
    api_key = _load_api_key(settings)
    url = settings.ingredient_match_llm_base_url or "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = max(1, int(settings.ingredient_match_llm_timeout_sec or 20))
    max_retries = max(0, int(settings.ingredient_match_llm_max_retries or 0))
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        for include_reasoning in (True, False):
            request_body = json.dumps(
                _openai_responses_payload(message, settings, include_reasoning=include_reasoning),
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(url, data=request_body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                payload = json.loads(body)
                text = _extract_openai_response_text(payload)
                if not text:
                    raise RuntimeError("OpenAI ingredient match LLM returned empty content")
                return text
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"OpenAI ingredient match LLM request failed ({exc.code}): {_safe_openai_error(body)}")
                if exc.code == 400 and include_reasoning:
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                break
        if attempt < max_retries:
            time.sleep(0.5 + attempt)
    raise RuntimeError(f"OpenAI ingredient match LLM request failed: {last_error}")


def call_ingredient_match_llm(message: str) -> str:
    settings = get_settings()
    provider = str(settings.ingredient_match_llm_provider or "openai").strip().lower()
    if provider in {"local", "local_llm"}:
        return call_local_llm(message)
    if provider in {"openai", "chatgpt", "gpt"}:
        try:
            return _call_openai_responses(message, settings)
        except Exception:
            if settings.ingredient_match_llm_fallback_to_local:
                return call_local_llm(message)
            raise
    raise RuntimeError(f"Unsupported ingredient match LLM provider: {provider}")

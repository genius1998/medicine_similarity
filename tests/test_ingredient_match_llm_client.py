import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import api.ingredient_match_llm_client as client


def make_settings(tmp_path: Path, **overrides):
    values = {
        "root_dir": tmp_path,
        "ingredient_match_llm_provider": "openai",
        "ingredient_match_llm_model": "gpt-5-nano",
        "ingredient_match_llm_api_key_env": "OPENAI_API_KEY",
        "ingredient_match_llm_env_path": Path(""),
        "ingredient_match_llm_base_url": "https://api.openai.com/v1/responses",
        "ingredient_match_llm_timeout_sec": 3,
        "ingredient_match_llm_max_retries": 0,
        "ingredient_match_llm_max_output_tokens": 200,
        "ingredient_match_llm_reasoning_effort": "minimal",
        "ingredient_match_llm_fallback_to_local": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_openai_responses_call_extracts_output_text(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return (
                b'{"output":[{"content":[{"type":"output_text","text":"'
                b'{\\"decision\\":\\"no_match\\"}'
                b'"}]}]}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = request.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)

    content = client._call_openai_responses("{}", settings)

    assert content == '{"decision":"no_match"}'
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["timeout"] == 3
    assert '"model": "gpt-5-nano"' in captured["body"]
    assert '"reasoning": {"effort": "minimal"}' in captured["body"]


def test_openai_api_key_can_be_loaded_from_configured_env_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    settings = make_settings(tmp_path, ingredient_match_llm_env_path=env_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert client._load_api_key(settings) == "dotenv-key"


def test_configured_env_file_overrides_existing_process_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    settings = make_settings(tmp_path, ingredient_match_llm_env_path=env_path)
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")

    assert client._load_api_key(settings) == "dotenv-key"


def test_openai_error_redacts_api_key_fragments():
    message = "Incorrect API key provided: AIzaSyDSabcdef1234567890. Check your key."

    body = json.dumps({"error": {"message": message}})

    assert client._safe_openai_error(body) == "Incorrect API key provided."


def test_ingredient_match_llm_can_fallback_to_local(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, ingredient_match_llm_fallback_to_local=True)
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    monkeypatch.setattr(client, "_call_openai_responses", lambda message, settings: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(client, "call_local_llm", lambda message: '{"decision":"no_match"}')

    assert client.call_ingredient_match_llm("{}") == '{"decision":"no_match"}'

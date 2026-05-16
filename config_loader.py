from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CONFIG_CANDIDATES = [
    Path(__file__).resolve().parent / "deploy_ec2" / "config.json",
    Path(__file__).resolve().parent.parent / "config.json",
    Path.cwd() / "config.json",
]


def load_config(explicit_path: str | Path | None = None) -> dict:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(DEFAULT_CONFIG_CANDIDATES)

    for path in candidates:
        if path and path.exists():
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    return {}


def get_config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value in ("", None) else value

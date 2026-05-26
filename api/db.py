from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from scripts.enhance_similarity_with_explanation import resolve_runtime_paths


@contextmanager
def sqlite_connection(sqlite_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(sqlite_path)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def default_sqlite_path() -> Path:
    runtime = resolve_runtime_paths()
    return Path(runtime["sqlite_path"])

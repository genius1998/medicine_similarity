from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def sqlite_connection(sqlite_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(sqlite_path)
    try:
        yield conn
    finally:
        conn.close()

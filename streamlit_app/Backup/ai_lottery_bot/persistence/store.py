import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data") / "ai_lottery.db"


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

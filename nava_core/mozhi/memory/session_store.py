"""SQLite-backed session store for chat messages, state, and summaries."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_state (
                    session_id TEXT PRIMARY KEY,
                    last_summarized_id INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_context (
                    session_id TEXT PRIMARY KEY,
                    field_id INTEGER,
                    crop_id INTEGER
                )
            """)
            conn.commit()

    def create_session_id(self) -> str:
        session_id = uuid.uuid4().hex
        self._ensure_state(session_id)
        return session_id

    def _ensure_state(self, session_id: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chat_state (session_id, last_summarized_id) VALUES (?, 0)",
                (session_id,),
            )
            conn.commit()

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.commit()

    def fetch_messages(self, session_id: str, limit: int = 10) -> List[dict]:
        self._ensure_state(session_id)
        last_id = self.get_last_summarized_id(session_id)
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT role, content FROM chat_messages
                   WHERE session_id = ? AND id > ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, last_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def fetch_messages_with_ids(self, session_id: str, after_id: int, limit: int) -> List[tuple]:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            return conn.execute(
                """SELECT id, role, content FROM chat_messages
                   WHERE session_id = ? AND id > ?
                   ORDER BY id ASC LIMIT ?""",
                (session_id, after_id, limit),
            ).fetchall()

    def fetch_message_history(self, session_id: str, limit: Optional[int] = None) -> List[dict]:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            if limit is None:
                rows = conn.execute(
                    """SELECT role, content, created_at FROM chat_messages
                       WHERE session_id = ? ORDER BY id ASC""",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT role, content, created_at FROM chat_messages
                       WHERE session_id = ? ORDER BY id ASC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
        return [{"role": r, "content": c, "created_at": t} for r, c, t in rows]

    def count_messages_after(self, session_id: str, after_id: int) -> int:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = ? AND id > ?",
                (session_id, after_id),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_last_summarized_id(self, session_id: str) -> int:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_summarized_id FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def set_last_summarized_id(self, session_id: str, last_id: int) -> None:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE chat_state SET last_summarized_id = ? WHERE session_id = ?",
                (last_id, session_id),
            )
            conn.commit()

    def add_summary(self, session_id: str, level: int, content: str) -> None:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_summaries (session_id, level, content) VALUES (?, ?, ?)",
                (session_id, level, content),
            )
            conn.commit()

    def count_summaries(self, session_id: str, level: int) -> int:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chat_summaries WHERE session_id = ? AND level = ?",
                (session_id, level),
            ).fetchone()
        return int(row[0]) if row else 0

    def fetch_recent_summaries(self, session_id: str, level: int, limit: int) -> List[str]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT content FROM chat_summaries
                   WHERE session_id = ? AND level = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, level, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def fetch_oldest_summaries(self, session_id: str, level: int, limit: int) -> List[tuple]:
        with _connect(self.db_path) as conn:
            return conn.execute(
                """SELECT id, content FROM chat_summaries
                   WHERE session_id = ? AND level = ?
                   ORDER BY id ASC LIMIT ?""",
                (session_id, level, limit),
            ).fetchall()

    def delete_summaries(self, summary_ids: List[int]) -> None:
        if not summary_ids:
            return
        with _connect(self.db_path) as conn:
            conn.executemany("DELETE FROM chat_summaries WHERE id = ?", [(i,) for i in summary_ids])
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        with _connect(self.db_path) as conn:
            for table in ("chat_messages", "chat_summaries", "chat_state", "chat_context"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            conn.commit()

    def set_session_context(self, session_id: str, field_id: int | None, crop_id: int | None) -> None:
        self._ensure_state(session_id)
        with _connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO chat_context (session_id, field_id, crop_id)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       field_id = excluded.field_id, crop_id = excluded.crop_id""",
                (session_id, field_id, crop_id),
            )
            conn.commit()

    def get_session_context(self, session_id: str) -> dict | None:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT field_id, crop_id FROM chat_context WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {"field_id": row[0], "crop_id": row[1]}

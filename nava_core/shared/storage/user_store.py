"""User storage with password hashing, session tokens, and TTL expiry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from nava_core.shared.utils.logging import get_logger
from nava_core.shared.utils.paths import logs_dir

log = get_logger("storage.user")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _hash_password(password: str, *, iterations: int = 120_000) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, raw_iter, salt_hex, digest_hex = stored.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    try:
        iterations = int(raw_iter)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return secrets.compare_digest(expected, computed)


@dataclass(frozen=True)
class UserRecord:
    id: int
    name: str
    email: str
    password_hash: str
    onboarded: bool
    location: Optional[str]
    goals: Optional[str]
    created_at: Optional[str]
    db_path: str


class UserStore:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        session_ttl_hours: int = 168,
    ) -> None:
        self.base_dir = logs_dir() / "users"
        self.db_path = db_path or (self.base_dir / "users.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl_hours = session_ttl_hours
        self._init_db()

    def _init_db(self) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    onboarded INTEGER DEFAULT 0,
                    location TEXT,
                    goals TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    db_path TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.commit()

    def _user_db_path(self, user_id: int) -> Path:
        return self.base_dir / f"user_{user_id}" / "user_data.db"

    def _ensure_user_db(self, user_id: int) -> str:
        path = self._user_db_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _row_to_record(self, row: tuple) -> UserRecord:
        db_path = row[8] or self._ensure_user_db(row[0])
        return UserRecord(
            id=row[0],
            name=row[1],
            email=row[2],
            password_hash=row[3],
            onboarded=bool(row[4]),
            location=row[5],
            goals=row[6],
            created_at=row[7],
            db_path=db_path,
        )

    def create_user(self, name: str, email: str, password: str) -> UserRecord:
        if not _EMAIL_RE.match(email):
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        password_hash = _hash_password(password)
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, password_hash),
            )
            user_id = int(cursor.lastrowid)
            db_path = self._ensure_user_db(user_id)
            conn.execute(
                "UPDATE users SET db_path = ? WHERE id = ?",
                (db_path, user_id),
            )
            conn.commit()
        log.info("User created: id=%d email=%s", user_id, email)
        user = self.get_user(user_id)
        if not user:
            raise RuntimeError("Failed to create user")
        return user

    def get_user(self, user_id: int) -> Optional[UserRecord]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT id, name, email, password_hash, onboarded,
                          location, goals, created_at, db_path
                   FROM users WHERE id = ?""",
                (user_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update_user(self, user_id: int, name: str) -> Optional[UserRecord]:
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
            conn.commit()
        return self.get_user(user_id)

    def update_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        user = self.get_user(user_id)
        if not user or not _verify_password(current_password, user.password_hash):
            return False
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters")
        new_hash = _hash_password(new_password)
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            conn.commit()
        return True

    def delete_user(self, user_id: int) -> bool:
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        # Note: We should ideally delete the user's sqlite db file as well, 
        # but leaving it orphaned or soft-deleting might be safer for now.
        return True

    def authenticate(self, email: str, password: str) -> Optional[UserRecord]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT id, name, email, password_hash, onboarded,
                          location, goals, created_at, db_path
                   FROM users WHERE email = ?""",
                (email,),
            ).fetchone()
        if not row:
            return None
        record = self._row_to_record(row)
        if not _verify_password(password, record.password_hash):
            log.warning("Failed login attempt for %s", email)
            return None
        return record

    def create_session(self, user_id: int) -> str:
        token = secrets.token_hex(32)
        expires_at = None
        if self.session_ttl_hours > 0:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(hours=self.session_ttl_hours)
            ).isoformat()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at),
            )
            conn.commit()
        return token

    def delete_session(self, token: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()

    def get_user_by_token(self, token: str) -> Optional[UserRecord]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT u.id, u.name, u.email, u.password_hash, u.onboarded,
                          u.location, u.goals, u.created_at, u.db_path
                   FROM users u
                   JOIN sessions s ON s.user_id = u.id
                   WHERE s.token = ?""",
                (token,),
            ).fetchone()
            if not row:
                return None
            # Check expiry
            sess = conn.execute(
                "SELECT expires_at FROM sessions WHERE token = ?", (token,)
            ).fetchone()
            if sess and sess[0]:
                exp = datetime.fromisoformat(sess[0])
                if datetime.now(timezone.utc) > exp:
                    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                    conn.commit()
                    log.info("Expired session cleaned for token")
                    return None
        return self._row_to_record(row)

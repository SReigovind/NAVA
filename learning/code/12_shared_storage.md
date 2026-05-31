# Shared Storage: `user_store.py` and `field_store.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/08_database_design.md](../technical/08_database_design.md) | [03_gathi_deps.md](03_gathi_deps.md) | [04_gathi_routers_auth_and_fields.md](04_gathi_routers_auth_and_fields.md)

**Source files:**
- [`user_store.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/storage/user_store.py)
- [`field_store.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/storage/field_store.py)

---

## `user_store.py` — User and Session Management

### Password Security

```python
def _hash_password(password: str, *, iterations: int = 120_000) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"
```

**PBKDF2-HMAC-SHA256:** The stored password format is `pbkdf2_sha256$iterations$salt_hex$digest_hex`. Each component is stored inline so the verification function is self-contained — the iteration count and salt are read from the stored string, not from a global config.

- **Salt (`secrets.token_bytes(16)`):** 16 random bytes, unique per user. Without a salt, two users with the same password would produce identical hashes, enabling precomputed rainbow table attacks.
- **120,000 iterations:** PBKDF2 is designed to be slow (computationally expensive). 120,000 iterations means an attacker brute-forcing a leaked hash database must compute 120,000 SHA-256 operations per password guess. This is the OWASP recommended minimum as of 2023.

```python
def _verify_password(password: str, stored: str) -> bool:
    scheme, raw_iter, salt_hex, digest_hex = stored.split("$", 3)
    ...
    computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return secrets.compare_digest(expected, computed)
```

`secrets.compare_digest()` performs a constant-time string comparison. Standard `==` comparison short-circuits on the first differing byte — this timing difference is theoretically exploitable in a timing attack. `compare_digest` always takes the same time regardless of where the strings differ.

### `UserRecord` Dataclass

```python
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
```

`frozen=True` — immutable after construction. `UserRecord` is the internal user object passed between layers (deps → routers → service). The `db_path` field is the path to the user's per-user SQLite database file — a critical piece of information for constructing `FieldStore`.

The `password_hash` field is included because `UserRecord` is used in `_verify_password()`. The `auth.py` router's `_to_user_response()` function explicitly excludes it from the API response.

### `UserStore` Initialization

```python
def __init__(self, db_path=None, session_ttl_hours=168):
    self.base_dir = logs_dir() / "users"
    self.db_path = db_path or (self.base_dir / "users.db")
    ...
```

- `users.db` — the central database containing the `users` and `sessions` tables
- `base_dir/user_{id}/user_data.db` — per-user database for fields, crops, plants, events

Session TTL is 168 hours (7 days). After expiry, `get_user_by_token()` deletes the expired session and returns `None`, triggering a 401 response that prompts the frontend to redirect to login.

### `create_user()` — Two-Step Insertion

```python
def create_user(self, name, email, password):
    password_hash = _hash_password(password)
    cursor = conn.execute("INSERT INTO users (...) VALUES (?, ?, ?)", ...)
    user_id = int(cursor.lastrowid)
    db_path = self._ensure_user_db(user_id)          # creates the directory
    conn.execute("UPDATE users SET db_path = ? WHERE id = ?", (db_path, user_id))
```

Two-step because the user's `db_path` depends on their `user_id` (which is only known after the INSERT). The directory is created eagerly — `_ensure_user_db()` calls `path.parent.mkdir(parents=True, exist_ok=True)` — so the path is valid immediately after `create_user()` returns.

### `authenticate()` — Timing-Safe Login

```python
def authenticate(self, email, password):
    row = conn.execute("SELECT ... FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        return None
    if not _verify_password(password, record.password_hash):
        log.warning("Failed login attempt for %s", email)
        return None
    return record
```

Note: the same code path executes `_verify_password()` whether or not the email exists. A timing-safe implementation would also run `_hash_password()` on a dummy string when the email is not found (to prevent email enumeration via timing). This is a known minor improvement noted for `futureWork.md`.

### `create_session()` — Token Generation

```python
def create_session(self, user_id: int) -> str:
    token = secrets.token_hex(32)   # 32 random bytes → 64 hex chars
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=self.session_ttl_hours)).isoformat()
    conn.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)", ...)
    return token
```

`secrets.token_hex(32)` generates 32 cryptographically random bytes, encoded as a 64-character hex string. This is the same quality of randomness as OAuth2 Bearer tokens.

### `get_user_by_token()` — The Auth Check

```python
def get_user_by_token(self, token: str) -> Optional[UserRecord]:
    # JOIN users and sessions in one query
    row = conn.execute("""
        SELECT u.id, u.name, ...
        FROM users u
        JOIN sessions s ON s.user_id = u.id
        WHERE s.token = ?
    """, (token,)).fetchone()
    
    if sess and sess[0]:  # check expiry
        exp = datetime.fromisoformat(sess[0])
        if datetime.now(timezone.utc) > exp:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None  # expired
    return self._row_to_record(row)
```

Two queries: the JOIN retrieves the user in one database round-trip, the second query checks the expiry timestamp. On expiry, the session is deleted immediately (lazy expiry cleanup) rather than requiring a scheduled cleanup job.

---

## `field_store.py` — Farm Data Storage

### `_connect()` — WAL Mode

```python
def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
```

WAL (Write-Ahead Logging) mode allows concurrent reads while a write is in progress. Without WAL, SQLite uses a write lock that blocks all reads. In NAVA's case, the weather background task (write) and a concurrent API request (read) could deadlock without WAL.

### `_init_db()` and `_migrate_schema()` — Schema Management

The base tables are created using `conn.executescript()` (a single multi-statement execution). The `_migrate_schema()` method handles schema evolution for existing databases:

```python
def _migrate_schema(self, conn):
    # Check existing columns
    field_cols = {row[1] for row in conn.execute("PRAGMA table_info(fields)").fetchall()}
    
    if "lat" not in field_cols:
        conn.execute("ALTER TABLE fields ADD COLUMN lat REAL DEFAULT NULL")
    if "weather_temp" not in field_cols:
        conn.execute("ALTER TABLE fields ADD COLUMN weather_temp REAL DEFAULT NULL")
    ...
```

`PRAGMA table_info(table_name)` returns column info as rows: `(cid, name, type, notnull, default, pk)`. The migration checks if specific columns exist before attempting to add them — making it idempotent and safe to run on every startup.

**The VNIR history migration (rename + recreate):**
```python
elif vh_cols.get("plant_id", "").upper() == "TEXT":
    # Old string-keyed table — rename and recreate
    conn.executescript("""
        ALTER TABLE vnir_history RENAME TO vnir_history_old;
        CREATE TABLE vnir_history (..., plant_id INTEGER NOT NULL, ...);
    """)
```

An earlier version of NAVA stored `plant_id` as `TEXT` in `vnir_history`. The migration renames the old table and creates a new one with `INTEGER` type. The old data is not migrated (it was incompatible), but it is preserved in `vnir_history_old` for forensic reference.

### Weather Storage Methods

```python
def set_field_coordinates(self, field_id, lat, lon):
    conn.execute("UPDATE fields SET lat = ?, lon = ? WHERE id = ?", (lat, lon, field_id))

def update_field_weather(self, field_id, temp, humidity, precipitation, wind_speed):
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "UPDATE fields SET weather_temp=?, weather_humidity=?, "
        "weather_precipitation=?, weather_wind_speed=?, weather_updated_at=? WHERE id=?",
        (temp, humidity, precipitation, wind_speed, updated_at, field_id),
    )
```

Both methods use simple `UPDATE` statements. `set_field_coordinates()` accepts `None` for both lat and lon (used when invalidating coordinates after a location change). `update_field_weather()` always sets `weather_updated_at` to the current UTC time.

### `delete_field()` — Manual Cascade

```python
def delete_field(self, field_id: int) -> None:
    crop_ids = [r[0] for r in conn.execute("SELECT id FROM crops WHERE field_id = ?", ...).fetchall()]
    for cid in crop_ids:
        plant_ids = [r[0] for r in conn.execute("SELECT id FROM plants WHERE crop_id = ?", ...).fetchall()]
        for pid in plant_ids:
            conn.execute("DELETE FROM vnir_history WHERE plant_id = ?", (pid,))
            conn.execute("DELETE FROM events WHERE plant_id = ?", (pid,))
        conn.execute("DELETE FROM plants WHERE crop_id = ?", (cid,))
        conn.execute("DELETE FROM events WHERE crop_id = ?", (cid,))
    conn.execute("DELETE FROM events WHERE field_id = ?", (field_id,))
    conn.execute("DELETE FROM crops WHERE field_id = ?", (field_id,))
    conn.execute("DELETE FROM fields WHERE id = ?", (field_id,))
```

SQLite foreign key enforcement is off by default (`PRAGMA foreign_keys = OFF`). Without enabling it, cascade deletes defined in the schema don't execute automatically. NAVA implements cascade explicitly in Python: VNIR history and events are deleted from innermost (plant level) to outermost (field level), respecting the dependency graph and preventing orphan rows.

### `auto_generate_field_context()` — LLM Context Builder

```python
def auto_generate_field_context(self, field_id: int) -> str:
    field = self.get_field(field_id)
    crops = self.list_crops(field_id)
    lines = []
    
    # Field metadata
    if field.get("location"): lines.append(f"Location: {field['location']}")
    if field.get("area"): lines.append(f"Size: {field['area']}")
    ...
    
    # Per-crop summary
    for c in crops:
        parts = [f"  • {c['name']}"]
        diag_events = self.list_events(crop_id=c["id"], event_type="diagnose", limit=1)
        if diag_events:
            label = diag_events[0]["payload"].get("class_label", "")
            parts.append(f"— last disease: {label}")
        vnir_events = self.list_events(crop_id=c["id"], event_type="vnir", limit=1)
        if vnir_events:
            status = vnir_events[0]["payload"].get("status", "")
            parts.append(f"— VNIR: {status}")
        lines.append(" ".join(parts))
    
    return "\n".join(lines)
```

This method generates the `shared_context` column value — the text that gets injected into LLM prompts for field-level chat. It includes only the most recent diagnosis and VNIR status per crop (not full history). This keeps the context concise and within the LLM's context window.

**User notes preservation:**
```python
existing = field.get("shared_context") or ""
marker = "--- User notes ---"
if marker in existing:
    user_notes = existing.split(marker, 1)[1].strip()
```

The auto-generated section is regenerated on every mutation; user notes (stored below the `--- User notes ---` separator) are extracted and reattached. This ensures manual notes survive the automatic context refresh.

### `update_field()` — Dynamic SQL Generation

```python
def update_field(self, field_id, name=None, location=None, area=None, soil_type=None):
    updates, values = [], []
    for col, val in [("name", name), ("location", location), ...]:
        if val is not None:
            updates.append(f"{col} = ?")
            values.append(val)
    values.append(field_id)
    conn.execute(f"UPDATE fields SET {', '.join(updates)} WHERE id = ?", values)
```

Only the provided non-None fields are updated. This pattern — building the SET clause dynamically from non-None values — avoids overwriting existing values with None when only a subset of fields is being edited. All values are passed as parameterised arguments; the only f-string interpolation is the column names (not user input).

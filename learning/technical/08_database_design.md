# Database Design

> **Subfolder:** `technical/`
> **Cross-references:** [01_system_architecture.md](01_system_architecture.md) | [10_api_and_auth_design.md](10_api_and_auth_design.md) | [code/12_shared_storage.md](../code/12_shared_storage.md)

---

## Why SQLite?

The most natural question when seeing SQLite as a production database is: why not PostgreSQL?

For NAVA specifically:
- **Zero infrastructure:** SQLite is a library embedded in the Python process. No separate database server, no connection pooling, no database admin required. Deployment is a file copy.
- **Single-user write pattern:** Each user has their own database file. Concurrent writes are per-user. SQLite's WAL (Write-Ahead Log) mode handles the read-write concurrency within a single user's session without contention.
- **Deployment target:** A Raspberry Pi running NAVA serves dozens of users, not thousands. SQLite scales adequately for this load.
- **Simplicity:** SQLite databases are single files. Backup is `cp`. Deletion is `rm`. Per-user data isolation is a filesystem permission.

The limitations of SQLite (no full-text search indices, limited concurrent write performance across many users, no built-in connection pool) are not limiting factors for NAVA's current scale.

---

## The Two-Database Architecture

NAVA splits its data across two SQLite databases:

### Global Database: `users.db`

Located at `logs/users.db`. Contains:

```sql
CREATE TABLE users (
    id           TEXT PRIMARY KEY,  -- UUID hex
    email        TEXT UNIQUE NOT NULL,
    name         TEXT NOT NULL,
    password_hash TEXT NOT NULL,    -- bcrypt hash
    session_token TEXT,
    token_expires_at TEXT,          -- ISO timestamp
    db_path      TEXT,              -- path to this user's farm DB
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
```

One row per user. The `db_path` column stores the path to the user's individual farm database.

### Per-User Database: `user_{hash}.db`

Located at `logs/user_{hash}.db` where `{hash}` is a deterministic hash of the user's ID. Contains all farm data: fields, crops, plants, events, VNIR history, chat messages, and sessions.

**Why separate the user authentication DB from the farm data DBs?**

1. **Security isolation:** A bug that allows reading another user's DB cannot leak authentication credentials — those are in a separate file with no foreign keys to the per-user DBs.
2. **Simplicity of deletion:** Deleting a user account means deleting their DB file. No cascading foreign key deletions across a shared schema.
3. **Per-user backup:** A user's farm data can be backed up independently of all other users.
4. **No cross-user lock contention:** Per-user databases are completely independent. Two users writing simultaneously don't compete for the same write lock.

---

## WAL Mode

All NAVA SQLite databases use WAL (Write-Ahead Log) mode:

```sql
PRAGMA journal_mode=WAL;
```

Standard SQLite uses a rollback journal: writes are performed in-place, with the original data copied to a journal file first. In WAL mode, writes are appended to a separate WAL file; readers read from the main database (without being blocked by an in-progress write); the WAL is periodically checkpointed back into the main file.

**Why WAL for NAVA?**
FastAPI's async architecture means that multiple coroutines may be reading from the database while another coroutine is writing (for example, reading chat history while adding a new event). In standard journal mode, a write blocks all readers. In WAL mode, readers and a single writer can operate concurrently. This eliminates most lock contention in NAVA's async request handlers.

---

## The Farm Data Schema

The per-user database contains these tables:

### `fields`
```sql
CREATE TABLE fields (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    location      TEXT,
    area          TEXT,
    soil_type     TEXT,
    field_notes   TEXT DEFAULT '',       -- manual user notes
    shared_context TEXT DEFAULT '',      -- auto-generated LLM context
    lat           REAL,                  -- latitude (from Nominatim)
    lon           REAL,                  -- longitude (from Nominatim)
    weather_temp           REAL,         -- °C from Open-Meteo
    weather_humidity       REAL,         -- %
    weather_precipitation  REAL,         -- mm
    weather_wind_speed     REAL,         -- km/h
    weather_updated_at     TEXT,         -- ISO timestamp
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
```

The weather columns were added in Migration 5. They store the last-fetched weather values from Open-Meteo, allowing zero-latency weather reads during chat context assembly.

### `crops`
```sql
CREATE TABLE crops (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id  INTEGER NOT NULL REFERENCES fields(id),
    name      TEXT NOT NULL,
    variety   TEXT,
    season    TEXT,
    stage     TEXT,
    notes     TEXT DEFAULT '',    -- manual notes + NAVA auto-notes
    context   TEXT DEFAULT '',    -- auto-generated crop context
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### `plants`
```sql
CREATE TABLE plants (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_id  INTEGER NOT NULL REFERENCES crops(id),
    name     TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### `events`
```sql
CREATE TABLE events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id  INTEGER,
    crop_id   INTEGER,
    plant_id  INTEGER,
    event_type TEXT NOT NULL,  -- 'diagnose' | 'vnir' | 'chat_note'
    payload   TEXT,            -- JSON blob (base64 images, predictions, etc.)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Events are the append-only log of everything that has happened to the farm. They are never updated, only created and deleted.

### `vnir_history`
```sql
CREATE TABLE vnir_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id  INTEGER NOT NULL REFERENCES plants(id),
    ratio     REAL NOT NULL,    -- NIR/Green ratio; 0.0 = no leaf detected
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

A separate table (not embedded in `events`) for VNIR readings. This allows efficient `ORDER BY created_at` queries for the ratio timeseries without parsing JSON from the events table.

---

## Schema Migrations

NAVA does not use Alembic or any migration framework. Instead, the `FieldStore` constructor runs idempotent migration checks:

```python
def _run_migrations(self, conn):
    # Migration 1: add shared_context to fields
    if 'shared_context' not in columns:
        conn.execute("ALTER TABLE fields ADD COLUMN shared_context TEXT DEFAULT ''")

    # Migration 2: add context to crops
    if 'context' not in columns:
        conn.execute("ALTER TABLE crops ADD COLUMN context TEXT DEFAULT ''")

    # ... migrations 3, 4, 5 ...
```

Each migration checks whether the column already exists (via `PRAGMA table_info`) before attempting to add it. This is idempotent — running it on an already-migrated database has no effect.

**Why not Alembic?** Alembic is the standard solution for production database schema management, and it is listed in [futureWork.md](../../futureWork.md) as a future improvement. For the current single-server deployment, the manual migration approach is sufficient: it is simple, requires no configuration, and runs automatically on startup.

---

## The `shared_context` Field

The `fields.shared_context` column stores an auto-generated text summary of the field, its crops, and plant histories. This text is regenerated by `_refresh_field_context()` after every mutation (adding a plant, updating a crop, logging a scan event).

The generated text is injected into the LLM prompt's farm context block. This means the chat assistant always sees an up-to-date summary of the farm, without NAVA having to query multiple tables in the hot path of every chat request.

**Why cache in a column rather than querying at chat time?**
Computing `get_rich_crop_context()` requires joining `fields`, `crops`, `plants`, `events`, and `vnir_history`. This is fast (milliseconds) but involves multiple SELECT statements. Caching the result in `shared_context` means the chat path reads one column from one row. The trade: slightly stale context (updated on mutation, not real-time) in exchange for simpler, faster chat request handling.

---

## Cascade Deletion

When a field is deleted, all associated data must be cleaned up:

```
DELETE field → DELETE crops → DELETE plants → DELETE events → DELETE vnir_history
```

NAVA implements this cascade explicitly in `FieldStore.delete_field()` rather than relying on SQLite's `ON DELETE CASCADE` constraint. This is because the `events` table uses non-enforced foreign keys (SQLite foreign key enforcement must be explicitly enabled per connection). Explicit deletion in Python code is clearer and not dependent on connection-level `PRAGMA foreign_keys=ON` being set.

The deletion order matters: child records must be deleted before parent records. Events and vnir_history reference plants; plants reference crops; crops reference fields.

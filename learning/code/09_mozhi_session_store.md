# Mozhi: `session_store.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/07_hierarchical_memory.md](../technical/07_hierarchical_memory.md) | [08_mozhi_client_and_service.md](08_mozhi_client_and_service.md) | [technical/08_database_design.md](../technical/08_database_design.md)

**Source file:** [`session_store.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mozhi/memory/session_store.py)

---

## What `SessionStore` Manages

`SessionStore` is a SQLite-backed data access object for everything related to chat session state:
- **Messages** (`chat_messages`) — the raw conversation history
- **Summaries** (`chat_summaries`) — L1 and L2 compression summaries
- **State** (`chat_state`) — the `last_summarized_id` pointer per session
- **Context** (`chat_context`) — the field_id/crop_id association for each session

All four tables live in the same user DB file as `FieldStore`'s tables (the per-user `user_{hash}.db`). `SessionStore` and `FieldStore` both accept the same `db_path` and operate on disjoint table sets within it.

---

## Schema Initialisation

```python
def _init_db(self) -> None:
    with _connect(self.db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN metadata TEXT DEFAULT NULL")
        except Exception:
            pass  # Already exists — safe to ignore
        ...
```

**`CREATE TABLE IF NOT EXISTS`:** Makes schema creation idempotent — safe to call on every instantiation without checking if tables exist first.

**Inline migration for `metadata`:** The `metadata` column was added after initial deployment. Rather than adding a formal migration step, the pattern is:
1. `CREATE TABLE IF NOT EXISTS` includes the column in the schema
2. `ALTER TABLE ... ADD COLUMN` is attempted in a try/except
3. If the column already exists, SQLite raises an error which is silently caught

This is safe but brittle for complex migrations. As noted in [futureWork.md](../../futureWork.md), Alembic is the intended long-term replacement.

---

## The `_ensure_state` Pattern

Almost every method calls `self._ensure_state(session_id)` before its main logic:

```python
def _ensure_state(self, session_id: str) -> None:
    with _connect(self.db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chat_state (session_id, last_summarized_id) VALUES (?, 0)",
            (session_id,),
        )
        conn.commit()
```

`INSERT OR IGNORE` is SQLite's upsert-skip: if a row with this `session_id` already exists (PRIMARY KEY), the insert is silently skipped. This creates the state row on first access, guaranteeing that `get_last_summarized_id()` always finds a row and doesn't return null.

**Why not create the state row at session creation?** Sessions may be created client-side (the frontend generates a UUID for a new conversation) and sent to the server for the first time only on the first message. `_ensure_state` handles this lazy initialisation pattern cleanly.

---

## Message Storage and Retrieval

### `append_message()` — Writing

```python
def append_message(self, session_id, role, content, metadata=None):
    import json
    meta_json = json.dumps(metadata) if metadata else None
    with _connect(self.db_path) as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, meta_json),
        )
        conn.commit()
```

Messages are appended in insertion order — the `id AUTOINCREMENT` column provides a total ordering that is used throughout for "messages after ID X" queries. `metadata` (the RAG chunk information) is serialised to JSON and stored as a TEXT column.

### `fetch_messages()` — For Context Window Assembly

```python
def fetch_messages(self, session_id, limit=10):
    last_id = self.get_last_summarized_id(session_id)
    rows = conn.execute(
        """SELECT role, content FROM chat_messages
           WHERE session_id = ? AND id > ?
           ORDER BY id DESC LIMIT ?""",
        (session_id, last_id, limit),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]
```

**`id > last_id`:** Only fetches messages that have not been summarised yet. Messages with IDs ≤ `last_id` are represented by summaries in `chat_summaries` — including them again would double-count them in the context window.

**`ORDER BY id DESC LIMIT ?` then `reversed`:** Fetching in descending order and then reversing is a common SQL pattern for "get the last N records in chronological order." Fetching descending with a LIMIT gets the most recent N records; reversing puts them back in time order for the LLM context.

### `fetch_message_history()` — For the UI

```python
def fetch_message_history(self, session_id, limit=None):
    # Returns ALL messages (no last_id filter), with created_at and metadata
    for r, c, t, meta_json in rows:
        item = {"role": r, "content": c, "created_at": t}
        if meta_json:
            item["metadata"] = json.loads(meta_json)
        result.append(item)
    return result
```

Unlike `fetch_messages()`, this method:
- Has no `last_id` filter (returns the complete history including summarised messages)
- Includes `created_at` timestamps (displayed in the chat UI)
- Includes `metadata` (RAG chunks, displayed as citation tooltips)

This method is used by `GET /api/chat/history` — the frontend endpoint that loads conversation history when a user reopens a chat session.

---

## The `last_summarized_id` Pointer

```python
def get_last_summarized_id(self, session_id) -> int:
    row = conn.execute(
        "SELECT last_summarized_id FROM chat_state WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0]) if row else 0

def set_last_summarized_id(self, session_id, last_id) -> None:
    conn.execute(
        "UPDATE chat_state SET last_summarized_id = ? WHERE session_id = ?",
        (last_id, session_id),
    )
```

`last_summarized_id` is the maximum `id` of the messages that have been compressed into a L1 summary. Messages with `id > last_summarized_id` are "unsummarised" — they are included directly in the context window.

When `_summarize_if_needed()` compresses a batch:
1. It finds the max `id` in the batch: `max_id = max(row[0] for row in batch)`
2. It calls `set_last_summarized_id(session, max_id)`
3. Future `fetch_messages()` calls exclude all messages up to `max_id`

This single pointer replaces a "summarised" flag on each message row — simpler and faster.

---

## Summary Storage

### L1 and L2 Summaries

Both levels are stored in the same `chat_summaries` table, distinguished by the `level` column (1 or 2).

**`add_summary()`:** Simple INSERT. Each call creates a new row.

**`fetch_recent_summaries(session_id, level, limit)`:**
```sql
SELECT content FROM chat_summaries
WHERE session_id = ? AND level = ?
ORDER BY id DESC LIMIT ?
```
Returns the most recent N summaries of a given level. For L2, `limit=1` returns the single long-term rollup. For L1, `limit=2` returns the two most recent short-term summaries.

**`fetch_oldest_summaries(session_id, level, limit)`:**
```sql
ORDER BY id ASC LIMIT ?
```
Returns the oldest N summaries — used to identify which L1 summaries to compress into an L2 rollup.

**`delete_summaries(summary_ids)`:**
```python
conn.executemany("DELETE FROM chat_summaries WHERE id = ?", [(i,) for i in summary_ids])
```
`executemany` executes the DELETE once per ID in the list. The L1 summaries that were rolled into L2 are deleted to prevent them from being included in future L2 rollups.

---

## Session Context

```python
def set_session_context(self, session_id, field_id, crop_id):
    conn.execute(
        """INSERT INTO chat_context (session_id, field_id, crop_id)
           VALUES (?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               field_id = excluded.field_id, crop_id = excluded.crop_id""",
        (session_id, field_id, crop_id),
    )
```

`ON CONFLICT ... DO UPDATE` is SQLite's upsert: if a row with this `session_id` exists, update it; otherwise insert. This is equivalent to "set or replace" semantics.

The session context persists which field and crop the session is associated with. If the user navigates away and returns to the same session, the server remembers the context without the frontend having to re-send `field_id` and `crop_id` on every request.

```python
def get_session_context(self, session_id) -> dict | None:
    row = conn.execute(
        "SELECT field_id, crop_id FROM chat_context WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return {"field_id": row[0], "crop_id": row[1]} if row else None
```

In `ChatService.chat()`, if `field_id` and `crop_id` are not provided in the request body, the session context is retrieved from the DB and used as the default. This means the frontend only needs to send them once (at session start) — subsequent messages automatically inherit the session's context.

---

## Session Deletion

```python
def delete_session(self, session_id: str) -> None:
    for table in ("chat_messages", "chat_summaries", "chat_state", "chat_context"):
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
```

All session data is deleted atomically in a single connection (within one `with _connect(...) as conn:` block). This is called by `ChatService.clear_session()` when the user clicks "Start fresh" in the UI.

Note: `f"DELETE FROM {table}..."` uses an f-string to insert the table name. This is **not** a SQL injection risk because `table` is a hardcoded string from a Python list in the same file, not user input. The `session_id` itself is still passed as a parameterised query argument.

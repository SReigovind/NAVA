"""Field and crop storage — per-user SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class FieldStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with _connect(self.db_path) as conn:
            # Base tables — only created if they don't exist
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    location TEXT,
                    area TEXT,
                    soil_type TEXT,
                    shared_context TEXT,
                    field_notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS crops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    variety TEXT,
                    season TEXT,
                    stage TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (field_id) REFERENCES fields(id)
                );

                CREATE TABLE IF NOT EXISTS plants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crop_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (crop_id) REFERENCES crops(id),
                    UNIQUE(crop_id, name)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    field_id INTEGER,
                    crop_id INTEGER,
                    plant_id INTEGER,
                    payload TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (field_id) REFERENCES fields(id),
                    FOREIGN KEY (crop_id) REFERENCES crops(id),
                    FOREIGN KEY (plant_id) REFERENCES plants(id)
                );
            """)
            # Run incremental migrations for existing DBs
            self._migrate_schema(conn)
            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Non-destructive incremental migration for existing databases."""
        # 1. Add plant_id column to events if it doesn't exist
        try:
            event_cols = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "plant_id" not in event_cols:
                conn.execute("ALTER TABLE events ADD COLUMN plant_id INTEGER REFERENCES plants(id)")
        except Exception:
            pass

        # 1b. Add field_notes column to fields if missing (manual notes, hidden from auto-context)
        try:
            field_cols = {row[1] for row in conn.execute("PRAGMA table_info(fields)").fetchall()}
            if "field_notes" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN field_notes TEXT")
            # 1c. Geo-coordinates — stored once after Nominatim geocoding, used for weather context
            if "lat" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN lat REAL DEFAULT NULL")
            if "lon" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN lon REAL DEFAULT NULL")
            # 1d. Weather cache — refreshed on login and manual refresh
            if "weather_temp" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN weather_temp REAL DEFAULT NULL")
            if "weather_humidity" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN weather_humidity REAL DEFAULT NULL")
            if "weather_precipitation" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN weather_precipitation REAL DEFAULT NULL")
            if "weather_wind_speed" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN weather_wind_speed REAL DEFAULT NULL")
            if "weather_updated_at" not in field_cols:
                conn.execute("ALTER TABLE fields ADD COLUMN weather_updated_at TEXT DEFAULT NULL")
        except Exception:
            pass

        # 2. Handle vnir_history — must have INTEGER plant_id (not TEXT)
        try:
            vh_cols = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(vnir_history)").fetchall()}
            if not vh_cols:
                # Table doesn't exist at all — create it
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vnir_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plant_id INTEGER NOT NULL,
                        ratio REAL NOT NULL,
                        avg_green REAL,
                        avg_vnir REAL,
                        status TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (plant_id) REFERENCES plants(id)
                    )
                """)
            elif vh_cols.get("plant_id", "").upper() == "TEXT":
                # Old string-keyed table — rename and recreate (data is incompatible)
                conn.executescript("""
                    ALTER TABLE vnir_history RENAME TO vnir_history_old;
                    CREATE TABLE vnir_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plant_id INTEGER NOT NULL,
                        ratio REAL NOT NULL,
                        avg_green REAL,
                        avg_vnir REAL,
                        status TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (plant_id) REFERENCES plants(id)
                    );
                """)
        except Exception:
            pass



    # ── Fields ──────────────────────────────────────────────────────────

    def create_field(self, name: str, location: Optional[str] = None, area: Optional[str] = None,
                     soil_type: Optional[str] = None, shared_context: Optional[str] = None) -> int:
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO fields (name, location, area, soil_type, shared_context) VALUES (?, ?, ?, ?, ?)",
                (name, location, area, soil_type, shared_context),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_fields(self) -> list[dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, location, area, soil_type, shared_context, field_notes, created_at, lat, lon,"
                " weather_temp, weather_humidity, weather_precipitation, weather_wind_speed, weather_updated_at"
                " FROM fields ORDER BY id ASC"
            ).fetchall()
        keys = ("id", "name", "location", "area", "soil_type", "shared_context", "field_notes", "created_at",
                "lat", "lon", "weather_temp", "weather_humidity", "weather_precipitation",
                "weather_wind_speed", "weather_updated_at")
        return [dict(zip(keys, r)) for r in rows]

    def get_field(self, field_id: int) -> Optional[dict]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, name, location, area, soil_type, shared_context, field_notes, created_at, lat, lon,"
                " weather_temp, weather_humidity, weather_precipitation, weather_wind_speed, weather_updated_at"
                " FROM fields WHERE id = ?",
                (field_id,),
            ).fetchone()
        if not row:
            return None
        keys = ("id", "name", "location", "area", "soil_type", "shared_context", "field_notes", "created_at",
                "lat", "lon", "weather_temp", "weather_humidity", "weather_precipitation",
                "weather_wind_speed", "weather_updated_at")
        return dict(zip(keys, row))

    def update_field_context(self, field_id: int, shared_context: str) -> None:
        """Update the AUTO-GENERATED context (not shown in UI)."""
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE fields SET shared_context = ? WHERE id = ?", (shared_context, field_id))
            conn.commit()

    def update_field_notes(self, field_id: int, notes: str) -> None:
        """Update manually entered field notes (shown in UI, separate from auto-context)."""
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE fields SET field_notes = ? WHERE id = ?", (notes, field_id))
            conn.commit()

    def set_field_coordinates(self, field_id: int, lat: float, lon: float) -> None:
        """Persist geocoded lat/lon for a field so Nominatim is only called once."""
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE fields SET lat = ?, lon = ? WHERE id = ?", (lat, lon, field_id))
            conn.commit()

    def update_field_weather(
        self,
        field_id: int,
        temp: float | None,
        humidity: float | None,
        precipitation: float | None,
        wind_speed: float | None,
    ) -> None:
        """Write fresh weather values and timestamp to the fields table."""
        from datetime import datetime, timezone
        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE fields SET weather_temp=?, weather_humidity=?, "
                "weather_precipitation=?, weather_wind_speed=?, weather_updated_at=? WHERE id=?",
                (temp, humidity, precipitation, wind_speed, updated_at, field_id),
            )
            conn.commit()

    def update_field(self, field_id: int, name: Optional[str] = None, location: Optional[str] = None,
                     area: Optional[str] = None, soil_type: Optional[str] = None) -> None:
        updates, values = [], []
        for col, val in [("name", name), ("location", location), ("area", area), ("soil_type", soil_type)]:
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if not updates:
            return
        values.append(field_id)
        with _connect(self.db_path) as conn:
            conn.execute(f"UPDATE fields SET {', '.join(updates)} WHERE id = ?", values)
            conn.commit()

    def delete_field(self, field_id: int) -> None:
        """Delete a field and cascade-delete all crops, plants, events, and VNIR history."""
        with _connect(self.db_path) as conn:
            crop_ids = [r[0] for r in conn.execute("SELECT id FROM crops WHERE field_id = ?", (field_id,)).fetchall()]
            for cid in crop_ids:
                plant_ids = [r[0] for r in conn.execute("SELECT id FROM plants WHERE crop_id = ?", (cid,)).fetchall()]
                for pid in plant_ids:
                    conn.execute("DELETE FROM vnir_history WHERE plant_id = ?", (pid,))
                    conn.execute("DELETE FROM events WHERE plant_id = ?", (pid,))
                conn.execute("DELETE FROM plants WHERE crop_id = ?", (cid,))
                conn.execute("DELETE FROM events WHERE crop_id = ?", (cid,))
            conn.execute("DELETE FROM events WHERE field_id = ?", (field_id,))
            conn.execute("DELETE FROM crops WHERE field_id = ?", (field_id,))
            conn.execute("DELETE FROM fields WHERE id = ?", (field_id,))
            conn.commit()

    def auto_generate_field_context(self, field_id: int) -> str:
        """Build shared context with field metadata + latest status per crop (not full history)."""
        field = self.get_field(field_id)
        if not field:
            return ""
        crops = self.list_crops(field_id)
        lines = []
        if field.get("location"):
            lines.append(f"Location: {field['location']}")
        if field.get("area"):
            lines.append(f"Size: {field['area']}")
        if field.get("soil_type"):
            lines.append(f"Soil type: {field['soil_type']}")
        if crops:
            lines.append(f"\nActive crops ({len(crops)}):")
            for c in crops:
                parts = [f"  • {c['name']}"]
                if c.get("variety"):
                    parts.append(f"({c['variety']})")
                if c.get("stage"):
                    parts.append(f"[{c['stage']}]")
                # Latest diagnose event for this crop
                diag_events = self.list_events(crop_id=c["id"], event_type="diagnose", limit=1)
                if diag_events:
                    p = diag_events[0].get("payload") or {}
                    label = p.get("class_label", "")
                    conf = p.get("confidence")
                    if label:
                        conf_str = f" ({conf*100:.0f}%)" if conf else ""
                        parts.append(f"— last disease: {label}{conf_str}")
                # Latest VNIR event
                vnir_events = self.list_events(crop_id=c["id"], event_type="vnir", limit=1)
                if vnir_events:
                    p = vnir_events[0].get("payload") or {}
                    status = p.get("status", "")
                    if status:
                        parts.append(f"— VNIR: {status}")
                lines.append(" ".join(parts))
        # Preserve any manual user notes below the auto section
        existing = field.get("shared_context") or ""
        marker = "--- User notes ---"
        user_notes = ""
        if marker in existing:
            user_notes = existing.split(marker, 1)[1].strip()
        elif existing.strip() and not existing.startswith("Location:") and not existing.startswith("Active crops") and not existing.startswith("Size:") and not existing.startswith("Soil type:"):
            user_notes = existing.strip()
        auto_section = "\n".join(lines)
        if user_notes:
            return f"{auto_section}\n{marker}\n{user_notes}" if auto_section else f"{marker}\n{user_notes}"
        return auto_section

    # ── Crops ───────────────────────────────────────────────────────────

    def create_crop(self, field_id: int, name: str, variety: Optional[str] = None,
                    season: Optional[str] = None, stage: Optional[str] = None, notes: Optional[str] = None) -> int:
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO crops (field_id, name, variety, season, stage, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (field_id, name, variety, season, stage, notes),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_crops(self, field_id: int) -> list[dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, field_id, name, variety, season, stage, notes, created_at FROM crops WHERE field_id = ? ORDER BY id ASC",
                (field_id,),
            ).fetchall()
        return [dict(zip(("id", "field_id", "name", "variety", "season", "stage", "notes", "created_at"), r)) for r in rows]

    def get_crop(self, crop_id: int) -> Optional[dict]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, field_id, name, variety, season, stage, notes, created_at FROM crops WHERE id = ?",
                (crop_id,),
            ).fetchone()
        if not row:
            return None
        return dict(zip(("id", "field_id", "name", "variety", "season", "stage", "notes", "created_at"), row))

    def update_crop_context(self, crop_id: int, notes: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE crops SET notes = ? WHERE id = ?", (notes, crop_id))
            conn.commit()

    def update_crop(self, crop_id: int, name: Optional[str] = None, variety: Optional[str] = None,
                    season: Optional[str] = None, stage: Optional[str] = None, notes: Optional[str] = None) -> None:
        updates, values = [], []
        for col, val in [("name", name), ("variety", variety), ("season", season), ("stage", stage), ("notes", notes)]:
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if not updates:
            return
        values.append(crop_id)
        with _connect(self.db_path) as conn:
            conn.execute(f"UPDATE crops SET {', '.join(updates)} WHERE id = ?", values)
            conn.commit()

    def get_crop_context(self, crop_id: int) -> Optional[dict]:
        crop = self.get_crop(crop_id)
        if not crop:
            return None
        field = self.get_field(crop["field_id"])
        if not field:
            return None
        return {"field": field, "crop": crop}

    def get_field_context(self, field_id: int) -> Optional[dict]:
        field = self.get_field(field_id)
        if not field:
            return None
        return {"field": field}

    # ── Plants (crop-scoped, shared between detect + VNIR) ───────────────

    def create_plant(self, crop_id: int, name: str, description: Optional[str] = None) -> int:
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO plants (crop_id, name, description) VALUES (?, ?, ?)",
                (crop_id, name, description),
            )
            if cursor.lastrowid:
                conn.commit()
                return int(cursor.lastrowid)
            # Already exists — fetch it
            row = conn.execute("SELECT id FROM plants WHERE crop_id = ? AND name = ?", (crop_id, name)).fetchone()
            return int(row[0]) if row else -1

    def list_plants(self, crop_id: int) -> list[dict]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, crop_id, name, description, created_at FROM plants WHERE crop_id = ? ORDER BY name ASC",
                (crop_id,),
            ).fetchall()
        return [dict(zip(("id", "crop_id", "name", "description", "created_at"), r)) for r in rows]

    def get_plant(self, plant_id: int) -> Optional[dict]:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, crop_id, name, description, created_at FROM plants WHERE id = ?",
                (plant_id,),
            ).fetchone()
        if not row:
            return None
        return dict(zip(("id", "crop_id", "name", "description", "created_at"), row))

    def update_plant(self, plant_id: int, description: Optional[str] = None) -> None:
        if description is None:
            return
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE plants SET description = ? WHERE id = ?", (description, plant_id))
            conn.commit()

    def delete_plant(self, plant_id: int) -> None:
        """Delete a plant and all associated vnir_history and events."""
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM vnir_history WHERE plant_id = ?", (plant_id,))
            conn.execute("DELETE FROM events WHERE plant_id = ?", (plant_id,))
            conn.execute("DELETE FROM plants WHERE id = ?", (plant_id,))
            conn.commit()

    def delete_crop(self, crop_id: int) -> None:
        """Delete a crop and all its plants, events, and vnir history."""
        with _connect(self.db_path) as conn:
            # Get all plant IDs for this crop
            plant_ids = [r[0] for r in conn.execute("SELECT id FROM plants WHERE crop_id = ?", (crop_id,)).fetchall()]
            for pid in plant_ids:
                conn.execute("DELETE FROM vnir_history WHERE plant_id = ?", (pid,))
                conn.execute("DELETE FROM events WHERE plant_id = ?", (pid,))
            conn.execute("DELETE FROM plants WHERE crop_id = ?", (crop_id,))
            conn.execute("DELETE FROM events WHERE crop_id = ?", (crop_id,))
            conn.execute("DELETE FROM crops WHERE id = ?", (crop_id,))
            conn.commit()

    def delete_events_for_plant(self, plant_id: int, event_type: Optional[str] = None) -> None:
        """Delete events for a plant, optionally filtered by type."""
        with _connect(self.db_path) as conn:
            if event_type:
                conn.execute("DELETE FROM events WHERE plant_id = ? AND event_type = ?", (plant_id, event_type))
            else:
                conn.execute("DELETE FROM events WHERE plant_id = ?", (plant_id,))
            conn.commit()

    def get_event(self, event_id: int) -> Optional[dict]:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT id, event_type, field_id, crop_id, plant_id, payload, created_at FROM events WHERE id = ?", (event_id,)).fetchone()
            if not row:
                return None
            payload = None
            if row[5]:
                try:
                    payload = json.loads(row[5])
                except json.JSONDecodeError:
                    payload = {"raw": row[5]}
            return {
                "id": row[0], "event_type": row[1], "field_id": row[2],
                "crop_id": row[3], "plant_id": row[4], "payload": payload, "created_at": row[6],
            }

    def delete_event(self, event_id: int) -> None:
        """Delete a single event by ID."""
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            conn.commit()


    def add_event(self, event_type: str, field_id: Optional[int], crop_id: Optional[int],
                  payload: dict, plant_id: Optional[int] = None) -> int:
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO events (event_type, field_id, crop_id, plant_id, payload) VALUES (?, ?, ?, ?, ?)",
                (event_type, field_id, crop_id, plant_id, json.dumps(payload)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_events(self, field_id: Optional[int] = None, crop_id: Optional[int] = None,
                    plant_id: Optional[int] = None, event_type: Optional[str] = None,
                    limit: int = 50) -> list[dict]:
        clauses, values = [], []
        if field_id is not None:
            clauses.append("field_id = ?")
            values.append(field_id)
        if crop_id is not None:
            clauses.append("crop_id = ?")
            values.append(crop_id)
        if plant_id is not None:
            clauses.append("plant_id = ?")
            values.append(plant_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            values.append(event_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, event_type, field_id, crop_id, plant_id, payload, created_at "
                f"FROM events {where_sql} ORDER BY id DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        results = []
        for row in rows:
            payload = None
            if row[5]:
                try:
                    payload = json.loads(row[5])
                except json.JSONDecodeError:
                    payload = {"raw": row[5]}
            results.append({
                "id": row[0], "event_type": row[1], "field_id": row[2],
                "crop_id": row[3], "plant_id": row[4], "payload": payload, "created_at": row[6],
            })
        return results

    def get_rich_crop_context(self, crop_id: int) -> str:
        """Full context for crop-level chat — includes ALL crops in field + plant history with priority weighting."""
        crop = self.get_crop(crop_id)
        if not crop:
            return ""
        field = self.get_field(crop["field_id"])
        lines = []

        # Field metadata
        if field:
            meta = []
            if field.get("location"): meta.append(f"Location: {field['location']}")
            if field.get("area"): meta.append(f"Size: {field['area']}")
            if field.get("soil_type"): meta.append(f"Soil type: {field['soil_type']}")
            if meta:
                lines.append("=== FIELD: " + field.get("name", "") + " ===")
                lines.extend(meta)

        # All crops in this field (with latest health snapshots for awareness)
        all_crops = self.list_crops(crop["field_id"])
        if len(all_crops) >= 1:
            lines.append(f"\nAll crops in this field ({len(all_crops)}):")
            for c in all_crops:
                marker = "► (CURRENT)" if c["id"] == crop_id else "  "
                parts = [f"{marker} {c['name']}"]
                if c.get("variety"): parts.append(f"({c['variety']})")
                if c.get("stage"): parts.append(f"[{c['stage']}]")
                lines.append(" ".join(parts))
                # For sibling crops, also include health summary
                if c["id"] != crop_id:
                    sibling_plants = self.list_plants(c["id"])
                    for sp in sibling_plants:
                        sp_diag = self.list_events(plant_id=sp["id"], event_type="diagnose", limit=1)
                        sp_vnir = self.list_events(plant_id=sp["id"], event_type="vnir", limit=1)
                        sp_diag_label = (sp_diag[0].get("payload") or {}).get("class_label", "No scan") if sp_diag else "No scan"
                        sp_vnir_status = (sp_vnir[0].get("payload") or {}).get("status", "No scan") if sp_vnir else "No scan"
                        lines.append(f"      Plant '{sp['name']}': Disease={sp_diag_label} | VNIR={sp_vnir_status}")

        # Current crop details
        lines.append(f"\n=== CURRENT CROP: {crop['name']} ===")
        crop_meta = []
        if crop.get("variety"): crop_meta.append(f"Variety: {crop['variety']}")
        if crop.get("stage"): crop_meta.append(f"Growth stage: {crop['stage']}")
        if crop.get("season"): crop_meta.append(f"Season: {crop['season']}")
        if crop_meta: lines.extend(crop_meta)
        if crop.get("notes"): lines.append(f"Crop notes: {crop['notes']}")

        # Priority note for NAVA
        lines.append("\nPRIORITY RULES:")
        lines.append("  - Disease detection results have HIGHER priority than stress monitoring.")
        lines.append("  - Stress monitoring (VNIR) is precautionary; a 'healthy' VNIR result does NOT override a disease detection result.")
        lines.append("  - If disease detected: treat as active concern even if VNIR shows healthy.")

        # Plants with history — disease detection FIRST (higher priority section)
        plants = self.list_plants(crop_id)
        if plants:
            lines.append(f"\n=== PLANT MONITORING ({len(plants)} plants) ===")
            for plant in plants:
                pid = plant["id"]
                lines.append(f"\n  Plant '{plant['name']}':" + (f" — {plant['description']}" if plant.get("description") else ""))

                # Disease detection (HIGH PRIORITY)
                diag_events = self.list_events(plant_id=pid, event_type="diagnose", limit=5)
                if diag_events:
                    lines.append("    [HIGH PRIORITY] Disease Detection History:")
                    for e in reversed(diag_events):
                        p = e.get("payload") or {}
                        label = p.get("class_label", "Unknown")
                        conf = p.get("confidence")
                        conf_str = f" ({conf*100:.0f}% confidence)" if conf else ""
                        rel = p.get("reliability", "")
                        ts = e.get("created_at", "")[:10]
                        lines.append(f"      [{ts}] {label}{conf_str} — {rel}")
                else:
                    lines.append("    [Disease Detection] No scans yet.")

                # VNIR monitoring (LOWER PRIORITY — precautionary)
                vnir_events = self.list_events(plant_id=pid, event_type="vnir", limit=5)
                if vnir_events:
                    lines.append("    [LOWER PRIORITY] Stress Monitoring (VNIR — precautionary):")
                    for e in reversed(vnir_events):
                        p = e.get("payload") or {}
                        status = p.get("status", "Unknown")
                        ratio = p.get("ratio")
                        ts = e.get("created_at", "")[:10]
                        ratio_str = f" ratio={ratio:.4f}" if ratio else ""
                        lines.append(f"      [{ts}] {status}{ratio_str}")
                else:
                    lines.append("    [Stress Monitoring] No scans yet.")

        return "\n".join(lines)

    # ── Per-plant VNIR history ──────────────────────────────────────────

    def add_vnir_reading(self, plant_id: int, ratio: float, avg_green: float,
                         avg_vnir: float, status: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO vnir_history (plant_id, ratio, avg_green, avg_vnir, status) VALUES (?, ?, ?, ?, ?)",
                (plant_id, ratio, avg_green, avg_vnir, status),
            )
            conn.commit()

    def get_vnir_ratios(self, plant_id: int) -> list[float]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT ratio FROM vnir_history WHERE plant_id = ? ORDER BY id ASC",
                (plant_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def clear_vnir_history(self, plant_id: int) -> None:
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM vnir_history WHERE plant_id = ?", (plant_id,))
            conn.commit()

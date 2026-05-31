"""Weather router — serves weather from the DB, supports manual per-field refresh."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import field_store_for_user, require_user

router = APIRouter(prefix="/api", tags=["weather"])


@router.get("/weather")
def get_weather(
    field_id: int,
    user: UserRecord = Depends(require_user),
) -> dict:
    """Return weather for a field from the DB.

    If the DB already has weather values (populated on login or manual refresh),
    they are returned immediately — no API call is made.

    If the DB has no weather yet but lat/lon are stored, a fresh fetch is triggered
    and the result is written to the DB before returning.

    Returns:
        On success:          { temp, humidity, precipitation, wind_speed, updated_at, location }
        No location set:     { error: "no_location" }
        No coordinates yet:  { error: "no_coordinates" }
        Fetch failed:        { error: "unavailable" }
    """
    store = field_store_for_user(user)
    field = store.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    location = (field.get("location") or "").strip()
    if not location:
        return {"error": "no_location"}

    # ── Return DB-cached values if already populated ─────────────────────────
    if field.get("weather_updated_at") is not None:
        return {
            "temp":          field["weather_temp"],
            "humidity":      field["weather_humidity"],
            "precipitation": field["weather_precipitation"],
            "wind_speed":    field["weather_wind_speed"],
            "updated_at":    field["weather_updated_at"],
            "location":      location,
        }

    # ── No DB value yet — try to fetch and store now ─────────────────────────
    lat = field.get("lat")
    lon = field.get("lon")

    if lat is None or lon is None:
        # Coordinates not yet resolved — need Nominatim first
        try:
            from nava_core.shared.utils.geo_context import get_field_weather_context
            wx, new_lat, new_lon = get_field_weather_context(location, None, None)
            if new_lat is not None and new_lon is not None:
                store.set_field_coordinates(field_id, new_lat, new_lon)
                lat, lon = new_lat, new_lon
        except Exception:
            return {"error": "unavailable"}

    if lat is None or lon is None:
        return {"error": "no_coordinates"}

    try:
        from nava_core.shared.utils.geo_context import get_weather_context
        wx = get_weather_context(lat, lon)
        if wx is None:
            return {"error": "unavailable"}
        store.update_field_weather(field_id, wx["temp"], wx["humidity"], wx["precipitation"], wx["wind_speed"])
        from datetime import datetime, timezone
        return {
            "temp":          wx["temp"],
            "humidity":      wx["humidity"],
            "precipitation": wx["precipitation"],
            "wind_speed":    wx["wind_speed"],
            "updated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "location":      location,
        }
    except Exception:
        return {"error": "unavailable"}


@router.post("/weather/refresh")
def refresh_weather(
    field_id: int,
    user: UserRecord = Depends(require_user),
) -> dict:
    """Force-fetch fresh weather for a single field and update the DB.

    Requires lat/lon to already be stored (i.e. field must have been
    geocoded at least once). Returns the fresh values immediately.

    Returns:
        On success:     { temp, humidity, precipitation, wind_speed, updated_at, location }
        No coordinates: { error: "no_coordinates" }
        Fetch failed:   { error: "unavailable" }
    """
    store = field_store_for_user(user)
    field = store.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    location = (field.get("location") or "").strip()
    lat = field.get("lat")
    lon = field.get("lon")

    if lat is None or lon is None:
        return {"error": "no_coordinates"}

    try:
        from nava_core.shared.utils.geo_context import get_weather_context
        wx = get_weather_context(lat, lon)
        if wx is None:
            return {"error": "unavailable"}
        store.update_field_weather(field_id, wx["temp"], wx["humidity"], wx["precipitation"], wx["wind_speed"])
        # Re-read updated_at from DB so it's the exact stored timestamp
        refreshed = store.get_field(field_id)
        return {
            "temp":          wx["temp"],
            "humidity":      wx["humidity"],
            "precipitation": wx["precipitation"],
            "wind_speed":    wx["wind_speed"],
            "updated_at":    refreshed.get("weather_updated_at"),
            "location":      location,
        }
    except Exception:
        return {"error": "unavailable"}

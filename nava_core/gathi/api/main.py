"""NAVA API — FastAPI application with CORS, routers, and SPA fallback."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# Add the nava directory to sys.path so `nava_core` is importable
_nava_dir = str(Path(__file__).resolve().parents[3])
if _nava_dir not in sys.path:
    sys.path.insert(0, _nava_dir)

from nava_core.gathi.api.routers import auth, chat, diagnose, fields, vnir
from nava_core.gathi.api.startup import lifespan
from nava_core.shared.utils.paths import project_root

app = FastAPI(title="NAVA API", version="0.2.0", lifespan=lifespan)

# CORS — allow the Vite dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routers
app.include_router(auth.router)
app.include_router(diagnose.router)
app.include_router(vnir.router)
app.include_router(chat.router)
app.include_router(fields.router)

# Directories
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"
PROJECT_DIR = project_root()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serve the logo
@app.get("/api/logo")
def logo() -> FileResponse:
    logo_path = PROJECT_DIR / "NAVA-Logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    # Transparent 1x1 pixel fallback
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


# Serve static assets (JS, CSS) — must be BEFORE the catch-all
@app.get("/assets/{file_path:path}")
async def serve_assets(file_path: str) -> FileResponse:
    asset_path = FRONTEND_DIR / "assets" / file_path
    if asset_path.exists() and asset_path.is_file():
        suffix = asset_path.suffix.lower()
        media_types = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
        }
        return FileResponse(asset_path, media_type=media_types.get(suffix, "application/octet-stream"))
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


# SPA fallback — serves index.html for all non-API, non-asset routes
@app.get("/{path:path}", response_class=HTMLResponse)
def spa_fallback(path: str) -> HTMLResponse:
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<html><body><h1>NAVA</h1><p>Run <code>npm run build</code> in "
        "nava/nava_core/gathi/frontend</p></body></html>"
    )

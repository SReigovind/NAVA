"""Launch the NAVA API server."""

import sys
from pathlib import Path

# Ensure nava_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "nava_core.gathi.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent / "nava_core")],
    )

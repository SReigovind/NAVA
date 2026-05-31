# Entry Points: `run.py` and `ingest.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/01_system_architecture.md](../technical/01_system_architecture.md) | [02_gathi_main_and_startup.md](02_gathi_main_and_startup.md) | [10_yukthi_pipeline_and_store.md](10_yukthi_pipeline_and_store.md)

---

## `run.py` — The Server Entry Point

`run.py` is a 19-line script. Despite its brevity, every line serves a purpose.

```python
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
```

### Line-by-line Explanation

**`sys.path.insert(0, str(Path(__file__).resolve().parent))`**

This inserts the project root directory at the beginning of Python's module search path. Without this, `import nava_core` would fail unless NAVA was installed as a package (which it can be, via `pip install -e .`, but `run.py` should work even without installation).

`Path(__file__).resolve().parent` evaluates to the absolute path of the directory containing `run.py` — i.e., the project root. Inserting it at position 0 (not appending) ensures it takes priority over any other version of `nava_core` that might be installed system-wide.

**`uvicorn.run("nava_core.gathi.api.main:app", ...)`**

This is uvicorn's programmatic API. The first argument is a module:attribute import string — uvicorn will import `nava_core.gathi.api.main` and find the `app` attribute (the FastAPI application object). 

Why pass a string instead of the object directly? Because `reload=True` requires uvicorn to be able to re-import the module from scratch when source files change. If you pass the live object, uvicorn has no way to re-construct it. The string lets uvicorn manage the import lifecycle for hot-reloading.

**`host="0.0.0.0"`**

Binds to all network interfaces (not just localhost). This is required for the server to be accessible from other machines on the network — essential for a farming tool deployed on a local server that users access from their phones.

**`port=8000`**

Standard development port. The frontend Vite dev server runs on 5173; the API server on 8000. In production (serving the built SPA from FastAPI), only port 8000 is needed.

**`reload=True, reload_dirs=[...]`**

Hot-reload: uvicorn watches the `nava_core/` directory for file changes and restarts the server automatically. This is invaluable during development — edit a router, save, and the change is live in under 1 second without restarting manually.

**`if __name__ == "__main__"`**

Standard Python entry point guard. This ensures that `uvicorn.run()` is only called when the script is run directly (`python run.py`), not when it is imported as a module.

---

## `ingest.py` — The RAG Ingestion CLI

`ingest.py` is a standalone command-line tool for building and managing the ChromaDB vector knowledge base. It is completely independent of the running FastAPI server.

### What It Does

It ingests text and PDF documents from `ragsource/<crop>/` into ChromaDB, making them retrievable by the RAG system.

### Structure: Lazy Imports

```python
def _build_pipeline():
    from nava_core.yukthi.pipeline import RAGPipeline
    return RAGPipeline.from_settings()
```

The pipeline is imported inside a function, not at the top of the file. **Why?** ChromaDB and SentenceTransformers take several seconds to load. If they were imported at module level, even `python ingest.py --help` would wait for these imports. Lazy imports ensure that `--help` is instant and that heavy imports only happen when actually needed.

### The `cmd_ingest()` Function

```python
def cmd_ingest(crop: str, force: bool) -> None:
    pipeline = _build_pipeline()

    if not force and pipeline.store.collection_exists(crop):
        print(f"  ✓ Collection 'nava_{crop}' already exists. Skipping.")
        return

    sources = pipeline._find_sources(crop)
    if not sources:
        print(f"  ✗ No source files found.")
        return

    t0 = time.time()
    count = pipeline.ingest(crop=crop, force=force)
    elapsed = time.time() - t0

    print(f"  ✓ Done: {count} chunks ingested in {elapsed:.1f}s")
```

**Skip if exists:** Without `--force`, a collection that already exists is skipped with a message. This makes `python ingest.py --all` idempotent — you can run it any time and it won't re-process crops that are already ingested. This is important because ingestion is slow (1–5 minutes per crop, depending on document size and embedding model speed).

**`_find_sources()`:** Called before `ingest()` to give the user a preview of what will be ingested. If the crop folder doesn't exist or contains no supported files, the error is printed clearly before any ChromaDB operations begin.

**Timing:** The elapsed time is printed to help the user estimate how long re-ingestion of all crops will take.

### The `cmd_status()` Function

```python
def cmd_status() -> None:
    store, chroma_dir = _build_store()
    pipeline = _build_pipeline()
    available_crops = pipeline.list_available_crops()
    # ... print ingestion status and collection chunk counts
```

`--status` gives a quick overview: which crops have source files in `ragsource/`, which are ingested, and how many chunks each ChromaDB collection contains. A typical output:

```
══════════════════════════════════════════════════════════
  Yukthi RAG Knowledge Base Status
  Vector store : /Users/.nava/chroma
  Source dir   : /path/to/ragsource

  ragsource/ subfolders (crops with source files):
    [✓ ingested]  banana/  (2 file(s))
    [✓ ingested]  rice/    (1 file(s))
    [✗ not ingested yet]  tomato/  (1 file(s))

  ChromaDB collections:
  Collection                      Chunks
  ──────────────────────────────   ────────
  nava_banana                          487
  nava_rice                            312
```

This is the only way to inspect the RAG knowledge base state without looking at the raw ChromaDB files.

### The `main()` Function and Argument Parsing

```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--crop", metavar="NAME", ...)
group.add_argument("--all", action="store_true", ...)
group.add_argument("--status", action="store_true", ...)

parser.add_argument("--force", action="store_true", default=False, ...)
```

A mutually exclusive group ensures the user can only specify one primary action (`--crop`, `--all`, or `--status`). `--force` is independent and can be combined with `--crop` or `--all`.

**`--all` auto-discovery:**
```python
crops = pipeline.list_available_crops()
```
Rather than maintaining a hardcoded list of crop names, `list_available_crops()` scans the `ragsource/` directory and returns the names of all subfolders. Adding a new crop requires only creating a new folder in `ragsource/` — no code change.

**Error handling:**
Each crop ingestion is wrapped in a try/except:
```python
for crop in crops:
    try:
        cmd_ingest(crop=crop, force=args.force)
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error ingesting '{crop}': {e}")
```
If one crop fails (e.g., a corrupted PDF), the loop continues with the next crop. Keyboard interrupt (Ctrl+C) is handled gracefully.

---

## The Relationship Between `run.py` and `ingest.py`

These two scripts are the only entry points to the NAVA system. They are intentionally decoupled:

- `run.py` starts the live server. It reads from ChromaDB but never writes to it.
- `ingest.py` builds ChromaDB. It does not start any server.

This decoupling means you can run ingestion at any time — even while the server is running — without restarting it. The ChromaDB `PersistentClient` uses file locking; concurrent access from the server and the ingestion CLI is safe as long as different collections are not being mutated simultaneously.

In practice, the workflow is:
1. Add or update source documents in `ragsource/<crop>/`
2. Run `python ingest.py --crop <crop> --force` to rebuild that collection
3. The running server will pick up the new data on the next RAG query (no restart required)

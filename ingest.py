#!/usr/bin/env python
"""
NAVA-AG · Yukthi RAG Ingestion CLI
====================================
Builds or updates the ChromaDB knowledge base from source files in ragsource/.

Folder structure expected
-------------------------
  ragsource/
  ├── banana/
  │   ├── banana.txt
  │   └── KAU Package of Practices.pdf
  ├── tomato/
  │   └── tomato.txt
  └── <any crop name>/
      └── <any supported files>

Each subfolder name becomes the crop name. ALL supported files inside it are
ingested into a dedicated ChromaDB collection (nava_<crop>).

Currently supported file formats: .txt, .pdf
To add a new format: register a handler in nava_core/yukthi/chunker.py.

Usage
-----
  # Ingest a specific crop (skips if already ingested)
  python ingest.py --crop banana

  # Force re-ingest a crop (wipes collection and rebuilds from scratch)
  python ingest.py --crop banana --force

  # Ingest ALL crops (auto-discovers from ragsource/ subfolders)
  python ingest.py --all

  # Force re-ingest all crops
  python ingest.py --all --force

  # Show current vector store status
  python ingest.py --status

Notes
-----
  - Completely independent of run.py / the FastAPI server.
  - The server reads from the vector store at query time; it never writes to it.
  - Ingestion is repeatable: --force wipes and rebuilds; without it, existing
    collections are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_pipeline():
    from nava_core.yukthi.pipeline import RAGPipeline
    return RAGPipeline.from_settings()


def _build_store():
    from nava_core.yukthi.store import YukthiStore
    from nava_core.shared.config import get_settings
    s = get_settings()
    return YukthiStore(s.yukthi_chroma_dir), s.yukthi_chroma_dir


def cmd_ingest(crop: str, force: bool) -> None:
    print(f"\n{'─'*60}")
    print(f"  Crop : {crop.upper()}")
    print(f"  Mode : {'force re-ingest (wipe + rebuild)' if force else 'skip if exists'}")
    print(f"{'─'*60}")

    pipeline = _build_pipeline()

    if not force and pipeline.store.collection_exists(crop):
        print(f"  ✓ Collection 'nava_{crop}' already exists. Skipping.")
        print(f"    Use --force to wipe and rebuild it.\n")
        return

    sources = pipeline._find_sources(crop)
    if not sources:
        crop_dir = pipeline.source_dir / crop
        print(f"  ✗ No source files found.")
        print(f"    Expected folder: {crop_dir}")
        print(f"    Create it and place .txt or .pdf files inside.\n")
        return

    print(f"  Source folder : {pipeline.source_dir / crop}")
    print(f"  Files ({len(sources)}):")
    for f in sources:
        print(f"    · {f.name}  ({f.stat().st_size // 1024} KB)")

    t0 = time.time()
    count = pipeline.ingest(crop=crop, force=force)
    elapsed = time.time() - t0

    if count > 0:
        print(f"\n  ✓ Done: {count} chunks ingested in {elapsed:.1f}s\n")
    else:
        print(f"\n  ✗ Ingestion produced 0 chunks — check source file content.\n")


def cmd_status() -> None:
    store, chroma_dir = _build_store()
    pipeline = _build_pipeline()
    available_crops = pipeline.list_available_crops()

    print(f"\n{'═'*60}")
    print(f"  Yukthi RAG Knowledge Base Status")
    print(f"  Vector store : {chroma_dir}")
    print(f"  Source dir   : {pipeline.source_dir}")
    print(f"{'═'*60}")

    if available_crops:
        print(f"\n  ragsource/ subfolders (crops with source files):")
        for crop in available_crops:
            sources = pipeline._find_sources(crop)
            ingested = store.collection_exists(crop)
            status = "✓ ingested" if ingested else "✗ not ingested yet"
            print(f"    [{status}]  {crop}/  ({len(sources)} file(s))")
    else:
        print(f"\n  No crop subfolders found in {pipeline.source_dir}")

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = client.list_collections()
        if collections:
            print(f"\n  ChromaDB collections:")
            print(f"  {'Collection':<30} {'Chunks':>8}")
            print(f"  {'─'*30}   {'─'*8}")
            for col in sorted(collections, key=lambda c: c.name):
                print(f"  {col.name:<30} {col.count():>8}")
        else:
            print(f"\n  No ChromaDB collections found yet.")
            print(f"  Run: python ingest.py --crop <name>  to build.")
    except Exception as e:
        print(f"\n  ✗ Could not read ChromaDB: {e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAVA-AG Yukthi RAG ingestion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--crop",
        metavar="NAME",
        help="Crop to ingest (e.g. banana, tomato). Must have a matching <name>.txt in ragsource/.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest all known crops listed in KNOWN_CROPS.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show the current state of the vector store (collections + chunk counts).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-ingestion even if a collection already exists (wipes and rebuilds).",
    )

    args = parser.parse_args()

    if args.status:
        cmd_status()
        return

    if args.all:
        # Auto-discover from ragsource/ subfolders — no hardcoded list needed
        pipeline = _build_pipeline()
        crops = pipeline.list_available_crops()
        if not crops:
            print(f"\nNo crop subfolders found in {pipeline.source_dir}")
            print("Create ragsource/<cropname>/ folders and place source files inside.")
            sys.exit(1)
        print(f"\nAuto-discovered crops: {', '.join(crops)}")
    else:
        crops = [args.crop.lower().strip()]
        print(f"\nNAVA-AG Yukthi Ingestion")

    for crop in crops:
        try:
            cmd_ingest(crop=crop, force=args.force)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ✗ Error ingesting '{crop}': {e}\n")

    print("All done.\n")


if __name__ == "__main__":
    main()

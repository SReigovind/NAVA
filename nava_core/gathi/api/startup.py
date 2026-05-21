"""NAVA application startup — preloads all heavy singletons at server boot.

All ML models and ChromaDB are initialized here once, in the main process,
before any request is handled. This avoids:
  - ChromaDB Rust FFI failures when PersistentClient is created in a worker thread
  - Embedding model being reloaded on every request
  - EfficientNet / VNIR models loading on the first POST (login latency)

Usage (in main.py lifespan):
    from nava_core.gathi.api.startup import lifespan
    app = FastAPI(lifespan=lifespan)
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger("nava.startup")


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """FastAPI lifespan context — runs startup before yield, shutdown after."""
    _startup(app)
    yield
    # No explicit shutdown needed for these singletons


def _startup(app: "FastAPI") -> None:
    """Synchronously preload all heavy singletons in the main thread."""
    log.info("=== NAVA startup: preloading models and vector store ===")

    # ── 1. Disease detection model (EfficientNet-B0) ────────────────────────
    def _load_predictor():
        try:
            from nava_core.gathi.api.deps import get_predictor
            get_predictor()
            log.info("Startup: EfficientNet-B0 disease detection model ready.")
        except Exception as e:
            log.warning("Startup: EfficientNet-B0 failed to load: %s", e)

    # ── 2. VNIR stress model ────────────────────────────────────────────────
    def _load_vnir():
        try:
            from nava_core.gathi.api.deps import get_vnir_pipeline
            get_vnir_pipeline()
            log.info("Startup: VNIR stress model ready.")
        except Exception as e:
            log.warning("Startup: VNIR model failed to load: %s", e)

    # ── 3. Yukthi — ChromaDB store + RAG retriever (eager init) ────────────
    def _load_yukthi():
        try:
            from nava_core.shared.config import get_settings
            s = get_settings()
            if not s.yukthi_enabled:
                log.info("Startup: Yukthi RAG is disabled (NAVA_YUKTHI_ENABLED=false).")
                return

            from nava_core.yukthi.store import YukthiStore
            from nava_core.yukthi.retriever import RAGRetriever

            store = YukthiStore(s.yukthi_chroma_dir)
            retriever = RAGRetriever(
                store=store,
                embed_model=s.yukthi_embed_model,
                top_k=s.yukthi_top_k,
                distance_threshold=s.yukthi_distance_threshold,
            )
            # Warm up the embedding model now (first encode triggers model load)
            retriever.warm_up()

            # Store singletons on app.state for reuse across requests
            app.state.yukthi_store = store
            app.state.rag_retriever = retriever
            log.info("Startup: Yukthi RAG store and retriever ready.")

        except Exception as e:
            log.warning("Startup: Yukthi RAG failed to initialise: %s", e)
            app.state.yukthi_store = None
            app.state.rag_retriever = None

    # Run ML loaders in background threads so server becomes available immediately
    # ChromaDB MUST run in main thread first (Rust FFI), then requests are fine.
    # We call it synchronously here (in the lifespan coroutine which runs in main),
    # then kick the slower ML loaders into background threads.
    _load_yukthi()  # synchronous — ChromaDB Rust FFI needs main thread

    t1 = threading.Thread(target=_load_predictor, daemon=True, name="startup-efficientnet")
    t2 = threading.Thread(target=_load_vnir, daemon=True, name="startup-vnir")
    t1.start()
    t2.start()

    log.info("=== NAVA startup complete (ML models loading in background) ===")

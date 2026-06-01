"""RAG ingestion pipeline — parse → chunk → embed → upsert into ChromaDB.

Usage (programmatic):
    pipeline = RAGPipeline.from_settings()
    pipeline.ingest(crop="banana")

The pipeline is idempotent: re-running upserts the same deterministic chunk IDs
without creating duplicates.

Source layout expected:
    ragsource/
    └── banana/          ← one subfolder per crop
        ├── banana.txt
        ├── guidebook.pdf
        └── notes.txt    ← ALL supported files in the folder are ingested

To add a new file format, register a handler in chunker.py's CHUNKER_REGISTRY.
No changes needed here in pipeline.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .chunker import chunk_file, Chunk, SUPPORTED_EXTENSIONS
from .store import YukthiStore
from nava_core.shared.config import get_settings
from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.pipeline")


class RAGPipeline:
    def __init__(
        self,
        store: YukthiStore,
        source_dir: Path,
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.store = store
        self.source_dir = source_dir
        self.embed_model = embed_model
        self._encoder = None  # lazy-loaded SentenceTransformer

    @classmethod
    def from_settings(cls) -> "RAGPipeline":
        s = get_settings()
        return cls(
            store=YukthiStore(s.yukthi_chroma_dir),
            source_dir=s.yukthi_source_dir,
            embed_model=s.yukthi_embed_model,
        )

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Loading embedding model: %s", self.embed_model)
                self._encoder = SentenceTransformer(self.embed_model)
                log.info("Embedding model loaded.")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required. Install it with: pip install sentence-transformers"
                )
        return self._encoder

    def _find_sources(self, crop: str) -> list[Path]:
        """Return all ingestible files inside ragsource/{crop}/.

        Every file with a supported extension in the crop's dedicated subfolder
        is returned. The user controls exactly which files belong to each crop
        by placing (or symlinking) them into that subfolder.

        Supported extensions are driven by chunker.SUPPORTED_EXTENSIONS —
        add a new format there; no change needed here.
        """
        crop_dir = self.source_dir / crop.lower().strip()
        if not crop_dir.exists() or not crop_dir.is_dir():
            return []
        return sorted(
            f for f in crop_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def list_available_crops(self) -> list[str]:
        """Auto-discover crops by listing non-hidden subdirectories in source_dir.

        Used by ingest.py --all so it doesn't need a hardcoded crop list.
        """
        if not self.source_dir.exists():
            return []
        return sorted(
            d.name for d in self.source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def ingest(self, crop: str, force: bool = False) -> int:
        """Ingest all source files for the given crop into ChromaDB.

        Args:
            crop: Crop name (e.g. "banana")
            force: If True, re-ingest even if collection already exists.

        Returns:
            Number of chunks upserted.
        """
        if not force and self.store.collection_exists(crop):
            collection_name = self.store.collection_name(crop)
            log.info("Collection '%s' already exists. Skipping ingestion (use force=True to re-ingest).", collection_name)
            return 0

        if force and self.store.collection_exists(crop):
            log.info("Force re-ingest: deleting existing collection 'nava_%s'.", crop)
            self.store.delete_collection(crop)

        sources = self._find_sources(crop)
        crop_dir = self.source_dir / crop.lower().strip()
        if not sources:
            log.warning(
                "No source files found for crop '%s'. "
                "Expected a folder at: %s  containing .txt or .pdf files.",
                crop, crop_dir,
            )
            return 0

        log.info("Ingesting %d source file(s) for crop '%s' from %s:",
                 len(sources), crop, crop_dir)
        for f in sources:
            log.info("  · %s", f.name)

        all_chunks: list[Chunk] = []
        for source_path in sources:
            t0 = time.time()
            chunks = chunk_file(source_path, crop)
            elapsed = time.time() - t0
            log.info("  %s → %d chunks (%.2fs)", source_path.name, len(chunks), elapsed)
            all_chunks.extend(chunks)

        if not all_chunks:
            log.warning("No chunks produced for crop '%s'.", crop)
            return 0

        # Embed all chunks in one batch call
        encoder = self._get_encoder()
        texts = [f"[{c.section}]\n{c.text}" for c in all_chunks]
        log.info("Embedding %d chunks...", len(texts))
        t0 = time.time()
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False).tolist()
        log.info("Embedding complete (%.2fs).", time.time() - t0)

        # Build deterministic IDs: {source}_{chunk_index}
        ids = [f"{c.source}_{c.chunk_index}" for c in all_chunks]
        metadatas = [
            {"crop": crop, "source": c.source, "section": c.section, "chunk_index": c.chunk_index}
            for c in all_chunks
        ]

        self.store.upsert(crop=crop, ids=ids, embeddings=embeddings,
                          documents=texts, metadatas=metadatas)
        log.info("Ingestion complete: %d chunks for crop '%s'.", len(all_chunks), crop)
        return len(all_chunks)

    def ingest_if_missing(self, crop: str) -> int:
        """Ingest only if the collection doesn't already exist."""
        return self.ingest(crop, force=False)

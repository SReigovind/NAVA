"""ChromaDB wrapper for Yukthi vector store management.

Per-crop collections: nava_{crop_name}
Persistent client stored at logs/chroma/

ChromaDB is initialized eagerly at construction time (not lazily per-call)
to avoid threading issues with the Rust bindings inside FastAPI request threads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.store")


class YukthiStore:
    def __init__(self, chroma_dir: Path) -> None:
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._init_client()

    def _init_client(self):
        """Initialize ChromaDB client eagerly at construction time.

        ChromaDB 1.5.x has threading issues when PersistentClient is first
        created inside a FastAPI/uvicorn request thread. Initializing it in
        __init__ (typically the main thread or startup hook) avoids this.
        Falls back to EphemeralClient (in-memory, non-persistent) if the
        persistent client fails — so the server stays up even if ChromaDB
        has an issue, though data won't persist across restarts in that case.
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            log.info("ChromaDB PersistentClient ready at %s", self.chroma_dir)
            return client
        except Exception as e:
            log.error(
                "ChromaDB PersistentClient failed (%s). "
                "Falling back to EphemeralClient — data will NOT persist. "
                "Try upgrading chromadb: pip install --upgrade chromadb",
                e,
            )
            try:
                import chromadb
                return chromadb.EphemeralClient()
            except Exception as e2:
                log.error("ChromaDB EphemeralClient also failed: %s. RAG will be unavailable.", e2)
                return None

    def _get_client(self):
        if self._client is None:
            raise RuntimeError("ChromaDB client is not available.")
        return self._client

    def collection_name(self, crop: str) -> str:
        """Normalise crop name to a valid ChromaDB collection name."""
        return "nava_" + crop.lower().strip().replace(" ", "_").replace("-", "_")

    def collection_exists(self, crop: str) -> bool:
        try:
            client = self._get_client()
            existing = [c.name for c in client.list_collections()]
            return self.collection_name(crop) in existing
        except Exception as e:
            log.warning("Failed to check collections: %s", e)
            return False

    def delete_collection(self, crop: str) -> None:
        """Delete a crop's collection entirely (used for force re-ingestion)."""
        try:
            client = self._get_client()
            name = self.collection_name(crop)
            client.delete_collection(name)
            log.info("Deleted collection '%s'.", name)
        except Exception as e:
            log.warning("Could not delete collection '%s': %s", self.collection_name(crop), e)

    def get_or_create_collection(self, crop: str):
        client = self._get_client()
        name = self.collection_name(crop)
        # cosine distance is standard for sentence-transformers embeddings
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        log.debug("Collection '%s' ready (%d items)", name, collection.count())
        return collection

    def upsert(self, crop: str, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None:
        collection = self.get_or_create_collection(crop)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log.info("Upserted %d chunks into collection '%s'", len(ids), self.collection_name(crop))

    def query(self, crop: str, embedding: list[float], n_results: int = 5
              ) -> Optional[dict]:
        """Run a similarity query. Returns raw ChromaDB results dict or None."""
        if not self.collection_exists(crop):
            return None
        collection = self.get_or_create_collection(crop)
        try:
            count = collection.count()
            if count == 0:
                return None
            return collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("ChromaDB query failed for crop '%s': %s", crop, e)
            return None

    def query_keyword_filtered(
        self,
        crop: str,
        embedding: list[float],
        term: str,
        n_results: int = 3,
    ) -> Optional[dict]:
        """Keyword-filtered semantic search: rank by embedding similarity but
        restrict to documents that contain *term* as a substring.

        ChromaDB's where_document={"$contains": term} acts as a hard filter
        (case-sensitive). Semantic ranking still applies within the matching set,
        so we get the most semantically relevant documents that also contain the
        keyword — the best of both worlds.
        """
        if not self.collection_exists(crop):
            return None
        collection = self.get_or_create_collection(crop)
        try:
            count = collection.count()
            if count == 0:
                return None
            return collection.query(
                query_embeddings=[embedding],
                where_document={"$contains": term},
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # where_document filter may fail if no documents match — that's OK
            log.debug("Keyword-filtered query failed for term='%s' crop='%s': %s", term, crop, e)
            return None

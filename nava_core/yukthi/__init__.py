"""Yukthi — RAG knowledge retrieval module for NAVA.

Intentionally minimal init: router and retriever depend on mozhi (chat client),
so importing them at package level would create circular imports when mozhi
imports back from yukthi. Import them directly from their submodules instead.

    from nava_core.yukthi.retriever import RAGRetriever, RAGChunk
    from nava_core.yukthi.router import QueryRouter
    from nava_core.yukthi.pipeline import RAGPipeline
"""

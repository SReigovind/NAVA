"""Text and PDF chunking for RAG ingestion.

Produces deterministic, semantically coherent chunks from:
  - Structured .txt files (PlantVillage-style: disease/pest entries)
  - PDF files (KAU guidebook: page+paragraph block extraction via PyMuPDF)

Adding a new file format
------------------------
Register a handler function in CHUNKER_REGISTRY at the bottom of this file:

    def chunk_csv(path, crop):
        ...
        return [Chunk(...), ...]

    CHUNKER_REGISTRY[".csv"] = chunk_csv

chunk_file() and pipeline.py will automatically pick it up.
SUPPORTED_EXTENSIONS is derived from CHUNKER_REGISTRY — no other changes needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Chunk:
    text: str
    source: str       # filename only (not full path)
    section: str      # heading or entry name
    chunk_index: int  # sequential within this source file


# ── TXT chunking ────────────────────────────────────────────────────────────

# Section headings found in PlantVillage-style documents
_TXT_SECTION_HEADERS = re.compile(
    r"^(Symptoms|Cause|Comments|Management|Description|Uses\s*&?\s*Benefits|"
    r"Varieties|Propagation|Basic\s*Requirement|Seeding|General\s*Care|"
    r"Harvesting|References|Diseases|Pests|Category\s*:|Common\s*Pests)\s*$",
    re.IGNORECASE,
)

# Disease/Pest entry line: e.g. "Panama disease (Fusarium wilt) Fusarium oxysporum"
# These are lines that start a new disease/pest record.
_ENTRY_LINE = re.compile(
    r"^(?:Anthracnose|Black\s+sigatoka|Cigar\s+end|Cordana|Panama|Rhizome|"
    r"Yellow\s+sigatoka|Banana\s+bacterial|Moko|Banana\s+mosaic|Bunchy\s+top|"
    r"Banana\s+aphid|Banana\s+skipper|Banana\s+weevil|Coconut\s+scale|"
    r"[A-Z][a-z]+\s+(?:disease|rot|wilt|blight|spot|mosaic|streak|top))",
    re.IGNORECASE,
)


def chunk_txt(path: Path, crop: str) -> list[Chunk]:
    """Chunk a PlantVillage-style .txt file into semantic units.

    Strategy:
    - Each disease/pest entry (from its name line through its Management line)
      is one chunk — preserving symptom+cause+management coherence.
    - General agronomy sections (Description, Propagation, etc.) are each
      one chunk (split at section headers).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    source = path.name

    chunks: list[Chunk] = []
    idx = 0
    buffer: list[str] = []
    section = "General"
    in_disease_block = False

    def _flush(label: str) -> None:
        nonlocal idx
        content = "\n".join(buffer).strip()
        if len(content) > 40:  # skip near-empty buffers
            chunks.append(Chunk(text=content, source=source, section=label, chunk_index=idx))
            idx += 1

    for line in lines:
        stripped = line.strip()

        # Detect a new disease/pest entry
        if _ENTRY_LINE.match(stripped) and not stripped.startswith("#"):
            if buffer:
                _flush(section)
            buffer = [stripped]
            section = stripped
            in_disease_block = True
            continue

        # Detect a top-level section header
        if _TXT_SECTION_HEADERS.match(stripped):
            if in_disease_block:
                # Don't break disease blocks on sub-headers; accumulate
                buffer.append(line)
                continue
            if buffer:
                _flush(section)
            buffer = []
            section = stripped
            in_disease_block = False
            continue

        buffer.append(line)

    if buffer:
        _flush(section)

    return chunks


# ── PDF chunking ─────────────────────────────────────────────────────────────

# Malayalam Unicode block (U+0D00–U+0D7F) — used to detect non-English pages
_MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")

# Minimum fraction of Malayalam chars to consider a block non-English
_MALAYALAM_THRESHOLD = 0.4

# Target token count per PDF chunk (rough: 1 token ≈ 4 chars)
_PDF_TARGET_CHARS = 2000   # ~500 tokens
_PDF_OVERLAP_CHARS = 200   # ~50 tokens overlap


def _is_malayalam(text: str) -> bool:
    if not text:
        return False
    mal_count = len(_MALAYALAM_RE.findall(text))
    return mal_count / len(text) > _MALAYALAM_THRESHOLD


def chunk_pdf(path: Path, crop: str) -> list[Chunk]:
    """Chunk a PDF using PyMuPDF block-level extraction.

    Strategy:
    - Extract text blocks per page, preserving reading order.
    - Detect section headings by font-size heuristics (larger = heading).
    - Skip blocks that are predominantly Malayalam.
    - Accumulate blocks into chunks targeting ~500 tokens, with 50-token overlap.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF chunking. Install it with: pip install pymupdf")

    source = path.name
    chunks: list[Chunk] = []
    idx = 0
    current_section = "Introduction"
    accumulator = ""
    last_tail = ""  # overlap buffer

    doc = fitz.open(str(path))

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue

            block_text_parts = []
            max_font_size = 0.0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > max_font_size:
                        max_font_size = size
                    block_text_parts.append(span.get("text", ""))

            block_text = " ".join(block_text_parts).strip()
            if not block_text or len(block_text) < 10:
                continue
            if _is_malayalam(block_text):
                continue

            # Heuristic: larger font → section heading
            is_heading = max_font_size >= 13.0 and len(block_text) < 120

            if is_heading:
                # Flush current accumulator as a chunk
                content = (last_tail + " " + accumulator).strip()
                if len(content) > 80:
                    chunks.append(Chunk(
                        text=content,
                        source=source,
                        section=current_section,
                        chunk_index=idx,
                    ))
                    idx += 1
                    # Carry last ~200 chars as overlap
                    last_tail = content[-_PDF_OVERLAP_CHARS:] if len(content) > _PDF_OVERLAP_CHARS else content
                accumulator = ""
                current_section = block_text[:100]  # truncate very long headings
                continue

            accumulator += " " + block_text

            # Flush when accumulator exceeds target size
            if len(accumulator) >= _PDF_TARGET_CHARS:
                content = (last_tail + " " + accumulator).strip()
                chunks.append(Chunk(
                    text=content,
                    source=source,
                    section=current_section,
                    chunk_index=idx,
                ))
                idx += 1
                last_tail = accumulator[-_PDF_OVERLAP_CHARS:]
                accumulator = ""

    # Flush any remaining text
    if accumulator.strip():
        content = (last_tail + " " + accumulator).strip()
        if len(content) > 80:
            chunks.append(Chunk(
                text=content,
                source=source,
                section=current_section,
                chunk_index=idx,
            ))

    doc.close()
    return chunks


# ── Format registry ──────────────────────────────────────────────────────────
# Maps file extension → chunker function (path, crop) -> list[Chunk]
# To support a new format: add an entry here. Nothing else needs to change.

CHUNKER_REGISTRY: dict[str, Callable[[Path, str], list[Chunk]]] = {
    ".txt": chunk_txt,
    ".pdf": chunk_pdf,
}

# Derived from the registry — used by pipeline._find_sources() to filter files
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(CHUNKER_REGISTRY.keys())


def chunk_file(path: Path, crop: str) -> list[Chunk]:
    """Dispatch to the appropriate chunker based on file extension.

    Returns an empty list for unsupported formats (logged as a warning).
    """
    handler = CHUNKER_REGISTRY.get(path.suffix.lower())
    if handler is None:
        import logging
        logging.getLogger("yukthi.chunker").warning(
            "No chunker registered for '%s' — skipping %s. "
            "Register a handler in CHUNKER_REGISTRY to support this format.",
            path.suffix, path.name,
        )
        return []
    return handler(path, crop)

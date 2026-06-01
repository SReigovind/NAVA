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

_COMMON_HEADERS = re.compile(
    r"^(Symptoms|Cause|Comments|Management|Description|Uses\s*&?\s*Benefits|"
    r"Varieties|Propagation|Basic\s*Requirement|Seeding|General\s*Care|"
    r"Harvesting|References|Diseases|Pests|Category\s*:|Common\s*Pests|"
    r"Biology|Effect\s+on\s+Crop|Mode\s+of\s+spread.*|Favourable\s+conditions.*|"
    r"Identification.*|Nature\s+of\s+Damage.*|Life\s+history|Importance|Management.*)\s*$",
    re.IGNORECASE,
)


def chunk_txt(path: Path, crop: str) -> list[Chunk]:
    """Chunk a .txt file into semantic units using a dual-strategy approach.

    Strategy A (Delimited): If the file contains '---' on a blank line, it's treated
    as a structured institutional document where each block is a separate topic.
    Strategy B (Prose): Otherwise, it's treated as unstructured prose (e.g., Wikipedia)
    and chunked by paragraphs and heuristic headings.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    source = path.stem

    has_delimiter = any(line.strip() == "---" for line in lines)

    if has_delimiter:
        return _chunk_txt_delimited(lines, source)
    else:
        return _chunk_txt_prose(lines, source)


def _chunk_txt_delimited(lines: list[str], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0

    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == "---":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        main_topic = "General"
        # The first non-empty line in a block is the Main Topic
        for i, line in enumerate(block):
            if line.strip():
                main_topic = line.strip()
                block = block[i + 1:]
                break

        buffer = []
        sub_header = "Overview"

        def _flush(label: str):
            nonlocal idx
            content = "\n".join(buffer).strip()
            if len(content) > 40:
                chunks.append(Chunk(
                    text=content,
                    source=source,
                    section=f"{main_topic} - {label}",
                    chunk_index=idx,
                ))
                idx += 1

        for line in block:
            stripped = line.strip()
            # Heuristic for sub-header
            if stripped and len(stripped) < 60 and (
                stripped.endswith(":") or _COMMON_HEADERS.match(stripped)
            ):
                if buffer:
                    _flush(sub_header)
                buffer = []
                sub_header = stripped.rstrip(":")
                continue

            buffer.append(line)

        if buffer:
            _flush(sub_header)

    return chunks


def _chunk_txt_prose(lines: list[str], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    buffer = []
    current_header = "General"
    part_count = 1

    def _flush(label: str, part: int):
        nonlocal idx
        content = "\n".join(buffer).strip()
        if len(content) > 40:
            sec_name = f"{label} (Part {part})" if part > 1 else label
            chunks.append(Chunk(
                text=content,
                source=source,
                section=sec_name,
                chunk_index=idx,
            ))
            idx += 1

    for line in lines:
        stripped = line.strip()

        is_header = False
        if stripped and len(stripped) < 60 and not stripped.endswith("."):
            if _COMMON_HEADERS.match(stripped) or (len(stripped.split()) <= 4 and stripped[0].isupper()):
                is_header = True

        if is_header and len("\n".join(buffer).strip()) > 40:
            _flush(current_header, part_count)
            buffer = []
            current_header = stripped
            part_count = 1
            continue
        elif is_header and len("\n".join(buffer).strip()) <= 40:
            # Overwrite header if we haven't accumulated much text
            current_header = stripped
            buffer = []
            continue

        buffer.append(line)

        # Soft split for very large sections on paragraph boundaries
        if len("\n".join(buffer)) > 1200 and not stripped:
            _flush(current_header, part_count)
            buffer = []
            part_count += 1

    if buffer:
        _flush(current_header, part_count)

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

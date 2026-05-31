# Yukthi: `chunker.py` and `keywords.py`

> **Subfolder:** `code/`
> **Cross-references:** [10_yukthi_pipeline_and_store.md](10_yukthi_pipeline_and_store.md) | [technical/05_rag_and_knowledge_grounding.md](../technical/05_rag_and_knowledge_grounding.md)

**Source files:**
- [`chunker.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/chunker.py)
- [`keywords.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/keywords.py)

---

## `chunker.py` — Document Chunking

### The `Chunk` Dataclass

```python
@dataclass
class Chunk:
    text: str
    source: str      # filename only (not full path)
    section: str     # heading or entry name
    chunk_index: int # sequential within this source file
```

A chunk is the unit of retrieval — the piece of text that gets stored in ChromaDB and returned to the LLM. The `section` field (the heading the chunk falls under) is stored in ChromaDB metadata and displayed in the UI's source citation tooltips. `chunk_index` is used in the deterministic ID: `"{source}_{chunk_index}"`.

### The Format Registry Pattern

```python
CHUNKER_REGISTRY: dict[str, Callable[[Path, str], list[Chunk]]] = {
    ".txt": chunk_txt,
    ".pdf": chunk_pdf,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(CHUNKER_REGISTRY.keys())

def chunk_file(path: Path, crop: str) -> list[Chunk]:
    handler = CHUNKER_REGISTRY.get(path.suffix.lower())
    if handler is None:
        logging.warning("No chunker registered for '%s'", path.suffix)
        return []
    return handler(path, crop)
```

This is the **Registry Pattern**: a dictionary maps file extensions to handler functions. Adding support for a new format (e.g., `.csv`, `.docx`) requires:
1. Writing a `chunk_csv(path, crop)` function
2. Adding `".csv": chunk_csv` to `CHUNKER_REGISTRY`

Nothing else changes. The pipeline, the CLI, and the ingestion logic all call `chunk_file()` — they don't know or care which format is being processed.

**Why derive `SUPPORTED_EXTENSIONS` from the registry?** The pipeline uses `SUPPORTED_EXTENSIONS` to filter files in `_find_sources()`. Deriving it from the registry means adding a new format automatically makes its files discoverable, with zero additional changes.

---

## `chunk_txt()` — PlantVillage-Style Text Chunking

The `.txt` source files follow a PlantVillage-style format: structured text with named disease/pest entries and section headers like "Symptoms", "Management", "Cause".

The chunking strategy preserves disease coherence: everything from a disease name line through its Management section stays in one chunk. This is critical for RAG: a query about "Black Sigatoka management" should retrieve a chunk that contains *both* the symptom description and the management protocol, not two separate chunks where one describes symptoms and the other describes management.

**The two regex patterns:**

```python
# Section headings
_TXT_SECTION_HEADERS = re.compile(
    r"^(Symptoms|Cause|Comments|Management|Description|Uses\s*&?\s*Benefits|...)$",
    re.IGNORECASE,
)

# Disease/pest entry lines
_ENTRY_LINE = re.compile(
    r"^(?:Anthracnose|Black\s+sigatoka|Cigar\s+end|...|[A-Z][a-z]+\s+(?:disease|rot|wilt|...))...",
    re.IGNORECASE,
)
```

`_ENTRY_LINE` has two parts:
1. Explicit named diseases (for known common diseases in NAVA's target crops)
2. A heuristic pattern: `[A-Z][a-z]+` followed by common disease suffix words (`disease`, `rot`, `wilt`, `blight`, etc.) — catches novel disease entries not in the explicit list

**The line-by-line state machine:**

```python
for line in lines:
    stripped = line.strip()
    
    if _ENTRY_LINE.match(stripped):    # New disease entry
        if buffer:
            _flush(section)            # Flush previous chunk
        buffer = [stripped]
        section = stripped
        in_disease_block = True
        
    elif _TXT_SECTION_HEADERS.match(stripped):
        if in_disease_block:
            buffer.append(line)        # Don't split disease blocks on sub-headers
        else:
            if buffer:
                _flush(section)        # New general section
            section = stripped
            in_disease_block = False
            
    else:
        buffer.append(line)            # Accumulate
```

The `in_disease_block` flag prevents splitting disease entries at section headers within them (e.g., "Symptoms" and "Management" within the same Black Sigatoka entry). When `in_disease_block=True` and a section header is seen, it's accumulated into the current buffer rather than triggering a flush. When a new `_ENTRY_LINE` is seen, the previous disease block is flushed and a new block begins.

**The `_flush()` function:**
```python
def _flush(label: str) -> None:
    content = "\n".join(buffer).strip()
    if len(content) > 40:   # skip near-empty buffers
        chunks.append(Chunk(text=content, source=source, section=label, chunk_index=idx))
        idx += 1
```

The 40-character minimum prevents adding tiny one-line chunks (category headings, separator lines) that provide no retrieval value.

---

## `chunk_pdf()` — KAU Guidebook PDF Chunking

KAU (Kerala Agricultural University) Package of Practices documents are PDFs with rich typographic structure: large-font section headings, numbered subsections, paragraph-level content.

**PyMuPDF block extraction:**
```python
doc = fitz.open(str(path))
for page in doc:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                max_font_size = max(max_font_size, span["size"])
                block_text_parts.append(span["text"])
```

PyMuPDF's `"dict"` output mode returns a structured representation: pages → blocks → lines → spans. Each span has a `size` (font size in points). This structure allows NAVA to detect headings by font size — a heuristic that works reliably for the KAU guidebook layout.

**Malayalam detection and filtering:**
```python
_MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")

def _is_malayalam(text: str) -> bool:
    mal_count = len(_MALAYALAM_RE.findall(text))
    return mal_count / len(text) > 0.4  # >40% Malayalam chars
```

KAU guidebooks are sometimes bilingual (English + Malayalam). Blocks with more than 40% Malayalam Unicode characters are skipped. The threshold (40%) prevents false positives from words that contain a few Unicode characters (e.g., proper nouns with diacritics).

**Why skip Malayalam?** The retrieval embedding model (`BAAI/bge-small-en-v1.5`) is optimised for English text. Embedding Malayalam text would produce poor embeddings that don't cluster meaningfully with English queries. Including Malayalam blocks would add noise to the knowledge base without improving retrieval quality.

**Heading detection:**
```python
is_heading = max_font_size >= 13.0 and len(block_text) < 120
```

`font_size >= 13.0` — PDF body text is typically 10–12pt; headings are 13pt or larger. `len(block_text) < 120` — headings are short; paragraphs that happen to be in a slightly larger font (e.g., a styled callout box) are not treated as headings.

**Overlap between chunks:**
```python
_PDF_TARGET_CHARS = 2000   # ~500 tokens
_PDF_OVERLAP_CHARS = 200   # ~50 tokens overlap

if len(accumulator) >= _PDF_TARGET_CHARS:
    content = (last_tail + " " + accumulator).strip()
    chunks.append(Chunk(...))
    last_tail = accumulator[-_PDF_OVERLAP_CHARS:]  # carry last 200 chars
    accumulator = ""
```

Each chunk includes the last 200 characters of the previous chunk (`last_tail`). This overlap prevents information loss at chunk boundaries — a sentence that spans the boundary of two chunks will appear in both, ensuring queries that match that sentence retrieve at least one relevant chunk.

**Why overlap for PDF but not for TXT?** The TXT chunker uses semantic boundaries (disease entry lines, section headers) — there's no information loss at boundaries because each chunk is a complete semantic unit. PDF chunking is positional (target character count), so boundary information loss is possible and overlap mitigates it.

---

## `keywords.py` — LLM-Based Keyword Extraction

### `KeywordExtractor.extract()`

```python
def extract(self, enriched_query: str) -> list[str]:
    user_prompt = (
        f"Extract exactly {self.n} search keywords from this agricultural query context:\n\n"
        f"{enriched_query}\n\n"
        f"Output {self.n} keywords, one per line:"
    )
    reply, error = self.client.send(
        prompt,
        model_override=self.model,
        temperature_override=0.0,
        max_new_tokens_override=25,
    )
    if error or not reply:
        return []   # caller falls back to heuristic extraction
    return self._parse_keywords(reply)
```

**`max_new_tokens=25`:** 3 keywords × approximately 5 tokens each (word + spaces + newline) = 15 tokens. The `25` token budget gives headroom for slightly longer disease names while still preventing verbose output.

**`temperature=0.0`:** Keywords should be deterministic — the same query should produce the same keywords on every call. Temperature 0 selects the highest-probability token at each step (greedy decoding).

**Returns `[]` on failure:** The caller (`RAGRetriever.query()`) treats an empty keyword list as "fall back to heuristic". This means the keyword extraction is a best-effort enhancement, not a required step.

### `_parse_keywords()`

```python
def _parse_keywords(self, reply: str) -> list[str]:
    _GENERIC = frozenset({"banana", "crop", "plant", "field", ...})
    
    for line in reply.strip().splitlines():
        clean = re.sub(r"^[\s\d\.\-\*\•]+", "", line).strip()
        if ":" in clean:
            clean = clean.split(":", 1)[0].strip()  # strip explanations after colon
        if not clean or len(clean) < 3:
            continue
        if clean.lower() in _GENERIC:
            continue
        keywords.append(clean)
        if len(keywords) >= self.n:
            break
    return keywords
```

The parsing is defensive:
- `re.sub(r"^[\s\d\.\-\*\•]+", "", line)` removes leading bullets, numbers, punctuation — even if the model adds them despite the instruction not to
- `split(":", 1)[0]` removes explanations the model sometimes appends after a colon (e.g., "Sigatoka: a fungal disease")
- `len(clean) < 3` filters out single-character or two-character tokens
- `_GENERIC` filters words that would produce useless keyword searches (e.g., "banana" matches every document in the banana collection, providing no additional precision)

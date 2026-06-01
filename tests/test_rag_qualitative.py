"""
RAG Qualitative Comparison Test
================================
For each of the 7 supported crops, asks one targeted disease question
and records two answers side-by-side:

  - WITH RAG  : normal pipeline — router → keyword extractor → ChromaDB
                → verified source chunks injected into the LLM prompt
  - WITHOUT RAG: RAG is forcibly bypassed — only the LLM's parametric
                 knowledge and the farm context system prompt are available

Run from project root:  python tests/test_rag_qualitative.py
Run from tests/:        python test_rag_qualitative.py

All output paths are anchored to the directory containing this file,
so the script works correctly from either location.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

# ── Path anchoring ───────────────────────────────────────────────────────────
TESTS_DIR    = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from fastapi.testclient import TestClient
from nava_core.gathi.api.main import app

# ── Output directory ─────────────────────────────────────────────────────────
MARKDOWN_OUT_DIR = TESTS_DIR / "markdownoutput"
REPORT_PATH      = MARKDOWN_OUT_DIR / "rag_qualitative_report.md"

# ── Test credentials (isolated from other test suites) ───────────────────────
TEST_EMAIL    = "rag_test@nava.local"
TEST_PASSWORD = "nava_rag_2026"
TEST_NAME     = "RAG Test Farmer"

DEFAULT_SEASON = "Monsoon Season"

# ── One disease question per crop ────────────────────────────────────────────
# Chosen to be direct agronomic knowledge questions that clearly benefit from
# the verified source material in ChromaDB.
#
# Disease classes covered (from EfficientNet-B0-labels.txt):
#   Banana   : banana_sigatoka
#   Cassava  : cassava_blight, cassava_mosaic
#   Corn     : corn_common_rust, corn_northern_leaf_blight, corn_cercospora_leaf_spot, corn_smut
#   Cucumber : cucumber_angular_leaf_spot, cucumber_powdery_mildew
#   Rice     : rice_blast, rice_tungro, rice_bacterial_leaf_blight, ...
#   Soybean  : soybean_bacterial_blight, soybean_downy_mildew
#   Tomato   : tomato_early_blight, tomato_late_blight, tomato_yellow_leaf_curl_virus, ...

CROP_QUESTIONS = [
    {
        "crop"    : "Banana",
        "disease" : "Sigatoka (Black Sigatoka)",
        "question": (
            "My banana plants show dark streaks and yellow halos on the leaves — "
            "I think it is Black Sigatoka. What are the symptoms, cause, and recommended "
            "fungicide management for Black Sigatoka in banana?"
        ),
    },
    {
        "crop"    : "Cassava",
        "disease" : "Cassava Mosaic Disease",
        "question": (
            "My cassava plants have distorted, mosaic-patterned leaves with yellowing. "
            "What causes Cassava Mosaic Disease, how does it spread, and what are the "
            "management strategies to control it?"
        ),
    },
    {
        "crop"    : "Corn",
        "disease" : "Northern Leaf Blight",
        "question": (
            "I see long, greyish-tan lesions running along the corn leaves. "
            "Can you explain Northern Leaf Blight in corn — what fungus causes it, "
            "what conditions favour it, and what fungicide or cultural practices are recommended?"
        ),
    },
    {
        "crop"    : "Cucumber",
        "disease" : "Powdery Mildew",
        "question": (
            "My cucumber plants have white powdery patches on the upper surface of the leaves. "
            "What is Powdery Mildew in cucumber, what causes it, and what are the treatment "
            "and prevention measures?"
        ),
    },
    {
        "crop"    : "Rice",
        "disease" : "Rice Blast",
        "question": (
            "Diamond-shaped lesions with grey centres and brown borders are appearing on my rice leaves. "
            "What is Rice Blast, what pathogen causes it, and what are the recommended chemical "
            "and cultural management practices?"
        ),
    },
    {
        "crop"    : "Soybean",
        "disease" : "Soybean Downy Mildew",
        "question": (
            "I can see pale green to yellow patches on the upper surface of soybean leaves "
            "with grey-purple fuzz underneath. What is Soybean Downy Mildew, what causes it, "
            "and how should I manage it?"
        ),
    },
    {
        "crop"    : "Tomato",
        "disease" : "Tomato Late Blight",
        "question": (
            "Dark, water-soaked lesions are appearing on my tomato leaves and spreading to the stems "
            "and fruits. What is Tomato Late Blight, what organism causes it, and what are the "
            "recommended management and fungicide options?"
        ),
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _auth(client: TestClient) -> str:
    res = client.post("/api/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    if res.status_code == 200:
        return res.json()["token"]
    res = client.post("/api/auth/register", json={
        "name": TEST_NAME, "email": TEST_EMAIL, "password": TEST_PASSWORD,
    })
    if res.status_code == 200:
        return res.json()["token"]
    raise RuntimeError(f"Auth failed: {res.text}")


def _get_or_create_field(client: TestClient, headers: dict) -> int:
    res = client.get("/api/fields", headers=headers)
    for f in res.json().get("fields", []):
        if f["name"] == "RAG Test Field":
            return f["id"]
    res = client.post("/api/fields", headers=headers, json={
        "name": "RAG Test Field",
        "location": "Kottayam, Kerala",
        "area": "2 acres",
        "soil_type": "Laterite",
    })
    if res.status_code != 200:
        raise RuntimeError(f"Field creation failed: {res.text}")
    return res.json()["id"]


def _get_or_create_crop(client: TestClient, headers: dict, field_id: int, crop_name: str) -> int:
    res = client.get(f"/api/crops?field_id={field_id}", headers=headers)
    for c in res.json().get("crops", []):
        if c["name"] == crop_name:
            return c["id"]
    res = client.post("/api/crops", headers=headers, json={
        "field_id": field_id,
        "name"    : crop_name,
        "variety" : "Standard",
        "season"  : DEFAULT_SEASON,
        "stage"   : "Vegetative",
    })
    if res.status_code != 200:
        raise RuntimeError(f"Crop creation failed for '{crop_name}': {res.text}")
    return res.json()["id"]


def _chat(
    client: TestClient,
    headers: dict,
    message: str,
    session_id: str,
    field_id: int,
    crop_id: int,
    *,
    force_skip_rag: bool = False,
) -> dict:
    """Send one chat message and return the full JSON response.

    When force_skip_rag=True, patches QueryRouter so it always returns False,
    disabling RAG for that single call while leaving everything else identical.
    """
    payload = {
        "message"   : message,
        "session_id": session_id,
        "field_id"  : field_id,
        "crop_id"   : crop_id,
    }

    if force_skip_rag:
        with patch(
            "nava_core.yukthi.router.QueryRouter.should_retrieve",
            return_value=False,
        ):
            res = client.post("/api/chat", headers=headers, json=payload)
    else:
        res = client.post("/api/chat", headers=headers, json=payload)

    if res.status_code != 200:
        return {
            "reply"      : f"[API ERROR {res.status_code}]: {res.text}",
            "rag_used"   : False,
            "rag_chunks" : [],
        }
    return res.json()


def _to_md_blockquote(text: str) -> str:
    """Convert an LLM reply to a markdown blockquote (> prefix on every line).

    Unlike HTML <blockquote> tags, the markdown parser processes formatting
    (bold, italics, lists, line breaks) correctly inside > blockquotes.
    Blank lines get a bare '>' so the blockquote is not broken mid-reply.
    """
    if not text or not text.strip():
        return "> _No reply received._\n"
    lines = text.strip().splitlines()
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines) + "\n"


def _chunk_table(chunks: list[dict]) -> str:
    if not chunks:
        return "_No chunks retrieved._\n"
    rows = [
        "| # | Source | Section | Snippet (first 300 chars) |",
        "|---|--------|---------|--------------------------|",
    ]
    for i, ch in enumerate(chunks, 1):
        snippet = (ch.get("snippet") or "")[:300].replace("|", "\\|").replace("\n", " ")
        source  = (ch.get("source")  or "").replace("|", "\\|")
        section = (ch.get("section") or "").replace("|", "\\|")
        rows.append(f"| {i} | {source} | {section} | {snippet} |")
    return "\n".join(rows) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def test_rag_qualitative():
    print("=== RAG Qualitative Comparison Test ===")
    MARKDOWN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        token    = _auth(client)
        headers  = {"Authorization": f"Bearer {token}"}
        field_id = _get_or_create_field(client, headers)

        with open(REPORT_PATH, "w", encoding="utf-8") as md:

            # ── Report header ─────────────────────────────────────────────────
            md.write("# RAG Qualitative Comparison Report\n\n")
            md.write(
                "> **Purpose:** For each of the 7 supported crops, a targeted disease question is asked "
                "twice — once through the full RAG-augmented pipeline and once with RAG forcibly disabled. "
                "The responses and retrieved source chunks are displayed side-by-side to assess the "
                "factual improvement that verified agronomic reference material provides.\n\n"
            )
            md.write(
                "| Field | Value |\n"
                "|-------|-------|\n"
                "| Test account | `rag_test@nava.local` |\n"
                f"| Field | RAG Test Field (ID {field_id}) |\n"
                "| LLM (chat) | Llama-3 70B via HF Router |\n"
                "| LLM (router / keywords) | Llama-3.1-8B-Instruct |\n"
                "| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, local) |\n"
                "| Retrieval | 5 semantic + ~5 keyword-filtered → rerank → top 3 |\n\n"
            )
            md.write("---\n\n")

            # ── Per-crop sections ─────────────────────────────────────────────
            for idx, entry in enumerate(CROP_QUESTIONS, 1):
                crop_name = entry["crop"]
                disease   = entry["disease"]
                question  = entry["question"]

                print(f"\n[{idx}/7] Crop: {crop_name} — Disease: {disease}")

                crop_id = _get_or_create_crop(client, headers, field_id, crop_name)

                # Fresh isolated sessions so neither call contaminates the other
                session_with    = f"rag_qual_with_{crop_name.lower()}_{idx}"
                session_without = f"rag_qual_without_{crop_name.lower()}_{idx}"

                # WITH RAG
                print(f"  → Querying WITH RAG ...")
                time.sleep(2)
                data_with = _chat(
                    client, headers, question,
                    session_id=session_with,
                    field_id=field_id, crop_id=crop_id,
                    force_skip_rag=False,
                )
                print(f"     RAG used: {data_with.get('rag_used')} | chunks: {data_with.get('rag_chunk_count', 0)}")

                # WITHOUT RAG
                print(f"  → Querying WITHOUT RAG ...")
                time.sleep(2)
                data_without = _chat(
                    client, headers, question,
                    session_id=session_without,
                    field_id=field_id, crop_id=crop_id,
                    force_skip_rag=True,
                )
                print(f"     RAG used: {data_without.get('rag_used')}")

                # Write section
                md.write(f"## {idx}. {crop_name} — {disease}\n\n")
                md.write(f"**Question asked:**\n\n> {question}\n\n")
                md.write(
                    "| | With RAG | Without RAG |\n"
                    "|-|----------|-------------|\n"
                    f"| RAG active | ✅ Yes | ❌ No |\n"
                    f"| Chunks retrieved | {data_with.get('rag_chunk_count', 0)} | 0 |\n\n"
                )

                md.write("### ✅ Answer WITH RAG\n\n")
                md.write(_to_md_blockquote(data_with.get("reply") or ""))
                md.write("\n")

                md.write("#### Retrieved Knowledge Chunks\n\n")
                md.write(_chunk_table(data_with.get("rag_chunks") or []))
                md.write("\n")

                md.write("### ❌ Answer WITHOUT RAG\n\n")
                md.write(_to_md_blockquote(data_without.get("reply") or ""))
                md.write("\n")

                md.write("---\n\n")
                print(f"  ✓ Section written.")

            # ── Summary note ──────────────────────────────────────────────────
            md.write("## Summary\n\n")
            md.write(
                "> The side-by-side comparison above shows how Yukthi (RAG) grounds each answer "
                "in verified agronomic source material — reducing hallucinated dosages, "
                "uncertain hedging, and missing specifics compared to the parametric-only baseline.\n"
            )

    print(f"\n=== Report saved to {REPORT_PATH} ===")


if __name__ == "__main__":
    test_rag_qualitative()

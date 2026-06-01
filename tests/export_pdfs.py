"""PDF export script — converts all Markdown reports to styled PDFs.

Run from project root:  python tests/export_pdfs.py
Run from tests/:        python export_pdfs.py

All paths are anchored to the directory containing this file,
so the script works correctly from either location.
"""
import os
import subprocess
from pathlib import Path

# ── Path anchoring ────────────────────────────────────────────────────────────
TESTS_DIR    = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent

MARKDOWN_DIR = TESTS_DIR / "markdownoutput"
PDF_DIR      = TESTS_DIR / "pdfoutput"
MD2PDF_PATH  = PROJECT_ROOT / ".nava" / "bin" / "md2pdf"


def export_pdfs():
    print("--- Exporting Markdown Reports to PDF ---")

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    reports = [
        "disease_report.md",
        "vnir_report.md",
        "chat_report.md",
        "rag_qualitative_report.md",
    ]

    # Write the CSS into markdownoutput/ so that md2pdf (run from there)
    # resolves relative image paths (../imageoutputs/...) correctly.
    css_path = MARKDOWN_DIR / "pdf_styles.css"
    with open(css_path, "w") as f:
        f.write("""
        @page { size: A4 portrait; margin: 1.5cm; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.6; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; table-layout: fixed; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: left; word-wrap: break-word; overflow-wrap: break-word; }
        th { background-color: #f2f2f2; font-weight: bold; }
        td { max-width: 120px; }
        pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 12px; border-radius: 6px; font-size: 11px; }
        img { max-width: 100%; height: auto; }
        blockquote {
            border-left: 4px solid #888;
            margin: 0.5em 0;
            padding: 8px 16px;
            background-color: #f9f9f9;
            color: #333;
        }
        blockquote p { margin: 0.4em 0; }
        blockquote ul, blockquote ol { margin: 0.4em 0 0.4em 1.2em; padding: 0; }
        blockquote li { margin-bottom: 0.2em; }
        """)


    for r in reports:
        md_full  = MARKDOWN_DIR / r
        pdf_file = r.replace(".md", ".pdf")
        pdf_full = PDF_DIR / pdf_file

        if not md_full.exists():
            print(f"Skipping {r} (not found in markdownoutput/). Run its test script first.")
            continue

        print(f"Converting {r} -> pdfoutput/{pdf_file} ...")

        # Run md2pdf from inside markdownoutput/ so relative image src paths
        # (../imageoutputs/...) embedded in the markdown resolve correctly.
        result = subprocess.run(
            [
                str(MD2PDF_PATH),
                "-i", r,
                "-o", str(pdf_full),
                "-c", "pdf_styles.css",
                "-e", "tables",
            ],
            cwd=str(MARKDOWN_DIR),
        )

        if pdf_full.exists() and result.returncode == 0:
            print(f"  -> Success: tests/pdfoutput/{pdf_file}")
        else:
            print(f"  -> Error generating tests/pdfoutput/{pdf_file} (returncode={result.returncode})")


if __name__ == "__main__":
    export_pdfs()

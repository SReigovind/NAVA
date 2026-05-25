import os
import subprocess

def export_pdfs():
    print("--- Exporting Markdown Reports to PDF ---")
    
    # We must run md2pdf from inside the tests directory so that relative paths like './outputs/...' resolve correctly.
    tests_dir = os.path.abspath("tests")
    md2pdf_path = os.path.abspath(".nava/bin/md2pdf")
    
    reports = ["disease_report.md", "vnir_report.md", "chat_report.md"]
    
    # Create a basic CSS file for nicer PDF rendering (tables and pre tags)
    css_path = os.path.join(tests_dir, "pdf_styles.css")
    with open(css_path, "w") as f:
        f.write("""
        @page { size: A4 portrait; margin: 1.5cm; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.5; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; table-layout: fixed; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: left; word-wrap: break-word; overflow-wrap: break-word; }
        th { background-color: #f2f2f2; font-weight: bold; }
        /* Add specific max-widths for table cells to force wrapping of long class names */
        td { max-width: 120px; }
        pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 12px; border-radius: 6px; font-size: 11px; }
        img { max-width: 100%; height: auto; }
        """)
        
    for r in reports:
        md_file = r
        pdf_file = r.replace(".md", ".pdf")
        
        md_full = os.path.join(tests_dir, md_file)
        if not os.path.exists(md_full):
            print(f"Skipping {md_file} (not found). Run its test script first.")
            continue
            
        print(f"Converting {md_file} to PDF...")
        # Run md2pdf inside the tests directory
        result = subprocess.run(
            [md2pdf_path, "-i", md_file, "-o", pdf_file, "-c", "pdf_styles.css", "-e", "tables"],
            cwd=tests_dir
        )
        
        pdf_full = os.path.join(tests_dir, pdf_file)
        if os.path.exists(pdf_full) and result.returncode == 0:
            print(f"  -> Success: tests/{pdf_file}")
        else:
            print(f"  -> Error generating tests/{pdf_file}")

if __name__ == "__main__":
    export_pdfs()

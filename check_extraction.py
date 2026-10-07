"""
Standalone diagnostic script — checks extraction quality across several
pages spread through the document, not just the first couple.

Run: python check_extraction.py
Delete this file once we've confirmed extraction quality is acceptable;
it's a throwaway diagnostic, not part of the real pipeline.
"""

from src.ingestion.pdf_loader import load_all_pdfs

pages = load_all_pdfs("data")
print(f"Total pages loaded: {len(pages)}\n")

# Print per-file counts
counts = {}
for p in pages:
    counts[p.source_file] = counts.get(p.source_file, 0) + 1
for filename, count in counts.items():
    print(f"  {filename}: {count} pages")

# Sample a handful of pages spread across the full list, not just the start
sample_indices = [i for i in [0, 5, 10, 20, 30, 40, 50] if i < len(pages)]

for idx in sample_indices:
    p = pages[idx]
    print(f"\n{'=' * 60}")
    print(f"pages[{idx}] -- {p.source_file} / page {p.page_number}")
    print("=" * 60)
    print(p.text[:400])
    
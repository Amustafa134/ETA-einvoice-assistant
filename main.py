"""
Central entry point for eta-einvoice-assistant.

Run this file to execute the pipeline. Right now it only runs ingestion
(loading + extracting text from the PDFs) since that's the only stage
we've built — we'll extend this as we add chunking, embedding, retrieval,
and generation.

Usage:
    python main.py
"""

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.cleaner import strip_repeated_boilerplate


def run_ingestion():
    print("=" * 50)
    print("STAGE: Ingestion")
    print("=" * 50)

    pages = load_all_pdfs("data")
    print(f"Loaded {len(pages)} non-empty pages from data/\n")

    pages = strip_repeated_boilerplate(pages)
    print(f"{len(pages)} pages remain after boilerplate cleanup\n")

    # Show a quick per-file breakdown so you can eyeball whether the page
    # counts look right for each document.
    counts: dict[str, int] = {}
    for p in pages:
        counts[p.source_file] = counts.get(p.source_file, 0) + 1
    for filename, count in counts.items():
        print(f"  {filename}: {count} pages")

    print("\n--- Sample: first page of extracted text ---")
    if pages:
        print(pages[0].text[:300])

    return pages


def main():
    pages = run_ingestion()

    # TODO next stages, added as we build them:
    # chunks = chunk_pages(pages)
    # embed_and_store(chunks)
    # (then retrieval + generation go in their own scripts / the FastAPI app)

    print(f"\nDone. {len(pages)} pages ready for the next stage (chunking).")


if __name__ == "__main__":
    main()
    
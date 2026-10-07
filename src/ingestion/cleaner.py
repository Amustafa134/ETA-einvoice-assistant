"""
Post-OCR cleanup: strips repeated header/footer boilerplate.

OCR sometimes misreads a page's watermark, logo text, or document-control
header as garbled text (e.g. "9/1 -_*3 1015139371901..."). Since this
junk repeats near-identically across many pages while real body content
doesn't, we detect lines that show up on an unusually large fraction of
pages and drop them.
"""

from collections import Counter

from src.ingestion.pdf_loader import PageText


def strip_repeated_boilerplate(pages: list[PageText], threshold: float = 0.3) -> list[PageText]:
    """
    Remove lines that appear (near-)identically on more than `threshold`
    fraction of all pages -- a strong signal they're header/footer noise,
    not real content.
    """
    line_counts: Counter[str] = Counter()
    for p in pages:
        # count each distinct line once per page, not once per occurrence
        for line in set(p.text.split("\n")):
            line = line.strip()
            if line:
                line_counts[line] += 1

    min_pages = max(2, int(len(pages) * threshold))
    boilerplate = {line for line, count in line_counts.items() if count >= min_pages}

    cleaned = []
    for p in pages:
        kept_lines = [
            line for line in p.text.split("\n")
            if line.strip() not in boilerplate
        ]
        new_text = "\n".join(kept_lines).strip()
        if new_text:  # don't keep pages that were entirely boilerplate
            cleaned.append(PageText(source_file=p.source_file, page_number=p.page_number, text=new_text))

    if boilerplate:
        print(f"Stripped {len(boilerplate)} repeated boilerplate line(s):")
        for line in list(boilerplate)[:5]:
            print(f"  - {line[:60]}")

    return cleaned

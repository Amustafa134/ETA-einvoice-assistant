"""
Loads the ETA source PDFs and extracts text per page via OCR.

We switched from PyMuPDF's text-layer extraction to OCR after confirming
the source PDFs have broken font-to-Unicode mappings: the embedded fonts
produce consistent character-level corruption (e.g. "كتر" extracting as
"كير") that's baked into the PDF itself, not fixable by extraction
settings. OCR reads the rendered glyphs visually instead of trusting the
PDF's (broken) text layer, which sidesteps the problem entirely.

Requires:
    - Tesseract-OCR installed on the system (not just pip-installed),
      with the Arabic ("ara") language data included.
    - pytesseract + pillow (see requirements.txt)
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF -- used here only to render pages to images
import pytesseract
from PIL import Image

# Windows doesn't reliably add Tesseract to PATH, so point at it explicitly.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_DPI = 300  # higher = better accuracy, slower. 300 is a solid default for printed text.


@dataclass
class PageText:
    source_file: str
    page_number: int  # 1-indexed, matches what a human would cite
    text: str


def _render_page_to_image(page, dpi: int = OCR_DPI) -> Image.Image:
    """Render a single PDF page to a PIL Image for OCR."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def load_pdf(path: str | Path, dpi: int = OCR_DPI) -> list[PageText]:
    """OCR every page of a PDF, one PageText per page."""
    path = Path(path)
    doc = fitz.open(path)

    pages = []
    for i, page in enumerate(doc, start=1):
        img = _render_page_to_image(page, dpi=dpi)
        text = pytesseract.image_to_string(img, lang="ara")
        text = text.strip()
        if text:  # skip genuinely blank pages (e.g. section dividers)
            pages.append(PageText(source_file=path.name, page_number=i, text=text))

    doc.close()
    return pages


def load_all_pdfs(data_dir: str | Path, dpi: int = OCR_DPI) -> list[PageText]:
    """Load every PDF in the data directory via OCR."""
    data_dir = Path(data_dir)
    all_pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        all_pages.extend(load_pdf(pdf_path, dpi=dpi))
    return all_pages


if __name__ == "__main__":
    # Quick manual check: OCR just the first page of each PDF (fast) and
    # print it, so we can eyeball quality before running the full,
    # much slower, 58-page OCR pass.
    data_dir = Path("data")
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        img = _render_page_to_image(doc[0])
        text = pytesseract.image_to_string(img, lang="ara").strip()
        print(f"\n--- {pdf_path.name} / page 1 (OCR) ---")
        print(text[:400])
        doc.close()
        
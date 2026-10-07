"""
Splits cleaned page text into retrievable chunks.

Arabic needs slightly different split points than English: question marks
are "؟" not "?", commas are "،" not ",". We list these explicitly so the
splitter actually recognizes natural break points in the source text
instead of just falling back to blind character cuts.

Each page is chunked independently (rather than merging pages together
first) so every chunk keeps an accurate, single page_number -- this
matters later when we want the system to cite which page an answer
came from.

We also filter out low-quality chunks produced by OCR on diagram/flowchart
pages. Those pages have scattered icon labels and arrows rather than real
sentences, and OCR reliably turns them into noise (lots of isolated single
characters, no coherent words). Rather than try to "fix" OCR on diagrams
-- not realistically possible -- we detect and drop these chunks so they
don't pollute retrieval quality.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_loader import PageText

ARABIC_AWARE_SEPARATORS = [
    "\n\n", "\n", "؟", "!", "۔", ".", "،", "؛", " ", "",
]

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# A chunk is considered noise if more than this fraction of its
# whitespace-separated tokens are a single character long -- a strong
# signal of scattered OCR fragments rather than real words/sentences.
NOISE_TOKEN_RATIO = 0.35
MIN_TOKENS_TO_JUDGE = 6  # don't judge very short chunks, too easy to false-positive


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    page_number: int


def _is_noisy(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < MIN_TOKENS_TO_JUDGE:
        return False
    single_char_count = sum(1 for t in tokens if len(t) == 1)
    return (single_char_count / len(tokens)) > NOISE_TOKEN_RATIO


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list[Chunk], int]:
    """
    Split every page's text into overlapping, Arabic-aware chunks.
    Returns (chunks, number_of_noisy_chunks_dropped).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=ARABIC_AWARE_SEPARATORS,
    )

    chunks: list[Chunk] = []
    dropped = 0
    for page in pages:
        pieces = splitter.split_text(page.text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            if _is_noisy(piece):
                dropped += 1
                continue
            chunk_id = f"{page.source_file}::p{page.page_number}::c{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_file=page.source_file,
                    page_number=page.page_number,
                )
            )

    return chunks, dropped


if __name__ == "__main__":
    from src.ingestion.pdf_loader import load_all_pdfs
    from src.ingestion.cleaner import strip_repeated_boilerplate

    pages = load_all_pdfs("data")
    pages = strip_repeated_boilerplate(pages)
    chunks, dropped = chunk_pages(pages)

    print(f"{len(pages)} pages -> {len(chunks)} chunks ({dropped} noisy chunks dropped)")
    lengths = [len(c.text) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")

    print("\n--- Sample chunk (middle of document) ---")
    sample = chunks[len(chunks) // 2]
    print(f"id: {sample.id}")
    print(sample.text)

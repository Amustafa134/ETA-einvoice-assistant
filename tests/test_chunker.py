"""
Tests for the chunking logic -- uses synthetic PageText objects, not real
PDFs, so these run instantly with no external dependencies (no OCR, no
network, no API calls).
"""

from src.ingestion.pdf_loader import PageText
from src.ingestion.chunker import chunk_pages, _is_noisy


def test_chunk_pages_produces_chunks():
    pages = [
        PageText(
            source_file="test.pdf",
            page_number=1,
            text="هل يمكن اصدار اكثر من اشعار مرتبط بفاتورة واحدة سابقة؟ "
                 "نعم، يجب مراعاة ألا يتخطى مجموع مبالغ تلك الإشعارات مبلغ الفاتورة.",
        )
    ]
    chunks, dropped = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0
    assert dropped == 0
    assert all(c.source_file == "test.pdf" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_pages_respects_source_metadata_across_multiple_pages():
    pages = [
        PageText(source_file="doc_a.pdf", page_number=1, text="نص تجريبي على الصفحة الأولى من المستند الأول"),
        PageText(source_file="doc_b.pdf", page_number=5, text="نص تجريبي على الصفحة الخامسة من المستند الثاني"),
    ]
    chunks, _ = chunk_pages(pages, chunk_size=100)

    sources = {(c.source_file, c.page_number) for c in chunks}
    assert ("doc_a.pdf", 1) in sources
    assert ("doc_b.pdf", 5) in sources


def test_empty_page_produces_no_chunks():
    pages = [PageText(source_file="empty.pdf", page_number=1, text="")]
    chunks, dropped = chunk_pages(pages)
    assert len(chunks) == 0


def test_is_noisy_flags_scattered_single_characters():
    noisy_text = "ا ض ا ا ل ا ا ض ا ا ل ا ا ض ا"
    assert _is_noisy(noisy_text) is True


def test_is_noisy_does_not_flag_real_sentences():
    real_text = "هل هناك حد للاشعار الدائن المرتبط بفاتورة سبق اصدارها؟ نعم يوجد حد."
    assert _is_noisy(real_text) is False


def test_is_noisy_does_not_flag_short_chunks():
    short_text = "نعم ولا"
    assert _is_noisy(short_text) is False
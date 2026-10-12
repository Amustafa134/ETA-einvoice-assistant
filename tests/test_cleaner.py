"""
Tests for repeated-boilerplate stripping.
"""

from src.ingestion.pdf_loader import PageText
from src.ingestion.cleaner import strip_repeated_boilerplate


def test_strips_line_repeated_across_many_pages():
    repeated_line = "منظومة الفاتورة الإلكترونية"
    pages = [
        PageText(source_file="t.pdf", page_number=i, text=f"{repeated_line}\nمحتوى فريد للصفحة {i}")
        for i in range(1, 6)
    ]

    cleaned = strip_repeated_boilerplate(pages, threshold=0.3)

    for page in cleaned:
        assert repeated_line not in page.text
        assert "محتوى فريد" in page.text


def test_does_not_strip_content_that_only_appears_once():
    pages = [
        PageText(source_file="t.pdf", page_number=1, text="محتوى فريد تماما لهذه الصفحة فقط"),
        PageText(source_file="t.pdf", page_number=2, text="محتوى آخر مختلف تماما لهذه الصفحة"),
    ]

    cleaned = strip_repeated_boilerplate(pages, threshold=0.3)

    assert "محتوى فريد تماما لهذه الصفحة فقط" in cleaned[0].text
    assert "محتوى آخر مختلف تماما لهذه الصفحة" in cleaned[1].text


def test_page_that_is_entirely_boilerplate_is_dropped():
    repeated_line = "تكرار"
    pages = [
        PageText(source_file="t.pdf", page_number=i, text=repeated_line)
        for i in range(1, 6)
    ]

    cleaned = strip_repeated_boilerplate(pages, threshold=0.3)

    assert len(cleaned) == 0
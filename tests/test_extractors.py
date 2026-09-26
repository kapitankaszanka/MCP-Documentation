"""Tests for the generic file -> text pages extractor registry."""

from pathlib import Path

import pymupdf
import pytest

from mcp_documentation.ingest.extractors import (
    Page,
    _clean,
    supported_suffixes,
    to_pages,
)


def test_supported_suffixes_includes_registered_formats():
    assert {".pdf", ".txt", ".md"} <= supported_suffixes()


def test_to_pages_plain_txt(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello from a plain text file\n", encoding="utf-8")

    assert to_pages(path) == [Page(None, "hello from a plain text file")]


def test_to_pages_markdown(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    assert to_pages(path) == [Page(None, "# Title\n\nSome body text.")]


def test_to_pages_pdf_numbers_pages_and_drops_repeated_footer(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    bodies = ["alpha routing notes", "bravo switching notes", "charlie firewall notes"]
    doc = pymupdf.open()
    for number, body in enumerate(bodies, start=1):
        page = doc.new_page()
        page.insert_text((72, 400), body)
        page.insert_text((72, 800), f"(c) 2025 Example Corp. Page {number} of 3")
    doc.save(path)
    doc.close()

    pages = to_pages(path)

    assert [page.number for page in pages] == [1, 2, 3]
    for body, page in zip(bodies, pages, strict=True):
        assert body in page.text
        assert "Example Corp" not in page.text


def test_to_pages_unsupported_suffix_raises(tmp_path: Path):
    path = tmp_path / "notes.docx"
    path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.docx"):
        to_pages(path)


def test_to_pages_case_insensitive_suffix(tmp_path: Path):
    path = tmp_path / "SHOUTING.TXT"
    path.write_text("still plain text", encoding="utf-8")

    assert to_pages(path) == [Page(None, "still plain text")]


class TestClean:
    def test_strips_markup_leftovers_and_blank_runs(self):
        raw = (
            "<u>Underlined</u> text   \n\n\n\n"
            "<!-- Start of picture text -->\n-<br><!-- End of picture text -->\n"
            "•\nEnd."
        )
        assert _clean([Page(1, raw)]) == [Page(1, "Underlined text\n\nEnd.")]

    def test_drops_lines_repeated_on_most_pages(self):
        bodies = ["Alpha.", "Bravo.", "Charlie."]
        pages = [
            Page(n, f"{body}\nCopyright Page {n} of 3")
            for n, body in enumerate(bodies, start=1)
        ]
        assert [page.text for page in _clean(pages)] == bodies

    def test_keeps_repeated_headings_and_table_rows(self):
        texts = [
            f"## Caution\n| a | b |\n|---|---|\n{body}"
            for body in ("Alpha.", "Bravo.", "Charlie.")
        ]
        pages = [Page(n, text) for n, text in enumerate(texts, start=1)]
        assert [page.text for page in _clean(pages)] == texts

    def test_needs_several_pages_to_call_a_line_repeated(self):
        pages = [Page(n, f"Same line\nBody {n}") for n in (1, 2)]
        assert _clean(pages) == pages

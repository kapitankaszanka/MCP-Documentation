"""Generic file -> cleaned text pages conversion, decoupled from any one format.

Adding support for a new source type is adding one function decorated with
:func:`register`, not touching the orchestrator that calls :func:`to_pages`.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm


@dataclass(frozen=True)
class Page:
    """One page of extracted text.

    Attributes:
        number: 1-based page number, or None for formats without pages.
        text: The page's text (Markdown for PDFs and ``.md`` files).
    """

    number: int | None
    text: str


_EXTRACTORS: dict[str, Callable[[Path], list[Page]]] = {}

# A line repeated (digits ignored) on more than this share of a document's
# pages is treated as a running header/footer, e.g. "Page 11 of 181".
_REPEATED_LINE_SHARE = 0.5
_REPEATED_LINE_MIN_PAGES = 3
# Headers/footers are short; longer repeated lines are more likely content.
_REPEATED_LINE_MAX_CHARS = 100

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINE_BREAK_TAG = re.compile(r"<br\s*/?>")
_UNDERLINE_TAG = re.compile(r"</?u>")
_LONE_BULLET = re.compile(r"^[ \t]*[-•●▪][ \t]*$", re.MULTILINE)
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


def register(
    *suffixes: str,
) -> Callable[[Callable[[Path], list[Page]]], Callable[[Path], list[Page]]]:
    """Register a conversion function for one or more file suffixes.

    Args:
        suffixes: Lowercase suffixes the decorated function handles, e.g.
            ``".pdf"``.
    """

    def decorator(fn: Callable[[Path], list[Page]]) -> Callable[[Path], list[Page]]:
        for suffix in suffixes:
            _EXTRACTORS[suffix] = fn
        return fn

    return decorator


def supported_suffixes() -> set[str]:
    """Return every file suffix with a registered extractor."""
    return set(_EXTRACTORS)


@register(".pdf")
def _pdf_to_pages(path: Path) -> list[Page]:
    """Extract a PDF as Markdown per page via pymupdf4llm (no OCR).

    pymupdf4llm's layout model detects headings, tables and lists, and drops
    page headers/footers.
    """
    with pymupdf.open(path) as document:
        chunks = pymupdf4llm.to_markdown(
            document,
            page_chunks=True,
            header=False,
            footer=False,
            use_ocr=False,
        )
    assert isinstance(chunks, list)  # page_chunks=True returns one dict per page
    return [
        Page(number=chunk["metadata"]["page_number"], text=chunk["text"])
        for chunk in chunks
    ]


@register(".txt", ".md")
def _plain_text_to_pages(path: Path) -> list[Page]:
    """Read an already-plain-text file as UTF-8, as a single page."""
    return [Page(number=None, text=path.read_text(encoding="utf-8"))]


def _repeated_lines(pages: list[Page]) -> set[str]:
    """Return digit-normalized lines found on most pages (headers/footers).

    Only short lines are candidates, and never Markdown headings or table
    rows — a repeated table separator (``|---|``) is structure, not a footer.
    """
    if len(pages) < _REPEATED_LINE_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update(
            {
                _DIGITS.sub("#", stripped)
                for line in page.text.splitlines()
                if (stripped := line.strip())
                and len(stripped) <= _REPEATED_LINE_MAX_CHARS
                and not stripped.startswith(("#", "|"))
            }
        )
    threshold = len(pages) * _REPEATED_LINE_SHARE
    return {line for line, count in counts.items() if count > threshold}


def _clean(pages: list[Page]) -> list[Page]:
    """Strip extraction noise: markup leftovers, headers/footers, blank runs."""
    repeated = _repeated_lines(pages)
    cleaned = []
    for page in pages:
        text = _HTML_COMMENT.sub("", page.text)
        text = _LINE_BREAK_TAG.sub("\n", text)
        text = _UNDERLINE_TAG.sub("", text)
        if repeated:
            text = "\n".join(
                line
                for line in text.splitlines()
                if _DIGITS.sub("#", line.strip()) not in repeated
            )
        text = _LONE_BULLET.sub("", text)
        text = _TRAILING_SPACE.sub("", text)
        text = _EXTRA_BLANK_LINES.sub("\n\n", text).strip()
        cleaned.append(Page(number=page.number, text=text))
    return cleaned


def to_pages(path: Path) -> list[Page]:
    """Convert a source file into cleaned text pages.

    Args:
        path: File to convert.

    Returns:
        The file's pages in order; a single page for formats without pages.

    Raises:
        ValueError: No extractor is registered for the file's suffix.
    """
    try:
        extractor = _EXTRACTORS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"no text extractor for {path.suffix!r}: {path}") from None
    return _clean(extractor(path))

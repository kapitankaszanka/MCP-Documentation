"""Walks a file or directory and keeps the database in sync with it.

Two entry points, both incremental (skip unchanged, re-extract changed,
remove vanished):

- :func:`ingest_file` — one file, category/subcategory given by the caller.
- :func:`ingest_directory` — a whole tree, category/subcategory derived from
  folder names relative to the given directory.

Documents are keyed by (category, subcategory, file name) — only the file
name is stored in ``source_path``; the path as given on the command line is
kept in ``metadata.file_path``.

Plus two explicit removal entry points:

- :func:`remove_file` — by file name (the file need not exist anymore),
  optionally narrowed to a category/subcategory.
- :func:`remove_directory` — every document in the categories that are
  top-level folders of a directory.

And :func:`format_document_list` for printing what's currently indexed.

The ``mcp-documentation-ingest`` command itself lives in :mod:`.cli`.
"""

import bisect
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import aiosqlite

from ..config import get_config
from ..documents import (
    Chunk,
    add_document,
    delete_document,
    find_document_by_hash,
    get_document,
    list_documents,
    touch_document,
    update_document,
)
from . import extractors
from .extractors import Page

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024

_PAGE_SEPARATOR = "\n\n"
_HEADING = re.compile(r"#{1,6}[ \t]+(.+?)[ \t]*$")
_SENTENCE_END = re.compile(r"[.!?:;](?=\s)|\n")
_WHITESPACE = re.compile(r"\s+")

IngestOutcome = Literal["added", "updated", "skipped"]


def _headings(text: str) -> list[tuple[int, str]]:
    """Return (offset, title) of every Markdown heading outside code fences.

    Titles ending in ``:`` are skipped — PDF extraction turns bold callout
    labels ("Note:", "Caution:") into headings, and they aren't sections.
    """
    headings = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and (match := _HEADING.match(line)):
            title = match.group(1).replace("**", "").replace("__", "").strip(" #*_")
            if title and not title.endswith(":"):
                headings.append((offset, title))
        offset += len(line)
    return headings


def _cut(text: str, start: int, chunk_size: int, heading_offsets: list[int]) -> int:
    """Pick where the chunk starting at ``start`` ends.

    Once the chunk is at least half full, the first heading wins; otherwise
    the last blank line, then sentence end, then whitespace before
    ``chunk_size``; a hard cut only when there's none.
    """
    hard_end = start + chunk_size
    if hard_end >= len(text):
        return len(text)
    min_end = start + chunk_size // 2

    index = bisect.bisect_left(heading_offsets, min_end)
    if index < len(heading_offsets) and heading_offsets[index] <= hard_end:
        return heading_offsets[index]

    paragraph = text.rfind("\n\n", min_end, hard_end)
    if paragraph != -1:
        return paragraph + 2
    for boundary in (_SENTENCE_END, _WHITESPACE):
        if matches := list(boundary.finditer(text, min_end, hard_end)):
            return matches[-1].end()
    return hard_end


def _overlap_start(text: str, start: int, end: int, overlap: int) -> int:
    """Start the next chunk up to ``overlap`` chars back, on a word boundary."""
    candidate = max(end - overlap, start + 1)
    match = _WHITESPACE.search(text, candidate, end)
    return match.end() if match else end


def chunk_pages(
    pages: list[Page], chunk_size: int = 2048, overlap: int = 254
) -> list[Chunk]:
    """Split a document's pages into chunks cut at natural boundaries.

    Args:
        pages: The document's pages, in order.
        chunk_size: Maximum characters per chunk.
        overlap: Up to this many characters from the end of a chunk are
            repeated at the start of the next one, so a match sitting on a
            boundary still has context in at least one chunk. Not applied
            when a chunk ends at a heading — a new section needs none.

    Returns:
        Chunks in document order, stripped of surrounding whitespace, each
        with its page range and the nearest heading before it. Empty for
        blank input.
    """
    texts = [page for page in pages if page.text.strip()]
    text = _PAGE_SEPARATOR.join(page.text for page in texts)
    page_offsets = []
    offset = 0
    for page in texts:
        page_offsets.append(offset)
        offset += len(page.text) + len(_PAGE_SEPARATOR)

    headings = _headings(text)
    heading_offsets = [heading_offset for heading_offset, _ in headings]

    def page_at(position: int) -> int | None:
        return texts[bisect.bisect_right(page_offsets, position) - 1].number

    def section_at(position: int) -> str | None:
        index = bisect.bisect_right(heading_offsets, position) - 1
        return headings[index][1] if index >= 0 else None

    chunks = []
    start = 0
    while start < len(text):
        end = _cut(text, start, chunk_size, heading_offsets)
        raw = text[start:end]
        if body := raw.strip():
            first = start + len(raw) - len(raw.lstrip())
            last = start + len(raw.rstrip()) - 1
            chunks.append(
                Chunk(
                    text=body,
                    page_start=page_at(first),
                    page_end=page_at(last),
                    section=section_at(first),
                )
            )
        if end >= len(text):
            break
        start = (
            end if end in heading_offsets else _overlap_start(text, start, end, overlap)
        )
    return chunks


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Args:
        path: File to hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_category(directory: Path, path: Path) -> tuple[str, str | None]:
    """Derive (category, subcategory) from a file's folder path.

    The first path segment under ``directory`` becomes ``category``; any
    remaining segments are joined with ``"/"`` into ``subcategory`` (no depth
    limit — arbitrarily deep trees are fine).

    Args:
        directory: Root the file was discovered under.
        path: File to classify.

    Returns:
        Tuple of (category, subcategory).

    Raises:
        ValueError: ``path`` sits directly in ``directory``, with no category
            folder at all.
    """
    parts = path.relative_to(directory).parts[:-1]
    if not parts:
        raise ValueError(f"no category folder under {directory}: {path}")
    category = parts[0]
    subcategory = "/".join(parts[1:]) if len(parts) > 1 else None
    return category, subcategory


async def ingest_file(
    conn: aiosqlite.Connection,
    path: Path,
    category: str,
    subcategory: str | None = None,
) -> IngestOutcome:
    """Extract one file and insert/update/skip it in the database.

    Args:
        conn: Open db connection.
        path: File to ingest.
        category: Category to file the document under.
        subcategory: Slash-joined subcategory path, or None.

    Returns:
        "added", "updated", or "skipped".

    Raises:
        ValueError: Another indexed document already has identical content.
    """
    started = time.perf_counter()
    source_path = path.name
    stat = path.stat()
    existing = await get_document(conn, category, subcategory, source_path)

    if existing is not None and existing["file_mtime"] == stat.st_mtime:
        logger.info("Skipped %s (unchanged)", path)
        return "skipped"

    digest = file_hash(path)
    if existing is not None and existing["file_hash"] == digest:
        await touch_document(conn, existing["id"], stat.st_mtime)
        logger.info("Skipped %s (unchanged)", path)
        return "skipped"

    # existing (if any) has a different hash by now, so a match is another doc
    duplicate = await find_document_by_hash(conn, digest)
    if duplicate is not None:
        location = "/".join(
            part
            for part in (
                duplicate["category"],
                duplicate["subcategory"],
                duplicate["source_path"],
            )
            if part
        )
        raise ValueError(f"identical content already indexed as {location}")

    logger.info("Indexing %s ...", path)
    pages = extractors.to_pages(path)
    config = get_config()
    chunks = chunk_pages(
        pages, chunk_size=config.chunk_size, overlap=config.chunk_overlap
    )
    metadata = {
        "filename": path.stem,
        "file_path": str(path),
        "chunks_num": len(chunks),
        "char_num": sum(len(page.text) for page in pages),
    }

    if existing is None:
        await add_document(
            conn,
            category,
            subcategory,
            source_path,
            stat.st_mtime,
            digest,
            metadata,
            chunks,
        )
        outcome: IngestOutcome = "added"
    else:
        await update_document(
            conn,
            existing["id"],
            category,
            subcategory,
            stat.st_mtime,
            digest,
            metadata,
            chunks,
        )
        outcome = "updated"

    logger.info(
        "%s %s: %d page(s), %d chunk(s), %d chars in %.2fs",
        outcome.capitalize(),
        path,
        len(pages),
        len(chunks),
        metadata["char_num"],
        time.perf_counter() - started,
    )
    return outcome


def _is_hidden(directory: Path, path: Path) -> bool:
    """Check if file is hidden.

    Return True if ``path`` or any folder between it and ``directory`` is
    hidden (name starts with ``"."``). ``directory`` itself is not checked.
    """
    return any(part.startswith(".") for part in path.relative_to(directory).parts)


def _top_level_categories(directory: Path) -> set[str]:
    """Return the names of the non-hidden category folders."""
    return {
        p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
    }


@dataclass
class ReindexResult:
    """Summary counts for one :func:`ingest_directory` run."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


async def ingest_directory(
    conn: aiosqlite.Connection, directory: Path
) -> ReindexResult:
    """Scan a directory and bring the database in sync with it.

    Hidden files and folders (name starting with ``"."``) are ignored.

    Args:
        conn: Open db connection.
        directory: Root of the ``<category>/<subcategory>/.../*`` tree.

    Returns:
        Summary counts of the run.

    Raises:
        FileNotFoundError: ``directory`` doesn't exist or isn't a directory.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")

    started = time.perf_counter()
    categories = _top_level_categories(directory)
    suffixes = extractors.supported_suffixes()
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not _is_hidden(directory, p)
    )

    logger.info("Found %d supported file(s) in %s", len(files), directory)
    result = ReindexResult()

    def fail(path: Path, exc: Exception) -> None:
        result.failed += 1
        result.errors.append(f"{path}: {exc}")
        logger.warning("Failed to ingest %s: %s", path, exc)

    located: list[tuple[Path, str, str | None]] = []
    for path in files:
        result.scanned += 1
        try:
            located.append((path, *_infer_category(directory, path)))
        except ValueError as exc:
            fail(path, exc)

    # Remove vanished documents first, so a file moved to another folder
    # isn't rejected as a duplicate of its own stale entry.
    expected = {
        (category, subcategory, path.name) for path, category, subcategory in located
    }
    for row in await list_documents(conn):
        if row["category"] not in categories:
            continue
        triple = (row["category"], row["subcategory"], row["source_path"])
        if triple not in expected:
            await delete_document(conn, row["id"])
            logger.info(
                "Removed %s (no longer on disk)", "/".join(p for p in triple if p)
            )
            result.removed += 1

    for index, (path, category, subcategory) in enumerate(located, start=1):
        logger.info("[%d/%d] %s", index, len(located), path)
        try:
            outcome = await ingest_file(conn, path, category, subcategory)
        except Exception as exc:  # one bad file must not abort the whole run
            fail(path, exc)
            continue

        if outcome == "added":
            result.added += 1
        elif outcome == "updated":
            result.updated += 1
        else:
            result.skipped += 1

    logger.info(
        "Ingest complete in %.2fs: scanned=%d added=%d updated=%d removed=%d "
        "skipped=%d failed=%d",
        time.perf_counter() - started,
        result.scanned,
        result.added,
        result.updated,
        result.removed,
        result.skipped,
        result.failed,
    )
    for error in result.errors:
        logger.warning("Ingest error: %s", error)
    return result


async def remove_file(
    conn: aiosqlite.Connection,
    path: Path,
    category: str | None = None,
    subcategory: str | None = None,
) -> int:
    """Remove a file's documents from the database, matched by file name.

    Args:
        conn: Open db connection.
        path: Source file; only its name is used, so it need not exist
            anymore.
        category: If given, only remove the document filed under this
            category (and ``subcategory``); otherwise remove every document
            with this file name, whatever its category.
        subcategory: Slash-joined subcategory path, or None. Only used
            together with ``category``.

    Returns:
        Number of documents removed.
    """
    source_path = path.name
    removed = 0
    for row in await list_documents(conn):
        if row["source_path"] != source_path:
            continue
        if category is not None and (row["category"], row["subcategory"]) != (
            category,
            subcategory,
        ):
            continue
        await delete_document(conn, row["id"])
        removed += 1
    return removed


async def remove_directory(conn: aiosqlite.Connection, directory: Path) -> int:
    """Remove every document in the categories found under a directory.

    Args:
        conn: Open db connection.
        directory: Root of a ``<category>/...`` tree; each of its top-level
            folders names a category whose documents are all removed.

    Returns:
        Number of documents removed.

    Raises:
        FileNotFoundError: ``directory`` doesn't exist or isn't a directory.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")

    categories = _top_level_categories(directory)
    removed = 0
    for row in await list_documents(conn):
        if row["category"] in categories:
            await delete_document(conn, row["id"])
            removed += 1
    return removed


def format_document_list(rows: list[aiosqlite.Row]) -> str:
    """Render indexed documents as an aligned, sorted text table.

    Args:
        rows: Rows from :func:`list_documents`.

    Returns:
        One line per document (category, subcategory or ``-``, file name),
        sorted by those three columns, with a trailing newline. Empty string
        for no rows.
    """
    table = sorted(
        (row["category"], row["subcategory"] or "-", row["source_path"]) for row in rows
    )
    if not table:
        return ""
    header = ("CATEGORY", "SUBCATEGORY", "FILE")
    widths = [max(len(line[i]) for line in (header, *table)) for i in range(2)]
    return "".join(
        f"{category:<{widths[0]}}  {subcategory:<{widths[1]}}  {source_path}\n"
        for category, subcategory, source_path in (header, *table)
    )

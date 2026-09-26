"""Tests for the ingest orchestrator: chunking, hashing, category inference,
single-file ingestion, and directory-tree ingestion.
"""

import json
import logging
import os
import re
from pathlib import Path

import aiosqlite
import pymupdf
import pytest

from mcp_documentation import __version__
from mcp_documentation.db import session
from mcp_documentation.documents import Chunk, get_document, list_documents
from mcp_documentation.ingest.cli import _parse_args, _run
from mcp_documentation.ingest.extractors import Page
from mcp_documentation.ingest.orchestrator import (
    _infer_category,
    chunk_pages,
    file_hash,
    format_document_list,
    ingest_directory,
    ingest_file,
    remove_directory,
    remove_file,
)


def _texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


class TestChunkPages:
    def test_empty_and_blank_input(self):
        assert chunk_pages([]) == []
        assert chunk_pages([Page(None, "   \n\t  ")]) == []

    def test_short_text_is_a_single_chunk(self):
        assert chunk_pages([Page(None, "hello world")]) == [Chunk(text="hello world")]

    def test_hard_cut_with_overlap_when_no_boundary(self):
        text = "0123456789" * 3  # 30 chars, no whitespace to cut at
        chunks = chunk_pages([Page(None, text)], chunk_size=10, overlap=3)
        # no whitespace for the overlap to start at, so no overlap either
        assert _texts(chunks) == ["0123456789", "0123456789", "0123456789"]

    def test_never_cuts_mid_word(self):
        words = [f"word{i}" for i in range(200)]
        chunks = chunk_pages([Page(None, " ".join(words))], chunk_size=100, overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            assert all(part in words for part in chunk.text.split())
            assert len(chunk.text) <= 100

    def test_overlap_repeats_whole_words(self):
        text = " ".join(f"w{i:03}" for i in range(100))
        chunks = chunk_pages([Page(None, text)], chunk_size=100, overlap=20)

        first, second = chunks[0].text.split(), chunks[1].text.split()
        assert first[-1] in second
        assert second[0] in first

    def test_prefers_paragraph_then_sentence_boundaries(self):
        text = "First sentence here. Second one follows.\n\nNew paragraph text."
        chunks = chunk_pages([Page(None, text)], chunk_size=50, overlap=0)
        assert chunks[0].text == "First sentence here. Second one follows."

        text = "First sentence here. Second one follows on and on."
        chunks = chunk_pages([Page(None, text)], chunk_size=30, overlap=0)
        assert chunks[0].text == "First sentence here."

    def test_splits_at_headings_and_tracks_section(self):
        body = "Some text about it. " * 5  # 100 chars
        text = f"# Intro\n\n{body}\n\n## **Routing**\n\n{body}"
        chunks = chunk_pages([Page(None, text)], chunk_size=150, overlap=30)

        assert chunks[0].text.startswith("# Intro")
        assert chunks[0].section == "Intro"
        # the chunk starts exactly at the heading, with no overlap before it
        assert chunks[1].text.startswith("## **Routing**")
        assert chunks[1].section == "Routing"

    def test_section_carries_forward(self):
        body = " ".join(["text"] * 100)
        chunks = chunk_pages(
            [Page(None, f"# Only heading\n\n{body}")], chunk_size=100, overlap=10
        )

        assert len(chunks) > 2
        assert {chunk.section for chunk in chunks} == {"Only heading"}

    def test_callout_labels_are_not_sections(self):
        body = "Some text about it. " * 5  # 100 chars
        text = f"# Routing\n\n{body}\n\n## **Note:**\n\n{body}"
        chunks = chunk_pages([Page(None, text)], chunk_size=150, overlap=0)

        assert len(chunks) == 2
        assert {chunk.section for chunk in chunks} == {"Routing"}

    def test_headings_inside_code_fences_are_ignored(self):
        text = "Intro text.\n\n```\n# show running-config\n```\n\nMore."
        chunks = chunk_pages([Page(None, text)])
        assert chunks == [Chunk(text=text)]

    def test_page_ranges(self):
        pages = [
            Page(1, "one " * 20),
            Page(2, "two " * 20),
            Page(3, "three " * 20),
        ]
        # page breaks are paragraph breaks; 200 chars can't stop at the first
        chunks = chunk_pages(pages, chunk_size=200, overlap=0)

        assert chunks[0].page_start == 1
        assert chunks[-1].page_end == 3
        assert any(chunk.page_start != chunk.page_end for chunk in chunks)
        for chunk in chunks:
            assert chunk.page_start is not None
            assert chunk.page_end is not None
            if "two" in chunk.text:
                assert chunk.page_start <= 2 <= chunk.page_end

    def test_pageless_documents_have_no_page(self):
        (chunk,) = chunk_pages([Page(None, "hello")])
        assert (chunk.page_start, chunk.page_end) == (None, None)


def test_file_hash_stable_and_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same content", encoding="utf-8")
    b.write_text("same content", encoding="utf-8")

    assert file_hash(a) == file_hash(b)

    b.write_text("different content", encoding="utf-8")
    assert file_hash(a) != file_hash(b)


class TestInferCategory:
    def test_category_only(self, tmp_path: Path):
        path = tmp_path / "devices" / "guide.pdf"
        assert _infer_category(tmp_path, path) == ("devices", None)

    def test_nested_subcategory_joins_with_slash(self, tmp_path: Path):
        path = tmp_path / "devices" / "switches" / "access" / "guide.pdf"
        assert _infer_category(tmp_path, path) == ("devices", "switches/access")

    def test_no_category_folder_raises(self, tmp_path: Path):
        path = tmp_path / "guide.pdf"
        with pytest.raises(ValueError, match="no category folder"):
            _infer_category(tmp_path, path)


def _make_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


async def _metadata(
    conn: aiosqlite.Connection, category: str, source_path: str
) -> dict:
    cursor = await conn.execute(
        "SELECT metadata FROM documents WHERE category = ? AND source_path = ?",
        (category, source_path),
    )
    row = await cursor.fetchone()
    assert row is not None
    return json.loads(row["metadata"])


class TestIngestFile:
    async def test_adds_new_document(self, conn: aiosqlite.Connection, tmp_path: Path):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "hello world")

        outcome = await ingest_file(conn, path, "devices", "switches")
        assert outcome == "added"

        row = await get_document(conn, "devices", "switches", path.name)
        assert row is not None

    async def test_stores_file_name_and_given_path(
        self,
        conn: aiosqlite.Connection,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _make_pdf(tmp_path / "docs" / "guide.pdf", "hello world")
        monkeypatch.chdir(tmp_path / "docs")

        await ingest_file(conn, Path("../docs/guide.pdf"), "devices")

        assert await get_document(conn, "devices", None, "guide.pdf")
        metadata = await _metadata(conn, "devices", "guide.pdf")
        assert metadata["file_path"] == "../docs/guide.pdf"

    async def test_unchanged_mtime_is_skipped(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "hello world")

        assert await ingest_file(conn, path, "devices") == "added"
        assert await ingest_file(conn, path, "devices") == "skipped"

    async def test_changed_mtime_same_hash_touches_without_reextracting(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "hello world")
        await ingest_file(conn, path, "devices")

        # bump mtime without changing content
        new_mtime = path.stat().st_mtime + 100
        os.utime(path, (new_mtime, new_mtime))

        outcome = await ingest_file(conn, path, "devices")
        assert outcome == "skipped"

        row = await get_document(conn, "devices", None, path.name)
        assert row is not None
        assert row["file_mtime"] == pytest.approx(new_mtime)

    async def test_changed_content_updates(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "version one")
        await ingest_file(conn, path, "devices")

        _make_pdf(path, "version two, much longer than before")
        outcome = await ingest_file(conn, path, "devices")
        assert outcome == "updated"

    async def test_same_name_different_category_is_independent(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "devices guide")
        assert await ingest_file(conn, path, "devices") == "added"

        _make_pdf(path, "topology guide")
        assert await ingest_file(conn, path, "topology") == "added"

        assert await get_document(conn, "devices", None, path.name)
        assert await get_document(conn, "topology", None, path.name)

    async def test_identical_content_is_rejected(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        first = tmp_path / "guide.pdf"
        _make_pdf(first, "hello world")
        await ingest_file(conn, first, "devices", "switches")

        copy = tmp_path / "copy" / "guide-copy.pdf"
        copy.parent.mkdir()
        copy.write_bytes(first.read_bytes())

        with pytest.raises(
            ValueError, match=r"already indexed as devices/switches/guide\.pdf"
        ):
            await ingest_file(conn, copy, "topology")
        assert await get_document(conn, "topology", None, copy.name) is None

    async def test_stores_page_and_section_per_chunk(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "notes.md"
        path.write_text("# Setup\n\nInstall it.\n", encoding="utf-8")
        await ingest_file(conn, path, "devices")

        rows = await conn.execute_fetchall(
            "SELECT text, page_start, section FROM document_chunks"
        )
        assert [tuple(row) for row in rows] == [
            ("# Setup\n\nInstall it.", None, "Setup")
        ]

    async def test_logs_outcome_with_stats_and_time(
        self,
        conn: aiosqlite.Connection,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        path = tmp_path / "notes.md"
        path.write_text("# Setup\n\nInstall it.\n", encoding="utf-8")
        caplog.set_level(logging.INFO)

        await ingest_file(conn, path, "devices")
        assert f"Indexing {path} ..." in caplog.messages
        assert re.fullmatch(
            rf"Added {re.escape(str(path))}: 1 page\(s\), 1 chunk\(s\), "
            r"\d+ chars in \d+\.\d{2}s",
            caplog.messages[-1],
        )

        await ingest_file(conn, path, "devices")
        assert caplog.messages[-1] == f"Skipped {path} (unchanged)"


class TestIngestDirectory:
    async def test_scans_and_categorizes(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        _make_pdf(tmp_path / "devices" / "a.pdf", "device a")
        _make_pdf(tmp_path / "devices" / "switches" / "b.pdf", "device b")

        result = await ingest_directory(conn, tmp_path)

        assert result.scanned == 2
        assert result.added == 2
        assert result.failed == 0

    async def test_rerun_is_fully_skipped(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        _make_pdf(tmp_path / "devices" / "a.pdf", "device a")
        await ingest_directory(conn, tmp_path)

        result = await ingest_directory(conn, tmp_path)
        assert result.added == 0
        assert result.skipped == 1

    async def test_removed_file_is_deleted_from_db(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "devices" / "a.pdf"
        _make_pdf(path, "device a")
        await ingest_directory(conn, tmp_path)

        path.unlink()
        result = await ingest_directory(conn, tmp_path)

        assert result.removed == 1
        assert await get_document(conn, "devices", None, path.name) is None

    async def test_logs_progress_removals_and_total_time(
        self,
        conn: aiosqlite.Connection,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        gone = tmp_path / "devices" / "gone.pdf"
        _make_pdf(gone, "device gone")
        await ingest_directory(conn, tmp_path)
        gone.unlink()
        _make_pdf(tmp_path / "devices" / "a.pdf", "device a")
        caplog.set_level(logging.INFO)

        await ingest_directory(conn, tmp_path)

        assert f"Found 1 supported file(s) in {tmp_path}" in caplog.messages
        assert "Removed devices/gone.pdf (no longer on disk)" in caplog.messages
        assert f"[1/1] {tmp_path / 'devices' / 'a.pdf'}" in caplog.messages
        assert re.match(r"Ingest complete in \d+\.\d{2}s: ", caplog.messages[-1])

    async def test_file_without_category_folder_fails_without_aborting(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        _make_pdf(tmp_path / "no_category.pdf", "orphan")
        _make_pdf(tmp_path / "devices" / "ok.pdf", "device ok")

        result = await ingest_directory(conn, tmp_path)

        assert result.scanned == 2
        assert result.failed == 1
        assert result.added == 1
        assert len(result.errors) == 1

    async def test_hidden_files_and_folders_are_ignored(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        _make_pdf(tmp_path / "devices" / "a.pdf", "device a")
        _make_pdf(tmp_path / "devices" / ".hidden.pdf", "hidden file")
        _make_pdf(tmp_path / "devices" / ".cache" / "b.pdf", "hidden subfolder")
        _make_pdf(tmp_path / ".git" / "c.pdf", "hidden category")

        result = await ingest_directory(conn, tmp_path)

        assert result.scanned == 1
        assert result.added == 1
        assert [row["source_path"] for row in await list_documents(conn)] == ["a.pdf"]

    async def test_moved_file_is_not_a_duplicate_of_its_old_entry(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        old = tmp_path / "devices" / "a.pdf"
        _make_pdf(old, "device a")
        await ingest_directory(conn, tmp_path)

        new = tmp_path / "topology" / "a.pdf"
        new.parent.mkdir()
        old.rename(new)
        result = await ingest_directory(conn, tmp_path)

        assert (result.removed, result.added, result.failed) == (1, 1, 0)
        assert await get_document(conn, "topology", None, "a.pdf")

    async def test_duplicate_in_tree_fails_only_the_copy(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        original = tmp_path / "devices" / "a.pdf"
        _make_pdf(original, "device a")
        copy = tmp_path / "devices" / "z-copy.pdf"
        copy.write_bytes(original.read_bytes())

        result = await ingest_directory(conn, tmp_path)

        assert (result.added, result.failed) == (1, 1)
        assert "z-copy.pdf" in result.errors[0]
        assert "already indexed as devices/a.pdf" in result.errors[0]

    async def test_nonexistent_directory_raises(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        with pytest.raises(FileNotFoundError):
            await ingest_directory(conn, tmp_path / "does_not_exist")

    async def test_cleanup_skips_categories_outside_directory(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        docs = tmp_path / "docs"
        _make_pdf(docs / "devices" / "a.pdf", "device a")
        outside = tmp_path / "outside.pdf"
        _make_pdf(outside, "topology doc")
        await ingest_file(conn, outside, "topology")

        result = await ingest_directory(conn, docs)

        assert result.removed == 0
        assert await get_document(conn, "topology", None, "outside.pdf")

    async def test_stores_given_path_in_metadata(
        self,
        conn: aiosqlite.Connection,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _make_pdf(tmp_path / "docs" / "devices" / "a.pdf", "device a")
        monkeypatch.chdir(tmp_path)

        await ingest_directory(conn, Path("docs"))

        metadata = await _metadata(conn, "devices", "a.pdf")
        assert metadata["filename"] == "a"
        assert metadata["file_path"] == "docs/devices/a.pdf"


class TestRemove:
    async def test_remove_file(self, conn: aiosqlite.Connection, tmp_path: Path):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "hello world")
        await ingest_file(conn, path, "devices")

        assert await remove_file(conn, path) == 1
        assert await get_document(conn, "devices", None, path.name) is None

    async def test_remove_file_category_filter(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "devices guide")
        source_path = path.name
        await ingest_file(conn, path, "devices", "switches")
        _make_pdf(path, "topology guide")
        await ingest_file(conn, path, "topology")

        assert await remove_file(conn, path, "devices", "switches") == 1
        assert await get_document(conn, "devices", "switches", source_path) is None
        assert await get_document(conn, "topology", None, source_path)

        assert await remove_file(conn, path) == 1
        assert await get_document(conn, "topology", None, source_path) is None

    async def test_remove_file_missing_from_disk(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        path = tmp_path / "guide.pdf"
        _make_pdf(path, "hello world")
        await ingest_file(conn, path, "devices")
        path.unlink()

        assert await remove_file(conn, path) == 1

    async def test_remove_unknown_file(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        assert await remove_file(conn, tmp_path / "nope.pdf") == 0

    async def test_remove_directory_only_touches_its_categories(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        docs = tmp_path / "docs"
        other = tmp_path / "other"
        _make_pdf(docs / "devices" / "a.pdf", "device a")
        _make_pdf(docs / "devices" / "switches" / "b.pdf", "device b")
        _make_pdf(other / "topology" / "c.pdf", "topology c")
        await ingest_directory(conn, docs)
        await ingest_directory(conn, other)

        assert await remove_directory(conn, docs) == 2
        assert await get_document(conn, "topology", None, "c.pdf")

    async def test_remove_directory_missing_raises(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ):
        with pytest.raises(FileNotFoundError):
            await remove_directory(conn, tmp_path / "does_not_exist")


class TestFormatDocumentList:
    def test_empty(self):
        assert format_document_list([]) == ""

    async def test_sorted_and_aligned(self, conn: aiosqlite.Connection, tmp_path: Path):
        _make_pdf(tmp_path / "topology" / "b.pdf", "b")
        _make_pdf(tmp_path / "devices" / "switches" / "a.pdf", "a")
        await ingest_directory(conn, tmp_path)

        lines = format_document_list(await list_documents(conn)).splitlines()

        assert lines[0].split() == ["CATEGORY", "SUBCATEGORY", "FILE"]
        assert lines[1].split() == ["devices", "switches", "a.pdf"]
        assert lines[2].split() == ["topology", "-", "b.pdf"]
        # file names start in the same column
        assert lines[1].index("a.pdf") == lines[2].index("b.pdf")


class TestRunMultipleFiles:
    async def test_bad_file_does_not_stop_the_rest(self, configure, tmp_path: Path):
        configure(db_path=str(tmp_path / "test.db"))
        good = tmp_path / "good.pdf"
        _make_pdf(good, "good")
        missing = tmp_path / "missing.pdf"
        args = _parse_args(["-f", str(missing), str(good), "--category", "devices"])

        with pytest.raises(RuntimeError, match="1 of 2"):
            await _run(args)

        async with session() as conn:
            assert await get_document(conn, "devices", None, "good.pdf")

    async def test_remove_multiple(self, configure, tmp_path: Path):
        configure(db_path=str(tmp_path / "test.db"))
        a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
        _make_pdf(a, "a")
        _make_pdf(b, "b")
        await _run(_parse_args(["-f", str(a), str(b), "--category", "devices"]))

        await _run(_parse_args(["-f", str(a), str(b), "--remove"]))

        async with session() as conn:
            assert await list_documents(conn) == []


class TestParseArgs:
    def test_file_requires_category(self):
        with pytest.raises(SystemExit):
            _parse_args(["-f", "doc.pdf"])

    def test_directory_rejects_category(self):
        with pytest.raises(SystemExit):
            _parse_args(["-d", "docs/", "--category", "devices"])

    def test_mutually_exclusive_file_and_directory(self):
        with pytest.raises(SystemExit):
            _parse_args(["-f", "doc.pdf", "-d", "docs/", "--category", "devices"])

    def test_requires_one_target(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    @pytest.mark.parametrize("flag", ["-v", "--version"])
    def test_version(self, flag, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse_args([flag])
        assert exc.value.code == 0
        assert capsys.readouterr().out == (f"mcp-documentation-ingest {__version__}\n")

    def test_valid_file_mode(self):
        args = _parse_args(["-f", "doc.pdf", "--category", "devices"])
        assert args.file == [Path("doc.pdf")]
        assert args.category == "devices"

    def test_multiple_files(self):
        args = _parse_args(["-f", "a.pdf", "b.md", "--category", "devices"])
        assert args.file == [Path("a.pdf"), Path("b.md")]

    def test_valid_directory_mode(self):
        args = _parse_args(["-d", "docs/"])
        assert args.directory == Path("docs/")
        assert not args.remove

    def test_remove_file_without_category(self):
        args = _parse_args(["-f", "doc.pdf", "--remove"])
        assert args.remove
        assert args.category is None

    def test_remove_subcategory_requires_category(self):
        with pytest.raises(SystemExit):
            _parse_args(["-f", "doc.pdf", "--remove", "--subcategory", "x"])

    def test_list(self):
        args = _parse_args(["--list"])
        assert args.list
        assert args.file is None
        assert args.directory is None

    def test_list_rejects_other_options(self):
        with pytest.raises(SystemExit):
            _parse_args(["--list", "--remove"])
        with pytest.raises(SystemExit):
            _parse_args(["--list", "--category", "devices"])
        with pytest.raises(SystemExit):
            _parse_args(["--list", "-f", "doc.pdf"])

    def test_remove_directory_rejects_category(self):
        with pytest.raises(SystemExit):
            _parse_args(["-d", "docs/", "--remove", "--category", "devices"])

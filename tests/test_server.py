"""Tests for the MCP tool functions in server.py.

These call the tool functions directly (FastMCP's @mcp.tool decorator
leaves the underlying function callable) against a db connection wired in
by hand, bypassing the FastMCP lifespan entirely.
"""

import aiosqlite
import pytest

from mcp_documentation import server
from mcp_documentation.documents import Chunk, add_document


@pytest.fixture(autouse=True)
def _wire_up_conn(conn: aiosqlite.Connection, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_db_conn", conn)


async def _seed(conn, **kwargs):
    defaults = {
        "category": "devices",
        "subcategory": None,
        "source_path": "/a.pdf",
        "file_mtime": 1.0,
        "file_hash": "h",
        "metadata": {"filename": "a", "chunks_num": 2, "char_num": 20},
        "chunks": [
            Chunk("first chunk about switches"),
            Chunk(
                "second chunk about routers",
                page_start=2,
                page_end=2,
                section="Routers",
            ),
        ],
    }
    defaults.update(kwargs)
    return await add_document(
        conn,
        defaults["category"],
        defaults["subcategory"],
        defaults["source_path"],
        defaults["file_mtime"],
        defaults["file_hash"],
        defaults["metadata"],
        defaults["chunks"],
    )


def test_get_conn_raises_without_lifespan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "_db_conn", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        server.get_conn()


async def test_search_doc_returns_matching_chunks(conn, configure):
    configure()
    await _seed(conn)

    response = await server.search_doc("switches")
    assert len(response.results) == 1
    result = response.results[0]
    assert result.file_name == "a"
    assert result.category == "devices"
    assert result.snippet == "first chunk about **switches**"
    assert response.next_offset is None


async def test_search_doc_returns_page_and_section(conn, configure):
    configure()
    await _seed(conn)

    (result,) = (await server.search_doc("routers")).results
    assert (result.chunk_number, result.page_start, result.section) == (
        1,
        2,
        "Routers",
    )


async def test_search_doc_filters_by_category(conn, configure):
    configure()
    await _seed(conn, category="devices", file_hash="h1")
    await _seed(conn, category="topology", file_hash="h2")

    response = await server.search_doc("switches", category="topology")
    assert [r.category for r in response.results] == ["topology"]


async def test_search_doc_invalid_query_hints_at_quoting(conn, configure):
    configure()
    await _seed(conn)

    with pytest.raises(ValueError, match=r'"MP-BGP"'):
        await server.search_doc("MP-BGP")


async def test_search_doc_pages_with_next_offset(conn, configure):
    configure(max_search_results=1)
    await add_document(
        conn,
        "devices",
        None,
        "/a.pdf",
        1.0,
        "h1",
        {"filename": "a"},
        [Chunk("shared term")],
    )
    await add_document(
        conn,
        "devices",
        None,
        "/b.pdf",
        1.0,
        "h2",
        {"filename": "b"},
        [Chunk("shared term")],
    )

    first = await server.search_doc("shared")
    assert first.next_offset == 1
    second = await server.search_doc("shared", offset=first.next_offset)
    assert second.next_offset is None
    assert {first.results[0].file_name, second.results[0].file_name} == {"a", "b"}


async def test_search_doc_respects_max_search_results(conn, configure):
    configure(max_search_results=1)
    await add_document(
        conn,
        "devices",
        None,
        "/a.pdf",
        1.0,
        "h1",
        {"filename": "a"},
        [Chunk("shared term")],
    )
    await add_document(
        conn,
        "devices",
        None,
        "/b.pdf",
        1.0,
        "h2",
        {"filename": "b"},
        [Chunk("shared term")],
    )

    response = await server.search_doc("shared")
    assert len(response.results) == 1


async def test_list_doc_builds_tree_with_chunk_counts(conn, configure):
    configure()
    await _seed(
        conn, subcategory="switches", metadata={"filename": "a", "chunks_num": 2}
    )

    result = await server.list_doc()
    assert result.model_dump() == {
        "categories": [
            {
                "name": "devices",
                "files": [],
                "subcategories": [
                    {
                        "name": "switches",
                        "files": [{"name": "a", "chunks_num": 2}],
                        "subcategories": [],
                    }
                ],
            }
        ]
    }


async def test_list_categories_omits_files(conn, configure):
    configure()
    await _seed(conn)

    result = await server.list_categories()
    assert result.model_dump() == {
        "categories": [{"name": "devices", "files": [], "subcategories": []}]
    }


async def test_read_doc_returns_requested_chunks(conn, configure):
    configure()
    await _seed(conn)

    result = await server.read_doc("a", "devices", None, [1])
    assert result.model_dump() == {
        "chunks": [
            {
                "chunk_number": 1,
                "page_start": 2,
                "page_end": 2,
                "section": "Routers",
                "text": "second chunk about routers",
            }
        ]
    }


async def test_read_doc_rejects_too_many_chunk_numbers(conn, configure):
    configure(max_chunks_per_read=2)
    await _seed(conn)

    with pytest.raises(ValueError, match="at most 2"):
        await server.read_doc("a", "devices", None, [0, 1, 2])


async def test_read_doc_unknown_document_raises(conn, configure):
    configure()

    with pytest.raises(ValueError, match="no document"):
        await server.read_doc("missing", "devices", None, [0])


async def test_read_doc_ambiguous_filename_raises(conn, configure):
    configure()
    await _seed(conn, source_path="/a.pdf")
    await _seed(conn, source_path="/a.md")

    with pytest.raises(ValueError, match="ambiguous"):
        await server.read_doc("a", "devices", None, [0])

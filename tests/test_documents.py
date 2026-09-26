"""Tests for the shared documents.py DB layer (schema + CRUD + queries)."""

import aiosqlite
import pytest

from mcp_documentation.documents import (
    Chunk,
    add_document,
    delete_document,
    find_document_by_hash,
    get_category_tree,
    get_chunks,
    get_document,
    get_document_by_name,
    list_documents,
    search_chunks,
    touch_document,
    update_document,
)


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(text) for text in texts]


async def _add(
    conn, *, category="devices", subcategory=None, source_path="/a.pdf", filename="a"
):
    return await add_document(
        conn,
        category,
        subcategory,
        source_path,
        1000.0,
        "hash1",
        {"filename": filename, "chunks_num": 2, "char_num": 10},
        _chunks("chunk one text", "chunk two text"),
    )


async def test_add_document_round_trip(conn: aiosqlite.Connection):
    doc_id = await _add(conn)

    row = await get_document(conn, "devices", None, "/a.pdf")
    assert row is not None
    assert row["id"] == doc_id
    assert row["file_mtime"] == 1000.0
    assert row["file_hash"] == "hash1"

    chunks = await conn.execute_fetchall(
        "SELECT chunk_number, text FROM document_chunks ORDER BY chunk_number"
    )
    assert [tuple(r) for r in chunks] == [
        (0, "chunk one text"),
        (1, "chunk two text"),
    ]
    fts_rows = list(await conn.execute_fetchall("SELECT count(*) FROM chunks_fts"))
    assert fts_rows[0][0] == 2


async def test_get_document_subcategory_is_null_aware(conn: aiosqlite.Connection):
    await _add(conn, subcategory=None)

    assert await get_document(conn, "devices", None, "/a.pdf") is not None
    assert await get_document(conn, "devices", "sub1", "/a.pdf") is None


async def test_same_path_different_category_is_a_distinct_document(
    conn: aiosqlite.Connection,
):
    first = await _add(conn, category="devices", source_path="/shared.pdf")
    second = await _add(conn, category="topology", source_path="/shared.pdf")

    assert first != second
    docs = await list_documents(conn)
    assert {d["category"] for d in docs} == {"devices", "topology"}


async def test_update_document_replaces_chunks(conn: aiosqlite.Connection):
    doc_id = await _add(conn)

    await update_document(
        conn,
        doc_id,
        "devices",
        None,
        2000.0,
        "hash2",
        {"filename": "a", "chunks_num": 1, "char_num": 5},
        _chunks("new chunk"),
    )

    row = await get_document(conn, "devices", None, "/a.pdf")
    assert row is not None
    assert row["file_mtime"] == 2000.0
    assert row["file_hash"] == "hash2"

    chunks = await conn.execute_fetchall(
        "SELECT chunk_number, text FROM document_chunks WHERE document_id = ?",
        (doc_id,),
    )
    assert [tuple(r) for r in chunks] == [(0, "new chunk")]
    fts_rows = await conn.execute_fetchall(
        "SELECT text FROM chunks_fts WHERE document_id = ?", (doc_id,)
    )
    assert [r[0] for r in fts_rows] == ["new chunk"]


async def test_touch_document_only_updates_mtime(conn: aiosqlite.Connection):
    doc_id = await _add(conn)

    await touch_document(conn, doc_id, 9999.0)

    row = await get_document(conn, "devices", None, "/a.pdf")
    assert row is not None
    assert row["file_mtime"] == 9999.0
    assert row["file_hash"] == "hash1"  # unchanged
    chunks = list(
        await conn.execute_fetchall(
            "SELECT count(*) FROM document_chunks WHERE document_id = ?", (doc_id,)
        )
    )
    assert chunks[0][0] == 2  # unchanged


async def test_delete_document_removes_chunks_and_fts(conn: aiosqlite.Connection):
    doc_id = await _add(conn)

    await delete_document(conn, doc_id)

    assert await get_document(conn, "devices", None, "/a.pdf") is None
    remaining_chunks = list(
        await conn.execute_fetchall(
            "SELECT count(*) FROM document_chunks WHERE document_id = ?", (doc_id,)
        )
    )
    assert remaining_chunks[0][0] == 0
    remaining_fts = list(
        await conn.execute_fetchall(
            "SELECT count(*) FROM chunks_fts WHERE document_id = ?", (doc_id,)
        )
    )
    assert remaining_fts[0][0] == 0


async def test_search_chunks_ranks_and_caps_results(conn: aiosqlite.Connection):
    await add_document(
        conn,
        "devices",
        None,
        "/switch.pdf",
        1.0,
        "h1",
        {"filename": "switch"},
        _chunks("the network switch handles routing", "unrelated filler text"),
    )
    await add_document(
        conn,
        "devices",
        None,
        "/router.pdf",
        1.0,
        "h2",
        {"filename": "router"},
        _chunks("the router forwards packets"),
    )

    results = await search_chunks(conn, "switch", limit=5)
    assert len(results) == 1
    assert results[0]["filename"] == "switch"
    assert results[0]["snippet"] == "the network **switch** handles routing"
    assert results[0]["score"] > 0

    capped = await search_chunks(conn, "the", limit=1)
    assert len(capped) == 1

    everything = await search_chunks(conn, "the", limit=5)
    paged = await search_chunks(conn, "the", limit=5, offset=1)
    assert [r["filename"] for r in paged] == [r["filename"] for r in everything[1:]]


async def test_search_chunks_returns_page_and_section(conn: aiosqlite.Connection):
    await add_document(
        conn,
        "devices",
        None,
        "/a.pdf",
        1.0,
        "h1",
        {"filename": "a"},
        [Chunk("bgp peering", page_start=3, page_end=4, section="Routing")],
    )

    (row,) = await search_chunks(conn, "bgp")
    assert (row["page_start"], row["page_end"], row["section"]) == (3, 4, "Routing")


async def test_search_chunks_filters(conn: aiosqlite.Connection):
    for category, subcategory, name in [
        ("networking", None, "top"),
        ("networking", "cisco", "cisco"),
        ("networking", "cisco/aci", "aci"),
        ("networking", "ciscoish", "ciscoish"),
        ("networking", "ciscoX/deep", "wildcard-bait"),
        ("storage", "cisco", "storage"),
    ]:
        await add_document(
            conn,
            category,
            subcategory,
            f"/{name}.pdf",
            1.0,
            name,
            {"filename": name},
            _chunks("bgp"),
        )

    async def names(**filters) -> set[str]:
        return {
            r["filename"] for r in await search_chunks(conn, "bgp", limit=10, **filters)
        }

    assert await names(category="storage") == {"storage"}
    # descendants match, sibling names sharing a prefix don't, "_" isn't a wildcard
    assert await names(category="networking", subcategory="cisco") == {"cisco", "aci"}
    assert await names(subcategory="cisco/aci") == {"aci"}
    assert await names(subcategory="cisco_") == set()
    assert await names(file_name="aci") == {"aci"}


async def test_find_document_by_hash(conn: aiosqlite.Connection):
    doc_id = await _add(conn, subcategory="sub", source_path="a.pdf")

    found = await find_document_by_hash(conn, "hash1")
    assert found is not None
    assert (found["id"], found["category"], found["subcategory"]) == (
        doc_id,
        "devices",
        "sub",
    )
    assert found["source_path"] == "a.pdf"
    assert await find_document_by_hash(conn, "other") is None


async def test_get_document_by_name(conn: aiosqlite.Connection):
    await _add(conn, filename="switch-guide")

    found = await get_document_by_name(conn, "devices", None, "switch-guide")
    assert found is not None

    assert await get_document_by_name(conn, "devices", None, "nope") is None


async def test_get_document_by_name_ambiguous_raises(conn: aiosqlite.Connection):
    await _add(conn, source_path="/a.pdf", filename="notes")
    await _add(conn, source_path="/a.md", filename="notes")

    with pytest.raises(ValueError, match="ambiguous"):
        await get_document_by_name(conn, "devices", None, "notes")


async def test_get_chunks_filters_and_orders(conn: aiosqlite.Connection):
    doc_id = await add_document(
        conn,
        "devices",
        None,
        "/a.pdf",
        1.0,
        "h1",
        {"filename": "a"},
        _chunks("c0", "c1", "c2"),
    )

    chunks = await get_chunks(conn, doc_id, [2, 0])
    assert [(c["chunk_number"], c["text"]) for c in chunks] == [(0, "c0"), (2, "c2")]

    missing = await get_chunks(conn, doc_id, [99])
    assert missing == []


async def test_get_category_tree_nests_subcategories(conn: aiosqlite.Connection):
    await add_document(
        conn, "devices", None, "/root.pdf", 1.0, "h", {"filename": "root"}, _chunks("x")
    )
    await add_document(
        conn,
        "devices",
        "switches/access",
        "/nested.pdf",
        1.0,
        "h",
        {"filename": "nested", "chunks_num": 3},
        _chunks("x", "y", "z"),
    )

    tree = await get_category_tree(conn)
    assert len(tree) == 1
    devices = tree[0]
    assert devices["name"] == "devices"
    assert devices["files"] == [{"name": "root", "chunks_num": None}]
    assert len(devices["subcategories"]) == 1

    switches = devices["subcategories"][0]
    assert switches["name"] == "switches"
    assert switches["files"] == []
    assert len(switches["subcategories"]) == 1

    access = switches["subcategories"][0]
    assert access["name"] == "access"
    assert access["files"] == [{"name": "nested", "chunks_num": 3}]


async def test_get_category_tree_include_files_false(conn: aiosqlite.Connection):
    await _add(conn)

    tree = await get_category_tree(conn, include_files=False)
    assert tree[0]["files"] == []


async def test_get_category_tree_filters_by_category(conn: aiosqlite.Connection):
    await _add(conn, category="devices", source_path="/a.pdf")
    await _add(conn, category="topology", source_path="/b.pdf")

    tree = await get_category_tree(conn, category="devices")
    assert [node["name"] for node in tree] == ["devices"]

"""Schema and CRUD for the documents/chunks tables.

Shared DB-access layer: the ingest orchestrator writes through these
functions, and the MCP server's read-side tools (listing categories, looking
up a document, search) read through the same ones.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document's text, with where it came from.

    Attributes:
        text: The chunk's text.
        page_start: 1-based page the chunk starts on, or None for formats
            without pages.
        page_end: 1-based page the chunk ends on, or None.
        section: Nearest heading before the chunk's start, or None.
    """

    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    subcategory TEXT,
    source_path TEXT NOT NULL,
    file_mtime REAL NOT NULL,
    file_hash TEXT NOT NULL,
    metadata TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    UNIQUE (category, subcategory, source_path)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section TEXT,
    PRIMARY KEY (document_id, chunk_number)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    document_id UNINDEXED,
    chunk_number UNINDEXED
);
"""


async def init_schema(conn: aiosqlite.Connection) -> None:
    """Create the documents/chunks tables and FTS index if they're missing."""
    await conn.executescript(SCHEMA)
    await conn.commit()


async def _replace_chunks(
    conn: aiosqlite.Connection, document_id: int, chunks: list[Chunk]
) -> None:
    """Insert a document's chunks, both into the chunk table and the FTS index."""
    await conn.executemany(
        "INSERT INTO document_chunks "
        "(document_id, chunk_number, text, page_start, page_end, section) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (document_id, number, c.text, c.page_start, c.page_end, c.section)
            for number, c in enumerate(chunks)
        ],
    )
    await conn.executemany(
        "INSERT INTO chunks_fts (document_id, chunk_number, text) VALUES (?, ?, ?)",
        [(document_id, number, c.text) for number, c in enumerate(chunks)],
    )


async def add_document(
    conn: aiosqlite.Connection,
    category: str,
    subcategory: str | None,
    source_path: str,
    file_mtime: float,
    file_hash: str,
    metadata: dict[str, Any],
    chunks: list[Chunk],
) -> int:
    """Insert one document and its pre-split chunks into the db.

    Args:
        conn: Open db connection.
        category: Top-level category, e.g. "devices", "topology", "knowladge".
        subcategory: Slash-joined subcategory path (e.g. "sub1/sub2"), or
            None when the document has no subcategory.
        source_path: File name (with extension, no directory) of the source
            file this document was extracted from, used together with
            category/subcategory as the change-tracking key on re-ingestion.
        file_mtime: Source file's mtime at extraction time.
        file_hash: SHA-256 hex digest of the source file's contents.
        metadata: Freeform JSON-serializable metadata for the document (file
            name, tags, doc type, vendor/version, ...) — whatever a
            ``get_doc_info``-style lookup should return for it.
        chunks: The document's text, already split into chunks.

    Returns:
        The new document's id.
    """
    cursor = await conn.execute(
        "INSERT INTO documents "
        "(category, subcategory, source_path, file_mtime, file_hash, "
        "metadata, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            category,
            subcategory,
            source_path,
            file_mtime,
            file_hash,
            json.dumps(metadata),
            datetime.now(UTC).isoformat(),
        ),
    )
    document_id = cursor.lastrowid
    assert document_id is not None  # set by sqlite on every successful INSERT

    await _replace_chunks(conn, document_id, chunks)
    await conn.commit()
    return document_id


async def update_document(
    conn: aiosqlite.Connection,
    document_id: int,
    category: str,
    subcategory: str | None,
    file_mtime: float,
    file_hash: str,
    metadata: dict[str, Any],
    chunks: list[Chunk],
) -> None:
    """Replace an existing document's metadata and chunks in place.

    Args:
        conn: Open db connection.
        document_id: Id of the document to update.
        category: Top-level category.
        subcategory: Slash-joined subcategory path, or None.
        file_mtime: Source file's mtime at extraction time.
        file_hash: SHA-256 hex digest of the source file's contents.
        metadata: Freeform JSON-serializable metadata.
        chunks: The document's new text, already split into chunks.
    """
    await conn.execute(
        "UPDATE documents SET category = ?, subcategory = ?, file_mtime = ?, "
        "file_hash = ?, metadata = ?, indexed_at = ? WHERE id = ?",
        (
            category,
            subcategory,
            file_mtime,
            file_hash,
            json.dumps(metadata),
            datetime.now(UTC).isoformat(),
            document_id,
        ),
    )
    await conn.execute(
        "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
    )
    await conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
    await _replace_chunks(conn, document_id, chunks)
    await conn.commit()


async def touch_document(
    conn: aiosqlite.Connection, document_id: int, file_mtime: float
) -> None:
    """Record a fresh mtime for a document whose content hash hasn't changed.

    Args:
        conn: Open db connection.
        document_id: Id of the document to update.
        file_mtime: The source file's current mtime.
    """
    await conn.execute(
        "UPDATE documents SET file_mtime = ?, indexed_at = ? WHERE id = ?",
        (file_mtime, datetime.now(UTC).isoformat(), document_id),
    )
    await conn.commit()


async def delete_document(conn: aiosqlite.Connection, document_id: int) -> None:
    """Remove a document, its chunks, and its FTS index entries.

    Args:
        conn: Open db connection.
        document_id: Id of the document to remove. ``document_chunks`` rows
            cascade-delete via their foreign key; ``chunks_fts`` is a plain
            FTS5 table with no foreign key support, so its rows are deleted
            explicitly.
    """
    await conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
    await conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    await conn.commit()


async def get_document(
    conn: aiosqlite.Connection,
    category: str,
    subcategory: str | None,
    source_path: str,
) -> aiosqlite.Row | None:
    """Look up a document's id and change-tracking fields.

    Args:
        conn: Open db connection.
        category: Top-level category.
        subcategory: Slash-joined subcategory path, or None.
        source_path: Source file name as stored on ingestion.

    Returns:
        Row with id/file_mtime/file_hash, or None if not indexed yet. The
        same source_path may exist under a different category/subcategory as
        a distinct document, so all three are used as the lookup key.
    """
    cursor = await conn.execute(
        "SELECT id, file_mtime, file_hash FROM documents "
        "WHERE category = ? AND subcategory IS ? AND source_path = ?",
        (category, subcategory, source_path),
    )
    return await cursor.fetchone()


async def find_document_by_hash(
    conn: aiosqlite.Connection, file_hash: str
) -> aiosqlite.Row | None:
    """Look up a document whose source file has the given content hash.

    Args:
        conn: Open db connection.
        file_hash: SHA-256 hex digest of a source file's contents.

    Returns:
        Row with id/category/subcategory/source_path of one such document, or
        None if no indexed document has that content.
    """
    cursor = await conn.execute(
        "SELECT id, category, subcategory, source_path FROM documents "
        "WHERE file_hash = ? LIMIT 1",
        (file_hash,),
    )
    return await cursor.fetchone()


async def list_documents(conn: aiosqlite.Connection) -> list[aiosqlite.Row]:
    """Return every document's id/category/subcategory/source_path.

    Used by the ingest orchestrator's directory mode to detect documents
    whose source file has disappeared (or moved to a different category)
    since the last run.
    """
    cursor = await conn.execute(
        "SELECT id, category, subcategory, source_path FROM documents"
    )
    return list(await cursor.fetchall())


async def search_chunks(
    conn: aiosqlite.Connection,
    pattern: str,
    *,
    category: str | None = None,
    subcategory: str | None = None,
    file_name: str | None = None,
    limit: int = 5,
    offset: int = 0,
) -> list[aiosqlite.Row]:
    """Full-text search over indexed chunks, ranked by relevance.

    Args:
        conn: Open db connection.
        pattern: FTS5 MATCH query — supports "phrase", AND/OR/NOT, prefix*.
        category: Only search this category, or None for all.
        subcategory: Only search this subcategory and the ones nested below
            it (``"cisco"`` also matches ``"cisco/aci"``), or None for all.
        file_name: Only search documents with this extension-stripped
            filename, or None for all.
        limit: Maximum number of matching chunks to return (not distinct
            files — one file may contribute more than one).
        offset: Number of best matches to skip, for paging.

    Returns:
        Rows with category/subcategory/filename/chunk_number/page_start/
        page_end/section plus ``snippet`` (the match in context, hits wrapped
        in ``**``) and ``score`` (higher is better), best match first.

    Raises:
        aiosqlite.OperationalError: pattern is not valid FTS5 query syntax.
    """
    query = (
        "SELECT d.category, d.subcategory, "
        "json_extract(d.metadata, '$.filename') AS filename, "
        "f.chunk_number, c.page_start, c.page_end, c.section, "
        # 32 tokens of context around each hit (FTS5 caps it at 64)
        "snippet(chunks_fts, 0, '**', '**', '…', 32) AS snippet, "
        "-bm25(chunks_fts) AS score "
        "FROM chunks_fts f "
        "JOIN documents d ON d.id = f.document_id "
        "JOIN document_chunks c "
        "ON c.document_id = f.document_id AND c.chunk_number = f.chunk_number "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[Any] = [pattern]
    if category is not None:
        query += " AND d.category = ?"
        params.append(category)
    if subcategory is not None:
        # Prefix compare rather than LIKE, so "_"/"%" in names aren't wildcards.
        query += (
            " AND (d.subcategory = ? "
            "OR substr(d.subcategory, 1, length(?) + 1) = ? || '/')"
        )
        params += [subcategory, subcategory, subcategory]
    if file_name is not None:
        query += " AND json_extract(d.metadata, '$.filename') = ?"
        params.append(file_name)
    query += " ORDER BY rank LIMIT ? OFFSET ?"
    params += [limit, offset]

    cursor = await conn.execute(query, params)
    return list(await cursor.fetchall())


async def get_document_by_name(
    conn: aiosqlite.Connection,
    category: str,
    subcategory: str | None,
    file_name: str,
) -> aiosqlite.Row | None:
    """Look up a document's id by its user-facing name.

    Args:
        conn: Open db connection.
        category: Top-level category.
        subcategory: Slash-joined subcategory path, or None.
        file_name: Extension-stripped filename, as stored in
            ``metadata->>'filename'`` (what ``list_doc``/``search_doc``
            return).

    Returns:
        Row with the document's id, or None if nothing matches.

    Raises:
        ValueError: More than one document matches — two source files with
            the same stem but different extensions in the same
            category/subcategory. Failing explicitly beats silently picking
            one.
    """
    cursor = await conn.execute(
        "SELECT id FROM documents "
        "WHERE category = ? AND subcategory IS ? "
        "AND json_extract(metadata, '$.filename') = ?",
        (category, subcategory, file_name),
    )
    rows = list(await cursor.fetchall())
    if len(rows) > 1:
        raise ValueError(
            f"ambiguous file_name {file_name!r} under category={category!r} "
            f"subcategory={subcategory!r}: {len(rows)} documents match"
        )
    return rows[0] if rows else None


async def get_chunks(
    conn: aiosqlite.Connection, document_id: int, chunk_numbers: list[int]
) -> list[aiosqlite.Row]:
    """Fetch specific chunks of one document, in chunk-number order.

    Args:
        conn: Open db connection.
        document_id: Id of the document to read from.
        chunk_numbers: Which chunk numbers to return.

    Returns:
        Rows with chunk_number/text/page_start/page_end/section for the chunk
        numbers that exist; numbers with no matching chunk are silently
        omitted.
    """
    cursor = await conn.execute(
        "SELECT chunk_number, text, page_start, page_end, section "
        "FROM document_chunks "
        "WHERE document_id = ? "
        "AND chunk_number IN (SELECT value FROM json_each(?)) "
        "ORDER BY chunk_number",
        (document_id, json.dumps(chunk_numbers)),
    )
    return list(await cursor.fetchall())


def _tree_node(name: str) -> dict[str, Any]:
    return {"name": name, "files": [], "subcategories": {}}


def _finalize_tree(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": node["name"],
        "files": node["files"],
        "subcategories": [
            _finalize_tree(child) for child in node["subcategories"].values()
        ],
    }


async def get_category_tree(
    conn: aiosqlite.Connection,
    category: str | None = None,
    include_files: bool = True,
) -> list[dict[str, Any]]:
    """Build a nested category/subcategory/file tree.

    Args:
        conn: Open db connection.
        category: Restrict to one top-level category, or None for all.
        include_files: Whether to attach file entries to each node (for
            ``list_doc``) or leave every node's ``files`` empty (for
            ``list_categories``).

    Returns:
        List of category nodes, each shaped
        ``{"name": str, "files": [...], "subcategories": [...]}``, recursing
        one level per subcategory path segment. File entries are
        ``{"name": filename, "chunks_num": int}``.
    """
    query = (
        "SELECT category, subcategory, "
        "json_extract(metadata, '$.filename') AS filename, "
        "json_extract(metadata, '$.chunks_num') AS chunks_num "
        "FROM documents"
    )
    params: tuple[str, ...] = ()
    if category is not None:
        query += " WHERE category = ?"
        params = (category,)

    cursor = await conn.execute(query, params)
    rows = await cursor.fetchall()

    categories: dict[str, dict[str, Any]] = {}
    for row in rows:
        node = categories.setdefault(row["category"], _tree_node(row["category"]))
        if row["subcategory"]:
            for part in row["subcategory"].split("/"):
                node = node["subcategories"].setdefault(part, _tree_node(part))
        if include_files:
            node["files"].append(
                {"name": row["filename"], "chunks_num": row["chunks_num"]}
            )

    return [_finalize_tree(node) for node in categories.values()]

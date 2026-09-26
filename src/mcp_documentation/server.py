import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import aiosqlite
from fastmcp import FastMCP
from pydantic import Field

from .config import get_config
from .db import connect
from .documents import (
    get_category_tree,
    get_chunks,
    get_document_by_name,
    search_chunks,
)
from .models import (
    CategoryNode,
    CategoryTree,
    ChunkEntry,
    ReadDocResult,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)

_db_conn: aiosqlite.Connection | None = None


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncGenerator[None]:
    """Open the db connection for the server's lifetime, close it on shutdown."""
    global _db_conn
    _db_conn = await connect()
    try:
        yield
    finally:
        await _db_conn.close()
        _db_conn = None


mcp = FastMCP("mcp-documentation", lifespan=_lifespan)


def get_conn() -> aiosqlite.Connection:
    """Return the db connection opened for this server's lifetime."""
    if _db_conn is None:
        raise RuntimeError(
            "Database connection not initialized; server lifespan has not started."
        )
    return _db_conn


@mcp.tool
async def search_doc(
    pattern: Annotated[
        str,
        Field(
            description="SQLite FTS5 MATCH query. Examples: 'BGP AND OSPF', "
            "'\"MP-BGP\"', 'L3Out OR \"external routing\"', "
            "'NEAR(bgp ospf, 10)', 'config*'. Quote any term containing "
            "punctuation such as '-'."
        ),
    ],
    category: Annotated[
        str | None, Field(description="Only search this category, or null for all")
    ] = None,
    subcategory: Annotated[
        str | None,
        Field(
            description="Only search this subcategory, including those nested "
            "below it ('cisco' also matches 'cisco/aci'), or null for all"
        ),
    ] = None,
    file_name: Annotated[
        str | None,
        Field(
            description="Only search the document with this extension-stripped "
            "filename, or null for all"
        ),
    ] = None,
    offset: Annotated[
        int, Field(ge=0, description="Skip this many best matches, for paging")
    ] = 0,
) -> SearchResponse:
    """Full-text keyword search over indexed document chunks.

    Matching is case-insensitive and word-based: punctuation splits words, so
    MP-BGP is indexed as the two words "mp" and "bgp". Unquoted, a '-' is
    query syntax and fails, so write "MP-BGP" in double quotes (a phrase).
    Space-separated terms are implicitly ANDed; there are no synonyms, so
    widen a search with OR ('L3Out OR "external routing"').

    Each result carries a short snippet around the hits and a score; use
    read_doc with the result's chunk_number for the full chunk text.

    Returns:
        Up to `max_search_results` matching chunks (see config.yaml), best
        match first, plus `next_offset` when more matches exist.

    Raises:
        ValueError: pattern is not valid FTS5 query syntax.
    """
    limit = get_config().max_search_results
    try:
        rows = await search_chunks(
            get_conn(),
            pattern,
            category=category,
            subcategory=subcategory,
            file_name=file_name,
            limit=limit + 1,
            offset=offset,
        )
    except aiosqlite.OperationalError as exc:
        raise ValueError(
            f"invalid search query {pattern!r}: {exc}. Wrap terms containing "
            'punctuation in double quotes, e.g. "MP-BGP".'
        ) from exc

    return SearchResponse(
        results=[
            SearchResult(
                category=row["category"],
                subcategory=row["subcategory"],
                file_name=row["filename"],
                chunk_number=row["chunk_number"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section=row["section"],
                score=round(row["score"], 3),
                snippet=row["snippet"],
            )
            for row in rows[:limit]
        ],
        next_offset=offset + limit if len(rows) > limit else None,
    )


@mcp.tool
async def list_doc(
    category: Annotated[
        str | None,
        Field(description="Restrict to one top-level category, or null for all"),
    ] = None,
) -> CategoryTree:
    """List indexed categories, subcategories, and files.

    Returns:
        A nested category/subcategory tree. Each file entry carries its
        `chunks_num` so `read_doc`'s pagination bounds are visible up
        front.
    """
    tree = await get_category_tree(get_conn(), category=category, include_files=True)
    return CategoryTree(categories=[CategoryNode.model_validate(node) for node in tree])


@mcp.tool
async def list_categories() -> CategoryTree:
    """List indexed categories and subcategories, without file names."""
    tree = await get_category_tree(get_conn(), include_files=False)
    return CategoryTree(categories=[CategoryNode.model_validate(node) for node in tree])


@mcp.tool
async def read_doc(
    file_name: Annotated[
        str,
        Field(
            description="Document's extension-stripped filename, as returned by "
            "list_doc/search_doc"
        ),
    ],
    category: Annotated[str, Field(description="Document's category")],
    subcategory: Annotated[
        str | None, Field(description="Document's subcategory, or null if it has none")
    ],
    chunk_numbers: Annotated[
        list[int],
        Field(
            description="Which chunk numbers to return - capped at "
            "max_chunks_per_read per call (see config.yaml)"
        ),
    ],
) -> ReadDocResult:
    """Read specific chunks of one document.

    Raises:
        ValueError: Too many chunk_numbers requested, no document matches,
            or file_name is ambiguous within category/subcategory.
    """
    max_chunks_per_read = get_config().max_chunks_per_read
    if len(chunk_numbers) > max_chunks_per_read:
        raise ValueError(f"at most {max_chunks_per_read} chunk_numbers per call")

    conn = get_conn()
    document = await get_document_by_name(conn, category, subcategory, file_name)
    if document is None:
        raise ValueError(
            f"no document {file_name!r} under category={category!r} "
            f"subcategory={subcategory!r}"
        )

    chunks = await get_chunks(conn, document["id"], chunk_numbers)
    return ReadDocResult(
        chunks=[
            ChunkEntry(
                chunk_number=row["chunk_number"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section=row["section"],
                text=row["text"],
            )
            for row in chunks
        ]
    )

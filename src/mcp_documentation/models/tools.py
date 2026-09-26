"""Pydantic models for MCP tool outputs.

Giving each tool a typed return model (instead of a plain dict) lets FastMCP
generate a real output schema and return structured content, rather than an
untyped JSON blob the caller has to guess the shape of.
"""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One matching chunk returned by ``search_doc``."""

    category: str
    subcategory: str | None
    file_name: str
    chunk_number: int
    page_start: int | None
    page_end: int | None
    section: str | None
    score: float = Field(description="Relevance (BM25), higher is better")
    snippet: str = Field(
        description="The match in context, hits wrapped in ** — use read_doc "
        "for the whole chunk"
    )


class SearchResponse(BaseModel):
    """Result of ``search_doc``."""

    results: list[SearchResult] = Field(default_factory=list)
    next_offset: int | None = Field(
        default=None,
        description="Pass as offset to get the next page, or null when there "
        "are no more matches",
    )


class FileEntry(BaseModel):
    """One file leaf in a category tree, as returned by ``list_doc``."""

    name: str
    chunks_num: int | None = None


class CategoryNode(BaseModel):
    """One node in the category tree from ``list_doc``/``list_categories``."""

    name: str
    files: list[FileEntry] = Field(default_factory=list)
    subcategories: list["CategoryNode"] = Field(default_factory=list)


CategoryNode.model_rebuild()


class CategoryTree(BaseModel):
    """Top-level result of ``list_doc``/``list_categories``."""

    categories: list[CategoryNode] = Field(default_factory=list)


class ChunkEntry(BaseModel):
    """One chunk returned by ``read_doc``."""

    chunk_number: int
    page_start: int | None
    page_end: int | None
    section: str | None
    text: str


class ReadDocResult(BaseModel):
    """Result of ``read_doc``."""

    chunks: list[ChunkEntry] = Field(default_factory=list)

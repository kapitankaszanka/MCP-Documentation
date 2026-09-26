from functools import lru_cache

from pydantic import BaseModel

from .paths import load_config, user_data_dir


class Config(BaseModel):
    """Configuration object. Store all config variables."""

    db_path: str
    max_search_results: int = 5
    max_chunks_per_read: int = 5
    chunk_size: int = 2048
    chunk_overlap: int = 254


@lru_cache
def get_config() -> Config:
    """Return cached settings object."""
    config = load_config()
    db_path = config.get("db_path")
    if db_path is None:
        db_path = str(user_data_dir() / "mcp-documentation.db")

    defaults = Config(db_path=db_path)
    return Config(
        db_path=db_path,
        max_search_results=config.get(
            "max_search_results", defaults.max_search_results
        ),
        max_chunks_per_read=config.get(
            "max_chunks_per_read", defaults.max_chunks_per_read
        ),
        chunk_size=config.get("chunk_size", defaults.chunk_size),
        chunk_overlap=config.get("chunk_overlap", defaults.chunk_overlap),
    )

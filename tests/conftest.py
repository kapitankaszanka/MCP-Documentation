"""Shared fixtures: an isolated db connection and an isolated config."""

from collections.abc import AsyncIterator, Callable
from pathlib import Path

import aiosqlite
import pytest
import yaml

from mcp_documentation.config import get_config
from mcp_documentation.documents import init_schema


@pytest.fixture(autouse=True)
def _isolated_config_cache():
    """Clear get_config()'s lru_cache before and after every test.

    Without this, whichever test runs first would freeze its Config for the
    rest of the session, since get_config() is cached at module scope.
    """
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Point get_config() at a throwaway config.yaml for the duration of a test.

    Returns a function so a test can call it with only the keys it cares
    about; anything unset falls back to Config's own defaults.
    """

    def _configure(**overrides: object) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        monkeypatch.setenv("MCP_DOCUMENTATION_CONFIG", str(config_path))
        get_config.cache_clear()
        return config_path

    return _configure


@pytest.fixture
async def conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """Open a fresh sqlite db with the schema applied, closed after the test."""
    connection = await aiosqlite.connect(tmp_path / "test.db")
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys = ON")
    await init_schema(connection)
    try:
        yield connection
    finally:
        await connection.close()

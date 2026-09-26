"""Tests for config.py's get_config()."""

from mcp_documentation.config import get_config


def test_defaults_when_config_file_is_empty(configure):
    configure()

    config = get_config()
    assert config.max_search_results == 5
    assert config.max_chunks_per_read == 5
    assert config.chunk_size == 2048
    assert config.chunk_overlap == 254
    assert config.db_path  # falls back to a platform data path, just not empty


def test_overrides_from_config_file(configure, tmp_path):
    db_path = str(tmp_path / "custom.db")
    configure(
        db_path=db_path,
        max_search_results=2,
        max_chunks_per_read=3,
        chunk_size=100,
        chunk_overlap=10,
    )

    config = get_config()
    assert config.db_path == db_path
    assert config.max_search_results == 2
    assert config.max_chunks_per_read == 3
    assert config.chunk_size == 100
    assert config.chunk_overlap == 10


def test_get_config_is_cached(configure):
    configure(max_search_results=7)

    assert get_config() is get_config()


def test_cache_clear_picks_up_new_config(configure):
    configure(max_search_results=1)
    first = get_config()
    assert first.max_search_results == 1

    configure(max_search_results=2)
    second = get_config()
    assert second.max_search_results == 2

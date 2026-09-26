"""Logging setup — reads the resolved logging YAML and applies ``dictConfig``."""

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml

from .paths import discover_logging_config, user_logs_dir


def _ensure_log_dirs(config: dict[str, Any]) -> None:
    """Rewrite relative handler filenames into the platform log directory.

    ``dictConfig`` raises if a file handler's directory does not exist, so every
    parent is created here. Relative ``filename`` entries (e.g.
    ``logs/mcp-documentation.log``) are redirected to the platformdirs log
    directory — ``%LOCALAPPDATA%`` on Windows, ``~/.local/state`` or
    ``~/.cache`` on Linux — so the server never depends on the working
    directory being writable. Absolute paths are left untouched.

    Args:
        config: Parsed ``dictConfig`` mapping, modified in place.
    """
    base = user_logs_dir()
    for handler in (config.get("handlers") or {}).values():
        raw = handler.get("filename")
        if not raw:
            continue
        path = Path(raw)
        resolved = path if path.is_absolute() else base / path.name
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler["filename"] = str(resolved)


def setup_logging(config_path: Path | None = None) -> None:
    """Load the YAML logging config and apply it.

    Args:
        config_path: Optional override. When omitted, resolves through
            :func:`mcp_documentation.paths.discover_logging_config` (env var →
            user config dir). If that isn't an existing file, a plain
            ``basicConfig`` is installed as a safety net.
    """
    path = config_path or discover_logging_config()

    if not path.is_file():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        )
        logging.getLogger(__name__).warning(
            "Logging config %s not found — falling back to basicConfig", path
        )
        return

    with path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    _ensure_log_dirs(config)
    logging.config.dictConfig(config)
    logging.getLogger(__name__).debug("Logging configured from %s", path)

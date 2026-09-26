"""Cross-platform path, config and data resolution for mcp-documentation.

Every location is resolved through :mod:`platformdirs` so the server behaves
natively on both Windows (``%APPDATA%`` / ``%LOCALAPPDATA%``) and Linux
(``~/.config`` / ``~/.local/share``), and every path is handled as a
:class:`pathlib.Path` — never by string concatenation, never with a hardcoded
separator.

Discovery order for the config/logging file paths themselves (first match
wins):
    1. The dedicated environment variable.
    2. The user config directory.
"""

import logging
import os
from pathlib import Path

import platformdirs
import yaml

logger = logging.getLogger(__name__)

APP_NAME = "mcp/mcp-documentation"

ENV_CONFIG = "MCP_DOCUMENTATION_CONFIG"
ENV_LOGGING_CONFIG = "MCP_DOCUMENTATION_LOG_CONFIG"

CONFIG_FILENAME = "config.yaml"
LOGGING_FILENAME = "logging.yaml"


def user_config_dir() -> Path:
    r"""Return the OS-appropriate config directory.

    ``%APPDATA%\\"$APP_NAME"`` on Windows,
    ``~/.config/"$APP_NAME"`` on Linux.

    Returns:
        Path to the directory (not guaranteed to exist).
    """
    return Path(platformdirs.user_config_dir(APP_NAME, roaming=True))


def user_data_dir() -> Path:
    r"""Return the OS-appropriate data directory for the default database.

    ``%LOCALAPPDATA%\\"$APP_NAME"`` on Windows,
    ``~/.local/share/"$APP_NAME"`` on Linux.

    Returns:
        Path to the directory (not guaranteed to exist).
    """
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def user_logs_dir() -> Path:
    """Return the OS-appropriate log directory.

    Returns:
        Path to the directory (not guaranteed to exist).
    """
    return Path(platformdirs.user_log_dir(APP_NAME))


def _discover(env_var: str, filename: str) -> Path:
    """Resolve a YAML config path using the standard discovery order.

    Args:
        env_var: Environment variable that overrides discovery.
        filename: Bare filename to look for in the user config dir.

    Returns:
        The env var path if set, otherwise the user-config-dir path (which
        may not exist yet, so callers can bootstrap it or fall back to
        defaults).
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override)

    return user_config_dir() / filename


def discover_logging_config() -> Path:
    """Resolve the ``logging.yaml`` path."""
    return _discover(ENV_LOGGING_CONFIG, LOGGING_FILENAME)


def discover_config() -> Path:
    """Resolve the server ``config.yaml`` path."""
    return _discover(ENV_CONFIG, CONFIG_FILENAME)


def _read_yaml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data)}")
    return data


def load_config() -> dict[str, str]:
    """Load config file."""
    config_path = discover_config()
    return _read_yaml(config_path)

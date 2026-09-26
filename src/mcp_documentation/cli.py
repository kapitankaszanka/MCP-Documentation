"""``mcp-documentation`` command-line entry point.

Auto-bootstraps ``config.yaml`` and ``logging.yaml`` into the platform config
directory from the bundled templates on first run, then starts the MCP stdio
server.
"""

import logging
import shutil
import sys
from argparse import ArgumentParser
from importlib import resources
from pathlib import Path

from . import __version__
from .logging_config import setup_logging
from .paths import user_config_dir, user_data_dir, user_logs_dir
from .server import mcp

logger = logging.getLogger(__name__)

_TEMPLATES = ("logging.yaml", "config.yaml")


def _bootstrap_user_config() -> list[Path]:
    """Create the config/cache/log directories and seed missing templates.

    Templates are copied only when the destination does not already exist, so
    user edits are never clobbered. Failures (permission denied, missing
    template) are reported on stderr and swallowed — stdout belongs to the
    JSON-RPC stream — and downstream config discovery surfaces a clearer error
    if something required is still absent.

    Returns:
        Every directory and file this call created, in creation order. Logging
        is not configured yet when this runs, so the caller logs them once
        :func:`setup_logging` has been applied.
    """
    created: list[Path] = []
    config_target = user_config_dir()
    for directory in (config_target, user_data_dir(), user_logs_dir()):
        if directory.exists():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
        except OSError as exc:
            sys.stderr.write(
                f"mcp-documentation: could not create {directory}: {exc}\n"
            )

    template_root = resources.files("mcp_documentation").joinpath("templates")
    try:
        with resources.as_file(template_root) as src_path:
            for src in src_path.rglob("*"):
                if not src.is_file() or "__pycache__" in src.parts:
                    continue
                dst = config_target / src.relative_to(src_path)
                if dst.exists():
                    continue
                if not dst.parent.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    created.append(dst.parent)
                shutil.copyfile(src, dst)
                created.append(dst)
    except OSError as exc:
        sys.stderr.write(f"mcp-documentation: failed to bootstrap templates: {exc}\n")

    return created


def main() -> None:
    """Start the MCP server."""
    args = ArgumentParser(
        prog="MCP-Documentation",
        description=(
            "MCP server that exposes documentation with "
            "full-text search and paginated content retrieval."
        ),
    )
    args.add_argument(
        "--transport",
        default="stdio",
        type=str,
        choices=["stdio", "http"],
    )
    args.add_argument("--host", default="127.0.0.1", type=str)
    args.add_argument("--port", default=9002, type=int)
    args.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    # Parse before bootstrapping so --help/--version have no side effects.
    parsed = args.parse_args()

    created = _bootstrap_user_config()
    setup_logging()
    for path in created:
        kind = "directory" if path.is_dir() else "file"
        logger.info("Bootstrap created %s %s", kind, path)

    # host/port are forwarded to the transport, and the stdio transport takes
    # neither — pass them only for the network transports.
    transport_kwargs: dict[str, object] = {}
    if parsed.transport != "stdio":
        transport_kwargs = {"host": parsed.host, "port": parsed.port}

    logger.info(
        "Starting mcp-documentation %s (transport=%s)", __version__, parsed.transport
    )
    try:
        mcp.run(transport=parsed.transport, show_banner=False, **transport_kwargs)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
    except Exception:
        logger.exception("mcp-documentation terminated with an error")
        sys.exit(1)

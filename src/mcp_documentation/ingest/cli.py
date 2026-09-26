"""``mcp-documentation-ingest`` command-line entry point."""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiosqlite

from .. import __version__
from ..db import session
from ..documents import init_schema, list_documents
from ..logging_config import setup_logging
from .orchestrator import (
    format_document_list,
    ingest_directory,
    ingest_file,
    remove_directory,
    remove_file,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mcp-documentation-ingest",
        description=(
            "Extract a file (or a directory tree of files) into the "
            "mcp-documentation database."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "-f",
        "--file",
        type=Path,
        nargs="+",
        help="One or more files to ingest, all under the same "
        "--category/--subcategory.",
    )
    target.add_argument(
        "-d",
        "--directory",
        type=Path,
        help=(
            "Directory tree to ingest; category/subcategory "
            "are derived from folder names."
        ),
    )
    target.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Print every document in the database and exit.",
    )
    parser.add_argument(
        "--category",
        help="Required with --file (optional filter with --file --remove); "
        "not used with --directory.",
    )
    parser.add_argument(
        "--subcategory",
        help="Optional with --file (slash-joined); not used with --directory.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the files (matched by file name, need not exist anymore) "
        "or every document in the directory's category folders, instead of "
        "ingesting.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args(argv)

    if args.list and (
        args.remove or args.category is not None or args.subcategory is not None
    ):
        parser.error("--list takes no other options")
    if args.file is not None and args.category is None:
        if not args.remove:
            parser.error("--category is required with --file")
        if args.subcategory is not None:
            parser.error("--subcategory requires --category")
    if args.directory is not None and (
        args.category is not None or args.subcategory is not None
    ):
        parser.error(
            "--category/--subcategory aren't used with --directory "
            "(they're derived from folder names)"
        )
    return args


def _log_removed(removed: int, target: Path) -> None:
    if removed:
        logger.info("Removed %d document(s) for %s", removed, target)
    else:
        logger.warning("Nothing to remove for %s", target)


async def _ingest_files(
    conn: aiosqlite.Connection,
    paths: list[Path],
    category: str,
    subcategory: str | None,
) -> None:
    """Ingest several files under one category, continuing past failures.

    Raises:
        RuntimeError: At least one file failed; the others were still
            ingested.
    """
    started = time.perf_counter()
    counts = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}
    for index, path in enumerate(paths, start=1):
        logger.info("[%d/%d] %s", index, len(paths), path)
        try:
            outcome = await ingest_file(conn, path, category, subcategory)
        except Exception as exc:  # one bad file must not abort the rest
            counts["failed"] += 1
            logger.warning("Failed to ingest %s: %s", path, exc)
            continue
        counts[outcome] += 1

    logger.info(
        "Ingest complete in %.2fs: added=%d updated=%d skipped=%d failed=%d",
        time.perf_counter() - started,
        counts["added"],
        counts["updated"],
        counts["skipped"],
        counts["failed"],
    )
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} of {len(paths)} file(s) failed")


async def _run(args: argparse.Namespace) -> None:
    async with session() as conn:
        await init_schema(conn)
        if args.list:
            listing = format_document_list(await list_documents(conn))
            if listing:
                sys.stdout.write(listing)
            else:
                logger.warning("The database is empty")
        elif args.remove:
            if args.file is not None:
                for path in args.file:
                    removed = await remove_file(
                        conn, path, args.category, args.subcategory
                    )
                    _log_removed(removed, path)
            else:
                removed = await remove_directory(conn, args.directory)
                _log_removed(removed, args.directory)
        elif args.file is not None:
            await _ingest_files(conn, args.file, args.category, args.subcategory)
        else:
            await ingest_directory(conn, args.directory)


def main() -> None:
    """Run the ``mcp-documentation-ingest`` command."""
    args = _parse_args()
    setup_logging()
    logger.info("Starting mcp-documentation-ingest %s", __version__)

    try:
        asyncio.run(_run(args))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Ingest failed: %s", exc)
        raise SystemExit(1) from exc

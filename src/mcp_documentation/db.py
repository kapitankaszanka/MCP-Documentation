"""Shared async sqlite connection helpers.

Kept separate from :mod:`server` so any entry point that needs the db —
the MCP server, a standalone indexer/updater script, tests — opens
connections the same way instead of each rolling its own.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiosqlite

from .config import get_config


async def connect() -> aiosqlite.Connection:
    """Open a new connection to the documentation db.

    Returns:
        A connection with :class:`aiosqlite.Row` row factory set, ready for
        querying. Caller owns the connection and must close it.
    """
    conn = await aiosqlite.connect(get_config().db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


@asynccontextmanager
async def session() -> AsyncGenerator[aiosqlite.Connection]:
    """Yield a connection for the duration of the ``async with`` block.

    Use this for a short-lived script (e.g. a db-updating job) that opens a
    connection, does its work, and exits. Long-running servers that need a
    connection for their whole lifetime should use :func:`connect` directly
    and close it themselves on shutdown.
    """
    conn = await connect()
    try:
        yield conn
    finally:
        await conn.close()

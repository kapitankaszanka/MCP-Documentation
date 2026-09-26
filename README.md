# MCP Documentation

An MCP (Model Context Protocol) server that exposes indexed documentation —
PDFs today, plain text/Markdown too, and more formats can be added — with
full-text search and paginated chunk retrieval. Documents are extracted and
chunked ahead of time into a local SQLite database
(`documents`/`document_chunks`/`chunks_fts`); the server only ever reads
from it, it never touches your source files.

## Why this exists

A full RAG setup — embeddings, a vector store, re-indexing pipelines, chunk
strategy tuning — is a lot to build and keep running just to let an LLM
search a folder of documents. This server is the deliberately simpler
alternative: SQLite's FTS5 for keyword search, plain chunked text for
retrieval, and nothing else to operate. No embedding model to keep in sync,
no vector db to run, no ongoing tuning — just a database file and one CLI to
(re)index it.

## Available tools

| Tool | Description |
|---|---|
| `search_doc(pattern, category=None, subcategory=None, file_name=None, offset=0)` | Full-text search over indexed chunks (see [Search](#search)). Returns up to `max_search_results` matches, best first, each with a short snippet instead of the whole chunk. |
| `list_doc(category=None)` | List categories, subcategories, and files as a nested tree, optionally scoped to one category. Each file entry includes its chunk count. |
| `list_categories()` | Same tree, without file names — just the category/subcategory structure. |
| `read_doc(file_name, category, subcategory, chunk_numbers)` | Read specific chunks of one document, up to `max_chunks_per_read` chunk numbers per call. Each chunk comes with its `page_start`/`page_end` and `section`. |

### Search

`search_doc` returns `{results, next_offset}`. Each result has the
document's `category`/`subcategory`/`file_name`, the `chunk_number`, the
chunk's `page_start`/`page_end` (`null` for `.txt`/`.md`) and `section`
(nearest heading before the chunk, or `null`), a BM25 `score` (higher is
better), and a `snippet`: about 32 words around the hits, with the hits
wrapped in `**`. Fetch the whole chunk with `read_doc`.

- **Filters:** `category`, `subcategory` (also matches nested
  subcategories: `cisco` matches `cisco/aci`) and `file_name` narrow the
  search.
- **Paging:** when more matches exist, `next_offset` is set; pass it back as
  `offset` for the next page. It is `null` on the last page.

`pattern` is an [SQLite FTS5 query](https://www.sqlite.org/fts5.html#full_text_query_syntax)
over the default `unicode61` tokenizer:

- Matching is case-insensitive and word-based. Punctuation splits words, so
  `MP-BGP` is indexed as the two words `mp` and `bgp`.
- Space-separated terms must all match (implicit `AND`). `OR`, `NOT`,
  `"phrases"`, `prefix*` and `NEAR(a b, 10)` are supported.
- A bare `-` is query syntax, so `MP-BGP` unquoted is an error. Quote any
  term containing punctuation: `"MP-BGP"`. Invalid queries return an error
  saying so.
- There are no synonyms. Widen a search with `OR`, e.g.
  `L3Out OR "external routing"`.

Examples: `BGP AND OSPF AND L3Out`, `"MP-BGP"`, `NEAR(bgp ospf, 10)`,
`config*`.

### Document identity

Every tool identifies a document by `(category, subcategory, file_name)`:

- `category` and `subcategory` come from where the source file lives (see
  [Ingesting documents](#ingesting-documents) below). `subcategory` is a
  single slash-joined string (e.g. `"switches/access"`) or `null` for a file
  with no subcategory.
- `file_name` is the source file's name with its extension stripped (e.g.
  `config-guide.pdf` → `"config-guide"`), as returned by `list_doc`/`search_doc`.

If two different source files share the same stem in the same
category/subcategory (e.g. `notes.pdf` and `notes.md` side by side),
`read_doc` fails with an explicit error rather than silently picking one —
rename one of them, or ingest them under different categories.

## Ingesting documents

Documents get into the database via the standalone `mcp-documentation-ingest`
CLI, not through an MCP tool — indexing is a deliberate, out-of-band step.

```bash
# One or more files, category given explicitly (shared by all of them)
mcp-documentation-ingest -f /path/to/config-guide.pdf --category devices
mcp-documentation-ingest -f /path/to/notes.md /path/to/vlans.pdf --category devices --subcategory switches/access

# A whole tree — category/subcategory derived from folder names
mcp-documentation-ingest -d /path/to/documentation

# Removal — every entry for a file name, just one category's entry, or
# every document in the tree's category folders
mcp-documentation-ingest -f config-guide.pdf --remove
mcp-documentation-ingest -f notes.md vlans.pdf --remove --category devices --subcategory switches/access
mcp-documentation-ingest -d /path/to/documentation --remove

# List what's indexed: category, subcategory, file name
mcp-documentation-ingest --list

# Print the version and exit (also logged at the start of every run)
mcp-documentation-ingest --version
```

Only the **file name** (e.g. `config-guide.pdf`) is stored as a document's
source; together with category/subcategory it identifies the document. The
path as you passed it on the command line is kept in the document's
metadata as `file_path`. Consequences:

- Two files with the same name in the same category/subcategory are the
  same document — ingesting the second replaces the first.
- `-f … --remove` matches by file name only, so it works even after the file
  has been deleted from disk.
- `-d … --remove` removes every document in the categories that are
  top-level folders of the given directory (which must exist).

> **Upgrading from 0.8.x or older:** 0.9.0 changed the database schema and
> the PDF extraction. Delete the database file (`db_path`, see
> [Configuration](#configuration)) and re-ingest.

**Duplicate content is rejected.** If a file's SHA-256 content hash matches
a document that is already indexed (under any category or name), ingesting it
fails with `identical content already indexed as <category>/<subcategory>/<file>`.
In directory mode that counts as one failed file; the rest of the run
continues. This means the same file can't be indexed under two categories.
Moving a file to another folder in the tree is not a duplicate: vanished
documents are removed before anything is ingested.

In directory mode, the first folder under the given directory becomes
`category`, and every folder below that is joined with `/` into
`subcategory` (no depth limit) — e.g.
`documentation/devices/switches/access/config-guide.pdf` becomes
category `devices`, subcategory `switches/access`. A file sitting directly
in the given directory (no category folder at all) is reported as a failure
for that file, without aborting the rest of the run. Hidden files and
folders (name starting with `.`, e.g. `.git/`) are ignored at any depth.

Ingestion is **incremental**: a file's mtime is checked first, falling back
to a SHA-256 content hash if the mtime changed (so a `touch` with no real
edit doesn't trigger a re-extraction). In directory mode, files that
disappeared — or moved to a different category/subcategory — since the last
run are removed from the database; this cleanup only looks at the categories
that are top-level folders of the given directory, so documents in other
categories (e.g. added with `-f`) are left alone.

Progress is logged to stderr as the run goes:

```
Found 3 supported file(s) in documentation
Removed devices/old-guide.pdf (no longer on disk)
[1/3] documentation/devices/config-guide.pdf
Indexing documentation/devices/config-guide.pdf ...
Added documentation/devices/config-guide.pdf: 181 page(s), 412 chunk(s), 523104 chars in 58.31s
[2/3] documentation/devices/switches/intro.md
Skipped documentation/devices/switches/intro.md (unchanged)
...
Ingest complete in 61.02s: scanned=3 added=1 updated=0 removed=1 skipped=2 failed=0
```

Each run ends with that summary (`-f` mode has no scanned/removed), plus a
reason per failure.

Supported file types are a small registry in `ingest/extractors.py` —
`.pdf` and `.txt`/`.md` (read as plain UTF-8) today. Adding a new format is
one function in that file, no changes needed elsewhere.

PDFs are converted page by page to Markdown with
[pymupdf4llm](https://github.com/pymupdf/pymupdf4llm). Its layout model
detects headings, lists and tables and drops page headers and footers. OCR is
off. A cleanup pass then removes leftover markup, lines repeated on most
pages (e.g. `Page 11 of 181`), lone bullet characters and runs of blank
lines. Layout analysis costs about 0.3 s per page, so a large document takes
a few minutes to ingest.

Chunks are cut at natural boundaries rather than at a fixed offset. Once a
chunk is at least half of `chunk_size`, it ends at the next Markdown
heading. Otherwise it ends at the last blank line, then sentence end, then
whitespace before `chunk_size`. Each chunk stores the pages it spans and the
nearest heading before it (`section`).

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

```bash
uv tool install git+https://github.com/kapitankaszanka/MCP-Documentation.git
```

This creates two commands: `mcp-documentation` (the server) and
`mcp-documentation-ingest` (the indexer, see above). To upgrade later:

```bash
uv tool upgrade mcp-documentation
```

## Configuration

On first start the server copies the bundled `config.yaml` and `logging.yaml`
templates into the platform config directory, if they are not there already
(your edits are never overwritten):

- **Linux:** `~/.config/mcp/mcp-documentation/`
- **Windows:** `%APPDATA%\mcp\mcp-documentation\`

`config.yaml`:

```yaml
db_path: null  # optional — defaults to the platform data dir if unset
max_search_results: 5
max_chunks_per_read: 5
chunk_size: 2048
chunk_overlap: 254
```

- `db_path` — path to the SQLite database file. Defaults to
  `~/.local/share/mcp/mcp-documentation/mcp-documentation.db` (Linux) /
  `%LOCALAPPDATA%\mcp\mcp-documentation\mcp-documentation.db` (Windows) if
  unset. Shared by the server and the
  `mcp-documentation-ingest` CLI — both must point at the same file.
- `max_search_results` — cap on how many chunks `search_doc` returns, and
  the page size for its `offset` paging.
- `max_chunks_per_read` — cap on how many `chunk_numbers` `read_doc` accepts
  per call; request a wider range in several calls.
- `chunk_size` — maximum characters per chunk when ingesting a document.
- `chunk_overlap` — up to this many characters from the end of a chunk are
  repeated at the start of the next one (starting on a word boundary), so a
  match sitting on a chunk boundary still has context in at least one chunk.
  No overlap is added when a chunk ends at a heading.

Changing `chunk_size`/`chunk_overlap` only affects documents ingested (or
re-ingested) after the change — it doesn't retroactively re-chunk what's
already in the database.

### Environment variable overrides

| Variable | Overrides |
|---|---|
| `MCP_DOCUMENTATION_CONFIG` | Path to `config.yaml` itself |
| `MCP_DOCUMENTATION_LOG_CONFIG` | Path to `logging.yaml` |

Resolution order for the config/logging file *paths* themselves: env var →
user config dir. If neither file exists, built-in defaults are used. Everything inside
`config.yaml` (including `db_path`) is read only from that file — set it
there, not via an env var.

## Running the server

`mcp-documentation` supports two transports, chosen with `--transport`:

```bash
# stdio (default) — for a client that spawns the server as a subprocess
# and talks JSON-RPC over stdin/stdout (e.g. Claude Code, Claude Desktop).
mcp-documentation
mcp-documentation --transport stdio

# Streamable HTTP — for a client that connects over the network instead.
mcp-documentation --transport http --host 127.0.0.1 --port 9002
```

- `--transport` — `stdio` (default) or `http`.
- `--host` — bind address for `http` (default `127.0.0.1`); ignored on `stdio`.
- `--port` — bind port for `http` (default `9002`); ignored on `stdio`.
- `-v`, `--version` — print the version and exit.

The version is also logged on every start (`Starting mcp-documentation
<version> …`).

On `stdio`, stdout is reserved entirely for the JSON-RPC stream — nothing
else may write to it (see [Logging](#logging)). On `http`, the server runs
as a Streamable HTTP endpoint at `http://<host>:<port>/mcp/`, served by
Uvicorn.

## Supported MCP protocol

Built on [FastMCP](https://github.com/jlowin/fastmcp) 4.x
(`fastmcp>=4.0.0`), which implements the MCP specification's Streamable HTTP
and stdio transports and negotiates the protocol version per-connection with
the client — no version needs to be picked or configured here. The
underlying `mcp`/`mcp-types` packages this pulls in support protocol
versions `2024-11-05` through `2025-11-25` (handshake-compatible) and
`2026-07-28` (latest); a client requesting an older or unrecognized version
gets FastMCP's standard negotiation-failure response rather than a silent
mismatch.

## Logging

`logging.yaml` follows `logging.config.dictConfig`. A relative handler
`filename` is rewritten at startup into the platform log directory, so it works
unchanged on both OSes. `console_stdout` is disabled by default on purpose:
stdout carries the JSON-RPC stream on stdio transport and any extra output
would corrupt it.

## Development

```bash
uv sync
uv run pytest
uvx ruff check
uvx pyrefly check
```

A `pre-push` hook in `.githooks/` runs all three checks and blocks the push
if any fails. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Out of scope

- OCR of scanned PDFs.
- Semantic/embedding search.
- Shelling out to external binaries.
- Writing to the source documents.
- Authentication — assume local/trusted use.

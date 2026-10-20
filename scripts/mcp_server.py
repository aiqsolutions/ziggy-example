#!/usr/bin/env python3
# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Serve the index to an AI client over MCP: Claude Code, Claude Desktop, or anything else that speaks it.

Two tools, both extraction rather than interpretation:

- `search` — passages with a `path:line` on each, so a caller can check a claim instead of trusting it.
- `get_document` — one whole document by id, for when a passage is not enough. The escalation, not the
  default: it is a much larger read.

Nothing here writes prose. A model paraphrasing into another model's context produces premises nothing
downstream can check; a quoted passage with a citation can be checked in seconds.

**stdio, not HTTP.** The client starts this script as a subprocess and exchanges JSON-RPC messages with
it over stdin and stdout, one per line. Nothing listens on the network, so there is nothing to
authenticate: the only caller is whoever is running the client. Serving one index to a whole team means
HTTP, and HTTP means authentication and deciding who may see what. That is a larger job, and not one
this example attempts.

No MCP SDK. A server this size needs four methods, and they are easier to understand read than imported.

    claude mcp add ziggy -- python3 /absolute/path/to/scripts/mcp_server.py    # register with Claude Code
    python3 scripts/mcp_server.py --backend lexical                              # no embedding model needed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus import BACKGROUND_COLLECTIONS, INDEX, Chunk, get_backend, load_index, search  # noqa: E402

# Only a fallback: `handle` echoes whichever revision the client asks for.
PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "ziggy-example", "version": "1.0"}

# Sent once, at the handshake, and passed to the model by the client. This is the place to say how the
# answers should be weighed, which no single tool description has room for.
INSTRUCTIONS = """\
Search this corpus before answering questions about the work it covers. `search` returns passages that
each cite `path:line`; cite them in your answer so a reader can check.

Each passage carries its source and the date it was last updated. When two passages disagree, say so and
show both, rather than silently choosing one.

The `archive` collection describes how things used to work. It is excluded from search unless named, and
it is never a guide to how anything works now.
"""

# Every tool here reads and nothing writes. Declaring it lets a client run them without asking a person
# to approve each call, which is the difference between a tool that gets used and one that gets abandoned.
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
TOOL_NAMES = ("search", "get_document")


def tool_definitions(collections: list[str]) -> list[dict]:
    background = ", ".join(f"'{c}'" for c in sorted(BACKGROUND_COLLECTIONS))
    return [
        {
            "name": "search",
            "description": (
                "Search the corpus. Returns the passages that best match, each citing the file and line it "
                "came from, its source, and when it was last updated."
            ),
            "annotations": READ_ONLY,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A question or keywords, in plain language."},
                    "top_k": {"type": "integer", "description": "Passages to return (default 5, at most 20)."},
                    "collection": {
                        "type": "string",
                        "enum": collections,
                        "description": f"Search one collection only. {background} is excluded unless named here.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_document",
            "description": (
                "Fetch one whole document by its id (for example NOTIFY-118), when a passage from `search` "
                "is not enough. A much larger read than a passage."
            ),
            "annotations": READ_ONLY,
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Document id, as `search` shows it."}},
                "required": ["id"],
            },
        },
    ]


class Index:
    """The built index and the backend that searches it, loaded once at startup."""

    def __init__(self, chunks: list[Chunk], backend, backend_name: str, root: Path):
        self.chunks, self.backend, self.backend_name = chunks, backend, backend_name
        # Chunk paths are relative to the directory that holds corpus/ and index/.
        self.root = root
        self.documents = {c.doc_id.lower(): (c.doc_id, c.path) for c in chunks}
        self.collections = sorted({c.collection for c in chunks})

    @classmethod
    def load(cls, index_dir: Path, backend_name: str) -> Index:
        chunks, vectors = load_index(index_dir)
        if backend_name != "lexical" and not vectors:
            print("No embeddings in the index — serving lexical search instead.", file=sys.stderr)
            backend_name = "lexical"
        return cls(chunks, get_backend(backend_name, chunks, vectors), backend_name, index_dir.parent)

    def search(self, query: str, top_k: int, collection: str | None) -> str:
        results = search(query, self.chunks, self.backend, top_k, collection)
        if not results:
            return f"No passages matched {query!r}."
        parts = [f"{len(results)} passage{'' if len(results) == 1 else 's'}, {self.backend_name} search. Each "
                 f"cites its source: check it before relying on anything that matters."]
        for chunk, _score in results:
            archive = "  «archive»" if chunk.collection in BACKGROUND_COLLECTIONS else ""
            body = chunk.text.split("\n\n", 1)[-1]
            parts.append(f"--- {chunk.cite()}  [{chunk.source} · {chunk.doc_id} · updated {chunk.updated or '?'}]"
                         f"{archive}\n{body}")
        return "\n\n".join(parts)

    def document(self, doc_id: str) -> str:
        # By id, so this reaches archive documents too: naming one is asking for it.
        found = self.documents.get(doc_id.strip().lower())
        if not found:
            known = ", ".join(sorted(shown for shown, _ in self.documents.values())[:10])
            raise ValueError(f"no document {doc_id!r} in the index. Known ids include: {known}")
        path = self.root / found[1]
        if not path.is_file():
            raise ValueError(f"{found[0]} is indexed as {found[1]}, which no longer exists. Rebuild the index.")
        return f"{found[1]}\n\n{path.read_text(encoding='utf-8')}"


def call_tool(index: Index, name: str, arguments: dict) -> str:
    """A tool's answer as text. Raises ValueError for a call the model can correct by asking differently."""
    if name == "search":
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("search needs a non-empty query")
        top_k = max(1, min(int(arguments.get("top_k") or 5), 20))
        return index.search(query, top_k, arguments.get("collection") or None)
    return index.document(str(arguments.get("id") or ""))


def result(request_id, value: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(index: Index, message: dict) -> dict | None:
    """One JSON-RPC message in, one response out, or None for a notification, which takes no reply."""
    method, request_id = message.get("method"), message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        # Echo the client's protocol version. A server that insists on its own revision fails handshakes
        # with clients that are otherwise entirely compatible.
        return result(request_id, {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return result(request_id, {})
    if method == "tools/list":
        return result(request_id, {"tools": tool_definitions(index.collections)})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOL_NAMES:
            return error(request_id, -32602, f"Unknown tool: {name}")
        try:
            text, failed = call_tool(index, name, params.get("arguments") or {}), False
        except (ValueError, RuntimeError) as exc:
            # Returned as a result the model reads, not a protocol error a client merely reports, so the
            # model can correct itself: ask again differently, or name a document that exists.
            text, failed = f"{name} failed: {exc}", True
        return result(request_id, {"content": [{"type": "text", "text": text}], "isError": failed})
    if request_id is None:  # a notification, `notifications/initialized` included: nothing to answer
        return None
    return error(request_id, -32601, f"Method not found: {method}")


def send(message: dict) -> None:
    # json.dumps escapes everything outside ASCII, so a message is always exactly one line.
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--backend", choices=["lexical", "vector", "hybrid"], default="vector")
    args = parser.parse_args()

    # stdout is the protocol channel, and anything else printed there corrupts it. Every diagnostic in
    # this process goes to stderr, which clients keep in their MCP logs.
    sys.stdin.reconfigure(encoding="utf-8")  # the protocol is UTF-8 whatever the platform default is
    index = Index.load(args.index, args.backend)
    print(f"ziggy-example MCP on stdio: {len(index.chunks)} passages, {index.backend_name} search", file=sys.stderr)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            send(error(None, -32700, "Parse error"))
            continue
        if not isinstance(message, dict):
            send(error(None, -32600, "Invalid request: one JSON-RPC object per line"))
            continue
        try:
            response = handle(index, message)
        except Exception:
            traceback.print_exc()
            response = error(message["id"], -32603, "Internal error") if message.get("id") is not None else None
        if response is not None:
            send(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())

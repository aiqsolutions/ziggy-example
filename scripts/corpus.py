#!/usr/bin/env python3
# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Chunk everything under corpus/, index it, and answer queries with passages that cite their source.

Retrieval is where the token savings come from: "read thirty documents to find the four that matter"
becomes three quoted passages with a `path:line` on each. That citation is not decoration. It lets a
reader check a passage cheaply instead of trusting a summary, and it is the difference between an
extraction layer and an unaccountable paraphrase.

Each source's ingest script writes markdown with frontmatter to `corpus/<source>/`. This script does
not care which source wrote what; it reads the frontmatter, chunks the body, and keeps the provenance
on every chunk.

Three backends behind one interface:

- **lexical** — BM25, standard library only, no model, no network. Also the control: if embeddings
  do not beat keyword search on your golden set, they are not earning their complexity.
- **vector** — an embedding model served by Ollama, exact cosine similarity. Better on paraphrase.
- **hybrid** — reciprocal rank fusion of the two.

The vector search is a pure-Python loop over an in-memory list. That is correct and fast enough for
thousands of chunks. It is not a vector database, and the backend interface is where that swap would
happen if you outgrow it.

    python3 scripts/corpus.py build                   # chunk + embed, writes index/
    python3 scripts/corpus.py build --lexical-only    # no model, no network
    python3 scripts/corpus.py query "receiver got the same webhook twice"
    python3 scripts/corpus.py stats
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
INDEX = ROOT / "index"

# Ollama's own API has no authentication. Keep it on localhost, or behind something that authenticates.
OLLAMA = os.environ.get("ZIGGY_OLLAMA", "http://127.0.0.1:11434")
# An embedding model turns text into vectors for meaning-search; it never writes a word. It is not
# interchangeable with a generative (chat) model. Changing it means a full rebuild: vectors from two
# models are not comparable.
EMBED_MODEL = os.environ.get("ZIGGY_EMBED_MODEL", "nomic-embed-text")

# Below MIN a section is merged into the next — a two-line section retrieves as noise. Above MAX it is
# split on `###`, so a long document never arrives as one undifferentiated blob.
MIN_CHARS, MAX_CHARS = 200, 2000

# Retrievable only when asked for by name. Old tickets are accurate about the past, which is exactly
# why they must not answer "how do we do this now" by default: a search that returns a three-year-old
# ticket describing a retired process teaches the old way, with a citation.
BACKGROUND_COLLECTIONS = frozenset({"archive"})

# Most passages one document may contribute to a result. One, and that was measured rather than
# assumed: on our own golden questions a cap of one beat two, three and no cap. A caller who needs more
# of a document should open the document, not receive it a section at a time at the cost of the other
# sources that would have filled those slots.
MAX_PER_DOC = 1

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
TOKEN = re.compile(r"[a-z0-9_]+")
STOP = set("the a an and or of to in is it that this for on with as at by from be are was not but if "
           "so than then which what how where when who does do can will its their there".split())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    path: str
    line: int
    text: str
    # Provenance, carried on every chunk because it cannot be recovered from the text afterwards. A
    # reader weighing two passages that disagree needs to know that one is a reviewed document and the
    # other is a ticket comment from two years ago.
    source: str
    collection: str
    container: str = ""
    updated: str = ""

    def cite(self) -> str:
        return f"{self.path}:{self.line}"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def split_document(rel: str, meta: dict, body: str, body_start: int) -> list[Chunk]:
    """Split on `##`, then `###` if a section is oversized; merge undersized sections forward.

    Every line keeps its own line number through the merging and splitting, so a citation is always
    the real line its passage starts on. Recomputing it from offsets is how a citation ends up pointing
    at the blank line above the text, or at the heading of the section before.

    The heading trail is prepended to each chunk's text. A bare paragraph loses the fact that it is
    about, say, retries; the trail is cheap and it is what makes a mid-document passage findable by a
    query that names its subject rather than its wording.
    """
    sections: list[tuple[str, int, list[tuple[int, str]]]] = []
    heading, start, buffer = "", None, []
    for number, line in enumerate(body.splitlines(), start=body_start):
        if line.startswith("## "):
            if buffer:
                sections.append((heading, start, buffer))
            heading, start, buffer = line[3:].strip(), number, []
        elif buffer or line.strip():  # leading blank lines carry nothing, and must not become the citation
            start = number if start is None else start
            buffer.append((number, line))
    if buffer:
        sections.append((heading, start, buffer))

    chunks: list[Chunk] = []

    def emit(head: str, line: int, items: list[tuple[int, str]]) -> None:
        text = "\n".join(t for _, t in items).strip()
        if text:
            trail = " › ".join(x for x in (meta["title"], head) if x)
            chunks.append(Chunk(chunk_id=f"{meta['doc_id']}#{len(chunks)}", doc_id=meta["doc_id"],
                                title=meta["title"], heading=head, path=rel, line=line,
                                text=f"{trail}\n\n{text}", source=meta["source"], collection=meta["collection"],
                                container=meta["container"], updated=meta["updated"]))

    pending = None
    for head, line, items in sections:
        if pending:
            head = " / ".join(h for h in (pending[0], head) if h)
            line, items, pending = pending[1], pending[2] + items, None
        size = len("\n".join(t for _, t in items).strip())
        if size < MIN_CHARS:
            pending = (head, line, items)
        elif size <= MAX_CHARS:
            emit(head, line, items)
        else:
            sub_head, sub_line, sub_items = "", line, []
            for number, text in items:
                if text.startswith("### "):
                    if sub_items:
                        emit(" / ".join(h for h in (head, sub_head) if h), sub_line, sub_items)
                    sub_head, sub_line, sub_items = text[4:].strip(), number, []
                else:
                    sub_items.append((number, text))
            if sub_items:
                emit(" / ".join(h for h in (head, sub_head) if h), sub_line, sub_items)
    if pending:
        emit(*pending)
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source_dir in sorted(d for d in corpus_dir.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for path in sorted(source_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            match = FRONTMATTER.match(text)
            if not match:
                print(f"  skipped {path.name}: no frontmatter", file=sys.stderr)
                continue
            data = yaml.safe_load(match.group(1)) or {}
            meta = {
                "doc_id": str(data.get("id") or path.stem),
                "title": str(data.get("title", path.stem)),
                "source": str(data.get("source") or source_dir.name),
                "collection": str(data.get("collection") or source_dir.name),
                "container": str(data.get("container", "")),
                "updated": str(data.get("updated", "")),
            }
            body_start = text[: match.end()].count("\n") + 1
            chunks += split_document(str(path.relative_to(corpus_dir.parent)), meta, text[match.end():], body_start)
    return chunks


def embed(texts: list[str], host: str, model: str) -> list[list[float]] | None:
    """Embed via Ollama. None when the host is unreachable, rather than raising: a build that falls back
    to lexical is useful, a build that crashes is not."""
    try:
        request = urllib.request.Request(f"{host}/api/embed", data=json.dumps({"model": model, "input": texts}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response).get("embeddings")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  embedding unavailable ({exc})", file=sys.stderr)
        return None


class Lexical:
    """BM25 over the chunk texts."""

    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[Chunk]):
        docs = [tokenize(c.text) for c in chunks]
        self.freqs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg = sum(self.lengths) / len(docs) if docs else 0.0
        appears = Counter(term for d in docs for term in set(d))
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in appears.items()}

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scored = []
        for i, freq in enumerate(self.freqs):
            score = sum(self.idf[t] * freq[t] * (self.K1 + 1)
                        / (freq[t] + self.K1 * (1 - self.B + self.B * self.lengths[i] / (self.avg or 1)))
                        for t in terms if freq.get(t))
            if score > 0:
                scored.append((i, score))
        return sorted(scored, key=lambda r: -r[1])[:top_k]


class Vector:
    """Exact cosine similarity against stored embeddings."""

    def __init__(self, vectors: list[list[float]], host: str, model: str):
        self.vectors, self.host, self.model = [self._unit(v) for v in vectors], host, model

    @staticmethod
    def _unit(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        embedded = embed([query], self.host, self.model)
        if not embedded:
            raise RuntimeError("vector search needs the embedding host — use --backend lexical, which needs nothing")
        q = self._unit(embedded[0])
        return sorted(((i, sum(a * b for a, b in zip(q, v))) for i, v in enumerate(self.vectors)),
                      key=lambda r: -r[1])[:top_k]


class Hybrid:
    """Reciprocal rank fusion: each chunk scores the sum of 1 / (K + rank) across backends.

    Rank rather than score is the point. BM25 scores and cosine similarities are not on comparable
    scales, and normalising them would invent a relationship that is not there. RRF needs no tuning,
    which matters when there is not enough evaluation data to tune a weight honestly. (It is fusion,
    not reranking: nothing here re-reads the passages.)
    """

    K = 60

    def __init__(self, *backends):
        self.backends = backends

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for backend in self.backends:
            for rank, (i, _) in enumerate(backend.search(query, max(top_k * 5, 50)), start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.K + rank)
        return sorted(fused.items(), key=lambda r: -r[1])[:top_k]


def build(corpus_dir: Path, index_dir: Path, host: str, model: str, lexical_only: bool) -> bool:
    """False if embeddings were wanted and could not be had. The caller must be able to tell: an index
    that silently lost its vectors still looks built."""
    chunks = chunk_corpus(corpus_dir)
    sources = Counter(c.source for c in chunks)
    print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents "
          f"({', '.join(f'{s}: {n}' for s, n in sources.items())})")

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    (index_dir / "vectors.json").unlink(missing_ok=True)
    if lexical_only:
        print("lexical-only, as asked")
        return True

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 32):
        batch = embed([c.text for c in chunks[start:start + 32]], host, model)
        if batch is None:
            print(f"DEGRADED: no embeddings — {host} unreachable. The index is lexical-only.", file=sys.stderr)
            return False
        vectors += batch
    (index_dir / "vectors.json").write_text(json.dumps(vectors), encoding="utf-8")
    print(f"embedded with {model}")
    return True


def load_index(index_dir: Path) -> tuple[list[Chunk], list[list[float]] | None]:
    if not (index_dir / "chunks.json").is_file():
        sys.exit(f"No index at {index_dir} — run: python3 scripts/corpus.py build")
    chunks = [Chunk(**row) for row in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
    vectors = index_dir / "vectors.json"
    return chunks, json.loads(vectors.read_text(encoding="utf-8")) if vectors.is_file() else None


def get_backend(name: str, chunks: list[Chunk], vectors, host: str = OLLAMA, model: str = EMBED_MODEL):
    if name == "lexical":
        return Lexical(chunks)
    if not vectors:
        sys.exit("No embeddings in the index — rebuild without --lexical-only, or use --backend lexical")
    if name == "hybrid":
        return Hybrid(Lexical(chunks), Vector(vectors, host, model))
    return Vector(vectors, host, model)


def search(query: str, chunks: list[Chunk], backend, top_k: int, collection: str | None = None,
           max_per_doc: int = MAX_PER_DOC) -> list[tuple[Chunk, float]]:
    """Top passages, at most `max_per_doc` per document, background collections only when named.

    Over-fetches, because the per-document cap and the collection filter both discard from the same
    pool, and one dominant document could otherwise exhaust it before `top_k` distinct ones appear.
    """
    results: list[tuple[Chunk, float]] = []
    per_doc: Counter[str] = Counter()
    for i, score in backend.search(query, max(top_k * 10, 50)):
        chunk = chunks[i]
        if collection and chunk.collection != collection:
            continue
        if not collection and chunk.collection in BACKGROUND_COLLECTIONS:
            continue
        if per_doc[chunk.doc_id] >= max_per_doc:
            continue
        per_doc[chunk.doc_id] += 1
        results.append((chunk, score))
        if len(results) >= top_k:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build", help="chunk and embed")
    build_cmd.add_argument("--corpus", type=Path, default=CORPUS)
    build_cmd.add_argument("--lexical-only", action="store_true")
    query_cmd = sub.add_parser("query", help="search the index")
    query_cmd.add_argument("text")
    query_cmd.add_argument("--backend", choices=["lexical", "vector", "hybrid"], default="vector")
    query_cmd.add_argument("--top", type=int, default=3)
    query_cmd.add_argument("--collection", help=f"one collection only; the only way to reach {', '.join(sorted(BACKGROUND_COLLECTIONS))}")
    query_cmd.add_argument("--chars", type=int, default=500, help="passage characters to print")
    sub.add_parser("stats", help="describe the index")
    args = parser.parse_args()

    if args.command == "build":
        return 0 if build(args.corpus, args.index, OLLAMA, EMBED_MODEL, args.lexical_only) else 2

    chunks, vectors = load_index(args.index)
    if args.command == "stats":
        print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents, embeddings: {'yes' if vectors else 'no'}")
        print("by collection: " + ", ".join(f"{k} ({v})" for k, v in Counter(c.collection for c in chunks).items()))
        return 0

    if args.backend != "lexical" and not vectors:
        print("(no embeddings in the index — searching lexically)", file=sys.stderr)
        args.backend = "lexical"
    results = search(args.text, chunks, get_backend(args.backend, chunks, vectors), args.top, args.collection)
    if not results:
        print("No matches.")
    for chunk, score in results:
        archive = "  «archive»" if chunk.collection in BACKGROUND_COLLECTIONS else ""
        print(f"\n── {chunk.cite()}  [{chunk.source} · {chunk.doc_id} · updated {chunk.updated or '?'}]{archive}  {score:.3f}")
        body = chunk.text.split("\n\n", 1)[-1]
        print(body[: args.chars] + ("…" if len(body) > args.chars else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

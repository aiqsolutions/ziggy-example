#!/usr/bin/env python3
# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Score retrieval against a golden set of questions.

This measures retrieval only — the cheap, objective half: either the document that owns the answer
came back or it did not. No model judges anything.

- **owner recall@k** — did the document that owns the answer appear in the top k?
- **MRR** — how high did it rank? This is the token-cost metric: a hit at rank 1 means the rest of the
  results never need reading.
- **leaks** — for questions marked `must_not_return_collection`, did that collection reach the top k
  *before* the default filter removed it? A leak means the filter is doing real work. No leaks, ever,
  means it is decoration.

Re-run it every time you add a source, and compare. Adding knowledge can make search worse — a
batch of tickets that all share a template's headings will crowd out the documents people actually
wanted — and this is the only thing that will tell you.

Write the questions the way people ask them, not in the documents' own vocabulary. Paraphrase is
where keyword search should lose, and a golden set written in the source's words hides that.

    python3 scripts/eval_retrieval.py                      # every backend the index supports
    python3 scripts/eval_retrieval.py --backend lexical --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from corpus import INDEX, ROOT, get_backend, load_index, search  # noqa: E402


def unfiltered_top(backend, chunks, query: str, k: int) -> list[tuple[str, str]]:
    """The top k distinct documents with no collection filter: what *would* surface without it."""
    seen: dict[str, str] = {}
    for i, _ in backend.search(query, k * 10):
        seen.setdefault(chunks[i].doc_id, chunks[i].collection)
        if len(seen) >= k:
            break
    return list(seen.items())


def score(backend, chunks, goldens: list[dict], k: int) -> dict:
    """Scores the *served* path — `corpus.search`, with its filters and per-document cap.

    Scoring the raw backend instead is tempting and wrong: an evaluation that differs from what users
    actually hit reports numbers that are quietly optimistic, and every decision made on them inherits
    the error.
    """
    hits = partial = leaks = 0
    reciprocal = 0.0
    rows = []
    for golden in goldens:
        top = [c.doc_id for c, _ in search(golden["q"], chunks, backend, k)]
        also = set(golden.get("also_ok") or [])
        if golden["expect"] in top:
            rank = top.index(golden["expect"]) + 1
            hits, reciprocal = hits + 1, reciprocal + 1 / rank
            verdict, detail = "hit", f"rank {rank}"
        elif also & set(top):
            first = next(p for p in top if p in also)
            rank = top.index(first) + 1
            partial, reciprocal = partial + 1, reciprocal + 0.5 / rank
            verdict, detail = "partial", f"{first} at rank {rank}; owner {golden['expect']} absent"
        else:
            verdict, detail = "MISS", f"expected {golden['expect']}, got {', '.join(top) or 'nothing'}"

        forbidden = golden.get("must_not_return_collection")
        if forbidden:
            leaked = [doc for doc, coll in unfiltered_top(backend, chunks, golden["q"], k) if coll == forbidden]
            if leaked:
                leaks += 1
                detail += f" · without the filter, {forbidden} would have returned {', '.join(leaked)}"
        rows.append((verdict, golden["q"], detail))

    total = len(goldens)
    return {"rows": rows, "hits": hits, "partial": partial, "leaks": leaks, "total": total,
            "owner_recall": hits / total, "recall": (hits + partial) / total, "mrr": reciprocal / total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goldens", type=Path, default=ROOT / "eval" / "golden_queries.yaml")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--backend", choices=["lexical", "vector", "hybrid", "all"], default="all")
    parser.add_argument("--k", type=int, default=3, help="distinct documents that count as retrieved")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-under", type=float, metavar="FRACTION",
                        help="exit non-zero if owner-or-peer recall drops below this — makes a regression a failure, not a diff")
    args = parser.parse_args()

    goldens = (yaml.safe_load(args.goldens.read_text(encoding="utf-8")) or {}).get("queries", [])
    if not goldens:
        sys.exit(f"{args.goldens} has no queries")
    chunks, vectors = load_index(args.index)

    results = {}
    for name in (["lexical", "vector", "hybrid"] if args.backend == "all" else [args.backend]):
        if name != "lexical" and not vectors:
            print(f"\n{name}: no embeddings in the index — skipped")
            continue
        r = results[name] = score(get_backend(name, chunks, vectors), chunks, goldens, args.k)
        print(f"\n{name} @ k={args.k}: owner {r['hits']}/{r['total']} ({r['owner_recall']:.0%}) · "
              f"owner-or-peer {r['recall']:.0%} · MRR {r['mrr']:.3f} · {r['leaks']} filtered leak(s)")
        if args.verbose:
            for verdict, query, detail in r["rows"]:
                print(f"  {dict(hit='✓', partial='~', MISS='✗')[verdict]} {query[:58]:58} {detail}")

    if args.fail_under is not None and results and min(r["recall"] for r in results.values()) < args.fail_under:
        print(f"\nFAIL: recall below {args.fail_under:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

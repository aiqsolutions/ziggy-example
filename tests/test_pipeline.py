# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Tests for the parts that fail quietly when they are wrong.

    python3 -m unittest discover tests
"""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from corpus import Chunk, Hybrid, build, search, split_document  # noqa: E402
from guard import blocking, redact, scan_text  # noqa: E402
from ingest_jira import drop_empty_rows, drop_repeated, issue_body, to_document, wiki_to_markdown  # noqa: E402
from mcp_server import Index, handle  # noqa: E402


class Guard(unittest.TestCase):
    def test_blocks_a_pasted_password(self):
        found = scan_text('config:\npassword = "correct-horse-battery-staple"\n', "t")
        self.assertEqual([f["rule"] for f in blocking(found)], ["assigned-secret"])

    def test_never_prints_the_secret(self):
        found = scan_text('password = "correct-horse-battery-staple"', "t")
        self.assertNotIn("correct-horse-battery-staple", found[0]["sample"])
        self.assertEqual(redact("AKIAIOSFODNN7EXAMPLE"), "AKIAIO…PLE (20 chars)")

    def test_internal_address_warns_but_does_not_block(self):
        found = scan_text("the load balancer at 10.0.4.17", "t")
        self.assertEqual(len(found), 1)
        self.assertEqual(blocking(found), [])

    def test_restricted_in_frontmatter_blocks_whether_or_not_quoted(self):
        # The ingest writes quoted values; this rule once matched only the bare word, and every
        # restricted ticket went straight through.
        for value in ("restricted", '"restricted"', "'restricted'"):
            doc = f"---\ntitle: x\nclassification: {value}\n---\nbody\n"
            self.assertTrue(blocking(scan_text(doc, "t")), value)

    def test_restricted_in_prose_is_only_a_description(self):
        doc = "---\ntitle: x\n---\nclassification: restricted tickets are never ingested\n"
        self.assertEqual(blocking(scan_text(doc, "t")), [])

    def test_inline_allow_for_a_known_false_positive(self):
        self.assertEqual(scan_text("SSN example 123-45-6789  <!-- guard: allow ssn -->", "t"), [])
        self.assertTrue(scan_text("SSN example 123-45-6789  <!-- guard: allow jwt -->", "t"))

    def test_the_ingested_document_is_what_gets_guarded(self):
        issue = {"key": "X-1", "fields": {"summary": "s", "labels": ["Restricted"], "description": "text " * 30}}
        document = to_document(issue, issue_body(issue), "tickets", "")
        self.assertEqual([f["rule"] for f in blocking(scan_text(document, "X-1"))], ["restricted-frontmatter"])


class JiraConversion(unittest.TestCase):
    def test_headings_survive_alongside_numbered_lists(self):
        # Jira numbers list items with `#`. Converting headings first turned every `## Heading` into a
        # numbered list item.
        self.assertEqual(wiki_to_markdown("h2. Root cause\n# first\n## nested"), "## Root cause\n1. first\n   1. nested")

    def test_links_and_monospace(self):
        self.assertEqual(wiki_to_markdown("[runbook|https://x.example/r]"), "[runbook](https://x.example/r)")
        self.assertEqual(wiki_to_markdown("[https://x.example|https://x.example|smart-link]"), "https://x.example")
        self.assertEqual(wiki_to_markdown("set {{NOTIFY_PORT}}"), "set `NOTIFY_PORT`")

    def test_an_empty_template_collapses_to_nothing(self):
        template = "|*User need*| |\n|*Acceptance criteria*| |\n|*Notes*| |"
        self.assertEqual(drop_empty_rows(wiki_to_markdown(template)), "")

    def test_a_filled_row_survives(self):
        kept = drop_empty_rows(wiki_to_markdown("|*User need*|Fewer retries.|\n|*Notes*| |"))
        self.assertEqual(kept, "| *User need* | Fewer retries. |")

    def test_a_paragraph_stored_twice_is_kept_once(self):
        once = "Events over the size limit are rejected at intake and never queued."
        self.assertEqual(drop_repeated(f"{once}\n\n{once.replace('.', '')}"), once)

    def test_short_repeats_are_left_alone(self):
        self.assertEqual(drop_repeated("Done.\n\nDone."), "Done.\n\nDone.")


class Chunking(unittest.TestCase):
    META = {"doc_id": "d", "title": "T", "source": "jira", "collection": "tickets", "container": "", "updated": ""}

    def lines_of(self, body, start=1):
        return [c.line for c in split_document("d.md", self.META, body, start)]

    def test_leading_blank_lines_do_not_become_the_citation(self):
        self.assertEqual(self.lines_of("\n\n## First\n" + "a " * 150, start=10), [12])

    def test_subsections_cite_their_own_heading(self):
        body = "## Big\n" + "### One\n" + "x " * 600 + "\n### Two\n" + "y " * 600
        self.assertEqual(self.lines_of(body), [2, 4])

    def test_subsection_citations_survive_a_merge(self):
        body = "## Tiny\nshort\n## Big\n" + "### One\n" + "x " * 600 + "\n### Two\n" + "y " * 600
        chunks = split_document("d.md", self.META, body, 1)
        self.assertEqual([(c.heading, c.line) for c in chunks],
                         [("Tiny / Big", 1), ("Tiny / Big / One", 4), ("Tiny / Big / Two", 6)])

    def test_heading_trail_is_part_of_the_text(self):
        chunk = split_document("d.md", self.META, "## Retries\n" + "z " * 150, 1)[0]
        self.assertTrue(chunk.text.startswith("T › Retries"))


class Ranked:
    def __init__(self, order):
        self.order = order

    def search(self, query, top_k):
        return [(i, 1.0) for i in self.order[:top_k]]


class Retrieval(unittest.TestCase):
    def chunk(self, doc_id, collection="tickets"):
        return Chunk(doc_id, doc_id, "", "", "d.md", 1, "text", "jira", collection)

    def test_rank_fusion_rewards_agreement(self):
        # Second in both lists beats first in only one.
        self.assertEqual(Hybrid(Ranked([0, 1, 2, 3]), Ranked([4, 1, 5, 6])).search("q", 3)[0][0], 1)

    def test_one_passage_per_document(self):
        chunks = [self.chunk("a"), self.chunk("a"), self.chunk("b")]
        self.assertEqual([c.doc_id for c, _ in search("q", chunks, Ranked([0, 1, 2]), 3)], ["a", "b"])

    def test_archive_only_when_asked_for_by_name(self):
        chunks = [self.chunk("old", "archive"), self.chunk("new")]
        self.assertEqual([c.doc_id for c, _ in search("q", chunks, Ranked([0, 1]), 2)], ["new"])
        self.assertEqual([c.doc_id for c, _ in search("q", chunks, Ranked([0, 1]), 2, collection="archive")], ["old"])


class McpServer(unittest.TestCase):
    BODY = "# X-1 Retry limit\n\n" + "A delivery is attempted three times in total, then marked dead. " * 5 + "\n"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        doc = self.root / "corpus" / "jira" / "x-1.md"
        doc.parent.mkdir(parents=True)
        doc.write_text(f"---\ntitle: X-1 Retry limit\nid: X-1\n---\n{self.BODY}", encoding="utf-8")
        chunks = [Chunk("X-1#0", "X-1", "", "", "corpus/jira/x-1.md", 5, "trail\n\nthree attempts", "jira", "tickets",
                        updated="2025-01-02"),
                  Chunk("OLD-1#0", "OLD-1", "", "", "corpus/jira/old-1.md", 5, "trail\n\nretried forever", "jira", "archive")]
        self.index = Index(chunks, Ranked([1, 0]), "lexical", self.root)

    def call(self, method, params=None):
        return handle(self.index, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})

    def tool(self, name, **arguments):
        return self.call("tools/call", {"name": name, "arguments": arguments})["result"]

    def test_handshake_echoes_the_clients_protocol_version(self):
        reply = self.call("initialize", {"protocolVersion": "2099-01-01"})["result"]
        self.assertEqual(reply["protocolVersion"], "2099-01-01")
        self.assertIn("tools", reply["capabilities"])

    def test_notifications_get_no_reply(self):
        self.assertIsNone(handle(self.index, {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_every_tool_is_declared_read_only(self):
        tools = self.call("tools/list")["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["search", "get_document"])
        self.assertTrue(all(t["annotations"]["readOnlyHint"] for t in tools))

    def test_search_cites_path_line_and_date_and_leaves_out_the_archive(self):
        text = self.tool("search", query="retries")["content"][0]["text"]
        self.assertIn("corpus/jira/x-1.md:5", text)
        self.assertIn("updated 2025-01-02", text)
        self.assertNotIn("OLD-1", text)

    def test_a_bad_call_is_an_error_the_model_can_read(self):
        self.assertTrue(self.tool("search", query=" ")["isError"])
        missing = self.tool("get_document", id="NOPE-9")
        self.assertTrue(missing["isError"])
        self.assertIn("X-1", missing["content"][0]["text"])  # names ids that do exist

    def test_get_document_returns_the_whole_file(self):
        self.assertIn("then marked dead", self.tool("get_document", id="x-1")["content"][0]["text"])

    def test_unknown_tool_and_method_are_protocol_errors(self):
        self.assertEqual(self.call("tools/call", {"name": "delete_everything"})["error"]["code"], -32602)
        self.assertEqual(self.call("resources/list")["error"]["code"], -32601)

    def test_stdout_carries_nothing_but_protocol_messages(self):
        # Over stdio, one stray print on stdout breaks the client. So run the real script, end to end.
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.root / "corpus", self.root / "index", "", "", lexical_only=True)
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "search", "arguments": {"query": "how many times is a delivery attempted"}}},
        ]
        run = subprocess.run([sys.executable, str(SCRIPTS / "mcp_server.py"), "--index", str(self.root / "index")],
                             input="".join(json.dumps(r) + "\n" for r in requests),
                             capture_output=True, text=True, timeout=60)
        replies = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual([r["id"] for r in replies], [1, 2])
        self.assertIn("corpus/jira/x-1.md:", replies[1]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()

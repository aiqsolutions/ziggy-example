#!/usr/bin/env python3
# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Pull Jira issues into the corpus as markdown — one file per issue, guarded before it is written.

    python3 scripts/ingest_jira.py --sample sample/jira_issues.json    # offline demo, no Jira needed
    python3 scripts/ingest_jira.py --measure                           # body lengths; writes nothing
    python3 scripts/ingest_jira.py                                     # live, per sources.yml

Live mode reads three environment variables. The token is never printed or logged.

    JIRA_BASE_URL   https://your-site.atlassian.net
    JIRA_EMAIL      the account a classic API token belongs to
    JIRA_TOKEN      the token (read access is all this needs)

With JIRA_EMAIL unset the token is sent as a Bearer token instead. That is what Atlassian's *scoped*
tokens need, and they need it against `https://api.atlassian.com/ex/jira/<cloud-id>`, not the site
URL — as basic auth against the site they return 401, which looks exactly like a wrong token.

Output goes to `corpus/jira/`, replaced wholesale on a successful run. Writing to a temporary
directory and swapping it in means a run that dies halfway leaves yesterday's corpus intact, and a
ticket that was deleted or became restricted since does not linger.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from guard import blocking, scan_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources.yml"
OUT_DIR = ROOT / "corpus" / "jira"

# --- talking to Jira ----------------------------------------------------------------------------


class Jira:
    """GETs, throttled and retried. Errors never include the token."""

    MIN_INTERVAL = 0.2

    def __init__(self):
        base, token = os.environ.get("JIRA_BASE_URL", "").rstrip("/"), os.environ.get("JIRA_TOKEN")
        if not (base and token):
            sys.exit("Set JIRA_BASE_URL and JIRA_TOKEN (and JIRA_EMAIL for a classic token), or use --sample.")
        email = os.environ.get("JIRA_EMAIL")
        auth = (f"Basic {base64.b64encode(f'{email}:{token}'.encode()).decode()}" if email else f"Bearer {token}")
        self.base, self.headers, self.last = base, {"Authorization": auth, "Accept": "application/json"}, 0.0

    def get(self, path: str, retries: int = 3) -> dict:
        wait = self.MIN_INTERVAL - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        request = urllib.request.Request(self.base + path, headers=self.headers)
        endpoint = path.split("?")[0]
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Distinguished because the remedies differ and the codes are easy to conflate.
                if exc.code == 401:
                    sys.exit(f"Jira 401 on {endpoint}: token expired or invalid — or a scoped token sent the wrong way (see the docstring).")
                if exc.code == 403:
                    sys.exit(f"Jira 403 on {endpoint}: the token authenticated but lacks read access.")
                if exc.code == 404:
                    return {}
                if exc.code != 429 and exc.code < 500:
                    sys.exit(f"Jira {exc.code} on {endpoint}")
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2 ** attempt)
        sys.exit(f"Jira unreachable after {retries} attempts: {endpoint}")

    def search_keys(self, jql: str) -> list[str]:
        """Every issue key matching the JQL.

        The v3 search endpoint pages with a `nextPageToken` cursor and reports **no total**, so the only
        way to know how many results there are is to walk them. (The older v2 `/search` now returns
        410 Gone.)
        """
        keys, cursor = [], ""
        while True:
            params = {"jql": jql, "maxResults": 100, "fields": "key", **({"nextPageToken": cursor} if cursor else {})}
            page = self.get(f"/rest/api/3/search/jql?{urllib.parse.urlencode(params)}")
            keys += [issue["key"] for issue in page.get("issues", [])]
            cursor = page.get("nextPageToken") or ""
            if page.get("isLast") or not cursor:
                return keys

    def issue(self, key: str) -> dict:
        """One issue from the **v2** API, deliberately. v2 returns the description as a plain string of
        wiki markup; v3 returns a nested JSON document tree that is several times the size for the
        same text and needs a converter of its own."""
        fields = "summary,description,issuetype,status,project,labels,created,updated,comment"
        issue = self.get(f"/rest/api/2/issue/{key}?fields={fields}")
        if issue:
            fields = issue["fields"]
            fields["comment"] = {**(fields.get("comment") or {}), "comments": self.all_comments(key, issue)}
        return issue

    def all_comments(self, key: str, issue: dict) -> list[dict]:
        """Every comment, not the first page.

        The most damaging trap in this source. The issue call includes only the first page of comments
        (20 by default) while reporting the true `total` — and the longest threads are the most valuable
        ones. An ingest that trusts the first page silently drops most of its best material and looks
        like it worked.
        """
        block = issue["fields"].get("comment") or {}
        comments = list(block.get("comments") or [])
        while len(comments) < block.get("total", 0):
            page = self.get(f"/rest/api/2/issue/{key}/comment?startAt={len(comments)}&maxResults=100")
            if not page.get("comments"):
                break  # a `total` that lies must not spin forever
            comments += page["comments"]
        return comments


# --- turning an issue into markdown -------------------------------------------------------------

HEADING = re.compile(r"^h([1-6])\.\s*", re.MULTILINE)
BULLET = re.compile(r"^(\*+)\s+", re.MULTILINE)
NUMBERED = re.compile(r"^(#+)\s+", re.MULTILINE)
TABLE_ROW = re.compile(r"^\|\|?(.+?)\|\|?\s*$", re.MULTILINE)
# `[text|url]`, and `[url|url|smart-link]` — Jira appends a rendering hint as a third segment, and a
# two-group pattern swallows it into the href, producing a link that goes nowhere.
LINK = re.compile(r"\[([^|\]]+)\|([^|\]]+)(?:\|[^\]]*)?\]")


def wiki_to_markdown(text: str) -> str:
    """Jira wiki markup to markdown, structurally only.

    Headings, lists, tables and code blocks are converted, because the chunker splits on headings and a
    passage that keeps its structure reads far better when it comes back. Inline emphasis is left
    alone: Jira's `*bold*` and its bullet `*` are the same character, and guessing between them mangles
    prose for no retrieval benefit.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\{(?:code(?::[^}]*)?|noformat)\}", "```", text)
    text = re.sub(r"\{\{(.+?)\}\}", r"`\1`", text)  # {{monospace}} — unambiguous, unlike *bold*
    text = LINK.sub(lambda m: m.group(2) if m.group(1).strip() == m.group(2).strip() else f"[{m.group(1)}]({m.group(2)})", text)
    # Numbered lists *before* headings. Jira numbers list items with `#`, and markdown headings are
    # made of `#` — convert `h2.` first and the list rule then turns every new `## Heading` into
    # `1. Heading`, silently flattening the structure the chunker splits on.
    text = NUMBERED.sub(lambda m: "   " * (len(m.group(1)) - 1) + "1. ", text)
    text = HEADING.sub(lambda m: "#" * int(m.group(1)) + " ", text)
    text = BULLET.sub(lambda m: "   " * (len(m.group(1)) - 1) + "- ", text)
    text = TABLE_ROW.sub(lambda m: "| " + " | ".join(c.strip() for c in re.split(r"\|\|?", m.group(1))) + " |", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def drop_empty_rows(text: str) -> str:
    """Remove table rows whose every cell after the label is blank.

    Without this, no length filter can work on tickets. An unfilled story template is a table of
    labels with nothing in them — and it is *longer* than a genuinely useful one-line bug report. Any
    threshold that drops the empty template also drops the real reports. Remove the rows that carry no
    information first, and length means what it should: the empty template collapses to nothing, and
    the short real report stays.

    Left in, those labels do worse than waste space. Every unfilled template says "Acceptance
    Criteria", so a search using those words returns a page of boilerplate.
    """
    kept = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else []
        if len(cells) >= 2 and not any(cells[1:]):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def drop_repeated(text: str) -> str:
    """Remove a paragraph that only restates what came before it — an editor migration or a bad paste
    can store a description twice, and indexing both copies doubles that text's weight for nothing."""
    kept: list[str] = []
    for paragraph in (p for p in re.split(r"\n\s*\n", text) if p.strip()):
        signature = re.sub(r"[^a-z0-9]", "", paragraph.lower())
        if len(signature) > 40 and signature in re.sub(r"[^a-z0-9]", "", " ".join(kept).lower()):
            continue
        kept.append(paragraph)
    return "\n\n".join(kept)


def issue_body(issue: dict) -> str:
    """The text worth indexing: the description, then the discussion."""
    fields = issue["fields"]
    description = drop_empty_rows(drop_repeated(wiki_to_markdown(fields.get("description") or "")))
    parts = ["## Description", "", description] if description else []
    comments = (fields.get("comment") or {}).get("comments") or []
    if comments:
        parts += ["", "## Discussion", ""]
        for comment in comments:
            text = drop_repeated(wiki_to_markdown(comment.get("body") or ""))
            if text:
                who = (comment.get("author") or {}).get("displayName", "unknown")
                parts += [f"**{who}** ({(comment.get('created') or '')[:10]})", "", text, ""]
    return "\n".join(parts).strip()


def to_document(issue: dict, body: str, collection: str, base_url: str) -> str:
    """Frontmatter, then the body. The frontmatter is the provenance: every passage retrieved from this
    file can say which ticket it came from, how current it was, and whether it is archive material."""
    fields = issue["fields"]
    key = issue["key"]
    labels = {label.lower() for label in fields.get("labels") or []}
    meta = {
        "title": f"{key} {fields.get('summary', '')}".strip(),
        "id": key,
        "source": "jira",
        "url": f"{base_url}/browse/{key}" if base_url else "",
        "container": (fields.get("project") or {}).get("key", ""),
        "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        "status": (fields.get("status") or {}).get("name", ""),
        "created": (fields.get("created") or "")[:10],
        "updated": (fields.get("updated") or "")[:10],
        "collection": collection,
        # A label at the source is how someone says "never index this". The guard enforces it.
        "classification": "restricted" if "restricted" in labels else "internal",
    }
    front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items())
    return f"---\n{front}\n---\n\n# {meta['title']}\n\n{body}\n"


def slug(key: str, summary: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", summary.lower()).strip("-")[:60]
    return f"{key.lower()}-{stem}.md" if stem else f"{key.lower()}.md"


# --- the run ------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, help="read issues from a JSON file instead of Jira")
    parser.add_argument("--measure", action="store_true", help="report body lengths and write nothing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    config = (yaml.safe_load(SOURCES.read_text(encoding="utf-8")) or {}).get("jira") or {}
    archive_before = str(config.get("archive_before") or "")
    min_body = int(config.get("min_body") or 0)
    excluded_types = set(config.get("exclude_types") or [])

    if args.sample:
        issues, base_url = json.loads(args.sample.read_text(encoding="utf-8")), ""
    else:
        jira = Jira()
        base_url = jira.base
        keys = jira.search_keys(config["jql"])
        print(f"{len(keys)} issues match: {config['jql'].strip()}", file=sys.stderr)
        issues = (jira.issue(key) for key in keys)

    tmp = args.out.parent / f".{args.out.name}.tmp"
    if not (args.measure or args.dry_run):
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)

    written, too_short, excluded, warned, blocked = 0, [], 0, [], []
    lengths: list[tuple[int, str, str]] = []
    for issue in issues:
        if not issue:
            continue
        fields, key = issue["fields"], issue["key"]
        if (fields.get("issuetype") or {}).get("name") in excluded_types:
            excluded += 1
            continue
        body = issue_body(issue)
        lengths.append((len(body), key, (fields.get("summary") or "")[:50]))
        if args.measure:
            continue
        if len(body) < min_body:
            too_short.append(key)
            continue

        created = (fields.get("created") or "")[:10]
        collection = "archive" if archive_before and created < archive_before else "tickets"
        document = to_document(issue, body, collection, base_url)

        # The guard runs here, in the ingest, rather than as a step someone remembers to run after it.
        # A ticket is written by whoever filed it. It has to be impossible to write a blocked document,
        # not merely customary to check.
        findings = scan_text(document, key)
        if blocking(findings):
            blocked.append((key, sorted({f["rule"] for f in blocking(findings)})))
            continue
        warned += [f for f in findings if f not in blocking(findings)]
        if not args.dry_run:
            (tmp / slug(key, fields.get("summary") or "")).write_text(document, encoding="utf-8")
        written += 1

    if args.measure:
        # Measure before choosing min_body. A guessed threshold silently discards exactly the content
        # that is short *because* it is precise — a one-line definition, a one-line root cause. Look
        # for the gap between the longest placeholder and the shortest thing worth keeping.
        lengths.sort()
        print(f"{len(lengths)} issues. Shortest:")
        for size, key, summary in lengths[:15]:
            print(f"  {size:>6}  {key:<12} {summary}")
        for pct in (10, 25, 50, 75, 90):
            print(f"  p{pct:<3} {lengths[min(int(len(lengths) * pct / 100), len(lengths) - 1)][0]:>6}")
        return 0

    if not args.dry_run:
        shutil.rmtree(args.out, ignore_errors=True)
        tmp.rename(args.out)

    print(f"{'would write' if args.dry_run else 'wrote'} {written} issue(s) to {args.out}: "
          f"{len(too_short)} under min_body {min_body}, {excluded} excluded by type, {len(blocked)} blocked by the guard")
    for key, rules in blocked:
        print(f"  BLOCKED {key}: {', '.join(rules)} — not written")
    for finding in warned:
        print(f"  warning {finding['where']}: {finding['why']} ({finding['sample']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

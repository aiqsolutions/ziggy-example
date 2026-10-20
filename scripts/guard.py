#!/usr/bin/env python3
# Copyright (c) 2026 AIQ Global, Inc. (AIQ Solutions). SPDX-License-Identifier: MIT
# Unsupported example code, provided "as is" without warranty of any kind. See LICENSE and README.md.
"""Refuse to ingest anything that must not leave the building.

**An example, not a control.** Regular expressions catch the obvious and miss the rest. Do not rely
on this to keep personal, health or other regulated data out of anything; see the README.

Three families of rule, in descending order of how much they matter:

1. **Classification** — content someone explicitly marked restricted at its source.
2. **Secrets** — API keys, tokens, private keys, passwords. Tickets really do contain these: someone
   debugging a problem pastes a config block without thinking about it.
3. **Personal data** — identifiers that point at a person. These should never fire. The day one does
   is the day something went wrong upstream, and you want to find out before it is searchable.

**Matches are redacted in the output.** A guard that prints the secret it found into a build log has
moved the secret, not caught it.

Severity is the whole design. BLOCK stops the document; WARN prints and lets it through. A guard
that cries wolf gets switched off, and a switched-off guard is worse than none — so anything that is
usually harmless (an internal IP address in a bug report, say) warns rather than blocks.

The ingest scripts call `scan_text` on every document before writing it. Run directly, this also
scans files, or what git is about to commit, so it doubles as a pre-commit hook. Exit status is 1 if
anything blocks.

    python3 scripts/guard.py corpus/
    python3 scripts/guard.py --staged
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BLOCK, WARN = "block", "warn"

# Rule = (id, pattern, why, severity). Add your own; these are a starting point, not a policy.
CLASSIFICATION = [
    # Frontmatter only — matched against the leading `---` block, never prose. A document that
    # *describes* the rule ("restricted tickets are never ingested") is not itself restricted.
    # Quotes optional, and that is not pedantry: the ingest writes YAML values quoted, and an earlier
    # version of this pattern matched only the bare word — so every restricted ticket went straight
    # through, and the guard reported nothing, because from its side there was nothing to report.
    ("restricted-frontmatter", re.compile(r"""^classification:\s*["']?restricted\b""", re.I | re.M),
     "marked restricted at its source", BLOCK),
]

SECRETS = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), "private key block", BLOCK),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id", BLOCK),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "GitHub token", BLOCK),
    ("atlassian-token", re.compile(r"\bATATT[A-Za-z0-9_\-=]{20,}"), "Atlassian API token", BLOCK),
    ("notion-token", re.compile(r"\b(?:ntn_|secret_)[A-Za-z0-9]{36,}\b"), "Notion integration token", BLOCK),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "Slack token", BLOCK),
    ("azure-conn", re.compile(r"DefaultEndpointsProtocol=.{0,200}?AccountKey=[A-Za-z0-9+/=]{20,}"),
     "Azure storage connection string", BLOCK),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "JWT", BLOCK),
    ("assigned-secret", re.compile(
        r"""(?ix)\b(?:api[_-]?key|apikey|secret|password|passwd|pwd|access[_-]?token)\b\s*[:=]\s*
            ['"][^'"\s]{12,}['"]"""), "a secret-looking value assigned to a secret-looking name", BLOCK),
    ("private-address", re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b"),
     "an internal network address — rarely sensitive alone, worth a look", WARN),
]

PERSONAL = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "US SSN format", BLOCK),
    ("dob", re.compile(r"(?i)\b(?:DOB|date of birth)\b.{0,24}?\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "date of birth", BLOCK),
    ("record-number", re.compile(r"(?i)\b(?:MRN|patient\s*(?:id|number)|member\s*id)\b[\s:#]*\d{5,}"),
     "a record number that identifies a person", BLOCK),
]

RULES = {"classification": CLASSIFICATION, "secrets": SECRETS, "personal": PERSONAL}

# An inline escape hatch, because without one the first false positive gets the whole guard switched
# off. Annotate the line in the *source* — so the exception is visible and reviewable where it lives.
ALLOW = re.compile(r"guard:\s*allow(?:\s+(?P<rules>[\w,\s-]+))?")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".ico", ".woff", ".woff2"}


def redact(text: str) -> str:
    """Enough to find it, never enough to use it."""
    text = text.strip()
    return text[:3] + "…" if len(text) <= 12 else f"{text[:6]}…{text[-3:]} ({len(text)} chars)"


def allowed(line: str, rule_id: str) -> bool:
    match = ALLOW.search(line)
    if not match:
        return False
    rules = match.group("rules")
    return not rules or rule_id in rules.replace(",", " ").split()


def scan_text(text: str, where: str, rules: dict | None = None) -> list[dict]:
    """Every finding in `text`, each with its line and a redacted sample."""
    rules = RULES if rules is None else rules
    fm = FRONTMATTER.match(text)
    frontmatter_lines = fm.group(1).count("\n") + 2 if fm else 0
    findings = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for family, family_rules in rules.items():
            for rule_id, pattern, why, severity in family_rules:
                if rule_id.endswith("-frontmatter") and line_no > frontmatter_lines:
                    continue
                match = pattern.search(line)
                if match and not allowed(line, rule_id):
                    findings.append({"family": family, "rule": rule_id, "where": f"{where}:{line_no}",
                                     "why": why, "sample": redact(match.group(0)), "severity": severity})
    return findings


def blocking(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["severity"] == BLOCK]


def targets(paths: list[Path], staged: bool) -> list[Path]:
    if staged:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit("not a git repository, or git failed")
        files = [Path(n) for n in out.stdout.splitlines() if n]
    else:
        files = []
        for path in paths:
            if not path.exists():
                sys.exit(f"no such path: {path}")
            files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return [f for f in files if f.is_file() and f.suffix.lower() not in SKIP_SUFFIXES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--staged", action="store_true", help="scan what git is about to commit")
    args = parser.parse_args()
    if not args.paths and not args.staged:
        sys.exit("give paths to scan, or --staged")

    findings, scanned = [], 0
    for path in targets(args.paths, args.staged):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        findings += scan_text(text, str(path))

    for finding in sorted(findings, key=lambda f: f["severity"] != BLOCK):
        print(f"{finding['severity'].upper():5}  {finding['where']}  [{finding['rule']}] {finding['why']}\n"
              f"       matched: {finding['sample']}")
    stopped = blocking(findings)
    print(f"\nguard: {scanned} files, {len(stopped)} blocking, {len(findings) - len(stopped)} advisory")
    if stopped:
        print("Fix the source, or annotate the line `guard: allow <rule-id>` if it is a genuine false positive.")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())

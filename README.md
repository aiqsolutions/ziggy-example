# Ziggy — example code

> **Unsupported and unmaintained.** This repository is published once, as an illustration, and will
> not be updated. Issues, pull requests and questions will not be answered, and no fixes — including
> security fixes — will be made. It is provided "as is", without warranty of any kind, under the
> [MIT License](LICENSE). Use of it is entirely at your own risk.
>
> It is not part of any AIQ Solutions product, is not a medical device, and has not been validated for
> any purpose. Do not use it with patient data or any other regulated data. In particular, `guard.py`
> is an example of the idea, not a compliance control: pattern matching catches the obvious and misses
> the rest.

Simplified versions of the scripts behind Ziggy, the on-premises AI context system at AIQ Solutions,
written up in [Building Ziggy: how we got real value out of AI without handing over our data](https://www.aiq-solutions.com/building-ziggy-how-we-got-real-value-out-of-ai-without-handing-over-our-data/).

This example covers one path: pulling content out of a source system (Jira, here), keeping out what
must not go in, turning the rest into searchable passages that cite where they came from, measuring
whether search actually works, and handing those passages to an AI client such as Claude. It is not
the production code — everything specific to us is gone — and it runs end to end on a bundled set of
fictional tickets, so you can watch it work before deciding whether any of it is worth borrowing.

Python 3.9 or later, standard library plus PyYAML. No framework, no MCP SDK, no vector database, no
build step.

## The pipeline

| Step | Script | What it is for |
|---|---|---|
| Pull | `scripts/ingest_jira.py` | Jira issues and their full comment threads, as markdown with provenance frontmatter |
| Guard | `scripts/guard.py` | Refuses secrets, personal data and restricted content — called *inside* the ingest |
| Chunk and index | `scripts/corpus.py` | Splits documents into passages that cite `path:line`; keyword and embedding search |
| Evaluate | `scripts/eval_retrieval.py` | Scores search against questions whose right answer you already know |
| Serve | `scripts/mcp_server.py` | Hands the index to an MCP client — Claude Code, Claude Desktop — as two read-only tools |

`sources.yml` says what to pull and how to label it. Each script's docstring explains why it is
shaped the way it is, and the comments are most of the value.

## Try it

```bash
pip install pyyaml
python3 scripts/ingest_jira.py --sample sample/jira_issues.json --measure
python3 scripts/ingest_jira.py --sample sample/jira_issues.json
python3 scripts/corpus.py build --lexical-only
python3 scripts/corpus.py query "how do we ship a release to production"
python3 scripts/eval_retrieval.py --verbose
python3 -m unittest discover tests
```

The sample is sixteen tickets for a made-up webhook service, and most of them exist to show one
behaviour:

- **`--measure` first.** It lists the shortest ticket bodies. An empty story template and a "TBD"
  sit at 0 and 19 characters; the shortest ticket worth keeping, a one-line root cause, is over a
  hundred. `min_body: 60` in `sources.yml` sits in that gap. Choose the threshold from measurements,
  not a guess: too high, and it silently drops your most precise content.
- **The guard blocks two tickets.** One has a password pasted into a config snippet; the other is
  labelled `restricted` in Jira. Neither is written. A third mentions an internal IP address, which
  warns and goes through — a guard that blocks on everything gets switched off. Findings are
  redacted in the output, so the guard never copies the secret into a log.
- **Two tickets are from 2023**, and they describe how releases and retries *used* to work. They are
  indexed in the `archive` collection, which search leaves out unless asked for by name. The eval
  shows the filter is doing real work: without it, both old tickets would have answered current
  questions.
- **One question shares no words with its answer** ("HMAC key" vs. "signing secret"), so keyword
  search can't find it. That is the case an embedding model exists for — and whether it actually
  helps on *your* data is what the eval is for.

Every passage comes back with a `path:line` citation. That is the point of the whole thing: a reader
can check a passage in seconds, which they cannot do with a model's summary of it.

## Serving it to Claude

`scripts/mcp_server.py` is a minimal MCP server over the index, with two tools: `search` returns
passages with their citations, and `get_document` returns one whole ticket by id. Neither writes
anything. Once the index is built, register it with Claude Code from the repository root:

```bash
claude mcp add ziggy -- python3 "$(pwd)/scripts/mcp_server.py"
```

Then ask Claude something the sample tickets answer — how many times a failed delivery is retried,
say — and the answer should cite the ticket it came from. For Claude Desktop, add the same command to
`claude_desktop_config.json`, with absolute paths:

```json
{"mcpServers": {"ziggy": {"command": "python3", "args": ["/absolute/path/to/scripts/mcp_server.py"]}}}
```

Claude Desktop does not see your shell's `PATH`, so if it cannot start the server, give the absolute
path of the `python3` that has PyYAML installed.

The server speaks MCP over stdio: the client starts it as a subprocess, and nothing listens on the
network. That is why it has no authentication — the only caller is the person running the client.
Serving one index to a team means HTTP, and HTTP means authentication and deciding who may see what.
That is a bigger job than this example takes on.

## With an embedding model

Install [Ollama](https://ollama.com), then:

```bash
ollama pull nomic-embed-text
python3 scripts/corpus.py build
python3 scripts/eval_retrieval.py
```

The eval now scores keyword (BM25), vector, and the two fused, side by side. An embedding model turns
text into vectors for meaning-search; it does not write anything, and it is a different thing from
the chat model you might also run in Ollama. Keyword search stays in as the control: if embeddings
don't beat it on your own questions, they aren't earning their keep.

Ollama's API has no authentication. Keep it on localhost, or behind something that authenticates.

## With your own Jira

```bash
export JIRA_BASE_URL=https://your-site.atlassian.net
export JIRA_EMAIL=you@example.com
export JIRA_TOKEN=...                          # an API token with read access
python3 scripts/ingest_jira.py --measure       # writes nothing
python3 scripts/ingest_jira.py --dry-run
python3 scripts/ingest_jira.py
```

Edit `sources.yml` first: the JQL, the date before which tickets count as archive, and `min_body`
once you have measured. Then replace `eval/golden_queries.yaml` with questions people at your company
actually ask, phrased the way they ask them, each naming the ticket that answers it.

## Adding another source

An ingest script for anything else — Notion, Confluence, a folder of documents — only has to do two
things: write markdown with frontmatter to `corpus/<source>/`, and run every document through
`guard.scan_text` before writing it, so a blocked document cannot be written rather than merely
being checked afterwards. `corpus.py` picks up the new directory on the next build and tags every
passage with its source.

```yaml
---
title: "Release process"
id: "DOC-12"
source: "notion"
collection: "docs"        # "archive" keeps it out of default results
updated: "2026-01-15"
---
```

Then rebuild and re-run the eval, and compare. Adding a source can make search *worse* — a few
hundred documents sharing the same boilerplate headings will crowd out the ones people wanted — and
the eval is the only thing that will tell you.

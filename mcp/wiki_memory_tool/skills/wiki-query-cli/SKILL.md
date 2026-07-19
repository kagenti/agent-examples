# kwiki Query Agent (CLI)

Query the wiki memory service using the `wiki_cli.py` command-line tool.

## Prerequisites

The wiki service must be running on `http://localhost:8321`. Run from the `wiki_memory_tool/` directory.

## Procedure

### 1. List Topics

```bash
uv run python wiki_cli.py query list-topics
```

### 2. List Pages in a Topic

```bash
uv run python wiki_cli.py query list-pages {topic_id}
```

### 3. Search a Topic

```bash
uv run python wiki_cli.py query search {topic_id} "search terms"
```

Optional: `--limit N` to control result count.

### 4. Search All Topics (Global)

```bash
uv run python wiki_cli.py query search-all "search terms"
```

### 5. Read a Page

```bash
uv run python wiki_cli.py query read {topic_id} {path}
```

Returns page content with frontmatter metadata if present.

### 6. Activity Feed

```bash
uv run python wiki_cli.py query activity              # Global
uv run python wiki_cli.py query activity {topic_id}   # Topic-specific
```

### 7. Backlinks

```bash
uv run python wiki_cli.py query backlinks {topic_id} {path}
```

### 8. Tags

```bash
uv run python wiki_cli.py query tags {topic_id}              # List all tags
uv run python wiki_cli.py query tag {topic_id} {tag_name}    # Pages by tag
```

### 9. Page Graph

```bash
uv run python wiki_cli.py query graph {topic_id}
```

### 10. Drafts

```bash
uv run python wiki_cli.py query drafts {topic_id}
```

## Authentication

### GitHub Login (recommended for users)

```bash
uv run python wiki_cli.py --base-url https://wiki-service.example.com login
uv run python wiki_cli.py whoami
uv run python wiki_cli.py logout
```

Once logged in, the CLI uses your GitHub identity for all requests. Token is cached at `~/.wiki-memory/token.json`.

### SPIFFE Headers (agent mode)

Without a cached token, the CLI uses simulated Query Agent SPIFFE headers with OBO user.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8321` | Service URL |
| `--user` | `alice@example.com` | User identity for OBO |
| `--trust-domain` | `rossoctl.example.com` | SPIFFE trust domain |

## Example Flow

```bash
uv run python wiki_cli.py query list-topics
uv run python wiki_cli.py query search ai "attention mechanism"
uv run python wiki_cli.py query search-all "transformer architecture"
uv run python wiki_cli.py query read ai transformers.md
uv run python wiki_cli.py query activity ai
uv run python wiki_cli.py query backlinks ai transformers.md
uv run python wiki_cli.py query tags ai
uv run python wiki_cli.py query tag ai paper
uv run python wiki_cli.py query graph ai
uv run python wiki_cli.py query drafts ai
```

## Notes

- Authentication is handled automatically (Query Agent SPIFFE + user OBO headers)
- Change the user with `--user bob@example.com`
- Returns exit code 1 on errors (403 forbidden, 404 not found)

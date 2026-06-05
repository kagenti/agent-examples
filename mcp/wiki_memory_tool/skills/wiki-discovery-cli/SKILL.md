# kwiki Discovery Agent (CLI)

Write new knowledge to the wiki memory service using the `wiki_cli.py` command-line tool.

## Prerequisites

The wiki service must be running on `http://localhost:8321`. Run from the `wiki_memory_tool/` directory.

## Procedure

### 1. Check Novelty First

```bash
uv run python wiki_cli.py discover novelty {topic_id} "Page Title" "Brief summary of content"
```

Output: `NOVEL: ...` or `NOT NOVEL: ...` with similar pages listed.

If NOT NOVEL, do not write.

### 2. Get a Template (optional)

```bash
uv run python wiki_cli.py discover template                  # List all
uv run python wiki_cli.py discover template paper-summary    # Get specific
```

Available templates: `paper-summary`, `concept-overview`, `how-to-guide`, `comparison`.

### 3. Write a Page

```bash
uv run python wiki_cli.py discover write {topic_id} {path} --content "# Title\n\nContent"
uv run python wiki_cli.py discover write {topic_id} {path} --file content.md
```

Options:
- `--message "commit message"` — customize git commit message
- `--draft` — submit as draft for review instead of publishing immediately

On success, the CLI shows suggested links to related pages.

### 4. Submit as Draft (optional)

```bash
uv run python wiki_cli.py discover write {topic_id} {path} --draft --content "..."
```

Drafts are stored pending review. An admin can approve or reject them.

## Authentication

### GitHub Login (recommended)

```bash
uv run python wiki_cli.py --base-url https://wiki-service.example.com login
uv run python wiki_cli.py whoami
```

### SPIFFE Headers (agent mode)

Without a cached token, the CLI uses simulated SPIFFE headers.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8321` | Service URL |
| `--topic` | `ai` | Default topic (used for SPIFFE ID) |
| `--trust-domain` | `kagenti.example.com` | SPIFFE trust domain |

## Example Flow

```bash
# Get template
uv run python wiki_cli.py discover template paper-summary

# Check novelty
uv run python wiki_cli.py discover novelty ai "LoRA" "Low-rank adaptation for fine-tuning"

# Write if novel
uv run python wiki_cli.py discover write ai lora.md --content "# LoRA\n\n..."

# Write as draft for review
uv run python wiki_cli.py discover write ai experimental.md --draft --file ./notes/draft.md
```

## Notes

- Returns exit code 1 on errors (403 if agent lacks write access to topic)
- Content is committed to git and pushed to remote if configured
- Suggested links are shown after writing to help connect related content

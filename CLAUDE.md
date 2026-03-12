# CLAUDE.md - Agent Examples

This file provides guidance for AI assistants working with this repository.

## Commit Attribution Policy

When creating git commits, do NOT use `Co-Authored-By` trailers for AI attribution.
Instead, use `Assisted-By` to acknowledge AI assistance without inflating contributor stats:

    Assisted-By: Claude (Anthropic AI) <noreply@anthropic.com>

Never add `Co-authored-by`, `Made-with`, or similar trailers that GitHub parses as co-authorship.

A `commit-msg` hook in `scripts/hooks/commit-msg` enforces this automatically.
Developers can install it by running:

```sh
git config core.hooksPath scripts/hooks
```

## Testing

Tests use `pytest` and live in the `tests/` directory:

```sh
# Run all tests
python -m pytest tests/ -v

# Run A2A agent tests only
python -m pytest tests/a2a/ -v

# Run MCP tool tests only
python -m pytest tests/mcp/ -v
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

Tests mock heavy dependencies (langchain, opentelemetry, fastmcp) to run
without installing agent-specific packages.

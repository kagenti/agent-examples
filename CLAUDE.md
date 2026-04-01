# CLAUDE.md - Agent Examples

This file provides guidance for AI assistants working with this repository.

## Overview

Community examples of AI agents (A2A protocol) and MCP tools for the Kagenti platform.
Each agent/tool is a self-contained Python project with its own `pyproject.toml` and `Dockerfile`.

## Repository Structure

```
agent-examples/
├── a2a/                        # A2A protocol agents
│   ├── a2a_contact_extractor/  # Contact extraction agent
│   ├── a2a_currency_converter/ # Currency conversion agent
│   ├── cheerup_agent/          # Cheerup agent
│   ├── file_organizer/         # File organizer agent
│   ├── generic_agent/          # Generic agent template
│   ├── git_issue_agent/        # GitHub issue agent
│   ├── image_service/          # Image generation agent
│   ├── recipe_agent/           # Recipe agent
│   ├── reservation_service/    # Reservation agent
│   ├── simple_generalist/      # Simple generalist agent
│   ├── slack_researcher/       # Slack research agent
│   ├── trivia_agent/           # Trivia agent
│   └── weather_service/        # Weather agent
├── mcp/                        # MCP protocol tools
│   ├── cloud_storage_tool/     # Cloud storage tool
│   ├── flight_tool/            # Flight search tool
│   ├── github_tool/            # GitHub tool
│   ├── image_tool/             # Image tool
│   ├── movie_tool/             # Movie tool
│   ├── reservation_tool/       # Reservation tool
│   ├── shopping_tool/          # Shopping tool
│   ├── slack_tool/             # Slack tool
│   └── weather_tool/           # Weather tool
├── scripts/hooks/              # Git hooks
├── sample-environments.yaml    # K8s environment configs
└── pyproject.toml              # Root ruff config
```

## Key Commands

| Task | Command |
|------|---------|
| Lint | `make lint` |
| Format | `make fmt` |
| Pre-commit install | `pre-commit install` |

## Code Style

- Python 3.11+ across all agents/tools
- Linting: `ruff` (config in root `pyproject.toml`)
- Pre-commit hooks: `pre-commit install`
- Each agent/tool has its own `pyproject.toml` for dependencies

## DCO Sign-Off (Mandatory)

All commits must include a `Signed-off-by` trailer:

```sh
git commit -s -m "feat: add new agent"
```

## Commit Attribution Policy

When creating git commits, do NOT use `Co-Authored-By` trailers for AI attribution.
Instead, use `Assisted-By` to acknowledge AI assistance without inflating contributor stats:

    Assisted-By: Claude (Anthropic AI) <noreply@anthropic.com>

Never add `Co-authored-by`, `Made-with`, or similar trailers that GitHub parses as co-authorship.

A `commit-msg` hook in `scripts/hooks/commit-msg` enforces this automatically.
Install it via:

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

## Adding a New Agent

1. Create a directory under `a2a/` or `mcp/`
2. Add a `pyproject.toml` with dependencies
3. Add a `Dockerfile` for container builds
4. Add environment config to `sample-environments.yaml` if needed

## Orchestration

This repo includes orchestrate skills for enhancing related repos:

| Skill | Description |
|-------|-------------|
| `orchestrate` | Run `/orchestrate <repo-path>` to start |
| `orchestrate:scan` | Assess repo structure and gaps |
| `orchestrate:plan` | Create phased enhancement plan |
| `orchestrate:precommit` | Add pre-commit hooks and linting |
| `orchestrate:tests` | Add test infrastructure |
| `orchestrate:ci` | Add CI workflows and security scanning |
| `orchestrate:security` | Add security governance files |
| `orchestrate:replicate` | Bootstrap skills into target |
| `orchestrate:review` | Review all orchestration PRs |

Skills management:

| Skill | Description |
|-------|-------------|
| `skills` | Skill management router |
| `skills:scan` | Audit and bootstrap skills |
| `skills:write` | Create or edit skills |
| `skills:validate` | Validate skill format |

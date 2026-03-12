---
name: skills
description: Skill management - create, validate, and improve Claude Code skills
---

```mermaid
flowchart TD
    START(["/skills"]) --> NEED{"What do you need?"}
    NEED -->|New skill| WRITE["skills:write"]:::skills
    NEED -->|Edit skill| WRITE
    NEED -->|Audit all| SCAN["skills:scan"]:::skills

    WRITE --> VALIDATE["skills:validate"]:::skills
    VALIDATE -->|Issues| WRITE
    VALIDATE -->|Pass| PR["Create PR"]:::git

    SCAN -->|Gaps found| WRITE

    classDef skills fill:#607D8B,stroke:#333,color:white
    classDef git fill:#FF9800,stroke:#333,color:white
```

> Follow this diagram as the workflow.

# Skills Management

Skills for managing the skill system itself.

## Worktree-First Gate

Before creating or editing any skill, consider creating a worktree:

```bash
git worktree add .worktrees/skills-<topic> -b docs/skills-<topic> main
```

Then work in the worktree, validate, and create a PR.

## Available Skills

| Skill | Purpose |
|-------|---------|
| `skills:write` | Create new skills or edit existing ones following the standard template |
| `skills:validate` | Validate skill format, naming, and structure |
| `skills:scan` | Audit repository skills — gaps, quality, connections, diagrams |

## Related Skills

- `orchestrate` — Orchestrate other repos using skills
- `orchestrate:replicate` — Bootstrap skills into target repos

# kwiki Discovery Agent (MCP)

Write new knowledge to the wiki memory service after checking for novelty.

## Prerequisites

The wiki-memory MCP server must be registered and running (stdio or streamable-http on port 8322).

## Procedure

### 1. Get a Template (optional)

Call `wiki_get_template` to get a structured starting point:
- No args: list available templates
- `template_id`: get specific template (paper-summary, concept-overview, how-to-guide, comparison)

### 2. Check Novelty First

Call `wiki_check_novelty` with:
- `topic_id`: target topic (e.g. "ai")
- `title`: title of the new content
- `abstract`: brief summary or first paragraph

If `"novel": false`, do NOT write.

### 3. Write the Page

Call `wiki_write` with:
- `topic_id`: target topic
- `path`: filename (e.g. "transformers.md")
- `content`: full markdown content (use frontmatter with tags)
- `message`: commit message (optional)
- `draft`: set to `true` to submit for review instead of publishing

Returns suggested links to related pages.

### 4. Submit as Draft (optional)

Call `wiki_write` with `draft=True` to submit for review:
```
wiki_write(topic_id="ai", path="draft.md", content="...", draft=True)
→ "Draft: ai/draft.md"
```

### 5. Approve/Reject Drafts (admin)

- `wiki_list_drafts(topic_id)` — list pending drafts
- `wiki_approve_draft(topic_id, path)` — approve and publish

## Example Flow

```
wiki_get_template(template_id="paper-summary") → template markdown

wiki_check_novelty(topic_id="ai", title="LoRA", abstract="Low-rank adaptation...") 
  → {"novel": true}

wiki_write(topic_id="ai", path="lora.md", content="---\ntags: [paper, fine-tuning]\n---\n# LoRA\n\n...")
  → "Written: ai/lora.md\nSuggested links:\n- ai/fine-tuning.md (score=0.08)"
```

## Content Guidelines

- Use YAML frontmatter with tags: `---\ntags: [paper, topic]\n---`
- Include a top-level `# Title` heading
- Use `## Sections` for organization
- Link to related pages with `[[page-name]]` or `[text](page.md)`
- Keep pages focused on one concept

## Notes

- **Local mode** (default): No authentication needed
- **Remote mode** (`WIKI_SERVICE_URL` set): Uses cached GitHub token
- Always check novelty to avoid duplicates
- Use drafts when content needs human review before publishing

# kwiki Query Agent (MCP)

Query the wiki memory service to find and read information on behalf of a user.

## Prerequisites

The wiki-memory MCP server must be registered and running (stdio or streamable-http on port 8322).

## Procedure

1. **List available topics** — Call `wiki_list_topics` to see what topics exist and their page counts.

2. **Search for information** — Call `wiki_query` with:
   - `topic_id`: the topic to search (e.g. "ai", "security")
   - `query`: natural language search terms
   - `limit`: max results (default 10)

3. **Search across all topics** — Call `wiki_search_all` with:
   - `query`: search terms
   - `limit`: max results (default 10)

4. **Read specific pages** — Call `wiki_read` with:
   - `topic_id`: the topic
   - `path`: page filename (e.g. "transformers.md")
   Returns content with frontmatter metadata.

5. **Activity feed** — Call `wiki_activity` with:
   - `topic_id`: (optional) specific topic, or empty for global
   - `limit`: max entries (default 20)

6. **Find backlinks** — Call `wiki_backlinks` with:
   - `topic_id`: the topic
   - `path`: page to find references to

7. **Browse by tags** — Call `wiki_list_tags` with `topic_id`, or `wiki_pages_by_tag` with `topic_id` and `tag`.

8. **Page graph** — Call `wiki_graph` with `topic_id` to get nodes and edges.

9. **List drafts** — Call `wiki_list_drafts` with `topic_id`.

10. **Get templates** — Call `wiki_get_template` with optional `template_id`.

## Example Flow

```
wiki_list_topics → "ai (4 pages), security (0 pages)"
wiki_query(topic_id="ai", query="attention mechanism") → ranked results
wiki_search_all(query="transformer") → results across all topics
wiki_read(topic_id="ai", path="transformers.md") → content + frontmatter
wiki_activity(topic_id="ai") → recent commits
wiki_backlinks(topic_id="ai", path="transformers.md") → pages linking here
wiki_list_tags(topic_id="ai") → tags with counts
wiki_pages_by_tag(topic_id="ai", tag="paper") → pages with tag
wiki_graph(topic_id="ai") → {"nodes": [...], "edges": [...]}
wiki_list_drafts(topic_id="ai") → pending drafts
wiki_get_template(template_id="paper-summary") → template content
```

## Notes

- **Local mode** (default): No authentication needed — MCP is a trusted local channel
- **Remote mode** (`WIKI_SERVICE_URL` set): Uses cached GitHub token from `~/.wiki-memory/token.json`
- Search uses TF-IDF ranking over markdown content

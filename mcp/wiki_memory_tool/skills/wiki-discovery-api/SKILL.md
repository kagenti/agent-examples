# kwiki Discovery Agent (REST API)

Write new knowledge to the wiki memory service via REST API after checking novelty.

## Prerequisites

The wiki service must be running on `http://localhost:8321`.

## Authentication

### Option A: GitHub OAuth Token

```
Authorization: Bearer <wiki-jwt-token>
```

### Option B: SPIFFE Headers (agent-to-agent)

```
X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-{topic_id}/sa/discovery-agent
```

## Procedure

### 1. Get a Template (optional)

```bash
curl -s http://localhost:8321/templates
curl -s http://localhost:8321/templates/paper-summary
```

### 2. Check Novelty First

```bash
curl -s -X POST http://localhost:8321/topics/{topic_id}/check-novelty \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-{topic_id}/sa/discovery-agent" \
  -H "Content-Type: application/json" \
  -d '{"title": "Page Title", "abstract": "Brief summary"}'
```

If `"novel": false`, do NOT write.

### 3. Write the Page

```bash
curl -s -X POST http://localhost:8321/topics/{topic_id}/pages/{path} \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-{topic_id}/sa/discovery-agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "---\ntags: [paper]\n---\n# Title\n\nContent...", "message": "commit message"}'
```

Response includes `suggested_links` to related pages.

### 4. Write as Draft

```bash
curl -s -X POST "http://localhost:8321/topics/{topic_id}/pages/{path}?draft=true" \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-{topic_id}/sa/discovery-agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "# Draft\n\nPending review..."}'
```

### 5. Approve/Reject Drafts (admin)

```bash
# List drafts
curl -s http://localhost:8321/topics/{topic_id}/drafts \
  -H "Authorization: Bearer <admin-jwt>"

# Approve
curl -s -X POST http://localhost:8321/topics/{topic_id}/drafts/{path}/approve \
  -H "Authorization: Bearer <admin-jwt>"

# Reject
curl -s -X POST http://localhost:8321/topics/{topic_id}/drafts/{path}/reject \
  -H "Authorization: Bearer <admin-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"reason": "needs more references"}'
```

## Example Flow

```bash
# Get template
curl -s http://localhost:8321/templates/paper-summary

# Check novelty
curl -s -X POST http://localhost:8321/topics/ai/check-novelty \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-ai/sa/discovery-agent" \
  -H "Content-Type: application/json" \
  -d '{"title": "LoRA", "abstract": "Low-rank adaptation for efficient fine-tuning"}'

# Write if novel (with tags)
curl -s -X POST http://localhost:8321/topics/ai/pages/lora.md \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-ai/sa/discovery-agent" \
  -H "Content-Type: application/json" \
  -d '{"content": "---\ntags: [paper, fine-tuning]\n---\n# LoRA\n\nLow-Rank Adaptation...", "message": "Add LoRA overview"}'
```

## Notes

- The Discovery Agent SPIFFE ID must be in the topic's `writers` list
- Writing to a different topic returns 403
- Response includes `suggested_links` for related content
- Use `?draft=true` to submit for review instead of publishing directly
- Use YAML frontmatter with tags for better discoverability

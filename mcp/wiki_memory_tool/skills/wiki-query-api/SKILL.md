# kwiki Query Agent (REST API)

Query the wiki memory service via REST API on behalf of a user.

## Prerequisites

The wiki service must be running on `http://localhost:8321`.

## Authentication

### Option A: GitHub OAuth Token

```
Authorization: Bearer <wiki-jwt-token>
```

### Option B: SPIFFE Headers (agent OBO user)

```
X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent
X-Original-Subject: alice@example.com
```

## Procedure

### 1. List Topics

```bash
curl -s http://localhost:8321/topics \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 2. Search a Topic

```bash
curl -s -X POST http://localhost:8321/topics/{topic_id}/query \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com" \
  -H "Content-Type: application/json" \
  -d '{"query": "search terms", "limit": 10}'
```

### 3. Global Search (across all topics)

```bash
curl -s -X POST http://localhost:8321/search \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com" \
  -H "Content-Type: application/json" \
  -d '{"query": "search terms", "limit": 10}'
```

### 4. Read a Page

```bash
curl -s http://localhost:8321/topics/{topic_id}/pages/{path} \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 5. Activity Feed

```bash
# Global
curl -s http://localhost:8321/activity \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"

# Topic-specific
curl -s "http://localhost:8321/topics/{topic_id}/activity?limit=10" \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 6. Backlinks

```bash
curl -s http://localhost:8321/topics/{topic_id}/backlinks/{path} \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 7. Tags

```bash
# List tags
curl -s http://localhost:8321/topics/{topic_id}/tags \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"

# Pages by tag
curl -s http://localhost:8321/topics/{topic_id}/tags/{tag} \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 8. Graph

```bash
curl -s http://localhost:8321/topics/{topic_id}/graph \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent" \
  -H "X-Original-Subject: alice@example.com"
```

### 9. Templates

```bash
curl -s http://localhost:8321/templates
curl -s http://localhost:8321/templates/paper-summary
```

### 10. Drafts

```bash
curl -s http://localhost:8321/topics/{topic_id}/drafts \
  -H "X-Spiffe-Id: spiffe://rossoctl.example.com/ns/topic-ai/sa/discovery-agent"
```

## Notes

- The Query Agent SPIFFE ID must be in the topic's `readers` list in the ACL
- Wildcard `*` in readers allows all authenticated users
- Returns 403 if identity lacks access

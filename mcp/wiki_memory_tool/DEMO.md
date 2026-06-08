# Wiki Memory Tool Demo

A quick walkthrough of the wiki CLI — login, query, discover, and verify.

## 1. Login

```bash
$ kwiki login
==================================================
  GitHub Device Authorization
==================================================

  1. Open this URL in your browser:

     https://github.com/login/device

  2. Enter this code:

     ABCD-1234

  Code expires in 15 minutes.
==================================================

Waiting for authorization.....

Logged in as aslom
Groups: kaslomorg/ml-writers, kaslomorg/platform-admins
```

## 2. Check Identity

```bash
$ kwiki whoami
User:   aslom
Status: valid (expires in 6d 23h)
Groups: kaslomorg/ml-writers, kaslomorg/platform-admins
Server: https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com

Access:
  ai: read, write, admin
    read  <- *
    write <- github:team:kaslomorg/ml-writers
    admin <- github:user:aslom
  security: read, write
    read  <- github:org:kaslomorg
    write <- github:team:kaslomorg/platform-admins
```

## 3. Query Existing Pages

```bash
$ kwiki query list-topics
  ai (7 pages)
  security (0 pages)
  ml (0 pages)

$ kwiki query list-pages ai
  evaluation.md
  fine-tuning.md
  rag-patterns.md
  transformers.md

$ kwiki query search ai "attention mechanism"
  ai/transformers.md (score=0.1652)
    Self-attention mechanisms for parallel sequence processing.

$ kwiki query read ai transformers.md
---
tags: [paper, architecture]
---
# Transformer Architecture

Self-attention mechanisms for parallel sequence processing.
...
```

## 4. Discover — Write a New Page

```bash
$ kwiki discover novelty ai "Mixture of Experts" "Sparse gating for efficient model scaling"
NOVEL: No sufficiently similar content found

$ kwiki discover write ai mixture-of-experts.md --content "$(cat <<'EOF'
---
tags: [paper, architecture, scaling]
---
# Mixture of Experts

Source: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)

## Summary

Mixture of Experts routes each token to a subset of specialist sub-networks,
enabling model scaling without proportional compute increase.

## Key Ideas

- Sparse gating: each token activates only top-k experts
- Load balancing loss prevents routing collapse
- Capacity factor limits tokens per expert

## Results

- Switch Transformer: 7x speedup over T5-Base at same compute
- Mixtral 8x7B: competitive with GPT-3.5 using 2 active experts per token
EOF
)"
Written: ai/mixture-of-experts.md by discovery-agent
Suggested links:
  ai/transformers.md (score=0.0923)
  ai/fine-tuning.md (score=0.0412)
```

## 5. Verify It Was Added

```bash
$ kwiki query search ai "mixture of experts sparse gating"
  ai/mixture-of-experts.md (score=0.2841)
    Mixture of Experts routes each token to a subset of specialist sub-networks.

$ kwiki query read ai mixture-of-experts.md
---
tags: [paper, architecture, scaling]
---
# Mixture of Experts

Source: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
...

$ kwiki query tags ai
  paper: 3 pages
  architecture: 2 pages
  scaling: 1 pages
  retrieval: 1 pages
  technique: 1 pages

$ kwiki query backlinks ai mixture-of-experts.md
  (none yet)
```

## 6. Optional — Initialize GitHub Pages

```bash
$ kwiki admin init-pages
GitHub Pages initialized (6 files):
  _config.yml
  index.md
  _layouts/default.html
  _layouts/page.html
  _includes/nav.html
  assets/css/style.css
```

After ~60 seconds, browse: https://kaslom.github.io/kagenti-wiki-research/

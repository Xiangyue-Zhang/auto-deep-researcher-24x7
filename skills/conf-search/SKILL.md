---
name: conf-search
description: "Search papers from top AI/ML conferences"
---

# conf-search

Search for papers from top venues.

## Usage

```
Claude Code: /conf-search --venue CVPR2025 --query "gesture generation"
Claude Code: /conf-search --venue NeurIPS2025 --query "diffusion models"
Codex: $conf-search
```

## Supported Venues

CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, IJCAI, ACL, EMNLP, NAACL, SIGGRAPH, and more.

## Behavior

1. Parse venue and query from user input
2. Search via Semantic Scholar API with venue filter
3. Sort by relevance and citation count
4. Present results with title, authors, abstract snippet, citation count
5. Offer to do deep analysis on any paper of interest

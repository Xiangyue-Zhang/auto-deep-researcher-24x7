---
name: daily-papers
description: "Daily arXiv paper recommendations with automatic deduplication"
---

# daily-papers

Search arXiv for the latest papers relevant to the user's research interests.

Invoke as `/daily-papers` in Claude Code or `$daily-papers` in Codex.

## Behavior

1. Ask the user for topics if not provided (or use defaults from config)
2. Search arXiv for papers published in the last 1-3 days
3. Check against previously recommended papers (dedup)
4. Rank by relevance to the user's research
5. For top 5: provide detailed analysis (motivation, method, key results, insights)
6. For next 5: provide brief summaries
7. Save recommendations to a dated markdown file

## Deduplication

Maintain a list of previously recommended paper IDs. Never recommend the same paper twice.

## Output Format

```markdown
# Daily Paper Recommendations — YYYY-MM-DD

## Top Picks (Detailed)

### 1. [Paper Title](arxiv_url)
**Authors**: ...
**Relevance**: Why this matters for your research
**Motivation**: What problem they solve
**Method**: Key technical approach
**Results**: Main findings
**Insight**: What you can learn from this

### 2. ...

## Also Worth Reading

### 6. [Paper Title](arxiv_url) — One-line summary
...
```

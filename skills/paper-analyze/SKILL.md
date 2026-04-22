---
name: paper-analyze
description: "Deep analysis of a single paper with figure extraction from arXiv source"
---

# paper-analyze

Perform deep analysis of a single academic paper.

## Usage

```
Claude Code: /paper-analyze <arxiv_id or url>
Codex: $paper-analyze
```

## Behavior

1. Fetch paper metadata from arXiv API
2. Attempt to download arXiv source package (.tar.gz)
3. Extract actual figures from source (not screenshots)
4. Read the full paper (PDF if source unavailable)
5. Generate structured analysis

## Figure Extraction

Priority order:
1. arXiv source package → extract .png/.jpg/.pdf figures
2. PDF extraction as fallback
3. Name files with arxiv_id prefix to avoid collisions

## Output Format

```markdown
# [Paper Title]

**arXiv**: [id] | **Authors**: ... | **Year**: ...

## Problem
What specific problem does this paper address?

## Motivation  
Why is this problem important? What gap exists?

## Method
Detailed technical approach with key equations/algorithms.

![Figure 1](figures/arxiv_id_fig1.png)

## Experiments
- Datasets, baselines, metrics
- Key results table
- Ablation findings

## Insights
- What can we learn and apply?
- Strengths and limitations
- Connections to our research
```

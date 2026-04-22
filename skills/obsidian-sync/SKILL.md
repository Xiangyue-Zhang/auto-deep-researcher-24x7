---
name: obsidian-sync
description: "Refresh Obsidian dashboard and daily notes from current experiment state"
---

# obsidian-sync

Refresh progress notes for a Deep Researcher project.

## Usage

```bash
Claude Code: /obsidian-sync --project /path/to/project
Claude Code: /obsidian-sync --project /path/to/project --dashboard-only
Claude Code: /obsidian-sync --project /path/to/project --daily-only
Codex: $obsidian-sync
```

## Behavior

1. Read project config and check `obsidian.enabled`
2. Read `PROJECT_BRIEF.md`, `workspace/MEMORY_LOG.md`, `workspace/state.json`, and `.cycle_counter`
3. Refresh `Dashboard.md` in Obsidian, or `workspace/progress_tracking/Dashboard.txt` if no vault is configured
4. Optionally append a new daily note entry

## Command

```bash
python -m core.obsidian --project /path/to/project
```

If progress export is disabled, tell the user to set `obsidian.enabled: true`. If `obsidian.vault_path` is empty, notes fall back to project-local text files.

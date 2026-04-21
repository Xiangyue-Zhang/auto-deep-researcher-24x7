---
name: auto-experiment
description: "Launch an autonomous THINK→EXECUTE→REFLECT experiment loop on a GPU project"
argument-hint: "[--project <path>] [--gpu <id>] [--max-cycles <n>]"
---

# /auto-experiment

Launch an autonomous experiment agent that runs your deep learning experiments 24/7.

## What This Does

This skill starts a **THINK → EXECUTE → REFLECT** loop that:
1. Reads your `PROJECT_BRIEF.md` to understand the research goal
2. Analyzes previous results in `MEMORY_LOG.md`
3. Plans the next experiment (hypothesis + success criteria)
4. Implements code changes and runs a **mandatory dry-run**
5. Launches GPU training via `nohup` (tracks PID)
6. **Monitors at zero LLM cost** (only `kill -0 PID` + `tail log` + `nvidia-smi`)
7. Wakes up when training finishes to analyze results
8. Updates memory and decides: iterate, pivot, or report
9. Repeats

## Usage

```
/auto-experiment
/auto-experiment --project /path/to/my_project --gpu 0
/auto-experiment --project . --max-cycles 5
```

## Prerequisites

The project directory must contain:

### `PROJECT_BRIEF.md` (required)
A frozen reference describing your research goal. Example:

```markdown
# Goal
Train a ViT-B/16 on ImageNet to reach 78%+ top-1 accuracy.

# Codebase
- Training: train.py
- Config: configs/vit_base.yaml
- Data: /data/imagenet/

# Constraints
- GPU 0-3 available (use DDP)
- Max 90 epochs per run
- Report val accuracy after each run

# Current Best
- ResNet-50 baseline: 76.1%
```

### `config.yaml` (optional)
Override default agent settings:

```yaml
agent:
  model: "claude-sonnet-4-6"
  max_cycles: -1          # -1 = unlimited
  max_steps_per_cycle: 3  # max sub-agent dispatches per cycle
  cooldown_interval: 300  # 5 min smart polling

memory:
  brief_max_chars: 3000
  log_max_chars: 2000

monitor:
  poll_interval: 900      # check every 15 min during training
  zero_llm: true

experiment:
  mandatory_dry_run: true
```

Optional remote execution over SSH:

```yaml
execution:
  mode: "ssh"
  ssh_host: "user@server"
  remote_workspace: "/home/user/my_project/workspace"
  remote_python: "python3"
```

In SSH mode, the controller state stays local (`PROJECT_BRIEF.md`,
`workspace/MEMORY_LOG.md`, `workspace/HUMAN_DIRECTIVE.md`, `state.json`),
while code edits, shell commands, training, log tailing, PID checks, and GPU
queries run on the configured remote host.

## Workflow Details

### Phase 1: THINK
- Read `PROJECT_BRIEF.md` (frozen, max 3000 chars)
- Read `MEMORY_LOG.md` (rolling, auto-compacted)
- Check for `HUMAN_DIRECTIVE.md` (highest priority, auto-archived after reading)
- Analyze: What's the current best? What hasn't been tried? What's most promising?
- Output: experiment plan with hypothesis and success criteria

### Phase 2: EXECUTE
- Dispatch to Code Agent (5 tools: `run_shell`, `launch_experiment`, `write_file`, `read_file`, `list_files`)
- Code Agent implements changes
- **Mandatory dry-run** (2-step verify, abort if fails)
- Launch training via `nohup`, capture PID
- Enter zero-cost monitoring loop:
  - backend PID check — is process alive?
  - backend `nvidia-smi` — GPU utilization
  - backend `tail -50 logfile` — latest training output
  - **Zero LLM API calls during this phase**

### Phase 3: REFLECT
- Parse training logs for metrics (loss, accuracy, FGD, FID, etc.)
- Compare against previous best
- Log milestone if improved (auto-compacted at 1200 chars)
- Log decision (rolling last 15 entries)
- Decide: try another config / pivot direction / generate report

### Human Override (anytime)
```bash
# Drop a directive file — agent reads it next cycle with highest priority
echo "Try learning rate 1e-5 with cosine schedule" > workspace/HUMAN_DIRECTIVE.md
```

## Memory System

Two-Tier, constant size (~5K chars / ~1500 tokens), no matter how long the agent runs:

| Tier | File | Content | Cap |
|------|------|---------|-----|
| 1 | `PROJECT_BRIEF.md` | Frozen project reference | 3,000 chars |
| 2 | `MEMORY_LOG.md` | Key Results + Recent Decisions | 2,000 chars |

**Auto-compaction rules:**
- Key Results: oldest dropped when section > 1,200 chars
- Recent Decisions: only last 15 entries kept
- Total log hard-capped at 2,000 chars

## Cost

| Phase | Duration | LLM Cost |
|-------|----------|----------|
| THINK | 5-10 min | ~$0.05 |
| EXECUTE (training) | hours/days | **$0.00** |
| REFLECT | 5-10 min | ~$0.03 |
| **24h cycle total** | | **~$0.08** |

## Example Output

After a few cycles, your `workspace/MEMORY_LOG.md` will look like:

```markdown
# Memory Log

## Key Results
[04-07 14:30] Exp001: ResNet-50 baseline, lr=0.1, acc=76.1%
[04-07 22:15] Exp002: ViT-B/16, lr=1e-3, acc=74.8% (underperforming, lr too high)
[04-08 06:00] Exp003: ViT-B/16, lr=3e-4 + cosine, acc=77.9% (new best!)
[04-08 14:45] Exp004: ViT-B/16, lr=3e-4 + cosine + mixup, acc=78.3% (target reached!)

## Recent Decisions
[04-07 14:30] Start with ResNet-50 baseline to establish reference
[04-07 22:15] ViT lr=1e-3 too high, try 3e-4 next
[04-08 06:00] Cosine schedule helped significantly, try adding regularization
[04-08 14:45] Target reached! Generate final report.
```

---
name: experiment-status
description: "Check status of running autonomous experiment loops"
---

# /experiment-status

Check the current status of your autonomous experiment agent.

## Usage

```
/experiment-status
/experiment-status --project /path/to/project
```

## Behavior

1. Read `PROJECT_BRIEF.md` — show the research goal
2. Read `MEMORY_LOG.md` — show key results and recent decisions  
3. Read `.cycle_counter` — show how many cycles completed
4. Check for running training processes via the configured execution backend
5. If training is running, tail the log file for latest output
6. Show GPU utilization through the configured backend
7. Check if `HUMAN_DIRECTIVE.md` exists (pending directive)

If `execution.mode=ssh`, controller state still comes from the local project
directory, but PID checks, training logs, and GPU status come from the
configured remote host.

## Output Format

```markdown
# Experiment Status — my-project

## Goal
Train ViT-B/16 on ImageNet to 78%+ accuracy

## Progress
- Cycles completed: 4
- Current best: 78.3% (Exp004, ViT-B/16 + cosine + mixup)
- Status: TRAINING (PID 12345, GPU 0, running 3.2h)

## Latest Training Log
Epoch 45/90 | loss: 2.134 | acc: 77.1% | lr: 1.2e-4

## Recent Decisions
1. [04-08 14:45] Target reached with mixup, trying stronger augmentation
2. [04-08 06:00] Cosine schedule helped, adding regularization

## Pending Directive
None (drop a file at workspace/HUMAN_DIRECTIVE.md to intervene)
```

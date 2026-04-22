---
name: gpu-monitor
description: "Check GPU status, running experiments, and available resources"
---

# gpu-monitor

Quick GPU status check for experiment management.

## Usage

```
Claude Code: /gpu-monitor
Claude Code: /gpu-monitor --server user@remote-host
Codex: $gpu-monitor
```

## Behavior

1. Run `nvidia-smi` to get current GPU status
2. Display a clean summary table:
   - GPU ID, Name, Memory (used/total), Utilization %, Temperature
   - Running processes on each GPU
3. Identify which GPUs are free (< 1GB memory used)
4. Identify which GPUs are running experiments (check for python/torchrun processes)
5. If `--server` is provided, SSH to remote server first

## Output Format

```
GPU Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GPU  Name          Memory         Util  Temp
  0   L20X 144GB    45123/147456   98%   72°C  ← training (PID 12345)
  1   L20X 144GB      234/147456    0%   35°C  ← FREE
  2   L20X 144GB    43210/147456   95%   70°C  ← training (PID 12346)
  3   L20X 144GB     1024/147456   12%   40°C  ← keeper

Free GPUs: [1]
Training: GPU 0 (PID 12345), GPU 2 (PID 12346)
```

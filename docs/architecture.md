# Architecture

> Detailed architecture documentation for Deep Researcher Agent.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep Researcher Agent                      │
│                                                              │
│  ┌─────────────┐                                             │
│  │ config.yaml │──→ Configuration for all components         │
│  └─────────────┘                                             │
│                                                              │
│  ┌──────────── Core Loop (loop.py) ────────────────────┐     │
│  │                                                      │     │
│  │  ┌───────┐    ┌─────────┐    ┌─────────┐           │     │
│  │  │ THINK │───→│ EXECUTE │───→│ REFLECT │──→ repeat  │     │
│  │  └───┬───┘    └────┬────┘    └────┬────┘           │     │
│  │      │             │              │                 │     │
│  │      ↓             ↓              ↓                 │     │
│  │  ┌───────────────────────────────────────┐          │     │
│  │  │        Agent Dispatcher (agents.py)   │          │     │
│  │  │                                       │          │     │
│  │  │  Leader ──→ Idea / Code / Writing     │          │     │
│  │  └───────────────────────────────────────┘          │     │
│  │      │             │              │                 │     │
│  │      ↓             ↓              ↓                 │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │     │
│  │  │ Memory   │ │ Monitor  │ │  Tools   │           │     │
│  │  │ Manager  │ │ (Zero$)  │ │ Registry │           │     │
│  │  └──────────┘ └──────────┘ └──────────┘           │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────── GPU Layer ──────────────────────────────┐     │
│  │  detect.py  │  keeper.py                            │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────── Skills Layer ───────────────────────────┐     │
│  │  daily-papers │ paper-analyze │ conf-search │ report │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Core Loop (`core/loop.py`)

The main orchestrator. Runs the THINK → EXECUTE → REFLECT cycle indefinitely.

**Key design decisions:**
- **Signal handling**: SIGTERM/SIGINT trigger graceful shutdown
- **Cycle counter**: Persisted to `.cycle_counter` file (survives restarts)
- **Smart cooldown**: Polls every N seconds instead of fixed sleep
- **Directive consumption**: Human directives are archived after reading (no re-reads)
- **Error backoff**: Doubles cooldown after errors to prevent burn loops

### 2. Agent Dispatcher (`core/agents.py`)

**Leader-Worker pattern** where:
- Leader persists conversation within a cycle (for coherent multi-step reasoning)
- Workers are stateless (each dispatch is independent)
- Only one worker runs at a time

**Why this works:**
- Leader sees the full picture without re-reading everything each step
- Workers are cheap (no accumulated context)
- Switching workers costs nothing (previous worker's context is gone)

### 3. Memory Manager (`core/memory.py`)

**Two tiers with automatic compaction:**

- **Tier 1 (Brief)**: Human-written, frozen. The "constitution" of the project.
- **Tier 2 (Log)**: Agent-written, rolling. Milestones and decisions.

**Compaction rules:**
1. Milestones: Drop oldest when section exceeds 1,200 chars
2. Decisions: Keep only last 15 entries
3. Total log: Hard cap at 2,000 chars (aggressive compaction if exceeded)

### 3b. Experiment Ledger & Research Journals (`core/ledger.py`, `core/journal.py`)

The compacting two-tier memory keeps context small but **drops detail**. The v2
autonomy layer adds durable, append-only records that complement it:

- **Experiment ledger** (`workspace/experiments.jsonl`): one JSON line per cycle
  (hypothesis, action, status, metrics, pid, conclusion). Append-only, crash-safe,
  queryable, zero LLM cost. Pure readers derive signals from it:
  - `summary(n)` — recent experiments block for the THINK context
  - `detect_stagnation(...)` — has the tracked metric stopped improving?
  - `check_phase_gate(...)` — does the best metric clear a configured bar?
- **Research journals**: `DEAD_ENDS.md` (failed approaches — do not retry) and
  `INSIGHTS.md` (durable observations). Never compacted; rotated to a dated
  `.bak` when oversized, so history is preserved, not lost.

These signals (plus a zero-cost violation scan from `core/safety.py`) are injected
into the leader's THINK/REFLECT context by `loop._enrich_context`. They are
**advisory and additive**: with default config they enrich planning without
changing control flow. `core/safety.py` also provides the proactive anti-burn
rate limiter (`seconds_until_allowed`) used by `loop._throttle_if_needed`.

### 4. Experiment Monitor (`core/monitor.py`)

**The zero-cost innovation.** During training:
- backend PID check — is process alive? (zero cost)
- backend GPU query — `nvidia-smi` when available (zero cost)
- backend file tail — last log lines (zero cost)

No LLM API calls until training completes.

### 5. Execution Backend (`core/execution.py`)

Execution is pluggable:

- **LocalExecutionBackend**: current behavior, runs everything inside `project.workspace`
- **SSHExecutionBackend**: keeps controller state local, but runs the tool-visible
  workspace, training commands, log reads, PID checks, and GPU queries on one
  remote host over SSH
- **SlurmExecutionBackend**: for Slurm-managed clusters. Subclasses the SSH backend
  (file/repo/`run_command` ops run on the login node, which shares the NFS
  workspace) and only changes job handling: training is submitted with
  `sbatch --parsable` over one transient ssh call that exits immediately, so no
  process is ever left running on the login node. The Slurm job id is carried in
  the `pid` field, `sacct` is the sole liveness authority (Slurm enforces
  `--time`, so a job is guaranteed terminal by `--time` + a buffer), and GPU
  status is reported from the partition's `squeue` occupancy.

The SSH/Slurm backends are intentionally narrow:
- one remote host (the login node, in Slurm mode)
- one remote workspace root
- single-cluster Slurm submission; no multi-host orchestration

### 6. Tool Registry (`core/tools.py`)

**Per-agent minimal tool sets** reduce token overhead:
- Each tool definition is ~200 tokens in the API call
- 15 tools = 3,000 extra tokens per call
- 5 tools = 1,000 extra tokens per call
- Over 100 API calls/day, that's still ~200K tokens saved

Tools are grouped so each worker only carries what it needs:
- **Codebase comprehension** (code/writing agents): `list_tree` (recursive,
  depth-limited repo map), `search_code` (regex grep across files), and
  `read_file` with optional line ranges for large files.
- **Literature** (idea agent): `search_papers` (Semantic Scholar),
  `search_arxiv` (freshest preprints), and `get_paper` (full details plus
  reference/citation snowballing).

ToolRegistry still owns command parsing and path safety. It validates
relative paths and parses shell text into argv before delegating execution
to the selected backend. The repo-reading tools skip `.git`, `__pycache__`,
and similar noise directories and run identically in local and SSH modes.

### 7. Tool-Use Protocol (`core/agents.py::dispatch_worker`)

Workers drive tool calls through a provider-agnostic text protocol rather
than each SDK's native tool-use API:

1. The dispatcher renders the worker's tool schemas as a plain-text
   `## Tool-Use Protocol` section and appends it to the system prompt.
2. The worker emits zero or more `<tool_call>{"name": "...", "args": {...}}</tool_call>`
   blocks in its response.
3. For each block, the dispatcher calls `ToolRegistry.execute_tool` and
   packages the JSON result into a `<tool_result name="...">...</tool_result>`
   block appended to the next user turn.
4. The loop iterates until the worker returns a message with no tool calls
   (the final answer) or `max_turns` is reached.

Design rationale:

- **Uniform behaviour across four providers.** The same protocol works
  whether the LLM is reached via the Anthropic SDK, the OpenAI SDK, the
  `claude` CLI, or the `codex` CLI. The execution loop contains no
  per-provider branching.
- **Authoritative experiment hand-off.** `pid` and `log_file` flow from
  the `launch_experiment` tool result (structured JSON) to
  `_parse_worker_response`, which promotes them onto the top-level result
  dict read by `loop._monitor_experiment`. Regex-on-prose remains as a
  fallback only.
- **CLI lock-down.** `claude_cli` is invoked with `--tools ""` so the
  Claude Code CLI cannot bypass the protocol using its built-in tools.
  `codex_cli` has no equivalent flag, so it may silently act on its own;
  `dispatch_worker` logs a warning when `codex_cli` is used as a worker
  provider, and the README compatibility table flags it accordingly.
- **Fence stripping.** Tool-call blocks inside triple-backtick code fences
  are removed before parsing so that models illustrating the protocol in
  their prose do not trigger real side-effectful tool execution.
- **Bounded execution.** `max_turns` is configured per-worker
  (`idea=12`, `code=40`, `writing=30`); on overflow the loop exits cleanly
  and the last response is returned with a warning.

### 8. GPU Utilities (`gpu/`)

- **detect.py**: Auto-detect GPUs, check availability, reserve last GPU
- **keeper.py**: Keep cloud instances alive with minimal GPU activity

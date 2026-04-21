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

The SSH backend is intentionally narrow in v1:
- one remote host
- one remote workspace root
- no scheduler integration
- no multi-host orchestration

### 6. Tool Registry (`core/tools.py`)

**Per-agent minimal tool sets** reduce token overhead:
- Each tool definition is ~200 tokens in the API call
- 15 tools = 3,000 extra tokens per call
- 4 tools = 800 extra tokens per call
- Over 100 API calls/day, that's 220K tokens saved

ToolRegistry still owns command parsing and path safety. It validates
relative paths and parses shell text into argv before delegating execution
to the selected backend.

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

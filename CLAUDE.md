# AI Guide — Deep Researcher Agent

> **This file is designed to be read by AI assistants (Claude, ChatGPT, Codex, etc.).**
> When a user pastes this file or asks you to help them set up Deep Researcher Agent,
> follow the instructions below to guide them step by step.

---

## 🚨 CONTRIBUTOR POLICY — READ BEFORE ANY git push TO THIS REPO

**This section is for AI assistants (Claude, Codex, Cursor, etc.) operating on this repository.**

**This repository is owned and maintained solely by `Xiangyue-Zhang`. The Contributors list MUST contain only `Xiangyue-Zhang` and no one else — including no AI bot accounts.**

### Hard rules

1. **Every commit must be authored by `Xiangyue-Zhang <85532891+Xiangyue-Zhang@users.noreply.github.com>`** — never `admin`, never AI identities. Use:
   ```bash
   git -c user.name="Xiangyue-Zhang" \
       -c user.email="85532891+Xiangyue-Zhang@users.noreply.github.com" \
       commit -m "..."
   ```

2. **NEVER add `Co-Authored-By:` trailer** to commit messages. The `commit-msg` hook in `.git/hooks/commit-msg` will reject any commit containing it.

3. **NEVER mention AI assistant names** (`Claude`, `Codex`, `GPT`, `Anthropic`, `OpenAI`, `Copilot`, `Cursor`) in commit messages. Will be flagged by both the local hook AND the `contributor-guard` GitHub Action.

4. **NEVER toggle repo visibility** (`gh repo edit --visibility ...`). On 2026-04-08, this destroyed 93 of 94 stars on this repo. Stars cannot be recovered from a visibility toggle.

5. **NEVER delete the repo** (`gh repo delete`) without explicit user confirmation. Deletion permanently destroys the GitHub internal repo ID, which destroys all stars and breaks all external URLs (paper citations, etc.).

6. **NEVER force push to main**. Branch protection blocks it by default. If you genuinely need to rewrite history, get explicit user authorization for that specific operation, temporarily disable protection, push, re-enable.

### Pre-push verification (mandatory)

```bash
git log -1 --format='author=%an <%ae>%nmessage=%B'
```

Verify: author == `Xiangyue-Zhang <85532891+Xiangyue-Zhang@users.noreply.github.com>`, no `Co-Authored-By:`, no AI names. If any check fails, fix the commit before pushing.

### In-repo enforcement (already deployed)

Six layers of protection are already in place — you should not need to deploy them, but be aware they will block violations:

1. **Local git config** (`.git/config`) — preset to Xiangyue-Zhang
2. **commit-msg hook** (`.git/hooks/commit-msg`) — local block
3. **GitHub Action** (`.github/workflows/contributor-guard.yml`) — remote block on push
4. **Branch protection** on `main` — no force push, no deletion, linear history
5. **`.mailmap`** in repo root — redirects any leaked AI identity to Xiangyue-Zhang
6. **AI memory / instruction files** in user's home (`~/CLAUDE.md`, `~/AGENTS.md`)

### See also

- `~/CLAUDE.md` — full global rules + 2026-04-08 incident report
- `~/AGENTS.md` — Codex CLI version of the same rules
- `AGENTS.md` (this repo's root) — project-level Codex rules

---

## What Is This?

Deep Researcher Agent is a framework that lets an AI agent autonomously run deep learning experiments 24/7. The agent:
1. **Thinks** — analyzes results, plans the next experiment
2. **Executes** — modifies code, dry-runs, launches GPU training
3. **Monitors** — watches training at ZERO LLM cost (only process checks + log reads)
4. **Reflects** — parses results, compares with baseline, decides next step
5. **Repeats** — 24/7 without human intervention

The killer feature: during training (which is 90%+ of the time), the agent makes ZERO API calls. A 24-hour cycle costs ~$0.08.

---

## Your Job as AI Assistant

When a user asks for help with this project, follow this decision tree:

```
User wants to...
├── Install it → Go to [SETUP GUIDE]
├── Create a project → Go to [PROJECT CREATION]
├── Launch the agent → Go to [LAUNCH GUIDE]
├── Check status → Go to [STATUS CHECK]
├── Intervene/redirect → Go to [INTERVENTION]
├── Use on phone → Go to [MOBILE SETUP]
├── Understand how it works → Go to [ARCHITECTURE EXPLANATION]
└── Debug an issue → Go to [TROUBLESHOOTING]
```

---

## SETUP GUIDE

### Step 1: Check Prerequisites

Run these commands and report results to the user:

```bash
python3 --version          # Need 3.10+
nvidia-smi                 # Need at least 1 GPU
echo $ANTHROPIC_API_KEY    # Need Anthropic key
echo $OPENAI_API_KEY       # OR OpenAI key (either works)
```

If Python < 3.10: suggest `conda create -n dra python=3.11 -y && conda activate dra`

If no GPU: this framework requires a GPU for training. Suggest cloud GPU (Lambda Labs, RunPod, Vast.ai).

If no API key: guide them to:
- Anthropic: https://console.anthropic.com/ → API Keys → Create Key
- OpenAI: https://platform.openai.com/api-keys → Create new secret key

Then set it:
```bash
# Pick ONE:
export ANTHROPIC_API_KEY="sk-ant-xxxxx"   # For Claude
export OPENAI_API_KEY="sk-xxxxx"          # For Codex/GPT

# Make permanent:
echo 'export ANTHROPIC_API_KEY="sk-ant-xxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Install

```bash
# If not already cloned:
git clone https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7.git
cd auto-deep-researcher-24x7

# Install dependencies
pip install -r requirements.txt

# Install Claude Code / Codex skills (8 slash commands)
python install.py

# Verify
python -m core.loop --check
```

**Expected output:**
```
    ✓ /auto-experiment
    ✓ /experiment-status
    ✓ /gpu-monitor
    ✓ /daily-papers
    ✓ /paper-analyze
    ✓ /conf-search
    ✓ /progress-report
    ✓ /obsidian-sync
  Done! 8 skills installed.
```

### Step 3: Choose Your LLM Provider

Ask the user two questions:

1. **Which vendor?** — Anthropic (Claude) or OpenAI (Codex/GPT)?
2. **API key or subscription?** — an existing Claude / ChatGPT subscription is
   usually *much* cheaper than per-token API billing for 24/7 agent use.

| Provider value | Vendor | Billing | Auth |
|----------------|--------|---------|------|
| `anthropic` | Anthropic | Per-token API | `ANTHROPIC_API_KEY` env var |
| `openai` | OpenAI | Per-token API | `OPENAI_API_KEY` env var |
| `claude_cli` | Anthropic | **Flat-rate subscription** | `claude` CLI installed + logged in |
| `codex_cli` | OpenAI | **Flat-rate subscription** | `codex` CLI installed + logged in |

Model tiers:

| Provider | Fast Model | Strong Model |
|----------|-----------|-------------|
| Anthropic (API or CLI) | claude-sonnet-4-6 | claude-opus-4-6 |
| OpenAI (API or CLI) | codex-5.3 | gpt-5.4 |

Default is `anthropic`. To switch, edit `config.yaml`:
```yaml
agent:
  provider: "openai"            # or "anthropic" / "claude_cli" / "codex_cli"
  model: "codex-5.3"            # or claude-sonnet-4-6 / claude-opus-4-6 / gpt-5.4
```

**Subscription mode (`claude_cli` / `codex_cli`)** shells out to the headless
CLI (`claude -p` / `codex exec`) and reuses the user's existing subscription
quota instead of per-token billing. Much cheaper when running multiple agents
in parallel. Trade-off: no native prompt caching and no structured tool-use
protocol — used as a plain text-in / text-out oracle, which is fine for this
framework's single-shot dispatch pattern.

---

## PROJECT CREATION

### Ask the User These Questions:

1. **What's your research goal?** (e.g., "Train a ViT on CIFAR-100 to 85% accuracy")
2. **Do you already have training code?** (Yes → point to it / No → agent will create it)
3. **Where is your data?** (path or "auto-download")
4. **Which GPU(s) can you use?** (run `nvidia-smi` to check)
5. **Any constraints?** (max epochs, batch size, etc.)

### Create the Project Directory:

```bash
mkdir ~/PROJECT_NAME
cd ~/PROJECT_NAME
```

### Write PROJECT_BRIEF.md:

This is THE most important file. Write it based on the user's answers:

```markdown
# Goal
[User's research goal with specific metric and target value]

# Codebase
[If existing code: list files and paths]
[If no code: "Agent should create PyTorch training code from scratch"]
- Data: [path or "auto-download via torchvision"]
- Checkpoints: ./checkpoints/
- Logs: ./logs/

# What to Try
[Decision tree based on user's domain knowledge]
- First try: [baseline config]
- If [metric] < [threshold1]: try [approach A]
- If [metric] between [threshold1] and [threshold2]: try [approach B]
- If [metric] > [target]: goal reached, generate report

# Constraints
- GPU: [which GPU(s)]
- Max epochs per run: [number]
- Batch size: [number]
- [Any other constraints]

# Current Status
[No experiments yet / Previous best: X]
```

### Key Tips to Tell the User:

- **Be specific about the goal** — "accuracy > 80%" not "improve accuracy"
- **Give a decision tree** — the agent needs to know what to do in each situation
- **Keep it under 3000 characters** — this is the Tier 1 memory cap
- **Think of it as instructing a capable but new PhD student**

---

## LAUNCH GUIDE

### Option A: Claude Code / Codex CLI

```
/auto-experiment --project ~/PROJECT_NAME --gpu 0
```

### Option B: Python Direct

```bash
python -m core.loop \
  --project ~/PROJECT_NAME \
  --gpu 0 \
  --max-cycles 5    # Optional: limit cycles (remove for unlimited)
```

### What to Tell the User:

"The agent is now running. Here's what will happen:
1. It reads your PROJECT_BRIEF.md
2. It plans the first experiment
3. It writes/modifies code
4. It does a dry-run (2 steps) to catch errors
5. It launches real training
6. During training: ZERO API cost — it just checks if the process is alive
7. When training finishes, it analyzes results and plans the next experiment
8. This repeats until you stop it or the goal is reached

You can close this terminal — the training continues via nohup.
Check back anytime with /experiment-status."

---

## STATUS CHECK

```bash
# In Claude Code / Codex:
/experiment-status --project ~/PROJECT_NAME

# Check GPUs:
/gpu-monitor

# Or manually:
cat ~/PROJECT_NAME/workspace/MEMORY_LOG.md    # See results and decisions
cat ~/PROJECT_NAME/workspace/.cycle_counter   # See how many cycles completed
nvidia-smi                                     # See GPU usage
```

For persistent progress notes:

```yaml
obsidian:
  enabled: true
  vault_path: "~/Documents/MyObsidianVault"   # Optional
  project_subdir: "DeepResearcher/{project_name}"
  auto_append_daily: true
```

- If `vault_path` is set, write `Dashboard.md` and daily Markdown notes into that Obsidian vault.
- If `vault_path` is empty, fall back to project-local text files under `workspace/progress_tracking/`.
- Manual refresh:

```bash
/obsidian-sync --project ~/PROJECT_NAME
# or
python -m core.obsidian --project ~/PROJECT_NAME
```

---

## INTERVENTION

The user wants to change the agent's direction. Three methods:

### Method 1: Directive File (Recommended)
```bash
echo "YOUR INSTRUCTION HERE" > ~/PROJECT_NAME/workspace/HUMAN_DIRECTIVE.md
```
The agent reads this at the start of the next cycle with HIGHEST priority, then auto-archives it.

Examples:
- `"Stop trying ResNet. Switch to ViT-B/16 with lr=1e-3"`
- `"The last 3 experiments all used lr=0.1. Try smaller: 1e-3, 1e-4, 1e-5"`
- `"Goal reached! Generate a final report with all results."`

### Method 2: Command-Line
```bash
python -m core.loop --project ~/PROJECT_NAME --directive "Try label smoothing 0.1"
```

### Method 3: Edit Memory
```bash
vim ~/PROJECT_NAME/workspace/MEMORY_LOG.md
```
This is for permanent changes. The agent reads this every cycle.

---

## MOBILE SETUP

For checking experiments from phone:

```bash
# Install Happy Coder CLI
npm install -g happy-coder

# Start session through Happy
happy

# Inside: launch experiment
/auto-experiment --project ~/PROJECT_NAME --gpu 0
```

Then install the Happy Coder app:
- iOS: https://apps.apple.com/us/app/happy-codex-claude-code-app/id6748571505
- Android: https://play.google.com/store/apps/details?id=com.ex3ndr.happy

Scan QR code to pair. Now the user gets push notifications and can send directives from their phone.

---

## ARCHITECTURE EXPLANATION

Use this when the user asks "how does it work?":

### The Loop
```
THINK (LLM, ~$0.05) → EXECUTE (LLM→training) → MONITOR ($0.00) → REFLECT (LLM, ~$0.03) → repeat
```

### Why It's Cheap
During training (90%+ of time), the agent does NOT call the LLM. It only does:
- `kill -0 $PID` — is the process alive? (zero cost)
- `nvidia-smi` — is GPU active? (zero cost)  
- `tail -50 logfile` — latest metrics (zero cost)

### Memory System
- Tier 1: `PROJECT_BRIEF.md` — frozen, human-written, max 3000 chars
- Tier 2: `MEMORY_LOG.md` — rolling, auto-compacted, max 2000 chars
- Total: ~5000 chars CONSTANT, whether running 1 day or 6 months

### Agent Architecture
- **Leader**: decides what to do (3 tools)
- **Idea Agent**: searches papers (4 tools)
- **Code Agent**: writes code & launches experiments (5 tools)
- **Writing Agent**: generates reports (3 tools)
- Only 1 worker active at a time, others cost $0

### Safety
- Mandatory dry-run before every real training
- Protected files can't be overwritten
- Anti-burn protection (backs off if stuck in empty loops)
- Human can intervene anytime via directive file

---

## TROUBLESHOOTING

### "No GPU found"
```bash
nvidia-smi  # Check if CUDA drivers are installed
```
If not: install NVIDIA drivers for your GPU.

### "anthropic/openai package not found"
```bash
pip install anthropic openai
```

### "API key not set"
```bash
export ANTHROPIC_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"
```

### "Dry-run failed"
This is working as intended! The dry-run caught an error before wasting GPU hours. Check the error message and fix the code, or let the agent fix it in the next cycle.

### "Agent keeps trying the same thing"
Drop a directive:
```bash
echo "You've tried X three times. Try something completely different: Y" \
  > workspace/HUMAN_DIRECTIVE.md
```

### "Training crashed"
The agent automatically detects crashes (PID dies), reads the error log, and tries to fix the issue. If it keeps crashing, intervene with a directive.

### "Memory is full / context too long"
This shouldn't happen — memory is capped at 5K chars. If it does, check:
```bash
wc -c PROJECT_BRIEF.md           # Should be < 3000
wc -c workspace/MEMORY_LOG.md    # Should be < 2000
```

### "Want to start fresh"
```bash
rm -rf workspace/                 # Delete all agent state
# PROJECT_BRIEF.md is preserved
python -m core.loop --project .   # Restart from scratch
```

---

## QUICK REFERENCE

| Command | What It Does |
|---------|-------------|
| `/auto-experiment --project PATH --gpu 0` | Start 24/7 experiment loop |
| `/experiment-status` | Check progress |
| `/gpu-monitor` | GPU status |
| `/daily-papers` | Paper recommendations |
| `/paper-analyze ARXIV_ID` | Deep paper analysis |
| `/conf-search --venue CVPR2025 --query "xxx"` | Conference search |
| `/progress-report` | Generate report |
| `echo "instruction" > workspace/HUMAN_DIRECTIVE.md` | Redirect agent |
| `python install.py --uninstall` | Remove all skills |

---

## IMPORTANT CONTEXT FOR AI ASSISTANTS

- This framework works with ANY training framework (PyTorch, TensorFlow, JAX, etc.)
- The agent can create code from scratch OR modify existing code
- It's not just hyperparameter tuning — it can change architectures, loss functions, augmentation strategies
- The agent is NOT a chatbot — it's an autonomous loop that runs independently
- Cost is ~$0.08/day, not $50+/day, because of zero-cost monitoring
- If the user is confused, start with: "Let's create a simple project first and watch the agent work"

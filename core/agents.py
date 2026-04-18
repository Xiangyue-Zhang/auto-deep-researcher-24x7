"""
AutoResearcher Agent Dispatcher

Leader-Worker architecture for efficient token usage:
- Leader: Central decision-maker, persistent conversation within a cycle
- Workers: Specialized agents (idea/code/writing), spawned on demand

Only ONE worker runs at a time. Others idle at zero token cost.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.agents")


# Agent definitions directory
AGENTS_DIR = Path(__file__).parent.parent / "agents"


class AgentDispatcher:
    """Dispatches tasks to specialized agents.

    The Leader agent decides what to do, then dispatches to workers:
    - idea_agent: Literature search, hypothesis formation
    - code_agent: Experiment implementation and execution
    - writing_agent: Report generation and paper writing

    Each worker has a minimal tool set (3-5 tools) to reduce token overhead.
    """

    WORKER_CONFIGS = {
        "idea": {
            "prompt_file": "idea_agent.md",
            "max_turns": 12,
            "tools": ["search_papers", "get_paper", "write_file", "read_file"],
        },
        "code": {
            "prompt_file": "code_agent.md",
            "max_turns": 40,
            "tools": ["run_shell", "launch_experiment", "write_file", "read_file", "list_files"],
        },
        "writing": {
            "prompt_file": "writing_agent.md",
            "max_turns": 30,
            "tools": ["write_file", "read_file", "list_files"],
        },
    }

    # Model mapping between providers
    MODEL_MAP = {
        # Anthropic ↔ OpenAI equivalents
        "claude-sonnet-4-6": "codex-5.3",     # Fast tier
        "claude-opus-4-6": "gpt-5.4",          # Strongest tier
        "codex-5.3": "claude-sonnet-4-6",
        "gpt-5.4": "claude-opus-4-6",
    }

    # Supported providers:
    #   "anthropic"  — Anthropic SDK, per-token API billing (needs ANTHROPIC_API_KEY)
    #   "openai"     — OpenAI SDK, per-token API billing (needs OPENAI_API_KEY)
    #   "claude_cli" — `claude -p` subprocess, uses Claude Code / Pro / Max subscription
    #   "codex_cli"  — `codex exec` subprocess, uses ChatGPT Plus / Pro subscription
    SUPPORTED_PROVIDERS = ("anthropic", "openai", "claude_cli", "codex_cli")

    def __init__(self, model: str = "claude-sonnet-4-6", provider: str = "anthropic", max_steps: int = 3):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported: {self.SUPPORTED_PROVIDERS}"
            )
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self._leader_history = []

    def dispatch_leader(self, task: str, context: dict) -> dict:
        """Send a task to the Leader agent.

        The Leader maintains conversation history within a cycle for
        coherent multi-step reasoning. History is cleared between cycles.

        Args:
            task: "think" or "reflect"
            context: Current state (brief, memory, results, etc.)

        Returns:
            Leader's decision as a dict
        """
        system_prompt = self._load_prompt("leader.md")

        messages = list(self._leader_history)
        messages.append({
            "role": "user",
            "content": self._format_leader_input(task, context),
        })

        response = self._call_llm(
            system=system_prompt,
            messages=messages,
            max_turns=10,
        )

        # Persist conversation for within-cycle coherence
        self._leader_history = messages + [{"role": "assistant", "content": response}]

        return self._parse_leader_response(response)

    def dispatch_worker(self, agent_type: str, task: str, tools: list) -> dict:
        """Dispatch a task to a worker agent.

        Workers are stateless — each dispatch is independent.
        This keeps token costs predictable.

        Args:
            agent_type: "idea", "code", or "writing"
            task: Task description from the Leader
            tools: Tool definitions to provide

        Returns:
            Worker's result as a dict
        """
        if agent_type not in self.WORKER_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")

        config = self.WORKER_CONFIGS[agent_type]
        system_prompt = self._load_prompt(config["prompt_file"])

        logger.info(f"Dispatching {agent_type} agent: {task[:100]}...")

        response = self._call_llm(
            system=system_prompt,
            messages=[{"role": "user", "content": task}],
            tools=tools,
            max_turns=config["max_turns"],
        )

        result = self._parse_worker_response(response, agent_type)
        logger.info(f"Worker {agent_type} completed: {str(result)[:200]}")
        return result

    def reset_leader_history(self):
        """Clear leader conversation history between cycles."""
        self._leader_history = []

    def _call_llm(self, system: str, messages: list, tools: list = None, max_turns: int = 10) -> str:
        """Call the LLM. Four providers are supported.

        - "anthropic":  Claude SDK, per-token API billing
        - "openai":     OpenAI SDK, per-token API billing
        - "claude_cli": `claude -p` subprocess, uses Claude Code / Pro / Max subscription
        - "codex_cli":  `codex exec` subprocess, uses ChatGPT Plus / Pro subscription

        CLI providers let you reuse existing subscriptions instead of paying per-token,
        which is much cheaper when running many agents in parallel or doing heavy
        Think/Reflect cycles. Trade-off: no native prompt caching, no tool-use protocol —
        the CLI is used as a plain text-in / text-out oracle.
        """
        if self.provider == "claude_cli":
            return self._call_claude_cli(system, messages)
        if self.provider == "codex_cli":
            return self._call_codex_cli(system, messages)
        if self.provider == "openai":
            return self._call_openai(system, messages)
        return self._call_anthropic(system, messages)

    def _call_anthropic(self, system: str, messages: list) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            client = anthropic.Anthropic()

            api_messages = []
            for msg in messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": api_messages,
            }

            response = client.messages.create(**kwargs)
            return response.content[0].text

        except ImportError:
            logger.warning("anthropic package not installed. Trying openai fallback.")
            return self._call_openai(system, messages)

    def _call_openai(self, system: str, messages: list) -> str:
        """Call OpenAI API (Codex 5.3 / GPT 5.4)."""
        try:
            import openai

            client = openai.OpenAI()

            # Map model name if it's an Anthropic model name
            model = self.MODEL_MAP.get(self.model, self.model) if self.provider != "openai" else self.model

            api_messages = [{"role": "system", "content": system}]
            for msg in messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=api_messages,
            )

            return response.choices[0].message.content

        except ImportError:
            logger.warning("openai package not installed. Using mock response.")
            return json.dumps({"action": "wait", "reason": "LLM not available"})

    @staticmethod
    def _flatten_for_cli(system: str, messages: list) -> str:
        """Serialize (system + chat history) into a single prompt for CLI subprocess.

        The headless CLI tools (claude -p / codex exec) take one blob of text and
        return the assistant reply. We rebuild the conversation using simple
        section markers rather than a structured role schema — good enough for
        single-turn dispatches, which is how the loop already uses the LLM.
        """
        parts = [f"===== SYSTEM =====\n{system.strip()}\n"]
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = str(msg.get("content", "")).strip()
            parts.append(f"===== {role} =====\n{content}\n")
        parts.append("===== ASSISTANT =====\n")
        return "\n".join(parts)

    def _run_cli(self, argv: list, prompt: str, tool_label: str, install_hint: str,
                 use_stdin: bool = False) -> str:
        """Invoke a headless CLI tool and return its stdout as the assistant reply."""
        import subprocess

        try:
            if use_stdin:
                result = subprocess.run(
                    argv,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
            else:
                result = subprocess.run(
                    argv + [prompt],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
        except FileNotFoundError:
            logger.warning(
                f"{tool_label} CLI not found on PATH. "
                f"Install: {install_hint}. Falling back to mock response."
            )
            return json.dumps({"action": "wait", "reason": f"{tool_label} CLI missing"})
        except subprocess.TimeoutExpired:
            logger.error(f"{tool_label} CLI timed out after 600s")
            return json.dumps({"action": "wait", "reason": f"{tool_label} CLI timeout"})
        except OSError as e:
            # argv too large (E2BIG) — retry via stdin
            if not use_stdin and getattr(e, "errno", None) == 7:
                logger.info(f"{tool_label} argv exceeded OS limit; retrying via stdin.")
                return self._run_cli(argv, prompt, tool_label, install_hint, use_stdin=True)
            raise

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
            logger.error(
                f"{tool_label} CLI exited {result.returncode}. "
                f"Stderr tail: {' | '.join(stderr_tail)}"
            )
            return json.dumps({"action": "wait", "reason": f"{tool_label} CLI error"})

        return (result.stdout or "").strip()

    def _call_claude_cli(self, system: str, messages: list) -> str:
        """Headless dispatch via the `claude` CLI, billed against a Pro / Max subscription."""
        prompt = self._flatten_for_cli(system, messages)
        # claude -p reads the prompt from stdin when no argument is given,
        # which avoids argv size limits for long memory contexts.
        return self._run_cli(
            argv=["claude", "-p", "--output-format", "text"],
            prompt=prompt,
            tool_label="claude",
            install_hint="npm i -g @anthropic-ai/claude-code && run `claude` once to sign in",
            use_stdin=True,
        )

    def _call_codex_cli(self, system: str, messages: list) -> str:
        """Headless dispatch via the `codex` CLI, billed against a ChatGPT subscription."""
        prompt = self._flatten_for_cli(system, messages)
        return self._run_cli(
            argv=["codex", "exec"],
            prompt=prompt,
            tool_label="codex",
            install_hint="brew install codex (or see upstream project) then run `codex login`",
            use_stdin=False,
        )

    def _load_prompt(self, filename: str) -> str:
        """Load agent prompt from agents/ directory."""
        prompt_path = AGENTS_DIR / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return f"You are the {filename.replace('.md', '')} agent."

    def _format_leader_input(self, task: str, context: dict) -> str:
        """Format context into a structured input for the Leader."""
        parts = [f"## Task: {task.upper()}\n"]

        if context.get("directive"):
            parts.append(f"## Human Directive (HIGHEST PRIORITY)\n{context['directive']}\n")

        parts.append(f"## Project Brief\n{context.get('brief', 'N/A')}\n")
        parts.append(f"## Memory Log\n{context.get('memory_log', 'N/A')}\n")
        parts.append(f"## Cycle: {context.get('cycle', 'N/A')}\n")

        if context.get("experiment_result"):
            parts.append(f"## Experiment Result\n{json.dumps(context['experiment_result'], indent=2)}\n")

        return "\n".join(parts)

    def _parse_leader_response(self, response: str) -> dict:
        """Parse Leader's response into structured action."""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract action from text
        response_lower = response.lower()
        if "wait" in response_lower or "no experiment" in response_lower:
            return {"action": "wait", "reason": response[:200]}

        return {
            "action": "experiment",
            "agent": "code",
            "task": response,
        }

    def _parse_worker_response(self, response: str, agent_type: str) -> dict:
        """Parse worker response into structured result."""
        result = {"agent": agent_type, "response": response}

        # Check for experiment launch indicators
        if agent_type == "code":
            if "PID" in response or "launched" in response.lower():
                result["experiment_launched"] = True
                # Try to extract PID
                import re
                pid_match = re.search(r"PID[=:\s]+(\d+)", response)
                if pid_match:
                    result["pid"] = int(pid_match.group(1))

        return result

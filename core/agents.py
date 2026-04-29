"""
AutoResearcher Agent Dispatcher

Leader-Worker architecture for efficient token usage:
- Leader: Central decision-maker, persistent conversation within a cycle
- Workers: Specialized agents (idea/code/writing), spawned on demand

Only ONE worker runs at a time. Others idle at zero token cost.

Tool use is implemented via a provider-agnostic text protocol. The LLM
emits <tool_call>{...}</tool_call> blocks, the dispatcher executes each
call through the ToolRegistry, and results are fed back as
<tool_result name="...">...</tool_result> blocks in the next user turn.
The loop runs until the worker produces a response with no tool calls
(the final answer) or max_turns is exceeded. This works uniformly
across all four providers — the API SDKs don't use their native
tool-use protocol, and the CLI providers are simply text oracles.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

# json_repair for lenient JSON parsing of LLM output
try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None

logger = logging.getLogger("autoresearcher.agents")


# Agent definitions directory
AGENTS_DIR = Path(__file__).parent.parent / "agents"


# Tool-use text protocol
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Triple-backtick fenced blocks are stripped before parsing so that LLMs can
# illustrate the protocol inside code fences without triggering real tool
# execution. Matches ``` with an optional language tag through the next ```.
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


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
    #   "anthropic"  — Anthropic-compatible SDK endpoint (default auth env: ANTHROPIC_API_KEY)
    #   "openai"     — OpenAI-compatible SDK endpoint (default auth env: OPENAI_API_KEY)
    #   "claude_cli" — `claude -p` subprocess, uses Claude Code / Pro / Max subscription
    #   "codex_cli"  — `codex exec` subprocess, uses ChatGPT Plus / Pro subscription
    SUPPORTED_PROVIDERS = ("anthropic", "openai", "claude_cli", "codex_cli")

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        provider: str = "anthropic",
        max_steps: int = 3,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "",
        auth_token: Optional[str] = None,
        auth_token_env: str = "",
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported: {self.SUPPORTED_PROVIDERS}"
            )
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.base_url = (base_url or "").strip() or None
        self.api_key = api_key or self._resolve_secret(api_key_env)
        self.auth_token = auth_token or self._resolve_secret(auth_token_env)
        self._leader_history = []

    @staticmethod
    def _resolve_secret(env_name: str) -> Optional[str]:
        env_name = (env_name or "").strip()
        if not env_name:
            return None
        return os.environ.get(env_name)

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

        response = self._call_llm(system=system_prompt, messages=messages)

        # Persist conversation for within-cycle coherence
        self._leader_history = messages + [{"role": "assistant", "content": response}]

        return self._parse_leader_response(response)

    def dispatch_worker(self, agent_type: str, task: str, tool_registry) -> dict:
        """Dispatch a task to a worker agent and run its tool-use loop.

        Workers are stateless across dispatches — each call starts with a
        fresh conversation. Within a single dispatch the conversation is
        multi-turn: the worker may emit tool calls, receive results, and
        continue reasoning until it produces a final answer (a response
        containing no <tool_call> blocks).

        Args:
            agent_type: "idea", "code", or "writing".
            task: Task description from the Leader.
            tool_registry: ToolRegistry that provides `get_tools_for` and
                `execute_tool`. The registry itself is passed in so this
                module does not have a hard import dependency on tools.py.

        Returns:
            Dict with at minimum `agent` and `response`. If the worker
            called `launch_experiment`, the PID and log_file from that
            tool result are also surfaced at the top level so the loop's
            EXECUTE → MONITOR handoff keeps working.
        """
        if agent_type not in self.WORKER_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if tool_registry is None:
            raise TypeError(
                "dispatch_worker requires a tool_registry with "
                "`get_tools_for(agent_type)` and `execute_tool(name, args)` "
                "methods. Pass a ToolRegistry configured with an empty tool "
                "list if you want a tool-less worker."
            )

        config = self.WORKER_CONFIGS[agent_type]
        base_prompt = self._load_prompt(config["prompt_file"])
        tool_defs = tool_registry.get_tools_for(agent_type)
        system_prompt = base_prompt + "\n\n" + self._render_tools_section(tool_defs)
        max_turns = config["max_turns"]

        # codex_cli hard-codes its own agentic tool loop; it will ignore the
        # <tool_call> protocol and silently act on its own. That breaks the
        # EXECUTE → MONITOR handoff (no PID, no log_file from ToolRegistry).
        # Leader/think dispatches are fine (they do not use tools) but worker
        # dispatches will likely return a non-authoritative summary. Warn once
        # per dispatch so users see it in the log without it becoming noise.
        if self.provider == "codex_cli" and tool_defs:
            logger.warning(
                "codex_cli is being used as a worker provider; its CLI does "
                "not support disabling built-in tools, so it may bypass the "
                "ToolRegistry and the resulting PID/log_file cannot be "
                "recovered. For worker dispatches prefer claude_cli, "
                "anthropic, or openai."
            )

        logger.info(f"Dispatching {agent_type} agent: {task[:100]}...")

        messages = [{"role": "user", "content": task}]
        last_response = ""
        tool_results_log: list[dict] = []

        for turn in range(1, max_turns + 1):
            last_response = self._call_llm(system=system_prompt, messages=messages)

            tool_calls = self._parse_tool_calls(last_response)
            if not tool_calls:
                # No tool calls → worker has produced its final answer.
                break

            # Echo the assistant turn so the next LLM call sees the history.
            messages.append({"role": "assistant", "content": last_response})

            # Execute each call and build a single user turn with all results.
            result_blocks = []
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                if not isinstance(args, dict):
                    tool_output = json.dumps({"error": "`args` must be a JSON object"})
                else:
                    tool_output = tool_registry.execute_tool(name, args)
                tool_results_log.append({"name": name, "args": args, "output": tool_output})
                result_blocks.append(
                    f'<tool_result name="{name}">\n{tool_output}\n</tool_result>'
                )

            messages.append({
                "role": "user",
                "content": "\n\n".join(result_blocks),
            })
        else:
            # for/else: executed only when the loop exhausts max_turns without break.
            logger.warning(
                f"Worker {agent_type} hit max_turns={max_turns} "
                f"with tool calls still pending; returning last response."
            )

        result = self._parse_worker_response(last_response, agent_type, tool_results_log)
        logger.info(f"Worker {agent_type} completed: {str(result)[:200]}")
        return result

    def reset_leader_history(self):
        """Clear leader conversation history between cycles."""
        self._leader_history = []

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        """Extract <tool_call>{...}</tool_call> blocks from an LLM response.

        Silently skips blocks whose JSON body is malformed. An empty list
        means the response is a final answer (no tool calls requested).

        Tool-call blocks inside triple-backtick code fences are deliberately
        ignored: LLMs routinely illustrate the protocol inside fenced blocks
        when explaining what they are about to do, and executing those
        illustrations as real side-effectful calls has caused accidental
        writes in practice.
        """
        stripped = _FENCED_BLOCK_RE.sub("", text or "")
        calls: list[dict] = []

        _fix_json = _repair_json if _repair_json is not None else (lambda x: x)

        for match in _TOOL_CALL_RE.finditer(stripped):
            body = match.group(1)
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                # Fallback: try json_repair for lenient parsing
                try:
                    fixed = _fix_json(body)
                    parsed = json.loads(fixed) if fixed and fixed.strip() else None
                except (json.JSONDecodeError, ValueError, TypeError):
                    logger.warning(
                        f"Skipping malformed tool_call block (even after repair): {body[:200]}"
                    )
                    continue
            if isinstance(parsed, dict) and parsed.get("name"):
                calls.append(parsed)
            else:
                logger.warning(
                    "Skipping tool_call without a string `name` field: "
                    f"{str(parsed)[:120]}"
                )
        return calls

    @staticmethod
    def _render_tools_section(tool_defs: list[dict]) -> str:
        """Render tool schemas as a plain-text block appended to the system prompt.

        The worker's own prompt already has a short 'Tools Available' list;
        this auto-generated section provides the exact machine-readable
        schemas and protocol instructions so the LLM emits calls in the
        format the dispatcher can parse.
        """
        if not tool_defs:
            return ""

        lines = [
            "## Tool-Use Protocol",
            "",
            "You have NO direct access to the filesystem, shell, or network.",
            "To act on the environment you MUST emit `<tool_call>` blocks and",
            "wait for the framework to return `<tool_result>` blocks in the",
            "next user turn. Example:",
            "",
            "    <tool_call>",
            '    {"name": "read_file", "args": {"path": "config.yaml"}}',
            "    </tool_call>",
            "",
            "You may emit multiple `<tool_call>` blocks in one message; each",
            "will be executed and its result returned. When you are finished,",
            "produce a plain-text message with NO `<tool_call>` blocks — that",
            "is how the framework knows you are done.",
            "",
            "Emit `<tool_call>` blocks at the top level of the message. Do NOT",
            "wrap them in triple-backtick code fences — fenced blocks are",
            "treated as illustration, not as real calls.",
            "",
            "### Available tools",
            "",
        ]
        for tool in tool_defs:
            name = tool.get("name", "<unnamed>")
            desc = tool.get("description", "")
            schema = tool.get("input_schema", {})
            lines.append(f"- `{name}` — {desc}")
            props = schema.get("properties", {}) or {}
            required = set(schema.get("required", []) or [])
            for pname, pspec in props.items():
                ptype = pspec.get("type", "any")
                pdesc = pspec.get("description", "")
                flag = "required" if pname in required else "optional"
                lines.append(f"    - `{pname}` ({ptype}, {flag}): {pdesc}")
        return "\n".join(lines)

    def _call_llm(self, system: str, messages: list) -> str:
        """Call the LLM. Four providers are supported.

        - "anthropic":  Anthropic-compatible SDK endpoint, per-token API billing
        - "openai":     OpenAI-compatible SDK endpoint, per-token API billing
        - "claude_cli": `claude -p` subprocess, uses Claude Code / Pro / Max subscription
        - "codex_cli":  `codex exec` subprocess, uses ChatGPT Plus / Pro subscription

        CLI providers let you reuse existing subscriptions instead of paying per-token,
        which is much cheaper when running many agents in parallel or doing heavy
        Think/Reflect cycles. Trade-off: no native prompt caching, no native tool-use
        protocol — the LLM is driven purely as a text-in / text-out oracle, and tool
        use is layered on top via the <tool_call> text protocol (see dispatch_worker).
        """
        if self.provider == "claude_cli":
            return self._call_claude_cli(system, messages)
        if self.provider == "codex_cli":
            return self._call_codex_cli(system, messages)
        if self.provider == "openai":
            return self._call_openai(system, messages)
        return self._call_anthropic(system, messages)

    def _call_anthropic(self, system: str, messages: list) -> str:
        """Call an Anthropic-compatible Messages API."""
        try:
            import anthropic

            client_kwargs = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.auth_token:
                client_kwargs["auth_token"] = self.auth_token
            # When using auth_token without api_key, prevent the SDK from
            # reading an empty ANTHROPIC_API_KEY env var (sends empty
            # x-api-key header that can interfere with Bearer auth)
            _clear_key = (
                self.auth_token
                and not self.api_key
                and os.environ.get("ANTHROPIC_API_KEY", "") == ""
            )
            if _clear_key:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                client = anthropic.Anthropic(**client_kwargs)
            finally:
                if _clear_key:
                    os.environ["ANTHROPIC_API_KEY"] = ""

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
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        except ImportError:
            logger.warning("anthropic package not installed. Trying openai fallback.")
            return self._call_openai(system, messages)

    def _call_openai(self, system: str, messages: list) -> str:
        """Call an OpenAI-compatible chat completions API."""
        try:
            import openai

            client_kwargs = {}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            client = openai.OpenAI(**client_kwargs)

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
        """Headless dispatch via the `claude` CLI, billed against a Pro / Max subscription.

        `--tools ""` disables every built-in tool so the CLI degrades to a
        pure text oracle. This is required for our <tool_call> protocol
        to work: the CLI must be unable to act on its own, otherwise it
        will bypass our ToolRegistry (and the loop loses visibility over
        what actually happened, especially for launch_experiment PIDs).

        The prompt is piped via stdin to sidestep argv-size limits on
        large conversation histories.
        """
        prompt = self._flatten_for_cli(system, messages)
        return self._run_cli(
            argv=["claude", "-p", "--output-format", "text", "--tools", ""],
            prompt=prompt,
            tool_label="claude",
            install_hint="npm i -g @anthropic-ai/claude-code && run `claude` once to sign in",
            use_stdin=True,
        )

    def _call_codex_cli(self, system: str, messages: list) -> str:
        """Headless dispatch via the `codex` CLI, billed against a ChatGPT subscription.

        Unlike `claude -p`, `codex exec` is fully agentic by default — it runs
        its own internal tool-use loop and there is no CLI flag to disable
        the built-in tools. That means the framework's <tool_call> protocol
        is unreliable under this provider: codex will often act on its own
        and return a final summary. Workers that need to launch experiments
        (and recover a PID from the ToolRegistry) should therefore prefer
        claude_cli / anthropic / openai; codex_cli is best kept for the
        leader/think path where we only need free-text output.

        Flags:
          - `-o <tempfile>`       captures only the final assistant message
                                  instead of the full agentic trace,
          - `--skip-git-repo-check` allows codex to run in arbitrary dirs
                                    (the workspace is typically not a repo).
        """
        import subprocess
        import tempfile

        prompt = self._flatten_for_cli(system, messages)

        try:
            with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out:
                out_path = out.name
            try:
                result = subprocess.run(
                    [
                        "codex", "exec",
                        "--skip-git-repo-check",
                        "-o", out_path,
                        prompt,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
            except FileNotFoundError:
                logger.warning(
                    "codex CLI not found on PATH. "
                    "Install: brew install codex (or see upstream) then `codex login`. "
                    "Falling back to mock response."
                )
                return json.dumps({"action": "wait", "reason": "codex CLI missing"})
            except subprocess.TimeoutExpired:
                logger.error("codex CLI timed out after 600s")
                return json.dumps({"action": "wait", "reason": "codex CLI timeout"})

            if result.returncode != 0:
                stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
                logger.error(
                    f"codex CLI exited {result.returncode}. "
                    f"Stderr tail: {' | '.join(stderr_tail)}"
                )
                return json.dumps({"action": "wait", "reason": "codex CLI error"})

            try:
                with open(out_path, "r") as f:
                    return f.read().strip()
            except OSError:
                # Fall back to stdout if --output-last-message didn't produce a file.
                return (result.stdout or "").strip()
        finally:
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass

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

    def _parse_worker_response(self, response: str, agent_type: str,
                               tool_results: Optional[list] = None) -> dict:
        """Parse worker response into a structured result dict.

        When the worker used the `launch_experiment` tool, the PID and
        log_file come directly from that tool's JSON result — this is
        authoritative. The regex-on-free-text path is retained as a
        fallback for responses that report an experiment launch purely
        in prose (or for older prompts that predate the tool-use loop).
        """
        result = {"agent": agent_type, "response": response}
        if tool_results:
            result["tool_calls"] = len(tool_results)

        if agent_type == "code":
            # Prefer authoritative tool-result data over text parsing.
            launch_result = None
            if tool_results:
                for entry in reversed(tool_results):
                    if entry.get("name") == "launch_experiment":
                        launch_result = entry
                        break
            if launch_result is not None:
                try:
                    payload = json.loads(launch_result.get("output", "{}"))
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("pid") is not None:
                    result["experiment_launched"] = True
                    result["pid"] = int(payload["pid"])
                    if payload.get("log_file"):
                        result["log_file"] = payload["log_file"]

            # Fallback: scrape PID from free-text response.
            if "pid" not in result and ("PID" in response or "launched" in response.lower()):
                result["experiment_launched"] = True
                pid_match = re.search(r"PID[=:\s]+(\d+)", response)
                if pid_match:
                    result["pid"] = int(pid_match.group(1))

        return result

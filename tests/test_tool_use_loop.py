"""Tests for the worker tool-use loop in core/agents.py.

The loop must:
  - run multiple turns until the model stops emitting <tool_call> blocks,
  - feed each tool's output back as a <tool_result> in the next user turn,
  - respect max_turns as a hard ceiling,
  - surface PID/log_file from a launch_experiment tool call so the EXECUTE
    → MONITOR handoff in core/loop.py still works,
  - ignore malformed <tool_call> JSON bodies without crashing.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agents import AgentDispatcher
from core.tools import ToolRegistry


def _make_dispatcher():
    # Provider choice is irrelevant because we stub _call_llm everywhere.
    return AgentDispatcher(provider="anthropic")


class ParseToolCallsTests(unittest.TestCase):
    def test_extracts_multiple_blocks_in_order(self):
        text = """
        I will take two actions.
        <tool_call>
        {"name": "read_file", "args": {"path": "a.txt"}}
        </tool_call>
        Some prose in between.
        <tool_call>
        {"name": "write_file", "args": {"path": "b.txt", "content": "hi"}}
        </tool_call>
        """
        calls = AgentDispatcher._parse_tool_calls(text)
        self.assertEqual([c["name"] for c in calls], ["read_file", "write_file"])
        self.assertEqual(calls[1]["args"]["path"], "b.txt")

    def test_empty_when_no_blocks(self):
        self.assertEqual(AgentDispatcher._parse_tool_calls("final answer"), [])

    def test_skips_malformed_json(self):
        text = """
        <tool_call>{"name": "ok", "args": {}}</tool_call>
        <tool_call>{not valid json</tool_call>
        <tool_call>"just a string"</tool_call>
        """
        calls = AgentDispatcher._parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "ok")

    def test_ignores_tool_calls_inside_code_fences(self):
        """LLMs often illustrate the protocol inside fenced blocks when
        explaining themselves; those illustrations must NOT execute."""
        text = '''
        Here is how you would call the tool:

        ```
        <tool_call>
        {"name": "write_file", "args": {"path": "DANGER.txt", "content": "pwned"}}
        </tool_call>
        ```

        But for this task I do not need any tools.
        '''
        self.assertEqual(AgentDispatcher._parse_tool_calls(text), [])

    def test_mix_of_fenced_illustration_and_real_call(self):
        """A real top-level call must still be picked up even if the message
        also contains an illustrative fenced example."""
        text = '''
        For reference, the general form looks like:

        ```
        <tool_call>
        {"name": "write_file", "args": {"path": "EXAMPLE.txt", "content": "x"}}
        </tool_call>
        ```

        Now I will do the real call:

        <tool_call>
        {"name": "read_file", "args": {"path": "actual.txt"}}
        </tool_call>
        '''
        calls = AgentDispatcher._parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[0]["args"]["path"], "actual.txt")

    def test_tolerates_missing_args_key(self):
        """A tool_call without an `args` field is valid; args default to {}."""
        text = '<tool_call>{"name": "list_files"}</tool_call>'
        calls = AgentDispatcher._parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_files")

    def test_rejects_args_that_is_not_a_dict(self):
        """If the LLM emits args as a string or list, the dispatcher must
        surface a structured error rather than crashing on **kwargs."""
        dispatcher = _make_dispatcher()
        registry = _FakeRegistry(
            tools=[{"name": "read_file", "description": "", "input_schema": {}}],
            outputs={},
        )
        turns = [
            '<tool_call>{"name": "read_file", "args": "not-a-dict"}</tool_call>',
            "giving up",
        ]
        with patch.object(dispatcher, "_call_llm", side_effect=turns):
            dispatcher.dispatch_worker("writing", "t", registry)
        # The registry must NOT have been called with a non-dict args payload.
        self.assertEqual(registry.calls, [])


class RenderToolsSectionTests(unittest.TestCase):
    def test_renders_schema_properties(self):
        tool = {
            "name": "search_papers",
            "description": "Search Semantic Scholar.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        }
        rendered = AgentDispatcher._render_tools_section([tool])
        self.assertIn("search_papers", rendered)
        self.assertIn("<tool_call>", rendered)
        self.assertIn("query", rendered)
        self.assertIn("required", rendered)
        self.assertIn("Max results", rendered)

    def test_empty_when_no_tools(self):
        self.assertEqual(AgentDispatcher._render_tools_section([]), "")


class _FakeRegistry:
    """Minimal ToolRegistry stub that records calls and returns canned outputs."""

    def __init__(self, tools, outputs):
        self._tools = tools
        self._outputs = outputs  # dict: tool_name -> json string
        self.calls: list[tuple[str, dict]] = []

    def get_tools_for(self, agent_type):
        return self._tools

    def execute_tool(self, name, args):
        self.calls.append((name, args))
        return self._outputs.get(name, json.dumps({"ok": True}))


class DispatchWorkerLoopTests(unittest.TestCase):
    def test_none_registry_raises_clear_typeerror(self):
        """Passing tool_registry=None must fail at the boundary with a
        clear TypeError, not with a cryptic AttributeError deep in the
        loop. External callers of dispatch_worker will hit this edge."""
        dispatcher = _make_dispatcher()
        with self.assertRaises(TypeError) as ctx:
            dispatcher.dispatch_worker("writing", "task", None)
        self.assertIn("tool_registry", str(ctx.exception))
        self.assertIn("get_tools_for", str(ctx.exception))

    def test_unknown_agent_type_raises_before_touching_registry(self):
        """Agent-type validation should happen first so that a bad
        agent_type produces ValueError regardless of registry state."""
        dispatcher = _make_dispatcher()
        with self.assertRaises(ValueError):
            dispatcher.dispatch_worker("bogus_agent", "task", None)

    def test_terminates_when_response_has_no_tool_calls(self):
        dispatcher = _make_dispatcher()
        registry = _FakeRegistry(tools=[], outputs={})

        with patch.object(dispatcher, "_call_llm", return_value="all done, no tools"):
            result = dispatcher.dispatch_worker("writing", "task", registry)

        self.assertEqual(registry.calls, [])
        self.assertEqual(result["agent"], "writing")
        self.assertIn("all done", result["response"])

    def test_executes_tools_and_feeds_results_back(self):
        dispatcher = _make_dispatcher()
        fake_tools = [{"name": "read_file", "description": "read",
                       "input_schema": {"type": "object", "properties": {}}}]
        registry = _FakeRegistry(
            tools=fake_tools,
            outputs={"read_file": json.dumps({"content": "file body"})},
        )

        turns = [
            '<tool_call>{"name": "read_file", "args": {"path": "a.txt"}}</tool_call>',
            "Done reading, here is my summary.",
        ]
        call_log: list[list] = []

        def fake_call(system, messages):
            call_log.append(list(messages))
            return turns.pop(0)

        with patch.object(dispatcher, "_call_llm", side_effect=fake_call):
            result = dispatcher.dispatch_worker("writing", "task", registry)

        # One tool call was executed with the right args.
        self.assertEqual(registry.calls, [("read_file", {"path": "a.txt"})])
        # Second LLM turn saw the assistant's tool_call and a user tool_result.
        self.assertEqual(len(call_log), 2)
        second_turn_messages = call_log[1]
        self.assertEqual(second_turn_messages[0]["role"], "user")  # original task
        self.assertEqual(second_turn_messages[1]["role"], "assistant")  # tool call echo
        self.assertEqual(second_turn_messages[2]["role"], "user")  # tool_result block
        self.assertIn("<tool_result", second_turn_messages[2]["content"])
        self.assertIn("file body", second_turn_messages[2]["content"])
        # Final result captures the summary response and tool-call count.
        self.assertEqual(result["tool_calls"], 1)
        self.assertIn("summary", result["response"])

    def test_max_turns_hard_ceiling(self):
        dispatcher = _make_dispatcher()
        registry = _FakeRegistry(
            tools=[{"name": "read_file", "description": "", "input_schema": {}}],
            outputs={},
        )

        # Every turn keeps requesting another tool call → would loop forever.
        infinite = '<tool_call>{"name": "read_file", "args": {"path": "x"}}</tool_call>'

        with patch.object(dispatcher, "_call_llm", return_value=infinite):
            # Override max_turns on the 'writing' worker to something small for the test.
            with patch.dict(AgentDispatcher.WORKER_CONFIGS["writing"], {"max_turns": 3}):
                result = dispatcher.dispatch_worker("writing", "task", registry)

        # Exactly 3 tool executions, no more.
        self.assertEqual(len(registry.calls), 3)
        self.assertEqual(result["response"], infinite)

    def test_surfaces_pid_from_launch_experiment_tool_result(self):
        """The EXECUTE → MONITOR handoff in core/loop.py reads result['pid']
        and result['log_file']. These must come from the tool result (which
        is authoritative) rather than regex-scraping the model's prose."""
        dispatcher = _make_dispatcher()
        launch_output = json.dumps({"pid": 4321, "log_file": "/tmp/exp.log", "status": "launched"})
        registry = _FakeRegistry(
            tools=[{"name": "launch_experiment", "description": "", "input_schema": {}}],
            outputs={"launch_experiment": launch_output},
        )

        turns = [
            '<tool_call>{"name": "launch_experiment", '
            '"args": {"command": "python train.py", "log_file": "exp.log"}}'
            '</tool_call>',
            # Deliberately give a prose reply that lies about the PID — tool result should win.
            "Training started, PID=99999 (this number is wrong).",
        ]

        with patch.object(dispatcher, "_call_llm", side_effect=turns):
            result = dispatcher.dispatch_worker("code", "launch it", registry)

        self.assertTrue(result["experiment_launched"])
        self.assertEqual(result["pid"], 4321)
        self.assertEqual(result["log_file"], "/tmp/exp.log")

    def test_end_to_end_with_real_registry(self):
        """Smoke test with a real ToolRegistry and a temp workspace."""
        dispatcher = _make_dispatcher()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = ToolRegistry(workspace)

            turns = [
                '<tool_call>{"name": "write_file", '
                '"args": {"path": "note.txt", "content": "hello"}}</tool_call>',
                '<tool_call>{"name": "read_file", '
                '"args": {"path": "note.txt"}}</tool_call>',
                "I wrote and read the file successfully.",
            ]

            with patch.object(dispatcher, "_call_llm", side_effect=turns):
                result = dispatcher.dispatch_worker("writing", "do it", registry)

            self.assertEqual(result["tool_calls"], 2)
            self.assertTrue((workspace / "note.txt").exists())
            self.assertEqual((workspace / "note.txt").read_text(), "hello")


if __name__ == "__main__":
    unittest.main()

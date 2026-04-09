import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_tool_registry():
    module_path = Path(__file__).resolve().parents[1] / "core" / "tools.py"
    spec = importlib.util.spec_from_file_location("core.tools", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ToolRegistry


ToolRegistry = load_tool_registry()


class ToolRegistrySecurityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        self.registry = ToolRegistry(self.workspace)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_write_file_rejects_path_traversal(self):
        result = json.loads(
            self.registry.execute_tool("write_file", {"path": "../escape.txt", "content": "owned"})
        )
        self.assertIn("error", result)
        self.assertIn("escapes workspace", result["error"])
        self.assertFalse((self.workspace.parent / "escape.txt").exists())

    def test_read_file_rejects_absolute_path(self):
        result = json.loads(self.registry.execute_tool("read_file", {"path": "/etc/hosts"}))
        self.assertIn("error", result)
        self.assertIn("relative to workspace", result["error"])

    def test_list_files_rejects_parent_escape(self):
        result = json.loads(self.registry.execute_tool("list_files", {"path": ".."}))
        self.assertIn("error", result)
        self.assertIn("escapes workspace", result["error"])

    def test_run_shell_does_not_execute_shell_injection_payload(self):
        result = json.loads(
            self.registry.execute_tool("run_shell", {"command": "echo hello; touch injected.txt"})
        )
        self.assertEqual(result["returncode"], 0)
        self.assertIn("hello; touch injected.txt", result["stdout"])
        self.assertFalse((self.workspace / "injected.txt").exists())

    def test_run_shell_blocks_dangerous_binaries(self):
        result = json.loads(self.registry.execute_tool("run_shell", {"command": "rm -rf tmp"}))
        self.assertIn("error", result)
        self.assertIn("Blocked executable", result["error"])

    def test_launch_experiment_rejects_log_path_traversal(self):
        result = json.loads(
            self.registry.execute_tool(
                "launch_experiment",
                {
                    "command": 'python3 -c "print(\'hi\')"',
                    "log_file": "../outside.log",
                },
            )
        )
        self.assertIn("error", result)
        self.assertIn("escapes workspace", result["error"])


if __name__ == "__main__":
    unittest.main()

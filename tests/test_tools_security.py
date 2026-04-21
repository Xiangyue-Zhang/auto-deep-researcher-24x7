import json
import tempfile
import unittest
from pathlib import Path

from core.execution import LocalExecutionBackend
from core.tools import ToolRegistry


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def read_file(self, path):
        self.calls.append(("read_file", path))
        return ""

    def write_file(self, path, content):
        self.calls.append(("write_file", path, content))
        return {"status": "written"}

    def list_files(self, path="."):
        self.calls.append(("list_files", path))
        return []

    def run_command(self, argv, timeout=120, env=None):
        self.calls.append(("run_command", argv, timeout, env))
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def launch_command(self, argv, log_file, env=None):
        self.calls.append(("launch_command", argv, log_file, env))
        return {"pid": 1, "log_file": log_file, "status": "launched"}


class ToolRegistrySecurityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name) / "workspace"
        self.workspace.mkdir()
        self.registry = ToolRegistry(LocalExecutionBackend(self.workspace))

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
        backend = RecordingBackend()
        registry = ToolRegistry(backend)
        result = json.loads(registry.execute_tool("read_file", {"path": "/etc/hosts"}))
        self.assertIn("error", result)
        self.assertIn("relative to workspace", result["error"])
        self.assertEqual(backend.calls, [])

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

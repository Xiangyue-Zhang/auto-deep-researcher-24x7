import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.execution import (
    LocalExecutionBackend,
    SSHExecutionBackend,
    build_execution_backend,
)
from core.monitor import ExperimentMonitor
from core.obsidian import ObsidianExporter
from core.memory import MemoryManager


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeBackend:
    def __init__(self, alive=None, tail=None, gpu=None):
        self.alive = list(alive or [])
        self.tail = list(tail or [])
        self.gpu = gpu or {"utilization": "N/A"}
        self.calls = []

    def validate(self):
        self.calls.append(("validate",))

    def read_file(self, path):
        self.calls.append(("read_file", path))
        return ""

    def write_file(self, path, content):
        self.calls.append(("write_file", path, content))
        return {"status": "written", "path": path, "bytes": len(content)}

    def list_files(self, path="."):
        self.calls.append(("list_files", path))
        return []

    def run_command(self, argv, timeout=120, env=None):
        self.calls.append(("run_command", argv, timeout, env))
        return {"stdout": "", "stderr": "", "returncode": 0}

    def launch_command(self, argv, log_file, env=None):
        self.calls.append(("launch_command", argv, log_file, env))
        return {"pid": 123, "log_file": log_file, "status": "launched"}

    def is_process_alive(self, pid):
        self.calls.append(("is_process_alive", pid))
        if self.alive:
            return self.alive.pop(0)
        return False

    def tail_file(self, path, lines=50):
        self.calls.append(("tail_file", path, lines))
        if self.tail:
            return self.tail.pop(0)
        return []

    def get_gpu_status(self):
        self.calls.append(("get_gpu_status",))
        return self.gpu


class BuildExecutionBackendTests(unittest.TestCase):
    def test_build_local_backend_by_default(self):
        backend = build_execution_backend(config={}, controller_workspace=Path("/tmp/workspace"))
        self.assertIsInstance(backend, LocalExecutionBackend)

    def test_build_ssh_backend(self):
        backend = build_execution_backend(
            config={
                "execution": {
                    "mode": "ssh",
                    "ssh_host": "user@example.com",
                    "remote_workspace": "/remote/ws",
                }
            },
            controller_workspace=Path("/tmp/workspace"),
        )
        self.assertIsInstance(backend, SSHExecutionBackend)
        self.assertEqual(backend.ssh_host, "user@example.com")
        self.assertEqual(backend.remote_workspace, "/remote/ws")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            build_execution_backend(
                config={"execution": {"mode": "bogus"}},
                controller_workspace=Path("/tmp/workspace"),
            )


class SSHExecutionBackendTests(unittest.TestCase):
    @patch("core.execution.shutil.which", return_value="/usr/bin/ssh")
    @patch("core.execution.subprocess.run")
    def test_validate_invokes_remote_helper(self, run_mock, _which_mock):
        run_mock.return_value = _Completed(stdout=json.dumps({"ok": True, "result": {"status": "ok"}}))
        backend = SSHExecutionBackend(
            ssh_host="user@example.com",
            remote_workspace="/remote/ws",
            remote_python="python3",
            ssh_args=["-p", "2222"],
        )

        backend.validate()

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][:4], ["ssh", "-p", "2222", "user@example.com"])
        self.assertIn("python3 -c", args[0][4])
        self.assertNotIn("import json", args[0][4])
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "validate")
        self.assertEqual(payload["remote_workspace"], "/remote/ws")
        self.assertIn("timeout", kwargs)
        self.assertFalse(kwargs["check"])

    @patch("core.execution.subprocess.run")
    def test_run_command_uses_json_stdin_and_no_shell(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps({"ok": True, "result": {"stdout": "hi", "stderr": "", "returncode": 0}})
        )
        backend = SSHExecutionBackend("user@example.com", "/remote/ws")

        result = backend.run_command(["python", "train.py"], timeout=42, env={"CUDA_VISIBLE_DEVICES": "0"})

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][0], "ssh")
        self.assertIn("base64", args[0][-1])
        self.assertNotIn("shell", kwargs)
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "run_command")
        self.assertEqual(payload["argv"], ["python", "train.py"])
        self.assertEqual(payload["timeout_seconds"], 42)
        self.assertEqual(payload["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(result["stdout"], "hi")

    @patch("core.execution.subprocess.run")
    def test_remote_file_not_found_maps_to_python_exception(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps({"ok": False, "error_type": "FileNotFoundError", "error": "File not found: x.txt"})
        )
        backend = SSHExecutionBackend("user@example.com", "/remote/ws")

        with self.assertRaises(FileNotFoundError):
            backend.read_file("x.txt")


class MonitorAndObsidianBackendTests(unittest.TestCase):
    def test_monitor_uses_backend_for_pid_log_and_gpu(self):
        backend = FakeBackend(
            alive=[True, False],
            tail=[["epoch 1"], ["epoch 1", "epoch 2 accuracy: 0.9"]],
            gpu={"utilization": "88%"},
        )
        monitor = ExperimentMonitor(poll_interval=0, backend=backend)
        monitor._active_experiments[123] = {"start_time": time.time(), "status": "running"}

        with patch("core.monitor.time.sleep", return_value=None):
            result = monitor.wait_for_completion(pid=123, log_file="logs/exp.log", notify=False)

        self.assertEqual(result["status"], "completed")
        self.assertIn("epoch 2", result["log_tail"])
        self.assertIn(("get_gpu_status",), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 5), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 50), backend.calls)

    def test_obsidian_dashboard_reads_remote_status_via_backend(self):
        backend = FakeBackend(alive=[True], tail=[["remote epoch 7"]])
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "PROJECT_BRIEF.md").write_text("Train model")
            workspace = project_dir / "workspace"
            workspace.mkdir()
            (workspace / "state.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "pid": 321,
                        "log_file": "logs/exp.log",
                        "started_at": time.time(),
                    }
                )
            )
            memory = MemoryManager(project_dir=project_dir)
            exporter = ObsidianExporter(
                config={"obsidian": {"enabled": True}},
                project_dir=project_dir,
                backend=backend,
            )

            result = exporter.refresh_dashboard(memory=memory, cycle_count=2)
            dashboard = Path(result["path"]).read_text()

        self.assertIn("TRAINING (PID 321", dashboard)
        self.assertIn("remote epoch 7", dashboard)
        self.assertIn(("is_process_alive", 321), backend.calls)
        self.assertIn(("tail_file", "logs/exp.log", 8), backend.calls)


if __name__ == "__main__":
    unittest.main()

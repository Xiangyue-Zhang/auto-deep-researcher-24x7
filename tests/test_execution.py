import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core.execution import (
    LocalExecutionBackend,
    REMOTE_HELPER,
    SSHExecutionBackend,
    SlurmExecutionBackend,
    build_execution_backend,
    _parse_slurm_time_seconds,
    _SLURM_RUNNING_STATES,
    _SLURM_OK_STATES,
    _SLURM_FAIL_STATES,
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
    def __init__(self, alive=None, tail=None, gpu=None, final=None):
        self.alive = list(alive or [])
        self.tail = list(tail or [])
        self.gpu = gpu or {"utilization": "N/A"}
        self.final = final or {"state": "unknown", "success": None}
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

    def final_status(self, pid):
        self.calls.append(("final_status", pid))
        return self.final


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
    def test_remote_helper_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            os.symlink(outside, root / "escape")

            payload = {
                "action": "write_file",
                "remote_workspace": str(root),
                "path": "escape/pwned.txt",
                "content": "x",
            }
            proc = subprocess.run(
                ["python3", "-c", REMOTE_HELPER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0)
            body = json.loads(proc.stdout)
            self.assertFalse(body["ok"])
            self.assertIn("escapes workspace", body["error"])
            self.assertFalse((outside / "pwned.txt").exists())

    def _run_helper(self, payload):
        proc = subprocess.run(
            ["python3", "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_remote_helper_grep_tree_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            (root / "pkg").mkdir(parents=True)
            (root / "pkg" / "m.py").write_text("def main():\n    return 1\n")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "x.py").write_text("def main(): pass\n")

            tree = self._run_helper(
                {"action": "list_tree", "remote_workspace": str(root), "path": "."}
            )
            self.assertTrue(tree["ok"])
            self.assertIn("pkg/", tree["result"]["entries"])
            self.assertIn("pkg/m.py", tree["result"]["entries"])
            self.assertNotIn("__pycache__/", tree["result"]["entries"])

            grep = self._run_helper(
                {"action": "grep_files", "remote_workspace": str(root), "pattern": "def main"}
            )
            self.assertTrue(grep["ok"])
            files = {h["file"] for h in grep["result"]["hits"]}
            self.assertEqual(files, {"pkg/m.py"})
            self.assertEqual(grep["result"]["hits"][0]["line"], 1)

            ranged = self._run_helper(
                {
                    "action": "read_file_range",
                    "remote_workspace": str(root),
                    "path": "pkg/m.py",
                    "start_line": 2,
                    "end_line": 2,
                }
            )
            self.assertTrue(ranged["ok"])
            self.assertEqual(ranged["result"]["content"], "2\t    return 1")

    def test_remote_helper_walk_and_grep_skip_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            outside = Path(tmp) / "outside"
            (outside / "sub").mkdir(parents=True)
            (outside / "creds.txt").write_text("TOPSECRET token\n")
            os.symlink(outside, root / "leakdir")
            os.symlink(outside / "creds.txt", root / "leak.txt")

            tree = self._run_helper({"action": "list_tree", "remote_workspace": str(root), "path": "."})
            self.assertTrue(tree["ok"])
            self.assertNotIn("leakdir/", tree["result"]["entries"])
            self.assertNotIn("leak.txt", tree["result"]["entries"])

            grep = self._run_helper(
                {"action": "grep_files", "remote_workspace": str(root), "pattern": "TOPSECRET"}
            )
            self.assertTrue(grep["ok"])
            self.assertEqual(grep["result"]["hits"], [])

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

    def test_monitor_reports_failed_from_backend_final_status(self):
        # A backend that reports a failed terminal state -> status "failed",
        # not a silent "completed".
        backend = FakeBackend(
            alive=[True, False],
            tail=[["epoch 1"], ["epoch 1", "Traceback: boom"]],
            final={"state": "FAILED", "success": False},
        )
        monitor = ExperimentMonitor(poll_interval=0, backend=backend)
        monitor._active_experiments[7] = {"start_time": time.time(), "status": "running"}

        with patch("core.monitor.time.sleep", return_value=None):
            result = monitor.wait_for_completion(pid=7, log_file="logs/exp.log", notify=False)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["terminal_state"], "FAILED")
        self.assertFalse(result["success"])
        self.assertIn(("final_status", 7), backend.calls)

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

    def test_obsidian_status_surfaces_failure(self):
        # A failed run must NOT render as IDLE on the dashboard.
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "PROJECT_BRIEF.md").write_text("Train model")
            (project_dir / "workspace").mkdir()
            exporter = ObsidianExporter(
                config={"obsidian": {"enabled": True}},
                project_dir=project_dir,
                backend=FakeBackend(),
            )
        self.assertEqual(
            exporter._format_status({"status": "failed", "terminal_state": "TIMEOUT"}),
            "FAILED (TIMEOUT)",
        )
        self.assertEqual(exporter._format_status({"status": "failed"}), "FAILED")
        self.assertEqual(exporter._format_status({"status": "no_pid"}), "FAILED (no PID)")
        self.assertEqual(exporter._format_status({"status": "completed"}), "COMPLETED")


class SlurmExecutionBackendTests(unittest.TestCase):
    LOGIN = "user@login-node"

    def _backend(self, **kw):
        defaults = dict(
            ssh_host=self.LOGIN,
            remote_workspace="/nfs/ws",
            slurm_partition="gpu",
            slurm_time="24:00:00",
            slurm_gpus_per_job=1,
        )
        defaults.update(kw)
        return SlurmExecutionBackend(**defaults)

    # --- factory + validation ---

    def test_factory_builds_slurm_backend(self):
        backend = build_execution_backend(
            config={
                "execution": {
                    "mode": "slurm",
                    "ssh_host": self.LOGIN,
                    "remote_workspace": "/nfs/ws",
                    "slurm_partition": "gpu-h200",
                    "slurm_time": "12:00:00",
                    "slurm_gpus_per_job": 2,
                    "ssh_args": ["-p", "2222"],
                }
            },
            controller_workspace=Path("/tmp/workspace"),
        )
        self.assertIsInstance(backend, SlurmExecutionBackend)
        self.assertEqual(backend.slurm_partition, "gpu-h200")
        self.assertEqual(backend.slurm_time, "12:00:00")
        self.assertEqual(backend.slurm_gpus_per_job, 2)
        self.assertEqual(backend.ssh_args, ["-p", "2222"])

    def test_unknown_mode_message_lists_slurm(self):
        with self.assertRaisesRegex(ValueError, "local, ssh, slurm"):
            build_execution_backend(
                config={"execution": {"mode": "bogus"}},
                controller_workspace=Path("/tmp/workspace"),
            )

    def test_validate_requires_partition_and_time(self):
        # partition missing -> raises before any ssh round-trip
        with self.assertRaisesRegex(ValueError, "slurm_partition is required"):
            self._backend(slurm_partition="").validate()
        with self.assertRaisesRegex(ValueError, "slurm_time is required"):
            self._backend(slurm_time="").validate()

    # --- launch (submit-and-exit) ---

    @patch("core.execution.subprocess.run")
    def test_launch_submits_and_parses_job_id(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps(
                {"ok": True, "result": {"slurm_job_id": 12345, "log_file": "logs/exp.log"}}
            )
        )
        backend = self._backend(slurm_gpus_per_job=2)

        result = backend.launch_command(
            ["python", "train.py"],
            "logs/exp.log",
            env={"CUDA_VISIBLE_DEVICES": "3", "FOO": "bar"},
        )

        self.assertEqual(result["pid"], 12345)
        self.assertEqual(result["slurm_job_id"], 12345)
        self.assertEqual(result["status"], "submitted")

        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][0], "ssh")            # transport is ssh, no local shell
        self.assertNotIn("shell", kwargs)
        payload = json.loads(kwargs["input"])
        self.assertEqual(payload["action"], "submit_slurm")
        self.assertEqual(payload["argv"], ["python", "train.py"])
        self.assertEqual(payload["partition"], "gpu")
        self.assertEqual(payload["gres"], 2)
        self.assertEqual(payload["env"]["FOO"], "bar")  # remote helper does the CUDA strip

    @patch("core.execution.subprocess.run")
    def test_launch_failure_raises(self, run_mock):
        run_mock.return_value = _Completed(
            stdout=json.dumps(
                {"ok": False, "error_type": "RuntimeError",
                 "error": "sbatch: error: invalid partition specified"}
            )
        )
        with self.assertRaises(RuntimeError):
            self._backend().launch_command(["python", "t.py"], "logs/exp.log")

    # --- liveness: sacct state map + anti-hang bounds ---

    def _alive_with_state(self, sacct_stdout):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout=sacct_stdout)):
            return backend.is_process_alive(12345)

    def test_is_alive_state_map(self):
        # Drive every enumerated state from the maps themselves so dropping a
        # state from its bucket (e.g. removing COMPLETING from running) regresses.
        for state in _SLURM_RUNNING_STATES:
            self.assertTrue(self._alive_with_state(state + "\n"), state)
        for state in _SLURM_OK_STATES:
            self.assertFalse(self._alive_with_state(state + "\n"), state)
        for state in _SLURM_FAIL_STATES:
            self.assertFalse(self._alive_with_state(state + "\n"), state)
        # Normalization edges + a non-fail indeterminate state.
        self.assertFalse(self._alive_with_state("CANCELLED+\n"))          # '+' suffix stripped
        self.assertFalse(self._alive_with_state("CANCELLED by 1001\n"))   # ' by <uid>' stripped
        # PREEMPTED is not a fail state -> indeterminate -> kept alive (1st grace poll)
        self.assertTrue(self._alive_with_state("PREEMPTED\n"))

    def test_is_alive_sacct_nonzero_rc_is_unknown_grace(self):
        # sacct exits non-zero (transient accounting error) -> indeterminate,
        # NOT dead: keep the job alive for the bounded grace window.
        backend = self._backend(slurm_unknown_grace_polls=2)
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="", returncode=1)):
            self.assertEqual([backend.is_process_alive(555) for _ in range(3)], [True, True, False])

    def test_is_alive_ssh_failure_is_unknown_grace(self):
        # ssh timeout -> indeterminate, NOT dead.
        backend = self._backend(slurm_unknown_grace_polls=2)
        with patch.object(backend, "_ssh_shell",
                          side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15)):
            self.assertEqual([backend.is_process_alive(556) for _ in range(3)], [True, True, False])

    def test_is_alive_pending_never_reaped_by_wallclock(self):
        # A job sacct still reports PENDING must NOT be reaped even long past
        # --time + buffer (queue wait is not bounded by --time).
        backend = self._backend(slurm_time="00:01:00", slurm_time_buffer=0)  # 60s cap
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="PENDING\n")):
            with patch("core.execution.time.time", side_effect=[1000.0, 1000.0 + 100000]):
                self.assertTrue(backend.is_process_alive(99))   # first poll
                self.assertTrue(backend.is_process_alive(99))   # 100000s later, still PENDING

    def test_is_alive_unknown_is_bounded(self):
        """Regression guard: a vanished/unreachable job must NOT hang forever."""
        backend = self._backend(slurm_unknown_grace_polls=3)
        # sacct empty AND squeue empty on every probe -> 'unknown' every time.
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="")):
            results = [backend.is_process_alive(777) for _ in range(4)]
        self.assertEqual(results, [True, True, True, False])

    @patch("core.execution.time.time")
    def test_is_alive_wallclock_cap(self, time_mock):
        backend = self._backend(slurm_time="00:01:00", slurm_time_buffer=0)  # 60s cap
        time_mock.side_effect = [1000.0, 1000.0 + 120]  # first seeds, second is past cap
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="")):
            self.assertTrue(backend.is_process_alive(42))   # within cap, unknown -> grace
            self.assertFalse(backend.is_process_alive(42))  # past --time+buffer -> reaped

    @patch("core.execution.subprocess.run")
    def test_liveness_reuses_host_and_args(self, run_mock):
        run_mock.return_value = _Completed(stdout="RUNNING\n")
        backend = self._backend(ssh_args=["-p", "2222"])

        self.assertTrue(backend.is_process_alive(12345))

        args, _ = run_mock.call_args
        self.assertEqual(args[0][:4], ["ssh", "-p", "2222", self.LOGIN])
        self.assertIn("sacct -j 12345", args[0][4])
        self.assertIn("State%30", args[0][4])              # explicit width, no truncation

    def test_final_status_reflects_terminal_state(self):
        backend = self._backend()
        # A COMPLETED job -> success True
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="COMPLETED\n")):
            self.assertFalse(backend.is_process_alive(1))   # records terminal state
        self.assertEqual(backend.final_status(1), {"state": "COMPLETED", "success": True})
        # A TIMEOUT job -> success False (not silently "completed")
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="TIMEOUT\n")):
            self.assertFalse(backend.is_process_alive(2))
        self.assertEqual(backend.final_status(2), {"state": "TIMEOUT", "success": False})
        # Never observed reaching a terminal state -> indeterminate
        self.assertEqual(backend.final_status(999), {"state": "unknown", "success": None})

    def test_get_gpu_status_parses_queue(self):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(stdout="   2 PENDING\n   1 RUNNING\n")):
            status = backend.get_gpu_status()
        self.assertEqual(status["utilization"], "slurm")
        self.assertEqual(status["pending"], 2)
        self.assertEqual(status["running"], 1)

    def test_cancel_calls_scancel(self):
        backend = self._backend()
        with patch.object(backend, "_ssh_shell", return_value=_Completed(returncode=0)) as shell:
            self.assertTrue(backend.cancel(12345))
        shell.assert_called_once()
        self.assertIn("scancel 12345", shell.call_args[0][0])
        # non-zero scancel -> False (not "return True unconditionally")
        with patch.object(backend, "_ssh_shell", return_value=_Completed(returncode=1)):
            self.assertFalse(backend.cancel(12345))
        # transport failure is swallowed -> False, never propagated
        with patch.object(backend, "_ssh_shell",
                          side_effect=subprocess.TimeoutExpired(cmd="scancel", timeout=8)):
            self.assertFalse(backend.cancel(12345))

    def test_parse_slurm_time_seconds(self):
        self.assertEqual(_parse_slurm_time_seconds("60"), 3600)            # bare minutes
        self.assertEqual(_parse_slurm_time_seconds("01:30"), 90)           # minutes:seconds
        self.assertEqual(_parse_slurm_time_seconds("12:00:00"), 43200)     # h:m:s
        self.assertEqual(_parse_slurm_time_seconds("2-00:00:00"), 172800)  # days-h:m:s
        self.assertEqual(_parse_slurm_time_seconds("1-12"), 129600)        # days-hours
        self.assertEqual(_parse_slurm_time_seconds("garbage"), 10 ** 9)    # sentinel


class SlurmRemoteHelperTests(unittest.TestCase):
    """Run the embedded REMOTE_HELPER as a subprocess (sbatch is absent here, so
    submission fails AFTER the script is written — we assert on the script)."""

    def _run_helper(self, payload):
        proc = subprocess.run(
            ["python3", "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_submit_slurm_builds_safe_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            self._run_helper(
                {
                    "action": "submit_slurm",
                    "remote_workspace": str(root),
                    "argv": ["python", "t.py", "--x", "a b"],
                    "log_file": "logs/exp.log",
                    "env": {"CUDA_VISIBLE_DEVICES": "3", "FOO": "b a r"},
                    "partition": "gpu",
                    "time": "01:00:00",
                    "gres": 2,
                    "raw_gres": "",
                    "qos": "",
                    "account": "",
                    "job_name": "ar_exp",
                    "setup": "module load cuda/12.4",
                    "extra_sbatch": ["--nodes=1"],
                }
            )
            # The output-log parent must be pre-created (Slurm won't make it).
            self.assertTrue((root / "logs").is_dir())
            script = (root / ".sbatch_ar_exp").read_text()

        self.assertIn("#SBATCH --partition=gpu", script)
        self.assertIn("#SBATCH --time=01:00:00", script)
        self.assertIn('#SBATCH --output="logs/exp.log"', script)   # quoted (whitespace-safe)
        self.assertIn("#SBATCH --gres=gpu:2", script)
        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("module load cuda/12.4", script)
        # env quoted safely; injection-prone arg quoted; GPU mask stripped.
        self.assertIn("export FOO='b a r'", script)
        self.assertIn("'a b'", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", script)
        # No persistent login-node construct (the 2026-05-29 MIL invariant).
        for forbidden in ("tmux", "srun", "--wait", "squeue", "while "):
            self.assertNotIn(forbidden, script)

    def _run_helper_with_path(self, payload, extra_path):
        env = {**os.environ, "PATH": extra_path + os.pathsep + os.environ["PATH"]}
        proc = subprocess.run(
            ["python3", "-c", REMOTE_HELPER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    @staticmethod
    def _fake_sbatch(bindir, body_line):
        fake = bindir / "sbatch"
        fake.write_text("#!/bin/bash\n" + body_line + "\n")
        fake.chmod(0o755)

    def _submit_payload(self, root, job_name):
        return {
            "action": "submit_slurm", "remote_workspace": str(root),
            "argv": ["python", "t.py"], "log_file": "out.log", "env": {},
            "partition": "gpu", "time": "01:00:00", "gres": 1,
            "raw_gres": "", "job_name": job_name,
        }

    def test_submit_slurm_parses_federated_job_id(self):
        # sbatch --parsable can emit the federated "<id>;<cluster>" form.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"; root.mkdir()
            binp = Path(tmp) / "bin"; binp.mkdir()
            self._fake_sbatch(binp, "printf '12345;cluster0\\n'")
            body = self._run_helper_with_path(self._submit_payload(root, "ar_fed"), str(binp))
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["result"]["slurm_job_id"], 12345)

    def test_submit_slurm_rejects_non_numeric_output(self):
        # A non --parsable line (e.g. "Submitted batch job 99") must be rejected,
        # not mis-parsed into a bogus job id.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"; root.mkdir()
            binp = Path(tmp) / "bin"; binp.mkdir()
            self._fake_sbatch(binp, "printf 'Submitted batch job 99\\n'")
            body = self._run_helper_with_path(self._submit_payload(root, "ar_bad"), str(binp))
        self.assertFalse(body["ok"])
        self.assertIn("did not return a job id", body["error"])

    def test_submit_slurm_raw_gres_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            root.mkdir()
            self._run_helper(
                {
                    "action": "submit_slurm",
                    "remote_workspace": str(root),
                    "argv": ["python", "t.py"],
                    "log_file": "out.log",
                    "env": {},
                    "partition": "gpu",
                    "time": "01:00:00",
                    "gres": 1,
                    "raw_gres": "gpu:a100:4",
                    "job_name": "ar_raw",
                }
            )
            script = (root / ".sbatch_ar_raw").read_text()
        self.assertIn("#SBATCH --gres=gpu:a100:4", script)
        self.assertNotIn("--gres=gpu:1", script)


if __name__ == "__main__":
    unittest.main()

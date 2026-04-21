"""
Execution backends for Deep Researcher Agent.

Local mode preserves the current behavior. SSH mode keeps the controller
state local while running file operations, shell commands, training, log
tailing, PID checks, and GPU inspection on one remote host.
"""

from __future__ import annotations

import json
import logging
import os
import base64
import shutil
import shlex
import subprocess
import textwrap
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger("autoresearcher.execution")


REMOTE_HELPER = textwrap.dedent(
    """
    import json
    import os
    import pathlib
    import subprocess
    import sys


    def normalize_rel(raw):
        if raw is None or not str(raw).strip():
            raise ValueError("Path cannot be empty")
        rel = pathlib.PurePosixPath(str(raw))
        if rel.is_absolute():
            raise ValueError("Path must be relative to workspace")
        if any(part == ".." for part in rel.parts):
            raise ValueError(f"Path escapes workspace: {raw}")
        parts = [part for part in rel.parts if part not in ("", ".")]
        return pathlib.Path(*parts)


    def resolve_path(root, raw):
        rel = normalize_rel(raw)
        return (root / rel).resolve(strict=False)


    def gpu_status():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "utilization": f"{parts[0]}%",
                                "memory": f"{parts[1]}MB/{parts[2]}MB",
                            }
                        )
                return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
        except Exception:
            pass
        return {"utilization": "N/A"}


    def main():
        payload = json.load(sys.stdin)
        root = pathlib.Path(payload["remote_workspace"]).expanduser().resolve(strict=False)
        action = payload["action"]
        result = None

        if action == "validate":
            root.mkdir(parents=True, exist_ok=True)
            result = {"status": "ok"}
        elif action == "read_file":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {payload['path']}")
            result = {"content": path.read_text()}
        elif action == "write_file":
            path = resolve_path(root, payload["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            content = payload["content"]
            path.write_text(content)
            result = {"status": "written", "path": payload["path"], "bytes": len(content)}
        elif action == "list_files":
            raw = payload.get("path", ".")
            if raw in ("", "."):
                path = root
            else:
                path = resolve_path(root, raw)
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {raw}")
            result = {"files": sorted(p.name for p in path.iterdir())[:100]}
        elif action == "run_command":
            try:
                proc = subprocess.run(
                    payload["argv"],
                    capture_output=True,
                    text=True,
                    timeout=int(payload.get("timeout_seconds", 120)),
                    cwd=str(root),
                    env={**os.environ, **(payload.get("env") or {})},
                    check=False,
                )
                result = {
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-500:],
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                result = {"error": f"Command timed out after {int(payload.get('timeout_seconds', 120))}s"}
        elif action == "launch_command":
            log_file = payload["log_file"]
            log_path = resolve_path(root, log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as handle:
                proc = subprocess.Popen(
                    payload["argv"],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **(payload.get("env") or {})},
                    start_new_session=True,
                    cwd=str(root),
                )
            result = {"pid": proc.pid, "log_file": log_file, "status": "launched"}
        elif action == "is_process_alive":
            try:
                os.kill(int(payload["pid"]), 0)
                result = {"alive": True}
            except OSError:
                result = {"alive": False}
        elif action == "tail_file":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                result = {"lines": []}
            else:
                lines = path.read_text().splitlines()
                result = {"lines": lines[-int(payload.get('lines', 50)) :]}
        elif action == "get_gpu_status":
            result = gpu_status()
        else:
            raise ValueError(f"Unknown action: {action}")

        json.dump({"ok": True, "result": result}, sys.stdout)


    if __name__ == "__main__":
        try:
            main()
        except Exception as exc:
            json.dump(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                sys.stdout,
            )
    """
).strip()

REMOTE_HELPER_B64 = base64.b64encode(REMOTE_HELPER.encode("utf-8")).decode("ascii")
REMOTE_LAUNCHER = "import base64,sys;exec(base64.b64decode(sys.argv[1]).decode())"


def normalize_relative_path(path: str) -> str:
    """Normalize a workspace-relative path and reject traversal."""
    if path is None or not str(path).strip():
        raise ValueError("Path cannot be empty")

    pure = PurePosixPath(str(path))
    if pure.is_absolute():
        raise ValueError("Path must be relative to workspace")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"Path escapes workspace: {path}")

    normalized = str(pure)
    return "." if normalized in ("", ".") else normalized


def _resolve_under_root(root: Path, rel_path: str) -> Path:
    parts = [part for part in PurePosixPath(rel_path).parts if part not in ("", ".")]
    resolved = (root / Path(*parts)).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc
    return resolved


class ExecutionBackend:
    """Abstract execution backend."""

    def validate(self):
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        raise NotImplementedError

    def write_file(self, path: str, content: str) -> dict:
        raise NotImplementedError

    def list_files(self, path: str = ".") -> list[str]:
        raise NotImplementedError

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def is_process_alive(self, pid: int) -> bool:
        raise NotImplementedError

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        raise NotImplementedError

    def get_gpu_status(self) -> dict:
        raise NotImplementedError


class LocalExecutionBackend(ExecutionBackend):
    """Current on-machine behavior."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def validate(self):
        self.workspace.mkdir(parents=True, exist_ok=True)

    def read_file(self, path: str) -> str:
        file_path = _resolve_under_root(self.workspace, normalize_relative_path(path))
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_text()

    def write_file(self, path: str, content: str) -> dict:
        rel_path = normalize_relative_path(path)
        file_path = _resolve_under_root(self.workspace, rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return {"status": "written", "path": rel_path, "bytes": len(content)}

    def list_files(self, path: str = ".") -> list[str]:
        rel_path = normalize_relative_path(path)
        dir_path = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted([f.name for f in dir_path.iterdir()])[:100]

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                env={**os.environ, **(env or {})},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s"}

        return {
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-500:],
            "returncode": result.returncode,
        }

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        rel_path = normalize_relative_path(log_file)
        log_path = _resolve_under_root(self.workspace, rel_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as handle:
            proc = subprocess.Popen(
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
                start_new_session=True,
                cwd=str(self.workspace),
            )

        return {"pid": proc.pid, "log_file": rel_path, "status": "launched"}

    def is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        rel_path = normalize_relative_path(path)
        file_path = _resolve_under_root(self.workspace, rel_path)
        if not file_path.exists():
            return []
        return file_path.read_text().splitlines()[-lines:]

    def get_gpu_status(self) -> dict:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "utilization": f"{parts[0]}%",
                                "memory": f"{parts[1]}MB/{parts[2]}MB",
                            }
                        )
                return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
        except Exception:
            pass
        return {"utilization": "N/A"}


class SSHExecutionBackend(ExecutionBackend):
    """Run the tool-visible workspace on a remote host over SSH."""

    def __init__(
        self,
        ssh_host: str,
        remote_workspace: str,
        remote_python: str = "python3",
        ssh_args: Optional[list[str]] = None,
    ):
        self.ssh_host = ssh_host
        self.remote_workspace = remote_workspace
        self.remote_python = remote_python or "python3"
        self.ssh_args = [str(arg) for arg in (ssh_args or [])]

    def validate(self):
        if not self.ssh_host:
            raise ValueError("execution.ssh_host is required when execution.mode=ssh")
        if not self.remote_workspace:
            raise ValueError("execution.remote_workspace is required when execution.mode=ssh")
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh binary not found on PATH")
        self._invoke("validate", transport_timeout=30)

    def read_file(self, path: str) -> str:
        payload = self._invoke("read_file", path=normalize_relative_path(path))
        return payload["content"]

    def write_file(self, path: str, content: str) -> dict:
        return self._invoke("write_file", path=normalize_relative_path(path), content=content)

    def list_files(self, path: str = ".") -> list[str]:
        payload = self._invoke("list_files", path=normalize_relative_path(path))
        return payload["files"]

    def run_command(self, argv: list[str], timeout: int = 120, env: Optional[dict] = None) -> dict:
        return self._invoke(
            "run_command",
            argv=argv,
            timeout_seconds=timeout,
            env=env or {},
            transport_timeout=timeout + 10,
        )

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        return self._invoke(
            "launch_command",
            argv=argv,
            log_file=normalize_relative_path(log_file),
            env=env or {},
            transport_timeout=30,
        )

    def is_process_alive(self, pid: int) -> bool:
        payload = self._invoke("is_process_alive", pid=int(pid), transport_timeout=15)
        return bool(payload["alive"])

    def tail_file(self, path: str, lines: int = 50) -> list[str]:
        payload = self._invoke("tail_file", path=normalize_relative_path(path), lines=lines, transport_timeout=15)
        return payload["lines"]

    def get_gpu_status(self) -> dict:
        return self._invoke("get_gpu_status", transport_timeout=20)

    def _invoke(self, action: str, transport_timeout: int = 60, **kwargs) -> dict:
        payload = {
            "action": action,
            "remote_workspace": self.remote_workspace,
            **kwargs,
        }
        remote_command = (
            f"{shlex.quote(self.remote_python)} -c {shlex.quote(REMOTE_LAUNCHER)} "
            f"{shlex.quote(REMOTE_HELPER_B64)}"
        )
        command = ["ssh", *self.ssh_args, self.ssh_host, remote_command]
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=transport_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SSH backend action '{action}' timed out after {transport_timeout}s") from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
            message = " | ".join(stderr_tail) if stderr_tail else "unknown ssh error"
            raise RuntimeError(f"SSH backend action '{action}' failed: {message}")

        try:
            payload = json.loads((result.stdout or "").strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SSH backend action '{action}' returned invalid JSON") from exc

        if not payload.get("ok"):
            error = payload.get("error", "unknown remote error")
            error_type = payload.get("error_type", "RuntimeError")
            if error_type == "FileNotFoundError":
                raise FileNotFoundError(error)
            if error_type == "NotADirectoryError":
                raise NotADirectoryError(error)
            if error_type == "ValueError":
                raise ValueError(error)
            raise RuntimeError(error)

        return payload.get("result", {})


def build_execution_backend(config: Optional[dict], controller_workspace: Path) -> ExecutionBackend:
    """Construct the execution backend from project config."""
    config = config or {}
    execution = config.get("execution", {}) or {}
    mode = execution.get("mode", "local")

    if mode == "ssh":
        return SSHExecutionBackend(
            ssh_host=execution.get("ssh_host", ""),
            remote_workspace=execution.get("remote_workspace", ""),
            remote_python=execution.get("remote_python", "python3"),
            ssh_args=execution.get("ssh_args", []) or [],
        )
    if mode != "local":
        raise ValueError(f"Unknown execution.mode '{mode}'. Supported: local, ssh")
    return LocalExecutionBackend(controller_workspace)

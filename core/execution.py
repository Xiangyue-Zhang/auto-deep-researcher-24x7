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
import time
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger("autoresearcher.execution")


# Directories and files that repo-reading tools (list_tree / grep_files) skip,
# so the agent sees source code instead of VCS metadata and build caches.
WALK_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".ipynb_checkpoints",
}
# grep_files skips files larger than this (likely data/binaries, not source).
GREP_MAX_FILE_BYTES = 2_000_000


# --- Slurm liveness taxonomy (used by SlurmExecutionBackend) ---
# We map a job's `sacct` State to three buckets. Reference: `man sacct`
# JOB STATE CODES. PENDING/RUNNING/etc. occupy a slot ("running"); COMPLETED
# is "completed"; the rest are "failed". PREEMPTED is intentionally ABSENT:
# under a requeue policy a preempted job returns to PENDING, so we let it fall
# through to "unknown" (bounded grace) rather than reaping it early.
_SLURM_RUNNING_STATES = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED",
    "CONFIGURING", "COMPLETING",
}
_SLURM_OK_STATES = {"COMPLETED"}
_SLURM_FAIL_STATES = {
    "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY",
    "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}


def _parse_slurm_time_seconds(spec: str) -> int:
    """Parse a Slurm ``--time`` spec to seconds.

    Accepts the documented forms: ``minutes``, ``minutes:seconds``,
    ``hours:minutes:seconds``, ``days-hours``, ``days-hours:minutes``,
    ``days-hours:minutes:seconds``. Returns a large sentinel when unparseable
    so the wall-clock liveness cap never fires spuriously (the consecutive
    -unknown grace still bounds the loop).
    """
    s = str(spec or "").strip()
    if not s:
        return 10 ** 9
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = s.split(":") if s else []
        if days:
            # days-hours[:minutes[:seconds]]
            hours = int(parts[0]) if len(parts) >= 1 else 0
            minutes = int(parts[1]) if len(parts) >= 2 else 0
            seconds = int(parts[2]) if len(parts) >= 3 else 0
        elif len(parts) == 1:
            hours, minutes, seconds = 0, int(parts[0]), 0          # bare minutes
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])  # minutes:seconds
        else:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError, IndexError):
        return 10 ** 9


REMOTE_HELPER = textwrap.dedent(
    """
    import json
    import os
    import pathlib
    import shlex
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
        resolved = (root / rel).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {raw}") from exc
        return resolved


    WALK_SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".idea", ".ipynb_checkpoints",
    }
    GREP_MAX_FILE_BYTES = 2000000


    def walk_tree(root, max_depth, max_entries):
        max_depth = max(1, int(max_depth))
        max_entries = max(1, int(max_entries))
        entries = []

        def walk(current, depth):
            if depth > max_depth or len(entries) >= max_entries:
                return
            try:
                children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
            except OSError:
                return
            for child in children:
                if len(entries) >= max_entries:
                    return
                if child.name in WALK_SKIP_DIRS:
                    continue
                if child.is_symlink():
                    continue
                rel = child.relative_to(root).as_posix()
                if child.is_dir():
                    entries.append(rel + "/")
                    walk(child, depth + 1)
                else:
                    entries.append(rel)

        walk(root, 1)
        return entries


    def grep_tree(root, base, pattern, max_results, ignore_case):
        import re
        if not pattern:
            raise ValueError("Search pattern cannot be empty")
        max_results = max(1, int(max_results))
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError("Invalid search pattern: " + str(exc))
        targets = []
        if root.is_file():
            targets = [root]
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in WALK_SKIP_DIRS)
                for name in sorted(filenames):
                    targets.append(pathlib.Path(dirpath) / name)
        hits = []
        for file_path in targets:
            if len(hits) >= max_results:
                break
            try:
                if file_path.is_symlink():
                    continue
                if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
                with open(file_path, "r", errors="strict") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        if regex.search(line):
                            hits.append({
                                "file": file_path.relative_to(base).as_posix(),
                                "line": lineno,
                                "text": line.rstrip("\\n")[:300],
                            })
                            if len(hits) >= max_results:
                                break
            except (UnicodeDecodeError, OSError, ValueError):
                continue
        return hits


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
        elif action == "read_file_range":
            path = resolve_path(root, payload["path"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {payload['path']}")
            lines = path.read_text().splitlines()
            start = max(1, int(payload.get("start_line", 1)))
            end_raw = payload.get("end_line")
            end = len(lines) if end_raw is None else min(len(lines), int(end_raw))
            if end < start:
                result = {"content": ""}
            else:
                selected = lines[start - 1:end]
                result = {"content": "\\n".join(str(start + i) + "\\t" + t for i, t in enumerate(selected))}
        elif action == "list_tree":
            raw = payload.get("path", ".")
            base = root if raw in ("", ".") else resolve_path(root, raw)
            if not base.is_dir():
                raise NotADirectoryError("Not a directory: " + str(raw))
            result = {"entries": walk_tree(base, payload.get("max_depth", 3), payload.get("max_entries", 300))}
        elif action == "grep_files":
            raw = payload.get("path", ".")
            base = root if raw in ("", ".") else resolve_path(root, raw)
            result = {"hits": grep_tree(base, root, payload["pattern"], payload.get("max_results", 50), payload.get("ignore_case", False))}
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
        elif action == "submit_slurm":
            # Build the sbatch script HERE, in Python, with shell=False — no
            # remote shell string is ever assembled from caller-supplied argv,
            # so there is no injection surface. Then `sbatch --parsable` and
            # EXIT: nothing persistent is left on the login node (the v7
            # submit-and-exit invariant). Slurm enforces --time.
            argv = payload["argv"]
            if not isinstance(argv, list) or not argv:
                raise ValueError("submit_slurm requires a non-empty argv list")
            log_file = payload["log_file"]
            log_path = resolve_path(root, log_file)        # reuses traversal guard
            log_path.parent.mkdir(parents=True, exist_ok=True)
            root.mkdir(parents=True, exist_ok=True)
            # Slurm assigns GPUs via --gres; an inherited CUDA_VISIBLE_DEVICES /
            # GPU would pin every job to the wrong physical device. Strip them.
            env = {
                k: v for k, v in (payload.get("env") or {}).items()
                if k not in ("CUDA_VISIBLE_DEVICES", "GPU")
            }
            job_name = str(payload.get("job_name") or "ar_job")
            # #SBATCH directive lines are tokenized by Slurm on whitespace
            # (honoring double quotes), NOT run through a shell — so a path with
            # spaces must be double-quoted. Strip any embedded double-quote to
            # keep quoting unambiguous (paths realistically never contain one).
            def _q(value):
                return chr(34) + str(value).replace(chr(34), "") + chr(34)
            lines = ["#!/bin/bash"]
            lines.append("#SBATCH --job-name=" + _q(job_name))
            lines.append("#SBATCH --partition=" + str(payload["partition"]))
            lines.append("#SBATCH --chdir=" + _q(str(root)))
            # --output is relative; Slurm resolves it against --chdir, matching
            # how tail_file(log_file) resolves it under the workspace root.
            lines.append("#SBATCH --output=" + _q(log_file))
            lines.append("#SBATCH --time=" + str(payload["time"]))
            raw_gres = payload.get("raw_gres") or ""
            gres = payload.get("gres")
            if raw_gres:
                lines.append("#SBATCH --gres=" + str(raw_gres))
            elif isinstance(gres, int) and gres >= 1:
                lines.append("#SBATCH --gres=gpu:" + str(gres))
            if payload.get("qos"):
                lines.append("#SBATCH --qos=" + str(payload["qos"]))
            if payload.get("account"):
                lines.append("#SBATCH --account=" + str(payload["account"]))
            for extra in (payload.get("extra_sbatch") or []):
                lines.append("#SBATCH " + str(extra))
            setup = payload.get("setup") or ""
            if setup:
                lines.append(str(setup))
            for k, v in env.items():
                lines.append("export " + str(k) + "=" + shlex.quote(str(v)))
            lines.append(" ".join(shlex.quote(str(a)) for a in argv))
            script = chr(10).join(lines) + chr(10)
            script_path = root / (".sbatch_" + job_name)
            script_path.write_text(script)
            try:
                proc = subprocess.run(
                    ["sbatch", "--parsable", str(script_path)],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(root), check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("sbatch not found on remote host: " + str(exc))
            if proc.returncode != 0:
                raise RuntimeError(
                    "sbatch failed: " + (proc.stderr or proc.stdout).strip()[:400]
                )
            token = ""
            if proc.stdout.strip():
                token = proc.stdout.strip().splitlines()[0].split(";")[0].strip()
            if not token.isdigit():
                raise RuntimeError(
                    "sbatch did not return a job id: " + proc.stdout.strip()[:200]
                )
            result = {
                "slurm_job_id": int(token),
                "log_file": log_file,
                "script_path": str(script_path),
            }
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


def _walk_tree(root: Path, base: Path, max_depth: int, max_entries: int) -> list[str]:
    """Depth-limited recursive listing relative to `base`, skipping noise dirs."""
    max_depth = max(1, int(max_depth))
    max_entries = max(1, int(max_entries))
    entries: list[str] = []

    def walk(current: Path, depth: int):
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except (PermissionError, OSError):
            return
        for child in children:
            if len(entries) >= max_entries:
                return
            if child.name in WALK_SKIP_DIRS:
                continue
            # Never follow or list symlinks: they can point outside the
            # workspace, which would defeat the sandbox enforced elsewhere.
            if child.is_symlink():
                continue
            rel = child.relative_to(base).as_posix()
            if child.is_dir():
                entries.append(rel + "/")
                walk(child, depth + 1)
            else:
                entries.append(rel)

    walk(root, 1)
    return entries


def _grep_tree(root: Path, base: Path, pattern: str, max_results: int, ignore_case: bool) -> list[dict]:
    """Scan text files under `root` for `pattern`, returning file/line/text hits."""
    import re as _re

    if not pattern:
        raise ValueError("Search pattern cannot be empty")
    max_results = max(1, int(max_results))
    flags = _re.IGNORECASE if ignore_case else 0
    try:
        regex = _re.compile(pattern, flags)
    except _re.error as exc:
        raise ValueError(f"Invalid search pattern: {exc}") from exc

    targets: list[Path] = []
    if root.is_file():
        targets = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in WALK_SKIP_DIRS)
            for name in sorted(filenames):
                targets.append(Path(dirpath) / name)

    hits: list[dict] = []
    for file_path in targets:
        if len(hits) >= max_results:
            break
        try:
            # os.walk does not descend symlinked dirs, but symlinked *files*
            # still appear and would otherwise be opened — that could read a
            # file outside the workspace. Skip any symlink target.
            if file_path.is_symlink():
                continue
            if file_path.stat().st_size > GREP_MAX_FILE_BYTES:
                continue
            with open(file_path, "r", errors="strict") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if regex.search(line):
                        hits.append(
                            {
                                "file": file_path.relative_to(base).as_posix(),
                                "line": lineno,
                                "text": line.rstrip("\n")[:300],
                            }
                        )
                        if len(hits) >= max_results:
                            break
        except (UnicodeDecodeError, PermissionError, OSError, ValueError):
            # Binary file, unreadable, or escaped base — skip silently.
            continue
    return hits


class ExecutionBackend:
    """Abstract execution backend."""

    def validate(self):
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        raise NotImplementedError

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        raise NotImplementedError

    def write_file(self, path: str, content: str) -> dict:
        raise NotImplementedError

    def list_files(self, path: str = ".") -> list[str]:
        raise NotImplementedError

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        raise NotImplementedError

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
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

    def final_status(self, pid: int) -> dict:
        """Outcome of a finished job: ``{"state": <str>, "success": <bool|None>}``.

        Default is indeterminate (``success=None``): backends that only track an
        OS pid cannot recover an exit code after the process is gone, so the
        caller keeps treating the run as "completed". The Slurm backend overrides
        this with the real ``sacct`` terminal state so FAILED / TIMEOUT / CANCELLED
        are not silently reported as success.
        """
        return {"state": "unknown", "success": None}


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

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        file_path = _resolve_under_root(self.workspace, normalize_relative_path(path))
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = file_path.read_text().splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        if end < start:
            return ""
        selected = lines[start - 1 : end]
        return "\n".join(f"{start + i}\t{text}" for i, text in enumerate(selected))

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

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        rel_path = normalize_relative_path(path)
        root = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return _walk_tree(root, root, max_depth=max_depth, max_entries=max_entries)

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
        rel_path = normalize_relative_path(path)
        root = self.workspace if rel_path == "." else _resolve_under_root(self.workspace, rel_path)
        return _grep_tree(root, self.workspace, pattern, max_results=max_results, ignore_case=ignore_case)

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

    def read_file_range(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        payload = self._invoke(
            "read_file_range",
            path=normalize_relative_path(path),
            start_line=int(start_line),
            end_line=None if end_line is None else int(end_line),
        )
        return payload["content"]

    def write_file(self, path: str, content: str) -> dict:
        return self._invoke("write_file", path=normalize_relative_path(path), content=content)

    def list_files(self, path: str = ".") -> list[str]:
        payload = self._invoke("list_files", path=normalize_relative_path(path))
        return payload["files"]

    def list_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 300) -> list[str]:
        payload = self._invoke(
            "list_tree",
            path=normalize_relative_path(path),
            max_depth=int(max_depth),
            max_entries=int(max_entries),
        )
        return payload["entries"]

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        max_results: int = 50,
        ignore_case: bool = False,
    ) -> list[dict]:
        payload = self._invoke(
            "grep_files",
            pattern=pattern,
            path=normalize_relative_path(path),
            max_results=int(max_results),
            ignore_case=bool(ignore_case),
            transport_timeout=60,
        )
        return payload["hits"]

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

    def _ssh_shell(self, remote_cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
        """Run ONE transient remote shell command, reusing this backend's host
        and ssh_args (single source of truth — no split-brain transport).

        Used by the Slurm subclass for ``sacct`` / ``squeue`` / ``scancel``, the
        only places an arbitrary remote shell string is needed. Each call runs
        one command and returns immediately; nothing persistent is started on
        the remote. The only values interpolated into these strings are
        validated integers (job ids) or operator-controlled config.
        """
        return subprocess.run(
            ["ssh", *self.ssh_args, self.ssh_host, remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

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


class SlurmExecutionBackend(SSHExecutionBackend):
    """Run experiments on a Slurm-managed cluster via a login node.

    The login node shares an NFS workspace with the compute nodes, so every
    file / repo-reading / ``run_command`` operation is inherited unchanged from
    :class:`SSHExecutionBackend` (they run on the login node over the same
    JSON-over-stdin helper transport). Only three things differ on a scheduler:

      - **launch** — instead of starting a process, submit an ``sbatch`` job
        with ``--parsable`` over ONE transient ssh call that exits immediately.
        The integer Slurm job id is returned in the ``pid`` field so the
        existing PID-keyed monitor / state.json plumbing works unchanged. No
        ``srun --wait``, no ``tmux``, no polling loop is ever left on the login
        node (the 2026-05-29 Tokyo-U MIL incident: a persistent login-node
        process is impermissible).
      - **liveness** — ``sacct`` is the sole authority while the cluster is
        reachable; the controller polls it transiently. Slurm enforces
        ``--time`` (reporting ``TIMEOUT``), so a running job always reaches a
        terminal state on its own.
      - **gpu status** — the login node has no usable ``nvidia-smi``; report
        the partition's queue occupancy from ``squeue`` instead.

    Two safeguards live INSIDE :meth:`is_process_alive` so the monitor's
    unbounded ``while is_process_alive(pid): sleep`` loop provably terminates
    even if the cluster becomes unreachable. They apply ONLY when sacct cannot
    confirm the job's state — a job sacct still reports as queued/running is
    never reaped (a long PENDING queue wait is not bounded by ``--time``):

      1. *Bounded unknown grace* — after ``slurm_unknown_grace_polls``
         consecutive indeterminate probes (ssh down / sacct purged), the job is
         declared dead.
      2. *Wall-clock backstop* — if the job is still unconfirmable once
         ``--time`` + ``slurm_time_buffer`` has elapsed since the first poll,
         it is declared dead (Slurm would have produced a terminal state by
         then for any job that actually ran).
    """

    def __init__(
        self,
        ssh_host: str,
        remote_workspace: str,
        remote_python: str = "python3",
        ssh_args: Optional[list[str]] = None,
        slurm_partition: str = "",
        slurm_time: str = "",
        slurm_gpus_per_job: Optional[int] = None,
        slurm_gres: str = "",
        slurm_qos: str = "",
        slurm_account: str = "",
        slurm_setup: str = "",
        slurm_extra_sbatch: Optional[list[str]] = None,
        slurm_unknown_grace_polls: int = 4,
        slurm_time_buffer: int = 1800,
    ):
        super().__init__(ssh_host, remote_workspace, remote_python, ssh_args)
        self.slurm_partition = slurm_partition
        self.slurm_time = slurm_time
        self.slurm_gpus_per_job = slurm_gpus_per_job
        self.slurm_gres = slurm_gres
        self.slurm_qos = slurm_qos
        self.slurm_account = slurm_account
        self.slurm_setup = slurm_setup
        self.slurm_extra_sbatch = list(slurm_extra_sbatch or [])
        self.slurm_unknown_grace_polls = int(slurm_unknown_grace_polls)
        self.slurm_time_buffer = int(slurm_time_buffer)
        self._time_cap_seconds = _parse_slurm_time_seconds(slurm_time)
        # Per-job liveness state, keyed by Slurm job id.
        self._first_seen: dict[int, float] = {}
        self._unknown_count: dict[int, int] = {}
        self._last_terminal: dict[int, str] = {}

    def validate(self):
        if not self.ssh_host:
            raise ValueError("execution.ssh_host is required when execution.mode=slurm")
        if not self.remote_workspace:
            raise ValueError("execution.remote_workspace is required when execution.mode=slurm")
        if not self.slurm_partition:
            raise ValueError("execution.slurm_partition is required when execution.mode=slurm")
        if not self.slurm_time:
            raise ValueError("execution.slurm_time is required when execution.mode=slurm")
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh binary not found on PATH")
        # Workspace reachable + remote python OK (inherited helper transport).
        self._invoke("validate", transport_timeout=30)
        # Require ALL three tools: `command -v a b c` succeeds if ANY one
        # resolves, so chain a check per tool.
        probe = self._ssh_shell(
            "command -v sbatch >/dev/null 2>&1 "
            "&& command -v sacct >/dev/null 2>&1 "
            "&& command -v squeue >/dev/null 2>&1 && echo OK",
            timeout=15,
        )
        if probe.returncode != 0 or "OK" not in (probe.stdout or ""):
            raise RuntimeError(
                "Slurm tools (sbatch/sacct/squeue) not found on the login node; "
                "is execution.ssh_host a Slurm submit host?"
            )

    def launch_command(self, argv: list[str], log_file: str, env: Optional[dict] = None) -> dict:
        normalized_log = normalize_relative_path(log_file)
        job_name = "ar_" + (Path(normalized_log).stem or "job")
        payload = self._invoke(
            "submit_slurm",
            argv=list(argv),
            log_file=normalized_log,
            env=env or {},                       # remote helper strips CUDA_VISIBLE_DEVICES/GPU
            partition=self.slurm_partition,
            time=self.slurm_time,
            gres=self.slurm_gpus_per_job,
            raw_gres=self.slurm_gres,
            qos=self.slurm_qos,
            account=self.slurm_account,
            job_name=job_name,
            setup=self.slurm_setup,
            extra_sbatch=list(self.slurm_extra_sbatch),
            transport_timeout=90,
        )
        job_id = int(payload["slurm_job_id"])
        # `pid` carries the Slurm job id so the existing monitor / state.json /
        # obsidian plumbing (which keys on `pid`) works without changes.
        return {
            "pid": job_id,
            "slurm_job_id": job_id,
            "log_file": payload.get("log_file", normalized_log),
            "status": "submitted",
        }

    def _sacct_state(self, job_id: int) -> tuple[str, str]:
        """Return (bucket, raw_state) for a Slurm job; bucket in
        {running, completed, failed, unknown}. One transient sacct query, with
        a squeue fallback for a job too new / already purged from accounting."""
        cmd = f"sacct -j {int(job_id)} --format=State%30 -X -n -P 2>/dev/null | head -1"
        try:
            r = self._ssh_shell(cmd, timeout=15)
        except (subprocess.TimeoutExpired, OSError):
            return "unknown", "ssh_failed"
        if r.returncode != 0:
            return "unknown", f"sacct_rc={r.returncode}"
        out = (r.stdout or "").strip()
        # split()[0] drops a trailing " by <uid>" (e.g. "CANCELLED by 1001");
        # .replace("+","") strips the "CANCELLED+" suffix Slurm appends.
        raw = out.split()[0].replace("+", "").upper() if out else ""
        if not raw:
            sq = f"squeue -j {int(job_id)} -h -o '%T' 2>/dev/null | head -1"
            try:
                r2 = self._ssh_shell(sq, timeout=15)
                raw = (r2.stdout or "").strip().upper()
            except (subprocess.TimeoutExpired, OSError):
                raw = ""
            if not raw:
                return "unknown", "sacct_empty"
        if raw in _SLURM_RUNNING_STATES:
            return "running", raw
        if raw in _SLURM_OK_STATES:
            return "completed", raw
        if raw in _SLURM_FAIL_STATES:
            return "failed", raw
        return "unknown", raw

    def is_process_alive(self, pid: int) -> bool:
        """Alive iff the Slurm job is in a running-bucket state. Indeterminate
        probes keep the job alive only for a bounded number of consecutive
        polls; a job is also force-reaped past ``--time`` + buffer. Both bounds
        guarantee the monitor's polling loop always terminates."""
        job_id = int(pid)
        now = time.time()
        first = self._first_seen.setdefault(job_id, now)
        bucket, raw = self._sacct_state(job_id)
        if bucket == "running":
            # PENDING/RUNNING/etc. are authoritative. A long queue wait is NOT
            # bounded by --time (which only counts while RUNNING), so never reap
            # a job sacct still confirms is queued or running.
            self._unknown_count[job_id] = 0
            return True
        if bucket in ("completed", "failed"):
            self._last_terminal[job_id] = raw
            return False
        # Indeterminate (ssh/sacct unreachable, or the job purged from both
        # sacct and squeue). Two bounds keep the monitor's polling loop finite
        # WITHOUT ever reaping a job sacct confirms is live:
        #   - a wall-clock backstop: Slurm enforces --time, so once --time +
        #     buffer has elapsed and we STILL cannot confirm the job, it is
        #     almost certainly gone;
        #   - a consecutive-unknown grace for shorter outages.
        if now - first > self._time_cap_seconds + self.slurm_time_buffer:
            return False
        self._unknown_count[job_id] = self._unknown_count.get(job_id, 0) + 1
        return self._unknown_count[job_id] <= self.slurm_unknown_grace_polls

    def get_gpu_status(self) -> dict:
        """Report the partition's queue occupancy (login node has no usable
        nvidia-smi). Advisory only — the monitor just logs ``utilization``."""
        cmd = (
            "squeue --me -p " + shlex.quote(self.slurm_partition)
            + " --states=PD,R -h -o '%T' 2>/dev/null | sort | uniq -c"
        )
        pending = running = 0
        try:
            r = self._ssh_shell(cmd, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            return {
                "utilization": "slurm", "partition": self.slurm_partition,
                "pending": 0, "running": 0, "note": "squeue unavailable",
            }
        if r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    count, state = int(parts[0]), parts[1].upper()
                    if state.startswith("PEND") or state == "PD":
                        pending = count
                    elif state.startswith("R"):
                        running = count
        return {
            "utilization": "slurm", "partition": self.slurm_partition,
            "pending": pending, "running": running,
        }

    def cancel(self, pid: int) -> bool:
        """Best-effort ``scancel`` for a Slurm job. Not wired into a caller yet
        (orphaned jobs are otherwise reclaimed by ``--time``); available for a
        future kill-on-shutdown path."""
        try:
            r = self._ssh_shell("scancel " + str(int(pid)), timeout=8)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def last_terminal_state(self, pid: int) -> Optional[str]:
        """Raw sacct state of a finished job, if observed (e.g. ``TIMEOUT``)."""
        return self._last_terminal.get(int(pid))

    def final_status(self, pid: int) -> dict:
        """Real outcome from the observed ``sacct`` terminal state.

        ``success`` is True only for ``COMPLETED``; ``FAILED`` / ``TIMEOUT`` /
        ``CANCELLED`` / ``OUT_OF_MEMORY`` / … are reported as failures. If the
        job was never observed reaching a terminal state (e.g. the cluster went
        unreachable and it was reaped by the wall-clock backstop), the outcome
        is indeterminate.
        """
        raw = self._last_terminal.get(int(pid))
        if raw is None:
            return {"state": "unknown", "success": None}
        return {"state": raw, "success": raw in _SLURM_OK_STATES}


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
    if mode == "slurm":
        return SlurmExecutionBackend(
            ssh_host=execution.get("ssh_host", ""),
            remote_workspace=execution.get("remote_workspace", ""),
            remote_python=execution.get("remote_python", "python3"),
            ssh_args=execution.get("ssh_args", []) or [],
            slurm_partition=execution.get("slurm_partition", ""),
            slurm_time=execution.get("slurm_time", ""),
            slurm_gpus_per_job=execution.get("slurm_gpus_per_job"),
            slurm_gres=execution.get("slurm_gres", ""),
            slurm_qos=execution.get("slurm_qos", ""),
            slurm_account=execution.get("slurm_account", ""),
            slurm_setup=execution.get("slurm_setup", ""),
            slurm_extra_sbatch=execution.get("slurm_extra_sbatch", []) or [],
            slurm_unknown_grace_polls=int(execution.get("slurm_unknown_grace_polls", 4)),
            slurm_time_buffer=int(execution.get("slurm_time_buffer", 1800)),
        )
    if mode != "local":
        raise ValueError(f"Unknown execution.mode '{mode}'. Supported: local, ssh, slurm")
    return LocalExecutionBackend(controller_workspace)

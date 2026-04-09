"""
AutoResearcher Tool Registry

Each agent gets a minimal tool set (3-5 tools) instead of all tools.
This reduces token overhead per API call significantly.
"""

import os
import subprocess
import json
import logging
import shlex
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.tools")


class ToolRegistry:
    """Manages tools available to agents.

    Design principle: minimal tool sets per agent.
    - Leader: log_memory, write_file, read_file, send_email (4 tools)
    - Idea Agent: search_papers, get_paper, write_file, read_file (4 tools)
    - Code Agent: run_shell, launch_experiment, write_file, read_file, list_files (5 tools)
    - Writing Agent: write_file, read_file, list_files (3 tools)

    Fewer tools = fewer tokens in each API call = lower cost.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self._protected_files = {"state.json", "MEMORY_LOG.md", "PROJECT_BRIEF.md", ".lock"}

    def get_tools_for(self, agent_type: str) -> list[dict]:
        """Get tool definitions for a specific agent type."""
        tool_map = {
            "leader": [self._tool_log_memory, self._tool_write_file, self._tool_read_file],
            "idea": [self._tool_search_papers, self._tool_write_file, self._tool_read_file],
            "code": [
                self._tool_run_shell,
                self._tool_launch_experiment,
                self._tool_write_file,
                self._tool_read_file,
                self._tool_list_files,
            ],
            "writing": [self._tool_write_file, self._tool_read_file, self._tool_list_files],
        }
        return tool_map.get(agent_type, [])

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name and return the result."""
        handlers = {
            "run_shell": self._exec_run_shell,
            "launch_experiment": self._exec_launch_experiment,
            "write_file": self._exec_write_file,
            "read_file": self._exec_read_file,
            "list_files": self._exec_list_files,
            "search_papers": self._exec_search_papers,
            "log_memory": self._exec_log_memory,
        }

        handler = handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            return handler(**args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return json.dumps({"error": str(e)})

    # --- Tool Definitions (for API schema) ---

    @property
    def _tool_run_shell(self) -> dict:
        return {
            "name": "run_shell",
            "description": "Run a shell command and return output. Use for quick checks, file ops, git commands. For long-running training, use launch_experiment instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)", "default": 120},
                },
                "required": ["command"],
            },
        }

    @property
    def _tool_launch_experiment(self) -> dict:
        return {
            "name": "launch_experiment",
            "description": "Launch a long-running experiment via nohup. Returns PID for monitoring. Use this for training runs, not run_shell.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Training command to run"},
                    "log_file": {"type": "string", "description": "Path for stdout/stderr log"},
                    "gpu": {"type": "string", "description": "CUDA_VISIBLE_DEVICES value"},
                },
                "required": ["command", "log_file"],
            },
        }

    @property
    def _tool_write_file(self) -> dict:
        return {
            "name": "write_file",
            "description": "Write content to a file. Cannot overwrite protected files (state.json, MEMORY_LOG.md, PROJECT_BRIEF.md).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        }

    @property
    def _tool_read_file(self) -> dict:
        return {
            "name": "read_file",
            "description": "Read a file's contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                },
                "required": ["path"],
            },
        }

    @property
    def _tool_list_files(self) -> dict:
        return {
            "name": "list_files",
            "description": "List files in a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
                },
            },
        }

    @property
    def _tool_search_papers(self) -> dict:
        return {
            "name": "search_papers",
            "description": "Search for academic papers via Semantic Scholar API.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                    "year": {"type": "string", "description": "Year filter, e.g. '2024-2026'"},
                },
                "required": ["query"],
            },
        }

    @property
    def _tool_log_memory(self) -> dict:
        return {
            "name": "log_memory",
            "description": "Log an entry to the memory system. Use 'milestone' for key results, 'decision' for routine decisions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["milestone", "decision"]},
                    "entry": {"type": "string", "description": "Content to log"},
                },
                "required": ["type", "entry"],
            },
        }

    # --- Tool Implementations ---

    def _resolve_workspace_path(self, path: str) -> Path:
        """Resolve a user-supplied relative path and keep it inside the workspace."""
        if path is None or not str(path).strip():
            raise ValueError("Path cannot be empty")

        requested = Path(path)
        if requested.is_absolute():
            raise ValueError("Path must be relative to workspace")

        resolved = (self.workspace / requested).resolve(strict=False)

        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path}") from exc

        return resolved

    def _parse_command(self, command: str) -> list[str]:
        """Parse command text into argv without invoking a shell."""
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Invalid command syntax: {exc}") from exc

        if not argv:
            raise ValueError("Command cannot be empty")

        dangerous_bins = {
            "rm",
            "sudo",
            "su",
            "mkfs",
            "dd",
            "shutdown",
            "reboot",
            "poweroff",
            "halt",
        }
        if Path(argv[0]).name in dangerous_bins:
            raise ValueError(f"Blocked executable: {argv[0]}")

        return argv

    def _exec_run_shell(self, command: str, timeout: int = 120) -> str:
        """Execute a shell command with timeout."""
        try:
            argv = self._parse_command(command)
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            return json.dumps({
                "stdout": result.stdout[-2000:],  # Cap output
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Command timed out after {timeout}s"})

    def _exec_launch_experiment(self, command: str, log_file: str, gpu: str = None) -> str:
        """Launch experiment via nohup."""
        env = os.environ.copy()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu

        argv = self._parse_command(command)
        log_path = self._resolve_workspace_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                argv,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                cwd=str(self.workspace),
            )

        return json.dumps({"pid": proc.pid, "log_file": str(log_path), "status": "launched"})

    def _exec_write_file(self, path: str, content: str) -> str:
        """Write file with protection check."""
        if Path(path).name in self._protected_files:
            return json.dumps({"error": f"Cannot overwrite protected file: {path}"})

        file_path = self._resolve_workspace_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return json.dumps({"status": "written", "path": str(file_path), "bytes": len(content)})

    def _exec_read_file(self, path: str) -> str:
        """Read file contents."""
        file_path = self._resolve_workspace_path(path)
        if not file_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        content = file_path.read_text()
        return content[:10000]  # Cap at 10K chars

    def _exec_list_files(self, path: str = ".") -> str:
        """List directory contents."""
        dir_path = self._resolve_workspace_path(path)
        if not dir_path.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"})
        files = sorted([f.name for f in dir_path.iterdir()])
        return json.dumps({"files": files[:100]})  # Cap at 100 entries

    def _exec_search_papers(self, query: str, limit: int = 10, year: str = None) -> str:
        """Search Semantic Scholar."""
        import urllib.request
        import urllib.parse

        params = {"query": query, "limit": limit, "fields": "title,year,authors,abstract,citationCount,url"}
        if year:
            params["year"] = year

        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AutoResearcher/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                papers = data.get("data", [])
                return json.dumps({"papers": papers[:limit]}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    def _exec_log_memory(self, type: str, entry: str) -> str:
        """Log to memory (delegated to MemoryManager)."""
        return json.dumps({"status": "logged", "type": type, "entry": entry[:200]})

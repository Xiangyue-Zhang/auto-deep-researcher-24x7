"""
Obsidian export helpers for Deep Researcher Agent.

Turns current project state into Obsidian-friendly Markdown:
- Dashboard.md: current snapshot, overwritten on refresh
- Daily/YYYY-MM-DD.md: append-only daily cycle notes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import yaml

from .execution import ExecutionBackend, LocalExecutionBackend, build_execution_backend
from .memory import MemoryManager


class ObsidianExporter:
    """Project-level Obsidian Markdown exporter."""

    def __init__(
        self,
        config: dict,
        project_dir: str | Path,
        backend: Optional[ExecutionBackend] = None,
    ):
        self.config = config or {}
        self.project_dir = Path(project_dir).resolve()
        self.project_name = self.project_dir.name
        self.workspace = self.project_dir / self.config.get("project", {}).get("workspace", "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_path = self.workspace / "state.json"
        self.backend = backend or LocalExecutionBackend(self.workspace)

        self.obsidian_config = self.config.get("obsidian", {})
        self.enabled = bool(self.obsidian_config.get("enabled", False))
        self.vault_path = self.obsidian_config.get("vault_path", "")
        self.project_subdir = self.obsidian_config.get("project_subdir", "DeepResearcher/{project_name}")
        self.dashboard_note = self.obsidian_config.get("dashboard_note", "Dashboard.md")
        self.daily_dir = self.obsidian_config.get("daily_dir", "Daily")
        self.auto_append_daily = bool(self.obsidian_config.get("auto_append_daily", True))
        self.local_fallback_dir = self.obsidian_config.get("local_fallback_dir", "progress_tracking")

    def is_enabled(self) -> bool:
        return self.enabled

    def refresh_all(self, memory: MemoryManager, cycle_count: int) -> dict:
        if not self.is_enabled():
            return {"status": "disabled"}

        dashboard = self.refresh_dashboard(memory=memory, cycle_count=cycle_count)
        daily = self.append_daily_entry(memory=memory, cycle_count=cycle_count, event_type="manual_refresh")
        return {"status": "ok", "dashboard": dashboard, "daily": daily}

    def refresh_dashboard(self, memory: MemoryManager, cycle_count: int) -> dict:
        if not self.is_enabled():
            return {"status": "disabled"}

        base_dir = self._base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = base_dir / self._dashboard_filename()
        state = self._load_state()
        dashboard_path.write_text(self._render_dashboard(memory=memory, state=state, cycle_count=cycle_count))
        return {"status": "written", "path": str(dashboard_path)}

    def append_daily_entry(
        self,
        memory: MemoryManager,
        cycle_count: int,
        event_type: str = "cycle_complete",
        reflection: Optional[dict] = None,
        directive: Optional[str] = None,
    ) -> dict:
        if not self.is_enabled() or (event_type == "cycle_complete" and not self.auto_append_daily):
            return {"status": "disabled"}

        base_dir = self._base_dir()
        daily_path = base_dir / self.daily_dir / self._daily_filename()
        daily_path.parent.mkdir(parents=True, exist_ok=True)

        state = self._load_state()
        entry = self._render_daily_entry(
            memory=memory,
            state=state,
            cycle_count=cycle_count,
            event_type=event_type,
            reflection=reflection or {},
            directive=directive,
        )

        if daily_path.exists():
            existing = daily_path.read_text().rstrip()
            daily_path.write_text(f"{existing}\n\n{entry}\n")
        else:
            header = f"# {self.project_name} — Daily Log — {time.strftime('%Y-%m-%d')}\n\n"
            daily_path.write_text(f"{header}{entry}\n")

        return {"status": "written", "path": str(daily_path)}

    def _base_dir(self) -> Path:
        if self.vault_path:
            subdir = self.project_subdir.format(project_name=self.project_name)
            return Path(self.vault_path).expanduser() / subdir
        return self.workspace / self.local_fallback_dir

    def _dashboard_filename(self) -> str:
        return self.dashboard_note if self.vault_path else "Dashboard.txt"

    def _daily_filename(self) -> str:
        return f"{time.strftime('%Y-%m-%d')}.md" if self.vault_path else f"{time.strftime('%Y-%m-%d')}.txt"

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _parse_log_sections(self, memory: MemoryManager) -> tuple[list[str], list[str]]:
        milestones: list[str] = []
        decisions: list[str] = []
        current = None
        for line in memory.get_log().splitlines():
            stripped = line.strip()
            if stripped == "## Key Results":
                current = "milestones"
            elif stripped == "## Recent Decisions":
                current = "decisions"
            elif stripped.startswith("["):
                if current == "milestones":
                    milestones.append(stripped)
                elif current == "decisions":
                    decisions.append(stripped)
        return milestones, decisions

    def _read_pending_directive(self) -> str:
        directive_path = self.workspace / "HUMAN_DIRECTIVE.md"
        if directive_path.exists():
            return directive_path.read_text().strip()
        return ""

    def _read_log_tail(self, log_file: str, lines: int = 8) -> str:
        if not log_file:
            return ""
        try:
            return "\n".join(self.backend.tail_file(log_file, lines=lines))
        except Exception:
            path = Path(log_file)
            if path.is_absolute() and path.exists():
                return "\n".join(path.read_text().splitlines()[-lines:])
        return ""

    def _pid_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            return self.backend.is_process_alive(int(pid))
        except Exception:
            return False

    def _format_status(self, state: dict) -> str:
        status = state.get("status", "idle")
        pid = state.get("pid")
        if status == "running" and self._pid_alive(pid):
            started_at = state.get("started_at")
            elapsed = ""
            if started_at:
                elapsed = f", {((time.time() - float(started_at)) / 3600):.1f}h"
            return f"TRAINING (PID {pid}{elapsed})"
        if status == "completed":
            return "COMPLETED"
        if status == "error":
            return "ERROR"
        return "IDLE"

    def _render_dashboard(self, memory: MemoryManager, state: dict, cycle_count: int) -> str:
        milestones, decisions = self._parse_log_sections(memory)
        pending_directive = self._read_pending_directive()
        best_result = milestones[-1] if milestones else "None yet"
        latest_decisions = decisions[-3:] if decisions else []
        log_tail = state.get("last_training_logs") or self._read_log_tail(state.get("log_file", ""))
        latest_snapshot = log_tail or "No active or recent training log."
        suggested_next = state.get("suggested_next_step") or (latest_decisions[-1] if latest_decisions else "Continue with current research direction.")

        lines = [
            f"# {self.project_name} Dashboard",
            "",
            f"_Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            f"- Output target: {'Obsidian vault' if self.vault_path else 'project-local text fallback'}",
            "",
            "## Project",
            f"- Name: {self.project_name}",
            f"- Path: `{self.project_dir}`",
            "",
            "## Goal",
            memory.get_brief().strip() or "PROJECT_BRIEF.md is empty.",
            "",
            "## Current Status",
            f"- Status: {self._format_status(state)}",
            f"- Cycles completed: {cycle_count}",
            "",
            "## Best Result",
            f"- {best_result}",
            "",
            "## Latest Training Snapshot",
            "```text",
            latest_snapshot,
            "```",
            "",
            "## Recent Decisions",
        ]

        if latest_decisions:
            lines.extend([f"- {entry}" for entry in latest_decisions])
        else:
            lines.append("- None yet")

        lines.extend(
            [
                "",
                "## Pending Directive",
                pending_directive if pending_directive else "None",
                "",
                "## Suggested Next Step",
                suggested_next,
                "",
            ]
        )
        return "\n".join(lines)

    def _render_daily_entry(
        self,
        memory: MemoryManager,
        state: dict,
        cycle_count: int,
        event_type: str,
        reflection: dict,
        directive: Optional[str],
    ) -> str:
        milestones, decisions = self._parse_log_sections(memory)
        latest_metric = state.get("last_metrics", {})
        latest_metric_text = ", ".join(f"{k}={v}" for k, v in latest_metric.items()) if latest_metric else "none"
        last_milestone = reflection.get("milestone") or state.get("last_milestone") or (milestones[-1] if milestones else "none")
        last_decision = reflection.get("decision") or state.get("last_decision") or (decisions[-1] if decisions else "none")
        blocker = state.get("last_error") or "none"
        consumed = directive or state.get("last_directive") or "none"

        lines = [
            f"## {time.strftime('%H:%M:%S')} — Cycle {cycle_count} ({event_type})",
            "",
            f"- Status: {self._format_status(state)}",
            f"- Best/new result: {last_milestone}",
            f"- Metrics: {latest_metric_text}",
            f"- Decision: {last_decision}",
            f"- Directive consumed: {consumed}",
            f"- Blocker: {blocker}",
        ]
        return "\n".join(lines)


def _load_config(project: Path, config_name: str) -> dict:
    config_path = project / config_name
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Refresh Obsidian notes from Deep Researcher project state")
    parser.add_argument("--project", type=str, required=True, help="Path to project directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Project config file")
    parser.add_argument("--dashboard-only", action="store_true", help="Only refresh Dashboard.md")
    parser.add_argument("--daily-only", action="store_true", help="Only append daily note")
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    config = _load_config(project_dir, args.config)
    memory = MemoryManager(
        project_dir=project_dir,
        brief_max=config.get("memory", {}).get("brief_max_chars", 3000),
        log_max=config.get("memory", {}).get("log_max_chars", 2000),
        milestone_max=config.get("memory", {}).get("milestone_max_chars", 1200),
        max_recent=config.get("memory", {}).get("max_recent_entries", 15),
    )
    backend = build_execution_backend(
        config=config,
        controller_workspace=project_dir / config.get("project", {}).get("workspace", "workspace"),
    )
    exporter = ObsidianExporter(config=config, project_dir=project_dir, backend=backend)
    cycle_path = project_dir / config.get("project", {}).get("workspace", "workspace") / ".cycle_counter"
    cycle_count = int(cycle_path.read_text().strip()) if cycle_path.exists() else 0

    if not exporter.is_enabled():
        print("Progress export disabled. Set obsidian.enabled=true in project config.")
        return

    if args.dashboard_only:
        result = exporter.refresh_dashboard(memory=memory, cycle_count=cycle_count)
    elif args.daily_only:
        result = exporter.append_daily_entry(memory=memory, cycle_count=cycle_count, event_type="manual_refresh")
    else:
        result = exporter.refresh_all(memory=memory, cycle_count=cycle_count)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

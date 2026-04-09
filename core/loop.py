"""
AutoResearcher Core Loop

The autonomous THINK → EXECUTE → REFLECT cycle that drives experiments 24/7.
"""

import os
import sys
import time
import json
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional

from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .agents import AgentDispatcher
from .obsidian import ObsidianExporter
from .tools import ToolRegistry

logger = logging.getLogger("autoresearcher")


class ResearchLoop:
    """Main autonomous research loop.

    Implements the THINK → EXECUTE → REFLECT cycle:
    - THINK: Analyze state, form hypothesis, plan experiment
    - EXECUTE: Dispatch code agent to implement and run experiment
    - REFLECT: Evaluate results, update memory, decide next action
    """

    def __init__(self, config: dict, project_dir: str):
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        self.workspace = self.project_dir / config.get("project", {}).get("workspace", "workspace")
        self.workspace.mkdir(exist_ok=True)
        self.state_path = self.workspace / "state.json"

        # Core components
        self.memory = MemoryManager(
            project_dir=self.project_dir,
            brief_max=config.get("memory", {}).get("brief_max_chars", 3000),
            log_max=config.get("memory", {}).get("log_max_chars", 2000),
            milestone_max=config.get("memory", {}).get("milestone_max_chars", 1200),
            max_recent=config.get("memory", {}).get("max_recent_entries", 15),
        )
        self.monitor = ExperimentMonitor(
            poll_interval=config.get("monitor", {}).get("poll_interval", 900),
            zero_llm=config.get("monitor", {}).get("zero_llm", True),
        )
        self.dispatcher = AgentDispatcher(
            model=config.get("agent", {}).get("model", "claude-sonnet-4-6"),
            provider=config.get("agent", {}).get("provider", "anthropic"),
            max_steps=config.get("agent", {}).get("max_steps_per_cycle", 3),
        )
        self.tools = ToolRegistry(self.workspace)
        self.obsidian = ObsidianExporter(config=config, project_dir=self.project_dir)

        # State
        self.cycle_count = self._load_cycle_counter()
        self.max_cycles = config.get("agent", {}).get("max_cycles", -1)
        self.cooldown = config.get("agent", {}).get("cooldown_interval", 300)
        self.no_progress_fallback_threshold = config.get("agent", {}).get("no_progress_fallback_threshold", 3)
        self._running = True
        self._no_progress_streak = 0
        self._last_no_progress_signature = ""

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def run(self):
        """Main entry point. Runs the THINK → EXECUTE → REFLECT loop."""
        logger.info(f"AutoResearcher starting | project={self.project_dir} | cycle={self.cycle_count}")

        while self._running:
            if self.max_cycles > 0 and self.cycle_count >= self.max_cycles:
                logger.info(f"Reached max cycles ({self.max_cycles}). Stopping.")
                break

            self.cycle_count += 1
            self._save_cycle_counter()
            logger.info(f"=== Cycle {self.cycle_count} ===")

            try:
                # Keep leader context bounded to one cycle.
                self.dispatcher.reset_leader_history()

                # Check for human directive
                directive = self._consume_directive()
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "status": "planning",
                        "updated_at": time.time(),
                        "last_directive": directive or "",
                    }
                )

                # THINK: Analyze and plan
                think_result = self._think(directive)
                think_result = self._apply_no_progress_fallback(think_result, directive)

                if think_result.get("action") == "wait":
                    logger.info("THINK decided to wait. Entering cooldown.")
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "status": "waiting",
                            "updated_at": time.time(),
                            "suggested_next_step": think_result.get("reason", ""),
                        }
                    )
                    self._smart_cooldown()
                    continue

                # EXECUTE: Run the plan
                execute_result = self._execute(think_result)

                if execute_result.get("experiment_launched"):
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "status": "running",
                            "pid": execute_result.get("pid"),
                            "log_file": execute_result.get("log_file", ""),
                            "started_at": time.time(),
                            "updated_at": time.time(),
                        }
                    )
                    # Monitor experiment (zero LLM cost)
                    monitor_result = self._monitor_experiment(execute_result)
                    execute_result["training_logs"] = monitor_result.get("log_tail", "")
                    execute_result["final_metrics"] = monitor_result.get("metrics", {})
                    self._update_state(
                        {
                            "status": "completed",
                            "pid": execute_result.get("pid"),
                            "log_file": execute_result.get("log_file", ""),
                            "updated_at": time.time(),
                            "last_training_logs": monitor_result.get("log_tail", ""),
                            "last_metrics": monitor_result.get("metrics", {}),
                            "elapsed_hours": monitor_result.get("elapsed_hours"),
                        }
                    )

                # REFLECT: Evaluate and update
                reflect_result = self._reflect(execute_result)
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "updated_at": time.time(),
                        "last_milestone": reflect_result.get("milestone", ""),
                        "last_decision": reflect_result.get("decision", ""),
                        "suggested_next_step": reflect_result.get("decision")
                        or reflect_result.get("reason")
                        or reflect_result.get("task", ""),
                        "last_error": "",
                    }
                )
                self._record_cycle_outcome(think_result, execute_result, reflect_result)
                self._refresh_obsidian(reflect_result=reflect_result, directive=directive)

            except Exception as e:
                logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
                self.memory.log_decision(f"Cycle {self.cycle_count} error: {str(e)[:200]}")
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "status": "error",
                        "updated_at": time.time(),
                        "last_error": str(e)[:500],
                    }
                )
                self._cooldown_after_error()

        logger.info("AutoResearcher stopped.")

    def _think(self, directive: Optional[str] = None) -> dict:
        """THINK phase: analyze current state and plan next experiment."""
        logger.info("THINK phase starting...")

        context = {
            "brief": self.memory.get_brief(),
            "memory_log": self.memory.get_log(),
            "cycle": self.cycle_count,
            "directive": directive,
        }

        result = self.dispatcher.dispatch_leader(
            task="think",
            context=context,
        )

        logger.info(f"THINK result: action={result.get('action', 'unknown')}")
        return result

    def _execute(self, plan: dict) -> dict:
        """EXECUTE phase: implement and run the planned experiment."""
        logger.info("EXECUTE phase starting...")

        agent_type = plan.get("agent", "code")
        task_description = plan.get("task", "")

        result = self.dispatcher.dispatch_worker(
            agent_type=agent_type,
            task=task_description,
            tools=self.tools.get_tools_for(agent_type),
        )

        return result

    def _monitor_experiment(self, execute_result: dict) -> dict:
        """Monitor running experiment with ZERO LLM calls."""
        pid = execute_result.get("pid")
        log_file = execute_result.get("log_file")

        if not pid:
            return {"status": "no_pid"}

        logger.info(f"Monitoring experiment PID={pid}, log={log_file}")
        return self.monitor.wait_for_completion(
            pid=pid,
            log_file=log_file,
            notify=self.config.get("monitor", {}).get("notify_on_complete", True),
        )

    def _reflect(self, execute_result: dict) -> dict:
        """REFLECT phase: evaluate results and update memory."""
        logger.info("REFLECT phase starting...")

        context = {
            "brief": self.memory.get_brief(),
            "memory_log": self.memory.get_log(),
            "experiment_result": execute_result,
            "cycle": self.cycle_count,
        }

        result = self.dispatcher.dispatch_leader(
            task="reflect",
            context=context,
        )

        # Update memory based on reflection
        if result.get("milestone"):
            self.memory.log_milestone(result["milestone"])
        if result.get("decision"):
            self.memory.log_decision(result["decision"])

        return result

    def _refresh_obsidian(self, reflect_result: dict, directive: Optional[str]):
        if not self.obsidian.is_enabled():
            return
        self.obsidian.refresh_dashboard(memory=self.memory, cycle_count=self.cycle_count)
        self.obsidian.append_daily_entry(
            memory=self.memory,
            cycle_count=self.cycle_count,
            event_type="cycle_complete",
            reflection=reflect_result,
            directive=directive,
        )

    def _plan_signature(self, plan: dict) -> str:
        """Build a stable signature for repeated-plan detection."""
        normalized = {
            "action": plan.get("action", ""),
            "agent": plan.get("agent", ""),
            "task": " ".join(plan.get("task", "").split())[:300],
            "hypothesis": " ".join(plan.get("hypothesis", "").split())[:200],
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    def _apply_no_progress_fallback(self, think_result: dict, directive: Optional[str]) -> dict:
        """Back off if the same experiment plan keeps repeating without progress."""
        if directive or self.no_progress_fallback_threshold <= 0:
            return think_result

        if think_result.get("action") != "experiment":
            return think_result

        signature = self._plan_signature(think_result)
        if (
            self._no_progress_streak >= self.no_progress_fallback_threshold
            and signature == self._last_no_progress_signature
        ):
            reason = (
                f"Fallback triggered after {self._no_progress_streak} no-progress cycles on the same plan. "
                "Backing off to avoid empty loops until new signal arrives."
            )
            logger.warning(reason)
            self.memory.log_decision(reason)
            return {
                "action": "wait",
                "reason": reason,
                "decision": reason,
            }

        return think_result

    def _record_cycle_outcome(self, think_result: dict, execute_result: dict, reflect_result: dict):
        """Track whether repeated cycles are producing real progress."""
        if think_result.get("action") != "experiment":
            if think_result.get("action") != "wait":
                self._no_progress_streak = 0
                self._last_no_progress_signature = ""
            return

        signature = self._plan_signature(think_result)
        made_progress = bool(
            execute_result.get("experiment_launched")
            or execute_result.get("final_metrics")
            or reflect_result.get("milestone")
        )

        if made_progress:
            self._no_progress_streak = 0
            self._last_no_progress_signature = ""
            return

        if signature == self._last_no_progress_signature:
            self._no_progress_streak += 1
        else:
            self._last_no_progress_signature = signature
            self._no_progress_streak = 1

    def _smart_cooldown(self):
        """Poll at short intervals instead of fixed long wait."""
        logger.info(f"Smart cooldown: polling every {self.cooldown}s")
        elapsed = 0
        while elapsed < self.cooldown and self._running:
            time.sleep(min(60, self.cooldown - elapsed))
            elapsed += 60

            # Check if any experiment just finished
            if self.monitor.has_completed_experiments():
                logger.info("Experiment completed during cooldown. Waking up.")
                return

    def _cooldown_after_error(self):
        """Back off after an error to prevent burn loops."""
        backoff = min(self.cooldown * 2, 1800)  # Max 30 min
        logger.warning(f"Error backoff: waiting {backoff}s")
        time.sleep(backoff)

    def _consume_directive(self) -> Optional[str]:
        """Read and consume HUMAN_DIRECTIVE.md if present."""
        directive_path = self.workspace / "HUMAN_DIRECTIVE.md"
        if directive_path.exists():
            content = directive_path.read_text().strip()
            if content:
                # Archive the directive
                archive_dir = self.workspace / "directive_archive"
                archive_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                directive_path.rename(archive_dir / f"directive_{timestamp}.md")
                logger.info(f"Consumed directive: {content[:100]}...")
                return content
        return None

    def _load_cycle_counter(self) -> int:
        counter_file = self.workspace / ".cycle_counter"
        if counter_file.exists():
            return int(counter_file.read_text().strip())
        return 0

    def _save_cycle_counter(self):
        counter_file = self.workspace / ".cycle_counter"
        counter_file.write_text(str(self.cycle_count))

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _update_state(self, updates: dict):
        state = self._load_state()
        state.update(updates)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="AutoResearcher - Autonomous ML Experiment Agent")
    parser.add_argument("--project", type=str, required=True, help="Path to project directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--max-cycles", type=int, default=None, help="Override max cycles")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s) to use")
    parser.add_argument("--check", action="store_true", help="Verify installation and exit")

    args = parser.parse_args()

    if args.check:
        print("AutoResearcher installation check:")
        print(f"  Python: {sys.version}")
        print(f"  Project: {args.project}")
        print("  Status: OK")
        return

    # Load config
    import yaml
    config_path = Path(args.project) / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    if args.max_cycles is not None:
        config.setdefault("agent", {})["max_cycles"] = args.max_cycles

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.project) / "autoresearcher.log"),
        ],
    )

    # Run
    loop = ResearchLoop(config=config, project_dir=args.project)
    loop.run()


if __name__ == "__main__":
    main()

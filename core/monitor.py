"""
AutoResearcher Experiment Monitor

The key innovation: ZERO LLM calls during experiment training.

While your model trains (hours/days), the monitor only does:
- Process alive check
- Log file tail read
- GPU utilization check

This means running AutoResearcher 24/7 costs the same as running it
only during the THINK and REFLECT phases.
"""

import logging
import shlex
import time
from typing import Optional

from .execution import ExecutionBackend, LocalExecutionBackend

logger = logging.getLogger("autoresearcher.monitor")


class ExperimentMonitor:
    """Zero-LLM experiment monitoring.

    Design principle: During training, the agent is effectively "sleeping"
    at zero cost. It only wakes up (calls LLM) when training completes
    and results need analysis.
    """

    def __init__(
        self,
        poll_interval: int = 900,
        zero_llm: bool = True,
        backend: Optional[ExecutionBackend] = None,
    ):
        self.poll_interval = poll_interval  # seconds between checks
        self.zero_llm = zero_llm
        self.backend = backend or LocalExecutionBackend(".")
        self._active_experiments: dict[int, dict] = {}

    def launch_experiment(self, command: str, log_file: str, gpu: Optional[str] = None) -> dict:
        """Launch an experiment via nohup and track its PID.

        Args:
            command: The training command to run
            log_file: Path to redirect stdout/stderr
            gpu: CUDA_VISIBLE_DEVICES value

        Returns:
            dict with pid, log_file, start_time
        """
        env = {}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        experiment = self.backend.launch_command(
            argv=shlex.split(command),
            log_file=log_file,
            env=env,
        )
        experiment.update({
            "start_time": time.time(),
            "command": command,
            "status": "running",
        })
        self._active_experiments[experiment["pid"]] = experiment

        logger.info(f"Launched experiment: PID={experiment['pid']}, cmd={command[:80]}...")
        return experiment

    def wait_for_completion(self, pid: int, log_file: str, notify: bool = True) -> dict:
        """Wait for experiment to complete. ZERO LLM calls during wait.

        This is the core cost-saving mechanism. Instead of asking the LLM
        "is training done?", we just check if the process is alive.
        """
        logger.info(f"Monitoring PID={pid}, polling every {self.poll_interval}s")

        while self._is_process_alive(pid):
            time.sleep(self.poll_interval)

            # Log current status (no LLM involved)
            gpu_info = self._safe_gpu_status()
            log_tail = self._safe_tail_file(log_file, lines=5)
            elapsed = time.time() - self._active_experiments.get(pid, {}).get("start_time", time.time())

            logger.info(
                f"PID={pid} alive | elapsed={elapsed/3600:.1f}h | "
                f"GPU={gpu_info.get('utilization', 'N/A')} | "
                f"last_log: {log_tail[-1] if log_tail else 'N/A'}"
            )

        # Experiment finished — ask the backend for the real outcome. Slurm
        # reports the sacct terminal state (so FAILED/TIMEOUT are not mislabelled
        # as success); pid-only backends return unknown and we keep "completed".
        elapsed = time.time() - self._active_experiments.get(pid, {}).get("start_time", time.time())
        log_tail = self._safe_tail_file(log_file, lines=50)

        final = self._safe_final_status(pid)
        success = final.get("success")
        status = "failed" if success is False else "completed"

        if pid in self._active_experiments:
            self._active_experiments[pid]["status"] = status

        result = {
            "pid": pid,
            "status": status,
            "success": success,
            "terminal_state": final.get("state", "unknown"),
            "elapsed_hours": elapsed / 3600,
            "log_tail": "\n".join(log_tail),
            "metrics": self._extract_metrics(log_tail),
        }

        logger.info(
            f"Experiment PID={pid} {status} after {result['elapsed_hours']:.1f}h "
            f"(state={result['terminal_state']})"
        )

        if notify:
            self._notify_completion(result)

        return result

    def has_completed_experiments(self) -> bool:
        """Check if any tracked experiment has finished."""
        for pid, exp in list(self._active_experiments.items()):
            if exp["status"] == "running" and not self._is_process_alive(pid):
                exp["status"] = "completed"
                return True
        return False

    def _is_process_alive(self, pid: int) -> bool:
        """Check if process is still running (zero cost)."""
        return self.backend.is_process_alive(pid)

    def _safe_gpu_status(self) -> dict:
        try:
            return self.backend.get_gpu_status()
        except Exception:
            return {"utilization": "N/A"}

    def _safe_final_status(self, pid: int) -> dict:
        try:
            return self.backend.final_status(pid) or {}
        except Exception:
            # Backend without final_status support -> treat as indeterminate.
            return {"state": "unknown", "success": None}

    def _safe_tail_file(self, filepath: str, lines: int = 50) -> list[str]:
        try:
            return self.backend.tail_file(filepath, lines=lines)
        except Exception:
            return []

    def _extract_metrics(self, log_lines: list[str]) -> dict:
        """Try to extract common metrics from training logs.

        Looks for patterns like:
        - loss: 0.123
        - accuracy: 95.2%
        - FGD: 0.582
        - epoch 100/200
        """
        import re
        metrics = {}
        for line in reversed(log_lines):
            # Common metric patterns
            for pattern, key in [
                (r"loss[:\s]+([0-9.]+)", "loss"),
                (r"acc(?:uracy)?[:\s]+([0-9.]+)", "accuracy"),
                (r"FGD[:\s]+([0-9.]+)", "FGD"),
                (r"FID[:\s]+([0-9.]+)", "FID"),
                (r"epoch[:\s]+(\d+)", "epoch"),
                (r"step[:\s]+(\d+)", "step"),
            ]:
                if key not in metrics:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        metrics[key] = match.group(1)
        return metrics

    def _notify_completion(self, result: dict):
        """Send notification when experiment finishes (success or failure)."""
        outcome = result.get("status", "completed").upper()
        logger.info(
            f"EXPERIMENT {outcome} | PID={result['pid']} | "
            f"Time={result['elapsed_hours']:.1f}h | "
            f"State={result.get('terminal_state', '?')} | "
            f"Metrics={result.get('metrics', {})}"
        )

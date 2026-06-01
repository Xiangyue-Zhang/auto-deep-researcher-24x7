"""
Experiment ledger — append-only record of every research cycle.

Stored as ``workspace/experiments.jsonl`` (one JSON object per line). The
append-only design means it survives controller crashes, never needs a
parse-and-rewrite, and stays human- and tool-readable at zero LLM cost. Thin
pure-Python readers (``recent`` / ``summary`` / ``best_metric`` /
``detect_stagnation`` / ``check_phase_gate``) turn the raw trajectory into
compact signals that the loop injects into the THINK context.

This is the spine of v2: persistent memory of *what was tried and what
happened*, which the agent previously lacked (the two-tier MEMORY_LOG is
auto-compacted, so detail was silently dropped).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.ledger")


class ExperimentLedger:
    """Append-only JSONL ledger of experiment cycles."""

    def __init__(self, workspace: Path, filename: str = "experiments.jsonl"):
        self.path = Path(workspace) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        cycle: int,
        hypothesis: str = "",
        action: str = "",
        status: str = "",
        metrics: Optional[dict] = None,
        pid: Optional[int] = None,
        log_file: str = "",
        conclusion: str = "",
        ts: Optional[float] = None,
    ) -> Optional[dict]:
        """Append one cycle's outcome. Never raises — a logging failure must
        not crash the research loop."""
        entry = {
            "ts": time.time() if ts is None else float(ts),
            "cycle": int(cycle),
            "action": str(action or ""),
            "status": str(status or ""),
            "hypothesis": str(hypothesis or "")[:500],
            "metrics": {k: v for k, v in (metrics or {}).items()},
            "pid": pid,
            "log_file": str(log_file or ""),
            "conclusion": str(conclusion or "")[:500],
        }
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover - disk failure path
            logger.warning(f"Failed to append to experiment ledger: {exc}")
            return None
        return entry

    def all(self) -> list[dict]:
        """Return every well-formed entry; malformed lines are skipped."""
        if not self.path.exists():
            return []
        entries: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
        return entries

    def recent(self, n: int = 5) -> list[dict]:
        n = int(n)
        return self.all()[-n:] if n > 0 else []

    def summary(self, n: int = 5) -> str:
        """Render the last ``n`` experiments as a compact context block."""
        entries = self.recent(n)
        if not entries:
            return ""
        lines = []
        for e in entries:
            metrics = e.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "no metrics"
            hypo = (e.get("hypothesis") or "").strip()
            if len(hypo) > 160:
                hypo = hypo[:157] + "..."
            status = e.get("status") or e.get("action") or "?"
            line = f"- cycle {e.get('cycle', '?')} [{status}] {hypo} ({metric_str})"
            conclusion = (e.get("conclusion") or "").strip()
            if conclusion:
                conclusion = conclusion[:160]
                line += f" -> {conclusion}"
            lines.append(line)
        return "\n".join(lines)

    def best_metric(self, metric_key: str, direction: str = "higher_better") -> Optional[float]:
        return best_metric(self.all(), metric_key, direction)


def _metric_values(entries: list[dict], metric_key: str) -> list[tuple[int, float]]:
    """Extract (index, value) pairs for entries that carry a numeric metric."""
    out: list[tuple[int, float]] = []
    for i, e in enumerate(entries):
        metrics = e.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metric_key in metrics:
            try:
                out.append((i, float(metrics[metric_key])))
            except (TypeError, ValueError):
                continue
    return out


def best_metric(entries: list[dict], metric_key: str, direction: str = "higher_better") -> Optional[float]:
    values = [v for _, v in _metric_values(entries, metric_key)]
    if not values:
        return None
    return max(values) if direction == "higher_better" else min(values)


def detect_stagnation(
    entries: list[dict],
    metric_key: str,
    direction: str = "higher_better",
    threshold_cycles: int = 3,
    min_delta: float = 0.0,
) -> dict:
    """Data-driven stagnation signal over the metric trajectory.

    Returns a verdict dict; ``stagnating`` is True when the best metric has
    not improved by more than ``min_delta`` for at least ``threshold_cycles``
    metric-bearing cycles. Advisory only — the caller decides what to do.
    """
    verdict = {
        "stagnating": False,
        "metric_key": metric_key,
        "best": None,
        "recent_best": None,
        "cycles_since_improvement": 0,
        "n_points": 0,
    }
    if not metric_key:
        verdict["reason"] = "no metric_key configured"
        return verdict

    points = _metric_values(entries, metric_key)
    verdict["n_points"] = len(points)
    if len(points) <= threshold_cycles:
        verdict["reason"] = "not enough metric points yet"
        if points:
            verdict["best"] = best_metric(entries, metric_key, direction)
        return verdict

    higher = direction == "higher_better"
    best_val = points[0][1]
    cycles_since_improvement = 0
    for _, val in points[1:]:
        improved = (val > best_val + min_delta) if higher else (val < best_val - min_delta)
        if improved:
            best_val = val
            cycles_since_improvement = 0
        else:
            cycles_since_improvement += 1

    recent_vals = [v for _, v in points[-threshold_cycles:]]
    verdict["best"] = best_val
    verdict["recent_best"] = max(recent_vals) if higher else min(recent_vals)
    verdict["cycles_since_improvement"] = cycles_since_improvement
    verdict["stagnating"] = cycles_since_improvement >= threshold_cycles
    return verdict


def check_phase_gate(
    entries: list[dict],
    metric_key: str,
    threshold: float,
    direction: str = "higher_better",
) -> dict:
    """Advisory promotion gate: is the best metric good enough to proceed?"""
    best = best_metric(entries, metric_key, direction)
    if best is None:
        return {"gate_met": False, "best_metric": None, "blocker_reason": "no metric recorded yet"}
    met = best >= threshold if direction == "higher_better" else best <= threshold
    reason = "" if met else (
        f"best {metric_key}={best} has not cleared the gate threshold {threshold} ({direction})"
    )
    return {"gate_met": met, "best_metric": best, "blocker_reason": reason}

"""
Zero-cost safety helpers — pure functions over state + ledger, no GPU/network.

These keep a long-running agent honest without spending tokens:

- ``scan_violations`` surfaces bad states (repeated no-progress, stale
  "running" state) as advisory strings the loop injects into the THINK context.
- ``seconds_until_allowed`` is the proactive anti-burn rate limiter: given the
  recent cycle-start timestamps, it returns how long to wait so the agent never
  exceeds ``max_per_hour`` cycles (protecting budget when stuck in a loop).

Everything here is deliberately pure and side-effect-free so it is unit-testable
with crafted inputs — no nvidia-smi, no subprocess, no clock.
"""

from __future__ import annotations


def scan_violations(
    state: dict,
    fail_count: int,
    now: float,
    fail_threshold: int = 3,
    stale_state_hours: int = 6,
) -> list[str]:
    """Return advisory violation messages for the current state."""
    violations: list[str] = []
    state = state if isinstance(state, dict) else {}

    if fail_threshold and fail_count >= fail_threshold:
        violations.append(
            f"{fail_count} consecutive no-progress cycles on the same plan — "
            "try a materially different approach or wait for new signal."
        )

    updated = state.get("updated_at")
    status = state.get("status")
    if updated is not None and status == "running" and stale_state_hours:
        try:
            age_hours = (float(now) - float(updated)) / 3600.0
        except (TypeError, ValueError):
            age_hours = 0.0
        if age_hours > stale_state_hours:
            violations.append(
                f"State has been 'running' for {age_hours:.1f}h without an update "
                f"(> {stale_state_hours}h) — training may be stuck or the process died."
            )

    return violations


def seconds_until_allowed(
    timestamps: list[float],
    now: float,
    max_per_hour: int,
    window: int = 3600,
) -> float:
    """How long to wait before starting another cycle, given recent starts.

    Returns 0.0 when rate limiting is disabled (``max_per_hour`` <= 0) or the
    recent count is under budget. Otherwise returns the seconds until the
    oldest in-window timestamp rolls past ``window``.
    """
    if not max_per_hour or max_per_hour <= 0:
        return 0.0
    recent = [t for t in (timestamps or []) if (now - t) < window]
    if len(recent) < max_per_hour:
        return 0.0
    # Wait until enough of the oldest in-window starts roll off to bring the
    # count back under max_per_hour — not just the single oldest one.
    recent_sorted = sorted(recent)
    target = recent_sorted[len(recent) - max_per_hour]
    return max(0.0, float(window) - (float(now) - float(target)))


def prune_timestamps(timestamps: list[float], now: float, window: int = 3600) -> list[float]:
    """Drop timestamps older than ``window`` seconds."""
    return [t for t in (timestamps or []) if (now - t) < window]

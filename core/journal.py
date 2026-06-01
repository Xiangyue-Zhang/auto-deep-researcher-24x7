"""
Research journals — append-only DEAD_ENDS.md and INSIGHTS.md.

Unlike the two-tier MEMORY_LOG (which auto-compacts and silently drops old
detail), these journals are append-only. They are never compacted; when a file
exceeds its size cap it is rotated to a dated ``.bak`` archive and a fresh file
is started, so no history is lost — it is just moved aside.

- DEAD_ENDS.md: approaches that failed and must not be retried.
- INSIGHTS.md: durable observations worth carrying across cycles.

The loop injects the tail of each into the THINK context so the agent stops
repeating known dead ends and keeps its hard-won insights in view.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("autoresearcher.journal")


class _AppendOnlyDoc:
    def __init__(self, path: Path, title: str, max_chars: int):
        self.path = Path(path)
        self.title = title
        self.max_chars = max_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._init()

    def _init(self):
        self.path.write_text(f"# {self.title}\n\n", encoding="utf-8")

    def append(self, entry: str, ts: str = None) -> None:
        """Append a timestamped entry. Never raises."""
        entry = (entry or "").strip()
        if not entry:
            return
        stamp = ts if ts is not None else time.strftime("%Y-%m-%d %H:%M")
        try:
            if not self.path.exists():
                self._init()
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(f"- [{stamp}] {entry}\n")
            if self.path.stat().st_size > self.max_chars:
                self._rotate(stamp)
        except OSError as exc:  # pragma: no cover - disk failure path
            logger.warning(f"Failed to append to {self.path.name}: {exc}")

    def _rotate(self, stamp: str) -> None:
        """Archive the full file to a dated backup, then keep only the tail."""
        try:
            content = self.path.read_text(encoding="utf-8")
            safe_stamp = stamp.replace(" ", "_").replace(":", "")
            backup = self.path.with_name(f"{self.path.stem}.{safe_stamp}.bak")
            n = 0
            while backup.exists():
                n += 1
                backup = self.path.with_name(f"{self.path.stem}.{safe_stamp}.{n}.bak")
            backup.write_text(content, encoding="utf-8")
            # Restart with the header plus the most recent half of the entries.
            tail = content[-(self.max_chars // 2):]
            self.path.write_text(
                f"# {self.title}\n\n_(rotated; full history in {backup.name})_\n{tail}",
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - disk failure path
            logger.warning(f"Failed to rotate {self.path.name}: {exc}")

    def tail(self, max_chars: int) -> str:
        """Return the last ``max_chars`` of the file. Never raises."""
        try:
            max_chars = int(max_chars)
        except (TypeError, ValueError):
            max_chars = self.max_chars
        try:
            if not self.path.exists():
                return ""
            content = self.path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk failure path
            logger.warning(f"Failed to read {self.path.name}: {exc}")
            return ""
        return content[-max_chars:] if len(content) > max_chars else content


class ResearchJournal:
    """Manages the DEAD_ENDS and INSIGHTS append-only journals."""

    def __init__(self, workspace: Path, max_chars: int = 4000):
        workspace = Path(workspace)
        self.dead_ends = _AppendOnlyDoc(workspace / "DEAD_ENDS.md", "Dead Ends", max_chars)
        self.insights = _AppendOnlyDoc(workspace / "INSIGHTS.md", "Insights", max_chars)

    def append_dead_end(self, entry: str, ts: str = None) -> None:
        self.dead_ends.append(entry, ts=ts)

    def append_insight(self, entry: str, ts: str = None) -> None:
        self.insights.append(entry, ts=ts)

    def dead_ends_tail(self, max_chars: int = 1500) -> str:
        return self.dead_ends.tail(max_chars)

    def insights_tail(self, max_chars: int = 1500) -> str:
        return self.insights.tail(max_chars)

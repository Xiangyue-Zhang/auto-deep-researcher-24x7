"""Live integration check: drive a real CLI-provider worker end-to-end.

Run manually:  python -m tests.integration_cli_tool_use

Burns one subscription round-trip per provider. Skipped automatically
if the CLI is not on PATH. Not wired into the normal unittest suite.
"""

import shutil
import tempfile
from pathlib import Path

from core.agents import AgentDispatcher
from core.tools import ToolRegistry


TASK = (
    "Your one job: create a file named hello.txt in the workspace "
    "containing exactly the three-word sentence 'integration test ok', "
    "then confirm by listing the files. Once done, reply with a short "
    "success message and no further tool calls."
)


def _run(provider: str) -> dict:
    binary = {"claude_cli": "claude", "codex_cli": "codex"}[provider]
    if shutil.which(binary) is None:
        return {"provider": provider, "skipped": f"{binary} not on PATH"}

    dispatcher = AgentDispatcher(provider=provider)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        registry = ToolRegistry(workspace)
        try:
            result = dispatcher.dispatch_worker("writing", TASK, registry)
        except Exception as exc:
            return {"provider": provider, "error": repr(exc)}

        hello = workspace / "hello.txt"
        return {
            "provider": provider,
            "tool_calls": result.get("tool_calls", 0),
            "file_created": hello.exists(),
            "file_content": hello.read_text() if hello.exists() else None,
            "response_tail": (result.get("response", "") or "")[-200:],
        }


def main():
    for provider in ("claude_cli", "codex_cli"):
        print(f"\n=== {provider} ===")
        outcome = _run(provider)
        for k, v in outcome.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

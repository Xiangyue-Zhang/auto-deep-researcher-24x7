"""
E2E smoke test for LiteLLM provider integration.

Requires a working LLM API key. Set one of:
  - ANTHROPIC_API_KEY + optional ANTHROPIC_FOUNDRY_BASE_URL
  - OPENAI_API_KEY
  - Any provider key supported by litellm

Usage:
    python tests/e2e_litellm.py
"""

import os
import sys

from core.agents import AgentDispatcher


def main():
    api_key = (
        os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL", "")
    model = os.environ.get("LITELLM_MODEL", "anthropic/claude-sonnet-4-6")

    if not api_key:
        print("SKIP: no API key found in environment")
        sys.exit(0)

    print(f"Provider: litellm")
    print(f"Model:    {model}")
    print(f"Base URL: {base_url or '(default)'}")
    print()

    dispatcher = AgentDispatcher(
        provider="litellm",
        model=model,
        base_url=base_url or None,
        api_key=api_key,
    )

    result = dispatcher._call_litellm(
        "You are a helpful assistant. Respond concisely.",
        [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    )

    print(f"Response: \"{result}\"")
    assert result.strip(), "Empty response"
    assert "4" in result, f"Expected '4' in response, got: {result}"
    print()
    print("E2E PASSED - litellm provider works end-to-end")


if __name__ == "__main__":
    main()

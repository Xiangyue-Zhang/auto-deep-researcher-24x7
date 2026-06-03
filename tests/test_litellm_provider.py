import os
import types
import unittest
from unittest.mock import MagicMock, patch

from core.agents import AgentDispatcher


class _LiteLLMResponse:
    def __init__(self, content: str):
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )
        ]


class LiteLLMProviderTests(unittest.TestCase):
    def test_litellm_in_supported_providers(self):
        self.assertIn("litellm", AgentDispatcher.SUPPORTED_PROVIDERS)

    def test_dispatcher_accepts_litellm_provider(self):
        dispatcher = AgentDispatcher(
            provider="litellm",
            model="anthropic/claude-sonnet-4-20250514",
        )
        self.assertEqual(dispatcher.provider, "litellm")
        self.assertEqual(dispatcher.model, "anthropic/claude-sonnet-4-20250514")

    def test_call_litellm_dispatches_to_litellm_completion(self):
        completion = MagicMock(return_value=_LiteLLMResponse("litellm ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="anthropic/claude-sonnet-4-20250514",
            )
            result = dispatcher._call_litellm(
                "system prompt",
                [{"role": "user", "content": "hello"}],
            )

        completion.assert_called_once()
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "anthropic/claude-sonnet-4-20250514")
        self.assertEqual(result, "litellm ok")

    def test_call_litellm_includes_drop_params_true(self):
        completion = MagicMock(return_value=_LiteLLMResponse("ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
            )
            dispatcher._call_litellm(
                "system",
                [{"role": "user", "content": "hi"}],
            )

        kwargs = completion.call_args.kwargs
        self.assertTrue(kwargs["drop_params"])

    def test_call_litellm_passes_api_key_when_set(self):
        completion = MagicMock(return_value=_LiteLLMResponse("ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
                api_key="sk-test-key",
            )
            dispatcher._call_litellm(
                "system",
                [{"role": "user", "content": "hi"}],
            )

        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "sk-test-key")

    def test_call_litellm_omits_api_key_when_not_set(self):
        completion = MagicMock(return_value=_LiteLLMResponse("ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
            )
            dispatcher._call_litellm(
                "system",
                [{"role": "user", "content": "hi"}],
            )

        kwargs = completion.call_args.kwargs
        self.assertNotIn("api_key", kwargs)

    def test_call_litellm_passes_base_url_when_set(self):
        completion = MagicMock(return_value=_LiteLLMResponse("ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
                base_url="http://my-proxy:4000/v1",
            )
            dispatcher._call_litellm(
                "system",
                [{"role": "user", "content": "hi"}],
            )

        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["api_base"], "http://my-proxy:4000/v1")

    def test_call_litellm_formats_messages_correctly(self):
        completion = MagicMock(return_value=_LiteLLMResponse("ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
            )
            dispatcher._call_litellm(
                "you are helpful",
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "thanks"},
                ],
            )

        kwargs = completion.call_args.kwargs
        messages = kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "you are helpful")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")

    def test_call_litellm_falls_back_to_openai_on_import_error(self):
        with patch.dict("sys.modules", {"litellm": None}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="openai/gpt-4o",
            )
            with patch.object(dispatcher, "_call_openai", return_value="fallback ok") as mock_openai:
                result = dispatcher._call_litellm(
                    "system",
                    [{"role": "user", "content": "hi"}],
                )

        mock_openai.assert_called_once()
        self.assertEqual(result, "fallback ok")

    def test_call_llm_routes_to_litellm(self):
        completion = MagicMock(return_value=_LiteLLMResponse("routed ok"))
        fake_litellm = types.SimpleNamespace(completion=completion)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            dispatcher = AgentDispatcher(
                provider="litellm",
                model="anthropic/claude-haiku-4-5",
            )
            result = dispatcher._call_llm(
                "system",
                [{"role": "user", "content": "test"}],
            )

        self.assertEqual(result, "routed ok")


if __name__ == "__main__":
    unittest.main()

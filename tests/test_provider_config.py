import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from core.agents import AgentDispatcher
from core.loop import ResearchLoop


class _OpenAIResponse:
    def __init__(self, content: str):
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )
        ]


class _AnthropicResponse:
    def __init__(self, content: str):
        self.content = [types.SimpleNamespace(text=content)]


class CompatibleProviderConfigTests(unittest.TestCase):
    def test_openai_compatible_provider_passes_base_url_and_custom_key_env(self):
        create = MagicMock(return_value=_OpenAIResponse("qwen ok"))
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )
        )
        ctor = MagicMock(return_value=client)
        fake_openai = types.SimpleNamespace(OpenAI=ctor)

        with patch.dict("sys.modules", {"openai": fake_openai}):
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret-key"}, clear=False):
                dispatcher = AgentDispatcher(
                    provider="openai",
                    model="qwen-plus",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key_env="DASHSCOPE_API_KEY",
                )
                result = dispatcher._call_openai(
                    "system prompt",
                    [{"role": "user", "content": "hello"}],
                )

        ctor.assert_called_once_with(
            api_key="secret-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["model"], "qwen-plus")
        self.assertEqual(result, "qwen ok")

    def test_anthropic_compatible_provider_passes_base_url_and_auth(self):
        create = MagicMock(return_value=_AnthropicResponse("minimax ok"))
        client = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=create)
        )
        ctor = MagicMock(return_value=client)
        fake_anthropic = types.SimpleNamespace(Anthropic=ctor)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            with patch.dict(
                os.environ,
                {
                    "MINIMAX_API_KEY": "secret-key",
                    "MINIMAX_AUTH_TOKEN": "secret-token",
                },
                clear=False,
            ):
                dispatcher = AgentDispatcher(
                    provider="anthropic",
                    model="MiniMax-M2.1",
                    base_url="https://api.minimaxi.com/anthropic",
                    api_key_env="MINIMAX_API_KEY",
                    auth_token_env="MINIMAX_AUTH_TOKEN",
                )
                result = dispatcher._call_anthropic(
                    "system prompt",
                    [{"role": "user", "content": "hello"}],
                )

        ctor.assert_called_once_with(
            api_key="secret-key",
            auth_token="secret-token",
            base_url="https://api.minimaxi.com/anthropic",
        )
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["model"], "MiniMax-M2.1")
        self.assertEqual(result, "minimax ok")


class ResearchLoopProviderConfigTests(unittest.TestCase):
    @patch("core.loop.AgentDispatcher")
    @patch("core.loop.ToolRegistry")
    @patch("core.loop.ObsidianExporter")
    @patch("core.loop.ExperimentMonitor")
    @patch("core.loop.MemoryManager")
    @patch("core.loop.build_execution_backend")
    def test_loop_passes_compatible_provider_config(
        self,
        build_backend_mock,
        _memory_mock,
        _monitor_mock,
        _obsidian_mock,
        _tool_registry_mock,
        dispatcher_mock,
    ):
        backend = MagicMock()
        build_backend_mock.return_value = backend

        with tempfile.TemporaryDirectory() as tmp:
            ResearchLoop(
                config={
                    "project": {"workspace": "workspace"},
                    "agent": {
                        "provider": "openai",
                        "model": "glm-4.5",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                        "api_key_env": "ZHIPUAI_API_KEY",
                        "auth_token_env": "",
                        "max_steps_per_cycle": 5,
                    },
                },
                project_dir=tmp,
            )

        dispatcher_mock.assert_called_once_with(
            model="glm-4.5",
            provider="openai",
            max_steps=5,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key_env="ZHIPUAI_API_KEY",
            auth_token_env="",
        )
        backend.validate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

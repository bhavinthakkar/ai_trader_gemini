import os
import unittest
from unittest.mock import Mock, patch

import httpx2

from llamacpp_provider import (
    LlamaCppPromptBudgetError,
    fit_messages_to_budget,
    load_config,
    query_llamacpp,
)
from llm_service import MODEL_REGISTRY, get_model_label, normalize_model_key, query_llm


class FakeHttpClient:
    def __init__(self, response: httpx2.Response):
        self._response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, *, json):
        self.calls.append({"url": url, "json": json})
        return self._response


class LlamaCppProviderTests(unittest.TestCase):
    def test_load_config_uses_local_llamacpp_defaults(self):
        env = {
            "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1/",
            "LLAMACPP_MODEL": "qwen2.5",
            "LLAMACPP_MAX_TOKENS": "1536",
            "LLAMACPP_CONTEXT_TOKENS": "8192",
            "LLAMACPP_READ_TIMEOUT": "900",
        }

        with patch.dict(os.environ, env, clear=True):
            config = load_config()

        self.assertEqual(config.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.model, "qwen2.5")
        self.assertEqual(config.max_output_tokens, 1536)
        self.assertEqual(config.context_tokens, 8192)
        self.assertEqual(config.read_timeout_seconds, 900.0)

    def test_query_posts_openai_compatible_json_request(self):
        request = httpx2.Request(
            "POST",
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        response = httpx2.Response(
            200,
            request=request,
            json={
                "model": "qwen2.5",
                "choices": [{"message": {"content": '{"decision":"HOLD"}'}}],
            },
        )
        client = FakeHttpClient(response)

        with (
            patch.dict(
                os.environ,
                {
                    "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1",
                    "LLAMACPP_MODEL": "qwen2.5",
                    "LLAMACPP_API_KEY": "",
                    "LLAMACPP_MAX_TOKENS": "1024",
                    "LLAMACPP_CONTEXT_TOKENS": "8192",
                },
                clear=True,
            ),
            patch("llamacpp_provider.create_client", return_value=client),
        ):
            result = query_llamacpp(
                system_instruction="Return JSON only.",
                user_prompt="Analyze AAPL.",
                temperature=0.2,
            )

        self.assertEqual(result, '{"decision":"HOLD"}')
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["url"], "chat/completions")
        payload = client.calls[0]["json"]
        self.assertEqual(payload["model"], "qwen2.5")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Analyze AAPL."},
            ],
        )

    def test_prompt_fitting_preserves_schema_tail(self):
        config = load_config_from_values(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5",
            max_output_tokens=1024,
            context_tokens=4096,
            prompt_safety_margin_tokens=128,
        )
        system_instruction = "Compact system rules. " * 120
        user_prompt = (
            "STRUCTURED DATA\n"
            + ("x" * 30000)
            + "\nRETURN EXACTLY THIS JSON SHAPE: {\"decision\":\"BUY|SELL|HOLD\"}"
        )

        fitted_system, fitted_user, estimated_total = fit_messages_to_budget(
            system_instruction,
            user_prompt,
            config,
        )

        self.assertEqual(fitted_system, system_instruction)
        self.assertIn("STRUCTURED DATA", fitted_user)
        self.assertIn("RETURN EXACTLY THIS JSON SHAPE", fitted_user)
        self.assertIn("[...LOCAL PROMPT TRUNCATED...]", fitted_user)
        self.assertLessEqual(
            estimated_total,
            config.context_tokens - config.prompt_safety_margin_tokens,
        )

    def test_prompt_fitting_rejects_system_instruction_that_cannot_fit(self):
        config = load_config_from_values(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5",
            max_output_tokens=2048,
            context_tokens=4096,
            prompt_safety_margin_tokens=128,
        )

        with self.assertRaises(LlamaCppPromptBudgetError):
            fit_messages_to_budget(
                "system " * 5000,
                "user",
                config,
            )


def load_config_from_values(**overrides):
    from llamacpp_provider import LlamaCppConfig

    values = {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5",
        "api_key": None,
        "connect_timeout_seconds": 5.0,
        "read_timeout_seconds": 900.0,
        "write_timeout_seconds": 30.0,
        "pool_timeout_seconds": 10.0,
        "max_output_tokens": 2048,
        "context_tokens": 8192,
        "prompt_safety_margin_tokens": 256,
    }
    values.update(overrides)
    return LlamaCppConfig(**values)


class ModelRegistryTests(unittest.TestCase):
    def test_registry_targets_xingchen_openrouter(self):
        for key in MODEL_REGISTRY:
            self.assertEqual(MODEL_REGISTRY[key]["provider"], "openrouter")
            self.assertEqual(MODEL_REGISTRY[key]["model"], "xingchen-agi/xing4.0-29b-a4b:free")
        self.assertIn("xingchen", MODEL_REGISTRY)

    def test_query_llm_routes_qwen_to_xingchen(self):
        self.assertEqual(normalize_model_key("qwen"), "xingchen")
        self.assertIn("xingchen", get_model_label("qwen").lower())


if __name__ == "__main__":
    unittest.main()

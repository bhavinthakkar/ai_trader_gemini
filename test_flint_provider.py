import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

import llm_service
from llm_service import (
    MODEL_REGISTRY,
    TimeBudgetExceeded,
    get_model_label,
    normalize_model_key,
    query_llm,
)


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode("utf-8")


def mock_openrouter_stream_response(chunks, reasoning=(), status=200, text=""):
    resp = Mock()
    resp.status_code = status
    resp.text = text
    resp.headers = {"content-type": "text/event-stream"}
    lines = [_sse({"choices": [{"index": 0, "delta": {"reasoning": r}, "finish_reason": None}]})
             for r in reasoning]
    lines += [_sse({"choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})
              for c in chunks]
    lines.append(b"data: [DONE]")
    resp.iter_lines.return_value = iter(lines)
    return resp


class FlintRoutingTests(unittest.TestCase):
    def test_flint_aliases_resolve_to_flint(self):
        for alias in ("flint", "flint-v1.0", "recursive/flint-v1.0:free", "recursive", "flint-free", "default"):
            self.assertEqual(normalize_model_key(alias), "flint")

    def test_legacy_claude_aliases_redirect_to_flint(self):
        for alias in ("haiku", "haike", "haiku-5.5", "haike-5.5", "claude-haiku-5-5", "claude-5-5-haiku", "claude", "anthropic"):
            self.assertEqual(normalize_model_key(alias), "flint")

    def test_empty_model_defaults_to_flint(self):
        self.assertEqual(normalize_model_key(""), "flint")
        self.assertEqual(normalize_model_key(None), "flint")

    def test_flint_registry_config(self):
        config = MODEL_REGISTRY["flint"]
        self.assertEqual(config["provider"], "openrouter")
        self.assertEqual(config["model"], "recursive/flint-v1.0:free")
        self.assertIn("Recursive: Flint (free)", config["label"])
        self.assertIn("openrouter/free", config.get("fallbacks", []))

    def test_claude_not_in_registry(self):
        for key in ("claude-haiku-5-5", "haiku", "haike", "claude", "anthropic"):
            self.assertNotIn(key, MODEL_REGISTRY)

    def test_flint_label(self):
        self.assertIn("Recursive: Flint (free)", get_model_label("flint"))
        self.assertIn("recursive/flint-v1.0:free", get_model_label(""))


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}, clear=False)
class FlintExecutionTests(unittest.TestCase):
    def test_query_llm_calls_openrouter_endpoint_with_flint(self):
        recorded = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            recorded["url"] = url
            recorded["headers"] = headers
            recorded["payload"] = json
            recorded["timeout"] = timeout
            recorded["stream"] = stream
            return mock_openrouter_stream_response(['{"decision": "HOLD", "confidence": 0.85}'])

        with patch("llm_service.requests.post", side_effect=responder):
            out = query_llm("System prompt", "User prompt", model_choice="flint")

        self.assertEqual(recorded["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(recorded["headers"]["Authorization"], "Bearer test-openrouter-key")
        self.assertEqual(recorded["payload"]["model"], "recursive/flint-v1.0:free")
        self.assertEqual(recorded["payload"]["messages"][0]["content"], "System prompt")
        self.assertEqual(recorded["payload"]["messages"][1]["content"], "User prompt")
        self.assertEqual(recorded["payload"]["response_format"], {"type": "json_object"})
        self.assertIn("decision", out)

    def test_query_llm_defaults_to_flint_when_unspecified(self):
        recorded = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            recorded["payload"] = json
            return mock_openrouter_stream_response(['{"decision": "BUY"}'])

        with patch("llm_service.requests.post", side_effect=responder):
            out = query_llm("System", "User")

        self.assertEqual(recorded["payload"]["model"], "recursive/flint-v1.0:free")
        self.assertIn("BUY", out)

    def test_missing_openrouter_api_key_raises_value_error(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                query_llm("Sys", "User", model_choice="flint")
            self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_rate_limit_retries_and_falls_back(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == "recursive/flint-v1.0:free":
                limited = Mock()
                limited.status_code = 429
                limited.text = "Rate limit reached"
                return limited
            return mock_openrouter_stream_response(['{"decision": "HOLD"}'])

        with patch("llm_service.requests.post", side_effect=responder), patch("llm_service.time.sleep"):
            out = query_llm("Sys", "User", model_choice="flint")

        self.assertIn("recursive/flint-v1.0:free", calls)
        self.assertIn("openrouter/free", calls)
        self.assertIn("HOLD", out)


if __name__ == "__main__":
    unittest.main()

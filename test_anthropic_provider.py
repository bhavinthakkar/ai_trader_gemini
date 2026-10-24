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


def _anthropic_sse_lines(text_chunks, is_stream_stop=True):
    lines = [
        b"event: message_start",
        b'data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant"}}',
        b"event: content_block_start",
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    ]
    for chunk in text_chunks:
        lines.append(b"event: content_block_delta")
        lines.append(f'data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{json.dumps(chunk)}}}}}'.encode("utf-8"))
    lines.append(b"event: content_block_stop")
    lines.append(b'data: {"type":"content_block_stop","index":0}')
    lines.append(b"event: message_delta")
    lines.append(b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}')
    if is_stream_stop:
        lines.append(b"event: message_stop")
        lines.append(b'data: {"type":"message_stop"}')
    return lines


def mock_anthropic_stream_response(chunks, status=200, text=""):
    resp = Mock()
    resp.status_code = status
    resp.text = text
    resp.headers = {"content-type": "text/event-stream"}
    resp.iter_lines.return_value = iter(_anthropic_sse_lines(chunks))
    return resp


class AnthropicRoutingTests(unittest.TestCase):
    def test_haiku_aliases_resolve_to_claude_haiku_5_5(self):
        for alias in ("haiku", "haike", "haiku-5.5", "haike-5.5", "claude-haiku-5-5", "claude-5-5-haiku", "claude", "anthropic", "default"):
            self.assertEqual(normalize_model_key(alias), "claude-haiku-5-5")

    def test_empty_model_defaults_to_claude_haiku_5_5(self):
        self.assertEqual(normalize_model_key(""), "claude-haiku-5-5")
        self.assertEqual(normalize_model_key(None), "claude-haiku-5-5")

    def test_haiku_registry_config(self):
        config = MODEL_REGISTRY["claude-haiku-5-5"]
        self.assertEqual(config["provider"], "anthropic")
        self.assertEqual(config["model"], "claude-haiku-5-5")
        self.assertIn("Haiku 5.5", config["label"])

    def test_haiku_label(self):
        self.assertIn("Anthropic Claude Haiku 5.5", get_model_label("haiku"))
        self.assertIn("Anthropic Claude Haiku 5.5", get_model_label("haike"))


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}, clear=False)
class AnthropicExecutionTests(unittest.TestCase):
    def test_query_llm_calls_anthropic_messages_endpoint(self):
        recorded = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            recorded["url"] = url
            recorded["headers"] = headers
            recorded["payload"] = json
            recorded["timeout"] = timeout
            recorded["stream"] = stream
            return mock_anthropic_stream_response(['{"decision": "HOLD", "confidence": 0.85}'])

        with patch("llm_service.requests.post", side_effect=responder):
            out = query_llm("System prompt", "User prompt", model_choice="haiku")

        self.assertEqual(recorded["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(recorded["headers"]["x-api-key"], "test-anthropic-key")
        self.assertEqual(recorded["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(recorded["payload"]["model"], "claude-haiku-5-5")
        self.assertEqual(recorded["payload"]["system"], "System prompt")
        self.assertEqual(recorded["payload"]["messages"], [{"role": "user", "content": "User prompt"}])
        # Verify temperature is omitted / deprecated for Haiku 5.5
        self.assertNotIn("temperature", recorded["payload"])
        # Verify default thinking is disabled
        self.assertEqual(recorded["payload"]["thinking"], {"type": "disabled"})
        self.assertIn("decision", out)

    def test_adaptive_thinking_when_effort_specified(self):
        recorded = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            recorded["payload"] = json
            return mock_anthropic_stream_response(['{"decision": "BUY"}'])

        with patch("llm_service.requests.post", side_effect=responder):
            query_llm("System", "User", model_choice="haiku", reasoning_effort="high")

        self.assertEqual(recorded["payload"]["thinking"], {"type": "adaptive"})

    def test_missing_api_key_raises_value_error(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                query_llm("Sys", "User", model_choice="haiku")
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_non_streaming_json_fallback(self):
        resp = Mock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "content": [{"type": "text", "text": '{"decision": "SELL", "confidence": 0.9}'}]
        }

        with patch("llm_service.requests.post", return_value=resp):
            out = query_llm("Sys", "User", model_choice="haiku")
        self.assertIn("SELL", out)

    def test_rate_limit_retries_and_succeeds(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(len(calls))
            if len(calls) == 1:
                limited = Mock()
                limited.status_code = 429
                limited.text = "Rate limit reached"
                return limited
            return mock_anthropic_stream_response(['{"decision": "HOLD"}'])

        with patch("llm_service.requests.post", side_effect=responder), patch("llm_service.time.sleep"):
            out = query_llm("Sys", "User", model_choice="haiku")

        self.assertEqual(len(calls), 2)
        self.assertIn("HOLD", out)

    def test_overload_529_retries_and_succeeds(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(len(calls))
            if len(calls) == 1:
                overloaded = Mock()
                overloaded.status_code = 529
                overloaded.text = "Anthropic is temporarily overloaded"
                return overloaded
            return mock_anthropic_stream_response(['{"decision": "BUY"}'])

        with patch("llm_service.requests.post", side_effect=responder), patch("llm_service.time.sleep"):
            out = query_llm("Sys", "User", model_choice="haiku")

        self.assertEqual(len(calls), 2)
        self.assertIn("BUY", out)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
import time
import unittest
from unittest.mock import Mock, patch

import requests

import llm_service
from llm_service import MODEL_REGISTRY, get_model_label, normalize_model_key, query_llm


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode()


def stream_response(pieces=(), reasoning=(), status=200, text=""):
    response = Mock()
    response.status_code = status
    response.text = text
    lines = [_sse({"choices": [{"index": 0, "delta": {"reasoning": r}, "finish_reason": None}]})
             for r in reasoning]
    lines += [_sse({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
              for p in pieces]
    lines.append(b"data: [DONE]")
    response.iter_lines.return_value = iter(lines)
    return response


class OpenRouterRoutingTests(unittest.TestCase):
    def test_existing_openrouter_aliases_resolve_to_openrouter(self):
        for alias in ("openrouter", "openrouter/free", "or", "minimax", "m3"):
            self.assertEqual(normalize_model_key(alias), "openrouter")

    def test_openrouter_targets_the_model_via_openrouter(self):
        config = MODEL_REGISTRY["openrouter"]
        self.assertEqual(config["provider"], "openrouter")
        self.assertEqual(config["model"], "openrouter/free")
        self.assertIsNone(config.get("default_reasoning_effort"))

    def test_openrouter_label_mentions_free(self):
        self.assertIn("free", get_model_label("openrouter").lower())

    def test_bunny_is_removed_and_unsupported(self):
        for alias in ("bunny", "space-bunny", "space-bunny-alpha", "SPACE-BUNNY-ALPHA", "sb", "stealth/space-bunny-alpha"):
            self.assertNotIn(alias, MODEL_REGISTRY)
        with self.assertRaises(ValueError):
            query_llm("sys", "user", model_choice="bunny")


@patch.dict(__import__("os").environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
class OpenRouterRequestTests(unittest.TestCase):
    def call(self, responder, model_choice="openrouter", **kwargs):
        with patch("llm_service.requests.post", side_effect=responder) as post, patch(
            "llm_service.time.sleep"
        ):
            return query_llm("sys", "user", model_choice=model_choice, **kwargs), post

    def test_default_openrouter_route_sends_no_reasoning_effort(self):
        seen = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen.update(json)
            return stream_response(['{"decision": "HOLD"}'])

        self.call(responder)
        self.assertEqual(seen["model"], "openrouter/free")
        self.assertNotIn("reasoning_effort", seen)
        self.assertIs(seen["stream"], True)
        self.assertEqual(seen["response_format"], {"type": "json_object"})

    def test_explicit_reasoning_effort_overrides_default(self):
        seen = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen.update(json)
            return stream_response(['{"decision": "HOLD"}'])

        self.call(responder, reasoning_effort="medium")
        self.assertEqual(seen["reasoning_effort"], "medium")

    def test_reasoning_deltas_are_not_returned(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(['{"a":1}'], reasoning=["lots of thinking", "more"])

        out, _ = self.call(responder)
        self.assertEqual(out, '{"a":1}')

    def test_finish_reason_without_done_terminates(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            response = Mock()
            response.status_code = 200
            response.text = ""

            def lines():
                yield _sse({"choices": [{"index": 0, "delta": {"content": '{"a":1}'}, "finish_reason": None}]})
                yield _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self.fail("iterator kept yielding after finish_reason")

            response.iter_lines.side_effect = lines
            return response

        out, _ = self.call(responder)
        self.assertEqual(out, '{"a":1}')

    def test_stream_uses_openrouter_budget_tuple(self):
        seen = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen["timeout"] = timeout
            return stream_response(['{"a":1}'])

        self.call(responder)
        self.assertEqual(
            seen["timeout"],
            (llm_service.OPENROUTER_CONNECT_TIMEOUT, llm_service.OPENROUTER_READ_TIMEOUT),
        )

    def test_invalid_json_falls_back(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == "openrouter/free":
                return stream_response(["not json at all"])
            return stream_response(['{"decision": "HOLD"}'])

        with patch.dict(MODEL_REGISTRY["openrouter"], {"fallbacks": ["vendor/other"]}, clear=False):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')
        self.assertEqual(calls, ["openrouter/free", "vendor/other"])

    def test_read_timeout_moves_to_next_candidate(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == "openrouter/free":
                raise requests.exceptions.ReadTimeout("read timed out")
            return stream_response(['{"decision": "HOLD"}'])

        with patch.dict(MODEL_REGISTRY["openrouter"], {"fallbacks": ["vendor/other"]}, clear=False):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')
        self.assertEqual(calls, ["openrouter/free", "vendor/other"])

    def test_rate_limit_moves_to_next_candidate(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == "openrouter/free":
                return stream_response(status=429, text="rate limited")
            return stream_response(['{"decision": "HOLD"}'])

        with patch.dict(MODEL_REGISTRY["openrouter"], {"fallbacks": ["vendor/other"]}, clear=False):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')

    def test_total_budget_stops_before_next_candidate(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            return stream_response(['{"a":1}'])

        with patch.dict(MODEL_REGISTRY["openrouter"], {"fallbacks": ["vendor/other"]}, clear=False), \
                patch.object(llm_service, "OPENROUTER_TOTAL_TIMEOUT", -1):
            with self.assertRaises(RuntimeError):
                self.call(responder)
        self.assertEqual(calls, [])

    def test_progress_is_reported(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(['{"decision"', ': "HOLD"', '}'])

        buf = io.StringIO()
        with patch.object(llm_service, "OPENROUTER_PROGRESS_INTERVAL", 0), contextlib.redirect_stdout(buf):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')
        self.assertIn("still generating", buf.getvalue())


class SharedStreamReaderTests(unittest.TestCase):
    def test_budget_check_survives_heartbeats(self):
        response = Mock()
        response.iter_lines.side_effect = lambda: iter([b": ping"] * 1000)
        with self.assertRaises(llm_service.TimeBudgetExceeded):
            llm_service._collect_sse_content(
                response, "x", time.monotonic() - 1, 0, 600)

    def test_malformed_lines_are_skipped(self):
        response = Mock()
        response.iter_lines.return_value = iter([
            b": keep-alive", b"data: not-json{", b"data: [DONE]",
        ])
        self.assertEqual(
            llm_service._collect_sse_content(
                response, "x", time.monotonic() + 60, 0, 600),
            "",
        )


if __name__ == "__main__":
    unittest.main()

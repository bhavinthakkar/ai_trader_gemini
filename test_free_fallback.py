import contextlib
import io
import json
import os
import time
import unittest
from unittest.mock import Mock, patch

import requests

import llm_service
from llm_service import MODEL_REGISTRY, get_model_label, normalize_model_key, query_llm

PRIMARY = "openrouter/free"
FLINT = PRIMARY
XINGCHEN = PRIMARY
FALLBACK = "recursive/flint-v1.0:free"


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode()


def stream_response(pieces=(), reasoning=(), status=200, text=""):
    response = Mock()
    response.status_code = status
    response.text = text
    lines = [_sse({"choices": [{"index": 0, "delta": {"reasoning_content": r}, "finish_reason": None}]})
             for r in reasoning]
    lines += [_sse({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
              for p in pieces]
    lines.append(b"data: [DONE]")
    response.iter_lines.return_value = iter(lines)
    return response


def ok_response(content='{"decision": "HOLD"}'):
    return stream_response([content])


class FreeRoutingTests(unittest.TestCase):
    def test_free_is_aliased_to_openrouter_free(self):
        self.assertEqual(normalize_model_key("free"), "openrouter/free")

    def test_free_targets_openrouter_free_then_fallback(self):
        config = MODEL_REGISTRY["free"]
        self.assertEqual(config["provider"], "openrouter")
        self.assertEqual(config["model"], "openrouter/free")
        self.assertIn(FALLBACK, config["fallbacks"])

    def test_free_label_advertises_openrouter_free(self):
        self.assertIn("OpenRouter Free", get_model_label("free"))

    def test_openrouter_aliases_redirect_to_openrouter_free(self):
        for alias in ("openrouter", "openrouter/free", "minimax", "minimax-m3", "m3", "or"):
            self.assertEqual(normalize_model_key(alias), "openrouter/free")
            self.assertEqual(MODEL_REGISTRY[normalize_model_key(alias)]["provider"], "openrouter")

    def test_other_legacy_aliases_redirect_to_openrouter_free(self):
        for alias in ("ultra", "kimi", "ling", "nemotron", "gemini", "qwen", "flint", "xingchen"):
            self.assertEqual(normalize_model_key(alias), "openrouter/free")
            self.assertEqual(MODEL_REGISTRY[normalize_model_key(alias)]["provider"], "openrouter")

    def test_free_case_insensitive(self):
        self.assertEqual(normalize_model_key("FREE"), "openrouter/free")


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
class ReadTimeoutFallbackTests(unittest.TestCase):
    def call(self, responder, model_choice="free"):
        """Run one query_llm with requests.post and sleep patched, so nothing leaves the process."""
        with patch("llm_service.requests.post", side_effect=responder) as post, patch(
            "llm_service.time.sleep"
        ):
            return query_llm("sys", "user", model_choice=model_choice), post

    def test_read_timeout_falls_back_to_next(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == FLINT:
                raise requests.exceptions.ReadTimeout("read timed out")
            return ok_response('{"decision": "from fallback"}')

        out, post = self.call(responder)
        self.assertEqual(out, '{"decision": "from fallback"}')
        self.assertEqual(calls, [FLINT, FALLBACK])
        self.assertEqual(post.call_count, 2)

    def test_read_timeout_does_not_retry_same_model(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            raise requests.exceptions.ReadTimeout("read timed out")

        with self.assertRaises(requests.exceptions.ReadTimeout):
            self.call(responder)
        self.assertEqual(calls, [PRIMARY] + MODEL_REGISTRY["free"]["fallbacks"])

    def test_xingchen_success_does_not_touch_fallback(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            return ok_response('{"decision": "from xingchen"}')

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "from xingchen"}')
        self.assertEqual(calls, [XINGCHEN])

    def test_generic_connection_error_advances_to_fallback(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == XINGCHEN:
                raise requests.exceptions.ConnectionError("reset")
            return ok_response('{"decision": "recovered"}')

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "recovered"}')
        self.assertEqual(calls, [XINGCHEN, FALLBACK])

    def test_rate_limit_still_advances_to_fallback(self):
        calls = []

        def limited():
            response = Mock()
            response.status_code = 429
            response.text = "rate limited"
            return response

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == FLINT:
                return limited()
            return ok_response('{"decision": "from fallback"}')

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "from fallback"}')
        self.assertEqual(calls, [FLINT, FLINT, FALLBACK])

    def test_timeout_value_uses_constants(self):
        seen = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen.append(timeout)
            return ok_response()

        self.call(responder)
        self.assertEqual(
            seen,
            [(llm_service.OPENROUTER_CONNECT_TIMEOUT, llm_service.OPENROUTER_READ_TIMEOUT)],
        )

    def test_streaming_is_requested(self):
        seen = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen["stream_arg"] = stream
            seen["payload_stream"] = json.get("stream")
            return ok_response()

        self.call(responder)
        self.assertIs(seen["stream_arg"], True)
        self.assertIs(seen["payload_stream"], True)

    def test_reasoning_content_is_excluded_from_output(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(['{"decision": "HOLD"}'], reasoning=["thinking hard", "more thoughts"])

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')

    def test_multiple_content_chunks_are_concatenated(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(['{"stock":', ' "ORCL",', ' "decision": "HOLD"}'])

        out, _ = self.call(responder)
        self.assertEqual(out, '{"stock": "ORCL", "decision": "HOLD"}')

    def test_malformed_sse_lines_are_skipped(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            response = Mock()
            response.status_code = 200
            response.text = ""
            response.iter_lines.return_value = iter([
                b": keep-alive comment",
                b"data: not-json{",
                _sse({"choices": [{"index": 0, "delta": {"content": '{"decision": "HOLD"}'}, "finish_reason": None}]}),
                b"data: [DONE]",
            ])
            return response

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')

    def test_mid_stream_read_timeout_falls_back(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == FLINT:
                response = ok_response()

                def raising_iter(*_args, **_kwargs):
                    yield _sse({"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]})
                    raise requests.exceptions.ReadTimeout("stalled mid-stream")

                response.iter_lines.side_effect = raising_iter
                return response
            return ok_response('{"decision": "from fallback"}')

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "from fallback"}')
        self.assertEqual(calls, [FLINT, FALLBACK])

    def test_gateway_504_retries_and_falls_back(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == FLINT:
                return stream_response(status=504, text="gateway timeout")
            return ok_response('{"decision": "from fallback"}')

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "from fallback"}')
        self.assertEqual(calls, [FLINT, FLINT, FALLBACK])

    def test_finish_reason_without_done_still_terminates(self):
        """Regression: the stream must not require a trailing [DONE] to stop iterating."""
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            response = Mock()
            response.status_code = 200
            response.text = ""
            emitted = []

            def lines():
                emitted.append(_sse({"choices": [{"index": 0, "delta": {"content": '{"decision": "HOLD"}'}, "finish_reason": None}]}))
                yield emitted[-1]
                # Server signals completion but never sends [DONE] and holds the connection open.
                yield _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self.fail("iterator kept yielding after finish_reason")

            response.iter_lines.side_effect = lines
            return response

        out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')

    def test_total_budget_exceeded_falls_back(self):
        seen = []

        def fake_collect(response, model_name, deadline, progress_interval, total_budget):
            seen.append(model_name)
            if model_name == FLINT:
                raise llm_service.TimeBudgetExceeded("budget")
            return '{"decision": "from fallback"}'

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return ok_response()

        with patch.object(llm_service, "_collect_sse_content", fake_collect):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "from fallback"}')
        self.assertEqual(seen, [FLINT, FALLBACK])

    def test_budget_exhaustion_stops_before_next_candidate(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            return ok_response()

        with patch.object(llm_service, "OPENROUTER_TOTAL_TIMEOUT", -1):
            with self.assertRaises(RuntimeError):
                self.call(responder)
        self.assertEqual(calls, [])

    def test_budget_exceeded_on_last_candidate_raises(self):
        def fake_collect(response, model_name, deadline, progress_interval, total_budget):
            raise llm_service.TimeBudgetExceeded("budget")

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return ok_response()

        with patch.object(llm_service, "_collect_sse_content", fake_collect):
            with self.assertRaises(llm_service.TimeBudgetExceeded):
                self.call(responder, model_choice="flint")

    def test_heartbeat_lines_do_not_prevent_budget_check(self):
        response = Mock()
        response.iter_lines.side_effect = lambda: iter([b": ping"] * 1000)
        with self.assertRaises(llm_service.TimeBudgetExceeded):
            llm_service._collect_sse_content(
                response, XINGCHEN, time.monotonic() - 1, 0, 600)

    def test_progress_is_reported_while_streaming(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(['{"decision":', ' "HOLD"', '}'])

        buf = io.StringIO()
        with patch.object(llm_service, "OPENROUTER_PROGRESS_INTERVAL", 0), contextlib.redirect_stdout(buf):
            out, _ = self.call(responder)
        self.assertEqual(out, '{"decision": "HOLD"}')
        self.assertIn("still generating", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

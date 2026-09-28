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

ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
SUPER = "nvidia/nemotron-3-super-120b-a12b"


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


def ok_response(content="hello"):
    return stream_response([content])


class FreeRoutingTests(unittest.TestCase):
    def test_free_is_not_aliased_to_openrouter(self):
        self.assertEqual(normalize_model_key("free"), "free")

    def test_free_targets_ultra_then_super(self):
        config = MODEL_REGISTRY["free"]
        self.assertEqual(config["provider"], "nvidia")
        self.assertEqual(config["model"], ULTRA)
        self.assertEqual(config["fallbacks"], [SUPER])

    def test_free_label_advertises_fallback(self):
        self.assertIn("Super", get_model_label("free"))

    def test_openrouter_aliases_unchanged(self):
        for alias in ("openrouter", "openrouter/free", "minimax", "minimax-m3", "m3", "or"):
            self.assertEqual(normalize_model_key(alias), "openrouter")
            self.assertEqual(MODEL_REGISTRY[normalize_model_key(alias)]["provider"], "openrouter")

    def test_other_nvidia_aliases_unchanged(self):
        self.assertEqual(normalize_model_key("ultra"), "ultra")
        self.assertEqual(normalize_model_key("kimi"), "kimi")
        self.assertEqual(MODEL_REGISTRY["ultra"]["provider"], "nvidia")
        self.assertIsNone(MODEL_REGISTRY["ultra"].get("fallbacks"))

    def test_free_case_insensitive(self):
        self.assertEqual(normalize_model_key("FREE"), "free")


@patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=False)
class ReadTimeoutFallbackTests(unittest.TestCase):
    def call(self, responder, model_choice="free"):
        """Run one query_llm with requests.post and sleep patched, so nothing leaves the process."""
        with patch("llm_service.requests.post", side_effect=responder) as post, patch(
            "llm_service.time.sleep"
        ):
            return query_llm("sys", "user", model_choice=model_choice), post

    def test_read_timeout_falls_back_to_super(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == ULTRA:
                raise requests.exceptions.ReadTimeout("read timed out")
            return ok_response("from super")

        out, post = self.call(responder)
        self.assertEqual(out, "from super")
        self.assertEqual(calls, [ULTRA, SUPER])
        self.assertEqual(post.call_count, 2)

    def test_read_timeout_does_not_retry_same_model(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            raise requests.exceptions.ReadTimeout("read timed out")

        with self.assertRaises(requests.exceptions.ReadTimeout):
            self.call(responder)
        self.assertEqual(calls, [ULTRA, SUPER])

    def test_ultra_success_does_not_touch_super(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            return ok_response("from ultra")

        out, _ = self.call(responder)
        self.assertEqual(out, "from ultra")
        self.assertEqual(calls, [ULTRA])

    def test_generic_connection_error_still_retries(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if calls.count(json["model"]) == 1:
                raise requests.exceptions.ConnectionError("reset")
            return ok_response("recovered")

        out, _ = self.call(responder)
        self.assertEqual(out, "recovered")
        self.assertEqual(calls, [ULTRA, ULTRA])

    def test_rate_limit_still_advances_to_fallback(self):
        calls = []

        def limited():
            response = Mock()
            response.status_code = 429
            response.text = "rate limited"
            return response

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == ULTRA:
                return limited()
            return ok_response("from super")

        out, _ = self.call(responder)
        self.assertEqual(out, "from super")
        self.assertEqual(calls, [ULTRA, ULTRA, SUPER])

    def test_timeout_value_uses_constants(self):
        seen = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen.append(timeout)
            return ok_response()

        self.call(responder)
        self.assertEqual(
            seen,
            [(llm_service.NVIDIA_CONNECT_TIMEOUT, llm_service.NVIDIA_READ_TIMEOUT)],
        )

    def test_streaming_is_requested(self):
        seen = {}

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            seen["stream_arg"] = stream
            seen["payload_stream"] = json.get("stream")
            return ok_response("ok")

        self.call(responder)
        self.assertIs(seen["stream_arg"], True)
        self.assertIs(seen["payload_stream"], True)

    def test_reasoning_content_is_excluded_from_output(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(["final answer"], reasoning=["thinking hard", "more thoughts"])

        out, _ = self.call(responder)
        self.assertEqual(out, "final answer")

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
                b"data: [DONE]",
            ])
            return response

        out, _ = self.call(responder)
        self.assertEqual(out, "")

    def test_mid_stream_read_timeout_falls_back(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == ULTRA:
                response = ok_response("partial")

                def raising_iter(*_args, **_kwargs):
                    yield _sse({"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]})
                    raise requests.exceptions.ReadTimeout("stalled mid-stream")

                response.iter_lines.side_effect = raising_iter
                return response
            return ok_response("from super")

        out, _ = self.call(responder)
        self.assertEqual(out, "from super")
        self.assertEqual(calls, [ULTRA, SUPER])

    def test_gateway_504_is_not_a_read_timeout(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            if json["model"] == ULTRA:
                return stream_response(status=504, text="gateway timeout")
            return ok_response("from super")

        out, _ = self.call(responder)
        self.assertEqual(out, "from super")
        self.assertEqual(calls, [ULTRA, ULTRA, SUPER])


    def test_finish_reason_without_done_still_terminates(self):
        """Regression: the stream must not require a trailing [DONE] to stop iterating."""
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            response = Mock()
            response.status_code = 200
            response.text = ""
            emitted = []

            def lines():
                emitted.append(_sse({"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": None}]}))
                yield emitted[-1]
                # Server signals completion but never sends [DONE] and holds the connection open.
                yield _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self.fail("iterator kept yielding after finish_reason")

            response.iter_lines.side_effect = lines
            return response

        out, _ = self.call(responder)
        self.assertEqual(out, "done")

    def test_total_budget_exceeded_falls_back(self):
        seen = []

        def fake_collect(response, model_name, deadline, progress_interval, total_budget):
            seen.append(model_name)
            if model_name == ULTRA:
                raise llm_service.TimeBudgetExceeded("budget")
            return "from super"

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return ok_response()

        with patch.object(llm_service, "_collect_sse_content", fake_collect):
            out, _ = self.call(responder)
        self.assertEqual(out, "from super")
        self.assertEqual(seen, [ULTRA, SUPER])

    def test_budget_exhaustion_stops_before_next_candidate(self):
        calls = []

        def responder(url, headers=None, json=None, timeout=None, stream=None):
            calls.append(json["model"])
            return ok_response("ok")

        with patch.object(llm_service, "NVIDIA_TOTAL_TIMEOUT", -1):
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
                self.call(responder, model_choice="ultra")

    def test_heartbeat_lines_do_not_prevent_budget_check(self):
        response = Mock()
        response.iter_lines.side_effect = lambda: iter([b": ping"] * 1000)
        with self.assertRaises(llm_service.TimeBudgetExceeded):
            llm_service._collect_sse_content(
                response, ULTRA, time.monotonic() - 1, 0, 600)

    def test_progress_is_reported_while_streaming(self):
        def responder(url, headers=None, json=None, timeout=None, stream=None):
            return stream_response(["a", "b", "c"])

        buf = io.StringIO()
        with patch.object(llm_service, "NVIDIA_PROGRESS_INTERVAL", 0), contextlib.redirect_stdout(buf):
            out, _ = self.call(responder)
        self.assertEqual(out, "abc")
        self.assertIn("still generating", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

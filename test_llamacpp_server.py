import os
import signal
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from llamacpp_provider import LlamaCppConfigurationError
from llamacpp_server import (
    LlamaCppServer,
    LlamaCppServerConfig,
    LlamaCppServerStartupError,
    is_loopback_host,
    load_server_config,
    managed_llamacpp_server,
)


def build_config(**overrides) -> LlamaCppServerConfig:
    values = {
        "binary": Path("/opt/llama.cpp/bin/llama-server"),
        "model_path": Path("/models/qwen2.5-14b-instruct-q4_k_m.gguf"),
        "base_url": "http://127.0.0.1:11434/v1",
        "host": "127.0.0.1",
        "port": 11434,
        "alias": "qwen2.5",
        "context_tokens": 8192,
        "gpu_layers": 99,
        "parallel_slots": 1,
        "startup_timeout_seconds": 0.01,
        "shutdown_timeout_seconds": 5.0,
        "log_path": Path("/tmp/llamacpp_server_test.log"),
    }
    values.update(overrides)
    return LlamaCppServerConfig(**values)


class FakePopen:
    """Minimal stand-in for a llama-server subprocess."""

    def __init__(self, exit_code: int | None = None, ignore_sigterm: bool = False):
        self.pid = 4242
        self.returncode = exit_code
        self.terminated = False
        self.killed = False
        self._ignore_sigterm = ignore_sigterm
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._ignore_sigterm and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class ServerConfigTests(unittest.TestCase):
    def test_config_derives_endpoint_and_reuses_provider_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server"
            binary.touch()
            gguf = Path(tmp) / "model.gguf"
            gguf.touch()
            env = {
                "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1/",
                "LLAMACPP_MODEL": "qwen2.5",
                "LLAMACPP_CONTEXT_TOKENS": "8192",
                "LLAMACPP_SERVER_BIN": str(binary),
                "LLAMACPP_GGUF_PATH": str(gguf),
            }
            with patch.dict(os.environ, env, clear=True):
                config = load_server_config()

        self.assertEqual(config.binary, binary)
        self.assertEqual(config.model_path, gguf)
        self.assertEqual(config.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual((config.host, config.port), ("127.0.0.1", 11434))
        # The client and the server must agree on the model name.
        self.assertEqual(config.alias, "qwen2.5")
        self.assertEqual(config.context_tokens, 8192)

    def test_default_gpu_layers_never_uses_auto_detect(self):
        """-ngl -1 hangs on this Vulkan build, so the default must stay explicit."""
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server"
            binary.touch()
            gguf = Path(tmp) / "model.gguf"
            gguf.touch()
            env = {
                "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1",
                "LLAMACPP_SERVER_BIN": str(binary),
                "LLAMACPP_GGUF_PATH": str(gguf),
            }
            with patch.dict(os.environ, env, clear=True):
                config = load_server_config()
            self.assertEqual(config.gpu_layers, 99)
            self.assertEqual(config.parallel_slots, 1)

            with patch.dict(os.environ, {**env, "LLAMACPP_SERVER_NGL": "-1"}, clear=True):
                with self.assertRaises(LlamaCppConfigurationError):
                    load_server_config()

    def test_missing_binary_override_is_rejected_clearly(self):
        with patch.dict(
            os.environ,
            {
                "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1",
                "LLAMACPP_SERVER_BIN": "/nonexistent/llama-server",
            },
            clear=True,
        ):
            with self.assertRaises(LlamaCppConfigurationError) as ctx:
                load_server_config()
        self.assertIn("LLAMACPP_SERVER_BIN", str(ctx.exception))

    def test_missing_gguf_model_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server"
            binary.touch()
            with patch.dict(
                os.environ,
                {
                    "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1",
                    "LLAMACPP_SERVER_BIN": str(binary),
                    "LLAMACPP_GGUF_PATH": "/nonexistent/model.gguf",
                },
                clear=True,
            ):
                with self.assertRaises(LlamaCppConfigurationError) as ctx:
                    load_server_config()
        self.assertIn("LLAMACPP_GGUF_PATH", str(ctx.exception))

    def test_loopback_detection(self):
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertFalse(is_loopback_host("192.168.1.50"))
        self.assertFalse(is_loopback_host("llm.internal.example.com"))


class ManagedServerSelectionTests(unittest.TestCase):
    def test_cloud_models_get_a_no_op_context(self):
        for model in ("gemini", "kimi", "nemotron", "openrouter", "super"):
            self.assertIsInstance(managed_llamacpp_server(model), nullcontext)

    def test_all_llamacpp_aliases_are_managed(self):
        for model in ("qwen", "llamacpp", "qwen-llamacpp", "QWEN-LLAMACPP"):
            with tempfile.TemporaryDirectory() as tmp:
                binary = Path(tmp) / "llama-server"
                binary.touch()
                gguf = Path(tmp) / "model.gguf"
                gguf.touch()
                env = {
                    "LLAMACPP_BASE_URL": "http://127.0.0.1:11434/v1",
                    "LLAMACPP_SERVER_BIN": str(binary),
                    "LLAMACPP_GGUF_PATH": str(gguf),
                }
                with patch.dict(os.environ, env, clear=True):
                    self.assertIsInstance(
                        managed_llamacpp_server(model), LlamaCppServer
                    )

    def test_non_loopback_endpoint_is_left_to_its_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server"
            binary.touch()
            gguf = Path(tmp) / "model.gguf"
            gguf.touch()
            env = {
                "LLAMACPP_BASE_URL": "http://192.168.1.50:11434/v1",
                "LLAMACPP_SERVER_BIN": str(binary),
                "LLAMACPP_GGUF_PATH": str(gguf),
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertIsInstance(
                    managed_llamacpp_server("qwen-llamacpp"), nullcontext
                )


class ServerLifecycleTests(unittest.TestCase):
    def test_command_matches_the_verified_vulkan_recipe(self):
        command = LlamaCppServer(build_config())._command()

        self.assertEqual(command[0], "/opt/llama.cpp/bin/llama-server")
        self.assertIn("-m", command)
        self.assertIn("/models/qwen2.5-14b-instruct-q4_k_m.gguf", command)
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--port", command)
        self.assertIn("11434", command)
        self.assertEqual(command[command.index("-c") + 1], "8192")
        self.assertEqual(command[command.index("-ngl") + 1], "99")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(
            command[command.index("--flash-attn") + 1 : command.index("--flash-attn") + 2],
            ["on"],
        )

    @patch("llamacpp_server.subprocess.Popen")
    def test_context_manager_starts_polls_and_stops(self, popen):
        process = FakePopen()
        popen.return_value = process
        server = LlamaCppServer(build_config())

        with (
            patch.object(LlamaCppServer, "_probe", side_effect=[False, False, True]) as probe,
            patch("llamacpp_server.os.getpgid", return_value=4242),
            patch("llamacpp_server.os.killpg") as killpg,
            patch("llamacpp_server.time.sleep"),
        ):
            with server:
                self.assertFalse(server.adopted)
                self.assertEqual(probe.call_count, 3)

        self.assertTrue(popen.called)
        self.assertTrue(killpg.called)
        signals = [call.args[1] for call in killpg.call_args_list]
        self.assertIn(signal.SIGTERM, signals)
        self.assertFalse(process.terminated)

        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["cwd"], "/opt/llama.cpp/bin")

    @patch("llamacpp_server.subprocess.Popen")
    def test_running_server_is_adopted_and_never_stopped(self, popen):
        server = LlamaCppServer(build_config())

        with patch.object(LlamaCppServer, "_probe", return_value=True):
            with server:
                self.assertTrue(server.adopted)

        popen.assert_not_called()
        # __exit__ ran, but an adopted server must survive it.
        self.assertIsNone(server._process)

    @patch("llamacpp_server.subprocess.Popen")
    def test_early_exit_reports_code_and_log_tail(self, popen):
        popen.return_value = FakePopen(exit_code=1)
        server = LlamaCppServer(build_config())
        log_path = Path(tempfile.gettempdir()) / "llamacpp_early_exit.log"
        log_path.write_text("ggml_vulkan: device lost\nline two\n")
        server = LlamaCppServer(build_config(log_path=log_path))
        self.addCleanup(log_path.unlink)

        with (
            patch.object(LlamaCppServer, "_probe", return_value=False),
            patch("llamacpp_server.time.sleep"),
        ):
            with self.assertRaises(LlamaCppServerStartupError) as ctx:
                with server:
                    pass

        message = str(ctx.exception)
        self.assertIn("code 1", message)
        self.assertIn("ggml_vulkan: device lost", message)
        server.stop()

    def test_startup_timeout_names_the_log_file(self):
        server = LlamaCppServer(
            build_config(log_path=Path("/tmp/does-not-matter.log"))
        )
        with (
            patch.object(LlamaCppServer, "_probe", return_value=False),
            patch("llamacpp_server.time.sleep"),
            patch("llamacpp_server.subprocess.Popen", return_value=FakePopen()),
        ):
            with self.assertRaises(LlamaCppServerStartupError) as ctx:
                server.start()
                server.wait_until_ready()
                server.stop()

        self.assertIn("/tmp/does-not-matter.log", str(ctx.exception))
        server.stop()

    @patch("llamacpp_server.subprocess.Popen")
    def test_sigterm_ignored_escalates_to_sigkill(self, popen):
        process = FakePopen(ignore_sigterm=True)
        popen.return_value = process
        server = LlamaCppServer(build_config())

        with (
            patch.object(LlamaCppServer, "_probe", return_value=False),
            patch("llamacpp_server.os.getpgid", return_value=4242),
            patch("llamacpp_server.os.killpg") as killpg,
        ):
            server.start()
            server.stop()

        signals = [call.args[1] for call in killpg.call_args_list]
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])

    @patch("llamacpp_server.subprocess.Popen")
    def test_teardown_runs_when_the_body_raises(self, popen):
        process = FakePopen()
        popen.return_value = process
        server = LlamaCppServer(build_config())

        not_running_then_ready = Mock(side_effect=[False, True])
        with (
            patch.object(LlamaCppServer, "_probe", not_running_then_ready),
            patch("llamacpp_server.os.getpgid", return_value=4242),
            patch("llamacpp_server.os.killpg") as killpg,
        ):
            with self.assertRaises(RuntimeError):
                with server:
                    raise RuntimeError("pipeline exploded")

        self.assertFalse(server.adopted)
        self.assertTrue(killpg.called)
        self.assertIsNone(server._process)

    @patch("llamacpp_server.subprocess.Popen")
    def test_stop_is_idempotent(self, popen):
        popen.return_value = FakePopen()
        server = LlamaCppServer(build_config())

        with (
            patch.object(LlamaCppServer, "_probe", return_value=False),
            patch("llamacpp_server.os.getpgid", return_value=4242),
            patch("llamacpp_server.os.killpg"),
        ):
            server.start()
            server.stop()
            server.stop()

        self.assertIsNone(server._process)


class MainEntryPointWiringTests(unittest.TestCase):
    def test_both_main_paths_use_the_managed_server(self):
        source = Path(__file__).with_name("main.py").read_text()

        self.assertEqual(
            source.count("with managed_llamacpp_server("),
            2,
            "expected the analysis and portfolio paths to both manage the server",
        )
        self.assertIn("with managed_llamacpp_server(model_choice):", source)
        self.assertIn("with managed_llamacpp_server(llm_choice):", source)


if __name__ == "__main__":
    unittest.main()

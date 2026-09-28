"""Managed llama.cpp server lifecycle for the local Qwen provider.

When a llama.cpp-backed model alias is selected, `main.py` starts the local
llama-server on demand, blocks until the OpenAI-compatible endpoint answers,
and stops the process again so it never keeps VRAM resident between runs.

An already-listening server is adopted rather than duplicated, and is left
running untouched because this process does not own it.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Callable, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from llamacpp_provider import LlamaCppConfigurationError, LlamaCppError, load_config
from llm_service import uses_compact_prompt_profile

_BINARY_CANDIDATES: Final[tuple[str, ...]] = (
    "~/Desktop/git/Bonsai-demo/bin/vulkan/llama-server",
    "~/Desktop/git/Bonsai-demo/bin/cpu/llama-server",
    "llama.cpp/build/bin/llama-server",
)
_DEFAULT_GGUF_PATH: Final[str] = "~/models/qwen2.5-14b-instruct-q4_k_m.gguf"
_DEFAULT_LOG_PATH: Final[str] = "~/models/llamacpp_server.log"
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "::1"})
_LOG_TAIL_LINES: Final[int] = 20
_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
_PROBE_INTERVAL_SECONDS: Final[float] = 1.0


class LlamaCppServerStartupError(LlamaCppError):
    """Raised when the managed llama.cpp server cannot be brought up."""


@dataclass(frozen=True, slots=True)
class LlamaCppServerConfig:
    binary: Path
    model_path: Path
    base_url: str
    host: str
    port: int
    alias: str
    context_tokens: int
    gpu_layers: int
    parallel_slots: int
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float
    log_path: Path


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise LlamaCppConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise LlamaCppConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise LlamaCppConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise LlamaCppConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _resolve_binary() -> Path:
    override = os.getenv("LLAMACPP_SERVER_BIN", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise LlamaCppConfigurationError(
                f"LLAMACPP_SERVER_BIN does not point to a file: {candidate}"
            )
        return candidate

    tried: list[str] = []
    for raw in _BINARY_CANDIDATES:
        candidate = Path(raw).expanduser()
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate

    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)

    raise LlamaCppConfigurationError(
        "Could not find a llama-server binary. Set LLAMACPP_SERVER_BIN to its "
        f"absolute path. Tried: {', '.join(tried)}, then $PATH."
    )


def _resolve_model_path() -> Path:
    raw = os.getenv("LLAMACPP_GGUF_PATH", "").strip() or _DEFAULT_GGUF_PATH
    candidate = Path(raw).expanduser()
    if not candidate.is_file():
        raise LlamaCppConfigurationError(
            f"GGUF model not found: {candidate}. Set LLAMACPP_GGUF_PATH to the model file."
        )
    return candidate


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().strip("[]")
    return normalized in _LOOPBACK_HOSTS or normalized.startswith("127.")


def load_server_config() -> LlamaCppServerConfig:
    """Build the launch config, reusing the provider's env contract for the endpoint."""
    client_config = load_config()
    parsed = urlparse(client_config.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    return LlamaCppServerConfig(
        binary=_resolve_binary(),
        model_path=_resolve_model_path(),
        base_url=client_config.base_url,
        host=host,
        port=port,
        alias=os.getenv("LLAMACPP_SERVER_ALIAS", "").strip() or client_config.model,
        context_tokens=client_config.context_tokens,
        # -ngl -1 (auto) hangs on this Vulkan build and crashes the host, so the
        # default is an explicit full offload rather than an automatic guess.
        gpu_layers=_env_int("LLAMACPP_SERVER_NGL", 99, minimum=0),
        # More than one slot exhausts UMA memory and severely slows Vulkan.
        parallel_slots=_env_int("LLAMACPP_SERVER_PARALLEL", 1, minimum=1),
        startup_timeout_seconds=_env_float("LLAMACPP_STARTUP_TIMEOUT", 300.0, minimum=1.0),
        shutdown_timeout_seconds=_env_float("LLAMACPP_SHUTDOWN_TIMEOUT", 30.0, minimum=1.0),
        log_path=Path(
            os.getenv("LLAMACPP_SERVER_LOG", "").strip() or _DEFAULT_LOG_PATH
        ).expanduser(),
    )


class LlamaCppServer:
    """Owns one llama-server subprocess for the duration of a run."""

    def __init__(self, config: LlamaCppServerConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None
        self._adopted = False
        self._previous_sigterm: object | None = None

    @property
    def adopted(self) -> bool:
        """True when a pre-existing server was reused instead of started here."""
        return self._adopted

    def _command(self) -> list[str]:
        config = self._config
        return [
            str(config.binary),
            "-m",
            str(config.model_path),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--alias",
            config.alias,
            "-c",
            str(config.context_tokens),
            "-ngl",
            str(config.gpu_layers),
            "--flash-attn",
            "on",
            "-np",
            str(config.parallel_slots),
        ]

    def _probe(self, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
        try:
            with urlopen(f"{self._config.base_url}/models", timeout=timeout) as response:
                return response.status == 200
        except HTTPError:
            # llama.cpp answers 503 while the model is still loading.
            return False
        except (URLError, OSError, ValueError):
            return False

    def _log_tail(self) -> str:
        try:
            lines = self._config.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    def start(self) -> None:
        if self._adopted or self._process is not None:
            return
        if self._probe(timeout=1.0):
            self._adopted = True
            print(
                f"[llama.cpp] Reusing the server already listening on "
                f"{self._config.base_url}; it will be left running."
            )
            return

        config = self._config
        command = self._command()
        try:
            self._log_handle = open(config.log_path, "ab", buffering=0)
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                # Own process group so teardown reaches any child llama.cpp forks.
                start_new_session=True,
                cwd=str(config.binary.parent),
            )
        except OSError as exc:
            self._close_log()
            raise LlamaCppServerStartupError(
                f"Failed to launch {config.binary}: {exc}"
            ) from exc

        atexit.register(self.stop)
        self._install_sigterm_handler()
        print(
            f"[llama.cpp] Starting {config.binary.name} for {config.model_path.name} "
            f"on {config.host}:{config.port} (ctx={config.context_tokens}, "
            f"ngl={config.gpu_layers}, slots={config.parallel_slots})"
        )
        print(f"[llama.cpp] Server log: {config.log_path}")

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._config.startup_timeout_seconds
        while True:
            if self._process is not None and self._process.poll() is not None:
                return_code = self._process.returncode
                tail = self._log_tail()
                raise LlamaCppServerStartupError(
                    f"llama-server exited during startup with code {return_code}."
                    + (f"\nLast log lines:\n{tail}" if tail else "")
                )
            if self._probe():
                print(f"[llama.cpp] Server ready at {self._config.base_url}")
                return
            if time.monotonic() >= deadline:
                raise LlamaCppServerStartupError(
                    f"llama-server did not answer {self._config.base_url}/models within "
                    f"{self._config.startup_timeout_seconds:.0f}s. "
                    f"Check {self._config.log_path}."
                )
            time.sleep(_PROBE_INTERVAL_SECONDS)

    def stop(self) -> None:
        atexit.unregister(self.stop)
        self._restore_sigterm_handler()
        process, self._process = self._process, None
        if process is None or self._adopted:
            self._close_log()
            return
        if process.poll() is None:
            self._signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=self._config.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                print("[llama.cpp] Server ignored SIGTERM; sending SIGKILL.")
                self._signal_group(process, signal.SIGKILL)
                process.wait(timeout=5)
        self._close_log()
        print("[llama.cpp] Server stopped.")

    def _signal_group(
        self, process: "subprocess.Popen[bytes]", sig: signal.Signals
    ) -> None:
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except OSError:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            handle.close()

    def _install_sigterm_handler(self) -> None:
        if self._previous_sigterm is not None:
            return
        try:
            self._previous_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:
            # Signal handlers can only be installed on the main thread.
            self._previous_sigterm = None

    def _restore_sigterm_handler(self) -> None:
        if self._previous_sigterm is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous_sigterm)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            pass
        self._previous_sigterm = None

    def _on_sigterm(self, signum: int, frame: object) -> None:
        previous = self._previous_sigterm
        self.stop()
        restored: Callable[[int, object], None] | int = (
            previous if callable(previous) else signal.SIG_DFL
        )
        signal.signal(signal.SIGTERM, restored)  # type: ignore[arg-type]
        os.kill(os.getpid(), signum)

    def __enter__(self) -> None:
        self.start()
        if not self._adopted:
            self.wait_until_ready()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


def managed_llamacpp_server(model_choice: str) -> AbstractContextManager[None]:
    """Auto-start the local llama.cpp server for llama.cpp-backed model choices.

    Returns a no-op context manager for cloud models and for endpoints that are
    not loopback, because those servers are managed outside this project.
    """
    if not uses_compact_prompt_profile(model_choice):
        return nullcontext()

    config = load_server_config()
    if not is_loopback_host(config.host):
        print(
            f"[llama.cpp] {config.base_url} is not loopback; "
            "assuming the server is managed externally."
        )
        return nullcontext()
    return LlamaCppServer(config)

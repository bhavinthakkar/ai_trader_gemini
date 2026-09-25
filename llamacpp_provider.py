from __future__ import annotations

import math
import os
import socket
import threading
from dataclasses import dataclass
from typing import Final, TypeAlias
from urllib.parse import urlparse

import httpx2

_LOCAL_REQUEST_LOCK: Final = threading.Lock()
_CHARS_PER_TOKEN_ESTIMATE: Final = 3
_MESSAGE_OVERHEAD_TOKENS: Final = 64
_JSON_SCALAR: TypeAlias = str | int | float | bool | None
_JSON_VALUE: TypeAlias = _JSON_SCALAR | list["_JSON_VALUE"] | dict[str, "_JSON_VALUE"]


class LlamaCppError(RuntimeError):
    """Base error for the local llama.cpp provider."""


class LlamaCppConfigurationError(LlamaCppError):
    """Raised when local-provider environment configuration is invalid."""


class LlamaCppPromptBudgetError(LlamaCppError):
    """Raised when the system contract cannot fit inside the model context."""


class LlamaCppHTTPError(LlamaCppError):
    """Raised when the local HTTP request fails."""


class LlamaCppResponseError(LlamaCppError):
    """Raised when the local server returns an unusable response."""


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    base_url: str
    model: str
    api_key: str | None
    connect_timeout_seconds: float
    read_timeout_seconds: float
    write_timeout_seconds: float
    pool_timeout_seconds: float
    max_output_tokens: int
    context_tokens: int
    prompt_safety_margin_tokens: int


def _read_float(name: str, default: float, *, minimum: float) -> float:
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


def _read_int(name: str, default: int, *, minimum: int) -> int:
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


def load_config(model_override: str | None = None) -> LlamaCppConfig:
    base_url = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LlamaCppConfigurationError(
            f"LLAMACPP_BASE_URL must be an absolute HTTP(S) URL, got {base_url!r}"
        )

    model = (model_override or os.getenv("LLAMACPP_MODEL", "qwen2.5")).strip()
    if not model:
        raise LlamaCppConfigurationError("LLAMACPP_MODEL must not be empty")

    context_tokens = _read_int("LLAMACPP_CONTEXT_TOKENS", 8192, minimum=2048)
    max_output_tokens = _read_int("LLAMACPP_MAX_TOKENS", 2048, minimum=128)
    prompt_safety_margin_tokens = _read_int(
        "LLAMACPP_PROMPT_SAFETY_TOKENS", 256, minimum=64
    )
    if max_output_tokens + prompt_safety_margin_tokens >= context_tokens:
        raise LlamaCppConfigurationError(
            "LLAMACPP_MAX_TOKENS + LLAMACPP_PROMPT_SAFETY_TOKENS must be smaller "
            "than LLAMACPP_CONTEXT_TOKENS"
        )

    api_key = os.getenv("LLAMACPP_API_KEY", "").strip() or None
    return LlamaCppConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        connect_timeout_seconds=_read_float("LLAMACPP_CONNECT_TIMEOUT", 5.0, minimum=0.1),
        read_timeout_seconds=_read_float("LLAMACPP_READ_TIMEOUT", 900.0, minimum=1.0),
        write_timeout_seconds=_read_float("LLAMACPP_WRITE_TIMEOUT", 30.0, minimum=0.1),
        pool_timeout_seconds=_read_float("LLAMACPP_POOL_TIMEOUT", 10.0, minimum=0.1),
        max_output_tokens=max_output_tokens,
        context_tokens=context_tokens,
        prompt_safety_margin_tokens=prompt_safety_margin_tokens,
    )


def estimate_text_tokens(text: str) -> int:
    """Conservatively estimate Qwen tokens without loading a tokenizer."""
    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN_ESTIMATE)


def _truncate_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n\n[...LOCAL PROMPT TRUNCATED...]\n\n"
    available = max_chars - len(marker)
    if available < 200:
        raise LlamaCppPromptBudgetError(
            "The local model context leaves too little room for the user prompt"
        )
    head_chars = max(100, int(available * 0.72))
    tail_chars = max(100, available - head_chars)
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def fit_messages_to_budget(
    system_instruction: str,
    user_prompt: str,
    config: LlamaCppConfig,
) -> tuple[str, str, int]:
    """Fit messages into context while preserving the system and output tail."""
    system_tokens = estimate_text_tokens(system_instruction)
    user_tokens = estimate_text_tokens(user_prompt)
    fixed_tokens = (
        system_tokens
        + config.max_output_tokens
        + config.prompt_safety_margin_tokens
        + _MESSAGE_OVERHEAD_TOKENS
    )
    target_tokens = config.context_tokens - config.prompt_safety_margin_tokens
    available_user_tokens = config.context_tokens - fixed_tokens

    if available_user_tokens < 128:
        raise LlamaCppPromptBudgetError(
            "The system instruction leaves insufficient room in LLAMACPP_CONTEXT_TOKENS"
        )
    if system_tokens + user_tokens + _MESSAGE_OVERHEAD_TOKENS + config.max_output_tokens <= target_tokens:
        return system_instruction, user_prompt, system_tokens + user_tokens + _MESSAGE_OVERHEAD_TOKENS + config.max_output_tokens

    max_user_chars = max(256, available_user_tokens * _CHARS_PER_TOKEN_ESTIMATE)
    fitted_user = _truncate_head_tail(user_prompt, max_user_chars)
    for _ in range(8):
        estimated_total = (
            system_tokens
            + estimate_text_tokens(fitted_user)
            + _MESSAGE_OVERHEAD_TOKENS
            + config.max_output_tokens
        )
        if estimated_total <= target_tokens:
            return system_instruction, fitted_user, estimated_total
        max_user_chars = int(max_user_chars * 0.9)
        fitted_user = _truncate_head_tail(user_prompt, max_user_chars)

    raise LlamaCppPromptBudgetError(
        f"Unable to fit local prompt into {config.context_tokens} tokens"
    )


def create_client(config: LlamaCppConfig) -> httpx2.Client:
    limits = httpx2.Limits(
        max_connections=1,
        max_keepalive_connections=1,
        keepalive_expiry=60.0,
    )
    timeout = httpx2.Timeout(
        connect=config.connect_timeout_seconds,
        read=config.read_timeout_seconds,
        write=config.write_timeout_seconds,
        pool=config.pool_timeout_seconds,
    )
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=1,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return httpx2.Client(
        transport=transport,
        timeout=timeout,
        base_url=f"{config.base_url}/",
        headers=headers,
        follow_redirects=True,
    )


def _extract_content(response_json: _JSON_VALUE) -> str:
    if not isinstance(response_json, dict):
        raise LlamaCppResponseError("llama.cpp returned a non-object JSON response")
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaCppResponseError("llama.cpp response did not contain any choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LlamaCppResponseError("llama.cpp returned an invalid first choice")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LlamaCppResponseError("llama.cpp response choice did not contain a message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    raise LlamaCppResponseError("llama.cpp response message content was not text")


def query_llamacpp(
    *,
    system_instruction: str,
    user_prompt: str,
    model_override: str | None = None,
    temperature: float | None = None,
) -> str:
    config = load_config(model_override=model_override)
    fitted_system, fitted_user, estimated_total = fit_messages_to_budget(
        system_instruction,
        user_prompt,
        config,
    )
    if estimated_total > config.context_tokens - config.prompt_safety_margin_tokens:
        raise LlamaCppPromptBudgetError("Local prompt budget calculation exceeded the context window")

    effective_temperature = (
        float(temperature)
        if temperature is not None
        else _read_float("LLAMACPP_TEMPERATURE", 0.2, minimum=0.0)
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": fitted_system},
            {"role": "user", "content": fitted_user},
        ],
        "temperature": effective_temperature,
        "max_tokens": config.max_output_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    try:
        with _LOCAL_REQUEST_LOCK:
            with create_client(config) as client:
                response = client.post("chat/completions", json=payload)
                response.raise_for_status()
                response_json = response.json()
    except httpx2.HTTPError as exc:
        raise LlamaCppHTTPError(f"llama.cpp request failed: {exc}") from exc
    except ValueError as exc:
        raise LlamaCppResponseError("llama.cpp returned invalid JSON") from exc

    return _extract_content(response_json)

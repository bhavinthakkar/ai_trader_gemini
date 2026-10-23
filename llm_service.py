import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

NVIDIA_READ_TIMEOUT = int(os.getenv("NVIDIA_READ_TIMEOUT", "300"))
NVIDIA_CONNECT_TIMEOUT = int(os.getenv("NVIDIA_CONNECT_TIMEOUT", "30"))
NVIDIA_TOTAL_TIMEOUT = int(os.getenv("NVIDIA_TOTAL_TIMEOUT", "600"))
NVIDIA_PROGRESS_INTERVAL = int(os.getenv("NVIDIA_PROGRESS_INTERVAL", "15"))

OPENROUTER_READ_TIMEOUT = int(os.getenv("OPENROUTER_READ_TIMEOUT", "300"))
OPENROUTER_CONNECT_TIMEOUT = int(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "30"))
OPENROUTER_TOTAL_TIMEOUT = int(os.getenv("OPENROUTER_TOTAL_TIMEOUT", "600"))
OPENROUTER_PROGRESS_INTERVAL = int(os.getenv("OPENROUTER_PROGRESS_INTERVAL", "15"))


class TimeBudgetExceeded(RuntimeError):
    """A streamed response outlived the total wall-clock budget."""


def _collect_sse_content(response, label, deadline, progress_interval, total_budget) -> str:
    parts = []
    started = time.monotonic()
    last_report = started
    token_count = 0

    for raw_line in response.iter_lines():
        now = time.monotonic()
        if now > deadline:
            raise TimeBudgetExceeded(
                f"{label} exceeded the {total_budget}s total budget"
            )
        if now - last_report >= progress_interval:
            last_report = now
            print(f"[llm_service] {label} still generating: {now - started:.0f}s elapsed, "
                  f"~{token_count} content tokens received")
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", "ignore") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for choice in chunk.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content") or ""
            if piece:
                parts.append(piece)
                token_count += 1
            if choice.get("finish_reason"):
                return "".join(parts)
    return "".join(parts)
    return "".join(parts)


MODEL_REGISTRY = {
    "flint": {
        "provider": "openrouter",
        "model": "recursive/flint-v1.0:free",
        "fallbacks": ["openrouter/free"],
        "label": "Recursive: Flint v1.0 (OpenRouter, free)"
    },
    "recursive/flint-v1.0:free": {
        "provider": "openrouter",
        "model": "recursive/flint-v1.0:free",
        "fallbacks": ["openrouter/free"],
        "label": "Recursive: Flint v1.0 (OpenRouter, free)"
    },
    "openrouter": {
        "provider": "openrouter",
        "model": "recursive/flint-v1.0:free",
        "fallbacks": ["openrouter/free"],
        "label": "Recursive: Flint v1.0 (OpenRouter, free)"
    },
    "free": {
        "provider": "openrouter",
        "model": "recursive/flint-v1.0:free",
        "fallbacks": ["openrouter/free"],
        "label": "Recursive: Flint v1.0 (OpenRouter, free)"
    },
}


def clean_think_tags(text: str) -> str:
    """
    Strips reasoning tokens from reasoning models before JSON parsing:
    - XML-style <think>...</think> blocks
    - model-specific " thinking ... response " markers
    and removes markdown code fence blocks if present.
    """
    if not text:
        return ""
    cleaned = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r' thinking.*? response', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def extract_json(text: str):
    """
    Safely parses JSON from an LLM response string.
    Strips markdown code fences, think tags, leading/trailing commentary,
    handles unescaped control characters, missing commas, unescaped quotes,
    trailing commas, and extra trailing data.
    """
    if not text or not str(text).strip():
        raise ValueError("Empty response string from LLM")

    cleaned = clean_think_tags(text)

    # Attempt 1: Standard json.loads
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Attempt 2: Use json_repair library if available
    try:
        from json_repair import repair_json
        parsed = repair_json(cleaned, return_objects=True)
        if parsed:
            return parsed
    except Exception:
        pass

    start_brace = cleaned.find("{")
    start_bracket = cleaned.find("[")

    if start_brace == -1 and start_bracket == -1:
        raise ValueError(f"No JSON object or array found in response: {cleaned[:100]}")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
    else:
        start_idx = start_bracket

    end_brace = cleaned.rfind("}")
    end_bracket = cleaned.rfind("]")
    end_idx = max(end_brace, end_bracket)

    if end_idx > start_idx:
        json_candidate = cleaned[start_idx:end_idx + 1]
        try:
            return json.loads(json_candidate, strict=False)
        except Exception:
            pass

        # Attempt to repair missing commas and trailing commas
        fixed_candidate = re.sub(r'("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null)\s*\n\s*(")', r'\1,\n\2', json_candidate)
        fixed_candidate = re.sub(r',\s*([\}\]])', r'\1', fixed_candidate)
        try:
            return json.loads(fixed_candidate, strict=False)
        except Exception:
            pass

        try:
            from json_repair import repair_json
            parsed = repair_json(json_candidate, return_objects=True)
            if parsed:
                return parsed
        except Exception:
            pass

    try:
        decoder = json.JSONDecoder(strict=False)
        obj, _ = decoder.raw_decode(cleaned[start_idx:])
        return obj
    except Exception as e:
        raise e


def normalize_model_key(model_choice: str) -> str:
    if not model_choice:
        return "flint"
    key = str(model_choice).strip().lower()
    if key in ["flint", "flint-v1.0", "recursive/flint-v1.0:free"]:
        return "flint"
    if key in ["openrouter", "openrouter/free", "or", "free"]:
        return "flint"
    # Gracefully redirect any other model request to flint
    return "flint"


def get_model_label(model_choice: str) -> str:
    key = normalize_model_key(model_choice)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["label"]
    return "Recursive: Flint v1.0 (OpenRouter, free)"


def uses_compact_prompt_profile(model_choice: str) -> bool:
    if not model_choice:
        return False
    return str(model_choice).strip().lower() in ["qwen", "llamacpp", "qwen-llamacpp"]


def query_llm(
    system_instruction: str,
    user_prompt: str,
    model_choice: str = "flint",
    temperature: float = None,
    reasoning_budget: int = None,
    reasoning_effort: str = None
) -> str:
    """
    Executes an LLM chat query using Recursive: Flint v1.0 via OpenRouter.
    """
    key = normalize_model_key(model_choice or "flint")
    config = MODEL_REGISTRY.get(key, MODEL_REGISTRY["flint"])
    provider = config.get("provider", "openrouter")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set in .env")

    eff_temp = float(temperature) if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))
    eff_effort = (str(reasoning_effort) if reasoning_effort is not None
                  else config.get("default_reasoning_effort"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bhavinthakkar/ai_trader_gemini",
        "X-Title": "AI Trader Gemini"
    }

    candidates = [config["model"]] + config.get("fallbacks", [])
    last_error = None
    chain_deadline = time.monotonic() + OPENROUTER_TOTAL_TIMEOUT

    for candidate_model in candidates:
        if time.monotonic() > chain_deadline:
            print(f"[llm_service] OpenRouter total budget of {OPENROUTER_TOTAL_TIMEOUT}s exhausted; "
                  f"not attempting '{candidate_model}'.")
            break

        payload = {
            "model": candidate_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": eff_temp,
            "response_format": {"type": "json_object"},
            "stream": True
        }
        if eff_effort:
            payload["reasoning_effort"] = eff_effort
        for attempt in range(2):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(OPENROUTER_CONNECT_TIMEOUT, OPENROUTER_READ_TIMEOUT),
                    stream=True
                )
                if response.status_code == 200:
                    cleaned = clean_think_tags(_collect_sse_content(
                        response, candidate_model, chain_deadline,
                        OPENROUTER_PROGRESS_INTERVAL, OPENROUTER_TOTAL_TIMEOUT))
                    try:
                        _ = extract_json(cleaned)
                        return cleaned
                    except Exception as parse_err:
                        print(f"[llm_service] OpenRouter model '{candidate_model}' returned "
                              f"non-JSON/invalid output ({parse_err}). Trying fallback...")
                        last_error = parse_err
                        break
                elif response.status_code == 429:
                    if attempt == 0:
                        print(f"[llm_service] OpenRouter model '{candidate_model}' rate limited (429). Retrying in 2s...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"[llm_service] OpenRouter model {candidate_model} returned 429: {response.text[:200]}. Trying fallback...")
                        last_error = RuntimeError(f"OpenRouter call failed (429): {response.text}")
                        break
                elif response.status_code in [502, 503, 504]:
                    if attempt == 0:
                        print(f"[llm_service] OpenRouter model '{candidate_model}' returned {response.status_code}. Retrying in 2s...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"[llm_service] OpenRouter model {candidate_model} returned {response.status_code}. Trying fallback...")
                        last_error = RuntimeError(f"OpenRouter call failed ({response.status_code}): {response.text}")
                        break
                else:
                    print(f"[llm_service] OpenRouter model {candidate_model} returned {response.status_code}. Trying fallback...")
                    last_error = RuntimeError(f"OpenRouter API call failed ({response.status_code}): {response.text}")
                    break
            except TimeBudgetExceeded as e:
                last_error = e
                print(f"[llm_service] OpenRouter model {candidate_model} hit the "
                      f"{OPENROUTER_TOTAL_TIMEOUT}s total budget. Trying fallback...")
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"[llm_service] OpenRouter connection error with {candidate_model}: {e}. Trying fallback...")
                break

    if last_error:
        raise last_error
    raise RuntimeError("OpenRouter query failed across all candidate models.")

import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_REGISTRY = {
    "ultra": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "label": "Nemotron-3 Ultra 550B (NVIDIA)"
    },
    "nemotron-ultra": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "label": "Nemotron-3 Ultra 550B (NVIDIA)"
    },
    "550b": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "label": "Nemotron-3 Ultra 550B (NVIDIA)"
    },
    "kimi": {
        "provider": "nvidia",
        "model": "moonshotai/kimi-k3",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"],
        "label": "Moonshot AI Kimi-K3 (NVIDIA)"
    },
    "kimi-k3": {
        "provider": "nvidia",
        "model": "moonshotai/kimi-k3",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"],
        "label": "Moonshot AI Kimi-K3 (NVIDIA)"
    },
    "k3": {
        "provider": "nvidia",
        "model": "moonshotai/kimi-k3",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"],
        "label": "Moonshot AI Kimi-K3 (NVIDIA)"
    },
    "nemotron": {
        "provider": "nvidia",
        "model": os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b", "moonshotai/kimi-k3"],
        "label": "Nemotron-3 Ultra 550B (NVIDIA)"
    },
    "nvidia": {
        "provider": "nvidia",
        "model": os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b", "moonshotai/kimi-k3"],
        "label": "Nemotron-3 Ultra 550B (NVIDIA)"
    },
    "super": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "moonshotai/kimi-k3"],
        "label": "Nemotron-3 Super 120B (NVIDIA)"
    },
    "nemotron-super": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "moonshotai/kimi-k3"],
        "label": "Nemotron-3 Super 120B (NVIDIA)"
    },
    "120b": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "fallbacks": ["nvidia/nemotron-3-ultra-550b-a55b", "moonshotai/kimi-k3"],
        "label": "Nemotron-3 Super 120B (NVIDIA)"
    },
    "gemini": {
        "provider": "gemini",
        "primary": "gemini-3.1-pro-preview",
        "fallbacks": ["gemini-3.5-flash", "gemini-2.5-flash"],
        "label": "Gemini 3.1 Pro"
    },
    "qwen": {
        "provider": "llamacpp",
        "model": os.getenv("LLAMACPP_MODEL", "qwen2.5"),
        "label": "Qwen 2.5 14B (llama.cpp)"
    },
    "qwen-llamacpp": {
        "provider": "llamacpp",
        "model": os.getenv("LLAMACPP_MODEL", "qwen2.5"),
        "label": "Qwen 2.5 14B (llama.cpp)"
    },
    "llamacpp": {
        "provider": "llamacpp",
        "model": os.getenv("LLAMACPP_MODEL", "qwen2.5"),
        "label": "Qwen 2.5 14B (llama.cpp)"
    },
    "openrouter": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b:free", "minimax/minimax-m3:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    },
    "free": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b:free", "minimax/minimax-m3:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    },
    "openrouter/free": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b:free", "minimax/minimax-m3:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    },
    "minimax": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    },
    "minimax-m3": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    },
    "m3": {
        "provider": "openrouter",
        "model": "openrouter/free",
        "fallbacks": ["minimax/minimax-m3:free", "nvidia/nemotron-3-super-120b-a12b:free"],
        "label": "OpenRouter Free Models Router (openrouter/free)"
    }
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
    key = str(model_choice).strip().lower()
    if key in ["free", "openrouter/free", "or", "minimax", "minimax-m3", "minimax_m3", "m3"]:
        key = "openrouter"
    if key in ["550b", "nemotron-ultra", "ultra"]:
        key = "ultra"
    if key in ["120b", "nemotron-super", "super"]:
        key = "super"
    if key in ["kimi", "kimi-k3", "k3", "moonshot"]:
        key = "kimi"
    if key in ["qwen-llamacpp", "llamacpp", "qwen2.5", "qwen2.5-14b"]:
        key = "qwen"
    return key


def get_model_label(model_choice: str) -> str:
    key = normalize_model_key(model_choice)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["label"]
    return model_choice


def uses_compact_prompt_profile(model_choice: str) -> bool:
    key = normalize_model_key(model_choice)
    config = MODEL_REGISTRY.get(key)
    return bool(config and config.get("provider") == "llamacpp")


def query_llm(
    system_instruction: str,
    user_prompt: str,
    model_choice: str,
    temperature: float = None,
    reasoning_budget: int = None,
    reasoning_effort: str = None
) -> str:
    """
    Executes an LLM chat query based on the model short name.
    Supported model short names:
      - 'nemotron' / 'ultra' / '550b': Cloud Nemotron-3 Ultra 550B (NVIDIA API)
      - 'kimi' / 'kimi-k3' / 'k3': Moonshot AI Kimi-K3 (NVIDIA API)
      - 'super' / '120b': Cloud Nemotron-3 Super 120B (NVIDIA API)
      - 'gemini': Cloud Gemini 3.1 Pro
      - 'openrouter' / 'free': OpenRouter Free Models Router (openrouter/free)
      - 'qwen' / 'llamacpp': Local Qwen model through the OpenAI-compatible llama.cpp server
    """
    if not model_choice or not str(model_choice).strip():
        raise ValueError("Model choice argument is required. Valid choices: 'nemotron', 'ultra', 'kimi', 'gemini', 'openrouter', 'qwen', 'llamacpp'")

    key = normalize_model_key(model_choice)

    if key not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model choice '{model_choice}'. Choose one of: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[key]
    provider = config["provider"]

    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set in .env")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        candidates = [config.get("model")] + config.get("fallbacks", [])
        eff_temp = float(temperature) if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.6"))
        eff_budget = int(reasoning_budget) if reasoning_budget is not None else int(os.getenv("REASONING_BUDGET", "16000"))
        eff_effort = str(reasoning_effort) if reasoning_effort is not None else os.getenv("REASONING_EFFORT", "high")

        import time
        last_error = None

        for model_idx, model_name in enumerate(candidates):
            if not model_name:
                continue

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": eff_temp,
                "top_p": 0.95,
                "max_tokens": eff_budget
            }

            if "kimi" in model_name.lower():
                payload["reasoning_effort"] = eff_effort if eff_effort in ["low", "medium", "high", "max"] else "max"
            elif "ultra" not in model_name.lower():
                payload["reasoning_effort"] = eff_effort
                payload["reasoning_budget"] = eff_budget

            for attempt in range(2):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=120)
                    if response.status_code == 200:
                        res_json = response.json()
                        content = res_json["choices"][0]["message"].get("content") or ""
                        return clean_think_tags(content)
                    elif response.status_code == 400 and "thinking_token_budget" in response.text:
                        payload.pop("reasoning_effort", None)
                        payload.pop("reasoning_budget", None)
                        continue
                    elif response.status_code == 429:
                        if attempt == 0:
                            print(f"[llm_service] NVIDIA NIM model '{model_name}' rate limited (429). Retrying in 2s...")
                            time.sleep(2)
                            continue
                        elif model_idx < len(candidates) - 1:
                            next_model = candidates[model_idx + 1]
                            print(f"[llm_service] NVIDIA NIM model '{model_name}' rate limited (429). Switching to fallback model '{next_model}'...")
                            break
                        else:
                            last_error = RuntimeError(f"NVIDIA NIM model '{model_name}' rate limited (429): {response.text}")
                    elif response.status_code in [502, 503, 504]:
                        print(f"[llm_service] NVIDIA NIM model '{model_name}' returned {response.status_code}. Retrying...")
                        time.sleep(3)
                        continue
                    else:
                        raise RuntimeError(f"NVIDIA API call failed ({response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    last_error = e
                    print(f"[llm_service] NVIDIA connection error for '{model_name}': {e}. Retrying...")
                    time.sleep(3)
                    continue

        if last_error:
            raise last_error
        raise RuntimeError("All NVIDIA NIM candidate models failed or were rate-limited.")

    elif provider == "llamacpp":
        from llamacpp_provider import query_llamacpp

        return clean_think_tags(
            query_llamacpp(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                model_override=config["model"],
                temperature=temperature,
            )
        )

    elif provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("The 'google-genai' python package is required for Gemini model execution. Run: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set in .env")

        eff_temp = float(temperature) if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))

        client = genai.Client(api_key=api_key)
        candidate_models = [config["primary"]] + config.get("fallbacks", [])

        last_error = None
        for m in candidate_models:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=eff_temp,
                        response_mime_type="application/json"
                    )
                )
                return response.text
            except Exception as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[llm_service] Quota/rate limit reached for {m}. Trying fallback model...")
                    continue
                raise e

        if last_error:
            raise last_error

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set in .env")

        eff_temp = float(temperature) if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bhavinthakkar/ai_trader_gemini",
            "X-Title": "AI Trader Gemini"
        }

        candidates = [config["model"]] + config.get("fallbacks", [])
        last_error = None

        for candidate_model in candidates:
            payload = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": eff_temp,
                "response_format": {"type": "json_object"}
            }
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    res_json = response.json()
                    choices = res_json.get("choices", [])
                    actual_model = res_json.get("model", candidate_model)
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "")
                        cleaned = clean_think_tags(content)
                        # Validate that returned text can be parsed as JSON
                        try:
                            _ = extract_json(cleaned)
                            return cleaned
                        except Exception as parse_err:
                            print(f"[llm_service] OpenRouter model '{candidate_model}' (routed to '{actual_model}') returned non-JSON/invalid output ({parse_err}). Trying fallback...")
                            last_error = parse_err
                            continue
                    else:
                        print(f"[llm_service] OpenRouter model '{candidate_model}' returned empty choices. Trying fallback...")
                        continue
                elif response.status_code in [429, 502, 503, 504]:
                    print(f"[llm_service] OpenRouter model {candidate_model} returned {response.status_code}: {response.text[:200]}. Trying fallback...")
                    last_error = RuntimeError(f"OpenRouter call failed ({response.status_code}): {response.text}")
                    continue
                else:
                    print(f"[llm_service] OpenRouter model {candidate_model} returned {response.status_code}. Trying fallback...")
                    last_error = RuntimeError(f"OpenRouter API call failed ({response.status_code}): {response.text}")
                    continue
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"[llm_service] OpenRouter connection error with {candidate_model}: {e}. Trying fallback...")
                continue

        if last_error:
            raise last_error
        raise RuntimeError("OpenRouter query failed across all candidate models.")

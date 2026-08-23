import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_REGISTRY = {
    "nemotron": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "label": "Nemotron-3 Super 120B (NVIDIA)"
    },
    "nvidia": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "label": "Nemotron-3 Super 120B (NVIDIA)"
    },
    "gemini": {
        "provider": "gemini",
        "primary": "gemini-3.1-pro-preview",
        "fallbacks": ["gemini-3.5-flash", "gemini-2.5-flash"],
        "label": "Gemini 3.1 Pro"
    },
    "gemma": {
        "provider": "ollama",
        "model": "gemma4:12b",
        "label": "gemma4:12b (Ollama)"
    },
    "qwen": {
        "provider": "ollama",
        "model": "qwen2.5:14b",
        "label": "qwen2.5:14b (Ollama)"
    },
    "twostage": {
        "provider": "ollama_twostage",
        "extraction_model": os.getenv("TWOSTAGE_EXTRACTION_MODEL", "qwen2.5:14b"),
        "reasoning_model": os.getenv("TWOSTAGE_REASONING_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
        "label": "2-Stage Local (qwen2.5:14b + qwen3:30b-a3b-instruct-2507-q4_K_M)"
    }
}


def clean_think_tags(text: str) -> str:
    """
    Strips <think>...</think> reasoning tokens from reasoning models (e.g., DeepSeek-R1)
    and removes markdown code fence blocks if present.
    """
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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


def get_model_label(model_choice: str) -> str:
    key = str(model_choice).strip().lower()
    if key == "local":
        key = "twostage"
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["label"]
    return model_choice


def query_llm(system_instruction: str, user_prompt: str, model_choice: str) -> str:
    """
    [System: Do not output <think> tags or internal reasoning steps. Go directly to the answer.]
    Executes an LLM chat query based on the model short name.
    Supported model short names:
      - 'nemotron' / 'nvidia': Cloud Nemotron-3 Super 120B (NVIDIA API)
      - 'gemini': Cloud Gemini 3.1 Pro
      - 'twostage' / 'local': 2-Stage Local (Stage 1 extraction via qwen2.5:14b, Stage 2 reasoning via qwen3:30b-a3b-instruct-2507-q4_K_M)
      - 'gemma': Local Ollama model 'gemma4:12b'
      - 'qwen': Local Ollama model 'qwen2.5:14b'
    """
    if not model_choice or not str(model_choice).strip():
        raise ValueError("Model choice argument is required. Valid choices: 'nemotron', 'gemini', 'twostage', 'gemma', 'qwen'")

    key = str(model_choice).strip().lower()
    if key == "local":
        key = "twostage"

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
        model_name = config.get("model", "nvidia/nemotron-3-super-120b-a12b")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 16000,
            "reasoning_effort": "high",
            "reasoning_budget": 16000
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"NVIDIA API call failed ({response.status_code}): {response.text}")

        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return clean_think_tags(content)

    elif provider == "ollama_twostage":
        try:
            from ollama import chat as ollama_chat
        except ImportError:
            raise ImportError("The 'ollama' python package is required for local model execution. Run: pip install ollama")

        # Stage 1: Key Metrics & Data Extraction Specialist (qwen2.5:14b)
        extraction_model = config["extraction_model"]
        print(f"[llm_service] [Stage 1: Extraction] Running {extraction_model}...")
        extraction_prompt = f"""
Analyze the following multi-agent stock market intelligence payload containing data from MarketAgent, InstitutionalDataAgent, MacroDataAgent, NewsAgent, RiskAgent, and AnalystAgent.
Extract and preserve ALL key quantitative metrics, SEC EDGAR filings, FRED yield curve & CFTC COT macro data, technical indicators, news catalysts, risk factors, and analyst ratings into a structured JSON object.

Input Data:
{user_prompt}
"""
        stage1_response = ollama_chat(
            model=extraction_model,
            messages=[
                {"role": "system", "content": "You are a financial data extraction specialist. Extract and preserve ALL 6 sub-agent datasets into a single clean JSON object."},
                {"role": "user", "content": extraction_prompt}
            ],
            format="json",
            keep_alive=0
        )
        extracted_data_str = stage1_response["message"]["content"]
        
        # Validate / Parse Stage 1 JSON if possible
        try:
            extracted_json = extract_json(extracted_data_str)
            extracted_input = json.dumps(extracted_json)
        except Exception:
            extracted_input = extracted_data_str

        # Pause briefly to ensure Ollama releases Stage 1 VRAM
        import time
        time.sleep(1)

        # Stage 2: Reasoning Engine
        reasoning_model = config["reasoning_model"]
        print(f"[llm_service] [Stage 2: Reasoning Engine] Running {reasoning_model}...")
        
        reasoning_num_gpu_env = os.getenv("TWOSTAGE_REASONING_NUM_GPU") or os.getenv("OLLAMA_NUM_GPU")
        reasoning_options = {}
        if reasoning_num_gpu_env is not None:
            reasoning_options["num_gpu"] = int(reasoning_num_gpu_env)
        elif "30b" in reasoning_model.lower():
            reasoning_options["num_gpu"] = 12

        try:
            stage2_response = ollama_chat(
                model=reasoning_model,
                messages=[
                    {"role": "system", "content": f"{system_instruction}\n\nCRITICAL MANDATE: Your output MUST be a valid JSON object matching the requested schema. Ensure 'decision' (BUY|SELL|HOLD), 'confidence' (float 0 to 1), 'reason', 'institutional_data', 'macro_data', 'news', 'investment_bank_coverage', 'risk_assessment', and 'PE_and_PEG' are all populated."},
                    {"role": "user", "content": f"Structured Financial Intelligence Data:\n{extracted_input}\n\nSynthesize the insights from all sub-agents and return ONLY a valid JSON object matching the required schema."}
                ],
                format="json",
                options=reasoning_options if reasoning_options else None,
                keep_alive=0
            )
            raw_reasoning_out = stage2_response["message"]["content"]
        except Exception as e:
            fallback_model = "deepseek-r1:14b" if reasoning_model != "deepseek-r1:14b" else "qwen2.5:14b"
            print(f"[llm_service] Model {reasoning_model} failed (likely GPU VRAM OOM: {e}). Falling back to {fallback_model}...")
            try:
                stage2_response = ollama_chat(
                    model=fallback_model,
                    messages=[
                        {"role": "system", "content": f"{system_instruction}\n\nCRITICAL MANDATE: Your output MUST be a valid JSON object matching the requested schema. Ensure 'decision' (BUY|SELL|HOLD), 'confidence' (float 0 to 1), 'reason', 'institutional_data', 'macro_data', 'news', 'investment_bank_coverage', 'risk_assessment', and 'PE_and_PEG' are all populated."},
                        {"role": "user", "content": f"Structured Financial Intelligence Data:\n{extracted_input}\n\nSynthesize the insights from all sub-agents and return ONLY a valid JSON object matching the required schema."}
                    ],
                    format="json",
                    keep_alive=0
                )
                raw_reasoning_out = stage2_response["message"]["content"]
            except Exception as e2:
                secondary_fallback = "qwen2.5:14b"
                print(f"[llm_service] Fallback model {fallback_model} also failed ({e2}). Falling back to {secondary_fallback}...")
                stage2_response = ollama_chat(
                    model=secondary_fallback,
                    messages=[
                        {"role": "system", "content": f"{system_instruction}\n\nCRITICAL MANDATE: Your output MUST be a valid JSON object matching the requested schema. Ensure 'decision' (BUY|SELL|HOLD), 'confidence' (float 0 to 1), 'reason', 'institutional_data', 'macro_data', 'news', 'investment_bank_coverage', 'risk_assessment', and 'PE_and_PEG' are all populated."},
                        {"role": "user", "content": f"Structured Financial Intelligence Data:\n{extracted_input}\n\nSynthesize the insights from all sub-agents and return ONLY a valid JSON object matching the required schema."}
                    ],
                    format="json",
                    keep_alive=0
                )
                raw_reasoning_out = stage2_response["message"]["content"]

        cleaned_out = clean_think_tags(raw_reasoning_out)
        return cleaned_out

    elif provider == "ollama":
        try:
            from ollama import chat as ollama_chat
        except ImportError:
            raise ImportError("The 'ollama' python package is required for local model execution. Run: pip install ollama")

        model_name = config["model"]
        response_obj = ollama_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            format="json"
        )
        return clean_think_tags(response_obj["message"]["content"])

    elif provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("The 'google-genai' python package is required for Gemini model execution. Run: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set in .env")

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

    elif provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set in .env")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "top_p": 0.7
        }
        try:
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Error querying NVIDIA NIM API: {e}")

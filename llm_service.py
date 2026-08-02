import os
import json
from dotenv import load_dotenv

load_dotenv()

MODEL_REGISTRY = {
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
        "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "label": "qwen3:30b-a3b (Ollama)"
    }
}


def get_model_label(model_choice: str) -> str:
    key = str(model_choice).strip().lower()
    if key == "local":
        key = "gemma"
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["label"]
    return model_choice


def query_llm(system_instruction: str, user_prompt: str, model_choice: str) -> str:
    """
    Executes an LLM chat query based on the model short name.
    Supported model short names:
      - 'gemini': Gemini 3.1 Pro (via Gemini API)
      - 'gemma': Local Ollama model 'gemma4:12b'
      - 'qwen': Local Ollama model 'qwen3:30b-a3b-instruct-2507-q4_K_M'
    """
    if not model_choice or not str(model_choice).strip():
        raise ValueError("Model choice argument is required. Valid choices: 'gemini', 'gemma', 'qwen'")

    key = str(model_choice).strip().lower()
    if key == "local":
        key = "gemma"

    if key not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model choice '{model_choice}'. Choose one of: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[key]
    provider = config["provider"]

    if provider == "ollama":
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
        return response_obj["message"]["content"]

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

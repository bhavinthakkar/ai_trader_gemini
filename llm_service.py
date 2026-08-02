import os
import json
from dotenv import load_dotenv

load_dotenv()


def query_llm(system_instruction: str, user_prompt: str, model_choice: str = "gemini") -> str:
    """
    Executes an LLM chat query.
    - If model_choice.lower() == 'local', uses Ollama with model 'gemma4:12b'.
    - Otherwise, uses Gemini API with model 'gemini-3.6-flash' (with automatic fallback to 'gemini-2.5-flash' on 429 rate limit).
    """
    is_local = (str(model_choice).strip().lower() == "local")

    if is_local:
        try:
            from ollama import chat as ollama_chat
        except ImportError:
            raise ImportError("The 'ollama' python package is required for local model execution. Run: pip install ollama")

        model_name = "gemma4:12b"
        response_obj = ollama_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            format="json"
        )
        return response_obj["message"]["content"]
    else:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("The 'google-genai' python package is required for Gemini model execution. Run: pip install google-genai")

        model_name = "gemini-3.6-flash"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set in .env")

        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json"
                )
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[llm_service] Quota reached for {model_name}. Falling back to gemini-2.5-flash...")
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json"
                    )
                )
                return response.text
            raise e

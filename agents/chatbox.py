import os
import time

import openai
from openai import APIStatusError, BadRequestError, InternalServerError

# Settings come from environment variables. main.py loads them from the .env file.
#   OPENAI_API_KEY      required
#   OPENAI_MODEL        default "gpt-5-mini" (the model used in the paper)
#   OPENAI_BASE_URL     optional, for an OpenAI-compatible endpoint
#   OPENAI_TEMPERATURE  optional, empty = model default
DEFAULT_MODEL = "gpt-5-mini"
PLACEHOLDER_KEY = "your-openai-api-key-here"

_client = None
_send_temperature = True


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key or api_key == PLACEHOLDER_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put your key in the .env file "
                "(copy .env.example to .env if the file does not exist)."
            )
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        _client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_model():
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def get_temperature():
    value = os.environ.get("OPENAI_TEMPERATURE", "").strip()
    return float(value) if value else None


def openai_chat_completion(prompt, model=None, temperature=None, max_attempts=5):
    global _send_temperature
    client = get_client()
    model = model or get_model()
    if temperature is None:
        temperature = get_temperature()

    for attempt in range(max_attempts):
        kwargs = {"model": model, "messages": prompt}
        if temperature is not None and _send_temperature:
            kwargs["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except BadRequestError as e:
            # Some models (for example gpt-5-mini on api.openai.com) accept only the
            # default temperature. Stop sending it and try again.
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                print(f"⚠️ Model {model} does not accept temperature={temperature}. "
                      "Using the model default temperature.")
                _send_temperature = False
                continue
            raise
        except (APIStatusError, InternalServerError) as e:
            # Retry on 5xx
            if getattr(e, "status_code", 500) >= 500:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except Exception as e:
            # Retry if server returned HTML instead of JSON
            if "unexpected mimetype" in str(e).lower():
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("ChatCompletion failed after retries")

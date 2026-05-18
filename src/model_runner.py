"""
model_runner.py
---------------
Sends prompts to Ollama and returns raw model responses.

Responsibilities:
  - Call Ollama's REST API with system + user prompt
  - Handle retries on timeout or connection failure
  - Return structured response dict with timing and metadata
  - Verify Ollama is running before experiments start
  - Never modify the prompt — only sends what it receives

Rules:
  - All timeouts and retries come from config/models.yaml
  - No hardcoded values
  - Imports only from utils.py and stdlib/third-party
"""

import time
from typing import Optional

import requests
from dotenv import load_dotenv

from utils import load_yaml

# Load .env so OLLAMA_BASE_URL is available if set
load_dotenv()

import os

# ─────────────────────────────────────────────────────────────
# OLLAMA ENDPOINT
# ─────────────────────────────────────────────────────────────

def _get_ollama_url() -> str:
    """
    Returns the Ollama base URL from environment or default.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_generate_url() -> str:
    return f"{_get_ollama_url()}/api/generate"


def _get_chat_url() -> str:
    return f"{_get_ollama_url()}/api/chat"


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────

def check_ollama_running() -> dict:
    """
    Checks if the Ollama server is running and reachable.

    Returns:
        {
            'running':  bool,
            'url':      str,
            'message':  str,
        }
    """
    url = _get_ollama_url()
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return {"running": True, "url": url, "message": "Ollama is running."}
        else:
            return {
                "running": False,
                "url": url,
                "message": f"Ollama responded with status {resp.status_code}.",
            }
    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "url": url,
            "message": (
                f"Could not connect to Ollama at {url}.\n"
                f"  Start it with: ollama serve"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "running": False,
            "url": url,
            "message": f"Connection to Ollama at {url} timed out.",
        }


def check_model_available(model_name: str) -> dict:
    """
    Checks if a specific model is available in Ollama.

    Args:
        model_name: e.g. 'llama3.2:3b'

    Returns:
        {
            'available': bool,
            'model':     str,
            'message':   str,
        }
    """
    url = f"{_get_ollama_url()}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]

        # Check for exact or prefix match (e.g. 'llama3.2:3b' may appear as 'llama3.2:3b-instruct-q4_0')
        found = any(model_name in m for m in model_names)

        if found:
            return {"available": True, "model": model_name, "message": "Model is available."}
        else:
            return {
                "available": False,
                "model": model_name,
                "message": (
                    f"Model '{model_name}' not found in Ollama.\n"
                    f"  Available models: {model_names}\n"
                    f"  Pull it with: ollama pull {model_name}"
                ),
            }
    except Exception as e:
        return {
            "available": False,
            "model": model_name,
            "message": f"Error checking model availability: {e}",
        }


# ─────────────────────────────────────────────────────────────
# CORE RUNNER
# ─────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    system_text: str,
    user_text: str,
    temperature: float = 0.0,
    config_path: str = "config/models.yaml",
) -> dict:
    """
    Sends a prompt to Ollama and returns the response.

    Uses the /api/chat endpoint with system + user roles.
    Retries on connection/timeout errors as per models.yaml config.

    Args:
        model_name:   Ollama model name, e.g. 'llama3.2:3b'
        system_text:  System role content (can be empty string).
        user_text:    User role content.
        temperature:  Sampling temperature (0.0 = deterministic).
        config_path:  Path to models.yaml for retry/timeout settings.

    Returns:
        {
            'success':       bool,
            'response_text': str,    # model's raw output (empty string on failure)
            'model_name':    str,
            'temperature':   float,
            'latency_ms':    int,    # wall-clock time in milliseconds
            'error':         str,    # empty string on success
            'attempt':       int,    # which attempt succeeded (1-based)
        }
    """
    cfg = load_yaml(config_path)
    request_cfg   = cfg.get("request", {})
    timeout_s     = request_cfg.get("timeout_seconds", 120)
    max_retries   = request_cfg.get("max_retries", 3)
    retry_delay_s = request_cfg.get("retry_delay_seconds", 5)

    # Build the messages list for /api/chat
    messages = []
    if system_text and system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": user_text.strip()})

    payload = {
        "model":   model_name,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "seed":        42,   # fixed seed for reproducibility
        },
    }

    last_error = ""
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            resp = requests.post(
                _get_chat_url(),
                json=payload,
                timeout=timeout_s,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if resp.status_code != 200:
                last_error = (
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                )
                print(
                    f"[model_runner] Attempt {attempt}/{max_retries} failed: {last_error}"
                )
                time.sleep(retry_delay_s)
                continue

            data = resp.json()
            response_text = (
                data.get("message", {}).get("content", "")
                or data.get("response", "")
            ).strip()

            if not response_text:
                last_error = "Model returned empty response."
                print(
                    f"[model_runner] Attempt {attempt}/{max_retries}: empty response from model."
                )
                time.sleep(retry_delay_s)
                continue

            return {
                "success":       True,
                "response_text": response_text,
                "model_name":    model_name,
                "temperature":   temperature,
                "latency_ms":    latency_ms,
                "error":         "",
                "attempt":       attempt,
            }

        except requests.exceptions.Timeout:
            latency_ms = int((time.time() - start_time) * 1000)
            last_error = f"Request timed out after {timeout_s}s."
            print(f"[model_runner] Attempt {attempt}/{max_retries}: timeout.")
            time.sleep(retry_delay_s)

        except requests.exceptions.ConnectionError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            last_error = f"Connection error: {str(e)[:200]}"
            print(
                f"[model_runner] Attempt {attempt}/{max_retries}: connection error. "
                f"Is Ollama running? (ollama serve)"
            )
            time.sleep(retry_delay_s)

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            last_error = f"Unexpected error: {str(e)[:200]}"
            print(f"[model_runner] Attempt {attempt}/{max_retries}: unexpected error: {e}")
            time.sleep(retry_delay_s)

    # All retries exhausted
    return {
        "success":       False,
        "response_text": "",
        "model_name":    model_name,
        "temperature":   temperature,
        "latency_ms":    0,
        "error":         last_error,
        "attempt":       max_retries,
    }


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: RUN FROM PROMPT DICT
# ─────────────────────────────────────────────────────────────

def run_from_prompt_dict(
    prompt: dict,
    model_id: str,
    temperature: float = 0.0,
    models_config_path: str = "config/models.yaml",
) -> dict:
    """
    Runs a model using a prompt dict returned by prompt_builder.build_prompt().

    Args:
        prompt:             Dict from build_prompt() with 'system' and 'user' keys.
        model_id:           Model ID, e.g. 'model_3b' or 'model_8b'.
        temperature:        Sampling temperature.
        models_config_path: Path to models.yaml.

    Returns:
        run_model() result dict, extended with:
          - 'strategy':    str
          - 'task_type':   str
          - 'noise_level': str
          - 'record_id':   str  (if present in prompt dict)
          - 'model_id':    str
          - 'token_counts': dict
    """
    # Resolve model_name from model_id
    cfg = load_yaml(models_config_path)
    models_list = cfg.get("models", [])
    model_cfg = next((m for m in models_list if m["id"] == model_id), None)

    if model_cfg is None:
        valid = [m["id"] for m in models_list]
        raise ValueError(
            f"[model_runner] Model ID '{model_id}' not found in models.yaml. "
            f"Valid IDs: {valid}"
        )

    model_name = model_cfg["name"]

    result = run_model(
        model_name=model_name,
        system_text=prompt.get("system", ""),
        user_text=prompt["user"],
        temperature=temperature,
        config_path=models_config_path,
    )

    # Enrich result with experiment metadata
    result["strategy"]     = prompt.get("strategy", "")
    result["task_type"]    = prompt.get("task_type", "")
    result["noise_level"]  = prompt.get("noise_level", "")
    result["record_id"]    = prompt.get("record_id", "")
    result["model_id"]     = model_id
    result["token_counts"] = prompt.get("token_counts", {})

    return result


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/model_runner.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== model_runner.py self-test ===\n")

    # 1. Health check
    health = check_ollama_running()
    print(f"[1] Ollama running: {health['running']} — {health['message']}")

    if not health["running"]:
        print("\n⚠️  Ollama is not running. Start it with: ollama serve")
        print("   Skipping live model tests.\n")
    else:
        # 2. Check model availability
        model_status = check_model_available("llama3.2:3b")
        print(f"[2] Model available: {model_status['available']} — {model_status['message']}")

        if model_status["available"]:
            # 3. Simple prompt test
            result = run_model(
                model_name="llama3.2:3b",
                system_text="You are a helpful assistant.",
                user_text="Reply with exactly: OLLAMA_OK",
                temperature=0.0,
            )
            print(f"[3] run_model success: {result['success']}")
            print(f"    Response: {result['response_text'][:100]}")
            print(f"    Latency: {result['latency_ms']}ms")
            assert result["success"], f"Model call failed: {result['error']}"

            # 4. run_from_prompt_dict
            mock_prompt = {
                "system":      "",
                "user":        "Reply with exactly: PROMPT_DICT_OK",
                "strategy":    "monolithic",
                "task_type":   "json_task",
                "noise_level": "short",
                "record_id":   "test_001",
                "token_counts": {"system": 0, "user": 10, "total": 10},
            }
            result2 = run_from_prompt_dict(mock_prompt, model_id="model_3b", temperature=0.0)
            print(f"[4] run_from_prompt_dict success: {result2['success']}")
            assert result2["strategy"] == "monolithic"
            assert result2["model_id"] == "model_3b"
            print(f"    Metadata fields present ✓")
        else:
            print("   Skipping live run tests — model not pulled yet.")

    # 5. Invalid model_id raises
    try:
        run_from_prompt_dict({"system": "", "user": "test"}, model_id="bad_model")
        assert False, "Should raise"
    except ValueError as e:
        print(f"[5] Invalid model_id raises ValueError ✓")

    print("\n✅  model_runner.py self-test complete.")

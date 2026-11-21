"""Model clients. Every call degrades gracefully: no key or no network -> None.

  heron_sql()   Heron (our fine-tuned Qwen2.5-Coder-7B) on Modal, OpenAI-compatible vLLM
  chat_json()   understanding: Groq first, Gemini free tier as backup
  chat_text()   free-text completion on the same providers (SQL fallback, polishing)
"""

import json
import re
import threading
import time

import requests

from .settings import (GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL, HERON_MODE,
                       HERON_MODEL, HERON_TIMEOUT, HERON_URL)

_state = {"heron_down_until": 0.0, "heron_last_ms": None, "heron_last_error": None,
          "groq_down_until": 0.0, "gemini_down_until": 0.0}
_lock = threading.Lock()
DOWN_FOR = 300            # after a failure, skip that provider for 5 minutes


def heron_enabled():
    return HERON_MODE != "off" and time.time() >= _state["heron_down_until"]


def heron_status():
    return dict(_state)


def heron_sql(system, user, timeout=None, max_tokens=320):
    """-> (completion_text | None, info dict)."""
    info = {"provider": "heron", "model": HERON_MODEL}
    if not heron_enabled():
        info["error"] = "Heron disabled or recently unreachable"
        return None, info
    # Heron's offline evaluation decoded with plain greedy SamplingParams. `vllm serve`
    # otherwise applies the merged repo's generation_config.json (repetition_penalty 1.1),
    # which penalises reusing ")" in SQL. Pin the offline settings explicitly.
    body = {"model": HERON_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "repetition_penalty": 1.0, "top_k": -1, "top_p": 1.0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{HERON_URL}/v1/chat/completions", json=body,
                          timeout=timeout or HERON_TIMEOUT)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        info["ms"] = (time.perf_counter() - t0) * 1000
        _state["heron_last_ms"] = info["ms"]
        _state["heron_last_error"] = None
        return text, info
    except Exception as e:                                         # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        info["ms"] = (time.perf_counter() - t0) * 1000
        with _lock:
            _state["heron_down_until"] = time.time() + DOWN_FOR
            _state["heron_last_error"] = info["error"]
        return None, info


def heron_wake(timeout=None):
    """One tiny request to pull Heron out of scale-to-zero (~77 s cold)."""
    _state["heron_down_until"] = 0.0
    return heron_sql("Respond with SQL only.", "Schema:\nCREATE TABLE t (x INTEGER)\n\n"
                     "External knowledge: \nQuestion: How many rows are in t?",
                     timeout=timeout or HERON_TIMEOUT, max_tokens=24)


# --------------------------------------------------------------------- hosted LLMs
def hosted_available():
    now = time.time()
    return bool((GROQ_API_KEY and now >= _state["groq_down_until"])
                or (GEMINI_API_KEY and now >= _state["gemini_down_until"]))


def _groq(system, user, json_mode, max_tokens=700):
    body = {"model": GROQ_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=30,
                      headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _gemini(system, user, json_mode, max_tokens=700):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY}, json=body, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def chat_text(system, user, json_mode=False, max_tokens=700):
    """-> (text | None, provider | None). Groq first, then Gemini."""
    now = time.time()
    for name, key, fn in (("groq", GROQ_API_KEY, _groq), ("gemini", GEMINI_API_KEY, _gemini)):
        if not key or now < _state[f"{name}_down_until"]:
            continue
        try:
            return fn(system, user, json_mode, max_tokens), name
        except Exception:                                          # noqa: BLE001
            _state[f"{name}_down_until"] = time.time() + 60
    return None, None


def chat_json(system, user):
    text, provider = chat_text(system, user, json_mode=True)
    if not text:
        return None, None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0) if m else text), provider
    except (json.JSONDecodeError, AttributeError):
        return None, None

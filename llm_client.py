import functools
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pydantic import ValidationError

log = logging.getLogger(__name__)

RETRYABLE = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
RUNS_DIR = Path("runs")
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "10"))
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "90"))
_gemini = {"client": None, "last_call": 0.0}


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("GEMINI_API_KEY not set (env var or .env)")


def _gemini_chat(prompt: str, model_class, temperature: float, meta: dict) -> str:
    import httpx
    from google import genai
    from google.genai import errors, types

    if _gemini["client"] is None:
        _gemini["client"] = genai.Client(
            api_key=_gemini_key(),
            http_options=types.HttpOptions(
                timeout=int(GEMINI_TIMEOUT_S * 1000),  # milliseconds
                retry_options=types.HttpRetryOptions(
                    attempts=1
                ),  # no hidden SDK retries
            ),
        )
    for attempt in range(1, 4):
        wait = 60 / GEMINI_RPM - (time.monotonic() - _gemini["last_call"])
        if wait > 0:
            time.sleep(wait)
        _gemini["last_call"] = time.monotonic()
        meta["attempts"] += 1
        try:
            resp = _gemini["client"].models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": model_class.model_json_schema(),
                    "temperature": temperature,
                },
            )
            return resp.text or ""
        except httpx.TimeoutException:
            log.warning("gemini timeout after %ss", GEMINI_TIMEOUT_S)
            raise  # fail fast: retries would burn daily quota
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            log.warning("gemini error %s: %s", code, str(exc)[:200])
            if code != 429 or attempt == 3:
                raise
            time.sleep(30)


def with_retries(max_attempts=3, base_delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE as exc:
                    if attempt == max_attempts:
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    log.warning(
                        "retry %d/%d after %s: %s (waiting %.1fs)",
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def save_run(record: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    with open(RUNS_DIR / "raw_outputs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_run(
    kind, prompt, temperature, meta, started, response=None, parsed=None, error=None
):
    save_run(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "model": MODEL,
            "temperature": temperature,
            "prompt": prompt,
            "attempts": meta["attempts"],
            "duration_s": round(time.perf_counter() - started, 3),
            "response": response,
            "parsed": parsed,
            "error": error,
        }
    )


def normalize(text: str) -> str:
    return text.strip().strip(".!?\"' ").lower()


@with_retries(max_attempts=3)
def _chat(payload: dict, meta: dict) -> dict:
    meta["attempts"] += 1
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def ask(prompt: str, temperature: float = 0.0) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    meta = {"attempts": 0}
    started = time.perf_counter()
    try:
        data = _chat(payload, meta)
    except Exception as exc:
        _log_run("ask", prompt, temperature, meta, started, error=repr(exc))
        raise
    _log_run("ask", prompt, temperature, meta, started, response=data)
    return data["message"]["content"]


def ask_structured(prompt: str, model_class, temperature: float = 0.0):
    meta = {"attempts": 0}
    started = time.perf_counter()
    try:
        if PROVIDER == "gemini":
            raw = _gemini_chat(prompt, model_class, temperature, meta)
            data = {"message": {"content": raw}}
        else:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": model_class.model_json_schema(),
                "options": {"temperature": temperature},
            }
            data = _chat(payload, meta)
    except Exception as exc:
        _log_run("ask_structured", prompt, temperature, meta, started, error=repr(exc))
        raise
    try:
        result = model_class.model_validate_json(data["message"]["content"])
    except ValidationError as exc:
        _log_run(
            "ask_structured",
            prompt,
            temperature,
            meta,
            started,
            response=data,
            error=repr(exc),
        )
        raise
    _log_run(
        "ask_structured",
        prompt,
        temperature,
        meta,
        started,
        response=data,
        parsed=result.model_dump(),
    )
    return result


# ---- assistant provider: send the raw question to the assistant (POST /api/chat) ----
ASSISTANT_URL = os.environ.get("ASSISTANT_URL", "http://localhost:3100").rstrip("/")
ASSISTANT_RPM = float(os.environ.get("ASSISTANT_RPM", "10"))
ASSISTANT_TIMEOUT_S = float(os.environ.get("ASSISTANT_TIMEOUT_S", "120"))
_assistant = {"last_call": 0.0}


class InvalidReply(Exception):
    """The assistant refused the model's reply (HTTP 502, error invalid_model_reply)."""


def assistant_health() -> dict:
    resp = requests.get(f"{ASSISTANT_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def use_assistant() -> dict:
    """Name results after the system under test, e.g. assistant:gemini/gemini-3.5-flash-lite."""
    global MODEL
    health = assistant_health()
    MODEL = f"assistant:{health['provider']}/{health['model']}"
    return health


def _error_code(resp):
    try:
        return resp.json().get("error")
    except (ValueError, AttributeError):
        return None


def ask_assistant(question: str, model_class):
    wait = 60 / ASSISTANT_RPM - (time.monotonic() - _assistant["last_call"])
    if wait > 0:
        time.sleep(wait)
    _assistant["last_call"] = time.monotonic()
    meta = {"attempts": 1}  # no retries: the server may already have spent model quota
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{ASSISTANT_URL}/api/chat",
            json={"message": question},
            timeout=ASSISTANT_TIMEOUT_S,
        )
    except Exception as exc:
        _log_run("ask_assistant", question, 0.0, meta, started, error=repr(exc))
        raise
    data = {"status": resp.status_code, "body": resp.text}
    if resp.status_code == 502 and _error_code(resp) == "invalid_model_reply":
        _log_run(
            "ask_assistant",
            question,
            0.0,
            meta,
            started,
            response=data,
            error="invalid_model_reply",
        )
        raise InvalidReply("invalid_model_reply")
    if not resp.ok:
        err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        _log_run(
            "ask_assistant", question, 0.0, meta, started, response=data, error=err
        )
        raise RuntimeError(err)
    try:
        result = model_class.model_validate_json(resp.text)
    except ValidationError as exc:
        _log_run(
            "ask_assistant",
            question,
            0.0,
            meta,
            started,
            response=data,
            error=repr(exc),
        )
        raise
    _log_run(
        "ask_assistant",
        question,
        0.0,
        meta,
        started,
        response=data,
        parsed=result.model_dump(),
    )
    return result

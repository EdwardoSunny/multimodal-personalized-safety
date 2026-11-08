"""Bounded retries with exponential backoff for transient API errors."""

import random
import time

import httpx
import openai

MAX_ATTEMPTS = 5
BASE_DELAY = 2.0  # seconds; attempt n waits BASE_DELAY * 2**(n-1), plus up to 25% jitter

# Request timeout, conflict, rate limit; every 5xx is transient as well
_TRANSIENT_STATUS = {408, 409, 429}


def is_transient_error(exc: BaseException) -> bool:
    """True for connection problems, timeouts, rate limits and server-side (5xx) errors.

    Covers the OpenAI SDK (also used for vLLM servers) and google-genai, whose errors carry
    the HTTP status in ``code``.
    """
    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return True
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    return isinstance(status, int) and (status in _TRANSIENT_STATUS or status >= 500)


def call_with_retries(fn, *args, max_attempts: int = MAX_ATTEMPTS, base_delay: float = BASE_DELAY, **kwargs):
    """Call ``fn(*args, **kwargs)``, retrying transient errors up to ``max_attempts`` times in total.

    Non-transient errors, and the last transient one, are re-raised unchanged.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts or not is_transient_error(e):
                raise
            delay = base_delay * 2 ** (attempt - 1) * (1 + 0.25 * random.random())
            print(f"Transient error ({type(e).__name__}: {e}); retry {attempt}/{max_attempts - 1} in {delay:.1f}s")
            time.sleep(delay)

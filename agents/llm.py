"""Thin provider-agnostic LLM client (Anthropic / OpenAI via REST). Runs in heuristic mode if no key is set."""
import json
import logging
import re
import threading
import time

import requests

from settings import settings

log = logging.getLogger(__name__)


class LLMRateLimitError(RuntimeError):
    """Provider is temporarily refusing requests; callers should use their fallback path."""


class LLM:
    def __init__(self):
        p = settings.LLM_PROVIDER.lower()
        if p == "auto":
            p = ("anthropic" if settings.ANTHROPIC_API_KEY else
                 "openai" if (settings.OPENAI_API_KEY or settings.LLM_API_KEY or settings.LLM_BASE_URL) else "none")
        self.provider = "openai" if p == "openai_compatible" else p
        self.base = (settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        self.key = settings.LLM_API_KEY or settings.OPENAI_API_KEY
        self.model = settings.LLM_MODEL or ("claude-haiku-5-5" if p == "anthropic" else "gpt-4o-mini")
        self._blocked_until = 0.0
        self._rate_lock = threading.Lock()

    @property
    def available(self):
        return self.provider in ("anthropic", "openai")

    def complete(self, prompt, system="", max_tokens=1500, temperature=0.2):
        if not self.available:
            raise RuntimeError("LLM not configured")
        with self._rate_lock:
            if time.time() < self._blocked_until:
                raise LLMRateLimitError("LLM provider is temporarily rate-limited")
        for attempt in range(3):
            try:
                if self.provider == "anthropic":
                    r = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                              "system": system, "messages": [{"role": "user", "content": prompt}]},
                        timeout=90)
                    if r.status_code == 429:
                        self._mark_rate_limited(r)
                        raise LLMRateLimitError("LLM provider rate-limited the request")
                    if r.status_code in (500, 502, 503, 529):
                        raise requests.HTTPError(str(r.status_code))
                    r.raise_for_status()
                    return "".join(b.get("text", "") for b in r.json()["content"])
                r = requests.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                    json={"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                    timeout=90)
                if r.status_code == 429:
                    self._mark_rate_limited(r)
                    raise LLMRateLimitError("LLM provider rate-limited the request")
                if r.status_code in (500, 502, 503):
                    raise requests.HTTPError(str(r.status_code))
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except LLMRateLimitError:
                raise
            except requests.HTTPError as e:
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise RuntimeError(f"LLM provider rejected the request (HTTP {e.response.status_code})") from e
                log.warning("LLM call failed (%s), attempt %d", e, attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt)
            except Exception as e:  # retry transient connection/server failures
                log.warning("LLM call failed (%s), attempt %d", e, attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError("LLM call failed after retries")

    def _mark_rate_limited(self, response):
        try:
            delay = min(120, max(15, int(response.headers.get("Retry-After", "60"))))
        except (TypeError, ValueError):
            delay = 60
        with self._rate_lock:
            self._blocked_until = max(self._blocked_until, time.time() + delay)

    def json(self, prompt, system="", max_tokens=1500):
        """Return parsed JSON or None (callers must have a heuristic fallback)."""
        if not self.available:
            return None
        system = (system + "\nRespond with a single valid JSON object only. No prose, no markdown fences.").strip()
        try:
            txt = self.complete(prompt, system, max_tokens, temperature=0.1)
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
            return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        except Exception as e:
            log.warning("LLM JSON parse failed: %s", e)
            return None

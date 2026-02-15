"""
Google Gemini API client for answer generation.

The pipeline flow:
  1. RAG retrieves relevant paper chunks
  2. ScaleDown compresses the context (reduces tokens & cost)
  3. Gemini generates the actual answer / COT / verification / critique
"""

from __future__ import annotations

import json
import time
import requests
from typing import Optional

from src.core.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiRateLimitError(Exception):
    """Raised when Gemini API rate limit is exhausted after all retries."""
    pass


class GeminiClient:
    """Thin wrapper around the Gemini REST API (generateContent)."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "your_gemini_api_key_here")

    def chat(
        self, system: str, user: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        thinking_budget: int = 2048,
    ) -> str:
        """
        Send a request to Gemini generateContent and return the text response.

        Maps (system, user) to Gemini's content format:
          - system_instruction for the system prompt
          - contents for the user message
        """
        if not self.is_configured:
            raise RuntimeError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY in your .env file. "
                "Get one free at https://aistudio.google.com/apikey"
            )

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": thinking_budget},
            },
        }

        # Retry with exponential backoff on 429 / 5xx
        max_retries = 5
        for attempt in range(max_retries):
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(5 * (2 ** attempt), 60)  # 5, 10, 20, 40, 60s
                try:
                    detail = resp.json().get("error", {}).get("message", "")
                except Exception:
                    detail = resp.text[:200]
                print(f"  [Gemini] {resp.status_code} — retrying in {wait}s … ({detail[:120]})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            # All retries exhausted — raise typed error for fallback logic
            if resp.status_code == 429:
                raise GeminiRateLimitError(
                    f"Gemini rate limit exhausted after {max_retries} retries. "
                    f"Falling back to ScaleDown."
                )
            resp.raise_for_status()  # 5xx or other error
        data = resp.json()

        # Extract text from response (skip thinking parts from 2.5-flash)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            # Gemini 2.5 returns thinking + text parts; grab the last text part
            text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
            if text_parts:
                return text_parts[-1]
            # Fallback: return any text part
            for p in parts:
                if "text" in p:
                    return p["text"]
            return f"**Empty response from Gemini**\n\n```json\n{json.dumps(data, indent=2)}\n```"
        except (KeyError, IndexError) as exc:
            return f"**Gemini response parse error:** {exc}\n\n```json\n{json.dumps(data, indent=2)}\n```"


def make_gemini_llm_fn(
    api_key: Optional[str] = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.3,
):
    """
    Create an LLMFunction compatible with the workflow handler factories.

    Returns a callable with signature (system: str, user: str) -> str.
    """
    client = GeminiClient(api_key=api_key, model=model)

    def llm_fn(system: str, user: str) -> str:
        return client.chat(system, user, temperature=temperature)

    return llm_fn


def make_gemini_fast_fn(
    api_key: Optional[str] = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    thinking_budget: int = 1024,
):
    """
    Create a faster/cheaper LLM function for verification & critique.
    Uses lower output cap and thinking budget to reduce latency.
    """
    client = GeminiClient(api_key=api_key, model=model)

    def llm_fn(system: str, user: str) -> str:
        return client.chat(
            system, user,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

    return llm_fn

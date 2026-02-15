"""ScaleDown API client for context compression and LLM-backed generation fallback."""

import json
import requests
from typing import Optional

from src.core.config import SCALEDOWN_URL, SCALEDOWN_API_KEY, SCALEDOWN_MODEL, SCALEDOWN_TIMEOUT


class ScaleDownClient:
    """Wrapper around the ScaleDown compression API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = SCALEDOWN_MODEL,
        url: str = SCALEDOWN_URL,
        timeout: int = SCALEDOWN_TIMEOUT,
    ):
        self.api_key = api_key or SCALEDOWN_API_KEY
        self.model = model
        self.url = url
        self.timeout = timeout
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def compress(
        self,
        context: str,
        prompt: str = "",
        rate: str = "auto",
    ) -> dict:
        """
        Compress context through ScaleDown.

        Returns dict with keys:
            compressed_text  – the compressed output
            successful       – bool
            original_tokens  – int
            compressed_tokens – int
            latency_ms       – int
        """
        payload = {
            "context": context,
            "prompt": prompt or "Compress this content preserving key information",
            "model": self.model,
            "scaledown": {"rate": rate},
        }

        try:
            resp = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("successful"):
                return {
                    "compressed_text": data["compressed_prompt"],
                    "successful": True,
                    "original_tokens": data.get("original_prompt_tokens", 0),
                    "compressed_tokens": data.get("compressed_prompt_tokens", 0),
                    "latency_ms": data.get("latency_ms", 0),
                }
            return {
                "compressed_text": context,
                "successful": False,
                "original_tokens": 0,
                "compressed_tokens": 0,
                "latency_ms": 0,
            }

        except (requests.RequestException, KeyError, ValueError) as exc:
            return {
                "compressed_text": context,
                "successful": False,
                "error": str(exc),
                "original_tokens": 0,
                "compressed_tokens": 0,
                "latency_ms": 0,
            }

    def generate(self, system: str, user: str) -> str:
        """
        Use ScaleDown compression as a pseudo-generation fallback.

        Strategy: We pass the system prompt as context and the user query as
        the prompt.  ScaleDown's internal model (gpt-4o) processes both,
        producing a compressed output that captures the key information
        relevant to the query.  We then wrap it in markdown.

        This is NOT a full LLM generation — it's a smart extraction/summary
        that uses ScaleDown's compression intelligence as a fallback when
        the primary LLM (Gemini) is unavailable due to rate limits.
        """
        # Build a rich context from system + user so ScaleDown's model sees both
        context = f"{system}\n\n---\n\nUser Question: {user}"
        prompt_instruction = (
            "Answer the user question using the context provided. "
            "Preserve all key facts, citations, and technical details. "
            "Format as clear markdown."
        )

        result = self.compress(
            context=context,
            prompt=prompt_instruction,
            rate="auto",
        )

        if result["successful"] and result["compressed_text"]:
            compressed = result["compressed_text"]
            return (
                f"{compressed}\n\n"
                f"---\n"
                f"*Generated via ScaleDown context compression "
                f"(Gemini unavailable — {result['original_tokens']} → "
                f"{result['compressed_tokens']} tokens)*"
            )
        else:
            # Last resort: return whatever raw text we have
            return (
                f"> **Note:** Both Gemini and ScaleDown generation are unavailable.\n\n"
                f"**Your question:** {user}\n\n"
                f"**System context provided:**\n{system[:2000]}"
            )

    @property
    def is_configured(self) -> bool:
        """Check whether a real API key is set."""
        return bool(self.api_key) and self.api_key != "YOUR_API_KEY"


def make_scaledown_llm_fn(
    api_key: Optional[str] = None,
    model: str = SCALEDOWN_MODEL,
):
    """
    Create an LLMFunction that uses ScaleDown compression as a fallback.

    Returns a callable with signature (system: str, user: str) -> str.
    """
    client = ScaleDownClient(api_key=api_key, model=model)

    def llm_fn(system: str, user: str) -> str:
        return client.generate(system, user)

    return llm_fn

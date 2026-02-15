"""
LLM-agnostic stage handlers for the reasoning workflow.

Provides pluggable handler factories for chain-of-thought, self-verification,
and self-critique.  By default handlers produce structured markdown from the
retrieved context — swap in any LLM backend by passing a custom `llm_fn`.
"""

from __future__ import annotations

import textwrap
from typing import Callable, Optional

# Type alias: any function that takes (system_prompt, user_prompt) → str
LLMFunction = Callable[[str, str], str]


# ── Default (offline) LLM stub ──────────────────────────────────────────────── #

def default_llm_fn(system: str, user: str) -> str:
    """
    Template-based fallback when no external LLM is wired in.
    Extracts structure from the user prompt and wraps it in markdown.
    """
    return (
        f"> *Generated without an external LLM — plug one in via `llm_fn`*\n\n"
        f"### System instructions\n{textwrap.shorten(system, 500)}\n\n"
        f"### Input received\n{textwrap.shorten(user, 1000)}\n"
    )


# ── Handler factories ──────────────────────────────────────────────────────── #

def make_cot_handler(
    context: str,
    llm_fn: Optional[LLMFunction] = None,
) -> Callable[[str], str]:
    """Return a handler that produces a chain-of-thought markdown trace."""
    fn = llm_fn or default_llm_fn

    def handler(question: str) -> str:
        system = (
            "You are a scientific research assistant. "
            "Think step-by-step and format your response as markdown.\n\n"
            "CITATION RULES (mandatory):\n"
            "- Every factual claim MUST have an inline citation like [arxiv:XXXX.XXXXX]\n"
            "- Quote or closely paraphrase the source text that supports each claim\n"
            "- If a claim has no supporting source, explicitly mark it as "
            "[unsupported — general knowledge]\n"
            "- End with a ## References section listing all cited sources\n\n"
            f"## Context\n{context}"
        )
        user = (
            f"## Research Question\n{question}\n\n"
            "Provide a detailed chain-of-thought analysis with proper citations."
        )
        return fn(system, user)

    return handler


def make_verify_handler(
    context: str,
    llm_fn: Optional[LLMFunction] = None,
) -> Callable[[str], str]:
    """Return a handler that verifies factual claims against source context."""
    fn = llm_fn or default_llm_fn

    def handler(prior_output: str) -> str:
        system = (
            "You are a scientific citation verifier. For EACH citation in the "
            "draft answer:\n"
            "1. Find the exact passage in the Source Context that supports it\n"
            "2. Quote that passage verbatim\n"
            "3. Rate: SUPPORTED / PARTIALLY SUPPORTED / NOT FOUND IN SOURCES\n"
            "4. If a claim is marked [unsupported], confirm it is general knowledge\n\n"
            "Also flag any claims that SHOULD have a citation but don't.\n\n"
            f"## Source Context\n{context}"
        )
        user = (
            f"## Draft Answer to Verify\n{prior_output}\n\n"
            "Verify every citation. Output a markdown table with columns: "
            "Claim | Citation | Source Quote | Verdict"
        )
        return fn(system, user)

    return handler


def make_critique_handler(
    llm_fn: Optional[LLMFunction] = None,
) -> Callable[[str], str]:
    """Return a handler that critiques completeness and quality."""
    fn = llm_fn or default_llm_fn

    def handler(prior_output: str) -> str:
        system = (
            "You are a senior research reviewer. "
            "Evaluate the answer for completeness, accuracy, clarity, "
            "and suggest improvements. Format as markdown."
        )
        user = (
            f"## Answer to Critique\n{prior_output}\n\n"
            "Provide a detailed critique with scores and suggestions."
        )
        return fn(system, user)

    return handler

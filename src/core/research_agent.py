"""
Smart research agent — extracts keywords via Gemini, discovers papers on ArXiv,
falls back to web search + GitHub when ArXiv has no results.
"""

from __future__ import annotations

import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.core.gemini import GeminiClient, GeminiRateLimitError
from src.papers.fetcher import PaperFetcher


class ResearchAgent:
    """
    Autonomous research agent that:
    0. Classifies whether a question needs papers or can be answered directly
    1. Uses Gemini to extract search keywords from a question
    2. Searches ArXiv for relevant papers
    3. Falls back to web / GitHub search if ArXiv has nothing
    4. Downloads & returns paper text for ingestion
    """

    # Questions matching these patterns NEVER need papers
    _GENERAL_PATTERNS = [
        r"^(what|who|when|where)\s+(is|are|was|were)\b",
        r"^(explain|define|describe|summarize)\b",
        r"^(tell me about|give me an overview)\b",
    ]

    def __init__(
        self,
        gemini: Optional[GeminiClient] = None,
        fetcher: Optional[PaperFetcher] = None,
    ):
        self.gemini = gemini or GeminiClient()
        self.fetcher = fetcher or PaperFetcher()

    # ── 0. Question triage ──────────────────────────────────────────────── #

    def classify_and_extract(self, question: str, history: str = "") -> dict:
        """
        Single Gemini call that both classifies the question AND extracts
        keywords.  This halves the initial latency vs two separate calls.

        Returns dict with:
            needs_papers   – bool
            complexity     – 'general' | 'conceptual' | 'research'
            reason         – short explanation
            keywords       – list[str]
            arxiv_query    – str for ArXiv API
            github_query   – str for GitHub search
            topic_summary  – 1-line summary
        """
        if not self.gemini.is_configured:
            q = question.strip().lower()
            for pat in self._GENERAL_PATTERNS:
                if re.match(pat, q):
                    words = re.findall(r"\b[a-zA-Z]{3,}\b", q)
                    stop = {"what", "how", "does", "the", "tell", "about", "with",
                            "from", "this", "that", "which", "where", "when", "why",
                            "can", "are", "was", "were", "been", "being", "have",
                            "has", "had", "for", "explain", "define", "describe"}
                    kw = [w for w in words if w not in stop][:5]
                    return {
                        "needs_papers": False, "complexity": "general",
                        "reason": "General knowledge question (heuristic)",
                        "keywords": kw, "arxiv_query": " AND ".join(kw[:3]),
                        "github_query": " ".join(kw[:3]), "topic_summary": question,
                    }
            words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
            stop = {"what", "how", "does", "the", "tell", "about", "with", "from",
                    "this", "that", "which", "where", "when", "why", "can", "are",
                    "was", "were", "been", "being", "have", "has", "had", "for"}
            kw = [w for w in words if w not in stop][:5]
            return {
                "needs_papers": True, "complexity": "research",
                "reason": "Defaulting to paper search",
                "keywords": kw, "arxiv_query": " AND ".join(kw[:3]),
                "github_query": " ".join(kw[:3]), "topic_summary": question,
            }

        system = (
            "You classify research questions AND extract search keywords in one step.\n\n"
            "Categories:\n"
            "  general    — foundational concept any LLM knows (e.g. 'What is CNN?')\n"
            "  conceptual — needs depth but not specific papers (e.g. 'Compare RNN vs Transformer')\n"
            "  research   — needs actual papers (e.g. 'Latest approaches to protein folding?')\n\n"
            "Return ONLY valid JSON with ALL these keys:\n"
            '  "needs_papers": true/false,\n'
            '  "complexity": "general|conceptual|research",\n'
            '  "reason": "one-line explanation",\n'
            '  "keywords": ["term1", "term2", ...],\n'
            '  "arxiv_query": "simple keyword query for ArXiv (just terms with AND/OR, '
            'NO field prefixes like all: or ti:)",\n'
            '  "github_query": "query for GitHub repo search",\n'
            '  "topic_summary": "one-line summary"\n'
            "No markdown, no explanation, just JSON."
        )
        user = question
        if history:
            user = f"History:\n{history}\n\nQuestion: {question}"

        try:
            raw = self.gemini.chat(system, user, temperature=0.0)
        except GeminiRateLimitError:
            # Gemini rate-limited — fall back to heuristic keyword extraction
            words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
            stop = {"what", "how", "does", "the", "tell", "about", "with", "from",
                    "this", "that", "which", "where", "when", "why", "can", "are",
                    "was", "were", "been", "being", "have", "has", "had", "for"}
            kw = [w for w in words if w not in stop][:5]
            return {
                "needs_papers": True, "complexity": "research",
                "reason": "Gemini rate-limited — using heuristic fallback",
                "keywords": kw, "arxiv_query": " AND ".join(kw[:3]),
                "github_query": " ".join(kw[:3]), "topic_summary": question,
            }
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        defaults = {
            "needs_papers": True, "complexity": "research", "reason": "",
            "keywords": re.findall(r"\b[a-zA-Z]{3,}\b", question)[:5],
            "arxiv_query": question, "github_query": question,
            "topic_summary": question,
        }
        try:
            parsed = json.loads(raw)
            for k in defaults:
                if k not in parsed:
                    parsed[k] = defaults[k]
            parsed["needs_papers"] = bool(parsed["needs_papers"])
            return parsed
        except json.JSONDecodeError:
            return defaults

    # Keep backward-compat aliases
    def classify_question(self, question: str, history: str = "") -> dict:
        r = self.classify_and_extract(question, history)
        return {"needs_papers": r["needs_papers"], "reason": r["reason"], "complexity": r["complexity"]}

    def extract_keywords(self, question: str, history: str = "") -> dict:
        r = self.classify_and_extract(question, history)
        return {"keywords": r["keywords"], "arxiv_query": r["arxiv_query"],
                "github_query": r["github_query"], "topic_summary": r["topic_summary"]}

    # ── 2. ArXiv discovery ──────────────────────────────────────────────── #

    def search_arxiv(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search ArXiv and return paper summaries.
        Returns list of {arxiv_id, title, summary, authors}.
        """
        papers = self.fetcher.search(query, max_results=max_results)
        return [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "summary": p.summary[:300],
                "authors": p.authors[:3],
            }
            for p in papers
        ]

    def fetch_arxiv_papers(self, arxiv_ids: list[str]) -> dict[str, str]:
        """Download and extract text from ArXiv papers in parallel."""
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(arxiv_ids), 4)) as pool:
            futures = {
                pool.submit(self.fetcher.fetch_by_id, aid): aid
                for aid in arxiv_ids
            }
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    results[aid] = future.result()
                except Exception:
                    continue
        return results

    # ── 3. Web / GitHub fallback ────────────────────────────────────────── #

    def web_search(self, query: str) -> list[dict]:
        """
        Use Gemini to generate a knowledge summary when no ArXiv papers found.
        Also searches for relevant GitHub repos.
        Returns list of {source, title, content}.
        """
        results: list[dict] = []

        # Ask Gemini to provide knowledge + GitHub repos
        if self.gemini.is_configured:
            system = (
                "You are a knowledgeable research assistant. "
                "The user is asking about a topic with no ArXiv papers found. "
                "Provide:\n"
                "1. A comprehensive technical overview of the topic (formatted as markdown)\n"
                "2. Any relevant GitHub repositories (name, URL, description)\n"
                "3. Key references or documentation links\n\n"
                "Format your response as markdown with clear sections."
            )
            user = f"Provide detailed technical information about: {query}"
            content = self.gemini.chat(system, user, temperature=0.3)
            results.append({
                "source": "gemini_knowledge",
                "title": f"Gemini knowledge: {query}",
                "content": content,
            })

        # Try GitHub API search (no auth required for basic search)
        github_results = self._search_github(query)
        for repo in github_results:
            results.append({
                "source": f"github:{repo['full_name']}",
                "title": repo.get("name", ""),
                "content": (
                    f"# {repo.get('full_name', '')}\n\n"
                    f"**Description:** {repo.get('description', 'N/A')}\n"
                    f"**URL:** {repo.get('html_url', '')}\n"
                    f"**Stars:** {repo.get('stargazers_count', 0)}\n"
                    f"**Language:** {repo.get('language', 'N/A')}\n"
                    f"**Topics:** {', '.join(repo.get('topics', []))}\n"
                ),
            })

        return results

    @staticmethod
    def _search_github(query: str, max_results: int = 5) -> list[dict]:
        """Search GitHub repos via their public API (no auth needed)."""
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": max_results},
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("items", [])
        except requests.RequestException:
            pass
        return []

    # ── 4. Full discovery pipeline ──────────────────────────────────────── #

    def discover(
        self,
        question: str,
        history: str = "",
        max_arxiv: int = 3,
    ) -> dict:
        """
        Full discovery: triage → extract keywords → search ArXiv → fallback to web.

        Returns:
            triage         – classification result (needs_papers, complexity, reason)
            keywords       – extracted keyword info
            arxiv_papers   – list of ArXiv paper dicts found
            arxiv_texts    – {id: full_text} for fetched papers
            web_results    – list of web/github results (only if ArXiv empty)
            source_type    – 'direct' | 'arxiv' | 'web' | 'none'
        """
        # Step 0: Classify + extract keywords in ONE Gemini call
        combined = self.classify_and_extract(question, history=history)

        triage = {
            "needs_papers": combined["needs_papers"],
            "reason": combined["reason"],
            "complexity": combined["complexity"],
        }
        kw = {
            "keywords": combined["keywords"],
            "arxiv_query": combined["arxiv_query"],
            "github_query": combined["github_query"],
            "topic_summary": combined["topic_summary"],
        }

        if not triage["needs_papers"]:
            return {
                "triage": triage,
                "keywords": {"keywords": [], "arxiv_query": "", "github_query": "", "topic_summary": question},
                "arxiv_papers": [],
                "arxiv_texts": {},
                "web_results": [],
                "source_type": "direct",
            }

        # Step 1: Search ArXiv (keywords already extracted above)
        arxiv_papers = self.search_arxiv(kw["arxiv_query"], max_results=max_arxiv)

        if arxiv_papers:
            # Fetch the actual papers
            ids = [p["arxiv_id"] for p in arxiv_papers]
            texts = self.fetch_arxiv_papers(ids)
            return {
                "triage": triage,
                "keywords": kw,
                "arxiv_papers": arxiv_papers,
                "arxiv_texts": texts,
                "web_results": [],
                "source_type": "arxiv",
            }

        # Step 3: Fallback to web + GitHub
        web_results = self.web_search(kw.get("github_query", question))
        return {
            "triage": triage,
            "keywords": kw,
            "arxiv_papers": [],
            "arxiv_texts": {},
            "web_results": web_results,
            "source_type": "web" if web_results else "none",
        }

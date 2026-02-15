"""
Session manager — persists conversation history across interactions.

Each session stores:
- session ID & timestamps
- conversation turns (question + answer + metadata)
- ingested paper IDs for that session
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.core.config import PROJECT_ROOT

SESSIONS_DIR = PROJECT_ROOT / "sessions"


class Session:
    """A single conversation session with history."""

    def __init__(self, session_id: Optional[str] = None, path: Optional[Path] = None):
        self.session_id = session_id or uuid.uuid4().hex[:10]
        self.path = path or SESSIONS_DIR / f"{self.session_id}.json"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at
        self.turns: list[dict] = []
        self.ingested_papers: list[str] = []  # ArXiv IDs or web sources

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing session if it exists
        if self.path.exists():
            self._load()

    # ── conversation history ────────────────────────────────────────────── #

    def add_turn(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a Q&A turn in this session."""
        self.turns.append({
            "turn": len(self.turns) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "metadata": metadata or {},
        })
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def add_paper(self, paper_id: str) -> None:
        """Track that a paper was ingested in this session."""
        if paper_id not in self.ingested_papers:
            self.ingested_papers.append(paper_id)
            self._save()

    def get_history_context(self, max_turns: int = 5) -> str:
        """
        Build a markdown string of recent conversation history
        for inclusion in the LLM prompt.
        """
        recent = self.turns[-max_turns:] if self.turns else []
        if not recent:
            return ""

        lines = ["## Previous Conversation\n"]
        for t in recent:
            lines.append(f"**Q{t['turn']}:** {t['question']}\n")
            # Truncate long answers
            answer = t["answer"]
            if len(answer) > 500:
                answer = answer[:500] + "…"
            lines.append(f"**A{t['turn']}:** {answer}\n")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    # ── persistence ─────────────────────────────────────────────────────── #

    def _save(self) -> None:
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ingested_papers": self.ingested_papers,
            "turns": self.turns,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.session_id = data.get("session_id", self.session_id)
        self.created_at = data.get("created_at", self.created_at)
        self.updated_at = data.get("updated_at", self.updated_at)
        self.turns = data.get("turns", [])
        self.ingested_papers = data.get("ingested_papers", [])

    def describe(self) -> dict:
        """Summary for listing."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": self.turn_count,
            "papers": len(self.ingested_papers),
        }


def list_sessions() -> list[dict]:
    """List all saved sessions."""
    if not SESSIONS_DIR.exists():
        return []
    results = []
    for f in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            s = Session(path=f)
            results.append(s.describe())
        except Exception:
            continue
    return results


def get_latest_session() -> Optional[Session]:
    """Load the most recently updated session, or None."""
    if not SESSIONS_DIR.exists():
        return None
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        return Session(path=files[0])
    return None

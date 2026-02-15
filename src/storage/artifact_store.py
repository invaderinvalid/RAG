"""
Markdown artifact storage with ScaleDown compression + local fallback.

Stores COT traces, self-verification, and self-critique logs as compressed
markdown files with companion metadata.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.core.config import ARTIFACTS_DIR
from src.core.scaledown import ScaleDownClient


class ArtifactStore:
    """Persist reasoning artifacts (COT / verify / critique) as markdown."""

    VALID_TYPES = {"cot", "self_verify", "self_critique"}

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        scaledown: Optional[ScaleDownClient] = None,
    ):
        self.base_dir = base_dir or ARTIFACTS_DIR
        self.scaledown = scaledown or ScaleDownClient()

    # ── public API ──────────────────────────────────────────────────────── #

    def store(
        self,
        content: str,
        artifact_type: str = "cot",
        prompt_context: str = "",
    ) -> dict:
        """
        Compress (if possible) and persist a markdown artifact.

        Returns metadata dict including storage path, compression stats, etc.
        """
        if artifact_type not in self.VALID_TYPES:
            raise ValueError(
                f"artifact_type must be one of {self.VALID_TYPES}, got '{artifact_type}'"
            )

        artifact_id = self._make_id(artifact_type, content)
        now = datetime.now(timezone.utc).isoformat()

        meta = {
            "id": artifact_id,
            "type": artifact_type,
            "timestamp": now,
            "storage": None,
            "compressed": False,
            "original_tokens": 0,
            "compressed_tokens": 0,
        }

        content_to_save = content

        # ── Attempt compression via ScaleDown ──────────────────────────── #
        if self.scaledown.is_configured:
            result = self.scaledown.compress(
                context=content,
                prompt=prompt_context or f"Compress {artifact_type} reasoning trace",
            )
            if result["successful"]:
                content_to_save = result["compressed_text"]
                meta["compressed"] = True
                meta["original_tokens"] = result["original_tokens"]
                meta["compressed_tokens"] = result["compressed_tokens"]

        # ── Write to disk ──────────────────────────────────────────────── #
        store_dir = self.base_dir / artifact_type
        store_dir.mkdir(parents=True, exist_ok=True)

        md_path = store_dir / f"{artifact_id}.md"
        md_path.write_text(content_to_save, encoding="utf-8")

        meta["storage"] = str(md_path)

        meta_path = store_dir / f"{artifact_id}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return meta

    def load(self, artifact_id: str, artifact_type: str = "cot") -> Optional[str]:
        """Load a stored artifact by ID."""
        md_path = self.base_dir / artifact_type / f"{artifact_id}.md"
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")
        return None

    def load_meta(self, artifact_id: str, artifact_type: str = "cot") -> Optional[dict]:
        """Load companion metadata for an artifact."""
        meta_path = self.base_dir / artifact_type / f"{artifact_id}.meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    def list_artifacts(self, artifact_type: Optional[str] = None) -> list[dict]:
        """List all stored artifacts, optionally filtered by type."""
        results = []
        types = [artifact_type] if artifact_type else self.VALID_TYPES
        for atype in types:
            adir = self.base_dir / atype
            if not adir.exists():
                continue
            for meta_file in sorted(adir.glob("*.meta.json")):
                results.append(json.loads(meta_file.read_text(encoding="utf-8")))
        return results

    # ── helpers ──────────────────────────────────────────────────────────── #

    @staticmethod
    def _make_id(artifact_type: str, content: str) -> str:
        raw = f"{artifact_type}:{datetime.now(timezone.utc).isoformat()}:{content[:128]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

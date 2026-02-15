"""
Configurable reasoning workflow engine.

Supports chain-of-thought, self-verification, and self-critique stages.
Stages can be added, removed, reordered, and toggled at runtime.
The full pipeline configuration can be serialised/loaded from JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.core.scaledown import ScaleDownClient
from src.storage.artifact_store import ArtifactStore


@dataclass
class WorkflowStage:
    """A single stage in the reasoning pipeline."""
    name: str                          # e.g. "cot", "self_verify", "self_critique"
    handler: Callable[[str], str]      # receives prior output → returns markdown
    enabled: bool = True
    description: str = ""


@dataclass
class ReasoningWorkflow:
    """
    Ordered pipeline of reasoning stages.

    Each stage receives the previous stage's output, produces a markdown
    artifact, and passes it downstream.

    When a ScaleDownClient is provided, the output of each stage is compressed
    before being passed to the next stage — this saves tokens on subsequent
    LLM calls (verify sees compressed COT, critique sees compressed verify).
    """

    stages: list[WorkflowStage] = field(default_factory=list)
    store: ArtifactStore = field(default_factory=ArtifactStore)
    compressor: Optional[ScaleDownClient] = field(default=None, repr=False)

    # ── stage management ────────────────────────────────────────────────── #

    def add_stage(
        self,
        stage: WorkflowStage,
        position: Optional[int] = None,
    ) -> None:
        """Insert a stage at *position* (default: end)."""
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)

    def remove_stage(self, name: str) -> None:
        """Remove a stage by name."""
        self.stages = [s for s in self.stages if s.name != name]

    def toggle_stage(self, name: str, enabled: bool) -> None:
        """Enable or disable a stage without removing it."""
        for s in self.stages:
            if s.name == name:
                s.enabled = enabled

    def reorder(self, name_order: list[str]) -> None:
        """Reorder stages to match *name_order*. Un-listed stages are dropped."""
        index = {s.name: s for s in self.stages}
        self.stages = [index[n] for n in name_order if n in index]

    def get_stage(self, name: str) -> Optional[WorkflowStage]:
        """Retrieve a stage by name."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    # ── execution ───────────────────────────────────────────────────────── #

    def run(self, initial_input: str, prompt_context: str = "") -> dict[str, dict]:
        """
        Execute all enabled stages sequentially.

        Returns a dict mapping stage name → artifact metadata (includes timing).
        """
        results: dict[str, dict] = {}
        current = initial_input

        for stage in self.stages:
            if not stage.enabled:
                continue

            t0 = time.perf_counter()
            output = stage.handler(current)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            meta = self.store.store(
                content=output,
                artifact_type=stage.name,
                prompt_context=prompt_context,
            )
            meta["stage_latency_ms"] = elapsed_ms
            results[stage.name] = meta
            current = output  # chain into next stage

        return results

    # ── serialisation ───────────────────────────────────────────────────── #

    def describe(self) -> list[dict]:
        """Return a JSON-friendly description of the pipeline."""
        return [
            {
                "name": s.name,
                "enabled": s.enabled,
                "description": s.description,
            }
            for s in self.stages
        ]

    def save_config(self, path: str | Path) -> None:
        """Persist current pipeline layout to a JSON file."""
        Path(path).write_text(json.dumps(self.describe(), indent=2), encoding="utf-8")

    def load_config(self, path: str | Path) -> None:
        """
        Restore pipeline layout from a JSON file.

        Only affects ordering & enabled flags; handlers stay as-is.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name_to_enabled = {d["name"]: d["enabled"] for d in data}
        order = [d["name"] for d in data]

        # Toggle enabled flags
        for stage in self.stages:
            if stage.name in name_to_enabled:
                stage.enabled = name_to_enabled[stage.name]

        # Reorder
        self.reorder(order)

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from .base import Evolver


@dataclass
class WorkingToEpisodicConfig(MemoryBaseModel):
    """Typed config for WorkingToEpisodic evolver."""
    threshold: int = 40


@dataclass
class WorkingToEpisodicRequest(StageRequest):
    """Typed request DTO for WorkingToEpisodic evolver execution."""
    threshold: int = 40
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class WorkingToEpisodicEvolver(Evolver):
    """
    Evolves (summarizes) working memory buffer to episodic memory when overflowed (N turns).
    """

    def evolve(self, memory, logger=None):
        # Example logic: summarize working memory if buffer exceeds threshold
        # Support both legacy dict config and typed config
        threshold = 40
        cfg = getattr(self, "config") or {}
        # Require typed config (fail-fast). No legacy dict support.
        if hasattr(cfg, "threshold"):
            threshold = int(getattr(cfg, "threshold", 40))
        else:
            raise TypeError(
                "WorkingToEpisodicEvolver requires a typed config with a 'threshold' attribute. "
                "Please provide WorkingToEpisodicConfig or a compatible typed config."
            )
        working_items = memory.working.get_buffer()
        if len(working_items) >= threshold:
            summary = memory.working.summarize_buffer()
            memory.episodic.add(summary)
            memory.working.clear_buffer()
            if logger:
                logger.info(f"Promoted {len(working_items)} working items to episodic as summary.")

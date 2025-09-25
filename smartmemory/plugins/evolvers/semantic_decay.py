from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from .base import Evolver


@dataclass
class SemanticDecayConfig(MemoryBaseModel):
    threshold: float = 0.2


@dataclass
class SemanticDecayRequest(StageRequest):
    threshold: float = 0.2
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class SemanticDecayEvolver(Evolver):
    """
    Prunes/archives semantic facts based on low relevance, age, or feedback.
    """

    def evolve(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "threshold"):
            raise TypeError(
                "SemanticDecayEvolver requires a typed config with 'threshold'. "
                "Provide SemanticDecayConfig or a compatible typed config."
            )
        threshold = float(getattr(cfg, "threshold"))
        old_facts = memory.semantic.get_low_relevance(threshold=threshold)
        for fact in old_facts:
            memory.semantic.archive(fact)
            if logger:
                logger.info(f"Archived low-relevance semantic fact: {fact}")

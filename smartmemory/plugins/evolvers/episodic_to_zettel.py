from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest
from .base import Evolver


@dataclass
class EpisodicToZettelConfig(MemoryBaseModel):
    period: int = 1  # days


@dataclass
class EpisodicToZettelRequest(StageRequest):
    period: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class EpisodicToZettelEvolver(Evolver):
    """
    Rolls up episodic events into zettels (notes) on a periodic basis.
    """

    def evolve(self, memory, logger=None):
        cfg = getattr(self, "config")
        if not hasattr(cfg, "period"):
            raise TypeError(
                "EpisodicToZettelEvolver requires a typed config with 'period'. "
                "Provide EpisodicToZettelConfig or a compatible typed config."
            )
        period = int(getattr(cfg, "period"))
        events = memory.episodic.get_events_since(days=period)
        for event in events:
            zettel = memory.zettel.create_note_from_event(event)
            memory.zettel.add(zettel)
            if logger:
                logger.info(f"Rolled up episodic event into zettel: {event}")

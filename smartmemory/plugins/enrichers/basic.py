import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smartmemory.models.base import MemoryBaseModel, StageRequest


@dataclass
class BasicEnricherConfig(MemoryBaseModel):
    enable_entity_tags: bool = True
    enable_summary: bool = True


@dataclass
class BasicEnricherRequest(StageRequest):
    enable_entity_tags: bool = True
    enable_summary: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class BasicEnricher:
    """
    Basic enrichment: tag with entities and generate summary.
    - Tags: Use entities from node_ids if present.
    - Summary: First sentence of item.content.
    Returns enrichment result dict.
    """

    def __init__(self, config: Optional[BasicEnricherConfig] = None):
        self.config = config or BasicEnricherConfig()

    def enrich(self, item, node_ids=None) -> Dict[str, Any]:
        # Fail fast on typed config
        if not isinstance(self.config, BasicEnricherConfig):
            raise TypeError("BasicEnricher requires a typed config (BasicEnricherConfig)")
        content = getattr(item, 'content', str(item))
        result: Dict[str, Any] = {'new_items': []}

        # Summary
        if self.config.enable_summary:
            result['summary'] = content.split('.')[0] + '.' if '.' in content else content[:100]

        # Entity tags and temporal annotations if node_ids provided
        if isinstance(node_ids, dict) and self.config.enable_entity_tags:
            entities = node_ids.get('semantic_entities') or []
            result['tags'] = list(entities)
            from datetime import datetime
            temporal = {}
            now = datetime.now()
            for entity in entities:
                temporal[entity] = {
                    'valid_start': now,
                    'valid_end': None,
                    'transaction_time': now
                }
            if temporal:
                result['temporal'] = temporal

        return result

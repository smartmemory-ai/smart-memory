"""
SmartGraph Node Operations Module

Handles all node-related operations for the SmartGraph system.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from smartmemory.graph.models.schema_validator import get_validator
from smartmemory.models.compat.dataclass_model import get_field_names
from smartmemory.observability.instrumentation import emit_ctx
from smartmemory.utils import unflatten_dict, flatten_dict

logger = logging.getLogger(__name__)


class SmartGraphNodes:
    """Handles all node-related operations for SmartGraph."""

    def __init__(self, backend, item_cls, enable_caching=True, cache_size=1000):
        self.backend = backend
        self.item_cls = item_cls
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._node_cache = {} if enable_caching else None
        self._cache_hits = 0
        self._cache_misses = 0

    def add_node(self,
                 item_id: Optional[str],
                 properties: Dict[str, Any],
                 valid_time: Optional[Tuple] = None,
                 transaction_time: Optional[Tuple] = None,
                 memory_type: Optional[str] = None):
        """Add a node to the graph."""
        if item_id is None:
            item_id = str(uuid.uuid4())

        # Use mapping logic to ensure properties are flat and complete
        node_dict = dict(properties)
        node_dict['item_id'] = item_id

        # Schema validation
        validator = get_validator()
        if not validator.validate_node(node_dict, memory_type):
            # Validation failed but we're in warning mode, continue
            pass

        result = self.backend.add_node(item_id, node_dict, valid_time, transaction_time, memory_type)

        # Invalidate cache for this node
        if self.enable_caching:
            cache_key = f"{item_id}:current"
            if cache_key in self._node_cache:
                del self._node_cache[cache_key]

        # Emit graph stats update
        try:
            self._emit_graph_stats(
                "add_node",
                details={"item_id": item_id, "memory_type": memory_type or ""},
                delta_nodes=1,
                delta_edges=0,
            )
        except Exception:
            pass

        if isinstance(result, dict) and "item_id" in result:
            return result
        return {"item_id": item_id, "properties": node_dict}

    def add_dual_node(self, item_id: str, memory_properties: Dict[str, Any], memory_type: str, entity_nodes: List[Dict[str, Any]] = None, is_global: bool = False):
        """Add a dual-node structure through the backend.
        
        Returns a dict containing at least:
          - memory_node_id: str
          - entity_node_ids: List[str]
        """
        # Check if backend supports dual-node architecture
        if hasattr(self.backend, 'add_dual_node'):
            result = self.backend.add_dual_node(
                item_id=item_id,
                memory_properties=memory_properties,
                memory_type=memory_type,
                entity_nodes=entity_nodes,
                is_global=is_global
            )

            # Invalidate cache for affected nodes
            if self.enable_caching:
                # Clear memory node cache
                memory_cache_key = f"{item_id}:current"
                if memory_cache_key in self._node_cache:
                    del self._node_cache[memory_cache_key]

                # Clear entity node caches
                if result and 'entity_node_ids' in result:
                    for entity_id in result['entity_node_ids']:
                        entity_cache_key = f"{entity_id}:current"
                        if entity_cache_key in self._node_cache:
                            del self._node_cache[entity_cache_key]

            # Emit graph stats update for dual-node creation
            try:
                # Compute deltas based on provided entity_nodes and relationships
                entity_count = 0
                if isinstance(result, dict):
                    entity_count = int(result.get('entity_count') or 0)
                if not entity_count and entity_nodes is not None:
                    try:
                        entity_count = len(entity_nodes)
                    except Exception:
                        entity_count = 0
                sem_rel_count = 0
                try:
                    if entity_nodes:
                        for en in entity_nodes:
                            rels = en.get('relationships', []) if isinstance(en, dict) else []
                            try:
                                sem_rel_count += len(rels)
                            except Exception:
                                pass
                except Exception:
                    sem_rel_count = 0
                total_new_nodes = 1 + (entity_count or 0)
                total_new_edges = (entity_count or 0) + (sem_rel_count or 0)
                self._emit_graph_stats(
                    "add_dual_node",
                    details={
                        "item_id": item_id,
                        "memory_type": memory_type,
                        "entity_count": entity_count,
                        "semantic_relationships": sem_rel_count,
                    },
                    delta_nodes=total_new_nodes,
                    delta_edges=total_new_edges,
                )
            except Exception:
                pass

            return result
        else:
            # Fallback to legacy single-node creation with normalized return shape
            node_result = self.add_node(item_id, memory_properties, memory_type=memory_type)
            mem_id = item_id
            if isinstance(node_result, dict):
                mem_id = node_result.get('item_id', item_id)
            return {"memory_node_id": mem_id, "entity_node_ids": []}

    def get_node(self, item_id: str, as_of_time: Optional[str] = None):
        """Get a node by ID."""
        # Check cache first (only for current time queries)
        cache_key = f"{item_id}:{as_of_time or 'current'}"
        if self.enable_caching and as_of_time is None and cache_key in self._node_cache:
            self._cache_hits += 1
            return self._node_cache[cache_key]

        # Cache miss - fetch from backend
        self._cache_misses += 1
        node = self.backend.get_node(item_id, as_of_time)
        if node:
            result = self._from_node_dict(self.item_cls, node)

            # Cache the result (only for current time queries)
            if self.enable_caching and as_of_time is None:
                self._manage_cache_size()
                self._node_cache[cache_key] = result

            return result
        return None

    def get_neighbors(self, item_id: str, edge_type: Optional[str] = None, as_of_time: Optional[str] = None):
        """Get neighbors of a node."""
        neighbors = self.backend.get_neighbors(item_id, edge_type, as_of_time)
        return [self._from_node_dict(self.item_cls, n) for n in neighbors]

    def remove_node(self, item_id: str):
        """Remove a node from the graph."""
        # Estimate number of incident edges for delta, best-effort
        incident_edges = None
        try:
            if hasattr(self.backend, 'get_edges_for_node'):
                edges = self.backend.get_edges_for_node(item_id)  # type: ignore[attr-defined]
                try:
                    incident_edges = len(edges) if edges is not None else 0
                except Exception:
                    incident_edges = None
        except Exception:
            incident_edges = None

        result = self.backend.remove_node(item_id)
        try:
            self._emit_graph_stats(
                "remove_node",
                details={"item_id": item_id, "incident_edges": incident_edges},
                delta_nodes=-1,
                delta_edges=(-incident_edges if isinstance(incident_edges, int) else None),
            )
        except Exception:
            pass
        return result

    def nodes(self):
        """Return all node IDs for compatibility with memory store iteration."""
        try:
            if hasattr(self.backend, 'get_all_node_ids'):
                return self.backend.get_all_node_ids()
            elif hasattr(self.backend, 'get_all_nodes'):
                nodes = self.backend.get_all_nodes()
                return [node.get('item_id', node.get('id', '')) for node in nodes]
            else:
                # Fallback: try to search for all nodes and extract IDs
                results = self.backend.search_nodes({})
                return [node.get('item_id', node.get('id', '')) for node in results]
        except Exception:
            return []

    @staticmethod
    def _to_node_dict(obj):
        """Convert object to node dictionary."""
        # Supports dataclasses/mixins, Pydantic models, and dicts
        if hasattr(obj, "to_dict"):
            node = obj.to_dict()
            if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
                node.update(obj.metadata)
            node = flatten_dict(node)
            return node
        if hasattr(obj, "dict"):
            node = obj.to_dict(exclude_unset=True)
            if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
                node.update(obj.metadata)
            node = flatten_dict(node)
            return node
        elif isinstance(obj, dict):
            return flatten_dict(obj)
        else:
            raise TypeError("Unsupported type for _to_node_dict")

    @staticmethod
    def _from_node_dict(item_cls, node):
        """Convert node dictionary to object."""
        # Handles both flat and nested dicts (Neo4j sometimes nests under 'properties')
        props = node.get("properties") or {} if "properties" in node else node

        # Determine known field names across dataclasses and Pydantic
        known_fields = get_field_names(item_cls)

        # Separate known fields from metadata
        known = {}
        metadata = {}

        for k, v in props.items():
            if k in known_fields and k != 'metadata':  # Don't include metadata in known fields
                known[k] = v
            else:
                metadata[k] = v

        # Ensure required fields have defaults
        if 'content' not in known:
            known['content'] = props.get('name', props.get('description', ''))

        # Handle transaction_time - ensure it's never None
        if 'transaction_time' not in known or known.get('transaction_time') is None:
            # Use current time if not provided or None
            from datetime import datetime
            known['transaction_time'] = datetime.now()

        # Handle other datetime fields that might be None
        datetime_fields = ['valid_start_time', 'valid_end_time']
        for field in datetime_fields:
            if field in known and known[field] is None:
                # Remove None datetime fields rather than setting defaults
                del known[field]

        # Unflatten metadata if needed
        metadata = unflatten_dict(metadata)

        # Construct the item with known fields and metadata
        known['metadata'] = metadata
        return item_cls(**known)

    def _emit_graph_stats(self, operation: str, details: Dict[str, Any], delta_nodes: Optional[int] = None, delta_edges: Optional[int] = None) -> None:
        """Best-effort emission of graph stats update events."""
        try:
            backend_name = type(self.backend).__name__
            node_count: Optional[int] = None
            edge_count: Optional[int] = None

            # Prefer explicit fast counters if backend provides them
            if hasattr(self.backend, 'get_counts'):
                try:
                    counts = self.backend.get_counts()  # type: ignore[attr-defined]
                    if isinstance(counts, dict):
                        node_count = counts.get('node_count')
                        edge_count = counts.get('edge_count')
                except Exception:
                    pass
            else:
                # Fallbacks
                try:
                    if hasattr(self.backend, 'get_node_count'):
                        node_count = self.backend.get_node_count()  # type: ignore[attr-defined]
                    elif hasattr(self.backend, 'get_all_nodes'):
                        nodes = self.backend.get_all_nodes()  # type: ignore[attr-defined]
                        try:
                            node_count = len(nodes) if nodes is not None else None
                        except Exception:
                            node_count = None
                except Exception:
                    node_count = None
                try:
                    if hasattr(self.backend, 'get_edge_count'):
                        edge_count = self.backend.get_edge_count()  # type: ignore[attr-defined]
                except Exception:
                    edge_count = None

            data: Dict[str, Any] = {
                "backend": backend_name,
                "node_count": node_count,
                "edge_count": edge_count,
                "delta_nodes": delta_nodes,
                "delta_edges": delta_edges,
                "details": details or {},
            }
            emit_ctx("graph_stats_update", component="graph", operation=operation, data=data)
        except Exception:
            # Observability must never break graph operations
            pass

    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries when cache is full."""
        if not self.enable_caching:
            return

        # Remove oldest entries if cache is too large
        if len(self._node_cache) >= self.cache_size:
            # Remove 20% of oldest entries
            remove_count = max(1, self.cache_size // 5)
            oldest_keys = list(self._node_cache.keys())[:remove_count]
            for key in oldest_keys:
                del self._node_cache[key]

    def clear_cache(self):
        """Clear node cache."""
        if self.enable_caching:
            self._node_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def get_cache_stats(self):
        """Get node cache performance statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0.0

        return {
            "caching_enabled": True,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "node_cache_size": len(self._node_cache),
            "max_cache_size": self.cache_size
        }

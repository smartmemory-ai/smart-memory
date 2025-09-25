import logging
from typing import Optional, Any, Dict, List

from smartmemory.configuration import MemoryConfig
from smartmemory.graph.types.zettel import ZettelMemoryGraph
from smartmemory.memory.base import GraphBackedMemory
from smartmemory.memory.mixins import CRUDMixin, ArchivingMixin, ValidationMixin, ConfigurableMixin
from smartmemory.memory.types.zettel_extensions import (
    ZettelBacklinkSystem, ZettelEmergentStructure, ZettelDiscoveryEngine,
    KnowledgeCluster, DiscoveryPath
)
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


# ---
# Zettel (Note) Memory Graph Schema
#
# Node Types:
#   - Note: title, content, created_at, tags, author, references, updated_at, summary
#   - Tag (optional): name
#   - Concept/Entity (optional): name, type
# Edge Types (extensible):
#   - LINKED_TO: (Note) <-> (Note) (bidirectional)
#   - TAGGED_WITH: (Note) -> (Tag)
#   - MENTIONS/REFERS_TO: (Note) -> (Concept/Entity/Note)
#   - ANY_DYNAMIC_RELATION: any new relation type as needed (e.g., INSPIRED_BY, CONTRADICTS, ELABORATES_ON)
#
# The agent/LLM can introduce new relation types at any time.
# ---

class Link:
    def __init__(self, src_id: str, tgt_id: str, link_type: str, score: float = 1.0, metadata: Optional[dict] = None):
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.link_type = link_type
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            'src_id': self.src_id,
            'tgt_id': self.tgt_id,
            'link_type': self.link_type,
            'score': self.score,
            'metadata': self.metadata
        }

    @staticmethod
    def create_graph_link(graph, src_id, tgt_id, link_type: str, score: float = 1.0, metadata: Optional[dict] = None):
        props = {"score": score, "metadata": metadata or {}, "type": link_type}
        return graph.add_edge(src_id, tgt_id, link_type.upper(), **props)


def create_links_graph(zettel_graph, strategy: str = 'semantic', threshold: float = 0.5):
    """
    Advanced linking logic for graph-backed Zettel. Supports semantic, keyword, and manual linking.
    Creates edges in the graph DB, not in memory.
    """
    notes = zettel_graph.get_all()
    for i, n1 in enumerate(notes):
        for j, n2 in enumerate(notes):
            if i == j:
                continue
            if strategy == 'semantic':
                emb1 = getattr(n1, 'embedding', None)
                emb2 = getattr(n2, 'embedding', None)
                if emb1 is not None and emb2 is not None:
                    score = MemoryItem.cosine_similarity(emb1, emb2)
                    if score >= threshold:
                        Link.create_graph_link(zettel_graph.store, n1.uuid, n2.uuid, "RELATED", score)
            elif strategy == 'keyword':
                words1 = set(str(n1.content).lower().split())
                words2 = set(str(n2.content).lower().split())
                shared = words1.intersection(words2)
                score = len(shared) / max(1, len(words1.union(words2)))
                if score >= threshold:
                    Link.create_graph_link(zettel_graph.store, n1.uuid, n2.uuid, "RELATED", score)


default_store_class = ZettelMemoryGraph


class ZettelMemory(GraphBackedMemory, CRUDMixin, ArchivingMixin, ValidationMixin, ConfigurableMixin):
    """
    Complete Zettelkasten memory system with bidirectional linking, emergent structure detection,
    and serendipitous discovery capabilities.
    """

    _memory_type = "zettel"

    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.graph = ZettelMemoryGraph()

        # Initialize Zettelkasten extensions
        self.backlinks = ZettelBacklinkSystem(self)
        self.structure = ZettelEmergentStructure(self)
        self.discovery = ZettelDiscoveryEngine(self)

    def _add_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Zettel-specific add logic."""
        zettel_data = {
            "item_id": item.item_id,
            "title": item.metadata.get("title") or item.metadata.get("description") or item.item_id,
            "content": item.content,
            "zettel_body": item.content,
            "description": item.metadata.get("description", "Zettel note"),
            "tags": item.metadata.get("tags", []),
            "group_id": getattr(item, "group_id", ""),
        }

        try:
            # add returns the zettel node directly
            zettel_node = self.graph.add(zettel_data, **kwargs)

            # Update item with zettel data
            if zettel_node:
                # Ensure item_id is preserved
                if not item.item_id:
                    item.item_id = zettel_node.get("item_id")
                item.group_id = zettel_node.get("group_id", "")
                item.metadata["group_id"] = item.group_id
                item.metadata["tags"] = zettel_node.get("tags", [])
                item.metadata["_node"] = zettel_node

            return item
        except Exception as e:
            logger.error(f"Failed to add zettel item: {e}")
            return None

    def _get_impl(self, key: str) -> Optional[MemoryItem]:
        """Zettel-specific get logic."""
        try:
            # Delegate directly to SmartGraph to retrieve a fully-formed MemoryItem
            # ZettelMemoryGraph wraps SmartGraph as `.graph`
            if hasattr(self.graph, 'graph') and hasattr(self.graph.graph, 'get_node'):
                item = self.graph.graph.get_node(key)
                return item

            # Fallbacks for unexpected environments
            if hasattr(self.graph, 'get_node'):
                node_dict = self.graph.get_node(key)
                if node_dict is None:
                    return None
                return MemoryItem(
                    item_id=key,
                    content=node_dict.get("content", ""),
                    metadata=node_dict
                )

            # Last resort: attempt a generic `get` and adapt
            node = getattr(self.graph, 'get', lambda k: None)(key)
            if not node:
                return None
            return node if isinstance(node, MemoryItem) else MemoryItem(
                item_id=key,
                content=getattr(node, "content", ""),
                metadata=getattr(node, "metadata", {}) if hasattr(node, 'metadata') else {}
            )
        except Exception as e:
            logger.error(f"Failed to get zettel item {key}: {e}")
            return None

    def _update_impl(self, item: MemoryItem, **kwargs) -> Optional[MemoryItem]:
        """Zettel-specific update logic."""
        try:
            # Update the graph node with new data
            zettel_data = {
                "item_id": item.item_id,
                "title": item.metadata.get("title") or item.metadata.get("description") or item.item_id,
                "content": item.content,
                "zettel_body": item.content,
                "description": item.metadata.get("description", "Zettel note"),
                "tags": item.metadata.get("tags", []),
                "group_id": getattr(item, "group_id", ""),
            }

            # Update in graph
            if hasattr(self.graph, 'update'):
                updated_node = self.graph.update(item.item_id, zettel_data, **kwargs)
            else:
                # Fallback: remove and re-add
                self.graph.remove(item.item_id)
                updated_node = self.graph.add(zettel_data, **kwargs)

            if updated_node:
                # Update item with new data
                item.metadata["_node"] = updated_node
                return item
            return None
        except Exception as e:
            logger.error(f"Failed to update zettel item {item.item_id}: {e}")
            return None

    # Keep advanced/dynamic methods for compatibility
    def add_dynamic_relation(self, source_id: Any, target_id: Any, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        return self.store.add_dynamic_relation(source_id, target_id, relation_type, properties)

    def find_notes_by_tag(self, tag_name: str) -> List[MemoryItem]:
        return self.store.find_notes_by_tag(tag_name)

    def find_notes_by_property(self, key: str, value: Any) -> List[MemoryItem]:
        return self.store.find_notes_by_property(key, value)

    def notes_linked_to(self, note_id: Any) -> List[MemoryItem]:
        return self.store.notes_linked_to(note_id)

    def notes_mentioning(self, entity_id: Any) -> List[MemoryItem]:
        return self.store.notes_mentioning(entity_id)

    def query_by_dynamic_relation(self, source_id: Any, relation_type: str) -> List[MemoryItem]:
        return self.store.query_by_dynamic_relation(source_id, relation_type)

    # === COMPLETE ZETTELKASTEN FUNCTIONALITY ===

    def get_backlinks(self, note_id: str) -> List[MemoryItem]:
        """Get all notes that link TO this note (backlinks)."""
        return self.backlinks.get_backlinks(note_id)

    def get_bidirectional_connections(self, note_id: str) -> Dict[str, List[MemoryItem]]:
        """Get complete bidirectional view of note connections."""
        return self.backlinks.get_bidirectional_connections(note_id)

    def create_bidirectional_link(self, source_id: str, target_id: str, link_type: str = "LINKS_TO"):
        """Create automatic bidirectional links for wikilinks."""
        return self.backlinks.create_bidirectional_link(source_id, target_id, link_type)

    def detect_knowledge_clusters(self, min_cluster_size: int = 3) -> List[KnowledgeCluster]:
        """Detect emergent knowledge clusters from connection patterns."""
        return self.structure.detect_knowledge_clusters(min_cluster_size)

    def find_knowledge_bridges(self) -> List[tuple]:
        """Find notes that bridge different knowledge domains."""
        return self.structure.find_knowledge_bridges()

    def detect_concept_emergence(self) -> Dict[str, float]:
        """Detect emerging concepts based on connection patterns."""
        return self.structure.detect_concept_emergence()

    def find_knowledge_paths(self, start_note_id: str, end_note_id: str, max_depth: int = 5) -> List[DiscoveryPath]:
        """Find paths between notes for knowledge discovery."""
        return self.discovery.find_knowledge_paths(start_note_id, end_note_id, max_depth)

    def suggest_related_notes(self, note_id: str, suggestion_count: int = 5) -> List[tuple]:
        """Suggest related notes for serendipitous discovery."""
        return self.discovery.suggest_related_notes(note_id, suggestion_count)

    def discover_missing_connections(self, note_id: str) -> List[tuple]:
        """Suggest connections that might be missing."""
        return self.discovery.discover_missing_connections(note_id)

    def random_walk_discovery(self, start_note_id: str, walk_length: int = 5) -> List[str]:
        """Perform random walk for serendipitous discovery."""
        return self.discovery.random_walk_discovery(start_note_id, walk_length)

    def get_zettelkasten_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of the Zettelkasten system state."""
        try:
            # Prefer direct graph-level stats for accuracy
            total_notes = 0
            total_connections = 0
            if hasattr(self.graph, 'graph'):
                # Count notes
                if hasattr(self.graph.graph, 'get_all_nodes'):
                    nodes = self.graph.graph.get_all_nodes()
                    # Treat anything with label 'Note' as a zettel note when dict-like
                    if nodes and isinstance(nodes[0], dict):
                        total_notes = sum(1 for n in nodes if n.get('label') == 'Note')
                    else:
                        total_notes = len(nodes)
                # Count edges
                if hasattr(self.graph.graph, 'get_all_edges'):
                    edges = self.graph.graph.get_all_edges()
                    total_connections = len(edges)

            # Fallback to connection-based counting if graph-level API is unavailable
            if total_notes == 0:
                all_notes = self.structure._get_all_notes()
                total_notes = len(all_notes)
                # Approximate connections by summing bidirectional buckets
                total_connections = 0
                for note in all_notes:
                    connections = self.get_bidirectional_connections(note.item_id)
                    total_connections += sum(len(conn_list) for conn_list in connections.values()) // 2

            # Get clusters and concepts via extensions
            clusters = self.detect_knowledge_clusters()
            emerging_concepts = self.detect_concept_emergence()

            return {
                'total_notes': total_notes,
                'total_connections': total_connections,
                'connection_density': (total_connections / max(total_notes * (total_notes - 1), 1)) if total_notes > 1 else 0.0,
                'knowledge_clusters': len(clusters),
                'top_clusters': [
                    {
                        'id': cluster.cluster_id,
                        'size': len(cluster.note_ids),
                        'concepts': cluster.central_concepts,
                        'emergence_score': cluster.emergence_score
                    }
                    for cluster in clusters[:5]
                ],
                'emerging_concepts': dict(list(emerging_concepts.items())[:10]),
                'system_health': 'healthy' if total_notes > 0 and total_connections > 0 else 'sparse'
            }
        except Exception as e:
            logger.error(f"Failed to get Zettelkasten overview: {e}")
            return {'error': str(e)}

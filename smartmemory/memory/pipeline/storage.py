"""
StorageEngine component for componentized memory ingestion pipeline.
Handles dual-node memory storage and triple processing with graph relationships.
"""
import logging
from typing import Dict, Any, Optional, List

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import StorageConfig
from smartmemory.memory.pipeline.state import ExtractionState
from smartmemory.memory.pipeline.transactions.change_set import ChangeOp, ChangeSet
from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)


class StorageEngine(PipelineComponent[StorageConfig]):
    """
    Component responsible for storing memory items and entities in the memory system.
    Handles dual-node storage (memory + entity nodes) and relationship processing.
    """

    def __init__(self, memory_instance=None):
        if memory_instance is None:
            raise ValueError("StorageEngine requires a valid memory_instance")
        self.memory = memory_instance

    def _process_extracted_triples(self, context: Dict[str, Any], item_id: str, triples: List) -> int:
        """Process extracted triples to create relationships in the graph"""
        edges_created = 0

        try:
            for triple in triples:
                if len(triple) != 3:
                    continue

                subject, predicate, obj = triple

                # Create relationship in graph
                try:
                    # Use SmartGraph's add_edge method (subject=source_id, obj=target_id, predicate=edge_type)
                    self.memory._graph.add_edge(
                        source_id=str(subject),
                        target_id=str(obj),
                        edge_type=str(predicate),
                        properties={"source": "extraction"}
                    )
                    edges_created += 1
                except Exception as e:
                    logger.warning(f"Failed to create edge {subject} -> {predicate} -> {obj}: {e}")
                    continue

            context['edges_created'] = edges_created
            return edges_created

        except Exception as e:
            logger.error(f"Error processing triples: {e}")
            context['edges_created'] = 0
            return 0

    def _build_ontology_extraction(self, entities: List, relations: List) -> Optional[Dict[str, Any]]:
        """Build ontology extraction payload from entities and relations"""
        if not entities and not relations:
            return None

        try:
            # Build ontology_extraction payload
            ontology_entities = []
            for entity in entities:
                if isinstance(entity, MemoryItem):
                    ontology_entities.append({
                        'id': getattr(entity, 'item_id', ''),
                        'name': entity.metadata.get('name', entity.content),
                        'type': entity.memory_type,
                        'content': entity.content,
                        'metadata': entity.metadata
                    })

            ontology_relations = []
            for relation in relations:
                if isinstance(relation, dict):
                    ontology_relations.append({
                        'source_id': relation.get('source_id', ''),
                        'target_id': relation.get('target_id', ''),
                        'relation_type': relation.get('relation_type', ''),
                        'metadata': relation.get('metadata', {})
                    })

            return {
                'entities': ontology_entities,
                'relations': ontology_relations
            }

        except Exception as e:
            logger.error(f"Error building ontology extraction: {e}")
            return None

    def validate_config(self, config: StorageConfig) -> bool:
        """Validate StorageEngine configuration using typed config"""
        try:
            # storage_strategy is validated in StorageConfig.__post_init__
            # relationship_creation and ontology_extraction are booleans
            if not isinstance(getattr(config, 'relationship_creation', True), bool):
                return False
            if not isinstance(getattr(config, 'ontology_extraction', True), bool):
                return False
            return True
        except Exception:
            return False

    def run(self, extraction_state: ExtractionState, config: StorageConfig) -> ComponentResult:
        """
        Execute StorageEngine with given extraction state and configuration.
        
        Args:
            extraction_state: ExtractionState from previous stage with entities/relations
            config: Storage configuration dict
        
        Returns:
            ComponentResult with storage results and metadata
        """
        try:
            if not extraction_state or not extraction_state.success:
                return ComponentResult(
                    success=False,
                    data={'error': 'Invalid or failed extraction state'},
                    metadata={'stage': 'storage_engine'}
                )

            # Extract data from extraction state
            entities = extraction_state.data.get('entities', [])
            relations = extraction_state.data.get('relations', [])
            triples = extraction_state.data.get('triples', [])

            # Get memory item from extraction state data
            memory_item = extraction_state.data.get('memory_item')
            if not memory_item:
                return ComponentResult(
                    success=False,
                    data={'error': 'No memory_item available for storage'},
                    metadata={'stage': 'storage_engine'}
                )

            # Detect preview mode (may be injected in config by API layer)
            preview_mode = bool(getattr(config, 'preview', False))
            proposed_ops: List[ChangeOp] = []

            # Build ontology extraction payload
            ontology_extraction = self._build_ontology_extraction(entities, relations)

            # Store the main memory item with dual-node strategy
            try:
                entity_ids = {}
                if preview_mode:
                    # Propose graph nodes for memory and entities
                    item_id = getattr(memory_item, 'item_id', None) or 'preview_item'
                    proposed_ops.append(ChangeOp(op_type='add_node', args={
                        'node_id': item_id,
                        'properties': {
                            'type': 'memory_node',
                            'content': getattr(memory_item, 'content', None),
                            'memory_type': getattr(memory_item, 'memory_type', None),
                        }
                    }))
                    # Propose entity nodes
                    for i, entity in enumerate(entities):
                        if not isinstance(entity, MemoryItem):
                            continue
                        entity_name = None
                        try:
                            entity_name = entity.metadata.get('name') if getattr(entity, 'metadata', None) else None
                        except Exception:
                            entity_name = None
                        ent_id = f"{item_id}_entity_{i}"
                        if entity_name:
                            entity_ids[entity_name] = ent_id
                        proposed_ops.append(ChangeOp(op_type='add_node', args={
                            'node_id': ent_id,
                            'properties': {
                                'type': 'entity_node',
                                'name': entity_name or f'entity_{i}',
                            }
                        }))
                else:
                    add_result = self.memory._crud.add(
                        memory_item,
                        ontology_extraction=ontology_extraction
                    )

                    # Handle different return formats
                    if isinstance(add_result, dict):
                        item_id = add_result.get('memory_node_id')
                        created_entity_ids = add_result.get('entity_node_ids', []) or []

                        # Map entity names to their created IDs
                        for i, entity in enumerate(entities):
                            if not isinstance(entity, MemoryItem):
                                continue
                            if not getattr(entity, 'metadata', None) or 'name' not in entity.metadata:
                                continue

                            entity_name = entity.metadata['name']
                            real_id = created_entity_ids[i] if i < len(created_entity_ids) else f"{item_id}_entity_{i}"
                            entity_ids[entity_name] = real_id
                    else:
                        # Legacy return format (string item_id)
                        item_id = add_result
                        for i, entity in enumerate(entities):
                            if not isinstance(entity, MemoryItem):
                                continue
                            if not getattr(entity, 'metadata', None) or 'name' not in entity.metadata:
                                continue

                            entity_name = entity.metadata['name']
                            entity_ids[entity_name] = f"{item_id}_entity_{i}"

                    # Update memory item with generated ID
                    memory_item.item_id = item_id
                    memory_item.update_status('created', notes='Item ingested')

            except Exception as e:
                return create_error_result('storage_engine', e, error_context='Storage failed')

            # Process triples/relationships
            edges_created = 0
            if triples and len(triples) > 0:
                if preview_mode:
                    for t in triples:
                        if len(t) != 3:
                            continue
                        subject, predicate, obj = t
                        try:
                            proposed_ops.append(ChangeOp(op_type='add_edge', args={
                                'source': subject,
                                'target': obj,
                                'relation_type': self._sanitize_relation_type(predicate) if hasattr(self, '_sanitize_relation_type') else (predicate or 'RELATED'),
                                'properties': {}
                            }))
                            edges_created += 1
                        except Exception:
                            # keep preview robust
                            pass
                else:
                    context = {
                        'item_id': item_id,
                        'entity_ids': entity_ids
                    }
                    edges_created = self._process_extracted_triples(context, item_id, triples)

            # Build stored nodes list
            stored_nodes = []
            if item_id:
                stored_nodes.append({
                    'id': item_id,
                    'type': 'memory_node',
                    'content': memory_item.content,
                    'memory_type': memory_item.memory_type
                })

            for entity_name, entity_id in entity_ids.items():
                stored_nodes.append({
                    'id': entity_id,
                    'type': 'entity_node',
                    'name': entity_name
                })

            # Build stored triples list
            stored_triples = []
            for triple in triples:
                if len(triple) == 3:
                    stored_triples.append({
                        'subject': triple[0],
                        'predicate': triple[1],
                        'object': triple[2]
                    })

            storage_metadata = {
                'storage_strategy': getattr(config, 'storage_strategy', 'dual_node'),
                'item_id': item_id,
                'nodes_created': len(stored_nodes),
                'relationships_created': edges_created,
                'ontology_extraction_used': ontology_extraction is not None
            }

            entities_processed = len(entities)
            logger.info(f"Successfully stored item {item_id} with {entities_processed} entities and {edges_created} edges")

            result_data: Dict[str, Any] = {
                'stored_nodes': stored_nodes,
                'stored_triples': stored_triples,
                'storage_metadata': storage_metadata,
                'item_id': item_id,
                'entity_ids': entity_ids,
                'edges_created': edges_created
            }

            # Attach change_set skeleton and ops in preview mode
            if preview_mode and proposed_ops:
                cs = ChangeSet.new(stage='storage', plugin='StorageEngine', run_id=getattr(config, 'run_id', None))
                cs.ops.extend(proposed_ops)
                result_data['change_set'] = {
                    'change_set_id': cs.change_set_id,
                    'ops_count': len(cs.ops),
                }
                result_data['change_set_ops'] = [
                    {'op_type': op.op_type, 'args': op.args}
                    for op in proposed_ops
                ]

            return ComponentResult(
                success=True,
                data=result_data,
                metadata={
                    'stage': 'storage_engine',
                    'item_id': item_id,
                    'nodes_count': len(stored_nodes),
                    'edges_count': edges_created
                }
            )

        except Exception as e:
            return create_error_result('storage_engine', e)

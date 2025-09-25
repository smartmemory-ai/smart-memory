"""
ExtractorPipeline component for componentized memory ingestion pipeline.
Handles entity/relation extraction with fallback chain and multiple extractor support.
"""
import logging
from typing import Dict, Any, Optional, List

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import ExtractionConfig
from smartmemory.memory.pipeline.state import ClassificationState
from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils import get_config
from smartmemory.utils.pipeline_utils import create_error_result

logger = logging.getLogger(__name__)


class ExtractorPipeline(PipelineComponent[ExtractionConfig]):
    """
    Component responsible for entity and relation extraction with fallback chain.
    Supports multiple extractors (llm, spacy, gliner, relik, ontology) with automatic fallback.
    """

    def __init__(self, memory=None):
        self._logger = logging.getLogger(__name__)
        self.memory = memory
        # Registry maps extractor name -> factory (zero-arg callable) or concrete callable
        self.extractor_registry: Dict[str, Any] = {}
        # Cache for resolved extractors (name -> callable)
        self._extractor_cache: Dict[str, Any] = {}
        self._last_validation_errors: List[str] = []
        self._register_default_extractors()

    def _register_default_extractors(self):
        """Register default extractors as LAZY factories (import on first use)."""

        # Each factory imports the plugin on demand and returns the extractor callable

        def _lazy_spacy():
            from smartmemory.plugins.extractors.spacy import make_spacy_extractor
            return make_spacy_extractor()

        def _lazy_gliner():
            from smartmemory.plugins.extractors.gliner import make_gliner_extractor
            return make_gliner_extractor()

        def _lazy_relik():
            from smartmemory.plugins.extractors.relik import make_relik_extractor
            return make_relik_extractor()

        def _lazy_llm():
            from smartmemory.plugins.extractors.llm import make_llm_extractor
            return make_llm_extractor()

        def _lazy_ontology():
            # Gate on config at resolution time
            from smartmemory.extraction.extractor import create_ontology_aware_extractor
            llm_cfg = {}
            try:
                llm_cfg = get_config('extractor').get('llm') or {}
            except Exception:
                llm_cfg = {}
            if not llm_cfg.get('openai_api_key'):
                raise ImportError("ontology extractor requires extractor.llm.openai_api_key")
            return create_ontology_aware_extractor()

        def _lazy_rebel():
            from smartmemory.plugins.extractors.rebel import make_rebel_extractor
            return make_rebel_extractor()

        # Register factories (no imports executed yet)
        self.register_extractor_factory('spacy', _lazy_spacy)
        self.register_extractor_factory('gliner', _lazy_gliner)
        self.register_extractor_factory('relik', _lazy_relik)
        self.register_extractor_factory('llm', _lazy_llm)
        self.register_extractor_factory('ontology', _lazy_ontology)
        self.register_extractor_factory('rebel', _lazy_rebel)

        # Back-compat alias
        self.register_extractor_factory('gpt4o_triple', _lazy_llm)

        # Log lazy registry keys
        try:
            self._logger.info("Lazy-registered extractor factories: %s", sorted(self.extractor_registry.keys()))
        except Exception:
            pass

    def register_extractor(self, name: str, extractor_fn):
        """Register a new extractor by name"""
        self.extractor_registry[name] = extractor_fn

    def register_extractor_factory(self, name: str, factory_fn):
        """Register a lazy factory that resolves the extractor on first use."""
        self.extractor_registry[name] = factory_fn

    def _resolve_extractor(self, name: str):
        """Resolve an extractor by name, importing on demand if registered as a factory."""
        if name in self._extractor_cache:
            return self._extractor_cache[name]
        val = self.extractor_registry.get(name)
        if val is None:
            return None
        # If it's a zero-arg factory, call it
        try:
            candidate = val()
            self._extractor_cache[name] = candidate
            # Overwrite registry with concrete callable to avoid repeated factory calls
            self.extractor_registry[name] = candidate
            return candidate
        except TypeError:
            # Not callable or not a zero-arg factory; assume it's already an extractor callable
            self._extractor_cache[name] = val
            return val
        except Exception as e:
            self._logger.error("Failed to resolve extractor '%s': %s", name, e)
            return None

    def _get_fallback_order(self, primary: str = None) -> List[str]:
        """Get fallback order for extractors, excluding primary"""
        # Read from config but be resilient to ConfigDict fail-fast semantics
        try:
            cfg = get_config('extractor')  # ConfigDict
            try:
                fallback_order = cfg['fallback_order']  # type: ignore[index]
            except KeyError:
                fallback_order = ['llm', 'spacy', 'gliner', 'relik']
        except Exception:
            fallback_order = ['llm', 'spacy', 'gliner', 'relik']

        if primary and isinstance(fallback_order, list):
            try:
                return [x for x in fallback_order if x != primary]
            except Exception:
                return []
        return fallback_order if isinstance(fallback_order, list) else ['llm', 'spacy', 'gliner', 'relik']

    def validate_config(self, config: ExtractionConfig) -> bool:
        """Validate ExtractorPipeline configuration using typed config"""
        try:
            errors: List[str] = []

            # extractor_name: optional string
            extractor_name = getattr(config, 'extractor_name', None)
            if extractor_name is not None and not isinstance(extractor_name, str):
                errors.append("extractor_name must be a string if provided")

            # max_entities: positive int
            max_entities = getattr(config, 'max_entities', None)
            if max_entities is not None and (not isinstance(max_entities, int) or max_entities <= 0):
                errors.append("max_entities must be a positive integer")

            # enable_relations: bool if provided
            er = getattr(config, 'enable_relations', None)
            if er is not None and not isinstance(er, bool):
                errors.append("enable_relations must be a boolean")

            if errors:
                try:
                    self._logger.warning("ExtractorPipeline config validation failed: %s", "; ".join(errors))
                except Exception:
                    pass
                self._last_validation_errors = errors
                return False

            self._last_validation_errors = []
            return True
        except Exception:
            return False

    def _select_default_extractor(self) -> Optional[str]:
        """Select the default extractor using config and availability"""
        try:
            cfg = get_config('extractor') or {}
        except Exception:
            cfg = {}

        default = cfg.get('default', 'llm')
        if default in self.extractor_registry:
            return default

        # Use first available from fallback order
        for name in self._get_fallback_order(primary=None):
            if name in self.extractor_registry:
                return name

        # Last resort - ontology or any registered
        if 'ontology' in self.extractor_registry:
            return 'ontology'

        return next(iter(self.extractor_registry.keys()), None)

    def _extract_with_ontology(self, memory_item, fallback_order: List[str]) -> Dict[str, Any]:
        """Handle ontology-aware extraction with fallback"""
        try:
            extractor = self._resolve_extractor('ontology')
            if extractor is None:
                raise ImportError("ontology extractor not available")
            user_id = getattr(memory_item, 'user_id', None) or memory_item.metadata.get('user_id')
            # Pull optional knobs from the last validated config if available
            # (Run() sets config; here we grab from a private attr if present, else default)
            cfg = getattr(self, "_current_cfg", None)
            kwargs = {}
            if cfg is not None:
                try:
                    if getattr(cfg, 'ontology_enabled', None) is not None:
                        kwargs['ontology_enabled'] = bool(cfg.ontology_enabled)
                    if getattr(cfg, 'ontology_constraints', None):
                        kwargs['ontology_constraints'] = list(cfg.ontology_constraints)
                    if getattr(cfg, 'model', None):
                        kwargs['model'] = str(cfg.model)
                    if getattr(cfg, 'temperature', None) is not None:
                        kwargs['temperature'] = float(cfg.temperature)
                    if getattr(cfg, 'max_tokens', None) is not None:
                        kwargs['max_tokens'] = int(cfg.max_tokens)
                    if getattr(cfg, 'reasoning_effort', None):
                        kwargs['reasoning_effort'] = str(cfg.reasoning_effort)
                    if getattr(cfg, 'max_reasoning_tokens', None) is not None:
                        kwargs['max_reasoning_tokens'] = int(cfg.max_reasoning_tokens)
                except Exception:
                    pass
            result = extractor.extract_entities_and_relations(
                memory_item.content,
                user_id,
                **kwargs,
            )

            # Convert OntologyNode objects to MemoryItem objects
            entities = []
            for node in result['entities']:
                memory_item_converted = node.to_memory_item()
                entities.append(memory_item_converted)

            relations = result['relations']

            # If ontology returns nothing, fall back
            if not entities and not relations:
                return self._extract_with_fallback(memory_item, fallback_order)

            return {
                'entities': entities,
                'triples': [(r['source_id'], r['relation_type'], r['target_id']) for r in relations],
                'relations': relations
            }

        except Exception:
            # On failure, fall back to configured extractors
            return self._extract_with_fallback(memory_item, fallback_order)

    def _extract_with_fallback(self, memory_item, fallback_order: List[str]) -> Dict[str, Any]:
        """Try extractors in fallback order until one succeeds"""
        for fb_name in fallback_order:
            fb = self._resolve_extractor(fb_name)
            if not fb:
                continue

            try:
                fb_res = fb(memory_item)
            except Exception as e:
                logger.warning(f"Extractor {fb_name} failed: {e}")
                continue

            # Normalize different output formats
            if isinstance(fb_res, dict):
                fb_entities = fb_res.get('entities', [])
                fb_triples = fb_res.get('triples', [])
                fb_relations = fb_res.get('relations', [])
            elif isinstance(fb_res, tuple) and len(fb_res) == 3:
                _it, fb_entities, fb_relations = fb_res
                fb_triples = [rel for rel in fb_relations if isinstance(rel, (list, tuple)) and len(rel) == 3]
            else:
                fb_entities, fb_triples, fb_relations = [], [], []

            # Convert entities to MemoryItem objects for consistency
            converted_entities = []
            for i, ent in enumerate(fb_entities):
                if isinstance(ent, MemoryItem):
                    converted_entities.append(ent)
                elif isinstance(ent, dict):
                    entity_content = ent.get('name', ent.get('text', f'entity_{i}'))
                    entity_type = ent.get('type', ent.get('label', 'entity'))
                    metadata = {
                        'name': ent.get('name', entity_content),
                        'confidence': ent.get('confidence', 1.0),
                        'source': 'legacy_extractor'
                    }
                    # Copy additional fields to metadata
                    for k, v in ent.items():
                        if k not in ['name', 'text', 'type', 'label', 'confidence']:
                            metadata[k] = v
                    converted_entities.append(MemoryItem(
                        content=entity_content,
                        memory_type=entity_type,
                        metadata=metadata
                    ))
                elif isinstance(ent, str):
                    converted_entities.append(MemoryItem(
                        content=ent,
                        memory_type='entity',
                        metadata={'name': ent, 'confidence': 1.0, 'source': 'legacy_extractor'}
                    ))
                else:
                    ent_str = str(ent)
                    converted_entities.append(MemoryItem(
                        content=ent_str,
                        memory_type='entity',
                        metadata={'name': ent_str, 'confidence': 1.0, 'source': 'legacy_extractor'}
                    ))

            # Return if we got any results
            if converted_entities or fb_triples or fb_relations:
                return {
                    'entities': converted_entities,
                    'triples': fb_triples,
                    'relations': fb_relations
                }

        # Nothing worked - return empty results
        return {
            'entities': [],
            'triples': [],
            'relations': []
        }

    def run(self, classification_state: ClassificationState, config: ExtractionConfig) -> ComponentResult:
        """
        Execute ExtractorPipeline with given classification state and configuration.
        
        Args:
            classification_state: ClassificationState from previous stage
            config: Extraction configuration dict with optional 'extractor_name'
        
        Returns:
            ComponentResult with entities, relations, and extraction metadata
        """
        try:
            # Accept either ClassificationState or InputState; require success
            if not classification_state or not getattr(classification_state, 'success', False):
                return create_error_result(
                    'extractor_pipeline',
                    ValueError('Invalid or failed prior state')
                )

            # Validate config with clear failure signaling
            if not self.validate_config(config):
                # Surface concrete reasons to caller/UI
                msg = "; ".join(self._last_validation_errors) if getattr(self, "_last_validation_errors", None) else "invalid configuration"
                return create_error_result('extractor_pipeline', ValueError(f'Extractor config invalid: {msg}'))

            # Get memory item from prior state (prefer data field, fallback to attribute)
            memory_item = None
            try:
                memory_item = classification_state.data.get('memory_item')
            except Exception:
                memory_item = None
            if not memory_item:
                memory_item = getattr(classification_state, 'memory_item', None)
            if not memory_item:
                return create_error_result(
                    'extractor_pipeline',
                    ValueError('No memory_item available for extraction')
                )

            # Cache config for ontology subcall knobs
            try:
                self._current_cfg = config
            except Exception:
                self._current_cfg = None

            # Determine extractor to use
            extractor_name = getattr(config, 'extractor_name', None)
            if not extractor_name:
                extractor_name = self._select_default_extractor()

            if not extractor_name:
                return create_error_result('extractor_pipeline', ValueError('No extractor available: none registered and no default could be selected'))

            # Get fallback order and filter to registered ones
            fallback_order = [n for n in self._get_fallback_order(primary=extractor_name) if n in self.extractor_registry]

            # Handle extraction based on extractor type
            if extractor_name == 'ontology':
                extraction_result = self._extract_with_ontology(memory_item, fallback_order)
            else:
                # Try primary extractor first, then fallback
                all_extractors = [n for n in ([extractor_name] + fallback_order) if n in self.extractor_registry]
                if not all_extractors:
                    available = ", ".join(sorted(self.extractor_registry.keys()))
                    return create_error_result('extractor_pipeline', ValueError(f"Requested extractor '{extractor_name}' is not registered; available: [{available}]"))
                extraction_result = self._extract_with_fallback(memory_item, all_extractors)

            # Build extraction metadata
            extraction_metadata = {
                'extractor_requested': extractor_name,
                'extractor_used': extractor_name,  # TODO: track actual extractor used in fallback
                'entities_found': len(extraction_result['entities']),
                'triples_found': len(extraction_result['triples']),
                'relations_found': len(extraction_result['relations']),
                'fallback_available': len(fallback_order) > 0
            }

            return ComponentResult(
                success=True,
                data={
                    'memory_item': memory_item,  # Pass memory_item forward for storage
                    'entities': extraction_result['entities'],
                    'relations': extraction_result['relations'],
                    'triples': extraction_result['triples'],
                    'extraction_metadata': extraction_metadata,
                    'extractor_used': extractor_name
                },
                metadata={
                    'stage': 'extractor_pipeline',
                    'extractor_name': extractor_name,
                    'results_count': len(extraction_result['entities']) + len(extraction_result['relations'])
                }
            )

        except Exception as e:
            return create_error_result(
                'extractor_pipeline',
                e,
                extractor_name=getattr(config, 'extractor_name', 'unknown')
            )

# Plugin registry for enrichers and other agentic memory plugins
# This avoids circular imports and centralizes plugin management.
from smartmemory.plugins.enrichers import WikipediaEnricher, TemporalEnricher
from smartmemory.plugins.extractors import make_spacy_extractor, make_gliner_extractor, make_rebel_extractor, make_llm_extractor
from smartmemory.plugins.extractors.relik import make_relik_extractor

ENRICHER_REGISTRY = {
    'wikipedia': WikipediaEnricher,
    'temporal': TemporalEnricher,
}

EXTRACTOR_REGISTRY = {
    'spacy': make_spacy_extractor,
    'gliner': make_gliner_extractor,
    'relik': make_relik_extractor,
    'rebel': make_rebel_extractor,
    'llm': make_llm_extractor,
    # Add future extractors here
}

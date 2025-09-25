# Enricher submodule registry for agentic memory (typed-config pipeline compliant)
import importlib
from typing import Any, Dict, Optional, Type

from .basic import BasicEnricher
from .sentiment import SentimentEnricher
from .skills_tools import ExtractSkillsToolsEnricher
from .temporal import TemporalEnricher
from .topic import TopicEnricher
from .wikipedia import WikipediaEnricher

# Public list of Enricher classes (not instances)
ENRICHERS = [
    BasicEnricher,
    SentimentEnricher,
    TemporalEnricher,
    ExtractSkillsToolsEnricher,
    TopicEnricher,
    WikipediaEnricher,
]

# Backward-compatible registry returning bound callables with default typed configs when available
ENRICHER_REGISTRY = {
    'basic_enricher': BasicEnricher().enrich,
    'sentiment_enricher': SentimentEnricher().enrich,
    'temporal_enricher': TemporalEnricher().enrich,
    'extract_skills_tools': ExtractSkillsToolsEnricher().enrich,
    'topic_enricher': TopicEnricher().enrich,
    'wikipedia_enricher': WikipediaEnricher().enrich,
}


def _build_typed_config(enricher_cls: Type, config_snapshot: Optional[Dict[str, Any]]):
    """Infer and construct the typed Config class co-located with the enricher class.
    Convention: replace suffix 'Enricher' with 'EnricherConfig' within the same module.
    Supports per-enricher snapshots via keys matching the Enricher or Config class name.
    """
    module = importlib.import_module(enricher_cls.__module__)
    enricher_name = enricher_cls.__name__
    if not enricher_name.endswith("Enricher"):
        raise TypeError(
            f"Enricher class {enricher_name} does not follow naming convention with 'Enricher' suffix"
        )
    config_name = enricher_name.replace("Enricher", "EnricherConfig")
    ConfigType: Optional[Type] = getattr(module, config_name, None)
    if ConfigType is None:
        # If no typed config exists, instantiate without config (legacy plugin)
        # The class may not require a config; pipeline will still run.
        return None
    if config_snapshot is None:
        return ConfigType()
    # Support a mapping of {EnricherName or ConfigName: {..params..}}
    if isinstance(config_snapshot, dict):
        sub = None
        if enricher_name in config_snapshot and isinstance(config_snapshot[enricher_name], dict):
            sub = config_snapshot[enricher_name]
        elif config_name in config_snapshot and isinstance(config_snapshot[config_name], dict):
            sub = config_snapshot[config_name]
        elif all(k in ConfigType.__annotations__ for k in config_snapshot.keys()):
            # Looks like a flat config for this enricher only
            sub = config_snapshot
        if sub is None:
            return ConfigType()
        return ConfigType(**sub)
    if isinstance(config_snapshot, ConfigType):
        return config_snapshot
    raise TypeError("Config snapshot must be a dict or the typed Config instance for enricher")


def run_enrichment_cycle(item, configs: Optional[Dict[str, Any]] = None, logger=None) -> Dict[str, Any]:
    """Run all enrichers over a single item, using typed configs by convention.
    Returns a merged enrichment result dict.
    """
    merged: Dict[str, Any] = {}
    for EnricherClass in ENRICHERS:
        typed_config = _build_typed_config(EnricherClass, configs)
        try:
            enricher = EnricherClass(config=typed_config) if typed_config is not None else EnricherClass()
            result = enricher.enrich(item)
            if isinstance(result, dict):
                # Shallow merge; later enrichers can override keys
                merged.update(result)
        except Exception as e:
            if logger:
                logger.error(f"Error in {EnricherClass.__name__}: {e}")
    return merged

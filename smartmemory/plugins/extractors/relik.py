def make_relik_extractor():
    """
    Factory for a Relik-based relation extractor.
    Uses the relik library to extract entity-relation-entity triples from text.
    Returns (item, entities, relations).
    """

    def relik_extractor(item):
        try:
            from relik import Relik
        except ImportError:
            raise ImportError("relik is not installed. Please install with 'pip install relik'.")
        from smartmemory.configuration import MemoryConfig
        config = MemoryConfig().extractor
        relik_cfg = config.get('relik') or {}
        model_name = relik_cfg.get('model_name')
        if not model_name:
            raise ValueError("No model_name specified in config under extractor['relik']['model_name'].")
        model = Relik.from_pretrained(model_name)
        content = item.content if hasattr(item, 'content') else str(item)
        output = model(content)
        triples = output.triples
        entities = list(set([t[0] for t in triples] + [t[2] for t in triples]))
        relations = [(t[0], t[1], t[2]) for t in triples]
        triples = []
        for rel in relations:
            if isinstance(rel, dict) and {'subject', 'predicate', 'object'} <= set(rel):
                triples.append((rel['subject'], rel['predicate'], rel['object']))
            elif isinstance(rel, (list, tuple)) and len(rel) == 3:
                triples.append(tuple(rel))
        return {'entities': entities, 'triples': triples}

    return relik_extractor

from smartmemory.utils import get_config


def make_gliner_extractor():
    """
    Factory for a GLiNER-based entity extractor.
    Uses the gliner library to extract entities from text.
    Returns (item, entities, []) with no relations.
    """

    def gliner_extractor(item):
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError("gliner is not installed. Please install with 'pip install gliner'.")
        config = get_config('extractor')
        gliner_cfg = config.get('gliner') or {}
        model_name = gliner_cfg.get('model_name')
        if not model_name:
            raise ValueError("No model_name specified in config under extractor['gliner']['model_name'].")
        model = GLiNER.from_pretrained(model_name)
        content = item.content if hasattr(item, 'content') else str(item)
        entities = [ent['text'] for ent in model.predict_entities(content, labels=None)]
        return {'entities': entities, 'triples': []}

    return gliner_extractor

"""
REBEL-based entity and relation extractor.
Extracts entities and relations from text using the REBEL models (via huggingface transformers pipeline).
"""

from smartmemory.utils import get_config


def make_rebel_extractor():
    """
    Factory for a REBEL-based entity and relation extractor.
    Uses the transformers pipeline with a REBEL models to extract subject-relation-object triples from text.
    Returns (item, entities, relations).
    """

    def rebel_extractor(item):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers is not installed. Please install with 'pip install transformers'.")
        config = get_config('extractor')
        rebel_cfg = config.get('rebel') or {}
        model_name = rebel_cfg.get('model_name')
        if not model_name:
            raise ValueError("No model_name specified in config under extractor['rebel']['model_name'].")
        nlp = pipeline('text2text-generation', model=model_name)
        content = item.content if hasattr(item, 'content') else str(item)
        result = nlp(content, max_length=512, clean_up_tokenization_spaces=True)[0]['generated_text']
        # Parse REBEL output (subject, relation, object triples)
        import re
        triple_pattern = r"\(.*?\)"
        triples = re.findall(triple_pattern, result)
        entities = set()
        relations = []
        for triple in triples:
            parts = triple.strip("() ").split(",")
            if len(parts) == 3:
                subj, rel, obj = [p.strip().strip('"') for p in parts]
                entities.add(subj)
                entities.add(obj)
                relations.append((subj, rel, obj))
        triples = []
        for rel in relations:
            if isinstance(rel, dict) and {'subject', 'predicate', 'object'} <= set(rel):
                triples.append((rel['subject'], rel['predicate'], rel['object']))
            elif isinstance(rel, (list, tuple)) and len(rel) == 3:
                triples.append(tuple(rel))
        return {'entities': list(entities), 'triples': triples}

    return rebel_extractor

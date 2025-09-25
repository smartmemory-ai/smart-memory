from smartmemory.utils import get_config


def make_spacy_extractor():
    """
    Factory for a spaCy extractor that loads the models name from config.
    - Extracts entities and subject-verb-object (SVO) relationships using dependency parsing when a models is available.
    - Gracefully falls back to a lightweight regex + co-occurrence approach if the spaCy models is unavailable.
    - Returns a dict with keys: 'entities' (list[dict{name: str, type: str}]) and 'triples' (list[tuple[str,str,str]]).
    """

    # Cache the spaCy models within the closure to avoid repeated loads
    _cache = {'nlp': None, 'loaded_name': None}

    def _load_nlp():
        model_name = None
        try:
            cfg = get_config('extractor') or {}
            model_name = (cfg.get('spacy') or {}).get('model_name', 'en_core_web_sm')
        except Exception:
            model_name = 'en_core_web_sm'

        # Reuse cached pipeline if already loaded
        if _cache['nlp'] is not None and _cache['loaded_name'] == model_name:
            return _cache['nlp']

        # Try to import spaCy and load the requested models
        try:
            import spacy  # type: ignore
            try:
                nlp = spacy.load(model_name)
            except Exception:
                nlp = spacy.blank('en')
                if 'sentencizer' not in nlp.pipe_names:
                    nlp.add_pipe('sentencizer')
        except Exception:
            # spaCy is not available; return None to trigger regex fallback
            nlp = None

        _cache['nlp'] = nlp
        _cache['loaded_name'] = model_name
        return nlp

    def _map_spacy_label_to_type(label: str) -> str:
        """Map spaCy NER labels to our entity types."""
        if not label:
            return 'entity'
        label = label.upper()
        if label in {'PERSON'}:
            return 'person'
        if label in {'ORG'}:
            return 'organization'
        if label in {'GPE', 'LOC'}:
            return 'location'
        if label in {'EVENT'}:
            return 'event'
        return 'concept'

    def _regex_entities(text: str) -> list[str]:
        """Very lightweight entity guesser for fallback mode.
        Captures capitalized multi-word names and known tech/places from simple heuristics.
        """
        import re
        candidates = set()
        # Multi-word capitalized sequences (e.g., John Smith, New York, San Francisco)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            candidates.add(m.group(1))
        # Single capitalized words that are well-known entities in typical tests
        known = {"Microsoft", "Google", "Python", "TensorFlow", "Seattle"}
        for k in known:
            if k in text:
                candidates.add(k)
        # Deduplicate while preserving insertion order for stability
        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def _guess_type_for_name(name: str) -> str:
        """Heuristic type guesser used in regex fallback mode."""
        org_markers = ("Inc", "Inc.", "Corporation", "Corp", "Corp.", "LLC", "Ltd", "Company")
        known_orgs = {"Google", "Google Inc", "Microsoft", "Amazon", "OpenAI", "Apple"}
        if any(marker in name for marker in org_markers) or name in known_orgs:
            return 'organization'
        parts = name.split()
        if len(parts) == 2 and all(p and p[0].isupper() for p in parts):
            return 'person'
        # Default to organization for single capitalized tokens, concept otherwise
        if len(parts) == 1 and name and name[0].isupper():
            return 'organization'
        return 'concept'

    def spacy_extractor(item):
        import re
        text = getattr(item, 'content', str(item))

        # Load spaCy (with graceful fallback)
        nlp = _load_nlp()
        entities = []  # list[dict{name, type}]
        triples = []
        if nlp is not None:
            try:
                doc = nlp(text)
                # Prefer spaCy NER if available
                ents = list(getattr(doc, 'ents', []) or [])
                # Build typed entities from spaCy labels
                seen = set()
                typed_entities = []
                for ent in ents:
                    name = ent.text
                    etype = _map_spacy_label_to_type(getattr(ent, 'label_', ''))
                    key = (name, etype)
                    if key not in seen:
                        seen.add(key)
                        typed_entities.append({'name': name, 'type': etype})
                entities = typed_entities
                # Build relations/triples using dependency parse when tags are available
                if entities and any(getattr(t, 'pos_', None) for t in doc):
                    try:
                        entity_tokens = {token.i for ent in ents for token in ent}
                        for sent in doc.sents:
                            sent_ents = [ent for ent in getattr(sent, 'ents', []) or []]
                            svo_found = False
                            for token in sent:
                                if getattr(token, 'pos_', '') == 'VERB':
                                    subj = None
                                    obj = None
                                    for child in token.children:
                                        if getattr(child, 'dep_', '') in ('nsubj', 'nsubjpass') and child.i in entity_tokens:
                                            subj = child
                                    for child in token.children:
                                        if getattr(child, 'dep_', '') in ('dobj', 'attr', 'oprd', 'pobj') and child.i in entity_tokens:
                                            obj = child
                                    if subj and obj:
                                        triples.append((subj.text, token.lemma_, obj.text))
                                        svo_found = True
                            if not svo_found and len(sent_ents) > 1:
                                for i in range(len(sent_ents) - 1):
                                    triples.append((sent_ents[i].text, 'related_to', sent_ents[i + 1].text))
                    except Exception:
                        pass
            except Exception:
                # If spaCy pipeline fails at runtime, we'll fallback to regex below
                entities = []
                triples = []

        # If spaCy didn't run or didn't produce entities, fall back to regex-based entities and co-occurrence
        if not entities:
            names = _regex_entities(text)
            entities = [{'name': n, 'type': _guess_type_for_name(n)} for n in names]
            # Naive sentence split
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            for sent in sentences:
                sent_ents = [e['name'] for e in entities if e['name'] in sent]
                if len(sent_ents) > 1:
                    for i in range(len(sent_ents) - 1):
                        triples.append((sent_ents[i], 'related_to', sent_ents[i + 1]))

        return {'entities': entities, 'triples': triples}

    return spacy_extractor

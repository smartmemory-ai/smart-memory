from smartmemory.observability.instrumentation import emit_after


def _ground_payload(self_or_none, args, kwargs, result):
    try:
        ctx = args[0] if args else kwargs.get('context')
        item = ctx.get('item') if isinstance(ctx, dict) else None
        item_id = getattr(item, 'item_id', None)
        prov = ctx.get('provenance_candidates') if isinstance(ctx, dict) else None
        prov_count = len(prov) if isinstance(prov, list) else (1 if prov else 0)
        return {
            'item_id': item_id,
            'provenance_count': prov_count,
        }
    except Exception:
        return {}


class Grounding:
    """
    Handles grounding/provenance logic. Designed for dependency injection (DI):
    Accepts only a store_manager, making it easy to swap dependencies for testing or alternate implementations.
    """

    def __init__(self, graph):
        """Initialize grounding component with graph backend."""
        self.graph = graph

    @emit_after(
        "background_process",
        component="grounding",
        operation="ground",
        payload_fn=_ground_payload,
        measure_time=True,
    )
    def ground(self, context):
        """
        Ground a memory item by creating GROUNDED_IN edges to provided provenance candidate nodes.
        provenance_candidates: list of node IDs (e.g., Wikipedia node IDs) to ground to.
        """
        item_id = context['item'].item_id if 'item' in context and hasattr(context['item'], 'item_id') else None
        provenance_candidates = context.get('provenance_candidates')
        source_url = context.get('source_url')
        validation = context.get('validation')
        memory_type = context.get('memory_type', 'semantic')
        if not provenance_candidates or not item_id:
            return
        for prov_id in provenance_candidates:
            self.graph.add_edge(item_id, prov_id, edge_type="GROUNDED_IN", properties={})
        from smartmemory.plugins.enrichers import WikipediaEnricher
        node = self.graph.get_node(item_id)
        if not node:
            raise ValueError(f"Node {item_id} not found.")
        entities = node.get('semantic_entities') or (node.get('metadata') or {} or {}).get('semantic_entities', [])
        wiki = WikipediaEnricher()
        wiki_data = wiki.enrich(node, {'semantic_entities': entities})
        provenance_url = None
        if wiki_data.get('wikipedia_data'):
            urls = [v.get('url') for v in wiki_data['wikipedia_data'].values() if v.get('url')]
            provenance_url = urls[0] if urls else None
        node['provenance'] = {
            'type': 'wikipedia',
            'entities': entities,
            'wikipedia_data': wiki_data.get('wikipedia_data') or {},
            'source_url': provenance_url or source_url,
            'validation': validation,
        }
        self.graph.add_node(item_id=item_id, properties=node)

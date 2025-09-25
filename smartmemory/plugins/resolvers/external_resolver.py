from smartmemory.memory.registry import ENRICHER_REGISTRY
from smartmemory.models.memory_item import MemoryItem
from smartmemory.observability.instrumentation import emit_after


class ExternalResolver:
    def __init__(self):
        self.enricher_registry = ENRICHER_REGISTRY

    def _payload(self, node, results):
        try:
            ref_count = 0
            if node and getattr(node, 'metadata', None):
                refs = node.metadata.get('external_refs')
                if isinstance(refs, list):
                    ref_count = len(refs)
                elif node.metadata.get('external_ref'):
                    ref_count = 1
            resolved_count = len(results) if isinstance(results, list) else (0 if results is None else 1)
            return {"references": ref_count, "resolved_count": resolved_count}
        except Exception:
            return {}

    @emit_after(
        "performance_metrics",
        component="resolver",
        operation="resolve_external",
        payload_fn=lambda self, args, kwargs, result: self._payload(args[0] if args else kwargs.get('node'), result),
        measure_time=True,
    )
    def resolve_external(self, node: MemoryItem):
        """
        Resolve linked resources from other registered memory backends (hybrid memory).
        If node.metadata contains 'external_refs' (list of dicts with 'ref' and 'type'), resolve all.
        For backward compatibility, also supports single 'external_ref' and 'external_type'.
        Returns a list of resolved MemoryItems (empty if none resolved).
        """
        results = []
        if node and getattr(node, 'metadata', None):
            refs = node.metadata.get('external_refs')
            if isinstance(refs, list):
                for ref_entry in refs:
                    ext = ref_entry.get('ref')
                    ext_type = ref_entry.get('type')
                    if ext and ext_type:
                        backend = None  # No memory type registry; implement if needed
                        if backend and hasattr(backend, "get"):
                            resolved = backend.get(ext)
                            if resolved:
                                results.append(resolved)
            else:
                # Backward compatibility: single external_ref/external_type
                ext = node.metadata.get("external_ref")
                ext_type = node.metadata.get("external_type")
                if ext and ext_type:
                    backend = None  # No memory type registry; implement if needed
                    if backend and hasattr(backend, "get"):
                        resolved = backend.get(ext)
                        if resolved:
                            results.append(resolved)
        return results if results else None

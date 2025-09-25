from datetime import datetime, timedelta

from smartmemory.models.memory_item import MemoryItem


class Monitoring:
    def __init__(self, graph):
        self._graph = graph

    def summary(self):
        insights = {}
        type_names = ["semantic", "episodic", "procedural", "working"]
        for name in type_names:
            items = self._graph.search_nodes({"$or": [{"type": name}, {"tags": name}]})
            keyword_count = {}
            for node in items:
                for word in str(node.get("content", "")).split():
                    keyword_count[word] = keyword_count.get(word, 0) + 1
            insights[name] = {
                "count": len(items),
                "top_keywords": sorted(keyword_count, key=keyword_count.get, reverse=True)[:5],
            }
        return insights

    def orphaned_notes(self):
        orphaned = []
        for note in self._graph.search_nodes({"$or": [{"type": "note"}, {"tags": "note"}]}):
            neighbors = self._graph.get_neighbors(note.get('item_id'))
            if not neighbors:
                orphaned.append(note)
        return orphaned

    def prune(self, strategy="old", days=365, **kwargs):
        if strategy == "old":
            old_notes = self.find_old_notes(days=days)
            for note in old_notes:
                self._graph.remove_node(note.item_id)
            return len(old_notes)
        else:
            raise ValueError(f"Unknown prune strategy: {strategy}")

    def find_old_notes(self, days: int = 365):
        cutoff = datetime.now() - timedelta(days=days)
        old_notes = []
        for node in self._graph.search_nodes({"$or": [{"type": "note"}, {"tags": "note"}]}):
            created_at = None
            if 'created_at' in node:
                created_at = node['created_at']
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except Exception:
                        continue
            if created_at and created_at < cutoff:
                old_notes.append(MemoryItem(item_id=node.get("item_id"), content=node.get("content", ""), metadata=node))
        return old_notes

    def self_monitor(self):
        status = {}
        type_names = ["semantic", "episodic", "procedural", "working"]
        for name in type_names:
            items = self._graph.search_nodes({"$or": [{"type": name}, {"tags": name}]})
            count = len(items)
            warnings = []
            status[name] = {
                "count": count,
                "warnings": warnings
            }
        return status

    def reflect(self, top_k: int = 5):
        insights = {}
        type_names = ["semantic", "episodic", "procedural", "working"]
        for name in type_names:
            items = self._graph.search_nodes({"$or": [{"type": name}, {"tags": name}]})
            keyword_count = {}
            for node in items:
                content = str(node.get("content", ""))
                for w in content.lower().split():
                    keyword_count[w] = keyword_count.get(w, 0) + 1
            sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)
            insights[name] = {
                "top_keywords": sorted_keywords[:top_k],
                "total_items": len(items)
            }
        return insights

    def summarize(self, max_items: int = 10):
        summaries = {}
        type_names = ["semantic", "episodic", "procedural", "working"]
        for name in type_names:
            items = self._graph.search_nodes({"$or": [{"type": name}, {"tags": name}]})
            content_preview = [str(item.get("content", ""))[:100] for item in items[:max_items]]
            summaries[name] = {
                "count": len(items),
                "examples": content_preview
            }
        return summaries

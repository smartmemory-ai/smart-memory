"""
SmartGraph Search Operations Module

Handles all search-related operations for the SmartGraph system.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SmartGraphSearch:
    """Handles all search-related operations for SmartGraph."""

    def __init__(self, backend, nodes_module, enable_caching=True, cache_size=1000):
        self.backend = backend
        self.nodes = nodes_module  # Reference to nodes module for conversion
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._search_cache = {} if enable_caching else None

    def search_nodes(self, query: Dict[str, Any]):
        """
        Search for nodes matching the query dict. Supports operators:
            - $lt, $lte, $gt, $gte for range queries
            - $in for membership
            - $or, $and for logical combinations
        Example:
            {"$or": [{"type": "note"}, {"tags": "note"}], "created_at": {"$lt": "2024-01-01T00:00:00"}}
        """
        nodes = self.backend.search_nodes(query)
        return [self.nodes._from_node_dict(self.nodes.item_cls, n) for n in nodes]

    def search(self, query_str: str, top_k: int = 5, **kwargs):
        """
        Enhanced search method using vector embeddings as primary method with text-based fallbacks.
        Provides semantic similarity search for better relevance.
        """
        # Multi-level fallback strategy - vector search first, then text-based
        fallback_attempts = [
            self._search_with_vector_embeddings,  # Primary vector-based search
            self._search_with_regex,
            self._search_with_simple_contains,
            self._search_with_keyword_matching,
            self._get_all_nodes_fallback
        ]

        for i, fallback_method in enumerate(fallback_attempts):
            try:
                results = fallback_method(query_str, top_k, **kwargs)
                # Check if we got actual results, not just an empty list
                if results is not None and len(results) > 0:
                    if i > 0:  # Log if we had to use fallback
                        logger.info(f"Search succeeded using fallback method {i}: {fallback_method.__name__}")
                    return results
                elif results is not None:
                    logger.debug(f"Fallback method {i} ({fallback_method.__name__}) returned empty results")
            except Exception as e:
                logger.warning(f"Search fallback {i} failed ({fallback_method.__name__}): {e}")
                continue

        # All fallbacks failed
        logger.error(f"All search fallbacks failed for query: {query_str}")
        return []

    def _search_with_vector_embeddings(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using vector embeddings for semantic similarity."""
        try:
            # Import required modules
            from smartmemory.stores.vector.vector_store import VectorStore
            from smartmemory.plugins.embedding import create_embeddings

            # Get vector store instance
            vector_store = VectorStore()

            # Generate embedding for the query
            query_embedding = create_embeddings(query_str)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding")
                return None

            # Convert to list if numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Search vector store for similar embeddings
            vector_results = vector_store.search(query_embedding, top_k=top_k * 2)  # Get more for filtering

            if not vector_results:
                logger.debug(f"No vector results found for query: {query_str}")
                return []

            # Convert vector results to MemoryItems by retrieving from graph
            memory_items = []
            for result in vector_results[:top_k]:
                node_id = result.get('id')
                if node_id:
                    try:
                        # Retrieve the full MemoryItem from the graph
                        item = self.nodes.get_node(node_id)
                        if item:
                            memory_items.append(item)
                            logger.debug(f"Retrieved item {node_id} with score {result.get('score', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Failed to retrieve node {node_id}: {e}")
                        continue

            logger.info(f"Vector search found {len(memory_items)} results for query: {query_str}")
            return memory_items

        except Exception as e:
            logger.warning(f"Vector embedding search failed: {e}")
            return None  # Signal fallback should be used

    def _search_with_regex(self, query_str: str, top_k: int = 5, **kwargs):
        """Primary search method using FalkorDB/Cypher-compatible text search."""
        # Handle wildcard or empty queries
        if query_str in ["*", "", None]:
            # Return all nodes for wildcard queries
            results = self.backend.search_nodes({})
        else:
            # Use FalkorDB's text search capabilities
            results = self._search_text_falkordb(query_str)

        # Convert to MemoryItem objects
        memory_items = [self.nodes._from_node_dict(self.nodes.item_cls, node) for node in results]
        return memory_items[:top_k] if top_k else memory_items

    def _search_text_falkordb(self, query_str: str):
        """FalkorDB-compatible text search using CONTAINS operator with multi-word support."""
        # Split query into individual terms for better matching
        terms = [term.strip() for term in query_str.split() if term.strip()]

        if not terms:
            return []

        # Build dynamic query for multiple terms
        # Use OR logic for individual terms to be more permissive
        where_conditions = []
        params = {}

        for i, term in enumerate(terms):
            param_name = f"term{i}"
            params[param_name] = term

            # Add condition for this term across multiple fields
            term_condition = f"("
            term_condition += f"toLower(n.content) CONTAINS toLower(${param_name}) OR "
            term_condition += f"toLower(n.title) CONTAINS toLower(${param_name}) OR "
            term_condition += f"toLower(n.description) CONTAINS toLower(${param_name})"
            term_condition += f")"

            where_conditions.append(term_condition)

        # Combine conditions with OR for broader matching
        where_clause = " OR ".join(where_conditions)

        cypher = f"""
        MATCH (n) 
        WHERE {where_clause}
        RETURN n
        """

        res = self.backend._query(cypher, params)

        result = []
        for record in res:
            node = record[0]
            if hasattr(node, 'properties'):
                props = dict(node.properties)
            else:
                # Fallback to direct attribute access
                props = {k: v for k, v in vars(node).items()
                         if not k.startswith('_') and k != 'properties'}

            # Remove internal properties
            props.pop('is_global', None)
            result.append(props)

        return result

    def _search_with_simple_contains(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using simple contains logic."""
        if query_str in ["*", "", None]:
            # Get all nodes
            if hasattr(self.backend, 'get_all_nodes'):
                nodes = self.backend.get_all_nodes()
                results = []
                for n in nodes[:top_k]:
                    try:
                        item = self.nodes._from_node_dict(self.nodes.item_cls, n)
                        if item:
                            results.append(item)
                    except Exception as e:
                        logger.warning(f"Failed to convert node to MemoryItem: {e}")
                        continue
                return results
            return []

        # Get all nodes and filter manually
        if hasattr(self.backend, 'get_all_nodes'):
            nodes = self.backend.get_all_nodes()
            filtered_nodes = []
            query_lower = query_str.lower()

            logger.info(f"Searching through {len(nodes)} nodes for query: {query_str}")

            for node in nodes:
                try:
                    # Convert node to MemoryItem
                    node_dict = self.nodes._from_node_dict(self.nodes.item_cls, node)
                    if not node_dict:
                        continue

                    # Extract searchable text from the MemoryItem
                    content = getattr(node_dict, 'content', '') or ''
                    metadata = getattr(node_dict, 'metadata', {}) or {}

                    # Build searchable text
                    searchable_text = str(content)
                    if isinstance(metadata, dict):
                        # Add metadata values to searchable text
                        for key, value in metadata.items():
                            if isinstance(value, str):
                                searchable_text += f" {value}"

                    # Check if query matches (case-insensitive)
                    if query_lower in searchable_text.lower():
                        filtered_nodes.append(node_dict)
                        logger.debug(f"Found match: {getattr(node_dict, 'item_id', 'No ID')}")
                        if len(filtered_nodes) >= top_k:
                            break

                except Exception as e:
                    logger.warning(f"Error processing node during search: {e}")
                    continue

            logger.info(f"Search found {len(filtered_nodes)} matches")
            return filtered_nodes
        return []

    def _search_with_keyword_matching(self, query_str: str, top_k: int = 5, **kwargs):
        """Fallback search using keyword matching."""
        if query_str in ["*", "", None]:
            return None  # Let next fallback handle this

        # Get all nodes and use keyword matching
        if hasattr(self.backend, 'get_all_nodes'):
            nodes = self.backend.get_all_nodes()
            query_words = set(query_str.lower().split())
            scored_nodes = []

            for node in nodes:
                node_dict = self.nodes._from_node_dict(self.nodes.item_cls, node)
                content = getattr(node_dict, 'content', '').lower()
                title = getattr(node_dict, 'metadata', {}).get('title', '').lower()
                description = getattr(node_dict, 'metadata', {}).get('description', '').lower()

                all_text = f"{content} {title} {description}"
                text_words = set(all_text.split())

                if query_words and text_words:
                    intersection = len(query_words & text_words)
                    union = len(query_words | text_words)
                    score = intersection / union if union > 0 else 0.0

                    if score > 0.1:  # Only include reasonable matches
                        scored_nodes.append((node_dict, score))

            # Sort by score and return top_k
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            return [node for node, score in scored_nodes[:top_k]]

        return None

    def _get_all_nodes_fallback(self, query_str: str, top_k: int = 5, **kwargs):
        """Final fallback - just return all available nodes."""
        if hasattr(self.backend, 'get_all_nodes'):
            nodes = self.backend.get_all_nodes()
            return [self.nodes._from_node_dict(self.nodes.item_cls, n) for n in nodes[:top_k]]
        return []

    def _manage_cache_size(self):
        """Manage cache size by removing oldest entries when cache is full."""
        if not self.enable_caching:
            return

        if len(self._search_cache) >= self.cache_size:
            # Remove 20% of oldest entries
            remove_count = max(1, self.cache_size // 5)
            oldest_keys = list(self._search_cache.keys())[:remove_count]
            for key in oldest_keys:
                del self._search_cache[key]

    def clear_cache(self):
        """Clear search cache."""
        if self.enable_caching:
            self._search_cache.clear()

    def get_cache_stats(self):
        """Get search cache performance statistics."""
        if not self.enable_caching:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "search_cache_size": len(self._search_cache),
            "max_cache_size": self.cache_size
        }

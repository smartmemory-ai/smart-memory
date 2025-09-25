from abc import ABC, abstractmethod
from typing import Dict, List, Type, Optional


class VectorBackend(ABC):
    """Backend interface for vector operations."""

    @abstractmethod
    def add(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        ...

    @abstractmethod
    def upsert(self, *, item_id: str, embedding: List[float], metadata: Dict) -> None:
        ...

    @abstractmethod
    def search(self, *, query_embedding: List[float], top_k: int) -> List[Dict]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


# --- Lazy backend registry and factory ---
_BACKENDS: Optional[Dict[str, Type[VectorBackend]]] = None


def _ensure_registry() -> None:
    global _BACKENDS
    if _BACKENDS is not None:
        return
    # Import locally to avoid circular imports at module import time
    from .chroma import ChromaVectorBackend
    from .falkor import FalkorVectorBackend
    _BACKENDS = {
        "chromadb": ChromaVectorBackend,
        "falkordb": FalkorVectorBackend,
    }


def create_backend(name: str, collection_name: str, persist_directory: str | None) -> VectorBackend:
    _ensure_registry()
    key = (name or "falkordb").lower()
    assert _BACKENDS is not None  # for type checkers
    if key not in _BACKENDS:
        raise ValueError(f"Unknown vector backend: {name}")
    cls = _BACKENDS[key]
    return cls(collection_name, persist_directory)

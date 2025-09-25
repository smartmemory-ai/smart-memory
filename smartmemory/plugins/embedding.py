import logging
import numpy as np

from smartmemory.configuration import MemoryConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Abstracts embedding computation for different providers (OpenAI, Ollama, etc).
    """

    def __init__(self, config=None):
        if config is None:
            config = MemoryConfig().vector["embedding"]
        self.provider = config.get('provider', 'openai')
        self.model = config.get('models', 'text-embedding-ada-002')
        self.api_key = config.get("openai_api_key")
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')

    def embed(self, text):
        # Try Redis cache first for significant performance improvement
        try:
            from smartmemory.utils.cache import get_cache
            cache = get_cache()

            # Check cache for existing embedding
            cached_embedding = cache.get_embedding(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return np.array(cached_embedding)

            logger.debug(f"Cache miss for embedding: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Redis cache unavailable for embeddings: {e}")
            cache = None

        # Generate embedding via API
        if self.provider == 'openai':
            if not self.api_key:
                # Generate mock embedding for testing when no API key is available
                logger.warning("No OpenAI API key provided, generating mock embedding for testing")
                # Generate deterministic mock embedding based on text hash
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                # Convert hash to 1536-dimensional vector (OpenAI embedding size)
                mock_embedding = []
                for i in range(1536):
                    # Use hash characters cyclically to generate values between -1 and 1
                    char_val = ord(text_hash[i % len(text_hash)]) / 255.0 * 2 - 1
                    mock_embedding.append(char_val)
                embedding = np.array(mock_embedding)
            else:
                import openai
                openai.api_key = self.api_key
                resp = openai.embeddings.create(input=text, model=self.model)
                embedding = np.array(resp.data[0].embedding)
        elif self.provider == 'ollama':
            import requests
            url = f"{self.ollama_url}/api/embeddings"
            resp = requests.post(url, json={"models": self.model, "prompt": text})
            resp.raise_for_status()
            embedding = np.array(resp.json()['embedding'])
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        # Cache the result for future use
        if cache is not None:
            try:
                cache.set_embedding(text, embedding.tolist())
                logger.debug(f"Cached embedding for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

        return embedding


def create_embeddings(text):
    return EmbeddingService().embed(text)

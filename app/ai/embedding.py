"""
Embedding Service using Google GenAI (gemini-embedding-001).
Produces 1536-dimensional vectors for pgvector storage and cosine similarity search.
"""

import asyncio
import hashlib
import math
from typing import List, Optional
from google import genai
from google.genai import types

from app.core.config import settings


class EmbeddingService:
    """
    Asynchronous embedding generator using Google GenAI SDK.
    Generates 1536-dimensional vectors compatible with the pgvector column.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client: Optional[genai.Client] = None
        self.model_name = "gemini-embedding-001"
        self.dimension = 1536
        self._cache: dict[str, List[float]] = {}
        self._max_cache_size = 1024

    def _get_client(self) -> Optional[genai.Client]:
        if not self.api_key:
            return None
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _fallback_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic 1536-dimensional pseudo-embedding from text.
        Used when API key is unconfigured or during offline/mock test executions.
        """
        # Seeded pseudo-random generation based on SHA-256
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            # Spread across float space and normalize
            val = math.sin((i + 1) * byte_val)
            vec.append(val)

        # L2-normalize vector
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous call to Google GenAI embed_content with fast cache lookup."""
        if text in self._cache:
            return self._cache[text]

        client = self._get_client()
        if not client:
            emb = self._fallback_embedding(text)
            self._put_cache(text, emb)
            return emb

        try:
            resp = client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            if resp.embeddings and len(resp.embeddings) > 0:
                values = resp.embeddings[0].values
                if len(values) == self.dimension:
                    emb = values
                elif len(values) > self.dimension:
                    emb = values[:self.dimension]
                else:
                    emb = values + [0.0] * (self.dimension - len(values))
            else:
                emb = self._fallback_embedding(text)
        except Exception:
            emb = self._fallback_embedding(text)

        self._put_cache(text, emb)
        return emb

    def _put_cache(self, text: str, emb: List[float]):
        if len(self._cache) >= self._max_cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[text] = emb

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Asynchronously generates a 1536-dimensional embedding vector for text.
        Executes via asyncio.to_thread to maintain non-blocking FastAPI loops.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return [0.0] * self.dimension

        if clean_text in self._cache:
            return self._cache[clean_text]

        return await asyncio.to_thread(self._embed_sync, clean_text)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for multiple texts asynchronously."""
        tasks = [self.generate_embedding(t) for t in texts]
        return await asyncio.gather(*tasks)

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two vector lists."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


# Singleton instance
embedding_service = EmbeddingService()

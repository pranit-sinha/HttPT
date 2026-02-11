import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_OPENSEARCH_HOST = "localhost"
DEFAULT_OPENSEARCH_PORT = 9200
DEFAULT_OPENSEARCH_AUTH = ("admin", "admin")

INDEX_NAME = "compliance_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  

class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.dimension = EMBEDDING_DIM

    def _load(self):
        if self.model is None:
            logger.info("Loading embedding model %s …", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model ready (dim=%d)", self.dimension)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vectors = self.model.encode(texts, normalize_embeddings=True) #l2-normalization
        return vectors.tolist()

    def embed(self, text: str) -> List[float]: #convenience - for single string
        return self.embed_batch([text])[0]

class DocumentProcessor:

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunk_id = hashlib.sha256(
                f"{document_id}:{chunk_index}:{chunk_text[:128]}".encode()
            ).hexdigest()[:16] #hashing is deterministic so can ingest previously seen content

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "word_count": end - start,
                    "metadata": metadata or {},
                }
            )

            chunk_index += 1
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start
            # otherwise infinite loop when overlap >= chunk_size

            if end >= len(words):
                break

        return chunks

class OpenSearchStore:
    pass

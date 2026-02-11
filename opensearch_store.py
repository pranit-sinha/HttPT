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
    """
    Hybrid index storing both analysed text (for BM25 retrieval)
    and dense embeddings (for approximate kNN retrieval)
    """

    def __init__(self, host: str = DEFAULT_OPENSEARCH_HOST, port: int = DEFAULT_OPENSEARCH_PORT, auth: tuple = DEFAULT_OPENSEARCH_AUTH, use_ssl: bool = True, verify_certs: bool = False, index_name: str = INDEX_NAME, embedding_dim: int = EMBEDDING_DIM, embedding_model: str = MODEL_NAME):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
        )
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        self.embedder = EmbeddingService(model_name=embedding_model)
        self.chunker = DocumentProcessor()

    def _index_body(self) -> Dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 256,
                },
                "analysis": {
                    "analyzer": {
                        "regulatory_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "snowball",
                                "regulatory_synonyms",
                            ],
                        },
                    },
                    "filter": {
                        "regulatory_synonyms": {
                            "type": "synonym",
                            "synonyms": [
                                "KYC, Know Your Customer, customer identification",
                                "AML, Anti Money Laundering, money laundering prevention",
                                "GDPR, General Data Protection Regulation",
                                "PII, Personally Identifiable Information, personal data",
                                "SOX, Sarbanes Oxley, Sarbanes-Oxley Act",
                                "PCI DSS, Payment Card Industry Data Security Standard",
                                "HIPAA, Health Insurance Portability and Accountability Act",
                                "CCPA, California Consumer Privacy Act",
                            ],
                        },
                    },
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "text": {
                        "type": "text",
                        "analyzer": "regulatory_analyzer",
                        "search_analyzer": "regulatory_analyzer",
                    },
                    "text_raw": {
                        "type": "keyword",
                        "ignore_above": 8191,
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16,
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "title": {"type": "text"},
                            "category": {"type": "keyword"},
                            "effective_date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                            "version": {"type": "keyword"},
                        },
                    },
                    "word_count": {"type": "integer"},
                    "ingested_at": {"type": "date"},
                },
            },
        }

    def create_index(self, recreate: bool = False) -> None:
        exists = self.client.indices.exists(index=self.index_name)

        if exists and recreate:
            self.client.indices.delete(index=self.index_name)
            logger.info("Dropped existing index '%s'", self.index_name)
            exists = False

        if exists:
            logger.info("Index '%s' already exists – skipping creation", self.index_name)
            return

        self.client.indices.create(index=self.index_name, body=self._index_body())
        logger.info("Created hybrid index '%s'", self.index_name)

    def ingest_document(self, text: str, document_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        if chunk_size is not None:
            self.chunker.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunker.chunk_overlap = chunk_overlap

        chunks = self.chunker.chunk(text, document_id=document_id, metadata=metadata)
        if not chunks:
            return {"document_id": document_id, "chunk_count": 0, "errors": []}

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        now = datetime.now(timezone.utc).isoformat()

        actions = []
        for chunk, vector in zip(chunks, embeddings):
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": f"{document_id}_{chunk['chunk_id']}",
                    "_source": {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "text_raw": chunk["text"],  
                        "embedding": vector,
                        "metadata": metadata,
                        "word_count": chunk["word_count"],
                        "ingested_at": now,
                    },
                }
            )

        success, errors = helpers.bulk(
            self.client, actions, raise_on_error=False, refresh="wait_for"
        )

        if errors:
            logger.error("Bulk‑index errors for doc %s: %s", document_id, errors)

        logger.info(
            "Ingested document '%s' → %d chunks (%d indexed successfully)",
            document_id,
            len(chunks),
            success,
        )

        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "indexed": success,
            "errors": errors if errors else [],
        }

    def ingest_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for doc in documents:
            result = self.ingest_document(
                text=doc["text"],
                document_id=doc.get("document_id"),
                metadata=doc.get("metadata"),
            )
            results.append(result)
        return results

    def delete_document(self, document_id: str) -> int:
        body = {"query": {"term": {"document_id": document_id}}}
        resp = self.client.delete_by_query(
            index=self.index_name, body=body, refresh=True
        )
        deleted = resp.get("deleted", 0)
        logger.info("Deleted %d chunks for document '%s'", deleted, document_id)
        return deleted

    def index_stats(self) -> Dict[str, Any]:
        if not self.client.indices.exists(index=self.index_name):
            return {"exists": False}

        stats = self.client.indices.stats(index=self.index_name)
        count = self.client.count(index=self.index_name)

        idx_stats = stats["indices"][self.index_name]["total"]
        return {
            "exists": True,
            "index_name": self.index_name,
            "document_count": count["count"],
            "store_size_bytes": idx_stats["store"]["size_in_bytes"],
        }

    def health_check(self) -> Dict[str, Any]:
        try:
            info = self.client.info()
            health = self.client.cluster.health()
            return {
                "status": "connected",
                "cluster_name": health.get("cluster_name"),
                "cluster_status": health.get("status"),
                "opensearch_version": info.get("version", {}).get("number"),
                "index_exists": self.client.indices.exists(index=self.index_name),
            }
        except Exception as exc:
            logger.exception("OpenSearch health‑check failed")
            return {"status": "disconnected", "error": str(exc)}

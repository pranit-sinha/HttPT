import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from sentence_transformers import CrossEncoder
from opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    retrieval: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CrossEncoderReranker:

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _load(self):
        if self._model is None:
            logger.info("Loading cross-encoder %s …", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder ready")

    def rerank(self, query: str, results: List[SearchResult], top_k: Optional[int] = None) -> List[SearchResult]:
        if not results:
            return []

        self._load()

        pairs = [(query, r.text) for r in results]
        ce_scores = self._model.predict(pairs)

        reranked: List[SearchResult] = []
        for result, ce_score in zip(results, ce_scores):
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    score=float(ce_score),
                    chunk_index=result.chunk_index,
                    metadata=result.metadata.copy() if result.metadata else {},
                    word_count=result.word_count,
                    retrieval={
                        **result.retrieval,
                        "reranker_score": float(ce_score),
                        "pre_rerank_score": result.score,
                    },
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked

class HybridSearcher:

    def __init__(self, store: OpenSearchStore, reranker: Optional[CrossEncoderReranker] = None, enable_reranker: bool = True):
        self.store = store
        if reranker is not None:
            self.reranker = reranker
        elif enable_reranker:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

    @staticmethod
    def _build_metadata_filters(metadata_filter: Optional[Dict[str, Any]]) -> Optional[List[Dict]]:
        """``{"category": "AML"}`` → ``[{"term": {"metadata.category": "AML"}}]``"""
        if not metadata_filter:
            return None
        return [{"term": {f"metadata.{k}": v}} for k, v in metadata_filter.items()]

    def build_bm25_query(self, query_text: str, top_k: int = 20, filters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        should = [
            {
                "match": {
                    "text": {
                        "query": query_text,
                        "analyzer": "regulatory_analyzer",
                    }
                }
            },
            {
                "match_phrase": {
                    "text": {
                        "query": query_text,
                        "slop": 2,
                        "boost": 2.0,
                    }
                }
            },
        ]

        bool_body: Dict[str, Any] = {
            "should": should,
            "minimum_should_match": 1,
        }
        if filters:
            bool_body["filter"] = filters

        return {
            "size": top_k,
            "query": {"bool": bool_body},
            "_source": {"excludes": ["embedding"]},
        }

    def build_knn_query(self, query_vector: List[float], top_k: int = 20, filters: Optional[List[Dict]] = None) -> Dict[str, Any]:
        knn_clause: Dict[str, Any] = {
            "vector": query_vector,
            "k": top_k,
        }
        if filters:
            knn_clause["filter"] = {"bool": {"must": filters}}

        return {
            "size": top_k,
            "query": {"knn": {"embedding": knn_clause}},
            "_source": {"excludes": ["embedding"]},
        }

    @staticmethod
    def _parse_hits(response: Dict[str, Any]) -> List[SearchResult]:
        results: List[SearchResult] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            results.append(
                SearchResult(
                    chunk_id=src.get("chunk_id", hit["_id"]),
                    document_id=src.get("document_id", ""),
                    text=src.get("text", ""),
                    score=float(hit.get("_score") or 0.0),
                    chunk_index=src.get("chunk_index", 0),
                    metadata=src.get("metadata", {}),
                    word_count=src.get("word_count", 0),
                )
            )
        return results

    def bm25_search(self, query: str, top_k: int = 20, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        filters = self._build_metadata_filters(metadata_filter)
        body = self.build_bm25_query(query, top_k, filters)
        response = self.store.client.search(index=self.store.index_name, body=body)
        results = self._parse_hits(response)

        for rank_0, r in enumerate(results):
            r.retrieval["bm25_rank"] = rank_0 + 1
            r.retrieval["bm25_score"] = r.score
        return results

    def knn_search(self, query: str, top_k: int = 20, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        query_vector = self.store.embedder.embed(query)
        filters = self._build_metadata_filters(metadata_filter)
        body = self.build_knn_query(query_vector, top_k, filters)
        response = self.store.client.search(index=self.store.index_name, body=body)
        results = self._parse_hits(response)

        for rank_0, r in enumerate(results):
            r.retrieval["knn_rank"] = rank_0 + 1
            r.retrieval["knn_score"] = r.score
        return results

    @staticmethod
    def reciprocal_rank_fusion(result_lists: List[List[SearchResult]], list_names: Optional[List[str]] = None, k: int = 60) -> List[SearchResult]:
        if list_names is None:
            list_names = [f"list_{i}" for i in range(len(result_lists))]

        rrf_scores: Dict[str, float] = {}
        canonical: Dict[str, SearchResult] = {}
        rank_info: Dict[str, Dict[str, Any]] = {}

        for name, result_list in zip(list_names, result_lists):
            for rank_0, result in enumerate(result_list):
                rank = rank_0 + 1
                cid = result.chunk_id

                rrf_scores.setdefault(cid, 0.0)
                rrf_scores[cid] += 1.0 / (k + rank)

                if cid not in canonical:
                    canonical[cid] = SearchResult(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        text=result.text,
                        score=0.0,
                        chunk_index=result.chunk_index,
                        metadata=result.metadata.copy() if result.metadata else {},
                        word_count=result.word_count,
                        retrieval={},
                    )
                    rank_info[cid] = {}

                rank_info[cid][f"{name}_rank"] = rank
                rank_info[cid][f"{name}_score"] = result.score

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        merged: List[SearchResult] = []
        for cid in sorted_ids:
            doc = canonical[cid]
            doc.score = rrf_scores[cid]
            doc.retrieval = {"rrf_score": rrf_scores[cid], **rank_info[cid]}
            merged.append(doc)
        return merged

    def search(self, query: str, bm25_top_k: int = 20, knn_top_k: int = 20, rrf_k: int = 60, final_top_k: int = 10, use_reranker: bool = True, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:

        if not query or not query.strip():
            return []

        bm25_results = self.bm25_search(query, bm25_top_k, metadata_filter)
        knn_results = self.knn_search(query, knn_top_k, metadata_filter)

        logger.info(
            "Hybrid retrieval: %d BM25 hits, %d kNN hits",
            len(bm25_results),
            len(knn_results),
        )

        if not bm25_results and not knn_results:
            return []

        fused = self.reciprocal_rank_fusion(
            [bm25_results, knn_results],
            list_names=["bm25", "knn"],
            k=rrf_k,
        )

        if use_reranker and self.reranker and fused:
            return self.reranker.rerank(query, fused, top_k=final_top_k)

        return fused[:final_top_k]

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from opensearch_store import OpenSearchStore
from hybrid_search import HybridSearcher, CrossEncoderReranker, SearchResult
from gerri_kellman_agent import ComplianceAuditor
from llm_provide import LLMManager

logger = logging.getLogger(__name__)

class DocumentIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: Optional[int] = Field(None, ge=64, le=2048)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=512)

class BatchIngestRequest(BaseModel):
    documents: List[DocumentIngestRequest] = Field(..., min_length=1, max_length=50)

class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    indexed: int
    errors: list

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    use_reranker: bool = True
    metadata_filter: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total: int

class AuditRequest(BaseModel):
    query: str = Field(..., min_length=1)
    organization_context: str = ""
    metadata_filter: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

class AuditResponse(BaseModel):
    status: str
    report: str
    findings: list
    risk_summary: dict
    requirements: list
    retrieved_document_count: int
    error: str = ""

class BatchAuditRequest(BaseModel):
    audits: List[AuditRequest] = Field(..., min_length=1, max_length=10)

class BatchAuditResponse(BaseModel):
    results: List[AuditResponse]
    total: int
    succeeded: int
    failed: int

class AuditBatchProcessor:

    def __init__(self, max_concurrent: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self.total_processed = 0

    async def _run_one(self, auditor: ComplianceAuditor, request: AuditRequest) -> AuditResponse:
        async with self.semaphore:
            self.active_count += 1
            try:
                result = await auditor.run(
                    query=request.query,
                    organization_context=request.organization_context,
                    metadata_filter=request.metadata_filter,
                    config=request.config,
                )
                return AuditResponse(
                    status=result.get("status", "unknown"),
                    report=result.get("report", ""),
                    findings=result.get("findings", []),
                    risk_summary=result.get("risk_summary", {}),
                    requirements=result.get("requirements", []),
                    retrieved_document_count=len(
                        result.get("retrieved_documents", [])
                    ),
                    error=result.get("error", ""),
                )
            except Exception as exc:
                logger.exception("Audit failed for query: %s", request.query)
                return AuditResponse(
                    status="error", report="", findings=[], risk_summary={},
                    requirements=[], retrieved_document_count=0, error=str(exc),
                )
            finally:
                self.active_count -= 1
                self.total_processed += 1

    async def run_batch(self, auditor: ComplianceAuditor, requests: List[AuditRequest]) -> BatchAuditResponse:
        tasks = [self._run_one(auditor, req) for req in requests]
        results = list(await asyncio.gather(*tasks))
        succeeded = sum(1 for r in results if r.status == "complete")
        return BatchAuditResponse(
            results=results,
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )

class ComplianceService:

    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        opensearch_auth: tuple = ("admin", "admin"),
        opensearch_ssl: bool = True,
        opensearch_verify_certs: bool = False,
        index_name: str = "compliance_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_reranker: bool = True,
        max_concurrent_audits: int = 4,
    ):
        self.store = OpenSearchStore(
            host=opensearch_host,
            port=opensearch_port,
            auth=opensearch_auth,
            use_ssl=opensearch_ssl,
            verify_certs=opensearch_verify_certs,
            index_name=index_name,
            embedding_model=embedding_model,
        )
        self.searcher = HybridSearcher(store=self.store, enable_reranker=enable_reranker)
        self.auditor: Optional[ComplianceAuditor] = None
        self.batch_processor = AuditBatchProcessor(max_concurrent=max_concurrent_audits)
        self._ready = False

    def bootstrap(self, llm_manager: LLMManager):
        try:
            self.store.create_index(recreate=False)
            self.auditor = ComplianceAuditor(
                searcher=self.searcher, llm_manager=llm_manager,
            )
            self._ready = True
            logger.info("Compliance service ready")
        except Exception:
            logger.exception(
                "Compliance service bootstrap failed — "
                "endpoints will return 503"
            )
            self._ready = False

    def shutdown(self):
        self._ready = False
        logger.info("Compliance service shut down")

    @property
    def ready(self) -> bool:
        return self._ready and self.auditor is not None

#to be set by main during lifespan
_service: Optional[ComplianceService] = None

def set_service(service: Optional[ComplianceService]):
    global _service
    _service = service

def _svc() -> ComplianceService:
    if _service is None or not _service.ready:
        raise HTTPException(
            status_code=503, detail="Compliance service not available",
        )
    return _service

router = APIRouter(prefix="/compliance", tags=["compliance"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    svc = _svc()
    try:
        result = svc.store.ingest_document(
            text=request.text,
            document_id=request.document_id,
            metadata=request.metadata,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.post("/ingest/batch", response_model=List[IngestResponse])
async def ingest_batch(request: BatchIngestRequest):
    svc = _svc()
    try:
        docs = [
            {"text": d.text, "document_id": d.document_id, "metadata": d.metadata}
            for d in request.documents
        ]
        results = svc.store.ingest_batch(docs)
        return [IngestResponse(**r) for r in results]
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Batch ingestion failed: {exc}",
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    svc = _svc()
    try:
        results = svc.searcher.search(
            query=request.query,
            final_top_k=request.top_k,
            use_reranker=request.use_reranker,
            metadata_filter=request.metadata_filter,
        )
        return SearchResponse(
            query=request.query,
            results=[r.to_dict() for r in results],
            total=len(results),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

def _audit_state_to_response(state: dict) -> AuditResponse:
    return AuditResponse(
        status=state.get("status", "unknown"),
        report=state.get("report", ""),
        findings=state.get("findings", []),
        risk_summary=state.get("risk_summary", {}),
        requirements=state.get("requirements", []),
        retrieved_document_count=len(state.get("retrieved_documents", [])),
        error=state.get("error", ""),
    )

@router.post("/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    svc = _svc()
    try:
        result = await svc.auditor.run(
            query=request.query,
            organization_context=request.organization_context,
            metadata_filter=request.metadata_filter,
            config=request.config,
        )
        return _audit_state_to_response(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit failed: {exc}")

@router.post("/audit/stream")
async def stream_audit(request: AuditRequest):
    svc = _svc()

    async def event_generator():
        try:
            async for step in svc.auditor.run_stream(
                query=request.query,
                organization_context=request.organization_context,
                metadata_filter=request.metadata_filter,
                config=request.config,
            ):
                node = step.get("node", "")
                data = step.get("data", {})

                payload: Dict[str, Any] = {
                    "node": node,
                    "status": step.get("status", ""),
                    "error": step.get("error", ""),
                }

                if node == "retrieve":
                    docs = data.get("retrieved_documents", [])
                    payload["retrieved_document_count"] = len(docs)
                    payload["preview"] = [
                        {
                            "chunk_id": d.get("chunk_id"),
                            "text": d.get("text", "")[:120],
                        }
                        for d in docs[:3]
                    ]
                elif node == "analyze":
                    payload["requirements"] = data.get("requirements", [])
                elif node == "audit":
                    payload["findings"] = data.get("findings", [])
                elif node == "report":
                    payload["report"] = data.get("report", "")
                    payload["risk_summary"] = data.get("risk_summary", {})

                yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
    )

@router.post("/audit/batch", response_model=BatchAuditResponse)
async def batch_audit(request: BatchAuditRequest):
    svc = _svc()
    try:
        return await svc.batch_processor.run_batch(svc.auditor, request.audits)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Batch audit failed: {exc}",
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    svc = _svc()
    try:
        deleted = svc.store.delete_document(document_id)
        return {"document_id": document_id, "deleted_chunks": deleted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {exc}")

@router.get("/health")
async def compliance_health():
    svc = _svc()
    return svc.store.health_check()

@router.get("/stats")
async def compliance_stats():
    svc = _svc()
    stats = svc.store.index_stats()
    stats["batch_processor"] = {
        "max_concurrent": svc.batch_processor.max_concurrent,
        "active_audits": svc.batch_processor.active_count,
        "total_processed": svc.batch_processor.total_processed,
    }
    return stats

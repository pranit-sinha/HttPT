import json
import logging
from typing import TypedDict, List, Dict, Any, Optional, AsyncGenerator
from langgraph.graph import StateGraph, END
from hybrid_search import HybridSearcher, SearchResult
from llm_provide import LLMManager

logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = """\
You are a regulatory compliance analyst. You receive retrieved regulatory document passages and an audit query.

Extract every concrete regulatory requirement that is relevant to the query.

Return ONLY a JSON array (no markdown fencing, no preamble) where each element
has these keys:
  - "requirement_id"       : short id, e.g. "REQ-1"
  - "text"                 : the requirement verbatim or closely paraphrased
  - "source_chunk_ids"     : list of chunk_ids this was derived from
  - "category"             : "mandatory" | "recommended" | "informational"
  - "regulatory_reference" : article / section numbers if mentioned
"""

AUDIT_SYSTEM_PROMPT = """\
You are a compliance auditor. You receive:
  1. A JSON list of regulatory requirements.
  2. A description of the organisation's current practices.

For each requirement, determine compliance status and produce a finding.

Return ONLY a JSON array (no markdown fencing, no preamble) where each element
has these keys:
  - "requirement_id"  : matches the requirement
  - "status"          : "compliant" | "non_compliant" | "partially_compliant" | "insufficient_info"
  - "severity"        : "critical" | "high" | "medium" | "low" | "info"
  - "finding"         : brief explanation of why the status was assigned
  - "evidence"        : what from the organisation context supports the assessment
  - "recommendation"  : specific remediation action (if non/partially compliant)
"""

REPORT_SYSTEM_PROMPT = """\
You are a compliance reporting specialist. Given audit findings (JSON), the original query, organisation context, and risk summary, produce a professional compliance audit report in **Markdown**.

The report MUST include these sections:
1. **Executive Summary** – overall posture, headline stats
2. **Scope** – what was audited and against which regulations
3. **Findings** – each finding with status, severity, evidence, recommendation
4. **Risk Summary** – counts by severity and status
5. **Recommendations** – prioritised action items

Be precise and reference specific requirement IDs.
"""

class AuditState(TypedDict):
    query: str
    organization_context: str
    metadata_filter: dict
    config: dict

    retrieved_documents: list   
    requirements: list          
    findings: list              
    report: str
    risk_summary: dict

    status: str
    error: str

class ComplianceAuditor:

    DEFAULT_CONFIG: Dict[str, Any] = {
        "bm25_top_k": 20,
        "knn_top_k": 20,
        "rrf_k": 60,
        "final_top_k": 10,
        "use_reranker": True,
        "llm_model": None,
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    def __init__(self, searcher: HybridSearcher, llm_manager: LLMManager):
        self.searcher = searcher
        self.llm_manager = llm_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AuditState)

        builder.add_node("retrieve", self.retrieve)
        builder.add_node("analyze", self.analyze)
        builder.add_node("audit", self.audit)
        builder.add_node("report", self.report_node)

        builder.set_entry_point("retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
        )
        builder.add_edge("analyze", "audit")
        builder.add_edge("audit", "report")
        builder.add_edge("report", END)

        return builder.compile()

    @staticmethod
    def _route_after_retrieve(state: AuditState) -> str:
        if state.get("error") or not state.get("retrieved_documents"):
            return END
        return "analyze"

    def _cfg(self, state: AuditState) -> dict:
        merged = {**self.DEFAULT_CONFIG}
        merged.update(state.get("config") or {})
        return merged

    async def _llm_call(
        self, system: str, user: str, cfg: dict,
    ) -> str:
        provider = self.llm_manager.get_provider(cfg.get("llm_model"))
        if provider is None:
            raise RuntimeError("No LLM provider registered")

        result = await provider.generate(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 4000),
            stream=False,
        )
        return result["content"]

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Best-effort extraction of a JSON array/object from LLM output."""
        cleaned = text.strip()

        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                block = parts[1]
                first_nl = block.find("\n") # cause there's an optional language tag on the first line
                if first_nl != -1:
                    block = block[first_nl + 1:]
                cleaned = block.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for open_ch, close_ch in ("[", "]"), ("{", "}"):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue

        raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)

    @staticmethod
    def _compute_risk_summary(findings: List[dict]) -> dict:
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "unknown")
            st = f.get("status", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_status[st] = by_status.get(st, 0) + 1

        total = len(findings)
        compliant = by_status.get("compliant", 0)

        return {
            "total_findings": total,
            "by_severity": by_severity,
            "by_status": by_status,
            "compliance_rate": round(compliant / total * 100, 1) if total else 0.0,
            "critical_issues": by_severity.get("critical", 0),
            "high_issues": by_severity.get("high", 0),
        }

    async def retrieve(self, state: AuditState) -> dict:
        query = state["query"]
        cfg = self._cfg(state)

        try:
            results: List[SearchResult] = self.searcher.search(
                query=query,
                bm25_top_k=cfg["bm25_top_k"],
                knn_top_k=cfg["knn_top_k"],
                rrf_k=cfg["rrf_k"],
                final_top_k=cfg["final_top_k"],
                use_reranker=cfg["use_reranker"],
                metadata_filter=state.get("metadata_filter") or None,
            )
            docs = [r.to_dict() for r in results]

            if not docs:
                logger.warning("No documents found for query: %s", query)
                return {
                    "retrieved_documents": [],
                    "status": "no_documents",
                    "error": f"No regulatory documents found for: {query}",
                }

            logger.info("Retrieved %d documents for audit", len(docs))
            return {"retrieved_documents": docs, "status": "retrieved"}

        except Exception as exc:
            logger.exception("Retrieval failed")
            return {
                "retrieved_documents": [],
                "status": "retrieval_error",
                "error": str(exc),
            }

    async def analyze(self, state: AuditState) -> dict:
        cfg = self._cfg(state)
        docs = state["retrieved_documents"]

        passages = "\n\n".join(
            f"[chunk_id={d['chunk_id']}, doc={d['document_id']}]\n{d['text']}"
            for d in docs
        )
        user_prompt = (
            f"Audit Query: {state['query']}\n\n"
            f"Retrieved Regulatory Passages ({len(docs)}):\n\n{passages}"
        )

        try:
            raw = await self._llm_call(ANALYZE_SYSTEM_PROMPT, user_prompt, cfg)
            requirements = self._parse_json(raw)
            logger.info("Extracted %d requirements", len(requirements))
            return {"requirements": requirements, "status": "analyzed"}

        except json.JSONDecodeError:
            logger.error("LLM returned unparseable JSON in analyze step")
            return {
                "requirements": [],
                "status": "analyze_parse_error",
                "error": "Requirements extraction produced invalid JSON",
            }
        except Exception as exc:
            logger.exception("Analyze step failed")
            return {
                "requirements": [],
                "status": "analyze_error",
                "error": str(exc),
            }

    async def audit(self, state: AuditState) -> dict:
        cfg = self._cfg(state)
        requirements = state.get("requirements", [])
        org_ctx = state.get("organization_context", "")

        if not requirements:
            return {
                "findings": [],
                "status": "skipped_audit",
                "error": "No requirements extracted — nothing to audit",
            }

        user_prompt = (
            f"Regulatory Requirements:\n{json.dumps(requirements, indent=2)}\n\n"
            f"Organisation Practices:\n"
            f"{org_ctx or 'No specific context provided — assess against general best practice.'}"
        )

        try:
            raw = await self._llm_call(AUDIT_SYSTEM_PROMPT, user_prompt, cfg)
            findings = self._parse_json(raw)
            logger.info("Produced %d audit findings", len(findings))
            return {"findings": findings, "status": "audited"}

        except json.JSONDecodeError:
            logger.error("LLM returned unparseable JSON in audit step")
            return {
                "findings": [],
                "status": "audit_parse_error",
                "error": "Audit findings produced invalid JSON",
            }
        except Exception as exc:
            logger.exception("Audit step failed")
            return {
                "findings": [],
                "status": "audit_error",
                "error": str(exc),
            }

    async def report_node(self, state: AuditState) -> dict:
        cfg = self._cfg(state)
        findings = state.get("findings", [])
        risk_summary = self._compute_risk_summary(findings)

        user_prompt = (
            f"Audit Query: {state['query']}\n\n"
            f"Organisation Context:\n{state.get('organization_context') or 'Not provided'}\n\n"
            f"Findings:\n{json.dumps(findings, indent=2)}\n\n"
            f"Risk Summary:\n{json.dumps(risk_summary, indent=2)}"
        )

        try:
            report_text = await self._llm_call(
                REPORT_SYSTEM_PROMPT, user_prompt, cfg,
            )
            return {
                "report": report_text,
                "risk_summary": risk_summary,
                "status": "complete",
            }
        except Exception as exc:
            logger.exception("Report generation failed")
            return {
                "report": "",
                "risk_summary": risk_summary,
                "status": "report_error",
                "error": str(exc),
            }

    async def run(self, query: str, organization_context: str = "", metadata_filter: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> AuditState:
        initial: AuditState = {
            "query": query,
            "organization_context": organization_context,
            "metadata_filter": metadata_filter or {},
            "config": config or {},
            "retrieved_documents": [],
            "requirements": [],
            "findings": [],
            "report": "",
            "risk_summary": {},
            "status": "started",
            "error": "",
        }
        return await self.graph.ainvoke(initial)

    async def run_stream(self, query: str, organization_context: str = "", metadata_filter: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        initial: AuditState = {
            "query": query,
            "organization_context": organization_context,
            "metadata_filter": metadata_filter or {},
            "config": config or {},
            "retrieved_documents": [],
            "requirements": [],
            "findings": [],
            "report": "",
            "risk_summary": {},
            "status": "started",
            "error": "",
        }
        async for step in self.graph.astream(initial, stream_mode="updates"):
            for node_name, update in step.items():
                yield {
                    "node": node_name,
                    "status": update.get("status", ""),
                    "error": update.get("error", ""),
                    "data": update,
                }

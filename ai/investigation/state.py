"""
investigation/state.py
=======================
Shared LangGraph state for the AML Investigation workflow.

Design Rules
------------
• Every field is typed explicitly — LangGraph serialises state to JSON for
  checkpointing, so all values must be JSON-compatible or wrapped in a
  custom serialiser (handled by SqliteSaver).

• Fields annotated with `Annotated[list, operator.add]` support parallel
  fan-out: when two nodes run concurrently and both append to the same list,
  LangGraph merges them automatically via operator.add rather than clobbering.

• `iteration_count` tracks revision cycles to prevent infinite loops.

• `guardrail_violations` accumulates any NeMo Guardrails rejections so the
  final report can document which legal citations were blocked.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


class NeighbourNode(TypedDict):
    """A single node returned from the Neo4j 2-hop neighbourhood query."""
    node_id: str
    node_type: str          # Customer | Account | Transaction
    relationship: str       # Transfer | Shared_IP | Shared_Phone
    properties: Dict[str, Any]
    hop: int                # 1 or 2


class RetrievedChunk(TypedDict):
    """A single chunk returned from Milvus vector search."""
    doc_id: str
    source_type: str        # "kyc_document" | "negative_news" | "sar_case"
    content: str
    score: float            # cosine similarity score
    metadata: Dict[str, Any]


class SARReport(TypedDict):
    """Bilingual SAR report structure."""
    report_id: str
    node_id: str
    risk_score: float
    typologies: List[str]
    narrative_en: str       # English SAR narrative
    narrative_fr: str       # French SAR narrative (regulatory requirement)
    evidence_refs: List[str]
    analyst_notes: str
    status: str             # DRAFT | PENDING_REVIEW | FILED


class AMLState(TypedDict):
    """
    Central state object flowing through all LangGraph agent nodes.

    Lifecycle:
      START
        → fetch_context   (populates neighbourhood, graph_summary)
        → rag_search      (populates kyc_chunks, news_chunks)
        → analyze         (populates risk_verdict, risk_rationale)
        → [conditional]
            ├─ report     (if risk_verified → populates sar_report)
            └─ END        (if risk_not_verified → closes case)
    """

    # ── Input (set once at invocation) ──────────────────────────────── #
    node_id: str                        # Account node ID from GNN
    risk_score: float                   # GNN suspicion probability (0–1)
    typology_scores: Dict[str, float]   # {smurfing: 0.7, structuring: 0.4, ...}
    thread_id: str                      # LangGraph checkpoint thread ID

    # ── Node: fetch_context ──────────────────────────────────────────── #
    neighbourhood: Annotated[List[NeighbourNode], operator.add]
    graph_summary: str                  # Short LLM summary of the subgraph

    # ── Node: rag_search ─────────────────────────────────────────────── #
    kyc_chunks: Annotated[List[RetrievedChunk], operator.add]
    news_chunks: Annotated[List[RetrievedChunk], operator.add]
    similar_sars: Annotated[List[RetrievedChunk], operator.add]

    # ── Node: analyze ─────────────────────────────────────────────────── #
    risk_verdict: Optional[str]         # "VERIFIED" | "DISMISSED" | "ESCALATE"
    risk_rationale: Optional[str]       # LLM reasoning string
    risk_level: Optional[str]           # LOW | MEDIUM | HIGH | CRITICAL
    rule_hits: Annotated[List[str], operator.add]   # FATF / FinCEN rule refs

    # ── Node: report ──────────────────────────────────────────────────── #
    sar_report: Optional[SARReport]

    # ── Control flow ─────────────────────────────────────────────────── #
    iteration_count: int
    max_iterations: int                 # Ceiling for revision loops
    human_feedback: Optional[str]       # Injected on HITL resume

    # ── Safety & Audit ───────────────────────────────────────────────── #
    guardrail_violations: Annotated[List[str], operator.add]
    messages: Annotated[List[Dict[str, str]], operator.add]  # agent message log


def initial_state(
    node_id: str,
    risk_score: float,
    typology_scores: Dict[str, float],
    thread_id: str,
    max_iterations: int = 3,
) -> AMLState:
    """
    Factory: create a clean AMLState for a new investigation.
    Avoids mutable default arguments in TypedDict.
    """
    return AMLState(
        node_id=node_id,
        risk_score=risk_score,
        typology_scores=typology_scores,
        thread_id=thread_id,
        neighbourhood=[],
        graph_summary="",
        kyc_chunks=[],
        news_chunks=[],
        similar_sars=[],
        risk_verdict=None,
        risk_rationale=None,
        risk_level=None,
        rule_hits=[],
        sar_report=None,
        iteration_count=0,
        max_iterations=max_iterations,
        human_feedback=None,
        guardrail_violations=[],
        messages=[],
    )

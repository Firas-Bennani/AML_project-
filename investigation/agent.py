"""
investigation/agent.py
========================
AMLInvestigatorAgent — LangGraph stateful multi-agent orchestrator.

Graph Topology
--------------
                        ┌─────────────────────┐
         START ─────────► fetch_context         │
                        │  (Neo4j 2-hop query)  │
                        └────────┬────────────┘
                                 │
                        ┌────────▼────────────┐
                        │ rag_search           │
                        │ (Milvus KYC + News)  │
                        └────────┬────────────┘
                                 │
                        ┌────────▼────────────┐
                        │ analyze              │
                        │ (NIM LLM + Rails)    │
                        └────────┬────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │  route_after_analyze       │  ← conditional edge
                   └──┬──────────┬─────────────┘
                      │          │           │
                  VERIFIED   ESCALATE    DISMISSED
                      │          │           │
              ┌───────▼──┐   ┌───▼──┐     END
              │  report  │   │ HITL │
              │(SAR gen) │   │interrupt
              └───────┬──┘   └───┬──┘
                      │          │ (human resumes)
                      └────►  END

Checkpointing
-------------
All intermediate state is persisted via SqliteSaver after each node.
This means:
  • The agent can be interrupted after analyze for human review.
  • If the LLM API fails mid-flight, the run can be resumed from the
    last completed node rather than restarting from scratch.
  • Audit trail is maintained in the SQLite checkpoint DB.

Human-in-the-Loop
-----------------
If risk_verdict == "ESCALATE", the graph pauses at the `human_review`
interrupt. The compliance officer can:
  a) Approve → resume(), which routes to report generation.
  b) Dismiss → resume() with feedback="DISMISS", routes to END.
  c) Add notes → human_feedback is injected into state, report uses it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.graph import CompiledGraph

from investigation.state import AMLState, initial_state
from investigation.nodes.fetch_context import FetchContextNode
from investigation.nodes.rag_search import RAGSearchNode
from investigation.nodes.analyze import AnalyzeNode
from investigation.nodes.report import ReportNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_analyze(state: AMLState) -> str:
    """
    Conditional edge called after the 'analyze' node.

    Decision table:
      VERIFIED  → report   (SAR generation)
      ESCALATE  → human_review (HITL interrupt — analyst decides)
      DISMISSED → END      (close case, no SAR)
      fallback  → report   (fail-safe: generate SAR if uncertain)
    """
    verdict = state.get("risk_verdict", "ESCALATE")
    iterations = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if iterations >= max_iter:
        logger.warning(
            "[router] Max iterations (%d) reached for node=%s — routing to report.",
            max_iter, state["node_id"],
        )
        return "report"

    routes = {
        "VERIFIED":  "report",
        "ESCALATE":  "human_review",
        "DISMISSED": END,
    }
    route = routes.get(verdict, "report")
    logger.info("[router] Verdict=%s → routing to '%s'", verdict, route)
    return route


def route_after_human_review(state: AMLState) -> str:
    """
    Conditional edge after human review node.
    Human feedback is read from state.human_feedback.
      "APPROVE"  or "VERIFIED" → report
      "DISMISS"               → END
      anything else           → analyze  (re-run with feedback injected)
    """
    feedback = (state.get("human_feedback") or "").upper().strip()
    if "DISMISS" in feedback:
        return END
    if "APPROVE" in feedback or "VERIFIED" in feedback:
        return "report"
    # Default: re-analyze with human feedback in context
    return "analyze"


# ---------------------------------------------------------------------------
# Passthrough human_review node
# ---------------------------------------------------------------------------

async def human_review_node(state: AMLState) -> Dict[str, Any]:
    """
    Interrupt node — LangGraph pauses here when `interrupt_before` is set.
    In streaming / async mode, the graph yields control back to the caller.
    The analyst uses AMLInvestigatorAgent.resume() to inject feedback and continue.
    """
    logger.info(
        "[human_review] Investigation of node=%s paused for analyst review. "
        "Risk Level=%s | Violations=%d",
        state["node_id"],
        state.get("risk_level", "UNKNOWN"),
        len(state.get("guardrail_violations", [])),
    )
    # State is unchanged — the caller injects human_feedback via resume()
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"[human_review] Awaiting analyst decision for {state['node_id']}. "
                    f"Risk={state.get('risk_level')} | Verdict={state.get('risk_verdict')}"
                ),
            }
        ]
    }


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------

class AMLInvestigatorAgent:
    """
    Stateful LangGraph orchestrator for AML case investigation.

    Instantiate once and call `investigate()` per flagged node.
    The compiled graph is reused across all investigations.

    Args:
        fetch_node    : FetchContextNode  (Neo4j)
        rag_node      : RAGSearchNode     (Milvus)
        analyze_node  : AnalyzeNode       (LLM + Guardrails)
        report_node   : ReportNode        (LLM + Guardrails, bilingual)
        checkpoint_db : Path to SQLite checkpoint file
        max_iterations: Max analyze→rag revision cycles before forcing report
    """

    def __init__(
        self,
        fetch_node: FetchContextNode,
        rag_node: RAGSearchNode,
        analyze_node: AnalyzeNode,
        report_node: ReportNode,
        checkpoint_db: str = "aml_checkpoints.db",
        max_iterations: int = 3,
    ) -> None:
        self.fetch_node = fetch_node
        self.rag_node = rag_node
        self.analyze_node = analyze_node
        self.report_node = report_node
        self.max_iterations = max_iterations
        self._checkpointer = SqliteSaver.from_conn_string(checkpoint_db)
        self._graph: CompiledGraph = self._build_graph()
        logger.info("AMLInvestigatorAgent ready. Checkpoint DB: %s", checkpoint_db)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> CompiledGraph:
        """
        Compile the LangGraph StateGraph.

        Node registration order does not matter — edges define execution order.
        """
        builder = StateGraph(AMLState)

        # Register nodes
        builder.add_node("fetch_context",  self.fetch_node)
        builder.add_node("rag_search",     self.rag_node)
        builder.add_node("analyze",        self.analyze_node)
        builder.add_node("human_review",   human_review_node)
        builder.add_node("report",         self.report_node)

        # Linear edges
        builder.set_entry_point("fetch_context")
        builder.add_edge("fetch_context", "rag_search")
        builder.add_edge("rag_search",    "analyze")

        # Conditional edge after analysis
        builder.add_conditional_edges(
            "analyze",
            route_after_analyze,
            {
                "report":       "report",
                "human_review": "human_review",
                END:            END,
            },
        )

        # Conditional edge after human review
        builder.add_conditional_edges(
            "human_review",
            route_after_human_review,
            {
                "report":  "report",
                "analyze": "analyze",
                END:       END,
            },
        )

        builder.add_edge("report", END)

        return builder.compile(
            checkpointer=self._checkpointer,
            # Pause BEFORE human_review so the graph can be inspected/resumed
            interrupt_before=["human_review"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def investigate(
        self,
        node_id: str,
        risk_score: float,
        typology_scores: Dict[str, float],
        thread_id: Optional[str] = None,
    ) -> AMLState:
        """
        Start a new AML investigation.

        Args:
            node_id          : Account node ID from the GNN detector
            risk_score       : Overall suspicion probability [0-1]
            typology_scores  : {smurfing: 0.7, structuring: 0.4, layering: 0.6}
            thread_id        : Optional identifier for checkpointing.
                               If None, a new UUID is generated.

        Returns:
            Final AMLState (includes sar_report if risk was verified)
        """
        import uuid
        thread_id = thread_id or f"aml-{uuid.uuid4().hex[:8]}"

        state = initial_state(
            node_id=node_id,
            risk_score=risk_score,
            typology_scores=typology_scores,
            thread_id=thread_id,
            max_iterations=self.max_iterations,
        )

        config = {"configurable": {"thread_id": thread_id}}

        logger.info(
            "Starting investigation | node=%s | score=%.3f | thread=%s",
            node_id, risk_score, thread_id,
        )

        final_state = await self._graph.ainvoke(state, config=config)
        logger.info(
            "Investigation complete | node=%s | verdict=%s | thread=%s",
            node_id, final_state.get("risk_verdict"), thread_id,
        )
        return final_state

    async def resume(
        self,
        thread_id: str,
        human_feedback: str,
    ) -> AMLState:
        """
        Resume a paused investigation after human review.

        Args:
            thread_id      : The thread ID of the paused investigation.
            human_feedback : Analyst decision string.
                             "APPROVE"  → generate SAR
                             "DISMISS"  → close case
                             Any other text → re-analyze with feedback injected
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Inject human_feedback into the checkpointed state
        await self._graph.aupdate_state(
            config,
            {"human_feedback": human_feedback},
            as_node="human_review",
        )

        logger.info(
            "Resuming investigation | thread=%s | feedback=%r",
            thread_id, human_feedback[:80],
        )
        final_state = await self._graph.ainvoke(None, config=config)
        return final_state

    async def investigate_batch(
        self,
        flagged_nodes: Dict[str, float],
        typology_scores_map: Dict[str, Dict[str, float]],
    ) -> Dict[str, AMLState]:
        """
        Process multiple flagged nodes concurrently.

        Args:
            flagged_nodes       : {node_id: risk_score}
            typology_scores_map : {node_id: {typology: score}}

        Returns:
            {node_id: final_AMLState}
        """
        import asyncio

        tasks = {
            node_id: self.investigate(
                node_id=node_id,
                risk_score=score,
                typology_scores=typology_scores_map.get(node_id, {}),
            )
            for node_id, score in flagged_nodes.items()
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: Dict[str, AMLState] = {}

        for node_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error("Investigation failed for node=%s: %s", node_id, result)
            else:
                output[node_id] = result

        return output

    def get_investigation_history(self, thread_id: str) -> list:
        """
        Retrieve the full checkpoint history for a given thread.
        Useful for audit logging and human review interfaces.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return list(self._graph.get_state_history(config))

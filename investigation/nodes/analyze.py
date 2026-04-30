"""
investigation/nodes/analyze.py
================================
LangGraph node: Analyze

The LLM receives:
  • The GNN risk score + typology probabilities
  • The Neo4j graph summary (topology signals)
  • Top-K KYC document chunks
  • Top-K negative news chunks

It must produce:
  • risk_verdict  : "VERIFIED" | "DISMISSED" | "ESCALATE"
  • risk_level    : LOW | MEDIUM | HIGH | CRITICAL
  • risk_rationale: structured reasoning string
  • rule_hits     : list of FATF/FinCEN rules triggered

NeMo Guardrails Integration
----------------------------
Before the LLM call, the prompt is passed through a NeMo Guardrails
RailsConfig that enforces:

  1. NO_HALLUCINATED_STATUTES — blocks any legal statute citation that
     is not in the approved AML statute whitelist (FATF Rec 1-40,
     FinCEN SAR rules 31 CFR §1020, EU AMLD 5/6).

  2. NO_PII_LEAKAGE — strips PII not needed for SAR (e.g. raw SSN,
     passport numbers) before sending to the LLM API.

  3. STRUCTURED_OUTPUT_ONLY — enforces that the LLM output is parseable
     JSON matching the RiskAssessment schema. If not, the Guardrail
     triggers a retry rather than letting malformed text propagate.

Output Parsing
--------------
The LLM is instructed to respond with a structured JSON block.
We parse it with a Pydantic model (RiskAssessment) and fall back to
regex extraction if JSON parsing fails. If both fail, verdict = ESCALATE
to ensure a human analyst reviews the case.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from investigation.state import AMLState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema (Pydantic)
# ---------------------------------------------------------------------------

VALID_VERDICTS  = {"VERIFIED", "DISMISSED", "ESCALATE"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


class RiskAssessment(BaseModel):
    """Structured output expected from the Analyze LLM call."""
    risk_verdict:   str = Field(..., description="VERIFIED | DISMISSED | ESCALATE")
    risk_level:     str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    risk_rationale: str = Field(..., description="Reasoning paragraph (max 400 words)")
    rule_hits:      List[str] = Field(default_factory=list,
                                      description="FATF / FinCEN rule references triggered")

    @field_validator("risk_verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_VERDICTS:
            raise ValueError(f"Invalid verdict: {v!r}. Must be one of {VALID_VERDICTS}")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {v!r}")
        return v


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior AML compliance analyst at a Tier-1 bank.
Your role is to evaluate flagged accounts and produce structured risk assessments.

RULES (strictly enforced):
1. Only cite legal statutes from this approved list:
   FATF Recommendations 1-40, FinCEN SAR rules (31 CFR §1020.320),
   EU AMLD5 (Directive 2015/849), EU AMLD6 (Directive 2018/1673),
   Basel AML Guidelines, Wolfsberg AML Principles.
2. Do NOT invent statistics, transaction amounts, or dates not present in the context.
3. Do NOT include raw PII (SSN, passport numbers) in your output.
4. Respond ONLY with valid JSON matching the RiskAssessment schema.
5. risk_verdict must be one of: VERIFIED, DISMISSED, ESCALATE.
6. risk_level must be one of: LOW, MEDIUM, HIGH, CRITICAL.
"""

HUMAN_PROMPT_TEMPLATE = """
## Flagged Account: {node_id}

### GNN Detection Results
- Overall risk score : {risk_score:.3f}
- Smurfing score    : {smurfing:.3f}
- Structuring score : {structuring:.3f}
- Layering score    : {layering:.3f}

### Graph Topology Summary
{graph_summary}

### KYC Document Context
{kyc_context}

### Negative News / Adverse Media
{news_context}

---
Based on all of the above, produce your structured JSON risk assessment.
"""


# ---------------------------------------------------------------------------
# Analyze node
# ---------------------------------------------------------------------------

class AnalyzeNode:
    """
    LangGraph node: LLM risk analysis with NeMo Guardrails enforcement.

    Constructor accepts any LangChain-compatible LLM (NIM, OpenAI, etc.)
    and an optional NeMo Guardrails LLMRails instance.
    """

    def __init__(
        self,
        llm: Any,                     # LangChain BaseChatModel (NIM, etc.)
        guardrails: Optional[Any],    # nemoguardrails.LLMRails or None
        max_context_chars: int = 4000,
    ) -> None:
        self.llm = llm
        self.guardrails = guardrails
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------
    # LangGraph callable
    # ------------------------------------------------------------------

    async def __call__(self, state: AMLState) -> Dict[str, Any]:
        """Analyze flagged account and return structured risk assessment."""
        prompt = self._build_prompt(state)
        logger.info("[analyze] Calling LLM for node=%s", state["node_id"])

        raw_response = await self._call_llm_with_guardrails(prompt, state)
        assessment = self._parse_response(raw_response, state["node_id"])

        logger.info(
            "[analyze] Verdict=%s | Level=%s | Rules=%s",
            assessment.risk_verdict, assessment.risk_level, assessment.rule_hits
        )

        return {
            "risk_verdict":   assessment.risk_verdict,
            "risk_level":     assessment.risk_level,
            "risk_rationale": assessment.risk_rationale,
            "rule_hits":      assessment.rule_hits,
            "iteration_count": state["iteration_count"] + 1,
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"[analyze] Verdict: {assessment.risk_verdict} | "
                        f"Level: {assessment.risk_level}"
                    ),
                }
            ],
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, state: AMLState) -> str:
        """Truncate and format context chunks into the human prompt."""
        kyc_text  = self._format_chunks(state.get("kyc_chunks", []))
        news_text = self._format_chunks(state.get("news_chunks", []))

        return HUMAN_PROMPT_TEMPLATE.format(
            node_id=state["node_id"],
            risk_score=state["risk_score"],
            smurfing=state["typology_scores"].get("smurfing", 0.0),
            structuring=state["typology_scores"].get("structuring", 0.0),
            layering=state["typology_scores"].get("layering", 0.0),
            graph_summary=state.get("graph_summary", "Not available"),
            kyc_context=kyc_text[: self.max_context_chars],
            news_context=news_text[: self.max_context_chars],
        )

    @staticmethod
    def _format_chunks(chunks: list) -> str:
        if not chunks:
            return "No relevant documents found."
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] (score={c['score']:.3f}, src={c['source_type']}) "
                f"{c['content'][:500]}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call with NeMo Guardrails
    # ------------------------------------------------------------------

    async def _call_llm_with_guardrails(
        self, human_prompt: str, state: AMLState
    ) -> str:
        """
        Route the prompt through NeMo Guardrails if configured,
        then call the LLM. Captures guardrail violations in state.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": human_prompt},
        ]

        if self.guardrails is not None:
            try:
                # NeMo Guardrails checks input rails BEFORE the LLM call
                # and output rails AFTER — blocking hallucinated statute refs
                response = await self.guardrails.generate_async(messages=messages)
                return response
            except Exception as exc:
                # Guardrail hard-block → log violation, escalate case
                violation_msg = f"Guardrail blocked LLM call: {exc}"
                logger.warning("[analyze] %s", violation_msg)
                state["guardrail_violations"].append(violation_msg)
                # Return a safe fallback that forces ESCALATE verdict
                return json.dumps({
                    "risk_verdict": "ESCALATE",
                    "risk_level": "HIGH",
                    "risk_rationale": "Guardrail violation prevented automated analysis. Human review required.",
                    "rule_hits": [],
                })

        # Direct LLM call (no guardrails)
        response = await self.llm.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, node_id: str) -> RiskAssessment:
        """
        Parse LLM JSON output → RiskAssessment.
        Falls back to regex extraction, then to safe ESCALATE default.
        """
        # Attempt 1: direct JSON parse
        try:
            json_str = self._extract_json(raw)
            return RiskAssessment.model_validate_json(json_str)
        except Exception as e1:
            logger.warning("[analyze] JSON parse failed for %s: %s", node_id, e1)

        # Attempt 2: regex field extraction
        try:
            return self._regex_parse(raw)
        except Exception as e2:
            logger.error("[analyze] Regex parse also failed for %s: %s", node_id, e2)

        # Fallback: human escalation
        return RiskAssessment(
            risk_verdict="ESCALATE",
            risk_level="HIGH",
            risk_rationale="Automated parsing failed. Escalated for human review.",
            rule_hits=[],
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON block from markdown-fenced or raw LLM output."""
        # Try ```json ... ``` block first
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            return fence.group(1)
        # Try first bare {...} block
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return brace.group(0)
        return text

    @staticmethod
    def _regex_parse(text: str) -> RiskAssessment:
        """Last-resort field extraction via regex."""
        def _find(pattern: str) -> Optional[str]:
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(1).strip() if m else None

        verdict = _find(r'"?risk_verdict"?\s*[:\-]\s*"?([A-Z]+)"?') or "ESCALATE"
        level   = _find(r'"?risk_level"?\s*[:\-]\s*"?([A-Z]+)"?')   or "HIGH"
        rationale = _find(r'"?risk_rationale"?\s*[:\-]\s*"(.+?)"')  or text[:200]
        return RiskAssessment(
            risk_verdict=verdict,
            risk_level=level,
            risk_rationale=rationale,
            rule_hits=[],
        )

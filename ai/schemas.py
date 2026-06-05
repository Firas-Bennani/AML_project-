"""
ai/schemas.py
==============
Single source of truth for the HTTP contract between backend/ and ai/.
Mirrored verbatim in backend/app/schemas/ai.py — keep them in sync.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Detection ────────────────────────────────────────────────────────────────


class DetectTransaction(BaseModel):
    id: str
    sender_account: str
    receiver_account: str
    amount: float
    currency: str = "USD"
    type: str = Field(..., description="DEPOSIT | WITHDRAWAL | TRANSFER")
    timestamp_hour: Optional[int] = None
    timestamp_dow: Optional[int] = None


class DetectRequest(BaseModel):
    transactions: List[DetectTransaction]


class TypologyScores(BaseModel):
    smurfing: float
    structuring: float
    layering: float


class TransactionScore(BaseModel):
    transaction_id: str
    risk_score: float
    typologies: TypologyScores


class FlaggedAccount(BaseModel):
    account_id: str
    risk_score: float
    dominant_typology: str
    typologies: TypologyScores


class DetectResponse(BaseModel):
    scores: List[TransactionScore]
    flagged_accounts: List[FlaggedAccount]
    threshold_used: float


# ── Investigation ────────────────────────────────────────────────────────────


class InvestigateRequest(BaseModel):
    node_id: str = Field(..., description="Account ID flagged by /detect")
    risk_score: float
    typology_scores: TypologyScores


class InvestigateResponse(BaseModel):
    report_id: str
    node_id: str
    verdict: str = Field(..., description="VERIFIED | DISMISSED | ESCALATE")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    sar_en: str
    sar_fr: str
    rule_hits: List[str]
    evidence_refs: List[str]
    status: str


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mock_external: bool

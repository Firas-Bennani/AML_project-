"""
backend/app/services/scoring.py
================================
Shared scoring pipeline.

Bug #2 fix: prior to this module, the scoring logic lived only inside the
POST /transactions handler. The backfill script and the periodic rescoring
job had no clean entry point to reuse it. Centralising it here means:

  * one source of truth for FLAG_THRESHOLD application
  * one source of truth for "WARMING_UP" gating
  * one fallback path when the AI service is unreachable

All callers pass the SQLAlchemy session — we never open one ourselves so
this module stays embeddable in routes, scripts, and jobs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import FLAG_THRESHOLD
from app.models.transaction import Transaction, TransactionStatus
from app.risk_scorer import score_transaction as local_score
from app.schemas.ai import DetectRequest, DetectTransaction, TypologyScores
from app.services import ai_client
from app.services.ai_client import AIServiceUnavailable

logger = logging.getLogger("backend.scoring")

_AI_HISTORY_LIMIT = 500
_AI_HISTORY_WINDOW_DAYS = 90
_WARMUP_MIN_SENDER_TXS = 3

# Per-typology flag thresholds. Must match the calibrated thresholds saved in
# the AI checkpoint (see GNNConfig.thresholds — currently [0.75, 0.80, 0.75]).
_TYP_THR_SMURFING = 0.75
_TYP_THR_STRUCTURING = 0.80
_TYP_THR_LAYERING = 0.75


def _gather_history(db: Session, tx: Transaction) -> list[Transaction]:
    cutoff = datetime.now() - timedelta(days=_AI_HISTORY_WINDOW_DAYS)
    return (
        db.query(Transaction)
        .filter(
            or_(
                Transaction.sender_account.in_(
                    [tx.sender_account, tx.receiver_account]
                ),
                Transaction.receiver_account.in_(
                    [tx.sender_account, tx.receiver_account]
                ),
            ),
            Transaction.created_at >= cutoff,
        )
        .order_by(Transaction.created_at.desc())
        .limit(_AI_HISTORY_LIMIT)
        .all()
    )


async def _score_via_ai(
    tx: Transaction, history: list[Transaction]
) -> Tuple[float, Optional[TypologyScores]]:
    detect_txs = [
        DetectTransaction(
            id=h.id,
            sender_account=h.sender_account,
            receiver_account=h.receiver_account,
            amount=float(h.amount),
            currency=h.currency,
            type=h.type.value,
        )
        for h in history
    ]
    if not any(t.id == tx.id for t in detect_txs):
        detect_txs.append(
            DetectTransaction(
                id=tx.id,
                sender_account=tx.sender_account,
                receiver_account=tx.receiver_account,
                amount=float(tx.amount),
                currency=tx.currency,
                type=tx.type.value,
            )
        )
    resp = await ai_client.detect(DetectRequest(transactions=detect_txs))
    match = next((s for s in resp.scores if s.transaction_id == tx.id), None)
    if match is None:
        return local_score(tx.amount, tx.type.value), None
    return match.risk_score, match.typologies


def _classify(
    score: float, typologies: Optional[TypologyScores], is_warming_up: bool
) -> TransactionStatus:
    if is_warming_up:
        return TransactionStatus.WARMING_UP
    typology_flag = typologies is not None and (
        typologies.smurfing >= _TYP_THR_SMURFING
        or typologies.structuring >= _TYP_THR_STRUCTURING
        or typologies.layering >= _TYP_THR_LAYERING
    )
    if typology_flag or score >= FLAG_THRESHOLD:
        return TransactionStatus.FLAGGED
    return TransactionStatus.SCORED


async def score_existing_transaction(
    db: Session, tx: Transaction, *, commit: bool = True
) -> Transaction:
    """Score a transaction that already lives in the database.

    Used by the backfill script (bug #2) and the periodic rescoring job. The
    POST /transactions route still has its own slightly-different code path
    because it needs to know about WARMING_UP at insert time.
    """
    history = _gather_history(db, tx)
    sender_count = sum(1 for h in history if h.sender_account == tx.sender_account)
    is_warming_up = sender_count < _WARMUP_MIN_SENDER_TXS

    try:
        score, typologies = await _score_via_ai(tx, history)
    except AIServiceUnavailable as exc:
        logger.warning("AI unavailable, using local fallback: %s", exc)
        score, typologies = local_score(tx.amount, tx.type.value), None
    except Exception as exc:
        logger.warning("AI raised, using local fallback: %s", exc)
        score, typologies = local_score(tx.amount, tx.type.value), None

    tx.risk_score = score
    if typologies is not None:
        tx.smurfing_score = typologies.smurfing
        tx.structuring_score = typologies.structuring
        tx.layering_score = typologies.layering
    tx.status = _classify(score, typologies, is_warming_up)
    if commit:
        db.commit()
        db.refresh(tx)
    return tx

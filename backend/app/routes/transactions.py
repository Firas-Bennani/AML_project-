import logging
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db
from app.models.transaction import Transaction, TransactionStatus
from app.models.user import User, UserRole
from app.models.alert import Alert
from app.schemas.transaction import (
    TransactionCreate,
    TransactionOut,
    PaginatedTransactions,
)
from app.schemas.ai import (
    DetectRequest,
    DetectTransaction,
    InvestigateRequest,
    TypologyScores,
)
from app.services import ai_client
from app.services.ai_client import AIServiceUnavailable
from app.services.alert_factory import create_alert_from_transaction
from app.dependencies import get_current_user, required_role
from app.risk_scorer import score_transaction
from app.audit import log_action
from typing import Optional, Tuple
from app.config import FLAG_THRESHOLD, DEFAULT_PAGE_SIZE, DEFAULT_PAGE_START

logger = logging.getLogger("backend.routes.transactions")

router = APIRouter(prefix="/transactions", tags=["Transactions"])


_AI_HISTORY_LIMIT = 500          # cap rows sent to /detect
_AI_HISTORY_WINDOW_DAYS = 90     # also cap by recency
_WARMUP_MIN_SENDER_TXS = 3       # below this, mark new tx as WARMING_UP

# Per-typology flag thresholds. Must match the calibrated thresholds saved in
# the AI checkpoint (see GNNConfig.thresholds — currently [0.75, 0.80, 0.75]).
_TYP_THR_SMURFING    = 0.75
_TYP_THR_STRUCTURING = 0.80
_TYP_THR_LAYERING    = 0.75


async def _score_via_ai_or_fallback(
    tx_id: str,
    payload: TransactionCreate,
    history: list,
) -> Tuple[float, Optional[TypologyScores]]:
    """Call AI /detect with the sender's and receiver's recent transaction
    window already including the new transaction (caller is expected to insert
    first, then query). The AI service reconstructs per-account aggregates
    matching the training schema. Fall back to local risk_scorer on AI failure."""
    try:
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
        if not any(t.id == tx_id for t in detect_txs):
            # Safety net: caller forgot insert-first. Append so /detect still
            # sees the new tx, but the per-account aggregate will be weaker.
            detect_txs.append(
                DetectTransaction(
                    id=tx_id,
                    sender_account=payload.sender_account,
                    receiver_account=payload.receiver_account,
                    amount=payload.amount,
                    currency=payload.currency,
                    type=payload.type.value,
                )
            )
        req = DetectRequest(transactions=detect_txs)
        resp = await ai_client.detect(req)
        new_score = next(
            (s for s in resp.scores if s.transaction_id == tx_id), None
        )
        if new_score is not None:
            return new_score.risk_score, new_score.typologies
    except AIServiceUnavailable as exc:
        logger.warning("AI /detect unavailable, using local fallback: %s", exc)
    except Exception as exc:
        logger.warning("AI /detect raised, using local fallback: %s", exc)
    return score_transaction(payload.amount, payload.type.value), None


def _run_investigation(alert_id: str, node_id: str, risk_score: float,
                       typologies: TypologyScores) -> None:
    """Background task: call /investigate and persist SAR fields onto the Alert."""
    import asyncio
    import json

    async def _go() -> None:
        try:
            req = InvestigateRequest(
                node_id=node_id,
                risk_score=risk_score,
                typology_scores=typologies,
            )
            result = await ai_client.investigate(req)
        except Exception as exc:
            logger.error("Investigation failed for alert %s: %s", alert_id, exc)
            return

        db = SessionLocal()
        try:
            from app.models.alert import SARStatus
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if alert is None:
                return
            alert.sar_en = result.sar_en
            alert.sar_fr = result.sar_fr
            alert.verdict = result.verdict
            alert.rule_hits = result.rule_hits
            alert.sar_generated_at = datetime.now()
            # If the analyst hasn't already validated/submitted, leave it as
            # DRAFT so the SAR Reports tab knows the narrative is ready for
            # review.
            if alert.sar_status is None:
                alert.sar_status = SARStatus.DRAFT
            db.commit()
            logger.info(
                "Alert %s populated with SAR (verdict=%s, rules=%d)",
                alert_id, result.verdict, len(result.rule_hits),
            )
        finally:
            db.close()

    asyncio.run(_go())


@router.post("/", response_model=TransactionOut, status_code=201)
async def create_transaction(
    payload: TransactionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    tx_id = str(uuid.uuid4())

    # Insert first so the history query naturally sees the new transaction.
    # The model is account-level: it needs an aggregate over a window of txs
    # per account; a single-tx call collapses 6/10 features to constants.
    new_transaction = Transaction(
        id=tx_id,
        sender_name=payload.sender_name,
        sender_account=payload.sender_account,
        receiver_name=payload.receiver_name,
        receiver_account=payload.receiver_account,
        amount=payload.amount,
        currency=payload.currency,
        type=payload.type,
        status=TransactionStatus.PENDING,
        risk_score=None,
    )
    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)

    # Pull the sender/receiver tx window: last 90 days, capped at 500 rows.
    cutoff = datetime.now() - timedelta(days=_AI_HISTORY_WINDOW_DAYS)
    history = (
        db.query(Transaction)
        .filter(
            or_(
                Transaction.sender_account.in_(
                    [payload.sender_account, payload.receiver_account]
                ),
                Transaction.receiver_account.in_(
                    [payload.sender_account, payload.receiver_account]
                ),
            ),
            Transaction.created_at >= cutoff,
        )
        .order_by(Transaction.created_at.desc())
        .limit(_AI_HISTORY_LIMIT)
        .all()
    )

    # If the sender has barely any history yet, AI scores are unreliable —
    # the account-level aggregates collapse. Mark the transaction so the UI
    # can render a "warming up" indicator instead of a misleading score.
    sender_tx_count = sum(
        1 for h in history if h.sender_account == payload.sender_account
    )
    is_warming_up = sender_tx_count < _WARMUP_MIN_SENDER_TXS

    score, typologies = await _score_via_ai_or_fallback(tx_id, payload, history)

    # Persist per-typology breakdown so the list view can render badges
    # without re-calling /detect for every row.
    if typologies is not None:
        new_transaction.smurfing_score = typologies.smurfing
        new_transaction.structuring_score = typologies.structuring
        new_transaction.layering_score = typologies.layering

    # FLAGGED if any typology exceeds its calibrated threshold; otherwise
    # SCORED. WARMING_UP wins because the score itself is unreliable.
    typology_flag = typologies is not None and (
        typologies.smurfing    >= _TYP_THR_SMURFING
        or typologies.structuring >= _TYP_THR_STRUCTURING
        or typologies.layering  >= _TYP_THR_LAYERING
    )
    if is_warming_up:
        status = TransactionStatus.WARMING_UP
    elif typology_flag or score >= FLAG_THRESHOLD:
        status = TransactionStatus.FLAGGED
    else:
        status = TransactionStatus.SCORED
    new_transaction.status = status
    new_transaction.risk_score = score
    db.commit()
    db.refresh(new_transaction)

    log_action(
        db=db,
        action="CREATE_TRANSACTION",
        user_id=current_user.id,
        entity_type="TRANSACTION",
        entity_id=new_transaction.id,
        details=f"Amount: {payload.amount} {payload.currency} , Score: {new_transaction.risk_score}, Status:{status.value}",
    )

    if status == TransactionStatus.FLAGGED:
        alert = create_alert_from_transaction(db, new_transaction.id)
        if alert is not None:
            log_action(
                db=db,
                action="CREATE_ALERT",
                user_id=current_user.id,
                entity_type="ALERT",
                entity_id=alert.id,
                details=(
                    f"Alert auto-created for transaction {new_transaction.id} "
                    f"with score {new_transaction.risk_score}"
                ),
            )

            # Fire-and-forget /investigate. Uses real AI typologies if /detect
            # returned them; otherwise synthesizes a single-axis score breakdown
            # so the SAR still gets generated.
            if typologies is None:
                typologies = TypologyScores(
                    smurfing=score, structuring=score, layering=0.0
                )
            background_tasks.add_task(
                _run_investigation,
                alert.id,
                payload.sender_account,
                new_transaction.risk_score,
                typologies,
            )

    return new_transaction


@router.get("/", response_model=PaginatedTransactions)
def get_all_transactions(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    status: Optional[TransactionStatus] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Transaction)
    if status:
        query = query.filter(Transaction.status == status)
    if min_amount:
        query = query.filter(Transaction.amount >= min_amount)
    if max_amount:
        query = query.filter(Transaction.amount <= max_amount)
    total = query.count()
    # Newest first so freshly-AI-scored rows (with typology populated) land on
    # page 1 instead of being buried under months of seed data.
    items = (
        query.order_by(Transaction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"total": total, "items": items}


@router.get("/{transaction_id}", response_model=TransactionOut)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction

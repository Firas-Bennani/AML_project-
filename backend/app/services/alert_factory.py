"""
backend/app/services/alert_factory.py
======================================
Single entry point for creating Alert rows from a FLAGGED transaction.

Bug #3 fix: prior to this module, alert creation was inlined in the POST
/transactions handler, which meant background rescoring jobs and backfill
scripts had no way to create an alert. Centralising the logic here gives us:

  - A single place to enforce the "no duplicate OPEN alert per transaction"
    invariant.
  - A single place to set sar_status=DRAFT so the SAR list endpoint can find
    every alert that still needs a SAR generated.
  - Reuse from scripts/backfill_alerts.py without re-implementing the rules.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.models.alert import Alert, AlertStatus, SARStatus
from app.models.transaction import Transaction, TransactionStatus

logger = logging.getLogger("backend.alert_factory")


def create_alert_from_transaction(
    db: Session,
    transaction_id: str,
    *,
    reason: Optional[str] = None,
) -> Optional[Alert]:
    """Create an Alert for a FLAGGED transaction.

    Returns the existing alert (without modifying it) if one is already OPEN
    or UNDER_REVIEW. Returns None when the transaction isn't FLAGGED or
    doesn't exist — the caller decides whether to log/raise.
    """
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if tx is None:
        logger.warning("alert_factory: transaction %s not found", transaction_id)
        return None
    if tx.status != TransactionStatus.FLAGGED:
        logger.debug(
            "alert_factory: tx %s status=%s not FLAGGED, skipping",
            transaction_id, tx.status,
        )
        return None
    if tx.risk_score is None:
        logger.warning(
            "alert_factory: tx %s has no risk_score, cannot create alert",
            transaction_id,
        )
        return None

    existing = (
        db.query(Alert)
        .filter(
            Alert.transaction_id == transaction_id,
            Alert.status.in_([AlertStatus.OPEN, AlertStatus.UNDER_REVIEW]),
        )
        .first()
    )
    if existing is not None:
        logger.debug(
            "alert_factory: open alert %s already exists for tx %s",
            existing.id, transaction_id,
        )
        return existing

    if reason is None:
        reason = (
            f"Transaction of {tx.amount} {tx.currency} scored {tx.risk_score} "
            f"- above the threshold."
        )

    alert = Alert(
        id=str(uuid.uuid4()),
        transaction_id=transaction_id,
        risk_score=tx.risk_score,
        reason=reason,
        status=AlertStatus.OPEN,
        # SAR starts as DRAFT; the /investigate background task populates the
        # actual narrative + bumps sar_generated_at when the LLM returns.
        sar_status=SARStatus.DRAFT,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(
        "alert_factory: created alert %s for tx %s (score=%.3f)",
        alert.id, transaction_id, tx.risk_score,
    )
    return alert

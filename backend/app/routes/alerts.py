from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from app.database import get_db
from app.models.alert import Alert, AlertStatus
from app.models.transaction import Transaction
from app.models.user import User, UserRole
from app.schemas.alert import AlertOut, AlertUpdate, PaginatedAlerts
from app.dependencies import get_current_user, required_role
from app.audit import log_action
from app.config import DEFAULT_PAGE_SIZE, DEFAULT_PAGE_START

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/", response_model=PaginatedAlerts)
def get_all_alerts(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    state: Optional[AlertStatus] = Query(
        None,
        description="Filter by alert status (OPEN/UNDER_REVIEW/CONFIRMED/DISMISSED)",
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List alerts.

    Sorted by ``risk_score`` DESC so the highest-risk items land at the top of
    page 1. Optional ``state`` filter mirrors the four UI tabs on the Alerts
    page.
    """
    query = db.query(Alert)
    if state is not None:
        query = query.filter(Alert.status == state)
    total = query.count()
    items = (
        query.order_by(Alert.risk_score.desc(), Alert.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"total": total, "items": items}


@router.get("/open", response_model=PaginatedAlerts, deprecated=True)
def get_open_alerts(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Kept for backward compatibility — prefer ``GET /alerts/?state=OPEN``."""
    query = db.query(Alert).filter(Alert.status == AlertStatus.OPEN)
    total = query.count()
    items = (
        query.order_by(Alert.risk_score.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"total": total, "items": items}


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.patch("/{alert_id}", response_model=AlertOut)
def update_alert(
    alert_id: str,
    payload: AlertUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    old_status = alert.status

    if payload.status is not None:
        alert.status = payload.status
        if payload.status == AlertStatus.UNDER_REVIEW and alert.assigned_to is None:
            alert.assigned_to = current_user.id
    if payload.notes is not None:
        alert.notes = payload.notes
    if payload.assigned_to is not None:
        alert.assigned_to = payload.assigned_to

    alert.updated_at = datetime.now()

    db.commit()
    db.refresh(alert)

    final_statuses = [AlertStatus.CONFIRMED, AlertStatus.DISMISSED]
    if payload.status in final_statuses:
        transaction = (
            db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
        )
        if transaction:
            transaction.reviewed_by = current_user.id
            db.commit()

            log_action(
                db=db,
                action="TRANSACTION_REVIEWED",
                user_id=current_user.id,
                entity_type="TRANSACTION",
                entity_id=transaction.id,
                details=f"Transaction marked as reviewed by {current_user.email} - alert {alert_id} {payload.status.value}",
            )

    log_action(
        db=db,
        action="UPDATE_ALERT",
        user_id=current_user.id,
        entity_type="ALERT",
        entity_id=alert_id,
        details=f"Status changed from {old_status.value} to {alert.status.value} by {current_user.email}",
    )

    return alert

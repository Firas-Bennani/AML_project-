from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
from app.database import get_db
from app.models.user import User, UserRole
from app.models.transaction import Transaction, TransactionStatus
from app.models.alert import Alert, AlertStatus, SARStatus
from app.models.audit_log import AuditLog
from app.dependencies import required_role
from app.audit import log_action
from app.schemas.reports import (
    AnalystPerformanceReport,
    MissedFlagsReport,
    ActivitySummaryReport,
    PaginatedSARReports,
    SARDetail,
    SARUpdate,
)
from app.services import sar_service
from app.config import (
    RED_FLAG_DISMISSAL_RATE,
    RED_FLAG_MIN_TRANSACTIONS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    ALL_TIME_START,
    DEFAULT_PAGE_SIZE,
    DEFAULT_PAGE_START,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/analyst-performance", response_model=AnalystPerformanceReport)
def get_analyst_performance(
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.AUDITOR])),
):
    analysts = db.query(User).filter(User.role == UserRole.ANALYST).all()
    performance = []

    for analyst in analysts:

        assigned = db.query(Alert).filter(Alert.assigned_to == analyst.id).count()
        confirmed = (
            db.query(Alert)
            .filter(
                Alert.assigned_to == analyst.id, Alert.status == AlertStatus.CONFIRMED
            )
            .count()
        )
        dismissed = (
            db.query(Alert)
            .filter(
                Alert.assigned_to == analyst.id, Alert.status == AlertStatus.DISMISSED
            )
            .count()
        )
        under_review = (
            db.query(Alert)
            .filter(
                Alert.assigned_to == analyst.id,
                Alert.status == AlertStatus.UNDER_REVIEW,
            )
            .count()
        )
        transactions_reviewed = (
            db.query(Transaction).filter(Transaction.reviewed_by == analyst.id).count()
        )
        total_resolved = confirmed + dismissed
        dismissal_rate = (
            (dismissed / total_resolved) * 100 if total_resolved > 0 else 0.0
        )

        login_count = (
            db.query(AuditLog)
            .filter(AuditLog.user_id == analyst.id, AuditLog.action == "LOGIN")
            .count()
        )

        performance.append(
            {
                "analyst_id": analyst.id,
                "analyst_name": analyst.name,
                "analyst_email": analyst.email,
                "is_active": analyst.is_active,
                "alerts_assigned": assigned,
                "alerts_confirmed": confirmed,
                "alerts_dismissed": dismissed,
                "alerts_under_review": under_review,
                "transactions_reviewed": transactions_reviewed,
                "dismissal_rate_percent": dismissal_rate,
                "total_logins": login_count,
                "red_flag": dismissal_rate > RED_FLAG_DISMISSAL_RATE
                and transactions_reviewed >= RED_FLAG_MIN_TRANSACTIONS,
            }
        )

    return {
        "report_type": "analyst_performance",
        "generated_at": datetime.now().isoformat(),
        "total_analysts": len(analysts),
        "analysts": performance,
    }


@router.get("/missed-flags", response_model=MissedFlagsReport)
def get_missed_flags(
    threshold: float = 0.6,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.AUDITOR])),
):
    dismissed_high_risk = (
        db.query(Alert)
        .filter(Alert.status == AlertStatus.DISMISSED, Alert.risk_score >= threshold)
        .all()
    )
    missed_flags = []
    for alert in dismissed_high_risk:
        transaction = (
            db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()
        )
        reviewer = None
        if transaction and transaction.reviewed_by:
            reviewer_user = (
                db.query(User).filter(User.id == transaction.reviewed_by).first()
            )
            if reviewer_user:
                reviewer = reviewer_user.email

        missed_flags.append(
            {
                "alert_id": alert.id,
                "transaction_id": alert.transaction_id,
                "risk_score": alert.risk_score,
                "alert_reason": alert.reason,
                "alert_notes": alert.notes,
                "dismissed_by": reviewer,
                "amount": transaction.amount if transaction else None,
                "currency": transaction.currency if transaction else None,
                "sender_name": transaction.sender_name if transaction else None,
                "receiver_name": transaction.receiver_name if transaction else None,
                "created_at": alert.created_at.isoformat(),
                "resolved_at": alert.updated_at.isoformat(),
                "severity": (
                    "CRITICAL"
                    if alert.risk_score >= SEVERITY_CRITICAL
                    else "HIGH" if alert.risk_score >= SEVERITY_HIGH else "MEDIUM"
                ),
            }
        )

    missed_flags.sort(key=lambda x: x["risk_score"], reverse=True)

    return {
        "report_type": "missed_flags",
        "generated_at": datetime.now().isoformat(),
        "threshold_used": threshold,
        "total_missed_flags": len(missed_flags),
        "critical_count": sum(1 for f in missed_flags if f["severity"] == "CRITICAL"),
        "high_count": sum(1 for f in missed_flags if f["severity"] == "HIGH"),
        "medium_count": sum(1 for f in missed_flags if f["severity"] == "MEDIUM"),
        "missed_flags": missed_flags,
    }


@router.get("/summary", response_model=ActivitySummaryReport)
def get_activity_summary(
    period: str = "monthly",
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.AUDITOR])),
):
    now = datetime.now()

    period_map = {
        "daily": (now - timedelta(days=1), "Last 24 hours"),
        "weekly": (now - timedelta(days=7), "Last 7 days"),
        "monthly": (now - timedelta(days=30), "Last 30 days"),
        "all": (datetime.fromisoformat(ALL_TIME_START), "All time"),
    }

    since, period_label = period_map.get(
        period, (now - timedelta(days=30), "Last 30 days (default)")
    )

    total_transactions = (
        db.query(Transaction).filter(Transaction.created_at >= since).count()
    )
    flagged_transactions = (
        db.query(Transaction)
        .filter(
            Transaction.created_at >= since,
            Transaction.status == TransactionStatus.FLAGGED,
        )
        .count()
    )
    # Bug #4: "scored" reports the count of transactions that have any
    # risk_score (i.e. the AI pipeline has touched them) — not just the
    # SCORED status bucket. Otherwise FLAGGED rows didn't count as scored
    # and the dashboard read 0 even when scoring was working.
    scored_transactions = (
        db.query(Transaction)
        .filter(
            Transaction.created_at >= since,
            Transaction.risk_score.isnot(None),
        )
        .count()
    )
    # "reviewed" means an analyst has at least picked the alert up.
    reviewed_alert_tx_ids = (
        db.query(Alert.transaction_id)
        .filter(
            Alert.status.in_(
                [
                    AlertStatus.UNDER_REVIEW,
                    AlertStatus.CONFIRMED,
                    AlertStatus.DISMISSED,
                ]
            )
        )
        .distinct()
        .subquery()
    )
    reviewed_transactions = (
        db.query(Transaction)
        .filter(
            Transaction.created_at >= since,
            Transaction.id.in_(reviewed_alert_tx_ids),
        )
        .count()
    )
    flag_rate = (
        (flagged_transactions / total_transactions) * 100
        if total_transactions > 0
        else 0.0
    )

    total_alerts = db.query(Alert).filter(Alert.created_at >= since).count()
    open_alerts = (
        db.query(Alert)
        .filter(Alert.created_at >= since, Alert.status == AlertStatus.OPEN)
        .count()
    )
    confirmed_alerts = (
        db.query(Alert)
        .filter(Alert.created_at >= since, Alert.status == AlertStatus.CONFIRMED)
        .count()
    )
    dismissed_alerts = (
        db.query(Alert)
        .filter(Alert.created_at >= since, Alert.status == AlertStatus.DISMISSED)
        .count()
    )
    under_review_alerts = (
        db.query(Alert)
        .filter(Alert.created_at >= since, Alert.status == AlertStatus.UNDER_REVIEW)
        .count()
    )
    total_resolved = confirmed_alerts + dismissed_alerts
    resolution_rate = (total_resolved / total_alerts) * 100 if total_alerts > 0 else 0.0

    active_users = (
        db.query(AuditLog.user_id)
        .filter(AuditLog.timestamp >= since, AuditLog.action == "LOGIN")
        .distinct()
        .count()
    )
    total_logins = (
        db.query(AuditLog)
        .filter(AuditLog.timestamp >= since, AuditLog.action == "LOGIN")
        .count()
    )
    failed_logins = (
        db.query(AuditLog)
        .filter(AuditLog.timestamp >= since, AuditLog.action == "LOGIN_Failed")
        .count()
    )

    return {
        "report_type": "activity_summary",
        "period": period_label,
        "generated_at": datetime.now().isoformat(),
        "since": since.isoformat(),
        "transactions": {
            "total": total_transactions,
            "flagged": flagged_transactions,
            "scored_normal": scored_transactions,
            "reviewed": reviewed_transactions,
            "flag_rate_percent": flag_rate,
        },
        "alerts": {
            "total": total_alerts,
            "open": open_alerts,
            "under_review": under_review_alerts,
            "confirmed": confirmed_alerts,
            "dismissed": dismissed_alerts,
            "resolution_rate_percent": resolution_rate,
        },
        "user_activity": {
            "active_users": active_users,
            "total_logins": total_logins,
            "failed_logins": failed_logins,
        },
    }


# ── SAR Reports tab (Bug #4) ─────────────────────────────────────────────────

# Analysts also need read access to SARs from the Reports view, so this is
# scoped to ADMIN/AUDITOR/ANALYST rather than ADMIN/AUDITOR-only.
_SAR_ALLOWED_ROLES = [UserRole.ADMIN, UserRole.AUDITOR, UserRole.ANALYST]


@router.get("/sar", response_model=PaginatedSARReports)
def list_sar(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    status: Optional[SARStatus] = Query(
        None, description="Filter by sar_status (DRAFT/VALIDATED/SUBMITTED)"
    ),
    search: Optional[str] = Query(
        None, description="Filter by sender/receiver name or account substring"
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role(_SAR_ALLOWED_ROLES)),
):
    total, items = sar_service.list_sar_reports(
        db,
        skip=skip,
        limit=limit,
        status=status.value if status else None,
        search=search,
    )
    return {"total": total, "items": items}


@router.get("/sar/{alert_id}", response_model=SARDetail)
def get_sar(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role(_SAR_ALLOWED_ROLES)),
):
    detail = sar_service.get_sar_detail(db, alert_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="SAR not found")
    return detail


@router.get("/sar/{alert_id}/pdf")
def get_sar_pdf(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role(_SAR_ALLOWED_ROLES)),
):
    try:
        pdf_bytes = sar_service.render_sar_pdf(db, alert_id)
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF rendering dependency (reportlab) is not installed. "
                "Run `pip install -r backend/requirements.txt`."
            ),
        ) from exc
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail="SAR not found")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="SAR-{alert_id}.pdf"'
        },
    )


@router.patch("/sar/{alert_id}", response_model=SARDetail)
def update_sar(
    alert_id: str,
    payload: SARUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.ANALYST])),
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=404, detail="SAR not found")

    if payload.sar_status is not None:
        try:
            new_status = SARStatus(payload.sar_status)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sar_status. Expected one of {[s.value for s in SARStatus]}",
            ) from exc
        old_status = (
            alert.sar_status.value if alert.sar_status is not None else None
        )
        alert.sar_status = new_status
        if new_status == SARStatus.SUBMITTED:
            alert.sar_submitted_at = datetime.now()
        db.commit()
        db.refresh(alert)
        log_action(
            db=db,
            action="UPDATE_SAR",
            user_id=current_user.id,
            entity_type="ALERT",
            entity_id=alert.id,
            details=(
                f"sar_status {old_status} → {new_status.value} by {current_user.email}"
            ),
        )

    detail = sar_service.get_sar_detail(db, alert_id)
    return detail

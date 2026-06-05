from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User, UserRole
from app.schemas.audit_log import AuditLogOut
from app.dependencies import required_role
from app.config import DEFAULT_PAGE_SIZE, DEFAULT_PAGE_START

router = APIRouter(prefix="/logs", tags=["Audit Logs"])


@router.get("/", response_model=list[AuditLogOut])
def get_all_logs(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.AUDITOR])),
):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return logs


@router.get("/my", response_model=list[AuditLogOut])
def get_my_logs(
    skip: int = DEFAULT_PAGE_START,
    limit: int = DEFAULT_PAGE_SIZE,
    db: Session = Depends(get_db),
    current_user: User = Depends(
        required_role([UserRole.ADMIN, UserRole.ANALYST, UserRole.AUDITOR])
    ),
):
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return logs


@router.get("/user/{user_id}", response_model=list[AuditLogOut])
def get_logs_by_user(
    user_id: str,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(required_role([UserRole.ADMIN, UserRole.AUDITOR])),
):

    target_user = db.query(User).filter(User.id == user_id).first()

    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    logs = (
        db.query(AuditLog)
        .filter(AuditLog.user_id == user_id)
        .order_by(AuditLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return logs

import uuid
from sqlalchemy.orm import Session
from app.models.audit_log import AuditLog


def log_action(
    db: Session,
    action: str,
    user_id: str = None,
    entity_type: str = None,
    entity_id: str = None,
    details: str = None,
):

    log_entry = AuditLog(
        id=str(uuid.uuid4()),
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
    )

    db.add(log_entry)
    db.commit()

    # We log 3 events: login, transaction creation, alert update

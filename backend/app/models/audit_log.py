from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)  # what action was performed
    entity_type = Column(String, nullable=True)  # example: USER, TRANSACTION, ALERT...
    entity_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(
        DateTime, default=datetime.now, nullable=False, index=True
    )  # when the action happened
    user = relationship(
        "User", backref="audit_logs"
    )  # link to the user who performed the action

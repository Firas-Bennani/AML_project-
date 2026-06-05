from sqlalchemy import JSON, Column, String, Float, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.database import Base


class AlertStatus(str, enum.Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    DISMISSED = "DISMISSED"
    CONFIRMED = "CONFIRMED"


class SARStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, index=True)
    transaction_id = Column(String, ForeignKey("transactions.id"), nullable=False)
    risk_score = Column(Float, nullable=False)  # same as the one in transaction
    reason = Column(String, nullable=False)
    status = Column(Enum(AlertStatus), nullable=False, default=AlertStatus.OPEN)
    notes = Column(Text, nullable=True)
    assigned_to = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, nullable=False)

    # Investigation output (populated asynchronously after AI /investigate)
    sar_en = Column(Text, nullable=True)
    sar_fr = Column(Text, nullable=True)
    verdict = Column(String(16), nullable=True)
    rule_hits = Column(JSON, nullable=True)
    sar_status = Column(Enum(SARStatus), nullable=True)
    sar_generated_at = Column(DateTime, nullable=True)
    sar_submitted_at = Column(DateTime, nullable=True)

    transaction = relationship(
        "Transaction", backref="alert"
    )  # one alert for each transaction (w l3aks)
    assigned_analyst = relationship(
        "User", backref="assigned_alerts"
    )  # one user can be assigned to many alerts (l3aks 8alet)

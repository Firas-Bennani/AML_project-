from sqlalchemy import Column, String, Float, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.database import Base


class TransactionType(str, enum.Enum):
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    TRANSFER = "TRANSFER"


class TransactionStatus(str, enum.Enum):
    PENDING = "PENDING"
    WARMING_UP = "WARMING_UP"   # AI score unreliable: < 3 prior txs for sender
    SCORED = "SCORED"
    FLAGGED = "FLAGGED"


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(String, primary_key=True, index=True)

    sender_name = Column(String, nullable=False)
    sender_account = Column(String, nullable=False)

    receiver_name = Column(String, nullable=False)
    receiver_account = Column(String, nullable=False)

    amount = Column(Float, nullable=False)
    currency = Column(String, nullable=False, default="TND")
    type = Column(Enum(TransactionType), nullable=False)

    status = Column(
        Enum(TransactionStatus), nullable=False, default=TransactionStatus.PENDING
    )

    risk_score = Column(Float, nullable=True)
    smurfing_score = Column(Float, nullable=True)
    structuring_score = Column(Float, nullable=True)
    layering_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False)

    reviewed_by = Column(String, ForeignKey("users.id"), nullable=True)
    reviewer = relationship("User", backref="reviewed_transactions")

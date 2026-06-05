from pydantic import BaseModel, field_validator, model_validator
from datetime import datetime
from typing import Optional
from app.models.transaction import TransactionType, TransactionStatus


class TransactionCreate(BaseModel):

    @model_validator(mode="after")
    def accounts_must_be_different(self):
        if self.sender_account == self.receiver_account:
            raise ValueError("Sender and receiver accounts cannot be the same.")
        return self

    sender_name: str
    sender_account: str
    receiver_name: str
    receiver_account: str
    amount: float
    currency: str = "TND"
    type: TransactionType

    @field_validator("amount")
    def amount_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError("Amount must be greater than zero")
        return value

    @field_validator("currency")
    def currency_must_be_valid(cls, value):
        allowed = [
            "TND",
            "EUR",
            "USD",
            "GBP",
            "JPY",
        ]  # we can add other currencies if we want
        if value.upper() not in allowed:
            raise ValueError(f"Currency must be one of:{allowed}")
        return value.upper()


class TransactionOut(BaseModel):
    id: str
    sender_name: str
    sender_account: str
    receiver_name: str
    receiver_account: str
    amount: float
    currency: str
    type: TransactionType
    status: TransactionStatus
    risk_score: Optional[float]
    smurfing_score: Optional[float] = None
    structuring_score: Optional[float] = None
    layering_score: Optional[float] = None
    reviewed_by: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedTransactions(BaseModel):
    total: int
    items: list[TransactionOut]

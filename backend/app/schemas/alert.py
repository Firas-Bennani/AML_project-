from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from app.models.alert import AlertStatus


class AlertOut(BaseModel):
    id: str
    transaction_id: str
    risk_score: float
    reason: str
    status: AlertStatus
    notes: Optional[str]
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime
    sar_en: Optional[str] = None
    sar_fr: Optional[str] = None
    verdict: Optional[str] = None
    rule_hits: Optional[List[str]] = None
    model_config = {"from_attributes": True}


class AlertUpdate(BaseModel):
    status: Optional[AlertStatus] = None
    notes: Optional[str] = None
    assigned_to: Optional[str] = None


class PaginatedAlerts(BaseModel):
    total: int
    items: list[AlertOut]

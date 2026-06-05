from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class AnalystPerformanceItem(BaseModel):
    analyst_id: str
    analyst_name: str
    analyst_email: str
    is_active: bool
    alerts_assigned: int
    alerts_confirmed: int
    alerts_dismissed: int
    alerts_under_review: int
    transactions_reviewed: int
    dismissal_rate_percent: float
    total_logins: int
    red_flag: bool


class AnalystPerformanceReport(BaseModel):
    report_type: str
    generated_at: str
    total_analysts: int
    analysts: list[AnalystPerformanceItem]


class MissedFlagItem(BaseModel):
    alert_id: str
    transaction_id: str
    risk_score: float
    alert_reason: str
    alert_notes: Optional[str]
    dismissed_by: Optional[str]
    amount: Optional[float]
    currency: Optional[str]
    sender_name: Optional[str]
    receiver_name: Optional[str]
    created_at: str
    resolved_at: str
    severity: str


class MissedFlagsReport(BaseModel):
    report_type: str
    generated_at: str
    threshold_used: float
    total_missed_flags: int
    critical_count: int
    high_count: int
    medium_count: int
    missed_flags: list[MissedFlagItem]


class TransactionStats(BaseModel):
    total: int
    flagged: int
    scored_normal: int
    reviewed: int
    flag_rate_percent: float


class AlertStats(BaseModel):
    total: int
    open: int
    under_review: int
    confirmed: int
    dismissed: int
    resolution_rate_percent: float


class UserActivityStats(BaseModel):
    active_users: int
    total_logins: int
    failed_logins: int


class ActivitySummaryReport(BaseModel):
    report_type: str
    period: str
    generated_at: str
    since: str
    transactions: TransactionStats
    alerts: AlertStats
    user_activity: UserActivityStats


# ── SAR Reports (Bug #4) ────────────────────────────────────────────────────

class SARListItem(BaseModel):
    alert_id: str
    transaction_id: str
    sar_status: str
    verdict: Optional[str] = None
    risk_score: float
    generated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    sender_name: Optional[str] = None
    sender_account: Optional[str] = None
    receiver_name: Optional[str] = None
    receiver_account: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    typologies: list[str] = []


class PaginatedSARReports(BaseModel):
    total: int
    items: list[SARListItem]


class SARDetail(BaseModel):
    alert_id: str
    transaction_id: str
    sar_status: str
    verdict: Optional[str] = None
    risk_score: float
    smurfing_score: Optional[float] = None
    structuring_score: Optional[float] = None
    layering_score: Optional[float] = None
    generated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    rule_hits: list[str] = []
    typologies: list[str] = []

    # Reporting institution / declarant identification.
    reporting_institution: str
    jurisdiction: str

    # Suspect identification (sender side of the suspicious transaction).
    suspect_name: Optional[str] = None
    suspect_account: Optional[str] = None
    counterparty_name: Optional[str] = None
    counterparty_account: Optional[str] = None

    # Activity description.
    amount: Optional[float] = None
    currency: Optional[str] = None
    transaction_type: Optional[str] = None
    transaction_date: Optional[str] = None

    # Narrative.
    sar_en: Optional[str] = None
    sar_fr: Optional[str] = None
    analyst_notes: Optional[str] = None


class SARUpdate(BaseModel):
    sar_status: Optional[str] = None

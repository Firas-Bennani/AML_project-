"""
backend/app/services/sar_service.py
=====================================
Bug #4 helper layer for the SAR Reports tab.

Responsibilities:
  * Build the list payload (alert join transaction) for ``GET /reports/sar``.
  * Build the detail payload for ``GET /reports/sar/{id}``.
  * Render the SAR as a PDF (FinCEN/TRACFIN-aligned structure) for
    ``GET /reports/sar/{id}/pdf``. Uses reportlab so it ships with no
    system-level dependency.

The SAR storage lives on the Alert row itself (``sar_en``, ``sar_fr``,
``verdict``, ``rule_hits``, ``sar_status``, ``sar_generated_at``,
``sar_submitted_at``). We deliberately don't introduce a separate
``sar_reports`` table — every SAR is 1:1 with an Alert, and the LangGraph
investigation agent already targets the alert row.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.alert import Alert, SARStatus
from app.models.transaction import Transaction

logger = logging.getLogger("backend.sar")

REPORTING_INSTITUTION = "AML Compliance Dept."
JURISDICTION = "FR (TRACFIN) / TN (CTAF)"

_TYP_THRESHOLD = 0.4  # mirror investigation/nodes/report.py


def _typology_list(tx: Optional[Transaction]) -> list[str]:
    if tx is None:
        return []
    out: list[str] = []
    if tx.smurfing_score is not None and tx.smurfing_score >= _TYP_THRESHOLD:
        out.append("smurfing")
    if tx.structuring_score is not None and tx.structuring_score >= _TYP_THRESHOLD:
        out.append("structuring")
    if tx.layering_score is not None and tx.layering_score >= _TYP_THRESHOLD:
        out.append("layering")
    return out


def _sar_status(alert: Alert) -> str:
    if alert.sar_status is not None:
        return alert.sar_status.value
    # Pre-bug-#4 rows never had sar_status set; if the narrative exists, call
    # it a DRAFT so the UI surfaces it instead of hiding it.
    return SARStatus.DRAFT.value if alert.sar_en else SARStatus.DRAFT.value


def list_sar_reports(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> tuple[int, list[dict]]:
    """Return ``(total, items)`` for the SAR Reports tab."""
    query = (
        db.query(Alert, Transaction)
        .join(Transaction, Transaction.id == Alert.transaction_id)
        .filter(Alert.sar_en.isnot(None) | Alert.sar_status.isnot(None))
    )
    if status:
        query = query.filter(Alert.sar_status == status)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                Transaction.sender_name.ilike(like),
                Transaction.receiver_name.ilike(like),
                Transaction.sender_account.ilike(like),
                Transaction.receiver_account.ilike(like),
            )
        )

    total = query.count()
    rows = (
        query.order_by(Alert.sar_generated_at.desc().nullslast(), Alert.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    items: list[dict] = []
    for alert, tx in rows:
        items.append(
            {
                "alert_id": alert.id,
                "transaction_id": alert.transaction_id,
                "sar_status": _sar_status(alert),
                "verdict": alert.verdict,
                "risk_score": alert.risk_score,
                "generated_at": (
                    alert.sar_generated_at.isoformat()
                    if alert.sar_generated_at
                    else None
                ),
                "submitted_at": (
                    alert.sar_submitted_at.isoformat()
                    if alert.sar_submitted_at
                    else None
                ),
                "sender_name": tx.sender_name,
                "sender_account": tx.sender_account,
                "receiver_name": tx.receiver_name,
                "receiver_account": tx.receiver_account,
                "amount": tx.amount,
                "currency": tx.currency,
                "typologies": _typology_list(tx),
            }
        )
    return total, items


def get_sar_detail(db: Session, alert_id: str) -> Optional[dict]:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        return None
    tx = (
        db.query(Transaction)
        .filter(Transaction.id == alert.transaction_id)
        .first()
    )
    return {
        "alert_id": alert.id,
        "transaction_id": alert.transaction_id,
        "sar_status": _sar_status(alert),
        "verdict": alert.verdict,
        "risk_score": alert.risk_score,
        "smurfing_score": getattr(tx, "smurfing_score", None) if tx else None,
        "structuring_score": getattr(tx, "structuring_score", None) if tx else None,
        "layering_score": getattr(tx, "layering_score", None) if tx else None,
        "generated_at": (
            alert.sar_generated_at.isoformat() if alert.sar_generated_at else None
        ),
        "submitted_at": (
            alert.sar_submitted_at.isoformat() if alert.sar_submitted_at else None
        ),
        "rule_hits": list(alert.rule_hits or []),
        "typologies": _typology_list(tx),
        "reporting_institution": REPORTING_INSTITUTION,
        "jurisdiction": JURISDICTION,
        "suspect_name": tx.sender_name if tx else None,
        "suspect_account": tx.sender_account if tx else None,
        "counterparty_name": tx.receiver_name if tx else None,
        "counterparty_account": tx.receiver_account if tx else None,
        "amount": tx.amount if tx else None,
        "currency": tx.currency if tx else None,
        "transaction_type": tx.type.value if tx else None,
        "transaction_date": tx.created_at.isoformat() if tx else None,
        "sar_en": alert.sar_en,
        "sar_fr": alert.sar_fr,
        "analyst_notes": alert.notes,
    }


def render_sar_pdf(db: Session, alert_id: str) -> Optional[bytes]:
    """Render a SAR as a PDF byte stream.

    Mirrors the FinCEN Form 111 / TRACFIN CERFA 10534 four-part structure:
      I.   Reporting institution
      II.  Subject (suspect) information
      III. Suspicious activity (amount, currency, type, dates, typologies)
      IV.  Narrative (bilingual EN/FR) + rule hits
    """
    detail = get_sar_detail(db, alert_id)
    if detail is None:
        return None

    # reportlab is imported lazily — if it isn't available the endpoint will
    # raise ImportError which the route translates to a 500 with a helpful
    # message, instead of breaking module import.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"SAR {detail['alert_id']}",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    mono = ParagraphStyle(
        "mono", parent=body, fontName="Courier", fontSize=9, leading=12
    )

    story: list = []
    story.append(Paragraph("Suspicious Activity Report (SAR)", h1))
    story.append(
        Paragraph(
            f"<b>Alert ID:</b> {detail['alert_id']} &nbsp;&nbsp; "
            f"<b>Status:</b> {detail['sar_status']} &nbsp;&nbsp; "
            f"<b>Verdict:</b> {detail.get('verdict') or '—'}",
            body,
        )
    )
    story.append(Spacer(1, 12))

    # Part I — Reporting institution
    story.append(Paragraph("Part I — Reporting Institution", h2))
    table_i = Table(
        [
            ["Institution", detail["reporting_institution"]],
            ["Jurisdiction", detail["jurisdiction"]],
            ["Generated", detail.get("generated_at") or "—"],
            ["Submitted", detail.get("submitted_at") or "—"],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    table_i.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table_i)
    story.append(Spacer(1, 12))

    # Part II — Subject
    story.append(Paragraph("Part II — Subject (Suspect) Information", h2))
    table_ii = Table(
        [
            ["Suspect name", detail.get("suspect_name") or "—"],
            ["Suspect account", detail.get("suspect_account") or "—"],
            ["Counterparty name", detail.get("counterparty_name") or "—"],
            ["Counterparty account", detail.get("counterparty_account") or "—"],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    table_ii.setStyle(table_i.style)
    story.append(table_ii)
    story.append(Spacer(1, 12))

    # Part III — Activity
    story.append(Paragraph("Part III — Suspicious Activity", h2))
    table_iii = Table(
        [
            [
                "Amount",
                (
                    f"{detail['amount']:.2f} {detail['currency']}"
                    if detail.get("amount") is not None
                    else "—"
                ),
            ],
            ["Type", detail.get("transaction_type") or "—"],
            ["Transaction date", detail.get("transaction_date") or "—"],
            ["Risk score (GNN)", f"{detail['risk_score']:.3f}"],
            [
                "Typology breakdown",
                (
                    f"smurfing={detail.get('smurfing_score'):.3f} "
                    f"structuring={detail.get('structuring_score'):.3f} "
                    f"layering={detail.get('layering_score'):.3f}"
                    if detail.get("smurfing_score") is not None
                    else "—"
                ),
            ],
            ["Typologies detected", ", ".join(detail["typologies"]) or "—"],
            ["Rule hits", ", ".join(detail["rule_hits"]) or "—"],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    table_iii.setStyle(table_i.style)
    story.append(table_iii)
    story.append(Spacer(1, 12))

    # Part IV — Narrative
    story.append(Paragraph("Part IV — Narrative (EN)", h2))
    if detail.get("sar_en"):
        for line in detail["sar_en"].splitlines():
            story.append(Paragraph(line or "&nbsp;", mono))
    else:
        story.append(Paragraph("(SAR narrative not yet generated.)", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Part IV — Narrative (FR)", h2))
    if detail.get("sar_fr"):
        for line in detail["sar_fr"].splitlines():
            story.append(Paragraph(line or "&nbsp;", mono))
    else:
        story.append(Paragraph("(Récit non encore généré.)", body))

    if detail.get("analyst_notes"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("Analyst notes", h2))
        story.append(Paragraph(detail["analyst_notes"], body))

    doc.build(story)
    return buf.getvalue()

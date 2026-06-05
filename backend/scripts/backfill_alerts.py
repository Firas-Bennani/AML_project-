"""
backend/scripts/backfill_alerts.py
===================================
Bug #3 backfill — every FLAGGED transaction must have a matching alert.

Old POSTs that fell through the cracks (or seed data flipped to FLAGGED by
the rescoring backfill) currently exist without alerts, so they're invisible
to the analyst workflow. This script closes that gap by walking every
FLAGGED transaction with no OPEN/UNDER_REVIEW alert and creating one via
the same factory the live handler uses.

Run inside the backend container:
    docker compose exec backend python -m scripts.backfill_alerts
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.alert import Alert, AlertStatus
from app.models.transaction import Transaction, TransactionStatus
from app.services.alert_factory import create_alert_from_transaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_alerts")

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore
        return it


def _select_orphan_flagged(db: Session) -> list[Transaction]:
    """FLAGGED transactions with no active alert."""
    # Subquery of transactions that already have an OPEN/UNDER_REVIEW alert.
    active_alert_tx_ids = (
        db.query(Alert.transaction_id)
        .filter(Alert.status.in_([AlertStatus.OPEN, AlertStatus.UNDER_REVIEW]))
        .subquery()
    )
    return (
        db.query(Transaction)
        .filter(
            Transaction.status == TransactionStatus.FLAGGED,
            ~Transaction.id.in_(active_alert_tx_ids),
        )
        .order_by(Transaction.created_at.asc())
        .all()
    )


def run(dry_run: bool) -> int:
    db = SessionLocal()
    try:
        targets = _select_orphan_flagged(db)
        total = len(targets)
        if total == 0:
            logger.info("Every FLAGGED transaction already has an active alert.")
            return 0
        logger.info("Backfilling alerts for %d FLAGGED transaction(s)", total)

        created = 0
        for tx in tqdm(targets, desc="alerts"):
            alert = create_alert_from_transaction(db, tx.id)
            if alert is not None:
                created += 1
            if dry_run:
                db.rollback()

        logger.info(
            "Alert backfill complete: created=%d skipped=%d dry_run=%s",
            created, total - created, dry_run,
        )
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

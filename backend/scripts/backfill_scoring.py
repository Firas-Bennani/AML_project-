"""
backend/scripts/backfill_scoring.py
====================================
Bug #2 backfill — re-score every transaction whose risk_score IS NULL.

The seed data and any rows that came in while the AI service was down end up
with NULL risk_score, which renders as "—" in the Transactions and Reports
screens. This script walks them in chronological order, calls the AI /detect
endpoint (with the same per-account history window the live POST handler
uses), and writes the result back.

Run inside the backend container:
    docker compose exec backend python -m scripts.backfill_scoring

Optional flags:
    --batch-size N      rows per AI call (default 50)
    --limit N           stop after N rows (default: until table is drained)
    --dry-run           score but don't commit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.alert import AlertStatus
from app.models.transaction import Transaction, TransactionStatus
from app.services.alert_factory import create_alert_from_transaction
from app.services.scoring import score_existing_transaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_scoring")

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover — tqdm is optional polish.
    def tqdm(it, **_):  # type: ignore
        return it


def _select_unscored(db: Session, limit: int | None) -> list[Transaction]:
    q = (
        db.query(Transaction)
        .filter(Transaction.risk_score.is_(None))
        .order_by(Transaction.created_at.asc())
    )
    if limit is not None:
        q = q.limit(limit)
    return q.all()


async def run(batch_size: int, limit: int | None, dry_run: bool) -> int:
    db = SessionLocal()
    try:
        targets = _select_unscored(db, limit)
        total = len(targets)
        if total == 0:
            logger.info("No transactions with NULL risk_score. Nothing to do.")
            return 0

        logger.info("Backfilling %d transactions (batch_size=%d)", total, batch_size)
        scored = 0
        errors = 0

        progress = tqdm(range(0, total, batch_size), desc="batches")
        for start in progress:
            batch = targets[start:start + batch_size]
            for tx in batch:
                try:
                    await score_existing_transaction(db, tx, commit=not dry_run)
                    scored += 1
                    if (
                        not dry_run
                        and tx.status == TransactionStatus.FLAGGED
                    ):
                        # Maintain the FLAGGED → Alert invariant so the
                        # historical data plays nicely with the rest of the
                        # platform. The factory is idempotent for OPEN alerts.
                        create_alert_from_transaction(db, tx.id)
                except Exception as exc:
                    errors += 1
                    logger.exception("scoring failed for tx %s: %s", tx.id, exc)
            if dry_run:
                db.rollback()

        logger.info(
            "Backfill complete: scored=%d errors=%d dry_run=%s",
            scored, errors, dry_run,
        )
        return errors
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args.batch_size, args.limit, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())

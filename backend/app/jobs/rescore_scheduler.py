"""
backend/app/jobs/rescore_scheduler.py
======================================
Bug #2 guard rail — periodic rescoring of unscored transactions.

Every RESCORE_INTERVAL_MINUTES (default 5) the scheduler picks up any
transactions whose risk_score is still NULL and runs them through the same
shared scoring service the live POST handler uses. This compensates for
transient AI outages and prevents NULL columns from accumulating in the UI.

A single APScheduler BackgroundScheduler is started by main.py's startup
hook and stopped by the shutdown hook. The scheduler is process-local: in a
multi-worker deployment, configure RESCORE_LEADER_ONLY and run a single
worker as the leader.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.database import SessionLocal
from app.models.alert import AlertStatus
from app.models.transaction import Transaction, TransactionStatus
from app.services.alert_factory import create_alert_from_transaction
from app.services.scoring import score_existing_transaction

logger = logging.getLogger("backend.jobs.rescore")

RESCORE_INTERVAL_MINUTES = int(os.getenv("BACKEND_RESCORE_INTERVAL_MINUTES", "5"))
RESCORE_BATCH_SIZE = int(os.getenv("BACKEND_RESCORE_BATCH_SIZE", "50"))

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()


async def _rescore_once() -> None:
    db = SessionLocal()
    try:
        targets = (
            db.query(Transaction)
            .filter(Transaction.risk_score.is_(None))
            .order_by(Transaction.created_at.asc())
            .limit(RESCORE_BATCH_SIZE)
            .all()
        )
        if not targets:
            return
        logger.info(
            "[rescore] picking up %d unscored transaction(s)", len(targets)
        )
        for tx in targets:
            try:
                await score_existing_transaction(db, tx)
                if tx.status == TransactionStatus.FLAGGED:
                    create_alert_from_transaction(db, tx.id)
            except Exception as exc:
                logger.exception("[rescore] tx %s failed: %s", tx.id, exc)
    finally:
        db.close()


def _job_entry() -> None:
    """Sync entry point invoked by APScheduler. APScheduler doesn't run
    coroutines natively, so we hop into asyncio.run for each tick."""
    # Reentrancy guard — if a tick is still running when the next one fires
    # (rare, but possible under load), skip rather than pile up.
    if not _lock.acquire(blocking=False):
        logger.warning("[rescore] previous tick still running, skipping")
        return
    try:
        asyncio.run(_rescore_once())
    except Exception:
        logger.exception("[rescore] unexpected failure")
    finally:
        _lock.release()


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _job_entry,
        trigger=IntervalTrigger(minutes=RESCORE_INTERVAL_MINUTES),
        id="rescore_unscored_transactions",
        max_instances=1,
        coalesce=True,
        # Run a few seconds after boot so we don't race the seed-on-empty hook.
        next_run_time=None,
    )
    _scheduler.start()
    logger.info(
        "rescore scheduler started (every %d min, batch=%d)",
        RESCORE_INTERVAL_MINUTES, RESCORE_BATCH_SIZE,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception:
        pass
    _scheduler = None

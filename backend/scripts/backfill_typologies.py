"""backend/scripts/backfill_typologies.py
==========================================
One-off offline backfill: populates smurfing/structuring/layering/risk on
transaction rows that still have NULL typology, by running each touched
account's 90-day window through /detect — exactly like the live pipeline.

Design points
-------------
- Per-account, not per-transaction (matches how the model was trained).
- Warming-up accounts (< 3 txs in the 90-day window) are skipped and leave
  typology NULL — same gate as the live pipeline's WARMING_UP path.
- Idempotent: only rows where smurfing_score IS NULL are inspected and
  updated. Re-running after a partial crash picks up where it left off.
- AI errors on a single account are logged and the loop continues.
- Per-tx risk_score is reused from the /detect response (which itself is
  max(sender_max_typology, receiver_max_typology)).

Run:
    docker compose exec backend python /app/scripts/backfill_typologies.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models.transaction import Transaction
# Register sibling models so SQLAlchemy resolves Transaction's relationships.
import app.models.user   # noqa: F401
import app.models.alert  # noqa: F401


AI_URL          = os.getenv("BACKEND_AI_BASE_URL", "http://ai:8001")
HISTORY_LIMIT   = 500
HISTORY_DAYS    = 90
WARMUP_MIN_TXS  = 3
PROGRESS_EVERY  = 50


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [backfill] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill").info


def _tx_to_detect_dict(t: Transaction) -> dict:
    return {
        "id": t.id,
        "sender_account": t.sender_account,
        "receiver_account": t.receiver_account,
        "amount": float(t.amount),
        "currency": t.currency,
        "type": t.type.value,
    }


def main() -> int:
    t_start = time.perf_counter()
    db = SessionLocal()
    client = httpx.Client(timeout=60.0)

    try:
        # AI reachability pre-check — fail loud instead of hammering errors.
        try:
            r = client.get(f"{AI_URL}/healthz", timeout=5.0)
            r.raise_for_status()
        except Exception as exc:
            log(f"[fatal] AI service not reachable at {AI_URL}: {exc}")
            return 1

        null_rows = (
            db.query(Transaction)
            .filter(Transaction.smurfing_score.is_(None))
            .all()
        )
        if not null_rows:
            log("Nothing to backfill — all rows already have typology scores.")
            return 0

        accounts: set[str] = set()
        for r in null_rows:
            accounts.add(r.sender_account)
            accounts.add(r.receiver_account)
        log(
            f"{len(null_rows)} NULL-typology rows across "
            f"{len(accounts)} unique accounts."
        )

        cutoff = datetime.now() - timedelta(days=HISTORY_DAYS)

        # account_id -> { tx_id -> (risk, sm, st, la) } for txs in that
        # account's /detect window. Same tx may appear in multiple accounts'
        # windows — values are symmetric (AI does max(sender, receiver)).
        per_account_scores: dict[str, dict[str, tuple[float, float, float, float]]] = {}
        warming_up: set[str] = set()
        ai_errors: list[tuple[str, str]] = []

        for idx, account in enumerate(sorted(accounts), start=1):
            window = (
                db.query(Transaction)
                .filter(
                    (Transaction.sender_account == account)
                    | (Transaction.receiver_account == account),
                    Transaction.created_at >= cutoff,
                )
                .order_by(Transaction.created_at.desc())
                .limit(HISTORY_LIMIT)
                .all()
            )
            if len(window) < WARMUP_MIN_TXS:
                warming_up.add(account)
            else:
                try:
                    payload = {"transactions": [_tx_to_detect_dict(t) for t in window]}
                    resp = client.post(f"{AI_URL}/detect", json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                    per_tx: dict[str, tuple[float, float, float, float]] = {}
                    for s in body.get("scores", []):
                        typ = s["typologies"]
                        per_tx[s["transaction_id"]] = (
                            float(s["risk_score"]),
                            float(typ["smurfing"]),
                            float(typ["structuring"]),
                            float(typ["layering"]),
                        )
                    per_account_scores[account] = per_tx
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    ai_errors.append((account, str(exc)))
                    log(f"  [warn] account={account}: AI error — {exc}")

            if idx % PROGRESS_EVERY == 0:
                log(
                    f"  progress {idx}/{len(accounts)} — "
                    f"scored={len(per_account_scores)} "
                    f"warming_up={len(warming_up)} "
                    f"errors={len(ai_errors)}"
                )

        # Walk each NULL row and pick a side's score. If both sides are
        # warming-up / errored, leave the row NULL (per the user's rule).
        updated = 0
        still_null = 0
        for row in null_rows:
            sender_score = per_account_scores.get(row.sender_account, {}).get(row.id)
            receiver_score = per_account_scores.get(row.receiver_account, {}).get(row.id)
            picked = sender_score if sender_score is not None else receiver_score
            if picked is None:
                still_null += 1
                continue
            risk, sm, st, la = picked
            row.risk_score = risk
            row.smurfing_score = sm
            row.structuring_score = st
            row.layering_score = la
            updated += 1

        db.commit()

        elapsed = time.perf_counter() - t_start
        log("─" * 60)
        log(f"DONE  accounts processed   = {len(accounts)}")
        log(f"      accounts scored      = {len(per_account_scores)}")
        log(f"      accounts warming-up  = {len(warming_up)}")
        log(f"      AI errors            = {len(ai_errors)}")
        log(f"      rows updated         = {updated}")
        log(f"      rows still NULL      = {still_null}")
        log(f"      elapsed              = {elapsed:.2f}s")
        return 0

    finally:
        client.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""
backend/scripts/probe_detector.py
==================================
Send a diverse batch of synthetic transactions through the AI /detect
endpoint and print how the model scores each behavioural persona.

Usage (inside the backend container):
    docker compose exec backend python scripts/probe_detector.py

Or against a different host:
    BACKEND_AI_BASE_URL=http://localhost:8001 python scripts/probe_detector.py

The personas span the typology axes the model was trained on:
  - smurfing   : one sender → many small fan-out receivers
  - structuring: amounts clustered in the $9k-$10k CTR-proximity band
  - layering   : multi-hop chain with cross-border + currency churn
plus a control set (quiet retail, weekend casual, large legitimate wires)
so the contrast is visible.
"""
from __future__ import annotations

import os
import statistics
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import httpx


AI_BASE_URL = os.getenv("BACKEND_AI_BASE_URL", "http://ai:8001")


# ── Persona generators ──────────────────────────────────────────────────────
#
# Each persona returns a tuple (sender_account, [tx_dict, ...]). Receivers
# are persona-local except where shared receivers are part of the signal.

@dataclass
class Persona:
    name: str
    sender: str
    expectation: str
    txs: List[dict]


def _tx(sender: str, receiver: str, amount: float, currency: str = "USD",
        ttype: str = "TRANSFER", hour: int = 12, dow: int = 2) -> dict:
    return {
        "id": f"tx-{uuid.uuid4().hex[:8]}",
        "sender_account": sender,
        "receiver_account": receiver,
        "amount": float(amount),
        "currency": currency,
        "type": ttype,
        "timestamp_hour": hour,
        "timestamp_dow": dow,
    }


def quiet_retail() -> Persona:
    s = "ACC-QUIET-001"
    return Persona(
        name="Quiet retail",
        sender=s,
        expectation="LOW (single small in-country tx)",
        txs=[_tx(s, "ACC-MERCHANT-A", 124.50, hour=14, dow=2)],
    )


def weekend_casual() -> Persona:
    s = "ACC-WEEKEND-001"
    return Persona(
        name="Weekend casual",
        sender=s,
        expectation="LOW (small Saturday spend)",
        txs=[
            _tx(s, "ACC-RESTO-A", 1_220.00, hour=13, dow=5),
            _tx(s, "ACC-RESTO-B", 980.00,   hour=20, dow=5),
        ],
    )


def round_amount_spender() -> Persona:
    s = "ACC-ROUND-001"
    return Persona(
        name="Round-amount spender",
        sender=s,
        expectation="MEDIUM (round numbers but legit-looking)",
        txs=[
            _tx(s, "ACC-VENDOR-A",  5_000.00, hour=10, dow=1),
            _tx(s, "ACC-VENDOR-B", 10_000.00, hour=11, dow=1),
            _tx(s, "ACC-VENDOR-C", 15_000.00, hour=15, dow=1),
        ],
    )


def structuring_suspect() -> Persona:
    """Eight deposits sitting just under the $10k CTR threshold."""
    s = "ACC-STRUCT-001"
    amounts = [9_100, 9_300, 9_500, 9_650, 9_750, 9_850, 9_900, 9_950]
    return Persona(
        name="Structuring suspect",
        sender=s,
        expectation="HIGH structuring (sub-CTR clustering)",
        txs=[
            _tx(s, f"ACC-DST-S{i:02d}", a, ttype="DEPOSIT",
                hour=(8 + i) % 24, dow=(i % 5))
            for i, a in enumerate(amounts)
        ],
    )


def smurfing_fan_out() -> Persona:
    """One sender → twelve different receivers in small amounts."""
    s = "ACC-SMURF-001"
    return Persona(
        name="Smurfing fan-out",
        sender=s,
        expectation="HIGH smurfing (many small destinations)",
        txs=[
            _tx(s, f"ACC-MULE-{i:02d}", 350 + (i * 47),
                hour=(7 + i) % 24, dow=(i % 5))
            for i in range(12)
        ],
    )


def layering_chain() -> Persona:
    """A→B→C→D→E chain with currency + cross-border churn.
    The named sender is A; the chain hops are visible to the GNN as edges."""
    chain = ["ACC-LAYER-A", "ACC-LAYER-B", "ACC-LAYER-C",
             "ACC-LAYER-D", "ACC-LAYER-E"]
    currencies = ["EUR", "GBP", "JPY", "USD"]
    txs = []
    for i in range(len(chain) - 1):
        txs.append(_tx(
            chain[i], chain[i + 1],
            amount=42_000 + (i * 1_500),
            currency=currencies[i % len(currencies)],
            hour=3 + i, dow=3,
        ))
    return Persona(
        name="Layering chain",
        sender=chain[0],
        expectation="HIGH layering (multi-hop cross-border)",
        txs=txs,
    )


def crossborder_whale() -> Persona:
    s = "ACC-WHALE-001"
    return Persona(
        name="Cross-border whale",
        sender=s,
        expectation="HIGH any (huge non-USD wires)",
        txs=[
            _tx(s, "ACC-OFFSHORE-A", 250_000, currency="EUR",  hour=11, dow=2),
            _tx(s, "ACC-OFFSHORE-B", 500_000, currency="JPY",  hour=14, dow=2),
            _tx(s, "ACC-OFFSHORE-C", 1_000_000, currency="CHF", hour=16, dow=2),
        ],
    )


def offhours_rapidfire() -> Persona:
    """Six wires from same sender between 02:00 and 04:00."""
    s = "ACC-NIGHT-001"
    return Persona(
        name="Off-hours rapid-fire",
        sender=s,
        expectation="MEDIUM (volume + odd hours)",
        txs=[
            _tx(s, f"ACC-NIGHT-DST-{i:02d}",
                amount=2_500 + (i * 100),
                hour=2 + (i % 3), dow=4)
            for i in range(6)
        ],
    )


PERSONA_FACTORIES = [
    quiet_retail,
    weekend_casual,
    round_amount_spender,
    structuring_suspect,
    smurfing_fan_out,
    layering_chain,
    crossborder_whale,
    offhours_rapidfire,
]


# ── Probe ───────────────────────────────────────────────────────────────────


def main() -> int:
    personas = [factory() for factory in PERSONA_FACTORIES]
    all_txs = [tx for p in personas for tx in p.txs]
    print(f"AI base URL    : {AI_BASE_URL}")
    print(f"Personas       : {len(personas)}")
    print(f"Transactions   : {len(all_txs)}")
    print()

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{AI_BASE_URL}/detect",
                               json={"transactions": all_txs})
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        print(f"ERROR calling /detect: {exc}", file=sys.stderr)
        return 2

    threshold = payload.get("threshold_used", 0.45)
    flagged_ids = {f["account_id"] for f in payload.get("flagged_accounts", [])}

    # Collapse per-tx scores → per-account (every tx of a sender shares its score)
    by_account: dict = {}
    for s in payload["scores"]:
        sender = next(
            (tx["sender_account"] for tx in all_txs if tx["id"] == s["transaction_id"]),
            None,
        )
        if sender is None:
            continue
        by_account[sender] = s  # last one wins; they're identical anyway

    # ── Per-persona table ─────────────────────────────────────────────────
    print("=" * 110)
    print(f"{'PERSONA':<24} {'N':>3} {'MEAN $':>11} {'RISK':>7}  "
          f"{'SMURF':>7} {'STRUCT':>7} {'LAYER':>7}  {'FLAG':>4}  {'EXPECT'}")
    print("-" * 110)

    for p in personas:
        score = by_account.get(p.sender)
        if score is None:
            print(f"{p.name:<24} {'(no score returned)':<60}")
            continue
        mean_amt = statistics.mean(t["amount"] for t in p.txs)
        typ = score["typologies"]
        flagged = "YES" if p.sender in flagged_ids else "no"
        print(
            f"{p.name:<24} {len(p.txs):>3} {mean_amt:>11,.0f} "
            f"{score['risk_score']:>7.3f}  "
            f"{typ['smurfing']:>7.3f} {typ['structuring']:>7.3f} {typ['layering']:>7.3f}  "
            f"{flagged:>4}  {p.expectation}"
        )

    # ── Aggregate distribution ────────────────────────────────────────────
    risks = [s["risk_score"] for s in payload["scores"]]
    smurf = [s["typologies"]["smurfing"]    for s in payload["scores"]]
    struct = [s["typologies"]["structuring"] for s in payload["scores"]]
    layer = [s["typologies"]["layering"]    for s in payload["scores"]]

    def stats(xs):
        return f"min={min(xs):.3f} mean={statistics.mean(xs):.3f} " \
               f"median={statistics.median(xs):.3f} max={max(xs):.3f}"

    print("-" * 110)
    print(f"Score distribution across {len(risks)} transactions:")
    print(f"  risk        : {stats(risks)}")
    print(f"  smurfing    : {stats(smurf)}")
    print(f"  structuring : {stats(struct)}")
    print(f"  layering    : {stats(layer)}")
    print(f"  trigger thr : {threshold} (per-typology cutoffs come from checkpoint;"
          f" check ai-service startup log for exact values)")

    unique_senders = {tx["sender_account"] for tx in all_txs}
    flagged_senders = flagged_ids & unique_senders
    flagged_receivers = flagged_ids - unique_senders
    print(
        f"  flagged     : {len(flagged_ids)} accounts "
        f"({len(flagged_senders)}/{len(unique_senders)} named senders; "
        f"{len(flagged_receivers)} receivers tripped via graph propagation)"
    )
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

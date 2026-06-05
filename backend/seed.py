"""
seed.py — populate aml.db with a realistic demo dataset.

Creates 3 fixed users + 30 named account profiles organised across three
behavioural tiers (CLEAN / MEDIUM / SUSPECT), ~540 profile transactions
plus ~30 isolated singletons for graph noise — all spread over the last
90 days.

Risk + typology scores are left NULL; status=PENDING. The companion
script `scripts/backfill_typologies.py` runs each account's window
through /detect to populate real model scores afterwards.

Run paths:
  • Auto: called from app.main on first boot when the users table is empty
    (see seed_if_empty()).
  • Manual: `docker compose exec backend python seed.py` wipes and reseeds.

Deterministic — random.seed is fixed so two fresh installs produce the
same demo dataset.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from faker import Faker

from app.database import SessionLocal, engine, Base
from app.models.user import User, UserRole
from app.models.transaction import Transaction, TransactionType, TransactionStatus
from app.models.alert import Alert
from app.models.audit_log import AuditLog
from app.auth import hash_password


fake = Faker("fr_FR")
random.seed(20260511)
Faker.seed(20260511)


# ─────────────────────────────────────────────────────────────────────────
# Profile catalogue — each entry is a coherent behavioural pattern.
# Field meanings:
#   tier        — bucket label, drives downstream filtering only.
#   name        — display name (also used as the "owner" sender_name/receiver_name).
#   acct        — stable account id string repeated across all rows of the profile.
#   n_txs       — how many transactions to generate.
#   amount      — (lo, hi) uniform range; rounded to 2 dp so amounts look real.
#   currs       — currency choices per tx; multi-currency profiles drive slot 8.
#   types       — TransactionType choices per tx.
#   direction   — "sender" / "receiver" / "mixed" — where the profile sits.
#   hours       — "business" 09-17, "mixed" 07-22 + a small night tail,
#                 "night" 60% in 02-05.
#   dow         — "weekday" Mon-Fri only, "mixed" all 7 days.
#   signature   — informational tag (smurfing/structuring/layering/mixed).
# ─────────────────────────────────────────────────────────────────────────

_DOM   = ["EUR"]
_USD   = ["USD"]
_MX2   = ["USD", "EUR"]
_MX3   = ["USD", "EUR", "GBP"]
_MX4   = ["USD", "EUR", "GBP", "JPY"]

PROFILES = [
    # ── TIER 1 — CLEAN (10 accounts) ───────────────────────────────────
    dict(tier="CLEAN", name="Retraité fixe",            acct="ACC-CLN-RET-001",
         n_txs=15, amount=(50, 300),     currs=_DOM, types=["DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Salarié régulier",         acct="ACC-CLN-SAL-001",
         n_txs=15, amount=(200, 1500),   currs=_DOM, types=["TRANSFER","DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Étudiant",                 acct="ACC-CLN-ETU-001",
         n_txs=15, amount=(20, 200),     currs=_DOM, types=["TRANSFER","DEPOSIT","WITHDRAWAL"],
         direction="sender",   hours="mixed",    dow="mixed"),
    dict(tier="CLEAN", name="Petit commerçant local",   acct="ACC-CLN-COM-001",
         n_txs=15, amount=(100, 500),    currs=_USD, types=["DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Cadre supérieur",          acct="ACC-CLN-CDR-001",
         n_txs=15, amount=(1000, 3000),  currs=_DOM, types=["TRANSFER","DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Famille avec enfants",     acct="ACC-CLN-FAM-001",
         n_txs=15, amount=(30, 400),     currs=_DOM, types=["TRANSFER","DEPOSIT","WITHDRAWAL"],
         direction="sender",   hours="mixed",    dow="mixed"),
    dict(tier="CLEAN", name="Senior retraité aisé",     acct="ACC-CLN-SEN-001",
         n_txs=15, amount=(1500, 2500),  currs=_DOM, types=["DEPOSIT"],
         direction="receiver", hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Auto-entrepreneur",        acct="ACC-CLN-AUT-001",
         n_txs=15, amount=(200, 1200),   currs=_DOM, types=["TRANSFER","DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Jeune actif salarié",      acct="ACC-CLN-JEU-001",
         n_txs=15, amount=(100, 1000),   currs=_DOM, types=["TRANSFER","DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="CLEAN", name="Coopérative locale",       acct="ACC-CLN-COO-001",
         n_txs=18, amount=(200, 800),    currs=_DOM, types=["DEPOSIT"],
         direction="receiver", hours="business", dow="weekday"),

    # ── TIER 2 — MEDIUM (10 accounts) ──────────────────────────────────
    dict(tier="MEDIUM", name="PME moyenne",             acct="ACC-MED-PME-001",
         n_txs=20, amount=(5000, 50000),  currs=_MX2, types=["TRANSFER","DEPOSIT"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="MEDIUM", name="Expatrié actif",          acct="ACC-MED-EXP-001",
         n_txs=20, amount=(1000, 15000),  currs=_MX3, types=["TRANSFER"],
         direction="sender",   hours="mixed",    dow="mixed"),
    dict(tier="MEDIUM", name="Cabinet médical",         acct="ACC-MED-MED-001",
         n_txs=20, amount=(1000, 10000),  currs=_DOM, types=["DEPOSIT"],
         direction="receiver", hours="business", dow="weekday"),
    dict(tier="MEDIUM", name="Cabinet d'avocat",        acct="ACC-MED-AVO-001",
         n_txs=20, amount=(5000, 30000),  currs=_MX2, types=["TRANSFER","DEPOSIT"],
         direction="receiver", hours="business", dow="weekday"),
    dict(tier="MEDIUM", name="Restaurant cash-intensive", acct="ACC-MED-RES-001",
         n_txs=22, amount=(500, 3000),    currs=_USD, types=["DEPOSIT"],
         direction="receiver", hours="mixed",    dow="mixed"),
    dict(tier="MEDIUM", name="E-commerce moyen",        acct="ACC-MED-ECM-001",
         n_txs=20, amount=(500, 20000),   currs=_MX3, types=["TRANSFER"],
         direction="receiver", hours="mixed",    dow="mixed"),
    dict(tier="MEDIUM", name="Consultant international", acct="ACC-MED-CON-001",
         n_txs=20, amount=(2000, 15000),  currs=_MX3, types=["TRANSFER"],
         direction="receiver", hours="mixed",    dow="mixed"),
    dict(tier="MEDIUM", name="Bureau de change autorisé", acct="ACC-MED-BDC-001",
         n_txs=22, amount=(1000, 50000),  currs=_MX4, types=["TRANSFER","DEPOSIT","WITHDRAWAL"],
         direction="mixed",    hours="business", dow="weekday"),
    dict(tier="MEDIUM", name="Société de transport",    acct="ACC-MED-TRP-001",
         n_txs=20, amount=(5000, 30000),  currs=_USD, types=["TRANSFER"],
         direction="sender",   hours="business", dow="weekday"),
    dict(tier="MEDIUM", name="Cabinet d'expertise comptable", acct="ACC-MED-CPT-001",
         n_txs=20, amount=(500, 15000),   currs=_MX2, types=["TRANSFER","DEPOSIT"],
         direction="receiver", hours="business", dow="weekday"),

    # ── TIER 3 — SUSPECT (10 accounts) ─────────────────────────────────
    # SMURFING (3) — fan-in of many small inbound DEPOSITs
    dict(tier="SUSPECT", name="Mule réseau 1",          acct="ACC-SUS-MUL-001",
         n_txs=25, amount=(500, 3000),    currs=_USD, types=["DEPOSIT"],
         direction="receiver", hours="mixed",    dow="mixed", signature="smurfing"),
    dict(tier="SUSPECT", name="Mule réseau 2",          acct="ACC-SUS-MUL-002",
         n_txs=25, amount=(600, 2800),    currs=_USD, types=["DEPOSIT"],
         direction="receiver", hours="mixed",    dow="mixed", signature="smurfing"),
    dict(tier="SUSPECT", name="Collecteur cash",        acct="ACC-SUS-COL-001",
         n_txs=25, amount=(1500, 4500),   currs=_USD, types=["DEPOSIT"],
         direction="receiver", hours="mixed",    dow="mixed", signature="smurfing"),
    # STRUCTURING (3) — sub-CTR-threshold band
    dict(tier="SUSPECT", name="Structureur classique",  acct="ACC-SUS-STR-001",
         n_txs=20, amount=(9000, 9999),   currs=_USD, types=["DEPOSIT"],
         direction="sender",   hours="business", dow="weekday", signature="structuring"),
    dict(tier="SUSPECT", name="Structureur prudent",    acct="ACC-SUS-STR-002",
         n_txs=20, amount=(9300, 9800),   currs=_USD, types=["DEPOSIT"],
         direction="sender",   hours="mixed",    dow="mixed", signature="structuring"),
    dict(tier="SUSPECT", name="Structureur multi-comptes", acct="ACC-SUS-STR-003",
         n_txs=20, amount=(9000, 9999),   currs=_MX2, types=["DEPOSIT"],
         direction="sender",   hours="mixed",    dow="mixed", signature="structuring"),
    # LAYERING (3) — multi-currency, varied types, night tails
    dict(tier="SUSPECT", name="Layering offshore",      acct="ACC-SUS-LAY-001",
         n_txs=20, amount=(5000, 50000),  currs=_MX3, types=["TRANSFER"],
         direction="sender",   hours="mixed",    dow="mixed", signature="layering"),
    dict(tier="SUSPECT", name="Layering rapide",        acct="ACC-SUS-LAY-002",
         n_txs=22, amount=(2000, 30000),  currs=_MX3, types=["TRANSFER","WITHDRAWAL"],
         direction="sender",   hours="mixed",    dow="mixed", signature="layering"),
    dict(tier="SUSPECT", name="Layering complexe",      acct="ACC-SUS-LAY-003",
         n_txs=25, amount=(3000, 40000),  currs=_MX4, types=["TRANSFER","DEPOSIT","WITHDRAWAL"],
         direction="sender",   hours="night",    dow="mixed", signature="layering"),
    # MIXED (1) — features of all three typologies
    dict(tier="SUSPECT", name="Cas atypique",           acct="ACC-SUS-MIX-001",
         n_txs=28, amount=(7000, 12000),  currs=_MX3, types=["TRANSFER","DEPOSIT","WITHDRAWAL"],
         direction="mixed",    hours="night",    dow="mixed", signature="mixed"),
]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _gen_timestamp(hours_dist: str, dow_dist: str, days_back_max: int = 90) -> datetime:
    """Pick a timestamp within the last `days_back_max` days, respecting the
    profile's hour and day-of-week distribution. Loops to reject weekend
    timestamps when dow=='weekday'."""
    for _ in range(20):
        ts = _now() - timedelta(days=random.randint(0, days_back_max))
        if hours_dist == "business":
            h = random.randint(9, 17)
        elif hours_dist == "mixed":
            h = random.choices(
                population=[*range(7, 22), *range(0, 5)],
                weights=[*[3] * 15, *[1] * 5],
            )[0]
        elif hours_dist == "night":
            h = random.randint(2, 5) if random.random() < 0.6 else random.randint(0, 23)
        else:
            h = random.randint(0, 23)
        ts = ts.replace(
            hour=h, minute=random.randint(0, 59), second=random.randint(0, 59)
        )
        if dow_dist == "weekday" and ts.weekday() >= 5:
            continue
        return ts
    return ts  # fallback


def _gen_amount(lo: float, hi: float) -> float:
    """Uniform on [lo, hi), rounded to 2 dp — never a perfectly round int."""
    return round(random.uniform(lo, hi), 2)


def _gen_counterparty() -> tuple[str, str]:
    """Return (display_name, bban). Names are person-like 70% of the time,
    company-like 30%, to look organic on the dashboard."""
    name = fake.name() if random.random() > 0.3 else fake.company()
    return name, fake.bban()


# ─────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────

def _build_profile_txs(profile: dict) -> List[Transaction]:
    """Build a profile's N transactions. The profile account sits on one side
    of every row; the other side is a fresh singleton each time."""
    out: List[Transaction] = []
    owner_name = profile["name"]
    for _ in range(profile["n_txs"]):
        ts = _gen_timestamp(profile["hours"], profile["dow"])
        amount = _gen_amount(*profile["amount"])
        currency = random.choice(profile["currs"])
        ttype = TransactionType(random.choice(profile["types"]))
        cp_name, cp_acct = _gen_counterparty()

        d = profile["direction"]
        if d == "mixed":
            d = random.choice(["sender", "receiver"])

        if d == "sender":
            sender_name, sender_acct = owner_name, profile["acct"]
            receiver_name, receiver_acct = cp_name, cp_acct
        else:
            sender_name, sender_acct = cp_name, cp_acct
            receiver_name, receiver_acct = owner_name, profile["acct"]

        out.append(
            Transaction(
                id=str(uuid.uuid4()),
                sender_name=sender_name,
                sender_account=sender_acct,
                receiver_name=receiver_name,
                receiver_account=receiver_acct,
                amount=amount,
                currency=currency,
                type=ttype,
                status=TransactionStatus.PENDING,
                risk_score=None,
                smurfing_score=None,
                structuring_score=None,
                layering_score=None,
                created_at=ts,
            )
        )
    return out


def _build_singleton_noise(n: int = 30) -> List[Transaction]:
    """One-off pairs of random accounts. These won't be backfilled (warming-up
    by definition); they exist purely so the dashboard has some random graph
    noise and the profile signatures don't look isolated."""
    out: List[Transaction] = []
    for _ in range(n):
        amount = round(random.uniform(20, 5000), 2)
        currency = random.choice(["EUR", "USD", "TND"])
        ttype = TransactionType(random.choice(["TRANSFER", "DEPOSIT", "WITHDRAWAL"]))
        s_name, s_acct = _gen_counterparty()
        r_name, r_acct = _gen_counterparty()
        while r_acct == s_acct:
            r_name, r_acct = _gen_counterparty()
        ts = _gen_timestamp("mixed", "mixed")
        out.append(
            Transaction(
                id=str(uuid.uuid4()),
                sender_name=s_name, sender_account=s_acct,
                receiver_name=r_name, receiver_account=r_acct,
                amount=amount, currency=currency, type=ttype,
                status=TransactionStatus.PENDING,
                risk_score=None,
                smurfing_score=None, structuring_score=None, layering_score=None,
                created_at=ts,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ─────────────────────────────────────────────────────────────────────────

USERS_SPEC = [
    ("Fares Admin",    "admin@aml.com",    "Admin123",    UserRole.ADMIN),
    ("Sara Analyst",   "analyst@aml.com",  "Analyst123",  UserRole.ANALYST),
    ("Karim Auditor",  "auditor@aml.com",  "Auditor123",  UserRole.AUDITOR),
]


def seed_users(db) -> List[User]:
    users = []
    for name, email, pw, role in USERS_SPEC:
        u = User(
            id=str(uuid.uuid4()),
            name=name, email=email,
            password_hash=hash_password(pw),
            role=role, is_active=True, failed_attempts=0,
            created_at=_now() - timedelta(days=60),
        )
        db.add(u)
        users.append(u)
    db.commit()
    return users


def seed_transactions(db) -> int:
    count = 0
    for profile in PROFILES:
        rows = _build_profile_txs(profile)
        for tx in rows:
            db.add(tx)
        count += len(rows)
    for tx in _build_singleton_noise(30):
        db.add(tx)
        count += 1
    db.commit()
    return count


def _is_db_empty(db) -> bool:
    """First-boot detector. We key on the users table because new users
    are always created at seed time; empty users => fresh database."""
    return db.query(User).count() == 0


def seed_if_empty() -> None:
    """Idempotent auto-seed used by the FastAPI startup hook. No-ops if
    the users table is non-empty."""
    db = SessionLocal()
    try:
        if not _is_db_empty(db):
            return
        users = seed_users(db)
        n_tx = seed_transactions(db)
        print(f"[seed] auto-seeded {len(users)} users + {n_tx} transactions")
    finally:
        db.close()


def main() -> None:
    """CLI: wipe and reseed (used by `make seed`)."""
    db = SessionLocal()
    try:
        print("[seed] clearing existing data...")
        db.query(AuditLog).delete()
        db.query(Alert).delete()
        db.query(Transaction).delete()
        db.query(User).delete()
        db.commit()

        users = seed_users(db)
        n_tx = seed_transactions(db)
        print(f"[seed] created {len(users)} users + {n_tx} transactions")
        print("       login: admin@aml.com / Admin123  "
              "(analyst@aml.com / Analyst123, auditor@aml.com / Auditor123)")
    finally:
        db.close()


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    main()

import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.database import engine, Base
from app.models import user, transaction, alert, audit_log
from app.routes import auth, transactions, alerts, audit_logs, reports, users
from fastapi.middleware.cors import CORSMiddleware

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ANTI MONEY LAUNDERING PLATFORM",
    description="Our platform offers an innovative and AI powered solution to help banks and financial institutions to combat money laundering more efficiently ",
    version="1.0",
)

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "BACKEND_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again later."
        },
    )


Base.metadata.create_all(bind=engine)


def _ensure_alert_columns() -> None:
    """Add SAR columns to an existing alerts table on SQLite (idempotent).

    Reason: create_all() only creates missing tables, not missing columns.
    A pre-existing aml.db from before the AI integration would lack
    sar_en/sar_fr/verdict/rule_hits and crash the transactions route.
    """
    try:
        from sqlalchemy import inspect, text
        insp = inspect(engine)
        if "alerts" not in insp.get_table_names():
            return
        existing = {c["name"] for c in insp.get_columns("alerts")}
        wanted = {
            "sar_en": "TEXT",
            "sar_fr": "TEXT",
            "verdict": "VARCHAR(16)",
            "rule_hits": "JSON",
            # SAR Reports tab (bug #4) — track lifecycle on the alert row.
            "sar_status": "VARCHAR(16)",
            "sar_generated_at": "DATETIME",
            "sar_submitted_at": "DATETIME",
        }
        with engine.begin() as conn:
            for col, ddl in wanted.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE alerts ADD COLUMN {col} {ddl}"))
    except Exception:
        traceback.print_exc()


_ensure_alert_columns()


def _seed_if_empty() -> None:
    """First-boot demo data: if the users table is empty, run seed.py.
    No-ops on subsequent restarts (users already exist)."""
    try:
        import sys
        sys.path.insert(0, "/app")
        from seed import seed_if_empty as _do_seed
        _do_seed()
    except Exception:
        traceback.print_exc()


_seed_if_empty()


# Periodic rescoring of NULL risk_score rows. See app/jobs/rescore_scheduler.py
# for the bug-fix rationale.
try:
    from app.jobs.rescore_scheduler import start_scheduler, stop_scheduler

    @app.on_event("startup")
    def _start_rescore_scheduler() -> None:  # pragma: no cover
        start_scheduler()

    @app.on_event("shutdown")
    def _stop_rescore_scheduler() -> None:  # pragma: no cover
        stop_scheduler()
except Exception:
    traceback.print_exc()


app.include_router(auth.router)
app.include_router(users.router)
app.include_router(transactions.router)
app.include_router(alerts.router)
app.include_router(audit_logs.router)
app.include_router(reports.router)


@app.get("/")
def root():
    return {"message": "AML Platform is running", "status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

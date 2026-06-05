"""
backend/app/services/ai_client.py
==================================
Async httpx client for the AI microservice.

Behaviour:
  • /detect       → 10s timeout, 2x retries with exponential backoff
  • /investigate  → 60s timeout, 2x retries with exponential backoff
  • /healthz      → 2s timeout, no retries

If BACKEND_AI_BASE_URL is unset OR the AI service is unreachable, callers
can fall back to the local risk_scorer stub by catching AIServiceUnavailable.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.schemas.ai import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    InvestigateRequest,
    InvestigateResponse,
)

logger = logging.getLogger("backend.ai_client")

AI_BASE_URL: Optional[str] = os.getenv("BACKEND_AI_BASE_URL")
DETECT_TIMEOUT = float(os.getenv("BACKEND_AI_DETECT_TIMEOUT_SECONDS", "10"))
INVESTIGATE_TIMEOUT = float(os.getenv("BACKEND_AI_INVESTIGATE_TIMEOUT_SECONDS", "60"))


class AIServiceUnavailable(Exception):
    """Raised when the AI service cannot be reached or returns 5xx after retries."""


def _retry_policy() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(3),  # initial + 2 retries
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError, httpx.TimeoutException)
        ),
        reraise=True,
    )


def _ensure_configured() -> str:
    if not AI_BASE_URL:
        raise AIServiceUnavailable("BACKEND_AI_BASE_URL is not configured")
    return AI_BASE_URL


async def detect(req: DetectRequest) -> DetectResponse:
    base = _ensure_configured()
    try:
        async for attempt in _retry_policy():
            with attempt:
                async with httpx.AsyncClient(timeout=DETECT_TIMEOUT) as client:
                    resp = await client.post(f"{base}/detect", json=req.model_dump())
                    resp.raise_for_status()
                    return DetectResponse(**resp.json())
    except (RetryError, httpx.HTTPError) as exc:
        logger.warning("AI /detect failed: %s", exc)
        raise AIServiceUnavailable(str(exc)) from exc
    raise AIServiceUnavailable("retry policy exited without producing a response")


async def investigate(req: InvestigateRequest) -> InvestigateResponse:
    base = _ensure_configured()
    try:
        async for attempt in _retry_policy():
            with attempt:
                async with httpx.AsyncClient(timeout=INVESTIGATE_TIMEOUT) as client:
                    resp = await client.post(
                        f"{base}/investigate", json=req.model_dump()
                    )
                    resp.raise_for_status()
                    return InvestigateResponse(**resp.json())
    except (RetryError, httpx.HTTPError) as exc:
        logger.warning("AI /investigate failed: %s", exc)
        raise AIServiceUnavailable(str(exc)) from exc
    raise AIServiceUnavailable("retry policy exited without producing a response")


async def healthz() -> Optional[HealthResponse]:
    if not AI_BASE_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{AI_BASE_URL}/healthz")
            if resp.status_code == 200:
                return HealthResponse(**resp.json())
    except httpx.HTTPError:
        return None
    return None

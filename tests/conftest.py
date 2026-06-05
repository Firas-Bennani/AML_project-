"""tests/conftest.py
==================
Shared pytest fixtures. When running inside a container (detected by
`DOCKER_CONTAINER` env var), use container network DNS names. Otherwise use localhost.

Override URLs via env vars when running against a non-default deployment:
    E2E_BACKEND_URL=http://my-host:8000
    E2E_AI_URL=http://my-host:8001
"""
import os

import pytest


@pytest.fixture(scope="session")
def backend_url() -> str:
    env_url = os.getenv("E2E_BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")
    # Inside container: use service DNS name; outside: use localhost
    if os.getenv("DOCKER_CONTAINER"):
        return "http://backend:8000"
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def ai_url() -> str:
    env_url = os.getenv("E2E_AI_URL")
    if env_url:
        return env_url.rstrip("/")
    # Inside container: use service DNS name; outside: use localhost
    if os.getenv("DOCKER_CONTAINER"):
        return "http://ai:8001"
    return "http://localhost:8001"

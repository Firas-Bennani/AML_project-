# =============================================================================
# AML Hybrid Platform — Convenience Makefile
# =============================================================================
# Requires Docker Desktop (or compatible) on PATH.
# .env must exist (cp .env.example .env first).

.PHONY: help env up down logs build ps restart wait health test test-e2e seed clean backfill backfill-scoring backfill-alerts

help:
	@echo "AML Hybrid Platform"
	@echo "  make env        Copy .env.example to .env if missing"
	@echo "  make build      Build all docker images"
	@echo "  make up         Start the full stack (detached)"
	@echo "  make down       Stop and remove containers"
	@echo "  make logs       Tail logs from all services"
	@echo "  make ps         Show service status"
	@echo "  make wait       Wait until ai+backend+frontend are healthy"
	@echo "  make health     Curl healthz on ai and backend"
	@echo "  make seed       Run backend/seed.py inside the backend container"
	@echo "  make backfill   Backfill risk scores + alerts (bug #2 + #3)"
	@echo "  make test       Run unit tests (pytest in backend/tests)"
	@echo "  make test-e2e   Run end-to-end smoke test (tests/e2e)"
	@echo "  make clean      Remove containers, volumes, and images"

env:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from template"; else echo ".env already exists"; fi

build:
	docker compose build

up: env
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

ps:
	docker compose ps

wait:
	@echo "Waiting for ai + backend + frontend to be healthy..."
	@for i in $$(seq 1 30); do \
		ai=$$(docker inspect -f '{{.State.Health.Status}}' aml-ai 2>/dev/null || echo "starting"); \
		be=$$(docker inspect -f '{{.State.Health.Status}}' aml-backend 2>/dev/null || echo "starting"); \
		fe=$$(docker inspect -f '{{.State.Health.Status}}' aml-frontend 2>/dev/null || echo "starting"); \
		echo "  ai=$$ai backend=$$be frontend=$$fe"; \
		[ "$$ai" = "healthy" ] && [ "$$be" = "healthy" ] && [ "$$fe" = "healthy" ] && echo "All healthy." && exit 0; \
		sleep 3; \
	done; \
	echo "Timed out waiting for healthchecks." && exit 1

health:
	curl -fsS http://localhost:$${AI_PORT:-8001}/healthz && echo
	curl -fsS http://localhost:$${BACKEND_PORT:-8000}/healthz && echo

seed:
	docker compose exec backend python seed.py

backfill-scoring:
	docker compose exec backend python -m scripts.backfill_scoring

backfill-alerts:
	docker compose exec backend python -m scripts.backfill_alerts

backfill: backfill-scoring backfill-alerts

test:
	docker compose exec backend python -m pytest -q

test-e2e:
	@python -c "import httpx, pytest" 2>/dev/null || pip install -r tests/requirements.txt
	python -m pytest -v -s tests/e2e

clean:
	docker compose down -v --rmi local

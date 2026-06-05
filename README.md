# AML Hybrid Platform

> Hybrid platform for anti-money-laundering detection and investigation,
> combining a Graph Neural Network detector and an LLM-based investigation
> agent, orchestrated as containerized microservices.

## Overview

Money laundering moves an estimated **2–5 trillion USD** through the global
financial system every year. The rule-based transaction-monitoring systems
most banks still run flag huge volumes of activity but are notoriously
imprecise — **95–98% of the alerts they raise are false positives**, which
buries real cases under manual review backlogs and inflates compliance cost.

This platform is a demonstrator for a different approach, built in three
layers: (1) an **AI detector** that scores transactions for laundering
typologies using a graph neural network rather than static rules; (2) an
**LLM-based investigation agent** that, for a flagged entity, gathers context,
searches internal knowledge, reasons about the case, and drafts a bilingual
Suspicious Activity Report; and (3) a **web platform** (FastAPI backend +
React frontend) where analysts triage alerts, inspect transactions, and
validate reports before any external submission.

The deliverable is a production-shaped, locally runnable demonstrator: every
component ships with a Dockerfile and the whole stack comes up from a single
`docker-compose.yml`. The detector runs a real trained model; the
investigation agent runs end-to-end in a mocked-external mode by default so
the system works without any third-party credentials.

## Architecture

Three layers, wired as HTTP microservices. The browser talks only to the
frontend; the frontend proxies `/api/*` to the backend; the backend calls the
AI service over async HTTP.

```
Browser :3000 ─► frontend (Vite, container :5173)
                     │  /api/* proxied to
                     ▼
                backend :8000  ── async httpx ──►  ai :8001
                     │                               ├── POST /detect      (real GNN)
                     ▼                               └── POST /investigate  (LangGraph agent,
                SQLite (volume                            mocked SAR EN+FR by default)
                 backend_data)
```

PostgreSQL, Redis, Neo4j, and Milvus (with etcd + MinIO) are provisioned in
the compose stack for the target architecture, but **v1 runs on SQLite** and
keeps the investigation agent in mocked-external mode (`AI_MOCK_EXTERNAL=true`).
Switching to the real Neo4j + Milvus + NIM pipeline is a documented follow-up.

### Core components

- **ai-detector** — `HeteroGraphSAGE` multi-label classifier (PyTorch +
  PyTorch Geometric), wrapped by a FastAPI service (`ai/service.py`).
- **ai-investigator** — LangGraph agent with 4 nodes (`fetch_context`,
  `rag_search`, `analyze`, `report`) producing a bilingual SAR draft.
- **backend** — FastAPI with OAuth2 / JWT auth (PyJWT + python-jose),
  SQLAlchemy 2.0, and an APScheduler-based rescoring job.
- **frontend** — React 19 + Vite + Tailwind CSS, Zustand state, axios client.
- **Data stores** — SQLite (active in v1); PostgreSQL, Redis, Neo4j, Milvus
  provisioned for the target architecture.
- **LLM** — NVIDIA NIM serving Llama 3 70B (on-premise; mocked by default).

## Tech stack

- **AI** — Python, PyTorch 2.3, PyTorch Geometric (HeteroGraphSAGE), LangGraph
  + LangChain, FastAPI, Pydantic.
- **Backend** — FastAPI, SQLAlchemy 2.0, PyJWT / python-jose (OAuth2 + JWT),
  httpx + tenacity (AI client with retry/backoff), APScheduler, SQLite.
- **Frontend** — React 19, Vite, Tailwind CSS, React Router, Zustand, axios,
  lucide-react.
- **Infrastructure** — Docker / Docker Compose; PostgreSQL, Redis, Neo4j,
  Milvus (etcd + MinIO) provisioned for the target architecture.
- **Observability** — container healthchecks on ai / backend / frontend; an
  append-only audit log with hash chaining in the backend.

## Getting started

### Prerequisites

- Docker Desktop with WSL2 (Windows) or Docker Engine + Compose v2 (Linux/macOS)
- At least 8 GB free RAM
- (Optional) NVIDIA GPU for fast inference — the default build uses CPU wheels

### Quick start with Docker Compose

```bash
git clone https://github.com/Firas-Bennani/AML_project-.git
cd AML_project-
cp .env.example .env      # then edit values for any service you enable
docker compose build      # first build pulls ~2 GB of PyTorch CPU wheels
docker compose up -d
```

Or use the bundled Makefile (it copies `.env` for you and adds healthcheck
waits):

```bash
make env       # copy .env.example -> .env if missing
make build     # build ai, backend, frontend images
make up        # start the full stack (detached)
make wait      # block until ai + backend + frontend report 'healthy'
make health    # curl /healthz on ai (8001) and backend (8000)
```

Wait ~60–90 seconds for all services to become healthy, then visit:

- Frontend: http://localhost:3000
- Backend Swagger (OpenAPI docs): http://localhost:8000/docs
- AI service health: http://localhost:8001/healthz

### First-time setup

```bash
make seed       # create demo users + sample data in the backend container
```

Default seed credentials (**DEV ONLY — change before any real deployment**):

| Role    | Email             | Password    |
|---------|-------------------|-------------|
| Admin   | admin@aml.com     | Admin123    |
| Analyst | analyst@aml.com   | Analyst123  |
| Auditor | auditor@aml.com   | Auditor123  |

Optional backfill / validation:

```bash
make backfill    # backfill risk scores + alerts on existing data
make test-e2e    # register an analyst, post a flagged tx, assert a SAR is produced
```

## Project structure

```
.
├── ai/                     FastAPI service: HeteroGraphSAGE detector + LangGraph agent
│   ├── detection/          feature engineering, GNN detector, Triton inference
│   ├── investigation/      LangGraph agent + nodes (fetch_context/rag_search/analyze/report)
│   ├── service.py          HTTP API: /healthz, /detect, /investigate
│   ├── schemas.py          Pydantic wire contract (mirrored in backend)
│   ├── train.py / demo.py  model training / local demo
│   └── Dockerfile
├── backend/                FastAPI + SQLAlchemy + SQLite
│   ├── app/
│   │   ├── routes/         auth, transactions, alerts, reports, users, audit_logs
│   │   ├── services/       ai_client (async httpx+tenacity), scoring, sar_service
│   │   ├── models/         user, transaction, alert, audit_log
│   │   ├── jobs/           APScheduler rescoring
│   │   └── audit.py        hash-chained audit log
│   ├── scripts/            backfill + detector-probe utilities
│   ├── seed.py             demo users + sample data
│   └── Dockerfile
├── frontend/               React 19 + Vite + Tailwind (proxies /api to backend)
├── tests/                  unit + e2e (pytest + httpx) smoke tests
├── docker-compose.yml      ai · backend · frontend · postgres · redis · neo4j · milvus
├── Makefile                env / build / up / wait / health / seed / test / clean
├── .env.example            environment template (copy to .env)
└── README.md
```

## Dataset

Trained on **SAML-D** (Oztas et al., *IEEE ICEBE 2023*): 200,000 transactions
subsampled from the full 9.5M-row set, 12 laundering typologies grouped into
3 detection targets (**smurfing, structuring, layering**). The data is
strongly imbalanced — roughly **0.17% positive** samples — which the training
pipeline addresses with class weighting and per-typology threshold calibration.

> The raw dataset is **not** committed (gitignored); place SAML-D CSVs under
> `ai/data/` locally to retrain.

## Results

Per-typology recall at calibrated decision thresholds:

| Typology     | Recall | Threshold |
|--------------|--------|-----------|
| Structuring  | 1.00   | 0.80      |
| Layering     | 0.70   | 0.75      |
| Smurfing     | 0.64   | 0.75      |

Calibrated thresholds are saved alongside the model `state_dict` in the
checkpoint, so inference uses the same cutoffs the model was calibrated for.

## Compliance & ethics

- SAR templates aligned with **FinCEN Form 111** and **TRACFIN CERFA 10534**.
- LLM deployed **on-premise via NVIDIA NIM** — no client data leaves the
  perimeter.
- **Human-in-the-loop**: every decision requires analyst validation before any
  external submission (the investigation graph interrupts before review).
- Append-only **audit log with hash chaining** for tamper evidence.

## Roadmap

- Migrate to the **AMLworld** dataset for a richer typology set.
- Integrate **NeMo Guardrails** around the investigation LLM.
- Add a **service mesh** (Linkerd) for inter-service mTLS and observability.
- **GNNExplainer** for detector interpretability in the analyst UI.
- **ISO 20022** message connector for real transaction ingestion.

## Academic context

Final-year engineering project (Projet de Fin d'Année), 2025–2026.

## License

All rights reserved — academic project (PFA 2025–2026), not for redistribution.

## Author

**Firas Bennani** — _[LinkedIn](www.linkedin.com/in/firas-bennani-a57482160) / contact: firas.bennani29@gmail.com

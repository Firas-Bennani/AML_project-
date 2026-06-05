"""
ai/service.py
==============
FastAPI wrapper around the HeteroGraphSAGE detector and the (mocked)
LangGraph investigation pipeline. See ai/schemas.py for the wire contract.

Operating modes
---------------
AI_MOCK_EXTERNAL=true   (default) — /investigate returns deterministic mock
                                    SAR text in EN/FR. Mirrors demo.py's
                                    run_agent_investigation flow.
AI_MOCK_EXTERNAL=false             — currently 501 Not Implemented. Real
                                    Neo4j + Milvus + NIM wiring is a
                                    follow-up; agent.py also needs an
                                    auto-resume around its HITL interrupt.

/detect always uses the real GNN (random weights if no checkpoint loaded).
"""
from __future__ import annotations

import logging
import os
import sys
import textwrap
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

_AI_ROOT = Path(__file__).parent
sys.path.insert(0, str(_AI_ROOT))

# Reuse the training-time feature builder verbatim. The model was trained on
# this exact schema; using anything else here causes a silent feature-meaning
# mismatch. Do NOT swap for detection.feature_engineering.build_account_features —
# that's a different schema (see prior bug diagnosis).
from train import (  # noqa: E402
    build_account_features_from_df,
    build_edge_index,
)
from detection.gnn_detector import (  # noqa: E402
    GNNConfig,
    HeteroGraphSAGEDetector,
    TYPOLOGY_LABELS,
)
from schemas import (  # noqa: E402
    DetectRequest,
    DetectResponse,
    FlaggedAccount,
    HealthResponse,
    InvestigateRequest,
    InvestigateResponse,
    TransactionScore,
    TypologyScores,
)

logging.basicConfig(level=os.getenv("AI_LOG_LEVEL", "INFO"))
logger = logging.getLogger("ai.service")

MOCK_EXTERNAL = os.getenv("AI_MOCK_EXTERNAL", "true").lower() == "true"
CHECKPOINT_PATH = _AI_ROOT / os.getenv("AI_GNN_CHECKPOINT", "aml_model.pt")
TRIGGER_THRESHOLD = float(os.getenv("AI_GNN_TRIGGER_THRESHOLD", "0.45"))
_env_thresholds = os.getenv("AI_DETECT_THRESHOLDS")
DETECT_THRESHOLDS_OVERRIDE = (
    [float(x) for x in _env_thresholds.split(",")] if _env_thresholds else None
)

_state: Dict[str, object] = {"model": None, "model_loaded": False}


# ── Model lifecycle ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            f"Set AI_GNN_CHECKPOINT to a valid file matching the "
            f"HeteroGraphSAGEDetector schema (saved via train.py)."
        )
    try:
        model = HeteroGraphSAGEDetector.load(str(CHECKPOINT_PATH), device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {CHECKPOINT_PATH}: {exc}"
        ) from exc
    if DETECT_THRESHOLDS_OVERRIDE is not None:
        model.cfg.thresholds = DETECT_THRESHOLDS_OVERRIDE
        logger.info(
            "Threshold override from AI_DETECT_THRESHOLDS: %s",
            DETECT_THRESHOLDS_OVERRIDE,
        )
    else:
        logger.info("Using thresholds from checkpoint: %s", model.cfg.thresholds)
    model.eval()
    _state["model"] = model
    _state["model_loaded"] = True
    logger.info("Loaded checkpoint %s successfully", CHECKPOINT_PATH)
    yield


app = FastAPI(title="AML AI Service", version="1.0.0", lifespan=lifespan)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=bool(_state["model_loaded"]),
        mock_external=MOCK_EXTERNAL,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest) -> DetectResponse:
    """Score accounts on a window of transactions.

    IMPORTANT: this is an *account-level* model. The training schema computes
    per-account aggregates (count, avg, max, currency diversity, structuring
    proximity ratio, …) over a window of transactions. A request with a single
    transaction collapses 6 of 10 features to constants and the model returns
    its prior. Callers should send the involved accounts' recent history along
    with the new tx — typically the last 90 days or last 500 rows, whichever
    is smaller. Below ~3 prior txs per sender, scores are not reliable and the
    backend marks the transaction WARMING_UP.
    """
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialised")
    if not req.transactions:
        return DetectResponse(scores=[], flagged_accounts=[], threshold_used=TRIGGER_THRESHOLD)

    # Build a SAML-D-shaped DataFrame from the request and run it through the
    # training-time feature builder. Everything downstream of this point — the
    # account feature schema, the edge_index_dict layout, the dummy customer/
    # transaction tensors — must match what train.py fed the model. See the
    # mismatch diagnosis in repo history if tempted to "simplify" this.
    df = pd.DataFrame([
        {
            "Sender_account":         tx.sender_account,
            "Receiver_account":       tx.receiver_account,
            "Amount":                 float(tx.amount),
            "Payment_currency":       tx.currency,
            # We don't carry sender/receiver bank locations on the wire, so
            # cross_border_ratio degrades to 0. That's a feature lost, not a
            # crash.
            "Sender_bank_location":   "UNKNOWN",
            "Receiver_bank_location": "UNKNOWN",
            "Payment_type":           tx.type,
        }
        for tx in req.transactions
    ])

    feats, acc_idx = build_account_features_from_df(df)
    edge_index = build_edge_index(df, acc_idx)

    x_account = torch.tensor(feats, dtype=torch.float32)
    # Customer and transaction nodes were *always* dummy zeros at training, and
    # their edges into account were *always* empty. Reproduce that exactly.
    x_customer    = torch.zeros(1, 8,  dtype=torch.float32)
    x_transaction = torch.zeros(1, 10, dtype=torch.float32)

    x_dict = {
        "customer":    x_customer,
        "account":     x_account,
        "transaction": x_transaction,
    }
    edge_index_dict = {
        ("customer",  "transfer",     "account"):     torch.zeros(2, 0, dtype=torch.long),
        ("account",   "transfer",     "account"):     edge_index,
        ("account",   "transfer",     "transaction"): torch.zeros(2, 0, dtype=torch.long),
        ("customer",  "shared_ip",    "customer"):    torch.zeros(2, 0, dtype=torch.long),
        ("customer",  "shared_phone", "customer"):    torch.zeros(2, 0, dtype=torch.long),
    }

    with torch.no_grad():
        proba = model.predict_proba(x_dict, edge_index_dict)  # [n_acc, 3]

    thresholds = torch.tensor(model.cfg.thresholds)

    # Per-tx risk = max over (sender_max_typology, receiver_max_typology).
    # We pick the typology breakdown from whichever side carried the higher
    # max — so the breakdown stays internally consistent with the risk_score.
    scores: List[TransactionScore] = []
    for tx in req.transactions:
        s_send = proba[acc_idx[tx.sender_account]]
        s_recv = proba[acc_idx[tx.receiver_account]]
        s = s_send if float(s_send.max()) >= float(s_recv.max()) else s_recv
        scores.append(
            TransactionScore(
                transaction_id=tx.id,
                risk_score=float(s.max().item()),
                typologies=TypologyScores(
                    smurfing=float(s[0].item()),
                    structuring=float(s[1].item()),
                    layering=float(s[2].item()),
                ),
            )
        )

    flagged: List[FlaggedAccount] = []
    for acc, idx in acc_idx.items():
        s = proba[idx]
        if bool((s >= thresholds).any().item()):
            flagged.append(
                FlaggedAccount(
                    account_id=acc,
                    risk_score=float(s.max().item()),
                    dominant_typology=TYPOLOGY_LABELS[int(s.argmax().item())],
                    typologies=TypologyScores(
                        smurfing=float(s[0].item()),
                        structuring=float(s[1].item()),
                        layering=float(s[2].item()),
                    ),
                )
            )

    return DetectResponse(
        scores=scores, flagged_accounts=flagged, threshold_used=TRIGGER_THRESHOLD
    )


@app.post("/investigate", response_model=InvestigateResponse)
async def investigate(req: InvestigateRequest) -> InvestigateResponse:
    if not MOCK_EXTERNAL:
        raise HTTPException(
            status_code=501,
            detail=(
                "Real-services investigation not implemented in v1. "
                "Set AI_MOCK_EXTERNAL=true."
            ),
        )

    node_id = req.node_id
    score = req.risk_score
    typo = req.typology_scores

    sar_en = textwrap.dedent(
        f"""
        [SUBJECT]
        Account {node_id} is held by a customer classified as HIGH RISK under our
        Customer Due Diligence framework. The account is associated with a network
        showing structuring and smurfing indicators consistent with a coordinated
        mule operation.

        [ACTIVITY]
        The GNN risk model assigned a suspicion score of {score:.3f}. Typology
        breakdown: smurfing={typo.smurfing:.3f}, structuring={typo.structuring:.3f},
        layering={typo.layering:.3f}. Transaction amounts cluster near the USD
        10,000 Currency Transaction Report threshold, suggesting deliberate
        sub-threshold deposits.

        [EVIDENCE]
        - 2-hop subgraph reveals counter-parties sharing IP infrastructure.
        - Average transaction amount sits within the FinCEN structuring proximity
          band (USD 9,000–10,000).
        - KYC profile flagged for enhanced due diligence (FATF-monitored
          jurisdiction).

        [RECOMMENDATION]
        Immediate account freeze pending law enforcement liaison. SAR to be
        transmitted to FinCEN within 30 days per 31 CFR §1020.320.

        This SAR is filed in accordance with 31 CFR §1020.320.
        """
    ).strip()

    sar_fr = textwrap.dedent(
        f"""
        [SUJET]
        Le compte {node_id} est détenu par un client classé RISQUE ÉLEVÉ dans
        notre dispositif de vigilance client (LCB-FT). Le compte est associé à
        un réseau présentant des indicateurs de structuration et de smurfing
        caractéristiques d'une opération coordonnée de mules financières.

        [ACTIVITÉ]
        Le modèle GNN a attribué un score de suspicion de {score:.3f}.
        Détail typologique : smurfing={typo.smurfing:.3f},
        structuration={typo.structuring:.3f},
        blanchiment en couches={typo.layering:.3f}. Les montants se concentrent
        autour du seuil de déclaration de 10 000 USD.

        [PREUVES]
        - Sous-graphe à 2 sauts révélant des contreparties partageant la même
          infrastructure IP.
        - Montants moyens dans la bande de proximité FinCEN (9 000–10 000 USD).
        - Profil KYC marqué pour vigilance renforcée.

        [RECOMMANDATION]
        Gel immédiat du compte dans l'attente d'une liaison avec les autorités
        judiciaires. Déclaration de soupçon transmise à TRACFIN conformément à
        l'article L561-15 du Code Monétaire et Financier.

        Cette déclaration est déposée conformément à l'article L561-15 du CMF.
        """
    ).strip()

    if score >= 0.85:
        risk_level = "CRITICAL"
    elif score >= 0.6:
        risk_level = "HIGH"
    elif score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    verdict = "VERIFIED" if score >= 0.5 else "ESCALATE"

    return InvestigateResponse(
        report_id=f"SAR-{uuid.uuid4().hex[:12].upper()}",
        node_id=node_id,
        verdict=verdict,
        risk_level=risk_level,
        sar_en=sar_en,
        sar_fr=sar_fr,
        rule_hits=["FATF Rec. 20", "31 CFR §1020.320", "FATF Rec. 10 (CDD)"],
        evidence_refs=["KYC-MOCK-001", "NEWS-MOCK-001"],
        status="PENDING_REVIEW",
    )

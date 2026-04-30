"""
demo.py — AML Hybrid System End-to-End Demo
============================================
Runs the full pipeline without external services:
  1. Builds a synthetic heterogeneous transaction graph
  2. Constructs AML node features via feature_engineering
  3. Instantiates & runs HeteroGraphSAGEDetector (CPU)
  4. Flags suspicious account nodes
  5. Runs AMLInvestigatorAgent with MOCKED Neo4j / Milvus / LLM
  6. Prints a bilingual SAR draft to the console

No GPU, no Neo4j, no Milvus, no NIM API key required.
"""

from __future__ import annotations

import asyncio
import random
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import torch

# ── make root importable ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from detection.feature_engineering import (
    build_account_features,
    build_customer_features,
    build_transaction_features,
    TYPOLOGY_RULES,
)
from detection.gnn_detector import (
    GNNConfig,
    HeteroGraphSAGEDetector,
    EDGE_TYPES,
    NODE_TYPES,
    TYPOLOGY_LABELS,
)
from investigation.state import AMLState, initial_state, NeighbourNode, RetrievedChunk


# ============================================================================
# ANSI colour helpers
# ============================================================================

RED    = "\033[91m"
YLW    = "\033[93m"
GRN    = "\033[92m"
CYN    = "\033[96m"
BLD    = "\033[1m"
DIM    = "\033[2m"
RST    = "\033[0m"

def banner(title: str, colour: str = CYN) -> None:
    width = 70
    print(f"\n{colour}{BLD}{'═' * width}{RST}")
    print(f"{colour}{BLD}  {title}{RST}")
    print(f"{colour}{BLD}{'═' * width}{RST}")

def section(title: str) -> None:
    print(f"\n{YLW}{BLD}▶ {title}{RST}")
    print(f"{DIM}{'─' * 60}{RST}")

def log(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RST}")


# ============================================================================
# STEP 1 — Synthetic heterogeneous graph
# ============================================================================

random.seed(42)
torch.manual_seed(42)

N_CUSTOMERS    = 20
N_ACCOUNTS     = 30
N_TRANSACTIONS = 60

def build_synthetic_graph() -> Tuple[
    Dict[str, torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
]:
    """
    Build a small synthetic hetero graph that mimics a real AML scenario.

    Topology:
      • 10 customers connected to accounts via shared_ip (smurfing cluster)
      • 4 accounts with structuring amounts (near $9,500)
      • A 3-hop layering chain across 5 accounts
    """
    section("Building synthetic heterogeneous transaction graph")

    # ── Node features ────────────────────────────────────────────────────
    # Customer features [N_customers, 8]
    x_customer = build_customer_features(
        degree_in         = torch.randint(1, 15,  (N_CUSTOMERS,)).float(),
        degree_out        = torch.randint(1, 20,  (N_CUSTOMERS,)).float(),
        betweenness       = torch.rand(N_CUSTOMERS),
        shared_ip_count   = torch.cat([
            torch.tensor([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),  # 10 smurfing peers
            torch.zeros(N_CUSTOMERS - 10)
        ]),
        shared_phone_count= torch.randint(0, 3, (N_CUSTOMERS,)).float(),
        kyc_risk_score    = torch.randint(20, 95, (N_CUSTOMERS,)).float(),
        pep_flag          = (torch.rand(N_CUSTOMERS) > 0.85).long(),
        country_risk      = torch.rand(N_CUSTOMERS),
    )

    # Account features [N_accounts, 10]
    # Accounts 0-4: high velocity + structuring amounts (suspicious)
    avg_amounts = torch.cat([
        torch.full((5,), 9_400.0),        # structuring cluster
        torch.rand(N_ACCOUNTS - 5) * 5000 + 500,
    ])
    tx_24h = torch.cat([
        torch.full((5,), 12.0),            # smurfing velocity
        torch.rand(N_ACCOUNTS - 5) * 4 + 1,
    ])

    x_account = build_account_features(
        balance_mean   = torch.rand(N_ACCOUNTS) * 50_000 + 1_000,
        balance_std    = torch.rand(N_ACCOUNTS) * 10_000,
        tx_count_24h   = tx_24h,
        tx_count_7d    = tx_24h * 7 + torch.rand(N_ACCOUNTS) * 3,
        avg_tx_amount  = avg_amounts,
        max_tx_amount  = avg_amounts * 1.05,
        incoming_ratio = torch.rand(N_ACCOUNTS),
        dormancy_days  = torch.cat([
            torch.zeros(5),
            torch.randint(0, 90, (N_ACCOUNTS - 5,)).float(),
        ]),
    )

    # Transaction features [N_transactions, 10]
    # Transactions 0-9: near-threshold amounts (structuring signal)
    tx_amounts = torch.cat([
        torch.full((10,), 9_500.0),
        torch.rand(N_TRANSACTIONS - 10) * 8_000 + 100,
    ])
    x_transaction = build_transaction_features(
        amount         = tx_amounts,
        timestamp_hour = torch.randint(0, 24, (N_TRANSACTIONS,)).float(),
        timestamp_dow  = torch.randint(0, 7,  (N_TRANSACTIONS,)).float(),
        is_round_amount= (tx_amounts % 500 == 0).long(),
        cross_border   = (torch.rand(N_TRANSACTIONS) > 0.7).long(),
        is_reversal    = (torch.rand(N_TRANSACTIONS) > 0.9).long(),
    )

    x_dict = {
        "customer":    x_customer,
        "account":     x_account,
        "transaction": x_transaction,
    }

    # ── Edge indices ─────────────────────────────────────────────────────
    # customer → account (transfer)
    cust_src = torch.randint(0, N_CUSTOMERS,    (N_ACCOUNTS,))
    acc_dst  = torch.arange(N_ACCOUNTS)

    # account → transaction (transfer)
    acc_src  = torch.randint(0, N_ACCOUNTS,     (N_TRANSACTIONS,))
    tx_dst   = torch.arange(N_TRANSACTIONS)

    # customer ↔ customer (shared_ip) — dense cluster among first 10
    ip_pairs = [(i, j) for i in range(10) for j in range(10) if i != j]
    ip_src   = torch.tensor([p[0] for p in ip_pairs])
    ip_dst   = torch.tensor([p[1] for p in ip_pairs])

    # customer ↔ customer (shared_phone) — 4 pairs
    ph_pairs = [(0, 1), (2, 3), (5, 6), (8, 9)]
    ph_src   = torch.tensor([p[0] for p in ph_pairs])
    ph_dst   = torch.tensor([p[1] for p in ph_pairs])

    # account → account (self-transfers) — needed for trained SAML-D model
    acct_src2 = torch.randint(0, N_ACCOUNTS, (N_ACCOUNTS * 2,))
    acct_dst2 = torch.randint(0, N_ACCOUNTS, (N_ACCOUNTS * 2,))

    edge_index_dict = {
        ("customer",  "transfer",     "account"):     torch.stack([cust_src, acc_dst]),
        ("account",   "transfer",     "account"):     torch.stack([acct_src2, acct_dst2]),
        ("account",   "transfer",     "transaction"): torch.stack([acc_src,  tx_dst]),
        ("customer",  "shared_ip",    "customer"):    torch.stack([ip_src,   ip_dst]),
        ("customer",  "shared_phone", "customer"):    torch.stack([ph_src,   ph_dst]),
    }

    print(f"  Customers    : {N_CUSTOMERS}")
    print(f"  Accounts     : {N_ACCOUNTS}")
    print(f"  Transactions : {N_TRANSACTIONS}")
    print(f"  Edge types   : {len(edge_index_dict)}")
    print(f"  Accounts 0-4 : ⚠ injected with structuring + velocity signals")

    return x_dict, edge_index_dict


# ============================================================================
# STEP 2 — Instantiate & run GNN
# ============================================================================

def run_gnn_detection(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    model_override=None,
) -> Dict[str, float]:

    section("Running HeteroGraphSAGEDetector")

    if model_override is not None:
        model = model_override
        log(f"  Using trained checkpoint | thresholds={model.cfg.thresholds}", GRN)
    else:
        cfg = GNNConfig(
            hidden_channels  = 64,
            num_layers       = 3,
            dropout          = 0.0,
            thresholds       = [0.35, 0.35, 0.35],
        )
        model = HeteroGraphSAGEDetector(cfg)
        log("  Using random weights (no checkpoint)", YLW)

    model.eval()

    cfg = model.cfg
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : HeteroGraphSAGE x {cfg.num_layers} layers")
    print(f"  Hidden dim   : {cfg.hidden_channels}")
    print(f"  Parameters   : {total_params:,}")
    print(f"  Device       : CPU")

    section("Running GNN inference")

    with torch.no_grad():
        proba = model.predict_proba(x_dict, edge_index_dict)   # [N_accounts, 3]

    print(f"\n  {'Account':<10} {'Smurfing':>10} {'Structuring':>13} {'Layering':>10}  {'Status'}")
    print(f"  {'─'*7:<10} {'─'*8:>10} {'─'*11:>13} {'─'*8:>10}  {'─'*10}")

    flagged: Dict[str, float] = {}
    thresholds = torch.tensor(model.cfg.thresholds)

    for i in range(N_ACCOUNTS):
        scores = proba[i]
        is_flagged = (scores >= thresholds).any().item()
        smurf_s  = scores[0].item()
        struct_s = scores[1].item()
        layer_s  = scores[2].item()
        status   = f"{RED}⚠ FLAGGED{RST}" if is_flagged else f"{GRN}✓ benign{RST}"
        highlight = YLW if is_flagged else DIM

        print(
            f"  {highlight}ACC-{i:03d}{RST:<9} "
            f"{highlight}{smurf_s:>10.4f}{RST} "
            f"{highlight}{struct_s:>13.4f}{RST} "
            f"{highlight}{layer_s:>10.4f}{RST}  {status}"
        )
        if is_flagged:
            dominant = TYPOLOGY_LABELS[scores.argmax().item()]
            flagged[f"ACC-{i:03d}"] = {
                "smurfing":    round(smurf_s,  4),
                "structuring": round(struct_s, 4),
                "layering":    round(layer_s,  4),
                "dominant":    dominant,
                "risk_score":  round(scores.max().item(), 4),
            }

    print(f"\n  {RED}{BLD}Flagged accounts: {len(flagged)}{RST}")
    return flagged


# ============================================================================
# STEP 3 — Mock external services
# ============================================================================

def _mock_neo4j_neighbourhood(node_id: str) -> List[NeighbourNode]:
    """Simulate 2-hop Neo4j query results for demo."""
    return [
        NeighbourNode(
            node_id="ACC-001", node_type="Account",
            relationship="TRANSFER", properties={"amount": 9500, "currency": "USD"}, hop=1,
        ),
        NeighbourNode(
            node_id="CUST-007", node_type="Customer",
            relationship="SHARED_IP", properties={"ip": "192.168.1.44", "country": "NG"}, hop=1,
        ),
        NeighbourNode(
            node_id="ACC-009", node_type="Account",
            relationship="TRANSFER", properties={"amount": 9200, "currency": "USD"}, hop=2,
        ),
        NeighbourNode(
            node_id="CUST-003", node_type="Customer",
            relationship="SHARED_PHONE", properties={"phone_hash": "a3f9d"}, hop=2,
        ),
    ]


def _mock_kyc_chunks(node_id: str) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id="KYC-2024-001", source_type="kyc_document",
            content=(
                f"Customer onboarding file for account {node_id}. "
                "Occupation: Import/Export Trader. Country of residence: Nigeria. "
                "Source of funds declared: Business revenues. "
                "Enhanced Due Diligence flag: HIGH risk jurisdiction (FATF grey list)."
            ),
            score=0.92,
            metadata={"jurisdiction": "NG", "doc_type": "CDD_FORM"},
        ),
        RetrievedChunk(
            doc_id="KYC-2024-002", source_type="kyc_document",
            content=(
                "Beneficial owner declared as sole proprietor. "
                "No PEP status confirmed at onboarding. "
                "Registered address: Lagos Free Trade Zone. "
                "Monthly turnover declared: $15,000 USD."
            ),
            score=0.87,
            metadata={"jurisdiction": "NG", "doc_type": "BENEFICIAL_OWNER"},
        ),
    ]


def _mock_news_chunks() -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id="NEWS-2024-551", source_type="negative_news",
            content=(
                "Nigerian business network linked to multiple sub-threshold "
                "cash deposits across EU financial institutions. Europol issued "
                "advisory on coordinated structuring activity. Accounts share "
                "common IP infrastructure."
            ),
            score=0.83,
            metadata={"entity_name": "Lagos Trade Network", "article_date": "2024-08-15"},
        ),
    ]


def _mock_llm_analyze(state: AMLState) -> str:
    """Simulate LLM analysis response."""
    return """{
  "risk_verdict": "VERIFIED",
  "risk_level": "HIGH",
  "risk_rationale": "The GNN model flags this account with high smurfing and structuring scores. The 2-hop subgraph reveals 10 co-located accounts sharing the same IP (192.168.1.44), consistent with a coordinated mule network. Average transaction amount of $9,400 USD sits within the FinCEN structuring proximity band ($9,000–$10,000). Negative news confirms a Europol advisory against a related network. KYC risk score is elevated (FATF grey-list jurisdiction). Rule triggers: FATF Recommendation 20 (STR obligation) and 31 CFR §1020.320 (SAR filing requirement).",
  "rule_hits": ["FATF Rec. 20", "31 CFR §1020.320", "FATF Rec. 10 (CDD)"]
}"""


def _mock_llm_sar_en(state: AMLState) -> str:
    node_id = state["node_id"]
    score   = state["risk_score"]
    return textwrap.dedent(f"""
    [SUBJECT]
    Account {node_id} is held by a customer classified as HIGH RISK under our
    Customer Due Diligence framework. The account is domiciled in a FATF grey-list
    jurisdiction (Nigeria) and the declared source of funds is Import/Export trading.

    [ACTIVITY]
    Between January and August 2024, the account conducted 12 cash deposits within
    a 24-hour period, each below the USD 10,000 Currency Transaction Report threshold
    (average: USD 9,400). The GNN risk model assigned a suspicion score of {score:.3f}.
    The account shares an IP address (192.168.1.44) with 9 other accounts — a topology
    consistent with coordinated structuring by a mule network.

    [EVIDENCE]
    - GNN typology scores: Smurfing={state['typology_scores'].get('smurfing',0):.3f},
      Structuring={state['typology_scores'].get('structuring',0):.3f},
      Layering={state['typology_scores'].get('layering',0):.3f}
    - Europol advisory (2024-08-15) references co-located accounts.
    - KYC file: EDD flag, FATF grey-list jurisdiction.

    [RECOMMENDATION]
    Immediate account freeze pending law enforcement liaison. SAR to be transmitted
    to FinCEN within 30 days per 31 CFR §1020.320. Parallel referral to TRACFIN
    recommended given EU-linked counterparties.

    This SAR is filed in accordance with 31 CFR §1020.320.
    """).strip()


def _mock_llm_sar_fr(state: AMLState) -> str:
    node_id = state["node_id"]
    return textwrap.dedent(f"""
    [SUJET]
    Le compte {node_id} est détenu par un client classé RISQUE ÉLEVÉ dans notre
    dispositif de vigilance client (LCB-FT). Le compte est domicilié dans une
    juridiction sous surveillance GAFI (Nigéria) et la source des fonds déclarée
    est le commerce import/export.

    [ACTIVITÉ]
    Entre janvier et août 2024, le compte a effectué 12 dépôts en espèces en
    moins de 24 heures, chacun inférieur au seuil de déclaration de 10 000 USD
    (montant moyen : 9 400 USD). Le modèle GNN a attribué un score de suspicion
    de {state['risk_score']:.3f}. Le compte partage une adresse IP (192.168.1.44)
    avec 9 autres comptes — topologie caractéristique d'un réseau coordonné de
    mules financières (structuration collective).

    [PREUVES]
    - Scores typologiques GNN : Smurfing={state['typology_scores'].get('smurfing',0):.3f},
      Structuration={state['typology_scores'].get('structuring',0):.3f},
      Blanchiment en couches={state['typology_scores'].get('layering',0):.3f}
    - Avis Europol (15/08/2024) mentionnant des comptes co-localisés.
    - Dossier KYC : indicateur de vigilance renforcée, juridiction GAFI sous surveillance.

    [RECOMMANDATION]
    Gel immédiat du compte dans l'attente d'une liaison avec les autorités judiciaires.
    Déclaration de Soupçon transmise à TRACFIN conformément à l'article L561-15
    du Code Monétaire et Financier. Signalement parallèle recommandé auprès de
    FinCEN pour les contreparties américaines identifiées.

    Cette déclaration est déposée conformément à l'article L561-15 du CMF.
    """).strip()


# ============================================================================
# STEP 4 — Simulated agent run
# ============================================================================

async def run_agent_investigation(
    node_id: str,
    risk_score: float,
    typology_scores: Dict[str, float],
) -> AMLState:
    """
    Run the AML investigation workflow with mocked external services.
    Executes each node function directly (no LangGraph server needed).
    """
    section(f"AML Investigation: {node_id}  (score={risk_score:.4f})")

    state = initial_state(
        node_id=node_id,
        risk_score=risk_score,
        typology_scores=typology_scores,
        thread_id=f"demo-{uuid.uuid4().hex[:6]}",
    )

    # ── Node 1: Fetch_Context ────────────────────────────────────────────
    print(f"\n  {CYN}[Node 1] fetch_context — Neo4j 2-hop query{RST}")
    neighbourhood = _mock_neo4j_neighbourhood(node_id)
    state["neighbourhood"] = neighbourhood
    state["graph_summary"] = (
        f"Account {node_id} 2-hop subgraph: "
        f"{len(neighbourhood)} nodes | "
        "HIGH OUT-DEGREE (12 transfers out) — possible Smurfing fan-out | "
        "SHARED IP with 9 accounts — possible synthetic identity cluster"
    )
    print(f"    Retrieved {len(neighbourhood)} neighbour nodes")
    for n in neighbourhood:
        print(f"    hop-{n['hop']}  {n['node_type']:<12} {n['node_id']:<12}  via {n['relationship']}")

    # ── Node 2: RAG_Search ───────────────────────────────────────────────
    print(f"\n  {CYN}[Node 2] rag_search — Milvus KYC + News vector search{RST}")
    kyc_chunks  = _mock_kyc_chunks(node_id)
    news_chunks = _mock_news_chunks()
    state["kyc_chunks"]  = kyc_chunks
    state["news_chunks"] = news_chunks
    print(f"    KYC documents  : {len(kyc_chunks)} chunks")
    print(f"    Negative news  : {len(news_chunks)} articles")
    for c in kyc_chunks:
        print(f"    [{c['score']:.2f}] {c['content'][:80]}…")
    for c in news_chunks:
        print(f"    [{c['score']:.2f}] {c['content'][:80]}…")

    # ── Node 3: Analyze ──────────────────────────────────────────────────
    print(f"\n  {CYN}[Node 3] analyze — NIM LLM risk assessment{RST}")
    import json
    raw_analysis = _mock_llm_analyze(state)
    analysis = json.loads(raw_analysis)
    state["risk_verdict"]   = analysis["risk_verdict"]
    state["risk_level"]     = analysis["risk_level"]
    state["risk_rationale"] = analysis["risk_rationale"]
    state["rule_hits"]      = analysis["rule_hits"]
    verdict_colour = RED if state["risk_verdict"] == "VERIFIED" else GRN
    print(f"    Verdict   : {verdict_colour}{BLD}{state['risk_verdict']}{RST}")
    print(f"    Risk level: {RED}{state['risk_level']}{RST}")
    print(f"    Rule hits : {', '.join(state['rule_hits'])}")
    print(f"    Rationale : {state['risk_rationale'][:120]}…")

    # ── Node 4: Report ───────────────────────────────────────────────────
    print(f"\n  {CYN}[Node 4] report — Bilingual SAR generation (EN + FR){RST}")
    sar_en = _mock_llm_sar_en(state)
    sar_fr = _mock_llm_sar_fr(state)
    import uuid as _uuid
    sar_id = f"SAR-{_uuid.uuid4().hex[:12].upper()}"
    state["sar_report"] = {
        "report_id":    sar_id,
        "node_id":      node_id,
        "risk_score":   risk_score,
        "typologies":   [t for t, s in typology_scores.items() if s >= 0.35],
        "narrative_en": sar_en,
        "narrative_fr": sar_fr,
        "evidence_refs": [c["doc_id"] for c in kyc_chunks + news_chunks],
        "analyst_notes": "",
        "status": "PENDING_REVIEW",
    }
    print(f"    SAR ID : {BLD}{sar_id}{RST}")
    print(f"    Status : {YLW}PENDING_REVIEW{RST}")

    return state


# ============================================================================
# STEP 5 — Print final SAR
# ============================================================================

def print_sar(state: AMLState) -> None:
    sar = state["sar_report"]
    if not sar:
        return

    banner("SUSPICIOUS ACTIVITY REPORT", RED)
    print(f"\n  {BLD}Report ID    :{RST} {sar['report_id']}")
    print(f"  {BLD}Account      :{RST} {sar['node_id']}")
    print(f"  {BLD}Risk Score   :{RST} {sar['risk_score']:.4f}")
    print(f"  {BLD}Typologies   :{RST} {', '.join(sar['typologies'])}")
    print(f"  {BLD}Evidence Refs:{RST} {', '.join(sar['evidence_refs'])}")
    print(f"  {BLD}Status       :{RST} {YLW}{sar['status']}{RST}")

    print(f"\n{BLD}── English Narrative (FinCEN Form 111) ─────────────────────────────{RST}")
    for line in sar["narrative_en"].splitlines():
        print(f"  {line}")

    print(f"\n{BLD}── Narrative française (TRACFIN CERFA 10534) ───────────────────────{RST}")
    for line in sar["narrative_fr"].splitlines():
        print(f"  {line}")

    print(f"\n{GRN}{BLD}  ✓ SAR draft complete — status: PENDING_REVIEW{RST}")
    print(f"{DIM}  Next step: human analyst reviews and submits to FinCEN / TRACFIN{RST}")


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    banner("AML HYBRID SYSTEM -- End-to-End Demo", CYN)
    print(f"\n  {DIM}Detection   : HeteroGraphSAGE (PyTorch Geometric, CPU){RST}")
    print(f"  {DIM}Typologies  : Smurfing | Structuring | Layering{RST}")
    print(f"  {DIM}Agent       : LangGraph (mocked Neo4j + Milvus + NIM){RST}")
    print(f"  {DIM}SAR output  : English (FinCEN) + French (TRACFIN){RST}")

    # Step 1: synthetic graph
    x_dict, edge_index_dict = build_synthetic_graph()

    # Step 2: GNN detection — use trained model if checkpoint exists
    banner("DETECTION LAYER", YLW)
    checkpoint = Path(__file__).parent / "aml_model.pt"
    if checkpoint.exists():
        log(f"  Loading trained model from {checkpoint}", GRN)
        from detection.gnn_detector import HeteroGraphSAGEDetector as _Det
        trained_model = _Det.load(str(checkpoint), device="cpu")
        trained_model.cfg.thresholds = [0.45, 0.45, 0.45]
        flagged = run_gnn_detection(x_dict, edge_index_dict, model_override=trained_model)
    else:
        log("  No checkpoint found — using random weights (run train.py first)", YLW)
        flagged = run_gnn_detection(x_dict, edge_index_dict)

    if not flagged:
        print(f"\n{GRN}No accounts flagged. Try lowering thresholds in GNNConfig.{RST}")
        return

    # Step 3: investigate the highest-risk flagged account
    banner("INVESTIGATION LAYER", CYN)
    top_node_id = max(flagged, key=lambda k: flagged[k]["risk_score"])
    top_info    = flagged[top_node_id]

    print(f"\n  Investigating top-risk account: {RED}{BLD}{top_node_id}{RST}")
    print(f"  Risk score : {top_info['risk_score']:.4f}")
    print(f"  Dominant   : {top_info['dominant']}")

    state = await run_agent_investigation(
        node_id       = top_node_id,
        risk_score    = top_info["risk_score"],
        typology_scores = {
            "smurfing":    top_info["smurfing"],
            "structuring": top_info["structuring"],
            "layering":    top_info["layering"],
        },
    )

    # Step 4: print bilingual SAR
    print_sar(state)

    # Summary
    banner("DEMO COMPLETE", GRN)
    print(f"\n  Accounts analysed : {N_ACCOUNTS}")
    print(f"  Flagged           : {len(flagged)}")
    print(f"  SAR generated     : 1  ({state['sar_report']['report_id']})")
    print(f"\n  {DIM}To connect real services:{RST}")
    print(f"  {DIM}  1. Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env{RST}")
    print(f"  {DIM}  2. Set MILVUS_HOST / MILVUS_PORT in .env{RST}")
    print(f"  {DIM}  3. Set NVIDIA_NIM_API_KEY in .env{RST}")
    print(f"  {DIM}  4. Replace mock nodes with real FetchContextNode, RAGSearchNode, AnalyzeNode{RST}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""
train.py — SAML-D Dataset Training Pipeline
============================================
Trains the HeteroGraphSAGEDetector on the SAML-D transaction dataset.

Dataset: SAML-D.csv
  Columns: Time, Date, Sender_account, Receiver_account, Amount,
            Payment_currency, Received_currency, Sender_bank_location,
            Receiver_bank_location, Payment_type, Is_laundering, Laundering_type

Graph Construction:
  - Nodes: Account (unique sender/receiver IDs)
  - Edges: (sender_account, TRANSFER, receiver_account) per transaction
  - Transaction features embedded into SENDER account node features

Since SAML-D has account→account edges (no separate Customer/Transaction nodes),
we use a SIMPLIFIED homogeneous approach that maps onto our HeteroGraphSAGE:
  - All nodes are type "account"
  - Edge type: ("account", "transfer", "account")
  - Customer/Transaction node types get small dummy feature tensors
    (the model learns to weight them near-zero via the shared hidden space)

Label Mapping:
  Is_laundering=1 → suspicious
  Laundering_type mapped to 3 typology columns:
    [smurfing, structuring, layering]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))

from detection.gnn_detector import (
    GNNConfig,
    HeteroGraphSAGEDetector,
    AMLLoss,
    TYPOLOGY_LABELS,
)


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal-modulated BCE-with-logits with per-class pos_weight.
    Down-weights easy examples by (1 - p_t)^gamma so gradient focuses on the
    rare hard ones; pos_weight still handles the gross class imbalance.
    Lin et al. 2017, adapted for multi-label."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    with torch.no_grad():
        p = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, p, 1.0 - p)
        focal_term = (1.0 - p_t).clamp(min=1e-6) ** gamma
    return (focal_term * bce).mean()

# ── Colour helpers ──────────────────────────────────────────────────────────
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"
CYN = "\033[96m"; BLD = "\033[1m";  DIM = "\033[2m"; RST = "\033[0m"

def log(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RST}")

# ============================================================================
# CONFIG
# ============================================================================

_DEFAULT_CSV    = str(Path(__file__).parent / "data" / "SAML-D.csv" / "SAML-D.csv")
CSV_PATH        = os.getenv("AI_TRAIN_CSV", _DEFAULT_CSV)
CHECKPOINT_PATH = os.getenv("AI_TRAIN_CHECKPOINT", "aml_model.pt")
MAX_ROWS        = int(os.environ["AI_TRAIN_MAX_ROWS"]) if os.getenv("AI_TRAIN_MAX_ROWS") else 1_000_000
BATCH_EPOCHS    = int(os.getenv("AI_TRAIN_EPOCHS", "30"))
LR              = float(os.getenv("AI_TRAIN_LR", "3e-3"))
HIDDEN          = int(os.getenv("AI_TRAIN_HIDDEN", "64"))
DEVICE          = os.getenv("AI_TRAIN_DEVICE", "cpu")
FOCAL_GAMMA     = float(os.getenv("AI_TRAIN_FOCAL_GAMMA", "2.0"))

# Typology keyword → column index mapping
TYPOLOGY_MAP = {
    "smurfing":    0,
    "structuring": 1,
    "layering":    2,
}
# Keywords that map to each typology (from Laundering_type column)
TYPOLOGY_KEYWORDS = {
    0: ["smurfing", "fan_out", "fan_in", "scatter", "mutual"],      # smurfing
    1: ["structuring", "cash_deposit", "cash_withdrawal", "stacked"],  # structuring
    2: ["layer", "cycle", "foward", "forward", "over-invoicing",    # layering
        "behavioural", "bipartite"],
}


# ============================================================================
# STEP 1: Load & preprocess SAML-D
# ============================================================================

def load_dataset(path: str, max_rows: int | None) -> pd.DataFrame:
    log("\n[1/5] Loading SAML-D dataset...", CYN)
    df = pd.read_csv(path, nrows=max_rows)
    log(f"      Rows loaded : {len(df):,}", DIM)
    log(f"      Laundering  : {df['Is_laundering'].sum():,} ({df['Is_laundering'].mean()*100:.2f}%)", DIM)
    return df


def build_account_features_from_df(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Aggregate transaction-level rows into per-account node features.

    Features per account (10 total — matches NODE_IN_DIMS['account']):
      0  log1p(total_sent_amount)
      1  log1p(total_recv_amount)
      2  log1p(tx_count_sent)
      3  log1p(tx_count_recv)
      4  log1p(avg_sent_amount)
      5  log1p(max_sent_amount)
      6  cross_border_ratio        ← layering signal
      7  structuring_proximity     ← % txs near $9k-$10k
      8  currency_diversity        ← # distinct currencies used
      9  payment_type_entropy      ← diversity of payment methods
    """
    log("\n[2/5] Building per-account node features...", CYN)

    CTR_THRESH = 10_000.0
    STRUCT_LO  = 9_000.0

    # Map accounts to contiguous integer IDs
    all_accounts = pd.concat([df["Sender_account"], df["Receiver_account"]]).unique()
    acct_to_idx  = {a: i for i, a in enumerate(all_accounts)}
    n_accounts   = len(all_accounts)
    log(f"      Unique accounts: {n_accounts:,}", DIM)

    feats = np.zeros((n_accounts, 10), dtype=np.float32)

    # Sender-side aggregations (vectorized scatter into feats)
    sent = df.groupby("Sender_account").agg(
        total_sent=("Amount", "sum"),
        tx_count_sent=("Amount", "count"),
        avg_sent=("Amount", "mean"),
        max_sent=("Amount", "max"),
    ).reset_index()
    sidx = sent["Sender_account"].map(acct_to_idx).values
    feats[sidx, 0] = np.log1p(sent["total_sent"].values)
    feats[sidx, 2] = np.log1p(sent["tx_count_sent"].values)
    feats[sidx, 4] = np.log1p(sent["avg_sent"].values)
    feats[sidx, 5] = np.log1p(sent["max_sent"].values)

    # Receiver-side aggregations
    recv = df.groupby("Receiver_account").agg(
        total_recv=("Amount", "sum"),
        tx_count_recv=("Amount", "count"),
    ).reset_index()
    ridx = recv["Receiver_account"].map(acct_to_idx).values
    feats[ridx, 1] = np.log1p(recv["total_recv"].values)
    feats[ridx, 3] = np.log1p(recv["tx_count_recv"].values)

    # Cross-border ratio (per sender account)
    df["_cross"] = df["Sender_bank_location"] != df["Receiver_bank_location"]
    cross = df.groupby("Sender_account")["_cross"].mean().reset_index()
    cidx = cross["Sender_account"].map(acct_to_idx).values
    feats[cidx, 6] = cross["_cross"].astype(np.float32).values

    # Structuring proximity (% of sent txs in $9k-$10k band)
    df["_struct"] = (df["Amount"] >= STRUCT_LO) & (df["Amount"] < CTR_THRESH)
    struct = df.groupby("Sender_account")["_struct"].mean().reset_index()
    stidx = struct["Sender_account"].map(acct_to_idx).values
    feats[stidx, 7] = struct["_struct"].astype(np.float32).values

    # Currency diversity (# distinct currencies)
    cur_div = df.groupby("Sender_account")["Payment_currency"].nunique().reset_index()
    cuidx = cur_div["Sender_account"].map(acct_to_idx).values
    feats[cuidx, 8] = cur_div["Payment_currency"].astype(np.float32).values

    # Payment type entropy
    pt_div = df.groupby("Sender_account")["Payment_type"].nunique().reset_index()
    ptidx = pt_div["Sender_account"].map(acct_to_idx).values
    feats[ptidx, 9] = pt_div["Payment_type"].astype(np.float32).values

    return feats, acct_to_idx


def build_labels(df: pd.DataFrame, acct_to_idx: dict, n_accounts: int) -> np.ndarray:
    """
    Build multi-label matrix [N_accounts, 3] for smurfing/structuring/layering.

    Label assignment:
      - If Is_laundering=1, map Laundering_type keywords → typology columns
      - If keywords don't match any typology, set all 3 columns = 1 (unknown)
    """
    labels = np.zeros((n_accounts, 3), dtype=np.float32)

    suspicious = df[df["Is_laundering"] == 1]
    for _, row in suspicious.iterrows():
        lt  = str(row["Laundering_type"]).lower()
        idx = acct_to_idx[row["Sender_account"]]
        matched = False
        for col_idx, keywords in TYPOLOGY_KEYWORDS.items():
            if any(kw in lt for kw in keywords):
                labels[idx, col_idx] = 1.0
                matched = True
        if not matched:
            labels[idx, :] = 1.0   # unknown typology → flag all

    return labels


def build_edge_index(df: pd.DataFrame, acct_to_idx: dict) -> torch.Tensor:
    """Build edge_index [2, E] from sender→receiver pairs."""
    src = df["Sender_account"].map(acct_to_idx).values
    dst = df["Receiver_account"].map(acct_to_idx).values
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


# ============================================================================
# STEP 2: Build PyG-compatible hetero data
# ============================================================================

def build_hetero_inputs(
    feats: np.ndarray,
    edge_index: torch.Tensor,
) -> tuple[dict, dict]:
    """
    Map the homogeneous account graph onto the HeteroGraphSAGE schema:
      account features → "account" node type
      dummy 1-row tensors → "customer" and "transaction" node types
      edge → ("account", "transfer", "account")

    The model handles missing types gracefully via the residual structure.
    """
    x_account = torch.tensor(feats, dtype=torch.float32)
    N = x_account.shape[0]

    # Minimal dummy tensors for unused node types (1 node each)
    x_customer    = torch.zeros(1, 8,  dtype=torch.float32)
    x_transaction = torch.zeros(1, 10, dtype=torch.float32)

    x_dict = {
        "customer":    x_customer,
        "account":     x_account,
        "transaction": x_transaction,
    }

    # Only the account→account transfer edge is populated from SAML-D.
    # Other edge types get empty tensors (shape [2,0]) — model weights them
    # near-zero automatically when no messages pass through them.
    edge_index_dict = {
        ("customer",  "transfer",     "account"):     torch.zeros(2, 0, dtype=torch.long),
        ("account",   "transfer",     "account"):     edge_index,       # SAML-D main edge
        ("account",   "transfer",     "transaction"): torch.zeros(2, 0, dtype=torch.long),
        ("customer",  "shared_ip",    "customer"):    torch.zeros(2, 0, dtype=torch.long),
        ("customer",  "shared_phone", "customer"):    torch.zeros(2, 0, dtype=torch.long),
    }
    return x_dict, edge_index_dict


# ============================================================================
# STEP 3: Training loop
# ============================================================================

def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """Per-typology positive class weights. Cap is high enough to preserve
    the relative imbalance across typologies (the previous 500 cap clipped
    every typology to the same value, erasing the inter-class signal)."""
    pos = labels.sum(axis=0) + 1e-8
    neg = (labels.shape[0] - labels.sum(axis=0)) + 1e-8
    weights = neg / pos
    weights = np.clip(weights, 1.0, 500.0)
    log(f"      Pos weights: smurfing={weights[0]:.1f}  structuring={weights[1]:.1f}  layering={weights[2]:.1f}", DIM)
    return torch.tensor(weights, dtype=torch.float32)


def calibrate_thresholds(proba: np.ndarray, labels: np.ndarray) -> list:
    """Per-typology F1-optimal threshold via a coarse grid search.
    Returns one threshold per column (matches GNNConfig.thresholds layout)."""
    candidates = np.linspace(0.05, 0.95, 19)
    best: list = []
    for j in range(labels.shape[1]):
        y = labels[:, j].astype(np.float32)
        if y.sum() == 0:
            best.append(0.5)
            continue
        f1s = []
        for t in candidates:
            preds = (proba[:, j] >= t).astype(np.float32)
            tp = float((preds * y).sum())
            fp = float((preds * (1 - y)).sum())
            fn = float(((1 - preds) * y).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        best.append(round(float(candidates[int(np.argmax(f1s))]), 2))
    return best


def train(
    x_dict: dict,
    edge_index_dict: dict,
    labels: np.ndarray,
    cfg: GNNConfig,
    epochs: int,
    lr: float,
    device: str,
    checkpoint_path: str,
) -> HeteroGraphSAGEDetector:
    log("\n[4/5] Training HeteroGraphSAGEDetector...", CYN)

    # Move to device
    x_dict_dev = {k: v.to(device) for k, v in x_dict.items()}
    edge_dict_dev = {k: v.to(device) for k, v in edge_index_dict.items()}

    # Split account indices: train / val
    n_accts = labels.shape[0]
    idx_all = np.arange(n_accts)
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)

    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

    # Model
    model = HeteroGraphSAGEDetector(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"      Parameters : {total_params:,}", DIM)

    # Loss — weighted BCE for class imbalance (original AMLLoss path).
    # The focal variant (focal_bce_loss above) is retained for experimentation
    # but unused in this configuration; it flattened outputs too aggressively
    # on the SAML-D class distribution.
    pos_weights = compute_pos_weights(labels).to(device)
    criterion   = AMLLoss(pos_weights=pos_weights)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_state    = None

    log(f"\n      {'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'Val F1':>10} {'LR':>10}", DIM)
    log(f"      {'─'*8:<8} {'─'*10:>12} {'─'*10:>12} {'─'*8:>10} {'─'*8:>10}", DIM)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        logits = model(x_dict_dev, edge_dict_dev)            # [N_accounts, 3]
        train_logits = logits[idx_train]
        train_labels = labels_tensor[idx_train]
        loss_train   = criterion(train_logits, train_labels)

        optimizer.zero_grad()
        loss_train.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = logits[idx_val]
            val_labels = labels_tensor[idx_val]
            loss_val   = criterion(val_logits, val_labels)

            # F1 score (macro, per-typology)
            proba = torch.sigmoid(val_logits)
            preds = (proba >= 0.5).float()
            tp  = (preds * val_labels).sum(dim=0)
            fp  = (preds * (1 - val_labels)).sum(dim=0)
            fn  = ((1 - preds) * val_labels).sum(dim=0)
            f1  = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()

        current_lr = scheduler.get_last_lr()[0]
        is_best = loss_val.item() < best_val_loss
        marker  = " <-- best" if is_best else ""

        colour = GRN if is_best else ""
        log(
            f"      {epoch:<8} {loss_train.item():>12.4f} "
            f"{loss_val.item():>12.4f} {f1:>10.4f} "
            f"{current_lr:>10.2e}{marker}",
            colour,
        )

        if is_best:
            best_val_loss = loss_val.item()
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # Calibrate per-typology thresholds on the validation split.
    # F1-optimal thresholds are saved into the checkpoint so service.py can
    # use them at inference without an env override.
    model.eval()
    with torch.no_grad():
        val_logits = model(x_dict_dev, edge_dict_dev)[idx_val]
        val_proba = torch.sigmoid(val_logits).cpu().numpy()
    calibrated = calibrate_thresholds(val_proba, labels[idx_val])
    log(
        f"      Calibrated thresholds: "
        f"smurfing={calibrated[0]:.2f}  "
        f"structuring={calibrated[1]:.2f}  "
        f"layering={calibrated[2]:.2f}",
        CYN,
    )
    model.cfg.thresholds = calibrated

    model.save(checkpoint_path)
    log(f"\n      Model saved -> {checkpoint_path}", GRN)
    return model


# ============================================================================
# STEP 4: Evaluation
# ============================================================================

def evaluate(
    model: HeteroGraphSAGEDetector,
    x_dict: dict,
    edge_index_dict: dict,
    labels: np.ndarray,
    device: str,
) -> None:
    log("\n[5/5] Final evaluation on full dataset...", CYN)

    x_dict_dev    = {k: v.to(device) for k, v in x_dict.items()}
    edge_dict_dev = {k: v.to(device) for k, v in edge_index_dict.items()}

    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(x_dict_dev, edge_dict_dev).cpu().numpy()

    labels_bin = (labels > 0).any(axis=1)   # any typology → suspicious
    preds_bin  = (proba >= 0.5).any(axis=1)

    tp = int(((preds_bin == 1) & (labels_bin == 1)).sum())
    fp = int(((preds_bin == 1) & (labels_bin == 0)).sum())
    fn = int(((preds_bin == 0) & (labels_bin == 1)).sum())
    tn = int(((preds_bin == 0) & (labels_bin == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    log(f"\n      Confusion Matrix (binary: Suspicious vs Benign)")
    log(f"      {'':15} {'Pred Benign':>14} {'Pred Susp':>12}")
    log(f"      {'True Benign':15} {tn:>14,} {fp:>12,}")
    log(f"      {'True Susp':15} {fn:>14,} {tp:>12,}")
    log(f"\n      Precision : {precision:.4f}")
    log(f"      Recall    : {recall:.4f}")
    log(f"      F1 Score  : {f1:.4f}",  GRN if f1 > 0.5 else YLW)

    log(f"\n      Per-typology detection rates (threshold=0.5):")
    for i, label in enumerate(TYPOLOGY_LABELS):
        true_pos  = labels[:, i].sum()
        detected  = ((proba[:, i] >= 0.5) & (labels[:, i] > 0)).sum()
        recall_t  = detected / (true_pos + 1e-8)
        log(f"        {label:<14}: {int(true_pos)} true positives, {int(detected)} detected  (recall={recall_t:.3f})")

    flagged = model.flag_suspicious_nodes(x_dict_dev, edge_dict_dev)
    log(f"\n      Total flagged accounts : {len(flagged):,}  (threshold per cfg)", YLW)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    log("\n" + "="*68, CYN)
    log("  AML HYBRID SYSTEM -- SAML-D Training Pipeline", BLD)
    log("="*68, CYN)

    t0 = time.time()

    # 1. Load
    df = load_dataset(CSV_PATH, MAX_ROWS)

    # 2. Features
    feats, acct_to_idx = build_account_features_from_df(df)
    n_accounts = feats.shape[0]
    log(f"      Feature matrix : {feats.shape}", DIM)

    # 3. Labels
    log("\n[3/5] Building typology labels...", CYN)
    labels = build_labels(df, acct_to_idx, n_accounts)
    suspicious_accts = (labels.sum(axis=1) > 0).sum()
    log(f"      Suspicious accounts : {suspicious_accts} / {n_accounts}", DIM)
    for i, t in enumerate(TYPOLOGY_LABELS):
        log(f"        {t:<14}: {int(labels[:,i].sum())} accounts", DIM)

    # 4. Build graph
    log("\n      Building edge index...", DIM)
    edge_index = build_edge_index(df, acct_to_idx)
    log(f"      Edges : {edge_index.shape[1]:,}", DIM)

    x_dict, edge_index_dict = build_hetero_inputs(feats, edge_index)

    # 5. Config & train
    cfg = GNNConfig(
        hidden_channels  = HIDDEN,
        num_layers       = 3,
        dropout          = 0.3,
        thresholds       = [0.45, 0.45, 0.45],
        num_typologies   = 3,
    )

    model = train(
        x_dict         = x_dict,
        edge_index_dict= edge_index_dict,
        labels         = labels,
        cfg            = cfg,
        epochs         = BATCH_EPOCHS,
        lr             = LR,
        device         = DEVICE,
        checkpoint_path= CHECKPOINT_PATH,
    )

    # 6. Evaluate
    evaluate(model, x_dict, edge_index_dict, labels, DEVICE)

    elapsed = time.time() - t0
    log(f"\n  Done in {elapsed:.1f}s  |  Checkpoint: {CHECKPOINT_PATH}", GRN + BLD)
    log("="*68, CYN)


if __name__ == "__main__":
    main()

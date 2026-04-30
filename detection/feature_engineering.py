"""
detection/feature_engineering.py
=================================
AML Feature Engineering for Heterogeneous Transaction Graphs.

AML Context
-----------
Graph-based AML detection relies on behavioural signals that are invisible
at the individual transaction level but emerge when viewed as a network:

  Velocity   — How fast is money moving through an account?
               High velocity with small amounts is a hallmark of Smurfing
               (breaking large sums into sub-reporting-threshold deposits).

  Amount     — Raw and derived statistics (mean, std, max) over a rolling
               window expose Structuring patterns (keeping transactions just
               under the CTR threshold of $10k in the US).

  Centrality — Nodes with abnormally high betweenness centrality sit on
               many shortest paths between accounts, typical of Layering
               hubs that funnel funds across multiple shell entities.

  Shared     — Accounts sharing an IP address or phone number that are
  Identity     otherwise unrelated are a strong indicator of synthetic
  Signals      identities or coordinated mule networks.

Node Feature Dimensions
------------------------
  Customer    : [degree_in, degree_out, betweenness, shared_ip_count,
                 shared_phone_count, kyc_risk_score, pep_flag, country_risk]
                → 8 features

  Account     : [balance_mean, balance_std, tx_count_24h, tx_count_7d,
                 avg_tx_amount, max_tx_amount, velocity_zscore,
                 structuring_flag, dormancy_days, incoming_ratio]
                → 10 features

  Transaction : [amount, amount_log, amount_bin_9k,  # ← CTR proximity
                 hour_of_day_sin, hour_of_day_cos,    # cyclic time encoding
                 day_of_week_sin, day_of_week_cos,
                 is_round_amount, cross_border_flag, reversal_flag]
                → 10 features
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US FinCEN Currency Transaction Report (CTR) threshold — $10,000.
# Structuring: deliberately staying below this to avoid reporting.
CTR_THRESHOLD_USD = 10_000.0

# "Proximity band" — transactions within this delta of the CTR threshold
# are flagged as potential structuring attempts.
STRUCTURING_PROXIMITY_USD = 1_000.0

# Smurfing burst window: number of transactions within a rolling 24-hour
# window that warrants a velocity alert.
SMURFING_VELOCITY_THRESHOLD = 8

# Feature dimensions per node type (must match model in_channels config)
NODE_FEATURE_DIMS: Dict[str, int] = {
    "customer": 8,
    "account": 10,
    "transaction": 10,
}


# ---------------------------------------------------------------------------
# Feature Builders
# ---------------------------------------------------------------------------


def build_customer_features(
    degree_in: Tensor,           # In-degree from graph analytics
    degree_out: Tensor,          # Out-degree from graph analytics
    betweenness: Tensor,         # Betweenness centrality (normalised 0-1)
    shared_ip_count: Tensor,     # # of other accounts sharing same IP
    shared_phone_count: Tensor,  # # of other accounts sharing same phone
    kyc_risk_score: Tensor,      # Raw KYC risk score (0-100)
    pep_flag: Tensor,            # Politically Exposed Person binary flag
    country_risk: Tensor,        # FATF country risk score (0-1)
) -> Tensor:
    """
    Concatenate raw signals into the Customer node feature matrix.

    AML Relevance:
      - High betweenness → potential money-mule coordinator / layering hub
      - shared_ip / shared_phone → synthetic identity clusters
      - pep_flag + country_risk → FATF risk amplifiers (Enhanced Due Diligence)
    """
    # Normalise degree by log to dampen scale (real graphs are power-law)
    deg_in_norm = torch.log1p(degree_in.float()).unsqueeze(-1)
    deg_out_norm = torch.log1p(degree_out.float()).unsqueeze(-1)

    return torch.cat(
        [
            deg_in_norm,
            deg_out_norm,
            betweenness.unsqueeze(-1),
            torch.log1p(shared_ip_count.float()).unsqueeze(-1),
            torch.log1p(shared_phone_count.float()).unsqueeze(-1),
            (kyc_risk_score / 100.0).unsqueeze(-1),
            pep_flag.float().unsqueeze(-1),
            country_risk.unsqueeze(-1),
        ],
        dim=-1,
    )  # → [N_customers, 8]


def build_account_features(
    balance_mean: Tensor,
    balance_std: Tensor,
    tx_count_24h: Tensor,
    tx_count_7d: Tensor,
    avg_tx_amount: Tensor,
    max_tx_amount: Tensor,
    incoming_ratio: Tensor,   # Fraction of volume that is incoming
    dormancy_days: Tensor,    # Days since last activity before current burst
) -> Tensor:
    """
    Build Account node features with derived AML signals.

    AML Relevance:
      - velocity_zscore: z-score of tx_count_24h over account history.
        Extreme values (>3σ) indicate Smurfing bursts.
      - structuring_flag: accounts whose avg_tx_amount sits in the
        $9,000–$10,000 band are statistically anomalous under normal behaviour.
      - dormancy_days: reactivated dormant accounts are a classic Layering
        indicator — shell accounts that go quiet then suddenly burst active.
    """
    # --- Velocity Z-Score ---------------------------------------------------
    # We approximate population stats; in production these come from cuGraph
    # rolling aggregates stored per account.
    eps = 1e-8
    velocity_zscore = (tx_count_24h.float() - tx_count_7d.float() / 7.0) / (
        tx_count_7d.float().std() + eps
    )
    velocity_zscore = velocity_zscore.clamp(-10, 10)  # robust clip

    # --- Structuring Proximity Flag -----------------------------------------
    # Binary feature: 1 if average transaction amount is in the "danger zone"
    # between (CTR_THRESHOLD - proximity) and CTR_THRESHOLD.
    lo = CTR_THRESHOLD_USD - STRUCTURING_PROXIMITY_USD
    structuring_flag = (
        (avg_tx_amount >= lo) & (avg_tx_amount < CTR_THRESHOLD_USD)
    ).float()

    return torch.cat(
        [
            torch.log1p(balance_mean.float().abs()).unsqueeze(-1),
            torch.log1p(balance_std.float()).unsqueeze(-1),
            torch.log1p(tx_count_24h.float()).unsqueeze(-1),
            torch.log1p(tx_count_7d.float()).unsqueeze(-1),
            torch.log1p(avg_tx_amount.float()).unsqueeze(-1),
            torch.log1p(max_tx_amount.float()).unsqueeze(-1),
            velocity_zscore.unsqueeze(-1),
            structuring_flag.unsqueeze(-1),
            torch.log1p(dormancy_days.float()).unsqueeze(-1),
            incoming_ratio.float().unsqueeze(-1),
        ],
        dim=-1,
    )  # → [N_accounts, 10]


def build_transaction_features(
    amount: Tensor,          # Raw USD amount
    timestamp_hour: Tensor,  # Hour of day (0–23)
    timestamp_dow: Tensor,   # Day of week (0–6)
    is_round_amount: Tensor, # Binary: amount is a round number (e.g. $5,000)
    cross_border: Tensor,    # Binary: crosses national jurisdictions
    is_reversal: Tensor,     # Binary: reversal/chargeback transaction
) -> Tensor:
    """
    Build Transaction node features with AML-sensitive encodings.

    AML Relevance:
      - amount_log: log-transform compresses the heavy tail; scale-invariant
        detection across micro-transactions and large wire transfers.
      - amount_bin_9k: binary proximity flag — same logic as structuring
        but at the individual transaction level (FinCEN SAR trigger).
      - Cyclic time encoding (sin/cos): captures temporal smurfing patterns
        (e.g. multiple deposits at ATMs at 11:45 PM to reset daily limits)
        without the discontinuity of raw hour/day integers.
      - is_round_amount: human behaviour produces ragged amounts; perfectly
        round sums are a money-laundering indicator in FinCEN red flags.
      - cross_border: FATF Recommendation 16 — cross-border wires above
        threshold require enhanced scrutiny.
      - is_reversal: layering schemes often use reversals to obscure the
        audit trail.
    """
    amount_f = amount.float()

    # Log-amount: stabilises gradient flow across 5+ orders of magnitude
    amount_log = torch.log1p(amount_f)

    # Structuring proximity at transaction level
    lo = CTR_THRESHOLD_USD - STRUCTURING_PROXIMITY_USD
    amount_bin_9k = ((amount_f >= lo) & (amount_f < CTR_THRESHOLD_USD)).float()

    # Cyclic hour encoding — maps 0–23 to unit circle to preserve periodicity
    hour_rad = (timestamp_hour.float() / 24.0) * 2 * math.pi
    hour_sin = torch.sin(hour_rad)
    hour_cos = torch.cos(hour_rad)

    # Cyclic day-of-week encoding
    dow_rad = (timestamp_dow.float() / 7.0) * 2 * math.pi
    dow_sin = torch.sin(dow_rad)
    dow_cos = torch.cos(dow_rad)

    return torch.cat(
        [
            amount_log.unsqueeze(-1),
            (amount_f / CTR_THRESHOLD_USD).unsqueeze(-1),  # normalised amount
            amount_bin_9k.unsqueeze(-1),
            hour_sin.unsqueeze(-1),
            hour_cos.unsqueeze(-1),
            dow_sin.unsqueeze(-1),
            dow_cos.unsqueeze(-1),
            is_round_amount.float().unsqueeze(-1),
            cross_border.float().unsqueeze(-1),
            is_reversal.float().unsqueeze(-1),
        ],
        dim=-1,
    )  # → [N_transactions, 10]


# ---------------------------------------------------------------------------
# Typology Thresholds (used at inference time for rule-based pre-screening)
# ---------------------------------------------------------------------------

TYPOLOGY_RULES = {
    # Smurfing: many small transactions in a short window
    "smurfing": {
        "max_amount": 9_999.99,
        "min_tx_count_24h": SMURFING_VELOCITY_THRESHOLD,
        "description": "Multiple sub-threshold deposits to avoid CTR filing",
    },
    # Structuring: amounts deliberately near but below CTR threshold
    "structuring": {
        "amount_lo": CTR_THRESHOLD_USD - STRUCTURING_PROXIMITY_USD,
        "amount_hi": CTR_THRESHOLD_USD,
        "description": "Transactions clustered just below $10k CTR threshold",
    },
    # Layering: complex multi-hop transfers across jurisdictions
    "layering": {
        "min_hops": 3,
        "cross_border_required": True,
        "description": "Multi-jurisdictional fund movement to obscure audit trail",
    },
}

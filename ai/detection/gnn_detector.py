"""
detection/gnn_detector.py
==========================
Heterogeneous GraphSAGE model for AML node classification.

Graph Schema
------------
Node types  : customer (8-d) | account (10-d) | transaction (10-d)
Edge types  : (customer,  transfer,     account)      — money movement
              (account,   transfer,     transaction)  — account owns tx
              (customer,  shared_ip,    customer)     — same IP cluster
              (customer,  shared_phone, customer)     — same phone cluster

Output Typologies (multi-label sigmoid)
---------------------------------------
  [0] Smurfing   — many sub-threshold deposits by a coordinated mule network
  [1] Structuring — amounts engineered to stay just below the CTR threshold
  [2] Layering    — complex multi-hop, multi-jurisdiction fund movement

Design Choices
--------------
  • HeteroConv wraps SAGEConv per edge type → one message-passing kernel
    per relation; weights are NOT shared across edge types, which lets the
    model learn distinct aggregation semantics for Transfer vs. Shared_IP.
  • Linear projection layers normalise heterogeneous feature spaces to a
    common hidden dimension before the first conv layer.
  • LazyLinear input projections allow us to defer specifying exact input
    dims until the first forward pass (useful during prototyping).
  • BatchNorm + ELU after each conv layer for stable deep training.
  • The output head is a per-typology binary classifier (BCEWithLogitsLoss),
    NOT a softmax, because a single subgraph can exhibit multiple typologies
    simultaneously (e.g. smurfing + layering in parallel).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, BatchNorm, Linear
from torch_geometric.transforms import ToUndirected

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must align with feature_engineering.py NODE_FEATURE_DIMS
# ---------------------------------------------------------------------------

NODE_TYPES: List[str] = ["customer", "account", "transaction"]

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("customer",     "transfer",     "account"),
    ("account",      "transfer",     "account"),      # SAML-D: acct → acct
    ("account",      "transfer",     "transaction"),
    ("customer",     "shared_ip",    "customer"),
    ("customer",     "shared_phone", "customer"),
]

TYPOLOGY_LABELS: List[str] = ["smurfing", "structuring", "layering"]

# Raw input feature dims per node type (see feature_engineering.py)
NODE_IN_DIMS: Dict[str, int] = {
    "customer":    8,
    "account":    10,
    "transaction": 10,
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GNNConfig:
    """All hyperparameters for the HeteroGraphSAGE detector."""

    hidden_channels: int   = 128
    num_layers: int        = 3      # GraphSAGE depth (3 = 3-hop receptive field)
    dropout: float         = 0.3
    aggr: str              = "mean" # SAGEConv neighbourhood aggregation

    # Classifier head
    num_typologies: int    = 3      # smurfing | structuring | layering
    classifier_hidden: int = 64

    # Inference thresholds (per typology, tuned on validation set)
    thresholds: List[float] = field(
        default_factory=lambda: [0.45, 0.50, 0.40]
    )

    # Target node type for classification output
    target_node_type: str  = "account"


# ---------------------------------------------------------------------------
# Sub-module: Input Projection
# ---------------------------------------------------------------------------

class HeteroInputProjection(nn.Module):
    """
    Project each node type's raw feature vector into a shared hidden space.

    AML Rationale: Customer features (8-d) and Transaction features (10-d)
    live in incompatible spaces. Projecting to a common `hidden_channels`
    dimension before message passing ensures the SAGEConv kernels receive
    commensurable representations when aggregating across node types.
    """

    def __init__(self, in_dims: Dict[str, int], hidden_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(in_dim, hidden_channels, bias=True),
                    nn.LayerNorm(hidden_channels),
                    nn.ELU(),
                )
                for ntype, in_dim in in_dims.items()
            }
        )

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {ntype: self.projections[ntype](x) for ntype, x in x_dict.items()}


# ---------------------------------------------------------------------------
# Sub-module: Heterogeneous GraphSAGE Layer
# ---------------------------------------------------------------------------

class HeteroSAGELayer(nn.Module):
    """
    One layer of Heterogeneous GraphSAGE.

    Uses HeteroConv to apply a dedicated SAGEConv kernel per edge type,
    then sums contributions from all edge types into each node's embedding.

    The `aggr="sum"` in HeteroConv controls HOW contributions from multiple
    edge types are merged at the destination node (not the neighbourhood agg).
    Neighbourhood aggregation within each edge type is controlled by SAGEConv's
    own `aggr` parameter (default: "mean").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        aggr: str = "mean",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # One SAGEConv per edge relation — weights are NOT shared.
        # This lets (customer→account) and (customer→customer via shared_ip)
        # learn completely different aggregation behaviours.
        self.conv = HeteroConv(
            {
                etype: SAGEConv(
                    in_channels=(in_channels, in_channels),
                    out_channels=out_channels,
                    aggr=aggr,
                    normalize=True,   # L2-normalise output embeddings
                    bias=True,
                )
                for etype in edge_types
            },
            aggr="sum",  # sum edge-type contributions at destination node
        )

        # LayerNorm instead of BatchNorm: works on any batch size including 1
        # (BatchNorm fails when dummy node types have only 1 node)
        self.norms = nn.ModuleDict(
            {
                ntype: nn.LayerNorm(out_channels)
                for ntype in NODE_TYPES
            }
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        # Message passing across all edge types simultaneously
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply norm → activation → dropout per node type
        return {
            ntype: self.dropout(
                self.activation(
                    self.norms[ntype](h)
                )
            )
            for ntype, h in out_dict.items()
            if ntype in self.norms
        }


# ---------------------------------------------------------------------------
# Sub-module: AML Typology Classifier Head
# ---------------------------------------------------------------------------

class TypologyClassifierHead(nn.Module):
    """
    MLP classifier that maps account-node embeddings to per-typology logits.

    Multi-label design: each of the 3 typologies gets an independent sigmoid,
    so the model can simultaneously flag smurfing AND layering on the same node
    — a real-world scenario where smurfing feeds a layering chain.

    Loss: BCEWithLogitsLoss (numerically stable sigmoid + BCE combined).
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_typologies: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, num_typologies),  # raw logits
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns raw logits of shape [N_accounts, num_typologies]."""
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Main Model: HeteroGraphSAGEDetector
# ---------------------------------------------------------------------------

class HeteroGraphSAGEDetector(nn.Module):
    """
    3-layer Heterogeneous GraphSAGE for AML node classification.

    Forward pass pipeline:
      1. Project each node type to shared hidden_channels space
      2. 3× HeteroSAGELayer (message passing with residual skip connections)
      3. TypologyClassifierHead on target node type ('account')
      4. Return logits (training) OR sigmoid probabilities (inference)

    The 3-hop receptive field means each account aggregates information from:
      Hop 1 → its own transactions and linked customers
      Hop 2 → customers sharing IPs/phones with linked customers
      Hop 3 → transactions of those shared-identity neighbours

    This is exactly the subgraph structure relevant to smurfing rings and
    coordinated mule networks.
    """

    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_channels

        # Stage 1: project raw features into shared latent space
        self.input_proj = HeteroInputProjection(NODE_IN_DIMS, H)

        # Stage 2: 3× heterogeneous SAGEConv layers
        self.layers = nn.ModuleList(
            [
                HeteroSAGELayer(
                    in_channels=H,
                    out_channels=H,
                    edge_types=EDGE_TYPES,
                    aggr=cfg.aggr,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # Residual projection (identity if dims match — they do here)
        self.res_proj = nn.Identity()

        # Stage 3: typology classifier on account embeddings
        self.classifier = TypologyClassifierHead(
            in_channels=H,
            hidden_channels=cfg.classifier_hidden,
            num_typologies=cfg.num_typologies,
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        return_embeddings: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            x_dict           : {node_type: feature_tensor}
            edge_index_dict  : {edge_type_tuple: edge_index [2, E]}
            return_embeddings: if True, also return account embeddings
                               (used for RAG vector indexing)

        Returns:
            logits [N_accounts, 3]  — training mode (raw logits)
            OR
            (logits, embeddings)    — when return_embeddings=True
        """
        # --- Input projection -------------------------------------------
        h_dict = self.input_proj(x_dict)

        # --- Stacked GraphSAGE layers with residual connections ----------
        for layer in self.layers:
            h_new = layer(h_dict, edge_index_dict)
            # Residual skip: add previous hidden state where node type exists
            # in both dicts. This helps gradients flow through deep stacks
            # and prevents over-smoothing (embeddings collapsing to mean).
            h_dict = {
                ntype: self.res_proj(h_dict[ntype]) + h_new[ntype]
                if ntype in h_new
                else h_dict[ntype]
                for ntype in h_dict
            }

        # --- Extract target node (account) embeddings -------------------
        account_emb = h_dict[self.cfg.target_node_type]  # [N_accounts, H]

        # --- Classifier head → raw logits -------------------------------
        logits = self.classifier(account_emb)             # [N_accounts, 3]

        if return_embeddings:
            return logits, account_emb
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Tensor:
        """
        Returns sigmoid probabilities [N_accounts, 3] for each typology.
        Safe to call at inference time (no gradient tracking).
        """
        self.eval()
        logits = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_typologies(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[int, Dict[str, float]]:
        """
        Threshold probabilities using per-typology thresholds from GNNConfig.

        Returns:
            {account_node_idx: {typology_name: probability}}
            Only includes accounts that breach at least one threshold.

        AML use: downstream AMLInvestigatorAgent only receives genuinely
        suspicious accounts — reduces false-positive investigation load.
        """
        proba = self.predict_proba(x_dict, edge_index_dict)  # [N, 3]
        thresholds = torch.tensor(
            self.cfg.thresholds, device=proba.device
        )                                                      # [3]
        flagged_mask = (proba >= thresholds).any(dim=-1)      # [N] bool

        result: Dict[int, Dict[str, float]] = {}
        flagged_indices = flagged_mask.nonzero(as_tuple=True)[0].tolist()

        for idx in flagged_indices:
            scores = proba[idx].tolist()
            result[idx] = {
                label: round(score, 4)
                for label, score in zip(TYPOLOGY_LABELS, scores)
            }

        return result

    @torch.no_grad()
    def flag_suspicious_nodes(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        threshold: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Public alias for predict_typologies — returns flagged account dict.
        Optionally overrides the per-typology thresholds with a single value.
        """
        if threshold is not None:
            orig = self.cfg.thresholds
            self.cfg.thresholds = [threshold] * self.cfg.num_typologies
            result = self.predict_typologies(x_dict, edge_index_dict)
            self.cfg.thresholds = orig
            return result
        return self.predict_typologies(x_dict, edge_index_dict)

    @torch.no_grad()
    def get_embeddings(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Tensor:
        """
        Return account-level GNN embeddings for RAG vector indexing in
        Milvus / Neo4j. These embeddings encode both local features AND
        2-hop neighbourhood structure, making them semantically rich for
        similarity search over past SAR cases.
        """
        self.eval()
        _, embeddings = self.forward(
            x_dict, edge_index_dict, return_embeddings=True
        )
        return embeddings  # [N_accounts, hidden_channels]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize weights + config to a single checkpoint file."""
        checkpoint = {
            "config": self.cfg.__dict__,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "HeteroGraphSAGEDetector":
        """Restore model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        cfg = GNNConfig(**checkpoint["config"])
        model = cls(cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        logger.info("Model loaded ← %s  [device=%s]", path, device)
        return model


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class AMLLoss(nn.Module):
    """
    Weighted BCEWithLogitsLoss for imbalanced AML datasets.

    Positive-class weights address extreme class imbalance:
    in real transaction datasets suspicious accounts are typically
    < 0.1 % of all accounts. Without weighting the model collapses
    to predicting 'benign' for everything.

    Weights are computed per typology as:
        pos_weight[i] = (N_negative[i] / N_positive[i])
    and passed to BCEWithLogitsLoss for each typology column.
    """

    def __init__(self, pos_weights: Optional[Tensor] = None) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            logits : [N_accounts, 3]  raw model output
            labels : [N_accounts, 3]  multi-hot ground truth {0, 1}
        Returns:
            scalar loss
        """
        return self.criterion(logits, labels.float())

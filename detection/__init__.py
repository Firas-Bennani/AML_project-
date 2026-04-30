"""detection package — AML GNN Detection Layer."""
from detection.gnn_detector import (
    GNNConfig,
    HeteroGraphSAGEDetector,
    AMLLoss,
    TYPOLOGY_LABELS,
    NODE_TYPES,
    EDGE_TYPES,
)
from detection.feature_engineering import (
    build_customer_features,
    build_account_features,
    build_transaction_features,
    TYPOLOGY_RULES,
    NODE_FEATURE_DIMS,
)
from detection.triton_inference import (
    AMLInferenceBackend,
    TensorRTInferenceEngine,
    TritonInferenceClient,
    FlattenedGNNWrapper,
)

__all__ = [
    "GNNConfig",
    "HeteroGraphSAGEDetector",
    "AMLLoss",
    "TYPOLOGY_LABELS",
    "NODE_TYPES",
    "EDGE_TYPES",
    "build_customer_features",
    "build_account_features",
    "build_transaction_features",
    "TYPOLOGY_RULES",
    "NODE_FEATURE_DIMS",
    "AMLInferenceBackend",
    "TensorRTInferenceEngine",
    "TritonInferenceClient",
    "FlattenedGNNWrapper",
]

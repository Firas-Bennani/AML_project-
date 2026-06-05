"""
detection/triton_inference.py
==============================
NVIDIA TensorRT + Triton Inference Server optimisation wrapper
for HeteroGraphSAGEDetector.

Two execution paths are provided:

  1. TensorRTInferenceEngine
     ─────────────────────────
     Exports the GNN to ONNX, then compiles with TensorRT for maximum
     single-GPU throughput. Best for: real-time streaming inference
     where latency < 5 ms per subgraph is required.

     Pipeline:
       PyTorch model → torch.onnx.export → TensorRT engine (FP16/INT8)
       → pycuda runtime → numpy results

  2. TritonInferenceClient
     ──────────────────────
     Sends inference requests to a running Triton Inference Server
     (e.g. deployed on a separate GPU node or as a K8s sidecar).
     Best for: production micro-service architectures where the GNN
     model is hosted centrally and multiple AML agents call it via gRPC.

     Pipeline:
       Preprocessed tensors → gRPC InferRequest → Triton Server
       → InferResponse → sigmoid proba

     Triton model repository layout expected:
       model_repository/
         aml_hetero_sage/
           config.pbtxt
           1/
             model.onnx

Architectural Note on Heterogeneous Graphs + ONNX
--------------------------------------------------
PyG's HeteroConv uses Python dicts as I/O which are not natively
ONNX-traceable. The export strategy here is to:
  a) Flatten the hetero graph into a fixed-layout tensor batch
     (node_feats stacked by type, offsets tracked by a type_mask tensor).
  b) Export a thin `FlattenedGNNWrapper` that internally splits the
     flat tensor back into per-type dicts before calling the real model.
This avoids ONNX dynamic dict limitations while keeping the PyG model
unchanged.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from detection.gnn_detector import (
    HeteroGraphSAGEDetector,
    GNNConfig,
    NODE_TYPES,
    TYPOLOGY_LABELS,
    NODE_IN_DIMS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module loads even without GPU libs
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt              # type: ignore[import]  # GPU-only
    import pycuda.driver as cuda        # type: ignore[import]  # GPU-only
    import pycuda.autoinit              # type: ignore[import]  # noqa: F401
    _TRT_AVAILABLE = True
except ImportError:
    trt = None       # type: ignore[assignment]
    cuda = None      # type: ignore[assignment]
    _TRT_AVAILABLE = False
    logger.warning("TensorRT not found — TensorRTInferenceEngine disabled.")

try:
    import tritonclient.grpc as grpcclient                      # type: ignore[import]
    import tritonclient.grpc.model_config_pb2 as mc             # type: ignore[import]
    _TRITON_AVAILABLE = True
except ImportError:
    grpcclient = None   # type: ignore[assignment]
    mc = None           # type: ignore[assignment]
    _TRITON_AVAILABLE = False
    logger.warning("tritonclient not found — TritonInferenceClient disabled.")


# ---------------------------------------------------------------------------
# Helper: Flatten heterogeneous graph to fixed-shape tensors
# ---------------------------------------------------------------------------

class FlattenedGNNWrapper(nn.Module):
    """
    ONNX-traceable wrapper around HeteroGraphSAGEDetector.

    The GNN internally uses dicts, which ONNX cannot trace directly.
    This wrapper:
      1. Accepts a single flat node feature matrix [N_total, max_feat_dim]
         and a type_ids vector [N_total] indicating node type per row.
      2. Splits by type_ids → reconstructs x_dict.
      3. Calls the real model.
      4. Returns flattened logits for the account nodes.

    Edge indices are passed as separate flat tensors with a corresponding
    edge_type_ids vector.
    """

    # Maximum feature dimension across all node types (pad shorter types)
    MAX_FEAT_DIM: int = max(NODE_IN_DIMS.values())  # 10

    # Numeric type IDs matching NODE_TYPES order
    TYPE_ID: Dict[str, int] = {nt: i for i, nt in enumerate(NODE_TYPES)}

    def __init__(self, model: HeteroGraphSAGEDetector) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def pack_hetero(
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict]:
        """
        Convert PyG hetero dicts → flat tensors for ONNX export.

        Returns:
            flat_x        : [N_total, MAX_FEAT_DIM] — padded node features
            type_ids      : [N_total] — node type index
            flat_edges    : [2, E_total] — concatenated edge indices (global)
            edge_type_ids : [E_total] — edge type index per column
            meta          : dict with node/edge counts for unpacking
        """
        max_dim = FlattenedGNNWrapper.MAX_FEAT_DIM
        parts_x, parts_tid = [], []
        node_offsets: Dict[str, int] = {}
        cursor = 0

        for i, ntype in enumerate(NODE_TYPES):
            x = x_dict[ntype]                          # [N_i, d_i]
            N, d = x.shape
            # Pad feature dim to max_dim
            pad = torch.zeros(N, max_dim - d, device=x.device)
            parts_x.append(torch.cat([x, pad], dim=-1))
            parts_tid.append(torch.full((N,), i, dtype=torch.long, device=x.device))
            node_offsets[ntype] = cursor
            cursor += N

        flat_x = torch.cat(parts_x, dim=0)       # [N_total, max_dim]
        type_ids = torch.cat(parts_tid, dim=0)    # [N_total]

        # Flatten edge indices — remap local node indices to global offsets
        edge_types = list(edge_index_dict.keys())
        parts_ei, parts_etid = [], []
        for j, etype in enumerate(edge_types):
            src_type, _, dst_type = etype
            ei = edge_index_dict[etype].clone()    # [2, E_j]
            ei[0] += node_offsets[src_type]
            ei[1] += node_offsets[dst_type]
            parts_ei.append(ei)
            parts_etid.append(
                torch.full((ei.shape[1],), j, dtype=torch.long, device=ei.device)
            )

        flat_edges = torch.cat(parts_ei, dim=1)       # [2, E_total]
        edge_type_ids = torch.cat(parts_etid, dim=0)  # [E_total]

        meta = {
            "node_offsets": node_offsets,
            "node_counts": {nt: x_dict[nt].shape[0] for nt in NODE_TYPES},
            "edge_types": edge_types,
        }
        return flat_x, type_ids, flat_edges, edge_type_ids, meta

    def forward(
        self,
        flat_x: Tensor,        # [N_total, MAX_FEAT_DIM]
        type_ids: Tensor,      # [N_total]
        flat_edges: Tensor,    # [2, E_total]
        edge_type_ids: Tensor, # [E_total]
    ) -> Tensor:
        """
        Unpack flat tensors → hetero dicts → run GNN → return account logits.
        """
        # Reconstruct x_dict
        x_dict: Dict[str, Tensor] = {}
        for i, ntype in enumerate(NODE_TYPES):
            mask = type_ids == i
            d = NODE_IN_DIMS[ntype]
            x_dict[ntype] = flat_x[mask, :d]

        # Reconstruct edge_index_dict (uses stored edge type list from model)
        # NOTE: edge_types order must match what was used in pack_hetero()
        from detection.gnn_detector import EDGE_TYPES
        edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
        for j, etype in enumerate(EDGE_TYPES):
            emask = edge_type_ids == j
            edge_index_dict[etype] = flat_edges[:, emask]

        logits = self.model(x_dict, edge_index_dict)  # [N_accounts, 3]
        return logits


# ---------------------------------------------------------------------------
# 1. TensorRT Inference Engine
# ---------------------------------------------------------------------------

class TensorRTInferenceEngine:
    """
    Compiles HeteroGraphSAGEDetector to a TensorRT FP16 engine and
    runs low-latency inference via pycuda.

    Usage:
        engine = TensorRTInferenceEngine.from_pytorch(model, cfg)
        proba = engine.infer(x_dict, edge_index_dict)

    Optimisation flags:
        • FP16 mode  — ~2× throughput vs FP32, negligible accuracy loss
          for well-trained GNNs (embeddings are smooth, not quantisation-
          sensitive like activations in image models).
        • Workspace  — 1 GB TensorRT workspace for layer fusion search.
        • Dynamic shapes — batch axis (N_total) is dynamic via
          OptimizationProfile, supporting variable-size transaction graphs.
    """

    def __init__(self, engine_path: str) -> None:
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        logger.info("TensorRT engine loaded from %s", engine_path)

    # ------------------------------------------------------------------
    # Build from PyTorch model
    # ------------------------------------------------------------------

    @classmethod
    def from_pytorch(
        cls,
        model: HeteroGraphSAGEDetector,
        engine_path: str,
        fp16: bool = True,
        workspace_gb: int = 1,
        max_nodes: int = 100_000,
        max_edges: int = 500_000,
    ) -> "TensorRTInferenceEngine":
        """
        Export PyTorch model → ONNX → TensorRT engine, then save.

        Args:
            max_nodes : upper bound for dynamic shape OptimizationProfile.
                        Set to the 99th-percentile subgraph size in production.
            max_edges : upper bound for edge count in dynamic shape profile.
        """
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed.")

        onnx_path = engine_path.replace(".trt", ".onnx")
        cls._export_onnx(model, onnx_path, max_nodes, max_edges)
        cls._build_engine(onnx_path, engine_path, fp16, workspace_gb,
                          max_nodes, max_edges)
        return cls(engine_path)

    @staticmethod
    def _export_onnx(
        model: HeteroGraphSAGEDetector,
        onnx_path: str,
        max_nodes: int,
        max_edges: int,
    ) -> None:
        """Wrap model in FlattenedGNNWrapper and export to ONNX."""
        wrapper = FlattenedGNNWrapper(model).eval().cuda()
        N = 64          # representative batch for tracing
        E = 256
        max_d = FlattenedGNNWrapper.MAX_FEAT_DIM

        dummy_flat_x = torch.randn(N, max_d).cuda()
        dummy_type_ids = torch.randint(0, 3, (N,)).cuda()
        dummy_edges = torch.randint(0, N, (2, E)).cuda()
        dummy_etids = torch.randint(0, 4, (E,)).cuda()

        torch.onnx.export(
            wrapper,
            (dummy_flat_x, dummy_type_ids, dummy_edges, dummy_etids),
            onnx_path,
            input_names=["flat_x", "type_ids", "flat_edges", "edge_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "flat_x":        {0: "num_nodes"},
                "type_ids":      {0: "num_nodes"},
                "flat_edges":    {1: "num_edges"},
                "edge_type_ids": {0: "num_edges"},
                "logits":        {0: "num_accounts"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info("ONNX model exported → %s", onnx_path)

    @staticmethod
    def _build_engine(
        onnx_path: str,
        engine_path: str,
        fp16: bool,
        workspace_gb: int,
        max_nodes: int,
        max_edges: int,
    ) -> None:
        """Parse ONNX and compile TensorRT engine with dynamic shape profile."""
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("ONNX parsing failed.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
        )
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Dynamic shape optimisation profile
        profile = builder.create_optimization_profile()
        profile.set_shape("flat_x",
            min=(1,  FlattenedGNNWrapper.MAX_FEAT_DIM),
            opt=(512, FlattenedGNNWrapper.MAX_FEAT_DIM),
            max=(max_nodes, FlattenedGNNWrapper.MAX_FEAT_DIM),
        )
        profile.set_shape("type_ids",        min=(1,), opt=(512,), max=(max_nodes,))
        profile.set_shape("flat_edges",      min=(2, 1), opt=(2, 2048), max=(2, max_edges))
        profile.set_shape("edge_type_ids",   min=(1,), opt=(2048,), max=(max_edges,))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized)
        logger.info("TensorRT engine saved → %s", engine_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> np.ndarray:
        """
        Run TensorRT inference.

        Returns:
            proba [N_accounts, 3] numpy array — sigmoid probabilities
            per typology (smurfing, structuring, layering).
        """
        flat_x, type_ids, flat_edges, edge_type_ids, meta = (
            FlattenedGNNWrapper.pack_hetero(x_dict, edge_index_dict)
        )

        inputs_np = {
            "flat_x":        flat_x.cpu().numpy().astype(np.float32),
            "type_ids":      type_ids.cpu().numpy().astype(np.int32),
            "flat_edges":    flat_edges.cpu().numpy().astype(np.int32),
            "edge_type_ids": edge_type_ids.cpu().numpy().astype(np.int32),
        }

        N_accounts = meta["node_counts"]["account"]
        output_np = np.empty((N_accounts, 3), dtype=np.float32)

        # Allocate GPU buffers and run inference
        bindings = []
        gpu_buffers = []

        for name in ["flat_x", "type_ids", "flat_edges", "edge_type_ids"]:
            arr = inputs_np[name]
            gpu_buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(gpu_buf, arr)
            bindings.append(int(gpu_buf))
            gpu_buffers.append(gpu_buf)
            idx = self.engine.get_binding_index(name)
            self.context.set_binding_shape(idx, arr.shape)

        out_buf = cuda.mem_alloc(output_np.nbytes)
        bindings.append(int(out_buf))

        stream = cuda.Stream()
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_np, out_buf, stream)
        stream.synchronize()

        # Sigmoid activation (model outputs raw logits)
        proba = 1.0 / (1.0 + np.exp(-output_np))
        return proba   # [N_accounts, 3]


# ---------------------------------------------------------------------------
# 2. Triton gRPC Inference Client
# ---------------------------------------------------------------------------

class TritonInferenceClient:
    """
    Sends pre-processed graph tensors to a Triton Inference Server
    hosting the ONNX-exported GNN model.

    Expected Triton model config (config.pbtxt) input/output names:
        Inputs : flat_x, type_ids, flat_edges, edge_type_ids
        Outputs: logits

    The client handles:
      • tensor packing via FlattenedGNNWrapper.pack_hetero()
      • gRPC InferRequest construction
      • response parsing + sigmoid conversion
      • latency logging for SLO monitoring

    Usage:
        client = TritonInferenceClient(url="localhost:8001",
                                       model_name="aml_hetero_sage")
        proba = client.infer(x_dict, edge_index_dict)
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "aml_hetero_sage",
        model_version: str = "1",
        ssl: bool = False,
    ) -> None:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("tritonclient[grpc] is not installed.")

        self.model_name = model_name
        self.model_version = model_version
        self.client = grpcclient.InferenceServerClient(url=url, ssl=ssl)

        if not self.client.is_model_ready(model_name, model_version):
            raise RuntimeError(
                f"Triton model '{model_name}' v{model_version} is not ready at {url}."
            )
        logger.info("Triton client ready → %s @ %s", model_name, url)

    def infer(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        request_id: str = "aml_req",
    ) -> np.ndarray:
        """
        Pack graph tensors, send to Triton, return sigmoid proba.

        Returns:
            proba [N_accounts, 3] numpy array
        """
        t0 = time.perf_counter()

        flat_x, type_ids, flat_edges, edge_type_ids, meta = (
            FlattenedGNNWrapper.pack_hetero(x_dict, edge_index_dict)
        )

        # Build Triton InferInput objects
        def _make_input(name: str, arr: np.ndarray, dtype: str):
            inp = grpcclient.InferInput(name, arr.shape, dtype)
            inp.set_data_from_numpy(arr)
            return inp

        inputs = [
            _make_input("flat_x",        flat_x.cpu().numpy().astype(np.float32), "FP32"),
            _make_input("type_ids",      type_ids.cpu().numpy().astype(np.int32),  "INT32"),
            _make_input("flat_edges",    flat_edges.cpu().numpy().astype(np.int32),"INT32"),
            _make_input("edge_type_ids", edge_type_ids.cpu().numpy().astype(np.int32), "INT32"),
        ]

        outputs = [grpcclient.InferRequestedOutput("logits")]

        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
        )

        logits = response.as_numpy("logits")          # [N_accounts, 3]
        proba = 1.0 / (1.0 + np.exp(-logits))        # sigmoid

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Triton infer complete | accounts=%d | latency=%.2f ms",
            meta["node_counts"]["account"], latency_ms,
        )
        return proba   # [N_accounts, 3]

    def health_check(self) -> bool:
        """Returns True if Triton server is live and model is ready."""
        return (
            self.client.is_server_live()
            and self.client.is_model_ready(self.model_name, self.model_version)
        )


# ---------------------------------------------------------------------------
# Unified inference facade
# ---------------------------------------------------------------------------

class AMLInferenceBackend:
    """
    Thin façade that delegates to either the TensorRT engine (local GPU)
    or the Triton client (remote inference server), with a PyTorch fallback.

    Priority: TensorRT > Triton > PyTorch (CPU/CUDA)

    Returns a structured result dict ready for downstream AMLInvestigatorAgent.
    """

    def __init__(
        self,
        pytorch_model: HeteroGraphSAGEDetector,
        trt_engine: Optional[TensorRTInferenceEngine] = None,
        triton_client: Optional[TritonInferenceClient] = None,
    ) -> None:
        self.model = pytorch_model
        self.trt = trt_engine
        self.triton = triton_client

    def infer(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, object]:
        """
        Run inference with best available backend.

        Returns:
            {
              "proba":    np.ndarray [N_accounts, 3],
              "labels":   List[str] ["smurfing", "structuring", "layering"],
              "backend":  str,
              "flagged":  Dict[int, Dict[str, float]],  # node_idx → scores
            }
        """
        if self.trt is not None:
            proba = self.trt.infer(x_dict, edge_index_dict)
            backend = "tensorrt"
        elif self.triton is not None:
            proba = self.triton.infer(x_dict, edge_index_dict)
            backend = "triton"
        else:
            with torch.no_grad():
                proba = self.model.predict_proba(x_dict, edge_index_dict).cpu().numpy()
            backend = "pytorch"

        # Apply per-typology thresholds to build flagged dict
        thresholds = np.array(self.model.cfg.thresholds)
        flagged_mask = (proba >= thresholds).any(axis=-1)
        flagged_indices = np.where(flagged_mask)[0].tolist()

        flagged = {
            int(idx): {
                label: round(float(proba[idx, j]), 4)
                for j, label in enumerate(TYPOLOGY_LABELS)
            }
            for idx in flagged_indices
        }

        return {
            "proba":   proba,
            "labels":  TYPOLOGY_LABELS,
            "backend": backend,
            "flagged": flagged,
        }

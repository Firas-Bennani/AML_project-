"""
investigation/nodes/rag_search.py
===================================
LangGraph node: RAG_Search

Performs parallel vector search over two Milvus collections:
  1. `kyc_documents`  — PDF-parsed KYC forms, onboarding files, CDD records
  2. `negative_news`  — Scraped news articles flagging the entity or
                        related persons for financial crime, sanctions, PEP

Optionally also queries a `past_sars` collection to retrieve similar
previously filed Suspicious Activity Reports as few-shot context.

Vector Search Strategy
----------------------
  Query embedding is constructed by concatenating:
    a) Node ID string → embedded via the same encoder used for KYC ingestion
    b) Graph summary text → embedded to capture topology-level semantics
  This "bimodal query" retrieves documents relevant to BOTH the entity's
  identity and its transaction behaviour.

  Collection-specific filters (Milvus scalar filtering) are applied to
  restrict KYC results to the entity's jurisdiction and account type.

AML Relevance of Negative News
--------------------------------
  • Sanctions matches (OFAC SDN list, EU Consolidated) → direct legal hold
  • PEP mentions → triggers Enhanced Due Diligence (EDD) under FATF R.12
  • Adverse media (fraud, bribery, corruption) → risk amplifier for SAR
  • Shell company registry hits → Layering investigation signal

Collection Schemas (expected)
------------------------------
  kyc_documents  : {id, entity_id, jurisdiction, doc_type, text, embedding}
  negative_news  : {id, entity_name, source_url, article_date, text, embedding}
  past_sars      : {id, typology, jurisdiction, narrative_en, embedding}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pymilvus import Collection, connections, MilvusException
from pymilvus import utility as milvus_utility

from investigation.state import AMLState, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Milvus collection names
# ---------------------------------------------------------------------------
COLLECTION_KYC   = "kyc_documents"
COLLECTION_NEWS  = "negative_news"
COLLECTION_SARS  = "past_sars"

# Search parameters — HNSW index tuning
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 128}}


# ---------------------------------------------------------------------------
# RAGSearchNode
# ---------------------------------------------------------------------------

class RAGSearchNode:
    """
    LangGraph node: vector search over Milvus KYC + Negative News collections.

    Parallel search: KYC and news queries are issued concurrently via
    asyncio.gather to minimise latency (typically 2 Milvus round-trips
    in parallel instead of sequential = ~50% wall-clock reduction).
    """

    def __init__(
        self,
        milvus_host: str,
        milvus_port: int,
        embedding_model: Any,       # Any encoder with .encode(text) → np.ndarray
        kyc_top_k: int = 5,
        news_top_k: int = 5,
        sar_top_k: int = 3,
        score_threshold: float = 0.60,
        kyc_filter: Optional[str] = None,   # Milvus scalar filter expression
    ) -> None:
        self.embedding_model = embedding_model
        self.kyc_top_k = kyc_top_k
        self.news_top_k = news_top_k
        self.sar_top_k = sar_top_k
        self.score_threshold = score_threshold
        self.kyc_filter = kyc_filter

        # Establish Milvus connection
        connections.connect(alias="default", host=milvus_host, port=milvus_port)
        self._validate_collections()
        logger.info(
            "Milvus connected → %s:%d | collections: %s, %s, %s",
            milvus_host, milvus_port, COLLECTION_KYC, COLLECTION_NEWS, COLLECTION_SARS
        )

    def _validate_collections(self) -> None:
        """Ensure required Milvus collections exist at startup."""
        for name in [COLLECTION_KYC, COLLECTION_NEWS, COLLECTION_SARS]:
            if not milvus_utility.has_collection(name):
                logger.warning("Milvus collection '%s' not found — RAG may return empty results.", name)

    # ------------------------------------------------------------------
    # LangGraph callable interface
    # ------------------------------------------------------------------

    async def __call__(self, state: AMLState) -> Dict[str, Any]:
        """
        Called by LangGraph as the 'rag_search' node.
        Returns updated state keys: kyc_chunks, news_chunks, similar_sars.
        """
        node_id      = state["node_id"]
        graph_summary = state.get("graph_summary", "")

        # Build a bimodal query string:
        # Entity identity + topology signals → single semantic query vector
        query_text = self._build_query(node_id, state["typology_scores"], graph_summary)
        query_vec  = self._embed(query_text)

        logger.info("[rag_search] Running parallel Milvus search for node=%s", node_id)

        # Parallel search — all three collections simultaneously
        kyc_task   = asyncio.to_thread(self._search_kyc,  query_vec, node_id)
        news_task  = asyncio.to_thread(self._search_news, query_vec, node_id)
        sars_task  = asyncio.to_thread(self._search_sars, query_vec)

        kyc_chunks, news_chunks, similar_sars = await asyncio.gather(
            kyc_task, news_task, sars_task
        )

        logger.info(
            "[rag_search] Retrieved: kyc=%d, news=%d, sars=%d",
            len(kyc_chunks), len(news_chunks), len(similar_sars),
        )

        return {
            "kyc_chunks":   kyc_chunks,
            "news_chunks":  news_chunks,
            "similar_sars": similar_sars,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"[rag_search] KYC={len(kyc_chunks)} chunks, "
                        f"News={len(news_chunks)} articles, "
                        f"SAR precedents={len(similar_sars)}."
                    ),
                }
            ],
        }

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(
        node_id: str,
        typology_scores: Dict[str, float],
        graph_summary: str,
    ) -> str:
        """
        Construct a rich natural-language query for embedding.

        Encodes:
          • The entity ID (for name-matching in KYC docs)
          • The dominant AML typology (to bias retrieval toward relevant regs)
          • Key graph topology signals from graph_summary
        """
        dominant_typology = max(typology_scores, key=typology_scores.get)
        score_str = ", ".join(
            f"{t}: {s:.2f}" for t, s in typology_scores.items()
        )
        # Truncate graph_summary to first 300 chars to keep query concise
        summary_snippet = (graph_summary or "")[:300]

        return (
            f"Account {node_id} AML investigation. "
            f"Suspected typologies: {score_str}. "
            f"Primary concern: {dominant_typology}. "
            f"Graph context: {summary_snippet}"
        )

    def _embed(self, text: str):
        """Embed query text using the configured encoder."""
        vec = self.embedding_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ------------------------------------------------------------------
    # Milvus search helpers
    # ------------------------------------------------------------------

    def _search_collection(
        self,
        collection_name: str,
        query_vec: list,
        top_k: int,
        output_fields: List[str],
        expr: Optional[str] = None,
        source_type: str = "unknown",
    ) -> List[RetrievedChunk]:
        """
        Generic Milvus ANN search helper.

        Args:
            expr : Milvus boolean expression for scalar pre-filtering.
                   Example: 'jurisdiction == "MA" and doc_type == "KYC"'
                   Pre-filtering happens on the inverted index BEFORE
                   the ANN search — more efficient than post-filtering.
        """
        try:
            col = Collection(collection_name)
            col.load()  # ensures collection is loaded into GPU/CPU memory

            results = col.search(
                data=[query_vec],
                anns_field="embedding",
                param=SEARCH_PARAMS,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
            )

            chunks: List[RetrievedChunk] = []
            for hit in results[0]:
                if hit.score < self.score_threshold:
                    continue   # Discard low-confidence retrievals
                entity = hit.entity
                chunks.append(
                    RetrievedChunk(
                        doc_id=str(hit.id),
                        source_type=source_type,
                        content=entity.get("text", ""),
                        score=round(float(hit.score), 4),
                        metadata={f: entity.get(f) for f in output_fields if f != "text"},
                    )
                )
            return chunks

        except MilvusException as exc:
            logger.error("[rag_search] Milvus error on '%s': %s", collection_name, exc)
            return []

    def _search_kyc(self, query_vec: list, node_id: str) -> List[RetrievedChunk]:
        """
        Search KYC documents.

        Scalar pre-filter: restrict to documents for this specific entity.
        Falls back to unfiltered search if entity_id yields no results.
        """
        entity_filter = f'entity_id == "{node_id}"'
        chunks = self._search_collection(
            COLLECTION_KYC, query_vec, self.kyc_top_k,
            output_fields=["text", "entity_id", "jurisdiction", "doc_type"],
            expr=entity_filter,
            source_type="kyc_document",
        )
        if not chunks:
            # Broaden search — entity might be stored under a related ID
            logger.debug("[rag_search] KYC entity filter returned 0 — broadening search.")
            chunks = self._search_collection(
                COLLECTION_KYC, query_vec, self.kyc_top_k,
                output_fields=["text", "entity_id", "jurisdiction", "doc_type"],
                expr=self.kyc_filter,   # user-provided broader filter or None
                source_type="kyc_document",
            )
        return chunks

    def _search_news(self, query_vec: list, node_id: str) -> List[RetrievedChunk]:
        """
        Search negative news articles.

        No entity_id filter — news is retrieved purely by semantic similarity
        to the query (entity name + typology context) so we surface adverse
        media about related persons/companies not in our KYC system.
        """
        return self._search_collection(
            COLLECTION_NEWS, query_vec, self.news_top_k,
            output_fields=["text", "entity_name", "source_url", "article_date"],
            source_type="negative_news",
        )

    def _search_sars(self, query_vec: list) -> List[RetrievedChunk]:
        """
        Search past SAR cases for similar typology + entity patterns.

        Used as few-shot examples for the SAR_Drafter LLM node.
        Retrieves narrative_en to give the LLM stylistic/structural guidance
        on how compliant SARs are written for the same typology.
        """
        return self._search_collection(
            COLLECTION_SARS, query_vec, self.sar_top_k,
            output_fields=["narrative_en", "typology", "jurisdiction"],
            source_type="sar_case",
        )

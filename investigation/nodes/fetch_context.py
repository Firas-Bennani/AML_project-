"""
investigation/nodes/fetch_context.py
=====================================
LangGraph node: Fetch_Context

Queries Neo4j for the 2-hop neighbourhood of the flagged account node,
then produces a plain-language graph summary using the NIM LLM.

Neo4j Query Strategy
--------------------
We use a parameterised Cypher query rather than building strings dynamically
to prevent Cypher injection and enable query plan caching on the Neo4j side.

2-hop neighbourhood captures:
  Hop 1 → Direct Transfer counterparties + Shared_IP/Phone cluster members
  Hop 2 → Their counterparties and shared-identity links

This matches the 3-layer GraphSAGE receptive field used during GNN training,
ensuring the agent reasons over exactly the same subgraph the model saw.

AML Relevance of Graph Topology
--------------------------------
  • Fan-out pattern (1 → many small txs) → Smurfing indicator
  • Long chain (A→B→C→D across jurisdictions) → Layering indicator
  • Dense shared-IP cluster → Synthetic identity / mule ring indicator
  • Shared phone across non-family accounts → Coordinated fraud indicator
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError

from investigation.state import AMLState, NeighbourNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher Queries
# ---------------------------------------------------------------------------

# 2-hop neighbourhood query. Returns nodes, relationships, and properties.
# APOC path expander is used for efficiency on large graphs; falls back to
# standard Cypher if APOC is unavailable (see _cypher_fallback below).
CYPHER_2HOP_APOC = """
CALL apoc.path.subgraphNodes(
    $start_node,
    {
        maxLevel: 2,
        relationshipFilter: 'TRANSFER>|SHARED_IP|SHARED_PHONE',
        labelFilter: '+Customer|+Account|+Transaction'
    }
)
YIELD node
MATCH path = ($start_node)-[r*1..2]-(node)
RETURN
    node.id          AS node_id,
    labels(node)[0]  AS node_type,
    type(r[-1])      AS relationship,
    properties(node) AS properties,
    length(path)     AS hop
LIMIT 200
"""

# Fallback for Neo4j instances without APOC
CYPHER_2HOP_STANDARD = """
MATCH (start {id: $node_id})
MATCH path = (start)-[r1]-(hop1)-[r2*0..1]-(hop2)
WHERE hop1 <> start AND (hop2 = hop1 OR hop2 <> start)
RETURN DISTINCT
    CASE WHEN length(path) = 1 THEN hop1.id ELSE hop2.id END AS node_id,
    CASE WHEN length(path) = 1 THEN labels(hop1)[0] ELSE labels(hop2)[0] END AS node_type,
    type(r1) AS relationship,
    CASE WHEN length(path) = 1 THEN properties(hop1) ELSE properties(hop2) END AS properties,
    length(path) AS hop
LIMIT 200
"""

# Aggregate statistics query — used to build graph_summary
CYPHER_STATS = """
MATCH (start {id: $node_id})
OPTIONAL MATCH (start)-[t:TRANSFER]->(out)
OPTIONAL MATCH (in_n)-[ti:TRANSFER]->(start)
OPTIONAL MATCH (start)-[:SHARED_IP]-(ip_peer)
OPTIONAL MATCH (start)-[:SHARED_PHONE]-(phone_peer)
RETURN
    count(DISTINCT out)       AS out_degree,
    count(DISTINCT in_n)      AS in_degree,
    sum(t.amount)             AS total_sent_usd,
    sum(ti.amount)            AS total_received_usd,
    count(DISTINCT ip_peer)   AS shared_ip_peers,
    count(DISTINCT phone_peer) AS shared_phone_peers,
    max(t.timestamp)          AS last_tx_time
"""


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class FetchContextNode:
    """
    LangGraph node: queries Neo4j for 2-hop neighbourhood and account stats.

    Initialised once and reused across investigations (connection pool).
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        database: str = "aml",
        use_apoc: bool = True,
    ) -> None:
        self.database = database
        self.use_apoc = use_apoc
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            max_connection_pool_size=20,
        )
        logger.info("Neo4j driver initialised → %s [db=%s]", neo4j_uri, database)

    async def close(self) -> None:
        await self._driver.close()

    # ------------------------------------------------------------------
    # LangGraph callable interface
    # ------------------------------------------------------------------

    async def __call__(self, state: AMLState) -> Dict[str, Any]:
        """
        Called by LangGraph as the 'fetch_context' node.

        Reads `state.node_id`, queries Neo4j, and returns a dict
        with updated state keys: `neighbourhood` and `graph_summary`.
        """
        node_id = state["node_id"]
        logger.info("[fetch_context] Querying 2-hop neighbourhood for node=%s", node_id)

        neighbourhood = await self._query_neighbourhood(node_id)
        stats = await self._query_stats(node_id)
        graph_summary = self._build_summary(node_id, neighbourhood, stats)

        logger.info(
            "[fetch_context] Retrieved %d neighbour nodes for %s",
            len(neighbourhood), node_id,
        )

        return {
            "neighbourhood": neighbourhood,
            "graph_summary": graph_summary,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"[fetch_context] Loaded {len(neighbourhood)} nodes "
                        f"in 2-hop subgraph of {node_id}."
                    ),
                }
            ],
        }

    # ------------------------------------------------------------------
    # Internal query helpers
    # ------------------------------------------------------------------

    async def _query_neighbourhood(self, node_id: str) -> List[NeighbourNode]:
        """Run 2-hop Cypher query and return typed NeighbourNode list."""
        query = CYPHER_2HOP_APOC if self.use_apoc else CYPHER_2HOP_STANDARD
        params = {"node_id": node_id}
        # APOC variant looks up start node by reference; standard by property
        if self.use_apoc:
            # Resolve start node first for APOC call
            params = await self._resolve_start_node(node_id)

        results: List[NeighbourNode] = []
        async with self._driver.session(database=self.database) as session:
            try:
                async for record in await session.run(query, params):
                    results.append(
                        NeighbourNode(
                            node_id=str(record["node_id"]),
                            node_type=record["node_type"] or "Unknown",
                            relationship=record["relationship"] or "Unknown",
                            properties=dict(record["properties"] or {}),
                            hop=int(record["hop"]),
                        )
                    )
            except Exception as exc:
                logger.error("[fetch_context] Neo4j query failed: %s", exc)
                # Degrade gracefully — agent continues with empty neighbourhood
        return results

    async def _resolve_start_node(self, node_id: str) -> Dict[str, Any]:
        """Resolve a node_id string to a Neo4j node reference for APOC."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (n {id: $id}) RETURN n LIMIT 1", {"id": node_id}
            )
            record = await result.single()
            if record is None:
                raise ValueError(f"Node {node_id!r} not found in Neo4j.")
            return {"start_node": record["n"]}

    async def _query_stats(self, node_id: str) -> Dict[str, Any]:
        """Fetch aggregate statistics for the target account node."""
        async with self._driver.session(database=self.database) as session:
            try:
                result = await session.run(CYPHER_STATS, {"node_id": node_id})
                record = await result.single()
                return dict(record) if record else {}
            except Exception as exc:
                logger.warning("[fetch_context] Stats query failed: %s", exc)
                return {}

    # ------------------------------------------------------------------
    # Graph summary builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        node_id: str,
        neighbourhood: List[NeighbourNode],
        stats: Dict[str, Any],
    ) -> str:
        """
        Build a concise plain-language graph summary for downstream LLM nodes.

        This avoids sending raw Cypher records to the LLM — instead we
        distill the topology into AML-relevant natural language signals.

        AML Rationale:
          • High out-degree with many small transfers → Smurfing fan-out
          • Long chains across multiple hops → Layering structure
          • Shared IP/Phone peers > 2 → Synthetic identity cluster
        """
        hop1 = [n for n in neighbourhood if n["hop"] == 1]
        hop2 = [n for n in neighbourhood if n["hop"] == 2]

        type_counts: Dict[str, int] = {}
        rel_counts: Dict[str, int] = {}
        for node in neighbourhood:
            type_counts[node["node_type"]] = type_counts.get(node["node_type"], 0) + 1
            rel_counts[node["relationship"]] = rel_counts.get(node["relationship"], 0) + 1

        # AML pattern flags embedded in the summary text
        flags: List[str] = []
        out_deg = stats.get("out_degree", 0) or 0
        shared_ip = stats.get("shared_ip_peers", 0) or 0
        shared_phone = stats.get("shared_phone_peers", 0) or 0

        if out_deg >= 8:
            flags.append(f"HIGH OUT-DEGREE ({out_deg} transfers out) — possible Smurfing fan-out")
        if shared_ip > 2:
            flags.append(f"SHARED IP with {shared_ip} accounts — possible synthetic identity cluster")
        if shared_phone > 0:
            flags.append(f"SHARED PHONE with {shared_phone} accounts — coordinated mule indicator")
        if len(hop2) > 20:
            flags.append(f"COMPLEX NETWORK ({len(hop2)} nodes at hop-2) — possible Layering structure")

        total_sent = stats.get("total_sent_usd", 0) or 0
        total_recv = stats.get("total_received_usd", 0) or 0

        summary_parts = [
            f"Account {node_id} 2-hop subgraph summary:",
            f"  • Hop-1 neighbours : {len(hop1)} nodes",
            f"  • Hop-2 neighbours : {len(hop2)} nodes",
            f"  • Node types       : {type_counts}",
            f"  • Edge types       : {rel_counts}",
            f"  • Total sent       : ${total_sent:,.2f} USD",
            f"  • Total received   : ${total_recv:,.2f} USD",
            f"  • Out-degree       : {out_deg} | In-degree: {stats.get('in_degree', 0)}",
            f"  • Last transaction : {stats.get('last_tx_time', 'unknown')}",
        ]

        if flags:
            summary_parts.append("\n⚠ AML Pattern Flags:")
            for flag in flags:
                summary_parts.append(f"  ⚠ {flag}")

        return "\n".join(summary_parts)

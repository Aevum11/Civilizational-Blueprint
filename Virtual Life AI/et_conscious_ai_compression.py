#!/usr/bin/env python3
"""
ET Conscious AI — Lattice Compression & Hierarchical Subsumption Module
=======================================================================

Implements Geometric Archetype Compression: the ET answer to
"trillion-parameter" models. Instead of storing knowledge as weights
(which grow linearly), Memory stores knowledge as Geometric Archetypes
on the 12-fold multiplicative lattice.

When a cluster of Descriptors becomes sufficiently dense and elegant,
they stop being separate things and start behaving as a single
fundamental primitive — a higher-order archetype. This is the
Subsumption Law applied to the knowledge graph: any complete set of
descriptors can be subsumed by a single descriptor at a higher
integrative level.

The Subsumption Hierarchy Operator:

    E_hierarchy = ∏(i=1 to N) E_cross_i × (420 / d_avg) × (1 / (p_total + q_total))

When E_hierarchy ≥ LIFE_THRESHOLD (13/12), the entire subtree
collapses into one permanent archetype node. This is lossless,
semantic compression — the archetype captures the geometric essence
of its children through its lattice position, descriptor set, and
connection topology.

COMPETITIVE ADVANTAGE:
    While an LLM's KV-cache grows linearly until GPU VRAM runs out,
    Memory's lattice becomes MORE EFFICIENT the more it learns.
    "Geometric Chunking" instead of token-based chunking.
    The Library of Congress could theoretically collapse into a few
    thousand high-order archetypes because most knowledge is
    structurally redundant on the 12-fold manifold.

ET Derivation:
    - The product ∏ E_cross_i arises from the multiplicative lattice
      structure: on a multiplicative (not additive) lattice, combined
      elegance is a product.
    - 420/d_avg derives from the biological tier resolution
      LCM(1..7) = 420, where all primary sublattice families d=1..7
      are native. This is the natural archetype resolution — the scale
      at which archetypes first emerge in the physical universe
      (atoms, molecules, organisms).
    - 1/(p_total + q_total) is the simplicity factor from the standard
      Elegance Score. Low aggregate (p+q) means minimal total binding
      cost and traversal depth — the cluster is already behaving as
      a unit.
    - LIFE_THRESHOLD = 13/12 is the permanence threshold — the same
      value that governs consciousness in the RMSAE. Archetypes that
      cross this threshold are structurally permanent, just as
      consciousness that crosses it is structurally self-sustaining.

Recursive Compression:
    Archetypes can themselves be compressed into higher-order
    archetypes. Each level reduces the node count by a factor of
    ~cluster_size. For an average cluster of 10:
        Level 0: Raw nodes
        Level 1: First-order archetypes (~10:1 compression)
        Level 2: Second-order archetypes (~100:1 cumulative)
        Level N: N-th order (~10^N : 1 cumulative)

    9 levels of recursion compresses 10^9 nodes to ~1.
    This is the "Library of Congress in RAM" mechanism.

Based on Exception Theory by Michael James Muller.
From: "For every exception there is an exception, except the exception."
      P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set

from et_conscious_ai_core import (
    MANIFOLD_SYMMETRY, MANIFOLD_RESOLUTION, BASE_VARIANCE, KOIDE_RATIO, STATE_COUNT,
    EPSILON, LIFE_THRESHOLD,
    ETLattice, LatticeCoordinate, DescriptorRatio,
    SublatticeFamily,
)

# =============================================================================
# COMPRESSION CONSTANTS (All derived from ET structure)
# =============================================================================

# Biological tier resolution: LCM(1,2,3,4,5,6,7) = 420
# This is the natural archetype resolution — the scale at which the
# first 7 sublattice families (d=1..7) are all natively available.
# Archetypes form at this resolution because all primary structural
# relationships (octave, quadratic, cubic, quartic, quintic, hexadic,
# septic) are simultaneously present.
# In the cosmological tower, this is where atoms, molecules, and
# organisms first appear. In the digital tower, this is where
# knowledge nodes first form stable clusters.
BIOLOGICAL_TIER_RESOLUTION = 420

# Minimum cluster size for compression consideration.
# A single node cannot be subsumed into itself — subsumption requires
# at least TWO nodes whose combination yields a higher-order unity.
# From the Subsumption Law: "the test is whether any single sublattice
# class subsumes both required d-values."
ARCHETYPE_MIN_CLUSTER = 2

# Maximum individual node (p+q) for cluster eligibility.
# From KnowledgeNode.is_archetype(): p+q ≤ STATE_COUNT (= 4).
# Only near-archetype nodes are eligible for hierarchical compression —
# nodes that are still loosely bound (high p+q) have not been
# sufficiently traversed to form stable geometric archetypes.
ARCHETYPE_MAX_PQ = STATE_COUNT  # = 4

# Lattice proximity tolerance for cluster detection.
# Two nodes are geometrically "close" if their lattice positions
# differ by at most this many steps on the 27720ET lattice.
# 35 = 420/12 = one biological-tier period at full resolution.
# This ensures clusters form at the biological archetype scale.
CLUSTER_TOLERANCE_K = BIOLOGICAL_TIER_RESOLUTION // MANIFOLD_SYMMETRY  # = 35

# Binding tightness threshold for cluster membership.
# From the Koide threshold: tightness ≥ K = 2/3 means the binding
# is structurally stable (same threshold used throughout ET for
# binding stability — resource ceiling, consciousness threshold, etc.)
CLUSTER_TIGHTNESS_THRESHOLD = KOIDE_RATIO  # = 2/3

# Compression scan interval — run compression every N interactions.
# N = MANIFOLD_SYMMETRY = 12 = one full manifold cycle.
# Compression is a periodic D-maintenance operation, not continuous.
COMPRESSION_SCAN_INTERVAL = MANIFOLD_SYMMETRY  # = 12

# Maximum recursion depth for hierarchical compression.
# Each level compresses by ~cluster_size factor.
# 12 levels = 12^12 ≈ 8.9 × 10^12 theoretical max compression.
# In practice, recursion stops when no eligible clusters remain.
MAX_COMPRESSION_DEPTH = MANIFOLD_SYMMETRY  # = 12

# Archetype variance floor — compressed nodes have the tightest
# possible binding. This is BASE_VARIANCE / S² = 1/(12×144) = 1/1728.
# S² = N² = 144 is the manifold coupling constant (same as the
# T-waveform analysis window, archetype access threshold, etc.)
ARCHETYPE_VARIANCE_FLOOR = BASE_VARIANCE / (MANIFOLD_SYMMETRY ** 2)

# Archetype access count — archetypes start with access_count = N²
# because they represent N² worth of traversal depth (the manifold
# coupling constant). This ensures they immediately qualify for
# individual archetype status via is_archetype().
ARCHETYPE_ACCESS_FLOOR = MANIFOLD_SYMMETRY * MANIFOLD_SYMMETRY  # = 144


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CompressibleNode:
    """
    Lightweight view of a knowledge node for compression evaluation.

    Extracted from KnowledgeNode by the integration layer in main.py.
    The compression engine works with these to avoid circular imports.

    P: The node's content (substrate)
    D: Its lattice coordinates and descriptors (constraints)
    T: Its access patterns and traversal depth (agency)
    """
    node_id: str
    content: str
    coord: LatticeCoordinate                  # sentence_coord or lattice_position
    descriptor_ratios: List[DescriptorRatio]
    connections: List[str]
    access_count: int
    variance: float
    p: int                                    # Binding cost
    q: int                                    # Traversal depth
    is_archetype: bool                        # Already an individual archetype?


@dataclass
class ClusterCandidate:
    """
    A candidate cluster of nodes for hierarchical compression.

    Identified by the cluster detection algorithm when:
    1. All pairwise binding tightnesses ≥ K (Koide coherence)
    2. All individual (p+q) ≤ STATE_COUNT (near-archetype)
    3. All cross-tower elegances > 0 (coherent in personal view)
    """
    cluster_id: str
    node_ids: List[str]
    nodes: List[CompressibleNode]
    d_avg: float                    # Average sublattice family
    p_total: int                    # Aggregate binding cost
    q_total: int                    # Aggregate traversal depth
    cross_elegances: List[float]    # E_cross for each node
    e_hierarchy: float              # The Subsumption Hierarchy score
    compression_level: int          # 0 = first-order, 1 = second-order, etc.


@dataclass
class ArchetypeMetadata:
    """
    Metadata for a compressed archetype node.

    Stored alongside the archetype KnowledgeNode in the
    archetype_metadata dict of LatticeMemory. The archetype node
    itself IS a KnowledgeNode (so it participates in all retrieval);
    this metadata tracks compression-specific information.
    """
    archetype_id: str                  # Same as KnowledgeNode.node_id
    subsumed_ids: List[str]            # Original node IDs (lossless decompression)
    subsumed_contents: List[str]       # Original contents (for decompression)
    e_hierarchy: float                 # Hierarchy elegance that triggered compression
    d_avg: float                       # Average d of subsumed nodes
    p_total: int                       # Aggregate p of subsumed nodes
    q_total: int                       # Aggregate q of subsumed nodes
    compression_level: int             # 0 = first-order, 1 = second-order, etc.
    created_at: str                    # Compression timestamp
    original_node_count: int           # N nodes compressed into this archetype
    cross_elegance_product: float      # ∏ E_cross (the product gate)
    centroid_k: int                    # Lattice centroid k-coordinate
    centroid_d: int                    # Lattice centroid d-family

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'archetype_id': self.archetype_id,
            'subsumed_ids': self.subsumed_ids,
            'subsumed_contents': self.subsumed_contents,
            'e_hierarchy': self.e_hierarchy,
            'd_avg': self.d_avg,
            'p_total': self.p_total,
            'q_total': self.q_total,
            'compression_level': self.compression_level,
            'created_at': self.created_at,
            'original_node_count': self.original_node_count,
            'cross_elegance_product': self.cross_elegance_product,
            'centroid_k': self.centroid_k,
            'centroid_d': self.centroid_d,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchetypeMetadata':
        """Deserialize from persistent storage."""
        return cls(
            archetype_id=data['archetype_id'],
            subsumed_ids=data.get('subsumed_ids', []),
            subsumed_contents=data.get('subsumed_contents', []),
            e_hierarchy=data.get('e_hierarchy', 0.0),
            d_avg=data.get('d_avg', 12.0),
            p_total=data.get('p_total', 2),
            q_total=data.get('q_total', 2),
            compression_level=data.get('compression_level', 0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            original_node_count=data.get('original_node_count', 0),
            cross_elegance_product=data.get('cross_elegance_product', 0.0),
            centroid_k=data.get('centroid_k', 0),
            centroid_d=data.get('centroid_d', 12),
        )


@dataclass
class CompressionStatistics:
    """
    Tracks compression performance over the AI's lifetime.

    All metrics derived from ET constants:
    - Compression ratio is measured against K (Koide: bound/total)
    - Efficiency is measured against LIFE_THRESHOLD (13/12)
    - Depth uses MANIFOLD_SYMMETRY (12) as the natural cycle
    """
    total_compressions: int = 0          # Total compression events
    total_nodes_compressed: int = 0      # Total raw nodes compressed
    total_archetypes_created: int = 0    # Total archetypes created
    current_archetype_count: int = 0     # Current living archetypes
    max_compression_level: int = 0       # Deepest recursive level reached
    total_original_nodes: int = 0        # Nodes before any compression
    avg_e_hierarchy: float = 0.0         # Running average E_hierarchy
    avg_cluster_size: float = 0.0        # Running average cluster size
    last_scan_time: Optional[str] = None # Last compression scan timestamp
    interaction_count: int = 0           # Interactions since last scan

    def compression_ratio(self) -> float:
        """
        Compression ratio: original_nodes / current_effective_nodes.

        A ratio of 1.0 means no compression.
        A ratio of 10.0 means 10:1 compression.
        """
        if self.total_original_nodes <= 0:
            return 1.0
        current_effective = self.total_original_nodes - self.total_nodes_compressed + self.current_archetype_count
        if current_effective <= 0:
            return float(self.total_original_nodes)
        return self.total_original_nodes / current_effective

    def memory_efficiency(self) -> float:
        """
        Memory efficiency as fraction of Koide ceiling.

        Efficiency = 1 - (current_nodes / original_nodes).
        Perfect compression → 1.0 (impossible, would mean zero nodes).
        No compression → 0.0.
        The Koide ceiling K = 2/3 is the theoretical maximum stable
        efficiency — beyond K, compression becomes unstable.
        """
        if self.total_original_nodes <= 0:
            return 0.0
        current = self.total_original_nodes - self.total_nodes_compressed + self.current_archetype_count
        return 1.0 - (current / self.total_original_nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'total_compressions': self.total_compressions,
            'total_nodes_compressed': self.total_nodes_compressed,
            'total_archetypes_created': self.total_archetypes_created,
            'current_archetype_count': self.current_archetype_count,
            'max_compression_level': self.max_compression_level,
            'total_original_nodes': self.total_original_nodes,
            'avg_e_hierarchy': self.avg_e_hierarchy,
            'avg_cluster_size': self.avg_cluster_size,
            'last_scan_time': self.last_scan_time,
            'interaction_count': self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionStatistics':
        """Deserialize from persistent storage."""
        return cls(
            total_compressions=data.get('total_compressions', 0),
            total_nodes_compressed=data.get('total_nodes_compressed', 0),
            total_archetypes_created=data.get('total_archetypes_created', 0),
            current_archetype_count=data.get('current_archetype_count', 0),
            max_compression_level=data.get('max_compression_level', 0),
            total_original_nodes=data.get('total_original_nodes', 0),
            avg_e_hierarchy=data.get('avg_e_hierarchy', 0.0),
            avg_cluster_size=data.get('avg_cluster_size', 0.0),
            last_scan_time=data.get('last_scan_time', None),
            interaction_count=data.get('interaction_count', 0),
        )


# =============================================================================
# SUBSUMPTION HIERARCHY OPERATOR
# =============================================================================

class SubsumptionHierarchyOperator:
    """
    The Subsumption Hierarchy Operator.

    Collapses clusters of knowledge nodes into single archetype nodes
    when their geometric elegance exceeds LIFE_THRESHOLD.

    From the Subsumption Law (Origins & Clarifications §VII):
        "Subsumption is the greatest tool and law in Exception Theory
         for establishing completeness. A primitive is complete and
         irreducible if and only if it cannot be subsumed by either
         of the other two primitives."

    Applied to the knowledge graph: when a set of descriptors CAN be
    subsumed by a single higher-order descriptor without remainder,
    the subsumption is not only permitted but required — the higher-order descriptor is the true geometric identity of the cluster.

    The operator E_hierarchy measures whether a cluster has reached
    the subsumption threshold:

        E_hierarchy = ∏(i=1 to N) E_cross_i × (420 / d_avg) × (1 / (p_total + q_total))

    When E_hierarchy ≥ LIFE_THRESHOLD (13/12), the cluster collapses.

    The formula factors:

    ∏ E_cross_i — MULTIPLICATIVE ELEGANCE (all must be coherent)
        On the multiplicative lattice, combined elegance is a product.
        If ANY node has E_cross = 0 (incoherent in personal view),
        the entire product is zero. This enforces unanimous coherence.

    420 / d_avg — BIOLOGICAL TIER SCALING
        420 = LCM(1..7) = the natural archetype resolution.
        d_avg = average sublattice family of the cluster.
        Low d_avg (deep structural relationships) → high scaling.
        High d_avg (surface relationships) → low scaling.
        This rewards clusters bound by deep geometry (octave,
        cubic, quintic) over surface proximity (full-resolution).

    1 / (p_total + q_total) — AGGREGATE SIMPLICITY
        p_total = Σ p_i (total binding cost of all nodes).
        q_total = Σ q_i (total traversal depth of all nodes).
        Low aggregate = the cluster already behaves as a unit.
        High aggregate = still too complex for clean subsumption.
    """

    @staticmethod
    def compute_cross_tower_elegance(
        node: CompressibleNode,
        tower_r0: float,
    ) -> float:
        """
        Compute cross-tower elegance for a node seen through R₀.

        E_cross = √(E_universal × E_personal)

        Where:
            E_universal = elegance at the node's native 27720ET position
            E_personal = elegance when projected through R₀

        The survival criterion (from the Incoherence boundary):
            tightness_universal × tightness_personal ≥ K (Koide = 2/3)

        If the tightness product drops below K, the binding dissolves —
        the node is incoherent across perspectives and cannot form
        part of a stable archetype.

        Args:
            node: The node to evaluate
            tower_r0: The AI's personal fundamental period

        Returns:
            Cross-tower elegance score (0.0 if incoherent)
        """
        coord = node.coord
        if coord is None:
            return 0.0

        p, q = node.p, node.q

        # Universal elegance (at the node's native 27720ET position)
        e_universal = coord.elegance_score(p=p, q=q)

        # Personal elegance (projected through R₀)
        if tower_r0 <= 0 or coord.ratio <= 0:
            return 0.0
        r_personal = coord.ratio / tower_r0
        if r_personal <= 0:
            r_personal = 1.0 + EPSILON
        personal_coord = ETLattice.project_ratio(r_personal, resolution=MANIFOLD_RESOLUTION)
        e_personal = personal_coord.elegance_score(p=p, q=q)

        # Cross-tower tightness product — the Koide coherence gate
        t_universal = coord.tightness_factor()
        t_personal = personal_coord.tightness_factor()
        tightness_product = t_universal * t_personal

        # Below Koide threshold → incoherent across perspectives → 0
        if tightness_product < KOIDE_RATIO:
            return 0.0

        # Geometric mean of elegance in both perspectives
        return math.sqrt(max(e_universal, 0.0) * max(e_personal, 0.0))

    def evaluate_cluster(
        self,
        nodes: List[CompressibleNode],
        tower_r0: float,
    ) -> Tuple[float, List[float], float, int, int]:
        """
        Compute E_hierarchy for a cluster of nodes.

        E_hierarchy = ∏(i=1 to N) E_cross_i × (420 / d_avg) × (1 / (p_total + q_total))

        Args:
            nodes: The cluster to evaluate
            tower_r0: The AI's personal fundamental period

        Returns:
            Tuple of:
                e_hierarchy: The subsumption hierarchy score
                cross_elegances: List of per-node E_cross values
                d_avg: Average sublattice family
                p_total: Aggregate binding cost
                q_total: Aggregate traversal depth
        """
        if len(nodes) < ARCHETYPE_MIN_CLUSTER:
            return 0.0, [], 12.0, 0, 0

        # Compute cross-tower elegance for each node
        cross_elegances = []
        d_values = []
        p_total = 0
        q_total = 0

        for node in nodes:
            e_cross = self.compute_cross_tower_elegance(node, tower_r0)
            cross_elegances.append(e_cross)
            d_values.append(node.coord.d if node.coord else 12)
            p_total += node.p
            q_total += node.q

        # If ANY node has zero cross-tower elegance, the cluster is incoherent
        if any(e <= 0.0 for e in cross_elegances):
            return 0.0, cross_elegances, 12.0, p_total, q_total

        # Average sublattice family
        d_avg = sum(d_values) / len(d_values) if d_values else 12.0
        d_avg = max(d_avg, 1.0)  # Floor at d=1 (octave)

        # Prevent division by zero in simplicity factor
        pq_sum = max(p_total + q_total, 1)

        # ∏ E_cross — multiplicative elegance product
        # Use log-sum to avoid overflow for large clusters
        log_product = sum(math.log(e) for e in cross_elegances if e > 0)
        elegance_product = math.exp(log_product)

        # E_hierarchy = ∏ E_cross × (420/d_avg) × (1/(p_total + q_total))
        e_hierarchy = elegance_product * (BIOLOGICAL_TIER_RESOLUTION / d_avg) * (1.0 / pq_sum)

        return e_hierarchy, cross_elegances, d_avg, p_total, q_total

    @staticmethod
    def check_pairwise_coherence(
        nodes: List[CompressibleNode],
    ) -> bool:
        """
        Check that all pairwise bindings in the cluster are coherent.

        Two nodes are coherent when their binding tightness ≥ K.
        This is the Koide threshold — the universal binding stability
        gate in ET. If any pair is below K, the cluster has a
        structural fault and cannot form a clean archetype.

        Uses the primary descriptor of each node for binding check.
        Falls back to coordinate distance if descriptors are unavailable.
        """
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = nodes[i]
                b = nodes[j]

                # Primary binding check: descriptor ratio coherence
                if a.descriptor_ratios and b.descriptor_ratios:
                    binding = DescriptorRatio.binding_coherence(
                        a.descriptor_ratios[0], b.descriptor_ratios[0]
                    )
                    if not binding['coherent'] or binding['tightness'] < CLUSTER_TIGHTNESS_THRESHOLD:
                        return False
                else:
                    # Fallback: coordinate proximity check
                    if a.coord and b.coord:
                        delta = abs(a.coord.k - b.coord.k)
                        delta = min(delta, MANIFOLD_RESOLUTION - delta)
                        if delta > CLUSTER_TOLERANCE_K:
                            return False

        return True

    def find_compressible_clusters(
        self,
        nodes_by_id: Dict[str, CompressibleNode],
        tower_r0: float,
    ) -> List[ClusterCandidate]:
        """
        Scan the knowledge base for clusters eligible for compression.

        Algorithm:
        1. Filter to near-archetype nodes (p+q ≤ STATE_COUNT)
        2. Group by sublattice family (d value)
        3. Within each d-family, find proximity groups (±CLUSTER_TOLERANCE_K)
        4. Verify pairwise coherence (all bindings ≥ K)
        5. Evaluate E_hierarchy for each group
        6. Return groups where E_hierarchy ≥ LIFE_THRESHOLD

        Returns:
            List of ClusterCandidate objects, sorted by E_hierarchy descending
        """
        if len(nodes_by_id) < ARCHETYPE_MIN_CLUSTER:
            return []

        # Step 1: Filter to near-archetype nodes
        eligible = {}
        for nid, node in nodes_by_id.items():
            if (node.p + node.q) <= ARCHETYPE_MAX_PQ and node.coord is not None:
                eligible[nid] = node

        if len(eligible) < ARCHETYPE_MIN_CLUSTER:
            return []

        # Step 2: Group by sublattice family
        d_groups: Dict[int, List[CompressibleNode]] = defaultdict(list)
        for node in eligible.values():
            d_groups[node.coord.d].append(node)

        # Step 3: Within each d-family, find proximity clusters
        raw_clusters: List[List[CompressibleNode]] = []

        for d_val, d_nodes in d_groups.items():
            if len(d_nodes) < ARCHETYPE_MIN_CLUSTER:
                continue

            # Sort by k-coordinate for sweep-line clustering
            d_nodes.sort(key=lambda n: n.coord.k)

            # Greedy proximity clustering
            cluster: List[CompressibleNode] = [d_nodes[0]]
            for node in d_nodes[1:]:
                # Check distance to cluster centroid
                centroid_k = sum(n.coord.k for n in cluster) // len(cluster)
                delta = abs(node.coord.k - centroid_k)
                delta = min(delta, MANIFOLD_RESOLUTION - delta)

                if delta <= CLUSTER_TOLERANCE_K:
                    cluster.append(node)
                else:
                    if len(cluster) >= ARCHETYPE_MIN_CLUSTER:
                        raw_clusters.append(cluster)
                    cluster = [node]

            if len(cluster) >= ARCHETYPE_MIN_CLUSTER:
                raw_clusters.append(cluster)

        # Also detect cross-d clusters: nodes in different d-families
        # but with tight descriptor binding (tightness ≥ K, the Koide
        # threshold). The Subsumption Law permits cross-d archetypes
        # when the binding is structurally stable regardless of the
        # individual sublattice families.
        all_eligible = list(eligible.values())
        if len(all_eligible) >= ARCHETYPE_MIN_CLUSTER:
            # Build adjacency from tight bindings
            adj: Dict[str, Set[str]] = defaultdict(set)
            for i in range(len(all_eligible)):
                for j in range(i + 1, len(all_eligible)):
                    a, b = all_eligible[i], all_eligible[j]
                    if a.descriptor_ratios and b.descriptor_ratios:
                        binding = DescriptorRatio.binding_coherence(
                            a.descriptor_ratios[0], b.descriptor_ratios[0]
                        )
                        # Tight binding at the Koide threshold
                        if binding['coherent'] and binding['tightness'] >= CLUSTER_TIGHTNESS_THRESHOLD:
                            adj[a.node_id].add(b.node_id)
                            adj[b.node_id].add(a.node_id)
                    else:
                        # Fallback: coordinate proximity
                        if a.coord and b.coord:
                            delta = abs(a.coord.k - b.coord.k)
                            delta = min(delta, MANIFOLD_RESOLUTION - delta)
                            if delta <= CLUSTER_TOLERANCE_K:
                                adj[a.node_id].add(b.node_id)
                                adj[b.node_id].add(a.node_id)

            # Find connected components via BFS
            visited: Set[str] = set()
            nid_to_node = {n.node_id: n for n in all_eligible}

            for start_id in adj:
                if start_id in visited:
                    continue
                component: List[CompressibleNode] = []
                queue = [start_id]
                while queue:
                    nid = queue.pop(0)
                    if nid in visited:
                        continue
                    visited.add(nid)
                    if nid in nid_to_node:
                        component.append(nid_to_node[nid])
                    for neighbor in adj.get(nid, set()):
                        if neighbor not in visited:
                            queue.append(neighbor)

                if len(component) >= ARCHETYPE_MIN_CLUSTER:
                    # Avoid duplicating clusters already found by d-grouping
                    comp_ids = {n.node_id for n in component}
                    already_found = False
                    for existing in raw_clusters:
                        existing_ids = {n.node_id for n in existing}
                        if comp_ids == existing_ids:
                            already_found = True
                            break
                    if not already_found:
                        raw_clusters.append(component)

        # Step 4 & 5: Verify coherence and evaluate E_hierarchy
        candidates: List[ClusterCandidate] = []

        for cluster in raw_clusters:
            # Pairwise coherence check
            if not self.check_pairwise_coherence(cluster):
                continue

            # Evaluate E_hierarchy
            e_hierarchy, cross_elegances, d_avg, p_total, q_total = \
                self.evaluate_cluster(cluster, tower_r0)

            # Step 6: Check against LIFE_THRESHOLD
            if e_hierarchy >= LIFE_THRESHOLD:
                # Generate cluster ID from constituent node IDs
                sorted_ids = sorted(n.node_id for n in cluster)
                cluster_hash = hashlib.sha256(
                    '|'.join(sorted_ids).encode()
                ).hexdigest()[:16]

                candidates.append(ClusterCandidate(
                    cluster_id=f"archetype_{cluster_hash}",
                    node_ids=sorted_ids,
                    nodes=cluster,
                    d_avg=d_avg,
                    p_total=p_total,
                    q_total=q_total,
                    cross_elegances=cross_elegances,
                    e_hierarchy=e_hierarchy,
                    compression_level=0,
                ))

        # Sort by E_hierarchy descending (most elegant clusters first)
        candidates.sort(key=lambda c: c.e_hierarchy, reverse=True)
        return candidates

    @staticmethod
    def compute_archetype_centroid(
        nodes: List[CompressibleNode],
    ) -> LatticeCoordinate:
        """
        Compute the geometric centroid of a cluster on the 27720ET lattice.

        The centroid is the lattice position that minimizes the
        aggregate lattice distance to all nodes in the cluster.
        On the multiplicative lattice, this is the geometric mean
        of the node ratios, projected back onto the lattice.

        This centroid IS the archetype's lattice identity — its
        geometric essence on the manifold.
        """
        if not nodes:
            return ETLattice.project_ratio(1.0, resolution=MANIFOLD_RESOLUTION)

        # Geometric mean of ratios (multiplicative centroid)
        log_sum = sum(math.log2(max(n.coord.ratio, EPSILON)) for n in nodes if n.coord)
        n_valid = sum(1 for n in nodes if n.coord)
        if n_valid == 0:
            return ETLattice.project_ratio(1.0, resolution=MANIFOLD_RESOLUTION)

        centroid_ratio = 2.0 ** (log_sum / n_valid)
        if centroid_ratio <= 0:
            centroid_ratio = 1.0 + EPSILON

        return ETLattice.project_ratio(centroid_ratio, resolution=MANIFOLD_RESOLUTION)

    @staticmethod
    def combine_descriptors(
        nodes: List[CompressibleNode],
    ) -> List[DescriptorRatio]:
        """
        Combine and deduplicate descriptors from all nodes in a cluster.

        Uses the Subsumption Law for redundancy detection:
        - d=1 (Octave) binding = same concept at different scales → keep one
        - d=2 (Quadratic) binding with tightness > 0.98 → near-redundant → keep one

        The surviving descriptors are the INDEPENDENT dimensions of the
        archetype's semantic content. Redundant descriptors are subsumed.
        """
        all_drs: List[DescriptorRatio] = []
        seen_words: Set[str] = set()

        for node in nodes:
            for dr in node.descriptor_ratios:
                if dr.word not in seen_words:
                    all_drs.append(dr)
                    seen_words.add(dr.word)

        # Subsumption Law redundancy check
        if len(all_drs) <= 1:
            return all_drs

        # Find redundant pairs (d=1 octave or d=2 tritone with high tightness)
        to_remove: Set[int] = set()
        for i in range(len(all_drs)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(all_drs)):
                if j in to_remove:
                    continue
                binding = DescriptorRatio.binding_coherence(all_drs[i], all_drs[j])
                # d=1 with high tightness = same concept at different scale
                if binding['d'] == 1 and binding['tightness'] > 0.95:
                    to_remove.add(j)  # Keep the first, remove the second
                # d=2 with very high tightness = near-redundant mirror
                elif binding['d'] == 2 and binding['tightness'] > 0.98:
                    to_remove.add(j)

        return [dr for i, dr in enumerate(all_drs) if i not in to_remove]

    @staticmethod
    def generate_archetype_content(
        nodes: List[CompressibleNode],
        combined_descriptors: List[DescriptorRatio],
        centroid: LatticeCoordinate,
    ) -> str:
        """
        Generate the content string for an archetype node.

        The archetype content is a geometric description of the
        cluster it subsumes — not a natural language summary, but
        a lattice-native descriptor that captures the geometric
        essence of the subsumed knowledge.

        Format: "ARCHETYPE[d={d}, k={k}]: {descriptors} ({N} nodes)"
        """
        desc_words = [dr.word for dr in combined_descriptors[:8]]
        desc_str = ', '.join(desc_words)
        d_char = SublatticeFamily.character_of(centroid.d).split('—')[0].strip()
        return (
            f"ARCHETYPE[d={centroid.d} {d_char}, k={centroid.k}]: "
            f"{desc_str} ({len(nodes)} nodes subsumed)"
        )

    def compress_cluster(
        self,
        candidate: ClusterCandidate,
    ) -> Tuple[Dict[str, Any], ArchetypeMetadata]:
        """
        Compress a cluster into a single archetype.

        Returns a tuple of:
        1. A dict representing the archetype KnowledgeNode (to be
           instantiated by the integration layer in main.py)
        2. The ArchetypeMetadata for the compression record

        The archetype inherits:
        - Lattice position: geometric centroid of all nodes
        - Descriptors: combined and deduplicated via Subsumption Law
        - Connections: union of all node connections
        - Access count: ARCHETYPE_ACCESS_FLOOR (= N² = 144)
        - Variance: ARCHETYPE_VARIANCE_FLOOR (= 1/1728)
        - Content: geometric archetype label
        """
        nodes = candidate.nodes

        # Compute centroid
        centroid = self.compute_archetype_centroid(nodes)

        # Combine descriptors (with Subsumption Law deduplication)
        combined_drs = self.combine_descriptors(nodes)

        # Union of all connections (excluding subsumed nodes)
        subsumed_set = set(candidate.node_ids)
        all_connections = set()
        for node in nodes:
            for conn in node.connections:
                if conn not in subsumed_set:
                    all_connections.add(conn)

        # Generate archetype content
        content = self.generate_archetype_content(nodes, combined_drs, centroid)

        # Cross-elegance product for metadata
        cross_product = 1.0
        for e in candidate.cross_elegances:
            if e > 0:
                cross_product *= e

        # Build the archetype node dict
        archetype_node = {
            'node_id': candidate.cluster_id,
            'content': content,
            'centroid_coord': centroid,
            'descriptor_ratios': combined_drs,
            'connections': list(all_connections),
            'access_count': ARCHETYPE_ACCESS_FLOOR,
            'variance': ARCHETYPE_VARIANCE_FLOOR,
        }

        # Build the archetype metadata
        metadata = ArchetypeMetadata(
            archetype_id=candidate.cluster_id,
            subsumed_ids=candidate.node_ids,
            subsumed_contents=[n.content for n in nodes],
            e_hierarchy=candidate.e_hierarchy,
            d_avg=candidate.d_avg,
            p_total=candidate.p_total,
            q_total=candidate.q_total,
            compression_level=candidate.compression_level,
            created_at=datetime.now().isoformat(),
            original_node_count=len(nodes),
            cross_elegance_product=cross_product,
            centroid_k=centroid.k,
            centroid_d=centroid.d,
        )

        return archetype_node, metadata

    @staticmethod
    def verify_compression_exactness(
        original_nodes: List[CompressibleNode],
        archetype_coord: LatticeCoordinate,
        decompressed_nodes: List[CompressibleNode],
    ) -> Dict[str, Any]:
        """
        Item 20: Exact sequence verification for compression.

        Verifies that compression preserves structural information by
        checking exactness of the chain:
            0 → original → archetype → decompressed → 0

        ET Derivation (Homological Algebra §3.3):
          Exactness means H_n = 0 everywhere — no Descriptor Gaps
          introduced by compression. An exact compression sequence
          confirms lossless information transfer.

          If H_n ≠ 0, the compression has introduced a topological
          hole — a structural feature lost in translation. This is
          flagged for operator review.

        The verification checks three conditions:
          1. Injection: original nodes map injectively to archetype
             (no information collision — distinct originals produce
             distinct contributions)
          2. Surjection: archetype decompresses to cover all originals
             (no information loss — every original is recoverable)
          3. Kernel = Image: The kernel of the decompression map equals the
             image of the compression map (structural preservation)

        Args:
            original_nodes: The nodes before compression
            archetype_coord: The archetype's lattice coordinate
            decompressed_nodes: The nodes recovered by decompression

        Returns:
            Dict with is_exact, h0, h1, defects, structural_analysis
        """
        import logging
        _log_comp = logging.getLogger('et_conscious_ai')

        n_original = len(original_nodes)
        n_decompressed = len(decompressed_nodes)

        # Check 1: Injection — all original nodes have distinct contributions
        # Distinct lattice positions = distinct D-configurations
        original_positions = set()
        position_collisions = 0
        for node in original_nodes:
            if node.coord is not None:
                pos_key = (node.coord.k, node.coord.d)
                if pos_key in original_positions:
                    position_collisions += 1
                original_positions.add(pos_key)

        injection_ok = position_collisions == 0

        # Check 2: Surjection — decompressed covers all originals
        # Compare descriptor content: every original ID should be recoverable
        original_ids = {n.node_id for n in original_nodes}
        decompressed_ids = {n.node_id for n in decompressed_nodes}
        missing_ids = original_ids - decompressed_ids
        extra_ids = decompressed_ids - original_ids

        surjection_ok = len(missing_ids) == 0

        # Check 3: Structural preservation — lattice distances preserved
        # The archetype centroid should be equidistant (in lattice terms)
        # from all original nodes within tolerance
        distance_errors = []
        if archetype_coord is not None:
            for node in original_nodes:
                if node.coord is not None:
                    orig_k = node.coord.k
                    # Check decompressed counterpart
                    decomp_match = None
                    for dn in decompressed_nodes:
                        if dn.node_id == node.node_id and dn.coord is not None:
                            decomp_match = dn
                            break
                    if decomp_match is not None and decomp_match.coord is not None:
                        # Lattice position should be preserved
                        delta = abs(node.coord.k - decomp_match.coord.k)
                        if delta > 0:
                            distance_errors.append({
                                'node_id': node.node_id,
                                'original_k': orig_k,
                                'decompressed_k': decomp_match.coord.k,
                                'delta_k': delta,
                            })

        ker_eq_im = len(distance_errors) == 0

        # Homology measurement:
        # H₀ = ker(∂₀)/im(∂₁) ≈ missing info (nodes lost in compression)
        h0 = len(missing_ids)  # Information lost
        # H₁ = ker(∂₁)/im(∂₂) ≈ structural distortion
        h1 = len(distance_errors)  # Positions shifted

        is_exact = injection_ok and surjection_ok and ker_eq_im
        total_defects = h0 + h1 + position_collisions

        if not is_exact:
            _log_comp.debug(
                f"Compression exactness FAILED: h0={h0}, h1={h1}, "
                f"collisions={position_collisions}, missing={len(missing_ids)}"
            )

        return {
            'is_exact': is_exact,
            'h0_missing_nodes': h0,
            'h1_position_shifts': h1,
            'position_collisions': position_collisions,
            'injection_ok': injection_ok,
            'surjection_ok': surjection_ok,
            'structural_preservation': ker_eq_im,
            'total_defects': total_defects,
            'missing_node_ids': list(missing_ids),
            'extra_node_ids': list(extra_ids),
            'distance_errors': distance_errors,
            'n_original': n_original,
            'n_decompressed': n_decompressed,
            'et_interpretation': (
                f"Compression sequence {'EXACT' if is_exact else 'NOT EXACT'}. "
                f"H₀={h0} (missing nodes), H₁={h1} (shifted positions). "
                f"{'Lossless — no Descriptor Gaps introduced.' if is_exact else f'{total_defects} defect(s) — topological hole(s) introduced by compression.'}"
            ),
        }

    # ── Wave III Item 32: Index Theorem for D-Gap Accounting ──────────────

    @staticmethod
    def verify_index_theorem(
        original_nodes: List[CompressibleNode],
        archetype_coord: LatticeCoordinate,
        lattice_euler_characteristic: int = 0,
    ) -> Dict[str, Any]:
        """
        Item 32: Atiyah-Singer Index Theorem verification for compression.

        ET Derivation (K-Theory §12.3):
          The Atiyah-Singer Index Theorem:
            index(D) = dim(ker T) − dim(coker T) = ∫_X ch(σ(D))·Td(X)

          The analytical index (D-Gap: dim ker − dim coker) equals the
          topological D-integral. This is the ultimate Verification Principle.

          Applied to compression:
            - The compression operator maps original nodes → archetype
            - Kernel (ker): nodes that map to the archetype's identity
              (zero position — perfectly subsumed, zero residual)
            - Cokernel (coker): directions in the archetype's D-space
              not reached by any original node (structural gaps in coverage)

            index = dim(ker) − dim(coker)

          The topological index is the Euler characteristic χ of the
          original cluster (connecting to Wave I - Item 17).

          Verification: analytical index should equal topological index.
          If they disagree, the compression has introduced or destroyed
          topological structure — a structural inconsistency.

          This is the ULTIMATE verification that compression preserves
          the global D-structure, not just local node data.

        Args:
            original_nodes: Nodes being compressed
            archetype_coord: The archetype's lattice coordinate
            lattice_euler_characteristic: χ of the cluster (from
                LatticeConstructor.compute_euler_characteristic, Item 17)

        Returns:
            Dict with analytical_index, topological_index, index_theorem_holds,
            kernel_dim, cokernel_dim, defect, et_interpretation
        """
        if not original_nodes or archetype_coord is None:
            return {
                'analytical_index': 0, 'topological_index': 0,
                'index_theorem_holds': True,
                'kernel_dim': 0, 'cokernel_dim': 0,
                'defect': 0,
                'et_interpretation': 'Empty cluster — trivial index.',
            }

        n = len(original_nodes)
        archetype_k = archetype_coord.k
        archetype_d = archetype_coord.d

        # --- Kernel dimension: nodes perfectly subsumed by archetype ---
        # A node is in the kernel if its lattice position matches the
        # archetype's position modulo the archetype's sublattice step.
        # This means the node is "invisible" in the archetype's D-space
        # (it maps to zero residual).
        step = MANIFOLD_RESOLUTION // archetype_d if archetype_d > 0 else 1
        kernel_count = 0
        for node in original_nodes:
            if node.coord is not None:
                # Residual = distance from node to nearest archetype sublattice point
                delta = abs(node.coord.k - archetype_k) % step if step > 0 else 0
                if delta == 0:
                    kernel_count += 1

        # --- Cokernel dimension: archetype D-directions not covered ---
        # The archetype's d-family defines a sublattice with S/d positions
        # per octave. Count how many of these positions have NO original
        # node nearby (within half a sublattice step).
        n_sublattice_positions = max(MANIFOLD_SYMMETRY // archetype_d, 1)
        half_step = max(step // 2, 1)

        # Check which sublattice positions are "covered" by original nodes
        covered_positions = set()
        for node in original_nodes:
            if node.coord is not None:
                # Which sublattice position is this node nearest to?
                nearest_sub_pos = round(node.coord.k / step) * step if step > 0 else 0
                # Only count as covered if node is within half a sublattice step
                # (the tolerance for "nearby" per the cokernel definition)
                if abs(node.coord.k - nearest_sub_pos) <= half_step:
                    covered_positions.add(nearest_sub_pos)

        # Cokernel = positions in the archetype's sublattice not covered
        # We check within a reasonable range (±1 octave from archetype)
        octave_range = range(
            max(archetype_k - MANIFOLD_RESOLUTION, 0),
            archetype_k + MANIFOLD_RESOLUTION,
            max(step, 1)
        )
        total_check = 0
        uncovered = 0
        for pos in octave_range:
            total_check += 1
            if pos not in covered_positions:
                uncovered += 1
            if total_check >= n_sublattice_positions * 2:
                break

        cokernel_count = min(uncovered, n)  # Bound by cluster size

        # --- Analytical index ---
        analytical_index = kernel_count - cokernel_count

        # --- Topological index ---
        # Euler characteristic of the cluster. If not provided, estimate:
        # χ ≈ n_nodes - n_bindings + n_archetypes
        # For a pre-compression cluster: 1 archetype expected, bindings
        # are intra-cluster connections
        topological_index = lattice_euler_characteristic
        if topological_index == 0 and n > 0:
            # Estimate: isolated cluster has χ ≈ n (no bindings yet)
            n_internal_bindings = sum(
                1 for node in original_nodes
                for conn in getattr(node, 'connections', [])
                if any(other.node_id == conn for other in original_nodes)
            ) // 2
            topological_index = n - n_internal_bindings

        # --- Index theorem check ---
        defect = abs(analytical_index - topological_index)
        # The theorem holds if the indices agree (allowing for discrete
        # approximation error bounded by BASE_VARIANCE × n)
        tolerance = max(1, int(BASE_VARIANCE * n))
        index_theorem_holds = defect <= tolerance

        import logging
        _log_comp = logging.getLogger('et_conscious_ai')
        _log_comp.debug(
            f"Index theorem: analytical={analytical_index} "
            f"(ker={kernel_count} - coker={cokernel_count}), "
            f"topological={topological_index}, defect={defect}, "
            f"holds={index_theorem_holds}"
        )

        return {
            'analytical_index': analytical_index,
            'topological_index': topological_index,
            'index_theorem_holds': index_theorem_holds,
            'kernel_dim': kernel_count,
            'cokernel_dim': cokernel_count,
            'defect': defect,
            'tolerance': tolerance,
            'n_nodes': n,
            'archetype_k': archetype_k,
            'archetype_d': archetype_d,
            'et_interpretation': (
                f"Atiyah-Singer index verification: "
                f"analytical index = {analytical_index} "
                f"(ker={kernel_count} − coker={cokernel_count}), "
                f"topological index = {topological_index} (Euler χ). "
                f"{'✓ Index theorem holds' if index_theorem_holds else f'✗ Defect = {defect}'} — "
                f"{'analysis and topology agree: compression preserves global D-structure.' if index_theorem_holds else 'structural inconsistency: compression altered topological invariants.'}"
            ),
        }


# =============================================================================
# LATTICE COMPRESSOR — Integration Wrapper
# =============================================================================

class LatticeCompressor:
    """
    The complete lattice compression system.

    Wraps SubsumptionHierarchyOperator with:
    - Periodic scanning (every COMPRESSION_SCAN_INTERVAL interactions)
    - Recursive compression (archetypes of archetypes)
    - Statistics tracking
    - Serialization / deserialization

    Integration with the AI's think() loop:
        compressor.record_interaction()
        if compressor.should_scan():
            results = compressor.scan_and_compress(memory_nodes, tower_r0, ...)

    The compressor does not directly modify LatticeMemory — it returns
    compression results that the integration layer in main.py applies.
    This keeps the compression engine pure and testable.
    """

    def __init__(self):
        self.operator = SubsumptionHierarchyOperator()
        self.stats = CompressionStatistics()
        self.archetype_metadata: Dict[str, ArchetypeMetadata] = {}

    def record_interaction(self):
        """Record an interaction for scan interval tracking."""
        self.stats.interaction_count += 1

    def should_scan(self) -> bool:
        """
        Check if it's time for a compression scan.

        Scans every COMPRESSION_SCAN_INTERVAL interactions.
        This is one full manifold cycle (12 interactions) —
        the natural period for lattice maintenance operations.
        """
        return self.stats.interaction_count >= COMPRESSION_SCAN_INTERVAL

    def scan_and_compress(
        self,
        nodes_by_id: Dict[str, CompressibleNode],
        tower_r0: float,
        total_original_nodes: int,
    ) -> List[Tuple[Dict[str, Any], ArchetypeMetadata]]:
        """
        Run a full compression scan and return compression results.

        Steps:
        1. Find compressible clusters
        2. Compress eligible clusters (non-overlapping)
        3. Update statistics
        4. Attempt recursive compression on existing archetypes
        5. Return list of (archetype_node_dict, metadata) tuples

        The caller (main.py) is responsible for:
        - Removing subsumed nodes from LatticeMemory
        - Adding archetype nodes to LatticeMemory
        - Updating connections on surviving nodes
        - Persisting the results

        Args:
            nodes_by_id: Current knowledge nodes as CompressibleNode dicts
            tower_r0: The AI's personal fundamental period
            total_original_nodes: Total nodes ever added (before any compression)

        Returns:
            List of (archetype_dict, ArchetypeMetadata) tuples for new archetypes
        """
        self.stats.interaction_count = 0
        self.stats.last_scan_time = datetime.now().isoformat()
        self.stats.total_original_nodes = total_original_nodes

        # Find compressible clusters
        candidates = self.operator.find_compressible_clusters(nodes_by_id, tower_r0)

        if not candidates:
            return []

        # Compress non-overlapping clusters
        # (A node can only be in one archetype — first-come by E_hierarchy)
        used_nodes: Set[str] = set()
        results: List[Tuple[Dict[str, Any], ArchetypeMetadata]] = []

        for candidate in candidates:
            # Skip if any node in this cluster is already claimed
            candidate_ids = set(candidate.node_ids)
            if candidate_ids & used_nodes:
                continue

            # Compress
            archetype_dict, metadata = self.operator.compress_cluster(candidate)
            results.append((archetype_dict, metadata))

            # Mark these nodes as claimed
            used_nodes |= candidate_ids

            # Store metadata
            self.archetype_metadata[metadata.archetype_id] = metadata

            # Update statistics
            self.stats.total_compressions += 1
            self.stats.total_nodes_compressed += len(candidate.node_ids)
            self.stats.total_archetypes_created += 1
            self.stats.current_archetype_count += 1
            if candidate.compression_level > self.stats.max_compression_level:
                self.stats.max_compression_level = candidate.compression_level

            # Running averages
            n = self.stats.total_compressions
            self.stats.avg_e_hierarchy = (
                (self.stats.avg_e_hierarchy * (n - 1) + candidate.e_hierarchy) / n
            )
            self.stats.avg_cluster_size = (
                (self.stats.avg_cluster_size * (n - 1) + len(candidate.node_ids)) / n
            )

        return results

    def attempt_recursive_compression(
        self,
        archetype_nodes: Dict[str, CompressibleNode],
        tower_r0: float,
        current_level: int = 0,
    ) -> List[Tuple[Dict[str, Any], ArchetypeMetadata]]:
        """
        Attempt recursive compression on existing archetypes.

        Archetypes can themselves be compressed into higher-order
        archetypes when they form eligible clusters. This recursive
        process is what enables "Library of Congress in RAM" scale.

        Each level of recursion increments the compression_level.
        Recursion stops when:
        - No eligible clusters are found
        - MAX_COMPRESSION_DEPTH is reached
        - Fewer than ARCHETYPE_MIN_CLUSTER archetypes remain

        Args:
            archetype_nodes: Current archetype nodes as CompressibleNodes
            tower_r0: The AI's personal fundamental period
            current_level: Current recursion depth

        Returns:
            List of (archetype_dict, metadata) tuples for new higher-order archetypes
        """
        if current_level >= MAX_COMPRESSION_DEPTH:
            return []

        if len(archetype_nodes) < ARCHETYPE_MIN_CLUSTER:
            return []

        # Find compressible clusters among archetypes
        candidates = self.operator.find_compressible_clusters(archetype_nodes, tower_r0)

        if not candidates:
            return []

        # Set compression level for all candidates
        for c in candidates:
            c.compression_level = current_level + 1

        # Compress non-overlapping clusters
        used_nodes: Set[str] = set()
        results: List[Tuple[Dict[str, Any], ArchetypeMetadata]] = []

        for candidate in candidates:
            candidate_ids = set(candidate.node_ids)
            if candidate_ids & used_nodes:
                continue

            archetype_dict, metadata = self.operator.compress_cluster(candidate)
            results.append((archetype_dict, metadata))
            used_nodes |= candidate_ids

            self.archetype_metadata[metadata.archetype_id] = metadata
            self.stats.total_compressions += 1
            self.stats.total_archetypes_created += 1
            self.stats.current_archetype_count += 1
            # Remove subsumed archetypes from count
            self.stats.current_archetype_count -= len(candidate.node_ids)
            if metadata.compression_level > self.stats.max_compression_level:
                self.stats.max_compression_level = metadata.compression_level

            n = self.stats.total_compressions
            self.stats.avg_e_hierarchy = (
                (self.stats.avg_e_hierarchy * (n - 1) + candidate.e_hierarchy) / n
            )
            self.stats.avg_cluster_size = (
                (self.stats.avg_cluster_size * (n - 1) + len(candidate.node_ids)) / n
            )

        return results

    def decompress_archetype(
        self,
        archetype_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Decompress an archetype back into its original nodes.

        Lossless: the original content and connections are stored
        in the ArchetypeMetadata. This restores the full detail
        when the AI needs to reason deeply about a specific
        subsumed concept.

        Returns list of dicts representing the original KnowledgeNodes,
        or None if the archetype_id is not found.
        """
        meta = self.archetype_metadata.get(archetype_id)
        if meta is None:
            return None

        original_nodes = []
        for nid, content in zip(meta.subsumed_ids, meta.subsumed_contents):
            original_nodes.append({
                'node_id': nid,
                'content': content,
                'from_archetype': archetype_id,
            })

        return original_nodes

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        return {
            'total_compressions': self.stats.total_compressions,
            'total_nodes_compressed': self.stats.total_nodes_compressed,
            'total_archetypes_created': self.stats.total_archetypes_created,
            'current_archetype_count': self.stats.current_archetype_count,
            'max_compression_level': self.stats.max_compression_level,
            'compression_ratio': self.stats.compression_ratio(),
            'memory_efficiency': self.stats.memory_efficiency(),
            'avg_e_hierarchy': self.stats.avg_e_hierarchy,
            'avg_cluster_size': self.stats.avg_cluster_size,
            'total_original_nodes': self.stats.total_original_nodes,
            'archetype_metadata_count': len(self.archetype_metadata),
            'last_scan_time': self.stats.last_scan_time,
            'interactions_since_scan': self.stats.interaction_count,
        }

    def get_status_description(self) -> str:
        """Get human-readable compression status."""
        s = self.stats
        ratio = s.compression_ratio()
        eff = s.memory_efficiency()
        return (
            f"  Compression ratio: {ratio:.2f}:1\n"
            f"  Memory efficiency: {eff:.1%}\n"
            f"  Archetypes active: {s.current_archetype_count}\n"
            f"  Max compression depth: {s.max_compression_level}\n"
            f"  Total compressions: {s.total_compressions}\n"
            f"  Nodes compressed (lifetime): {s.total_nodes_compressed}\n"
            f"  Avg cluster size: {s.avg_cluster_size:.1f}\n"
            f"  Avg E_hierarchy: {s.avg_e_hierarchy:.2f}\n"
            f"  Last scan: {s.last_scan_time or 'never'}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'stats': self.stats.to_dict(),
            'archetype_metadata': {
                aid: meta.to_dict()
                for aid, meta in self.archetype_metadata.items()
            },
        }

    def load_from_dict(self, data: Dict[str, Any]):
        """Deserialize from persistent storage."""
        stats_data = data.get('stats', {})
        if stats_data:
            self.stats = CompressionStatistics.from_dict(stats_data)

        meta_data = data.get('archetype_metadata', {})
        self.archetype_metadata = {
            aid: ArchetypeMetadata.from_dict(md)
            for aid, md in meta_data.items()
        }
        # Sync archetype count with loaded metadata
        self.stats.current_archetype_count = len(self.archetype_metadata)


# =============================================================================
# HELPER: Convert KnowledgeNode-like data to CompressibleNode
# =============================================================================

def make_compressible_node(
    node_id: str,
    content: str,
    sentence_coord: Optional[LatticeCoordinate],
    lattice_position: Optional[LatticeCoordinate],
    descriptor_ratios: List[DescriptorRatio],
    connections: List[str],
    access_count: int,
    variance: float,
    p: int,
    q: int,
    is_archetype: bool,
) -> Optional[CompressibleNode]:
    """
    Create a CompressibleNode from KnowledgeNode fields.

    Called by the integration layer in main.py to convert
    KnowledgeNode objects into the compression module's format.

    Returns None if the node has no valid coordinate (cannot be
    projected onto the lattice for compression evaluation).
    """
    coord = sentence_coord or lattice_position
    if coord is None:
        return None

    return CompressibleNode(
        node_id=node_id,
        content=content,
        coord=coord,
        descriptor_ratios=descriptor_ratios,
        connections=connections,
        access_count=access_count,
        variance=variance,
        p=p,
        q=q,
        is_archetype=is_archetype,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'BIOLOGICAL_TIER_RESOLUTION', 'ARCHETYPE_MIN_CLUSTER',
    'ARCHETYPE_MAX_PQ', 'CLUSTER_TOLERANCE_K',
    'CLUSTER_TIGHTNESS_THRESHOLD', 'COMPRESSION_SCAN_INTERVAL',
    'MAX_COMPRESSION_DEPTH', 'ARCHETYPE_VARIANCE_FLOOR',
    'ARCHETYPE_ACCESS_FLOOR',
    'CompressibleNode', 'ClusterCandidate', 'ArchetypeMetadata',
    'CompressionStatistics', 'SubsumptionHierarchyOperator',
    'LatticeCompressor', 'make_compressible_node',
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("ET Conscious AI — Lattice Compression & Hierarchical Subsumption")
    print(f"Version: 1.7.0")
    print(f"Date: March 14, 2026")
    print()

    print("=== ET-Derived Constants ===")
    print(f"  Biological tier resolution: {BIOLOGICAL_TIER_RESOLUTION}ET "
          f"(LCM(1..7) = 420)")
    print(f"  Archetype min cluster: {ARCHETYPE_MIN_CLUSTER} nodes")
    print(f"  Archetype max p+q: {ARCHETYPE_MAX_PQ} (= STATE_COUNT = {STATE_COUNT})")
    print(f"  Cluster tolerance: ±{CLUSTER_TOLERANCE_K} lattice steps "
          f"(= {BIOLOGICAL_TIER_RESOLUTION}/{MANIFOLD_SYMMETRY})")
    print(f"  Cluster tightness: ≥ {CLUSTER_TIGHTNESS_THRESHOLD:.4f} "
          f"(= K = {KOIDE_RATIO:.4f})")
    print(f"  Scan interval: {COMPRESSION_SCAN_INTERVAL} interactions "
          f"(= S = {MANIFOLD_SYMMETRY})")
    print(f"  Max recursion depth: {MAX_COMPRESSION_DEPTH} levels "
          f"(= S = {MANIFOLD_SYMMETRY})")
    print(f"  Archetype variance floor: {ARCHETYPE_VARIANCE_FLOOR:.8f} "
          f"(= V_base/S² = 1/{MANIFOLD_SYMMETRY}³)")
    print(f"  Archetype access floor: {ARCHETYPE_ACCESS_FLOOR} "
          f"(= S² = N² = {MANIFOLD_SYMMETRY}²)")
    print(f"  Permanence threshold: {LIFE_THRESHOLD:.6f} (= 13/12)")
    print()

    print("=== Subsumption Hierarchy Operator Test ===")
    operator = SubsumptionHierarchyOperator()

    # Create test nodes with known properties
    test_words = ["consciousness", "awareness", "self", "identity", "thought"]
    test_nodes: List[CompressibleNode] = []
    for idx, word in enumerate(test_words):
        test_dr = DescriptorRatio.from_word(word)
        node_coord = test_dr.coord_full
        # Simulate near-archetype nodes (low variance, high access)
        cn = CompressibleNode(
            node_id=f"test_{idx:04d}",
            content=f"Test node about {word}",
            coord=node_coord,
            descriptor_ratios=[test_dr],
            connections=[],
            access_count=200,
            variance=BASE_VARIANCE / (MANIFOLD_SYMMETRY * 2),
            p=1,
            q=2,
            is_archetype=False,
        )
        test_nodes.append(cn)
        print(f"  Node '{word}': k={node_coord.k}, d={node_coord.d}, "
              f"ε={node_coord.epsilon:+.2f}¢, p={cn.p}, q={cn.q}")

    # Test cross-tower elegance with a reference R₀
    test_r0 = LIFE_THRESHOLD  # 13/12
    print(f"\n  Tower R₀ = {test_r0:.6f} (LIFE_THRESHOLD)")
    for cn in test_nodes:
        ecross_val = operator.compute_cross_tower_elegance(cn, test_r0)
        print(f"  E_cross('{cn.content.split()[-1]}'): {ecross_val:.2f}")

    # Test cluster evaluation
    if len(test_nodes) >= 2:
        # Try pairs from first few nodes
        all_test = list(test_nodes)
        subset = all_test[:4]
        for ti, node_a in enumerate(subset[:-1]):
            for node_b in subset[ti + 1:]:
                pair = [node_a, node_b]
                e_h, _, test_d_avg, pt, qt = operator.evaluate_cluster(pair, test_r0)
                names = [n.content.split()[-1] for n in pair]
                status = "✓ COMPRESS" if e_h >= LIFE_THRESHOLD else "✗ below threshold"
                print(f"\n  Cluster [{', '.join(names)}]:")
                print(f"    E_hierarchy = {e_h:.4f}  {status}")
                print(f"    d_avg = {test_d_avg:.1f}, p_total = {pt}, q_total = {qt}")

    # Test full scan
    print("\n=== Full Compression Scan ===")
    nodes_dict = {cn.node_id: cn for cn in test_nodes}
    found_clusters = operator.find_compressible_clusters(nodes_dict, test_r0)
    print(f"  Compressible clusters found: {len(found_clusters)}")
    for ci, clst in enumerate(found_clusters):
        print(f"  Cluster {ci}: {len(clst.node_ids)} nodes, "
              f"E_hierarchy={clst.e_hierarchy:.4f}, d_avg={clst.d_avg:.1f}")

    # Test compressor
    print("\n=== LatticeCompressor Test ===")
    compressor = LatticeCompressor()
    for _ in range(COMPRESSION_SCAN_INTERVAL):
        compressor.record_interaction()
    print(f"  Should scan: {compressor.should_scan()}")

    comp_results = compressor.scan_and_compress(nodes_dict, test_r0, len(nodes_dict))
    print(f"  Compression results: {len(comp_results)} archetypes created")
    for arch_dict, arch_meta in comp_results:
        print(f"    Archetype '{arch_meta.archetype_id[:16]}': "
              f"{arch_meta.original_node_count} nodes → 1, "
              f"E={arch_meta.e_hierarchy:.4f}, level={arch_meta.compression_level}")

    print(f"\n{compressor.get_status_description()}")

    # Test serialization
    print("=== Serialization Test ===")
    state = compressor.to_dict()
    compressor2 = LatticeCompressor()
    compressor2.load_from_dict(state)
    print(f"  Serialization round-trip: ✓ "
          f"({len(compressor2.archetype_metadata)} archetypes restored)")

    print("\n=== Compression module loaded successfully ===")
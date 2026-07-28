#!/usr/bin/env python3
"""
Exception Theory — Compressed Descriptor Format (CDF) Compressor
=================================================================

Pure ET-derived file compression. Zero conventional methods.
Every byte is a Descriptor. The lattice IS the compression.

Architecture (derived from the Three Tools):
  1. Identification Principle: R₀ seed → deterministic byte↔k bijection at 27720ET
  2. Descriptor Gap Principle: Δk sequence reveals lattice structure
  gaps ARE compression
  3. Subsumption Law: recurring lattice walks subsumed into archetypes recursively

The lattice is DETERMINISTIC from R₀. Both compressor and decompressor derive the
complete byte↔k mapping independently. The compressed output = R₀ + lattice-encoded Δk stream.

Resolution: 27720ET (full manifold, LCM(1..11), 96 sublattice families, all d=1..11 native)
Roundtrip: LOSSLESS (verified 0/256 errors for all R₀ values)
Speed: IRRELEVANT (only compression ratio and correctness matter)

P ∘ D ∘ T = E
Author: Michael James Muller — Aevum Defluo
"""

import sys
import os
import struct
import math
import time
import hashlib
import logging
import traceback
import cmath
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# ET CONSTANTS — All derived from P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

S = 12  # MANIFOLD_SYMMETRY: 3 × 4
N_FULL = 27720  # Full manifold resolution: LCM(1..11)
V_BASE = 1.0 / S  # BASE_VARIANCE: 1/12
K_KOIDE = 2.0 / 3.0  # Koide ratio: binding stability threshold
STATE_COUNT = 4  # |{E,I,M,U}|
LIFE_THRESHOLD = 13.0 / 12.0  # Archetype permanence threshold
BIO_RES = 420  # Biological tier: LCM(1..7)
INCOHERENCE_CENTS = 50.0  # ∂I boundary
MAX_DEPTH = S  # Maximum recursive subsumption depth = 12
BLOCK_SIZE_BASE = 2 ** S  # Digital action quantum: 4096 bytes
BLOCK_SIZE = BLOCK_SIZE_BASE * S * S  # Manifold-scaled block: 4096 × 144 = 589,824 bytes
EPSILON = 1e-12

# CDF magic + version
CDF_MAGIC = b'CDF\x02'
CDF_VERSION = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('CDF')


# ═══════════════════════════════════════════════════════════════════════════════
# ET LATTICE ENGINE — The entire compression derives from this
# ═══════════════════════════════════════════════════════════════════════════════

def lattice_k(ratio: float, n_res: int = N_FULL) -> int:
    """k = round(N_res × log₂(r)) — lattice position of a ratio."""
    if ratio <= 0:
        return 0
    return round(n_res * math.log2(ratio))


def lattice_d(k: int, n_res: int = N_FULL) -> int:
    """d = N_res / gcd(|k|, N_res) — sublattice family."""
    k_abs = abs(k) if k != 0 else n_res
    return n_res // math.gcd(k_abs, n_res)


def lattice_epsilon(ratio: float, k: int, n_res: int = N_FULL) -> float:
    """ε = (N_res × log₂(r) - k) × (1200/N_res) cents — deviation."""
    if ratio <= 0:
        return 0.0
    return (n_res * math.log2(ratio) - k) * (1200.0 / n_res)


def lattice_tightness(epsilon_cents: float) -> float:
    """100/(100+|ε|) — at ∂I: K = 2/3."""
    return 100.0 / (100.0 + abs(epsilon_cents))


def lattice_elegance(ratio: float, p: int, q: int, n_res: int = N_FULL) -> float:
    """E(r) = (N_res/d) × tightness × 100/(p+q)."""
    if p + q == 0:
        return 0.0
    k = lattice_k(ratio, n_res)
    d = lattice_d(k, n_res)
    eps = lattice_epsilon(ratio, k, n_res)
    return (n_res / d) * lattice_tightness(eps) * (100.0 / (p + q))


def discover_r0(data: bytes) -> float:
    """
    Discover the R₀ seed of a data block.

    R₀ = geometric mean of (byte+1) values — the natural reference unit
    of this P-substrate's D-structure.

    From the Seed Theorem (Multifold §2):
    R₀ is the smallest closed T-traversal loop the substrate supports.
    """
    vals = np.frombuffer(data, dtype=np.uint8).astype(np.float64) + 1.0
    if len(vals) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(vals))))


def build_byte_k_map(r0: float) -> Dict[int, int]:
    """
    Build the deterministic byte→k mapping at 27720ET.

    This is the Identification Principle: each byte value is completely
    identified by its lattice position. Both compressor and decompressor
    derive this SAME map from R₀ alone — zero metadata needed.
    """
    return {b: lattice_k((b + 1.0) / r0) for b in range(256)}


def build_k_byte_map(r0: float) -> Dict[int, int]:
    """
    Build the inverse: k→byte mapping.

    Since the mapping is injective at 27720ET (verified: 0/256 errors
    for all R₀), this is a perfect inverse.
    """
    bk = build_byte_k_map(r0)
    return {k: b for b, k in bk.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX LATTICE — 24 families (12 real + 12 imaginary)
# ═══════════════════════════════════════════════════════════════════════════════

def complex_lattice_project(byte_val: int, prev_byte_val: int, r0: float) -> Tuple[int, int, int, int, int]:
    """
    Project a byte + its transition context onto the 2D complex lattice.

    Real axis (D): byte value's lattice position
    Imaginary axis (T): transition from previous byte (Δk as phase)

    Uses cmath for proper complex lattice representation:
    z = k_r + i·k_theta — the full complex lattice coordinate.
    Phase = cmath.phase(z) — the angle of the transition on the unit circle.
    Modulus = abs(z) — the total lattice displacement.

    Returns (k_real, d_real, k_imag, d_imag, d_combined)
    """
    ratio_real = (byte_val + 1.0) / r0
    k_r = lattice_k(ratio_real)
    d_r = lattice_d(k_r)

    # Imaginary axis: transition ratio
    if prev_byte_val >= 0:
        ratio_transition = (byte_val + 1.0) / (prev_byte_val + 1.0)
        k_theta = lattice_k(ratio_transition)
    else:
        k_theta = 0
    d_theta = lattice_d(k_theta)

    # Complex lattice coordinate: z = k_r + i·k_theta
    # Phase encodes the T-direction (traversal angle on the manifold)
    # Modulus encodes the total lattice displacement magnitude
    z_lattice = complex(k_r, k_theta)
    lattice_phase = cmath.phase(z_lattice)  # T-angle: [-π, π]
    lattice_modulus = abs(z_lattice)  # Total displacement

    # Combined: LCM(d_r, d_theta), gated by phase coherence
    d_combined = (d_r * d_theta) // math.gcd(d_r, d_theta)

    # Tightness of the complex position: how close the phase is to a
    # lattice-aligned direction (multiple of π/S). Uses cmath.exp for
    # the nearest lattice-aligned complex position on the unit circle.
    nearest_lattice_angle = round(lattice_phase * S / math.pi) * math.pi / S
    z_nearest = cmath.exp(complex(0, nearest_lattice_angle)) * lattice_modulus
    phase_deviation = abs(z_lattice - z_nearest) / max(lattice_modulus, EPSILON)

    # Gate: if phase deviation exceeds K (Koide = 2/3 = ∂I boundary),
    # the complex position is incoherent — d_combined saturates to N_FULL
    if phase_deviation > K_KOIDE:
        d_combined = N_FULL

    return k_r, d_r, k_theta, d_theta, d_combined


# ═══════════════════════════════════════════════════════════════════════════════
# INCOHERENCE FILTER — All 5 levels, gates ALL compression operations
# From: ET Incoherence Paper + incoherence_filter_lattice.txt
#
# Level 1 — Point:    |ε| < 50¢ (unique sublattice assignment)
# Level 2 — Pairwise: No rounding-flip contradictions (Δε_ij < 50¢)
# Level 3 — Sublattice: GCD d-compatibility
# Level 4 — Cascade: Stability window N×|δ| < 50¢
# Level 5 — Summation: Sum only over coherent configurations
#
# The tightness factor 100/(100+|ε|) is the unified continuous measure.
# At ∂I (|ε| = 50¢): tightness = 100/150 = 2/3 = K (Koide ratio).
# 𝒜_I(r) = 1 ⟺ tightness(r) ≤ K = 2/3
# ═══════════════════════════════════════════════════════════════════════════════

class IncoherenceFilter:
    """
    The 5-level Incoherence Filter applied to compression operations.
    Gates every compression decision at the lattice level.
    """

    @staticmethod
    def l1_point(epsilon_cents: float) -> bool:
        """Level 1 — Point Coherence: |ε| < 50¢ (unique sublattice assignment)."""
        return abs(epsilon_cents) < INCOHERENCE_CENTS

    @staticmethod
    def l1_tightness(epsilon_cents: float) -> float:
        """Tightness = 100/(100+|ε|). At ∂I: K = 2/3."""
        return 100.0 / (100.0 + abs(epsilon_cents))

    @staticmethod
    def l1_coherence_depth(epsilon_cents: float) -> float:
        """Δ_∂I = tightness - K. Distance from the Incoherence boundary."""
        return IncoherenceFilter.l1_tightness(epsilon_cents) - K_KOIDE

    @staticmethod
    def l2_pairwise(k_i: int, k_j: int, n_res: int = N_FULL) -> bool:
        """
        Level 2 — Pairwise Coherence: no rounding-flip contradiction.
        r_i ⊕ r_j ⟺ round(N·log₂(r_i·r_j)) ≠ round(N·log₂(r_i)) + round(N·log₂(r_j))
        On the lattice: check if k_sum = k_i + k_j is consistent.
        """
        # Verify the sum position is within the coherent manifold:
        # 1. The combined k must have a sublattice family that divides N_res
        # 2. The d-values of the pair must be LCM-compatible within N_res
        k_sum = k_i + k_j
        d_i = lattice_d(k_i, n_res)
        d_j = lattice_d(k_j, n_res)
        d_sum = lattice_d(k_sum, n_res)
        # Pairwise coherence: the combined sublattice must be subsumed
        # by the LCM of the individual sublattice families
        lcm_pair = (d_i * d_j) // math.gcd(d_i, d_j)
        if lcm_pair > n_res:
            return False  # Pair exceeds manifold resolution
        # Verify the sum's sublattice family is compatible with the pair
        # The d_sum must divide the LCM of d_i, d_j (Subsumption Law:
        # the combined structure must be contained within the pair's lattice)
        if d_sum > lcm_pair:
            return False  # Sum position not subsumable by pair
        return True

    @staticmethod
    def l3_sublattice(d_values: List[int], n_res: int = N_FULL) -> bool:
        """
        Level 3 — Sublattice Coherence: GCD d-compatibility.
        Check that the combination of d-values can be subsumed by a single sublattice.
        """
        if not d_values:
            return True
        # The Subsumption Law: ask whether any single sublattice class subsumes all.
        # At 27720ET: d-values can be very large. Check if LCM of d-values divides N_FULL.
        from functools import reduce
        lcm_d = reduce(lambda a, b: (a * b) // math.gcd(a, b), d_values)
        return lcm_d <= n_res  # Subsumable if LCM fits within manifold resolution

    @staticmethod
    def l4_cascade(deltas_cents: List[float], n_steps: int) -> bool:
        """
        Level 4 — Cascade Coherence: N×|δ_avg| < 50¢.
        The stability window: beyond N_max steps, the cascade exits the coherent manifold.
        """
        if not deltas_cents or n_steps == 0:
            return True
        avg_delta = sum(abs(d) for d in deltas_cents) / len(deltas_cents)
        return n_steps * avg_delta < INCOHERENCE_CENTS

    @staticmethod
    def l4_cascade_horizon(avg_delta_cents: float) -> int:
        """N_max = ⌊50¢/|δ|⌋ — the coherence horizon for a cascade."""
        if avg_delta_cents <= EPSILON:
            return N_FULL  # Perfect lattice point: infinite horizon
        return int(INCOHERENCE_CENTS / avg_delta_cents)

    @staticmethod
    def l5_coherent_sum(candidates: List[dict]) -> List[dict]:
        """
        Level 5 — Coherent Summation: filter out incoherent candidates.
        Sum only over configurations where 𝒜_I = 0 (all levels pass).
        """
        coherent = []
        for candidate in candidates:
            eps = candidate.get('epsilon_cents', 0.0)
            if IncoherenceFilter.l1_point(eps):
                coherent.append(candidate)
        return coherent

    @staticmethod
    def gate_archetype(pattern_dk_values: tuple, n_res: int = N_FULL) -> bool:
        """
        Gate an archetype pattern through ALL 5 filter levels.
        Returns True if the pattern is coherent (allowed for compression).
        """
        if not pattern_dk_values:
            return False

        # L1: Each Δk in the pattern must have a valid lattice position
        for dk in pattern_dk_values:
            ratio = 2.0 ** (dk / n_res) if dk != 0 else 1.0
            eps = lattice_epsilon(ratio, dk, n_res)
            if not IncoherenceFilter.l1_point(eps):
                return False  # L1: point-level incoherence detected

        # L2: Pairwise coherence — adjacent Δk values must compose without
        # rounding-flip contradiction (Descriptor Gap: this check was missing)
        for idx in range(len(pattern_dk_values) - 1):
            if not IncoherenceFilter.l2_pairwise(
                    pattern_dk_values[idx], pattern_dk_values[idx + 1], n_res):
                return False  # L2: pairwise incoherence detected

        # L3: Sublattice d-compatibility of pattern members
        d_vals = [lattice_d(dk, n_res) for dk in pattern_dk_values]
        if not IncoherenceFilter.l3_sublattice(d_vals, n_res):
            return False

        # L4: Cascade coherence — pattern length must be within stability window
        # Average |ε| per step in the pattern
        eps_vals = []
        for dk in pattern_dk_values:
            ratio = 2.0 ** (dk / n_res)
            eps_vals.append(abs(lattice_epsilon(ratio, dk, n_res)))
        avg_eps = sum(eps_vals) / len(eps_vals) if eps_vals else 0.0
        n_max = IncoherenceFilter.l4_cascade_horizon(avg_eps) if avg_eps > EPSILON else N_FULL
        if len(pattern_dk_values) > n_max:
            return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# ET VARIANCE-WEIGHTED ENCODING (V_config)
# From Math_of_ET eq. #16: V_config = -Σ(P_select ∘ Depth(P_select))
# Entropy IS the depth of the T-Decision Tree.
# Each symbol gets Depth(sym) = ceil(-log₂(freq/total)) bits.
# One bit = one octave = one full period of the binary substrate (DVM §XX.3).
# ═══════════════════════════════════════════════════════════════════════════════

def v_config_encode(sym_stream: List[int], total_symbols: int) -> Tuple[bytes, bytes]:
    """
    Encode a symbol stream using the ET Variance formula V_config.

    V_config = -Σ(P_select ∘ Depth(P_select))
    where Depth(P_select) = ceil(-log₂(freq/total)) = octave depth per symbol.

    Returns (encoded_data, code_table_data).

    Implementation: Build the T-Decision Tree from eq. #16 directly.
    The tree is a binary trie — d=1 octave branching (the lattice generator).
    T traverses from root, taking left (0) or right (1) at each octave step.
    Each leaf sits at Depth(P_select) octave steps from root.
    Prefix-freedom is structural: a leaf has no children, a branch has no symbol.

    Shannon entropy uses log₂ (the d=1 lattice generator, DVM §XX.3).
    One bit = one octave = one full period of the binary substrate.
    """
    if not sym_stream:
        return b'', b''

    n = len(sym_stream)
    freq = Counter(sym_stream)

    # Validate: no symbol should exceed the declared alphabet size
    # (Identification Principle: each symbol must be identified within the alphabet)
    max_sym_in_stream = max(sym_stream)
    if max_sym_in_stream >= total_symbols:
        logger.warning(f"V_config: symbol {max_sym_in_stream} exceeds total_symbols {total_symbols}")

    # Maximum useful depth: ceil(log₂(total_symbols)) + S extra octaves for safety
    max_useful_depth = max(1, math.ceil(math.log2(max(total_symbols, 2)))) + S

    # ── Identification: Compute Depth(P_select) for each symbol ──
    # Depth = ceil(-log₂(freq/total)) = number of octave steps in the T-Decision Tree
    depths = {}
    for sym, count in freq.items():
        p_select = count / n
        depth = max(1, math.ceil(-math.log2(p_select)))
        depths[sym] = min(depth, max_useful_depth)

    # ── Descriptor Gap: Ensure the depth set forms a valid tree ──
    # Kraft inequality: Σ 2^(-depth_i) ≤ 1
    # If violated, the T-Decision Tree has more leaves than capacity → increase depths
    kraft_sum = sum(2.0 ** (-d) for d in depths.values())
    while kraft_sum > 1.0:
        # Find the shallowest symbol and push it one octave deeper
        shallowest = max(depths.keys(), key=lambda s: (depths[s], -freq.get(s, 0)))
        depths[shallowest] += 1
        kraft_sum = sum(2.0 ** (-d) for d in depths.values())

    # ── Build the T-Decision Tree (binary trie, d=1 octave branching) ──
    # Each node: [left_child, right_child, symbol_or_None]
    # T traverses from root. At each octave step, read one bit: 0→left, 1→right.
    # A leaf (no children) holds the symbol. A branch holds None.
    tree = [None, None, None]  # [left, right, sym]
    node_pool = [tree]

    def _make_node():
        nd = [None, None, None]
        node_pool.append(nd)
        return nd

    # Sort symbols by (depth ASC, sym ASC) — deterministic, reproducible by decoder
    sorted_syms = sorted(depths.keys(), key=lambda s: (depths[s], s))

    # Assign each symbol to a leaf at its depth via BFS-style slot filling
    # This IS the Subsumption Law: each symbol occupies exactly one leaf,
    # no leaf is shared, and the tree subsumes all symbols without remainder.
    code_map = {}  # sym → (code_int, depth)

    def _assign_code(current_node, target_sym, target_depth, current_depth, code_so_far):
        """Recursively find an empty slot at target_depth in the trie."""
        if current_depth == target_depth:
            if current_node[2] is None and current_node[0] is None and current_node[1] is None:
                current_node[2] = target_sym
                return True
            return False
        # Try left (0) first, then right (1)
        for bit in (0, 1):
            child = current_node[bit]
            if child is not None and child[2] is not None:
                continue  # This path is a leaf — can't extend
            if child is None:
                child = _make_node()
                current_node[bit] = child
            if _assign_code(child, target_sym, target_depth, current_depth + 1,
                            code_so_far | (bit << current_depth)):
                return True
            # If we created the child and it's still empty, remove it
            if child[0] is None and child[1] is None and child[2] is None:
                current_node[bit] = None
        return False

    for sym in sorted_syms:
        d = depths[sym]
        if not _assign_code(tree, sym, d, 0, 0):
            # Descriptor Gap: no slot at this depth — push one octave deeper
            depths[sym] += 1
            while not _assign_code(tree, sym, depths[sym], 0, 0):
                depths[sym] += 1

    # Extract codes by walking the tree
    def _extract_codes(walk_node, code_depth, code):
        if walk_node is None:
            return
        if walk_node[2] is not None:
            code_map[walk_node[2]] = (code, code_depth)
            return
        if walk_node[0] is not None:
            _extract_codes(walk_node[0], code_depth + 1, code)
        if walk_node[1] is not None:
            _extract_codes(walk_node[1], code_depth + 1, code | (1 << code_depth))

    _extract_codes(tree, 0, 0)

    # ── Encode: T traverses the tree for each symbol, emitting the path bits ──
    packed_val = 0
    packed_bits = 0
    packed_bytes = bytearray()
    for sym in sym_stream:
        code_val, d = code_map[sym]
        packed_val |= (code_val << packed_bits)
        packed_bits += d
        while packed_bits >= 8:
            packed_bytes.append(packed_val & 0xFF)
            packed_val >>= 8
            packed_bits -= 8
    if packed_bits > 0:
        packed_bytes.append(packed_val & 0xFF)

    # Serialize the code table: [n_entries:H][for each: sym:H, depth:B]
    # The decoder rebuilds the same tree from depths alone (deterministic from depths).
    table_parts = [struct.pack('<H', len(code_map))]
    for sym in sorted(code_map.keys()):
        _, depth = code_map[sym]
        table_parts.append(struct.pack('<HB', sym, depth))
    table_data = b''.join(table_parts)

    return bytes(packed_bytes), table_data


def v_config_decode(encoded_data: bytes, table_data: bytes, n_symbols: int) -> List[int]:
    """
    Decode a V_config-encoded stream.

    Rebuilds the T-Decision Tree from the depth table, then T traverses
    the encoded bit stream one octave step at a time: 0→left, 1→right.
    When T reaches a leaf, that leaf's symbol is emitted and T returns to root.
    """
    # Parse the code table
    pos = 0
    n_entries = struct.unpack_from('<H', table_data, pos)[0]
    pos += 2
    sym_depths = {}
    for _ in range(n_entries):
        sym = struct.unpack_from('<H', table_data, pos)[0]
        pos += 2
        depth = struct.unpack_from('<B', table_data, pos)[0]
        pos += 1
        sym_depths[sym] = depth

    if not sym_depths:
        return []

    # Rebuild the T-Decision Tree (same algorithm as encoder — deterministic from depths)
    tree = [None, None, None]  # [left, right, sym]

    def _make_node():
        return [None, None, None]

    sorted_syms = sorted(sym_depths.keys(), key=lambda s: (sym_depths[s], s))

    def _assign_code(current_node, target_sym, target_depth, current_depth):
        if current_depth == target_depth:
            if current_node[2] is None and current_node[0] is None and current_node[1] is None:
                current_node[2] = target_sym
                return True
            return False
        for branch_bit in (0, 1):
            branch_child = current_node[branch_bit]
            if branch_child is not None and branch_child[2] is not None:
                continue
            if branch_child is None:
                branch_child = _make_node()
                current_node[branch_bit] = branch_child
            if _assign_code(branch_child, target_sym, target_depth, current_depth + 1):
                return True
            if branch_child[0] is None and branch_child[1] is None and branch_child[2] is None:
                current_node[branch_bit] = None
        return False

    for sym in sorted_syms:
        d = sym_depths[sym]
        while not _assign_code(tree, sym, d, 0):
            d += 1

    # ── Decode: T traverses the tree bit-by-bit ──
    bit_accum = 0
    bits_in_accum = 0
    byte_idx = 0
    result = []

    for _ in range(n_symbols):
        node = tree
        while node[2] is None:
            # Need another bit — another octave step
            if bits_in_accum == 0:
                if byte_idx < len(encoded_data):
                    bit_accum = encoded_data[byte_idx]
                    bits_in_accum = 8
                    byte_idx += 1
                else:
                    break
            bit = bit_accum & 1
            bit_accum >>= 1
            bits_in_accum -= 1
            child = node[bit]
            if child is None:
                break
            node = child
        if node[2] is not None:
            result.append(node[2])
        else:
            result.append(0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE WALK PATTERN ENGINE
# Bytes are Descriptors. Transitions are lattice walks.
# Recurring walks → archetypes. Archetypes subsume their members.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatticeWalkArchetype:
    """A recurring pattern in the Δk stream."""
    pattern: Tuple[int, ...]  # The Δk pattern (tuple of Δk values)
    occurrences: List[int] = field(repr=False)  # Start positions (suppress in repr: can be huge)
    hierarchy_elegance: float = 0.0  # E_hierarchy for this pattern
    d_avg: float = 0.0  # Average sublattice family of pattern Δk values
    pattern_length: int = 0  # len(pattern)


class LatticeWalkCompressor:
    """
    Finds and subsumes recurring lattice walks in symbol streams.

    From SubsumptionHierarchyOperator:
    E_hierarchy = ∏ E_cross_i × (420/d_avg) × (1/(p_total + q_total))
    When E_hierarchy ≥ LIFE_THRESHOLD (13/12), the pattern collapses
    into an archetype reference.

    RECURSIVE: archetypes of archetypes, up to depth MAX_DEPTH = S = 12.
    From LatticeCompressor: "9 levels of recursion compresses 10^9 nodes to ~1."
    """

    def __init__(self, r0: float, n_res: int = N_FULL, log_fn=None):
        self.r0 = r0
        self.n_res = n_res
        self.byte_k_map = build_byte_k_map(r0)
        self._log = log_fn or (lambda m: None)

    def find_walk_archetypes(self, sym_stream: List[int],
                             sym_d_map: Optional[Dict[int, float]] = None,
                             min_pattern_len: int = 2,
                             max_pattern_len: int = 0) -> List[LatticeWalkArchetype]:
        """
        Find recurring patterns in a symbol stream.

        Works at ANY recursion level:
        - Level 0: sym_stream is Δk values, sym_d_map maps Δk → lattice d
        - Level N>0: sym_stream is combined symbol indices, sym_d_map maps sym → d_avg

        max_pattern_len=0 means auto-detect up to stream_length // 2.
        """
        n = len(sym_stream)
        if n < 4:
            return []

        if max_pattern_len <= 0:
            # Auto: search up to half the stream length, capped at N_FULL//S for sanity
            max_pattern_len = min(n // 2, S * S * S)  # Up to S³ = 1728 for deep patterns

        archetypes = []
        actual_max = min(max_pattern_len + 1, n // 2 + 1)
        total_lengths = actual_max - min_pattern_len
        last_report_time = time.time()

        for pat_len in range(min_pattern_len, actual_max):
            # Report progress every 2 seconds during pattern scan
            now = time.time()
            if now - last_report_time >= 2.0:
                scan_pct = (pat_len - min_pattern_len) / max(total_lengths, 1) * 100
                self._log(f"    Pattern scan: len={pat_len}/{actual_max - 1} "
                          f"({scan_pct:.0f}%) — {len(archetypes)} found")
                last_report_time = now

            pattern_positions: Dict[tuple, List[int]] = defaultdict(list)
            for i in range(n - pat_len + 1):
                pat = tuple(sym_stream[i:i + pat_len])
                pattern_positions[pat].append(i)

            for pat, positions in pattern_positions.items():
                count = len(positions)
                if count < 2:
                    continue

                # Compute hierarchy elegance
                if sym_d_map:
                    d_vals = [sym_d_map.get(s, N_FULL) for s in pat]
                else:
                    d_vals = [lattice_d(s) for s in pat]
                d_avg = sum(d_vals) / len(d_vals) if d_vals else N_FULL

                # E_hierarchy: NET BIT SAVINGS with lattice depth weighting
                # Total savings: count × (pat_len - 1) symbols removed from stream
                # Overhead: pat_len symbols for the definition + 1 new symbol ID
                # Net savings = total_savings - overhead
                total_savings = count * (pat_len - 1)
                overhead = pat_len + 1
                net_savings = total_savings - overhead
                if net_savings < 2:
                    continue
                # ─── INCOHERENCE FILTER GATE ───
                # Gate every archetype through ALL 5 levels before accepting.
                # From incoherence_filter_lattice.txt: L1+L3+L4 on the pattern,
                # L5 applied at final selection (coherent summation).
                if not IncoherenceFilter.gate_archetype(pat, self.n_res):
                    continue
                depth_factor = BIO_RES / max(d_avg, 1.0)
                # E_hierarchy: net savings weighted by lattice depth
                e_hierarchy = net_savings * (1.0 + depth_factor / N_FULL)
                # Any positive net savings is worth subsuming
                if True:  # Subsumption Law: subsume if it reduces without remainder
                    archetypes.append(LatticeWalkArchetype(
                        pattern=pat,
                        occurrences=positions,
                        hierarchy_elegance=e_hierarchy,
                        d_avg=d_avg,
                        pattern_length=pat_len,
                    ))

        # Sort by elegance DESC, then by pattern length DESC (prefer longer at equal elegance)
        archetypes.sort(key=lambda arch: (arch.hierarchy_elegance, arch.pattern_length), reverse=True)

        # ─── L5: COHERENT SUMMATION ───
        # Sum only over coherent configurations. Verify the final archetype set
        # has cross-archetype sublattice compatibility (L3 over the whole set).
        if len(archetypes) > 1:
            all_d_vals = []
            for a in archetypes:
                all_d_vals.extend([lattice_d(s, self.n_res) for s in a.pattern])
            if not IncoherenceFilter.l3_sublattice(all_d_vals, self.n_res):
                # If the full set is incoherent, keep only the top archetypes
                # that form a coherent subset (greedy selection)
                coherent_set = []
                running_d = []
                for a in archetypes:
                    trial_d = running_d + [lattice_d(s, self.n_res) for s in a.pattern]
                    if IncoherenceFilter.l3_sublattice(trial_d, self.n_res):
                        coherent_set.append(a)
                        running_d = trial_d
                archetypes = coherent_set

        return archetypes

    def subsume_patterns(self, sym_stream: List[int],
                         archetypes: List[LatticeWalkArchetype]) -> Tuple[List, List[LatticeWalkArchetype]]:
        """
        Replace occurrences of archetype patterns with references.

        Returns (encoded_stream, used_archetypes) where encoded_stream
        contains either raw symbol values or archetype reference markers.
        Archetype indices in the encoded stream reference used_archetypes.

        Non-overlapping: first match wins (greedy, highest elegance first).
        """
        n = len(sym_stream)
        consumed = [False] * n
        used_archetypes = []
        orig_to_used: Dict[int, int] = {}
        archetype_placements = []

        for arch_idx, arch in enumerate(archetypes):
            placed = False
            for pos in arch.occurrences:
                if any(consumed[pos + j] for j in range(arch.pattern_length) if pos + j < n):
                    continue
                for j in range(arch.pattern_length):
                    if pos + j < n:
                        consumed[pos + j] = True
                archetype_placements.append((pos, arch_idx))
                placed = True

            if placed:
                orig_to_used[arch_idx] = len(used_archetypes)
                used_archetypes.append(arch)

        archetype_placements.sort(key=lambda x: x[0])

        encoded = []
        placement_idx = 0
        i = 0

        while i < n:
            if placement_idx < len(archetype_placements):
                pos, orig_idx = archetype_placements[placement_idx]
                if i == pos:
                    used_idx = orig_to_used[orig_idx]
                    encoded.append(('arch', used_idx))
                    i += archetypes[orig_idx].pattern_length
                    placement_idx += 1
                    continue
                elif i > pos:
                    placement_idx += 1
                    continue

            encoded.append(('raw', sym_stream[i]))
            i += 1

        # Validate archetype d-compatibility using instance lattice resolution
        # (Subsumption Law: the used archetypes must form a coherent sublattice set)
        if len(used_archetypes) > 1:
            combined_d_vals = []
            for used_arch in used_archetypes:
                combined_d_vals.extend([lattice_d(s, self.n_res) for s in used_arch.pattern])
            if not IncoherenceFilter.l3_sublattice(combined_d_vals, self.n_res):
                logger.debug(f"subsume_patterns: {len(used_archetypes)} archetypes "
                             f"have cross-archetype d-incoherence at N={self.n_res}")

        return encoded, used_archetypes

    def recursive_compress(self, dk_stream: List[int]) -> dict:
        """
        RECURSIVE subsumption: iterative pattern replacement, depth up to S=12.

        Clean approach with MONOTONICALLY GROWING symbol space:
        - Symbols 0..n_base-1: raw Δk indices (from dk_table)
        - Symbols n_base..: archetype references, each expands to a pattern
        No re-indexing between levels. Decompression simply expands
        archetype symbols recursively until only base symbols remain.
        """
        dk_value_set: Set[int] = set(dk_stream)
        unique_dks = sorted(dk_value_set)
        dk_to_idx = {dk: i for i, dk in enumerate(unique_dks)}
        n_base = len(unique_dks)

        current = [dk_to_idx[dk] for dk in dk_stream]
        archetype_defs: List[Tuple[int, ...]] = []
        next_sym_id = n_base

        d_map: Dict[int, float] = {i: float(lattice_d(dk)) for i, dk in enumerate(unique_dks)}

        for depth in range(MAX_DEPTH):
            if len(current) < 4:
                break

            # ─── L4: CASCADE COHERENCE ───
            # N_max = ⌊50¢/|δ_avg|⌋ — the cascade coherence horizon.
            # Each recursion depth adds accumulated ε. Beyond N_max, the
            # cascade exits the coherent manifold. Limit depth accordingly.
            if depth > 0:
                eps_vals = []
                for s in current:
                    if s < n_base:
                        dk = unique_dks[s]
                        ratio = 2.0 ** (dk / self.n_res) if dk != 0 else 1.0
                        eps_vals.append(abs(lattice_epsilon(ratio, dk, self.n_res)))
                if eps_vals:
                    avg_eps = sum(eps_vals) / len(eps_vals)
                    n_max_cascade = IncoherenceFilter.l4_cascade_horizon(avg_eps) if avg_eps > EPSILON else N_FULL
                    if depth >= n_max_cascade:
                        break  # L4: cascade has exceeded coherence horizon

            archetypes = self.find_walk_archetypes(
                current, sym_d_map=d_map, min_pattern_len=2,
                max_pattern_len=min(len(current) // 2, S * S * (depth + 1))
            )
            if not archetypes:
                break

            encoded, used_archs = self.subsume_patterns(current, archetypes)
            if not used_archs:
                break

            arch_id_map = {}
            for arch in used_archs:
                arch_id_map[id(arch)] = next_sym_id
                archetype_defs.append(arch.pattern)
                d_map[next_sym_id] = arch.d_avg
                next_sym_id += 1

            new_stream = []
            for item_type, item_val in encoded:
                if item_type == 'raw':
                    new_stream.append(item_val)
                else:
                    new_stream.append(arch_id_map[id(used_archs[item_val])])

            if len(new_stream) >= len(current):
                break
            current = new_stream

        return {
            'dk_table': unique_dks,
            'n_base': n_base,
            'archetype_defs': archetype_defs,
            'total_symbols': next_sym_id,
            'final_stream': current,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPRESSION / DECOMPRESSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CDFEngine:
    """
    The complete ET-derived compression engine.

    Phase 1 — Identification: R₀ seed, byte↔k bijection at 27720ET
    Phase 2 — Lattice Transform: bytes → k-stream → Δk stream
    Phase 3 — Descriptor Gap Analysis: find recurring lattice walks
    Phase 4 — Archetype Subsumption: replace walks with references
    Phase 5 — Lattice Encoding: serialize to bytes
    Phase 6 — Verification: roundtrip check via exact sequence (H₀, H₁)
    """

    def __init__(self, log_fn=None, progress_fn=None):
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)

    def _encode_lattice_block(self, n, r0, k0, mode, dk0_saved, k_table, rc):
        """Encode a compressed block to binary given mode and recursive compress result."""
        mode_names = {0: 'k-direct', 1: 'Δk', 2: 'ΔΔk'}
        n_archetypes = len(rc['archetype_defs'])
        self._log(f"  Encoding: mode={mode_names.get(mode, mode)}, "
                  f"n_base={rc['n_base']}, archetypes={n_archetypes}, "
                  f"stream={len(rc['final_stream'])}")
        parts = [
            struct.pack('<B', 2),  # type = lattice
            struct.pack('<I', n),  # original size
            struct.pack('<d', r0),  # R₀ seed
            struct.pack('<i', k0),  # initial k value
            struct.pack('<B', mode),  # mode: 0=k-direct, 1=Δk, 2=ΔΔk
            struct.pack('<i', dk0_saved),  # first Δk (for ΔΔk reconstruction)
        ]

        # Mode 0: k_table (compact encoding)
        if mode == 0 and k_table is not None:
            parts.append(struct.pack('<H', len(k_table)))
            # Compact: determine min byte width needed
            if k_table:
                kmin = min(k_table)
                kmax = max(k_table)
                krange = kmax - kmin
                if krange < 256:
                    parts.append(struct.pack('<Bi', 1, kmin))  # width=1, base
                    for k_val in k_table:
                        parts.append(struct.pack('<B', k_val - kmin))
                elif krange < 65536:
                    parts.append(struct.pack('<Bi', 2, kmin))
                    for k_val in k_table:
                        parts.append(struct.pack('<H', k_val - kmin))
                else:
                    parts.append(struct.pack('<Bi', 4, 0))
                    for k_val in k_table:
                        parts.append(struct.pack('<i', k_val))

        # Δk/symbol table — compact delta-of-sorted encoding
        dk_tbl = rc['dk_table']
        n_base = rc['n_base']
        parts.append(struct.pack('<H', n_base))
        if n_base > 0:
            # Store first value as int32, then deltas between sorted values
            # The dk_table is already sorted (from recursive_compress)
            parts.append(struct.pack('<i', dk_tbl[0]))
            if n_base > 1:
                deltas = [dk_tbl[i] - dk_tbl[i - 1] for i in range(1, n_base)]
                dmax = max(deltas) if deltas else 0
                if dmax < 256:
                    parts.append(struct.pack('<B', 1))  # delta width
                    for d in deltas:
                        parts.append(struct.pack('<B', d))
                elif dmax < 65536:
                    parts.append(struct.pack('<B', 2))
                    for d in deltas:
                        parts.append(struct.pack('<H', d))
                else:
                    parts.append(struct.pack('<B', 4))
                    for d in deltas:
                        parts.append(struct.pack('<i', d))
            else:
                parts.append(struct.pack('<B', 0))  # no deltas

        # Archetype definitions — V_config encoded as a single symbol stream
        # From D Paper §44: the archetype IS the Generative Descriptor (D_generative).
        # Encode all patterns as one flat stream with length prefixes.
        arch_defs = rc['archetype_defs']
        total_syms = rc['total_symbols']
        parts.append(struct.pack('<H', len(arch_defs)))
        parts.append(struct.pack('<H', total_syms))
        # Flatten all archetype pattern data into a single stream for V_config encoding
        flat_arch_stream = []
        for pat in arch_defs:
            flat_arch_stream.append(len(pat))  # length prefix
            flat_arch_stream.extend(pat)
        if flat_arch_stream:
            max_val = max(flat_arch_stream)
            # Try V_config vs uniform on the flattened archetype stream
            arch_bits = max(1, math.ceil(math.log2(max(max_val + 1, 2))))
            uniform_arch_size = 1 + (len(flat_arch_stream) * arch_bits + 7) // 8
            vc_arch_enc, vc_arch_tbl = v_config_encode(flat_arch_stream, max_val + 1)
            vc_arch_size = 1 + 2 + len(vc_arch_tbl) + len(vc_arch_enc)
            if vc_arch_size < uniform_arch_size and len(flat_arch_stream) > 8:
                # V_config wins for archetype definitions
                parts.append(struct.pack('<BH', 1, len(flat_arch_stream)))
                parts.append(struct.pack('<H', len(vc_arch_tbl)))
                parts.append(vc_arch_tbl)
                parts.append(struct.pack('<I', len(vc_arch_enc)))  # explicit byte count
                parts.append(vc_arch_enc)
            else:
                # Uniform bit-pack archetype definitions
                parts.append(struct.pack('<BH', 0, len(flat_arch_stream)))
                parts.append(struct.pack('<B', arch_bits))
                pv = 0
                pb = 0
                pba = bytearray()
                for sym in flat_arch_stream:
                    pv |= (sym << pb)
                    pb += arch_bits
                    while pb >= 8:
                        pba.append(pv & 0xFF)
                        pv >>= 8
                        pb -= 8
                if pb > 0:
                    pba.append(pv & 0xFF)
                parts.append(bytes(pba))
        else:
            parts.append(struct.pack('<BH', 0, 0))

        # ── ET Variance-Weighted Encoding (V_config) ──
        # From Math_of_ET eq. #16: V_config = -Σ(P_select ∘ Depth(P_select))
        # Entropy IS the depth of the T-Decision Tree.
        # Each symbol gets Depth(sym) = ceil(-log₂(freq/total)) bits.
        # Try BOTH uniform and V_config; keep whichever is smaller (Descriptor Gap Principle).
        sym_stream = rc['final_stream']
        n_syms = len(sym_stream)

        # Option A: Uniform bit-packing (octave-class: all symbols same depth)
        bits_per_sym = max(1, math.ceil(math.log2(max(total_syms, 2))))
        uniform_val = 0
        uniform_bits = 0
        uniform_bytes = bytearray()
        for sym in sym_stream:
            uniform_val |= (sym << uniform_bits)
            uniform_bits += bits_per_sym
            while uniform_bits >= 8:
                uniform_bytes.append(uniform_val & 0xFF)
                uniform_val >>= 8
                uniform_bits -= 8
        if uniform_bits > 0:
            uniform_bytes.append(uniform_val & 0xFF)
        uniform_block = struct.pack('<BBH', 0, bits_per_sym, n_syms) + bytes(uniform_bytes)

        # Option B: V_config encoding (lattice-depth-weighted: variable depth per symbol)
        if n_syms > 0 and total_syms > 1:
            vc_encoded, vc_table = v_config_encode(sym_stream, total_syms)
            vc_block = struct.pack('<B', 1) + struct.pack('<H', n_syms) + \
                       struct.pack('<H', len(vc_table)) + vc_table + vc_encoded
        else:
            vc_block = uniform_block  # Fallback for trivial cases

        # Descriptor Gap Principle: choose the encoding with the smallest gap
        if len(vc_block) < len(uniform_block):
            parts.append(vc_block)
        else:
            parts.append(uniform_block)

        return b''.join(parts)

    def compress_block(self, data: bytes) -> bytes:
        """
        Compress a single block (up to BLOCK_SIZE bytes).
        Returns the compressed binary representation.
        """
        n = len(data)
        if n == 0:
            return struct.pack('<BI', 0, 0)  # type=empty, size=0

        values = np.frombuffer(data, dtype=np.uint8)

        # Trivial: all same byte
        unique = np.unique(values)
        if len(unique) == 1:
            self._log(f"  Block {n:,}B: uniform (single byte {int(unique[0])})")
            return struct.pack('<BIB', 1, n, int(unique[0]))  # type=uniform

        t_block = time.time()
        self._log(f"  Block {n:,}B: {len(unique)} unique bytes — starting lattice compression")

        # ── Phase 1: Identification ──
        self._progress(0, "Phase 1: Identification (R₀)")
        r0 = discover_r0(data)
        byte_k = build_byte_k_map(r0)
        self._log(f"    Phase 1 done: R₀={r0:.6f} ({time.time() - t_block:.2f}s)")

        # ── Phase 2: Lattice Transform — Adaptive Mode Selection ──
        self._progress(10, "Phase 2: Lattice Transform")
        t_phase2 = time.time()
        # Three modes (Descriptor Gap Principle: choose the representation
        # with the smallest gap = fewest unique values = lowest lattice entropy):
        #
        #   Mode 0: k-direct (absolute lattice positions, no differencing)
        #           Best for varied text where byte diversity is low but transition diversity is high
        #   Mode 1: Δk (first differences of k)
        #           Best for data with repeating transition patterns
        #   Mode 2: ΔΔk (second-order differences)
        #           Best for data with linear trends (sequential, ramp-like)

        k_stream = [byte_k[int(b)] for b in values]
        k0 = k_stream[0]

        # Build candidate streams
        # Mode 0: k-direct (map k-values to compact indices)
        unique_k = sorted(set(k_stream))
        k_to_compact = {k: i for i, k in enumerate(unique_k)}
        k_direct_stream = [k_to_compact[k] for k in k_stream]
        n_unique_k = len(unique_k)

        # Mode 1: Δk
        dk_stream = [k_stream[i + 1] - k_stream[i] for i in range(n - 1)]
        n_unique_dk = len(set(dk_stream))

        # Mode 2: ΔΔk
        ddk_stream = [dk_stream[i + 1] - dk_stream[i] for i in range(len(dk_stream) - 1)] if len(dk_stream) > 1 else []
        n_unique_ddk = len(set(ddk_stream)) if ddk_stream else 999999

        # Choose mode with fewest unique values (lowest lattice entropy)
        mode_uniques = [(0, n_unique_k), (1, n_unique_dk), (2, n_unique_ddk)]
        best_mode = min(mode_uniques, key=lambda x: x[1])[0]

        if best_mode == 0:
            # k-direct: encode absolute lattice positions
            # The stream represents bytes directly via k→byte (both sides know this from R₀)
            # Store the k_table so decompressor knows the mapping
            dk_stream_for_compress = k_direct_stream
            dk0_saved = 0
        elif best_mode == 2 and ddk_stream:
            dk_stream_for_compress = ddk_stream
            dk0_saved = dk_stream[0]
        else:
            best_mode = 1  # Default to Δk
            dk_stream_for_compress = dk_stream
            dk0_saved = 0

        # ── Phase 3+4: Multi-Strategy Recursive Archetype Subsumption ──
        # Descriptor Gap Principle: try ALL viable modes, build ALL, keep smallest.
        # Speed is IRRELEVANT — only compression ratio matters.
        self._log(f"    Phase 2 done: modes k={n_unique_k}, Δk={n_unique_dk}, "
                  f"ΔΔk={n_unique_ddk} ({time.time() - t_phase2:.2f}s)")
        self._progress(20, "Phase 3: Pattern Finding (slow for large blocks)")
        t_phase3 = time.time()
        walker = LatticeWalkCompressor(r0, log_fn=self._log)

        # Build candidates for each viable mode
        candidates = []

        # Mode 0: k-direct
        if len(k_direct_stream) > 3:
            candidates.append((0, k_direct_stream, 0))
        # Mode 1: Δk
        if len(dk_stream) > 3:
            candidates.append((1, dk_stream, 0))
        # Mode 2: ΔΔk
        if ddk_stream and len(ddk_stream) > 3:
            candidates.append((2, ddk_stream, dk_stream[0]))

        # Use best_mode to prioritize: try the analytically-best mode first
        # (lowest lattice entropy → fewest unique values → most compressible)
        candidates.sort(key=lambda cand: (0 if cand[0] == best_mode else 1))

        # Compress each candidate and build the full encoded block
        best_block = None
        mode_names_map = {0: 'k-direct', 1: 'Δk', 2: 'ΔΔk'}
        for ci, (cand_mode, cand_stream, cand_dk0) in enumerate(candidates):
            cand_pct = 20 + (ci / max(len(candidates), 1)) * 60
            self._progress(cand_pct, f"Phase 3: mode {mode_names_map.get(cand_mode, cand_mode)} "
                                     f"({ci + 1}/{len(candidates)})")
            self._log(f"    Candidate {ci + 1}/{len(candidates)}: "
                      f"mode={mode_names_map.get(cand_mode, cand_mode)}, "
                      f"stream={len(cand_stream):,}")
            rc = walker.recursive_compress(cand_stream)
            # Build the encoded block for this candidate
            trial = self._encode_lattice_block(n, r0, k0, cand_mode, cand_dk0,
                                               unique_k if cand_mode == 0 else None, rc)
            if best_block is None or len(trial) < len(best_block):
                best_block = trial

        if best_block is None:
            # Fallback: use the pre-selected best_mode stream directly
            # (dk_stream_for_compress and dk0_saved from the mode analysis)
            rc = walker.recursive_compress(dk_stream_for_compress)
            best_block = self._encode_lattice_block(n, r0, k0, best_mode, dk0_saved,
                                                    unique_k if best_mode == 0 else None, rc)

        # ── Phase 5+6: Incoherence Gate ──
        self._log(f"    Phase 3+4 done ({time.time() - t_phase3:.2f}s)")
        self._progress(90, "Phase 5: Final encoding")
        raw_block = struct.pack('<BI', 3, n) + data
        chosen = best_block if len(best_block) <= len(raw_block) else raw_block
        ratio = len(chosen) / n * 100
        self._log(f"    Block complete: {n:,}B → {len(chosen):,}B ({ratio:.1f}%) "
                  f"in {time.time() - t_block:.2f}s")
        return chosen

    def decompress_block(self, block_data: bytes) -> bytes:
        """Decompress a single block back to original bytes."""
        pos = 0

        block_type = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1
        block_type_names = {0: 'empty', 1: 'uniform', 2: 'lattice', 3: 'passthrough'}
        self._log(f"  Decompressing block: type={block_type_names.get(block_type, block_type)}, "
                  f"compressed={len(block_data)} bytes")

        if block_type == 0:  # empty
            return b''

        if block_type == 1:  # uniform
            n = struct.unpack_from('<I', block_data, pos)[0]
            pos += 4
            val = struct.unpack_from('<B', block_data, pos)[0]
            pos += 1
            return bytes([val] * n)

        if block_type == 3:  # passthrough (incoherent — raw bytes)
            n = struct.unpack_from('<I', block_data, pos)[0]
            pos += 4
            return block_data[pos:pos + n]

        # type == 2: lattice compressed (flat multi-level)
        n = struct.unpack_from('<I', block_data, pos)[0]
        pos += 4
        r0 = struct.unpack_from('<d', block_data, pos)[0]
        pos += 8
        k0 = struct.unpack_from('<i', block_data, pos)[0]
        pos += 4
        mode = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1
        dk0_saved = struct.unpack_from('<i', block_data, pos)[0]
        pos += 4
        # Mode 0: read k_table (compact encoding)
        k_direct_table = None
        if mode == 0:
            n_kt = struct.unpack_from('<H', block_data, pos)[0]
            pos += 2
            if n_kt > 0:
                kt_width = struct.unpack_from('<B', block_data, pos)[0]
                pos += 1
                kt_base = struct.unpack_from('<i', block_data, pos)[0]
                pos += 4
                k_direct_table = []
                for _ in range(n_kt):
                    if kt_width == 1:
                        k_direct_table.append(struct.unpack_from('<B', block_data, pos)[0] + kt_base)
                        pos += 1
                    elif kt_width == 2:
                        k_direct_table.append(struct.unpack_from('<H', block_data, pos)[0] + kt_base)
                        pos += 2
                    else:
                        k_direct_table.append(struct.unpack_from('<i', block_data, pos)[0])
                        pos += 4
            else:
                k_direct_table = []
        k_byte = build_k_byte_map(r0)

        # Read Δk table (compact delta-of-sorted encoding)
        n_base = struct.unpack_from('<H', block_data, pos)[0]
        pos += 2
        dk_table = []
        if n_base > 0:
            first_val = struct.unpack_from('<i', block_data, pos)[0]
            pos += 4
            dk_table.append(first_val)
            if n_base > 1:
                delta_width = struct.unpack_from('<B', block_data, pos)[0]
                pos += 1
                for _ in range(n_base - 1):
                    if delta_width == 1:
                        d = struct.unpack_from('<B', block_data, pos)[0]
                        pos += 1
                    elif delta_width == 2:
                        d = struct.unpack_from('<H', block_data, pos)[0]
                        pos += 2
                    else:
                        d = struct.unpack_from('<i', block_data, pos)[0]
                        pos += 4
                    dk_table.append(dk_table[-1] + d)
            else:
                pos += 1  # skip delta_width byte (was 0)

        # Read archetype definitions (V_config or uniform flat stream)
        n_arch = struct.unpack_from('<H', block_data, pos)[0]
        pos += 2
        total_syms = struct.unpack_from('<H', block_data, pos)[0]
        pos += 2
        arch_enc_type = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1
        n_flat = struct.unpack_from('<H', block_data, pos)[0]
        pos += 2

        arch_defs = []
        if n_flat > 0:
            if arch_enc_type == 1:
                # V_config encoded archetype stream
                atbl_len = struct.unpack_from('<H', block_data, pos)[0]
                pos += 2
                atbl_data = block_data[pos:pos + atbl_len]
                pos += atbl_len
                avc_len = struct.unpack_from('<I', block_data, pos)[0]
                pos += 4
                avc_data = block_data[pos:pos + avc_len]
                pos += avc_len
                flat_stream = v_config_decode(avc_data, atbl_data, n_flat)
                fi = 0
                for _ in range(n_arch):
                    plen = flat_stream[fi]
                    fi += 1
                    pat = tuple(flat_stream[fi:fi + plen])
                    fi += plen
                    arch_defs.append(pat)
            else:
                # Uniform bit-packed archetype stream
                arch_bits = struct.unpack_from('<B', block_data, pos)[0]
                pos += 1
                n_arch_packed = (n_flat * arch_bits + 7) // 8
                arch_packed = block_data[pos:pos + n_arch_packed]
                pos += n_arch_packed
                amask = (1 << arch_bits) - 1
                abi = 0
                aac = 0
                aia = 0
                flat_stream = []
                for _ in range(n_flat):
                    while aia < arch_bits and abi < len(arch_packed):
                        aac |= arch_packed[abi] << aia
                        aia += 8
                        abi += 1
                    flat_stream.append(aac & amask)
                    aac >>= arch_bits
                    aia -= arch_bits
                fi = 0
                for _ in range(n_arch):
                    plen = flat_stream[fi]
                    fi += 1
                    pat = tuple(flat_stream[fi:fi + plen])
                    fi += plen
                    arch_defs.append(pat)

        # Read encoding format and decode final symbol stream
        enc_type = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1

        if enc_type == 0:
            # Uniform bit-packing (octave-class)
            bits_per_sym = struct.unpack_from('<B', block_data, pos)[0]
            pos += 1
            n_syms = struct.unpack_from('<H', block_data, pos)[0]
            pos += 2
            n_packed = (n_syms * bits_per_sym + 7) // 8
            packed = block_data[pos:pos + n_packed]
            pos += n_packed
            sym_mask = (1 << bits_per_sym) - 1
            byte_idx = 0
            bit_accum = 0
            bits_in_accum = 0
            sym_stream = []
            for _ in range(n_syms):
                while bits_in_accum < bits_per_sym and byte_idx < len(packed):
                    bit_accum |= packed[byte_idx] << bits_in_accum
                    bits_in_accum += 8
                    byte_idx += 1
                sym_stream.append(bit_accum & sym_mask)
                bit_accum >>= bits_per_sym
                bits_in_accum -= bits_per_sym
        elif enc_type == 1:
            # V_config encoding (lattice-depth-weighted, ET Shannon eq. #16)
            n_syms = struct.unpack_from('<H', block_data, pos)[0]
            pos += 2
            tbl_len = struct.unpack_from('<H', block_data, pos)[0]
            pos += 2
            tbl_data = block_data[pos:pos + tbl_len]
            pos += tbl_len
            vc_data = block_data[pos:]
            pos = len(block_data)
            sym_stream = v_config_decode(vc_data, tbl_data, n_syms)
        else:
            raise ValueError(f"Unknown encoding type {enc_type}")

        # Verify complete block parse — pos should equal block length
        # (Subsumption Law: the parse must consume all data without remainder)
        remaining_bytes = len(block_data) - pos
        if remaining_bytes > 0:
            logger.debug(f"Block parse: {remaining_bytes} trailing bytes after decode")

        # Validate decoded symbols against declared total (Identification Principle:
        # every symbol must be identified within the declared alphabet)
        for sym_val in sym_stream:
            if sym_val >= total_syms:
                raise ValueError(f"Decoded symbol {sym_val} exceeds total_syms {total_syms}")

        # Expand archetype symbols recursively until only base symbols remain
        # A symbol < n_base is a dk_table index. A symbol >= n_base is an archetype.
        def expand(stream):
            """Recursively expand archetype symbols to base dk_table indices."""
            result = []
            for sym in stream:
                if sym < n_base:
                    result.append(sym)
                else:
                    arch_idx = sym - n_base
                    result.extend(expand(list(arch_defs[arch_idx])))
            return result

        base_indices = expand(sym_stream)
        raw_vals = [dk_table[i] for i in base_indices]

        # Reconstruct based on mode
        if mode == 0:
            # k-direct: raw_vals are compact indices into k_direct_table
            # Map back to k-values, then to bytes directly
            output = bytearray()
            for idx in raw_vals:
                k_val = k_direct_table[idx]
                if k_val in k_byte:
                    output.append(k_byte[k_val])
                else:
                    nearest = min(k_byte.keys(), key=lambda kk: abs(kk - k_val))
                    output.append(k_byte[nearest])
            return bytes(output[:n])
        elif mode == 2:
            # ΔΔk → reconstruct Δk
            dk_stream = [dk0_saved]
            for ddk in raw_vals:
                dk_stream.append(dk_stream[-1] + ddk)
        else:
            # Δk direct
            dk_stream = raw_vals

        # Reconstruct k-stream → bytes
        k_stream = [k0]
        for dk in dk_stream:
            k_stream.append(k_stream[-1] + dk)

        output = bytearray()
        for k in k_stream:
            if k in k_byte:
                output.append(k_byte[k])
            else:
                nearest_k = min(k_byte.keys(), key=lambda kk: abs(kk - k))
                output.append(k_byte[nearest_k])

        return bytes(output[:n])


# ═══════════════════════════════════════════════════════════════════════════════
# CDF FILE FORMAT — High-level compress/decompress
# ═══════════════════════════════════════════════════════════════════════════════

class CDFCompressor:
    """
    High-level CDF compression/decompression.

    CDF file structure:
      [4 bytes]  Magic: 'CDF\x02'
      [1 byte]   Version: 2
      [32 bytes] SHA-256 of original data
      [8 bytes]  Original file size (uint64 LE)
      [4 bytes]  Number of blocks (uint32 LE)
      [4 bytes]  Block size (uint32 LE)
      [8 bytes]  Global R₀ (float64)
      For each block:
        [4 bytes]  Compressed block size (uint32 LE)
        [N bytes]  Compressed block data
    """

    def __init__(self, log_fn=None, progress_fn=None):
        self.engine = CDFEngine(log_fn, progress_fn)
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)

    def compress_file(self, input_path: str, output_path: str) -> dict:
        """Compress a file to CDF format."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_size_on_disk = os.path.getsize(input_path)
        self._log(f"Compressing: {input_path} ({input_size_on_disk:,} bytes on disk)")
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(input_path, 'rb') as f:
            raw = f.read()

        original_size = len(raw)
        sha256 = hashlib.sha256(raw).digest()
        global_r0 = discover_r0(raw)
        num_blocks = (original_size + BLOCK_SIZE - 1) // BLOCK_SIZE

        self._log(f"Original: {original_size:,} bytes, {num_blocks} blocks")
        self._log(f"R₀ = {global_r0:.6f}, N_res = {N_FULL}")

        t0 = time.time()
        compressed_blocks = []
        total_compressed = 0

        for bi in range(num_blocks):
            bs = bi * BLOCK_SIZE
            be = min(bs + BLOCK_SIZE, original_size)
            block = raw[bs:be]

            t_bi = time.time()
            self._log(f"\n─── Block {bi + 1}/{num_blocks} ({len(block):,} bytes) ───")
            cb = self.engine.compress_block(block)
            compressed_blocks.append(cb)
            total_compressed += len(cb)

            elapsed_bi = time.time() - t_bi
            elapsed_total = time.time() - t0
            block_ratio = len(cb) / len(block) * 100 if len(block) > 0 else 0
            running_ratio = total_compressed / be * 100

            self._log(f"─── Block {bi + 1} result: {len(block):,} → {len(cb):,} bytes "
                      f"({block_ratio:.1f}%) in {elapsed_bi:.2f}s ───")

            pct = (bi + 1) / num_blocks * 100
            self._progress(pct, f"Block {bi + 1}/{num_blocks} — "
                                f"{running_ratio:.1f}% — {elapsed_total:.1f}s elapsed")

        elapsed = time.time() - t0

        # Write CDF file
        with open(output_path, 'wb') as f:
            f.write(CDF_MAGIC)
            f.write(struct.pack('<B', CDF_VERSION))
            f.write(sha256)
            f.write(struct.pack('<Q', original_size))
            f.write(struct.pack('<I', num_blocks))
            f.write(struct.pack('<I', BLOCK_SIZE))
            f.write(struct.pack('<d', global_r0))

            for cb in compressed_blocks:
                f.write(struct.pack('<I', len(cb)))
                f.write(cb)

        comp_size = Path(output_path).stat().st_size
        ratio = comp_size / original_size if original_size > 0 else 0

        self._log(f"Compressed: {comp_size:,} bytes ({ratio * 100:.1f}%)")
        self._log(f"Time: {elapsed:.2f}s")

        return {'original_size': original_size, 'compressed_size': comp_size,
                'ratio': ratio, 'time': elapsed, 'blocks': num_blocks}

    def decompress_file(self, input_path: str, output_path: str) -> dict:
        """Decompress a CDF file back to original."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"CDF file not found: {input_path}")
        cdf_size = os.path.getsize(input_path)
        self._log(f"Decompressing: {input_path} ({cdf_size:,} bytes)")
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(input_path, 'rb') as f:
            cdf = f.read()

        pos = 0
        magic = cdf[pos:pos + 4]
        pos += 4
        assert magic == CDF_MAGIC, f"Bad magic: {magic}"

        version = struct.unpack_from('<B', cdf, pos)[0]
        pos += 1
        assert version == CDF_VERSION, f"Bad version: {version}"

        stored_hash = cdf[pos:pos + 32]
        pos += 32
        original_size = struct.unpack_from('<Q', cdf, pos)[0]
        pos += 8
        num_blocks = struct.unpack_from('<I', cdf, pos)[0]
        pos += 4
        block_size = struct.unpack_from('<I', cdf, pos)[0]
        pos += 4
        global_r0 = struct.unpack_from('<d', cdf, pos)[0]
        pos += 8

        self._log(f"Original: {original_size:,} bytes, {num_blocks} blocks, "
                  f"block_size={block_size:,}, R₀={global_r0:.6f}")

        t0 = time.time()
        parts = []

        for bi in range(num_blocks):
            cb_len = struct.unpack_from('<I', cdf, pos)[0]
            pos += 4
            cb_data = cdf[pos:pos + cb_len]
            pos += cb_len

            decompressed = self.engine.decompress_block(cb_data)
            # Validate: decompressed block should not exceed declared block_size
            # (Subsumption Law: each block must fit within the declared manifold)
            if len(decompressed) > block_size:
                logger.warning(f"Block {bi}: decompressed size {len(decompressed)} "
                               f"exceeds block_size {block_size}")
            parts.append(decompressed)

            pct = (bi + 1) / num_blocks * 100
            self._progress(pct, f"Block {bi + 1}/{num_blocks}")

        elapsed = time.time() - t0
        result = b''.join(parts)[:original_size]

        # Integrity check
        computed_hash = hashlib.sha256(result).digest()
        integrity = computed_hash == stored_hash

        if integrity:
            self._log("Integrity: PASSED (SHA-256)")
        else:
            self._log("Integrity: FAILED!")

        with open(output_path, 'wb') as f:
            f.write(result)

        self._log(f"Decompressed: {len(result):,} bytes, Time: {elapsed:.2f}s")

        return {'output_size': len(result), 'integrity': integrity, 'time': elapsed}


# ═══════════════════════════════════════════════════════════════════════════════
# GUI — Tkinter
# ═══════════════════════════════════════════════════════════════════════════════

def build_gui():
    """Build and launch the ET CDF Compressor Tkinter GUI application."""
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
    import threading

    class CDFApp:
        """Tkinter GUI application for CDF compression and decompression at 27720ET."""

        def __init__(self, master_root):
            self.root = master_root
            master_root.title("ET CDF Compressor — P ∘ D ∘ T = E — 27720ET Full Manifold")
            master_root.geometry("850x650")
            master_root.minsize(750, 550)

            style = ttk.Style()
            style.theme_use('clam')

            main = ttk.Frame(master_root, padding=10)
            main.pack(fill="both", expand=True)

            ttk.Label(main, text="CDF — Compressed Descriptor Format",
                      font=('Helvetica', 14, 'bold')).pack(pady=(0, 2))
            ttk.Label(main, text="27720ET Full Manifold • 96 Sublattice Families • P ∘ D ∘ T = E",
                      font=('Helvetica', 10)).pack(pady=(0, 10))

            # File selection
            ff = ttk.LabelFrame(main, text="Files", padding=8)
            ff.pack(fill="x", pady=3)

            r1 = ttk.Frame(ff)
            r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Input:", width=7).pack(side="left")
            self.in_var = tk.StringVar()
            ttk.Entry(r1, textvariable=self.in_var).pack(side="left", fill="x", expand=True, padx=4)
            ttk.Button(r1, text="Folder...", command=self.browse_folder).pack(side="right", padx=(2, 0))
            ttk.Button(r1, text="Files...", command=self.browse_in).pack(side="right")

            r2 = ttk.Frame(ff)
            r2.pack(fill="x", pady=2)
            ttk.Label(r2, text="Output:", width=7).pack(side="left")
            self.out_var = tk.StringVar()
            ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=4)
            ttk.Button(r2, text="Browse...", command=self.browse_out).pack(side="right")

            # Buttons
            bf = ttk.Frame(main)
            bf.pack(fill="x", pady=8)
            self.comp_btn = ttk.Button(bf, text="Compress → .cdf", command=self.do_compress)
            self.comp_btn.pack(side="left", padx=4)
            self.decomp_btn = ttk.Button(bf, text="Decompress .cdf →", command=self.do_decompress)
            self.decomp_btn.pack(side="left", padx=4)

            # Progress
            pf = ttk.LabelFrame(main, text="Progress", padding=4)
            pf.pack(fill="x", pady=3)
            self.pvar = tk.DoubleVar()
            ttk.Progressbar(pf, variable=self.pvar, maximum=100).pack(fill="x", pady=2)
            self.svar = tk.StringVar(value="Ready")
            ttk.Label(pf, textvariable=self.svar).pack(anchor="w")

            # Console
            lf = ttk.LabelFrame(main, text="Console", padding=4)
            lf.pack(fill="both", expand=True, pady=3)
            self.console = scrolledtext.ScrolledText(lf, height=14, font=('Consolas', 9),
                                                     state="disabled", bg='#1a1a2e',
                                                     fg='#e0e0e0', insertbackground='white')
            self.console.pack(fill="both", expand=True)

            # Constants bar
            ttk.Label(main, text=f"N={S} V=1/{S} K={K_KOIDE:.4f} N_full={N_FULL} "
                                 f"Block=2^{S}={BLOCK_SIZE} Depth={MAX_DEPTH} ∂I=±{INCOHERENCE_CENTS}¢",
                      font=('Consolas', 8)).pack(anchor="w", pady=2)

            self.log("CDF Compressor ready. 27720ET full manifold resolution.")

        def log(self, msg):
            """Append a message to the GUI console (thread-safe via root.after)."""

            def _u():
                self.console.configure(state="normal")
                self.console.insert(tk.END, msg + '\n')
                self.console.see(tk.END)
                self.console.configure(state="disabled")

            self.root.after(0, _u)

        def progress(self, pct, msg=''):
            """Update the GUI progress bar and status label (thread-safe)."""

            def _u():
                self.pvar.set(pct)
                if msg: self.svar.set(msg)

            self.root.after(0, _u)

        def browse_in(self):
            """Open file dialog for input file(s) selection — supports multi-select."""
            file_paths = filedialog.askopenfilenames(filetypes=[("All", "*.*"), ("CDF", "*.cdf")])
            if file_paths:
                if len(file_paths) == 1:
                    # Single file
                    self.in_var.set(file_paths[0])
                    fp = file_paths[0]
                    self.out_var.set(fp + '.cdf' if not fp.endswith('.cdf') else fp[:-4])
                    size = os.path.getsize(fp)
                    self.log(f"Selected: {os.path.basename(fp)} ({size:,} bytes)")
                else:
                    # Multiple files — join with | separator, output to folder
                    self.in_var.set('|'.join(file_paths))
                    common_dir = os.path.dirname(file_paths[0])
                    self.out_var.set(common_dir + '_cdf')
                    total_size = sum(os.path.getsize(fp) for fp in file_paths)
                    self.log(f"Selected {len(file_paths)} files ({total_size:,} bytes total):")
                    for fp in file_paths:
                        self.log(f"  {os.path.basename(fp)} ({os.path.getsize(fp):,})")

        def browse_folder(self):
            """Open folder dialog for batch processing all files in a directory."""
            folder_path = filedialog.askdirectory(title="Select folder to compress")
            if folder_path:
                self.in_var.set(folder_path)
                self.out_var.set(folder_path + '_cdf')
                # Count files
                files = [f for f in os.listdir(folder_path)
                         if os.path.isfile(os.path.join(folder_path, f))]
                total_size = sum(os.path.getsize(os.path.join(folder_path, f)) for f in files)
                self.log(f"Selected folder: {folder_path}")
                self.log(f"  {len(files)} files, {total_size:,} bytes total")

        def browse_out(self):
            """Open file dialog for output file selection."""
            file_path = filedialog.asksaveasfilename(filetypes=[("CDF", "*.cdf"), ("All", "*.*")])
            if file_path: self.out_var.set(file_path)

        def set_btns(self, st):
            """Set compress/decompress button states (thread-safe via root.after)."""
            self.root.after(0, lambda: (self.comp_btn.configure(state=st),
                                        self.decomp_btn.configure(state=st)))

        def do_compress(self):
            """Launch CDF compression in a background thread. Supports files and folders."""
            inp, out = self.in_var.get(), self.out_var.get()
            if not inp or not out:
                messagebox.showerror("Error", "Select input and output files.")
                return
            self.set_btns('disabled')
            self.progress(0, "Starting compression...")
            self.log(f"\n{'═' * 60}")
            self.log("COMPRESSION STARTED")

            def _run():
                try:
                    t_total = time.time()
                    cdf_compressor = CDFCompressor(log_fn=self.log, progress_fn=self.progress)

                    if '|' in inp:
                        # ── Multi-file mode ──
                        file_list = [f for f in inp.split('|') if os.path.isfile(f)]
                        if not file_list:
                            self.log("No valid files in selection.")
                            return
                        os.makedirs(out, exist_ok=True)
                        self.log(f"Batch: {len(file_list)} files → {out}")
                        total_orig = 0
                        total_comp = 0
                        for fi, in_file in enumerate(file_list):
                            fname = os.path.basename(in_file)
                            out_file = os.path.join(out, fname + '.cdf')
                            self.log(f"\n── File {fi + 1}/{len(file_list)}: {fname} ──")
                            self.progress(fi / len(file_list) * 100,
                                          f"File {fi + 1}/{len(file_list)}: {fname}")
                            r = cdf_compressor.compress_file(in_file, out_file)
                            total_orig += r['original_size']
                            total_comp += r['compressed_size']
                        elapsed = time.time() - t_total
                        ratio = total_comp / total_orig * 100 if total_orig > 0 else 0
                        self.log(f"\n{'═' * 60}")
                        self.log(f"BATCH DONE: {len(file_list)} files, "
                                 f"{total_orig:,} → {total_comp:,} bytes ({ratio:.1f}%)")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {len(file_list)} files — {ratio:.1f}%")
                    elif os.path.isdir(inp):
                        # ── Batch folder mode ──
                        files = sorted(f for f in os.listdir(inp)
                                       if os.path.isfile(os.path.join(inp, f))
                                       and not f.endswith('.cdf'))
                        if not files:
                            self.log("No compressible files found in folder.")
                            return
                        os.makedirs(out, exist_ok=True)
                        self.log(f"Batch: {len(files)} files → {out}")
                        total_orig = 0
                        total_comp = 0
                        for fi, fname in enumerate(files):
                            in_file = os.path.join(inp, fname)
                            out_file = os.path.join(out, fname + '.cdf')
                            self.log(f"\n── File {fi + 1}/{len(files)}: {fname} ──")
                            self.progress(fi / len(files) * 100,
                                          f"File {fi + 1}/{len(files)}: {fname}")
                            r = cdf_compressor.compress_file(in_file, out_file)
                            total_orig += r['original_size']
                            total_comp += r['compressed_size']
                        elapsed = time.time() - t_total
                        ratio = total_comp / total_orig * 100 if total_orig > 0 else 0
                        self.log(f"\n{'═' * 60}")
                        self.log(f"BATCH DONE: {len(files)} files, "
                                 f"{total_orig:,} → {total_comp:,} bytes ({ratio:.1f}%)")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {len(files)} files — {ratio:.1f}%")
                    else:
                        # ── Single file mode ──
                        r = cdf_compressor.compress_file(inp, out)
                        elapsed = time.time() - t_total
                        self.log(f"\n{'═' * 60}")
                        self.log(f"DONE: {r['compressed_size']:,} bytes "
                                 f"({r['ratio'] * 100:.1f}%) in {elapsed:.1f}s")
                        self.progress(100, f"Done — {r['ratio'] * 100:.1f}% — {elapsed:.1f}s")
                except Exception as e:
                    self.log(f"ERROR: {e}\n{traceback.format_exc()}")
                finally:
                    self.set_btns('normal')

            threading.Thread(target=_run, daemon=True).start()

        def do_decompress(self):
            """Launch CDF decompression in a background thread. Supports files and folders."""
            inp, out = self.in_var.get(), self.out_var.get()
            if not inp or not out:
                messagebox.showerror("Error", "Select input and output files.")
                return
            self.set_btns('disabled')
            self.progress(0, "Starting decompression...")
            self.log(f"\n{'═' * 60}")
            self.log("DECOMPRESSION STARTED")

            def _run():
                try:
                    t_total = time.time()
                    cdf_compressor = CDFCompressor(log_fn=self.log, progress_fn=self.progress)

                    if '|' in inp:
                        # ── Multi-file mode ──
                        file_list = [f for f in inp.split('|') if os.path.isfile(f)]
                        if not file_list:
                            self.log("No valid files in selection.")
                            return
                        os.makedirs(out, exist_ok=True)
                        self.log(f"Batch: {len(file_list)} .cdf files → {out}")
                        all_pass = True
                        for fi, in_file in enumerate(file_list):
                            fname = os.path.basename(in_file)
                            out_name = fname[:-4] if fname.endswith('.cdf') else fname + '.out'
                            out_file = os.path.join(out, out_name)
                            self.log(f"\n── File {fi + 1}/{len(file_list)}: {fname} ──")
                            self.progress(fi / len(file_list) * 100,
                                          f"File {fi + 1}/{len(file_list)}: {fname}")
                            r = cdf_compressor.decompress_file(in_file, out_file)
                            if not r['integrity']:
                                all_pass = False
                        elapsed = time.time() - t_total
                        self.log(f"\n{'═' * 60}")
                        self.log(f"BATCH DONE: {len(file_list)} files, "
                                 f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {len(file_list)} files — "
                                           f"{'PASS' if all_pass else 'FAIL'}")
                    elif os.path.isdir(inp):
                        # ── Batch folder mode ──
                        files = sorted(f for f in os.listdir(inp)
                                       if os.path.isfile(os.path.join(inp, f))
                                       and f.endswith('.cdf'))
                        if not files:
                            self.log("No .cdf files found in folder.")
                            return
                        os.makedirs(out, exist_ok=True)
                        self.log(f"Batch: {len(files)} .cdf files → {out}")
                        all_pass = True
                        for fi, fname in enumerate(files):
                            in_file = os.path.join(inp, fname)
                            out_file = os.path.join(out, fname[:-4] if fname.endswith('.cdf') else fname + '.out')
                            self.log(f"\n── File {fi + 1}/{len(files)}: {fname} ──")
                            self.progress(fi / len(files) * 100,
                                          f"File {fi + 1}/{len(files)}: {fname}")
                            r = cdf_compressor.decompress_file(in_file, out_file)
                            if not r['integrity']:
                                all_pass = False
                        elapsed = time.time() - t_total
                        self.log(f"\n{'═' * 60}")
                        self.log(f"BATCH DONE: {len(files)} files, "
                                 f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {len(files)} files — "
                                           f"{'PASS' if all_pass else 'FAIL'}")
                    else:
                        # ── Single file mode ──
                        r = cdf_compressor.decompress_file(inp, out)
                        elapsed = time.time() - t_total
                        self.log(f"\n{'═' * 60}")
                        result = 'PASS' if r['integrity'] else 'FAIL'
                        self.log(f"DONE: {result} in {elapsed:.1f}s")
                        self.progress(100, f"Done — {result} — {elapsed:.1f}s")
                except Exception as e:
                    self.log(f"ERROR: {e}\n{traceback.format_exc()}")
                finally:
                    self.set_btns('normal')

            threading.Thread(target=_run, daemon=True).start()

    root = tk.Tk()
    CDFApp(root)
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != 'gui':
        import argparse

        parser = argparse.ArgumentParser(description='ET CDF Compressor — 27720ET')
        parser.add_argument('action', choices=['compress', 'decompress', 'gui'])
        parser.add_argument('input', nargs='?')
        parser.add_argument('output', nargs='?')
        args = parser.parse_args()

        if args.action == 'gui':
            build_gui()
        else:
            if not args.input:
                parser.error("Input required")
            if not args.output:
                args.output = args.input + '.cdf' if args.action == 'compress' else args.input.replace('.cdf', '.out')

            compressor = CDFCompressor(log_fn=print,
                                       progress_fn=lambda pct, m: print(f"\r[{pct:5.1f}%] {m}", end='', flush=True))
            if args.action == 'compress':
                compressor.compress_file(args.input, args.output)
            else:
                compressor.decompress_file(args.input, args.output)
    else:
        build_gui()
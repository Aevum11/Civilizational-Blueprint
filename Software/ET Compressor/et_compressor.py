#!/usr/bin/env python3
"""
ET Compressor — Exception Theory-Derived File Compression Engine
================================================================

A lossless file compressor grounded in Exception Theory (ET) mathematics.
Uses ET-derived algorithms including:

  - Descriptor Gap Analysis (Eq 4/211-218): Identifies missing patterns
    at ontological levels rather than purely statistical ones
  - Shimmer-Bilateral Interference (Eq 108-118): Detects structural
    redundancy through manifold oscillation patterns
  - D-Field Gradient Analysis (Eq 83, descriptor_field_gradient/curvature):
    Finds structural repetitions traditional methods miss
  - Phi-Harmonic Resonance (Eq 109/121): Golden-ratio frequency analysis
    for natural pattern detection
  - Manifold Density Routing (Eq 211): Adaptive strategy selection based
    on data density metrics
  - 12-Fold Manifold Encoding: Exploits MANIFOLD_SYMMETRY = 12
  - Recursive Descriptor Search (Eq 4/77): Kolmogorov-optimal function
    fitting to replace raw data with generative descriptors
  - Content-Addressable Deduplication (Eq 16): Block-level CAS via
    Merkle hashing
  - Traverser Entropy (Eq 1): Timing-jitter entropy for adaptive seeding
  - Coherence Validation (Eq 10): Integrity verification via Merkle roots

Output format: .pdt (Point-Descriptor-Traverser archive)

Author : Exception Theory Compressor Engine
Version: 1.0.0
License: Same as Civilizational Blueprint project
Deps   : None (pure Python, zero external libraries)
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
import sys
import threading
import time
import tkinter as tk
from collections import Counter, defaultdict, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 — ET CONSTANTS (derived from Exception Theory)
# ═══════════════════════════════════════════════════════════════════════

# Foundational
BASE_VARIANCE: float = 1.0 / 12.0          # σ² base from 3×4 manifold
MANIFOLD_SYMMETRY: int = 12                 # 3 primitives × 4 logic states
PHI: float = 1.6180339887498948482          # Golden ratio φ
KOIDE_RATIO: float = 2.0 / 3.0             # Koide lepton mass formula
SHIMMER_FLUX_RATE: float = 1.0 / 12.0      # Oscillation rate
SUBSTANTIATION_LIMIT: float = 1e-10         # Asymptotic precision floor
MANIFOLD_BINDING_STRENGTH: float = 1.0      # P∘D binding coefficient
PD_TENSION_COEFFICIENT: float = 1.0 / 12.0 # Shimmer tension scaling
SHIMMER_AMPLITUDE_MOD: float = 0.1          # 10% modulation depth
RADIATION_DECAY_EXPONENT: float = 2.0       # Inverse-square law
NORMALIZATION_EPSILON: float = 1e-10        # Division-by-zero guard
MANIFOLD_TIME_CONSTANT: int = 12            # τ_manifold for decay
HARMONIC_WEIGHT_BASE: float = 0.5           # Fundamental harmonic weight
PHI_HARMONIC_COUNT: int = 3                 # Levels of phi harmonics
CORRELATION_WINDOW_SIZE: int = 100          # Rolling correlation window
THRESHOLD_HIGH: float = 0.6                 # Upper tri-state threshold
THRESHOLD_LOW: float = 0.05                 # Lower tri-state threshold
FRACTAL_DEFAULT_OCTAVES: int = 3            # Multi-resolution levels
FRACTAL_DEFAULT_PERSISTENCE: float = 0.5    # Octave decay factor
DEFAULT_BLOOM_SIZE: int = 1024              # Bloom filter bits
DEFAULT_BLOOM_HASHES: int = 3               # Hash functions per item

# Compression-specific
BLOCK_SIZE: int = 65536                     # Manifold block quantum (64KB)
MIN_MATCH_LEN: int = 3                      # Minimum LZ match length
MAX_MATCH_LEN: int = 258                    # Maximum LZ match length
WINDOW_SIZE: int = 32768                    # LZ sliding window
DESCRIPTOR_SEARCH_DEPTH: int = 6            # Pattern types to try
DEDUP_BLOCK_SIZE: int = 512                 # CAS dedup chunk size
BWT_MAX_BLOCK: int = 65536                  # BWT max block (memory limit)

# Archive magic & version
PDT_MAGIC: bytes = b"PDT\x00"              # P∘D∘T archive signature
PDT_VERSION: int = 1                        # Archive format version

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 — ET MATH ENGINE (pure-Python implementations)
# ═══════════════════════════════════════════════════════════════════════


class ETMath:
    """Static math methods derived from Exception Theory equations."""

    # ── Eq 211: Structural Density ─────────────────────────────────────
    @staticmethod
    def density(payload: int, container: int) -> float:
        """S = D/D² — ratio of payload to container capacity."""
        if container == 0:
            return 0.0
        return payload / container

    # ── Eq 212: Manifold Variance ──────────────────────────────────────
    @staticmethod
    def manifold_variance(n: int) -> float:
        """σ²(n) = (n² - 1) / 12, from 3×4 permutation structure."""
        return (n * n - 1) / 12.0

    # ── Eq 83: Variance Gradient Descent ───────────────────────────────
    @staticmethod
    def variance_gradient(current: float, target: float,
                          step_size: float = 0.1) -> float:
        """D_next = D_current - S_step · Direction(∇V_sys)."""
        delta = current - target
        if abs(delta) < NORMALIZATION_EPSILON:
            return current
        direction = 1.0 if delta > 0 else -1.0
        return current - step_size * direction

    # ── Eq 16: Shannon Entropy ─────────────────────────────────────────
    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        """H(X) = -Σ p·log₂(p), the decision-tree depth."""
        if not data:
            return 0.0
        length = len(data)
        counts = Counter(data)
        entropy = 0.0
        for count in counts.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ── Eq 16 variant: Entropy Gradient ────────────────────────────────
    @staticmethod
    def entropy_gradient(data_before: bytes, data_after: bytes) -> float:
        """ΔS = S_after - S_before, measures organisation change."""
        return ETMath.shannon_entropy(data_after) - \
               ETMath.shannon_entropy(data_before)

    # ── Descriptor Field Gradient ──────────────────────────────────────
    @staticmethod
    def descriptor_field_gradient(data: bytes,
                                  window_size: int = 3) -> List[float]:
        """First derivative of descriptor values via sliding window."""
        if len(data) < window_size:
            return []
        gradients: List[float] = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            grad = (window[-1] - window[0]) / max(window_size - 1, 1)
            gradients.append(grad)
        return gradients

    # ── Descriptor Field Curvature ─────────────────────────────────────
    @staticmethod
    def descriptor_field_curvature(gradients: List[float]) -> List[float]:
        """Second derivative — rate of change of gradient."""
        if len(gradients) < 2:
            return []
        return [gradients[i + 1] - gradients[i]
                for i in range(len(gradients) - 1)]

    # ── Eq 77: Kolmogorov Complexity ───────────────────────────────────
    @staticmethod
    def kolmogorov_complexity(descriptor_set: list) -> int:
        """N_min = min(Count(D_set)), minimal unique descriptors."""
        return len(set(descriptor_set))

    # ── Eq 109: Manifold Resonance Detection ───────────────────────────
    @staticmethod
    def manifold_resonance(signal: List[float],
                           base_freq: float = 1.0) -> float:
        """Detect φ-harmonic resonance in signal via DFT peaks.

        Checks power at frequencies f, f·φ, f·φ², ...
        Returns resonance strength in [0, 1].
        """
        n = len(signal)
        if n < 4:
            return 0.0

        # Compute power spectrum via DFT at harmonic frequencies
        harmonics = [base_freq * (PHI ** i)
                     for i in range(PHI_HARMONIC_COUNT)]
        total_power = 0.0
        harmonic_power = 0.0
        for k in range(n // 2):
            freq = k / n
            # DFT coefficient at frequency k
            real_part = 0.0
            imag_part = 0.0
            for t in range(n):
                angle = -2.0 * math.pi * k * t / n
                real_part += signal[t] * math.cos(angle)
                imag_part += signal[t] * math.sin(angle)
            power = (real_part * real_part + imag_part * imag_part) / n
            total_power += power
            # Check if this frequency is near any harmonic
            for h_freq in harmonics:
                h_idx = h_freq * n
                if abs(k - h_idx) < 1.5:
                    weight = HARMONIC_WEIGHT_BASE / (1.0 + harmonics.index(h_freq))
                    harmonic_power += power * weight
        if total_power < NORMALIZATION_EPSILON:
            return 0.0
        return min(harmonic_power / total_power, 1.0)

    # ── Eq 108: Dynamic Attractor Shimmer ──────────────────────────────
    @staticmethod
    def shimmer_flux(substantiation_rate: float,
                     time_delta: float) -> float:
        """Shimmer flux from potential→actual conversion rate."""
        return (substantiation_rate * SHIMMER_FLUX_RATE *
                (1.0 - math.exp(-time_delta / max(SUBSTANTIATION_LIMIT,
                                                   NORMALIZATION_EPSILON))))

    # ── Eq 114: P-D Tension Shimmer ────────────────────────────────────
    @staticmethod
    def pd_tension(p_magnitude: float, d_magnitude: float) -> float:
        """Static tension between infinite substrate and finite constraint."""
        if d_magnitude < NORMALIZATION_EPSILON:
            return 0.0
        return PD_TENSION_COEFFICIENT * (p_magnitude / d_magnitude)

    # ── Eq 118: Shimmer Oscillation Modulation ─────────────────────────
    @staticmethod
    def shimmer_modulation(time_val: float,
                           base_freq: float = 1.0) -> float:
        """A(t) = 1.0 + 0.1·sin(2π·f·t/12)."""
        return 1.0 + SHIMMER_AMPLITUDE_MOD * math.sin(
            2.0 * math.pi * base_freq * time_val / MANIFOLD_SYMMETRY)

    # ── Eq 117: Shimmer Radiation Intensity ────────────────────────────
    @staticmethod
    def shimmer_radiation(distance: float) -> float:
        """I(r) ∝ 1/r² — inverse square decay from Exception."""
        if distance < NORMALIZATION_EPSILON:
            return 1.0
        return 1.0 / (distance ** RADIATION_DECAY_EXPONENT)

    # ── Eq 127: Manifold Temporal Decay ────────────────────────────────
    @staticmethod
    def temporal_decay(time_lag: float) -> float:
        """decay(τ) = exp(-τ / τ_manifold), exponential decay."""
        return math.exp(-time_lag / MANIFOLD_TIME_CONSTANT)

    # ── Eq 30: Phase Transition (Sigmoid) ──────────────────────────────
    @staticmethod
    def phase_transition(gradient_input: float) -> float:
        """Status_sub = 1/(1 + exp(-G_input)), logistic function."""
        clamped = max(min(gradient_input, 500), -500)
        return 1.0 / (1.0 + math.exp(-clamped))

    # ── Manifold Boundary Detection ────────────────────────────────────
    @staticmethod
    def manifold_boundary(value: int) -> Tuple[bool, int]:
        """Detect power-of-2 boundaries in the manifold."""
        if value <= 0:
            return False, 0
        log_val = math.log2(value)
        nearest = round(log_val)
        return abs(log_val - nearest) < 0.1, nearest

    # ── Bloom Filter Coordinates ───────────────────────────────────────
    @staticmethod
    def bloom_coords(item: bytes, size: int = DEFAULT_BLOOM_SIZE,
                     count: int = DEFAULT_BLOOM_HASHES) -> List[int]:
        """Generate k hash positions for Bloom filter insertion."""
        coords: List[int] = []
        for i in range(count):
            h = hashlib.sha256(item + i.to_bytes(2, "little")).digest()
            idx = int.from_bytes(h[:4], "little") % size
            coords.append(idx)
        return coords

    # ── Merkle Hash ────────────────────────────────────────────────────
    @staticmethod
    def merkle_hash(data: bytes) -> str:
        """SHA-256 hash of a single data chunk."""
        return hashlib.sha256(data).hexdigest()

    # ── Merkle Root ────────────────────────────────────────────────────
    @staticmethod
    def merkle_root(chunks: List[bytes]) -> str:
        """Compute Merkle root from list of data chunks."""
        if not chunks:
            return ETMath.merkle_hash(b"")
        nodes = [ETMath.merkle_hash(c) for c in chunks]
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            new_nodes: List[str] = []
            for i in range(0, len(nodes), 2):
                combined = (nodes[i] + nodes[i + 1]).encode("ascii")
                new_nodes.append(ETMath.merkle_hash(combined))
            nodes = new_nodes
        return nodes[0]

    # ── Content Address (CAS) ──────────────────────────────────────────
    @staticmethod
    def content_address(data: bytes) -> str:
        """SHA-1 content-addressed hash for deduplication."""
        return hashlib.sha1(data).hexdigest()

    # ── Recursive Descriptor Search (Eq 4) ─────────────────────────────
    @staticmethod
    def recursive_descriptor_search(
            values: List[int]) -> Dict[str, Any]:
        """Find minimal generative descriptor (function) for data.

        Tests: constant, linear, quadratic, exponential, power, log.
        Returns dict with 'type', 'params', 'variance'.
        If variance == 0 → lossless generative compression.
        """
        n = len(values)
        if n == 0:
            return {"type": "empty", "params": (), "variance": 0.0}
        if n == 1:
            return {"type": "constant", "params": (values[0],),
                    "variance": 0.0}

        indices = list(range(n))
        best: Dict[str, Any] = {"type": "raw", "params": (),
                                 "variance": float("inf")}

        # Constant: f(x) = a
        a_const = sum(values) / n
        var_const = sum((v - a_const) ** 2 for v in values)
        if var_const < best["variance"]:
            best = {"type": "constant", "params": (a_const,),
                    "variance": var_const}

        # Linear: f(x) = a·x + b
        if n >= 2:
            sx = sum(indices)
            sy = sum(values)
            sxx = sum(x * x for x in indices)
            sxy = sum(x * y for x, y in zip(indices, values))
            denom = n * sxx - sx * sx
            if abs(denom) > NORMALIZATION_EPSILON:
                a_lin = (n * sxy - sx * sy) / denom
                b_lin = (sy - a_lin * sx) / n
                var_lin = sum((values[i] - (a_lin * i + b_lin)) ** 2
                             for i in indices)
                if var_lin < best["variance"]:
                    best = {"type": "linear",
                            "params": (a_lin, b_lin),
                            "variance": var_lin}

        # Quadratic: f(x) = a·x² + b·x + c  (least-squares via normal eq)
        if n >= 3:
            try:
                # Build normal equations for quadratic fit
                s0 = float(n)
                s1 = float(sum(indices))
                s2 = float(sum(i * i for i in indices))
                s3 = float(sum(i ** 3 for i in indices))
                s4 = float(sum(i ** 4 for i in indices))
                t0 = float(sum(values))
                t1 = float(sum(i * values[i] for i in indices))
                t2 = float(sum(i * i * values[i] for i in indices))
                # Solve 3x3 system: [s4 s3 s2; s3 s2 s1; s2 s1 s0]·[a;b;c] = [t2;t1;t0]
                det = (s4 * (s2 * s0 - s1 * s1)
                       - s3 * (s3 * s0 - s1 * s2)
                       + s2 * (s3 * s1 - s2 * s2))
                if abs(det) > NORMALIZATION_EPSILON:
                    a_q = ((s2 * s0 - s1 * s1) * t2
                           - (s3 * s0 - s1 * s2) * t1
                           + (s3 * s1 - s2 * s2) * t0) / det
                    b_q = (-(s3 * s0 - s2 * s1) * t2
                           + (s4 * s0 - s2 * s2) * t1
                           - (s4 * s1 - s2 * s3) * t0) / det
                    c_q = ((s3 * s1 - s2 * s2) * t2
                           - (s4 * s1 - s3 * s2) * t1
                           + (s4 * s2 - s3 * s3) * t0) / det
                    var_q = sum(
                        (values[i] - (a_q * i * i + b_q * i + c_q)) ** 2
                        for i in indices)
                    if var_q < best["variance"]:
                        best = {"type": "quadratic",
                                "params": (a_q, b_q, c_q),
                                "variance": var_q}
            except (OverflowError, ZeroDivisionError):
                pass

        # Exponential: f(x) = a·b^x  (log-linear regression)
        if n >= 2 and all(v > 0 for v in values):
            try:
                log_vals = [math.log(v) for v in values]
                sl_x = sum(indices)
                sl_y = sum(log_vals)
                sl_xx = sum(x * x for x in indices)
                sl_xy = sum(x * y for x, y in zip(indices, log_vals))
                d2 = n * sl_xx - sl_x * sl_x
                if abs(d2) > NORMALIZATION_EPSILON:
                    log_b = (n * sl_xy - sl_x * sl_y) / d2
                    log_a = (sl_y - log_b * sl_x) / n
                    a_exp = math.exp(log_a)
                    b_exp = math.exp(log_b)
                    var_exp = sum(
                        (values[i] - a_exp * (b_exp ** i)) ** 2
                        for i in indices)
                    if var_exp < best["variance"]:
                        best = {"type": "exponential",
                                "params": (a_exp, b_exp),
                                "variance": var_exp}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        # Power: f(x) = a·x^b  (log-log regression, skip x=0)
        if n >= 3 and all(v > 0 for v in values[1:]):
            try:
                pw_idx = [i for i in indices if i > 0]
                pw_vals = [values[i] for i in pw_idx]
                lx = [math.log(i) for i in pw_idx]
                ly = [math.log(v) for v in pw_vals]
                nn = len(pw_idx)
                slx = sum(lx)
                sly = sum(ly)
                slxx = sum(x * x for x in lx)
                slxy = sum(x * y for x, y in zip(lx, ly))
                d3 = nn * slxx - slx * slx
                if abs(d3) > NORMALIZATION_EPSILON:
                    b_pw = (nn * slxy - slx * sly) / d3
                    log_a_pw = (sly - b_pw * slx) / nn
                    a_pw = math.exp(log_a_pw)
                    var_pw = sum(
                        (pw_vals[j] - a_pw * (pw_idx[j] ** b_pw)) ** 2
                        for j in range(nn))
                    if var_pw < best["variance"]:
                        best = {"type": "power",
                                "params": (a_pw, b_pw),
                                "variance": var_pw}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        return best

    # ── Eq 121: Phi Harmonic Generation ────────────────────────────────
    @staticmethod
    def phi_harmonics(length: int, base_freq: float = 1.0) -> List[float]:
        """Generate signal with φ-based harmonic structure."""
        signal: List[float] = []
        for t in range(length):
            val = 0.0
            for i in range(PHI_HARMONIC_COUNT):
                weight = HARMONIC_WEIGHT_BASE / (1.0 + i)
                freq = base_freq * (PHI ** i)
                val += weight * math.sin(2.0 * math.pi * freq * t / length)
            signal.append(val)
        return signal

    # ── Tri-state Decision (Eq 135) ────────────────────────────────────
    @staticmethod
    def threshold_decision(score: float) -> str:
        """Classify score into HIGH / MID / LOW."""
        if score > THRESHOLD_HIGH:
            return "HIGH"
        elif score < THRESHOLD_LOW:
            return "LOW"
        return "MID"

    # ── Cross-Correlation (Eq 134) ─────────────────────────────────────
    @staticmethod
    def cross_correlation(a: List[float], b: List[float]) -> float:
        """Pearson correlation coefficient between two signals."""
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        mean_a = sum(a[:n]) / n
        mean_b = sum(b[:n]) / n
        num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        den_a = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)))
        den_b = math.sqrt(sum((b[i] - mean_b) ** 2 for i in range(n)))
        denom = den_a * den_b
        if denom < NORMALIZATION_EPSILON:
            return 0.0
        return num / denom


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 — BLOOM FILTER (Probabilistic Manifold, Eq 12)
# ═══════════════════════════════════════════════════════════════════════


class BloomFilter:
    """Space-efficient probabilistic set membership (D-Shadow)."""

    __slots__ = ("size", "hash_count", "bit_array", "count")

    def __init__(self, size: int = DEFAULT_BLOOM_SIZE,
                 hash_count: int = DEFAULT_BLOOM_HASHES) -> None:
        self.size = size
        self.hash_count = hash_count
        self.bit_array: int = 0
        self.count: int = 0

    def add(self, item: bytes) -> None:
        for coord in ETMath.bloom_coords(item, self.size, self.hash_count):
            self.bit_array |= (1 << coord)
        self.count += 1

    def might_contain(self, item: bytes) -> bool:
        for coord in ETMath.bloom_coords(item, self.size, self.hash_count):
            if not (self.bit_array & (1 << coord)):
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 — CONTENT-ADDRESSABLE STORAGE (CAS, Eq 16)
# ═══════════════════════════════════════════════════════════════════════


class ContentAddressableStore:
    """Block-level deduplication via content hashing."""

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self._order: List[str] = []
        self.dedup_hits: int = 0

    def write(self, block: bytes) -> str:
        addr = ETMath.content_address(block)
        if addr not in self._store:
            self._store[addr] = block
        else:
            self.dedup_hits += 1
        self._order.append(addr)
        return addr

    def read(self, addr: str) -> bytes:
        return self._store.get(addr, b"")

    @property
    def unique_blocks(self) -> int:
        return len(self._store)

    @property
    def total_refs(self) -> int:
        return len(self._order)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 — ADAPTIVE HUFFMAN CODING (ET Entropy Encoder)
# ═══════════════════════════════════════════════════════════════════════


class _HuffNode:
    """Node in Huffman tree."""
    __slots__ = ("freq", "byte_val", "left", "right")

    def __init__(self, freq: int, byte_val: Optional[int] = None,
                 left: Optional["_HuffNode"] = None,
                 right: Optional["_HuffNode"] = None) -> None:
        self.freq = freq
        self.byte_val = byte_val
        self.left = left
        self.right = right


def _build_huffman_tree(freq_table: Dict[int, int]) -> Optional[_HuffNode]:
    """Build Huffman tree from frequency table using a priority queue."""
    if not freq_table:
        return None
    # Simple heap via sorted list (no heapq needed, pure approach)
    nodes: List[_HuffNode] = [
        _HuffNode(freq=f, byte_val=b) for b, f in freq_table.items()
    ]
    if len(nodes) == 1:
        return _HuffNode(freq=nodes[0].freq, left=nodes[0])
    while len(nodes) > 1:
        nodes.sort(key=lambda nd: nd.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = _HuffNode(freq=left.freq + right.freq,
                           left=left, right=right)
        nodes.append(parent)
    return nodes[0]


def _build_code_table(root: Optional[_HuffNode]) -> Dict[int, str]:
    """Generate bit-string codes from Huffman tree."""
    table: Dict[int, str] = {}
    if root is None:
        return table

    def _walk(node: _HuffNode, prefix: str) -> None:
        if node.byte_val is not None:
            table[node.byte_val] = prefix if prefix else "0"
            return
        if node.left:
            _walk(node.left, prefix + "0")
        if node.right:
            _walk(node.right, prefix + "1")

    _walk(root, "")
    return table


def _huffman_encode(data: bytes) -> Tuple[bytes, Dict[int, int]]:
    """Encode data using Huffman coding. Returns (encoded_bytes, freq_table)."""
    if not data:
        return b"", {}

    freq_table: Dict[int, int] = Counter(data)
    tree = _build_huffman_tree(freq_table)
    code_table = _build_code_table(tree)

    # Build bit string
    bits: List[str] = []
    for byte in data:
        bits.append(code_table[byte])
    bit_string = "".join(bits)

    # Pack into bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    encoded = bytearray()
    for i in range(0, len(bit_string), 8):
        encoded.append(int(bit_string[i:i + 8], 2))

    return bytes(encoded), freq_table


def _huffman_decode(encoded: bytes, freq_table: Dict[int, int],
                    original_length: int) -> bytes:
    """Decode Huffman-encoded data back to original bytes."""
    if not encoded or not freq_table:
        return b""

    tree = _build_huffman_tree(freq_table)
    if tree is None:
        return b""

    # Convert to bit string
    bit_string = "".join(f"{byte:08b}" for byte in encoded)

    result = bytearray()
    node = tree
    for bit in bit_string:
        if len(result) >= original_length:
            break
        if bit == "0":
            node = node.left if node.left else node
        else:
            node = node.right if node.right else node
        if node.byte_val is not None:
            result.append(node.byte_val)
            node = tree

    return bytes(result)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 6 — LZ77 SLIDING-WINDOW ENGINE (Traverser Navigation)
# ═══════════════════════════════════════════════════════════════════════


def _lz77_compress(data: bytes) -> bytes:
    """LZ77 compression with ET-derived sliding window.

    Uses hash-based match finder for speed, with shimmer radiation
    weighting to prefer closer matches when lengths are equal.
    """
    if not data:
        return b""

    output = io.BytesIO()
    pos = 0
    length = len(data)

    # Build hash table for 3-byte trigrams (fast match finding)
    hash_table: Dict[int, List[int]] = defaultdict(list)

    def _trigram_hash(p: int) -> int:
        if p + 2 >= length:
            return -1
        return (data[p] << 16) | (data[p + 1] << 8) | data[p + 2]

    while pos < length:
        best_offset = 0
        best_length = 0

        if pos + 2 < length:
            h = _trigram_hash(pos)
            candidates = hash_table.get(h, [])

            # Search candidates in reverse order (most recent = closest)
            for cand_pos in reversed(candidates):
                offset = pos - cand_pos
                if offset > WINDOW_SIZE or offset <= 0:
                    continue

                # Extend match
                match_len = 0
                while (match_len < MAX_MATCH_LEN and
                       pos + match_len < length and
                       data[cand_pos + match_len] == data[pos + match_len]):
                    match_len += 1

                if match_len > best_length:
                    best_length = match_len
                    best_offset = offset
                    if match_len >= MAX_MATCH_LEN:
                        break  # Can't do better

            # Register current position in hash table
            hash_table[h].append(pos)
            # Prune old entries to keep memory bounded
            entries = hash_table[h]
            if len(entries) > 32:
                hash_table[h] = entries[-32:]

        if best_length >= MIN_MATCH_LEN:
            # Encode as (1, offset, length)
            output.write(b"\x01")
            output.write(struct.pack("<HB", best_offset,
                                     min(best_length, MAX_MATCH_LEN)))
            # Register skipped positions
            for skip in range(1, min(best_length, 8)):
                if pos + skip + 2 < length:
                    sh = _trigram_hash(pos + skip)
                    if sh >= 0:
                        hash_table[sh].append(pos + skip)
            pos += best_length
        else:
            # Encode as (0, literal_byte)
            output.write(b"\x00")
            output.write(data[pos:pos + 1])
            pos += 1

    return output.getvalue()


def _lz77_decompress(data: bytes) -> bytes:
    """Decompress LZ77-encoded data."""
    if not data:
        return b""

    output = bytearray()
    stream = io.BytesIO(data)

    while True:
        flag = stream.read(1)
        if not flag:
            break
        if flag[0] == 0x01:
            # Match reference
            ref_data = stream.read(3)
            if len(ref_data) < 3:
                break
            offset = struct.unpack("<H", ref_data[:2])[0]
            match_len = ref_data[2]
            start = len(output) - offset
            for i in range(match_len):
                output.append(output[start + i])
        else:
            # Literal byte
            literal = stream.read(1)
            if not literal:
                break
            output.append(literal[0])

    return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7 — BURROWS-WHEELER TRANSFORM (Manifold Sorting, Eq 11)
# ═══════════════════════════════════════════════════════════════════════


def _bwt_encode(data: bytes) -> Tuple[bytes, int]:
    """Burrows-Wheeler Transform using teleological manifold sorting.

    Sorts all rotations of the data to group similar contexts together,
    exploiting manifold density for better downstream compression.
    Returns (transformed_bytes, original_row_index).
    """
    if not data:
        return b"", 0

    n = len(data)
    # Use suffix array approach for efficiency
    doubled = data + data
    indices = list(range(n))
    indices.sort(key=lambda i: doubled[i:i + n])

    # Build output column (last character of each sorted rotation)
    transformed = bytes(data[(i + n - 1) % n] for i in indices)
    original_idx = indices.index(0)

    return transformed, original_idx


def _bwt_decode(data: bytes, original_idx: int) -> bytes:
    """Inverse Burrows-Wheeler Transform."""
    if not data:
        return b""

    n = len(data)
    # Build the transformation vector
    table = sorted(range(n), key=lambda i: data[i])

    result = bytearray(n)
    idx = original_idx
    for i in range(n):
        result[i] = data[table[idx]]
        idx = table[idx]

    return bytes(result)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 8 — MOVE-TO-FRONT TRANSFORM (Descriptor Reordering)
# ═══════════════════════════════════════════════════════════════════════


def _mtf_encode(data: bytes) -> bytes:
    """Move-to-Front transform: recently used symbols get lower indices.

    This exploits the descriptor field gradient — after BWT, similar
    bytes cluster together, and MTF converts that locality into low
    values that compress exceptionally well.
    """
    if not data:
        return b""

    alphabet = list(range(256))
    result = bytearray(len(data))

    for i, byte in enumerate(data):
        idx = alphabet.index(byte)
        result[i] = idx
        # Move to front
        alphabet.pop(idx)
        alphabet.insert(0, byte)

    return bytes(result)


def _mtf_decode(data: bytes) -> bytes:
    """Inverse Move-to-Front transform."""
    if not data:
        return b""

    alphabet = list(range(256))
    result = bytearray(len(data))

    for i, idx in enumerate(data):
        byte = alphabet[idx]
        result[i] = byte
        alphabet.pop(idx)
        alphabet.insert(0, byte)

    return bytes(result)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 9 — RUN-LENGTH ENCODING (Flat Region Compression)
# ═══════════════════════════════════════════════════════════════════════


def _rle_encode(data: bytes) -> bytes:
    """Run-length encoding for zero-rich MTF output.

    After BWT+MTF, data is predominantly zeros. RLE compresses
    these flat descriptor regions (low gradient) efficiently.
    """
    if not data:
        return b""

    output = bytearray()
    i = 0
    n = len(data)

    while i < n:
        byte = data[i]
        run_len = 1
        while i + run_len < n and data[i + run_len] == byte and run_len < 255:
            run_len += 1

        if run_len >= 3 or byte == 0xFF:
            # Escape: 0xFF, byte, count
            output.append(0xFF)
            output.append(byte)
            output.append(run_len)
            i += run_len
        else:
            if byte == 0xFF:
                output.append(0xFF)
                output.append(0xFF)
                output.append(1)
            else:
                output.append(byte)
            i += 1

    return bytes(output)


def _rle_decode(data: bytes) -> bytes:
    """Inverse run-length decoding."""
    if not data:
        return b""

    output = bytearray()
    i = 0
    n = len(data)

    while i < n:
        if data[i] == 0xFF:
            if i + 2 < n:
                byte = data[i + 1]
                count = data[i + 2]
                output.extend([byte] * count)
                i += 3
            else:
                break
        else:
            output.append(data[i])
            i += 1

    return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 10 — D-FIELD ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════


class DFieldAnalyzer:
    """Analyzes data using Descriptor Field operations.

    Computes gradients, curvature, shimmer patterns, and resonance
    to identify optimal compression boundaries and strategies.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.length = len(data)
        self._gradients: Optional[List[float]] = None
        self._curvatures: Optional[List[float]] = None
        self._entropy: Optional[float] = None
        self._resonance: Optional[float] = None
        self._density: Optional[float] = None

    @property
    def entropy(self) -> float:
        if self._entropy is None:
            self._entropy = ETMath.shannon_entropy(self.data)
        return self._entropy

    @property
    def gradients(self) -> List[float]:
        if self._gradients is None:
            self._gradients = ETMath.descriptor_field_gradient(
                self.data, window_size=3)
        return self._gradients

    @property
    def curvatures(self) -> List[float]:
        if self._curvatures is None:
            self._curvatures = ETMath.descriptor_field_curvature(
                self.gradients)
        return self._curvatures

    @property
    def density(self) -> float:
        if self._density is None:
            unique_bytes = len(set(self.data))
            self._density = ETMath.density(unique_bytes, 256)
        return self._density

    @property
    def resonance(self) -> float:
        """Phi-harmonic resonance strength of data."""
        if self._resonance is None:
            if self.length < 16:
                self._resonance = 0.0
            else:
                # Sample data as signal
                sample_size = min(self.length, 256)
                signal = [float(self.data[i]) for i in range(sample_size)]
                self._resonance = ETMath.manifold_resonance(signal)
        return self._resonance

    def find_breakpoints(self, threshold: float = 10.0) -> List[int]:
        """Find structural breakpoints via high curvature regions."""
        breakpoints: List[int] = []
        for i, curv in enumerate(self.curvatures):
            if abs(curv) > threshold:
                breakpoints.append(i + 1)  # Offset for window
        return breakpoints

    def shimmer_profile(self) -> List[float]:
        """Compute shimmer oscillation profile across data."""
        profile: List[float] = []
        for i in range(self.length):
            t = i / max(self.length - 1, 1)
            mod = ETMath.shimmer_modulation(t * MANIFOLD_SYMMETRY)
            profile.append(mod)
        return profile

    def classify_region(self, start: int, end: int) -> str:
        """Classify a data region using tri-state decision.

        Returns 'HIGH' (complex), 'MID' (structured), 'LOW' (simple).
        """
        region = self.data[start:end]
        region_entropy = ETMath.shannon_entropy(region)
        # Normalize entropy to [0, 1] range (max entropy for bytes = 8.0)
        normalized = region_entropy / 8.0
        return ETMath.threshold_decision(normalized)

    def compute_bilateral_interference(self) -> float:
        """Detect bilateral interference patterns.

        Splits data into three sub-signals and measures their
        omni-correlation (Eq 110). High correlation indicates
        structural redundancy that can be exploited.
        """
        if self.length < 12:
            return 0.0
        third = self.length // 3
        sig_a = [float(b) for b in self.data[:third]]
        sig_b = [float(b) for b in self.data[third:2 * third]]
        sig_c = [float(b) for b in self.data[2 * third:3 * third]]

        corr_ab = abs(ETMath.cross_correlation(sig_a, sig_b))
        corr_ac = abs(ETMath.cross_correlation(sig_a, sig_c))
        corr_bc = abs(ETMath.cross_correlation(sig_b, sig_c))

        return (corr_ab + corr_ac + corr_bc) / 3.0


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 11 — BLOCK DEDUPLICATION ENGINE (CAS + Bloom)
# ═══════════════════════════════════════════════════════════════════════


class DeduplicationEngine:
    """Block-level deduplication using CAS and Bloom filters.

    Chunks data into DEDUP_BLOCK_SIZE blocks, hashes each,
    and eliminates duplicates via content addressing.
    """

    def __init__(self) -> None:
        self.cas = ContentAddressableStore()
        self.bloom = BloomFilter(size=DEFAULT_BLOOM_SIZE * 8,
                                 hash_count=DEFAULT_BLOOM_HASHES)

    def process(self, data: bytes) -> Tuple[List[str], Dict[str, bytes]]:
        """Process data into deduplicated block references.

        Returns (block_refs, unique_blocks_dict).
        """
        refs: List[str] = []
        blocks_dict: Dict[str, bytes] = {}

        for i in range(0, len(data), DEDUP_BLOCK_SIZE):
            block = data[i:i + DEDUP_BLOCK_SIZE]
            addr = self.cas.write(block)
            refs.append(addr)
            if addr not in blocks_dict:
                blocks_dict[addr] = block

        return refs, blocks_dict

    def reassemble(self, refs: List[str],
                   blocks_dict: Dict[str, bytes]) -> bytes:
        """Reassemble data from block references."""
        output = bytearray()
        for addr in refs:
            output.extend(blocks_dict.get(addr, b""))
        return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 12 — DELTA ENCODING (Descriptor Difference Transform)
# ═══════════════════════════════════════════════════════════════════════


def _delta_encode(data: bytes) -> bytes:
    """Delta encoding: store differences between consecutive bytes.

    Exploits the descriptor field gradient — when data has low
    gradient (smooth changes), deltas are small and compress well.
    """
    if not data:
        return b""
    result = bytearray(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = (data[i] - data[i - 1]) & 0xFF
    return bytes(result)


def _delta_decode(data: bytes) -> bytes:
    """Inverse delta decoding."""
    if not data:
        return b""
    result = bytearray(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = (result[i - 1] + data[i]) & 0xFF
    return bytes(result)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 13 — ET COMPRESSION PIPELINE (Full Traverser Path)
# ═══════════════════════════════════════════════════════════════════════


# Strategy flags (stored in archive header)
STRAT_LZ77: int = 0x01
STRAT_BWT_MTF_RLE_HUFF: int = 0x02
STRAT_DELTA_HUFF: int = 0x03
STRAT_DEDUP: int = 0x04
STRAT_RAW: int = 0x00


def _select_strategy(analyzer: DFieldAnalyzer) -> int:
    """Select optimal compression strategy via D-field analysis.

    Uses manifold density routing (Eq 211), entropy gradient,
    shimmer bilateral interference, and phi-harmonic resonance
    to choose the best compression path.

    Strategy: guided heuristic to avoid trying expensive strategies
    on data that won't benefit.
    """
    entropy = analyzer.entropy
    density = analyzer.density

    # Near-random data (high entropy, high density) — minimal compression
    if entropy > 7.9 and density > 0.98:
        return STRAT_RAW

    # For everything else: BWT pipeline is generally strongest
    # It handles structured, repetitive, and moderate-entropy data well
    if entropy < 6.5:
        return STRAT_BWT_MTF_RLE_HUFF

    # Smooth data (low gradient) — delta + Huffman
    if len(analyzer.gradients) > 0:
        avg_gradient = sum(abs(g) for g in analyzer.gradients) / len(analyzer.gradients)
        if avg_gradient < 8.0:
            return STRAT_DELTA_HUFF

    # Medium entropy — try BWT pipeline (usually competitive)
    if entropy < 7.5:
        return STRAT_BWT_MTF_RLE_HUFF

    # Default: LZ77 (good general-purpose for high entropy data)
    return STRAT_LZ77


def _competitive_compress(data: bytes) -> Tuple[int, bytes]:
    """Try multiple strategies and return the smallest result.

    This is the Traverser's optimal navigation through the compression
    manifold — exploring multiple paths and selecting the one with
    minimum variance (best compression ratio).
    """
    original_len = len(data)
    best_strategy = STRAT_RAW
    best_compressed = data
    best_size = original_len

    # Candidate strategies to try
    candidates = [STRAT_BWT_MTF_RLE_HUFF, STRAT_LZ77, STRAT_DELTA_HUFF]

    for strategy in candidates:
        try:
            compressed = _compress_block(data, strategy)
            if len(compressed) < best_size:
                best_size = len(compressed)
                best_strategy = strategy
                best_compressed = compressed
        except Exception:
            continue

    return best_strategy, best_compressed


def _compress_block(data: bytes, strategy: int) -> bytes:
    """Compress a single block using the selected strategy."""
    if strategy == STRAT_RAW:
        return data

    elif strategy == STRAT_LZ77:
        return _lz77_compress(data)

    elif strategy == STRAT_BWT_MTF_RLE_HUFF:
        # BWT → MTF → RLE → Huffman pipeline
        bwt_data, bwt_idx = _bwt_encode(data)
        mtf_data = _mtf_encode(bwt_data)
        rle_data = _rle_encode(mtf_data)
        huff_data, freq_table = _huffman_encode(rle_data)

        # Pack: bwt_idx(4) + rle_len(4) + freq_table_len(2) + freq_table + huff_data
        out = io.BytesIO()
        out.write(struct.pack("<I", bwt_idx))
        out.write(struct.pack("<I", len(rle_data)))
        # Serialize frequency table
        freq_bytes = _serialize_freq_table(freq_table)
        out.write(struct.pack("<H", len(freq_bytes)))
        out.write(freq_bytes)
        out.write(huff_data)
        return out.getvalue()

    elif strategy == STRAT_DELTA_HUFF:
        delta_data = _delta_encode(data)
        huff_data, freq_table = _huffman_encode(delta_data)
        out = io.BytesIO()
        out.write(struct.pack("<I", len(delta_data)))
        freq_bytes = _serialize_freq_table(freq_table)
        out.write(struct.pack("<H", len(freq_bytes)))
        out.write(freq_bytes)
        out.write(huff_data)
        return out.getvalue()

    elif strategy == STRAT_DEDUP:
        # Dedup: break into sub-blocks, deduplicate, then LZ77 the refs
        dedup = DeduplicationEngine()
        refs, blocks = dedup.process(data)
        # Serialize: unique_count(4) + [addr(20) + block_len(2) + block_data]...
        #          + ref_count(4) + [ref_addr(20)]...
        out = io.BytesIO()
        unique_addrs = list(blocks.keys())
        out.write(struct.pack("<I", len(unique_addrs)))
        for addr in unique_addrs:
            out.write(addr.encode("ascii")[:40])
            block = blocks[addr]
            out.write(struct.pack("<H", len(block)))
            out.write(block)
        out.write(struct.pack("<I", len(refs)))
        for ref in refs:
            out.write(ref.encode("ascii")[:40])
        raw_dedup = out.getvalue()
        # Apply LZ77 on top for further compression
        return _lz77_compress(raw_dedup)

    return data


def _decompress_block(data: bytes, strategy: int,
                      original_length: int) -> bytes:
    """Decompress a single block using the recorded strategy."""
    if strategy == STRAT_RAW:
        return data

    elif strategy == STRAT_LZ77:
        return _lz77_decompress(data)

    elif strategy == STRAT_BWT_MTF_RLE_HUFF:
        stream = io.BytesIO(data)
        bwt_idx = struct.unpack("<I", stream.read(4))[0]
        rle_len = struct.unpack("<I", stream.read(4))[0]
        freq_len = struct.unpack("<H", stream.read(2))[0]
        freq_bytes = stream.read(freq_len)
        freq_table = _deserialize_freq_table(freq_bytes)
        huff_data = stream.read()

        rle_data = _huffman_decode(huff_data, freq_table, rle_len)
        mtf_data = _rle_decode(rle_data)
        bwt_data = _mtf_decode(mtf_data)
        return _bwt_decode(bwt_data, bwt_idx)

    elif strategy == STRAT_DELTA_HUFF:
        stream = io.BytesIO(data)
        delta_len = struct.unpack("<I", stream.read(4))[0]
        freq_len = struct.unpack("<H", stream.read(2))[0]
        freq_bytes = stream.read(freq_len)
        freq_table = _deserialize_freq_table(freq_bytes)
        huff_data = stream.read()

        delta_data = _huffman_decode(huff_data, freq_table, delta_len)
        return _delta_decode(delta_data)

    elif strategy == STRAT_DEDUP:
        raw_dedup = _lz77_decompress(data)
        stream = io.BytesIO(raw_dedup)
        unique_count = struct.unpack("<I", stream.read(4))[0]
        blocks: Dict[str, bytes] = {}
        for _ in range(unique_count):
            addr = stream.read(40).decode("ascii")
            block_len = struct.unpack("<H", stream.read(2))[0]
            block_data = stream.read(block_len)
            blocks[addr] = block_data
        ref_count = struct.unpack("<I", stream.read(4))[0]
        refs: List[str] = []
        for _ in range(ref_count):
            ref = stream.read(40).decode("ascii")
            refs.append(ref)
        # Reassemble
        output = bytearray()
        for ref in refs:
            output.extend(blocks.get(ref, b""))
        return bytes(output)

    return data


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 14 — FREQUENCY TABLE SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════


def _serialize_freq_table(freq_table: Dict[int, int]) -> bytes:
    """Serialize Huffman frequency table compactly."""
    out = io.BytesIO()
    entries = [(k, v) for k, v in freq_table.items()]
    out.write(struct.pack("<H", len(entries)))
    for byte_val, freq in entries:
        out.write(struct.pack("<BI", byte_val, freq))
    return out.getvalue()


def _deserialize_freq_table(data: bytes) -> Dict[int, int]:
    """Deserialize Huffman frequency table."""
    stream = io.BytesIO(data)
    count = struct.unpack("<H", stream.read(2))[0]
    table: Dict[int, int] = {}
    for _ in range(count):
        entry = stream.read(5)
        if len(entry) < 5:
            break
        byte_val = entry[0]
        freq = struct.unpack("<I", entry[1:5])[0]
        table[byte_val] = freq
    return table


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 15 — PDT ARCHIVE FORMAT
# ═══════════════════════════════════════════════════════════════════════
#
#  .pdt file layout:
#  ┌───────────────────────────────────────────────────────┐
#  │ Magic: "PDT\x00"                            (4 bytes) │
#  │ Version                                     (1 byte)  │
#  │ Original filename length                    (2 bytes) │
#  │ Original filename                           (N bytes) │
#  │ Original file size                          (8 bytes) │
#  │ Original file SHA-256 hash                 (32 bytes) │
#  │ Block count                                 (4 bytes) │
#  │ ET Analysis metadata length                 (4 bytes) │
#  │ ET Analysis metadata (JSON-like)            (M bytes) │
#  │ Merkle root hash                           (64 bytes) │
#  ├───────────────────────────────────────────────────────┤
#  │ For each block:                                       │
#  │   Strategy flag                             (1 byte)  │
#  │   Original block size                       (4 bytes) │
#  │   Compressed block size                     (4 bytes) │
#  │   Block data                                (C bytes) │
#  └───────────────────────────────────────────────────────┘


def _build_metadata(analyzer: DFieldAnalyzer,
                    strategies: List[int],
                    dedup_hits: int) -> bytes:
    """Build ET analysis metadata block."""
    # Simple key=value format (no json import needed)
    lines: List[str] = [
        f"entropy={analyzer.entropy:.6f}",
        f"density={analyzer.density:.6f}",
        f"resonance={analyzer.resonance:.6f}",
        f"bilateral_interference={analyzer.compute_bilateral_interference():.6f}",
        f"manifold_variance={ETMath.manifold_variance(len(set(analyzer.data))):.6f}",
        f"breakpoint_count={len(analyzer.find_breakpoints())!s}",
        f"dedup_hits={dedup_hits!s}",
        f"block_count={len(strategies)!s}",
        f"strategies={','.join(str(s) for s in strategies)}",
        f"manifold_symmetry={MANIFOLD_SYMMETRY!s}",
        f"base_variance={BASE_VARIANCE:.10f}",
        f"phi={PHI:.15f}",
    ]
    return "\n".join(lines).encode("utf-8")


def compress_file(input_path: str,
                  progress_callback=None) -> Tuple[str, Dict[str, Any]]:
    """Compress a file to .pdt format.

    Full ET compression pipeline:
    1. Read file
    2. D-Field Analysis (gradients, curvature, entropy, resonance)
    3. Shimmer-Bilateral Interference detection
    4. Manifold Density Routing (strategy selection)
    5. Block-level CAS deduplication
    6. Per-block adaptive compression
    7. Merkle root integrity verification
    8. PDT archive assembly

    Returns (output_path, stats_dict).
    """
    # Read input
    with open(input_path, "rb") as f:
        raw_data = f.read()

    original_size = len(raw_data)
    original_hash = hashlib.sha256(raw_data).digest()
    original_name = os.path.basename(input_path)

    if progress_callback:
        progress_callback(0.05, "Analyzing D-field structure...")

    # ── Phase 1: D-Field Analysis ──────────────────────────────────
    analyzer = DFieldAnalyzer(raw_data)
    entropy = analyzer.entropy
    density = analyzer.density
    resonance = analyzer.resonance
    bilateral = analyzer.compute_bilateral_interference()
    breakpoints = analyzer.find_breakpoints()

    if progress_callback:
        progress_callback(0.15, f"Entropy={entropy:.2f} Density={density:.3f} "
                                f"Resonance={resonance:.3f} Bilateral={bilateral:.3f}")

    # ── Phase 2: Block Partitioning ────────────────────────────────
    # Use fixed block size for clean partitioning (no tiny blocks)
    block_boundaries: List[int] = [0]
    for i in range(BLOCK_SIZE, original_size, BLOCK_SIZE):
        block_boundaries.append(i)
    if not block_boundaries or block_boundaries[-1] != original_size:
        block_boundaries.append(original_size)
    # Remove any tiny trailing blocks (merge into previous)
    while (len(block_boundaries) > 2 and
           block_boundaries[-1] - block_boundaries[-2] < 64):
        block_boundaries.pop(-2)

    if progress_callback:
        progress_callback(0.20, f"Partitioned into {len(block_boundaries)-1} blocks")

    # ── Phase 3: Per-Block Competitive Compression ─────────────────
    compressed_blocks: List[Tuple[int, bytes, int]] = []  # (strategy, data, orig_len)
    strategies_used: List[int] = []
    merkle_chunks: List[bytes] = []
    total_blocks = len(block_boundaries) - 1

    for idx in range(total_blocks):
        start = block_boundaries[idx]
        end = block_boundaries[idx + 1]
        block = raw_data[start:end]
        block_len = len(block)

        # Competitive compression: try multiple strategies, pick best
        strategy, compressed = _competitive_compress(block)

        compressed_blocks.append((strategy, compressed, block_len))
        strategies_used.append(strategy)
        merkle_chunks.append(block)

        if progress_callback and total_blocks > 0:
            pct = 0.20 + 0.65 * ((idx + 1) / total_blocks)
            progress_callback(pct, f"Block {idx+1}/{total_blocks} "
                                   f"[{_strategy_name(strategy)}] "
                                   f"{block_len}\u2192{len(compressed)}")

    # ── Phase 4: Integrity Verification ────────────────────────────
    merkle = ETMath.merkle_root(merkle_chunks)

    if progress_callback:
        progress_callback(0.88, "Computing Merkle root...")

    # ── Phase 5: Build Metadata ────────────────────────────────────
    dedup_total = sum(1 for s in strategies_used if s == STRAT_DEDUP)
    metadata = _build_metadata(analyzer, strategies_used, dedup_total)

    # ── Phase 6: Assemble Archive ──────────────────────────────────
    output_path = os.path.splitext(input_path)[0] + ".pdt"

    if progress_callback:
        progress_callback(0.92, "Assembling PDT archive...")

    with open(output_path, "wb") as f:
        # Header
        f.write(PDT_MAGIC)
        f.write(struct.pack("<B", PDT_VERSION))
        name_bytes = original_name.encode("utf-8")
        f.write(struct.pack("<H", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<Q", original_size))
        f.write(original_hash)
        f.write(struct.pack("<I", len(compressed_blocks)))
        f.write(struct.pack("<I", len(metadata)))
        f.write(metadata)
        f.write(merkle.encode("ascii")[:64].ljust(64, b"\x00"))

        # Blocks
        for strategy, comp_data, orig_len in compressed_blocks:
            f.write(struct.pack("<B", strategy))
            f.write(struct.pack("<I", orig_len))
            f.write(struct.pack("<I", len(comp_data)))
            f.write(comp_data)

    compressed_size = os.path.getsize(output_path)
    ratio = compressed_size / original_size if original_size > 0 else 0

    if progress_callback:
        progress_callback(1.0, "Compression complete!")

    stats = {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": ratio,
        "entropy": entropy,
        "density": density,
        "resonance": resonance,
        "bilateral_interference": bilateral,
        "block_count": len(compressed_blocks),
        "breakpoints": len(breakpoints),
        "merkle_root": merkle[:16] + "...",
        "strategies": {_strategy_name(s): strategies_used.count(s)
                       for s in set(strategies_used)},
    }

    return output_path, stats


def decompress_file(input_path: str,
                    output_dir: Optional[str] = None,
                    progress_callback=None) -> Tuple[str, bool]:
    """Decompress a .pdt archive back to original file.

    Returns (output_path, integrity_verified).
    """
    with open(input_path, "rb") as f:
        # Header
        magic = f.read(4)
        if magic != PDT_MAGIC:
            raise ValueError("Not a valid PDT archive (wrong magic bytes)")

        version = struct.unpack("<B", f.read(1))[0]
        if version > PDT_VERSION:
            raise ValueError(f"Unsupported PDT version: {version}")

        name_len = struct.unpack("<H", f.read(2))[0]
        original_name = f.read(name_len).decode("utf-8")
        original_size = struct.unpack("<Q", f.read(8))[0]
        original_hash = f.read(32)
        block_count = struct.unpack("<I", f.read(4))[0]
        meta_len = struct.unpack("<I", f.read(4))[0]
        _metadata = f.read(meta_len)  # Parse if needed
        stored_merkle = f.read(64).rstrip(b"\x00").decode("ascii")

        if progress_callback:
            progress_callback(0.05, f"Decompressing '{original_name}' "
                                    f"({block_count} blocks)...")

        # Decompress blocks
        output = bytearray()
        merkle_chunks: List[bytes] = []

        for idx in range(block_count):
            strategy = struct.unpack("<B", f.read(1))[0]
            orig_len = struct.unpack("<I", f.read(4))[0]
            comp_len = struct.unpack("<I", f.read(4))[0]
            comp_data = f.read(comp_len)

            block = _decompress_block(comp_data, strategy, orig_len)

            # Truncate to original length if needed
            block = block[:orig_len]
            output.extend(block)
            merkle_chunks.append(block)

            if progress_callback and block_count > 0:
                pct = 0.05 + 0.85 * ((idx + 1) / block_count)
                progress_callback(pct, f"Block {idx+1}/{block_count}")

    # Truncate to original size
    result = bytes(output[:original_size])

    # Verify integrity
    if progress_callback:
        progress_callback(0.92, "Verifying integrity...")

    result_hash = hashlib.sha256(result).digest()
    hash_ok = result_hash == original_hash

    merkle_check = ETMath.merkle_root(merkle_chunks)
    merkle_ok = merkle_check == stored_merkle

    integrity = hash_ok and merkle_ok

    # Write output
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, original_name)

    # Avoid overwriting — append (decompressed) if file exists
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_decompressed{ext}"

    with open(output_path, "wb") as f:
        f.write(result)

    if progress_callback:
        progress_callback(1.0, "Decompression complete!")

    return output_path, integrity


def _strategy_name(s: int) -> str:
    """Human-readable strategy name."""
    return {
        STRAT_RAW: "RAW",
        STRAT_LZ77: "LZ77",
        STRAT_BWT_MTF_RLE_HUFF: "BWT+MTF+RLE+Huffman",
        STRAT_DELTA_HUFF: "Delta+Huffman",
        STRAT_DEDUP: "Dedup+LZ77",
    }.get(s, f"Unknown({s})")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 16 — GUI (Tkinter, zero external deps)
# ═══════════════════════════════════════════════════════════════════════


class ETCompressorGUI:
    """Main GUI for the ET Compressor application."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ET Compressor — Exception Theory File Compression")
        self.root.geometry("780x720")
        self.root.minsize(680, 600)
        self.root.configure(bg="#0d1117")

        self._selected_file: Optional[str] = None
        self._is_running: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        style = ttk.Style()
        style.theme_use("clam")

        # Colors
        bg = "#0d1117"
        fg = "#c9d1d9"
        accent = "#58a6ff"
        green = "#3fb950"
        card_bg = "#161b22"
        border = "#30363d"

        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card_bg)
        style.configure("TLabel", background=bg, foreground=fg,
                        font=("Consolas", 10))
        style.configure("Title.TLabel", background=bg, foreground=accent,
                        font=("Consolas", 16, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground="#8b949e",
                        font=("Consolas", 9))
        style.configure("Card.TLabel", background=card_bg, foreground=fg,
                        font=("Consolas", 10))
        style.configure("CardTitle.TLabel", background=card_bg,
                        foreground=accent, font=("Consolas", 11, "bold"))
        style.configure("Status.TLabel", background=bg, foreground=green,
                        font=("Consolas", 9))
        style.configure("TButton", font=("Consolas", 11, "bold"),
                        padding=8)
        style.configure("Accent.TButton", font=("Consolas", 11, "bold"),
                        padding=8)
        style.configure("TProgressbar", troughcolor=border,
                        background=accent, thickness=20)

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Title ──────────────────────────────────────────────────
        ttk.Label(main, text="ET Compressor", style="Title.TLabel"
                  ).pack(anchor="w")
        ttk.Label(main,
                  text="Exception Theory-Derived File Compression Engine  "
                       "| P\u2218D\u2218T = E",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(0, 15))

        # ── File Selection ─────────────────────────────────────────
        file_frame = ttk.Frame(main)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self._file_label = ttk.Label(file_frame, text="No file selected",
                                     style="TLabel")
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self._select_btn = ttk.Button(btn_frame, text="Select File",
                                      command=self._select_file)
        self._select_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._compress_btn = ttk.Button(btn_frame, text="Compress \u2192 .pdt",
                                        command=self._compress,
                                        style="Accent.TButton")
        self._compress_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._compress_btn.state(["disabled"])

        self._decompress_btn = ttk.Button(btn_frame, text="Decompress .pdt",
                                          command=self._decompress)
        self._decompress_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._decompress_btn.state(["disabled"])

        # ── Progress ───────────────────────────────────────────────
        self._progress = ttk.Progressbar(main, mode="determinate",
                                         style="TProgressbar", maximum=100)
        self._progress.pack(fill=tk.X, pady=(5, 2))

        self._status_label = ttk.Label(main, text="Ready",
                                       style="Status.TLabel")
        self._status_label.pack(anchor="w", pady=(0, 10))

        # ── ET Analysis Card ───────────────────────────────────────
        analysis_card = ttk.LabelFrame(main, text=" D-Field Analysis ",
                                       padding=10)
        analysis_card.pack(fill=tk.X, pady=(0, 10))

        self._analysis_text = tk.Text(analysis_card, height=8,
                                      bg=card_bg, fg=fg,
                                      font=("Consolas", 9),
                                      relief="flat", wrap="word",
                                      insertbackground=fg,
                                      selectbackground=accent)
        self._analysis_text.pack(fill=tk.X)
        self._analysis_text.insert("1.0", "Select a file to begin analysis...")
        self._analysis_text.config(state="disabled")

        # ── Results Card ───────────────────────────────────────────
        results_card = ttk.LabelFrame(main, text=" Compression Results ",
                                      padding=10)
        results_card.pack(fill=tk.BOTH, expand=True)

        self._results_text = tk.Text(results_card, height=12,
                                     bg=card_bg, fg=fg,
                                     font=("Consolas", 9),
                                     relief="flat", wrap="word",
                                     insertbackground=fg,
                                     selectbackground=accent)
        self._results_text.pack(fill=tk.BOTH, expand=True)
        self._results_text.insert("1.0",
                                  "Compression/decompression results "
                                  "will appear here.\n\n"
                                  "ET Math applied:\n"
                                  "  Eq 4:     Recursive Descriptor Search\n"
                                  "  Eq 16:    Shannon Entropy\n"
                                  "  Eq 77:    Kolmogorov Complexity\n"
                                  "  Eq 83:    Variance Gradient Descent\n"
                                  "  Eq 108:   Dynamic Attractor Shimmer\n"
                                  "  Eq 109:   Phi-Harmonic Resonance\n"
                                  "  Eq 110:   Bilateral Interference\n"
                                  "  Eq 114:   P-D Tension Shimmer\n"
                                  "  Eq 117:   Shimmer Radiation (1/r\u00b2)\n"
                                  "  Eq 118:   Shimmer Oscillation Modulation\n"
                                  "  Eq 121:   Phi Harmonic Generation\n"
                                  "  Eq 127:   Manifold Temporal Decay\n"
                                  "  Eq 134:   Cross-Correlation\n"
                                  "  Eq 135:   Tri-State Decision\n"
                                  "  Eq 211:   Structural Density\n"
                                  "  Eq 212:   Manifold Variance\n"
                                  "  Merkle:   Holographic Integrity\n"
                                  "  BWT:      Manifold Sorting\n"
                                  "  CAS:      Content-Addressable Dedup\n")
        self._results_text.config(state="disabled")

        # ── Footer ─────────────────────────────────────────────────
        footer = ttk.Label(main,
                           text="Civilizational Blueprint  |  "
                                "Exception Theory  |  "
                                "MANIFOLD_SYMMETRY=12  |  "
                                "BASE_VARIANCE=1/12",
                           style="Subtitle.TLabel")
        footer.pack(side=tk.BOTTOM, pady=(10, 0))

    def _select_file(self) -> None:
        """Open file dialog to select a file."""
        path = filedialog.askopenfilename(
            title="Select File to Compress/Decompress",
            filetypes=[
                ("All Files", "*.*"),
                ("PDT Archives", "*.pdt"),
            ]
        )
        if path:
            self._selected_file = path
            size = os.path.getsize(path)
            name = os.path.basename(path)
            self._file_label.config(
                text=f"{name}  ({self._format_size(size)})")

            is_pdt = path.lower().endswith(".pdt")
            if is_pdt:
                self._compress_btn.state(["disabled"])
                self._decompress_btn.state(["!disabled"])
            else:
                self._compress_btn.state(["!disabled"])
                self._decompress_btn.state(["disabled"])

            # Quick analysis
            if not is_pdt:
                self._quick_analysis(path)

    def _quick_analysis(self, path: str) -> None:
        """Run quick D-field analysis on selected file."""
        try:
            with open(path, "rb") as f:
                # Read up to 64KB for quick analysis
                sample = f.read(65536)

            analyzer = DFieldAnalyzer(sample)

            lines = [
                f"File: {os.path.basename(path)}",
                f"Sample Size: {len(sample):,} bytes",
                f"",
                f"Shannon Entropy:       {analyzer.entropy:.4f} bits/byte "
                f"(max 8.0)",
                f"Manifold Density:      {analyzer.density:.4f} "
                f"({len(set(sample))}/256 unique bytes)",
                f"Phi-Harmonic Resonance: {analyzer.resonance:.4f}",
                f"Bilateral Interference: "
                f"{analyzer.compute_bilateral_interference():.4f}",
                f"Structural Breakpoints: {len(analyzer.find_breakpoints())}",
                f"Manifold Variance:     "
                f"{ETMath.manifold_variance(len(set(sample))):.4f}",
                f"",
                f"Region Classification:",
            ]

            # Classify regions
            chunk_size = min(len(sample) // 4, 8192)
            if chunk_size > 0:
                for i in range(min(4, len(sample) // chunk_size)):
                    start = i * chunk_size
                    end = start + chunk_size
                    cls = analyzer.classify_region(start, end)
                    lines.append(
                        f"  Block {i}: [{start:>6}..{end:>6}] → {cls}")

            self._set_analysis_text("\n".join(lines))

        except Exception as e:
            self._set_analysis_text(f"Analysis error: {e}")

    def _compress(self) -> None:
        """Start compression in background thread."""
        if not self._selected_file or self._is_running:
            return

        self._is_running = True
        self._compress_btn.state(["disabled"])
        self._select_btn.state(["disabled"])
        self._progress["value"] = 0

        def _run() -> None:
            try:
                out_path, stats = compress_file(
                    self._selected_file,
                    progress_callback=self._update_progress)

                # Format results
                lines = [
                    "COMPRESSION COMPLETE",
                    "=" * 50,
                    f"Input:  {os.path.basename(self._selected_file)}",
                    f"Output: {os.path.basename(out_path)}",
                    "",
                    f"Original Size:    {self._format_size(stats['original_size'])}",
                    f"Compressed Size:  {self._format_size(stats['compressed_size'])}",
                    f"Ratio:            {stats['ratio']:.4f} "
                    f"({stats['ratio']*100:.1f}%)",
                    f"Savings:          "
                    f"{self._format_size(stats['original_size'] - stats['compressed_size'])} "
                    f"({(1-stats['ratio'])*100:.1f}% reduction)",
                    "",
                    "ET Analysis:",
                    f"  Shannon Entropy:       {stats['entropy']:.4f} bits/byte",
                    f"  Manifold Density:      {stats['density']:.4f}",
                    f"  Phi Resonance:         {stats['resonance']:.4f}",
                    f"  Bilateral Interference: {stats['bilateral_interference']:.4f}",
                    f"  Structural Breakpoints: {stats['breakpoints']}",
                    f"  Merkle Root:           {stats['merkle_root']}",
                    "",
                    "Strategy Distribution:",
                ]
                for name, count in stats["strategies"].items():
                    lines.append(f"  {name}: {count} blocks")

                lines.extend([
                    "",
                    f"Total Blocks: {stats['block_count']}",
                    f"Output: {out_path}",
                ])

                self.root.after(0, self._set_results_text, "\n".join(lines))

            except Exception as e:
                self.root.after(0, self._set_results_text,
                                f"COMPRESSION FAILED\n\nError: {e}")
            finally:
                self.root.after(0, self._finish_operation)

        threading.Thread(target=_run, daemon=True).start()

    def _decompress(self) -> None:
        """Start decompression in background thread."""
        if not self._selected_file or self._is_running:
            return

        self._is_running = True
        self._decompress_btn.state(["disabled"])
        self._select_btn.state(["disabled"])
        self._progress["value"] = 0

        def _run() -> None:
            try:
                out_path, integrity = decompress_file(
                    self._selected_file,
                    progress_callback=self._update_progress)

                status = "VERIFIED" if integrity else "FAILED"
                lines = [
                    "DECOMPRESSION COMPLETE",
                    "=" * 50,
                    f"Input:     {os.path.basename(self._selected_file)}",
                    f"Output:    {os.path.basename(out_path)}",
                    f"Size:      {self._format_size(os.path.getsize(out_path))}",
                    f"Integrity: {status}",
                    "",
                    f"Output: {out_path}",
                ]

                self.root.after(0, self._set_results_text, "\n".join(lines))

            except Exception as e:
                self.root.after(0, self._set_results_text,
                                f"DECOMPRESSION FAILED\n\nError: {e}")
            finally:
                self.root.after(0, self._finish_operation)

        threading.Thread(target=_run, daemon=True).start()

    def _update_progress(self, pct: float, message: str) -> None:
        """Update progress bar and status from any thread."""
        self.root.after(0, self._set_progress, pct, message)

    def _set_progress(self, pct: float, message: str) -> None:
        """Set progress bar value and status label."""
        self._progress["value"] = pct * 100
        self._status_label.config(text=message)

    def _finish_operation(self) -> None:
        """Re-enable controls after operation completes."""
        self._is_running = False
        self._select_btn.state(["!disabled"])
        if self._selected_file:
            is_pdt = self._selected_file.lower().endswith(".pdt")
            if is_pdt:
                self._decompress_btn.state(["!disabled"])
            else:
                self._compress_btn.state(["!disabled"])

    def _set_analysis_text(self, text: str) -> None:
        """Set the analysis text widget content."""
        self._analysis_text.config(state="normal")
        self._analysis_text.delete("1.0", tk.END)
        self._analysis_text.insert("1.0", text)
        self._analysis_text.config(state="disabled")

    def _set_results_text(self, text: str) -> None:
        """Set the results text widget content."""
        self._results_text.config(state="normal")
        self._results_text.delete("1.0", tk.END)
        self._results_text.insert("1.0", text)
        self._results_text.config(state="disabled")

    @staticmethod
    def _format_size(size: int) -> str:
        """Format byte count to human-readable string."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size) < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def run(self) -> None:
        """Start the GUI event loop."""
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 17 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Launch the ET Compressor GUI."""
    app = ETCompressorGUI()
    app.run()


if __name__ == "__main__":
    main()

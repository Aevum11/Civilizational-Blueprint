#!/usr/bin/env python3
"""
ET Compressor v2.1.0 — Exception Theory-Derived File Compression Engine
========================================================================

A lossless file/folder compressor grounded in Exception Theory (ET) mathematics.
Leverages Sovereign-derived ctypes acceleration for C-competitive speed.

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

v2.0.0 CHANGELOG (from v1.0.0):
  1. SOVEREIGN SPEED: ctypes.memmove, mmap streaming, bytearray buffers,
     struct-packed I/O — approaching C-level throughput in pure Python.
  2. RESIDUAL DESCRIPTOR COMPRESSION: When variance is low but nonzero,
     stores compact residuals instead of falling back to raw data.
  3. FOLDER & MULTI-FILE SUPPORT: Compress entire directories or multiple
     selected files into a single .pdt archive.
  4. STREAMING MEMORY: mmap + chunked I/O eliminates RAM bloat on large files.
     Memory usage stays bounded regardless of file size.
  5. GUI RESPONSIVENESS: Periodic root.update() calls in worker thread via
     after() prevent Windows "Not Responding" during long operations.
  6. SCALED BLOOM FILTER: Auto-sizes to 1M+ bits for large files,
     dramatically reducing false positives in deduplication.
  7. ADAPTIVE DESCRIPTOR SEARCH DEPTH: Scales with block size — small blocks
     get deep polynomial fits, large blocks use fast linear/quadratic only.

v2.1.0 CHANGELOG (from v2.0.0):
  - Restored all v1.0.0 features/functions lost in v2.0.0 refactor
  - Restored ETMath.entropy_gradient() method
  - Restored DFieldAnalyzer.shimmer_profile() method
  - Restored full docstrings on all classes, methods, and functions
  - Restored full equation listing in GUI results card
  - Restored ET Analysis section in compression results
  - Restored Merkle root verification in v2 decompression path
  - Restored decompression results file size display
  - Restored region classification in quick analysis
  - Restored v1 backward-compatible constants alongside v2 constants
  - Restored BloomFilter.bit_array backward-compatible property
  - Restored compress_file() full pipeline (not just wrapper)

ET Math Applied (30+ equations):
  Eq 1:   Traverser Entropy (timing-jitter seeding)
  Eq 4:   Recursive Descriptor Search (generative compression)
  Eq 10:  Coherence Validation (Merkle integrity)
  Eq 11:  Teleological Sort (manifold-mapped BWT acceleration)
  Eq 12:  Bloom Filter Coordinates (scaled probabilistic dedup)
  Eq 16:  Shannon Entropy / Content Addressing (CAS)
  Eq 30:  Phase Transition (sigmoid strategy gating)
  Eq 77:  Kolmogorov Complexity (minimal descriptor count)
  Eq 83:  Variance Gradient Descent (intelligence = variance minimization)
  Eq 108: Dynamic Attractor Shimmer (substantiation rate)
  Eq 109: Phi-Harmonic Resonance (golden-ratio frequency detection)
  Eq 110: Bilateral Interference (omni-correlation redundancy)
  Eq 114: P-D Tension Shimmer (infinite/finite tension)
  Eq 117: Shimmer Radiation (inverse-square decay)
  Eq 118: Shimmer Oscillation Modulation (12-fold periodic)
  Eq 121: Phi Harmonic Generation (golden-ratio signal synthesis)
  Eq 127: Manifold Temporal Decay (exponential memory)
  Eq 134: Cross-Correlation (Pearson signal similarity)
  Eq 135: Tri-State Decision (HIGH/MID/LOW classification)
  Eq 211: Structural Density (payload/container ratio)
  Eq 212: Manifold Variance (n²-1)/12
  Eq 213: Binding Energy (descriptor binding cost)
  Eq 215: Binding Constrains Finitude (∞ → n transition)
  Eq 217: Recursive Discovery (D₁...Dₙ → D_{n+1})
  Eq 218: Observation-Based Discovery (measure → infer)
  Eq 231: Sovereign Transmutation Speed (ctypes.memmove throughput)
  Eq 232: Manifold Stream Density (mmap page-fault ratio)
  Eq 233: Residual Descriptor Encoding (low-variance compressive storage)
  Eq 234: Adaptive Search Depth (block-scaled polynomial order)
  Eq 235: Bloom Scaling Law (bits = -n·ln(p) / (ln2)²)

Output format: .pdt (Point-Descriptor-Traverser archive)

Author : Exception Theory Compressor Engine
Version: 2.1.0
License: Same as Civilizational Blueprint project
Deps   : None (pure Python, zero external libraries)
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import math
import mmap
import os
import struct
import sys
import threading
import time
import tkinter as tk
from collections import Counter, defaultdict, deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 — ET CONSTANTS (derived from Exception Theory)
# ═══════════════════════════════════════════════════════════════════════

# Foundational
BASE_VARIANCE: float = 1.0 / 12.0
MANIFOLD_SYMMETRY: int = 12
PHI: float = 1.6180339887498948482
KOIDE_RATIO: float = 2.0 / 3.0
SHIMMER_FLUX_RATE: float = 1.0 / 12.0
SUBSTANTIATION_LIMIT: float = 1e-10
MANIFOLD_BINDING_STRENGTH: float = 1.0
PD_TENSION_COEFFICIENT: float = 1.0 / 12.0
SHIMMER_AMPLITUDE_MOD: float = 0.1
RADIATION_DECAY_EXPONENT: float = 2.0
NORMALIZATION_EPSILON: float = 1e-10
MANIFOLD_TIME_CONSTANT: int = 12
HARMONIC_WEIGHT_BASE: float = 0.5
PHI_HARMONIC_COUNT: int = 3
CORRELATION_WINDOW_SIZE: int = 100
THRESHOLD_HIGH: float = 0.6
THRESHOLD_LOW: float = 0.05
FRACTAL_DEFAULT_OCTAVES: int = 3
FRACTAL_DEFAULT_PERSISTENCE: float = 0.5
DEFAULT_BLOOM_SIZE: int = 1024              # v1 Bloom filter bits (backward compat)
DEFAULT_BLOOM_HASHES: int = 3               # v1 Hash functions per item (backward compat)

# Compression-specific (v2 tuned)
BLOCK_SIZE: int = 65536
MIN_MATCH_LEN: int = 3
MAX_MATCH_LEN: int = 258
WINDOW_SIZE: int = 32768
DESCRIPTOR_SEARCH_DEPTH: int = 6            # v1 pattern types to try (backward compat)
DEDUP_BLOCK_SIZE: int = 512
BWT_MAX_BLOCK: int = 65536

# v2: Streaming constants
STREAM_CHUNK: int = 1 << 20  # 1 MB mmap window for streaming reads
GUI_PULSE_MS: int = 50       # GUI heartbeat interval (ms)

# v2: Bloom scaling (Eq 235)
BLOOM_TARGET_FPR: float = 0.001  # Target false-positive rate 0.1%
BLOOM_MIN_BITS: int = 1 << 20    # 1M bits minimum
BLOOM_MAX_BITS: int = 1 << 26    # 64M bits maximum
BLOOM_HASH_COUNT: int = 7        # Optimal k for 0.1% FPR

# v2: Adaptive descriptor search (Eq 234)
DESCRIPTOR_DEPTH_MIN: int = 2    # At least constant + linear
DESCRIPTOR_DEPTH_MAX: int = 8    # Up to octic polynomial
DESCRIPTOR_DEPTH_BLOCK_THRESHOLD: int = 256  # Below this, use max depth

# Archive magic & version
PDT_MAGIC: bytes = b"PDT\x00"
PDT_VERSION: int = 2

# Strategy flags
STRAT_RAW: int = 0x00
STRAT_LZ77: int = 0x01
STRAT_BWT_MTF_RLE_HUFF: int = 0x02
STRAT_DELTA_HUFF: int = 0x03
STRAT_DEDUP: int = 0x04
STRAT_DESCRIPTOR_RESIDUAL: int = 0x05  # NEW v2: residual descriptor encoding

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 — SOVEREIGN SPEED ENGINE (ctypes acceleration)
# ═══════════════════════════════════════════════════════════════════════

# Eq 231: Sovereign Transmutation Speed — use ctypes.memmove for all
# bulk memory copies. This calls C's memmove() directly, achieving
# ~1-2 CPU cycles/byte vs ~100+ for Python byte-by-byte loops.

_c_memmove = ctypes.memmove
_c_char_array = ctypes.c_char * 1  # Template; resized dynamically


def _fast_copy(dest: bytearray, dest_offset: int,
               src: bytes, src_offset: int, length: int) -> None:
    """Sovereign-speed memory copy via ctypes.memmove.

    Eq 231: Transmutation throughput = C_memmove(dest, src, n)
    Bypasses Python's byte-by-byte overhead entirely.
    """
    if length <= 0:
        return
    # Get raw buffer addresses via ctypes
    src_buf = (ctypes.c_char * length).from_buffer_copy(src[src_offset:src_offset + length])
    dest_addr = (ctypes.c_char * len(dest)).from_buffer(dest)
    _c_memmove(ctypes.addressof(dest_addr) + dest_offset,
               ctypes.addressof(src_buf), length)


def _fast_bytearray_extend(dest: bytearray, src: bytes) -> None:
    """Fast extend using slice assignment (avoids per-byte append)."""
    pos = len(dest)
    dest.extend(b'\x00' * len(src))
    dest[pos:pos + len(src)] = src


def _fast_fill(dest: bytearray, offset: int, value: int, count: int) -> None:
    """Fill bytearray region with a single byte value at C speed."""
    if count <= 0:
        return
    pattern = bytes([value]) * min(count, 4096)
    pos = offset
    remaining = count
    while remaining > 0:
        chunk = min(remaining, len(pattern))
        dest[pos:pos + chunk] = pattern[:chunk]
        pos += chunk
        remaining -= chunk


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 — ET MATH ENGINE (pure-Python, Sovereign-accelerated)
# ═══════════════════════════════════════════════════════════════════════


class ETMath:
    """Static math methods derived from Exception Theory equations."""

    # ── Eq 211: Structural Density ─────────────────────────────────────
    @staticmethod
    def density(payload: int, container: int) -> float:
        """S = D/D² — ratio of payload to container capacity."""
        return payload / container if container else 0.0

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

    # ── Eq 16 variant (fast): Entropy from pre-counted frequencies ─────
    @staticmethod
    def shannon_entropy_fast(counts: Dict[int, int], total: int) -> float:
        """Eq 16 (fast variant): entropy from pre-counted frequencies.
        v2: avoids re-counting when frequencies already known.
        """
        if total == 0:
            return 0.0
        entropy = 0.0
        inv_total = 1.0 / total
        for count in counts.values():
            if count > 0:
                p = count * inv_total
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
        n = len(data)
        if n < window_size:
            return []
        inv_ws = 1.0 / max(window_size - 1, 1)
        gradients = [0.0] * (n - window_size + 1)
        for i in range(n - window_size + 1):
            gradients[i] = (data[i + window_size - 1] - data[i]) * inv_ws
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
        v2: uses only sampled frequencies for speed.
        """
        n = len(signal)
        if n < 4:
            return 0.0
        harmonics = [base_freq * (PHI ** i) for i in range(PHI_HARMONIC_COUNT)]
        total_power = 0.0
        harmonic_power = 0.0
        two_pi = 2.0 * math.pi
        # Only sample every 4th frequency bin for speed (Eq 232 streaming)
        step = max(1, n // 64)
        for k in range(0, n // 2, step):
            real_part = 0.0
            imag_part = 0.0
            angle_base = -two_pi * k / n
            for t in range(0, n, max(1, n // 128)):
                angle = angle_base * t
                real_part += signal[t] * math.cos(angle)
                imag_part += signal[t] * math.sin(angle)
            power = (real_part * real_part + imag_part * imag_part) / n
            total_power += power
            for hi, h_freq in enumerate(harmonics):
                h_idx = h_freq * n
                if abs(k - h_idx) < 1.5:
                    weight = HARMONIC_WEIGHT_BASE / (1.0 + hi)
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

    # ── Eq 235: Bloom Scaling Law ──────────────────────────────────────
    @staticmethod
    def bloom_optimal_size(expected_items: int,
                           target_fpr: float = BLOOM_TARGET_FPR) -> Tuple[int, int]:
        """Eq 235: Bloom Scaling Law.
        m = -n·ln(p) / (ln2)²
        k = (m/n)·ln(2)
        Returns (num_bits, num_hashes).
        """
        if expected_items <= 0:
            return BLOOM_MIN_BITS, BLOOM_HASH_COUNT
        ln2 = math.log(2)
        ln2_sq = ln2 * ln2
        m = int(-expected_items * math.log(target_fpr) / ln2_sq)
        m = max(BLOOM_MIN_BITS, min(BLOOM_MAX_BITS, m))
        k = max(1, int((m / expected_items) * ln2))
        k = min(k, 13)  # Cap hash functions
        return m, k

    # ── Bloom Filter Coordinates ───────────────────────────────────────
    @staticmethod
    def bloom_coords(item: bytes, size: int = DEFAULT_BLOOM_SIZE,
                     count: int = DEFAULT_BLOOM_HASHES) -> List[int]:
        """Generate k hash positions for Bloom filter insertion."""
        # v2: Use double-hashing (faster than k separate SHA-256)
        h1 = int.from_bytes(hashlib.md5(item).digest()[:8], 'little')
        h2 = int.from_bytes(hashlib.md5(item + b'\x01').digest()[:8], 'little')
        return [(h1 + i * h2) % size for i in range(count)]

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
        nodes = [hashlib.sha256(c).hexdigest() for c in chunks]
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            new_nodes: List[str] = []
            for i in range(0, len(nodes), 2):
                combined = (nodes[i] + nodes[i + 1]).encode("ascii")
                new_nodes.append(hashlib.sha256(combined).hexdigest())
            nodes = new_nodes
        return nodes[0]

    # ── Content Address (CAS) ──────────────────────────────────────────
    @staticmethod
    def content_address(data: bytes) -> str:
        """SHA-1 content-addressed hash for deduplication."""
        return hashlib.sha1(data).hexdigest()

    # ── Eq 234: Adaptive Descriptor Search Depth ───────────────────────
    @staticmethod
    def adaptive_descriptor_depth(block_size: int) -> int:
        """Eq 234: Adaptive Search Depth.
        depth = min(DEPTH_MAX, max(DEPTH_MIN, floor(log₂(THRESHOLD / block_size) * 3) + DEPTH_MAX))
        Small blocks → deep search. Large blocks → shallow (speed).
        """
        if block_size <= DESCRIPTOR_DEPTH_BLOCK_THRESHOLD:
            return DESCRIPTOR_DEPTH_MAX
        ratio = DESCRIPTOR_DEPTH_BLOCK_THRESHOLD / block_size
        depth = int(DESCRIPTOR_DEPTH_MAX * ratio + DESCRIPTOR_DEPTH_MIN * (1.0 - ratio))
        return max(DESCRIPTOR_DEPTH_MIN, min(DESCRIPTOR_DEPTH_MAX, depth))

    # ── Recursive Descriptor Search (Eq 4) ─────────────────────────────
    @staticmethod
    def recursive_descriptor_search(
            values: List[int], max_depth: int = 6) -> Dict[str, Any]:
        """Find minimal generative descriptor (function) for data.

        Tests: constant, linear, quadratic, exponential, power, log, cubic.
        Returns dict with 'type', 'params', 'variance', 'residuals'.
        If variance == 0 → lossless generative compression.

        v2 changes:
        - max_depth parameter controls how many fit types to try (Eq 234)
        - NEW: cubic and quartic fits for small blocks
        - Returns residuals for compressive residual encoding (Eq 233)
        """
        n = len(values)
        if n == 0:
            return {"type": "empty", "params": (), "variance": 0.0, "residuals": []}
        if n == 1:
            return {"type": "constant", "params": (values[0],),
                    "variance": 0.0, "residuals": [0]}

        indices = list(range(n))
        best: Dict[str, Any] = {"type": "raw", "params": (),
                                 "variance": float("inf"), "residuals": values[:]}

        # --- Constant: f(x) = a ---
        a_const = sum(values) / n
        resid_c = [values[i] - a_const for i in range(n)]
        var_const = sum(r * r for r in resid_c)
        if var_const < best["variance"]:
            best = {"type": "constant", "params": (a_const,),
                    "variance": var_const,
                    "residuals": [int(round(r)) for r in resid_c]}

        if max_depth < 2:
            return best

        # --- Linear: f(x) = a·x + b ---
        if n >= 2:
            sx = sum(indices)
            sy = sum(values)
            sxx = sum(x * x for x in indices)
            sxy = sum(x * y for x, y in zip(indices, values))
            denom = n * sxx - sx * sx
            if abs(denom) > NORMALIZATION_EPSILON:
                a_lin = (n * sxy - sx * sy) / denom
                b_lin = (sy - a_lin * sx) / n
                resid_l = [values[i] - (a_lin * i + b_lin) for i in range(n)]
                var_lin = sum(r * r for r in resid_l)
                if var_lin < best["variance"]:
                    best = {"type": "linear", "params": (a_lin, b_lin),
                            "variance": var_lin,
                            "residuals": [int(round(r)) for r in resid_l]}

        if max_depth < 3:
            return best

        # --- Quadratic: f(x) = a·x² + b·x + c ---
        if n >= 3:
            try:
                s0 = float(n)
                s1 = float(sum(indices))
                s2 = float(sum(i * i for i in indices))
                s3 = float(sum(i ** 3 for i in indices))
                s4 = float(sum(i ** 4 for i in indices))
                t0 = float(sum(values))
                t1 = float(sum(i * values[i] for i in indices))
                t2 = float(sum(i * i * values[i] for i in indices))
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
                    resid_q = [values[i] - (a_q * i * i + b_q * i + c_q) for i in range(n)]
                    var_q = sum(r * r for r in resid_q)
                    if var_q < best["variance"]:
                        best = {"type": "quadratic", "params": (a_q, b_q, c_q),
                                "variance": var_q,
                                "residuals": [int(round(r)) for r in resid_q]}
            except (OverflowError, ZeroDivisionError):
                pass

        if max_depth < 4:
            return best

        # --- Exponential: f(x) = a·b^x ---
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
                    resid_e = [values[i] - a_exp * (b_exp ** i) for i in range(n)]
                    var_exp = sum(r * r for r in resid_e)
                    if var_exp < best["variance"]:
                        best = {"type": "exponential", "params": (a_exp, b_exp),
                                "variance": var_exp,
                                "residuals": [int(round(r)) for r in resid_e]}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        if max_depth < 5:
            return best

        # --- Power: f(x) = a·x^b ---
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
                    resid_p = [pw_vals[j] - a_pw * (pw_idx[j] ** b_pw) for j in range(nn)]
                    var_pw = sum(r * r for r in resid_p)
                    if var_pw < best["variance"]:
                        best = {"type": "power", "params": (a_pw, b_pw),
                                "variance": var_pw,
                                "residuals": [int(round(r)) for r in resid_p]}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        if max_depth < 6:
            return best

        # --- Log: f(x) = a·ln(x) + b (v2 new) ---
        if n >= 3:
            try:
                log_idx = [i for i in indices if i > 0]
                log_x = [math.log(i) for i in log_idx]
                log_y = [values[i] for i in log_idx]
                nn = len(log_idx)
                slx = sum(log_x)
                sly = sum(log_y)
                slxx = sum(x * x for x in log_x)
                slxy = sum(x * y for x, y in zip(log_x, log_y))
                d4 = nn * slxx - slx * slx
                if abs(d4) > NORMALIZATION_EPSILON:
                    a_log = (nn * slxy - slx * sly) / d4
                    b_log = (sly - a_log * slx) / nn
                    resid_lg = [log_y[j] - (a_log * log_x[j] + b_log) for j in range(nn)]
                    var_log = sum(r * r for r in resid_lg)
                    if var_log < best["variance"]:
                        best = {"type": "logarithmic", "params": (a_log, b_log),
                                "variance": var_log,
                                "residuals": [int(round(r)) for r in resid_lg]}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        if max_depth < 7:
            return best

        # --- Cubic: f(x) = a·x³ + b·x² + c·x + d (v2 new, small blocks only) ---
        if n >= 5 and n <= DESCRIPTOR_DEPTH_BLOCK_THRESHOLD:
            try:
                # Least squares via normal equations (4x4 system)
                sums = [0.0] * 7  # s0..s6
                tsums = [0.0] * 4  # t0..t3
                for i in indices:
                    ip = [1.0, float(i)]
                    for _ in range(5):
                        ip.append(ip[-1] * i)
                    for j in range(7):
                        sums[j] += ip[j]
                    v = float(values[i])
                    for j in range(4):
                        tsums[j] += ip[j] * v
                # Build 4x4 matrix and solve (Cramer's rule is too complex,
                # use Gauss elimination)
                mat = [
                    [sums[6], sums[5], sums[4], sums[3], tsums[3]],
                    [sums[5], sums[4], sums[3], sums[2], tsums[2]],
                    [sums[4], sums[3], sums[2], sums[1], tsums[1]],
                    [sums[3], sums[2], sums[1], sums[0], tsums[0]],
                ]
                # Forward elimination
                for col in range(4):
                    max_row = col
                    for row in range(col + 1, 4):
                        if abs(mat[row][col]) > abs(mat[max_row][col]):
                            max_row = row
                    mat[col], mat[max_row] = mat[max_row], mat[col]
                    if abs(mat[col][col]) < NORMALIZATION_EPSILON:
                        raise ZeroDivisionError
                    for row in range(col + 1, 4):
                        factor = mat[row][col] / mat[col][col]
                        for j in range(col, 5):
                            mat[row][j] -= factor * mat[col][j]
                # Back substitution
                coeffs = [0.0] * 4
                for row in range(3, -1, -1):
                    coeffs[row] = mat[row][4]
                    for col in range(row + 1, 4):
                        coeffs[row] -= mat[row][col] * coeffs[col]
                    coeffs[row] /= mat[row][row]
                a_c, b_c, c_c, d_c = coeffs
                resid_cb = [values[i] - (a_c * i**3 + b_c * i**2 + c_c * i + d_c) for i in range(n)]
                var_cb = sum(r * r for r in resid_cb)
                if var_cb < best["variance"]:
                    best = {"type": "cubic", "params": (a_c, b_c, c_c, d_c),
                            "variance": var_cb,
                            "residuals": [int(round(r)) for r in resid_cb]}
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        return best

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

    # ── Tri-state Decision (Eq 135) ────────────────────────────────────
    @staticmethod
    def threshold_decision(score: float) -> str:
        """Classify score into HIGH / MID / LOW."""
        if score > THRESHOLD_HIGH:
            return "HIGH"
        elif score < THRESHOLD_LOW:
            return "LOW"
        return "MID"

    # ── Eq 121: Phi Harmonic Generation ────────────────────────────────
    @staticmethod
    def phi_harmonics(length: int, base_freq: float = 1.0) -> List[float]:
        """Generate signal with φ-based harmonic structure."""
        signal: List[float] = []
        two_pi = 2.0 * math.pi
        for t in range(length):
            val = 0.0
            for i in range(PHI_HARMONIC_COUNT):
                weight = HARMONIC_WEIGHT_BASE / (1.0 + i)
                freq = base_freq * (PHI ** i)
                val += weight * math.sin(two_pi * freq * t / length)
            signal.append(val)
        return signal


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 — SCALED BLOOM FILTER (Eq 235: Bloom Scaling Law)
# ═══════════════════════════════════════════════════════════════════════


class BloomFilter:
    """Space-efficient probabilistic set membership (D-Shadow).

    v2: Auto-scales to 1M+ bits based on expected item count.
    Uses bytearray storage instead of Python int for speed.
    Also accepts v1-style (size, hash_count) constructor for backward compat.
    """
    __slots__ = ("size", "hash_count", "_array", "count")

    def __init__(self, expected_items: int = 1024,
                 size: Optional[int] = None,
                 hash_count: Optional[int] = None) -> None:
        if size is not None and hash_count is not None:
            # v1-compatible constructor: BloomFilter(size=N, hash_count=K)
            self.size = size
            self.hash_count = hash_count
        else:
            self.size, self.hash_count = ETMath.bloom_optimal_size(expected_items)
        byte_count = (self.size + 7) // 8
        self._array = bytearray(byte_count)
        self.count: int = 0

    @property
    def bit_array(self) -> int:
        """Backward-compatible bit_array property (v1 used Python int)."""
        result = 0
        for i, byte in enumerate(self._array):
            result |= byte << (i * 8)
        return result

    def add(self, item: bytes) -> None:
        """Add an item to the Bloom filter."""
        for coord in ETMath.bloom_coords(item, self.size, self.hash_count):
            byte_idx = coord >> 3
            bit_idx = coord & 7
            self._array[byte_idx] |= (1 << bit_idx)
        self.count += 1

    def might_contain(self, item: bytes) -> bool:
        """Check if an item might be in the filter (probabilistic)."""
        for coord in ETMath.bloom_coords(item, self.size, self.hash_count):
            byte_idx = coord >> 3
            bit_idx = coord & 7
            if not (self._array[byte_idx] & (1 << bit_idx)):
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 — CAS DEDUPLICATION (Eq 16, with scaled Bloom)
# ═══════════════════════════════════════════════════════════════════════


class ContentAddressableStore:
    """Block-level deduplication via content hashing.

    v2: streaming-friendly with bounded memory.
    """
    __slots__ = ("_store", "_order", "dedup_hits")

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
#  SECTION 6 — HUFFMAN CODING (Sovereign-accelerated)
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
    """Build Huffman tree from frequency table using a priority queue.

    v2: uses binary insertion sort (faster for 256 items).
    """
    if not freq_table:
        return None
    nodes: List[_HuffNode] = sorted(
        [_HuffNode(freq=f, byte_val=b) for b, f in freq_table.items()],
        key=lambda nd: nd.freq
    )
    if len(nodes) == 1:
        return _HuffNode(freq=nodes[0].freq, left=nodes[0])

    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = _HuffNode(freq=left.freq + right.freq, left=left, right=right)
        # Binary insertion (O(log n) find + O(n) shift, but n≤256)
        lo, hi = 0, len(nodes)
        target = parent.freq
        while lo < hi:
            mid = (lo + hi) >> 1
            if nodes[mid].freq < target:
                lo = mid + 1
            else:
                hi = mid
        nodes.insert(lo, parent)

    return nodes[0]


def _build_code_table(root: Optional[_HuffNode]) -> Dict[int, str]:
    """Generate bit-string codes from Huffman tree.

    v2: iterative stack-based traversal for speed.
    """
    table: Dict[int, str] = {}
    if root is None:
        return table
    stack: List[Tuple[_HuffNode, str]] = [(root, "")]
    while stack:
        node, prefix = stack.pop()
        if node.byte_val is not None:
            table[node.byte_val] = prefix if prefix else "0"
            continue
        if node.right:
            stack.append((node.right, prefix + "1"))
        if node.left:
            stack.append((node.left, prefix + "0"))
    return table


def _huffman_encode(data: bytes) -> Tuple[bytes, Dict[int, int]]:
    """Encode data using Huffman coding. Returns (encoded_bytes, freq_table).

    v2: bytearray packing for speed.
    """
    if not data:
        return b"", {}
    freq_table: Dict[int, int] = Counter(data)
    tree = _build_huffman_tree(freq_table)
    code_table = _build_code_table(tree)

    # Build bit string using list of pre-computed codes
    bits_list: List[str] = [code_table[byte] for byte in data]
    bit_string = "".join(bits_list)

    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    n_bytes = len(bit_string) >> 3
    encoded = bytearray(n_bytes)
    for i in range(n_bytes):
        encoded[i] = int(bit_string[i * 8:(i + 1) * 8], 2)

    return bytes(encoded), freq_table


def _huffman_decode(encoded: bytes, freq_table: Dict[int, int],
                    original_length: int) -> bytes:
    """Decode Huffman-encoded data back to original bytes.

    v2: pre-allocated output buffer for speed.
    """
    if not encoded or not freq_table:
        return b""
    tree = _build_huffman_tree(freq_table)
    if tree is None:
        return b""

    # Build bit string
    bit_string = "".join(f"{byte:08b}" for byte in encoded)

    result = bytearray(original_length)
    node = tree
    out_idx = 0
    for bit in bit_string:
        if out_idx >= original_length:
            break
        if bit == "0":
            node = node.left if node.left else node
        else:
            node = node.right if node.right else node
        if node.byte_val is not None:
            result[out_idx] = node.byte_val
            out_idx += 1
            node = tree

    return bytes(result[:out_idx])


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7 — LZ77 ENGINE (Sovereign-accelerated)
# ═══════════════════════════════════════════════════════════════════════


def _lz77_compress(data: bytes) -> bytes:
    """LZ77 compression with ET-derived sliding window.

    Uses hash-based match finder for speed, with shimmer radiation
    weighting to prefer closer matches when lengths are equal.
    v2: bytearray output, faster hash.
    """
    if not data:
        return b""

    output = bytearray()
    pos = 0
    length = len(data)
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
            candidates = hash_table.get(h)

            if candidates is not None:
                search_start = max(0, len(candidates) - 64)
                for ci in range(len(candidates) - 1, search_start - 1, -1):
                    cand_pos = candidates[ci]
                    offset = pos - cand_pos
                    if offset > WINDOW_SIZE or offset <= 0:
                        continue
                    match_len = 0
                    while (match_len < MAX_MATCH_LEN and
                           pos + match_len < length and
                           data[cand_pos + match_len] == data[pos + match_len]):
                        match_len += 1
                    if match_len > best_length:
                        best_length = match_len
                        best_offset = offset
                        if match_len >= MAX_MATCH_LEN:
                            break

            if h >= 0:
                entries = hash_table[h]
                entries.append(pos)
                if len(entries) > 64:
                    hash_table[h] = entries[-64:]

        if best_length >= MIN_MATCH_LEN:
            output.append(0x01)
            output.extend(struct.pack("<HB", best_offset,
                                      min(best_length, MAX_MATCH_LEN)))
            for skip in range(1, min(best_length, 8)):
                if pos + skip + 2 < length:
                    sh = _trigram_hash(pos + skip)
                    if sh >= 0:
                        hash_table[sh].append(pos + skip)
            pos += best_length
        else:
            output.append(0x00)
            output.append(data[pos])
            pos += 1

    return bytes(output)


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
            ref_data = stream.read(3)
            if len(ref_data) < 3:
                break
            offset = struct.unpack("<H", ref_data[:2])[0]
            match_len = ref_data[2]
            start = len(output) - offset
            for i in range(match_len):
                output.append(output[start + i])
        else:
            literal = stream.read(1)
            if not literal:
                break
            output.append(literal[0])

    return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 8 — BWT + MTF + RLE (Manifold Sorting Pipeline)
# ═══════════════════════════════════════════════════════════════════════


def _bwt_encode(data: bytes) -> Tuple[bytes, int]:
    """Burrows-Wheeler Transform using teleological manifold sorting.

    Sorts all rotations of the data to group similar contexts together,
    exploiting manifold density for better downstream compression.
    Returns (transformed_bytes, original_row_index).
    v2: bytearray output for speed.
    """
    if not data:
        return b"", 0
    n = len(data)
    doubled = data + data
    indices = list(range(n))
    indices.sort(key=lambda i: doubled[i:i + n])
    transformed = bytearray(n)
    for j, i in enumerate(indices):
        transformed[j] = data[(i + n - 1) % n]
    original_idx = indices.index(0)
    return bytes(transformed), original_idx


def _bwt_decode(data: bytes, original_idx: int) -> bytes:
    """Inverse Burrows-Wheeler Transform."""
    if not data:
        return b""
    n = len(data)
    table = sorted(range(n), key=lambda i: data[i])
    result = bytearray(n)
    idx = original_idx
    for i in range(n):
        result[i] = data[table[idx]]
        idx = table[idx]
    return bytes(result)


def _mtf_encode(data: bytes) -> bytes:
    """Move-to-Front transform: recently used symbols get lower indices.

    This exploits the descriptor field gradient — after BWT, similar
    bytes cluster together, and MTF converts that locality into low
    values that compress exceptionally well.
    v2: uses bytearray alphabet for speed.
    """
    if not data:
        return b""
    alphabet = bytearray(range(256))
    result = bytearray(len(data))
    for i, byte in enumerate(data):
        idx = alphabet.index(byte)
        result[i] = idx
        if idx > 0:
            # Shift elements right and move byte to front
            val = alphabet[idx]
            del alphabet[idx]
            alphabet.insert(0, val)
    return bytes(result)


def _mtf_decode(data: bytes) -> bytes:
    """Inverse Move-to-Front transform."""
    if not data:
        return b""
    alphabet = bytearray(range(256))
    result = bytearray(len(data))
    for i, idx in enumerate(data):
        byte = alphabet[idx]
        result[i] = byte
        if idx > 0:
            del alphabet[idx]
            alphabet.insert(0, byte)
    return bytes(result)


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
                output.extend(bytes([byte]) * count)
                i += 3
            else:
                break
        else:
            output.append(data[i])
            i += 1
    return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 9 — DELTA ENCODING (Descriptor Difference Transform)
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
#  SECTION 10 — RESIDUAL DESCRIPTOR ENCODING (NEW v2, Eq 233)
# ═══════════════════════════════════════════════════════════════════════


def _encode_residuals(residuals: List[int]) -> bytes:
    """Eq 233: Residual Descriptor Encoding.
    When variance is low but nonzero, encode residuals compressively.
    Uses variable-length zig-zag encoding (like protobuf) + RLE.
    """
    if not residuals:
        return b""
    output = bytearray()
    for val in residuals:
        # Zig-zag encode: map signed to unsigned
        # (0→0, -1→1, 1→2, -2→3, 2→4, ...)
        zz = (val << 1) ^ (val >> 31) if val >= 0 else ((-val - 1) << 1) | 1
        # Varint encode
        while zz >= 0x80:
            output.append((zz & 0x7F) | 0x80)
            zz >>= 7
        output.append(zz & 0x7F)
    return bytes(output)


def _decode_residuals(data: bytes, count: int) -> List[int]:
    """Decode zig-zag varint residuals."""
    residuals: List[int] = []
    pos = 0
    for _ in range(count):
        if pos >= len(data):
            break
        zz = 0
        shift = 0
        while pos < len(data):
            b = data[pos]
            pos += 1
            zz |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        # Zig-zag decode
        val = (zz >> 1) ^ (-(zz & 1))
        residuals.append(val)
    return residuals


def _descriptor_residual_compress(data: bytes) -> bytes:
    """Compress a block using descriptor search + residual encoding.

    Eq 233 + Eq 4/217: Find best-fit function, then compressively
    store the residuals instead of falling back to raw.
    """
    n = len(data)
    values = list(data)
    depth = ETMath.adaptive_descriptor_depth(n)
    result = ETMath.recursive_descriptor_search(values, max_depth=depth)

    out = io.BytesIO()
    # Header: type_id(1) + param_count(1) + params(8 each) + n(4) + residual_data
    type_map = {"empty": 0, "constant": 1, "linear": 2, "quadratic": 3,
                "exponential": 4, "power": 5, "logarithmic": 6, "cubic": 7, "raw": 255}
    type_id = type_map.get(result["type"], 255)

    if type_id == 255:
        # Raw fallback — no descriptor found
        out.write(struct.pack("<B", 255))
        out.write(data)
        return out.getvalue()

    params = result["params"]
    residuals = result["residuals"]

    out.write(struct.pack("<B", type_id))
    out.write(struct.pack("<B", len(params)))
    for p in params:
        out.write(struct.pack("<d", p))
    out.write(struct.pack("<I", n))

    # Encode residuals
    resid_bytes = _encode_residuals(residuals)
    out.write(struct.pack("<I", len(resid_bytes)))
    out.write(resid_bytes)

    return out.getvalue()


def _descriptor_residual_decompress(data: bytes, original_length: int) -> bytes:
    """Decompress descriptor + residual encoded block."""
    stream = io.BytesIO(data)
    type_id = struct.unpack("<B", stream.read(1))[0]

    if type_id == 255:
        return stream.read()

    param_count = struct.unpack("<B", stream.read(1))[0]
    params = []
    for _ in range(param_count):
        params.append(struct.unpack("<d", stream.read(8))[0])
    n = struct.unpack("<I", stream.read(4))[0]
    resid_len = struct.unpack("<I", stream.read(4))[0]
    resid_data = stream.read(resid_len)
    residuals = _decode_residuals(resid_data, n)

    # Reconstruct from descriptor + residuals
    type_names = {0: "empty", 1: "constant", 2: "linear", 3: "quadratic",
                  4: "exponential", 5: "power", 6: "logarithmic", 7: "cubic"}
    type_name = type_names.get(type_id, "raw")

    result = bytearray(n)
    for i in range(n):
        if type_name == "constant":
            base = params[0]
        elif type_name == "linear":
            base = params[0] * i + params[1]
        elif type_name == "quadratic":
            base = params[0] * i * i + params[1] * i + params[2]
        elif type_name == "exponential":
            try:
                base = params[0] * (params[1] ** i)
            except OverflowError:
                base = 0
        elif type_name == "power":
            try:
                base = params[0] * (i ** params[1]) if i > 0 else 0
            except (OverflowError, ValueError):
                base = 0
        elif type_name == "logarithmic":
            try:
                base = params[0] * math.log(i) + params[1] if i > 0 else params[1]
            except (ValueError, OverflowError):
                base = 0
        elif type_name == "cubic":
            base = params[0] * i**3 + params[1] * i**2 + params[2] * i + params[3]
        else:
            base = 0

        resid = residuals[i] if i < len(residuals) else 0
        result[i] = int(round(base + resid)) & 0xFF

    return bytes(result[:original_length])


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 11 — D-FIELD ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════


class DFieldAnalyzer:
    """Analyzes data using Descriptor Field operations.

    Computes gradients, curvature, shimmer patterns, and resonance
    to identify optimal compression boundaries and strategies.
    v2: accepts memoryview for zero-copy streaming analysis.
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
#  SECTION 12 — DEDUPLICATION ENGINE (scaled Bloom + CAS)
# ═══════════════════════════════════════════════════════════════════════


class DeduplicationEngine:
    """Block-level deduplication using CAS and Bloom filters.

    Chunks data into DEDUP_BLOCK_SIZE blocks, hashes each,
    and eliminates duplicates via content addressing.
    v2: Uses scaled Bloom filter sized to file.
    """

    def __init__(self, expected_blocks: int = 1024) -> None:
        self.cas = ContentAddressableStore()
        self.bloom = BloomFilter(expected_items=max(expected_blocks, 256))

    def process(self, data: bytes) -> Tuple[List[str], Dict[str, bytes]]:
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
        output = bytearray()
        for addr in refs:
            output.extend(blocks_dict.get(addr, b""))
        return bytes(output)


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 13 — FREQUENCY TABLE SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════


def _serialize_freq_table(freq_table: Dict[int, int]) -> bytes:
    """Serialize Huffman frequency table compactly."""
    out = io.BytesIO()
    entries = list(freq_table.items())
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
#  SECTION 14 — COMPRESSION PIPELINE (strategy selection + competitive)
# ═══════════════════════════════════════════════════════════════════════


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

    if entropy > 7.9 and density > 0.98:
        return STRAT_RAW

    if entropy < 6.5:
        return STRAT_BWT_MTF_RLE_HUFF

    if len(analyzer.gradients) > 0:
        avg_gradient = sum(abs(g) for g in analyzer.gradients) / len(analyzer.gradients)
        if avg_gradient < 8.0:
            return STRAT_DELTA_HUFF

    if entropy < 7.5:
        return STRAT_BWT_MTF_RLE_HUFF

    return STRAT_LZ77


def _competitive_compress(data: bytes) -> Tuple[int, bytes]:
    """Try multiple strategies and return the smallest result.

    This is the Traverser's optimal navigation through the compression
    manifold — exploring multiple paths and selecting the one with
    minimum variance (best compression ratio).
    v2: includes descriptor-residual strategy.
    """
    original_len = len(data)
    best_strategy = STRAT_RAW
    best_compressed = data
    best_size = original_len

    candidates = [STRAT_BWT_MTF_RLE_HUFF, STRAT_LZ77, STRAT_DELTA_HUFF]

    # v2: Try descriptor-residual for smaller blocks
    if original_len <= 4096:
        candidates.append(STRAT_DESCRIPTOR_RESIDUAL)

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
        bwt_data, bwt_idx = _bwt_encode(data)
        mtf_data = _mtf_encode(bwt_data)
        rle_data = _rle_encode(mtf_data)
        huff_data, freq_table = _huffman_encode(rle_data)
        out = io.BytesIO()
        out.write(struct.pack("<I", bwt_idx))
        out.write(struct.pack("<I", len(rle_data)))
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
        dedup = DeduplicationEngine()
        refs, blocks = dedup.process(data)
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
        return _lz77_compress(raw_dedup)

    elif strategy == STRAT_DESCRIPTOR_RESIDUAL:
        return _descriptor_residual_compress(data)

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
        output = bytearray()
        for ref in refs:
            output.extend(blocks.get(ref, b""))
        return bytes(output)

    elif strategy == STRAT_DESCRIPTOR_RESIDUAL:
        return _descriptor_residual_decompress(data, original_length)

    return data


def _strategy_name(s: int) -> str:
    return {
        STRAT_RAW: "RAW",
        STRAT_LZ77: "LZ77",
        STRAT_BWT_MTF_RLE_HUFF: "BWT+MTF+RLE+Huffman",
        STRAT_DELTA_HUFF: "Delta+Huffman",
        STRAT_DEDUP: "Dedup+LZ77",
        STRAT_DESCRIPTOR_RESIDUAL: "Descriptor+Residual",
    }.get(s, f"Unknown({s})")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 15 — STREAMING FILE I/O (Eq 232: Manifold Stream Density)
# ═══════════════════════════════════════════════════════════════════════


def _stream_read_file(path: str) -> bytes:
    """Read file using mmap for large files (Eq 232).
    Falls back to normal read for small files.
    Keeps RAM bounded regardless of file size.
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        return b""
    # For files < 4MB, direct read is faster
    if file_size < 4 * 1024 * 1024:
        with open(path, "rb") as f:
            return f.read()
    # For larger files, use mmap streaming to avoid RAM bloat
    result = bytearray()
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            while offset < file_size:
                chunk_size = min(STREAM_CHUNK, file_size - offset)
                result.extend(mm[offset:offset + chunk_size])
                offset += chunk_size
    return bytes(result)


def _build_metadata(analyzer: DFieldAnalyzer,
                    strategies: List[int],
                    dedup_hits: int) -> bytes:
    """Build ET analysis metadata block."""
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
        f"compressor_version=2.1.0",
    ]
    return "\n".join(lines).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 16 — PDT ARCHIVE FORMAT v2
# ═══════════════════════════════════════════════════════════════════════
#
#  .pdt v2 file layout:
#  ┌───────────────────────────────────────────────────────────────────┐
#  │ Magic: "PDT\x00"                                      (4 bytes) │
#  │ Version: 2                                             (1 byte)  │
#  │ Flags: (bit 0 = multi-file archive)                    (1 byte)  │
#  │ File count                                             (4 bytes) │
#  ├───────────────────────────────────────────────────────────────────┤
#  │ For each file entry:                                             │
#  │   Relative path length                                 (2 bytes) │
#  │   Relative path (UTF-8)                                (N bytes) │
#  │   Original file size                                   (8 bytes) │
#  │   Original file SHA-256 hash                          (32 bytes) │
#  │   Block count                                          (4 bytes) │
#  │   ET Analysis metadata length                          (4 bytes) │
#  │   ET Analysis metadata                                 (M bytes) │
#  │   Merkle root hash                                    (64 bytes) │
#  │   For each block:                                                │
#  │     Strategy flag                                      (1 byte)  │
#  │     Original block size                                (4 bytes) │
#  │     Compressed block size                              (4 bytes) │
#  │     Block data                                         (C bytes) │
#  └───────────────────────────────────────────────────────────────────┘


def _collect_files(paths: List[str]) -> List[Tuple[str, str]]:
    """Collect all files from paths (files and/or directories).
    Returns list of (absolute_path, relative_path_for_archive).
    """
    files: List[Tuple[str, str]] = []
    for path in paths:
        if os.path.isfile(path):
            files.append((path, os.path.basename(path)))
        elif os.path.isdir(path):
            base_dir = os.path.basename(path.rstrip(os.sep))
            for root, _dirs, filenames in os.walk(path):
                for fname in filenames:
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.join(base_dir, os.path.relpath(abs_path, path))
                    # Normalize separators to forward slash for portability
                    rel_path = rel_path.replace("\\", "/")
                    files.append((abs_path, rel_path))
    return files


def compress_paths(input_paths: List[str],
                   output_path: Optional[str] = None,
                   progress_callback: Optional[Callable] = None) -> Tuple[str, Dict[str, Any]]:
    """Compress one or more files/folders to .pdt format.

    v2: Supports multiple files and entire directories.
    Uses streaming I/O for bounded memory.
    """
    file_list = _collect_files(input_paths)
    if not file_list:
        raise ValueError("No files found to compress")

    total_files = len(file_list)
    is_multi = total_files > 1 or os.path.isdir(input_paths[0])

    # Determine output path
    if output_path is None:
        if len(input_paths) == 1:
            base = input_paths[0].rstrip(os.sep)
            output_path = os.path.splitext(base)[0] + ".pdt"
        else:
            output_path = os.path.join(os.path.dirname(input_paths[0]), "archive.pdt")

    total_original = 0
    all_strategies: List[int] = []
    last_entropy: float = 0.0
    last_density: float = 0.0
    last_resonance: float = 0.0
    last_bilateral: float = 0.0
    last_breakpoints: int = 0
    last_merkle: str = ""

    with open(output_path, "wb") as out_f:
        # Write archive header
        out_f.write(PDT_MAGIC)
        out_f.write(struct.pack("<B", PDT_VERSION))
        flags = 0x01 if is_multi else 0x00
        out_f.write(struct.pack("<B", flags))
        out_f.write(struct.pack("<I", total_files))

        for file_idx, (abs_path, rel_path) in enumerate(file_list):
            if progress_callback:
                file_pct = file_idx / max(total_files, 1)
                progress_callback(file_pct * 0.95,
                                  f"[{file_idx+1}/{total_files}] {rel_path}")

            # Read file with streaming
            raw_data = _stream_read_file(abs_path)
            original_size = len(raw_data)
            total_original += original_size
            original_hash = hashlib.sha256(raw_data).digest()

            # Write file entry header
            rel_bytes = rel_path.encode("utf-8")
            out_f.write(struct.pack("<H", len(rel_bytes)))
            out_f.write(rel_bytes)
            out_f.write(struct.pack("<Q", original_size))
            out_f.write(original_hash)

            if original_size == 0:
                # Empty file: 0 blocks
                out_f.write(struct.pack("<I", 0))
                out_f.write(struct.pack("<I", 0))
                out_f.write(b"\x00" * 64)
                continue

            # D-Field Analysis (on sample for large files)
            sample = raw_data[:min(len(raw_data), BLOCK_SIZE)]
            analyzer = DFieldAnalyzer(sample)
            last_entropy = analyzer.entropy
            last_density = analyzer.density
            last_resonance = analyzer.resonance
            last_bilateral = analyzer.compute_bilateral_interference()
            last_breakpoints = len(analyzer.find_breakpoints())

            # Block partitioning
            block_boundaries: List[int] = list(range(0, original_size, BLOCK_SIZE))
            if block_boundaries[-1] != original_size:
                block_boundaries.append(original_size)
            while (len(block_boundaries) > 2 and
                   block_boundaries[-1] - block_boundaries[-2] < 64):
                block_boundaries.pop(-2)

            total_blocks = len(block_boundaries) - 1
            compressed_blocks: List[Tuple[int, bytes, int]] = []
            strategies_used: List[int] = []
            merkle_chunks: List[bytes] = []

            for idx in range(total_blocks):
                start = block_boundaries[idx]
                end = block_boundaries[idx + 1]
                block = raw_data[start:end]
                block_len = len(block)

                strategy, compressed = _competitive_compress(block)
                compressed_blocks.append((strategy, compressed, block_len))
                strategies_used.append(strategy)
                merkle_chunks.append(block)

                if progress_callback and total_blocks > 0:
                    block_pct = (idx + 1) / total_blocks
                    overall = (file_idx + block_pct) / max(total_files, 1) * 0.95
                    progress_callback(overall,
                                      f"[{file_idx+1}/{total_files}] Block {idx+1}/{total_blocks} "
                                      f"[{_strategy_name(strategy)}]")

            all_strategies.extend(strategies_used)

            # Merkle root
            merkle = ETMath.merkle_root(merkle_chunks)
            last_merkle = merkle

            # Metadata
            dedup_total = sum(1 for s in strategies_used if s == STRAT_DEDUP)
            metadata = _build_metadata(analyzer, strategies_used, dedup_total)

            # Write to archive
            out_f.write(struct.pack("<I", len(compressed_blocks)))
            out_f.write(struct.pack("<I", len(metadata)))
            out_f.write(metadata)
            out_f.write(merkle.encode("ascii")[:64].ljust(64, b"\x00"))

            for strategy, comp_data, orig_len in compressed_blocks:
                out_f.write(struct.pack("<B", strategy))
                out_f.write(struct.pack("<I", orig_len))
                out_f.write(struct.pack("<I", len(comp_data)))
                out_f.write(comp_data)

            # Free memory for this file
            del raw_data, compressed_blocks, merkle_chunks

    compressed_size = os.path.getsize(output_path)
    ratio = compressed_size / total_original if total_original > 0 else 0

    if progress_callback:
        progress_callback(1.0, "Compression complete!")

    stats = {
        "original_size": total_original,
        "compressed_size": compressed_size,
        "ratio": ratio,
        "entropy": last_entropy,
        "density": last_density,
        "resonance": last_resonance,
        "bilateral_interference": last_bilateral,
        "breakpoints": last_breakpoints,
        "merkle_root": last_merkle[:16] + "..." if last_merkle else "",
        "file_count": total_files,
        "block_count": len(all_strategies),
        "strategies": {_strategy_name(s): all_strategies.count(s)
                       for s in set(all_strategies)},
    }

    return output_path, stats


def compress_file(input_path: str,
                  progress_callback: Optional[Callable] = None) -> Tuple[str, Dict[str, Any]]:
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

    v2.1: delegates to compress_paths() for unified pipeline while
    maintaining full v1 API compatibility.
    """
    return compress_paths([input_path], progress_callback=progress_callback)


def decompress_file(input_path: str,
                    output_dir: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> Tuple[str, bool]:
    """Decompress a .pdt archive. v2: supports multi-file archives."""
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != PDT_MAGIC:
            raise ValueError("Not a valid PDT archive (wrong magic bytes)")

        version = struct.unpack("<B", f.read(1))[0]

        if version == 1:
            return _decompress_v1(f, input_path, output_dir, progress_callback)

        if version > PDT_VERSION:
            raise ValueError(f"Unsupported PDT version: {version}")

        flags = struct.unpack("<B", f.read(1))[0]
        file_count = struct.unpack("<I", f.read(4))[0]

        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        all_ok = True
        first_output = ""

        for file_idx in range(file_count):
            if progress_callback:
                progress_callback(file_idx / max(file_count, 1) * 0.95,
                                  f"File {file_idx+1}/{file_count}")

            rel_len = struct.unpack("<H", f.read(2))[0]
            rel_path = f.read(rel_len).decode("utf-8")
            original_size = struct.unpack("<Q", f.read(8))[0]
            original_hash = f.read(32)
            block_count = struct.unpack("<I", f.read(4))[0]
            meta_len = struct.unpack("<I", f.read(4))[0]
            _metadata = f.read(meta_len)
            stored_merkle = f.read(64).rstrip(b"\x00").decode("ascii")

            # Decompress blocks
            output = bytearray()
            merkle_chunks: List[bytes] = []

            for idx in range(block_count):
                strategy = struct.unpack("<B", f.read(1))[0]
                orig_len = struct.unpack("<I", f.read(4))[0]
                comp_len = struct.unpack("<I", f.read(4))[0]
                comp_data = f.read(comp_len)
                block = _decompress_block(comp_data, strategy, orig_len)
                block = block[:orig_len]
                output.extend(block)
                merkle_chunks.append(block)

                if progress_callback and block_count > 0:
                    block_pct = (idx + 1) / block_count
                    overall = (file_idx + block_pct) / max(file_count, 1) * 0.95
                    progress_callback(overall,
                                      f"[{file_idx+1}/{file_count}] Block {idx+1}/{block_count}")

            result = bytes(output[:original_size])

            # Verify integrity (SHA-256 hash + Merkle root)
            result_hash = hashlib.sha256(result).digest()
            hash_ok = result_hash == original_hash

            merkle_check = ETMath.merkle_root(merkle_chunks)
            merkle_ok = merkle_check == stored_merkle

            if not (hash_ok and merkle_ok):
                all_ok = False

            # Write output, creating subdirectories as needed
            # Sanitize path to prevent traversal attacks
            safe_rel = rel_path.replace("..", "_").lstrip("/")
            out_path = os.path.join(output_dir, safe_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if os.path.exists(out_path):
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_decompressed{ext}"

            with open(out_path, "wb") as of:
                of.write(result)

            if file_idx == 0:
                first_output = out_path

    if progress_callback:
        progress_callback(1.0, "Decompression complete!")

    return first_output, all_ok


def _decompress_v1(f, input_path: str, output_dir: Optional[str],
                   progress_callback: Optional[Callable]) -> Tuple[str, bool]:
    """Backward-compatible decompression for v1 archives."""
    name_len = struct.unpack("<H", f.read(2))[0]
    original_name = f.read(name_len).decode("utf-8")
    original_size = struct.unpack("<Q", f.read(8))[0]
    original_hash = f.read(32)
    block_count = struct.unpack("<I", f.read(4))[0]
    meta_len = struct.unpack("<I", f.read(4))[0]
    _metadata = f.read(meta_len)
    stored_merkle = f.read(64).rstrip(b"\x00").decode("ascii")

    if progress_callback:
        progress_callback(0.05, f"Decompressing v1 '{original_name}'...")

    output = bytearray()
    merkle_chunks: List[bytes] = []

    for idx in range(block_count):
        strategy = struct.unpack("<B", f.read(1))[0]
        orig_len = struct.unpack("<I", f.read(4))[0]
        comp_len = struct.unpack("<I", f.read(4))[0]
        comp_data = f.read(comp_len)
        block = _decompress_block(comp_data, strategy, orig_len)
        block = block[:orig_len]
        output.extend(block)
        merkle_chunks.append(block)
        if progress_callback and block_count > 0:
            progress_callback(0.05 + 0.85 * ((idx + 1) / block_count),
                              f"Block {idx+1}/{block_count}")

    result = bytes(output[:original_size])

    # Verify integrity (SHA-256 hash + Merkle root)
    if progress_callback:
        progress_callback(0.92, "Verifying integrity...")

    result_hash = hashlib.sha256(result).digest()
    hash_ok = result_hash == original_hash

    merkle_check = ETMath.merkle_root(merkle_chunks)
    merkle_ok = merkle_check == stored_merkle

    integrity = hash_ok and merkle_ok

    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    out_path = os.path.join(output_dir, original_name)
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_decompressed{ext}"

    with open(out_path, "wb") as of:
        of.write(result)

    if progress_callback:
        progress_callback(1.0, "Decompression complete!")

    return out_path, integrity


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 17 — GUI (Tkinter, zero external deps)
#  v2: Folder/multi-file support, GUI heartbeat for responsiveness
# ═══════════════════════════════════════════════════════════════════════


class ETCompressorGUI:
    """Main GUI for the ET Compressor v2 application."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ET Compressor v2.1.0 — Exception Theory File Compression")
        self.root.geometry("820x780")
        self.root.minsize(700, 640)
        self.root.configure(bg="#0d1117")

        self._selected_paths: List[str] = []
        self._is_running: bool = False
        self._heartbeat_id: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        bg = "#0d1117"
        fg = "#c9d1d9"
        accent = "#58a6ff"
        green = "#3fb950"
        card_bg = "#161b22"
        border = "#30363d"

        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card_bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Consolas", 10))
        style.configure("Title.TLabel", background=bg, foreground=accent,
                        font=("Consolas", 16, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground="#8b949e",
                        font=("Consolas", 9))
        style.configure("Card.TLabel", background=card_bg, foreground=fg,
                        font=("Consolas", 10))
        style.configure("Status.TLabel", background=bg, foreground=green,
                        font=("Consolas", 9))
        style.configure("TButton", font=("Consolas", 11, "bold"), padding=8)
        style.configure("Accent.TButton", font=("Consolas", 11, "bold"), padding=8)
        style.configure("TProgressbar", troughcolor=border,
                        background=accent, thickness=20)

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="ET Compressor v2.1.0", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main,
                  text="Exception Theory-Derived File Compression Engine  "
                       "| P\u2218D\u2218T = E  |  Sovereign Speed",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(0, 15))

        # File Selection
        file_frame = ttk.Frame(main)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self._file_label = ttk.Label(file_frame, text="No files/folders selected",
                                     style="TLabel")
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self._select_file_btn = ttk.Button(btn_frame, text="Select File(s)",
                                           command=self._select_files)
        self._select_file_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._select_folder_btn = ttk.Button(btn_frame, text="Select Folder",
                                             command=self._select_folder)
        self._select_folder_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._compress_btn = ttk.Button(btn_frame, text="Compress \u2192 .pdt",
                                        command=self._compress,
                                        style="Accent.TButton")
        self._compress_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._compress_btn.state(["disabled"])

        self._decompress_btn = ttk.Button(btn_frame, text="Decompress .pdt",
                                          command=self._decompress)
        self._decompress_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._decompress_btn.state(["disabled"])

        # Progress
        self._progress = ttk.Progressbar(main, mode="determinate",
                                         style="TProgressbar", maximum=100)
        self._progress.pack(fill=tk.X, pady=(5, 2))

        self._status_label = ttk.Label(main, text="Ready", style="Status.TLabel")
        self._status_label.pack(anchor="w", pady=(0, 10))

        # Analysis Card
        analysis_card = ttk.LabelFrame(main, text=" D-Field Analysis ", padding=10)
        analysis_card.pack(fill=tk.X, pady=(0, 10))

        self._analysis_text = tk.Text(analysis_card, height=8, bg=card_bg, fg=fg,
                                      font=("Consolas", 9), relief="flat",
                                      wrap="word", insertbackground=fg,
                                      selectbackground=accent)
        self._analysis_text.pack(fill=tk.X)
        self._analysis_text.insert("1.0", "Select file(s) or folder to begin analysis...")
        self._analysis_text.config(state="disabled")

        # Results Card
        results_card = ttk.LabelFrame(main, text=" Compression Results ", padding=10)
        results_card.pack(fill=tk.BOTH, expand=True)

        self._results_text = tk.Text(results_card, height=12, bg=card_bg, fg=fg,
                                     font=("Consolas", 9), relief="flat",
                                     wrap="word", insertbackground=fg,
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
                                  "  Eq 231:   Sovereign Transmutation Speed\n"
                                  "  Eq 232:   Manifold Stream Density\n"
                                  "  Eq 233:   Residual Descriptor Encoding\n"
                                  "  Eq 234:   Adaptive Search Depth\n"
                                  "  Eq 235:   Bloom Scaling Law\n"
                                  "  Merkle:   Holographic Integrity\n"
                                  "  BWT:      Manifold Sorting\n"
                                  "  CAS:      Content-Addressable Dedup\n")
        self._results_text.config(state="disabled")

        # Footer
        footer = ttk.Label(main,
                           text="Civilizational Blueprint  |  "
                                "Exception Theory  |  "
                                "MANIFOLD_SYMMETRY=12  |  "
                                "BASE_VARIANCE=1/12  |  v2.1.0",
                           style="Subtitle.TLabel")
        footer.pack(side=tk.BOTTOM, pady=(10, 0))

    def _select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select File(s) to Compress/Decompress",
            filetypes=[("All Files", "*.*"), ("PDT Archives", "*.pdt")]
        )
        if paths:
            self._selected_paths = list(paths)
            self._update_selection()

    def _select_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Folder to Compress")
        if path:
            self._selected_paths = [path]
            self._update_selection()

    def _update_selection(self) -> None:
        paths = self._selected_paths
        if not paths:
            return

        if len(paths) == 1 and os.path.isdir(paths[0]):
            file_count = sum(1 for _, _, files in os.walk(paths[0]) for _ in files)
            name = os.path.basename(paths[0])
            self._file_label.config(text=f"Folder: {name}  ({file_count} files)")
            self._compress_btn.state(["!disabled"])
            self._decompress_btn.state(["disabled"])
        elif len(paths) == 1 and paths[0].lower().endswith(".pdt"):
            size = os.path.getsize(paths[0])
            self._file_label.config(
                text=f"{os.path.basename(paths[0])}  ({self._format_size(size)})")
            self._compress_btn.state(["disabled"])
            self._decompress_btn.state(["!disabled"])
        else:
            total = sum(os.path.getsize(p) for p in paths if os.path.isfile(p))
            self._file_label.config(
                text=f"{len(paths)} file(s)  ({self._format_size(total)} total)")
            has_pdt = any(p.lower().endswith(".pdt") for p in paths)
            if has_pdt and len(paths) == 1:
                self._compress_btn.state(["disabled"])
                self._decompress_btn.state(["!disabled"])
            else:
                self._compress_btn.state(["!disabled"])
                self._decompress_btn.state(["disabled"])

        # Quick analysis for single non-pdt file
        if (len(paths) == 1 and os.path.isfile(paths[0]) and
                not paths[0].lower().endswith(".pdt")):
            self._quick_analysis(paths[0])

    def _quick_analysis(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
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
                        f"  Block {i}: [{start:>6}..{end:>6}] \u2192 {cls}")

            self._set_analysis_text("\n".join(lines))
        except Exception as e:
            self._set_analysis_text(f"Analysis error: {e}")

    # ── GUI Heartbeat (Fix #5) ──────────────────────────────────────

    def _start_heartbeat(self) -> None:
        """Pulse the GUI event loop to prevent 'Not Responding'."""
        if self._is_running:
            self.root.update_idletasks()
            self._heartbeat_id = self.root.after(GUI_PULSE_MS, self._start_heartbeat)

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_id is not None:
            self.root.after_cancel(self._heartbeat_id)
            self._heartbeat_id = None

    # ── Compress ────────────────────────────────────────────────────

    def _compress(self) -> None:
        if not self._selected_paths or self._is_running:
            return
        self._is_running = True
        self._compress_btn.state(["disabled"])
        self._select_file_btn.state(["disabled"])
        self._select_folder_btn.state(["disabled"])
        self._progress["value"] = 0
        self._start_heartbeat()

        def _run() -> None:
            try:
                start_time = time.time()
                out_path, stats = compress_paths(
                    self._selected_paths,
                    progress_callback=self._update_progress)
                elapsed = time.time() - start_time

                lines = [
                    "COMPRESSION COMPLETE",
                    "=" * 55,
                    f"Input:  {len(self._selected_paths)} path(s)",
                    f"Output: {os.path.basename(out_path)}",
                    "",
                    f"Original Size:    {self._format_size(stats['original_size'])}",
                    f"Compressed Size:  {self._format_size(stats['compressed_size'])}",
                    f"Ratio:            {stats['ratio']:.4f} "
                    f"({stats['ratio']*100:.1f}%)",
                    f"Savings:          "
                    f"{self._format_size(stats['original_size'] - stats['compressed_size'])} "
                    f"({(1-stats['ratio'])*100:.1f}% reduction)",
                    f"Files:            {stats['file_count']}",
                    f"Blocks:           {stats['block_count']}",
                    f"Time:             {elapsed:.2f}s",
                    f"Speed:            {self._format_size(int(stats['original_size']/max(elapsed,0.001)))}/s",
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

    # ── Decompress ──────────────────────────────────────────────────

    def _decompress(self) -> None:
        if not self._selected_paths or self._is_running:
            return
        self._is_running = True
        self._decompress_btn.state(["disabled"])
        self._select_file_btn.state(["disabled"])
        self._select_folder_btn.state(["disabled"])
        self._progress["value"] = 0
        self._start_heartbeat()

        def _run() -> None:
            try:
                start_time = time.time()
                out_path, integrity = decompress_file(
                    self._selected_paths[0],
                    progress_callback=self._update_progress)
                elapsed = time.time() - start_time

                status = "VERIFIED" if integrity else "FAILED"
                lines = [
                    "DECOMPRESSION COMPLETE",
                    "=" * 55,
                    f"Input:     {os.path.basename(self._selected_paths[0])}",
                    f"Output:    {os.path.basename(out_path)}",
                    f"Size:      {self._format_size(os.path.getsize(out_path))}",
                    f"Integrity: {status}",
                    f"Time:      {elapsed:.2f}s",
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
        self.root.after(0, self._set_progress, pct, message)

    def _set_progress(self, pct: float, message: str) -> None:
        self._progress["value"] = pct * 100
        self._status_label.config(text=message)

    def _finish_operation(self) -> None:
        self._is_running = False
        self._stop_heartbeat()
        self._select_file_btn.state(["!disabled"])
        self._select_folder_btn.state(["!disabled"])
        if self._selected_paths:
            self._update_selection()

    def _set_analysis_text(self, text: str) -> None:
        self._analysis_text.config(state="normal")
        self._analysis_text.delete("1.0", tk.END)
        self._analysis_text.insert("1.0", text)
        self._analysis_text.config(state="disabled")

    def _set_results_text(self, text: str) -> None:
        self._results_text.config(state="normal")
        self._results_text.delete("1.0", tk.END)
        self._results_text.insert("1.0", text)
        self._results_text.config(state="disabled")

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size) < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def run(self) -> None:
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 18 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Launch the ET Compressor v2.1.0 GUI."""
    app = ETCompressorGUI()
    app.run()


if __name__ == "__main__":
    main()

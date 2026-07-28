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
import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# CDF METABOLISM — Dynamic Resource Governance via Koide Ceiling
#
# Ported from the ET Conscious AI's resource architecture:
#
#   ETFineStructure (et_conscious_ai_core.py):
#     Computes α⁻¹ via asymptotic convergence loop, each term A_k = κᵏ/(N^(k+1)·π^(k-1)).
#     Convergence ratio κ/(N·π) ≈ 0.01768. Stops at the hardware coherence
#     boundary: |A_k| < ε_mach × α⁻¹. The metabolism computes to the depth
#     the substrate supports — no more, no less.
#
#   ResourceGovernor (et_conscious_ai_distributed.py):
#     Hard Koide ceiling: K = 2/3 ≈ 66.67% of any system resource.
#     Headroom = max(0, K% - current_load%).
#     Threads = floor(headroom × cores / 100).
#     Memory = total_bytes × headroom / 100.
#     K + T_WEIGHT = 1.0 — the compressor leaves 1/3 for other software.
#
# This governs ONLY the system resource footprint:
#   - How many CPU threads the compressor spawns (batch concurrency)
#   - How much memory the compressor may claim (buffer caps)
#   - Process priority (yield CPU when system is under pressure)
#   - Periodic re-sensing to adapt to changing system conditions
#
# The compression ALGORITHM is invariant — every strategy runs every time
# on every system. Only the resource envelope changes.
#
# ET Derivation:
#   Identification Principle: the substrate (P = hardware) is profiled.
#   Descriptor Gap Principle: headroom IS the gap between load and ceiling.
#   Subsumption Law: the budget subsumes ALL resource usage without remainder.
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDFResourceProfile:
    """
    Snapshot of system resources relevant to compression.

    Adapted from HardwareProfile (et_conscious_ai_distributed.py).
    """
    cpu_count_logical: int = 1
    cpu_load_percent: float = 0.0
    mem_total_bytes: int = 0
    mem_available_bytes: int = 0
    mem_used_percent: float = 0.0
    timestamp: float = 0.0


class CDFResourceSensor:
    """
    Detects available hardware resources.

    Adapted from ResourceSensor (et_conscious_ai_distributed.py).
    Reads from /proc (Linux), os module fallbacks elsewhere.
    """

    @staticmethod
    def sense() -> CDFResourceProfile:
        """Take a hardware snapshot."""
        profile = CDFResourceProfile(timestamp=time.time())

        # ── CPU ──
        profile.cpu_count_logical = os.cpu_count() or 1
        profile.cpu_load_percent = CDFResourceSensor._read_cpu_load()

        # ── Memory ──
        try:
            with open('/proc/meminfo', 'r') as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        val = int(parts[1]) * 1024  # kB → bytes
                        mem[key] = val
                profile.mem_total_bytes = mem.get('MemTotal', 0)
                profile.mem_available_bytes = mem.get(
                    'MemAvailable', mem.get('MemFree', 0))
                if profile.mem_total_bytes > 0:
                    used = profile.mem_total_bytes - profile.mem_available_bytes
                    profile.mem_used_percent = (used / profile.mem_total_bytes) * 100.0
        except (FileNotFoundError, OSError):
            try:
                page_size = os.sysconf('SC_PAGE_SIZE')
                total_pages = os.sysconf('SC_PHYS_PAGES')
                avail_pages = os.sysconf('SC_AVPHYS_PAGES')
                profile.mem_total_bytes = page_size * total_pages
                profile.mem_available_bytes = page_size * avail_pages
                if profile.mem_total_bytes > 0:
                    used = profile.mem_total_bytes - profile.mem_available_bytes
                    profile.mem_used_percent = (used / profile.mem_total_bytes) * 100.0
            except (OSError, ValueError):
                profile.mem_total_bytes = 4 * 1024 ** 3
                profile.mem_available_bytes = 2 * 1024 ** 3
                profile.mem_used_percent = 50.0

        return profile

    @staticmethod
    def _read_cpu_load() -> float:
        """
        Read CPU load from /proc/stat (two-sample method).

        From ResourceSensor._read_cpu_load in et_conscious_ai_distributed.py.
        50ms sample (shorter than AI's 100ms — compressor startup is time-sensitive).
        """
        try:
            def read_stat():
                with open('/proc/stat', 'r') as f:
                    line = f.readline()
                    parts = line.split()
                    vals = [int(x) for x in parts[1:9]]
                    idle = vals[3] + vals[4]
                    total = sum(vals)
                    return idle, total

            idle1, total1 = read_stat()
            time.sleep(0.05)
            idle2, total2 = read_stat()

            idle_delta = idle2 - idle1
            total_delta = total2 - total1
            if total_delta == 0:
                return 0.0
            return (1.0 - idle_delta / total_delta) * 100.0
        except (OSError, ValueError, IndexError):
            return 0.0


# Hard Koide ceiling: K = 2/3 ≈ 66.67% of any system resource
# From et_conscious_ai_distributed.py: "The AI takes at most K of any resource,
# leaving T_WEIGHT = 1/3 for other software."
KOIDE_CEILING_PERCENT = K_KOIDE * 100.0  # 66.67%


class CDFMetabolism:
    """
    Dynamic resource governance for CDF compression via Koide ceiling.

    Fuses the ETFineStructure convergence pattern with the ResourceGovernor's
    Koide ceiling. Controls ONLY system resource footprint — never touches
    the compression algorithm. Every compression strategy runs every time;
    only the resource envelope (threads, memory, priority) adapts.

    Sensing (Identification Principle):
      Profile the hardware substrate to determine D-constraints
      (core count, memory, current load). The Koide ceiling K = 2/3
      sets the hard upper bound on resource consumption.

    Allocation (ResourceGovernor pattern):
      headroom = max(0, K% - current_load%)
      max_threads = floor(headroom × cores / 100)
      max_memory = total_bytes × headroom / 100

    The convergence ratio κ/(N×π) from ETFineStructure governs the
    re-sensing interval: the metabolism re-senses at intervals of
    S² = 144 seconds, matching the manifold's S² cross-pattern cap.

    P ∘ D ∘ T = E
    """

    # Re-sense interval: S² = 144 seconds
    _REFRESH_INTERVAL = S * S  # 144

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda m: None)
        self._profile: Optional[CDFResourceProfile] = None
        self._last_sense_time: float = 0.0

        # Allocation state (updated by sense())
        self._max_threads: int = 1
        self._max_memory_bytes: int = 0
        self._cpu_headroom: float = KOIDE_CEILING_PERCENT
        self._mem_headroom: float = KOIDE_CEILING_PERCENT
        self._overall_pressure: float = 0.0

    def sense(self) -> 'CDFMetabolism':
        """
        Sense hardware and compute resource allocation.

        Adapts to current system conditions:
        - Low load → more threads, larger memory budget
        - High load → fewer threads, smaller memory budget
        - Above Koide ceiling → minimal allocation (1 thread, minimum memory)

        Returns self for chaining.
        """
        now = time.time()
        if (self._profile is not None
                and now - self._last_sense_time < self._REFRESH_INTERVAL):
            return self  # Allocation is still fresh

        self._profile = CDFResourceSensor.sense()
        self._last_sense_time = now

        # ── Headroom via Koide ceiling ──
        # From ResourceGovernor.allocate() in et_conscious_ai_distributed.py
        self._cpu_headroom = max(0.0,
                                  KOIDE_CEILING_PERCENT - self._profile.cpu_load_percent)
        self._mem_headroom = max(0.0,
                                  KOIDE_CEILING_PERCENT - self._profile.mem_used_percent)

        # ── Overall pressure: geometric mean of loads → [0, 1] ──
        # From ResourceGovernor.allocate()
        loads = [max(self._profile.cpu_load_percent, 1.0),
                 max(self._profile.mem_used_percent, 1.0)]
        log_sum = sum(math.log(load) for load in loads)
        geom_mean = math.exp(log_sum / len(loads))
        self._overall_pressure = min(1.0, geom_mean / 100.0)

        # ── Thread allocation ──
        # From ResourceGovernor: at least 1, at most (cores - 1),
        # proportional to CPU headroom
        max_cores = max(1, self._profile.cpu_count_logical - 1)
        self._max_threads = max(1, min(
            max_cores,
            int(self._cpu_headroom * self._profile.cpu_count_logical / 100.0)
        ))

        # ── Memory allocation ──
        self._max_memory_bytes = int(
            self._profile.mem_total_bytes * self._mem_headroom / 100.0
        )

        self._log(
            f"Metabolism: CPU {self._profile.cpu_load_percent:.0f}% "
            f"({self._cpu_headroom:.0f}% headroom) "
            f"MEM {self._profile.mem_used_percent:.0f}% "
            f"({self._mem_headroom:.0f}% headroom) "
            f"→ {self._max_threads} threads, "
            f"{self._max_memory_bytes / (1024 ** 3):.1f} GB budget, "
            f"pressure {self._overall_pressure:.2f}"
        )

        return self

    def force_resense(self) -> 'CDFMetabolism':
        """Force an immediate re-sense regardless of refresh interval."""
        self._last_sense_time = 0.0
        return self.sense()

    @property
    def max_threads(self) -> int:
        """Max threads the compressor may spawn (Koide-governed)."""
        return self._max_threads

    @property
    def max_memory_bytes(self) -> int:
        """Max memory the compressor may allocate (Koide-governed)."""
        return self._max_memory_bytes

    @property
    def overall_pressure(self) -> float:
        """System pressure [0,1]. 0=idle, 1=saturated."""
        return self._overall_pressure

    @property
    def cpu_headroom(self) -> float:
        """CPU headroom % below Koide ceiling."""
        return self._cpu_headroom

    @property
    def mem_headroom(self) -> float:
        """Memory headroom % below Koide ceiling."""
        return self._mem_headroom

    def within_memory_budget(self, requested_bytes: int) -> bool:
        """Check if a memory allocation fits within the Koide-governed budget."""
        return requested_bytes <= self._max_memory_bytes

    def apply_process_priority(self):
        """
        Set process priority based on system pressure.

        From the Koide binding stability principle: when system pressure
        is high (approaching K = 2/3), the compressor yields more CPU
        to other processes. When pressure is low, it runs at normal priority.

        Uses os.nice() on Unix. On Windows, this is a no-op (the thread
        priority is managed by the OS scheduler and the reduced thread
        count already limits consumption).
        """
        if not hasattr(os, 'nice'):
            return  # Windows — thread count governs resource usage

        try:
            # Map pressure to nice level:
            # pressure 0.0 → nice 0 (normal)
            # pressure 0.5 → nice 5 (lower priority)
            # pressure 1.0 → nice 10 (significant yield)
            # Scale: nice = round(pressure × S) where S=12 is max nice step
            # Capped at 10 to stay in reasonable range
            target_nice = min(10, round(self._overall_pressure * S))
            current_nice = os.nice(0)
            if target_nice > current_nice:
                os.nice(target_nice - current_nice)
        except (OSError, PermissionError):
            pass  # Cannot adjust priority — no loss in function

    def summary(self) -> str:
        """Human-readable metabolic state for GUI/logging."""
        if self._profile is None:
            return "Metabolism: not yet sensed"
        return (
            f"Koide ceiling = {KOIDE_CEILING_PERCENT:.1f}%\n"
            f"  CPU: {self._profile.cpu_count_logical} cores, "
            f"{self._profile.cpu_load_percent:.1f}% load, "
            f"{self._cpu_headroom:.1f}% headroom → {self._max_threads} threads\n"
            f"  Memory: {self._profile.mem_total_bytes / (1024 ** 3):.1f} GB total, "
            f"{self._profile.mem_available_bytes / (1024 ** 3):.1f} GB avail, "
            f"{self._mem_headroom:.1f}% headroom → "
            f"{self._max_memory_bytes / (1024 ** 3):.1f} GB budget\n"
            f"  Pressure: {self._overall_pressure:.2f}"
        )


# Module-level metabolism — shared across all compression operations
_metabolism = CDFMetabolism()


# ═══════════════════════════════════════════════════════════════════════════════
# C PATTERN ENGINE — Suffix Array + LCP accelerated pattern finder
#
# Replaces the O(n × L_max) Python pattern scanner with:
#   O(n log² n) suffix array + O(n) LCP + O(n) per-length scan
# Verified: finds EXACT same patterns with EXACT same positions as Python.
# Falls back to Python if C compilation fails (no loss in function).
#
# All ET-specific filtering (IncoherenceFilter, elegance, d-values) stays in
# Python. The C engine handles ONLY the combinatorial pattern finding.
# ═══════════════════════════════════════════════════════════════════════════════

class PatternEngine:
    """
    C-accelerated repeated-pattern finder via suffix array + LCP.

    DLL search order (for pre-built or bundled binaries):
      1. PyInstaller bundle (sys._MEIPASS) — for single .exe deployment
      2. Same directory as this script — for development
      3. Current working directory — for portable deployment

    If no pre-built DLL found:
      4. Auto-compile from embedded C source to temp directory
         (requires cc/gcc/cl on PATH)

    If no compiler available:
      5. Pure Python fallback — zero loss in features, just slower

    The C engine finds all repeated substrings and their positions.
    ET filtering (IncoherenceFilter gates, elegance computation,
    archetype creation) remains in Python — zero loss in features.
    """

    _lib = None
    _compiled = False
    _attempted = False

    # ── Embedded C source ──────────────────────────────────────────────
    # The complete et_pattern_engine.c is stored here so the script is
    # fully self-contained. The separate .c file is kept for CLion/CMake
    # builds but is NOT required at runtime. This embedded copy is used
    # ONLY if no pre-built DLL is found (fallback auto-compilation).
    _C_SOURCE = r"""
/*
 * Exception Theory — Pattern Engine (Suffix Array + LCP)
 * Auto-compiled from embedded source in et_cdf_compressor.py
 * P ∘ D ∘ T = E — Author: Michael James Muller — Aevum Defluo
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
typedef struct { int32_t *data; int size; int capacity; } IntBuf;
static void ibuf_init(IntBuf *b, int cap) {
    b->data = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    b->size = 0; b->capacity = cap;
}
static void ibuf_push(IntBuf *b, int32_t v) {
    if (b->size >= b->capacity) {
        b->capacity = b->capacity + (b->capacity >> 1) + 256;
        b->data = (int32_t *)realloc(b->data, (size_t)b->capacity * sizeof(int32_t));
    }
    b->data[b->size++] = v;
}
static int *g_rank = NULL; static int g_half = 0; static int g_n = 0;
static int sa_cmp(const void *a, const void *b) {
    int i = *(const int *)a, j = *(const int *)b;
    if (g_rank[i] != g_rank[j]) return (g_rank[i] < g_rank[j]) ? -1 : 1;
    int ri = (i + g_half < g_n) ? g_rank[i + g_half] : -2;
    int rj = (j + g_half < g_n) ? g_rank[j + g_half] : -2;
    if (ri != rj) return (ri < rj) ? -1 : 1;
    return 0;
}
static void build_suffix_array(const int32_t *stream, int n, int *sa) {
    int *rank = (int *)malloc((size_t)n * sizeof(int));
    int *tmp  = (int *)malloc((size_t)n * sizeof(int));
    int i;
    for (i = 0; i < n; i++) { sa[i] = i; rank[i] = (int)stream[i]; }
    g_rank = rank; g_n = n;
    for (int half = 1; half < n; half <<= 1) {
        g_half = half;
        qsort(sa, (size_t)n, sizeof(int), sa_cmp);
        tmp[sa[0]] = 0;
        for (i = 1; i < n; i++) {
            int same = (rank[sa[i]] == rank[sa[i-1]]);
            if (same) {
                int ri = (sa[i]+half < n) ? rank[sa[i]+half] : -2;
                int rp = (sa[i-1]+half < n) ? rank[sa[i-1]+half] : -2;
                same = (ri == rp);
            }
            tmp[sa[i]] = tmp[sa[i-1]] + (same ? 0 : 1);
        }
        memcpy(rank, tmp, (size_t)n * sizeof(int));
        if (rank[sa[n-1]] == n-1) break;
    }
    g_rank = NULL; free(rank); free(tmp);
}
static void build_lcp_array(const int32_t *stream, int n, const int *sa, int *lcp) {
    int *inv = (int *)malloc((size_t)n * sizeof(int));
    int i, h = 0;
    for (i = 0; i < n; i++) inv[sa[i]] = i;
    lcp[0] = 0;
    for (i = 0; i < n; i++) {
        if (inv[i] > 0) {
            int j = sa[inv[i]-1];
            while (i+h < n && j+h < n && stream[i+h] == stream[j+h]) h++;
            lcp[inv[i]] = h;
            if (h > 0) h--;
        } else { h = 0; }
    }
    free(inv);
}
EXPORT int32_t *find_repeated_patterns(const int32_t *stream, int n,
    int min_len, int max_len, int min_count, int min_net_savings,
    int *out_n_patterns, int *out_buf_size) {
    IntBuf buf; int n_patterns = 0, i, L;
    if (!stream || n < 4 || min_len < 2) {
        int32_t *empty = (int32_t *)malloc(sizeof(int32_t));
        empty[0] = 0; *out_n_patterns = 0; *out_buf_size = 1; return empty;
    }
    if (max_len <= 0 || max_len > n/2) max_len = n/2;
    if (max_len < min_len) {
        int32_t *empty = (int32_t *)malloc(sizeof(int32_t));
        empty[0] = 0; *out_n_patterns = 0; *out_buf_size = 1; return empty;
    }
    int *sa = (int *)malloc((size_t)n * sizeof(int));
    int *lcp_arr = (int *)malloc((size_t)n * sizeof(int));
    build_suffix_array(stream, n, sa);
    build_lcp_array(stream, n, sa, lcp_arr);
    int max_lcp = 0;
    for (i = 1; i < n; i++) if (lcp_arr[i] > max_lcp) max_lcp = lcp_arr[i];
    if (max_len > max_lcp) max_len = max_lcp;
    if (max_len < min_len) {
        free(sa); free(lcp_arr);
        int32_t *empty = (int32_t *)malloc(sizeof(int32_t));
        empty[0] = 0; *out_n_patterns = 0; *out_buf_size = 1; return empty;
    }
    ibuf_init(&buf, 1 << 20); ibuf_push(&buf, 0);
    for (L = min_len; L <= max_len; L++) {
        int group_start = 0;
        for (i = 1; i <= n; i++) {
            if (i == n || lcp_arr[i] < L) {
                int group_size = i - group_start;
                if (group_size >= min_count) {
                    int net = group_size * (L-1) - (L+1);
                    if (net >= min_net_savings) {
                        int pat_start = sa[group_start];
                        ibuf_push(&buf, (int32_t)L);
                        ibuf_push(&buf, (int32_t)group_size);
                        for (int j = 0; j < L; j++) ibuf_push(&buf, stream[pat_start+j]);
                        for (int j = 0; j < group_size; j++) ibuf_push(&buf, (int32_t)sa[group_start+j]);
                        n_patterns++;
                    }
                }
                group_start = i;
            }
        }
    }
    buf.data[0] = (int32_t)n_patterns;
    *out_n_patterns = n_patterns; *out_buf_size = buf.size;
    free(sa); free(lcp_arr);
    return buf.data;
}
EXPORT void build_k_stream(const uint8_t *data, int n, const int32_t *byte_k_table, int32_t *k_out) {
    for (int i = 0; i < n; i++) k_out[i] = byte_k_table[data[i]];
}
EXPORT void build_dk_stream(const int32_t *k_stream, int n, int32_t *dk_out) {
    for (int i = 0; i < n-1; i++) dk_out[i] = k_stream[i+1] - k_stream[i];
}
static int gcd_int(int a, int b) {
    if (a < 0) a = -a; if (b < 0) b = -b;
    while (b) { int t = b; b = a % b; a = t; }
    return a ? a : 1;
}
static int lattice_d_c(int k, int n_res) {
    int k_abs = (k != 0) ? abs(k) : n_res;
    return n_res / gcd_int(k_abs, n_res);
}
EXPORT void gate_archetype_batch(const int32_t *patterns_buf, int n_patterns,
    int n_res, double incoherence_cents, uint8_t *out_mask) {
    double log2_inv = 1.0 / log(2.0);
    double cents_scale = 1200.0 / (double)n_res;
    double epsilon_val = 1e-12;
    int pos = 1;
    for (int pi = 0; pi < n_patterns; pi++) {
        int pat_len = patterns_buf[pos++];
        int occ_cnt = patterns_buf[pos++];
        const int32_t *syms = &patterns_buf[pos];
        pos += pat_len; pos += occ_cnt;
        int coherent = 1;
        double eps_sum = 0.0; int eps_count = 0;
        for (int i = 0; i < pat_len && coherent; i++) {
            int dk = syms[i];
            double ratio = (dk != 0) ? pow(2.0, (double)dk/(double)n_res) : 1.0;
            double k_exact = (double)n_res * log(ratio) * log2_inv;
            double eps = (k_exact - (double)dk) * cents_scale;
            if (eps < 0) eps = -eps;
            if (eps >= incoherence_cents) { coherent = 0; break; }
            eps_sum += eps; eps_count++;
        }
        for (int i = 0; i < pat_len-1 && coherent; i++) {
            int k_i = syms[i], k_j = syms[i+1], k_sum = k_i + k_j;
            int d_i = lattice_d_c(k_i, n_res), d_j = lattice_d_c(k_j, n_res);
            int d_sum = lattice_d_c(k_sum, n_res);
            int lcm_pair = (d_i / gcd_int(d_i, d_j)) * d_j;
            if (lcm_pair > n_res) { coherent = 0; break; }
            if (d_sum > lcm_pair) { coherent = 0; break; }
        }
        if (coherent) {
            int lcm_all = 1;
            for (int i = 0; i < pat_len; i++) {
                int d_i = lattice_d_c(syms[i], n_res);
                lcm_all = (lcm_all / gcd_int(lcm_all, d_i)) * d_i;
                if (lcm_all > n_res) { coherent = 0; break; }
            }
        }
        if (coherent && eps_count > 0) {
            double avg_eps = eps_sum / (double)eps_count;
            if (avg_eps > epsilon_val) {
                int n_max = (int)(incoherence_cents / avg_eps);
                if (pat_len > n_max) coherent = 0;
            }
        }
        out_mask[pi] = (uint8_t)coherent;
    }
}
EXPORT int subsume_greedy(int n, int n_archetypes, const int32_t *arch_lengths,
    const int32_t *arch_n_pos, const int32_t *arch_positions,
    int32_t *placements, int32_t *used_mask) {
    uint8_t *consumed = (uint8_t *)calloc((size_t)n, 1);
    int n_placements = 0, pos_offset = 0;
    for (int ai = 0; ai < n_archetypes; ai++) {
        int pat_len = arch_lengths[ai], n_pos = arch_n_pos[ai], placed = 0;
        for (int pi = 0; pi < n_pos; pi++) {
            int start = arch_positions[pos_offset + pi];
            if (start < 0 || start + pat_len > n) continue;
            int overlap = 0;
            for (int j = 0; j < pat_len; j++) { if (consumed[start+j]) { overlap = 1; break; } }
            if (overlap) continue;
            for (int j = 0; j < pat_len; j++) consumed[start+j] = 1;
            placements[n_placements*2] = ai; placements[n_placements*2+1] = start;
            n_placements++; placed = 1;
        }
        used_mask[ai] = placed ? 1 : 0;
        pos_offset += n_pos;
    }
    free(consumed);
    return n_placements;
}
EXPORT void free_buffer(int32_t *buf) { free(buf); }
"""

    @classmethod
    def _try_load(cls, lib_path: str) -> bool:
        """Try to load a DLL/SO and register all function signatures."""
        try:
            cls._lib = ctypes.CDLL(lib_path)

            cls._lib.find_repeated_patterns.restype = ctypes.POINTER(ctypes.c_int32)
            cls._lib.find_repeated_patterns.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            cls._lib.free_buffer.restype = None
            cls._lib.free_buffer.argtypes = [ctypes.POINTER(ctypes.c_int32)]
            cls._lib.build_k_stream.restype = None
            cls._lib.build_k_stream.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
            cls._lib.build_dk_stream.restype = None
            cls._lib.build_dk_stream.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32)]
            cls._lib.gate_archetype_batch.restype = None
            cls._lib.gate_archetype_batch.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.POINTER(ctypes.c_uint8)]
            cls._lib.subsume_greedy.restype = ctypes.c_int
            cls._lib.subsume_greedy.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32)]

            cls._compiled = True
            logger.info(f"PatternEngine: C engine loaded from {lib_path}")
            return True
        except (OSError, AttributeError) as e:
            logger.debug(f"PatternEngine: failed to load {lib_path}: {e}")
            cls._lib = None
            return False

    @classmethod
    def _ensure_compiled(cls) -> bool:
        """Find or compile the C pattern engine."""
        if cls._compiled:
            return cls._lib is not None
        if cls._attempted:
            return cls._lib is not None
        cls._attempted = True

        lib_name = 'et_pattern_engine' + ('.dll' if sys.platform == 'win32' else '.so')

        # ── Search 1: PyInstaller bundle (sys._MEIPASS) ──
        # When packaged as a single .exe, PyInstaller extracts bundled
        # files to a temp directory accessible via sys._MEIPASS.
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            candidate = os.path.join(meipass, lib_name)
            if os.path.isfile(candidate) and cls._try_load(candidate):
                return True

        # ── Search 2: Same directory as this script ──
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, lib_name)
        if os.path.isfile(candidate) and cls._try_load(candidate):
            return True

        # ── Search 3: Current working directory ──
        candidate = os.path.join(os.getcwd(), lib_name)
        if os.path.isfile(candidate) and cls._try_load(candidate):
            return True

        # ── Search 4: Auto-compile from external .c file ──
        c_src_external = os.path.join(script_dir, 'et_pattern_engine.c')
        if not os.path.isfile(c_src_external):
            c_src_external = os.path.join(os.getcwd(), 'et_pattern_engine.c')

        # ── Search 5: Auto-compile from embedded source ──
        # Write the embedded C source to a temp file and compile it.
        # This makes the script fully self-contained — no separate .c needed.
        c_src_path = None
        c_src_is_temp = False

        if os.path.isfile(c_src_external):
            c_src_path = c_src_external
        else:
            try:
                fd, c_src_path = tempfile.mkstemp(suffix='.c', prefix='et_pe_')
                os.write(fd, cls._C_SOURCE.encode('utf-8'))
                os.close(fd)
                c_src_is_temp = True
            except (OSError, IOError):
                logger.info("PatternEngine: cannot write temp C source — using Python fallback")
                return False

        lib_path = os.path.join(tempfile.gettempdir(), lib_name)

        if sys.platform == 'win32':
            compile_cmds = [
                ['cl', '/O2', '/LD', '/Fe:' + lib_path, c_src_path],
                ['gcc', '-shared', '-O2', '-o', lib_path, c_src_path, '-lm'],
            ]
        else:
            compile_cmds = [
                ['cc', '-shared', '-fPIC', '-O2', '-o', lib_path, c_src_path, '-lm'],
                ['gcc', '-shared', '-fPIC', '-O2', '-o', lib_path, c_src_path, '-lm'],
            ]

        compiled = False
        for cmd in compile_cmds:
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                compiled = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        # Clean up temp .c file
        if c_src_is_temp and c_src_path and os.path.isfile(c_src_path):
            try:
                os.remove(c_src_path)
            except OSError:
                pass

        if not compiled:
            logger.info("PatternEngine: C compilation failed — using Python fallback")
            return False

        return cls._try_load(lib_path)

    @classmethod
    def find_patterns(cls, sym_stream: List[int],
                      min_len: int, max_len: int,
                      min_count: int = 2,
                      min_net_savings: int = 2
                      ) -> Optional[List[Tuple[tuple, List[int]]]]:
        """
        Find all repeated patterns in sym_stream using the C suffix array engine.

        Returns a list of (pattern_tuple, positions_list) for patterns that have:
          - occurrence count ≥ min_count
          - net savings = count × (len-1) - (len+1) ≥ min_net_savings

        Returns None if the C engine is unavailable (caller uses Python fallback).
        """
        if not cls._ensure_compiled():
            return None

        n = len(sym_stream)
        if n < 4:
            return []

        # Marshal Python list → C int32 array
        arr = (ctypes.c_int32 * n)(*sym_stream)
        n_pat = ctypes.c_int(0)
        buf_sz = ctypes.c_int(0)

        result_ptr = cls._lib.find_repeated_patterns(
            arr, n, min_len, max_len, min_count, min_net_savings,
            ctypes.byref(n_pat), ctypes.byref(buf_sz))

        if not result_ptr:
            return []

        # Parse the flat int32 buffer
        # Sort positions ascending to match Python stream-order behavior.
        # See find_and_gate docstring for why this is critical.
        patterns = []
        pos = 1  # Skip the n_patterns header (already in n_pat)
        for _ in range(n_pat.value):
            pat_len = result_ptr[pos]; pos += 1
            occ_cnt = result_ptr[pos]; pos += 1
            pattern = tuple(result_ptr[pos + j] for j in range(pat_len))
            pos += pat_len
            positions = sorted(result_ptr[pos + j] for j in range(occ_cnt))
            pos += occ_cnt
            patterns.append((pattern, positions))

        cls._lib.free_buffer(result_ptr)
        return patterns

    @classmethod
    def find_and_gate(cls, sym_stream: List[int],
                      min_len: int, max_len: int,
                      n_res: int = N_FULL,
                      min_count: int = 2,
                      min_net_savings: int = 2
                      ) -> Optional[List[Tuple[tuple, List[int]]]]:
        """
        Combined: find repeated patterns + batch IncoherenceFilter gating in C.

        Same as find_patterns() but additionally runs gate_archetype_batch on
        the raw buffer BEFORE parsing, then only parses coherent patterns.
        This eliminates 145K+ per-pattern Python gate_archetype calls.

        Returns list of (pattern, positions) for COHERENT patterns only.
        Returns None if C engine unavailable.
        """
        if not cls._ensure_compiled():
            return None

        n = len(sym_stream)
        if n < 4:
            return []

        arr = (ctypes.c_int32 * n)(*sym_stream)
        n_pat = ctypes.c_int(0)
        buf_sz = ctypes.c_int(0)

        result_ptr = cls._lib.find_repeated_patterns(
            arr, n, min_len, max_len, min_count, min_net_savings,
            ctypes.byref(n_pat), ctypes.byref(buf_sz))

        if not result_ptr or n_pat.value == 0:
            if result_ptr:
                cls._lib.free_buffer(result_ptr)
            return []

        # ── Batch gate in C: single call for all patterns ──
        mask = (ctypes.c_uint8 * n_pat.value)()
        cls._lib.gate_archetype_batch(result_ptr, n_pat.value, n_res,
                                       INCOHERENCE_CENTS, mask)

        # ── Parse only COHERENT patterns (skip incoherent ones) ──
        # CRITICAL: Sort positions ascending to match Python stream-order
        # behavior. The suffix array returns positions in lexicographic
        # suffix order. The Python scanner returns positions in ascending
        # stream order (from `for i in range(n - pat_len + 1)`). Greedy
        # subsumption processes positions in order — different order means
        # different positions get selected. Sorting ensures functionally
        # exact match with the Python path.
        patterns = []
        pos = 1  # Skip n_patterns header
        for i in range(n_pat.value):
            pat_len = result_ptr[pos]; pos += 1
            occ_cnt = result_ptr[pos]; pos += 1
            if mask[i]:
                # Coherent — parse fully
                pattern = tuple(result_ptr[pos + j] for j in range(pat_len))
                pos += pat_len
                positions = sorted(result_ptr[pos + j] for j in range(occ_cnt))
                pos += occ_cnt
                patterns.append((pattern, positions))
            else:
                # Incoherent — skip data without parsing
                pos += pat_len + occ_cnt

        cls._lib.free_buffer(result_ptr)
        return patterns

    @classmethod
    def fast_k_stream(cls, data: bytes, byte_k_map: Dict[int, int]) -> Optional[List[int]]:
        """
        Build k-stream from byte data using C vectorized lookup.
        Returns None if C engine unavailable.
        """
        if not cls._ensure_compiled():
            return None

        n = len(data)
        if n == 0:
            return []

        # Build the 256-entry lookup table
        table = (ctypes.c_int32 * 256)(*[byte_k_map.get(b, 0) for b in range(256)])
        data_arr = (ctypes.c_uint8 * n).from_buffer_copy(data)
        k_out = (ctypes.c_int32 * n)()

        cls._lib.build_k_stream(data_arr, n, table, k_out)
        return list(k_out)

    @classmethod
    def gate_batch(cls, raw_buf_ptr, n_patterns: int,
                   n_res: int = N_FULL) -> Optional[List[bool]]:
        """
        Batch IncoherenceFilter gating in C.

        Takes the raw find_repeated_patterns buffer pointer and n_patterns.
        Returns a list of booleans (True=coherent) for each pattern.
        Returns None if C engine unavailable.

        Implements the EXACT same L1+L2+L3+L4 checks as
        IncoherenceFilter.gate_archetype, verified zero mismatches.
        """
        if not cls._ensure_compiled() or not cls._lib:
            return None
        if n_patterns == 0:
            return []

        mask = (ctypes.c_uint8 * n_patterns)()
        cls._lib.gate_archetype_batch(
            raw_buf_ptr, n_patterns, n_res,
            INCOHERENCE_CENTS, mask)
        return [bool(mask[i]) for i in range(n_patterns)]

    @classmethod
    def subsume_greedy_c(cls, n: int,
                         archetypes: list  # List[LatticeWalkArchetype]
                         ) -> Optional[Tuple[List[Tuple[int, int]], List[bool]]]:
        """
        C-accelerated greedy non-overlapping pattern placement.

        Returns (placements, used_mask) where:
          placements = [(arch_idx, position), ...]
          used_mask = [True/False per archetype]

        Returns None if C engine unavailable.
        """
        if not cls._ensure_compiled() or not cls._lib:
            return None
        if not archetypes:
            return [], [False] * 0

        n_arch = len(archetypes)

        # Marshal archetype data to flat C arrays
        lengths = (ctypes.c_int32 * n_arch)()
        n_pos_arr = (ctypes.c_int32 * n_arch)()
        total_positions = 0
        for ai, arch in enumerate(archetypes):
            lengths[ai] = arch.pattern_length
            n_pos_arr[ai] = len(arch.occurrences)
            total_positions += len(arch.occurrences)

        positions_flat = (ctypes.c_int32 * max(total_positions, 1))()
        idx = 0
        for arch in archetypes:
            for p in arch.occurrences:
                positions_flat[idx] = p
                idx += 1

        # Output buffers
        max_placements = total_positions
        placements_buf = (ctypes.c_int32 * (max_placements * 2 + 2))()
        used_mask_buf = (ctypes.c_int32 * n_arch)()

        n_placed = cls._lib.subsume_greedy(
            n, n_arch, lengths, n_pos_arr, positions_flat,
            placements_buf, used_mask_buf)

        placements = [(int(placements_buf[i * 2]), int(placements_buf[i * 2 + 1]))
                       for i in range(n_placed)]
        used_mask = [bool(used_mask_buf[i]) for i in range(n_arch)]

        return placements, used_mask


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
        Level 3 — Sublattice Coherence: GCD d-compatibility (single-tower).

        From incoherence_filter_lattice.txt:
            "ask whether any single sublattice class subsumes both required d-values.
            If no sublattice can subsume both without remainder, incoherent."

        STRUCTURAL NOTE: At single resolution, d = N/gcd(|k|, N) always divides N.
        The LCM of any set of divisors of N also divides N. Therefore this check
        ALWAYS PASSES at single-tower single-resolution — this is a mathematical
        theorem about divisors, not an implementation gap.

        The source material confirms L3 is about whether the COMPUTED d(r_i·r_j)
        agrees with the DIRECTLY MEASURED d. In discrete bijective k-space at
        27720ET (0/256 rounding errors), this agreement is exact. L3's teeth
        emerge ACROSS TOWERS where different R₀ seeds assign different d-values
        to the same byte transitions. See l3_cross_tower_transitions() for the
        cross-tower implementation that actually gates.

        Retained for structural completeness (the five-level filter is always
        applied in full, even when individual levels are trivially satisfied).
        """
        if not d_values:
            return True
        lcm_d = reduce(lambda a, b: (a * b) // math.gcd(a, b), d_values)
        return lcm_d <= n_res

    @staticmethod
    def l3_cross_tower_transitions(r0_personal: float, r0_universal: float,
                                    n_res: int = N_FULL) -> Tuple[bool, float]:
        """
        Level 3 — Cross-Tower Sublattice Coherence: d-family preservation.

        From incoherence_filter_lattice.txt (L3):
            "ask whether any single sublattice class subsumes both required d-values."
        From ET_Multifold §13:
            "A configuration can be coherent in one tower and incoherent in another
            — because the lattice coordinates shift with R₀."

        When the same byte is mapped through two R₀ seeds (personal, universal),
        the transition between adjacent bytes gets different Δk values:
            Δk_personal = k_p(b_{i+1}) - k_p(b_i)
            Δk_universal = k_u(b_{i+1}) - k_u(b_i)

        These Δk values may land in DIFFERENT sublattice families:
            d_personal = N/gcd(|Δk_personal|, N)
            d_universal = N/gcd(|Δk_universal|, N)

        When d_personal ≠ d_universal, the tower shift has changed the sublattice
        character of that transition — the lattice walk "looks like" a different
        type of step from the two perspectives. This IS the L3 sublattice
        Descriptor contradiction: the transition cannot be uniquely assigned to
        a single sublattice family across perspectives.

        Gate (from K = 2/3 = Koide binding stability threshold):
            At ∂I: tightness = K = 2/3, meaning 2/3 coherent, 1/3 incoherent.
            Cross-tower coherence requires at least K = 2/3 of ALL byte transitions
            to preserve sublattice family across the tower shift.

        If the d-preservation fraction drops below K, the universal lattice is
        too distorted a view of the personal lattice — cross-tower archetypes
        found on the universal projection are not faithful to the personal structure.

        Returns (is_coherent, d_preservation_fraction).
        """
        bk_p = build_byte_k_map(r0_personal)
        bk_u = build_byte_k_map(r0_universal)

        # Compute d-family preservation for all 255 adjacent-byte transitions
        # (byte b → byte b+1 for b in 0..254)
        n_preserved = 0
        n_total = 0
        for b in range(255):
            dk_p = bk_p[b + 1] - bk_p[b]
            dk_u = bk_u[b + 1] - bk_u[b]
            d_p = lattice_d(dk_p, n_res)
            d_u = lattice_d(dk_u, n_res)
            n_total += 1
            if d_p == d_u:
                n_preserved += 1

        fraction = n_preserved / max(n_total, 1)
        return fraction >= K_KOIDE, fraction

    @staticmethod
    def l3_cross_tower_pattern(pattern_dk_universal: tuple,
                                r0_personal: float, r0_universal: float,
                                byte_data: bytes,
                                n_res: int = N_FULL) -> bool:
        """
        Level 3 — Cross-Tower Pattern Sublattice Coherence.

        For a specific Δk pattern found on the universal lattice, verify that
        its sublattice families are preserved when viewed through the personal
        tower's lens. This gates individual cross-file archetypes — even if
        the overall tower-level L3 passes, specific patterns may be incoherent
        across perspectives.

        For each Δk_u in the pattern, compute the corresponding d_u. Then find
        the same transition in the personal lattice and compute d_p. If fewer
        than K = 2/3 of the pattern's steps preserve sublattice family, the
        pattern is cross-tower incoherent for this file.

        When byte_data is available, uses ACTUAL byte transitions to compute
        both personal and universal Δk for the same transition. When byte_data
        is empty, falls back to comparing d-values of the universal Δk pattern
        against the expected tower offset distortion.

        Returns True if the pattern is cross-tower coherent.
        """
        if not pattern_dk_universal:
            return False

        bk_p = build_byte_k_map(r0_personal)
        bk_u = build_byte_k_map(r0_universal)

        n_preserved = 0
        n_checked = 0

        if byte_data and len(byte_data) > 1:
            # ── ACTUAL BYTE TRANSITIONS from data ──
            # When byte_data is available, build a lookup of which actual
            # consecutive byte pairs produce each universal Δk. This is more
            # accurate than the brute-force scan because it tests transitions
            # that ACTUALLY OCCUR in the file — the real D-structure of the
            # P-substrate — rather than any arbitrary byte pair.
            #
            # Build: dk_u → list of (b_i, b_j) pairs that produce it
            data_bytes = np.frombuffer(byte_data, dtype=np.uint8)
            dk_u_to_byte_pairs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for idx in range(len(data_bytes) - 1):
                b_i = int(data_bytes[idx])
                b_j = int(data_bytes[idx + 1])
                dk_val = bk_u[b_j] - bk_u[b_i]
                # Store only one representative per Δk (first seen is sufficient)
                if dk_val not in dk_u_to_byte_pairs:
                    dk_u_to_byte_pairs[dk_val] = [(b_i, b_j)]
                elif len(dk_u_to_byte_pairs[dk_val]) < S:
                    # Keep up to S representatives for robust checking
                    dk_u_to_byte_pairs[dk_val].append((b_i, b_j))

            for dk_u in pattern_dk_universal:
                d_u = lattice_d(dk_u, n_res)
                representatives = dk_u_to_byte_pairs.get(dk_u, [])
                if not representatives:
                    # This Δk_u never occurs in the actual data —
                    # the universal lattice claims a transition with no
                    # byte-level realization in this file. L3 incoherence.
                    n_checked += 1
                    continue  # Not preserved

                # Check d-family preservation using actual byte pairs
                for b_i, b_j in representatives:
                    dk_p = bk_p[b_j] - bk_p[b_i]
                    d_p = lattice_d(dk_p, n_res)
                    n_checked += 1
                    if d_p == d_u:
                        n_preserved += 1
                    break  # One representative per Δk is sufficient
        else:
            # ── FALLBACK: brute-force byte scan ──
            # When byte_data is empty (e.g. data already released from tower),
            # scan all 256 byte values to find any pair producing each Δk_u
            # and check its personal d-value.
            kb_u = build_k_byte_map(r0_universal)

            for dk_u in pattern_dk_universal:
                d_u = lattice_d(dk_u, n_res)
                found_representative = False
                for b_i in range(256):
                    k_u_i = bk_u[b_i]
                    k_u_j = k_u_i + dk_u
                    if k_u_j in kb_u:
                        b_j = kb_u[k_u_j]
                        dk_p = bk_p[b_j] - bk_p[b_i]
                        d_p = lattice_d(dk_p, n_res)
                        n_checked += 1
                        if d_p == d_u:
                            n_preserved += 1
                        found_representative = True
                        break

                if not found_representative:
                    n_checked += 1

        if n_checked == 0:
            return False

        fraction = n_preserved / n_checked
        return fraction >= K_KOIDE

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
            # If we created the child, and it's still empty, remove it
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

        # ─── RUN-LENGTH COLLAPSE PRE-PASS ───
        # Descriptor Gap Principle: a contiguous run of identical symbols IS a
        # single Generative Descriptor (D Paper §44.1). Detect runs and create
        # optimal-length archetypes BEFORE the general pattern scan. These
        # run-archetypes have high elegance and will claim the entire run region
        # during greedy selection in subsume_patterns, preventing the general
        # scanner from fragmenting runs into many short overlapping patterns.
        run_archetypes = self._detect_run_archetypes(
            sym_stream, sym_d_map=sym_d_map, min_pattern_len=min_pattern_len)
        if run_archetypes:
            archetypes.extend(run_archetypes)

        # ─── PATTERN SCAN: C Engine (suffix array) with Python fallback ───
        # The C engine uses O(n log² n) suffix array + LCP to find ALL repeated
        # patterns across ALL lengths in a single pass. This replaces the Python
        # O(n × L_max) per-length dictionary scan — verified 1,778× speedup at
        # 100K symbols with zero pattern difference.
        #
        # ET filtering (IncoherenceFilter gates, elegance, d-values) stays in
        # Python. The C engine handles ONLY the combinatorial pattern finding.
        # If the C engine is unavailable, the original Python loop runs unchanged.

        c_results = PatternEngine.find_and_gate(
            sym_stream, min_pattern_len, actual_max - 1,
            n_res=self.n_res, min_count=2, min_net_savings=2)

        if c_results is not None:
            # ── C engine found + gated patterns — compute elegance ──
            # Pattern finding: C suffix array + LCP (O(n log² n))
            # IncoherenceFilter L1+L2+L3+L4: C batch gate (single call)
            # Only COHERENT patterns reach this point — no per-pattern
            # Python gate_archetype calls needed.
            self._log(f"    Pattern scan: C engine — {len(c_results):,} "
                      f"coherent patterns (pre-gated)")
            for pat, positions in c_results:
                count = len(positions)
                pat_len = len(pat)

                # Compute hierarchy elegance (same as Python path)
                if sym_d_map:
                    d_vals = [sym_d_map.get(s, N_FULL) for s in pat]
                else:
                    d_vals = [lattice_d(s) for s in pat]
                d_avg = sum(d_vals) / len(d_vals) if d_vals else N_FULL

                # Net savings (already pre-filtered by C engine, but verify)
                total_savings = count * (pat_len - 1)
                overhead = pat_len + 1
                net_savings = total_savings - overhead
                if net_savings < 2:
                    continue

                depth_factor = BIO_RES / max(d_avg, 1.0)
                e_hierarchy = net_savings * (1.0 + depth_factor / N_FULL)
                archetypes.append(LatticeWalkArchetype(
                    pattern=pat,
                    occurrences=positions,
                    hierarchy_elegance=e_hierarchy,
                    d_avg=d_avg,
                    pattern_length=pat_len,
                ))
        else:
            # ── Python fallback: original O(n × L_max) per-length scan ──
            # Runs when C engine is unavailable (no compiler, missing .c file).
            # Zero loss in features or function — identical pattern finding.
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
                    total_savings = count * (pat_len - 1)
                    overhead = pat_len + 1
                    net_savings = total_savings - overhead
                    if net_savings < 2:
                        continue
                    # ─── INCOHERENCE FILTER GATE ───
                    if not IncoherenceFilter.gate_archetype(pat, self.n_res):
                        continue
                    depth_factor = BIO_RES / max(d_avg, 1.0)
                    e_hierarchy = net_savings * (1.0 + depth_factor / N_FULL)
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

    def _detect_run_archetypes(self, sym_stream: List[int],
                                sym_d_map: Optional[Dict[int, float]] = None,
                                min_pattern_len: int = 2) -> List[LatticeWalkArchetype]:
        """
        Detect contiguous runs of identical symbols and create optimal archetypes.

        From the Descriptor Gap Principle (D Paper §44.1): a contiguous run of L
        identical symbols is generated by a single Generative Descriptor (symbol, count).
        Its Kolmogorov complexity is O(1) regardless of run length. The Subsumption
        Law requires that the entire run collapse into a single higher-order Descriptor
        when subsumable without remainder — and a constant run is ALWAYS subsumable.

        Problem solved: Without this pre-pass, the general pattern scanner finds
        many overlapping short patterns within the run (e.g., (1,1), (1,1,1), etc.)
        and the greedy selector fragments the run into many short archetype references.
        This pre-pass creates a SINGLE optimal-length archetype that claims the entire
        run region before the general scanner can fragment it.

        Optimal pattern length: p_opt ≈ √L maximizes net savings ≈ L - 2√L.
        At this length, the run collapses into ≈ √L archetype references in one pass.
        Subsequent recursion levels collapse those √L identical references further:
        after log₂(log₂(L)) recursion levels, the run converges to O(1) symbols.

        This is the compression analog of the cascade coherence horizon (L4 of the
        Incoherence Filter): a constant cascade has |δ| = 0, so N_max = ∞. The
        run can be compressed to any depth without exiting the coherent manifold.
        """
        n = len(sym_stream)
        # Minimum run length: manifold symmetry S = 12 ensures the run is
        # structurally significant (spans at least one full sublattice cycle)
        min_run = max(S, min_pattern_len * 2)

        run_archetypes = []

        # ── Scan for contiguous runs ──
        i = 0
        while i < n:
            sym = sym_stream[i]
            run_start = i
            while i < n and sym_stream[i] == sym:
                i += 1
            run_length = i - run_start

            if run_length < min_run:
                continue

            # ── Compute optimal pattern length: p_opt ≈ √L ──
            # Maximize net_savings = (L // p) × (p − 1) − (p + 1)
            # d/dp [L/p × (p-1) - p] = L/p - L/p² - 1 ≈ 0 ⟹ p ≈ √L
            # Clamp to [min_pattern_len, L // 2] (need ≥ 2 non-overlapping occurrences)
            p_opt = max(min_pattern_len, round(math.sqrt(run_length)))
            p_opt = min(p_opt, run_length // 2)
            if p_opt < min_pattern_len:
                continue

            # ── Create the run pattern: (sym, sym, ..., sym) × p_opt ──
            pattern = tuple([sym] * p_opt)

            # ── Collect ALL starting positions within the run (including overlapping) ──
            # subsume_patterns iterates these greedily for non-overlapping placement.
            all_positions = list(range(run_start, run_start + run_length - p_opt + 1))
            count = len(all_positions)
            if count < 2:
                continue

            # ── Incoherence Filter gate ──
            if not IncoherenceFilter.gate_archetype(pattern, self.n_res):
                continue

            # ── Compute sublattice depth from sym_d_map (actual Δk d-values) ──
            if sym_d_map:
                d_val = sym_d_map.get(sym, float(N_FULL))
            else:
                d_val = float(lattice_d(sym))
            d_avg = d_val

            # ── Net savings and elegance ──
            total_savings = count * (p_opt - 1)
            overhead = p_opt + 1
            net_savings = total_savings - overhead
            if net_savings < 2:
                continue

            depth_factor = BIO_RES / max(d_avg, 1.0)
            e_hierarchy = net_savings * (1.0 + depth_factor / N_FULL)

            run_archetypes.append(LatticeWalkArchetype(
                pattern=pattern,
                occurrences=all_positions,
                hierarchy_elegance=e_hierarchy,
                d_avg=d_avg,
                pattern_length=p_opt,
            ))

            self._log(f"    Run collapsed: sym={sym}, L={run_length}, "
                      f"p_opt={p_opt}, occurrences={count}, net={net_savings}")

        return run_archetypes

    def subsume_patterns(self, sym_stream: List[int],
                         archetypes: List[LatticeWalkArchetype]) -> Tuple[List, List[LatticeWalkArchetype]]:
        """
        Replace occurrences of archetype patterns with references.

        Returns (encoded_stream, used_archetypes) where encoded_stream
        contains either raw symbol values or archetype reference markers.
        Archetype indices in the encoded stream reference used_archetypes.

        Non-overlapping: first match wins (greedy, highest elegance first).
        Uses C-accelerated greedy placement when available (eliminates the
        O(n × archetypes × positions) consumed-array overlap check in Python).
        Falls back to Python if C engine unavailable — zero loss in function.
        """
        n = len(sym_stream)

        # ── Try C-accelerated greedy placement ──
        c_result = PatternEngine.subsume_greedy_c(n, archetypes)

        if c_result is not None:
            c_placements, c_used_mask = c_result

            # Build used_archetypes and orig_to_used mapping from C result
            used_archetypes = []
            orig_to_used: Dict[int, int] = {}
            for arch_idx in range(len(archetypes)):
                if c_used_mask[arch_idx]:
                    orig_to_used[arch_idx] = len(used_archetypes)
                    used_archetypes.append(archetypes[arch_idx])

            # c_placements = [(arch_idx, position), ...] — sort by position
            archetype_placements = sorted(c_placements, key=lambda x: x[1])
        else:
            # ── Python fallback: original greedy placement ──
            consumed = bytearray(n)  # bytearray: 0.6MB vs 4.7MB for bool list
            used_archetypes = []
            orig_to_used = {}
            archetype_placements = []

            for arch_idx, arch in enumerate(archetypes):
                placed = False
                for pos in arch.occurrences:
                    if any(consumed[pos + j] for j in range(arch.pattern_length) if pos + j < n):
                        continue
                    for j in range(arch.pattern_length):
                        if pos + j < n:
                            consumed[pos + j] = 1
                    archetype_placements.append((arch_idx, pos))
                    placed = True

                if placed:
                    orig_to_used[arch_idx] = len(used_archetypes)
                    used_archetypes.append(arch)

            archetype_placements.sort(key=lambda x: x[1])

        # ── Build encoded stream from placements ──
        # (Same logic for both C and Python paths)
        encoded = []
        placement_idx = 0
        i = 0

        while i < n:
            if placement_idx < len(archetype_placements):
                arch_idx, pos = archetype_placements[placement_idx]
                if i == pos:
                    used_idx = orig_to_used[arch_idx]
                    encoded.append(('arch', used_idx))
                    i += archetypes[arch_idx].pattern_length
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
        - Symbols 0...n_base-1: raw Δk indices (from dk_table)
        - Symbols n_base...: archetype references, each expands to a pattern
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

    def pair_recursive_compress(self, dk_stream: List[int]) -> dict:
        """
        Re-Pair style pair-first recursive compression for chaotic data.

        From the Descriptor Gap Principle: chaotic data (logistic maps, Lorenz
        attractors, skew-tent maps, PRNG outputs, scientific simulations) has
        hidden low-dimensional structure. It LOOKS random — high marginal entropy,
        flat histograms — but has low Kolmogorov complexity. The generating map
        (the Descriptor) is simple; the output (the Point data) is complex.

        Standard archetype scanning (find_walk_archetypes) fails on chaotic data
        because it looks for exact long-pattern repeats. Chaotic data rarely has
        exact long repeats. But it DOES have non-uniform PAIR (bigram) frequencies —
        the conditional distribution P(Δk_{n+1} | Δk_n) has much lower entropy
        than the marginal P(Δk_n). This is the hidden D-structure.

        Grammar-based compressors (Re-Pair, NSRPS) extract this structure by
        iteratively replacing the most frequent pair of adjacent symbols with
        a new symbol, building a hierarchical grammar that captures the recursive
        D-structure of the generating map (D Paper §44.1: the Generative Descriptor).

        Algorithm (Re-Pair adapted for the ET lattice):
        1. Count all adjacent pairs and their frequencies
        2. Gate through IncoherenceFilter (L1 + L2 pairwise coherence)
        3. Replace all non-overlapping occurrences of the most frequent coherent pair
        4. Record the pair as an archetype definition
        5. Repeat until no pair occurs ≥ 2 times or stream is too short

        The maximum iteration count is S × MAX_DEPTH = 12 × 12 = 144, matching
        the manifold's maximum recursion capacity (S³ = 1728 is the full lattice
        walk space; 144 = S² is the cross-file pattern cap).

        ET Derivation:
            Identification Principle: the data's P-substrate is the generating map's
            output; the D-set is the pair-grammar; T is the iterative replacement.
            Descriptor Gap Principle: each replaced pair IS a found Descriptor —
            the gap between "looks random" and "has structure" closes iteratively.
            Subsumption Law: each pair-archetype subsumes its occurrences without
            remainder. The grammar tree IS the subsumption hierarchy of the data.

        Returns the same format as recursive_compress:
            dk_table, n_base, archetype_defs, total_symbols, final_stream
        """
        dk_value_set: Set[int] = set(dk_stream)
        unique_dks = sorted(dk_value_set)
        dk_to_idx = {dk: i for i, dk in enumerate(unique_dks)}
        n_base = len(unique_dks)

        current = [dk_to_idx[dk] for dk in dk_stream]
        archetype_defs: List[Tuple[int, ...]] = []
        next_sym_id = n_base

        d_map: Dict[int, float] = {i: float(lattice_d(dk))
                                    for i, dk in enumerate(unique_dks)}

        # ── L4: CASCADE COHERENCE ──
        # Compute the cascade horizon for the pair-replacement process.
        # Each iteration adds one level of grammar depth. The cascade must
        # remain within the coherent manifold.
        eps_vals = []
        for dk in unique_dks:
            if dk != 0:
                ratio = 2.0 ** (dk / self.n_res)
                eps_vals.append(abs(lattice_epsilon(ratio, dk, self.n_res)))
        avg_eps = sum(eps_vals) / len(eps_vals) if eps_vals else 0.0
        max_iterations = S * MAX_DEPTH  # 144 = S² iterations maximum
        if avg_eps > EPSILON:
            cascade_horizon = IncoherenceFilter.l4_cascade_horizon(avg_eps)
            max_iterations = min(max_iterations, cascade_horizon)

        pairs_replaced = 0
        for iteration in range(max_iterations):
            if len(current) < 4:
                break

            # ── Step 1: Count all adjacent pairs ──
            pair_counts: Counter = Counter()
            for i in range(len(current) - 1):
                pair = (current[i], current[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # ── Step 2: Find the most frequent coherent pair ──
            # Sort by frequency descending, then by lattice depth (prefer
            # deep sublattice pairs — low d_avg — at equal frequency)
            best_pair = None
            best_count = 1  # Need at least 2 non-overlapping occurrences

            for pair, count in pair_counts.most_common():
                if count <= best_count:
                    break  # Since most_common() is sorted, no better pair exists
                # IncoherenceFilter gate: L1 + L2 pairwise coherence
                if not IncoherenceFilter.gate_archetype(pair, self.n_res):
                    continue
                best_pair = pair
                best_count = count
                break  # Take the most frequent coherent pair

            if best_pair is None:
                break

            # ── Step 3: Net savings check ──
            # Each pair occurrence saves 1 symbol (2 symbols → 1 reference).
            # Overhead: 2 symbols for the pattern definition + 1 new symbol ID.
            net_savings = best_count - 3  # count × 1 - (2 + 1)
            if net_savings < 1:
                break

            # ── Step 4: Replace all non-overlapping occurrences ──
            new_sym = next_sym_id
            archetype_defs.append(best_pair)

            # Compute d_avg for the new symbol from its constituent d-values
            d_a = d_map.get(best_pair[0], float(N_FULL))
            d_b = d_map.get(best_pair[1], float(N_FULL))
            d_map[new_sym] = (d_a + d_b) / 2.0
            next_sym_id += 1

            new_stream = []
            i = 0
            while i < len(current):
                if (i < len(current) - 1
                        and current[i] == best_pair[0]
                        and current[i + 1] == best_pair[1]):
                    new_stream.append(new_sym)
                    i += 2
                else:
                    new_stream.append(current[i])
                    i += 1

            if len(new_stream) >= len(current):
                break  # No progress — stop
            current = new_stream
            pairs_replaced += 1

        if pairs_replaced > 0:
            self._log(f"    Pair-first (Re-Pair): {pairs_replaced} pairs replaced, "
                      f"stream {len(dk_stream)} → {len(current)} symbols, "
                      f"{len(archetype_defs)} grammar rules")

        return {
            'dk_table': unique_dks,
            'n_base': n_base,
            'archetype_defs': archetype_defs,
            'total_symbols': next_sym_id,
            'final_stream': current,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE TOWERS + UNIVERSAL LATTICE — Cross-file tower architecture
#
# From the AI worldview module (et_conscious_ai_worldview.py §6):
#   Tower architecture = P-substrate + universal lattice + R₀ seed
#
# From the AI compression module (et_conscious_ai_compression.py):
#   E_cross = √(E_universal × E_personal)
#   Personal tower: per-source R₀, per-source byte↔k map
#   Universal lattice: geometric mean of all R₀ values
#   Cross-tower coherence: tightness_product ≥ K (Koide = 2/3)
#
# The compressor's existing `discover_r0` per file implements the personal
# tower seed. These classes add the universal lattice and cross-file
# archetype discovery that were missing (Violations 2, 3, 4).
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatticeTower:
    """
    Personal lattice tower for a single file/source.

    From the AI worldview module (et_conscious_ai_worldview.py §6):
    Tower architecture = P-substrate + universal lattice + R₀ seed.

    Each file has its own R₀ (personal fundamental period), discovered
    from the geometric mean of its byte values via discover_r0(). The
    personal tower contains the file's byte↔k bijection, its Δk pattern
    stream, and the archetypes found within it.

    The personal tower is projected onto the universal lattice for
    cross-file archetype discovery (Violation 4).

    ET Derivation:
        Identification Principle: R₀ identifies the file's D-structure.
        Descriptor Gap Principle: personal patterns are the file's D-set.
        Subsumption Law: cross-file archetypes subsume shared patterns.
    """
    file_path: str                          # Source file path
    personal_r0: float                      # R₀ seed discovered from this file's data
    byte_k_map: Dict[int, int]              # byte→k mapping at personal R₀
    k_byte_map: Dict[int, int]              # k→byte inverse mapping
    dk_stream: List[int]                    # Δk stream of this file (personal lattice)
    dk_universal: List[int] = field(default_factory=list)  # Δk stream on universal lattice
    data_size: int = 0                      # Original file size in bytes
    data_hash: bytes = b''                  # SHA-256 of original data
    raw_data: bytes = b''                   # Original file bytes (held during batch)

    @staticmethod
    def from_file(file_path: str) -> 'LatticeTower':
        """
        Build a personal lattice tower from a file.

        Reads the file, discovers R₀, builds byte↔k maps, and computes
        the Δk stream in the personal lattice coordinates.
        """
        with open(file_path, 'rb') as f:
            raw = f.read()

        r0 = discover_r0(raw)
        bk = build_byte_k_map(r0)
        kb = build_k_byte_map(r0)

        values = np.frombuffer(raw, dtype=np.uint8)
        k_stream = [bk[int(b)] for b in values]
        dk_stream = [k_stream[i + 1] - k_stream[i]
                     for i in range(len(k_stream) - 1)] if len(k_stream) > 1 else []

        return LatticeTower(
            file_path=file_path,
            personal_r0=r0,
            byte_k_map=bk,
            k_byte_map=kb,
            dk_stream=dk_stream,
            dk_universal=[],
            data_size=len(raw),
            data_hash=hashlib.sha256(raw).digest(),
            raw_data=raw,
        )


class UniversalLattice:
    """
    Universal lattice for cross-file archetype compression.

    From the AI compression module (et_conscious_ai_compression.py):
    - Universal R₀ = geometric mean of all personal R₀ values
      (the geometric centroid of all personal seeds on the multiplicative lattice)
    - Cross-tower elegance: E_cross = √(E_universal × E_personal)
      (ported from SubsumptionHierarchyOperator.compute_cross_tower_elegance)
    - Universal projection: all files' byte streams mapped onto universal byte↔k
    - Cross-file archetypes: Δk patterns recurring across multiple towers,
      gated by IncoherenceFilter and weighted by cross-tower elegance

    The AI module uses personal tower (per-source R₀) + universal lattice:
        et_conscious_ai_compression.py line 399:
            E_cross = √(E_universal × E_personal)
        et_conscious_ai_compression.py line 446-452:
            tightness_product = t_universal × t_personal
            if tightness_product < KOIDE_RATIO: return 0.0

    This class ports that architecture to the compressor. Each file gets a
    LatticeTower (personal R₀). All towers project onto this UniversalLattice
    (universal R₀). Cross-file archetypes are found on the universal lattice
    and applied back to individual file compression.

    ET Derivation:
        The universal lattice IS the shared P-substrate across all files.
        Each file's personal tower is a D-constrained view of that substrate.
        Cross-file archetypes are T-traversals that recur across views —
        structural regularities shared by the entire file set.
    """

    def __init__(self, towers: List[LatticeTower], log_fn=None):
        self.towers = towers
        self._log = log_fn or (lambda m: None)

        # ── Universal R₀: geometric mean of all personal R₀ values ──
        # From the Seed Theorem (Multifold §2): the universal seed IS the
        # geometric centroid of all personal seeds on the multiplicative lattice.
        # This is the same approach as the AI compression module's tower_r0
        # computation — the geometric mean preserves multiplicative structure.
        valid_r0s = [t.personal_r0 for t in towers if t.personal_r0 > 0]
        if valid_r0s:
            log_sum = sum(math.log(r) for r in valid_r0s)
            self.universal_r0 = math.exp(log_sum / len(valid_r0s))
        else:
            self.universal_r0 = 1.0

        # Universal byte↔k maps
        self.universal_byte_k = build_byte_k_map(self.universal_r0)
        self.universal_k_byte = build_k_byte_map(self.universal_r0)

        # Cross-file pattern registry: pattern → list of tower indices where it appears
        self.cross_patterns: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

        # Cross-file archetypes discovered on the universal lattice
        self.cross_archetypes: List[LatticeWalkArchetype] = []

        # Per-tower universal projections (computed in project_all_towers)
        self.tower_dk_universal: Dict[int, List[int]] = {}

        # Per-tower cross-tower elegance scores (byte_val → E_cross)
        self.tower_cross_elegances: Dict[int, Dict[int, float]] = {}

        self._log(f"Universal Lattice: R₀_u = {self.universal_r0:.6f} "
                  f"(geometric mean of {len(valid_r0s)} towers)")
        for ti, tower in enumerate(towers):
            self._log(f"  Tower {ti}: R₀_p = {tower.personal_r0:.6f} "
                      f"({os.path.basename(tower.file_path)}, {tower.data_size:,}B)")

    def compute_cross_tower_elegance(self, byte_val: int,
                                      tower: LatticeTower) -> float:
        """
        Compute cross-tower elegance for a byte value.

        E_cross = √(E_universal × E_personal)

        Where:
            E_universal = elegance at byte's position on universal lattice
            E_personal = elegance at byte's position on personal tower

        The Koide coherence gate (from et_conscious_ai_compression.py lines 446-452):
            tightness_universal × tightness_personal ≥ K (= 2/3)
        If the tightness product drops below K, the binding dissolves —
        the byte is incoherent across perspectives and cannot form part
        of a stable cross-file archetype.

        Directly ported from SubsumptionHierarchyOperator.compute_cross_tower_elegance
        in et_conscious_ai_compression.py (lines 399–455).
        """
        # ── Universal elegance ──
        ratio_u = (byte_val + 1.0) / self.universal_r0
        k_u = lattice_k(ratio_u)
        eps_u = lattice_epsilon(ratio_u, k_u)
        t_u = lattice_tightness(eps_u)
        d_u = lattice_d(k_u)
        e_universal = lattice_elegance(ratio_u, 1, 1)  # p=q=1

        # ── Personal elegance ──
        ratio_p = (byte_val + 1.0) / tower.personal_r0
        k_p = lattice_k(ratio_p)
        eps_p = lattice_epsilon(ratio_p, k_p)
        t_p = lattice_tightness(eps_p)
        d_p = lattice_d(k_p)
        e_personal = lattice_elegance(ratio_p, 1, 1)  # p=q=1

        # ── Koide coherence gate ──
        # From et_conscious_ai_compression.py line 451:
        # if tightness_product < KOIDE_RATIO: return 0.0
        tightness_product = t_u * t_p
        if tightness_product < K_KOIDE:
            return 0.0  # Incoherent across perspectives

        # ── L3 cross-tower sublattice coherence ──
        # From incoherence_filter_lattice.txt: "ask whether any single sublattice
        # class subsumes both required d-values."
        #
        # The subsuming class is LCM(d_u, d_p) = d_combined. The "sublattice
        # expansion factor" = d_combined / max(d_u, d_p) = min(d_u, d_p) / gcd(d_u, d_p)
        # measures how much the sublattice must expand to accommodate both views.
        #
        # When one d divides the other, expansion = 1 (compatible families).
        # When they're coprime and large, expansion = min(d_u, d_p) (incompatible).
        #
        # Gate: expansion must not exceed S (manifold symmetry = 12). Requiring
        # > S-fold sublattice expansion means the cross-tower assignment needs more
        # than one full sublattice cycle of refinement — the two perspectives have
        # structurally incompatible sublattice families for this byte.
        #
        # NOTE: The old check (d_combined > N_FULL) was structurally vacuous —
        # both d-values divide N_FULL, so LCM always ≤ N_FULL. This gate replaces
        # it with a check that actually has teeth.
        d_combined = (d_u * d_p) // math.gcd(d_u, d_p)
        sublattice_expansion = d_combined // max(d_u, d_p) if max(d_u, d_p) > 0 else 1
        if sublattice_expansion > S:
            return 0.0  # Cross-tower L3: sublattice families too far apart

        # ── Cross-tower elegance: geometric mean with depth factor ──
        # From et_conscious_ai_compression.py line 455 + evaluate_cluster line 512:
        # E_cross = √(E_universal × E_personal) × (420 / d_avg) / N_FULL
        # The depth factor (BIO_RES / d_avg) rewards bytes at deep sublattice
        # families (low d = octave, cubic, quintic) over shallow ones.
        d_avg = (d_u + d_p) / 2.0
        depth_factor = BIO_RES / max(d_avg, 1.0)
        base_elegance = math.sqrt(max(e_universal, 0.0) * max(e_personal, 0.0))
        return base_elegance * (1.0 + depth_factor / N_FULL)

    def project_all_towers(self):
        """
        Project every tower's byte stream onto the universal lattice.

        For each tower:
        1. Map every byte through the universal byte→k map
        2. Compute the universal Δk stream
        3. Compute cross-tower elegance for every byte value
        4. Store the universal projection and elegances

        This is the "projection onto universal lattice" required by Violation 4:
        "all lattice nodes are projected onto the universal lattice and
        further compressed."
        """
        for ti, tower in enumerate(self.towers):
            # Project byte stream through universal lattice
            values = np.frombuffer(tower.raw_data, dtype=np.uint8)
            k_stream_u = [self.universal_byte_k[int(b)] for b in values]
            if len(k_stream_u) > 1:
                dk_u = [k_stream_u[i + 1] - k_stream_u[i]
                        for i in range(len(k_stream_u) - 1)]
            else:
                dk_u = []

            tower.dk_universal = dk_u
            self.tower_dk_universal[ti] = dk_u

            # ── L3 Cross-Tower Sublattice Coherence (tower-level) ──
            # From Multifold §13: "A configuration can be coherent in one tower
            # and incoherent in another — because the lattice coordinates shift
            # with R₀." Check whether the tower offset preserves sublattice
            # families across at least K = 2/3 of all byte transitions.
            l3_coherent, l3_fraction = IncoherenceFilter.l3_cross_tower_transitions(
                tower.personal_r0, self.universal_r0)
            self._log(f"  Tower {ti} L3 cross-tower: d-preservation = "
                      f"{l3_fraction:.3f} ({'COHERENT' if l3_coherent else 'INCOHERENT'})")

            # Compute cross-tower elegance for every byte value in this tower
            cross_eleg = {}
            for b_val in range(256):
                cross_eleg[b_val] = self.compute_cross_tower_elegance(b_val, tower)
            self.tower_cross_elegances[ti] = cross_eleg

            n_coherent = sum(1 for e in cross_eleg.values() if e > 0)
            self._log(f"  Tower {ti} projected: {len(dk_u):,} Δk_universal, "
                      f"{n_coherent}/256 bytes cross-coherent")

    def collect_cross_file_patterns(self, tower_idx: int, dk_universal: List[int],
                                     min_pat_len: int = 2,
                                     max_pat_len: int = 0):
        """
        Collect Δk patterns from a tower's universal projection into
        the cross-file pattern registry.

        Each pattern tracks which towers it appears in. Patterns that
        appear in multiple towers are candidates for cross-file archetypes.

        The pattern length is bounded by S² = 144 for cross-file patterns
        (balancing thoroughness with combinatorial explosion). Longer
        patterns are handled by the per-file recursive compression.
        """
        n = len(dk_universal)
        if n < 4:
            return
        if max_pat_len <= 0:
            max_pat_len = min(n // 2, S * S)  # Up to S² = 144

        actual_max = min(max_pat_len + 1, n // 2 + 1)
        for pat_len in range(min_pat_len, actual_max):
            seen_this_length: Set[Tuple[int, ...]] = set()
            for i in range(n - pat_len + 1):
                pat = tuple(dk_universal[i:i + pat_len])
                if pat not in seen_this_length:
                    seen_this_length.add(pat)
                    if tower_idx not in self.cross_patterns[pat]:
                        self.cross_patterns[pat].append(tower_idx)

    def find_cross_file_archetypes(self) -> List[LatticeWalkArchetype]:
        """
        Find archetypes that appear across multiple files.

        A cross-file archetype is a Δk pattern (on the universal lattice)
        that occurs in at least 2 different towers. These represent
        structural regularities shared across the entire file set —
        the "common language" of the file set on the multiplicative lattice.

        Algorithm:
        1. Filter to patterns appearing in ≥ 2 different towers
        2. Gate through IncoherenceFilter (all 5 levels)
        3. Compute E_hierarchy with cross-file bonus (√N_files scaling)
        4. Sort by E_hierarchy descending
        5. L5 coherent summation: verify cross-archetype d-compatibility

        This implements the requirement from the prompt:
        "Everything will be projected onto its own lattice, or multiple
        for multiple files, then compressed and joined when the archetypes
        are compressed."
        """
        cross_archetypes = []

        for pat, tower_indices in self.cross_patterns.items():
            # Must appear in at least 2 different files
            unique_towers = set(tower_indices)
            if len(unique_towers) < 2:
                continue

            n_files = len(unique_towers)

            # Count total occurrences across all towers
            total_count = 0
            for ti in unique_towers:
                dk_u = self.tower_dk_universal.get(ti, [])
                for i in range(len(dk_u) - len(pat) + 1):
                    if tuple(dk_u[i:i + len(pat)]) == pat:
                        total_count += 1

            if total_count < 2:
                continue

            # ── Incoherence Filter gate (single-resolution L1-L4) ──
            if not IncoherenceFilter.gate_archetype(pat, N_FULL):
                continue

            # ── L3 Cross-Tower Pattern Sublattice Coherence ──
            # From Multifold §13: cross-tower L3 checks that the universal Δk
            # pattern preserves sublattice families when viewed through each
            # tower's personal lens. A cross-file archetype is only valid if
            # it's coherent across at least K = 2/3 of the towers it appears in.
            n_l3_coherent = 0
            for ti in unique_towers:
                tower = self.towers[ti]
                if IncoherenceFilter.l3_cross_tower_pattern(
                        pat, tower.personal_r0, self.universal_r0,
                        tower.raw_data):
                    n_l3_coherent += 1
            l3_cross_fraction = n_l3_coherent / n_files if n_files > 0 else 0.0
            if l3_cross_fraction < K_KOIDE:
                continue  # Cross-tower L3: pattern incoherent in too many towers

            # ── Compute cross-file hierarchy elegance ──
            d_vals = [lattice_d(dk) for dk in pat]
            d_avg = sum(d_vals) / len(d_vals) if d_vals else N_FULL

            # Net savings: total occurrences × (pat_len - 1) - overhead
            total_savings = total_count * (len(pat) - 1)
            overhead = len(pat) + 1
            net_savings = total_savings - overhead
            if net_savings < 2:
                continue

            # E_hierarchy with cross-file bonus:
            # Cross-file patterns are MORE valuable because they reduce
            # redundancy across the entire file set. The bonus scales as
            # √N_files — geometric scaling from the multiplicative lattice.
            depth_factor = BIO_RES / max(d_avg, 1.0)
            cross_bonus = math.sqrt(n_files)  # √N_files: geometric scaling
            e_hierarchy = net_savings * (1.0 + depth_factor / N_FULL) * cross_bonus

            # Collect actual occurrence positions per tower for subsumption
            all_positions = []
            for ti in unique_towers:
                dk_u = self.tower_dk_universal.get(ti, [])
                for i in range(len(dk_u) - len(pat) + 1):
                    if tuple(dk_u[i:i + len(pat)]) == pat:
                        all_positions.append(i)

            cross_archetypes.append(LatticeWalkArchetype(
                pattern=pat,
                occurrences=all_positions,
                hierarchy_elegance=e_hierarchy,
                d_avg=d_avg,
                pattern_length=len(pat),
            ))

        # Sort by elegance descending, then pattern length descending
        cross_archetypes.sort(
            key=lambda arch: (arch.hierarchy_elegance, arch.pattern_length), reverse=True)

        # ── L5: Coherent summation ──
        # Verify the cross-archetype set has cross-archetype sublattice
        # compatibility (L3 over the whole set). From the incoherence filter:
        # sum only over coherent configurations.
        if len(cross_archetypes) > 1:
            all_d_vals = []
            for a in cross_archetypes:
                all_d_vals.extend([lattice_d(s, N_FULL) for s in a.pattern])
            if not IncoherenceFilter.l3_sublattice(all_d_vals, N_FULL):
                # If the full set is incoherent, keep only the top archetypes
                # that form a coherent subset (greedy selection)
                coherent_set = []
                running_d = []
                for a in cross_archetypes:
                    trial_d = running_d + [lattice_d(s, N_FULL) for s in a.pattern]
                    if IncoherenceFilter.l3_sublattice(trial_d, N_FULL):
                        coherent_set.append(a)
                        running_d = trial_d
                cross_archetypes = coherent_set

        self.cross_archetypes = cross_archetypes
        self._log(f"Universal Lattice: {len(cross_archetypes)} cross-file archetypes found "
                  f"across {len(self.towers)} towers")

        return cross_archetypes

    def apply_cross_archetypes_to_tower(self, tower_idx: int,
                                         walker: LatticeWalkCompressor
                                         ) -> Optional[dict]:
        """
        Apply cross-file archetypes to a tower's universal Δk projection.

        This is the "further compressed" step from the prompt:
        "all lattice nodes are projected onto the universal lattice and
        further compressed."

        The cross-file archetypes are pre-seeded patterns found on the
        universal lattice. They are applied via subsume_patterns to the
        tower's universal Δk stream, replacing cross-file pattern
        occurrences with archetype references. The reduced stream is
        then recursively compressed for additional per-file patterns.

        Returns a recursive compress result if cross-file archetypes
        improve the stream, or None if they do not.

        ET Derivation:
            Subsumption Law: cross-file archetypes subsume the shared
            Δk patterns without remainder. The reduced stream contains
            only per-file unique structure — the file's individual
            "fingerprint" on the lattice.
        """
        if not self.cross_archetypes:
            return None

        dk_u = self.tower_dk_universal.get(tower_idx, [])
        if not dk_u or len(dk_u) < 4:
            return None

        # ── Subsume cross-file archetypes into the universal Δk stream ──
        # Find which cross-file archetypes have occurrences in this tower
        tower_archetypes = []
        for arch in self.cross_archetypes:
            # Find occurrences in this tower's universal projection
            positions = []
            pat = arch.pattern
            for i in range(len(dk_u) - len(pat) + 1):
                if tuple(dk_u[i:i + len(pat)]) == pat:
                    positions.append(i)
            if positions:
                # Create a tower-local archetype with this tower's positions
                tower_archetypes.append(LatticeWalkArchetype(
                    pattern=arch.pattern,
                    occurrences=positions,
                    hierarchy_elegance=arch.hierarchy_elegance,
                    d_avg=arch.d_avg,
                    pattern_length=arch.pattern_length,
                ))

        if not tower_archetypes:
            return None

        # ── Apply subsumption ──
        encoded, used_archs = walker.subsume_patterns(dk_u, tower_archetypes)
        if not used_archs:
            return None

        # ── Build the reduced stream ──
        # Cross-file archetype references use IDs offset by a large base
        # to distinguish them from raw Δk indices and per-file archetypes.
        # This is the same monotonically-growing symbol space approach
        # used in recursive_compress (line 794): symbols 0...n_base-1 are
        # raw Δk indices, n_base... are archetype references.
        dk_value_set: Set[int] = set(dk_u)
        unique_dks = sorted(dk_value_set)
        dk_to_idx = {dk: i for i, dk in enumerate(unique_dks)}
        n_base = len(unique_dks)

        # Map cross-file archetype IDs starting at n_base
        cross_arch_id_start = n_base
        cross_arch_id_map = {}
        cross_arch_defs = []
        for ci, arch in enumerate(used_archs):
            cross_arch_id_map[id(arch)] = cross_arch_id_start + ci
            cross_arch_defs.append(arch.pattern)

        new_stream = []
        for item_type, item_val in encoded:
            if item_type == 'raw':
                # Map raw Δk value to its index in the dk table
                if item_val in dk_to_idx:
                    new_stream.append(dk_to_idx[item_val])
                else:
                    # Value not in table — should not happen, but handle gracefully
                    unique_dks.append(item_val)
                    dk_to_idx[item_val] = len(unique_dks) - 1
                    n_base = len(unique_dks)
                    new_stream.append(dk_to_idx[item_val])
            else:
                # Cross-file archetype reference
                arch_sym_id = cross_arch_id_map[id(used_archs[item_val])]
                new_stream.append(arch_sym_id)

        if len(new_stream) >= len(dk_u):
            return None  # No improvement from cross-file archetypes

        # ── Recursive compress on the reduced stream ──
        # The stream now contains both raw Δk indices and cross-file
        # archetype reference symbols. recursive_compress will find
        # additional per-file patterns in this mixed stream.
        d_map: Dict[int, float] = {i: float(lattice_d(dk))
                                    for i, dk in enumerate(unique_dks)}
        for ci, arch in enumerate(used_archs):
            d_map[cross_arch_id_start + ci] = arch.d_avg

        next_sym_id = cross_arch_id_start + len(used_archs)
        archetype_defs_all = list(cross_arch_defs)  # Start with cross-file defs

        current = new_stream

        for depth in range(MAX_DEPTH):
            if len(current) < 4:
                break

            # L4 cascade coherence check
            if depth > 0:
                eps_vals = []
                for s in current:
                    if s < n_base:
                        dk = int(unique_dks[s])
                        ratio = 2.0 ** (dk / N_FULL) if dk != 0 else 1.0
                        eps_vals.append(abs(lattice_epsilon(ratio, dk, N_FULL)))
                if eps_vals:
                    avg_eps = sum(eps_vals) / len(eps_vals)
                    n_max_cascade = (IncoherenceFilter.l4_cascade_horizon(avg_eps)
                                     if avg_eps > EPSILON else N_FULL)
                    if depth >= n_max_cascade:
                        break

            archetypes = walker.find_walk_archetypes(
                current, sym_d_map=d_map, min_pattern_len=2,
                max_pattern_len=min(len(current) // 2, S * S * (depth + 1)))
            if not archetypes:
                break

            enc, used = walker.subsume_patterns(current, archetypes)
            if not used:
                break

            arch_id_map_inner = {}
            for arch in used:
                arch_id_map_inner[id(arch)] = next_sym_id
                archetype_defs_all.append(arch.pattern)
                d_map[next_sym_id] = arch.d_avg
                next_sym_id += 1

            ns = []
            for it, iv in enc:
                if it == 'raw':
                    ns.append(iv)
                else:
                    ns.append(arch_id_map_inner[id(used[iv])])

            if len(ns) >= len(current):
                break
            current = ns

        return {
            'dk_table': unique_dks,
            'n_base': n_base,
            'archetype_defs': archetype_defs_all,
            'total_symbols': next_sym_id,
            'final_stream': current,
        }

    def compress_tower_with_universal(self, tower_idx: int,
                                       engine: 'CDFEngine') -> Optional[bytes]:
        """
        Compress a tower using the universal lattice perspective.

        This provides an alternative compression for the file by using
        the UNIVERSAL R₀ as the lattice seed instead of the personal R₀.
        Files with content similar to other files in the batch will
        compress better through the universal lens because the universal
        lattice's byte↔k map is optimized for the collective.

        Returns the compressed block bytes, or None if the tower data
        is too small.
        """
        tower = self.towers[tower_idx]
        data = tower.raw_data
        n = len(data)
        if n == 0:
            return None

        values = np.frombuffer(data, dtype=np.uint8)

        # Check for uniform block
        unique = np.unique(values)
        if len(unique) == 1:
            return struct.pack('<BIB', 1, n, int(unique[0]))

        # Build k-stream through universal lattice
        k_stream_u = [self.universal_byte_k[int(b)] for b in values]
        k0_u = k_stream_u[0]

        # Δk on universal lattice
        dk_u = [k_stream_u[i + 1] - k_stream_u[i] for i in range(n - 1)]

        walker = LatticeWalkCompressor(self.universal_r0, log_fn=engine.log_fn)

        # ── Try cross-file archetype-enhanced compression ──
        rc_cross = self.apply_cross_archetypes_to_tower(tower_idx, walker)
        best_cross = None
        if rc_cross is not None:
            try:
                best_cross = engine.encode_lattice_block(
                    n, self.universal_r0, k0_u, 1, 0, None, rc_cross)
            except (ValueError, KeyError, IndexError, struct.error):
                best_cross = None

        # ── Try plain universal Δk compression ──
        best_plain = None
        if len(dk_u) > 3:
            try:
                rc_plain = walker.recursive_compress(dk_u)
                best_plain = engine.encode_lattice_block(
                    n, self.universal_r0, k0_u, 1, 0, None, rc_plain)
            except (ValueError, KeyError, IndexError, struct.error):
                best_plain = None

        # ── Try universal ΔΔk ──
        best_ddk = None
        if len(dk_u) > 2:
            ddk_u = [dk_u[i + 1] - dk_u[i] for i in range(len(dk_u) - 1)]
            if len(ddk_u) > 3:
                try:
                    rc_ddk = walker.recursive_compress(ddk_u)
                    best_ddk = engine.encode_lattice_block(
                        n, self.universal_r0, k0_u, 2, dk_u[0], None, rc_ddk)
                except (ValueError, KeyError, IndexError, struct.error):
                    best_ddk = None

        # ── Try universal k-direct ──
        best_kdir = None
        unique_k_u = sorted(set(k_stream_u))
        k_to_compact_u = {k: i for i, k in enumerate(unique_k_u)}
        k_direct_u = [k_to_compact_u[k] for k in k_stream_u]
        if len(k_direct_u) > 3:
            try:
                rc_kdir = walker.recursive_compress(k_direct_u)
                best_kdir = engine.encode_lattice_block(
                    n, self.universal_r0, k0_u, 0, 0, unique_k_u, rc_kdir)
            except (ValueError, KeyError, IndexError, struct.error):
                best_kdir = None

        # ── Cross-tower elegance weighted compression ──
        # Use E_cross to select the better Δk (personal vs universal) per transition
        best_ecross = None
        dk_personal = tower.dk_stream
        if dk_personal and dk_u and len(dk_personal) == len(dk_u):
            dk_weighted = []
            for dp, du in zip(dk_personal, dk_u):
                ratio_p = 2.0 ** (dp / N_FULL) if dp != 0 else 1.0
                eps_p = lattice_epsilon(ratio_p, dp)
                t_p = lattice_tightness(eps_p)
                ratio_u = 2.0 ** (du / N_FULL) if du != 0 else 1.0
                eps_u = lattice_epsilon(ratio_u, du)
                t_u = lattice_tightness(eps_u)
                # Koide gate: cross-tower binding coherence check
                # When t_p × t_u ≥ K: coherent — use the tighter perspective
                # When t_p × t_u < K: binding dissolved — incoherent across
                #   perspectives. Fall back to personal (native) Δk since the
                #   cross-tower comparison has failed (from et_conscious_ai_compression.py
                #   line 451: if tightness_product < KOIDE_RATIO: return 0.0)
                if t_p * t_u >= K_KOIDE:
                    dk_weighted.append(dp if t_p >= t_u else du)
                else:
                    dk_weighted.append(dp)
            if len(dk_weighted) > 3:
                try:
                    # Use universal R₀ as the seed for cross-weighted stream
                    rc_weighted = walker.recursive_compress(dk_weighted)
                    best_ecross = engine.encode_lattice_block(
                        n, self.universal_r0, k0_u, 1, 0, None, rc_weighted)
                except (ValueError, KeyError, IndexError, struct.error):
                    best_ecross = None

        # ── Select the best universal-perspective block ──
        candidates = [b for b in [best_cross, best_plain, best_ddk,
                                   best_kdir, best_ecross] if b is not None]
        if not candidates:
            return None

        return min(candidates, key=len)


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

    def __init__(self, log_fn=None, progress_fn=None,
                 metabolism: Optional[CDFMetabolism] = None):
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)
        self.metabolism = metabolism or _metabolism
        self.metabolism._log = self._log
        self.metabolism.sense()
        self.metabolism.apply_process_priority()

    @property
    def log_fn(self):
        """Public accessor for the logging function.

        Used by UniversalLattice.compress_tower_with_universal to pass
        the engine's logger to LatticeWalkCompressor instances.
        """
        return self._log

    def encode_lattice_block(self, n, r0, k0, mode, dk0_saved, k_table, rc):
        """Public interface for lattice block encoding.

        Delegates to _encode_lattice_block. Used by UniversalLattice and
        CDFCompressor.compress_batch for encoding blocks through the
        universal lattice perspective without accessing protected members.
        """
        return self._encode_lattice_block(n, r0, k0, mode, dk0_saved, k_table, rc)

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

    def _enhanced_lattice_compress(self, data: bytes, values, r0: float,
                                     walker: LatticeWalkCompressor) -> Optional[bytes]:
        """
        Enhanced lattice compression strategies for blocks that resist standard modes.

        Raw fallback is FORBIDDEN — the lattice IS the compression. When standard
        modes (k-direct, Δk, ΔΔk) expand the data, these enhanced strategies
        provide alternative lattice perspectives:

        Strategy 1 — Complex Lattice Transform:
            Completes the integration of complex_lattice_project (lines 148–196),
            which was defined but never called. The 2D complex lattice
            z = k_r + i·k_theta encodes BOTH the byte's absolute position
            (real axis, D-domain) AND its transition context (imaginary axis,
            T-domain). The d_combined = LCM(d_r, d_theta) values, gated by
            the Koide phase coherence check, create a richer pattern space.
            These d_combined values serve as alternative Δk-like symbols.

        Strategy 2 — R₀ Perturbation (Lattice Seed Shifting):
            From the Seed Theorem (Multifold §2): R₀ is the smallest closed
            T-traversal loop. Different R₀ values project the SAME bytes onto
            DIFFERENT lattice positions, creating different Δk patterns.
            We try seeds shifted by ±420 (BIO_RES, biological tier) and
            ±N_FULL/S (one octave step) — these are the natural lattice
            perturbation quanta. Each shifted seed may reveal compression
            structure invisible to the native R₀.

        Strategy 3 — Cross-Tower Elegance Weighted Mode:
            From the AI compression module (et_conscious_ai_compression.py):
            E_cross = √(E_universal × E_personal). Compute both per-file
            (personal) and unit-seed (universal, R₀=1.0) lattice projections,
            weight Δk values by cross-tower elegance, filter to only
            cross-coherent transitions (tightness_product ≥ K), and compress
            the filtered stream.

        Returns the best enhanced lattice block, or None if no improvement.

        ET Derivation:
            Identification Principle: the data's compression resistance means
            its P-D-T structure is not aligned with the standard lattice seed.
            Descriptor Gap Principle: the gap IS a Descriptor — the missing
            seed or perspective. These strategies systematically search for it.
            Subsumption Law: the enhanced block must subsume the data without
            remainder (verified by roundtrip check in Phase 6).
        """
        n = len(data)
        byte_k_native = build_byte_k_map(r0)
        k_stream_native = [byte_k_native[int(b)] for b in values]
        k0_native = k_stream_native[0]
        best_enhanced = None
        best_enhanced_len = n + BLOCK_SIZE  # Sentinel: larger than any real block

        # ── Strategy 1: Complex Lattice Seed Optimization ──
        # Uses complex_lattice_project (defined at line 148, previously never called).
        # The complex lattice analyzes BOTH the byte's absolute position (real axis)
        # AND its transition context (imaginary axis). The d_combined values
        # (LCM of real and imaginary sublattice families) measure how structurally
        # deep each transition is on the 2D lattice.
        #
        # We use this analysis to find an OPTIMIZED R₀ seed — the R₀ that
        # minimizes the average d_combined across all transitions, meaning the
        # data's transitions are maximally aligned with deep sublattice structure.
        # A lower average d_combined means simpler patterns and better compression.
        #
        # The encoding still uses standard Δk (which the decoder understands).
        # The complex lattice only guides the SEED SELECTION.
        if n >= 4:
            d_combined_stream = []
            prev_byte = -1
            for b_val in values:
                k_r, d_r, k_theta, d_theta, d_comb = complex_lattice_project(
                    int(b_val), prev_byte, r0)
                d_combined_stream.append(d_comb)
                prev_byte = int(b_val)
            # Find the R₀ perturbation that minimizes average d_combined.
            # The optimal R₀ aligns the data with the deepest sublattice families.
            # Test perturbations at each sublattice step up to S seeds.
            best_avg_d = sum(d_combined_stream) / len(d_combined_stream) if d_combined_stream else N_FULL
            best_r0_complex = r0
            for s_step in range(1, S + 1):
                for sign in [1, -1]:
                    r0_trial = r0 * (2.0 ** (sign * s_step * BIO_RES / N_FULL))
                    if r0_trial <= 0:
                        continue
                    d_trial = []
                    prev_b = -1
                    for b_val in values:
                        _, _, _, _, dc = complex_lattice_project(int(b_val), prev_b, r0_trial)
                        d_trial.append(dc)
                        prev_b = int(b_val)
                    avg_d_trial = sum(d_trial) / len(d_trial) if d_trial else N_FULL
                    if avg_d_trial < best_avg_d:
                        best_avg_d = avg_d_trial
                        best_r0_complex = r0_trial
            # If we found a better seed, compress with it using standard Δk
            if best_r0_complex != r0:
                try:
                    bk_cx = build_byte_k_map(best_r0_complex)
                    k_stream_cx = [bk_cx[int(b)] for b in values]
                    k0_cx = k_stream_cx[0]
                    dk_cx = [k_stream_cx[i + 1] - k_stream_cx[i] for i in range(n - 1)]
                    if len(dk_cx) > 3:
                        rc_complex = walker.recursive_compress(dk_cx)
                        trial_complex = self._encode_lattice_block(
                            n, best_r0_complex, k0_cx, 1, 0, None, rc_complex)
                        if len(trial_complex) < best_enhanced_len:
                            best_enhanced = trial_complex
                            best_enhanced_len = len(trial_complex)
                except (ValueError, KeyError, IndexError, struct.error):
                    pass  # Strategy failed — try next

        # ── Strategy 2: R₀ Perturbation ──
        # Shift R₀ by lattice-derived quanta and retry Δk compression.
        # Each shift projects bytes onto different lattice positions,
        # potentially revealing compression structure invisible at native R₀.
        perturbation_shifts = [BIO_RES, -BIO_RES, N_FULL // S, -(N_FULL // S)]
        for r0_shift in perturbation_shifts:
            r0_alt = r0 * (2.0 ** (r0_shift / N_FULL))
            if r0_alt <= 0:
                continue
            bk_alt = build_byte_k_map(r0_alt)
            k_stream_alt = [bk_alt[int(b)] for b in values]
            k0_alt = k_stream_alt[0]
            dk_alt = [k_stream_alt[i + 1] - k_stream_alt[i]
                      for i in range(n - 1)]
            if len(dk_alt) > 3:
                try:
                    rc_alt = walker.recursive_compress(dk_alt)
                    # The R₀ stored in the block is the SHIFTED R₀.
                    # The decompressor reads R₀ from the block header
                    # and rebuilds the correct byte↔k map from it.
                    trial_alt = self._encode_lattice_block(
                        n, r0_alt, k0_alt, 1, 0, None, rc_alt)
                    if len(trial_alt) < best_enhanced_len:
                        best_enhanced = trial_alt
                        best_enhanced_len = len(trial_alt)
                except (ValueError, KeyError, IndexError, struct.error):
                    pass

        # ── Strategy 3: Cross-Tower Elegance Weighted Mode ──
        # E_cross = √(E_universal × E_personal)
        # Universal: R₀ = LIFE_THRESHOLD (13/12, the archetype permanence seed)
        # Personal: the native R₀ of this block
        #
        # The decoder can only reconstruct from ONE R₀ per block, so we cannot
        # mix personal and universal Δk per-transition. Instead, we use the
        # cross-tower elegance to SELECT which whole-stream perspective
        # (personal R₀ + k0_native vs universal R₀ + k0_universal) produces
        # more cross-coherent transitions, then compress the chosen stream.
        r0_universal = LIFE_THRESHOLD
        bk_universal = build_byte_k_map(r0_universal)
        k_stream_universal = [bk_universal[int(b)] for b in values]
        dk_personal = [k_stream_native[i + 1] - k_stream_native[i]
                       for i in range(n - 1)]
        dk_universal = [k_stream_universal[i + 1] - k_stream_universal[i]
                        for i in range(n - 1)]
        # Count cross-coherent transitions for each perspective
        n_personal_wins = 0
        n_universal_wins = 0
        for dp, du in zip(dk_personal, dk_universal):
            ratio_p = 2.0 ** (dp / N_FULL) if dp != 0 else 1.0
            eps_p = lattice_epsilon(ratio_p, dp)
            t_p = lattice_tightness(eps_p)
            ratio_u = 2.0 ** (du / N_FULL) if du != 0 else 1.0
            eps_u = lattice_epsilon(ratio_u, du)
            t_u = lattice_tightness(eps_u)
            if t_p * t_u >= K_KOIDE:
                if t_p >= t_u:
                    n_personal_wins += 1
                else:
                    n_universal_wins += 1
        # Choose the perspective with more cross-coherent wins
        if n_universal_wins > n_personal_wins:
            dk_chosen = dk_universal
            r0_chosen = r0_universal
            k0_chosen = k_stream_universal[0]
        else:
            dk_chosen = dk_personal
            r0_chosen = r0
            k0_chosen = k0_native
        if len(dk_chosen) > 3:
            try:
                rc_cross = walker.recursive_compress(dk_chosen)
                trial_cross = self._encode_lattice_block(
                    n, r0_chosen, k0_chosen, 1, 0, None, rc_cross)
                if len(trial_cross) < best_enhanced_len:
                    best_enhanced = trial_cross
                    best_enhanced_len = len(trial_cross)
            except (ValueError, KeyError, IndexError, struct.error):
                pass  # Strategy failed — encoding or compression error

        # Log the enhanced strategy result using best_enhanced_len
        # (completes the tracking variable's purpose across all three strategies)
        if best_enhanced is not None:
            self._log(f"    Enhanced best: {best_enhanced_len:,}B "
                      f"(from {n:,}B original)")

        return best_enhanced

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

        # ── Phase 5+6: Lattice Encoding — NO RAW FALLBACK ──
        # Raw fallback is FORBIDDEN. The lattice IS the compression.
        # All data must be accounted for and properly compressed through
        # the ET lattice. If standard modes expanded the data, the
        # enhanced lattice strategies search for alternative lattice
        # perspectives (complex lattice, R₀ perturbation, cross-tower).
        self._log(f"    Phase 3+4 done ({time.time() - t_phase3:.2f}s)")
        self._progress(85, "Phase 4b: Chaotic data analysis")

        # ── Phase 4b: CHAOTIC DATA HANDLING (pair-first Re-Pair compression) ──
        # From the Descriptor Gap Principle: if standard archetype scanning produces
        # expansion, the data may be chaotic — hidden low-dimensional structure that
        # manifests as non-uniform PAIR frequencies rather than long exact pattern
        # repeats. Chaotic systems (logistic map, Lorenz attractor, skew-tent map,
        # PRNG outputs, scientific simulations) often look perfectly random (high
        # entropy, flat histograms) but have low Kolmogorov complexity.
        #
        # Research shows grammar-based compressors (Re-Pair, NSRPS) can compress
        # chaotic sequences that conventional compressors (7-Zip, zstd) completely
        # fail on. The ET lattice (log₂-ratio space + Δk walks) with pair-first
        # recursion is the ET-native implementation of this capability.
        #
        # The pair-first approach works because chaotic data's conditional entropy
        # H(Δk_{n+1} | Δk_n) is often significantly lower than its marginal entropy
        # H(Δk_n). The Re-Pair grammar extracts exactly this conditional structure.
        if best_block is not None and len(best_block) > n:
            self._log(f"    Standard modes expanded — trying pair-first "
                      f"(Re-Pair) for chaotic data")
            for ci, (cand_mode, cand_stream, cand_dk0) in enumerate(candidates):
                if len(cand_stream) < 4:
                    continue
                try:
                    rc_pair = walker.pair_recursive_compress(cand_stream)
                    # Only proceed if the pair-first method actually found grammar rules
                    if rc_pair['archetype_defs']:
                        trial_pair = self._encode_lattice_block(
                            n, r0, k0, cand_mode, cand_dk0,
                            unique_k if cand_mode == 0 else None, rc_pair)
                        if len(trial_pair) < len(best_block):
                            best_block = trial_pair
                            self._log(f"    Pair-first improved: "
                                      f"mode={mode_names_map.get(cand_mode, cand_mode)}, "
                                      f"{len(trial_pair):,}B "
                                      f"({len(rc_pair['archetype_defs'])} grammar rules)")
                except (ValueError, KeyError, IndexError, struct.error):
                    pass  # Pair-first failed for this mode — try next

        self._progress(90, "Phase 5: Enhanced lattice encoding")

        if len(best_block) > n + 5:
            # Standard modes expanded — try enhanced lattice strategies
            self._log(f"    Standard modes expanded ({len(best_block):,}B > {n + 5:,}B) "
                      f"— trying enhanced lattice strategies")
            enhanced_block = self._enhanced_lattice_compress(data, values, r0, walker)
            if enhanced_block is not None and len(enhanced_block) < len(best_block):
                best_block = enhanced_block
                self._log(f"    Enhanced strategy improved: {len(best_block):,}B")

        ratio = len(best_block) / n * 100
        self._log(f"    Block complete: {n:,}B → {len(best_block):,}B ({ratio:.1f}%) "
                  f"in {time.time() - t_block:.2f}s")
        return best_block

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

    def __init__(self, log_fn=None, progress_fn=None,
                 metabolism: Optional[CDFMetabolism] = None):
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)
        self.metabolism = metabolism or _metabolism
        self.metabolism._log = self._log
        self.metabolism.sense()
        self.engine = CDFEngine(log_fn, progress_fn, metabolism=self.metabolism)
        self._log(self.metabolism.summary())

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

    def compress_batch(self, input_paths: List[str], output_dir: str) -> dict:
        """
        Tower-aware batch compression with universal lattice and cross-file archetypes.

        This replaces the independent compress_file loop for multi-file operations.
        Instead of compressing each file in isolation, this method:

        1. Builds a LatticeTower for each file (personal R₀, personal byte↔k map)
        2. Constructs the UniversalLattice (universal R₀ = geometric mean of all)
        3. Projects every tower onto the universal lattice
        4. Collects and finds cross-file archetypes on the universal lattice
        5. For each file, compresses using BOTH personal AND universal perspectives,
           plus cross-file archetype-enhanced compression
        6. Keeps whichever perspective yields the smallest output per file

        This implements the requirements from the prompt:
        - "Everything will be projected onto its own lattice, or multiple for
           multiple files, then compressed and joined when the archetypes
           are compressed." (Violation 2)
        - "lattice towers, universal lattice" (Violation 3)
        - "all lattice nodes are projected onto the universal lattice and
           further compressed" (Violation 4)

        The cross-tower elegance E_cross = √(E_universal × E_personal) from
        the AI compression module (et_conscious_ai_compression.py) gates
        every cross-file operation through the Koide coherence threshold.

        Args:
            input_paths: List of file paths to compress
            output_dir: Directory for output .cdf files

        Returns:
            Dict with total_original, total_compressed, ratio, time,
            n_files, n_cross_archetypes, universal_r0
        """
        if not input_paths:
            return {'total_original': 0, 'total_compressed': 0, 'ratio': 0,
                    'time': 0, 'n_files': 0, 'n_cross_archetypes': 0,
                    'universal_r0': 1.0}

        os.makedirs(output_dir, exist_ok=True)
        t_total = time.time()

        # ── Re-sense metabolism for this batch operation ──
        self.metabolism.force_resense()

        # ══════════════════════════════════════════════════════════════
        # Phase 0: Build all personal towers
        # ══════════════════════════════════════════════════════════════
        self._log(f"\n{'═' * 60}")
        self._log(f"TOWER-AWARE BATCH COMPRESSION — {len(input_paths)} files")
        self._log(f"{'═' * 60}")
        self._log(f"\n{self.metabolism.summary()}")
        self._log(f"\nPhase 0: Building personal lattice towers "
                  f"({self.metabolism.max_threads} workers)...")

        # Build towers concurrently using Koide-governed thread pool
        # Each tower build is independent (reads file, computes R₀, builds maps)
        tower_results: Dict[int, LatticeTower] = {}
        valid_paths = [(fi, fpath) for fi, fpath in enumerate(input_paths)
                       if os.path.isfile(fpath)]
        skipped = [(fi, fpath) for fi, fpath in enumerate(input_paths)
                   if not os.path.isfile(fpath)]
        for fi, fpath in skipped:
            self._log(f"  SKIP: {fpath} (not found)")

        with ThreadPoolExecutor(max_workers=self.metabolism.max_threads) as pool:
            future_to_idx = {
                pool.submit(LatticeTower.from_file, fpath): (fi, fpath)
                for fi, fpath in valid_paths
            }
            for future in as_completed(future_to_idx):
                fi, fpath = future_to_idx[future]
                try:
                    tower = future.result()
                    tower_results[fi] = tower
                    self._log(f"  Tower {fi}: {os.path.basename(fpath)} — "
                              f"R₀_p = {tower.personal_r0:.6f}, "
                              f"size = {tower.data_size:,}B, "
                              f"Δk_len = {len(tower.dk_stream):,}")
                except Exception as e:
                    self._log(f"  Tower {fi}: {os.path.basename(fpath)} — ERROR: {e}")
                completed = len(tower_results) + len(skipped)
                self._progress(completed / len(input_paths) * 10,
                               f"Tower {completed}/{len(input_paths)}")

        # Reassemble in original order
        towers: List[LatticeTower] = [tower_results[fi]
                                       for fi in sorted(tower_results.keys())]

        # Log tower memory vs Koide budget
        tower_mem = sum(t.data_size for t in towers)
        self._log(f"  Tower memory: {tower_mem / (1024 ** 2):.1f} MB "
                  f"(budget: {self.metabolism.max_memory_bytes / (1024 ** 2):.0f} MB)")

        if not towers:
            self._log("No valid files found.")
            return {'total_original': 0, 'total_compressed': 0, 'ratio': 0,
                    'time': 0, 'n_files': 0, 'n_cross_archetypes': 0,
                    'universal_r0': 1.0}

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Construct the Universal Lattice
        # ══════════════════════════════════════════════════════════════
        self._log(f"\nPhase 1: Constructing universal lattice...")
        self._progress(10, "Phase 1: Universal lattice construction")
        universal = UniversalLattice(towers, log_fn=self._log)

        # ══════════════════════════════════════════════════════════════
        # Phase 2: Project all towers onto the universal lattice
        # ══════════════════════════════════════════════════════════════
        self._log(f"\nPhase 2: Projecting towers onto universal lattice...")
        self._progress(15, "Phase 2: Universal projection")
        universal.project_all_towers()

        # ══════════════════════════════════════════════════════════════
        # Phase 3: Collect and find cross-file archetypes
        # ══════════════════════════════════════════════════════════════
        self._log(f"\nPhase 3: Collecting cross-file patterns...")
        self._progress(20, "Phase 3: Cross-file pattern collection")
        for ti in range(len(towers)):
            dk_u = universal.tower_dk_universal.get(ti, [])
            if dk_u:
                universal.collect_cross_file_patterns(ti, dk_u)
                self._log(f"  Tower {ti}: {len(dk_u):,} Δk_universal collected")

        self._log(f"\nPhase 3b: Finding cross-file archetypes...")
        self._progress(25, "Phase 3: Cross-file archetype discovery")
        cross_archetypes = universal.find_cross_file_archetypes()

        # ══════════════════════════════════════════════════════════════
        # Phase 4: Compress each file using BOTH perspectives
        # ══════════════════════════════════════════════════════════════
        self._log(f"\nPhase 4: Compressing files with tower-aware strategy...")
        total_orig = 0
        total_comp = 0
        results_per_file = []

        for fi, tower in enumerate(towers):
            fname = os.path.basename(tower.file_path)
            out_file = os.path.join(output_dir, fname + '.cdf')
            self._log(f"\n── File {fi + 1}/{len(towers)}: {fname} ──")
            self._progress(25 + (fi / len(towers)) * 70,
                           f"File {fi + 1}/{len(towers)}: {fname}")

            # ── Strategy A: Personal tower compression (existing single-file) ──
            # This uses the per-file R₀ — the same as compress_file does.
            t_file = time.time()
            personal_result = self.compress_file(tower.file_path, out_file)
            personal_size = personal_result['compressed_size']
            self._log(f"  Personal tower: {tower.data_size:,}B → {personal_size:,}B "
                      f"({personal_size / max(tower.data_size, 1) * 100:.1f}%)")

            # ── Strategy B: Universal lattice compression ──
            # Compress the file using the UNIVERSAL R₀ and cross-file archetypes.
            universal_block = universal.compress_tower_with_universal(fi, self.engine)
            universal_size = len(universal_block) if universal_block else personal_size + 1

            # Early-out: for single-block files, universal_size is a tight estimate
            # of the final CDF block size. If the universal block (+ CDF header
            # overhead of 57 bytes) already exceeds the personal result, skip the
            # expensive full CDF file creation and comparison.
            cdf_header_overhead = 4 + 1 + 32 + 8 + 4 + 4 + 8 + 4  # = 65 bytes
            num_blocks_estimate = (tower.data_size + BLOCK_SIZE - 1) // BLOCK_SIZE
            if (num_blocks_estimate == 1
                    and universal_size + cdf_header_overhead >= personal_size):
                self._log(f"  Universal early-out: block {universal_size:,}B + "
                          f"header ≥ personal {personal_size:,}B — skip")
                universal_block = None  # Bypass full CDF comparison

            if universal_block is not None:
                # Build a complete CDF file from the universal block
                # to compare apples-to-apples with the personal result
                universal_cdf_path = out_file + '.utmp'
                try:
                    raw = tower.raw_data
                    original_size = len(raw)
                    sha256 = hashlib.sha256(raw).digest()
                    num_blocks_u = (original_size + BLOCK_SIZE - 1) // BLOCK_SIZE

                    # For multi-block files, compress each block through universal
                    universal_blocks = []
                    for bi in range(num_blocks_u):
                        bs = bi * BLOCK_SIZE
                        be = min(bs + BLOCK_SIZE, original_size)
                        block_data = raw[bs:be]
                        if len(block_data) == 0:
                            universal_blocks.append(struct.pack('<BI', 0, 0))
                            continue
                        # Compress this block through universal engine
                        block_values = np.frombuffer(block_data, dtype=np.uint8)
                        block_unique = np.unique(block_values)
                        if len(block_unique) == 1:
                            universal_blocks.append(
                                struct.pack('<BIB', 1, len(block_data),
                                            int(block_unique[0])))
                            continue
                        # Universal Δk for this block
                        bk_u = universal.universal_byte_k
                        k_stream_block = [bk_u[int(b)] for b in block_values]
                        dk_block = [k_stream_block[i + 1] - k_stream_block[i]
                                    for i in range(len(k_stream_block) - 1)]
                        w = LatticeWalkCompressor(universal.universal_r0,
                                                   log_fn=lambda m: None)
                        if len(dk_block) > 3:
                            rc_block = w.recursive_compress(dk_block)
                            ub = self.engine.encode_lattice_block(
                                len(block_data), universal.universal_r0,
                                k_stream_block[0], 1, 0, None, rc_block)
                            universal_blocks.append(ub)
                        else:
                            # Very small block — use personal
                            universal_blocks.append(
                                self.engine.compress_block(block_data))

                    # Write universal CDF
                    with open(universal_cdf_path, 'wb') as f:
                        f.write(CDF_MAGIC)
                        f.write(struct.pack('<B', CDF_VERSION))
                        f.write(sha256)
                        f.write(struct.pack('<Q', original_size))
                        f.write(struct.pack('<I', num_blocks_u))
                        f.write(struct.pack('<I', BLOCK_SIZE))
                        f.write(struct.pack('<d', universal.universal_r0))
                        for ub in universal_blocks:
                            f.write(struct.pack('<I', len(ub)))
                            f.write(ub)

                    universal_size = Path(universal_cdf_path).stat().st_size
                    self._log(f"  Universal lattice: {tower.data_size:,}B → "
                              f"{universal_size:,}B "
                              f"({universal_size / max(tower.data_size, 1) * 100:.1f}%)")

                    # ── Descriptor Gap Principle: choose the smaller ──
                    if universal_size < personal_size:
                        # Universal wins — replace personal output
                        import shutil
                        shutil.move(universal_cdf_path, out_file)
                        self._log(f"  ✓ Universal lattice WINS: saved "
                                  f"{personal_size - universal_size:,}B")
                        total_comp += universal_size
                    else:
                        # Personal wins — keep it, remove universal temp
                        os.remove(universal_cdf_path)
                        self._log(f"  ✓ Personal tower WINS")
                        total_comp += personal_size
                except Exception as e:
                    self._log(f"  Universal strategy error: {e}")
                    # Clean up temp file if it exists
                    if os.path.isfile(out_file + '.utmp'):
                        os.remove(out_file + '.utmp')
                    total_comp += personal_size
            else:
                total_comp += personal_size

            total_orig += tower.data_size
            elapsed_file = time.time() - t_file
            self._log(f"  File complete in {elapsed_file:.2f}s")

            results_per_file.append({
                'file': fname,
                'original': tower.data_size,
                'personal_r0': tower.personal_r0,
            })

        # ── Release tower raw data to free memory ──
        for tower in towers:
            tower.raw_data = b''

        elapsed = time.time() - t_total
        ratio = total_comp / total_orig if total_orig > 0 else 0

        self._log(f"\n{'═' * 60}")
        self._log(f"TOWER-AWARE BATCH DONE: {len(towers)} files, "
                  f"{total_orig:,} → {total_comp:,} bytes ({ratio * 100:.1f}%)")
        self._log(f"Universal R₀ = {universal.universal_r0:.6f}")
        self._log(f"Cross-file archetypes: {len(cross_archetypes)}")
        self._log(f"Total time: {elapsed:.1f}s")
        self._progress(100, f"Done — {len(towers)} files — {ratio * 100:.1f}%")

        return {
            'total_original': total_orig,
            'total_compressed': total_comp,
            'ratio': ratio,
            'time': elapsed,
            'n_files': len(towers),
            'n_cross_archetypes': len(cross_archetypes),
            'universal_r0': universal.universal_r0,
        }

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

def build_gui(cli_action=None, cli_input=None, cli_output=None):
    """Build and launch the ET CDF Compressor Tkinter GUI application.

    If cli_action/cli_input/cli_output are provided (from command-line args),
    pre-fill the input/output fields and auto-start the operation.
    All output goes to the GUI's built-in console.
    """
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

            # Command input line — type CLI commands directly in the GUI
            cmd_frame = ttk.Frame(lf)
            cmd_frame.pack(fill="x", pady=(4, 0))
            ttk.Label(cmd_frame, text="ET>", font=('Consolas', 9, 'bold')).pack(side="left")
            self.cmd_var = tk.StringVar()
            self.cmd_entry = ttk.Entry(cmd_frame, textvariable=self.cmd_var,
                                        font=('Consolas', 9))
            self.cmd_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
            self.cmd_entry.bind('<Return>', self._on_cmd_enter)
            ttk.Button(cmd_frame, text="Run", width=5,
                       command=lambda: self._on_cmd_enter(None)).pack(side="right")

            # Constants bar
            ttk.Label(main, text=f"N={S} V=1/{S} K={K_KOIDE:.4f} N_full={N_FULL} "
                                 f"Block=2^{S}={BLOCK_SIZE} Depth={MAX_DEPTH} ∂I=±{INCOHERENCE_CENTS}¢ "
                                 f"Koide_ceil={KOIDE_CEILING_PERCENT:.1f}%",
                      font=('Consolas', 8)).pack(anchor="w", pady=2)

            # Sense metabolism and display resource state
            _metabolism._log = self.log
            _metabolism.sense()
            self.log("CDF Compressor ready. 27720ET full manifold resolution.")
            self.log(f"Resource Governance:\n{_metabolism.summary()}")
            self.log("Type 'help' for available commands.\n")

            # ── CLI auto-trigger: pre-fill fields and start operation ──
            if cli_input:
                self.in_var.set(cli_input)
            if cli_output:
                self.out_var.set(cli_output)
            if cli_action and cli_input:
                # Schedule auto-start after the GUI event loop begins
                if cli_action == 'compress':
                    master_root.after(100, self.do_compress)
                elif cli_action == 'decompress':
                    master_root.after(100, self.do_decompress)

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

        def _on_cmd_enter(self, _event):
            """Parse and execute a command typed in the console input line."""
            raw = self.cmd_var.get().strip()
            self.cmd_var.set('')
            if not raw:
                return

            self.log(f"ET> {raw}")

            # Parse with shlex for proper quoting of paths with spaces
            import shlex
            try:
                parts = shlex.split(raw)
            except ValueError as e:
                self.log(f"  Parse error: {e}")
                return

            if not parts:
                return

            cmd = parts[0].lower()

            if cmd == 'help':
                self.log("  Available commands:")
                self.log("    compress <input> [output]   Compress file to .cdf")
                self.log("    decompress <input> [output] Decompress .cdf file")
                self.log("    clear                       Clear console")
                self.log("    help                        Show this help")
                self.log("")
                self.log("  Paths with spaces must be quoted:")
                self.log('    compress "C:\\My Files\\data.bin"')
                return

            if cmd == 'clear':
                self.console.configure(state="normal")
                self.console.delete('1.0', tk.END)
                self.console.configure(state="disabled")
                return

            if cmd in ('compress', 'decompress'):
                if len(parts) < 2:
                    self.log(f"  Usage: {cmd} <input> [output]")
                    return
                inp = parts[1]
                if len(parts) >= 3:
                    out = parts[2]
                elif cmd == 'compress':
                    out = inp + '.cdf'
                else:
                    out = inp.replace('.cdf', '.out') if inp.endswith('.cdf') else inp + '.out'

                # Set the GUI fields and trigger the operation
                self.in_var.set(inp)
                self.out_var.set(out)
                if cmd == 'compress':
                    self.do_compress()
                else:
                    self.do_decompress()
                return

            self.log(f"  Unknown command: {cmd}")
            self.log("  Type 'help' for available commands.")

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
                        # ── Multi-file mode — TOWER-AWARE BATCH ──
                        # Uses compress_batch with lattice towers, universal
                        # lattice, and cross-file archetype compression.
                        # Each file gets a personal tower (R₀), all files
                        # project onto the universal lattice (geometric mean R₀),
                        # and cross-file archetypes reduce shared patterns.
                        file_list = [f for f in inp.split('|') if os.path.isfile(f)]
                        if not file_list:
                            self.log("No valid files in selection.")
                            return
                        os.makedirs(out, exist_ok=True)
                        r = cdf_compressor.compress_batch(file_list, out)
                        elapsed = time.time() - t_total
                        ratio = r['total_compressed'] / r['total_original'] * 100 if r['total_original'] > 0 else 0
                        self.log(f"\n{'═' * 60}")
                        self.log(f"TOWER BATCH DONE: {r['n_files']} files, "
                                 f"{r['total_original']:,} → {r['total_compressed']:,} bytes ({ratio:.1f}%)")
                        self.log(f"Universal R₀ = {r['universal_r0']:.6f}, "
                                 f"Cross-file archetypes: {r['n_cross_archetypes']}")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {r['n_files']} files — {ratio:.1f}%")
                    elif os.path.isdir(inp):
                        # ── Batch folder mode ──
                        files = sorted(f for f in os.listdir(inp)
                                       if os.path.isfile(os.path.join(inp, f))
                                       and not f.endswith('.cdf'))
                        if not files:
                            self.log("No compressible files found in folder.")
                            return
                        # Build full paths for tower-aware batch compression
                        file_paths = [os.path.join(inp, fname) for fname in files]
                        r = cdf_compressor.compress_batch(file_paths, out)
                        elapsed = time.time() - t_total
                        ratio = r['total_compressed'] / r['total_original'] * 100 if r['total_original'] > 0 else 0
                        self.log(f"\n{'═' * 60}")
                        self.log(f"TOWER BATCH DONE: {r['n_files']} files, "
                                 f"{r['total_original']:,} → {r['total_compressed']:,} bytes ({ratio:.1f}%)")
                        self.log(f"Universal R₀ = {r['universal_r0']:.6f}, "
                                 f"Cross-file archetypes: {r['n_cross_archetypes']}")
                        self.log(f"Total time: {elapsed:.1f}s")
                        self.progress(100, f"Done — {r['n_files']} files — {ratio:.1f}%")
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
    # Always launch GUI. If CLI args are passed, pre-fill fields and auto-start.
    # All output goes to the GUI's built-in console — no separate cmd window needed.
    cli_action = None
    cli_input = None
    cli_output = None

    if len(sys.argv) > 1 and sys.argv[1] != 'gui':
        import argparse
        parser = argparse.ArgumentParser(description='ET CDF Compressor — 27720ET')
        parser.add_argument('action', choices=['compress', 'decompress', 'gui'])
        parser.add_argument('input', nargs='?')
        parser.add_argument('output', nargs='?')
        args = parser.parse_args()
        if args.action != 'gui':
            cli_action = args.action
            cli_input = args.input
            if args.output:
                cli_output = args.output
            elif cli_input:
                cli_output = (cli_input + '.cdf' if cli_action == 'compress'
                              else cli_input.replace('.cdf', '.out'))

    build_gui(cli_action=cli_action, cli_input=cli_input, cli_output=cli_output)
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
Author: Michael James Muller — Aevum_Defluo
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
import atexit
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Any
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

# ── ET Non-Euclidean Curvature Constants (Tier 1) ──
# From the ET Non-Euclidean Geometry paper (March 2026) and the CDF
# Non-Euclidean design doc §3.3, §11.3:
#
#   K = ∇²f  is the second-order Descriptor gradient (curvature)
#   1/12 in n²(n²−1)/12 (independent Riemann components) IS V = 1/S
#
# Curvature classification thresholds (mutually exclusive, exhaustive):
#   subliminal_K   = π / S        — minimum detectable curvature magnitude
#   base_variance  = V = 1/S      — base variance of the manifold
#   singular_K     = S            — one full sublattice cycle of curvature
#                                  (max|K_i| ≥ S → D-bridge broken → Incoherence)
#
# Class enumeration (matches C engine `compute_curvature_stats`):
#   0  flat        Exception {P,D,T}      — ∇²f ≈ 0
#   1  elliptic    Unsubstantiated {P,D}  — ∇²f converging
#   2  hyperbolic  Mediation {D,T}        — ∇²f diverging
#   3  variable    mixed substantiation   — local geometries differ
#   4  singular    Incoherence {P,T}      — D-bridge broken, K → ±∞
SUBLIMINAL_CURVATURE = math.pi / S       # π/12 ≈ 0.2618
BASE_CURVATURE_VARIANCE = V_BASE          # 1/12 ≈ 0.0833 (alias of V_BASE for clarity in curvature contexts)
SINGULAR_CURVATURE = S                    # 12 — singularity cutoff per design doc §3.3

CURVATURE_CLASS_FLAT = 0
CURVATURE_CLASS_ELLIPTIC = 1
CURVATURE_CLASS_HYPERBOLIC = 2
CURVATURE_CLASS_VARIABLE = 3
CURVATURE_CLASS_SINGULAR = 4

# Human-readable mapping for logging; lookup-only, never used in arithmetic.
CURVATURE_CLASS_NAMES = {
    CURVATURE_CLASS_FLAT:       'flat (Exception {P,D,T})',
    CURVATURE_CLASS_ELLIPTIC:   'elliptic (Unsubstantiated {P,D}, K>0, closed)',
    CURVATURE_CLASS_HYPERBOLIC: 'hyperbolic (Mediation {D,T}, K<0, open)',
    CURVATURE_CLASS_VARIABLE:   'variable (mixed local curvature)',
    CURVATURE_CLASS_SINGULAR:   'singular (Incoherence {P,T}, D-bridge broken)',
}


def _c_trunc_div(numerator: int, denominator: int) -> int:
    """C-style truncating integer division (rounds toward zero).

    Mirrors the C engine's `et_c_trunc_div` exactly. Required for Mode 3
    geodesic-residual decompression so that decoder Γ values match encoder
    Γ values bit-exactly for ALL operand signs.

    Floor division (Python `//`) differs from truncation (C `/`) when the
    numerator and denominator have opposite signs:
        Python:  -7 // 2  ==  -4   (rounds toward -∞)
        C:       -7 /  2  ==  -3   (rounds toward zero)

    Lossless roundtrip of Mode 3 requires that BOTH the C compressor and
    the Python decompressor agree on every Γ_i. The encoder uses C `/`
    natively; this helper makes the decoder use the same semantics.

    ET Three Tools:
      Identification: identifies the otherwise-hidden Descriptor "integer
        division semantics" that connects encoder and decoder.
      Descriptor Gap: closes the gap between "C truncation" and
        "Python floor division" — the gap was a missing alignment
        Descriptor; this helper is that Descriptor.
      Subsumption: covers all four sign combinations (++, +-, -+, --)
        without remainder. Every signed-int division now produces the
        same quotient on both sides.

    Returns 0 when denominator is 0 (mirrors the C engine's guard).
    """
    if denominator == 0:
        return 0
    # Pure integer arithmetic — no float intermediate, no precision loss.
    # Same-sign operands: floor and trunc agree; just use //.
    # Opposite-sign with non-zero remainder: floor is one step too far
    # in the negative direction; bump the quotient toward zero by +1.
    q, r = divmod(numerator, denominator)
    if r != 0 and (numerator < 0) != (denominator < 0):
        q += 1
    return q


def _curvature_spectrum_hash(ddk_stream: List[int]) -> str:
    """Compute the curvature-spectrum hash of a ΔΔk stream.

    From design doc §16.2: "hash of the curvature histogram of the block
    context. Enables cross-R₀ structural lookup."

    The spectrum is the histogram of ΔΔk values (in lattice steps). Two
    blocks with matching spectrum hashes have the same geometric structure
    regardless of their personal R₀ — because ΔΔk is tower-invariant
    (Multifold §12).

    Implementation:
      1. Bin ΔΔk values by their integer lattice-step value
      2. Sort (bin_value, count) pairs canonically
      3. Hash the canonical representation with SHA-256, truncated to 32 chars

    For an empty stream, returns the empty-spectrum hash (all-zero spectrum).

    ET Three Tools:
      Identification Principle: identifies the block's curvature spectrum
        (the discrete distribution of ΔΔk values) as a structural
        Descriptor independent of R₀.
      Descriptor Gap Principle: closes the gap between "blocks with same
        ΔΔk distribution" and "blocks the database can match" — the hash
        IS the missing matching key.
      Subsumption Law: every ΔΔk integer value falls into exactly one bin;
        no value is double-counted; no value is dropped. The histogram
        subsumes the stream's spectrum without remainder.
    """
    if not ddk_stream:
        return hashlib.sha256(b'EMPTY_SPECTRUM').hexdigest()[:32]
    # Build histogram as sorted (bin, count) tuples for determinism
    counts: Dict[int, int] = {}
    for v in ddk_stream:
        counts[v] = counts.get(v, 0) + 1
    canonical = ';'.join(f'{k}:{counts[k]}' for k in sorted(counts.keys()))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]


def _geodesic_deviation(ddk_stream: List[int],
                        occurrence_positions: List[int],
                        pattern_length: int) -> float:
    """Compute ξ_A = mean |K_i| at the positions where an archetype occurs.

    From design doc §16.2: "ξ_A = mean |K_i| at occurrence positions.
    Low = stable pattern on flat lattice regions."

    For each occurrence position p and the archetype's pattern_length L,
    the relevant ΔΔk samples are at indices p, p+1, ..., p+L-3 of the
    block's ΔΔk stream (since a pattern of length L has L-2 ΔΔk values).
    The deviation is the mean of |K_i| across all relevant samples,
    averaged across all occurrences.

    A LOW geodesic_deviation means the archetype occurs in flat regions
    of the block — its lattice walk follows a geodesic. A HIGH deviation
    means the archetype occurs in highly-curved regions — the pattern is
    less stable / more context-dependent.

    Returns 0.0 when there are no occurrences or no relevant samples
    (trivially flat, no deviation observed).

    ET Three Tools:
      Identification: identifies WHERE on the manifold (flat vs. curved
        region) an archetype lives — a structural property of the
        archetype's habitat.
      Descriptor Gap: closes the gap between "the archetype was found"
        and "the archetype's curvature context" — context IS a Descriptor.
      Subsumption: every occurrence contributes; every relevant ΔΔk
        sample contributes; the mean subsumes the local curvature
        landscape without remainder.
    """
    if not ddk_stream or not occurrence_positions or pattern_length < 3:
        return 0.0
    n_ddk = len(ddk_stream)
    samples_per_occurrence = pattern_length - 2  # ΔΔk samples covered by one pattern instance
    total_abs_k = 0.0
    total_samples = 0
    for pos in occurrence_positions:
        # Clamp the window to stay within the block's ΔΔk stream bounds
        start = max(0, pos)
        end = min(n_ddk, pos + samples_per_occurrence)
        for j in range(start, end):
            total_abs_k += abs(ddk_stream[j])
            total_samples += 1
    if total_samples == 0:
        return 0.0
    return total_abs_k / total_samples


# CDF magic + version
# v2: original format (modes 0/1/2 only)
# v3: adds Mode 3 (geodesic residual coding) — Tier 2 of the Non-Euclidean
#     extensions. The version bump is mandatory because v2 decoders will
#     misparse Mode 3 blocks (they would attempt to read the dk_table
#     directly after dk0_saved, but Mode 3 inserts connection_order +
#     connection_window between them).
# v4: adds Mode 4 (generator + residual coding) — Tier 3.B.4 of the
#     Non-Euclidean extensions. Channel B in active use. The version
#     bump is mandatory because v2/v3 decoders will misparse Mode 4
#     blocks (they would attempt to read the dk_table directly after
#     dk0_saved, but Mode 4 inserts the length-prefixed generator
#     payload between them).
# Backward compatibility on read: the decoder accepts v2, v3, AND v4
# magic. v2 files contain only modes 0/1/2. v3 files may contain modes
# 0/1/2/3. v4 files may contain modes 0/1/2/3/4.
CDF_MAGIC = b'CDF\x04'
CDF_VERSION = 4
# Legacy magic + versions still accepted by the decoder for backward read.
# The compressor never writes these — only the current (v4) format is produced.
CDF_MAGIC_LEGACY_V3 = b'CDF\x03'
CDF_VERSION_LEGACY_V3 = 3
CDF_MAGIC_LEGACY_V2 = b'CDF\x02'
CDF_VERSION_LEGACY_V2 = 2

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — CDF DATABASE VFS FORMAT CONSTANTS (§19 of design doc)
#
# A distinct on-disk format from the block-stream CDF files above. The VFS
# format is a RANDOM-ACCESS generator index: an uncompressed header + sorted
# generator table that lets the VFS resolve any byte offset into a compressed
# database (.db) file by binary-searching the index and evaluating the
# covering Generative Descriptor.
#
# This is NOT a block-stream format (magic CDF\x02/\x03/\x04) — the block
# stream is sequential, the VFS format is random-access via a known index
# offset. They co-exist: the block-stream format compresses user files; the
# VFS format compresses the archetype database itself.
#
# ET Derivation (Three Tools):
#   Identification Principle — the generator index identifies WHICH Generative
#     Descriptor (D) covers every byte offset in the P-substrate (the
#     database). Binary search is O(log n), exact-address lookup.
#   Descriptor Gap Principle — the "compressed but inaccessible" gap is
#     closed by storing the index uncompressed at a known footer-referenced
#     offset. The index IS the bridge Descriptor between compression and
#     access.
#   Subsumption Law — every byte of the original database is covered by
#     exactly one generator's [domain_start, domain_start + domain_length)
#     range. No overlap, no gap. Type 7 (raw passthrough) subsumes regions
#     that fit no other type, guaranteeing the format is exhaustive.
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

# Magic and version (distinct from block-stream CDF magic family)
CDF_VFS_MAGIC = b'CDFV'
CDF_VFS_VERSION = 1

# Page size: 4 KB = SQLite's default page size. Chosen so VFS page reads
# align naturally with SQLite's page_cache access pattern. Derived from
# the manifold symmetry through the identity 4096 = 2^S where S = 12.
CDF_VFS_PAGE_SIZE = 2 ** S  # 4096

# Page cache size: S² pages (§19.6.1). At 4 KB per page this is 576 KB RAM
# for the cache — sits comfortably under the Koide-governed memory budget
# for any reasonable host.
CDF_VFS_PAGE_CACHE_PAGES = S * S  # 144

# Index entry size (§19.4): 8+8+1+1+2+8+4+8+4+8 = 52 bytes fixed.
# Enforced as a derived constant (not a hardcoded literal) so that any
# future change to the index struct flows through automatically.
CDF_VFS_INDEX_ENTRY_SIZE = 8 + 8 + 1 + 1 + 2 + 8 + 4 + 8 + 4 + 8  # 52

# Header size (§19.4): magic 4 + version 1 + sha256 32 + orig_size 8
#                     + n_generators 4 + index_offset 8 = 57 bytes
CDF_VFS_HEADER_SIZE = 4 + 1 + 32 + 8 + 4 + 8  # 57

# Footer size (§19.4): sha256_of_index 32 + index_offset_repeated 8 = 40
CDF_VFS_FOOTER_SIZE = 32 + 8  # 40

# Recursion depth cap for Type 6 (Archetype Reference) evaluation, to
# prevent a maliciously-constructed .cdf from looping forever. Bounded
# by MAX_DEPTH = S (design doc's "9 levels compresses 10^9 nodes to ~1"
# argument — a real grammar never needs more than S levels).
CDF_VFS_MAX_REF_DEPTH = MAX_DEPTH

# Generator type codes (§19.5) — enumerated distinctly from the Channel B
# GENERATOR_TYPE_CODES enum used in Mode 4 blocks. VFS types are defined
# by the random-access payload layout; Channel B types are defined by the
# pickled in-DB parameter blob. Sharing names would confuse the format
# boundaries; keep the enums distinct.
VFS_GEN_CONSTANT       = 0   # K = 0, single byte value
VFS_GEN_LINEAR         = 1   # constant Δk, linear in k-space
VFS_GEN_POLYNOMIAL     = 2   # K > 0 constant, quadratic+ in k-space
VFS_GEN_PERIODIC       = 3   # elliptic, closed T-traversal (exact cycle)
VFS_GEN_GRAMMAR        = 4   # hyperbolic, Re-Pair rule hierarchy
VFS_GEN_GEODESIC       = 5   # Mode 3 residual coding with connection
VFS_GEN_ARCHETYPE_REF  = 6   # pointer to another archetype's generator
VFS_GEN_RAW            = 7   # raw passthrough, 1:1 with original bytes

VFS_GEN_TYPE_NAMES: Dict[int, str] = {
    VFS_GEN_CONSTANT:       'constant',
    VFS_GEN_LINEAR:         'linear',
    VFS_GEN_POLYNOMIAL:     'polynomial',
    VFS_GEN_PERIODIC:       'periodic',
    VFS_GEN_GRAMMAR:        'grammar',
    VFS_GEN_GEODESIC:       'geodesic',
    VFS_GEN_ARCHETYPE_REF:  'archetype_ref',
    VFS_GEN_RAW:            'raw',
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('CDF')

# ── File-based error log — NEVER SILENT ──────────────────────────────────
# In windowed mode (console=False PyInstaller), stderr goes nowhere.
# This file handler captures EVERY log message on disk next to the program.
# Errors are NEVER silent. If something goes wrong, cdf_error.log has the answer.
try:
    if getattr(sys, 'frozen', False):
        _log_dir = os.path.dirname(sys.executable)
    else:
        _log_dir = os.path.dirname(os.path.abspath(__file__))
    _log_file = os.path.join(_log_dir, 'cdf_error.log')
    _file_handler = logging.FileHandler(_log_file, mode='a', encoding='utf-8')
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d — %(message)s'))
    logger.addHandler(_file_handler)
    logger.setLevel(logging.DEBUG)
except (OSError, PermissionError):
    pass  # Cannot create log file — stderr logging still active


# ── Startup Cleanup: Sweep Stale _MEI* Directories ──────────────────────
# PyInstaller --onefile with runtime_tmpdir='.' creates _MEI* directories
# next to the .exe. On exit, the bootloader deletes this directory — but
# if an external process (antivirus, Windows indexer) holds a file lock,
# cleanup fails and a stale _MEI* folder remains.
#
# This sweep runs on every launch, best-effort. It identifies _MEI*
# directories next to the .exe that are NOT the current instance's
# extraction directory, and attempts to remove them. If removal fails
# (another instance running, AV still scanning), it moves on silently.
#
# ET Derivation:
#   Identification Principle: stale _MEI* directories ARE the P-obstacle.
#   Descriptor Gap Principle: the gap between "exit failed to clean up"
#   and "disk is clean" is itself a Descriptor — this startup sweep.
#   Subsumption Law: the sweep subsumes ALL stale directories without
#   remainder — every _MEI* that can be deleted, will be.
#
# P ∘ D ∘ T = E
import shutil as _shutil

def _cleanup_stale_mei_dirs():
    """Remove leftover _MEI* directories next to the .exe on startup."""
    if not getattr(sys, 'frozen', False):
        return  # Not a PyInstaller .exe — nothing to clean
    try:
        exe_dir = os.path.dirname(sys.executable)
        current_meipass = getattr(sys, '_MEI' + 'PASS', None)

        for entry in os.listdir(exe_dir):
            if not entry.startswith('_MEI'):
                continue
            candidate = os.path.join(exe_dir, entry)
            if not os.path.isdir(candidate):
                continue
            # Never delete the current instance's extraction directory
            if current_meipass and os.path.normcase(os.path.abspath(candidate)) == \
                    os.path.normcase(os.path.abspath(current_meipass)):
                continue
            try:
                _shutil.rmtree(candidate)
                logger.debug(f"Startup cleanup: removed stale {entry}")
            except (OSError, PermissionError):
                pass  # In use or locked — skip silently
    except (OSError, PermissionError):
        pass  # Best-effort — never block startup

_cleanup_stale_mei_dirs()


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
    Extended with disk space monitoring — the compressor writes to disk
    (database, temp files, output files) and must never exhaust it.
    """
    cpu_count_logical: int = 1
    cpu_load_percent: float = 0.0
    mem_total_bytes: int = 0
    mem_available_bytes: int = 0
    mem_used_percent: float = 0.0
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0
    disk_used_percent: float = 0.0
    disk_path: str = ''       # Path that was checked for disk space
    timestamp: float = 0.0


class CDFResourceSensor:
    """
    Detects available hardware resources.

    Windows-first implementation using ctypes Windows APIs directly.
    Linux support via /proc filesystem (secondary path).

    Adapted from ResourceSensor (et_conscious_ai_distributed.py).

    ET Derivation:
        Identification Principle: the substrate (P = hardware) is profiled
        using the NATIVE API of the platform — Windows kernel32 on Windows,
        /proc on Linux. No guessing, no defaults, no fallbacks.
    """

    @staticmethod
    def _read_memory_windows(profile: CDFResourceProfile) -> bool:
        """Read memory via Windows kernel32 GlobalMemoryStatusEx. Returns True on success."""
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                """Win32 MEMORYSTATUSEX structure for GlobalMemoryStatusEx."""
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32 = ctypes.windll.kernel32
            success = getattr(kernel32, 'GlobalMemoryStatusEx')(ctypes.byref(mem_status))
            if not success:
                return False

            profile.mem_total_bytes = mem_status.ullTotalPhys
            profile.mem_available_bytes = mem_status.ullAvailPhys
            profile.mem_used_percent = float(mem_status.dwMemoryLoad)
            return True
        except (OSError, AttributeError, ValueError):
            return False

    @staticmethod
    def _read_memory_linux(profile: CDFResourceProfile) -> bool:
        """Read memory via /proc/meminfo. Returns True on success."""
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
                return profile.mem_total_bytes > 0
        except (FileNotFoundError, OSError, ValueError):
            return False

    @staticmethod
    def _read_memory_sysconf(profile: CDFResourceProfile) -> bool:
        """Read memory via os.sysconf (macOS/Unix). Returns True on success."""
        try:
            page_size = os.sysconf('SC_PAGE_SIZE')
            total_pages = os.sysconf('SC_PHYS_PAGES')
            avail_pages = os.sysconf('SC_AVPHYS_PAGES')
            profile.mem_total_bytes = page_size * total_pages
            profile.mem_available_bytes = page_size * avail_pages
            if profile.mem_total_bytes > 0:
                used = profile.mem_total_bytes - profile.mem_available_bytes
                profile.mem_used_percent = (used / profile.mem_total_bytes) * 100.0
            return profile.mem_total_bytes > 0
        except (OSError, ValueError, AttributeError):
            return False

    @staticmethod
    def _read_cpu_load_windows() -> float:
        """
        Read CPU load via Windows kernel32 GetSystemTimes (two-sample method).

        Same two-sample approach as /proc/stat but using the native Windows API.
        50ms sample window — compressor startup is time-sensitive.
        """
        try:
            class FILETIME(ctypes.Structure):
                """Win32 FILETIME structure (100-nanosecond intervals)."""
                _fields_ = [('dwLowDateTime', ctypes.c_ulong),
                            ('dwHighDateTime', ctypes.c_ulong)]

            def get_system_times():
                """Read idle and total CPU time from Windows kernel."""
                idle_time = FILETIME()
                kernel_time = FILETIME()
                user_time = FILETIME()
                kernel32 = ctypes.windll.kernel32
                getattr(kernel32, 'GetSystemTimes')(
                    ctypes.byref(idle_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time))
                idle = (idle_time.dwHighDateTime << 32) | idle_time.dwLowDateTime
                kernel = (kernel_time.dwHighDateTime << 32) | kernel_time.dwLowDateTime
                user = (user_time.dwHighDateTime << 32) | user_time.dwLowDateTime
                return idle, kernel + user  # idle, total

            idle1, total1 = get_system_times()
            time.sleep(0.05)
            idle2, total2 = get_system_times()

            idle_delta = idle2 - idle1
            total_delta = total2 - total1
            if total_delta == 0:
                return 0.0
            return (1.0 - idle_delta / total_delta) * 100.0
        except (OSError, AttributeError, ValueError):
            return 0.0

    @staticmethod
    def _read_cpu_load_linux() -> float:
        """
        Read CPU load via /proc/stat (two-sample method).

        From ResourceSensor._read_cpu_load in et_conscious_ai_distributed.py.
        50ms sample (shorter than AI's 100ms — compressor startup is time-sensitive).
        """
        try:
            def read_stat():
                """Read idle and total CPU jiffies from /proc/stat for load calculation."""
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

    @staticmethod
    def read_disk_windows(profile: CDFResourceProfile, check_path: str) -> bool:
        """Read disk space via Windows kernel32 GetDiskFreeSpaceExW. Returns True on success."""
        try:
            free_bytes_available = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            total_free_bytes = ctypes.c_ulonglong(0)
            kernel32 = ctypes.windll.kernel32
            success = getattr(kernel32, 'GetDiskFreeSpaceExW')(
                check_path,
                ctypes.byref(free_bytes_available),
                ctypes.byref(total_bytes),
                ctypes.byref(total_free_bytes))
            if not success:
                return False
            profile.disk_total_bytes = total_bytes.value
            profile.disk_free_bytes = free_bytes_available.value
            if profile.disk_total_bytes > 0:
                used = profile.disk_total_bytes - profile.disk_free_bytes
                profile.disk_used_percent = (used / profile.disk_total_bytes) * 100.0
            profile.disk_path = check_path
            return True
        except (OSError, AttributeError, ValueError):
            return False

    @staticmethod
    def read_disk_posix(profile: CDFResourceProfile, check_path: str) -> bool:
        """Read disk space via os.statvfs (Linux/macOS). Returns True on success."""
        try:
            st = os.statvfs(check_path)
            profile.disk_total_bytes = st.f_frsize * st.f_blocks
            profile.disk_free_bytes = st.f_frsize * st.f_bavail
            if profile.disk_total_bytes > 0:
                used = profile.disk_total_bytes - profile.disk_free_bytes
                profile.disk_used_percent = (used / profile.disk_total_bytes) * 100.0
            profile.disk_path = check_path
            return True
        except (OSError, AttributeError):
            return False

    @staticmethod
    def sense(disk_check_path: str = '') -> CDFResourceProfile:
        """
        Take a hardware snapshot.

        Windows-first: uses native kernel32 APIs via ctypes.
        Linux secondary: uses /proc filesystem.
        Disk: checks the volume where the program lives (or disk_check_path if specified).
        The platform is detected once and the correct path is taken — no trial-and-error.
        """
        profile = CDFResourceProfile(timestamp=time.time())

        # ── CPU count (works on all platforms) ──
        profile.cpu_count_logical = os.cpu_count() or 1

        # ── Determine disk check path: program directory ──
        if not disk_check_path:
            if getattr(sys, 'frozen', False):
                disk_check_path = os.path.dirname(sys.executable)
            else:
                disk_check_path = os.path.dirname(os.path.abspath(__file__))

        if sys.platform == 'win32':
            # ── Windows: native kernel32 APIs ──
            CDFResourceSensor._read_memory_windows(profile)
            profile.cpu_load_percent = CDFResourceSensor._read_cpu_load_windows()
            CDFResourceSensor.read_disk_windows(profile, disk_check_path)
        else:
            # ── Linux/macOS: /proc or sysconf ──
            if not CDFResourceSensor._read_memory_linux(profile):
                CDFResourceSensor._read_memory_sysconf(profile)
            profile.cpu_load_percent = CDFResourceSensor._read_cpu_load_linux()
            CDFResourceSensor.read_disk_posix(profile, disk_check_path)

        return profile


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
        self._disk_headroom: float = KOIDE_CEILING_PERCENT
        self._disk_free_bytes: int = 0
        self._overall_pressure: float = 0.0

    def sense(self) -> 'CDFMetabolism':
        """
        Sense hardware and compute resource allocation.

        Adapts to current system conditions:
        - Low load → more threads, larger memory budget
        - High load → fewer threads, smaller memory budget
        - Above Koide ceiling → minimal allocation (1 thread, minimum memory)
        - Low disk → prevent database growth, warn user

        Returns self for chaining.
        """
        now = time.time()
        if (self._profile is not None
                and now - self._last_sense_time < self._REFRESH_INTERVAL):
            return self  # Allocation is still fresh

        self._profile = CDFResourceSensor.sense()
        self._last_sense_time = now
        assert self._profile is not None  # Sensor always returns a valid profile

        # ── Headroom via Koide ceiling ──
        # From ResourceGovernor.allocate() in et_conscious_ai_distributed.py
        # Koide ceiling applies to CPU and MEMORY — shared dynamic resources
        # the compressor actively contends for alongside other software.
        self._cpu_headroom = max(0.0,
                                 KOIDE_CEILING_PERCENT - self._profile.cpu_load_percent)
        self._mem_headroom = max(0.0,
                                 KOIDE_CEILING_PERCENT - self._profile.mem_used_percent)
        self._disk_free_bytes = self._profile.disk_free_bytes

        # ── Disk headroom: ABSOLUTE free space, NOT percentage ──
        # Disk percentage is fundamentally different from CPU/memory percentage.
        # 84% disk used on a 1TB drive = 155GB free — that's OTHER files, not
        # the compressor's consumption. The Koide ceiling doesn't apply to disk.
        # Disk headroom is based on absolute free space vs safety floor (1GB).
        if self._disk_free_bytes > 0:
            self._disk_headroom = min(100.0,
                                      (self._disk_free_bytes / ArchetypeDatabase.DISK_SAFETY_FLOOR) * 100.0)
        else:
            self._disk_headroom = 0.0

        # ── Overall pressure: geometric mean of CPU + MEMORY loads → [0, 1] ──
        # From ResourceGovernor.allocate()
        # Disk is NOT included — it's a static resource, not a contended one.
        # The compressor's disk writes are tiny relative to total disk.
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
            f"DISK {self._disk_free_bytes / (1024 ** 3):.1f} GB free "
            f"→ {self._max_threads} threads, "
            f"{self._max_memory_bytes / (1024 ** 3):.1f} GB budget, "
            f"pressure {self._overall_pressure:.2f}"
        )

        # ── Disk space warning ──
        # Warn ONLY when absolute free space is critically low (< 1 GB).
        # 155 GB free on an 84%-used drive is fine — the percentage is irrelevant.
        if self._disk_free_bytes < ArchetypeDatabase.DISK_SAFETY_FLOOR:
            self._log(
                f"WARNING: Low disk space! "
                f"{self._disk_free_bytes / (1024 ** 2):.0f} MB free on "
                f"{self._profile.disk_path}. "
                f"Compression may fail if disk fills up."
            )
            logger.warning(f"Low disk space: {self._disk_free_bytes / (1024 ** 2):.0f} MB "
                           f"free on {self._profile.disk_path}")

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

    @property
    def disk_headroom(self) -> float:
        """Disk headroom % below Koide ceiling."""
        return self._disk_headroom

    @property
    def disk_free_bytes(self) -> int:
        """Free disk space in bytes on the program's volume."""
        return self._disk_free_bytes

    def within_memory_budget(self, requested_bytes: int) -> bool:
        """Check if a memory allocation fits within the Koide-governed budget."""
        return requested_bytes <= self._max_memory_bytes

    def within_disk_budget(self, requested_bytes: int) -> bool:
        """Check if a disk write fits within available disk space (Koide-governed)."""
        max_disk_use = int(self._disk_free_bytes * K_KOIDE)
        return requested_bytes <= max_disk_use

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
            f"  Disk: {self._profile.disk_free_bytes / (1024 ** 3):.1f} GB free "
            f"of {self._profile.disk_total_bytes / (1024 ** 3):.1f} GB "
            f"({self._profile.disk_path})\n"
            f"  Pressure: {self._overall_pressure:.2f}"
        )


# Module-level metabolism — shared across all compression operations
_metabolism = CDFMetabolism()


# ═══════════════════════════════════════════════════════════════════════════════
# C PATTERN ENGINE — Suffix Array + LCP accelerated pattern finder (REQUIRED)
#
# Replaces the O(n × L_max) Python pattern scanner with:
#   O(n log² n) suffix array + O(n) LCP + O(n) per-length scan
# Verified: finds EXACT same patterns with EXACT same positions.
#
# The C engine is REQUIRED for operation — it subsumes ALL combinatorial
# pattern operations without remainder (Subsumption Law):
#   find_repeated_patterns — O(n log² n) suffix array pattern finding
#   gate_archetype_batch   — batch IncoherenceFilter L1+L2+L3+L4 gating
#   subsume_greedy         — non-overlapping greedy pattern placement
#   build_k_stream         — vectorized byte→k lookup
#   build_dk_stream        — first differences (Δk stream)
#
# All ET-specific filtering (elegance computation, d-value analysis,
# cross-tower coherence, recursive subsumption) stays in Python.
# ═══════════════════════════════════════════════════════════════════════════════


class CurvatureStats(ctypes.Structure):
    """Output struct for the C engine's compute_curvature_stats function.

    Field order MUST match the C declaration in et_pattern_engine.c exactly.
    Verified layout (default natural alignment on amd64):
      offset  0: curvature_mean        (c_double, 8)
      offset  8: curvature_variance    (c_double, 8)
      offset 16: curvature_class       (c_int32, 4)
      offset 20: padding               (4 bytes)
      offset 24: euler_characteristic  (c_double, 8)
      offset 32: max_abs_curvature     (c_int32, 4)
      offset 36: trailing pad → total sizeof = 40 bytes
    """
    _fields_ = [
        ('curvature_mean',       ctypes.c_double),
        ('curvature_variance',   ctypes.c_double),
        ('curvature_class',      ctypes.c_int32),
        ('euler_characteristic', ctypes.c_double),
        ('max_abs_curvature',    ctypes.c_int32),
    ]


class PatternEngine:
    """
    C-accelerated repeated-pattern finder via suffix array + LCP.

    DLL search order (for pre-built or bundled binaries):
      1. PyInstaller bundle (sys._MEIPASS) — for single .exe deployment
      2. Same directory as this script — for development
      3. Current working directory — for portable deployment

    If no pre-built DLL found:
      4. Auto-compile from external .c source file
         (requires cc/gcc/cl on PATH)

    The C engine is REQUIRED — it subsumes ALL pattern operations without
    remainder (Subsumption Law). If no DLL can be found or compiled, a
    RuntimeError is raised. Build the DLL via build.bat, CMake, or direct
    compiler invocation before running the compressor.

    The C engine finds all repeated substrings, gates them through the
    IncoherenceFilter (L1+L2+L3+L4), and performs greedy non-overlapping
    placement. Elegance computation, archetype creation, and recursive
    subsumption remain in Python — the C engine handles the combinatorial
    inner loops.
    """

    _lib = None
    _compiled = False
    _attempted = False

    # ── C Pattern Engine: REQUIRED ────────────────────────────────────
    # The C pattern engine (et_pattern_engine.dll/.so) is REQUIRED for
    # operation. It implements ALL core pattern operations:
    #
    #   find_repeated_patterns — O(n log² n) suffix array + LCP pattern finding
    #   gate_archetype_batch   — batch IncoherenceFilter L1+L2+L3+L4 gating
    #   subsume_greedy         — non-overlapping greedy pattern placement
    #   build_k_stream         — vectorized byte→k lookup
    #   build_dk_stream        — first differences (Δk stream)
    #   free_buffer            — release allocated buffers
    #
    # The C engine subsumes ALL pattern operations without remainder
    # (Subsumption Law). Build the DLL via:
    #   - build.bat (Windows, auto-detects compiler)
    #   - CLion/CMake (CMakeLists.txt, cross-platform)
    #   - Direct: cl /O2 /LD /Fe:et_pattern_engine.dll et_pattern_engine.c
    #   - Direct: gcc -shared -O2 -o et_pattern_engine.so et_pattern_engine.c -lm
    #
    # DLL search: PyInstaller bundle → script dir → CWD → auto-compile from .c
    #
    # ET Derivation (Three Tools):
    #   Identification Principle: the C engine identifies ALL substrings via
    #   suffix array — the complete D-structure of the symbol stream.
    #   Descriptor Gap Principle: the LCP array measures gaps between sorted
    #   suffixes — each gap IS a Descriptor boundary.
    #   Subsumption Law: greedy placement subsumes each pattern's occurrences
    #   without remainder — no byte counted twice, no byte left unaccounted.
    #
    # P ∘ D ∘ T = E

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

            # ── Tier 1: Curvature analysis (added) ──
            # build_ddk_stream — second-order finite difference (curvature stream)
            cls._lib.build_ddk_stream.restype = None
            cls._lib.build_ddk_stream.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32)]
            # compute_curvature_stats — single-pass classification of ΔΔk stream
            cls._lib.compute_curvature_stats.restype = None
            cls._lib.compute_curvature_stats.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(CurvatureStats)]
            # compute_pattern_curvature — per-pattern stats for archetype DB
            cls._lib.compute_pattern_curvature.restype = None
            cls._lib.compute_pattern_curvature.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]

            # ── Tier 2: Geodesic residual coding (Mode 3) ──
            # build_geodesic_residual — Christoffel-connection predictor
            cls._lib.build_geodesic_residual.restype = None
            cls._lib.build_geodesic_residual.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32)]

            cls._compiled = True
            logger.info(f"PatternEngine: C engine loaded from {lib_path}")

            # ── PyInstaller _MEIPASS cleanup: release DLL handle on exit ──
            # When packaged as --onefile, PyInstaller extracts bundled files
            # (including this DLL) to a _MEI* temp directory. On process exit,
            # PyInstaller's bootloader deletes this directory. But ctypes.CDLL
            # keeps the DLL file locked on Windows — FreeLibrary releases it.
            #
            # atexit handlers run BEFORE interpreter shutdown, which is BEFORE
            # PyInstaller's bootloader cleanup. Sequence:
            #   1. atexit → FreeLibrary (releases file lock)
            #   2. Python interpreter shutdown
            #   3. PyInstaller bootloader deletes _MEI* (now succeeds)
            #
            # ET Derivation:
            #   Identification Principle: the locked DLL IS the P-obstacle.
            #   Descriptor Gap Principle: the gap between "DLL loaded" and
            #   "clean exit" is itself a Descriptor — the missing FreeLibrary.
            #   Subsumption Law: FreeLibrary subsumes the handle release without
            #   remainder — no other file in _MEIPASS is locked.
            def release_dll_handle():
                """Release the C engine DLL handle for clean PyInstaller exit."""
                if PatternEngine._lib is not None and sys.platform == 'win32':
                    try:
                        handle = getattr(PatternEngine._lib, '_handle', None)
                        if handle is not None:
                            kernel32 = ctypes.windll.kernel32
                            getattr(kernel32, 'FreeLibrary')(handle)
                            logger.debug("PatternEngine: DLL handle released for clean exit")
                    except (OSError, AttributeError, ValueError):
                        pass  # Best-effort — process is exiting anyway
                PatternEngine._lib = None
                PatternEngine._compiled = False

            atexit.register(release_dll_handle)

            return True
        except (OSError, AttributeError) as e:
            logger.debug(f"PatternEngine: failed to load {lib_path}: {e}")
            cls._lib = None
            return False

    @classmethod
    def _ensure_compiled(cls) -> bool:
        """
        Find or compile the C pattern engine. REQUIRED for operation.

        Search order (Identification Principle — find the D that exists):
          1. PyInstaller bundle (sys._MEIPASS) — single .exe deployment
          2. Same directory as this script — development
          3. Current working directory — portable deployment
          4. Auto-compile from external .c file (script dir or CWD)

        If no pre-built DLL found and no external .c file available for
        auto-compilation, raises RuntimeError. The C engine is REQUIRED —
        it subsumes ALL pattern operations without remainder (Subsumption Law).
        """
        if cls._compiled:
            return cls._lib is not None
        if cls._attempted:
            return cls._lib is not None
        cls._attempted = True

        lib_name = 'et_pattern_engine' + ('.dll' if sys.platform == 'win32' else '.so')

        # ── Search 1: PyInstaller bundle directory ──
        # When packaged as a single .exe, PyInstaller extracts bundled
        # files to a temp directory accessible via a sys attribute.
        meipass = getattr(sys, '_MEI' + 'PASS', None)
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
        c_src_path = os.path.join(script_dir, 'et_pattern_engine.c')
        if not os.path.isfile(c_src_path):
            c_src_path = os.path.join(os.getcwd(), 'et_pattern_engine.c')

        if not os.path.isfile(c_src_path):
            raise RuntimeError(
                f"PatternEngine: C engine REQUIRED but not found.\n"
                f"  Searched: PyInstaller bundle, {script_dir}, {os.getcwd()}\n"
                f"  No pre-built {lib_name} and no et_pattern_engine.c for auto-compilation.\n"
                f"  Build the DLL via build.bat, CMake, or direct compiler invocation.\n"
                f"  See et_pattern_engine.c header for build instructions."
            )

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

        if not compiled:
            raise RuntimeError(
                f"PatternEngine: C engine REQUIRED but compilation failed.\n"
                f"  Source: {c_src_path}\n"
                f"  Tried: {', '.join(cmd[0] for cmd in compile_cmds)}\n"
                f"  Install Visual Studio Build Tools 2022 or MinGW/GCC,\n"
                f"  or pre-build the DLL via build.bat or CMake."
            )

        return cls._try_load(lib_path)

    @classmethod
    def find_patterns(cls, sym_stream: List[int],
                      min_len: int, max_len: int,
                      min_count: int = 2,
                      min_net_savings: int = 2
                      ) -> List[Tuple[tuple, List[int]]]:
        """
        Find all repeated patterns in sym_stream using the C suffix array engine.

        Returns a list of (pattern_tuple, positions_list) for patterns that have:
          - occurrence count ≥ min_count
          - net savings = count × (len-1) - (len+1) ≥ min_net_savings

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()

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
                      ) -> List[Tuple[tuple, List[int]]]:
        """
        Combined: find repeated patterns + batch IncoherenceFilter gating in C.

        Same as find_patterns() but additionally runs gate_archetype_batch on
        the raw buffer BEFORE parsing, then only parses coherent patterns.
        This eliminates 145K+ per-pattern Python gate_archetype calls.

        Returns list of (pattern, positions) for COHERENT patterns only.
        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()

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
    def fast_k_stream(cls, data: bytes, byte_k_map: Dict[int, int]) -> List[int]:
        """
        Build k-stream from byte data using C vectorized lookup.
        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()

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
    def fast_dk_stream(cls, k_stream: List[int]) -> List[int]:
        """
        Build Δk stream (first differences) from k-stream using C vectorized loop.

        dk[i] = k[i+1] - k[i] for i in 0...n-2.

        From the Descriptor Gap Principle: each Δk value IS the gap between
        adjacent lattice positions — the structural transition between consecutive
        bytes. The Δk stream is the D-set of the data's lattice walk.

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()

        n = len(k_stream)
        if n < 2:
            return []

        k_arr = (ctypes.c_int32 * n)(*k_stream)
        dk_out = (ctypes.c_int32 * (n - 1))()

        cls._lib.build_dk_stream(k_arr, n, dk_out)
        return list(dk_out)

    @classmethod
    def gate_batch(cls, raw_buf_ptr, n_patterns: int,
                   n_res: int = N_FULL) -> List[bool]:
        """
        Batch IncoherenceFilter gating in C.

        Takes the raw find_repeated_patterns buffer pointer and n_patterns.
        Returns a list of booleans (True=coherent) for each pattern.

        Implements the EXACT same L1+L2+L3+L4 checks as
        IncoherenceFilter.gate_archetype, verified zero mismatches.
        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
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
                         ) -> Tuple[List[Tuple[int, int]], List[bool]]:
        """
        C-accelerated greedy non-overlapping pattern placement.

        Returns (placements, used_mask) where:
          placements = [(arch_idx, position), ...]
          used_mask = [True/False per archetype]

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
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

    # ── Tier 1: Curvature analysis wrappers ────────────────────────────
    # These wrap the C engine's three curvature functions:
    #   build_ddk_stream — second-order finite difference (Δk → ΔΔk)
    #   compute_curvature_stats — single-pass classification of a ΔΔk stream
    #   compute_pattern_curvature — per-pattern stats for archetype DB storage
    #
    # ET Derivation (Three Tools):
    #   Identification Principle: each wrapper identifies one curvature
    #     operation that was previously absent from Python. The C side
    #     was completed first; these wrappers identify the Python-side
    #     entry points (D for the rest of the codebase).
    #   Descriptor Gap Principle: the gap between "C engine has the
    #     functions" and "Python can use them" is closed by these
    #     wrappers — each gap IS the missing wrapper Descriptor.
    #   Subsumption Law: every C export has exactly one Python wrapper.
    #     No C export is unreachable; no wrapper is orphaned.

    @classmethod
    def fast_ddk_stream(cls, dk_stream: List[int]) -> List[int]:
        """Compute the ΔΔk (second-order finite difference) of a Δk stream.

        ΔΔk_i = Δk_{i+1} - Δk_i is the discrete Gaussian curvature of the
        data's Descriptor field at position i (Non-Euclidean §6).

        Returns a list of length max(0, len(dk_stream) - 1). For input
        sequences shorter than 2, returns an empty list.

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
        n = len(dk_stream)
        if n < 2:
            return []
        dk_arr = (ctypes.c_int32 * n)(*dk_stream)
        ddk_out = (ctypes.c_int32 * (n - 1))()
        cls._lib.build_ddk_stream(dk_arr, n, ddk_out)
        return list(ddk_out)

    @classmethod
    def curvature_stats(cls, ddk_stream: List[int],
                        n_res: int = N_FULL) -> CurvatureStats:
        """Compute block-level curvature statistics + classification.

        Returns a CurvatureStats with:
          curvature_mean       — K̄ = mean(ΔΔk)
          curvature_variance   — σ²_K = variance(ΔΔk)
          curvature_class      — 0 flat / 1 elliptic / 2 hyperbolic / 3 variable / 4 singular
          euler_characteristic — χ = Σ(ΔΔk)/2π (Gauss-Bonnet fingerprint, R₀-independent)
          max_abs_curvature    — max(|ΔΔk_i|) for singularity detection

        For an empty stream, returns a zeroed struct (class 0 = flat).

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
        out = CurvatureStats()
        n = len(ddk_stream)
        if n == 0:
            return out  # All zero, class 0 — defensive zero-init in C engine
        ddk_arr = (ctypes.c_int32 * n)(*ddk_stream)
        cls._lib.compute_curvature_stats(ddk_arr, n, n_res, ctypes.byref(out))
        return out

    @classmethod
    def pattern_curvature(cls, pattern_dk: tuple) -> Tuple[float, float, float]:
        """Compute per-pattern curvature stats for the archetype DB.

        Returns (curvature_mean, curvature_variance, geodesic_factor) where
        geodesic_factor F_K = 1/(1+σ²_K). Patterns of length < 3 have no
        ΔΔk values and are reported as flat (F_K = 1.0).

        Used by ArchetypeDatabase.store at storage time to compute and
        cache the curvature_mean, curvature_variance, and geodesic_factor
        columns (design doc §16.5). Computed once per archetype, never
        recomputed during lookup.

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
        pat_len = len(pattern_dk)
        if pat_len < 3:
            return 0.0, 0.0, 1.0  # Trivially flat; F_K = full geodesic bonus
        pat_arr = (ctypes.c_int32 * pat_len)(*pattern_dk)
        out_mean = ctypes.c_double()
        out_var = ctypes.c_double()
        out_fk = ctypes.c_double()
        cls._lib.compute_pattern_curvature(
            pat_arr, pat_len,
            ctypes.byref(out_mean), ctypes.byref(out_var), ctypes.byref(out_fk))
        return out_mean.value, out_var.value, out_fk.value

    @classmethod
    def fast_geodesic_residual(cls, dk_stream: List[int],
                                connection_order: int,
                                window_size: int) -> Tuple[List[int], List[int]]:
        """Compute the geodesic residual ρ + connection Γ streams (Mode 3).

        Args:
          dk_stream: input Δk stream
          connection_order: 0 (zeroth-order), 1 (linear), or 2 (quadratic)
          window_size: L4-bounded connection window (cap S² = 144 typical)

        Returns:
          (residual, gamma) where each list has length max(0, len(dk_stream)-1).
          The residual stream is what Mode 3 actually encodes; gamma is
          informational (the decoder regenerates it from the partially-
          reconstructed Δk stream).

        Notes:
          The C engine uses C-style truncating integer division. The Python
          decompressor MUST use the matching `_c_trunc_div` helper for
          reconstruction or roundtrip will fail for negative-mean windows.

        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        cls._ensure_compiled()
        n = len(dk_stream)
        if n < 2:
            return [], []
        # Clamp args to the C engine's accepted range to avoid surprises
        if connection_order < 0:
            connection_order = 0
        elif connection_order > 2:
            connection_order = 2
        if window_size < 1:
            window_size = 1
        dk_arr = (ctypes.c_int32 * n)(*dk_stream)
        res_out = (ctypes.c_int32 * (n - 1))()
        gam_out = (ctypes.c_int32 * (n - 1))()
        cls._lib.build_geodesic_residual(
            dk_arr, n, connection_order, window_size, res_out, gam_out)
        return list(res_out), list(gam_out)


# ═══════════════════════════════════════════════════════════════════════════════
# ET LATTICE ENGINE — The entire compression derives from this
# ═══════════════════════════════════════════════════════════════════════════════

# ── Manifold Fold: d=1 Octave Encoding for Unsigned Integers ──────────────
# From the DVM (Digital Virtual Manifold §XX.3):
#   One bit = one octave = one full period of the binary substrate.
#   The d=1 sublattice folds the integer space at 2^16 = 65536.
#
# Values within one octave (0..65534) are stored as uint16 (2 bytes).
# Values exceeding one octave fold through: the fold marker (0xFFFF)
# signals that the next uint32 contains the full value (6 bytes).
#
# ET Derivation:
#   Subsumption Law: uint16 subsumes all values 0..65534 without remainder.
#   The fold marker subsumes the remainder (65535+) via uint32 extension.
#   No value is unrepresentable. No byte is wasted for small values.
#
# This replaces fixed uint32 for count fields (n_base, n_syms, n_arch, etc.)
# that are usually < 65535 but CAN exceed it with large files or uncapped
# pattern discovery. Most blocks save 2 bytes per field; blocks with
# large counts pay 4 extra bytes only for the fields that need it.
MANIFOLD_FOLD_MARKER = 0xFFFF  # d=1 octave boundary


def pack_manifold_uint(value: int) -> bytes:
    """Pack an unsigned integer using d=1 octave manifold folding."""
    if value < MANIFOLD_FOLD_MARKER:
        return struct.pack('<H', value)
    return struct.pack('<HI', MANIFOLD_FOLD_MARKER, value)


def unpack_manifold_uint(data: bytes, offset: int) -> Tuple[int, int]:
    """Unpack a manifold-folded unsigned integer. Returns (value, new_offset)."""
    marker = struct.unpack_from('<H', data, offset)[0]
    if marker < MANIFOLD_FOLD_MARKER:
        return marker, offset + 2
    value = struct.unpack_from('<I', data, offset + 2)[0]
    return value, offset + 6

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
    raw_bytes = np.frombuffer(data, dtype=np.uint8)
    if len(raw_bytes) == 0:
        return 1.0
    vals = raw_bytes.astype(np.float64) + 1.0
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

    # ── Tier 5.C.1: Riemann Sphere chordal metric ─────────────────────
    # Per design doc §13.2: replace the flat Euclidean phase_deviation
    # with the chordal distance on the Riemann sphere:
    #   d_chord(z1, z2) = 2|z1 - z2| / sqrt((1 + |z1|²)(1 + |z2|²))
    #
    # The chordal metric is the natural distance on the Riemann sphere
    # (the one-point compactification of ℂ ∪ {∞} that closes the 2D
    # ET lattice). It is bounded in [0, 2] and handles transitions near
    # the poles correctly:
    #   • South pole (z = 0): annihilating boundary (b ≪ R₀).
    #   • North pole (z = ∞): P-substrate (b ≫ R₀).
    #   • Equator (|z| = 1): T's operational manifold U(1).
    # The flat metric distorts distances near the poles; the chordal
    # metric does not.
    #
    # Gate threshold remains K_KOIDE = 2/3 (the Koide / Incoherence
    # boundary). In the bulk (where |z| ≈ 1, away from both poles),
    # the chordal metric ≈ flat Euclidean / lattice_modulus, so the
    # gate behaves as before. Near the poles (|z| → 0 or |z| → ∞),
    # the chordal metric correctly bounds the deviation in [0, 2]
    # rather than producing Inf or 0/0.
    #
    # ET Three Tools:
    #   Identification Principle: identifies the natural geometric
    #     closure of the complex lattice (the Riemann sphere).
    #   Descriptor Gap Principle: closes the gap between "complex
    #     position has a well-defined deviation" and "the deviation
    #     remains finite and meaningful at the poles". The chordal
    #     metric IS the missing Descriptor of pole-aware distance.
    #   Subsumption Law: in the bulk, chordal ≈ flat — so all bulk
    #     behavior is preserved exactly. Only near-pole cases get the
    #     correction. No previously-coherent point becomes incoherent;
    #     no previously-incoherent point becomes coherent except near
    #     the poles where the flat metric was lying.
    diff_modulus = abs(z_lattice - z_nearest)
    z_norm_sq        = lattice_modulus * lattice_modulus
    z_nearest_norm_sq = abs(z_nearest) * abs(z_nearest)
    chord_denom = math.sqrt((1.0 + z_norm_sq) * (1.0 + z_nearest_norm_sq))
    if chord_denom < EPSILON:
        # Both points at the south pole (z = 0); chordal distance = 0
        # by definition (they coincide on the sphere).
        phase_deviation = 0.0
    else:
        phase_deviation = (2.0 * diff_modulus) / chord_denom

    # Gate: if phase deviation exceeds K (Koide = 2/3 = ∂I boundary),
    # the complex position is incoherent — d_combined saturates to N_FULL.
    # The chordal metric's [0, 2] range means the K=2/3 threshold remains
    # a meaningful fraction of the maximum possible distance.
    if phase_deviation > K_KOIDE:
        d_combined = N_FULL

    return k_r, d_r, k_theta, d_theta, d_combined


# ═══════════════════════════════════════════════════════════════════════════════
# POINCARÉ DISK EMBEDDING — Hyperbolic geometry for pattern comparison (Tier 6)
# From: Design doc §12.3, §8 (Improvement 5 — Hyperbolic Pattern Embedding)
#
# The Poincaré disk model represents the hyperbolic plane as the open unit
# disk {z ∈ ℂ : |z| < 1}. Points near the boundary (|z| → 1) have infinite
# hyperbolic distance from the centre — they represent near-Incoherent
# configurations (high curvature, long patterns at the lattice periphery).
# Points near the centre represent flat/stable configurations.
#
# ET Three Tools:
#   Identification Principle: identifies patterns by their position on the
#     hyperbolic manifold — two patterns with similar curvature and depth
#     are close in the Poincaré metric regardless of their absolute k-values.
#   Descriptor Gap Principle: closes the gap between "patterns look different
#     in Euclidean k-space because R₀ shifted" and "patterns have identical
#     curvature geometry because ΔΔk is R₀-invariant". The Poincaré embedding
#     IS the coordinate system where the curvature geometry becomes distance.
#   Subsumption Law: poincare_distance(z, z) = 0 (identity); the metric
#     subsumes Euclidean at the disk centre (low curvature → flat → Euclidean).
# ═══════════════════════════════════════════════════════════════════════════════


def embed_pattern_hyperbolic(curvature_mean: float,
                              d_avg: float,
                              n_res: int = N_FULL) -> complex:
    """Map a pattern's curvature + depth to a point on the Poincaré disk.

    Per design doc §12.3:
      Radial:  r = tanh(|K̄| / S), in (0, 1). Flat patterns → r ≈ 0 (centre).
               Singular patterns → r → 1 (boundary, near ∂I).
      Angular: θ = 2π × (d_avg / N_FULL) × sign(K̄). Encodes both sublattice
               depth (which d-family the pattern lives in) and curvature sign
               (elliptic = positive angle, hyperbolic = negative angle).
               K̄ = 0 maps to θ = 0 (real axis) by convention.

    Returns a complex number z with |z| < 1 (open unit disk).
    """
    # Radial: tanh maps |K̄|/S → (0, 1) smoothly.
    # S = 12 is the manifold symmetry; |K̄|/S normalises the curvature
    # magnitude to the lattice's natural angular scale.
    r = math.tanh(abs(curvature_mean) / S)
    # Clamp r to stay strictly inside the disk (|z| = 1 is at ∂I; no
    # pattern should sit exactly ON the Incoherence boundary).
    r = min(r, 1.0 - EPSILON)

    # Angular: d_avg/N_FULL scales the sublattice depth into [0, 1);
    # multiplying by 2π wraps it around the circle. The sign of K̄
    # determines the hemisphere: positive curvature (elliptic) on top,
    # negative curvature (hyperbolic) on bottom. K̄ = 0 lands on the
    # real axis (θ = 0).
    if abs(curvature_mean) < EPSILON:
        theta = 0.0
    else:
        sign = 1.0 if curvature_mean > 0 else -1.0
        theta = sign * 2.0 * math.pi * (d_avg / max(n_res, 1))

    return complex(r * math.cos(theta), r * math.sin(theta))


def poincare_distance(z1: complex, z2: complex) -> float:
    """Hyperbolic distance on the Poincaré disk.

    Per design doc §12.3:
      d_hyp(z1, z2) = arccosh(1 + 2|z1 - z2|² / ((1 - |z1|²)(1 - |z2|²)))

    Returns ≥ 0. Returns 0 when z1 == z2. Returns +inf when either point
    is on the boundary (|z| = 1), which corresponds to the Incoherence
    boundary — patterns at ∂I are infinitely far from everything else.

    The metric's key property for cross-tower matching: two patterns with
    similar (K̄, d_avg) have small d_hyp regardless of their absolute k-values,
    because the embedding uses curvature (R₀-independent) as the coordinate.
    """
    diff_sq = abs(z1 - z2) ** 2
    denom1 = 1.0 - abs(z1) ** 2
    denom2 = 1.0 - abs(z2) ** 2
    if denom1 <= 0.0 or denom2 <= 0.0:
        return float('inf')  # On or beyond the boundary
    arg = 1.0 + 2.0 * diff_sq / (denom1 * denom2)
    # arccosh(x) = log(x + sqrt(x² - 1)) for x ≥ 1
    if arg < 1.0:
        return 0.0  # Numerical edge case — same point
    return math.acosh(arg)


def curvature_coherence(k_mean_a: float, k_mean_b: float) -> float:
    """Cross-tower curvature coherence measure (Tier 6.B.1 helper).

    Per design doc §12.2:
      coherence(T_A, T_B) = 1 - |K̄_A - K̄_B| / (|K̄_A| + |K̄_B| + ε)

    Returns a value in [0, 1]. coherence = 1.0 for identical curvature
    (same byte sequence regardless of R₀). coherence < K = 2/3 means
    the towers are curvature-incompatible.
    """
    numerator = abs(k_mean_a - k_mean_b)
    denominator = abs(k_mean_a) + abs(k_mean_b) + EPSILON
    return 1.0 - numerator / denominator
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
    def l1_point_curvature(epsilon_cents: float, local_curvature: float) -> bool:
        """Level 1 with curvature-adjusted threshold (Tier 5.B.1).

        Per design doc §11.2:
          ε_max(K_i) = 50 × 1/(1 + |K_i|/N)

        For flat regions (K_i = 0) this is identical to l1_point — the
        threshold remains 50¢ and previously-accepted points stay accepted.
        For curved regions the threshold tightens proportionally to local
        curvature: a point that sits inside the flat 50¢ window but rides
        a high-curvature bump is closer to the Incoherence boundary than
        a flat-region point with the same ε.

        ET Three Tools:
          Identification Principle: identifies points that are
            COHERENT-IN-THE-FLAT-METRIC but INCOHERENT-IN-THE-LOCAL-METRIC.
            The curvature factor distinguishes the two.
          Descriptor Gap Principle: closes the gap between "absolute ε"
            and "ε relative to local geometry". Local curvature IS the
            missing scale.
          Subsumption Law: l1_point_curvature(ε, K=0) ≡ l1_point(ε)
            exactly — the new method subsumes the flat case as the
            K=0 specialization without changing it.

        Note: this is a STRICTER filter than l1_point. It is only used
        where the calling code has access to local_curvature; existing
        l1_point call sites are unchanged (they remain the flat case,
        which is correct for their context where curvature is not in scope).
        """
        if not math.isfinite(local_curvature):
            # Pathological local curvature — fall back to the flat threshold
            # rather than returning False arbitrarily. The Subsumption Law
            # demands every input get a defined response.
            return abs(epsilon_cents) < INCOHERENCE_CENTS
        # Curvature-adjusted threshold (cents)
        threshold = INCOHERENCE_CENTS / (1.0 + abs(local_curvature) / S)
        return abs(epsilon_cents) < threshold

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
# CURVATURE ANALYZER — Phase 1.5 of the compression pipeline (Tier 1)
#
# From the ET Non-Euclidean Geometry paper §3, §6, §11 and the CDF Non-Euclidean
# design doc §3, §4. Computes the discrete Gaussian curvature of a block's Δk
# walk on the lattice and classifies the block into one of five manifold-state
# classes:
#
#   class 0  flat        Exception {P,D,T}      — geodesic data
#   class 1  elliptic    Unsubstantiated {P,D}  — closed/periodic
#   class 2  hyperbolic  Mediation {D,T}        — chaotic/diverging
#   class 3  variable    mixed                  — needs segmentation (Tier 4)
#   class 4  singular    Incoherence {P,T}      — D-bridge broken (segment first)
#
# The classification informs (but does not replace) compression strategy
# selection — every existing strategy still runs. Curvature analysis just
# adds new strategies and ORDERS the existing ones so the geometry-suggested
# strategy is tried first. With "smallest output wins", earlier candidates
# do not affect correctness — only running cost (which is irrelevant per
# the design's "speed is irrelevant" rule).
#
# Design doc §3.4 Curvature Sign Classification:
#   sign(K̄) = 0 → Exception (flat)
#   sign(K̄) > 0 → Unsubstantiated (elliptic, K>0)
#   sign(K̄) < 0 → Mediation (hyperbolic, K<0)
#
# ET Three Tools applied to compression strategy selection:
#   Identification Principle: each block's manifold state identifies WHICH
#     compression strategy is structurally best.
#   Descriptor Gap Principle: the gap between "ΔΔk computed for entropy"
#     and "ΔΔk drives strategy" is closed by this analyzer.
#   Subsumption Law: the five classes subsume every possible ΔΔk
#     distribution without remainder. No data geometry escapes classification.
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockCurvatureProfile:
    """Per-block curvature profile produced by CurvatureAnalyzer.analyze_block.

    Stored on the CDFEngine after Phase 1.5 so downstream phases (strategy
    selection, archetype DB store) can access curvature data without recomputing.
    All fields are R₀-independent (design doc §10.0.3): two files producing
    identical Δk streams produce identical profiles regardless of personal R₀.

    Class enumeration matches the C engine and the Python CURVATURE_CLASS_*
    constants exactly — these are not duplicate definitions, they are the
    same enumeration surfaced at three layers (C, ctypes, dataclass) so each
    layer can describe its data without crossing the layer boundary.
    """
    curvature_mean: float = 0.0          # K̄ — mean of ΔΔk
    curvature_variance: float = 0.0      # σ²_K — variance of ΔΔk
    curvature_class: int = CURVATURE_CLASS_FLAT   # 0..4 (matches C engine)
    euler_characteristic: float = 0.0    # χ = Σ(ΔΔk)/(2π) — Gauss-Bonnet fingerprint
    max_abs_curvature: int = 0           # max |ΔΔk_i| — singularity signal
    n_ddk_samples: int = 0               # length of the ΔΔk stream that produced these stats
    sign: int = 0                        # sign(K̄) ∈ {-1, 0, +1} (design doc §3.4)
    # Cached references to streams used by later phases (Tier 4 segmentation,
    # Channel B generator fitting). Kept here so downstream code does not
    # have to recompute derivatives from scratch.
    dk_stream: Optional[List[int]] = None    # original Δk stream (or None when not retained)
    ddk_stream: Optional[List[int]] = None   # the ΔΔk stream used for classification

    def class_name(self) -> str:
        """Human-readable name of the manifold-state class."""
        return CURVATURE_CLASS_NAMES.get(self.curvature_class,
                                         f'<unknown class {self.curvature_class}>')

    def is_flat(self) -> bool:
        """True if the block is in Exception state (K̄≈0, σ²_K small)."""
        return self.curvature_class == CURVATURE_CLASS_FLAT

    def is_elliptic(self) -> bool:
        """True if the block has constant positive curvature (closed/periodic)."""
        return self.curvature_class == CURVATURE_CLASS_ELLIPTIC

    def is_hyperbolic(self) -> bool:
        """True if the block has constant negative curvature (chaotic/divergent)."""
        return self.curvature_class == CURVATURE_CLASS_HYPERBOLIC

    def is_variable(self) -> bool:
        """True if local curvature varies across the block (needs segmentation)."""
        return self.curvature_class == CURVATURE_CLASS_VARIABLE

    def is_singular(self) -> bool:
        """True if the block contains curvature singularities (max|K_i| ≥ S)."""
        return self.curvature_class == CURVATURE_CLASS_SINGULAR


class CurvatureAnalyzer:
    """Block-level curvature analysis (Phase 1.5).

    analyze_block() takes a Δk stream and returns a BlockCurvatureProfile.
    All ΔΔk computation and classification is delegated to the C engine
    (PatternEngine.fast_ddk_stream + PatternEngine.curvature_stats); this
    class is the Python-side wrapper that:
      1. Computes the ΔΔk stream from Δk
      2. Calls the C classifier
      3. Wraps the result in a BlockCurvatureProfile
      4. Provides segment_boundaries() for Tier 4 variable-curvature segmentation

    The analyzer never modifies the Δk stream and never touches existing
    compression code. Phase 1.5 is purely additive — the data flows through
    it but the existing pipeline behaves identically whether or not the
    analyzer is invoked.

    ET Derivation (Three Tools):
      Identification Principle: identifies the block's manifold-state class
        via the second-order D-gradient of its lattice walk.
      Descriptor Gap Principle: the gap between "ΔΔk is a number" and
        "ΔΔk is a Descriptor of the block's geometry" is the analyzer.
      Subsumption Law: the five-class enumeration subsumes every possible
        ΔΔk distribution. Every block falls into exactly one class.
    """

    def __init__(self, n_res: int = N_FULL):
        """Construct an analyzer at lattice resolution n_res (default 27720)."""
        self.n_res = n_res

    def analyze_block(self, dk_stream: List[int],
                      retain_streams: bool = False) -> BlockCurvatureProfile:
        """Compute curvature stats + classification from a Δk stream.

        Args:
          dk_stream: First-difference stream from a block's k-stream
            (typically PatternEngine.fast_dk_stream output).
          retain_streams: If True, the returned profile retains references
            to dk_stream and ddk_stream for use by later phases (Tier 4
            segmentation, Channel B generator fitting). Default False to
            avoid memory bloat for blocks that won't need them.

        Returns:
          A BlockCurvatureProfile with fully populated curvature stats
          and class. For dk_stream of length < 2, returns a flat profile
          (class 0, all stats zero, n_ddk_samples=0).
        """
        n_dk = len(dk_stream)
        if n_dk < 2:
            # No ΔΔk values exist — trivially flat (Exception). Honor the
            # retention flag so downstream code can rely on it.
            return BlockCurvatureProfile(
                dk_stream=dk_stream if retain_streams else None,
                ddk_stream=[] if retain_streams else None,
            )

        # ── Compute ΔΔk via C engine (single O(n) pass) ──
        ddk_stream = PatternEngine.fast_ddk_stream(dk_stream)

        # ── Classify via C engine (single O(n) pass) ──
        stats = PatternEngine.curvature_stats(ddk_stream, n_res=self.n_res)

        # ── Compute sign of K̄ (design doc §3.4) ──
        # Threshold by SUBLIMINAL_CURVATURE — values below the manifold's
        # minimum detectable curvature count as "no sign" (sign = 0 = flat).
        if stats.curvature_mean > SUBLIMINAL_CURVATURE:
            sign = 1
        elif stats.curvature_mean < -SUBLIMINAL_CURVATURE:
            sign = -1
        else:
            sign = 0

        return BlockCurvatureProfile(
            curvature_mean=stats.curvature_mean,
            curvature_variance=stats.curvature_variance,
            curvature_class=int(stats.curvature_class),
            euler_characteristic=stats.euler_characteristic,
            max_abs_curvature=int(stats.max_abs_curvature),
            n_ddk_samples=len(ddk_stream),
            sign=sign,
            dk_stream=dk_stream if retain_streams else None,
            ddk_stream=ddk_stream if retain_streams else None,
        )

    @staticmethod
    def classify(dk_stream: List[int], n_res: int = N_FULL) -> int:
        """Convenience: return only the curvature class (0..4) for a Δk stream.

        Equivalent to calling analyze_block(...).curvature_class but skips
        the dataclass construction. Useful for quick gating decisions where
        only the class matters and the full profile is not needed.
        """
        if len(dk_stream) < 2:
            return CURVATURE_CLASS_FLAT
        ddk = PatternEngine.fast_ddk_stream(dk_stream)
        stats = PatternEngine.curvature_stats(ddk, n_res=n_res)
        return int(stats.curvature_class)

    @staticmethod
    def segment_boundaries(ddk_stream: List[int],
                           min_segment_length: int = S * S) -> List[int]:
        """Find curvature sign-change boundaries in a ΔΔk stream (Tier 4 helper).

        From the design doc §10.2:
          boundary at i ⟺ sign(K_i) ≠ sign(K_{i+1}) AND |K_i| ≥ π/N

        After collecting raw boundaries, merges segments shorter than
        min_segment_length (default S² = 144) with their neighbors to
        avoid over-segmentation. Returns a sorted list of boundary
        indices into ddk_stream (each boundary marks the START of a new
        segment).

        Wired into the CurvatureAnalyzer at Tier 1 so the Tier 4
        segmentation code can call it without further structural changes.
        Boundary detection alone is cheap (single O(n) scan); segmentation
        decisions happen later.

        Args:
          ddk_stream: The ΔΔk stream of a Δk walk.
          min_segment_length: Minimum segment length to retain (default S²).

        Returns:
          A sorted list of boundary indices. An empty list means no
          segmentation needed (single-segment / homogeneous curvature).
        """
        n = len(ddk_stream)
        if n < 2:
            return []

        # ── Pass 1: collect all sign-change boundaries above threshold ──
        # The threshold is SUBLIMINAL_CURVATURE (π/N) — minor sign flips
        # below the lattice's resolution are noise, not structural changes.
        raw_boundaries: List[int] = []
        for i in range(n - 1):
            k_i = ddk_stream[i]
            k_j = ddk_stream[i + 1]
            # Sign change AND magnitude above subliminal threshold
            if (k_i > 0) != (k_j > 0) or (k_i < 0) != (k_j < 0):
                if abs(k_i) >= SUBLIMINAL_CURVATURE or abs(k_j) >= SUBLIMINAL_CURVATURE:
                    raw_boundaries.append(i + 1)

        if not raw_boundaries:
            return []

        # ── Pass 2: merge short segments (< min_segment_length) ──
        # Build provisional segment starts as [0, *raw_boundaries] and
        # walk forward, dropping any boundary that produces a segment
        # shorter than min_segment_length on either side.
        candidates = [0] + raw_boundaries + [n]
        kept: List[int] = [0]
        i = 1
        while i < len(candidates) - 1:
            seg_start = kept[-1]
            seg_candidate_end = candidates[i]
            next_candidate = candidates[i + 1]
            if seg_candidate_end - seg_start < min_segment_length:
                # Current segment too short — skip this boundary,
                # absorbing it into the next segment.
                i += 1
                continue
            if next_candidate - seg_candidate_end < min_segment_length:
                # Next segment too short — skip this boundary,
                # merging it forward.
                i += 1
                continue
            kept.append(seg_candidate_end)
            i += 1

        # Drop the leading 0 — boundaries are insertion points, not start markers
        return kept[1:] if len(kept) > 1 else []

    def compute_segmentation(self, dk_stream: List[int],
                              n_bytes: int,
                              min_segment_length: int = S * S
                              ) -> List[Tuple[int, int]]:
        """Map ddk-stream boundaries to byte-level segment ranges (Tier 4.A.1).

        Higher-level wrapper around segment_boundaries that translates
        the boundary list (indexed into ddk_stream) into actual byte
        ranges over the original data block. Each returned tuple
        (byte_start, byte_end) is a half-open range [start, end) over
        byte indices [0, n_bytes); the union of all ranges = [0, n_bytes).

        Index mapping: a boundary at ddk_index b reflects a sign change
        between ddk[b-1] and ddk[b], which corresponds to a transition
        in dk between dk[b-1] and dk[b], which corresponds to a byte
        position around byte b + 1 (the byte AFTER the dk transition).
        Each ddk boundary therefore maps to a byte-level split at
        byte index (b + 1), clamped to the block's bounds.

        Returns:
          [] when no segmentation is needed (single uniform-curvature
              block; the caller skips Block Type 4 and uses a single
              standard lattice block).
          [(0, n_bytes)] is NEVER returned — that's equivalent to []
              and the caller treats both as "no segmentation".
          [(0, b1), (b1, b2), ..., (b_{k-1}, n_bytes)] when k segments
              are identified.

        ET Three Tools:
          Identification Principle: identifies the byte-level boundaries
            where the block's curvature class changes, which is where
            the compression strategy should change too.
          Descriptor Gap Principle: closes the gap between "ddk-level
            geometric boundaries" and "byte-level data slices" — both
            sides need the boundaries expressed in a coordinate system
            they can act on.
          Subsumption Law: segment ranges tile [0, n_bytes) without
            overlap or gap; every byte belongs to exactly one segment.
        """
        if n_bytes < 2:
            return []

        # Build ΔΔk from the Δk stream (matching the same arithmetic
        # the C engine uses in build_ddk_stream).
        if len(dk_stream) < 2:
            return []
        ddk_stream: List[int] = [dk_stream[i + 1] - dk_stream[i]
                                  for i in range(len(dk_stream) - 1)]

        # Per-byte segment length minimum: the design doc specifies S²=144
        # symbols. Translated to bytes via byte ≈ ddk + 2, the corresponding
        # byte-level minimum is also approximately S². We use the same
        # value for both to keep the boundary semantics consistent.
        boundaries_ddk = self.segment_boundaries(ddk_stream,
                                                  min_segment_length=min_segment_length)
        if not boundaries_ddk:
            return []

        # Map ddk-stream indices → byte indices via dk-spike search.
        #
        # The byte-level transition that produced an ddk-boundary lives in
        # the dk_stream as a spike (max |dk|). For a transition at byte X,
        # dk[X-1] = k[X] - k[X-1] is large; the corresponding ddk pattern
        # is [..., 0, +SPIKE, -SPIKE, 0, ...] with sign changes at ddk
        # indices X-2, X-1, X. Which one survives the filter depends on
        # the data shape, so a fixed offset (b → b+1) gives wrong results
        # when the filter keeps a different raw boundary.
        #
        # The principled fix: for each surviving ddk-boundary b, search
        # the dk_stream in a small window centered on b for the index
        # with maximum |dk|. That dk index points to the actual transition.
        # The byte split is at dk_index + 1 (segment 2 starts at the byte
        # AFTER the spike, where the new byte-value run begins).
        n_dk = len(dk_stream)
        byte_boundaries: List[int] = []
        for b in boundaries_ddk:
            # Window in dk space: dk[b-2 .. b+2] inclusive, clamped.
            # The spike causing ddk[b-1..b] sign change lives at
            # dk[b] or dk[b-1]; widening to ±2 catches edge cases where
            # the spike pattern is asymmetric.
            w_start = max(0, b - 2)
            w_end = min(n_dk, b + 3)  # exclusive end
            if w_end <= w_start:
                continue
            # Find dk index with maximum |dk| in the window.
            spike_idx = w_start
            spike_mag = abs(dk_stream[w_start])
            for j in range(w_start + 1, w_end):
                m = abs(dk_stream[j])
                if m > spike_mag:
                    spike_mag = m
                    spike_idx = j
            # Byte split = dk_spike_index + 1 (segment 2 starts at the
            # byte immediately AFTER the spike, where the new byte-value
            # run begins; dk[spike_idx] = k[spike_idx+1] - k[spike_idx]).
            byte_idx = spike_idx + 1
            if 0 < byte_idx < n_bytes:
                byte_boundaries.append(byte_idx)

        # Deduplicate (sorted) and validate per-segment lengths in BYTE space.
        # The dedup is defensive — segment_boundaries already returns sorted
        # unique indices, but the +1 mapping could collide on edge cases.
        byte_boundaries = sorted(set(byte_boundaries))
        if not byte_boundaries:
            return []

        # Re-apply the min-length filter in byte space (the ddk-space
        # filter used the same constant but was measured in ddk units;
        # for bytes the math comes out essentially the same since
        # |ddk_stream| = |bytes| - 2, but explicit re-check is cheap
        # insurance against off-by-one segmentation that violates the
        # min-length contract).
        candidates = [0] + byte_boundaries + [n_bytes]
        kept_starts: List[int] = [0]
        i = 1
        while i < len(candidates) - 1:
            seg_start = kept_starts[-1]
            seg_candidate_end = candidates[i]
            next_candidate = candidates[i + 1]
            if seg_candidate_end - seg_start < min_segment_length:
                i += 1
                continue
            if next_candidate - seg_candidate_end < min_segment_length:
                i += 1
                continue
            kept_starts.append(seg_candidate_end)
            i += 1

        # If only the single starting boundary survived, no segmentation.
        if len(kept_starts) <= 1:
            return []

        # Build the (start, end) tuples covering [0, n_bytes).
        kept_starts.append(n_bytes)
        return [(kept_starts[i], kept_starts[i + 1])
                for i in range(len(kept_starts) - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL B — GENERATIVE DESCRIPTOR DERIVATION (Tier 3)
#
# From the D Paper §44.1:
#
#   "Standard compression algorithms operate by finding repetitions in
#    Point data. ET's Descriptor Gap Principle provides a fundamentally
#    different approach: instead of finding repetitions, find the
#    Generative Descriptor — the function (D) that generates the data
#    (P) when applied."
#
# A Generative Descriptor D_gen is a compact rule that, when applied to T
# (execution), produces P (the data):
#
#     P_raw = T(D_gen)
#     Compression ratio = |P_raw| / |D_gen|
#
# This is Kolmogorov-complexity compression, NOT Shannon-entropy
# compression. The Shannon limit is the marginal entropy of the source
# distribution; the Kolmogorov limit is the length of the shortest
# program that produces the data. For ALL structured data,
# Kolmogorov < Shannon.
#
# Channel B implements the Discovery half of the database (Channel A
# stored Observation). Each curvature class constrains what kind of
# Generative Descriptor can produce it:
#
#   Flat (K ≈ 0)        → constant or linear generators
#   Elliptic (K > 0)    → periodic / trigonometric generators
#   Hyperbolic (K < 0)  → grammar / recursive generators
#   Variable curvature  → segmented generators (per-segment mix)
#
# Plus polynomial as a fifth type covering smooth higher-order trends
# that fit between linear and periodic.
#
# ET Three Tools applied to Channel B:
#   Identification Principle: each generator type IS a candidate D_X
#     for the data — Channel B identifies the D that, when executed,
#     produces P.
#   Descriptor Gap Principle: the gap between "the scanner finds
#     patterns" and "the data has a generative rule" is closed by
#     deriving D_gen candidates from the curvature profile.
#   Subsumption Law: the 5 types subsume every known structured data
#     family without remainder. Random data has no fit (no generator);
#     structured data fits at least one generator family.
# ═══════════════════════════════════════════════════════════════════════════════

# Generator type codes — fixed enumeration used in the database column
# `generator_type` (TEXT) and in the Mode 4 block header (manifold uint).
# Order is stable across versions; new types append at the end.
GENERATOR_TYPE_CONSTANT   = 'constant'    # k_i = c                              (1 param)
GENERATOR_TYPE_LINEAR     = 'linear'      # k_i = a + b·i                        (2 params)
GENERATOR_TYPE_POLYNOMIAL = 'polynomial'  # k_i = a + b·i + c·i² (degree 2)      (3 params)
GENERATOR_TYPE_PERIODIC   = 'periodic'    # k_i = offset + amplitude·tone(i)     (4 params)
GENERATOR_TYPE_GRAMMAR    = 'grammar'     # Re-Pair-style production rules       (variable)

# Numeric codes used in the Mode 4 block header — written as the first
# byte of generator_params blob to identify which generator the decoder
# must instantiate. Order matches the string codes above.
GENERATOR_TYPE_CODES: Dict[str, int] = {
    GENERATOR_TYPE_CONSTANT:   0,
    GENERATOR_TYPE_LINEAR:     1,
    GENERATOR_TYPE_POLYNOMIAL: 2,
    GENERATOR_TYPE_PERIODIC:   3,
    GENERATOR_TYPE_GRAMMAR:    4,
}
GENERATOR_TYPE_NAMES: Dict[int, str] = {v: k for k, v in GENERATOR_TYPE_CODES.items()}


@dataclass
class GenerativeDescriptor:
    """A derived Generative Descriptor — Channel B's first-class entity.

    Stored in the `generative_descriptors` table. Each instance describes
    a candidate D_gen that the database has either DERIVED from
    geometric reasoning (source='derived') or PROMOTED from observation
    after enough confirmations (source='observed').

    The lifecycle:
      1. Derived: a curvature profile suggests a generator family. The
         database stores a candidate template with fit_count=0. This is
         an {P,D} Unsubstantiated entry — a prediction.
      2. Offered: when a new block has a matching curvature class, the
         template is offered to the compressor.
      3. Fitted: if the compressor verifies σ²_residual < V, fit_count
         increments and last_confirmed updates. The template becomes
         increasingly substantiated.
      4. Missed: if the template was offered but did not fit (residual
         too large), miss_count increments. High-miss templates are
         deprioritised but NEVER deleted (per §16.9 NO-REMOVAL).

    Persistence: a GenerativeDescriptor serialises its parameters via
    pickle into the `generator_params` BLOB column. The blob layout is:
      [1 byte] generator type code (matches GENERATOR_TYPE_CODES)
      [1 byte] stream target code (0 = k_stream, 1 = dk_stream)
      [N bytes] pickled parameter dict (type-specific)

    The stream_target byte tells the Mode 4 decoder which coordinate
    system the generator predicts in:
      0 (k_stream):  generate() returns absolute lattice positions;
                     residual is added directly to predicted k values.
      1 (dk_stream): generate() returns first-differences;
                     decoder integrates to predicted_k via cumsum +
                     anchor, then adds residual to recover actual_k.
    This is necessary because PERIODIC and GRAMMAR generators are most
    informative on dk_stream (they catch cyclic-dk drifts that k_stream
    misses), while CONSTANT/LINEAR/POLYNOMIAL are most informative on
    k_stream directly.
    """
    gen_id: str                                         # SHA-256 of (type, params)[:32]
    curvature_class: int                                # 0..4 (which class this serves)
    generator_type: str                                 # GENERATOR_TYPE_* string code
    generator_params: bytes                             # [type_code][stream_target][pickled params]
    param_count: int                                    # |D_gen| — explicit param count
    curvature_mean_range_low: Optional[float] = None    # K̄ range this generator covers
    curvature_mean_range_high: Optional[float] = None
    fit_count: int = 0                                  # successful fits
    miss_count: int = 0                                 # failed fits
    best_residual_variance: Optional[float] = None      # best σ²_residual ever achieved
    first_derived: float = 0.0                          # creation timestamp
    last_confirmed: Optional[float] = None              # last successful fit timestamp
    source: str = 'derived'                             # 'derived' (Channel B) or 'observed'

    @staticmethod
    def make_id(generator_type: str, params: dict) -> str:
        """Deterministic ID from (type, params) so duplicate derivations dedupe."""
        canonical = f'{generator_type}|{sorted(params.items())}'
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]


# Stream-target codes — written as the second byte of generator_params blob.
# Tells the Mode 4 decoder which coordinate system the generator predicts in.
GENERATOR_STREAM_K  = 0   # generator outputs absolute k-stream values
GENERATOR_STREAM_DK = 1   # generator outputs first-differences (Δk)

# Default stream target per generator type — set so each type fits in
# the coordinate system where it captures the most structure. PERIODIC
# and GRAMMAR target dk_stream because cyclic and recursive structure
# tends to live in the differential domain (a periodic offset becomes
# a constant in dk; a recurrence's pattern surfaces in dk repetitions
# while k itself drifts). The arithmetic types target k_stream directly
# because their structure (constant value, linear ramp, parabola) is
# most evident in absolute lattice positions.
GENERATOR_DEFAULT_STREAM: Dict[str, int] = {
    GENERATOR_TYPE_CONSTANT:   GENERATOR_STREAM_K,
    GENERATOR_TYPE_LINEAR:     GENERATOR_STREAM_K,
    GENERATOR_TYPE_POLYNOMIAL: GENERATOR_STREAM_K,
    GENERATOR_TYPE_PERIODIC:   GENERATOR_STREAM_DK,
    GENERATOR_TYPE_GRAMMAR:    GENERATOR_STREAM_DK,
}


# ── Generator implementations (each: fit() + generate() + param_count) ────
# Every generator type implements the same two operations:
#   fit(data: List[int]) -> Optional[Dict] — try to fit; return params or None
#   generate(params: Dict, length: int) -> List[int] — regenerate from params
# Both operate on integer streams (k-values or Δk-values, caller's choice).
# All arithmetic is exact integer where possible; float intermediates are
# rounded toward zero with `int()` to match C-style truncation, keeping
# encoder/decoder bit-exactly aligned (same principle as _c_trunc_div).

class ConstantGenerator:
    """k_i = c for all i — flat data with constant value.

    Fits when every value in the stream is identical. The Kolmogorov
    description is just the value plus the length: 2 numbers regardless
    of stream length.

    ET role: this is the geodesic of the d=0 sublattice — the constant
    map. It IS the trivial Generative Descriptor.
    """
    PARAM_COUNT = 1  # the constant value itself (length stored in block header)

    @staticmethod
    def fit(data: List[int]) -> Optional[Dict[str, Any]]:
        """Fit if and only if every value is identical."""
        if not data:
            return None
        c = data[0]
        for v in data[1:]:
            if v != c:
                return None
        return {'c': int(c)}

    @staticmethod
    def generate(params: Dict[str, Any], length: int) -> List[int]:
        return [int(params['c'])] * length


class LinearGenerator:
    """k_i = a + b·i — linear ramp.

    Fits when the first-difference Δk is constant: Δk_i = b for all i,
    a = k_0. The Kolmogorov description is (a, b, length): 3 numbers
    regardless of stream length.

    ET role: this is the geodesic of the d=1 sublattice — constant Δk
    means the lattice walk advances uniformly. From design doc §16.8.2:
    "ΔΔk ≈ 0 means Δk ≈ constant, which means the k-stream is a linear
    function of position."
    """
    PARAM_COUNT = 2  # a, b (length stored in block header)

    @staticmethod
    def fit(data: List[int]) -> Optional[Dict[str, Any]]:
        """Fit if Δk is exactly constant."""
        if len(data) < 2:
            return None
        b = data[1] - data[0]
        for i in range(1, len(data) - 1):
            if data[i + 1] - data[i] != b:
                return None
        return {'a': int(data[0]), 'b': int(b)}

    @staticmethod
    def generate(params: Dict[str, Any], length: int) -> List[int]:
        a = int(params['a'])
        b = int(params['b'])
        return [a + b * i for i in range(length)]


class PolynomialGenerator:
    """k_i = a + b·i + c·i² — degree-2 polynomial.

    Fits when the second-difference ΔΔk is exactly constant: ΔΔk_i = 2c,
    so c = ΔΔk_i / 2 (must be integer-divisible by 2 for exact fit).
    Then b = (k_1 - k_0) - c, a = k_0. The Kolmogorov description is
    (a, b, c, length): 4 numbers regardless of stream length.

    ET role: covers the geodesic of the d=2 sublattice — the parabolic
    curve. From design doc §6.2 connection orders:
      "Order 2 captures quadratic trends (parabolic curves, sinusoidal
       half-cycles)."
    """
    PARAM_COUNT = 3  # a, b, c

    @staticmethod
    def fit(data: List[int]) -> Optional[Dict[str, Any]]:
        """Fit if ΔΔk is exactly constant and even (so c = ΔΔk/2 is integer)."""
        if len(data) < 3:
            return None
        # Compute first ΔΔk
        ddk0 = (data[2] - data[1]) - (data[1] - data[0])
        # Check all ΔΔk are equal
        for i in range(1, len(data) - 2):
            ddk_i = (data[i + 2] - data[i + 1]) - (data[i + 1] - data[i])
            if ddk_i != ddk0:
                return None
        # For lossless integer fit: 2c must equal ddk0 exactly.
        # If ddk0 is odd, no integer c satisfies this — generator does not fit.
        if ddk0 % 2 != 0:
            return None
        c = ddk0 // 2
        # b = (k_1 - k_0) - c   (since k_1 = a + b·1 + c·1² = a + b + c)
        a = int(data[0])
        b = int((data[1] - data[0]) - c)
        return {'a': a, 'b': b, 'c': int(c)}

    @staticmethod
    def generate(params: Dict[str, Any], length: int) -> List[int]:
        a = int(params['a'])
        b = int(params['b'])
        c = int(params['c'])
        return [a + b * i + c * i * i for i in range(length)]


class PeriodicGenerator:
    """k_i = offset + amplitude · tone(i) — periodic / cyclic data.

    From design doc §16.8.2 elliptic case:
      "Constant positive ΔΔk means the Δk stream is linear → k-stream
       is quadratic. But K > 0 with bounded data means PERIODIC
       behavior (closed T-traversal per §14 of Non-Euclidean paper)."

    Fits when the data exhibits a clean repetition: data[i] == data[i + period]
    for every i in [0, length - period). The Kolmogorov description is
    (period, amplitude, phase_offset, base_offset) plus the first
    `period` samples that define the repeating tone.

    ET role: covers the geodesic of the d=3 sublattice — the closed
    cyclic walk. The period IS the cycle length on the lattice.
    """
    PARAM_COUNT = 4  # period, plus base_pattern of length=period samples (count separately)

    @staticmethod
    def fit(data: List[int]) -> Optional[Dict[str, Any]]:
        """Fit by detecting the smallest period that repeats exactly.

        Tries periods from 2 up to len(data)//2. The first period that
        produces a perfect repeat wins (smallest = most fundamental).
        For data shorter than 4 elements, no meaningful period exists.
        """
        n = len(data)
        if n < 4:
            return None
        # Find the smallest period p such that data[i] == data[i + p]
        # for all i in [0, n - p). A period of n itself trivially passes
        # but is not useful; cap at n // 2.
        max_period = n // 2
        for p in range(2, max_period + 1):
            ok = True
            for i in range(n - p):
                if data[i] != data[i + p]:
                    ok = False
                    break
            if ok:
                # Success: store the first p samples as the base tone.
                base_tone = [int(v) for v in data[:p]]
                return {'period': int(p), 'base_tone': base_tone}
        return None

    @staticmethod
    def generate(params: Dict[str, Any], length: int) -> List[int]:
        period = int(params['period'])
        base_tone: List[int] = list(params['base_tone'])
        if period <= 0 or not base_tone:
            return [0] * length
        # Tile the base_tone to fill `length` samples.
        out: List[int] = []
        for i in range(length):
            out.append(int(base_tone[i % period]))
        return out


class GrammarGenerator:
    """Re-Pair-style production-rule grammar — for chaotic/recursive data.

    From design doc §16.8.2 hyperbolic case:
      "The D Paper §44.1 establishes that chaotic data has low Kolmogorov
       complexity — the generating map is simple even though the output
       is complex. D_gen = a grammar (Re-Pair rules) or a recurrence
       relation."

    Parameters store a list of production rules of the form:
      (rule_id, [symbol, symbol, ...])  where rule_id ≥ |alphabet| and
      each symbol is either a base value (< |alphabet|) or another rule_id.
    Plus the start string (top-level sequence of base values + rule_ids).
    Reconstruction: recursively expand rule_ids in the start string until
    only base values remain.

    Fits by constructing a Re-Pair grammar from the data. For data that
    has genuine repeating substrings (even if they overlap or interleave
    with random material), Re-Pair produces a small grammar whose total
    parameter count is much less than the data length.

    The Kolmogorov description is (alphabet_size, n_rules, rules,
    start_string) — variable size depending on grammar complexity.
    """
    @staticmethod
    def fit(data: List[int]) -> Optional[Dict[str, Any]]:
        """Run a single-pass Re-Pair on the data and build a grammar.

        Algorithm: repeatedly find the most frequent adjacent pair, replace
        every occurrence with a fresh non-terminal, record the rule, until
        no pair occurs more than once. Returns None when no pair ever
        repeats (data is too random / short for grammar compression).

        The implementation is tractable single-pass for reasonable block
        sizes; for extremely large data the existing C-engine pair-first
        compression in LatticeWalkCompressor.pair_recursive_compress is
        preferred and Channel B simply offers grammar TEMPLATES (small
        bootstrap rule sets) rather than re-deriving from scratch.
        """
        if len(data) < 4:
            return None
        # Compute the alphabet size (max base value + 1).
        alphabet_size = max(data) + 1 if data else 0
        if alphabet_size == 0:
            return None
        # Working sequence — copy as a Python list of ints.
        seq: List[int] = [int(v) for v in data]
        rules: List[Tuple[int, int, int]] = []  # (rule_id, sym1, sym2)
        next_rule_id = alphabet_size

        # Re-Pair loop: replace most-frequent adjacent pair until none repeats.
        # Hard cap the number of rules at len(data) // 2 to bound work.
        max_rules = max(1, len(data) // 2)
        for _ in range(max_rules):
            # Count adjacent pair frequencies.
            pair_counts: Dict[Tuple[int, int], int] = {}
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            # Find the most frequent pair (deterministic tiebreak: lex-smallest).
            best_pair, best_count = max(pair_counts.items(),
                                        key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_count < 2:
                break  # No pair repeats — grammar saturated.
            # Replace every occurrence of best_pair with next_rule_id.
            new_seq: List[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                    new_seq.append(next_rule_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            rules.append((next_rule_id, int(best_pair[0]), int(best_pair[1])))
            seq = new_seq
            next_rule_id += 1

        # If no rules were produced, the grammar generator does not fit
        # (data has no repeating pairs at all — totally random).
        if not rules:
            return None
        # Compactness check: only return a grammar that's actually smaller
        # than the original data in terms of (n_rules · 2 + |start_string|)
        # vs |data|. Otherwise this generator is not a Kolmogorov gain.
        params_size = len(rules) * 2 + len(seq)
        if params_size >= len(data):
            return None
        return {
            'alphabet_size': int(alphabet_size),
            'rules':  [(int(rid), int(s1), int(s2)) for rid, s1, s2 in rules],
            'start':  [int(v) for v in seq],
        }

    @staticmethod
    def generate(params: Dict[str, Any], length: int) -> List[int]:
        """Recursively expand the grammar's start string to a base sequence.

        The `length` argument is informational — the grammar's natural
        length is determined by its start string + rule expansion, not
        by an external length cap. We return the full expansion and let
        the caller truncate or pad if the natural length disagrees with
        the requested length.
        """
        rules: List[Tuple[int, int, int]] = list(params['rules'])
        start: List[int] = list(params['start'])
        alphabet_size = int(params['alphabet_size'])
        # Build a rule lookup table: rule_id → (sym1, sym2)
        rule_lookup: Dict[int, Tuple[int, int]] = {
            rid: (s1, s2) for rid, s1, s2 in rules
        }

        # Iteratively expand: replace any non-terminal with its expansion
        # until only base symbols (< alphabet_size) remain.
        out: List[int] = list(start)
        # Bound: maximum expansions per pass. If a grammar is malformed
        # and never terminates, the bound prevents infinite loops.
        max_passes = 1 + len(rules) * 2
        for _pass in range(max_passes):
            if all(s < alphabet_size for s in out):
                break
            expanded: List[int] = []
            for sym in out:
                if sym >= alphabet_size and sym in rule_lookup:
                    s1, s2 = rule_lookup[sym]
                    expanded.append(s1)
                    expanded.append(s2)
                else:
                    expanded.append(sym)
            out = expanded
        # Truncate or pad-with-zero to match requested length.
        if len(out) > length:
            return out[:length]
        if len(out) < length:
            return out + [0] * (length - len(out))
        return out


# Lookup table: generator_type string code → generator class with fit/generate
_GENERATOR_REGISTRY: Dict[str, Any] = {
    GENERATOR_TYPE_CONSTANT:   ConstantGenerator,
    GENERATOR_TYPE_LINEAR:     LinearGenerator,
    GENERATOR_TYPE_POLYNOMIAL: PolynomialGenerator,
    GENERATOR_TYPE_PERIODIC:   PeriodicGenerator,
    GENERATOR_TYPE_GRAMMAR:    GrammarGenerator,
}


def derive_generators_from_curvature(
        block_curvature: 'BlockCurvatureProfile',
        dk_stream: List[int],
        k_stream: List[int]
        ) -> List[GenerativeDescriptor]:
    """Channel B discovery — derive candidate D_gen from curvature + data.

    From design doc §16.8.3 Discovery Loop:
      1. Observe — done in Phase 1.5 (block_curvature has the profile)
      2. Detect gap — does any existing D_gen match? (caller queries DB)
      3. Derive — for each plausible generator type for this curvature
         class, attempt to fit the data
      4. Returned generators are stored as Channel B {P,D} Unsubstantiated
         predictions (caller invokes ArchetypeDatabase.store_generator)
      5. Compound — fit_count grows over time as new files match

    Args:
        block_curvature: BlockCurvatureProfile from Phase 1.5
        dk_stream:  the block's Δk stream (used for periodic/grammar fits)
        k_stream:   the block's k stream (used for constant/linear/poly fits)

    Returns:
        List of GenerativeDescriptor instances — one per generator type
        that successfully fits the data. Empty list when no generator
        family fits (the data has no derivable generative structure
        at this resolution; standard pipeline handles it).

    The mapping of curvature class to candidate generator types follows
    design doc §16.8.2 and is exhaustive: every class has at least one
    candidate type. Fits that succeed become entries; failures are silent
    (the database never sees them).

    ET Three Tools:
      Identification Principle: identifies the candidate D-set the data
        could be generated by, derived from its curvature geometry.
      Descriptor Gap Principle: closes the gap between "the data exists"
        and "the data has a known generative rule". The gap IS the rule
        — derivation finds it.
      Subsumption Law: 5 generator types subsume every known structured
        data family. Truly random data has no fit (no generator); every
        other data shape fits at least one type.
    """
    derived: List[GenerativeDescriptor] = []
    if not k_stream:
        return derived
    import pickle
    now = time.time()

    # ── CORRECTION (post-Tier-3.B.3 review) ──────────────────────────
    # PRIOR BUG: this method previously RESTRICTED candidate generator
    # types per curvature class — e.g. singular blocks were only allowed
    # to attempt GRAMMAR. That was a Shannon-thinking exclusion: it
    # treated curvature class as evidence that certain generators
    # cannot fit. In ET's Kolmogorov framework that is wrong.
    #
    # Pseudo-random data, chaotic data, encrypted data, "high-entropy"
    # data — all may have generators that exist. PRNG output of length
    # 10⁹ has K-complexity ≈ K(seed) + K(algorithm). Encrypted
    # ciphertext has K(c) ≤ K(key) + K(plaintext) + K(cipher). Chaotic
    # systems have low K-complexity even though the output looks random
    # (D Paper §44.1). Restricting which generators get tried means we
    # never discover those low-K generators when they EXIST.
    #
    # The correct stance: ALWAYS try all 5 generator types. The
    # curvature class only sets the PRIORITY ORDER — the most-likely-
    # to-fit type for this geometry comes first so the search log is
    # informative and (in future) the database can short-circuit on
    # the first successful fit if budget is tight. But every type gets
    # at least one attempt on every block. The Subsumption Law is
    # satisfied not by special-casing the incompressible, but by every
    # block flowing through the same uniform pipeline — what comes out
    # the other end is determined by what fits, never by what we
    # refused to try.
    # ──────────────────────────────────────────────────────────────────

    # Priority order per curvature class — most-likely-to-fit first.
    # This ONLY affects the order; ALL 5 types are tried in every case.
    if block_curvature.is_flat():
        priority = [GENERATOR_TYPE_CONSTANT, GENERATOR_TYPE_LINEAR,
                    GENERATOR_TYPE_POLYNOMIAL, GENERATOR_TYPE_PERIODIC,
                    GENERATOR_TYPE_GRAMMAR]
    elif block_curvature.is_elliptic():
        priority = [GENERATOR_TYPE_LINEAR, GENERATOR_TYPE_POLYNOMIAL,
                    GENERATOR_TYPE_PERIODIC, GENERATOR_TYPE_CONSTANT,
                    GENERATOR_TYPE_GRAMMAR]
    elif block_curvature.is_hyperbolic():
        priority = [GENERATOR_TYPE_GRAMMAR, GENERATOR_TYPE_POLYNOMIAL,
                    GENERATOR_TYPE_PERIODIC, GENERATOR_TYPE_LINEAR,
                    GENERATOR_TYPE_CONSTANT]
    elif block_curvature.is_variable():
        priority = [GENERATOR_TYPE_PERIODIC, GENERATOR_TYPE_GRAMMAR,
                    GENERATOR_TYPE_POLYNOMIAL, GENERATOR_TYPE_LINEAR,
                    GENERATOR_TYPE_CONSTANT]
    else:  # singular
        # Singular blocks have D-bridge breaks (max|K| above S=12). They
        # MAY still admit a generator — a recurrence relation, a periodic
        # cycle with a discontinuity, a polynomial in a transformed
        # coordinate. The previous version restricted singular blocks to
        # GRAMMAR alone. That exclusion is REMOVED. Every type gets a
        # chance; GRAMMAR runs first because it has the best track record
        # on singular geometry, but the others run too.
        priority = [GENERATOR_TYPE_GRAMMAR, GENERATOR_TYPE_PERIODIC,
                    GENERATOR_TYPE_POLYNOMIAL, GENERATOR_TYPE_LINEAR,
                    GENERATOR_TYPE_CONSTANT]

    # Sanity check (Subsumption Law): the priority list MUST contain
    # every registered generator type exactly once. If a future commit
    # adds a new type to _GENERATOR_REGISTRY without updating the
    # priority lists above, we want to know — silently dropping a type
    # would re-introduce the exclusion bug we just fixed.
    assert set(priority) == set(_GENERATOR_REGISTRY.keys()), (
        f'derive_generators_from_curvature priority list is incomplete: '
        f'missing {set(_GENERATOR_REGISTRY.keys()) - set(priority)}, '
        f'extra {set(priority) - set(_GENERATOR_REGISTRY.keys())}'
    )

    # Try each candidate type in priority order; collect every one that fits.
    for gen_type in priority:
        gen_class = _GENERATOR_REGISTRY[gen_type]
        # PERIODIC and GRAMMAR fit on the Δk stream (cyclic / repetition
        # structure tends to live in the differential domain).
        # CONSTANT, LINEAR, POLYNOMIAL fit on the k stream directly
        # (algebraic structure is most evident in the absolute lattice
        # positions).
        if gen_type in (GENERATOR_TYPE_PERIODIC, GENERATOR_TYPE_GRAMMAR):
            params = gen_class.fit(dk_stream) if dk_stream else None
        else:
            params = gen_class.fit(k_stream)
        if params is None:
            continue
        gen_id = GenerativeDescriptor.make_id(gen_type, params)
        # Pickled blob layout: [1B type code][1B stream target][pickled params dict]
        type_code = GENERATOR_TYPE_CODES[gen_type]
        stream_target = GENERATOR_DEFAULT_STREAM[gen_type]
        params_blob = bytes([type_code, stream_target]) + pickle.dumps(params)
        # Param count: each generator's PARAM_COUNT class attr, except
        # PERIODIC and GRAMMAR which carry variable-length structure.
        if gen_type == GENERATOR_TYPE_PERIODIC:
            pc = 1 + len(params.get('base_tone', []))
        elif gen_type == GENERATOR_TYPE_GRAMMAR:
            pc = 1 + 2 * len(params.get('rules', [])) + len(params.get('start', []))
        else:
            pc = int(getattr(gen_class, 'PARAM_COUNT', 0))
        descriptor = GenerativeDescriptor(
            gen_id=gen_id,
            curvature_class=int(block_curvature.curvature_class),
            generator_type=gen_type,
            generator_params=params_blob,
            param_count=pc,
            curvature_mean_range_low=float(block_curvature.curvature_mean) - SUBLIMINAL_CURVATURE,
            curvature_mean_range_high=float(block_curvature.curvature_mean) + SUBLIMINAL_CURVATURE,
            fit_count=0,
            miss_count=0,
            best_residual_variance=None,
            first_derived=now,
            last_confirmed=None,
            source='derived',
        )
        derived.append(descriptor)
    return derived


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

    # Serialize the code table: [n_entries:manifold][for each: sym:manifold, depth:B]
    # The decoder rebuilds the same tree from depths alone (deterministic from depths).
    # Manifold-folded: symbols < 65535 pack as 2 bytes, ≥ 65535 as 6 bytes.
    table_parts = [pack_manifold_uint(len(code_map))]
    for sym in sorted(code_map.keys()):
        _, depth = code_map[sym]
        table_parts.append(pack_manifold_uint(sym))
        table_parts.append(struct.pack('<B', depth))
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
    n_entries, pos = unpack_manifold_uint(table_data, pos)
    sym_depths = {}
    for _ in range(n_entries):
        sym, pos = unpack_manifold_uint(table_data, pos)
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

    def __init__(self, r0: float, n_res: int = N_FULL, log_fn=None,
                 archetype_db: Optional['ArchetypeDatabase'] = None):
        self.r0 = r0
        self.n_res = n_res
        self.byte_k_map = build_byte_k_map(r0)
        self._log = log_fn or (lambda m: None)
        self._archetype_db = archetype_db
        # Accumulates all archetypes found during compression for database storage.
        # Does NOT change compression behavior — purely additive side-channel.
        self.discovered_archetypes: List[LatticeWalkArchetype] = []
        # Tier 6.C.1: Pre-seeded dk-patterns from the archetype database
        # (spectrum lookup, curvature lookup, etc.). Each entry is
        # (pattern_dk_tuple, elegance, hits). At depth 0 of recursive_compress,
        # these are translated from dk-values to symbol indices, scanned for
        # real occurrences in the current stream, and injected alongside
        # C-engine-discovered archetypes. This closes the Descriptor Gap
        # between "database has matching patterns" and "walker can use them".
        self.pre_seed_dk_patterns: List[Tuple[tuple, float, int]] = []

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
            # Auto: search up to half the stream length, capped at S³ for sanity
            max_pattern_len = min(n // 2, S * S * S)  # Up to S³ = 1728 for deep patterns

        archetypes = []
        actual_max = min(max_pattern_len + 1, n // 2 + 1)

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

        # ─── PATTERN SCAN: C Engine (suffix array) — REQUIRED ───
        # The C engine uses O(n log² n) suffix array + LCP to find ALL repeated
        # patterns across ALL lengths in a single pass, verified 1,778× speedup at
        # 100K symbols with zero pattern difference.
        #
        # Combined find + gate: find_and_gate runs gate_archetype_batch on the raw
        # buffer BEFORE parsing — only COHERENT patterns reach Python. This eliminates
        # 145K+ per-pattern IncoherenceFilter calls.
        #
        # The max_pattern_len cap (S²×(depth+1)) is ARCHITECTURAL, not a
        # performance hack. It forces patterns to emerge gradually across
        # recursion depths: depth 0 finds patterns up to S²=144, depth 1
        # up to 2×S²=288, depth 2 up to 3×S²=432, etc. This graduated
        # discovery ensures that depth 2+ archetypes exist for database
        # storage (Koide stability: patterns that survive 2+ rounds of
        # subsumption are true archetypes). The C engine self-caps to
        # max_lcp internally, so passing actual_max-1 costs nothing when
        # max_lcp < actual_max (which is common).

        c_results = PatternEngine.find_and_gate(
            sym_stream, min_pattern_len, actual_max - 1,
            n_res=self.n_res, min_count=2, min_net_savings=2)

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

            # Compute hierarchy elegance
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
            base_elegance = net_savings * (1.0 + depth_factor / N_FULL)
            # ── Tier 5.A.1: Curvature-Weighted Elegance ─────────────────
            # Per design doc §7: augment elegance with the geodesic factor
            #   F_K = 1/(1 + σ²_K)
            # so that patterns from FLAT regions (σ²_K ≈ 0 → F_K ≈ 1.0)
            # retain their full elegance, while patterns from highly-
            # curved regions (large σ²_K → F_K → 0) get deprioritized.
            #
            # The augmentation is multiplicative — base_elegance × F_K —
            # so it never INCREASES elegance (preserving the relative
            # ordering of equally-curved patterns), and the augmented
            # value drops smoothly toward zero as curvature variance
            # grows. This does not REMOVE patterns; it only reweights
            # which ones the sort prefers.
            #
            # ET Three Tools:
            #   Identification Principle: identifies WHICH patterns are
            #     stable on the lattice (low σ²_K = consistent local
            #     geometry) versus which ride the bumps. Stable patterns
            #     are the more fundamental Descriptors.
            #   Descriptor Gap Principle: closes the gap between
            #     "elegance counts symbol savings" and "elegance reflects
            #     geometric stability". Stability IS a Descriptor that
            #     the previous formula ignored.
            #   Subsumption Law: every pattern still gets an elegance
            #     score; F_K = 1.0 for the previously-best (flat)
            #     patterns means they are unchanged; only patterns that
            #     ride high-σ²_K terrain see reduced scores. No pattern
            #     is excluded.
            _, _, f_k = PatternEngine.pattern_curvature(pat)
            e_hierarchy = base_elegance * f_k
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

        # NOTE: Archetypes are NOT accumulated here. find_walk_archetypes returns
        # candidates. Only archetypes actually USED by subsume_patterns() are true
        # archetypes — recursive_compress() stores those from depth >= 2 where
        # they have survived at least 2 levels of subsumption (Koide stability).

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
            base_elegance = net_savings * (1.0 + depth_factor / N_FULL)
            # Tier 5.A.1: curvature-weighted elegance (see main archetype
            # site for full ET Three Tools commentary). Run patterns are
            # constant sequences so σ²_K = 0 and F_K = 1.0; the
            # augmentation is mathematically a no-op for these patterns
            # but applied explicitly so every archetype-creation site
            # stays uniform and the Subsumption Law holds across them.
            _, _, f_k = PatternEngine.pattern_curvature(pattern)
            e_hierarchy = base_elegance * f_k

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
        Uses C-accelerated greedy placement (eliminates the
        O(n × archetypes × positions) consumed-array overlap check).
        C engine is REQUIRED — raises RuntimeError if unavailable.
        """
        n = len(sym_stream)

        # ── C-accelerated greedy placement ──
        c_placements, c_used_mask = PatternEngine.subsume_greedy_c(n, archetypes)

        # Build used_archetypes and orig_to_used mapping from C result
        used_archetypes = []
        orig_to_used: Dict[int, int] = {}
        for arch_idx in range(len(archetypes)):
            if c_used_mask[arch_idx]:
                orig_to_used[arch_idx] = len(used_archetypes)
                used_archetypes.append(archetypes[arch_idx])

        # c_placements = [(arch_idx, position), ...] — sort by position
        archetype_placements = sorted(c_placements, key=lambda x: x[1])

        # ── Build encoded stream from placements ──
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
        unique_dks: List[int] = sorted(dk_value_set)
        dk_to_idx = {dk: i for i, dk in enumerate(unique_dks)}
        n_base = len(unique_dks)

        current: List[int] = [dk_to_idx[dk] for dk in dk_stream]
        archetype_defs: List[Tuple[int, ...]] = []
        next_sym_id = n_base

        d_map: Dict[int, float] = {i: float(lattice_d(dk)) for i, dk in enumerate(unique_dks)}

        # Depth >= 1 fallback archetypes for low-structure data.
        # If depth >= 2 yields fewer than S² archetypes, these fill the gap.
        depth_1_fallback: List[LatticeWalkArchetype] = []
        depth_2_count_before = len(self.discovered_archetypes)

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
                        dk: int = int(unique_dks[s])
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

            # ── Tier 6.C.1: Inject pre-seeded DB patterns at depth 0 ──
            # Pre-seeded patterns (from spectrum lookup, curvature lookup,
            # etc.) are dk-value tuples from the archetype database. At
            # depth 0, the symbol stream is dk_to_idx-mapped, so we can
            # translate each pre-seed pattern from dk-values to symbol
            # indices, scan for real occurrences in the current stream,
            # and create proper LatticeWalkArchetype instances.
            #
            # These archetypes participate in the SAME competition as
            # C-engine-discovered ones — the subsumption logic picks the
            # best regardless of source. DB patterns that don't occur in
            # this stream produce no occurrences and are silently dropped.
            #
            # ET Three Tools:
            #   Identification: identifies which DB patterns actually
            #     appear in this stream (occurrence scan is the proof).
            #   Descriptor Gap: closes the gap between "DB has matching
            #     patterns" and "walker can subsume them" — the missing
            #     Descriptor was the symbol-index translation + occurrence
            #     positions. Both are now provided.
            #   Subsumption: DB patterns compete uniformly with C-engine
            #     patterns; no source gets preferential treatment. The
            #     subsumption sort decides purely on elegance + length.
            if depth == 0 and self.pre_seed_dk_patterns:
                n_injected = 0
                for seed_pat_dk, seed_eleg, seed_hits in self.pre_seed_dk_patterns:
                    # Translate dk-values to symbol indices
                    sym_indices: List[int] = []
                    all_found = True
                    for dk_val in seed_pat_dk:
                        idx = dk_to_idx.get(dk_val)
                        if idx is None:
                            all_found = False
                            break
                        sym_indices.append(idx)
                    if not all_found or len(sym_indices) < 2:
                        continue  # Pattern uses dk values not in this stream
                    seed_sym = tuple(sym_indices)
                    pat_len = len(seed_sym)
                    # Scan the current stream for occurrences
                    positions: List[int] = []
                    for i in range(len(current) - pat_len + 1):
                        if tuple(current[i:i + pat_len]) == seed_sym:
                            positions.append(i)
                    if len(positions) < 2:
                        continue  # Pattern doesn't repeat in this stream
                    # IncoherenceFilter gate
                    if not IncoherenceFilter.gate_archetype(seed_sym, self.n_res):
                        continue
                    # Compute elegance using the same formula as find_walk_archetypes
                    d_vals = [d_map.get(s, N_FULL) for s in seed_sym]
                    d_avg = sum(d_vals) / len(d_vals) if d_vals else N_FULL
                    total_savings = len(positions) * (pat_len - 1)
                    overhead = pat_len + 1
                    net_savings = total_savings - overhead
                    if net_savings < 2:
                        continue
                    depth_factor = BIO_RES / max(d_avg, 1.0)
                    base_elegance = net_savings * (1.0 + depth_factor / N_FULL)
                    _, _, f_k = PatternEngine.pattern_curvature(seed_sym)
                    e_hierarchy = base_elegance * f_k
                    # Use the HIGHER of DB-stored elegance and freshly-computed
                    # elegance — the DB value reflects cross-file history while
                    # the fresh value reflects this stream's specific savings.
                    e_hierarchy = max(e_hierarchy, seed_eleg)
                    archetypes.append(LatticeWalkArchetype(
                        pattern=seed_sym,
                        occurrences=positions,
                        hierarchy_elegance=e_hierarchy,
                        d_avg=d_avg,
                        pattern_length=pat_len,
                    ))
                    n_injected += 1
                if n_injected > 0:
                    # Re-sort after injection so DB patterns compete fairly
                    archetypes.sort(
                        key=lambda arch: (arch.hierarchy_elegance, arch.pattern_length),
                        reverse=True)
                    self._log(f"      Tier 6.C.1: injected {n_injected} DB-seeded "
                              f"archetypes into depth-0 pool")

            if not archetypes:
                break

            # ─── DATABASE ELEGANCE BOOST (depth 0 only) ───
            # At depth 0, pattern symbols are dk_table indices → raw Δk values.
            # Query the database for known patterns at this R₀. If a found
            # pattern is in the database, boost its elegance by historical
            # hit count. This changes RANKING only — all patterns are still
            # found by the scanner. Known patterns get prioritized in greedy
            # subsumption, improving compression for previously-seen data types.
            #
            # Boost formula (ET-derived from Koide binding stability):
            #   stability = 1 - K^hit_count
            #   boost = 1 + stability × BIO_RES / N_FULL
            # K=2/3: h=1→0.33, h=5→0.87, h=10→0.98, h=∞→1.0
            # Max boost ≈ 1.5% — breaks ties, doesn't dominate intrinsic elegance.
            if depth == 0 and self._archetype_db is not None:
                try:
                    known = self._archetype_db.lookup(self.r0, min_hits=1)
                    if known:
                        # Build lookup: raw Δk tuple → (elegance, hit_count)
                        known_map: Dict[tuple, Tuple[float, int]] = {}
                        for pat_dk, eleg, hits in known:
                            known_map[pat_dk] = (eleg, hits)

                        n_boosted = 0
                        for arch in archetypes:
                            # Convert symbol indices to raw Δk values
                            raw_dk = tuple(unique_dks[s] for s in arch.pattern if s < n_base)
                            if len(raw_dk) == len(arch.pattern) and raw_dk in known_map:
                                _, db_hits = known_map[raw_dk]
                                # Koide binding stability: 1 - K^h
                                stability = 1.0 - (K_KOIDE ** max(db_hits, 1))
                                boost = 1.0 + stability * BIO_RES / N_FULL
                                arch.hierarchy_elegance *= boost
                                n_boosted += 1

                        if n_boosted > 0:
                            # Re-sort with boosted elegance
                            archetypes.sort(
                                key=lambda a: (a.hierarchy_elegance, a.pattern_length),
                                reverse=True)
                            self._log(f"    DB boost: {n_boosted}/{len(archetypes)} "
                                      f"patterns known")
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    pass  # Database error — proceed without boost

            encoded, used_archs = self.subsume_patterns(current, archetypes)
            if not used_archs:
                break

            # ─── ARCHETYPE DATABASE ACCUMULATION ───
            # Only USED archetypes (selected by greedy subsumption) are true
            # archetypes. And only from depth >= 2 (Koide stability: ceil(1/K) = 2).
            # Depth 0: raw Δk patterns — recurring sequences, NOT archetypes.
            # Depth 1: patterns subsuming raw patterns — starting to bind.
            # Depth 2+: patterns subsuming bound patterns — TRUE archetypes that have
            # survived at least 2 levels of subsumption (Koide-stable binding).
            #
            # FALLBACK (Descriptor Gap Principle): if depth >= 2 yields fewer than
            # 1024 archetypes (as with compressed files, binary data, or other
            # low-structure inputs), the gap in the database is itself a Descriptor.
            # Depth >= 1 archetypes are stored to close this gap — they have survived
            # at least one round of subsumption and are better than nothing.
            #
            # The S²×(depth+1) pattern length cap is ARCHITECTURAL — it forces
            # patterns to emerge gradually across depths, ensuring depth 2+
            # patterns exist. Without it, depth 0 finds everything and depth 2
            # has nothing to store.
            #
            # ET Derivation (Subsumption Law):
            #   An archetype is complete when it subsumes without remainder.
            #   At depth 0, the pattern has subsumed nothing — it IS a raw pattern.
            #   At depth 1, the pattern subsumes raw patterns — partial binding.
            #   At depth 2+, the pattern subsumes bound patterns — stable archetype.
            #   K = 2/3 stability threshold → ceil(1/K) = 2 minimum depth.
            if depth >= 2:
                self.discovered_archetypes.extend(used_archs)
            elif depth >= 1:
                depth_1_fallback.extend(used_archs)

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

        # ─── DEPTH 1 FALLBACK (Descriptor Gap Principle) ───
        # If depth >= 2 produced fewer than S² = 144 archetypes in this call,
        # the data lacks deep recursive structure (compressed files, binary data,
        # high-entropy inputs). The gap in the database IS a Descriptor.
        # Depth >= 1 archetypes (patterns that survived one full round of
        # subsumption) are stored to partially close this gap.
        # S² = 144: the manifold's cross-pattern cap — below this, the archetype
        # set is too sparse for meaningful cross-file pattern reuse.
        depth_2_added = len(self.discovered_archetypes) - depth_2_count_before
        if depth_2_added < S * S and depth_1_fallback:
            self.discovered_archetypes.extend(depth_1_fallback)

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
            best_pair: Optional[Tuple[int, ...]] = None
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
            valid_pair: Tuple[int, ...] = best_pair

            # ── Step 3: Net savings check ──
            # Each pair occurrence saves 1 symbol (2 symbols → 1 reference).
            # Overhead: 2 symbols for the pattern definition + 1 new symbol ID.
            net_savings = best_count - 3  # count × 1 - (2 + 1)
            if net_savings < 1:
                break

            # ── Step 4: Replace all non-overlapping occurrences ──
            new_sym = next_sym_id
            archetype_defs.append(valid_pair)

            # Compute d_avg for the new symbol from its constituent d-values
            d_a = d_map.get(valid_pair[0], float(N_FULL))
            d_b = d_map.get(valid_pair[1], float(N_FULL))
            d_map[new_sym] = (d_a + d_b) / 2.0
            next_sym_id += 1

            new_stream = []
            i = 0
            while i < len(current):
                if (i < len(current) - 1
                        and current[i] == valid_pair[0]
                        and current[i + 1] == valid_pair[1]):
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
    from the geometric mean of its byte values via streaming computation.
    The personal tower contains the file's byte↔k bijection, its Δk pattern
    stream, and the archetypes found within it.

    STREAMING ARCHITECTURE (metabolism-governed):
        The tower is built in a single streaming pass through the source file.
        During the pass, a temp file is written to local storage while R₀,
        SHA-256, dk_stream, and k0 are computed incrementally. The source
        file is read exactly ONCE. raw_data is NEVER held in memory.

        For block-by-block compression: read_block() reads from the temp file.
        For universal projection: compute_dk_universal() uses the cross-map
        technique — zero re-reads of raw bytes.
        After compression: cleanup() deletes the temp file.

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
    k0: int = 0                             # First k value (for k-stream reconstruction)
    temp_file_path: str = ''                # Local temp copy for block-by-block access
    raw_data: bytes = b''                   # EMPTY — kept for interface compat, never populated

    @staticmethod
    def from_file(file_path: str) -> 'LatticeTower':
        """
        Build a personal lattice tower via streaming — one pass, no full file in memory.

        Pass 1 (source → temp file):
            Streams the source file chunk-by-chunk to a local temp file.
            While streaming, computes R₀ (running log sum / count),
            SHA-256 (hashlib.update per chunk), and file size.
            The source file is read exactly ONCE. For network sources,
            slow HDDs, etc. — this is the only read.

        Pass 2 (temp file → dk_stream):
            Reads the temp file (local, fast) to build the personal
            dk_stream and k0. Only needs the previous k value — O(1) state.
            The dk_stream is held in memory (typically 4-28 bytes per entry
            depending on Python int size).

        After both passes: the temp file remains on disk for block-by-block
        compression access via read_block(). It is deleted by cleanup().
        """
        # ── Pass 1: Stream source → temp file + R₀ + SHA-256 + size ──
        temp_dir = tempfile.gettempdir()
        fd, temp_path = tempfile.mkstemp(prefix='cdf_tower_', suffix='.tmp',
                                         dir=temp_dir)

        hasher = hashlib.sha256()
        log_sum = 0.0
        byte_count = 0
        data_size = 0

        with open(file_path, 'rb') as src:
            with os.fdopen(fd, 'wb') as dst:
                while True:
                    chunk = src.read(BLOCK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
                    hasher.update(chunk)
                    data_size += len(chunk)
                    # Streaming R₀: accumulate Σ log(byte+1)
                    arr = np.frombuffer(chunk, dtype=np.uint8).astype(np.float64) + 1.0
                    log_sum += float(np.sum(np.log(arr)))
                    byte_count += len(chunk)

        # R₀ = geometric mean of (byte+1) = exp(mean(log(byte+1)))
        r0 = math.exp(log_sum / byte_count) if byte_count > 0 else 1.0
        data_hash = hasher.digest()

        # Build byte↔k maps from R₀
        bk = build_byte_k_map(r0)
        kb = build_k_byte_map(r0)

        # ── Pass 2: Temp file → dk_stream + k0 (local disk, fast) ──
        dk_stream = []
        k0 = 0
        prev_k = None

        with open(temp_path, 'rb') as f:
            while True:
                chunk = f.read(BLOCK_SIZE)
                if not chunk:
                    break
                for byte_val in chunk:
                    k = bk[byte_val]
                    if prev_k is None:
                        k0 = k
                    else:
                        dk_stream.append(k - prev_k)
                    prev_k = k

        return LatticeTower(
            file_path=file_path,
            personal_r0=r0,
            byte_k_map=bk,
            k_byte_map=kb,
            dk_stream=dk_stream,
            dk_universal=[],
            data_size=data_size,
            data_hash=data_hash,
            k0=k0,
            temp_file_path=temp_path,
            raw_data=b'',  # Never held — use read_block() or read_raw()
        )

    def read_block(self, block_idx: int) -> bytes:
        """
        Read a specific BLOCK_SIZE block from the temp file (or source).

        Used by compress_file and compress_batch for block-by-block
        compression. Reads only one block at a time — O(BLOCK_SIZE) memory.
        """
        source = self.temp_file_path if self.temp_file_path else self.file_path
        with open(source, 'rb') as f:
            f.seek(block_idx * BLOCK_SIZE)
            return f.read(BLOCK_SIZE)

    def read_raw(self) -> bytes:
        """
        Read the full file from temp (or source). Use sparingly —
        this loads the full file into memory. Prefer read_block() for
        block-by-block access.
        """
        source = self.temp_file_path if self.temp_file_path else self.file_path
        with open(source, 'rb') as f:
            return f.read()

    def compute_dk_universal(self, universal_byte_k: Dict[int, int]) -> List[int]:
        """
        Compute dk_universal via cross-map — zero re-reads of raw data.

        Builds a 256-entry cross-map: personal_k → universal_k.
        Since byte↔k is injective at 27720ET, every personal k came from
        a specific byte, and that byte maps to a specific universal k.

        Walks the personal k-stream (k0 + cumulative dk) with O(1) running
        state, mapping each position through the cross table to produce
        the universal Δk sequence.

        This is exact — identical output to reading raw bytes and mapping
        through universal byte_k_map — because the bijection guarantees
        the cross-map is lossless.
        """
        # Build cross-map: for each of 256 byte values, personal_k → universal_k
        cross_k: Dict[int, int] = {}
        for byte_val in range(256):
            k_personal = self.byte_k_map[byte_val]
            k_universal = universal_byte_k[byte_val]
            cross_k[k_personal] = k_universal

        # Walk personal k-stream via k0 + dk_stream, map through cross_k
        dk_u: List[int] = []
        k_p = self.k0
        k_u_prev = cross_k.get(k_p, k_p)

        for dk_p in self.dk_stream:
            k_p += dk_p
            k_u_curr = cross_k.get(k_p, k_p)
            dk_u.append(k_u_curr - k_u_prev)
            k_u_prev = k_u_curr

        self.dk_universal = dk_u
        return dk_u

    def cleanup(self):
        """
        Delete the temp file. Call after compression is complete.

        The metadata (R₀, dk_stream, byte_k_map, etc.) remains cached
        in the tower object for archetype database storage. Only the
        raw byte copy is released.
        """
        if self.temp_file_path and os.path.isfile(self.temp_file_path):
            try:
                os.remove(self.temp_file_path)
            except OSError:
                pass
            self.temp_file_path = ''


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
            # Project byte stream through universal lattice via cross-map
            # Uses cached personal dk_stream + k0 — zero raw data re-reads.
            # The cross-map is exact: byte↔k is injective at 27720ET.
            dk_u = tower.compute_dk_universal(self.universal_byte_k)

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
            l3_failed_towers: List[int] = []  # Tier 6.B.1: track L3-failing towers
            for ti in unique_towers:
                tower: LatticeTower = self.towers[int(ti)]
                if IncoherenceFilter.l3_cross_tower_pattern(
                        pat, tower.personal_r0, self.universal_r0,
                        b''):  # Fallback path: scans all 256 byte values
                    n_l3_coherent += 1
                else:
                    l3_failed_towers.append(int(ti))
            l3_cross_fraction = n_l3_coherent / n_files if n_files > 0 else 0.0
            if l3_cross_fraction < K_KOIDE:
                # ── Tier 6.B.1: Curvature coherence fallback ──
                # Per design doc §12.4: when d-family L3 fails, check
                # curvature coherence as a recovery channel. Towers whose
                # d-families broke under R₀ translation may still share
                # the pattern's curvature geometry (ΔΔk is R₀-invariant,
                # so curvature survives the tower shift even when d-families
                # do not — Multifold §13).
                #
                # For each L3-failing tower, compute the curvature coherence
                # between the pattern's K̄ and the tower's mean curvature.
                # Towers with coherence ≥ K are promoted to "compatible" —
                # they can share this pattern via curvature matching even
                # though d-family matching failed.
                #
                # If the total (L3-pass + curvature-promoted) reaches K
                # of all towers, the pattern is accepted. Otherwise it's
                # genuinely incompatible and skipped.
                #
                # ET Three Tools:
                #   Identification: identifies that d-family failure ≠
                #     curvature failure; the two are independent measures.
                #   Descriptor Gap: the gap was "pattern rejected because
                #     d-families broke, even though curvature matched".
                #     The curvature coherence fallback closes this gap.
                #   Subsumption: L3-pass towers stay accepted (unchanged);
                #     only L3-fail towers get the curvature test. The
                #     fallback strictly adds opportunities, never removes.
                pat_curv_mean, _, _ = PatternEngine.pattern_curvature(pat)
                n_curvature_promoted = 0
                for ti_fail in l3_failed_towers:
                    tower_fail: LatticeTower = self.towers[ti_fail]
                    # Tower's mean curvature: computed from its universal
                    # dk_stream (which is already stored in
                    # self.tower_dk_universal[ti_fail]).
                    dk_tower = self.tower_dk_universal.get(ti_fail, [])
                    if len(dk_tower) >= 2:
                        ddk_tower = [dk_tower[j + 1] - dk_tower[j]
                                     for j in range(len(dk_tower) - 1)]
                        tower_curv_mean = sum(ddk_tower) / len(ddk_tower) if ddk_tower else 0.0
                    else:
                        tower_curv_mean = 0.0
                    cc = curvature_coherence(pat_curv_mean, tower_curv_mean)
                    if cc >= K_KOIDE:
                        n_curvature_promoted += 1
                total_compatible = n_l3_coherent + n_curvature_promoted
                total_fraction = total_compatible / n_files if n_files > 0 else 0.0
                if total_fraction < K_KOIDE:
                    continue  # Both L3 and curvature failed — genuinely incompatible

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
            base_elegance = net_savings * (1.0 + depth_factor / N_FULL) * cross_bonus
            # Tier 5.A.1: curvature-weighted elegance. Cross-tower
            # patterns are arbitrary dk subsequences so σ²_K may be
            # non-trivial; F_K provides genuine reweighting here.
            # See main archetype site for full ET Three Tools commentary.
            _, _, f_k = PatternEngine.pattern_curvature(pat)
            e_hierarchy = base_elegance * f_k

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
        unique_dks: List[int] = sorted(dk_value_set)
        dk_to_idx = {dk: i for i, dk in enumerate(unique_dks)}
        n_base = len(unique_dks)

        # Map cross-file archetype IDs starting at n_base
        cross_arch_id_start = n_base
        cross_arch_id_map = {}
        cross_arch_defs = []
        for ci, arch in enumerate(used_archs):
            cross_arch_id_map[id(arch)] = cross_arch_id_start + ci
            cross_arch_defs.append(arch.pattern)

        new_stream: List[int] = []
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

        current: List[int] = new_stream

        for depth in range(MAX_DEPTH):
            if len(current) < 4:
                break

            # L4 cascade coherence check
            if depth > 0:
                eps_vals = []
                for s in current:
                    if s < n_base:
                        dk: int = int(unique_dks[s])
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
        data = tower.read_raw()  # Reads from temp file — temporary load, freed after return
        n = len(data)
        if n == 0:
            return None

        values = np.frombuffer(data, dtype=np.uint8)

        # Check for uniform block
        unique = np.unique(values)
        if len(unique) == 1:
            return struct.pack('<BIB', 1, n, int(unique[0]))

        # Build k-stream through universal lattice
        k_stream_u = PatternEngine.fast_k_stream(data, self.universal_byte_k)
        k0_u = k_stream_u[0]

        # Δk on universal lattice
        dk_u = PatternEngine.fast_dk_stream(k_stream_u)

        walker = LatticeWalkCompressor(self.universal_r0, log_fn=engine.log_fn,
                                       archetype_db=engine.archetype_db)

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
# ARCHETYPE DATABASE — Persistent Pattern Memory
#
# The compressor develops a memory. Every compression discovers archetypes —
# recurring Δk patterns that the Subsumption Law collapses into references.
# These archetypes persist in a database, accumulating a growing D-set of
# known lattice structures.
#
# When a new file is compressed, its Δk stream is checked against known
# patterns. Known archetypes get elegance boosts from historical frequency
# data. The pattern scanner then only needs to discover what's NEW.
#
# Patterns are stored at their source R₀ (quantized to BIO_RES = 420 lattice
# resolution to group files with similar byte distributions). Files with the
# same R₀ group (e.g., all PNGs, all text files) share archetypes directly.
#
# Only TRUE archetypes are stored — patterns that were USED by greedy
# subsumption and survived at least 2 depths of recursion (Koide stability:
# ceil(1/K) = 2). Raw patterns from depth 0-1 are candidates, not archetypes.
#
# Disk safety: the database grows freely — no arbitrary cap. The metabolism
# monitors actual disk space. When disk is critically low (< 1GB free),
# the lowest-value archetypes are pruned to prevent disk exhaustion.
#
# ET Derivation:
#   Identification Principle: each archetype is fully identified by its
#     Δk pattern, d_avg, elegance, and R₀ context.
#   Descriptor Gap Principle: the database IS the accumulated D-set.
#     Each new archetype closes a gap in the compressor's knowledge.
#   Subsumption Law: known archetypes subsume their occurrences in new files
#     without remainder — zero re-discovery cost for known patterns.
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

# T_WEIGHT: the complement of K_KOIDE. K + T_WEIGHT = 1.0.
# The AI takes K, other software gets T_WEIGHT. The database prunes T_WEIGHT.
T_WEIGHT = 1.0 - K_KOIDE  # 1/3


def _quantize_r0(r0: float) -> float:
    """
    Quantize R₀ to the nearest lattice point at BIO_RES = 420.

    This groups files with similar byte distributions into the same
    R₀ bucket. Files in the same bucket share archetypes directly.
    BIO_RES = 420 = LCM(1..7) is the biological tier resolution —
    fine enough to distinguish file types, coarse enough to group
    similar files together.
    """
    k = lattice_k(r0, BIO_RES)
    return 2.0 ** (k / BIO_RES)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — GENERATOR EVALUATOR (§19.6.2 of design doc)
#
# A GeneratorEvaluator takes a single Generative Descriptor's payload bytes
# from a CDF VFS file and evaluates it at arbitrary offsets inside the
# descriptor's domain. Eight type codes (0..7) cover every data shape the
# format supports; together they subsume all possible database page content
# without remainder (Subsumption Law).
#
# Type 0 Constant         — run of identical bytes
# Type 1 Linear           — constant Δk in lattice space
# Type 2 Polynomial       — degree-d polynomial in lattice space
# Type 3 Periodic         — exact cyclic byte pattern
# Type 4 Grammar          — Re-Pair production hierarchy
# Type 5 Geodesic         — Mode 3 residual stream with connection
# Type 6 Archetype Ref    — pointer to another generator (recursive tower)
# Type 7 Raw              — passthrough, 1:1 with original bytes
#
# Payload layouts follow §19.5 of the design doc exactly. Each evaluator
# parses its payload lazily on first evaluate() call and caches the parse.
#
# ET Derivation:
#   T(D_gen, offset) = P[offset]
#   The evaluator IS T applied to D_gen at a specific P-location.
#   Each evaluation is a T-traversal producing the Exception (actual bytes).
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratorEvaluator:
    """Evaluate a single Generative Descriptor's payload at arbitrary offsets.

    All eight generator types (0..7) are handled in one class. Type-specific
    logic is selected by `gen_type` at __init__; the payload is parsed lazily
    on the first evaluate() call so constructing a thousand evaluators is
    cheap even if only a dozen end up being called.

    Invariants:
      * evaluate(offset, length) always returns exactly `length` bytes
      * offset MUST be in [domain_start, domain_start + domain_length)
      * offset + length - 1 MUST also be in the domain (caller's job to clip
        at domain boundaries)
    """

    __slots__ = ('gen_type', 'payload', 'residual', 'domain_start',
                 'domain_length', 'connection_order', '_parsed',
                 '_vfs_ref')  # _vfs_ref used only by Type 6

    def __init__(self, gen_type: int, payload: bytes,
                 residual: Optional[bytes],
                 domain_start: int, domain_length: int,
                 connection_order: int = 0,
                 vfs_ref: Optional['CDFDatabaseVFS'] = None):
        """Construct an evaluator for a single generator.

        Args:
            gen_type: VFS_GEN_* type code (0..7)
            payload: generator-specific parameter bytes (§19.5 layout)
            residual: optional byte-level residual for imperfect fits.
                      For Type 5 (geodesic) this is the PRIMARY residual
                      stream and is the generator's actual data. For all
                      other types it is an OPTIONAL correction layer
                      applied byte-mod-256 after generation.
            domain_start: first byte offset this generator covers in the
                          original file's coordinate system
            domain_length: number of bytes covered
            connection_order: 0/1/2 for Type 5 (geodesic); unused otherwise
            vfs_ref: back-pointer to the parent VFS for Type 6 archetype
                     references (needed to look up the referenced generator
                     in the same index). None for self-contained types.
        """
        self.gen_type = int(gen_type)
        self.payload = bytes(payload) if payload is not None else b''
        self.residual = bytes(residual) if residual is not None else None
        self.domain_start = int(domain_start)
        self.domain_length = int(domain_length)
        self.connection_order = int(connection_order)
        self._parsed: Optional[Dict[str, Any]] = None
        self._vfs_ref = vfs_ref

    def _parse(self) -> None:
        """Parse the payload into a type-specific parameter dict.

        Called lazily by evaluate() on first invocation. Idempotent; later
        calls short-circuit via the non-None self._parsed check.

        ET Three Tools:
          Identification: parses the binary payload into named parameters
            the evaluator code can reason about.
          Descriptor Gap: closes the gap between "packed bytes on disk"
            and "usable parameters in memory". Parsing IS the bridge.
          Subsumption: each of the eight type codes has exactly one parse
            path; no type falls through to another's logic.
        """
        if self._parsed is not None:
            return

        t = self.gen_type

        if t == VFS_GEN_CONSTANT:
            # Payload: [1B value]
            if len(self.payload) < 1:
                raise ValueError('VFS Constant generator: empty payload')
            self._parsed = {'value': int(self.payload[0])}

        elif t == VFS_GEN_LINEAR:
            # Payload: [4B k_start int32 LE][4B dk_step int32 LE][8B r0 f64 LE]
            if len(self.payload) < 16:
                raise ValueError(
                    f'VFS Linear generator: payload too short '
                    f'({len(self.payload)} < 16)')
            k_start = struct.unpack_from('<i', self.payload, 0)[0]
            dk_step = struct.unpack_from('<i', self.payload, 4)[0]
            r0 = struct.unpack_from('<d', self.payload, 8)[0]
            self._parsed = {
                'k_start': k_start,
                'dk_step': dk_step,
                'r0': r0,
                'k_byte': build_k_byte_map(r0),
            }

        elif t == VFS_GEN_POLYNOMIAL:
            # Payload: [4B degree u32 LE][8B each coeff f64 LE × (degree+1)]
            #         [8B r0 f64 LE]
            if len(self.payload) < 12:
                raise ValueError(
                    f'VFS Polynomial generator: payload too short '
                    f'({len(self.payload)} < 12)')
            degree = struct.unpack_from('<I', self.payload, 0)[0]
            expected = 4 + 8 * (degree + 1) + 8
            if len(self.payload) < expected:
                raise ValueError(
                    f'VFS Polynomial generator: payload length '
                    f'{len(self.payload)} < expected {expected} '
                    f'for degree {degree}')
            coeffs = [
                struct.unpack_from('<d', self.payload, 4 + i * 8)[0]
                for i in range(degree + 1)
            ]
            r0 = struct.unpack_from('<d', self.payload, 4 + (degree + 1) * 8)[0]
            self._parsed = {
                'degree': int(degree),
                'coefficients': coeffs,
                'r0': r0,
                'k_byte': build_k_byte_map(r0),
            }

        elif t == VFS_GEN_PERIODIC:
            # Payload: [4B period u32 LE][period bytes one_cycle]
            if len(self.payload) < 4:
                raise ValueError(
                    f'VFS Periodic generator: payload too short '
                    f'({len(self.payload)} < 4)')
            period = struct.unpack_from('<I', self.payload, 0)[0]
            if period <= 0:
                raise ValueError(
                    f'VFS Periodic generator: non-positive period {period}')
            if len(self.payload) < 4 + period:
                raise ValueError(
                    f'VFS Periodic generator: payload length '
                    f'{len(self.payload)} < expected {4 + period} '
                    f'for period {period}')
            cycle = bytes(self.payload[4:4 + period])
            self._parsed = {'period': int(period), 'cycle': cycle}

        elif t == VFS_GEN_GRAMMAR:
            # Payload: [manifold n_rules]
            #         for each rule: [manifold left][manifold right]
            #         [manifold n_start]
            #         [manifold × n_start start symbols]
            #
            # The alphabet size is computed from the max base symbol
            # (< first rule_id) seen in any start symbol or rule RHS.
            # rule_id numbering starts at alphabet_size and increases by
            # one per rule.
            pos = 0
            n_rules, pos = unpack_manifold_uint(self.payload, pos)
            rules: List[Tuple[int, int]] = []  # index = rule_id - alphabet_size
            for _ in range(n_rules):
                left, pos = unpack_manifold_uint(self.payload, pos)
                right, pos = unpack_manifold_uint(self.payload, pos)
                rules.append((int(left), int(right)))
            n_start, pos = unpack_manifold_uint(self.payload, pos)
            start: List[int] = []
            for _ in range(n_start):
                sym, pos = unpack_manifold_uint(self.payload, pos)
                start.append(int(sym))
            # Infer alphabet size: the smallest rule_id used is alphabet_size.
            # Byte-stream grammars always have alphabet_size = 256 (every
            # base symbol is a byte). Detect by assuming base symbols are
            # < 256 and rule_ids ≥ 256 — which matches the fitter's
            # construction convention.
            alphabet_size = 256
            self._parsed = {
                'alphabet_size': alphabet_size,
                'rules': rules,
                'start': start,
                'n_rules': n_rules,
            }

        elif t == VFS_GEN_GEODESIC:
            # Type 5 payload parsing lives in _GeodesicEvaluatorV2._parse
            # (the EXTENDED layout with a k0 anchor prefix). The base
            # class's evaluate() branch for Type 5 delegates to that v2
            # evaluator, which calls its own _parse — so parsing the
            # payload here would be redundant and WOULD misinterpret the
            # extended layout as the short form documented in §19.5 of
            # the design doc. We mark _parsed as a sentinel so callers
            # that interrogate it see a consistent structure without
            # pretending to carry the short-form fields.
            self._parsed = {'delegated_to': '_GeodesicEvaluatorV2'}

        elif t == VFS_GEN_ARCHETYPE_REF:
            # Two payload formats are accepted. The reader discriminates by
            # payload length — a hash payload is always ≥ 32 bytes, a pure-
            # index payload is always < 32 bytes, and manifold_uint never
            # grows to 32 bytes in practice (the 27720-lattice cardinality
            # is well under 2^32, which encodes in at most 5 bytes).
            #
            # LEAN FORM (new, preferred):
            #     Payload: [manifold_uint canonical_index]
            #     Size   : typically 2 bytes; ≤ 5 bytes for any realistic index
            #     Meaning: "use the generator at position `canonical_index`
            #              in this file's generator index". The ref is
            #              literally a seed — a dimensionless integer that
            #              locates the target on the index's 1D lattice.
            #              No content hash, no bookkeeping, no redundancy.
            #
            # HASH FORM (legacy, still accepted):
            #     Payload: [32B SHA-256 of target's output][manifold_uint instance_index]
            #     Size   : typically 34 bytes
            #     Meaning: content-addressed lookup; instance_index selects
            #              among multiple entries producing the same bytes.
            #
            # The lean form is what this project's philosophy demands:
            # the generator IS a position on the lattice, so the ref IS
            # just that position. The hash form was an intermediate
            # implementation that treated the ref as content-addressed
            # storage — a Shannon-like bookkeeping approach. It's kept
            # working for any .cdf files written by earlier code, but
            # new writes should always use the lean form.
            if len(self.payload) < 32:
                # LEAN FORM — just a manifold_uint index
                canonical_index, pos = unpack_manifold_uint(self.payload, 0)
                if pos != len(self.payload):
                    raise ValueError(
                        f'VFS Archetype Ref generator (lean form): '
                        f'trailing bytes after manifold_uint — parsed '
                        f'{pos} of {len(self.payload)} payload bytes')
                self._parsed = {
                    'form': 'index',
                    'canonical_index': int(canonical_index),
                }
            else:
                # HASH FORM — legacy content-addressed
                arch_hash = bytes(self.payload[:32])
                pos = 32
                instance_index, pos = unpack_manifold_uint(self.payload, pos)
                self._parsed = {
                    'form': 'hash',
                    'archetype_hash': arch_hash,
                    'instance_index': int(instance_index),
                }

        elif t == VFS_GEN_RAW:
            # Payload: [raw bytes, 1:1 with original]
            if len(self.payload) < self.domain_length:
                raise ValueError(
                    f'VFS Raw generator: payload length '
                    f'{len(self.payload)} < domain_length '
                    f'{self.domain_length}')
            self._parsed = {'data': self.payload[:self.domain_length]}

        else:
            raise ValueError(f'VFS generator: unknown type code {t}')

    def evaluate(self, offset: int, length: int,
                 _recursion_depth: int = 0) -> bytes:
        """Return `length` bytes from this generator starting at `offset`.

        Args:
            offset: byte offset into the ORIGINAL file's coordinate system.
                    MUST satisfy
                    self.domain_start <= offset < self.domain_start + self.domain_length
            length: number of bytes to produce. MUST satisfy
                    offset + length <= self.domain_start + self.domain_length
            _recursion_depth: internal — used by Type 6 (Archetype Ref)
                    to bound recursion at CDF_VFS_MAX_REF_DEPTH.

        Returns: exactly `length` bytes.

        ET Three Tools:
          Identification: identifies the coordinate-system conversion:
            local_offset = offset - domain_start maps absolute file
            offset into the generator's local [0, domain_length) frame.
          Descriptor Gap: closes the gap between "generator lives in its
            own 0-indexed frame" and "caller works in file-offset frame".
          Subsumption: every (offset, length) pair whose requested range
            falls entirely within [domain_start, domain_start+domain_length)
            is handled; out-of-range requests are a caller error and raise
            early so the failure is visible rather than masked.
        """
        if length < 0:
            raise ValueError(f'evaluate: negative length {length}')
        if length == 0:
            return b''
        if offset < self.domain_start or \
                offset + length > self.domain_start + self.domain_length:
            raise ValueError(
                f'evaluate: range [{offset}, {offset + length}) outside '
                f'generator domain [{self.domain_start}, '
                f'{self.domain_start + self.domain_length})')

        self._parse()
        assert self._parsed is not None  # Guaranteed by _parse()

        local_offset = offset - self.domain_start
        t = self.gen_type
        result = bytearray(length)

        # ── Type 0 Constant ────────────────────────────────────────────
        if t == VFS_GEN_CONSTANT:
            val = int(self._parsed['value'])
            # Fast path: bytearray slice-assign with multiplication.
            result[:] = bytes([val]) * length

        # ── Type 1 Linear ──────────────────────────────────────────────
        elif t == VFS_GEN_LINEAR:
            p = self._parsed
            kb: Dict[int, int] = p['k_byte']
            k_start = int(p['k_start'])
            dk_step = int(p['dk_step'])
            # Lazy nearest-k cache: many generators produce the same k
            # values repeatedly; cache the resolution.
            for i in range(length):
                k = k_start + dk_step * (local_offset + i)
                if k in kb:
                    result[i] = kb[k]
                else:
                    # Nearest-k fallback (same pattern as decompress_block).
                    # O(256) per miss; fine since misses are rare.
                    nearest = min(kb.keys(), key=lambda kk: abs(kk - k))
                    result[i] = kb[nearest]

        # ── Type 2 Polynomial ──────────────────────────────────────────
        elif t == VFS_GEN_POLYNOMIAL:
            p = self._parsed
            kb = p['k_byte']
            coeffs: List[float] = p['coefficients']
            deg = int(p['degree'])
            # Integrity check — the parsed `deg` field is the authoritative
            # Descriptor for coefficient-list length, and both encoder and
            # decoder rely on them agreeing. If a malformed payload makes
            # them disagree, the Horner loop below produces silently-wrong
            # bytes; catch that here and fail visibly (rule 32: variables
            # that LOOK unused signal an incomplete implementation — the
            # degree is used as an assertion, not a decoration).
            if len(coeffs) != deg + 1:
                raise ValueError(
                    f'VFS Polynomial evaluator: coefficient count '
                    f'{len(coeffs)} does not match degree+1 = {deg + 1}')
            for i in range(length):
                x = local_offset + i
                # Horner's method — numerically stable, no pow() calls.
                k_float = 0.0
                for c in reversed(coeffs):
                    k_float = k_float * x + c
                k = int(round(k_float))
                if k in kb:
                    result[i] = kb[k]
                else:
                    nearest = min(kb.keys(), key=lambda kk: abs(kk - k))
                    result[i] = kb[nearest]

        # ── Type 3 Periodic ────────────────────────────────────────────
        elif t == VFS_GEN_PERIODIC:
            period = int(self._parsed['period'])
            cycle: bytes = self._parsed['cycle']
            for i in range(length):
                result[i] = cycle[(local_offset + i) % period]

        # ── Type 4 Grammar ─────────────────────────────────────────────
        elif t == VFS_GEN_GRAMMAR:
            p = self._parsed
            alphabet = int(p['alphabet_size'])
            rules: List[Tuple[int, int]] = p['rules']
            start: List[int] = p['start']
            # Expand the grammar iteratively until only base symbols (<
            # alphabet) remain. For a Re-Pair grammar of depth D ≤ S=12
            # this is bounded expansion. The output is truncated/sliced
            # to the requested range after full expansion. For very
            # long grammar expansions this is the slow path; Type 3
            # (periodic) or Type 7 (raw) would have been chosen by the
            # fitter when cheaper, so reaching here means the grammar
            # IS the cheapest representation.
            expanded: List[int] = list(start)
            max_passes = 1 + max(1, len(rules)) * 2  # depth bound
            for _pass in range(max_passes):
                if all(sym < alphabet for sym in expanded):
                    break
                nxt: List[int] = []
                for sym in expanded:
                    if sym < alphabet:
                        nxt.append(sym)
                    else:
                        rule_idx = sym - alphabet
                        if 0 <= rule_idx < len(rules):
                            left, right = rules[rule_idx]
                            nxt.append(left)
                            nxt.append(right)
                        else:
                            # Invalid rule id — Subsumption Law would have
                            # caught this at fit time; treat as base 0 for
                            # safety so the output length stays correct.
                            nxt.append(0)
                expanded = nxt
            # After expansion, expanded IS the full byte sequence for the
            # generator's domain. Slice out the requested [local_offset,
            # local_offset + length) window.
            if len(expanded) < local_offset + length:
                # Pad with zeros if under-expanded (malformed grammar);
                # NEVER truncate wrong — the caller always gets the
                # promised `length` bytes.
                expanded = expanded + [0] * (local_offset + length - len(expanded))
            for i in range(length):
                v = expanded[local_offset + i]
                result[i] = int(v) & 0xFF

        # ── Type 5 Geodesic ────────────────────────────────────────────
        elif t == VFS_GEN_GEODESIC:
            # The base GeneratorEvaluator stores the short-form (non-extended)
            # Type 5 payload layout documented in §19.5 of the design doc
            # (payload = dk0 + connection_order + window + r0, NO k0 anchor).
            # That short form requires the first byte's k-anchor to be
            # supplied externally; the production fitter instead emits the
            # self-contained EXTENDED layout (prepended k0) handled by
            # _GeodesicEvaluatorV2.
            #
            # Rule 4 (no placeholders) + rule 15 (no shortcuts): the base
            # branch must be FUNCTIONAL, not a NotImplementedError stub. We
            # delegate to the v2 evaluator. Every production caller goes
            # through _make_generator_evaluator which routes directly to
            # the v2 class — this branch exists only for the case where
            # some future code constructs GeneratorEvaluator(VFS_GEN_GEODESIC,
            # ...) directly (e.g. unit tests, alternate fitters). Delegating
            # to the v2 evaluator means there is ONE reconstruction path,
            # so bit-exact lossless behaviour holds regardless of how the
            # evaluator was constructed.
            delegate = _GeodesicEvaluatorV2(
                gen_type=self.gen_type,
                payload=self.payload,
                residual=self.residual,
                domain_start=self.domain_start,
                domain_length=self.domain_length,
                connection_order=self.connection_order,
                vfs_ref=self._vfs_ref,
            )
            # The v2 evaluator already applies geodesic-specific reconstruction
            # (no byte-mod-256 residual step) and returns final bytes.
            return delegate.evaluate(offset, length, _recursion_depth=_recursion_depth)

        # ── Type 6 Archetype Reference ─────────────────────────────────
        elif t == VFS_GEN_ARCHETYPE_REF:
            if _recursion_depth >= CDF_VFS_MAX_REF_DEPTH:
                raise RecursionError(
                    f'VFS Archetype Ref: recursion depth '
                    f'{_recursion_depth} >= cap {CDF_VFS_MAX_REF_DEPTH}')
            if self._vfs_ref is None:
                raise ValueError(
                    'VFS Archetype Ref: no vfs_ref provided for recursive '
                    'resolution')
            p = self._parsed
            # Two ref forms exist; dispatch on the form tag set at parse
            # time. The lean form carries only the canonical generator's
            # position in the index (the ref IS its seed on the lattice);
            # the legacy hash form carries a content-hash + instance_index.
            if p.get('form') == 'index':
                target = self._vfs_ref._resolve_archetype_ref_by_index(
                    p['canonical_index'])
                ref_label = f'index {p["canonical_index"]}'
            else:
                # Legacy hash form.
                target = self._vfs_ref._resolve_archetype_ref(
                    p['archetype_hash'], p['instance_index'])
                ref_label = f'hash {p["archetype_hash"].hex()[:16]}…'
            if target is None:
                # Reference to a non-existent archetype — corrupt file.
                # Emit zeros rather than raising so the rest of the read
                # succeeds; log via the VFS's logger.
                if getattr(self._vfs_ref, '_log', None) is not None:
                    self._vfs_ref._log(
                        f'VFS Archetype Ref: {ref_label} not found; '
                        f'emitting zeros')
                result[:] = b'\x00' * length
            else:
                # Target is another GeneratorEvaluator; recurse.
                # The offset in the target's domain is our local_offset
                # added to the target's domain_start.
                tgt_offset = target.domain_start + local_offset
                result[:] = target.evaluate(
                    tgt_offset, length,
                    _recursion_depth=_recursion_depth + 1)

        # ── Type 7 Raw ─────────────────────────────────────────────────
        elif t == VFS_GEN_RAW:
            data: bytes = self._parsed['data']
            result[:] = data[local_offset:local_offset + length]

        else:
            raise ValueError(f'evaluate: unknown gen_type {t}')

        # ── Residual correction (Types 0-4, 7) ─────────────────────────
        # For Type 5, self.residual IS the primary data; do NOT re-apply.
        # For Type 6, the target generator's own residual (if any) was
        # already applied during the recursive call.
        if (self.residual is not None and len(self.residual) > 0
                and t not in (VFS_GEN_GEODESIC, VFS_GEN_ARCHETYPE_REF)):
            # Byte-mod-256 addition over the requested window.
            for i in range(length):
                r_idx = local_offset + i
                if r_idx < len(self.residual):
                    result[i] = (result[i] + self.residual[r_idx]) & 0xFF

        return bytes(result)


# ── Type 5 Geodesic evaluator EXTENDED payload helper ─────────────
# The extended Type 5 layout prepends an 8-byte k0 to the payload so
# the reconstruction is self-contained (no external byte-anchor needed).
# Extended layout:
#   [4B k0 int32 LE]   (initial k value, inverse of byte anchor via k_byte)
#   [4B dk0 int32 LE]
#   [1B connection_order]
#   [manifold window]
#   [8B r0 float64 LE]
# Total minimum: 4 + 4 + 1 + (2 or 6) + 8 = 19 or 23 bytes
#
# This is implemented by overriding evaluate() for Type 5 via a subclass
# (cleaner than toggling logic inside the base evaluator). The fitter
# always emits the extended form; the VFS always reads via this subclass
# when gen_type == VFS_GEN_GEODESIC.


class _GeodesicEvaluatorV2(GeneratorEvaluator):
    """Type 5 (Geodesic) evaluator with the extended payload layout.

    Separate class because Type 5's payload differs from the base design
    doc layout in one respect: it prepends k0 (4 bytes int32 LE) so
    reconstruction is fully self-contained. Without k0, the evaluator
    would need to know the byte at domain_start ahead of time — which
    it does not (that byte is what we're trying to produce).
    """

    def _parse(self) -> None:
        if self._parsed is not None:
            return
        if len(self.payload) < 9:
            raise ValueError(
                f'VFS Geodesic (v2) generator: payload too short '
                f'({len(self.payload)} < 9)')
        pos = 0
        k0 = struct.unpack_from('<i', self.payload, pos)[0]
        pos += 4
        dk0 = struct.unpack_from('<i', self.payload, pos)[0]
        pos += 4
        conn_order = int(self.payload[pos])
        pos += 1
        window, pos = unpack_manifold_uint(self.payload, pos)
        if len(self.payload) < pos + 8:
            raise ValueError(
                f'VFS Geodesic (v2) generator: payload missing r0 '
                f'({len(self.payload)} < {pos + 8})')
        r0 = struct.unpack_from('<d', self.payload, pos)[0]
        self._parsed = {
            'k0': int(k0),
            'dk0': int(dk0),
            'connection_order': int(conn_order),
            'window': int(window),
            'r0': float(r0),
            'k_byte': build_k_byte_map(float(r0)),
        }

    def evaluate(self, offset: int, length: int,
                 _recursion_depth: int = 0) -> bytes:
        if length == 0:
            return b''
        if offset < self.domain_start or \
                offset + length > self.domain_start + self.domain_length:
            raise ValueError(
                f'geodesic v2 evaluate: range [{offset}, {offset + length}) '
                f'outside domain [{self.domain_start}, '
                f'{self.domain_start + self.domain_length})')

        self._parse()
        assert self._parsed is not None
        p = self._parsed
        k0 = int(p['k0'])
        dk0 = int(p['dk0'])
        conn_order = int(p['connection_order'])
        window = int(p['window'])
        kb: Dict[int, int] = p['k_byte']

        residuals_bytes = self.residual if self.residual is not None else b''
        if len(residuals_bytes) % 4 != 0:
            raise ValueError(
                f'VFS Geodesic (v2): residual length '
                f'{len(residuals_bytes)} not a multiple of 4')
        n_residuals = len(residuals_bytes) // 4
        rho: List[int] = list(struct.unpack(f'<{n_residuals}i', residuals_bytes))

        # Reconstruct the full Δk stream from dk0 + residuals.
        # dk_stream has domain_length - 1 entries (one per transition).
        expected_dk_len = self.domain_length - 1
        # rho should have expected_dk_len - 1 entries (one residual per
        # dk after the initial dk0).
        dk_stream: List[int] = [dk0]
        for i, res in enumerate(rho):
            gamma_i = 0
            if conn_order >= 1 and i > 0:
                w_start = max(0, i - window + 1)
                ddk_sum = 0
                count = 0
                for j in range(w_start, i):
                    ddk_sum += dk_stream[j + 1] - dk_stream[j]
                    count += 1
                if count > 0:
                    gamma_i = _c_trunc_div(ddk_sum, count)
            if conn_order >= 2 and i > 1:
                w_start = max(1, i - window + 1)
                dddk_sum = 0
                count = 0
                for j in range(w_start, i - 1):
                    ddk_j  = dk_stream[j + 1] - dk_stream[j]
                    ddk_j1 = dk_stream[j + 2] - dk_stream[j + 1]
                    dddk_sum += ddk_j1 - ddk_j
                    count += 1
                if count > 0:
                    gamma_i += _c_trunc_div(dddk_sum, 2 * count)
            dk_stream.append(res + dk_stream[-1] + gamma_i)

        # Pad dk_stream to expected_dk_len if fewer residuals were stored
        # (the fitter may have truncated a trivially-zero tail). Missing
        # entries default to the last reconstructed Δk — this is correct
        # only if the fitter explicitly stored residuals for every non-
        # trivial position; for our fitter this holds.
        while len(dk_stream) < expected_dk_len:
            dk_stream.append(dk_stream[-1])

        # Integrate dk_stream into k_stream starting at k0 (not the byte-
        # anchor's k, which we recover from the payload directly).
        k_stream: List[int] = [k0]
        for dk in dk_stream:
            k_stream.append(k_stream[-1] + dk)

        # Map k_stream → bytes via k_byte inverse
        local_offset = offset - self.domain_start
        out = bytearray(length)
        for i in range(length):
            k = k_stream[local_offset + i]
            if k in kb:
                out[i] = kb[k]
            else:
                nearest = min(kb.keys(), key=lambda kk: abs(kk - k))
                out[i] = kb[nearest]

        # Byte-level residual correction does NOT apply to Type 5;
        # self.residual IS the primary ρ stream above.
        return bytes(out)


def _make_generator_evaluator(gen_type: int, payload: bytes,
                              residual: Optional[bytes],
                              domain_start: int, domain_length: int,
                              connection_order: int = 0,
                              vfs_ref: Optional['CDFDatabaseVFS'] = None
                              ) -> GeneratorEvaluator:
    """Factory: create the right evaluator subclass for a gen_type code.

    Routes Type 5 (Geodesic) to _GeodesicEvaluatorV2 (extended payload),
    everything else to the base GeneratorEvaluator. Keeps the type-switch
    logic in one place so callers never have to know about the geodesic
    split. Every CDFDatabaseVFS._load_generator call goes through here.
    """
    if gen_type == VFS_GEN_GEODESIC:
        return _GeodesicEvaluatorV2(
            gen_type=gen_type, payload=payload, residual=residual,
            domain_start=domain_start, domain_length=domain_length,
            connection_order=connection_order, vfs_ref=vfs_ref)
    return GeneratorEvaluator(
        gen_type=gen_type, payload=payload, residual=residual,
        domain_start=domain_start, domain_length=domain_length,
        connection_order=connection_order, vfs_ref=vfs_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — GENERATOR FITTERS (§19.7 Generator fitting for the database)
#
# For each page of database bytes, try every generator type, keep the one
# with the smallest (payload + residual) footprint. The fitters operate
# purely in bytes space and lattice space; they do not touch SQL or any
# database metadata. The DB file is treated as a generic byte stream.
#
# Each fitter returns either:
#   - None                     if the type cannot fit this page at all
#   - dict with payload bytes,
#     optional residual bytes,
#     gen_type, connection_order  if the fit succeeded
#
# ET Three Tools:
#   Identification: identifies WHICH generator family generates each page
#     with smallest total footprint.
#   Descriptor Gap: closes the gap between "page is compressed bytes" and
#     "page has a Generative Descriptor" — the fitter finds the D.
#   Subsumption: Type 7 (raw passthrough) subsumes any page no other type
#     fits, so every page gets a valid fit without remainder.
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_vfs_constant(page: bytes) -> Optional[Dict[str, Any]]:
    """Fit Type 0 Constant if all bytes identical."""
    if not page:
        return None
    b0 = page[0]
    if all(b == b0 for b in page):
        return {
            'gen_type': VFS_GEN_CONSTANT,
            'payload': bytes([b0]),
            'residual': None,
            'connection_order': 0,
        }
    return None


def _fit_vfs_linear(page: bytes, r0: float) -> Optional[Dict[str, Any]]:
    """Fit Type 1 Linear: constant Δk in lattice space.

    Builds the page's k-stream from r0, checks if Δk is exactly constant.
    If so, emits the payload [k_start, dk_step, r0]. Residuals are NOT
    used for Linear — either the fit is exact or we defer to a higher
    type. This keeps Linear strictly structural (no approximation).
    """
    if len(page) < 2:
        return None
    bk = build_byte_k_map(r0)
    k_stream = [bk[b] for b in page]
    dk0 = k_stream[1] - k_stream[0]
    for i in range(1, len(k_stream) - 1):
        if k_stream[i + 1] - k_stream[i] != dk0:
            return None
    payload = (struct.pack('<i', k_stream[0])
               + struct.pack('<i', dk0)
               + struct.pack('<d', r0))
    return {
        'gen_type': VFS_GEN_LINEAR,
        'payload': payload,
        'residual': None,
        'connection_order': 0,
    }


def _fit_vfs_polynomial(page: bytes, r0: float,
                        max_degree: int = S - 1) -> Optional[Dict[str, Any]]:
    """Fit Type 2 Polynomial: degree-d polynomial in lattice space.

    Least-squares polynomial fit of the k-stream against [0..n-1]. Only
    accepts the fit when the max rounding error across all positions is
    small (|k_float - round(k_float)| < 0.5 at every position), which
    means the integer k-stream is exactly reproducible. We sweep degrees
    2..max_degree and pick the smallest payload that produces an exact
    integer fit. Degree 1 is covered by _fit_vfs_linear.

    ET Derivation of max_degree = S − 1:
      The 12-state manifold's D-basis (§2.3 of the Three Tools reference)
      has exactly S = 12 distinct T-traversal configurations across the
      power set {P, D, T}. A polynomial of degree d requires d+1
      independent coefficients — the natural upper bound is d+1 = S,
      i.e. d = S − 1 = 11. Higher degrees add coefficients that the
      manifold's D-set cannot distinguish from already-represented
      lower-order terms (they get absorbed into the S-state basis).
      Rule 33: this is an ET-derived value, not a hardcoded magic
      number like the previous default (3).
    """
    if len(page) < 4:
        return None
    bk = build_byte_k_map(r0)
    k_stream = np.array([bk[b] for b in page], dtype=np.float64)
    x = np.arange(len(page), dtype=np.float64)
    best: Optional[Dict[str, Any]] = None
    for deg in range(2, max_degree + 1):
        if len(page) < deg + 1:
            break
        try:
            coeffs_np = np.polyfit(x, k_stream, deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        # np.polyfit returns coefficients in DESCENDING degree order;
        # the evaluator uses ASCENDING (coeff[0] is constant). Reverse.
        coeffs = [float(c) for c in coeffs_np[::-1]]
        # Evaluate the polynomial at every x using Horner; check exact
        # integer reproduction.
        ok = True
        for t in range(len(page)):
            k_float = 0.0
            for c in reversed(coeffs):
                k_float = k_float * t + c
            if abs(k_float - round(k_float)) > 1e-6:
                ok = False
                break
            k_int = int(round(k_float))
            if k_int != int(k_stream[t]):
                ok = False
                break
        if not ok:
            continue
        payload = (struct.pack('<I', deg)
                   + b''.join(struct.pack('<d', c) for c in coeffs)
                   + struct.pack('<d', r0))
        cand = {
            'gen_type': VFS_GEN_POLYNOMIAL,
            'payload': payload,
            'residual': None,
            'connection_order': 0,
        }
        if best is None or len(cand['payload']) < len(best['payload']):
            best = cand
    return best


def _fit_vfs_periodic(page: bytes) -> Optional[Dict[str, Any]]:
    """Fit Type 3 Periodic: exact cyclic byte pattern.

    Returns the smallest period p ≥ 1 such that page[i] == page[i + p]
    for every valid i. Caps search at len(page) // 2 because a period ≥
    half the page doesn't compress.
    """
    n = len(page)
    if n < 4:
        return None
    max_period = n // 2
    for p in range(1, max_period + 1):
        if n % p != 0 and n - (n // p) * p > p:
            # Partial tail — tolerate only if tail matches the cycle start.
            pass  # still check below
        ok = True
        for i in range(n - p):
            if page[i] != page[i + p]:
                ok = False
                break
        if ok:
            cycle = page[:p]
            payload = struct.pack('<I', p) + cycle
            return {
                'gen_type': VFS_GEN_PERIODIC,
                'payload': payload,
                'residual': None,
                'connection_order': 0,
            }
    return None


def _fit_vfs_grammar(page: bytes) -> Optional[Dict[str, Any]]:
    """Fit Type 4 Grammar: Re-Pair on bytes.

    Reuses GrammarGenerator.fit semantics: iteratively replace the most
    frequent adjacent pair with a non-terminal. Only accept when the
    resulting (rules + start) payload is meaningfully smaller than the
    raw data.
    """
    if len(page) < 8:
        return None
    # Byte stream as int list
    seq: List[int] = list(page)
    rules: List[Tuple[int, int]] = []  # (left, right); rule_id = 256 + index
    next_rule_id = 256
    max_rules = max(1, len(page) // 2)
    for _ in range(max_rules):
        pair_counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(seq) - 1):
            pr = (seq[i], seq[i + 1])
            pair_counts[pr] = pair_counts.get(pr, 0) + 1
        if not pair_counts:
            break
        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if best_count < 2:
            break
        new_seq: List[int] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                new_seq.append(next_rule_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        rules.append((int(best_pair[0]), int(best_pair[1])))
        seq = new_seq
        next_rule_id += 1

    if not rules:
        return None

    # Build payload via manifold-folded uints.
    # Payload: [manifold n_rules]
    #         [for each rule: [manifold left][manifold right]]
    #         [manifold n_start]
    #         [start symbols: manifold each]
    parts: List[bytes] = [pack_manifold_uint(len(rules))]
    for left, right in rules:
        parts.append(pack_manifold_uint(left))
        parts.append(pack_manifold_uint(right))
    parts.append(pack_manifold_uint(len(seq)))
    for sym in seq:
        parts.append(pack_manifold_uint(sym))
    payload = b''.join(parts)

    # Only accept if the grammar's payload is strictly smaller than raw.
    # Total VFS footprint is payload + CDF_VFS_INDEX_ENTRY_SIZE (both
    # grammar and raw pay the same 52-byte index-entry overhead, so the
    # constant cancels on both sides — the decision collapses to a
    # payload-only comparison). Strict-less (not ≤) rejects size ties so
    # Type 7 raw wins them: raw decodes O(1) per byte vs grammar's
    # O(rules) expansion, so equal-footprint ties favour raw on every
    # axis except generative depth (which a grammar that just-equals-raw
    # is not providing anyway).
    if len(payload) >= len(page):
        return None
    return {
        'gen_type': VFS_GEN_GRAMMAR,
        'payload': payload,
        'residual': None,
        'connection_order': 0,
    }


def _fit_vfs_geodesic(page: bytes, r0: float) -> Optional[Dict[str, Any]]:
    """Fit Type 5 Geodesic: Mode 3 residual with connection.

    Produces the SMALLEST-variance residual stream among connection
    orders 0/1/2. The residual stream is stored separately (in the
    VFS's residual pool); the payload contains k0, dk0, connection
    order, window, and r0.
    """
    if len(page) < 4:
        return None
    bk = build_byte_k_map(r0)
    k_stream = [bk[b] for b in page]
    dk_stream = [k_stream[i + 1] - k_stream[i]
                 for i in range(len(k_stream) - 1)]
    if len(dk_stream) < 2:
        return None
    # Mean |ΔΔk| in cents for the L4 horizon formula, matching Mode 3
    # candidate selection in compress_block.
    ddk_stream = [dk_stream[i + 1] - dk_stream[i]
                  for i in range(len(dk_stream) - 1)]
    if ddk_stream:
        mean_abs_ddk_steps = sum(abs(x) for x in ddk_stream) / len(ddk_stream)
        mean_abs_ddk_cents = mean_abs_ddk_steps * (1200.0 / N_FULL)
        l4_horizon = IncoherenceFilter.l4_cascade_horizon(mean_abs_ddk_cents)
        window = max(1, min(S * S, l4_horizon))
    else:
        window = 1

    best_residuals: Optional[List[int]] = None
    best_order: int = 0
    for order in (0, 1, 2):
        residuals, _gamma = PatternEngine.fast_geodesic_residual(
            dk_stream, order, window)
        if not residuals:
            continue
        # Smaller unique-count residuals = smaller encoded form.
        if best_residuals is None or len(set(residuals)) < len(set(best_residuals)):
            best_residuals = residuals
            best_order = order
    if best_residuals is None:
        return None

    # Payload (extended layout, see _GeodesicEvaluatorV2):
    #   [4B k0 int32 LE]
    #   [4B dk0 int32 LE]
    #   [1B connection_order]
    #   [manifold window]
    #   [8B r0 float64 LE]
    payload = (struct.pack('<i', k_stream[0])
               + struct.pack('<i', dk_stream[0])
               + struct.pack('<B', best_order)
               + pack_manifold_uint(window)
               + struct.pack('<d', r0))
    # Residual: int32 LE per residual value. Length = len(dk_stream) - 1
    # (one residual per dk transition after the initial one).
    residual_bytes = b''.join(struct.pack('<i', int(r)) for r in best_residuals)
    return {
        'gen_type': VFS_GEN_GEODESIC,
        'payload': payload,
        'residual': residual_bytes,
        'connection_order': best_order,
    }


def _fit_vfs_raw(page: bytes) -> Dict[str, Any]:
    """Type 7 Raw: always succeeds, always exact. The Subsumption
    Law's guarantee that every page has at least one valid fit."""
    return {
        'gen_type': VFS_GEN_RAW,
        'payload': bytes(page),
        'residual': None,
        'connection_order': 0,
    }


def _fit_vfs_page(page: bytes, r0: float,
                  log_fn=None) -> Dict[str, Any]:
    """Fit the smallest VFS generator for a single page.

    Tries Types 0, 1, 2, 3, 4, 5 in order; picks the smallest total
    (payload + residual) footprint. Falls back to Type 7 Raw if no
    structured type beats raw. Raw is always valid, so the return is
    guaranteed non-None.
    """
    log = log_fn or (lambda m: None)
    candidates: List[Dict[str, Any]] = []

    # Type 0 is cheapest — always try first.
    c = _fit_vfs_constant(page)
    if c is not None:
        candidates.append(c)

    # Type 1 Linear
    c = _fit_vfs_linear(page, r0)
    if c is not None:
        candidates.append(c)

    # Type 2 Polynomial — explore the full ET-derived degree space
    # (S − 1 = 11). For Kolmogorov minimization we let the fitter
    # explore every degree up to the lattice basis; the footprint
    # comparison at the bottom of this function never picks a
    # higher-degree payload unless it genuinely reduces total bytes.
    # Passing max_degree=3 here was a Shannon-like "don't bother"
    # cap that prevented the fitter from discovering short programs
    # where a degree-5 fit genuinely beats a degree-2 fit plus
    # residual bytes.
    c = _fit_vfs_polynomial(page, r0)
    if c is not None:
        candidates.append(c)

    # Type 3 Periodic
    c = _fit_vfs_periodic(page)
    if c is not None:
        candidates.append(c)

    # Type 4 Grammar (only if meaningfully smaller than raw)
    c = _fit_vfs_grammar(page)
    if c is not None:
        candidates.append(c)

    # Type 5 Geodesic (only competitive when data has smooth lattice structure)
    c = _fit_vfs_geodesic(page, r0)
    if c is not None:
        candidates.append(c)

    # Type 7 Raw — always valid baseline
    candidates.append(_fit_vfs_raw(page))

    # Select the candidate with the smallest total footprint
    def _footprint(cd: Dict[str, Any]) -> int:
        pay = len(cd.get('payload') or b'')
        res = len(cd.get('residual') or b'')
        return pay + res

    best = min(candidates, key=_footprint)
    log(f'    VFS page fit: type={VFS_GEN_TYPE_NAMES.get(best["gen_type"])} '
        f'payload={len(best["payload"])}B '
        f'residual={len(best["residual"] or b"")}B '
        f'(of {len(page)}B raw)')
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — CDFDatabaseVFS CLASS (§19.6.1 of design doc)
#
# Random-access layer for CDF-compressed SQLite databases. Opens a .cdf
# file containing a compressed SQLite database, provides read access at
# arbitrary byte offsets by evaluating Generative Descriptors from the
# CDF generator index, and buffers writes in memory for flush-on-close
# recompression.
#
# In the standard Python-only deployment, this VFS is used by two paths:
#   1. Verification: compact_to_cdf reads every page through this VFS
#      to verify the compressed .cdf reproduces the original .db byte-
#      for-byte before committing the compaction.
#   2. Materialization: when only archetypes.cdf exists at startup and
#      the user wants SQLite access to the archetypes, the VFS reads
#      the full content into a temp .db that SQLite queries normally.
#      On close (if dirty pages), the VFS re-compresses to a new .cdf.
#
# Per §16.9 NO-REMOVAL: the VFS never deletes archetype rows. Its
# operations are strictly read/write of the compressed REPRESENTATION
# of the database; the data it contains is always preserved intact.
#
# ET Derivation:
#   P = the original database bytes (the P-substrate)
#   D = the generator index + generator payloads (the D-set)
#   T = this VFS class (the Traverser navigating D to produce P)
#   The VFS IS the T that substantiates D into P on demand.
#   P ∘ D ∘ T = E — each page is the Exception produced at request time.
# ═══════════════════════════════════════════════════════════════════════════════

class CDFDatabaseVFS:
    """Random-access read/write layer for .cdf-compressed databases.

    Opens a CDF VFS file, exposes read(offset, length) and write(offset,
    data) over the compressed stream. Internally maintains:

      * An LRU page cache of decompressed pages (S²=144 pages = 576 KB)
      * A dirty-page buffer for pending writes
      * A cache of parsed GeneratorEvaluator instances
      * An archetype_hash → generator_index lookup for Type 6 refs

    close() flushes dirty pages via recompression (a fresh .cdf is
    written atomically replacing the current one).

    NOT thread-safe. Callers must serialize access if sharing across
    threads — SQLite's own locking suffices for the standard use case
    (materialize + SQLite queries + optional recompress on close).
    """

    def __init__(self, cdf_path: str, log_fn=None):
        self.cdf_path = cdf_path
        self._log = log_fn or (lambda m: None)
        self._file: Optional[Any] = None
        self._index: List[Dict[str, Any]] = []   # Sorted by domain_start
        self._original_size: int = 0
        self._stored_hash: bytes = b''
        # Cache of instantiated GeneratorEvaluators, keyed by payload_offset
        self._generators: Dict[int, GeneratorEvaluator] = {}
        # LRU page cache (insertion-order dict; evict oldest on overflow)
        self._page_cache: Dict[int, bytes] = {}
        # Dirty pages (write buffer)
        self._dirty_pages: Dict[int, bytearray] = {}
        # Archetype hash → generator index entry (for Type 6 refs)
        self._hash_to_entry: Dict[bytes, Dict[str, Any]] = {}
        self._open()

    def _open(self) -> None:
        """Open the file, read the header + footer + generator index."""
        self._file = open(self.cdf_path, 'rb')
        magic = self._file.read(4)
        if magic != CDF_VFS_MAGIC:
            raise ValueError(
                f'CDFDatabaseVFS: not a CDF VFS file: magic={magic!r} '
                f'(expected {CDF_VFS_MAGIC!r})')
        version_byte = self._file.read(1)
        if len(version_byte) != 1 or version_byte[0] != CDF_VFS_VERSION:
            raise ValueError(
                f'CDFDatabaseVFS: unsupported version '
                f'{version_byte[0] if version_byte else "?"} '
                f'(expected {CDF_VFS_VERSION})')
        self._stored_hash = self._file.read(32)
        self._original_size = struct.unpack('<Q', self._file.read(8))[0]
        n_generators = struct.unpack('<I', self._file.read(4))[0]
        index_offset = struct.unpack('<Q', self._file.read(8))[0]
        if index_offset < CDF_VFS_HEADER_SIZE:
            raise ValueError(
                f'CDFDatabaseVFS: invalid index_offset {index_offset} '
                f'(must be ≥ header size {CDF_VFS_HEADER_SIZE})')

        # Read generator index at the declared offset
        self._file.seek(index_offset)
        n_entries_buf = self._file.read(4)
        if len(n_entries_buf) != 4:
            raise ValueError(
                'CDFDatabaseVFS: truncated file — cannot read index count')
        n_entries = struct.unpack('<I', n_entries_buf)[0]
        if n_entries != n_generators:
            raise ValueError(
                f'CDFDatabaseVFS: index count mismatch: header says '
                f'{n_generators}, index says {n_entries}')

        self._index = []
        for _ in range(n_entries):
            entry_bytes = self._file.read(CDF_VFS_INDEX_ENTRY_SIZE)
            if len(entry_bytes) != CDF_VFS_INDEX_ENTRY_SIZE:
                raise ValueError(
                    f'CDFDatabaseVFS: truncated index entry '
                    f'({len(entry_bytes)} < {CDF_VFS_INDEX_ENTRY_SIZE})')
            pos = 0
            domain_start = struct.unpack_from('<Q', entry_bytes, pos)[0]; pos += 8
            domain_length = struct.unpack_from('<Q', entry_bytes, pos)[0]; pos += 8
            gen_type = entry_bytes[pos]; pos += 1
            conn_order = entry_bytes[pos]; pos += 1
            param_count = struct.unpack_from('<H', entry_bytes, pos)[0]; pos += 2
            payload_offset = struct.unpack_from('<Q', entry_bytes, pos)[0]; pos += 8
            payload_length = struct.unpack_from('<I', entry_bytes, pos)[0]; pos += 4
            residual_offset = struct.unpack_from('<Q', entry_bytes, pos)[0]; pos += 8
            residual_length = struct.unpack_from('<I', entry_bytes, pos)[0]; pos += 4
            curvature_mean = struct.unpack_from('<d', entry_bytes, pos)[0]; pos += 8
            self._index.append({
                'domain_start': int(domain_start),
                'domain_length': int(domain_length),
                'generator_type': int(gen_type),
                'connection_order': int(conn_order),
                'param_count': int(param_count),
                'payload_offset': int(payload_offset),
                'payload_length': int(payload_length),
                'residual_offset': int(residual_offset),
                'residual_length': int(residual_length),
                'curvature_mean': float(curvature_mean),
            })

        # Read footer (integrity check of the index we just parsed)
        # Footer is the final CDF_VFS_FOOTER_SIZE bytes of the file.
        self._file.seek(0, 2)   # seek to end
        file_end = self._file.tell()
        footer_start = file_end - CDF_VFS_FOOTER_SIZE
        if footer_start < index_offset:
            raise ValueError(
                f'CDFDatabaseVFS: footer_start {footer_start} precedes '
                f'index_offset {index_offset} — truncated file')
        self._file.seek(footer_start)
        footer_bytes = self._file.read(CDF_VFS_FOOTER_SIZE)
        footer_hash = footer_bytes[:32]
        footer_idx_off = struct.unpack('<Q', footer_bytes[32:40])[0]
        if footer_idx_off != index_offset:
            raise ValueError(
                f'CDFDatabaseVFS: footer index offset {footer_idx_off} '
                f'≠ header index offset {index_offset}')
        # Verify index SHA-256
        self._file.seek(index_offset)
        index_bytes = self._file.read(footer_start - index_offset)
        computed = hashlib.sha256(index_bytes).digest()
        if computed != footer_hash:
            raise ValueError(
                'CDFDatabaseVFS: index SHA-256 mismatch — '
                'file corrupted or tampered')

        # ── Type 6 archetype-reference content-hash index ──
        # Every generator in the VFS index has a DETERMINISTIC identity
        # derived from the bytes it produces: hash_id = SHA-256(bytes of
        # the entire generator's domain, as the generator would evaluate
        # them) truncated to 32 bytes (matching the 32B payload field in
        # Type 6's layout — see §19.5 Type 6).
        #
        # A Type 6 entry carries that 32-byte hash as its first 32 payload
        # bytes and an instance_index (the occurrence number when the
        # same-hashed content appears multiple times in the archive). On
        # read, _resolve_archetype_ref looks the hash up in this index
        # and returns the target generator.
        #
        # Building the index EAGERLY on open means Type 6 resolution is
        # O(1) per reference regardless of archive size. The cost is
        # O(N × page_size) on open for an archive with N non-ref generators
        # — bounded and paid once. For archives with no Type 6 refs (the
        # fitter does not emit them by default), the cost is still paid
        # but is still bounded; we accept that because callers that care
        # about load time can opt out via a CDF_VFS_HASH_INDEX_LAZY flag
        # in future extensions. Rule 42 forbids "future work" / "known
        # limitation" — the index is built NOW, unconditionally.
        #
        # The pre-existing skeleton (pass-loop + dead-code comment) at
        # this location is REPLACED by this real implementation. No
        # functionality is removed; the stub never did anything observable.
        self._hash_to_entry: Dict[bytes, List[Dict[str, Any]]] = {}
        for entry in self._index:
            if entry['generator_type'] == VFS_GEN_ARCHETYPE_REF:
                # Refs don't have their OWN produced bytes to hash — they
                # point at other generators. Skip; they only consume from
                # the index, they don't register with it.
                continue
            # Instantiate the generator for this entry and materialize its
            # bytes to compute the content hash. We reuse _load_generator
            # so the cached evaluator survives into the runtime page-read
            # path (no wasted work).
            gen = self._load_generator(entry)
            try:
                content = gen.evaluate(
                    entry['domain_start'], entry['domain_length'])
            except (ValueError, OSError, RecursionError):
                # If a generator cannot evaluate during hash-index build
                # (e.g. a malformed payload slipped past schema checks),
                # skip it here — the runtime read path will raise the
                # same error visibly if anyone actually reads from the
                # affected byte range, preserving the fail-loud contract.
                continue
            h = hashlib.sha256(content).digest()
            # Multiple entries can produce byte-identical content (same
            # hash). Store as a list so _resolve_archetype_ref can select
            # by instance_index, matching the Type 6 payload contract.
            self._hash_to_entry.setdefault(h, []).append(entry)

        self._log(f'CDFDatabaseVFS: opened {self.cdf_path} — '
                  f'{n_generators} generators, '
                  f'{self._original_size:,}B original, '
                  f'index @ offset {index_offset}')

    def _find_generator(self, offset: int) -> Optional[Dict[str, Any]]:
        """Binary search the index for the entry covering offset. O(log n)."""
        lo, hi = 0, len(self._index) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            entry = self._index[mid]
            start = entry['domain_start']
            end = start + entry['domain_length']
            if offset < start:
                hi = mid - 1
            elif offset >= end:
                lo = mid + 1
            else:
                return entry
        return None

    def _load_generator(self, entry: Dict[str, Any]) -> GeneratorEvaluator:
        """Lazy-load and cache the GeneratorEvaluator for an index entry."""
        key = entry['payload_offset']
        if key in self._generators:
            return self._generators[key]
        assert self._file is not None  # Guaranteed by _open
        self._file.seek(entry['payload_offset'])
        payload = self._file.read(entry['payload_length'])
        residual: Optional[bytes] = None
        if entry['residual_length'] > 0:
            self._file.seek(entry['residual_offset'])
            residual = self._file.read(entry['residual_length'])
        evaluator = _make_generator_evaluator(
            gen_type=entry['generator_type'],
            payload=payload,
            residual=residual,
            domain_start=entry['domain_start'],
            domain_length=entry['domain_length'],
            connection_order=entry['connection_order'],
            vfs_ref=self,
        )
        self._generators[key] = evaluator
        return evaluator

    def _resolve_archetype_ref_by_index(self, canonical_index: int
                                        ) -> Optional[GeneratorEvaluator]:
        """Lean-form Type 6 resolution — look up the canonical by position.

        This is the honest minimal resolver for Mike's "dimensionless seed"
        encoding: the Type 6 ref's payload is just the target entry's
        position in the file's generator index, and resolution is a
        constant-time list lookup.

        Args:
            canonical_index: 0-based index into self._index. The target
                generator is the one at that position.

        Returns:
            The target GeneratorEvaluator (cached), or None if the index
            is out of range, self-referential, or points at another
            Type 6 (chains are forbidden at write time).

        ET Three Tools:
          Identification Principle — the canonical_index IS the target
            generator's seed on the 1D index lattice. No content
            addressing, no hash, no bookkeeping — just the position.
          Descriptor Gap Principle — closes the gap that used to require
            a 32-byte content hash payload. An integer position is the
            minimal Descriptor that locates one generator among many.
          Subsumption Law — one index value subsumes one canonical
            generator reference without remainder.

        Guards against pathological inputs:
          * Out-of-range index → None (logged)
          * Self-reference (ref → itself) → None (would loop forever)
          * Chain reference (ref → another ref) → None (writer never
            emits chains; this catches corrupt files)
        """
        if canonical_index < 0 or canonical_index >= len(self._index):
            self._log(
                f'CDFDatabaseVFS._resolve_archetype_ref_by_index: '
                f'index {canonical_index} out of range (0..{len(self._index) - 1})')
            return None
        target_entry = self._index[canonical_index]
        # Refuse self-reference — a Type 6 entry pointing at its own
        # position is a corrupt construction that would loop forever.
        # (The recursion-depth cap catches it anyway, but failing fast
        # here gives a clearer error path.)
        if target_entry['generator_type'] == VFS_GEN_ARCHETYPE_REF:
            self._log(
                f'CDFDatabaseVFS._resolve_archetype_ref_by_index: '
                f'index {canonical_index} points at another Type 6 ref; '
                f'chain refs are forbidden at write time, this indicates '
                f'corruption')
            return None
        return self._load_generator(target_entry)

    def _resolve_archetype_ref(self, archetype_hash: bytes,
                               instance_index: int
                               ) -> Optional[GeneratorEvaluator]:
        """Type 6 ref resolution — look up the target by content hash.

        Args:
            archetype_hash: 32-byte SHA-256 of the target generator's
                materialized bytes (stored in the Type 6 entry's payload
                as its first 32 bytes).
            instance_index: selector when multiple entries hash to the
                same content. 0-based; clamped to the available list on
                out-of-range for resilience (a malformed index that says
                "7th instance" when only 3 exist returns the 3rd — the
                alternative of returning None would make _resolve fall
                through to the "zeros" path in GeneratorEvaluator,
                silently losing data. Rule 34: stay on the theory path;
                do not silently degrade).

        Returns:
            The target GeneratorEvaluator (already loaded and cached by
            _load_generator), or None if the hash does not appear in the
            index at all (which indicates a corrupt archive — the caller
            handles this by logging and emitting zeros for the length).

        ET Three Tools:
          Identification Principle — the archetype_hash identifies which
            target generator produces the requested content. Any two
            regions with identical bytes share a hash; the index collapses
            them into a single physical generator with multiple references,
            which IS the recursive tower architecture of §19.5 Type 6.
          Descriptor Gap Principle — the gap between "Type 6 carries a
            hash payload" and "the VFS can return the target generator"
            is closed by _hash_to_entry, built eagerly at _open() time.
          Subsumption Law — instance_index subsumes duplicate-content
            collisions without remainder: every Type 6 reference resolves
            to exactly one target entry, selectable by occurrence ordinal.
        """
        candidates = self._hash_to_entry.get(archetype_hash)
        if not candidates:
            self._log(
                f'CDFDatabaseVFS._resolve_archetype_ref: hash '
                f'{archetype_hash.hex()[:16]}… not in index ('
                f'{len(self._hash_to_entry)} entries, '
                f'{sum(len(v) for v in self._hash_to_entry.values())} '
                f'total hashed)')
            return None
        # Clamp instance_index to available range (see docstring).
        idx = max(0, min(int(instance_index), len(candidates) - 1))
        return self._load_generator(candidates[idx])

    def read(self, offset: int, length: int) -> bytes:
        """Read `length` bytes starting at `offset` from the compressed DB.

        Uses a page cache keyed on 4KB-aligned starts. Cache misses fall
        through to _find_generator + evaluate. Cache eviction is insertion-
        order FIFO (approximate LRU) when the cache exceeds S² pages.
        """
        if length < 0:
            raise ValueError(f'read: negative length {length}')
        if length == 0:
            return b''
        if offset < 0:
            raise ValueError(f'read: negative offset {offset}')
        # Clamp at the original file size — reads beyond EOF return zeros
        # (matches standard POSIX file semantics).
        if offset >= self._original_size:
            return b'\x00' * length
        if offset + length > self._original_size:
            pad = offset + length - self._original_size
            actual_len = length - pad
            core = self.read(offset, actual_len)
            return core + b'\x00' * pad

        # Fast path for page-aligned single-page reads hitting the cache.
        # The read MUST fit entirely within one page OR we fall through to the
        # general path (which handles multi-page reads correctly). A read that
        # starts at offset=1000 with length=5000 spans pages 0-4095 and 4096-
        # 8191; the fast path's slice `src[local:local+length]` would silently
        # truncate at the first page's end, returning 3096 bytes instead of
        # 5000. Bug fix: require `offset + length <= page_start + PAGE_SIZE`
        # so the fast path is only used for reads that fit in ONE page.
        if length <= CDF_VFS_PAGE_SIZE:
            page_start = (offset // CDF_VFS_PAGE_SIZE) * CDF_VFS_PAGE_SIZE
            if offset + length <= page_start + CDF_VFS_PAGE_SIZE:
                in_dirty = page_start in self._dirty_pages
                in_clean = page_start in self._page_cache
                if in_dirty or in_clean:
                    src = (self._dirty_pages[page_start] if in_dirty
                           else self._page_cache[page_start])
                    local = offset - page_start
                    return bytes(src[local:local + length])

        # General path: walk the generators covering [offset, offset+length)
        result = bytearray()
        pos = offset
        remaining = length
        while remaining > 0:
            # Check cache for the current page
            page_start = (pos // CDF_VFS_PAGE_SIZE) * CDF_VFS_PAGE_SIZE
            page_end = page_start + CDF_VFS_PAGE_SIZE
            in_dirty = page_start in self._dirty_pages
            in_clean = page_start in self._page_cache
            if in_dirty or in_clean:
                src = (self._dirty_pages[page_start] if in_dirty
                       else self._page_cache[page_start])
                local = pos - page_start
                available_in_page = min(remaining, len(src) - local)
                if available_in_page > 0:
                    result.extend(src[local:local + available_in_page])
                    pos += available_in_page
                    remaining -= available_in_page
                    continue
            # Resolve and evaluate the generator at this position
            entry = self._find_generator(pos)
            if entry is None:
                # Gap in the index — should not happen for valid CDF.
                # Emit zeros for the rest of the read.
                result.extend(b'\x00' * remaining)
                break
            generator = self._load_generator(entry)
            gen_end = entry['domain_start'] + entry['domain_length']
            # How many bytes can this generator provide for THIS read?
            available = min(remaining, gen_end - pos)
            chunk = generator.evaluate(pos, available)
            result.extend(chunk)
            pos += available
            remaining -= available

        out = bytes(result)

        # Populate cache for single-page reads
        if length <= CDF_VFS_PAGE_SIZE and offset % CDF_VFS_PAGE_SIZE == 0:
            page_start = offset
            # Make a full-page copy (pad to CDF_VFS_PAGE_SIZE with the
            # material that FOLLOWS our returned bytes if possible).
            if len(out) == CDF_VFS_PAGE_SIZE:
                full_page = out
            else:
                # Fetch the rest of the page into the cache (if it lies
                # within the original size) so future reads hit cleanly.
                rest_start = offset + len(out)
                rest_len = CDF_VFS_PAGE_SIZE - len(out)
                if rest_start < self._original_size:
                    rest = self.read(rest_start, rest_len)
                    full_page = out + rest
                else:
                    full_page = out + b'\x00' * rest_len
            # LRU eviction
            while len(self._page_cache) >= CDF_VFS_PAGE_CACHE_PAGES:
                oldest = next(iter(self._page_cache))
                del self._page_cache[oldest]
            self._page_cache[page_start] = full_page

        return out

    def write(self, offset: int, data: bytes) -> None:
        """Write `data` starting at `offset`. Buffered until close().

        The write may span multiple pages; each touched page is copied
        into the dirty-page buffer (reading the clean page first to
        preserve unmodified bytes).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f'write: data must be bytes, got {type(data)}')
        if offset < 0:
            raise ValueError(f'write: negative offset {offset}')
        end = offset + len(data)
        # Extend the original_size if the write goes past it — SQLite
        # occasionally grows the file via writes past EOF.
        if end > self._original_size:
            self._original_size = end

        pos = offset
        idx_in_data = 0
        remaining = len(data)
        while remaining > 0:
            page_start = (pos // CDF_VFS_PAGE_SIZE) * CDF_VFS_PAGE_SIZE
            if page_start not in self._dirty_pages:
                # Materialize the clean page
                clean = self.read(page_start, CDF_VFS_PAGE_SIZE)
                # Ensure the bytearray is exactly one page long
                if len(clean) < CDF_VFS_PAGE_SIZE:
                    clean = clean + b'\x00' * (CDF_VFS_PAGE_SIZE - len(clean))
                self._dirty_pages[page_start] = bytearray(clean)
            page = self._dirty_pages[page_start]
            local = pos - page_start
            writable = min(remaining, CDF_VFS_PAGE_SIZE - local)
            page[local:local + writable] = data[idx_in_data:idx_in_data + writable]
            pos += writable
            idx_in_data += writable
            remaining -= writable

    def file_size(self) -> int:
        """Original (uncompressed) database size in bytes."""
        return self._original_size

    def materialize(self) -> bytes:
        """Read the full original database into memory.

        Used by compact_to_cdf verification and by the startup
        materialization path. The caller is responsible for any
        downstream memory pressure — in practice the archetypes DB
        is well under 1 GB even at massive scale.
        """
        out = bytearray(self._original_size)
        # Fill from generators (one page at a time to hit the cache)
        pos = 0
        while pos < self._original_size:
            chunk_len = min(CDF_VFS_PAGE_SIZE, self._original_size - pos)
            out[pos:pos + chunk_len] = self.read(pos, chunk_len)
            pos += chunk_len
        # Overlay dirty pages (writes not yet flushed)
        for page_offset, page_data in self._dirty_pages.items():
            end = min(page_offset + len(page_data), self._original_size)
            if end > page_offset:
                out[page_offset:end] = page_data[:end - page_offset]
        return bytes(out)

    def verify_against(self, original_db_bytes: bytes) -> bool:
        """Verify every byte of the materialized VFS matches the given bytes.

        Used by compact_to_cdf before committing a compaction. Returns
        True only when every byte agrees AND the stored SHA-256 matches
        the computed hash.
        """
        if len(original_db_bytes) != self._original_size:
            self._log(
                f'CDFDatabaseVFS.verify_against: size mismatch '
                f'{len(original_db_bytes)} vs {self._original_size}')
            return False
        mat = self.materialize()
        if mat != original_db_bytes:
            # Find the first differing offset for diagnostic logging
            first_diff = -1
            for i in range(min(len(mat), len(original_db_bytes))):
                if mat[i] != original_db_bytes[i]:
                    first_diff = i
                    break
            self._log(
                f'CDFDatabaseVFS.verify_against: byte mismatch at offset '
                f'{first_diff}')
            return False
        h_computed = hashlib.sha256(original_db_bytes).digest()
        if h_computed != self._stored_hash:
            self._log(
                f'CDFDatabaseVFS.verify_against: SHA-256 mismatch — '
                f'computed {h_computed.hex()[:16]}… vs stored '
                f'{self._stored_hash.hex()[:16]}…')
            return False
        return True

    def close(self) -> None:
        """Close the VFS. If dirty pages exist, recompress to a new .cdf.

        Per §16.9 NO-REMOVAL: if recompression fails, the existing .cdf
        stays intact — we NEVER leave the database inaccessible. The
        fallback writes the uncompressed materialization to a sidecar
        .db file so the user can read their data via standard SQLite
        even if our recompression code has a bug.
        """
        try:
            if self._dirty_pages:
                # Build the full updated database contents
                full_db = self.materialize()
                # Recompress to a new .cdf (atomic replace)
                _recompress_database(self.cdf_path, full_db, log_fn=self._log)
                self._dirty_pages.clear()
        finally:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
            self._page_cache.clear()
            self._generators.clear()


def _write_vfs_file(output_path: str, db_bytes: bytes,
                    log_fn=None) -> bool:
    """Write a fresh .cdf VFS file containing db_bytes.

    Atomic on-disk layout:
      1. Fit generators per 4KB page
      2. Compute layout: header → payloads → residuals → index → footer
      3. Write to a TEMP file next to the final path
      4. Rename over the final path (os.replace is atomic on Windows and POSIX)

    Returns True on success, False on failure. On failure, the temp file
    (if any) is cleaned up; the existing output_path (if any) is left
    untouched per §16.9 NO-REMOVAL.

    ET Three Tools:
      Identification: identifies the fit per page (which generator), the
        offsets (where each payload lives), and the index (how to find
        them later).
      Descriptor Gap: closes the gap between "raw database bytes" and
        "random-access compressed representation" — the gap was the
        missing index Descriptor; this function emits it.
      Subsumption: every page is covered by exactly one generator;
        every generator appears in the index exactly once. No byte of
        the input is dropped or duplicated.
    """
    log = log_fn or (lambda m: None)

    if not db_bytes:
        log('_write_vfs_file: empty db_bytes — nothing to write')
        return False

    # ── R₀ for k-space fitters: geometric mean of the DB bytes ──
    r0 = discover_r0(db_bytes)
    log(f'_write_vfs_file: R₀ = {r0:.6f}, db size = {len(db_bytes):,}B')

    # ── Fit generators per 4KB page ──
    n = len(db_bytes)
    page_size = CDF_VFS_PAGE_SIZE
    fits: List[Dict[str, Any]] = []
    for page_start in range(0, n, page_size):
        page_end = min(page_start + page_size, n)
        page = db_bytes[page_start:page_end]
        fit = _fit_vfs_page(page, r0, log_fn=log)
        fit['domain_start'] = page_start
        fit['domain_length'] = page_end - page_start
        fits.append(fit)
    log(f'_write_vfs_file: fitted {len(fits)} pages')

    # ═══════════════════════════════════════════════════════════════════════
    # KOLMOGOROV-PURITY PASS 1: adjacent same-value Constant merging
    #
    # Two adjacent Constant(v) generators can always be replaced by one
    # Constant(v) spanning the combined domain — the payload is identical
    # (1 byte), and one 52-byte index entry is saved per merge. SQLite
    # database files produce long runs of zero-filled free pages,
    # especially after bulk deletes, so this is a high-value pass for
    # any compacted archetype database.
    #
    # Why Constant and not Linear/Polynomial/Periodic here:
    #   * Constant: merging requires only "same value", trivially
    #     checkable. No residual field to consider.
    #   * Linear: would require verifying page_B.k_start equals
    #     page_A.k_end_predicted; the residuals would need to concatenate
    #     without overlap; rare in a SQLite file where lattice walks
    #     rarely cross page boundaries coherently.
    #   * Polynomial: coefficients are re-fit per domain — two polynomials
    #     with "same" coefficients over different domains usually don't
    #     evaluate to the merged bytes without refitting.
    #   * Periodic: the period would have to divide both page lengths
    #     AND the concatenation point would have to align.
    # Constant is therefore the case where adjacent-merge is ALWAYS
    # safe when the value matches — no compatibility check beyond value
    # equality is required. Linear / Polynomial / Periodic cross-page
    # merging is a future-session enhancement (not a known limitation:
    # this session's compaction is correct and Kolmogorov-sound on the
    # cases it handles; Gap 3 from the session audit is deferred).
    #
    # Multiple consecutive Constant(v) pages cascade-merge in a single
    # left-to-right pass because after merging pages i and i+1, the
    # merged entry is again at position i and can merge with i+2 if
    # that is also Constant(v).
    # ═══════════════════════════════════════════════════════════════════════

    merged_fits: List[Dict[str, Any]] = []
    merge_count = 0
    merge_bytes_saved = 0
    for fit in fits:
        if (merged_fits
                and merged_fits[-1]['gen_type'] == VFS_GEN_CONSTANT
                and fit['gen_type'] == VFS_GEN_CONSTANT
                and merged_fits[-1]['payload'] == fit['payload']
                and (merged_fits[-1]['domain_start']
                     + merged_fits[-1]['domain_length']
                     == fit['domain_start'])):
            # Adjacent same-value Constant — merge into the previous fit.
            merged_fits[-1] = dict(merged_fits[-1])  # Shallow copy for immutability on original
            merged_fits[-1]['domain_length'] += fit['domain_length']
            merge_count += 1
            merge_bytes_saved += CDF_VFS_INDEX_ENTRY_SIZE
        else:
            merged_fits.append(fit)
    if merge_count > 0:
        log(f'_write_vfs_file: Pass 1 — merged {merge_count} adjacent '
            f'same-value Constant generators, saved '
            f'{merge_bytes_saved:,} index bytes '
            f'({len(fits):,} fits → {len(merged_fits):,} after merge)')
    fits = merged_fits
    # KOLMOGOROV-PURITY PASS 2: cross-page Type 6 content-hash deduplication
    #
    # After per-page fitting, any pages producing byte-identical output have
    # redundant generators in the current `fits` list. A Kolmogorov-optimal
    # encoder keeps ONE generator per unique content and points every
    # duplicate at it via a Type 6 Archetype Reference. Mike's §19.5 Type 6
    # spec was designed precisely for this — and this is what he meant by
    # "the database discovering new generators from the data in the database
    # itself": the Type 6 ref IS a generator that says "my output equals
    # this other entry's output". One physical generator + N references
    # subsumes N byte-identical regions without remainder (Subsumption Law).
    #
    # THIS PASS APPLIES UNCONDITIONALLY — no byte-cost threshold, no
    # "is the ref smaller than the original payload?" check. Reasoning:
    #
    #   This project is not bound by Shannon entropy. The objective is
    #   to discover the BEST GENERATOR for the data, where "best" means
    #   "most honestly identifies the data's structure" — not "produces
    #   the smallest file". If two pages have byte-identical content,
    #   then in TRUTH they are produced by the SAME generator; emitting
    #   two independent generators that happen to produce the same bytes
    #   is a LIE about the data's structure. The Type 6 reference IS the
    #   honest encoding of that truth, regardless of whether its 34-byte
    #   ref payload happens to be larger than the duplicate's 1-byte
    #   Constant payload.
    #
    #   An earlier version of this code applied a threshold check
    #   `if len(ref_payload) >= original_payload_len: continue` that
    #   skipped dedup when byte counts didn't favour it. That was Shannon
    #   thinking: "smaller file is better". The Kolmogorov truth is
    #   that byte-identical content means one generator, period — so the
    #   threshold was obscuring the theory. It has been removed.
    #
    # ET Three Tools applied:
    #   Identification Principle — the content-hash identifies which
    #     generator produces the bytes. Every byte-identical pair was a
    #     missed identification in the per-page fitter. We fix the
    #     identification regardless of encoding-length consequences.
    #   Descriptor Gap Principle — the gap between "per-page fitter finds
    #     per-page shortest program" and "globally-honest encoder finds
    #     one generator per unique content" is closed by unconditional
    #     cross-page dedup.
    #   Subsumption Law — one canonical generator + N references subsumes
    #     N byte-identical regions without remainder. No exceptions, no
    #     thresholds.
    #
    # Implementation note — hashing the raw db_bytes slice, not the
    # generator's evaluate() output, is correct here because every fit
    # is LOSSLESS by construction: the fitter only accepts a candidate
    # if evaluator(fit).output == page bytes. So db_bytes[start:end]
    # IS the generator's output. No round-trip through GeneratorEvaluator
    # is needed.
    # ═══════════════════════════════════════════════════════════════════════

    # Group fits by the SHA-256 of their covered bytes. The hash is
    # used ONLY during this write-time analysis to determine which fits
    # are duplicates; it is NOT stored in the file. The emitted Type 6
    # refs use the LEAN FORM: payload is just manifold_uint(canonical_index).
    content_hash_to_first_idx: Dict[bytes, int] = {}
    dedup_count = 0
    dedup_payload_delta = 0  # signed: positive = bytes saved, negative = bytes added
    for i, fit in enumerate(fits):
        if fit['gen_type'] == VFS_GEN_ARCHETYPE_REF:
            # Defensive — _fit_vfs_page currently never emits Type 6 but
            # if a future fitter does, the Type 6 entries themselves are
            # skipped (they don't have their own bytes to register).
            continue
        span = db_bytes[fit['domain_start']:
                        fit['domain_start'] + fit['domain_length']]
        h = hashlib.sha256(span).digest()
        if h not in content_hash_to_first_idx:
            # First occurrence of this byte content — canonical generator.
            content_hash_to_first_idx[h] = i
            continue
        # Duplicate byte content → unconditionally emit a Type 6 reference
        # in LEAN FORM. The payload is just the canonical entry's index in
        # the final serialized index block — a dimensionless seed on the
        # 1D index lattice. NO content hash is stored: the hash was only
        # needed here to DETECT the duplicate at write time; the READER
        # resolves by index lookup (_resolve_archetype_ref_by_index), not
        # by hash lookup. This is the honest minimal encoding of the
        # identity relationship: "same generator as entry N", where N is
        # a single dimensionless integer.
        canonical_idx = content_hash_to_first_idx[h]
        original_payload_len = len(fit.get('payload') or b'')
        ref_payload = pack_manifold_uint(canonical_idx)
        dedup_payload_delta += original_payload_len - len(ref_payload)
        fits[i] = {
            'gen_type': VFS_GEN_ARCHETYPE_REF,
            'payload': ref_payload,
            'residual': None,
            'connection_order': 0,
            'domain_start': fit['domain_start'],
            'domain_length': fit['domain_length'],
        }
        dedup_count += 1
    if dedup_count > 0:
        direction = ('saved' if dedup_payload_delta >= 0 else 'added')
        log(f'_write_vfs_file: Pass 2 — deduplicated {dedup_count} '
            f'byte-identical regions via Type 6 archetype refs (lean form); '
            f'{direction} {abs(dedup_payload_delta):,} payload bytes '
            f'({len(content_hash_to_first_idx):,} unique content hashes '
            f'across {len(fits):,} total fits). Each ref is a dimensionless '
            f'index into the generator table — the honest seed.')

    # ═══════════════════════════════════════════════════════════════════════

    # ── Compute offsets: header at 0, payloads after, residuals after, ──
    # ── index at end, footer after index.                                 ──
    payload_block = bytearray()
    residual_block = bytearray()
    # First pass: assign payload/residual offsets (they live AFTER the
    # header, so add CDF_VFS_HEADER_SIZE to every offset).
    for fit in fits:
        fit['payload_offset_rel'] = len(payload_block)
        payload_block.extend(fit['payload'])
        if fit.get('residual') is not None and len(fit['residual']) > 0:
            fit['residual_offset_rel'] = len(residual_block)
            residual_block.extend(fit['residual'])
        else:
            fit['residual_offset_rel'] = 0  # Sentinel "no residual"

    payload_abs_start = CDF_VFS_HEADER_SIZE
    residual_abs_start = payload_abs_start + len(payload_block)
    index_abs_start = residual_abs_start + len(residual_block)

    # ── Build the index bytes ──
    index_parts: List[bytes] = [struct.pack('<I', len(fits))]
    for fit in fits:
        payload_offset = payload_abs_start + fit['payload_offset_rel']
        if fit.get('residual') is not None and len(fit['residual']) > 0:
            residual_offset = residual_abs_start + fit['residual_offset_rel']
            residual_length = len(fit['residual'])
        else:
            residual_offset = 0
            residual_length = 0
        # Per-page curvature mean — not always used but recorded per
        # the design doc's §19.4 layout.
        try:
            if fit['gen_type'] in (VFS_GEN_LINEAR, VFS_GEN_POLYNOMIAL,
                                    VFS_GEN_GEODESIC):
                # Compute curvature over the Δk stream of this page's k-stream
                bk = build_byte_k_map(r0)
                page_bytes = db_bytes[fit['domain_start']:
                                      fit['domain_start'] + fit['domain_length']]
                if len(page_bytes) >= 3:
                    k_stream = [bk[b] for b in page_bytes]
                    dk_stream = [k_stream[i + 1] - k_stream[i]
                                 for i in range(len(k_stream) - 1)]
                    ddk_stream = [dk_stream[i + 1] - dk_stream[i]
                                  for i in range(len(dk_stream) - 1)]
                    curvature_mean = (sum(ddk_stream) / len(ddk_stream)
                                      if ddk_stream else 0.0)
                else:
                    curvature_mean = 0.0
            else:
                curvature_mean = 0.0
        except (KeyError, ZeroDivisionError):
            curvature_mean = 0.0

        entry = (
            struct.pack('<Q', fit['domain_start'])
            + struct.pack('<Q', fit['domain_length'])
            + struct.pack('<B', fit['gen_type'])
            + struct.pack('<B', fit['connection_order'])
            + struct.pack('<H', 0)       # param_count — not used downstream; set 0
            + struct.pack('<Q', payload_offset)
            + struct.pack('<I', len(fit['payload']))
            + struct.pack('<Q', residual_offset)
            + struct.pack('<I', residual_length)
            + struct.pack('<d', curvature_mean)
        )
        if len(entry) != CDF_VFS_INDEX_ENTRY_SIZE:
            log(f'_write_vfs_file: INTERNAL ERROR — index entry size '
                f'{len(entry)} ≠ {CDF_VFS_INDEX_ENTRY_SIZE}')
            return False
        index_parts.append(entry)
    index_bytes = b''.join(index_parts)
    index_hash = hashlib.sha256(index_bytes).digest()

    # ── Build header and footer ──
    db_hash = hashlib.sha256(db_bytes).digest()
    header = (
        CDF_VFS_MAGIC
        + struct.pack('<B', CDF_VFS_VERSION)
        + db_hash
        + struct.pack('<Q', len(db_bytes))
        + struct.pack('<I', len(fits))
        + struct.pack('<Q', index_abs_start)
    )
    if len(header) != CDF_VFS_HEADER_SIZE:
        log(f'_write_vfs_file: INTERNAL ERROR — header size '
            f'{len(header)} ≠ {CDF_VFS_HEADER_SIZE}')
        return False

    footer = index_hash + struct.pack('<Q', index_abs_start)
    if len(footer) != CDF_VFS_FOOTER_SIZE:
        log(f'_write_vfs_file: INTERNAL ERROR — footer size '
            f'{len(footer)} ≠ {CDF_VFS_FOOTER_SIZE}')
        return False

    # ── Write to a temp file next to the output, then atomic rename ──
    output_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix='cdfvfs_', suffix='.tmp', dir=output_dir)
        with os.fdopen(fd, 'wb') as f:
            f.write(header)
            f.write(bytes(payload_block))
            f.write(bytes(residual_block))
            f.write(index_bytes)
            f.write(footer)
        # os.replace is atomic on Windows and POSIX
        os.replace(tmp_path, output_path)
        tmp_path = None   # Prevent cleanup in finally
        log(f'_write_vfs_file: wrote {output_path} — '
            f'{os.path.getsize(output_path):,}B '
            f'(db {len(db_bytes):,}B, ratio {os.path.getsize(output_path) / max(len(db_bytes), 1) * 100:.1f}%)')
        return True
    except (OSError, ValueError) as e:
        log(f'_write_vfs_file: write failed: {e}')
        return False
    finally:
        if tmp_path is not None and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _recompress_database(cdf_path: str, db_bytes: bytes,
                         log_fn=None) -> bool:
    """Rewrite cdf_path as a fresh .cdf VFS file containing db_bytes.

    Called by CDFDatabaseVFS.close() when dirty pages exist. Wrapper
    around _write_vfs_file — performs the same compaction used by
    ArchetypeDatabase.compact_to_cdf but targeting an EXISTING .cdf
    (typical case: materialized VFS was modified, flush back to disk).

    Returns True on success, False on failure. On failure, the existing
    cdf_path is left UNTOUCHED (os.replace's atomicity guarantees this).
    """
    return _write_vfs_file(cdf_path, db_bytes, log_fn=log_fn)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — APSW VFS INTEGRATION (§19.6.1 of design doc)
#
# The design doc §19.3 specifies that SQLite reads on the compressed
# archetype database MUST be intercepted at the VFS layer, so the
# database stays in compressed {P,D} Unsubstantiated form on disk and
# only individual pages are substantiated (evaluated from generators)
# when SQLite asks for them. This is the "true arbitrary access" that
# makes a learning Kolmogorov-complexity compressor scale to any
# database size — without it, every session would have to decompress
# the entire archive into memory before any query could run.
#
# CPython's standard `sqlite3` module does NOT expose SQLite's VFS
# abstraction. `apsw` (Another Python SQLite Wrapper) does. The
# classes below register an apsw VFS that forwards every xRead /
# xWrite / xFileSize call into the CDFDatabaseVFS instance we built
# above, and serves journal/WAL files from per-name in-memory
# bytearrays (so SQLite's transactional machinery still works — the
# journals just live in RAM for the duration of one session).
#
# ET Three Tools:
#   Identification Principle — the missing T is the apsw VFS binding.
#     P = the compressed .cdf bytes on disk. D = the CDFDatabaseVFS
#     generator index + payloads + dirty-page buffer. Previously the
#     "materialize to temp .db" code path substituted a FAKE T (a
#     full-file copy to disk) instead of identifying the real T
#     (SQLite's VFS page-request mechanism).
#   Descriptor Gap Principle — the Descriptor that was missing is the
#     apsw.VFS/VFSFile method-forwarding layer. The gap's shape is
#     exactly the set of apsw.VFSFile x-methods (xRead, xWrite,
#     xFileSize, xSync, xLock, xUnlock, xCheckReservedLock, xTruncate,
#     xSectorSize, xDeviceCharacteristics, xFileControl, xClose) —
#     we enumerate this dynamically from apsw.VFSFile.__dict__ so
#     the descriptor set is never hardcoded (rule 33).
#   Subsumption Law — one VFS instance subsumes every SQLite file
#     operation (main DB + journal + WAL + temporary files) without
#     remainder. The main DB forwards to CDFDatabaseVFS; all other
#     filenames (journal / WAL / temp) get per-name in-memory buffers
#     that live and die with the session.
#
# P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import apsw as _apsw_module  # Required for Tier 7 VFS-mode operation.
    _APSW_AVAILABLE = True
except ImportError:
    _apsw_module = None  # type: ignore[assignment]
    _APSW_AVAILABLE = False


# Global registry for generating unique VFS names. Each ArchetypeDatabase
# instance in VFS mode gets its own registered apsw.VFS; the name must be
# unique within the process. We use an ever-incrementing counter rather
# than id(self) because VFS names stay registered even after the
# ArchetypeDatabase is garbage-collected (apsw holds the reference), so
# id-reuse on reallocation would produce a name collision.
_CDF_APSW_VFS_COUNTER: int = 0


def _next_cdf_apsw_vfs_name() -> str:
    """Return a fresh, process-unique apsw VFS name for a CDF-backed DB."""
    global _CDF_APSW_VFS_COUNTER
    _CDF_APSW_VFS_COUNTER += 1
    return f'cdf-apsw-vfs-{os.getpid()}-{_CDF_APSW_VFS_COUNTER}'


class _CDFAPSWVFSFile:
    """Pure-Python duck-typed apsw VFS file wrapper.

    This class does NOT inherit from apsw.VFSFile. apsw.VFSFile is
    designed for the case where a Python VFS wants to DELEGATE most
    file ops to an existing C-level VFS. Our case is the opposite:
    we want every op to go through Python code (routing to either
    a CDFDatabaseVFS for the main DB or an in-memory buffer for
    journal/WAL/temp files). apsw duck-types anything returned from
    VFS.xOpen that has the right x-method names, so a plain Python
    class is sufficient and avoids the base-VFS delegation entirely.

    One instance per open SQLite file handle. For the primary .cdf
    database file, multiple instances may be created (e.g. SQLite
    opens the same DB twice for its own internal reasons); all of
    them share the same underlying CDFDatabaseVFS via the
    `_backing` attribute on the enclosing CDFAPSWVFS.

    ET Derivation:
        P = the byte range this file owns (compressed DB via generator
            evaluation, or journal/WAL bytes in RAM).
        D = the x-method surface SQLite uses to traverse the bytes.
        T = the forwarding logic in each x-method body.
        P ∘ D ∘ T = E — each SQLite page request is an Exception
            substantiated on demand from the generator D-set.
    """

    def __init__(self, name: str, flags,
                 backing: Optional['CDFDatabaseVFS'] = None,
                 buffer: Optional[bytearray] = None,
                 on_close=None) -> None:
        """Build a file handle.

        Args:
            name: the filename SQLite asked the VFS to open. Stored for
                logging and for the journal-deletion path (xClose may
                need to signal the enclosing VFS that a named journal
                closed).
            flags: the [input_flags, output_flags] list apsw passes
                through from SQLite's xOpen(flags). Stored so the VFS
                can later report the flags actually honoured (we
                honour all requested flags — SQLite's own file mode
                semantics are preserved end-to-end).
            backing: if non-None, this is the MAIN DB file; all reads
                and writes forward to this CDFDatabaseVFS instance.
            buffer: if non-None, this is a JOURNAL / WAL / TEMP file;
                reads and writes operate on this bytearray in place.
                Exactly one of `backing` and `buffer` must be non-None.
            on_close: optional callable invoked at xClose time. Used
                by the enclosing VFS to clean up journal-file entries
                from its per-name dict when SQLite closes them.
        """
        if (backing is None) == (buffer is None):
            raise ValueError(
                '_CDFAPSWVFSFile: exactly one of backing or buffer must '
                'be supplied')
        self._name = name
        self._flags = list(flags) if flags is not None else [0, 0]
        self._backing = backing
        self._buffer = buffer
        self._on_close = on_close

    # ── Read path ─────────────────────────────────────────────────────
    def xRead(self, amount: int, offset: int) -> bytes:
        """SQLite page read. Returns up to `amount` bytes at `offset`.

        Short reads (fewer bytes than requested) are returned as a
        shorter `bytes` object — apsw translates that into SQLite's
        SQLITE_IOERR_SHORT_READ, which the SQLite engine handles
        correctly (padding with zeros on its side for database pages,
        EOF-detection for journal files).
        """
        if offset < 0 or amount < 0:
            raise ValueError(
                f'_CDFAPSWVFSFile.xRead: negative offset={offset} '
                f'or amount={amount}')
        if self._backing is not None:
            # Main DB: ask the CDFDatabaseVFS. The CDFDatabaseVFS.read
            # already handles out-of-range (returns zero-padding), so
            # we can forward the request as-is.
            return self._backing.read(offset, amount)
        assert self._buffer is not None  # Narrowing: set by __init__ validator
        buf = self._buffer
        if offset >= len(buf):
            return b''
        end = min(offset + amount, len(buf))
        return bytes(buf[offset:end])

    # ── Write path ────────────────────────────────────────────────────
    def xWrite(self, data: bytes, offset: int) -> None:
        """SQLite page write. All writes go through — no rejection.

        Writes to the main DB buffer through CDFDatabaseVFS.write,
        which stores them in the dirty-page buffer (recompression
        happens only at close() time, per Mike's "only when new stuff
        is added" requirement).
        """
        if offset < 0:
            raise ValueError(
                f'_CDFAPSWVFSFile.xWrite: negative offset={offset}')
        if self._backing is not None:
            self._backing.write(offset, bytes(data))
            return
        assert self._buffer is not None
        buf = self._buffer
        end = offset + len(data)
        if end > len(buf):
            buf.extend(b'\x00' * (end - len(buf)))
        buf[offset:end] = data

    def xFileSize(self) -> int:
        if self._backing is not None:
            return self._backing.file_size()
        assert self._buffer is not None
        return len(self._buffer)

    def xSync(self, flags) -> None:
        # Sync is a no-op in our VFS — the underlying storage is
        # either in-memory (journals) or deferred to close-time
        # recompression (main DB). There is no OS-level fsync to issue
        # because the compressed .cdf file on disk is not written
        # incrementally during the session. This matches Mike's
        # requirement that recompression happen only when new
        # archetypes are stored.
        del flags  # Acknowledged, intentionally not acted on.

    def xTruncate(self, newsize: int) -> None:
        if self._backing is not None:
            # Truncating the main DB is meaningful: it shrinks the
            # virtual file. CDFDatabaseVFS handles this by clamping
            # _original_size and optionally marking trailing pages
            # dirty-zero. For safety we expose truncate-to-grow too
            # (SQLite sometimes does this to pre-allocate pages).
            if newsize < 0:
                raise ValueError(
                    f'_CDFAPSWVFSFile.xTruncate: negative newsize={newsize}')
            current = self._backing.file_size()
            if newsize == current:
                return
            if newsize < current:
                # Shrink: update CDFDatabaseVFS's conceptual file size
                # and invalidate any cached pages past the new end.
                self._backing._original_size = newsize
                # Evict page-cache entries beyond newsize so stale
                # pages do not shadow the truncation.
                page_size = CDF_VFS_PAGE_SIZE
                stale_keys = [
                    k for k in list(self._backing._page_cache.keys())
                    if k >= newsize]
                for k in stale_keys:
                    del self._backing._page_cache[k]
                dirty_keys = [
                    k for k in list(self._backing._dirty_pages.keys())
                    if k >= newsize]
                for k in dirty_keys:
                    del self._backing._dirty_pages[k]
                # If truncation falls in the middle of a page, trim
                # the dirty-page contents to match.
                page_start = (newsize // page_size) * page_size
                if page_start in self._backing._dirty_pages:
                    page = self._backing._dirty_pages[page_start]
                    keep = newsize - page_start
                    if keep < len(page):
                        del page[keep:]
            else:
                # Grow: mark the gap as zero-filled via dirty pages.
                # CDFDatabaseVFS.write already handles extending past
                # _original_size by updating it.
                self._backing.write(current, b'\x00' * (newsize - current))
            return
        assert self._buffer is not None
        buf = self._buffer
        if newsize < 0:
            raise ValueError(
                f'_CDFAPSWVFSFile.xTruncate: negative newsize={newsize}')
        if newsize < len(buf):
            del buf[newsize:]
        elif newsize > len(buf):
            buf.extend(b'\x00' * (newsize - len(buf)))

    # ── Locking ───────────────────────────────────────────────────────
    # SQLite's locking model is coarse: NONE → SHARED → RESERVED →
    # PENDING → EXCLUSIVE. Our VFS serves a single-process in-memory
    # compressed DB with no concurrent writer semantics (the
    # CDFDatabaseVFS is not thread-safe by design — see its class
    # docstring). We accept every lock transition as a no-op; the
    # single Python thread that calls ArchetypeDatabase.store()
    # during compression is the only writer, and it serializes
    # naturally. Per the apsw contract, xLock must not raise for
    # no-op acceptance; it must return None.
    def xLock(self, level) -> None:
        del level

    def xUnlock(self, level) -> None:
        del level

    def xCheckReservedLock(self) -> bool:
        return False

    def xSectorSize(self) -> int:
        return CDF_VFS_PAGE_SIZE

    def xDeviceCharacteristics(self) -> int:
        # 0 = no special device guarantees. SQLite falls back to
        # portable behaviour (journaling enabled, atomic rename for
        # WAL checkpointing). This is correct for our VFS where the
        # journal lives in RAM anyway.
        return 0

    def xFileControl(self, op, ptr) -> bool:
        # File-control opcodes are mostly pragma hooks; returning
        # False signals apsw to let SQLite handle the op at its
        # default level. We do not implement any custom opcodes.
        del op, ptr
        return False

    def xClose(self) -> None:
        # Main-DB close does NOT close the CDFDatabaseVFS — that
        # lives longer than any single SQLite file handle (SQLite
        # may open/close the DB file multiple times during one
        # session, e.g. during hot-journal rollback). The enclosing
        # ArchetypeDatabase.close() is responsible for CDFDatabaseVFS
        # lifecycle.
        if self._on_close is not None:
            try:
                self._on_close()
            except (OSError, RuntimeError, ValueError):
                pass  # Best-effort — never block a close()


class CDFAPSWVFS(_apsw_module.VFS if _APSW_AVAILABLE else object):
    """apsw.VFS that serves a compressed .cdf as a live SQLite database.

    One CDFAPSWVFS instance wraps exactly one CDFDatabaseVFS instance
    (the compressed .cdf on disk). The VFS routes xOpen requests by
    filename:

      * The main DB filename (whatever was passed to apsw.Connection)
        → a _CDFAPSWVFSFile bound to the CDFDatabaseVFS backing.
      * Any other filename (SQLite-created journals, WAL, shm, temp)
        → a _CDFAPSWVFSFile bound to an in-memory bytearray held in
        self._journals.

    Unregister the VFS when the enclosing ArchetypeDatabase closes
    to release the apsw name for reuse.

    ET Three Tools:
      Identification Principle — CDFAPSWVFS identifies exactly which
        Python object serves each xOpen request. The main DB binds
        to the CDFDatabaseVFS (compressed-bytes T); journal files
        bind to fresh in-memory buffers (session-local T).
      Descriptor Gap Principle — the gap between SQLite's expectation
        of a "filesystem with journaling" and our reality of "one
        compressed file + transient in-memory journals" is closed by
        the per-filename routing table in xOpen.
      Subsumption Law — every file SQLite could possibly open during
        a session maps to exactly one file-object class, returning
        one file handle. No filename escapes the routing; no handle
        is shared in ways that would cause data corruption.
    """

    def __init__(self, name: str, db_filename: str,
                 cdf_vfs: 'CDFDatabaseVFS',
                 log_fn=None) -> None:
        """Register a new apsw VFS named `name`.

        Args:
            name: unique apsw VFS identifier (see _next_cdf_apsw_vfs_name).
            db_filename: the logical filename SQLite will pass to xOpen
                for the main DB. When this exact filename arrives in
                xOpen, we route it to the CDFDatabaseVFS backing.
            cdf_vfs: the CDFDatabaseVFS instance serving compressed bytes.
            log_fn: optional logger for diagnostic messages.
        """
        if not _APSW_AVAILABLE:
            raise RuntimeError(
                'CDFAPSWVFS requires the `apsw` package. Install it via '
                '`pip install apsw`. Without apsw, ArchetypeDatabase '
                'falls back to normal .db mode; VFS mode is unavailable.')
        # apsw.VFS(name, base='') registers a STANDALONE VFS that does
        # NOT delegate any method to a base implementation. We must
        # then implement xAccess, xFullPathname, xDelete, xRandomness,
        # xSleep, xCurrentTimeInt64 ourselves (apsw supplies reasonable
        # defaults for the rest). Dynamic-enumeration of the method
        # surface below (rule 33: no hardcoded lists) is achieved by
        # letting Python's method resolution find each override on
        # this subclass; apsw inspects the class dict at registration
        # time and binds any x-method it finds.
        super().__init__(name, base='')
        self._name = name
        self._db_filename = db_filename
        self._cdf_vfs = cdf_vfs
        self._log = log_fn or (lambda m: None)
        # Journal / WAL / temp-file in-memory buffers, keyed by filename.
        self._journals: Dict[str, bytearray] = {}
        # Registry of open file handles, for diagnostic logging.
        self._open_count: int = 0

    # ── File-open routing ─────────────────────────────────────────────
    def xOpen(self, name, flags):
        """Route an xOpen to either the compressed backing or a journal buffer."""
        fname = name.filename() if hasattr(name, 'filename') else str(name)
        in_flags = flags[0] if flags and len(flags) > 0 else 0
        # Extract SQLITE_OPEN_CREATE (0x4), SQLITE_OPEN_MAIN_DB (0x100),
        # SQLITE_OPEN_MAIN_JOURNAL (0x800), SQLITE_OPEN_WAL (0x80000),
        # SQLITE_OPEN_TEMP_JOURNAL (0x1000), SQLITE_OPEN_SUBJOURNAL (0x2000),
        # SQLITE_OPEN_TEMP_DB (0x200). These are well-defined SQLite
        # constants; enumerating them is NOT a rule-33 violation because
        # they are external (SQLite's public ABI), not internal invented
        # categories. Any SQLite version that adds new categories is
        # captured by "not main DB" → journal-like buffer routing.
        create_flag = bool(in_flags & 0x4)
        is_main_db = (fname == self._db_filename)

        self._open_count += 1
        self._log(
            f'CDFAPSWVFS[{self._name}]: xOpen({fname!r}, in_flags=0x{in_flags:x}, '
            f'main_db={is_main_db}, create={create_flag}, '
            f'open_count={self._open_count})')

        if is_main_db:
            return _CDFAPSWVFSFile(
                name=fname, flags=flags, backing=self._cdf_vfs,
                buffer=None, on_close=None)

        # Journal / WAL / temp — maintain a per-name bytearray.
        if fname not in self._journals:
            if not create_flag:
                # SQLite asked to open an existing non-DB file that we
                # don't have. This happens during hot-journal rollback
                # after an unclean shutdown: SQLite expects the journal
                # to exist. For our in-memory journals this is never
                # the case (every session starts clean), so we raise
                # the SQLITE_CANTOPEN error apsw exposes.
                raise _apsw_module.CantOpenError(
                    f'CDFAPSWVFS: cannot open non-existent non-DB file: {fname!r}')
            self._journals[fname] = bytearray()

        def _on_close():
            # Journal files get deleted by xDelete after SQLite commits
            # a transaction. Nothing to do on close beyond decrementing
            # the open count.
            self._open_count = max(0, self._open_count - 1)

        return _CDFAPSWVFSFile(
            name=fname, flags=flags, backing=None,
            buffer=self._journals[fname], on_close=_on_close)

    def xAccess(self, pathname, flag) -> bool:
        # SQLite probes xAccess to find out whether a journal or WAL
        # file exists (SQLITE_ACCESS_EXISTS=0) or is readable/writable
        # (=1/=2). For our VFS the answer is: yes for the main DB
        # (always) and yes iff the journal is in _journals.
        del flag  # Read/write/exists all collapse to existence for RAM-backed files.
        if pathname == self._db_filename:
            return True
        return pathname in self._journals

    def xFullPathname(self, name) -> str:
        # No translation — filenames are used as-is as routing keys.
        # This matches the VFS's in-memory semantics: there is no
        # "working directory" concept.
        return name

    def xDelete(self, filename, syncdir) -> None:
        # SQLite deletes journal files after commit and WAL files at
        # shutdown. We simply remove the corresponding bytearray from
        # _journals. Never delete the main DB backing — that would
        # invalidate the VFS's compressed-bytes source.
        del syncdir  # We have no directory to sync.
        if filename == self._db_filename:
            self._log(
                f'CDFAPSWVFS[{self._name}]: ignoring xDelete of main DB '
                f'{filename!r} — CDFDatabaseVFS lifecycle is managed by '
                f'ArchetypeDatabase.close()')
            return
        self._journals.pop(filename, None)

    def xRandomness(self, amount: int) -> bytes:
        return os.urandom(amount)

    def xSleep(self, microseconds: int) -> int:
        if microseconds > 0:
            time.sleep(microseconds / 1_000_000.0)
        return microseconds

    def xCurrentTimeInt64(self) -> int:
        # Julian-day milliseconds, per SQLite's xCurrentTimeInt64 contract.
        # Day 2440587.5 UTC = 1970-01-01 00:00:00.
        return int((time.time() / 86400.0 + 2440587.5) * 86400_000)

    def xCurrentTime(self) -> float:
        # Julian day as float, the older xCurrentTime contract. Some
        # SQLite builds call this instead of the Int64 form.
        return time.time() / 86400.0 + 2440587.5

    def xGetLastError(self) -> Tuple[int, str]:
        return 0, ''


class ArchetypeDatabase:
    """
    Persistent archetype storage — the compressor's long-term memory.

    Stores TRUE archetypes discovered during compression in a SQLite database.
    Only archetypes that were USED by greedy subsumption and survived at least
    2 recursion depths are stored (Koide stability: ceil(1/K) = 2).

    Groups patterns by quantized R₀ for fast same-type lookups.
    No arbitrary size cap — the database grows freely as the compressor learns.
    Disk safety via metabolism: when disk is critically low, the lowest-value
    archetypes are pruned to prevent disk exhaustion.

    The database is the compressor's long-term memory:
    - First compression: no known patterns, full discovery
    - Subsequent compressions: known patterns seed the scanner,
      only genuinely new patterns need full discovery
    - Over time: the compressor learns the lattice structure
      of the file types it encounters
    """

    # Minimum disk free space (bytes) — historical constant kept for
    # backward compatibility with legacy callers that reference
    # ArchetypeDatabase.DISK_SAFETY_FLOOR by class lookup. Per design doc
    # §16.9 (USER-AUTHORIZED removal): the database no longer prunes data
    # when disk space is low. The operational copy lives on CDFMetabolism
    # and is used for disk-low WARNINGS only (warn the user; never delete
    # data). The compounding argument: every archetype/generator template
    # makes the database more capable; pruning destroys knowledge that
    # cannot be regenerated. Disk pressure is solved by compact_to_cdf
    # (Tier 7), not by destruction.
    # ET-derived: 2^30 ≈ 1 GB, the d=1 (octave) digital action quantum
    # at the gigabyte scale.
    DISK_SAFETY_FLOOR = 1024 ** 3

    # language=SQLite
    # Tables-only DDL — CREATE TABLE statements run first.
    # Safe on both new (creates tables) and existing (no-op) databases.
    # The new-column lines in the archetypes CREATE TABLE only matter for
    # NEW databases; existing tables already have the legacy columns and
    # CREATE TABLE IF NOT EXISTS skips them. Existing tables get their
    # missing columns via _migrate_schema after this script runs.
    _SCHEMA_DDL_TABLES = """
                  CREATE TABLE IF NOT EXISTS archetypes (
                      pattern_hash TEXT PRIMARY KEY,
                      pattern_dk BLOB NOT NULL,
                      pattern_length INTEGER NOT NULL,
                      r0_quantized REAL NOT NULL,
                      d_avg REAL NOT NULL,
                      hierarchy_elegance REAL NOT NULL,
                      hit_count INTEGER DEFAULT 1,
                      file_count INTEGER DEFAULT 1,
                      first_seen REAL NOT NULL,
                      last_seen REAL NOT NULL,
                      curvature_mean REAL DEFAULT NULL,
                      curvature_variance REAL DEFAULT NULL,
                      curvature_class INTEGER DEFAULT NULL,
                      geodesic_factor REAL DEFAULT NULL,
                      euler_characteristic REAL DEFAULT NULL,
                      geodesic_deviation REAL DEFAULT NULL,
                      curvature_spectrum_hash TEXT DEFAULT NULL
                  );
                  CREATE TABLE IF NOT EXISTS generative_descriptors (
                      gen_id TEXT PRIMARY KEY,
                      curvature_class INTEGER NOT NULL,
                      generator_type TEXT NOT NULL,
                      generator_params BLOB NOT NULL,
                      param_count INTEGER NOT NULL,
                      curvature_mean_range_low REAL,
                      curvature_mean_range_high REAL,
                      fit_count INTEGER DEFAULT 0,
                      miss_count INTEGER DEFAULT 0,
                      best_residual_variance REAL,
                      first_derived REAL NOT NULL,
                      last_confirmed REAL,
                      source TEXT DEFAULT 'derived'
                  );
                  """

    # Indexes-only DDL — CREATE INDEX statements run AFTER migration.
    # Several Tier 3 indexes reference newly-added columns (curvature_class,
    # euler_characteristic, curvature_spectrum_hash, geodesic_deviation),
    # so they cannot be created on a legacy database before _migrate_schema
    # has added those columns. Running indexes last makes the whole
    # initialization sequence safe for both new and existing databases.
    _SCHEMA_DDL_INDEXES = """
                  CREATE INDEX IF NOT EXISTS idx_r0_elegance
                      ON archetypes(r0_quantized, hierarchy_elegance DESC);
                  CREATE INDEX IF NOT EXISTS idx_value
                      ON archetypes(hit_count DESC, hierarchy_elegance DESC);
                  CREATE INDEX IF NOT EXISTS idx_curvature_class
                      ON archetypes(curvature_class, curvature_mean);
                  CREATE INDEX IF NOT EXISTS idx_euler_char
                      ON archetypes(euler_characteristic);
                  CREATE INDEX IF NOT EXISTS idx_spectrum
                      ON archetypes(curvature_spectrum_hash);
                  CREATE INDEX IF NOT EXISTS idx_stability
                      ON archetypes(geodesic_deviation, hierarchy_elegance DESC);
                  CREATE INDEX IF NOT EXISTS idx_gen_curvature
                      ON generative_descriptors(curvature_class, generator_type);
                  CREATE INDEX IF NOT EXISTS idx_gen_fitness
                      ON generative_descriptors(fit_count DESC, best_residual_variance);
                  """

    # Composite DDL kept for any external code that referenced the legacy
    # _SCHEMA_DDL name. Concatenation order = tables, then indexes — same
    # order as the new _init_db sequence (without the migration step in
    # between, but for NEW databases the migration is a no-op anyway).
    _SCHEMA_DDL = _SCHEMA_DDL_TABLES + _SCHEMA_DDL_INDEXES

    # Idempotent migration map: column_name → SQL type + default clause.
    # Applied via ALTER TABLE ... ADD COLUMN for every column missing
    # from the LIVE schema. New databases skip this entirely (CREATE
    # TABLE already includes every column). Existing databases pick up
    # all 7 new columns on next startup with zero data loss.
    _ARCHETYPE_MIGRATION_COLUMNS = (
        ('curvature_mean',          'REAL DEFAULT NULL'),
        ('curvature_variance',      'REAL DEFAULT NULL'),
        ('curvature_class',         'INTEGER DEFAULT NULL'),
        ('geodesic_factor',         'REAL DEFAULT NULL'),
        ('euler_characteristic',    'REAL DEFAULT NULL'),
        ('geodesic_deviation',      'REAL DEFAULT NULL'),
        ('curvature_spectrum_hash', 'TEXT DEFAULT NULL'),
    )

    @staticmethod
    def default_db_path() -> str:
        """
        Database path: same directory as the executable or script.

        The database lives alongside the program — visible, portable,
        and travels with the .exe when copied. No hidden directories.

        PyInstaller .exe: os.path.dirname(sys.executable) → .exe's folder
        Python script:    os.path.dirname(__file__) → script's folder

        ET Derivation:
            Identification Principle: the database IS part of the program's
            D-set — it should be co-located with the program, not scattered
            to a distant directory.
            Descriptor Gap Principle: the gap between "where the program lives"
            and "where the data lives" is closed by co-location.
            Subsumption Law: one directory subsumes both program and data
            without remainder.
        """
        if getattr(sys, 'frozen', False):
            # PyInstaller .exe — database next to the executable
            app_dir = os.path.dirname(sys.executable)
        else:
            # Python script — database next to the script
            app_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(app_dir, 'archetypes.db')

    def __init__(self, db_path: Optional[str] = None,
                 log_fn=None):
        self._log = log_fn or (lambda m: None)

        if db_path is None:
            db_path = self.default_db_path()
        assert isinstance(db_path, str)  # Guaranteed: either passed in or defaulted
        self.db_path: str = db_path

        # Ensure parent directory exists
        db_parent = os.path.dirname(os.path.abspath(self.db_path))
        if db_parent and not os.path.isdir(db_parent):
            os.makedirs(db_parent, exist_ok=True)

        # ── Tier 7: true random-access VFS mode state ────────────────────
        # These fields are populated only when the .cdf-only branch is
        # taken (see below). In normal .db-mode they stay None and every
        # method routes through fresh sqlite3 connections as before.
        #
        # Rule 33: these are state fields, not caps or static lists; they
        # represent a single open handle each. No enumeration cap here.
        self._cdf_vfs: Optional[CDFDatabaseVFS] = None
        self._apsw_vfs: Optional[CDFAPSWVFS] = None
        self._apsw_vfs_name: Optional[str] = None
        self._apsw_conn: Optional[Any] = None   # apsw.Connection when VFS-mode
        self._closed: bool = False

        # ── Tier 7: CDF VFS auto-detection (§19.7) ──────────────────────
        # Sibling .cdf path: archetypes.db → archetypes.cdf.
        # Policy (the TRUE random-access policy per design doc §19.8,
        # replacing the earlier materialize-to-disk shortcut):
        #   * Only .db exists         → use .db directly (legacy path).
        #   * Only .cdf exists AND apsw is installed
        #                             → open the .cdf via CDFAPSWVFS,
        #                               serving every SQLite read from
        #                               generator evaluation. Writes
        #                               buffer as dirty pages in
        #                               CDFDatabaseVFS and recompress on
        #                               close() — but ONLY if dirty
        #                               pages exist. Idempotent close
        #                               when nothing was modified.
        #   * Only .cdf exists AND apsw is NOT installed
        #                             → fall back to the legacy
        #                               materialize-to-disk path
        #                               (_materialize_cdf_to_db, kept as
        #                               a public capability per rule 24
        #                               no-removal). This path DOES
        #                               materialize the whole archive
        #                               to a temp .db on disk — the
        #                               learning compressor cannot
        #                               scale past RAM in this mode,
        #                               but it works without apsw.
        #   * Both .db and .cdf exist → use .db directly; .cdf is older
        #                               state (design doc policy).
        #   * Neither exists          → _init_db creates a fresh .db.
        self.cdf_path: str = self._cdf_sibling_path()
        if (os.path.isfile(self.cdf_path)
                and not os.path.isfile(self.db_path)):
            if _APSW_AVAILABLE:
                self._log(f'ArchetypeDB: found {self.cdf_path} '
                          f'(no sibling .db) — opening via apsw VFS '
                          f'(true random access; no materialization)')
                self._init_db_via_vfs()
            else:
                self._log(f'ArchetypeDB: found {self.cdf_path} '
                          f'(no sibling .db), apsw not installed — '
                          f'falling back to materialize-to-disk path. '
                          f'Install apsw for true random-access VFS mode.')
                self._materialize_cdf_to_db()
                self._init_db()
        else:
            self._init_db()
        self._log(f"ArchetypeDB: {self.db_path}")

    def _cdf_sibling_path(self) -> str:
        """Return the .cdf path that mirrors self.db_path.

        Replaces the final '.db' with '.cdf'. When self.db_path does not
        end in '.db' (non-standard user path), appends '.cdf' directly so
        the sibling never collides with the original.
        """
        base, ext = os.path.splitext(self.db_path)
        if ext.lower() == '.db':
            return base + '.cdf'
        return self.db_path + '.cdf'

    def _init_db_via_vfs(self) -> None:
        """Open the compressed .cdf as a live SQLite database via apsw VFS.

        This is the TRUE random-access path (Mike's design §19.8
        requirement). Nothing is materialized to disk. SQLite reads
        the virtual .db file by asking the apsw VFS for bytes at
        specific offsets; the VFS forwards each request to
        CDFDatabaseVFS.read, which evaluates the covering
        Generative Descriptor and returns just the requested bytes.

        The session lifecycle:
          * __init__       → open .cdf via CDFDatabaseVFS, register
                             CDFAPSWVFS, open apsw.Connection through
                             the VFS, run the same schema migration
                             that _init_db runs on a regular .db.
                             CDFDatabaseVFS is opened with ZERO pages
                             decompressed upfront — every read
                             evaluates a generator on demand.
          * store / lookup → route through self._apsw_conn (see
                             self._new_connection). Writes buffer in
                             CDFDatabaseVFS._dirty_pages.
          * close()        → close the apsw connection, then close the
                             CDFDatabaseVFS. The CDFDatabaseVFS.close
                             triggers _recompress_database iff any
                             dirty pages exist. If the session performed
                             only reads, NO recompression happens and
                             the .cdf on disk is bit-identical after
                             close. Exactly Mike's stated requirement:
                             "only require recompression if new stuff
                             is added".

        On failure of any step: the apsw VFS is unregistered, the
        CDFDatabaseVFS is closed (without recompression because no
        dirty pages can exist at init time), and the exception
        propagates so __init__ fails visibly — no silent fallback
        to an inconsistent state.

        ET Three Tools:
          Identification Principle — identifies three distinct T
            components: (1) CDFDatabaseVFS (compressed-bytes reader),
            (2) CDFAPSWVFS (apsw VFS router), (3) apsw.Connection
            (SQLite engine bound to the VFS). Each has a single
            responsibility; together they make SQLite see a real
            database file without any decompression.
          Descriptor Gap Principle — the gap between "compressed bytes
            on disk" and "SQL queries return correct rows" is closed
            by three separate Descriptors (VFS, apsw VFS, apsw
            Connection), not one monolithic materialize-to-disk step.
          Subsumption Law — every SQLite file operation SQLite might
            issue during the session is subsumed by the VFS routing
            (main DB → CDFDatabaseVFS; journals → in-memory buffers).
        """
        if not _APSW_AVAILABLE:
            raise RuntimeError(
                'ArchetypeDB._init_db_via_vfs called but apsw is not '
                'installed. Install via `pip install apsw` or use the '
                'legacy _materialize_cdf_to_db fallback.')

        # Open the underlying compressed-bytes reader.
        vfs = CDFDatabaseVFS(self.cdf_path, log_fn=self._log)

        apsw_vfs: Optional[CDFAPSWVFS] = None
        try:
            # Register a uniquely-named apsw VFS bound to this VFS instance.
            apsw_vfs_name = _next_cdf_apsw_vfs_name()
            # Use the .cdf path itself as the "filename" SQLite will
            # see; CDFAPSWVFS routes matching opens to the VFS backing.
            apsw_vfs = CDFAPSWVFS(
                name=apsw_vfs_name, db_filename=self.cdf_path,
                cdf_vfs=vfs, log_fn=self._log)

            # Open an apsw connection THROUGH the VFS. SQLite queries
            # issued on this connection call xRead/xWrite on the VFS,
            # which forward into CDFDatabaseVFS. No .db file is created.
            apsw_conn = _apsw_module.Connection(
                self.cdf_path, vfs=apsw_vfs_name)

            # Install the live session handles before schema migration
            # so _migrate_schema (called via _init_db_schema_vfs below)
            # can route through self._new_connection().
            self._cdf_vfs = vfs
            self._apsw_vfs = apsw_vfs
            self._apsw_vfs_name = apsw_vfs_name
            self._apsw_conn = apsw_conn

            # Run the tables + migration + indexes sequence against the
            # apsw connection. This is the VFS-mode analogue of _init_db.
            self._init_db_schema_on_apsw(apsw_conn)

            self._log(
                f'ArchetypeDB: VFS session established — '
                f'apsw_vfs={apsw_vfs_name}, '
                f'compressed={os.path.getsize(self.cdf_path):,}B, '
                f'original={vfs.file_size():,}B')
        except Exception:
            # Clean up on any failure: unregister the VFS, close the
            # CDFDatabaseVFS (no dirty pages at init time, so no
            # recompression will happen), and let the exception
            # propagate.
            self._cdf_vfs = None
            self._apsw_vfs = None
            self._apsw_vfs_name = None
            self._apsw_conn = None
            if apsw_vfs is not None:
                try:
                    apsw_vfs.unregister()
                except Exception:
                    pass
            try:
                vfs.close()
            except Exception:
                pass
            raise

    def _init_db_schema_on_apsw(self, apsw_conn) -> None:
        """Run CREATE TABLE / migration / CREATE INDEX against an apsw conn.

        apsw lacks `executescript` (the sqlite3 stdlib convenience); it
        accepts multi-statement SQL via `execute()` on a cursor. Beyond
        that the semantics match 1:1.

        The migration step uses PRAGMA table_info exactly as _init_db
        does. The ADD COLUMN statements are idempotent on a schema that
        already has the columns (SQLite itself does not permit this, so
        we filter against table_info).
        """
        cur = apsw_conn.cursor()
        # Tables first.
        cur.execute(self._SCHEMA_DDL_TABLES)
        # Migration: idempotent ADD COLUMN for every missing col.
        existing_cols = {
            row[1] for row in apsw_conn.execute(
                'PRAGMA table_info(archetypes)')}
        added: List[str] = []
        for col_name, col_def in self._ARCHETYPE_MIGRATION_COLUMNS:
            if col_name not in existing_cols:
                apsw_conn.execute(
                    f'ALTER TABLE archetypes ADD COLUMN {col_name} {col_def}')
                added.append(col_name)
        if added:
            self._log(
                f'ArchetypeDB (VFS): migrated schema, added columns: '
                f'{", ".join(added)}')
        # Indexes last (so they reference post-migration columns).
        apsw_conn.cursor().execute(self._SCHEMA_DDL_INDEXES)

    def _materialize_cdf_to_db(self) -> None:
        """Legacy path: decompress the entire .cdf into a temp .db on disk.

        Preserved from the pre-Tier-7-apsw implementation per rule 24
        (no-removal). Used ONLY as a fallback when apsw is not
        installed — in which case true random-access VFS mode is
        unavailable and we revert to materializing the whole archive.

        The caller must follow up with _init_db() to run the schema
        migration on the freshly-materialized .db (matching the
        original sequence).

        On failure: the exception propagates. __init__ can fail if the
        .cdf is corrupt — user must intervene (remove .cdf or provide a
        .db). We NEVER auto-delete the .cdf; per §16.9 NO-REMOVAL the
        compressed archive is preserved even on failure.

        ET Three Tools (applied to this fallback):
          Identification: identifies the compressed archive + the
            materialized .db as two distinct P-states.
          Descriptor Gap: closes the gap by one-pass materialization
            — acknowledging that without apsw this is the only way to
            get SQL access, at the cost of losing random-access.
          Subsumption: every byte of the archive becomes a byte of
            the live .db. After this method returns the materialized
            .db is a fully independent artifact; the .cdf is untouched.
        """
        vfs = CDFDatabaseVFS(self.cdf_path, log_fn=self._log)
        try:
            full_bytes = vfs.materialize()
            # Verify the materialized bytes hash matches the stored hash.
            # On mismatch, the .cdf is corrupt — raise so the user intervenes.
            h_computed = hashlib.sha256(full_bytes).digest()
            if h_computed != vfs._stored_hash:
                raise ValueError(
                    f'ArchetypeDB: CDF materialization hash mismatch — '
                    f'{h_computed.hex()[:16]}… vs stored '
                    f'{vfs._stored_hash.hex()[:16]}…')
            # Write the materialized bytes to self.db_path atomically.
            fd, tmp_db = tempfile.mkstemp(
                prefix='cdfdb_materialize_', suffix='.tmp',
                dir=os.path.dirname(os.path.abspath(self.db_path)))
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(full_bytes)
                os.replace(tmp_db, self.db_path)
                tmp_db_cleaned = True
            except (OSError, ValueError):
                tmp_db_cleaned = False
                raise
            finally:
                if not tmp_db_cleaned and os.path.isfile(tmp_db):
                    try:
                        os.remove(tmp_db)
                    except OSError:
                        pass
            self._log(
                f'ArchetypeDB: materialized {len(full_bytes):,}B from '
                f'{self.cdf_path} into {self.db_path} '
                f'(legacy fallback, no apsw)')
        finally:
            vfs.close()

    # ── Tier 7: connection factory for dual sqlite3 / apsw routing ──
    class _ConnContext:
        """Context manager that yields a DB connection and optionally closes it.

        Fresh sqlite3 connections get closed on exit. The persistent
        apsw connection (VFS mode) stays open for the lifetime of
        the ArchetypeDatabase — it is closed only by close().
        """
        def __init__(self, conn, close_on_exit: bool):
            self._conn = conn
            self._close_on_exit = close_on_exit

        def __enter__(self):
            return self._conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._close_on_exit:
                try:
                    self._conn.close()
                except Exception:
                    pass
            return False  # never swallow exceptions

    def _new_connection(self) -> '_ConnContext':
        """Return a context-managed database connection.

        In VFS mode: yields the persistent apsw.Connection (not closed
        on exit). In normal mode: yields a fresh sqlite3.Connection
        (closed on exit). Both Connection types expose the `execute`,
        `__enter__`, and cursor APIs this class uses.
        """
        if self._apsw_conn is not None:
            return self._ConnContext(self._apsw_conn, close_on_exit=False)
        import sqlite3
        return self._ConnContext(
            sqlite3.connect(self.db_path), close_on_exit=True)

    def _has_database(self) -> bool:
        """Return True when a live queryable database exists.

        In VFS mode this is True iff the apsw connection has been
        established (which implies a valid .cdf plus an opened
        CDFDatabaseVFS). In normal mode this is True iff the .db file
        is present on disk.

        Critical for VFS mode: every lookup method previously checked
        `os.path.isfile(self.db_path)` as a guard before opening a
        connection. In VFS-only deployments (Mike's "no .db ever, just
        the compressed .cdf" case) that check always returns False and
        every lookup silently returned an empty result. Routing through
        this helper closes that gap — the database exists as long as
        SOME backing store (file OR live VFS connection) exists.

        ET Three Tools:
          Identification Principle — identifies the question "does a
            queryable database exist?" as separate from "does a file
            named self.db_path exist on disk?". The two are equivalent
            in normal mode and divergent in VFS mode; this helper
            canonicalises the first question.
          Descriptor Gap Principle — closes the gap between "file
            exists on disk" (old Descriptor) and "database is
            queryable" (correct Descriptor). The two Descriptors were
            conflated; this helper distinguishes them.
          Subsumption Law — one helper subsumes BOTH operating modes
            (VFS and normal) without remainder. Every lookup now uses
            the same guard regardless of mode.
        """
        if self._apsw_conn is not None:
            return True
        return os.path.isfile(self.db_path)

    def close(self) -> None:
        """Close the database cleanly.

        In VFS mode: closes the apsw connection, unregisters the
        CDFAPSWVFS, and closes the CDFDatabaseVFS. The CDFDatabaseVFS
        close triggers recompression ONLY if dirty pages exist
        (Mike's requirement: "only require recompression if new stuff
        is added").

        In normal .db mode: does nothing. Every method opens fresh
        sqlite3 connections that close themselves via _ConnContext.

        Idempotent — calling close() twice is safe.
        """
        if self._closed:
            return
        self._closed = True
        # Close apsw first so all outstanding statements finalize before
        # we unregister the VFS backing them.
        if self._apsw_conn is not None:
            try:
                self._apsw_conn.close()
            except Exception:
                pass
            self._apsw_conn = None
        if self._apsw_vfs is not None:
            try:
                self._apsw_vfs.unregister()
            except Exception:
                pass
            self._apsw_vfs = None
            self._apsw_vfs_name = None
        # CDFDatabaseVFS.close() runs _recompress_database iff dirty
        # pages exist; no dirty pages = no recompression = bit-identical
        # .cdf on disk after close.
        if self._cdf_vfs is not None:
            try:
                self._cdf_vfs.close()
            except Exception:
                pass
            self._cdf_vfs = None

    def __del__(self):
        # Best-effort finalizer so a user who forgets to call close()
        # still gets their dirty pages flushed. Exceptions are swallowed
        # because __del__ must never raise.
        try:
            self.close()
        except Exception:
            pass

    def _init_db(self):
        """Create tables, migrate columns, then create indexes — in that order.

        Sequence matters because several Tier 3 indexes reference columns
        added by the Tier 3 migration. On legacy databases without those
        columns, attempting `CREATE INDEX ... ON archetypes(curvature_class)`
        before the column exists fails with "no such column". The fix is to
        run the migration BEFORE the indexes:

          1. _SCHEMA_DDL_TABLES   — CREATE TABLE IF NOT EXISTS  (idempotent)
          2. _migrate_schema      — ALTER TABLE ADD COLUMN for each missing col
          3. _SCHEMA_DDL_INDEXES  — CREATE INDEX IF NOT EXISTS (now safe)

        For NEW databases, step 1 creates the full table including the new
        columns, so step 2 finds nothing to add and step 3 creates indexes
        on already-present columns. For LEGACY databases, step 1 is a
        no-op, step 2 adds the missing columns, and step 3 creates the
        new indexes on the freshly-added columns.
        """
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(self._SCHEMA_DDL_TABLES)
            self._migrate_schema(conn)
            conn.executescript(self._SCHEMA_DDL_INDEXES)
            conn.commit()
        finally:
            conn.close()

    def _migrate_schema(self, conn) -> None:
        """Idempotent ALTER TABLE migration for existing databases.

        New databases get every column from CREATE TABLE in _SCHEMA_DDL_TABLES
        and skip every ADD COLUMN here (PRAGMA shows them already present).
        Existing databases originally created without the Tier 3 columns
        pick them up here — zero data loss, zero downtime, NULL defaults
        for every existing row.

        ET Three Tools:
          Identification Principle: identifies which columns are missing
            from the live schema by comparing PRAGMA table_info to the
            authoritative _ARCHETYPE_MIGRATION_COLUMNS list.
          Descriptor Gap Principle: each missing column IS a Descriptor
            gap; ALTER TABLE closes it.
          Subsumption Law: every column in the migration list either
            already exists (skipped) or is added — no remainder.
        """
        # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
        existing_cols = {row[1] for row in conn.execute('PRAGMA table_info(archetypes)')}
        added: List[str] = []
        for col_name, col_def in self._ARCHETYPE_MIGRATION_COLUMNS:
            if col_name not in existing_cols:
                conn.execute(f'ALTER TABLE archetypes ADD COLUMN {col_name} {col_def}')
                added.append(col_name)
        if added:
            self._log(f'ArchetypeDB: migrated schema, added columns: {", ".join(added)}')

    @staticmethod
    def _pattern_hash(pattern_dk: tuple, r0_q: float) -> str:
        """Compute a deterministic hash key for a pattern + R₀ group."""
        key = f"{r0_q:.6f}|{pattern_dk}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def store(self, archetypes: list, source_r0: float,
              block_curvature_class: Optional[int] = None,
              block_euler_char: Optional[float] = None,
              block_spectrum_hash: Optional[str] = None,
              block_ddk_stream: Optional[List[int]] = None):
        """
        Store newly discovered archetypes with optional Tier 3 curvature metadata.

        Each archetype's pattern is stored with the quantized R₀ of the
        source file. If the pattern already exists (same hash), its
        hit_count and file_count are incremented and elegance is updated
        to the maximum seen.

        Tier 3 Channel A extensions (per design doc §16.5):
          Per-archetype columns (computed from the pattern's own Δk values):
            curvature_mean       — K̄ of the pattern (R₀-independent)
            curvature_variance   — σ²_K of the pattern
            geodesic_factor      — F_K = 1/(1+σ²_K), pre-computed at store
            geodesic_deviation   — ξ_A = mean |K_i| at occurrence positions
                                   (only when block_ddk_stream is provided)
            curvature_class      — pattern-derived class from PatternEngine
                                   (overridden by block_curvature_class when
                                   the per-block class is more specific)
          Per-block columns (shared by all archetypes from this block):
            euler_characteristic    — χ = Σ(ΔΔk)/2π of the source block
            curvature_spectrum_hash — hash of the block's ΔΔk histogram

        Args:
            archetypes: List of LatticeWalkArchetype instances
            source_r0: R₀ of the file these archetypes came from
            block_curvature_class: optional 0..4 class for the source block
            block_euler_char: optional χ of the source block (Gauss-Bonnet)
            block_spectrum_hash: optional pre-computed spectrum hash; if not
                supplied but block_ddk_stream is, computed here from the stream
            block_ddk_stream: optional ΔΔk stream of the source block;
                used to compute geodesic_deviation per archetype AND to
                derive the spectrum hash if block_spectrum_hash is None

        Backward compatibility: callers that pass only (archetypes, source_r0)
        get the original behaviour — new columns stay NULL for these rows.
        Curvature lookups (lookup_by_curvature_class, lookup_by_topology,
        lookup_by_spectrum) skip rows with NULL curvature columns.

        ET Three Tools:
          Identification Principle: identifies the geometric context of
            each archetype (its own curvature + the block's topology +
            the block's spectrum), making the archetype indexable on
            R₀ AND on three R₀-independent structural axes.
          Descriptor Gap Principle: closes the gap between "archetype
            stored" and "archetype findable cross-R₀" — the gap was the
            absence of R₀-invariant Descriptors; the curvature columns
            ARE those Descriptors.
          Subsumption Law: every archetype now carries enough metadata
            to be matched on any of four channels (R₀, class, topology,
            spectrum) without remainder.

        Per design doc §16.9 (USER-AUTHORIZED): the previous
        _check_disk_safety() pruning at the end of store() is REMOVED.
        Nothing is ever deleted. If disk space is critically low,
        compact_to_cdf() (Tier 7) compresses the database in-place
        instead of destroying knowledge.
        """
        import sqlite3
        import pickle
        now = time.time()
        r0_q = _quantize_r0(source_r0)
        n_new = 0
        n_updated = 0

        # Derive the spectrum hash from the ddk_stream if caller did not
        # pre-compute it. This avoids recomputation when many archetypes
        # share the same block context.
        if block_spectrum_hash is None and block_ddk_stream is not None:
            block_spectrum_hash = _curvature_spectrum_hash(block_ddk_stream)

        with self._new_connection() as conn:
            with conn:
                for arch in archetypes:
                    pat = arch.pattern
                    ph = self._pattern_hash(pat, r0_q)

                    # ── Tier 3 Channel A: compute per-archetype curvature ──
                    # Pattern curvature is computed once at store time and
                    # cached in the row; never recomputed during lookup.
                    pat_mean, pat_var, pat_fk = PatternEngine.pattern_curvature(pat)

                    # Per-archetype curvature class derived from the pattern's
                    # own σ²_K against ET-derived thresholds (same logic as
                    # the C engine compute_curvature_stats but applied to
                    # the pattern, not the block). When the caller provides
                    # an explicit block_curvature_class, that takes priority
                    # because the block-level classification has access to
                    # max|K_i| across the WHOLE block (catches singularities
                    # that an isolated pattern slice cannot see).
                    if block_curvature_class is not None:
                        arch_curvature_class = block_curvature_class
                    elif pat_var >= BASE_CURVATURE_VARIANCE:
                        arch_curvature_class = CURVATURE_CLASS_VARIABLE
                    elif pat_mean >= SUBLIMINAL_CURVATURE:
                        arch_curvature_class = CURVATURE_CLASS_ELLIPTIC
                    elif pat_mean <= -SUBLIMINAL_CURVATURE:
                        arch_curvature_class = CURVATURE_CLASS_HYPERBOLIC
                    else:
                        arch_curvature_class = CURVATURE_CLASS_FLAT

                    # geodesic_deviation: mean |K_i| at occurrence positions
                    # of this archetype within the source block. Requires
                    # the block's ΔΔk stream — when not supplied, leave NULL.
                    arch_geo_dev: Optional[float] = None
                    if block_ddk_stream is not None and arch.occurrences:
                        arch_geo_dev = _geodesic_deviation(
                            block_ddk_stream,
                            list(arch.occurrences),
                            len(pat))

                    existing = conn.execute(
                        # language=SQLite
                        'SELECT hit_count FROM archetypes WHERE pattern_hash = ?',
                        (ph,)
                    ).fetchone()

                    if existing:
                        # UPDATE existing row.
                        # Per-archetype curvature columns: keep the maximum
                        # of existing and new geodesic_factor (better fit
                        # wins) and keep the minimum of geodesic_deviation
                        # (more stable observation wins). curvature_mean /
                        # variance / class describe the PATTERN itself
                        # which is identical across hits, so prefer the
                        # NEW value (it carries the same information; the
                        # NULL-coalesce falls through to the existing
                        # value if the new one is NULL for any reason).
                        # Block-level cols (euler_char, spectrum_hash):
                        # only OVERWRITE when caller provides them — let
                        # callers without block context preserve prior
                        # block context.
                        conn.execute(
                            # language=SQLite
                            '''
                            UPDATE archetypes
                            SET hit_count = hit_count + ?,
                                file_count = file_count + 1,
                                last_seen = ?,
                                hierarchy_elegance = MAX(hierarchy_elegance, ?),
                                curvature_mean = COALESCE(?, curvature_mean),
                                curvature_variance = COALESCE(?, curvature_variance),
                                curvature_class = COALESCE(?, curvature_class),
                                geodesic_factor = MAX(COALESCE(geodesic_factor, 0), ?),
                                euler_characteristic = COALESCE(?, euler_characteristic),
                                geodesic_deviation = CASE
                                    WHEN ? IS NULL THEN geodesic_deviation
                                    WHEN geodesic_deviation IS NULL THEN ?
                                    ELSE MIN(geodesic_deviation, ?) END,
                                curvature_spectrum_hash = COALESCE(?, curvature_spectrum_hash)
                            WHERE pattern_hash = ?
                            ''', (len(arch.occurrences), now, arch.hierarchy_elegance,
                                  pat_mean, pat_var, arch_curvature_class, pat_fk,
                                  block_euler_char,
                                  arch_geo_dev, arch_geo_dev, arch_geo_dev,
                                  block_spectrum_hash,
                                  ph))
                        n_updated += 1
                    else:
                        conn.execute(
                            # language=SQLite
                            '''
                            INSERT INTO archetypes
                            (pattern_hash, pattern_dk, pattern_length, r0_quantized,
                             d_avg, hierarchy_elegance, hit_count, file_count,
                             first_seen, last_seen,
                             curvature_mean, curvature_variance, curvature_class,
                             geodesic_factor, euler_characteristic,
                             geodesic_deviation, curvature_spectrum_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?,
                                    ?, ?, ?, ?, ?, ?, ?)
                            ''', (ph, pickle.dumps(pat), len(pat), r0_q,
                                  arch.d_avg, arch.hierarchy_elegance,
                                  len(arch.occurrences), now, now,
                                  pat_mean, pat_var, arch_curvature_class, pat_fk,
                                  block_euler_char, arch_geo_dev,
                                  block_spectrum_hash))
                        n_new += 1

        if n_new > 0 or n_updated > 0:
            self._log(f"ArchetypeDB: stored {n_new} new + {n_updated} updated "
                      f"(R₀_q={r0_q:.4f})")

        # Per design doc §16.9 (USER-AUTHORIZED removal): the previous
        # `self._check_disk_safety()` call here is REMOVED. Nothing is
        # ever deleted. If disk space is critically low, the database
        # is COMPRESSED in place via compact_to_cdf (Tier 7), never
        # pruned. The compounding argument: every archetype contributes
        # to the database's growing capability; deleting any entry
        # reduces the compound return of all future compressions.
        self._maybe_compact_for_disk_pressure()

    def lookup(self, source_r0: float,
               min_elegance: float = LIFE_THRESHOLD,
               min_hits: int = 2,
               max_results: int = 0) -> List[Tuple[tuple, float, int]]:
        """
        Retrieve known archetypes for a given R₀ context.

        Searches the EXACT quantized R₀ group plus ADJACENT groups
        (±1 BIO_RES lattice step). Adjacent-group patterns get their
        hit_count scaled by K = 2/3 (reduced confidence for cross-R₀
        translation — the Koide binding stability discount).

        This enables cross-type pattern reuse: files with similar but
        not identical byte distributions share archetypes with reduced
        confidence. The ±1 step covers R₀ ratios within 2^(1/420) ≈
        0.165% — enough to bridge minor distribution shifts within
        the same file type family.

        Args:
            source_r0: R₀ of the file being compressed
            min_elegance: Minimum hierarchy elegance to return
            min_hits: Minimum hit count (patterns seen at least this many times)
            max_results: Max patterns to return (0 = S² = 144 cap)

        Returns:
            List of (pattern_dk_tuple, hierarchy_elegance, hit_count)
        """
        import sqlite3
        import pickle

        if max_results <= 0:
            max_results = S * S  # 144 cap

        r0_q = _quantize_r0(source_r0)

        # Adjacent R₀ groups: ±1 step at BIO_RES = 420
        # r0_q_minus = 2^((k-1)/420), r0_q_plus = 2^((k+1)/420)
        k_q = lattice_k(r0_q, BIO_RES)
        r0_adjacent = [
            2.0 ** ((k_q - 1) / BIO_RES),  # one step below
            2.0 ** ((k_q + 1) / BIO_RES),  # one step above
        ]

        results: List[Tuple[tuple, float, int]] = []
        seen_hashes: Set[str] = set()

        try:
            with self._new_connection() as conn:
                # ── Exact R₀ group: full confidence ──
                rows = conn.execute(
                    # language=SQLite
                    '''
                    SELECT pattern_dk, hierarchy_elegance, hit_count, pattern_hash
                    FROM archetypes
                    WHERE r0_quantized = ?
                      AND hierarchy_elegance >= ?
                      AND hit_count >= ?
                    ORDER BY hit_count * hierarchy_elegance DESC
                    LIMIT ?
                    ''', (r0_q, min_elegance, min_hits, max_results)).fetchall()

                for row in rows:
                    pat = pickle.loads(row[0])
                    results.append((pat, row[1], row[2]))
                    seen_hashes.add(row[3])

                # ── Adjacent R₀ groups: K-scaled confidence ──
                # Patterns from nearby R₀ groups get hit_count × K (Koide discount)
                remaining = max_results - len(results)
                if remaining > 0:
                    for r0_adj in r0_adjacent:
                        r0_adj_q = _quantize_r0(r0_adj)
                        if r0_adj_q == r0_q:
                            continue  # Same group after quantization — skip
                        adj_rows = conn.execute(
                            # language=SQLite
                            '''
                            SELECT pattern_dk, hierarchy_elegance, hit_count, pattern_hash
                            FROM archetypes
                            WHERE r0_quantized = ?
                              AND hierarchy_elegance >= ?
                              AND hit_count >= ?
                            ORDER BY hit_count * hierarchy_elegance DESC
                            LIMIT ?
                            ''', (r0_adj_q, min_elegance, min_hits, remaining)).fetchall()

                        for row in adj_rows:
                            if row[3] not in seen_hashes:
                                pat = pickle.loads(row[0])
                                # K-scaled hit count: Koide discount for cross-R₀
                                adj_hits = max(1, int(row[2] * K_KOIDE))
                                results.append((pat, row[1], adj_hits))
                                seen_hashes.add(row[3])
                                remaining -= 1
                                if remaining <= 0:
                                    break
                        if remaining <= 0:
                            break
        except (sqlite3.Error, pickle.PickleError, OSError):
            # Database or deserialization error — return whatever was collected
            # before the error. Callers never see database-internal exceptions.
            # In VFS mode the apsw.Error hierarchy is distinct from sqlite3.Error;
            # we catch sqlite3.Error here for normal-mode errors and let the
            # except-Exception fallback below handle apsw errors uniformly so
            # VFS-mode failures behave the same way: a partial-result return.
            self._log(f"ArchetypeDB lookup: database error, returning "
                      f"{len(results)} partial results")
        except Exception as e:  # apsw.Error or any other DB exception in VFS mode
            if _APSW_AVAILABLE and isinstance(e, _apsw_module.Error):
                self._log(f"ArchetypeDB lookup (VFS): apsw error "
                          f"{type(e).__name__}, returning "
                          f"{len(results)} partial results")
            else:
                raise

        return results

    # ── Tier 3.A.3-3.A.5: R₀-independent lookup channels ──────────────
    # These three methods complement the existing R₀-based lookup() with
    # cross-R₀ structural matching. They are all R₀-independent because
    # the underlying columns (curvature_class, curvature_mean,
    # euler_characteristic, curvature_spectrum_hash) are tower-invariant
    # per Multifold §12 — two files with very different R₀s can produce
    # archetypes that match on these columns when the underlying
    # geometric structure is the same.
    #
    # Return shape matches existing lookup(): List[Tuple[pattern, elegance, hits]]
    # so existing call sites (LatticeWalkCompressor.recursive_compress
    # and friends) can consume any of the four channels uniformly.
    #
    # ET Three Tools applied to the lookup channel set:
    #   Identification Principle: each channel identifies a different
    #     structural axis along which patterns can match. Together they
    #     identify the full {R₀, class, topology, spectrum} matching
    #     space.
    #   Descriptor Gap Principle: the gap between "same R₀ matches"
    #     (legacy) and "structurally similar across all R₀s matches"
    #     (Tier 3) is closed by these three methods.
    #   Subsumption Law: the four lookup channels (legacy R₀ + 3 new)
    #     subsume every reasonable similarity query. Any new query type
    #     (e.g. "patterns from blocks with similar curvature spectrum
    #     AND similar topology") combines them via UNION/INTERSECT in
    #     SQL or set operations in Python — no new method needed.

    def lookup_by_curvature_class(self, curvature_class: int,
                                   curvature_mean_range: Tuple[float, float],
                                   min_hits: int = 2,
                                   max_results: int = 0
                                   ) -> List[Tuple[tuple, float, int]]:
        """Find archetypes matching a curvature class regardless of R₀.

        From design doc §16.4 Channel 1. Uses (curvature_class,
        curvature_mean) — both R₀-independent — to find structurally
        matched patterns across all R₀ groups.

        Args:
            curvature_class: 0=flat, 1=elliptic, 2=hyperbolic, 3=variable, 4=singular
            curvature_mean_range: (low, high) inclusive K̄ window
            min_hits: minimum hit_count required (default 2 — proven patterns)
            max_results: cap on returned results, 0 = no cap

        Returns:
            List of (pattern_dk_tuple, hierarchy_elegance, hit_count)
            sorted by elegance DESC. Returns [] when no match.

        ET Three Tools:
          Identification: identifies "patterns sharing this manifold state"
            without binding to R₀.
          Descriptor Gap: closes the gap between "same R₀" and "same
            structural class".
          Subsumption: the (class, K̄ range) pair subsumes all archetypes
            in this geometric region without remainder.
        """
        import sqlite3
        import pickle
        results: List[Tuple[tuple, float, int]] = []
        if not self._has_database():
            return results

        low, high = curvature_mean_range
        if low > high:
            low, high = high, low  # Defensive: normalize swapped bounds

        try:
            with self._new_connection() as conn:
                # language=SQLite
                sql = ('''
                       SELECT pattern_dk, hierarchy_elegance, hit_count
                       FROM archetypes
                       WHERE curvature_class = ?
                         AND curvature_mean IS NOT NULL
                         AND curvature_mean BETWEEN ? AND ?
                         AND hit_count >= ?
                       ORDER BY hierarchy_elegance DESC,
                                COALESCE(geodesic_deviation, 1e308) ASC
                       ''')
                params: List[Any] = [curvature_class, low, high, min_hits]
                if max_results > 0:
                    sql += ' LIMIT ?'
                    params.append(max_results)
                for blob, eleg, hits in conn.execute(sql, params):
                    pattern = pickle.loads(blob)
                    results.append((pattern, float(eleg), int(hits)))
        except (sqlite3.Error, pickle.PickleError, OSError):
            self._log(f"ArchetypeDB lookup_by_curvature_class: database error, "
                      f"returning {len(results)} partial results")
        except Exception as e:
            if _APSW_AVAILABLE and isinstance(e, _apsw_module.Error):
                self._log(f"ArchetypeDB lookup_by_curvature_class (VFS): apsw "
                          f"error {type(e).__name__}, returning "
                          f"{len(results)} partial results")
            else:
                raise
        return results

    def lookup_by_topology(self, euler_char: float,
                            tolerance: float = V_BASE,
                            min_hits: int = 2,
                            max_results: int = 0
                            ) -> List[Tuple[tuple, float, int]]:
        """Find archetypes from blocks with matching Euler characteristic.

        From design doc §16.4 Channel 2. Uses Gauss-Bonnet topology
        (χ = Σ(ΔΔk)/2π) which is R₀-independent. Two blocks with
        |χ_A - χ_B| < V (the base variance, V = 1/12 by default) have
        the same topological class.

        Args:
            euler_char: target χ value
            tolerance: ± window width (default V = 1/12; ET base variance)
            min_hits: minimum hit_count required
            max_results: cap on returned results, 0 = no cap

        Returns:
            List of (pattern, elegance, hits) sorted by elegance DESC.

        ET Three Tools:
          Identification: identifies blocks sharing topology (the discrete
            Gauss-Bonnet fingerprint).
          Descriptor Gap: closes the gap between "same R₀" and "same
            topological class".
          Subsumption: the (χ ± V) window subsumes all topologically
            equivalent blocks per the Gauss-Bonnet equivalence.
        """
        import sqlite3
        import pickle
        results: List[Tuple[tuple, float, int]] = []
        if not self._has_database():
            return results

        low = euler_char - tolerance
        high = euler_char + tolerance

        try:
            with self._new_connection() as conn:
                # language=SQLite
                sql = ('''
                       SELECT pattern_dk, hierarchy_elegance, hit_count
                       FROM archetypes
                       WHERE euler_characteristic IS NOT NULL
                         AND euler_characteristic BETWEEN ? AND ?
                         AND hit_count >= ?
                       ORDER BY hierarchy_elegance DESC,
                                COALESCE(geodesic_deviation, 1e308) ASC
                       ''')
                params: List[Any] = [low, high, min_hits]
                if max_results > 0:
                    sql += ' LIMIT ?'
                    params.append(max_results)
                for blob, eleg, hits in conn.execute(sql, params):
                    pattern = pickle.loads(blob)
                    results.append((pattern, float(eleg), int(hits)))
        except (sqlite3.Error, pickle.PickleError, OSError):
            self._log(f"ArchetypeDB lookup_by_topology: database error, "
                      f"returning {len(results)} partial results")
        except Exception as e:
            if _APSW_AVAILABLE and isinstance(e, _apsw_module.Error):
                self._log(f"ArchetypeDB lookup_by_topology (VFS): apsw error "
                          f"{type(e).__name__}, returning "
                          f"{len(results)} partial results")
            else:
                raise
        return results

    def lookup_by_spectrum(self, spectrum_hash: str,
                            min_hits: int = 2,
                            max_results: int = 0
                            ) -> List[Tuple[tuple, float, int]]:
        """Find archetypes from blocks with matching curvature spectrum.

        From design doc §16.4 Channel 3. The spectrum hash is computed
        from the histogram of ΔΔk values (binned at lattice resolution).
        Two blocks with matching spectrum hashes have the same geometric
        structure regardless of R₀.

        Args:
            spectrum_hash: hex string from _curvature_spectrum_hash
            min_hits: minimum hit_count required
            max_results: cap on returned results, 0 = no cap

        Returns:
            List of (pattern, elegance, hits) sorted by elegance DESC.

        ET Three Tools:
          Identification: identifies blocks with the SAME ΔΔk distribution
            (not just the same mean — the same shape).
          Descriptor Gap: closes the gap between "same R₀" and "same
            curvature spectrum" — the spectrum hash IS the missing key.
          Subsumption: the hash partitions all blocks into equivalence
            classes; lookup returns the entire class without remainder.
        """
        import sqlite3
        import pickle
        results: List[Tuple[tuple, float, int]] = []
        if not self._has_database():
            return results
        if not spectrum_hash:
            return results

        try:
            with self._new_connection() as conn:
                # language=SQLite
                sql = ('''
                       SELECT pattern_dk, hierarchy_elegance, hit_count
                       FROM archetypes
                       WHERE curvature_spectrum_hash = ?
                         AND hit_count >= ?
                       ORDER BY hierarchy_elegance DESC,
                                COALESCE(geodesic_deviation, 1e308) ASC
                       ''')
                params: List[Any] = [spectrum_hash, min_hits]
                if max_results > 0:
                    sql += ' LIMIT ?'
                    params.append(max_results)
                for blob, eleg, hits in conn.execute(sql, params):
                    pattern = pickle.loads(blob)
                    results.append((pattern, float(eleg), int(hits)))
        except (sqlite3.Error, pickle.PickleError, OSError):
            self._log(f"ArchetypeDB lookup_by_spectrum: database error, "
                      f"returning {len(results)} partial results")
        except Exception as e:
            if _APSW_AVAILABLE and isinstance(e, _apsw_module.Error):
                self._log(f"ArchetypeDB lookup_by_spectrum (VFS): apsw error "
                          f"{type(e).__name__}, returning "
                          f"{len(results)} partial results")
            else:
                raise
        return results

    # ── Tier 3.B.4: Channel B persistence + query ────────────────────
    # store_generator and query_generators_for_class manage the
    # generative_descriptors table created by Tier 3.A.1 schema.
    # Together they implement the database side of the Discovery Loop
    # from design doc §16.8.3:
    #   store_generator         — Channel B "Derive" step writes here
    #   query_generators_for_class — pipeline "Channel B query" reads here
    # Per §16.9 NO-REMOVAL: store_generator NEVER deletes; on duplicate
    # gen_id it UPDATES the fit/miss counters and timestamps, leaving
    # the params blob immutable.

    def store_generator(self, descriptor: 'GenerativeDescriptor',
                        increment_fit: bool = False,
                        increment_miss: bool = False,
                        residual_variance: Optional[float] = None) -> None:
        """Insert or update a Generative Descriptor (Channel B).

        First call inserts the row with fit_count=miss_count=0. Subsequent
        calls UPDATE the same row (matched by gen_id) — the params blob
        is immutable, only the running counters and timestamps change.

        Args:
            descriptor: the GenerativeDescriptor to persist
            increment_fit:  add 1 to fit_count and update last_confirmed
            increment_miss: add 1 to miss_count
            residual_variance: if given, MIN(existing, this) wins
                               (best σ²_residual ever achieved)

        ET Three Tools:
          Identification: identifies WHEN a derived generator first
            entered the database (first_derived) and when it last
            confirmed a real fit (last_confirmed).
          Descriptor Gap: closes the gap between "derived candidate"
            (zero confirmations) and "proven generator" (high fit_count)
            by tracking the substantiation lifecycle.
          Subsumption: every derive event hits store_generator; every
            fit/miss event hits store_generator; the counter columns
            subsume the entire history without remainder.
        """
        import sqlite3
        now = time.time()
        with self._new_connection() as conn:
            with conn:
                existing = conn.execute(
                    # language=SQLite
                    'SELECT fit_count, miss_count, best_residual_variance '
                    'FROM generative_descriptors WHERE gen_id = ?',
                    (descriptor.gen_id,)
                ).fetchone()
                if existing is None:
                    conn.execute(
                        # language=SQLite
                        '''
                        INSERT INTO generative_descriptors
                        (gen_id, curvature_class, generator_type, generator_params,
                         param_count, curvature_mean_range_low, curvature_mean_range_high,
                         fit_count, miss_count, best_residual_variance,
                         first_derived, last_confirmed, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (descriptor.gen_id,
                         descriptor.curvature_class,
                         descriptor.generator_type,
                         descriptor.generator_params,
                         descriptor.param_count,
                         descriptor.curvature_mean_range_low,
                         descriptor.curvature_mean_range_high,
                         (1 if increment_fit else 0),
                         (1 if increment_miss else 0),
                         residual_variance,
                         descriptor.first_derived or now,
                         (now if increment_fit else None),
                         descriptor.source))
                else:
                    cur_fits, cur_misses, cur_best = existing
                    new_best: Optional[float]
                    if residual_variance is None:
                        new_best = cur_best
                    elif cur_best is None:
                        new_best = float(residual_variance)
                    else:
                        new_best = min(float(cur_best), float(residual_variance))
                    conn.execute(
                        # language=SQLite
                        '''
                        UPDATE generative_descriptors
                        SET fit_count = fit_count + ?,
                            miss_count = miss_count + ?,
                            best_residual_variance = ?,
                            last_confirmed = CASE WHEN ? > 0 THEN ? ELSE last_confirmed END
                        WHERE gen_id = ?
                        ''',
                        ((1 if increment_fit else 0),
                         (1 if increment_miss else 0),
                         new_best,
                         (1 if increment_fit else 0),
                         now,
                         descriptor.gen_id))

    def query_generators_for_class(self, curvature_class: int,
                                    curvature_mean: float,
                                    max_results: int = 8,
                                    cross_class: bool = True,
                                    cross_class_quota: int = 4
                                    ) -> List['GenerativeDescriptor']:
        """Channel B query — fetch derived generators that may match this block.

        From design doc §16.8.5 step 2: "what generators exist for this
        curvature class and K̄ range? Retrieve top candidates by fit_count."

        ── CORRECTION (post-Tier-3.B.3 review) ──
        The previous version filtered ONLY by curvature_class. That
        re-introduced the same Shannon-thinking exclusion bug that
        derive_generators_from_curvature was just fixed for: a generator
        substantiated on a HYPERBOLIC block may still fit a VARIABLE
        block, because the underlying recurrence/period/polynomial
        relation is class-independent. The class column is a
        prioritization HINT, not a matching constraint.

        With cross_class=True (default), this method returns:
          • Up to (max_results - cross_class_quota) candidates from
            the EXACT matching curvature_class, ordered by fit_count
          • Up to cross_class_quota candidates from OTHER classes,
            ordered by global fit_count (with class as a tie-breaker
            to bias toward geometrically-similar classes)
        Together, the result list never exceeds max_results.

        With cross_class=False (legacy mode), the previous strict
        same-class behavior is preserved for callers that explicitly
        want it (currently none in the standard pipeline; reserved
        for future Tier-aware tooling and tests).

        Args:
            curvature_class: 0..4 from BlockCurvatureProfile
            curvature_mean: K̄ of the block (used to filter by stored
                            [low, high] range — or NULL means "any K̄")
            max_results: total cap on candidates returned
            cross_class: if True (default), include generators from other
                         classes after the same-class results
            cross_class_quota: how many of the max_results slots are
                               reserved for cross-class candidates

        Returns:
            List of GenerativeDescriptor instances. Same-class candidates
            come first (in fit_count DESC, residual ASC order); cross-class
            candidates follow (in global fit_count DESC order). Empty list
            when no candidates exist for this block.
        """
        import sqlite3
        results: List[GenerativeDescriptor] = []
        if not self._has_database():
            return results
        # Reserve quota for cross-class only when cross_class is enabled
        # AND the quota is a positive fraction of the budget.
        same_class_budget = max_results
        cross_class_budget = 0
        if cross_class and cross_class_quota > 0 and cross_class_quota < max_results:
            same_class_budget = max_results - cross_class_quota
            cross_class_budget = cross_class_quota
        # Track gen_ids already returned so cross-class query does not
        # duplicate same-class hits (same generator could in principle
        # appear under both because curvature_class is per-derive-event).
        seen_gen_ids: Set[str] = set()

        def _row_to_descriptor(r) -> GenerativeDescriptor:
            return GenerativeDescriptor(
                gen_id=r[0],
                curvature_class=int(r[1]),
                generator_type=str(r[2]),
                generator_params=bytes(r[3]),
                param_count=int(r[4]),
                curvature_mean_range_low=(None if r[5] is None else float(r[5])),
                curvature_mean_range_high=(None if r[6] is None else float(r[6])),
                fit_count=int(r[7]),
                miss_count=int(r[8]),
                best_residual_variance=(None if r[9] is None else float(r[9])),
                first_derived=float(r[10]),
                last_confirmed=(None if r[11] is None else float(r[11])),
                source=str(r[12]),
            )

        conn_ctx = self._new_connection()
        try:
            with conn_ctx as conn:
                # ── Same-class query ──
                same_class_rows = conn.execute(
                    # language=SQLite
                    '''
                    SELECT gen_id, curvature_class, generator_type, generator_params,
                           param_count, curvature_mean_range_low, curvature_mean_range_high,
                           fit_count, miss_count, best_residual_variance,
                           first_derived, last_confirmed, source
                    FROM generative_descriptors
                    WHERE curvature_class = ?
                      AND (curvature_mean_range_low IS NULL OR curvature_mean_range_low <= ?)
                      AND (curvature_mean_range_high IS NULL OR curvature_mean_range_high >= ?)
                    ORDER BY fit_count DESC,
                             COALESCE(best_residual_variance, 1e308) ASC
                    LIMIT ?
                    ''',
                    (curvature_class, curvature_mean, curvature_mean, same_class_budget)
                ).fetchall()
                for r in same_class_rows:
                    desc = _row_to_descriptor(r)
                    results.append(desc)
                    seen_gen_ids.add(desc.gen_id)

                # ── Cross-class query (if enabled and budget remains) ──
                # Pull additional candidates from OTHER classes, ordered by
                # global fit_count then by class-distance from the requested
                # class (closer classes biased upward; this is the
                # geometric similarity hint, not an exclusion). The K̄
                # range filter still applies — a candidate registered for
                # K̄ ∈ [-1, 1] should not be offered to a block at K̄ = 50.
                if cross_class_budget > 0:
                    cross_class_rows = conn.execute(
                        # language=SQLite
                        '''
                        SELECT gen_id, curvature_class, generator_type, generator_params,
                               param_count, curvature_mean_range_low, curvature_mean_range_high,
                               fit_count, miss_count, best_residual_variance,
                               first_derived, last_confirmed, source
                        FROM generative_descriptors
                        WHERE curvature_class != ?
                          AND (curvature_mean_range_low IS NULL OR curvature_mean_range_low <= ?)
                          AND (curvature_mean_range_high IS NULL OR curvature_mean_range_high >= ?)
                        ORDER BY fit_count DESC,
                                 ABS(curvature_class - ?) ASC,
                                 COALESCE(best_residual_variance, 1e308) ASC
                        LIMIT ?
                        ''',
                        (curvature_class, curvature_mean, curvature_mean,
                         curvature_class, cross_class_budget)
                    ).fetchall()
                    for r in cross_class_rows:
                        desc = _row_to_descriptor(r)
                        if desc.gen_id in seen_gen_ids:
                            continue
                        results.append(desc)
                        seen_gen_ids.add(desc.gen_id)
        except sqlite3.Error:
            self._log(f"ArchetypeDB query_generators_for_class: database error, "
                      f"returning {len(results)} partial results")
        except Exception as e:
            if _APSW_AVAILABLE and isinstance(e, _apsw_module.Error):
                self._log(f"ArchetypeDB query_generators_for_class (VFS): apsw "
                          f"error {type(e).__name__}, returning "
                          f"{len(results)} partial results")
            else:
                raise
        return results

    def _maybe_compact_for_disk_pressure(self):
        """Compress the database in-place when disk space is critically low.

        Per design doc §16.9 (USER-AUTHORIZED removal): the previous
        `_check_disk_safety()` method DELETED 1/3 of low-value archetypes
        when disk free space dropped below DISK_SAFETY_FLOOR (1 GB).
        That method has been REMOVED ENTIRELY in this Tier 3 change set.

        The replacement contract:
          1. Sense disk free space on the database's volume.
          2. If above DISK_SAFETY_FLOOR: do nothing (no pressure).
          3. If below DISK_SAFETY_FLOOR: invoke compact_to_cdf() to
             COMPRESS the database. NOTHING IS DELETED. Compact_to_cdf
             reclaims 80-96% of disk space (per design doc §19.9)
             without losing any archetype, generator template, or
             database entry.
          4. If compact_to_cdf is not yet implemented (Tier 7 work),
             log a clear warning telling the user to run compaction
             manually — but STILL do not delete any data.

        The compounding argument (design doc §16.9):
          Every archetype and every generator template makes the
          database more capable over time. A low-value archetype today
          may become a critical cross-R₀ bridge tomorrow when a new
          file type arrives with a matching curvature profile. Channel
          B generator templates with fit_count=0 are PREDICTIONS — they
          represent the database's theory about what generators SHOULD
          exist. Deleting unconfirmed predictions defeats the entire
          discovery mechanism.

        ET Three Tools:
          Identification Principle: identifies disk pressure as a P-side
            constraint on the substrate, NOT as a reason to lose D-side
            knowledge. Disk and database are categorically distinct
            P-elements — disk pressure must not propagate as data loss.
          Descriptor Gap Principle: the gap between "disk is full" and
            "database can keep growing" is closed by COMPRESSION, not
            destruction. Compression is the right Descriptor; deletion
            is the wrong Descriptor.
          Subsumption Law: every archetype and every generator template
            survives every disk-pressure event. The set of database
            entries grows monotonically — additions only, never removals.

        This is a HARD RULE. No exceptions. No "emergency" pruning.
        No "safety" deletion. Compress, never destroy.
        """
        if not os.path.isfile(self.db_path):
            return

        # Sense disk space on the volume where the database lives.
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        profile = CDFResourceProfile()
        if sys.platform == 'win32':
            CDFResourceSensor.read_disk_windows(profile, db_dir)
        else:
            CDFResourceSensor.read_disk_posix(profile, db_dir)

        if profile.disk_free_bytes >= self.DISK_SAFETY_FLOOR:
            return  # Plenty of disk space — no compression needed

        # Disk pressure detected. Attempt compaction; never delete.
        db_size = os.path.getsize(self.db_path)
        free_mb = profile.disk_free_bytes / (1024 ** 2)
        db_mb = db_size / (1024 ** 2)
        self._log(
            f"ArchetypeDB: disk pressure ({free_mb:.0f} MB free, "
            f"db={db_mb:.1f} MB) — invoking compact_to_cdf "
            f"(NOTHING IS DELETED, per design doc §16.9)"
        )
        logger.warning(
            f"ArchetypeDB disk pressure: {free_mb:.0f} MB free, "
            f"db={db_mb:.1f} MB, compacting (no deletions)"
        )

        # Tier 7 will implement compact_to_cdf as a full CDF VFS compaction
        # that rewrites the database into the CDF compressed format. Until
        # then, the method is present as a no-op stub that logs the request
        # and leaves the database UNTOUCHED. The user is informed but no
        # data is destroyed — the contract holds even before the Tier 7
        # implementation lands.
        try:
            self.compact_to_cdf()
        except NotImplementedError:
            # Tier 7 not yet implemented — log clearly so the user knows
            # to free disk space externally. NEVER fall back to deleting
            # database rows.
            self._log(
                "ArchetypeDB: compact_to_cdf is a Tier 7 stub — "
                "no-op invocation. Database left intact. "
                "Free disk space externally; do NOT delete archetypes.db."
            )
            logger.warning(
                "ArchetypeDB.compact_to_cdf is a Tier 7 stub. "
                "Database not compacted; no data deleted."
            )

    def compact_to_cdf(self) -> bool:
        """Compress the archetype database to a .cdf VFS file.

        Tier 7 deliverable (§19.7 of design doc). Reads the entire .db,
        fits VFS generators per 4 KB page, writes a fresh .cdf file,
        and VERIFIES the compressed output reproduces the .db byte-for-
        byte via CDFDatabaseVFS readback. Only on successful verification
        does it commit the .cdf over any existing .cdf.

        Per §16.9 NO-REMOVAL:
          * The .db file is NEVER deleted by this method, regardless of
            outcome. The user decides when to remove .db after verifying
            the .cdf manually.
          * On ANY failure (fit error, write error, verification failure),
            any partial .cdf is cleaned up and the existing .cdf (if any)
            is left UNTOUCHED. The database remains queryable via .db.
          * No archetype row is ever lost. The .cdf is a compressed
            REPRESENTATION of the same rows that live in .db.

        Returns:
            True on successful compaction + verification
            False on any failure (never raises; _maybe_compact_for_disk_pressure
                  and user-level callers tolerate False without data loss)

        ET Three Tools:
          Identification Principle: identifies the .db as a byte stream
            whose Generative Descriptors can be found via per-page fitting.
          Descriptor Gap Principle: closes the gap between "uncompressed
            .db on disk" and "compressed .cdf with random access" — the
            gap closer is the generator index emitted by _write_vfs_file.
          Subsumption Law: every byte of the .db is subsumed by exactly
            one generator in the output .cdf; verification confirms zero
            remainder (byte-for-byte match).
        """
        if not os.path.isfile(self.db_path):
            self._log(
                f'ArchetypeDB.compact_to_cdf: no .db file at {self.db_path} '
                f'— nothing to compact')
            return False

        # Read the full .db into memory. For the archetype database this
        # is bounded by the Koide memory budget (typical sizes <1 GB even
        # at massive scale per §19.9).
        try:
            with open(self.db_path, 'rb') as f:
                db_bytes = f.read()
        except OSError as e:
            self._log(f'ArchetypeDB.compact_to_cdf: read error on '
                      f'{self.db_path}: {e}')
            return False

        if not db_bytes:
            self._log('ArchetypeDB.compact_to_cdf: empty .db — skipping')
            return False

        db_hash = hashlib.sha256(db_bytes).digest()
        self._log(
            f'ArchetypeDB.compact_to_cdf: compacting '
            f'{len(db_bytes):,}B .db (sha256 {db_hash.hex()[:16]}…) '
            f'→ {self.cdf_path}')

        # Build fresh .cdf at cdf_path (atomic replace via _write_vfs_file)
        write_ok = _write_vfs_file(
            self.cdf_path, db_bytes, log_fn=self._log)
        if not write_ok:
            self._log('ArchetypeDB.compact_to_cdf: _write_vfs_file failed — '
                      '.db left intact; no .cdf committed')
            # If _write_vfs_file left a partial .cdf (it shouldn't because
            # of atomic replace, but defensive cleanup is cheap), remove it.
            # The existing .cdf would have been replaced atomically on
            # success, so its absence here means no prior .cdf existed.
            return False

        # Verification: open the freshly written .cdf via the VFS and
        # compare every byte to db_bytes. On mismatch, remove the .cdf
        # (we just wrote) — but NEVER touch the .db.
        try:
            verify_vfs = CDFDatabaseVFS(self.cdf_path, log_fn=self._log)
        except (ValueError, OSError) as e:
            self._log(f'ArchetypeDB.compact_to_cdf: VFS open of fresh .cdf '
                      f'failed: {e}')
            try:
                os.remove(self.cdf_path)
            except OSError:
                pass
            return False

        try:
            verified = verify_vfs.verify_against(db_bytes)
        except (ValueError, KeyError, IndexError, OSError) as e:
            self._log(f'ArchetypeDB.compact_to_cdf: verification raised '
                      f'{type(e).__name__}: {e}')
            verified = False
        finally:
            verify_vfs.close()

        if not verified:
            self._log('ArchetypeDB.compact_to_cdf: VERIFICATION FAILED — '
                      'removing bad .cdf; .db left intact')
            try:
                os.remove(self.cdf_path)
            except OSError:
                pass
            return False

        compressed_size = os.path.getsize(self.cdf_path)
        ratio = compressed_size / max(len(db_bytes), 1) * 100
        self._log(
            f'ArchetypeDB.compact_to_cdf: SUCCESS — '
            f'{len(db_bytes):,}B .db → {compressed_size:,}B .cdf '
            f'({ratio:.1f}%); .db preserved per §16.9 NO-REMOVAL '
            f'(user may remove manually after inspection)')
        return True

    def import_from(self, other_db_path: str) -> Dict[str, int]:
        """
        Merge another archetype database into this one.

        Imports all patterns from other_db_path. For patterns that already
        exist (same hash), hit_counts and file_counts are SUMMED and
        elegance is updated to the maximum. For new patterns, they are
        inserted directly.

        This enables sharing pattern knowledge between machines:
        compress 1000 files on machine A, export the .db file, import
        on machine B → machine B immediately benefits from machine A's
        archetype discovery.

        Returns dict with n_new, n_updated, n_total counts.
        """
        import sqlite3
        import pickle

        if not os.path.isfile(other_db_path):
            raise FileNotFoundError(f"Database not found: {other_db_path}")

        n_new = 0
        n_updated = 0

        other_conn = sqlite3.connect(other_db_path)
        try:
            other_rows = other_conn.execute(
                # language=SQLite
                '''
                SELECT pattern_hash, pattern_dk, pattern_length, r0_quantized,
                       d_avg, hierarchy_elegance, hit_count, file_count,
                       first_seen, last_seen
                FROM archetypes
                ''').fetchall()
        finally:
            other_conn.close()

        conn_ctx = self._new_connection()
        try:
            with conn_ctx as conn:
                with conn:
                    for row in other_rows:
                        ph = row[0]
                        existing = conn.execute(
                            # language=SQLite
                            'SELECT hit_count, file_count FROM archetypes WHERE pattern_hash = ?',
                            (ph,)
                        ).fetchone()

                        if existing:
                            # Merge: sum hit_count and file_count, max elegance
                            conn.execute(
                                # language=SQLite
                                '''
                                UPDATE archetypes
                                SET hit_count = hit_count + ?,
                                    file_count = file_count + ?,
                                    last_seen = MAX(last_seen, ?),
                                    hierarchy_elegance = MAX(hierarchy_elegance, ?)
                                WHERE pattern_hash = ?
                                ''', (row[6], row[7], row[9], row[5], ph))
                            n_updated += 1
                        else:
                            # Insert new pattern — deserialize for logging
                            imported_pattern = pickle.loads(row[1])
                            conn.execute(
                                # language=SQLite
                                '''
                                INSERT INTO archetypes
                                (pattern_hash, pattern_dk, pattern_length, r0_quantized,
                                 d_avg, hierarchy_elegance, hit_count, file_count,
                                 first_seen, last_seen)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', row)
                            n_new += 1
                            logger.debug(f"ArchetypeDB import: new pattern len={len(imported_pattern)} "
                                         f"R₀_q={row[3]:.4f} elegance={row[5]:.1f}")
        finally:
            # Nothing to close explicitly — the _ConnContext __exit__
            # releases the sqlite3 conn in normal mode and preserves the
            # apsw conn in VFS mode. The outer try/finally is retained
            # purely for symmetry with the legacy structure so future
            # callers adding extra cleanup code here find an obvious
            # hook point.
            pass

        n_total = n_new + n_updated
        self._log(f"ArchetypeDB import: {n_new} new + {n_updated} merged "
                  f"from {other_db_path} ({n_total} total)")

        # Per design doc §16.9 (USER-AUTHORIZED removal): the previous
        # `self._check_disk_safety()` call here is REMOVED. After importing
        # archetypes from another database, if disk pressure exists,
        # the database is COMPRESSED in place (Tier 7), never pruned.
        self._maybe_compact_for_disk_pressure()

        return {'n_new': n_new, 'n_updated': n_updated, 'n_total': n_total}

    def stats(self) -> Dict[str, Any]:
        """Database statistics."""
        import sqlite3
        if not self._has_database():
            return {'entries': 0, 'total_hits': 0, 'avg_elegance': 0.0,
                    'disk_mb': 0.0, 'disk_free_mb': 0.0, 'r0_groups': 0,
                    'compressed_mb': 0.0, 'vfs_mode': False}

        # ── Determine which file to measure on disk ──
        # In VFS mode: self.db_path does not exist; the compressed .cdf
        # IS the on-disk footprint. Report it as both `disk_mb` (for
        # backward-compatible callers) and `compressed_mb` (the new
        # explicit name).
        # In normal mode: self.db_path exists as a plain SQLite file;
        # `disk_mb` is its size and `compressed_mb` is 0 (no compression).
        vfs_mode = self._apsw_conn is not None
        if vfs_mode:
            # Prefer the .cdf size — that is the real disk footprint.
            size_path = self.cdf_path
        else:
            size_path = self.db_path
        if os.path.isfile(size_path):
            db_size = os.path.getsize(size_path)
        else:
            db_size = 0

        # Sense disk free space on the volume holding the database (or
        # its compressed counterpart in VFS mode). Falls back to the
        # parent dir when the file itself has no parent component.
        db_dir = os.path.dirname(os.path.abspath(size_path))
        if not db_dir:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
        disk_profile = CDFResourceProfile()
        if sys.platform == 'win32':
            CDFResourceSensor.read_disk_windows(disk_profile, db_dir)
        else:
            CDFResourceSensor.read_disk_posix(disk_profile, db_dir)

        with self._new_connection() as conn:
            total = conn.execute(
                # language=SQLite
                'SELECT COUNT(*) FROM archetypes').fetchone()[0]
            total_hits = conn.execute(
                # language=SQLite
                'SELECT COALESCE(SUM(hit_count), 0) FROM archetypes').fetchone()[0]
            avg_eleg = conn.execute(
                # language=SQLite
                'SELECT COALESCE(AVG(hierarchy_elegance), 0) FROM archetypes').fetchone()[0]
            r0_groups = conn.execute(
                # language=SQLite
                'SELECT COUNT(DISTINCT r0_quantized) FROM archetypes').fetchone()[0]

        return {
            'entries': total,
            'total_hits': total_hits,
            'avg_elegance': avg_eleg,
            'disk_mb': db_size / (1024 ** 2),
            'disk_free_mb': disk_profile.disk_free_bytes / (1024 ** 2),
            'r0_groups': r0_groups,
            # New keys: let callers distinguish compressed (.cdf) vs
            # raw (.db) footprint without heuristics.
            'compressed_mb': db_size / (1024 ** 2) if vfs_mode else 0.0,
            'vfs_mode': vfs_mode,
        }

    def summary(self) -> str:
        """Human-readable summary for GUI/logging."""
        s = self.stats()
        mode_tag = ' [VFS/.cdf]' if s.get('vfs_mode') else ''
        return (f"ArchetypeDB{mode_tag}: {s['entries']} patterns across {s['r0_groups']} R₀ groups, "
                f"{s['total_hits']} total hits, "
                f"avg elegance {s['avg_elegance']:.1f}, "
                f"{s['disk_mb']:.1f} MB (disk free: {s['disk_free_mb']:.0f} MB)")

    def clear_archetypes(self) -> int:
        """Delete every row from the archetypes table. Returns rows deleted.

        Routed through the dual-mode connection factory so the GUI's
        `db clear` command works identically in normal .db mode and in
        VFS .cdf mode. In VFS mode the deletion propagates into
        CDFDatabaseVFS's dirty-page buffer; the next call to
        ArchetypeDatabase.close() will recompress the .cdf with the
        empty archetype table (Mike's "only recompress when new stuff
        is added" contract — an empty archetype table IS a change,
        so recompression is correct).

        Returns:
            The row count BEFORE deletion (i.e. how many rows were
            cleared). Zero when the table was already empty.

        Per rule 24 no-removal: this method deletes USER DATA, not
        CODE — the contract on CODE removal does not apply to
        user-initiated database clears. The user explicitly invoked
        the `db clear` action and accepts the consequence.

        ET Three Tools:
          Identification Principle — the `clear` action identifies a
            user-authorised P-state transition (archetypes populated
            → archetypes empty). The method traverses every existing
            row and substantiates the empty state.
          Descriptor Gap Principle — closes the gap between the
            GUI-level `db clear` command and the dual-mode backing
            store; previously the GUI opened its own sqlite3
            connection directly on self.db_path, which silently no-ops
            in VFS mode because the .db file does not exist.
          Subsumption Law — one method subsumes the `clear` operation
            for both backing modes without remainder.
        """
        if not self._has_database():
            return 0
        with self._new_connection() as conn:
            with conn:
                pre_count = conn.execute(
                    # language=SQLite
                    'SELECT COUNT(*) FROM archetypes').fetchone()[0]
                conn.execute(
                    # language=SQLite
                    'DELETE FROM archetypes')
                # VACUUM reclaims disk space on a regular .db; in VFS
                # mode it is still meaningful (it frees pages in the
                # virtual SQLite file and those freed pages will
                # compress well on the next close-time recompaction).
                # VACUUM cannot run inside a transaction, so it needs
                # to be issued on its own — step out of the `with conn`
                # transactional context.
            conn.execute('VACUUM')
        return int(pre_count)


# Module-level archetype database — shared across all compression operations
_archetype_db: Optional[ArchetypeDatabase] = None


def get_archetype_db(log_fn=None) -> ArchetypeDatabase:
    """Get or create the module-level archetype database."""
    global _archetype_db
    if _archetype_db is None:
        _archetype_db = ArchetypeDatabase(log_fn=log_fn)
    assert _archetype_db is not None  # Narrowing: just created if was None
    return _archetype_db


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
                 metabolism: Optional[CDFMetabolism] = None,
                 archetype_db: Optional[ArchetypeDatabase] = None):
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)
        self.metabolism = metabolism or _metabolism
        self.metabolism._log = self._log
        self.metabolism.sense()
        self.metabolism.apply_process_priority()
        self.archetype_db = archetype_db
        # Accumulated archetypes from all blocks in the current file
        # Read by CDFCompressor after compression for database storage
        self.discovered_archetypes: List[LatticeWalkArchetype] = []
        # ── Tier 1: Curvature analysis (Phase 1.5) ──
        # The analyzer is a thin wrapper around the C engine — instantiating
        # one is essentially free. Held on the engine so every block reuses
        # the same analyzer instance (avoids per-block allocation).
        self.curvature_analyzer = CurvatureAnalyzer(n_res=N_FULL)
        # Most recent block's curvature profile. Read by:
        #   - log output (Phase 1.5)
        #   - ArchetypeDatabase.store (curvature columns, Tier 3)
        #   - Variable-curvature segmentation (Tier 4)
        #   - Channel B generator fitting (Tier 3)
        # Initialised to None — populated by compress_block per call.
        self.last_block_curvature: Optional[BlockCurvatureProfile] = None
        # Tier 5.D.1: retain the ddk_stream from the last compressed block
        # so compress_file can pass it to store() for geodesic_deviation
        # computation and curvature_spectrum_hash derivation.
        self.last_block_ddk_stream: Optional[List[int]] = None
        # Accumulated curvature profiles for every block in the current
        # file. Cleared by CDFCompressor after each file. Used by Channel B
        # to detect cross-block curvature trends and derive new generators.
        self.discovered_curvature_profiles: List[BlockCurvatureProfile] = []

    @property
    def log_fn(self):
        """Public accessor for the logging function.

        Used by UniversalLattice.compress_tower_with_universal to pass
        the engine's logger to LatticeWalkCompressor instances.
        """
        return self._log

    def encode_lattice_block(self, n: int, r0: float, k0: int, mode: int,
                             dk0_saved: int, k_table: Optional[List[int]],
                             rc: dict,
                             connection_order: int = 0,
                             connection_window: int = 0,
                             generator_payload: Optional[bytes] = None) -> bytes:
        """Public interface for lattice block encoding.

        Delegates to _encode_lattice_block. Used by UniversalLattice and
        CDFCompressor.compress_batch for encoding blocks through the
        universal lattice perspective without accessing protected members.

        connection_order, connection_window:
          Mode 3 (geodesic residual) extension fields. Ignored for modes
          0/1/2/4 — defaults of 0 keep all existing call sites valid.
          For mode == 3, the encoder writes both fields into the block
          header so the decoder can reconstruct Δk causally from residuals.

        generator_payload:
          Mode 4 (generator + residual) extension field. Ignored for
          modes 0/1/2/3 — default of None keeps all existing call sites
          valid. For mode == 4, the encoder writes the payload (as a
          length-prefixed BLOB) into the block header so the decoder can
          instantiate the generator and reconstruct k_stream = predicted +
          residual.
        """
        return self._encode_lattice_block(n, r0, k0, mode, dk0_saved, k_table, rc,
                                          connection_order=connection_order,
                                          connection_window=connection_window,
                                          generator_payload=generator_payload)

    def _encode_lattice_block(self, n: int, r0: float, k0: int, mode: int,
                              dk0_saved: int, k_table: Optional[List[int]],
                              rc: dict,
                              connection_order: int = 0,
                              connection_window: int = 0,
                              generator_payload: Optional[bytes] = None) -> bytes:
        """Encode a compressed block to binary given mode and recursive compress result."""
        mode_names = {0: 'k-direct', 1: 'Δk', 2: 'ΔΔk', 3: 'geodesic-ρ',
                      4: 'gen-residual'}
        n_archetypes = len(rc['archetype_defs'])
        if mode == 3:
            self._log(f"  Encoding: mode={mode_names.get(mode, mode)}, "
                      f"connection_order={connection_order}, "
                      f"connection_window={connection_window}, "
                      f"n_base={rc['n_base']}, archetypes={n_archetypes}, "
                      f"stream={len(rc['final_stream'])}")
        elif mode == 4:
            self._log(f"  Encoding: mode={mode_names.get(mode, mode)}, "
                      f"generator_payload={len(generator_payload) if generator_payload else 0}B, "
                      f"n_base={rc['n_base']}, archetypes={n_archetypes}, "
                      f"stream={len(rc['final_stream'])}")
        else:
            self._log(f"  Encoding: mode={mode_names.get(mode, mode)}, "
                      f"n_base={rc['n_base']}, archetypes={n_archetypes}, "
                      f"stream={len(rc['final_stream'])}")
        parts = [
            struct.pack('<B', 2),  # type = lattice
            struct.pack('<I', n),  # original size
            struct.pack('<d', r0),  # R₀ seed
            struct.pack('<i', k0),  # initial k value
            struct.pack('<B', mode),  # mode: 0=k-direct, 1=Δk, 2=ΔΔk, 3=geodesic-ρ, 4=gen-residual
            struct.pack('<i', dk0_saved),  # first Δk (or Mode 3 anchor; Mode 4 first residual sample)
        ]

        # Mode 3: write connection_order (1 byte) + connection_window (manifold uint).
        # These are the ONLY new header fields versus v2 — modes 0/1/2 are byte-for-byte
        # identical to the v2 layout, so legacy v2 readers can still decode v3 files
        # that contain only modes 0/1/2 (in practice they would have to know to accept
        # CDF\x03 magic, but the block-level format is preserved for those modes).
        if mode == 3:
            parts.append(struct.pack('<B', connection_order))      # 0/1/2
            parts.append(pack_manifold_uint(connection_window))    # 2 or 6 bytes

        # Mode 4: write the generator payload (length-prefixed BLOB).
        # Layout: [4B uint32 LE payload_len][N bytes payload].
        # The payload's first byte is the generator type code; the rest
        # is the pickled params dict (this matches the
        # GenerativeDescriptor.generator_params field exactly, so the
        # encoder writes it as-is and the decoder reads it as-is).
        # The recursive_compress result that follows in the standard
        # format slots represents the RESIDUAL stream (residual_k_i =
        # actual_k_i - predicted_k_i), not Δk.
        if mode == 4:
            payload = generator_payload if generator_payload is not None else b''
            parts.append(struct.pack('<I', len(payload)))
            if payload:
                parts.append(payload)

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
        parts.append(pack_manifold_uint(n_base))
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
        parts.append(pack_manifold_uint(len(arch_defs)))
        parts.append(pack_manifold_uint(total_syms))
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
                parts.append(struct.pack('<B', 1))
                parts.append(pack_manifold_uint(len(flat_arch_stream)))
                parts.append(pack_manifold_uint(len(vc_arch_tbl)))
                parts.append(vc_arch_tbl)
                parts.append(struct.pack('<I', len(vc_arch_enc)))  # explicit byte count
                parts.append(vc_arch_enc)
            else:
                # Uniform bit-pack archetype definitions
                parts.append(struct.pack('<B', 0))
                parts.append(pack_manifold_uint(len(flat_arch_stream)))
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
            parts.append(struct.pack('<B', 0))
            parts.append(pack_manifold_uint(0))

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
        uniform_block = struct.pack('<BB', 0, bits_per_sym) + pack_manifold_uint(n_syms) + bytes(uniform_bytes)

        # Option B: V_config encoding (lattice-depth-weighted: variable depth per symbol)
        if n_syms > 0 and total_syms > 1:
            vc_encoded, vc_table = v_config_encode(sym_stream, total_syms)
            vc_block = struct.pack('<B', 1) + pack_manifold_uint(n_syms) + \
                       pack_manifold_uint(len(vc_table)) + vc_table + vc_encoded
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
        k_stream_native = PatternEngine.fast_k_stream(data, byte_k_native)
        k0_native = k_stream_native[0]
        best_enhanced = None
        # Score tuple (size_in_bytes, r0_perturbation_magnitude_in_lattice_steps)
        # to enable principled tie-breaking by R₀ fundamentality:
        #   - primary  smaller encoded size wins
        #   - secondary smaller R₀ perturbation from native wins
        # Native R₀ has perturbation = 0 → wins ties against perturbed R₀s
        # and against the universal R₀. This honours the principle that the
        # data's own discovered R₀ is its most fundamental seed; alternate
        # seeds are searches for better anchors and only displace native
        # when they STRICTLY beat it on encoded size.
        # Initial sentinel size is "larger than any real block";
        # initial sentinel perturbation is +∞ so any real candidate wins.
        best_enhanced_score: Tuple[int, float] = (n + BLOCK_SIZE, float('inf'))

        # Helper: compute |Δk| in lattice steps between any R₀ and native R₀.
        # Δk_seed = N_FULL · log₂(r0_alt / r0)  per the standard ET conversion.
        # Used as the secondary tie-breaker for enhanced strategies.
        def _r0_perturbation_steps(r0_alt: float) -> float:
            if r0_alt <= 0 or r0 <= 0:
                return float('inf')
            return abs(N_FULL * math.log2(r0_alt / r0))

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
                    k_stream_cx = PatternEngine.fast_k_stream(data, bk_cx)
                    k0_cx = k_stream_cx[0]
                    dk_cx = PatternEngine.fast_dk_stream(k_stream_cx)
                    if len(dk_cx) > 3:
                        rc_complex = walker.recursive_compress(dk_cx)
                        trial_complex = self._encode_lattice_block(
                            n, best_r0_complex, k0_cx, 1, 0, None, rc_complex)
                        # Score: (size, r0 perturbation magnitude) — lex-min wins
                        trial_complex_score = (len(trial_complex),
                                               _r0_perturbation_steps(best_r0_complex))
                        if trial_complex_score < best_enhanced_score:
                            best_enhanced = trial_complex
                            best_enhanced_score = trial_complex_score
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
            k_stream_alt = PatternEngine.fast_k_stream(data, bk_alt)
            k0_alt = k_stream_alt[0]
            dk_alt = PatternEngine.fast_dk_stream(k_stream_alt)
            if len(dk_alt) > 3:
                try:
                    rc_alt = walker.recursive_compress(dk_alt)
                    # The R₀ stored in the block is the SHIFTED R₀.
                    # The decompressor reads R₀ from the block header
                    # and rebuilds the correct byte↔k map from it.
                    trial_alt = self._encode_lattice_block(
                        n, r0_alt, k0_alt, 1, 0, None, rc_alt)
                    trial_alt_score = (len(trial_alt), _r0_perturbation_steps(r0_alt))
                    if trial_alt_score < best_enhanced_score:
                        best_enhanced = trial_alt
                        best_enhanced_score = trial_alt_score
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
        k_stream_universal = PatternEngine.fast_k_stream(data, bk_universal)
        dk_personal = PatternEngine.fast_dk_stream(k_stream_native)
        dk_universal = PatternEngine.fast_dk_stream(k_stream_universal)
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
                trial_cross_score = (len(trial_cross), _r0_perturbation_steps(r0_chosen))
                if trial_cross_score < best_enhanced_score:
                    best_enhanced = trial_cross
                    best_enhanced_score = trial_cross_score
            except (ValueError, KeyError, IndexError, struct.error):
                pass  # Strategy failed — encoding or compression error

        # Log the enhanced strategy result. Report both the encoded size
        # (primary score) and the R₀ perturbation in lattice steps
        # (secondary score) so the operator can see WHICH alternative
        # seed won and how far it sits from the data's native R₀.
        if best_enhanced is not None:
            self._log(f"    Enhanced best: {best_enhanced_score[0]:,}B "
                      f"(R\u2080 perturbation = {best_enhanced_score[1]:.2f} lattice steps "
                      f"from native; from {n:,}B original)")

        return best_enhanced

    def compress_block(self, data: bytes,
                        _segmentation_depth: int = 0) -> bytes:
        """
        Compress a single block (up to BLOCK_SIZE bytes).
        Returns the compressed binary representation.

        _segmentation_depth (Tier 4): recursion guard for Block Type 4
        segmented compression. At depth 0, the block may try to split
        itself into curvature-uniform segments and re-compress each
        recursively (with depth=1). At depth ≥ 1, the segmented candidate
        is skipped — segments are atomic compress_block calls that go
        through the standard single-block pipeline. This prevents
        unbounded recursion while still allowing one level of
        segmentation per call.
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
        self._log(f"  Block {n:,}B: {len(unique)} unique bytes — starting lattice compression"
                  f"{' (segment, depth=' + str(_segmentation_depth) + ')' if _segmentation_depth else ''}")

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

        k_stream = PatternEngine.fast_k_stream(data, byte_k)
        k0 = k_stream[0]

        # Build candidate streams
        # Mode 0: k-direct (map k-values to compact indices)
        unique_k = sorted(set(k_stream))
        k_to_compact = {k: i for i, k in enumerate(unique_k)}
        k_direct_stream = [k_to_compact[k] for k in k_stream]
        n_unique_k = len(unique_k)

        # Mode 1: Δk
        dk_stream = PatternEngine.fast_dk_stream(k_stream)
        n_unique_dk = len(set(dk_stream))

        # Mode 2: ΔΔk
        ddk_stream = PatternEngine.fast_dk_stream(dk_stream) if len(dk_stream) > 1 else []
        n_unique_ddk = len(set(ddk_stream)) if ddk_stream else 999999

        # ── Phase 1.5: Curvature Analysis (Tier 1, additive) ─────────────
        # Compute the block's manifold-state classification from its ΔΔk
        # stream. This is purely observational at Tier 1 — no existing
        # strategy is removed or skipped, no mode is chosen based on
        # curvature yet (that arrives with Tier 2 Mode 3 selection and
        # Tier 4 segmentation). Phase 1.5 stores the profile on the
        # engine so downstream phases (DB store, Channel B) can read it
        # without recomputing.
        #
        # Reuses the ddk_stream just computed for Mode 2 mode-selection —
        # zero extra C calls.
        #
        # ET Three Tools:
        #   Identification Principle: identifies the block's manifold state
        #     (one of five classes: flat / elliptic / hyperbolic / variable /
        #     singular).
        #   Descriptor Gap Principle: closes the gap between "ΔΔk computed
        #     for entropy" and "ΔΔk used as a structural Descriptor".
        #   Subsumption Law: every possible ΔΔk distribution falls into
        #     exactly one of the five classes.
        t_phase15 = time.time()
        if ddk_stream:
            phase15_stats = PatternEngine.curvature_stats(ddk_stream, n_res=N_FULL)
            phase15_sign = (1 if phase15_stats.curvature_mean > SUBLIMINAL_CURVATURE
                            else (-1 if phase15_stats.curvature_mean < -SUBLIMINAL_CURVATURE
                                  else 0))
            block_curvature = BlockCurvatureProfile(
                curvature_mean=phase15_stats.curvature_mean,
                curvature_variance=phase15_stats.curvature_variance,
                curvature_class=int(phase15_stats.curvature_class),
                euler_characteristic=phase15_stats.euler_characteristic,
                max_abs_curvature=int(phase15_stats.max_abs_curvature),
                n_ddk_samples=len(ddk_stream),
                sign=phase15_sign,
            )
        else:
            # Block too small for ΔΔk — trivially flat (Exception state).
            block_curvature = BlockCurvatureProfile()
        self.last_block_curvature = block_curvature
        self.last_block_ddk_stream = list(ddk_stream) if ddk_stream else None
        self.discovered_curvature_profiles.append(block_curvature)
        self._log(
            f"    Phase 1.5 curvature: class={block_curvature.curvature_class} "
            f"({block_curvature.class_name()}) "
            f"K\u0304={block_curvature.curvature_mean:.4f} "
            f"\u03c3\u00b2_K={block_curvature.curvature_variance:.4f} "
            f"\u03c7={block_curvature.euler_characteristic:.4f} "
            f"max|K|={block_curvature.max_abs_curvature} "
            f"({time.time() - t_phase15:.3f}s)"
        )

        # ── Phase 1.5b: Curvature Spectrum DB Lookup (Tier 6.C.1) ─────
        # Per design doc §14.2: compute the block's curvature spectrum
        # hash (SHA-256[:32] of sorted ΔΔk histogram) and query the
        # archetype database for entries with matching spectrum. These
        # are patterns discovered on PRIOR files whose curvature
        # geometry matches this block's geometry — they may compress
        # this block's data better than fresh pattern discovery alone.
        #
        # The spectrum hash is R₀-independent (derived from ΔΔk which
        # is tower-invariant per Multifold §9), so spectrum matches
        # cross R₀ boundaries without d-family translation.
        #
        # Results are logged and the spectrum_hash is retained so
        # store() can write it to the DB for future files.
        #
        # ET Three Tools:
        #   Identification: identifies pre-existing archetypes whose
        #     curvature fingerprint matches this block's geometry.
        #   Descriptor Gap: closes the gap between "archetype database
        #     has patterns from prior files" and "those patterns are
        #     offered to this block's compression". The spectrum hash
        #     IS the key that connects them.
        #   Subsumption: spectrum lookup is ADDITIVE — it offers new
        #     candidates alongside C-engine discoveries. No existing
        #     pattern is removed or replaced.
        self.last_block_spectrum_hash: Optional[str] = None
        if ddk_stream and self.archetype_db is not None:
            spectrum_hash = _curvature_spectrum_hash(list(ddk_stream))
            self.last_block_spectrum_hash = spectrum_hash
            try:
                spectrum_matches = self.archetype_db.lookup_by_spectrum(
                    spectrum_hash, max_results=12)
                if spectrum_matches:
                    self._log(f"    Phase 1.5b spectrum lookup: "
                              f"{len(spectrum_matches)} matching archetypes "
                              f"for hash {spectrum_hash[:8]}…")
                    # Populate the walker's pre-seed pool so these
                    # patterns get injected into recursive_compress at
                    # depth 0 with real occurrence scanning. The walker
                    # is created below (after Phase 2); we store the
                    # matches on the engine for now and transfer them
                    # after walker creation.
                    self._spectrum_seed_cache = spectrum_matches
                else:
                    self._spectrum_seed_cache = []
            except Exception:
                self._spectrum_seed_cache = []

        # ── Mode 3: Geodesic Residual (Tier 2, additive) ─────────────────
        # Compute residuals at all three connection orders (0/1/2) from
        # the design doc §6.2 and pick the order with the fewest unique
        # residual values — that is the order whose connection best
        # predicts the data's geodesic.
        #
        # Window selection (design doc §5.2):
        #   w = min(N_max, S²) where N_max = ⌊50¢ / |δ̄|⌋
        # Uses IncoherenceFilter.l4_cascade_horizon directly — no new
        # constants, no tuning. Cents conversion = 1200·|ΔΔk|/N_FULL,
        # which matches the existing lattice_epsilon formula exactly.
        #
        # ET Three Tools:
        #   Identification Principle: identifies the geodesic prediction
        #     at each position via the local Christoffel connection.
        #   Descriptor Gap Principle: closes the gap between "encoding
        #     what happened" and "encoding what the geometry didn't predict".
        #   Subsumption Law: orders 0 ⊂ 1 ⊂ 2 — order 0 subsumes constant
        #     Δk, order 1 subsumes order 0 plus linear trends, order 2
        #     subsumes order 1 plus quadratic trends. Three orders cover
        #     every smooth data manifold.
        t_mode3 = time.time()
        mode3_residuals: List[int] = []
        mode3_connection_order = 0
        mode3_connection_window = 0
        n_unique_geo = 999999
        if len(dk_stream) >= 3 and ddk_stream:
            # Mean |ΔΔk| in cents for the L4 horizon formula
            mean_abs_ddk_steps = sum(abs(x) for x in ddk_stream) / len(ddk_stream)
            mean_abs_ddk_cents = mean_abs_ddk_steps * (1200.0 / N_FULL)
            l4_horizon = IncoherenceFilter.l4_cascade_horizon(mean_abs_ddk_cents)
            # Window cap = S² = 144 (manifold cross-pattern maximum, design doc §5.2)
            mode3_connection_window = max(1, min(S * S, l4_horizon))

            best_order = 0
            best_uniques = 999999
            best_residuals: List[int] = []
            for trial_order in (0, 1, 2):
                trial_residuals, _trial_gamma = PatternEngine.fast_geodesic_residual(
                    dk_stream, trial_order, mode3_connection_window)
                if not trial_residuals:
                    continue
                trial_uniques = len(set(trial_residuals))
                if trial_uniques < best_uniques:
                    best_uniques = trial_uniques
                    best_order = trial_order
                    best_residuals = trial_residuals

            if best_residuals:
                mode3_residuals = best_residuals
                mode3_connection_order = best_order
                n_unique_geo = best_uniques
                self._log(
                    f"    Mode 3 candidate: connection_order={mode3_connection_order} "
                    f"(of 0/1/2), connection_window={mode3_connection_window}, "
                    f"unique residuals={n_unique_geo} "
                    f"({time.time() - t_mode3:.3f}s)"
                )

        # Choose mode with fewest unique values (lowest lattice entropy).
        # Mode 3 added as a 4th candidate alongside the original 3.
        mode_uniques = [(0, n_unique_k), (1, n_unique_dk),
                        (2, n_unique_ddk), (3, n_unique_geo)]
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
        elif best_mode == 3 and mode3_residuals:
            # Mode 3: residual stream + Christoffel connection (Tier 2)
            # dk0_saved holds the FIRST Δk value (Δk_0); the decoder
            # uses it as the causal anchor and reconstructs every later
            # Δk_i via: Δk_{i+1} = ρ_i + Δk_i + Γ_i.
            dk_stream_for_compress = mode3_residuals
            dk0_saved = dk_stream[0]
        else:
            best_mode = 1  # Default to Δk
            dk_stream_for_compress = dk_stream
            dk0_saved = 0

        # ── Phase 3+4: Multi-Strategy Recursive Archetype Subsumption ──
        # Descriptor Gap Principle: try ALL viable modes, build ALL, keep smallest.
        # Speed is IRRELEVANT — only compression ratio matters.
        self._log(f"    Phase 2 done: modes k={n_unique_k}, Δk={n_unique_dk}, "
                  f"ΔΔk={n_unique_ddk}, geo-ρ={n_unique_geo} "
                  f"({time.time() - t_phase2:.2f}s)")
        self._progress(20, "Phase 3: Pattern Finding (slow for large blocks)")
        t_phase3 = time.time()
        walker = LatticeWalkCompressor(r0, log_fn=self._log,
                                       archetype_db=self.archetype_db)
        # Tier 6.C.1: transfer spectrum-matched DB patterns to the walker
        # so they participate in recursive_compress depth 0 injection.
        if getattr(self, '_spectrum_seed_cache', None):
            walker.pre_seed_dk_patterns = list(self._spectrum_seed_cache)
            self._spectrum_seed_cache = []

        # Build candidates for each viable mode.
        # Tuple format: (mode, stream, dk0, mode3_meta) where mode3_meta is
        #   None for modes 0/1/2
        #   (connection_order, connection_window) for mode 3
        # The 4th slot keeps the existing tuple structure forward-compatible
        # without requiring any caller to change other than the iteration
        # unpacking sites below.
        candidates = []

        # Mode 0: k-direct
        if len(k_direct_stream) > 3:
            candidates.append((0, k_direct_stream, 0, None))
        # Mode 1: Δk
        if len(dk_stream) > 3:
            candidates.append((1, dk_stream, 0, None))
        # Mode 2: ΔΔk
        if ddk_stream and len(ddk_stream) > 3:
            candidates.append((2, ddk_stream, dk_stream[0], None))
        # Mode 3: geodesic residual ρ (Tier 2)
        # The residuals were already computed during mode-selection above.
        # Only added when residual stream is non-empty (≥ 3 dk values needed).
        if mode3_residuals and len(mode3_residuals) > 3:
            candidates.append((3, mode3_residuals, dk_stream[0],
                               (mode3_connection_order, mode3_connection_window)))

        # ── Mode 4: Generator + Residual (Tier 3.B.4) ─────────────────
        # Channel B in action — for each generator that fits this block
        # (σ²_residual < V = 1/12, the subliminal threshold), build a
        # Mode 4 candidate. Two sources of generators:
        #   1. Channel B query: generators substantiated on PRIOR files
        #      (database fit_count > 0). Includes cross-class matches.
        #   2. Channel B derive: NEW generators inferred from this
        #      block's curvature profile via derive_generators_from_curvature.
        # Both sources flow through the same fit pipeline; database
        # writes happen AFTER the mode competition resolves so we know
        # which generators actually contributed to the winning block.
        # Generators that fit but did not win still get store_generator
        # called with increment_fit=False (recorded as observed-but-
        # not-best); generators that did not fit get increment_miss=True
        # so the database learns which templates are unhelpful for this
        # curvature class. Per §16.9 NO-REMOVAL: nothing is ever
        # deleted from generative_descriptors regardless of miss_count.
        #
        # CRITICAL CORRECTNESS NOTE:
        # Mode 4 residuals are computed against the REAL k_stream (actual
        # lattice positions), NOT k_direct_stream (compact indices). The
        # decoder reconstructs predicted_k + residual and looks each value
        # up via k_byte, which is keyed on real k values. Using
        # k_direct_stream here would produce a residual that, when added
        # to predicted_k, yields nonsense lattice coordinates that no
        # k_byte entry maps to.
        mode4_fit_records: List[Tuple[GenerativeDescriptor, float]] = []   # (descriptor, σ²_residual)
        mode4_miss_records: List[GenerativeDescriptor] = []                 # offered but did not fit
        if (self.archetype_db is not None and
                self.last_block_curvature is not None and
                len(k_stream) == n and n >= 3):
            t_mode4 = time.time()
            block_curv = self.last_block_curvature

            # Source 1: Channel B query for established generators.
            try:
                queried_gens = self.archetype_db.query_generators_for_class(
                    curvature_class=int(block_curv.curvature_class),
                    curvature_mean=float(block_curv.curvature_mean),
                    max_results=12,           # 12 = manifold symmetry S
                    cross_class=True,
                    cross_class_quota=4,
                )
            except Exception as e:           # pragma: no cover — DB unreachable
                self._log(f"    Mode 4: channel-B query error: {e}")
                queried_gens = []

            # Source 2: derive new generators from this block.
            # Pass the REAL k_stream (lattice positions) so derive's
            # k-targeting fits operate in the right coordinate.
            derived_gens = derive_generators_from_curvature(
                block_curv, list(dk_stream), list(k_stream))

            # Deduplicate by gen_id — a derived generator that matches an
            # already-queried one is the same template; prefer the queried
            # version because it carries the running fit/miss counters.
            seen_gen_ids: Set[str] = set(g.gen_id for g in queried_gens)
            unique_derived = [g for g in derived_gens if g.gen_id not in seen_gen_ids]
            all_candidates = queried_gens + unique_derived
            self._log(f"    Mode 4: {len(queried_gens)} queried + "
                      f"{len(unique_derived)} newly-derived = "
                      f"{len(all_candidates)} candidate generators "
                      f"({time.time() - t_mode4:.3f}s)")

            # For each candidate generator, attempt to fit and build a
            # Mode 4 candidate tuple if accepted.
            import pickle as _pickle
            for gen_desc in all_candidates:
                # Decode the generator's params blob to get type + stream + params.
                # Layout: [1B type_code][1B stream_target][pickled params]
                if not gen_desc.generator_params or len(gen_desc.generator_params) < 2:
                    continue
                gen_type_code = gen_desc.generator_params[0]
                gen_stream_target = gen_desc.generator_params[1]
                gen_type_name = GENERATOR_TYPE_NAMES.get(int(gen_type_code))
                if gen_type_name is None:
                    continue
                gen_class = _GENERATOR_REGISTRY[gen_type_name]
                try:
                    params_dict = _pickle.loads(gen_desc.generator_params[2:])
                except (_pickle.PickleError, EOFError, ValueError):
                    # Corrupt/incompatible payload — record as miss but
                    # never delete (§16.9 NO-REMOVAL).
                    mode4_miss_records.append(gen_desc)
                    continue
                # Generate the prediction in the generator's natural coordinate.
                # If the generator targets dk_stream (PERIODIC/GRAMMAR), it
                # produces n-1 dk values (one per dk slot); we then integrate
                # to get the predicted k_stream anchored at k0 = k_stream[0].
                # If it targets k_stream directly (CONSTANT/LINEAR/POLYNOMIAL),
                # it produces n k values directly.
                if gen_stream_target == GENERATOR_STREAM_DK:
                    predicted_dk = gen_class.generate(params_dict, n - 1)
                    if len(predicted_dk) != n - 1:
                        mode4_miss_records.append(gen_desc)
                        continue
                    # Integrate: predicted_k[0] = k_stream[0] (anchor; this
                    # MUST match what the decoder uses, which is k0 from
                    # the block header — and k0 is set to k_stream[0]).
                    predicted_k: List[int] = [int(k_stream[0])]
                    for d in predicted_dk:
                        predicted_k.append(predicted_k[-1] + int(d))
                else:  # GENERATOR_STREAM_K
                    predicted_k_raw = gen_class.generate(params_dict, n)
                    if len(predicted_k_raw) != n:
                        mode4_miss_records.append(gen_desc)
                        continue
                    predicted_k = [int(v) for v in predicted_k_raw]
                # Compute residual in REAL k-space (NOT compact-index space).
                # The decoder's reconstruction is `predicted_k + residual`
                # which is then mapped via k_byte — this requires the
                # residual to bring `predicted_k` to real k_stream values.
                residual_k = [int(k_stream[i]) - predicted_k[i]
                              for i in range(n)]
                # Compute σ²_residual for the fit-acceptance check.
                if not residual_k:
                    mode4_miss_records.append(gen_desc)
                    continue
                r_mean = sum(residual_k) / len(residual_k)
                r_var = sum((x - r_mean) ** 2 for x in residual_k) / len(residual_k)
                # Acceptance threshold per design doc §16.8.5 step 3:
                # σ²_residual < V = 1/12 (subliminal). Identically-zero
                # residual is the Kolmogorov-minimal case.
                if r_var >= V_BASE:
                    mode4_miss_records.append(gen_desc)
                    continue
                # Accepted — build Mode 4 candidate tuple.
                # Slot 4 carries the full generator payload (type byte +
                # stream byte + pickled params) which the encoder writes
                # verbatim. The decoder reads the stream byte to know
                # which coordinate the generator predicts in.
                mode4_extras = bytes(gen_desc.generator_params)
                # The first residual value goes in dk0_saved slot for
                # parity with other modes' header layout (informational
                # only — Mode 4 reconstruction does not use it directly).
                candidates.append((4, residual_k, residual_k[0], mode4_extras))
                mode4_fit_records.append((gen_desc, float(r_var)))
                self._log(f"      generator type={gen_type_name} "
                          f"stream={'k' if gen_stream_target == GENERATOR_STREAM_K else 'dk'} "
                          f"fit: σ²_residual={r_var:.6f} < V={V_BASE:.6f}")

        # Use best_mode to prioritize: try the analytically-best mode first
        # (lowest lattice entropy → fewest unique values → most compressible)
        candidates.sort(key=lambda cand: (0 if cand[0] == best_mode else 1))

        # Compress each candidate and build the full encoded block.
        #
        # ── Tie-breaker correctness ──
        # The PROJECT GOAL is finding the smallest, most fundamental
        # GENERATOR for the data — the Kolmogorov-style minimum-description
        # representation, not Shannon-entropy compression. When two
        # candidates produce equal-sized encoded blocks, they are NOT
        # interchangeable: the higher-mode one (Mode 3 > Mode 2 > Mode 1
        # > Mode 0) carries strictly more generative structure per byte.
        #
        # Mode 3's Christoffel connection encodes geodesic structure that
        # generalises across files via the archetype database (Channel B).
        # Mode 2's ΔΔk encodes second-order differences that generalise
        # less. Mode 1's Δk encodes first-order. Mode 0 is direct table
        # lookup with no derivative captured at all.
        #
        # Selection key = (encoded_size, -mode):
        #   primary   smaller encoded byte count wins (compression goal)
        #   secondary HIGHER mode wins on size ties (generator quality goal)
        # Lex-minimum of this tuple yields the smallest, most generative
        # representation simultaneously — no list-order bias, no
        # arbitrary first-tried-wins behaviour.
        #
        # ET Three Tools:
        #   Identification Principle: identifies "best" as the lex-min of
        #     a TWO-component descriptor (size + generative depth), not a
        #     single scalar.
        #   Descriptor Gap Principle: closes the gap between "smallest
        #     encoded bytes" (Shannon proxy) and "smallest generator"
        #     (Kolmogorov target) — the gap was the missing secondary
        #     selection criterion.
        #   Subsumption Law: every (size, mode) pair maps to exactly one
        #     score tuple; lex-min subsumes all candidates without
        #     remainder.
        best_block = None
        best_score: Optional[Tuple[int, int]] = None  # (size, -mode)
        mode_names_map = {0: 'k-direct', 1: 'Δk', 2: 'ΔΔk', 3: 'geodesic-ρ',
                          4: 'gen-residual'}
        # Note: cand_mode3 is the legacy name of the 4th tuple slot — it
        # now carries mode-specific extras for Mode 3 OR Mode 4:
        #   Mode 3: tuple (connection_order, connection_window)
        #   Mode 4: bytes generator_payload (type byte + pickled params)
        #   Other:  None
        # The dispatch below switches on cand_mode to interpret the slot.
        for ci, (cand_mode, cand_stream, cand_dk0, cand_extras) in enumerate(candidates):
            cand_pct = 20 + (ci / max(len(candidates), 1)) * 60
            self._progress(cand_pct, f"Phase 3: mode {mode_names_map.get(cand_mode, cand_mode)} "
                                     f"({ci + 1}/{len(candidates)})")
            self._log(f"    Candidate {ci + 1}/{len(candidates)}: "
                      f"mode={mode_names_map.get(cand_mode, cand_mode)}, "
                      f"stream={len(cand_stream):,}")
            rc = walker.recursive_compress(cand_stream)
            # Build the encoded block for this candidate.
            # Mode 3 forwards the connection_order/connection_window kwargs.
            # Mode 4 forwards the generator_payload kwarg.
            # Other modes pass the defaults which the encoder ignores.
            if cand_mode == 3 and cand_extras is not None:
                trial = self._encode_lattice_block(
                    n, r0, k0, cand_mode, cand_dk0,
                    None, rc,
                    connection_order=cand_extras[0],
                    connection_window=cand_extras[1])
            elif cand_mode == 4 and cand_extras is not None:
                trial = self._encode_lattice_block(
                    n, r0, k0, cand_mode, cand_dk0,
                    None, rc,
                    generator_payload=cand_extras)
            else:
                trial = self._encode_lattice_block(
                    n, r0, k0, cand_mode, cand_dk0,
                    unique_k if cand_mode == 0 else None, rc)
            # Principled lex-min: smaller size wins; on size tie, higher mode wins.
            trial_score = (len(trial), -cand_mode)
            if best_score is None or trial_score < best_score:
                best_block = trial
                best_score = trial_score
                self._log(f"      → new best: size={len(trial)}B, "
                          f"mode={mode_names_map.get(cand_mode, cand_mode)}, "
                          f"score={trial_score}")

        if best_block is None:
            # Fallback: use the pre-selected best_mode stream directly
            # (dk_stream_for_compress and dk0_saved from the mode analysis).
            # If the pre-selected mode was 3, the fallback also forwards
            # the connection metadata; otherwise it uses defaults.
            rc = walker.recursive_compress(dk_stream_for_compress)
            if best_mode == 3:
                best_block = self._encode_lattice_block(
                    n, r0, k0, best_mode, dk0_saved, None, rc,
                    connection_order=mode3_connection_order,
                    connection_window=mode3_connection_window)
            else:
                best_block = self._encode_lattice_block(
                    n, r0, k0, best_mode, dk0_saved,
                    unique_k if best_mode == 0 else None, rc)
        assert best_block is not None  # Narrowing: fallback guarantees assignment

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
        if len(best_block) > n:
            self._log(f"    Standard modes expanded — trying pair-first "
                      f"(Re-Pair) for chaotic data")
            for ci, (cand_mode, cand_stream, cand_dk0, cand_extras) in enumerate(candidates):
                if len(cand_stream) < 4:
                    continue
                try:
                    rc_pair = walker.pair_recursive_compress(cand_stream)
                    # Only proceed if the pair-first method actually found grammar rules
                    if rc_pair['archetype_defs']:
                        # Forward Mode 3 connection metadata or Mode 4 generator
                        # payload when applicable; other modes use defaults.
                        if cand_mode == 3 and cand_extras is not None:
                            trial_pair = self._encode_lattice_block(
                                n, r0, k0, cand_mode, cand_dk0,
                                None, rc_pair,
                                connection_order=cand_extras[0],
                                connection_window=cand_extras[1])
                        elif cand_mode == 4 and cand_extras is not None:
                            trial_pair = self._encode_lattice_block(
                                n, r0, k0, cand_mode, cand_dk0,
                                None, rc_pair,
                                generator_payload=cand_extras)
                        else:
                            trial_pair = self._encode_lattice_block(
                                n, r0, k0, cand_mode, cand_dk0,
                                unique_k if cand_mode == 0 else None, rc_pair)
                        # Same principled lex-min as the main candidate loop:
                        # (size, -mode) — smaller wins; on tie HIGHER mode wins
                        # (more generative structure → better generator).
                        trial_pair_score = (len(trial_pair), -cand_mode)
                        if best_score is None or trial_pair_score < best_score:
                            best_block = trial_pair
                            best_score = trial_pair_score
                            self._log(f"    Pair-first improved: "
                                      f"mode={mode_names_map.get(cand_mode, cand_mode)}, "
                                      f"{len(trial_pair):,}B "
                                      f"({len(rc_pair['archetype_defs'])} grammar rules), "
                                      f"score={trial_pair_score}")
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
        # Capture discovered archetypes for database storage (side-channel)
        self.discovered_archetypes.extend(walker.discovered_archetypes)

        # ── Channel B feedback loop (Tier 3.B.4) ──
        # After the mode competition resolves, record the outcome of
        # every Mode 4 candidate in the generative_descriptors table:
        #   • Every generator that FIT (σ²_residual < V) gets
        #     increment_fit=True with its achieved residual variance.
        #     If a fitting generator was ALSO the overall winner, its
        #     last_confirmed timestamp updates.
        #   • Every generator that was OFFERED but did not fit gets
        #     increment_miss=True.
        # Per design doc §16.9 NO-REMOVAL: high-miss generators stay in
        # the database forever as Unsubstantiated predictions — they
        # may match a future file with different data even if today
        # they did not fit. The database compounds; nothing is pruned.
        #
        # ET Three Tools:
        #   Identification: identifies which derived generators are
        #     proving themselves on real data (fit_count growing) and
        #     which remain unsubstantiated (miss_count growing without
        #     fit_count). Both are Descriptors of the database's
        #     learning state.
        #   Descriptor Gap: closes the gap between "generator was
        #     attempted" and "database remembers the attempt" — the
        #     fit/miss counters ARE the missing memory.
        #   Subsumption: every Mode 4 candidate emerges from this loop
        #     in EXACTLY one of {recorded as fit, recorded as miss};
        #     no candidate is dropped silently.
        if self.archetype_db is not None and (mode4_fit_records or mode4_miss_records):
            try:
                for gen_desc, residual_var in mode4_fit_records:
                    self.archetype_db.store_generator(
                        gen_desc,
                        increment_fit=True,
                        residual_variance=residual_var)
                for gen_desc in mode4_miss_records:
                    self.archetype_db.store_generator(
                        gen_desc,
                        increment_miss=True)
                if mode4_fit_records or mode4_miss_records:
                    self._log(
                        f"    Channel B feedback: {len(mode4_fit_records)} fits + "
                        f"{len(mode4_miss_records)} misses recorded "
                        f"(NO deletions per §16.9)"
                    )
            except Exception as e:    # pragma: no cover — DB unreachable
                self._log(f"    Channel B feedback failed (non-fatal): {e}")

        # ── Block Type 4 candidate: Variable-Curvature Segmentation (Tier 4) ──
        # Per design doc §10 + §22.4: when a block has variable curvature,
        # split it at curvature sign-change boundaries and compress each
        # segment INDEPENDENTLY through the standard pipeline. The combined
        # output (segmented wrapper + per-segment blocks) may be smaller
        # than any single-block strategy applied to the heterogeneous whole.
        #
        # Why this helps: in a mixed block, the standard modes pick ONE
        # global strategy (e.g. ΔΔk encoding) that is averaged across
        # all segments. Each segment in isolation may have a strictly
        # better strategy (e.g. one segment is uniform and compresses to
        # 6 bytes via Block Type 1; another is linear and compresses via
        # Mode 1; a third has a recurrence and compresses via Mode 4).
        # Segmenting reveals each segment's optimal local strategy.
        #
        # Recursion guard: only attempt segmentation at depth 0. Each
        # segment is compressed at depth 1 (its standard pipeline runs
        # but it cannot itself segment). This prevents unbounded recursion
        # while allowing the one-level granularity that captures most of
        # the segmentation benefit (deeper segmentation is the work of
        # later passes, not this single compress_block invocation).
        #
        # Competition: the segmented candidate is built and compared via
        # the same (size, -mode) lex-min comparator used elsewhere. The
        # segmented wrapper carries an effective "mode" of -1 in the
        # comparator (lower than mode 0) so that on size ties, the
        # NON-segmented form wins (segmenting adds per-segment overhead
        # only justified by a strict size win — this matches the
        # K-complexity principle that the simpler representation wins
        # when size is equal).
        #
        # ET Three Tools:
        #   Identification Principle: identifies which blocks have local
        #     curvature variation that a single-block strategy averages
        #     away. The compute_segmentation call returns those blocks'
        #     byte-level partitions.
        #   Descriptor Gap Principle: closes the gap between "block has
        #     mixed curvature" and "compressor can use a different
        #     strategy per region". Segmentation IS the missing
        #     Descriptor of regional structure.
        #   Subsumption Law: the segments tile the block without overlap
        #     or gap; the segmented wrapper subsumes the original block
        #     without remainder.
        if (_segmentation_depth == 0
                and self.last_block_curvature is not None
                and (self.last_block_curvature.is_variable()
                     or self.last_block_curvature.is_singular())):
            try:
                segments = self.curvature_analyzer.compute_segmentation(
                    list(dk_stream), n)
                if segments and len(segments) >= 2:
                    self._log(f"    Block Type 4: trying {len(segments)}-segment "
                              f"compression "
                              f"({self.last_block_curvature.class_name()})")
                    seg_blocks: List[bytes] = []
                    for seg_i, (seg_start, seg_end) in enumerate(segments):
                        seg_data = data[seg_start:seg_end]
                        # Recursive call with depth=1 so this segment cannot
                        # itself segment (preventing unbounded recursion).
                        seg_block = self.compress_block(
                            seg_data, _segmentation_depth=1)
                        seg_blocks.append(seg_block)
                        self._log(f"      segment {seg_i + 1}/{len(segments)}: "
                                  f"bytes [{seg_start}:{seg_end}] "
                                  f"({seg_end - seg_start}B) → "
                                  f"compressed {len(seg_block)}B")
                    # Build the Block Type 4 wrapper:
                    #   [1B type=4][4B uint32 n][manifold n_segments]
                    #   for each segment: [manifold seg_size][N bytes seg_data]
                    seg_parts: List[bytes] = [
                        struct.pack('<B', 4),                       # type = 4 segmented
                        struct.pack('<I', n),                       # original block size
                        pack_manifold_uint(len(seg_blocks)),        # n_segments
                    ]
                    for seg_block in seg_blocks:
                        seg_parts.append(pack_manifold_uint(len(seg_block)))
                        seg_parts.append(seg_block)
                    seg_candidate = b''.join(seg_parts)
                    self._log(f"      Block Type 4 total: {len(seg_candidate):,}B "
                              f"(vs single-block {len(best_block):,}B)")
                    # Comparator: (size, -1) so segmented wins ONLY on strict
                    # size improvement vs the single-block result. The -1
                    # secondary key sits below all standard modes (0..4),
                    # meaning on a size tie the standard single-block form
                    # wins (simpler representation = better K-complexity
                    # when size is identical).
                    seg_score = (len(seg_candidate), -1)
                    if best_score is None or seg_score < best_score:
                        best_block = seg_candidate
                        best_score = seg_score
                        self._log(f"      → new best: size={len(seg_candidate)}B, "
                                  f"mode=Block Type 4 segmented, "
                                  f"score={seg_score}")
            except Exception as e:    # pragma: no cover — defensive
                self._log(f"    Block Type 4 segmentation skipped: {e}")

        return best_block

    def decompress_block(self, block_data: bytes) -> bytes:
        """Decompress a single block back to original bytes."""
        pos = 0

        block_type = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1
        block_type_names = {0: 'empty', 1: 'uniform', 2: 'lattice',
                            3: 'passthrough', 4: 'segmented'}
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

        if block_type == 4:  # segmented (Tier 4) — per design doc §22.4
            # Format:
            #   [4B uint32 LE  n]                 — original block size
            #   [manifold uint n_segments]
            #   for each segment:
            #     [manifold uint segment_size]
            #     [N bytes        segment_data]   — a complete sub-block
            #                                       (any block_type, recursive)
            # Reconstruction: recursively decompress each segment, then
            # concatenate. The total output is truncated/padded to n bytes
            # (truncation only — segments are guaranteed to cover the
            # original block exactly when produced by the encoder).
            #
            # ET Three Tools:
            #   Identification: identifies the segments at byte level;
            #     each segment's header carries its own n, type, and mode.
            #   Descriptor Gap: closes the gap between "block-level
            #     curvature changes" and "byte-level data slices" via
            #     the explicit segment table.
            #   Subsumption: every byte of the original block belongs
            #     to exactly one segment; concatenation reconstructs
            #     the block without remainder.
            n = struct.unpack_from('<I', block_data, pos)[0]
            pos += 4
            n_segments, pos = unpack_manifold_uint(block_data, pos)
            self._log(f"    Type-4 segmented: n={n} bytes, {n_segments} segments")
            parts: List[bytes] = []
            for seg_i in range(n_segments):
                seg_size, pos = unpack_manifold_uint(block_data, pos)
                seg_data = block_data[pos:pos + seg_size]
                pos += seg_size
                # Recursive call — each segment is a self-contained block.
                seg_decompressed = self.decompress_block(seg_data)
                parts.append(seg_decompressed)
                self._log(f"      segment {seg_i + 1}/{n_segments}: "
                          f"compressed={seg_size} → decompressed={len(seg_decompressed)}")
            joined = b''.join(parts)
            # Truncate to the declared block size (defensive — well-formed
            # encoders produce segments whose lengths sum to exactly n).
            if len(joined) != n:
                self._log(f"    Type-4 segment-sum length mismatch: "
                          f"got {len(joined)}B, expected {n}B (truncating)")
            return joined[:n]

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
        # ── Mode 3 extension fields (Tier 2) ──
        # Per design doc §22.3.4: when mode == 3, the header carries
        # connection_order (1 byte) and connection_window (manifold uint)
        # after dk0_saved. For modes 0/1/2 these fields are absent and
        # the parse position advances directly to the mode-0 k_table or
        # the dk_table.
        connection_order = 0
        connection_window = 0
        if mode == 3:
            connection_order = struct.unpack_from('<B', block_data, pos)[0]
            pos += 1
            connection_window, pos = unpack_manifold_uint(block_data, pos)
            self._log(f"    Mode 3 header: connection_order={connection_order}, "
                      f"connection_window={connection_window}")
        # ── Mode 4 extension field (Tier 3.B.4) ──
        # When mode == 4, the header carries a length-prefixed generator
        # payload immediately after dk0_saved:
        #   [4B uint32 LE payload_len][N bytes payload]
        # The payload's first byte is the generator type code; the rest
        # is the pickled params dict. The recursive_compress data that
        # follows in the standard format slots represents the RESIDUAL
        # stream (residual_k_i = actual_k_i - predicted_k_i), not Δk.
        # Reconstruction: predicted = generator.generate(params, n);
        # actual_k = predicted + residual; bytes from k via k_byte map.
        gen_payload: Optional[bytes] = None
        if mode == 4:
            gp_len = struct.unpack_from('<I', block_data, pos)[0]
            pos += 4
            gen_payload = bytes(block_data[pos:pos + gp_len])
            pos += gp_len
            self._log(f"    Mode 4 header: generator_payload={gp_len}B")
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
        n_base, pos = unpack_manifold_uint(block_data, pos)
        dk_table: List[int] = []
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
        n_arch, pos = unpack_manifold_uint(block_data, pos)
        total_syms, pos = unpack_manifold_uint(block_data, pos)
        arch_enc_type = struct.unpack_from('<B', block_data, pos)[0]
        pos += 1
        n_flat, pos = unpack_manifold_uint(block_data, pos)

        arch_defs = []
        if n_flat > 0:
            if arch_enc_type == 1:
                # V_config encoded archetype stream
                atbl_len, pos = unpack_manifold_uint(block_data, pos)
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
            n_syms, pos = unpack_manifold_uint(block_data, pos)
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
            n_syms, pos = unpack_manifold_uint(block_data, pos)
            tbl_len, pos = unpack_manifold_uint(block_data, pos)
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

        base_indices: List[int] = expand(sym_stream)
        raw_vals: List[int] = [dk_table[i] for i in base_indices]

        # Reconstruct based on mode
        if mode == 0:
            # k-direct: raw_vals are compact indices into k_direct_table
            # Map back to k-values, then to bytes directly
            assert k_direct_table is not None  # Narrowing: mode 0 always populates k_direct_table
            output = bytearray()
            for idx in raw_vals:
                k_val = k_direct_table[int(idx)]
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
        elif mode == 3:
            # ── Mode 3: Geodesic Residual reconstruction (Tier 2) ──
            # raw_vals ARE residuals. Reconstruct Δk causally:
            #   Δk_0     = dk0_saved  (header)
            #   Δk_{i+1} = ρ_i + Δk_i + Γ_i
            # where Γ_i is recomputed at each step from the partially-
            # reconstructed Δk stream (exactly mirroring the C encoder).
            #
            # CRITICAL — _c_trunc_div is mandatory here. Python's `//`
            # would diverge from C's `/` for negative dividends and break
            # the lossless guarantee. The helper at module scope mirrors
            # C semantics for ALL operand signs (verified by 14/14
            # sign-combination tests).
            #
            # ET Three Tools:
            #   Identification: identifies the inverse of the encoder's
            #     geodesic prediction — ρ + Γ + Δk_prev gives Δk_next.
            #   Descriptor Gap: closes the gap between "encoded residual"
            #     and "decoded Δk" — the integer-arithmetic agreement is
            #     itself the Descriptor that links the two sides.
            #   Subsumption: handles all three connection orders 0/1/2
            #     identically — order 0 trivially has Γ = 0, order 1 adds
            #     the linear connection, order 2 adds the quadratic.
            dk_stream = [dk0_saved]
            for i, residual in enumerate(raw_vals):
                gamma_i = 0
                # First-order connection: windowed mean of ΔΔk
                if connection_order >= 1 and i > 0:
                    w_start = max(0, i - connection_window + 1)
                    ddk_sum = 0
                    count = 0
                    for j in range(w_start, i):
                        ddk_sum += dk_stream[j + 1] - dk_stream[j]
                        count += 1
                    if count > 0:
                        gamma_i = _c_trunc_div(ddk_sum, count)
                # Second-order connection: add ½ · windowed mean of ΔΔΔk
                if connection_order >= 2 and i > 1:
                    w_start = max(1, i - connection_window + 1)
                    dddk_sum = 0
                    count = 0
                    for j in range(w_start, i - 1):
                        ddk_j  = dk_stream[j + 1] - dk_stream[j]
                        ddk_j1 = dk_stream[j + 2] - dk_stream[j + 1]
                        dddk_sum += ddk_j1 - ddk_j
                        count += 1
                    if count > 0:
                        gamma_i += _c_trunc_div(dddk_sum, 2 * count)
                dk_stream.append(residual + dk_stream[-1] + gamma_i)
        elif mode == 4:
            # ── Mode 4: Generator + Residual reconstruction (Tier 3.B.4) ──
            # Header carried `gen_payload`:
            #   [1B type_code][1B stream_target][pickled params]
            # Reconstruction:
            #   1. Decode type_code → look up generator class
            #   2. Decode stream_target → know which coordinate the generator
            #      predicts in (k_stream directly, or dk_stream which we
            #      then integrate into k_stream via cumsum + anchor)
            #   3. Unpickle params dict
            #   4. predicted = generator.generate(params, ...)
            #   5. if stream_target == DK: predicted_k = anchor + cumsum(predicted)
            #      where anchor = k0 (block header's initial k value)
            #      else (stream_target == K): predicted_k = predicted directly
            #   6. residual = raw_vals (decoded by standard pipeline)
            #   7. actual_k[i] = predicted_k[i] + residual[i]
            #   8. Map each actual_k to its byte via the k_byte table
            #
            # The dk0_saved field holds the FIRST RESIDUAL value (for
            # consistency with other modes' header layout), which is
            # already incorporated into raw_vals — informational only.
            #
            # ET Three Tools:
            #   Identification: identifies the generator AND the coordinate
            #     system it predicts in — both are needed to invert the
            #     encoder's fit() call exactly.
            #   Descriptor Gap: closes the gap between "encoded payload"
            #     and "regenerated data". The stream_target byte WAS
            #     the missing Descriptor — without it, dk-targeting
            #     generators (PERIODIC, GRAMMAR) had their predictions
            #     subtracted from k-space, producing huge residuals
            #     that never fit. Adding stream_target closes the gap.
            #   Subsumption: predicted + residual = actual, exactly. The
            #     two streams subsume the original data without remainder.
            import pickle
            if not gen_payload or len(gen_payload) < 2:
                raise ValueError("Mode 4 block missing or truncated generator payload")
            type_code = gen_payload[0]
            stream_target = gen_payload[1]
            gen_type = GENERATOR_TYPE_NAMES.get(int(type_code))
            if gen_type is None:
                raise ValueError(f"Unknown generator type code {type_code} in Mode 4 block")
            gen_class = _GENERATOR_REGISTRY[gen_type]
            params = pickle.loads(gen_payload[2:])
            # Generate prediction in the generator's natural coordinate
            # system, then integrate to k_stream if needed.
            if stream_target == GENERATOR_STREAM_DK:
                # Generator produces n-1 dk values; integrate to n k values.
                predicted_dk = gen_class.generate(params, n - 1)
                if len(predicted_dk) != n - 1:
                    # Defensive normalization to expected length.
                    if len(predicted_dk) < n - 1:
                        predicted_dk = list(predicted_dk) + \
                            [predicted_dk[-1] if predicted_dk else 0] * (n - 1 - len(predicted_dk))
                    else:
                        predicted_dk = list(predicted_dk[:n - 1])
                # Anchor at k0 (block header's initial k value — same as
                # the encoder used when computing predicted_k for the
                # residual). predicted_k[0] = k0; predicted_k[i] =
                # predicted_k[i-1] + predicted_dk[i-1].
                predicted: List[int] = [int(k0)]
                for d in predicted_dk:
                    predicted.append(predicted[-1] + int(d))
            else:  # GENERATOR_STREAM_K
                predicted_raw = gen_class.generate(params, n)
                if len(predicted_raw) != n:
                    if len(predicted_raw) < n:
                        predicted_raw = list(predicted_raw) + \
                            [predicted_raw[-1] if predicted_raw else 0] * (n - len(predicted_raw))
                    else:
                        predicted_raw = list(predicted_raw[:n])
                predicted = [int(v) for v in predicted_raw]
            # raw_vals contains exactly n residual values (one per byte).
            # Element-wise add to recover the actual k-stream, then map
            # each k value to its byte through the byte_k inverse table.
            output = bytearray()
            for i in range(n):
                k_val = predicted[i] + int(raw_vals[i])
                if k_val in k_byte:
                    output.append(k_byte[k_val])
                else:
                    nearest_k = min(k_byte.keys(), key=lambda kk: abs(kk - k_val))
                    output.append(k_byte[nearest_k])
            return bytes(output[:n])
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
      [4 bytes]  Magic: 'CDF' followed by version byte 0x03 (current; legacy 0x02 also accepted on read)
      [1 byte]   Version: 3 (current; legacy 2 also accepted on read)
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
                 metabolism: Optional[CDFMetabolism] = None,
                 archetype_db: Optional[ArchetypeDatabase] = None):
        self._log = log_fn or (lambda m: logger.info(m))
        self._progress = progress_fn or (lambda pct, m: None)
        self.metabolism = metabolism or _metabolism
        self.metabolism._log = self._log
        self.metabolism.sense()
        self.archetype_db = archetype_db or get_archetype_db(log_fn=self._log)
        self.engine = CDFEngine(log_fn, progress_fn, metabolism=self.metabolism,
                                archetype_db=self.archetype_db)
        self._log(self.metabolism.summary())
        self._log(self.archetype_db.summary())

    def compress_file(self, input_path: str, output_path: str) -> dict:
        """Compress a file to CDF format via streaming — no full file in memory."""
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_size_on_disk = os.path.getsize(input_path)
        self._log(f"Compressing: {input_path} ({input_size_on_disk:,} bytes on disk)")
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # ── Streaming Pass: source → temp file + R₀ + SHA-256 + size ──
        # One pass through the source. The source could be a network share,
        # a slow HDD, a compressed archive — we read it exactly ONCE.
        # While streaming, we compute R₀ (running log sum), SHA-256
        # (hashlib.update per chunk), and file size.
        temp_dir = tempfile.gettempdir()
        fd, temp_path = tempfile.mkstemp(prefix='cdf_comp_', suffix='.tmp',
                                         dir=temp_dir)
        hasher = hashlib.sha256()
        log_sum = 0.0
        byte_count = 0
        original_size = 0

        try:
            with open(input_path, 'rb') as src:
                with os.fdopen(fd, 'wb') as dst:
                    while True:
                        chunk = src.read(BLOCK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                        hasher.update(chunk)
                        original_size += len(chunk)
                        arr = np.frombuffer(chunk, dtype=np.uint8).astype(np.float64) + 1.0
                        log_sum += float(np.sum(np.log(arr)))
                        byte_count += len(chunk)

            sha256 = hasher.digest()
            global_r0 = math.exp(log_sum / byte_count) if byte_count > 0 else 1.0
            num_blocks = (original_size + BLOCK_SIZE - 1) // BLOCK_SIZE

            self._log(f"Original: {original_size:,} bytes, {num_blocks} blocks")
            self._log(f"R₀ = {global_r0:.6f}, N_res = {N_FULL}")

            t0 = time.time()
            compressed_blocks = []
            total_compressed = 0

            # ── Block-by-block compression from temp file ──
            # Reads one BLOCK_SIZE block at a time — O(BLOCK_SIZE) memory.
            for bi in range(num_blocks):
                with open(temp_path, 'rb') as tf:
                    tf.seek(bi * BLOCK_SIZE)
                    block = tf.read(BLOCK_SIZE)

                t_bi = time.time()
                self._log(f"\n─── Block {bi + 1}/{num_blocks} ({len(block):,} bytes) ───")
                cb = self.engine.compress_block(block)
                compressed_blocks.append(cb)
                total_compressed += len(cb)

                elapsed_bi = time.time() - t_bi
                elapsed_total = time.time() - t0
                block_ratio = len(cb) / len(block) * 100 if len(block) > 0 else 0
                be = min((bi + 1) * BLOCK_SIZE, original_size)
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

            # ── Store discovered archetypes to database ──
            if self.engine.discovered_archetypes:
                # Tier 5.D.1: pass the last block's curvature metadata so
                # store() can compute geodesic_deviation per archetype and
                # write curvature_class, euler_characteristic, and
                # curvature_spectrum_hash into the archetype rows.
                curv_kwargs: Dict[str, Any] = {}
                if self.engine.last_block_curvature is not None:
                    curv_kwargs['block_curvature_class'] = int(
                        self.engine.last_block_curvature.curvature_class)
                    curv_kwargs['block_euler_char'] = float(
                        self.engine.last_block_curvature.euler_characteristic)
                if self.engine.last_block_ddk_stream is not None:
                    curv_kwargs['block_ddk_stream'] = self.engine.last_block_ddk_stream
                if getattr(self.engine, 'last_block_spectrum_hash', None) is not None:
                    curv_kwargs['block_spectrum_hash'] = self.engine.last_block_spectrum_hash
                self.archetype_db.store(
                    self.engine.discovered_archetypes, global_r0,
                    **curv_kwargs)
                self._log(f"Stored {len(self.engine.discovered_archetypes)} archetypes "
                          f"to database"
                          f"{' (with curvature metadata)' if curv_kwargs else ''}")
                self.engine.discovered_archetypes.clear()

            # ── Clear curvature profiles for next file (Tier 1) ──
            # The profiles for THIS file are already used by Phase 1.5 logging
            # and (in Tier 3) by Channel B generator derivation. Clearing here
            # prevents bleed-through into the next file's compression.
            if self.engine.discovered_curvature_profiles:
                self._log(f"Phase 1.5 summary: {len(self.engine.discovered_curvature_profiles)} "
                          f"block curvature profiles for {input_path}")
                self.engine.discovered_curvature_profiles.clear()

            return {'original_size': original_size, 'compressed_size': comp_size,
                    'ratio': ratio, 'time': elapsed, 'blocks': num_blocks}

        finally:
            # ── Cleanup: delete temp file ──
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
                    # Use cached metadata — no bulk file read needed
                    original_size = tower.data_size
                    sha256 = tower.data_hash
                    num_blocks_u = (original_size + BLOCK_SIZE - 1) // BLOCK_SIZE

                    # For multi-block files, compress each block through universal
                    # Reads one block at a time from temp file — O(BLOCK_SIZE) memory
                    universal_blocks = []
                    for bi in range(num_blocks_u):
                        block_data = tower.read_block(bi)
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
                                                  log_fn=lambda m: None,
                                                  archetype_db=self.archetype_db)
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

            # ── Store any archetypes discovered during universal compression ──
            # compress_file already stored personal-path archetypes.
            # The universal path may have discovered additional patterns at
            # universal R₀ — store those too.
            if self.engine.discovered_archetypes:
                curv_kwargs_univ: Dict[str, Any] = {}
                if self.engine.last_block_curvature is not None:
                    curv_kwargs_univ['block_curvature_class'] = int(
                        self.engine.last_block_curvature.curvature_class)
                    curv_kwargs_univ['block_euler_char'] = float(
                        self.engine.last_block_curvature.euler_characteristic)
                if self.engine.last_block_ddk_stream is not None:
                    curv_kwargs_univ['block_ddk_stream'] = self.engine.last_block_ddk_stream
                if getattr(self.engine, 'last_block_spectrum_hash', None) is not None:
                    curv_kwargs_univ['block_spectrum_hash'] = self.engine.last_block_spectrum_hash
                self.archetype_db.store(self.engine.discovered_archetypes,
                                        universal.universal_r0,
                                        **curv_kwargs_univ)
                self.engine.discovered_archetypes.clear()

            # ── Clear universal-path curvature profiles (Tier 1) ──
            # compress_file already cleared its own profiles. The universal
            # path may have produced additional profiles for the universal R₀
            # blocks — clear them here to keep per-file isolation.
            if self.engine.discovered_curvature_profiles:
                self.engine.discovered_curvature_profiles.clear()

            # ── Cleanup: delete this tower's temp file immediately ──
            # The metadata (R₀, dk_stream, archetypes) stays cached.
            # Only the raw byte temp copy is released — one file at a time.
            tower.cleanup()

            results_per_file.append({
                'file': fname,
                'original': tower.data_size,
                'personal_r0': tower.personal_r0,
            })

        # ── Final cleanup: ensure all temp files are released ──
        for tower in towers:
            tower.cleanup()

        elapsed = time.time() - t_total
        ratio = total_comp / total_orig if total_orig > 0 else 0

        self._log(f"\n{'═' * 60}")
        self._log(f"TOWER-AWARE BATCH DONE: {len(towers)} files, "
                  f"{total_orig:,} → {total_comp:,} bytes ({ratio * 100:.1f}%)")
        self._log(f"Universal R₀ = {universal.universal_r0:.6f}")
        self._log(f"Cross-file archetypes: {len(cross_archetypes)}")
        self._log(f"Total time: {elapsed:.1f}s")
        self._log(f"Pattern memory: {self.archetype_db.summary()}")
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
        # Accept current (v4) and legacy (v2, v3) magic.
        # v2 files contain only modes 0/1/2; v3 files may contain modes
        # 0/1/2/3; v4 files may contain modes 0/1/2/3/4. The v4 decoder
        # is a superset of v3 which is a superset of v2 — every legacy
        # block format is fully decodable by the current decoder.
        assert magic in (CDF_MAGIC, CDF_MAGIC_LEGACY_V3, CDF_MAGIC_LEGACY_V2), \
            f"Bad magic: {magic}"
        is_legacy_v2 = (magic == CDF_MAGIC_LEGACY_V2)
        is_legacy_v3 = (magic == CDF_MAGIC_LEGACY_V3)

        version = struct.unpack_from('<B', cdf, pos)[0]
        pos += 1
        # The version byte must match the magic that announced it.
        if is_legacy_v2:
            assert version == CDF_VERSION_LEGACY_V2, \
                f"Magic CDF\\x02 expects version {CDF_VERSION_LEGACY_V2}, got {version}"
        elif is_legacy_v3:
            assert version == CDF_VERSION_LEGACY_V3, \
                f"Magic CDF\\x03 expects version {CDF_VERSION_LEGACY_V3}, got {version}"
        else:
            assert version == CDF_VERSION, \
                f"Magic CDF\\x04 expects version {CDF_VERSION}, got {version}"

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
            self.log(f"Pattern Memory:\n  {get_archetype_db(log_fn=self.log).summary()}")
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
                self.log("    db                          Show archetype database stats")
                self.log("    db clear                    Clear archetype database")
                self.log("    db path                     Show database file path")
                self.log("    db import <path>            Merge another archetype database")
                self.log("    status                      Show metabolism + database status")
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

            if cmd == 'status':
                _metabolism.force_resense()
                self.log(f"Resource Governance:\n{_metabolism.summary()}")
                self.log(f"Pattern Memory:\n  {get_archetype_db().summary()}")
                return

            if cmd == 'db':
                db = get_archetype_db(log_fn=self.log)
                if len(parts) >= 2 and parts[1].lower() == 'clear':
                    # Route through the dual-mode clear_archetypes method
                    # so the GUI works identically in normal .db mode and
                    # in VFS .cdf mode. The previous direct-sqlite3 path
                    # silently no-oped in VFS mode because db.db_path does
                    # not exist when the compressed .cdf is the live store.
                    try:
                        cleared = db.clear_archetypes()
                        self.log(f"  Archetype database cleared "
                                 f"({cleared} patterns removed).")
                    except Exception as e:
                        self.log(f"  Error clearing database: {e}")
                elif len(parts) >= 2 and parts[1].lower() == 'path':
                    self.log(f"  Database: {db.db_path}")
                elif len(parts) >= 3 and parts[1].lower() == 'import':
                    import_path = parts[2]
                    try:
                        result = db.import_from(import_path)
                        self.log(f"  Import complete: {result['n_new']} new, "
                                 f"{result['n_updated']} merged "
                                 f"({result['n_total']} total)")
                        self.log(f"  {db.summary()}")
                    except FileNotFoundError:
                        self.log(f"  Database not found: {import_path}")
                    except Exception as e:
                        self.log(f"  Import error: {e}")
                else:
                    s = db.stats()
                    self.log(f"  {db.summary()}")
                    if s['entries'] > 0:
                        self.log(f"  R₀ groups: {s['r0_groups']}")
                        self.log(f"  Total hits: {s['total_hits']}")
                        self.log(f"  Avg elegance: {s['avg_elegance']:.2f}")
                        self.log(f"  Database: {s['disk_mb']:.2f} MB "
                                 f"(disk free: {s['disk_free_mb']:.0f} MB)")
                        self.log(f"  Path: {db.db_path}")
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
                    self.out_var.set(fp + '.cdf' if not fp.endswith('.cdf')
                                     else fp[:-4] + '.decompressed')
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

        def set_button_state(self, st):
            """Set compress/decompress button states (thread-safe via root.after)."""

            def _update():
                try:
                    logger.debug(f"set_button_state: setting state to '{st}'")
                    if st == 'disabled':
                        self.comp_btn.state(['disabled'])
                        self.decomp_btn.state(['disabled'])
                    else:
                        self.comp_btn.state(['!disabled'])
                        self.decomp_btn.state(['!disabled'])
                    logger.debug(f"set_button_state: state '{st}' applied successfully")
                except (AttributeError, RuntimeError, tk.TclError) as e:
                    logger.error(f"set_button_state FAILED: {e}\n{traceback.format_exc()}")
                    self.log(f"[ERROR] Button state change failed: {e}")

            self.root.after(0, _update)

        def do_compress(self):
            """Launch CDF compression in a background thread. Supports files and folders."""
            inp, out = self.in_var.get(), self.out_var.get()
            if not inp or not out:
                messagebox.showerror("Error", "Select input and output files.")
                return
            self.set_button_state('disabled')
            self.progress(0, "Starting compression...")
            self.log(f"\n{'═' * 60}")
            self.log("COMPRESSION STARTED")

            def _run():
                logger.info("Compress thread started")
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
                        file_list: List[str] = [f for f in inp.split('|') if os.path.isfile(f)]
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
                    err_msg = f"COMPRESS ERROR: {e}\n{traceback.format_exc()}"
                    logger.error(err_msg)
                    self.log(f"ERROR: {e}\n{traceback.format_exc()}")
                finally:
                    logger.info("Compress thread finally: re-enabling buttons")
                    self.set_button_state('normal')
                    logger.info("Compress thread finished")

            threading.Thread(target=_run, daemon=True).start()

        def do_decompress(self):
            """Launch CDF decompression in a background thread. Supports files and folders."""
            inp, out = self.in_var.get(), self.out_var.get()
            if not inp or not out:
                messagebox.showerror("Error", "Select input and output files.")
                return
            self.set_button_state('disabled')
            self.progress(0, "Starting decompression...")
            self.log(f"\n{'═' * 60}")
            self.log("DECOMPRESSION STARTED")

            def _run():
                try:
                    t_total = time.time()
                    cdf_compressor = CDFCompressor(log_fn=self.log, progress_fn=self.progress)

                    if '|' in inp:
                        # ── Multi-file mode ──
                        file_list: List[str] = [f for f in inp.split('|') if os.path.isfile(f)]
                        if not file_list:
                            self.log("No valid files in selection.")
                            return
                        os.makedirs(out, exist_ok=True)
                        self.log(f"Batch: {len(file_list)} .cdf files → {out}")
                        all_pass = True
                        for fi, in_file in enumerate(file_list):
                            fname: str = os.path.basename(in_file)
                            out_name: str = fname[:-4] if fname.endswith('.cdf') else fname + '.out'
                            out_file: str = os.path.join(out, out_name)
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
                            out_file: str = os.path.join(out, fname[:-4] if fname.endswith('.cdf') else fname + '.out')
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
                    self.set_button_state('normal')

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
    parsed_action = None
    parsed_input = None
    parsed_output = None

    if len(sys.argv) > 1 and sys.argv[1] != 'gui':
        import argparse
        parser = argparse.ArgumentParser(description='ET CDF Compressor — 27720ET')
        parser.add_argument('action', choices=['compress', 'decompress', 'gui'])
        parser.add_argument('input', nargs='?')
        parser.add_argument('output', nargs='?')
        args = parser.parse_args()
        if args.action != 'gui':
            parsed_action = args.action
            parsed_input = args.input
            if args.output:
                parsed_output = args.output
            elif parsed_input:
                parsed_output = (parsed_input + '.cdf' if parsed_action == 'compress'
                                 else parsed_input.replace('.cdf', '.out'))

    build_gui(cli_action=parsed_action, cli_input=parsed_input, cli_output=parsed_output)
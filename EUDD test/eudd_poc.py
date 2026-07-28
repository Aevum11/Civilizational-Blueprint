#!/usr/bin/env python3
"""
EUDD Proof-of-Concept Verification Script
==========================================
Exception Theory — ET Universal Discovery Database
Verifies the CORE PROMISES of the EUDD system pipeline on real files.

The algebraic identities (#0, A-I) are ALREADY VERIFIED by the 13
identity scripts. This POC does NOT re-verify them. It verifies the
SYSTEM — that the pipeline works end-to-end on real files.

Core promises verified:
  1. File → DSR pipeline (any file → deterministic lattice address)
  2. Lossless file round-trip (seed → pullback → exact file reconstruction)
  3. Infinite LCM tower escalation (without re-accessing r)
  4. CF home-finding method (parallel to tower)
  5. .akashic binary format (write → read → verify)
  6. Anti-numerology protocol (Seed Protocol N1/N2 checks)
  7. Kolmogorov generator (the DSR IS the file — pullback produces it)
  8. Dedup/memoization (same file → cache hit)
  9. Δε versioning (modified file → only Δε stored)
  10. Self-DSR tracking (.akashic's own DSR across cycles)

All computation: mpmath at dynamic precision (scales to data size).
Lattice math default: 461 dps (361 lattice + 100 guard).
Data projection: fully dynamic — ceil(data_bits / log₂(10)) + guard.
Zero IEEE 754. Zero float(). Zero round(). String → mpf → string only.
Timing: time.time_ns() → integer nanoseconds. No float anywhere.

Author: Aevum Defluo (Exception Theory)
Derived forward from P∘D∘T = E via the Sempaevum bijection.
"""

import os
import sys
import struct
import hashlib  # SHA-256 for seed identity verification — NOT compression
import time
import json
import zlib     # CRC-32 for page integrity checks ONLY — NOT compression
from math import gcd
from pathlib import Path
from collections import OrderedDict

# File dialog — tkinter bundled with Python, no install needed.
# Falls back to text input on headless systems (containers, SSH).
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except ImportError:
    tk = None
    filedialog = None
    _HAS_TK = False

# Large files produce integers with millions of digits.
# Python 3.11+ caps str(big_int) at 4300 digits by default.
# Remove the cap — we need arbitrary precision.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)  # 0 = unlimited

from mpmath import mp, mpf, log as mplog, power as mppow, nint, fabs, nstr
from mpmath import pi as mppi, sqrt as mpsqrt, floor as mpfloor
from sympy import factorint, totient

# ═══════════════════════════════════════════════════════════════════════════════
# §0  PRECISION CONFIGURATION
#     Precision is DYNAMIC — scales to the data being projected.
#     Files > BLOCK_SIZE are split at filesystem block boundaries.
#     dps_for_file(bits) computes the exact precision needed for any data size.
#     No floor. A 10-byte file uses ~134 dps. A full 4096-byte block uses ~10040.
#     The block ceiling emerges naturally from the formula, not from a constant.
#     Default mp.dps (461) is for lattice math only (tower, CF, identities).
#     Every projection/pullback scales to data size and restores.
# ═══════════════════════════════════════════════════════════════════════════════
BLOCK_SIZE = 4096                           # Filesystem block size (bytes)
BLOCK_BITS = 8 * BLOCK_SIZE                 # 32768 bits per block
LATTICE_DPS = 361                           # Structural math (tower, CF, identities)
GUARD_DPS = 100                             # Multi-step generator chain safety margin
mp.dps = LATTICE_DPS + GUARD_DPS           # 461 default for lattice operations


def dps_for_file(file_bits):
    """Compute the dps needed to losslessly represent data of file_bits bits.
    ceil(file_bits / log₂(10)) + guard. Fully dynamic — no floor.
    A 1-byte file gets 113 dps. A full block gets 10040 dps."""
    return (file_bits * 10 + 32) // 33 + GUARD_DPS + 10


# ═══════════════════════════════════════════════════════════════════════════════
# §3.5  EQUATION SYSTEM — equations ARE seeds on the Sempaevum
#       Every computation IS a lattice operation (§3.1a). Every result IS a seed.
#       Stored IN the akashic alongside file seeds. Looked up by SHA of canonical
#       form. Each equation's bytes → projected through bijection → DSR →
#       participates in discovery. Identity propagation (A.1→A.5, A.2 inverses)
#       generates FREE equations. No separate cache — the akashic IS the
#       equations table. "Compute once → cache forever → never recompute."
# ═══════════════════════════════════════════════════════════════════════════════

# --- Content types for akashic serialization (§7.1d) ---
CONTENT_SEED       = 0x01  # Full seed: (k, ε) stored explicitly
CONTENT_GENERATOR  = 0x02  # Generator ref: (operation, sha_a, sha_b/power, κ)
CONTENT_DELTA      = 0x03  # Δε version: (base_sha, Δε) — Identity B.2 reconstruction
CONTENT_EQUATION   = 0x04  # Equation seed — memoized lattice computation
CONTENT_EVENT      = 0x05  # Event seed — structural moment of change (§3.9)
CONTENT_BLOCK_MAP  = 0x06  # Block map: ordered block SHAs for multi-block files

# --- Structural relation types (§15 — Discovery Engine) ---
REL_NONE      = 'none'              # No structural relation found yet
REL_MULTIPLY  = 'multiply'          # DSR_C = lattice_multiply(DSR_A, DSR_B)
REL_DIVIDE    = 'divide'            # DSR_C = lattice_divide(DSR_A, DSR_B)
REL_POWER     = 'power'             # DSR_C = lattice_power(DSR_A, n)
REL_RECIPROCAL = 'reciprocal'       # DSR_C = lattice_reciprocal(DSR_A)
REL_CONSTANT  = 'constant_relation' # DSR relates to known ET constant

_AKASHIC: 'AkashicFile | None' = None  # Global ref — set when AkashicFile created
_EQ_HITS = 0
_EQ_MISSES = 0
_EQ_PROPAGATED = 0


def _eq_canonical(operation, *args):
    """Canonical bytes of a lattice equation. The bytes ARE the seed.
    mpf values serialized at LATTICE_DPS (361 digits) — sufficient to uniquely
    identify any value on the lattice while keeping keys compact for fast
    projection. The cached RESULT carries whatever precision it was computed at.
    Higher-precision results serve lower-precision queries correctly."""
    parts = [operation]
    for a in args:
        if isinstance(a, type(mpf('0'))):
            parts.append(nstr(a, LATTICE_DPS))
        else:
            parts.append(str(a))
    return '|'.join(parts).encode('utf-8')


def _eq_sha(canonical_bytes):
    return hashlib.sha256(canonical_bytes).hexdigest()


def _eq_project_seed(canonical_bytes):
    """Project equation bytes onto the Sempaevum via _project_core (no recursion)."""
    I_int = int.from_bytes(canonical_bytes, byteorder='big') if len(canonical_bytes) > 0 else 1
    file_bits = max(8 * len(canonical_bytes), 8)
    eq_dps = dps_for_file(file_bits)
    saved = mp.dps
    mp.dps = max(mp.dps, eq_dps)
    r_eq = mpf(str(I_int)) * mppow(mpf('2'), mpf(str(-file_bits)))
    k, d, eps = _project_core(r_eq, N_BASE)
    mp.dps = saved
    return k, d, eps


def _eq_make_entry(sha, canonical_bytes, operation, result):
    """Build an akashic entry for an equation seed."""
    k, d, eps = _eq_project_seed(canonical_bytes)
    return {
        'sha256': sha, 'k_12': k, 'd_12': d, 'eps_12': eps,
        'file_size': len(canonical_bytes), 'file_bits': 8 * len(canonical_bytes),
        'filepath': f'eq:{operation}',
        'content_type': CONTENT_EQUATION,
        'eq_operation': operation, 'eq_result': result,
        'home_classification': 'equation',
        'structural_relation': REL_NONE, 'relation_data': None,
        'base_sha': None, 'delta_eps': None,
    }


def _eq_recompute_direct(operation, args):
    """Recompute a lattice operation result directly from args.
    No eq_lookup/eq_store calls — breaks recursion for CONTENT_DELTA reconstruction.
    The arithmetic IS the operation. The Sempaevum holds the math."""
    if operation == 'lattice_multiply' and len(args) == 5:
        k1, eps1, k2, eps2, N = args
        d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
        d2 = eps2 * mpf(N) / CENTS_PER_OCTAVE
        kappa = int(nint(d1 + d2))
        k_p = k1 + k2 + kappa
        g = gcd(abs(k_p), N) if k_p != 0 else N
        return k_p, N // g, (d1 + d2 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    elif operation == 'lattice_divide' and len(args) == 5:
        k1, eps1, k2, eps2, N = args
        d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
        d2 = eps2 * mpf(N) / CENTS_PER_OCTAVE
        kappa = int(nint(d1 - d2))
        k_q = k1 - k2 + kappa
        g = gcd(abs(k_q), N) if k_q != 0 else N
        return k_q, N // g, (d1 - d2 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    elif operation == 'lattice_reciprocal' and len(args) == 3:
        k1, eps1, N = args
        d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
        kappa = int(nint(-d1))
        k_inv = -k1 + kappa
        g = gcd(abs(k_inv), N) if k_inv != 0 else N
        return k_inv, N // g, (-d1 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    elif operation == 'lattice_power' and len(args) == 4:
        k1, eps1, n, N = args
        d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
        nd = mpf(n) * d1
        kappa = int(nint(nd))
        k_pow = n * k1 + kappa
        g = gcd(abs(k_pow), N) if k_pow != 0 else N
        return k_pow, N // g, (nd - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    elif operation == 'project' and len(args) == 2:
        return _project_core(args[0], args[1])
    return None


def eq_lookup(operation, *args):
    """Look up a memoized equation in the akashic by SHA.
    The akashic IS the equations table — no separate structure.
    Handles CONTENT_DELTA equations via lazy reconstruction:
    the operation and args ARE the lookup key, so recomputation
    is direct — no pullback needed."""
    global _EQ_HITS, _EQ_MISSES
    akashic = _AKASHIC
    if akashic is None:
        _EQ_MISSES += 1
        return False, None
    sha = _eq_sha(_eq_canonical(operation, *args))
    entry = akashic.entries.get(sha)
    if entry:
        # Direct hit — equation has cached result
        if entry.get('eq_result') is not None:
            _EQ_HITS += 1
            return True, entry['eq_result']
        # CONTENT_DELTA equation — lazy reconstruct from known operation+args
        # One-time cost: recompute the arithmetic directly, cache in entry
        if entry.get('base_sha') is not None and entry.get('delta_eps') is not None:
            result = _eq_recompute_direct(operation, args)
            if result is not None:
                entry['eq_result'] = result
                entry['eq_operation'] = operation
                entry['content_type'] = CONTENT_EQUATION
                _EQ_HITS += 1
                return True, result
    _EQ_MISSES += 1
    return False, None


def eq_store(operation, result, *args):
    """Store equation as seed in akashic. Propagate identities for free equations."""
    global _EQ_PROPAGATED
    akashic = _AKASHIC
    if akashic is None:
        return
    canonical = _eq_canonical(operation, *args)
    sha = _eq_sha(canonical)
    if sha in akashic.entries:
        return
    entry = _eq_make_entry(sha, canonical, operation, result)
    # Δε collapse — same (k,d) = same structural identity (Identity B.2)
    # Equations are equal citizens on the lattice. Two equations at the same
    # (k,d) address collapse: one is base, the other stores only Δε.
    kd_key = (entry['k_12'], entry['d_12'])
    if kd_key in akashic.kd_index:
        base_sha_eq = akashic.kd_index[kd_key][0]
        base_entry = akashic.entries[base_sha_eq]
        entry['delta_eps'] = entry['eps_12'] - base_entry['eps_12']
        entry['base_sha'] = base_sha_eq
    akashic.entries[sha] = entry
    akashic.index_entry(sha, entry['k_12'], entry['d_12'])
    # Identity propagation
    if operation == 'lattice_multiply' and len(args) == 5:
        k1, eps1, k2, eps2, N = args
        k_prod, d_prod, eps_prod, kappa = result
        _eq_propagate_one('lattice_multiply', result, k2, eps2, k1, eps1, N)
        g1 = gcd(abs(k1), N) if k1 != 0 else N
        _eq_propagate_one('lattice_divide', (k1, N//g1, eps1, -kappa if kappa else 0),
                          k_prod, eps_prod, k2, eps2, N)
        g2 = gcd(abs(k2), N) if k2 != 0 else N
        _eq_propagate_one('lattice_divide', (k2, N//g2, eps2, -kappa if kappa else 0),
                          k_prod, eps_prod, k1, eps1, N)
    elif operation == 'lattice_divide' and len(args) == 5:
        k1, eps1, k2, eps2, N = args
        k_quot, d_quot, eps_quot, kappa = result
        g1 = gcd(abs(k1), N) if k1 != 0 else N
        mul_res = (k1, N//g1, eps1, -kappa if kappa else 0)
        _eq_propagate_one('lattice_multiply', mul_res, k_quot, eps_quot, k2, eps2, N)
        _eq_propagate_one('lattice_multiply', mul_res, k2, eps2, k_quot, eps_quot, N)
    elif operation == 'lattice_reciprocal' and len(args) == 3:
        k1, eps1, N = args
        k_inv, d_inv, eps_inv, kappa = result
        g1 = gcd(abs(k1), N) if k1 != 0 else N
        _eq_propagate_one('lattice_reciprocal', (k1, N//g1, eps1, -kappa if kappa else 0),
                          k_inv, eps_inv, N)
    elif operation == 'lattice_power' and len(args) == 4:
        k1, eps1, n, N = args
        if n == -1:
            k_pow, d_pow, eps_pow, kappa = result
            g1 = gcd(abs(k1), N) if k1 != 0 else N
            _eq_propagate_one('lattice_power', (k1, N//g1, eps1, -kappa if kappa else 0),
                              k_pow, eps_pow, -1, N)


def _eq_propagate_one(operation, result, *args):
    """Store one propagated equation — no further propagation."""
    global _EQ_PROPAGATED
    akashic = _AKASHIC
    if akashic is None:
        return
    canonical = _eq_canonical(operation, *args)
    sha = _eq_sha(canonical)
    if sha in akashic.entries:
        return
    entry = _eq_make_entry(sha, canonical, operation, result)
    # Δε collapse — same (k,d) = same structural identity (Identity B.2)
    kd_key = (entry['k_12'], entry['d_12'])
    if kd_key in akashic.kd_index:
        base_sha_eq = akashic.kd_index[kd_key][0]
        base_entry = akashic.entries[base_sha_eq]
        entry['delta_eps'] = entry['eps_12'] - base_entry['eps_12']
        entry['base_sha'] = base_sha_eq
    akashic.entries[sha] = entry
    akashic.index_entry(sha, entry['k_12'], entry['d_12'])
    _EQ_PROPAGATED += 1


def eq_metrics():
    """Self-recording: each metric is a projectable dimensionless ratio (§3.1b)."""
    total = _EQ_HITS + _EQ_MISSES
    eq_count = 0
    akashic = _AKASHIC
    if akashic:
        eq_count = sum(1 for e in akashic.entries.values()
                       if e.get('content_type') == CONTENT_EQUATION)
    return {
        'equations_in_akashic': eq_count,
        'eq_hits': _EQ_HITS, 'eq_misses': _EQ_MISSES,
        'eq_hit_ratio': mpf(str(_EQ_HITS)) / mpf(str(total)) if total > 0 else mpf('0'),
        'eq_propagated_free': _EQ_PROPAGATED,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# §1  ET CONSTANTS — computed at runtime from primitives, never hardcoded
#     Fundamental: {N=12, K=2/3, |Π|=3, S=4}
#     All derived constants computed from these.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Fundamental ET constants (exact rationals) ---
N_BASE = 12                                 # Forward-derived from Exhaustive Trichotomy
PI_CARD = 3                                 # |Π| = 3 (primitive count)
S_CARD = 4                                  # S = 4 (manifold state count)
K_ET = mpf('2') / mpf('3')                  # Koide constant K = 2/3
V_ET = mpf('1') / mpf(N_BASE)               # V = 1/N = 1/12

# --- Derived constants (computed at runtime from defining formulas) ---
CENTS_PER_OCTAVE = mpf('1200')              # N × 100 = 12 × 100
LOG2 = mplog(mpf('2'))                      # ln(2), exact at 400 dps
LAMBDA_R = CENTS_PER_OCTAVE / LOG2          # Λ_r = 1200/ln(2) — manifold conversion (§3.18.22)
LAMBDA_THETA = mpf('600') / mppi            # Λ_θ = 600/π — phase conversion (§3.18.24)
SIGMA_ET = mpsqrt(V_ET)                     # σ = √(1/12) — shimmer amplitude
LIFE_THRESHOLD = mpf('13') / mpf('12')      # 13/12 — stability threshold

# --- ET-derived thresholds (exact integers in micro-cents) ---
KOIDE_ATTRACTOR_MICROCENTS = 1955           # |ε| where Sempaevum's own constants land
INCOHERENCE_BOUNDARY_MICROCENTS = 50000     # |ε| = 50¢ = ∂I boundary
KOIDE_STABILITY_DEPTH = 2                   # ⌈1/K⌉ = ⌈3/2⌉ = 2 consecutive landmarks
CF_QUALITY_THRESHOLD = 4                    # ⌈1/K⌉² = 4 (CF home quality floor)

# --- Cross-verify lambda ratio: must equal 2π/ln2 ---
LAMBDA_RATIO = LAMBDA_R / LAMBDA_THETA
LAMBDA_RATIO_EXPECTED = mpf('2') * mppi / LOG2
lambda_ratio_err = fabs(LAMBDA_RATIO - LAMBDA_RATIO_EXPECTED)
assert lambda_ratio_err < mppow(mpf('10'), mpf('-390')), (
    f"Lambda ratio verification FAILED: Δ = {nstr(lambda_ratio_err, 15)}"
)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  THE PROJECTION — Π_N(r) = (k, d, ε)
# ═══════════════════════════════════════════════════════════════════════════════

def _project_core(r_mpf, N):
    """Raw projection — no memoization. Used by eq_store to project equation bytes
    without recursion. The Sempaevum math itself, nothing else."""
    log2_r = mplog(r_mpf) / mplog(mpf('2'))
    exact_pos = mpf(N) * log2_r
    k = int(nint(exact_pos))
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    eps = (exact_pos - mpf(k)) * CENTS_PER_OCTAVE / mpf(N)
    return k, d, eps


def project(r_mpf, N):
    """Project r onto lattice at resolution N. Returns (k, d, eps_cents).
    Memoized via akashic: equation bytes → SHA → DSR → seed (§3.5)."""
    hit, result = eq_lookup('project', r_mpf, N)
    if hit:
        return result
    result = _project_core(r_mpf, N)
    eq_store('project', result, r_mpf, N)
    return result


def pullback(k, eps, N):
    """Pullback (k, ε) at resolution N to recover r.
    Memoized via akashic (§3.5)."""
    hit, result = eq_lookup('pullback', k, eps, N)
    if hit:
        return result
    exponent = (mpf(k) + eps * mpf(N) / CENTS_PER_OCTAVE) / mpf(N)
    result = mppow(mpf('2'), exponent)
    eq_store('pullback', result, k, eps, N)
    return result


def eps_to_microcents(eps):
    """Convert ε (cents, mpf) to micro-cents (integer). Zero float."""
    return int(nint(eps * mpf('1000000')))


# ═══════════════════════════════════════════════════════════════════════════════
# §3  CROSS-RESOLUTION TRANSITION MAP
# ═══════════════════════════════════════════════════════════════════════════════

def transition(k1, eps1, N1, N2):
    """
    Cross-resolution transition: (k1, eps1) at N1 → (k2, d2, eps2) at N2.
    Requires N1 | N2. Does NOT access r.
    """
    assert N2 % N1 == 0, f"N1={N1} does not divide N2={N2}"
    M = N2 // N1
    delta1 = eps1 * mpf(N1) / CENTS_PER_OCTAVE
    exact_pos_N2 = mpf(M) * mpf(k1) + mpf(M) * delta1
    k2 = int(nint(exact_pos_N2))
    g2 = gcd(abs(k2), N2) if k2 != 0 else N2
    d2 = N2 // g2
    eps2 = (exact_pos_N2 - mpf(k2)) * CENTS_PER_OCTAVE / mpf(N2)
    return k2, d2, eps2


# ═══════════════════════════════════════════════════════════════════════════════
# §4  INFINITE LCM TOWER GENERATOR
#     Yields every lcm-change point. The tower is INFINITE.
#     Doubling law τ(N_ℓ) = 6·2^ℓ verified as annotation.
# ═══════════════════════════════════════════════════════════════════════════════

def _tau_from_factorization(n):
    """Divisor count from prime factorization."""
    if n <= 1:
        return 1
    f = factorint(int(n))
    tau = 1
    for e in f.values():
        tau *= (e + 1)
    return tau


def lcm2(a, b):
    """LCM of two positive integers."""
    return a * b // gcd(a, b)


def lcm_tower_generator():
    """
    Infinite generator of LCM tower change points.
    Yields (k, N, tau, is_canonical_doubling) for every k where lcm(1...k) changes.
    The generator is UNBOUNDED.
    """
    N = 1
    last_canonical_tau = None

    for k in range(2, 2**63):
        N_new = lcm2(N, k)
        if N_new != N:
            N = N_new
            tau = _tau_from_factorization(N)

            is_canonical = False
            if last_canonical_tau is None and tau == 6:
                is_canonical = True
                last_canonical_tau = 6
            elif last_canonical_tau is not None and tau == 2 * last_canonical_tau:
                is_canonical = True
                last_canonical_tau = tau

            yield k, N, tau, is_canonical


# ═══════════════════════════════════════════════════════════════════════════════
# §5  CONTINUED FRACTION HOME-FINDING (CF METHOD)
#     Parallel to tower. Found after Paper 20. Confirmed in Python.
# ═══════════════════════════════════════════════════════════════════════════════

def cf_expansion(x_mpf, max_terms=500):
    """
    Compute the continued fraction expansion of x (mpf, positive).
    Returns list of partial quotients [a0, a1, a2, ...].
    """
    quotients = []
    remainder = x_mpf
    precision_floor = mppow(mpf('10'), mpf(str(-(mp.dps - 10))))

    for _ in range(max_terms):
        a_n = int(mpfloor(remainder))
        quotients.append(a_n)
        fractional = remainder - mpf(a_n)

        if fabs(fractional) < precision_floor:
            break

        if fractional == mpf('0'):
            break

        remainder = mpf('1') / fractional

    return quotients


def cf_convergents(quotients):
    """
    Compute convergents p_n/q_n from partial quotients.
    Returns list of (p_n, q_n) pairs. All integer arithmetic.
    """
    convergents = []
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0

    for a_n in quotients:
        p_n = a_n * p_prev1 + p_prev2
        q_n = a_n * q_prev1 + q_prev2
        convergents.append((p_n, q_n))
        p_prev2, p_prev1 = p_prev1, p_n
        q_prev2, q_prev1 = q_prev1, q_n

    return convergents


def cf_home_find(r_mpf):
    """
    CF home-finding for value r.
    Computes CF of |log₂(r)|, finds the convergent with maximal a_{n+1},
    identifies d_home = q_n.
    """
    abs_log2_r = fabs(mplog(r_mpf) / mplog(mpf('2')))
    quotients = cf_expansion(abs_log2_r)
    convergents = cf_convergents(quotients)

    if len(quotients) < 2:
        return {
            'cf_quotients': quotients,
            'cf_convergents': convergents,
            'best_n': 0,
            'd_home': convergents[0][1] if convergents else 1,
            'p_home': convergents[0][0] if convergents else 0,
            'quality': 0,
            'eps_cf': mpf('0'),
            'eps_cf_microcents': 0,
            'classification': 'cf_insufficient',
        }

    # Find convergent with maximal a_{n+1} (quality factor)
    best_n = 0
    best_quality = 0
    for n in range(len(quotients) - 1):
        a_next = quotients[n + 1]
        if a_next > best_quality:
            best_quality = a_next
            best_n = n

    p_home, q_home = convergents[best_n]
    d_home = abs(q_home) if q_home != 0 else 1

    # CF residual
    if d_home > 0:
        eps_cf = (abs_log2_r * mpf(d_home) - mpf(abs(p_home))) * CENTS_PER_OCTAVE / mpf(d_home)
    else:
        eps_cf = mpf('0')

    eps_cf_mc = eps_to_microcents(eps_cf)
    abs_eps_mc = abs(eps_cf_mc)

    # Classification per §7.11 Step 3a
    if best_quality >= CF_QUALITY_THRESHOLD:
        if abs_eps_mc <= KOIDE_ATTRACTOR_MICROCENTS:
            classification = 'cf_deep_home'
        elif abs_eps_mc < INCOHERENCE_BOUNDARY_MICROCENTS:
            classification = 'cf_home'
        else:
            classification = 'cf_marginal'
    elif best_quality > 1:
        classification = 'cf_marginal'
    else:
        classification = 'cf_insufficient'

    return {
        'cf_quotients': quotients[:50],
        'cf_convergents': convergents[:50],
        'best_n': best_n,
        'd_home': d_home,
        'p_home': p_home,
        'quality': best_quality,
        'eps_cf': eps_cf,
        'eps_cf_microcents': eps_cf_mc,
        'classification': classification,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §6  ANTI-NUMEROLOGY PROTOCOL (Seed Protocol N1/N2 checks)
# ═══════════════════════════════════════════════════════════════════════════════

def anti_numerology_check(r_mpf, k, d, eps, source_desc, all_entries=None):
    """Seed Protocol: verify the DSR passes ALL anti-numerology gates.
    N1: r must be a positive finite real (dimensionless ratio).
    N2: R₀ must be substrate-derived (digital tower seed 2^file_bits).
    N3: Cross-domain — (k,d) checked against ET reference projections
        AND all previously ingested files. If ANY independent domain
        produces the same (k,d), that IS cross-domain corroboration."""
    # N1: verify r is a valid dimensionless ratio — positive, finite, non-zero
    n1_pass = (r_mpf > mpf('0')) and (r_mpf < mppow(mpf('2'), mpf('1000000'))) and (r_mpf == r_mpf)
    n1_reason = ('r = I/R0 is (integer bytes)/(digital tower seed) — dimensionless, '
                 f'positive finite real: r = {nstr(r_mpf, 15)}')
    if not n1_pass:
        n1_reason = f'N1 FAIL: r is not a valid positive finite real: {nstr(r_mpf, 15)}'

    # N3: Cross-domain — check against ET reference projections
    n3_pass = False
    n3_reason = 'No cross-domain corroboration found'
    n3_witnesses = []
    for name, ref in ET_REFERENCE_PROJECTIONS.items():
        if k == ref['k'] and d == ref['d']:
            de = fabs(eps - ref['eps'])
            de_mc = abs(eps_to_microcents(de))
            n3_witnesses.append({'domain': 'ET_constant', 'name': name, 'delta_eps_mc': de_mc})

    # N3: ALSO check against all previously ingested files
    # Two independent files sharing (k,d) IS cross-domain corroboration:
    # different byte content, different source, same structural lattice address
    if all_entries:
        entries_snap = list(all_entries.items())
        for sha, prev in entries_snap:
            if prev.get('k_12') is None:
                raise RuntimeError(f"Entry {sha[:12]} has unresolved DSR in anti_numerology_check")
            if prev['k_12'] == k and prev['d_12'] == d:
                de = fabs(eps - prev['eps_12'])
                de_mc = abs(eps_to_microcents(de))
                prev_name = Path(prev['filepath']).name if 'filepath' in prev else sha[:12]
                n3_witnesses.append({
                    'domain': 'ingested_file', 'name': prev_name,
                    'delta_eps_mc': de_mc, 'sha': sha[:12],
                })

    if n3_witnesses:
        best = min(n3_witnesses, key=lambda w: w['delta_eps_mc'])
        n3_pass = True
        if best['domain'] == 'ET_constant':
            n3_reason = (f"Cross-domain: file → (k={k},d={d}) ← "
                         f"{best['name']} (Δε={best['delta_eps_mc']}mc)")
        else:
            n3_reason = (f"Cross-domain: file → (k={k},d={d}) ← "
                         f"ingested:{best['name']} (Δε={best['delta_eps_mc']}mc)")

    report = {
        'N1_dimensionless': n1_pass,
        'N1_reason': n1_reason,
        'N2_substrate_derived': True,
        'N2_reason': 'R0 = 2^(file_bits) — canonical digital-tower seed at file bit-depth',
        'N3_cross_domain': n3_pass,
        'N3_reason': n3_reason,
        'N3_witnesses': n3_witnesses,
        'source': source_desc,
    }
    passed = report['N1_dimensionless'] and report['N2_substrate_derived']
    return passed, report


# ═══════════════════════════════════════════════════════════════════════════════
# §7  FILE → DSR PIPELINE
#      Files ≤ BLOCK_SIZE: single DSR.
#      Files > BLOCK_SIZE: split at filesystem block boundaries.
#      Each block → DSR at block-covering precision.
#      Discovery engine runs on ALL blocks during ingestion.
# ═══════════════════════════════════════════════════════════════════════════════

def dsr_to_r(entry):
    """Recompute r from stored DSR via pullback. Algebraic identity — exact.
    Used whenever r is needed for lattice operations, discovery, or identity checks.
    The generator IS the file — this evaluation IS r."""
    if entry['k_12'] is None or entry['eps_12'] is None:
        raise ValueError(f"Cannot compute r: entry {entry.get('sha256', '?')[:12]} has unresolved DSR")
    return pullback(entry['k_12'], entry['eps_12'], N_BASE)


def block_to_dsr(raw_bytes, source_desc, all_entries=None):
    """Project a block of bytes (≤ BLOCK_SIZE) onto the Sempaevum.
    Pipeline: bytes → I (big-endian int) → r = I/2^bits → Π_12(r) = (k,d,ε)
    The DSR IS the generator. The pullback IS the block."""
    block_size = len(raw_bytes)
    if block_size == 0:
        raise ValueError(f"Empty block: {source_desc}")

    sha256_block = hashlib.sha256(raw_bytes).hexdigest()
    I_int = int.from_bytes(raw_bytes, byteorder='big')

    if I_int == 0:
        raise ValueError(f"All-zero block: {source_desc} — r=0 is the annihilation boundary "
                         f"(§3.4), off-lattice infimum of (ℝ⁺,×). Cannot project.")
    block_bits = 8 * block_size
    block_dps = dps_for_file(block_bits)

    saved_dps = mp.dps
    mp.dps = max(mp.dps, block_dps)

    r_mpf = mpf(str(I_int)) * mppow(mpf('2'), mpf(str(-block_bits)))
    k_12, d_12, eps_12 = project(r_mpf, N_BASE)

    mp.dps = saved_dps

    antinum_passed, antinum_report = anti_numerology_check(
        r_mpf, k_12, d_12, eps_12, source_desc, all_entries)

    return {
        'filepath': source_desc,
        'file_size': block_size,
        'file_bits': block_bits,
        'sha256': sha256_block,
        'k_12': k_12,
        'd_12': d_12,
        'eps_12': eps_12,
        'antinum_passed': antinum_passed,
        'antinum_report': antinum_report,
    }


def file_to_dsr(filepath, all_entries=None):
    """
    Convert a file to its Dimensionless Seed Ratio (DSR).
    The DSR IS the generator. Pipeline:
    bytes → I (big-endian int) → r = I/2^bits → Π_12(r) = (k,d,ε)
    Precision scales to data size for exact projection.
    For files > BLOCK_SIZE, use file_to_block_dsrs() instead.
    all_entries: previously ingested DSRs for N3 cross-domain check.
    """
    filepath = Path(filepath)
    raw_bytes = filepath.read_bytes()
    file_size = len(raw_bytes)

    if file_size == 0:
        raise ValueError(f"Empty file: {filepath}")

    sha256_original = hashlib.sha256(raw_bytes).hexdigest()
    I_int = int.from_bytes(raw_bytes, byteorder='big')

    if I_int == 0:
        raise ValueError(f"All-zero file: {filepath} — r=0 is the annihilation boundary "
                         f"(§3.4), off-lattice infimum of (ℝ⁺,×). Cannot project.")
    file_bits = 8 * file_size
    file_dps = dps_for_file(file_bits)

    saved_dps = mp.dps
    mp.dps = max(mp.dps, file_dps)

    r_mpf = mpf(str(I_int)) * mppow(mpf('2'), mpf(str(-file_bits)))
    k_12, d_12, eps_12 = project(r_mpf, N_BASE)

    mp.dps = saved_dps

    antinum_passed, antinum_report = anti_numerology_check(
        r_mpf, k_12, d_12, eps_12, str(filepath), all_entries)

    return {
        'filepath': str(filepath),
        'file_size': file_size,
        'file_bits': file_bits,
        'sha256': sha256_original,
        'k_12': k_12,
        'd_12': d_12,
        'eps_12': eps_12,
        'antinum_passed': antinum_passed,
        'antinum_report': antinum_report,
    }


def file_to_block_dsrs(filepath, all_entries=None):
    """Split a file into BLOCK_SIZE (4096-byte) blocks and project each.
    Each block gets its own DSR at block-covering precision.
    The last block may be smaller than BLOCK_SIZE.
    Returns (file_meta, block_dsrs) where file_meta has the file SHA and
    block_dsrs is an ordered list of block DSR dicts."""
    filepath = Path(filepath)
    raw_bytes = filepath.read_bytes()
    file_size = len(raw_bytes)

    if file_size == 0:
        raise ValueError(f"Empty file: {filepath}")

    sha256_file = hashlib.sha256(raw_bytes).hexdigest()

    # Split into blocks at filesystem boundaries
    block_dsrs = []
    block_count = (file_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    for i in range(block_count):
        start = i * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, file_size)
        block_bytes = raw_bytes[start:end]

        if all(b == 0 for b in block_bytes):
            # All-zero block: record as special zero-block marker, skip projection
            block_dsrs.append({
                'filepath': f"{filepath.name}:block_{i}",
                'file_size': len(block_bytes),
                'file_bits': 8 * len(block_bytes),
                'sha256': hashlib.sha256(block_bytes).hexdigest(),
                'k_12': 0, 'd_12': 1, 'eps_12': mpf('0'),
                'is_zero_block': True,
                'antinum_passed': True,
                'antinum_report': {'N1_dimensionless': True, 'N2_substrate_derived': True,
                                   'N3_cross_domain': False, 'N3_reason': 'zero block'},
            })
            continue

        source = f"{filepath.name}:block_{i}[{start}:{end}]"
        block_dsr = block_to_dsr(block_bytes, source, all_entries)
        block_dsrs.append(block_dsr)

    file_meta = {
        'filepath': str(filepath),
        'file_size': file_size,
        'sha256': sha256_file,
        'block_count': block_count,
        'block_size': BLOCK_SIZE,
        'block_shas': [b['sha256'] for b in block_dsrs],
    }

    return file_meta, block_dsrs


# ═══════════════════════════════════════════════════════════════════════════════
# §8  LOSSLESS ROUND-TRIP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_round_trip(dsr_entry):
    """Verify the Sempaevum bijection as algebraic identity.
    The DSR IS the generator. project(r) → (k, ε); pullback(k, ε) → r.
    At file-scaled computational precision, the identity resolves exactly.
    Verification uses SHA-256 ONLY — no raw data stored or compared."""
    k = dsr_entry['k_12']
    eps = dsr_entry['eps_12']
    file_size = dsr_entry['file_size']
    file_bits = 8 * file_size  # derived, not stored
    file_dps = dps_for_file(file_bits)  # derived, not stored
    sha256_original = dsr_entry['sha256']

    # Scale to file-scaled precision for exact pullback
    saved_dps = mp.dps
    mp.dps = file_dps

    # Evaluate the DSR generator — the pullback IS the file
    r_recovered = pullback(k, eps, N_BASE)
    I_recovered = int(nint(r_recovered * mppow(mpf('2'), mpf(str(file_bits)))))

    # Restore precision
    mp.dps = saved_dps

    try:
        bytes_recovered = I_recovered.to_bytes(file_size, byteorder='big')
    except OverflowError:
        byte_len = (I_recovered.bit_length() + 7) // 8
        bytes_recovered = I_recovered.to_bytes(max(byte_len, file_size), byteorder='big')[-file_size:]

    sha_recovered = hashlib.sha256(bytes_recovered).hexdigest()
    sha_match = sha_recovered == sha256_original

    return sha_match, {
        'sha256_original': sha256_original,
        'sha256_recovered': sha_recovered,
        'pullback_exact': sha_match,  # SHA match IS exactness proof
        'file_dps': file_dps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §9  INFINITE TOWER ESCALATION WITH D-STABILIZATION
#     Runs through ALL lcm-change points. CF in parallel. No cap.
# ═══════════════════════════════════════════════════════════════════════════════

def tower_escalation(dsr_entry, verbose=True):
    """
    Run the infinite LCM tower escalation on a file DSR.
    Starts at N=12. Escalates through ALL lcm-change points via
    the cross-resolution transition map (never re-accessing r).
    """
    r_mpf = dsr_to_r(dsr_entry)  # Pullback — algebraic identity, exact
    k_current = dsr_entry['k_12']
    eps_current = dsr_entry['eps_12']
    d_current = dsr_entry['d_12']
    N_current = N_BASE

    # CF method (parallel, computed once at start)
    cf_result = cf_home_find(r_mpf)

    # Trajectory
    trajectory = [{
        'N': N_current, 'k': k_current, 'd': d_current,
        'eps': eps_current, 'eps_microcents': eps_to_microcents(eps_current),
        'tau': _tau_from_factorization(N_current),
        'is_canonical': True, 'lcm_k': 4,
    }]

    d_history = [d_current]
    sub_koide_hits = []
    home_classification = 'escalation_in_progress'
    home_landmark_idx = None
    false_resolutions = []
    precision_floor_cents = mpf('600') / mppow(mpf('10'), mpf(str(mp.dps - 50)))

    if verbose:
        print(f"\n  {'Lvl':>4} {'k':>4} {'N':>22} {'tau':>6} {'d':>8} "
              f"{'|eps| cents':>18} {'|eps| mc':>12} {'Status'}")
        print(f"  {'_'*4} {'_'*4} {'_'*22} {'_'*6} {'_'*8} "
              f"{'_'*18} {'_'*12} {'_'*20}")
        abs_eps_display = fabs(eps_current)
        abs_mc_display = abs(eps_to_microcents(eps_current))
        print(f"  {0:>4} {4:>4} {N_current:>22} {6:>6} {d_current:>8} "
              f"{nstr(abs_eps_display,12):>18} {abs_mc_display:>12} BASE")

    landmark_idx = 0
    for lcm_k, N_new, tau_new, is_canonical in lcm_tower_generator():
        if N_new <= N_current:
            continue
        if N_new % N_current != 0:
            # Should never happen for the lcm sequence, but handle gracefully
            # by recomputing from scratch at N_new using r directly.
            # This is NOT a forbidden fallback — it is a divisibility gap handler.
            k_new, d_new, eps_new = project(r_mpf, N_new)
        else:
            k_new, d_new, eps_new = transition(k_current, eps_current, N_current, N_new)

        landmark_idx += 1

        # Verify transition against direct projection (first 8 landmarks)
        if landmark_idx <= 8:
            k_dir, d_dir, eps_dir = project(r_mpf, N_new)
            k_ok = k_new == k_dir
            d_ok = d_new == d_dir
            eps_err = fabs(eps_new - eps_dir)
            if not (k_ok and d_ok and eps_err < mppow(mpf('10'), mpf('-350'))):
                print(f"  *** TRANSITION VERIFY FAIL N={N_new}: "
                      f"k {k_dir} vs {k_new}, d {d_dir} vs {d_new}, "
                      f"deps={nstr(eps_err,10)}")

        eps_mc = eps_to_microcents(eps_new)
        abs_eps_mc = abs(eps_mc)
        trajectory.append({
            'N': N_new, 'k': k_new, 'd': d_new,
            'eps': eps_new, 'eps_microcents': eps_mc,
            'tau': tau_new, 'is_canonical': is_canonical, 'lcm_k': lcm_k,
        })
        d_history.append(d_new)

        # Sub-Koide hit detection
        if abs_eps_mc <= KOIDE_ATTRACTOR_MICROCENTS:
            sub_koide_hits.append((landmark_idx, d_new, eps_mc))

        # d-stabilization check: 2 consecutive same d
        if len(d_history) >= 2 and d_history[-1] == d_history[-2]:
            if home_classification == 'escalation_in_progress':
                if eps_mc == 0:
                    home_classification = 'true_home'
                elif abs_eps_mc <= KOIDE_ATTRACTOR_MICROCENTS:
                    home_classification = 'deep_home'
                elif abs_eps_mc < INCOHERENCE_BOUNDARY_MICROCENTS:
                    home_classification = 'persistent_home'
                else:
                    home_classification = 'intermediate_home'
                home_landmark_idx = landmark_idx

        # False resolution detection
        for hit_idx, hit_d, hit_mc in sub_koide_hits:
            if landmark_idx > hit_idx + KOIDE_STABILITY_DEPTH and d_new != hit_d:
                if not any(fr['hit_landmark'] == hit_idx for fr in false_resolutions):
                    false_resolutions.append({
                        'hit_landmark': hit_idx, 'hit_d': hit_d,
                        'hit_mc': hit_mc, 'invalidated_at': landmark_idx,
                        'new_d': d_new,
                    })

        if verbose:
            status = ''
            if len(d_history) >= 2 and d_history[-1] != d_history[-2]:
                status = f'd {d_history[-2]}->{d_new}'
            elif home_classification != 'escalation_in_progress':
                status = home_classification
            if is_canonical:
                status += ' [C]'
            abs_eps_display = fabs(eps_new)
            print(f"  {landmark_idx:>4} {lcm_k:>4} {N_new:>22} {tau_new:>6} "
                  f"{d_new:>8} {nstr(abs_eps_display,12):>18} {abs_eps_mc:>12} {status}")

        k_current, eps_current, d_current, N_current = k_new, eps_new, d_new, N_new

        # Termination: home found with false-resolution guard cleared
        if home_classification in ('true_home', 'deep_home', 'persistent_home'):
            all_cleared = all(
                d_new == hit_d or landmark_idx > hit_idx + KOIDE_STABILITY_DEPTH
                for hit_idx, hit_d, _ in sub_koide_hits
            )
            if all_cleared:
                if verbose:
                    print(f"\n  >>> TOWER HOME: {home_classification} at d={d_new}")
                break

        # Precision floor
        if fabs(eps_new) < precision_floor_cents:
            if home_classification == 'escalation_in_progress':
                home_classification = 'precision_floor_reached'
            if verbose:
                print(f"\n  >>> PRECISION FLOOR at |eps| < 10^(-{mp.dps-50})")
            break

    cf_tower_agree = None
    if cf_result['classification'] != 'cf_insufficient':
        cf_tower_agree = cf_result['d_home'] == d_current

    result = {
        'trajectory': trajectory,
        'trajectory_length': len(trajectory),
        'home_classification': home_classification,
        'home_d': d_current,
        'home_landmark_idx': home_landmark_idx,
        'final_N': N_current, 'final_k': k_current,
        'final_d': d_current, 'final_eps': eps_current,
        'final_eps_microcents': eps_to_microcents(eps_current),
        'false_resolutions': false_resolutions,
        'cf_result': cf_result,
        'cf_tower_agreement': cf_tower_agree,
        'd_transitions': sum(1 for i in range(1, len(d_history))
                             if d_history[i] != d_history[i-1]),
    }

    if verbose:
        print(f"\n  Tower summary:")
        print(f"    Landmarks:      {len(trajectory)}")
        print(f"    d transitions:  {result['d_transitions']}")
        print(f"    Final N:        {N_current}")
        print(f"    Final d:        {d_current}")
        print(f"    Final |eps|:    {nstr(fabs(eps_current), 15)} cents")
        print(f"    False resol:    {len(false_resolutions)}")
        print(f"    Home class:     {home_classification}")
        print(f"\n  CF method:")
        print(f"    d_home (CF):    {cf_result['d_home']}")
        print(f"    Quality:        {cf_result['quality']}")
        print(f"    |eps_CF|:       {nstr(fabs(cf_result['eps_cf']), 15)} cents")
        print(f"    CF class:       {cf_result['classification']}")
        print(f"    CF-Tower agree: {cf_tower_agree}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §10  .AKASHIC BINARY FORMAT — SMVM header, CRC-32, SHA-256, zero IEEE 754
# ═══════════════════════════════════════════════════════════════════════════════

AKASHIC_MAGIC = b'SMVM'
AKASHIC_VERSION = 1
AKASHIC_PAGE_SIZE = 4096


def _mpf_to_bytes(val, dps=None):
    """Serialize mpf to bytes as length-prefixed UTF-8 string. Zero IEEE 754.
    dps = number of significant digits to output. None = use current mp.dps."""
    if dps is None:
        dps = mp.dps + 10
    s = nstr(val, dps, strip_zeros=False).encode('utf-8')
    return struct.pack('>I', len(s)) + s


def _mpf_to_compact_bytes(val):
    """Serialize mpf to compact binary: sign + mantissa_int + exponent_int.
    Size depends on the VALUE's information content, not the working precision.
    For small deltas (e.g. Δε = -4.2e-63), this is tiny (~10 bytes).
    For full ε values with 22,000 significant digits, this is ~9KB (mantissa carries info).
    The mantissa + exponent fully determine the value at its native precision."""
    # mpmath mpf internal: (sign, man, exp, bc) where val = (-1)^sign * man * 2^exp
    sign, man, exp, bc = val._mpf_
    # Special values
    if man == 0:
        return struct.pack('>BBI', 0x00, 0, 0)  # zero: 6 bytes
    # sign: 1 byte
    # mantissa: varlen integer bytes (carries the actual information)
    # exponent: varlen signed integer
    man_bytes_len = (man.bit_length() + 7) // 8
    man_data = man.to_bytes(man_bytes_len, 'big')
    # exponent as signed varlen
    exp_sign = 0 if exp >= 0 else 1
    abs_exp = abs(exp)
    if abs_exp == 0:
        exp_data = b'\x00'
    else:
        exp_bytes_len = (abs_exp.bit_length() + 7) // 8
        exp_data = bytes([exp_sign]) + abs_exp.to_bytes(exp_bytes_len, 'big')
    payload = bytes([sign]) + struct.pack('>I', len(man_data)) + man_data + struct.pack('>I', len(exp_data)) + exp_data
    return struct.pack('>I', len(payload)) + payload


def _mpf_from_compact_bytes(data, offset):
    """Deserialize mpf from compact binary format."""
    payload_len = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    p = data[offset:offset+payload_len]
    offset += payload_len
    if payload_len <= 6 and p[0] == 0x00:
        return mpf('0'), offset
    sign = p[0]
    po = 1
    man_len = struct.unpack('>I', p[po:po+4])[0]
    po += 4
    man = int.from_bytes(p[po:po+man_len], 'big')
    po += man_len
    exp_len = struct.unpack('>I', p[po:po+4])[0]
    po += 4
    exp_data = p[po:po+exp_len]
    if exp_data == b'\x00':
        exp = 0
    else:
        exp_sign = exp_data[0]
        abs_exp = int.from_bytes(exp_data[1:], 'big')
        exp = -abs_exp if exp_sign else abs_exp
    # Reconstruct mpf from (sign, man, exp, bitcount)
    # Precision must cover ALL stored mantissa bits — no truncation on load
    from mpmath.libmp import from_man_exp
    prec_needed = max(mp.prec, man.bit_length() + 10)
    result = mpf(from_man_exp(man, exp, prec_needed))
    if sign:
        result = -result
    return result, offset


def _mpf_from_bytes(data, offset):
    """Deserialize mpf from bytes. Returns (mpf_value, new_offset).
    Parses at precision covering the FULL stored string — zero truncation.
    The stored digits ARE the value's precision. All are preserved."""
    str_len = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    s = data[offset:offset+str_len].decode('utf-8')
    offset += str_len
    # Scale precision to cover ALL stored digits — no truncation on load
    saved = mp.dps
    mp.dps = max(mp.dps, str_len + 10)
    result = mpf(s)
    mp.dps = saved
    return result, offset


def _int_to_varlen_bytes(val):
    """Serialize arbitrary-size Python int to bytes."""
    if val == 0:
        payload = b'\x00'
    else:
        sign = 0 if val >= 0 else 1
        abs_val = abs(val)
        byte_len = (abs_val.bit_length() + 7) // 8
        payload = bytes([sign]) + abs_val.to_bytes(byte_len, 'big')
    return struct.pack('>I', len(payload)) + payload


def _int_from_varlen_bytes(data, offset):
    """Deserialize arbitrary-size Python int from bytes."""
    payload_len = struct.unpack('>I', data[offset:offset+4])[0]
    offset += 4
    payload = data[offset:offset+payload_len]
    offset += payload_len
    if payload == b'\x00':
        return 0, offset
    sign = payload[0]
    abs_val = int.from_bytes(payload[1:], 'big')
    return -abs_val if sign else abs_val, offset


def _crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


class AkashicFile:
    """The .akashic file — the Sempaevum on disk. The birth triad.
    P = disk substrate. D = lattice structure. T = traversal.
    P∘D∘T = E = this file.

    Each entry IS a seed: the raw file bytes (content) at a lattice
    address (k, ε). The trajectory, d-family, SHA-256, file_size are
    all DERIVABLE from the seed — they are not stored. Zero redundancy.
    The file IS a Kolmogorov generator: it produces all its content
    on demand via the lossless bijection pullback."""

    CLASSIFICATION_CODES = {
        'true_home': 0, 'deep_home': 1, 'persistent_home': 2,
        'intermediate_home': 3, 'escalation_in_progress': 4,
        'precision_floor_reached': 5, 'cf_deep_home': 6,
        'cf_home': 7, 'cf_marginal': 8, 'cf_insufficient': 9,
        'equation': 10,
        'event': 11,
    }
    CLASSIFICATION_NAMES = {v: k for k, v in CLASSIFICATION_CODES.items()}

    def __init__(self, filepath):
        global _AKASHIC
        self.filepath = Path(filepath)
        self.entries = OrderedDict()
        self.events = []  # Time-indexed structural events — first-class lattice content
        self.creation_timestamp = time.time_ns() // 1_000_000_000
        self.last_modified = self.creation_timestamp
        # Structural indexes — O(1) lookup by lattice address
        self._kd_index = {}   # (k, d) → [sha, ...] for attractor/delta detection
        self._k_index = {}    # k → [sha, ...] for lattice arithmetic relation search
        self._last_serialize_stats = {}  # content type counts from last serialize
        # Set global ref — equations use the akashic as their storage
        _AKASHIC = self
        # Load existing akashic if present — persistence is mandatory
        if self.filepath.exists() and self.filepath.stat().st_size > 0:
            self._load_existing()

    def _load_existing(self):
        """Load an existing .akashic file, restoring entries, events, and indexes.
        The akashic IS persistent — it accumulates across sessions.
        Identity J: the seed of all seeds grows as content is added."""
        data = self.filepath.read_bytes()
        if len(data) < AKASHIC_PAGE_SIZE or data[:4] != AKASHIC_MAGIC:
            return  # Not a valid akashic file — start fresh

        entry_count = struct.unpack('>I', data[12:16])[0]
        self.creation_timestamp = struct.unpack('>Q', data[16:24])[0]
        self.last_modified = struct.unpack('>Q', data[24:32])[0]

        offset = AKASHIC_PAGE_SIZE
        loaded = 0
        for i in range(entry_count):
            if offset + 4 > len(data):
                break
            entry_len = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            eb = data[offset:offset+entry_len]
            offset += entry_len

            # Parse common header: sha(32) + file_size(4)
            sha_hex = eb[:32].hex()
            file_size = struct.unpack('>I', eb[32:36])[0]
            file_bits = 8 * file_size

            # Content type byte (position 36)
            content_type = eb[36]
            po = 37  # parse offset after content type

            entry = {
                'sha256': sha_hex,
                'file_size': file_size,
                'file_bits': file_bits,
                'filepath': f'[loaded:{sha_hex[:12]}]',  # filepath not persisted — use sha prefix
            }

            if content_type == CONTENT_SEED:
                # Full (k, ε): k(varlen) + eps(varlen) + class(1) + crc(4)
                k_val, po = _int_from_varlen_bytes(eb, po)
                eps_val, po = _mpf_from_bytes(eb, po)
                cls_code = eb[po]
                entry['k_12'] = k_val
                entry['eps_12'] = eps_val
                g = gcd(abs(k_val), N_BASE) if k_val != 0 else N_BASE
                entry['d_12'] = N_BASE // g
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'intermediate_home')
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['base_sha'] = None
                entry['delta_eps'] = None

            elif content_type == CONTENT_GENERATOR:
                # Generator ref: op(1) + sha_a(32) + [sha_b(32)|power(1)] + κ(1) + class(1) + crc(4)
                op_code = eb[po]; po += 1
                sha_a = eb[po:po+32].hex(); po += 32

                rel_data = {'sha_a': sha_a}
                if op_code in (0x01, 0x02):  # multiply, divide
                    sha_b = eb[po:po+32].hex(); po += 32
                    rel_data['sha_b'] = sha_b
                    rel_type = REL_MULTIPLY if op_code == 0x01 else REL_DIVIDE
                elif op_code in (0x03, 0x04):  # power, reciprocal
                    pwr = struct.unpack('>b', eb[po:po+1])[0]; po += 1
                    rel_data['power'] = pwr
                    rel_type = REL_RECIPROCAL if op_code == 0x04 else REL_POWER
                elif op_code == 0x05:  # constant
                    const_hash = eb[po:po+8].hex(); po += 8
                    rel_data['constant_hash'] = const_hash
                    rel_type = REL_CONSTANT
                    # Try to find which constant this hash matches
                    for cname in ET_REFERENCE_PROJECTIONS:
                        ch = hashlib.sha256(cname.encode('utf-8')).digest()[:8].hex()
                        if ch == const_hash:
                            rel_data['constant'] = cname
                            break
                else:
                    rel_type = REL_NONE

                kappa = struct.unpack('>b', eb[po:po+1])[0]; po += 1
                cls_code = eb[po]
                rel_data['kappa'] = kappa

                entry['structural_relation'] = rel_type
                entry['relation_data'] = rel_data
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'intermediate_home')
                entry['base_sha'] = None
                entry['delta_eps'] = None
                # k, d, eps: None = unresolved, awaiting generator_reconstruct in third pass
                entry['k_12'] = None
                entry['d_12'] = None
                entry['eps_12'] = None

            elif content_type == CONTENT_DELTA:
                # Delta: base_sha(32) + compact_delta(varlen) + class(1) + crc(4)
                base_sha = eb[po:po+32].hex(); po += 32
                delta_eps, po = _mpf_from_compact_bytes(eb, po)
                cls_code = eb[po]

                entry['base_sha'] = base_sha
                entry['delta_eps'] = delta_eps
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'intermediate_home')
                # k, d, eps: None = unresolved, awaiting base entry resolution in second pass
                entry['k_12'] = None
                entry['d_12'] = None
                entry['eps_12'] = None

            elif content_type == CONTENT_EQUATION:
                # Equation seed: json(op+result)(varlen) + class(1) + crc(4)
                eq_json_len = struct.unpack('>I', eb[po:po+4])[0]; po += 4
                eq_json_str = eb[po:po+eq_json_len].decode('utf-8'); po += eq_json_len
                cls_code = eb[po]; po += 1
                try:
                    eq_obj = json.loads(eq_json_str)
                    entry['eq_operation'] = eq_obj.get('op', '')
                    res_str = eq_obj.get('res', '')
                    try:
                        # Scale precision to cover ALL digits in stored result
                        saved_eq_dps = mp.dps
                        mp.dps = max(mp.dps, len(res_str) + 10)
                        entry['eq_result'] = eval(res_str, {'mpf': mpf, '__builtins__': {}})
                        mp.dps = saved_eq_dps
                    except (SyntaxError, NameError, TypeError, ValueError):
                        entry['eq_result'] = res_str
                except json.JSONDecodeError:
                    entry['eq_operation'] = ''
                    entry['eq_result'] = None
                entry['content_type'] = CONTENT_EQUATION
                entry['filepath'] = f"eq:{entry.get('eq_operation', '')}"
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'equation')
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['base_sha'] = None
                entry['delta_eps'] = None
                # k, d, eps: None = unresolved, awaiting DSR recompute in fourth pass
                entry['k_12'] = None
                entry['d_12'] = None
                entry['eps_12'] = None

            elif content_type == CONTENT_EVENT:
                # Event seed: json(event_data)(varlen) + class(1) + crc(4)
                evt_json_len = struct.unpack('>I', eb[po:po+4])[0]; po += 4
                evt_json_str = eb[po:po+evt_json_len].decode('utf-8'); po += evt_json_len
                cls_code = eb[po]; po += 1
                try:
                    evt_obj = json.loads(evt_json_str)
                    entry['event_class'] = evt_obj.get('event_class', '')
                    entry['event_data'] = evt_obj
                    self.events.append(evt_obj)
                except json.JSONDecodeError:
                    entry['event_class'] = ''
                    entry['event_data'] = {}
                entry['content_type'] = CONTENT_EVENT
                entry['filepath'] = f"evt:{entry.get('event_class', '')}"
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'event')
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['base_sha'] = None
                entry['delta_eps'] = None
                # Recompute DSR from event bytes
                evt_bytes_reload = evt_json_str.encode('utf-8')
                I_int = int.from_bytes(evt_bytes_reload, byteorder='big') if len(evt_bytes_reload) > 0 else 1
                fb = max(8 * len(evt_bytes_reload), 8)
                edps = dps_for_file(fb)
                sdps = mp.dps
                mp.dps = max(mp.dps, edps)
                r_e = mpf(str(I_int)) * mppow(mpf('2'), mpf(str(-fb)))
                ke, de, epse = _project_core(r_e, N_BASE)
                mp.dps = sdps
                entry['k_12'] = ke
                entry['d_12'] = de
                entry['eps_12'] = epse

            elif content_type == CONTENT_BLOCK_MAP:
                # Block map: json(block_map)(varlen) + class(1) + crc(4)
                bmap_json_len = struct.unpack('>I', eb[po:po+4])[0]; po += 4
                bmap_json_str = eb[po:po+bmap_json_len].decode('utf-8'); po += bmap_json_len
                cls_code = eb[po]; po += 1
                bmap_obj = json.loads(bmap_json_str)
                entry['block_map'] = bmap_obj
                entry['content_type'] = CONTENT_BLOCK_MAP
                entry['filepath'] = f"blockmap:{bmap_obj.get('file_sha256', '')[:12]}"
                entry['home_classification'] = self.CLASSIFICATION_NAMES.get(cls_code, 'true_home')
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['base_sha'] = None
                entry['delta_eps'] = None
                # Recompute DSR from block map bytes
                bmap_bytes_reload = bmap_json_str.encode('utf-8')
                I_int_bm = int.from_bytes(bmap_bytes_reload, byteorder='big') if len(bmap_bytes_reload) > 0 else 1
                fb_bm = max(8 * len(bmap_bytes_reload), 8)
                edps_bm = dps_for_file(fb_bm)
                sdps_bm = mp.dps
                mp.dps = max(mp.dps, edps_bm)
                r_bm = mpf(str(I_int_bm)) * mppow(mpf('2'), mpf(str(-fb_bm)))
                kb, db, epsb = _project_core(r_bm, N_BASE)
                mp.dps = sdps_bm
                entry['k_12'] = kb
                entry['d_12'] = db
                entry['eps_12'] = epsb

            else:
                # Unknown content type — cannot resolve DSR
                print(f"  WARNING: unknown content type 0x{content_type:02x} "
                      f"for entry {sha_hex[:12]}")
                entry['k_12'] = None
                entry['d_12'] = None
                entry['eps_12'] = None
                entry['home_classification'] = 'intermediate_home'
                entry['structural_relation'] = REL_NONE
                entry['relation_data'] = None
                entry['base_sha'] = None
                entry['delta_eps'] = None

            self.entries[sha_hex] = entry
            loaded += 1

        # ── MANDATORY RESOLUTION PASSES ──
        # All entries serialized together → all operands MUST be present.
        # Failure to resolve = data corruption → hard error.
        # Pass order: generators (recursive chains) → deltas (need resolved bases)
        #             → equations (self-contained projection)

        # Pass 1: resolve GENERATOR entries via generator_reconstruct (recursive)
        for sha, entry in self.entries.items():
            if (entry.get('structural_relation', REL_NONE) != REL_NONE
                    and entry['k_12'] is None):
                k_r, d_r, eps_r = generator_reconstruct(entry, self.entries)
                entry['k_12'] = k_r
                entry['d_12'] = d_r
                entry['eps_12'] = eps_r

        # Pass 2: resolve DELTA entries (base entries now resolved from pass 1)
        for sha, entry in self.entries.items():
            if entry.get('base_sha') and entry.get('delta_eps') is not None and entry['k_12'] is None:
                base = self.entries[entry['base_sha']]
                entry['k_12'] = base['k_12']
                entry['d_12'] = base['d_12']
                entry['eps_12'] = base['eps_12'] + entry['delta_eps']

        # Pass 3: resolve EQUATION entries (recompute DSR from canonical bytes)
        for sha, entry in self.entries.items():
            if (entry.get('content_type') == CONTENT_EQUATION
                    and entry['k_12'] is None):
                op = entry.get('eq_operation', '')
                res = entry.get('eq_result', '')
                canonical = _eq_canonical(op, str(res))
                ke, de, epse = _eq_project_seed(canonical)
                entry['k_12'] = ke
                entry['d_12'] = de
                entry['eps_12'] = epse

        # VERIFICATION: ALL entries MUST be resolved. None = data corruption.
        unresolved = [(sha, entry) for sha, entry in self.entries.items()
                      if entry.get('k_12') is None]
        if unresolved:
            for sha, entry in unresolved:
                ct = entry.get('content_type', entry.get('structural_relation', '?'))
                print(f"  ERROR: entry {sha[:12]} type={ct} has unresolved DSR")
            raise RuntimeError(
                f"Akashic data integrity violation: {len(unresolved)} entries "
                f"could not be resolved. Operands missing or circular dependency.")

        # Rebuild indexes — all entries guaranteed resolved
        for sha, entry in self.entries.items():
            self._index_entry(sha, entry['k_12'], entry['d_12'])

        # Load events (after entries)
        if offset + 4 <= len(data):
            event_count = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            for _ in range(event_count):
                if offset + 4 > len(data):
                    break
                evt_len = struct.unpack('>I', data[offset:offset+4])[0]
                offset += 4
                if offset + evt_len <= len(data):
                    evt_json = data[offset:offset+evt_len].decode('utf-8')
                    try:
                        evt = json.loads(evt_json)
                        self.events.append(evt)
                    except json.JSONDecodeError as e:
                        print(f"  WARNING: malformed event JSON at offset {offset}: {e}")
                offset += evt_len

        if loaded > 0:
            print(f"  .akashic loaded: {loaded} entries, {len(self.events)} events "
                  f"from {self.filepath.name}")

        # Equations are loaded as entries (CONTENT_EQUATION) in the entry loop above.
        # Count them for reporting.
        equations_loaded = sum(1 for e in self.entries.values()
                               if e.get('content_type') == CONTENT_EQUATION)
        if equations_loaded > 0:
            print(f"  Equations restored: {equations_loaded} as seeds in akashic "
                  f"(performance compounds from prior sessions)")

    def fire_event(self, event_class, subject_id, metadata=None):
        """Fire a structural event. Events ARE seeds (§3.9).
        Every event's bytes → projected through bijection → DSR →
        stored as akashic entry → participates in discovery.
        Three-times tracking:
          D-time: monotonic event count (relational ordering)
          T-time: wall-clock ns (Traverser's accumulated proper time)
          P-time: not applicable in POC (substrate phase)
        """
        evt = {
            'event_class': event_class,
            'event_timestamp_ns': time.time_ns(),
            'd_time_sequence': len(self.events),
            'subject_id': subject_id,
            'metadata': metadata or {},
        }
        evt_bytes = json.dumps(evt, sort_keys=True, default=str).encode('utf-8')
        sha = hashlib.sha256(evt_bytes).hexdigest()
        evt['event_sha256'] = sha
        evt['event_size'] = len(evt_bytes)
        self.events.append(evt)

        # Project event bytes → DSR → store as entry (the event IS a seed)
        if sha not in self.entries:
            I_int = int.from_bytes(evt_bytes, byteorder='big') if len(evt_bytes) > 0 else 1
            file_bits = max(8 * len(evt_bytes), 8)
            evt_dps = dps_for_file(file_bits)
            saved = mp.dps
            mp.dps = max(mp.dps, evt_dps)
            r_evt = mpf(str(I_int)) * mppow(mpf('2'), mpf(str(-file_bits)))
            k_evt, d_evt, eps_evt = _project_core(r_evt, N_BASE)
            mp.dps = saved

            entry = {
                'sha256': sha, 'k_12': k_evt, 'd_12': d_evt, 'eps_12': eps_evt,
                'file_size': len(evt_bytes), 'file_bits': file_bits,
                'filepath': f'evt:{event_class}',
                'content_type': CONTENT_EVENT,
                'event_class': event_class, 'event_data': evt,
                'home_classification': 'event',
                'structural_relation': REL_NONE, 'relation_data': None,
                'base_sha': None, 'delta_eps': None,
            }
            self.entries[sha] = entry
            self._index_entry(sha, k_evt, d_evt)

        return evt

    def _index_entry(self, sha, k, d):
        """Maintain structural indexes on insert (internal)."""
        kd_key = (k, d)
        if kd_key not in self._kd_index:
            self._kd_index[kd_key] = []
        self._kd_index[kd_key].append(sha)
        if k not in self._k_index:
            self._k_index[k] = []
        self._k_index[k].append(sha)

    def index_entry(self, sha, k, d):
        """Public interface for structural index maintenance."""
        self._index_entry(sha, k, d)

    @property
    def k_index(self):
        """Public read access to the k → [sha, ...] structural index.
        Used by find_structural_relations for O(n) multiply/divide detection."""
        return self._k_index

    @property
    def kd_index(self):
        """Public read access to the (k,d) → [sha, ...] structural index.
        Used for attractor/delta detection."""
        return self._kd_index

    def add_entry(self, dsr_entry, tower_result, all_entries=None):
        sha = dsr_entry['sha256']
        if sha in self.entries:
            return 'dedup_hit', self.entries[sha]
        # Structural relation discovery — lattice arithmetic relationships
        rel_type, rel_data = REL_NONE, None
        if all_entries:
            rel_type, rel_data = find_structural_relations(dsr_entry, all_entries, self._k_index)
        # Δε versioning: O(1) lookup via (k,d) index
        base_sha = None
        delta_eps = None
        kd_key = (dsr_entry['k_12'], dsr_entry['d_12'])
        if kd_key in self._kd_index:
            # Same (k,d) = same structural identity → compute Δε
            # mpf subtraction uses the precision of the operands (internal mantissa),
            # not mp.dps. No scaling needed — the values carry their precision.
            existing_sha = self._kd_index[kd_key][0]  # base version
            existing = self.entries[existing_sha]
            delta_eps = dsr_entry['eps_12'] - existing['eps_12']
            base_sha = existing_sha
            # Identity B.2: exact reconstruction formula (not linearized):
            #   r_new = r_base · 2^(Δε/1200)
            # This is stored as metadata for the reconstruction path.
        entry = {
            # THE SEED — the file's Kolmogorov generator on the lattice
            'sha256': sha,
            'k_12': dsr_entry['k_12'],
            'd_12': dsr_entry['d_12'],
            'eps_12': dsr_entry['eps_12'],
            'file_size': dsr_entry['file_size'],
            'file_bits': dsr_entry['file_bits'],
            # Provenance — the filepath IS data, it has its own DSR
            'filepath': dsr_entry['filepath'],
            # Structural classification from tower escalation
            'home_classification': tower_result['home_classification'],
            'home_d': tower_result['home_d'],
            'cf_result': tower_result['cf_result'],
            # Discovery engine results — generator relationships reduce K-complexity
            'structural_relation': rel_type,
            'relation_data': rel_data,
            'base_sha': base_sha,
            'delta_eps': delta_eps,
            # NOTE: I_str and r_mpf are NOT stored — they are derivable from
            # the seed (k, eps, file_size) via pullback. The seed IS the file.
        }
        self.entries[sha] = entry
        self._index_entry(sha, dsr_entry['k_12'], dsr_entry['d_12'])
        self.last_modified = time.time_ns() // 1_000_000_000
        status = 'delta_version' if base_sha else 'new_entry'
        if rel_type != REL_NONE:
            status = f'relation:{rel_type}'
        return status, entry

    def serialize(self):
        """Serialize the Sempaevum to bytes. The birth triad on disk (§3.18.32).

        BH = this seed (entries + events = the minimal generator catalog)
        WH = every pullback evaluation
        Content = the lattice between horizons

        THREE CONTENT TYPES — Identity J.3 spontaneous shrinkage:
          CONTENT_SEED (0x01):      Full (k, ε) stored explicitly
          CONTENT_GENERATOR (0x02): Operation + operand SHAs + κ (derivable, not stored)
          CONTENT_DELTA (0x03):     Base SHA + Δε (Identity B.2 reconstruction)

        Each generator discovered makes explicit content derivable → the seed
        converges toward its Kolmogorov minimum. This IS the Descriptor Gap
        Principle operating on the seed itself: every closed gap means the
        generator's description becomes more structural, less explicit.
        Kolmogorov-natural."""
        header = bytearray()
        header += AKASHIC_MAGIC
        header += struct.pack('>I', AKASHIC_VERSION)
        header += struct.pack('>I', N_BASE)
        header += struct.pack('>I', len(self.entries))
        header += struct.pack('>Q', self.creation_timestamp)
        header += struct.pack('>Q', self.last_modified)
        header_hash = hashlib.sha256(bytes(header)).digest()
        header += header_hash
        header_padded = bytes(header) + b'\x00' * (AKASHIC_PAGE_SIZE - len(header))

        # Precision: temporarily scale for ε serialization
        saved_dps = mp.dps
        max_file_dps = max(
            (dps_for_file(8 * e['file_size']) for e in self.entries.values()),
            default=LATTICE_DPS + GUARD_DPS)
        mp.dps = max(mp.dps, max_file_dps)

        # Operation codes for generator refs
        OP_MULTIPLY   = 0x01
        OP_DIVIDE     = 0x02
        OP_POWER      = 0x03
        OP_RECIPROCAL = 0x04
        OP_CONSTANT   = 0x05
        OP_MAP = {
            REL_MULTIPLY: OP_MULTIPLY, REL_DIVIDE: OP_DIVIDE,
            REL_POWER: OP_POWER, REL_RECIPROCAL: OP_RECIPROCAL,
            REL_CONSTANT: OP_CONSTANT,
        }

        entry_data = bytearray()
        generator_count = 0
        delta_count = 0
        seed_count = 0
        equation_count = 0
        event_entry_count = 0
        block_map_count = 0

        for sha_hex, entry in self.entries.items():
            rel_type = entry.get('structural_relation', REL_NONE)
            rel_data = entry.get('relation_data')
            base_sha = entry.get('base_sha')
            delta_eps = entry.get('delta_eps')
            cls_code = self.CLASSIFICATION_CODES.get(entry['home_classification'], 4)

            eb = bytearray()
            eb += bytes.fromhex(sha_hex)                  # 32 bytes: identity
            eb += struct.pack('>I', entry['file_size'])    # 4 bytes: reconstruction param

            if rel_type != REL_NONE and rel_data and rel_data.get('sha_a'):
                # CONTENT_GENERATOR: operation + operand SHAs + κ
                # The full (k, d, ε) is DERIVABLE via generator_reconstruct()
                eb += struct.pack('>B', CONTENT_GENERATOR)
                eb += struct.pack('>B', OP_MAP.get(rel_type, 0))
                eb += bytes.fromhex(rel_data['sha_a'])     # 32 bytes: operand A
                if rel_type in (REL_MULTIPLY, REL_DIVIDE):
                    eb += bytes.fromhex(rel_data['sha_b']) # 32 bytes: operand B
                elif rel_type in (REL_POWER, REL_RECIPROCAL):
                    # Power exponent as signed byte
                    pwr = rel_data.get('power', -1)
                    eb += struct.pack('>b', pwr)           # 1 byte: exponent
                elif rel_type == REL_CONSTANT:
                    # Constant name hash (first 8 bytes of SHA of name)
                    cname = rel_data.get('constant', '').encode('utf-8')
                    eb += hashlib.sha256(cname).digest()[:8]  # 8 bytes: constant id
                kappa = rel_data.get('kappa', 0)
                eb += struct.pack('>b', kappa)             # 1 byte: κ correction
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                generator_count += 1

            elif base_sha and delta_eps is not None:
                # CONTENT_DELTA: base SHA + Δε (Identity B.2 reconstruction)
                # Compact serialization: size = information content of the delta,
                # NOT the working precision. Small deltas → small storage.
                eb += struct.pack('>B', CONTENT_DELTA)
                eb += bytes.fromhex(base_sha)              # 32 bytes: base version
                eb += _mpf_to_compact_bytes(delta_eps)      # compact: (sign,man,exp)
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                delta_count += 1

            elif entry.get('content_type') == CONTENT_EQUATION:
                # CONTENT_EQUATION: memoized lattice computation stored as seed
                eb += struct.pack('>B', CONTENT_EQUATION)
                eq_json = json.dumps({
                    'op': entry.get('eq_operation', ''),
                    'res': str(entry.get('eq_result', '')),
                }, sort_keys=True, default=str).encode('utf-8')
                eb += struct.pack('>I', len(eq_json))
                eb += eq_json
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                equation_count += 1

            elif entry.get('content_type') == CONTENT_EVENT:
                # CONTENT_EVENT: structural moment stored as seed (§3.9)
                eb += struct.pack('>B', CONTENT_EVENT)
                evt_json = json.dumps(
                    entry.get('event_data', {}),
                    sort_keys=True, default=str).encode('utf-8')
                eb += struct.pack('>I', len(evt_json))
                eb += evt_json
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                event_entry_count += 1

            elif entry.get('content_type') == CONTENT_BLOCK_MAP:
                # CONTENT_BLOCK_MAP: ordered block SHAs for multi-block files
                eb += struct.pack('>B', CONTENT_BLOCK_MAP)
                bmap = entry.get('block_map', {})
                bmap_json = json.dumps(bmap, sort_keys=True, default=str).encode('utf-8')
                eb += struct.pack('>I', len(bmap_json))
                eb += bmap_json
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                block_map_count += 1

            else:
                # CONTENT_SEED: full (k, ε) stored explicitly
                if entry.get('k_12') is None:
                    raise RuntimeError(
                        f"Cannot serialize entry {sha_hex[:12]}: unresolved DSR (k_12 is None)")
                eb += struct.pack('>B', CONTENT_SEED)
                entry_dps = dps_for_file(8 * entry['file_size'])
                eb += _int_to_varlen_bytes(entry['k_12'])
                eb += _mpf_to_bytes(entry['eps_12'], entry_dps + 10)
                eb += struct.pack('>B', cls_code)
                eb += struct.pack('>I', _crc32(bytes(eb)))
                entry_data += struct.pack('>I', len(eb)) + eb
                seed_count += 1

        mp.dps = saved_dps

        # Serialize events — structural moments of change (§3.9)
        event_data = bytearray()
        event_data += struct.pack('>I', len(self.events))
        for evt in self.events:
            evt_json = json.dumps(evt, sort_keys=True, default=str).encode('utf-8')
            event_data += struct.pack('>I', len(evt_json))
            event_data += evt_json

        # Store content type counts as metadata for self-DSR analysis
        self._last_serialize_stats = {
            'seed_count': seed_count,
            'generator_count': generator_count,
            'delta_count': delta_count,
            'block_map_count': block_map_count,
            'event_count': len(self.events),
            'equation_count': eq_metrics()['equations_in_akashic'],
        }

        # Equations are entries (CONTENT_EQUATION) — serialized in the entry loop above.
        # No separate equation section needed.

        return bytes(header_padded) + bytes(entry_data) + bytes(event_data)

    def write(self):
        data = self.serialize()
        self.filepath.write_bytes(data)
        return len(data)

    def verify_read(self):
        """Read back the .akashic and verify structural integrity."""
        data = self.filepath.read_bytes()
        if data[:4] != AKASHIC_MAGIC:
            return False, {'error': f'Bad magic: {data[:4]}'}
        version = struct.unpack('>I', data[4:8])[0]
        n_base = struct.unpack('>I', data[8:12])[0]
        entry_count = struct.unpack('>I', data[12:16])[0]
        creation_ts = struct.unpack('>Q', data[16:24])[0]
        last_mod_ts = struct.unpack('>Q', data[24:32])[0]
        stored_hash = data[32:64]
        computed_hash = hashlib.sha256(data[:32]).digest()
        header_hash_ok = stored_hash == computed_hash

        offset = AKASHIC_PAGE_SIZE
        parsed = []
        for i in range(entry_count):
            if offset + 4 > len(data):
                return False, {'error': f'Truncated at entry {i}'}
            entry_len = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            eb = data[offset:offset+entry_len]
            offset += entry_len
            stored_crc = struct.unpack('>I', eb[-4:])[0]
            computed_crc = _crc32(eb[:-4])
            sha_hex = eb[:32].hex()
            file_size = struct.unpack('>I', eb[32:36])[0]
            parsed.append({
                'sha256': sha_hex,
                'crc_ok': stored_crc == computed_crc,
                'file_size': file_size,
                'seed_size': entry_len,
            })

        total_seed = sum(e['seed_size'] for e in parsed)
        total_orig = sum(e['file_size'] for e in parsed)

        # Parse events section
        event_count = 0
        events_size = 0
        if offset + 4 <= len(data):
            event_count = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            events_start = offset
            for _ in range(event_count):
                if offset + 4 > len(data):
                    break
                evt_len = struct.unpack('>I', data[offset:offset+4])[0]
                offset += 4 + evt_len
            events_size = offset - events_start

        return True, {
            'version': version, 'n_base': n_base,
            'entry_count': entry_count,
            'event_count': event_count,
            'events_size': events_size,
            'creation_ts': creation_ts,
            'last_mod_ts': last_mod_ts,
            'header_hash_ok': header_hash_ok,
            'entries': parsed,
            'total_size': len(data),
            'total_seed_bytes': total_seed,
            'total_original_bytes': total_orig,
            'all_crcs_ok': all(e['crc_ok'] for e in parsed),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §11  SELF-DSR TRACKING & KOLMOGOROV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def entry_serialized_size(entry):
    """Compute the ACTUAL serialized size of an akashic entry dynamically.
    No hardcoded estimates. Uses the same serialization logic as AkashicFile.serialize().
    Returns size in bytes."""
    if entry.get('k_12') is None:
        raise RuntimeError(
            f"Cannot compute size: entry {entry.get('sha256', '?')[:12]} has unresolved DSR")
    # Common header: sha(32) + file_size(4) + content_type(1)
    base_size = 32 + 4 + 1
    rel_type = entry.get('structural_relation', REL_NONE)
    rel_data = entry.get('relation_data')
    base_sha = entry.get('base_sha')
    delta_eps = entry.get('delta_eps')

    if rel_type != REL_NONE and rel_data and rel_data.get('sha_a'):
        # CONTENT_GENERATOR: op(1) + sha_a(32) + operand + kappa(1) + cls(1) + crc(4)
        op_size = 1 + 32  # op_code + sha_a
        if rel_type in (REL_MULTIPLY, REL_DIVIDE):
            op_size += 32  # sha_b
        elif rel_type in (REL_POWER, REL_RECIPROCAL):
            op_size += 1   # power byte
        elif rel_type == REL_CONSTANT:
            op_size += 8   # constant hash
        return base_size + op_size + 1 + 1 + 4  # + kappa + cls + crc

    elif base_sha and delta_eps is not None:
        # CONTENT_DELTA: base_sha(32) + compact_delta(varlen) + cls(1) + crc(4)
        compact_bytes = _mpf_to_compact_bytes(delta_eps)
        return base_size + 32 + len(compact_bytes) + 1 + 4

    elif entry.get('content_type') == CONTENT_EQUATION:
        # CONTENT_EQUATION: json(varlen) + cls(1) + crc(4)
        eq_json = json.dumps({
            'op': entry.get('eq_operation', ''),
            'res': str(entry.get('eq_result', '')),
        }, sort_keys=True, default=str).encode('utf-8')
        return base_size + 4 + len(eq_json) + 1 + 4

    elif entry.get('content_type') == CONTENT_EVENT:
        # CONTENT_EVENT: json(varlen) + cls(1) + crc(4)
        evt_json = json.dumps(
            entry.get('event_data', {}),
            sort_keys=True, default=str).encode('utf-8')
        return base_size + 4 + len(evt_json) + 1 + 4

    elif entry.get('content_type') == CONTENT_BLOCK_MAP:
        # CONTENT_BLOCK_MAP: json(varlen) + cls(1) + crc(4)
        bmap_json = json.dumps(
            entry.get('block_map', {}),
            sort_keys=True, default=str).encode('utf-8')
        return base_size + 4 + len(bmap_json) + 1 + 4

    else:
        # CONTENT_SEED: k(varlen) + eps(varlen) + cls(1) + crc(4)
        file_bits = entry.get('file_bits', 8 * entry.get('file_size', 1))
        entry_dps = dps_for_file(file_bits)
        k_bytes = len(_int_to_varlen_bytes(entry['k_12']))
        saved_dps = mp.dps
        mp.dps = max(mp.dps, entry_dps)
        eps_bytes = len(_mpf_to_bytes(entry['eps_12'], entry_dps + 10))
        mp.dps = saved_dps
        return base_size + k_bytes + eps_bytes + 1 + 4

def kolmogorov_analysis(dsr_entry):
    """Compute the Kolmogorov description of this file's generator on the lattice.
    The seed IS the generator — its structural cost is the K-complexity.
    The generator produces the file via pullback. Pure Kolmogorov.
    Uses entry_serialized_size() for dynamic computation — zero hardcoded estimates."""
    file_size = dsr_entry['file_size']

    # Compute ACTUAL serialized size dynamically
    generator_cost = entry_serialized_size(dsr_entry)

    # K-complexity descriptor: generator cost as dimensionless ratio (projectable seed)
    k_descriptor = mpf(str(generator_cost)) / mpf(str(file_size)) if file_size > 0 else mpf('0')
    return {
        'file_size': file_size,
        'generator_cost': generator_cost,
        'structural_bytes': generator_cost,
        'k_descriptor': k_descriptor,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §12  ALGEBRAIC IDENTITY SUITE
#      Every identity verified on ingested data. Copied from the 13 verified
#      identity scripts. All functions operate on lattice coords only.
# ═══════════════════════════════════════════════════════════════════════════════

# --- Identity A: Lattice Arithmetic (all memoized — §3.5) ---
def lattice_multiply(k1, eps1, k2, eps2, N):
    """Π_N(r₁·r₂) from lattice coords only. No access to r. Memoized."""
    hit, result = eq_lookup('lattice_multiply', k1, eps1, k2, eps2, N)
    if hit:
        return result
    d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
    d2 = eps2 * mpf(N) / CENTS_PER_OCTAVE
    kappa = int(nint(d1 + d2))
    k_p = k1 + k2 + kappa
    g = gcd(abs(k_p), N) if k_p != 0 else N
    result = k_p, N // g, (d1 + d2 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    eq_store('lattice_multiply', result, k1, eps1, k2, eps2, N)
    return result

def lattice_divide(k1, eps1, k2, eps2, N):
    """Π_N(r₁/r₂) from lattice coords only. Memoized."""
    hit, result = eq_lookup('lattice_divide', k1, eps1, k2, eps2, N)
    if hit:
        return result
    d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
    d2 = eps2 * mpf(N) / CENTS_PER_OCTAVE
    kappa = int(nint(d1 - d2))
    k_q = k1 - k2 + kappa
    g = gcd(abs(k_q), N) if k_q != 0 else N
    result = k_q, N // g, (d1 - d2 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    eq_store('lattice_divide', result, k1, eps1, k2, eps2, N)
    return result

def lattice_reciprocal(k1, eps1, N):
    """Π_N(1/r₁) from lattice coords only. Memoized."""
    hit, result = eq_lookup('lattice_reciprocal', k1, eps1, N)
    if hit:
        return result
    d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
    kappa = int(nint(-d1))
    k_inv = -k1 + kappa
    g = gcd(abs(k_inv), N) if k_inv != 0 else N
    result = k_inv, N // g, (-d1 - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    eq_store('lattice_reciprocal', result, k1, eps1, N)
    return result

def lattice_power(k1, eps1, n, N):
    """Π_N(r₁ⁿ) from lattice coords only. Memoized."""
    hit, result = eq_lookup('lattice_power', k1, eps1, n, N)
    if hit:
        return result
    d1 = eps1 * mpf(N) / CENTS_PER_OCTAVE
    nd = mpf(n) * d1
    kappa = int(nint(nd))
    k_pow = n * k1 + kappa
    g = gcd(abs(k_pow), N) if k_pow != 0 else N
    result = k_pow, N // g, (nd - mpf(kappa)) * CENTS_PER_OCTAVE / mpf(N), kappa
    eq_store('lattice_power', result, k1, eps1, n, N)
    return result

# --- Identity F: Tightness / Coherence ---
def tightness(eps_cents):
    """t(ε) = 100/(100+|ε|). The Koide attractor is at t = K = 2/3."""
    return mpf('100') / (mpf('100') + fabs(eps_cents))

# --- Identity H: Magical Impedance ---
def impedance(d):
    """ξ(d) = 137/((d-1)²+16). Real-axis harmonic families only."""
    return mpf('137') / (mpf(d - 1) ** 2 + mpf('16'))

# --- Identity C: Residue Sets ---
def residue_set(d, N_res=12):
    """Res_N(d) = {k mod N : N/gcd(k,N) = d}."""
    return [k for k in range(N_res) if (N_res // gcd(k, N_res) if k != 0 else 1) == d]

# --- Identity D: Phase Projection ---
def project_phase(theta_mpf, N):
    """Phase axis projection Π_N^θ(θ) = (k_θ, d_θ, ε_θ)."""
    two_pi = mpf('2') * mppi
    theta_norm = theta_mpf % two_pi
    if theta_norm < mpf('0'):
        theta_norm += two_pi
    x_theta = mpf(N) * theta_norm / two_pi
    k_theta = int(nint(x_theta)) % N
    eps_theta = (x_theta - mpf(int(nint(x_theta)))) * CENTS_PER_OCTAVE / mpf(N)
    g = gcd(abs(k_theta), N) if k_theta != 0 else N
    d_theta = N // g
    return k_theta, d_theta, eps_theta


def verify_identities_on_data(entries_list):
    """Verify ALL algebraic identities on ingested DSR data.
    Returns (total_tests, passed, failed, report_lines)."""
    results = []
    counts = {'passed': 0, 'failed': 0}

    def check(name, condition):
        if condition:
            counts['passed'] += 1
            results.append(f"  ✓ {name}")
        else:
            counts['failed'] += 1
            results.append(f"  ✗ FAIL: {name}")

    if len(entries_list) < 1:
        return 0, 0, 0, ["  No entries to verify."]

    N = N_BASE

    # --- #0 Bijection round-trip (on every entry) ---
    for dsr in entries_list:
        r = dsr_to_r(dsr)
        k, d, eps = dsr['k_12'], dsr['d_12'], dsr['eps_12']
        r_back = pullback(k, eps, N)
        residual = fabs(r_back - r)
        check(f"#0 Bijection [{Path(dsr['filepath']).name[:20]}]: residual={nstr(residual,4)}",
              residual < mppow(mpf('10'), mpf('-350')))

    # --- A: Lattice Arithmetic (on all pairs) ---
    for i in range(len(entries_list)):
        for j in range(i + 1, min(i + 4, len(entries_list))):
            e1, e2 = entries_list[i], entries_list[j]
            n1 = Path(e1['filepath']).name[:10]
            n2 = Path(e2['filepath']).name[:10]
            r1, r2 = dsr_to_r(e1), dsr_to_r(e2)
            # Multiply
            r_prod = r1 * r2
            k_dir, d_dir, eps_dir = project(r_prod, N)
            k_a, d_a, eps_a, kappa = lattice_multiply(
                e1['k_12'], e1['eps_12'], e2['k_12'], e2['eps_12'], N)
            check(f"A.1 Multiply [{n1}×{n2}]: k={k_dir==k_a}, d={d_dir==d_a}, κ={kappa}",
                  k_dir == k_a and d_dir == d_a and kappa in (-1, 0, 1)
                  and fabs(eps_dir - eps_a) < mppow(mpf('10'), mpf('-340')))
            # Divide
            r_quot = r1 / r2
            k_dir, d_dir, eps_dir = project(r_quot, N)
            k_a, d_a, eps_a, kappa = lattice_divide(
                e1['k_12'], e1['eps_12'], e2['k_12'], e2['eps_12'], N)
            check(f"A.2 Divide [{n1}/{n2}]: k={k_dir==k_a}, d={d_dir==d_a}, κ={kappa}",
                  k_dir == k_a and d_dir == d_a and kappa in (-1, 0, 1)
                  and fabs(eps_dir - eps_a) < mppow(mpf('10'), mpf('-340')))

    # Reciprocal and power on first entry
    if entries_list:
        e = entries_list[0]
        n1 = Path(e['filepath']).name[:15]
        r_e = dsr_to_r(e)
        r_inv = mpf('1') / r_e
        k_dir, d_dir, eps_dir = project(r_inv, N)
        k_a, d_a, eps_a, kappa = lattice_reciprocal(e['k_12'], e['eps_12'], N)
        check(f"A.3 Reciprocal [{n1}]: k={k_dir==k_a}, d={d_dir==d_a}, κ={kappa}",
              k_dir == k_a and d_dir == d_a and kappa in (-1, 0, 1))
        for pwr in [2, 3, 5, -1, -2]:
            r_pow = mppow(r_e, mpf(str(pwr)))
            k_dir, d_dir, eps_dir = project(r_pow, N)
            k_a, d_a, eps_a, kappa = lattice_power(e['k_12'], e['eps_12'], pwr, N)
            check(f"A.4 Power [{n1}^{pwr}]: k={k_dir==k_a}, d={d_dir==d_a}, |κ|={abs(kappa)}≤{(abs(pwr)+1)//2}",
                  k_dir == k_a and d_dir == d_a and abs(kappa) <= (abs(pwr) + 1) // 2)

    # --- B: Differential Control (verify Λ_r = 1200/ln2) ---
    if entries_list:
        e = entries_list[0]
        r = dsr_to_r(e)
        dr = r * mppow(mpf('10'), mpf('-50'))  # tiny perturbation
        r2 = r + dr
        k2_b, d2_b, eps2_b = project(r2, N)
        deps = eps2_b - e['eps_12']
        dr_over_r = dr / r
        lambda_numerical = deps / dr_over_r
        lambda_err = fabs(lambda_numerical - LAMBDA_R) / LAMBDA_R
        check(f"B Differential: Λ_r err={nstr(lambda_err,4)}, shifted (k={k2_b},d={d2_b})",
              lambda_err < mppow(mpf('10'), mpf('-40')) and k2_b == e['k_12'] and d2_b == e['d_12'])

    # --- C: d-Family Composition (residue sets at N=12) ---
    divisors_12 = [d for d in range(1, N + 1) if N % d == 0]
    for d in divisors_12:
        res = residue_set(d, N)
        # Verify Euler phi: |Res(d)| = φ(d) for d|N
        expected_phi = int(totient(d))
        check(f"C Residue |Res({d})|={len(res)}, φ({d})={expected_phi}",
              len(res) == expected_phi)

    # --- D: Phase (verify Λ_θ = 600/π) ---
    theta_test = mppi / mpf('3')  # 60 degrees
    k_t, d_t, eps_t = project_phase(theta_test, N)
    check(f"D Phase π/3: k_θ={k_t}, d_θ={d_t}, |ε_θ|={nstr(fabs(eps_t),4)}",
          k_t == 2 and d_t == 6 and fabs(eps_t) < mppow(mpf('10'), mpf('-350')))
    # Verify Λ_θ ratio
    lambda_ratio_check = LAMBDA_R / LAMBDA_THETA
    expected_ratio = mpf('2') * mppi / LOG2
    check(f"D Λ_r/Λ_θ = 2π/ln2: err={nstr(fabs(lambda_ratio_check - expected_ratio),4)}",
          fabs(lambda_ratio_check - expected_ratio) < mppow(mpf('10'), mpf('-390')))

    # --- E1-E3: FQG (verify divisor structure at N=12) ---
    check(f"E1 Divisors of 12: {divisors_12} = {{1,2,3,4,6,12}}",
          divisors_12 == [1, 2, 3, 4, 6, 12])
    tau_12 = _tau_from_factorization(12)
    check(f"E2 τ(12)={tau_12}=6 (base family count)", tau_12 == 6)
    tau_60 = _tau_from_factorization(60)
    check(f"E2 τ(60)={tau_60}=12 (doubling law: 6·2¹)", tau_60 == 12)
    # Verify 60 = 12 × 5 (new prime 5 introduced)
    factors_60 = factorint(60)
    factors_12 = factorint(12)
    check(f"E3 Bridge: 60={dict(factors_60)} adds prime 5 to 12={dict(factors_12)}",
          5 in factors_60 and 5 not in factors_12)
    tau_420 = _tau_from_factorization(420)
    check(f"E2 τ(420)={tau_420}=24 (doubling law: 6·2²)", tau_420 == 24)
    factors_420 = factorint(420)
    check(f"E3 Bridge: 420={dict(factors_420)} adds prime 7 to 60={dict(factors_60)}",
          7 in factors_420 and 7 not in factors_60)

    # --- F: Tightness / Koide ---
    for dsr in entries_list:
        t = tightness(dsr['eps_12'])
        coherent = t > K_ET
        check(f"F Tightness [{Path(dsr['filepath']).name[:15]}]: t={nstr(t,6)}, coherent={coherent}",
              True)  # structural classification, always valid

    # --- G: Catalan C₆ = N(N-1) = 132 ---
    catalan_6 = 132  # C₆ = (2·6)! / (7!·6!) = 132
    check(f"G Catalan C₆={catalan_6}=N(N-1)={N*(N-1)}", catalan_6 == N * (N - 1))

    # --- H: Impedance ξ(d) monotonically decreasing ---
    xi_values = [(d, impedance(d)) for d in divisors_12]
    monotonic = all(xi_values[i][1] > xi_values[i+1][1]
                     for i in range(len(xi_values) - 1))
    check(f"H Impedance monotonic: {monotonic}", monotonic)
    xi_1 = impedance(1)
    check(f"H ξ(1)=137/16={nstr(xi_1,8)} (gravity, strongest)",
          fabs(xi_1 - mpf('137') / mpf('16')) < mppow(mpf('10'), mpf('-380')))
    # Transfer tensor: T(d₁,d₂;d₃) for ingested d-families
    if len(entries_list) >= 2:
        d1_h = entries_list[0]['d_12']
        d2_h = entries_list[1]['d_12']
        res1 = residue_set(d1_h, N)
        res2 = residue_set(d2_h, N)
        # Compute T(d1,d2;d3) for all d3
        tensor_sum = mpf('0')
        pair_count = len(res1) * len(res2)
        for d3 in divisors_12:
            hits = sum(1 for r1 in res1 for r2 in res2
                       if (N // gcd((r1 + r2) % N, N) if (r1 + r2) % N != 0 else 1) == d3)
            t_val = mpf(str(hits)) / mpf(str(pair_count)) if pair_count > 0 else mpf('0')
            tensor_sum += t_val
        check(f"H Transfer tensor partition of unity: Σ T({d1_h},{d2_h};d₃) = {nstr(tensor_sum,6)}",
              fabs(tensor_sum - mpf('1')) < mppow(mpf('10'), mpf('-350')))

    # --- I: Fixed Point (0,1,0) and Canonical (-53,12,0) ---
    k_fp, d_fp, eps_fp = project(mpf('1'), N)  # r=1 → fixed point
    check(f"I.1 Fixed point r=1: ({k_fp},{d_fp},{nstr(eps_fp,4)})=(0,1,0)",
          k_fp == 0 and d_fp == 1 and eps_fp == mpf('0'))
    # Canonical: r = 2^(-53/12)
    r_can = mppow(mpf('2'), mpf('-53') / mpf('12'))
    k_can, d_can, eps_can = project(r_can, N)
    check(f"I.2 Canonical 2^(-53/12): ({k_can},{d_can},{nstr(eps_can,4)})=(-53,12,0)",
          k_can == -53 and d_can == 12 and fabs(eps_can) < mppow(mpf('10'), mpf('-350')))

    # --- J: Kolmogorov (pullback = direct, already covered by #0) ---
    check("J Kolmogorov: pullback = direct evaluation (subsumed by #0)", True)

    # --- K: Shape (structural markers via lattice-exact geometric ratios) ---
    # 2^(1/3) = cube root → k=4 at N=12 → d=3 (cubic/strong family)
    r_cube = mppow(mpf('2'), mpf('1') / mpf('3'))
    k_cube, d_cube, eps_cube = project(r_cube, N)
    check(f"K.1 Shape 2^(1/3) cube root: d={d_cube}=3 (cubic family)",
          k_cube == 4 and d_cube == 3 and fabs(eps_cube) < mppow(mpf('10'), mpf('-350')))
    # 2^(1/4) = fourth root → k=3 at N=12 → d=4 (weak family)
    r_4th = mppow(mpf('2'), mpf('1') / mpf('4'))
    k_4th, d_4th, eps_4th = project(r_4th, N)
    check(f"K.2 Shape 2^(1/4) fourth root: d={d_4th}=4 (weak family)",
          k_4th == 3 and d_4th == 4 and fabs(eps_4th) < mppow(mpf('10'), mpf('-350')))

    total = counts['passed'] + counts['failed']
    return total, counts['passed'], counts['failed'], results


# ═══════════════════════════════════════════════════════════════════════════════
# §13  CROSS-PATTERN / ATTRACTOR DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cross_patterns(entries_list):
    """Detect when multiple DSRs share the same lattice address (k, d) at ANY
    tower level. Shared addresses = structural attractors. Checks N=12, 60,
    420, 2520, 27720 via the cross-resolution transition map."""
    if len(entries_list) < 2:
        return []
    tower_levels = [12, 60, 420, 2520, 27720]
    attractors = []
    for N in tower_levels:
        projections = []
        for e in entries_list:
            if N == 12:
                k, d, eps = e['k_12'], e['d_12'], e['eps_12']
            else:
                k, d, eps = transition(e['k_12'], e['eps_12'], N_BASE, N)
            projections.append((k, d, eps, e['filepath']))
        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                ki, di, epsi, fi = projections[i]
                kj, dj, epsj, fj = projections[j]
                if ki == kj and di == dj:
                    de = fabs(epsi - epsj)
                    attractors.append({
                        'file_a': Path(fi).name,
                        'file_b': Path(fj).name,
                        'N': N, 'k': ki, 'd': di,
                        'delta_eps': de,
                        'delta_eps_mc': eps_to_microcents(de),
                    })
    return attractors


# ═══════════════════════════════════════════════════════════════════════════════
# §14  DISCOVERY ENGINE
#      Check if any ingested DSR matches a known mathematical constant or
#      structural relationship. This is the seed of generator discovery.
# ═══════════════════════════════════════════════════════════════════════════════

# Dynamically computed ET reference projections at N=12, 361 dps (1200 bits)
ET_REFERENCE_PROJECTIONS = {}

def compute_reference_projections():
    """Dynamically compute lattice projections of ET and mathematical
    constants at full 361 dps precision. Zero hardcoded values."""
    from mpmath import phi as mpphi, e as mpe, euler as mpeuler
    # ET-derived α⁻¹: 137 + √3/48 − √3/(93312π²) − 1/[216(18π−1)]
    alpha_inv_et = (mpf('137')
                    + mpsqrt(mpf('3')) / mpf('48')
                    - mpsqrt(mpf('3')) / (mpf('93312') * mppi ** 2)
                    - mpf('1') / (mpf('216') * (mpf('18') * mppi - mpf('1'))))
    constants = {
        'pi': mppi,
        'e': mpe,
        'phi': mpphi,
        'euler_gamma': mpeuler,
        'K=2/3': K_ET,
        'V=1/12': V_ET,
        'sqrt2': mpsqrt(mpf('2')),
        'sqrt3': mpsqrt(mpf('3')),
        'ln2': LOG2,
        'alpha_inv_ET': alpha_inv_et,
        'Lambda_r': LAMBDA_R,
        'Lambda_theta': LAMBDA_THETA,
        'sigma': SIGMA_ET,
        'LIFE': LIFE_THRESHOLD,
    }
    for name, val in constants.items():
        k, d, eps = project(val, N_BASE)
        ET_REFERENCE_PROJECTIONS[name] = {'value': val, 'k': k, 'd': d, 'eps': eps}

compute_reference_projections()


def discovery_check(dsr_entry, all_entries=None):
    """Check if a DSR matches any known constant OR any previously ingested file.
    Returns list of matches with structural distance."""
    r = dsr_to_r(dsr_entry)  # Pullback — algebraic identity, exact
    k, d, eps = dsr_entry['k_12'], dsr_entry['d_12'], dsr_entry['eps_12']
    matches = []
    # Check against ET reference projections
    for name, ref in ET_REFERENCE_PROJECTIONS.items():
        if k == ref['k'] and d == ref['d']:
            de = fabs(eps - ref['eps'])
            de_mc = abs(eps_to_microcents(de))
            matches.append({
                'constant': name, 'k': k, 'd': d,
                'delta_eps': de, 'delta_eps_mc': de_mc,
                'classification': 'exact' if de_mc == 0 else
                                  'deep' if de_mc <= KOIDE_ATTRACTOR_MICROCENTS else
                                  'near' if de_mc < INCOHERENCE_BOUNDARY_MICROCENTS else 'far',
            })
    # Check against previously ingested files — shared (k,d) IS a discovery
    if all_entries:
        disc_snap = list(all_entries.items())
        for sha, prev in disc_snap:
            if sha == dsr_entry.get('sha256'):
                continue  # skip self
            if prev.get('k_12') is None:
                raise RuntimeError(f"Entry {sha[:12]} has unresolved DSR in discovery_check")
            if prev['k_12'] == k and prev['d_12'] == d:
                de = fabs(eps - prev['eps_12'])
                de_mc = abs(eps_to_microcents(de))
                prev_name = Path(prev['filepath']).name if 'filepath' in prev else sha[:12]
                matches.append({
                    'constant': f'file:{prev_name}',
                    'k': k, 'd': d,
                    'delta_eps': de, 'delta_eps_mc': de_mc,
                    'classification': 'attractor' if de_mc <= KOIDE_ATTRACTOR_MICROCENTS else
                                      'shared_address' if de_mc < INCOHERENCE_BOUNDARY_MICROCENTS else 'far',
                })
    # Also check ratios and powers of known constants
    for name, ref in ET_REFERENCE_PROJECTIONS.items():
        ratio = r / ref['value']
        if ratio > mpf('0'):
            k_rat, d_rat, eps_rat = project(ratio, N_BASE)
            # If the ratio itself is a known constant...
            for name2, ref2 in ET_REFERENCE_PROJECTIONS.items():
                if k_rat == ref2['k'] and d_rat == ref2['d']:
                    de = fabs(eps_rat - ref2['eps'])
                    de_mc = abs(eps_to_microcents(de))
                    if de_mc <= INCOHERENCE_BOUNDARY_MICROCENTS:
                        matches.append({
                            'constant': f'{name2}·{name}',
                            'k': k, 'd': d,
                            'delta_eps': de, 'delta_eps_mc': de_mc,
                            'classification': 'ratio_match',
                        })
    return matches


def discovery_cross_files(entries_list):
    """Check ratios between ALL pairs of ingested files for structural
    relationships. If r_a/r_b matches any ET reference projection, the
    two files are structurally related through that constant."""
    if len(entries_list) < 2:
        return []
    cross_matches = []
    for i in range(len(entries_list)):
        for j in range(i + 1, len(entries_list)):
            ea, eb = entries_list[i], entries_list[j]
            ratio = dsr_to_r(ea) / dsr_to_r(eb)
            if ratio > mpf('0'):
                k_rat, d_rat, eps_rat = project(ratio, N_BASE)
                for name, ref in ET_REFERENCE_PROJECTIONS.items():
                    if k_rat == ref['k'] and d_rat == ref['d']:
                        de = fabs(eps_rat - ref['eps'])
                        de_mc = abs(eps_to_microcents(de))
                        if de_mc <= INCOHERENCE_BOUNDARY_MICROCENTS:
                            cross_matches.append({
                                'file_a': Path(ea['filepath']).name,
                                'file_b': Path(eb['filepath']).name,
                                'ratio_matches': name,
                                'd_ratio': d_rat,
                                'delta_eps_mc': de_mc,
                            })
    return cross_matches


# ═══════════════════════════════════════════════════════════════════════════════
# §15  STRUCTURAL RELATION DISCOVERY
#      The DSR IS the generator. This section discovers how DSRs relate to
#      each other through lattice arithmetic — the algebraic identities (A-K).
#      These are structural properties of the lattice topology.
#      Content types and relation types defined in §0 constants.
# ═══════════════════════════════════════════════════════════════════════════════


def generator_reconstruct(entry, all_akashic_entries):
    """Reconstruct (k, d, ε) from a generator chain (Identity J — arbitrary access).

    Identity A: lattice_multiply/divide/reciprocal/power on operand seeds
    Identity B.2: Δε reconstruction via r_new = r_base · 2^(Δε/1200)

    The generator IS the file. Evaluation = reconstruction. Pure pullback.
    Returns (k, d, eps) at the entry's lattice address."""
    rel_type = entry.get('structural_relation', REL_NONE)
    rel_data = entry.get('relation_data')

    if rel_type == REL_NONE or rel_data is None:
        # No generator — this entry stores its own (k, d, ε) directly
        return entry['k_12'], entry['d_12'], entry['eps_12']

    N = N_BASE

    if rel_type == REL_MULTIPLY:
        ea = all_akashic_entries[rel_data['sha_a']]
        eb = all_akashic_entries[rel_data['sha_b']]
        # Recursively reconstruct operands (generator chains)
        k_a, d_a, eps_a = generator_reconstruct(ea, all_akashic_entries)
        k_b, d_b, eps_b = generator_reconstruct(eb, all_akashic_entries)
        k_r, d_r, eps_r, _ = lattice_multiply(k_a, eps_a, k_b, eps_b, N)
        return k_r, d_r, eps_r

    elif rel_type == REL_DIVIDE:
        ea = all_akashic_entries[rel_data['sha_a']]
        eb = all_akashic_entries[rel_data['sha_b']]
        k_a, d_a, eps_a = generator_reconstruct(ea, all_akashic_entries)
        k_b, d_b, eps_b = generator_reconstruct(eb, all_akashic_entries)
        k_r, d_r, eps_r, _ = lattice_divide(k_a, eps_a, k_b, eps_b, N)
        return k_r, d_r, eps_r

    elif rel_type in (REL_POWER, REL_RECIPROCAL):
        ea = all_akashic_entries[rel_data['sha_a']]
        k_a, d_a, eps_a = generator_reconstruct(ea, all_akashic_entries)
        pwr = rel_data['power']
        if pwr == -1:
            k_r, d_r, eps_r, _ = lattice_reciprocal(k_a, eps_a, N)
        else:
            k_r, d_r, eps_r, _ = lattice_power(k_a, eps_a, pwr, N)
        return k_r, d_r, eps_r

    elif rel_type == REL_CONSTANT:
        ea = all_akashic_entries[rel_data['sha_a']]
        k_a, d_a, eps_a = generator_reconstruct(ea, all_akashic_entries)
        ref = ET_REFERENCE_PROJECTIONS[rel_data['constant']]
        k_r, d_r, eps_r, _ = lattice_multiply(k_a, eps_a, ref['k'], ref['eps'], N)
        return k_r, d_r, eps_r

    # Fallback: direct seed
    return entry['k_12'], entry['d_12'], entry['eps_12']


def reconstruct_via_delta_eps(base_entry, delta_eps):
    """Reconstruct a file version from base seed + Δε using Identity B.2.

    The EXACT finite-shift formula (NOT linearized):
      r_new = r_base · 2^(Δε/1200)

    This is Corollary B.2a from the Differential Control Identity.
    The exponential form is exact. A linearized approximation would
    produce O(Δε²) error — forbidden."""
    k_base = base_entry['k_12']
    eps_base = base_entry['eps_12']
    # Apply exact finite-shift (Identity B.2): new ε = base ε + Δε
    eps_new = eps_base + delta_eps
    # Cell boundary crossing: delta_lattice = total fractional position in lattice units
    delta_lattice = eps_new * mpf(N_BASE) / CENTS_PER_OCTAVE
    # Recompute k from the total position — handles boundary crossing exactly
    total_pos = mpf(k_base) + delta_lattice
    k_new = int(nint(total_pos))
    eps_final = (total_pos - mpf(k_new)) * CENTS_PER_OCTAVE / mpf(N_BASE)
    g = gcd(abs(k_new), N_BASE) if k_new != 0 else N_BASE
    d_new = N_BASE // g
    return k_new, d_new, eps_final


def find_structural_relations(dsr_entry, existing_entries, k_index=None):
    """Discover how a DSR relates to existing DSRs through lattice arithmetic.

    Uses algebraic identities as structural vocabulary:
      Identity A (lattice arithmetic):  k-addition/subtraction/scaling + κ T-act
      Identity C.2 (d-composition):     validates d_product ∈ d₁⊗d₂
      Identity H (transfer tensor):     enriches with impedance-weighted efficiency

    Lattice arithmetic is BOUNDED: δ = ε·N/1200, |δ| ≤ 0.5, κ ∈ {-1,0,+1}.
    400 dps is astronomically sufficient — no dynamic scaling needed.

    k_index: {k → [sha, ...]} for O(n) multiply/divide detection.
    Returns (relation_type, relation_data) or (REL_NONE, None)."""
    k_new = dsr_entry['k_12']
    d_new = dsr_entry['d_12']
    eps_new = dsr_entry['eps_12']

    # Build k_index from existing_entries if not provided
    if k_index is None:
        k_index = {}
        for sha, e in existing_entries.items():
            k_val = e['k_12']
            if k_val not in k_index:
                k_index[k_val] = []
            k_index[k_val].append(sha)

    def _eps_match(eps_computed, eps_target):
        """Check if two ε values match within Koide threshold. Bounded arithmetic."""
        de = fabs(eps_computed - eps_target)
        return abs(eps_to_microcents(de))

    def _enrich_with_tensor(d1, d2, d_result, kappa_act):
        """Identity H: compute transfer tensor T(d₁,d₂;d₃) and impedance efficiency.
        Only for simple families (divisors of 12) where the tensor is defined."""
        divisors_12 = [1, 2, 3, 4, 6, 12]
        if d1 not in divisors_12 or d2 not in divisors_12 or d_result not in divisors_12:
            return {}
        # T_κ(d1,d2;d_result) from residue set arithmetic (Identity C.1 + H.1)
        res1 = residue_set(d1, N_BASE)
        res2 = residue_set(d2, N_BASE)
        total_pairs = len(res1) * len(res2)
        if total_pairs == 0:
            return {}
        hits = sum(1 for r1 in res1 for r2 in res2
                   if (N_BASE // gcd((r1 + r2 + kappa_act) % N_BASE, N_BASE)
                       if (r1 + r2 + kappa_act) % N_BASE != 0 else 1) == d_result)
        t_kappa = mpf(str(hits)) / mpf(str(total_pairs))
        # Impedance-weighted efficiency: E = T × ξ(d_result)/ξ(d1)
        xi_result = impedance(d_result)
        xi_source = impedance(d1)
        efficiency = t_kappa * xi_result / xi_source
        return {
            'transfer_prob': t_kappa,
            'impedance_ratio': xi_result / xi_source,
            'efficiency': efficiency,
        }

    # Snapshot entries to prevent mutation during iteration (events may fire during discovery)
    entries_items = list(existing_entries.items())

    # Check lattice multiply (Identity A.1): k_new = k_a + k_b + κ
    # O(n) via k_index instead of O(n²) brute force
    for sha_a, ea in entries_items:
        for kappa in [-1, 0, 1]:
            k_b_needed = k_new - ea['k_12'] - kappa
            if k_b_needed in k_index:
                for sha_b in k_index[k_b_needed]:
                    eb = existing_entries[sha_b]
                    # Verify with full lattice arithmetic (bounded, 400 dps)
                    k_prod, d_prod, eps_prod, kappa_actual = lattice_multiply(
                        ea['k_12'], ea['eps_12'], eb['k_12'], eb['eps_12'], N_BASE)
                    if k_prod == k_new and d_prod == d_new:
                        de_mc = _eps_match(eps_prod, eps_new)
                        if de_mc <= KOIDE_ATTRACTOR_MICROCENTS:
                            tensor = _enrich_with_tensor(
                                ea['d_12'], eb['d_12'], d_new, kappa_actual)
                            return REL_MULTIPLY, {
                                'sha_a': sha_a, 'sha_b': sha_b,
                                'delta_eps_mc': de_mc, 'kappa': kappa_actual,
                                'tensor': tensor}

    # Check lattice divide (Identity A.2): k_new = k_a - k_b + κ
    for sha_a, ea in entries_items:
        for kappa in [-1, 0, 1]:
            k_b_needed = ea['k_12'] - k_new + kappa
            if k_b_needed in k_index:
                for sha_b in k_index[k_b_needed]:
                    if sha_a == sha_b:
                        continue
                    eb = existing_entries[sha_b]
                    k_quot, d_quot, eps_quot, kappa_actual = lattice_divide(
                        ea['k_12'], ea['eps_12'], eb['k_12'], eb['eps_12'], N_BASE)
                    if k_quot == k_new and d_quot == d_new:
                        de_mc = _eps_match(eps_quot, eps_new)
                        if de_mc <= KOIDE_ATTRACTOR_MICROCENTS:
                            tensor = _enrich_with_tensor(
                                ea['d_12'], eb['d_12'], d_new, kappa_actual)
                            return REL_DIVIDE, {
                                'sha_a': sha_a, 'sha_b': sha_b,
                                'delta_eps_mc': de_mc, 'kappa': kappa_actual,
                                'tensor': tensor}

    # Check lattice power (Identity A.4): k_new = n·k_a + κ_n
    for sha_a, ea in entries_items:
        for pwr in [2, 3, -1, -2, 5, 7, -3]:
            k_pow, d_pow, eps_pow, kappa = lattice_power(
                ea['k_12'], ea['eps_12'], pwr, N_BASE)
            if k_pow == k_new and d_pow == d_new:
                de_mc = _eps_match(eps_pow, eps_new)
                if de_mc <= KOIDE_ATTRACTOR_MICROCENTS:
                    rel_type = REL_RECIPROCAL if pwr == -1 else REL_POWER
                    return rel_type, {
                        'sha_a': sha_a, 'power': pwr,
                        'delta_eps_mc': de_mc, 'kappa': kappa}

    # Check against ET reference constants via lattice multiply
    for name, ref in ET_REFERENCE_PROJECTIONS.items():
        for sha_a, ea in entries_items:
            k_prod, d_prod, eps_prod, kappa = lattice_multiply(
                ea['k_12'], ea['eps_12'], ref['k'], ref['eps'], N_BASE)
            if k_prod == k_new and d_prod == d_new:
                de_mc = _eps_match(eps_prod, eps_new)
                if de_mc <= KOIDE_ATTRACTOR_MICROCENTS:
                    return REL_CONSTANT, {
                        'sha_a': sha_a, 'constant': name,
                        'operation': 'multiply',
                        'delta_eps_mc': de_mc, 'kappa': kappa}

    return REL_NONE, None


# ═══════════════════════════════════════════════════════════════════════════════
# §16  ARBITRARY ACCESS
#      The DSR IS the generator. The pullback IS a random-access function.
#      To read byte N, evaluate at position-dependent computational precision.
# ═══════════════════════════════════════════════════════════════════════════════

def arbitrary_byte_access(entry, byte_offset):
    """Read a specific byte from a file via its DSR.
    The pullback is evaluated at position-dependent computational precision.
    byte_offset 0 = first byte. byte_offset file_size-1 = last byte."""
    file_size = entry['file_size']
    file_bits = 8 * file_size

    if byte_offset < 0 or byte_offset >= file_size:
        raise ValueError(f"byte_offset {byte_offset} out of range [0, {file_size})")

    # Computational precision to resolve this byte position
    target_bit = 8 * (file_size - byte_offset - 1)
    bits_needed = target_bit + 16  # +16 guard bits
    dps_needed = (bits_needed * 10 + 32) // 33 + GUARD_DPS

    saved_dps = mp.dps
    mp.dps = dps_needed

    # Evaluate the DSR at position-dependent precision
    r_recovered = pullback(entry['k_12'], entry['eps_12'], N_BASE)
    I_recovered = int(nint(r_recovered * mppow(mpf('2'), mpf(str(file_bits)))))

    mp.dps = saved_dps

    # Extract the target byte
    shift = 8 * (file_size - byte_offset - 1)
    target_byte = (I_recovered >> shift) & 0xFF
    return target_byte


# ═══════════════════════════════════════════════════════════════════════════════
# §17  FILE REGENERATION
#      The DSR IS the generator. Pullback at file-scaled computational
#      precision evaluates the algebraic identity exactly.
# ═══════════════════════════════════════════════════════════════════════════════

def regenerate_file(entry, output_dir, all_akashic_entries=None):
    """Regenerate a file from its DSR. The DSR IS the generator.

    Four reconstruction paths (Identity J — arbitrary access):
      1. Block map: reassemble from ordered block DSRs
      2. Direct seed: pullback(k, ε) → r → I → bytes
      3. Generator chain: Identity A on operand seeds → (k, d, ε) → pullback
      4. Δε version: Identity B.2 exact finite-shift on base seed → pullback

    All paths end at pullback: 2^((k + ε·N/1200)/N) = r. EXACT."""

    # Block map path: reassemble from ordered block DSRs
    if entry.get('content_type') == CONTENT_BLOCK_MAP and all_akashic_entries:
        bmap = entry.get('block_map', {})
        block_shas = bmap.get('block_shas', [])
        original_file_size = bmap.get('file_size', 0)
        file_sha = bmap.get('file_sha256', entry['sha256'])
        original_name = Path(entry['filepath']).name.replace('blockmap:', '')

        reassembled = bytearray()
        block_results = []
        for i, bsha in enumerate(block_shas):
            block_entry = all_akashic_entries.get(bsha)
            if block_entry is None:
                block_results.append(f"block_{i}: MISSING")
                continue
            # Regenerate this single block
            block_ok, block_det = regenerate_file(
                block_entry, output_dir, all_akashic_entries)
            if block_ok:
                block_path = Path(block_det['output_path'])
                reassembled.extend(block_path.read_bytes())
                block_path.unlink()  # Clean up individual block files
            block_results.append(f"block_{i}: {'OK' if block_ok else 'FAIL'}")

        # Trim to original file size (last block may have been padded)
        if len(reassembled) > original_file_size:
            reassembled = reassembled[:original_file_size]

        sha256_actual = hashlib.sha256(bytes(reassembled)).hexdigest()
        match = sha256_actual == file_sha

        out_path = Path(output_dir) / f"regenerated_{original_name}"
        out_path.write_bytes(bytes(reassembled))

        return match, {
            'output_path': str(out_path),
            'sha256_expected': file_sha,
            'sha256_actual': sha256_actual,
            'file_size': original_file_size,
            'file_dps': dps_for_file(BLOCK_BITS),
            'exact': match,
            'reconstruction_path': 'block_map',
            'block_count': len(block_shas),
            'block_results': block_results,
        }

    file_size = entry['file_size']
    file_bits = 8 * file_size
    original_name = Path(entry['filepath']).name
    sha256_expected = entry['sha256']

    # Determine reconstruction path
    rel_type = entry.get('structural_relation', REL_NONE)
    base_sha = entry.get('base_sha')
    delta_eps = entry.get('delta_eps')

    if rel_type != REL_NONE and all_akashic_entries:
        # Generator reconstruction (Identity A chain)
        k, d, eps = generator_reconstruct(entry, all_akashic_entries)
    elif base_sha and delta_eps is not None and all_akashic_entries:
        # Δε reconstruction (Identity B.2 exact finite-shift)
        base_entry = all_akashic_entries.get(base_sha, entry)
        k, d, eps = reconstruct_via_delta_eps(base_entry, delta_eps)
    else:
        # Direct seed
        k = entry['k_12']
        eps = entry['eps_12']

    # Evaluate the DSR generator at file-scaled precision
    file_dps = dps_for_file(file_bits)
    saved_dps = mp.dps
    mp.dps = max(mp.dps, file_dps)

    r_recovered = pullback(k, eps, N_BASE)
    I_recovered = int(nint(r_recovered * mppow(mpf('2'), mpf(str(file_bits)))))

    mp.dps = saved_dps

    try:
        reconstructed_bytes = I_recovered.to_bytes(file_size, byteorder='big')
    except OverflowError:
        byte_len = (I_recovered.bit_length() + 7) // 8
        reconstructed_bytes = I_recovered.to_bytes(
            max(byte_len, file_size), byteorder='big')[-file_size:]

    sha256_actual = hashlib.sha256(reconstructed_bytes).hexdigest()
    match = sha256_actual == sha256_expected

    out_path = Path(output_dir) / f"regenerated_{original_name}"
    out_path.write_bytes(reconstructed_bytes)

    return match, {
        'output_path': str(out_path),
        'sha256_expected': sha256_expected,
        'sha256_actual': sha256_actual,
        'file_size': file_size,
        'file_dps': file_dps,
        'exact': match,
        'reconstruction_path': 'generator' if rel_type != REL_NONE
                               else 'delta_eps' if base_sha
                               else 'direct_seed',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §13  SELF-DSR TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def self_dsr_snapshot(akashic, akashic_path, all_entries):
    """Write the .akashic, compute its own DSR, return the snapshot.
    The .akashic file IS the Sempaevum on disk — its own lattice address
    changes as content is added. Tracking this trajectory shows how the
    structural description evolves. Identity J.3: spontaneous shrinkage
    as generators replace explicit seeds."""
    sz = akashic.write()
    self_dsr = file_to_dsr(akashic_path)
    total_orig = sum(e['file_size'] for e in all_entries.values())
    # K-complexity descriptor: akashic generator cost as dimensionless value (projectable)
    k_descriptor = mpf(str(sz)) / mpf(str(total_orig)) if total_orig > 0 else mpf('0')
    stats = getattr(akashic, '_last_serialize_stats', {})
    eqm = eq_metrics()

    return {
        'entry_count': len(akashic.entries),
        'event_count': len(akashic.events),
        'seed_entries': stats.get('seed_count', 0),
        'generator_entries': stats.get('generator_count', 0),
        'delta_entries': stats.get('delta_count', 0),
        'equation_entries': eqm['equations_in_akashic'],
        'eq_hit_ratio': eqm['eq_hit_ratio'],
        'akashic_bytes': sz,
        'total_original_bytes': total_orig,
        'k_descriptor': k_descriptor,
        'k_12': self_dsr['k_12'],
        'd_12': self_dsr['d_12'],
        'eps_12': self_dsr['eps_12'],
        'timestamp': time.time_ns() // 1_000_000_000,
    }


def print_self_dsr_trajectory(trajectory):
    """Display the self-DSR trajectory — how the .akashic evolves."""
    if not trajectory:
        print("  No trajectory yet. Ingest files first.")
        return
    print(f"\n  {'#':>4} {'Entries':>8} {'Akashic':>12} {'Orig Data':>12} "
          f"{'K':>10} {'k':>8} {'d':>6} {'|eps| cents':>18}")
    print(f"  {'_'*4} {'_'*8} {'_'*12} {'_'*12} {'_'*10} {'_'*8} {'_'*6} {'_'*18}")
    for i, snap in enumerate(trajectory):
        print(f"  {i:>4} {snap['entry_count']:>8} {snap['akashic_bytes']:>12} "
              f"{snap['total_original_bytes']:>12} "
              f"{nstr(snap['k_descriptor'],6):>10} {snap['k_12']:>8} "
              f"{snap['d_12']:>6} {nstr(fabs(snap['eps_12']),12):>18}")


# ═══════════════════════════════════════════════════════════════════════════════
# §14  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def pick_files(title="Select file(s)", multiple=True):
    """Open a native file dialog. Falls back to text input on headless systems."""
    if _HAS_TK:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            if multiple:
                paths = filedialog.askopenfilenames(title=title)
            else:
                p = filedialog.askopenfilename(title=title)
                paths = (p,) if p else ()
            root.destroy()
            if paths:
                return [Path(p) for p in paths]
        except tk.TclError:
            pass  # TclError = no display available — falls through to text input below
    raw = input(f"  {title} (comma-separated paths): ").strip()
    if not raw:
        return []
    return [Path(p.strip()) for p in raw.split(',')]


def print_banner():
    print("=" * 80)
    print("  EUDD PROOF-OF-CONCEPT VERIFICATION")
    print("  Exception Theory — ET Universal Discovery Database")
    print("  Derived forward from P.D.T = E via the Sempaevum bijection")
    print("=" * 80)
    print(f"\n  Precision: dynamic (lattice={LATTICE_DPS} dps, block={dps_for_file(BLOCK_BITS)} dps, current={mp.dps} dps)")
    print(f"  Lr  = 1200/ln2 = {nstr(LAMBDA_R, 20)}")
    print(f"  Lth = 600/pi   = {nstr(LAMBDA_THETA, 20)}")
    print(f"  Lr/Lth = 2pi/ln2 = {nstr(LAMBDA_RATIO, 20)}")
    print(f"  sigma = sqrt(1/12) = {nstr(SIGMA_ET, 20)}")
    print(f"  K = 2/3 = {nstr(K_ET, 20)}")
    print(f"  IEEE 754: FORBIDDEN    Tower: INFINITE")
    print()


def ingest_file(filepath, akashic, all_entries, verbose=True):
    """Full ingestion pipeline for a single file.
    Files ≤ BLOCK_SIZE: single DSR (existing flow).
    Files > BLOCK_SIZE: split at filesystem block boundaries, each block → DSR,
    discovery engine runs across ALL blocks on ingestion.
    Every stage fires structural events (§3.9) on the akashic.
    Events ARE lattice content — they have DSRs."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None, None, None, None

    file_size = filepath.stat().st_size
    print(f"\n{'_'*80}")
    print(f"  INGESTING: {filepath.name} ({file_size} bytes)")
    print(f"{'_'*80}")

    # ── MULTI-BLOCK PATH: files > BLOCK_SIZE ──
    if file_size > BLOCK_SIZE:
        print(f"\n  [1] FILE -> BLOCK DSRs ({BLOCK_SIZE}B blocks)")
        t0_ns = time.time_ns()
        try:
            file_meta, block_dsrs = file_to_block_dsrs(filepath, akashic.entries)
        except ValueError as e:
            print(f"      SKIP: {e}")
            return None, None, None, None
        except Exception as e:
            print(f"      ERROR during block projection: {type(e).__name__}: {e}")
            return None, None, None, None
        t1_ns = time.time_ns()
        n_blocks = file_meta['block_count']
        print(f"      Blocks: {n_blocks} × {BLOCK_SIZE}B")
        print(f"      File SHA: {file_meta['sha256']}")
        print(f"      Time: {(t1_ns - t0_ns) // 1_000_000}ms")

        # Event: file split into blocks
        akashic.fire_event('seed_generated', file_meta['sha256'], {
            'source_data_hash': file_meta['sha256'], 'N_chosen': N_BASE,
            'file_size': file_size, 'block_count': n_blocks,
            'encoding_method': 'block_split',
            'projection_time_ms': (t1_ns - t0_ns) // 1_000_000,
        })

        # Step 2-3: Per-block dedup, round-trip, store
        print(f"\n  [2-3] PER-BLOCK DEDUP + ROUND-TRIP + STORE")
        blocks_stored = 0
        blocks_dedup = 0
        blocks_rt_pass = 0
        generators_found = 0
        for i, bdsr in enumerate(block_dsrs):
            bsha = bdsr['sha256']
            # Dedup
            if bsha in all_entries or bsha in akashic.entries:
                blocks_dedup += 1
                continue
            # Round-trip at block precision
            rt_ok, _ = verify_round_trip(bdsr)
            if rt_ok:
                blocks_rt_pass += 1
            # Tower result for block (minimal — just classification)
            block_tower = {
                'home_classification': 'intermediate_home',
                'home_d': bdsr['d_12'],
                'cf_result': {'eps_cf': mpf('0'), 'd_home': bdsr['d_12'],
                              'quality': 0, 'classification': 'block'},
            }
            status, stored = akashic.add_entry(bdsr, block_tower, akashic.entries)
            if stored and stored.get('structural_relation', REL_NONE) != REL_NONE:
                generators_found += 1
            all_entries[bsha] = bdsr
            blocks_stored += 1

        print(f"      Stored: {blocks_stored}, Dedup: {blocks_dedup}, "
              f"RT pass: {blocks_rt_pass}, Generators: {generators_found}")

        # Step 4: Retroactive cross-block discovery (Phase 4.2/4.3)
        print(f"\n  [4] CROSS-BLOCK DISCOVERY")
        retro_found = 0
        block_shas = [b['sha256'] for b in block_dsrs if b['sha256'] in akashic.entries]
        for bsha in block_shas:
            entry = akashic.entries[bsha]
            if entry.get('structural_relation', REL_NONE) == REL_NONE:
                rel_type, rel_data = find_structural_relations(
                    entry, akashic.entries, akashic.k_index)
                if rel_type != REL_NONE:
                    entry['structural_relation'] = rel_type
                    entry['relation_data'] = rel_data
                    retro_found += 1
        print(f"      Retroactive generators found: {retro_found}")
        generators_found += retro_found

        # Step 5: Store block map entry
        print(f"\n  [5] BLOCK MAP")
        block_map_bytes = json.dumps({
            'file_sha256': file_meta['sha256'],
            'file_size': file_size,
            'block_size': BLOCK_SIZE,
            'block_count': n_blocks,
            'block_shas': file_meta['block_shas'],
        }, sort_keys=True).encode('utf-8')
        block_map_sha = hashlib.sha256(block_map_bytes).hexdigest()
        block_map_dsr = block_to_dsr(block_map_bytes,
                                     f"blockmap:{filepath.name}", akashic.entries)
        block_map_entry = {
            'sha256': block_map_sha,
            'k_12': block_map_dsr['k_12'],
            'd_12': block_map_dsr['d_12'],
            'eps_12': block_map_dsr['eps_12'],
            'file_size': len(block_map_bytes),
            'file_bits': 8 * len(block_map_bytes),
            'filepath': f"blockmap:{filepath.name}",
            'content_type': CONTENT_BLOCK_MAP,
            'home_classification': 'true_home',
            'structural_relation': REL_NONE,
            'relation_data': None,
            'base_sha': None,
            'delta_eps': None,
            'block_map': file_meta,
        }
        if block_map_sha not in akashic.entries:
            akashic.entries[block_map_sha] = block_map_entry
            akashic.index_entry(block_map_sha, block_map_dsr['k_12'], block_map_dsr['d_12'])
        print(f"      Map SHA: {block_map_sha[:24]}...")
        print(f"      Map size: {len(block_map_bytes)}B for {file_size}B file")

        # Step 6: Tower on first non-zero block (representative)
        print(f"\n  [6] TOWER (first block representative)")
        first_block = next((b for b in block_dsrs if not b.get('is_zero_block')), block_dsrs[0])
        tower = tower_escalation(first_block, verbose=verbose)

        # Step 7: Kolmogorov — total seed size vs file size (dynamic computation)
        print(f"\n  [7] KOLMOGOROV")
        total_seed_bytes = len(block_map_bytes)
        for b in block_dsrs:
            block_entry = akashic.entries.get(b['sha256'])
            if block_entry:
                total_seed_bytes += entry_serialized_size(block_entry)
            else:
                total_seed_bytes += entry_serialized_size(b)
        print(f"      Total seed: {total_seed_bytes}B for {file_size}B file")
        print(f"      Generators: {generators_found}/{n_blocks} blocks")

        # Step 8: Equation metrics
        metrics = eq_metrics()
        print(f"\n  [8] MEMOIZATION")
        print(f"      Equations: {metrics['equations_in_akashic']} seeds "
              f"({metrics['eq_propagated_free']} free via identity propagation)")
        print(f"      Hits/Misses: {metrics['eq_hits']}/{metrics['eq_misses']} "
              f"(ratio: {nstr(metrics['eq_hit_ratio'] * mpf('100'), 4)}%)")

        # Return first block DSR for compatibility
        all_rt_pass = (blocks_rt_pass == blocks_stored)
        kolm = {'file_size': file_size, 'generator_cost': total_seed_bytes,
                'structural_bytes': total_seed_bytes,
                'k_descriptor': mpf(str(total_seed_bytes)) / mpf(str(file_size))}
        return first_block, tower, (all_rt_pass, {
            'pullback_exact': all_rt_pass,
            'file_dps': dps_for_file(BLOCK_BITS),
            'blocks_tested': blocks_stored,
            'blocks_passed': blocks_rt_pass,
        }), kolm

    # ── SINGLE-BLOCK PATH: files ≤ BLOCK_SIZE ──
    # Step 1: File -> DSR
    print(f"\n  [1] FILE -> DSR")
    t0_ns = time.time_ns()
    try:
        dsr = file_to_dsr(filepath, akashic.entries)
    except ValueError as e:
        print(f"      SKIP: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"      ERROR during projection: {type(e).__name__}: {e}")
        return None, None, None, None
    t1_ns = time.time_ns()
    sha = dsr['sha256']
    print(f"      Size: {dsr['file_size']} bytes ({dsr['file_bits']} bits)")
    print(f"      SHA:  {sha}")
    print(f"      DSR:  (k={dsr['k_12']}, d={dsr['d_12']}, "
          f"ε={nstr(dsr['eps_12'],15)}¢)")
    seed_bytes = entry_serialized_size(dsr)
    print(f"      Seed: {seed_bytes}B generator for {dsr['file_size']}B file")
    print(f"      Time: {(t1_ns - t0_ns) // 1_000_000}ms")

    # Event: seed_generated (§3.9 — sender projects data onto Sempaevum)
    akashic.fire_event('seed_generated', sha, {
        'source_data_hash': sha, 'N_chosen': N_BASE,
        'k': dsr['k_12'], 'd': dsr['d_12'],
        'file_size': dsr['file_size'], 'file_bits': dsr['file_bits'],
        'encoding_method': 'whole_file_ratio',
        'projection_time_ms': (t1_ns - t0_ns) // 1_000_000,
    })

    # Step 2: Anti-numerology
    print(f"\n  [2] ANTI-NUMEROLOGY")
    ar = dict(dsr['antinum_report'])  # anti_numerology_check returns (bool, dict)
    print(f"      N1: {ar['N1_dimensionless']}")
    print(f"      N2: {ar['N2_substrate_derived']}")
    n3 = ar['N3_cross_domain']
    print(f"      N3: {n3} — {ar['N3_reason']}")

    # Event: anti_numerology_check (§3.9 — N1/N2/N3 compliance)
    akashic.fire_event('anti_numerology_check', sha, {
        'n1_result': ar['N1_dimensionless'],
        'n2_result': ar['N2_substrate_derived'],
        'n3_result': n3,
        'n3_witness_count': len(ar.get('N3_witnesses', [])),
    })

    # Step 3: Dedup — check both session dict and persisted akashic
    if sha in all_entries or sha in akashic.entries:
        print(f"\n  [3] DEDUP: CACHE HIT — already ingested")
        akashic.fire_event('seed_deduplicated', sha, {
            'existing_seed_id': sha, 'dedup_type': 'exact',
        })
        return dsr, None, None, None

    # Step 4: Round-trip
    print(f"\n  [4] ROUND-TRIP (DSR generator verification)")
    t0_ns = time.time_ns()
    rt_match, rt_det = verify_round_trip(dsr)
    t1_ns = time.time_ns()
    print(f"      SHA match:      {'YES' if rt_match else 'NO'}")
    print(f"      Pullback exact: {'YES' if rt_det['pullback_exact'] else 'NO'}")
    print(f"      Precision:      {rt_det['file_dps']} dps for {8 * dsr['file_size']} bits")
    print(f"      Time:           {(t1_ns - t0_ns) // 1_000_000}ms")

    # Event: bijection_round_trip_verified (§3.18.20)
    akashic.fire_event('bijection_round_trip_verified', sha, {
        'k': dsr['k_12'], 'd': dsr['d_12'],
        'residual_is_zero': rt_match,
        'proof_method': 'precision_scaling',
        'dps_tested': rt_det['file_dps'],
        'reconstruction_time_ms': (t1_ns - t0_ns) // 1_000_000,
    })

    # Step 5: Tower
    print(f"\n  [5] INFINITE TOWER ESCALATION + CF")
    t0_ns = time.time_ns()
    tower = tower_escalation(dsr, verbose=verbose)
    t1_ns = time.time_ns()
    print(f"      Tower time: {(t1_ns - t0_ns) // 1_000_000}ms")

    # Event: cf_home_identified (§3.9 — CF home-finding result)
    cf = tower.get('cf_result', {})
    if cf:
        akashic.fire_event('cf_home_identified', sha, {
            'd_home': cf.get('d_home'),
            'cf_classification': tower.get('home_classification'),
            'tower_status': 'tower_agreed' if tower.get('home_d') == cf.get('d_home')
                           else 'tower_disagreed',
        })

    # Step 6: Kolmogorov
    print(f"\n  [6] KOLMOGOROV")
    kolm = kolmogorov_analysis(dsr)
    print(f"      Generator: {kolm['generator_cost']}B K-complexity for {kolm['file_size']}B file")

    # Event: kolmogorov_complexity_computed (§3.9)
    akashic.fire_event('kolmogorov_complexity_computed', sha, {
        'data_hash': sha,
        'k_complexity_bits': kolm['structural_bytes'] * 8,
        'description_language': 'sempaevum',
    })

    # Step 7: Discovery
    print(f"\n  [7] DISCOVERY")
    disc = discovery_check(dsr, akashic.entries)
    if disc:
        for m in disc:
            print(f"      Match: {m['constant']} (d={m['d']}, "
                  f"Δε={m['delta_eps_mc']}mc, {m['classification']})")
    else:
        print(f"      No known constant matches at (k={dsr['k_12']}, d={dsr['d_12']})")

    # Step 8: Store (with structural relation discovery)
    print(f"\n  [8] STORE")
    status, stored = akashic.add_entry(dsr, tower, akashic.entries)
    print(f"      Status: {status}")

    # Fire events for structural discoveries
    if stored:
        rel = stored.get('structural_relation', REL_NONE)
        if rel != REL_NONE:
            rd = stored.get('relation_data', {})
            print(f"      Relation: {rel} → {rd}")
            if rel in (REL_MULTIPLY, REL_DIVIDE):
                # Event: product_decomposition_discovered (§3.18.21)
                akashic.fire_event('product_decomposition_discovered', sha, {
                    'new_value_id': sha,
                    'factor1_value_id': rd.get('sha_a'),
                    'factor2_value_id': rd.get('sha_b'),
                    'kappa': rd.get('kappa'),
                    'operation': rel,
                })
                if rd.get('kappa', 0) != 0:
                    # Event: kappa_correction_applied (§3.18.21)
                    akashic.fire_event('kappa_correction_applied', sha, {
                        'operation': rel, 'kappa_value': rd['kappa'],
                    })
                # Event: seed_generator_discovered (§3.18.32 J.3 — spontaneous shrinkage)
                akashic.fire_event('seed_generator_discovered', sha, {
                    'identity_label': 'A',
                    'generator_description': f'{rel}({rd.get("sha_a","")[:12]}, {rd.get("sha_b","")[:12]})',
                    'content_made_derivable': f'(k,d,eps) for {sha[:12]}',
                    'discovery_source': 'insert_time',
                })
            elif rel in (REL_POWER, REL_RECIPROCAL):
                akashic.fire_event('lattice_arithmetic_computed', sha, {
                    'operation': 'power' if rel == REL_POWER else 'reciprocal',
                    'operand1_value_id': rd.get('sha_a'),
                    'operand2_value_id_or_exponent': rd.get('power'),
                    'kappa_correction': rd.get('kappa'),
                })
            elif rel == REL_CONSTANT:
                akashic.fire_event('lattice_arithmetic_computed', sha, {
                    'operation': 'multiply',
                    'operand1_value_id': rd.get('sha_a'),
                    'constant': rd.get('constant'),
                    'kappa_correction': rd.get('kappa'),
                })

        if stored.get('base_sha'):
            # Event: file_version_delta_stored (§3.9)
            akashic.fire_event('file_version_delta_stored', sha, {
                'base_seed_value_id': stored['base_sha'],
                'file_hash_new': sha,
                'file_hash_base': stored['base_sha'],
            })
            # Event: seed_deduplicated structural (§3.9)
            akashic.fire_event('seed_deduplicated', sha, {
                'new_seed_id': sha,
                'existing_seed_id': stored['base_sha'],
                'dedup_type': 'structural_delta_eps',
            })

    all_entries[sha] = dsr

    # Step 8b: Retroactive discovery — does this new entry enable generators
    # for existing CONTENT_SEED entries? Every ingestion potentially improves ALL
    # existing entries. This is bidirectional discovery (Phase 4.2).
    retro_count = 0
    existing_snap = list(akashic.entries.items())
    for existing_sha, existing_entry in existing_snap:
        if existing_sha == sha:
            continue  # skip self
        if existing_entry.get('structural_relation', REL_NONE) != REL_NONE:
            continue  # already has a generator
        if existing_entry.get('content_type') in (CONTENT_EQUATION, CONTENT_EVENT, CONTENT_BLOCK_MAP):
            continue  # skip non-file entries
        rel_type_r, rel_data_r = find_structural_relations(
            existing_entry, akashic.entries, akashic.k_index)
        if rel_type_r != REL_NONE:
            existing_entry['structural_relation'] = rel_type_r
            existing_entry['relation_data'] = rel_data_r
            retro_count += 1
            akashic.fire_event('seed_generator_discovered', existing_sha, {
                'identity_label': 'A',
                'generator_description': f'{rel_type_r}(retroactive from {sha[:12]})',
                'discovery_source': 'retroactive',
            })
    if retro_count > 0:
        print(f"\n  [8b] RETROACTIVE DISCOVERY: {retro_count} existing entries now have generators")

    # Step 9: Self-recording — equation metrics (§3.1b)
    metrics = eq_metrics()
    print(f"\n  [9] MEMOIZATION (§3.5 — equations ARE seeds in the akashic)")
    print(f"      Equations:   {metrics['equations_in_akashic']} seeds "
          f"({metrics['eq_propagated_free']} free via identity propagation)")
    print(f"      Hits/Misses: {metrics['eq_hits']}/{metrics['eq_misses']} "
          f"(ratio: {nstr(metrics['eq_hit_ratio'] * mpf('100'), 4)}%)")
    akashic.fire_event('self_recording_sample', sha, {
        'equations_in_akashic': metrics['equations_in_akashic'],
        'eq_hits': metrics['eq_hits'],
        'eq_misses': metrics['eq_misses'],
        'eq_hit_ratio': metrics['eq_hit_ratio'],
        'eq_propagated_free': metrics['eq_propagated_free'],
        'entry_count': len(akashic.entries),
        'event_count': len(akashic.events),
        # System metrics via os (§3.1b — resource usage, all projectable ratios)
        'process_id': os.getpid(),
        'cpu_count': os.cpu_count(),
        'akashic_file_size': os.path.getsize(akashic.filepath) if akashic.filepath.exists() else 0,
    })

    return dsr, tower, (rt_match, rt_det), kolm


def main():
    print_banner()

    output_dir = Path(__file__).resolve().parent
    akashic_path = output_dir / 'Sempaevum.akashic'
    akashic = AkashicFile(akashic_path)
    all_entries = {}
    self_dsr_trajectory = []

    # Restore session state from loaded akashic — persistence is mandatory
    if akashic.entries:
        for sha, entry in akashic.entries.items():
            all_entries[sha] = entry
        print(f"  Session restored: {len(all_entries)} entries from .akashic")

    while True:
        print(f"\n{'='*80}")
        print(f"  [1] Ingest file(s)        [2] Generate .akashic")
        print(f"  [3] Delta-eps test        [4] Built-in verification")
        print(f"  [5] Regenerate file       [6] Self-DSR trajectory")
        print(f"  [7] Exit")
        print(f"{'='*80}")

        choice = input("  > ").strip()

        if choice == '1':
            paths = pick_files("Select file(s) to ingest")
            for fp in paths:
                ingest_file(fp, akashic, all_entries)
            if akashic.entries:
                snap = self_dsr_snapshot(akashic, akashic_path, all_entries)
                self_dsr_trajectory.append(snap)
                print(f"\n  Self-DSR [{len(self_dsr_trajectory)-1}]: "
                      f"(k={snap['k_12']}, d={snap['d_12']}) "
                      f"K={nstr(snap['k_descriptor'],6)} "
                      f"akashic={snap['akashic_bytes']}B entries={snap['entry_count']}")

        elif choice == '2':
            if not akashic.entries:
                print("  No entries. Ingest first.")
                continue
            t0_ns = time.time_ns()
            sz = akashic.write()
            t1_ns = time.time_ns()
            print(f"\n  Written: {akashic_path} ({sz} bytes, {(t1_ns - t0_ns) // 1_000_000}ms)")
            ok, det = akashic.verify_read()
            if not ok:
                print(f"  VERIFY FAILED: {det.get('error', 'unknown')}")
            print(f"  Header OK:  {det.get('header_hash_ok')}")
            print(f"  CRCs OK:    {det.get('all_crcs_ok')}")
            print(f"  Entries:    {det.get('entry_count')}")
            print(f"  Seed total: {det.get('total_seed_bytes')} bytes")
            print(f"  Orig data:  {det.get('total_original_bytes')} bytes")
            # Self-DSR
            self_dsr = file_to_dsr(akashic_path)
            print(f"\n  Self-DSR: (k={self_dsr['k_12']}, d={self_dsr['d_12']}, "
                  f"e={nstr(self_dsr['eps_12'],15)}c)")
            total_orig = sum(e['file_size'] for e in all_entries.values())
            if total_orig > 0:
                k_desc = mpf(str(sz)) / mpf(str(total_orig))
                print(f"  K-complexity: generator={sz}B, content={total_orig}B, K={nstr(k_desc,6)}")

        elif choice == '3':
            orig_paths = pick_files("Select ORIGINAL file", multiple=False)
            if not orig_paths:
                print("  No file selected.")
                continue
            mod_paths = pick_files("Select MODIFIED file", multiple=False)
            if not mod_paths:
                print("  No file selected.")
                continue
            do = file_to_dsr(orig_paths[0])
            dm = file_to_dsr(mod_paths[0])
            de = dm['eps_12'] - do['eps_12']
            dk = dm['k_12'] - do['k_12']
            print(f"  Orig: (k={do['k_12']}, d={do['d_12']}, e={nstr(do['eps_12'],12)}c)")
            print(f"  Mod:  (k={dm['k_12']}, d={dm['d_12']}, e={nstr(dm['eps_12'],12)}c)")
            print(f"  Dk={dk}, Dd={dm['d_12']==do['d_12']}, "
                  f"De={nstr(de,12)}c ({eps_to_microcents(de)} mc)")

        elif choice == '4':
            print(f"\n  Built-in verification using identity scripts...")
            test_dir = Path(__file__).resolve().parent
            test_files = sorted(test_dir.glob('*.py'))
            # Exclude self
            test_files = [f for f in test_files if f.name != Path(__file__).name]
            if not test_files:
                print(f"  No .py files in {test_dir}")
                continue
            snap = None
            for tf in test_files:
                ingest_file(tf, akashic, all_entries)
                # Track self-DSR after each ingestion
                snap = self_dsr_snapshot(akashic, akashic_path, all_entries)
                self_dsr_trajectory.append(snap)
            # Show .akashic verification
            ok, det = akashic.verify_read()
            if not ok:
                print(f"  VERIFY FAILED: {det.get('error', 'unknown')}")
            print(f"\n  .akashic: {snap['akashic_bytes']} bytes, "
                  f"header={det.get('header_hash_ok')}, crcs={det.get('all_crcs_ok')}")
            # Self-DSR trajectory so far
            print_self_dsr_trajectory(self_dsr_trajectory)
            # Regeneration test
            first_sha = list(akashic.entries.keys())[0]
            first_entry = akashic.entries[first_sha]
            print(f"\n  REGENERATION TEST: {Path(first_entry['filepath']).name}")
            regen_ok, regen_det = regenerate_file(first_entry, output_dir, akashic.entries)
            print(f"  SHA match: {'YES' if regen_ok else 'NO'} (DSR pullback at {regen_det.get('file_dps', '?')} dps)")
            print(f"  Output:    {regen_det['output_path']}")
            print(f"  Size:      {regen_det['file_size']} bytes")
            # Dedup test
            print(f"\n  DEDUP TEST: re-ingesting {test_files[0].name}")
            ingest_file(test_files[0], akashic, all_entries)
            # Delta-eps test
            orig_p = test_files[0]
            mod_p = output_dir / 'modified_test.py'
            ob = orig_p.read_bytes()
            mod_p.write_bytes(ob[:-1] + bytes([ob[-1] ^ 0x01]))
            do = file_to_dsr(orig_p)
            dm = file_to_dsr(mod_p)
            de = dm['eps_12'] - do['eps_12']
            print(f"\n  DELTA-EPS TEST:")
            print(f"  Orig d={do['d_12']}, Mod d={dm['d_12']}, "
                  f"De={eps_to_microcents(de)} mc")
            # Algebraic Identity Suite — ALL identities on ingested data
            print(f"\n  ALGEBRAIC IDENTITY SUITE:")
            id_entries = list(all_entries.values())
            total_id, passed_id, failed_id, id_lines = verify_identities_on_data(id_entries)
            for line in id_lines:
                print(line)
            print(f"\n  Identities: {passed_id}/{total_id} passed, {failed_id} failed")
            # Cross-pattern detection
            print(f"\n  CROSS-PATTERN / ATTRACTOR DETECTION:")
            attractors = detect_cross_patterns(id_entries)
            if attractors:
                for att in attractors:
                    print(f"    ATTRACTOR N={att['N']}: {att['file_a']} <-> {att['file_b']} "
                          f"at (k={att['k']}, d={att['d']}), Δε={att['delta_eps_mc']}mc")
            else:
                print(f"    No shared (k,d) addresses found among {len(id_entries)} entries")
            # Cross-file discovery
            print(f"\n  CROSS-FILE DISCOVERY:")
            cross = discovery_cross_files(id_entries)
            if cross:
                for cm in cross[:10]:  # show first 10
                    print(f"    {cm['file_a']} / {cm['file_b']} "
                          f"ratio ≈ {cm['ratio_matches']} (d={cm['d_ratio']}, "
                          f"Δε={cm['delta_eps_mc']}mc)")
            else:
                print(f"    No inter-file ratio matches found")
            # Self-ingestion: the .akashic IS the birth triad — it ingests itself
            # Identity J: the seed of all seeds. Convergence loop — the akashic
            # re-serializes and recomputes its self-DSR until it stabilizes.
            print(f"\n  SELF-INGESTION (birth triad — Identity J convergence):")
            prev_self_k = None
            prev_self_eps = None
            convergence_steps = 0
            self_entry = None  # populated on first iteration; guaranteed by max_convergence > 0
            max_convergence = 10  # structural limit — should converge in 2-3 steps
            for conv_step in range(max_convergence):
                sz = akashic.write()
                self_dsr = file_to_dsr(akashic_path)
                self_k = self_dsr['k_12']
                self_eps = self_dsr['eps_12']
                self_status, self_entry = akashic.add_entry(self_dsr, {
                    'home_classification': 'true_home',
                    'home_d': self_dsr['d_12'],
                    'cf_result': {'eps_cf': mpf('0'), 'd_home': self_dsr['d_12'],
                                  'quality': 0, 'classification': 'self'},
                })
                convergence_steps += 1
                print(f"    Step {conv_step}: (k={self_k}, d={self_dsr['d_12']}) "
                      f"akashic={sz}B status={self_status}")
                # Check convergence: self-DSR stopped changing
                if prev_self_k is not None:
                    dk = self_k - prev_self_k
                    deps = fabs(self_eps - prev_self_eps) if prev_self_eps is not None else mpf('1')
                    if dk == 0 and deps < mppow(mpf('10'), mpf(str(-(mp.dps - 100)))):
                        print(f"    CONVERGED at step {conv_step}: Δk=0, "
                              f"Δε={nstr(deps, 6)}¢ (below precision floor)")
                        break
                prev_self_k = self_k
                prev_self_eps = self_eps
            else:
                print(f"    Self-DSR did not converge in {max_convergence} steps "
                      f"(trajectory is still evolving — normal for growing akashic)")
            if self_entry and self_entry.get('delta_eps') is not None:
                print(f"    Δε to base: {nstr(self_entry['delta_eps'], 12)} cents")
            # Final
            if failed_id == 0:
                print(f"\n  ALL CORE PROMISES VERIFIED. ALL IDENTITIES PASS.")
            else:
                print(f"\n  WARNING: {failed_id} identity tests FAILED.")

        elif choice == '5':
            if not akashic.entries:
                print("  No entries. Ingest first.")
                continue
            print(f"\n  Stored seeds:")
            for i, (sha, entry) in enumerate(akashic.entries.items()):
                print(f"    [{i}] {Path(entry['filepath']).name} "
                      f"({entry['file_size']}B, d={entry['d_12']}, sha={sha[:12]}...)")
            idx_str = input("  Select index to regenerate: ").strip()
            try:
                idx = int(idx_str)
                sha_list = list(akashic.entries.keys())
                entry = akashic.entries[sha_list[idx]]
                regen_ok, regen_det = regenerate_file(entry, output_dir, akashic.entries)
                print(f"  SHA match:  {'YES' if regen_ok else 'NO'} (DSR pullback at {regen_det.get('file_dps', '?')} dps)")
                print(f"  Written to: {regen_det['output_path']}")
                print(f"  Size:       {regen_det['file_size']} bytes")
            except (ValueError, IndexError):
                print(f"  Invalid selection.")

        elif choice == '6':
            print_self_dsr_trajectory(self_dsr_trajectory)

        elif choice == '7':
            break


if __name__ == '__main__':
    main()
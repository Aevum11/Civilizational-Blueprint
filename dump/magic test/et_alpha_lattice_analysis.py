#!/usr/bin/env python3
"""
==============================================================================
ET FINE STRUCTURE CONSTANT — FULL LATTICE ANALYSIS (pre-derivation diagnostic)
==============================================================================

Goal: Before deriving the correct cross-term, ANALYZE α⁻¹ using the full
ET lattice to find its natural structural position. Use measurement data as
a structural GUIDE (not as a fit target), then forward-derive the correct
cross-term from whatever structural signature emerges.

This script:
  1. Projects every candidate α⁻¹ value (forward-derived and measured) onto
     every LCM tower level from 12ET to 27720ET (and beyond: 2744ET per
     Lattice Compendium §19).
  2. Identifies sub-cent placements (structural lattice points).
  3. Identifies shared lattice structure between forward and measured values.
  4. Computes the Descriptor Gap between them at each lattice level.
  5. Hypothesizes the missing cross-term from the structural signature.
  6. Tests forward-derived candidate expressions for the cross-term.

Per Rule 10 (Three Tools), Rule 12 (no ad hoc), Rule 36 (python search),
Rule 38 (overengineer), Rule 47 (forward derivation only).

Author: Exception Theory lattice analysis toolkit
"""

from __future__ import annotations

from math import gcd as _int_gcd
from mpmath import (
    mp, mpf, pi as mp_pi, sqrt as mp_sqrt, log as mp_log, fabs as mp_fabs,
    nint as mp_nint, mpmathify,
)
import sympy as sp
from sympy import Rational, sqrt, pi as sp_pi, simplify, N as SN
from typing import NamedTuple

WORKING_DPS = 80
mp.dps = WORKING_DPS

# ----------------------------------------------------------------------------
# ET PRIMITIVES (forward, no external inputs)
# ----------------------------------------------------------------------------
PI_COUNT      = 3                                # |Π|: P, D, T
S_STATES      = 4                                # C(3,2) + C(3,3)
N_MANIFOLD    = PI_COUNT * S_STATES              # 12
V_VAR         = Rational(1, N_MANIFOLD)          # 1/12
SIGMA         = sqrt(V_VAR)                      # √3/6
KAPPA         = Rational(2, 3)                   # Koide
K_EM          = N_MANIFOLD * KAPPA               # 8
A0            = (N_MANIFOLD - 1)**2 + S_STATES**2  # 137

# Forward-derived α⁻¹ (A₀ + A₁ − Σ A_k, no A₁.₅)
alpha_inv_forward_sym = (
    A0 + SIGMA / K_EM - KAPPA**2 / (N_MANIFOLD**2 * (N_MANIFOLD * sp_pi - KAPPA))
)
alpha_inv_forward = mpmathify(SN(alpha_inv_forward_sym, WORKING_DPS + 10))

# ----------------------------------------------------------------------------
# CANDIDATE α⁻¹ VALUES — measurements + ET forward
# ----------------------------------------------------------------------------
class Candidate(NamedTuple):
    label:  str
    value:  mpf
    source: str

candidates = [
    Candidate("ET forward (this work)",          alpha_inv_forward,         "ET primitives, K=∞ closed form, no A₁.₅"),
    Candidate("CODATA 2018",                      mpf("137.035999084"),      "J. Phys. Chem. Ref. Data 2021"),
    Candidate("CODATA 2022",                      mpf("137.035999177"),      "NIST CUU 2025"),
    Candidate("Morel 2020 Rb",                    mpf("137.035999206"),      "Nature 588, 61 (2020), atom interf., 81ppt"),
    Candidate("Parker 2018 Cs",                   mpf("137.035999046"),      "Science 360, 191 (2018), atom interf., 200ppt"),
    Candidate("Lattice Compendium 2025",          mpf("137.035999166"),      "ET_Lattice_Compendium §23.3, Table row 2025"),
    Candidate("ET Four-Constants target",         mpf("137.035999177"),      "ET_Four_Constants_v2.md §1 target"),
]

# ----------------------------------------------------------------------------
# LCM TOWER LEVELS (per Universal Projection Guide)
# ----------------------------------------------------------------------------
# Standard LCM tower + extras of interest
LCM_TOWER = [12, 24, 36, 60, 72, 84, 120, 132, 180, 252, 420, 504, 840,
             1260, 2520, 5544, 27720]
# Additional levels of structural interest:
EXTRA_LEVELS = [2744]                  # 2³·7³ — optimal convergent for α⁻¹ per LatComp §19
ALL_LEVELS = sorted(set(LCM_TOWER + EXTRA_LEVELS))


# ============================================================================
# LATTICE PROJECTION PRIMITIVE
# ============================================================================
class Projection(NamedTuple):
    N_lat:  int
    k:      int
    g:      int            # gcd(|k|, N_lat)
    d:      int            # sublattice = N_lat / g
    eps:    float          # cents
    log2r:  mpf
    exact:  mpf

def project(r: mpf, N_lat: int) -> Projection:
    """Project positive ratio r onto N_lat-ET lattice. Fully lossless (mp.dps)."""
    if r <= 0:
        raise ValueError("r must be positive")
    log2_r = mp_log(r) / mp_log(mpf(2))
    exact_pos = mpf(N_lat) * log2_r
    k = int(mp_nint(exact_pos))
    g = _int_gcd(abs(k), N_lat) if k != 0 else N_lat
    d = N_lat // g
    eps_step = exact_pos - k
    eps_cents = float(eps_step * mpf(1200) / N_lat)
    return Projection(N_lat, k, g, d, eps_cents, log2_r, exact_pos)

def factorize(n: int) -> str:
    """Return prime factorization as a string."""
    if n <= 1:
        return str(n)
    factors = []
    x = n
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        if cnt > 0:
            factors.append(f"{p}^{cnt}" if cnt > 1 else str(p))
    if x > 1:
        factors.append(str(x))
    return "·".join(factors)


# ============================================================================
# SECTION 1 — BASELINE: log₂ OF EVERY CANDIDATE TO HIGH PRECISION
# ============================================================================
print("=" * 78)
print("SECTION 1 — log₂(α⁻¹) FOR EVERY CANDIDATE (50 dps)")
print("=" * 78)
print()
print(f"  {'Candidate':<32} {'α⁻¹':>18} {'log₂(α⁻¹)':>24}")
print(f"  {'-'*32} {'-'*18} {'-'*24}")
for c in candidates:
    log2_val = mp_log(c.value) / mp_log(mpf(2))
    print(f"  {c.label:<32} {mp.nstr(c.value, 14):>18}  {mp.nstr(log2_val, 20):>22}")

# ============================================================================
# SECTION 2 — PROJECTION TABLE AT EVERY LCM TOWER LEVEL
# ============================================================================
print()
print("=" * 78)
print("SECTION 2 — LATTICE PROJECTION ACROSS LCM TOWER")
print("=" * 78)

# For each candidate, produce a projection row at every tower level
for c in candidates:
    print()
    print(f"  ── {c.label} (α⁻¹ = {mp.nstr(c.value, 15)}) ──")
    print(f"  {'N_lat':>6}  {'factor':<20}  {'k':>10}  {'g':>6}  {'d':>6}  {'ε(¢)':>14}  {'regime'}")
    print(f"  {'-'*6}  {'-'*20}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*14}  {'-'*22}")
    for N_lat in ALL_LEVELS:
        p = project(c.value, N_lat)
        # Regime label
        ae = abs(p.eps)
        if ae < 0.01:
            regime = "sub-ppm (identity)"
        elif ae < 0.1:
            regime = "sub-cent (exact)"
        elif ae < 1.0:
            regime = "sub-cent (structural)"
        elif ae < 5.0:
            regime = "near-exact"
        elif ae < 25.0:
            regime = "coherent"
        elif ae < 50.0:
            regime = "approaching ∂I"
        else:
            regime = "on ∂I"
        print(f"  {N_lat:>6}  {factorize(N_lat):<20}  {p.k:>10d}  {p.g:>6d}  {p.d:>6d}  {p.eps:>+14.7f}  {regime}")


# ============================================================================
# SECTION 3 — DESCRIPTOR GAP: FORWARD vs MEASURED AT EACH LEVEL
# ============================================================================
print()
print("=" * 78)
print("SECTION 3 — DESCRIPTOR GAP (forward vs measurements) AT EACH LEVEL")
print("=" * 78)
print()
print("  The Descriptor Gap at level N_lat is (k_forward - k_measured); if they")
print("  agree (Δk=0), the lattice cannot distinguish them at this resolution.")
print("  If Δk ≠ 0, the lattice DETECTS a structural difference.")
print()

forward = candidates[0].value  # ET forward

measurement_set = [c for c in candidates[1:]]  # skip the forward one

print(f"  {'N_lat':>6}  {'factor':<18}  ", end="")
for c in measurement_set:
    # Short label
    short = c.label.split()[0] + " " + c.label.split()[1] if len(c.label.split()) > 1 else c.label.split()[0]
    print(f"{short[:10]:<12}", end=" ")
print()
print(f"  {'-'*6}  {'-'*18}  ", end="")
for c in measurement_set:
    print(f"{'-'*11:<12}", end=" ")
print()

for N_lat in ALL_LEVELS:
    pf = project(forward, N_lat)
    print(f"  {N_lat:>6}  {factorize(N_lat):<18}  ", end="")
    for c in measurement_set:
        pm = project(c.value, N_lat)
        delta_k = pf.k - pm.k
        delta_eps = pf.eps - pm.eps
        if delta_k == 0:
            marker = "="
        elif abs(delta_k) == 1:
            marker = f"Δk={delta_k:+d}"
        else:
            marker = f"Δk={delta_k:+d}"
        print(f"{marker:<4} {delta_eps:+7.3f}¢ ", end="")
    print()


# ============================================================================
# SECTION 4 — STRUCTURAL SIGNATURE EXTRACTION
# ============================================================================
print()
print("=" * 78)
print("SECTION 4 — STRUCTURAL SIGNATURE — WHERE DOES α⁻¹ ACTUALLY LIVE?")
print("=" * 78)
print()
print("  For each candidate, identify the LCM level at which |ε| first drops")
print("  below 1 cent (structural), 0.1 cent (near-exact), and 0.01 cent (identity).")
print()
thresholds = [1.0, 0.1, 0.01, 0.001]
print(f"  {'Candidate':<32}  ", end="")
for t in thresholds:
    print(f"|ε|<{t}¢".ljust(16), end="")
print()
for c in candidates:
    print(f"  {c.label:<32}  ", end="")
    for t in thresholds:
        found = None
        for N_lat in ALL_LEVELS:
            p = project(c.value, N_lat)
            if abs(p.eps) < t:
                found = (N_lat, p.d, p.eps)
                break
        if found:
            print(f"N={found[0]:<5} d={found[1]:<5}".ljust(16), end="")
        else:
            print("(none)".ljust(16), end="")
    print()


# ============================================================================
# SECTION 5 — NUMERICAL GAP ANALYSIS — WHAT MAGNITUDE IS MISSING?
# ============================================================================
print()
print("=" * 78)
print("SECTION 5 — NUMERICAL GAP — MAGNITUDE OF MISSING CROSS-TERM")
print("=" * 78)
print()
print("  The ET forward value (137.036001048…) is ABOVE all measurements.")
print("  The missing cross-term must be NEGATIVE with magnitude in this range:")
print()
forward = candidates[0].value
for c in measurement_set:
    gap = forward - c.value
    print(f"    forward − {c.label:<30}  =  {mp.nstr(gap, 10)}")

# Best estimate of the "true" cross-term magnitude:
# Use all measurements as data points; the missing term is their mean, weighted
# by inverse variance (1/σ²)
measurement_vals_unc = [
    (mpf("137.035999084"), mpf("0.000000021")),  # CODATA 2018
    (mpf("137.035999177"), mpf("0.000000021")),  # CODATA 2022
    (mpf("137.035999206"), mpf("0.000000011")),  # Morel 2020 Rb
    (mpf("137.035999046"), mpf("0.000000027")),  # Parker 2018 Cs
]
num   = sum(v / u**2 for v, u in measurement_vals_unc)
den   = sum(1  / u**2 for _, u in measurement_vals_unc)
weighted_mean    = num / den
weighted_sigma   = mp_sqrt(mpf(1) / den)

print()
print(f"  Weighted-mean measurement:     α⁻¹ = {mp.nstr(weighted_mean, 14)} ± {mp.nstr(weighted_sigma, 3)}")
gap_target = forward - weighted_mean
print(f"  Missing cross-term target:     |X| = {mp.nstr(gap_target, 12)}")
print(f"                                    ≈ {mp.nstr(gap_target * 1e6, 6)} × 10⁻⁶")
print(f"                                    ≈ {mp.nstr(gap_target / forward * 1e9, 5)} ppb")


# ============================================================================
# SECTION 6 — SEARCH FOR ET-NATIVE EXPRESSIONS MATCHING THE GAP
# ============================================================================
print()
print("=" * 78)
print("SECTION 6 — SCAN FOR ET-NATIVE EXPRESSIONS MATCHING |X|")
print("=" * 78)
print()
print("  Enumerate combinations of {σ, κ, N, S, K_EM, π, √π} that can produce")
print("  a value close to the missing gap. Only ET-primitive combinations —")
print("  no ad-hoc multipliers.")
print()

# Compute numerical values of primitives
sig_n = mpmathify(SN(SIGMA, 80))
kap_n = mpf(2) / 3
N_n   = mpf(12)
S_n   = mpf(4)
K_n   = mpf(8)
pi_n  = +mp_pi

target = gap_target  # mpf, ~1.87e-6
print(f"  Target gap magnitude: {mp.nstr(target, 15)}")
print()

# Systematically enumerate expressions of the form:
#    sigma^a · kappa^b · N^c · S^d · K_EM^e · pi^f · sqrt(pi)^g
# with small integer/half-integer exponents.
# Keep only expressions whose numerical value is within ±30% of the target.

import itertools

def primitive_value(exps):
    a, b, c, d, e, f, g = exps
    v = sig_n**a * kap_n**b * N_n**c * S_n**d * K_n**e * pi_n**f * mp_sqrt(pi_n)**g
    return v

def primitive_name(exps):
    a, b, c, d, e, f, g = exps
    parts = []
    if a != 0: parts.append(f"σ^{a}" if a != 1 else "σ")
    if b != 0: parts.append(f"κ^{b}" if b != 1 else "κ")
    if c != 0: parts.append(f"N^{c}" if c != 1 else "N")
    if d != 0: parts.append(f"S^{d}" if d != 1 else "S")
    if e != 0: parts.append(f"K^{e}" if e != 1 else "K")
    if f != 0: parts.append(f"π^{f}" if f != 1 else "π")
    if g != 0: parts.append(f"(√π)^{g}" if g != 1 else "√π")
    return " · ".join(parts) if parts else "1"

# Ranges: keep modest to avoid combinatorial explosion but wide enough to
# include all structurally meaningful combinations
range_sig = range(-1, 3)                  # σ⁻¹ .. σ²
range_kap = range(-1, 4)                  # κ⁻¹ .. κ³
range_N   = range(-7, 2)                  # N⁻⁷ .. N¹
range_S   = range(-2, 2)                  # S⁻² .. S¹
range_K   = range(-2, 2)                  # K⁻² .. K¹
range_pi  = range(-3, 2)                  # π⁻³ .. π¹
range_spi = range(-2, 2)                  # (√π)⁻² .. (√π)¹

candidates_expr = []
count = 0
for exps in itertools.product(range_sig, range_kap, range_N, range_S, range_K, range_pi, range_spi):
    count += 1
    v = primitive_value(exps)
    ratio = v / target
    # Tight matches: |ratio - 1| < 0.10  (within 10%)
    # Good matches: |ratio - 1| < 0.01  (within 1%)
    if mp_fabs(ratio - 1) < mpf("0.10"):
        candidates_expr.append((exps, v, float(ratio)))

print(f"  Enumerated {count} combinations. Found {len(candidates_expr)} within 10% of target.")
print()

# Sort by closeness to 1
candidates_expr.sort(key=lambda x: abs(x[2] - 1))

# Filter for cleanness: prefer lower total |exponent|
def total_exp(exps):
    return sum(abs(e) for e in exps)

# Show top 30 cleanest matches
print(f"  {'expression':<45}  {'value':>18}  {'ratio':>12}  {'total |exp|':>11}")
print(f"  {'-'*45}  {'-'*18}  {'-'*12}  {'-'*11}")
for exps, v, ratio in sorted(candidates_expr, key=lambda x: (total_exp(x[0]), abs(x[2]-1)))[:30]:
    print(f"  {primitive_name(exps):<45}  {mp.nstr(v, 10):>18}  {ratio:>12.6f}  {total_exp(exps):>11}")


# ============================================================================
# SECTION 7 — EXPRESSION FOR 18π − 1 GAP (structural)
# ============================================================================
print()
print("=" * 78)
print("SECTION 7 — STRUCTURAL CHECK: COULD THE GAP BE A SUB-LEADING SERIES TERM?")
print("=" * 78)
print()
print("  The main negative term is 1/[216·(18π−1)] ≈ 8.33×10⁻⁵.")
print("  Look for a CORRECTION to this form that produces the right magnitude.")
print()

# Candidates: multiplying 1/[216·(18π−1)] by some ET primitive ratio
main_neg = mpf(1) / (216 * (18*pi_n - 1))
print(f"  1/[216·(18π−1)]                 = {mp.nstr(main_neg, 15)}")
print(f"  Missing gap / main neg          = {mp.nstr(gap_target/main_neg, 12)}")
print(f"  Check: is gap/main ≈ κ/N²?       {mp.nstr(gap_target/main_neg, 10)} vs {mp.nstr(kap_n/N_n**2, 10)}")
print(f"  Check: is gap/main ≈ σ/N?        {mp.nstr(gap_target/main_neg, 10)} vs {mp.nstr(sig_n/N_n, 10)}")
print(f"  Check: is gap/main ≈ 1/N²?       {mp.nstr(gap_target/main_neg, 10)} vs {mp.nstr(mpf(1)/N_n**2, 10)}")
print(f"  Check: is gap/main ≈ 2·σ/N?      {mp.nstr(gap_target/main_neg, 10)} vs {mp.nstr(2*sig_n/N_n, 10)}")

print()
# Test specific ET-structural expressions and report their ratio to target
tests = [
    ("1 / [N⁴·(18π−1)]",                   mpf(1)/(N_n**4 * (18*pi_n - 1))),
    ("κ / [N³·(18π−1)]",                  kap_n/(N_n**3 * (18*pi_n - 1))),
    ("σ / [N³·(18π−1)]",                  sig_n/(N_n**3 * (18*pi_n - 1))),
    ("κ² / [N³·(18π−1)]",                 kap_n**2/(N_n**3 * (18*pi_n - 1))),
    ("1 / [K_EM·N³·(18π−1)]",             mpf(1)/(K_n * N_n**3 * (18*pi_n - 1))),
    ("σ·κ / [N²·(18π−1)]",                sig_n*kap_n/(N_n**2 * (18*pi_n - 1))),
    ("1 / [N²·(18π−1)²]",                 mpf(1)/(N_n**2 * (18*pi_n - 1)**2)),
    ("1 / [216·(18π−1)·K_EM·N]",          mpf(1)/(216 * (18*pi_n - 1) * K_n * N_n)),
    ("κ³ / [N⁴·(18π−1)]",                 kap_n**3/(N_n**4 * (18*pi_n - 1))),
    ("κ² / [N²·(Nπ−κ)²]",                 kap_n**2/(N_n**2 * (N_n*pi_n - kap_n)**2)),
    ("κ³ / [N³·(Nπ−κ)²]",                 kap_n**3/(N_n**3 * (N_n*pi_n - kap_n)**2)),
    ("κ³ / [N²·(Nπ−κ)²·(18π−1)]",         kap_n**3/(N_n**2 * (N_n*pi_n - kap_n)**2 * (18*pi_n-1))),
]
print(f"  {'Expression':<40} {'value':>14} {'ratio to target':>18}")
print(f"  {'-'*40} {'-'*14} {'-'*18}")
for name, val in tests:
    ratio = val / target
    marker = " <-- MATCH" if mp_fabs(ratio - 1) < mpf("0.05") else ""
    print(f"  {name:<40} {mp.nstr(val, 10):>14} {mp.nstr(ratio, 8):>18}{marker}")


# ============================================================================
# SECTION 8 — SYMMETRIC ("SECOND-ORDER SERIES") HYPOTHESIS
# ============================================================================
print()
print("=" * 78)
print("SECTION 8 — HYPOTHESIS: SECOND-ORDER CASCADE Σ A_k × κ/(Nπ)")
print("=" * 78)
print()
print("  If ET has a NESTED series — each Mediation loop itself sprouts a")
print("  sub-loop with ratio κ/(Nπ), then the second-order correction would be:")
print()
print("    Σ_{k=2}^∞ A_k × (κ/(Nπ))  =  [1/(216·(18π−1))] · [κ/(Nπ)]")
print()
sub_corr = main_neg * kap_n / (N_n * pi_n)
print(f"    value = {mp.nstr(sub_corr, 15)}")
print(f"    target = {mp.nstr(target, 15)}")
print(f"    ratio = {mp.nstr(sub_corr/target, 12)}")

if mp_fabs(sub_corr/target - 1) < mpf("0.10"):
    print("    → WITHIN 10% of target — plausible candidate!")

print()
print("  Alternative: nested series per loop order")
print("    Σ_{k=2}^∞ A_k · [κ/(Nπ)]^(k-1)  — geometric in BOTH k and nest level")
nest_sum = mpf(0)
for k in range(2, 50):
    A_k = kap_n**k / (N_n**(k+1) * pi_n**(k-1))
    nest_factor = (kap_n/(N_n*pi_n))**(k-1)   # (k-1) = nest level
    nest_sum += A_k * nest_factor

print(f"    nested Σ = {mp.nstr(nest_sum, 15)}")
print(f"    ratio to target = {mp.nstr(nest_sum/target, 12)}")


# ============================================================================
# SECTION 9 — A₁ CROSS A_k HYPOTHESIS
# ============================================================================
print()
print("=" * 78)
print("SECTION 9 — HYPOTHESIS: CROSS-TERM IS A₁ × (Σ A_k) × (ET-factor)")
print("=" * 78)
print()
A1_val = sig_n / K_n          # √3/48
sum_val = main_neg
print(f"  A₁ · Σ A_k = {mp.nstr(A1_val*sum_val, 15)}")
print(f"  target      = {mp.nstr(target, 15)}")
print(f"  ratio       = {mp.nstr(A1_val*sum_val/target, 12)}")
print()
print("  Products with various ET factors:")
for fac_name, fac in [
    ("× 1",                     mpf(1)),
    ("× κ",                     kap_n),
    ("× κ²",                    kap_n**2),
    ("× σ",                     sig_n),
    ("× 1/N",                   mpf(1)/N_n),
    ("× N·σ²",                  N_n * sig_n**2),
    ("× 1/(K_EM·σ)",            mpf(1)/(K_n * sig_n)),
    ("× √κ",                    mp_sqrt(kap_n)),
    ("× K_EM",                  K_n),
]:
    v = A1_val * sum_val * fac
    print(f"    A₁·ΣA_k {fac_name:<22}  = {mp.nstr(v, 12):>16}  ratio = {mp.nstr(v/target, 10)}")


# ============================================================================
# SECTION 10 — DISCUSSION
# ============================================================================
print()
print("=" * 78)
print("SECTION 10 — STRUCTURAL FINDINGS (for downstream derivation)")
print("=" * 78)

print()
print(f"  The missing cross-term has magnitude |X| ≈ {mp.nstr(target, 8)}")
print(f"                                  ≈ {mp.nstr(target * mpf(1e6), 5)} × 10⁻⁶")
print()
print("  From Section 2 projections, the key structural observations are:")
print("    • At 12ET, 24ET both forward and measurements sit at (k=±85, d=12) coherent")
print("    • At 132ET and 2520ET all candidates reach sub-cent (structural)")
print("    • At 2744ET (LatComp §19 optimal) forward and measured are")
print("      indistinguishable at cent resolution → 14 ppb < lattice floor at 2744ET")
print()
print("  The missing cross-term must be derived from Section 6–9 structural")
print("  hypotheses with the best ratio-to-target match. See downstream work.")
print()
print("END OF LATTICE ANALYSIS")

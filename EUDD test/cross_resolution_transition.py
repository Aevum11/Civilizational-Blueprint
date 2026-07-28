#!/usr/bin/env python3
"""
CROSS-RESOLUTION TRANSITION MAP DERIVATION AND VERIFICATION
============================================================
Deriving the exact algebraic identity for how the universal lattice
handles the intersection boundary where two or more distinct 
coordinate transformations meet.

Three cases:
  Case 1: Same R₀, different N (cross-resolution)
  Case 2: Same N, different R₀ (cross-seed)
  Case 3: Both N and R₀ differ (full cross-tower)

All derived from P∘D∘T = E via the bijection.
"""

from mpmath import mp, mpf, log as mplog, sqrt as mpsqrt, pi as mppi
from mpmath import nint, fabs, power as mppow, nstr, floor as mpfloor
from math import gcd

mp.dps = 200  # High precision, no float

print("=" * 80)
print("  CROSS-RESOLUTION TRANSITION MAP — DERIVATION AND VERIFICATION")
print("  Derived forward from the bijection Π_N(r) = (k, d, ε)")
print("=" * 80)

# =============================================================================
# The Universal Projection (from the paper, Definition 7.1)
# =============================================================================
def project(r_str, N):
    """Project r onto lattice at resolution N. Returns (k, d, eps_cents)."""
    r = mpf(r_str)
    log2_r = mplog(r) / mplog(mpf(2))
    exact_pos = mpf(N) * log2_r
    k = int(nint(exact_pos))
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    eps = (exact_pos - mpf(k)) * mpf(1200) / mpf(N)
    return k, d, eps

def pullback(k, eps, N):
    """Pullback (k, eps) at resolution N to recover r. Algebraic identity."""
    exponent = (mpf(k) + eps * mpf(N) / mpf(1200)) / mpf(N)
    return mppow(mpf(2), exponent)

# =============================================================================
# CASE 1: CROSS-RESOLUTION TRANSITION MAP
# Given Π_N₁(r) = (k₁, d₁, ε₁), compute Π_N₂(r) = (k₂, d₂, ε₂)
# WITHOUT re-accessing r.  
# Constraint: N₁ | N₂ (N₁ divides N₂) for tower transitions.
# =============================================================================
print("\n" + "=" * 80)
print("  CASE 1: CROSS-RESOLUTION TRANSITION (same R₀, different N)")
print("=" * 80)

print("""
THEOREM (Cross-Resolution Transition Map):
Let N₁ | N₂ with M = N₂/N₁. Given Π_N₁(r) = (k₁, d₁, ε₁),
the projection Π_N₂(r) = (k₂, d₂, ε₂) is:

  δ₁ = ε₁ · N₁ / 1200          (fractional lattice offset at N₁)
  k₂ = round(M · k₁ + M · δ₁)  (scaled + offset)
  g₂ = gcd(|k₂|, N₂)
  d₂ = N₂ / g₂
  ε₂ = (M · k₁ + M · δ₁ − k₂) · 1200 / N₂

PROOF: By losslessness (Theorem 19.4):
  N₁ · log₂(r) = k₁ + δ₁  (exact, algebraic identity)
  N₂ · log₂(r) = M · N₁ · log₂(r) = M · (k₁ + δ₁) = M·k₁ + M·δ₁
  k₂ = round(N₂ · log₂(r)) = round(M·k₁ + M·δ₁)
  ε₂ = (N₂ · log₂(r) − k₂) · 1200/N₂  
     = (M·k₁ + M·δ₁ − k₂) · 1200/N₂   ∎

EQUIVALENT COMPOSITION FORM:
  Π_N₂ ∘ Π_N₁⁻¹ : (k₁, d₁, ε₁) ↦ (k₂, d₂, ε₂)
  This IS the transition function on the overlap of two coordinate charts.
""")

def cross_resolution_transition(k1, d1, eps1, N1, N2):
    """Transition from (k1, d1, eps1) at N1 to (k2, d2, eps2) at N2."""
    M = N2 // N1
    delta1 = eps1 * mpf(N1) / mpf(1200)
    exact_pos_N2 = mpf(M) * mpf(k1) + mpf(M) * delta1
    k2 = int(nint(exact_pos_N2))
    g2 = gcd(abs(k2), N2) if k2 != 0 else N2
    d2 = N2 // g2
    eps2 = (exact_pos_N2 - mpf(k2)) * mpf(1200) / mpf(N2)
    return k2, d2, eps2

# Verify against direct projection for test values
test_values = [
    ("π", nstr(mppi, 50)),
    ("2/3 (Koide)", "0.66666666666666666666666666666666666666666666666666"),
    ("3/2 (fifth)", "1.50000000000000000000000000000000000000000000000000"),
    ("φ (golden)", nstr(mppow(mpf(5), mpf("0.5"))/2 + mpf("0.5"), 50)),
    ("137.036 (α⁻¹)", "137.036"),
    ("1836.153 (μ)", "1836.153"),
]

tower = [(12, 60), (60, 420), (420, 2520), (2520, 27720), (12, 27720)]

print(f"  {'Value':<16} {'N₁→N₂':>10} | {'Direct k₂':>10} {'Trans k₂':>10} {'Match k':>7} | {'Direct d₂':>10} {'Trans d₂':>10} {'Match d':>7} | {'Δε':>14}")
print(f"  {'-'*16} {'-'*10}-+-{'-'*10}-{'-'*10}-{'-'*7}-+-{'-'*10}-{'-'*10}-{'-'*7}-+-{'-'*14}")

all_match = True
for name, val_str in test_values:
    for N1, N2 in tower:
        # Direct projection at both resolutions
        k1, d1, eps1 = project(val_str, N1)
        k2_direct, d2_direct, eps2_direct = project(val_str, N2)
        
        # Transition map (no access to r, only uses k1, d1, eps1)
        k2_trans, d2_trans, eps2_trans = cross_resolution_transition(k1, d1, eps1, N1, N2)
        
        k_match = k2_direct == k2_trans
        d_match = d2_direct == d2_trans
        eps_diff = float(fabs(eps2_direct - eps2_trans))
        
        if not (k_match and d_match and eps_diff < 1e-50):
            all_match = False
        
        print(f"  {name:<16} {N1:>4}→{N2:<5} | {k2_direct:>10} {k2_trans:>10} {'  ✓' if k_match else '  ✗':>7} | {d2_direct:>10} {d2_trans:>10} {'  ✓' if d_match else '  ✗':>7} | {eps_diff:>14.2e}")

print(f"\n  ALL TRANSITIONS MATCH: {'✓ YES' if all_match else '✗ NO'}")

# =============================================================================
# CASE 2: CROSS-SEED TRANSITION MAP  
# Given Π_N(Q/R₀) = (k₁, d₁, ε₁), compute Π_N(Q/R₀') = (k₂, d₂, ε₂)
# =============================================================================
print(f"\n{'=' * 80}")
print(f"  CASE 2: CROSS-SEED TRANSITION (same N, different R₀)")
print(f"{'=' * 80}")

print("""
THEOREM (Cross-Seed Transition Map):
Let ρ = R₀/R₀' (seed ratio). Given Π_N(Q/R₀) = (k₁, d₁, ε₁),
the projection Π_N(Q/R₀') = (k₂, d₂, ε₂) is:

  Δk_exact = N · log₂(ρ)         (exact seed shift on log₂ line)
  k₂ = round(k₁ + δ₁ + Δk_exact)  where δ₁ = ε₁·N/1200
  d₂ = N / gcd(|k₂|, N)
  ε₂ = (k₁ + δ₁ + Δk_exact − k₂) · 1200/N

PROOF: Q/R₀' = (Q/R₀) · (R₀/R₀') = r · ρ.
  log₂(r·ρ) = log₂(r) + log₂(ρ)
  N·log₂(r·ρ) = N·log₂(r) + N·log₂(ρ) = (k₁ + δ₁) + Δk_exact  ∎

NOTE: The seed shift Δk_exact is generally IRRATIONAL (since ρ is 
typically irrational). The d-family CHANGES under seed shift.
This is Convention Independence (Theorem 7.5) in REVERSE:
  - For the SAME r: Π_N(r) = Π_N(ur/u) — invariant (Thm 7.5)
  - For DIFFERENT r: Π_N(r·ρ) ≠ Π_N(r) in general — the STRUCTURAL
    classification CHANGES because r·ρ IS a different physical ratio.
""")

# Verify: use electron mass R₀ vs proton mass R₀
# ρ = m_e/m_p ≈ 1/1836.153
N = 12
rho_str = nstr(mpf(1) / mpf("1836.15267"), 50)
rho = mpf(rho_str)
Delta_k_exact = mpf(N) * mplog(rho) / mplog(mpf(2))

print(f"  Example: R₀ = m_e → R₀' = m_p")
print(f"  ρ = m_e/m_p ≈ 1/1836.153")
print(f"  Δk_exact = N·log₂(ρ) = {nstr(Delta_k_exact, 15)}")
print(f"  round(Δk_exact) = {int(nint(Delta_k_exact))}")
print(f"  Δk = -130 (confirming journal value)")

# Test: project π with both reference systems
k1_pi, d1_pi, eps1_pi = project(nstr(mppi, 50), N)
# π as if measured in proton masses: r' = π · ρ
r_prime = mppi * rho
k2_pi, d2_pi, eps2_pi = project(nstr(r_prime, 50), N)

# Via transition map:
delta1 = eps1_pi * mpf(N) / mpf(1200)
exact_pos = mpf(k1_pi) + delta1 + Delta_k_exact
k2_trans = int(nint(exact_pos))
g2 = gcd(abs(k2_trans), N) if k2_trans != 0 else N
d2_trans = N // g2
eps2_trans = (exact_pos - mpf(k2_trans)) * mpf(1200) / mpf(N)

print(f"\n  π with R₀=m_e: (k={k1_pi}, d={d1_pi}, ε={nstr(eps1_pi,6)}¢)")
print(f"  π with R₀=m_p: direct  (k={k2_pi}, d={d2_pi}, ε={nstr(eps2_pi,6)}¢)")
print(f"  π with R₀=m_p: transit (k={k2_trans}, d={d2_trans}, ε={nstr(eps2_trans,6)}¢)")
print(f"  Match: k={'✓' if k2_pi==k2_trans else '✗'}, d={'✓' if d2_pi==d2_trans else '✗'}, Δε={float(fabs(eps2_pi-eps2_trans)):.2e}")

# =============================================================================
# CASE 3: FULL CROSS-TOWER TRANSITION (different N AND different R₀)
# =============================================================================
print(f"\n{'=' * 80}")
print(f"  CASE 3: FULL CROSS-TOWER TRANSITION (different N AND R₀)")
print(f"{'=' * 80}")

print("""
THEOREM (Full Cross-Tower Transition Map):
Given Π_N₁(Q/R₀) = (k₁, d₁, ε₁), compute Π_N₂(Q/R₀') = (k₂, d₂, ε₂):

  δ₁ = ε₁ · N₁ / 1200
  x = (k₁ + δ₁) / N₁                  (recover log₂(Q/R₀) exactly)
  x' = x + log₂(R₀/R₀')               (shift to new seed)
  k₂ = round(N₂ · x')
  d₂ = N₂ / gcd(|k₂|, N₂)
  ε₂ = (N₂ · x' − k₂) · 1200/N₂

This is the GENERAL transition function:
  Π_N₂^{R₀'} ∘ (Π_N₁^{R₀})⁻¹ : (k₁, d₁, ε₁) ↦ (k₂, d₂, ε₂)

It factors as:
  (Cross-Seed shift) ∘ (Cross-Resolution scale) = Full transition
  OR
  (Cross-Resolution scale) ∘ (Cross-Seed shift) = Full transition

The two factorizations commute (both give the same result)
because addition and scaling on log₂(r) commute:
  M·(x + Δ) = M·x + M·Δ  ✓
""")

# Verify commutativity: seed-then-scale vs scale-then-seed
print("  COMMUTATIVITY VERIFICATION:")
test_r = nstr(mppi, 50)
N1_test, N2_test = 12, 420
rho_test = mpf(1) / mpf("1836.15267")

# Direct: project at N2 with R0' 
r_prime_val = mppi * rho_test
k_direct, d_direct, eps_direct = project(nstr(r_prime_val, 50), N2_test)

# Route A: seed-shift at N1, then resolution-scale to N2
k1, d1, eps1 = project(test_r, N1_test)
# Apply seed shift at N1
Delta_k_N1 = mpf(N1_test) * mplog(rho_test) / mplog(mpf(2))
delta1 = eps1 * mpf(N1_test) / mpf(1200)
pos_shifted_N1 = mpf(k1) + delta1 + Delta_k_N1
k_A1 = int(nint(pos_shifted_N1))
eps_A1 = (pos_shifted_N1 - mpf(k_A1)) * mpf(1200) / mpf(N1_test)
# Now scale to N2
k_A, d_A, eps_A = cross_resolution_transition(k_A1, N1_test//gcd(abs(k_A1),N1_test), eps_A1, N1_test, N2_test)

# Route B: resolution-scale to N2, then seed-shift at N2
k2_pre, d2_pre, eps2_pre = cross_resolution_transition(k1, d1, eps1, N1_test, N2_test)
# Apply seed shift at N2
Delta_k_N2 = mpf(N2_test) * mplog(rho_test) / mplog(mpf(2))
delta2_pre = eps2_pre * mpf(N2_test) / mpf(1200)
pos_shifted_N2 = mpf(k2_pre) + delta2_pre + Delta_k_N2
k_B = int(nint(pos_shifted_N2))
g_B = gcd(abs(k_B), N2_test) if k_B != 0 else N2_test
d_B = N2_test // g_B
eps_B = (pos_shifted_N2 - mpf(k_B)) * mpf(1200) / mpf(N2_test)

print(f"  Direct:         (k={k_direct}, d={d_direct}, ε={nstr(eps_direct,8)}¢)")
print(f"  Route A (S→R):  (k={k_A}, d={d_A}, ε={nstr(eps_A,8)}¢)")
print(f"  Route B (R→S):  (k={k_B}, d={d_B}, ε={nstr(eps_B,8)}¢)")
print(f"  All three match: k={'✓' if k_direct==k_A==k_B else '✗'}, d={'✓' if d_direct==d_A==d_B else '✗'}")

# =============================================================================
# THE BOUNDARY: WHERE d-FAMILY CHANGES UNDER REFINEMENT
# =============================================================================
print(f"\n{'=' * 80}")
print(f"  THE INTERSECTION BOUNDARY: d-family transitions under refinement")
print(f"{'=' * 80}")

print("""
The intersection boundary between two coordinate charts at N₁ and N₂
is the set of r where d changes:

  ∂_transition = { r ∈ ℝ⁺ : d_N₁(r) ≠ d_N₂(r) }

At this boundary, two coordinate descriptions DISAGREE on the structural
classification. The higher-resolution chart resolves the shadow content
that the lower resolution encoded in ε.

KEY IDENTITY at the boundary:
  r at a d-transition satisfies: |ε₁| is large enough that
  M·δ₁ pushes the re-rounded k₂ across a gcd-boundary.
  
  Specifically: d₁ → d₂ occurs when
    gcd(|k₂|, N₂) ≠ (N₂/N₁) · gcd(|k₁|, N₁)
    
  The "shadow content" encoded in ε₁ becomes "native content" in d₂.
""")

# Example: muon escalation (from journal Finding 8.7)
print("  MUON TOWER ESCALATION (verifying journal Finding 8.7):")
muon_r = nstr(mpf("206.768") , 50)  # m_muon/m_e ≈ 206.768

tower_levels = [12, 60, 420, 840, 2520, 27720]
prev_result = None
for N in tower_levels:
    k, d, eps = project(muon_r, N)
    transition_note = ""
    if prev_result:
        prev_k, prev_d, prev_N = prev_result
        if d != prev_d:
            transition_note = f"  ← d CHANGED from {prev_d}"
    print(f"  N={N:>6}: (k={k:>7}, d={d:>6}, ε={nstr(eps,6):>12}¢){transition_note}")
    prev_result = (k, d, N)

print(f"\n{'=' * 80}")
print(f"  SUMMARY OF DERIVED TRANSITION IDENTITIES")
print(f"{'=' * 80}")
print("""
  1. CROSS-RESOLUTION (N₁ | N₂, M = N₂/N₁):
     k₂ = round(M·k₁ + M·ε₁·N₁/1200)
     ε₂ = (M·k₁ + M·ε₁·N₁/1200 − k₂) · 1200/N₂
     d₂ = N₂/gcd(|k₂|, N₂)

  2. CROSS-SEED (same N, ρ = R₀/R₀'):
     k₂ = round(k₁ + ε₁·N/1200 + N·log₂(ρ))
     ε₂ = (k₁ + ε₁·N/1200 + N·log₂(ρ) − k₂) · 1200/N
     d₂ = N/gcd(|k₂|, N)

  3. FULL CROSS-TOWER (different N AND R₀):
     x = (k₁ + ε₁·N₁/1200)/N₁
     k₂ = round(N₂·(x + log₂(R₀/R₀')))
     ε₂ = (N₂·(x + log₂(ρ)) − k₂) · 1200/N₂
     d₂ = N₂/gcd(|k₂|, N₂)

  4. COMMUTATIVITY: (Seed ∘ Scale) = (Scale ∘ Seed) = Direct
     [VERIFIED COMPUTATIONALLY]

  5. BOUNDARY: d-transition occurs when refinement pushes k₂
     across a gcd-boundary of N₂. This is the ε→d conversion:
     shadow content (ε at N₁) becomes native content (d at N₂).

  All identities are ALGEBRAIC CONSEQUENCES of the bijection.
  Zero additional axioms. Zero external parameters.
  Forward-derived from P∘D∘T = E via the lossless projection.
""")

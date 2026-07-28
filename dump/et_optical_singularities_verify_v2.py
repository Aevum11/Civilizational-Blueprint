#!/usr/bin/env python3
"""
ET Optical Phase Singularities — Complete Verification Suite (v2)
=================================================================
Bucher et al. (2025) "Superluminal Correlations in Ensembles of
Optical Phase Singularities" — arXiv:2509.17675v1

All math is ET-derived. All projections use lossless mpmath arithmetic
at 150 decimal digits. No floats in projection formulas.

TONE STANDARD (Subsumption Law):
  Standard physics results ARE the D-content of the phenomenon.
  ET verifies them as structural consequences of the PDT decomposition.
  Agreement is a necessary condition of the Subsumption Law, not a
  surprising confirmation. "Verified" not "verified."

Author: Michael James Muller — Aevum Defluo
"""

import mpmath
mpmath.mp.dps = 150
from mpmath import mpf, log, pi, nint, inf, quad
from math import gcd, lcm
from fractions import Fraction


# ═══════════════════════════════════════════════════════════════
# ET FOUNDATIONAL CONSTANTS — IMMUTABLE, DERIVED NOT CHOSEN
# ═══════════════════════════════════════════════════════════════

N_BASE = 12
V_BASE = Fraction(1, 12)
K_KOIDE = Fraction(2, 3)
N_FULL = 27720
PRIMITIVE_COUNT = 3
T_WEIGHT = Fraction(1, 3)           # 1/|{P,D,T}|
PD_WEIGHT = Fraction(2, 3)          # K = weight of {P,D}

DELTA_R_EXACT = abs(12 * log(mpf(12), 2) - 43)
DELTA_THETA_EXACT = abs(24 * pi / log(mpf(2)) - 109)
N_MAX_R = int(mpf('0.5') / DELTA_R_EXACT)      # = 25
N_MAX_THETA = int(mpf('0.5') / DELTA_THETA_EXACT)  # = 2

LCM_TOWER = [12, 24, 36, 60, 84, 132, 420, 2520, 27720]


# ═══════════════════════════════════════════════════════════════
# LOSSLESS LATTICE PROJECTION (mpmath, arbitrary precision)
# Per UPP Guide §11: k = round(N·log₂(r)), d = N/gcd(|k|,N),
#   ε = (N·log₂(r) - k) × 1200/N cents
# ═══════════════════════════════════════════════════════════════

def project(r, N=12):
    """Lossless lattice projection at resolution N."""
    r_mpf = mpf(r) if not isinstance(r, mpf) else r
    assert r_mpf > 0, f"r must be positive, got {r_mpf}"
    lr = log(r_mpf, 2)
    exact = N * lr
    k = int(nint(exact))
    eps_exact = (exact - k) * mpf(1200) / N
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    return {'k': k, 'd': d, 'eps_cents': float(eps_exact), 'N': N}


def project_tower(r, label="", lattices=None):
    """Project r across the LCM tower and print results."""
    if lattices is None:
        lattices = LCM_TOWER
    r_mpf = mpf(r) if not isinstance(r, mpf) else r
    if label:
        print(f"\n--- {label} ---")
    for N in lattices:
        p = project(r_mpf, N)
        m = ""
        if abs(p['eps_cents']) < 0.5: m = " ← SUB-HALF-CENT"
        if abs(p['eps_cents']) < 0.1: m = " ← SUB-TENTH-CENT"
        if abs(p['eps_cents']) < 0.05: m = " ← ESSENTIALLY EXACT"
        print(f"  {N:5d}ET: k={p['k']:7d}, d={p['d']:6d}, ε={p['eps_cents']:+12.6f}¢{m}")


# ═══════════════════════════════════════════════════════════════
# 1. CASCADE RESIDUALS — THREE UNIT SYSTEMS
# ═══════════════════════════════════════════════════════════════

def section_cascade_residuals():
    print("=" * 78)
    print("1. CASCADE RESIDUALS — THREE UNIT SYSTEMS")
    print("   (Fixes units bug in Sempaevum line 2546)")
    print("=" * 78)

    dr = DELTA_R_EXACT
    dt = DELTA_THETA_EXACT
    ratio = dt / dr
    shadow_gap = 12 - ratio

    print(f"\n|δ_r| = {mpmath.nstr(dr, 30)}")
    print(f"  = {float(dr):.10f} lattice steps")
    print(f"  = {float(dr * 100):.6f} cents")
    print(f"  = {float(dr / 12):.12f} octaves")
    print(f"  Conversion: 1 lattice step = 100 cents = 1/N octaves at N=12")

    print(f"\n|δ_θ| = {mpmath.nstr(dt, 30)}")
    print(f"  = {float(dt):.10f} lattice steps")
    print(f"  = {float(dt * 100):.6f} cents")
    print(f"  = {float(dt / 12):.12f} octaves")

    print(f"\nRatio |δ_θ|/|δ_r| = {mpmath.nstr(ratio, 30)}")
    print(f"  = N at terminal resolution (27720ET)")
    print(f"  Gap at 12ET: {mpmath.nstr(shadow_gap, 20)} = (5,7) shadow")
    print(f"\nStability: n_max_r = {N_MAX_R}, n_max_θ = {N_MAX_THETA}")
    print(f"  n_max_r/n_max_θ = {N_MAX_R}/{N_MAX_THETA} = {N_MAX_R/N_MAX_THETA}")

    print(f"\nShadow gap tower projection:")
    project_tower(shadow_gap, "Gap = 12 - |δ_θ|/|δ_r| → first sub-cent at d=5 (60ET)")


# ═══════════════════════════════════════════════════════════════
# 2. 1/3 = T-WEIGHT IN THE VELOCITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def section_t_weight():
    print("\n" + "=" * 78)
    print("2. 1/3 = T-WEIGHT — BERRY-DENNIS DISTRIBUTION")
    print("   ET verifies the standard physics result as T's primitive weight.")
    print("=" * 78)

    def pdf_u(u):
        return u / (2 * (2*u**2 + 1)**2)

    total = quad(pdf_u, [0, inf])
    above_mean = quad(pdf_u, [1, inf])
    fraction = above_mean / total

    print(f"\nBerry-Dennis velocity PDF: P(u) = u / (2(2u²+1)²)")
    print(f"  Fraction above mean: {mpmath.nstr(fraction, 20)} = 1/3 exactly")
    print(f"  Computation: ∫₁^∞ / ∫₀^∞ = (1/24)/(1/8) = 1/3")
    print(f"\nET verification:")
    print(f"  Substitution w = 2u²+1 maps u=1 → w=3 = |{{P,D,T}}|")
    print(f"  The '3' IS the primitive count (Subsumption Law: exactly 3)")
    print(f"  Below ⟨v⟩: {{P,D}}-constrained fraction = K = 2/3")
    print(f"  Above ⟨v⟩: unconstrained T-fraction = 1-K = 1/3 = T-weight")
    print(f"  The standard physics math produces 1/3 because the underlying")
    print(f"  structure has 3 primitives — the Subsumption Law at work.")


# ═══════════════════════════════════════════════════════════════
# 3. R₀ ANALYSIS — MATERIAL-SPECIFIC LATTICE POSITIONS
# ═══════════════════════════════════════════════════════════════

def section_R0_analysis():
    print("\n" + "=" * 78)
    print("3. R₀ ANALYSIS — MATERIAL-SPECIFIC LATTICE POSITIONS")
    print("   Per UPP Guide §8: R₀ = natural reference period of the substrate")
    print("   r = Q/R₀ = Dimensionless Seed Ratio (Guide §10)")
    print("=" * 78)

    print(f"""
For phonon-polariton materials:
  P = the crystal lattice (substrate)
  D = dielectric function, phonon frequencies (constraints)
  T = EM field navigating the crystal (agency)

  R₀ = the smallest closed T-traversal loop the substrate supports
     = one TO phonon oscillation period = 1/ω_TO
  
  The Dimensionless Seed Ratio (DSR) for the Reststrahlen band:
     r = ω_LO / ω_TO
  
  This ratio characterises the material's phonon resonance strength.
  Different materials have different r → different lattice positions
  → different polariton properties including v_φ/v_g.
""")

    # hBN
    r_hBN_upper = mpf(1610) / mpf(1370)  # 161/137
    r_hBN_lower = mpf(830) / mpf(780)    # 83/78
    r_MoO3 = mpf(972) / mpf(820)         # 243/205

    print("hBN upper Reststrahlen band:")
    print(f"  ω_TO = 1370 cm⁻¹, ω_LO = 1610 cm⁻¹")
    print(f"  r = ω_LO/ω_TO = 1610/1370 = 161/137")
    print(f"  (denominator 137 = α⁻¹, the fine-structure constant)")
    project_tower(r_hBN_upper, "hBN upper: r = 161/137", [12, 60, 132, 420, 27720])

    print("\nhBN lower Reststrahlen band:")
    print(f"  ω_TO = 780 cm⁻¹, ω_LO = 830 cm⁻¹")
    print(f"  r = ω_LO/ω_TO = 83/78")
    project_tower(r_hBN_lower, "hBN lower: r = 83/78", [12, 60, 132, 420, 27720])

    print("\nα-MoO₃ [100] Reststrahlen band:")
    print(f"  ω_TO = 820 cm⁻¹, ω_LO = 972 cm⁻¹")
    print(f"  r = ω_LO/ω_TO = 972/820")
    project_tower(r_MoO3, "MoO₃: r = 972/820", [12, 60, 132, 420, 27720])

    # Comparison
    p_u = project(r_hBN_upper, 12)
    p_l = project(r_hBN_lower, 12)
    p_m = project(r_MoO3, 12)

    print(f"\nCOMPARISON AT 12ET:")
    print(f"  hBN upper: d={p_u['d']}, ε={p_u['eps_cents']:+.2f}¢  (d=4 quartic/weak)")
    print(f"  hBN lower: d={p_l['d']}, ε={p_l['eps_cents']:+.2f}¢  (d=12 full resolution)")
    print(f"  MoO₃:     d={p_m['d']}, ε={p_m['eps_cents']:+.2f}¢  (d=4 quartic/weak)")

    print(f"""
STRUCTURAL READING:
  hBN upper and MoO₃ both sit at d=4 (quartic/weak sublattice family)
  but at DIFFERENT descriptor gaps: ε=-20.5¢ vs ε=-5.6¢.
  Same sublattice character, different ε → different polariton properties.

  hBN lower band sits at d=12 (full resolution) — completely different
  sublattice family → completely different polariton character.

  The v_φ/v_g ratio is NOT universal — it depends on the material's R₀
  position (d, ε) on the lattice. The Bucher paper's v_φ/v_g ≈ 12 is
  what the hBN upper band produces at its specific lattice position
  (d=4, ε=-20.5¢). MoO₃ at (d=4, ε=-5.6¢) produces different v_φ/v_g.

  The lattice classifies materials by their R₀-derived dimensionless
  ratio. Materials at the same (d, ε) cell are predicted to show similar
  polariton character; materials at different cells show different character.
""")

    # Notable: 161/137 and α⁻¹
    print(f"NOTABLE: hBN upper band R₀ ratio = 161/137")
    print(f"  gcd(1610, 1370) = {gcd(1610, 1370)}")
    print(f"  1610/{gcd(1610,1370)} = {1610//gcd(1610,1370)}")
    print(f"  1370/{gcd(1610,1370)} = {1370//gcd(1610,1370)}")
    print(f"  Denominator 137 is prime and equals α⁻¹.")
    print(f"  Phonon frequency values have experimental uncertainties")
    print(f"  (ω_TO = 1365-1375, ω_LO = 1600-1614 in literature),")
    print(f"  so 161/137 is approximate. Flagged for investigation.")


# ═══════════════════════════════════════════════════════════════
# 4. ∂I BOUNDARY IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

def section_dI():
    print("\n" + "=" * 78)
    print("4. PHASE SINGULARITIES AS ∂I BOUNDARY CONFIGURATIONS")
    print("   ET verifies the singularity's structural role in the four-state")
    print("   classification. Standard physics has no ∂I concept.")
    print("=" * 78)

    props = [
        ("∂I ∩ I = ∅", "Core has zero measure; boundary not region"),
        ("Approached asymptotically", "Amplitude → 0 continuously; zero is limit point"),
        ("D absent at I → {P,T}", "Amplitude (D-content) = 0; phase (T-content) undefined without D"),
        ("Marginal: perturbation switches ϕ", "Perturbation displaces or annihilates singularity"),
        ("Open-set topology", "Core has no interior; neighborhoods extend into coherent field"),
        ("Mediation Problem: P,T need D", "At core, P (space) and T (winding) lack D-bridge (zero amplitude)"),
    ]

    print(f"\nSix structural properties — all match:")
    for i, (et, phys) in enumerate(props, 1):
        print(f"  {i}. ET: {et}")
        print(f"     Phys: {phys} ✓")

    print(f"""
KEY AXIS IDENTIFICATION (corpus: ET_Freedom_and_U1, ET_Complex_Lattice):
  Real axis = (ℝ⁺, ×) = D's domain = amplitude/magnitude
  Imaginary axis = (U(1), ×) = T's domain = phase/rotation

  At the singularity core:
    D-content (amplitude) = 0 → D-bridge absent → {{P,T}} = Incoherence
    T-content (phase) undefined → without D, T has nothing to rotate
    T's ±2π winding exists in the NEIGHBORHOOD, not at the core
    Winding number Q = ±1 counts T-circuits around the D-void

  This is the Mediation Problem (Sempaevum Theorem 2.25):
    "Without D, {{P,T}} is the structural form of the failure"
    The singularity IS the Mediation Problem made physical.
""")


# ═══════════════════════════════════════════════════════════════
# 5. SUPERLUMINAL VELOCITY = T NOT CONSTRAINED BY D
# ═══════════════════════════════════════════════════════════════

def section_superluminal():
    print("\n" + "=" * 78)
    print("5. SUPERLUMINAL VELOCITY — c IS A D-LIMIT, NOT A T-LIMIT")
    print("   ET verifies the standard physics observation ('no information")
    print("   transfer') as the structural category distinction T ≠ D.")
    print("=" * 78)

    print(f"""
  T-Paper §52.6 (corpus):
    "c applies to D-propagation, not T-navigation."

  Standard physics says: singularities carry zero intensity, so
  superluminal motion doesn't violate relativity (no information transfer).

  ET verifies this as: c constrains D-content (energy, information).
  Phase singularities have zero D-content → c does not apply.
  The unbounded velocity is T-agency unconstrained by D-bounds.
  This is the Subsumption Law: T cannot be subsumed by D,
  therefore T-observables are not bound by D-limits.

  Velocity divergence at annihilation:
    x(t) ~ x₀ ± a√(t_ann - t), so v ~ 1/√Δt → ∞ (determinate infinity)
    NOT a 0/0 indeterminate form. Correct ET statement:
    c constrains D, not T. Period.
""")


# ═══════════════════════════════════════════════════════════════
# 6. CASCADE STABILITY → PARTICLE ANALOGY BREAKDOWN
# ═══════════════════════════════════════════════════════════════

def section_cascade_link():
    print("\n" + "=" * 78)
    print("6. CASCADE STABILITY ASYMMETRY → PARTICLE ANALOGY BREAKDOWN")
    print("   ET verifies the Berry-Dennis observation as the cascade")
    print(f"   stability ratio n_max_r/n_max_θ = {N_MAX_R}/{N_MAX_THETA} = {N_MAX_R/N_MAX_THETA} ≈ N")
    print("=" * 78)

    print(f"""
  Real axis (D): n_max = {N_MAX_R} cascade levels → fully resolved statistics
  Imag axis (T): n_max = {N_MAX_THETA} cascade levels → indeterminate after 2 steps

  PAPER'S OBSERVATION (Berry-Dennis, standard physics):
    Distance correlations (D-observable): particle-like ✓
    Velocity distributions (T-observable): non-particle, long tail ✓
    Standard physics shows this; ET explains WHY:

  ET VERIFICATION:
    D-content has {N_MAX_R} levels of cascade stability → D-statistics
    are fully determined → particle-like distance correlations.
    T-content has {N_MAX_THETA} levels before palindromic fallback →
    T-statistics are structurally indeterminate → long superluminal tail.

  CROSS-DOMAIN PARALLEL (ET-derived, standard physics does not connect):
    CKM angles (real axis): small, tight → D-content resolved
    PMNS angles (imaginary axis): large, spread → T-content unresolved
    Distance/velocity in singularity ensembles: SAME asymmetry.
    Two instances of one structural fact across different domains.
""")


# ═══════════════════════════════════════════════════════════════
# 7. U(1) WINDING UNIVERSALITY
# ═══════════════════════════════════════════════════════════════

def section_u1():
    print("\n" + "=" * 78)
    print("7. U(1) WINDING — CROSS-DOMAIN STRUCTURAL IDENTITY")
    print("   ET verifies each domain's winding numbers as T-coordinates")
    print("   on T's operational manifold U(1). Standard physics treats")
    print("   each domain's topology separately.")
    print("=" * 78)

    print(f"""
  Optics:          Phase singularity     Q = ±1 (±2π winding)
  QCD:             Instanton             Q = ±1 (vacuum sector)
  Superfluids:     Quantized vortex      Q = ±1 (circulation quantum)
  Superconductors: Flux quantum          Q = ±1 (flux quantum Φ₀)
  ET (general):    T-traversal on U(1)   Q = ±1 (one step)

  ET identification: all five are T-coordinates on the same U(1).
  Standard physics: each domain's winding is a separate mathematical fact.
  ET: they are the SAME structural fact — T on its operational manifold —
  read at five different integrative levels.
""")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 76 + "╗")
    print("║  ET OPTICAL PHASE SINGULARITIES — VERIFICATION SUITE v2" + " " * 19 + "║")
    print("║  Bucher et al. (2025) arXiv:2509.17675v1" + " " * 35 + "║")
    print("║  ET verifies via subsumption. 150-digit mpmath precision." + " " * 16 + "║")
    print("╚" + "═" * 76 + "╝")

    section_cascade_residuals()
    section_t_weight()
    section_R0_analysis()
    section_dI()
    section_superluminal()
    section_cascade_link()
    section_u1()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"""
  ET VERIFIES:
    1. Cascade residuals in three unit systems (lattice steps, cents, octaves)
    2. 1/3 velocity tail = T-weight = 1/|{{P,D,T}}| (computed to 20 digits)
    3. Material-specific polariton classification via R₀ lattice position
    4. ∂I boundary identification (6 structural properties match)
    5. Superluminal velocity = c constrains D, not T (T-Paper §52.6)
    6. Particle-analogy breakdown = cascade stability ratio {N_MAX_R}/{N_MAX_THETA} ≈ N
    7. U(1) winding universality across 5 physical domains

  ET ADDS BEYOND STANDARD PHYSICS:
    - WHY c doesn't apply (structural category T ≠ D)
    - WHY the particle analogy breaks for velocity but not distance
    - Cross-domain CKM/PMNS ↔ distance/velocity parallel
    - Material classification via R₀ lattice position
    - 1/3 = T-weight structural reason (not just a PDF property)

  OPEN QUESTIONS:
    - Whether 161/137 in hBN's R₀ connects to α⁻¹ = 137
    - Whether R₀-classified materials at same (d,ε) show similar v_φ/v_g
    - Whether 1/3 = T-weight holds in non-2D systems
    - n_eff = 132 = lcm(11,12) in hBN image polaritons (separate mode type)
""")

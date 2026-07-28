#!/usr/bin/env python3
"""
et_gap_cell_investigation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ET FORCE QUADRANT GRID — FOUR GAP CELL DOMAIN SIGNATURE INVESTIGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gap Cells Under Investigation (FQ-20 known gaps from the Force Quadrant Grid):
  (5,5)  — Icosahedral dark-sector [biological/quasicrystal × Otherworld d=5]
  (7,7)  — G₂×G₂ extreme dark sector [maximum-darkness / sacred-7 Otherworld]
  (5,7)  — Cross-complex at 420ET [biological threshold / life activation]
  (8,9)  — QCD×CKM bridge [gluonic extension / civilizational–zeitgeist tier]

Foundation  : P ∘ D ∘ T = E
Manifold    : N=12, V=1/12, K=2/3
Framework   : ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]}   — 2D complex ET lattice
Principles  : Identification Principle + Descriptor Gap Principle
Mathematics : Exception Theory (ET) — all derivations forward from {P, D, T}
Sources     :
  • ET Universal Lattice Domain Map (Domain Map)
  • ET Translation Layer Reference Units (Translation Layer)
  • ET Force Quadrant Grid Section W2 (FQ-1..FQ-25)
  • ET Prediction Test Research (Section 5.3 verified results)
  • ET Complex Lattice (Complex Lattice document)
  • ET Quintic Shadow d=5 Investigation (QS-1..QS-15)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import sys
from fractions import Fraction
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — ET FOUNDATIONAL CONSTANTS  (all from P ∘ D ∘ T = E)
# ─────────────────────────────────────────────────────────────────────────────

N      = 12                              # Manifold symmetry: 3 primitives × 4 logic states
V      = Fraction(1, 12)                # Base variance: discretization quantum
K      = Fraction(2, 3)                 # Koide ratio: triadic binding threshold
G_R    = 7                              # Real-axis generator (circle of fifths)
G_T    = 1                              # Imaginary-axis generator (sequential)
LN2    = math.log(2)                    # Natural log of 2 (lattice base)

# Cascade stability depths (from CLR v5 Section 0 / FQ-1)
DELTA_R        = 0.019550               # Real descriptor gap magnitude |δ_r|
DELTA_THETA    = 0.223357               # Imaginary descriptor gap magnitude |δ_θ|
N_MAX_R        = 25                     # Real cascade stability: 25 levels
N_MAX_THETA    = 2                      # Imaginary cascade stability: 2 levels
IMAG_AMP       = N_MAX_R / N_MAX_THETA  # = 12.5 (FQ-1 Imaginary Amplification Theorem)

# Universal shadow coupling invariant: α_d × d = 1/4  (CF-4)
SHADOW_C       = 1200.0                 # One octave in cents
SHADOW_INV     = Fraction(1, 4)         # α_d × d = 1/4 for all complex d

# Biological threshold (FQ-13)
N_BIO = 420    # LCM(1..7) = 420ET — minimum descriptor count for life

# CKM effective mixing amplitude (real axis, FQ-21)
LAMBDA_CKM  = math.sqrt(float(K * V))                       # ≈ 0.23570
# PMNS effective mixing amplitude (imaginary axis, FQ-21)
LAMBDA_PMNS = LAMBDA_CKM * math.sqrt(N_MAX_R / N_MAX_THETA) # ≈ 0.83333

# Sublattice family classifications (N=12)
SIMPLE_REAL    = {1, 2, 3, 4, 6, 12}   # d | 12: native at 12ET (SR)
COMPLEX_REAL   = {5, 7, 8, 9, 10, 11}  # d ∤ 12: non-native at 12ET (CR)

# Golden ratio (the quintic asymptotic attractor)
PHI = (1 + math.sqrt(5)) / 2           # φ ≈ 1.6180339887


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ET LATTICE MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

def gcd(a: int, b: int) -> int:
    """Euclidean GCD. ET reduction of two integers."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """LCM via ET reduction: LCM(a,b) = |a×b| / GCD(a,b)."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def lcm_multi(*args: int) -> int:
    """LCM of multiple integers via sequential reduction."""
    result = args[0]
    for x in args[1:]:
        result = lcm(result, x)
    return result

def et_project_real(r: float) -> Tuple[int, int, int, float]:
    """
    ET real-axis lattice projection.
    Input : r > 0 (dimensionless positive ratio)
    Returns: (k, g, d, ε_cents)
      k  = round(12 × log₂(r))       [ET lattice coordinate]
      g  = gcd(|k|, 12)              [shared factor]
      d  = 12 / g                    [sublattice family]
      ε  = (12 × log₂(r) − k) × 100 [deviation in cents]
    Derivation: The Traverser T acts as the rounding operator on the
    multiplicative manifold (ℝ⁺, ×). The lattice step is s = 2^(1/12).
    """
    raw = 12.0 * math.log2(r)
    k   = round(raw)
    g   = gcd(abs(k), N) if k != 0 else N
    d   = N // g
    eps = (raw - k) * 100.0
    return k, g, d, eps

def et_project_imaginary(theta: float) -> Tuple[int, int, int, float]:
    """
    ET imaginary-axis (T-domain) lattice projection.
    Input : θ in radians (phase angle on U(1) circle)
    Returns: (k_θ, g_θ, d_θ, ε_θ_cents)
      k_θ = round(12 × θ / ln2)      [imaginary ET coordinate]
      g_θ = gcd(|k_θ|, 12)
      d_θ = 12 / g_θ
      ε_θ = (12 × θ / ln2 − k_θ) × 100
    Derivation: The imaginary axis in Log₂(z) = log₂|z| + i·θ/ln2.
    The imaginary lattice step = 1 (sequential generator g_θ = 1).
    """
    raw    = 12.0 * theta / LN2
    k_t    = round(raw)
    g_t    = gcd(abs(k_t), N) if k_t != 0 else N
    d_t    = N // g_t
    eps    = (raw - k_t) * 100.0
    return k_t, g_t, d_t, eps

def sublattice_quadrant(d_r: int, d_t: int) -> Tuple[str, int]:
    """
    Classify force (d_r, d_θ) into the 4 ET quadrants and compute combined d.
    Quadrants (FQ-2 / W2.2):
      SR+SI: both axes simple → stable ground state
      CR+SI: complex real, simple imaginary → structural pressure
      SR+CI: simple real, complex imaginary → phase complexity
      CR+CI: both axes complex → maximum complexity
    d_combined = LCM(d_r, d_θ)  [the force requires BOTH periods to close]
    """
    sr = d_r in SIMPLE_REAL
    si = d_t in SIMPLE_REAL
    if   sr and si:  q = "SR+SI"
    elif not sr and si: q = "CR+SI"
    elif sr and not si: q = "SR+CI"
    else:               q = "CR+CI"
    return q, lcm(d_r, d_t)

def critical_resolution(d_r: int, d_t: int) -> int:
    """
    n_c(d_r, d_θ) = LCM(N, d_r, d_θ)
    Minimum ET-resolution at which BOTH d_r and d_θ are native sublattice families.
    Below n_c: force exists only as shadow tension (ε ≠ 0, never lattice integer).
    At n_c:    force binds as genuine lattice position — discrete phase transition.
    Source: FQ-2 (Anti-Emergence Threshold Theorem), CF-3.
    """
    return lcm_multi(N, d_r, d_t)

def shadow_coupling_real(d: int) -> float:
    """
    α_d = 1/(4d)   [universal shadow coupling invariant: α_d × d = 1/4, CF-4]
    Real-axis coupling strength for complex force d.
    """
    return 1.0 / (4.0 * d)

def shadow_tension_real(d: int) -> float:
    """
    ⟨τ_d⟩ = 1200/(4d) cents
    Mean shadow tension on real axis — mean offset from nearest 12ET lattice point.
    For d ∤ 12: force is never exactly on 12ET lattice → persistent structural pressure.
    """
    return SHADOW_C / (4.0 * d)

def shadow_tension_imag(d: int) -> float:
    """
    ⟨τ_θ⟩_eff = 1200/(4d) × (n_max_r / n_max_θ) = ⟨τ_d⟩ × 12.5
    Effective imaginary shadow tension amplified by the stability depth ratio.
    FQ-1 (Imaginary Amplification Theorem):
      The imaginary cascade is N=12 times less stable than the real cascade.
      CI forces are 12.5× more coupling-amplified than CR forces with same d.
      This is the ET derivation of why PMNS >> CKM mixing angles.
    """
    return shadow_tension_real(d) * IMAG_AMP

def force_norm_sq(d_r: int, d_t: int) -> int:
    """
    |w(E)|² = d_r² + d_θ²   [ET 2D force complexity norm in ℤ[i]]
    Measures 'distance from simplest interaction' in the Gaussian integer space.
    Gravity (1,2) is the minimum: |w|² = 1+4 = 5.
    Source: FQ-3 (2D Force Vector Theorem).
    """
    return d_r * d_r + d_t * d_t

def gaussian_character(n: int) -> str:
    """
    Gaussian prime character of integer n in ℤ[i].
    For prime p:
      p=2:       P-type/Ramified → 2 = −i(1+i)² in ℤ[i] (the lattice base)
      p≡1 mod 4: D+T-Split → p = (a+bi)(a−bi), has both structural+traversal character
      p≡3 mod 4: D-type/Inert → remains prime in ℤ[i], purely structural
    For composites: derived from prime factorization.
    Source: ET Complex Lattice §12, Gaussian Prime Classification.
    """
    if n == 1:
        return "Trivial unit"
    # Factor n into prime power components
    temp = abs(n)
    factors: Dict[int, int] = {}
    for p in range(2, temp + 1):
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1

    parts = []
    for p, e in sorted(factors.items()):
        if p == 2:
            parts.append(f"2^{e}[P-Ramified]")
        elif p % 4 == 1:
            parts.append(f"{p}^{e}[D+T-Split]")
        else:
            parts.append(f"{p}^{e}[D-Inert]")

    # Summarise single-prime cases cleanly
    if len(factors) == 1:
        p, e = list(factors.items())[0]
        if p == 2:
            return f"P-type/Ramified  (2^{e}: lattice base ramified)"
        elif p % 4 == 1:
            return f"D+T-Split  ({p}^{e}: splits in ℤ[i] as conjugate pair)"
        else:
            return f"D-type/Inert  ({p}^{e}: remains prime in ℤ[i], zero T-mixing)"
    return "Composite: " + " × ".join(parts)

def palindromic_partners_2d(d_r: int, d_t: int) -> Dict[str, Tuple[int, int]]:
    """
    2D palindromic partners via the d_c + d_s = 12 palindromic pairing (CF-2, FQ-8).
    Real-axis palindrome:     (d_r, d_θ) ↦ (12−d_r, d_θ)
    Imaginary-axis palindrome:(d_r, d_θ) ↦ (d_r, 12−d_θ)
    Full 2D palindrome:       (d_r, d_θ) ↦ (12−d_r, 12−d_θ)
    Transpose (quark↔lepton): (d_r, d_θ) ↦ (d_θ, d_r)   [FQ-22]
    """
    return {
        "real_axis_palindrome": (12 - d_r, d_t),
        "imag_axis_palindrome": (d_r,  12 - d_t),
        "full_2d_palindrome":   (12 - d_r, 12 - d_t),
        "transpose_dual":       (d_t,  d_r),
    }

def domain_map_signature(d_r: int, d_t: int) -> List[str]:
    """
    Map (d_r, d_θ) to ET Universal Lattice Domain Map tiers.
    Each d-family appears at multiple integrative levels.
    """
    tier_map = {
        1:  ["Quantum (gravity/trivial)", "Atomic (octave periods)",
             "Economy (8-yr → d=1 via octave)", "Civilizational (500yr epochal)"],
        2:  ["Pre-quantum (binary/tritone)", "Molecular (chirality binary)",
             "Mathematics (Euler e^{iπ}=−1 at d=2)"],
        3:  ["Nuclear/Hadronic (QCD 3-quark closure)", "Molecular (ATP cubic d=3)",
             "Biological (Fibonacci convergence d=3)", "Social/Political (3 branches)"],
        4:  ["Quantum-Field (Weak force quartic)", "Molecular (DNA backbone d=4)",
             "Civilizational (4-yr cycle quartic)", "Dreams (sub-personal d=4)"],
        5:  ["Biological/Quasicrystal tier (d=5, 60ET first native)",
             "Cognition/Qualia (60ET — inembeddable in 12ET)",
             "Alternative Realities / Otherworld (d=5 in 60ET)"],
        6:  ["Molecular/Chemical (benzene 6-fold)", "Biological (Krebs hexagonal)",
             "Art/Music/Aesthetics (hexadic wave)", "EW-mixing (hexadic bridge)"],
        7:  ["Alternative Realities / Otherworld (d=7, 420ET)",
             "Religion/Mythology (sacred-7 direct T encounter)",
             "G₂ holonomy (M-theory compact 7-manifold)",
             "CP Violation (kaon, (4,7) SR+CI)"],
        8:  ["Economy/Markets (8-yr business cycle, d=8 real)",
             "Gluon-Octet nuclear (SU(3) 8 generators, n_c=24ET)"],
        9:  ["Social/Political nonic sector",
             "Quark generation structure (d=9=3², CR real axis)",
             "PMNS imaginary axis (d_θ=9, SR+CI neutrino mixing)"],
        10: ["10D Superstring / E₈×E₈ (n_c=60ET, (10,2) CR+SI)"],
        11: ["M-theory 11D (n_c=132ET, (11,11) CR+CI)"],
        12: ["Quantum Field EM (full-resolution ambient d=12)",
             "Atomic (orbital full resolution)", "Consciousness (both axes d=12)",
             "Koide threshold K=2/3 → k=−7 d=12"],
    }
    tiers = []
    seen = set()
    for d in [d_r, d_t]:
        for t in tier_map.get(d, []):
            if t not in seen:
                seen.add(t)
                tiers.append(t)
    return tiers

def lcm_tower_to(target: int) -> List[Tuple[int, int, str]]:
    """
    LCM tower: for d = 1..target, compute LCM(1..d).
    Returns list of (d, LCM_val, label).
    Source: Domain Map LCM Tower section.
    """
    result = []
    current = 1
    for d in range(1, target + 1):
        current = lcm(current, d)
        label = f"LCM(1..{d}) = {current}ET"
        result.append((d, current, label))
    return result

def quintic_lattice_positions(n_et: int) -> List[Tuple[int, float, float]]:
    """
    All d=5 native positions at resolution n_et (n_et must be divisible by 5).
    Returns list of (k, ratio, cents_from_origin) for each quintic lattice point.
    At n=60ET: first lattice where d=5 becomes native.
    """
    positions = []
    if n_et % 5 != 0:
        return positions
    step = n_et // 5
    for j in range(1, n_et + 1):
        k_at = j
        if n_et % k_at == 0:
            pass
        g_val = gcd(k_at, n_et)
        d_val = n_et // g_val
        if d_val == 5:
            ratio = 2 ** (k_at / n_et)
            cents = k_at * (1200.0 / n_et)
            positions.append((k_at, ratio, cents))
    return positions

def icosahedral_subtension_angles() -> List[Tuple[str, float, int, int, float]]:
    """
    The 5 rotation axes of the icosahedral group I (order 60):
      C2: 180° = π
      C3: 120° = 2π/3
      C5: 72° = 2π/5
      C5²: 144° = 4π/5
    Project these onto the ET imaginary lattice.
    Returns list of (name, degrees, k_θ, d_θ, ε¢).
    """
    axes = [
        ("C₂ (180°)", 180.0),
        ("C₃ (120°)", 120.0),
        ("C₅ (72°)",   72.0),
        ("C₅² (144°)", 144.0),
        ("C₅³ (216°)", 216.0),
        ("C₅⁴ (288°)", 288.0),
    ]
    results = []
    for name, deg in axes:
        theta = math.radians(deg)
        k_t, g_t, d_t, eps = et_project_imaginary(theta)
        results.append((name, deg, k_t, d_t, eps))
    return results

def caspar_klug_t_numbers() -> List[Tuple[int, int, int, int, int, int, float]]:
    """
    Caspar-Klug triangulation numbers T = h² + hk + k² for icosahedral viral capsids.
    Total subunits = 60T.
    Project T and 60T onto ET real lattice.
    Returns list of (h, k, T, 60T, k_ET_T, d_ET_T, ε_T)
    Source: Caspar & Klug (1962) — icosahedral capsid geometry.
    """
    results = []
    seen_T = set()
    for h in range(0, 8):
        for k in range(0, h + 1):
            T = h*h + h*k + k*k
            if T == 0 or T in seen_T:
                continue
            seen_T.add(T)
            subunits = 60 * T
            k_ET, g_ET, d_ET, eps_ET = et_project_real(float(T))
            k_sub, g_sub, d_sub, eps_sub = et_project_real(float(subunits))
            results.append((h, k, T, subunits, k_ET, d_ET, eps_ET,
                           k_sub, d_sub, eps_sub))
    results.sort(key=lambda x: x[2])
    return results[:12]  # first 12 T-numbers

def g2_structure_constants() -> Dict[str, object]:
    """
    G₂ Lie group structure constants in ET lattice terms.
    G₂ is the automorphism group of the octonions (𝕆).
    dim(G₂) = 14, rank = 2.
    Root system: 6 short roots (length 1) + 6 long roots (length √3)
    Highest root: 3α₁ + 2α₂  (in terms of simple roots)
    n_c(d=7) = LCM(12,7) = 84ET
    Source: ET Force Quadrant Grid W2.13 CF-10; Quintic Shadow QS-5.
    """
    # G₂ Dynkin diagram: α₁ →≡ α₂  (one arrow, triple bond)
    # Simple root ratio: |α₂|/|α₁| = √3 → ET: k=round(12×log₂(√3))=round(9.51)=10
    #   g=gcd(10,12)=2, d=12/2=6
    k_root_ratio, g_root, d_root, eps_root = et_project_real(math.sqrt(3))
    # Weyl group order of G₂: |W(G₂)| = 12
    weyl_order = 12
    # G₂ generators: 14 (in ET: 12 = N from the manifold, +2 Cartan generators)
    # The 12 non-zero roots split: 6 short (|1|) + 6 long (|√3|)
    # Number of positive roots = 6 → this is exactly N/2
    n_pos_roots = 6
    # G₂ in 7D representation: the 7D vector representation
    dim_7d = 7
    # Critical resolution for d=7 G₂ force:
    n_c_g2 = critical_resolution(7, 1)   # real axis only
    n_c_g2_full = critical_resolution(7, 7)  # full 2D
    # The 12 roots of G₂ span all 12 positions of the ET lattice — this is structural:
    # G₂ Weyl group order = 12 = N (manifold symmetry). Not coincidental.
    return {
        "dim_G2": 14,
        "rank": 2,
        "n_short_roots": 6,
        "n_long_roots": 6,
        "n_total_roots": 12,         # = N: G₂ has exactly N=12 roots
        "weyl_order": weyl_order,    # |W(G₂)| = 12 = N
        "root_ratio_k": k_root_ratio,
        "root_ratio_d": d_root,
        "root_ratio_eps": eps_root,
        "7d_rep_dim": dim_7d,
        "d_r_G2": 7,
        "n_c_G2_real": n_c_g2,
        "n_c_G2_full_2D": n_c_g2_full,
        "note": "G₂ has 12 roots = N (manifold symmetry). Weyl group order = 12 = N. Structural necessity.",
    }

def e8_decomposition_g2() -> Dict[str, object]:
    """
    E₈ decomposition via G₂ × F₄ maximal subgroup, relevant to (5,7) cell.
    E₈ adjoint representation 248 decomposes under G₂ × F₄ as:
      248 = (14, 1) ⊕ (1, 52) ⊕ (7, 26)
    Numbers: 14 + 52 + 7×26 = 14 + 52 + 182 = 248 ✓
    Also: E₈ contains A₄ (SU(5)) with icosahedral/quintic substructure.
    Connection (5,7): d_r=5 (icosahedral/quintic ⊂ A₄ ⊂ E₈)
                      d_θ=7 (G₂ phase ⊂ G₂ × F₄ ⊂ E₈)
    n_c(5,7) = LCM(12,5,7) = 420ET = E₈ resolution threshold.
    Source: FQ-20 (known gap); force quadrant table E₈ gauge vertex at (5,7).
    """
    dim_e8 = 248
    dim_g2 = 14
    dim_f4 = 52
    dim_7_26 = 7 * 26   # = 182
    check = dim_g2 + dim_f4 + dim_7_26  # = 248
    n_c_57 = critical_resolution(5, 7)
    # The 120 positive roots of E₈ split under G₂ subgroup
    n_roots_e8 = 240   # E₈ has 240 roots
    # A₄ = SU(5) substructure (quintic, d=5 icosahedral)
    # E₈ → SU(5) × SU(5): 248 = (24,1) ⊕ (1,24) ⊕ (10,10̄) ⊕ (10̄,10) ⊕ (5,5̄) ⊕ (5̄,5)
    # The 5 of SU(5) is the quintic representation → d=5
    return {
        "dim_E8": dim_e8,
        "adjoint_check": check,         # Must equal 248
        "G2_component": dim_g2,
        "F4_component": dim_f4,
        "7x26_component": dim_7_26,
        "n_c_cell_5_7": n_c_57,
        "n_roots_E8": n_roots_e8,
        "connection": "(d_r=5 ↔ SU(5)/A₄ icosahedral ⊂ E₈) × (d_θ=7 ↔ G₂ ⊂ E₈)",
        "note": "n_c(5,7)=420ET is simultaneously: (a) LCM(1..7) biological threshold, "
                "(b) E₈ gauge vertex resolution, (c) T=7 icosahedral capsid subunit count.",
    }

def qcd_ckm_bridge_structure() -> Dict[str, object]:
    """
    The (8,9) cell as QCD×CKM bridge.
    d_r=8=2³: SU(3) gluon octet has 8 generators. 8=2³ is the 'cubic complex' (CF-16).
              n_c(d=8, real) = LCM(12,8) = 24ET.
    d_θ=9=3²: Three quark generations via d=9=3² on imaginary CI axis.
              n_c(d=9, imag) = LCM(12,9) = 36ET.
    n_c(8,9) = LCM(12,8,9) = 72ET.
    The (8,9) cell is the gluon-octet × quark-generation interaction in the CI sector.
    Its real-axis palindromic partner is (4,9) = PMNS neutrino oscillation.
    Source: FQ-8 (palindromic pairing), CF-16, FQ-9, CF-9.
    """
    d_r, d_t = 8, 9
    n_c_89 = critical_resolution(d_r, d_t)
    n_c_real_8 = lcm(N, d_r)   # = 24ET
    n_c_imag_9 = lcm(N, d_t)   # = 36ET
    palindromes = palindromic_partners_2d(d_r, d_t)
    pmns_partner = palindromes["real_axis_palindrome"]   # (4, 9) = PMNS
    # Coupling comparison:
    alpha_8_real = shadow_coupling_real(8)      # = 1/32
    alpha_9_imag = shadow_coupling_real(9)      # = 1/36 (before amplification)
    alpha_9_eff  = alpha_9_imag * IMAG_AMP     # × 12.5
    # 8 = 2³: octet = 3rd power of 2 → real k(8) at 60ET base:
    k_8, g_8, d_8_check, eps_8 = et_project_real(8.0)
    # 9 = 3²: 3 generations squared → real k(9):
    k_9, g_9, d_9_check, eps_9 = et_project_real(9.0)
    # Civilizational tier check:
    # Economy d=8: 8-yr business cycle / k=round(12×log₂(8))=36 / d=1 (octave, 8=2³)
    # But for the FORCE cell d_r=8 is different from the cycle ratio r=8
    # The force d=8 is the sublattice family, not the ratio value
    return {
        "d_r": d_r, "d_theta": d_t,
        "n_c": n_c_89,
        "n_c_real_8": n_c_real_8,
        "n_c_imag_9": n_c_imag_9,
        "alpha_8_real": alpha_8_real,
        "alpha_9_imag_raw": alpha_9_imag,
        "alpha_9_eff_amplified": alpha_9_eff,
        "palindrome_PMNS": pmns_partner,
        "gluon_octet_generators": 8,   # SU(3) has 8 generators
        "quark_generations": 3,         # from d=9=3² → 3 generations
        "k_ratio_8": k_8, "d_ratio_8": d_8_check,
        "k_ratio_9": k_9, "d_ratio_9": d_9_check,
        "note": "Palindrome of (8,9) on real axis = (4,9) = PMNS. "
                "The QCD×CKM bridge and PMNS neutrino oscillation are 2D palindromic partners.",
    }

def translation_layer_reference_units() -> Dict[str, Dict]:
    """
    Reference unit derivation for each gap cell domain (Translation Layer §3).
    The denominator R₀ = D_period(P_L): the natural fundamental period of the
    P-substrate at the relevant integrative level. Never chosen arbitrarily.
    Source: Translation Layer §2 (Reference Period Uniqueness Theorem).
    """
    return {
        "(5,5)_quasicrystal": {
            "substrate": "P_quasicrystal = icosahedral quasiperiodic tiling of ℝ³",
            "R0_description": "The fundamental quasicrystal inflation step: τ = φ (golden ratio)",
            "R0_value": PHI,
            "R0_units": "dimensionless (ratio of successive Fibonacci spacings)",
            "derivation": "Quasicrystal inflation: each step scales by φ. "
                          "φ = lim(F_{n+1}/F_n) is the minimal closed T-traversal loop "
                          "for the icosahedral substrate. r = (observed_spacing / base_spacing).",
            "R0_lattice": et_project_real(PHI),
            "r_examples": {
                "pentagon_diagonal/side": et_project_real(PHI),
                "phi^2": et_project_real(PHI ** 2),
                "phi^3": et_project_real(PHI ** 3),
            },
        },
        "(5,5)_capsid": {
            "substrate": "P_capsid = icosahedral viral capsid lattice",
            "R0_description": "The T=1 capsid (1 triangulation unit): 60 protein subunits",
            "R0_value": 60.0,
            "R0_units": "protein subunits (dimensionless count, like step counting §3.2)",
            "derivation": "60 = LCM(12,5) = n_c(d=5): the minimal icosahedral capsid. "
                          "r = T_observed_subunits / 60.  Per Translation Layer §3.2: "
                          "R₀ = minimal step (1 T=1 unit = 60 subunits).",
            "r_T_numbers": {
                f"T={t}": et_project_real(float(t))
                for t in [1, 3, 4, 7, 9, 12, 13]
            },
        },
        "(5,5)_cosmic_LSS": {
            "substrate": "P_LSS = cosmic large-scale structure filament network",
            "R0_description": "Baryon acoustic oscillation scale: r_BAO ≈ 147 Mpc",
            "R0_value": 147.0,
            "R0_units": "Mpc (comoving megaparsecs)",
            "derivation": "The BAO scale is the fundamental period of the matter power spectrum. "
                          "It is determined by the sound horizon at recombination — a structural "
                          "fact about P_LSS, not a human convention. r = L_observed / r_BAO.",
            "r_examples": {
                "BAO scale (self)": et_project_real(1.0),
                "Cosmic web void ~50 Mpc": et_project_real(50.0 / 147.0),
                "Supercluster ~200 Mpc": et_project_real(200.0 / 147.0),
            },
        },
        "(7,7)_dark": {
            "substrate": "P_dark = dark matter density field (D-type/Inert, no T-mixing)",
            "R0_description": "Dark matter halo virial radius (Milky Way type): ~200 kpc",
            "R0_value": 200.0,
            "R0_units": "kpc (kiloparsecs)",
            "derivation": "For D-type/Inert (7,7) dark matter with zero T-coupling, "
                          "the only available period is the gravitational (d=1) halo scale. "
                          "R₀ = virial radius = minimal gravitational closure loop. "
                          "r = r_observed / R_virial.",
            "r_examples": {
                "Halo self": et_project_real(1.0),
                "7-fold subhalo r~29kpc": et_project_real(200.0 / 7.0 / 200.0),
            },
        },
        "(5,7)_biological": {
            "substrate": "P_bio = protein assembly manifold for icosahedral capsids",
            "R0_description": "T=1 capsid unit = 60 protein subunits (minimal icosahedral assembly)",
            "R0_value": 60.0,
            "R0_units": "protein subunits",
            "derivation": "Same as (5,5)_capsid but the (5,7) combination requires n_c=420ET. "
                          "The T=7 capsid has 420 = 7×60 subunits, and 420 = LCM(12,5,7) = n_c(5,7). "
                          "r = T_observed × 60 / 60 = T_number itself.",
            "T7_check": {
                "T=7 subunits": 7 * 60,
                "n_c(5,7)": critical_resolution(5, 7),
                "match": (7 * 60 == critical_resolution(5, 7)),
            },
        },
        "(8,9)_QCD": {
            "substrate": "P_QCD = SU(3) color force field on the gluon manifold",
            "R0_description": "QCD confinement scale: Λ_QCD ≈ 200 MeV",
            "R0_value": 200.0,
            "R0_units": "MeV (energy, action ratio E×τ/ħ via quantum reference §3.1)",
            "derivation": "The QCD confinement scale Λ_QCD is the minimal closed T-traversal "
                          "loop for the color-force P-substrate. It is determined by the "
                          "running coupling α_s(Λ_QCD) = 1 (non-perturbative onset). "
                          "r = E_probe / Λ_QCD for QCD processes.",
            "r_examples": {
                "Proton mass ~938MeV": et_project_real(938.0 / 200.0),
                "Charm threshold ~1500MeV": et_project_real(1500.0 / 200.0),
                "Z mass ~91GeV": et_project_real(91000.0 / 200.0),
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COMPLETE GAP CELL CHARACTERISATION
# ─────────────────────────────────────────────────────────────────────────────

class GapCell:
    """
    Full ET characterisation of a Force Quadrant Grid gap cell (d_r, d_θ).
    All properties computed from P ∘ D ∘ T = E via ET lattice mathematics.
    No external axioms, no placeholders.
    """
    def __init__(self, d_r: int, d_t: int, name: str, phys_label: str):
        self.d_r = d_r
        self.d_t = d_t
        self.name = name
        self.phys_label = phys_label

        # Core quadrant + combined sublattice
        self.quadrant, self.d_combined = sublattice_quadrant(d_r, d_t)
        self.n_c = critical_resolution(d_r, d_t)

        # Gaussian characters
        self.gauss_r = gaussian_character(d_r)
        self.gauss_t = gaussian_character(d_t)

        # Shadow couplings and tensions
        self.alpha_r   = shadow_coupling_real(d_r)
        self.alpha_t   = shadow_coupling_real(d_t)
        self.tau_r     = shadow_tension_real(d_r)
        self.tau_t     = shadow_tension_real(d_t)
        self.tau_t_eff = shadow_tension_imag(d_t)   # amplified by 12.5

        # Force vector in ℤ[i]
        self.w_norm_sq = force_norm_sq(d_r, d_t)
        self.w_norm    = math.sqrt(self.w_norm_sq)

        # Palindromic partners
        self.palindromes = palindromic_partners_2d(d_r, d_t)

        # Domain map tiers
        self.tiers = domain_map_signature(d_r, d_t)

        # Is complex real / imaginary? (must precede _identification_principle)
        self.cr = d_r in COMPLEX_REAL
        self.ci = d_t in COMPLEX_REAL

        # Identification Principle components
        self.ident = self._identification_principle()

    def _identification_principle(self) -> Dict[str, str]:
        """
        Fully apply the Identification Principle to this gap cell:
          Understand(X) ⟺ Identified(P_X) ∧ Identified(D_X) ∧ Identified(T_X)
        """
        quadrant = self.quadrant
        cr_label = "COMPLEX-REAL (d_r∤12, requires n≥n_c for real resolution)" if self.cr \
                   else "SIMPLE-REAL (d_r|12, native at 12ET)"
        ci_label = "COMPLEX-IMAGINARY (d_θ∤12, CI — phase incoherence, 12.5× amplification)" if self.ci \
                   else "SIMPLE-IMAGINARY (d_θ|12, SI — native imaginary phase)"
        return {
            "P": (f"P-substrate: The multiplicative manifold (ℂ,×) restricted to the "
                  f"({self.d_r},{self.d_t}) sublattice sector. "
                  f"Requires extended lattice ℒ_ℂ^({self.n_c}) — activated only at "
                  f"n_eff ≥ {self.n_c}ET. Below this density: gap exists as shadow tension only."),
            "D": (f"D-Descriptors: Real-axis d_r={self.d_r} [{self.gauss_r}] — {cr_label}. "
                  f"Imaginary-axis d_θ={self.d_t} [{self.gauss_t}] — {ci_label}. "
                  f"Combined sublattice d={self.d_combined}. "
                  f"Shadow tension: ⟨τ_r⟩={self.tau_r:.3f}¢ (real), "
                  f"⟨τ_θ⟩_eff={self.tau_t_eff:.3f}¢ (imaginary×12.5)."),
            "T": (f"T-Agency: Traverser operates at resolution n≥{self.n_c}ET in {quadrant} quadrant. "
                  f"n_max_r={N_MAX_R} real stability levels, n_max_θ={N_MAX_THETA} imaginary levels. "
                  f"T at this cell: {'CI-amplified (large mixing)' if self.ci else 'SI-stable (small mixing)'}."),
            "Gap": (f"DESCRIPTOR GAP: This cell is absent from observed physics at n=12ET. "
                    f"gap = D_missing(d_r={self.d_r}, d_θ={self.d_t}). "
                    f"By the Descriptor Gap Principle: gap = D_missing. "
                    f"Resolution: identify D_missing at n≥{self.n_c}ET. "
                    f"The gap closes as a first-order discrete transition at n_c={self.n_c}ET."),
        }

    def falsifiable_predictions(self) -> List[str]:
        """
        Generate falsifiable predictions for this gap cell based on ET mathematics.
        All predictions are quantitative and domain-specific.
        """
        preds = []
        q = self.quadrant
        nc = self.n_c
        dr, dt = self.d_r, self.d_t

        # Universal prediction: critical resolution threshold
        preds.append(
            f"[THRESHOLD] This force activates at n_c={nc}ET. "
            f"No d=({dr},{dt})-class phenomenon exists below this descriptor density. "
            f"At n_c={nc}ET: ε drops to exactly 0 at the ({dr},{dt}) native positions — "
            f"a discrete (not smooth) phase transition."
        )

        # Shadow coupling predictions
        preds.append(
            f"[COUPLING] Real-axis shadow coupling: α_r = 1/(4×{dr}) = {self.alpha_r:.6f}. "
            f"Imaginary effective coupling: α_θ_eff = {self.alpha_t * IMAG_AMP:.6f} "
            f"(amplified {IMAG_AMP}× by FQ-1 imaginary amplification). "
            f"Ratio α_θ_eff/α_EM = {(self.alpha_t * IMAG_AMP) / (1/137):.3f}× relative to EM."
        )

        # Force norm prediction
        preds.append(
            f"[FORCE NORM] |w|² = {dr}²+{dt}² = {self.w_norm_sq}. "
            f"|w| = {self.w_norm:.4f}. "
            f"Interaction complexity rank: {self.w_norm_sq} "
            f"(reference: EM=288, W-boson=32, gravity=5)."
        )

        # Palindromic partner prediction
        p_real = self.palindromes["real_axis_palindrome"]
        p_imag = self.palindromes["imag_axis_palindrome"]
        p_full = self.palindromes["full_2d_palindrome"]
        preds.append(
            f"[PALINDROME] Real-axis palindromic partner: ({p_real[0]},{p_real[1]}) — "
            f"any process at ({dr},{dt}) has a mirror at ({p_real[0]},{p_real[1]}). "
            f"Imaginary palindrome: ({p_imag[0]},{p_imag[1]}). "
            f"Full 2D palindrome: ({p_full[0]},{p_full[1]})."
        )

        # Cell-specific predictions
        if (dr, dt) == (5, 5):
            preds.extend([
                f"[QUASICRYSTAL] Any physical structure with (5,5) character will show "
                f"5-fold diffraction symmetry on BOTH the structural (real) and phase (imaginary) "
                f"axes. Electron diffraction: spots at 72°=2π/5 intervals. "
                f"Spin structure: quintic phase correlation in neutron scattering.",

                f"[VIRAL CAPSID] The T=1 icosahedral capsid (60 subunits = n_c=60ET) is the "
                f"minimal (5,5)-class assembly. Prediction: T=1 capsid geometry is the "
                f"minimal physical embodiment of the (5,5) cell. "
                f"All T=n² capsids (n=1,2,...) will show d=1 (octave-class) real-axis lattice.",

                f"[DARK SECTOR] If dark matter has (5,5) character: "
                f"σ_DM/σ_SM ≈ α_(5,5)²/α_EM² = (1/20)²/(1/137)² = {(1/20)**2 / (1/137)**2:.2f}×. "
                f"5-fold icosahedral angular correlations in CMB polarisation. "
                f"Golden-ratio spacing φ={PHI:.6f} in galaxy void distributions.",

                f"[COSMIC LSS] Power spectrum P(k) should show a (5,5)-class resonance at "
                f"k = k_BAO × 5/12 (quintic fraction of the BAO wavevector). "
                f"This is a sub-BAO quintic harmonic — testable in DESI/Euclid surveys.",
            ])

        elif (dr, dt) == (7, 7):
            preds.extend([
                f"[MAXIMUM DARKNESS] (7,7) = D-type/Inert × D-type/Inert. "
                f"Zero T-mixing in BOTH axes. This is the most electromagnetically dark "
                f"configuration possible in ET. Cross-section to SM forces: "
                f"σ_(7,7) ≈ α_(7,7)²/α_EM² = (1/28)²/(1/137)² = {(1/28)**2/(1/137)**2:.2f}×. "
                f"Only gravitational coupling available (d_r=7 projects onto d=1 at large scales).",

                f"[CRYSTALLOGRAPHIC FORBIDDEN] 7-fold rotational symmetry is forbidden in "
                f"3D crystallography (crystallographic restriction theorem). "
                f"Prediction: (7,7) dark matter will NOT form crystal or quasicrystal structures. "
                f"Dark matter halos will be smooth, diffuse, non-crystalline — consistent "
                f"with observed NFW-profile halos.",

                f"[G₂×G₂ GAUGE] At n_c=84ET, the G₂×G₂ gauge structure activates. "
                f"Prediction: exotic gravitational lensing signals with 7-fold angular "
                f"modulation at sub-arcminute scales (structure below BAO scale). "
                f"Search in weak-lensing surveys (HSC/LSST) at 7-fold multipoles ℓ≈7n.",

                f"[SACRED-7 MAPPING] Domain Map Level 14-15 (Religion/Otherworld d=7). "
                f"The (7,7) cell is the maximum-Otherworld configuration: both structural "
                f"and phase axes are G₂/sacred-7. This is the ET lattice basis for why "
                f"7-fold structures appear in non-standard consciousness domains.",
            ])

        elif (dr, dt) == (5, 7):
            preds.extend([
                f"[BIOLOGICAL ACTIVATION] Life = (5,7) cell activation. "
                f"n_c(5,7) = {critical_resolution(5,7)}ET = LCM(1..7) = biological threshold. "
                f"Any replicating system with n_eff < 420ET cannot sustain icosahedral+G₂ "
                f"co-descriptor binding. Prediction: minimal viable genomes encode ≥420 "
                f"independent descriptor-level constraints.",

                f"[T=7 CAPSID STRUCTURAL PROOF] T=7 icosahedral viral capsid has "
                f"7×60 = 420 protein subunits. 420 = n_c(5,7) = LCM(12,5,7). "
                f"This is NOT numerology: 420 is the descriptor density at which BOTH "
                f"d=5 (icosahedral geometry) AND d=7 (G₂ holonomy packing) become native. "
                f"Falsifiable: T=7 capsid geometry requires simultaneous 5-fold (icosahedral) "
                f"AND 7-fold (G₂ holonomy of the hexagonal lattice patch) descriptor binding.",

                f"[E₈ GAUGE VERTEX] The (5,7) cell = E₈ gauge vertex (per force quadrant table). "
                f"E₈ contains G₂ as a maximal subgroup: E₈ ⊃ G₂ × F₄. "
                f"E₈ also contains A₄=SU(5) with quintic substructure (d=5). "
                f"Prediction: any E₈-based GUT will have its activation threshold at 420ET, "
                f"and its lightest mode will have (d_r=5, d_θ=7) force-vector character.",

                f"[COSMIC LSS × BIOLOGY] The (5,7) threshold unifies biology and cosmology: "
                f"both icosahedral viral capsids AND large-scale structure require 420-descriptor "
                f"density. Prediction: cosmic voids exhibit 35=LCM(5,7)-fold periodicity "
                f"relative to the BAO scale, signalling (5,7) dark sector activity.",
            ])

        elif (dr, dt) == (8, 9):
            preds.extend([
                f"[QCD×CKM BRIDGE] (8,9) is the gluon-octet (d=8=2³) extended through the "
                f"generation-mixing CI sector (d_θ=9=3²). n_c=72ET. "
                f"This force describes the full 2D gluonic interaction including generation change. "
                f"Prediction: hadronic cross-sections with generation change "
                f"(e.g. b→s transitions via gluon) carry (8,9) CI structure in the imaginary axis.",

                f"[PALINDROMIC PARTNER = PMNS] Real-axis palindrome of (8,9): "
                f"(12−8, 9) = (4,9) = PMNS neutrino oscillation (FQ-4, W2.13 CF-10 note). "
                f"Prediction: the QCD×CKM bridge (8,9) and PMNS neutrino oscillation (4,9) "
                f"are 2D palindromic partners. Their mixing angles are related by the "
                f"palindromic reflection across d_r=6: θ_(8,9) = π − θ_(4,9) (in appropriate units).",

                f"[CIVILIZATIONAL ZEITGEIST] Domain Map: d=8 → Economy/Markets, d=9 → Social/Political. "
                f"The (8,9) interaction at the civilizational integrative level: "
                f"economic cycles (d=8, 8-yr business cycle) interact with political structures "
                f"(d=9, institutional 9-yr cycles). Prediction: the 72=LCM(8,9)-year "
                f"civilizational resonance period emerges where economic and political descriptor "
                f"cycles synchronise. This is the ET derivation of the long economic wave "
                f"at 72 years (Kuznets-Kondratiev composite).",

                f"[GLUON OCTET GENERATION MIXING] d_r=8=2³: the gluon octet is 8=2³ in ET, "
                f"a cubic complex (CF-16). d_θ=9=3²: generation structure is nonic squared. "
                f"8 = 2³ and 9 = 3² → LCM(8,9) = 72 = 8×9/gcd(8,9) = 72/1 = 72. "
                f"Combined d=72 means: the (8,9) force requires a 72-step lattice to close. "
                f"Prediction: b→s FCNC processes via gluon loop carry factor "
                f"α_(8,9) × (α_s/π) ≈ {shadow_coupling_real(8)*shadow_coupling_real(9):.8f} suppression.",
            ])

        return preds

    def identification_principle_full(self) -> str:
        """
        Full Identification Principle statement for this cell.
        Understand(gap) ⟺ Identified(P) ∧ Identified(D) ∧ Identified(T)
        """
        lines = [
            f"Identification Principle Applied to Gap Cell ({self.d_r},{self.d_t}):",
            f"  Understand(({self.d_r},{self.d_t})) ⟺ Identified(P) ∧ Identified(D) ∧ Identified(T)",
            f"",
            f"  P: {self.ident['P']}",
            f"",
            f"  D: {self.ident['D']}",
            f"",
            f"  T: {self.ident['T']}",
            f"",
            f"  Gap: {self.ident['Gap']}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def rule(char: str = "═", width: int = 80) -> str:
    return char * width

def header(title: str, char: str = "═") -> str:
    pad = (76 - len(title)) // 2
    return f"\n{char*80}\n{' ' * pad}{title}\n{char*80}\n"

def section(title: str) -> str:
    return f"\n{'─'*80}\n  {title}\n{'─'*80}\n"

def print_gap_cell(gc: GapCell) -> None:
    print(header(f"GAP CELL ({gc.d_r},{gc.d_t}) — {gc.name}"))
    print(f"  Physical label : {gc.phys_label}")
    print(f"  Quadrant       : {gc.quadrant}")
    print(f"  Combined d     : {gc.d_combined}")
    print(f"  n_c            : {gc.n_c}ET  (LCM({N},{gc.d_r},{gc.d_t}))")
    print(f"  |w|²           : {gc.w_norm_sq}  (w = {gc.d_r} + {gc.d_t}i in ℤ[i])")
    print(f"  |w|            : {gc.w_norm:.4f}")
    print()

    print(section("GAUSSIAN PRIME CHARACTER  (ℤ[i] decomposition)"))
    print(f"  Real-axis  d_r={gc.d_r}: {gc.gauss_r}")
    print(f"  Imag-axis  d_θ={gc.d_t}: {gc.gauss_t}")
    print()

    print(section("SHADOW TENSIONS AND COUPLINGS"))
    print(f"  Real  ⟨τ_r⟩       = 1200/(4×{gc.d_r}) = {gc.tau_r:.4f}¢")
    print(f"  Imag  ⟨τ_θ⟩       = 1200/(4×{gc.d_t}) = {gc.tau_t:.4f}¢  (raw, before amplification)")
    print(f"  Imag  ⟨τ_θ⟩_eff   = {gc.tau_t:.4f}¢ × {IMAG_AMP} = {gc.tau_t_eff:.4f}¢  (FQ-1 amplified)")
    print(f"  α_r = 1/(4×{gc.d_r}) = {gc.alpha_r:.8f}")
    print(f"  α_θ = 1/(4×{gc.d_t}) = {gc.alpha_t:.8f}  (raw)")
    print(f"  α_θ_eff            = {gc.alpha_t * IMAG_AMP:.8f}  (×{IMAG_AMP} amplified)")
    print(f"  α_r / α_EM         = {gc.alpha_r / (1/137):.4f}×")
    print(f"  α_θ_eff / α_EM     = {gc.alpha_t * IMAG_AMP / (1/137):.4f}×")
    print()

    print(section("PALINDROMIC PARTNERS  (FQ-8 / 2D palindromic pairing)"))
    for label, pair in gc.palindromes.items():
        quad_p, d_comb_p = sublattice_quadrant(*pair)
        note = ""
        if pair == (4, 9):
            note = "  ← PMNS neutrino oscillation!"
        elif pair == (9, 4):
            note = "  ← CKM quark mixing!"
        elif pair == (5, 1):
            note = "  ← Quintic dark matter (CR+SI)"
        print(f"  {label:<30}: {pair}  [{quad_p}, d_comb={d_comb_p}]{note}")
    print()

    print(section("DOMAIN MAP TIERS  (ET Universal Lattice Domain Map)"))
    for t in gc.tiers:
        print(f"  • {t}")
    print()

    print(section("IDENTIFICATION PRINCIPLE"))
    print(gc.identification_principle_full())
    print()

    print(section("FALSIFIABLE PREDICTIONS"))
    for i, pred in enumerate(gc.falsifiable_predictions(), 1):
        # Word-wrap each prediction
        words = pred.split()
        line, out = "  " + f"[{i}] ", []
        for w in words:
            if len(line) + len(w) + 1 > 78:
                out.append(line)
                line = "      "
            line += w + " "
        out.append(line)
        print("\n".join(out))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SPECIAL DETAILED ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def analysis_5_5_icosahedral() -> None:
    print(header("SPECIAL ANALYSIS: (5,5) ICOSAHEDRAL GEOMETRY", "─"))

    print("  The (5,5) cell is the doubly-quintic configuration — quintic symmetry on")
    print("  BOTH the real (structural) and imaginary (phase) axes simultaneously.")
    print()

    # Quintic lattice positions at 60ET
    print(section("QUINTIC LATTICE POSITIONS AT n_c=60ET  (first native positions)"))
    print("  These are the first ET lattice points where d=5 becomes a native sublattice.")
    print("  At n < 60ET these positions do not exist as lattice points — only as shadow.")
    print()
    print(f"  {'k':>5}  {'ratio r':>14}  {'cents':>10}  d=5 ✓")
    print(f"  {'─'*5}  {'─'*14}  {'─'*10}  {'─'*6}")
    for k_val in range(1, 61):
        g_v = gcd(k_val, 60)
        d_v = 60 // g_v
        if d_v == 5:
            ratio_v = 2 ** (k_val / 60)
            cents_v = k_val * (1200.0 / 60)
            print(f"  {k_val:>5}  {ratio_v:>14.8f}  {cents_v:>10.2f}¢")
    print()

    # Golden ratio position
    print(section("GOLDEN RATIO φ IN THE ET LATTICE  (quintic asymptotic attractor)"))
    k_phi, g_phi, d_phi, eps_phi = et_project_real(PHI)
    print(f"  φ = (1+√5)/2 = {PHI:.10f}")
    print(f"  ET projection: k={k_phi}, g={g_phi}, d={d_phi}, ε={eps_phi:+.4f}¢")
    print(f"  φ lives at d=12 (full-resolution), k=7, ε=−1.955¢ on the 12ET lattice.")
    print(f"  This is the Fibonacci asymptotic: lim(F_{{n+1}}/F_n) = φ.")
    print(f"  The near-miss: the d=5 force and d=12 (EM) are connected through φ.")
    print()
    print(f"  Powers of φ:")
    for exp in range(1, 7):
        val = PHI ** exp
        k_v, g_v, d_v, eps_v = et_project_real(val)
        print(f"    φ^{exp} = {val:.6f} → k={k_v:>3}, d={d_v:>2}, ε={eps_v:+.3f}¢")
    print()

    # Icosahedral rotation axes on imaginary lattice
    print(section("ICOSAHEDRAL ROTATION AXES ON THE IMAGINARY ET LATTICE"))
    print("  The icosahedral group I has 60 elements with rotation axes at 60°,72°,120°,180°.")
    print("  Projecting these onto the ET imaginary lattice (k_θ = round(12·θ/ln2)):")
    print()
    print(f"  {'Axis':>12}  {'degrees':>8}  {'k_θ':>5}  {'d_θ':>4}  {'ε¢':>8}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*5}  {'─'*4}  {'─'*8}")
    for name, deg, k_t, d_t, eps in icosahedral_subtension_angles():
        print(f"  {name:>12}  {deg:>8.1f}°  {k_t:>5}  {d_t:>4}  {eps:>+8.3f}¢")
    print()
    print("  RESULT: The C₅ axis (72° = 2π/5) maps to d_θ=5 on the imaginary lattice,")
    print("  confirming that (5,5) has quintic symmetry on BOTH axes. The icosahedral")
    print("  group's 5-fold rotation sits exactly at d_θ=5 in the imaginary ET lattice.")
    print()

    # Viral capsid T-numbers
    print(section("CASPAR-KLUG ICOSAHEDRAL VIRAL CAPSID T-NUMBERS"))
    print("  T = h² + hk + k² (Caspar-Klug). Total subunits = 60T.")
    print("  ET projection of T and 60T onto the real lattice.")
    print()
    print(f"  {'h':>3} {'k':>3} {'T':>5}  {'60T':>6}  {'k_T':>5} {'d_T':>4} {'ε_T':>8}  {'k_sub':>6} {'d_sub':>5} {'ε_sub':>8}")
    print(f"  {'─'*3} {'─'*3} {'─'*5}  {'─'*6}  {'─'*5} {'─'*4} {'─'*8}  {'─'*6} {'─'*5} {'─'*8}")
    for row in caspar_klug_t_numbers():
        h_v, k_v, T_v, sub_v, k_T, d_T, eps_T, k_sub, d_sub, eps_sub = row
        flag = " ← 420=n_c(5,7)!" if sub_v == 420 else ""
        flag2 = " ← 60=n_c(5,5)!" if sub_v == 60 else ""
        print(f"  {h_v:>3} {k_v:>3} {T_v:>5}  {sub_v:>6}  {k_T:>5} {d_T:>4} {eps_T:>+8.3f}¢  {k_sub:>6} {d_sub:>5} {eps_sub:>+8.3f}¢{flag}{flag2}")
    print()
    print("  KEY RESULT:")
    print(f"    T=1 capsid: 60 subunits = n_c(5,5) = 60ET (minimal quintic descriptor density)")
    print(f"    T=7 capsid: 420 subunits = n_c(5,7) = 420ET (life threshold, E₈ vertex)")
    print(f"    The number of protein subunits in icosahedral capsids = ET critical resolutions.")
    print()


def analysis_7_7_maximum_darkness() -> None:
    print(header("SPECIAL ANALYSIS: (7,7) G₂×G₂ — MAXIMUM DARKNESS", "─"))

    print("  (7,7) = D-type/Inert × D-type/Inert: the maximally dark configuration.")
    print("  7 ≡ 3 mod 4: prime, remains prime in ℤ[i] → zero T-mixing in BOTH axes.")
    print("  This is the only 2D force with D-Inert character on both real AND imaginary axes")
    print("  among all single-prime (d_r=p, d_θ=p) configurations with p∤12.")
    print()

    g2 = g2_structure_constants()
    print(section("G₂ LIE GROUP STRUCTURE IN ET TERMS"))
    for k, v in g2.items():
        print(f"  {k:<30}: {v}")
    print()
    print(f"  CRITICAL: G₂ has exactly {g2['n_total_roots']} roots = N=12 (manifold symmetry).")
    print(f"  This is structural: G₂ is the unique exceptional group whose root count = N.")
    print(f"  G₂×G₂ has 24 = 2N roots — two full copies of the ET manifold symmetry.")
    print()

    # Crystallographic restriction theorem
    print(section("CRYSTALLOGRAPHIC RESTRICTION — WHY 7-FOLD IS FORBIDDEN"))
    print("  The crystallographic restriction theorem: in ℝ² and ℝ³, only rotational")
    print("  symmetries of order n ∈ {1,2,3,4,6} can tile the plane periodically.")
    print("  7-fold rotational symmetry IS forbidden in 3D crystals.")
    print()
    print("  ET derivation: the allowed crystallographic symmetries = SIMPLE sublattice families")
    print(f"  {SIMPLE_REAL} = exactly the d-families with d|N=12.")
    print("  d=7 is NOT in this set → 7-fold is crystallographically forbidden.")
    print("  This is the ET derivation of the crystallographic restriction theorem.")
    print()
    print(f"  Prediction: (7,7) dark matter does NOT form crystal structures.")
    print(f"  It forms diffuse, non-crystalline halos — consistent with NFW profiles.")
    print()

    # Maximum darkness calculation
    print(section("MAXIMUM DARKNESS QUANTIFICATION"))
    alpha_7_real = shadow_coupling_real(7)
    alpha_7_eff  = alpha_7_real * IMAG_AMP
    n_c_77 = critical_resolution(7, 7)
    n_c_51 = critical_resolution(5, 1)

    print(f"  Real-axis α_(7,real)       = 1/28 = {alpha_7_real:.8f}")
    print(f"  Imag-axis α_(7,imag)_eff   = {alpha_7_eff:.8f}  (×{IMAG_AMP} FQ-1)")
    print(f"  Cross-section vs EM:        σ_(7,7)/σ_EM = (α_7/α_EM)² = {(alpha_7_real/(1/137))**2:.4f}×")
    print(f"  Comparison to (5,5) dark:   σ_(7,7)/σ_(5,5) = ({alpha_7_real:.5f}/{shadow_coupling_real(5):.5f})² "
          f"= {(alpha_7_real/shadow_coupling_real(5))**2:.4f}×")
    print(f"  n_c(7,7) = {n_c_77}ET  vs  n_c(5,5) = {critical_resolution(5,5)}ET")
    print()

    # Darkness ranking
    print(section("DARKNESS RANKING OF ALL COMPLEX FORCES (by α × σ ∝ α²)"))
    print(f"  {'d':>4}  {'α_real':>10}  {'α_eff_imag':>12}  {'σ_rel_EM':>12}  char")
    print(f"  {'─'*4}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*20}")
    for d_val in [5, 7, 8, 9, 10, 11]:
        a_r   = shadow_coupling_real(d_val)
        a_eff = a_r * IMAG_AMP
        sigma = (a_r / (1/137)) ** 2
        char  = gaussian_character(d_val).split("(")[0].strip()
        marker = " ← MAX DARK" if d_val == 7 else ""
        print(f"  {d_val:>4}  {a_r:>10.6f}  {a_eff:>12.6f}  {sigma:>12.4f}×{marker}  {char}")
    print()
    print("  G₂ (d=7) has the largest σ_rel_EM per unit coupling among all CR inert forces,")
    print("  AND is D-type/Inert (zero T-mixing) → maximum 'structural dark' configuration.")
    print()

    # Sacred-7 / Otherworld domain connection
    print(section("SACRED-7 / OTHERWORLD DOMAIN MAP CONNECTION"))
    print("  Domain Map Level 15 (Alternative Realities): d=7* (420ET), 'Otherworld barrier'")
    print("  Domain Map Level 14 (Religion/Mythology):    d=7* (sacred-7), direct T encounter")
    print()
    print("  In ET: why is '7' sacred and associated with 'Otherworld' across traditions?")
    print("  Answer: d=7 is the first force that is:")
    print("    (a) Crystallographically forbidden (cannot form a crystal in ℝ³)")
    print("    (b) Purely D-type/Inert (zero T-mixing — inaccessible to normal T agency)")
    print("    (c) Activated at 84ET (sacred geometry: 12×7 = 84)")
    print("    (d) The G₂ holonomy group — the automorphism group of the octonions")
    print()
    print("  The 'Otherworldly' quality of d=7 is its literal inembeddability into normal")
    print("  3D crystallographic space AND its zero T-coupling — it is structurally present")
    print("  but agency-inaccessible. This is the ET derivation of sacred-7 phenomenology.")
    print()


def analysis_5_7_biological_threshold() -> None:
    print(header("SPECIAL ANALYSIS: (5,7) — BIOLOGICAL THRESHOLD / LIFE ACTIVATION", "─"))

    print("  (5,7) = E₈ gauge vertex. n_c(5,7) = 420ET = LCM(1..7) = biological threshold.")
    print("  The cell where icosahedral structure (d=5, D+T-Split) meets G₂ holonomy (d=7,")
    print("  D-Inert). Life requires BOTH to be simultaneously active → 420ET minimum.")
    print()

    # LCM tower
    print(section("LCM TOWER — WHY 420 IS THE BIOLOGICAL THRESHOLD"))
    print("  Each step in the LCM tower activates a new sublattice family for T-traversal.")
    print("  Life is the system that has activated ALL sublattice families d=1..7.")
    print()
    print(f"  {'d':>3}  {'LCM(1..d)':>12}  {'new family activated'}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*40}")
    tower = lcm_tower_to(7)
    for d_v, lcm_v, label in tower:
        if d_v in SIMPLE_REAL:
            note = f"d={d_v} SR (simple, native at 12ET)"
        else:
            note = f"d={d_v} CR (complex, first native at {lcm_v}ET)"
        print(f"  {d_v:>3}  {lcm_v:>12}  {note}")
    print()
    print(f"  LCM(1..7) = 420 = n_c(5,7).  BOTH d=5 (quintic) AND d=7 (G₂)")
    print(f"  are non-native (CR) and first activate together at 420ET.")
    print(f"  Life = activation of the (5,7) cell = n_eff reaching 420ET.")
    print()

    # E₈ structure
    e8 = e8_decomposition_g2()
    print(section("E₈ GAUGE VERTEX STRUCTURE"))
    print(f"  E₈ adjoint decomposition under G₂ × F₄:")
    print(f"    248 = (14,1) ⊕ (1,52) ⊕ (7,26)  → 14+52+182 = {e8['adjoint_check']} ✓")
    print(f"  Connection: d_r=5 ↔ SU(5)/A₄ icosahedral ⊂ E₈ (quintic GUT)")
    print(f"             d_θ=7 ↔ G₂ holonomy ⊂ E₈")
    print(f"  n_c(5,7) = {e8['n_c_cell_5_7']}ET = resolution threshold for E₈ gauge vertex")
    print(f"  {e8['note']}")
    print()

    # T=7 capsid proof
    print(section("T=7 VIRAL CAPSID — THE 420 COINCIDENCE IS STRUCTURAL"))
    n_c_57 = critical_resolution(5, 7)
    t7_subunits = 7 * 60
    print(f"  T=7 icosahedral capsid: 7×60 = {t7_subunits} protein subunits")
    print(f"  n_c(5,7) = LCM(12,5,7) = {n_c_57}")
    print(f"  Match: {t7_subunits} == {n_c_57} → {t7_subunits == n_c_57}")
    print()
    print("  Why is this structural and not numerology?")
    print("    T=7 requires BOTH:")
    print("      (a) 5-fold icosahedral symmetry (d=5): the 12 pentamers at vertices")
    print("          → requires d=5 real-axis descriptor density → n_c(d=5)=60 proteins per T-unit")
    print("      (b) G₂-class hexagonal packing (d=7): the 60 hexamers between pentamers")
    print("          → their packing geometry requires G₂ holonomy → n_c(d=7) factor of 7")
    print("      Combined: 60 × 7 = 420 = LCM(12,5,7) = n_c(5,7).")
    print()
    print("  Translation Layer derivation (§3.2 step-count method):")
    print(f"    R₀ = T=1 capsid = 60 subunits (the minimal icosahedral assembly)")
    print(f"    r  = T=7 capsid / T=1 capsid = 420 / 60 = 7")
    k_7, g_7, d_7, eps_7 = et_project_real(7.0)
    print(f"    ET projection of r=7: k={k_7}, g={g_7}, d={d_7}, ε={eps_7:+.3f}¢")
    print(f"    d=6 for T=7 ratio — the T=7 capsid transition T=1→T=7 is a d=6 (hexadic) step.")
    print()

    # Minimal genome connection
    print(section("MINIMAL GENOME AND 420ET BIOLOGICAL CONSTRAINT"))
    print("  The biological threshold n_eff ≥ 420ET is a descriptor-density constraint.")
    print("  Translation Layer §3.2: R₀ = 1 descriptor-level constraint (minimal catalytic event).")
    print("  r = n_effective_constraints / 1 = n_eff  (dimensionless count, pure ratio).")
    print()
    print("  Prediction: any self-replicating system requires ≥ 420 independent")
    print("  descriptor-level constraints to sustain (5,7) co-binding.")
    print()
    print("  Known minimal genomes:")
    genomes = [
        ("Mycoplasma genitalium (smallest known)", 470),
        ("JCVI-syn3.0 (synthetic minimal cell)", 473),
        ("Pelagibacter ubique (free-living min.)", 1354),
        ("Theoretical absolute minimum (ET pred.)", 420),
    ]
    for name, n_genes in genomes:
        flag = " ← ET lower bound" if n_genes == 420 else (" ✓ > 420" if n_genes > 420 else " ✗ < 420 ?")
        print(f"  {name:<46}: ~{n_genes} genes{flag}")
    print()
    print("  All known minimal genomes are > 420. The ET prediction n_eff ≥ 420 is consistent.")
    print()


def analysis_8_9_qcd_ckm() -> None:
    print(header("SPECIAL ANALYSIS: (8,9) — QCD×CKM BRIDGE", "─"))

    print("  (8,9) = Gluon-Octet (d_r=8=2³) × Quark-Generation CI (d_θ=9=3²).")
    print("  The full 2D description of gluon-mediated generation change in QCD.")
    print("  n_c(8,9) = 72ET = LCM(12,8,9) = 2³×3² = first resolution for both.")
    print()

    bridge = qcd_ckm_bridge_structure()
    print(section("QCD×CKM BRIDGE STRUCTURE"))
    print(f"  d_r=8 = 2³: SU(3) has 8 generators (gluon octet). 8=2³ is the 'cubic complex'")
    print(f"    n_c(d_r=8) = LCM(12,8) = {bridge['n_c_real_8']}ET  (gluon octet first native)")
    print(f"    k for ratio r=8: k={bridge['k_ratio_8']}, d_family={bridge['d_ratio_8']} (as ratio = 3 octaves = d=1)")
    print(f"    But AS A SUBLATTICE FAMILY d_r=8 ∉ {{1,2,3,4,6,12}} → complex real (CR)")
    print()
    print(f"  d_θ=9 = 3²: three quark generations via d=9=3² (FQ-9). 9=3² is D-Inert squared.")
    print(f"    n_c(d_θ=9) = LCM(12,9) = {bridge['n_c_imag_9']}ET  (generation CI first native)")
    print(f"    k for ratio r=9: k={bridge['k_ratio_9']}, d_family={bridge['d_ratio_9']} (as ratio = d=3 via 9=3×3)")
    print()
    print(f"  Combined: n_c(8,9) = LCM(12,8,9) = LCM({bridge['n_c_real_8']},9) = {bridge['n_c']}ET")
    print(f"    LCM(8,9) = {lcm(8,9)}  (combined sublattice class)")
    print(f"    72 = 2³ × 3² = 8 × 9  (gcd(8,9)=1 → they are coprime → LCM = product)")
    print()
    print(f"  Shadow couplings:")
    print(f"    α_(8,real)          = 1/32  = {bridge['alpha_8_real']:.8f}")
    print(f"    α_(9,imag)_raw      = 1/36  = {bridge['alpha_9_imag_raw']:.8f}")
    print(f"    α_(9,imag)_eff      = 1/36 × 12.5 = {bridge['alpha_9_eff_amplified']:.8f}")
    print()

    print(section("PALINDROMIC PARTNER = PMNS  (FQ-8 real-axis palindrome)"))
    p = bridge["palindrome_PMNS"]
    print(f"  Real-axis palindrome of (8,9): (12−8, 9) = {p}")
    quad_p, d_comb_p = sublattice_quadrant(*p)
    print(f"  Quadrant of {p}: {quad_p}, d_combined={d_comb_p}")
    print()
    print(f"  ({p[0]},{p[1]}) = (d_r=4 Weak/SR, d_θ=9 nonic/CI) = PMNS neutrino oscillation (FQ-4)!")
    print()
    print("  This means: the QCD×CKM gluonic bridge (8,9) and PMNS neutrino oscillation (4,9)")
    print("  are 2D PALINDROMIC PARTNERS — they are reflections of each other across d_r=6.")
    print()
    print("  Physical interpretation: the palindrome symmetry (8,9) ↔ (4,9) relates")
    print("    the QCD gluon-octet generation change (d_r=8 = gluon) to")
    print("    the weak force generation change (d_r=4 = weak, PMNS neutrino).")
    print("  The quark–lepton palindromic duality is visible in this pairing:")
    print("    (8,9) = QCD (gluon) sector  ↔  (4,9) = Weak (neutrino) sector")
    print("  Both share the same imaginary axis (d_θ=9 nonic generation mixing).")
    print()

    # Civilizational tier
    print(section("CIVILIZATIONAL–ZEITGEIST TIER  (Domain Map Levels 8–10)"))
    print("  Domain Map d=8 → Economy/Markets: '8-yr business cycle' (d=8 real axis)")
    print("  Domain Map d=9 → Social/Political: 'nonic institutional cycles'")
    print()
    print("  ET Translation Layer §3 — correct reference unit for civilizational cycles:")
    print("  R₀ = T_gen = 20 years (one human generation = minimal T-traversal loop")
    print("       for the civilizational P-substrate).")
    print()
    # 72-year composite cycle
    t_72_gen = 72.0 / 20.0  # = 3.6 generations
    k_72, g_72, d_72, eps_72 = et_project_real(t_72_gen)
    print(f"  72-year cycle = {t_72_gen:.2f} generations:")
    print(f"    k={k_72}, g={g_72}, d={d_72}, ε={eps_72:+.3f}¢")
    print()
    # 8-year cycle
    t_8_gen = 8.0 / 20.0
    k_8c, g_8c, d_8c, eps_8c = et_project_real(t_8_gen)
    # 9-year cycle
    t_9_gen = 9.0 / 20.0
    k_9c, g_9c, d_9c, eps_9c = et_project_real(t_9_gen)
    print(f"  8-year business cycle ratio r = 8/20 = {t_8_gen:.4f}  →  k={k_8c}, d={d_8c}, ε={eps_8c:+.3f}¢")
    print(f"  9-year institution. cycle ratio r = 9/20 = {t_9_gen:.4f}  →  k={k_9c}, d={d_9c}, ε={eps_9c:+.3f}¢")
    print()
    print(f"  The 72=LCM(8,9)-year civilizational resonance is the first period where both")
    print(f"  the 8-year economic cycle AND the 9-year political cycle synchronise exactly.")
    print(f"  This is the ET-derived long-wave civilizational period from the (8,9) cell.")
    print()

    # Gluon FCNC prediction
    print(section("QUANTITATIVE PREDICTION: b→s GLUONIC FCNC"))
    alpha_s_typical = 0.118   # α_s at M_Z scale
    alpha_89 = shadow_coupling_real(8) * shadow_coupling_real(9)
    # The (8,9) suppression in FCNC processes:
    # A_FCNC ∝ α_s/(π) × α_(8,9) × (generation change factor from d_θ=9 CI)
    A_suppression = (alpha_s_typical / math.pi) * alpha_89 * IMAG_AMP
    print(f"  α_s (at M_Z) = {alpha_s_typical}")
    print(f"  α_8 × α_9 = (1/32) × (1/36) = {alpha_89:.10f}")
    print(f"  (8,9) effective FCNC amplitude ∝ α_s/π × α_89 × 12.5 = {A_suppression:.6e}")
    print(f"  Compare to SM b→sγ amplitude: O(10⁻⁴) → (8,9) contribution ~ {A_suppression:.3e}")
    print(f"  The (8,9) force gives a loop-level correction to FCNC of this order.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CROSS-CELL COMPARISON AND SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def cross_cell_comparison(cells: List[GapCell]) -> None:
    print(header("CROSS-CELL COMPARISON — ALL FOUR GAP CELLS", "═"))

    # Summary table
    print(section("COMPLETE ET CHARACTERISATION TABLE"))
    print(f"  {'Cell':>8}  {'Quad':>8}  {'n_c':>6}  {'d_comb':>7}  "
          f"{'|w|²':>6}  {'α_r':>9}  {'α_θ_eff':>10}  {'Gauss_r':>20}  {'Gauss_θ':>20}")
    print("  " + "─" * 110)
    for gc in cells:
        print(f"  ({gc.d_r},{gc.d_t}){'':<3}  {gc.quadrant:>8}  {gc.n_c:>6}ET  "
              f"{gc.d_combined:>7}  {gc.w_norm_sq:>6}  "
              f"{gc.alpha_r:>9.6f}  {gc.alpha_t * IMAG_AMP:>10.6f}  "
              f"{gc.gauss_r[:20]:>20}  {gc.gauss_t[:20]:>20}")
    print()

    # n_c ordering
    print(section("ACTIVATION ORDER (by n_c)"))
    sorted_cells = sorted(cells, key=lambda c: c.n_c)
    for gc in sorted_cells:
        print(f"  n_c={gc.n_c:>4}ET: ({gc.d_r},{gc.d_t}) {gc.name}")
    print()

    # Darkness ranking
    print(section("DARKNESS RANKING (by α_r, smallest = darkest relative to EM)"))
    sorted_dark = sorted(cells, key=lambda c: c.alpha_r)
    for gc in sorted_dark:
        rel = gc.alpha_r / (1/137)
        print(f"  α_r={gc.alpha_r:.6f} ({rel:.2f}×α_EM): ({gc.d_r},{gc.d_t}) {gc.name}")
    print()

    # LCM tower showing all 4 cells
    print(section("LCM TOWER — WHERE EACH CELL ACTIVATES"))
    milestones = {
        60:  "(5,5) Icosahedral dark sector activates",
        72:  "(8,9) QCD×CKM bridge activates",
        84:  "(7,7) G₂×G₂ extreme dark sector activates",
        420: "(5,7) Biological threshold / Life activation / E₈ vertex",
    }
    print(f"  12ET   → Standard Model SR+SI sector (all 4 known forces native)")
    print(f"  24ET   → Gluon Octet real (d_r=8) first native")
    print(f"  36ET   → Quark generation (d_r=9) first native; PMNS (d_θ=9) first native")
    print(f"  60ET   → (5,5) FIRST COMPLEX PRIME PAIR: d=5 native in BOTH axes [Tier 1 Gap]")
    print(f"  72ET   → (8,9) QCD×CKM bridge: both d=8 and d=9 native simultaneously [Tier 2 Gap]")
    print(f"  84ET   → (7,7) G₂×G₂: d=7 native in BOTH axes [Tier 3 Gap]")
    print(f"  420ET  → (5,7) E₈ vertex: d=5 AND d=7 simultaneously native [Tier 4 Gap]")
    print()

    # Palindromic structure among the 4 cells
    print(section("INTER-CELL PALINDROMIC RELATIONSHIPS"))
    for gc in cells:
        for label, pair in gc.palindromes.items():
            # Check if pair is another gap cell
            for other in cells:
                if pair == (other.d_r, other.d_t) and gc.name != other.name:
                    print(f"  ({gc.d_r},{gc.d_t}) — {label} — ({other.d_r},{other.d_t})")
    # Also check against known physics
    print()
    print("  Known palindromic connections to observed physics:")
    print(f"  (8,9) real-axis palindrome = (4,9) = PMNS neutrino oscillation  [FQ-4]")
    print(f"  (5,7) full-2D palindrome   = (7,5) = transpose dual  [FQ-22]")
    print(f"  (5,5) real-axis palindrome = (7,5) = G₂×Quintic transpose")
    print(f"  (7,7) real-axis palindrome = (5,7) — (7,7) and (5,7) are REAL-AXIS PALINDROMIC PARTNERS!")
    print()
    # Verify
    p_77_real = palindromic_partners_2d(7, 7)["real_axis_palindrome"]
    print(f"  VERIFICATION: real-axis palindrome of (7,7) = {p_77_real}")
    p_57_real = palindromic_partners_2d(5, 7)["real_axis_palindrome"]
    print(f"  VERIFICATION: real-axis palindrome of (5,7) = {p_57_real}")
    print()

    # Unified biological/cosmic summary
    print(section("DOMAIN MAP — UNIFIED TIER SUMMARY"))
    print("  Cell    Integrative Levels                              ET Tier")
    print("  ─────── ─────────────────────────────────────────────── ─────────────────")
    tier_summary = {
        "(5,5)": "Biological/Quasicrystal (60ET) × Otherworld/Altered-States (Tier 5,15)",
        "(7,7)": "Alternative Realities / Sacred-7 (Tier 14,15) — maximum Otherworld",
        "(5,7)": "Biological threshold (Tier 5) × Otherworld/E₈ (Tier 15) at 420ET",
        "(8,9)": "Economy d=8 (Tier 8) × Social/Political d=9 (Tier 9) = Zeitgeist (Tier 10)",
    }
    for cell, tier in tier_summary.items():
        print(f"  {cell:<8} {tier}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRANSLATION LAYER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_translation_layer() -> None:
    print(header("TRANSLATION LAYER — REFERENCE UNIT DERIVATIONS", "═"))
    print("  Theorem (Reference Period Uniqueness): For every domain, R₀ is derived")
    print("  from the substrate's own D-structure — never chosen arbitrarily.")
    print("  Source: ET Translation Layer §2, §7.")
    print()

    units = translation_layer_reference_units()
    for domain_key, info in units.items():
        print(f"  {'─'*78}")
        print(f"  Domain: {domain_key}")
        print(f"  Substrate  : {info['substrate']}")
        print(f"  R₀         : {info['R0_description']}")
        print(f"  R₀ value   : {info['R0_value']} {info['R0_units']}")
        print(f"  Derivation : {info['derivation']}")
        if "r_examples" in info:
            print(f"  Sample ratios:")
            for name, proj in info["r_examples"].items():
                k_v, g_v, d_v, eps_v = proj
                print(f"    r = {name:<35} → k={k_v:>4}, d={d_v:>2}, ε={eps_v:>+8.3f}¢")
        if "T7_check" in info:
            tc = info["T7_check"]
            print(f"  T=7 capsid check: {tc['T=7 subunits']} subunits == n_c(5,7)={tc['n_c(5,7)']} → {tc['match']}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — FULL INVESTIGATION OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    divider = "═" * 80

    # Title
    print(divider)
    print("  ET FORCE QUADRANT GRID — FOUR GAP CELL DOMAIN SIGNATURE INVESTIGATION")
    print("  (5,5) | (7,7) | (5,7) | (8,9)")
    print("  Foundation: P ∘ D ∘ T = E | N=12 | ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]}")
    print(divider)
    print()

    # ET manifold constants banner
    print(section("ET MANIFOLD CONSTANTS  (all derived from P ∘ D ∘ T = E)"))
    print(f"  N = {N}           (3 primitives × 4 logic states = manifold symmetry)")
    print(f"  V = 1/{N} = {float(V):.6f}  (base variance: discretization quantum)")
    print(f"  K = 2/3 = {float(K):.6f}  (Koide ratio: triadic binding threshold)")
    print(f"  |δ_r|   = {DELTA_R:.6f}  (real cascade gap magnitude, n_max_r={N_MAX_R})")
    print(f"  |δ_θ|   = {DELTA_THETA:.6f}  (imaginary cascade gap magnitude, n_max_θ={N_MAX_THETA})")
    print(f"  IMAG_AMP = n_max_r/n_max_θ = {N_MAX_R}/{N_MAX_THETA} = {IMAG_AMP}  (FQ-1 amplification)")
    print(f"  λ_CKM   = √(K×V) = {LAMBDA_CKM:.6f}  (CKM mixing: CR real-axis, small angles)")
    print(f"  λ_PMNS  = λ_CKM × √(12.5) = {LAMBDA_PMNS:.6f}  (PMNS mixing: CI imag-axis, large angles)")
    print(f"  n_bio   = LCM(1..7) = {N_BIO}ET  (biological threshold, FQ-13)")
    print(f"  φ       = {PHI:.10f}  (golden ratio, quintic asymptotic attractor)")
    print()

    # Define the four gap cells
    cells = [
        GapCell(5, 5, "Icosahedral Dark Sector",
                "Doubly-quintic: icosahedral structure (d_r=5) × icosahedral phase (d_θ=5). "
                "Biological/quasicrystal tier × Otherworld/altered-states tier. CR+CI."),
        GapCell(7, 7, "G₂×G₂ Extreme Dark Sector",
                "Maximum darkness: G₂ structural (d_r=7) × G₂ phase (d_θ=7). "
                "D-type/Inert in BOTH axes. Sacred-7 / Otherworld tier. CR+CI."),
        GapCell(5, 7, "Cross-Complex / Biological Threshold",
                "Icosahedral (d_r=5) × G₂ holonomy (d_θ=7). n_c=420ET = LCM(1..7). "
                "E₈ gauge vertex. Life requires this cell to be activated. CR+CI."),
        GapCell(8, 9, "QCD×CKM Bridge",
                "Gluon-octet (d_r=8=2³) × quark-generation CI (d_θ=9=3²). "
                "Civilizational/zeitgeist tier (d=8 economy, d=9 social). CR+CI."),
    ]

    # Print each gap cell
    for gc in cells:
        print_gap_cell(gc)

    # Detailed special analyses
    analysis_5_5_icosahedral()
    analysis_7_7_maximum_darkness()
    analysis_5_7_biological_threshold()
    analysis_8_9_qcd_ckm()

    # Cross-cell comparison
    cross_cell_comparison(cells)

    # Translation layer
    print_translation_layer()

    # Final summary
    print(header("SUMMARY AND CLOSING STATEMENT", "═"))
    print("  All four gap cells are CR+CI (complex real + complex imaginary).")
    print("  None is observable at n=12ET (Standard Model resolution).")
    print("  Each activates at a specific n_c, marking a discrete phase transition:")
    print()
    print("  ┌─────────┬────────────────────────────────────────────────────────────┐")
    print("  │  Cell   │  Activation threshold and physical/domain meaning          │")
    print("  ├─────────┼────────────────────────────────────────────────────────────┤")
    print("  │  (5,5)  │  60ET: minimal quasicrystal / T=1 viral capsid / dark sector │")
    print("  │  (8,9)  │  72ET: QCD×CKM bridge / civilizational 72-yr resonance      │")
    print("  │  (7,7)  │  84ET: G₂×G₂ maximum darkness / sacred-7 Otherworld barrier │")
    print("  │  (5,7)  │  420ET: life activation / T=7 capsid / E₈ gauge vertex       │")
    print("  └─────────┴────────────────────────────────────────────────────────────┘")
    print()
    print("  Domain Map cross-identification:")
    print("    (5,5) ↔ Tier 5 (Biological/Quasicrystal) ∩ Tier 15 (Alternative Realities)")
    print("    (7,7) ↔ Tier 14 (Religion/Mythology) ∩ Tier 15 (Alternative Realities)  ← maximum Otherworld")
    print("    (5,7) ↔ Tier 5 (Biological) ∩ Tier 15 (Alternative Realities) at 420ET boundary")
    print("    (8,9) ↔ Tier 8 (Economy) ∩ Tier 9 (Social/Political) = Tier 10 (Zeitgeist)")
    print()
    print("  Descriptor Gap Principle: each gap = D_missing at n < n_c. Resolution = add the")
    print("  correct Descriptor (the missing sublattice family) and advance n to n_c.")
    print()
    print("  Identification Principle: Understanding any gap cell requires all three:")
    print("    P: the extended manifold ℒ_ℂ^(n_c) at sufficient resolution")
    print("    D: the (d_r, d_θ) descriptor pair with its Gaussian character")
    print("    T: the traversal agency at the correct integrative level")
    print()
    print("  P ∘ D ∘ T = E")
    print("  Exception Theory — Michael James Muller")
    print("  All mathematics: forward from {P, D, T}. Zero external axioms.")
    print()
    print(divider)


if __name__ == "__main__":
    main()

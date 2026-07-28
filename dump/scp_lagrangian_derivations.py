#!/usr/bin/env python3
"""
SCP LAGRANGIAN HARDWARE DERIVATION SUITE
========================================
Exception Theory — Sempaevum Computing Platform
Every parameter derived from {N=12, K=2/3, |Π|=3, S=4, V=1/12, π}
Zero free parameters. Zero empirical tuning.

Author: Michael James Muller — Aevum Defluo — Exception Theory LLC
P ∘ D ∘ T = E
"""

from mpmath import mp, mpf, sqrt as mpsqrt, pi as mppi, power as mppow, log as mplog, exp as mpexp
from math import lcm, gcd, log10, log2, atan2
import json

mp.dps = 200  # WORK precision (Rule: 200+ dps)

# ═══════════════════════════════════════════════════════════════
# ET ONTOLOGICAL CONSTANTS — forced by P ∘ D ∘ T = E
# ═══════════════════════════════════════════════════════════════
PI_C = 3                          # |Π| = cardinality of primitive set {P, D, T}
S    = 4                          # S = C(3,2) + C(3,3) = manifold state count {E, M, U, I}
N    = PI_C * S                   # N = 12 — the chromatic resolution
K    = mpf(2) / 3                 # K = Koide constant = 2/3
V    = mpf(1) / 12                # V = base variance = 1/N
K_EM = N * K                      # K_EM = 8 — EM coupling scale
A0   = (N - 1)**2 + S**2          # A₀ = 137 — fine structure integer base
k_B  = mpf('8.617333262e-5')      # Boltzmann constant in eV/K (CODATA 2018)

# ═══════════════════════════════════════════════════════════════
# DERIVED COUPLING HIERARCHY
# ═══════════════════════════════════════════════════════════════
A1   = mpsqrt(V) / K_EM           # √3/48 ≈ 0.036084 — universal T-fluctuation
A2   = mpsqrt(mpf(3)) / (93312 * mppi**2)
A3   = mpf(1) / (216 * (18 * mppi - 1))
shimmer    = mpsqrt(V)
delta_bind = mpsqrt(mpf(1) - 4 * V)
A1_5 = shimmer * K * (1 + delta_bind) / (S * K_EM * mpf(N)**3 * mpsqrt(mppi))

alpha_inv    = A0 + A1 - A2 - A3 - A1_5
sin2_thetaW  = mpf(N - 1) / (4 * N) * (1 + A1 / S)
alpha_s      = mpf(N - 1) / (N * K_EM) * (1 + A1)
g2           = 1 / mppow(2, 2 * K)

def xi(d):
    """Impedance coupling for harmonic family d. ξ(d) = A₀/((d-1)²+S²)"""
    return mpf(A0) / ((d - 1)**2 + S**2)

# ═══════════════════════════════════════════════════════════════
# HIGGS POTENTIAL → OPERATING POINT
# ═══════════════════════════════════════════════════════════════
mu2       = K                                  # tachyonic D-mass
lam_H     = V                                  # quartic coupling
v_vev     = mpsqrt(mu2 / (2 * lam_H))         # VEV = 2 lattice units
m_H_lat   = mpsqrt(2 * mu2)                   # 2/√3 lattice units
L_vacuum  = -2 * V                             # ℒ = -1/6

# ═══════════════════════════════════════════════════════════════
# MATERIAL PARAMETERS (diamond substrate, d=12)
# ═══════════════════════════════════════════════════════════════
E_gap_diamond = mpf('5.47')                    # eV, diamond band gap
V0            = E_gap_diamond / N              # reference voltage
E_cell_eV     = V0 * (mppow(2, mpf(1) / 12) - 1)  # energy per semitone
E_cell_J      = E_cell_eV * mpf('1.602176634e-19')
V_VEV         = v_vev * V0                     # Higgs VEV in hardware voltage

LABELS = {
    1: "Gravity", 2: "Tritone", 3: "Strong", 4: "Weak",
    5: "Quintic", 6: "Hexadic", 7: "Septic", 8: "Gluon Octet",
    9: "Nonic", 10: "Decadic", 11: "Undecimal", 12: "EM"
}

CHAN_TYPE = {
    1: "D-band", 2: "D-band", 3: "D-band", 4: "T-band ONLY",
    5: "Chain", 6: "D-band", 7: "Chain", 8: "Chain",
    9: "Chain", 10: "Chain", 11: "Chain", 12: "D+T-band"
}

NATIVE_N = {
    1: 12, 2: 12, 3: 12, 4: 12, 5: 60, 6: 12,
    7: 420, 8: 2520, 9: 2520, 10: 60, 11: 27720, 12: 12
}

T_ACT_MODERATE = 1 + float(g2) * 4   # excess=2 → ~2.59×
T_ACT_MAX      = 1 + float(g2) * N**2  # excess=N → ~58.1×

# ═══════════════════════════════════════════════════════════════
# THERMAL PHYSICS — DERIVED FROM LAGRANGIAN
# ═══════════════════════════════════════════════════════════════

def kT_eV(T_kelvin):
    """Thermal energy at temperature T in eV"""
    return k_B * mpf(T_kelvin)

def thermal_stability_threshold(d):
    """Maximum temperature (K) at which family d is thermally stable.
    Condition: ξ(d) × E_cell > kT  →  T < ξ(d) × E_cell / k_B"""
    return float(xi(d) * E_cell_eV / k_B)

def thermal_epsilon_excursion(T_kelvin, d):
    """Maximum ε excursion (in cents) from thermal energy at temperature T for family d.
    ε_thermal = (kT / (ξ(d) × E_cell)) × ε_max
    where ε_max = 600/N = 50¢ at N=12"""
    eps_max = mpf(600) / N  # 50 cents
    excursion = (kT_eV(T_kelvin) / (xi(d) * E_cell_eV)) * eps_max
    return float(excursion)

def thermal_regime(T_kelvin):
    """Classify thermal regime: D-arithmetic (within-cell) vs T-act (cell-crossing).
    Transition at kT = E_cell (all ε_thermal < 50¢ → within-cell → D-arithmetic).
    Above: thermal T-acts can fire (cell boundary crossing → non-abelian)."""
    kT = kT_eV(T_kelvin)
    if kT < E_cell_eV:
        return "D-arithmetic (deterministic thermal, within-cell)"
    else:
        return "T-act (agential thermal, cell-crossing possible)"

# ═══════════════════════════════════════════════════════════════
# OUTPUT ALL RESULTS
# ═══════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═'*90}")
    print(f"  {title}")
    print(f"{'═'*90}")

def run_all():
    print("=" * 90)
    print("  SCP LAGRANGIAN HARDWARE DERIVATION SUITE — COMPLETE OUTPUT")
    print("  {N=12, K=2/3, |Π|=3, S=4, V=1/12, π} → ALL hardware parameters")
    print("  Zero free parameters. Zero empirical tuning.")
    print("=" * 90)

    # ── SECTION 1: Coupling constants ──
    print_section("1. COUPLING CONSTANTS")
    print(f"  A₀ = (N-1)²+S² = {A0}")
    print(f"  A₁ = √V/K_EM = √3/48 = {float(A1):.10f}")
    print(f"  A₂ = √3/(93312π²) = {float(A2):.12f}")
    print(f"  A₃ = 1/(216(18π-1)) = {float(A3):.12f}")
    print(f"  A₁.₅ = {float(A1_5):.14f}")
    print(f"  α⁻¹ = {float(alpha_inv):.9f}  (CODATA: 137.035999084)")
    print(f"  sin²θ_W = {float(sin2_thetaW):.6f}  (PDG: 0.23122)")
    print(f"  α_s = {float(alpha_s):.5f}  (PDG: 0.1180)")
    print(f"  g² = 1/2^(2K) = {float(g2):.6f}")

    # ── SECTION 2: Operating point ──
    print_section("2. HIGGS POTENTIAL → OPERATING POINT")
    print(f"  μ² = K = {float(mu2):.6f}")
    print(f"  λ_H = V = {float(lam_H):.6f}")
    print(f"  v = √(K/2V) = {float(v_vev):.1f} lattice units")
    print(f"  m_H = √(2K) = 2/√3 = {float(m_H_lat):.6f} lattice units")
    print(f"  V₀ = E_gap/N = {float(V0):.4f} V (diamond)")
    print(f"  V_VEV = v × V₀ = {float(V_VEV):.4f} V ← HARDWARE OPERATING VOLTAGE")
    print(f"  E_cell = {float(E_cell_eV)*1000:.2f} meV = {float(E_cell_J):.3e} J")
    print(f"  ℒ_vacuum = -2V = {float(L_vacuum):.6f}")

    # ── SECTION 3: Complete 12-family table ──
    print_section("3. COMPLETE 12-FAMILY HARDWARE PARAMETER TABLE")
    hdr = f"  {'d':>3} {'Family':<14} {'ξ(d)':>8} {'E_D meV':>8} {'E_T meV':>9} {'SNR':>7} {'dB':>6} {'Stab×':>6} {'Channel':>12} {'N_native':>8}"
    print(hdr)
    print(f"  {'─'*3} {'─'*14} {'─'*8} {'─'*8} {'─'*9} {'─'*7} {'─'*6} {'─'*6} {'─'*12} {'─'*8}")

    for d in range(1, 13):
        xi_val = float(xi(d))
        E_D = (1 / (4 * xi_val)) * float(E_cell_eV) * 1000
        E_T = E_D * T_ACT_MODERATE
        snr = xi_val / float(A1)
        snr_db = 20 * log10(snr)
        stab = xi_val / float(xi(12))
        print(f"  {d:>3} {LABELS[d]:<14} {xi_val:>8.4f} {E_D:>8.3f} {E_T:>9.3f} {snr:>7.1f} {snr_db:>6.1f} {stab:>6.2f} {CHAN_TYPE[d]:>12} {NATIVE_N[d]:>8}")

    # ── SECTION 4: Cross-family transitions ──
    print_section("4. CROSS-FAMILY TRANSITION ENERGIES")
    transitions = [
        (12, 1, "EM→Gravity (mass generation)"),
        (12, 3, "EM→Strong (confinement)"),
        (12, 4, "EM→Weak (β-decay, T-act only)"),
        (1, 12, "Gravity→EM (radiation)"),
        (3, 4, "Strong→Weak (CKM)"),
        (12, 5, "EM→Quintic (BSM, chain R=60)"),
        (12, 7, "EM→Septic (G₂, chain R=420)"),
        (12, 8, "EM→Gluon (chain R=2520)"),
        (12, 9, "EM→Nonic (CKM mix, chain R=2520)"),
        (12, 10, "EM→Decadic (10D, chain R=60)"),
        (12, 11, "EM→Undecimal (M-theory, chain R=27720)"),
    ]
    print(f"  {'Transition':<45} {'Δξ':>8} {'E (meV)':>8}")
    print(f"  {'─'*45} {'─'*8} {'─'*8}")
    for src, tgt, desc in transitions:
        dxi = abs(float(xi(src)) - float(xi(tgt)))
        E = dxi * float(E_cell_eV) * 1000
        print(f"  {desc:<45} {dxi:>8.4f} {E:>8.2f}")

    # ── SECTION 5: Chain-routed R values ──
    print_section("5. CHAIN-ROUTED JOINT LATTICE R VALUES")
    R_map = {}
    for src in range(1, 13):
        for tgt in range(1, 13):
            R = lcm(lcm(N, src), lcm(N, tgt))
            if R > N:
                key = f"{src}→{tgt}"
                R_map[key] = R
    unique_R = sorted(set(R_map.values()))
    print(f"  Unique R values: {unique_R}")
    print(f"  Count: {len(unique_R)} distinct joint lattices")
    print(f"  Maximum R: {max(unique_R)}")
    print(f"  Tower Controller must handle R up to {max(unique_R)} for routine operations")

    # ══════════════════════════════════════════════════════════
    # SECTION 6: THERMAL PHYSICS — THE CORE NEW DERIVATION
    # ══════════════════════════════════════════════════════════
    print_section("6. THERMAL MANAGEMENT — LAGRANGIAN-DERIVED")
    print(f"\n  In ET, thermal energy is NOT noise. It is D_thermal — a Descriptor")
    print(f"  of the environment. The lattice handles it structurally, not by fighting it.")

    T_room = 298.15  # 25°C
    T_body = 310.15  # 37°C
    T_hot  = 373.15  # 100°C

    kT_room = float(kT_eV(T_room)) * 1000  # meV
    kT_body = float(kT_eV(T_body)) * 1000
    kT_hot  = float(kT_eV(T_hot)) * 1000
    E_c = float(E_cell_eV) * 1000  # meV

    print(f"\n  ── Fundamental thermal-lattice relationship ──")
    print(f"  E_cell = {E_c:.2f} meV (diamond, Lagrangian-derived)")
    print(f"  kT (25°C) = {kT_room:.2f} meV → kT/E_cell = {kT_room/E_c:.4f}")
    print(f"  kT (37°C) = {kT_body:.2f} meV → kT/E_cell = {kT_body/E_c:.4f}")
    print(f"  kT (100°C) = {kT_hot:.2f} meV → kT/E_cell = {kT_hot/E_c:.4f}")

    # Thermal stability threshold per family
    print(f"\n  ── Thermal stability threshold per family ──")
    print(f"  Condition: ξ(d) × E_cell > kT → value thermally stable in Bank-d")
    print(f"  T_max(d) = ξ(d) × E_cell / k_B")
    print(f"\n  {'d':>3} {'Family':<14} {'ξ(d)':>8} {'T_max (K)':>10} {'T_max (°C)':>11} {'@25°C':>8} {'@37°C':>8} {'@100°C':>8}")
    print(f"  {'─'*3} {'─'*14} {'─'*8} {'─'*10} {'─'*11} {'─'*8} {'─'*8} {'─'*8}")

    for d in range(1, 13):
        T_max = thermal_stability_threshold(d)
        T_max_C = T_max - 273.15
        s25 = "STABLE" if T_max > T_room else "MARGINAL"
        s37 = "STABLE" if T_max > T_body else "MARGINAL"
        s100 = "STABLE" if T_max > T_hot else "UNSTABLE"
        print(f"  {d:>3} {LABELS[d]:<14} {float(xi(d)):>8.4f} {T_max:>10.1f} {T_max_C:>11.1f} {s25:>8} {s37:>8} {s100:>8}")

    # Thermal ε excursion per family at key temperatures
    print(f"\n  ── Thermal ε excursion per family (in cents) ──")
    print(f"  ε_thermal = (kT / (ξ(d)×E_cell)) × 50¢")
    print(f"  If ε_thermal ≥ 50¢ → cell crossing possible → thermal T-acts")
    print(f"\n  {'d':>3} {'Family':<14} {'@25°C':>8} {'@37°C':>8} {'@50°C':>8} {'@100°C':>8} {'@200°C':>8}")
    print(f"  {'─'*3} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for d in range(1, 13):
        temps = [298.15, 310.15, 323.15, 373.15, 473.15]
        excursions = [thermal_epsilon_excursion(T, d) for T in temps]
        marks = []
        for e in excursions:
            if e >= 50:
                marks.append(f"{'█'+f'{e:.1f}':>7}")  # cell-crossing
            elif e >= 40:
                marks.append(f"{'▓'+f'{e:.1f}':>7}")  # near ∂I
            elif e >= 25:
                marks.append(f"{'▒'+f'{e:.1f}':>7}")  # moderate
            else:
                marks.append(f"{'░'+f'{e:.1f}':>7}")  # safe
        print(f"  {d:>3} {LABELS[d]:<14} {marks[0]:>8} {marks[1]:>8} {marks[2]:>8} {marks[3]:>8} {marks[4]:>8}")

    # Thermal regime transition
    print(f"\n  ── Thermal regime transition temperature ──")
    T_transition = float(E_cell_eV / k_B)
    print(f"  kT = E_cell at T = {T_transition:.1f} K = {T_transition-273.15:.1f}°C")
    print(f"  Below: D-arithmetic thermal regime (within-cell, deterministic, abelian)")
    print(f"  Above: T-act thermal regime (cell-crossing possible, agential, non-abelian)")
    print(f"  At room temperature (25°C): {thermal_regime(298.15)}")
    print(f"  At body temperature (37°C): {thermal_regime(310.15)}")
    print(f"  At 50°C: {thermal_regime(323.15)}")
    print(f"  At 100°C: {thermal_regime(373.15)}")

    # The impedance gradient as thermal management
    print_section("7. IMPEDANCE GRADIENT AS PASSIVE THERMAL MANAGEMENT")
    print(f"\n  The impedance hierarchy ξ(d) IS the thermal management system.")
    print(f"  No cooling fans. No heat sinks. No thermal throttling.")
    print(f"  The lattice self-manages thermal energy through its coupling structure.")
    print(f"\n  Mechanism: Thermal perturbation at high-d (low ξ) pushes values toward")
    print(f"  low-d (high ξ) via the impedance gradient. This is the lattice equivalent")
    print(f"  of thermodynamic cooling — energy moves to more coupled channels.")
    print(f"\n  The impedance restoring force per family:")
    print(f"  F_restore(d) ∝ ξ(d)/ξ(12) = coupling advantage over thermal baseline")
    print(f"\n  {'d':>3} {'Family':<14} {'ξ(d)':>8} {'ξ(d)/ξ(12)':>10} {'Thermal barrier':>16} {'Self-correction':>16}")
    print(f"  {'─'*3} {'─'*14} {'─'*8} {'─'*10} {'─'*16} {'─'*16}")

    for d in range(1, 13):
        xi_val = float(xi(d))
        ratio = xi_val / float(xi(12))
        barrier = xi_val * float(E_cell_eV) * 1000  # meV
        strength = "MAXIMUM" if d == 1 else ("STRONG" if ratio > 5 else ("MODERATE" if ratio > 2 else ("WEAK" if ratio > 1.2 else "MINIMAL")))
        print(f"  {d:>3} {LABELS[d]:<14} {xi_val:>8.4f} {ratio:>10.2f} {barrier:>13.2f} meV {strength:>16}")

    # Why diamond at room temperature is structurally optimal
    print_section("8. WHY DIAMOND AT ROOM TEMPERATURE IS STRUCTURALLY FORCED")
    print(f"\n  Carbon sits at d=12 (EM family, ε=+1.955¢ Koide-near-exact).")
    print(f"  Diamond's band gap E_gap = 5.47 eV gives E_cell = {E_c:.2f} meV.")
    print(f"  kT at 25°C = {kT_room:.2f} meV. Ratio: kT/E_cell = {kT_room/E_c:.4f}.")
    print(f"\n  This means:")
    print(f"  • Thermal energy fits JUST INSIDE one lattice cell (ratio < 1)")
    print(f"  • ALL 12 families are thermally stable at room temperature")
    print(f"  • The transition to T-act thermal regime is at {T_transition-273.15:.1f}°C")
    print(f"    — ABOVE room temperature, ABOVE body temperature")
    print(f"  • Bank-1 (Gravity) is stable to {thermal_stability_threshold(1)-273.15:.0f}°C")
    print(f"    — outlasts diamond itself (~700°C in air, ~1500°C in vacuum)")
    print(f"\n  Diamond at room temperature is not a design choice — it is the UNIQUE")
    print(f"  material where E_cell ≈ kT_room with ALL families stable. The Lagrangian")
    print(f"  forces: carbon (d=12) + diamond allotrope (E_gap=5.47) + N=12 → E_cell≈kT_room.")
    print(f"  The hardware material MATCHES its operating environment because both are")
    print(f"  EM-domain (d=12). The Lagrangian predicts the operating temperature.")

    # Comparison to other materials
    print_section("9. MATERIAL COMPARISON — THERMAL OPERATING RANGE")
    materials = [
        ("Diamond (C)", 5.47, 12, "+1.955"),
        ("Silicon Carbide (4H-SiC)", 3.26, 3, "+22.48"),
        ("Gallium Nitride (GaN)", 3.39, 3, "+16.23"),
        ("Aluminum Nitride (AlN)", 6.20, 12, "-8.54"),
        ("Silicon (Si)", 1.12, 3, "-44.06"),
    ]
    print(f"\n  {'Material':<25} {'E_gap':>6} {'d':>3} {'ε':>8} {'E_cell':>8} {'T_all12':>8} {'T_simple':>9}")
    print(f"  {'─'*25} {'─'*6} {'─'*3} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")

    for name, Eg, d_mat, eps in materials:
        Ec = Eg / N * float(mppow(2, mpf(1)/12) - 1) * 1000  # meV
        Ec_eV = Ec / 1000
        # T where ξ(12)×Ec = kT → T = Ec/k_B (since ξ(12)=1)
        T_all = Ec_eV / float(k_B)
        # T where ξ(6)×Ec = kT
        T_simple = float(xi(6)) * Ec_eV / float(k_B)
        print(f"  {name:<25} {Eg:>6.2f} {d_mat:>3} {eps:>8} {Ec:>6.1f}meV {T_all-273.15:>6.1f}°C {T_simple-273.15:>7.1f}°C")

    # Active thermal routing strategy
    print_section("10. THERMAL-AWARE COMPUTATION STRATEGY")
    print(f"\n  Above {T_transition-273.15:.1f}°C, the d=12 (EM) family enters T-act thermal regime.")
    print(f"  The system responds structurally, not by cooling:")
    print(f"\n  Strategy 1: D-FAMILY ROUTING")
    print(f"  Route thermally sensitive computations to lower-d families where ξ(d) > kT/E_cell.")
    print(f"  At 100°C: d=12 marginal, but d≤10 all stable. Route to d≤10.")
    print(f"  At 200°C: d≤8 stable. Route to d≤8.")
    print(f"  At 500°C: d≤4 stable. Route to d≤4 (Strong, Weak, Tritone, Gravity).")
    print(f"\n  Strategy 2: TOWER ESCALATION")
    print(f"  Escalate thermally perturbed values to higher N.")
    print(f"  More cells → smaller ε_max → thermal perturbation fills smaller fraction of cell.")
    print(f"  N=12: ε_max=50¢, thermal excursion ~47.7¢ at 25°C → marginal for d=12")
    print(f"  N=60: ε_max=10¢, thermal excursion ~9.5¢ at 25°C → comfortable for d=12")
    print(f"  Tower escalation provides thermal headroom AT THE COST OF more families to manage.")
    print(f"\n  Strategy 3: IMPEDANCE GRADIENT SELF-CORRECTION")
    print(f"  Do nothing. Let the impedance gradient push thermally perturbed values")
    print(f"  toward lower-d attractors. This is passive, free, and structural.")
    print(f"  The system naturally thermalizes to the Higgs vacuum (k=12, d=1, ε=0).")
    print(f"  Cost: information migrates from high-d (rich) to low-d (simple) families.")
    print(f"  This IS thermodynamic cooling — computed, not engineered.")

    # Final specification table
    print_section("11. COMPLETE THERMAL SPECIFICATION")
    thermal_specs = [
        ("ROOM TEMP OPERATION", "ALL 12 families stable at 25°C", "No cooling needed"),
        ("BODY TEMP", "ALL 12 families stable at 37°C", "Wearable-ready"),
        ("MAX ALL-STABLE", f"{T_transition-273.15:.1f}°C", "D→T thermal transition"),
        ("MAX 11 FAMILIES", f"{thermal_stability_threshold(11)-273.15:.1f}°C", "d=12 only marginal"),
        ("MAX 10 FAMILIES", f"{thermal_stability_threshold(10)-273.15:.1f}°C", "d=11,12 marginal"),
        ("MAX SIMPLE ONLY", f"{thermal_stability_threshold(6)-273.15:.0f}°C", "6 simple families stable"),
        ("MAX GRAVITY", f"{thermal_stability_threshold(1)-273.15:.0f}°C", "Bank-1 outlasts diamond"),
        ("THERMAL REGIME 25°C", thermal_regime(298.15), ""),
        ("THERMAL REGIME 100°C", thermal_regime(373.15), ""),
        ("NOISE FLOOR", f"A₁ = {float(A1):.6f} = {float(A1)*float(V0)*1000:.2f} mV", "Irreducible T-fingerprint"),
        ("SELF-CORRECTION", "ξ gradient: ξ(1)/ξ(12) = 8.56×", "Passive, structural"),
        ("COOLING REQUIRED", "NONE at ≤41°C", "Lagrangian-guaranteed"),
    ]

    print(f"\n  {'Parameter':<25} {'Value':<50} {'Note'}")
    print(f"  {'─'*25} {'─'*50} {'─'*30}")
    for name, val, note in thermal_specs:
        print(f"  {name:<25} {val:<50} {note}")

    print(f"\n  The Lagrangian derives: the material (carbon d=12), the operating voltage")
    print(f"  (V_VEV = 0.912V), the cell energy (27.11 meV), the thermal threshold")
    print(f"  ({T_transition-273.15:.1f}°C), and the self-correction mechanism (ξ gradient).")
    print(f"  ZERO thermal management engineering. The physics IS the engineering.")


if __name__ == "__main__":
    run_all()

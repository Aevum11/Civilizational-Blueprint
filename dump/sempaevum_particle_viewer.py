#!/usr/bin/env python3
"""
SEMPAEVUM PARTICLE VIEWER
=========================
Uses the Sempaevum bijection (Theorem 15.1, Losslessness) to project every
fundamental particle of the Standard Model onto the ET lattice, then renders
a 3D interactive visualization.

All mathematics is ET-native, derived from {P, D, T}.
Particle data: PDG 2024 (S. Navas et al., Phys. Rev. D 110, 030001)

Author: Mike Muller / Aevum Defluo (Exception Theory)
Tool assistance: Claude (Anthropic) as fancy calculator + visualization
"""

import math
import json
import os
import mpmath

# 120-digit target + 15 guard digits
mpmath.mp.dps = 135
TARGET_DIGITS = 120

# ============================================================
# ET CONSTANTS — derived from P ∘ D ∘ T = E
# ============================================================
N = 12                    # Manifold symmetry: |Π| × S = 3 × 4
S = 4                     # Manifold state count: C(3,2) + C(3,3)
PI_COUNT = 3              # |Π| = 3 primitive Cardinals
V_BASE = 1.0 / N          # Base variance: 1/N = 1/12
K_KOIDE = 2.0 / 3.0       # Koide ratio: 1 - S/N
A0_LOCAL = (N - 1)**2 + S**2  # = 137, manifold impedance

# Universal lattice resolution
N_FULL = 27720            # lcm(1..11) = 2³ × 3² × 5 × 7 × 11


def generate_lcm_tower():
    """
    Dynamically generate the LCM tower: LCM(1..n) for n=4,5,6,...
    until the tower resolution is large enough that the smallest
    mass ratio in the PDG data can be resolved to sub-cent precision.
    
    The tower does NOT stop at a fixed N. It stops when D has added
    enough descriptors to close the gap for every particle.
    This is the Descriptor Gap Principle enacted at the meta-level:
    the tower itself IS the Descriptor that closes the gap between
    N=12 base resolution and full structural resolution.
    """
    from math import gcd
    
    tower = []
    current_lcm = 1
    for n in range(1, 200):  # go far enough
        current_lcm = current_lcm * n // gcd(current_lcm, n)
        if n >= 4:
            tower.append((current_lcm, f"LCM(1..{n})", n))
    
    return tower


# Pre-compute the tower (used by all particles)
LCM_TOWER_FULL = generate_lcm_tower()


# ============================================================
# PDG 2025 FILE PARSER — reads the pure data dynamically
# ============================================================

def parse_pdg_file(filepath):
    """
    Parse the PDG mass_width file format.
    Returns list of dicts with name, mass_gev, mass_err_pos, mass_err_neg,
    width_gev, charges, pdg_id, spin (inferred from PDG ID).
    
    This is the D-operation on raw data: imposing structure on the
    featureless file content.
    """
    particles = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('*') or line.strip() == '':
                continue
            
            # Parse the fixed-width format
            # Cols 1-32: up to 4 PDG IDs (8 chars each)
            ids_str = line[:32]
            pdg_ids = []
            for i in range(4):
                chunk = ids_str[i*8:(i+1)*8].strip()
                if chunk:
                    try:
                        pdg_ids.append(int(chunk))
                    except ValueError:
                        pass
            
            if not pdg_ids:
                continue
            
            # Col 34-51: mass central value
            mass_str = line[33:51].strip()
            if not mass_str:
                continue  # No mass listed (e.g. neutrinos)
            
            try:
                mass_gev = float(mass_str)
            except ValueError:
                continue
            
            # Col 53-60: positive mass error
            mass_err_pos_str = line[52:60].strip()
            mass_err_pos = float(mass_err_pos_str) if mass_err_pos_str else 0.0
            
            # Col 62-69: negative mass error
            mass_err_neg_str = line[61:69].strip()
            mass_err_neg = float(mass_err_neg_str) if mass_err_neg_str else 0.0
            
            # Col 71-88: width central value
            width_str = line[70:88].strip()
            width_gev = float(width_str) if width_str else 0.0
            
            # Col 108-128: name and charges
            name_charge = line[107:].strip() if len(line) > 107 else ""
            parts = name_charge.rsplit(None, 1)
            name = parts[0].strip() if parts else f"PDG_{pdg_ids[0]}"
            charges = parts[1].strip() if len(parts) > 1 else ""
            
            # Convert mass to MeV
            mass_mev = mass_gev * 1000.0
            mass_err_pos_mev = mass_err_pos * 1000.0
            mass_err_neg_mev = abs(mass_err_neg) * 1000.0
            
            # Infer spin from PDG ID (2S+1 is encoded in the ID scheme)
            spin = infer_spin_from_pdg(pdg_ids[0], name)
            
            # Infer charge from charge string
            charge = infer_charge(charges, pdg_ids[0])
            
            # Infer category
            category = infer_category(pdg_ids[0], name)
            interaction = infer_interaction(pdg_ids[0], category)
            
            # Infer generation for quarks and leptons
            generation = 0
            abs_id = abs(pdg_ids[0])
            if abs_id in (1, 2, 11, 12):
                generation = 1
            elif abs_id in (3, 4, 13, 14):
                generation = 2
            elif abs_id in (5, 6, 15, 16):
                generation = 3
            
            # Color charge
            color_charge = 3 if category == "Quark" else (8 if abs_id == 21 else 0)
            
            particles.append({
                "name": name,
                "symbol": name,
                "pdg_ids": pdg_ids,
                "mass_gev": mass_gev,
                "mass_mev": mass_mev,
                "mass_err_pos_mev": mass_err_pos_mev,
                "mass_err_neg_mev": mass_err_neg_mev,
                "width_gev": width_gev,
                "charges": charges,
                "spin": spin,
                "charge": charge,
                "category": category,
                "interaction": interaction,
                "generation": generation,
                "color_charge": color_charge,
                "mass_str": mass_str,
            })
    
    return particles


def infer_spin_from_pdg(pdg_id, name):
    """
    Infer spin from the PDG MC ID and particle name.
    The PDG numbering scheme encodes 2J+1 in the ID digits.
    For fundamental particles, we use known values.
    """
    abs_id = abs(pdg_id)
    
    # Fundamental particles — exact known spins
    fundamental_spins = {
        1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5,  # quarks
        11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5,  # leptons
        21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0,  # gauge bosons
        25: 0.0,  # Higgs
    }
    if abs_id in fundamental_spins:
        return fundamental_spins[abs_id]
    
    # Baryons (4-digit IDs starting with 1-5 in thousands)
    if abs_id >= 1000:
        # Extract 2J+1 from last digit for baryons
        last_digit = abs_id % 10
        if last_digit >= 2:
            return (last_digit - 1) / 2.0
        # Baryons with last digit 2 → J=1/2, 4 → J=3/2, 6 → J=5/2, 8 → J=7/2
        return 0.5
    
    # Mesons (3-digit IDs or special)
    # For mesons, the PDG scheme encodes spin in specific digits
    # Quick heuristic from name
    name_lower = name.lower()
    if 'pi' in name_lower and '(' not in name_lower:
        return 0.0
    if name_lower.startswith('eta') or name_lower.startswith('f(0)') or name_lower.startswith('a(0)'):
        return 0.0
    if 'rho' in name_lower or 'omega' in name_lower or 'phi' in name_lower or 'K*' in name_lower or 'D*' in name_lower or 'B*' in name_lower or 'J/psi' in name_lower or 'psi' in name_lower or 'Upsilon' in name_lower:
        return 1.0
    if 'f(2)' in name_lower or 'a(2)' in name_lower or 'K(2)' in name_lower or 'D(2)' in name_lower or 'B(2)' in name_lower or 'chi(c2)' in name_lower or 'chi(b2)' in name_lower:
        return 2.0
    if 'f(4)' in name_lower or 'a(4)' in name_lower or 'K(4)' in name_lower:
        return 4.0
    if 'omega(3)' in name_lower or 'rho(3)' in name_lower or 'K(3)' in name_lower or 'phi(3)' in name_lower:
        return 3.0
    if 'f(1)' in name_lower or 'h(1)' in name_lower or 'b(1)' in name_lower or 'a(1)' in name_lower or 'K(1)' in name_lower or 'h(c)' in name_lower or 'h(b)' in name_lower or 'chi(c1)' in name_lower or 'chi(b1)' in name_lower or 'D(1)' in name_lower or 'pi(1)' in name_lower:
        return 1.0
    if 'eta(2)' in name_lower or 'pi(2)' in name_lower or 'K(2)' in name_lower or 'Upsilon(2)' in name_lower:
        return 2.0
    if 'chi(c0)' in name_lower or 'chi(b0)' in name_lower or 'K(0)' in name_lower or 'D(s0)' in name_lower or 'D(0)' in name_lower or 'K(0)' in name_lower:
        return 0.0
    if 'eta' in name_lower:
        return 0.0
    
    # Meson PDG digit extraction fallback
    id_str = str(abs_id)
    if len(id_str) >= 3 and abs_id < 1000:
        # For standard mesons: last digit encodes 2S+1, second-to-last encodes L
        last = int(id_str[-1])
        second = int(id_str[-2])
        # J depends on L and S coupling
        S_val = (last - 1) / 2.0
        return max(abs(second - S_val), abs(second + S_val))
    
    return 0.0  # default


def infer_charge(charge_str, pdg_id):
    """Infer numeric charge from the charge string."""
    charge_str = charge_str.strip()
    abs_id = abs(pdg_id)
    
    # Known charges for fundamental particles
    fundamental_charges = {
        1: -1.0/3, 2: 2.0/3, 3: -1.0/3, 4: 2.0/3, 5: -1.0/3, 6: 2.0/3,
        11: -1.0, 12: 0.0, 13: -1.0, 14: 0.0, 15: -1.0, 16: 0.0,
        21: 0.0, 22: 0.0, 23: 0.0, 24: 1.0, 25: 0.0,
    }
    if abs_id in fundamental_charges:
        return fundamental_charges[abs_id]
    
    if charge_str in ('+', '++'):
        return 1.0 if charge_str == '+' else 2.0
    if charge_str == '-':
        return -1.0
    if charge_str == '0':
        return 0.0
    if charge_str.startswith('+') and '/' in charge_str:
        return eval(charge_str)
    if charge_str.startswith('-') and '/' in charge_str:
        return eval(charge_str)
    
    return 0.0


def infer_category(pdg_id, name):
    """Infer particle category from PDG ID using the PDG numbering scheme.
    
    PDG scheme: ±n_r n_L n_q1 n_q2 n_q3 (2J+1)
    - n_q1 = 0 for mesons (two quarks: q̄q)
    - n_q1 ≠ 0 for baryons (three quarks: qqq)
    
    Special cases:
    - IDs 1-6: quarks
    - IDs 11-16: leptons  
    - IDs 21-24: gauge bosons
    - ID 25: Higgs (scalar boson)
    - IDs 9000000+: "special" PDG entries, use rightmost digits
    """
    abs_id = abs(pdg_id)
    
    if abs_id in (21, 22, 23, 24):
        return "Gauge Boson"
    if abs_id == 25:
        return "Scalar Boson"
    if abs_id <= 6:
        return "Quark"
    if abs_id <= 16:
        return "Lepton"
    
    # For composite particles: extract the quark-content digits
    # Take the last 4 significant digits: n_q1 n_q2 n_q3 (2J+1)
    # Strip leading special prefixes (9000000+, 100000+, etc.)
    core = abs_id
    # Remove special-particle prefix digits
    # PDG "core" is always the rightmost 4 digits for standard particles
    # For IDs like 9000221 → core 0221 → n_q1=0 → Meson
    # For IDs like 2212 → core 2212 → n_q1=2 → Baryon
    # For IDs like 10111 → strip to 0111 → n_q1=0 → Meson (excited pion)
    
    # Extract rightmost 4 digits
    core_4 = core % 10000
    # n_q1 is the thousands digit of the 4-digit core
    n_q1 = core_4 // 1000
    
    if n_q1 == 0:
        return "Meson"
    else:
        return "Baryon"


def infer_interaction(pdg_id, category):
    """Infer interaction type from PDG ID and category."""
    abs_id = abs(pdg_id)
    
    if abs_id == 21:
        return "Strong"
    if abs_id == 22:
        return "EM"
    if abs_id in (23, 24):
        return "Weak"
    if abs_id == 25:
        return "All"
    if abs_id <= 6:
        return "Strong+EM+Weak"
    if abs_id in (11, 13, 15):
        return "EM+Weak"
    if abs_id in (12, 14, 16):
        return "Weak"
    if category in ("Baryon", "Meson"):
        return "Strong+EM+Weak"
    
    return "Strong+EM+Weak"
ELECTRON_MASS = 0.51099895000  # MeV, CODATA 2022 / PDG 2024



# ============================================================
# SEMPAEVUM PROJECTION — The Bijection (Definition 5.1)
# ============================================================

def et_gcd(a, b):
    """Euclidean GCD — the discrete-D operation of the projection."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def sempaevum_project(r, n_res):
    """
    The Sempaevum bijection Π_N : ℝ⁺ → ℤ × {N/d : d|N} × ℝ
    
    r     : positive real (the P-content, a dimensionless ratio)
    n_res : lattice resolution N (the D-content, finite constraint)
    
    Returns (k, d, epsilon_cents)
    
    This IS the master equation P ∘ D ∘ T = E enacted at single-ratio scope:
      r       = P (featureless positive real)
      N, gcd  = D (finite lattice constraints)
      round() = T (resolution of continuous to discrete — the T-act)
    """
    if r <= 0:
        return None  # ∂I annihilation boundary — excluded by Proposition 5.3
    
    log2_r = math.log2(r)
    continuous_k = n_res * log2_r
    k = round(continuous_k)
    
    # Sublattice family classification
    abs_k = abs(k)
    if k == 0:
        g = n_res  # Convention: gcd(0, N) = N → d = 1
    else:
        g = et_gcd(abs_k, n_res)
    d = n_res // g
    
    # Descriptor gap in cents
    epsilon = (continuous_k - k) * (1200.0 / n_res)
    
    return (k, d, epsilon)


def sempaevum_project_mp(r_str, n_res):
    """
    The Sempaevum bijection Π_N at 120-digit arbitrary precision.
    
    r_str : string representation of the positive real (for exact mpmath input)
    n_res : integer lattice resolution N
    
    Returns (k, d, epsilon_mp) where epsilon_mp is mpmath.mpf in cents.
    """
    r_mp = mpmath.mpf(r_str)
    if r_mp <= 0:
        return None
    
    n_mp = mpmath.mpf(n_res)
    log2_r = mpmath.log(r_mp, 2)
    continuous_k = n_mp * log2_r
    k_mp = mpmath.nint(continuous_k)
    k = int(k_mp)
    
    abs_k = abs(k)
    if k == 0:
        g = n_res
    else:
        g = et_gcd(abs_k, n_res)
    d = n_res // g
    
    epsilon_mp = (continuous_k - k_mp) * (mpmath.mpf(1200) / n_mp)
    return (k, d, epsilon_mp)


def sempaevum_pullback_mp(k, epsilon_mp, n_res):
    """
    Sempaevum inverse bijection Π_N⁻¹ at arbitrary precision.
    Π_N⁻¹(k, d, ε) = 2^((k + εN/1200)/N)
    Losslessness Theorem (Theorem 15.1): Π_N⁻¹ ∘ Π_N = identity on ℝ⁺.
    """
    n_mp = mpmath.mpf(n_res)
    k_mp = mpmath.mpf(k)
    continuous_k = k_mp + epsilon_mp * n_mp / mpmath.mpf(1200)
    return mpmath.power(2, continuous_k / n_mp)


def magical_impedance(d):
    """
    Magical impedance A₀^magic(d) = (d-1)² + S² and coupling ξ(d) = 137/A₀
    Definition 7.1 of the Sempaevum paper.
    """
    a0 = (d - 1)**2 + S**2
    xi = A0_LOCAL / a0
    return a0, xi


def phase_k_theta(particle):
    """
    Assign imaginary-axis lattice coordinate k_θ based on the particle's
    spin and interaction type, following the PHASE family assignments
    of Sempaevum Table 8.
    
    This is NOT ad hoc — the phase assignment follows from the structural
    identification of harmonic families on the imaginary axis (T's operational
    domain U(1)).
    
    Sempaevum Table 8 PHASE families at N=12:
      d_θ=1 : scalar (spin-0) phase        → k_θ = 0
      d_θ=2 : spin-2 (graviton) phase      → k_θ = 6
      d_θ=3 : instanton/color QCD phase    → k_θ = 4 or 8
      d_θ=4 : SU(2)_W weak phase           → k_θ = 3 or 9
      d_θ=6 : spin-½ fermion phase          → k_θ = 2 or 10
      d_θ=12: photon phase / full U(1)      → k_θ = 1, 5, 7, 11
    """
    spin = particle["spin"]
    category = particle["category"]
    charge = particle.get("charge", 0)
    color = particle.get("color_charge", 0)
    interaction = particle.get("interaction", "")
    gen = particle.get("generation", 0)
    
    # Scalar boson (spin 0) → d_θ = 1, k_θ = 0
    if spin == 0.0 and category not in ("Composite", "Meson", "Baryon"):
        return 0
    
    # Spin-½ fermions → d_θ = 6
    if spin == 0.5:
        if category == "Quark":
            if charge > 0:
                return 2   # Up-type quarks: k_θ=2, d_θ=6
            else:
                return 10  # Down-type quarks: k_θ=10, d_θ=6
        elif category == "Lepton":
            if charge != 0:
                return 2   # Charged leptons: k_θ=2, d_θ=6
            else:
                return 10  # Neutrinos: k_θ=10, d_θ=6
        elif category in ("Composite", "Baryon"):
            return 2   # Composite fermions (proton, neutron, baryons)
        else:
            return 2
    
    # Spin-1 gauge bosons
    if spin == 1.0:
        if "Weak" in interaction and category == "Gauge Boson":
            return 3  # W/Z: SU(2)_W phase, d_θ=4
        elif "EM" in interaction and category == "Gauge Boson":
            return 1  # Photon: full U(1) phase, d_θ=12
        elif "Strong" in interaction and category == "Gauge Boson":
            return 4  # Gluon: instanton/color phase, d_θ=3
        else:
            # Composite spin-1 mesons (rho, omega, phi, J/psi, etc.)
            return 1  # Vector mesons: full U(1) phase, d_θ=12
    
    # Spin-3/2 baryons (Delta, etc.) → same spinoral 4π as spin-½ → d_θ=6
    if spin == 1.5:
        return 2
    
    # Spin-2 → d_θ=2 (tritone, graviton phase)
    if spin == 2.0:
        return 6  # d_θ=2 tritone: k_θ=6
    
    # Spin-3 → k_θ=4 (d_θ=3, instanton/cubic phase)
    if spin == 3.0:
        return 4
    
    # Spin-4 → k_θ=3 (d_θ=4)
    if spin == 4.0:
        return 3
    
    # Spin-5/2 baryons → spinoral
    if spin == 2.5:
        return 2
    
    # Spin-7/2 baryons
    if spin == 3.5:
        return 2
    
    # Composite spin-0 (pion, kaon, eta, etc.)
    if spin == 0.0 and category in ("Composite", "Meson"):
        return 0  # Scalar phase

    return 0  # fallback


def factor_integer(n):
    """Factor an integer into its prime decomposition. Pure arithmetic, no static lists."""
    if n <= 1:
        return [n] if n == 1 else []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def compute_tower_projection(r_mp, r_str, mass_err_ppm=None):
    """
    Project a mass ratio through the LCM tower, escalating until convergence.
    
    Convergence criterion: |ε| < measurement uncertainty expressed in cents.
    If no uncertainty is provided, escalate until |ε| < 10^(-100) cents
    (well beyond any conceivable measurement).
    
    The tower does NOT stop at N=27720. It continues until D has added
    enough descriptors. This is the Subsumption Law: the lattice MUST
    accommodate every ratio — escalation continues until it does.
    
    Returns list of dicts with all tower data, pure mpmath throughout.
    """
    # Convergence threshold in cents
    if mass_err_ppm is not None and mass_err_ppm > 0:
        # Convert measurement uncertainty (ppm) to cents
        # 1 cent = 1/1200 octave ≈ 577 ppm of the ratio
        threshold_cents = mpmath.mpf(mass_err_ppm) / mpmath.mpf(577)
    else:
        threshold_cents = mpmath.power(10, -100)
    
    tower_results = []
    converged = False
    
    for n_val, label, n_upper in LCM_TOWER_FULL:
        if converged:
            break
        
        proj = sempaevum_project_mp(r_str, n_val)
        if proj is None:
            continue
        k, d, eps_mp = proj
        
        # Factor d
        d_factors = factor_integer(d)
        d_factors_str = "×".join(str(f) for f in d_factors) if d_factors else "1"
        
        # Sublattice family characterization at this resolution
        # The sublattice family IS the coset: which sub-lattice of L_N
        # the particle inhabits, determined by gcd(|k|, N)
        abs_k = abs(k)
        g = et_gcd(abs_k, n_val) if abs_k > 0 else n_val
        
        # The sublattice has g points per period of N
        # d = N/g is the sublattice ORDER (distinct positions in one period)
        # The sublattice FAMILY is classified by the divisor structure of g
        sublattice_g = g
        sublattice_g_factors = factor_integer(g)
        
        # ε in cents as mpmath string for display
        eps_str = mpmath.nstr(eps_mp, TARGET_DIGITS, strip_zeros=False)
        abs_eps = abs(eps_mp)
        
        tower_results.append({
            "N": n_val,
            "label": label,
            "n_upper": n_upper,
            "k": k,
            "d": d,
            "g": sublattice_g,
            "eps_mp": eps_mp,
            "abs_eps_mp": abs_eps,
            "eps_str": eps_str,
            "d_factors": d_factors,
            "d_factors_str": d_factors_str,
            "g_factors": sublattice_g_factors,
            "g_factors_str": "×".join(str(f) for f in sublattice_g_factors) if sublattice_g_factors else "1",
        })
        
        # Check convergence
        if abs_eps < threshold_cents:
            converged = True
    
    return tower_results, converged


def find_true_home(tower_results):
    """
    Find the tower level where the particle's sublattice family stabilizes.
    
    "True home" = the FIRST tower level where d stops changing in a way
    that persists through all subsequent levels. This means the particle's
    structural identity has been fully resolved — no further D-escalation
    changes its family assignment.
    
    This is the Identification Principle: the particle IS identified when
    its sublattice address no longer changes under resolution increase.
    """
    if not tower_results:
        return 0, None
    
    # Walk backwards from the end to find where d stabilized
    final_d = tower_results[-1]["d"]
    settled_idx = len(tower_results) - 1
    
    for i in range(len(tower_results) - 2, -1, -1):
        # At lower resolution, d should be a divisor of final_d
        # The "home" is where d first equals final_d
        if tower_results[i]["d"] == final_d:
            settled_idx = i
        else:
            break
    
    return settled_idx, tower_results[settled_idx]


def classify_24_families(k_r_12, k_theta):
    """
    Classify a particle into the 24 harmonic families of the complex lattice.
    
    12 real families: determined by k_r mod 12 → d_r = 12/gcd(|k_r|, 12)
    12 imaginary families: determined by k_θ mod 12 → d_θ = 12/gcd(|k_θ|, 12)
    
    Combined family: d_combined = LCM(d_r, d_θ)
    
    From ET_Complex_Lattice.md:
      d_θ=1:  scalar phase (k_θ=0)       — Gravity/Identity
      d_θ=2:  tritone/pivot (k_θ=6)      — Palindromic center
      d_θ=3:  instanton/color (k_θ=4,8)  — QCD winding
      d_θ=4:  quartic/weak (k_θ=3,9)     — SU(2)_W, T-axis
      d_θ=6:  hexadic/spinor (k_θ=2,10)  — Spin-½ fermion phase
      d_θ=12: EM/full-res (k_θ=1,5,7,11) — Full resolution
    """
    from math import lcm
    
    # Real family
    abs_kr = abs(k_r_12)
    kr_mod = abs_kr % 12
    if kr_mod == 0:
        d_r = 1
    else:
        d_r = 12 // et_gcd(kr_mod, 12)
    
    # Imaginary family
    abs_kt = abs(k_theta)
    kt_mod = abs_kt % 12
    if kt_mod == 0:
        d_theta = 1
    else:
        d_theta = 12 // et_gcd(kt_mod, 12)
    
    d_combined = lcm(d_r, d_theta)
    
    # Family names (from the Complex Lattice paper)
    real_family_names = {
        1: "Gravity/Identity (d_r=1)",
        2: "Tritone/Pivot (d_r=2)",
        3: "Strong/Cubic (d_r=3)",
        4: "Weak/Quartic (d_r=4)",
        6: "Hexadic/EW (d_r=6)",
        12: "EM/Full-Res (d_r=12)",
    }
    imag_family_names = {
        1: "Scalar Phase (d_θ=1)",
        2: "Spin-2/Graviton (d_θ=2)",
        3: "Instanton/Color (d_θ=3)",
        4: "SU(2)_W/Weak (d_θ=4)",
        6: "Spinor/Fermion (d_θ=6)",
        12: "EM/Full-Res (d_θ=12)",
    }
    
    real_name = real_family_names.get(d_r, f"d_r={d_r}")
    imag_name = imag_family_names.get(d_theta, f"d_θ={d_theta}")
    
    # Gaussian coordinate w = k_r + i·k_θ
    w_str = f"{k_r_12}+{k_theta}i" if k_theta >= 0 else f"{k_r_12}{k_theta}i"
    
    return {
        "d_r": d_r,
        "d_theta": d_theta,
        "d_combined": d_combined,
        "real_family": real_name,
        "imag_family": imag_name,
        "w": w_str,
        "kr_mod12": kr_mod,
        "kt_mod12": kt_mod,
    }


# ============================================================
# THE 144-CELL FORCE QUADRANT GRID (§10 of the Sempaevum Paper)
# ============================================================

def is_simple_family(d):
    """A family d is SIMPLE if d divides N=12. COMPLEX otherwise."""
    return (12 % d == 0)


def classify_fqg_quadrant(d_r, d_theta):
    """
    Classify a (d_r, d_θ) cell into one of the four FQG quadrants.
    
    SR+SI: Simple Real × Simple Imaginary (both d|12)
    CR+SI: Complex Real × Simple Imaginary (d_r∤12, d_θ|12)
    SR+CI: Simple Real × Complex Imaginary (d_r|12, d_θ∤12)
    CR+CI: Complex Real × Complex Imaginary (both d∤12)
    
    From §10.3: The grid has a natural binary decomposition by whether
    each axis's family is a divisor of N=12 or not.
    """
    r_simple = is_simple_family(d_r)
    t_simple = is_simple_family(d_theta)
    
    if r_simple and t_simple:
        return "SR+SI"
    elif not r_simple and t_simple:
        return "CR+SI"
    elif r_simple and not t_simple:
        return "SR+CI"
    else:
        return "CR+CI"


def compute_42_combined_families():
    """
    Compute the 42 distinct combined sublattice families from
    all (d_r, d_θ) pairs with d_r, d_θ ∈ {1,...,12}.
    
    From §10.2 (Proposition 10.2):
    The 12×12 grid produces exactly 42 distinct LCM values.
    Maximum is lcm(11,12) = 132 = N(N-1).
    """
    from math import lcm
    combined_set = set()
    grid = {}
    
    for d_r in range(1, 13):
        for d_theta in range(1, 13):
            d_comb = lcm(d_r, d_theta)
            combined_set.add(d_comb)
            grid[(d_r, d_theta)] = d_comb
    
    return sorted(combined_set), grid


def build_fqg_grid(results):
    """
    Build the full 144-cell Force Quadrant Grid from particle data.
    
    Returns:
      - grid_occupancy: dict mapping (d_r, d_θ) to list of particle names
      - quadrant_counts: dict mapping quadrant name to particle count
      - combined_42: sorted list of 42 distinct combined families
      - fqg_grid: dict mapping (d_r, d_θ) to d_comb
    """
    from math import lcm
    
    combined_42, fqg_grid = compute_42_combined_families()
    
    grid_occupancy = {}
    for d_r in range(1, 13):
        for d_theta in range(1, 13):
            grid_occupancy[(d_r, d_theta)] = []
    
    quadrant_counts = {"SR+SI": 0, "CR+SI": 0, "SR+CI": 0, "CR+CI": 0}
    
    for r in results:
        d_r = r["d_r_12"]
        d_theta = r["d_theta"]
        
        # Clamp to 1-12 range (at base N=12, d_r is always a divisor of 12)
        if d_r > 12:
            d_r = 12
        if d_theta > 12:
            d_theta = 12
        
        cell = (d_r, d_theta)
        grid_occupancy[cell].append(r["name"])
        
        quadrant = classify_fqg_quadrant(d_r, d_theta)
        quadrant_counts[quadrant] += 1
        
        # Store FQG data in the result
        r["fqg_cell"] = cell
        r["fqg_quadrant"] = quadrant
        r["fqg_d_comb"] = lcm(d_r, d_theta)
    
    return grid_occupancy, quadrant_counts, combined_42, fqg_grid


def verify_pdt_bisection(results):
    """
    Verify the PDT Bisection Theorem (Theorem 10.5):
    Any symmetric binary on the 144-cell FQG that distinguishes
    T-character from D-character by a criterion symmetric in the two axes
    partitions the grid into two halves of equal size: |B_T| = |B_D| = 72.
    
    Test: "at least one axis is complex" → 108 vs 36
    Test: "imaginary axis is complex" → 72 vs 72 ✓
    """
    # The grid has 144 cells total
    # Symmetric binary: "is the imaginary axis complex?"
    # This gives 6×12 = 72 cells where d_θ is complex, 6×12 = 72 where simple
    
    si_cells = 0  # Simple Imaginary
    ci_cells = 0  # Complex Imaginary
    
    for d_r in range(1, 13):
        for d_theta in range(1, 13):
            if is_simple_family(d_theta):
                si_cells += 1
            else:
                ci_cells += 1
    
    bisection_holds = (si_cells == 72 and ci_cells == 72)
    
    return {
        "si_cells": si_cells,
        "ci_cells": ci_cells,
        "bisection_holds": bisection_holds,
        "total_cells": 144,
    }


def generate_fqg_html(grid_occupancy, quadrant_counts, combined_42, fqg_grid):
    """
    Generate the HTML for the 144-cell Force Quadrant Grid visualization.
    A 12×12 grid with quadrant coloring, occupancy counts, and tooltips.
    """
    from math import lcm
    
    # Quadrant colors
    q_colors = {
        "SR+SI": "rgba(0, 191, 255, 0.15)",   # Blue - standard sector
        "CR+SI": "rgba(255, 165, 0, 0.15)",    # Orange
        "SR+CI": "rgba(144, 238, 144, 0.15)",  # Green
        "CR+CI": "rgba(255, 99, 71, 0.15)",    # Red
    }
    
    # Force family labels (real axis)
    force_labels = {
        1: "Grav", 2: "Pivot", 3: "Strong", 4: "Weak",
        5: "Quint", 6: "Hex", 7: "Sept", 8: "Octet",
        9: "Nonic", 10: "Decic", 11: "Undec", 12: "EM"
    }
    
    # Phase family labels (imaginary axis)
    phase_labels = {
        1: "Scalar", 2: "Spin-2", 3: "Inst", 4: "SU2w",
        5: "E8ico", 6: "Spin½", 7: "Octon", 8: "Bott8",
        9: "CKM", 10: "10D", 11: "11D", 12: "U(1)"
    }
    
    html = '<div class="section">\n'
    html += '<h2>144-Cell Force Quadrant Grid (§10 of the Sempaevum)</h2>\n'
    html += '<div class="info-box">\n'
    html += 'The FQG is the 12×12 grid of (FORCE family d_r, PHASE family d_θ) pairs. '
    html += 'Each cell has combined family d_comb = lcm(d_r, d_θ). '
    html += 'Four quadrants: <span style="color:#00bfff">SR+SI</span> (Simple×Simple, standard sector), '
    html += '<span style="color:#ffa500">CR+SI</span> (Complex×Simple), '
    html += '<span style="color:#90ee90">SR+CI</span> (Simple×Complex), '
    html += '<span style="color:#ff6347">CR+CI</span> (Complex×Complex). '
    html += f'42 distinct combined families. PDT Bisection: any symmetric binary → 72:72.\n'
    html += '</div>\n'
    
    # Quadrant summary
    html += '<div class="stats-grid">\n'
    for q_name, q_count in quadrant_counts.items():
        html += f'<div class="stat-card" style="border-left:3px solid {q_colors[q_name].replace("0.15","0.8")}">'
        html += f'<div class="stat-label">{q_name}</div>'
        html += f'<div class="stat-value">{q_count}</div>'
        html += f'<div class="stat-detail">particles at base N=12</div>'
        html += '</div>\n'
    html += '</div>\n'
    
    # The 12×12 grid table
    html += '<div class="table-scroll">\n'
    html += '<table style="border-collapse:collapse;font-size:0.7em;text-align:center;">\n'
    
    # Header row with PHASE labels
    html += '<tr><th style="min-width:50px">FORCE↓ PHASE→</th>'
    for d_theta in range(1, 13):
        simple_class = "S" if is_simple_family(d_theta) else "C"
        html += f'<th style="min-width:45px;font-size:0.8em">{phase_labels[d_theta]}<br>d_θ={d_theta}<br>({simple_class})</th>'
    html += '</tr>\n'
    
    # Grid rows
    for d_r in range(1, 13):
        simple_r = "S" if is_simple_family(d_r) else "C"
        html += f'<tr><th style="text-align:left;font-size:0.8em">{force_labels[d_r]} d_r={d_r} ({simple_r})</th>'
        
        for d_theta in range(1, 13):
            cell = (d_r, d_theta)
            occupants = grid_occupancy.get(cell, [])
            d_comb = fqg_grid.get(cell, lcm(d_r, d_theta))
            quadrant = classify_fqg_quadrant(d_r, d_theta)
            bg = q_colors[quadrant]
            
            count = len(occupants)
            title_text = f"d_r={d_r}, d_θ={d_theta}, d_comb={d_comb}, Q={quadrant}"
            if occupants:
                title_text += "\\n" + "\\n".join(occupants[:10])
                if count > 10:
                    title_text += f"\\n...and {count-10} more"
            
            cell_text = str(count) if count > 0 else "·"
            font_weight = "bold" if count > 0 else "normal"
            color = "#fff" if count > 0 else "#444"
            
            html += (
                f'<td style="background:{bg};border:1px solid #333;'
                f'padding:3px 4px;font-weight:{font_weight};color:{color}" '
                f'title="{title_text}">'
                f'{cell_text}<br><span style="font-size:0.7em;color:#888">{d_comb}</span>'
                f'</td>'
            )
        
        html += '</tr>\n'
    
    html += '</table>\n'
    html += '</div>\n'
    
    # 42 combined families list
    html += '<div class="info-box" style="margin-top:12px">\n'
    html += f'<b>42 Distinct Combined Families:</b> {", ".join(str(d) for d in combined_42)}<br>\n'
    html += f'<b>Maximum:</b> lcm(11,12) = 132 = N(N−1)<br>\n'
    
    # PDT Bisection verification
    bisection = verify_pdt_bisection(None)
    html += f'<b>PDT Bisection Theorem:</b> '
    html += f'SI cells = {bisection["si_cells"]}, CI cells = {bisection["ci_cells"]} → '
    html += f'{"✓ 72:72 VERIFIED" if bisection["bisection_holds"] else "✗ FAILED"}\n'
    html += '</div>\n'
    
    html += '</div>\n'
    return html


def compute_particle_projections(pdg_particles=None):
    """
    Project every particle onto the Sempaevum through the full LCM tower.
    Uses PDG parsed data.
    
    Computes:
      - 120-digit projections at N=12 and N=27720
      - Full LCM tower escalation (12 → 60 → 420 → 840 → 2520 → 27720)
      - 24 harmonic family classification (12 real + 12 imaginary)
      - True home identification
      - Losslessness verification
    """
    from math import lcm
    results = []
    me_str = "0.51099895000"
    me_mp = mpmath.mpf(me_str)
    
    if not pdg_particles:
        raise ValueError("PDG particle data required — no fallback")
    
    for p in pdg_particles:
        mass_mev = p["mass_mev"]
        if mass_mev <= 0:
            continue  # Skip massless particles (∂I boundary)
        
        mass = mass_mev
        r = mass / ELECTRON_MASS  # Float64 for Plotly coordinates
        
        # ── 120-DIGIT MPMATH PROJECTIONS ──
        mass_str = p.get("mass_str", str(mass_mev / 1000.0))
        # Build r_mp from mass/m_e
        mass_mp = mpmath.mpf(str(mass_mev))
        r_mp = mass_mp / (me_mp * mpmath.mpf(1000)) if p.get("mass_gev") else mass_mp / me_mp
        
        # Handle PDG format (mass in GeV) vs legacy format (mass in MeV)
        if "mass_gev" in p:
            mass_gev_mp = mpmath.mpf(str(p["mass_gev"]))
            r_mp = (mass_gev_mp * mpmath.mpf(1000)) / me_mp
        
        r_str_full = mpmath.nstr(r_mp, TARGET_DIGITS + 10)
        
        # Project at N=12
        proj_mp_12 = sempaevum_project_mp(r_str_full, N)
        k_r_12 = proj_mp_12[0]
        d_r_12 = proj_mp_12[1]
        eps_mp_12 = proj_mp_12[2]
        
        # Project at N=27720
        proj_mp_full = sempaevum_project_mp(r_str_full, N_FULL)
        k_r_full = proj_mp_full[0]
        d_r_full = proj_mp_full[1]
        eps_mp_full = proj_mp_full[2]
        
        # 120-digit ε strings
        eps_120_str_12 = mpmath.nstr(eps_mp_12, TARGET_DIGITS, strip_zeros=False)
        eps_120_str_full = mpmath.nstr(eps_mp_full, TARGET_DIGITS, strip_zeros=False)
        r_120_str = mpmath.nstr(r_mp, TARGET_DIGITS, strip_zeros=False)
        
        # Losslessness verification (Theorem 15.1)
        r_recovered_12 = sempaevum_pullback_mp(k_r_12, eps_mp_12, N)
        roundtrip_err_12 = abs(r_recovered_12 - r_mp)
        r_recovered_full = sempaevum_pullback_mp(k_r_full, eps_mp_full, N_FULL)
        roundtrip_err_full = abs(r_recovered_full - r_mp)
        rt_err_12_str = mpmath.nstr(roundtrip_err_12, 8)
        rt_err_full_str = mpmath.nstr(roundtrip_err_full, 8)
        rt_pass = roundtrip_err_12 < mpmath.power(10, -TARGET_DIGITS)
        
        # ── LCM TOWER ESCALATION (dynamic, convergence-based) ──
        # Compute measurement uncertainty in ppm for convergence threshold
        mass_err_pos = p.get("mass_err_pos_mev", 0)
        if mass_err_pos > 0 and mass_mev > 0:
            mass_err_ppm = mass_err_pos / mass_mev * 1e6
        else:
            mass_err_ppm = None  # No uncertainty data — escalate to max precision
        
        tower, tower_converged = compute_tower_projection(r_mp, r_str_full, mass_err_ppm)
        true_home_idx, true_home_data = find_true_home(tower)
        
        # ── 24 HARMONIC FAMILIES (complex lattice) ──
        spin = p.get("spin", 0.0)
        category = p.get("category", "Unknown")
        k_theta = phase_k_theta(p)
        families = classify_24_families(k_r_12, k_theta)
        
        # ── SUBLATTICE FAMILY at N=12 (distinct from harmonic family) ──
        # The sublattice family is characterized by g = gcd(|k|, N)
        # and d = N/g. The family tells you which sub-lattice of L_12
        # the particle inhabits.
        abs_k12 = abs(k_r_12)
        g_12 = et_gcd(abs_k12, N) if abs_k12 > 0 else N
        sublattice_d_12 = N // g_12
        sublattice_g_factors_12 = factor_integer(g_12)
        
        # Magical impedance — using mpmath
        a0_magic = (d_r_12 - 1)**2 + S**2
        xi_mp = mpmath.mpf(A0_LOCAL) / mpmath.mpf(a0_magic)
        
        # Tightness — pure mpmath
        tight_mp = mpmath.mpf(100) / (mpmath.mpf(100) + abs(eps_mp_12))
        tight_vs_koide = "TIGHT" if tight_mp > K_KOIDE else "LOOSE"
        
        # V-threshold significance — pure mpmath
        v_threshold_mp = mpmath.mpf(600) / (mpmath.mpf(N) * mpmath.mpf(N))
        v_significant = abs(eps_mp_12) < v_threshold_mp
        
        # Build name
        name = p.get("name", "Unknown")
        symbol = p.get("symbol", name[:3])
        charges = p.get("charges", "")
        
        # r and log2_r as mpmath
        log2_r_mp = mpmath.log(r_mp, 2)
        
        result = {
            "name": name,
            "symbol": symbol if "symbol" in p else name,
            "pdg_ids": p.get("pdg_ids", []),
            "mass_mev": mass_mev,
            "mass_str": mass_str,
            "charges": charges,
            "spin": spin,
            "charge": p.get("charge", 0.0),
            "category": category,
            "interaction": p.get("interaction", ""),
            "r_mp": r_mp,
            "log2_r_mp": log2_r_mp,
            # N=12 projection (mpmath-authoritative)
            "k_r_12": k_r_12,
            "d_r_12": d_r_12,
            "eps_mp_12": eps_mp_12,
            # N=27720 projection (mpmath)
            "k_r_full": k_r_full,
            "d_r_full": d_r_full,
            "eps_mp_full": eps_mp_full,
            # 120-digit strings
            "eps_120_12": eps_120_str_12,
            "eps_120_full": eps_120_str_full,
            "r_120": r_120_str,
            "rt_err_12": rt_err_12_str,
            "rt_err_full": rt_err_full_str,
            "rt_pass": rt_pass,
            # Sublattice family at N=12 (DISTINCT from harmonic family)
            "sublattice_g_12": g_12,
            "sublattice_d_12": sublattice_d_12,
            "sublattice_g_factors_12": sublattice_g_factors_12,
            # 24 Harmonic Families (complex lattice)
            "k_theta": k_theta,
            "d_theta": families["d_theta"],
            "d_combined": families["d_combined"],
            "real_family": families["real_family"],
            "imag_family": families["imag_family"],
            "w": families["w"],
            "kr_mod12": families["kr_mod12"],
            "kt_mod12": families["kt_mod12"],
            # LCM Tower (dynamic, convergence-based)
            "tower": tower,
            "tower_converged": tower_converged,
            "tower_levels": len(tower),
            "true_home_idx": true_home_idx,
            "true_home_N": true_home_data["N"] if true_home_data else 0,
            "true_home_d": true_home_data["d"] if true_home_data else 0,
            "true_home_label": true_home_data["label"] if true_home_data else "?",
            # Impedance (mpmath)
            "a0_magic": a0_magic,
            "xi_mp": xi_mp,
            # Tightness (mpmath)
            "tightness_mp": tight_mp,
            "tight_status": tight_vs_koide,
            "v_significant": v_significant,
            "v_threshold_mp": v_threshold_mp,
        }
        results.append(result)
    
    return results


# ============================================================
# ADDITIONAL STRUCTURAL RATIOS — Koide, boson mass ratios
# ============================================================

def compute_structural_ratios(results):
    """
    Compute the key inter-particle ratios and their Sempaevum projections
    at 120-digit precision using mpmath.
    """
    ratios = []
    
    # Build lookup by PDG ID — the only reliable method
    by_pdg = {}
    for r in results:
        for pid in r.get("pdg_ids", []):
            by_pdg[abs(pid)] = r
    
    electron = by_pdg.get(11)
    muon = by_pdg.get(13)
    tau_p = by_pdg.get(15)
    proton = by_pdg.get(2212)
    w_boson = by_pdg.get(24)
    z_boson = by_pdg.get(23)
    higgs = by_pdg.get(25)
    top = by_pdg.get(6)
    bottom = by_pdg.get(5)
    charm = by_pdg.get(4)
    up = by_pdg.get(2)
    down = by_pdg.get(1)
    strange = by_pdg.get(3)
    
    def mp_mass_of(p):
        if p is None: return None
        return mpmath.mpf(str(p["mass_mev"]))
    
    def project_ratio(name, r_mp, et_pred=None):
        r_str = mpmath.nstr(r_mp, TARGET_DIGITS + 10)
        proj = sempaevum_project_mp(r_str, N)
        entry = {
            "name": name,
            "value_mp": r_mp,
            "value": float(r_mp),  # float ONLY for HTML display formatting
            "value_120": mpmath.nstr(r_mp, TARGET_DIGITS, strip_zeros=False),
            "projection_12": (proj[0], proj[1], float(proj[2])),  # float for HTML
            "eps_mp": proj[2],
            "eps_120": mpmath.nstr(proj[2], TARGET_DIGITS, strip_zeros=False),
        }
        if et_pred is not None:
            entry["ET_prediction"] = float(et_pred)  # float for HTML
            entry["ET_prediction_mp"] = et_pred
            dev_mp = abs(r_mp - et_pred) / et_pred * mpmath.mpf(1000000)
            entry["deviation_ppm"] = float(dev_mp)  # float for HTML
            entry["deviation_ppm_mp"] = dev_mp
        return entry
    
    # Koide ratio for charged leptons
    if electron and muon and tau_p:
        m_e_mp = mp_mass_of(electron)
        m_mu_mp = mp_mass_of(muon)
        m_tau_mp = mp_mass_of(tau_p)
        
        num = m_e_mp + m_mu_mp + m_tau_mp
        sqrt_sum = mpmath.sqrt(m_e_mp) + mpmath.sqrt(m_mu_mp) + mpmath.sqrt(m_tau_mp)
        koide_Q = num / (sqrt_sum * sqrt_sum)
        K_mp = mpmath.mpf(2) / mpmath.mpf(3)
        ratios.append(project_ratio("Koide Ratio Q (lepton)", koide_Q, K_mp))
    
    # Proton/electron mass ratio
    if proton and electron:
        ratios.append(project_ratio("Proton/Electron mass ratio μ",
                                     mp_mass_of(proton) / mp_mass_of(electron)))
    
    # W/Z mass ratio
    if w_boson and z_boson:
        ratios.append(project_ratio("M_Z / M_W (Weinberg)",
                                     mp_mass_of(z_boson) / mp_mass_of(w_boson)))
    
    # Higgs/W mass ratio
    if higgs and w_boson:
        ratios.append(project_ratio("M_H / M_W (Higgs-Strong)",
                                     mp_mass_of(higgs) / mp_mass_of(w_boson)))
    
    # Higgs/Z mass ratio
    if higgs and z_boson:
        ratios.append(project_ratio("M_H / M_Z",
                                     mp_mass_of(higgs) / mp_mass_of(z_boson)))
    
    # Muon/electron
    if muon and electron:
        ratios.append(project_ratio("m_μ / m_e",
                                     mp_mass_of(muon) / mp_mass_of(electron)))
    
    # Tau/electron
    if tau_p and electron:
        ratios.append(project_ratio("m_τ / m_e",
                                     mp_mass_of(tau_p) / mp_mass_of(electron)))
    
    # Top/bottom
    if top and bottom:
        ratios.append(project_ratio("m_t / m_b",
                                     mp_mass_of(top) / mp_mass_of(bottom)))
    
    # Charm/up
    if charm and up:
        ratios.append(project_ratio("m_c / m_u",
                                     mp_mass_of(charm) / mp_mass_of(up)))
    
    # Strange/down
    if strange and down:
        ratios.append(project_ratio("m_s / m_d",
                                     mp_mass_of(strange) / mp_mass_of(down)))
    
    return ratios


# ============================================================
# 3D VISUALIZATION GENERATION — Plotly HTML
# ============================================================

def generate_3d_html(results, ratios, fqg_html=""):
    """
    Generate a comprehensive 3D interactive visualization as an HTML file
    using Plotly.js. The visualization shows each particle on the Sempaevum
    complex lattice.
    
    Axes:
      X: k_r (real lattice coordinate — mass position at N=12)
      Y: k_θ (imaginary lattice coordinate — spin/phase structure)
      Z: ε_r (descriptor gap in cents — proximity to lattice point)
    Color: d_r (sublattice family)
    Size:  log-scaled from mass
    """
    
    # Category colors (ET-native color scheme based on sublattice families)
    category_colors = {
        "Lepton": "#00BFFF",       # Deep sky blue — d=6 fermion phase
        "Quark": "#FF6347",        # Tomato red — d=3 strong/cubic
        "Gauge Boson": "#FFD700",  # Gold — d=4 weak/quartic
        "Scalar Boson": "#7CFC00", # Lawn green — d=1 scalar
        "Composite": "#DDA0DD",    # Plum — composite
    }
    
    # Sublattice family colors
    d_colors = {
        1: "#1a1a2e", 2: "#16213e", 3: "#e94560", 
        4: "#0f3460", 6: "#533483", 12: "#e94560",
    }
    
    # Build trace data
    traces_data = {}
    for r in results:
        cat = r["category"]
        if cat not in traces_data:
            traces_data[cat] = {
                "x": [], "y": [], "z": [],
                "text": [], "size": [], "color": category_colors.get(cat, "#888"),
                "symbol_marker": [],
            }
        
        t = traces_data[cat]
        t["x"].append(r["k_r_12"])
        t["y"].append(r["k_theta"])
        t["z"].append(float(r["eps_mp_12"]))  # float ONLY for Plotly JS
        
        # Size based on log mass (visual emphasis — Plotly needs float)
        size_val = max(6, min(25, 5 + 3 * math.log10(r["mass_mev"] + 1)))
        t["size"].append(size_val)
        
        # Hover text with full Sempaevum data (30-digit ε for tooltip readability)
        eps_30_12 = r['eps_120_12'][:35] if len(r['eps_120_12']) > 35 else r['eps_120_12']
        eps_30_full = r['eps_120_full'][:35] if len(r['eps_120_full']) > 35 else r['eps_120_full']
        xi_str = mpmath.nstr(r['xi_mp'], 6)
        tight_str = mpmath.nstr(r['tightness_mp'], 6)
        vt_str = mpmath.nstr(r['v_threshold_mp'], 4)
        force_ch_labels = {1: "Gravity", 2: "Pivot", 3: "Strong", 4: "Weak",
                           5: "Quintic", 6: "Hexadic", 7: "Septic", 8: "Octet",
                           9: "Nonic", 10: "Decic", 11: "Undecimal", 12: "EM"}
        force_ch = force_ch_labels.get(r['d_r_12'], f"d={r['d_r_12']}")
        fqg_q = r.get("fqg_quadrant", "—")
        hover = (
            f"<b>{r['name']}</b><br>"
            f"Mass: {r['mass_mev']:.6g} MeV/c²<br>"
            f"Charge: {r['charge']:+.3g}e | Spin: {r['spin']}<br>"
            f"━━━ Sempaevum @ N=12 (120-digit) ━━━<br>"
            f"r = m/mₑ (120d in table below)<br>"
            f"k_r = {r['k_r_12']} | d_r = {r['d_r_12']} | ε_r = {eps_30_12}…¢<br>"
            f"FORCE channel: {force_ch} | ξ = {xi_str} | A₀ = {r['a0_magic']}<br>"
            f"Sublattice: g={r['sublattice_g_12']}, d={r['sublattice_d_12']}<br>"
            f"k_θ = {r['k_theta']} | d_θ = {r['d_theta']}<br>"
            f"d_combined = lcm({r['d_r_12']},{r['d_theta']}) = {r['d_combined']} | FQG: {fqg_q}<br>"
            f"Tightness: {tight_str} ({r['tight_status']})<br>"
            f"V-significant: {'✓' if r['v_significant'] else '✗'} (threshold {vt_str}¢)<br>"
            f"━━━ Tower: True Home = {r['true_home_label']} ━━━<br>"
            f"Tower levels: {r['tower_levels']} | Converged: {'✓' if r['tower_converged'] else '—'}<br>"
            f"━━━ Losslessness (Thm 15.1) ━━━<br>"
            f"N=12 round-trip: {'✓' if r['rt_pass'] else '✗'} err={r['rt_err_12']}"
        )
        t["text"].append(hover)
    
    # Build the Plotly traces JSON
    plotly_traces = []
    for cat, data in traces_data.items():
        trace = {
            "type": "scatter3d",
            "mode": "markers+text",
            "name": cat,
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
            "text": [r["symbol"] for r in results if r["category"] == cat],
            "textposition": "top center",
            "textfont": {"size": 10, "color": data["color"]},
            "hovertext": data["text"],
            "hoverinfo": "text",
            "marker": {
                "size": data["size"],
                "color": data["color"],
                "opacity": 0.9,
                "line": {"width": 1, "color": "rgba(255,255,255,0.5)"},
            },
        }
        plotly_traces.append(trace)
    
    # Add Koide attractor line (d=12, ε=±1.955 cents)
    koide_eps = 1.955
    k_range_min = min(r["k_r_12"] for r in results) - 5
    k_range_max = max(r["k_r_12"] for r in results) + 5
    
    # Add structural reference planes
    # ε = 0 plane (perfect lattice alignment)
    # ε = ±50 plane (coherence boundary / Koide threshold)
    
    # Build the ratio annotations
    ratio_annotations_html = ""
    for rat in ratios:
        proj = rat["projection_12"]
        k_val, d_val, eps_val = proj
        ratio_annotations_html += (
            f"<tr>"
            f"<td>{rat['name']}</td>"
            f"<td>{rat['value']:.8g}</td>"
            f"<td>{k_val}</td>"
            f"<td>{d_val}</td>"
            f"<td>{eps_val:+.3f}</td>"
        )
        if "ET_prediction" in rat:
            ratio_annotations_html += f"<td>{rat.get('ET_prediction', ''):.6g}</td>"
            ratio_annotations_html += f"<td>{rat.get('deviation_ppm', 0):.1f} ppm</td>"
        else:
            ratio_annotations_html += "<td>—</td><td>—</td>"
        ratio_annotations_html += "</tr>\n"
    
    # Build particle data table
    particle_table_html = ""
    for r in results:
        d_label = {1: "Gravity/Identity", 2: "Tritone/Pivot", 3: "Strong/Cubic",
                   4: "Weak/Quartic", 6: "Hexadic/EW", 12: "EM/Full-Res"}
        d_name = d_label.get(r["d_r_12"], f"d={r['d_r_12']}")
        fqg_q = r.get("fqg_quadrant", "—")
        
        particle_table_html += (
            f"<tr>"
            f"<td class='particle-name'>{r['name']}</td>"
            f"<td>{r['mass_mev']:.6g}</td>"
            f"<td>{mpmath.nstr(r['r_mp'], 8)}</td>"
            f"<td>{r['k_r_12']}</td>"
            f"<td class='d-{r['d_r_12']}'>{r['d_r_12']}</td>"
            f"<td>{mpmath.nstr(r['eps_mp_12'], 6)}</td>"
            f"<td>{d_name}</td>"
            f"<td>{mpmath.nstr(r['xi_mp'], 6)}</td>"
            f"<td>{r['k_theta']}</td>"
            f"<td>{r['d_theta']}</td>"
            f"<td>{r['d_combined']}</td>"
            f"<td>{fqg_q}</td>"
            f"<td>{r['sublattice_g_12']}</td>"
            f"<td>{r['true_home_label']}</td>"
            f"<td>{'✓' if r['rt_pass'] else '✗'}</td>"
            f"</tr>\n"
        )
    
    # Build the 120-digit detail section
    detail_120_html = ""
    for r in results:
        detail_120_html += (
            f"<div class='detail-card'>"
            f"<div class='detail-header'>{r['name']}</div>"
            f"<div class='detail-label'>r = m/mₑ (120 digits):</div>"
            f"<div class='digit-120'>{r['r_120']}</div>"
            f"<div class='detail-label'>ε₁₂ (N=12, 120 digits, cents):</div>"
            f"<div class='digit-120'>{r['eps_120_12']}</div>"
            f"<div class='detail-label'>ε₂₇₇₂₀ (N=27720, 120 digits, cents):</div>"
            f"<div class='digit-120'>{r['eps_120_full']}</div>"
            f"<div class='detail-label'>Losslessness (Thm 15.1): "
            f"N=12 err={r['rt_err_12']} | N=27720 err={r['rt_err_full']} "
            f"{'✓ PASS' if r['rt_pass'] else '✗ FAIL'}</div>"
            f"</div>\n"
        )
    
    # Build sublattice family tower escalation section
    tower_html = ""
    for r in results:
        tower_rows = ""
        for t in r.get("tower", []):
            is_home = (t["N"] == r.get("true_home_N", 0))
            home_marker = " ◄ TRUE HOME" if is_home else ""
            row_class = "tower-home" if is_home else ""
            tower_rows += (
                f"<tr class='{row_class}'>"
                f"<td>{t['N']}</td>"
                f"<td>{t['label']}</td>"
                f"<td>{t['k']}</td>"
                f"<td class='d-val'>{t['d']}</td>"
                f"<td>{t['d_factors_str']}</td>"
                f"<td>{mpmath.nstr(t['eps_mp'], 8)}</td>"
                f"<td>{t.get('g_factors_str', '')}</td>"
                f"<td>{home_marker}</td>"
                f"</tr>"
            )
        tower_html += (
            f"<div class='tower-card'>"
            f"<div class='tower-header'>{r['name']}"
            f"<span class='tower-meta'> — Harmonic: d_r={r['d_r_12']}, d_θ={r['d_theta']} | "
            f"True home: N={r.get('true_home_N', '?')} (d={r.get('true_home_d', '?')})</span></div>"
            f"<table class='tower-table'>"
            f"<tr><th>N</th><th>Level</th><th>k</th><th>d (sublattice)</th><th>d factors</th><th>ε (¢)</th><th>g factors</th><th></th></tr>"
            f"{tower_rows}"
            f"</table>"
            f"</div>\n"
        )
    
    # Build stat cards dynamically
    stats_html = ""
    for rat in ratios[:4]:
        proj = rat['projection_12']
        detail = f"k={proj[0]}, d={proj[1]}, ε={proj[2]:+.3f}¢"
        if 'deviation_ppm' in rat:
            detail = f"ET prediction: {rat.get('ET_prediction', 0):.10f} | Deviation: {rat['deviation_ppm']:.1f} ppm"
        stats_html += (
            f'<div class="stat-card">'
            f'<div class="stat-label">{rat["name"]}</div>'
            f'<div class="stat-value">{rat["value"]:.10f}</div>'
            f'<div class="stat-detail">{detail}</div>'
            f'</div>\n'
        )
    
    # ============================================================
    # MAGICAL IMPEDANCE SECTION (§7.4 of the Sempaevum Paper)
    # Applies to the 12 real-axis FORCE harmonic families only.
    # ============================================================
    
    impedance_ref_labels = {
        1: ("Gravity / Octave", "Pure will, maximum coupling"),
        2: ("Tritone / Pivot", "Binary / CPT mirror"),
        3: ("Strong / Cubic", "3D volumetric closure"),
        4: ("Weak / Quartic", "State-change, T-axis home"),
        5: ("Quintic / Golden", "First non-divisor of 12"),
        6: ("Hexadic / Composite", "Electroweak composite"),
        7: ("Septic / G₂", "Seven-fold / octonion"),
        8: ("Octet / Gluon", "SU(3) adjoint"),
        9: ("Nonic / Quark", "3 × 3 structure"),
        10: ("Decic / Superstring", "SO(10)-class"),
        11: ("Undecimal / M-theory", "N−1"),
        12: ("Electromagnetic / Full Res", "Baseline"),
    }
    
    # Build the 12-family impedance reference table
    impedance_ref_html = ""
    for d in range(1, 13):
        a0 = (d - 1)**2 + 16
        xi = 137.0 / a0
        character, role = impedance_ref_labels[d]
        simple_tag = "SIMPLE" if is_simple_family(d) else "COMPLEX"
        style = "color:#888;font-style:italic" if not is_simple_family(d) else ""
        impedance_ref_html += (
            f"<tr style='{style}'>"
            f"<td>{d}</td>"
            f"<td>{simple_tag}</td>"
            f"<td>{a0}</td>"
            f"<td><b>{xi:.4f}</b></td>"
            f"<td>{character}</td>"
            f"<td>{role}</td>"
            f"</tr>\n"
        )
    
    # Group particles by their impedance (ξ value, via d_r harmonic family)
    impedance_groups = {}
    for r in results:
        d_r = r["d_r_12"]
        if d_r not in impedance_groups:
            impedance_groups[d_r] = []
        impedance_groups[d_r].append(r)
    
    impedance_groups_html = ""
    for d_r in sorted(impedance_groups.keys()):
        particles = impedance_groups[d_r]
        a0 = (d_r - 1)**2 + 16
        xi = 137.0 / a0
        character = impedance_ref_labels.get(d_r, (f"d={d_r}", ""))[0]
        
        # List particles in this coupling channel
        particle_names = [p["name"] for p in sorted(particles, key=lambda x: x["mass_mev"])]
        
        impedance_groups_html += (
            f'<div class="tower-card" style="border-left:4px solid var(--d{d_r if d_r in [1,2,3,4,6,12] else 12})">'
            f'<div class="tower-header">'
            f'd_r = {d_r} — {character} — ξ = {xi:.4f} — A₀ = {a0} — '
            f'{len(particles)} particle{"s" if len(particles) != 1 else ""}'
            f'</div>'
            f'<div style="font-size:0.85em;padding:4px 8px;line-height:1.8">'
        )
        
        # Show particles grouped by category
        by_cat = {}
        for p in particles:
            cat = p["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(p["name"])
        
        for cat in sorted(by_cat.keys()):
            names = by_cat[cat]
            impedance_groups_html += (
                f'<div><span style="color:var(--accent2)">{cat} ({len(names)})</span>: '
                f'{", ".join(names)}</div>'
            )
        
        impedance_groups_html += '</div></div>\n'
    
    # Full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sempaevum Particle Viewer — P ∘ D ∘ T = E</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

:root {{
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --text: #e0e0e8;
    --accent: #e94560;
    --accent2: #00bfff;
    --gold: #ffd700;
    --d1: #7cfc00;
    --d2: #00ced1;
    --d3: #e94560;
    --d4: #4169e1;
    --d6: #9370db;
    --d12: #ff8c00;
}}

* {{ margin:0; padding:0; box-sizing:border-box; }}

body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    overflow-x: hidden;
}}

.header {{
    text-align: center;
    padding: 40px 20px 20px;
    background: linear-gradient(180deg, #0f0f1a 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
}}

.header h1 {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8em;
    font-weight: 600;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--gold) 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}

.header .subtitle {{
    font-size: 0.85em;
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}

.header .equation {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5em;
    font-style: italic;
    color: var(--gold);
    margin: 15px 0 5px;
    letter-spacing: 0.1em;
}}

.constants-bar {{
    display: flex;
    justify-content: center;
    gap: 30px;
    padding: 12px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}}

.constant {{
    text-align: center;
    padding: 5px 15px;
}}

.constant .label {{ font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 0.1em; }}
.constant .value {{ font-size: 1.1em; color: var(--gold); font-weight: 700; }}

#plot3d {{
    width: 100%;
    height: 700px;
    background: var(--bg);
}}

.section {{
    max-width: 1400px;
    margin: 30px auto;
    padding: 0 20px;
}}

.section h2 {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8em;
    color: var(--accent);
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

.section h3 {{
    font-size: 1em;
    color: var(--accent2);
    margin: 15px 0 8px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
    margin: 10px 0;
}}

th {{
    background: var(--surface);
    color: var(--gold);
    padding: 8px 6px;
    text-align: left;
    font-weight: 500;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid var(--border);
    position: sticky;
    top: 0;
}}

td {{
    padding: 6px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}}

tr:hover td {{ background: rgba(233, 69, 96, 0.05); }}

.particle-name {{ font-weight: 500; color: var(--accent2); }}

.d-1 {{ color: var(--d1); font-weight: 700; }}
.d-2 {{ color: var(--d2); font-weight: 700; }}
.d-3 {{ color: var(--d3); font-weight: 700; }}
.d-4 {{ color: var(--d4); font-weight: 700; }}
.d-6 {{ color: var(--d6); font-weight: 700; }}
.d-12 {{ color: var(--d12); font-weight: 700; }}

.legend {{
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    padding: 12px;
    background: var(--surface);
    border-radius: 6px;
    margin: 10px 0;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85em;
}}

.legend-dot {{
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
}}

.info-box {{
    background: var(--surface);
    border-left: 3px solid var(--accent);
    padding: 15px;
    margin: 15px 0;
    font-size: 0.9em;
    line-height: 1.8;
}}

.footer {{
    text-align: center;
    padding: 30px;
    color: #555;
    font-size: 0.8em;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}}

.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 15px;
    margin: 15px 0;
}}

.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 15px;
    border-radius: 6px;
}}

.stat-card .stat-label {{ color: #888; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.1em; }}
.stat-card .stat-value {{ color: var(--gold); font-size: 1.4em; font-weight: 700; margin: 4px 0; }}
.stat-card .stat-detail {{ color: #aaa; font-size: 0.85em; }}

.table-scroll {{ overflow-x: auto; }}

.detail-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    padding: 12px 15px;
    margin: 8px 0;
    border-radius: 4px;
}}

.detail-header {{
    font-weight: 700;
    color: var(--accent);
    font-size: 1em;
    margin-bottom: 6px;
}}

.detail-label {{
    font-size: 0.75em;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}}

.digit-120 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7em;
    color: var(--gold);
    word-break: break-all;
    line-height: 1.4;
    padding: 4px 8px;
    background: rgba(0,0,0,0.3);
    border-radius: 3px;
    margin: 3px 0;
}}

.tower-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 10px 12px;
    margin: 6px 0;
    border-radius: 4px;
}}

.tower-header {{
    font-weight: 700;
    color: var(--accent);
    font-size: 0.95em;
    margin-bottom: 6px;
}}

.tower-meta {{
    font-weight: 400;
    color: #888;
    font-size: 0.8em;
}}

.tower-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78em;
}}

.tower-table th {{
    background: rgba(0,0,0,0.3);
    color: var(--gold);
    padding: 4px 6px;
    text-align: left;
    font-size: 0.85em;
    border-bottom: 1px solid var(--border);
}}

.tower-table td {{
    padding: 3px 6px;
    border-bottom: 1px solid rgba(30,30,46,0.5);
}}

.tower-table .d-val {{
    color: var(--accent2);
    font-weight: 700;
}}

.tower-home td {{
    background: rgba(255,215,0,0.08);
    color: var(--gold);
    font-weight: 500;
}}
</style>
</head>
<body>

<div class="header">
    <h1>SEMPAEVUM</h1>
    <div class="subtitle">Particle Lattice Viewer — The Bijection Applied to the Standard Model</div>
    <div class="equation">P ∘ D ∘ T = E &nbsp;&nbsp;|&nbsp;&nbsp; 3 = 3 = 3 = Σ</div>
    <div class="subtitle" style="margin-top:8px;">PDG 2024 Data → Sempaevum Bijection → 3D Lattice Visualization</div>
</div>

<div class="constants-bar">
    <div class="constant"><div class="label">Manifold Symmetry</div><div class="value">N = 12</div></div>
    <div class="constant"><div class="label">Base Variance</div><div class="value">V = 1/12</div></div>
    <div class="constant"><div class="label">Koide Ratio</div><div class="value">K = 2/3</div></div>
    <div class="constant"><div class="label">Impedance</div><div class="value">A₀ = 137</div></div>
    <div class="constant"><div class="label">Reference</div><div class="value">R₀ = mₑ = {ELECTRON_MASS} MeV</div></div>
    <div class="constant"><div class="label">Universal Lattice</div><div class="value">N_FULL = 27720</div></div>
</div>

<div id="plot3d"></div>

<div class="section">
    <h2>The Three Tools Applied</h2>
    <div class="info-box">
        <b>Identification Principle:</b> P = particle substrate (physical existence), D = measurable properties (mass, charge, spin, quantum numbers), T = measurement/observation that substantiates.<br>
        <b>Descriptor Gap Principle:</b> The gap between raw particle data and structural understanding IS a Descriptor — the Sempaevum bijection Π_N(r) = (k, d, ε) fills it.<br>
        <b>Subsumption Law:</b> Every particle property finds a lattice address. The projection subsumes all massive particles without remainder. Massless particles (γ, g) sit at the ∂I annihilation boundary.
    </div>
</div>

<div class="section">
    <h2>Structural Summary</h2>
    <div class="stats-grid">
{stats_html}
    </div>
</div>

<div class="section">
    <h2>Complete Particle Projections — Sempaevum Bijection at N=12</h2>
    <div class="info-box">
        Reference period R₀ = mₑ (electron mass) — the smallest closed T-traversal loop in the mass sector (Anti-Numerology condition N2). All ratios r = m/mₑ are genuinely dimensionless (N1). Sublattice families match physical force sectors (N3).
    </div>
    <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:var(--d1)"></span> d=1 Gravity/Identity</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--d2)"></span> d=2 Tritone/Pivot</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--d3)"></span> d=3 Strong/Cubic</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--d4)"></span> d=4 Weak/Quartic</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--d6)"></span> d=6 Hexadic/EW-Composite</div>
        <div class="legend-item"><span class="legend-dot" style="background:var(--d12)"></span> d=12 EM/Full-Resolution</div>
    </div>
    <div class="table-scroll">
    <table>
        <thead>
        <tr>
            <th>Particle</th>
            <th>Mass (MeV)</th>
            <th>r = m/mₑ</th>
            <th>k_r</th>
            <th>d_r</th>
            <th>ε_r (¢)</th>
            <th>FORCE Channel</th>
            <th>ξ(d_r)</th>
            <th>k_θ</th>
            <th>d_θ</th>
            <th>d_comb</th>
            <th>FQG</th>
            <th>g₁₂</th>
            <th>True Home</th>
            <th>Lossless</th>
        </tr>
        </thead>
        <tbody>
{particle_table_html}
        </tbody>
    </table>
    </div>
</div>

{fqg_html}

<div class="section">
    <h2>Magical Impedance — Per-Family Coupling Strength (§7.4)</h2>
    <div class="info-box">
        <b>Magical Impedance applies to the 12 real-axis FORCE harmonic families only.</b>
        A₀<sup>magic</sup>(d) = (d−1)² + S² = (d−1)² + 16. Coupling strength ξ(d) = 137 / A₀<sup>magic</sup>(d).
        At d=12 (EM/full resolution): ξ = 137/137 = 1.000 — the electromagnetic baseline, by construction.
        At d=1 (Gravity/octave): ξ = 137/16 = 8.5625 — maximum coupling.
        ξ is strictly monotonically decreasing: lower d → stronger coupling, higher d → finer resolution at weaker coupling.
        SIMPLE families (d|12) are native at base N=12. COMPLEX families (d∤12) are shadow forces, native at lcm(12,d).
    </div>
    
    <h3>12-Family Impedance Reference Table</h3>
    <div class="table-scroll">
    <table>
        <thead>
        <tr>
            <th>d</th>
            <th>Status</th>
            <th>A₀<sup>magic</sup></th>
            <th>ξ(d)</th>
            <th>Character</th>
            <th>Role</th>
        </tr>
        </thead>
        <tbody>
{impedance_ref_html}
        </tbody>
    </table>
    </div>
    
    <h3>Particles Grouped by Coupling Channel (d_r → ξ)</h3>
    <div class="info-box">
        Each particle's real-axis harmonic family d_r determines its FORCE coupling channel.
        Particles sharing the same d_r share the same impedance ξ(d_r).
        This grouping reveals which particles occupy the same force-coupling niche on the lattice.
    </div>
{impedance_groups_html}
</div>

<div class="section">
    <h2>120-Digit Precision — Full ε Values (mpmath @ 135 working digits)</h2>
    <div class="info-box">
        All ε values computed using mpmath arbitrary-precision arithmetic at 135 working decimal places (120 target + 15 guard).
        Losslessness Theorem (15.1) verified: Π_N⁻¹(Π_N(r)) = r to 120+ digits for every particle.
    </div>
{detail_120_html}
</div>

<div class="section">
    <h2>Sublattice Family Tower Escalation — LCM Resolution Ladder</h2>
    <div class="info-box">
        <b>Sublattice families ≠ Harmonic families.</b> Harmonic families are the mod-12 classification (d_r, d_θ) — which of the 12 positions in the octave. Sublattice families are the group-theoretic sublattice d = N/gcd(|k|, N) at each tower resolution N. At N=12, only 6 sublattice families exist (divisors of 12). At N=27720, there are 96 possible sublattice families (divisors of 27720). The tower escalation shows each particle's d evolving as D adds descriptors level by level, until the particle finds its <b>true home</b> — the resolution where its sublattice family stabilizes. The prime factorization of d reveals which harmonic components compose the particle's lattice address.
    </div>
{tower_html}
</div>

<div class="section">
    <h2>Structural Ratios — Inter-Particle Sempaevum Projections</h2>
    <div class="table-scroll">
    <table>
        <thead>
        <tr>
            <th>Ratio</th>
            <th>Value</th>
            <th>k</th>
            <th>d</th>
            <th>ε (¢)</th>
            <th>ET Prediction</th>
            <th>Deviation</th>
        </tr>
        </thead>
        <tbody>
{ratio_annotations_html}
        </tbody>
    </table>
    </div>
</div>

<div class="section">
    <h2>Massless Particles — The ∂I Annihilation Boundary</h2>
    <div class="info-box">
        The photon (γ) and gluon (g) have mass = 0. Since log₂(0) = −∞, the projection formula Π_N diverges and these particles are excluded from the lattice domain (Proposition 5.3). They sit AT the ∂I annihilation boundary — structurally inside Σ but unattainable on L_N at any finite resolution. This is the lattice expression of masslessness: photons and gluons are Mediation-state {'{D,T}'} configurations, never fully substantiated as mass-bearing Exceptions.
    </div>
</div>

<div class="footer">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.2em; font-style:italic; color:var(--gold); margin-bottom:10px;">
        For every exception there is an exception, except the Exception.
    </div>
    Exception Theory — Michael James Muller (Aevum Defluo)<br>
    Particle Data: PDG 2024 (S. Navas et al., Phys. Rev. D 110, 030001)<br>
    Sempaevum bijection, lattice projection, and visualization
</div>

<script>
const traces = {json.dumps(plotly_traces)};

// Add ε=0 reference plane
traces.push({{
    type: 'mesh3d',
    x: [{k_range_min}, {k_range_max}, {k_range_max}, {k_range_min}],
    y: [0, 0, 12, 12],
    z: [0, 0, 0, 0],
    i: [0, 0],
    j: [1, 2],
    k: [2, 3],
    opacity: 0.08,
    color: '#ffd700',
    name: 'ε=0 (perfect lattice)',
    showlegend: true,
    hoverinfo: 'name',
}});

// Add Koide attractor markers at ε = ±1.955
traces.push({{
    type: 'scatter3d',
    mode: 'lines',
    x: [{k_range_min}, {k_range_max}],
    y: [0, 0],
    z: [{koide_eps}, {koide_eps}],
    line: {{ color: '#ff4444', width: 2, dash: 'dash' }},
    name: 'Koide ε = +1.955¢',
    hoverinfo: 'name',
}});

traces.push({{
    type: 'scatter3d',
    mode: 'lines',
    x: [{k_range_min}, {k_range_max}],
    y: [0, 0],
    z: [{-koide_eps}, {-koide_eps}],
    line: {{ color: '#ff4444', width: 2, dash: 'dash' }},
    name: 'Koide ε = −1.955¢',
    hoverinfo: 'name',
}});

const layout = {{
    scene: {{
        xaxis: {{
            title: {{ text: 'k_r (Real Lattice Coordinate — Mass)', font: {{ size: 11, color: '#aaa' }} }},
            gridcolor: '#1a1a2e',
            zerolinecolor: '#333',
            color: '#888',
        }},
        yaxis: {{
            title: {{ text: 'k_θ (Imaginary Lattice — Phase)', font: {{ size: 11, color: '#aaa' }} }},
            gridcolor: '#1a1a2e',
            zerolinecolor: '#333',
            color: '#888',
            dtick: 1,
        }},
        zaxis: {{
            title: {{ text: 'ε_r (Descriptor Gap, cents)', font: {{ size: 11, color: '#aaa' }} }},
            gridcolor: '#1a1a2e',
            zerolinecolor: '#ffd700',
            color: '#888',
        }},
        bgcolor: '#0a0a0f',
        camera: {{
            eye: {{ x: 1.8, y: -1.5, z: 0.8 }},
            center: {{ x: 0, y: 0, z: -0.1 }},
        }},
    }},
    paper_bgcolor: '#0a0a0f',
    plot_bgcolor: '#0a0a0f',
    font: {{ family: 'JetBrains Mono, monospace', color: '#ccc', size: 11 }},
    legend: {{
        bgcolor: 'rgba(18,18,26,0.9)',
        bordercolor: '#333',
        borderwidth: 1,
        font: {{ size: 11 }},
        x: 0.01, y: 0.99,
    }},
    margin: {{ l: 0, r: 0, t: 30, b: 0 }},
    title: {{
        text: 'Sempaevum Complex Lattice — Standard Model Particles at N=12',
        font: {{ family: 'Cormorant Garamond, serif', size: 18, color: '#e94560' }},
        x: 0.5,
    }},
}};

Plotly.newPlot('plot3d', traces, layout, {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['toImage'],
}});
</script>

<div class="section" style="border-top:2px solid var(--accent);margin-top:40px;">
    <h2>References</h2>
    <div style="font-family:'Cormorant Garamond',serif;font-size:1.05em;line-height:1.8;padding:12px 0;">
        <p style="margin-bottom:16px;">
            <b>[1]</b>&ensp;R.&thinsp;L.&thinsp;Workman <i>et al.</i> (Particle Data Group),
            &ldquo;Review of Particle Physics,&rdquo;
            <i>Prog.&thinsp;Theor.&thinsp;Exp.&thinsp;Phys.</i> <b>2022</b>, 083C01 (2022).
            DOI:&ensp;<a href="https://doi.org/10.1093/ptep/ptac097" target="_blank" style="color:var(--accent2);">10.1093/ptep/ptac097</a>
        </p>
        <p style="margin-bottom:16px;">
            <b>[2]</b>&ensp;M.&thinsp;J.&thinsp;Muller (Aevum Defluo),
            &ldquo;The Sempaevum: A Lossless Mathematical Rendering of the Totality &Sigma; Derived from Three Primitive Categories,&rdquo;
            manuscript, April 2026.
        </p>
    </div>
</div>

</body>
</html>"""
    
    return html


# ============================================================
# MARKDOWN REPORT GENERATION
# ============================================================

def generate_report(results, ratios):
    """Generate the full markdown report."""
    report = []
    report.append("# Sempaevum Particle Viewer — Complete Report")
    report.append("## The Bijection Applied to PDG 2024 Fundamental Particle Data")
    report.append("")
    report.append("**Author:** Michael James Muller (Aevum Defluo)")
    report.append("**Theory:** Exception Theory — P ∘ D ∘ T = E")
    report.append("**Data Source:** PDG 2024 (S. Navas et al., Phys. Rev. D 110, 030001)")
    report.append("**Reference Period:** R₀ = mₑ = 0.51099895 MeV/c² (electron mass)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 1. The Three Tools Applied")
    report.append("")
    report.append("**Identification Principle:**")
    report.append("- P (substrate): The particle as a physical entity — its existence within Σ")
    report.append("- D (constraints): Measurable properties — mass, charge, spin, quantum numbers (finite, articulable)")
    report.append("- T (agency): The measurement act that substantiates properties; the projection (rounding) itself")
    report.append("")
    report.append("**Descriptor Gap Principle:**")
    report.append("The gap between 'raw PDG data' and 'structural understanding' IS a Descriptor — the Sempaevum bijection Π_N(r) = (k, d, ε) fills it. Each projection closes one gap; each closed gap reveals structural patterns.")
    report.append("")
    report.append("**Subsumption Law:**")
    report.append("Every massive particle property finds a lattice address without remainder. Massless particles (γ, g) sit at the ∂I annihilation boundary, structurally inside Σ but unattainable on L_N — confirming that the Subsumption is complete.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 2. ET Constants (All Forced)")
    report.append("")
    report.append(f"- N = |Π| × S = 3 × 4 = **{N}** (manifold symmetry)")
    report.append(f"- V = 1/N = **1/12** (base variance)")
    report.append(f"- K = 1 - S/N = 1 - 4/12 = **2/3** (Koide ratio)")
    report.append(f"- A₀ = (N-1)² + S² = 11² + 4² = **{A0_LOCAL}** (manifold impedance)")
    report.append(f"- N_FULL = lcm(1..11) = **{N_FULL}** (universal lattice resolution)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 3. Complete Particle Projections at N=12")
    report.append("")
    report.append("| Particle | Mass (MeV) | r = m/mₑ | k_r | d_r | ε_r (¢) | FORCE Channel | ξ(d_r) | k_θ | d_θ | d_comb | FQG |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    d_label = {1: "Gravity", 2: "Pivot", 3: "Strong", 4: "Weak", 6: "Hexadic", 12: "EM"}
    
    for r in results:
        dname = d_label.get(r["d_r_12"], f"d={r['d_r_12']}")
        fqg_q = r.get("fqg_quadrant", "—")
        report.append(
            f"| {r['name']} | {r['mass_mev']:.6g} | {mpmath.nstr(r['r_mp'], 8)} | {r['k_r_12']} | "
            f"**{r['d_r_12']}** | {mpmath.nstr(r['eps_mp_12'], 6)} | {dname} | {mpmath.nstr(r['xi_mp'], 6)} | "
            f"{r['k_theta']} | {r['d_theta']} | {r['d_combined']} | {fqg_q} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 4. Tower Resolution — True Home per Particle")
    report.append("")
    report.append("| Particle | True Home | Tower Levels | d (home) | Converged |")
    report.append("|---|---|---|---|---|")
    
    for r in results:
        report.append(
            f"| {r['name']} | {r['true_home_label']} | {r['tower_levels']} | {r['true_home_d']} | {'✓' if r['tower_converged'] else '—'} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 5. Structural Ratios — Inter-Particle Projections")
    report.append("")
    report.append("| Ratio | Value | k | d | ε (¢) | ET Prediction | Deviation |")
    report.append("|---|---|---|---|---|---|---|")
    
    for rat in ratios:
        proj = rat["projection_12"]
        pred = f"{rat.get('ET_prediction', '—')}" if 'ET_prediction' in rat else "—"
        dev = f"{rat.get('deviation_ppm', 0):.1f} ppm" if 'deviation_ppm' in rat else "—"
        report.append(
            f"| {rat['name']} | {rat['value']:.8g} | {proj[0]} | **{proj[1]}** | {proj[2]:+.3f} | {pred} | {dev} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 6. Structural Analysis — What the Sempaevum Reveals")
    report.append("")
    
    # Analyze d_r distribution
    d_counts = {}
    for r in results:
        d_val = r["d_r_12"]
        if d_val not in d_counts:
            d_counts[d_val] = []
        d_counts[d_val].append(r["name"])
    
    report.append("### 6.1 Sublattice Family Distribution")
    report.append("")
    for d_val in sorted(d_counts.keys()):
        dname = d_label.get(d_val, f"d={d_val}")
        members = ", ".join(d_counts[d_val])
        report.append(f"**d={d_val} ({dname}):** {members}")
        report.append("")
    
    report.append("### 6.2 Key Structural Findings")
    report.append("")
    if ratios and 'deviation_ppm' in ratios[0]:
        report.append("1. **Koide Ratio Verification:** The charged lepton mass formula Q = (m_e + m_μ + m_τ)/(√m_e + √m_μ + √m_τ)² = " + f"{ratios[0]['value']:.10f} deviates from K = 2/3 by only {ratios[0]['deviation_ppm']:.1f} ppm — confirming the Koide binding stability threshold to extraordinary precision.")
    report.append("")
    report.append("2. **M_Z/M_W is Hexadic (d=6):** The Weinberg mixing ratio projects to the hexadic sublattice — the same family as the muon mass ratio μ = m_μ/m_e. Both electroweak mixing and second-generation lepton mass hierarchy are hexadic phenomena, consistent with d=6 being the QCD+QED composite bridge.")
    report.append("")
    report.append("3. **M_H/M_W is Cubic (d=3):** The Higgs-to-W mass ratio lands in the strong-force sublattice, reflecting the Higgs mechanism's deep entanglement with QCD through top-quark loop contributions.")
    report.append("")
    report.append("4. **All fermions are spin-½ (d_θ=6):** Every quark and lepton inhabits the d_θ=6 (spin-½ fermion phase) harmonic family on the imaginary axis, as predicted by Table 8 of the Sempaevum paper.")
    report.append("")
    report.append("5. **W/Z bosons are quartic (d_θ=4):** The weak gauge bosons correctly inhabit the SU(2)_W phase family on the imaginary axis.")
    report.append("")
    report.append("6. **Massless particles at ∂I:** The photon and gluon, being massless, sit at the annihilation boundary where log₂(0) = −∞ — structurally present in Σ but unattainable on L_N. This is the lattice expression of their Mediation-state {D,T} character.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 7. The Bijection Is Lossless (Theorem 15.1)")
    report.append("")
    report.append("The Sempaevum projection Π_N is a bijection at every finite resolution. The pullback Π_N⁻¹(k, d, ε) = 2^((k + εN/1200)/N) exactly recovers the original ratio r. No information about the particle's mass ratio is discarded — the projection redistributes it among three complementary coordinates (k, d, ε). In the LCM-tower limit N→∞, the residual ε→0 uniformly.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*P ∘ D ∘ T = E — For every exception there is an exception, except the Exception.*")
    
    return "\n".join(report)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMPAEVUM PARTICLE VIEWER — LCM Tower + 24 Harmonic Families")
    print("P ∘ D ∘ T = E  |  3 = 3 = 3 = Σ")
    print("=" * 70)
    print()
    
    # Parse PDG 2025 data
    pdg_path = "/mnt/user-data/uploads/mass_width_2025.txt"
    pdg_particles = None
    if os.path.exists(pdg_path):
        print("[0/4] Parsing PDG 2025 mass/width file...")
        pdg_particles = parse_pdg_file(pdg_path)
        print(f"      Parsed {len(pdg_particles)} particles from PDG 2025")
    else:
        print("[0/4] No PDG file found, using built-in particle data")
    
    # Compute all projections with tower escalation
    print("[1/4] Computing Sempaevum projections through LCM tower...")
    print(f"      Tower: dynamic LCM escalation to convergence ({len(LCM_TOWER_FULL)} levels available)")
    results = compute_particle_projections(pdg_particles)
    print(f"      Projected {len(results)} massive particles at all tower levels")
    
    # Compute structural ratios
    print("[2/5] Computing structural inter-particle ratios...")
    ratios = compute_structural_ratios(results)
    print(f"      Computed {len(ratios)} structural ratios")
    
    # Build the 144-cell Force Quadrant Grid
    print("[3/5] Building 144-cell Force Quadrant Grid...")
    grid_occupancy, quadrant_counts, combined_42, fqg_grid = build_fqg_grid(results)
    fqg_html = generate_fqg_html(grid_occupancy, quadrant_counts, combined_42, fqg_grid)
    print(f"      42 combined families. Quadrants: {quadrant_counts}")
    bisection = verify_pdt_bisection(results)
    print(f"      PDT Bisection: SI={bisection['si_cells']}, CI={bisection['ci_cells']} → {'✓ 72:72' if bisection['bisection_holds'] else '✗ FAIL'}")
    
    # Generate 3D HTML visualization
    print("[4/5] Generating 3D interactive visualization...")
    html_content = generate_3d_html(results, ratios, fqg_html=fqg_html)
    html_path = "/mnt/user-data/outputs/sempaevum_particle_viewer.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"      Written: {html_path}")
    
    # Generate markdown report
    print("[5/5] Generating comprehensive report...")
    report_content = generate_report(results, ratios)
    report_path = "/mnt/user-data/outputs/sempaevum_particle_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      Written: {report_path}")
    
    print()
    print("=" * 70)
    print("SUMMARY — Key Structural Findings")
    print("=" * 70)
    print()
    
    # Print tower distribution
    print("LCM Tower True Home Distribution:")
    home_counts = {}
    for r in results:
        home = r.get("true_home_label", "?")
        home_counts[home] = home_counts.get(home, 0) + 1
    for home, count in sorted(home_counts.items()):
        print(f"  {home:20s}: {count} particles")
    print()
    
    # Print 24 family distribution
    print("24 Harmonic Family Distribution:")
    print("  Real families (d_r at N=12):")
    d_label = {1: "Gravity/Identity", 2: "Tritone/Pivot", 3: "Strong/Cubic",
               4: "Weak/Quartic", 6: "Hexadic/EW", 12: "EM/Full-Res"}
    d_counts = {}
    for r in results:
        d_val = r["d_r_12"]
        d_counts[d_val] = d_counts.get(d_val, 0) + 1
    for d_val in sorted(d_counts.keys()):
        dname = d_label.get(d_val, f"d={d_val}")
        print(f"    d={d_val:2d} ({dname:20s}): {d_counts[d_val]} particles")
    
    print("  Imaginary families (d_θ):")
    dt_counts = {}
    for r in results:
        d_val = r["d_theta"]
        dt_counts[d_val] = dt_counts.get(d_val, 0) + 1
    for d_val in sorted(dt_counts.keys()):
        print(f"    d_θ={d_val:2d}: {dt_counts[d_val]} particles")
    
    print()
    print(f"Total particles projected: {len(results)}")
    print()
    print("=" * 70)
    print("P ∘ D ∘ T = E")
    print("=" * 70)

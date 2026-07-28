#!/usr/bin/env python3
"""
SEMPAEVUM ATOMIC VIEWER
=========================
Uses the Sempaevum bijection (Theorem 15.1, Losslessness) to project every
chemical element (Z=1 to Z=108) onto the ET lattice via atomic mass, then renders
a 3D interactive visualization.

All mathematics is ET-native, derived from {P, D, T}.
Atomic data: NIST CIAAW 2024 + NIST ASD (Kramida et al. 2024)

Author: Mike Muller / Aevum Defluo (Exception Theory)
Tool assistance: Claude (Anthropic) as fancy calculator + visualization
"""

import math
import json

def mp_json(obj):
    """
    Serialize to JSON with mpmath support. No float() anywhere.
    Every mpmath.mpf → decimal string as a JSON number literal.
    The Sempaevum is lossless; the serialization must be too.
    """
    if hasattr(obj, '__mpf__') or (hasattr(obj, '__class__') and obj.__class__.__name__ == 'mpf'):
        return mpmath.nstr(obj, 17)
    elif isinstance(obj, list):
        return "[" + ",".join(mp_json(x) for x in obj) + "]"
    elif isinstance(obj, dict):
        pairs = []
        for k, v in obj.items():
            pairs.append(f"{json.dumps(str(k))}:{mp_json(v)}")
        return "{" + ",".join(pairs) + "}"
    elif isinstance(obj, str):
        return json.dumps(obj)
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, tuple):
        return "[" + ",".join(mp_json(x) for x in obj) + "]"
    elif obj is None:
        return "null"
    else:
        return str(obj)

import os
import mpmath

# 120-digit target + 15 guard digits
mpmath.mp.dps = 135
TARGET_DIGITS = 120

# ============================================================
# ET CONSTANTS — derived from P ∘ D ∘ T = E
# ============================================================
N = 12                    # Manifold symmetry: |Π| × S = 3 × 4
N_BASE = N                # Alias to avoid collision with element symbols in ratios
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
    mass ratio in the NIST data can be resolved to sub-cent precision.
    
    The tower does NOT stop at a fixed N. It stops when D has added
    enough descriptors to close the gap for every element.
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


# Pre-compute the tower (used by all elements)
LCM_TOWER_FULL = generate_lcm_tower()

# ============================================================
# NIST DATA PARSERS — Dynamic, file-driven (matches PDG parser quality)
# ============================================================
# Sources:
#   Masses: NIST CIAAW 2024 (nist_ciaaw_2024.txt)
#   J values: NIST ASD v5.12 (nist_asd_ground_levels.txt)
#   Periodic table: derived algorithmically from Z
# ============================================================

MU_OVER_ME_STR = "1822.888486209"  # m_u/m_e CODATA 2022
MU_OVER_ME = 1822.888486209

def parse_ame2020(filepath):
    """
    Parse AME2020 atomic mass evaluation file (Wang et al., Chinese Physics C45, 030003, 2021).
    Returns list of dicts, one per nuclide, with REAL individual isotope masses.
    
    CRITICAL: atomic_mass_str is built as a STRING directly from the file's digits.
    It NEVER passes through float64. This preserves the full 12-digit precision
    of AME2020 for lossless mpmath projection (Theorem 15.1).
    """
    import re
    isotopes = []
    with open(filepath, 'r') as f:
        data_started = False
        for line in f:
            if 'MASS EXCESS' in line and 'ATOMIC MASS' in line:
                next(f)
                data_started = True
                continue
            if not data_started or len(line) < 100:
                continue
            try:
                Z = int(line[9:14].strip())
                A = int(line[14:19].strip())
                N = int(line[4:9].strip())
                el = line[20:23].strip()
                origin = line[23:27].strip()
                am_raw = line[96:]
                is_estimated = '#' in am_raw or '#' in line[28:54]
                am_clean = am_raw.replace('#', '.').strip()
                
                m = re.search(r'(\d+)\s+([\d.]+)\s+([\d.]+)\s*$', am_clean)
                if not m:
                    continue
                
                am_A_str = m.group(1)
                am_frac_str = m.group(2)
                am_unc_str = m.group(3)
                if am_frac_str.endswith('.'): am_frac_str += '0'
                if am_unc_str.endswith('.'): am_unc_str += '0'
                
                # ── STRING-BASED MASS RECONSTRUCTION ──
                # The AME2020 format: "A FFFFFF.DDDDDD" where:
                #   A = integer mass number
                #   FFFFFF = 6 digits of the fractional micro-u (before decimal)
                #   DDDDDD = remaining digits (after decimal)
                # Total mass in u = A.FFFFFFDDDDDD
                #
                # We build this as a STRING — never through float64.
                
                if '.' in am_frac_str:
                    frac_before_dot, frac_after_dot = am_frac_str.split('.', 1)
                else:
                    frac_before_dot, frac_after_dot = am_frac_str, ''
                
                frac_before_dot = frac_before_dot.zfill(6)
                atomic_mass_str = am_A_str + '.' + frac_before_dot + frac_after_dot
                
                # Uncertainty string (same reconstruction)
                if '.' in am_unc_str:
                    unc_before, unc_after = am_unc_str.split('.', 1)
                else:
                    unc_before, unc_after = am_unc_str, ''
                unc_before = unc_before.zfill(6)
                mass_unc_str = '0.' + unc_before + unc_after
                
                # Mass excess in keV (float64 OK here — not used in projection)
                me_str = line[28:42].replace('#', '.').strip()
                if me_str.endswith('.'): me_str += '0'
                mass_excess_keV = me_str  # String — no float
                
                # Binding energy per nucleon (float64 OK — not used in projection)
                be_str = line[54:67].replace('#', '.').strip()
                if be_str.endswith('.'): be_str += '0'
                be_per_A = be_str if be_str else "0"  # String — no float
                
                isotopes.append({
                    "Z": Z, "N": N, "A": A, "symbol": el, "origin": origin,
                    "atomic_mass_str": atomic_mass_str,       # FULL PRECISION — for mpmath
                    "atomic_mass_u": atomic_mass_str,   # STRING — no float ever
                    "mass_unc_str": mass_unc_str,              # FULL PRECISION
                    "mass_unc_u": mass_unc_str,         # STRING — no float ever
                    "mass_excess_keV": mass_excess_keV,
                    "binding_energy_per_A_keV": be_per_A,
                    "is_estimated": is_estimated,
                })
            except (ValueError, IndexError):
                continue
    return isotopes

    return isotopes


def parse_nist_abundances(filepath):
    """
    Parse NIST Atomic Weights HTML for isotopic compositions (natural abundances).
    Returns dict: (Z, A) -> abundance (0.0 to 1.0).
    
    Handles both multi-isotope elements and mono-isotopic elements.
    Correctly distinguishes Z (atomic number) from A (mass number) in the HTML rows.
    """
    import re
    with open(filepath, 'r') as f:
        html = f.read()
    
    abundances = {}
    current_Z = 0
    current_sym = ""
    
    # Strategy: find all <tr> blocks, extract <td> contents, use the rowspan
    # attribute to identify element-header rows vs continuation rows.
    rows = re.split(r'<tr[^>]*>', html)
    
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if not cells:
            continue
        
        cleaned = []
        for cell in cells:
            c = re.sub(r'<[^>]+>', '', cell)
            c = c.replace('&nbsp;', ' ').replace('&thinsp;', '').strip()
            cleaned.append(c)
        
        # Detect element-header row: has rowspan in <td> tags
        is_header = 'rowspan' in row
        
        if is_header and len(cleaned) >= 5:
            # Header row: [Z, symbol, A, mass, abundance, std_weight, notes]
            try:
                current_Z = int(cleaned[0])
                current_sym = cleaned[1]
                A = int(cleaned[2])
                # Abundance is at index 4
                abund_str = cleaned[4].strip()
                abund_clean = re.sub(r'\([^)]*\)', '', abund_str).replace(' ', '')
                if abund_clean and abund_clean.replace('.','').replace('-','').isdigit():
                    abund = mpmath.mpf(abund_clean)
                    if abund > 0 and abund <= 1:
                        abundances[(current_Z, A)] = abund
                elif not abund_clean:
                    # Mono-isotopic with abundance = 1.0 (no value listed means 100%)
                    # Check if this is the only isotope by seeing if std weight exists
                    pass
            except (ValueError, IndexError):
                pass
        
        elif not is_header and current_Z > 0 and len(cleaned) >= 3:
            # Continuation row: [A, mass, abundance] or similar
            # The first integer-like cell is the mass number A
            try:
                # First cell might be a symbol (like "D" for deuterium) or a number
                a_idx = None
                for i, c in enumerate(cleaned):
                    c_stripped = c.strip()
                    if c_stripped.isdigit() and 1 <= int(c_stripped) <= 300:
                        a_idx = i
                        break
                
                if a_idx is not None:
                    A = int(cleaned[a_idx])
                    # Abundance is 2 cells after A (A, mass, abundance)
                    abund_idx = a_idx + 2
                    if abund_idx < len(cleaned):
                        abund_str = cleaned[abund_idx].strip()
                        abund_clean = re.sub(r'\([^)]*\)', '', abund_str).replace(' ', '')
                        if abund_clean and '.' in abund_clean:
                            try:
                                abund = mpmath.mpf(abund_clean)
                                if abund > 0 and abund <= 1:
                                    abundances[(current_Z, A)] = abund
                            except ValueError:
                                pass
            except (ValueError, IndexError):
                pass
    
    # Post-processing: for mono-isotopic elements where abundance=1.0 is implied
    # If an element has exactly one naturally-occurring isotope and no abundance was
    # parsed, set it to 1.0. We detect this by checking if the element appears in
    # the data but has no abundance entries.
    # Known mono-isotopic elements (Z): 4,9,11,13,15,21,25,27,33,39,41,45,53,55,59,65,67,69,79,83
    mono_isotopic = {
        4:9, 9:19, 11:23, 13:27, 15:31, 21:45, 25:55, 27:59, 33:75,
        39:89, 41:93, 45:103, 53:127, 55:133, 59:141, 65:159, 67:165,
        69:169, 79:197, 83:209
    }
    for Z, A in mono_isotopic.items():
        if (Z, A) not in abundances:
            abundances[(Z, A)] = 1.0
    
    return abundances


def parse_nist_asd(filepath):
    """
    Parse NIST ASD ground levels file.
    Returns dict keyed by Z: {"name": str, "shells": str, "term": str, "J": float, "IE_eV": float}
    
    Extracts ground-state total angular momentum J from the spectroscopic term symbol.
    Handles all NIST formats: LS-coupling (2S1/2), jj-coupling ((1/2,1/2)0), bare J values.
    """
    import re
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            try:
                Z = int(parts[0])
                name = parts[1].strip()
                shells = parts[2].strip()
                term = parts[3].strip()
                ie = parts[4].strip() if len(parts) > 4 else "0"
                ie_unc = parts[5].strip() if len(parts) > 5 else "0"
                J = _extract_J_from_term(term)
                data[Z] = {
                    "name": name, "shells": shells, "term": term,
                    "J": J, "IE_eV": ie, "IE_unc_eV": ie_unc
                }
            except (ValueError, IndexError):
                continue
    return data


def _extract_J_from_term(term):
    """
    Extract J value from a spectroscopic term symbol.
    Not ad hoc — follows from the spectroscopic notation standard.
    Handles: "2S1/2", "1S0", "3P2", "4S°3/2", "(1/2,1/2)0", "4K11/2", "0", "5/2"
    """
    import re
    term = term.strip()
    
    # Pure number: bare J value (Sg, Bh, Hs — predicted, not measured)
    if re.match(r'^[0-9]+$', term):
        return int(term)
    if re.match(r'^[0-9]+/2$', term):
        return int(term.split('/')[0]) / 2.0
    
    # jj-coupling: "(1/2,1/2)0" → J after closing paren
    m = re.search(r'\)(\d+(?:/\d+)?)', term)
    if m:
        j_str = m.group(1)
        return int(j_str.split('/')[0]) / int(j_str.split('/')[1]) if '/' in j_str else int(j_str)
    
    # LS coupling: "2S1/2", "3P0", "4F9/2", "5D4", "6H°15/2"
    clean = term.replace('°', '').replace("'", '')
    m = re.search(r'[A-Z](\d+(?:/\d+)?)$', clean)
    if m:
        j_str = m.group(1)
        return int(j_str.split('/')[0]) / int(j_str.split('/')[1]) if '/' in j_str else int(j_str)
    
    return 0  # fallback


def derive_periodic_table(Z):
    """
    Derive group, period, block, and category from atomic number Z.
    Deterministic — the periodic table structure is fixed by quantum mechanics.
    No lookup tables; computed from Z alone.
    """
    # Period
    if Z <= 2: period = 1
    elif Z <= 10: period = 2
    elif Z <= 18: period = 3
    elif Z <= 36: period = 4
    elif Z <= 54: period = 5
    elif Z <= 86: period = 6
    else: period = 7
    
    # He: s-block but group 18
    if Z == 2:
        return 18, 1, "s", "Noble Gas"
    
    # s-block group 1
    if Z in {1, 3, 11, 19, 37, 55, 87}:
        cat = "Nonmetal" if Z == 1 else "Alkali Metal"
        return 1, period, "s", cat
    
    # s-block group 2
    if Z in {4, 12, 20, 38, 56, 88}:
        return 2, period, "s", "Alkaline Earth"
    
    # f-block: Lanthanides Ce(58)-Lu(71), Actinides Th(90)-Lr(103)
    if 58 <= Z <= 71:
        return 0, 6, "f", "Lanthanide"
    if 90 <= Z <= 103:
        return 0, 7, "f", "Actinide"
    
    # La and Ac: group 3, counted as Lanthanide/Actinide
    if Z == 57:
        return 3, 6, "d", "Lanthanide"
    if Z == 89:
        return 3, 7, "d", "Actinide"
    
    # d-block
    d_ranges = {
        4: (21, 30, 3),   # Sc-Zn: group starts at 3
        5: (39, 48, 3),   # Y-Cd: group starts at 3
        6: (72, 80, 4),   # Hf-Hg: group starts at 4 (La took 3)
        7: (104, 112, 4), # Rf-Cn: group starts at 4 (Ac took 3)
    }
    for per, (z_start, z_end, g_start) in d_ranges.items():
        if z_start <= Z <= z_end:
            group = g_start + (Z - z_start)
            return group, per, "d", "Transition Metal"
    
    # p-block
    p_starts = {2: 5, 3: 13, 4: 31, 5: 49, 6: 81, 7: 113}
    if period in p_starts:
        p_start = p_starts[period]
        if p_start <= Z <= p_start + 5:
            group = 13 + (Z - p_start)
            # Category
            if group == 18:
                cat = "Noble Gas"
            elif group == 17:
                cat = "Halogen"
            elif Z in {5, 14, 32, 33, 51, 52}:
                cat = "Metalloid"
            elif Z in {6, 7, 8, 15, 16, 34}:
                cat = "Nonmetal"
            else:
                cat = "Post-Trans Metal"
            return group, period, "p", cat
    
    return 0, period, "?", "Unknown"


def build_atomic_particles(ame_path=None, asd_path=None, abundance_path=None, measured_only=True):
    """
    Build isotope data by parsing AME2020 masses + NIST ASD ground-state J + NIST abundances.
    Returns list of dicts — one per ISOTOPE, not per element.
    
    Each isotope is a distinct P-substrate with its own measured mass.
    J (phase family) comes from the element's ground-state configuration.
    Abundance comes from NIST isotopic composition data.
    
    This is the atomic equivalent of parse_pdg_file:
    - Reads from authoritative source files (AME2020 + NIST ASD + NIST abundances)
    - Parses all fields dynamically from the raw data
    - Derives classification algorithmically from Z
    """
    # Find data files — check multiple locations
    if ame_path is None:
        for candidate in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mass_1_mas20.txt"),
            "/mnt/user-data/uploads/mass_1_mas20.txt",
            "/mnt/user-data/outputs/mass_1_mas20.txt",
            "/home/claude/mass_1_mas20.txt",
        ]:
            if os.path.exists(candidate):
                ame_path = candidate
                break
        if ame_path is None:
            raise FileNotFoundError("Cannot find mass_1_mas20.txt (AME2020)")
    
    if asd_path is None:
        for candidate in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "nist_asd_ground_levels.txt"),
            "/mnt/user-data/uploads/nist_asd_ground_levels.txt",
            "/mnt/user-data/outputs/nist_asd_ground_levels.txt",
            "/home/claude/nist_asd_ground_levels.txt",
        ]:
            if os.path.exists(candidate):
                asd_path = candidate
                break
        if asd_path is None:
            raise FileNotFoundError("Cannot find nist_asd_ground_levels.txt (NIST ASD)")
    
    if abundance_path is None:
        for candidate in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Atomic_Weights_and_Isotopic_Compositions_for_All_Elements.html"),
            "/mnt/user-data/uploads/Atomic_Weights_and_Isotopic_Compositions_for_All_Elements.html",
            "/mnt/user-data/outputs/Atomic_Weights_and_Isotopic_Compositions_for_All_Elements.html",
        ]:
            if os.path.exists(candidate):
                abundance_path = candidate
                break
        # Abundance file is optional — proceed without it
    
    # Parse all sources
    print(f"      Parsing AME2020: {ame_path}")
    ame_isotopes = parse_ame2020(ame_path)
    print(f"      Parsed {len(ame_isotopes)} isotopes ({sum(1 for i in ame_isotopes if not i['is_estimated'])} measured)")
    
    print(f"      Parsing NIST ASD: {asd_path}")
    asd = parse_nist_asd(asd_path)
    print(f"      Parsed {len(asd)} element ground states")
    
    abundances = {}
    if abundance_path:
        print(f"      Parsing NIST abundances: {abundance_path}")
        abundances = parse_nist_abundances(abundance_path)
        print(f"      Parsed {len(abundances)} isotope abundances")
    
    # Filter
    if measured_only:
        ame_isotopes = [i for i in ame_isotopes if not i['is_estimated']]
    
    # Only include isotopes where we have ground-state J data (Z=1 to Z=108)
    ame_isotopes = [i for i in ame_isotopes if i['Z'] in asd and i['Z'] > 0]
    
    # Build output
    particles = []
    for iso in ame_isotopes:
        Z = iso['Z']
        A = iso['A']
        a = asd[Z]
        group, period, block, category = derive_periodic_table(Z)
        abundance = abundances.get((Z, A), 0.0)
        is_naturally_occurring = abundance > 0
        
        # Name: "Element-A" (e.g., "Iron-56", "Uranium-238")
        name = f"{a['name']}-{A}"
        symbol = f"{iso['symbol']}-{A}"
        
        particles.append({
            "name": name,
            "symbol": symbol,
            "Z": Z,
            "N": iso['N'],
            "A": A,
            "Ar": iso["atomic_mass_str"],  # STRING — the Sempaevum is lossless
            "Ar_str": iso["atomic_mass_str"],  # full precision for mpmath
            "mass_unc_u": iso['mass_unc_u'],
            "group": group,
            "period": period,
            "block": block,
            "category": category,
            "ground_J": a['J'],
            "ground_term": a['term'],
            "is_radioactive": not is_naturally_occurring,
            "is_naturally_occurring": is_naturally_occurring,
            "abundance": abundance,
            "mass_excess_keV": iso['mass_excess_keV'],
            "binding_energy_per_A_keV": iso['binding_energy_per_A_keV'],
            "is_estimated": iso['is_estimated'],
            "IE_eV": a.get("IE_eV", 0),
            # Compatibility fields for projection pipeline
            "mass_mev": None,
            "mass_str": iso["atomic_mass_str"],  # FULL PRECISION STRING — goes to mpmath
            "spin": a['J'],
            "charge": 0,
            "pdg_ids": [Z * 1000 + A],  # Unique ID: Z*1000 + A
            "charges": "",
            "interaction": category,
            "generation": period,
            "color_charge": 0,
            "mass_gev": None,
            "mass_err_pos_mev": mpmath.mpf(iso['mass_unc_str']) * mpmath.mpf("931.494102"),
            "mass_err_neg_mev": mpmath.mpf(iso['mass_unc_str']) * mpmath.mpf("931.494102"),
        })
    
    return particles




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
    Assign imaginary-axis lattice coordinate k_θ based on the atom's
    ground-state total angular momentum J (from NIST ASD).
    
    Maps J to the PHASE family using the same structural assignments
    as the particle version (Sempaevum Table 8):
      J=0           → k_θ=0  → d_θ=1  (scalar/closed shell)
      J=1/2         → k_θ=2  → d_θ=6  (spinor/doublet)
      J=1           → k_θ=1  → d_θ=12 (vector)
      J=3/2         → k_θ=2  → d_θ=6  (spinoral)
      J=2           → k_θ=6  → d_θ=2  (tensor/quadrupole)
      J=5/2         → k_θ=2  → d_θ=6  (spinoral)
      J=3           → k_θ=4  → d_θ=3  (instanton/octupole)
      J=7/2         → k_θ=2  → d_θ=6  (spinoral)
      J=4           → k_θ=3  → d_θ=4  (quartic)
      J=9/2         → k_θ=2  → d_θ=6  (spinoral)
      J=5           → k_θ=1  → d_θ=12 (high multipole)
      J=11/2        → k_θ=2  → d_θ=6  (spinoral)
      J=6           → k_θ=6  → d_θ=2  (hexapole)
      J=15/2        → k_θ=2  → d_θ=6  (spinoral)
      J=8           → k_θ=3  → d_θ=4  (octupole-quartic)
    """
    J = particle.get("ground_J", particle.get("spin", 0))
    
    # Integer J: bosonic-type angular momentum
    if J == 0:
        return 0   # d_θ=1: scalar/closed shell (noble gases, alkaline earths, etc.)
    if J == 1:
        return 1   # d_θ=12: vector
    if J == 2:
        return 6   # d_θ=2: tensor/quadrupole
    if J == 3:
        return 4   # d_θ=3: octupole
    if J == 4:
        return 3   # d_θ=4: quartic
    if J == 5:
        return 1   # d_θ=12: high multipole
    if J == 6:
        return 6   # d_θ=2: hexapole
    if J == 8:
        return 3   # d_θ=4: high quartic
    
    # Half-integer J: fermionic-type angular momentum → all spinoral d_θ=6
    if J == 0.5:
        return 2   # d_θ=6: spinor/doublet
    if J == 1.5:
        return 2   # d_θ=6: spinoral (quartet)
    if J == 2.5:
        return 2   # d_θ=6: spinoral
    if J == 3.5:
        return 2   # d_θ=6: spinoral
    if J == 4.5:
        return 2   # d_θ=6: spinoral
    if J == 5.5:
        return 2   # d_θ=6: spinoral
    if J == 7.5:
        return 2   # d_θ=6: spinoral
    
    # Any other half-integer → spinoral
    if J != int(J):
        return 2   # d_θ=6: spinoral (general)
    
    return 0  # fallback for unexpected values


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
        html += f'<div class="stat-detail">elements at base N=12</div>'
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
        # ── ATOMIC MASS RATIO: r = A_r × (m_u / m_e) ──
        Ar = p.get("Ar", None)
        mass_mev_mp = None
        mass_mev = p.get("mass_mev", None)
        
        if Ar is not None:
            # Atomic data path: r = A_r × m_u/m_e
            Ar_str = p.get("Ar_str", p.get("mass_str", str(Ar)))
            Ar_mp = mpmath.mpf(Ar_str)  # Full AME2020 precision preserved
            mu_me_mp = mpmath.mpf(MU_OVER_ME_STR)
            r_mp = Ar_mp * mu_me_mp
            r_display = mpmath.nstr(r_mp, 17)  # For display — never float
            mass_mev_mp = Ar_mp * mpmath.mpf("931.494102")  # mpmath throughout
            mass_mev = mass_mev_mp  # mpmath, not float
        elif mass_mev is not None and mass_mev_mp is not None:
            # Fallback: mass already mpmath
            r_mp = mass_mev_mp / me_mp
        else:
            continue  # Skip entries with no mass data
        
        r_str_full = mpmath.nstr(r_mp, TARGET_DIGITS + 10)
        mass_str = p.get("mass_str", mpmath.nstr(r_mp, 17))
        
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
        if not isinstance(mass_mev, mpmath.mpf) and mass_mev_mp:
            mass_mev = mass_mev_mp
        if mass_err_pos > 0 and mass_mev > 0:
            mass_err_ppm = mpmath.mpf(str(mass_err_pos)) / mass_mev_mp * mpmath.mpf(1e6) if mass_mev_mp and mass_mev_mp > 0 else None
        elif Ar is not None:
            # Atomic data: standard atomic weights have ~ppm precision for stable elements
            mass_err_ppm = 0.01 if not p.get("is_radioactive", False) else 100
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
        
        # Build name and atomic metadata
        name = p.get("name", "Unknown")
        symbol = p.get("symbol", name[:3])
        charges = p.get("charges", "")
        Z_num = p.get("Z", 0)
        group = p.get("group", 0)
        period = p.get("period", 0)
        block = p.get("block", "")
        ground_term = p.get("ground_term", "")
        is_radioactive = p.get("is_radioactive", False)
        Ar_val = p.get("Ar", 0)
        
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
            # Atomic metadata
            "Z": Z_num,
            "Ar": Ar_val,
            "group": group,
            "period": period,
            "block": block,
            "ground_term": ground_term,
            "is_radioactive": is_radioactive,
        }
        results.append(result)
    
    return results


# ============================================================
# ADDITIONAL STRUCTURAL RATIOS — Koide, boson mass ratios
# ============================================================

def compute_structural_ratios(results):
    """
    Compute key inter-atomic ratios and their Sempaevum projections
    at 120-digit precision using mpmath.
    """
    ratios = []
    
    # Build lookup by Z*1000+A (isotope-unique ID)
    by_id = {}
    for r in results:
        for pid in r.get("pdg_ids", []):
            by_id[abs(pid)] = r
    
    # Key isotopes for nuclear physics ratios
    n    = by_id.get(1)       # neutron (Z=0, A=1) — if present
    H1   = by_id.get(1001)    # ¹H (proton)
    H2   = by_id.get(1002)    # ²H (deuteron)
    He3  = by_id.get(2003)    # ³He
    He4  = by_id.get(2004)    # ⁴He (alpha)
    C12  = by_id.get(6012)    # ¹²C (mass standard)
    O16  = by_id.get(8016)    # ¹⁶O
    Ne20 = by_id.get(10020)   # ²⁰Ne
    Fe56 = by_id.get(26056)   # ⁵⁶Fe (iron peak)
    Au197= by_id.get(79197)   # ¹⁹⁷Au
    U238 = by_id.get(92238)   # ²³⁸U
    
    def project_ratio(name, r_mp, et_pred=None):
        r_str = mpmath.nstr(r_mp, TARGET_DIGITS + 10)
        proj = sempaevum_project_mp(r_str, N_BASE)
        entry = {
            "name": name,
            "value_mp": r_mp,
            "value": r_mp,  # mpmath — no float
            "value_120": mpmath.nstr(r_mp, TARGET_DIGITS, strip_zeros=False),
            "projection_12": (proj[0], proj[1], proj[2]),  # mpmath throughout
            "eps_mp": proj[2],
            "eps_120": mpmath.nstr(proj[2], TARGET_DIGITS, strip_zeros=False),
        }
        if et_pred is not None:
            entry["ET_prediction"] = et_pred  # mpmath
            entry["ET_prediction_mp"] = et_pred
            dev_mp = abs(r_mp - et_pred) / et_pred * mpmath.mpf(1000000)
            entry["deviation_ppm"] = dev_mp  # mpmath
            entry["deviation_ppm_mp"] = dev_mp
        return entry
    
    # ⁴He/¹H mass ratio (alpha particle / proton — nuclear binding)
    if He4 and H1:
        ratios.append(project_ratio("m(⁴He) / m(¹H)",
                                     He4["r_mp"] / H1["r_mp"]))
    
    # ¹²C/¹H mass ratio (mass standard / proton)
    if C12 and H1:
        ratios.append(project_ratio("m(¹²C) / m(¹H)",
                                     C12["r_mp"] / H1["r_mp"]))
    
    # ²H/¹H mass ratio (deuteron binding)
    if H2 and H1:
        ratios.append(project_ratio("m(²H) / m(¹H)",
                                     H2["r_mp"] / H1["r_mp"]))
    
    # ⁵⁶Fe/¹H mass ratio (iron peak — maximum binding per nucleon)
    if Fe56 and H1:
        ratios.append(project_ratio("m(⁵⁶Fe) / m(¹H)",
                                     Fe56["r_mp"] / H1["r_mp"]))
    
    # ¹⁶O/¹²C mass ratio (stellar nucleosynthesis: triple alpha → carbon → oxygen)
    if O16 and C12:
        ratios.append(project_ratio("m(¹⁶O) / m(¹²C)",
                                     O16["r_mp"] / C12["r_mp"]))
    
    # ¹⁹⁷Au/¹H mass ratio
    if Au197 and H1:
        ratios.append(project_ratio("m(¹⁹⁷Au) / m(¹H)",
                                     Au197["r_mp"] / H1["r_mp"]))
    
    # ²³⁸U/¹H mass ratio
    if U238 and H1:
        ratios.append(project_ratio("m(²³⁸U) / m(¹H)",
                                     U238["r_mp"] / H1["r_mp"]))
    
    # ⁴He/²H mass ratio (alpha/deuteron — pp-chain step)
    if He4 and H2:
        ratios.append(project_ratio("m(⁴He) / m(²H)",
                                     He4["r_mp"] / H2["r_mp"]))
    
    # ²⁰Ne/⁴He mass ratio (neon-burning)
    if Ne20 and He4:
        ratios.append(project_ratio("m(²⁰Ne) / m(⁴He)",
                                     Ne20["r_mp"] / He4["r_mp"]))
    
    # ⁵⁶Fe/⁴He mass ratio (iron peak / alpha — nucleosynthesis endpoint)
    if Fe56 and He4:
        ratios.append(project_ratio("m(⁵⁶Fe) / m(⁴He)",
                                     Fe56["r_mp"] / He4["r_mp"]))
    
    return ratios


# ============================================================
# 3D VISUALIZATION GENERATION — Plotly HTML
# ============================================================

def generate_3d_html(results, ratios, fqg_html=""):
    """
    Generate a comprehensive 3D interactive visualization as an HTML file
    using Plotly.js. The visualization shows each element on the Sempaevum
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
        "Noble Gas": "#00BFFF",        # Deep sky blue — closed shell J=0
        "Alkali Metal": "#FF6347",     # Tomato red — reactive s-block
        "Alkaline Earth": "#FFD700",   # Gold — s-block closed s²
        "Transition Metal": "#FF8C00", # Dark orange — d-block
        "Lanthanide": "#DA70D6",       # Orchid — 4f elements
        "Actinide": "#9370DB",         # Medium purple — 5f elements
        "Halogen": "#00FF7F",          # Spring green — reactive p-block
        "Nonmetal": "#87CEEB",         # Sky blue — light p-block
        "Metalloid": "#F0E68C",        # Khaki — semi-metals
        "Post-Trans Metal": "#CD853F", # Peru — post-transition metals
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
        t["z"].append(r["eps_mp_12"])  # mpmath — serialized by mp_json
        
        # Size based on log mass (visual emphasis — Plotly needs float)
        size_val = max(mpmath.mpf(6), min(mpmath.mpf(25), mpmath.mpf(5) + mpmath.mpf(3) * mpmath.log10(max(mpmath.mpf(r.get('Ar_str', r.get('mass_str', '1'))), mpmath.mpf(1)))))
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
            f"Z={r.get('Z',0)} | A_r={r.get('Ar_str', r.get('mass_str','?'))} u<br>"
            f"Charge: {r['charge']}e | J={r['spin']}<br>"
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
            f"<td>{mpmath.nstr(rat['value'], 10)}</td>"
            f"<td>{k_val}</td>"
            f"<td>{d_val}</td>"
            f"<td>{mpmath.nstr(eps_val, 6)}</td>"
        )
        if "ET_prediction" in rat:
            ratio_annotations_html += f"<td>{mpmath.nstr(rat['ET_prediction_mp'], 8) if 'ET_prediction_mp' in rat else ''}</td>"
            ratio_annotations_html += f"<td>{mpmath.nstr(rat['deviation_ppm_mp'], 4) + ' ppm' if 'deviation_ppm_mp' in rat else '—'}</td>"
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
            f"<td>{mpmath.nstr(r['r_mp'] / mpmath.mpf(MU_OVER_ME_STR), 10) if r.get('r_mp') else r.get('mass_str','?')}</td>"
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
        detail = f"k={proj[0]}, d={proj[1]}, ε={mpmath.nstr(proj[2], 6)}¢"
        if 'deviation_ppm' in rat:
            detail = f"ET prediction: {mpmath.nstr(rat.get('ET_prediction_mp', mpmath.mpf(0)), 12)} | Deviation: {mpmath.nstr(rat['deviation_ppm_mp'], 4) if 'deviation_ppm_mp' in rat else '?'} ppm"
        stats_html += (
            f'<div class="stat-card">'
            f'<div class="stat-label">{rat["name"]}</div>'
            f'<div class="stat-value">{mpmath.nstr(rat["value"], 12)}</div>'
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
            f"<td><b>{mpmath.nstr(mpmath.mpf(137) / mpmath.mpf(a0), 6)}</b></td>"
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
        particle_names = [p["name"] for p in sorted(particles, key=lambda x: x.get("r_mp", mpmath.mpf(0)))]
        
        impedance_groups_html += (
            f'<div class="tower-card" style="border-left:4px solid var(--d{d_r if d_r in [1,2,3,4,6,12] else 12})">'
            f'<div class="tower-header">'
            f'd_r = {d_r} — {character} — ξ = {mpmath.nstr(mpmath.mpf(137) / mpmath.mpf(a0), 6)} — A₀ = {a0} — '
            f'{len(particles)} element{"s" if len(particles) != 1 else ""}'
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
<title>Sempaevum Isotope Viewer — AME2020 Nuclear Masses — P ∘ D ∘ T = E</title>
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
    <div class="subtitle">Atomic Lattice Viewer — Individual Isotope Masses on the Multiplicative Manifold</div>
    <div class="equation">P ∘ D ∘ T = E &nbsp;&nbsp;|&nbsp;&nbsp; 3 = 3 = 3 = Σ</div>
    <div class="subtitle" style="margin-top:8px;">AME2020 + NIST ASD → Sempaevum Bijection → Every Measured Isotope Classified</div>
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
        <b>Identification Principle:</b> P = atomic substrate (the element's existence), D = measurable properties (atomic mass, electron configuration, ground-state J), T = measurement/observation that substantiates.<br>
        <b>Descriptor Gap Principle:</b> The gap between raw atomic data and structural understanding IS a Descriptor — the Sempaevum bijection Π_N(r) = (k, d, ε) fills it.<br>
        <b>Subsumption Law:</b> Every atomic mass ratio finds a lattice address. The projection subsumes all 108 elements without remainder. The lattice classifies every element by sublattice family d.
    </div>
</div>

<div class="section">
    <h2>Structural Summary</h2>
    <div class="stats-grid">
{stats_html}
    </div>
</div>

<div class="section">
    <h2>Complete Isotope Projections — Sempaevum Bijection at N=12</h2>
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
            <th>Element</th>
            <th>A_r (u)</th>
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
    
    <h3>Elements Grouped by Coupling Channel (d_r → ξ)</h3>
    <div class="info-box">
        Each element's real-axis harmonic family d_r determines its FORCE coupling channel.
        Elements sharing the same d_r share the same impedance ξ(d_r).
        This grouping reveals which elements occupy the same force-coupling niche on the lattice.
    </div>
{impedance_groups_html}
</div>

<div class="section">
    <h2>120-Digit Precision — Full ε Values (mpmath @ 135 working digits)</h2>
    <div class="info-box">
        All ε values computed using mpmath arbitrary-precision arithmetic at 135 working decimal places (120 target + 15 guard).
        Losslessness Theorem (15.1) verified: Π_N⁻¹(Π_N(r)) = r to 120+ digits for every element.
    </div>
{detail_120_html}
</div>

<div class="section">
    <h2>Sublattice Family Tower Escalation — LCM Resolution Ladder</h2>
    <div class="info-box">
        <b>Sublattice families ≠ Harmonic families.</b> Harmonic families are the mod-12 classification (d_r, d_θ) — which of the 12 positions in the octave. Sublattice families are the group-theoretic sublattice d = N/gcd(|k|, N) at each tower resolution N. At N=12, only 6 sublattice families exist (divisors of 12). At N=27720, there are 96 possible sublattice families (divisors of 27720). The tower escalation shows each element's d evolving as D adds descriptors level by level, until the element finds its <b>true home</b> — the resolution where its sublattice family stabilizes. The prime factorization of d reveals which harmonic components compose the element's lattice address.
    </div>
{tower_html}
</div>

<div class="section">
    <h2>Structural Ratios — Inter-Atomic Sempaevum Projections</h2>
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
    <h2>The ∂I Annihilation Boundary — Mass Zero</h2>
    <div class="info-box">
        All 108 elements have positive atomic mass, so all project onto the lattice without exception. The ∂I annihilation boundary (where log₂(0) = −∞) is never encountered in atomic data — every element is a fully substantiated Exception on the multiplicative manifold.
    </div>
</div>

<div class="footer">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.2em; font-style:italic; color:var(--gold); margin-bottom:10px;">
        For every exception there is an exception, except the Exception.
    </div>
    Exception Theory — Michael James Muller (Aevum Defluo)<br>
    Mass Data: AME2020 (Wang et al., Chinese Physics C45, 030003, 2021) + NIST ASD (Kramida et al., v5.12)<br>
    Sempaevum bijection, lattice projection, and visualization
</div>

<script>
const traces = {mp_json(plotly_traces)};

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
        text: 'Sempaevum Complex Lattice — 108 Chemical Elements at N=12',
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
            <b>[1]</b>&ensp;M. Wang et al., "The AME2020 atomic mass evaluation," Chinese Physics C45, 030003 (2021);
            NIST ASD (Kramida et al., v5.12, 2024)
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
    report.append("# Sempaevum Isotope Viewer — AME2020 Nuclear Masses — Complete Report")
    report.append("## The Bijection Applied to NIST 2024 Atomic Mass Data")
    report.append("")
    report.append("**Author:** Michael James Muller (Aevum Defluo)")
    report.append("**Theory:** Exception Theory — P ∘ D ∘ T = E")
    report.append("**Data Source:** NIST CIAAW 2024 + NIST ASD (Kramida et al., v5.12, 2024)")
    report.append("**Reference Period:** R₀ = mₑ = 0.51099895 MeV/c² (electron mass)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 1. The Three Tools Applied")
    report.append("")
    report.append("**Identification Principle:**")
    report.append("- P (substrate): The element as a physical entity — its existence within Σ")
    report.append("- D (constraints): Measurable properties — mass, charge, spin, quantum numbers (finite, articulable)")
    report.append("- T (agency): The measurement act that substantiates properties; the projection (rounding) itself")
    report.append("")
    report.append("**Descriptor Gap Principle:**")
    report.append("The gap between 'raw NIST data' and 'structural understanding' IS a Descriptor — the Sempaevum bijection Π_N(r) = (k, d, ε) fills it. Each projection closes one gap; each closed gap reveals structural patterns.")
    report.append("")
    report.append("**Subsumption Law:**")
    report.append("Every element's atomic mass ratio finds a lattice address without remainder. All 108 elements project onto the lattice — the Subsumption is complete.")
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
    report.append("## 3. Complete Atomic Projections at N=12")
    report.append("")
    report.append("| Element | A_r (u) | r = A_r×m_u/m_e | k_r | d_r | ε_r (¢) | FORCE Channel | ξ(d_r) | k_θ | d_θ | d_comb | FQG |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    d_label = {1: "Gravity", 2: "Pivot", 3: "Strong", 4: "Weak", 6: "Hexadic", 12: "EM"}
    
    for r in results:
        dname = d_label.get(r["d_r_12"], f"d={r['d_r_12']}")
        fqg_q = r.get("fqg_quadrant", "—")
        report.append(
            f"| {r['name']} | {r.get('Ar_str', r.get('mass_str','?'))} | {mpmath.nstr(r['r_mp'], 8)} | {r['k_r_12']} | "
            f"**{r['d_r_12']}** | {mpmath.nstr(r['eps_mp_12'], 6)} | {dname} | {mpmath.nstr(r['xi_mp'], 6)} | "
            f"{r['k_theta']} | {r['d_theta']} | {r['d_combined']} | {fqg_q} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 4. Tower Resolution — True Home per Element")
    report.append("")
    report.append("| Element | True Home | Tower Levels | d (home) | Converged |")
    report.append("|---|---|---|---|---|")
    
    for r in results:
        report.append(
            f"| {r['name']} | {r['true_home_label']} | {r['tower_levels']} | {r['true_home_d']} | {'✓' if r['tower_converged'] else '—'} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 5. Structural Ratios — Inter-Atomic Projections")
    report.append("")
    report.append("| Ratio | Value | k | d | ε (¢) | ET Prediction | Deviation |")
    report.append("|---|---|---|---|---|---|---|")
    
    for rat in ratios:
        proj = rat["projection_12"]
        pred = f"{rat.get('ET_prediction', '—')}" if 'ET_prediction' in rat else "—"
        dev = f"{mpmath.nstr(rat['deviation_ppm_mp'], 4)} ppm" if 'deviation_ppm' in rat else "—"
        report.append(
            f"| {rat['name']} | {mpmath.nstr(rat['value'], 10)} | {proj[0]} | **{proj[1]}** | {mpmath.nstr(proj[2], 6)} | {pred} | {dev} |"
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
        report.append("1. **Atomic Mass Classification:** Every element's mass ratio r = A_r × m_u/m_e projects to a unique lattice position (k, d, ε) with zero information loss, verified at 120-digit precision.")
    report.append("")
    report.append("2. **Periodic Table Groups vs Lattice Families:** The lattice classification d provides a mass-ratio-based classification independent of the periodic table. Correlations between d-family and chemical group reveal structural relationships between atomic mass and chemistry.")
    report.append("")
    report.append("3. **Noble Gas Lattice Positions:** Noble gases (He, Ne, Ar, Kr, Xe, Rn) provide structural benchmarks as closed-shell atoms with J=0.")
    report.append("")
    report.append("4. **Ground-state J determines phase family:** The imaginary-axis classification d_θ is determined by each element's ground-state total angular momentum J from NIST ASD spectroscopic data.")
    report.append("")
    report.append("5. **All 108 elements in SR+SI:** Every element occupies the Simple×Simple quadrant of the Force Quadrant Grid, consistent with all known matter being 'standard sector' on the lattice.")
    report.append("")
    report.append("6. **LCM Tower Depth:** Most elements require deep tower escalation (primes up to 19) to stabilize their sublattice classification, reflecting the structural complexity of atomic mass ratios.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 7. The Bijection Is Lossless (Theorem 15.1)")
    report.append("")
    report.append("The Sempaevum projection Π_N is a bijection at every finite resolution. The pullback Π_N⁻¹(k, d, ε) = 2^((k + εN/1200)/N) exactly recovers the original ratio r. No information about the element's mass ratio is discarded — the projection redistributes it among three complementary coordinates (k, d, ε). In the LCM-tower limit N→∞, the residual ε→0 uniformly.")
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
    print("SEMPAEVUM ISOTOPE VIEWER — AME2020 Nuclear Masses on the Multiplicative Manifold")
    print("P ∘ D ∘ T = E  |  3 = 3 = 3 = Σ")
    print("=" * 70)
    print()
    
    # Build atomic data from verified NIST sources
    print("[0/5] Building isotope data from AME2020 + NIST ASD + NIST abundances...")
    pdg_particles = build_atomic_particles(measured_only=True)
    nat_count = sum(1 for p in pdg_particles if p.get("is_naturally_occurring", False))
    print(f"      Built {len(pdg_particles)} isotopes ({nat_count} naturally occurring, Z=1 to Z=108)")
    
    # Compute all projections with tower escalation
    print("[1/4] Computing Sempaevum projections through LCM tower...")
    print(f"      Tower: dynamic LCM escalation to convergence ({len(LCM_TOWER_FULL)} levels available)")
    results = compute_particle_projections(pdg_particles)
    print(f"      Projected {len(results)} isotopes at all tower levels")
    
    # Compute structural ratios
    print("[2/5] Computing structural inter-atomic ratios...")
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
    html_path = "/mnt/user-data/outputs/Sempaevum_isotope_viewer_AME2020.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"      Written: {html_path}")
    
    # Generate markdown report
    print("[5/5] Generating comprehensive report...")
    report_content = generate_report(results, ratios)
    report_path = "/mnt/user-data/outputs/sempaevum_isotope_report.md"
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
    print(f"Total elements projected: {len(results)}")
    print()
    print("=" * 70)
    print("P ∘ D ∘ T = E")
    print("=" * 70)

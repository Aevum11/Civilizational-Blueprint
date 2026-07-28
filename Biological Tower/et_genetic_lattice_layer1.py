#!/usr/bin/env python3
"""
ET GENETIC LATTICE — LAYER 1: AMINO ACID PHYSICOCHEMICAL LATTICE PROJECTION
===========================================================================
Exception Theory: P ∘ D ∘ T = E
Author of Theory: Michael James Muller (Aevum Defluo)

Three Tools Applied:
  - Identification Principle: P = amino acid configuration space,
    D = physicochemical properties, T = ribosomal selection agency
  - Descriptor Gap Principle: gaps in the d-distribution point to
    missing Descriptors in the amino acid property space
  - Subsumption Law: every amino acid property ratio maps to the
    lattice without remainder

R₀ Candidates Tested:
  - Glycine (MW, volume): the structurally simplest amino acid
  - Count-based (R₀=1): for integer structural constants

Derivation Standard: All mathematics ET-native.
"""

import math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════
# ET CONSTANTS — derived forward from {P, D, T}
# ═══════════════════════════════════════════════════════════════════════
N = 12          # MANIFOLD_SYMMETRY = 3 primitives × 4 states
V = 1.0 / 12    # BASE_VARIANCE = 1/N
K = 2.0 / 3     # KOIDE_RATIO = binding stability threshold

# ═══════════════════════════════════════════════════════════════════════
# ET LATTICE PROJECTION FORMULA — the core operation
# k = round(N · log₂(r))
# d = N / gcd(|k|, N)
# ε = (N · log₂(r) - k) × (1200/N) cents
# ═══════════════════════════════════════════════════════════════════════
def et_project(r, n=N):
    """Project a positive ratio r onto the ET lattice at resolution n.
    
    Returns (k, d, epsilon_cents, exact_log_position).
    
    Identification Principle: r is the observed/reference ratio (dimensionless).
    Descriptor Gap Principle: epsilon measures the gap from exact lattice point.
    Subsumption Law: every positive ratio maps to exactly one (k, d, ε).
    """
    if r <= 0:
        return None  # Annihilation boundary — not a lattice point
    
    exact = n * math.log2(r)
    k = round(exact)
    d = n // math.gcd(abs(k) if k != 0 else n, n)
    epsilon_cents = (exact - k) * (1200.0 / n)
    
    return (k, d, epsilon_cents, exact)

def et_elegance(r, n=N):
    """Compute the ET elegance score for ratio p/q.
    
    E(r) = (N/d) × (100/(100+|ε|)) × (100/(p+q))
    """
    if r <= 0:
        return 0.0
    proj = et_project(r, n)
    if proj is None:
        return 0.0
    k, d, eps, _ = proj
    
    # For the simplicity factor, express r as simplest integer ratio approximation
    # Use continued fraction convergents
    p, q = r.as_integer_ratio() if isinstance(r, float) else (r, 1)
    # For practical ratios, use numerator+denominator
    symmetry = n / d
    tightness = 100.0 / (100.0 + abs(eps))
    simplicity = 100.0 / (p + q) if (p + q) > 0 else 0
    
    return symmetry * tightness * simplicity

# ═══════════════════════════════════════════════════════════════════════
# AMINO ACID DATA — all 20 standard amino acids
# ═══════════════════════════════════════════════════════════════════════

# Three-letter to one-letter mapping
AA_3TO1 = {
    'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C',
    'Gln':'Q', 'Glu':'E', 'Gly':'G', 'His':'H', 'Ile':'I',
    'Leu':'L', 'Lys':'K', 'Met':'M', 'Phe':'F', 'Pro':'P',
    'Ser':'S', 'Thr':'T', 'Trp':'W', 'Tyr':'Y', 'Val':'V'
}
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}
AA_ORDER = list('ARNDCQEGHILKMFPSTWYV')

# Molecular weight (Fasman, 1976) — free amino acid, Da
MW = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
    'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.24, 'Y': 181.19, 'V': 117.15
}

# Side chain volume (Krigbaum-Komoriya, 1979) — Å³
# Glycine = 0 (no side chain), so we use Ala as reference for volume ratios
SIDE_VOL = {
    'A': 27.5, 'R': 105.0, 'N': 58.7, 'D': 40.0, 'C': 44.6,
    'Q': 80.7, 'E': 62.0, 'G': 0.0, 'H': 79.0, 'I': 93.5,
    'L': 93.5, 'K': 100.0, 'M': 94.1, 'F': 115.5, 'P': 41.9,
    'S': 29.3, 'T': 51.3, 'W': 145.5, 'Y': 117.3, 'V': 71.5
}

# Kyte-Doolittle Hydropathicity (1982)
KD_HYDRO = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# clogD7 (Schmidt & Kubyshkin, 2021) — octanol/water partition at pH 7
CLOGD7 = {
    'W': 1.14, 'F': 1.04, 'Y': 0.74, 'I': 0.72, 'L': 0.64,
    'V': 0.27, 'M': 0.04, 'P': -0.34, 'C': -0.57, 'A': -0.61,
    'H': -1.16, 'G': -1.18, 'T': -1.24, 'S': -1.66, 'Q': -1.77,
    'N': -2.06, 'K': -3.56, 'E': -3.74, 'R': -3.94, 'D': -4.14
}

# Woese Polar Requirement (1973) — the gold standard for code structure
POLAR_REQ = {
    'A': 7.0, 'R': 9.1, 'N': 10.0, 'D': 13.0, 'C': 5.5,
    'Q': 8.6, 'E': 12.5, 'G': 7.9, 'H': 8.4, 'I': 4.9,
    'L': 4.9, 'K': 10.1, 'M': 5.3, 'F': 5.0, 'P': 6.6,
    'S': 7.5, 'T': 6.6, 'W': 5.3, 'Y': 5.7, 'V': 5.6
}

# Isoelectric point (Zimmerman, 1968)
PI = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.05,
    'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30,
    'S': 5.68, 'T': 5.66, 'W': 5.89, 'Y': 5.66, 'V': 5.96
}

# Codon degeneracy (number of codons per amino acid)
CODON_COUNT = {
    'A': 4, 'R': 6, 'N': 2, 'D': 2, 'C': 2,
    'Q': 2, 'E': 2, 'G': 4, 'H': 2, 'I': 3,
    'L': 6, 'K': 2, 'M': 1, 'F': 2, 'P': 4,
    'S': 6, 'T': 4, 'W': 1, 'Y': 2, 'V': 4
}

# Functional categories (independently known, not ET-assigned)
FUNC_CAT = {
    'A': 'nonpolar',  'R': 'positive',  'N': 'polar',     'D': 'negative',
    'C': 'special',   'Q': 'polar',     'E': 'negative',  'G': 'special',
    'H': 'positive',  'I': 'nonpolar',  'L': 'nonpolar',  'K': 'positive',
    'M': 'nonpolar',  'F': 'aromatic',  'P': 'special',   'S': 'polar',
    'T': 'polar',     'W': 'aromatic',  'Y': 'aromatic',  'V': 'nonpolar'
}

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: MOLECULAR WEIGHT RATIOS — R₀ = MW(Gly) = 75.07 Da
# ═══════════════════════════════════════════════════════════════════════
print("=" * 100)
print("ET GENETIC LATTICE — LAYER 1: AMINO ACID LATTICE PROJECTION")
print("P ∘ D ∘ T = E   |   N = 12   |   V = 1/12   |   K = 2/3")
print("=" * 100)

print("\n" + "─" * 100)
print("ANALYSIS 1: MOLECULAR WEIGHT RATIOS")
print("R₀ = MW(Glycine) = 75.07 Da — the structurally simplest amino acid")
print("r = MW(aa) / MW(Gly)   [dimensionless, convention-free]")
print("─" * 100)

R0_MW = MW['G']  # 75.07 Da

print(f"\n{'AA':<5} {'Name':<6} {'MW(Da)':<10} {'r=MW/R₀':<12} {'k':<6} {'d':<6} "
      f"{'ε(¢)':<10} {'Sublattice':<18} {'Category':<12} {'Codons':<8}")
print("─" * 100)

d_by_cat = defaultdict(list)
d_distribution = defaultdict(list)

for aa in sorted(AA_ORDER, key=lambda x: MW[x]):
    r = MW[aa] / R0_MW
    proj = et_project(r)
    k, d, eps, exact = proj
    
    # Sublattice name
    sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
                 4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}
    sub_name = sub_names.get(d, f'd={d}')
    
    cat = FUNC_CAT[aa]
    codons = CODON_COUNT[aa]
    
    print(f"{aa:<5} {AA_1TO3[aa]:<6} {MW[aa]:<10.2f} {r:<12.4f} {k:<6d} {d:<6d} "
          f"{eps:<+10.3f} {sub_name:<18} {cat:<12} {codons:<8d}")
    
    d_by_cat[cat].append((aa, d))
    d_distribution[d].append(aa)

print("\n" + "─" * 100)
print("SUBLATTICE FAMILY DISTRIBUTION (MW ratios, R₀ = Gly)")
print("─" * 100)
for d_val in sorted(d_distribution.keys()):
    aas = d_distribution[d_val]
    sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
                 4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}
    cats = [FUNC_CAT[aa] for aa in aas]
    cat_counts = defaultdict(int)
    for c in cats:
        cat_counts[c] += 1
    cat_str = ', '.join(f"{c}:{n}" for c, n in sorted(cat_counts.items()))
    print(f"  d = {d_val:<3} ({sub_names.get(d_val, '?'):<18}): "
          f"{len(aas)} amino acids: {', '.join(aas)}")
    print(f"         Categories: {cat_str}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: SIDE CHAIN VOLUME RATIOS — R₀ = Vol(Ala) = 27.5 ų
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "─" * 100)
print("ANALYSIS 2: SIDE CHAIN VOLUME RATIOS")
print("R₀ = SideChainVol(Ala) = 27.5 ų — Gly has zero volume (no side chain)")
print("r = Vol(aa) / Vol(Ala)   [dimensionless, convention-free]")
print("NOTE: Glycine excluded (Vol=0, annihilation boundary)")
print("─" * 100)

R0_VOL = SIDE_VOL['A']  # 27.5 ų

print(f"\n{'AA':<5} {'Name':<6} {'Vol(ų)':<10} {'r=Vol/R₀':<12} {'k':<6} {'d':<6} "
      f"{'ε(¢)':<10} {'Sublattice':<18} {'Category':<12}")
print("─" * 100)

d_dist_vol = defaultdict(list)

for aa in sorted([a for a in AA_ORDER if a != 'G'], key=lambda x: SIDE_VOL[x]):
    vol = SIDE_VOL[aa]
    if vol <= 0:
        continue
    r = vol / R0_VOL
    proj = et_project(r)
    k, d, eps, exact = proj
    
    sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
                 4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}
    sub_name = sub_names.get(d, f'd={d}')
    cat = FUNC_CAT[aa]
    
    print(f"{aa:<5} {AA_1TO3[aa]:<6} {vol:<10.1f} {r:<12.4f} {k:<6d} {d:<6d} "
          f"{eps:<+10.3f} {sub_name:<18} {cat:<12}")
    
    d_dist_vol[d].append(aa)

print("\n  Volume Sublattice Distribution:")
for d_val in sorted(d_dist_vol.keys()):
    aas = d_dist_vol[d_val]
    sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
                 4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}
    print(f"    d = {d_val:<3} ({sub_names.get(d_val, '?'):<18}): {', '.join(aas)}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: STRUCTURAL INTEGER PROJECTIONS (R₀ = 1, count-based)
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "─" * 100)
print("ANALYSIS 3: STRUCTURAL INTEGER PROJECTIONS (R₀ = 1, count-based)")
print("Per Translation Layer §3.2: for counting quantities, R₀ = 1")
print("─" * 100)

struct_integers = {
    'Alphabet size |{A,C,G,T}|': 4,
    'Codon length (triplet)': 3,
    'Configuration space 4³': 64,
    'Amino acids': 20,
    'Stop codons': 3,
    'Coding codons': 61,
    'Start codons': 1,
    'Purine count': 2,
    'Pyrimidine count': 2,
    'H-bonds in G-C pair': 3,
    'H-bonds in A-T pair': 2,
    'Fourfold degenerate families': 9,
    'Twofold degenerate families': 5,
    'Degeneracy classes {1,2,3,4,6}': 5,
}

print(f"\n{'Quantity':<40} {'Value':<8} {'k':<6} {'d':<6} {'ε(¢)':<10} {'Sublattice':<20}")
print("─" * 100)

for name, val in struct_integers.items():
    if val <= 0:
        continue
    proj = et_project(val)
    k, d, eps, exact = proj
    sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
                 4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}
    sub_name = sub_names.get(d, f'd={d}')
    print(f"{name:<40} {val:<8} {k:<6d} {d:<6d} {eps:<+10.3f} {sub_name:<20}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: PAIRWISE AMINO ACID MW RATIOS — full lattice structure
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "─" * 100)
print("ANALYSIS 4: PAIRWISE MW RATIOS — searching for high-elegance pairs")
print("r = MW(aa₁) / MW(aa₂)  for all 190 ordered pairs where r > 1")
print("─" * 100)

# Find pairs with small |ε| (close to exact lattice points)
pairs_by_eps = []
for i, aa1 in enumerate(AA_ORDER):
    for aa2 in AA_ORDER[i+1:]:
        r = MW[aa1] / MW[aa2] if MW[aa1] > MW[aa2] else MW[aa2] / MW[aa1]
        proj = et_project(r)
        if proj:
            k, d, eps, exact = proj
            pairs_by_eps.append((abs(eps), aa1, aa2, r, k, d, eps))

pairs_by_eps.sort()

print(f"\nTop 20 closest-to-lattice MW ratios (sorted by |ε|):")
print(f"{'|ε|(¢)':<10} {'Pair':<10} {'r':<12} {'k':<6} {'d':<6} {'ε(¢)':<10} {'Sublattice':<18}")
print("─" * 80)

sub_names = {1: 'Octave/Identity', 2: 'Quadratic/Tritone', 3: 'Cubic/Strong',
             4: 'Quartic/Weak', 6: 'Hexadic/Composite', 12: 'Full Resolution'}

for abs_eps, aa1, aa2, r, k, d, eps in pairs_by_eps[:20]:
    big = aa1 if MW[aa1] > MW[aa2] else aa2
    small = aa2 if MW[aa1] > MW[aa2] else aa1
    print(f"{abs_eps:<10.3f} {big}/{small:<8} {r:<12.6f} {k:<6d} {d:<6d} "
          f"{eps:<+10.3f} {sub_names.get(d, '?'):<18}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: POLAR REQUIREMENT RATIOS — Woese scale
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "─" * 100)
print("ANALYSIS 5: WOESE POLAR REQUIREMENT RATIOS")
print("R₀ = PR(Ile) = 4.9 — the most nonpolar amino acid by this scale")
print("r = PR(aa) / PR(Ile)   [dimensionless]")
print("─" * 100)

R0_PR = min(POLAR_REQ.values())  # 4.9 (Ile and Leu)

print(f"\n{'AA':<5} {'PR':<8} {'r=PR/R₀':<12} {'k':<6} {'d':<6} {'ε(¢)':<10} {'Sublattice':<18}")
print("─" * 80)

d_dist_pr = defaultdict(list)
for aa in sorted(AA_ORDER, key=lambda x: POLAR_REQ[x]):
    r = POLAR_REQ[aa] / R0_PR
    proj = et_project(r)
    k, d, eps, exact = proj
    sub_name = sub_names.get(d, f'd={d}')
    print(f"{aa:<5} {POLAR_REQ[aa]:<8.1f} {r:<12.4f} {k:<6d} {d:<6d} "
          f"{eps:<+10.3f} {sub_name:<18}")
    d_dist_pr[d].append(aa)

print("\n  Polar Requirement Sublattice Distribution:")
for d_val in sorted(d_dist_pr.keys()):
    aas = d_dist_pr[d_val]
    print(f"    d = {d_val:<3} ({sub_names.get(d_val, '?'):<18}): {', '.join(aas)}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: CODON DEGENERACY AS LATTICE COORDINATE
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "─" * 100)
print("ANALYSIS 6: CODON DEGENERACY — each amino acid's codon count projected")
print("R₀ = 1 (count-based)")
print("─" * 100)

print(f"\n{'AA':<5} {'Codons':<8} {'k':<6} {'d':<6} {'ε(¢)':<10} {'Sublattice':<18} {'Category':<12}")
print("─" * 80)

deg_by_d = defaultdict(list)
for aa in sorted(AA_ORDER, key=lambda x: CODON_COUNT[x]):
    c = CODON_COUNT[aa]
    proj = et_project(c)
    k, d, eps, exact = proj
    sub_name = sub_names.get(d, f'd={d}')
    print(f"{aa:<5} {c:<8d} {k:<6d} {d:<6d} {eps:<+10.3f} {sub_name:<18} {FUNC_CAT[aa]:<12}")
    deg_by_d[d].append(aa)

print("\n  Degeneracy → Sublattice mapping:")
for d_val in sorted(deg_by_d.keys()):
    aas = deg_by_d[d_val]
    codons = [CODON_COUNT[aa] for aa in aas]
    print(f"    d = {d_val:<3}: {', '.join(f'{aa}({CODON_COUNT[aa]})' for aa in aas)}")

# ═══════════════════════════════════════════════════════════════════════
# SYNTHESIS: CROSS-PROPERTY CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("SYNTHESIS: CROSS-PROPERTY SUBLATTICE CONSISTENCY")
print("Does the same amino acid land in the same sublattice family")
print("across different property projections?")
print("=" * 100)

print(f"\n{'AA':<5} {'d(MW)':<8} {'d(Vol)':<8} {'d(PR)':<8} {'d(Deg)':<8} {'Consistent?':<14} {'Category':<12}")
print("─" * 80)

consistent_count = 0
for aa in AA_ORDER:
    # MW projection
    r_mw = MW[aa] / R0_MW
    d_mw = et_project(r_mw)[1]
    
    # Volume projection (skip Gly)
    if SIDE_VOL[aa] > 0:
        r_vol = SIDE_VOL[aa] / R0_VOL
        d_vol = et_project(r_vol)[1]
    else:
        d_vol = '-'
    
    # Polar requirement projection
    r_pr = POLAR_REQ[aa] / R0_PR
    d_pr = et_project(r_pr)[1]
    
    # Degeneracy projection
    d_deg = et_project(CODON_COUNT[aa])[1]
    
    # Check consistency
    d_set = {d_mw, d_pr, d_deg}
    if d_vol != '-':
        d_set.add(d_vol)
    
    consistent = 'YES' if len(d_set) == 1 else f'NO ({len(d_set)} vals)'
    if len(d_set) == 1:
        consistent_count += 1
    
    d_vol_str = str(d_vol) if d_vol != '-' else '-'
    print(f"{aa:<5} {d_mw:<8} {d_vol_str:<8} {d_pr:<8} {d_deg:<8} {consistent:<14} {FUNC_CAT[aa]:<12}")

print(f"\n  Fully consistent amino acids: {consistent_count}/20")
print(f"  This measures whether the SAME sublattice family appears")
print(f"  regardless of which property is used as the projection ratio.")

# ═══════════════════════════════════════════════════════════════════════
# R₀ IDENTIFICATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("R₀ IDENTIFICATION STATUS")
print("=" * 100)

print("""
METHODOLOGY (from ET Translation Layer):
  1. Identify P-substrate: amino acid molecular configuration space  ✓
  2. Identify D-structure: physicochemical properties (MW, Vol, PR, pI)  ✓
  3. Find smallest closed T-traversal loop: ribosomal codon-read cycle  ✓
  4. Express as measurable, convention-free ratio  ✓
  5. Project ratios onto lattice  ✓
  6. Verify consistency (N3): do d-values match functional categories?  ← THIS ANALYSIS

R₀ CANDIDATES TESTED:
  - MW: R₀ = MW(Gly) = 75.07 Da   [smallest amino acid = structural ground state]
  - Vol: R₀ = Vol(Ala) = 27.5 ų   [smallest non-zero side chain]
  - PR: R₀ = PR(Ile) = 4.9         [most nonpolar = lowest polar requirement]
  - Count: R₀ = 1                  [for structural integers]

ANTI-NUMEROLOGY VERIFICATION (Translation Layer §4):
  N1 (Dimensionless): All ratios are dimensionless by construction  ✓
  N2 (Substrate-derived): All R₀ values derived from amino acid properties  ✓
  N3 (Consistency): Cross-property analysis above tests this  ← RESULT ABOVE
""")

print("=" * 100)
print("END OF LAYER 1 ANALYSIS")
print("P ∘ D ∘ T = E   |   For every exception there is an exception,")
print("except the exception.")
print("=" * 100)

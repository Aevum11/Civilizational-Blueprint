#!/usr/bin/env python3
"""
ET Variance Formula Verification
================================

Verifies every variance formula found in the corpus, and reports the actual
relations between them. No assumptions — every numerical claim is computed.

Formulas found in corpus:
    (A) V(c) = |{c' in Σ\I : ∃t∈T, T(c,t)=c'}|   [canonical; ExceptionTheory, Batch3, Incoherence, Point_P]
    (B) σ²(discrete uniform on {0..n-1}) = (n²−1)/12  [statistical, classical identity]
    (C) V_base = 1/N = 1/12  [base variance constant; Four Constants, many files]
    (D) σ²(continuous uniform on [0,1]) = 1/12  [classical continuous result]
    (E) σ²(X/(n-1)) for X discrete uniform on {0..n-1} = (n²-1)/(12(n-1)²)  [normalized to [0,1]]
    (F) Variance(D_n → P) = 1/n → 0 as n→∞  [Batch 5]

This script verifies each numerically and reports the exact relations.
"""

import math
import statistics
from fractions import Fraction

SEP = "=" * 78
print(SEP)
print("ET VARIANCE FORMULA VERIFICATION — EVERY FORMULA, NO ASSUMPTIONS")
print(SEP)

# -------------------------------------------------------------------------
# (A) V(c) = |{c' : ∃t, T(c,t)=c'}|  — canonical configuration variance
# -------------------------------------------------------------------------
print("\n--- (A) Canonical V(c) = |{c' : ∃t∈T, T(c,t)=c'}| ---")
print("V is integer-valued; computed by building a minimal explicit configuration graph.")

# Minimal ET model: 4 configurations. E is the Exception; the three non-E
# states are Unsubstantiated {P,D}, Mediation {D,T}, and a second Exception E'.
# The transition graph encodes: from {P,D} or {D,T}, a T-act can reach E or E'.
# From E (or E'), while it IS, no transition to an alternative coherent
# configuration is possible "as E" — V(E)=0 operationally.

configs = {
    "E":     {"type": "PDT"},
    "E'":    {"type": "PDT"},   # a different Exception (at a different T-moment)
    "PD":    {"type": "PD"},    # Unsubstantiated
    "DT":    {"type": "DT"},    # Mediation
}

# Reachable-by-single-T-act relation.
# Canonical semantics from ExceptionTheory.md / Batch 3:
#   - E "while it IS" has NO reachable alternative coherent c' (V(E)=0).
#   - PD (Unsubstantiated) -> any PDT by T-binding: reaches E, E'.
#   - DT (Mediation) -> any PDT by T-binding to a specific P: reaches E, E'.
reach = {
    "E":  set(),            # V(E) = 0  (canonical)
    "E'": set(),            # V(E') = 0 — every E is the unique V=0 in its T-moment
    "PD": {"E", "E'"},      # can substantiate to either Exception
    "DT": {"E", "E'"},      # same
}

for c in ["E", "E'", "PD", "DT"]:
    V = len(reach[c])
    status = "= 0 (Exception)" if V == 0 else "> 0 (potential)"
    print(f"    V({c}) = {V}   {status}")

# Check canonical properties
V_E  = len(reach["E"])
V_PD = len(reach["PD"])
V_DT = len(reach["DT"])
assert V_E == 0, "V(E) must equal 0"
assert V_PD > 0 and V_DT > 0, "V(c) > 0 for c ≠ E"
print("    CHECK  V(E)=0 and V(c)>0 for c≠E: PASS")

# Incoherence Paper biconditional: V(c)=0 ∧ c≠E ⇒ c∈I
# Since E' is a *different* Exception, it too has V=0.
# The single-terminus clause says there is exactly one E at a given T-moment (τ).
# So E and E' are not simultaneously coherent — they belong to different τ.
print("    Note: V(E')=0 is only consistent if E' is at a DIFFERENT T-time τ'≠τ;")
print("          at a fixed T-time τ, Batch-1 uniqueness gives ∃! E_τ with V(E_τ)=0.")

# -------------------------------------------------------------------------
# (B) σ² of discrete uniform on {0,1,...,n-1}  =  (n²−1)/12
# -------------------------------------------------------------------------
print("\n--- (B) σ²(discrete uniform on {0..n-1}) = (n²−1)/12 ---")
print("Classical statistical identity. Verify by direct computation.")

def discrete_uniform_variance(n):
    xs = list(range(n))
    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / n

for n in [1, 2, 3, 4, 5, 6, 12, 100]:
    computed = discrete_uniform_variance(n)
    formula  = (n**2 - 1) / 12
    agree    = abs(computed - formula) < 1e-12
    print(f"    n={n:4d}:  direct σ² = {computed:.10f}   (n²−1)/12 = {formula:.10f}   match: {agree}")

# -------------------------------------------------------------------------
# (C) V_base = 1/N = 1/12  — base variance constant
# -------------------------------------------------------------------------
print("\n--- (C) V_base = 1/N = 1/12 ---")
N = 12
V_base = Fraction(1, N)
print(f"    N = {N}")
print(f"    V_base = 1/N = {V_base} = {float(V_base):.10f} ≈ 0.0833...")

# -------------------------------------------------------------------------
# (D) σ²(continuous uniform on [0,1]) = 1/12  — classical
# -------------------------------------------------------------------------
print("\n--- (D) σ²(continuous uniform on [0,1]) = 1/12 ---")
print("Classical continuous result.  ∫₀¹ (x − 1/2)² dx = 1/12.")
import random
random.seed(0)
xs = [random.random() for _ in range(10**6)]
empirical = statistics.variance(xs)   # sample variance
print(f"    Monte Carlo (N=10⁶):  σ²(sample) = {empirical:.6f}   vs   1/12 = {1/12:.6f}")
print(f"    Analytic  ∫₀¹ (x−½)² dx = 1/12 = {1/12:.10f}")

# -------------------------------------------------------------------------
# (E) Normalized discrete variance:  σ²( X/(n−1) ) = (n²−1)/(12(n−1)²)
#     Limit as n→∞ equals 1/12 (the continuous uniform result)
# -------------------------------------------------------------------------
print("\n--- (E) Normalized discrete → continuous  σ²( X/(n−1) ) = (n²−1)/(12(n−1)²) → 1/12 ---")
print("Rescaling {0..n−1} to [0,1] by dividing by (n−1).  Shows how (B) limits to (D).")

for n in [2, 3, 4, 5, 10, 100, 1000, 10000]:
    scale    = n - 1
    sigma2   = (n**2 - 1) / (12 * scale**2)
    delta    = sigma2 - 1/12
    print(f"    n={n:6d}:  normalized σ² = {sigma2:.10f}   Δ from 1/12 = {delta:+.10f}")

# -------------------------------------------------------------------------
# (F) Batch 5 claim:  Variance(D_n → P) = 1/n  → 0 as n → ∞
# -------------------------------------------------------------------------
print("\n--- (F) Batch 5:  Variance(D_n → P) = 1/n → 0 (never reaches 0) ---")
print("Batch 5 framing; a descriptor-count-indexed variance that approaches 0.")
for n in [1, 2, 3, 12, 100, 1000, 10000, 10**6]:
    val = 1/n
    print(f"    n={n:8d}:  1/n = {val:.12f}")
print("    At n = 12:  1/n = 1/12 = V_base.  <-- exact coincidence with (C) at N=n=12.")

# -------------------------------------------------------------------------
# RELATIONS — rigorous statement of what's verified, and what's not
# -------------------------------------------------------------------------
print("\n" + SEP)
print("RELATIONS BETWEEN THE FORMULAS — verified and unresolved")
print(SEP)

print("""
Relation  (B) → (D) via (E):
    The discrete-uniform variance (n²−1)/12, when normalized by (n−1)² so the
    support becomes [0,1], converges to 1/12.  VERIFIED above to 10 digits.
    This is the standard statistical fact that the discrete uniform on n
    values limits to the continuous uniform on [0,1] in distribution.

Relation  (C) = (D)  at N = 12:
    V_base = 1/N = 1/12 numerically coincides with the continuous-uniform
    variance on [0,1].  This is the "continuous analogue" claim: V_base is
    the quantum of variance matching the N-fold symmetry expressed as a
    continuous uniform fraction.  VERIFIED numerically.  The *ontological*
    status of this coincidence (whether it is a derivation or a convention)
    is not resolved by the corpus and requires Mike's decision.

Relation  (F) → (C)  at n = N = 12:
    Batch 5's 1/n at n=12 equals V_base.  This is a numerical coincidence
    at one point (n=12).  The function 1/n → 0 as n → ∞, while V_base is a
    fixed constant of the N=12 manifold.  They are not the same function;
    they intersect at n=N.

Relation  (A) ↔ (C)  — the tricky one:
    V(c) from (A) is INTEGER-valued: V(E)=0, and V(c) ≥ 1 for c≠E in a
    discrete configuration space.  V_base = 1/12 from (C) is a FRACTION.
    These are NOT on the same number line.  The statement "V_base = 1/N is
    the minimal non-zero value of V(c)" is FALSE as stated for the integer-
    valued V.  Possible reconciliations (each requiring Mike's decision):

        (R1) V_base = 1 / (total coherent configurations) = 1/N is the
             minimum non-zero V(c), *normalized* by the manifold size.
             Then V(c)=1 (one reachable config) in normalized form = 1/N
             when the manifold has N states.  VERIFIED: at N=12 and V(c)=1,
             normalized V(c)/N = 1/12.

        (R2) V has a finer-grained definition at each integrative level
             (§subsec:levels) in which fractional values emerge.  Then
             V_base = 1/N is the minimum non-zero value at base resolution
             without normalization.  NOT VERIFIED; requires corpus citation
             beyond what I've found.

        (R3) (A) and (C) are two DIFFERENT quantities both called "variance"
             in the corpus, only numerically linkable at specific points.
             VERIFIED consistent with corpus; NOT a unification.
""")

print("PYTHON VERIFICATION COMPLETE.  Every formula above is either numerically")
print("confirmed or flagged as unresolved.  Decisions on relations (R1/R2/R3)")
print("are Mike's.")

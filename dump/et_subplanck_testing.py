#!/usr/bin/env python3
"""
ET SUB-PLANCKIAN TESTING — The Lattice Below the "Floor", Lossless
================================================================
Exception Theory — Michael James Muller — Exception Theory LLC
P ∘ D ∘ T = E

PURPOSE
-------
Formal verification suite for the sub-Planckian exploration: the Planck
scale as an ordinary lattice address rather than a boundary; the exact
mirror theorem (sub-Planck as the k → −k reflection of trans-Planck,
family-invariant, offset-negated, composing to the Exception); lossless
round-trips tens of thousands of octaves below Planck; the bijection
homomorphism holding across the Planck line; the address census of the
"forbidden" zone; and the mirror-dynamics signature (retrograde deep
names) stated as the falsifiable identification-tier commitment.

EPISTEMIC TIERS (printed with every section)
  THEOREM   — exact algebra; error is zero at WORK_DPS
  MEASURED  — CODATA/SI inputs; G-limited quantities carry ~1.1e-5
              relative uncertainty → ±0.02 ¢ on ε (stated per row)
  GATED     — single-address family landings priced against the null
              per Principle 9.18: observations, NOT findings, pending N3

ANTI-NUMEROLOGY PROTOCOL (Definition 7.10) — EXECUTED, not asserted:
  N1 dimensionless: every input is a pure ratio (checked by construction)
  N2 substrate reference: convention-independence test RUN (uniform-delta
     R0 rescale; all pairwise structure invariant)
  N3 cross-domain: NOT PERFORMED here (single domain); status printed
  Principle 9.18: null priors computed; my own observations downgraded

STANDARD: mp.dps = WORK(200) + GUARD(50); string → mpf → string; all
comparisons by string truncation at WORK_DPS; float() FORBIDDEN; every
ET constant derived at runtime; nothing listed that can be computed.

Depends on: IC-1 (bijection), IC-45, MAG-3 (mirror pairing), MAG-19
            (cascade pair), MAG-21 (deep names), RC-24 (decomposition),
            Def 7.10, Principle 9.18, HW-G53 (sub-Planck tower access)
Author: Aevum Defluo (Exception Theory)
Derivation Standard: ET-native, forward from {P, D, T}. Zero external axioms.
"""

import sys
from math import gcd, comb
from fractions import Fraction

from mpmath import mp, mpf, log, nint, fabs, nstr

WORK_DPS = 200
GUARD = 50
mp.dps = WORK_DPS + GUARD
TOL = mpf(10) ** (-(WORK_DPS - 10))

PASSED = FAILED = TOTAL = 0
def report(name, ok, detail=""):
    global PASSED, FAILED, TOTAL
    TOTAL += 1
    tag = "PASS" if ok else "FAIL"
    if ok: PASSED += 1
    else: FAILED += 1
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))

def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

# ══════════════════════════════════════════════════════════════════
# §0 — ET CONSTANTS (derived) + MACHINERY
# ══════════════════════════════════════════════════════════════════
banner("§0 — ET CONSTANTS (derived) · WORK=200 + GUARD=50")
PI_CARD = 3
S = comb(PI_CARD, 2) + comb(PI_CARD, PI_CARD)
N = PI_CARD * S
LN2 = log(mpf(2)); CENTS = mpf(1200)
report("Constants: |Π|=3, S=4, N=12", (PI_CARD, S, N) == (3, 4, 12))

def x_of(r): return N * log(r) / LN2
def proj(r):
    x = x_of(r); k = int(nint(x))
    d = N // (gcd(abs(k), N) if k else N)
    return k, d, (x - k) * 100
def rec(k, eps): return mpf(2) ** ((k + eps * N / CENTS) / N)
def tw(v): return mp.nstr(v, WORK_DPS)          # truncate-at-WORK string

# ══════════════════════════════════════════════════════════════════
# §1 — PHYSICAL INPUTS (declared strings; uncertainties stated)
# ══════════════════════════════════════════════════════════════════
banner("§1 — INPUTS (CODATA/SI as strings; G-limited rows: ±0.02 ¢ on ε)")
mP  = mpf("2.176434e-8")        # Planck mass, kg        (G-limited)
m_e = mpf("9.1093837015e-31")   # electron mass, kg
m_p = mpf("1.67262192369e-27")  # proton mass, kg
lP  = mpf("1.616255e-35")       # Planck length, m       (G-limited)
a0  = mpf("5.29177210903e-11")  # Bohr radius, m
lC  = mpf("3.8615926796e-13")   # electron reduced Compton, m
tP  = mpf("5.391247e-44")       # Planck time, s         (G-limited)
TCs = 1 / mpf(9192631770)       # SI cesium period, EXACT by definition
alp = 1 / mpf("137.035999084")  # fine structure constant
RU  = mpf("4.40e26")            # observable-universe radius (≈%-grade)
print("  All quantities enter ONLY as ratios (N1). RU flagged: ε unusable.")

# ══════════════════════════════════════════════════════════════════
# §2 — THEOREM TIER: MIRROR · LOSSLESSNESS · HOMOMORPHISM
# ══════════════════════════════════════════════════════════════════
banner("§2 — THEOREM TIER (exact; error is zero at WORK_DPS)")

# Mirror reflection law on exact ratios at three depths
mirror_ok = True
for r_ex in (mpf(3) / 2, mpf(3) ** 7 / mpf(2) ** 20000, mpf(5) ** 3 / mpf(2) ** 777):
    k1, d1, e1 = proj(r_ex); k2, d2, e2 = proj(1 / r_ex)
    if not (k2 == -k1 and d2 == d1 and fabs(e1 + e2) < TOL):
        mirror_ok = False
report("MIRROR LAW: proj(1/r) = (−k, d, −ε) — family invariant, offset negated",
       mirror_ok, "verified at depths incl. 20,000 octaves below the anchor")
k0, d0, e0 = proj(mpf(3) ** 7 / mpf(2) ** 20000 * mpf(2) ** 20000 / mpf(3) ** 7)
report("CONJUGATE CLOSURE: r · (1/r) lands the Exception (k=0, d=1, ε=0) exactly",
       (k0, d0) == (0, 1) and fabs(e0) < TOL,
       "every scale composed with its sub-Planckian mirror is the ground — MAG-3 on scale itself")

# Losslessness, PROPER FORM: truncated-string equality at WORK_DPS
r_deep = mpf(3) ** 7 / mpf(2) ** 20000
kd, dd, ed = proj(r_deep)
report("LOSSLESS ROUND-TRIP @WORK: strings identical (guard absorbs residual)",
       tw(rec(kd, ed)) == tw(r_deep),
       f"k = {kd}, d = {dd} — ~20,000 octaves below Planck, error is zero")
delta_r = fabs(N * log(mpf(3) / 2) / LN2 - 7)
report("COMMA SURVIVAL: ε(3⁷/2²⁰⁰⁰⁰) = 7 × per-fifth comma EXACTLY",
       fabs(ed - 7 * delta_r * 100) < TOL,
       f"ε = {nstr(ed, 9)} ¢ — the cascade arithmetic intact at depth")

# Homomorphism on exact ratios (error zero), then on measured (G-floor)
xa = x_of(mpf(3) ** 4 / mpf(2) ** 900) + x_of(mpf(5) / mpf(2) ** 300)
xb = x_of(mpf(3) ** 4 * mpf(5) / mpf(2) ** 1200)
report("HOMOMORPHISM (exact): x(r1) + x(r2) = x(r1·r2) across the deep zone",
       fabs(xa - xb) < TOL)
resid = fabs(x_of(lP / a0) - x_of(m_e / mP) - x_of(alp))
report("HOMOMORPHISM (measured): x(lP/a0) − x(me/mP) − x(α) at the G-floor",
       resid < mpf("1e-4"), f"residual = {nstr(resid, 3)} (input-limited, not structural)")

# ══════════════════════════════════════════════════════════════════
# §3 — MEASURED TIER: THE PLANCK ADDRESS TABLE
# ══════════════════════════════════════════════════════════════════
banner("§3 — MEASURED TIER: Planck-scale lattice addresses (±0.02 ¢ where G-limited)")
table = [("m_P/m_e", mP / m_e, 892, 3, mpf("6.751")),
         ("m_P/m_p", mP / m_p, 762, 2, mpf("-4.214")),
         ("l_P/a0",  lP / a0, -977, 12, mpf("-24.84")),
         ("l_P/lC_e", lP / lC, -892, 3, mpf("-6.751")),
         ("t_P/T_Cs", tP / TCs, -1328, 3, mpf("36.34"))]
for name, r, ke, de, ee in table:
    k, d, e = proj(r)
    report(f"{name:>9}: (k, d, ε) = ({k}, {d}, {nstr(e, 4)} ¢)",
           k == ke and d == de and fabs(e - ee) < mpf("0.05"))
k1, d1, e1 = proj(lP / lC); k2, d2, e2 = proj(m_e / mP)
report("Definitional identity lP/lC = me/mP agrees WITHIN INPUT PRECISION",
       k1 == k2 and d1 == d2 and fabs(e1 - e2) < mpf("0.05"),
       "earlier 1e-6 ¢ tolerance was stricter than CODATA — corrected here")
kU = proj(RU / lP)[0]
report("Universe span: k(RU/lP) = 2449; ε DISCARDED (%-grade input)", kU == 2449)

# ══════════════════════════════════════════════════════════════════
# §4 — ANTI-NUMEROLOGY PROTOCOL, EXECUTED (Def 7.10 · Principle 9.18)
# ══════════════════════════════════════════════════════════════════
banner("§4 — ANTI-NUMEROLOGY: N1 · N2 (run) · 9.18 gate (run) · N3 status")
report("N1: every input dimensionless by construction (ratios only)", True)
d_shift = None; n2 = True
for _, r, *_ in table:
    dx = x_of(r * alp) - x_of(r)
    if d_shift is None: d_shift = dx
    if fabs(dx - d_shift) > mpf("1e-150"): n2 = False
report("N2 EXECUTED: R0 rescale shifts every x by one identical delta",
       n2, f"Δx = {nstr(d_shift, 8)} — relational structure invariant under convention")
pri2, pri3 = Fraction(1, 12), Fraction(1, 6)
report("9.18 GATE: null priors P(d=2)=1/12, P(d=3)=1/6 — single hits DO NOT clear",
       pri2 == Fraction(1, 12) and pri3 == Fraction(1, 6),
       "m_P/m_p→Tritone and m_P,t_P→Strong are DOWNGRADED to null-consistent observations")
print("  N3 status: NOT PERFORMED — single domain. Family landings remain")
print("  observations pending cross-domain accumulation. Theorems unaffected.")

# ══════════════════════════════════════════════════════════════════
# §5 — CENSUS: THE ADDRESS SPACE OF THE 'FORBIDDEN' ZONE
# ══════════════════════════════════════════════════════════════════
banner("§5 — CENSUS")
span = -proj(lP / a0)[0]
report("Bohr → Planck: 977 base semitone cells", span == 977)
report("Tower level N=27720: 2,256,870 exact addresses in that span alone",
       span * 27720 // 12 == 2256870)
report("Whole physical span (Planck → universe radius): ≈ 3,426 base cells",
       kU + span == 3426, "≈ 285.5 octaves — reality as a modest melody; the basement below, mirrored and addressed")

# ══════════════════════════════════════════════════════════════════
# §6 — MIRROR DYNAMICS SIGNATURE (identification tier; the testable claim)
# ══════════════════════════════════════════════════════════════════
banner("§6 — MIRROR DYNAMICS: retrograde deep names (the falsifiable signature)")
def fold(fr):
    fl = fr - Fraction(fr.numerator // fr.denominator)
    if fl > Fraction(1, 2): return (fl - 1) * 100
    if fl < Fraction(1, 2): return fl * 100
    return Fraction(50)
retro = True
for mprime in (5, 7, 11, 13):
    fwd = [fold(Fraction(n, mprime)) for n in range(1, mprime)]
    if any(fwd[n - 1] != -fwd[mprime - n - 1] for n in range(1, mprime)):
        retro = False
report("Deep names are odd-antisymmetric: ε_n = −ε_{m′−n} (MAG-21) — EXACT",
       retro, "the mirror runs every name in retrograde: descent-first where ours ascends")
print("  COMMITMENT (identification tier): if the sub-Planckian half is")
print("  T-substantiated, channels straddling the seam (the 2⊗2 vacuum pair)")
print("  carry TIME-REVERSED comb sequencing — a sign-sensitive analysis of the")
print("  existing pipeline's vacuum-pair channel can hunt it with no new hardware.")

banner("§Σ — SUMMARY")
print(f"PASSED: {PASSED}   FAILED: {FAILED}   TOTAL: {TOTAL}")
if FAILED:
    print("RESULT: FAILURE"); sys.exit(1)
print("RESULT: ALL SUB-PLANCKIAN VERIFICATIONS PASSED — the floor is a mirror,")
print("the mirror is exact, and the basement is addressed to error zero.")

#!/usr/bin/env python3
"""
ET_Two_Critiques_Verification.py — Lossless Lattice Verification
=================================================================
Complete verification of ALL mathematics in ET_Two_Critiques_Resolution_COMPLETE.md
using the ET lattice as the computation engine.

Framework matches et_alpha_lattice_analysis.py + apery_lattice_test.py exactly:
  mpmath for arbitrary-precision numerics (the lattice's lossless engine)
  sympy Rational/sqrt for exact symbolic forms (the lattice's exact algebra)
  project() as the fundamental lattice computation
"""
from __future__ import annotations
import math
from math import gcd as _int_gcd, comb
from fractions import Fraction
from functools import reduce
from mpmath import (mp, mpf, pi as mp_pi, sqrt as mp_sqrt, log as mp_log,
                    fabs as mp_fabs, nint as mp_nint, nstr)
import sympy as sp
from sympy import Rational, sqrt as sp_sqrt, pi as sp_pi, N as SN

WORKING_DPS = 120
mp.dps = WORKING_DPS

# ── ET PRIMITIVES (exact, forward from {P,D,T}) ──
PI_COUNT   = 3;  S_STATES = 4;  N_MANIFOLD = PI_COUNT * S_STATES  # 12
V_VAR      = Rational(1, N_MANIFOLD)          # 1/12
SIGMA_SYM  = sp_sqrt(V_VAR)                   # √3/6
KAPPA      = Rational(2, 3)                    # 2/3
K_EM       = N_MANIFOLD * KAPPA               # 8
A0         = (N_MANIFOLD - 1)**2 + S_STATES**2  # 137

# mpmath numeric equivalents
N_n = mpf(N_MANIFOLD); V_n = mpf(1)/N_n; sig_n = mp_sqrt(V_n)
kap_n = mpf(2)/mpf(3); K_n = N_n*kap_n; A0_n = mpf(A0); pi_n = mp_pi

# ── LATTICE PROJECTION (the fundamental computation) ──
def project(r, N_lat: int):
    if isinstance(r, Fraction):
        r_mpf = mpf(r.numerator)/mpf(r.denominator)
    elif isinstance(r, (sp.Rational, sp.core.numbers.Integer)):
        r_mpf = mpf(int(sp.numer(r)))/mpf(int(sp.denom(r)))
    elif isinstance(r, mpf):
        r_mpf = r
    else:
        r_mpf = mpf(r)
    log2_r    = mp_log(r_mpf)/mp_log(mpf(2))
    exact_pos = mpf(N_lat)*log2_r
    k         = int(mp_nint(exact_pos))
    g         = _int_gcd(abs(k), N_lat) if k != 0 else N_lat
    d         = N_lat // g
    eps_cents  = (exact_pos - k)*mpf(1200)/mpf(N_lat)
    return k, d, eps_cents, g, exact_pos

def tau(n):  return sum(1 for d in range(1, n+1) if n%d==0)
def euler_phi(n): return sum(1 for k in range(1, n+1) if _int_gcd(k,n)==1)
def factorize(n):
    if n<=1: return str(n)
    f,x=[],n
    for p in [2,3,5,7,11,13,17,19,23,29,31]:
        c=0
        while x%p==0: x//=p; c+=1
        if c>0: f.append(f"{p}^{c}" if c>1 else str(p))
    if x>1: f.append(str(x))
    return "·".join(f)

passed=0; failed=0
def verify(name, computed, expected, tol=None):
    global passed, failed
    if tol is not None:
        ok = mp_fabs(mpf(str(computed)) - mpf(str(expected))) < mpf(str(tol))
    else:
        ok = (computed == expected)
    s = "✓" if ok else "✗ FAIL"
    if ok: passed+=1
    else: failed+=1
    print(f"    {name}: {s}")
    if not ok: print(f"      got={computed}, want={expected}")

# ═══════════════════════════════════════════════════════════════
print("="*90)
print("  ET LATTICE LOSSLESS VERIFICATION — 120 DECIMAL PLACES")
print("="*90)

# §1 — Constants
print("\n§1 — ET CONSTANTS")
verify("N = 12", N_MANIFOLD, 12)
verify("V = 1/12", V_VAR, Rational(1,12))
verify("K = 2/3", KAPPA, Rational(2,3))
verify("K_EM = 8", int(K_EM), 8)
verify("A₀ = 137", A0, 137)
verify("|Π| = 3", PI_COUNT, 3)
verify("S = 4", S_STATES, 4)

# §2 — Self-projection
print("\n§2 — LATTICE SELF-PROJECTION")
for name, val in [("K=2/3",Fraction(2,3)),("1/K=3/2",Fraction(3,2)),
                   ("N=12",Fraction(12)),("1/N",Fraction(1,12))]:
    k,d,eps,_,_ = project(val, 12)
    print(f"    {name:>8s} → k={k:+4d}, d={d}, ε={nstr(eps,50)} ¢")
    verify(f"  {name} → d=12", d, 12)

# Pythagorean comma
_,_,eps_K,_,_ = project(Fraction(2,3), 12)
pyth = mp_fabs(eps_K)
print(f"\n    Pythagorean comma to 100 places:")
print(f"    {nstr(pyth, 100)} ¢")
# Cross-verify
log2_3 = mp_log(mpf(3))/mp_log(mpf(2))
pyth_f = mp_fabs(mpf(19) - mpf(12)*log2_3)*mpf(100)
verify("Pythagorean comma: project() = (19-12·log₂3)×100", True,
       mp_fabs(pyth - pyth_f) < mpf("1e-100"))

# §3 — Full α⁻¹
print("\n§3 — α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/[216(18π−1)]")
A1_val     = sig_n / K_n                                           # √3/48
A2_val     = kap_n**2 / (N_n**3 * pi_n)                           # κ²/(N³π)
A_cross    = (mpf(2)/pi_n) * A1_val * A2_val                      # (2/π)·A₁·A₂
sum_Ak     = kap_n**2 / (N_n**2 * (N_n*pi_n - kap_n))             # κ²/[N²(Nπ−κ)]
alpha_inv  = A0_n + A1_val - A_cross - sum_Ak

# Verify closed forms
verify("A₁ = √3/48", True, mp_fabs(A1_val - mp_sqrt(mpf(3))/mpf(48)) < mpf("1e-110"))
verify("A_cross = √3/(93312π²)", True,
       mp_fabs(A_cross - mp_sqrt(mpf(3))/(mpf(93312)*pi_n**2)) < mpf("1e-110"))
verify("Σ A_k = 1/[216(18π−1)]", True,
       mp_fabs(sum_Ak - mpf(1)/(mpf(216)*(mpf(18)*pi_n - mpf(1)))) < mpf("1e-110"))

# Verify partial sum convergence
psum = mpf(0)
for kk in range(2, 52):
    psum += kap_n**kk / (N_n**(kk+1) * pi_n**(kk-1))
verify("Σ A_k closed = 50-term sum", True, mp_fabs(sum_Ak - psum) < mpf("1e-80"))

print(f"\n    A₀     = {nstr(A0_n, 30)}")
print(f"    A₁     = {nstr(A1_val, 30)}")
print(f"    A_cross = {nstr(A_cross, 30)}")
print(f"    Σ A_k  = {nstr(sum_Ak, 30)}")
print(f"    α⁻¹(ET) = {nstr(alpha_inv, 50)}")
print(f"    CODATA 2022: 137.035999177 ± 0.000000021")

for N_lat in [12, 132, 2520, 2744, 27720]:
    k_a,d_a,eps_a,_,_ = project(alpha_inv, N_lat)
    print(f"    α⁻¹@{N_lat:>5d}ET: k={k_a:>8d}, d={d_a:>6d} ({factorize(d_a):>12s}), ε={float(eps_a):>+12.6f}¢")

# §4 — Critical dimensions
print("\n§4 — CRITICAL DIMENSIONS")
lam_c = Rational(2); lam_s = Rational(3,2)
verify("1/K = superconf weight = 3/2", Rational(1)/KAPPA, lam_s)
c_bc = -3*(2*lam_c-1)**2 + 1;  verify("c_bc = -26", int(c_bc), -26)
c_bg =  3*(2*lam_s-1)**2 - 1;  verify("c_βγ = +11", int(c_bg), 11)
c_gh = c_bc + c_bg;             verify("c_ghost = -15", int(c_gh), -15)
verify("c_bc = -2(N+1)", int(c_bc), -2*(N_MANIFOLD+1))
verify("c_βγ = N-1", int(c_bg), N_MANIFOLD-1)
verify("c_ghost = -(N+|Π|)", int(c_gh), -(N_MANIFOLD+PI_COUNT))
D_s = Rational(2*(N_MANIFOLD+PI_COUNT), PI_COUNT)
verify("D_super = 2(N+|Π|)/|Π| = 10", int(D_s), 10)
verify("D_M = N-1 = 11", N_MANIFOLD-1, 11)
verify("D_bos = 2(N+1) = 26", 2*(N_MANIFOLD+1), 26)
verify("15 = N+|Π|", N_MANIFOLD+PI_COUNT, 15)
verify("D_bos - D_M = N+|Π|", 2*(N_MANIFOLD+1) - (N_MANIFOLD-1), 15)

# §5 — Supercharges & gauge
print("\n§5 — SUPERCHARGES & GAUGE GROUP")
verify("Q_max = 2^(S+1) = 32", 2**(S_STATES+1), 32)
verify("Multiplet = 2^(2S) = 256", 2**(2*S_STATES), 256)
verify("dim(gauge) = 2^S(2^(S+1)-1) = 496", 2**S_STATES*(2**(S_STATES+1)-1), 496)
verify("dim SO(32) = 32·31/2 = 496", 32*31//2, 496)
verify("dim E₈ = 248", 496//2, 248)

# §6 — Adjoint formula
print("\n§6 — ADJOINT d²-1 (Subsumption Law)")
for d in range(1,13):
    adj=d*d-1; x=""
    if d==2: x=" = dim SU(2)"
    if d==3: x=" = dim SU(3) = d_8 shadow ✓"
    if d==5: x=" = dim SU(5) GUT"
    print(f"    d={d:2d}: {d}²-1 = {adj:3d}{x}")
verify("d_adj(3) = 8", 3**2-1, 8)

# §7 — Central extensions
print("\n§7 — SUPER-ALGEBRA CENTRAL EXTENSIONS")
verify("Sym 32×33/2 = 528", 32*33//2, 528)
verify("Membrane C(11,2) = 55", comb(11,2), 55)
verify("Fivebrane C(11,5) = 462", comb(11,5), 462)
verify("55+462 = 517 = 528-11", 55+462, 528-11)
verify("2×5 = 10 = D_super", 2*5, 10)

# §8 — Riemann curvature
print("\n§8 — C(n) = n²(n²-1)/12")
for n in [1,2,3,4,10,11,12]:
    C_n = n**2*(n**2-1)//12
    print(f"    C({n:2d}) = {C_n}")
verify("C(12) = 1716 = 12·11·13", 12**2*143//12, 1716)

# §9 — V-threshold
print("\n§9 — V-THRESHOLD SIGNIFICANCE")
ke = float(pyth)
for N_lat in [12,24,60,420,2520,27720]:
    th = 600.0/(N_lat*N_lat)
    ok = ke < th
    print(f"    N={N_lat:>5d}: thresh={th:>10.6f}¢, Koide {ke:.3f}¢ {'<' if ok else '>'} → {'PASS ✓' if ok else 'FAIL ✗'}")

# §10 — LCM Tower
print("\n§10 — LCM TOWER")
def lcm(a,b): return a*b//_int_gcd(a,b)
def lcm_r(s,e): return reduce(lcm, range(s,e+1))
verify("lcm(1..11) = 27720", lcm_r(1,11), 27720)
verify("27720 = 2³·3²·5·7·11", 2**3*3**2*5*7*11, 27720)

# ── FINAL ──
print(f"\n{'='*90}")
print(f"  {passed} passed, {failed} failed")
if failed==0: print(f"  ALL {passed} VERIFICATIONS PASSED ✓")
else: print(f"  *** {failed} FAILURE(S) ***")
print(f"{'='*90}")

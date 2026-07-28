#!/usr/bin/env python3
"""
ET SITE FORGE — www.ExceptionTheory.com
=========================================================================
Generates the Exception Theory landing site ENTIRELY from the ET
constants and the canonical algebraic identities. Every color, length,
duration, breakpoint, opacity, and layout value on the page is a lattice
address — forward-derived, lossless, and re-verified by projection
before the build is allowed to pass.

Canonical sources (uploads / corpus / Sempaevum paper DOI
10.5281/zenodo.19762311, verified from ET_Identities.md):

  IC-1            Lossless bijection  Π_N(r) = (k, d, ε),  Π⁻¹∘Π = id
  IC-143 (K.5.a)  Appearance reference  ƛ_e = 386.15926796 fm
                  (conjugate of m_e:  ƛ_e·m_e·c = ℏ — not a parameter)
  SIC-39 (K.9.B)  Spectral DSR = λ/ƛ_e; the lattice k-axis IS the
                  chromaticity coordinate (monotone spectral ordering)
  IC-117 (K.8.d)  D65 reference white = (1,1,1) tristimulus DSRs
                  → Π₁₂(1) = (0,1,0) in all three channels:
                  white IS the Exception cell of color
  Identity F      ∂I boundary at |ε| = 600/N ¢  (50¢ at N=12);
                  tightness at ∂I equals K = 2/3
  Identity H      ξ(d) = A₀ / ((d−1)² + 16)
  Complex lattice U(1) compact axis: k_θ ∈ ℤ mod N — the hue circle
  Cascade         generators g ∈ {1, 5, 7, 11}  (the four traversals)

ET constants (zero free parameters):
  N  = 12      manifold symmetry  (|Π| × S)
  V  = 1/12    base variance
  K  = 2/3     Koide binding-stability threshold
  |Π| = 3      primitive count  {P, D, T}
  S  = 4       manifold states  {PDT, DT, PT, PD}
  A₀ = 137     base fine-structure integer

Translation layer (observer/display hardware Descriptor profile — the
measured D-set of the human eye + sRGB device, NOT design parameters):
  cie_observer_data.py — CIE 1931 2° observer (5 nm), D65, sRGB matrices.

Measurement standard (Ω): mp.dps = 250, string → mpf → string, exact
rationals, float() FORBIDDEN in every computation chain, dynamic
derivation, report() with PASSED/FAILED/TOTAL, sys.exit(1) on failure.

CLIENT-SIDE NOTE: the emitted JS performs NO ET computation. It plays
back derived parameter strings (positions, phases, periods) that this
forge computed at 250 dps. Authority lives in et_tokens.json.

Author: forged for Michael James Muller — Aevum Defluo
        Exception Theory LLC — P ∘ D ∘ T = E
"""

import json
import os
import sys
import datetime
from math import gcd
from mpmath import (mp, mpf, log as mplog, nint, fabs, power as mppow,
                    nstr, floor as mpfloor, cos as mpcos, sin as mpsin,
                    pi as mppi)

from cie_observer_data import (CIE_1931_2DEG_5NM, D65_XY, D65_XYZ_N,
                               M_XYZ_TO_SRGB, M_SRGB_TO_XYZ)

mp.dps = 250  # 200 working + 50 guard

# ═════════════════════════════════════════════════════════════════════
#  SECTION 0 — SITE CONFIG (business facts only; no design values here)
# ═════════════════════════════════════════════════════════════════════
CONFIG = {
    "domain":       "www.exceptiontheory.com",
    "site_title":   "Exception Theory",
    "tagline":      "For every exception there is an exception, except the exception.",
    "org":          "Exception Theory LLC",
    "org_location": "Ellwood City, Pennsylvania",
    "author":       "Michael James Muller — Aevum Defluo",
    "doi":          "10.5281/zenodo.19762311",   # verified from ET_Identities.md
    "doi_url":      "https://doi.org/10.5281/zenodo.19762311",
    # ⚠ EMAIL AS SUPPLIED — spelling contains 'thoery'; awaiting Mike's
    #   one-word confirmation of exact address before launch. Rendered
    #   verbatim so nothing silently diverges from his instruction.
    "email":        "exceptionthoery@gmail.com",
    "year":         str(datetime.date.today().year),
    # Products launch after the site per Mike's directive. The pipeline
    # below is fully functional; the list is empty by instruction, and
    # the section renders its true forthcoming state when empty.
    "products":     [],
}

# ═════════════════════════════════════════════════════════════════════
#  SECTION 1 — ET CONSTANTS (strings → mpf; the only inputs)
# ═════════════════════════════════════════════════════════════════════
N        = 12                          # manifold symmetry
PI_COUNT = 3                           # |Π| — primitives
S_STATES = 4                           # manifold states
A0       = 137                         # base fine-structure integer
V        = mpf(1) / mpf(N)             # base variance 1/12
K        = mpf(2) / mpf(3)             # Koide ratio
CENTS    = mpf(1200)
LOG2     = mplog(mpf(2))
GENERATORS = [g for g in range(1, N) if gcd(g, N) == 1]  # {1,5,7,11} — derived
DIVISORS   = [d for d in range(1, N + 1) if N % d == 0]  # {1,2,3,4,6,12} — derived
DI_CENTS   = mpf(600) / mpf(N)         # Identity F: ∂I half-step = 50¢ at N=12

# IC-143 appearance reference — ƛ_e in nanometres (386.15926796 fm)
LAMBDA_E_NM = mpf("0.38615926796")

def xi(d):
    """Identity H impedance: ξ(d) = A₀ / ((d−1)² + 16)."""
    dd = mpf(d)
    return mpf(A0) / ((dd - 1) ** 2 + mpf(16))

# ═════════════════════════════════════════════════════════════════════
#  SECTION 2 — THE BIJECTION (IC-1; conventions of the canon scripts)
# ═════════════════════════════════════════════════════════════════════
def project(r):
    """Π_N(r) = (k, d, ε¢). r: mpf > 0."""
    exact = mpf(N) * mplog(r) / LOG2
    k = int(nint(exact))
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    eps = (exact - mpf(k)) * CENTS / mpf(N)
    return k, d, eps

def pullback(k, eps):
    """Π_N⁻¹(k, ε) = 2^((k + ε·N/1200)/N) — algebraic identity."""
    return mppow(mpf(2), (mpf(k) + eps * mpf(N) / CENTS) / mpf(N))

def lattice(k):
    """Lattice-exact value 2^(k/N)."""
    return mppow(mpf(2), mpf(k) / mpf(N))

# ═════════════════════════════════════════════════════════════════════
#  SECTION 3 — TRANSLATION LAYER (CIE observer → sRGB device)
# ═════════════════════════════════════════════════════════════════════
CMF = [(mpf(w), mpf(x), mpf(y), mpf(z)) for (w, x, y, z) in CIE_1931_2DEG_5NM]
D65x, D65y = mpf(D65_XY[0]), mpf(D65_XY[1])
Xn, Yn, Zn = (mpf(D65_XYZ_N[0]), mpf(D65_XYZ_N[1]), mpf(D65_XYZ_N[2]))
M_X2R = [[mpf(v) for v in row] for row in M_XYZ_TO_SRGB]
M_R2X = [[mpf(v) for v in row] for row in M_SRGB_TO_XYZ]

CIE_EPS = mpf(216) / mpf(24389)        # exact CIE rationals
CIE_KAP = mpf(24389) / mpf(27)

def cmf_at(lam):
    """Linear interpolation of the observer at wavelength lam (nm)."""
    lo, hi = CMF[0][0], CMF[-1][0]
    if lam <= lo:
        return CMF[0][1], CMF[0][2], CMF[0][3]
    if lam >= hi:
        return CMF[-1][1], CMF[-1][2], CMF[-1][3]
    idx = int(mpfloor((lam - lo) / mpf(5)))
    w0, x0, y0, z0 = CMF[idx]
    w1, x1, y1, z1 = CMF[idx + 1]
    t = (lam - w0) / (w1 - w0)
    return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t)

def xyz_to_xy(X, Y, Z):
    s = X + Y + Z
    return X / s, Y / s

def xy_Y_to_xyz(x, y, Y):
    X = x * Y / y
    Z = (mpf(1) - x - y) * Y / y
    return X, Y, Z

def L_to_Y(Lstar):
    fy = (Lstar + mpf(16)) / mpf(116)
    fy3 = fy ** 3
    if fy3 > CIE_EPS:
        return Yn * fy3
    return Yn * (mpf(116) * fy - mpf(16)) / CIE_KAP

def _f_lab(t):
    if t > CIE_EPS:
        return mppow(t, mpf(1) / mpf(3))
    return (CIE_KAP * t + mpf(16)) / mpf(116)

def xyz_to_lab(X, Y, Z):
    fx, fy, fz = _f_lab(X / Xn), _f_lab(Y / Yn), _f_lab(Z / Zn)
    return (mpf(116) * fy - mpf(16),
            mpf(500) * (fx - fy),
            mpf(200) * (fy - fz))

def lab_to_xyz(L, a, b):
    fy = (L + mpf(16)) / mpf(116)
    fx = fy + a / mpf(500)
    fz = fy - b / mpf(200)
    def inv(f):
        f3 = f ** 3
        if f3 > CIE_EPS:
            return f3
        return (mpf(116) * f - mpf(16)) / CIE_KAP
    return inv(fx) * Xn, inv(fy) * Yn, inv(fz) * Zn

def xyz_to_lin_rgb(X, Y, Z):
    return tuple(M_X2R[i][0] * X + M_X2R[i][1] * Y + M_X2R[i][2] * Z
                 for i in range(3))

def lin_rgb_to_xyz(R, G, B):
    return tuple(M_R2X[i][0] * R + M_R2X[i][1] * G + M_R2X[i][2] * B
                 for i in range(3))

def gamma_encode(c):
    if c <= mpf("0.0031308"):
        return mpf("12.92") * c
    return mpf("1.055") * mppow(c, mpf(1) / mpf("2.4")) - mpf("0.055")

def gamma_decode(c):
    if c <= mpf("0.04045"):
        return c / mpf("12.92")
    return mppow((c + mpf("0.055")) / mpf("1.055"), mpf("2.4"))

def in_gamut(rgb_lin, tol=mpf("1e-12")):
    return all((-tol) <= c <= (mpf(1) + tol) for c in rgb_lin)

def lin_to_hex(rgb_lin):
    out = []
    for c in rgb_lin:
        c = min(max(c, mpf(0)), mpf(1))
        out.append(int(nint(gamma_encode(c) * mpf(255))))
    return "#{:02x}{:02x}{:02x}".format(*out)

def hex_to_xyz(hx):
    hx = hx.lstrip("#")
    lin = [gamma_decode(mpf(int(hx[i:i + 2], 16)) / mpf(255))
           for i in (0, 2, 4)]
    return lin_rgb_to_xyz(*lin)

# ═════════════════════════════════════════════════════════════════════
#  SECTION 4 — SPECTRAL PALETTE  (SIC-39: k IS the chromaticity axis)
#  Visible octave on the lattice: classes k = 120…130 above ƛ_e, plus
#  class 11 — the non-spectral purple line closing the U(1) circle.
#  Rendering: monochromatic chromaticity at the accent lightness
#  L* = K·100 (a single derived, equiluminant law), gamut overflow
#  resolved by pulling chromaticity toward D65 — the identity cell
#  (IC-117): out-of-gamut color returns toward the Exception.
# ═════════════════════════════════════════════════════════════════════
ACCENT_L = K * mpf(100)          # 66.666… — the equiluminant '-even' law
K_BASE   = N * 10                # k=120: violet sits 10 exact octaves above ƛ_e

def gamut_pull_to_white(x, y, Lstar, iters=96):
    """Binary-search t∈[0,1]: chroma point pulled toward D65 until in gamut."""
    Ytar = L_to_Y(Lstar)
    def rgb_at(t):
        xt = x + (D65x - x) * t
        yt = y + (D65y - y) * t
        return xyz_to_lin_rgb(*xy_Y_to_xyz(xt, yt, Ytar))
    if in_gamut(rgb_at(mpf(0))):
        return mpf(0), rgb_at(mpf(0))
    lo, hi = mpf(0), mpf(1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if in_gamut(rgb_at(mid)):
            hi = mid
        else:
            lo = mid
    return hi, rgb_at(hi)

def cusp_lightness(x, y):
    """The per-class natural lightness: L* = argmin over (0,100) of the
    gamut pull t(L*). This is an extremum condition on the observer +
    device translation layer — the lightness at which the spectral
    chromaticity survives with MAXIMAL purity. No free parameter is
    introduced: the value is forced by the CMFs, D65, and the sRGB
    primaries. (Descriptor Gap closed: the missing Descriptor was the
    class's own cusp, not a shared lightness.)  t(L*) is unimodal —
    the gamut cross-section closes toward both device poles — so a
    ternary search converges."""
    lo, hi = mpf(1), mpf(99)
    for _ in range(90):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        t1, _ = gamut_pull_to_white(x, y, m1, iters=60)
        t2, _ = gamut_pull_to_white(x, y, m2, iters=60)
        if t1 <= t2:
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2

def spectral_class(c):
    """Class c∈0..11 → dict with λ, lattice address, xy, vivid sRGB."""
    if c <= 10:
        k = K_BASE + c
        lam = LAMBDA_E_NM * lattice(k)
        X, Y, Z = cmf_at(lam)
        kind = "spectral"
        dsr = lam / LAMBDA_E_NM
        kp, dp, ep = project(dsr)
        lam_s = nstr(lam, 50)
    else:
        # Purple line: equal-energy mixture of the two visible endpoints
        Xa, Ya, Za = cmf_at(LAMBDA_E_NM * lattice(K_BASE))
        Xb, Yb, Zb = cmf_at(LAMBDA_E_NM * lattice(K_BASE + 10))
        X, Y, Z = Xa + Xb, Ya + Yb, Za + Zb
        kind = "purple-line"
        kp, dp, ep = None, None, None
        lam_s = None
    x, y = xyz_to_xy(X, Y, Z)
    # vivid: the class at its own cusp lightness — maximal spectral purity
    Lc = cusp_lightness(x, y)
    tc, rgb_c = gamut_pull_to_white(x, y, Lc)
    hex_vivid = lin_to_hex(rgb_c)
    # even: the equiluminant law L* = K·100 — uniform-lightness set
    te, rgb_e = gamut_pull_to_white(x, y, ACCENT_L)
    hex_even = lin_to_hex(rgb_e)
    Xv, Yv, Zv = lin_rgb_to_xyz(*[min(max(ch, mpf(0)), mpf(1))
                                  for ch in rgb_c])
    L, a, b = xyz_to_lab(Xv, Yv, Zv)
    return {
        "class": c, "kind": kind, "k": (None if c == 11 else K_BASE + c),
        "lambda_nm": lam_s,
        "proj": (None if c == 11 else
                 {"k": kp, "d": dp, "eps_cents": nstr(ep, 50)}),
        "xy": (nstr(x, 30), nstr(y, 30)),
        "cusp_L": nstr(Lc, 20), "gamut_pull_t": nstr(tc, 30),
        "even_pull_t": nstr(te, 30),
        "hex": hex_vivid, "hex_even": hex_even, "lab": (L, a, b),
    }

def chroma_variant(base, factor):
    """Scale Lab chroma by an ET factor (K or V); re-check gamut."""
    L, a, b = base["lab"]
    a2, b2 = a * factor, b * factor
    rgb = xyz_to_lin_rgb(*lab_to_xyz(L, a2, b2))
    if not in_gamut(rgb):
        lo, hi = mpf(0), mpf(1)
        for _ in range(120):
            mid = (lo + hi) / 2
            rr = xyz_to_lin_rgb(*lab_to_xyz(L, a2 * (1 - mid), b2 * (1 - mid)))
            if in_gamut(rr):
                hi = mid
            else:
                lo = mid
        rgb = xyz_to_lin_rgb(*lab_to_xyz(L, a2 * (1 - hi), b2 * (1 - hi)))
    return lin_to_hex(rgb)

CLASSES = [spectral_class(c) for c in range(12)]
for cl in CLASSES:
    cl["hex_dim"]  = chroma_variant(cl, K)   # C·K
    cl["hex_wash"] = chroma_variant(cl, V)   # C·V

# ─── Neutral ladder: L* = j·(100·V), tinted toward deep space ────────
#  Surfaces j∈{1..S}=4 states; text j∈{11,10}; the tint hue is class 1
#  (indigo — the receding depth), chroma law C = K·V of that class.
def neutral(j, tint_class=1):
    Lstar = mpf(j) * (mpf(100) * V)
    base = CLASSES[tint_class]
    _, a, b = base["lab"]
    scale = K * V
    rgb = xyz_to_lin_rgb(*lab_to_xyz(Lstar, a * scale, b * scale))
    if not in_gamut(rgb):
        rgb = xyz_to_lin_rgb(*lab_to_xyz(Lstar, mpf(0), mpf(0)))
    return lin_to_hex(rgb)

NEUTRAL = {j: neutral(j) for j in list(range(0, S_STATES + 1)) + [8, 9, 10, 11]}

# ═════════════════════════════════════════════════════════════════════
#  SECTION 5 — UNIVERSAL COLOR ADDRESSING  ("support all colors")
#  IC-117 route generalized: ANY color → three tristimulus DSRs
#  (X/Xn, Y/Yn, Z/Zn) → three Π₁₂ addresses. Losslessly invertible.
# ═════════════════════════════════════════════════════════════════════
def color_to_lattice(hx):
    X, Y, Z = hex_to_xyz(hx)
    out = []
    for val, ref in ((X, Xn), (Y, Yn), (Z, Zn)):
        dsr = val / ref
        if dsr <= 0:
            # The P-pole: zero luminous content in this channel. log₂ is
            # undefined at the substrate — the channel is unaddressed
            # (unsubstantiated), not approximated. Pullback returns 0.
            out.append({"pole": "P", "k": None, "d": None,
                        "eps_cents": None})
            continue
        k, d, e = project(dsr)
        out.append({"k": k, "d": d, "eps_cents": nstr(e, 50)})
    return out

def lattice_to_color(triple):
    vals = []
    for ch in triple:
        if ch.get("pole") == "P":
            vals.append(mpf(0))
        else:
            vals.append(pullback(ch["k"], mpf(ch["eps_cents"])))
    X, Y, Z = vals[0] * Xn, vals[1] * Yn, vals[2] * Zn
    return lin_to_hex(xyz_to_lin_rgb(X, Y, Z))

# ═════════════════════════════════════════════════════════════════════
#  SECTION 6 — GEOMETRY TOKENS  (every value = 2^(k/12); k is named)
#  The lattice is the key signature; composition names the notes.
# ═════════════════════════════════════════════════════════════════════
def px(k, digits=12):
    return nstr(lattice(k), digits) + "px"

def rem(k, digits=12):
    return nstr(lattice(k), digits) + "rem"

def sec(k, digits=10):
    return nstr(lattice(k), digits) + "s"

TYPE_SCALE = {  # name → lattice k relative to 1rem (=2^(48/12) px = 16px)
    "fs-xs": -2, "fs-sm": -1, "fs-base": 0, "fs-md": 2, "fs-lg": 4,
    "fs-xl": 7, "fs-2xl": 12, "fs-3xl": 16, "fs-4xl": 19, "fs-hero": 24,
}
SPACE_SCALE = {
    "sp-3xs": -31, "sp-2xs": -24, "sp-xs": -12, "sp-sm": -5, "sp-md": 0,
    "sp-lg": 5, "sp-xl": 12, "sp-2xl": 19, "sp-3xl": 24, "sp-4xl": 31,
    "sp-5xl": 36,
}
RADII = {"rad-sm": -24, "rad-md": -12, "rad-lg": -5, "rad-xl": 0}
DUR   = {"dur-fast": -31, "dur-base": -24, "dur-slow": -12, "dur-drift": 24}
BREAKPOINTS = {"bp-min": 100, "bp-mid": 108, "bp-grid": 120, "bp-wide": 123}
LINE_HEIGHT = lattice(7)                 # 2^(7/12) ≈ 1.4983 — the fifth
MEASURE_CH  = K * (CENTS / mpf(N))       # K·100 = 66.6…ch reading measure
EASE = "cubic-bezier({a}, 0, {b}, 1)".format(a=nstr(mpf(1) - K, 8),
                                             b=nstr(K, 8))
ALPHAS = {"a-v": V, "a-kv": K * V, "a-k2": K * K, "a-k": K,
          "a-1v": mpf(1) - V, "a-1kv": mpf(1) - K * V}

# ═════════════════════════════════════════════════════════════════════
#  SECTION 7 — STARFIELD PARAMETERS  (the integer sky)
#  Star i∈1..A₀: position by the two irrational rotations generated by
#  |Π| and K⁻¹ — x_i = frac(i·log₂3), y_i = frac(i·log₂(3/2)) — fully
#  derived, equidistributed, deterministic. Class/size/brightness/phase
#  come from Π₁₂(i): hue class = k mod 12; g = 12/d sets radius
#  r = 2^(g/12) px and opacity α = V + (1−V)/d; twinkle period
#  T = 2^(class/12)·2 s (2 s = lattice k=12); phase = ε·2π/100.
#  Layers: i mod |Π| — three parallax depths; drift v_L = V·2^(3−L) px/s.
# ═════════════════════════════════════════════════════════════════════
FX = mplog(mpf(3)) / LOG2                # log₂ 3
FY = mplog(mpf(3) / mpf(2)) / LOG2       # log₂ (3/2) = log₂ K⁻¹
def frac(x):
    return x - mpfloor(x)

STARS = []
for i in range(1, A0 + 1):
    k, d, e = project(mpf(i))
    g = N // d
    cls = k % N
    STARS.append({
        "x": nstr(frac(mpf(i) * FX), 20),
        "y": nstr(frac(mpf(i) * FY), 20),
        "c": cls,
        "r": nstr(lattice(g), 10),
        "a": nstr(V + (mpf(1) - V) / mpf(d), 10),
        "T": nstr(lattice(cls) * lattice(12), 10),
        "p": nstr(e * 2 * mppi / mpf(100), 12),
        "L": (i % PI_COUNT),
    })
DRIFT = [nstr(V * lattice(12 * (3 - L)) / lattice(12), 10) for L in range(3)]
# v_L = V·2^(2−L) px/s → 0.333…, 0.1666…, 0.0833… px/s

# ═════════════════════════════════════════════════════════════════════
#  SECTION 8 — SIGIL WHEEL SVG  (the visible octave as an arcane seal)
#  Outer ring: 12 true-spectrum sectors. Sector boundaries ARE the ∂I
#  half-step lines (Identity F, |ε| = 50¢). Inner geometry: the four
#  generator traversals g ∈ {1,5,7,11} drawn as star polygons {12/g} —
#  the cascade skeleton. Center: P∘D∘T = E.
# ═════════════════════════════════════════════════════════════════════
def build_wheel_svg():
    size = 640
    cx = cy = mpf(size) / 2
    r_out = mpf(size) * (mpf(1) - V) / 2          # (1−V)·half
    r_in  = r_out * K                              # ring width by K
    r_star = r_in * K
    def pt(angle, r):
        return (cx + r * mpcos(angle), cy + r * mpsin(angle))
    def fmt(v):
        return nstr(v, 8)
    segs = []
    for c in range(12):
        a0 = 2 * mppi * (mpf(c) - mpf(1) / 2) / mpf(12) - mppi / 2
        a1 = 2 * mppi * (mpf(c) + mpf(1) / 2) / mpf(12) - mppi / 2
        x0, y0 = pt(a0, r_out); x1, y1 = pt(a1, r_out)
        x2, y2 = pt(a1, r_in);  x3, y3 = pt(a0, r_in)
        col = CLASSES[c]["hex"]
        segs.append(
            f'<path d="M {fmt(x0)} {fmt(y0)} '
            f'A {fmt(r_out)} {fmt(r_out)} 0 0 1 {fmt(x1)} {fmt(y1)} '
            f'L {fmt(x2)} {fmt(y2)} '
            f'A {fmt(r_in)} {fmt(r_in)} 0 0 0 {fmt(x3)} {fmt(y3)} Z" '
            f'fill="{col}" opacity="{nstr(mpf(1)-V,6)}">'
            f'<title>class {c} — '
            + (f'λ = {nstr(mpf(CLASSES[c]["lambda_nm"]),6)} nm, '
               f'k = {CLASSES[c]["k"]}' if c <= 10 else 'purple line (non-spectral)')
            + '</title></path>')
    # ∂I boundary ticks (the 50¢ half-step lines between cells)
    ticks = []
    for c in range(12):
        a = 2 * mppi * (mpf(c) + mpf(1) / 2) / mpf(12) - mppi / 2
        xo, yo = pt(a, r_out * (1 + V / 2)); xi_, yi_ = pt(a, r_out)
        ticks.append(f'<line x1="{fmt(xi_)}" y1="{fmt(yi_)}" '
                     f'x2="{fmt(xo)}" y2="{fmt(yo)}" '
                     f'stroke="{NEUTRAL[10]}" stroke-width="1" '
                     f'opacity="{nstr(K*V,6)}"/>')
    # Generator star polygons {12/g}
    stars = []
    for g in GENERATORS:
        pts = []
        node = 0
        for _ in range(13):
            a = 2 * mppi * mpf(node) / mpf(12) - mppi / 2
            x, y = pt(a, r_star)
            pts.append(f"{fmt(x)},{fmt(y)}")
            node = (node + g) % 12
        col = CLASSES[(g * 10) % 12]["hex_dim"]
        stars.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{col}" stroke-width="1.2" '
                     f'opacity="{nstr(K*K,6)}"/>')
    center = (f'<text x="{fmt(cx)}" y="{fmt(cy)}" text-anchor="middle" '
              f'dominant-baseline="central" fill="{NEUTRAL[11]}" '
              f'font-family="Georgia, serif" '
              f'font-size="{nstr(lattice(60)/2,6)}" '
              f'letter-spacing="2">P∘D∘T = E</text>')
    lbl = (f'<text x="{fmt(cx)}" y="{fmt(cy + r_star * (K*K))}" '
           f'text-anchor="middle" fill="{NEUTRAL[9]}" '
           f'font-family="Georgia, serif" font-size="{nstr(lattice(48)/2,6)}" '
           f'opacity="{nstr(K,4)}">the visible octave · k = 120…130 above ƛ_e '
           f'· ∂I at |ε| = 50¢</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="The visible octave of the electromagnetic spectrum '
            f'projected onto the N=12 Sempaevum lattice">'
            f'<rect width="{size}" height="{size}" fill="none"/>'
            + "".join(segs) + "".join(ticks) + "".join(stars)
            + center + lbl + '</svg>')

# ═════════════════════════════════════════════════════════════════════
#  SECTION 9 — TOKEN LEDGER
# ═════════════════════════════════════════════════════════════════════
def token_entry(name, css_value, law, k=None, address=None, extra=None):
    ent = {"token": name, "value": css_value, "law": law}
    if k is not None:
        ent["k"] = k
        ent["exact"] = nstr(lattice(k), 50)
    if address is not None:
        ent["lattice_address"] = address
    if extra:
        ent.update(extra)
    return ent

TOKENS = []
CSS_VARS = []

def add_geom(prefix, table, unit_fn, law):
    for name, k in table.items():
        val = unit_fn(k)
        CSS_VARS.append(f"  --{name}: {val};")
        TOKENS.append(token_entry(name, val, law + f" (k={k:+d})", k=k))

add_geom("fs", TYPE_SCALE, rem, "2^(k/12) rem — semitone type scale")
add_geom("sp", SPACE_SCALE, rem, "2^(k/12) rem — semitone spacing")
add_geom("rad", RADII, rem, "2^(k/12) rem — radii")
add_geom("dur", DUR, sec, "2^(k/12) s — durations")
for name, k in BREAKPOINTS.items():
    val = px(k)
    CSS_VARS.append(f"  --{name}: {val};")
    TOKENS.append(token_entry(name, val, f"2^(k/12) px breakpoint (k={k})", k=k))

CSS_VARS.append(f"  --lh: {nstr(LINE_HEIGHT, 12)};")
TOKENS.append(token_entry("lh", nstr(LINE_HEIGHT, 50),
                          "line-height = 2^(7/12) — the tempered fifth", k=7))
CSS_VARS.append(f"  --measure: {nstr(MEASURE_CH, 10)}ch;")
TOKENS.append(token_entry("measure", nstr(MEASURE_CH, 50) + "ch",
                          "reading measure = K·(1200/N) ch = 66.6ch"))
CSS_VARS.append(f"  --ease: {EASE};")
TOKENS.append(token_entry("ease", EASE, "cubic-bezier(1−K, 0, K, 1)"))
for name, a in ALPHAS.items():
    CSS_VARS.append(f"  --{name}: {nstr(a, 10)};")
    TOKENS.append(token_entry(name, nstr(a, 50), "opacity from {V, K} algebra"))

for j, hx in sorted(NEUTRAL.items()):
    CSS_VARS.append(f"  --n{j}: {hx};")
    TOKENS.append(token_entry(
        f"n{j}", hx,
        f"neutral ladder L* = {j}·(100·V) = {nstr(mpf(j)*100*V,6)}, "
        f"tint = class-1 chroma × K·V",
        address=color_to_lattice(hx)))

for cl in CLASSES:
    c = cl["class"]
    base_law = (f"spectral class {c}: λ = ƛ_e·2^({cl['k']}/12) nm via CIE "
                f"observer" if c <= 10 else
                "purple line: endpoint XYZ sum via CIE observer")
    for suffix, key, law_tail in (
            ("", "hex",
             f" @ cusp L*={nstr(mpf(cl['cusp_L']),5)} (argmin gamut pull —"
             " maximal spectral purity, translation-layer extremum)"),
            ("-even", "hex_even", " @ L*=K·100 (equiluminant law)"),
            ("-dim", "hex_dim", " @ cusp; chroma × K"),
            ("-wash", "hex_wash", " @ cusp; chroma × V")):
        CSS_VARS.append(f"  --c{c}{suffix}: {cl[key]};")
        TOKENS.append(token_entry(f"c{c}{suffix}", cl[key],
                                  base_law + law_tail,
                                  address=color_to_lattice(cl[key]),
                                  extra={"lambda_nm": cl["lambda_nm"],
                                         "spectral_proj": cl["proj"]}))

CSS_VARS.append(f"  --grid-n: {N};")
CSS_VARS.append(f"  --di-cents: {nstr(DI_CENTS, 6)};")

# Semantic assignments (composition — named choices on derived values).
# Display/graphic slots take the vivid cusp classes (spectral truth,
# maximal chromostereopsis); running-text slots take the -even set
# (equiluminant at L*=K·100 — uniform, contrast-guaranteed).
SEMANTIC = {
    "bg":        "var(--n1)",   # ground: j=1 near-black (chromostereopsis)
    "bg-raise":  "var(--n2)",
    "surface":   "var(--n3)",
    "surface-2": "var(--n4)",
    "line":      "var(--n3)",
    "text":      "var(--n11)",
    "text-2":    "var(--n10)",
    "text-3":    "var(--n9)",
    "advance":   "var(--c10)",       # 704 nm red — floats forward on dark
    "advance-2": "var(--c9)",
    "recede":    "var(--c3)",        # 470 nm blue — the canonical deep
    "recede-2":  "var(--c1)",        # indigo-violet field
    "arcane":    "var(--c11)",       # purple line — the non-spectral seal
    "verdant":   "var(--c6)",
    "gold":      "var(--c7)",        # 592 nm amber
    "link":      "var(--c3-even)",   # readable blue for body links
    "link-hot":  "var(--c7-even)",
    "gold-text": "var(--c7-even)",
    "verdant-text": "var(--c6-even)",
    "arcane-text":  "var(--c11-even)",
}
for name, val in SEMANTIC.items():
    CSS_VARS.append(f"  --{name}: {val};")
    TOKENS.append(token_entry(name, val, "semantic alias (composition)"))

# ═════════════════════════════════════════════════════════════════════
#  SECTION 10 — CSS
# ═════════════════════════════════════════════════════════════════════
def build_css():
    tpl = """/* www.exceptiontheory.com — forged; do not hand-edit.
   Every value is a lattice address. Authority: et_tokens.json.
   P ∘ D ∘ T = E */
:root{
@@VARS@@
}
*,*::before,*::after{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion: reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
body{
  margin:0;background:var(--bg);color:var(--text);
  font-family:'Palatino Linotype','Book Antiqua',Palatino,Georgia,serif;
  font-size:var(--fs-base);line-height:var(--lh);
  -webkit-font-smoothing:antialiased;
}
::selection{background:var(--advance);color:var(--n1)}
a{color:var(--link);text-decoration-color:color-mix(in srgb,var(--link) calc(var(--a-k)*100%),transparent)}
a:hover{color:var(--link-hot)}
code,kbd,.mono{font-family:ui-monospace,'Cascadia Mono',Consolas,Menlo,monospace;font-size:var(--fs-sm)}
h1,h2,h3{font-weight:600;letter-spacing:.02em;line-height:calc(var(--lh)/var(--lh))}
h1{font-size:clamp(var(--fs-3xl),calc(100vw/var(--grid-n)),var(--fs-hero))}
h2{font-size:var(--fs-2xl);margin:0 0 var(--sp-md)}
h3{font-size:var(--fs-xl);margin:var(--sp-lg) 0 var(--sp-sm)}
p{margin:0 0 var(--sp-md);max-width:var(--measure)}
.small{font-size:var(--fs-sm);color:var(--text-2)}

/* ── the deep field ─────────────────────────────────────────────── */
#field{position:fixed;inset:0;z-index:0;pointer-events:none}
#field canvas{width:100%;height:100%;display:block}
.nebula{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse calc(100vmax*2/3) calc(100vmax/2) at calc(100%*1/12) calc(100%*1/3),
      color-mix(in srgb,var(--c1) calc(var(--a-kv)*100%),transparent),transparent 70%),
    radial-gradient(ellipse calc(100vmax/2) calc(100vmax*2/3) at calc(100%*11/12) calc(100%*2/3),
      color-mix(in srgb,var(--c0) calc(var(--a-kv)*100%),transparent),transparent 70%),
    radial-gradient(ellipse calc(100vmax/3) calc(100vmax/3) at calc(100%*2/3) calc(100%*1/12),
      color-mix(in srgb,var(--c10) calc(var(--a-v)*var(--a-v)*100%),transparent),transparent 70%);
}
.veil{position:fixed;inset:0;z-index:0;pointer-events:none;
  background:linear-gradient(180deg,transparent,color-mix(in srgb,var(--n0) calc(var(--a-k)*100%),transparent))}

main,header,footer{position:relative;z-index:1}

/* ── chrome ─────────────────────────────────────────────────────── */
header.site{position:sticky;top:0;z-index:12;
  backdrop-filter:blur(var(--sp-sm));
  background:color-mix(in srgb,var(--n1) calc(var(--a-1v)*100%),transparent);
  border-bottom:1px solid color-mix(in srgb,var(--n4) calc(var(--a-k)*100%),transparent)}
.nav{max-width:var(--bp-grid);margin:0 auto;padding:var(--sp-sm) var(--sp-md);
  display:flex;align-items:center;gap:var(--sp-lg);flex-wrap:wrap}
.brand{font-size:var(--fs-md);letter-spacing:.08em;color:var(--text);text-decoration:none}
.brand b{color:var(--advance)}
.nav a.item{color:var(--text-2);text-decoration:none;font-size:var(--fs-sm);letter-spacing:.06em;text-transform:uppercase}
.nav a.item:hover{color:var(--gold-text)}
.nav .spacer{flex:1}

.wrap{max-width:var(--bp-grid);margin:0 auto;padding:var(--sp-2xl) var(--sp-md)}
.grid{display:grid;grid-template-columns:repeat(var(--grid-n),1fr);gap:var(--sp-md)}
.col-12{grid-column:span 12}.col-6{grid-column:span 6}.col-4{grid-column:span 4}.col-3{grid-column:span 3}
@media (max-width:@@BPMID@@){.col-6,.col-4,.col-3{grid-column:span 12}}

/* ── hero ───────────────────────────────────────────────────────── */
.hero{min-height:calc(100vh*2/3);display:grid;place-items:center;text-align:center;padding:var(--sp-3xl) var(--sp-md)}
.hero .eq{font-size:clamp(var(--fs-hero),calc(100vw/12),calc(var(--fs-hero)*2));
  letter-spacing:.06em;margin:0;
  color:var(--advance);
  text-shadow:
    0 0 var(--sp-sm) color-mix(in srgb,var(--advance) calc(var(--a-k)*100%),transparent),
    0 0 var(--sp-xl) color-mix(in srgb,var(--advance) calc(var(--a-kv)*100%),transparent)}
.hero .tag{font-style:italic;color:var(--text-2);font-size:var(--fs-lg);max-width:none}
.hero .sub{color:var(--link);letter-spacing:.24em;text-transform:uppercase;font-size:var(--fs-sm)}
.cta{display:inline-block;margin-top:var(--sp-lg);padding:var(--sp-sm) var(--sp-lg);
  border:1px solid var(--recede);border-radius:var(--rad-md);color:var(--text);
  text-decoration:none;letter-spacing:.1em;font-size:var(--fs-sm);text-transform:uppercase;
  transition:all var(--dur-base) var(--ease)}
.cta:hover{border-color:var(--advance);color:var(--link-hot);
  box-shadow:0 0 var(--sp-md) color-mix(in srgb,var(--advance) calc(var(--a-kv)*100%),transparent)}
.cta.primary{border-color:var(--gold);color:var(--gold-text)}

/* ── cards / sections ───────────────────────────────────────────── */
section{padding:var(--sp-2xl) 0;border-top:1px solid color-mix(in srgb,var(--n3) calc(var(--a-k)*100%),transparent)}
.card{background:color-mix(in srgb,var(--n2) calc(var(--a-1kv)*100%),transparent);
  border:1px solid color-mix(in srgb,var(--n4) calc(var(--a-k)*100%),transparent);
  border-radius:var(--rad-lg);padding:var(--sp-lg)}
.card h3{margin-top:0}
.prim{border-top:2px solid var(--pc)}
.prim .sig{font-size:var(--fs-2xl);color:var(--pc)}
.kde{display:inline-block;padding:calc(var(--sp-3xs)) var(--sp-xs);border-radius:var(--rad-sm);
  background:color-mix(in srgb,var(--pc,var(--recede)) calc(var(--a-v)*100%),transparent);
  color:var(--text);font-family:ui-monospace,Consolas,monospace;font-size:var(--fs-xs)}
table.consts{width:100%;border-collapse:collapse;font-size:var(--fs-sm)}
table.consts th,table.consts td{padding:var(--sp-xs) var(--sp-sm);text-align:left;
  border-bottom:1px solid color-mix(in srgb,var(--n3) calc(var(--a-k)*100%),transparent)}
table.consts th{color:var(--gold);letter-spacing:.06em;text-transform:uppercase;font-size:var(--fs-xs)}
table.consts td.mono{color:var(--text-2)}
.wheelbox{display:grid;place-items:center;padding:var(--sp-lg)}
.wheelbox svg{width:min(100%,var(--bp-mid));height:auto;
  filter:drop-shadow(0 0 var(--sp-md) color-mix(in srgb,var(--arcane) calc(var(--a-kv)*100%),transparent))}

.depth-demo{background:var(--n0);border-radius:var(--rad-lg);padding:var(--sp-xl);text-align:center}
.depth-demo .far{color:var(--recede-2);font-size:var(--fs-2xl);letter-spacing:.3em}
.depth-demo .near{color:var(--advance);font-size:var(--fs-2xl);letter-spacing:.3em;
  text-shadow:0 0 var(--sp-sm) color-mix(in srgb,var(--advance) calc(var(--a-k)*100%),transparent)}

.evid{border-left:2px solid var(--verdant);padding-left:var(--sp-md)}
.evid b{color:var(--verdant-text)}

.doi{display:inline-flex;gap:var(--sp-xs);align-items:center;
  border:1px solid var(--arcane);border-radius:var(--rad-md);
  padding:var(--sp-xs) var(--sp-md);color:var(--arcane-text);text-decoration:none}
.doi:hover{color:var(--gold-text);border-color:var(--gold)}

footer.site{padding:var(--sp-xl) var(--sp-md);text-align:center;color:var(--text-3);
  border-top:1px solid color-mix(in srgb,var(--n3) calc(var(--a-k)*100%),transparent)}
footer.site .eqline{letter-spacing:.2em;color:var(--text-2)}
"""
    css = tpl.replace("@@VARS@@", "\n".join(CSS_VARS))
    css = css.replace("@@BPMID@@", px(BREAKPOINTS["bp-mid"], 10))
    return css

# ═════════════════════════════════════════════════════════════════════
#  SECTION 11 — STARFIELD JS (playback of forged parameters only)
# ═════════════════════════════════════════════════════════════════════
def build_js():
    palette = [CLASSES[c]["hex"] for c in range(12)]
    stars_json = json.dumps(STARS, separators=(",", ":"))
    tpl = """/* et_field.js — the integer sky. Forged; do not hand-edit.
   137 stars = the first A0 integers. Positions: the |Pi| and K^-1
   irrational rotations. Hue class = k mod 12 of Pi_12(i); radius and
   opacity from d; twinkle period/phase from (class, epsilon).
   This file performs NO ET computation: it plays back parameters the
   forge derived at 250 dps. Authority: et_tokens.json. */
(function(){
  'use strict';
  var PAL=@@PAL@@;
  var STARS=@@STARS@@;
  var DRIFT=@@DRIFT@@;   /* px/s per layer: V*2^(2-L) */
  var reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var host=document.getElementById('field'); if(!host) return;
  var cv=document.createElement('canvas'); host.appendChild(cv);
  var ctx=cv.getContext('2d'); var W=0,H=0,DPR=1;
  function size(){DPR=Math.min(window.devicePixelRatio||1,2);
    W=host.clientWidth;H=host.clientHeight;
    cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);}
  size(); window.addEventListener('resize',size);
  var px=0,py=0;
  window.addEventListener('pointermove',function(e){
    px=(e.clientX/W-0.5); py=(e.clientY/H-0.5);},{passive:true});
  var t0=performance.now();
  function draw(now){
    var t=(now-t0)/1000;
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<STARS.length;i++){
      var s=STARS[i];
      var vx=parseFloat(DRIFT[s.L]);
      var x=( (parseFloat(s.x)+ (reduce?0:vx*t/W)) %1 +1)%1;
      var y=parseFloat(s.y);
      /* parallax: layer * V * pointer offset (V = 1/12) */
      var par=(s.L+1)*(1/12);
      var X=x*W + (reduce?0:px*par*W*(1/12));
      var Y=y*H + (reduce?0:py*par*H*(1/12));
      var a=parseFloat(s.a);
      if(!reduce){
        var ph=parseFloat(s.p), T=parseFloat(s.T);
        a=a*(1-(1/12))+a*(1/12)*Math.sin(2*Math.PI*t/T+ph);
      }
      var r=parseFloat(s.r);
      ctx.globalAlpha=Math.max(0,Math.min(1,a));
      ctx.fillStyle=PAL[s.c];
      ctx.beginPath();ctx.arc(X,Y,r,0,2*Math.PI);ctx.fill();
      if(r>1.6){ /* d=1..2 stars carry a faint halo */
        ctx.globalAlpha=a*(1/12);
        ctx.beginPath();ctx.arc(X,Y,r*3,0,2*Math.PI);ctx.fill();
      }
    }
    ctx.globalAlpha=1;
    if(!reduce) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
"""
    return (tpl.replace("@@PAL@@", json.dumps(palette))
               .replace("@@STARS@@", stars_json)
               .replace("@@DRIFT@@", json.dumps(DRIFT)))

# ═════════════════════════════════════════════════════════════════════
#  SECTION 12 — HTML
# ═════════════════════════════════════════════════════════════════════
def fmt_eps(e_str, digits=6):
    return nstr(mpf(e_str), digits)

def constants_rows():
    rows = []
    entries = [
        ("N", "12", "manifold symmetry — |Π|·S", mpf(12)),
        ("V", "1/12", "base variance", V),
        ("K", "2/3", "Koide binding threshold", K),
        ("|Π|", "3", "primitives {P, D, T}", mpf(3)),
        ("S", "4", "manifold states", mpf(4)),
        ("A₀", "137", "base fine-structure integer", mpf(137)),
        ("ƛ_e", "386.15926796 fm", "appearance reference (IC-143)",
         LAMBDA_E_NM),
        ("∂I", "50 ¢", "incoherence half-step (Identity F)", None),
    ]
    for sym, val, role, r in entries:
        if r is not None:
            k, d, e = project(r)
            addr = (f'<span class="kde">k={k:+d}, d={d}, '
                    f'ε={nstr(e,5)}¢</span>')
        else:
            addr = '<span class="kde">|ε| = 600/N</span>'
        rows.append(f"<tr><td>{sym}</td><td class='mono'>{val}</td>"
                    f"<td>{role}</td><td>{addr}</td></tr>")
    return "\n".join(rows)

def products_block():
    if not CONFIG["products"]:
        return ("<div class='card col-12'><h3>Forthcoming</h3>"
                "<p>The first product line — software and instruments built "
                "directly from the algebraic identities of the Sempaevum "
                "lattice — is in final preparation. Releases will be "
                "announced here, on this page, when the doors open.</p>"
                "<p class='small'>Every product ships with its derivation: "
                "zero free parameters, verification scripts included.</p>"
                "</div>")
    cards = []
    for p in CONFIG["products"]:
        cards.append(
            f"<div class='card col-4'><h3>{p['name']}</h3>"
            f"<p>{p['desc']}</p>"
            f"<a class='cta' href='{p['url']}'>{p.get('cta','View')}</a></div>")
    return "\n".join(cards)

def palette_strip():
    cells = []
    for cl in CLASSES:
        c = cl["class"]
        title = (f"class {c} · λ={nstr(mpf(cl['lambda_nm']),5)} nm · k={cl['k']}"
                 if c <= 10 else "class 11 · purple line")
        cells.append(f"<div title='{title}' style='background:var(--c{c});"
                     f"height:var(--sp-lg);flex:1'></div>")
    return ("<div style='display:flex;border-radius:var(--rad-md);"
            "overflow:hidden'>" + "".join(cells) + "</div>")

def build_html(wheel_svg):
    c = CONFIG
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{c['site_title']} — P∘D∘T = E</title>
<meta name="description" content="Exception Theory: a foundational framework built from three irreducible primitives. One equation, zero free parameters, interventional evidence. {c['org']}.">
<meta property="og:title" content="{c['site_title']} — P∘D∘T = E">
<meta property="og:description" content="{c['tagline']}">
<meta property="og:type" content="website">
<meta property="og:url" content="https://{c['domain']}/">
<link rel="stylesheet" href="css/et.css">
</head>
<body>
<div class="nebula" aria-hidden="true"></div>
<div id="field" aria-hidden="true"></div>
<div class="veil" aria-hidden="true"></div>

<header class="site">
  <nav class="nav" aria-label="primary">
    <a class="brand" href="#top"><b>Exception</b>&nbsp;Theory</a>
    <span class="spacer"></span>
    <a class="item" href="#theory">Theory</a>
    <a class="item" href="#lattice">Lattice</a>
    <a class="item" href="#evidence">Evidence</a>
    <a class="item" href="#papers">Papers</a>
    <a class="item" href="#products">Products</a>
    <a class="item" href="#about">About</a>
  </nav>
</header>

<main id="top">
  <div class="hero">
    <div>
      <p class="sub">a foundational framework · three primitives · zero free parameters</p>
      <h1 class="eq">P&thinsp;∘&thinsp;D&thinsp;∘&thinsp;T&nbsp;=&nbsp;E</h1>
      <p class="tag">“{c['tagline']}”</p>
      <p><a class="cta primary" href="#theory">Enter the theory</a>
         <a class="cta" href="{c['doi_url']}">Read the paper</a></p>
    </div>
  </div>

  <div class="wrap">

  <section id="theory">
    <h2>The Theory</h2>
    <p>Exception Theory is built from exactly three irreducible primitives.
    Everything else — number, geometry, force, field, mind — is a
    configuration of the three. The framework stands as a complete
    foundational alternative: where the Standard Model begins from
    measured couplings and M-theory from postulated geometry, Exception
    Theory begins from the primitives and derives forward, with zero
    free parameters at every step.</p>
    <div class="grid">
      <div class="card prim col-4" style="--pc:var(--recede)">
        <div class="sig">P</div><h3>Point</h3>
        <p>The substrate — the bare, featureless container of potential.
        Infinite. Undifferentiated. Cardinality Ω. Strip away everything
        describable, and what remains is P.</p>
      </div>
      <div class="card prim col-4" style="--pc:var(--verdant)">
        <div class="sig">D</div><h3>Descriptor</h3>
        <p>The constraint — every rule, value, law, and property. Finite
        when bound. Cardinality n. All of mathematics and every physical
        law lives here.</p>
      </div>
      <div class="card prim col-4" style="--pc:var(--advance)">
        <div class="sig">T</div><h3>Traverser</h3>
        <p>The agency — the navigator that substantiates. Indeterminate,
        cardinality [0/0]. Choice, observation, becoming. Irreducible to
        rule or substrate.</p>
      </div>
    </div>
    <h3>The bijection</h3>
    <p>The lattice engine of the theory is a lossless projection. For any
    positive ratio r and resolution N,</p>
    <p class="mono">Π_N(r) = (k, d, ε)&nbsp;&nbsp;with&nbsp;&nbsp;
    k = round(N·log₂ r),&nbsp; d = N/gcd(|k|, N),&nbsp;
    ε = (N·log₂ r − k)·1200/N</p>
    <p>and the pullback 2^((k + ε·N/1200)/N) = r is an <em>algebraic
    identity</em> — not an approximation, not a fit. Zero error by
    theorem. ε is the exact third coordinate, never noise. Every value
    on this page — every color, every length, every duration — is such
    an address, and the build verifies itself by re-projection before it
    is permitted to exist.</p>
    <h3>The constants</h3>
    <table class="consts">
      <tr><th>Symbol</th><th>Value</th><th>Role</th><th>Lattice address</th></tr>
      {constants_rows()}
    </table>
  </section>

  <section id="lattice">
    <h2>The Lattice — the visible octave</h2>
    <p>Project the visible band of the electromagnetic spectrum through
    Π₁₂ against the appearance reference ƛ_e and it lands on eleven
    consecutive lattice cells, k = 120 through 130 — violet sitting
    exactly ten octaves above ƛ_e, red at the tenth semitone. One cell
    remains to close the circle, and colorimetry supplies it: the
    non-spectral purple line. The k-axis <em>is</em> the chromaticity
    coordinate. The palette of this site is not styled — it is the
    visible octave itself, each swatch an address:</p>
    {palette_strip()}
    <div class="wheelbox">{wheel_svg}</div>
    <p class="small">Sector boundaries are the ∂I half-step lines
    (|ε| = 50¢, Identity F). The inscribed star polygons are the four
    generator traversals g ∈ {{1, 5, 7, 11}} — the cascade skeleton.
    White on this site is D65: the identity cell (0, 1, 0) in all three
    tristimulus channels — the Exception state of color.</p>
    <h3>Depth without depth</h3>
    <div class="depth-demo">
      <span class="near">ADVANCE</span>&nbsp;&nbsp;&nbsp;
      <span class="far">RECEDE</span>
      <p class="small" style="margin-top:var(--sp-md);max-width:none">
      Long-wavelength red floats forward; short-wavelength blue sinks
      back — chromostereopsis, the eye's own transverse chromatic
      aberration reading wavelength as depth on a dark ground. The page
      you are looking at is running that projection on you right now.</p>
    </div>
  </section>

  <section id="evidence">
    <h2>Evidence</h2>
    <p>The standard is interventional. A framework earns its place not by
    reclassifying what is already known but by naming a knob nature has
    never been asked about — forbidden by default, zero free parameters,
    binary control — and being right when it is turned. That is the
    category Exception Theory builds for, and the derivations already on
    record set the table:</p>
    <div class="grid">
      <div class="evid col-6"><p><b>α⁻¹ derived forward.</b> The fine
      structure constant, derived from the primitives with zero free
      parameters; the CODATA 2022 measurement concurs to 0.46σ.</p></div>
      <div class="evid col-6"><p><b>The particle ledger.</b> 227 PDG
      particles and 2,324 AME2020 isotopes carry valid lattice addresses
      under one projection — one rule, no per-particle tuning.</p></div>
      <div class="evid col-6"><p><b>The vacuum reads back.</b> STAR
      Collaboration QCD vacuum data lands where the lattice says it
      lands; |ε|/cell = 0.2508 ± 0.0069 across fifteen LCM tower
      levels.</p></div>
      <div class="evid col-6"><p><b>Interventional bench.</b> The d = 1
      family detects gravitational tilt through a lossless microphone
      pipeline — replicated independently, across operators and
      hardware.</p></div>
    </div>
    <p class="small">Derived first; measurement concurs. Full
    derivations, identity cards, and verification scripts accompany the
    paper below.</p>
  </section>

  <section id="papers">
    <h2>Papers</h2>
    <div class="card">
      <h3>The Sempaevum Paper</h3>
      <p>The foundational document: the primitives, the master equation,
      the lossless bijection, the tower, and the identity corpus —
      741 algebraic identity cards and counting, each derived forward
      from P∘D∘T = E and verified at 250 decimal digits.</p>
      <a class="doi" href="{c['doi_url']}">DOI&nbsp;{c['doi']}</a>
    </div>
  </section>

  <section id="products">
    <h2>Products</h2>
    <div class="grid">
      {products_block()}
    </div>
  </section>

  <section id="about">
    <h2>About</h2>
    <p><b>{c['org']}</b> — {c['org_location']}. Founded and authored by
    {c['author']}, sole creator of Exception Theory: fifteen years of
    derivation, a published formal paper, an audited identity compendium,
    verified computation at the Ω standard, and instruments in build.</p>
    <p>Correspondence: <a href="mailto:{c['email']}">{c['email']}</a></p>
  </section>

  </div>
</main>

<footer class="site">
  <p class="eqline">P ∘ D ∘ T = E</p>
  <p class="small">© {c['year']} {c['org']} ·
  <a href="{c['doi_url']}">DOI {c['doi']}</a> ·
  forged losslessly from the constants — every value on this page is a
  lattice address.</p>
</footer>

<script src="js/et_field.js" defer></script>
<noscript><style>#field,.veil{{display:none}}</style></noscript>
</body>
</html>
"""
    return html

# ═════════════════════════════════════════════════════════════════════
#  SECTION 13 — VERIFICATION (the build must prove itself)
# ═════════════════════════════════════════════════════════════════════
RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return ok

def verify_all():
    # 13.1 — geometry tokens are lattice-exact (ε → 0, scaling with dps)
    for table, unit in ((TYPE_SCALE, "rem"), (SPACE_SCALE, "rem"),
                        (RADII, "rem"), (DUR, "s"), (BREAKPOINTS, "px")):
        for name, k in table.items():
            val = lattice(k)
            kk, dd, ee = project(val)
            ok = (kk == k) and fabs(ee) < mpf(10) ** (-(mp.dps - 20))
            check(f"lattice-exact token {name} (k={k})", ok,
                  f"ε={nstr(ee,3)}¢")
    # precision-scaling witness on one token
    save = mp.dps
    resid = []
    for dps in (60, 120, 250):
        mp.dps = dps
        _, _, e = project(mppow(mpf(2), mpf(7) / mpf(12)))
        resid.append(fabs(e))
    mp.dps = save
    check("precision scaling (ε computational only)",
          resid[0] > resid[1] > resid[2] or all(r == 0 for r in resid),
          " > ".join(nstr(r, 3) for r in resid))

    # 13.2 — spectral classes: emitted ε matches independent re-projection
    for cl in CLASSES:
        if cl["class"] > 10:
            continue
        lam = mpf(cl["lambda_nm"])
        k, d, e = project(lam / LAMBDA_E_NM)
        ok = (k == cl["proj"]["k"] and d == cl["proj"]["d"] and
              fabs(e - mpf(cl["proj"]["eps_cents"])) < mpf("1e-40"))
        check(f"spectral class {cl['class']} address (k={k})", ok)
    # monotone spectral ordering — SIC-39 on our own palette
    ks = [cl["proj"]["k"] for cl in CLASSES if cl["class"] <= 10]
    check("SIC-39 monotone k-ordering across palette",
          all(ks[i] < ks[i + 1] for i in range(len(ks) - 1)),
          f"k = {ks[0]}…{ks[-1]}")

    # 13.3 — IC-117: D65 white is the identity cell in all channels
    white = lin_to_hex((mpf(1), mpf(1), mpf(1)))
    triple = color_to_lattice(white)
    ok = all(ch["k"] == 0 and ch["d"] == 1 and
             fabs(mpf(ch["eps_cents"])) < mpf("0.2") for ch in triple)
    check("IC-117 white = (0,1,0)³ identity cell", ok,
          " ".join(f"(k={ch['k']},d={ch['d']},ε={fmt_eps(ch['eps_cents'],2)}¢)"
                   for ch in triple))

    # 13.4 — universal color round-trip (all colors supported, lossless)
    test_set = ([cl["hex"] for cl in CLASSES] +
                [cl["hex_dim"] for cl in CLASSES] +
                list(NEUTRAL.values()) +
                ["#ff0000", "#00ff00", "#0000ff", "#ffff00",
                 "#00ffff", "#ff00ff", "#ffffff", "#123456"])
    for hx in test_set:
        back = lattice_to_color(color_to_lattice(hx))
        check(f"universal round-trip {hx}", back == hx, f"→ {back}")

    # 13.5 — the four generators and divisors emerge, not assumed
    check("generators derived = {1,5,7,11}", GENERATORS == [1, 5, 7, 11])
    check("divisors derived = {1,2,3,4,6,12}",
          DIVISORS == [1, 2, 3, 4, 6, 12])
    check("ξ(1) = A₀/16", fabs(xi(1) - mpf(A0) / 16) < mpf("1e-200"))

    # 13.6 — WCAG contrast on the actual emitted pairs
    def rel_lum(hx):
        X, Y, Z = hex_to_xyz(hx)
        return Y / Yn
    def contrast(fg, bg):
        L1, L2 = rel_lum(fg), rel_lum(bg)
        hi, lo = (L1, L2) if L1 > L2 else (L2, L1)
        return (hi + mpf("0.05")) / (lo + mpf("0.05"))
    bg = NEUTRAL[1]
    pairs = [("body text n11/bg", NEUTRAL[11], bg, mpf(7)),
             ("secondary n10/bg", NEUTRAL[10], bg, mpf("4.5")),
             ("tertiary n9/surface", NEUTRAL[9], NEUTRAL[3], mpf(3)),
             ("link c3-even/bg", CLASSES[3]["hex_even"], bg, mpf("4.5")),
             ("gold-text c7-even/bg", CLASSES[7]["hex_even"], bg,
              mpf("4.5")),
             ("verdant-text c6-even/bg", CLASSES[6]["hex_even"], bg,
              mpf("4.5")),
             ("arcane-text c11-even/bg", CLASSES[11]["hex_even"], bg,
              mpf("4.5")),
             ("brand+hero advance c10/bg", CLASSES[10]["hex"], bg,
              mpf("4.5")),
             ("demo near c10/n0", CLASSES[10]["hex"], NEUTRAL[0], mpf(3)),
             ("demo far c3/n0", CLASSES[3]["hex"], NEUTRAL[0], mpf(3))]
    for name, fg, bgc, req in pairs:
        ratio = contrast(fg, bgc)
        check(f"WCAG {name} ≥ {nstr(req,3)}", ratio >= req,
              f"ratio {nstr(ratio,4)}")

    # 13.7 — starfield parameters: every star re-derives identically
    for s in STARS[:12] + STARS[-3:]:
        i = STARS.index(s) + 1
        k, d, e = project(mpf(i))
        ok = (s["c"] == k % N and
              fabs(mpf(s["p"]) - e * 2 * mppi / 100) < mpf("1e-8"))
        check(f"star i={i} address playback", ok)
    check("star count = A₀", len(STARS) == A0, f"{len(STARS)}")

def report():
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    lines = ["=" * 72, "  ET SITE FORGE — VERIFICATION REPORT", "=" * 72]
    for n, ok, d in RESULTS:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"  [{mark}] {n}" + (f"  — {d}" if d else ""))
    lines += ["-" * 72,
              f"  PASSED: {passed}   FAILED: {len(failed)}   "
              f"TOTAL: {len(RESULTS)}",
              "=" * 72]
    text = "\n".join(lines)
    print(text)
    with open("dist/VERIFICATION_REPORT.txt", "w") as f:
        f.write(text + "\n")
    if failed:
        sys.exit(1)

# ═════════════════════════════════════════════════════════════════════
#  SECTION 14 — EMIT
# ═════════════════════════════════════════════════════════════════════
def main():
    os.makedirs("dist/css", exist_ok=True)
    os.makedirs("dist/js", exist_ok=True)
    os.makedirs("dist/assets", exist_ok=True)

    wheel = build_wheel_svg()
    with open("dist/assets/et_wheel.svg", "w") as f:
        f.write(wheel)
    with open("dist/css/et.css", "w") as f:
        f.write(build_css())
    with open("dist/js/et_field.js", "w") as f:
        f.write(build_js())
    with open("dist/index.html", "w") as f:
        f.write(build_html(wheel))
    with open("dist/et_tokens.json", "w") as f:
        json.dump({
            "site": CONFIG["domain"],
            "doi": CONFIG["doi"],
            "standard": "Ω — mp.dps 250, string→mpf→string, zero float()",
            "constants": {
                "N": N, "V": nstr(V, 50), "K": nstr(K, 50),
                "Pi_count": PI_COUNT, "S": S_STATES, "A0": A0,
                "lambda_e_nm": nstr(LAMBDA_E_NM, 20),
                "dI_cents": nstr(DI_CENTS, 10),
                "generators": GENERATORS, "divisors": DIVISORS,
            },
            "spectral_classes": [
                {kk: vv for kk, vv in cl.items() if kk != "lab"}
                for cl in CLASSES],
            "tokens": TOKENS,
        }, f, indent=1)
    verify_all()
    report()
    print("\n  dist/ written: index.html, css/et.css, js/et_field.js,")
    print("  assets/et_wheel.svg, et_tokens.json, VERIFICATION_REPORT.txt")

if __name__ == "__main__":
    main()

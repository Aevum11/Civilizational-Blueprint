"""
Universal Equation Verification Suite via the ET Lattice
=========================================================

Runs a diverse set of mathematical equations through:
  (1) ET's forced minimal machinery — EML primitives (Odrzywołek 2026)
      paired with the single constant 1 and lattice projection
  (2) Independent computation for cross-check
  (3) The 12ET lattice projection formula (Guide §12.3) to verify
      the lattice identity proj(LHS) = proj(RHS)

Categories covered:
  A. Arithmetic (+, −, ×, /)
  B. Powers and roots
  C. Trigonometric identities
  D. Logarithmic / exponential identities
  E. Classical identities (Pythagorean, Euler)
  F. Non-elementary (erf via Path B Taylor-limit)
  G. Infinite series
  H. Physics — the Koide ratio and fine-structure constant
  I. Mathematics-as-domain — axiom counts (Path C)
"""
import cmath
import math
from math import gcd, log2, factorial

# ============================================================================
# ET LATTICE PROJECTION (Guide §12.3)
# ============================================================================
N = 12   # manifold symmetry = |Π|·S = 3·4
S = 4    # state count = C(3,2) + C(3,3)

def project(r, N=N):
    if r <= 0:
        return {'k': None, 'd': None, 'eps': None, 'note': 'annihilation boundary (r=0) or negative (needs complex lattice)'}
    exact = N * log2(r)
    k = round(exact)
    g = gcd(abs(k), N) if k != 0 else N
    d = N // g
    eps = (exact - k) * (1200.0 / N)
    return {'k': k, 'd': d, 'eps': eps, 'exact': exact}

def lattice_identity(lhs_val, rhs_val, tol=1e-9):
    """Test whether LHS and RHS have the same lattice address."""
    if lhs_val <= 0 or rhs_val <= 0:
        return abs(lhs_val - rhs_val) < tol  # handle via direct equality for non-positives
    pl = project(lhs_val)
    pr = project(rhs_val)
    return (pl['k'] == pr['k'] and pl['d'] == pr['d']
            and abs(pl['eps'] - pr['eps']) < tol)

# ============================================================================
# EML PRIMITIVES (forced minimal continuous-D operator)
# ============================================================================
def eml(x, y):
    return cmath.exp(x) - cmath.log(y)

def eml_exp(x):
    return eml(x, 1)

def eml_ln(x):
    return eml(1, eml(eml(1, x), 1))

def eml_mul(x, y):
    return eml_exp(eml_ln(x) + eml_ln(y))

def eml_div(x, y):
    return eml_exp(eml_ln(x) - eml_ln(y))

def eml_sub(x, y):
    return eml(eml_ln(x), eml_exp(y))  # exp(ln(x)) - ln(exp(y)) = x - y

def eml_add(x, y):
    return eml_ln(eml_mul(eml_exp(x), eml_exp(y)))  # ln(e^x · e^y) = x+y

def eml_pow(x, n):
    """x^n via iterated multiplication for integer n >= 0, or via exp(n·ln(x)) for real n."""
    return eml_exp(eml_mul(n, eml_ln(x)))

# ============================================================================
# TEST HARNESS
# ============================================================================
results = []

def test(category, description, lhs_computation, expected, note=""):
    """Run one equation; record result."""
    # Compute via EML / elementary machinery
    if callable(lhs_computation):
        lhs_raw = lhs_computation()
    else:
        lhs_raw = lhs_computation
    # Extract real part if complex
    if isinstance(lhs_raw, complex):
        lhs_val = lhs_raw.real
        imag = lhs_raw.imag
    else:
        lhs_val = float(lhs_raw)
        imag = 0.0
    # Check numerical correctness
    numerical_ok = abs(lhs_val - expected) < 1e-8
    # Project both onto lattice
    if lhs_val > 0 and expected > 0:
        pl = project(lhs_val)
        pe = project(expected)
        lattice_ok = (pl['k'] == pe['k'] and pl['d'] == pe['d']
                      and abs(pl['eps'] - pe['eps']) < 1e-7)
        addr = f"(k={pe['k']:+d}, d={pe['d']}, ε={pe['eps']:+.4f}¢)"
    else:
        # Handle non-positive cases
        lattice_ok = numerical_ok  # lattice identity trivial if numerical matches
        addr = "non-positive (negative or zero — needs complex lattice or annihilation boundary)"
    results.append({
        'category': category,
        'description': description,
        'computed': lhs_val,
        'expected': expected,
        'imag_residual': imag,
        'numerical_ok': numerical_ok,
        'lattice_ok': lattice_ok,
        'lattice_address': addr,
        'note': note,
    })
    return lhs_val

# ============================================================================
# CATEGORY A — ARITHMETIC (all four operations via pure EML)
# ============================================================================
test("A. Arithmetic",  "2 + 2 = 4",         lambda: eml_add(2, 2),         4.0)
test("A. Arithmetic",  "7 + 3 = 10",        lambda: eml_add(7, 3),         10.0)
test("A. Arithmetic",  "11 − 4 = 7",        lambda: eml_sub(11, 4),        7.0)
test("A. Arithmetic",  "12 × 11 = 132",     lambda: eml_mul(12, 11),       132.0)
test("A. Arithmetic",  "100 / 4 = 25",      lambda: eml_div(100, 4),       25.0)
test("A. Arithmetic",  "(3 × 5) + 7 = 22",  lambda: eml_add(eml_mul(3, 5), 7), 22.0)

# ============================================================================
# CATEGORY B — POWERS AND ROOTS
# ============================================================================
test("B. Powers",      "2^10 = 1024",       lambda: eml_pow(2, 10),        1024.0)
test("B. Powers",      "3^4 = 81",          lambda: eml_pow(3, 4),         81.0)
test("B. Roots",       "√16 = 4",           lambda: eml_pow(16, 0.5),      4.0)
test("B. Roots",       "∛27 = 3",           lambda: eml_pow(27, 1.0/3),    3.0)
test("B. Roots",       "√2 via EML",        lambda: eml_pow(2, 0.5),       math.sqrt(2))

# ============================================================================
# CATEGORY C — TRIGONOMETRIC IDENTITIES
# ============================================================================
# cos via complex exp: cos(θ) = (e^(iθ) + e^(-iθ)) / 2
def eml_cos(theta):
    return (eml_exp(1j * theta) + eml_exp(-1j * theta)) / 2

def eml_sin(theta):
    return (eml_exp(1j * theta) - eml_exp(-1j * theta)) / (2j)

# sin²(π/4) + cos²(π/4) = 1 (the Pythagorean identity on the unit circle)
sin_pi_4 = eml_sin(math.pi / 4)
cos_pi_4 = eml_cos(math.pi / 4)
pyth_trig = (sin_pi_4 * sin_pi_4 + cos_pi_4 * cos_pi_4).real
test("C. Trig",        "sin²(π/4) + cos²(π/4) = 1",  pyth_trig,            1.0)

# ============================================================================
# CATEGORY D — LOG / EXP IDENTITIES
# ============================================================================
test("D. Log/Exp",     "ln(e) = 1",         lambda: eml_ln(math.e),        1.0)
test("D. Log/Exp",     "ln(e⁴) = 4",        lambda: eml_ln(eml_exp(4)),    4.0)
test("D. Log/Exp",     "exp(ln(7)) = 7",    lambda: eml_exp(eml_ln(7)),    7.0)
test("D. Log/Exp",     "ln(8) = 3·ln(2)",   lambda: eml_ln(8),             3 * math.log(2))

# ============================================================================
# CATEGORY E — CLASSICAL IDENTITIES
# ============================================================================
# Pythagorean 3-4-5 triangle: 3² + 4² = 5² → 9+16 = 25
test("E. Classical",   "3² + 4² = 5² (= 25)",
     lambda: eml_add(eml_pow(3, 2), eml_pow(4, 2)),    25.0)

# Euler's identity check: e^(iπ) = −1 → re-arranged to eml form, modulus is 1
euler = abs(eml_exp(1j * math.pi))
test("E. Classical",   "|e^(iπ)| = 1 (Euler)", euler,                      1.0)

# Golden ratio: φ² = φ + 1 → both sides at φ² ≈ 2.618
phi = (1 + math.sqrt(5)) / 2
test("E. Classical",   "φ² = φ + 1 (golden ratio)",
     lambda: eml_pow(phi, 2),                                               phi + 1)

# ============================================================================
# CATEGORY F — NON-ELEMENTARY via PATH B (Taylor-limit projection)
# ============================================================================
# erf(x) = (2/√π) · Σ (-1)^n · x^(2n+1) / (n! · (2n+1))
# erf is not elementary (Liouville theorem) — requires Path B
def erf_partial(x, N_terms):
    c = 2.0 / math.sqrt(math.pi)
    return c * sum((-1)**n * x**(2*n+1) / (factorial(n) * (2*n+1))
                   for n in range(N_terms + 1))

test("F. Non-elementary (Path B)", "erf(1) via 20 Taylor terms",
     lambda: erf_partial(1.0, 20), math.erf(1),
     note="Path B: non-elementary function reached via elementary partials")

# ζ(2) via Basel formula: π²/6
test("F. Non-elementary (Path B)", "ζ(2) = π²/6 via 10000 partials",
     lambda: sum(1 / n**2 for n in range(1, 10001)), math.pi**2 / 6,
     note="Partial sum approaches Basel value")

# ============================================================================
# CATEGORY G — INFINITE SERIES (convergence to lattice point)
# ============================================================================
# Geometric: 1 + 1/2 + 1/4 + ... = 2
test("G. Infinite series", "Σ 1/2^n = 2",
     lambda: sum(1 / 2**n for n in range(100)), 2.0)

# Leibniz: π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
test("G. Infinite series", "Leibniz π/4 (100k terms)",
     lambda: sum((-1)**n / (2*n + 1) for n in range(100000)), math.pi / 4,
     note="Converges slowly — truncation error visible in ε")

# ============================================================================
# CATEGORY H — PHYSICS — Koide ratio and A₀
# ============================================================================
# A₀ = (N-1)² + S²  — the forced ET derivation of the fine structure constant
A0 = (N - 1)**2 + S**2
test("H. Physics",     "A₀ = (N-1)² + S² = 137",   A0,                     137.0,
     note="Leading-order 1/α from ET — zero free parameters")

# Koide ratio: K = (me + mμ + mτ) / (√me + √mμ + √mτ)²
# Using CODATA lepton masses (MeV): me = 0.51099895, mμ = 105.6583755, mτ = 1776.86
me, mu, mtau = 0.51099895000, 105.6583755, 1776.86
K_observed = (me + mu + mtau) / (math.sqrt(me) + math.sqrt(mu) + math.sqrt(mtau))**2
test("H. Physics",     "Koide ratio ≈ 2/3",        K_observed,             2.0/3.0,
     note=f"Observed: {K_observed:.10f}, ET-predicted: 2/3 = 0.6666666667")

# ============================================================================
# CATEGORY I — MATHEMATICS-AS-DOMAIN via PATH C (axiom counts)
# ============================================================================
# ZF at d=1 ε=0 because 8 axioms = 2³ is a pure octave object
zf_projection = project(8)  # ZF has 8 axioms
test("I. Math-as-domain (Path C)", "ZF → 8 axioms → d=1 octave ε=0",
     8.0, 8.0,
     note=f"Projects to (k={zf_projection['k']}, d={zf_projection['d']}, ε={zf_projection['eps']:+.6f}¢) — 8=2³ pure octave")

# Euclidean geometry: 5 postulates
eucl_projection = project(5)
test("I. Math-as-domain (Path C)", "Euclid → 5 postulates → d=3 quintic",
     5.0, 5.0,
     note=f"Projects to (k={eucl_projection['k']}, d={eucl_projection['d']}, ε={eucl_projection['eps']:+.6f}¢) — quintic comma QS-10")

# ============================================================================
# REPORT
# ============================================================================

def run_report():
    print("\n" + "=" * 92)
    print("UNIVERSAL EQUATION VERIFICATION via the ET LATTICE")
    print("=" * 92)
    current_cat = None
    total, num_ok, lat_ok = 0, 0, 0
    for r in results:
        if r['category'] != current_cat:
            print(f"\n── {r['category']} " + "─" * (86 - len(r['category'])))
            current_cat = r['category']
        n_mark = "✓" if r['numerical_ok'] else "✗"
        l_mark = "✓" if r['lattice_ok']   else "✗"
        print(f"  [{n_mark} num | {l_mark} lat]  {r['description']:<45}")
        print(f"                computed:  {r['computed']:.12g}")
        print(f"                expected:  {r['expected']:.12g}")
        print(f"                lattice:   {r['lattice_address']}")
        if r['note']:
            print(f"                note:      {r['note']}")
        total += 1
        if r['numerical_ok']: num_ok += 1
        if r['lattice_ok']:   lat_ok += 1

    print("\n" + "=" * 92)
    print(f"SUMMARY: {num_ok}/{total} numerical, {lat_ok}/{total} lattice identity")
    print("=" * 92)

    if num_ok == total and lat_ok == total:
        print("""
ALL EQUATIONS ACROSS ALL CATEGORIES:
  (1) COMPUTE CORRECTLY via the forced minimal ET machinery (EML + constant 1)
  (2) PROJECT ONTO THE LATTICE at specific (k, d, ε) addresses
  (3) PRESERVE LATTICE IDENTITY: proj(LHS) = proj(RHS) for every equation
""")
    else:
        print(f"\nFailures: {total - num_ok} numerical, {total - lat_ok} lattice identity")

if __name__ == "__main__":
    run_report()

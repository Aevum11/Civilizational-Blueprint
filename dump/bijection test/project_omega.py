"""
Chaitin's Omega — Full Sempaevum Lattice Projection
====================================================

Projects Chaitin's Omega_U onto the Sempaevum lattice through the complete
LCM tower escalation AND continued fraction home-finding. Finds Ω's home.

Source value: Calude, Dinneen, Shu (2002) "Computing a Glimpse of Randomness"
  Experimental Mathematics 11(3), 361-370.
  First 64 exact bits of Omega_U for their universal Chaitin machine.

All arithmetic: mpmath at 2000-bit precision (~600 dps).
Projection formula: Sempaevum Paper §6, Theorem 6.1.
Home-finding: TWO methods —
  1. LCM tower: d-family stabilization across ceil(1/K) = 2 consecutive
     landmarks. FAILS for algorithmically random values (d never stabilizes).
  2. Continued fraction: identifies the dominant convergent of |log₂(Ω)|.
     The convergent with the largest following partial quotient a_{n+1}
     identifies the home d-family. SUCCEEDS for all values.

Result: Ω's home is d = 87 = 3 × 29, ε = +0.001003 cents.
  CF convergent n=3 (608/87), quality a₄ = 157.
  Sub-Koide by a factor of ~1955.

Author: Exception Theory projection — Michael James Muller (Aevum Defluo)
Tools: Identification Principle, Descriptor Gap Principle, Subsumption Law
"""

from mpmath import mp, mpf, log, power, nint, fabs, floor
from math import gcd

# ============================================================
# Precision: 400 bits = 120+ decimal places. Uniform. No floats.
# ============================================================
mp.prec = 2000  # ~600 dps — needed for extended LCM tower (N up to lcm(1..97) ≈ 10^40)

# ============================================================
# ET Constants — all exact, all derived from {P, D, T}
# ============================================================
N_BASE = 12                    # Manifold symmetry — forced (Sempaevum §5)
K_NUM, K_DEN = 2, 3            # Koide ratio K = 2/3
STABILITY_DEPTH = 2            # ceil(1/K) = ceil(3/2) = 2
EPS_KOIDE_MICROCENTS = 1955    # |ε_Koide| = 1.955 cents = 1955 micro-cents
EPS_DI_MICROCENTS = 50000      # ∂I boundary = 50 cents = 50000 micro-cents
BASE_VARIANCE_NUM = 1          # V = 1/12
BASE_VARIANCE_DEN = 12
S = 4                          # State count = C(3,2) + C(3,3)

# ============================================================
# Step 1: Reconstruct Omega from the exact 64 bits
# ============================================================

# The 64 exact bits from Calude-Dinneen-Shu 2002 (Experimental Mathematics)
# Universal Chaitin (self-delimiting Turing) machine U
# These are PROVEN exact — not approximations, not lower bounds of unknown accuracy
OMEGA_BITS = "0000001000000100000110001000011010001111110010111011101000010000"

omega = mpf(0)
for i, bit in enumerate(OMEGA_BITS):
    if bit == '1':
        omega += power(2, -(i + 1))

log2_omega = log(omega, 2)

# ============================================================
# Utility functions
# ============================================================

def my_lcm(a, b):
    """LCM of two integers."""
    return a * b // gcd(a, b)


def factorize(n):
    """Full prime factorization. No shortcuts."""
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n < 0:
        return "-1·" + factorize(-n)
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        if e > 0:
            factors.append(f"{p}^{e}" if e > 1 else str(p))
        p += 1 if p == 2 else 2  # 2, 3, 5, 7, 9, 11, ...
    if temp > 1:
        factors.append(str(temp))
    return "·".join(factors) if factors else "1"


def divisor_count(n):
    """τ(n) = number of divisors, via prime factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        if e > 0:
            result *= (e + 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= 2
    return result


def project(r, n_res):
    """
    Sempaevum projection formula (Theorem 6.1).

    Input:  r ∈ ℝ⁺, n_res ∈ ℤ⁺ (lattice resolution N)
    Output: (k, d, ε_cents, ε_microcents)

    k = round(n_res · log₂(r))
    d = n_res / gcd(|k|, n_res)
    ε = (n_res · log₂(r) − k) · 1200 / n_res   [cents]

    All computation via mpmath at 400-bit precision.
    The only conversion to int is for k (which IS an integer by definition)
    and eps_microcents (for exact integer threshold comparison).
    """
    log2_r = log(r, 2)
    n_log2_r = mpf(n_res) * log2_r
    k = int(nint(n_log2_r))

    abs_k = abs(k)
    g = gcd(abs_k, n_res) if abs_k > 0 else n_res
    d = n_res // g

    eps_cents = (n_log2_r - mpf(k)) * mpf(1200) / mpf(n_res)
    eps_microcents = int(nint(eps_cents * mpf(1000)))

    return k, d, eps_cents, eps_microcents


def classify_eps(eps_microcents):
    """Classify ε by ET thresholds. All integer arithmetic."""
    a = abs(eps_microcents)
    if eps_microcents == 0:
        return "EXACT"
    elif a <= EPS_KOIDE_MICROCENTS:
        return "sub-Koide"
    elif a < EPS_DI_MICROCENTS:
        return "inside"
    else:
        return "near ∂I"


def impedance(d):
    """A₀_magic(d) = (d−1)² + S² where S=4."""
    return (d - 1) ** 2 + S ** 2


def coupling(d):
    """ξ(d) = 137 / A₀_magic(d). Returns as exact rational string."""
    a0 = impedance(d)
    return f"137/{a0}"


# ============================================================
# Step 2: Generate the CORRECT LCM tower
# ============================================================

def generate_lcm_tower(max_k=50):
    """
    Generate the canonical LCM landmarks: lcm(1, 2, ..., k) for k = 1, 2, 3, ...

    CRITICAL: Always accumulate the lcm. Only YIELD when >= 12 AND when the
    lcm has changed from the last yielded value. But ALWAYS update the running lcm.
    """
    running_lcm = 1
    last_yielded = 0

    for k in range(1, max_k + 1):
        running_lcm = my_lcm(running_lcm, k)  # ALWAYS accumulate

        if running_lcm >= 12 and running_lcm != last_yielded:
            last_yielded = running_lcm
            yield running_lcm, k


# ============================================================
# Step 3: Generate multiplicative refinements between landmarks
# ============================================================

def generate_multiplicative_refinements(lcm_landmarks, max_mult=27720):
    """
    Between each pair of LCM landmarks, generate the multiplicative
    refinements (multiples of N_BASE) that fall between them.
    These refine ε within the same family structure.

    max_mult defaults to 27720ET = lcm(1..11), the base lattice's full
    resolution. Multiplicative refinements above this level are redundant
    with LCM landmarks.
    """
    refinements = set()
    for n_landmark, _ in lcm_landmarks:
        refinements.add(n_landmark)

    # Add all multiples of N_BASE up to the lesser of the largest landmark and max_mult
    if lcm_landmarks:
        upper = min(lcm_landmarks[-1][0], max_mult)
        mult = N_BASE
        while mult <= upper:
            refinements.add(mult)
            mult += N_BASE

    return sorted(refinements)


# ============================================================
# Main projection
# ============================================================

def main():
    """
    Project Chaitin's Omega onto the Sempaevum lattice via the full LCM tower
    escalation. Prints the complete trajectory, home-finding result, sub-Koide
    hits, and structural analysis.
    """
    print("=" * 100)
    print("CHAITIN'S Ω — FULL SEMPAEVUM LATTICE PROJECTION")
    print("Calude-Dinneen-Shu Universal Machine (2002)")
    print("=" * 100)
    print()
    print(f"Binary (64 exact bits):  0.{OMEGA_BITS}")
    print(f"Decimal (full precision): {mp.nstr(omega, 30)}")
    print(f"log₂(Ω) = {mp.nstr(log2_omega, 35)}")
    print()
    print(f"Manifold state: {{P,D}} Unsubstantiated")
    print(f"  P = substrate of all possible computations (cardinality Ω)")
    print(f"  D = 'halting probability of universal prefix-free Turing machine U' (finite descriptor)")
    print(f"  T = ABSENT — no finite computational process can substantiate further bits")
    print(f"  The uncomputability IS the absence of T.")
    print()

    # Generate LCM tower
    lcm_landmarks = list(generate_lcm_tower(max_k=100))

    print("=" * 100)
    print("CANONICAL LCM TOWER VERIFICATION")
    print("=" * 100)
    print()
    for n_res, k_level in lcm_landmarks:
        print(f"  lcm(1..{k_level:>2}) = {n_res:>15}   τ = {divisor_count(n_res):>6}   factorization: {factorize(n_res)}")
    print()

    # Generate all resolutions to check (LCM + multiplicative refinements)
    all_resolutions = []
    for n_res, k_level in lcm_landmarks:
        all_resolutions.append((n_res, k_level, True))  # True = LCM landmark

    # Add multiplicative refinements between base landmarks (up to 27720ET)
    for mult in range(24, 27721, 12):
        if mult not in [n for n, _, _ in all_resolutions]:
            all_resolutions.append((mult, 0, False))  # False = multiplicative refinement

    all_resolutions.sort(key=lambda x: x[0])

    # ============================================================
    # Project at every resolution
    # ============================================================

    print("=" * 100)
    print("FULL TOWER ESCALATION — LCM LANDMARKS (★) AND MULTIPLICATIVE REFINEMENTS")
    print("=" * 100)
    print()

    header = (f"{'':>3} {'N':>12} {'Source':>14} {'τ(N)':>6} "
              f"{'k':>12} {'d':>10} {'d factors':>24} "
              f"{'ε (cents)':>20} {'|ε|μ¢':>10} {'Class':>10} {'Stab':>5}")
    print(header)
    print("-" * len(header) + "-" * 20)

    # Home-finding state
    last_lcm_d = None
    consec_lcm = 0
    home_found = False
    home_d = None
    home_resolution = None
    home_class = None
    verify_rem = 0
    false_resolutions = []
    sub_koide_hits = []

    for n_res, k_level, is_lcm in all_resolutions:
        k, d, eps, eps_micro = project(omega, n_res)
        tau_n = divisor_count(n_res)
        abs_micro = abs(eps_micro)
        cls = classify_eps(eps_micro)

        # Track sub-Koide hits
        if cls == "sub-Koide":
            sub_koide_hits.append((n_res, k, d, eps, is_lcm))

        # d-family stability (LCM landmarks only — these introduce new primes)
        stab_str = ""
        marker = ""

        if is_lcm:
            if last_lcm_d is not None:
                if d == last_lcm_d:
                    consec_lcm += 1
                else:
                    # d changed at LCM landmark
                    if home_found and verify_rem > 0:
                        false_resolutions.append({
                            'stable_resolution': home_resolution,
                            'stable_d': home_d,
                            'break_resolution': n_res,
                            'break_d': d,
                            'break_k_level': k_level
                        })
                        home_found = False
                        home_class = None
                        home_d = None
                    consec_lcm = 1
            else:
                consec_lcm = 1
            last_lcm_d = d

            # Check stabilization
            if consec_lcm >= STABILITY_DEPTH and not home_found:
                home_found = True
                home_d = d
                home_resolution = n_res
                verify_rem = STABILITY_DEPTH
                if eps_micro == 0:
                    home_class = "true_home"
                elif abs_micro <= EPS_KOIDE_MICROCENTS:
                    home_class = "deep_home"
                else:
                    home_class = "persistent_home"

            if home_found and verify_rem > 0:
                verify_rem -= 1

            stab_str = f"{consec_lcm}"

            if home_found and n_res == home_resolution:
                marker = " ◄ HOME?"
            elif home_found and verify_rem == 0 and n_res > home_resolution:
                marker = " ✓ VERIFIED"

        # Format source
        if is_lcm:
            source = f"lcm(1..{k_level})"
        else:
            source = f"{n_res // N_BASE}×{N_BASE}"

        # Format d factors (truncate if too long)
        d_fac = factorize(d)
        if len(d_fac) > 24:
            d_fac = d_fac[:21] + "..."

        lcm_mark = "★" if is_lcm else " "

        print(f"{lcm_mark}{' ':>2} {n_res:>12} {source:>14} {tau_n:>6} "
              f"{k:>12} {d:>10} {d_fac:>24} "
              f"{mp.nstr(eps, 12):>20} {abs_micro:>10} {cls:>10} "
              f"{stab_str:>5}{marker}")

    # ============================================================
    # Results
    # ============================================================

    print()
    print("=" * 100)
    print("HOME-FINDING RESULT")
    print("=" * 100)
    print()

    if false_resolutions:
        print(f"FALSE RESOLUTIONS: {len(false_resolutions)}")
        print()
        for fr_idx, fr in enumerate(false_resolutions, 1):
            print(f"  #{fr_idx}: d = {fr['stable_d']} appeared stable at N = {fr['stable_resolution']}")
            print(f"      but d changed to {fr['break_d']} at N = {fr['break_resolution']} = lcm(1..{fr['break_k_level']})")
            print(f"      Stable d factors: {factorize(fr['stable_d'])}")
            print(f"      New d factors:    {factorize(fr['break_d'])}")
            print()

    if home_found and verify_rem == 0:
        k_h, d_h, eps_h, eps_micro_h = project(omega, home_resolution)
        print(f"HOME FOUND:")
        print(f"  d-family:        d = {home_d}")
        print(f"  d factorization: {factorize(home_d)}")
        print(f"  Classification:  {home_class}")
        print(f"  Stable at:       N = {home_resolution} = {factorize(home_resolution)}")
        print(f"  k = {k_h}")
        print(f"  ε = {mp.nstr(eps_h, 20)} cents")
        print(f"  |ε| = {abs(eps_micro_h)} micro-cents")
        print(f"  A₀_magic = {impedance(home_d)}")
        print(f"  ξ = {coupling(home_d)}")
    else:
        print(f"LCM TOWER: d has not stabilized through {len([r for r in all_resolutions if r[2]])} landmarks")
        print(f"  Last d at LCM landmark: {last_lcm_d}")
        print(f"  Last d factorization:   {factorize(last_lcm_d)}")
        print(f"  Consecutive same d:     {consec_lcm}")
        print()

        # Analyze the d-family trajectory at LCM landmarks
        print("d-FAMILY TRAJECTORY AT LCM LANDMARKS:")
        print()
        for n_res, k_level, is_lcm in all_resolutions:
            if is_lcm:
                k, d, eps, eps_micro = project(omega, n_res)
                d_fac = factorize(d)
                print(f"  N = {n_res:>15} = lcm(1..{k_level:>2})  →  d = {d:>15}  = {d_fac}")

    # Sub-Koide analysis
    if sub_koide_hits:
        print()
        print("SUB-KOIDE HITS (|ε| ≤ 1.955¢):")
        print()
        for N_sk, k_sk, d_sk, eps_sk, is_lcm_sk in sub_koide_hits:
            mark = "★ LCM" if is_lcm_sk else "  mult"
            print(f"  {mark} N={N_sk:>10}: k={k_sk:>10}, d={d_sk:>10} = {factorize(d_sk):>24}, ε={mp.nstr(eps_sk, 10)} cents")

    # Structural analysis
    print()
    print("=" * 100)
    print("STRUCTURAL ANALYSIS")
    print("=" * 100)
    print()

    # Base resolution analysis
    k12, d12, eps12, eps_micro12 = project(omega, 12)
    print(f"BASE RESOLUTION (N=12):")
    print(f"  k = {k12}")
    print(f"  d = {d12}")
    print(f"  ε = {mp.nstr(eps12, 15)} cents")
    print(f"  |ε| = {abs(eps_micro12)} micro-cents")
    print(f"  Classification: {classify_eps(eps_micro12)}")
    print(f"  k = {k12} = {factorize(abs(k12))} × sign({'+' if k12 > 0 else '-'})")
    print(f"  gcd(|k|, N) = gcd({abs(k12)}, {12}) = {gcd(abs(k12), 12)}")
    print(f"  d = N/gcd = {12}/{gcd(abs(k12), 12)} = {d12}")

    eps_abs = fabs(eps12)
    tightness = mpf(100) / (mpf(100) + eps_abs)
    koide_ratio = mpf(K_NUM) / mpf(K_DEN)
    print(f"  Tightness t = 100/(100+|ε|) = {mp.nstr(tightness, 12)}")
    print(f"  K = 2/3 = {mp.nstr(koide_ratio, 12)}")
    print(f"  t vs K: {'t > K → inside lattice' if tightness > koide_ratio else 't ≤ K → near ∂I'}")

    print()
    print(f"MANIFOLD STATE: {{P,D}} Unsubstantiated")
    print(f"  Chaitin's Ω is the halting probability of a universal prefix-free Turing machine.")
    print(f"  It is algorithmically random — its binary expansion is an algorithmic random sequence.")
    print(f"  It is non-computable — no algorithm can produce all its bits.")
    print(f"  The descriptor 'halting probability of U' is complete and finite (D is present).")
    print(f"  The value r = Ω is determined (P is present).")
    print(f"  But no T can extend the known bits — T is absent → {{P,D}} Unsubstantiated.")
    print()
    print(f"  The gap between the 64 known bits and the unknowable remainder is the")
    print(f"  Descriptor Gap at the value level: gap(Ω) = D_missing = 'the remaining bits'.")
    print(f"  The Descriptor Gap Principle says: this gap IS a Descriptor.")
    print(f"  And it is — it is the Descriptor 'algorithmically random', which describes")
    print(f"  precisely what cannot be further described.")

    # ============================================================
    # Step 5: Continued Fraction of |log₂(Ω)| — Home-Finding
    # ============================================================
    #
    # The continued fraction expansion finds Ω's HOME on the Sempaevum.
    # Each convergent p_n/q_n is a candidate home: at N = q_n, the
    # projection ε is minimized among all denominators ≤ q_n.
    # The dominant convergent (largest following partial quotient a_{n+1})
    # identifies the actual home d-family.

    print()
    print("=" * 100)
    print("CONTINUED FRACTION OF |log₂(Ω)| — HOME-FINDING")
    print("The dominant convergent (largest a_{n+1}) identifies the home d-family")
    print("=" * 100)
    print()

    abs_log2 = -log2_omega  # positive

    # Compute CF convergents via standard recurrence
    # h_{-1}=1, h_0=a_0; k_{-1}=0, k_0=1; h_n=a_n·h_{n-1}+h_{n-2}
    MAX_CF_TERMS = 50
    cf_h = [mpf(0)] * (MAX_CF_TERMS + 2)
    cf_k = [mpf(0)] * (MAX_CF_TERMS + 2)
    cf_h[0] = mpf(1)  # h_{-1}
    cf_k[0] = mpf(0)  # k_{-1}

    cf_terms = []
    cf_convergents = []
    cf_remainder = abs_log2

    for cf_i in range(MAX_CF_TERMS):
        cf_a = int(floor(cf_remainder))
        cf_terms.append(cf_a)
        cf_idx = cf_i + 1  # offset for h_{-1} slot

        if cf_i == 0:
            cf_h[cf_idx] = mpf(cf_a)
            cf_k[cf_idx] = mpf(1)
        else:
            cf_h[cf_idx] = mpf(cf_a) * cf_h[cf_idx - 1] + cf_h[cf_idx - 2]
            cf_k[cf_idx] = mpf(cf_a) * cf_k[cf_idx - 1] + cf_k[cf_idx - 2]

        cf_p = int(cf_h[cf_idx])
        cf_q = int(cf_k[cf_idx])

        cf_residual_exact = abs_log2 * mpf(cf_q) - mpf(cf_p)
        cf_eps = float(cf_residual_exact * mpf(1200) / mpf(cf_q)) if cf_q > 0 else 0.0
        cf_gcd = gcd(abs(cf_p), cf_q)
        cf_d = cf_q // cf_gcd if cf_gcd > 0 else 0

        cf_convergents.append({
            'n': cf_i, 'a': cf_a, 'p': cf_p, 'q': cf_q,
            'd': cf_d, 'g': cf_gcd, 'eps': cf_eps,
            'residual': float(fabs(cf_residual_exact))
        })

        cf_frac = cf_remainder - mpf(cf_a)
        if cf_frac < mpf(10) ** (-500):
            break
        cf_remainder = mpf(1) / cf_frac

    # Display CF
    cf_display = f"[{cf_terms[0]}; " + ", ".join(str(a) for a in cf_terms[1:30]) + ", ...]"
    print(f"|log₂(Ω)| = {cf_display}")
    print()

    # Convergents table
    cf_header = (f"{'n':>3} {'a_n':>8} {'q_n (=N)':>20} {'p_n (≈|k|)':>20} "
                 f"{'d=q/gcd':>15} {'gcd(p,q)':>10} {'ε (cents)':>22} {'|residual|':>14}")
    print(cf_header)
    print("-" * len(cf_header))

    for c in cf_convergents[:30]:
        print(f"{c['n']:>3} {c['a']:>8} {c['q']:>20} {c['p']:>20} "
              f"{c['d']:>15} {c['g']:>10} {c['eps']:>22.15e} {c['residual']:>14.6e}")

    print()

    # Key resonances with quality assessment
    print("CONVERGENT HIERARCHY — HOME IDENTIFICATION (ε < 50¢, quality = a_{n+1}):")
    print("The convergent with the largest a_{n+1} is the HOME.")
    print()
    # Find the dominant convergent (largest a_{n+1}) for HOME marking
    max_quality_idx = -1
    max_quality_val = 0
    for c in cf_convergents[:20]:
        if c['q'] > 0 and abs(c['eps']) < 50:
            next_a = cf_terms[c['n'] + 1] if c['n'] + 1 < len(cf_terms) else 0
            if isinstance(next_a, int) and next_a > max_quality_val:
                max_quality_val = next_a
                max_quality_idx = c['n']

    for c in cf_convergents[:20]:
        if c['q'] > 0 and abs(c['eps']) < 50:
            next_a = cf_terms[c['n'] + 1] if c['n'] + 1 < len(cf_terms) else "?"
            quality_str = f"a_{c['n']+1} = {next_a}" if isinstance(next_a, int) else ""
            native = "divisor of 12" if c['d'] > 0 and 12 % c['d'] == 0 else "SHADOW"
            home_mark = " ◄◄◄ HOME" if c['n'] == max_quality_idx else ""
            print(f"  n={c['n']:>2}: N={c['q']}, d={c['d']} = {factorize(c['d'])}, "
                  f"ε={c['eps']:.15f}¢, quality: {quality_str}, family: {native}{home_mark}")

    print()

    # ============================================================
    # Step 6: Home Analysis — dynamically computed from CF
    # ============================================================
    #
    # The dominant convergent (largest a_{n+1}) identifies the home.
    # All values below are computed from max_quality_idx, not hardcoded.

    home_idx = max_quality_idx
    home_conv = cf_convergents[home_idx]
    home_d = home_conv['d']
    home_q = home_conv['q']
    home_p = home_conv['p']
    home_quality = cf_terms[home_idx + 1] if home_idx + 1 < len(cf_terms) else "?"
    next_q = cf_convergents[home_idx + 1]['q'] if home_idx + 1 < len(cf_convergents) else "?"

    print("=" * 100)
    print(f"d = {home_d} = {factorize(home_d)} — HOME ANALYSIS")
    print("=" * 100)
    print()

    k_home, d_home, eps_home, eps_micro_home = project(omega, home_q)
    print(f"Projection at N = {home_q}:")
    print(f"  k = {k_home}")
    print(f"  d = {d_home}")
    print(f"  ε = {mp.nstr(eps_home, 20)} cents ({abs(eps_micro_home)} micro-cents)")
    print(f"  gcd(|k|, N) = gcd({abs(k_home)}, {home_q}) = {gcd(abs(k_home), home_q)}")
    print(f"  Classification: {classify_eps(eps_micro_home)}")
    print()

    # CF identification
    print("CONTINUED FRACTION IDENTIFICATION:")
    print(f"  |log₂(Ω)| has CF = [{cf_terms[0]}; {', '.join(str(a) for a in cf_terms[1:home_idx+3])}, ...]")
    print(f"  Convergent n={home_idx}: p/q = {home_p}/{home_q}")
    print(f"  gcd({home_p}, {home_q}) = {home_conv['g']}")
    print(f"  d = q/gcd(p,q) = {home_q}/{home_conv['g']} = {home_d}")
    print()
    print(f"  a_{home_idx+1} = {home_quality} — THIS IS THE HOME QUALITY")
    print(f"  No denominator between {home_q} and ~{next_q} approximates |log₂(Ω)| better")
    print(f"  The home quality is proportional to a_{home_idx+1}: larger = tighter lock on the d-family")
    print()

    # Exact product
    exact_product = abs_log2 * mpf(home_q)
    deficit = mpf(home_p) - exact_product
    print(f"  {home_q} × |log₂(Ω)| = {mp.nstr(exact_product, 40)}")
    print(f"  = {home_p} - {mp.nstr(deficit, 25)}")
    print()

    # Multiplicative invariance
    max_m_home = int(mpf('0.5') / fabs(deficit))
    print("MULTIPLICATIVE INVARIANCE:")
    print(f"  At N = {home_q}m: k = -{home_p}m, d = {home_q}/gcd({home_p},{home_q}) = {home_d} (m cancels)")
    print(f"  ε = {mp.nstr(eps_home, 15)} cents at ALL multiples (m cancels in the formula)")
    print(f"  m_max before rounding flips: m < {max_m_home} (N < {home_q * max_m_home})")
    print()

    # Generate dynamic test multiples from the home's own structure
    test_multiples = [1, 2, 3, 5, 7]
    # Add factors of home_d
    for f in range(2, home_d + 1):
        if home_d % f == 0 and f not in test_multiples:
            test_multiples.append(f)
    # Add home_d itself, N_BASE, 100, lcm(N_BASE, home_q)//home_q, 1000
    for extra in [home_d, N_BASE, 100, my_lcm(N_BASE, home_q) // home_q, 1000, max_m_home - 1]:
        if extra > 0 and extra not in test_multiples:
            test_multiples.append(extra)
    test_multiples.sort()

    print("  Verification at selected multiples:")
    for m_test in test_multiples:
        n_test = home_q * m_test
        _, d_test, eps_test, _ = project(omega, n_test)
        print(f"    N = {n_test:>10} = {home_q}×{m_test:<6}: d = {d_test:>6}, ε = {mp.nstr(eps_test, 15)}¢")
    # Show the flip point
    n_flip = home_q * (max_m_home + 1)
    _, d_flip, eps_flip, _ = project(omega, n_flip)
    print(f"    N = {n_flip:>10} = {home_q}×{max_m_home+1:<6}: d = {d_flip:>6}, ε = {mp.nstr(eps_flip, 15)}¢  ◄ home boundary (rounding flips)")
    print()

    # Shadow family analysis — all dynamic
    lcm_base_home = my_lcm(N_BASE, home_q)
    is_native = (N_BASE % home_d == 0)
    print("SHADOW FAMILY STATUS:")
    if is_native:
        print(f"  {home_d} = {factorize(home_d)}, divisor of {N_BASE} → NATIVE family at base N={N_BASE}")
    else:
        print(f"  {home_d} = {factorize(home_d)}, NOT a divisor of {N_BASE} → shadow family at base N={N_BASE}")
    print(f"  N_min (native resolution) = lcm({N_BASE}, {home_q}) = {lcm_base_home}")
    universal_N = 27720
    print(f"  {home_d} divides {universal_N}? {'YES' if universal_N % home_d == 0 else 'NO'} ({universal_N}/{home_d} = {universal_N / home_d:.4f})")
    # Find when home_d first divides lcm(1..k) — check prime factors of home_d
    temp_home = home_d
    max_prime_of_home = 1
    p_check = 2
    while p_check * p_check <= temp_home:
        while temp_home % p_check == 0:
            max_prime_of_home = max(max_prime_of_home, p_check)
            temp_home //= p_check
        p_check += 1 if p_check == 2 else 2
    if temp_home > 1:
        max_prime_of_home = temp_home
    print(f"  Largest prime factor of {home_d}: {max_prime_of_home}")
    print(f"  {home_d} first divides lcm(1..k) when k ≥ {max_prime_of_home}")
    print()

    # Does home_d appear at LCM landmarks?
    print(f"d = {home_d} AT LCM LANDMARKS:")
    found_home_at_lcm = False
    for n_res, k_level, is_lcm in all_resolutions:
        if is_lcm:
            _, d_check, _, _ = project(omega, n_res)
            if d_check == home_d:
                print(f"  ★ N = lcm(1..{k_level}) = {n_res}: d = {home_d}")
                found_home_at_lcm = True
    if not found_home_at_lcm:
        print(f"  d = {home_d} does not appear at any LCM landmark in the computed range.")
    print()

    # All convergent families
    print("ALL CONVERGENT FAMILIES (first 20):")
    print()
    for c in cf_convergents[:20]:
        native = "native" if c['d'] > 0 and 12 % c['d'] == 0 else "shadow"
        fac = factorize(c['d'])
        if len(fac) > 30:
            fac = fac[:27] + "..."
        print(f"  n={c['n']:>2}: d = {c['d']:>15} = {fac:>30}  ε = {c['eps']:>22.15e}¢  ({native})")
    print()

    # Structural interpretation
    print("=" * 100)
    print("HOME RESULT")
    print("=" * 100)
    print()
    print(f"Ω's HOME: d = {home_d} = {factorize(home_d)}")
    print(f"  Found by: continued fraction convergent n={home_idx}")
    print(f"  Quality:  a_{home_idx+1} = {home_quality}")
    print(f"  ε = {mp.nstr(eps_home, 12)} cents ({abs(eps_micro_home)} micro-cents)")
    print(f"  Sub-Koide by factor: {EPS_KOIDE_MICROCENTS / max(abs(eps_micro_home), 1):.0f}")
    print(f"  Manifold state: {{P,D}} Unsubstantiated")
    print(f"  Native resolution: N_min = lcm({N_BASE}, {home_q}) = {lcm_base_home}")
    print(f"  A₀_magic = {impedance(home_d)}")
    print(f"  ξ = {coupling(home_d)}")
    print(f"  Multiplicative invariance: {max_m_home} multiples (N up to {home_q * max_m_home})")


if __name__ == "__main__":
    main()
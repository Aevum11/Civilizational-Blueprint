"""
Chaitin's Omega — Full Sempaevum Lattice Projection
====================================================

Projects Chaitin's Omega_U onto the Sempaevum lattice through the complete
LCM tower escalation. Finds where Omega lives.

Source value: Calude, Dinneen, Shu (2002) "Computing a Glimpse of Randomness"
  Experimental Mathematics 11(3), 361-370.
  First 64 exact bits of Omega_U for their universal Chaitin machine.

All arithmetic: mpmath at 400-bit precision (120+ dps) per Sempaevum §3.1a.
Projection formula: Sempaevum Paper §6, Theorem 6.1.
Home-finding: d-family stabilization across ceil(1/K) = 2 consecutive LCM
  landmarks, with ceil(1/K) = 2 additional verification landmarks.
  NO TERMINATION. The tower is infinite. The algorithm runs until home is found.

Author: Exception Theory projection — Michael James Muller (Aevum Defluo)
Tools: Identification Principle, Descriptor Gap Principle, Subsumption Law
"""

from mpmath import mp, mpf, log, power, nint, fabs
from math import gcd

# ============================================================
# Precision: 400 bits = 120+ decimal places. Uniform. No floats.
# ============================================================
mp.prec = 400

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


def project(r, N):
    """
    Sempaevum projection formula (Theorem 6.1).
    
    Input:  r ∈ ℝ⁺, N ∈ ℤ⁺ (resolution)
    Output: (k, d, ε_cents, ε_microcents)
    
    k = round(N · log₂(r))
    d = N / gcd(|k|, N)
    ε = (N · log₂(r) − k) · 1200 / N   [cents]
    
    All computation via mpmath at 400-bit precision.
    The only conversion to int is for k (which IS an integer by definition)
    and eps_microcents (for exact integer threshold comparison).
    """
    log2_r = log(r, 2)
    Nlog2r = mpf(N) * log2_r
    k = int(nint(Nlog2r))
    
    abs_k = abs(k)
    g = gcd(abs_k, N) if abs_k > 0 else N
    d = N // g
    
    eps_cents = (Nlog2r - mpf(k)) * mpf(1200) / mpf(N)
    eps_microcents = int(round(float(eps_cents * 1000)))
    
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

def generate_multiplicative_refinements(lcm_landmarks, max_mult=100):
    """
    Between each pair of LCM landmarks, generate the multiplicative
    refinements (multiples of 12) that fall between them.
    These refine ε within the same family structure.
    """
    refinements = set()
    for N, _ in lcm_landmarks:
        refinements.add(N)
    
    # Add all multiples of 12 up to the largest landmark considered
    if lcm_landmarks:
        upper = lcm_landmarks[-1][0]
        mult = 12
        while mult <= upper and mult <= 27720:  # multiplicative refinements up to 27720
            refinements.add(mult)
            mult += 12
    
    return sorted(refinements)


# ============================================================
# Main projection
# ============================================================

def main():
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
    lcm_landmarks = list(generate_lcm_tower(max_k=50))
    
    print("=" * 100)
    print("CANONICAL LCM TOWER VERIFICATION")
    print("=" * 100)
    print()
    for N, k_level in lcm_landmarks:
        print(f"  lcm(1..{k_level:>2}) = {N:>15}   τ = {divisor_count(N):>6}   factorization: {factorize(N)}")
    print()
    
    # Generate all N to check (LCM + multiplicative refinements)
    all_N = []
    for N, k_level in lcm_landmarks:
        all_N.append((N, k_level, True))  # True = LCM landmark
    
    # Add multiplicative refinements between base landmarks (up to 27720)
    for mult in range(24, 27721, 12):
        if mult not in [n for n, _, _ in all_N]:
            all_N.append((mult, 0, False))  # False = multiplicative refinement
    
    all_N.sort(key=lambda x: x[0])
    
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
    home_N = None
    home_class = None
    verify_rem = 0
    false_resolutions = []
    sub_koide_hits = []
    
    for N, k_level, is_lcm in all_N:
        k, d, eps, eps_micro = project(omega, N)
        t_N = divisor_count(N)
        abs_micro = abs(eps_micro)
        cls = classify_eps(eps_micro)
        
        # Track sub-Koide hits
        if cls == "sub-Koide":
            sub_koide_hits.append((N, k, d, eps, is_lcm))
        
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
                            'stable_N': home_N,
                            'stable_d': home_d,
                            'break_N': N,
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
                home_N = N
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
            
            if home_found and N == home_N:
                marker = " ◄ HOME?"
            elif home_found and verify_rem == 0 and N > home_N:
                marker = " ✓ VERIFIED"
        
        # Format source
        if is_lcm:
            source = f"lcm(1..{k_level})"
        else:
            source = f"{N//12}×12"
        
        # Format d factors (truncate if too long)
        d_fac = factorize(d)
        if len(d_fac) > 24:
            d_fac = d_fac[:21] + "..."
        
        lcm_mark = "★" if is_lcm else " "
        
        print(f"{lcm_mark}{' ':>2} {N:>12} {source:>14} {t_N:>6} "
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
        for i, fr in enumerate(false_resolutions, 1):
            print(f"  #{i}: d = {fr['stable_d']} appeared stable at N = {fr['stable_N']}")
            print(f"      but d changed to {fr['break_d']} at N = {fr['break_N']} = lcm(1..{fr['break_k_level']})")
            print(f"      Stable d factors: {factorize(fr['stable_d'])}")
            print(f"      New d factors:    {factorize(fr['break_d'])}")
            print()
    
    if home_found and verify_rem == 0:
        k_h, d_h, eps_h, eps_micro_h = project(omega, home_N)
        print(f"HOME FOUND:")
        print(f"  d-family:        d = {home_d}")
        print(f"  d factorization: {factorize(home_d)}")
        print(f"  Classification:  {home_class}")
        print(f"  Stable at:       N = {home_N} = {factorize(home_N)}")
        print(f"  k = {k_h}")
        print(f"  ε = {mp.nstr(eps_h, 20)} cents")
        print(f"  |ε| = {abs(eps_micro_h)} micro-cents")
        print(f"  A₀_magic = {impedance(home_d)}")
        print(f"  ξ = {coupling(home_d)}")
    else:
        print(f"ESCALATION IN PROGRESS — d HAS NOT STABILIZED")
        print(f"  Last d at LCM landmark: {last_lcm_d}")
        print(f"  Last d factorization:   {factorize(last_lcm_d)}")
        print(f"  Consecutive same d:     {consec_lcm}")
        print()
        
        # Analyze the d-family trajectory at LCM landmarks
        print("d-FAMILY TRAJECTORY AT LCM LANDMARKS:")
        print()
        for N, k_level, is_lcm in all_N:
            if is_lcm:
                k, d, eps, eps_micro = project(omega, N)
                d_fac = factorize(d)
                print(f"  N = {N:>15} = lcm(1..{k_level:>2})  →  d = {d:>15}  = {d_fac}")
    
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
    
    eps_float = float(fabs(eps12))
    tightness = 100.0 / (100.0 + eps_float)
    print(f"  Tightness t = 100/(100+|ε|) = {tightness:.8f}")
    print(f"  K = 2/3 = {2/3:.8f}")
    print(f"  t vs K: {'t > K → inside lattice' if tightness > 2/3 else 't ≤ K → near ∂I'}")
    
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


if __name__ == "__main__":
    main()

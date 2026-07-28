"""
T-SHADOW COMPLETE VERIFICATION SCRIPT
======================================
Comprehensive verification of all T-shadow findings from the ET Audio Analysis.

Findings verified:
 1. PDT Bisection (Theorem 12.8): 50.9% D / 49.8% T energy
 2. K = 2/3 in FQG d_c=12 energy: 66.651% (161 ppm from K)
 3. T's |ε|/cell ≈ 1/S when D removed: 0.24884 (0.47% from 1/4)
 4. D's |ε|/cell deviates 15× more from 1/S than T's
 5. Phase axis tighter than real axis: ratio 0.956
 6. T more phase-coherent than D: R_T/R_D = 1.071
 7. d_r=1 (gravity) unique spatial bias: +18.4° vs ≤2.2° all others
 8. F0 FQG address: (d_r=1, d_θ=12, d_c=12)
 9. T permeates all 6 d-families
10. Koide-comma region: T attracted (1.088×), D repelled (0.635×)
11. D/T sign asymmetry: D leans ε<0 (Koide), T leans ε>0 (comma)
12. T's resolution gradient: T/D drops from 1.69 to 0.57 center→∂I
13. ∂I edge: D accumulates (1.35×), T depletes (0.82×)
14. Tower harmonics at ∂I: 5.9% (48% excess over uniform 4.0%)
15. H7 family at ε=+48.29¢ resolved at N=420=LCM(1..7)
16. LCM tower |ε|/cell convergence: 1/S steady state

Data: REW V5.31.3 (65,535 bins × 2ch), ET report JSON (15 tower levels)
Recording: 3.wav (48kHz/32-bit/WASAPI Exclusive/HyperX QuadCast S)

Author: Analysis of Michael James Muller's Exception Theory
        P ∘ D ∘ T = E (The Sempaevum, Paper v20, April 2026)
"""

import numpy as np
import math
import json
import sys


# ============================================================
# ET CONSTANTS (from primitives, zero ad hoc)
# ============================================================
N = 12
S = 4
K = 2.0 / 3.0
T_WEIGHT = 1.0 / 3.0
V = 1.0 / N
C4 = 261.63
CELL_CENTS = 1200.0 / N
COMMA_CENTS = (12 * np.log2(3/2) - 7) * 100  # 1.955001 cents
TZ_START = CELL_CENTS / 3  # 33.33 cents — Twilight Zone boundary
DI_BOUNDARY = CELL_CENTS / 2  # 50 cents — ∂I boundary


# ============================================================
# DATA LOADING
# ============================================================
def parse_rew(path):
    freqs, spls, phases = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('*') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    freqs.append(float(parts[0]))
                    spls.append(float(parts[1]))
                    phases.append(float(parts[2]))
                except ValueError:
                    continue
    return np.array(freqs), np.array(spls), np.array(phases)


# ============================================================
# LATTICE PROJECTION
# ============================================================
def project_real(freqs, N=12, ref=261.63):
    k = np.zeros(len(freqs), dtype=int)
    d = np.zeros(len(freqs), dtype=int)
    eps = np.zeros(len(freqs))
    pmask = freqs > 0
    log2_r = np.log2(freqs[pmask] / ref)
    exact = N * log2_r
    k[pmask] = np.round(exact).astype(int)
    eps[pmask] = (exact - k[pmask]) * 1200.0 / N
    for i in np.where(pmask)[0]:
        ki = int(abs(k[i]))
        g = math.gcd(ki, N) if ki != 0 else N
        d[i] = N // g
    return k, d, eps


def project_phase(phases_deg, N=12):
    theta = np.deg2rad(phases_deg) % (2 * np.pi)
    exact = N * theta / (2 * np.pi)
    k_th = np.round(exact).astype(int) % N
    eps_th = (exact - np.round(exact)) * 1200.0 / N
    d_th = np.zeros(len(phases_deg), dtype=int)
    for i in range(len(k_th)):
        ki = int(abs(k_th[i]))
        g = math.gcd(ki, N) if ki != 0 else N
        d_th[i] = N // g
    return k_th, d_th, eps_th


def euler_phi(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


# ============================================================
# MAIN VERIFICATION
# ============================================================
def run_all_tests():
    print("=" * 80)
    print("T-SHADOW COMPLETE VERIFICATION")
    print("P ∘ D ∘ T = E — Exception Theory (The Sempaevum)")
    print("=" * 80)

    # Load data
    f1, s1, p1 = parse_rew("/mnt/user-data/uploads/Ch1_4.txt")
    f2, s2, p2 = parse_rew("/mnt/user-data/uploads/Ch2_4.txt")
    with open("/mnt/user-data/uploads/3_et_report.json") as f:
        report = json.load(f)

    F0 = 129.2724
    power1 = 10.0 ** (s1 / 10.0)
    power2 = 10.0 ** (s2 / 10.0)

    # Projections
    k_r, d_r, eps_r = project_real(f1)
    k_th, d_th, eps_th = project_phase(p1)
    d_c = np.array([math.lcm(int(dr), int(dt)) for dr, dt in zip(d_r, d_th)])
    delta_phase = ((p2 - p1 + 180) % 360) - 180

    # Masks
    harm_mask = np.zeros(len(f1), dtype=bool)
    for i, freq in enumerate(f1):
        if freq < F0 / 2:
            continue
        n = round(freq / F0)
        if n > 0 and abs(freq - n * F0) < F0 / 4:
            harm_mask[i] = True
    inter_mask = ~harm_mask & (f1 > F0 / 2) & (f1 < 20000)
    voice = (f1 > F0 / 2) & (f1 < 20000)

    tests_passed = 0
    tests_total = 0

    def test(name, condition, detail=""):
        nonlocal tests_passed, tests_total
        tests_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            tests_passed += 1
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")
        return condition

    # ==========================================
    # SECTION 1: PDT BISECTION (Theorem 12.8)
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 1: PDT BISECTION (Paper 20, Theorem 12.8)")
    print(f"{'='*70}")

    total_voice = power1[voice].sum()
    D_frac = power1[harm_mask].sum() / total_voice
    T_frac = power1[inter_mask].sum() / total_voice
    DT_ratio = D_frac / T_frac

    test("PDT Bisection: D:T ratio within 5% of 1:1",
         abs(DT_ratio - 1.0) < 0.05,
         f"D={D_frac*100:.3f}%, T={T_frac*100:.3f}%, ratio={DT_ratio:.4f}")

    test("PDT Bisection: D energy 45-55%",
         0.45 < D_frac < 0.55,
         f"D={D_frac*100:.3f}%")

    test("PDT Bisection: T energy 45-55%",
         0.45 < T_frac < 0.55,
         f"T={T_frac*100:.3f}%")

    # ==========================================
    # SECTION 2: K = 2/3 IN FQG
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 2: KOIDE RATIO IN FQG d_c=12 ENERGY")
    print(f"{'='*70}")

    total_power = power1.sum()
    dc12_energy = power1[d_c == 12].sum() / total_power

    # Verify FQG cell count
    dc12_cells = 0
    for kr_mod in range(N):
        for kth_mod in range(N):
            dr_c = N // math.gcd(kr_mod if kr_mod != 0 else N, N)
            dt_c = N // math.gcd(kth_mod if kth_mod != 0 else N, N)
            if math.lcm(dr_c, dt_c) == 12:
                dc12_cells += 1

    test("FQG: 96/144 cells have d_c=12 = K exactly",
         dc12_cells == 96,
         f"cells={dc12_cells}/144={dc12_cells/144:.6f}, K={K:.6f}")

    test("FQG: d_c=12 energy tracks K to <500 ppm",
         abs(dc12_energy - K) * 1e6 < 500,
         f"energy={dc12_energy*100:.4f}%, K={K*100:.4f}%, deviation={abs(dc12_energy-K)*1e6:.0f} ppm")

    # ==========================================
    # SECTION 3: T's |ε|/cell ≈ 1/S
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 3: T's |ε_r|/cell vs D's |ε_r|/cell vs 1/S")
    print(f"{'='*70}")

    D_eps = np.average(np.abs(eps_r[harm_mask]), weights=power1[harm_mask])
    T_eps = np.average(np.abs(eps_r[inter_mask]), weights=power1[inter_mask])
    D_ratio = D_eps / CELL_CENTS
    T_ratio = T_eps / CELL_CENTS
    one_over_S = 1.0 / S

    test("T's |ε|/cell closer to 1/S than D's",
         abs(T_ratio - one_over_S) < abs(D_ratio - one_over_S),
         f"T dev={abs(T_ratio-one_over_S)/one_over_S*100:.3f}%, D dev={abs(D_ratio-one_over_S)/one_over_S*100:.3f}%")

    test("T's |ε|/cell within 1% of 1/S",
         abs(T_ratio - one_over_S) / one_over_S < 0.01,
         f"T ratio={T_ratio:.6f}, 1/S={one_over_S:.6f}, dev={abs(T_ratio-one_over_S)/one_over_S*100:.4f}%")

    test("D's |ε|/cell deviates >5% from 1/S (D is NOT uniform)",
         abs(D_ratio - one_over_S) / one_over_S > 0.05,
         f"D ratio={D_ratio:.6f}, dev={abs(D_ratio-one_over_S)/one_over_S*100:.3f}%")

    # ==========================================
    # SECTION 4: AXIS ASYMMETRY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 4: REAL vs PHASE AXIS ASYMMETRY")
    print(f"{'='*70}")

    ew_eps_r = np.average(np.abs(eps_r[voice]), weights=power1[voice])
    ew_eps_th = np.average(np.abs(eps_th[voice]), weights=power1[voice])

    test("Phase axis |ε| tighter than real axis",
         ew_eps_th < ew_eps_r,
         f"|ε_θ|={ew_eps_th:.4f}¢, |ε_r|={ew_eps_r:.4f}¢, ratio={ew_eps_th/ew_eps_r:.4f}")

    # ==========================================
    # SECTION 5: PHASE COHERENCE
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 5: PHASE COHERENCE (R)")
    print(f"{'='*70}")

    R_D = np.abs(np.mean(np.exp(1j * np.deg2rad(p1[harm_mask]))))
    R_T = np.abs(np.mean(np.exp(1j * np.deg2rad(p1[inter_mask]))))

    test("T more phase-coherent than D",
         R_T > R_D,
         f"R_T={R_T:.6f}, R_D={R_D:.6f}, ratio={R_T/R_D:.4f}")

    # ==========================================
    # SECTION 6: GRAVITY SPATIAL BIAS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 6: d_r=1 GRAVITY SPATIAL BIAS")
    print(f"{'='*70}")

    gravity_bias = abs(delta_phase[d_r == 1].mean())
    max_other = max(abs(delta_phase[d_r == d].mean()) for d in [2, 3, 4, 6, 12])

    test("Gravity bias > 3× any other family",
         gravity_bias > 3 * max_other,
         f"gravity={gravity_bias:.1f}°, max_other={max_other:.1f}°, ratio={gravity_bias/max_other:.1f}×")

    # ==========================================
    # SECTION 7: F0 FQG ADDRESS
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 7: F0 COMPLETE FQG ADDRESS")
    print(f"{'='*70}")

    idx = np.argmin(np.abs(f1 - F0))
    test("F0 at d_r=1 (gravity on real axis)",
         d_r[idx] == 1, f"d_r={d_r[idx]}")
    test("F0 at d_θ=12 (EM on phase axis)",
         d_th[idx] == 12, f"d_θ={d_th[idx]}")
    test("F0 at d_c=12 (full EM combined)",
         d_c[idx] == 12, f"d_c={d_c[idx]}")

    # ==========================================
    # SECTION 8: T PERMEATES ALL FAMILIES
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 8: T PERMEATES ALL d-FAMILIES")
    print(f"{'='*70}")

    inter_d = d_r[inter_mask]
    inter_pow = power1[inter_mask]
    total_inter = inter_pow.sum()
    all_pop = all(
        inter_pow[inter_d == d].sum() / total_inter > 0.001
        for d in [1, 2, 3, 4, 6, 12]
    )
    test("All 6 d-families populated (>0.1%) in T's inter-harmonic energy",
         all_pop)

    # ==========================================
    # SECTION 9: KOIDE-COMMA ATTRACTOR REGION
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 9: KOIDE-COMMA ATTRACTOR PAIR")
    print(f"{'='*70}")

    abs_eps = np.abs(eps_r)
    uniform_comma = 2 * COMMA_CENTS / CELL_CENTS

    T_comma = power1[inter_mask & (abs_eps < COMMA_CENTS)].sum() / power1[inter_mask].sum()
    D_comma = power1[harm_mask & (abs_eps < COMMA_CENTS)].sum() / power1[harm_mask].sum()

    test("T over-populates Koide-comma region (>uniform)",
         T_comma > uniform_comma,
         f"T={T_comma*100:.3f}%, uniform={uniform_comma*100:.3f}%, ratio={T_comma/uniform_comma:.3f}×")

    test("D under-populates Koide-comma region (<uniform)",
         D_comma < uniform_comma,
         f"D={D_comma*100:.3f}%, uniform={uniform_comma*100:.3f}%, ratio={D_comma/uniform_comma:.3f}×")

    # D/T sign asymmetry
    D_pos = power1[harm_mask & (eps_r > 0)].sum()
    D_neg = power1[harm_mask & (eps_r < 0)].sum()
    T_pos = power1[inter_mask & (eps_r > 0)].sum()
    T_neg = power1[inter_mask & (eps_r < 0)].sum()

    test("D leans ε<0 (Koide sign) — D_neg > D_pos",
         D_neg > D_pos,
         f"D: {D_neg/(D_pos+D_neg)*100:.2f}% negative")

    test("T leans ε>0 (comma sign) — T_pos > T_neg",
         T_pos > T_neg,
         f"T: {T_pos/(T_pos+T_neg)*100:.2f}% positive")

    # ==========================================
    # SECTION 10: T's RESOLUTION GRADIENT AT ∂I
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 10: T's RESOLUTION GRADIENT (center → ∂I)")
    print(f"{'='*70}")

    # T/D ratio at |ε| = 0-5¢ vs 45-50¢
    center_T = power1[inter_mask & (abs_eps < 5)].sum()
    center_D = power1[harm_mask & (abs_eps < 5)].sum()
    edge_T = power1[inter_mask & (abs_eps >= 45)].sum()
    edge_D = power1[harm_mask & (abs_eps >= 45)].sum()

    center_ratio = center_T / center_D if center_D > 0 else 0
    edge_ratio = edge_T / edge_D if edge_D > 0 else 0

    test("T/D ratio higher at center than at ∂I edge",
         center_ratio > edge_ratio,
         f"center T/D={center_ratio:.3f}, edge T/D={edge_ratio:.3f}")

    test("T/D > 1 at cell centers (T dominates near nodes)",
         center_ratio > 1.0,
         f"center T/D={center_ratio:.3f}")

    test("T/D < 1 at ∂I edge (D dominates near boundary)",
         edge_ratio < 1.0,
         f"edge T/D={edge_ratio:.3f}")

    # ∂I zone populations
    di_D = power1[harm_mask & (abs_eps >= 48)].sum() / power1[harm_mask].sum()
    di_T = power1[inter_mask & (abs_eps >= 48)].sum() / power1[inter_mask].sum()
    uniform_di = 2 * 2 / CELL_CENTS  # 4% for last 2¢ on each side

    test("D accumulates at ∂I (above uniform)",
         di_D > uniform_di,
         f"D at ∂I={di_D*100:.3f}%, uniform={uniform_di*100:.1f}%, ratio={di_D/uniform_di:.3f}×")

    test("T depletes at ∂I (below uniform)",
         di_T < uniform_di,
         f"T at ∂I={di_T*100:.3f}%, uniform={uniform_di*100:.1f}%, ratio={di_T/uniform_di:.3f}×")

    # ==========================================
    # SECTION 11: TOWER HARMONICS AT ∂I
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 11: HARMONICS AT ∂I AND TOWER RESOLUTION")
    print(f"{'='*70}")

    di_edge_count = 0
    total_harmonics = 0
    h7_eps = None
    h7_resolved_at = None

    for n in range(1, 190):
        freq = F0 * n
        if freq > 24000:
            break
        total_harmonics += 1
        log2_r = math.log2(freq / C4)
        exact_k = N * log2_r
        eps_n = abs((exact_k - round(exact_k)) * 100)
        if eps_n > 48:
            di_edge_count += 1
        if n == 7:
            h7_eps = eps_n
            # Check tower resolution
            for test_N in [60, 420, 2520, 27720]:
                exact_tN = test_N * log2_r
                cell_tN = 1200.0 / test_N
                eps_tN = abs((exact_tN - round(exact_tN)) * cell_tN)
                if eps_tN / cell_tN < 1.0/3:
                    h7_resolved_at = test_N
                    break

    di_frac = di_edge_count / total_harmonics
    uniform_di_frac = 0.04  # 4% expected

    test("Voice has excess harmonics at ∂I (> 4% uniform)",
         di_frac > uniform_di_frac,
         f"measured={di_frac*100:.1f}%, uniform={uniform_di_frac*100:.1f}%, excess={di_frac/uniform_di_frac:.2f}×")

    test("H7 is at ∂I edge (|ε| > 48¢)",
         h7_eps is not None and h7_eps > 48,
         f"H7 |ε|={h7_eps:.2f}¢")

    test("H7 resolves at N=420 = LCM(1..7)",
         h7_resolved_at == 420,
         f"resolved at N={h7_resolved_at}")

    # ==========================================
    # SECTION 12: LCM TOWER CONVERGENCE
    # ==========================================
    print(f"\n{'='*70}")
    print(f"SECTION 12: LCM TOWER |ε|/cell CONVERGENCE")
    print(f"{'='*70}")

    ld = report['lattice_distribution']
    tower_ratios = []
    for N_str in sorted(ld.keys(), key=lambda x: int(x)):
        N_val = int(N_str)
        eps_c = float(ld[N_str]['weighted_mean_abs_epsilon_cents'])
        cell = 1200.0 / N_val
        tower_ratios.append(eps_c / cell)

    mean_ratio = np.mean(tower_ratios)
    # Tower is infinite — N=27720 is N_FULL, not the limit
    test("|ε|/cell mean across 15 tower levels near 1/S",
         abs(mean_ratio - one_over_S) / one_over_S < 0.01,
         f"mean={mean_ratio:.6f}, 1/S={one_over_S:.6f}, dev={abs(mean_ratio-one_over_S)/one_over_S*100:.3f}%")

    # |ε| decreases monotonically in ABSOLUTE terms
    abs_eps_tower = [float(ld[N_str]['weighted_mean_abs_epsilon_cents'])
                     for N_str in sorted(ld.keys(), key=lambda x: int(x))]
    monotonic = all(abs_eps_tower[i] > abs_eps_tower[i+1] for i in range(len(abs_eps_tower)-1))
    test("Absolute |ε| decreases monotonically through tower",
         monotonic,
         f"Range: {abs_eps_tower[0]:.4f}¢ → {abs_eps_tower[-1]:.6f}¢")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"VERIFICATION COMPLETE: {tests_passed}/{tests_total} TESTS PASSED")
    print(f"{'='*70}")

    if tests_passed == tests_total:
        print(f"\n  ALL TESTS PASSED. All T-shadow findings verified.")
    else:
        print(f"\n  {tests_total - tests_passed} TESTS FAILED — investigate.")
        sys.exit(1)

    return tests_passed, tests_total


if __name__ == "__main__":
    passed, total = run_all_tests()

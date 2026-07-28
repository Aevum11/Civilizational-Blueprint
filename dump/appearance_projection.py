"""
APPEARANCE PROJECTION: r = R_charge / ƛ_e
Project nuclear charge radii onto the Sempaevum.

R_charge = r₀ × A^(1/3) (nuclear physics, measured r₀ = 1.2 fm)
ƛ_e = ℏ/(m_e·c) = 386.15926796... fm (reduced electron Compton wavelength)

For key isotopes: use MEASURED charge radii from Angeli & Marinova (2013).
For all others: use R = r₀ × A^(1/3).

The appearance lattice should show:
- Shell closures as ε anomalies (magic nuclei are more compact)
- Size vs mass decorrelation
- Physical form on the lattice
"""
import sys
sys.path.insert(0, '/home/claude')
import mpmath
mpmath.mp.dps = 130
from collections import defaultdict

from sempaevum_nz_viewer import (
    build_atomic_particles, sempaevum_project_mp, cf_home_find, N, et_gcd
)

# Constants as mpmath strings — zero float
LAMBDA_E_FM = mpmath.mpf("386.15926796090585")  # ℏ/(m_e·c) in fm, CODATA 2018
R0_FM = mpmath.mpf("1.2")  # nuclear radius parameter in fm (measured)

# ── MEASURED charge radii (Angeli & Marinova 2013 + CODATA updates) ──
# Key isotopes with precision measurements. Format: (Z, A): R_rms in fm as string
MEASURED_RADII = {
    (1,1): "0.8783",   (1,2): "2.1421",   (1,3): "1.7591",
    (2,3): "1.9661",   (2,4): "1.6755",   (2,6): "2.0710",   (2,8): "1.9290",
    (3,6): "2.5890",   (3,7): "2.3905",   (3,8): "2.3390",   (3,9): "2.2170",   (3,11): "2.4670",
    (4,9): "2.5190",   (4,7): "2.6470",   (4,10): "2.3570",  (4,12): "2.5020",
    (5,10): "2.4277",  (5,11): "2.4060",
    (6,12): "2.4702",  (6,13): "2.4614",  (6,14): "2.4970",
    (7,14): "2.5582",
    (8,16): "2.6991",  (8,17): "2.6932",  (8,18): "2.7726",
    (10,20): "3.0055", (10,21): "2.9710", (10,22): "2.9525",
    (12,24): "3.0570", (12,25): "3.0284", (12,26): "3.0337",
    (14,28): "3.1224",
    (16,32): "3.2611",
    (18,36): "3.3905", (18,40): "3.4274",
    (20,40): "3.4776", (20,42): "3.5081", (20,43): "3.4952",
    (20,44): "3.5157", (20,48): "3.4771",  # Ca-48 ≈ Ca-40!
    (26,54): "3.6880", (26,56): "3.7377", (26,57): "3.7530", (26,58): "3.7740",
    (28,58): "3.7757", (28,60): "3.8118", (28,62): "3.8399", (28,64): "3.8604",
    (30,64): "3.9283", (30,66): "3.9491", (30,68): "3.9658",
    (38,88): "4.2191",
    (40,90): "4.2694", (40,91): "4.2847", (40,92): "4.3057", (40,96): "4.3512",
    (50,112): "4.5948",(50,114): "4.6096",(50,116): "4.6254",
    (50,118): "4.6396",(50,120): "4.6519",(50,122): "4.6631",
    (50,124): "4.6735",(50,132): "4.7093",
    (56,138): "4.8378",
    (82,204): "5.4803",(82,206): "5.4902",(82,207): "5.4945",
    (82,208): "5.5012",(82,210): "5.5243",
    (83,209): "5.5211",
}

print(f"Measured charge radii loaded: {len(MEASURED_RADII)} isotopes")

# Load AME2020 isotopes
particles = build_atomic_particles(measured_only=True)
print(f"AME2020 isotopes: {len(particles)}")

# ── Compute charge radii and project ──
print(f"\nProjecting r = R_charge / ƛ_e onto the Sempaevum at N=12...")

results = []
for p in particles:
    Z = p.get("Z", 0)
    A = p.get("A", 0)
    N_val = A - Z
    if Z == 0 or A == 0:
        continue
    
    # Get charge radius: measured if available, else formula
    measured = MEASURED_RADII.get((Z, A))
    if measured:
        R_fm = mpmath.mpf(measured)
        source = "measured"
    else:
        R_fm = R0_FM * mpmath.power(mpmath.mpf(A), mpmath.mpf(1) / mpmath.mpf(3))
        source = "formula"
    
    # DSR: r = R_charge / ƛ_e
    r_mp = R_fm / LAMBDA_E_FM
    r_str = mpmath.nstr(r_mp, 40)
    
    # Project
    proj = sempaevum_project_mp(r_str, 12)
    if proj is None:
        continue
    k, d, eps = proj
    
    # CF method
    cf = cf_home_find(r_mp)
    
    # Expected from formula
    R_formula = R0_FM * mpmath.power(mpmath.mpf(A), mpmath.mpf(1) / mpmath.mpf(3))
    r_formula = R_formula / LAMBDA_E_FM
    proj_formula = sempaevum_project_mp(mpmath.nstr(r_formula, 40), 12)
    k_formula = proj_formula[0] if proj_formula else None
    
    # Deviation from formula (in cents)
    if measured and proj_formula:
        delta_eps = eps - proj_formula[2]
    else:
        delta_eps = mpmath.mpf(0)
    
    results.append({
        "symbol": p["symbol"], "Z": Z, "A": A, "N": N_val,
        "R_fm": R_fm, "source": source,
        "r_mp": r_mp, "k": k, "d": d, "eps": eps,
        "k_formula": k_formula,
        "delta_eps": delta_eps,
        "cf_d": cf["cf_d_home"], "cf_q": cf["cf_quality"], "cf_class": cf["cf_class"],
        "is_nat": p.get("is_naturally_occurring", False),
    })

print(f"Projected {len(results)} isotopes\n")

# ================================================================
print("=" * 100)
print("1. APPEARANCE LATTICE OVERVIEW")
print("=" * 100)

k_vals = [r["k"] for r in results]
print(f"  k range: {min(k_vals)} to {max(k_vals)}")
print(f"  Measured radii projected: {sum(1 for r in results if r['source'] == 'measured')}")
print(f"  Formula radii projected: {sum(1 for r in results if r['source'] == 'formula')}")

# k distribution
k_counts = defaultdict(int)
k_nat = defaultdict(int)
for r in results:
    k_counts[r["k"]] += 1
    if r["is_nat"]:
        k_nat[r["k"]] += 1

print(f"\n  k distribution (appearance lattice at N=12):")
for k in sorted(k_counts.keys()):
    c = k_counts[k]
    n = k_nat.get(k, 0)
    d = 12 // et_gcd(abs(k), 12) if k != 0 else 1
    pct = 100 * n / c if c else 0
    bar = "█" * min(c // 2, 50)
    print(f"    k={k:>4d} (d={d:>2d}): {c:>5d} total, {n:>4d} nat ({pct:>5.1f}%) {bar}")

# ================================================================
print(f"\n{'=' * 100}")
print("2. DOUBLY-MAGIC NUCLEI — Shell closures on the appearance lattice")
print("   Measured radii vs formula. Deviation = shell effect.")
print("=" * 100)

doubly_magic = [(2,4,"⁴He"), (8,16,"¹⁶O"), (20,40,"⁴⁰Ca"), (20,48,"⁴⁸Ca"),
                (28,58,"⁵⁸Ni"), (50,132,"¹³²Sn"), (82,208,"²⁰⁸Pb")]

print(f"\n  {'Nucleus':>8s} {'R_meas(fm)':>10s} {'R_form(fm)':>10s} {'δR/R %':>8s} "
      f"{'k_meas':>6s} {'k_form':>6s} {'Δk':>4s} {'d':>3s} {'ε(¢)':>10s}")
for Z, A, name in doubly_magic:
    r = next((x for x in results if x["Z"] == Z and x["A"] == A), None)
    if r and r["source"] == "measured":
        R_form = R0_FM * mpmath.power(mpmath.mpf(A), mpmath.mpf(1) / mpmath.mpf(3))
        delta_pct = (r["R_fm"] - R_form) / R_form * 100
        dk = r["k"] - r["k_formula"] if r["k_formula"] else 0
        print(f"  {name:>8s} {mpmath.nstr(r['R_fm'],5):>10s} {mpmath.nstr(R_form,5):>10s} "
              f"{mpmath.nstr(delta_pct,3):>8s} {r['k']:>6d} {r['k_formula']:>6d} {dk:>4d} "
              f"{r['d']:>3d} {mpmath.nstr(r['eps'],6):>10s}")

# ================================================================
print(f"\n{'=' * 100}")
print("3. Ca-48 vs Ca-40 — The IDENTICAL radii anomaly")
print("   Two nuclei with 8 neutron difference but the SAME charge radius")
print("=" * 100)

ca40 = next((r for r in results if r["Z"]==20 and r["A"]==40), None)
ca48 = next((r for r in results if r["Z"]==20 and r["A"]==48), None)
if ca40 and ca48:
    print(f"\n  Ca-40: R = {mpmath.nstr(ca40['R_fm'],5)} fm → k={ca40['k']}, d={ca40['d']}, ε={mpmath.nstr(ca40['eps'],8)}¢")
    print(f"  Ca-48: R = {mpmath.nstr(ca48['R_fm'],5)} fm → k={ca48['k']}, d={ca48['d']}, ε={mpmath.nstr(ca48['eps'],8)}¢")
    print(f"  ΔR = {mpmath.nstr(abs(ca40['R_fm'] - ca48['R_fm']), 4)} fm")
    print(f"  Δk = {ca48['k'] - ca40['k']}, Δε = {mpmath.nstr(ca48['eps'] - ca40['eps'], 8)}¢")
    print(f"  8 extra neutrons, virtually ZERO change in charge radius!")
    print(f"  On the appearance lattice: {'SAME k' if ca40['k']==ca48['k'] else 'DIFFERENT k'}")

# ================================================================
print(f"\n{'=' * 100}")
print("4. SHELL CLOSURE ε ANOMALIES — Isotope chains crossing magic N")
print("   If shell closures make nuclei more compact, ε should show it")
print("=" * 100)

# Tin chain (Z=50) — crosses N=50 and N=82
by_Z = defaultdict(list)
for r in results:
    by_Z[r["Z"]].append(r)

for Z_target, name in [(20, "Calcium"), (50, "Tin"), (82, "Lead")]:
    chain = sorted(by_Z.get(Z_target, []), key=lambda r: r["A"])
    if not chain: continue
    
    print(f"\n  {name} (Z={Z_target}) isotope chain on the appearance lattice:")
    for r in chain:
        N_val = r["A"] - Z_target
        magic = " ◆" if N_val in {2,8,20,28,50,82,126} else "  "
        src = "M" if r["source"] == "measured" else "F"
        print(f"    A={r['A']:>3d} N={N_val:>3d}{magic} R={mpmath.nstr(r['R_fm'],5):>7s}fm "
              f"k={r['k']:>4d} d={r['d']:>2d} ε={mpmath.nstr(r['eps'],6):>10s}¢ [{src}]")

# ================================================================
print(f"\n{'=' * 100}")
print("5. MASS vs APPEARANCE — Two complementary projections")
print("   Same isotopes, different lattice positions")
print("=" * 100)

# For key isotopes, show (k_mass, k_appearance) pairs
from sempaevum_nz_viewer import compute_particle_projections
all_proj = compute_particle_projections(particles[:20])  # Just first 20 for speed

# Actually, let me compute mass projection for key isotopes manually
MU_ME = mpmath.mpf("1822.888486209")
print(f"\n  {'Isotope':>10s} {'k_mass':>7s} {'k_appear':>8s} {'Δk':>5s} {'d_mass':>6s} {'d_app':>5s}")
for r in results:
    if r["source"] != "measured": continue
    # Compute mass projection
    mass_str = None
    for p in particles:
        if p["Z"] == r["Z"] and p["A"] == r["A"]:
            mass_str = p.get("Ar_str", p.get("mass_str"))
            break
    if not mass_str: continue
    
    r_mass = mpmath.mpf(mass_str) * MU_ME
    proj_mass = sempaevum_project_mp(mpmath.nstr(r_mass, 40), 12)
    if proj_mass:
        k_mass, d_mass = proj_mass[0], proj_mass[1]
        dk = r["k"] - k_mass
        print(f"  {r['symbol']:>10s} {k_mass:>7d} {r['k']:>8d} {dk:>5d} {d_mass:>6d} {r['d']:>5d}")

# ================================================================
print(f"\n{'=' * 100}")
print("6. STABILITY ON THE APPEARANCE LATTICE")
print("   Does the appearance projection see stability?")
print("=" * 100)

# Stability separation by k
print(f"\n  Natural fraction by k (appearance):")
for k in sorted(k_counts.keys()):
    if k_counts[k] < 10: continue
    n = k_nat.get(k, 0)
    c = k_counts[k]
    pct = 100 * n / c
    bar = "█" * int(pct / 2)
    print(f"    k={k:>4d}: {n:>4d}/{c:>5d} = {pct:>5.1f}% {bar}")

print(f"\n{'=' * 100}")
print("APPEARANCE PROJECTION COMPLETE")
print("=" * 100)

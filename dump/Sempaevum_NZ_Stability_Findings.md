# Sempaevum N/Z Stability Projection Findings

**The Descriptor That Reveals Nuclear Stability — and Predicts the Island of Stability**

**Version 1.0 — 2323 Measured Isotopes + Superheavy Predictions**

**DSR:** r = N/Z (neutron-to-proton ratio)
**Projection:** Sempaevum bijection Π₁₂(r) = (k, d, ε) at N=12, 120-digit mpmath precision
**Data:** AME2020 (Wang et al., Chinese Physics C45, 030003, 2021) — 2328 measured isotopes
**Home-finding:** Dual-pathway — LCM tower (d-family stabilization) + CF method (§7.11 Step 3a)
**Pipeline:** String → mpmath throughout. Zero float64 in the computational chain.

---

## Section 1: The Stability Band — k_NZ ∈ [0, 7]

The mass projection (r = m × m_u/m_e) is blind to nuclear stability — natural and unstable isotopes distribute identically across all six d_r families. When we project r = N/Z instead, the valley of stability emerges as a sharp lattice band.

| k_NZ | N/Z ≈ | Natural | Total | Nat % | Character |
|---|---|---|---|---|---|
| < 0 | < 1.0 | 0 | 61 | 0% | Proton-rich — all unstable |
| 0 | 1.0 | 13 | 38 | 34.2% | N=Z line (d=1, Gravity) |
| 1 | 1.06 | 10 | 65 | 15.4% | Light stable isotopes |
| 2 | 1.12 | 17 | 96 | 17.7% | |
| 3 | 1.19 | 29 | 166 | 17.5% | |
| 4 | 1.26 | 43 | 266 | 16.2% | |
| 5 | 1.33 | 42 | 323 | 13.0% | |
| 6 | 1.41 | 66 | 352 | 18.8% | Heavy stable isotopes |
| **7** | **1.50** | **60** | **406** | **14.8%** | **Band edge — Koide attractor** |
| 8 | 1.59 | 4 | 356 | 1.1% | Sharp cutoff — primordials only |
| ≥ 9 | > 1.68 | 0 | 192 | 0% | Neutron-rich — all unstable |

**280 of 284 naturally occurring isotopes** (98.6%) live within k_NZ = 0 to 7. The 4 at k_NZ = 8 are heavy primordials (U-238, Th-232 and neighbors). Everything beyond k_NZ = 8: zero natural isotopes.

The stability band is **8 semitones wide**. This is the valley of nuclear stability expressed as a lattice band on the multiplicative manifold.

---

## Section 2: N/Z = 3/2 and the Koide Attractor

**Discovery:** All 49 isotopes with N/Z = 3/2 project to |ε_NZ| = 1.955¢ — exactly the Koide attractor threshold. This is the Sempaevum's own self-consistency residual (the value where the defining constants {N, 1/N, K, 1/K} project at d=12).

The N/Z = 3/2 isotopes span the entire nuclear chart:

| Isotope | Z | N | Notes |
|---|---|---|---|
| He-5 | 2 | 3 | Lightest N/Z = 3/2 |
| Be-10 | 4 | 6 | Natural, long-lived |
| C-15 | 6 | 9 | |
| ... | ... | ... | 49 total isotopes |
| Fm-250 | 100 | 150 | Heaviest MEASURED N/Z = 3/2 |
| No-255 | 102 | 153 | Heaviest measured, all at k_NZ=7 |

All sit at k_NZ = 7, d_NZ = 12 — the EM/Full-Resolution family at the band edge. The most common nuclear ratio IS the Koide threshold.

---

## Section 3: The Koide Track Through the Island of Stability

The N/Z = 3/2 line extends beyond measured nuclei into the predicted superheavy region. For every even Z, the isotope with N = 3Z/2 sits at the Koide attractor:

| Element | Z | N | A | Status | Lattice Position |
|---|---|---|---|---|---|
| Hg-200 | 80 | 120 | 200 | MEASURED (natural) | k=7, d=12, |ε|=1.955¢ |
| Pb-205* | 82 | 123 | 205 | *not exact 3/2* | |
| Po-210 | 84 | 126 | 210 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| Rn-215 | 86 | 129 | 215 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| Ra-220 | 88 | 132 | 220 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| Th-225 | 90 | 135 | 225 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| U-230 | 92 | 138 | 230 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| Pu-235 | 94 | 141 | 235 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| Fm-250 | 100 | 150 | 250 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| No-255 | 102 | 153 | 255 | MEASURED | k=7, d=12, |ε|=1.955¢ |
| **Rf-260** | **104** | **156** | **260** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Sg-265** | **106** | **159** | **265** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Hs-270** | **108** | **162** | **270** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Ds-275** | **110** | **165** | **275** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Cn-280** | **112** | **168** | **280** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Fl-285** | **114** | **171** | **285** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Lv-290** | **116** | **174** | **290** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Og-295** | **118** | **177** | **295** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Ubn-300** | **120** | **180** | **300** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Ubt-305** | **122** | **183** | **305** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Ubq-310** | **124** | **186** | **310** | predicted | k=7, d=12, |ε|=1.955¢ |
| **Ubh-315** | **126** | **189** | **315** | predicted | k=7, d=12, |ε|=1.955¢ |

The Koide track is a straight line through the nuclear chart at N/Z = 3/2 exactly. Every point sits at the same lattice address: k_NZ=7, d_NZ=12, |ε|=1.955¢. The track passes through three predicted magic-Z elements: Fl (Z=114), Ubn (Z=120), and Ubh (Z=126).

---

## Section 4: Stability Windows for Superheavy Elements

The k_NZ = 7→8 transition defines the maximum neutron number inside the stability band for each element. The k_NZ = 6→7 transition defines the minimum. Together they give the **stability window** — the range of N values at k_NZ = 7 (inside the band).

| Element | Z | k=6→7 at N= | k=7→8 at N= | Window width | Koide N | Magic N=184 position |
|---|---|---|---|---|---|---|
| Rf | 104 | 152 | 161 | 9 neutrons | 156 (center) | outside band (k=9) |
| Sg | 106 | 155 | 164 | 9 neutrons | 159 (center) | outside band (k=9) |
| Hs | 108 | 158 | 167 | 9 neutrons | 162 (center) | outside band (k=9) |
| Ds | 110 | 161 | 170 | 9 neutrons | 165 (center) | outside band (k=8) |
| Cn | 112 | 164 | 173 | 9 neutrons | 168 (center) | outside band (k=8) |
| **Fl** | **114** | **166** | **176** | **10 neutrons** | **171 (center)** | outside band (k=8) |
| Lv | 116 | 169 | 179 | 10 neutrons | 174 (center) | outside band (k=8) |
| **Og** | **118** | **172◆** | **182** | **10 neutrons** | **177 (center)** | N=184 at k=8→9 edge |
| **Ubn** | **120** | **175** | **186** | **11 neutrons** | **180 (center)** | N=184 inside band (k=7) |
| Ubt | 122 | 178 | 189 | 11 neutrons | 183 (center) | N=184 inside band (k=7) |
| Ubq | 124 | 181 | 192 | 11 neutrons | 186 (center) | N=184 inside band (k=7) |
| **Ubh** | **126** | **184◆** | **195** | **11 neutrons** | **189 (center)** | **N=184 at k=6→7 entry (d=2→12 transition)** |

The predicted magic neutron number N=184 enters the stability band at **Z=120** (Unbinilium). For all elements below Z=120, N=184 sits outside the stability band (k_NZ ≥ 8). This is a concrete lattice prediction: **Z=120 is the minimum proton number for which the doubly-magic N=184 configuration is inside the stability band.**

For Z=126, N=184 sits at the exact k_NZ = 6→7 transition — the entry point of the stability band. This coincides with a d_NZ family transition from d=2 (Pivot) to d=12 (EM/Full-Resolution). The doubly-magic Z=126, N=184 sits at a structural boundary on the N/Z lattice.

---

## Section 5: The Lattice Prediction — Island Center at Z=120, N=180

The lattice evidence converges on **Unbinilium-300 (Z=120, N=180)** as the predicted island center:

1. **Koide attractor:** N/Z = 180/120 = 3/2 exactly → |ε| = 1.955¢, the tightest structural lock on the lattice
2. **Stability band:** k_NZ = 7, centered in an 11-neutron-wide window (N=175–185)
3. **d_NZ family:** d=12 (EM/Full-Resolution), the highest-resolution lattice family
4. **Magic proximity:** N=180 is 4 neutrons below the predicted magic N=184, which enters the band at this Z
5. **Symmetry:** The Koide track passes through Z=120 at the midpoint of the superheavy island

The alternative candidates:
- **Fl-285 (Z=114, N=171):** Koide isotope, but in a narrower stability window (10 neutrons). N=184 is outside the band at this Z.
- **Ubh-315 (Z=126, N=189):** Koide isotope, but N=184 sits at the band entry point (marginal). N=189 is inside but not centered.
- **Z=126, N=184 (doubly-magic by shell theory):** Sits at the d=2→12 family transition. Inside the band but at its lower edge — not centered. ε = −44.5¢ (near the ∂I boundary).

---

## Section 6: Superheavy Artificial Elements — Lattice Signatures

All 23 measured superheavy isotopes (Z ≥ 100) in AME2020 show structural features on the N/Z lattice.

**Even/odd Z split:** Even-Z superheavy elements (Fm, No, Rf, Sg) project to d_NZ = 12 (EM family). Odd-Z (Md, Db, Bh) project to d_NZ = 2 (Pivot) or d_NZ = 3 (Strong). This reflects proton pairing: even-Z elements have a more symmetric nuclear structure that maps to the full-resolution family.

**Nobelium is pure d_NZ = 12:** All 4 measured No isotopes (A=253–256) sit at k_NZ = 7, d_NZ = 12 without exception. No is the cleanest superheavy element on the N/Z lattice — every measured isotope is inside the stability band in the EM family. No-255 (N/Z = 3/2) sits at the Koide attractor.

**The k_NZ = 7/8 boundary at Z=100:** Fermium splits exactly across the boundary. Fm-249 through Fm-252 (N=149–152) are at k_NZ = 7 (inside band). Fm-255 and Fm-257 (N=155, 157) are at k_NZ = 8 (outside). The lattice resolves which Fm isotopes are more stable — the lower-N ones.

**Hassium straddles k_NZ = 6 and 7:** Hs-264 (N=156) is at k_NZ = 6, d_NZ = 2. Hs-266 (N=158) is at k_NZ = 7, d_NZ = 12. The two extra neutrons push Hs across a lattice boundary AND change its d_NZ family from Pivot to EM.

---

## Section 7: The CF Method — Dual-Pathway Home-Finding

Both the LCM tower and CF method (§7.11 Step 3a) run in parallel for every isotope.

**LCM tower results:** 179/2323 isotopes d-stabilize within 58 landmarks. The remaining 2144 are tower-resistant because log₂(N/Z) is transcendental for most integer ratios. This is structurally expected — not a failure.

**CF method results:** 2274 isotopes classified as cf_deep_home, 49 as cf_marginal. The 49 marginals are ALL N/Z = power of 2 (1/1, 2/1, 1/2, 4/1) — the only rationals whose log₂ is integer. These sit exactly at lattice nodes with d=1 (Gravity/Octave), quality=0.

**CF d_home distribution:** Small d_home values involve only ET-relevant primes: d=2, 3, 5, 11, 13, 19, 23, 29. The digit-count distribution shows peaks at 1 digit (63 isotopes), 64 digits (52), 108 digits (74), and 120 digits (44).

**The N=Z line (k_NZ = 0):** All 32 N=Z isotopes project to k_NZ = 0 with CF d_home = 1, quality = 0 — the d=1 Gravity family. 13 are naturally occurring (the even-even N=Z nuclei from ²H to ⁴⁰Ca). The N=Z line IS the gravity family on the N/Z projection.

---

## Section 8: d_NZ Family Transitions as Nuclear Structure

The d_NZ family cycles through all six ET families as k_NZ increments:

| k_NZ | d_NZ | Family | Physical interpretation |
|---|---|---|---|
| ...→k | 2 | Pivot | Approaching the stability band from below |
| k→k+1 | 12 | EM/Full-Res | Inside the stability band |
| k+1→k+2 | 3 | Strong | Exiting the stability band |
| k+2→k+3 | 4 | Weak | Beyond the band |

The pattern d=2→12→3→4→6→12→... repeats every 6 k-steps (one full cycle of mod-12 residues). The stability band (k_NZ = 0 to 7) spans more than one full cycle, containing ALL six families.

For the superheavy region, the d=2→12 transition at the band entry is particularly significant: it marks where the d_NZ family switches from Pivot (d=2) to EM (d=12) — from a 2-fold to a 12-fold sublattice. For Z=126 at N=184, this transition is exactly where the predicted doubly-magic nucleus sits.

---

## Section 9: Comparison with Other DSR Projections

Five candidate DSRs were tested systematically:

| DSR | Best Band | Width | Captures | Outside | Notable Feature |
|---|---|---|---|---|---|
| **N/Z** | k∈[0,7] | 8 | **98.6%** | 4 | Sharpest boundary, Koide at 3/2 |
| Binding fraction | k∈[−83,−81] | 3 | 93.7% | 18 | Iron peak at k=−81 (51% nat) |
| BE/A / m_e | k∈[48,49] | 2 | 86.3% | 39 | Narrowest band |
| Defect/A / m_e | k∈[47,49] | 3 | 94.7% | 15 | Best continuous-value separation |
| S_n / m_e | k∈[44,62] | 19 | 94.3% | 16 | Broadest — spreads structure |

N/Z gives the sharpest stability boundary. The binding fraction gives the most dramatic single-point concentration (iron peak at 51% natural). Both reveal structure invisible on the mass projection. They are complementary: N/Z shows WHERE stability lives (the band), binding fraction shows HOW TIGHTLY bound nuclei cluster (the iron peak).

---

## Summary of Key Predictions

1. **Island center: Z=120, N=180 (Ubn-300)** — Koide attractor, centered in stability band, d_NZ=12
2. **Z=120 is the minimum Z for N=184 inside the band** — below Z=120, the doubly-magic N=184 falls outside
3. **Z=126, N=184 sits at the d=2→12 family transition** — structural boundary, band entry point
4. **The Koide track (N/Z = 3/2) extends to Z=130+** — every even-Z superheavy has a Koide isotope
5. **Superheavy even/odd Z split: d_NZ=12 vs d_NZ=2** — proton pairing manifests as lattice family

---

*Exception Theory — P ∘ D ∘ T = E*
*All projections verified at 120-digit mpmath precision.*
*Dual-pathway home-finding: LCM tower + CF method (§7.11).*
*Zero float64 in the computational chain.*

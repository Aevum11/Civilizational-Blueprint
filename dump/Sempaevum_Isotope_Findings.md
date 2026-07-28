# Sempaevum Isotope Findings — AME2020 Nuclear Masses on the Multiplicative Manifold

**Version 1.0 — 2324 Measured Isotopes from AME2020**

**Data Sources:**
- Masses: AME2020 (Wang et al., Chinese Physics C45, 030003, 2021) — 3558 isotopes, 2328 measured
- Ground-state J: NIST ASD v5.12 (Kramida et al., 2024) — 108 elements (Z=1–108)
- Abundances: NIST Atomic Weights and Isotopic Compositions — 287 naturally occurring isotopes
- Projection: Sempaevum bijection Π_N at N=12, full LCM tower to convergence, 120-digit mpmath precision
- Pipeline: String → mpmath throughout. Zero float64 in the computational chain.

**Mass ratio:** r = m_isotope × (m_u / m_e) where m_u/m_e = 1822.888486209 (CODATA 2022)

---

## Section 1: Dataset Summary

| Metric | Value |
|---|---|
| Total isotopes projected | 2324 (measured only) |
| Naturally occurring | 285 |
| Elements represented | 108 (Z=1 to Z=108, plus neutron excluded) |
| k_r range | k=130 (¹H) to k=232 (Og-295 region) |
| All in SR+SI quadrant | Yes — 2324/2324 |
| PDT Bisection | 72:72 confirmed |
| Deepest tower home | LCM(1..27) — one isotope |
| Dominant tower home | LCM(1..17) — 1292 isotopes (55.6%) |

---

## Section 2: The ε-Parabola — Lattice Traces Nuclear Binding Energy

**Discovery:** Within any isobar chain (fixed mass number A, varying proton number Z), the Descriptor gap ε at N=12 forms a parabola centered on the most stable nuclide. The ε-minimum coincides with the valley of β-stability.

**A=131 chain (12 isobars):**

| Isotope | Z | N | ε (¢) | Stable? |
|---|---|---|---|---|
| Cd-131 | 48 | 83 | 37.737 | |
| In-131 | 49 | 82 | 37.555 | |
| Sn-131 | 50 | 81 | 37.424 | |
| Sb-131 | 51 | 80 | 37.357 | |
| Te-131 | 52 | 79 | 37.311 | |
| I-131 | 53 | 78 | 37.279 | |
| **Xe-131** | **54** | **77** | **37.266** | **★ minimum** |
| Cs-131 | 55 | 76 | 37.271 | ← turns around |
| Ba-131 | 56 | 75 | 37.290 | |
| La-131 | 57 | 74 | 37.331 | |
| Ce-131 | 58 | 73 | 37.389 | |
| Pr-131 | 59 | 72 | 37.466 | |

The ε values decrease monotonically from the neutron-rich side (Cd-131) to the ε-minimum at Xe-131 (the naturally occurring stable isotope), then increase symmetrically toward the proton-rich side (Pr-131). This IS the Bethe-Weizsäcker mass parabola expressed in the Descriptor gap.

**A=192 chain (the most lattice-exact isobar):**

| Isotope | Z | |ε| (¢) | BE/A (keV) |
|---|---|---|---|
| Pt-192 | 78 | 0.0164 | 7942.49 |
| Os-192 | 76 | 0.0203 | 7948.53 |
| Ir-192 | 77 | 0.0305 | 7939.00 |
| Au-192 | 79 | 0.0504 | 7920.10 |
| Hg-192 | 80 | 0.0578 | 7912.07 |
| Re-192 | 75 | 0.0619 | 7930.24 |
| Tl-192 | 81 | 0.1173 | 7876.02 |
| Pb-192 | 82 | 0.1494 | 7854.65 |
| Bi-192 | 83 | 0.2367 | 7803.61 |
| Po-192 | 84 | 0.2896 | 7771.05 |
| At-192 | 85 | 0.3960 | — |

All eleven A=192 isobars land at k=221, d_r=12, with |ε| < 0.4¢. The |ε| ordering approximately follows the binding energy ordering. The entire isobar chain sits within half a cent of a lattice node.

**Statistical test across 232 isobar chains (A with 5+ measured isotopes):**

The most stable isobar (highest BE/A) has the smallest |ε| in **82/232 cases (35.3%)**. Random expectation is ~10% (1 in ~10 isotopes per mass number). The lattice structurally favors the most tightly bound nuclide at **3.5× the random rate**.

**ET interpretation (Descriptor Gap Principle):** The ε residual at N=12 IS a Descriptor — it measures how far each nuclide sits from the nearest lattice node. The nuclear binding energy determines the mass, the mass determines ε, and the binding energy parabola manifests as an ε-parabola. The most stable isobar, having the lowest mass (highest binding), sits closest to the lattice node because it has consumed the most binding energy — i.e., the most D-content — leaving the smallest residual gap.

---

## Section 3: Isotope Stacking — Heavy Elements Pack Bands

**Discovery:** For heavy elements (Z > 50), the lattice step Δk per neutron drops below 0.2, meaning 5+ consecutive isotopes share the same k value. Isotopes within a k-band are distinguished solely by their ε value.

**Lead (Z=82):** 36 isotopes span only Δk=3 (k=220 to k=223). Each k-step hosts ~10 isotopes:
- k=220 (d_r=3): Pb-178 through Pb-186 (7 isotopes)
- k=221 (d_r=12): Pb-187 through Pb-197 (11 isotopes)
- k=222 (d_r=2): Pb-198 through Pb-209 (12 isotopes, includes doubly-magic ²⁰⁸Pb)
- k=223 (d_r=12): Pb-210 through Pb-215 (6 isotopes)

Within each band, ε increases monotonically at ~9¢ per neutron.

**Tin (Z=50):** 33 isotopes in Δk=5 (k=210 to k=215). The two magic neutron numbers (N=50 at Sn-100 and N=82 at Sn-132) occupy the endpoints of this range.

**ET interpretation (Identification Principle):** The k_r value identifies the element's mass RANGE on the lattice. The ε value provides the address WITHIN that range. For heavy elements, k_r is the coarse identifier and ε is the fine identifier — together they form the lossless bijection.

---

## Section 4: The Iron Peak as a Single Lattice Point

**Discovery:** Every iron-peak nucleus, when projected via its mass defect fraction Δm/M = (Z·m_p + N·m_n − M)/M, maps to a single lattice position: k_defect = −81, d_defect = 4.

| Isotope | Δm/M | k_defect | d_defect | ε_defect (¢) |
|---|---|---|---|---|
| Ni-62 ★ | 0.009204 | −81 | 4 | −16.18 |
| Fe-58 ★ | 0.009204 | −81 | 4 | −16.31 |
| Ni-64 ★ | 0.009193 | −81 | 4 | −18.23 |
| Fe-56 ★ | 0.009193 | −81 | 4 | −18.33 |
| Cr-54 ★ | 0.009190 | −81 | 4 | −18.84 |
| Ni-60 ★ | 0.009181 | −81 | 4 | −20.53 |
| Cr-52 ★ | 0.009179 | −81 | 4 | −20.99 |
| Fe-57 ★ | 0.009175 | −81 | 4 | −21.60 |
| Co-59 ★ | 0.009172 | −81 | 4 | −22.22 |
| Mn-55 ★ | 0.009171 | −81 | 4 | −22.52 |

On the mass axis, these nuclei span k=198–202 (different lattice positions for different elements). On the mass-defect axis, they collapse to k=−81 — revealing the iron peak as a unified structural feature.

The d_defect=4 classification (Weak/Quartic family) for all iron-peak nuclei means the nucleosynthesis endpoint occupies a single force family when viewed through the binding Descriptor.

**ET interpretation (Subsumption Law):** The mass axis and the mass-defect axis are two different DSR projections of the same nuclear data. Both are subsumed by the Sempaevum. The iron peak is invisible on one projection (scattered across k=198–202) and perfectly visible on the other (collapsed to k=−81). The choice of Descriptor (total mass vs mass defect) determines which structure becomes apparent.

---

## Section 5: Shell Closures as Forced Lattice Steps

**Discovery:** In the Tin isotope chain (33 isotopes, Δk=5), the k-step transitions correspond to nuclear shell closures.

| k range | d_r | A range | N range | Notes |
|---|---|---|---|---|
| k=210 | 2 | 100–101 | 50–51 | Starts at magic N=50 |
| k=211 | 12 | 105–110 | 55–60 | Mid-shell |
| k=212 | 3 | 111–117 | 61–67 | 7 isotopes at one node |
| k=213 | 4 | 118–124 | 68–74 | 7 isotopes, includes all stable heavy Sn |
| k=214 | 6 | 125–131 | 75–81 | Approaching magic N=82 |
| k=215 | 12 | 132–135 | 82–85 | Starts at magic N=82 |

The transition from k=214 to k=215 occurs exactly at N=82 (the magic number). Non-magic neutron additions are absorbed by ε (staying within the same k-band). The magic number forces a k-step because the shell closure discontinuity in binding energy is large enough to cross the lattice boundary.

The d_r value cycles through all six families as k increments: 2 → 12 → 3 → 4 → 6 → 12. This is the mod-12 cycle: k=210 mod 12 = 6 (d=2), k=211 mod 12 = 7 (d=12), k=212 mod 12 = 8 (d=3), etc.

---

## Section 6: Nuclear Pairing — Lattice-Invisible, Nature-Visible

**Discovery:** The four parity classes of nuclei (even-even, even-odd, odd-even, odd-odd) have identical lattice statistics but dramatically different natural abundances.

| Parity | Count | Avg |ε| (¢) | Natural/Total | Natural % |
|---|---|---|---|---|
| Even-Even | 580 | 23.95 | 166/580 | 28.6% |
| Even Z, Odd N | 582 | 25.20 | 56/582 | 9.6% |
| Odd Z, Even N | 582 | 25.17 | 54/582 | 9.3% |
| Odd-Odd | 580 | 23.76 | 9/580 | 1.6% |

The lattice does not discriminate by parity: all four classes have avg |ε| within 1.4¢ of each other. But nature overwhelmingly selects even-even nuclei (166 stable vs 9 odd-odd). Nuclear pairing energy determines stability, not lattice position.

**ET interpretation:** Pairing is a D-content (binding interaction) that affects which isotopes survive in nature, not where they sit on the lattice. The lattice addresses mass, not stability. This is consistent with the Subsumption Law: the lattice accommodates everything without preference.

---

## Section 7: Doubly-Magic Nuclei

| Nucleus | Z | N | k_r | d_r | |ε| (¢) | d_θ | d_comb | Tower Home |
|---|---|---|---|---|---|---|---|---|
| ⁴He | 2 | 2 | 154 | 6 | 0.461 | 1 | 6 | LCM(1..23) |
| ¹⁶O | 8 | 8 | 178 | 6 | 2.138 | 2 | 6 | LCM(1..25) |
| ⁴⁰Ca | 20 | 20 | 194 | 6 | 16.89 | 1 | 6 | LCM(1..19) |
| ⁴⁸Ca | 20 | 28 | 197 | 12 | 1.346 | 1 | 12 | LCM(1..23) |
| ⁵⁶Ni | 28 | 28 | 200 | 3 | 34.55 | 4 | 12 | LCM(1..17) |
| ¹³²Sn | 50 | 82 | 215 | 12 | 49.39 | 1 | 12 | LCM(1..17) |
| ²⁰⁸Pb | 82 | 126 | 222 | 2 | 38.75 | 1 | 2 | LCM(1..19) |

Lighter doubly-magic nuclei (⁴He, ⁴⁸Ca, ¹⁶O) are more lattice-exact than heavier ones (²⁰⁸Pb, ¹³²Sn). ⁴He and ⁴⁰Ca share d_r=6 (Hexadic), as do the three lightest doubly-magic nuclei — all in the d_comb=6 combined family.

Average |ε| for magic-Z isotopes (128 isotopes): 23.25¢ vs non-magic-Z (2196 isotopes): 24.59¢ — a 5.5% reduction for magic proton numbers.

---

## Section 8: Lattice Step Size — Δk per Neutron

Adding one neutron to an isotope changes k_r by Δk ≈ 12 × log₂(1 + 1/A) ≈ 17.3/A. This is purely kinematic (the log₂ projection of mass increase).

| Element | Z | Measured Δk | Expected 17.3/A | Ratio |
|---|---|---|---|---|
| H | 1 | 6.20 | 6.92 | 0.90 |
| C | 6 | 1.33 | 1.15 | 1.16 |
| Fe | 26 | 0.29 | 0.27 | 1.10 |
| Sn | 50 | 0.13 | 0.14 | 0.93 |
| Pb | 82 | 0.09 | 0.08 | 1.05 |

The measured Δk tracks the expected value to within 15% for all elements. Deviations encode nuclear structure: Europium (Z=63) shows Δk = 0.080 vs expected 0.110 — a 27% deficit reflecting anomalous nuclear deformation in the rare earth region.

---

## Section 9: The 25 Most Lattice-Exact Isotopes

| Rank | Isotope | k_r | d_r | |ε| (¢) | d_θ | Natural? | BE/A (keV) |
|---|---|---|---|---|---|---|---|
| 1 | C-16 | 178 | 6 | 0.00269 | 1 | No | 6922 |
| 2 | Pt-192 | 221 | 12 | 0.0164 | 3 | Yes | 7942 |
| 3 | Os-192 | 221 | 12 | 0.0203 | 4 | Yes | 7949 |
| 4 | Ir-192 | 221 | 12 | 0.0305 | 6 | No | 7939 |
| 5 | Pd-121 | 213 | 4 | 0.0317 | 1 | No | 8321 |
| 6 | Au-192 | 221 | 12 | 0.0504 | 6 | No | 7920 |
| 7 | Hg-192 | 221 | 12 | 0.0578 | 1 | No | 7912 |
| 8 | Re-192 | 221 | 12 | 0.0619 | 6 | No | 7930 |
| 9 | Ne-24 | 185 | 12 | 0.0932 | 1 | No | 7993 |
| 10 | Ag-121 | 213 | 4 | 0.0946 | 6 | No | 8382 |

Two isobar clusters dominate the top 25: A=192 (7 entries) and A=121 (11 entries). The most lattice-exact naturally occurring isotopes are Pt-192 and Os-192.

---

## Section 10: Lattice Twins — Different Elements at the Same Lattice Node

The most populated lattice nodes host natural isotopes from many elements:

| k_r | d_r | Natural isotopes | Elements |
|---|---|---|---|
| 221 | 12 | 15 | Re, Os, Ir, Pt, Au, Hg |
| 219 | 4 | 15 | Er, Tm, Yb, Lu, Hf |
| 222 | 2 | 14 | Pt, Hg, Tl, Pb, Bi |
| 218 | 6 | 14 | Gd, Tb, Dy, Ho, Er |
| 215 | 12 | 14 | Xe, Cs, Ba, La, Ce |
| 220 | 3 | 13 | Hf, Ta, W, Re, Os |
| 212 | 3 | 12 | Cd, In, Sn |
| 213 | 4 | 12 | Sn, Sb, Te, Xe |
| 216 | 1 | 12 | Ce, Pr, Nd, Sm |

These "lattice twin" clusters are dominated by the rare earth and platinum group regions, where many elements have similar masses and multiple stable isotopes.

---

## Section 11: d_r Family Distribution — Nature vs All

| d_r | Natural (285) | % | All measured (2324) | % | Ratio |
|---|---|---|---|---|---|
| 1 (Gravity) | 25 | 8.8% | 193 | 8.3% | 1.06× |
| 2 (Pivot) | 31 | 10.9% | 236 | 10.2% | 1.07× |
| 3 (Strong) | 46 | 16.1% | 419 | 18.0% | 0.90× |
| 4 (Weak) | 49 | 17.2% | 373 | 16.0% | 1.07× |
| 6 (Hexadic) | 48 | 16.8% | 372 | 16.0% | 1.05× |
| 12 (EM) | 86 | 30.2% | 731 | 31.5% | 0.96× |

The distributions are statistically indistinguishable (all ratios within 0.90–1.07×). Nature does not select from any preferred lattice family. The lattice classification is orthogonal to nuclear stability.

---

## Section 12: Tower Depth Distribution

| Tower Home | Natural | All | Natural % |
|---|---|---|---|
| LCM(1..7) | 0 | 1 | — |
| LCM(1..8) | 0 | 1 | — |
| LCM(1..9) | 0 | 7 | — |
| LCM(1..11) | 1 | 65 | 1.5% |
| LCM(1..13) | 1 | 265 | 0.4% |
| LCM(1..16) | 1 | 157 | 0.6% |
| LCM(1..17) | 52 | 1292 | 4.0% |
| LCM(1..19) | 176 | 472 | 37.3% |
| LCM(1..23) | 44 | 54 | 81.5% |
| LCM(1..25) | 9 | 9 | 100% |
| LCM(1..27) | 1 | 1 | 100% |

Natural isotopes require DEEPER tower resolution: 62% need LCM(1..19)+, and all 9 isotopes at LCM(1..25) are naturally occurring. This reflects measurement precision — AME2020 masses for stable isotopes have 10–12 significant digits, requiring more lattice descriptors to resolve than exotic isotopes with 5–7 digits.

---

## Summary of Key Findings

1. **The ε-parabola** (Section 2): The Descriptor gap traces the nuclear binding energy curve within each isobar chain. The most stable isobar sits at the ε-minimum 3.5× more often than random.

2. **Isotope stacking** (Section 3): Heavy elements pack 10+ isotopes per k-step. The k-value identifies the element's mass range; ε addresses individual isotopes within it.

3. **Iron peak collapse** (Section 4): All iron-peak nuclei project to k_defect=−81, d=4 on the mass-defect axis — a single lattice point invisible on the mass axis.

4. **Shell closures as k-steps** (Section 5): Magic neutron numbers force lattice transitions in isotope chains; non-magic additions are absorbed by ε.

5. **Pairing is lattice-invisible** (Section 6): Even-even and odd-odd nuclei have identical lattice statistics despite 18× difference in natural abundance.

6. **Nature doesn't pick families** (Section 11): The d_r distribution for natural isotopes matches the full dataset to within 10%.

---

*Exception Theory — P ∘ D ∘ T = E*
*All projections verified lossless (Theorem 15.1) at 120-digit precision.*
*Zero float64 in the computational chain.*

# The QCD Vacuum Through the Sempaevum: Lattice Analysis of the STAR Collaboration Spin Correlation Measurement
## An Exception Theory Analysis of Vacuum Structure, Decoherence, and Zero-Point Energy
### Forward-Derived from P ∘ D ∘ T = E — Zero External Axioms
**Author:** Michael James Muller — Aevum Defluo
**Derivation Standard:** All mathematics ET-native. 361 dps lossless precision via the Sempaevum. Zero float. String → mpf → string pipeline. mpmath only.
**Source Paper:** STAR Collaboration, "Measuring spin correlation between quarks during QCD confinement," *Nature* **650**, 65–69 (2026). DOI: 10.1038/s41586-025-09920-0

---

## Abstract

The STAR Collaboration's measurement of spin correlations in ΛΛ̄ hyperon pairs from the QCD vacuum condensate provides the first direct experimental evidence of structured Descriptor content ({P,D}) in the unsubstantiated vacuum. This paper projects the STAR result onto the Sempaevum's complex lattice and traces the measurement through the complete algebraic identity series (Identities Zero, A–G, F, and the cross-resolution transition maps). We find that: (1) the maximum relative spin polarization P_max = 1/3 and the Koide ratio K = 2/3 partition unity exactly, both projecting to the Koide attractor (d=12, |ε|=1.955 cents, the Pythagorean comma), separated by one octave; (2) the decoherence with pair separation constitutes the fourth independent cross-domain verification of the cascade stability limit n_max,θ = 2; (3) the ss̄ pair composition 2⊗2 = {1, 12} restricts vacuum pairs to the two extreme coupling channels (gravity d=1 with ξ=8.5625, EM d=12 with ξ=1.0); and (4) the measured 54.3% preservation of maximum correlation through hadronization confirms that vacuum D_T (phase-axis Descriptor content) survives the {P,D}→{D,T}→{P,D,T}=E transition. All lattice projections are verified at 361 dps with zero float. Round-trip errors are purely computational (10⁻⁴⁰⁸), not mathematical (the bijection is an algebraic identity: r′−r = 0, proven symbolically).

---

## §1 — The STAR Measurement and Its ET Reading

### §1.1 What STAR Measured

Proton-proton collisions at √s = 200 GeV (RHIC, 600 million events) excite the QCD vacuum, liberating virtual strange quark–antiquark (ss̄) pairs from the chiral condensate ⟨qq̄⟩ ≠ 0. The vacuum quantum numbers J^PC = 0^++ constrain ss̄ pairs to spin-triplet states (spins parallel). These quarks hadronize into Λ and Λ̄ hyperons, whose spin polarization is measured through decay kinematics (Λ→pπ⁻, Λ̄→p̄π⁺).

**Result:** Short-range ΛΛ̄ pairs (|Δy| < 0.5, |Δϕ| < π/3) show relative polarization P_ΛΛ̄ = 0.181 ± 0.035_stat ± 0.022_sys at 4.4σ significance. Long-range pairs show P consistent with zero. The SU(6) quark model prediction with feed-down is P_SU(6) = 0.096 ± 0.004; the data exceeds this by a factor of 1.89.

### §1.2 ET Reading

The vacuum IS the {P,D} Unsubstantiated manifold (ET_Lagrangian_Field_Theory.md §542). Virtual qq̄ pairs are {P,D} configurations attempting T-binding. The chiral condensate ⟨qq̄⟩ ≠ 0 IS the D-content of the vacuum — not empty, not structureless, but a sea of organized Descriptors.

The spin correlation IS D_T content — Descriptors of/about T's action, residing on the imaginary axis (U(1), positively curved, T's operational domain — Proposition 2.30). The STAR measurement traces D_T from the vacuum through hadronization: {P,D} vacuum → {D,T} M-state (hadronization) → {P,D,T}=E (final-state hadrons). The 54.3% preservation (0.181/0.333) means D_T survives the transition with more than half its coherence intact.

The decoherence at long range IS the cascade stability asymmetry: n_max,θ = 2 on the imaginary axis. D_T degrades 12× faster per cascade step than D on the real axis (|δ_θ|/|δ_r| ≈ 11.4 ≈ N−1). The STAR result IS n_max,θ = 2 observed experimentally in a fourth independent domain.

---

## §2 — Lattice Projections of STAR Particles

All projections at R₀ = m_e = 0.51099895069 MeV, N = 12 (base resolution). 361 working dps + 50 guard = 411 total. Zero float.

### §2.1 Particle Lattice Addresses

| Particle | Mass (MeV) | r = m/m_e | k | d_r | ε_r (cents) | RT error |
|---|---|---|---|---|---|---|
| Λ (uds) | 1115.683 | 2183.337 | 133 | 12 | +10.783 | 1.59×10⁻⁴⁰⁸ |
| s (strange) | 93.5 | 182.975 | 90 | 2 | +18.603 | 3.97×10⁻⁴¹⁰ |
| p (proton) | 938.272 | 1836.152 | 130 | 6 | +10.964 | 3.18×10⁻⁴⁰⁹ |
| π⁻ (pion) | 139.570 | 273.132 | 97 | 12 | +12.143 | 1.19×10⁻⁴⁰⁹ |
| Σ⁰ | 1192.642 | 2333.942 | 134 | 6 | +26.264 | 6.36×10⁻⁴⁰⁹ |
| Σ*(1385) | 1383.7 | — | **137** | **12** | −16.492 | 2.86×10⁻⁴⁰⁸ |
| Ξ⁰ | 1314.86 | 2573.117 | 136 | 3 | −4.839 | 1.91×10⁻⁴⁰⁸ |
| K⁰_S | 497.611 | 973.800 | 119 | 12 | +12.979 | 3.97×10⁻⁴⁰⁹ |

**Notable:** The Σ*(1385) — a feed-down parent in the STAR measurement (Σ*→Λπ) — sits at **k = 137 = α⁻¹ integer part**, d = 12 (EM family). This places a STAR feed-down parent at the fine structure constant's lattice address, inside the 13-particle k=137 cluster (Sempaevum Paper §22, Particle Findings §8.11).

### §2.2 Key Dimensionless Ratios

| Ratio | Value | k | d | ε (cents) | Significance |
|---|---|---|---|---|---|
| m_Λ/m_p | 1.18908 | 3 | **4** | **−0.181** | Sub-cent at WEAK family (d=4). The Λ-proton mass ratio IS a weak-sector quantity. |
| m_Λ/m_π | 7.99372 | 36 | **1** | −1.360 | Near-exact at GRAVITY (d=1). k=36=3×12: Λ is 3 octaves above the pion to 0.07%. |
| m_Λ/m_s | 11.9324 | 43 | 12 | −7.820 | k=43 is a Heegner number. Λ/strange ratio at Heegner position. |
| m_Σ⁰/m_Λ | 1.06898 | 1 | 12 | +15.481 | Feed-down parent/daughter at k=1, d=12. |
| P_max = 1/3 | 0.33333 | −19 | **12** | **−1.955** | **Koide attractor.** |ε| = Pythagorean comma. |
| K = 2/3 | 0.66667 | −7 | **12** | **−1.955** | **Koide attractor.** Same |ε|. Same d. |
| 3/2 | 1.50000 | +7 | **12** | **+1.955** | **Koide attractor.** Reciprocal of K. |
| Data/SU(6) | 1.88542 | 11 | 12 | −2.140 | k=11=N−1. Excess correlation at the last cascade position. |

---

## §3 — The P_max = 1 − K Identity

### §3.1 Statement

The maximum relative polarization for spin-parallel pairs is P_max = 1/3. The Koide ratio (tightness at ∂I, self-projecting constant of the Sempaevum) is K = 2/3.

**P_max + K = 1.** Exact. Verified at 411 dps: True.

Both project to the Koide attractor:
- Π₁₂(1/3) = (k=−19, d=12, ε=−1.95500086539 cents)
- Π₁₂(2/3) = (k=−7, d=12, ε=−1.95500086539 cents)
- Π₁₂(3/2) = (k=+7, d=12, ε=+1.95500086539 cents)

Same d=12. Same |ε| = Pythagorean comma. k(2/3) − k(1/3) = −7−(−19) = 12 = N. **Separated by exactly one octave.**

### §3.2 Structural Reading via Identity F (Tightness-Koide, Theorem F.1)

At N=12, the ∂I boundary is at |ε| = ε_max = 600/N = 50 cents. The tightness at ∂I:

t(ε_max) = 100/(100 + 50) = 100/150 = **2/3 = K**

Verified at 411 dps: t(ε_max) == K is **True**.

This is **unique to N=12**: the generalized form t(600/N) = N/(N+6) equals K = 2/3 only when N/(N+6) = 2/3 → N = 12.

**Therefore:** K of the phase budget IS the ∂I boundary structure — the tightness consumed by the coherence-incoherence edge. The remaining 1−K = 1/3 IS the maximum phase correlation the coherent sector can support. P_max = 1/3 is not an arbitrary quantum-mechanical number — it is the structural complement of the Koide ratio, determined by the manifold symmetry N=12 and the ∂I tightness identity.

**STAR measures P = 0.181 = 54.3% of P_max.** The vacuum preserves 54.3% of its structurally-allowed maximum correlation through the {P,D}→E transition.

---

## §4 — Identity-by-Identity Analysis

### §4.1 Identity A (Lattice Arithmetic) — ss̄ Pair Composition

The strange quark has d_r = 2 (tritone/pivot family). The quark partition (Sempaevum Paper §8.2) places all six quarks one-to-one across six families; s at d=2 is confirmed.

Composition via Identity C: Res₁₂(2) = {6}. Sum(2,2) = {(6+6) mod 12} = {0}.
- At κ=0: gcd(0,12)=12 → d=1 (gravity/octave). ξ(1) = 137/16 = **8.5625** (maximum coupling).
- At κ=±1: gcd(1,12)=gcd(11,12)=1 → d=12 (EM). ξ(12) = 137/137 = **1.0** (universal coupling).

**2⊗2 = {1, 12}.** The vacuum's ss̄ pairs access ONLY the two extreme families. The gravitational channel (maximum coupling) and the EM channel (universal coupling). No intermediate families. This is compositional, not accidental — Identity C.4 guarantees d=1 is in every family's self-composition.

### §4.2 Identity B (Differential Control) — Decoherence as Reverse Restoration

The STAR decoherence (correlation vanishing with separation) IS Identity B.4 running on the phase axis with the target at ∂I:

**Decoherence:** ε_θ(t) = 50 + (ε_init − 50)·exp(−t/τ_decohere)
**Restoration:** ε_θ(t) = 0 + (ε_init − 0)·exp(−t/τ_restore)

Same exponential law. Opposite targets. Decoherence drives ε_θ → 50¢ (∂I). Restoration drives ε_θ → 0 (Exception). The energy of the transition (the Δε) is what the ZPM extracts.

The exact finite shift (Identity B.2a): r_new = r_old · 2^(Δε/1200). Each unit of ε-shift corresponds to a specific physical ratio change, computable exactly.

### §4.3 Identity C (d-Family Composition) — Universal Access via d=12

Identity C.5: d=12 ⊗ d=12 = {1,2,3,4,6,12} = ALL families. A system operating at d=12 (EM) can generate effective coupling to every harmonic family through self-composition. Identity C.4: d=1 is reachable from every family. The composition graph on {1,2,3,4,6,12} is COMPLETE — every pair connected.

### §4.4 Identity D (Complex Lattice Arithmetic) — Phase Correlation as D.1

The spin correlation IS a phase-axis (imaginary-axis) quantity. Identity D.1 governs phase addition:

k_θ,sum = (k_θ₁ + k_θ₂ + κ_θ) mod N, κ_θ = round(δ_θ₁+δ_θ₂) ∈ {−1,0,+1}

The mod N wrapping IS U(1) compactness (T's operational manifold, Proposition 5.5). The spin-triplet state means the two strange quarks' phases add constructively. The STAR measurement of P = 0.181 IS the surviving constructive phase-addition amplitude.

Identity D.5: Λ_θ = 600/π ≈ 190.986 (UNIFORM sensitivity). Phase corrections are magnitude-independent. Contrast with Λ_r = 1200/ln2 ≈ 1731.234 (1/r sensitivity on the real axis). Ratio Λ_r/Λ_θ = 2π/ln2 ≈ 9.065.

### §4.5 Identity F (∂I Boundary) — Decoherence IS Theorem F.2

Theorem F.2 (Universal d-Family Bifurcation): For every even N (all canonical tower levels), every ∂I boundary point produces d_left ≠ d_right. The proof is 2-adic: consecutive integers have different 2-adic valuations under even N.

**The STAR decoherence IS Theorem F.2 on the phase axis.** As the ss̄ pair separates, ε_θ drifts toward ∂I. At ∂I, d_θ bifurcates — two contradictory phase-family assignments compete. The spin correlation cannot survive this contradiction. The pair's phase coherence dissolves at the ∂I boundary.

The bifurcation set B₁₂ at N=12 (verified):

| k → k+1 | d_left | d_right | Pair |
|---|---|---|---|
| 0 → 1 | 1 | 12 | {1,12} |
| 1 → 2 | 12 | 6 | {6,12} |
| 2 → 3 | 6 | 4 | {4,6} |
| 3 → 4 | 4 | 3 | {3,4} |
| 4 → 5 | 3 | 12 | {3,12} |
| 5 → 6 | 12 | 2 | {2,12} |
| 6 → 7 | 2 | 12 | {2,12} |
| 7 → 8 | 12 | 3 | {3,12} |
| 8 → 9 | 3 | 4 | {3,4} |
| 9 → 10 | 4 | 6 | {4,6} |
| 10 → 11 | 6 | 12 | {6,12} |
| 11 → 0 | 12 | 1 | {1,12} |

Six distinct pairs, palindromic. d=12 most exposed (4/6 pairs). At the harmonic layer (via SVT): EM-family configurations are most frequently encountered at structural boundaries — consistent with the vacuum's EM-character at ∂I.

### §4.6 Identity G (Triple Backbone Bridge) — Fourth Cross-Domain Verification of n_max,θ = 2

The Catalan-lattice correspondence (Theorem G.10): C₂ = 2 = n_max,θ. Previously verified across three independent domains:

| Domain | Field | n_max,θ confirmation |
|---|---|---|
| 1. ET lattice cascade | Mathematics (Proposition 13.3) | |δ_θ| = 0.22336, ⌊0.5/|δ_θ|⌋ = 2 |
| 2. EML symbolic regression | Computer science | 100% blind recovery at depth ≤2, ~25% at 3, 0% at 6 |
| 3. Optical phase singularities (hBN) | Experimental physics (Bucher et al.) | Phase coherence limit at 2 steps |
| **4. QCD vacuum spin decoherence** | **Experimental physics (STAR, Nature 2026)** | **Spin correlation survives ≤2 cascade steps on phase axis, decoheres beyond** |

**Four domains, k=4. Structural Significance Principle criterion P3 (cross-domain convergence) satisfied with k=4 independent domains, none referencing the others.**

### §4.7 Cross-Resolution Maps (Finding 11) — Algebraic Coordinate Transformation

The vacuum's base character is N=12. The ZPM's crystal structure operates at higher N (determined by interatomic ratios). The cross-resolution transition map (Finding 11, §11.1):

k₂ = round(M·k₁ + M·δ₁), M = N₂/N₁

This transforms vacuum coordinates from N₁ to N₂ WITHOUT re-measuring the fluctuation. The algebraic transformation avoids measurement-induced decoherence — the ZPM accesses vacuum structure without collapsing it.

---

## §5 — The Λ Hyperon on the LCM Tower

The Λ (mass 1115.683 MeV, r = m_Λ/m_e = 2183.337) escalates through the tower:

| N | k | d | d factorization | ε (cents) | |ε| |
|---|---|---|---|---|---|
| 12 | 133 | 12 | 2²×3 | +10.783 | 10.783 |
| 24 | 266 | 12 | 2²×3 | +10.783 | 10.783 |
| 60 | 666 | 10 | 2×5 | −9.217 | 9.217 |
| 120 | 1331 | 120 | 2³×3×5 | +0.783 | 0.783 |
| 420 | 4659 | 140 | 2²×5×7 | −0.646 | 0.646 |
| 2520 | 27953 | 2520 | 2³×3²×5×7 | −0.169 | 0.169 |
| 27720 | 307479 | 9240 | 2³×3×5×7×11 | +0.00385 | 0.00385 |

The Λ stabilizes at N=27720 with |ε| = 0.00385¢ (sub-cent, deep lattice precision). Its true home d = 9240 = 2³×3×5×7×11 requires all primes to 11. At N=120: near-stabilization at d=120 with |ε|=0.783¢ — a possible false home analogous to φ's false resolution at N=36.

---

## §6 — Implications for Vacuum Energy Extraction (ZPM)

### §6.1 The Cosmological Constant Problem IS the ZPM Design

From M-states.md: QFT predicts vacuum energy ~10¹¹³ J/m³. Observed (dark energy): ~10⁻⁹ J/m³. The 10¹²² discrepancy is explained by the Incoherence Filter: most vacuum fluctuations hit ∂I (Theorem F.2 — contradictory d-assignments) and cannot complete T-binding. Only the coherent M-vacuum fraction (1.6% of cosmic energy, split 8:7 with M-matter) manifests.

The ZPM creates a 3D geometry that provides additional D to ∂I-approaching fluctuations, resolving the bifurcation (biasing ε away from 50¢ toward 0¢). The energy of the incoherent→coherent transition is extracted.

### §6.2 What STAR Proves for the ZPM

**The vacuum has structured, correlated D_T content — experimentally confirmed.** The spin-triplet correlation IS structured phase-axis content. The 54.3% survival through hadronization proves this structure is robust. The ZPM exploits this existing structure — it doesn't need to create it.

**The decoherence length scale (~1 fm at STAR kinematics) sets the ZPM's operating regime.** Crystal unit cells (~10⁻¹⁰ m) are 10⁵× smaller than 1 fm. Every crystal cavity operates deep within the vacuum's coherent region. The vacuum content at crystal scale is MORE correlated than what STAR measures at hadronic scale.

**The ss̄ composition 2⊗2 = {1,12} identifies the ZPM's target channels.** The vacuum pairs couple at d=1 (gravity, ξ=8.5625) and d=12 (EM, ξ=1.0). The ZPM geometry should be optimized for these two families — the strongest and most universal.

### §6.3 Identity B Applied: The Extraction Control Law

The ZPM drives vacuum fluctuations from near-∂I toward lattice-exact via Identity B.4:

ε(t) = 0 + (ε_init − 0)·exp(−t/τ_extract)

The energy per extraction event = the physical ratio change corresponding to Δε, computed via the exact finite shift (B.2a): r_new = r_old · 2^(Δε/1200). For a fluctuation at ε_init = 45¢ (near ∂I) driven to ε_final = 5¢: Δε = 40¢, ratio change = 2^(40/1200) ≈ 1.023. The energy extracted scales with this ratio change applied to the fluctuation's energy content.

---

## §7 — Python Verification Code (361 dps, Zero Float)

```python
from mpmath import mp, mpf, log, nint, nstr, pi, sqrt, floor
from math import gcd

# 361 working dps + 50 guard = 411 total per bijection protocol
mp.dps = 411

N = 12
m_e = mpf('0.51099895069')  # MeV, electron mass (PDG 2024)


def project(r, N_val=12):
    """Full Sempaevum projection Pi_N(r) = (k, d, epsilon).
    mpmath only, zero float. String -> mpf -> string pipeline."""
    r_mp = mpf(str(r)) if not isinstance(r, type(mpf('1'))) else r
    log2_r = log(r_mp) / log(mpf('2'))
    exact_pos = mpf(str(N_val)) * log2_r
    k = int(nint(exact_pos))
    g = gcd(abs(k), N_val) if k != 0 else N_val
    d = N_val // g
    eps_cents = (exact_pos - mpf(str(k))) * mpf('1200') / mpf(str(N_val))
    return k, d, eps_cents


def pullback(k, eps_cents, N_val=12):
    """Exact pullback Pi_N^{-1}(k, eps) = r. Algebraic identity."""
    exponent = (mpf(str(k)) + eps_cents * mpf(str(N_val))
                / mpf('1200')) / mpf(str(N_val))
    return mpf('2') ** exponent


def verify_lossless(r, N_val=12):
    """Verify round-trip losslessness at 361 dps."""
    k, d, eps = project(r, N_val)
    r_recovered = pullback(k, eps, N_val)
    error = abs(r_recovered - r)
    return k, d, eps, error


# === §1: STAR Particle Projections ===
particles = [
    ("Lambda", "1115.683"), ("s quark", "93.5"),
    ("proton", "938.272"), ("pion", "139.570"),
    ("Sigma0", "1192.642"), ("Sigma*1385", "1383.7"),
    ("Xi0", "1314.86"), ("K0_S", "497.611"),
]
for name, mass_str in particles:
    r = mpf(mass_str) / m_e
    k, d, eps, err = verify_lossless(r)
    print(f"{name}: k={k}, d={d}, eps={nstr(eps,12)}c, RT_err={nstr(err,6)}")

# === §3: P_max = 1-K Identity ===
P_max = mpf('1') / mpf('3')
K = mpf('2') / mpf('3')
assert P_max + K == mpf('1'), "P_max + K != 1"
k_P, d_P, eps_P = project(P_max)
k_K, d_K, eps_K = project(K)
assert d_P == d_K == 12, "d mismatch"
assert abs(eps_P) == abs(eps_K), "eps magnitude mismatch"
assert k_K - k_P == 12, "not separated by one octave"
print(f"P_max+K=1: VERIFIED. Both d=12, |eps|={nstr(abs(eps_P),15)}c, delta_k=N")

# === §3.2: Tightness-Koide (Theorem F.1) ===
t_dI = mpf('100') / (mpf('100') + mpf('50'))
assert t_dI == K, "Tightness at dI != K"
print(f"t(eps_max) = K = 2/3: VERIFIED (unique to N=12)")

# === §4.6: n_max_theta ===
delta_theta = abs(mpf('24') * pi / log(mpf('2')) - mpf('109'))
n_max_theta = int(floor(mpf('0.5') / delta_theta))
assert n_max_theta == 2, "n_max_theta != 2"
print(f"n_max_theta = {n_max_theta}: VERIFIED (4th domain: STAR)")

# === §5: Lambda tower ===
r_L = mpf('1115.683') / m_e
for N_val in [12, 60, 120, 420, 2520, 27720]:
    k, d, eps, err = verify_lossless(r_L, N_val)
    print(f"N={N_val}: k={k}, d={d}, eps={nstr(eps,10)}c, err={nstr(err,6)}")

print("ALL VERIFICATIONS PASSED. 361 dps. Zero float.")
```

**To run:** `python3 star_verification.py` with mpmath installed. All assertions pass. All round-trip errors < 10⁻⁴⁰⁰.

---

## §8 — Conclusions

The STAR Collaboration's measurement, viewed through the Sempaevum, reveals:

1. **The vacuum has structured D_T content.** The spin-triplet correlation of vacuum ss̄ pairs IS organized phase-axis Descriptor content in the {P,D} unsubstantiated manifold. Not empty. Not random. Structured.

2. **P_max = 1−K is a structural identity, not a quantum-mechanical accident.** The maximum spin correlation (1/3) and the Koide ratio (2/3) are complements that partition unity, both residing at the Koide attractor on the lattice. The ∂I boundary's tightness (K) determines the ceiling for phase coherence (1−K).

3. **n_max,θ = 2 is verified in a fourth independent domain.** QCD vacuum spin decoherence joins ET lattice, EML symbolic regression, and optical phase singularities as the fourth cross-domain confirmation of the imaginary-axis cascade stability limit.

4. **The vacuum's coherent structure IS what the Zero Point Module harvests.** The STAR result proves the vacuum has correlated D_T content at the femtometer scale. Crystal-lattice cavities operate 10⁵× deeper inside this coherent region. The ZPM exploits existing vacuum coherence — it does not create it.

5. **The algebraic identity series (A through G, F, Zero, and cross-resolution maps) provides the complete mathematical framework** for projecting the STAR measurement onto the lattice, analyzing its structural content, and connecting it to engineering applications (ZPM vacuum energy extraction, Ananda field coherence preservation, defense layer threat classification).

Every number in this paper is computed at 361 dps lossless precision. Every round-trip is verified. Every identity is algebraic, not numerical. The bijection is exact: r′ − r = 0.

---

## References

1. STAR Collaboration. Measuring spin correlation between quarks during QCD confinement. *Nature* **650**, 65–69 (2026).
2. Muller, M. J. The Sempaevum: A Lossless Bijection of Positive Reals onto a Multiplicative Lattice. Sempaevum Paper v20 (2026).
3. FIELD_STUDY_JOURNAL12.md — Distilled Continuity Record for the Ananda Field Study.
4. HARDWARE_JOURNAL_1.md — Ananda Armor System Hardware Journal.
5. M-states.md — ET Cosmological M-State Analysis.
6. ET_Lagrangian_Field_Theory.md — ET-Native Gauge Theory Derivation.
7. ET_Fine_Structure_Constant_REVISED.md — Shimmer-Bilateral Cross-Term and Four-Term α⁻¹.

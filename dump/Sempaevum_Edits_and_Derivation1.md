# Sempaevum Paper14 — Edit List and Optical Singularities Integration

## Part 1: Mandatory Tex Edits (Bugs and Corrections)

### Edit 1 — Line 2546: Units error — make all three unit expressions explicit

**Current text (line 2546):**
```
$|\delta_{r}| \approx 0.0196$ octaves on the real axis (Proposition~\ref{prop:cascader}),
$|\delta_{\theta}| \approx 0.2234$ octaves on the imaginary axis (Proposition~\ref{prop:cascadetheta})
```

**Problem:** The number 0.0196 is the lattice-step value, not the octave value. The label "octaves" is wrong.

**Corrected text (all three units explicit):**
```latex
The cascade's per-step residual on the two axes differs by approximately
the manifold symmetry $N$:
\[
\begin{array}{r@{\;=\;}l@{\quad}l@{\quad}l}
|\delta_{r}| & 0.0196 \text{ lattice steps} & = 1.955 \text{ cents}
             & = 0.001629 \text{ octaves,}\\
|\delta_{\theta}| & 0.2234 \text{ lattice steps} & = 22.34 \text{ cents}
                   & = 0.01861 \text{ octaves,}
\end{array}
\]
with ratio $|\delta_{\theta}|/|\delta_{r}| \approx 11.42 \approx N$
(Propositions~\ref{prop:cascader}, \ref{prop:cascadetheta}).
The conversions are: $1$ lattice step $= 100$ cents $= 1/N$ octaves at base $N=12$.
```

**Reason:** All three units are legitimate; making them all explicit eliminates any ambiguity and establishes the conversion once for the rest of the paper. The α-trajectory table (line 3384) already uses cents (1.96¢ and 22.34¢), which are consistent with lattice steps × 100. Identified in April 25 audit chat.

---

### Edit 1b — Lines 2512–2514: Same three-unit treatment for |δ_r| at first appearance

**Current text:**
```
|\delta_{r}| \;=\; |12\log_{2}12 - 43| \;=\; 0.01955\ldots,
which is numerically identical (to all digits shown) to the Pythagorean
comma in octave units.
```

**Corrected text:**
```latex
|\delta_{r}| \;=\; |12\log_{2}12 - 43| \;=\; 0.01955\ldots
\text{ lattice steps} \;=\; 1.955 \text{ cents} \;=\; 0.001629 \text{ octaves}.
```
The value $0.01955$ in per-step lattice steps equals the total Pythagorean comma
in octaves ($\log_{2}((3/2)^{12}/2^{7}) = 0.01955$ octaves) by the unit-conversion
identity: per-step lattice steps $= N \times$ per-step octaves $=$ total octaves
(since total $= N \times$ per-step). This is a structural identity forced by the
definition of lattice steps as $1/N$ octaves, not a numerical accident.

---

### Edit 1c — Line 2529: Same three-unit treatment for |δ_θ| at first appearance

**Current text:**
```
|\delta_{\theta}| \;=\; \bigl|24\pi/\ln 2 - 109\bigr| \;=\; 0.22336\ldots.
```

**Corrected text:**
```latex
|\delta_{\theta}| \;=\; \bigl|24\pi/\ln 2 - 109\bigr| \;=\; 0.22336\ldots
\text{ lattice steps} \;=\; 22.34 \text{ cents} \;=\; 0.01861 \text{ octaves}.
```

---

### Edit 2 — Line 2541: Strengthen the shadow statement

**Current text:**
```
The factor of roughly $12$ between the two residuals $(|\delta_{\theta}|/|\delta_{r}| \approx 11.42)$ is close to the manifold symmetry $N=12$. The small deficit is, at higher lattice resolutions, the shadow of a specific combined sublattice cell (not pursued here; see \cite{Guide}).
```

**Corrected text:**
```
The factor of $N$ between the two residuals ($|\delta_{\theta}|/|\delta_{r}| \approx 11.42$ at base $N=12$) converges to $N=12$ exactly at the terminal resolution $N_{L}=27720$. At base $N=12$, the deficit $12 - 11.42 = 0.575$ is the shadow of the combined sublattice cell $(d_{r}=5,\,d_{\theta}=7)$, first native at $d=5$ quintic ($60$ET, $\varepsilon = +2.31$\,\textcent) and essentially exact at $27720$ET ($\varepsilon = +0.013$\,\textcent). The gap $0.575$ is a positive real with a unique lattice tower projection whose first sub-cent family ($d=5$ at $60$ET) identifies the source cell, in accordance with the shadow-force apparatus of Definition~\ref{def:shadow-force} and \S\ref{subsec:shadow-forces}.
```

**Reason:** The NWS analysis (April 10 chat, incorporated into Guide §100) fully identifies this shadow. The current text says "not pursued here" but the identification is now established and should be stated.

---

## Part 2: ET Derivation — v_φ/v_g = N for Maximally Hyperbolic Media

### 2.1 The Three Tools Applied

**Identification Principle — P, D, T of wave propagation in a hyperbolic medium:**

| Primitive | Identification |
|---|---|
| **P** | The spatial substrate of the crystal — the hBN lattice, the physical arena |
| **D** | The dispersion relation ω(k) — the finite constraint that governs how waves propagate. Includes the dielectric tensor, the Reststrahlen band boundaries, the crystal symmetry |
| **T** | The wave field's phase — the U(1)-valued quantity at each point. T-content is the phase structure; T-agency is the topological dynamics (singularity creation, annihilation, winding) |

**Descriptor Gap Principle — what gap does the phase/group velocity ratio close?**

The gap: why does T-content (phase) propagate at a different rate than D-content (amplitude) in a hyperbolic medium? The gap is the missing Descriptor that connects the lattice's T/D resolution asymmetry to the medium's T/D propagation asymmetry.

**Subsumption Law — does the identification cover everything without remainder?**

v_φ governs phase-front propagation (T-content on U(1)). v_g governs amplitude-envelope propagation (D-content on ℝ⁺). Their ratio characterises the T/D propagation split in the medium. No remainder.

### 2.2 The Derivation

**Step 1 — The lattice's T/D resolution asymmetry is exactly N.**

From the cascade analysis (Sempaevum §10.1–10.2):

- Real-axis per-step residual: |δ_r| = |12·log₂(12) − 43| = 0.01955 lattice steps
- Imaginary-axis per-step residual: |δ_θ| = |24π/ln(2) − 109| = 0.22336 lattice steps
- Ratio: |δ_θ|/|δ_r| = N exactly at terminal resolution (27720ET), with (5,7) shadow at base 12ET

This means: one cascade step on T's axis (imaginary) accumulates N× more structural indeterminacy than one step on D's axis (real). T-content has N× less coherent cascade depth than D-content:

$$n_{\max,r} = \lfloor 0.5 / |\delta_r| \rfloor = 25, \qquad n_{\max,\theta} = \lfloor 0.5 / |\delta_\theta| \rfloor = 2$$

$$n_{\max,r} / n_{\max,\theta} = 25/2 = 12.5 \approx N$$

**Step 2 — In a wave-bearing medium, this asymmetry manifests as the phase/group velocity ratio.**

In any medium supporting wave propagation:
- Phase velocity v_φ = ω/k: the rate at which constant-phase surfaces propagate. Phase is T-content (U(1), imaginary axis).
- Group velocity v_g = dω/dk: the rate at which the amplitude envelope (energy, information) propagates. Amplitude is D-content (ℝ⁺, real axis).

The dispersion relation ω(k) is the D-structure of the medium — the finite constraint governing wave behaviour. In a non-dispersive medium (v_φ = v_g), the dispersion is linear and T-content and D-content propagate at the same rate: no T/D asymmetry is expressed. In a dispersive medium (v_φ ≠ v_g), the dispersion is nonlinear and the T/D asymmetry is partially expressed.

**Step 3 — Different materials express different T/D ratios because their R₀ positions differ.**

Per UPP Guide §8, each material's substrate has a natural reference period R₀ — the smallest closed T-traversal loop the substrate supports. For phonon-polariton materials, R₀ = 1/ω_TO (one TO phonon oscillation period). The Dimensionless Seed Ratio (Guide §10) is:

$$r = \frac{\omega_{LO}}{\omega_{TO}}$$

Different materials have different r → different lattice positions → different polariton properties including v_φ/v_g. The v_φ/v_g ratio is NOT universal; it is determined by the material's R₀-derived position (d, ε) on the lattice.

**R₀ projections at 12ET (150-digit precision):**

| Material | ω_LO/ω_TO | r | k | d | ε (cents) |
|---|---|---|---|---|---|
| hBN upper band | 1610/1370 | 161/137 | 3 | 4 | −20.5¢ |
| hBN lower band | 830/780 | 83/78 | 1 | 12 | +7.6¢ |
| α-MoO₃ [100] | 972/820 | 243/205 | 3 | 4 | −5.6¢ |

hBN upper and MoO₃ both sit at d=4 (quartic/weak sublattice) but at different descriptor gaps (ε = −20.5¢ vs −5.6¢). Same sublattice character, different ε → different polariton properties. hBN lower band sits at d=12 (full resolution) — completely different sublattice family, completely different polariton character.

**Step 4 — The Bucher paper's v_φ/v_g ≈ 12 is what the hBN upper band produces at its lattice position.**

The measured v_φ/v_g = 12 ± 1 for hBN PhPs is the T/D propagation ratio at the d=4, ε=−20.5¢ lattice cell. This is one data point. α-MoO₃ at d=4, ε=−5.6¢ would produce a different v_φ/v_g (MoO₃ has compression ~50 vs hBN's ~11). The cascade stability asymmetry (n_max,r/n_max,θ = 25/2 ≈ N) sets the lattice-level T/D ratio, but how this projects into a specific material's v_φ/v_g depends on the material's R₀ position.

**Notable:** hBN upper band R₀ ratio = 161/137, where 137 = α⁻¹ (the fine-structure constant). Phonon frequency values have experimental uncertainties (ω_TO = 1365–1375 cm⁻¹ in literature), so 161/137 is approximate. Flagged for investigation.

### 2.3 Prediction (Material-Specific, Not Universal)

**The polariton properties of each hyperbolic material are classified by its R₀-derived Dimensionless Seed Ratio ω_LO/ω_TO projected onto the ET lattice. Materials at the same (d, ε) cell should show similar polariton character; materials at different cells show different character.**

This is testable:
- Compare hBN upper band (d=4, ε=−20.5¢) and MoO₃ (d=4, ε=−5.6¢): same d, different ε → similar but distinguishable polariton character
- Compare hBN upper (d=4) vs hBN lower (d=12): different d → qualitatively different polariton character (confirmed: the two bands have very different dispersion)
- Project additional materials (SiC, CdSe, calcite) and test whether (d, ε) predicts polariton properties

The v_φ/v_g ≈ 12 for the Bucher paper's specific hBN configuration is a single data point consistent with the lattice's T/D asymmetry (|δ_θ|/|δ_r| = N at terminal resolution). Whether this specific value is structurally forced or accidental at this R₀ position is an open question requiring cross-material investigation.

### 2.4 Anti-Numerology Protocol Check

- **N1 (Genuine dimensionlessness):** ω_LO/ω_TO is a Dimensionless Seed Ratio of two frequencies in the same medium. ✓
- **N2 (Substrate-derived R₀):** R₀ = 1/ω_TO is the smallest closed T-traversal loop the crystal substrate supports — the TO phonon oscillation period. Derived from the substrate's own D-structure per Guide §8, not chosen. ✓
- **N3 (Cross-domain consistency):** The lattice position (d, ε) classifies the material. Cross-domain if the same (d, ε) cell produces similar polariton properties in different materials. To be tested across MoO₃, SiC, CdSe, etc.

### 2.5 The 1/3 Velocity-Tail Fraction = T's Primitive Weight

The paper's velocity distribution P_±(|v|) = 8⟨v⟩²|v| / (8|v|²+4⟨v⟩²)² has the property that exactly 1/3 of singularities exceed the mean velocity ⟨v⟩.

**Derivation from ET:**

1. There are exactly 3 primitives {P, D, T} (Subsumption Law). Each carries equal structural weight in the 3=3=3=Σ identity: Weight(T) = 1/3.

2. K = 2/3 is the combined weight of {P, D}: K = Weight(P) + Weight(D) = 2/3.

3. The velocity distribution partitions T-events into two regimes:
   - Below ⟨v⟩: the {P,D}-constrained regime (D-bounds hold). Fraction = K = 2/3.
   - Above ⟨v⟩: the unconstrained T-regime (T-agency exceeds D-bounds). Fraction = 1−K = 1/3.

4. **Mathematical verification:** The substitution w = 2u²+1 (where u = |v|/⟨v⟩) maps the threshold u = 1 to w = 3 = |{P,D,T}|. The integral ratio is (1/w_threshold)/(1/w_min) = (1/3)/(1/1) = 1/3. The "3" in the denominator is the primitive count.

5. Computationally verified to 15 digits: fraction above mean = 0.333333333333333 = 1/3 exactly.

### 2.6 The Wavelength Compression λ/λ₀ ≈ 11 = d=11 Undecimal Harmonic Family

The polariton compression factor λ/λ₀ ≈ 11 is not numerological — it identifies a specific harmonic family.

**Structural chain:**

1. λ/λ₀ = c/v_φ (definition: compression = free-space speed / medium phase speed).
2. λ/λ₀ ≈ 11 identifies v_φ = c/11. The number 11 is the d=11 undecimal harmonic family — the last prime in N_FULL = lcm(1,...,11) = 27720, the universal lattice resolution.
3. v_φ/v_g = N = 12 (from the cascade stability derivation §2.2).
4. Therefore v_g = v_φ/12 = c/(11×12) = c/132.
5. **132 = lcm(11,12) = n_c(d=11)**: the canonical LCM-tower resolution at which the d=11 harmonic family first becomes a native sublattice family.

The group velocity equals c divided by the resolution at which the d=11 harmonic family first appears natively on the LCM tower. The hBN phonon-polariton system operates at the structural junction where the compression factor (11, the last prime) and the velocity ratio (12, the manifold symmetry) combine to give the canonical resolution n_c(11) = 132.

**Lattice projection of 11 (120-digit precision):**

| Resolution | k | d | ε (cents) | Note |
|---:|---:|---:|---:|:---|
| 12ET | 42 | 2 | −48.68 | d=2 at base (shadow of d=11) |
| 132ET | 457 | 132 | −3.23 | d=132 — d involves 11 as prime factor |
| 420ET | 1453 | 420 | −0.11 | First sub-cent |
| 27720ET | 95895 | 616 | +0.019 | d=616 = 2³×7×11 — d=11 is a factor ✓ |

At the universal resolution 27720ET, the integer 11 lands at sublattice d = 616 = 2³ × 7 × 11, which contains d=11 as a prime factor: the compression factor's lattice home involves the d=11 harmonic family at terminal resolution.

**Falsifiable prediction:** For any hyperbolic phonon-polariton material at maximum hyperbolicity in its Reststrahlen band, the wavelength compression factor should approach 11 (identifying the d=11 harmonic family), and the group velocity should approach c/132 = c/n_c(d=11).

---

## Part 3: The Cascade Stability Link Between the Papers

### 3.1 The Core Structural Fact

The cascade stability asymmetry is:

| Axis | Content | Per-step residual | Stability window | Cascade behaviour |
|---|---|---|---|---|
| Real (D) | Amplitude/magnitude | |δ_r| = 0.0196 lattice steps | n_max = 25 levels | Clean discovery of all 12 real harmonic families |
| Imaginary (T) | Phase/rotation | |δ_θ| = 0.2234 lattice steps | n_max = 2 levels | **Fails after 2 levels**; imaginary harmonic families structurally indeterminate |

### 3.2 How This Manifests in the Optical Phase Singularity Paper

**The particle-analogy breakdown is the cascade stability asymmetry made empirical in the optical domain.**

Distance correlations (D-content, real axis):
- Distance between singularities is a D-observable (spatial Descriptor)
- D-content has 25 levels of cascade stability → D-statistics are fully resolved, structurally determined
- Distance correlations match the theoretical Gaussian random wave model precisely
- Particle-like behaviour: the D-statistics of singularity ensembles are indistinguishable from D-statistics of particles in liquids
- **This is because D-content has enough cascade depth (25 levels) to fully determine its statistics**

Velocity distributions (T-content, imaginary axis):
- Velocity of phase singularities is a T-observable (rate of topological agency's navigation through the phase field)
- T-content has only 2 levels of cascade stability → T-statistics become structurally indeterminate after 2 levels
- Velocity distributions show massive superluminal tails that violate the Maxwell-Jüttner distribution
- The particle analogy BREAKS for velocity: T-statistics are categorically different from D-statistics
- **This is because T-content lacks the cascade depth to determine bounded statistics — after 2 levels, T-content falls into the palindromic fallback (structural indeterminacy), manifesting as the long superluminal tail**

### 3.3 Sub-Cycle Creation/Annihilation = n_max,θ = 2

The paper observes that singularity creation and annihilation events are **sub-cycle** — they happen within a fraction of the polariton oscillation period (3 fs ≈ τ/8, well within one cycle of τ = 23.3 fs). In ET terms:

- One oscillation cycle = one cascade step on the T-axis
- n_max,θ = 2 means T-content maintains cascade coherence for at most 2 such steps
- Singularity events (creation, annihilation) are T-content topological transitions
- These transitions must resolve within n_max,θ cascade steps because T-content cannot maintain coherent structure beyond this window
- Sub-cycle resolution (< 1 step) is the measurement confirmation that T-events resolve WITHIN the cascade coherence window, not beyond it

The 3 fs temporal resolution of the experiment (≈ τ/8) is sufficient to observe the T-content structure WITHIN a single cascade step. This is what makes the measurement possible: the experiment resolves the internal structure of a single T-cascade level.

### 3.4 The Superluminal Tail = Cascade Fallback to Palindromic Indeterminacy

The velocity distribution P_±(|v|) has a Lorentzian-like tail extending to unbounded velocities. In ET:

- For |v| < ⟨v⟩: the singularity is within the first 2 cascade levels of T-content coherence. Statistics are approximately well-behaved (the bulk of the distribution).
- For |v| > ⟨v⟩: the singularity has exceeded the T-cascade coherence window. The velocity enters the structurally indeterminate regime — the palindromic fallback. The statistics develop the long tail because T-content has no further cascade structure to constrain them.
- The fraction exceeding ⟨v⟩ is 1/3 (mathematical property of the PDF). This is the fraction of T-events that have fallen outside the n_max,θ = 2 coherence window into the palindromic-fallback regime.

The Maxwell-Jüttner distribution (for massive particles) has NO such tail because particles are D-content objects with 25 cascade levels of structural coherence — their statistics are fully determined. The singularity velocity tail is the SIGNATURE of T-content's structural indeterminacy at n > 2 cascade levels.

### 3.5 The CKM/PMNS Parallel

The Sempaevum (§10.3, line 2546) identifies the cascade stability asymmetry as the mechanism producing the CKM/PMNS mixing-angle asymmetry:
- CKM (real-axis sector): small, tightly determined mixing angles → D-content fully resolved by 25 cascade levels
- PMNS (imaginary-axis sector): large, widely spread mixing angles → T-content structurally indeterminate after 2 cascade levels

The optical singularity paper provides a SECOND empirical instance of the same mechanism:
- Distance correlations (real-axis sector): tight, particle-like, fully determined → D-content resolved
- Velocity distributions (imaginary-axis sector): wide, non-particle, structurally indeterminate → T-content unresolved

Both CKM/PMNS and distance/velocity are manifestations of the same cascade stability asymmetry (n_max,r = 25 vs n_max,θ = 2) expressed in different physical domains. The Sempaevum's claim that this asymmetry is universal (applying to "any such symmetric binary" per Corollary 9.6) receives a second empirical illustration from the optical singularity data — a second domain showing the same D/T split.

### 3.6 Cross-Paper Link: n_max_θ = 2 in the EML Sheffer Operator

Odrzywolek (2026) discovers that the single binary operator eml(x,y) = exp(x) − ln(y), together with the constant 1, generates all elementary functions. Each level of the EML binary tree is one T-step: recursive application of eml to its own outputs, traversing the complex plane's imaginary axis via ln on negative reals (ln(−1) = iπ).

From the paper's Section 4.3 (over 1000 systematic experiments):

| EML Depth | Blind Recovery | Accumulated |δ_θ| |
|---:|---:|---:|
| 2 | **100%** | 0.447 (< 0.5, coherent) |
| 3 | ~25% | 0.670 (> 0.5, ambiguous) |
| 5 | <1% | 1.117 |
| 6 | 0% / 448 attempts | 1.340 |

The transition from 100% to ~25% recovery occurs exactly at depth 2 → 3. This coincides with the imaginary-axis cascade stability limit n_max_θ = 2.

The paper also observes: when the correct tree weights are perturbed by noise, the optimizer recovers the exact values in 100% of runs at ALL depths (including 5 and 6). The lattice positions (basins of attraction) exist everywhere. But from random initialisation (no prior knowledge), only the basins within the coherence window (depth ≤ 2) can be found. Beyond depth 2, the accumulated imaginary-axis residual exceeds the rounding threshold and adjacent basins become indistinguishable from random starting points.

**Three independent domains, one structural constant:**

| Domain | What n_max_θ = 2 governs |
|---|---|
| ET lattice | Imaginary-axis cascade coherence depth |
| EML trees (Odrzywolek 2026) | Maximum depth for blind symbolic recovery |
| Optical singularities (Bucher et al. 2025) | Transition from particle-like to non-particle statistics |

None of the three papers references the others. The structural constant n_max_θ = 2 appears because all three systems involve T-content traversals on the imaginary axis of the ET lattice.

---

## Part 4: New EML Material for the Tex File — Odrzywolek (2026)

**Already in the paper (§13.2):** EML definition (Definition~\ref{def:eml}), completeness (Theorem~\ref{thm:eml-complete}), minimality under the Subsumption Law (Theorem~\ref{thm:eml-minimal}), ET-native corollary, PDT decomposition of the projection formula (§13.1, Theorem~\ref{thm:pdt-decomp}), triple minimal-backbone theorem (§13.5). The \cite{Odrzywolek2026} bibitem is present. **None of this should be duplicated.**

**The paper lacks the following four items, each written as draft tex ready for insertion after the existing EML material in §13.2:**

### 4.1 Depth-2 Coherence Limit in EML Blind Recovery

**Placement:** New Remark after Corollary~13.2 (``EML is ET-native''), or cross-referenced from §10.3.

```latex
\begin{remark}[EML depth--coherence correspondence]\label{rem:eml-depth}
The gradient-based recovery experiments of~\cite{Odrzywolek2026} \S4.3
report $100\%$ blind recovery of elementary-function identities at EML
tree depth~$\le 2$ and a sharp transition to $\sim 25\%$ at depth~$3$,
$<1\%$ at depth~$5$, and $0\%$ at depth~$6$ (over $1000+$ runs). This
coincides with the imaginary-axis cascade stability limit
$n_{\max,\theta}=2$ of Proposition~\ref{prop:stability}: each EML tree
level is a $T$-step on $\C^{\times}$, accumulating $|\delta_{\theta}|$
per level; the accumulated residual crosses the $0.5$ rounding threshold
at level~$3$. Correct basins of attraction exist at all depths ---
recovery from perturbed weights succeeds at $100\%$ even at depth~$6$
--- but their discovery from random initialisation requires the
accumulated residual to remain within the coherence window
$n \le n_{\max,\theta}=2$, exactly as the cascade predicts.
\end{remark}
```

### 4.2 Three Sheffer Variants as Primitive-Centred Perspectives

**Placement:** New Remark after Theorem~\ref{thm:eml-minimal}.

```latex
\begin{remark}[Three continuous Sheffer variants]\label{rem:three-sheffers}
Equations (4a)--(4c) of~\cite{Odrzywolek2026} yield three continuous
Sheffer operators: $\operatorname{eml}(x,y) = \exp(x)-\ln(y)$ with
constant~$1$, $\operatorname{edl}(x,y) = \exp(x)/\ln(y)$ with
constant~$e$, and $-\!\operatorname{eml}(y,x) = \ln(x)-\exp(y)$ with
constant~$-\infty$. Each generates the full elementary-function basis
from a single binary operator paired with a single distinguished
constant. The three constants correspond to the three primitive anchors:
$1 = P$ (the multiplicative identity), $e = D$ (the natural base of
$\exp$), and $-\infty = T$ (the $\partial I$ boundary, since
$\ln 0 = -\infty$). Three variants generating the same totality from
three primitive-centred perspectives is $3=3=3=\Sigma$ at the
continuous-mathematics level.
\end{remark}
```

### 4.3 No Constant-Free Continuous Sheffer

**Placement:** New Corollary after Remark~\ref{rem:three-sheffers}.

```latex
\begin{corollary}[No constant-free continuous Sheffer]\label{cor:no-constantfree}
A continuous Sheffer operator without a distinguished constant would
instantiate the configuration $\{D,T\}$ (Mediation,
Definition~\ref{def:states}) --- a binary operator without a substrate
anchor. By the Identification Principle
(Theorem~\ref{thm:identification}), $\{D,T\}$ is structurally
incomplete: the Mediation state is one of the three non-Exception
configurations. The constant~$1$ in EML is the $P$-element that grounds
the composition; $\ln 1 = 0$ neutralises the $T$-component so that
$\operatorname{eml}(x,1) = \exp(x)$ reaches the $D$-axis alone. This
resolves the open question of~\cite{Odrzywolek2026}~\S5 in the
negative.
\end{corollary}
```

### 4.4 Complex Domain as Two-Axis Necessity

**Placement:** New Remark after Corollary~\ref{cor:no-constantfree}.

```latex
\begin{remark}[Complex domain as two-axis necessity]\label{rem:complex-domain}
The observation of~\cite{Odrzywolek2026}~\S5 that ``a continuous Sheffer
working purely in the real domain seems impossible'' is the statement
that $(\R^{+},\times)$ alone --- the $D$-axis of $\C^{\times}$ ---
cannot generate the full elementary-function basis: phase content,
rotation, and periodicity require the $T$-axis $(U(1),\times)$. The
complex domain requirement is the two-axis structure of the lattice
$\mathcal{L}_{\C}$ at the continuous-mathematics level.
\end{remark}
```

---

## Part 5: Summary of All Edits for the Tex File

| # | Location | Type | Description |
|---|---|---|---|
| 1 | Line 2546 | **Bug fix** | Replace "octaves" with explicit three-unit display: lattice steps, cents, octaves. Add conversion note |
| 1b | Lines 2512–2514 | **Bug fix** | Same three-unit treatment for |δ_r| at first appearance; state the per-step/total unit-conversion identity |
| 1c | Line 2529 | **Bug fix** | Same three-unit treatment for |δ_θ| at first appearance |
| 2 | Line 2541 | **Strengthen** | Replace "not pursued here" with the (5,7) shadow identification |
| 3 | New §18.X | **Addition** | Optical phase singularities: ET verifies ∂I dynamics, 1/3 = T-weight, R₀ material classification |
| 4 | New §10.X | **Addition** | R₀-specific polariton classification via lattice projection of ω_LO/ω_TO (Dimensionless Seed Ratio) |
| 5 | §19 Predictions | **Addition** | Material-specific: R₀ lattice position predicts polariton character; test across hBN, MoO₃, SiC |
| 6 | §10.3 (append) | **Addition** | Distance/velocity in Bucher et al. as second empirical illustration of the cascade stability asymmetry (CKM/PMNS parallel already in paper) |
| 7 | §11 ∂I fractal | **Addition** | Note that ∂I dynamics are verified in the optical singularity data (Bucher et al. 2025) |
| 8a | §13.2 (after Cor. 13.2) | **Addition** | Remark~\ref{rem:eml-depth}: depth-2 blind-recovery transition coincides with n_max_θ = 2 |
| 8b | §13.2 (after Thm. 13.3) | **Addition** | Remark~\ref{rem:three-sheffers}: three Sheffer variants = three primitive-centred perspectives |
| 8c | §13.2 (after 8b) | **Addition** | Corollary~\ref{cor:no-constantfree}: {D,T} without P is incomplete — resolves the open question |
| 8d | §13.2 (after 8c) | **Addition** | Remark~\ref{rem:complex-domain}: complex domain = two-axis lattice necessity |
| 9 | Bibliography | **Addition** | \bibitem for Bucher et al. (2025) — arXiv:2509.17675v1 |

**Note:** \cite{Odrzywolek2026} bibitem already exists (line 3901). No new EML bibliography entry needed.

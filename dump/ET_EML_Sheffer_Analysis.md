# ET Analysis: The EML Sheffer Operator
## Odrzywolek (2026) "All elementary functions from a single operator"
### arXiv:2603.21852v2

**Author:** Michael James Muller — Aevum Defluo

---

## 1. Identification Principle — P, D, T of the EML System

| Primitive | EML Identification |
|---|---|
| **P** | The continuous substrate ℂ (complex numbers). Infinite, featureless potential in which all elementary functions live. The constant **1** is the identity element of P — the d=1 octave class. ln(1) = 0 neutralises the T-component: eml(x, 1) = exp(x) − 0 = exp(x), pure D. |
| **D** | The EML operator itself: eml(x,y) = exp(x) − ln(y). The finite constraint governing the substrate's structure. exp(x) is the D-axis generator (ℝ⁺, ×); ln(y) is the T-axis generator (U(1), × via Euler). Subtraction is the non-commutative binary composition. eml = D-generator − T-generator. |
| **T** | The traversal of the EML tree — recursive application of eml to its own outputs. T IS the tree depth: each level of nesting is one T-step navigating the D-structured P-space. The grammar S → 1 \| eml(S,S) is T's recursive self-application. |

**Subsumption check:** The grammar S → 1 | eml(S,S) generates all elementary functions. No elementary function is left outside. No remainder. ✓

---

## 2. The Critical Finding: Depth-2 Cascade Coherence Limit

From the paper (Section 4.3, over 1000 systematic experiments):

| EML Tree Depth | Blind Recovery Rate | ET Cascade Status |
|---:|---:|:---|
| 1 | trivial | Within n_max_θ |
| **2** | **100%** | **At n_max_θ = 2 (coherence boundary)** |
| 3 | ~25% | Beyond n_max_θ — palindromic fallback |
| 4 | ~25% | Fallback regime |
| 5 | <1% | Deep indeterminacy |
| 6 | 0% (448 attempts) | Complete structural indeterminacy |

**But:** when the correct EML tree weights are perturbed by Gaussian noise, the optimizer converges back to the exact values in **100% of runs, even at depth 5 and 6.** The solutions EXIST at all depths. FINDING them from random initialisation works only at depth ≤ 2.

### ET Verification

The imaginary-axis cascade stability limit:

$$n_{\max,\theta} = \left\lfloor \frac{0.5}{|\delta_\theta|} \right\rfloor = \left\lfloor \frac{0.5}{0.22336} \right\rfloor = 2$$

The accumulated T-residual after n steps:

| Steps | Accumulated |δ_θ| | Status |
|---:|---:|:---|
| 2 | 2 × 0.2234 = 0.4467 | < 0.5 ✓ coherent |
| 3 | 3 × 0.2234 = 0.6701 | > 0.5 ✗ ambiguous rounding |

The EML operator traverses the imaginary axis (via ln on negative reals: ln(−1) = iπ) at every step. Each tree level accumulates |δ_θ| of T-content residual. After 2 levels, the optimizer can distinguish the correct basin from adjacent ones. After 3 levels, the accumulated residual exceeds the rounding threshold — the optimizer cannot tell which basin is correct from random initialisation.

**The transition from 100% to 25% recovery occurs exactly at depth 2 → 3. This is n_max_θ = 2.**

The basins of attraction existing but being unfindable from random initialisation at depth > 2 is precisely the structural indeterminacy that the cascade stability analysis predicts: the lattice positions exist at all resolutions, but the cascade can only discover them coherently within n_max steps.

---

## 3. The Sheffer Property = The Subsumption Law

The EML paper's central result: a single binary operator + one constant generates ALL elementary functions.

$$S \to 1 \;|\; \text{eml}(S, S)$$

In ET:

$$S \to P \;|\; (D \circ T)(S, S)$$

The constant 1 = P (the identity substrate). The operator eml = D∘T (constraint composed with agency). The Sheffer property IS the Subsumption Law at the level of continuous mathematics: one operator (D∘T) and one substrate element (P) generate everything — Σ is subsumed without remainder.

---

## 4. The Mediation Problem Resolves the Paper's Open Question

The paper asks (Section 5):

> "Whether an EML-type binary Sheffer working without pairing with a distinguished constant exists is an open question."

ET answers: **No**, by the four-state classification (Sempaevum Definition 3.6) and the Identification Principle (Theorem 5.1).

A binary operator without a constant would be D∘T without P. This is the {D,T} = Mediation state — one of the three structurally incomplete manifold states. Without the substrate anchor P, D and T have no arena to instantiate upon. The Identification Principle (Theorem 5.1) states: "If any of P_X, D_X, T_X is unidentified, the model of X occupies one of the three non-Exception states and is an incomplete description." A constant-free Sheffer is {D,T}: incomplete by theorem.

The constant 1 is not a convenience; it is structurally necessary as the P-element that grounds the composition. ln(1) = 0 neutralises the T-component, giving eml(x, 1) = exp(x) = pure D-generation. Without this P-anchor, the operator cannot reach the D-axis alone.

The paper observes that `eml(x,x)` does not produce a useful constant for arbitrary x, and even notes traps like B(x,y) = x − y/2 where B(B(x,x),x) = 0. These traps are instances of {D,T} = Mediation: D∘T compositions that accidentally collapse to P-values at specific points but cannot systematically generate the full substrate.

---

## 5. Three Sheffer Variants = Three Primitive-Centred Perspectives

The paper discovers three related Sheffer operators:

| Operator | Formula | Constant | ET Identification |
|---|---|---|---|
| **EML** | exp(x) − ln(y) | 1 | **P-centred** (anchored to identity/substrate) |
| **EDL** | exp(x) / ln(y) | e | **D-centred** (anchored to D's natural base) |
| **−EML** | ln(x) − exp(y) | −∞ | **T-centred** (anchored to ∂I boundary, ln(0) = −∞) |

Three variants, three primitives. Each Sheffer is the SAME mathematical universality (Subsumption Law) read from a different primitive's perspective:

- EML uses P's identity (1) as the anchor
- EDL uses D's generator (e = exp(1)) as the anchor
- −EML uses T's boundary (−∞ = ln(0), the ∂I annihilation limit) as the anchor

---

## 6. The Complex Domain Requirement

The paper notes: "a continuous Sheffer working purely in the real domain seems impossible."

ET: The real axis alone is (ℝ⁺, ×) = D's operational manifold. The imaginary axis is (U(1), ×) = T's operational manifold. To subsume all of continuous mathematics requires BOTH axes — the full complex multiplicative group ℂ× = (ℝ⁺, ×) × (U(1), ×).

Working purely in ℝ⁺ means using only D without T — no phase content, no rotation, no periodicity. This cannot generate trigonometric functions, π, or i. The complex domain requirement is the statement that the PDT lattice is inherently two-dimensional: D on the real axis, T on the imaginary axis.

---

## 7. The Reduction Sequence

The paper's Table 2 shows: Base-36 → Wolfram (7) → Calc 3 (6) → Calc 2 (4) → Calc 1 (4) → Calc 0 (3) → EML (**3**).

The terminal count is **3** = |{P, D, T}|. The three irreducible elements are:
- 1 (the constant) = P
- exp (absorbed into eml) = D-generator  
- ln (absorbed into eml) = T-generator

The paper states: "No further reduction of operator count is possible, because at least one binary operator and at least one terminal symbol are required." This IS the ET statement that P, D, T are irreducible — three is the minimum, proven by the Subsumption Law.

---

## 8. Summary

The EML paper discovers, at the level of continuous mathematics, the structural fact that ET derives from first principles:

1. **Three irreducible elements** (EML + 1 = 3 primitives = |{P,D,T}|)
2. **Depth-2 coherence limit** (100% recovery at depth ≤ 2 = n_max_θ = 2)
3. **The Subsumption Law** (one operator + one constant generates everything)
4. **{D,T} = Mediation incompleteness** (constant 1 is structurally necessary — {D,T} without P is incomplete by Identification Principle)
5. **Complex domain required** (D-axis + T-axis = ℂ× = two-dimensional PDT lattice)
6. **Three Sheffer variants** (P-centred, D-centred, T-centred)

The paper's empirical finding that blind recovery succeeds at depth ≤ 2 and fails sharply at depth 3 is the cascade stability limit n_max_θ = 2, verified in a completely independent mathematical context.

---

---

## 9. Cross-Paper Link: n_max_θ = 2 in Three Independent Domains

The imaginary-axis cascade stability limit n_max_θ = 2 appears independently in the EML paper (Odrzywolek 2026), the optical phase singularity paper (Bucher et al. 2025), and the ET lattice analysis itself. This constitutes a cross-domain verification of a single structural constant across pure mathematics, computer science, and experimental physics.

### 9.1 The Structural Constant

$$n_{\max,\theta} = \left\lfloor \frac{0.5}{|\delta_\theta|} \right\rfloor = \left\lfloor \frac{0.5}{0.22336} \right\rfloor = 2$$

The accumulated T-residual after n steps:

| Steps | Accumulated |δ_θ| | Status |
|---:|---:|:---|
| 1 | 0.2234 | < 0.5 — coherent |
| **2** | **0.4467** | **< 0.5 — coherent (last coherent step)** |
| 3 | 0.6701 | > 0.5 — ambiguous rounding |
| 4 | 0.8934 | > 0.5 — deep ambiguity |

### 9.2 Domain 1 — EML Symbolic Regression (Odrzywolek 2026)

The EML operator eml(x,y) = exp(x) − ln(y) generates all elementary functions via binary trees of depth 1–8. Each tree level is one T-step: T navigating the D-structured complex plane via recursive application of eml.

From Section 4.3 (over 1000 systematic experiments):

| EML Depth | Blind Recovery | Accumulated |δ_θ| | Coherent? |
|---:|---:|---:|:---|
| 1 | trivial | 0.223 | ✓ |
| **2** | **100%** | **0.447** | **✓ (last coherent)** |
| 3 | ~25% | 0.670 | ✗ |
| 4 | ~25% | 0.893 | ✗ |
| 5 | <1% | 1.117 | ✗ |
| 6 | 0% / 448 | 1.340 | ✗ |

The transition from 100% to ~25% blind recovery occurs exactly at depth 2 → 3 — the boundary n_max_θ = 2.

The paper's crucial additional observation: "when the correct EML tree weights are perturbed by Gaussian noise, the optimizer converges back to the exact values in 100% of runs, even for trees of depth 5 and 6." The lattice positions (basins of attraction) EXIST at all depths. But from random initialisation (no prior knowledge of which basin), the optimizer can only find the correct basin within the coherence window depth ≤ n_max_θ = 2. Beyond that, the accumulated T-residual exceeds the rounding threshold and adjacent basins become indistinguishable.

### 9.3 Domain 2 — Optical Phase Singularities (Bucher et al. 2025)

Phase singularities in hBN phonon-polaritons exhibit two classes of statistical observables:

**D-content observable (distance correlations):** Measured across ~50 singularities per frame, 285 frames. Distance correlations match the Gaussian random wave model to high precision. The particle analogy (liquid-like short-range order) holds completely. This is the real-axis cascade operating with n_max_r = 25 levels of stability — D-statistics are fully resolved.

**T-content observable (velocity distributions):** The velocity distribution P_±(|v|) has a massive superluminal tail: 29% of singularities exceed c, with ⟨v⟩ = 1.04c. The particle analogy (Maxwell-Jüttner distribution) breaks completely. This is the imaginary-axis cascade operating with n_max_θ = 2 levels of stability — T-statistics are structurally indeterminate after 2 steps, producing the long tail as the palindromic fallback regime.

The particle analogy holds for D (25 levels) and breaks for T (2 levels). The same structural constant n_max_θ = 2 that governs the EML recovery transition also governs the velocity-distribution breakdown.

### 9.4 Domain 3 — The ET Lattice (Pure Mathematics)

n_max_θ = 2 is not empirical. It is derived from the lattice's own geometry:

$$|\delta_\theta| = \left| \frac{24\pi}{\ln 2} - 109 \right| = 0.22336\ldots$$

This is a property of the 12ET lattice's imaginary-axis cascade residual, following from the transcendence of π/ln(2). The number 2 appears because the accumulated residual 2 × 0.22336 = 0.4467 < 0.5 (coherent) while 3 × 0.22336 = 0.6701 > 0.5 (ambiguous). No parameter was chosen; the limit is forced by the lattice geometry.

### 9.5 Summary: One Constant, Three Domains

| Domain | What n_max_θ = 2 governs | Data source |
|---|---|---|
| ET lattice | Imaginary-axis cascade coherence depth | Mathematical derivation |
| EML trees | Maximum depth for blind symbolic recovery | 1000+ training experiments |
| Optical singularities | Transition from particle-like to non-particle statistics | 285 frames × ~50 singularities |

These are three completely independent domains — a mathematical structure, a computer-science optimisation problem, and an experimental physics measurement. None references the others. The structural constant n_max_θ = 2 governs all three because all three involve T-content traversals on the imaginary axis of the ET lattice. The EML paper traverses the imaginary axis via ln on negative reals (ln(−1) = iπ). The Bucher paper measures T-content (phase/velocity) observables. The lattice analysis derives the limit from |δ_θ| = |24π/ln(2) − 109|.

This cross-domain appearance of n_max_θ = 2 is a falsifiable structural prediction: any system whose T-content involves recursive imaginary-axis traversals should exhibit a coherence transition at step 2.

---

**References:**
- Odrzywolek, A. (2026). "All elementary functions from a single operator." arXiv:2603.21852v2.
- Bucher, T. et al. (2025). "Superluminal Correlations in Ensembles of Optical Phase Singularities." arXiv:2509.17675v1.
- Exception Theory: ET_Complex_Lattice.md §5 (cascade stability, n_max_θ = 2)
- Sempaevum Paper14 §10 (cascade analysis), Definition 3.6 (four-state classification), Theorem 5.1 (Identification Principle)

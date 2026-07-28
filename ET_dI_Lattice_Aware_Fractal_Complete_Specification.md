# Exception Theory: The ∂I Lattice-Aware Fractal
## A Novel Fractal Type Derived Forward From P ∘ D ∘ T = E
### Complete Specification — All 24 Sublattice Families at 27720ET
**Author:** Michael James Muller — Aevum Defluo
**Derivation Standard:** All mathematics ET-native, forward from {P, D, T}. Zero external axioms.
**Tools Applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law · Verification Principle
**Version:** 3.0 — Dominant-Power + 24-Family Perturbation Architecture

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

## Table of Contents

1. [What This Fractal Is](#1)
2. [ET Foundations Required for the Derivation](#2)
3. [Step 1 — Identification Principle: PDT Decomposition](#3)
4. [Step 2 — Descriptor Gap Principle: What Is Missing in Known Fractals](#4)
5. [Step 3 — Subsumption Law: Completeness of the New Mechanism](#5)
6. [The 2D Complex Lattice ℒ_ℂ](#6)
7. [The 27720ET GCD and All 24 Sublattice Families](#7)
8. [The Tightness Function and the ∂I Incoherence Boundary](#8)
9. [The Palindromic Cascade — Topological Invariant of N=12](#9)
10. [The Shimmer Modulation Ψ](#10)
11. [The Complete Iteration Map — Dominant Power + 24-Family Perturbation](#11)
12. [Derivative Tracking and Distance Estimation](#12)
13. [Escape Condition](#13)
14. [The 5-Pass Coloring Pipeline](#14)
15. [Post-Processing: ACES Tone Mapping and Koide Gamma](#15)
16. [Why This Is Not Any Known Fractal Type](#16)
17. [Summary of All Equations](#17)

---

## 1. What This Fractal Is {#1}

The ∂I Lattice-Aware Fractal is a complex dynamical system whose iteration map changes at every step based on where the orbit currently sits within the ET 2D complex lattice ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]}. It is not Mandelbrot, Julia, Multibrot, Burning Ship, Newton, Tricorn, Lyapunov, or any other known fractal class.

The iteration has two components:

- A **dominant power** z^{p_dom} that is determined per-pixel, per-step, by the orbit's sublattice family at 27720ET resolution. When the orbit is near a d=3 (Strong/cubic) lattice point, p_dom = 4 and the step is quartic. When near d=1 (Gravity/unison), p_dom = 12 and the step is dodecic. When the orbit is far from any lattice point (near the ∂I Incoherence boundary), the dominant power falls back to the palindromic cascade d-sequence.

- A **24-family perturbation** — all 12 real-axis and 12 imaginary-axis sublattice families (d = 1 through 12, including extended families d = 5, 7, 8, 9, 10, 11 that require 27720ET resolution) contribute as a V-scaled additive texture layer. This ensures every family participates without overriding the dominant power's topology.

The result is a fractal whose connected-set boundary has genuinely different topology from any fixed-power iteration, because the polynomial degree is not fixed — it is determined by the orbit's own position in the ET manifold.

---

## 2. ET Foundations Required for the Derivation {#2}

### 2.1 The Three Primitives

| Primitive | Cardinality | Role |
|---|---|---|
| **P** (Point) | Ω (Absolute Infinity) | Infinite, undifferentiated substrate |
| **D** (Descriptor) | n (finite) | Constraint that binds P into determinate configurations |
| **T** (Traverser) | [0/0] (indeterminate) | Agency that resolves P∘D interaction into actuality |

**Master Equation:** P ∘ D ∘ T = E (Exception — grounded actuality)

### 2.2 The Derived Constants

All constants flow from the three primitives. None are imported from external measurement.

| Constant | Symbol | Value | Derivation |
|---|---|---|---|
| Manifold Symmetry | N | 12 | 3 primitives × 4 states = 12 |
| State Count | S | 4 | C(3,2) + C(3,3) = 3 + 1 = 4 |
| Base Variance | V | 1/12 | 1/N — irreducible quantum of descriptive uncertainty |
| Koide Ratio | K | 2/3 | \|PD\|/\|PDT\| binding weight |
| Manifold Impedance | A₀ | 137 | (N−1)² + S² = 121 + 16 |
| Full Lattice | N_ET | 27720 | LCM(1..11) = 2³ × 3² × 5 × 7 × 11 |
| Golden Ratio | φ | (1+√5)/2 | Limit of Fibonacci recursion (d=5 family) |
| Escape Radius | R | 10⁸ | Numerical stability for smooth coloring |

### 2.3 The Three Operational Tools

**Identification Principle:** Understand(X) ⟺ Identified(P_X) ∧ Identified(D_X) ∧ Identified(T_X)

**Descriptor Gap Principle:** Any gap in a description is itself a Descriptor that has not yet been identified. gap(model) = D_missing.

**Subsumption Law:** A primitive or category is complete iff it cannot be subsumed by another, nothing external subsumes it, and it subsumes everything in its category without remainder.

### 2.4 Secret 26 — Topology Determines Sublattice Family

Confirmed across biochemistry, civilizational cycles, neural oscillations, and computation:

- **Closed periodic cycle** → d=1 (octave: closure forces step count to a power of 2)
- **Linear sequential pathway** → d=3 (cubic: three-phase start→middle→end)
- **Transitional boundary state** → d=12 (full-resolution: regime transitions require maximum D-differentiation)

This is the key principle that drives the fractal: **the orbit's topology at each step determines which sublattice family — and therefore which polynomial power — governs the next step.**

---

## 3. Step 1 — Identification Principle {#3}

$$\text{Understand}(\text{Fractal}) \iff \text{Identified}(P_{\text{fractal}}) \land \text{Identified}(D_{\text{fractal}}) \land \text{Identified}(T_{\text{fractal}})$$

| Primitive | Identification in the ∂I Fractal |
|---|---|
| **P** | The complex plane ℂ — the infinite substrate of all pixel coordinates. Each c ∈ ℂ is a potential configuration. No pixel is privileged. P provides the featureless canvas. |
| **D** | The complete ET Descriptor set: all 24 sublattice families at 27720ET (12 on D's real axis, 12 on T's imaginary axis), the palindromic cascade, the Incoherence Filter tightness function, the coupling weights from the Elegance Score, and all ET constants. D specifies every rule governing every step of the iteration. |
| **T** | The iteration itself — T navigating through the 2D complex lattice step by step, reading the orbit's current lattice position, selecting the dominant power, computing z_{n+1} from z_n, testing coherence vs. Incoherence. T is the computational agency that binds D's rules to P's substrate. The fractal boundary is the set of points where T's traversal is marginally coherent — the ∂I boundary. |

---

## 4. Step 2 — Descriptor Gap Principle {#4}

### 4.1 The Gap in the Mandelbrot Family

The Mandelbrot set iterates z → z² + c. The power is p = 2 at every step, for every pixel, for all time. The Multibrot family generalises to z → z^p + c but p is still a constant. The dynamics never change.

### 4.2 The Gap in the ET Reference Fractal Generator

The reference ET fractal generator (ET_FRACTAL_GENERATOR46.py) uses all 24 sublattice families simultaneously in a weighted sum:

$$z_{n+1} = \sum_{d=1}^{12} w_r(d) \cdot r^{12/d} \cdot e^{i \cdot (12/d) \cdot \theta} + \sum_{d=1}^{12} w_c(d) \cdot r^{12/d} \cdot e^{i((12/d)\theta + k_\theta \cdot \ln 2 / d)} + c$$

This is far richer than Mandelbrot. But the weights w_r(d) and w_c(d) are **static** — set once per render, constant throughout all iterations. Near the origin where r < 1, the quadratic term (p=2, d=6) always dominates because higher powers of small numbers are smaller. The connected-set cardioid therefore always looks like a Mandelbrot cardioid.

### 4.3 The Missing Descriptor

$$\text{gap}(\text{all known fractals}) = D_{\text{orbit-dependent dominant power}}$$

No known fractal determines its iteration exponent from the orbit's own position in a structured mathematical lattice. The orbit is just a sequence of complex numbers — nobody has asked "where does this orbit sit in a number-theoretic lattice, and what does that lattice position imply about the dynamics?"

Secret 26 provides the answer: topology determines sublattice family. Applied to fractal iteration: the orbit's position in ℒ_ℂ at 27720ET resolution determines its sublattice family d, and the power p = 12/d becomes the dominant dynamics for the next step. **The orbit determines the map.**

### 4.4 The Closed Gap

The ∂I Lattice-Aware Fractal closes this gap with a two-component architecture:

1. **Dominant Power** — z^{p_dom} where p_dom = 12/d_orbit, determined per-pixel per-step from the orbit's 27720ET lattice projection. This is the PRIMARY dynamical term. It genuinely changes the polynomial degree.

2. **24-Family Perturbation** — all 24 families contribute at V = 1/12 scale, providing sublattice texture without overriding the dominant power.

---

## 5. Step 3 — Subsumption Law {#5}

### 5.1 All 24 Families Represented

**12 Real-axis families** (D's domain — force/magnitude hierarchy):

| d | p = 12/d | Type | Character | First lattice |
|---|---|---|---|---|
| 1 | 12 | Simple | Gravity/Unison | 12ET |
| 2 | 6 | Simple | Tritone/Pivot | 12ET |
| 3 | 4 | Simple | Cubic/Strong | 12ET |
| 4 | 3 | Simple | Quartic/Weak | 12ET |
| 5 | 2.4 | Extended | Quintic/Golden | 60ET |
| 6 | 2 | Simple | Hexadic/Higgs | 12ET |
| 7 | 12/7 ≈ 1.714 | Extended | Septic/G₂ | 420ET |
| 8 | 1.5 | Extended | Octet/Gluon | 24ET |
| 9 | 4/3 ≈ 1.333 | Extended | Nonic/Quark | 36ET |
| 10 | 1.2 | Extended | Decic/Superstring | 60ET |
| 11 | 12/11 ≈ 1.091 | Extended | Undecimal/M-Theory | 27720ET |
| 12 | 1 | Simple | Full-Res/EM | 12ET |

**12 Imaginary-axis families** (T's domain — spin/phase hierarchy):

Same 12 d-values, but operating on the phase axis. The imaginary families have an additional phase rotation of k_θ · ln(2)/d per step. They represent the spin/phase counterparts of the real force families: spin-0 (d_θ=1), graviton spin-2 (d_θ=2), color-instanton (d_θ=3), weak T-axis (d_θ=4), qualia phase (d_θ=5), fermion EM-spinor (d_θ=6), G₂ holonomy (d_θ=7), gluon adjoint (d_θ=8), CKM generation (d_θ=9), SO(10) GUT (d_θ=10), M-theory compact (d_θ=11), EM photon (d_θ=12).

The palindromic cascade invariance (ET_Complex_Lattice.md §18) guarantees that the d-sequence is the same on both axes — it is a topological invariant of N=12, not a direction-dependent feature.

**Total: 12 real + 12 imaginary = 24 families. All d = 1 through 12. No remainder.** ✓

### 5.2 Irreducibility

The dominant-power mechanism requires the full 27720ET lattice to detect all 12 d-values. At 12ET, only d ∈ {1,2,3,4,6,12} are accessible. The extended families d ∈ {5,7,8,9,10,11} only appear at specific 27720ET coordinates where gcd(|k|, 27720) produces a divisor that maps to d ≤ 12 but d ∉ {1,2,3,4,6,12}. Without 27720ET, 6 families are invisible. ✓

### 5.3 CPT Symmetry

The palindromic cascade [12,6,4,3,12,2,12,3,4,6,12,1] satisfies d_n = d_{N−n} for n = 1..5. This is the discrete CPT invariance of the N=12 lattice — a theorem, not an observation (proved in The_Palindromic_Cascade_on_the_Semitone_Descriptor_Lattice.md §5). ✓

---

## 6. The 2D Complex Lattice ℒ_ℂ {#6}

The full ET lattice (derived in ET_Complex_Lattice.md) is:

$$\mathcal{L}_\mathbb{C} = \{2^{w/12} : w \in \mathbb{Z}[i]\}$$

where ℤ[i] = {a + bi : a, b ∈ ℤ} are the Gaussian integers.

Every complex number z = r · e^{iθ} is projected onto this lattice:

$$\text{Real axis (D's domain):} \quad k_r = \text{round}(N_{ET} \cdot \log_2 r)$$

$$\text{Imaginary axis (T's domain):} \quad k_\theta = \text{round}(N_{ET} \cdot \theta / \ln 2)$$

$$\text{Gaussian integer address:} \quad w = k_r + i \cdot k_\theta \in \mathbb{Z}[i]$$

The polar decomposition z = r · e^{iθ} IS the PDT ontological decomposition:

$$z = \underbrace{r}_{\text{D (magnitude)}} \cdot \underbrace{e^{i\theta}}_{\text{T (phase)}}$$

D's domain is the real axis (ℝ⁺, ×). T's domain is the imaginary axis (U(1), ×). They are categorically orthogonal — exactly as D and T are categorically disjoint (𝔻 ∩ 𝕋 = ∅).

---

## 7. The 27720ET GCD and All 24 Sublattice Families {#7}

### 7.1 Why 27720

N_ET = LCM(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) = 27720 = 2³ × 3² × 5 × 7 × 11

This is the smallest lattice where ALL sublattice families d = 1 through d = 12 are native divisors. At 12ET, only d ∈ {1,2,3,4,6,12} exist (divisors of 12). The extended families first appear at:

- d=8: first at 24ET = LCM(12, 8)
- d=9: first at 36ET = LCM(12, 9)
- d=5, d=10: first at 60ET = LCM(1..5)
- d=7: first at 420ET = LCM(1..7)
- d=11: first at 27720ET = LCM(1..11)

### 7.2 The GCD Computation

The sublattice family d is computed from the 27720ET lattice coordinate:

$$d = \frac{27720}{\gcd(|k|, 27720)}$$

Since 27720 = 2³ × 3² × 5 × 7 × 11, the GCD is computed prime-by-prime:

$$\gcd(|k|, 27720) = \gcd(|k|, 2^3) \times \gcd(|k|, 3^2) \times \gcd(|k|, 5) \times \gcd(|k|, 7) \times \gcd(|k|, 11)$$

Each factor is determined by a single divisibility test:

- **2-adic:** if 8\||k| then 8, elif 4\||k| then 4, elif 2\||k| then 2, else 1
- **3-adic:** if 9\||k| then 9, elif 3\||k| then 3, else 1
- **5-adic:** if 5\||k| then 5, else 1
- **7-adic:** if 7\||k| then 7, else 1
- **11-adic:** if 11\||k| then 11, else 1

The product of these gives gcd(|k|, 27720), and d = 27720/gcd. If d > 12, the value falls back to the 12ET computation d = 12/gcd(|k mod 12|, 12).

This is the identical algorithm used in the reference generator's CUDA kernel `et_gcd_27720` (lines 1414–1431).

### 7.3 Example d-Values

| k (27720ET) | gcd(k, 27720) | d = 27720/gcd | Family |
|---|---|---|---|
| 0 | 27720 | 1 | Gravity/Unison |
| 2310 | 2310 | 12 | Full-Res/EM |
| 2520 | 2520 | 11 | Undecimal/M-Theory |
| 3960 | 3960 | 7 | Septic/G₂ |
| 5544 | 5544 | 5 | Quintic/Golden |
| 13860 | 13860 | 2 | Tritone/Pivot |

---

## 8. The Tightness Function and the ∂I Incoherence Boundary {#8}

### 8.1 The Descriptor Gap ε

At each step, the orbit z_n is projected onto the 27720ET lattice. The Descriptor Gap measures how far z_n is from the nearest lattice point:

$$\varepsilon_r = \left(N_{ET} \cdot \log_2 |z_n| - k_r\right) \times \frac{1200}{N_{ET}} \quad \text{(cents, real axis)}$$

$$\varepsilon_\theta = \left(N_{ET} \cdot \frac{\theta_n}{\ln 2} - k_\theta\right) \times \frac{1200}{N_{ET}} \quad \text{(angular cents, imaginary axis)}$$

### 8.2 Tightness

$$t_r = \frac{100}{100 + |\varepsilon_r|}, \quad t_\theta = \frac{100}{100 + |\varepsilon_\theta|}, \quad t = t_r \cdot t_\theta$$

Tightness is 1.0 at a perfect lattice point (ε = 0) and falls toward K = 2/3 as |ε| approaches 50¢.

### 8.3 The ∂I Boundary

From the Incoherence Paper and the Incoherence Filter on the lattice: the ∂I boundary is crossed when tightness drops to the Koide threshold:

$$t_r \leq K = \frac{2}{3} \iff |\varepsilon_r| \geq 50 \text{¢}$$

At this boundary, the Descriptor set becomes self-defeating — the orbit is equidistant between two lattice positions with different sublattice families. This is the {P,T} Incoherence configuration: substrate and agency present, but the D-bridge is contradictory.

The tightness function is used in the fractal to **select the dominant power**: orbits near coherent lattice points (t_r > K) use their own sublattice family's power. Orbits near ∂I (t_r ≤ K) fall back to the palindromic cascade.

---

## 9. The Palindromic Cascade — Topological Invariant of N=12 {#9}

The palindromic cascade is the sublattice d-sequence generated by the base variance cascade (1/12)^n projected onto the 12ET lattice:

$$d_k = \frac{12}{\gcd(7k \bmod 12, \; 12)} \quad \text{for } k = 1, 2, \ldots, 12$$

$$\text{PALINDROME} = [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]$$

The generator g = 7 comes from round(12 · log₂(12)) mod 12 = 43 mod 12 = 7. Since gcd(7, 12) = 1, the generator is a unit of ℤ/12ℤ, and the cascade visits every residue class. The Palindrome Theorem (proved in The_Palindromic_Cascade_on_the_Semitone_Descriptor_Lattice.md §5) states: d_n = d_{12−n} for all n = 1..11. This is the discrete CPT invariance of the N=12 manifold.

The corresponding power sequence:

$$p_k = \frac{12}{d_k} = [1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1, 12]$$

This cycles through linear (p=1), quadratic (p=2), cubic (p=3), quartic (p=4), sextic (p=6), and dodecic (p=12) dynamics in a palindromic pattern.

The palindromic cascade serves as the **fallback** when the orbit is near ∂I and its own lattice position cannot be resolved.

---

## 10. The Shimmer Modulation Ψ {#10}

From the RMSAE derivation (ET_RMSAE_Complete_Derivation.md), the manifold shimmers with a 12-fold periodic modulation:

$$\Psi_k = 1 + \sqrt{V} \cdot \sin\!\left(\frac{2\pi k}{N}\right) = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi k}{12}\right)$$

where k = n mod 12 is the step index.

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ψ_k** | 1.000 | 1.144 | 1.250 | 1.289 | 1.250 | 1.144 | 1.000 | 0.856 | 0.750 | 0.711 | 0.750 | 0.856 |

Range: Ψ ∈ [1 − 1/√12, 1 + 1/√12] ≈ [0.711, 1.289]

Shimmer amplitude = √V = √(1/12) = 1/√12. This is the square root of the PD tension coefficient (BASE_VARIANCE), derived from the ET manifold's 12-fold symmetric oscillation.

---

## 11. The Complete Iteration Map {#11}

This is the core of the fractal. For z_n at iteration step n:

### 11.1 Lattice Projection

$$k_r = \text{round}(27720 \cdot \log_2 |z_n|), \quad k_\theta = \text{round}\!\left(27720 \cdot \frac{\arg(z_n)}{\ln 2}\right)$$

$$d_r = \frac{27720}{\gcd(|k_r|, 27720)}, \quad d_\theta = \frac{27720}{\gcd(|k_\theta|, 27720)}$$

$$t_r = \frac{100}{100 + |\varepsilon_r|}$$

### 11.2 Dominant Power Selection

$$d_{\text{dom}} = \begin{cases} d_r & \text{if } t_r > K = \frac{2}{3} \quad \text{(orbit near coherent lattice point)} \\ \text{PALINDROME}[n \bmod 12] & \text{if } t_r \leq K \quad \text{(orbit near ∂I boundary)} \end{cases}$$

$$p_{\text{dom}} = \frac{12}{d_{\text{dom}}}$$

### 11.3 Primary Term

$$z_{\text{primary}} = \Psi_{n \bmod 12} \cdot z_n^{\,p_{\text{dom}}}$$

The complex power z^p is computed in polar form:

$$z^p = |z|^p \cdot e^{i \cdot p \cdot \arg(z)}$$

### 11.4 24-Family Perturbation

Let ALL = [1, 2, 3, 4, 6, 12, 5, 7, 8, 9, 10, 11] be the list of all sublattice families. The Elegance-Score baseline weight for family d is:

$$\bar{w}(d) = \frac{(N/d) \cdot 100/(100 + (p+q)_d)}{\sum_{d'} (N/d') \cdot 100/(100 + (p+q)_{d'})}$$

where (p+q)_d is the canonical numerator+denominator complexity for family d.

The 24-family perturbation sums all 12 real and 12 imaginary families:

$$z_{\text{perturb}} = V \cdot \left[\sum_{d \in \text{ALL}} \bar{w}(d) \cdot |z_n|^{12/d} \cdot e^{i \cdot (12/d) \cdot \theta_n} + \sum_{d \in \text{ALL}} \bar{w}(d) \cdot |z_n|^{12/d} \cdot e^{i((12/d) \cdot \theta_n + k_\theta \cdot \ln 2 / d)}\right]$$

The factor V = 1/12 ensures this is a genuine perturbation — it adds sublattice texture from all 24 families without overriding the dominant power's topology.

### 11.5 Full Iteration

$$\boxed{z_{n+1} = \underbrace{\Psi_{n \bmod 12} \cdot z_n^{\,p_{\text{dom}}}}_{\text{dominant power (changes dynamics)}} + \underbrace{V \cdot \sum_{\text{24 families}} \bar{w}(d) \cdot r_n^{12/d} \cdot e^{i(\ldots)}}_{\text{perturbation (adds texture)}} + c}$$

**Mandelbrot mode:** z₀ = 0, c = pixel coordinate.
**Julia mode:** z₀ = pixel coordinate, c = fixed ET-derived parameter.

---

## 12. Derivative Tracking and Distance Estimation {#12}

The Jacobian of the dominant power is:

$$f'(z_n) = \Psi_{n \bmod 12} \cdot p_{\text{dom}} \cdot z_n^{\,p_{\text{dom}} - 1}$$

The derivative is tracked alongside the iteration for distance estimation:

$$dz_{n+1} = f'(z_n) \cdot dz_n + \begin{cases} 1 & \text{Mandelbrot} \\ 0 & \text{Julia} \end{cases}$$

The distance estimate (DE) at escape:

$$\text{DE} = \frac{2 \, |z| \, \ln|z|}{|dz|}$$

DE provides infinitely sharp boundary rendering regardless of zoom level, and feeds the normal-map lighting pass.

---

## 13. Escape Condition {#13}

A pixel escapes when:

$$|z_n|^2 > R^2 = (10^8)^2$$

The smooth iteration count for anti-aliased coloring:

$$\mu = n + 1 - \frac{\ln(\ln|z_n|) - \ln(\ln R)}{\ln(p_{\text{eff}})}$$

where the effective power is the mean of the palindromic cascade powers:

$$p_{\text{eff}} = \frac{1}{12} \sum_{k=0}^{11} p_k = \frac{1+2+3+4+1+6+1+4+3+2+1+12}{12} = \frac{40}{12} = \frac{10}{3} \approx 3.333$$

---

## 14. The 5-Pass Coloring Pipeline {#14}

### Pass 1 — Escape Coloring (sublattice family hue)

Hue combines the ET EM Spectrum for d_r and d_θ with log-scale smooth_n cycling:

$$H = \left(\log\!\left(\frac{\mu + 1}{\text{max\_iter} + 1}\right) \cdot K \cdot N \cdot \Psi_{\text{col}} + K \cdot H_r + (1-K) \cdot H_\theta + 0.50\right) \bmod 1$$

where COLOR_CYCLES = K·N = 8 (ET-derived), and the Koide 2:1 weighting gives D's magnitude hue twice the weight of T's phase hue.

| d | Hue | Color | d | Hue | Color |
|---|---|---|---|---|---|
| 1 | 0.00 | Red (Gravity) | 7 | 0.72 | Indigo (Septic/G₂) |
| 2 | 0.05 | Dark-red (Tritone) | 8 | 0.80 | Purple (Octet) |
| 3 | 0.33 | Green (Strong) | 9 | 0.87 | Magenta (Nonic) |
| 4 | 0.07 | Orange (Weak) | 10 | 0.83 | Violet (Decic) |
| 5 | 0.13 | Gold (Quintic) | 11 | 0.95 | Crimson (Undecimal) |
| 6 | 0.22 | Yellow-Green (Hexadic) | 12 | 0.50 | Cyan (EM) |

Near the ∂I boundary (tightness < K), pixels are darkened and hue-shifted toward d=7 Otherworld indigo — the Incoherence signature.

### Pass 2 — Normal-Map Lighting

Surface-relief from the DE gradient. Light direction from ET:

- **Angle:** θ_L = (7/12) · 2π = 210° (circle-of-fifths generator g_r = 7)
- **Elevation:** K = 2/3 radians ≈ 38.2° (Koide binding angle)

$$n_{\text{complex}} = \frac{z_{\text{esc}}}{|z_{\text{esc}}| \cdot |dz_{\text{esc}}|}$$

$$h = \frac{\text{Re}(n) \cdot \cos\theta_L + \text{Im}(n) \cdot \sin\theta_L + \sin K}{1 + \sin K}$$

### Pass 3 — Interior Coloring ({P,D} Unsubstantiated = dark matter)

Non-escaped pixels are the {P,D} state: structured potential, no agency. "Dark matter gravitates (d=1) but does not emit (d≠12)." Very dark, with a subtle hue whisper from the final orbit angle.

### Pass 4 — Orbit Trap Coloring

Minimum distance during the orbit to 4 ET lattice ring radii:

- K = 2/3 (Koide binding stability threshold)
- V = 1/12 (base variance)
- 1/φ ≈ 0.618 (golden conjugate, d=5 Fibonacci)
- 1.0 (unison, the origin)

### Pass 5 — Multi-Pass Composition

1. Normal-map lighting applied to escape coloring
2. Orbit trap layer screen-blended (capped at 0.35 weight)
3. Interior pixels replaced with dark-matter coloring

---

## 15. Post-Processing {#15}

### ACES Filmic Tone Mapping (industry standard HDR)

$$\text{ACES}(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

### Koide Gamma

$$\text{output} = \text{ACES}(x)^K = \text{ACES}(x)^{2/3}$$

The Koide ratio K = 2/3 as the gamma exponent — ET-native perceptual encoding.

### Quartic Vignette

$$\text{vignette}(x, y) = 1 - 0.18 \cdot (x^2 + y^2)^2$$

The quartic (d=4) radial falloff corresponds to the Weak force sublattice geometry.

---

## 16. Why This Is Not Any Known Fractal Type {#16}

| Property | Mandelbrot | Multibrot | Reference ET Gen | **∂I Lattice-Aware** |
|---|---|---|---|---|
| Power at step n | 2 (fixed) | p (fixed) | Weighted sum (static) | **p_dom from orbit's lattice position** |
| Power changes per step? | No | No | No (weights fixed) | **Yes — every step** |
| Power determined by... | Nothing | Parameter | Mode weights | **Orbit's own 27720ET sublattice** |
| Families used | 1 | 1 | 24 (static) | **24 (1 dominant + 23 perturbing)** |
| Lattice resolution | N/A | N/A | 27720ET | **27720ET** |
| d-values at escape | N/A | N/A | All 12 | **All 12 (both axes)** |
| Connected-set topology | Cardioid | p-fold | Cardioid-like | **Variable: depends on local d** |
| Palindromic symmetry | None | None | None | **CPT-symmetric fallback** |
| Self-referential? | No | No | No | **Yes — orbit reads its own lattice** |

The ∂I Lattice-Aware Fractal is the only known fractal whose iteration polynomial degree is determined by the orbit's position within a number-theoretic lattice.

---

## 17. Summary of All Equations {#17}

### Constants

$$N = 12, \quad V = \frac{1}{12}, \quad K = \frac{2}{3}, \quad S = 4, \quad A_0 = 137, \quad N_{ET} = 27720$$

$$R = 10^8, \quad \phi = \frac{1+\sqrt{5}}{2}, \quad p_{\text{eff}} = \frac{10}{3}$$

### Lattice Projection

$$k_r = \text{round}(27720 \cdot \log_2|z_n|), \quad k_\theta = \text{round}(27720 \cdot \arg(z_n)/\ln 2)$$

$$d_r = 27720 / \gcd(|k_r|, 27720), \quad d_\theta = 27720 / \gcd(|k_\theta|, 27720)$$

### Tightness

$$\varepsilon_r = (N_{ET} \cdot \log_2|z_n| - k_r) \times 1200/N_{ET}$$

$$t_r = 100/(100 + |\varepsilon_r|)$$

### Dominant Power Selection

$$d_{\text{dom}} = \begin{cases} d_r & t_r > 2/3 \\ \text{PALINDROME}[n \bmod 12] & t_r \leq 2/3 \end{cases}$$

$$p_{\text{dom}} = 12 / d_{\text{dom}}$$

### Shimmer

$$\Psi_k = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi k}{12}\right), \quad k = n \bmod 12$$

### Iteration

$$z_{n+1} = \Psi_k \cdot z_n^{p_{\text{dom}}} + V \cdot \sum_{d \in \text{ALL}} \bar{w}(d) \left[r_n^{12/d} e^{i(12\theta_n/d)} + r_n^{12/d} e^{i(12\theta_n/d + k_\theta \ln 2/d)}\right] + c$$

### Derivative

$$dz_{n+1} = \Psi_k \cdot p_{\text{dom}} \cdot z_n^{p_{\text{dom}}-1} \cdot dz_n + 1$$

### Distance Estimation

$$\text{DE} = 2|z|\ln|z| / |dz|$$

### Smooth Iteration Count

$$\mu = n + 1 - \frac{\ln(\ln|z_n|) - \ln(\ln R)}{\ln(10/3)}$$

### Escape Coloring (Koide 2:1 D/T weighting)

$$H = (\log(\mu+1)/\log(I+1) \cdot 8 \cdot \Psi + K \cdot H_r + (1-K) \cdot H_\theta + 0.50) \bmod 1$$

### Normal-Map Lighting

$$h = (\text{Re}(n) \cos(7\pi/6) + \text{Im}(n) \sin(7\pi/6) + \sin(2/3)) / (1 + \sin(2/3))$$

### ACES + Koide Post

$$\text{final} = \left[\frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}\right]^{2/3} \cdot (1 - 0.18 \cdot r^4)$$

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

**Document Version:** ∂I Lattice-Aware Fractal — Complete Specification v3.0
**Tools Applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law · Verification Principle
**Derivation Standard:** All mathematics ET-native, forward from {P, D, T}. Zero external axioms.
**Author:** Michael James Muller — Aevum Defluo

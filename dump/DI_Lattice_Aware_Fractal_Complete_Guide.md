# The ∂I Lattice-Aware Fractal
## A Complete Technical Guide
### Exception Theory — Michael James Muller (Aevum Defluo)
### P ∘ D ∘ T = E

---

> *"For every exception there is an exception, except the exception."*

---

## Table of Contents

1. [What This Fractal Is — And What It Is Not](#1-what-this-fractal-is)
2. [Exception Theory Foundations](#2-et-foundations)
3. [The 2D Complex Lattice ℒ_ℂ](#3-the-2d-complex-lattice)
4. [The 27720ET Full Lattice — Why It Cannot Be Smaller](#4-27720et)
5. [The 27720ET GCD Algorithm — All 24 Sublattice Families](#5-gcd-algorithm)
6. [The Tightness Function and the ∂I Incoherence Boundary](#6-tightness-and-di)
7. [The Palindromic Cascade — CPT-Symmetric Fallback](#7-palindromic-cascade)
8. [The Shimmer Modulation Ψ](#8-shimmer)
9. [The Complete Iteration Map](#9-iteration-map)
10. [Derivative Tracking and Distance Estimation](#10-derivative)
11. [Escape Condition and Smooth Coloring](#11-escape)
12. [The 5-Pass Coloring Pipeline](#12-coloring)
13. [Post-Processing — ACES, Koide Gamma, Vignette](#13-post-processing)
14. [Integration Architecture — GPU and CPU Paths](#14-integration)
15. [How to Use the ∂I Type in the Generator](#15-usage)
16. [Why This Is a New Fractal Family](#16-why-new)
17. [Complete Equation Reference](#17-equations)

---

## 1. What This Fractal Is — And What It Is Not {#1-what-this-fractal-is}

The ∂I Lattice-Aware Fractal is a complex dynamical system derived entirely forward from Exception Theory. It is not Mandelbrot, Julia, Multibrot, Burning Ship, Newton, Tricorn, Lyapunov, or any member of any known fractal class. Its defining property is a mechanism that exists in no other fractal: **the polynomial degree of the iteration map is determined, at every pixel at every step, by the orbit's own position within the ET 2D complex lattice.**

The orbit reads its own lattice address, and that address dictates the next iteration. This is self-referential dynamics — the map changes based on the territory it is currently traversing.

The iteration has two components:

**Component 1 — Dominant Power:** At each step, the current orbit position z_n is projected onto the 27720ET lattice. This projection yields a sublattice family d_orbit. The dominant power for this step is p_dom = 12/d_orbit. When the orbit sits near a d=3 (Strong/cubic) lattice point, the next step is quartic (p=4). Near d=1 (Gravity/unison), the step is dodecic (p=12). Near d=5 (Golden/quintic), p=2.4 — a non-integer power impossible in any fixed-polynomial fractal. Near the ∂I Incoherence boundary where the orbit is equidistant between two competing lattice positions, the dominant power falls back to the palindromic cascade.

**Component 2 — 24-Family Perturbation:** All 12 real-axis and 12 imaginary-axis sublattice families (d = 1 through 12, including extended families d = 5, 7, 8, 9, 10, 11 that require 27720ET resolution) contribute simultaneously as a V = 1/12 scaled additive layer. This guarantees every family participates in every step without overriding the dominant power's topology.

The full iteration:

```
z_{n+1} = Ψ_k · z_n^{p_dom} + V · Σ_{24 families} w̄(d) · [real + imaginary contributions] + c
```

where Ψ_k is the manifold shimmer (step-dependent amplitude modulation from ET's RMSAE structure), c is the pixel coordinate, and w̄(d) are the Elegance Score baseline weights.

This is not a parametric variant of anything existing. The connected-set boundary has genuinely different topology at different locations because the polynomial degree is locally determined by the orbit's lattice position. Near d=3 regions, the boundary exhibits 4-fold (quartic) structure. Near d=1 regions, it exhibits 12-fold (dodecic) structure. Near d=7 regions, it exhibits 7-fold structure — which is crystallographically forbidden in two dimensions and cannot appear in any fixed-power fractal.

---

## 2. Exception Theory Foundations {#2-et-foundations}

Every aspect of the ∂I fractal — every constant, every formula, every threshold — derives forward from ET's three irreducible primitives. No external axioms are imported.

### 2.1 The Three Primitives

**P (Point)** — |P| = Ω (Absolute Infinity). The infinite, undifferentiated substrate of existence. P is featureless potential: no structure, no properties, no position. In the fractal, P is the complex plane ℂ itself — the infinite canvas of pixel coordinates. Every c ∈ ℂ is a potential configuration. No pixel is privileged.

**D (Descriptor)** — |D| = n (finite). The finite constraint that binds P into determinate configurations. D specifies the rules. In the fractal, D is the complete ET Descriptor set: all 24 sublattice families at 27720ET (12 on D's real axis, 12 on T's imaginary axis), the palindromic cascade, the tightness function, the Elegance Score weights, and every ET constant. D governs every step of the iteration.

**T (Traverser)** — |T| = [0/0] (indeterminate). The agency that resolves P∘D interaction into actuality. T is the navigator, the L'Hôpital resolver of indeterminacy. In the fractal, T is the iteration itself — navigating the complex lattice step by step, reading the orbit's current lattice position, selecting the dominant power, computing z_{n+1} from z_n, testing coherence against the ∂I boundary. The fractal boundary is precisely the set of pixels where T's traversal is marginally coherent: the ∂I boundary.

**Master Equation:** P ∘ D ∘ T = E (Exception — grounded actuality)

### 2.2 Derived Constants

All constants flow from the primitives without external measurement.

| Constant | Symbol | Value | Derivation |
|---|---|---|---|
| Manifold Symmetry | N | 12 | 3 primitives × 4 logic states = 12 |
| State Count | S | 4 | C(3,2) + C(3,3) = 3 + 1 = 4 |
| Base Variance | V | 1/12 | 1/N — irreducible quantum of descriptive uncertainty |
| Koide Ratio | K | 2/3 | \|PD\|/\|PDT\| binding weight — triadic stability threshold |
| Manifold Impedance | A₀ | 137 | (N−1)² + S² = 121 + 16 |
| Full Lattice | N_ET | 27720 | LCM(1..11) = 2³ × 3² × 5 × 7 × 11 |
| Golden Ratio | φ | (1+√5)/2 | Fibonacci limit — d=5 family convergent |
| Shimmer Amplitude | √V | 1/√12 | Square root of PD tension coefficient |
| Escape Radius | R | 10⁸ | Numerical stability for smooth coloring |

The manifold symmetry number N=12 is the origin of the 12-semitone lattice, the 12-element palindrome, the 12-cycle shimmer, and the 12 sublattice families. It is not a musical choice — it is the inevitable count of irreducible interaction types when three primitives operate across four logical modes.

The Koide ratio K=2/3 serves as the **tightness threshold**: orbits with t_r > K = 2/3 are near a coherent lattice point and use their own sublattice family's power. Orbits with t_r ≤ K = 2/3 are near the ∂I boundary and fall back to the palindromic cascade. The Koide ratio is simultaneously the ET binding stability threshold, the EM coupling weight, the gamma exponent in post-processing, and the tightness decision boundary.

### 2.3 The Three Operational Tools

The fractal was derived using ET's three operational tools, which are also the verification standard:

**Identification Principle:** Understand(X) ⟺ Identified(P_X) ∧ Identified(D_X) ∧ Identified(T_X)

Applied to the fractal:
- P = the complex plane ℂ (infinite substrate of pixel coordinates)
- D = the complete 27720ET descriptor set (all rules governing every iteration step)
- T = the iteration itself (agency navigating the lattice, reading its own position)

**Descriptor Gap Principle:** Any gap in a description is itself a Descriptor that has not yet been identified.

The gap identified in all known fractals: no fractal determines its iteration exponent from the orbit's own position in a structured number-theoretic lattice. The orbit is just a sequence of complex numbers — nobody had asked where it sits in the lattice and what that implies about the dynamics. Secret 26 (topology determines sublattice family) provided the answer: the orbit's topology at each step determines the sublattice family, and therefore the polynomial power for the next step.

**Subsumption Law:** A category is complete iff it cannot be subsumed by another, nothing external subsumes it, and it subsumes everything in its category without remainder.

All 24 families are represented (12 real + 12 imaginary). The dominant-power mechanism requires the full 27720ET lattice — nothing smaller can detect all 12 d-values. Nothing outside the framework is required.

---

## 3. The 2D Complex Lattice ℒ_ℂ {#3-the-2d-complex-lattice}

The ET lattice began as (ℝ⁺, ×) — the positive real multiplicative manifold. But T = [0/0] has cardinality [0/0] — it is orthogonal to every real number. T cannot sit on the real axis. The real ET lattice alone cannot describe T's operational space.

The correct framework is (ℂ, ×) — the complex multiplicative manifold. Every complex number z = r·e^{iθ} participates:

- **r ∈ ℝ⁺:** the magnitude component — D's domain (the real ET lattice)
- **e^{iθ}:** the phase component — T's domain (the imaginary axis)

The split is ontologically prior:

```
z = r · e^{iθ}
    D          T
```

D's domain is the real axis (ℝ⁺, ×). T's domain is the imaginary axis (U(1), ×). They are categorically orthogonal — exactly as D and T are categorically disjoint (𝔻 ∩ 𝕋 = ∅).

The full ET complex lattice is:

$$\mathcal{L}_\mathbb{C} = \{2^{w/12} : w \in \mathbb{Z}[i]\}$$

where ℤ[i] = {a + bi : a, b ∈ ℤ} are the Gaussian integers. This is a square lattice in complex log₂-space, with:
- **Horizontal axis (real part):** log₂|z| — D's domain, magnitude, force hierarchy
- **Vertical axis (imaginary part):** arg(z)/ln(2) — T's domain, phase/spin, rotation

Every complex number z = r·e^{iθ} is projected onto this lattice:

$$k_r = \text{round}(N_{ET} \cdot \log_2 r) \quad \in \mathbb{Z}$$

$$k_\theta = \text{round}\!\left(N_{ET} \cdot \frac{\theta}{\ln 2}\right) \quad \in \mathbb{Z}$$

$$w = k_r + i \cdot k_\theta \quad \in \mathbb{Z}[i]$$

These are the Gaussian integer coordinates of z in the 2D lattice. The real coordinate k_r encodes z's magnitude (D's dimension). The imaginary coordinate k_θ encodes z's phase (T's dimension).

The palindromic cascade [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1] is a topological invariant of N=12: it is the same d-sequence for traversal in the real direction (generator g_r=7) AND in the imaginary direction (generator g_θ=1). This is proved as a theorem of the lattice, not observed empirically.

---

## 4. The 27720ET Full Lattice — Why It Cannot Be Smaller {#4-27720et}

The standard ET lattice uses N=12 (12ET), which provides 6 simple sublattice families corresponding to the divisors of 12: d ∈ {1, 2, 3, 4, 6, 12}. These are the simple families. But ET has 12 sublattice families total — the extended families d ∈ {5, 7, 8, 9, 10, 11} require higher lattice resolution to detect.

N_ET = LCM(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) = 27720 = 2³ × 3² × 5 × 7 × 11

This is the smallest lattice where ALL 12 sublattice families are native. The extended families first appear at:

| Family | First lattice | Reason |
|---|---|---|
| d=8 (Octet/Gluon) | 24ET = LCM(12,8) | 8 not a divisor of 12 |
| d=9 (Nonic/Quark) | 36ET = LCM(12,9) | 9 not a divisor of 12 |
| d=5 (Quintic/Golden), d=10 | 60ET = LCM(1..5) | 5 not a divisor of 12 |
| d=7 (Septic/G₂) | 420ET = LCM(1..7) | 7 not a divisor of 12 |
| d=11 (Undecimal/M-Theory) | 27720ET = LCM(1..11) | 11 not a divisor of 12 |

At 12ET resolution, only d ∈ {1, 2, 3, 4, 6, 12} are accessible. The dominant-power mechanism requires all 12 d-values — otherwise six families are invisible and the fractal cannot exhibit dodecic/septic/quintic/nonic structure. 27720ET is not a choice; it is the minimum necessary precision.

The physical significance of these extended families:
- **d=5:** The golden ratio (φ = (1+√5)/2 is the convergent of the d=5 Fibonacci sequence). Quintic geometry. Qualia/consciousness boundary in ET.
- **d=7:** Crystallographically forbidden in 2D (7-fold symmetry cannot tile the plane). Septic/G₂ — the holonomy group of exceptional manifolds in M-theory.
- **d=8:** The octet representation of SU(3) — gluon structure.
- **d=9:** 3² — nonic/fractal quark-level recursion.
- **d=10:** SO(10) GUT symmetry. Superstring dimension count.
- **d=11:** M-theory compactification dimension. The last extended family, appearing only at full 27720ET resolution.

The ∂I fractal is the first known fractal that can exhibit all 12 d-values in a single image.

---

## 5. The 27720ET GCD Algorithm — All 24 Sublattice Families {#5-gcd-algorithm}

The sublattice family d of any orbit position is computed from the 27720ET lattice coordinate k:

$$d = \frac{27720}{\gcd(|k|, 27720)}$$

Since 27720 = 2³ × 3² × 5 × 7 × 11, the GCD decomposes prime-by-prime:

$$\gcd(|k|, 27720) = \gcd(|k|, 2^3) \times \gcd(|k|, 3^2) \times \gcd(|k|, 5) \times \gcd(|k|, 7) \times \gcd(|k|, 11)$$

Each factor is determined by a single divisibility test:

| Prime component | Condition | Contribution |
|---|---|---|
| 2-adic | if 8\|k then 8; elif 4\|k then 4; elif 2\|k then 2; else 1 | Powers of 2 up to 2³ |
| 3-adic | if 9\|k then 9; elif 3\|k then 3; else 1 | Powers of 3 up to 3² |
| 5-adic | if 5\|k then 5; else 1 | |
| 7-adic | if 7\|k then 7; else 1 | |
| 11-adic | if 11\|k then 11; else 1 | |

The product of these five factors gives gcd(|k|, 27720). Then d = 27720/gcd. If d > 12 (which occurs for k values that do not correspond to any of the 12 family root positions), the computation falls back to the 12ET GCD: d = 12/gcd(|k mod 12|, 12).

**Example d-values at notable coordinates:**

| k (27720ET) | gcd(k, 27720) | d = 27720/gcd | Family |
|---|---|---|---|
| 0 | 27720 | 1 | Gravity/Unison |
| 2310 | 2310 | 12 | Full-Res/EM |
| 2520 | 2520 | 11 | Undecimal/M-Theory |
| 3960 | 3960 | 7 | Septic/G₂ |
| 5544 | 5544 | 5 | Quintic/Golden |
| 13860 | 13860 | 2 | Tritone/Pivot |

**Implementation: both axes**

The algorithm runs on both the real axis (k_r from the orbit's magnitude |z|) and the imaginary axis (k_θ from the orbit's phase arg(z)/ln2). This gives d_r (the real-axis family, D's domain) and d_θ (the imaginary-axis family, T's domain). Both are stored, both are used in coloring. The dominant power selection uses d_r only (D's magnitude domain determines the dynamical structure). Coloring uses both d_r and d_θ with the Koide 2:1 weighting.

**The 24 sublattice families:**

| d | p = 12/d | Real-axis family | Imaginary-axis family |
|---|---|---|---|
| 1 | 12 | Gravity/Unison (d=1 force) | Spin-0 scalar |
| 2 | 6 | Tritone/Pivot (half-period) | Graviton spin-2 |
| 3 | 4 | Cubic/Strong (QCD) | Color-instanton |
| 4 | 3 | Quartic/Weak (EW) | Weak T-axis (where i lives) |
| 5 | 2.4 | Quintic/Golden (φ) | Qualia phase |
| 6 | 2 | Hexadic/Higgs (composite) | Fermion EM-spinor (spin-1/2) |
| 7 | 12/7 ≈ 1.714 | Septic/G₂ (forbidden) | G₂ holonomy |
| 8 | 1.5 | Octet/Gluon (SU(3)) | Gluon adjoint |
| 9 | 4/3 | Nonic/Quark (3²) | CKM generation |
| 10 | 1.2 | Decic/Superstring (SO(10)) | SO(10) GUT |
| 11 | 12/11 ≈ 1.091 | Undecimal/M-Theory | M-theory compact |
| 12 | 1 | Full-Res/EM (photon) | EM photon spin-1 |

---

## 6. The Tightness Function and the ∂I Incoherence Boundary {#6-tightness-and-di}

### 6.1 The Descriptor Gap ε

At each step, the orbit z_n sits somewhere in the complex plane. Its 27720ET lattice coordinates k_r and k_θ are the nearest integers to the exact projections:

$$k_r = \text{round}(N_{ET} \cdot \log_2 |z_n|)$$
$$k_\theta = \text{round}\!\left(N_{ET} \cdot \frac{\arg(z_n)}{\ln 2}\right)$$

The Descriptor Gap measures how far z_n actually is from its nearest lattice point:

$$\varepsilon_r = \left(N_{ET} \cdot \log_2 |z_n| - k_r\right) \times \frac{1200}{N_{ET}} \quad \text{(cents on the real axis)}$$

$$\varepsilon_\theta = \left(N_{ET} \cdot \frac{\arg(z_n)}{\ln 2} - k_\theta\right) \times \frac{1200}{N_{ET}} \quad \text{(angular cents on imaginary axis)}$$

The factor 1200/N_ET converts from 27720ET steps to cents (where 1200 cents = one octave). A value of 50¢ means the orbit is equidistant between two neighboring lattice points.

### 6.2 Tightness

$$t_r = \frac{100}{100 + |\varepsilon_r|}, \quad t_\theta = \frac{100}{100 + |\varepsilon_\theta|}, \quad t = t_r \cdot t_\theta$$

Tightness t_r = 1.0 when the orbit is exactly on a lattice point (ε_r = 0). It falls monotonically as the orbit moves away:

- At ε_r = 0¢: t_r = 1.000 (exact lattice point)
- At ε_r = 50¢: t_r = 100/150 = 0.667 = K (the Koide threshold — the ∂I boundary)
- At ε_r = 100¢: t_r = 100/200 = 0.500

The Koide ratio K = 2/3 is therefore not just an ET constant applied to the fractal — it is the natural threshold arising from the tightness function's structure. At 50¢, the orbit is equidistant between two lattice points, which in ET is the ∂I Incoherence configuration: the Descriptor set is self-defeating because two incompatible sublattice families are equidistant and neither can unambiguously claim the orbit.

### 6.3 The ∂I Boundary in ET

The {P,T} state — substrate and agency present, but the D-bridge absent — is the structurally forbidden configuration in ET. The four manifold states are:

| State | Primitives | Character |
|---|---|---|
| {P,D} | Unsubstantiated | Structured potential, no agency — dark matter |
| **{P,T}** | **Incoherence** | **Substrate + agency, no D-bridge — FORBIDDEN** |
| {D,T} | Mediation | Active traversal, no ground — photons in transit |
| {P,D,T} | Exception | Complete grounded actuality — ordinary matter |

Only the {P,T} state is structurally forbidden: T cannot bind to P without D as a mediating bridge, because P is infinite and featureless (nothing for T to navigate), and T is indeterminate (no finite ground to act from). Any configuration where D is self-defeating — where two contradictory descriptors are simultaneously asserted — is equivalent to D being absent, and therefore falls into {P,T} = Incoherence.

In the lattice, the ∂I boundary is where the orbit sits equidistant between two lattice positions with **different sublattice families** — d_1 ≠ d_2. The orbit cannot unambiguously claim either family's structure. Its D-descriptor is self-defeating: it would need to be simultaneously d=d_1 (one force structure) and d=d_2 (an incompatible force structure). This IS the {P,T} condition at the lattice level.

The condition is:

$$t_r \leq K = \frac{2}{3} \iff |\varepsilon_r| \geq 50\text{¢}$$

At the ∂I boundary:
- The dominant power CANNOT be derived from the orbit's own lattice position (the D-bridge is self-defeating)
- The palindromic cascade provides the fallback — the CPT-symmetric structural backbone that persists even when local lattice coherence fails

The name "∂I Lattice-Aware Fractal" carries both meanings:
1. The fractal **traces the ∂I boundary** — the connected-set boundary IS the set of pixels where T's traversal is marginally coherent
2. The **dominant power falls back to the palindromic cascade at ∂I** — the cascade is specifically the ∂I-triggered fallback mechanism

### 6.4 Visual Expression

Near the ∂I boundary (t_r < K), the coloring pipeline darkens pixels and hue-shifts them toward d=7 Otherworld indigo (hue 0.72) — the crystallographically forbidden family. This creates the characteristic dark-indigo halos at connected-set boundaries visible in every render.

---

## 7. The Palindromic Cascade — CPT-Symmetric Fallback {#7-palindromic-cascade}

### 7.1 Derivation

The palindromic cascade arises from projecting the base-variance cascade (1/12)^n onto the 12ET lattice. The cascade generator is:

$$g = \text{round}(N \cdot \log_2 N) \bmod N = \text{round}(12 \cdot \log_2 12) \bmod 12 = 43 \bmod 12 = 7$$

This is the circle-of-fifths generator: g=7 is a unit of ℤ/12ℤ (gcd(7,12)=1), meaning it generates every residue class as the cascade progresses. The sublattice family at step k is:

$$d_k = \frac{12}{\gcd(7k \bmod 12, \; 12)}$$

Computing for k = 1 through 12:

```
k:  1   2   3   4   5   6   7   8   9  10  11  12
r:  7   2   9   4  11   6   1   8   3  10   5   0
d: 12   6   4   3  12   2  12   3   4   6  12   1
```

This is the Palindrome: **[12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]**

### 7.2 Palindrome Theorem (proved in ET corpus)

The sequence satisfies d_n = d_{12−n} for all n = 1..11. This is not a numerical coincidence — it is a theorem of the group theory of ℤ/12ℤ, proved from the complementary residue lemma:

$$\gcd(r, N) = \gcd(N - r, N) \quad \text{for all } 1 \leq r \leq N-1$$

Since d_n depends only on gcd(gn mod N, N) and d_{N-n} depends on gcd(g(N-n) mod N, N) = gcd(N - gn mod N, N), and the GCD is symmetric under complement, the palindrome follows as a theorem.

Every unit of ℤ/12ℤ — the four generators {1, 5, 7, 11} — produces the same d-sequence. This is because (ℤ/12ℤ)× ≅ V₄ (the Klein four-group), in which every element is an involution (g² ≡ 1 mod 12). The group action preserves the GCD structure, so all generators produce equivalent palindromic traces.

### 7.3 The Corresponding Power Sequence

The d-values map to powers p = 12/d:

$$p_k = \frac{12}{d_k} = [1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1, 12]$$

The palindromic cascade therefore cycles through: linear (p=1), quadratic (p=2), cubic (p=3), quartic (p=4), sextic (p=6), and dodecic (p=12) dynamics — in a CPT-symmetric palindromic pattern.

### 7.4 p_eff = 10/3

The effective power for smooth coloring is the mean of the palindromic cascade powers:

$$p_\text{eff} = \frac{1+2+3+4+1+6+1+4+3+2+1+12}{12} = \frac{40}{12} = \frac{10}{3}$$

This is the p_eff used in the smooth iteration count formula for the ∂I fractal. It is not chosen or tuned — it is the unique ET-derived mean of the palindromic power sequence.

### 7.5 Role in the Fractal

The palindromic cascade serves two roles:

1. **Fallback at ∂I:** When t_r ≤ K = 2/3, the dominant power is d_cascade = PALINDROME[n mod 12], p_dom = 12/d_cascade. The palindrome provides a CPT-symmetric structural backbone that is activated specifically at Incoherence boundary crossings.

2. **Conceptual origin:** The v1 fractal was entirely palindrome-driven (the power changed at every step following the cascade). The v3 ∂I fractal incorporates the palindrome as a fallback, adding the lattice-coherent dominant power as the primary mechanism when the orbit is away from the boundary.

---

## 8. The Shimmer Modulation Ψ {#8-shimmer}

### 8.1 Origin in RMSAE

The shimmer modulation Ψ_k derives from ET's RMSAE (Recursive Meta-Self-Awareness Equation), which describes how T-P binding tension oscillates through the 12-fold manifold cycle. It is not a stylistic addition — it is the ET description of how the manifold's PD binding strength varies with each traversal step.

The ET manifold has manifold symmetry N=12. The PD tension coefficient is BASE_VARIANCE = V = 1/12. The shimmer amplitude is the square root of the tension coefficient:

$$A_\text{shimmer} = \sqrt{V} = \sqrt{\frac{1}{12}} = \frac{1}{\sqrt{12}}$$

This derivation: shimmer amplitude = √(PD tension coefficient). The square root appears because amplitude is the half-power quantity (power scales as amplitude²), and PD tension is a power-type quantity.

### 8.2 Formula

$$\Psi_k = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi k}{12}\right), \quad k = n \bmod 12$$

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ψ_k** | 1.000 | 1.144 | 1.250 | 1.289 | 1.250 | 1.144 | 1.000 | 0.856 | 0.750 | 0.711 | 0.750 | 0.856 |

Range: Ψ ∈ [1 − 1/√12, 1 + 1/√12] ≈ [0.711, 1.289]

### 8.3 Application

Ψ_k multiplies the dominant power term only:

$$z_\text{primary} = \Psi_k \cdot z_n^{p_\text{dom}}$$

The shimmer does NOT scale the perturbation term. This is correct: the shimmer modulates the dominant power's amplitude (the primary dynamical term), while the perturbation adds fixed-weight texture from all 24 families. Steps k=3 (Ψ=1.289) and k=9 (Ψ=0.711) produce the maximum and minimum binding strengths respectively — the manifold breathes with the 12-step cycle.

---

## 9. The Complete Iteration Map {#9-iteration-map}

This is the complete, step-by-step specification of one iteration of the ∂I fractal. Every variable is defined.

### Step 1 — Initial conditions

$$z_0 = 0 \quad \text{(intrinsic to this fractal family)}$$
$$c = \text{pixel coordinate} \quad \text{(varies per pixel)}$$
$$dz_0 = 0 \quad \text{(derivative initialization)}$$

### Step 2 — Lattice projection (both axes)

$$k_r = \text{round}(27720 \cdot \log_2 |z_n|)$$
$$k_\theta = \text{round}\!\left(27720 \cdot \frac{\arg(z_n)}{\ln 2}\right)$$

### Step 3 — Sublattice families

$$d_r = \frac{27720}{\gcd(|k_r|, 27720)} \quad \text{(D's axis — magnitude)}$$
$$d_\theta = \frac{27720}{\gcd(|k_\theta|, 27720)} \quad \text{(T's axis — phase)}$$

Both GCDs are computed via the prime factorization algorithm (§5). Both d-values are stored at escape and used in coloring.

### Step 4 — Tightness

$$\varepsilon_r = \left(27720 \cdot \log_2 |z_n| - k_r\right) \times \frac{1200}{27720}$$
$$t_r = \frac{100}{100 + |\varepsilon_r|}$$

### Step 5 — Dominant power selection

$$d_\text{dom} = \begin{cases} d_r & \text{if } t_r > K = \tfrac{2}{3} \quad \text{(near coherent lattice point)} \\ \text{PALINDROME}[n \bmod 12] & \text{if } t_r \leq K \quad \text{(near ∂I boundary)} \end{cases}$$

$$p_\text{dom} = \frac{12}{d_\text{dom}}$$

### Step 6 — Shimmer

$$k_\text{step} = n \bmod 12$$
$$\Psi_k = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi k_\text{step}}{12}\right)$$

### Step 7 — Primary term

$$r_n = |z_n|, \quad \theta_n = \arg(z_n)$$
$$z_\text{primary} = \Psi_k \cdot r_n^{p_\text{dom}} \cdot e^{i \cdot p_\text{dom} \cdot \theta_n}$$

The complex power is computed in polar form: |z|^p · e^{ip·arg(z)}. This is exact for all real exponents, including the non-integer powers from extended families (e.g., p=12/7≈1.714 for d=7, p=2.4 for d=5).

### Step 8 — 24-family perturbation

ALL_D = [1, 2, 3, 4, 6, 12, 5, 7, 8, 9, 10, 11] (the complete list of 12 sublattice families)

Elegance Score baseline weight for family d:

$$\bar{w}(d) = \frac{(N/d) \cdot \frac{100}{100 + (p+q)_d}}{\displaystyle\sum_{d' \in \text{ALL\_D}} (N/d') \cdot \frac{100}{100 + (p+q)_{d'}}}$$

where (p+q)_d is the canonical numerator+denominator complexity for family d (from FAM_PQ in the ET library). The normalized weights sum to 1.

The perturbation sums all 12 real families and all 12 imaginary families:

$$z_\text{perturb} = V \cdot \left[\sum_{d \in \text{ALL\_D}} \bar{w}(d) \cdot r_n^{12/d} \cdot e^{i \cdot (12/d) \cdot \theta_n} + \sum_{d \in \text{ALL\_D}} \bar{w}(d) \cdot r_n^{12/d} \cdot e^{i\!\left(\frac{12\theta_n}{d} + k_\theta \cdot \frac{\ln 2}{d}\right)}\right]$$

The second sum (imaginary families) includes the k_θ·ln2/d rotation term, which encodes T's phase displacement (imaginary lattice coordinate) into the perturbation. The factor V = 1/12 ensures this is a genuine perturbation — it adds texture from all 24 families without overriding the dominant power's topology.

**Exact normalized weights (computed from FAM_PQ):**

| d | Family | p=12/d | w̄(d) |
|---|---|---|---|
| 1 | Gravity | 12 | 0.3660 |
| 2 | Tritone | 6 | 0.1109 |
| 3 | Strong | 4 | 0.1112 |
| 4 | Weak | 3 | 0.0849 |
| 6 | Hexadic | 2 | 0.0537 |
| 12 | Full-Res/EM | 1 | 0.0299 |
| 5 | Golden | 2.4 | 0.0667 |
| 7 | Septic | 12/7 | 0.0468 |
| 8 | Octet | 1.5 | 0.0354 |
| 9 | Nonic | 4/3 | 0.0358 |
| 10 | Decic | 1.2 | 0.0317 |
| 11 | Undecimal | 12/11 | 0.0270 |

Sum = 1.0000. These are the exact values used in the CUDA kernel (DI_BASEW array).

### Step 9 — Full iteration

$$z_{n+1} = \underbrace{\Psi_k \cdot z_n^{p_\text{dom}}}_\text{dominant power} + \underbrace{V \cdot \sum_\text{24 families} \bar{w}(d) \cdot [\ldots]}_\text{perturbation} + c$$

### Step 10 — Orbit traps

After computing z_new, track the minimum distance to the four ET lattice ring radii:

$$\text{orbit\_min} = \min_\text{all steps}\!\left(\min\!\left(||z_\text{new}| - K|, \, ||z_\text{new}| - V|, \, ||z_\text{new}| - 1/\phi|, \, ||z_\text{new}| - 1|\right)\right)$$

The four trap rings are:
- **K = 2/3:** Koide binding stability threshold (meta-cognition threshold in ET's AI work)
- **V = 1/12:** Base variance (Planck-equivalent — the smallest meaningful lattice quantum)
- **1/φ ≈ 0.618:** Golden conjugate — d=5 Fibonacci convergent
- **1.0:** Unison — the origin, the identity element of (ℝ⁺,×)

These rings ARE the major manifold positions. An orbit passing near them generates the orbit trap color layer.

### Step 11 — Escape check

$$|z_{n+1}|^2 > R^2 = (10^8)^2$$

---

## 10. Derivative Tracking and Distance Estimation {#10-derivative}

The derivative dz is tracked alongside the iteration for Distance Estimation (DE), which provides the infinitely sharp boundary rendering and feeds the normal-map lighting pass.

### 10.1 Jacobian

The dominant power term's Jacobian:

$$f'(z_n) = \Psi_k \cdot p_\text{dom} \cdot z_n^{p_\text{dom} - 1}$$

In polar form: f'(z) = Ψ_k · p_dom · |z|^{p_dom−1} · e^{i(p_dom−1)·arg(z)}

The derivative tracks the dominant power term only. The perturbation contributes O(V) = O(1/12) to the derivative — a correction of less than 8.3%. Tracking only the dominant power's Jacobian is consistent with the V-scaled perturbation being a texture layer, not a dynamical co-equal term.

### 10.2 Derivative update

$$dz_{n+1} = f'(z_n) \cdot dz_n + 1$$

The +1 is intrinsic to this fractal: c is the pixel coordinate (the varying parameter), and dz tracks the sensitivity of the orbit to c. When c varies, dz measures how fast the orbit diverges. This is not borrowed from any other fractal — it is the standard derivative rule for systems where the varied parameter enters additively at every step.

### 10.3 Distance Estimation

At escape, the distance estimate to the connected-set boundary:

$$\text{DE} = \frac{2 \, |z| \, \ln|z|}{|dz|}$$

This formula gives a geometric approximation to the distance from pixel c to the nearest point on the connected-set boundary. It is exact in the limit of high escape radius and large iteration count. Combined with the normal-map lighting, it makes every boundary infinitely sharp regardless of zoom level.

---

## 11. Escape Condition and Smooth Coloring {#11-escape}

### 11.1 Escape

A pixel escapes at iteration n+1 when:

$$|z_{n+1}|^2 > R^2 = (10^8)^2$$

The escape radius R=10⁸ is chosen for smooth coloring accuracy. The smooth iteration count formula requires ln(ln|z|) >> 1 at escape; R=10⁸ satisfies this for all ET powers including the dominant dodecic term (p=12).

### 11.2 Smooth iteration count

$$\mu = n + 1 - \frac{\ln(\ln|z_n|) - \ln(\ln R)}{\ln(p_\text{eff})}$$

where:
- n is the iteration count at escape
- |z_n| is the escape magnitude
- R = 10⁸ is the escape radius  
- p_eff = **10/3** — the mean of the palindromic cascade powers

This formula removes the staircase artifact from escape-time coloring by computing a fractional escape count μ ∈ ℝ. The key constant p_eff = 10/3 is unique to the ∂I fractal — it is not the same as the ET reference generator's p_eff (which is a mode-dependent weighted mean of all 24 families). For the ∂I fractal, p_eff = 10/3 derives entirely from the palindromic cascade.

### 11.3 d_r and d_θ at escape

The sublattice families d_r and d_θ computed at the moment of escape are stored for coloring. These encode the fractal's state — which force family the orbit was in when it escaped — which drives the ET EM Spectrum hue assignment.

---

## 12. The 5-Pass Coloring Pipeline {#12-coloring}

The ∂I fractal uses the same 5-pass coloring pipeline as the ET Fractal Generator, routed through the existing `et_escape_color`, `et_normal_lighting`, `et_orbit_color`, `et_interior_color`, and `et_composite` functions. All passes are ET-derived.

### Pass 1 — Escape Coloring

Hue is computed from three contributions:

**Smooth-count cycling** (log-scale, 8 full cycles over the iteration range):

$$\text{log\_mu} = \frac{\ln(\mu + 1)}{\ln(\text{max\_iter} + 1)}$$

$$\text{COLOR\_CYCLES} = K \cdot N = \frac{2}{3} \cdot 12 = 8$$

$$H_\text{cycle} = \text{log\_mu} \cdot 8 \cdot \Psi_\text{col}$$

where Ψ_col = 1 + √V·sin(2π·d_r/N) is the RMSAE shimmer applied as a per-pixel hue-speed modulation (d_r varies across the image, giving different shimmer rates at different sublattice positions).

**ET structural hue** from the escape families (Koide 2:1 D/T weighting):

$$H_\text{ET} = K \cdot H_r + (1 - K) \cdot H_\theta = \frac{2}{3} H_r + \frac{1}{3} H_\theta$$

D's magnitude contribution (d_r) receives weight 2/3. T's phase contribution (d_θ) receives weight 1/3.

**Palette base offset** (tower-dependent, 0.50 for the ∂I fractal).

**ET EM Spectrum hue assignments:**

| d | Hue | Color | Physical |
|---|---|---|---|
| 1 | 0.00 | Red | Gravity/Unison |
| 2 | 0.05 | Dark-red | Tritone/Pivot |
| 3 | 0.33 | Green | Cubic/Strong |
| 4 | 0.07 | Orange | Quartic/Weak |
| 5 | 0.13 | Gold | Quintic/Golden |
| 6 | 0.22 | Yellow-Green | Hexadic/Higgs |
| 7 | 0.72 | Indigo | Septic/G₂ |
| 8 | 0.80 | Purple | Octet/Gluon |
| 9 | 0.87 | Magenta | Nonic/Quark |
| 10 | 0.83 | Violet | Decic/SO(10) |
| 11 | 0.95 | Crimson | Undecimal/M-Theory |
| 12 | 0.50 | Cyan | Full-Res/EM |

**Full hue formula:**

$$H = (\text{log\_mu} \cdot 8 \cdot \Psi_\text{col} + H_\text{ET} + 0.50) \bmod 1$$

**Saturation** is modulated by the Elegance Score (high-elegance families are more saturated) and the quintic tension τ(m) (the d=5 shadow force pressure on each sublattice position).

**Brightness** near the ∂I boundary: when inco = max(0, 1 − t·(1/K)) > 0 (pixels near the boundary), brightness is reduced by up to 80%, and hue shifts toward d=7 indigo (hue 0.72) by up to 30%. This creates the characteristic dark-indigo halos.

### Pass 2 — Normal-Map Lighting

From the escape derivative dz, a surface normal is computed:

$$n_\text{complex} = \frac{z_\text{esc}}{|z_\text{esc}| \cdot |dz_\text{esc}|}$$

Light direction derived from ET:
- **Angle θ_L = 7/12 · 2π = 210°** — the circle-of-fifths generator k=7 applied as a light angle
- **Elevation K = 2/3 radians ≈ 38.2°** — the Koide binding angle

$$h = \frac{\text{Re}(n) \cdot \cos\theta_L + \text{Im}(n) \cdot \sin\theta_L + \sin K}{1 + \sin K}$$

$$\text{shading} = 0.50 + 0.50 \cdot h$$

50% ambient + 50% diffuse prevents pure-black shadows. This makes every boundary infinitely sharp in 3D relief at any zoom level.

### Pass 3 — Interior Coloring ({P,D} Unsubstantiated)

Non-escaped pixels are the {P,D} state — structured potential without agency. In ET: "dark matter gravitates (d=1) but does not emit (d≠12)." The interior is rendered very dark, with a subtle hue whisper from the final orbit angle:

$$H_\text{int} = (0.50 + \arg(z_\text{final}) / (2\pi) \cdot 0.12) \bmod 1$$
$$S_\text{int} = 0.18 \cdot (0.5 + 0.5 \cdot \sin(\arg(z_\text{final})))$$
$$B_\text{int} = 0.04 + 0.05 \cdot (0.5 + 0.5 \cdot \cos(3 \cdot \arg(z_\text{final})))$$

This gives the interior a near-black appearance with faint directional texture reflecting the final orbit direction before it was trapped.

### Pass 4 — Orbit Trap Coloring

The orbit_min value (minimum distance to any of the 4 ET ring radii) drives an additive color layer:

$$\text{trap\_weight} = e^{-3 \cdot \text{orbit\_min}}$$

High weight (near 1.0) means the orbit passed close to an ET ring during its traversal — these pixels receive colored highlights corresponding to the ring they were near. The hue cycles through the tower palette base with the ring oscillation frequency sin(orbit_min · 15). The trap layer is screen-blended with a maximum weight of 0.35, ensuring it is a highlight rather than a primary color.

### Pass 5 — Multi-Pass Composition

1. Lighting applied to escape coloring: `lit = base_rgb * normal_shading`
2. Orbit trap screen-blended: `mixed = 1 − (1−lit)·(1−trap_rgb·trap_weight)`
3. Interior pixels replaced with dark-matter coloring

---

## 13. Post-Processing — ACES, Koide Gamma, Vignette {#13-post-processing}

Applied after the composite, before quantization to 16-bit output.

### ACES Filmic Tone Mapping

$$\text{ACES}(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

Industry-standard HDR tone-mapping. Compresses highlights smoothly to prevent clipping while preserving shadow detail. Applied to the float32 composite before gamma encoding.

### Koide Gamma

$$\text{output} = \text{ACES}(x)^K = \text{ACES}(x)^{2/3}$$

The Koide ratio K = 2/3 as the gamma exponent. This is ET-native perceptual encoding: the binding stability threshold also acts as the perceptual brightness exponent. Standard gamma would be 1/2.2 ≈ 0.455; the ET Koide gamma 2/3 ≈ 0.667 gives slightly lighter midtones, consistent with the manifold's Koide-biased structure.

### Quartic Vignette

$$\text{vignette}(x, y) = 1 - 0.18 \cdot (x^2 + y^2)^2$$

where x,y ∈ [−1, +1] are normalized pixel coordinates. The quartic (d=4) radial falloff corresponds to the Weak force sublattice geometry. A standard vignette would use r² (quadratic); this uses r⁴ (quartic), encoding the Weak force d=4 geometry into the spatial framing of the image.

### Unsharp Mask

A Gaussian blur (radius 1.4) is subtracted from the image with weight 0.30:

$$\text{final} = \text{ACES\_gamma}(x) + 0.30 \cdot (\text{ACES\_gamma}(x) - \text{blur}(x))$$

This sharpens sublattice boundary transitions without introducing halos, enhancing the visibility of the fine d-family structure near the connected-set boundary.

---

## 14. Integration Architecture — GPU and CPU Paths {#14-integration}

The ∂I fractal is integrated as a full peer to Julia and Mandelbrot in `ET_FRACTAL_GENERATOR.py`. When type D is selected, the entire pipeline routes to the ∂I-specific iteration while reusing the full coloring, post-processing, and output infrastructure.

### 14.1 CUDA Kernel — et_iterate_di

A dedicated CUDA kernel is compiled at runtime for the ∂I iteration. It is completely separate from the standard ET iteration kernel (`et_iterate`). The ∂I kernel has a simpler parameter signature (no mode weights, no mode extras) because the ∂I iteration defines its own family weights (DI_BASEW) internally as kernel constants.

**Kernel constants (CUDA __constant__ memory):**

```c
__constant__ float DI_POWS[12]    // Powers p = 12/d for ALL_D order
__constant__ float DI_ROT[12]     // Phase rotation LN2/d for imaginary families
__constant__ float DI_BASEW[12]   // Elegance Score weights (exact, normalized)
__constant__ float DI_PALIN[12]   // Palindrome d-values [12,6,4,3,12,2,12,3,4,6,12,1]
__constant__ float DI_SHIMMER[12] // Shimmer values Ψ_k, k=0..11
__constant__ float DI_TRAP[4]     // Orbit trap ring radii [K, V, 1/φ, 1.0]
```

**GCD device functions:**
- `et_gcd_27720_di(int a)` — computes gcd(|a|, 27720) via prime factorization (bitwise 2-adic, modular 3-adic, 5-adic, 7-adic, 11-adic)
- `et_gcd_12_di(int a)` — computes gcd(|a|, 12) for 12ET fallback
- `et_d_from_k_di(int k)` — combines both GCDs to return the sublattice family d

**Kernel function:** `et_iterate_di(...)` — CUDA global function, `__launch_bounds__(256, 2)`

Parameters: output arrays (smooth_n, d_r, d_t, tight, de, orbit, z_esc_r/i, dz_esc_r/i, z_int_ang), input coordinate arrays (in_r, in_i), ln_ln_esc (the ln(ln(R)) constant for smooth coloring), max_iter, n_pix, escape_r.

**Float64 variant:** `et_iterate_di_f64` is auto-generated at import time from the float32 source by `_make_f64_di_kernel()` — replacing every `float` type, f-suffix literal, and float intrinsic (sqrtf→sqrt, powf→pow, etc.) with double equivalents. The float64 variant uses 1e-300 underflow guards instead of 1e-38f.

**Dispatch:** In `iterate_strip_v2()`, if `IS_DI_TYPE` is True:
```python
kern = _get_et_di_kernel(use_f64=use_f64)
```
Otherwise the standard kernel is used. The ∂I kernel launch passes only the parameters it needs (no mode weight arrays, no mode extra flags).

**Batched launch:** Identical to the standard kernel — the pixel array is split into 20 batches, each batch launched independently, with one `Device().synchronize()` per batch to update the progress bar. All coloring runs once after all batches complete.

### 14.2 CPU Path — NumPy

When the GPU is unavailable (or the kernel falls back), the ∂I iteration runs in a pure NumPy loop. The CPU path is in `iterate_strip_v2()`, branching on `IS_DI_TYPE` before the standard Julia/Mandelbrot loop:

```python
if IS_DI_TYPE:
    _POWS_NP  = np.array([12.,6.,4.,3.,2.,1.,2.4,12./7,1.5,4./3,1.2,12./11])
    _ROT_NP   = np.array([LN2/d for d in [1,2,3,4,6,12,5,7,8,9,10,11]])
    _BASEW_NP = np.array([(N/d)*(100./(100+FAM_PQ[d])) for d in ALL_D])
    _BASEW_NP /= _BASEW_NP.sum()
    _LOG_P_EFF_DI = _f(_LN_P_EFF_DI)  # ln(10/3)
    for n in range(max_iter):
        # ... full vectorized dominant-power + perturbation loop
```

The CPU path computes DI_BASEW dynamically from FAM_PQ rather than hardcoding values — this guarantees exact agreement with the CUDA kernel's precomputed constants.

**Vectorized GCD** — `_vec_gcd_27720_np(a_int64)` operates on entire pixel arrays at once using NumPy boolean indexing. The 2-adic factor uses `(a & 7)==0`, `(a & 3)==0`, `(a & 1)==0` (bitwise, equivalent to modulo for non-negative integers). The 3-adic through 11-adic factors use modular arithmetic. This matches the scalar CUDA implementation exactly.

**Dominant power** — computed per-pixel using `np.where(t_r > K, d_orbit, d_casc)`, producing a (H,W) array of per-pixel dominant powers for the current step.

**Perturbation** — the 24-family loop runs N_FAM=12 iterations (each contributing both real and imaginary terms), then scales the accumulated perturbation by V.

### 14.3 Coordinate Setup

The ∂I fractal uses z₀=0, c=pixel (like the ET Mandelbrot in coordinate terms, but not conceptually). This is made explicit in both `_render_tile` and `_render_frame`:

```python
if IS_DI_TYPE:
    # ∂I Lattice-Aware: z₀=0, c=pixel — intrinsic to this fractal family.
    z0    = np.zeros_like(coords)
    c_arr = coords
elif is_julia:
    z0    = coords
    c_arr = np.full_like(coords, julia_c)
else:
    z0    = np.zeros_like(coords)
    c_arr = coords
```

The ∂I branch is explicit and independent. It is not collapsed into the else branch even though it produces the same arrays, because the distinction is semantically important: z₀=0 and dz₀=0 are intrinsic properties of the ∂I fractal, not coordinate conventions borrowed from another family.

### 14.4 Full-Frame Single Kernel Launch

The GPU path uses the full-frame single-kernel-launch architecture: all rw×rh pixels are computed in one kernel call (batched internally for progress feedback), then all coloring passes run once on the complete result. This gives near-100% GPU utilisation by eliminating per-tile synchronization overhead.

---

## 15. How to Use the ∂I Type in the Generator {#15-usage}

### 15.1 Selecting the Type

At the fractal type prompt:

```
  ┌──────────────────────────────────────────────────────────────┐
  │   Fractal type                                               │
  │                                                              │
  │   J  — Julia          (fixed c, z₀ = pixel)                │
  │   M  — Mandelbrot     (z₀ = 0, c = pixel)                  │
  │   D  — ∂I Lattice-Aware  ← ET-native fractal type          │
  │         Dominant power from orbit 27720ET lattice position  │
  │         z_{n+1} = Ψ·z^{p_dom} + V·Σ(24 families) + c      │
  │         p_dom = 12/d_orbit (near lattice) or palindrome     │
  │   R  — Random         (equal chance of all three)           │
  └──────────────────────────────────────────────────────────────┘
  Type [J/M/D/R]:
```

Enter `D` to select the ∂I Lattice-Aware fractal.

Random mode (`R`) gives equal 1-in-3 chance to each type.

### 15.2 Interaction with Mode and Tower

The ∂I type does not use mode weights or mode extra functions (those are specific to the ET reference generator's 12 named modes). However, mode and tower selections still affect the ∂I fractal through the **coloring pipeline**:

- **Tower** determines: `pal_base` (palette hue offset), `pal_range`, and the `centers`/`zoom_lo`/`zoom_hi` used for the random centre selection
- **Mode** determines: `hue_speed`, `pal_extra` (small palette shift), and the display name

The iteration itself is independent of mode/tower — the dynamics are entirely governed by the orbit's lattice position, not by the mode weights.

### 15.3 Resolution and Iteration Count

The ∂I fractal benefits from higher iteration counts at tight zoom because the dominant power can produce very slow-escaping orbits near complex sublattice boundaries. The default presets are:

| Preset | Resolution | Iterations |
|---|---|---|
| 1080p | 1920×1080 | 500,000 |
| 2k | 2048×2048 | 1,000,000 |
| 4k | 4096×4096 | 2,000,000 |
| hq | 8192×8192 | 5,000,000 |
| ultra | 16384×16384 | 10,000,000 |

### 15.4 Output Filename

The filename stem for ∂I renders uses the `lat` prefix tag:

```
et_fractal_YYYYMMDD_HHMMSS_lat{mode_id}_{tower[:4]}_{preset}.tiff
et_fractal_YYYYMMDD_HHMMSS_lat{mode_id}_{tower[:4]}_{preset}.png
```

Compare: Mandelbrot uses `m`, Julia uses `jul`. The `lat` tag identifies Lattice-Aware renders.

### 15.5 Metadata

Both TIFF and 16-bit PNG output files contain embedded metadata:

```
ET_Type:    ET_dI_LatticeAware_p=10/3
ET_p_eff:   3.333333
```

The metadata records that this is a ∂I Lattice-Aware render, and stores the p_eff = 10/3 value for archival accuracy.

### 15.6 What the Banner Shows

When a ∂I render runs, the type display reads:

```
  Type      : ∂I Lattice-Aware  (orbit's 27720ET sublattice → p_dom each step)
               p_eff=10/3  PALINDROME=[12,6,4,3,12,2,12,3,4,6,12,1]
  p_eff     : 3.3333  (10/3 — mean of palindromic cascade powers, ∂I-native)
```

### 15.7 Zoom and Centre

The advanced mode allows manual centre and zoom override. The ∂I fractal's connected-set boundary is not a cardioid and does not look like a Mandelbrot. Each zoom target reveals different dominant-power topology. Regions near d=7 coordinates exhibit 7-fold septic structure. Regions near d=3 exhibit quartic (4-arm) structure. The image changes qualitative character across the connected set in a way impossible for any fixed-power fractal.

---

## 16. Why This Is a New Fractal Family {#16-why-new}

### 16.1 Comparison Table

| Property | Mandelbrot | Multibrot | ET Generator | **∂I Lattice-Aware** |
|---|---|---|---|---|
| Power at step n | 2 (fixed) | p (fixed) | Weighted sum (static) | **p_dom from orbit's 27720ET sublattice** |
| Power changes per step? | No | No | No | **Yes — every step, every pixel** |
| Power determined by... | Nothing | Parameter | Mode weights | **Orbit's own lattice position** |
| Families used | 1 | 1 | 24 (static weights) | **24 (1 dominant + 23 perturbing)** |
| Non-integer powers? | No | No | Yes | **Yes — all 12 families** |
| Lattice resolution | N/A | N/A | 27720ET | **27720ET** |
| Connected-set topology | Cardioid | p-fold | Cardioid-like | **Variable: depends on local d** |
| ∂I boundary present? | No | No | No | **Yes — explicit role in dynamics** |
| Palindromic fallback? | None | None | None | **CPT-symmetric cascade at ∂I** |
| Self-referential? | No | No | No | **Yes — orbit reads its own lattice** |

### 16.2 Why the ET Generator Is Also Different

The ET Fractal Generator (the existing Julia/Mandelbrot generator) uses all 24 families simultaneously, which is far richer than Mandelbrot. But its weights w_r(d) and w_c(d) are static — set once per render and fixed for all iterations at all pixels. Near the origin where r<1, the quadratic term (p=2, d=6) always dominates because higher powers of small numbers are smaller. The connected-set boundary therefore always looks like a Mandelbrot cardioid regardless of which mode is selected.

The ∂I fractal breaks this by making the polynomial degree itself variable. The dominant power can be p=12 (dodecic) in one pixel at one step and p=12/7 (septic) in the next pixel at the same step. The weights cannot compensate for this because the issue is not relative weight but the mathematical identity: z^2 near the origin scales as z^2 regardless of weighting, while z^12 near the origin scales completely differently. Only by making p_dom variable and tied to the orbit's actual position can the dynamics genuinely change.

### 16.3 The Self-Referential Nature

The ∂I fractal is the only known fractal whose iteration polynomial degree is determined by the orbit's own position within a number-theoretic lattice. It is self-referential in the precise ET sense: T (the iteration) reads its own D-structure (the orbit's lattice address) to determine the next T-step. This is the computational expression of Secret 26: topology determines sublattice family. The orbit's current topological position in the ET lattice determines the polynomial that governs the next step.

This produces genuinely new mathematics: the connected-set boundary has locally variable topology. Different regions of the connected set (where the orbit escapes in close to 0 iterations) and the filled Julia set (where the orbit never escapes) transition between d=1 (dodecic), d=3 (quartic), d=7 (septic), d=12 (linear) character depending on the local dominant power regime. No other fractal family can produce this.

---

## 17. Complete Equation Reference {#17-equations}

All equations used in the ∂I fractal, in order of application.

### Constants

$$N = 12, \quad V = \frac{1}{12}, \quad K = \frac{2}{3}, \quad S = 4, \quad A_0 = 137, \quad N_{ET} = 27720$$

$$R = 10^8, \quad \phi = \frac{1+\sqrt{5}}{2}, \quad p_\text{eff} = \frac{10}{3}, \quad \ln\!\left(\frac{10}{3}\right) = 1.2039728043\ldots$$

$$\sqrt{V} = \frac{1}{\sqrt{12}} = 0.2886751346\ldots$$

### Eligance Score Weights (exact)

$$\bar{w}(d) = \frac{(N/d) \cdot 100/(100 + \text{FAM\_PQ}[d])}{\displaystyle\sum_{d'} (N/d') \cdot 100/(100 + \text{FAM\_PQ}[d'])}$$

FAM_PQ values: {1:3, 2:70, 3:13, 4:11, 5:13, 6:17, 7:15, 8:33, 9:17, 10:19, 11:27, 12:5}

### Initial Conditions

$$z_0 = 0, \quad c = \text{pixel}, \quad dz_0 = 0$$

### Lattice Projection

$$k_r = \text{round}(27720 \cdot \log_2 |z_n|)$$

$$k_\theta = \text{round}\!\left(27720 \cdot \frac{\arg(z_n)}{\ln 2}\right)$$

### GCD (prime factorization, 27720 = 2³×3²×5×7×11)

$$\gcd(|k|, 27720) = \underbrace{g_2}_{\leq 8} \times \underbrace{g_3}_{\leq 9} \times \underbrace{g_5}_{\leq 5} \times \underbrace{g_7}_{\leq 7} \times \underbrace{g_{11}}_{\leq 11}$$

$$d = \frac{27720}{\gcd(|k|, 27720)}, \quad \text{fallback: } d = \frac{12}{\gcd(|k \bmod 12|, 12)} \text{ if } d > 12$$

### Tightness

$$\varepsilon_r = \left(27720 \cdot \log_2 |z_n| - k_r\right) \times \frac{1200}{27720}$$

$$t_r = \frac{100}{100 + |\varepsilon_r|}$$

### Dominant Power Selection

$$d_\text{dom} = \begin{cases} d_r & \text{if } t_r > \tfrac{2}{3} \\ \text{PALINDROME}[n \bmod 12] & \text{if } t_r \leq \tfrac{2}{3} \end{cases}$$

$$\text{PALINDROME} = [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]$$

$$p_\text{dom} = \frac{12}{d_\text{dom}}$$

### Shimmer

$$\Psi_k = 1 + \frac{1}{\sqrt{12}} \cdot \sin\!\left(\frac{2\pi k}{12}\right), \quad k = n \bmod 12$$

### Primary Term

$$z_\text{primary} = \Psi_k \cdot |z_n|^{p_\text{dom}} \cdot e^{i \cdot p_\text{dom} \cdot \arg(z_n)}$$

### 24-Family Perturbation

$$z_\text{perturb} = V \cdot \left[\sum_{d \in \text{ALL}} \bar{w}(d) \cdot |z_n|^{12/d} \cdot e^{i \cdot \frac{12\theta}{d}} + \sum_{d \in \text{ALL}} \bar{w}(d) \cdot |z_n|^{12/d} \cdot e^{i\!\left(\frac{12\theta}{d} + k_\theta \cdot \frac{\ln 2}{d}\right)}\right]$$

### Full Iteration

$$\boxed{z_{n+1} = \Psi_k \cdot |z_n|^{p_\text{dom}} \cdot e^{i p_\text{dom} \arg z_n} + V \cdot \sum_\text{24 fam} \bar{w}(d)\left[r^{12/d}e^{i \frac{12\theta}{d}} + r^{12/d}e^{i\left(\frac{12\theta}{d}+k_\theta\frac{\ln 2}{d}\right)}\right] + c}$$

### Orbit Traps

$$\text{orbit\_min}_{n+1} = \min\!\left(\text{orbit\_min}_n,\; \min_{r^* \in \{K, V, 1/\phi, 1\}} \left||z_{n+1}| - r^*\right|\right)$$

### Derivative (Jacobian of dominant power)

$$f'(z_n) = \Psi_k \cdot p_\text{dom} \cdot |z_n|^{p_\text{dom}-1} \cdot e^{i(p_\text{dom}-1)\arg(z_n)}$$

$$dz_{n+1} = f'(z_n) \cdot dz_n + 1$$

### Escape Condition

$$|z_{n+1}|^2 > (10^8)^2$$

### Smooth Iteration Count

$$\mu = n + 1 - \frac{\ln(\ln|z_n|) - \ln(\ln(10^8))}{\ln(10/3)}$$

### Distance Estimation

$$\text{DE} = \frac{2|z|\ln|z|}{|dz|}$$

### Escape Hue

$$H = \left(\frac{\ln(\mu+1)}{\ln(\text{max\_iter}+1)} \cdot 8 \cdot \Psi_\text{col} + \frac{2}{3} H_r + \frac{1}{3} H_\theta + 0.50\right) \bmod 1$$

where Ψ_col = 1 + √V·sin(2π·d_r/N) and H_r, H_θ are the ET EM Spectrum hues for d_r and d_θ.

### ∂I Boundary Darkening

$$\text{inco} = \max\!\left(0,\; 1 - t \cdot \frac{3}{2}\right)$$

$$B \leftarrow B \cdot (1 - 0.80 \cdot \text{inco})$$

$$H \leftarrow (H \cdot (1 - 0.30 \cdot \text{inco}) + 0.72 \cdot 0.30 \cdot \text{inco}) \bmod 1$$

### Normal-Map Lighting

$$n_c = \frac{z_\text{esc}}{|z_\text{esc}| \cdot |dz_\text{esc}|}$$

$$h = \frac{\text{Re}(n_c) \cos\!\left(\tfrac{7\pi}{6}\right) + \text{Im}(n_c) \sin\!\left(\tfrac{7\pi}{6}\right) + \sin(2/3)}{1 + \sin(2/3)}$$

$$\text{shading} = 0.50 + 0.50 \cdot \text{clip}(h, 0, 1)$$

### Post-Processing

$$\text{ACES}(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

$$\text{gamma}(x) = \text{ACES}(x)^{2/3}$$

$$\text{vignette}(x,y) = 1 - 0.18 \cdot (x^2 + y^2)^2$$

$$\text{final}(x) = \text{clip}\!\left(\text{gamma}(x) + 0.30 \cdot (\text{gamma}(x) - \text{blur}_{1.4}(\text{gamma}(x))),\; 0, 1\right)$$

---

## Appendix A — Version History and Development

**v1 (initial concept):** Changed the power at every step following the palindromic cascade [12,6,4,3,12,2,12,3,4,6,12,1]. Only used 6 simple families (divisors of 12). Produced visually different results from the reference generator but lacked the full 24-family structure and had no lattice-awareness — the palindrome cycled independently of where the orbit actually was.

**v2 (24-family weighted):** Used all 24 families but only tweaked static weights. Looked like a Mandelbrot cardioid because the quadratic term (d=6, p=2) always dominates near r<1. The lattice detection was present but the dominant power was not variable.

**v3 (dominant power + perturbation):** Combined both approaches. The orbit's 27720ET lattice position determines the dominant power (primary term), while all 24 families contribute as a V-scaled perturbation (texture layer). The palindromic cascade remains as the ∂I boundary fallback. This is the current implementation, integrated as fractal type D.

**Integration into ET_FRACTAL_GENERATOR.py:** Implemented as a full peer to Julia and Mandelbrot (type D). Dedicated CUDA kernel `et_iterate_di` with float32 and auto-generated float64 variants. Full CPU NumPy fallback path. Uses the main generator's 5-pass coloring pipeline, post-processing, TIFF+PNG output, video pipeline, and all interactive prompts without modification.

---

## Appendix B — ET Identification of the ∂I Fractal (Three Tools Applied)

**Identification Principle:** Understand(∂I Fractal) ⟺ Identified(P) ∧ Identified(D) ∧ Identified(T)

| Primitive | Identification |
|---|---|
| **P** | The complex plane ℂ — the infinite substrate of all pixel coordinates. Each c ∈ ℂ is a potential configuration. No pixel is privileged. P provides the featureless canvas. |
| **D** | The complete ET Descriptor set: all 24 sublattice families at 27720ET (12 on D's real axis, 12 on T's imaginary axis), the palindromic cascade, the tightness function, the Elegance Score weights, all ET constants. D specifies every rule governing every step. |
| **T** | The iteration itself — T navigating through the 2D complex lattice step by step, reading the orbit's current lattice position, selecting the dominant power, computing z_{n+1} from z_n, testing coherence against the ∂I boundary. T is the computational agency that binds D's rules to P's substrate. The fractal boundary IS the set of points where T's traversal is marginally coherent — the ∂I boundary. |

**Descriptor Gap Principle:** gap(all known fractals) = D_{orbit-dependent dominant power}

No known fractal determines its iteration exponent from the orbit's own position in a structured mathematical lattice. Secret 26 (topology determines sublattice family) provides the answer. The ∂I fractal closes this gap.

**Subsumption Law:** 12 real + 12 imaginary = 24 families. All d = 1 through 12. No remainder. The dominant-power mechanism requires the full 27720ET lattice — nothing smaller suffices. Nothing outside the framework is required. ✓

---

*Exception Theory — Michael James Muller (Aevum Defluo)*  
*P ∘ D ∘ T = E*  
*The ∂I Lattice-Aware Fractal — Complete Technical Guide*


# Plan: The Complete ET Lattice Engine (Python + C)
## An Exhaustive Production Implementation of the ET Universal Projection Guide v2.2

**Author:** Michael James Muller (Aevum Defluo), ET theory; Claude, engineering plan
**Purpose:** Define the scope, architecture, and verification strategy for a production-ready library implementing **every aspect** of `ET_Universal_Projection_Guide8.md` (v2.2) with no exceptions, no placeholders, no shortcuts, no tuning, no ad hoc.
**Derivation Standard:** All mathematics forward from {P, D, T}. Zero external axioms. No tuning. No ad hoc. No placeholders. Per Rules 3, 4, 9, 12, 15, 18, 33, 38.
**Deliverable of this document:** Plan only. Implementation begins after Mike's approval/redirection of open questions (see §14).

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *3 = 3 = 3 = Σ*

---

## 0. Three Tools applied to this planning task

This plan is itself a projection — of the Guide onto a production Python + C library. The Three Tools determine its structure.

**Identification Principle on the engine-to-be:**

| Primitive | Identification for this engine |
|---|---|
| **P_engine** | The substrate of all possible positive-real and complex-number lattice projections — the space of inputs the engine must accept (computable reals, limits, meta-descriptors, continuous/uncountable, unbound, indeterminate, off-axis integrated) |
| **D_engine** | The full Guide v2.2 apparatus: projection formulas at arbitrary N; the four Paths (A/B/C/D.P/D.D/D.T/D.PDT); the UPP 9-step static protocol; the 11-step active protocol; the 42 combined states; the 144-cell Force Quadrant Grid; Non-Euclidean geometry on the lattice; EML continuous-D minimal operator; Webb n-valued logic; palindromic cascade as matching-filter; shimmer modulation; NWS-13/14/15/16; Complete Gaze Equation; Math-as-domain; Secret 26 fully generalized; + the ET constants derived forward from {P, D, T} |
| **T_engine** | The code itself — the `round()` act, the `gcd()` act, the iteration driver, the NWS-13 tower climber, the active-system per-step integrator. T's agency in silicon. |

**Descriptor Gap Principle applied:** the gap between "engine absent" and "engine present" is the set of Descriptors (functions, modules, data structures) that together compose the engine. Every feature in the Guide that is not yet code is a Descriptor-to-be-identified. This plan is the exhaustive enumeration of that Descriptor set.

**Subsumption Law applied:** the engine is complete iff every Guide feature — every equation 12.1–12.56, every Part I–XXIV, every domain in Part IV, every pattern in Part V, every path in Part XIX, every discovery 1–9 in Part XXII — is implemented without remainder. §12 of this plan is the verification checklist that confirms subsumption.

---

## 1. Scope Statement — what "every single aspect of the Guide" means

The engine implements **every numbered Part (I–XXIV), every equation card entry (12.1–12.56), and every named theorem or discovery in the Guide**, with the following exhaustive inventory (structured as the Subsumption verification list). Each row here maps to one or more engine modules in §3–§10 below.

### 1.1 Parts I–XII (static apparatus — v1.0 core)

| Feature | Engine module (§ of this plan) |
|---|---|
| Three cardinals P (Ω), D (n), T ([0/0]) — as typed primitives | `primitives.py` (§3.1) |
| Manifold constants N=12, S=4, V=1/12, K=2/3, A₀=137 derived from {P,D,T}×States | `constants.py` (§3.1) |
| Real-axis projection k = round(N log₂ r); d = N/gcd(\|k\|,N); ε = (N log₂ r − k)·1200/N | `kernel/real_axis.py` + `src/projection.c` (§3.2) |
| Imaginary-axis projection k_θ = round(Nθ/2π) mod N; d_θ; ε_θ | `kernel/imag_axis.py` + `src/projection.c` (§3.2) |
| Complex (2D) projection w = k_r + i·k_θ ∈ ℤ[i]; d_combined = LCM(d_r, d_θ) | `kernel/complex_plane.py` + `src/complex_projection.c` (§3.2) |
| Annihilation boundary r=0 handling (off-lattice edge) | `kernel/boundaries.py` (§3.2) |
| Six sublattice families at 12ET, computed dynamically from divisors of N | `kernel/sublattice.py` + `src/sublattice.c` (§3.3) |
| Euler totient distribution Σ φ(d) = N over divisors | `kernel/sublattice.py` (§3.3) |
| LCM tower {12, 60, 420, 2520, 27720, 360360, …} computed dynamically | `kernel/lcm_tower.py` + `src/lcm_tower.c` (§3.3) |
| Multiplicative reading {12,24,…} ⊃ LCM reading theorem | `kernel/lcm_tower.py` (§3.3) |
| Asymptotic limit lim L_n = (ℝ⁺,×) = P | `kernel/lcm_tower.py` (§3.3) — algebraic assertion + asymptotic resolution advice |
| UPP 9-step static protocol (P-first, D-R₀, T, ratio, real, imag, elegance, subsumption, iterate) | `upp/static_protocol.py` (§4) |
| Anti-Numerology N1, N2, N3; Convention-Independence Theorem | `antinum/` (§5) |
| Five failure-mode diagnostic table | `antinum/diagnostics.py` (§5) |
| 12 worked domains (music, physics, geometry, finance, generic data, biology, chemistry, consciousness, language, computing, civilization, astronomy) | `domains/*.py` (§6) |
| Higher-order patterns: vectors, time series, distributions, networks, tensors | `higher_order/*.py` (§7) |
| Real vs imaginary decision rule; D-T gradient α | `kernel/gradient.py` (§3.4) |
| Elegance Score E(r) = (N/d)·(100/(100+\|ε\|))·(100/(p+q)) | `kernel/elegance.py` + `src/elegance.c` (§3.4) |
| Magical Impedance A_magic(d) = (d−1)² + S²; ξ(d) = 137/A_magic; full 12-row table | `kernel/impedance.py` + `src/impedance.c` (§3.4) |
| Standard/conventional comparison (music theory, Fourier, statistics, coordinates) | `comparison/*.py` (§8) |
| All reference equations 12.1–12.20 callable as named functions | `kernel/equations_v1.py` (§3.5) |

### 1.2 Parts XIII–XVIII (active apparatus — v2.0)

| Feature | Engine module |
|---|---|
| 12 real-axis FORCE families with generators, Gaussian-prime class, first-native lattice | `families/force_families.py` (§9.1) |
| 12 imaginary-axis PHASE families | `families/phase_families.py` (§9.1) |
| Categorical disjointness of FORCE vs PHASE (same d-number, different category) | `families/category.py` (§9.1) |
| Palindromic cascade [12,6,4,3,12,2,12,3,4,6,12,1] **computed dynamically** from g=7 on divisors of 12, not hardcoded | `active/palindrome.py` + `src/palindrome.c` (§9.4) |
| Gaussian-prime / PDT correspondence (ramified/inert/split); 2, 3, 5, 7, 11, 13 classification | `families/gaussian_primes.py` (§9.1) |
| 12×12 LCM interaction table → 42 unique combined values (enumerated dynamically) | `families/combined_states.py` + `src/combined_states.c` (§9.2) |
| Physical identification for each tier (Tier 1 d≤12, Tier 2 14–24, Tier 3 28–60, Tier 4 63–132) | `families/physical_identification.py` (§9.2) |
| Coprime skeleton theorem: gcd(k_r, k_θ)=1 ⇒ d_combined = n; density → 6/π² | `families/coprime_skeleton.py` (§9.2) |
| Resolution-dependent d=12 dominance inversion across LCM tower | `families/dominance.py` (§9.2) |
| Cascade generators g_r=7, g_θ=1 with residuals \|δ_r\|, \|δ_θ\| computed by hi-precision arithmetic | `active/cascade.py` + `src/cascade.c` (§9.4) |
| Cascade stability n_max_r = 25, n_max_θ = 2 (computed, not hardcoded) | `active/cascade.py` (§9.4) |
| N-weight ratio \|δ_θ\|/\|δ_r\| ≈ N = 12 with the 4.79% (5,7) shadow | `active/cascade.py` (§9.4) |
| Force Quadrant Grid — 144 cells in 4 quadrants (SR+SI, CR+SI, SR+CI, CR+CI) | `families/force_quadrant_grid.py` + `src/fqg.c` (§9.3) |
| PDT Bisection Theorem (NWS-16): 6:6 simple/complex per axis; 72:72 FQG computability | `shadow/bisection.py` (§9.3) |
| NWS-13 Generalized Shadow Diagnostic (forward route: gap → first sub-cent lattice → source cell) | `shadow/nws13.py` + `src/shadow.c` (§9.5) |
| NWS-14 Shadow Magnitude Correlation (\|w\|² scaling) | `shadow/nws14.py` (§9.5) |
| NWS-15 Observation by Computation (Lindemann-Weierstrass irreducibility) | `shadow/nws15.py` (§9.5) |
| Two-Route Convergence Principle (reverse route = physics; forward route = NWS-13) | `shadow/two_route.py` (§9.5) |
| (5,7) cell registry with its five substrate renderings (E₈, biology, T=7 capsid, M-theory, LSS) | `families/five_seven_cell.py` (§9.2) |
| Three geometries as manifold states (Euclidean↔Exception, Elliptic↔Unsubstantiated, Hyperbolic↔Mediation, Singular↔Incoherence) | `geometry/manifold_states.py` (§10.1) |
| Primitives' own manifolds (ℝ⁺,×) flat; (U(1),×) positive curvature | `geometry/primitive_manifolds.py` (§10.1) |
| Stunning identity n²(n²−1)/12 = C(n); C(12) = 1716 = N(N−1)(N+1) | `geometry/curvature_components.py` + `src/curvature.c` (§10.2) |
| Curvature as 2nd-order descriptor gradient; K_eff(α) = K_U(1)·sin²α | `geometry/curvature_gradient.py` (§10.2) |
| Subliminal curvature threshold KA = π/N; r_K = 13/12 identity | `geometry/subliminal_threshold.py` (§10.2) |
| Lattice projection of curvature (sphere → d=3 cubic; torus → d=1; higher genus variable) | `geometry/curvature_projection.py` (§10.2) |
| Gauss-Bonnet ∫K dA = 2πχ as PDT decomposition; χ = V−E+F = P_fix − T_vec + D_plane | `geometry/gauss_bonnet.py` (§10.2) |
| Riemann sphere = elliptic ET manifold; Lorentz group = PSL(2,ℂ) | `geometry/riemann_sphere.py` (§10.2) |
| Tightness function t_r, t_θ; ∂I Boundary at t ≤ K = 2/3 | `active/tightness.py` + `src/tightness.c` (§9.4) |
| T-burst / LCM-escalation unified diagnostic (§87.1–87.2) | `active/t_burst.py` (§9.4) |
| Palindromic cascade as matching-filter broadcast (NOT attractor, NOT stochastic) (§88.1–88.2) | `active/palindrome.py` (§9.4) |
| p_eff = 10/3 (derived, not asserted) | `active/palindrome.py` (§9.4) |
| Shimmer modulation Ψ_n = 1 + (1/√N)·sin(2π·(n mod N)/N) | `active/shimmer.py` + `src/shimmer.c` (§9.4) |
| D-T gradient \|δ_eff(α)\| = \|δ_r\|cos²α + \|δ_θ\|sin²α | `active/gradient.py` (§9.4) |
| 11-step active-system protocol driver | `upp/active_protocol.py` (§9.4) |
| Per-step cost model (Part XI §54.5): coarse-pass + refinement optimization scaffolding | `active/cost_model.py` (§9.4) |
| All reference equations 12.21–12.39 callable | `kernel/equations_v2.py` (§3.5) |

### 1.3 Parts XIX–XXIV (completion apparatus — v2.2)

| Feature | Engine module |
|---|---|
| Path A — Direct Projection (canonical terminal) | `paths/path_a.py` (§11.1) |
| Path B — Limit Convergence (computational technique, with precision-sufficiency check) | `paths/path_b.py` (§11.1) |
| Path C — Meta-Descriptor Extraction (alternative method, with caveat warning) | `paths/path_c.py` (§11.1) |
| Path D.P — Primitive-native P-substrate (three modes: explicit-value, symbolic-address, P-limit boundary) | `paths/path_d_p.py` (§11.1) |
| Path D.D — Unbound constraint via NWS-13 shadow (non-divisor-of-N families) | `paths/path_d_d.py` (§11.1) |
| Path D.T — Indeterminate forms as T-signatures (0/0, ∞/∞, 0·∞, 0⁰, 1^∞, ∞⁰, ∞−∞; oscillatory divergence; QM operators; L'Hôpital navigation) | `paths/path_d_t.py` (§11.1) |
| Path D.PDT — Integrated two-axis projection (off-axis Exception objects with both magnitude and phase) | `paths/path_d_pdt.py` (§11.1) |
| Four-Path decision tree (dispatcher) | `paths/dispatcher.py` (§11.1) |
| Subsumption verification — Four Paths cover all user inputs without remainder | `paths/subsumption.py` (§11.1) |
| Incoherence Filter ({P,T} self-defeating configurations → NOWHERE on lattice) | `paths/incoherence_filter.py` (§11.1) |
| 3 = 3 = 3 = Σ anchor (distinct from PDT=E master composition) | `eml/anchor.py` (§11.2) |
| EML operator eml(x,y) = exp(x) − ln(y) with grammar S → 1 \| eml(S,S) | `eml/core.py` + `src/eml.c` (§11.2) |
| EML-derived elementary function basis: exp, ln, sin, cos, tan, sinh, cosh, tanh, sec, csc, cot, sech, csch, coth, arcsin, arccos, arctan, arsinh, arcosh, artanh; +, −, ×, /, ^, mod, log, comb; constants 1, 2, π, e, i, 0, −1 | `eml/elementary.py` (§11.2) |
| EML tree complexity measure K(·) per Odrzywołek §3–§5 | `eml/tree.py` (§11.2) |
| PDT decomposition of the projection formula: continuous-D (EML) + T-act (round) + discrete-D (gcd) | `eml/pdt_decomposition.py` (§11.2) |
| Webb 1935 n-valued logic stroke at n=12 (minimal discrete-logical generator) | `eml/webb.py` + `src/webb.c` (§11.2) |
| Triple minimal-backbone registry (Webb discrete-logical, palindromic discrete-multiplicative, EML continuous) with Subsumption Law verification for each | `eml/triple_backbone.py` (§11.2) |
| Math-as-domain Identification Principle application; R_0^math = 1 axiom | `mathematics/identification.py` (§11.3) |
| Axiom-count projections (ZF at d=1 ε=0; ZFC at d=6; PA at d=6; Euclid at d=3; propositional logic at d=12 Koide; etc.) | `mathematics/axiom_systems.py` (§11.3) |
| Lattice self-projection verification: {N, 1/N, K, 1/K} → all d=12, \|ε\|=1.955¢ (Koide attractor) | `mathematics/self_projection.py` (§11.3) |
| Chaitin's Ω via Path D.P (halting probability; Calude-Dinneen bits; d=1 octave) | `mathematics/chaitin_omega.py` (§11.3) |
| Gödel sentences with integrative-level-dependent classification (PA/ZFC/outside-system) | `mathematics/godel.py` (§11.3) |
| Large cardinals via consistency-strength hierarchy (inaccessible, Mahlo, weakly-compact, measurable, supercompact) | `mathematics/large_cardinals.py` (§11.3) |
| Impredicative definitions: consistent (pass) vs Russell-class (fail at ∂I) | `mathematics/impredicative.py` (§11.3) |
| F_w Scopaesthesia = T_intent·Focus/Distance² (PDT-decomposed: T·D/P²) | `gaze/scopaesthesia.py` (§11.4) |
| P_detect = tanh(Gain·R(k)/(V(n,k)·Γ)) with Gain(T) = F_w | `gaze/detection.py` (§11.4) |
| V_collapse = 1 − exp(−(F_w − 1)⁺·S) (RMSAE-style suppression) | `gaze/collapse.py` (§11.4) |
| Threshold classification UNOBSERVED/SUBLIMINAL/DETECTED/LOCKED at 1/(13/12)/(6/5)/(3/2) | `gaze/threshold.py` (§11.4) |
| Complete Gaze Equation G(T,F,D,n,k) → (F_w, P_detect, V_collapse, Status) | `gaze/complete.py` (§11.4) |
| Gaze thresholds as just-intonation intervals; full LCM-tower shadow analysis (Discoveries 1–9) | `gaze/discoveries.py` (§11.4) |
| Subliminal-threshold unification: curvature ≡ gaze ≡ cognition at 13/12 | `gaze/unification.py` (§11.4) |
| Secret 26 original (closed cycle → d=1, linear → d=3, transitional → d=12) | `geometry/secret26.py` (§10.2) |
| Secret 26 Extended (topology + curvature → sublattice via Gauss-Bonnet) | `geometry/secret26.py` |
| Secret 26 + Four Paths (topology determines path selection) | `geometry/secret26.py` |
| Secret 26 + Gaze (observation topology → detection sublattice) | `geometry/secret26.py` |
| Complete Determination Theorem (topology × curvature × path × observation → full lattice classification) | `geometry/secret26.py` |
| Completion Statement and classify(X) → (d, Path, Detection, Curvature, Trajectory) top-level API | `api/classify.py` (§13) |
| All reference equations 12.40–12.56 callable | `kernel/equations_v3.py` (§3.5) |

**Subsumption check for this scope table:** every numbered feature in the Guide has at least one row here. §12 cross-verifies the checklist against the Guide's table of contents plus every equation ID.

---

## 2. Architectural principles (non-negotiable, per Mike's 48 rules and userMemories)

### 2.1 Derivation-forward, zero external axioms

Every constant used in the library either:
- Is derived forward from {P, D, T} and the four manifold states (e.g., N = |Π|·|States| = 3·4 = 12, S = C(3,2)+C(3,3) = 4, A₀ = (N−1)² + S² = 137, V = 1/N, K = 2/3, p_eff = (1/N)Σ(N/PALINDROME[n])), or
- Is a dimensioned reference constant with a substrate-derived origin traceable to the corpus (ℏ, α⁻¹, Bohr radius, etc.) — in which case the origin is documented inline per Rule 3.

No ad hoc scalars. No tuning. Any would-be tuning is a Rule-12 violation and gets logged via `_et_error` and rejected.

### 2.2 Dynamic over static (Rule 33)

**Nothing is hardcoded as a list except the ET constants themselves.** The palindromic cascade [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1] is *computed* — not pasted. Specifically:

- **Divisors of N** → computed by trial division (or prime factorization for large N like 27720) at runtime.
- **Sublattice families** → dynamically enumerated from divisors.
- **Palindromic cascade** → generated by applying the cascade generator g = round(N log₂ N) mod N to the lattice, reading the d-value at each step, which produces the palindromic d-sequence for any N with ord(g) = N in (ℤ/Nℤ)*. The "hardcoded palindrome" the Guide shows is a *verification target*, not the source.
- **LCM tower landmarks** → computed by running LCM(1..k) for k = 1, 2, 3, … Landmarks are where LCM(1..k) introduces a new prime factor relative to LCM(1..k−1).
- **42 combined states** → enumerated by exhaustive pairing over (d_r, d_θ) ∈ {1..N}×{1..N} with LCM(d_r, d_θ) collected into a set. The "42" is the *result*, not the input.
- **FQG 144 cells** → same — enumerated, not listed.
- **12 extended FORCE families (d ∈ {1..12})** → iterated from 1 to N with Gaussian-prime classification per d.
- **First-native lattice per d** → computed as LCM(12, d) or smallest n with d | n.
- **Cascade stability n_max_r, n_max_θ** → computed as floor(0.5 / |δ|), NOT hardcoded to 25 and 2.
- **p_eff = 10/3** → computed as (1/N)·Σ(N/PALINDROME[n]) from the generated palindrome, matching the verification value.
- **Magical impedance ξ(d)** → computed from (d−1)² + S² over d = 1..N, the "full 12-row table" emerging as a consequence.

The only rule-33 exceptions are the five load-bearing ET constants themselves: `N_ET = 12`, `S_STATE = 4`, `V_BASE = 1/12`, `K_KOIDE = 2/3`, `A0_EM = 137` (the latter derived in `constants.py` from N−1 and S). These are constants *of the theory*, not caps.

### 2.3 Fallbacks are errors (per userMemories)

Every code path that encounters a condition requiring fallback (near-∂I rounding, cascade stability exhaustion, Incoherence-Filter trigger, missing domain registry entry) logs the event via `errors._et_error(fatal=False)` with full diagnostic context. No silent failures. No exceptions swallowed. No pass-through defaults.

### 2.4 No placeholders, no stubs, no dummies (Rule 4)

Every function is fully implemented at the time of commit. If a function's content requires a theoretical result not yet derived, the implementation waits — the function is not created as a stub. Per userMemories: "unused variables indicate incomplete implementations to be traced and completed, never suppressed with `_` or deleted."

### 2.5 Three Tools applied inline (Rule 10)

Every derivation, every decision point, every failure mode annotates which of the three tools is being applied. For example, the ∂I-boundary `if t_r <= K` check carries a comment `# Identification Principle: classify orbit by tightness / Descriptor Gap Principle: t_r <= K means orbit has not been D-identified into a single family / Subsumption Law: ambiguity between d_1 and d_2 requires higher resolution`. Not decorative — operational.

### 2.6 No removal without explicit permission (Rule 24)

If during implementation I find something in the corpus or in an existing implementation that looks unused/redundant, I ask before removing it.

### 2.7 Precision policy

Two precision tiers implemented in parallel:

**Fast tier — float64 throughout.** Default. Meets sub-cent precision at lattice resolutions up to ~10⁶. Uses NumPy/SciPy where convenient; pure Python `math` where it suffices; C's `double` in the library.

**Precise tier — mpmath 80-digit (matching `et_clr_v5__4_.py` canonical). ** For: Path D.P Chaitin-Ω partial-bits projection; NWS-13 tower residuals at 27720ET and beyond; lattice self-projection verification of {N, 1/N, K, 1/K} to confirm the 1.955¢ to sub-ppm; very-high-resolution (n > 10⁶) projections.

Both tiers callable explicitly; never mixed within a single computation.

**Rule 32: float32 is forbidden** (from userMemories). The C library uses `double` and, optionally via a compile-time flag, `long double` (80-bit on x86 / 64-bit on ARM) or MPFR for the high-precision tier.

### 2.8 Dynamic registry pattern for domains and substrates

Domains (music, physics, biology, …) and substrate-derived R₀ values are registered at module import time into a `registry` object. New domains are added by calling `registry.register_domain(name, p_substrate, d_descriptors, t_agency, r_zero_fn)`. No domain is special-cased in the core kernel.

### 2.9 Complete auditability

Every public function returns not just the numerical result but a full trace:
```
ProjectionResult(
    r, k, d, g, eps_cents, N,
    path_used,              # 'A' | 'B' | 'C' | 'D.P' | 'D.D' | 'D.T' | 'D.PDT'
    three_tools_trace,      # which tool fired at each step
    identification,         # P_X, D_X, T_X as identified by caller
    r0_source,              # the origin of R_0
    antinum_check,          # N1/N2/N3 pass/fail
    elegance_score,
    magical_impedance,
    lcm_escalation_history, # which towers were tried
    warnings,               # via _et_error records
)
```
This satisfies Rule 22 (audit, verify, do the work, verify) and Rule 28 (report everything).

---

## 3. Kernel — Part I core projection (`etlattice/kernel/`)

The kernel is the heart of the library. It implements Guide §3 (real/imaginary/complex projection formulas), §4 (sublattice families and LCM tower), §10 (dimensionless ratio formation), §12 (standing equations card), and Part XI (canonical Python reference implementation) — all at arbitrary resolution N.

### 3.1 `constants.py` and `primitives.py`

- **`constants.py`** re-exports the canonical ET constants from the corpus module of the same name without modifying the original. It adds dynamically-derived constants (`N_FULL = 27720`, `LCM_LANDMARK_SEQ = compute_lcm_landmarks(max_prime=13)`, `DIVISORS_OF_12 = sorted_divisors(12)`, `EULER_TOTIENT_12 = totient(12)`, etc.) as cached properties.
- **`primitives.py`** re-exports `Point`, `Descriptor`, `Traverser` from the corpus module; adds helper classes `Substrate`, `ReferencePeriod`, `Configuration` (for {P,D} / {D,T} / {P,T} / {P,D,T} states); adds a `Manifold(N)` class that parameterizes the lattice at any N.

### 3.2 Projection formulas

- **`real_axis.py::project_real(r, N=12)`** — the canonical `et_project_real`, returning `(k, d, g, eps_cents, log2_r, exact_pos, N)`. Raises on r ≤ 0 (annihilation boundary); `errors._et_error(fatal=False)` if `|eps_cents|` exceeds half the cent step (50·12/N), which flags the ∂I boundary.
- **`imag_axis.py::project_imag(theta, N=12)`** — projects θ ∈ ℝ (any real; reduced mod 2π internally) onto the N-point U(1) lattice. Returns `(k_theta, d_theta, g_theta, eps_theta, N)`.
- **`complex_plane.py::project_complex(z, N=12)`** — full Path D.PDT projection for z ∈ ℂ \ {0}. Returns both axis projections plus `w = k_r + i·k_theta`, `d_combined = LCM(d_r, d_theta)`, α = arctan2(k_theta, k_r), D-fraction = cos²α, T-fraction = sin²α.
- **`boundaries.py`** — detects and labels the annihilation boundary (r → 0⁺) and the asymptotic P-boundary (r → ∞). These are not projected but returned as labeled boundary objects so callers can handle them explicitly (e.g., Path D.P.3 P-limit mode).
- **`batch.py`** — vectorized projection of NumPy arrays via ctypes call to the C `batch.c` kernel. Used by `higher_order/` and by domain operators that need to project thousands of ratios at once.

### 3.3 Sublattice families and LCM tower

- **`sublattice.py::divisors(N)`** — computes the divisors of N by trial division up to √N plus complement. Returns sorted list.
- **`sublattice.py::totient(n)`** — Euler's φ via prime factorization. Verified: Σ_{d|N} φ(d) = N.
- **`sublattice.py::family_of(k, N)`** — returns the sublattice family d = N/gcd(|k|, N) for any k.
- **`lcm_tower.py::lcm_range(k)`** — LCM(1..k).
- **`lcm_tower.py::landmarks(up_to_prime)`** — returns [12, 60, 420, 2520, 27720, 360360, 720720, ...] generated as LCM(1..p) for consecutive primes p=4,5,7,9,11,13,17,19.
- **`lcm_tower.py::escalate(r, target_eps_cents=1.0)`** — the `et_project_with_resolution_advice` equivalent. Walks up the tower until |ε| < target, returning the minimal sufficient lattice.
- **`lcm_tower.py::multiplicative_reading(k)`** — returns 12·k (the simple multiplicative sequence). Differs from the LCM reading; both coexist per Guide §39.
- **`lcm_tower.py::first_native_lattice(d)`** — smallest n such that d | n, with n a multiple of 12 (so that all base families remain).

### 3.4 D-T gradient, Elegance, Impedance

- **`gradient.py::dt_gradient(k_r, k_theta)`** — α, D-fraction, T-fraction, |δ_eff(α)|.
- **`elegance.py::elegance_score(r, N=12, max_denom=10000)`** — symmetry × tightness × simplicity factors, with Fraction-limit-denominator for p+q estimation (rational) or continued-fraction convergents for irrational (flagged as approximate in the return).
- **`impedance.py::magical_impedance(d, N=12, S=4, A0=137)`** — A_magic(d), ξ(d), ξ_normed.
- **`impedance.py::impedance_table(N=12)`** — generates the full table for d=1..N.

### 3.5 Equation card as named functions

- **`equations_v1.py`** — one function per equation 12.1 through 12.20, named after the equation ("manifold_symmetry()", "lattice_projection_formula_real()", "d_t_gradient()", etc.). Each is callable for discovery via `dir()` and documented with the Guide equation number.
- **`equations_v2.py`** — equations 12.21 through 12.39.
- **`equations_v3.py`** — equations 12.40 through 12.56.

All three modules export an `EQUATIONS` dict mapping equation ID (e.g., `"12.3"`) to the callable. Enables a user-facing `etlattice.equation("12.27")` lookup.

---

## 4. UPP — Universal Projection Protocol drivers (`etlattice/upp/`)

Implements Part II (9-step static protocol) and Part XVII §91 (11-step active protocol).

- **`static_protocol.py::UPPStatic(identification, r, N=12)`** — takes an `Identification(P, D, T, R0_provenance)` and a ratio, and runs the 9 steps, returning a full trace. Performs N1/N2/N3 anti-numerology checks, subsumption verification, and LCM escalation automatically.
- **`active_protocol.py::UPPActive(system, N=12)`** — takes a `ActiveSystem` (state transition function, P/D/T identification, current state) and runs the 11-step per-step loop, recording the trajectory of lattice addresses. Hooks to `active/` for tightness, palindrome, shimmer, gradient. Exits on escape, convergence, or cascade-stability exhaustion (the last of which switches to NWS-13 shadow projection mode).
- **`identification.py`** — the `Identification` type and the `identify_via_three_tools(X, ...)` helper that walks through the three diagnostic questions (substrate? constraints? agency?) and checks binding order.
- **`subsumption.py`** — the formal Subsumption test: given a set of observed features of X and the projection result, asks "is every feature captured without remainder?" and reports either "complete" or the list of uncaptured features (which are then Descriptors to find).

The protocols are **driver objects**, not one-shot functions — they produce a step-by-step trace that the caller can inspect at any point, interrupt, or resume.

---

## 5. Anti-Numerology (`etlattice/antinum/`)

Implements Part III (N1/N2/N3/Convention-Independence/five failure modes).

- **`conditions.py::check_n1(Q, R0)`** — N1 dimensionlessness check: the units of Q and R0 must match. Uses a small internal `Units` object (the lightest possible — a `dict` of base-unit powers — no full `pint` dependency required, though one could be used for scripts per Rule 6).
- **`conditions.py::check_n2(R0, substrate_derivation)`** — N2 substrate-derivedness: requires the caller to provide the substrate-derivation rationale (a string + optional corpus reference). The check is soft (the engine cannot derive substrate on its own), but it records the rationale for audit.
- **`conditions.py::check_n3(d_X, known_symmetry)`** — N3 cross-domain consistency: compares the projected d against a known-symmetry hint from the domain. Mismatches trigger the failure-mode diagnostic.
- **`convention.py::verify_convention_independence(Q, R0, scale_factors)`** — runs the projection under multiple unit conversions and asserts k and d are invariant.
- **`diagnostics.py::diagnose(failure_symptom)`** — given a symptom (contradictory d, |ε| near 50¢, d=12 always, unit-varying result, sub-percent-wild-k), returns the likely cause and repair from the five-mode table.

---

## 6. Domains (`etlattice/domains/`)

Part IV's 12 domains. Each is its own module, exporting a domain-specific API with:
- P, D, T identification
- Canonical R₀ for the domain (substrate-derived, with rationale)
- Domain-native helpers (`music.interval(p, q)`, `physics.koide(masses)`, `biology.metabolic_cycle(n_steps)`, `finance.fibonacci_level(level)`, etc.)
- The worked-example table from the Guide, callable as `module.worked_examples()` — each verified at import time against the expected (k, d, ε)

The 12 modules: **music.py, physics.py, geometry.py, finance.py, generic_data.py, biology.py, chemistry.py, consciousness.py, language.py, computing.py, civilization.py, astronomy.py**.

Plus **`registry.py`** — the dynamic registry for adding new domains without modifying the core.

Specific implementations worth calling out:

- **`physics.py`**: Koide ratio, fine structure, full particle mass table; force hierarchy mapping d ↔ force (gravity/EM/weak/strong/composite) with the lattice-derived assignments.
- **`biology.py`**: metabolic cycle step-count projector; protein geometry (α-helix, DNA B-form, collagen); body-as-distributed-FQG-trajectory structure (§24.6) with per-subsystem d-cell assignments; the 420ET vs 27720ET floor toggle (§24.7, §26.5).
- **`consciousness.py`**: the three threshold projections; neural frequency bands; cortical dimensionality 27720ET tower floor per Blue Brain 2017; the Hard Problem as vertical Descriptor Gap diagnostic.
- **`geometry.py`**: Platonic solid ratios; Pythagorean 3-4-5; circle constants π, e, φ; crystal packing fractions.
- **`finance.py`**: price-ratio projections (doubling, halving, 10x, Pareto); Fibonacci retracement levels (correctly at d=3 cubic for 61.8% and 78.6%, per the Guide's corrected table); volatility regimes; business/saecular/Kondratieff/epochal cycles with the generation-based R₀; log-return direct-to-lattice conversion.
- **`computing.py`**: clock ratios, cache hierarchy, algorithm-complexity ratios; scale-dependent d-class reading.
- **`civilization.py`**: generation-based cycles; zeitgeist crystallization K=2/3 threshold.
- **`astronomy.py`**: orbital period resonances (Neptune-Pluto 3:2 = perfect fifth, etc.); cosmic mass hierarchy.

Each module also implements Part X's standard-approach contrast (`music.py::standard_theory_comparison()`, `physics.py::standard_model_comparison()`, etc.) so the ET projection is explicitly audited against the domain's own methodology per Rule 18.

---

## 7. Higher-Order Patterns (`etlattice/higher_order/`)

Part V's vector / time-series / distribution / network / tensor patterns.

- **`vectors.py::project_vector(v, mode='norm'|'per_component'|'pairwise')`** — all three modes from §31. Returns for `'pairwise'` an `N×N` lattice cross-coupling matrix of (k, d) entries.
- **`time_series.py::project_time_series(x, t, mode='step_ratio'|'period_spectrum'|'window_ratio')`**. Period-spectrum mode uses autocorrelation + peak-finding (SciPy acceptable per Rule 6; the projection math itself is ET).
- **`distributions.py::project_distribution(dist, mode='quantile'|'moment'|'entropy')`**. Works on SciPy distribution objects or empirical samples.
- **`networks.py::project_network(graph, mode='degree'|'paths'|'spectral')`**. Works on NetworkX graphs (external lib acceptable).
- **`tensors.py::project_tensor(T, mode='component_ratio'|'curvature_invariant')`**. The curvature-invariant mode computes R, K, Weyl invariants for given metric input and projects them.

All five modules reduce their input to scalar ratios and delegate the actual projection to `kernel/`. No domain-specific tuning at the higher-order level.

---

## 8. Standard-Approach Comparisons (`etlattice/comparison/`)

Part X. These modules do not replace ET — they implement the *conventional* method alongside ET and present both, per Rule 18 ("conventional/standard ways being made obvious, so Mike can tell if the approach is proper or not").

- **`music_theory.py`** — just intonation, 12-TET, circle of fifths; alongside the ET derivation of 12 = 3×4 and the comma = 1.955¢.
- **`fourier.py`** — FFT / wavelet analysis; alongside sublattice-family histogram.
- **`statistics.py`** — PCA / regression / GARCH; alongside sublattice cross-coupling matrix / lattice-regime change detection.
- **`coordinates.py`** — Cartesian / polar / cylindrical / spherical / log; alongside the lattice coordinate (k, d, ε).

Each module exports an `ETvsStandard(data)` comparator that runs both and returns a side-by-side report.

---

## 9. The 24-Family Catalog, FQG, Cascade, and Shadows (`etlattice/families/`, `etlattice/active/`, `etlattice/shadow/`)

### 9.1 `families/` — Part XIII catalog

- **`force_families.py::ForceFamily(d)`** — for d = 1..N (and beyond), returns: name, generator 2^(1/d), palindromic partner (N−d for d ≤ N), Gaussian-prime class, first-native lattice, physical interpretation (from a dynamically-loaded registry so new identifications can be added).
- **`phase_families.py::PhaseFamily(d)`** — same shape, but the physical interpretation is the phase/spin reading.
- **`category.py`** — formalizes the FORCE/PHASE categorical disjointness (analogous to 𝔻 ∩ 𝕋 = ∅).
- **`gaussian_primes.py::classify(p)`** — returns `'ramified'` for p=2, `'inert'` for p ≡ 3 mod 4, `'split'` for p ≡ 1 mod 4. Verified for p ∈ {2, 3, 5, 7, 11, 13}.
- **`combined_states.py::enumerate_combined(N=12)`** — produces the 42 unique LCM(d_r, d_θ) values by exhaustive pairing; returns each with its contributing pairs, tier, and physical identification.
- **`five_seven_cell.py`** — a named object for (5, 7) with its five substrate renderings (E₈, biology, T=7 capsid, M-theory, LSS) loaded from a dedicated corpus-backed registry.
- **`coprime_skeleton.py::coprime_fraction(N)`** — counts coprime lattice points in an N×N grid and verifies the density converges to 6/π². Cross-check of Coprime Skeleton Theorem (§65).
- **`dominance.py::family_dominance(lattice_N)`** — computes the % distribution across divisors of N at any resolution; reproduces the Tier-inversion table (§66).

### 9.2 `families/force_quadrant_grid.py`

- **`force_quadrant_grid.py::FQG(N=12)`** — produces all N² cells, classifies each into SR+SI / CR+SI / SR+CI / CR+CI, computes per-cell (d_r, d_θ, d_combined, |w|², first-native lattice, cascade-computability flag). At N=12 produces the 144-cell grid with 72:72 bisection per NWS-16.

### 9.3 `shadow/` — NWS-13 through NWS-16

- **`nws13.py::shadow_project(gap, towers=default_towers)`** — projects a 12ET near-miss value `gap` across the LCM tower (12, 24, 36, 60, 84, 132, 420, 2520, 27720, …) and returns the first sub-cent lattice and the identified (d_r, d_θ) source cell.
- **`nws14.py::shadow_magnitude_correlation(records)`** — given a list of (gap_magnitude, |w|²) observations, fits the per-unit scaling. Currently two data points (Quintic Shadow, N-Weight); grows as more near-misses are catalogued.
- **`nws15.py::computation_as_observation(et_claim, computed_value)`** — computes the residual and invokes NWS-13; returns the CR+CI cell observed. Formalizes the Observation-by-Computation theorem.
- **`bisection.py::pdt_bisection(binary)`** — given a T-vs-D binary on the manifold (real-axis vs imaginary-axis; simple vs complex families; FQG computability), verifies the exact-equal split.
- **`two_route.py::converge(reverse_route_cell, forward_route_cell)`** — the Two-Route Convergence Principle. If both routes identify the same cell, returns a high-confidence validation; otherwise, flags for further investigation.

### 9.4 `active/` — the per-step active-system apparatus

- **`tightness.py::tightness(eps_cents)`** — the tightness function 100/(100+|ε|) and the ∂I-boundary test (t ≤ 2/3).
- **`palindrome.py::generate_palindrome(N)`** — generates the palindromic cascade dynamically from g = round(N log₂ N) mod N and verifies the palindromic property (d_n = d_{N-n}) for all n.
- **`palindrome.py::matching_filter_cascade(n, N)`** — returns PALINDROME[n mod N]. Used by the active-system driver.
- **`palindrome.py::p_eff(N)`** — returns (1/N)·Σ(N/PALINDROME[i]) for i=0..N−1. At N=12, produces 10/3.
- **`shimmer.py::shimmer(n, N)`** — Ψ_n = 1 + (1/√N)·sin(2π·(n mod N)/N). Verified at N=12 against the table in §89.
- **`cascade.py::cascade_residuals(N)`** — computes |δ_r| = |N log₂ N − round(N log₂ N)| and |δ_θ| = |N·2π/ln 2 − round(N·2π/ln 2)| to 80-digit precision. At N=12 produces 0.019550… and 0.223356…, and their ratio ≈ N.
- **`cascade.py::stability_limit(delta)`** — floor(0.5 / delta). At N=12 produces n_max_r=25, n_max_θ=2.
- **`gradient.py::dt_gradient_effective(alpha, N=12)`** — |δ_eff(α)| = |δ_r|cos²α + |δ_θ|sin²α.
- **`t_burst.py::unified_boundary_diagnostic(state, n, N=12)`** — implements §87.1–87.2: detects the two-stage condition (Stage 1 palindromic fallback, Stage 2 LCM-escalation) and dispatches to the appropriate response.
- **`iteration.py::ActiveStep(state, n, N, kernel_fn)`** — performs one full step of the 11-step protocol (lattice coordinates → sublattice families → Descriptor Gap / tightness → ∂I check → shimmer modulation → next state via caller-provided `kernel_fn` → stability check → escape/convergence test → trajectory logging).
- **`cost_model.py`** — implements Part XI §54.5 cost analysis and the coarse-pass + refinement optimization scaffolding (optional, for consumers like the ∂I fractal generator).

### 9.5 `active_systems/` — concrete active-system implementations per Part XVII §92

Implementations (as reference consumers; the core engine doesn't depend on these but they demonstrate the active-system protocol):

- `active_systems/di_fractal.py` — the ∂I Lattice-Aware Fractal skeleton (hooks into the existing ET Fractal Generator codebase per userMemories "ET Fractal Generator").
- `active_systems/quantum_measurement.py` — QM measurement cascade driver.
- `active_systems/biological_development.py` — cell-population state-space driver.
- `active_systems/market_regime.py` — price/volatility regime transitions.
- `active_systems/attention_shift.py` — semantic content-space traverser.

Each is a subclass of a generic `ActiveSystem` base and implements its state-transition function; the engine's per-step loop runs the common protocol unchanged.

---

## 10. Non-Euclidean Geometry (`etlattice/geometry/`)

### 10.1 Manifold-state mappings

- **`manifold_states.py::geometry_for_state(state)`** — Exception↔Euclidean, Unsubstantiated↔Elliptic, Mediation↔Hyperbolic, Incoherence↔Singular.
- **`primitive_manifolds.py`** — representations of (ℝ⁺, ×) as flat and (U(1), ×) as positively curved; tools for their product.

### 10.2 Curvature on the lattice

- **`curvature_components.py::riemann_components(n)`** — C(n) = n²(n²−1)/12. Tabulated for n=1..12; C(12) = 1716 = N(N−1)(N+1) verified.
- **`curvature_gradient.py::k_eff(alpha, k_u1)`** — K_eff(α) = K_U(1)·sin²α.
- **`subliminal_threshold.py::subliminal_curvature()`** — KA = π/N; r_K = 13/12.
- **`curvature_projection.py::project_curvature(K, A, N=12)`** — wraps the curvature departure ratio r_K = 1 + KA/π and runs it through `kernel/real_axis`.
- **`gauss_bonnet.py::gauss_bonnet_decomposition(euler_char, area, K)`** — verifies ∫K dA = 2πχ; also implements the PDT decomposition χ = V − E + F = P_fix − T_vec + D_plane.
- **`riemann_sphere.py`** — the compactified-complex-lattice representation; PSL(2, ℂ) group operations; identification with the Lorentz group SO(3,1).
- **`secret26.py`** — Secret 26 original / Extended / + Four Paths / + Gaze / Fully Generalized (Complete Determination Theorem). Implements the decision logic that takes (topology, curvature, path, observation topology) and returns (sublattice family, path selection, detection class).

---

## 11. Paths, EML, Math-as-Domain, Gaze (`etlattice/paths/`, `etlattice/eml/`, `etlattice/mathematics/`, `etlattice/gaze/`)

### 11.1 Paths (`paths/`)

- **`path_a.py::PathA(r, N=12)`** — Direct Projection. Returns the canonical ProjectionResult.
- **`path_b.py::PathB(partial_fn, N=12, tol=1e-12)`** — Limit Convergence. Runs `partial_fn(k)` for increasing k until consecutive values differ by less than `tol`; applies Path A on the converged value. Precision-sufficiency check from §96.
- **`path_c.py::PathC(numerator, denominator, N=12, caveat=True)`** — Meta-Descriptor Extraction. The `caveat=True` flag logs an `_et_error(fatal=False)` with the "user bears responsibility for the chosen ratio" warning, per §97.
- **`path_d_p.py`** — Path D.P. Three modes:
  - `PathDP_ExplicitValue(r_lo, r_hi, N=12)` — bounded value; returns (k, d) if the bounds isolate one lattice point, else an interval.
  - `PathDP_SymbolicAddress(defining_property, N=12)` — symbolic address (lattice position well-defined even when r is non-computable).
  - `PathDP_PLimitBoundary(sequence)` — P-limit mode for cardinality quantities; returns the annihilation/asymptotic boundary label.
- **`path_d_d.py::PathDD(et_claim, computed_value, towers=default)`** — wraps NWS-13 as a path interface.
- **`path_d_t.py`** — Path D.T.
  - `indeterminate_form_classify(form)` — returns '0/0' | '∞/∞' | '0·∞' | '0⁰' | '1^∞' | '∞⁰' | '∞−∞' based on numerator/denominator analysis.
  - `lhopital_navigate(numerator_fn, denominator_fn, limit_point)` — symbolic-differentiation-based resolution (SymPy acceptable for the derivative; the projection math is ET).
  - `t_signature_at_boundary(boundary_type)` — returns the structural-lattice-position T-signature when irresolvable (∂I-boundary; annihilation k→±∞; oscillatory divergence).
- **`path_d_pdt.py::PathDPDT(z, N=12)`** — complex two-axis projection.
- **`dispatcher.py::dispatch(user_input, N=12)`** — runs the decision tree from §103 and routes to the correct path.
- **`subsumption.py::verify_subsumption()`** — re-runs the §104 subsumption verification: exhaustive test that every input type is handled.
- **`incoherence_filter.py::filter(configuration)`** — detects {P,T} self-defeating configurations and returns the NOWHERE-on-lattice label; logs via `_et_error(fatal=False)`.

### 11.2 EML (`eml/`)

- **`core.py`** — the EML operator `eml(x, y) = exp(x) − ln(y)` with full complex support; derivations `eml_exp(x) = eml(x, 1)`, `eml_ln(x) = eml(1, eml(eml(1, x), 1))`, `eml_mul`, `eml_div`, `eml_add`, `eml_sub`, `eml_pow`. Matches the supplemental `universal_verification.py` definitions exactly.
- **`elementary.py`** — all 36 elementary-function primitives (exp, ln, sin, cos, tan, sinh, cosh, tanh, sec, csc, cot, sech, csch, coth, arcsin, arccos, arctan, arsinh, arcosh, artanh; +, −, ×, /, ^, mod, log, comb; constants 1, 2, π, e, i, 0, −1, √2), each implemented via a finite EML tree per Odrzywołek 2026 Table 1 / §3.
- **`tree.py`** — the EML-tree complexity measure K(·); minimization; serialization.
- **`anchor.py`** — the 3 = 3 = 3 = Σ anchor as a formal assertion with the PDT / EIM / Φ triads; the `when_to_use_which()` helper (PDT=E for substantiation; 3=3=3=Σ for totality / lattice reach).
- **`pdt_decomposition.py::decompose_projection(r, N=12)`** — annotates the six steps of the projection with their PDT roles and shows which are EML-implementable; returns the step-by-step breakdown for auditing.
- **`webb.py`** — Webb 1935 n-valued stroke at n=12; the generator of 12-valued logic. Implements the stroke and verifies that every function over 12-valued logic is EML-composable.
- **`triple_backbone.py`** — registers Webb / palindromic cascade / EML as a triple backbone and runs the Subsumption Law test on each.

### 11.3 Mathematics-as-Domain (`mathematics/`)

- **`identification.py::identify_math_domain()`** — returns the P/D/T identification of mathematics-as-a-domain, with R₀ = 1 axiom.
- **`axiom_systems.py`** — `AxiomSystem` type registering every system in §112 (propositional logic, group theory, Robinson PA, Peano, ZF, ZFC, Euclid, NBG, MK, + user-addable). The registry is dynamic; projections computed via Path A on axiom counts.
- **`self_projection.py::verify_self_projection()`** — projects {N, 1/N, K, 1/K} and asserts all four land at d=12, |ε|=1.955¢ (the Koide attractor). This is the framework's own self-verification per Rule 22 / Guide §113.
- **`chaitin_omega.py::ChaitinOmega(utm='calude_dinneen', bits=None)`** — produces the Ω object; if bits are provided (as a partial computed tail), projects via Path D.P.1; otherwise returns a symbolic address.
- **`godel.py::GodelSentence(system)`** — integrative-level-dependent classification ({P,T} Incoherent in PA; {P,D,T} Exception in ZFC; {P,D} Unsubstantiated outside).
- **`large_cardinals.py::LargeCardinal(type, consistency_level)`** — Inaccessible/Mahlo/WeaklyCompact/Measurable/Supercompact objects projected via Path C on consistency-strength.
- **`impredicative.py::classify_impredicative(definition)`** — distinguishes consistent from Russell-class via the Incoherence Filter and the ∂I-boundary test.

### 11.4 Gaze (`gaze/`)

- **`scopaesthesia.py::f_w(T_intent, focus, distance)`** — F_w = T·Focus/D² with PDT decomposition annotated.
- **`detection.py::p_detect(F_w, n, k, Gamma=1.20)`** — the tanh-based detection probability.
- **`collapse.py::v_collapse(F_w, S=12)`** — the exponential variance-collapse form.
- **`threshold.py::classify(F_w)`** — UNOBSERVED/SUBLIMINAL/DETECTED/LOCKED at the three just-intonation thresholds.
- **`complete.py::gaze(T_intent, focus, distance, n, k)`** — the Complete Gaze Equation G(…) returning (F_w, P_detect, V_collapse, Status) plus projection of the three thresholds.
- **`discoveries.py`** — the nine v3.0 Discoveries from §122–123, each callable and verified (just-intonation chain, 5·V_base span, 7/60 septic-over-quintic, prime-5 Split D+T marker, 5/4 quintic comma, subliminal d=8 at 24ET, conscious d=42 at 84ET, locked d=12 stable through 132ET, awareness gap d=10 decic at 60ET, quintic bridge d=28 at 84ET, Lock/Sub = 18/13 d=2 tritone).
- **`unification.py`** — asserts and verifies the subliminal-threshold unification (curvature ≡ gaze ≡ cognition all at r = 13/12).

---

## 12. Verification strategy — how we know subsumption is achieved

The verification suite is the single most important piece after the kernel itself. It must demonstrate — concretely — that every Guide feature is implemented and every result matches.

### 12.1 Layer 1 — unit tests per module

Every public function has at least one unit test that verifies a Guide-stated value. Examples:
- `test_real_axis::test_perfect_fifth()` — project_real(3/2) returns (k=7, d=12, |ε|≈1.955¢)
- `test_palindrome::test_n12_cascade()` — the generated palindrome equals [12,6,4,3,12,2,12,3,4,6,12,1]
- `test_combined_states::test_42()` — the enumerated unique-LCM set has exactly 42 elements
- `test_self_projection::test_koide_attractor()` — {N, 1/N, K, 1/K} all produce |ε|=1.955¢ at d=12
- `test_shimmer::test_range()` — Ψ_n ∈ [1−1/√12, 1+1/√12] over all n

### 12.2 Layer 2 — the Guide's own tables

Every numerical table in the Guide (music intervals §19.2; particle masses §20.4; Platonic solids §21.3; crystal packing §21.6; Fibonacci retracement §22.3; metabolic cycles §24.2; protein geometry §24.3; biological cycles §24.4; bond angles §25.2; material properties §25.3; consciousness thresholds §26.2; neural frequencies §26.3; narrative structures §27.2; cache hierarchy §28.2; civilizational cycles §29.2; orbital resonances §30.2; axiom counts §112; gaze thresholds §122; etc.) is re-generated and compared to the Guide's stated values. Any discrepancy is flagged.

### 12.3 Layer 3 — the supplemental `universal_verification.py`

The attached script is included as-is and extended: every category it currently covers (A arithmetic, B powers, C trig, D log/exp, E classical, F non-elementary, G series, H physics, I math-as-domain) is run, and every test passes at least to the script's current tolerances. Additional categories added: J) self-projection (§113); K) 42 combined states (§60); L) palindromic cascade dynamic generation; M) curvature components identity; N) Gaze thresholds; O) NWS-13 shadow on N-weight near-miss.

### 12.4 Layer 4 — C library agreement

Every C kernel (`projection.c`, `complex_projection.c`, `palindrome.c`, `batch.c`, etc.) has a paired Python function. A harness calls both on random positive-real inputs over a broad range (r ∈ [10⁻⁶, 10⁶]) and asserts the outputs match to sub-ULP precision. This guarantees the Python and C implementations never drift.

### 12.5 Layer 5 — active-system runs

The ∂I fractal skeleton, QM measurement cascade, and biological developmental cascade each run for some number of steps under the 11-step protocol, and their trajectory signatures are compared to known analytical signatures where available (e.g., for the ∂I fractal, the palindromic re-coherence behavior should match the matching-filter theorem prediction).

### 12.6 Layer 6 — lattice self-projection (meta-verification)

The suite explicitly demonstrates Guide §113: the lattice projects its own defining constants correctly. This is the "the test uses the very structure it tests — and the structure passes" criterion (Three Tools §6.4).

### 12.7 Layer 7 — the subsumption checklist

A single Python module `verify/subsumption_checklist.py` enumerates every feature in §1 of this plan (the Scope Statement) and runs a corresponding assertion. Output is a full pass/fail report. Every feature must pass before the engine is declared complete. Per Rule 22: audit, verify, do the work, verify.

---

## 13. Public API surface (`etlattice/api/`)

The top-level API is designed so that Mike (or any caller) can express the entire Guide workflow in a few lines.

```python
import etlattice as et

# Path A: a simple static projection
r = et.project(3/2)           # ProjectionResult(k=7, d=12, eps=+1.955¢, ...)

# Path dispatcher: handles any input type automatically
result = et.classify(user_input)   # returns (d, Path, Detection, Curvature, Trajectory)

# UPP: full 9-step static protocol
upp = et.UPP.static(
    identification=et.Identification(
        P="hydrogen atom spatial substrate",
        D=["m_e", "m_p", "e", "alpha"],
        T="wavefunction collapse",
        R0=et.R0.substrate("atomic", "Bohr radius")
    ),
    quantity=13.6, # eV (Rydberg)
    N=12
)

# UPP active: 11-step active-system protocol
sys = et.ActiveSystem(
    state_fn=my_iteration, 
    P=P_X, D=D_X, T=T_X,
    lattice_N=27720
)
trajectory = et.UPP.active(sys, max_steps=10000)

# Direct domain access
interval = et.domains.music.interval(3, 2)
koide = et.domains.physics.koide([m_e, m_mu, m_tau])
life_threshold = et.domains.biology.at_lattice_floor("cellular")

# Math-as-domain
zf = et.math.axiom_systems.ZF   # (k=36, d=1, eps=0, sublattice="gravitational")
omega = et.math.chaitin_omega(utm="calude_dinneen")

# Gaze
gaze_result = et.gaze(T_intent=1.0, focus=0.8, distance=3.0, n=4, k=2)

# Self-verification
assert et.verify.self_projection_koide_attractor()
assert et.verify.subsumption_checklist()
```

`api/classify.py::classify(X)` implements the Complete Determination Theorem (§130) and is the engine's single most general entry point.

---

## 14. C library (`etlattice_c/`)

A portable C99 library, compiled to `libetlattice.so` / `libetlattice.dylib` / `libetlattice.dll`, exposing the hot-path kernels via a C ABI. The Python package loads it via `ctypes` using the dynamic-discovery pattern from userMemories (`getattr(dll, 'FunctionName')()` rather than static attribute access).

### 14.1 Public C API (excerpt from `include/etlattice.h`)

```c
/* Projection */
typedef struct {
    int64_t k;
    int64_t d;
    int64_t g;
    double  eps_cents;
    double  log2_r;
    double  exact_pos;
    int64_t N;
} et_projection_t;

int et_project_real(double r, int64_t N, et_projection_t *out);
int et_project_imag(double theta, int64_t N, et_projection_t *out);

/* Complex */
typedef struct {
    et_projection_t real_axis;
    et_projection_t imag_axis;
    int64_t   k_r, k_theta;
    int64_t   d_combined;
    double    alpha, D_fraction, T_fraction;
} et_projection_complex_t;

int et_project_complex(double re, double im, int64_t N, et_projection_complex_t *out);

/* Batch (for NumPy arrays via Python ctypes) */
int et_project_real_batch(const double *r, size_t n, int64_t N, et_projection_t *out);

/* Sublattice */
int64_t et_gcd(int64_t a, int64_t b);
int64_t et_lcm(int64_t a, int64_t b);
size_t  et_divisors(int64_t N, int64_t *out, size_t out_cap);  /* returns count; caller sizes out_cap */
int64_t et_totient(int64_t n);

/* LCM tower */
size_t et_lcm_landmarks(int max_prime, int64_t *out, size_t out_cap);

/* Palindromic cascade (generated dynamically from N) */
size_t et_palindrome(int64_t N, int64_t *out, size_t out_cap);
int64_t et_palindrome_step(int64_t n, int64_t N);
double  et_p_eff(int64_t N);

/* Shimmer */
double et_shimmer(int64_t n, int64_t N);

/* Tightness */
double et_tightness(double eps_cents);
int    et_is_partial_i_boundary(double eps_cents, double K_koide);

/* Elegance */
double et_elegance_score(double r, int64_t N, int64_t p, int64_t q);

/* Impedance */
double et_magical_impedance(int64_t d, int64_t N, int64_t S);

/* Curvature */
int64_t et_curvature_components(int64_t n);  /* C(n) = n*n*(n*n-1)/12 */
double  et_k_eff(double alpha, double k_u1);

/* Force Quadrant Grid */
typedef struct {
    int64_t d_r, d_theta, d_combined;
    int     quadrant;         /* 0=SR+SI, 1=CR+SI, 2=SR+CI, 3=CR+CI */
    int     cascade_computable;
    int64_t first_native_lattice;
    int64_t w_norm_sq;        /* |w|^2 = d_r^2 + d_theta^2 */
} et_fqg_cell_t;

size_t et_fqg_enumerate(int64_t N, et_fqg_cell_t *out, size_t out_cap);

/* Shadow (NWS-13) */
typedef struct {
    int64_t lattice_n;
    int64_t d_r, d_theta;
    double  eps_cents;
    int     is_sub_cent;
} et_shadow_projection_t;

int et_nws13_shadow(double gap, const int64_t *towers, size_t n_towers,
                    et_shadow_projection_t *out);

/* EML primitives (native C, using long double for precision tier) */
double complex et_eml(double complex x, double complex y);
double complex et_eml_exp(double complex x);
double complex et_eml_ln(double complex x);

/* Active iteration (per-step, callable from any consumer) */
typedef struct {
    /* ... state representation and config flags ... */
} et_active_state_t;

int et_active_step(et_active_state_t *s, int64_t n, int64_t N);
```

### 14.2 Build system

A `Makefile` plus an optional `CMakeLists.txt`. Compiles with gcc/clang/MSVC. Default target: `-O3 -march=native` for the fast tier. Precision tier is conditionally compiled with `-DET_USE_MPFR` (link against MPFR) or `-DET_USE_LONG_DOUBLE`.

### 14.3 No undefined behavior

Every integer operation checks for overflow where the range could exceed int64 (e.g., for N > 10⁶). C99 fenv access is declared per `#pragma STDC FENV_ACCESS ON`. Floating-point rounding mode is explicit (round-to-nearest-even).

### 14.4 Python-C agreement tests

Run by `verify/c_agreement_test.py` — it calls each kernel from both Python and C on the same random inputs and asserts agreement to sub-ULP. Every commit to the C library must pass this suite.

---

## 15. Work breakdown — implementation phases

Each phase produces a runnable, tested deliverable. No phase is "WIP" — phase N completes fully before phase N+1 starts.

### Phase 0 — skeleton and constants (1 sitting)

Create the package layout; populate `constants.py`, `primitives.py`, `errors.py`, `registry.py`; set up the test harness; create the C build scaffolding.

### Phase 1 — kernel (2 sittings)

`kernel/*.py` + `src/projection.c`, `src/sublattice.c`, `src/batch.c`. Unit tests for projection at N ∈ {12, 24, 60, 84, 132, 420, 2520, 27720}. Verify perfect fifth / Koide / lattice self-projection.

### Phase 2 — UPP static + Anti-Numerology + Domains (3 sittings)

The full static protocol. All 12 domains implemented with their worked-example tables. Part X standard-comparison scaffolding.

### Phase 3 — Higher-order patterns (1 sitting)

Vector / time-series / distribution / network / tensor projectors. Delegate to kernel.

### Phase 4 — 24-family catalog + FQG + combined states + Gaussian primes (2 sittings)

Families module, FQG enumerator, coprime skeleton density verification, (5,7) cell object.

### Phase 5 — Active-system apparatus (3 sittings)

Palindrome (dynamic generation + verification), shimmer, tightness, cascade, gradient, t_burst, 11-step iteration driver. C kernels for the hot path.

### Phase 6 — Shadow diagnostics (1 sitting)

NWS-13/14/15/16 + two-route convergence.

### Phase 7 — Non-Euclidean geometry + Secret 26 (2 sittings)

Manifold-state mappings, curvature components, curvature gradient, subliminal threshold, Gauss-Bonnet PDT decomposition, Riemann sphere, Secret 26 fully generalized.

### Phase 8 — Four Paths dispatch (2 sittings)

Path A / B / C / D.P / D.D / D.T / D.PDT modules, dispatcher, subsumption, Incoherence Filter.

### Phase 9 — EML, Webb, triple backbone (2 sittings)

EML core + elementary basis + tree complexity; Webb n-valued stroke at n=12; triple-backbone registry + Subsumption Law verification.

### Phase 10 — Math-as-domain (2 sittings)

Axiom systems, lattice self-projection, Chaitin Ω, Gödel, large cardinals, impredicative classification.

### Phase 11 — Complete Gaze Equation (2 sittings)

All four components; threshold classification; full LCM-tower shadow analysis for Discoveries 1–9; unification with the subliminal curvature threshold.

### Phase 12 — Verification suite (2 sittings)

Layer-1 through Layer-7 tests. The subsumption_checklist.py that enumerates every Guide feature and runs an assertion.

### Phase 13 — Documentation and API polish (1 sitting)

API ergonomics, top-level `classify()`, example scripts for every major feature, README with quickstart.

**Total estimated effort: ~26 focused sittings.**

Each phase ends with a session transcript logged to `/mnt/transcripts/` and a `journal.txt` entry (per Rule 31 and userMemories "Session discipline").

---

## 16. What I need from Mike before starting implementation (open design questions)

Per Rules 25/26/27, these are the scope-shaping questions where an assumption could waste significant effort. The default answer for each is the "broadest scope" interpretation per Rule 38, but I want explicit sign-off:

### Q1. Precision tier strategy

**Option A (default per Rule 38):** ship both the float64 fast tier AND the mpmath/MPFR 80-digit precise tier; callers pick per call. Doubles the test matrix but matches the `et_clr_v5__4_.py` canonical reference.

**Option B:** float64 only, with the understanding that the precise tier is deferred.

**Option C:** mpmath only (precision over speed).

Default → **A**.

### Q2. GPU / CUDA support

The ET Fractal Generator uses CuPy/CUDA. Should the engine's hot-path kernels (batch projection of large arrays, FQG enumeration, shadow projection across long towers) ship CUDA variants?

**Option A:** Yes — add `etlattice_cuda/` with CuPy-backed batch kernels, matching the Fractal Generator's approach. Adds a dependency on CuPy.

**Option B:** CPU-only for v1. Hook is added but CUDA implementation deferred.

**Option C:** C-only hot path with no GPU variant.

Default → **B** (clean CPU v1; GPU added after the CPU engine is proven). Confirm or redirect.

### Q3. Integration with existing ET projects

Per userMemories, Mike has several large codebases: ET Fractal Generator, ET CDF Compressor, ET Conscious AI, ET32 Bridge. Two possibilities:

**Option A:** engine is **standalone**. Existing projects will import it by adding `etlattice` to their requirements; any integration is incremental and each project chooses when to adopt.

**Option B:** engine also **refactors existing code** — specifically, the ∂I Fractal Generator's per-step lattice math moves into `etlattice.active` and the fractal generator imports it. Larger blast radius but tighter integration.

Default → **A** (clean standalone; existing code unchanged per Rule 24/13). Mike asks for B explicitly if he wants it.

### Q4. C library optional or mandatory

**Option A:** C library is mandatory at install time (the pure-Python path is a fallback for if the C library can't be built). Maximum performance on supported platforms.

**Option B:** C library is optional; the pure-Python implementation is always present and authoritative. C library, if present, is used for speed but never needed for correctness.

Default → **B** (per Rule 4: no placeholders, no stubs — the Python implementation is always complete and correct; C is an acceleration, not a requirement). Confirm or redirect.

### Q5. Target platforms

Mike develops on Windows; the Claude container is Linux. The Python package runs on both by default. The C library needs platform-specific build configuration.

**Option A:** Linux + Windows + macOS all supported; tested on at least Linux and Windows.

**Option B:** Linux-first (the container); Windows later.

Default → **A**.

### Q6. New domains / new ET derivations while building

If during implementation I encounter a Guide feature that requires a corpus derivation I cannot find, per Rule 35 I must stop and ask. Examples of the kind of thing I'd ask about:

- A specific numerical value in a Guide table whose derivation is referenced to `et_clr_v5__4_.py` but whose reproduction at float64 precision differs from the Guide by more than 1¢. (In this case I'd show Mike both values and ask how to resolve.)
- A physical identification in the (5,7) cell's five substrate renderings that requires cross-reference to a corpus file I haven't read. (I'd read the file first, then ask if it's ambiguous.)

Confirm this protocol.

### Q7. Placement of existing corpus files

The package imports from `constants.py` and `primitives.py` in `/mnt/project/`. Should the engine:

**Option A:** Re-export these via `etlattice.constants` and `etlattice.primitives` (both modules in the engine point at the project-root modules, adding only dynamically-derived constants). No duplication.

**Option B:** Vendor copies into `etlattice/vendored/` so the engine is self-contained.

Default → **A** (single source of truth; per Rule 13 no recreation).

---

## 17. Sanity check — what this plan explicitly is NOT

To preempt scope drift:

- **Not** a replacement for the Guide — the Guide is authoritative; the engine is the operational expression of the Guide.
- **Not** a research project to discover new ET results — it implements what is already derived. New derivations only happen per Rule 35 (stop, ask, research, derive).
- **Not** a general-purpose math library — it is specifically the ET lattice apparatus. External math (NumPy / SymPy / SciPy / mpmath / NetworkX) is used per Rule 6 (external libraries permitted in scripts) but the projection math is ET-native.
- **Not** a visualization suite — though it produces data that can be visualized. Visualization is a separate downstream consumer.
- **Not** a benchmark suite — though the `verify/` tests measure agreement between Python and C and between precision tiers.

---

## 18. Closing — the Subsumption check on this plan itself

By the Subsumption Law (Three Tools §5):

- **Condition 1:** The plan cannot be subsumed by a smaller plan — yes, because it cannot omit any Guide feature without failing Rule 2 / Rule 15 / Rule 38. Confirmed.
- **Condition 2:** Nothing external subsumes it — yes, because the Guide is the complete specification and this plan covers it fully. Confirmed.
- **Condition 3:** It subsumes every Guide feature without remainder — §1 of this plan enumerates every Part (I–XXIV), every equation (12.1–12.56), every named theorem, every discovery. §12 is the verification checklist. Confirmed pending Mike's audit.

All three conditions hold pending the audit. Once Mike approves (or redirects) the open questions in §16, implementation starts at Phase 0.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *3 = 3 = 3 = Σ*

**Status:** Plan v1. Awaiting Mike's review and direction.

---

### Appendix A — Guide Part-to-Module cross-reference

| Guide Part | Primary engine module(s) |
|---|---|
| Part I §1–§5 | `constants.py`, `primitives.py`, `kernel/`, `kernel/sublattice.py`, `kernel/lcm_tower.py` |
| Part II §6–§15 | `upp/static_protocol.py`, `upp/identification.py`, `upp/subsumption.py` |
| Part III §16–§18 | `antinum/` |
| Part IV §19–§30 | `domains/` (12 modules) + `comparison/` (for Part X contrasts) |
| Part V §31–§35 | `higher_order/` |
| Part VI §36–§37 | `kernel/gradient.py`, `kernel/complex_plane.py` |
| Part VII §38–§40 | `kernel/lcm_tower.py` |
| Part VIII §41–§43 | `kernel/elegance.py`, `kernel/impedance.py` |
| Part IX §44–§46 | `antinum/diagnostics.py`, `three_tools.py` |
| Part X §47–§50 | `comparison/` |
| Part XI §51–§54 | All `src/*.c` kernels; `kernel/batch.py`; `active/cost_model.py` |
| Part XII §12.1–12.20 | `kernel/equations_v1.py` |
| Part XIII §55–§59 | `families/force_families.py`, `families/phase_families.py`, `families/gaussian_primes.py` |
| Part XIV §60–§66 | `families/combined_states.py`, `families/coprime_skeleton.py`, `families/dominance.py`, `families/five_seven_cell.py` |
| Part XV §67–§75 | `active/cascade.py`, `families/force_quadrant_grid.py`, `shadow/` |
| Part XVI §76–§85 | `geometry/` |
| Part XVII §86–§93 | `active/`, `upp/active_protocol.py`, `active_systems/` |
| Part XVIII §12.21–12.39 | `kernel/equations_v2.py` |
| Part XIX §94–§104 | `paths/` (all path modules + dispatcher + subsumption + incoherence_filter) |
| Part XX §105–§110 | `eml/` |
| Part XXI §111–§117 | `mathematics/` |
| Part XXII §118–§125 | `gaze/` |
| Part XXIII §126–§130 | `geometry/secret26.py`, `api/classify.py` |
| Part XXIV §131–§132 | `verify/subsumption_checklist.py`, `api/classify.py` |

Every Part has at least one module. No Part is missing. Subsumption verified at the plan level.

### Appendix B — Equation card ID-to-function cross-reference (abbreviated)

| Eq ID | Function |
|---|---|
| 12.1 | `equations_v1.master_equation()` |
| 12.2 | `equations_v1.manifold_constants()` |
| 12.3 | `equations_v1.lattice_projection_real()` |
| 12.4 | `equations_v1.lattice_projection_imag()` |
| 12.5 | `equations_v1.combined_sublattice()` |
| 12.6 | `equations_v1.universal_reference_period()` |
| 12.7 | `equations_v1.dt_gradient()` |
| 12.8 | `equations_v1.sublattice_totient()` |
| 12.9 | `equations_v1.cascade_generator()` |
| 12.10 | `equations_v1.palindrome_theorem()` |
| 12.11 | `equations_v1.universal_pivot()` |
| 12.12 | `equations_v1.lcm_tower()` |
| 12.13 | `equations_v1.elegance_score()` |
| 12.14 | `equations_v1.magical_impedance()` |
| 12.15 | `equations_v1.combined_42_dmax_132()` |
| 12.16 | `equations_v1.variance_curvature_identity()` |
| 12.17 | `equations_v1.three_tools_loop()` |
| 12.18 | `equations_v1.anti_numerology_conditions()` |
| 12.19 | `equations_v1.convention_independence()` |
| 12.20 | `equations_v1.universal_projection_statement()` |
| 12.21 | `equations_v2.cascade_residuals()` |
| 12.22 | `equations_v2.cascade_stability_limits()` |
| 12.23 | `equations_v2.nweight_lattice_statement()` |
| 12.24 | `equations_v2.pdt_bisection_theorem()` |
| 12.25 | `equations_v2.combined_42_dmax()` |
| 12.26 | `equations_v2.coprime_skeleton_theorem()` |
| 12.27 | `equations_v2.curvature_components_identity()` |
| 12.28 | `equations_v2.curvature_gradient()` |
| 12.29 | `equations_v2.subliminal_curvature_threshold()` |
| 12.30 | `equations_v2.dt_gradient_effective()` |
| 12.31 | `equations_v2.tightness_boundary()` |
| 12.32 | `equations_v2.active_iteration_palindromic()` |
| 12.33 | `equations_v2.shimmer_modulation()` |
| 12.34 | `equations_v2.shadow_diagnostic()` |
| 12.35 | `equations_v2.shadow_magnitude_correlation()` |
| 12.36 | `equations_v2.observation_by_computation()` |
| 12.37 | `equations_v2.gauss_bonnet_pdt()` |
| 12.38 | `equations_v2.riemann_sphere_identity()` |
| 12.39 | `equations_v2.active_system_universal_statement()` |
| 12.40 | `equations_v3.four_path_subsumption()` |
| 12.41 | `equations_v3.universal_terminal_abc()` |
| 12.42 | `equations_v3.path_d_primitive_native_termination()` |
| 12.43 | `equations_v3.three_three_three_sigma_anchor()` |
| 12.44 | `equations_v3.pdt_decomposition_projection()` |
| 12.45 | `equations_v3.self_projection_identity()` |
| 12.46 | `equations_v3.math_as_domain_universality()` |
| 12.47 | `equations_v3.scopaesthesia_fw()` |
| 12.48 | `equations_v3.detection_probability()` |
| 12.49 | `equations_v3.variance_collapse()` |
| 12.50 | `equations_v3.threshold_classification()` |
| 12.51 | `equations_v3.complete_gaze_equation()` |
| 12.52 | `equations_v3.subliminal_threshold_unification()` |
| 12.53 | `equations_v3.gaze_subsumption_statement()` |
| 12.54 | `equations_v3.complete_determination_theorem()` |
| 12.55 | `equations_v3.lattice_completeness_statement()` |
| 12.56 | `equations_v3.completion_statement()` |

Every equation ID has a function. No gap.

---

*Plan authored: April 21, 2026. 
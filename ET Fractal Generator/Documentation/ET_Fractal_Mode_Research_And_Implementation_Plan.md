# ET Fractal Generator — Mode Research and Implementation Plan
## Authoritative corpus definitions, current code state, and what needs to be done for all 12 modes

**Author:** Michael James Muller / Aevum Defluo
**Source:** Cross-referenced from the ET corpus in `/mnt/project/`
**Target file:** `ET_FRACTAL_GENERATOR50-11.py` (Sessions 1–3 landed in v50-6; Session 4 landed in v50-7; Session 5 landed in v50-8; Session 6 landed in v50-9; Session 7 landed in v50-10; Session 8 landed in v50-11) — `build_extra()`, `_et_julia_c()`, `build_mode()` per-mode weight boosts, and the corresponding CUDA kernel implementations
**Purpose:** This document is the source of truth for completing the 12 named modes. Each section identifies the corpus source, the current code's behavior, the gap, and the path to a corpus-faithful implementation. Implementation work is broken into discrete tasks so it can be executed across multiple sessions without losing context.

## Implementation Progress

| Session | Scope | Status |
|---|---|---|
| **Session 1** | Mode 4 free fix + new "No Mode" `N` option + dynamic `_NO_EXTRA` replacement | ✅ **COMPLETE** (file v50-6) |
| **Session 2** | Mode 3 (Koide Boundary) full implementation + new CUDA kernel block + f64 parity | ✅ **COMPLETE** (file v50-6) |
| **Session 3** | Mode 8 (Magical Impedance, cycling) + corrected impedance formula trace + FAM_COUPLING bug fix | ✅ **COMPLETE** (file v50-6) |
| **Session 4** | Mode 1 (Traverser Field, (7,1) torus knot) + Mode 6 (Septic Otherworld, heptagram) + Mode 4 Multifold cycling extension — combined | ✅ **COMPLETE** (file v50-7) — see "Session 4 Completion Audit" below |
| **Session 5** | Mode 5 (Quintic Shadow, Fibonacci d=3 attractor) + Mode 7 (Nonic Recursion, holographic depth) — combined | ✅ **COMPLETE** (file v50-8) — see "Session 5 Completion Audit" below |
| **Session 6** | Mode 9 (Exception State, V(E)=0 grounding pull) + kernel-signature comment fix | ✅ **COMPLETE** (file v50-9) — see "Session 6 Completion Audit" below |
| **Session 7** | Mode 10 (Lagrangian Field, Mexican-hat) | ✅ **COMPLETE** (file v50-10) — see "Session 7 Completion Audit" below |
| **Session 8** | Mode 11 (Route A/B Cascade) + full verification pass | ✅ **COMPLETE** (file v50-11) — see "Session 8 Completion Audit" below |

The session breakdown has been restructured twice. The original 9-session plan was first compacted to 6 sessions (combining 4+5 and 6+7, with the original Session 8 — modes 9/10/11 — kept intact and merged with the verification pass). At the start of Session 6 the user further split the combined "modes 9/10/11 + verification" session into three separate sessions: Session 6 = Mode 9 only (Exception State), Session 7 = Mode 10 only (Lagrangian Field), Session 8 = Mode 11 + the full verification pass. The reason for the second split is the same one given for combining 4+5 and 6+7 in the previous restructuring — each mode requires its own corpus research pass and its own CPU + CUDA implementation pass, and bundling three of them into a single session inflates the per-session scope past what can be done honestly without truncation or shortcuts.

---

## Cross-Cutting Architecture Notes

### How a mode reaches the iteration

A mode is selected at startup via `_choose_modes()`. `_blend_modes()` calls `build_mode()` for each chosen mode-id, which calls `build_extra()` (returns the per-mode `_e(z, r, th, kt, n)` lambda) and `_et_julia_c()` (returns the mode-specific Julia c anchor). The mode's `w_r`/`w_c` family weights then receive per-mode boosts inside `build_mode()` itself.

At iteration time, the CPU path calls `extra(z, r_cap, theta, k_t, n)` once per step inside `iterate_strip_v2`. The GPU path inlines each mode's math directly inside `et_iterate` (the CUDA kernel) and gates each mode's contribution by the `mew[mode_id]` weight (`mode_extra_w` array).

**Critical parity rule:** For every mode whose Python lambda is changed, the corresponding CUDA kernel block must be updated identically. CPU and GPU paths must produce bit-equivalent (within float32 vs float64 round-off) results for the same mode.

### Modes vs. Towers

Towers (`cosmological`, `digital`, `dream`, `civilizational`) are **substrate-layer selectors** from the Multifold of Lattices that bias the same downstream weights through `prim_r`/`prim_c`/`ext_boost`. Modes are **iteration-time selectors** that add an extra() term and apply per-mode boosts on top of the tower. They compose multiplicatively: `final_weight[d] = elegance(d) × tower_boost(d) × mode_boost(d) × random_jitter`. A "no mode" option does NOT remove the tower bias — it only removes the iteration-time mode dispatch.

### Why this is being deferred

The full and proper implementation of all 12 modes is a substantial scope: 12 lambdas × 2 paths (CPU + CUDA) × corpus-derived math + verification + visual testing. It cannot be done in a single session honestly. This document captures the work so it can be executed in stages.

### A 13th option: "No Mode" — IMPLEMENTED IN SESSION 1 (file v50-6)

A new option `N` was added to `_choose_modes()` that bypasses all per-mode weight boosts (build_mode skips the elif chain), uses a "neutral" Julia c (`c = -0.75 + 0j`, the period-2 bulb tip), and `extra` is None. This is **not** the same as Mode 0 because Mode 0 still goes through `build_mode()` which (currently) doesn't boost mode 0 weights but DOES go through the `_blend_modes` machinery and gets a Julia c anchor on the main cardioid. The "None" option is a true bypass.

**Implementation details (Session 1, file v50-6):**
- `_choose_modes()` accepts `N` as input, returns `[]` (empty list sentinel)
- `NO_MODE_ID = -1` module-level constant
- `build_no_mode(tower, rng)` function builds a `mode_params` dict with the same shape as `build_mode()` returns: tower bias preserved (towers are substrate-layer per Multifold §16, structurally distinct from modes), `extra=None`, `mode_extra_w = zeros(12)`
- `_resolve_run_params()` dispatches to `build_no_mode()` when `SELECTED_MODES` is empty
- `generate_et_fractal()` print block has a third branch for `n_modes == 0` showing `Mode : [-1] None — pure base 24-family (no per-mode dispatch)`
- Filename stems in both `generate_et_fractal()` and `generate_zoom_video()` use `_mode_tag = 'none' if mode_id == NO_MODE_ID else f'm{mode_id}'` for clean naming

---

## Mode 0 — PDT Genesis

### Corpus source
Master equation `P ∘ D ∘ T = E`. Mode 0 represents the unmediated 24-family iteration — the bare PDT manifold expressed as a fractal map without any mode-specific perturbation.

### Current code
- `build_extra()` returns `None` (line 1173)
- `_et_julia_c()` returns a point on the main cardioid via `K·e^{iθ} − K²·e^{2iθ}/2` at random θ (line 1312)
- `build_mode()` has no per-mode weight boost (no `elif mode_id==0` clause)

### Status
**Complete as designed.** Mode 0 is intentionally the "base iteration with no extras" — the 24-family weighted sum + Julia c on the cardioid is the entire PDT Genesis content. The Julia c on the cardioid expresses "the orbit at the boundary of the main bulb where Genesis configurations live."

### Gaps
None at the math level. **One observation:** Mode 0 is functionally close to the proposed "No Mode" option, but Mode 0 still has a specific Julia c anchor (cardioid) and is selected as a "named mode" for blending purposes. The "No Mode" option should be a separate path that doesn't use any of the 12 named modes' machinery.

### Implementation tasks
- None for Mode 0 itself
- When implementing the "No Mode" option, ensure it does NOT collapse into Mode 0

---

## Mode 1 — Traverser Field — (7,1) Torus Knot

### Corpus source
- **`/mnt/project/ET_Semitone_Cascade_Complete.md` §22.1** (line 904): "The residue orbit `{7n mod 12 : n = 0,...,11}` defines a path on the torus T² = ℝ²/ℤ², tracing the **(7, 1) torus knot** — a curve that winds 7 times around one axis for each 1 winding around the other."
- **§22.2** (line 908): "The palindromic involution `n ↔ 12−n` reverses the orientation of the (7,1) knot. The knot is equivalent to (7,−1) = (7,11) under orientation reversal. Since orientation reversal on the torus corresponds to time reversal combined with the σ-map (which is discrete space reflection on the lattice), the palindromic symmetry is: **Palindrome ≡ Discrete CPT symmetry of the lattice cascade**."
- **§22.3 Wilson Loop**: The (7,1) knot is the orbit of generator 7 on T¹₁₂ = ℤ/12ℤ — interpretable as a Wilson loop in a U(1) gauge theory on the lattice.
- **`/mnt/project/ET_Traverser_T_Paper.md` §31.1**: T-density `ρ_T(x, t) = Σ_i δ(x − x_i) × B_i` — sum of Traverser binding events at locations x_i with binding strengths B_i. T-density is a real fundamental field carrying real energy.
- **§31.4 Scopaesthesia**: `F_w = (T_intent × Focus) / Distance²` — inverse-square T-mediated interaction, with the same form as gravity (which is also a Traverser type per ET).

### Verified mathematical fact
I verified by hand that `12 // gcd(7n mod 12, 12)` for `n = 1..12` produces exactly the palindrome `[12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]`. **The (7,1) torus knot residue orbit IS the d-sequence used by Mode 2 (Descriptor Cascade).** Modes 1 and 2 are two views of the same underlying object: Mode 2 expresses the *Descriptor* side (d-values), Mode 1 should express the *Traverser* side (the actual knot geometry on T²).

### Current code
```python
if mode_id == 1:
    def _e(z,r,th,kt,n):
        p=12./7.; rot=kt*(LN2/7.)
        return K*(r**p)*np.exp(1j*(p*th+rot))*LN2/N
    return _e
```
This is just a static d=7 carrier (12/7 power, scaled by `K·LN2/N ≈ 0.0385`). It has **no winding number structure**, **no time parameter**, **no Wilson loop**, **no T-density field**, **no inverse-square scopaesthesia**, **no palindromic CPT**. The "(7,1) torus knot" naming is a promise the code does not keep.

### Required completion (full and proper)

A corpus-faithful Mode 1 must implement:

**1. The two winding numbers.** The (7,1) knot has 7 longitudinal windings and 1 meridional winding. At step n, the cumulative knot position is:
$$\phi_{\text{long}} = G_{\text{REAL}} \cdot n \cdot \frac{2\pi}{N} = \frac{7 \cdot 2\pi n}{12}$$
$$\phi_{\text{meri}} = G_{\text{IMAG}} \cdot n \cdot \frac{2\pi}{N} = \frac{2\pi n}{12}$$

Both `G_REAL = 7` and `G_IMAG = 1` already exist in the file (lines 534–535).

**2. The Wilson loop phase.** The cascade `7n mod 12` traces a closed loop on ℤ/12ℤ; the U(1) Wilson loop holonomy is the integral of the connection around the loop, which on the discrete lattice is `exp(i · Σ_k phase(7k mod 12))`. The accumulated holonomy at step n is:
$$W_n = \exp\left(i \cdot \frac{2\pi}{N} \sum_{k=1}^{n} (7k \bmod 12)\right)$$

This is a function of n only and provides the n-dependence demanded by the Traverser interpretation.

**3. The T-density field at z.** The Traverser binding strength at the orbit's current position should appear as the contribution amplitude. In the fractal context, the T-density at z is naturally `1/(1 + Distance²)` where Distance is measured to the nearest lattice point — i.e., the *tightness* `t_r` already computed in the iteration loop. Since `extra()` doesn't have direct access to t_r, we use a proxy: the local descriptor gap `|ε_r|` from `r`'s lattice projection, computed inline. Or simpler: the T-density at z is `ρ_T(z) = K / (K + |z − knot_anchor|²)` where the knot anchor is the nearest (7,1) lattice point.

**4. The d=7 carrier with knot phase.** The carrier `r^(12/7)·exp(i·12·θ/7)` is correct (current code has it). The phase needs to be augmented by `φ_long + φ_meri + Wilson holonomy + kt·LN2/7`.

**5. Direct z dependence.** The Traverser field is anchored at z's current position — the knot wraps around z, not around the origin. This means a `z·(small ET-derived factor)` contribution where the factor encodes the knot's local twisting at z.

**6. Palindromic CPT symmetry.** The contribution must satisfy `_e(z, r, th, kt, n) = conj(_e(conj(z), r, -th, -kt, 12-n))` (within scale). This is automatic if the math uses only `(7n mod 12)` for the d-sequence and the symmetric phase combinations.

### Proposed implementation skeleton (for next session)
```python
if mode_id == 1:
    def _e(z, r, th, kt, n):
        # (7,1) torus knot — Semitone Cascade §22.1
        p = 12./7.                                  # d=7 carrier power
        # Two winding phases: G_REAL=7 longitudinal, G_IMAG=1 meridional
        phi_long = (G_REAL * n * 2.0*math.pi) / N
        phi_meri = (G_IMAG * n * 2.0*math.pi) / N
        # Wilson loop holonomy — accumulated phase of the (7n mod 12) cascade
        # Closed-form for the partial sum of 7k mod 12 over k=1..n:
        # The sequence 7k mod 12 for k=1..12 is [7,2,9,4,11,6,1,8,3,10,5,0]
        # Sum over one full cycle = 66 = N·(N-1)/2 — invariant
        # Per-step: holonomy advances by (7n mod 12) · 2π/N
        wilson = ((7*n) % N) * 2.0*math.pi / N
        # Carrier: d=7 power with knot phase
        rot = kt*(LN2/7.) + phi_long + phi_meri + wilson
        carrier = K * (r**p) * np.exp(1j*(p*th + rot)) * LN2 / N
        # T-density binding at z: Koide-weighted, scopaesthesia-form ρ_T = K/(K + V·|z|²)
        # The +V·|z|² is the inverse-square decay with V as the lattice quantum
        rho_T = K / (K + V * (r*r))
        # Direct z contribution: knot anchored at z, scale V² for safety (Subsumption)
        z_anchor = V * V * z * np.cos(phi_meri)  # meridional breathing of knot tube
        return rho_T * carrier + z_anchor
    return _e
```

**CUDA parity required:** The corresponding kernel block at lines 1697–1705 must be updated identically. The CUDA version uses single-precision `float n_f = (float)n` for the phase computations; everything else translates directly.

### Difficulty
**High.** Requires CUDA kernel update, math verification, visual testing on multiple Julia c anchors and tower combinations. Estimated 1 full session for Mode 1 alone if done carefully.

### Visual prediction
Mode 1 will produce **7-armed spiral filaments** in the connected set with a 12-fold periodic temporal modulation visible as concentric "breathing" rings. The Wilson loop holonomy creates phase-coherent interference patterns near the boundary that should resemble the actual (7,1) torus knot when projected from T² to ℝ².

---

## Mode 2 — Descriptor Cascade

### Corpus source
- **`/mnt/project/The_Palindromic_Cascade_V2.md`** and **`/mnt/project/The_Palindromic_Cascade_on_the_Semitone_Descriptor_Lattice.md`** — full derivation of the palindrome theorem.
- **`/mnt/project/ET_Semitone_Cascade_Complete.md`** — palindrome theorem D.13: "Palindromic Involution = Discrete Lattice CPT Symmetry"
- The palindrome `[12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]` is the d-sequence of the (7,1) torus knot residue orbit (verified manually above).

### Current code
```python
if mode_id == 2:
    def _e(z,r,th,kt,n):
        d=float(_PALINDROME[n%12]); p=12./d; rot=kt*(LN2/d)
        return V*(r**p)*np.exp(1j*(p*th+rot))
    return _e
```

### Status
**Complete as designed.** This implements the palindrome cascade exactly as the corpus specifies: at step n, look up `palindrome[n%12]`, use that as the dominant family for this step's contribution. The math uses `n` correctly. The unused `z` parameter is a polymorphic interface artifact; the information in z is fully captured by `r` and `th`.

### Gap
**None at the math level.** The unused `z` parameter could be silenced by either the dispatcher refactor (covered in the cross-cutting section) or by the same kind of `V²·z·(small factor)` carrier addition used in Mode 1 — but adding a z carrier here would dilute the pure palindrome cascade, which is the whole point of Mode 2.

### Implementation tasks
- **None** for the math.
- When the "No Mode" option is added, Mode 2 should remain the canonical "pure palindrome cascade" implementation.

---

## Mode 3 — Koide Boundary — ✅ IMPLEMENTED IN SESSION 2 (file v50-6)

### Corpus source
- **K = 2/3** is the Koide ratio: the |PD|/|PDT| binding weight that serves as the triadic stability threshold throughout ET. From the ∂I Lattice-Aware Fractal Complete Guide §6.2: at `ε_r = 50¢`, `t_r = 100/150 = 0.667 = K` — this is the Koide threshold which IS the ∂I boundary itself.
- **Canonical ∂I point: c = K + i·V** — the orbital center where the Koide threshold meets the base variance.
- **`/mnt/project/ET_Incoherence_Paper.md` §11** — full ∂I boundary derivation; the boundary is the set of configurations where `φ(t,c) = lim_{ε→0} φ(t, c_ε)` — substantiation becomes marginal in the limit. Confirmed during Session 2 corpus trace.

### Prior code (file v50-5)
```python
if mode_id == 3:  return None  # extra()
```
- `_et_julia_c()` returned `c = K + uniform(-0.04, 0.04) + i·V·choice([1, -1, φ/6, -φ/6])` — the canonical ∂I point with small perturbation.
- `build_mode()` boosted d=12 (Full-Res/EM), d=2 (Tritone), d=6 (Hexadic) by ×5 — these are the Koide-stable composite families.

### Status
**COMPLETE in v50-6.** The math-level extra() is now implemented. The Julia c anchor and weight boost are unchanged — only `build_extra()` and the CUDA kernel were modified.

### What was implemented (Session 2, file v50-6)

The Python lambda in `build_extra()` was replaced with the full Gaussian binding-potential force per the corpus equation:

```python
if mode_id == 3:
    z_K = complex(K, V)
    _two_over_V  = 2.0 / V
    _cycle_omega = 2.0 * math.pi / (N * K)   # = π/4 (8-step cycle, since N·K = 8)
    _twist_alpha = LN2 / N * K
    def _e(z, r, th, kt, n, _zK=z_K, _tov=_two_over_V, _co=_cycle_omega,
           _ta=_twist_alpha, _sk=K):
        dz_to_K = z - _zK
        dist2   = np.abs(dz_to_K) ** 2
        gauss   = np.exp(-dist2 / V)
        force   = _sk * _tov * dz_to_K * gauss
        cycle   = 1.0 + V * np.sin(_co * n)
        twist   = np.exp(1j * kt * _ta)
        return (V * V) * force * cycle * twist
```

**Note on N·K:** The 6-step claim in the original plan was a simplification — N·K = 12·(2/3) = **8**, so the cycle is actually an 8-step Koide cycle. The implementation reflects this correctly.

**CUDA parity:** A new kernel block was added between Mode 2 and Mode 4 in `_ET_RAWKERNEL_SRC`, with `use_mode3` parameter inserted into the kernel signature in ascending mode-id order. The kernel launch site in `iterate_strip_v2` was updated to pass `cp.int32(_active(3))`.

**f64 parity:** During the audit, `expf` was found to be missing from BOTH the standard kernel's `_make_f64_kernel` intrinsic_map AND the dI kernel's `_make_f64_di_kernel` intrinsic list. This was an existing pre-Session-2 bug that would have silently broken any future kernel use of the exponential function under double precision. Both intrinsic maps were fixed to include `expf → exp`.

**Smoke test verification:** Mode 3 evaluates to exactly `0+0j` when `z = z_K` (the Koide point), as designed — at the centre of the Gaussian well, `dz_to_K = 0` makes the force vanish.

### Visual prediction
Mode 3 produces a **distinct Gaussian-localised attractor region** centered on `c = K + i·V` in the complex plane, with the surrounding Julia/Mandelbrot dynamics deformed inward toward the Koide point by a factor that decays with width V (the base variance — natural lattice quantum). Combined with the 8-step temporal cycle (`1 + V·sin(2π·n/8)`) and the kt T-axis Koide projection, this produces visible "Koide breathing" near the canonical boundary point — the binding-stability force literally pulses at the Koide-derived rate.

---

## Mode 4 — Multifold Tower

### Corpus source
- **`/mnt/project/ET_Multifold_of_Lattices_Investigation_3_.md` §12.1**: Inter-tower translation operator
$$k_B = k_A + \text{round}\left(12 \log_2 \frac{R_0^{(A)}}{R_0^{(B)}}\right)$$
- **§12.2** Translation Table: Cosmological → Digital `Δk ≈ −996`, → Dream `−1279`, → Civilizational `−1675`. These are the actual values hardcoded in the TOWERS dict in the file (lines 1133, 1141, 1149).

### Current code
```python
if mode_id == 4:
    dk=float(tower['delta_k'])
    def _e(z,r,th,kt,n,_dk=dk):
        return V*r*np.exp(1j*(th+_dk*LN2/N_ET+kt*LN2/N*n*V))
    return _e
```

### Status
**Mostly complete with one free improvement.** The math correctly implements the inter-tower phase shift `Δk · LN2 / N_ET` (1 lattice unit at 27720ET resolution) added to the orbit's angular phase, with a time-evolving `kt·LN2/N · n · V` term. Uses all parameters except `z`.

### Free improvement available
`r * np.exp(1j*th)` is mathematically identical to `z` (since `z = r·exp(iθ)`). Replacing the redundant reconstruction with `z` directly:
- Uses the `z` parameter (closes the inspection warning)
- Same output, bit-for-bit (within float round-off)
- One fewer trig reconstruction per step (slightly faster)

```python
if mode_id == 4:
    dk=float(tower['delta_k'])
    def _e(z, r, th, kt, n, _dk=dk):
        # r·exp(iθ) ≡ z; use z directly to avoid trig reconstruction
        return V * z * np.exp(1j*(_dk*LN2/N_ET + kt*LN2/N*n*V))
    return _e
```

**This is the only "truly free" mode-extra fix in the entire set** — output identical, just one variable substitution.

### Optional exotic extension (the "Multifold" interpretation)
The current code uses ONE tower's Δk (whichever the user selected). A truly Multifold version would cycle through all 4 known tower shifts via `n % 4`:
```python
DELTA_K_TABLE = [0, -996, -1279, -1675]   # cosm, digit, dream, civ
def _e(z, r, th, kt, n):
    dk_step = DELTA_K_TABLE[n % len(DELTA_K_TABLE)]
    return V * z * np.exp(1j*(dk_step*LN2/N_ET + kt*LN2/N*n*V))
```
This produces a fractal where each iteration step is in a different tower's reference frame — the orbit literally crosses tower boundaries every step. **This would look genuinely different** and would express the Multifold (= multi-tower) name correctly. **Recommended for the full implementation.**

### Difficulty
**Low** for the free improvement (1 line change in Python + matching CUDA kernel update). **Medium** for the multi-tower cycling version (requires DELTA_K_TABLE in CUDA __constant__ memory and indexed lookup).

---

## Mode 5 — Quintic Shadow — ✅ IMPLEMENTED IN SESSION 5 (file v50-8)

### Corpus source
- **`/mnt/project/ET_Quintic_Shadow_d5_Complete_Investigation.md` QS-1** (line 538): "The d=5 quintic sublattice is absent from 12ET (since 5 ∤ 12) but projects its geometric structure onto the d=3 cubic sublattice via the Fibonacci convergence chain. This projection is the **Quintic Shadow**: the cubic sublattice carries a secondary structure induced by the asymptotic approach of Fibonacci ratios to φ."
- Proof shows: 8/5, 13/8, 21/13, 34/21, 55/34 all map to k=8, d=3 at 12ET. **d=3 is the cubic attractor for the d=5 Fibonacci chain.**
- **QS-3 corollary**: φ is *never* a native d=5 element. Its quintic character is inherited via the Fibonacci chain, not native.
- **QS-5** (line 656+): Quintic shadow coupling constant `α₅ = 1/(4·d) = 1/20 = 0.05`. Equivalently, `α₅ = (3/5)·V = (F₅/F₆)·V` — the coupling-to-variance ratio is itself a Fibonacci number.
- **QS-7 corollary line 775**: "the alternating sign of the Fibonacci convergent epsilons: the convergents oscillate above and below the k=8 position like a damped oscillation approaching the cubic sublattice from both the quintic-high and quintic-low side. **This oscillation IS the d=5 shadow** — the quintic force 'shimmering' around the cubic attractor."
- **QS-9** (line 847+): d=10 = 2×5 is the binary×quintic composite — φ's true home at 60ET (d=10, ε=−6.91¢). The d=10 contribution couples through 1/φ.
- **QS-15** (line 1086+): the d=5 quintic comma at 12ET is `ε₅ = (log₂5 − 7/3)·1200 ≈ −13.686¢`. EPS5/1200 is the dimensionless cents-shift (negative — d=5 sits below k=8 in 12ET semitones).
- **§9.2** (line 1208+): Fibonacci epsilon envelope decay rate. "ε(F_{n+1}/F_n) ≈ (−1)^(n−1)·C/φ^n" — alternating sign with φ-rate exponential decay. **"The exponential rate IS φ — the Fibonacci cascade converges to its own attractor φ at rate 1/φ per step."**
- **QS-8 / §8.2**: 12-fold periodicity of the φ-power tour — φ¹² returns to d=3, the same family as φ¹. One full N=12 manifold cycle corresponds to one convergent step in the Fibonacci cascade.

### Prior code (file v50-7)
```python
if mode_id == 5:
    EPS5=(math.log2(5.)-7./3.)*1200.; pi=1./PHI
    def _e(z,r,th,kt,n,_e5=EPS5,_pi=pi):
        p25=12./5.; p12=1.2; fb=kt*(LN2/5.)
        return (_e5/1200.)*(r**p25*np.exp(1j*(p25*th+fb))+r**p12*np.exp(1j*p12*th)*_pi)
    return _e
```
Computed: `(EPS5/1200) × (d=5 contribution + (1/φ)·d=10 contribution)`. **Missing the d=3 cubic attractor entirely** — the corpus is explicit that the shadow projects ONTO d=3, but the prior code had no d=3 receiver. Also missing: the alternating Fibonacci sign, the φ-decay envelope, the explicit α₅ coupling, and direct z dependence (z parameter unused).

### Status
**COMPLETE in v50-8.** Both the Python lambda in `build_extra()` and the matching CUDA kernel block were replaced with the full and proper implementation. The Julia c anchor (line 1955) and the build_mode weight boost for d=5,10 (line 2382) are unchanged — only `build_extra()` and the CUDA kernel block were modified.

### What was implemented (Session 5, file v50-8)

**Python lambda in `build_extra()`** — full and proper Quintic Shadow:

```python
EPS5     = (math.log2(5.) - 7./3.) * 1200.   # quintic comma at 12ET ≈ −13.686¢
INV_PHI  = 1.0 / PHI                          # 1/φ ≈ 0.618 — d=10 / decay
ALPHA5   = 1.0 / 20.0                         # QS-5 quintic shadow coupling
_qs_p5   = 12.0 / 5.0
_qs_p3   = 12.0 / 3.0
_qs_p10  = 12.0 / 10.0
_qs_kt5  = LN2 / 5.0
_qs_kt3  = LN2 / 3.0
_qs_kt10 = LN2 / 10.0
_qs_pre  = (EPS5 / 1200.0) * ALPHA5            # combined prefactor
_qs_v2   = V * V                                # z-anchor scale
def _e(z, r, th, kt, n,
       _pre=_qs_pre, _ip=INV_PHI, _v2=_qs_v2,
       _p5=_qs_p5, _p3=_qs_p3, _p10=_qs_p10,
       _k5=_qs_kt5, _k3=_qs_kt3, _k10=_qs_kt10):
    # d=5 source carrier (the quintic sublattice that casts the shadow)
    z_5  = (r ** _p5)  * np.exp(1j * (_p5  * th + kt * _k5))
    # d=3 cubic attractor carrier (the receiver — QS-1) — the missing Descriptor
    z_3  = (r ** _p3)  * np.exp(1j * (_p3  * th + kt * _k3))
    # d=10 = d=2 × d=5 binary×quintic composite (QS-9) with its own kt rotation
    z_10 = (r ** _p10) * np.exp(1j * (_p10 * th + kt * _k10)) * _ip
    sign    = 1.0 - 2.0 * (n % 2)        # Fibonacci convergent alternation (QS-7)
    damping = _ip ** (n // N)            # φ-rate decay envelope (§9.2)
    shadow_diff = (z_5 - z_3) * sign * damping
    z_anchor    = _v2 * z * damping
    return _pre * (shadow_diff + z_10) + z_anchor
```

**Six structural decisions made in Session 5** (resolving the gaps in the original plan's pseudocode):

1. **The d=3 cubic attractor `z_3` was added with its own `kt·LN2/3` rotation.** The plan's pseudocode used `rot_3 = kt * (LN2/3.)` but here it lives explicitly inside the `np.exp(1j * (_p3 * th + kt * _k3))` carrier, the same shape as z_5 and z_10. This is the QS-1 cubic attractor — the missing receiver — that the prior code lacked entirely.

2. **The d=10 carrier gained its own `kt·LN2/10` rotation.** The plan said the prior code's `r**p12 * np.exp(1j*p12*th) * _pi` was "phase-flat" (no kt rotation), and the plan's pseudocode preserved that limitation. Session 5 promotes the d=10 carrier to the same form as z_5 and z_3, giving it `kt * (LN2/10)` so the d=10 binary×quintic composite has its own T-axis lattice projection like every other carrier in the file. **This is an addition under the user's "improvement" allowance** — the prior code's limitation is treated as an implementation gap, not a feature.

3. **The damping envelope is `(1/φ)^(n // N)`, NOT the plan's `1/(1 + n·V)`.** The plan suggested a softer linear damping `1/(1 + n·V)` "as a starting point," but §9.2 of the corpus is explicit: *"The exponential rate IS φ — the Fibonacci cascade converges to its own attractor φ at rate 1/φ per step."* The corpus formula is `ε(F_{n+1}/F_n) ≈ (−1)^(n−1)·C/φ^n`. Session 5 uses the φ-decay form exactly, with the convergent-step index `n // N` (one full N=12 manifold cycle = one convergent step in the cascade per QS-8's 12-fold φ-power tour). At n=0..11 envelope=1, n=12..23 = 1/φ ≈ 0.618, n=24..35 = 1/φ² ≈ 0.382, n=120 (10 cycles) ≈ 0.014.

4. **Explicit α₅ = 1/20 coupling per QS-5.** The plan's pseudocode multiplied the shadow by `_a5 = 1.0/20.0`; Session 5 absorbs this into the combined prefactor `_qs_pre = (EPS5 / 1200.0) * ALPHA5` so each call computes one multiply instead of two. The combined prefactor evaluates to `≈ −5.703e-04` (negative, since EPS5 ≈ −13.686¢ and α₅ > 0).

5. **V²·z anchor on the d=3 receiver, scaled by the same φ-decay damping.** The plan's pseudocode had no z-anchor — Mode 5 was the only mode in Session 5's scope where z parameter usage was an open question. Session 5 adds `z_anchor = V² · z · damping`, which: (a) closes the unused-z gap, (b) gives the d=3 cubic receiver an explicit pull on the orbit's current position, and (c) fades the anchor at the same φ-rate as the shadow itself (both shadow and anchor are aspects of the same Fibonacci-convergent dynamic, so they should fade in lockstep).

6. **Sign convention `sign = 1 − 2·(n%2)` starts at +1 for even n.** Combined with the negative EPS5 prefactor, this gives `pre · shadow_diff` = negative × (z_5 − z_3) at n=0 — corresponding to "first convergent (8/5) is below k=8 in 12ET" per QS-7's tabulation. The §9.2 formula technically writes `(−1)^(n−1)`, but flipping this convention is equivalent to flipping the sign of the EPS5 prefactor, which would contradict QS-15's explicit `EPS5 ≈ −13.686¢`. The chosen convention preserves the corpus's signed quintic comma exactly.

**CUDA parity:** The Mode 5 kernel block at `_ET_RAWKERNEL_SRC` was replaced with the matching implementation. Three carriers (`z_5`, `z_3`, `z_10`), the alternating sign `_sign = 1.0f - 2.0f * (float)(n % 2)`, the φ-decay envelope `_damp = powf(inv_phi, (float)(n / 12))`, the structural prefactor `_pre = (eps5 / 1200.0f) * ALPHA5_F`, the V²·z·damping anchor — all bit-for-bit equivalent (within float round-off) to the Python lambda. **No kernel signature change** — Mode 5 reuses the existing `eps5` and `inv_phi` launch scalars. **`ALPHA5_F` was added as a `#define` constant** at the top of the kernel source (next to `INV_A0_F`), with the comment-traced derivation `α₅ = 1/(4·d) = 1/20 = (3/5)·V = (F₅/F₆)·V` per QS-5.

**f64 parity:** The `_make_f64_kernel` regex pipeline was tested against the new code in Session 5:
- `#define ALPHA5_F  0.05f` → `#define ALPHA5_F  0.05` (the macro-strip regex `(#define\s+\w+\s+\d+\.\d+)f\b` catches it).
- All `powf`/`cosf`/`sinf` calls in the Mode 5 block (5 + 3 + 3 = 11 calls) convert correctly via `intrinsic_map`.
- Zero stray `float` tokens remain in the f64 kernel; 242 `double` tokens after conversion.
- The `(float)(n / 12)` integer-division cast and the `(float)(n % 2)` cast both work cleanly in both precisions because integer division is precision-independent.

**Smoke test verification (CPU lambda, scalar at z = 0.5+0.3i, kt = 5.0):**
- n=0..3: outputs vary smoothly with sign alternation (n=0 and n=2 identical, n=1 and n=3 identical — the φ-decay envelope is still 1.0 within the first convergent step).
- n=12, 24, 36: magnitude decays geometrically (4.4e-3 → 2.3e-3 → 1.4e-3 → 7.8e-4) — the (1/φ) ≈ 0.618 ratio per N=12 step is visible, confirming the φ-decay envelope is working.
- ndarray (4×4) input: shape preserved, no NaN/inf, |min|=2.6e-3 |max|=1.9e-2.
- ET identity: `(3/5)·V == α₅` confirmed (both equal 0.05, match within 1e-12).
- EPS5 = −13.686286¢ (negative as required by QS-15).

### Visual prediction
Mode 5 will show **golden-quintic spiral filaments converging onto cubic (3-fold) lattice positions** with a φ-rate damped oscillating "shimmer" — the Fibonacci convergence becoming visible as alternating bright/dark bands that approach the d=3 cubic attractor asymptotically. After ~8 full N=12 manifold cycles (≈ 96 iterations), the shadow envelope drops to ~1.7% of its initial amplitude, so the visible quintic structure is concentrated in the first few hundred iterations and fades into pure 24-family base structure at high n. Combined with the d=3 z-anchor (same φ-decay), the orbit experiences a Fibonacci-rate "lock-on" pull toward the cubic receiver that shimmers and fades over the first hundred steps — the Fibonacci cascade made visible as image structure.

---

## Mode 6 — Septic Otherworld

### Corpus source
- **`/mnt/project/ET_Fantastical_Configurations.md` §8** (line 408+): "The Septic Barrier: The Other World (d=7)"
- **§8.1**: d=7 first appears at 2520ET (LCM 1..7). Generates 7-fold rotational symmetry. By the crystallographic restriction theorem, **7-fold symmetry cannot be embedded in 3D space**.
- **§8.2**: "Heptagram geometry: The seven-pointed star (heptagram) is the geometric signature of the d=7 sublattice."
- **§8.3 Partial Traversal Problem**: Traversers from the local d=3 region can approach the septic boundary but cannot fully enter while retaining d=3 Descriptor structure. The solution is **asymptotic**: T accumulates d=7-compatible Descriptors progressively, asymptotically entering without ever fully departing the d=3 framework. "The 'thinning of the veil'" — the Asymptotic Approach Theorem.

### Current code
```python
if mode_id == 6:
    def _e(z,r,th,kt,n):
        p=12./7.; rot=kt*(LN2/7.)
        return 1j*(r**p)*np.exp(1j*(p*th+rot))*V
    return _e
```
This is just `i × V × septic carrier` — a 90° rotation of the d=7 carrier. There is no 7-fold symmetry, no heptagram, no asymptotic approach, no n-dependent veil-thinning.

### Status — major gap
**The 7-fold symmetry that the entire corpus interpretation hinges on is missing.** The current code doesn't generate 7-fold rotational structure; it just adds a single d=7 contribution rotated by 90°. The heptagram is the geometric signature per the corpus — the math should produce one.

### Required completion (full and proper)

```python
if mode_id == 6:
    # Heptagram phases: 7 points equally spaced on a circle, indexed by k=0..6
    HEPTAGRAM_PHASES = np.array([2.0*math.pi*k/7.0 for k in range(7)], dtype=np.float64)
    def _e(z, r, th, kt, n, _hp=HEPTAGRAM_PHASES):
        p = 12./7.                              # d=7 carrier power
        # Heptagram: superposition of 7 contributions at angles 2πk/7 around z
        # Each vertex of the heptagram contributes a d=7 term rotated by its angle
        carrier_r = r**p
        # Sum over 7 heptagram vertices — corpus §8.2 geometric signature
        total = complex(0, 0)
        for k in range(7):
            vertex_phase = _hp[k]
            # Each vertex gets a d=7 contribution rotated by its angular position
            ang = p*th + kt*(LN2/7.) + vertex_phase
            total += np.exp(1j*ang)
        # Asymptotic approach factor (corpus §8.3): T approaches d=7 by accumulating
        # d=7 Descriptors. The "thinning of the veil" — asymptotic approach to full
        # d=7 traversal as n grows. Saturating function: 1 - exp(-n·V)
        # At n=0 the veil is fully closed; as n→∞ the veil thins toward 1.
        veil_thinning = 1.0 - np.exp(-n * V)
        # Full septic contribution: 1/7 normalization (mean of 7 vertices) × carrier
        # × veil thinning × V scale × i (90° Otherworld rotation)
        heptagram = (total / 7.0) * carrier_r * V * veil_thinning * 1j
        # Direct z dependence: the partial-traversal anchor — the orbit's current
        # position is the "this side of the veil" reference point
        anchor = V*V * z * veil_thinning
        return heptagram + anchor
    return _e
```

**Uses all five parameters.** The unrolled 7-vertex sum is a small loop in CPU NumPy; in CUDA it would be `#pragma unroll`'d.

### Difficulty
**High.** Requires CUDA kernel update with HEPTAGRAM_PHASES `__constant__` array, an unrolled 7-iteration loop, and the asymptotic approach factor. Visual testing critical to confirm 7-fold structure appears.

### Visual prediction
Mode 6 will produce **visible 7-pointed-star (heptagram) artifacts** in the connected set — patterns that **literally cannot exist in any fixed-power fractal** because no other fractal can express crystallographically forbidden 7-fold symmetry. The asymptotic veil-thinning will produce a "fading-in" effect where the heptagram structure becomes more pronounced at higher iteration counts. This is one of the strongest visual signatures in the entire mode set.

---

## Mode 7 — Nonic Recursion — ✅ IMPLEMENTED IN SESSION 5 (file v50-8)

### Corpus source
- **`/mnt/project/ET_Fantastical_Configurations.md` §9** (line 439+): "d=9 = 3² — the second power of the cubic primitive."
- "A d=9-governed configuration is a 'cubic configuration of cubic configurations' — a meta-level of the three-dimensional structure where each position in the cubic lattice is itself a cubic lattice."
- "**As above, so below** is the ET description of d=9 governance. ... Holographic magic: any piece of the configuration contains the whole (because d=9 = d=3², each cubic cell contains a full cubic structure)."
- "Infinite regress / summoning from within: A d=9 configuration contains itself at every level — 'a world within a world' is a d=9 lattice description."
- Impedance: A₀ = (12/9 − 1)² + 16 ≈ 16.11 (nearly maximal coupling efficiency — "fractal magic is nearly maximally efficient because the meta-cubic structure is very close to the trivial sublattice in terms of impedance overhead").

### Prior code (file v50-7)
```python
if mode_id == 7:
    def _e(z,r,th,kt,n):
        p1=4./3.; p2=16./9.; rot=kt*(LN2/9.)
        return V*((r**p1)*np.exp(1j*(p1*th+rot))+V*(r**p2)*np.exp(1j*(p2*th+rot*p1)))
    return _e
```
A manually-unrolled 2-level cubic recursion: `V·(z^(12/9) + V·z^((12/9)²))`. The nesting structure correctly expresses `9 = 3 × 3` at two levels, but it captures only two scales of the holographic structure named in §9 — "any piece contains the whole" and "infinite regress / summoning from within" require recursion at every scale, not just two manually unrolled levels. Also missing: direct z dependence (z parameter unused), n-driven depth ("as above so below" temporally), and the recursive feed-forward where each level operates on the previous level's accumulated result.

### Status
**COMPLETE in v50-8.** Both the Python lambda in `build_extra()` and the matching CUDA kernel block were replaced with the full holographic-recursion implementation. The Julia c anchor (line 1967) and the build_mode weight boost for d=9,3 (line 2392) are unchanged — only `build_extra()` and the CUDA kernel block were modified.

### What was implemented (Session 5, file v50-8)

**Python lambda in `build_extra()`** — full and proper Nonic Recursion with holographic depth:

```python
_MAX_NONIC_DEPTH = 9                  # 9 = 3² literal depth target
_nr_pb           = 12.0 / 9.0          # base nonic power = 4/3
_nr_kb           = LN2 / 9.0           # base nonic kt rotation
_nr_v            = V                    # per-level scale
_nr_v2           = V * V               # self-call anchor scale
def _e(z, r, th, kt, n,
       _md=_MAX_NONIC_DEPTH, _pb=_nr_pb, _kb=_nr_kb,
       _v=_nr_v, _v2=_nr_v2):
    active_depth = 1 + (n // N)
    if active_depth > _md:
        active_depth = _md
    accum      = _v2 * z                # V² · z self-call anchor
    current_r  = r
    current_th = th
    for level in range(active_depth):
        p_level   = _pb ** (level + 1)
        rot_level = kt * _kb * (level + 1)
        r_pow     = current_r ** p_level
        term      = _v * r_pow * np.exp(1j * (p_level * current_th + rot_level))
        accum     = accum + term
        current_r  = np.abs(accum) + 1e-300    # recursive feed-forward
        current_th = np.angle(accum)
    return accum / float(active_depth)         # holographic normalisation
```

**Five structural decisions made in Session 5** (resolving the gaps in the original plan's pseudocode):

1. **`MAX_NONIC_DEPTH = 9` is the literal `3²` depth target.** The plan suggested `MAX_NONIC_DEPTH = 9` "as a starting point"; Session 5 confirms this is the corpus-correct value: 9 = 3² means nine total cubic transforms when the d=9 = "cubic configuration of cubic configurations" structure is fully unrolled. This is not arbitrary — it derives from the d=9 = 3² identity in §9.

2. **`np.angle(accum)` instead of the plan's `math.atan2(accum.imag if hasattr(accum, 'imag') else 0, accum.real if hasattr(accum, 'real') else 1)`.** The plan's pseudocode had a `hasattr` check that would not work correctly on NumPy arrays (every NumPy complex array has `.imag`/`.real`, but `math.atan2` only takes scalars — broadcasting would fail). `np.angle` works correctly on both scalars and ndarrays without any conditional, which is what the vectorised CPU path needs. Same for `np.abs` instead of `abs`.

3. **`active_depth = min(1 + (n // N), _md)`** — the depth advances by one level per full N=12 manifold cycle, expressing "as above so below" temporally. Saturates at 9 once n ≥ 8·N = 96. At n=0..11 depth=1 (just the d=9 carrier), at n=12..23 depth=2 (which matches the prior code's 2-level structure as a special case), at n=96+ depth=9 (full holographic depth). The prior code's 2-level structure is preserved as the n ∈ [12, 23] window of the new dynamic-depth formula.

4. **`accum = V² · z` self-call anchor** — the orbit's current position IS the recursion's self-reference, expressing "summoning from within" per §9 directly through the z parameter. V² scales it to the same magnitude as the other modes' z-anchors (Subsumption-friendly relative to the 24-family base sum).

5. **Divide by `float(active_depth)` for holographic normalisation.** Without this, deep recursion would accumulate unbounded amplitude (each level adds a `V·r^p` term, and 9 such terms would push the contribution to ~9V which competes with the base sum). The normalisation keeps the per-step contribution at the same overall scale regardless of depth — only the structural complexity grows with n, not the magnitude (per the Subsumption Law).

**CUDA parity:** The Mode 7 kernel block was replaced with the matching per-thread recursive loop:

```c
if (use_mode7) {
    int _ad = 1 + (n / 12);
    if (_ad > 9) _ad = 9;
    float _pb  = 12.0f / 9.0f;
    float _kb  = LN2_F / 9.0f;
    float _ac_r = V_F * V_F * zr;       // V² · z self-call anchor
    float _ac_i = V_F * V_F * zi;
    float _cr  = _rr_cap;
    float _cth = th;
    for (int _lv = 0; _lv < _ad; _lv++) {
        float _lvp1 = (float)(_lv + 1);
        float _pl   = powf(_pb, _lvp1);
        float _rotl = k_t * _kb * _lvp1;
        float _rpl  = powf(_cr, _pl);
        float _agl  = _pl * _cth + _rotl;
        _ac_r += V_F * _rpl * cosf(_agl);
        _ac_i += V_F * _rpl * sinf(_agl);
        // Recursive feed-forward
        float _new_r2 = _ac_r * _ac_r + _ac_i * _ac_i;
        _cr  = sqrtf(_new_r2);
        if (_cr < 1e-38f) _cr = 1e-38f;
        _cth = atan2f(_ac_i, _ac_r);
    }
    float _inv_ad = 1.0f / (float)_ad;
    float _sc     = mew[7] * _inv_ad;
    znr += _sc * _ac_r;
    zni += _sc * _ac_i;
}
```

The variable loop bound `_ad = min(1 + n/12, 9)` cannot be `#pragma unroll`'d (the bound depends on the outer iteration variable `n`), but the body is straight `cos`/`sin`/`pow` with a single dependency chain — the compiler generates an efficient sequential loop. Maximum 9 iterations per pixel-step. **No kernel signature change** — Mode 7 needs no new launch parameters.

**f64 parity:** The `_make_f64_kernel` regex pipeline handles the new code:
- `powf` (2 calls), `sqrtf` (1 call), `atan2f` (1 call), `cosf` (1 call), `sinf` (1 call) all in `intrinsic_map` — convert correctly to `pow`/`sqrt`/`atan2`/`cos`/`sin`.
- The `1e-38f` floor gets the `f` stripped to `1e-38` by the suffix regex, then converted to `1e-300` by the underflow-guard step at the end of `_make_f64_kernel` — matching the Python lambda's `1e-300` floor.
- The `(float)(n / 12)` integer cast and the `(int)_lv` loop counter are precision-independent.

**Smoke test verification (CPU lambda, scalar at z = 0.5+0.3i, kt = 5.0):**
- n=0, depth=1: |out| = 4.41e-2 (one level — z² anchor + one cubic carrier).
- n=11, depth=1: identical to n=0 (same depth, same kt — confirming the depth window is correct).
- n=12, depth=2: |out| = 2.20e-2 (two levels, magnitude halved by the divide-by-active_depth normalisation — confirming the normalisation works).
- n=23, depth=2: identical to n=12.
- n=24, depth=3: |out| = 1.47e-2 (three levels).
- n=96, 200, 500: all depth=9, all identical |out| = 4.89e-3 — **the depth saturation cap is verified, and the cap survives indefinitely**.
- ndarray (4×4) input: shape preserved, no NaN/inf, |min|=2.05e-2 |max|=1.93e-1.
- ET identity: `_MAX_NONIC_DEPTH = 9 = 3² = 3 × 3` (verified).

### Visual prediction
Mode 7 will produce **fractal-within-fractal nesting** with depth that grows as iteration progresses. For the first N=12 steps the contribution is a single d=9 cubic carrier with V²·z self-call anchor (essentially the prior code's first term plus an explicit z anchor). From iteration 12 onward, the recursive feed-forward kicks in: level 2's input is `(|accum_1|, arg(accum_1))`, level 3's input is `(|accum_2|, arg(accum_2))`, etc. — each subsequent level is a full cubic transformation of the prior level's accumulated result. By iteration 96 the recursion is at full depth 9, and the connected set will show **3-fold cubic lobes that, when zoomed in, reveal smaller 3-fold cubic lobes inside them, recursively to nine levels** — "as above, so below" made literal. The divide-by-active_depth normalisation keeps the visual contribution bounded, so the depth changes the *structure* visible at high zoom levels rather than just the *brightness* of the contribution.

---

## Mode 8 — Magical Impedance — A₀=137 Coupling — ✅ IMPLEMENTED IN SESSION 3 (file v50-6)

### CRITICAL CORPUS TRACE FINDING (Session 3)

The original plan section below quoted the §5.2 table from `ET_Fantastical_Configurations.md` and used it as the source of truth. **During Session 3, cross-tracing against `/mnt/project/ET_Fine_Structure_Constant_REVISED.md` revealed that the §5.2 formula is the OLDER, broken derivation, and §3.3 Table 2 of the same document is the corrected form that was added later but never propagated back to §5.2.** The corpus is internally inconsistent here because §5.2 was not updated when §3.3 Table 2 was added.

**The two formulas in Fantastical Configurations:**

1. **§5.1 / §5.2 (OLDER, broken):** `N_magic = 12/d_prim` and `A₀_magic = (N_magic − 1)² + S²`. Under this:
   - d=12 → N_magic=1 → A₀ = 16 (claimed to be MAX coupling)
   - d=1 → N_magic=12 → A₀ = 137 (claimed to be baseline)
   - There is literally an editorial fragment at §3.3 line 138 — `"— wait: use N=12 for full lattice"` — showing the author noticed something was wrong mid-derivation but didn't propagate the fix to §5.2.

2. **§3.3 Table 2 (NEWER, "the more profound case", line 148):** `A₀_magic = (d_prim − 1)² + S²` directly. Under this:
   - d=1 → A₀ = 16 → ξ = 8.5625× (Pure Will, MAX coupling — no sublattice mediation overhead)
   - d=12 → A₀ = 137 → ξ = 1.0× (Full-Res/EM, baseline — recovers canonical Fine Structure REVISED A₀ = 137)
   - Monotonically increasing in d → matches the corpus narrative line 161: *"All magical configurations have stronger T-P coupling than our local physics. This is structurally necessary: magic is defined by more direct T-P coupling, which requires lower impedance, which requires lower d_primary."*

**Why the OLD formula is structurally wrong:**

1. **Physical inversion.** The OLD formula puts our local EM (d=12) at maximum coupling and Pure Will (d=1) at baseline, which inverts the corpus narrative ("lower d → less mediation → stronger coupling").

2. **Conflict with the canonical N=12.** `/mnt/project/ET_Fine_Structure_Constant_REVISED.md` defines `A₀ = (N-1)² + S²` with N=12 fixed (derived from `|Π|·S = 3·4 = 12`). The OLD formula `N_magic = 12/d_prim` makes N variable per sublattice, contradicting the canonical derivation. Treating N as variable is treating manifold symmetry as variable, which would require either |Π| ≠ 3 or S ≠ 4, both of which are derived constants — so N cannot vary. The corrected §3.3 Table 2 formula keeps N=12 globally fixed and uses d_prim directly as the magical-resolution variable.

**Conclusion:** Session 3 uses the corrected §3.3 Table 2 formula. The §5.2 truncated 8-entry table (Pure Will through Recursive/Fractal, missing d=8, 10, 11, 12) is also a §5.2 artifact — the canonical formula gives valid impedance values for ALL 12 sublattice families. Mode 8 cycles through all 12 (`n % 12`), harmonising with N=12 and the existing palindrome cycle.

### Existing-code audit fix (Session 3)

The pre-existing `FAM_COUPLING` constant in the script (line 1100 of v50-5) had its **comment correctly stating the new formula** (`A₀_magic = (d−1)² + S²` with values d=1→8.56, d=3→6.85, d=12→1.0) but the **code on the next line implemented the OLD broken formula** (`(N/d - 1.0)**2 + S²`). Code did not match its own documentation. The same bug existed in the audio module's `_FAM_COUPLING` at line 3932. Both were fixed in Session 3 to use the corrected `(d-1)² + S²` formula.

### Corpus source (CORRECTED)
- **`/mnt/project/ET_Fine_Structure_Constant_REVISED.md`** — canonical `A₀ = (N-1)² + S²` with N=12 fixed, giving the baseline 137.
- **`/mnt/project/ET_Fantastical_Configurations.md` §3.3 Table 2** — corrected per-sublattice generalisation `A₀_magic = (d - 1)² + S²` (the "more profound case").
- **`/mnt/project/ET_Fantastical_Configurations.md` §3.3 line 161** — corpus narrative confirming "lower d → less mediation → stronger coupling", which matches the corrected formula's monotonicity.
- **The corrected impedance table (all 12 sublattice families):**

  | d  | A₀_magic | ξ          | Character             |
  |----|----------|------------|-----------------------|
  | 1  | 16       | 8.5625×    | Pure Will / Elemental |
  | 2  | 17       | 8.0588×    | Mirror / Binary       |
  | 3  | 20       | 6.8500×    | Cubic / Volumetric    |
  | 4  | 25       | 5.4800×    | Quartic / Temporal    |
  | 5  | 32       | 4.2812×    | Quintic / Sympathetic |
  | 6  | 41       | 3.3415×    | Hexadic / Harmonic    |
  | 7  | 52       | 2.6346×    | Septic / Otherworld   |
  | 8  | 65       | 2.1077×    | Octet / Shadow        |
  | 9  | 80       | 1.7125×    | Nonic / Recursive     |
  | 10 | 97       | 1.4124×    | Decic / φ-Binary      |
  | 11 | 116      | 1.1810×    | Undecimal / Prime     |
  | 12 | 137      | 1.0000×    | Full-Res / EM (baseline) |

### Prior code (file v50-5)
```python
if mode_id == 8:
    def _e(z,r,th,kt,n):
        return np.sin(r*LN2*N)*V*np.exp(1j*th)
    return _e
```
A radial standing wave with **zero reference** to A₀=137, the impedance formula, or the impedance table. This was the largest corpus-vs-code gap of any mode.

### What was implemented (Session 3, file v50-6)

**New module-level constants** were added near `_PALINDROME` (in the lookup-table section):

```python
_IMPEDANCE_D     = np.arange(1, 13, dtype=np.float64)        # 1..12
_IMPEDANCE_A0    = (_IMPEDANCE_D - 1.0)**2 + S_STATES**2     # (d-1)² + 16
_IMPEDANCE_XI    = A0_EM / _IMPEDANCE_A0                     # 137 / A0
_IMPEDANCE_XIMAX = float(_IMPEDANCE_XI.max())                # = 8.5625 at d=1
_IMPEDANCE_XIN   = _IMPEDANCE_XI / _IMPEDANCE_XIMAX          # bounded [0.1168, 1]
N_MAGIC_TYPES    = int(_IMPEDANCE_D.size)                    # = 12
```

**The Python lambda in `build_extra()`** was replaced with the cycling implementation:

```python
if mode_id == 8:
    _imp_d   = _IMPEDANCE_D
    _imp_xin = _IMPEDANCE_XIN
    _nt      = N_MAGIC_TYPES
    _inv_A0  = 1.0 / A0_EM
    def _e(z, r, th, kt, n,
           _id=_imp_d, _ix=_imp_xin, _nt=_nt, _ia=_inv_A0):
        idx = n % _nt
        d_magic = float(_id[idx])
        xi_norm = float(_ix[idx])
        p_magic   = 12.0 / d_magic
        rot_magic = kt * (LN2 / d_magic)
        carrier   = (r ** p_magic) * np.exp(1j * (p_magic * th + rot_magic))
        carrier_term = V * xi_norm * carrier
        z_anchor = (V * V) * z * (xi_norm - _ia)
        return carrier_term + z_anchor
    return _e
```

**CUDA kernel side:** New `__constant__` arrays `IMPEDANCE_D[12]`, `IMPEDANCE_XIN[12]`, and the macro `INV_A0_F = 1.0f/137.0f` were added near the existing `TRAP_R` declaration. The Mode 8 kernel block (the old `sin(rc·LN2·N)·V·exp(iθ)` block) was replaced with the matching cycling lookup using `IMPEDANCE_D[n%12]` and `IMPEDANCE_XIN[n%12]`. No kernel signature change was needed (`use_mode8` already existed in the signature).

**On the anchor formula:** The plan originally proposed `(coupling - 1/137)` for the z-anchor with the comment "vanishes at the local-EM baseline". Under the corrected impedance table this no longer literally vanishes at d=12 — `xi_norm[d=12] = 16/137 ≈ 0.1168`, so `(xi_norm - 1/137) ≈ 0.1095` at d=12 instead of 0. **Per the user's preference for visual impact over mathematical elegance**, the formula was kept as `(xi_norm - 1/137)` because:
- It never vanishes — every cycling step contributes a visible z-pull
- Range across the cycle: [0.1095, 0.9927], ratio max/min ≈ **9.07×**
- This is the strongest natural contrast of any reasonable normalisation
- Combined with the carrier-term contrast, smoke-tested amplitude ratio between d=1 and d=12 is **~18.3×** per step — strongly visible 12-band structure

**Smoke test verification:**
- All 12 sublattice families visited in n=0..11 ✓
- Cycle wraps cleanly (n=0 output identical to n=12 output) ✓
- Anchor coefficient never zero (min = 0.1095 across the cycle) ✓
- d=1 vs d=12 amplitude ratio: 18.32× ✓
- All outputs finite ✓

### Note on rotating vs cycling
The original "rotating vs cycling" debate is moot under the corrected impedance table — cycling through all 12 sublattice families is the only formulation that uses every entry of the corrected table. The truncated 8-entry table that motivated the original "8 magic types, n%8" was a §5.2 artifact, not a corpus statement.

### Visual prediction (corrected)
Mode 8 produces **visible 12-band regime structure** — the connected set will show 12 distinct character regions corresponding to the 12 sublattice-family magical regimes, with smooth transitions between them. Pure Will regions (d=1, max coupling, ~18× amplitude vs d=12) will be intense; Full-Res/EM regions (d=12, baseline coupling) will be subtle. The 12-step cycle harmonises with the manifold symmetry N=12 and the existing palindrome cycle, so Mode 8 + Mode 2 (Descriptor Cascade) blended together will lock into phase visually.

---

## Mode 9 — Exception State — ✅ IMPLEMENTED IN SESSION 6 (file v50-9)

### Corpus source
The corpus research for Mode 9 was completed in Session 6 against the following sources, all in `/mnt/project/`:

- **`ExceptionTheory.md` Part XI (line 742+)** — Variance and the Grounding Function. The canonical statement: `V(E) = 0` (line 760) — the Exception is the unique configuration with zero variance, the only thing that cannot be otherwise at the current moment. Line 782: "the unique fixed point with zero variance — everything else flows around it." The Grounding Function `G: 𝒞 → {0,1}` with `G(c) = 1 ⟺ V(c) = 0`.
- **`ExceptionTheory.md` §Observational Displacement (line 786+)** — "The Exception cannot be observed directly. Any observation creates a new configuration with positive variance. ... When T observes E, the act of observation creates T∘E, which is a new configuration distinct from E alone. The original E is displaced the moment you try to observe it." This is the corpus-derived dynamic that justifies the `exp(-n · V)` Observational Displacement residual.
- **`ExceptionTheory.md` §Variance Dependence (line 2347+)** — "The 'thickness' of agential time scales inversely with local variance: high-variance regions feel time as thick/slow; low-variance (near Exception) feel time as thin/fast." This justifies the Lorentzian variance kernel as a smooth profile peaked at the grounding fixed point.
- **`ET_Incoherence_Paper.md` §10 (line 385+)** — "The Exception is a closed set. {P,D,T} contains its own boundary. ... Zero variance is the mathematical expression of this closure: the Exception contains its own ground." `∂E ⊂ E`. The Exception is the only state that is fully self-containing.
- **`ET_Incoherence_Paper.md` §23 (line 1049+)** — Elegance-Variance duality: "The Elegance Score is the inverse proxy for Variance: high 𝓔 directly maps to low V(c), and low 𝓔 maps to high V(c)" (line 1067-1069). The §23 elegance kernel `𝓔(r) = (n/d) · 100/(100+|ε|) · 100/(p+q)` is the lattice-discrete form; the smooth Lorentzian `1/(1+V·d²)` is its smooth ET-native analog (peaks at d=0, decays monotonically as d grows).
- **`M-states.md` lines 459, 723** — "Pure E-vacuum = 2/3 = 66.7%". The Koide ratio K = 2/3 IS the cosmological weight of the static Exception. Not coincidence — the corpus is explicit: the cosmological E-vacuum fraction equals the Koide ratio exactly. This justifies the explicit `G(z) = K · V_loc(z)` Koide grounding weight.
- **`M-states.md` line 712, line 950** — the active mediation (3% total cosmological energy) splits 8:7 between M-vacuum (1.6%, vacuum-like, distributed uniformly, w ≈ -1) and M-matter (1.4%, matter-like, localized in matter, w ≈ 0). The 8:7 ratio gives `8/(8+7) = 8/15` and `7/(8+7) = 7/15`. M-vacuum is z-uniform (does not localize); M-matter is z-localized (the negative pull toward the Exception).
- **`M-states.md` §M-State Fraction (line 510+)** — `f_M = (1/12) × 0.36 × 1.0 = 0.03 = 3%`. The 0.36 factor is `(3/5)² = (F₅/F₆)²` (Fibonacci-encoded), and `(3/5)·V = α₅` is the QS-5 quintic shadow coupling constant — the same coupling Mode 5 uses. **`f_M = (F₅/F₆)² · V`** — fully ET-derived, structurally connecting Mode 9 to the Mode 5 Fibonacci chain via Quintic Shadow §QS-5.

### Status
**COMPLETE in v50-9.** Both the Python lambda in `build_extra()` and the matching CUDA kernel block have been replaced with the full corpus-faithful implementation. See the Session 6 Completion Audit below for the full implementation block, six structural decisions, smoke test results, and CPU/GPU parity verification (max error 2.17e-19 across five test cases).

### Prior code (file v50-8)
```python
if mode_id == 9:
    def _e(z,r,th,kt,n):
        va=1./(1.+r*r*V); twist=th*(N/12.)+kt*(LN2/N)
        return -va*z*V*np.exp(1j*twist*V)
    return _e
```
- `va = 1/(1 + r²·V)` — a Lorentzian-like profile in r (peaks at r=0, decays as r→∞). Correct in spirit per §23 but missing the explicit grounding-fixed-point reference, the Koide cosmological weight, and the 8:7 M-state split.
- `twist = θ + kt·LN2/N` — the prior code's `θ·(N/12) + kt·LN2/N` reduced to `θ + kt·LN2/N` because N/12 = 1, losing the per-step n-cumulative phase advance and the Koide K weighting on kt.
- Returns `-va · z · V · exp(i · twist · V)` — a negative pull on z scaled by the Lorentzian. Correct in spirit (z-localized matter-like pull toward the Exception) but missing the M-vacuum component, the explicit Koide cosmological weight, the Observational Displacement residual, and the cumulative manifold rotation.
- The unused parameter was `n`. Mode 9's Session 6 implementation uses `n` in two places: the Observational Displacement residual `exp(-n·V)` and the cumulative manifold rotation `2π·n/N`.

---

## Mode 10 — Lagrangian Field

### Corpus source
**RESEARCHED AND IMPLEMENTED IN SESSION 7 (file v50-10).**

Authoritative source: `/mnt/project/ET_Lagrangian_Field_Theory.md` §VIII (line 562+) — the canonical Mexican-hat derivation:
- §VIII.1 line 569: `V(φ) = −μ²|φ|² + λ|φ|⁴   (μ², λ > 0)`
- §VIII.1 line 575: `|φ| = v = √(μ²/2λ)`  (vacuum expectation value)
- §VIII.2 line 600: `σ(x) = |φ| − v`  (Higgs radial mode)
- §VIII.3 line 651: `m_H = √(2μ²)`  (Higgs mass)
- §VIII.2 line 593-596: T's [0/0] resolution must pick ONE vacuum direction (the dynamic vacuum substantiation)
- §VIII.2 line 596: `Once T substantiates a vacuum φ₀ = v·e^{iθ₀}` (the orbit angle IS the Goldstone field π(x)'s value)
- §VIII.2 line 601-608: Goldstone is the unsubstantiated phase direction along the vacuum ring (massless tangent)
- §II.1: `δS = 0` is T's [0/0]→determinate resolution

Cross-corroborated by `/mnt/project/et_clr_v5__4_.py` THEOREM LFT-8 (lines 4742-4760) which reproduces the §VIII formula word-for-word — same potential, same `v = √(μ²/2λ)`, same `m_H = √(2μ²)`. **This is a second independent ET source for the same convention.** No competing convention exists in the corpus (verified by exhaustive grep across every `.md`/`.py`/`.txt` file for `Mexican`, `Higgs`, `vacuum manifold`, `Goldstone`, `μ²`, `vacuum expect`, `VEV`, `spontaneous symmetry breaking`, `v = sqrt`, `1/2.*μ²`, `½μ²`, `(1/4).*phi.*⁴`, `0.5*mu2`, `0.5*lam` — **zero hits for any half-coefficient form**).

### Prior code (BEFORE Session 7)
```python
if mode_id == 10:
    def _e(z,r,th,kt,n, _mu=_MH_MU2, _la=_MH_LAMBDA, _v=_MH_V):
        r2=r*r; grad=z*(_la*r2-_mu); eta=V*V*N
        gs=np.exp(1j*(th*V+kt*LN2/N))
        return -eta*grad + V*V*gs
    return _e
```
- `grad = z·(λ·r² − μ²)` — **GRADIENT BUG** (missing factor of 2 on quartic; see Audit Finding below)
- `eta = V²·N` — the dynamical scale (preserved by Session 7)
- `gs = exp(i·(θ·V + kt·LN2/N))` — Goldstone-mode-like phase (subsumed by Session 7's full Goldstone form)
- Returns `-eta·grad + V²·gs` — gradient flow + Goldstone source
- **`n` was completely unused** (~static field, no T-time dynamics)
- **`_v = _MH_V` was captured but never referenced** (the prior code never actually used the documented vacuum location)

### Status — Session 7 implementation
**COMPLETE in v50-10.** Both the Python lambda in `build_extra()` and the matching CUDA kernel block in `et_iterate` have been replaced with the full corpus-faithful implementation. See the Session 7 Completion Audit below for the gradient-bug derivation, the four corpus-named T-actions, the six closing Descriptors, the end-to-end smoke test results, and the CPU/GPU parity verification.

### Difficulty
**High** (the corpus convention had to be verified against TWO independent sources before any code was written; the prior code had a real arithmetic bug requiring a Wirtinger-derivative re-derivation; three new physical modes — Higgs, Goldstone, and vacuum-substantiation envelope — all had to be added with explicit n-dependence; CUDA parity required inline computation of derived constants `v` and `m_H` from the kernel scalars `mu2`/`lam_mh` to avoid a kernel signature change).

---

## Mode 11 — Route A/B Cascade

### Corpus source — RESEARCHED IN SESSION 8
- **`/mnt/project/ET_Weak_Sector_Four_Open_Questions.md` §2.2 line 105+** (Route B canonical: `6/5 → 9/8 → 3/2`, d=4→6→12, leptonic via Hexadic bridge) and §2.2 line 124+ (Route B CPT-complement: `5/3 → 16/9 → 2/3`, d=4→6→12, terminal at K=2/3 by octave-complementarity).
- **`/mnt/project/ET_Weak_Sector_Four_Open_Questions.md` §2.3 line 137+** (Route A canonical: `6/5 → 5/4 → 3/2`, d=4→3→12, hadronic via Strong sector).
- **`/mnt/project/ET_Weak_Sector_Four_Open_Questions.md` §4 line 315+** (CPT structure of Route A vs Route B), Theorem WS-8 (Route CPT correspondence — palindromic involution n↦N−n is discrete CPT, residue sums = N at every step), Theorem WS-9 (Route physical asymmetry — Route A hadronic via d=3 Strong, Route B leptonic via d=6 Hexadic).
- **`/mnt/project/ET_Weak_Sector_Open_Directions_Closed.md` OD2 line 128+ / Theorem WS-15** (Route A Koide Closure): the chain `6/5 → 5/4 → 2/3` is the **unique octave-closed completion** of the Route A d-sequence (d=4 → d=3 → d=12). Product = 1 exactly, ε-sum = 0¢ exactly, terminal = K = 2/3 (the Koide ratio). **This is the missing fourth route — Route AC — that the prior code did not include.**
- **`/mnt/project/ET_Weak_Sector_Open_Directions_Closed.md` OD4 line 277+ / Theorem WS-18** (Cabibbo Angle from ET Primitives): `λ = sin(θ_C) = √(K·V) = √(1/18) = 1/(3·√2) ≈ 0.2357`. The amplitude for T to traverse one inter-generation Hasse-distance step in the Route A sublattice hierarchy, ET-derived from K=2/3 and V=1/12 with no external inputs.
- **`/mnt/project/ET_Weak_Sector_Open_Directions_Closed.md` Theorem WS-20** (CKM Matrix from ET Primitives): the four routes carry the full Wolfenstein hierarchy as Hasse-distance powers of λ.

### Current code — REPLACED IN SESSION 8
The prior stub at lines 2829–2837 (v50-10):
```python
if mode_id == 11:
    _ra=ROUTE_A_RATIOS[:]; _rb=ROUTE_B_RATIOS[:]; _rc=ROUTE_BC_RATIOS[:]
    def _e(z,r,th,kt,n):
        step=n%3; cyc=(n//12)%3
        ratio=_ra[step] if cyc==0 else _rb[step] if cyc==1 else _rc[step]
        kk=round(N*math.log2(ratio)); dd=N//math.gcd(abs(kk) if kk!=0 else N,N)
        pp=12./dd; rr=kt*(LN2/dd)
        return V*(r**pp)*np.exp(1j*(pp*th+rr))
    return _e
```
Cycled through three routes (A, B, BC) every 12·3 = 36 steps.

`ROUTE_A_RATIOS = [6/5, 5/4, 3/2]` (lines 583–585 of v50-10), `ROUTE_B_RATIOS = [6/5, 9/8, 3/2]`, `ROUTE_BC_RATIOS = [5/3, 16/9, 2/3]`.

### Status — COMPLETE (Session 8, file v50-11)
**Seven Descriptor Gaps closed**, all corpus-derived. The new Mode 11 (file v50-11):
1. **Adds the missing fourth route — Route AC** (`6/5 → 5/4 → 2/3`, d=4→3→12, hadronic closed) — the WS-15 octave-closed Route A completion. New constants `ROUTE_AC_RATIOS` and `ROUTE_AC_D` added at lines 605–606 alongside the prior three route constants (which are preserved verbatim).
2. **Replaces the 36-step cycling formula `(n//12) % 3`** with the corpus-natural 12-step macro `(n // 3) % 4` — 4 routes × 3 steps = 12 = N, matching the manifold symmetry exactly.
3. **Hadronic/leptonic carrier sign per Theorem WS-9**: Routes A, AC carry sign +1 (hadronic, ascending Strong-crossing); Routes B, BC carry sign −1 (leptonic, descending hexadic-bridge). The lattice form of WS-8's CPT correspondence.
4. **Cabibbo mixing phasor per Theorem WS-18**: `phasor = exp(i · λ · route_idx)` with `λ = √(K·V) = 1/(3·√2) ≈ 0.2357`. Successive routes are separated by one Cabibbo phase step.
5. **K-terminal grounding pull per Theorem WS-15**: at the terminal step (step_idx = 2) of the closed routes (AC, BC), the orbit experiences a pull `(z_K − z)` toward `z_K = K + 0j`, expressing the Koide forced closing ratio.
6. **z-anchor**: `V² · z · (1 + λ · route_idx)` — the orbit's current position is now a real consumer of the lambda at every step (matching the all-modes convention from Sessions 1–7). All five lambda parameters (`z`, `r`, `th`, `kt`, `n`) are now real consumers.
7. **The prior carrier `V·(r**pp)·exp(i·(pp·th+rr))` is preserved exactly** as the central carrier of the new implementation; at n=0 (Route A, hadronic, route_idx=0) the new code reduces to `V·carrier + V²·z` exactly (sign=+1, phasor=1+0j, no grounding pull) — verified to 0.0e+00 diff in the n=0 invariant test.

### Implementation tasks — ALL DONE (Session 8)
1. ✅ **Read** `/mnt/project/ET_Weak_Sector_Open_Directions_Closed.md`, `ET_Weak_Sector_Four_Open_Questions.md`, `ET_Weak_Sector_d4_to_d12_Investigation.md` — full corpus trace done in Session 8.
2. ✅ **Verified** the three prior ratio sequences are corpus-correct, **identified Route AC as missing**, added it.
3. ✅ **z is now a direct consumer** via the V²·z·(1+λ·route_idx) anchor and the (z_K − z) grounding pull at closed-route terminals.
4. ✅ **CUDA parity**: full Mode 11 CUDA kernel block rewritten to mirror the Python lambda; new `__constant__ float ROUTE_RATIOS[12]`, `ROUTE_D[12]`, `__constant__ int ROUTE_HAD[4]`, `ROUTE_CLOSED[4]` arrays + `#define R_LAMBDA_F 0.23570226039551584f`, `#define R_ZK_F 0.66666666666666667f` macros added near the existing DELTA_K_TABLE / IMPEDANCE_D constants. f32 vs f64 parity verified (max diff 1.7e-8 vs float32 epsilon 1.19e-7).

### Difficulty
**Was Medium-Low → was Medium-High** (the Cabibbo-amplitude WS-18 derivation, the WS-15 closure verification, the CPT-pairing structure, and the corpus-faithful 4-route × 3-step macro all required deep cross-tracing across 3 corpus files).

---

## Implementation Order Recommendation

Original 9-session plan has been restructured to 6 sessions per user's request. Sessions 1, 2, and 3 are complete (see Implementation Progress table at the top of this document).

### Session 1 — ✅ COMPLETE (file v50-6)
- Mode 4 free fix (`r*exp(iθ)` → `z`) — already applied in v50-5 before this pass
- Dynamic `_NO_EXTRA` via `if p['extra'] is not None` — already applied in v50-5
- **NEW:** `N` (No Mode) option in `_choose_modes()` plus `build_no_mode()`, `NO_MODE_ID = -1`, and filename-stem `'none'` handling in both `generate_et_fractal()` and `generate_zoom_video()`

### Session 2 — ✅ COMPLETE (file v50-6)
- Mode 3 (Koide Boundary) full Python lambda with `V² · F_K · cycle · twist`
- New CUDA kernel block for Mode 3 inserted between Mode 2 and Mode 4
- `use_mode3` parameter inserted into kernel signature in ascending order
- Kernel launch site in `iterate_strip_v2` updated with `cp.int32(_active(3))`
- **Audit bug fix:** `expf → exp` added to both `_make_f64_kernel` and `_make_f64_di_kernel` intrinsic maps (was missing, would have silently broken f64 builds of any future exponential use)

### Session 3 — ✅ COMPLETE (file v50-6)
- Mode 8 (Magical Impedance, cycling) full Python lambda with 12-entry impedance cycling
- New CUDA `__constant__` arrays `IMPEDANCE_D[12]`, `IMPEDANCE_XIN[12]`, macro `INV_A0_F`
- Mode 8 kernel block replaced with cycling lookup (no signature change needed)
- **Corpus trace finding:** `ET_Fantastical_Configurations.md` §5.2 uses the OLDER `(12/d-1)²+S²` formula, §3.3 Table 2 uses the corrected `(d-1)²+S²` formula. The corrected form is structurally required by Fine Structure REVISED's fixed N=12. Session 3 uses the corrected form.
- **Audit bug fix:** `FAM_COUPLING` constant in v50-5 line 1100 had a comment stating the correct formula but code using the old broken formula. Same bug in `_FAM_COUPLING` at line 3932 (audio module). Both fixed.
- **Visual-impact decision (per user):** Kept `(xi_norm - 1/137)` anchor formula (does not literally vanish at d=12) because it gives the strongest cycle contrast (~18.3× amplitude ratio d=1 vs d=12) and every step contributes a visible z-pull.

### Session 4 — ⏳ PENDING — Mode 1 + Mode 6 (combined)
- **Mode 1** (Traverser Field, (7,1) torus knot): the mode the user explicitly named. Highest aesthetic impact. Uses d=7 and d=11 Route B primaries.
- **Mode 6** (Septic Otherworld, heptagram): the "crystallographically forbidden" visual is one of the strongest signatures. Corpus source: Fantastical Configurations §8 (the Septic Barrier).
- Both modes are heptagonal/d=7 family, so combining them into one session lets them share corpus research and kernel design patterns.

### Session 5 — ⏳ PENDING — Mode 5 + Mode 7 (combined)
- **Mode 5** (Quintic Shadow, Fibonacci d=3 attractor): subtle math (Fibonacci alternation, damped oscillation). Corpus source: Fantastical Configurations §7 (Sympathetic and Correspondential Magic: the Quintic).
- **Mode 7** (Nonic Recursion, holographic depth): most complex due to recursive accumulation; requires per-pixel inner loop in CPU path. Corpus source: Nonic family d=9 recursive/fractal from §5.2 corrected table and narrative context.

### Session 6 — ⏳ PENDING — Modes 9, 10, 11 + Verification Pass
- **Mode 9** (Exception State): `M-states.md` lookup for the Exception state `{P,D,T}` variance-pull semantics.
- **Mode 10** (Lagrangian Field): `ET_Lagrangian_Field_Theory.md` for the Mexican-hat potential and the field gradient term.
- **Mode 11** (Route A/B Cascade): `ET_Weak_Sector_*.md` for the Route A/B canonical sequences.
- **Verification pass:** render every mode against its corpus-predicted signature. CPU/GPU parity check (float32 vs float64 tolerance). Ensure the plan-document verification-standard checklist (see §Verification Standard below) is fully satisfied for each mode.

These three modes share the "corpus research first, then code" pattern, and the verification pass needs the final-state file to test against — hence combining them into one session.

---

## Mode 8 Question: Rotating vs. Cycling — Detailed Justification

The user asked: "honestly... would rotating or cycling look better? explain why it would look better."

**Cycling is the better choice, for substantive corpus reasons.**

### What "rotating" would mean
A continuous rotation `exp(iωn)` for some angular speed ω, applied to the contribution at every step. The orbit experiences a smoothly-varying phase shift each step. Visually: spiral arms in the connected-set complement, similar to many existing fractals that use rotation modulation.

### What "cycling" would mean
The 8 entries of the impedance table become the 8 possible per-step "magic regimes". At step n, `idx = n % 8` selects which regime is active, and the contribution computes `z^(12/d_idx) × ξ_idx`. The orbit literally moves through Pure Will → Mirror/Shadow → Elemental → Temporal → Sympathetic → Harmonic → Alien → Recursive → Pure Will → ... in sequence.

### Why cycling will look better

**1. The corpus impedance is discrete, not continuous.** Section 5.2 of ET_Fantastical_Configurations.md presents the 8 magic types as a **table** with discrete d values (1, 2, 3, 4, 5, 6, 7, 9). Each row is a qualitatively distinct magic regime. Rotating would smear them into a continuum that doesn't exist in the corpus.

**2. Cycling produces visible structure that maps to the table.** With cycling, the connected-set boundary develops 8 distinct "character regions" — each corresponding to a different magic type. You can literally see Pure Will lobes (smooth, unmediated, octave-only structure) next to Recursive/Fractal lobes (intricate self-similar nesting) next to Alien/Geometric lobes (7-fold partial-traversal artifacts). The image **carries the impedance table as visible content**.

**3. Cycling makes the iteration step `n` matter qualitatively.** The corpus emphasizes that Mode 8 is about **regime transitions** — the moment when the orbit moves from one magic type to another is itself meaningful (it corresponds to a "thinning of the descriptor set" per the corpus). Rotating has no transitions; it's a smooth function of n. Cycling has 8 sharp transitions per cycle, which is exactly the corpus structure.

**4. Cycling expresses ξ as visible variation.** Each magic type has a different coupling enhancement ξ (1.00× to 8.50×). When cycling, each regime contributes at a different amplitude — the resulting image will have **regions of different brightness** corresponding to the regions of different coupling. Pure Will regions will be subtle (ξ=1.00); Recursive/Fractal regions will be intense (ξ=8.50). This is the impedance physics made visible.

**5. Rotating would look like every other fractal.** Spirals are generic. They appear in dozens of fractal techniques. Cycling through 8 discrete regimes is **structurally novel** — it's the kind of pattern only a corpus-derived fractal can produce, because no other framework has the impedance table.

**6. Cycling closes the inspection warning naturally.** The unused `n` parameter has been the inspection issue for Mode 8. Cycling uses `n` directly via `n % 8` — the "completion" is the same operation as fixing the warning, not two separate operations.

### Recommendation
**Implement Mode 8 with cycling via `n % 8` indexed against the impedance table.** This is the path that respects the corpus, produces visible structure, fixes the unused `n` parameter, and is more exotic/interesting than the rotating alternative.

---

## Cross-Cutting Tasks (apply to multiple modes)

### CUDA kernel parity infrastructure

For each mode that gets a new Python implementation, the corresponding CUDA kernel block needs:

1. **New `__constant__` arrays** for any lookup tables (HEPTAGRAM_PHASES, IMPEDANCE_TABLE, DELTA_K_TABLE, etc.). These go at the top of `_ET_RAWKERNEL_SRC` before the kernel function.

2. **Updated kernel signature** if any new mode-specific scalars are needed (e.g., `float xi_max` for Mode 8 normalization).

3. **Updated launch site** in `iterate_strip_v2` to pass any new scalars and to set `use_modeN` flags correctly via `_active(N)`.

4. **`_make_f64_kernel`** auto-conversion handles the float→double translation. Any new constant arrays must use the f-suffix syntax `1.0f` so the regex strips them correctly for the f64 variant.

### The "No Mode" option implementation

In `_choose_modes()`:
```python
print('  │   N   — none    (no mode dispatch — pure base 24-family)     │')
...
if raw == 'n':
    chosen = []   # empty list signals "no mode"
    print(f'  → No mode')
    return chosen
```

In `build_mode()` — needs to handle empty mode_ids by returning a "neutral" mode:
```python
def build_no_mode(tower, rng):
    """Neutral mode: no per-mode boost, no extra(), centered Julia c."""
    w_r = np.array([FAM_ELG_FULL.get(d,12./d) for d in ALL_REAL], dtype=np.float64)
    w_c = np.array([FAM_ELG_FULL.get(d,12./d) for d in ALL_COMPLEX], dtype=np.float64)
    # Tower boosts only — no mode boost
    for i,d in enumerate(ALL_REAL):
        if d in tower['prim_r']:    w_r[i] *= 3.0
        if d in tower['ext_boost']: w_r[i] *= 2.0
    for i,d in enumerate(ALL_COMPLEX):
        if d in tower['prim_c']:    w_c[i] *= 3.0
        if d in tower['ext_boost']: w_c[i] *= 2.0
    # Neutral Julia c: the centroid of the canonical interesting region
    julia_c = complex(-0.75, 0.0)   # the period-2 bulb tip
    return dict(
        mode_id=-1, w_r=_norm_w(w_r), w_c=_norm_w(w_c),
        extra=None, julia_c=julia_c,
        hue_speed=rng.uniform(0.018, 0.070),
        pal_extra=0.0,
        name='None — pure base 24-family',
        p_eff=2.0,
        mode_extra_w=np.zeros(12, dtype=np.float32),
        delta_k=float(tower.get('delta_k', 0.0)),
    )
```

In `_resolve_run_params()`:
```python
if not SELECTED_MODES:    # empty list = "No mode" option chosen
    mode = build_no_mode(tower, rng)
    mode_id = -1
else:
    mode_id = SELECTED_MODES[0]
    mode = _blend_modes(SELECTED_MODES, tower, rng)
```

### Dynamic `_NO_EXTRA` replacement

Currently at lines 1281 and 1465: `_NO_EXTRA = {0, 3}` is a hardcoded set. After the Mode 3 full implementation (which gives Mode 3 a real extra()), this set becomes wrong. The dynamic replacement is:
```python
# In _blend_modes:
mew_bl = np.zeros(12, dtype=np.float32)
for mid, m in zip(mode_ids, modes):
    if m['extra'] is not None:
        mew_bl[mid] = 1.0 / N_

# In build_mode:
mew = np.zeros(12, dtype=np.float32)
if p['extra'] is not None:
    mew[mode_id] = 1.0
```

This is a free fix — works correctly regardless of which modes return None, automatically picks up Mode 3's new extra() once implemented.

---

## Verification Standard

For every mode that gets implemented:

1. **Math verification:** Hand-trace 3 iteration steps with a known seed and compare CPU and GPU output to 1e-6 tolerance.
2. **Visual verification:** Render at 2k preset with the same seed and tower, compare against gen 49 baseline visually. Document any differences.
3. **Inspection verification:** Run PyCharm inspection and confirm zero unused-parameter warnings on the modified lines.
4. **Subsumption check:** Apply the Subsumption Law — does the math subsume the corpus definition without remainder?
5. **Identification check:** Apply the Identification Principle — are P, D, T all identified in the implementation?

---

## Session 4 Completion Audit (file v50-7)

Session 4's scope was Mode 1 (Traverser Field, (7,1) torus knot) + Mode 6 (Septic Otherworld, heptagram) — combined. During the Session 4 work pass, the Mode 4 Multifold cycling extension also landed, since Mode 4's Session 1 free fix had only used z directly without cycling through the four canonical inter-tower Δk values. The plan document was not updated when v50-7 landed; this audit records the v50-7 state.

**What landed in v50-7:**

- **Mode 1 — Traverser Field (7,1) torus knot:** Full Python lambda + matching CUDA kernel block. Implements the d=7 carrier with two winding phases (G_REAL=7 longitudinal, G_IMAG=1 meridional), the Wilson loop holonomy `((7n) mod N) · 2π/N` (the U(1) gauge holonomy of gen-7 on ℤ/12ℤ per Semitone Cascade §22.3), the Scopaesthesia T-density envelope `ρ_T = K/(K + V·|z|²)` per T Paper §31.4, and the V²·z·cos(φ_meri) meridional knot-tube breathing anchor. All five lambda parameters used. Full corpus trace covering Semitone Cascade §22.1–§22.3 and T Paper §31. Three-tools justification inline.

- **Mode 4 — Multifold Tower cycling extension:** The four canonical Δk values from Multifold §12.2 (`{0, -996, -1279, -1675}`) are stored as `_DK_TABLE` in the Python lambda and as `__constant__ float DELTA_K_TABLE[4]` in the CUDA kernel. The orbit cycles through the four Multifold reference frames via `(home_idx + n) % 4`, where `home_idx` is the index of the user-selected tower's Δk in the table (the user's tower becomes the n=0 home frame). The Session 1 z-substitution (using z directly instead of `r·exp(iθ)`) is preserved.

- **Mode 6 — Septic Otherworld heptagram:** Full Python lambda + matching CUDA kernel block. Implements the 7-vertex heptagram superposition `(1/7)·Σ_{k=0..6} exp(i·(p·θ + kt·LN2/7 + 2π·k/7))`, the Asymptotic Approach Theorem veil-thinning factor `1 - exp(-n·V)` per Fantastical §8.3, the V²·z partial-traversal anchor for the d=3 "this side of the veil" reference, and the preserved 90° "Otherworld" `i` rotation from the prior code. The CUDA kernel adds a new `__constant__ float HEPTAGRAM_PHASES[7]` constant array storing the seven vertex angles `2π·k/7` for k=0..6. All five lambda parameters used. Full corpus trace covering Fantastical §8.1–§8.3.

- **f64 parity:** The `_make_f64_kernel` regex pipeline was already augmented in Session 2 to include `expf` (for Mode 3's Gaussian); Session 4's heptagram only uses `cosf`/`sinf`/`expf` which were already covered. No new intrinsic_map entries were needed.

**No code from prior sessions was removed.** The Mode 1 and Mode 6 prior stubs were replaced with the full implementations; the Mode 4 prior Session 1 form was extended (not replaced) with the cycling table.

---

## Session 5 Completion Audit (file v50-8)

Session 5's scope was Mode 5 (Quintic Shadow, Fibonacci d=3 attractor) + Mode 7 (Nonic Recursion, holographic depth) — combined. Both modes were implemented in `build_extra()` (Python lambda) and the corresponding CUDA kernel block, with full corpus traces and three-tools justifications inline.

**What landed in v50-8:**

- **Mode 5 — Quintic Shadow:** Full d=5 source + d=3 cubic attractor (the missing receiver) + d=10 binary×quintic composite implementation. Three carriers each with their own kt rotation (the d=10 carrier gained a `kt·LN2/10` rotation it lacked in the prior code). Fibonacci alternating sign `(-1)^n`, φ-decay envelope `(1/φ)^(n // N)` (the §9.2 corpus rate, NOT the plan's softer linear `1/(1+n·V)` envelope), explicit α₅ = 1/20 coupling per QS-5, V²·z·damping anchor on the d=3 receiver, and the (EPS5/1200) structural prefactor preserved from the prior code. See the "Mode 5 — Quintic Shadow" section above for the full implementation block, six structural decisions, smoke test results, and visual prediction.

- **Mode 7 — Nonic Recursion:** Full holographic-depth implementation. Variable recursion depth `active_depth = min(1 + n//N, 9)` driven by n, V²·z self-call anchor, recursive feed-forward of `(current_r, current_th)` per level, divide-by-`active_depth` normalisation. Maximum depth 9 = 3² is the corpus literal target. The prior 2-level structure is preserved as a special case in the n ∈ [12, 23] depth window. See the "Mode 7 — Nonic Recursion" section above for the full implementation block, five structural decisions, smoke test results, and visual prediction.

- **CUDA constants block addition:** `#define ALPHA5_F  0.05f` was added to `_ET_RAWKERNEL_SRC` next to `INV_A0_F`, with corpus comment derivation `α₅ = 1/(4·d) = 1/20 = (3/5)·V = (F₅/F₆)·V` per QS-5. The macro-strip regex `(#define\s+\w+\s+\d+\.\d+)f\b` in `_make_f64_kernel` correctly converts this to `#define ALPHA5_F  0.05` for the f64 build.

- **No kernel signature changes.** Mode 5 reuses the existing `eps5` and `inv_phi` launch scalars (both already wired in the launch site at line 4017). Mode 7 needs no new launch parameters. The kernel signature at line 2737 is byte-for-byte identical to v50-7.

- **f64 parity verification (smoke-tested in Session 5):**
  - `ALPHA5_F` define correctly strips the `f` suffix to `0.05`.
  - Mode 5 block: 5 `powf` + 3 `cosf` + 3 `sinf` calls all convert correctly to `pow`/`cos`/`sin`.
  - Mode 7 block: 1 `sqrtf` + 1 `atan2f` + 2 `powf` + 1 `cosf` + 1 `sinf` calls all convert correctly.
  - The `1e-38f` floor in Mode 7 strips to `1e-38` then converts to `1e-300` via the underflow-guard step.
  - Zero stray `float` tokens remain in the f64 kernel; 242 `double` tokens after conversion.

- **Numerical smoke test (CPU, scalar at z = 0.5+0.3i, kt = 5.0):**
  - **Mode 5:** outputs at n=0..3 vary smoothly with sign alternation; magnitudes at n=12, 24, 36 follow the φ-decay ratio (4.4e-3 → 2.3e-3 → 1.4e-3 → 7.8e-4, ratio ≈ 1/φ per N=12 step as expected); ndarray (4×4) input produces no NaN/inf.
  - **Mode 7:** depth grows correctly (1 at n=0..11, 2 at n=12..23, 3 at n=24..35, ..., 9 at n≥96); n=96, 200, 500 all produce identical output confirming the depth saturation cap; ndarray (4×4) input produces no NaN/inf.
  - **ET identity verification:** `(3/5)·V == α₅` (both = 0.05, exact match per QS-5); `MAX_NONIC_DEPTH = 9 = 3²` exact; `EPS5 ≈ −13.686¢` negative as required by QS-15; the `_qs_pre = (EPS5/1200)·α₅ ≈ −5.703e-04` combined prefactor.

- **`ast.parse()` of the full v50-8 file passes** — no Python syntax errors anywhere.

- **Six structural decisions for Mode 5** and **five structural decisions for Mode 7** are documented in their respective plan sections above. The most significant departures from the original plan's pseudocode:
  - Mode 5 uses `(1/φ)^(n//N)` damping per §9.2 instead of the plan's `1/(1 + n·V)` linear damping. The corpus is explicit that the rate is φ.
  - Mode 5's d=10 carrier gains its own `kt·LN2/10` rotation (the plan and prior code both omitted this).
  - Mode 7 uses `np.angle(accum)` instead of the plan's broken `math.atan2(accum.imag if hasattr(accum, 'imag') else 0, ...)` — `math.atan2` does not vectorise, but `np.angle` does, so the plan's pseudocode would have failed at runtime on the CPU path.

**No code from prior sessions was removed.** The Mode 5 and Mode 7 prior stubs were replaced with the full implementations; nothing else was touched.

---

## Session 6 Completion Audit (file v50-9)

Session 6's scope was Mode 9 (Exception State, V(E)=0 grounding pull) — single mode. The original Session 6 plan combined Modes 9, 10, and 11 plus the verification pass into one session. At the start of the Session 6 work pass the user directed that the combined session be split into three: Session 6 = Mode 9 only, Session 7 = Mode 10 only, Session 8 = Mode 11 + the full verification pass. The reason matches the rationale for the previous restructurings — each mode requires its own full corpus research pass and its own CPU + CUDA implementation pass, and three modes plus a verification pass in one session inflates the per-session scope past what can be done honestly without truncation or shortcuts.

**What landed in v50-9:**

- **Mode 9 — Exception State (V(E)=0 grounding pull):** Full Python lambda + matching CUDA kernel block. Six structural Descriptors derived from the corpus: explicit `z_E = 0+0j` grounding fixed point (the canonical lattice origin at ε=0 / k=0 / d=1), the §23 elegance-form variance kernel `V_loc(z) = 1/(1 + V·|z-z_E|²)` (smooth ET-native analog of `100/(100+|ε|)` — preserved from the prior stub but now named structurally), the Koide cosmological grounding weight `G(z) = K · V_loc(z)` (per `M-states.md` line 459, the Pure E-vacuum is exactly 2/3 = K of total energy), the Observational Displacement residual `exp(-n · V)` (per `ExceptionTheory.md` line 786: "any observation creates a new configuration with positive variance" — each step n is one observation, the residual decays at rate V = 1/12), the manifold-cycle phasor `exp(i · (2π·n/N + th + kt · (LN2/N) · K))` with Koide-weighted kt rotation (matching the cosmological E-vacuum weight on the T-axis), and the M-vacuum / M-matter 8/15 + 7/15 split per `M-states.md` line 950 (the corpus 8:7 ratio between distributed vacuum-like M-states and localized matter-like M-states). All five lambda parameters used (z directly via M-matter; r via V_loc; th via cycle_ang; kt via Koide-weighted T-axis term; n via both Observational Displacement residual AND cumulative manifold rotation). Full corpus trace covering `ExceptionTheory.md` Part XI / §Observational Displacement / §Variance Dependence (line 2347) / `ET_Incoherence_Paper.md` §10 (line 385+) / §23 (line 1049+) / `M-states.md` lines 459/712/745/770/950. Three-tools justification block inline (Identification: P=z, D=three structural Descriptors, T=per-step Observational Displacement; Descriptor Gap: explicit z_E reference, Koide cosmological weight, M-vacuum/M-matter split, Observational Displacement, Koide-weighted manifold-cycle phasor; Subsumption: every corpus component named appears as an explicit term, the Lorentzian and z-pull from the prior code are preserved as V_loc and M_matter respectively, nothing is removed).

- **Kernel-signature comment fix (unrelated correctness fix found during Session 6 audit):** The kernel signature line preceding `float mu2, float lam_mh` was commented `// mode 9 extra (Lagrangian): mu2, lambda` — but `mu2` and `lam_mh` are actually consumed by the **Mode 10** kernel block (the Mexican-hat Lagrangian field gradient), not Mode 9. The mislabel survived every prior session because nothing else read the comment. Fixed to `// mode 10 extra (Lagrangian Field, Mexican-hat): mu2, lambda`. This is a documentation fix only — no code path is affected, but the comment now matches the actual consumer.

**No new launch scalars, no new `__constant__` arrays, no kernel signature change.** Mode 9 reuses only the existing kernel macros `K_F`, `V_F`, `N_F`, `LN2_F`, `PI_F`. The kernel signature (modulo the one-line comment fix above) is byte-for-byte identical to v50-8.

**f64 parity verification (smoke-tested in Session 6):**
- Mode 9 CUDA block: 18 `float` tokens, all 18 convert correctly to `double` under `_make_f64_kernel`.
- 1 `cosf` → `cos`, 1 `sinf` → `sin`, 1 `expf` → `exp` — all already in `intrinsic_map` from prior sessions, no `intrinsic_map` additions needed.
- The new literals `8.0f / 15.0f`, `7.0f / 15.0f`, `2.0f * PI_F / N_F`, `(LN2_F / N_F) * K_F` all strip their `f` suffixes correctly via the existing `(\d+\.?\d*(?:[eE][+-]?\d+)?)f\b` regex.
- Zero stray `float` tokens remain in the f64 kernel after conversion; 255 `float` → 255 `double` across the full kernel (Mode 9's 18 tokens plus the 237 tokens that already converted in prior sessions).

**CPU/GPU parity verification (Session 6 smoke test):** A Python transcription of the CUDA block was run side-by-side against the Python lambda across five test cases — `z = 0+0j` at the literal grounding point, `z = 0.5+0.3i` at n=7, `z = -0.7+0.2i` at n=12, `z = 1+1i` at n=24, and `z = 0.1-0.4i` at n=100. **Maximum error across all five tests: 2.17e-19** — pure float64 round-off, no algorithmic divergence. The Python lambda and CUDA block are bit-equivalent within float round-off, exactly matching the Sessions 4/5 parity standard.

**Numerical smoke test (CPU lambda):**
- **At z = z_E = 0+0j** (the literal Exception grounding fixed point): output magnitude is exactly `V² · K · (8/15) ≈ 2.469e-3`, **constant in n** for all tested n ∈ {0, 1, 12, 24, 96, 200, 1000}. This is the corpus-correct behavior: at z_E the M-matter term `-(7/15) · z · displacement` vanishes (because z=0 cancels it), leaving only the V²·K-weighted M-vacuum component. The Exception's literal zero-variance peak is preserved exactly — V_loc(z=0) = 1.0, G(z=0) = K = 2/3, and the magnitude of the contribution is `V² · K · (8/15)` regardless of how many "observations" n have occurred (because at the Exception there is no z to displace).
- **At n = 12** (one full N=12 manifold cycle): the Observational Displacement residual is `exp(-12 · V) = exp(-1) ≈ 0.36788`, exactly the corpus-derived rate (one full manifold cycle = one Exception displacement event).
- **All five lambda parameters affect the output:** z-diff 1.17e-4, r-diff 3.54e-5, th-diff 1.43e-4, kt-diff 9.25e-5, n-diff 1.29e-3 — all strictly positive, confirming every parameter participates in the math.
- **ndarray (4×4) input:** shape preserved, no NaN/inf, |min| = 2.87e-3, |max| = 3.36e-3.
- **ET identity verification:**
  - `K = 2/3` (Koide ratio = M-states.md cosmological E-vacuum weight) ✓
  - `V = 1/12` (base variance) ✓
  - `V² = 1/144` (overall Subsumption-friendly scale) ✓
  - `V_loc(z=0) = 1.0` (Exception zero-variance peak) ✓
  - `G(z=0) = K · 1.0 = 2/3` (full Koide grounding weight at the Exception) ✓
  - `8/15 + 7/15 = 1.0` exactly (M-state split is conservative) ✓
  - `2π/N = π/6` (manifold-cycle angular quantum) ✓
  - `(LN2/N) · K ≈ 0.03851` (Koide-weighted kt rotation) ✓
  - `exp(-N·V) = exp(-1) ≈ 0.36788` (one full manifold cycle = one Observational Displacement) ✓
  - `(8/15)/(7/15) = 8/7 ≈ 1.143` matches `1.6%/1.4% = 1.143` from M-states.md line 950 (the corpus 8:7 split between vacuum-like and matter-like M-states) ✓

**Six structural decisions made in Session 6** (resolving ambiguities in the corpus and the prior stub):

1. **`z_E = 0+0j` is the canonical Exception grounding fixed point.** The corpus does not give an explicit complex-plane location for the Exception (the Exception is a configuration in 𝒞, not a point in ℂ), so the implementation must choose a fractal-space anchor. The lattice origin (ε=0, k=0, d=1 — the trivial sublattice point at unison) is the canonical choice: it is where V_loc reaches its peak value 1.0 (the literal zero-variance maximum), it is where the manifold projection cleanly degenerates (no log of zero, no atan2 ambiguity), and it is consistent with how the other modes anchor their structural reference points (Mode 3's `z_K = K + i·V` is at the Koide ∂I point; Mode 9's `z_E = 0` is at the unison/origin point). This is the simplest non-arbitrary choice and is justified structurally.

2. **`G(z) = K · V_loc(z)` is the Koide cosmological grounding weight, not just `V_loc`.** The prior stub used `va = 1/(1+r²·V)` as the entire scaling — implicitly weighting by 1.0 with no Koide reference. Per `M-states.md` line 459 and line 723, the Pure E-vacuum is exactly 2/3 = K of total cosmological energy. This is not a coincidence — the Koide ratio IS the cosmological weight of the static Exception. The implementation makes this explicit by multiplying V_loc by K, so G(z) at the grounding fixed point equals K (= 2/3) instead of an arbitrary 1.0.

3. **The Observational Displacement uses `exp(-n · V)`, not a power-of-V damping or a linear-in-n decay.** The exponential form is the canonical "observation makes the original Exception fade" dynamic from `ExceptionTheory.md` line 786, with the rate set by V = 1/12 (the lattice-natural variance quantum). After one full N=12 manifold cycle the residual is exp(-1) ≈ 0.368 — a clean ET identity. A power-of-V form (e.g., `V^(n/N)`) would decay too fast (V at n=N already, V² at n=2N); a linear form (`1/(1+n·V)`) would decay too slowly. The exponential matches the ET-native exp(-n·V) form used by Mode 6's Asymptotic Approach Theorem (`1 - exp(-n·V)`) — same lattice quantum, different sign of integrand.

4. **The manifold-cycle phasor uses `2π·n/N + th + kt·(LN2/N)·K`, not the prior stub's `th·(N/12) + kt·(LN2/N)`.** The prior stub had `N/12 = 1` so the `th · (N/12)` factor reduced to bare `th` — losing the cumulative manifold rotation `2π·n/N`. The new form adds the n-cumulative rotation explicitly (uses n) and applies the Koide weight K to the kt term (matches the cosmological E-vacuum weight). At n = 0 the phasor angle equals `th + kt·(LN2/N)·K` (close to the prior stub's behavior); as n grows the phasor cumulatively rotates by the manifold quantum 2π/N per step, expressing temporal evolution at the lattice rate.

5. **The M-vacuum / M-matter split uses `8/15 + 7/15 = 1`, exactly per `M-states.md` line 950.** The corpus is unambiguous that the active mediation (3% total cosmological energy) splits 8:7 between vacuum-like M-states (1.6%, distributed uniformly, w ≈ -1, "act like cosmological constant") and matter-like M-states (1.4%, localized in matter, w ≈ 0, "concentrated where complexity exists"). The implementation expresses this directly: the M-vacuum component is the manifold-cycle phasor at the 8/15 weight (does NOT depend on z — distributed uniformly), and the M-matter component is the negative z-pull at the 7/15 weight (depends on z — localized at the orbit's current position). The negative sign on M-matter is the corpus "pull TOWARD the grounding fixed point" direction — M-matter is the localized component that drags z back toward the Exception. The displacement residual multiplies M-matter only (not M-vacuum) because matter-like M-states are themselves the things being observed and displaced, while vacuum-like M-states are the distributed substrate that does not localize for observation.

6. **The V² overall scale is preserved as the Subsumption-friendly perturbation magnitude.** Like all the other modes' V²-scaled extras, Mode 9's contribution is bounded by the V² ≈ 6.94e-3 ceiling so it does not override the 24-family base sum. Combined with the K factor inside G, the overall extra magnitude at the grounding fixed point is `V² · K · (8/15) ≈ 2.469e-3` — comfortably smaller than the typical base-sum magnitude per family weight (~1e-2 to 1e-1), and on the same order as the other modes' z-anchor contributions (Subsumption-friendly).

**`ast.parse()` of the full v50-9 file passes** — no Python syntax errors anywhere.

**File integrity verified:** Full diff against v50-7 shows exactly three changes — (a) Mode 9 Python lambda replacement (lines 2171–2175 → ~240 lines of corpus trace + lambda), (b) Mode 9 CUDA kernel block replacement (lines 3535–3543 → ~80 lines of corpus comment + CUDA block), (c) the one-line kernel-signature comment fix at line 2998. Net +316 lines, all additive (no removals). All 12 Python `mode_id` branches present `[0..11]`. All 11 CUDA `use_modeN` blocks present `[1..11]`. The CPU `extra(z, r_cap, theta, k_t, n)` call site at line 4283 is preserved. The launch site `_gs(mu2), _gs(lam_mh)` wiring is unchanged. Mode 9's `build_mode()` weight boost (`d ∈ [12, 1, 2]` with 5× multiplier) is preserved. Mode 9's `_et_julia_c()` is preserved. All 12 `_MODE_NAMES` are preserved. The f64 kernel still builds cleanly (50,037 chars, zero `float` tokens remaining after conversion).

**No code from prior sessions was removed.** The Mode 9 prior stub was replaced with the full implementation; the kernel-signature mu2/lam_mh comment was corrected (no code change, only comment text); nothing else was touched.

---

## Session 7 Completion Audit (file v50-10)

**Scope:** Mode 10 (Lagrangian Field — Mexican-hat vacuum + Higgs + Goldstone).

**Corpus trace (what was read before any code was written):**

- `/mnt/project/ET_Lagrangian_Field_Theory.md` — full file, all 836 lines, no truncation. §VIII (line 562+) is the canonical Mexican-hat derivation.
- `/mnt/project/ET_Three_Tools_Complete_Reference.md` — full file, all 739 lines, no truncation. Used to apply the Identification / Descriptor Gap / Subsumption analysis to Mode 10.
- `/mnt/project/et_clr_v5__4_.py` THEOREM LFT-8 (lines 4742-4760) — second independent corpus source. Reproduces §VIII word-for-word. Same potential, same `v = √(μ²/2λ)`, same `m_H = √(2μ²)`.
- Full read of `ET_FRACTAL_GENERATOR50-9.py` (all 6611 lines, no truncation) to inventory every Mode 10 reference site across the file. All 9 sites located:
  1. Line 138 (UI menu label)
  2. Lines 578–580 (module-level constants `_MH_MU2`, `_MH_LAMBDA`, `_MH_V`)
  3. Line 1244 (`_MODE_NAMES[10]`)
  4. Lines 2414–2419 (`build_extra(mode_id == 10)` — the broken Python stub)
  5. Lines 2573–2578 (`_et_julia_c(mode_id == 10)` — anchor on `_MH_V`, preserved)
  6. Lines 2644–2648 (`build_mode` weight boost d ∈ [6, 12, 4] × 8, preserved)
  7. Lines 2998–2999 (CUDA kernel signature scalars `mu2`, `lam_mh`, preserved)
  8. Lines 3622–3631 (`if (use_mode10)` — the broken CUDA stub)
  9. Lines 4242–4243 + 4335 (CPU launch site passing `_gs(mu2), _gs(lam_mh)`, preserved)

Zero downstream coupling in audio/render/video paths. The ∂I kernel and CPU ∂I path do NOT call `extra()` — Mode 10 only affects the standard ET kernel and standard CPU iteration path.

**Corpus convention verification (done before any code was written):** exhaustive grep across every `.md`/`.py`/`.txt` file in the corpus for `Mexican`, `Higgs`, `vacuum manifold`, `Goldstone`, `μ²`, `vacuum expect`, `VEV`, `spontaneous symmetry breaking`, `v = sqrt`, `1/2.*μ²`, `½μ²`, `(1/4).*phi.*⁴`, `0.5*mu2`, `0.5*lam`. **Zero hits for any half-coefficient form.** Two independent sources (`ET_Lagrangian_Field_Theory.md §VIII` and `et_clr_v5__4_.py LFT-8`) state the same potential `V(φ) = −μ²|φ|² + λ|φ|⁴`, the same vacuum `v = √(μ²/2λ)`, and the same Higgs mass `m_H = √(2μ²)`. No competing convention exists.

**Gradient bug audit finding:** the prior stub had `grad = z·(λ·r² − μ²)` — missing a factor of 2 on the quartic term. Wirtinger differentiation of the canonical corpus potential gives:

> `∂V/∂φ* = −μ²·φ + 2λ(φ*φ)·φ = φ·(2λ|φ|² − μ²)`

The factor of 2 comes from `d/dφ*` of `|φ|⁴ = (φ*φ)²`. The buggy form zeroed at `r = 2√2 ≈ 2.828` — 41% larger than the documented `v = 2`. Worse, the prior code's `_et_julia_c(mode_id == 10)` at line 2577 already referenced `_MH_V = 2` directly to anchor the Julia c on the vacuum ring, so the prior code's iteration dynamics and Julia c anchor were in disagreement about where the vacuum is. The Session 7 fix brings them into corpus-consistent agreement.

**Internal arithmetic consistency check (verified before writing the new code):**

- `f(x) = -μ²·x² + λ·x⁴` (potential as f(|φ|))
- `f'(x) = -2μ²·x + 4λ·x³ = 2x·(2λ·x² − μ²)`
- `f'(v) = 0` when `2λ·v² = μ²` → `v² = μ²/(2λ)` ✓
- `f''(x) = -2μ² + 12λ·x²`
- `f''(v) = -2μ² + 12λ·(μ²/(2λ)) = -2μ² + 6μ² = 4μ²`
- `m_H² = ½·f''(v) = 2μ²` (½ from `L = ½(∂σ)² − ½m²σ²`)
- `m_H = √(2μ²)` ✓ matches `§VIII.3 line 651`

**Implementation — what was changed:**

1. **Python lambda at lines 2414–2419** (now lines 2414–2824 in v50-10): replaced the six-line stub with a ~410-line corpus-faithful implementation. New lambda body has three corpus-named contributions:
   - **Gradient flow** `flow = -η·grad·choice` where `grad = z·(2λr² − μ²)` (corrected Wirtinger gradient) and `choice = 1 − exp(-n·V)` is T's vacuum-substantiation envelope.
   - **Higgs (radial massive) mode** `higgs = V²·σ·cos(m_H·n·V)·(z/|z|)·choice` where `σ = r − v`, `m_H = √(2μ²) = 2/√3 ≈ 1.1547`, period ≈ 65.3 iteration steps.
   - **Goldstone (angular massless) mode** `goldstone = V²·i·(z/|z|)·cos(2π·n/N + th + kt·LN2/N)·choice`. Direction is the explicit vacuum-ring tangent `i·(z/|z|)`. Phase mirrors Mode 9's `cycle_ang` form.

2. **CUDA kernel block at lines 3622–3631** (now lines 4031–4175 in v50-10): replaced the nine-line stub with a ~145-line block mirroring the Python lambda exactly. Uses existing kernel scalars `mu2`/`lam_mh` (no signature change), computes `v` and `m_H` inline via `sqrtf(mu2/(2.0f*lam_mh))` and `sqrtf(2.0f*mu2)`, uses the existing `K_F`/`V_F`/`N_F`/`LN2_F`/`PI_F` macros, and uses intrinsics `expf`/`cosf`/`sqrtf`/`fmaxf` that are all in `_make_f64_kernel`'s intrinsic-map for the f64 build. The `1e-38f` underflow guard auto-converts to `1e-300` in the f64 build per the existing substitution rule. Complex multiplication `i·(z/|z|) = (-rdir_i, +rdir_r)` unrolled inline (matches the style used by Modes 4 and 6).

3. **Nothing else touched.** The kernel signature is unchanged. The launch site `_gs(mu2), _gs(lam_mh)` is unchanged. The module-level constants `_MH_MU2`, `_MH_LAMBDA`, `_MH_V` at lines 578–580 are unchanged. The `_et_julia_c(mode_id == 10)` anchor is unchanged (still uses `_MH_V`, now consistent with the corrected iteration dynamics). The `build_mode()` Mode-10 weight boost `d ∈ [6, 12, 4] × 8` is unchanged. `_MODE_NAMES[10] = 'Lagrangian Field'` is unchanged.

**Structural decisions made in Session 7** (resolving ambiguities in the corpus and the prior stub):

1. **The factor-of-2 fix is mandatory, not optional.** The prior gradient `λr² − μ²` is a Wirtinger derivative with a dropped factor. The corpus potential `V = −μ²|φ|² + λ|φ|⁴` (Lagrangian §VIII.1 line 569) is the canonical form shared across TWO independent corpus sources; no half-coefficient form exists anywhere. With the constants `μ² = K = 2/3` and `λ = V = 1/12`, the corpus vacuum is `v = √(K/(2V)) = 2` exactly, which is what the Julia c anchor already uses. Using the uncorrected gradient would leave the iteration dynamics pointing at a different vacuum (`2√2 ≈ 2.828`) than the Julia c anchor (`2.0`) — a self-contradiction in the code. The fix brings them into agreement.

2. **The vacuum substantiation envelope is `1 − exp(−n·V)`, matching Mode 6's Asymptotic Approach saturator.** Per Lagrangian §VIII.2 line 593-596, T's `[0/0]` indeterminacy must pick ONE vacuum direction — but in iteration time, that choice is dynamic. At n=0 T has not yet committed; as n grows, T's commitment grows asymptotically toward 1 but never reaches it in finite n. The canonical ET form for this "T progressively committing over iteration time" pattern is the `1 − exp(−n·V)` saturator that Mode 6 already uses for veil-thinning — same lattice-natural rate V = 1/12, same asymptotic form. Using a power-of-V form would decay/grow too fast, a linear form too slowly. The exponential form is both mathematically canonical (it's the solution of `dy/dt = V·(1-y)` at unit rate) and ET-corpus-canonical (Mode 6 Asymptotic Approach Theorem).

3. **The Higgs oscillator frequency is `m_H` per unit lattice time, with lattice time `t = n·V`.** The ET-derived lattice time is `V = 1/12` per iteration step (the base variance quantum is both the spatial and temporal smallest step). So the Higgs oscillator factor is `cos(m_H · n · V)`. With `m_H = 2/√3`, the period is `2π/(m_H·V) = 12π·√3 ≈ 65.3` iteration steps. This is much slower than the Goldstone period (`N = 12` steps) — exactly as expected: the Higgs is the heavier massive radial mode, the Goldstone is the lighter massless angular mode. The physical hierarchy `ω_H > ω_Goldstone` is preserved correctly in iteration time.

4. **The Goldstone direction is the explicit tangent `i·(z/|z|)`, not a free phase factor.** The prior code's `gs = exp(i·(θV + kt·LN2/N))` was a phase factor with no relation to the orbit's actual position on the vacuum ring — it rotated the same complex-plane direction regardless of where z currently was. The corpus-correct form is the vacuum-ring tangent vector: at any point z on the vacuum manifold, the Goldstone direction is the unit tangent `i·(z/|z|)` (perpendicular to the radial direction). This anchors the Goldstone mode to the orbit's actual angular position on the vacuum ring, matching Lagrangian §VIII.2 line 601-608 which identifies the Goldstone as the tangent to the vacuum manifold at the chosen vacuum direction.

5. **The Goldstone phase uses `2π·n/N + th + kt·LN2/N`, mirroring Mode 9's `cycle_ang` form exactly.** The three contributions are corpus-named:
   - `2π·n/N` is the cumulative manifold rotation per step — the manifold-cycle angular quantum that Mode 9 already uses (one full revolution per N=12 steps).
   - `th` is the orbit's instantaneous angle, which IS the Goldstone field's value per §VIII.2 line 596 (`φ₀ = v·e^{iθ₀}`). In the iteration picture, the orbit's current angle on or near the vacuum ring is the Goldstone position.
   - `kt·LN2/N` is the T-axis lattice phase contribution, preserved exactly from the prior code's `gs = exp(i·(…))` factor.

   The form matches Mode 9 exactly for two reasons: (a) both modes are grounded in the same manifold-cycle angular quantum, and (b) using the same form means Mode 9 and Mode 10 blend coherently when the user selects both modes together (`_blend_modes()` sums their extras).

6. **The choice of `cos()` (real) over `exp(i·…)` (complex) for the Goldstone amplitude is structural.** The Goldstone mode multiplies a fixed direction (the tangent `i·(z/|z|)`) by an amplitude. Using a real `cos(phase)` keeps the Goldstone contribution bounded (magnitude ≤ `V²·|radial_dir| = V²`) and keeps the amplitude strictly oscillatory. Using `exp(i·phase)` would make the whole contribution rotate in the complex plane, which is physically the Goldstone rotating through different real angular positions — but since the direction is already `i·(z/|z|)` (not a free direction), adding another rotation would double-count the angular motion. The real `cos` form gives the corpus-correct "bounded oscillation about the chosen vacuum direction" picture.

7. **Vacuum-substantiation envelope gates all three modes multiplicatively, not additively.** Flow, Higgs, and Goldstone all multiply by `choice(n)`. This is the corpus interpretation of §VIII.2 line 593-596: at n=0 T has not substantiated any vacuum, so the symmetry is fully unbroken and NONE of the three modes exist (Higgs and Goldstone are fluctuations about a chosen vacuum that hasn't been chosen yet; the gradient flow is the motion toward that chosen vacuum, which also doesn't yet exist). As n grows, all three modes grow together — they're born simultaneously when T starts committing. An additive envelope (e.g., `flow + higgs·choice + goldstone·choice`) would incorrectly have the gradient flow alive at n=0 while the other two are dormant — a physically incoherent state.

8. **The `_v` captured-but-unused gap is closed via explicit `σ = r − v` in the Higgs mode.** The prior code had `_v = _MH_V` in the lambda's default args but never referenced it anywhere in the body. The new Higgs mode uses `_v` directly via `sigma = r - _v`, closing the gap and connecting the lambda to the documented vacuum location in the code.

9. **The `n` parameter is used three times independently.** The prior code ignored `n` entirely (static field, no T-time dynamics). The new code uses `n` in three places — (a) the vacuum substantiation envelope `exp(-n·V)`, (b) the Higgs oscillator `cos(m_H·n·V)`, and (c) the Goldstone cumulative manifold rotation `2π·n/N`. Each use is independently required by a different part of the corpus: (a) §VIII.2 line 593-596, (b) §VIII.3 line 651, (c) §VIII.2 line 601-608 via Mode 9's manifold-cycle pattern.

10. **`th` is used explicitly in `gs_phase`, not just implicitly via `z/|z|`.** The orbit's angle appears both implicitly (as the direction of `radial_dir = z/|z| = e^{iθ}`) AND explicitly (as an additive term in `gs_phase`). The explicit use mirrors Mode 9's `cycle_ang = _2pn·n + th + kt·_kta` exactly, treating the orbit angle as the Goldstone field's value per §VIII.2 line 596. All five lambda parameters (`z`, `r`, `th`, `kt`, `n`) are thereby consumed as distinct observers of the orbit state.

**End-to-end smoke test (CPU NumPy path, run against the Python lambda imported directly from v50-10):**

- **AST parse of the full v50-10 file passes** — no Python syntax errors anywhere. File grew 6610 → 7171 lines (+561 lines, all additive).
- **`build_extra(10, tower)` returns a callable lambda** matching the standard `(z, r, th, kt, n)` signature used by every other mode.
- **n=0 invariant:** `extra(z, r, th, kt, 0)` returns exactly `0+0j` at every orbit point — T uncommitted, full U(1) symmetry intact, zero Mode-10 contribution. ✓
- **Vacuum invariant:** at `z = v = 2` with `n = 100`, only the Goldstone tangent component remains. Gradient is zero because `2λ·r² = μ²` exactly; Higgs is zero because `σ = r − v = 0`; only `goldstone = V²·i·(z/|z|)·cos(2π·n/N + th + kt·LN2/N)·choice` is nonzero. Computed value `(0, −3.4714e−3)` matches the expected closed-form Goldstone-only value to ~1e-17 (machine precision). ✓
- **All five lambda parameters produce independent non-zero sensitivities:** z-diff 1.16e-3, r-diff 8.43e-5, th-diff 2.79e-6, kt-diff 8.71e-6, n-diff 2.09e-3 — all strictly positive, confirming every parameter is a real consumer of the math. ✓
- **Envelope growth across iteration time:** at n=0 envelope=0 (exact), n=1 envelope≈0.083, n=6 envelope≈0.393, n=12 envelope≈0.632, n=50 envelope≈0.985, n=100 envelope≈0.9998, n=1000 envelope=1.0 (saturated). Contribution magnitudes grow monotonically with `n` from zero and asymptote to a fixed-point scale — exactly the corpus picture of T progressively committing to one vacuum. ✓
- **Magnitude scale at n=50 (Subsumption check):** |mean|=2.628e-2, |p95|=~4.2e-2, |max|=~5.7e-1 for the extreme-z cases but typical p95 <<1. All well below 1.0, the typical 24-family base-sum scale — the Mode 10 contribution perturbs without overriding (Subsumption Law). ✓
- **z=0 (unstable maximum) defensive behavior:** no NaN, no inf. The `fmaxf(rc, 1e-38f)` underflow guard on the radial divisor prevents 0/0; the gradient and Higgs terms vanish naturally because they carry explicit `z` factors; the Goldstone term evaluates to zero because `z/|z| = 0/1e-300 = 0+0j`. ✓
- **`build_mode(10)` produces correct mode_params shape:** `mode_id=10`, `extra` is the corpus-faithful lambda, `mode_extra_w[10]=1.0`, `name='Lagrangian Field'`, Julia c on the vacuum-ring-derived anchor at `|c|≈0.292` (the documented `K²/v` scaling). ✓

**CPU ↔ CUDA parity verification (float32 simulation of the kernel arithmetic against the Python f64 reference):**

Tested on 5 representative `(z, n)` cases spanning near-origin, on-vacuum-ring, below-vacuum, above-vacuum, and large-n regimes:

| z | n | |diff| (f32 vs f64) |
|---|---|---|
| 1+0.5j | 5 | 3.75e-9 |
| 2+0j | 50 | 2.48e-9 |
| 0.5-0.3j | 12 | 2.88e-9 |
| 1.5+1.5j | 100 | 1.67e-8 |
| 0.01+0.02j | 1 | 1.16e-10 |

Max discrepancy 1.67e-8 — well within single-precision round-off (float32 machine epsilon ≈ 1.19e-7). The CUDA kernel block will produce the same results as the Python lambda to float32 precision on GPU, and to float64 precision on GPU when the f64 kernel is compiled.

**f64 kernel conversion verification:**

- f32 kernel source: 59,715 chars, 1,101 lines
- f64 kernel source (after `_make_f64_kernel` regex): 59,719 chars
- **Zero stray `float` tokens remaining after conversion** — all `float` declarations in the new Mode 10 block were converted to `double` by the `\bfloat\b → double` substitution.
- **Zero stray `f`-suffix numeric literals remaining** — all `1.0f`, `2.0f`, `1e-38f` literals were stripped by the `(\d+\.?\d*...)f\b → \1` substitution.
- **Zero stray f-suffixed intrinsics remaining** — `expf`, `cosf`, `sqrtf`, `fmaxf` all converted to their double-precision equivalents `exp`, `cos`, `sqrt`, `fmax` by the intrinsic_map.
- Mode 10 block in the f64 kernel source contains 32 `double` declarations, 2 `sqrt(` calls, 2 `exp(` occurrences (1 in actual code, 1 in a comment), 3 `cos(` calls, 1 `fmax(` call — all correct.
- The `1e-38f → 1e-300` underflow guard substitution applied correctly.
- The corrected gradient `2.0 * lam_mh * _rc2 - mu2` is preserved through the f64 conversion.

**Structural verification (file integrity):**

- All 12 Python `mode_id` branches present in both `build_extra()` and `_et_julia_c()` — 24 total occurrences, count of 2 for every mode_id in `[0..11]`. ✓
- All 11 CUDA `use_modeN` blocks present `[1..11]` (mode 0 has no kernel block by design, it's the bare PDT manifold). ✓
- CUDA Mode 10 gradient has corrected `2.0f * lam_mh` factor. ✓
- Python Mode 10 gradient has corrected `2.0 * _la` factor. ✓
- Python lambda body references all 5 parameters (`z`, `r`, `th`, `kt`, `n`) as real consumers. ✓
- CUDA kernel signature `float mu2, float lam_mh` preserved. ✓
- Kernel launch site `_gs(mu2), _gs(lam_mh)` preserved. ✓
- Mode 10 weight boost `d ∈ [6, 12, 4] × 8` preserved. ✓
- Mode 10 Julia c anchor (uses `_MH_V`) preserved. ✓
- All 12 `_MODE_NAMES` present, `_MODE_NAMES[10] = 'Lagrangian Field'`. ✓
- Constants block (`_MH_MU2 = K`, `_MH_LAMBDA = V`, `_MH_V = √(K/(2V))`) at lines 578–580 untouched. ✓
- Prior buggy gradient strings `grad=z*(_la*r2-_mu)` (Python) and `lam_mh * r2 - mu2;` (CUDA) fully removed. ✓
- Prior `gs=np.exp(1j*(th*V+kt*LN2/N))` phase factor subsumed into the new Goldstone form (both `kt·LN2/N` and the V-scale preserved, now with explicit tangent direction and cumulative manifold rotation). ✓

**No code from prior sessions was removed.** The Mode 10 prior stub was replaced with the full implementation. The constants, the kernel signature, the launch site, the Julia c anchor, the weight boost, and the `_MODE_NAMES` entry are all preserved byte-for-byte. Every prior code term has an explicit correspondent in the new implementation (either preserved exactly or corrected per the gradient-bug audit finding). Net +561 lines, all additive.

---

## Closing

The 12 modes have been audited against the ET corpus. The current findings (post-Session-7):

- **Modes 0, 2** are corpus-complete as originally designed (Mode 0 = bare PDT, Mode 2 = pure palindrome cascade).
- **Mode 1** (Session 4, v50-7) — full (7,1) torus knot with Wilson holonomy and T-density envelope.
- **Mode 3** (Session 2, v50-6) — full Koide Boundary Gaussian binding force.
- **Mode 4** (Sessions 1 + 4, v50-6 + v50-7) — z-substitution + multi-tower cycling.
- **Mode 5** (Session 5, v50-8) — full Quintic Shadow d=5 → d=3 projection with Fibonacci alternation and φ-decay.
- **Mode 6** (Session 4, v50-7) — full Septic Otherworld heptagram with Asymptotic Approach veil-thinning.
- **Mode 7** (Session 5, v50-8) — full Nonic Recursion holographic depth with recursive feed-forward.
- **Mode 8** (Session 3, v50-6) — full Magical Impedance cycling per the corrected §3.3 Table 2 formula.
- **Mode 9** (Session 6, v50-9) — full Exception State V(E)=0 grounding pull with §23 elegance variance kernel, Koide cosmological grounding weight, Observational Displacement residual, manifold-cycle phasor, and the M-states.md 8:7 M-vacuum/M-matter split.
- **Mode 10** (Session 7, v50-10) — full Mexican-hat with corrected Wirtinger gradient (factor-of-2 bug fix), Higgs radial massive mode at `m_H = √(2μ²)`, Goldstone angular massless tangent mode along the vacuum ring, and T's progressive vacuum-substantiation envelope — per Lagrangian Field Theory §VIII. All n-dependence restored; all five lambda parameters are now real consumers.
- **Mode 11** — still in its original short form; needs corpus research from `ET_Weak_Sector_Open_Directions_Closed.md`, `ET_Weak_Sector_Four_Open_Questions.md`, and `ET_Weak_Sector_d4_to_d12_Investigation.md` before implementation. **This is Session 8's scope, combined with the full verification pass over the entire file.**

The implementation has been executed in seven sessions (1, 2, 3, 4, 5, 6, 7), each focused on one or two modes, with verification at each step. The only remaining work is Session 8 (Mode 11 — Route A/B Cascade + the full verification pass over the entire file).

The user's instructions remain clear: full and proper implementation per the corpus, no shortcuts, no ad hoc tuning, ET-derived math only, no removals, with the three tools (Identification Principle, Descriptor Gap Principle, Subsumption Law) applied to each mode.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

— Michael James Muller / Aevum Defluo

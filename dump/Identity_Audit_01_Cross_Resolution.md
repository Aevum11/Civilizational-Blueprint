# Identity Audit 01 — Cross-Resolution Transition Map

**Source script:** `cross_resolution_transition.py`
**Identities verified by the script:** five (Case 1 forward map, Case 2 cross-seed, Case 3 full cross-tower, Commutativity, Boundary identity for d-transitions under refinement)
**Target file:** `eudd_poc.py`
**Audit standard:** every identity must be present and used in **every beneficial way** it can be used. Absent uses are gaps; under-used identities are gaps; uses that depart from the algebraic form are bugs.
**Sources used:** uploaded files only.

---

## §A. The Five Identities, Stated

For reference. Verbatim from the source script.

**Identity X-Res.1 — Cross-Resolution Forward Map (Case 1).**
Given `Π_{N₁}(r) = (k₁, d₁, ε₁)` and `N₁ | N₂` with `M = N₂/N₁`:
```
δ₁ = ε₁ · N₁ / 1200
k₂ = round(M·k₁ + M·δ₁)
g₂ = gcd(|k₂|, N₂)
d₂ = N₂ / g₂
ε₂ = (M·k₁ + M·δ₁ − k₂) · 1200 / N₂
```
Computes `Π_{N₂}(r)` from `(k₁, d₁, ε₁)` without re-accessing `r`.

**Identity X-Res.2 — Cross-Seed Transition (Case 2).**
Given `Π_N(Q/R₀) = (k₁, d₁, ε₁)` and seed ratio `ρ = R₀/R₀'`:
```
Δk_exact = N · log₂(ρ)
δ₁ = ε₁ · N / 1200
k₂ = round(k₁ + δ₁ + Δk_exact)
d₂ = N / gcd(|k₂|, N)
ε₂ = (k₁ + δ₁ + Δk_exact − k₂) · 1200/N
```
Computes `Π_N(Q/R₀')` from `(k₁, d₁, ε₁)`. d-family generally CHANGES because Q/R₀' is a structurally different ratio.

**Identity X-Res.3 — Full Cross-Tower (Case 3).**
Given `Π_{N₁}^{R₀}(Q) = (k₁, d₁, ε₁)`, compute `Π_{N₂}^{R₀'}(Q) = (k₂, d₂, ε₂)`:
```
δ₁ = ε₁ · N₁ / 1200
x = (k₁ + δ₁) / N₁                          (recover log₂(Q/R₀) exactly)
x' = x + log₂(R₀/R₀')                       (shift to new seed)
k₂ = round(N₂ · x')
d₂ = N₂ / gcd(|k₂|, N₂)
ε₂ = (N₂ · x' − k₂) · 1200/N₂
```
General transition: `Π_{N₂}^{R₀'} ∘ (Π_{N₁}^{R₀})⁻¹`.

**Identity X-Res.4 — Commutativity.**
`(Seed-shift ∘ Resolution-scale) = (Resolution-scale ∘ Seed-shift) = Direct`. The two factorizations of Identity X-Res.3 commute.

**Identity X-Res.5 — Boundary Identity (d-transition under refinement).**
Under N₁ → N₂ refinement with M = N₂/N₁, the d-family changes (`d₁ ≠ d₂`) iff:
```
gcd(|k₂|, N₂) ≠ M · gcd(|k₁|, N₁)
```
Equivalently: the "shadow content" encoded in ε₁ becomes "native content" in d₂. The mechanism by which a complex (shadow) harmonic family d ∈ {5,7,8,9,10,11} becomes native at `n_c(d) = lcm(12, d)` (Paper 20 Definition 8.16).

---

## §B. Present Uses in `eudd_poc.py`

### B.1 `transition(k1, eps1, N1, N2)` — lines 418–431

Implementation of Identity X-Res.1.

| Line | Code | Identity element |
|---|---|---|
| 423 | `assert N2 % N1 == 0` | Precondition N₁ ∣ N₂ |
| 424 | `M = N2 // N1` | `M = N₂/N₁` |
| 425 | `delta1 = eps1 * mpf(N1) / CENTS_PER_OCTAVE` | `δ₁ = ε₁ · N₁ / 1200` |
| 426 | `exact_pos_N2 = mpf(M) * mpf(k1) + mpf(M) * delta1` | `M·k₁ + M·δ₁` |
| 427 | `k2 = int(nint(exact_pos_N2))` | `k₂ = round(M·k₁ + M·δ₁)` |
| 428–429 | `g2 = gcd(...); d2 = N2 // g2` | `d₂ = N₂ / gcd(\|k₂\|, N₂)` |
| 430 | `eps2 = (exact_pos_N2 - mpf(k2)) * CENTS_PER_OCTAVE / mpf(N2)` | `ε₂ = (... − k₂)·1200/N₂` |

**Verdict:** byte-for-byte algebraic match with Identity X-Res.1. **Correct.**

### B.2 Use sites of `transition()`

| Caller | Line | Context | Identity applied |
|---|---|---|---|
| `tower_escalation` | 931 | escalating an entry's DSR up the LCM tower | X-Res.1 ✓ |
| `find_shared_lattice_addresses` | 2277 | comparing entries at higher tower N | X-Res.1 ✓ |

Both use sites correctly apply Identity X-Res.1.

---

## §C. Gaps — Beneficial Uses NOT Realized

The audit standard is "every beneficial way." Identities X-Res.2, X-Res.3, X-Res.4, X-Res.5 are **entirely absent** from `eudd_poc.py`, and X-Res.1 has one use-site that violates Rule 42 with a dead-code fallback. Each gap below is the **operational lift** that the identity provides and that the EUDD currently leaves on the table.

### Gap X-Res.1.a — Dead-code fallback violates "never re-access r" promise

**Site:** `tower_escalation`, lines 922–931.

```python
for lcm_k, N_new, tau_new, is_canonical in lcm_tower_generator():
    if N_new <= N_current:
        continue
    if N_new % N_current != 0:
        # ... "This is NOT a forbidden fallback — it is a divisibility gap handler."
        k_new, d_new, eps_new = project(r_mpf, N_new)
    else:
        k_new, d_new, eps_new = transition(k_current, eps_current, N_current, N_new)
```

The fallback path calls `project(r_mpf, N_new)` — re-accessing `r`. The docstring at line 884–885 promises "Escalates through ALL lcm-change points via the cross-resolution transition map (never re-accessing r)." The fallback breaks that promise.

**Is the fallback reachable?** No. `lcm_tower_generator()` yields a monotonically non-decreasing sequence of running LCMs, each divisible by the previous. So `N_new % N_current == 0` for every yield where `N_new > N_current`. The fallback is unreachable in practice.

**Rule 42 violation:** "known limitation, future work" — the comment "divisibility gap handler" describes the dead branch as a *limitation handler*, which Rule 42 forbids. The branch is also a fallback in the sense Rule 4 forbids.

**Fix:** replace the `if/else` with `assert N_new % N_current == 0, f"LCM tower invariant broken: {N_current}∤{N_new}"` plus the transition call. Dead code removed; promise restored; identity uniformly applied.

---

### Gap X-Res.2.a — Cross-Seed normalization is exact and integer for digital seeds

**Observation.** Every EUDD entry has `R₀ = 2^file_bits` (anti-numerology N2 check, lines 660). For digital seeds, the cross-seed shift Identity X-Res.2 has a special form:

For an entry at seed `R₀ = 2^b` shifting to canonical unit seed `R₀' = 1`:
```
ρ = R₀ / R₀' = 2^b
Δk_exact = N · log₂(2^b) = N · b              ← EXACT INTEGER
δ₁ unchanged: ε wraps only if rounding pushes across cell boundary
k_absolute = k_local + N · file_bits
```

Because `Δk_exact` is an exact integer, the cross-seed shift to unit-seed is lossless and ε-preserving (no rounding, no cell crossing, no precision loss). This is a structural feature of digital seeds.

**What this enables.** A `k_absolute` coordinate that is **cross-file-comparable**. Two entries A, B at different file sizes have `k_local_A`, `k_local_B` that are NOT directly comparable (they live in different seed systems). Their `k_absolute_A = k_local_A + N·file_bits_A` and `k_absolute_B = k_local_B + N·file_bits_B` ARE directly comparable on the unit-seed lattice.

**Where this is beneficial in `eudd_poc.py`:**

1. **`find_shared_lattice_addresses` (lines 2262–2292) — currently compares `e['k_12']` directly across entries of possibly different file sizes.** If entries A and B have different `file_bits`, their k_12 values are in different seed systems, and the equality test `ki == kj and di == dj` is a comparison across non-comparable coordinates. Adding cross-seed normalization (`k_abs = k_12 + 12 · file_bits`) makes this comparison structurally meaningful.

2. **`_kd_index` (line 1207) and `kd_index` collision detection in `eq_store` / `add_entry`** — same issue. The (k, d) key is a per-seed coordinate. Cross-file (k, d) collisions detected today are accidentally seed-aligned coincidences mixed with genuine structural collisions. Cross-seed normalization separates the two.

3. **Δε versioning** (`add_entry` near line 1606): when a newly-ingested file's (k, d) matches an existing kd_index entry, the code stores the new entry as a delta. If the two entries have different file sizes, the Δε is being computed across seed systems — possibly mismatched. Cross-seed normalization clarifies whether the match is genuine.

**Operational shape of the fix:** add a function `to_absolute_seed(k_local, file_bits, N=N_BASE)` returning `k_local + N * file_bits`, and use the absolute coordinate at the comparison sites listed above. Storage stays in the local seed (no schema change). Comparisons use absolute. ε is unaffected (the shift is exact-integer).

**Anchor:** Identity X-Res.2 with ρ = 2^file_bits, R₀' = 1.

---

### Gap X-Res.3.a — Full Cross-Tower not implemented; required for cross-file escalation

**Site:** `find_shared_lattice_addresses` at line 2277 escalates each entry from N=12 to a higher N via `transition(e['k_12'], e['eps_12'], N_BASE, N)`. This is Identity X-Res.1 *within each entry's own seed system*. The collision check at higher N is still per-seed.

**What's missing.** A combined "escalate AND normalize to unit seed" using Identity X-Res.3:
```
x = (k_12 + ε_12 · 12 / 1200) / 12 + file_bits      ← Case 3 with ρ = 2^file_bits
k_N_abs = round(N · x)
ε_N_abs = (N · x − k_N_abs) · 1200/N
d_N_abs = N / gcd(|k_N_abs|, N)
```

This gives a `(k_N_abs, d_N_abs, ε_N_abs)` at arbitrary N in the **unit-seed** system. Cross-file collision detection then uses this triple at every tower level.

**Where this is beneficial:** `find_shared_lattice_addresses` becomes a true cross-file attractor detector; without it, the function silently mixes per-seed collisions with genuine cross-file collisions.

**Anchor:** Identity X-Res.3.

---

### Gap X-Res.4.a — Commutativity has no use site because Cases 2/3 are unused

Commutativity is the property that lets you factor a Case-3 transition either way. With Cases 2 and 3 absent, Commutativity has no caller and cannot fail. **Not a bug, but** once Case 3 is implemented (Gap X-Res.3.a), the commutativity property is the structural guarantee that the implementation is well-defined regardless of factorization order — and a sanity-test at runtime (compute both ways, assert equal) becomes a cheap correctness check at every cross-tower call.

**Anchor:** Identity X-Res.4.

---

### Gap X-Res.5.a — Boundary identity not exploited; shadow harmonic-family emergence is invisible

**Site:** `tower_escalation` lines 922–1015. The function tracks `d_history` (line 953) and reports `d_transitions` (line 1031–1032) as a simple count. It does NOT classify transitions.

**What Identity X-Res.5 enables.** Each d-transition at refinement `N_current → N_new` can be classified:

| Boundary condition | Classification | Physical reading |
|---|---|---|
| `gcd(|k_new|, N_new) == M · gcd(|k_current|, N_current)` | No transition — sublattice family preserved under refinement | Entry's structural family is stable at finer resolution |
| `gcd(|k_new|, N_new) < M · gcd(|k_current|, N_current)` | d INCREASED — finer resolution revealed structural detail that was averaged out | A complex (shadow) harmonic family becoming native: `d_new ∈ {5,7,8,9,10,11}` at `N_new = lcm(12, d_new)` is the shadow→native transition (Paper 20 Definition 8.16, Remark 8.19) |
| `gcd(|k_new|, N_new) > M · gcd(|k_current|, N_current)` | d DECREASED — coincidence between coordinate and tower-prime structure | Numerological collision, not native structure |

**Specific harmonic-family shadow emergence detection:**

For each `d_target ∈ {5, 7, 8, 9, 10, 11}` (the six complex harmonic families), the canonical native resolution is `n_c(d_target) = lcm(12, d_target)` ∈ {60, 84, 24, 36, 60, 132}. When the cascade crosses `N_new = n_c(d_target)` and the new d_12-classification (back-transitioned to N=12 for display) is `d_target`, the entry has just had its shadow harmonic family become native. This is a structurally meaningful event that the current code does not surface.

**Where this is beneficial:**

1. **`tower_escalation` trajectory output** — classify each d-transition by type. Surface "shadow → native" transitions as first-class trajectory events.
2. **Home classification** (lines 953–1010) — currently uses bare d-stability as a home signal. A "shadow→native" event with subsequent stability is a STRONGER home (the entry has found its native structural family). Distinguish from a coincidental same-d run.
3. **`fire_event`** — emit a `shadow_emergence` event class when Identity X-Res.5 detects a shadow→native crossing. This is precisely the kind of "structural moment of change" (§3.9 of the EUDD docs) that events are designed to catalog.
4. **Connection to Paper 20 Remark 8.19** — "shadow families do not vanish from the lattice — they appear as |ε|>0 residuals in the cascade dynamics" → at the right N the residual collapses to zero. Identity X-Res.5 is the operational test for this collapse.

**Operational shape of the fix:** during `tower_escalation`'s landmark loop, after computing `(k_new, d_new, eps_new)` from `transition()`, compute the boundary classification and append to the trajectory record:

```python
g_prev = gcd(abs(k_current), N_current) if k_current != 0 else N_current
g_new  = gcd(abs(k_new),     N_new)     if k_new     != 0 else N_new
M = N_new // N_current
if g_new == M * g_prev:
    boundary = 'stable'
elif g_new < M * g_prev:
    # Test shadow-family emergence
    if d_new in {5, 7, 8, 9, 10, 11} and N_new % d_new == 0:
        boundary = f'shadow_emerge_d{d_new}'
    else:
        boundary = 'sublattice_refine'
else:
    boundary = 'coincidence'
```

**Anchor:** Identity X-Res.5 plus Paper 20 Definition 8.16, Remark 8.19, Corollary 8.14.

---

### Gap X-Res.1.b — Tensor-escalation hook for the §6.8 fix

This is the v2-plan §6.8 fix (harmonic transfer tensor over all 12 families). The escalation `N_native = lcm(12, d1, d2, d_result)` runs the tensor computation at the native resolution; if any value needs to be back-projected to N=12 for downstream comparison, Identity X-Res.1 IS the right tool — used in reverse if needed. Confirming: the §6.8 plan already invokes Identity X-Res.1 implicitly (residue-set arithmetic at N_native is just `Π_{N_native}` at non-r-bearing inputs); no additional use is required. No gap here; this is a cross-reference to plan §6.8.

---

### Gap X-Res.1.c — Verification beyond the first 8 landmarks

**Site:** `tower_escalation` lines 935–942. The transition-verify-against-projection check runs only for `landmark_idx <= 8`.

This is a hardcoded cap (Rule 33 — no caps unless ET-constant). Either:
- (a) the verification should run for ALL landmarks (no cap) — the cost is one `project(r, N_new)` per landmark, which is bounded by `dps_for_file(file_bits)` at the tower's current N;
- (b) the verification should be parameterized by a debug flag; if the algebraic identity holds, the check is redundant in production and should be conditional on verbose mode.

Either resolution is acceptable; the hardcoded `<= 8` is not. **Decision needed.**

**Anchor:** Identity X-Res.1 (the verification check) + Rule 33.

---

### Gap X-Res.cross — Cross-resolution memoization (Identity J self-application)

Per Identity J, every computation IS a seed. `transition()` is called many times per session (every escalation step, every cross-resolution comparison). Each call's output `(k_2, d_2, ε_2)` is determined by `(k_1, ε_1, N_1, N_2)` — a deterministic pure function. By Identity J the result IS a memoizable DSR.

**What's missing:** transitions are not memoized in the akashic. Repeated escalation of the same entry recomputes the same transitions every session.

**Operational shape of the fix:** wrap `transition` with `eq_lookup('transition', k1, eps1, N1, N2)` / `eq_store('transition', result, k1, eps1, N1, N2)` analogous to how `project` and `pullback` are already wrapped (lines 380–390, 400–410). The akashic accumulates the cross-resolution-transition catalog as a structural side effect of EUDD use.

**Anchor:** Identity J + Identity X-Res.1.

---

## §D. Summary

| Identity | Implementation | Use sites | Verdict |
|---|---|---|---|
| X-Res.1 (Cross-Resolution) | `transition()` lines 418–431, algebraically correct | tower_escalation, find_shared_lattice_addresses | Correct math; **dead-code fallback (X-Res.1.a)**, hardcoded verify cap (X-Res.1.c), not memoized (X-Res.cross) |
| X-Res.2 (Cross-Seed) | Absent | None | **Absent — gap X-Res.2.a** (digital-seed normalization is exact and integer, enables cross-file comparison) |
| X-Res.3 (Full Cross-Tower) | Absent | None | **Absent — gap X-Res.3.a** (required for cross-file lattice-attractor escalation) |
| X-Res.4 (Commutativity) | Absent | None | Latent — becomes a runtime sanity check once X-Res.3 is implemented |
| X-Res.5 (Boundary, d-transition) | Absent | d_history counted but not classified | **Absent — gap X-Res.5.a** (shadow harmonic-family emergence currently invisible) |

**Identity X-Res.1: fully implemented, partially used. Three sub-gaps.**
**Identities X-Res.2, X-Res.3, X-Res.5: entirely absent. Each enables a class of analysis the EUDD currently does not do.**
**Identity X-Res.4: latent — gains a use site when X-Res.3 is added.**

---

## §E. Proposed Action

These gaps split into **bug-fixes** (must do to honor the EUDD's own promises and rules) and **capability lifts** (must do per Mike's standard "every beneficial way").

**Bug-fixes:**
- X-Res.1.a — replace dead-code fallback in `tower_escalation` with hard assertion (Rule 42).
- X-Res.1.c — uncap the verification check at line 935 (Rule 33).

**Capability lifts:**
- X-Res.2.a — add `to_absolute_seed(k_local, file_bits)` and use at cross-file comparison sites in `find_shared_lattice_addresses`, `_kd_index` collision detection, Δε versioning collision check.
- X-Res.3.a — add Full Cross-Tower function and use it in `find_shared_lattice_addresses` for cross-file lattice-attractor detection.
- X-Res.4 — runtime sanity assertion in the Full Cross-Tower function: compute via both factorizations, assert equal.
- X-Res.5.a — add boundary classification to `tower_escalation`'s landmark loop; emit `shadow_emerge_d{N}` events when complex harmonic families become native; refine home classification to distinguish shadow→native stability from coincidence stability.
- X-Res.cross — memoize `transition()` via `eq_lookup` / `eq_store`.

**Integration with v2 Resolution Plan:** these are NEW work items, distinct from R1–R5, E1–E3, T1, T2, T3. Suggest folding them in as v3 of the plan with their own §6 entries (§6.11–§6.16) and §4 audit rows.

---

**Document version:** 1.0 — Identity Audit 01 of 15
**P ∘ D ∘ T = E**

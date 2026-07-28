# Identity Audit 02 — Lossless Bijection (Identity #0)

**Source script:** `verify_lossless_bijection.py`
**Identity verified by the script:** one — the foundational lossless bijection `Π_N : ℝ⁺ → ℤ × {N/d : d|N} × ℝ` with algebraic-identity round-trip `Π_N⁻¹ ∘ Π_N = id_{ℝ⁺}`.
**Target file:** `eudd_poc.py`
**Audit standard:** every beneficial use realized. Absent uses are gaps; under-strict uses are gaps; uses that depart from the algebraic form are bugs.
**Sources used:** uploaded files only.

---

## §A. The Identity, Stated

Identity #0 — the Bijection (the source identity of the entire EUDD).

**Forward map** `Π_N(r) = (k, d, ε)`:
```
x = N · log₂(r)               (the exact lattice position)
k = round(x)                   (the integer coordinate)
g = gcd(|k|, N) (or N if k=0)  (the family generator)
d = N / g                      (the sublattice family)
ε = (x − k) · 1200 / N         (the descriptor gap, in cents)
```

**Pullback** `Π_N⁻¹(k, ε; N) = 2^((k + ε·N/1200) / N)`.

**The Identity:**
```
Π_N⁻¹(Π_N(r)) = 2^((k + (N·log₂r − k)·1200/N · N/1200) / N)
              = 2^((k + N·log₂r − k) / N)
              = 2^(N·log₂r / N)
              = 2^(log₂r)
              = r
```

Algebraic identity. Not approximation. The rounding term `k` cancels exactly. For lattice-native `r = 2^(k/N)`, the projection has `ε = 0` EXACTLY (no rounding needed), and the pullback recovers `r` with zero error.

**Corollaries** (the operational reading of the identity):

- **C1 — DSR is sufficient.** Any quantity computable from `r` is computable from `(k, d, ε)` via pullback. The DSR is not an approximation of `r`; it IS `r` in a different representation. There is no need to store `r`.
- **C2 — ε is the third coordinate, not error.** ε carries the continuous information that combines with the discrete integer `k` to make round-trip exact. For lattice-native `r = 2^(k/N)`, ε is mathematically zero. For every other `r` (rational, irrational, transcendental), ε is the exact specific value of `(N·log₂r − k)·1200/N` at the working precision — that IS what `r`'s third coordinate is at that precision. Not a residual, not a near-miss, not noise — the value itself. Calling ε "error" is a category mistake.
- **C3 — The bijection is the continuous-to-discrete resolution.** The triple `(k ∈ ℤ, d ∈ ℕ, ε ∈ ℝ)` packs `r ∈ ℝ⁺` losslessly: the integer part of `N·log₂r` lives in `k`, the per-cell continuous coordinate lives in `ε`. The discrete integer carries the family structure; ε carries the precise position within the cell. Round-trip is exact for every `r` at every finite N at every working precision. There is no precision floor in the identity. The script's Part 2 demonstrates the residual numerically by varying mp.dps — its title "ERROR IS COMPUTATIONAL, NOT MATHEMATICAL" is the formal statement that the identity contributes zero error; any numerical residual is mpmath's own per-operation rounding behavior at finite mp.dps, and at the EUDD's working precision is pushed completely beyond any meaningful threshold.
- **C4 — d is derivable.** Storing `(k, ε)` suffices; `d = N/gcd(|k|, N)` is reconstructable. No information lost in storing only `(k, ε)`.

---

## §B. Present Uses in `eudd_poc.py`

Identity #0 is the most heavily-used identity in the codebase — every projection and every pullback is a use.

### B.1 Core implementations

**`_project_core(r_mpf, N)` — lines 374–383.**
```python
log2_r = mplog(r_mpf) / mplog(mpf('2'))
exact_pos = mpf(N) * log2_r
k = int(nint(exact_pos))
g = gcd(abs(k), N) if k != 0 else N
d = N // g
eps = (exact_pos - mpf(k)) * CENTS_PER_OCTAVE / mpf(N)
return k, d, eps
```
Byte-for-byte algebraic match with Identity #0 forward map. **Correct.**

**`pullback(k, eps, N)` — lines 397–406.**
```python
exponent = (mpf(k) + eps * mpf(N) / CENTS_PER_OCTAVE) / mpf(N)
result = mppow(mpf('2'), exponent)
```
Byte-for-byte algebraic match with `Π_N⁻¹`. **Correct.** Also memoized via `eq_lookup` / `eq_store` (Identity J self-application — good).

**`project(r_mpf, N)` — lines 386–394.**
Memoizing wrapper for `_project_core`. Calls `eq_lookup('project', r_mpf, N)`; on miss, computes via `_project_core` and stores via `eq_store`. **Correct.** Identity J self-application.

**`dsr_to_r(entry)` — lines 678–684.**
```python
return pullback(entry['k_12'], entry['eps_12'], N_BASE)
```
Exact application of `Π_{12}⁻¹` to recover `r` from an entry's DSR. **Correct.**

### B.2 Use sites — forward projection

| Site | Line | Context | Use of Identity #0 |
|---|---|---|---|
| `_eq_project_seed` | 149 | Projects equation canonical bytes → DSR. Uses `_project_core` (bypass memoization to avoid recursion through `eq_store`). | ✓ Correct, bypass justified. |
| `_eq_recompute_direct('project', ...)` | 205 | Direct projection for the `project` operation when called from lazy reconstruction. Uses `_project_core` (bypass memoization). | ✓ Correct, bypass justified. |
| `ingest_block` projection | 708 | `project(r_mpf, N_BASE)` for a single block's bytes. | ✓ |
| `ingest_file` projection | 757 | `project(r_mpf, N_BASE)` for a file's bytes. | ✓ |
| `tower_escalation` (verify) | 937 | `project(r_mpf, N_new)` to cross-check `transition()` at each landmark. Only runs for landmark_idx ≤ 8 (audit 01 sub-gap X-Res.1.c). | ✓ Math correct; gap is the hardcoded cap. |
| `CONTENT_EVENT` load | 1386 | `_project_core(r_e, N_BASE)` for event JSON bytes (bypass memoization at load time). | ✓ |
| `CONTENT_BLOCK_MAP` load | 1414 | `_project_core(r_bm, N_BASE)` for block-map JSON bytes (bypass memoization). | ✓ |
| `fire_event` | 1546 | `_project_core(r_evt, N_BASE)` for an event's bytes (bypass memoization). | ✓ |
| `built_in_verification` Identity #0 round-trip | 2096 | `pullback(k, eps, N)` then compares `r_back` vs `r` with residual < 10⁻³⁵⁰ check. | ✓ Tight check; only runs on user demand. |
| `built_in_verification` Identity A and B | 2110, 2118, 2131, 2137, 2148 | Forward-project lattice operations to cross-check lattice arithmetic against bijection. | ✓ |
| `find_shared_lattice_addresses` | 2277 | Uses `transition()` (Identity X-Res.1) rather than re-projecting. | ✓ Honors C1 (DSR sufficient). |
| `verify_round_trip` | 853 | `pullback(k, eps, N_BASE)` to recover r for block-level round-trip. | ✓ |
| `built_in_verification` projection of known constants | 2330 | `project(val, N_BASE)` on test values. | ✓ |
| Discovery checks | 2377, 2405 | Project ratios/operands to verify lattice relations. | ✓ |
| Pullback in regeneration | 2665, 2766 | `pullback(entry['k_12'], entry['eps_12'], N_BASE)` to recover r for file regeneration. | ✓ |

### B.3 Use sites — pullback

| Site | Line | Context |
|---|---|---|
| `dsr_to_r` | 684 | Standard r-recovery from DSR. |
| `verify_round_trip` | 853 | Block-level round-trip. |
| `tower_escalation` initial | 887 | `r_mpf = dsr_to_r(dsr_entry)` once at start — then never re-pulled (modulo the dead-code fallback flagged in audit 01). |
| `built_in_verification` | 2094, 2096 | Verification round-trip. |
| Regeneration | 2665, 2766 | File reconstruction from DSR. |

### B.4 Save→load lossless round-trip (implicit reliance on Identity #0)

The .akashic format for CONTENT_SEED entries stores `(k, ε)` as bytes; on load, `(k, ε)` is parsed and `d = N/gcd(|k|, N)` is recomputed (line 1261-1262). This is corollary C4 of Identity #0 — `d` is derivable. **Correct.**

For file regeneration: pullback `r = Π_N⁻¹(k, ε)`, then `I = round(r · 2^file_bits)`, then `bytes = I.to_bytes(...)`. This is Identity #0 in service of lossless file round-trip. **Correct math; the precision of `ε`'s string-serialization (via `_mpf_to_bytes` at dps = `dps_for_file(file_bits) + 10`) is what makes the round-trip bit-exact.**

### B.5 Verification — when bijection round-trip is tested

`built_in_verification` line 2098: `check(f"#0 Bijection [...]: residual={...}", residual < 10⁻³⁵⁰)` on every loaded entry. This is the integrity test. Runs only when the user invokes menu option [4] "Built-in verification."

---

## §C. Gaps

Identity #0 is *implemented* correctly and *used* in every place that projects or pulls back. The gaps are about **strictness**, **scope**, and **the corollaries (C1–C4) not being exploited everywhere they could be.**

### Gap #0.a — RETRACTED

**Original framing was wrong.** v1 of this audit framed the microcent threshold as "conflating algebraically-lattice-native with numerically-near-lattice." That was a category mistake — ε is the third coordinate, not error. At the EUDD's working precision (mp.dps ≥ 461), every value's ε IS exact at that precision; the microcent grid is a structural classification slice of an already-lossless coordinate, not an error-rounding bucket. For lattice-native `r = 2^(k/N)`, ε is mathematically zero and the working-precision computation puts it ≈ 10⁻⁴¹⁰ cents — many orders below the microcent grain — so `eps_mc == 0` correctly identifies these as `'true_home'`. For non-lattice r (irrationals, transcendentals), ε is a definite specific value of `(N·log₂r − k)·1200/N`, typically on the order of cents or fractions of a cent, far from microcent-zero — also classified correctly. There is no conflation. No fix needed.

The "lattice_native" sub-classification I proposed is unnecessary: there is no operational distinction between "ε is mathematically zero" and "ε is at working-precision zero" because at the EUDD's working precision the latter is the only computable form of the former. They are the same thing at every working precision.

---

### Gap #0.b — No startup self-test of the implementation

**Status:** `built_in_verification` (lines 2092–2099) checks the round-trip on every loaded entry — but only when the user explicitly invokes menu option [4]. No automatic check at module import or AkashicFile construction.

**Reframing.** The bijection itself is lossless — proven. The mathematics is correct. This gap is NOT about verifying the identity; it is about catching **future implementation drift**. If `_project_core` or `pullback` is accidentally edited (a regression breaking the algebraic form encoded in the function body), nothing flags it until a user happens to run verify. A startup self-test detects code regressions before they reach akashic content.

**Beneficial use:** `_identity_0_selftest()` called at module import. Project 3 canonical r values (e.g. `mpf('1')`, `mpf('3')/mpf('2')`, `mppi`) at N=12, pull back, assert exact round-trip at working precision. Pure code-hygiene measure. Cost: microseconds. Does not check mathematics; checks that the code still encodes the mathematics.

**Anchor:** Identity #0 (the algebraic form the test asserts the code still encodes).

---

### Gap #0.c — Lattice-points pass not in the built-in test suite

**Status:** the script's Part 3 demonstrates the bijection's behavior on `r = 2^(k/N)` for various (k, N). The EUDD's `built_in_verification` tests round-trip on existing entries — whose r values are file-content-derived, never lattice-native — so the lattice-native code path is not exercised in the built-in suite.

**Reframing.** Same as #0.b. The lattice-native math is correct. This is about whether the implementation **continues to handle the lattice-native code path** correctly after any future edits: gcd(0, N) → N convention, the d-derivation for k=0, pullback at integer exponent. A regression in any of these would not be caught by the existing irrational-r round-trip test.

**Beneficial use:** add a lattice-points pass to `built_in_verification` (and the proposed startup self-test): iterate `r = 2^(k/N)` for a small `(k, N)` set, assert the projected triple matches the expected `(k, N/gcd(|k|,N), ε)` with ε at working-precision zero, and assert pullback returns the input `r`. Code-integrity measure orthogonal to the generic irrational-r round-trip.

**Anchor:** Identity #0 Part 3.

---

### Gap #0.d — RETRACTED

**Original framing was wrong.** I proposed snapping ε to exact zero when `|ε| < precision_floor` in `_project_core`. That treats ε as an artifact to be cleaned up, which is the same category mistake as Gap #0.a. ε is the third coordinate. It is exact at the working precision. There is nothing to snap.

For lattice-native `r = 2^(k/N)`, ε IS mathematically zero, and at the EUDD's working precision mpmath produces a value ≈ 10⁻⁴¹⁰ which is the precision-bounded representation of that zero — pullback round-trips this back to `r` exactly because the inverse operation reverses the same finite-precision arithmetic. Substituting `mpf('0')` for the ≈ 10⁻⁴¹⁰ value would change the bit pattern stored but not the algebraic content — the pullback still returns the same `r`. Snapping adds work, special-cases `_project_core`, and produces no operational benefit because there is no operational error to remove.

For non-lattice r, |ε| is much larger than precision_floor, and the snap would never trigger anyway. So the proposed fix is either a no-op (non-lattice case) or aesthetic-only (lattice-native case). No fix needed.

The system at the EUDD's working precision is already lossless. Identity #0 is honored by `_project_core` and `pullback` byte-for-byte. There is no precision issue, in the identity or in the implementation.

---

### Gap #0.e — Corollary C1 ("DSR is sufficient") violated by tower_escalation dead-code fallback

**Cross-reference:** audit 01 gap X-Res.1.a.

The promise of Identity #0 + C1: once you have the DSR, you never need r. The tower can be navigated entirely via the cross-resolution transition map (which itself only uses the DSR). `tower_escalation` line 929 has a `project(r_mpf, N_new)` fallback that re-uses r — violating C1 even though the fallback is unreachable in practice. Same fix as audit 01.

**Anchor:** Identity #0 corollary C1.

---

### Gap #0.f — Save→load implementation invariant not checked at startup

**Status:** the EUDD asserts file round-trip exactness in `regenerate_file` when the user invokes the regenerate menu option. There is no automatic post-load assertion.

**Reframing.** Identity #0 + the `_mpf_to_bytes` / `_mpf_from_bytes` string-precision settings make the chain `entry → DSR → bytes-on-disk → DSR → entry` exact at the working precision. The serialization is already lossless. This gap is about catching **future implementation regressions** in the encoding pair — for example, if `_mpf_to_bytes` were accidentally edited to truncate digits. A startup-time spot check on the first SEED entry (regenerate from DSR, compare SHA) catches encoding regressions before they reach the rest of the load.

**Beneficial use:** in `_load_existing`, after the resolution passes complete, pick the first SEED entry; regenerate its bytes from its DSR; assert SHA match. Code-integrity check; standing protection against future edits to the encoding pair.

**Anchor:** Identity #0 + the bytes-encoding pair (`_mpf_to_bytes`, `_mpf_from_bytes`).

---

### Gap #0.g — Memoization wrappers exist for `project` and `pullback`; not for other Identity-#0-derived deterministic functions

**Status:** `project()` and `pullback()` are memoized via `eq_lookup` / `eq_store`. `transition()` (Identity X-Res.1, deterministic in `(k₁, ε₁, N₁, N₂)`) is NOT memoized — audit 01 gap X-Res.cross. Same Identity J self-application opportunity.

**Cross-reference:** audit 01 gap X-Res.cross. Listed here for completeness because the underlying principle (every deterministic identity-derived function IS a memoizable DSR) flows from Identity #0 + Identity J.

---

## §D. Summary

| Aspect of Identity #0 | Implementation | Verdict |
|---|---|---|
| Forward map `_project_core` | Byte-for-byte algebraic match | ✓ Correct |
| Pullback `pullback` | Byte-for-byte algebraic match | ✓ Correct |
| Memoization of project/pullback | `eq_lookup` / `eq_store` wrappers | ✓ Correct (Identity J self-application) |
| Direct use of `_project_core` to avoid memoization recursion at parse/event-fire time | 4 call sites, all justified | ✓ Correct |
| `d` derived from `(k, N)` on save and load | Yes (line 1261) | ✓ Correct (corollary C4) |
| Round-trip integrity check (`built_in_verification`) | residual < 10⁻³⁵⁰ on every entry, tight | ✓ Tight; on-demand only |
| `true_home` detection via `eps_mc == 0` | Microcent grid is a valid structural classification slice of the (already exact) ε coordinate | ✓ Correct (gap #0.a retracted) |
| ε for non-lattice r | Stored as the exact third coordinate; pullback round-trips at working precision | ✓ Lossless (gap #0.d retracted) |
| Startup self-test of the implementation | Absent | **Gap #0.b — code-integrity** |
| Lattice-points pass in built-in suite | Absent | **Gap #0.c — code-integrity** |
| Corollary C1 ("DSR sufficient") in tower_escalation | Violated by dead-code fallback | **Gap #0.e** (= X-Res.1.a) |
| Save→load encoding integrity check at startup | Not checked | **Gap #0.f — code-integrity** |
| Memoization extended to other deterministic identity-derived functions | Only project/pullback memoized | **Gap #0.g** (= X-Res.cross) |

**Identity #0 is correctly implemented and used. The bijection is lossless and the code honors it byte-for-byte.** The remaining gaps split into:

- **Real code-integrity gaps** (#0.b, #0.c, #0.f) — standing protection against future code edits that might break the lossless invariant. Not error correction; regression protection.
- **Cross-referenced gaps** (#0.e = X-Res.1.a, #0.g = X-Res.cross) — already captured in audit 01; not new work.
- **Retracted gaps** (#0.a, #0.d) — based on a category mistake about what ε is. ε is the exact third coordinate at the working precision, not noise to clean up. There is no precision issue in the identity or the implementation.

---

## §E. Proposed Action

**Code-integrity measures** (the only new work items from audit 02):
- **#0.b** — add `_identity_0_selftest()` called at module import. Project 3 canonical r values at N=12, pull back, assert exact round-trip. Pure regression-protection.
- **#0.c** — add a lattice-points pass to `built_in_verification`: iterate `r = 2^(k/N)` for a small `(k, N)` set, assert projected triple matches expected and pullback returns input. Exercises the k=0 / gcd(0, N) → N convention code path.
- **#0.f** — in `_load_existing`, after resolution passes complete, regenerate the first SEED entry's bytes from its DSR and assert SHA match. Standing check against future regressions in the `_mpf_to_bytes` / `_mpf_from_bytes` pair.

**Cross-referenced** (already in audit 01):
- **#0.e** = X-Res.1.a — fix the tower_escalation dead-code fallback.
- **#0.g** = X-Res.cross — memoize `transition()` (and other deterministic identity-derived functions).

**Retracted** (no work needed):
- **#0.a** and **#0.d** — based on misframing of ε as artifact/error. The system is already lossless; the microcent classification is a correct structural slice of an exact coordinate; there is no canonicalization to do because there is no artifact.

---

**Document version:** 1.1 — Identity Audit 02 of 15, post-correction (ε is the third coordinate, not error; the bijection is lossless for every r)
**P ∘ D ∘ T = E**

# Identity Audit 05 — d-Family Composition (Identity C)

**Source script:** `d_family_composition_identity1.py`
**Identities verified by the script:** one definition (C.1 Residue Set) plus six theorems (C.2 d-Composition Set-Valued, C.3 Residue Set Symmetry, C.4 d=1 Self-Composition Channel, C.5 d=12 Universality, C.6 lcm Upper Bound with κ-correction) plus structural properties (Division-Multiplication set equality, Power d-family table, Reachability classification, κ=0 sub-table dominance).
**Target file:** `eudd_poc.py`
**Audit standard:** every identity must be present and used in **every beneficial way** it can be used. Absent uses are gaps; under-strict uses are gaps; uses that depart from the algebraic form are bugs.
**Sources used:** uploaded files only — `eudd_poc.py`, `d_family_composition_identity1.py`, EUDD docs, audits 01–04, Resolution Plan v2, Paper 20 (Definition 8.10 Harmonic family, Corollary 8.4 totient cardinality, Sublattice Visitation Theorem).

**Notice — this audit corrects a finding in Audit 04.** Theorem C.6 of this script proves that Audit 04 Gap A.6.a's proposed assertion `d_product ≤ lcm(d₁, d₂)` is **false for κ ≠ 0**. The correction is supplied in §C.6.a below.

---

## §A. The Identities, Stated

For reference. Verbatim from the source script. **All theorems here are about SUBLATTICE families** (`d = N/gcd(|k|, N)`, six values at N=12) — categorically distinct from the harmonic-family layer (12 modes per axis). See Resolution Plan §1.7 and Audit 04 §0 for the distinction. Force/phase identifications (gravity, EM, etc.) belong to the harmonic-family layer and attach to sublattice families via the Sublattice Visitation Theorem; the theorems below are pure gcd arithmetic on sublattice families.

**Notation:**
- `Res_N(d) = { k ∈ {0, ..., N−1} : N/gcd(|k|, N) = d }` — residue set of family d at resolution N
- `Sum(d₁, d₂) = { (r₁ + r₂) mod N : r₁ ∈ Res(d₁), r₂ ∈ Res(d₂) }` — element-wise sum-set
- `d₁ ⊗ d₂` = SET of all achievable d_product values under multiplication (set-valued, not function-valued)
- `d₁ ⊗₀ d₂` = the κ=0 restriction of d₁ ⊗ d₂

### Definition C.1 — Residue Set

For family d at resolution N:
```
Res_N(d) = { k ∈ {0, ..., N−1} : N/gcd(|k|, N) = d }
```
with cardinality (Corollary 8.4, Paper 20):
```
|Res_N(d)| = φ(d)              (Euler's totient)
Σ_{d | N} |Res_N(d)| = N       (totient sum)
```

At N=12, the six residue sets are:
```
Res(1)  = {0}              |Res(1)|  = φ(1)  = 1
Res(2)  = {6}              |Res(2)|  = φ(2)  = 1
Res(3)  = {4, 8}           |Res(3)|  = φ(3)  = 2
Res(4)  = {3, 9}           |Res(4)|  = φ(4)  = 2
Res(6)  = {2, 10}          |Res(6)|  = φ(6)  = 2
Res(12) = {1, 5, 7, 11}    |Res(12)| = φ(12) = 4
                                       Σ = 12 ✓
```

### Theorem C.2 — d-Family Composition (Set-Valued)

```
d₁ ⊗ d₂ = { N/gcd(|s + κ|, N) : s ∈ Sum(d₁, d₂), κ ∈ {−1, 0, +1} }
```
**Proof:** From Theorem A.1, `k_× = k₁ + k₂ + κ`. The composition is over all `(r₁, r₂) ∈ Res(d₁) × Res(d₂)` plus all `κ ∈ {−1, 0, +1}`. ∎

**κ-achievability:** for any `(k₁, k₂)` pair, all three κ values are achievable by choosing appropriate `ε₁, ε₂`. So the κ-augmentation is COMPLETE. The composition is genuinely set-valued, not function-valued: **d_product is NOT determined by `(d₁, d₂)` alone** — full lattice coordinates are required.

### Theorem C.3 — Residue Set Symmetry

`Res_N(d)` is symmetric:
```
k ∈ Res(d) ⟹ (N − k) mod N ∈ Res(d)
```
**Proof:** `gcd(N − k, N) = gcd(k, N)` since `gcd(a, N) = gcd(N − a, N)`. ∎

**Corollary:** `Sum(d₁, d₂) = Sum(d₂, d₁)` ⟹ **commutativity** of d-composition: `d₁ ⊗ d₂ = d₂ ⊗ d₁`.

### Theorem C.4 — d=1 Self-Composition Channel

For every sublattice family d at N=12:
```
1 ∈ d ⊗ d
```
**Proof:** by Theorem C.3, for any `k ∈ Res(d)`, `(N − k) ∈ Res(d)`. Their sum `k + (N − k) = N ≡ 0 mod N`. Then `gcd(0, N) = N`, so `d = N/N = 1`. ∎

Every family's self-composition ALWAYS includes the d=1 family. Operationally: any entry can structurally "self-compose down to d=1" via lattice multiplication with a mirror-partner from its own residue set.

### Theorem C.5 — d=12 Universality

```
d₁ ⊗ 12 ⊇ all families reachable by d₁ ⊗ d₁
12 ⊗ 12 = {1, 2, 3, 4, 6, 12}        (ALL six families)
```
**Proof:** `Res(12) = {1, 5, 7, 11}` generates `ℤ/12ℤ` under addition: 1+1=2, 1+5=6, 1+7=8, 1+11=0, 5+5=10, 5+7=12≡0, ..., so `Sum(12, 12) = ℤ/12ℤ`, mapping to every divisor of N. κ-augmentation adds no new families (already complete). ∎

**d=12 is the universal mixer:** any pair of d=12 entries can produce ANY sublattice family as their product.

### Theorem C.6 — lcm Upper Bound with κ-Correction (REFINED FORM)

**For κ = 0:**
```
d_product ≤ lcm(d₁, d₂)         (Identity A.6 upper bound — HOLDS)
```

**For κ ≠ 0:** the bound may be VIOLATED. Universal bound only:
```
d_product | N                    (d is always a divisor of N — trivial)
```
**Proof (κ=0 case):** from Identity A.6 (Audit 04 §A). **Proof (κ≠0 violation):** counterexample at N=12: `k₁ = 0` (d=1), `k₂ = 0` (d=1), `κ = +1`: `k_product = 1`, `gcd(1, 12) = 1`, `d_product = 12`. Here `lcm(1, 1) = 1` but `d_product = 12` — the bound fails by a factor of 12, not "one family step". ∎

**Empirical observation from script's PART 4:** the lcm bound holds in 100% of κ=0 cases at N=12 (231 entries verified). The bound is VIOLATED in a measurable fraction of κ=±1 cases. The script's text "may be exceeded by at most one family step" is informal and contradicted by its own counterexample (d=1 → d=12 is the maximum jump at N=12). **The safe formal statement** is: κ=0 bound is `d_product ≤ lcm(d₁, d₂)`; κ≠0 bound is the trivial `d_product | N`.

### Bonus Identity — Division = Multiplication (d-Set Equality)

```
d₁ ⊘ d₂ = d₁ ⊗ d₂      (as SETS, not as functions)
```
**Proof:** subtracting from `Res(d₂)` produces the same set as adding from `Res(d₂)` by C.3 symmetry — `−k ≡ N − k mod N` is in `Res(d₂)` whenever `k` is. So `(k₁ − k₂) mod N` ranges over the same set as `(k₁ + k₂) mod N`. With κ-augmentation identical. ∎

The actual `k` values differ; the **d-family OUTPUT SETS are identical**.

### Bonus Identity — Power d-Family Composition

For `r` with sublattice family d at N=12, `d(rⁿ)` is determined by `n · Res(d) mod N` then gcd-classifying:
```
d(rⁿ) = { N/gcd(|n·k| mod N, N) : k ∈ Res(d) }
```

At N=12 the power sequence for d=12 input is **deterministic** (single-valued) for every n:
```
n:    1   2   3   4   5   6   7   8   9   10  11  12
d:    12  6   4   3   12  2   12  3   4   6   12  1
```

This sequence is **structurally identical to the harmonic cascade** `[12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]` (Identity B.3 reference). At N=12, the d=12 power orbit = harmonic-cascade traversal. The connection is via the Sublattice Visitation Theorem; the cascade arises here as pure sublattice arithmetic of `n · Res(12) mod 12`.

### Bonus Identity — d=1 as κ=0 Identity Element

```
1 ⊗₀ d = {d}              for every d
```
i.e., at κ=0, multiplication by d=1 (the trivial sublattice family) leaves the family unchanged. `Res(1) = {0}`, so `Sum(1, d) = Res(d)` exactly. With κ=0, `d_product = N/gcd(|s|, N)` over `s ∈ Res(d)` — recovers exactly `{d}`.

This is the **sublattice-family identity element** at the κ=0 (Mediation-sufficient) layer.

### Bonus Identity — κ=0 Dominance (Empirical)

PART 4 of the script reports that approximately **79% of compositions land at κ=0** across the test suite. The κ=±1 cases are minority but structurally significant (they're the cases where T-correction fires — the Exception substantiates at the composition step).

This statistic is the d-composition analog of Audit 04 Gap κ.a (κ statistics from Identity A): the κ=0 rate IS the rate at which Mediation suffices. For Identity C this rate is observable directly in the comp_table_κ=0 vs comp_table sizes.

---

## §B. Present Uses in `eudd_poc.py`

### B.1 `residue_set(d, N_res=12)` — Definition C.1 implementation (lines 2053–2056)

```python
# --- Identity C: Residue Sets ---
def residue_set(d, N_res=12):
    """Res_N(d) = {k mod N : N/gcd(k,N) = d}."""
    return [k for k in range(N_res) if (N_res // gcd(k, N_res) if k != 0 else 1) == d]
```

**Math:** matches Definition C.1 byte-for-byte with the `k=0 ⟹ gcd → N` convention from the projection formula (Definition 7.1). ✓

**Implementation observations:**
- Pure enumeration; full `[0, N)` range scanned per call.
- `N_res` default is 12 but accepts any N (used at higher N in audit 04's proposed §6.8 fix and resolution plan §6.8).
- **Not memoized.** Each invocation re-enumerates the entire range. For N=12 this is 12 iterations; for the resolution-plan's `N_native = 27720`, this would be 27720 iterations per call.

### B.2 Use sites of `residue_set`

| Site | Line | Context | What's tested |
|---|---|---|---|
| `verify_identities_on_data` Identity C block | 2156–2163 | Verifies `|Res(d)| = φ(d)` for each `d ∈ divisors(N)` at N=12 | **Only the totient cardinality** (Corollary 8.4) — no other C-theorem tested |
| `verify_identities_on_data` Identity H tensor | 2213–2228 | Computes partition-of-unity `Σ T(d₁, d₂; d₃) = 1` over `d₃ ∈ divisors(N)` at **κ=0 only** (line 2224's formula has no κ term) | Implicitly tests C.2 SUPPORT-set at κ=0 via the partition-of-unity check |
| `_enrich_with_tensor` | 2547–2548 | Computes `T_κ(d₁, d₂; d_result)` with `kappa_act` ∈ {−1, 0, +1} | Per-call computation of C.2's κ-augmented composition probability |

**Coverage in built-in verification:** **1 of 7 structural properties** of Identity C is tested (totient cardinality only). C.2 (composition sets), C.3 (symmetry), C.4 (d=1 channel), C.5 (d=12 universality), C.6 (lcm bound with κ-correction), the division/multiplication set equality, the power d-table, and the κ=0 dominance statistic are all **untested**. See Gap C.cov.

### B.3 `_enrich_with_tensor` — partial Identity C.2 usage (lines 2540–2566)

```python
def _enrich_with_tensor(d1, d2, d_result, kappa_act):
    divisors_12 = [1, 2, 3, 4, 6, 12]
    if d1 not in divisors_12 or d2 not in divisors_12 or d_result not in divisors_12:
        return {}
    res1 = residue_set(d1, N_BASE)
    res2 = residue_set(d2, N_BASE)
    total_pairs = len(res1) * len(res2)
    if total_pairs == 0:
        return {}
    hits = sum(1 for r1 in res1 for r2 in res2
               if (N_BASE // gcd((r1 + r2 + kappa_act) % N_BASE, N_BASE)
                   if (r1 + r2 + kappa_act) % N_BASE != 0 else 1) == d_result)
    t_kappa = mpf(str(hits)) / mpf(str(total_pairs))
```

**This is Identity C.2 in probabilistic form.** The composition SUPPORT is `{d_result : hits > 0}`; the probability is `hits / total_pairs`. The composition is computed **per call** rather than precomputed.

**Connection to C.2:** the boolean predicate inside the `sum(...)` exactly computes membership of `d_result` in the κ-restricted slice of `d₁ ⊗ d₂`. So `_enrich_with_tensor` IS using Identity C.2, but:
1. As a **probability** (transfer-tensor entry) rather than a **set** (composition support);
2. **Per-call recomputation** rather than precomputed table lookup;
3. Filtered to **simple families only** (Audit 04 §T3 / Resolution Plan §6.8 — restricted to divisors of 12, not all 12 harmonic families; this is the same gap).

**Verdict:** C.2 is used in probability form for one specific subsystem (Identity H tensor). The set-valued composition itself is never directly available as a structural object.

### B.4 `find_structural_relations` — implicit C.2 use without precomputed support (lines 2569–2637)

```python
for sha_a, ea in entries_items:
    for kappa in [-1, 0, 1]:
        k_b_needed = k_new - ea['k_12'] - kappa
        if k_b_needed in k_index:
            for sha_b in k_index[k_b_needed]:
                eb = existing_entries[sha_b]
                k_prod, d_prod, eps_prod, kappa_actual = lattice_multiply(...)
                if k_prod == k_new and d_prod == d_new:
                    ...
```

The discovery engine searches every `(sha_a, sha_b)` pair against the new entry. **No filtering by `d_new ∈ d_a ⊗ d_b`.** Per Identity C.2, many `(d_a, d_b, d_new)` triples are structurally **impossible** — yet the search wastes work testing them via `lattice_multiply`.

**Cross-reference:** Audit 04 Gap A.6.b proposed `D_PRODUCT_SUPPORT[(d₁, d₂)]` filter. This is exactly Identity C.2's composition set. Audit 05 confirms the proposal AND extends it with κ-awareness: the filter should be:
- For κ=0 candidates: `d_new ∈ d_a ⊗₀ d_b` (tight, κ=0 sub-table)
- For κ=±1 candidates: `d_new ∈ d_a ⊗ d_b` (broader, full table)

### B.5 `divisors_12` enumeration (line 2157, 2178, 2206, 2222, 2543)

Five occurrences of `divisors_12 = [1, 2, 3, 4, 6, 12]` (either hardcoded or computed). This is the sublattice-family enumeration at N=12 (Corollary 8.2 / Resolution Plan §5.9). The list IS correct *as the divisors of 12*; the cosmetic gap (Resolution Plan T3b) replaces literal with `sympy.divisors(N_BASE)`. **No Identity C theorem is directly tested** at these sites — they're enumeration scaffolding.

### B.6 Summary of Present Uses

| Identity | Implementation | Use sites | Verification |
|---|---|---|---|
| C.1 Residue Set definition | `residue_set` (2054) ✓ correct | 3 sites (verify_C, verify_H, _enrich_with_tensor) | `|Res(d)| = φ(d)` only |
| C.2 d-composition set-valued | Implicit via `_enrich_with_tensor` (probability form per call) | 1 site (probabilistic) | Partition-of-unity at κ=0 (verify_H) |
| C.3 Residue symmetry | NOT exploited | — | NOT tested |
| C.4 d=1 self-composition channel | NOT surfaced | — | NOT tested |
| C.5 d=12 universality | NOT surfaced | — | NOT tested |
| C.6 lcm bound with κ-correction | NOT asserted (audit 04 A.6.a proposal needs correction) | — | NOT tested |
| Division = Multiplication (set) | NOT exploited | — | NOT tested |
| Power d-family table | `lattice_power` correctly computes per-call (audit 04 §B); table NOT precomputed | — | NOT tested |
| κ=0 dominance statistic | NOT surfaced | — | NOT tracked (cross-ref audit 04 κ.a) |

**Identity C is sparsely integrated.** The residue-set definition is implemented and used correctly; C.2 is partially used (as probability) in one subsystem; **the remaining five theorems plus three bonus identities are absent or unused.**

This is the lowest level of integration of any identity audited so far (compare: Identity #0 fully integrated; Identity A fully integrated; Identity B mostly integrated; Identity X-Res partially integrated; Identity C sparsely integrated).

---

## §C. Gaps — Beneficial Uses NOT Realized

### Gap C.cov — Built-in verification covers only `|Res(d)| = φ(d)`

**Severity:** coverage gap.
**Site:** `verify_identities_on_data` lines 2156–2163.

Only 1 of 7+ structural properties tested. The source script's PART 4 tests:
1. Totient sum `Σ φ(d) = N`
2. Residue set symmetry (C.3)
3. d=1 self-composition channel (C.4)
4. d=12 universality (C.5)
5. Commutativity (C.3 corollary)
6. d=1 as κ=0 identity element
7. lcm bound at κ=0 (C.6 κ=0 case)
8. lcm bound with κ-correction violations (C.6 counterexamples)
9. Computational verification across 12 test reals × all (d₁,d₂) pairs

The EUDD tests only (1) at line 2162. **The other eight are entirely untested.**

**Fix:** add a full C-block to `verify_identities_on_data` that runs PART 4 of the source script as a self-contained verification. The composition table is finite (6×6 = 36 cells at N=12), the test cost is negligible.

**Anchor:** every identity should be subsumed by the verification (Subsumption Law) — `verify_identities_on_data` is the operational expression of that subsumption for the EUDD.

---

### Gap C.1.a — `residue_set` not memoized as an Identity J seed

**Severity:** capability lift (cross-reference Audit 01 X-Res.cross, Audit 02 #0.g, Audit 04 A.6.b).
**Site:** `residue_set` (line 2054).

Per Identity J (the EUDD is a birth triad; every computation IS a seed), `residue_set(d, N)` is a **deterministic pure function** of `(d, N)`. Its output `Res_N(d)` is a memoizable structural DSR. The current implementation re-enumerates every call.

**At N=12, six (d, N) pairs cover all sublattice families — the residue-set CATALOG is six entries.** At higher N (60, 420, 27720), the catalog is larger but still finite. Memoizing the catalog as CONTENT_EQUATION seeds in the akashic means:
1. Subsequent `residue_set(d, N)` calls become O(1) lookups.
2. The catalog persists across sessions (akashic loads with it).
3. The catalog itself becomes a structural object that can participate in discovery (the residue set's own bytes have a DSR — Identity J self-application).

**Fix:** wrap `residue_set` with `eq_lookup` / `eq_store` analogous to `project`, `pullback`, `lattice_multiply`. Each `(d, N)` → list of residues becomes a CONTENT_EQUATION seed.

**Anchor:** Identity J + Identity C.1.

---

### Gap C.2.a — `comp_table[(d₁, d₂)]` set-valued composition table not precomputed

**Severity:** structural — extends and refines Audit 04 Gap A.6.b.
**Site:** `find_structural_relations` (line 2569, 2599); `_enrich_with_tensor` (line 2540–2566).

The full set-valued composition table at N=12 is a 6×6 → set-of-d-values structural object, finite, deterministic, and **precomputable once at module init**.

**Two tables are useful (distinct structural information):**

1. **`COMP_TABLE_KAPPA0[(d₁, d₂)]`** — the κ=0 sub-table. By Theorem C.6's κ=0 case, `d_product ≤ lcm(d₁, d₂)` bounds this. This is the **Mediation-sufficient** composition layer (~79% of cases per script's PART 4). When the discovery engine considers a candidate `(sha_a, sha_b)` with `κ=0`, the d-target MUST be in this set.

2. **`COMP_TABLE_FULL[(d₁, d₂)]`** — the full κ-augmented table. By Theorem C.6's κ≠0 case, this can include d-values exceeding `lcm(d₁, d₂)`. When the discovery engine considers κ=±1 candidates, the d-target must be in this (broader) set.

Both tables fit in tiny constant memory (36 cells × ≤6 entries = ≤216 d-values at N=12). Computed once at module init from `residue_set` + iterative κ-iteration. Stored as CONTENT_EQUATION seeds in the akashic (Identity J self-application).

**Refinement of Audit 04 A.6.b:** that audit proposed `D_PRODUCT_SUPPORT[(d₁, d₂)] = set of achievable d_products`. Audit 05 confirms this and **splits it into κ=0 and full-κ variants**, each with its own structural meaning.

**Beneficial uses:**
1. **Discovery filter** in `find_structural_relations`: for each iterated κ ∈ {−1, 0, +1}, check `d_new ∈ COMP_TABLE_KAPPA0[(d_a, d_b)]` (for κ=0) or `d_new ∈ COMP_TABLE_FULL[(d_a, d_b)]` (for κ≠0) before invoking `lattice_multiply`. O(1) filter cuts infeasible candidates.
2. **Tensor partition-of-unity sanity check** in `_enrich_with_tensor`: the SUM of `T_κ(d₁, d₂; d_result)` over `d_result ∈ COMP_TABLE_FULL[(d₁, d₂)]` must equal 1. Direct test of the residue-set arithmetic's completeness.
3. **Structural DSR storage:** each table cell is itself a small set with a deterministic byte representation → projectable seed.

**Anchor:** Identity C.2 + Identity J + Audit 04 A.6.b refinement.

---

### Gap C.2.b — κ=0 dominance not surfaced as a structural metric

**Severity:** observability (cross-reference Audit 04 κ.a).
**Site:** `eq_metrics()` (lines 319–332).

Audit 04 Gap κ.a proposed tracking per-operation κ counts in `eq_metrics`. Audit 05 extends this: **at the d-family layer**, the ratio `|COMP_TABLE_KAPPA0[(d₁, d₂)]| / |COMP_TABLE_FULL[(d₁, d₂)]|` per cell IS the κ=0 dominance for that specific input pair. Aggregated, this is the Mediation-sufficiency rate.

The script's PART 4 reports ~79% κ=0 across all (d₁, d₂) pairs at N=12. This number is a structural constant of the lattice at N=12 — computable from the tables alone, no runtime sampling needed.

**Fix:** add to `eq_metrics`:
```python
mediation_sufficiency_d_family = sum(
    1 for d1, d2 in COMP_TABLE_FULL if COMP_TABLE_KAPPA0[(d1,d2)] == COMP_TABLE_FULL[(d1,d2)]
) / len(COMP_TABLE_FULL)
```
This is the **fraction of (d₁, d₂) input pairs for which κ-correction adds no new output families** — the structural Mediation-sufficiency rate at the d-family layer. A standing structural metric (Rule 28).

**Anchor:** Identity C.6 (κ=0 vs κ≠0 distinction) + Theorem 15.1 (κ is the T-act).

---

### Gap C.3.a — Residue symmetry not exploited in `residue_set` enumeration

**Severity:** capability lift (cosmetic optimization).
**Site:** `residue_set` (line 2056).

Theorem C.3 proves `k ∈ Res(d) ⟹ (N − k) ∈ Res(d)`. The current implementation enumerates the full range `[0, N)`. By symmetry, **enumerating `[0, N/2]` and reflecting gives the same result** with half the work:

```python
def residue_set(d, N_res=12):
    """Res_N(d) = {k mod N : N/gcd(k,N) = d}, exploiting C.3 symmetry."""
    half = N_res // 2
    base = [k for k in range(half + 1) if (N_res // gcd(k, N_res) if k != 0 else 1) == d]
    # Reflect: for each k in [0, half], if N-k ≠ k and N-k ∈ Res(d), add it
    mirrored = [N_res - k for k in base if 0 < N_res - k < N_res and N_res - k != k]
    return sorted(set(base + mirrored))
```

**Caveat:** at N=12 this is a tiny optimization (12 iterations vs 7); at N=27720 (resolution plan §6.8) it's a factor-of-2 win on a 27720-iteration scan. Not critical for current scope, but principled.

**Anchor:** Identity C.3.

---

### Gap C.3.b — Residue symmetry not asserted as a structural postcondition

**Severity:** defensive correctness (cross-reference Audit 04 A.1.a pattern).
**Site:** `residue_set` (line 2056).

Symmetry is structural: any candidate residue set returned by `residue_set` MUST satisfy `k ∈ S ⟹ (N − k) mod N ∈ S`. The current implementation does not assert this as a postcondition. A bug that filtered residues asymmetrically would not be caught.

**Fix:** add after the return list construction:
```python
result = [k for k in range(N_res) if (N_res // gcd(k, N_res) if k != 0 else 1) == d]
assert all((N_res - k) % N_res in result for k in result), (
    f"C.3 violation: Res_{N_res}({d}) = {result} is not symmetric under k → N−k"
)
return result
```
Standing structural guard.

**Anchor:** Identity C.3 + Rule 14 (no silent failure).

---

### Gap C.4.a — d=1 self-composition channel not exploited in discovery

**Severity:** capability lift (structural shortcut).
**Site:** `find_structural_relations` (lines 2569–2588).

Theorem C.4 guarantees that **any d=1 entry (`true_home` per current EUDD classification) is structurally available as the self-product of mirror-partner pairs from any family**. Specifically:
- Given new entry with `d_new = 1`, the discovery engine should FIRST check whether `k_new` arises as `k_a + (−k_a) = 0` (mod N) for some existing pair `(sha_a, sha_b)` with `k_b = N − k_a` (or `k_b = −k_a`) and `d_a = d_b`.
- This is the structural test for the d=1 channel: if `d_new = 1`, the mirror-product test is the cheapest candidate.

**Cross-reference:** Audit 04 Gap A.6.c proposed `family_collision_multiply` events when `d_a = d_b` and `d_new != d_a`. Audit 05 sharpens this: the **specific case** of `d_a = d_b` and `d_new = 1` IS the C.4 channel firing — a distinct event class `d1_self_composition_channel_discovered`.

**Fix:** add to `find_structural_relations` a fast pre-pass for `d_new = 1` entries: check existence of `sha_a` with `k_a` such that `k_a + (N − k_a) + κ = k_new` for some κ ∈ {−1, 0, +1}. Hit ⟹ direct C.4 channel realization.

**Anchor:** Identity C.4.

---

### Gap C.4.b — d=1 channel not tested in `verify_identities_on_data`

**Severity:** coverage gap.

For every `d ∈ divisors(N_BASE)`, the script's PART 4 verifies `1 ∈ comp_table[(d, d)]`. The EUDD does not.

**Fix:** add to C-block:
```python
for d in divisors_12:
    # Build comp_table[(d, d)] on the fly
    res_d = residue_set(d, N)
    comp_set = set()
    for r1 in res_d:
        for r2 in res_d:
            for kappa in [-1, 0, 1]:
                s = (r1 + r2 + kappa) % N
                comp_set.add(N // gcd(s, N) if s > 0 else 1)
    check(f"C.4 d=1 channel: 1 ∈ {d}⊗{d}", 1 in comp_set)
```
Standing structural assertion.

**Anchor:** Identity C.4.

---

### Gap C.5.a — d=12 universal-mixer classification absent

**Severity:** structural classification.
**Site:** `home_classification` field; `anti_numerology_check`; `eq_metrics`.

Theorem C.5: `d=12` entries are **universal mixers** — they can compose with another d=12 entry to produce ANY sublattice family. This is operationally distinct from other families (which produce restricted output sets per C.2).

The EUDD's `home_classification` taxonomy (lines 1189–1196: `true_home`, `deep_home`, `persistent_home`, `intermediate_home`, etc.) does not include a "universal mixer" classifier. A d=12 entry passing `home_classification = 'intermediate_home'` is missing a structural label that C.5 supplies.

**Fix:** when classifying home, add a tag for d=12 entries: `universal_mixer = True`. Track in metrics. Optionally surface in event class `d12_universal_mixer_discovered`.

**Anchor:** Identity C.5.

---

### Gap C.5.b — `find_structural_relations` filter short-circuit absent for (12, 12) inputs

**Severity:** filter-edge-case efficiency.
**Site:** the proposed Gap C.2.a filter.

By Theorem C.5, `COMP_TABLE_FULL[(12, 12)] = {1, 2, 3, 4, 6, 12}` — every sublattice family. The Gap C.2.a filter `d_new ∈ COMP_TABLE_FULL[(d_a, d_b)]` is **vacuous** for `(d_a, d_b) = (12, 12)` (since the set is everything).

**Fix:** when implementing Gap C.2.a's filter, short-circuit for `(d_a, d_b) = (12, 12)` (don't filter — pass through). Equivalently, skip the filter when `len(COMP_TABLE_FULL[(d_a, d_b)]) == len(divisors(N))`.

**Anchor:** Identity C.5.

---

### Gap C.5.c — d=12 universality not tested in `verify_identities_on_data`

**Severity:** coverage gap.

**Fix:** add to C-block:
```python
res_12 = residue_set(12, N)
comp_set_12 = set()
for r1 in res_12:
    for r2 in res_12:
        for kappa in [-1, 0, 1]:
            s = (r1 + r2 + kappa) % N
            comp_set_12.add(N // gcd(s, N) if s > 0 else 1)
check(f"C.5 d=12 universality: 12⊗12 = all families",
      comp_set_12 == set(divisors_12))
```

**Anchor:** Identity C.5.

---

### Gap C.6.a — **CORRECTION TO AUDIT 04 GAP A.6.a**

**Severity:** **Material correction to a prior audit's proposed fix.**

**Audit 04 Gap A.6.a proposed:**
```python
assert d_prod <= math_lcm(d1_fam, d2_fam), (
    f"A.6 violation: d_product={d_prod} > lcm(d₁={d1_fam}, d₂={d2_fam}) = {math_lcm(d1_fam, d2_fam)}"
)
```

**This is FALSE for κ ≠ 0** per Theorem C.6 of the script. Counterexample:
- `(k₁, ε₁) = (0, +49¢)`, so d₁ = 1
- `(k₂, ε₂) = (0, +49¢)`, so d₂ = 1
- `δ₁ + δ₂ = 49·12/1200 + 49·12/1200 = 0.49 + 0.49 = 0.98`
- `κ = round(0.98) = +1`
- `k_product = 0 + 0 + 1 = 1`
- `d_product = 12 / gcd(1, 12) = 12 / 1 = 12`
- But `lcm(1, 1) = 1`. **The proposed assertion fires.**

This is a **legal, correct, exact** outcome of `lattice_multiply`. The proposed assertion would incorrectly raise on a valid call.

**Corrected assertion:**
```python
if kappa == 0:
    assert d_prod <= math_lcm(d1_fam, d2_fam), (
        f"C.6 κ=0 violation: d_product={d_prod} > lcm(d₁={d1_fam}, d₂={d2_fam}) "
        f"= {math_lcm(d1_fam, d2_fam)} (Identity C.6 κ=0 case)"
    )
else:
    # For κ ≠ 0: only the universal bound d_product | N holds (trivial)
    assert N % d_prod == 0, (
        f"d_product={d_prod} does not divide N={N} (lattice violation)"
    )
```

**Status:** Audit 04 Gap A.6.a as worded is **rescinded**; Audit 05 Gap C.6.a is the correct form. The composite fix proposed in Audit 04 Gap CROSS.d (`_assert_identity_a_postconditions` helper) must use the κ-conditional version.

**Anchor:** Theorem C.6 of the source script.

---

### Gap C.6.b — κ-induced lcm violation not tested as a structural counterexample

**Severity:** coverage gap. Audit 04 A.6.a missed this case entirely.

The κ-induced violation is a STRUCTURAL prediction of Identity C.6: the lcm bound MUST fail in some κ≠0 cases. Testing for this confirms the κ-correction is genuinely non-trivial.

**Fix:** add to C-block:
```python
# Specific C.6 counterexample: k=0 + k=0 + κ=+1 → k_product=1, d_product=12
k_test_1 = mpf('1')      # d_1 = 1 (input)
eps_test_1 = mpf('49')    # δ = 0.49, just below cell boundary
k_a, d_a, eps_a, kappa = lattice_multiply(0, eps_test_1, 0, eps_test_1, N)
expected_kappa_fires = (kappa == 1 and d_a == 12)
check(f"C.6 κ-induced lcm violation: (d=1)·(d=1) → d_prod=12 when κ=+1",
      expected_kappa_fires)
```

**Anchor:** Identity C.6 (κ≠0 case proof).

---

### Gap C.6.c — Audit 04 A.6.b filter needs κ-awareness

**Severity:** correction to Audit 04 Gap A.6.b.

Audit 04 Gap A.6.b proposed a filter `D_PRODUCT_SUPPORT[(d₁, d₂)]` for `find_structural_relations` multiply search. **Per Theorem C.6, the filter must distinguish κ=0 from κ≠0**:

```python
for sha_a, ea in entries_items:
    for kappa in [-1, 0, 1]:
        # κ-aware filter (C.6 correction to audit 04 A.6.b)
        if kappa == 0:
            if d_new not in COMP_TABLE_KAPPA0[(ea['d_12'], <implied d_b>)]:
                continue  # infeasible at κ=0
        else:
            if d_new not in COMP_TABLE_FULL[(ea['d_12'], <implied d_b>)]:
                continue  # infeasible
        # ... proceed with k_b_needed lookup and lattice_multiply trial ...
```

The two tables (κ=0 and full) are the right structural objects, not one combined table.

**Anchor:** Theorem C.6 + Audit 04 A.6.b.

---

### Gap C.div.a — Division-multiplication d-set equality not exploited

**Severity:** discovery optimization.
**Site:** `find_structural_relations` (multiply at line 2569, divide at line 2591 — two separate loops).

By the Division=Multiplication bonus identity, **the d-OUTPUT space for `lattice_divide(a, b)` is identical to `lattice_multiply(a, b)`**. The actual k values differ (subtract vs add), but the d-family target sets are equal.

Currently `find_structural_relations` runs **separate loops** for multiply and divide, each with its own k_index lookup. The d-family filter (Gap C.2.a / C.6.c) applies to both with the same table.

**Beneficial use:** consolidate the multiply and divide d-feasibility checks. Each pair `(sha_a, sha_b)` is checked once against the unified d-set; the k-direction (sum vs difference) is determined by which arithmetic produces the matching k_new. This halves the d-filter work.

**Caveat:** the k_new = k_a + k_b + κ (multiply) vs k_new = k_a − k_b + κ (divide) ARE different k-targets, so the k_index lookup is still per-operation. But the **d-filter precheck** is shared.

**Anchor:** Division=Multiplication bonus identity (C.3 corollary).

---

### Gap C.div.b — Division=Multiplication set equality not tested

**Severity:** coverage gap.

**Fix:** add to C-block:
```python
div_eq_mult = True
for d1 in divisors_12:
    for d2 in divisors_12:
        res1 = residue_set(d1, N)
        res2 = residue_set(d2, N)
        mult_set = set()
        div_set = set()
        for r1 in res1:
            for r2 in res2:
                for kappa in [-1, 0, 1]:
                    sm = (r1 + r2 + kappa) % N
                    sd = (r1 - r2 + kappa) % N
                    mult_set.add(N // gcd(sm, N) if sm > 0 else 1)
                    div_set.add(N // gcd(sd, N) if sd > 0 else 1)
        if mult_set != div_set:
            div_eq_mult = False
check(f"C.div: d₁⊘d₂ = d₁⊗d₂ as sets (residue symmetry)", div_eq_mult)
```

**Anchor:** C.3 corollary (Division=Multiplication identity).

---

### Gap C.pow.a — Power d-family table not precomputed; cycle structure not surfaced

**Severity:** capability lift + structural connection to Identity B.3.
**Site:** `lattice_power` (line 2029) — computes per-call.

The power d-family table at N=12 is `12 × 12 = 144` entries (for d ∈ divisors_12 × n ∈ [1, 12]), each entry being a small set (most are singletons by the script's PART 7 observation). Precomputable at startup. Storable as a CONTENT_EQUATION seed.

**The d=12 power orbit is structurally identical to the harmonic cascade.** From the script's PART 7:
```
d=12 power orbit (n=1..12):  [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]
Harmonic cascade (from B.3):  [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]   ← IDENTICAL
```

This is a meaningful structural connection: at N=12, the sublattice-power orbit of d=12 traces the harmonic cascade. The relationship is mediated by the Sublattice Visitation Theorem, but the orbit emerges purely from `n · Res(12) mod 12` gcd-classification.

**Beneficial uses:**
1. Precompute `POWER_D_TABLE[(d, n)]` at module init; store as a CONTENT_EQUATION seed.
2. Use as a structural classifier: if an entry's d traces the d=12 orbit under successive squaring (via lattice_power), it's structurally tied to the harmonic cascade. Cross-reference Audit 03 Gap B.c (sublattice palindrome class labeling).
3. Detect d-family fixed points: entries whose `d(rⁿ) = d` for some n are "self-similar at order n" — a structural classification.

**Anchor:** Power d-family bonus identity + Identity B.3 cascade reference + Identity J.

---

### Gap C.pow.b — Power d-cycle not used for d-family identification of unknown entries

**Severity:** capability lift.
**Site:** discovery engine; potential anti_numerology extension.

If an entry has `d = 6` (an "unknown" entry from the perspective of family-identification), the power-d-orbit can pin down WHICH d=6 family it is: by powering `r → r²`, the orbit's continuation traces a specific path. If the trace matches the d=12 cascade-path (`6 → ... ` per the table), the entry was originally d=12 squared. If it traces a different path, it has a different originating family.

This is a fingerprinting test: the d-family alone doesn't identify the structural class; the **d-power orbit** does.

**Caveat:** computationally substantial (each fingerprint test runs N=12 sequential `lattice_power` calls). Mark as opt-in discovery extension.

**Anchor:** Power d-family bonus identity.

---

### Gap C.reach.a — Per-family reachability classification absent

**Severity:** structural classification (mostly informational).
**Site:** none — `reachability` is not a concept in the EUDD.

The script's PART 6 computes: for each d, which families are reachable via composition with ANY other family. At N=12 with κ-augmentation, every family becomes universal (reaches all six). At κ=0, the reachability is **restricted** — gives a stratification of families by κ=0 expressive power.

**Beneficial uses:**
1. **Akashic-level metric:** "expressive reach" per family — how many distinct d-output families each ingested entry can participate in composing with others.
2. **Discovery sequencing:** prioritize composing high-reach families first (they yield more candidates per call).
3. **Cross-reference Audit 04 Gap A.6.b filter pruning:** entries from low-reach families have smaller filter sets.

**Anchor:** Reachability bonus property.

---

## §D. Summary

| Identity / Property | Implementation in `eudd_poc.py` | Verdict |
|---|---|---|
| C.1 — Residue Set definition | `residue_set` (2054); algebraically correct; full enumeration per call | ✓ Math; **not memoized (C.1.a)** |
| C.2 — d-Composition Set-Valued | Implicit per-call probability via `_enrich_with_tensor`; not precomputed as a SET object | **Not precomputed as table (C.2.a)**, **κ=0 sub-table not separated (C.2.b cross-ref κ.a)** |
| C.3 — Residue Symmetry | Not exploited; not asserted | **Not exploited in enumeration (C.3.a)**, **not asserted (C.3.b)** |
| C.4 — d=1 Self-Composition Channel | Not surfaced | **Not exploited in discovery (C.4.a)**, **not in verify (C.4.b)** |
| C.5 — d=12 Universality | Not surfaced | **Universal-mixer classification absent (C.5.a)**, **filter short-circuit absent (C.5.b)**, **not in verify (C.5.c)** |
| C.6 — lcm Bound with κ-correction | Not asserted; Audit 04 A.6.a's proposal is WRONG as worded | **CORRECTION TO AUDIT 04 (C.6.a)**, **κ-induced violation not tested (C.6.b)**, **Audit 04 A.6.b filter needs κ-awareness (C.6.c)** |
| Division = Multiplication (set equality) | Not exploited | **Discovery search optimization missed (C.div.a)**, **not in verify (C.div.b)** |
| Power d-family Table | Per-call computation; cycle structure not surfaced | **Not precomputed (C.pow.a)**, **cycle not used for identification (C.pow.b)** |
| κ=0 dominance statistic | Not tracked | **Cross-reference Audit 04 κ.a (C.2.b)** |
| Reachability classification | Not present | **Absent (C.reach.a)** |
| Built-in verification coverage | Only `|Res(d)| = φ(d)` tested | **1 of 7+ structural properties (C.cov)** |

**Identity C is the most sparsely-integrated identity audited so far.**

In numerical terms across the audits:
| Audit | Identity | Theorems | Theorems used as tools | Verification coverage |
|---|---|---|---|---|
| 02 | #0 Bijection | 1 | 1 (heavily) | round-trip + several code-integrity gaps |
| 03 | B Differential | 6 | 3 (B.1 verify only, B.2a in regenerate only, B.5 constant) | 1 of 6 tested rigorously |
| 04 | A Lattice Arith | 6 | 4 (A.1–A.4 in 6 subsystems each) | 4 of 6 with κ-bound checks |
| 01 | X-Res Cross-Resolution | 5 | 1 (X-Res.1 in 2 sites) | partial |
| **05** | **C d-Family Composition** | **6 + bonus** | **0** (C.1 used; C.2 only as probability via _enrich; C.3–C.6 unused) | **1 of 7+ tested (totient only)** |

Identity C is **conceptually central** (it's what makes the sublattice-family layer operational — the COMPLETE algebraic structure of how families compose) yet **operationally absent** from the EUDD beyond residue-set enumeration and one tensor probability calculation. This is a substantial gap-cluster.

**Material finding flagged to prior work:** Theorem C.6 corrects Audit 04 Gap A.6.a. The proposed `assert d_product ≤ lcm(d₁, d₂)` in Audit 04 fires false positives for κ ≠ 0 multiplications. The κ-conditional form is in §C.6.a. The Audit 04 CROSS.d composite-postcondition helper must use the corrected form.

---

## §E. Proposed Action

### Audit 04 corrections (urgent — prevents incorrect assertions in proposed fixes)

- **C.6.a** (rescinds Audit 04 A.6.a; ratifies Audit 04 CROSS.d composite helper conditional on this fix). Use κ-conditional form: `assert d_prod ≤ lcm` only when `κ == 0`; otherwise only the universal `d_product | N`.
- **C.6.c** (refines Audit 04 A.6.b). The proposed filter must distinguish κ=0 from κ≠0 — TWO tables, not one.

### Defensive-correctness fixes (Subsumption Law postconditions)

- **C.3.b** — add residue-set symmetry assertion at end of `residue_set`.
- **C.6.a/C.6.c** (above).

### Categorical structural detection (parallel to Audit 04's A.3 mirror property family)

- **C.4.a** — fast pre-pass in `find_structural_relations` for d_new=1 entries: try mirror-product pairs first.
- **C.5.a** — add `universal_mixer` classifier for d=12 entries in `home_classification` and `eq_metrics`.
- **C.5.b** — short-circuit filter for `(d=12, d=12)` input pairs (filter is vacuous).

### Structural table memoization (Identity J self-application, cross-reference Audit 01 X-Res.cross, Audit 02 #0.g, Audit 04 A.6.b)

- **C.1.a** — memoize `residue_set(d, N)` via `eq_lookup` / `eq_store`. Six entries at N=12; persists across sessions.
- **C.2.a** — precompute and store `COMP_TABLE_KAPPA0[(d₁, d₂)]` AND `COMP_TABLE_FULL[(d₁, d₂)]` at module init. Two CONTENT_EQUATION seeds.
- **C.pow.a** — precompute `POWER_D_TABLE[(d, n)]` table.

### Search optimization

- **C.div.a** — consolidate d-feasibility precheck across multiply/divide search loops in `find_structural_relations` (same d-set).
- **C.6.c** — use κ-conditional filter in discovery (above).
- **C.4.a** — fast pre-pass for d_new=1 (above).

### Observability

- **C.2.b** — surface κ=0 dominance rate at the d-family layer in `eq_metrics` (Mediation-sufficiency rate per Theorem 15.1).

### Coverage breadth

- **C.cov** — add full C-block to `verify_identities_on_data`:
  - C.3 symmetry (per-d residue mirror check)
  - C.4 d=1 channel (per-d self-composition)
  - C.5 d=12 universality (12⊗12 = all)
  - C.6 κ=0 lcm bound (held)
  - C.6 κ≠0 lcm bound violation (counterexample fires)
  - C.div division=multiplication set equality
  - d=1 as κ=0 identity element
  - Commutativity

### Power d-family identification (opt-in)

- **C.pow.b** — d-power-orbit fingerprinting for family identification. Computationally substantial; opt-in flag.

### Reachability metric (low priority, informational)

- **C.reach.a** — per-family reachability classification in `eq_metrics`.

### Cross-references with prior audits

| Audit 05 gap | Audit 01–04 cross-reference |
|---|---|
| C.1.a (residue_set memoization) | Audit 01 X-Res.cross, Audit 02 #0.g, Audit 04 A.6.b — same Identity J self-application |
| C.2.a (comp table precompute) | **Refines Audit 04 A.6.b** with κ-awareness; same Identity J target |
| C.2.b (κ=0 dominance) | **Extends Audit 04 κ.a** to the d-family layer |
| C.3.a/b (symmetry) | **Mirrors Audit 04 A.3.a** (mirror property at the bijection layer; same structural principle at the residue-set layer) |
| C.4.a (d=1 channel) | **Sharpens Audit 04 A.6.c** family-collision events to the specific `d=1` case |
| C.6.a (lcm with κ-correction) | **CORRECTS Audit 04 A.6.a** (composite postcondition CROSS.d must use this form) |
| C.6.c (κ-aware filter) | **Refines Audit 04 A.6.b** filter |
| C.pow.a (power d-table) | **Connects to Audit 03 B.c** (sublattice palindrome) — the d=12 power orbit IS the harmonic cascade |
| C.cov | Pattern matches Audit 04 A.cov, Audit 03 B.e/B.f (verification breadth) |

### Priority classification

- **High priority (corrections to active proposals):** C.6.a (Audit 04 A.6.a correction), C.6.c (Audit 04 A.6.b refinement).
- **High priority (Subsumption-Law completeness — only 1 of 7+ theorems verified):** C.cov.
- **High priority (defensive correctness):** C.3.b.
- **Medium priority (structural memoization, Identity J):** C.1.a, C.2.a, C.pow.a.
- **Medium priority (capability lifts):** C.4.a, C.5.a/b, C.div.a.
- **Medium priority (observability):** C.2.b, C.5.a, C.reach.a.
- **Low priority (cosmetic optimization):** C.3.a.
- **Opt-in (computational cost):** C.pow.b.

---

## §F. Identification Principle Closure

Applied to this audit:
- **P_audit:** the d-family composition surface in `eudd_poc.py` — identified as `residue_set` (1 function, 4 use sites), `_enrich_with_tensor` (probabilistic C.2 use), `verify_identities_on_data` C-block (totient-only coverage), `find_structural_relations` (Identity C not invoked as filter).
- **D_audit:** Definition C.1 plus theorems C.2 through C.6, plus the three bonus identities (Division=Multiplication, Power d-table, κ=0 dominance / d=1 κ=0 identity element / Reachability).
- **T_audit:** the traversal from each theorem/property to each implementation site, noting present uses (§B) and absent beneficial uses (§C).

**Subsumption Law check on the audit itself:**
- Every theorem traced through every subsystem. ✓
- Every absent beneficial use documented. ✓
- A material finding (C.6 correcting Audit 04 A.6.a) surfaced and integrated. ✓
- No remainder.

**Descriptor Gap Principle on the audit's own finding:** the κ-conditional lcm-bound (Gap C.6.a) was a Descriptor missing from the previous audit's proposal. The proposal failed (would have fired false positives); the failure IS the gap; the gap IS the κ-conditional form. Now closed.

**The C.6 correction is the most important finding of this audit.** Implementing Audit 04 Gap CROSS.d (the composite postcondition helper) without this correction would have introduced a regression into production code. Audits compounding rigorously over the identity sequence catch such issues before they reach the code.

---

**Document version:** 1.0 — Identity Audit 05 of 15
**P ∘ D ∘ T = E**
**For every exception there is an exception, except the exception.**

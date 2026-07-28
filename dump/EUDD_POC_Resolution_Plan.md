# EUDD Proof-of-Concept — Comprehensive Resolution Plan

**Subject:** `eudd_poc.py` — TypeError at line 1457 (`base['eps_12']` is `None`) plus complete audit of related defects.
**Standard:** All fixes ET-native, forward from `P ∘ D ∘ T = E`. No placeholders, no shortcuts, no tuning, no caps, nothing static unless explicitly an ET constant.
**Anchoring identities:** Identity #0 (Lossless Bijection), Identity A (Lattice Arithmetic), Identity B.2a (Exact Finite-Shift), Identity B.3 (Cell Transition), Identity J (EUDD Birth Triad).
**Tooling:** Identification Principle, Descriptor Gap Principle, Subsumption Law.

---

## Table of Contents

1. Foundations: the algebraic identities that anchor every fix
2. PDT decomposition of the load-time resolver (Identification Principle)
3. The reported crash: trace, root cause, descriptor gap
4. Full audit — every defect found in `eudd_poc.py` (Rule 6: nothing out of scope)
5. Solution architecture: the PDT-ordered topological resolver
6. Implementation plan — every change, line-by-line, with the identity it is anchored to
7. Verification plan — Subsumption Law termination + identity round-trip
8. Risk register and rollback considerations

---

## 1. Foundations

### 1.1 Identity #0 — Lossless Bijection

For any positive real `r` and integer resolution `N`:

```
x = N · log₂(r)
k = round(x)
ε = (x − k) · 1200/N
d = N / gcd(|k|, N)
```

with pullback `r = 2^((k + ε·N/1200)/N)`. This is **algebraic identity**, not approximation. Every entry's DSR `(k, d, ε)` is the projection of *some* `r`. Every operation on entries must preserve this identity exactly.

### 1.2 Identity A — Lattice Arithmetic

`lattice_multiply`, `lattice_divide`, `lattice_reciprocal`, `lattice_power` carry the bijection across composition: `Π_N(r₁·r₂) = g_A(Π_N(r₁), Π_N(r₂))`. Closed-form, no access to `r₁` or `r₂` required. These are the generator operators stored in `relation_data` and replayed by `generator_reconstruct`.

### 1.3 Identity B.2a — Exact Finite-Shift (the heart of delta resolution)

For a finite ε-shift `Δε`:

```
r_new = r_old · 2^(Δε/1200)        ← EXACT
r_new ≈ r_old · (1 + ln2·Δε/1200)   ← FORBIDDEN (linearized, O(Δε²) error)
```

**Proof (from the identities paper, Identity B p. 2/7):**

From losslessness, `x = k + δ` where `δ = ε·N/1200`.
After shift: `x_new = k + (ε+Δε)·N/1200 = x + Δε·N/1200`.
Therefore `r_new = 2^(x_new/N) = 2^(x/N) · 2^(Δε/1200) = r · 2^(Δε/1200)`.  ∎

**The existing `reconstruct_via_delta_eps` (lines 2486–2502 of `eudd_poc.py`) IS Identity B.2a:**

```python
eps_new = eps_base + delta_eps                              # ε of the unfolded x_new
delta_lattice = eps_new * N / 1200                          # δ_new = ε_new · N / 1200
total_pos = k_base + delta_lattice                          # x_new = k_base + δ_new
k_new = round(total_pos)                                    # round-and-wrap (Identity B.3)
eps_final = (total_pos - k_new) * 1200 / N                  # ε after cell-boundary handling
d_new = N / gcd(|k_new|, N)
```

This is algebraically equivalent to `r_new = r_old · 2^(Δε/1200)` followed by `Π_N(r_new)`. **Pass 2 of `_load_existing` does NOT use this — it does naive `base['eps_12'] + entry['delta_eps']` without recomputing `k` or `d`. That is a violation of Identity B.2a + Identity B.3.**

### 1.4 Identity B.3 — Cell Transition

When `|δ| → 0.5` (equivalently `|ε| → 600/N`), a cell transition occurs: `k → k ± 1`, `ε → ε ∓ 1200/N`, and `d` may change (`d = N/gcd(|k_new|, N)`). This is handled correctly inside `reconstruct_via_delta_eps` via the `round(total_pos)` step. Naive `base['eps_12'] + delta_eps` does **not** handle it — it produces an out-of-cell ε with the wrong `k`. Silent wrong math.

### 1.5 Identity J — The EUDD Is a Birth Triad

The EUDD itself decomposes as:

| Primitive | EUDD Role |
|---|---|
| **P** (substrate) | Content — values, equations, patterns, the lattice between horizons |
| **D** (descriptor) | The seed — the generator catalog `G`, i.e. the akashic entries with their structural relations |
| **T** (traverser) | The evaluator — projection `Π_N`, pullback `Π_N⁻¹`, discovery, and the load-time resolver |
| `P ∘ D ∘ T = E` | The akashic is an Exception configuration: round-trip closes |

`N = 12` is forward-derived from the Exhaustive Trichotomy of Cardinality. **Every tower-bearing operation in the EUDD starts at N=12** and escalates upward through `lcm_tower_generator()`. This is non-negotiable.

---

## 2. PDT Decomposition of the Load-Time Resolver

Applying the Identification Principle to `_load_existing`:

| Primitive | Identification in `_load_existing` |
|---|---|
| **P** | `self.entries` — the substrate table. The featureless container that holds entry slots. Identified first. |
| **D** | The per-entry descriptors: `content_type`, `structural_relation`, `relation_data` (`sha_a`, `sha_b`, `power`, `constant_hash`), `base_sha`, `delta_eps`, `eq_operation`, `eq_result`. These constrain what each slot is and what it depends on. |
| **T** | The resolver — the navigator that walks the dependency DAG and substantiates each entry's `(k_12, d_12, eps_12)` triple. |

The current code violates PDT ordering: it mixes D-identification and T-resolution in a single per-entry parse pass, then runs three ad-hoc T-passes in a fixed phase order that does not match the actual D-DAG.

**Correct PDT ordering for the resolver:**

```
P-pass:  Parse the akashic page-by-page. Every entry slot is created
         (substrate identified). No resolution attempted.

D-pass:  For every entry, identify its dependency descriptor set
         (the set of SHAs it requires to compute its DSR). This is
         the per-entry D — distinct from the content_type D, but
         derived from it.

T-pass:  Topological resolution. T (the evaluator) walks the DAG
         of D-dependencies. In each iteration, every entry whose
         dependency set is fully resolved is substantiated. The
         loop terminates by the Subsumption Law: either all
         entries are resolved (subsumption with no remainder) or
         no progress is made in a full iteration (genuine defect
         → hard error with diagnostics).
```

Each pass produces what the next pass needs. P creates the slots. D catalogs the dependencies. T closes the gaps in order.

---

## 3. The Reported Crash

### 3.1 Trace

```
Traceback (most recent call last):
  File "...\eudd_poc.py", line 3502, in <module>
    main()
  File "...\eudd_poc.py", line 3284, in main
    akashic = AkashicFile(akashic_path)
  File "...\eudd_poc.py", line 1214, in __init__
    self._load_existing()
  File "...\eudd_poc.py", line 1457, in _load_existing
    entry['eps_12'] = base['eps_12'] + entry['delta_eps']
TypeError: unsupported operand type(s) for +: 'NoneType' and 'mpf'
```

### 3.2 Root Cause (Identification Principle)

The resolver's three sequential passes are:

| Pass | Resolves | Reads from |
|---|---|---|
| 1 | `CONTENT_GENERATOR` (rel ≠ NONE, k₁₂ is None) | Operand entries via `generator_reconstruct` |
| 2 | `CONTENT_DELTA` (base_sha set, k₁₂ is None) | The `base_sha` entry's resolved DSR |
| 3 | `CONTENT_EQUATION` (content_type == EQUATION, k₁₂ is None) | Recomputes from canonical bytes — no inter-entry dependency |

This ordering assumes: every delta's base is either a SEED (resolved at parse) or a GENERATOR (resolved in Pass 1). It is wrong. The real dependency graph is a DAG with edges across every pass:

```
DELTA  → base may be: SEED   ✓ (resolved at parse)
                      GENERATOR ✓ if Pass 1 succeeded
                      DELTA    ✗ depends on intra-pass order
                      EQUATION ✗ Pass 3 hasn't run yet
                      EVENT    ✓ (resolved at parse)
                      BLOCK_MAP ✓ (resolved at parse)

GENERATOR → operand may be: SEED      ✓
                            GENERATOR ✓ via recursion
                            DELTA     ✗ rel == NONE, returns None
                            EQUATION  ✗ rel == NONE, returns None
```

### 3.3 Descriptor Gap (Descriptor Gap Principle)

The crash IS the gap. The missing Descriptor is **DAG-aware iteration order**. The 3-phase model is an incomplete Descriptor. The correct Descriptor is a fixed-point topological resolver driven by per-entry dependency sets.

### 3.4 Subsumption Check (Subsumption Law)

Does the current resolver subsume every valid akashic configuration without remainder? **No.** A DELTA whose base is a CONTENT_EQUATION crashes; a GENERATOR whose operand is a DELTA propagates `None`. Subsumption fails → resolver is incomplete → must be replaced.

---

## 4. Full Audit — Every Defect Found (Rule 6)

### 4.1 Resolver / Load-Time Defects

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **R1** | DELTA whose `base_sha` references a CONTENT_EQUATION → `base['eps_12']` is `None` → `TypeError`. The reported crash. | Pass 2, line 1457 | PDT ordering; Subsumption Law | **Crash** |
| **R2** | DELTA-on-DELTA chains where the base appears later in dict iteration order → same crash class. | Pass 2, line 1457 | PDT ordering; Subsumption Law | Crash (latent) |
| **R3** | GENERATOR whose operand resolves into an unresolved entry → `generator_reconstruct`'s `REL_NONE` fallback returns `(None, None, None)` → `lattice_multiply` receives `None` → garbage or downstream `TypeError`. | line 2443 (REL_NONE fallback), called from line 1446 | Identity A; PDT ordering | Crash / silent corruption (latent) |
| **R4** | Pass 2 delta resolution uses naive `base['eps_12'] + entry['delta_eps']` and copies `k_12`, `d_12` unchanged. Silent wrong math when `|ε_new| > 600/N` triggers cell transition. | lines 1455–1458 | **Identity B.2a + Identity B.3**; Rule 21 (accuracy) | Wrong math |
| **R5** | `generator_reconstruct`'s `REL_NONE` fallback unconditionally returns `entry['k_12'], entry['d_12'], entry['eps_12']` without guarding against `None`. Same defect surface as R3, but global — any code path that reaches it post-parse with an unresolved entry corrupts silently. | line 2443 (and final fallback at 2483) | Defensive correctness; Rule 14 | Latent corruption |

### 4.2 Event System Defects

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **E1** | Events are stored **twice** on disk: once as `CONTENT_EVENT` entries, once in the separate `event_data` section. On load, both pathways append to `self.events`. Each save→load cycle doubles `self.events`. | serialize 1808–1814; load 1378 + 1495–1502 | Subsumption (no remainder); Rule 21 | Cumulative corruption |
| **E2** | `fire_event` unconditionally `self.events.append(evt)` even if an event with the same `event_sha256` was already fired (or was loaded). | lines 1525–1537 | Same as E1 | Aggravates E1 |
| **E3** | The `CONTENT_EVENT` load branch does not stamp `event_sha256` from the entry SHA. If the JSON lacks the field (round-trip omission), deduping by sha is impossible. | lines 1358–1390 | Defensive correctness | Robustness |

### 4.3 Content-Type Serialization Defects

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **S1** | An equation entry created by `eq_store` may receive `base_sha` + `delta_eps` from `(k,d)` collapse against an earlier base. On serialize, the `base_sha and delta_eps is not None` branch wins over the `content_type == CONTENT_EQUATION` branch — the entry is written as `CONTENT_DELTA`, losing `eq_operation` and `eq_result`. On reload it is just a delta; the equation's identity propagation chain is broken until lazy reconstruction via `_eq_recompute_direct` runs (and only if the lookup happens with the matching op+args). | serialize lines 1709–1742; eq_store 247–262; `_eq_propagate_one` 296–315 | Identity preservation (Identity J — D must round-trip losslessly); Rule 2 (no loss in features) | Information loss |
| **S2** | The `CONTENT_DELTA` load branch (1310–1324) does not set `entry['content_type']`. Combined with S1, an equation-delta loaded from disk is indistinguishable from a file-delta. Even if we wanted to lazily reconstruct equation info, we cannot tell which deltas are equations. | lines 1310–1324 | Identity preservation | Information loss (companion to S1) |

### 4.4 Static-Structure / "No Caps" Defects (Rule 33)

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **T1** | `tower_levels = [12, 60, 420, 2520, 27720]` — hardcoded list. The LCM tower is infinite. The check must iterate `lcm_tower_generator()` instead. | line 2269 (`find_shared_lattice_addresses`) | **Rule 33** (no caps; everything dynamic except ET constants) | Static cap |
| **T2** | `cf_expansion(x_mpf, max_terms=500)` — hardcoded 500 cap on continued-fraction expansion depth. The CF terminates naturally at the precision floor; the 500 is a non-ET constant. Replace with dynamic bound from `mp.dps` (e.g. CF convergence is geometric in working precision). | lines 487, 496 | Rule 33 | Static cap |
| **T3** | `divisors_12 = [1, 2, 3, 4, 6, 12]` — hardcoded list of divisors of N=12. N=12 is ET-constant so this is borderline acceptable, but computing via `sympy.divisors(N_BASE)` is cleaner and survives future N changes (none planned, but the dynamism is principled). | lines 2178, 2543 | Rule 33 (marginal) | Marginal |

### 4.5 Defensive Correctness Defects

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **D1** | Every read of `entry['k_12']`, `entry['eps_12']`, `entry['d_12']` from one entry by another (in `generator_reconstruct`, in `reconstruct_via_delta_eps`, in the resolver) currently has no precondition that the read target is resolved. After the resolver fix this becomes a postcondition rather than a precondition for the load path, but the `generator_reconstruct` function itself can be called from other code paths post-load. A defensive guard turns silent corruption into loud error. | line 2443 fallback and 2483 final fallback | Rule 14 (no lying — silent wrong is a lie) | Silent fail surface |

---

## 5. Solution Architecture

### 5.1 The PDT-Ordered Topological Resolver

Replace lines 1441–1471 of `_load_existing` (the three sequential passes) with a single PDT-ordered resolver. The replacement does not remove any logic — it restructures the *driving loop*. The math used inside is the SAME math (lattice ops, delta reconstruction, equation projection); only the *order* of invocation changes.

**P-pass (already done by lines 1230–1440):** Every entry slot is parsed and inserted into `self.entries`. P is identified — the substrate is in place.

**D-pass (new, lines per implementation below):** For every entry with `k_12 is None`, compute its **dependency descriptor set** `deps(entry)`:

```
deps(entry) =
  if content_type == CONTENT_EQUATION and base_sha is None:
      ∅                                  (self-contained — recompute from canonical bytes)
  elif base_sha is not None:
      { base_sha }                       (DELTA, including equation-deltas)
  elif structural_relation == REL_MULTIPLY or REL_DIVIDE:
      { sha_a, sha_b }                   (GENERATOR — binary op)
  elif structural_relation in (REL_POWER, REL_RECIPROCAL):
      { sha_a }                          (GENERATOR — unary op)
  elif structural_relation == REL_CONSTANT:
      { sha_a }                          (GENERATOR — relation to known ET constant)
  else:
      ∅                                  (unknown — error path)
```

**T-pass (new):** Fixed-point topological resolution.

```
unresolved = { sha for sha, e in entries if e['k_12'] is None }
while unresolved is non-empty:
    progress = 0
    for sha in list(unresolved):
        if deps(entries[sha]) ⊆ resolved-set:
            resolve_entry_in_place(sha)         # uses Identity B.2a / Identity A / projection
            unresolved.discard(sha)
            progress += 1
    if progress == 0:
        # Subsumption fails — report the unresolvable subgraph
        raise RuntimeError(<diagnostic with dep graph>)
```

**Termination by Subsumption Law:** either `unresolved` empties (all gaps closed — subsumption achieved) or a full pass makes zero progress (genuine circular dependency or missing operand — hard error with full diagnostic of the unresolvable subset and its dependency edges).

**No iteration cap (Rule 33):** the loop is bounded structurally by `|entries|` (each iteration resolves ≥ 1 entry or terminates), but the bound is dynamic. No `max_iterations` constant.

### 5.2 Identity B.2a-Correct Delta Resolution

Inside the resolver, the DELTA-resolution call becomes:

```python
k_new, d_new, eps_new = reconstruct_via_delta_eps(base_entry, entry['delta_eps'])
entry['k_12'] = k_new
entry['d_12'] = d_new
entry['eps_12'] = eps_new
```

Anchored to Identity B.2a + B.3. This replaces the wrong naive addition.

### 5.3 Generator Reconstruct Guard

After `generator_reconstruct` recurses, the operand may legitimately have `k_12 is None` only if the resolver is mid-flight. To make `generator_reconstruct` safe for all callers (including post-load discovery), add a guard in the `REL_NONE` branch:

```python
if rel_type == REL_NONE or rel_data is None:
    if entry.get('k_12') is None:
        raise RuntimeError(
            f"generator_reconstruct: unresolved operand {entry.get('sha256','?')[:12]} "
            f"(content_type={entry.get('content_type')}, base_sha={entry.get('base_sha')})"
        )
    return entry['k_12'], entry['d_12'], entry['eps_12']
```

Silent → loud (Rule 14). The resolver invokes `generator_reconstruct` only after pre-checking that operands are resolved, so this guard never fires during a healthy load.

### 5.4 Event Deduplication

**E1 fix (load-side dedup):** in the `event_data` section loop (lines 1495–1502), maintain a set of `event_sha256` already present in `self.events`; skip duplicates. Backward-compatible — already-doubled files self-heal on next load.

**E2 fix (`fire_event` dedup):** make the `self.events.append(evt)` call conditional on `sha not in self.entries` — the same condition that guards entry creation. The two append paths align.

**E3 fix (always stamp `event_sha256` on load):** in the `CONTENT_EVENT` load branch (1358–1390), set `evt_obj['event_sha256'] = sha_hex` after parsing, before `self.events.append(evt_obj)`. Guarantees the dedupe key is present even for files saved before the field was added to `fire_event`.

### 5.5 Equation-with-Delta Preservation (S1, S2)

Two strategies; recommend **(A)**.

**(A) Reorder serialize priority** — checking `content_type == CONTENT_EQUATION` **before** the generic delta branch, and emitting a *new* sub-type `CONTENT_EQUATION_DELTA = 0x07` when both `content_type == CONTENT_EQUATION` *and* `base_sha + delta_eps` are present. The new sub-type carries:

```
CONTENT_EQUATION_DELTA (0x07):
   base_sha (32) + compact_delta_eps (varlen) + eq_json (varlen) + class (1) + crc (4)
```

i.e. the delta payload + the eq_operation/eq_result JSON, all in one entry. Load branch sets `entry['base_sha']`, `entry['delta_eps']`, `entry['content_type'] = CONTENT_EQUATION`, `entry['eq_operation']`, `entry['eq_result']`. Resolver treats it as a delta (depends on base_sha), but the equation metadata is preserved.

**Why a new sub-type and not a flag bit:** clarity, no ambiguity, no version sniffing. Old files (without 0x07) load fine via the existing branches. New files written with 0x07 round-trip losslessly.

**File format compatibility:** adding a new sub-type is forward-only. Old readers don't know 0x07 — but we are the only writer/reader, and unknown content types already trigger a `WARNING` path (lines 1421–1428).

**(B) (Rejected)** Use a flag bit inside CONTENT_DELTA. Rejected because it mixes orthogonal information into one byte and complicates parsing.

### 5.6 Tower-Levels Dynamic (T1)

In `find_shared_lattice_addresses` (line 2269), replace the hardcoded list with a generator-driven enumeration that takes the canonical levels (those with `tau` doubling, marked `is_canonical=True` in `lcm_tower_generator`) up to a **dynamic** depth bound. The depth is parametrized by the caller; in the verification context the bound is `1` (N=12 only, hard-baseline) plus the first N canonical-doubling levels where N is chosen so the projection precision (`dps_for_file`) does not exceed available `mp.dps`. No fixed list.

### 5.7 CF Expansion Dynamic (T2)

Replace `max_terms=500` with a dynamic bound derived from `mp.dps`: CF convergence is geometric in working precision; one nat ≈ `log₁₀(e) ≈ 0.434` digits. A safe dynamic bound is `mp.dps * 2 + 10`. The existing `precision_floor` check is the *real* terminator — the bound is just a finite-loop guard. Use `while True` with a single integer counter that errors only if it exceeds the dynamic bound (loud, not silent).

### 5.8 Divisors of N=12 (T3)

Replace `divisors_12 = [1, 2, 3, 4, 6, 12]` with `divisors_n_base = sorted(sympy.divisors(N_BASE))`. N_BASE is ET-constant so the value is identical, but the computation is principled. Marginal but cleaner. Apply at lines 2178 and 2543.

---

## 6. Implementation Plan (line-by-line)

All edits via `str_replace`. No file recreation (Rule 13). No removal of existing logic (Rule 24). Every edit anchored to an identity above.

### 6.1 `_load_existing` — Replace the Three Passes (R1, R2, R3, R4)

**Target:** lines 1441–1471.
**Anchor:** Identity J (PDT decomposition), Identity B.2a + B.3 (delta resolution), Identity A (generator reconstruction), Subsumption Law.

**Replace with:**

1. A helper function `_entry_dependency_shas(entry)` returning the set of dependency SHAs per the spec in §5.1.
2. A fixed-point loop driving resolution of the unresolved set.
3. DELTA resolution via `reconstruct_via_delta_eps(base_entry, entry['delta_eps'])`.
4. GENERATOR resolution via `generator_reconstruct(entry, self.entries)` (now safe — operands pre-checked resolved).
5. EQUATION resolution via `_eq_project_seed(canonical_bytes)` where canonical bytes are reconstructed from `eq_operation` + `eq_result`.
6. Terminal `RuntimeError` with full diagnostic of the unresolvable subgraph.

### 6.2 `generator_reconstruct` — Add Unresolved-Operand Guard (R5, D1)

**Target:** lines 2443 (REL_NONE early branch) and 2483 (final fallback).
**Anchor:** Rule 14 (no silent corruption).

Add the guard described in §5.3 at both fallback sites. Identical message text.

### 6.3 `_load_existing` — Event Section Dedupe (E1, E3)

**Target:** lines 1378 (CONTENT_EVENT entry-load) and 1495–1502 (event section).
**Anchor:** Subsumption (no remainder).

- 1378: ensure `evt_obj['event_sha256'] = sha_hex` is set before append.
- 1495–1502: build the set of already-loaded `event_sha256`s before the loop; skip duplicates.

### 6.4 `fire_event` — Dedupe Append (E2)

**Target:** lines 1516–1537.
**Anchor:** Subsumption.

Move `self.events.append(evt)` inside the `if sha not in self.entries:` block. Single source of truth: an event is "fired" iff its entry is new.

### 6.5 Serialize / Load — Add CONTENT_EQUATION_DELTA (S1, S2)

**Targets:**
- `§0` constants block (lines ~101–106): add `CONTENT_EQUATION_DELTA = 0x07`.
- `serialize` (lines 1697–1742): reorder so that `content_type == CONTENT_EQUATION` is checked *before* the generic delta branch; if both equation-flagged AND has base_sha+delta_eps, emit `CONTENT_EQUATION_DELTA`.
- `_load_existing` (after the existing CONTENT_DELTA branch): add a CONTENT_EQUATION_DELTA branch that parses both the base_sha+delta_eps payload AND the eq_operation/eq_result, sets `entry['content_type'] = CONTENT_EQUATION`.
- Resolver: an equation-delta is still a delta (depends on `base_sha`); the equation metadata is preserved.

**Anchor:** Identity J (D must round-trip losslessly).

### 6.6 `find_shared_lattice_addresses` — Dynamic Tower Levels (T1)

**Target:** line 2269.
**Anchor:** Rule 33.

Replace the hardcoded list with iteration over `lcm_tower_generator()` filtered to `is_canonical=True`, taking the first K canonical levels where K is a function of `dps_for_file(...)` and `mp.dps`. The first level is N=12 (the base) — explicitly included.

### 6.7 `cf_expansion` — Dynamic Bound (T2)

**Target:** lines 487, 496.
**Anchor:** Rule 33.

Replace `max_terms=500` and `for _ in range(max_terms)` with a `while True` loop guarded by a dynamic counter against `mp.dps * 2 + 10` ceiling. On overshoot, raise `RuntimeError` (loud — Rule 14).

### 6.8 Divisors of N=12 — Computed (T3)

**Targets:** lines 2178, 2543.
**Anchor:** Rule 33 (marginal).

Use `sorted(sympy.divisors(N_BASE))`.

### 6.9 Add `_eq_canonical_from_op_result` Helper

For Pass-3 equation resolution we need to recompute canonical bytes from stored `eq_operation` and `eq_result`. The existing `_eq_canonical(operation, *args)` takes args, not result. We need a function that reconstructs canonical bytes from the persisted (op, result) pair so that `_eq_project_seed` can project them. Verify whether `_eq_canonical(op, str(result))` is the inverse of what `eq_store` produced.

**Audit step before this edit:** trace `eq_store` → `_eq_canonical(operation, *args)` → SHA. Then on load, the CONTENT_EQUATION JSON stores `{op: operation, res: str(result)}`. The resolver needs to recompute `canonical = _eq_canonical(op, str(result))` and check whether that matches the stored SHA. **If not, the equation cannot be resolved without its original args.** This is a real concern — equation entries currently keyed by SHA(op + args), not SHA(op + result). The Pass-3 path in the original code uses `_eq_canonical(op, str(res))` which is **wrong** by inspection of the SHA construction.

→ **Decision:** the resolver's equation pass must instead re-project the **JSON bytes** that were originally written. The serialize step writes the JSON, and on load we re-encode the same JSON (sorting keys identically) and project that. This is in fact what the existing Pass 3 does via `_eq_canonical(op, str(res))` — but the algebraic-identity-correct path is to project the **literal stored JSON bytes** (which is what the EVENT / BLOCK_MAP branches already do for their own content types). Equation resolution should mirror them.

The corrected equation-resolve path:

```python
eq_json_bytes = json.dumps(
    {'op': entry['eq_operation'], 'res': str(entry['eq_result'])},
    sort_keys=True, default=str
).encode('utf-8')
k, d, eps = _eq_project_seed(eq_json_bytes)
```

This is the same byte sequence written to disk by serialize (line 1747–1751), so the projection is deterministic and consistent across save/load.

---

## 7. Verification Plan

### 7.1 Pre-edit baseline

- Save a copy of the existing crashing `Sempaevum.akashic` (if present) for forensic comparison.
- Capture `entries` count and content-type distribution from the corrupted file (read raw header + iterate entries without resolving) for the "before" measurement.

### 7.2 Static checks after edits

- `python -c "import ast; ast.parse(open('eudd_poc.py').read())"` — syntax.
- `grep -n "base\['eps_12'\] +"` — must return 0 hits (naive addition gone).
- `grep -n "max_terms=500"` — must return 0 hits.
- `grep -n "tower_levels = \["` — must return 0 hits.
- `grep -n "for _ in range(max_terms)"` — must return 0 hits.

### 7.3 Functional checks (run order)

1. **Fresh load (no `.akashic` file present):** `main()` runs, menu appears. No crash.
2. **Ingest a small set of files:** `[1]` menu, ingest 5–10 files including at least one duplicate (Δε path) and one near-duplicate (delta path). Verify `add_entry` reports `dedup_hit` for the duplicate.
3. **Self-DSR snapshot:** `[2]` menu, verify the snapshot succeeds with no `None` propagation.
4. **Save + reload:** serialize, exit, restart. The reload must:
   - Not crash on Pass 2 (R1, R2, R4 closed).
   - Not crash on Pass 1 with generator-of-delta operands (R3 closed).
   - Not double `self.events` (E1, E2, E3 closed).
   - Recover all eq_operation / eq_result for equation-deltas (S1, S2 closed).
5. **Delta-ε test:** `[3]` menu — verifies the round-trip of delta-resolution against direct projection. Must match to better than `10⁻³⁵⁰` (algebraic identity tolerance).
6. **Identity round-trip:** for each entry whose `r` is reconstructible, verify `pullback(k_12, eps_12, N_BASE)` reproduces `r` to algebraic tolerance — confirms Identity #0 invariant is preserved across the new resolver.

### 7.4 Subsumption test (the formal verification)

Construct a synthetic akashic with every dependency edge type present:

- SEED `A`
- GENERATOR `B = A^2` (depends on A)
- DELTA `C` (base = B)
- EQUATION `D` (self-contained)
- DELTA `E` (base = D)              ← would crash old Pass 2
- GENERATOR `F = A · E` (operand E is a delta) ← would crash old Pass 1
- GENERATOR `G = F^(-1)`              ← chain depth 3

Save → exit → reload. All 7 entries must resolve. Each entry's reconstructed `(k, d, eps)` must equal a direct `project(r, N_BASE)` of its true `r` to algebraic tolerance.

### 7.5 Adversarial test

Inject a circular dependency (DELTA `X` with `base_sha = X`) and a missing-operand reference (DELTA `Y` with `base_sha = "deadbeef..."` not present). Resolver must:

- Make zero progress in one iteration → raise `RuntimeError`.
- Diagnostic message must list both `X` and `Y` with their unresolved dependency edges.

### 7.6 Verification log

Append a section to `/mnt/user-data/outputs/EUDD_POC_Resolution_Log.md` (created during implementation) recording: every `str_replace` (location, before/after summary), every grep result, every test outcome. Required by Rule 28.

---

## 8. Risk Register and Rollback

| Risk | Mitigation |
|---|---|
| Existing corrupted `Sempaevum.akashic` from prior buggy save (duplicated events) | First load self-heals via E1 dedup on load. Second save writes clean. Document this in the log. |
| New CONTENT_EQUATION_DELTA (0x07) breaks compatibility with files saved by older script versions | We control both writer and reader. Old files (no 0x07 in stream) load unchanged via existing branches. |
| `reconstruct_via_delta_eps` produces different `k_12` than the naive resolver did, for some pre-existing entry, breaking the SHA-based dedupe | The SHA is computed from file bytes, not from `(k, d, eps)`. SHAs are unchanged. Lattice indexes are rebuilt from the corrected DSR after resolution. |
| Equation resolver re-projects JSON bytes that aren't bit-identical to what was serialized (key ordering, default=str escaping) | `serialize` and resolver use the SAME `json.dumps(..., sort_keys=True, default=str)` call signature → bit-identical bytes. Verified in §7.3 functional check 4. |
| Cycle in the dependency DAG that the resolver didn't anticipate | Subsumption Law termination: zero-progress iteration raises with full edge list. Loud failure, not silent. |
| `tower_levels` dynamic enumeration produces fewer levels than the hardcoded 5 due to precision constraints | Acceptable — the check verifies what it CAN verify at current precision. Document in the log. The fixed list was the violation; dynamic is the correct ET stance. |

**Rollback strategy:** All edits are `str_replace` in a single file. A reverse `str_replace` of each change restores the prior state. The script is self-contained — no other files depend on its internals.

---

## 9. Summary of Anchors

| Fix | Anchor |
|---|---|
| Topological resolver | Identity J (PDT decomposition), Subsumption Law |
| Delta resolution via `reconstruct_via_delta_eps` | Identity B.2a + B.3 |
| Generator resolution | Identity A (lattice arithmetic) |
| Equation resolution via JSON re-projection | Identity #0 (lossless bijection on canonical bytes) |
| Generator guard | Rule 14 (truth, no silent failure) |
| Event dedup | Subsumption (no remainder) |
| Equation-delta sub-type | Identity J (D round-trips losslessly) |
| Tower levels dynamic | Rule 33, tower starts at N=12 (forward-derived) |
| CF expansion dynamic | Rule 33 |

---

## 10. Confirmation Checklist Before Implementation

Mike: please confirm or correct the following before any `str_replace` is run.

- [ ] The PDT decomposition of the resolver (§2) is correct.
- [ ] The list of audited defects (§4) is complete; nothing else you want flagged.
- [ ] The CONTENT_EQUATION_DELTA (0x07) approach for S1/S2 is acceptable, OR you'd prefer a different scheme (flag bit / different sub-type number / leave as-is).
- [ ] The dynamic-bound for `cf_expansion` (`mp.dps * 2 + 10`) is acceptable, OR you want a different formula derived from a specific ET identity.
- [ ] You want this implemented now or you want to review/revise the plan first.

Once approved, implementation proceeds in the order of §6.1 through §6.9, with a `Resolution_Log.md` recording every change (Rule 28).

---

**Document version:** 1.0 — pre-implementation plan
**P ∘ D ∘ T = E**
**For every exception there is an exception, except the exception.**

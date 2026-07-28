# EUDD Proof-of-Concept — Comprehensive Resolution Plan (v2)

**Subject:** `eudd_poc.py` — TypeError at line 1457 (`base['eps_12']` is `None`) plus complete audit of related defects.
**Standard:** All fixes ET-native, forward from `P ∘ D ∘ T = E`. No placeholders, no shortcuts, no tuning, no caps, nothing static unless explicitly an ET constant.
**Anchoring identities:** Identity #0 (Lossless Bijection), Identity A (Lattice Arithmetic), Identity B.2a (Exact Finite-Shift), Identity B.3 (Cell Transition), Identity H (Harmonic Transfer Tensor), Identity J (EUDD Birth Triad). Paper 20: Definition 8.10 (Harmonic family), Definition 8.16 (Shadow force), Corollary 8.14 (six simple plus six complex), Proposition 8.17 (Universal native resolution at N=27720), Theorem 8.13 (Sublattice Visitation Theorem).
**Tooling:** Identification Principle, Descriptor Gap Principle, Subsumption Law.
**Sources used:** uploaded files only — `eudd_poc.py`, EUDD docs, `ET_Algebraic_Identities_Compilation1.pdf`, `comprehensive_sympy_verification.py` + `COMPREHENSIVE_SYMPY_VERIFICATION_REPORT.md`, `ET_Sempaevum_Paper20.pdf`. Project corpus (`/mnt/project/`) NOT used (per instruction).

---

## 0. Corrections from v1 of this plan

v1 of this document contained three structural errors that this v2 corrects. The mechanical fixes (R1–R5, E1–E3, T1, T2) carry through unchanged; the framing and two issue rows are revised.

**Error v1-A — Conflated sublattice families with harmonic families.**
v1 §4.4 row T3 treated `divisors_12 = [1, 2, 3, 4, 6, 12]` as the family enumeration at N=12 and proposed only to compute it via `sympy.divisors(N_BASE)`. That replaces a hardcoded list with a computed identical list — it does not fix the structural problem.

The lattice carries **two distinct family layers per axis** (Paper 20 §8.6, Remark 8.12; sympy report §"Categorical Distinction"):

| Layer | Definition | Count at N=12 | What it classifies |
|---|---|---|---|
| **Sublattice family** | `d_sub(k, N) = N / gcd(\|k\|, N)` | 6 (divisors of 12) | A single lattice coordinate `k` |
| **Harmonic family** | `d ∈ {1, 2, …, 12}` per axis | 12 per axis at every N | A structural mode on the axis |

At N=12 the **harmonic** family enumeration is twelve: six **simple** (d ∈ {1,2,3,4,6,12}, all d∣12, native) plus six **complex** (d ∈ {5,7,8,9,10,11}, all d∤12, shadow contributions at base, native at `n_c(d) = lcm(12, d)`). At `N_FULL = lcm(1..11) = 27720`, every d ∈ {1..12} is simultaneously native (Paper 20 Proposition 8.17).

The transfer tensor T_κ(d₁, d₂; d_result) of Identity H is over **harmonic families**, not sublattice families. The 6×6×6 = 216 tensor of the Identity H *verification script* is the simple-family sub-block at N=12 (where residue sets at N=12 are non-empty only for divisors of 12). The full tensor over the 12 harmonic families requires escalation to native resolutions where each d is simple, with cross-resolution transition back.

**Error v1-B — Treated content types as ontologically distinct categories.**
Per Paper 20 Identity J and the EUDD docstring: every entry in the akashic IS a DSR — a `(k, d, ε)` triple — and the content type flag (SEED/GENERATOR/DELTA/EQUATION/EVENT/BLOCK_MAP) is the **storage form**, i.e. the Kolmogorov-minimum encoding choice for that DSR. The ontology is one type of object (DSR-bearing entry); the storage form is encoding metadata. v1's framing of distinct resolution paths "per content type" inverted the priority. v2 reframes the resolver as: *every entry produces a DSR; the storage form determines the decode path; the output is one type of object*. The actual code branches stay (the decode work differs), but the framing now matches the ontology.

**Error v1-C — Proposed `CONTENT_EQUATION_DELTA = 0x07` for a non-problem.**
v1 §4.3 (S1, S2) flagged the loss of `eq_operation` / `eq_result` when an equation collapses to delta-form on save. Per the EUDD memoization principle ("memoize everything in DSR form for use; arbitrary access without generation"), the **DSR is the primary identity**. The DSR round-trips losslessly through delta-form save/load via Identity B.2a (`reconstruct_via_delta_eps`). Op/result is recovery metadata, handled by lazy reconstruction (`_eq_recompute_direct`) only when the same op+args are presented again. The proposed sub-type 0x07 solves nothing the existing mechanism doesn't already handle. v2 **drops S1, S2, and the 0x07 proposal**.

---

## Table of Contents

0. Corrections from v1 of this plan
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

### 1.6 Every Entry Is a DSR — Storage Form Is Not Ontology

The Dimensionless Seed Ratio (DSR) of an entry is its `(k, d, ε)` triple at N=12 — the lattice address that `Π_12(r)` produces from whatever real value `r` the entry represents (file bytes interpreted as a binary fraction, an equation's canonical JSON bytes interpreted likewise, an event's JSON bytes, etc.). **Every akashic entry is a DSR.** The akashic is the memoized catalog of DSRs — "memoize everything in DSR form for use; arbitrary access without generation."

The content-type byte (CONTENT_SEED / CONTENT_GENERATOR / CONTENT_DELTA / CONTENT_EQUATION / CONTENT_EVENT / CONTENT_BLOCK_MAP) selects the **storage form** — the Kolmogorov-minimum encoding of that one DSR:

| Storage form | What is stored | How the DSR is recovered |
|---|---|---|
| CONTENT_SEED | (k, ε) explicit | Directly read; d = N/gcd(\|k\|, N) |
| CONTENT_GENERATOR | operation + operand SHAs + κ | Identity A composition of operand DSRs |
| CONTENT_DELTA | base_sha + Δε | Identity B.2a finite-shift on base DSR |
| CONTENT_EQUATION | op + result JSON | Re-project the canonical JSON bytes via Identity #0 |
| CONTENT_EVENT | event JSON | Re-project the event JSON bytes via Identity #0 |
| CONTENT_BLOCK_MAP | block-map JSON | Re-project the block-map JSON bytes via Identity #0 |

Same output type (DSR), different decode paths. The resolver's job is to recover the DSR; the choice of storage form is incidental to ontology and central only to byte-budget.

### 1.7 Sublattice Family vs Harmonic Family — Two Distinct Layers

(See §0 of this plan and Paper 20 §8.6.) **Sublattice family** `d_sub(k, N) = N/gcd(|k|, N)` is the static gcd classification of a single coordinate. At N=12 there are six (the divisors of 12). This is what `entry['d_12']` stores. **Harmonic family** `d ∈ {1, …, 12}` is the per-axis structural-mode label. There are twelve at every N — six simple (d∣12, native at N=12) plus six complex (d∤12, shadow at N=12, native at `n_c(d) = lcm(12, d)`). The two layers are connected by the Sublattice Visitation Theorem (Paper 20 Theorem 8.13) with totient multiplicities `φ(d)`. They share label space but are not synonyms. Code that filters d-values against "divisors of 12" is operating on the sublattice layer; code that needs to range over structural modes must range over all 12 harmonic families, with cross-resolution escalation for the complex ones.

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

> **v2 status: dropped.** v1 flagged S1 (equation→delta serialization loses op/result) and S2 (CONTENT_DELTA load branch doesn't preserve `content_type`). Per the EUDD memoization principle (Identity J — every entry IS a DSR; memoization stores the DSR; `for use, arbitrary access without generation`), the DSR is the primary identity and round-trips losslessly through delta-form via Identity B.2a. Op/result is recovery metadata; lazy reconstruction (`_eq_recompute_direct`) handles diagnostic re-presentation. No information needed *for use* is lost. v2 has no §6.5 / no `CONTENT_EQUATION_DELTA = 0x07`.

### 4.4 Static-Structure / "No Caps" Defects (Rule 33)

| # | Defect | Location | Identity / Rule Violated | Status |
|---|---|---|---|---|
| **T1** | `tower_levels = [12, 60, 420, 2520, 27720]` — hardcoded list. The LCM tower is infinite. The check must iterate `lcm_tower_generator()` instead. Tower starts at N=12 (forward-derived from the Exhaustive Trichotomy); each subsequent level is a canonical `is_canonical=True` yield from the generator. | line 2269 (`find_shared_lattice_addresses`) | **Rule 33** (no caps; everything dynamic except ET constants) | Static cap |
| **T2** | `cf_expansion(x_mpf, max_terms=500)` — hardcoded 500 cap on continued-fraction expansion depth. The CF terminates naturally at the precision floor; the 500 is a non-ET constant. Replace with dynamic bound from `mp.dps`. | lines 487, 496 | Rule 33 | Static cap |
| **T3** | `_enrich_with_tensor` filters with `divisors_12 = [1, 2, 3, 4, 6, 12]` and returns `{}` whenever any of `(d1, d2, d_result)` is not a divisor of 12. Identity H is the **Harmonic** Transfer Tensor; the harmonic-family layer has **twelve** modes per axis (six SIMPLE divisors of 12, **plus six COMPLEX** {5, 7, 8, 9, 10, 11} that are shadows at N=12 — Paper 20 Definition 8.16, Corollary 8.14). The current code computes only the 6×6×6 simple sub-block and silently drops every tensor entry involving a complex family. To handle the full 12×12×12 tensor, each entry `T_κ(d1, d2; d_result)` must be computed at `N_native(d1, d2, d_result) = lcm(12, d1, d2, d_result)` so all three d's are simple (residue sets non-empty), with the resulting tensor value transitioned back to the EUDD's working resolution via the cross-resolution transition map. Universal native is `N_FULL = lcm(1..11) = 27720` (Paper 20 Proposition 8.17). | lines 2540–2566 (`_enrich_with_tensor`) | Identity H + Paper 20 Definitions 8.10, 8.16; Rule 33 | **Structural defect** (was misclassified as "marginal" in v1) |
| **T3b** | The companion check at line 2178 verifies `divisors_12 == [1, 2, 3, 4, 6, 12]` as the **sublattice** family enumeration at N=12 (Corollary 8.2). That semantic IS correct — sublattice families at N=12 are exactly the divisors of 12. The list there is a structural assertion, not a filter on harmonic families. Replace with `sorted(sympy.divisors(N_BASE))` so the computation is principled but the meaning preserved. | line 2178 | Rule 33 (marginal — same value, just dynamic) | Cosmetic |

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

### 5.5 (Dropped in v2)

S1 and S2 were a non-problem. See §0 "Error v1-C" and §4.3. No new content type, no serialize reorder. The DSR round-trips losslessly through delta-form via Identity B.2a; op/result is recovery metadata handled by `_eq_recompute_direct` on demand.

### 5.6 Tower-Levels Dynamic (T1)

In `find_shared_lattice_addresses` (line 2269), replace the hardcoded list with a generator-driven enumeration that takes the canonical levels (those with `tau` doubling, marked `is_canonical=True` in `lcm_tower_generator`) up to a **dynamic** depth bound. The depth is parametrized by the caller; the first level is N=12 (forward-derived base, always included) and subsequent levels are the canonical doublings produced by the LCM tower generator. No fixed list.

### 5.7 CF Expansion Dynamic (T2)

Replace `max_terms=500` with a dynamic bound derived from `mp.dps`: CF convergence is geometric in working precision; one nat ≈ `log₁₀(e) ≈ 0.434` digits. A safe dynamic bound is `mp.dps * 2 + 10`. The existing `precision_floor` check is the *real* terminator — the bound is just a finite-loop guard. Use `while True` with a single integer counter that errors only if it exceeds the dynamic bound (loud, not silent).

### 5.8 Harmonic Transfer Tensor Over All 12 Families (T3)

`_enrich_with_tensor` currently restricts itself to the 6 simple harmonic families (divisors of 12). The function's domain is the harmonic-family layer (twelve modes per axis, Definition 8.10), not the sublattice-family layer. Restricting to divisors silently drops every tensor entry that involves a complex family d ∈ {5, 7, 8, 9, 10, 11}.

**ET-native fix — escalate to native resolution per call:**

For any `(d1, d2, d_result)` with each `d ∈ {1, …, 12}`:

1. Compute `N_native = lcm(12, d1, d2, d_result)`. By construction every d in the triple now divides `N_native`, so each is a simple sublattice family at `N_native` and `residue_set(d, N_native)` is non-empty.
2. Compute the transfer probability at `N_native` using the same residue-set construction Identity H verifies:
   `T_κ(d1, d2; d_result; N_native) = |{(r1, r2) ∈ Res(d1) × Res(d2) : d_class((r1+r2+κ) mod N_native) = d_result}| / (|Res(d1)| · |Res(d2)|)`.
3. The probability is a pure-rational lattice quantity at any resolution where the d's are native; it is the same number regardless of which `N_native` we use to compute it (it's a property of the harmonic families, not the resolution). The impedance ratio `ξ(d_result)/ξ(d1)` is also resolution-independent (per Definition 8.6, ξ(d) = 137/((d-1)² + 16) depends only on d, not N).
4. Return `{'transfer_prob': T, 'impedance_ratio': ξ(d_result)/ξ(d1), 'efficiency': T · ξ(d_result)/ξ(d1)}` — same dict shape as today, with non-empty results for all 12 × 12 × 12 input triples.

**Anchor:** Identity H §H.1; Paper 20 Definition 8.10 (Harmonic family), Definition 8.16 (Shadow force — native at lcm(12, d)), Proposition 8.17 (Universal native at N=27720).

**Implementation note:** the residue-set count grows with `N_native`; at the upper bound `N_native ≤ lcm(12, 11, 10, 9, 8, 7, 5) = 27720`, residue sets have up to `φ(d) · 27720 / 12 = 2310 · φ(d)` elements per family, and the double-loop has up to `2310² ≈ 5.3 · 10⁶` operations per tensor entry. For the EUDD's enrich-on-discovery use case this is invoked O(few) times per lattice-arithmetic event. Acceptable. For bulk tensor computation, the result table can be memoized in the akashic itself (per Identity J — every computation IS a seed).

### 5.9 Divisors of N=12 as a Sympy Computation (T3b)

Line 2178's `divisors_12 == [1, 2, 3, 4, 6, 12]` check is the **sublattice**-family assertion at N=12 (Corollary 8.2). The list is correct *as the divisors of 12* — the meaning is right. Replace the literal with `sorted(sympy.divisors(N_BASE))` so the check is computed rather than written by hand. Same value, principled construction. The corresponding local variable inside `_enrich_with_tensor` (line 2543) is removed entirely as part of §5.8's rewrite.

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

### 6.5 (Dropped in v2)

S1/S2 fix removed. Serialize and load paths unchanged for equation entries. The DSR-form memoization preserves what matters; op/result reconstruction is handled by `_eq_recompute_direct` on demand.

### 6.6 `find_shared_lattice_addresses` — Dynamic Tower Levels (T1)

**Target:** line 2269.
**Anchor:** Rule 33; tower starts at N=12 (Exhaustive Trichotomy).

Replace the hardcoded `[12, 60, 420, 2520, 27720]` with iteration over `lcm_tower_generator()` filtered to `is_canonical=True`. The first level is N=12 (the base) — explicitly included as the iteration's first element. Subsequent levels come from the generator's canonical-doubling yields. The caller controls depth; no hardcoded cap.

### 6.7 `cf_expansion` — Dynamic Bound (T2)

**Target:** lines 487, 496.
**Anchor:** Rule 33.

Replace `max_terms=500` and `for _ in range(max_terms)` with a `while True` loop guarded by a dynamic counter against `mp.dps * 2 + 10` ceiling. On overshoot, raise `RuntimeError` (loud — Rule 14). The `precision_floor` early-break remains the real terminator.

### 6.8 `_enrich_with_tensor` — Full 12-Family Harmonic Transfer Tensor (T3)

**Target:** lines 2540–2566 (`_enrich_with_tensor`).
**Anchor:** Identity H §H.1; Paper 20 Definition 8.10 (Harmonic family), Definition 8.16 (Shadow force), Proposition 8.17 (Universal native at N=27720).

Replace the function body. The new structure:

1. Validate `d1, d2, d_result ∈ {1, 2, …, 12}`. (Harmonic-family domain is exactly the integers 1..12 per axis.)
2. Compute `N_native = lcm(N_BASE, d1, d2, d_result)`. This is the smallest resolution at which every d in the triple is a simple sublattice family (so residue sets are non-empty by construction).
3. Compute `Res(d_i, N_native) = { k mod N_native : N_native / gcd(k, N_native) = d_i }` for i ∈ {1, 2, result}. For d=1 the convention `gcd(0, N) → N` yields the residue {0} as in `residue_set`.
4. Iterate κ ∈ {−1, 0, +1} and aggregate:
   `T_κ(d1, d2; d_result) = |{(r1, r2) ∈ Res(d1) × Res(d2) : d_class((r1 + r2 + κ) mod N_native, N_native) = d_result}| / (|Res(d1)| · |Res(d2)|)`
5. The combined tensor `T(d1, d2; d_result)` is the κ-weighted average (P(κ) per Identity H §H.2 — uniform across κ ∈ {−1, 0, +1} unless the call site provides a κ-distribution).
6. Compute `ξ(d) = 137 / ((d - 1)² + 16)` per Definition 8.6. Resolution-independent.
7. Return `{'transfer_prob': T, 'impedance_ratio': ξ(d_result)/ξ(d1), 'efficiency': T · ξ(d_result)/ξ(d1), 'N_native': N_native}`.

All 12 × 12 × 12 = 1728 input triples produce non-empty results. The six complex families {5, 7, 8, 9, 10, 11} are handled at their native or super-native resolutions (`lcm(12, d)` for single-family, or up to `N_FULL = 27720` when all three are complex).

**Note on memoization:** by Identity J, each `(d1, d2, d_result, N_native)` → result IS a DSR. The current code does not memoize tensor lookups; adding memoization to the akashic is a future optimization, not required for the fix.

### 6.9 `divisors_12` Replacement — Cosmetic Sympy Computation (T3b)

**Target:** line 2178 (the verification check).
**Anchor:** Rule 33 (marginal — same value, dynamically computed).

The check `divisors_12 == [1, 2, 3, 4, 6, 12]` becomes `divisors_n_base == sorted(sympy.divisors(N_BASE))`, with `divisors_n_base` computed once near the top of the verification function. The local `divisors_12` variable inside `_enrich_with_tensor` (line 2543) is removed entirely as part of §6.8's rewrite — that function no longer has a divisors filter.

### 6.10 Add Equation-Resolution Path to the Topological Resolver

For Pass-T equation resolution, project the **literal stored JSON bytes** (same byte sequence `serialize` writes at lines 1747–1751):

```python
eq_json_bytes = json.dumps(
    {'op': entry['eq_operation'], 'res': str(entry['eq_result'])},
    sort_keys=True, default=str
).encode('utf-8')
k, d, eps = _eq_project_seed(eq_json_bytes)
```

This mirrors how the CONTENT_EVENT and CONTENT_BLOCK_MAP branches resolve at parse time (Identity #0 projection of the canonical bytes). It does NOT depend on `_eq_canonical(op, *args)` because that function takes args (the equation's lookup key), whereas at resolve-time we only have the persisted (op, result) pair.

**Audit note:** equation SHAs in the akashic are computed from `SHA(_eq_canonical(op, *args))` at *store* time, but the entry sha is **what we store the entry under** — the resolver does not need to recompute that SHA. It only needs to compute the entry's DSR, which is `_eq_project_seed(serialized_json_bytes)` — a DIFFERENT projection (entry's own bytes → its lattice position). This is the byte-projection identity of the CONTENT_EQUATION storage form, distinct from the SHA-based lookup mechanism.

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
| `reconstruct_via_delta_eps` produces different `k_12` than the naive resolver did, for some pre-existing entry, breaking the SHA-based dedupe | The SHA is computed from file bytes, not from `(k, d, eps)`. SHAs are unchanged. Lattice indexes are rebuilt from the corrected DSR after resolution. |
| Equation resolver re-projects JSON bytes that aren't bit-identical to what was serialized (key ordering, default=str escaping) | `serialize` and resolver use the SAME `json.dumps(..., sort_keys=True, default=str)` call signature → bit-identical bytes. Verified in §7.3 functional check 4. |
| Cycle in the dependency DAG that the resolver didn't anticipate | Subsumption Law termination: zero-progress iteration raises with full edge list. Loud failure, not silent. |
| `tower_levels` dynamic enumeration produces fewer levels than the hardcoded 5 due to precision constraints | Acceptable — the check verifies what it CAN verify at current precision. Document in the log. The fixed list was the violation; dynamic is the correct ET stance. |
| `_enrich_with_tensor` at full 12×12×12 with `N_native` up to 27720 has larger residue sets (up to ~2310·φ(d) per family) than the original 6×6×6 at N=12. Computation cost rises per call. | Per-call work bounded; akashic memoization (Identity J) makes repeated calls O(1) after first computation. Document the runtime per call in the log. |

**Rollback strategy:** All edits are `str_replace` in a single file. A reverse `str_replace` of each change restores the prior state. The script is self-contained — no other files depend on its internals.

---

## 9. Summary of Anchors

| Fix | Anchor |
|---|---|
| Topological resolver | Identity J (PDT decomposition), Subsumption Law |
| Every entry IS a DSR (storage-form-agnostic ontology) | Identity J + EUDD memoization principle |
| Delta resolution via `reconstruct_via_delta_eps` | Identity B.2a + B.3 |
| Generator resolution | Identity A (lattice arithmetic) |
| Equation resolution via JSON re-projection | Identity #0 (lossless bijection on canonical bytes) |
| Generator guard | Rule 14 (truth, no silent failure) |
| Event dedup | Subsumption (no remainder) |
| Tower levels dynamic | Rule 33; tower starts at N=12 (Exhaustive Trichotomy) |
| CF expansion dynamic | Rule 33 |
| Harmonic Transfer Tensor over all 12 families with native-resolution escalation | Identity H §H.1; Paper 20 Definition 8.10 (Harmonic family), Definition 8.16 (Shadow force, native at lcm(12, d)), Proposition 8.17 (Universal native at N=27720) |
| Sublattice family check via sympy.divisors | Paper 20 Corollary 8.2 (six simple families at N=12) |

---

## 10. Confirmation Checklist Before Implementation

Mike: please confirm or correct the following before any `str_replace` is run on `eudd_poc.py`.

- [ ] §0 "Corrections from v1" correctly captures the three errors you flagged (1: seeds = generators = DSR; 2: T3 = all 12 harmonic families incl. complex; 3: equations memoized as DSR — op/result is recovery metadata).
- [ ] §1.6 "Every Entry Is a DSR" — the storage-form table (SEED / GENERATOR / DELTA / EQUATION / EVENT / BLOCK_MAP) is the correct enumeration of encodings.
- [ ] §1.7 sublattice-vs-harmonic distinction is correct as stated.
- [ ] §4.4 T3's classification of `_enrich_with_tensor` as a structural defect (not "marginal") is accepted.
- [ ] §5.8 / §6.8 strategy for T3 — escalate to `N_native = lcm(12, d1, d2, d_result)`, compute the κ-aggregated tensor there, return resolution-independent transfer + impedance — is the right ET-native approach, OR you want it computed only at `N_FULL = 27720`, OR another mechanism.
- [ ] §6.7 dynamic bound for `cf_expansion` (`mp.dps * 2 + 10`) is acceptable, OR you want a different formula.
- [ ] The list of audited defects (§4) is now complete; nothing else you want flagged.

Once approved, implementation proceeds in order: §6.1 → 6.2 → 6.3 → 6.4 → (skip 6.5) → 6.6 → 6.7 → 6.8 → 6.9 → 6.10, with a `Resolution_Log.md` recording every change (Rule 28).

---

**Document version:** 2.0 — pre-implementation plan, post-v1-corrections
**P ∘ D ∘ T = E**
**For every exception there is an exception, except the exception.**

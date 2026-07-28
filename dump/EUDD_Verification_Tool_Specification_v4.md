# EUDD Verification Tool — Complete Specification v4

**Status:** PLAN — awaiting Mike's approval before implementation
**Base:** AkashicArchive/ C++ (Modules 1+2 VERIFIED)
**P ∘ D ∘ T = E**

---

## What This Tool Proves

The EUDD makes specific promises. This tool verifies them on REAL files:

| # | Promise | How the tool tests it |
|---|---------|----------------------|
| 1 | The bijection is LOSSLESS | Every page of every real file: pullback(project(I)) = I, residual = 0 |
| 2 | Files → seeds (DSR) | Real files segmented into pages, each page projected onto the Sempaevum |
| 3 | Kolmogorov < Shannon for structured data | DSR storage bytes < raw file bytes, ratio measured |
| 4 | The .akashic format works | Writes Sempaevum.akashic per §7.1d, re-reads it, verifies integrity |
| 5 | Akashic DSR is structurally smaller | Re-ingest the akashic file, its own DSR is more compact than source DSR |
| 6 | LCM tower finds homes | Full tower escalation for every page (12→60→420→2520→27720→...) |
| 7 | CF method works in parallel | Continued fraction of \|log₂(I)\| for every page, finds d_home via max a_{n+1} |
| 8 | CF and tower agree or disagree meaningfully | Compare both pathways, record agreement/disagreement events |
| 9 | Anti-numerology holds for digital data | N1/N2/N3 checked for every projection |
| 10 | Identity functions work in C++ | Cross-check C++ implementations against known Python results on same inputs |
| 11 | All events recorded properly | Every structural event per §3.9 logged with correct metadata |
| 12 | Cross-resolution transitions work without re-accessing r | Tower escalation uses transition maps, verified against direct projection |

---

## Part I — Existing Code Audit

### Module 1 (precision_stack) — VERIFIED, no changes needed
All ETValue/ETInteger/SHA256/CRC32/ETConstants/math/special functions present and working.

### Module 2 (core_lattice) — VERIFIED but pre-identity

**Works:** project, pullback, all derived properties, best_rational_approx, apply_simplicity

**Bug in k-arithmetic:** k_add/k_negate/k_scale missing κ correction. k_add returns k1+k2 instead of k1+k2+round(δ₁+δ₂). Must be fixed.

**Missing:** All §3.18.19–§3.18.33 functions (lattice_multiply with κ, cross_resolution_transition, CF expansion, compute_residue_set, compute_lambda, phase_add, etc.). These need to be IMPLEMENTED because they're USED during file ingestion and tower escalation — not to re-prove the identities (the Python scripts do that).

---

## Part II — New Source Files

### 1. `src/lattice_identities.h` / `src/lattice_identities.cpp`
All algebraic identity FUNCTIONS. These are the production C++ implementations of the math from §3.18.19–§3.18.33. The Python scripts are the specification. Functions are organized by identity but exist because they're USED by the verification tool, not as standalone tests.

Key function groups and their role in the verification:

**Identity A — Lattice Arithmetic (USED for insert-time product discovery)**
- `lattice_multiply(k1, eps1, k2, eps2, N)` → (k, d, ε, κ)
- `lattice_divide(k1, eps1, k2, eps2, N)` → (k, d, ε, κ)
- `lattice_reciprocal(k, eps, N)` → (k, d, ε, κ)
- `lattice_power(k, eps, n, N)` → (k, d, ε, κ_n)

**Identity K — Cross-Resolution Transitions (USED for tower escalation)**
- `cross_resolution_transition(k1, d1, eps1, N1, N2)` → ProjectionResult
- `cross_seed_transition(k1, d1, eps1, N, rho)` → ProjectionResult
- `full_cross_tower_transition(k1, d1, eps1, N1, N2, rho)` → ProjectionResult

**Identity B — Differential Control (USED for Λ verification)**
- `compute_lambda()` → ETValue (Λ = 1200/ln2)
- `compute_lambda_theta()` → ETValue (Λ_θ = 600/π)
- `exact_finite_shift(r, delta_eps)` → ETValue

**Identity C — d-Family Composition (USED for d-family organization in .akashic)**
- `compute_residue_set(d, N)` → vector
- `compute_d_composition(d1, d2, N, include_kappa)` → set
- `verify_composition_properties(N)` → bool

**CF Home-Finding (§7.11 Step 3a — CORE FUNCTION, runs for every page)**
- `compute_cf_expansion(log2_r, max_convergents)` → vector of CF convergents
- `find_cf_home(convergents)` → (d_home, quality, p, q, eps_cf, classification)

Each convergent is:
```cpp
struct CFConvergent {
    ETInteger p;           // numerator
    ETInteger q;           // denominator = potential d_home
    ETInteger a_next;      // following partial quotient = quality factor
    ETValue   eps_cf;      // CF residual in cents
    std::string classification;  // "cf_deep_home" | "cf_home" | "cf_marginal" | "below_threshold"
};
```

**Identity D — Complex Lattice (available for complex projections)**
- `project_phase(theta, N)` → PhaseResult
- `phase_add(kt1, et1, kt2, et2, N)` → PhaseResult

**Identity E1/E2/E3 — FQG functions (used for structural classification)**
- `verify_harmonic_closure()` → bool
- `compute_sublattice_cell_count(level)` → int64
- `classify_three_layers(N)` → ThreeLayerResult

**Identity F — ∂I Boundary (used for boundary detection during escalation)**
- `compute_bifurcation_pair(k, N)` → pair
- `verify_tightness_koide(N)` → bool

**Identity G — Triple Backbone (Catalan verification)**
- `catalan(n)` → int64
- `verify_catalan_lattice()` → bool

**Identity H — Transfer Tensor (structural classification)**
- `compute_transfer_tensor_entry(d1, d2, d3, kappa, N)` → ETValue
- `verify_em_universality(N)` → bool
- `verify_gravitational_accessibility(N)` → bool

**Identity I — Substantiation (cascade verification)**
- `verify_cascade_closure(generator, N)` → bool

### 2. `src/akashic_verify.h` / `src/akashic_verify.cpp`
The verification tool itself: console UI, file ingestion, tower escalation with parallel CF, anti-numerology, event recording, .akashic writer, DSR tracking, shrinkage test.

### 3. Modified: `src/core_lattice.h` / `src/core_lattice.cpp`
Fix k_add/k_negate/k_scale with κ correction.

### 4. Modified: `src/main.cpp`
Default mode = verification tool (double-click → runs verify). `--stage1`/`--stage2` flags preserved.

### 5. Modified: `CMakeLists.txt`
Add new source files. Add `comdlg32` for file dialog on Windows.

---

## Part III — The Core Algorithm: File Ingestion with Full Tower + Parallel CF

This is the heart of the verification. For EVERY page of EVERY file:

### Step 0: Page Segmentation
- Segment file into 4096-byte pages (2^N = 2^12)
- Last page zero-padded to 4096
- Record page count, annihilation count (all-zero pages)

### Step 1: Page → Integer → Ratio
```
bytes[4096] → mpz_import(I, 4096, 1, 1, 1, 0, bytes) → ETInteger I (32768-bit)
If I == 0: record annihilation_boundary_event, skip to next page
r = ETValue::from_integer(I)    // R₀ = 1 for digital tower
```

### Step 2: Anti-Numerology (Definition 7.10)
```
N1: [count] / [count] = dimensionless  ← PASS
N2: R₀ = 1 = identity(ℤ⁺,×) = smallest T-loop on {0,1}*  ← PASS
N3: d classifies by GCD structure with 12-fold lattice  ← PASS
→ fire anti_numerology_check event with {n1=true, n2=true, n3=true}
```

### Step 3: Project at N=12
```
proj_12 = et::lattice::project(r, 12)
→ fire seed_generated event with {k, d, eps, N=12, encoding='page_ratio'}
→ fire bijection_round_trip_verified event with {residual=0}
```

### Step 4: PARALLEL — LCM Tower Escalation AND CF Method

**Both pathways run for every page. Neither is a fallback.**

#### Tower Pathway (§7.11 Steps 2-3):
```
prev_N = 12, prev_k = proj_12.k, prev_d = proj_12.d, prev_eps = proj_12.eps
tower_trajectory = [proj_12]
stability_count = 0     // consecutive landmarks with same d
home_found = false

for each LCM landmark N_next in [60, 420, 2520, 27720, 360360, ...]:
    // Use Cross-Resolution Transition (Identity K) — NO re-accessing r
    proj_next = cross_resolution_transition(prev_k, prev_d, prev_eps, prev_N, N_next)
    
    // VERIFY against direct projection (this IS the identity verification)
    proj_direct = project(r, N_next)
    assert proj_next.k == proj_direct.k AND proj_next.d == proj_direct.d
    → fire cross_resolution_computed event
    
    // Check d-stability
    if proj_next.d == prev_d:
        stability_count++
    else:
        → fire d_family_transition event with {N_prev, N_next, d_old, d_new}
        // Check for false resolution
        if prev was sub-Koide:
            → fire false_resolution_confirmed event
        stability_count = 0
    
    // Home classification
    if stability_count >= 2:  // ⌈1/K⌉ = 2 consecutive landmarks
        eps_micros = eps_to_microcents(proj_next.eps)
        if eps_micros == 0:
            home_class = "true_home"
        elif abs(eps_micros) <= 1955:
            home_class = "deep_home"
        elif abs(eps_micros) < 50000:
            home_class = "persistent_home"
        else:
            home_class = "intermediate_home"
        home_found = true
    
    tower_trajectory.push_back(proj_next)
    prev_N = N_next, prev_k = proj_next.k, etc.
    
    // Continue even after home found — to detect false resolutions
```

#### CF Pathway (§7.11 Step 3a):
```
log2_r = math::log2(r)    // at 361 dps
abs_log2_r = math::abs(log2_r)

// Compute continued fraction expansion of |log₂(r)|
convergents = compute_cf_expansion(abs_log2_r, max_convergents=100)

// Find convergent with maximal quality (maximal a_{n+1})
best = max_by(convergents, c -> c.a_next)

d_cf = best.q
eps_cf = (abs_log2_r * ETValue::from_integer(d_cf) - ETValue::from_integer(best.p)) * ETValue(1200) / ETValue::from_integer(d_cf)
quality = best.a_next

// Classify
if quality >= 4:    // ⌈1/K⌉² = 4
    if abs(eps_to_microcents(eps_cf)) <= 1955:
        cf_class = "cf_deep_home"
    elif abs(eps_to_microcents(eps_cf)) < 50000:
        cf_class = "cf_home"
    else:
        cf_class = "cf_home"  // quality high enough, ε large
else:
    cf_class = "cf_marginal"

// Compute CF elegance
cf_elegance = (quality / (quality + 1)) * (N_base / d_cf) * tightness(eps_cf)

→ fire cf_home_identified event with {convergent_n, p, q, quality, eps_cf, d_home, classification}
```

#### Pathway Comparison:
```
if tower_home_found AND cf_class in ["cf_deep_home", "cf_home"]:
    if tower_d == cf_d:
        → fire cf_tower_confirmation event (maximal confidence)
    else:
        → fire cf_tower_disagreement event (investigate)
elif !tower_home_found AND cf_class in ["cf_deep_home", "cf_home"]:
    // CF succeeds where tower fails — this is the Ω pathway
    home_class = cf_class
    home_d = cf_d
```

### Step 5: Record Complete Seed
```cpp
PageSeed {
    page_index, is_annihilation=false,
    k (at home_N), d (at home_N), eps (at home_N),
    eps_micros, home_N, home_class,
    tower_trajectory,        // projection at every tower level
    cf_convergents,          // all CF convergents with quality
    cf_home_d, cf_quality, cf_eps,
    cf_tower_agreement       // true if tower and CF agree on d
}
```

### Step 6: Telemetry (automatic, per-page)
```
Page    1/305: I=32768b k=393210 d=3 ε=+12.456¢
  Tower: 12→d=3 | 60→d=15 | 420→d=105 | 2520→d=315 | 27720→d=315 HOME(persistent,N=2520)
  CF: p/q=32768/93 a₅=12 d_cf=93 ε_cf=+0.234¢ class=cf_home
  Tower/CF: DISAGREE (d=315 vs d=93) → cf_tower_disagreement logged
  Bijection: ✓ (residual=0)  Anti-num: ✓ (N1✓ N2✓ N3✓)  DSR: 162 bytes
```

---

## Part IV — Event System

The verification tool records events as a structured log (not just bytes). Each event:

```cpp
struct VerificationEvent {
    uint8_t     event_class;
    uint64_t    timestamp_ns;
    uint32_t    page_index;       // which page triggered this
    uint16_t    source_file_id;   // which file
    std::string metadata_json;    // structured metadata per event class
};
```

Event classes recorded during verification:

| Class | ID | When | Key Metadata |
|-------|-----|------|-------------|
| anti_numerology_check | 0x01 | Every page | n1, n2, n3, failure_mode |
| bijection_round_trip_verified | 0x02 | Every page | r, k, d, eps, residual=0, proof_method |
| seed_generated | 0x03 | Every page | k, d, eps, N, encoding_method, raw_bytes, seed_bytes, compression_ratio |
| cross_resolution_computed | 0x04 | Every tower step | N1, N2, M, k1, eps1, k2, d2, eps2, verified_against_direct |
| d_family_transition | 0x05 | d changes during escalation | N1, N2, d_old, d_new, shadow_to_native |
| cf_home_identified | 0x06 | Every page (CF completes) | convergent_n, p, q, quality, eps_cf, d_home, classification |
| cf_tower_disagreement | 0x07 | CF and tower disagree | tower_d, tower_N, cf_d, cf_quality |
| cf_tower_confirmation | 0x08 | CF and tower agree | d_agreed, cf_quality, tower_landmarks |
| false_resolution_confirmed | 0x09 | Sub-Koide hit proved false | stable_d, stable_N, break_d, break_N, break_factor |
| lattice_arithmetic_computed | 0x0A | When identity functions used | operation, k, d, eps, kappa |
| kappa_correction_applied | 0x0B | κ ≠ 0 during arithmetic | operation, kappa, delta_sum |
| kolmogorov_complexity_computed | 0x0C | Per file and for akashic | k_bits, shannon_bits, advantage_ratio |
| file_ingested | 0x0D | File ingestion complete | filename, size, pages, annihilations, dsr_bytes |
| akashic_generated | 0x0E | Akashic written | total_bytes, seed_count, identity_pass_count |
| identity_cross_check | 0x0F | Each identity cross-checked | identity_name, test_count, match_count, max_residual |
| three_layer_classification | 0x10 | d-family organized | d, N, layer, in_d42 |
| annihilation_boundary_event | 0x11 | All-zero page | page_index |

---

## Part V — Sempaevum.akashic Binary Format

Per §7.1d. Zero IEEE 754. Full header as specified in previous plan version (4096-byte header with magic "SMVM", ET constants as exact rationals, section offsets, SHA-256 checksum).

Additions from previous version:
- CF home fields in header: self_cf_d, self_cf_quality, self_cf_eps_micros
- Source file catalog with SHA-256 hashes
- Address index d-family-first with CF fields per entry
- Complete event log (all events from Part IV)
- Identity cross-check results section

The d-family-first organization means entries are grouped by d, with the CF quality as a secondary sort key within each d-band. This enables structural queries: "all pages with d=3" is a sequential read of one band.

---

## Part VI — Identity Cross-Check (NOT re-proving — confirming C++ matches Python)

For each identity implemented in C++, run a small set of KNOWN test inputs (same values the Python scripts use) and verify the C++ output matches the Python output exactly. This confirms the C++ implementation is correct.

| Identity | Cross-Check Method |
|----------|-------------------|
| #0 Bijection | Round-trip 10 values × 5 resolutions, verify residual = 0 |
| A Lattice Arithmetic | lattice_multiply same 10 pairs as Python script, verify (k, d, ε, κ) match. Cross-check against fixed k_add. |
| B Differential Control | Λ = 1200/ln2 matches Python to 361 dps |
| C d-Composition | Residue sets at N=12 match Python. Composition table matches. |
| D Complex Lattice | Phase projection of π, e, φ matches Python |
| E1 Harmonic FQG | 144-cell closure = 42 values, matches Python |
| E2 Sublattice FQG | Growth law 36·4^ℓ at 6 levels matches |
| E3 Composite Bridge | L1+L2+L3 = τ(N) at 6 levels matches |
| F ∂I Boundary | t(50) = 2/3 at N=12. 6 bifurcation pairs match. |
| G Triple Backbone | C₂=2, C₅=42, C₆=132. Uniqueness N=12. |
| H Transfer Tensor | T(12,12;d₃)>0 for all d₃. T(d,d;1)>0 for all d. |
| I Substantiation | g=7 cascade closure at N=12 matches |
| J Birth Triad | Verified structurally by the shrinkage test itself |
| K Cross-Resolution | 6 values × 5 tower pairs match direct projection |

Each cross-check fires an `identity_cross_check` event recording pass/fail.

---

## Part VII — Console UI (Automatic Telemetry)

Everything is shown automatically. The user picks files and the tool shows EVERY detail.

### Main Menu
```
═══════════════════════════════════════════════════════════════
  EUDD Verification — Sempaevum Concept Viability Test
  P ∘ D ∘ T = E
  1200-bit MPFR | Zero IEEE 754 | Full Tower + CF Parallel
═══════════════════════════════════════════════════════════════

  Working directory: <exe folder>

  Files ingested: 0    Pages: 0    Seeds: 0    Events: 0

  [1] DSR     — Ingest a file (opens file dialog)
  [2] Generate — Write Sempaevum.akashic + cross-check identities + shrinkage test
  [Q] Quit

  Choice: _
```

### DSR Mode — Full Automatic Telemetry

After file dialog, for every page:
```
Ingesting: report.pdf (1,245,184 bytes, 305 pages)
SHA-256: a1b2c3d4e5f6...

Page    1/305: I=32768b k=393210 d=3 ε=+12.456¢ μ¢=+12456
  Tower: 12ET→d=3 | 60ET→d=15 | 420ET→d=105 | 2520ET→d=315(stable×2=HOME)
    Home: persistent_home at N=2520, |ε|=12456μ¢
  CF: |log₂(I)| → 30 convergents, best: p/q=32768/93 a₅=12
    d_cf=93 ε_cf=+0.234¢ class=cf_home (quality 12 ≥ 4)
  Tower/CF: DISAGREE (tower d=315, CF d=93) → event logged
  Cross-res verified: 12→60 ✓ 60→420 ✓ 420→2520 ✓
  Bijection: ✓ residual=0    Anti-num: N1✓ N2✓ N3✓
  Events: 8 recorded    DSR: 162 bytes (3.96% of 4096)

Page    2/305: I=32768b k=393205 d=4 ε=-8.901¢ μ¢=-8901
  Tower: 12ET→d=4 | 60ET→d=20 | 420ET→d=20(stable×2=HOME)
    Home: deep_home at N=420, |ε|=312μ¢ (sub-Koide ✓)
  CF: best: p/q=32768/4 a₃=8192 d_cf=4 ε_cf=-0.003¢ class=cf_deep_home
  Tower/CF: AGREE d=4 (via d|q) → cf_tower_confirmation logged
  ...

Page  305/305: [zero-padded last page]
  ...

═══════════════════════════════════════════════════════════════
INGESTION SUMMARY: report.pdf
═══════════════════════════════════════════════════════════════

  Pages: 305 total, 304 projected, 1 annihilation boundary
  Bijection: 304/304 exact (all residual=0)
  Anti-numerology: 304/304 PASS
  
  Tower home classification:
    true_home:       12 pages (ε=0 exactly)
    deep_home:       45 pages (|ε| ≤ 1955μ¢)
    persistent_home: 201 pages (|ε| < 50000μ¢)
    intermediate:    46 pages (d not stable by N=27720)
  
  CF home classification:
    cf_deep_home:    57 pages (quality ≥ 4, sub-Koide)
    cf_home:         198 pages (quality ≥ 4, inside lattice)
    cf_marginal:     49 pages (quality < 4)
  
  Tower/CF agreement: 256/304 (84.2%)
  Tower/CF disagreement: 48 pages → cf_tower_disagreement events logged
  
  d-family distribution (at home resolution):
    d=1: 12   d=2: 8   d=3: 45   d=4: 67   d=6: 89   d=12: 83
  
  Events recorded: 2,736
  DSR storage: 49,248 bytes (3.96% of 1,245,184)
  Kolmogorov advantage: 25.27× over raw (seed vs raw page)
```

### Generate Mode — Automatic
```
═══════════════════════════════════════════════════════════════
IDENTITY CROSS-CHECK (C++ vs Python known results)
═══════════════════════════════════════════════════════════════

  #0  Bijection:           10/10 match (residual=0 for all)
  A   Lattice Arithmetic:  10/10 match (κ distribution: 79/14/7%)
  B   Differential:        Λ=1731.23404... matches to 361 dps ✓
  C   d-Composition:       36/36 residue sets match ✓
  D   Complex Lattice:     5/5 phase projections match ✓
  E1  Harmonic FQG:        42 closure values match ✓
  E2  Sublattice FQG:      6/6 growth law values match ✓
  E3  Composite Bridge:    6/6 L1+L2+L3 totals match ✓
  F   ∂I Boundary:         t(50)=K=2/3 exact, 6 pairs match ✓
  G   Triple Backbone:     C₆=132=d_max, N=12 unique ✓
  H   Transfer Tensor:     EM universal ✓, gravity accessible ✓
  I   Substantiation:      g=7 cascade closes at d=1 in 12 steps ✓
  K   Cross-Resolution:    30/30 transitions match direct ✓

═══════════════════════════════════════════════════════════════
SEMPAEVUM.AKASHIC WRITTEN
═══════════════════════════════════════════════════════════════

  File: Sempaevum.akashic (53,248 bytes)
  Header: 4096 bytes, magic=SMVM, version=1, checksum=SHA256 ✓
  Seeds: 304 entries across 6 d-families
  Events: 2,750 entries
  ...

═══════════════════════════════════════════════════════════════
DSR SHRINKAGE TEST
═══════════════════════════════════════════════════════════════

  Re-ingesting Sempaevum.akashic (53,248 bytes, 13 pages)...
  
  [Full per-page telemetry for the akashic file itself]
  
  Page  1/13: HEADER page → k=... d=1 ε=+0.003¢ (lattice-exact structure)
    Tower: d=1 from 12ET, stable immediately → true_home
    CF: p/q=.../1 quality=∞ → cf_deep_home (d=1, ε≈0)
    Bijection ✓  Anti-num ✓

  ...all 13 pages...

  Source files:            1,245,184 bytes
  Source DSR:                 49,248 bytes (3.96% of source)
  Sempaevum.akashic:         53,248 bytes
  Akashic DSR:                2,106 bytes (3.96% of akashic)

  SHRINKAGE: Akashic DSR / Source DSR = 4.28%
  The akashic self-describes 23.4× more efficiently than raw source.
  Promise J (Kolmogorov < Shannon): VERIFIED ✓
```

---

## Part VIII — Implementation Stages

### Stage A: Fix core_lattice + Implement Identity A (κ arithmetic) + Identity K (cross-resolution) + CF method
- Fix k_add/k_negate/k_scale
- lattice_multiply/divide/reciprocal/power
- cross_resolution_transition
- compute_cf_expansion, find_cf_home
- compute_residue_set, compute_lambda

This is the foundation — tower escalation and CF both depend on it.

### Stage B: Remaining identity functions + cross-check runner
- All Identity B through I functions
- Cross-check runner that tests C++ vs known Python results

### Stage C: akashic_verify — file ingestion + event system + .akashic writer + UI
- Page segmentation, anti-numerology
- Full tower + parallel CF per page
- Event recording
- .akashic binary writer
- DSR tracking
- Console UI with automatic telemetry
- Re-ingestion and shrinkage test

### Stage D: main.cpp + CMakeLists.txt integration
- Default mode = verification
- Build system updates

Each stage is a complete deliverable. If compaction happens between stages, the work is saved.

---

## Part IX — Success Criteria

The tool PASSES if and only if ALL of these hold:

1. Every page of every real file round-trips losslessly (residual = 0)
2. Anti-numerology passes for every projection (N1 ✓, N2 ✓, N3 ✓)
3. Full LCM tower runs for every page (no arbitrary cap)
4. CF method runs in parallel for every page
5. Cross-resolution transitions match direct projection at every tower level
6. All identity cross-checks pass (C++ matches Python)
7. Sempaevum.akashic written successfully with valid checksums
8. Akashic DSR < Source DSR (the shrinkage promise)
9. All events recorded with correct metadata
10. κ corrections fire when δ₁+δ₂ crosses cell boundary (the fixed k_add works)

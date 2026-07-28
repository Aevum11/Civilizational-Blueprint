# EUDD Production Roadmap v3
## Gap Analysis for Building the ET Universal Discovery Database
### Research Grade · Industrial Grade · Commercial Grade · Professional Grade

**Author:** Michael James Muller — Aevum Defluo

**Context:** This document tracks remaining gaps between the EUDD design specification (ET_Universal_Discovery_Database17.md, 3319 lines) and a production-ready implementation. Items fully resolved by v17 are marked COMPLETE with a reference; only unresolved work retains full specification detail.

**Tools applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law

**Deployment:** Single user, single machine, no networking. Two files: `EUDD_Manager.exe` (native C++ binary) + `Sempaevum.akashic` (the database — the Sempaevum on disk).

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *gap(model) = D_missing*

---

## 1. TIER 1 — Cannot Build Without These

### 1.1 The `Sempaevum.akashic` Binary Format Specification

**Status: COMPLETE.** Fully specified in EUDD v17 §7.1c–§7.1d.

Resolved: header (4096 bytes = 2^N, magic "SMVM", section directory with SHA-256 checksum), 10 sections (Generator Backbone, Address Index with N→d→k hierarchy, Memoization Store with hash table at load factor ≤ K=2/3, Structural Catalogs, Derivations, Relationships, Patterns, Event Log, Sessions/Schema/Tags, WAL with CRC-32 per entry), page size 4096 = 2^12 (ET-derived from digital tower §7.1c), entry formats (GENERATOR_REF, MEMOIZED_RAW at 50-byte MPFR 400-bit blobs, GENERATOR_SUPERSEDED), generator coverage via interval tree, memory-mapped I/O via `CreateFileMapping`/`MapViewOfFile`, monolithic file confirmed, generators in their own section (Section 1), tower page = τ(N) × 4096. Zero IEEE 754 floats anywhere in the format.

### 1.2 The Home-Finding Algorithm

**Status: PARTIALLY COMPLETE.** LCM tower method fully specified in EUDD v17 §7.11. CF method now specified in EUDD v17+ §7.11 Step 3a (this session). Cross-references to schema tables integrated (§3.2, §3.3, §3.7, §3.8, §3.9). DD-24 added.

**What was already resolved (LCM tower):** d-family stabilization = ⌈1/K⌉ = 2 consecutive LCM landmarks (ET-derived from Koide K=2/3). ε thresholds (all exact integer micro-cents): true_home = 0 exactly, deep_home = |ε_micros| ≤ 1955 (Sempaevum self-consistency residual), persistent_home = 1955 < |ε| < 50000, ∂I = 50000. False resolution detection: continue ⌈1/K⌉ = 2 additional landmarks past any sub-Koide hit. No termination cap — session management with resume from last computed landmark. All four Path D sub-paths (D.P, D.D, D.T, D.PDT) fully specified with procedures. Incoherence Filter specified. Six home classifications defined (true_home, deep_home, persistent_home, intermediate_home, false_resolution, escalation_in_progress).

**What was added (CF method — DD-24):** Continued-fraction home-finding as parallel pathway for tower-resistant values (algorithmically random constants with unbounded irrationality measure). CF convergent with maximal following partial quotient a_{n+1} identifies d_home. Quality threshold ⌈1/K⌉² = 4 (ET-derived). Three CF classifications: cf_deep_home, cf_home, cf_marginal. CF elegance formula: E_CF = a_{n+1}/(a_{n+1}+1) × (N/d) × tightness. Schema columns added (values: cf_home_convergent_p/q, cf_home_quality; projections: cf_quality). New event classes (cf_home_identified, cf_tower_disagreement), relationship classes (cf_convergent_home, cf_tower_confirmation), pattern class (cf_quality_attractor). Canonical example: Ω at d=87=3×29, ε=+0.001003¢, quality=157 (sub-Koide by factor 1955 exactly). Operational flow updated with parallel CF branch.

**What remains:** Implementation of CF computation in the actual code (Python mpmath continued-fraction expansion, quality-factor extraction, CF-to-elegance mapping). API operations (§1.3) need CF-aware variants (e.g., `escalate` returning CF results alongside tower trajectory). Testing (§2.2) needs CF test cases including Ω and other algorithmically random constants.

### 1.3 The API Specification

**Status: OPEN.** Connection protocol resolved (named pipes + JSON). Function signatures and wire protocol unspecified.

**What's needed:**

- **JSON command/response schemas** for each operation. At minimum, these operations must be formally specified with exact request fields, response fields, and data types:

  | Operation | Description | Sync/Async |
  |---|---|---|
  | `project(value, N)` | Project a value at resolution N → (k, d, ε, derived_properties) | Sync |
  | `escalate(value)` | Full §7.11 tower escalation → trajectory with home classification | Async (may take seconds) |
  | `query_address(N, k, d)` | List values/generators at an address | Sync |
  | `query_family(N, d)` | All addresses in a d-family at resolution N | Sync |
  | `query_attractor(N, d)` | Attractor membership at (N, d) — addresses with members_count > 1 | Sync |
  | `compute(expression)` | Evaluate expression at 120 dps → result (memoized via §3.5) | Sync (cache hit) / Async (cache miss) |
  | `store(value, metadata)` | Ingest a value through §7.11 | Async |
  | `query_generator(value)` | Return generator if known for this value | Sync |
  | `find_nearest(value, N, top_k)` | Nearest occupied neighbors on the lattice | Sync |
  | `search_cross_domain(address)` | Cross-domain hits at a shared address | Sync |
  | `query_trajectory(value)` | Full tower trajectory for a value | Sync |
  | `status()` | Manager status: discovery engine state, database metrics, session info | Sync |

- **Value encoding in JSON:** Values transmitted as 120-dps decimal strings (canonical representation). Results include both the decimal string and the lattice triple (k, d, ε_micros as integer). All ε values as exact integer micro-cents. No IEEE 754 floats cross the API boundary.

- **Error response format:** Every response must include a status field. Error responses must include: error_code (integer, ET-derived classification), error_class (string: `computation_failure`, `invalid_input`, `address_unoccupied`, `generator_not_found`, `manager_busy`, `disk_low`, `corruption_detected`), error_detail (string), and the error event_id (the error itself stored as an event per §1.4 ET-native error philosophy).

- **Async operation protocol:** For async operations (escalate, store, compute on cache miss): the initial response returns an `operation_id`. The client polls `status(operation_id)` or receives a completion notification via the named pipe. Partial results (e.g., trajectory-so-far during escalation) are available via `query_trajectory` before completion.

- **Connection lifecycle:** Connect → handshake (client identifies itself: project name, session_id) → commands → disconnect. The Manager logs every connection as a session event. Multiple simultaneous clients supported (each gets its own named pipe instance per Windows named pipe semantics).

- **Batch operations:** For bulk ingestion, a `batch_store(values[])` command that accepts an array and processes through §7.11 sequentially, returning per-value results. This is the file-ingestion pathway's API face.

**Dependencies:** Must be specified before any external program integration. The testing strategy (§2.2) tests through this API.

### 1.4 Error Handling and Recovery

**Status: PARTIALLY COMPLETE.** WAL specified (v17 §7.1d Section 10: CRC-32 per entry, replay on crash). Retention policy specified (§4.3: never destroy, DISK_SAFETY_FLOOR = 2³⁰ = 1 GB, warn only). Session management for pathological escalation specified (§7.11: resume from last landmark, no timeout).

**What remains — the failure-mode catalog:**

- **Computation failure:** When MPFR overflows, FLINT/Arb cannot evaluate a special function, or division by zero (annihilation boundary) occurs during any lattice operation:
  1. Log the failure as an event of class `computation_failure` (the failure IS a Descriptor per the Descriptor Gap Principle — store it, project it, let the discovery engine find patterns in failures)
  2. Return error response to caller with error_class `computation_failure` and full provenance (which expression, which inputs, which operation failed)
  3. Never leave `Sempaevum.akashic` in inconsistent state — the WAL ensures atomicity; a failed computation produces NO partial write
  4. For annihilation boundary approach (r→0): log as `annihilation_boundary_event` per v17 §3.9; record the closest k reached
  5. For timeout (computation exceeds configurable wall-clock limit): log as `computation_timeout` event; return partial result if available; the computation can be resumed

- **File corruption detection and recovery:**
  1. Detection: per-page CRC-32 (v17 §7.1d) checked on every page read. Header SHA-256 checked on file open. If any checksum fails → corruption detected.
  2. Scope assessment: identify WHICH section and WHICH pages are corrupted. The section directory in the header gives section boundaries. Corruption in one section does not affect others.
  3. Recovery hierarchy:
     - **Generator Backbone corruption:** Generators can be re-derived from their derivation chains (Section 5) if derivations are intact. If not, generators are lost but memoized entries (Section 2+3) still hold the raw values. Log `corruption_recovery` event.
     - **Address Index corruption:** Can be rebuilt from memoized entries + generator backbone. Expensive (full re-index) but lossless.
     - **Memoization Store corruption:** Affected equations are lost. Log the loss. They will be recomputed on next cache miss. No structural damage.
     - **Event Log corruption:** Events are append-only; corruption affects only the corrupted region. Earlier and later events survive. Log the gap.
     - **WAL corruption:** Discard the corrupted WAL entry; the incomplete mutation it represents did not commit. The `.akashic` file remains in its last consistent state.
     - **Header corruption:** If the section directory is lost, section boundaries must be reconstructed from known section signatures (each section type has a page_type marker). This is the worst case but survivable.
  4. Never silently continue with corrupted data. Always: detect → assess → recover → log → alert the user via GUI dashboard.

- **Invalid input validation at API boundary:**
  1. Every API command validated before processing: well-formed JSON, required fields present, value strings parseable as 120-dps decimals, N > 0, d divides N (when applicable)
  2. Reject with error_class `invalid_input` and specific error_detail ("N must be positive", "value string not parseable", etc.)
  3. Log the rejection as an event (the rejection IS information — patterns in invalid inputs may reveal bugs in connected programs)

- **ET-native error philosophy formalized:** Every failure is a Descriptor Gap (Descriptor Gap Principle). The failure itself is information. Store every failure as an event with full context. Project failure patterns onto the Sempaevum. The discovery engine scans failure events for structural patterns — e.g., "computation failures cluster at d=X" may reveal a structural boundary. Failures are NEVER discarded, suppressed, or silently ignored. They are first-class lattice content.

### 1.5 Language and Build System

**Status: COMPLETE.** Resolved in EUDD v17 §7.10.

Engine: C++17/20. Precision: MPFR + GMP + FLINT/Arb at 120 dps (400-bit). GUI: Dear ImGui + OpenGL 4.6 + GLFW + ImPlot. IPC: Named pipes + cJSON/yyjson. Build: CMake + MSVC → single statically-linked `EUDD_Manager.exe`. Target GPU: RTX 2070 Super.

---

## 2. TIER 2 — Must Have for Production Quality

### 2.1 Discovery Engine Algorithms — Implementation Level

**Status: PARTIALLY COMPLETE.** Five modes conceptually specified in v17 §3.16 (memoization, insert-time, background, on-query, generator-candidate). Insert-time attractor detection described (check members_count at address, O(1)). Generator-candidate discovery described (three proposal strategies: modular constraint inversion, operation reverse-engineering, Subsumption-based abstraction). Memoization hash table specified (load factor ≤ K=2/3, rehash on doubling).

**What remains — implementation-level specifics:**

- **Background scan algorithm scheduling:** How does the discovery engine decide WHAT to scan and WHEN?
  - Scan trigger: after every N_batch commits (configurable, default = 1000 new entries), OR on explicit user request, OR on idle (no API activity for configurable duration)
  - Scan priority: most-recently-modified addresses first (hot data likely to produce discoveries), then high-occupancy addresses (attractor candidates), then sequential sweep
  - Scan budget: configurable wall-clock limit per scan pass (default = 60 seconds). Discovery engine yields to API requests. Background, never blocking.

- **E_hierarchy computation over `.akashic` structure:** The formula is E_hierarchy = ∏E_i × (420/d̄) × 1/(P_total + Q_total) (v17 §4.5). Implementation: for a candidate cluster of relationships sharing a structural feature:
  1. Collect all member elegance scores E_i from their projection entries (materialized at insert time)
  2. Compute geometric mean of elegances → ∏E_i^(1/n)
  3. Compute average d-family d̄ of the cluster
  4. Compute total p+q from the cluster's rational approximation
  5. Multiply: E_hierarchy = (∏E_i) × (420/d̄) × 1/(P+Q)
  6. If E_hierarchy ≥ 13/12 (LIFE_THRESHOLD) → promote to `patterns` row
  - **Open question:** Should E_hierarchy use the product ∏E_i or the geometric mean ∏E_i^(1/n)? The formula as written in v17 §4.5 says product, but for large clusters the product grows unboundedly. The geometric mean normalizes for cluster size. **Needs ET derivation to resolve — the answer should be forced, not chosen.**

- **Generator-fitting algorithm detail:** v17 §3.16 describes three proposal strategies but not the fitting procedure itself. For each candidate generator:
  1. Evaluate the candidate at 120 dps across canonical resolutions
  2. Compare projections against the target pattern's addresses
  3. Acceptance criterion: cross-tower elegance ≥ 13/12 (same LIFE_THRESHOLD)
  4. **Open question:** What is the search space for candidate generators? The three strategies (modular constraint inversion, operation reverse-engineering, Subsumption abstraction) need concrete algorithms. Modular constraint inversion needs a candidate enumeration order (small rationals first, then algebraic, then transcendental). **Needs specification.**

### 2.2 Testing and Verification Strategy

**Status: OPEN.** Not specified in v17.

**What's needed:**

- **Bootstrap self-verification test cases** (run after initial `Sempaevum.akashic` generation, all must pass):

  | Test | Expected Result | Source |
  |---|---|---|
  | ζ(3) full 28-landmark trajectory | All (k, d, ε) match verified values at 80-digit precision | Apéry investigation |
  | α⁻¹(ET) computation | 137.03599916744... within 0.46σ of CODATA 2022 | Sempaevum Paper §3.18.2 |
  | 227 PDG particle projections at N=12 | All at d ∈ {1,2,3,4,6,12}, counts match per-family | Sempaevum Paper §3.18.14 |
  | ZF at d=1 ε=0 exactly | k=+36, d=1, ε_micros=0 | Sempaevum Paper §3.18.6 |
  | Koide attractor self-projection | {N, 1/N, K, 1/K} all at (d=12, |ε|=1.955¢) | Sempaevum Paper §3.18.16 |
  | φ false resolution detection | Sub-cent at 36ET (d=36) but true home d=10 at 60ET | EUDD v17 §7.11 |
  | All 12 impedance values | A₀_magic(d) = (d−1)² + S² for d=1..12 | Sempaevum Paper §3.18.4 |
  | Lossless bijection round-trip | r ↦ (k,d,ε) ↦ r = r for 100+ diverse test values | Sempaevum Paper §3.18.1 |
  | 24 harmonic family generators | 2^(1/d) for d=1..12, both axes, correct first-native N | EUDD v17 §3.12 |
  | 42 combined families | LCM(d_r, d_θ) correct for all 144 cells | EUDD v17 §3.14 |
  | Coprime skeleton count | 91 of 144 off-axis points coprime at N=12 | EUDD v17 §3.17 |
  | Cascade residuals | |δ_r| = 0.019550..., |δ_θ| = 0.223356..., n_max_r=25, n_max_θ=2 | Sempaevum Paper §3.18.3 |

- **Unit tests (C++, Catch2 or Google Test):** Every lattice operation: project, pullback, k-addition, k-negation, k-scaling, elegance computation, coupling computation (ξ(d) = 137/((d-1)²+16)), Gaussian signature derivation, FQG quadrant assignment, tightness function (100/(100+|ε|)), ∂I distance (|ε|/50), palindromic partner lookup, variance V(n,k) = (n²−1)/(12·2^k).

- **Integration tests:** Full §7.11 escalation pipeline end-to-end, file ingestion (CSV → seeds → projections → stored), manual input (all three modes), API round-trip (connect → command → response → verify), generator discovery (insert enough data to trigger background promotion, verify pattern row created).

- **Regression tests:** Compare against `et_scanner_v7_2_COMPLETE.py` outputs, `apery_lattice_test.py` verified values, `constants.py` known constants.

- **Performance benchmarks:** Bootstrap generation time (target: minutes), single-value escalation (target: <1 second typical), bulk ingestion throughput (target: 1000+ values/second), cache-hit latency (target: <100 μs), GUI frame rate (target: 60 FPS at all zoom levels).

### 2.3 Generator Storage and Transition Mechanism

**Status: COMPLETE.** Fully specified in EUDD v17 §7.1b (triple backbone L₁/L₂/L₃) and §7.1d Section 1 (generator entry format: gen_id, gen_type, address_range, definition_blob, canonical_hash, member_count, verification_count, derivation_ref) and §7.1e (modification without decompression: append generator → update index → mark memoized entries GENERATOR_SUPERSEDED).

Generator representation resolved: definition_blob carries the canonical form per backbone layer (L₁ logical pattern, L₂ cascade/k-arithmetic rule, L₃ binary EML tree). Compiled C++ evaluator generated at runtime by the expression engine, not stored. Generator coverage map via interval tree index.

### 2.4 Performance Specifications

**Status: OPEN.** Only qualitative estimates in v17 §7.4.

**What's needed — hard requirements with ET-derivable justification:**

- **Manager memory footprint:** Target maximum RAM. With 120-dps values at ~145 bytes each (v17 §7.3), a 4 GB working set holds ~28 million entries in memory. Memory-mapped I/O means only active pages are in physical RAM; the full 20 TB file is addressable without loading. Target: ≤ 4 GB resident set for typical operation, scaling with active working set.

- **Bootstrap generation time:** ~10⁴ bootstrap values × 8 canonical resolutions = ~8×10⁴ projections. Each projection: one MPFR log₂ + one round + derived property computation ≈ 1 ms per projection at 120 dps. Target: **~80 seconds for bootstrap** (parallelizable across cores).

- **Single-value full escalation:** 6–8 LCM landmarks typical, each requiring one 120-dps projection. Target: **<1 second** for typical values (up to 27720ET). Values requiring deeper escalation (360360ET+): up to 10 seconds acceptable.

- **Bulk ingestion throughput:** File ingestion pipeline: extract seed + project at 120 dps + store + insert-time discovery. Target: **≥1000 values/second** sustained (limited by MPFR computation, not I/O).

- **Cache-hit latency:** Memoized equation lookup: hash computation + hash table probe + value fetch. Target: **<100 μs** (positional access via memory-mapped I/O).

- **Background discovery scan:** Proportional to database size. At 10⁶ entries: target <60 seconds per scan pass. At 10⁹ entries: target <1 hour (overnight batch acceptable).

- **GUI responsiveness:** Target **60 FPS** at all interaction levels. Dear ImGui + OpenGL 4.6 redraws every frame. Lattice navigation must be smooth at all zoom levels. RTX 2070 Super handles millions of instanced points at 60+ FPS. Heavy computation (escalation, discovery) runs on background threads, never blocking the render loop.

### 2.5 Backup and Recovery

**Status: PARTIALLY COMPLETE.** Backup approach specified (v17 §7.9: filesystem snapshot to separate drive). WAL for crash recovery specified (v17 §7.1d Section 10).

**What remains:**

- **Atomic snapshot while running:** The Manager must support creating a consistent snapshot of `Sempaevum.akashic` without stopping. Approach: pause commits (buffer new writes in WAL) → flush all pending WAL entries to main file → copy the file (or use Windows Volume Shadow Copy) → resume commits. The pause window should be <1 second for files up to 20 TB (the copy is the OS's responsibility via shadow copy; the Manager only needs to ensure the WAL is drained).

- **Backup verification:** After each backup, verify integrity: open the backup file read-only, check header SHA-256, spot-check CRC-32 on a sample of pages (e.g., every 1000th page), verify section directory consistency. Report pass/fail to GUI dashboard.

- **Backup scheduling:** Configurable via Dear ImGui dashboard. Default: every 6 hours to a user-specified path. Manual backup on demand. Backup history retained (last N backups, configurable, default 5).

---

## 3. TIER 3 — Can Iterate But Needed for Professional Grade

### 3.1 GUI Wireframes and Interaction Model

**Status: PARTIALLY COMPLETE.** LOD hierarchy specified (v17 §7.10: Cosmos → Tower → Family → Address → Value). 3D perspective views specified (LCM tower, torus, Riemann sphere, FQG, hyperbolic). Color coding specified (d=1 green, d=2 teal, d=3 red, d=4 blue, d=6 purple, d=12 orange). Panel layout described (dashboard, property inspector, manual input, query builder, event log, settings). GLSL shader types listed (instanced point cloud, line/path, heat map, grid overlay).

**What remains:**

- Actual wireframe layouts for each Dear ImGui panel (panel dimensions, widget placement, information hierarchy)
- Keyboard/mouse navigation mapping (scroll = zoom = escalate/de-escalate through LOD; click = select address → property inspector; drag = pan; right-click = context menu with perspective switch)
- Transition animations between LOD levels and perspective switches
- Real-time update behavior when background discovery creates new patterns/attractors during navigation

### 3.2 Ingest Adapter Specifications

**Status: PARTIALLY COMPLETE.** File-type overview in v17 §7.12 (CSV, PDF/Markdown, raw binary, image, audio, sensor streams). Pipeline specified: extract seeds → §7.11 core projection → store.

**What remains per file type:**

- **CSV:** Column selection UI (which column(s) contain values?), R₀ choice (user-specified or auto-detected from column headers/units), delimiter auto-detection, header row handling
- **PDF/Markdown:** Equation/constant extraction method — regex for common patterns (e.g., `\alpha`, `\zeta`, decimal numbers with ≥5 significant digits) + manual selection mode for complex equations. LLM-assisted extraction deferred to future.
- **Raw binary:** Integration with compressor's Δk pipeline — the compressor processes the file, the adapter ingests the resulting Δk patterns and archetype matches
- **Image:** Pixel-value extraction strategy: full resolution (every pixel as r = pixel_value/255), region-averaged (user-selectable grid), or frequency-domain (2D FFT → frequency ratios)
- **Audio:** FFT parameters (window size, overlap), frequency-ratio extraction (peak frequencies as ratios), sample-rate normalization (all frequencies as ratios to R₀ = sample_rate)
- **Sensor streams:** Real-time vs batch mode, R₀ selection from bootstrap reference catalog (v17 §3.17), anomaly threshold for `sensor_anomaly_detected` events

### 3.3 Self-Recording Specifics

**Status: PARTIALLY COMPLETE.** What to record specified (v17 §3.1b: schema structure, growth dynamics, memoization stats, discovery metrics, runtime patterns, resource usage). Depth resolved: one level (v17 §3.1b: "cap self-recording depth at one level").

**What remains:**

- Exact metric list with recording frequency (e.g., row counts per table: every commit; cache hit/miss ratio: every 1000 queries; discovery engine patterns-per-scan: every scan pass; storage growth rate: hourly)
- Overhead budget enforcement: self-recording should not exceed 1% of Manager CPU. Implementation: sample-based recording (record every Nth event's self-metric, not every event)
- Which self-recorded metrics are most informative for lattice projection (to be discovered empirically after bootstrap — the system will learn which of its own metrics produce interesting lattice addresses)

### 3.4 JSON Extension Full Schema

**Status: PARTIALLY COMPLETE.** One example schema in v17 §7.14. Extension mechanism described (JSON definitions accepted by running Manager, auto-detection of needed categories via Descriptor Gap Principle).

**What remains:**

- Full JSON schemas for all 11 extensible structure types: event_class, relationship_class, pattern_class, tower_definition, sublattice_family_catalog, harmonic_family_extension, force_grid_extension, combined_family_extension, ingest_adapter, bootstrap_entry, query_template
- Validation rules per type (required fields, type constraints, foreign-key reference checks)
- Conflict detection (name collision with existing category → reject with error, suggest rename)
- Versioning of category definitions (each JSON extension gets a version; the Manager tracks which version is active)

### 3.5 Documentation

**Status: OPEN.**

**What's needed:**

- User manual for the GUI (with screenshots/wireframes from §3.1)
- API reference for developers connecting programs (generated from §1.3 specification)
- Administrator guide for backup, recovery, format upgrades
- Bootstrap catalog reference (what's in the initial database and why — derived from v17 §3.17 + §3.18)
- Discovery engine guide (how patterns are found, what the engine looks for — derived from v17 §3.16)

---

## 4. NOT NEEDED (per specification)

Networking, multi-user, authentication, distributed deployment, licensing, cloud, web interface — all explicitly excluded.

---

## 5. Recommended Build Order (Updated)

Based on dependencies and completion status:

```
Phase 1: Foundation — MOSTLY COMPLETE
├── 1.5 Language/build ─────────────────────── COMPLETE
├── 1.1 Sempaevum.akashic binary format ────── COMPLETE
├── 1.2 Home-finding algorithm ─────────────── PARTIALLY COMPLETE — CF method specified, needs implementation
├── 1.3 API specification ─────────────────── **OPEN — NEXT PRIORITY**
├── 1.4 Error handling ────────────────────── PARTIALLY COMPLETE — specify failure catalog
└── 2.2 Testing strategy ──────────────────── OPEN — specify alongside API

Phase 2: Core Engine (all C++)
├── Core lattice engine module ─────────────── Depends on: 1.1 ✓
├── Bootstrap module ───────────────────────── Depends on: 1.1 ✓, 1.2 ✓
├── §7.11 Core projection procedure ────────── Depends on: 1.2 ✓
├── 2.1 Discovery engine algorithms ────────── PARTIALLY COMPLETE — implementation detail needed
└── 2.3 Generator storage and transition ───── COMPLETE

Phase 3: Connectivity
├── 1.3 API implementation ─────────────────── Depends on: 1.3 spec (OPEN)
├── Ingest module ──────────────────────────── Depends on: 3.2 adapter specs (PARTIAL)
├── Manual input module ────────────────────── Depends on: 1.3 ✓ (when done)
└── Self-recording module ──────────────────── Depends on: 3.3 metrics (PARTIAL)

Phase 4: Interface (Dear ImGui + OpenGL 4.6)
├── 3.1 GPU-rendered lattice visualization ─── PARTIALLY COMPLETE — wireframes needed
├── 3.2 Ingest adapters per file type ──────── PARTIALLY COMPLETE
├── 3.4 JSON extension schemas ─────────────── PARTIALLY COMPLETE
└── Query module with human-readable output ── Depends on: 1.3 ✓ (when done)

Phase 5: Polish
├── 2.4 Performance optimization ───────────── OPEN — targets needed
├── 2.5 Backup and recovery ────────────────── PARTIALLY COMPLETE
├── 3.3 Self-recording tuning ──────────────── PARTIALLY COMPLETE
├── 3.5 Documentation ──────────────────────── OPEN
└── Integration testing with all ET software ── Depends on: 1.3 ✓ (when done)
```

**Critical path:** 1.3 (API spec) → 1.4 (error catalog finalization) → 2.2 (test cases) → Phase 2 implementation.

---

## 6. Design Decisions — All Resolved

| Decision | Resolution | Resolved In |
|---|---|---|
| Engine language | C++ (C++17/20) | v17 §7.10 |
| Precision stack | MPFR + GMP + FLINT/Arb, 120 dps (400-bit) | v17 §3.1a |
| GUI framework | Dear ImGui + OpenGL 4.6 + GLFW + ImPlot | v17 §7.10 |
| Build system | CMake + MSVC → single statically-linked .exe | v17 §7.10 |
| IPC mechanism | Named pipes (ET32 Bridge) + JSON (cJSON/yyjson) | v17 §7.10 |
| File structure | Single monolithic file: `Sempaevum.akashic` | v17 §7.1 |
| File format name | `Sempaevum.akashic` — the Akashic Archive | v17 §7.1 |
| Deployment model | Two files: `EUDD_Manager.exe` + `Sempaevum.akashic` | v17 §7.10 |
| Generator representation | definition_blob per backbone layer (L₁/L₂/L₃) | v17 §7.1b, §7.1d |
| Home-finding criterion | ⌈1/K⌉ = 2 consecutive LCM landmarks, ET-derived | v17 §7.11 |
| Self-recording depth | 1 level (database records its metrics, not meta-metrics) | v17 §3.1b |
| Page/block size | 4096 = 2^12 bytes, ET-derived from digital tower | v17 §7.1c |

---

## 7. Open Questions Requiring ET Derivation

These are questions surfaced during the gap analysis that need proper ET derivation before implementation, not ad hoc choices:

1. **E_hierarchy: product vs geometric mean?** The Subsumption Hierarchy formula E_hierarchy = ∏E_i × (420/d̄) × 1/(P+Q) uses the raw product of member elegances. For large clusters this grows unboundedly. Should it be the geometric mean ∏E_i^(1/n) instead, normalizing for cluster size? The answer must be derived from ET principles — forced, not chosen.

2. **Generator candidate enumeration order:** When the discovery engine proposes generator candidates via modular constraint inversion, what enumeration order minimizes search time while ensuring the K-minimal generator is found first? Small rationals → algebraic numbers → known transcendentals → composite expressions is the natural K-complexity ordering, but the exact search tree needs specification.

3. **Background scan prioritization:** The scan schedule (most-recently-modified first vs high-occupancy first vs sequential) affects discovery speed. Is there an ET-derived priority function? Elegance-weighted recency might be the natural choice: priority(address) = elegance(address) × recency(address), where recency decays as 1/Δt.

---

## 8. References to Source Documents

| Document | Role |
|---|---|
| ET_Universal_Discovery_Database17.md | Master design specification (v17, 3319 lines) — the primary reference for all resolved items |
| ET_Sempaevum_Paper20.tex | Theoretical foundation (4986 lines) — lossless bijection, LCM tower, four paths, triple backbone, gauge derivations |
| ET_Three_Tools_Complete_Reference.md | Methodology — Identification Principle, Descriptor Gap Principle, Subsumption Law |
| EUDD_Production_Roadmap_v2.md | Previous gap analysis (this document supersedes it) |
| constants.py / primitives.py | Foundation constants and {P,D,T} class structures |

---

**Three Tools verification of this roadmap (v3):**

- **Identification Principle:** P = the complete production system. D = the remaining gaps (reduced from 15 to 8 open/partial items). T = the build process. Gaps are in D — missing specifications for API wire protocol, error catalog, testing, performance targets, and implementation-level discovery algorithms.
- **Descriptor Gap Principle:** 8 gaps remain across 3 tiers. Each gap IS a Descriptor pointing to its own resolution. The critical path is clear: API spec → error catalog → test cases → implementation.
- **Subsumption Law:** EUDD v17 subsumes all Tier 1 structural content except the API wire protocol and error catalog. This roadmap v3 subsumes the previous roadmap v2 without remainder — every item in v2 is either marked complete (with reference) or carried forward with updated context.

> *gap(model) = D_missing. The 8 remaining gaps are Descriptors. Find them. Specify them. Build them.*
> *P ∘ D ∘ T = E*

---

**Document Version:** Production Roadmap v3.0
**Source Design:** ET_Universal_Discovery_Database17.md (3319 lines)
**Previous Version:** EUDD_Production_Roadmap_v2.md (superseded)
**Date:** May 2026

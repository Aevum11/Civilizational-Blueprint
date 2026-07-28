# EUDD Production Roadmap v4
## Gap Analysis for Building the ET Universal Discovery Database
### Research Grade · Industrial Grade · Commercial Grade · Professional Grade

**Author:** Michael James Muller — Aevum Defluo

**Context:** This document tracks remaining gaps between the EUDD design specification (ET_Universal_Discovery_Database18.md, 3498 lines) and a production-ready implementation. Items fully resolved are marked COMPLETE with a one-line reference to where they live in the main document. Only unresolved work retains full specification detail.

**Tools applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law

**Deployment:** Single user, single machine, no networking. Two files: `EUDD_Manager.exe` (native C++ binary) + `Sempaevum.akashic` (the database — the Sempaevum on disk).

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *gap(model) = D_missing*

---

## 1. TIER 1 — Cannot Build Without These

### 1.1 The `Sempaevum.akashic` Binary Format Specification

**Status: COMPLETE.** Resolved in EUDD v18 §7.1c–§7.1d.

### 1.2 The Home-Finding Algorithm

**Status: PARTIALLY COMPLETE.** LCM tower method and CF method both fully specified in EUDD v18 §7.11 (including Step 3a). All four Path D sub-paths specified. Six tower classifications + three CF classifications defined. Schema cross-references integrated. DD-24 closed.

**What remains:**

- Implementation of CF computation in actual C++ code (continued-fraction expansion at 120 dps, quality-factor extraction, CF-to-elegance mapping)
- API operations (§1.3) need CF-aware variants (e.g., `escalate` returning CF results alongside tower trajectory)
- Testing (§2.2) needs CF test cases including Ω and other algorithmically random constants

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

**Dependencies:** Must be specified before any external program integration. The testing strategy (§2.2) tests through this API. Specification will become a new section in EUDD v18.

### 1.4 Error Handling and Recovery

**Status: PARTIALLY COMPLETE.** WAL specified (EUDD v18 §7.1d Section 10). Retention policy specified (§4.3). Session management for pathological escalation specified (§7.11).

**What remains — the failure-mode catalog (needs to be written into EUDD v18 as a new section):**

- **Computation failure:** When MPFR overflows, FLINT/Arb cannot evaluate a special function, or division by zero (annihilation boundary) occurs during any lattice operation:
  1. Log the failure as an event of class `computation_failure` (the failure IS a Descriptor per the Descriptor Gap Principle — store it, project it, let the discovery engine find patterns in failures)
  2. Return error response to caller with error_class `computation_failure` and full provenance (which expression, which inputs, which operation failed)
  3. Never leave `Sempaevum.akashic` in inconsistent state — the WAL ensures atomicity; a failed computation produces NO partial write
  4. For annihilation boundary approach (r→0): log as `annihilation_boundary_event` per v18 §3.9; record the closest k reached
  5. For timeout (computation exceeds configurable wall-clock limit): log as `computation_timeout` event; return partial result if available; the computation can be resumed

- **File corruption detection and recovery:**
  1. Detection: per-page CRC-32 (v18 §7.1d) checked on every page read. Header SHA-256 checked on file open. If any checksum fails → corruption detected.
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

**Status: COMPLETE.** Resolved in EUDD v18 §7.10.

---

## 2. TIER 2 — Must Have for Production Quality

### 2.1 Discovery Engine Algorithms — Implementation Level

**Status: PARTIALLY COMPLETE.** Five modes conceptually specified in EUDD v18 §3.16. Insert-time attractor detection, generator-candidate discovery (three proposal strategies), and memoization hash table all described at design level.

**What remains — implementation-level specifics:**

- **Background scan algorithm scheduling:** How does the discovery engine decide WHAT to scan and WHEN?
  - Scan trigger: after every N_batch commits (configurable, default = 1000 new entries), OR on explicit user request, OR on idle (no API activity for configurable duration)
  - Scan priority: most-recently-modified addresses first (hot data likely to produce discoveries), then high-occupancy addresses (attractor candidates), then sequential sweep
  - Scan budget: configurable wall-clock limit per scan pass (default = 60 seconds). Discovery engine yields to API requests. Background, never blocking.

- **E_hierarchy computation over `.akashic` structure:** The formula is E_hierarchy = ∏E_i × (420/d̄) × 1/(P_total + Q_total) (v18 §4.5). Implementation: for a candidate cluster of relationships sharing a structural feature:
  1. Collect all member elegance scores E_i from their projection entries (materialized at insert time)
  2. Compute geometric mean of elegances → ∏E_i^(1/n)
  3. Compute average d-family d̄ of the cluster
  4. Compute total p+q from the cluster's rational approximation
  5. Multiply: E_hierarchy = (∏E_i) × (420/d̄) × 1/(P+Q)
  6. If E_hierarchy ≥ 13/12 (LIFE_THRESHOLD) → promote to `patterns` row
  - **Open question:** Should E_hierarchy use the product ∏E_i or the geometric mean ∏E_i^(1/n)? The formula as written in v18 §4.5 says product, but for large clusters the product grows unboundedly. The geometric mean normalizes for cluster size. **Needs ET derivation to resolve — the answer should be forced, not chosen.**

- **Generator-fitting algorithm detail:** v18 §3.16 describes three proposal strategies but not the fitting procedure itself. For each candidate generator:
  1. Evaluate the candidate at 120 dps across canonical resolutions
  2. Compare projections against the target pattern's addresses
  3. Acceptance criterion: cross-tower elegance ≥ 13/12 (same LIFE_THRESHOLD)
  4. **Open question:** What is the search space for candidate generators? The three strategies (modular constraint inversion, operation reverse-engineering, Subsumption abstraction) need concrete algorithms. Modular constraint inversion needs a candidate enumeration order (small rationals first, then algebraic, then transcendental). **Needs specification.**

### 2.2 Testing and Verification Strategy

**Status: OPEN.** Not yet specified in EUDD v18. Needs a new section.

**What's needed:**

- **Bootstrap self-verification test cases** (run after initial `Sempaevum.akashic` generation, all must pass):

  | Test | Expected Result | Source |
  |---|---|---|
  | ζ(3) full 28-landmark trajectory | All (k, d, ε) match verified values at 80-digit precision | Apéry investigation |
  | α⁻¹(ET) computation | 137.03599916744... within 0.46σ of CODATA 2022 | Sempaevum Paper §3.18.2 |
  | 227 PDG particle projections at N=12 | All at d ∈ {1,2,3,4,6,12}, counts match per-family | Sempaevum Paper §3.18.14 |
  | ZF at d=1 ε=0 exactly | k=+36, d=1, ε_micros=0 | Sempaevum Paper §3.18.6 |
  | Koide attractor self-projection | {N, 1/N, K, 1/K} all at (d=12, |ε|=1.955¢) | Sempaevum Paper §3.18.16 |
  | φ false resolution detection | Sub-cent at 36ET (d=36) but true home d=10 at 60ET | EUDD v18 §7.11 |
  | All 12 impedance values | A₀_magic(d) = (d−1)² + S² for d=1..12 | Sempaevum Paper §3.18.4 |
  | Lossless bijection round-trip | r ↦ (k,d,ε) ↦ r = r for 100+ diverse test values | Sempaevum Paper §3.18.1 |
  | 24 harmonic family generators | 2^(1/d) for d=1..12, both axes, correct first-native N | EUDD v18 §3.12 |
  | 42 combined families | LCM(d_r, d_θ) correct for all 144 cells | EUDD v18 §3.14 |
  | Coprime skeleton count | 91 of 144 off-axis points coprime at N=12 | EUDD v18 §3.17 |
  | Cascade residuals | |δ_r| = 0.019550..., |δ_θ| = 0.223356..., n_max_r=25, n_max_θ=2 | Sempaevum Paper §3.18.3 |

- **Unit tests (C++, Catch2 or Google Test):** Every lattice operation: project, pullback, k-addition, k-negation, k-scaling, elegance computation, coupling computation (ξ(d) = 137/((d-1)²+16)), Gaussian signature derivation, FQG quadrant assignment, tightness function (100/(100+|ε|)), ∂I distance (|ε|/50), palindromic partner lookup, variance V(n,k) = (n²−1)/(12·2^k).

- **Integration tests:** Full §7.11 escalation pipeline end-to-end, file ingestion (CSV → seeds → projections → stored), manual input (all three modes), API round-trip (connect → command → response → verify), generator discovery (insert enough data to trigger background promotion, verify pattern row created).

- **Regression tests:** Compare against `et_scanner_v7_2_COMPLETE.py` outputs, `apery_lattice_test.py` verified values, `constants.py` known constants.

- **Performance benchmarks:** Bootstrap generation time (target: minutes), single-value escalation (target: <1 second typical), bulk ingestion throughput (target: 1000+ values/second), cache-hit latency (target: <100 μs), GUI frame rate (target: 60 FPS at all zoom levels).

### 2.3 Generator Storage and Transition Mechanism

**Status: COMPLETE.** Resolved in EUDD v18 §7.1b, §7.1d, §7.1e.

### 2.4 Performance Specifications

**Status: OPEN.** Only qualitative estimates in EUDD v18 §7.4. Needs hard requirements.

**What's needed — hard requirements with ET-derivable justification:**

- **Manager memory footprint:** Target maximum RAM. With 120-dps values at ~145 bytes each (v18 §7.3), a 4 GB working set holds ~28 million entries in memory. Memory-mapped I/O means only active pages are in physical RAM; the full 20 TB file is addressable without loading. Target: ≤ 4 GB resident set for typical operation, scaling with active working set.

- **Bootstrap generation time:** ~10⁴ bootstrap values × 8 canonical resolutions = ~8×10⁴ projections. Each projection: one MPFR log₂ + one round + derived property computation ≈ 1 ms per projection at 120 dps. Target: **~80 seconds for bootstrap** (parallelizable across cores).

- **Single-value full escalation:** 6–8 LCM landmarks typical, each requiring one 120-dps projection. Target: **<1 second** for typical values (up to 27720ET). Values requiring deeper escalation (360360ET+): up to 10 seconds acceptable.

- **Bulk ingestion throughput:** File ingestion pipeline: extract seed + project at 120 dps + store + insert-time discovery. Target: **≥1000 values/second** sustained (limited by MPFR computation, not I/O).

- **Cache-hit latency:** Memoized equation lookup: hash computation + hash table probe + value fetch. Target: **<100 μs** (positional access via memory-mapped I/O).

- **Background discovery scan:** Proportional to database size. At 10⁶ entries: target <60 seconds per scan pass. At 10⁹ entries: target <1 hour (overnight batch acceptable).

- **GUI responsiveness:** Target **60 FPS** at all interaction levels. Dear ImGui + OpenGL 4.6 redraws every frame. Lattice navigation must be smooth at all zoom levels. RTX 2070 Super handles millions of instanced points at 60+ FPS. Heavy computation (escalation, discovery) runs on background threads, never blocking the render loop.

### 2.5 Backup and Recovery

**Status: PARTIALLY COMPLETE.** Backup approach and WAL crash recovery specified in EUDD v18 §7.9, §7.1d.

**What remains:**

- **Atomic snapshot while running:** The Manager must support creating a consistent snapshot of `Sempaevum.akashic` without stopping. Approach: pause commits (buffer new writes in WAL) → flush all pending WAL entries to main file → copy the file (or use Windows Volume Shadow Copy) → resume commits. The pause window should be <1 second for files up to 20 TB (the copy is the OS's responsibility via shadow copy; the Manager only needs to ensure the WAL is drained).

- **Backup verification:** After each backup, verify integrity: open the backup file read-only, check header SHA-256, spot-check CRC-32 on a sample of pages (e.g., every 1000th page), verify section directory consistency. Report pass/fail to GUI dashboard.

- **Backup scheduling:** Configurable via Dear ImGui dashboard. Default: every 6 hours to a user-specified path. Manual backup on demand. Backup history retained (last N backups, configurable, default 5).

---

## 3. TIER 3 — Can Iterate But Needed for Professional Grade

### 3.1 GUI Wireframes and Interaction Model

**Status: PARTIALLY COMPLETE.** LOD hierarchy, 3D perspectives, color coding, panel layout, and shader types all specified in EUDD v18 §7.10.

**What remains:**

- Actual wireframe layouts for each Dear ImGui panel (panel dimensions, widget placement, information hierarchy)
- Keyboard/mouse navigation mapping (scroll = zoom = escalate/de-escalate through LOD; click = select address → property inspector; drag = pan; right-click = context menu with perspective switch)
- Transition animations between LOD levels and perspective switches
- Real-time update behavior when background discovery creates new patterns/attractors during navigation

### 3.2 Ingest Adapter Specifications

**Status: PARTIALLY COMPLETE.** File-type overview and pipeline specified in EUDD v18 §7.12.

**What remains per file type:**

- **CSV:** Column selection UI (which column(s) contain values?), R₀ choice (user-specified or auto-detected from column headers/units), delimiter auto-detection, header row handling
- **PDF/Markdown:** Equation/constant extraction method — regex for common patterns (e.g., `\alpha`, `\zeta`, decimal numbers with ≥5 significant digits) + manual selection mode for complex equations. LLM-assisted extraction deferred to future.
- **Raw binary:** Integration with compressor's Δk pipeline — the compressor processes the file, the adapter ingests the resulting Δk patterns and archetype matches
- **Image:** Pixel-value extraction strategy: full resolution (every pixel as r = pixel_value/255), region-averaged (user-selectable grid), or frequency-domain (2D FFT → frequency ratios)
- **Audio:** FFT parameters (window size, overlap), frequency-ratio extraction (peak frequencies as ratios), sample-rate normalization (all frequencies as ratios to R₀ = sample_rate)
- **Sensor streams:** Real-time vs batch mode, R₀ selection from bootstrap reference catalog (v18 §3.17), anomaly threshold for `sensor_anomaly_detected` events

### 3.3 Self-Recording Specifics

**Status: PARTIALLY COMPLETE.** What to record and depth cap (one level) specified in EUDD v18 §3.1b.

**What remains:**

- Exact metric list with recording frequency (e.g., row counts per table: every commit; cache hit/miss ratio: every 1000 queries; discovery engine patterns-per-scan: every scan pass; storage growth rate: hourly)
- Overhead budget enforcement: self-recording should not exceed 1% of Manager CPU. Implementation: sample-based recording (record every Nth event's self-metric, not every event)
- Which self-recorded metrics are most informative for lattice projection (to be discovered empirically after bootstrap — the system will learn which of its own metrics produce interesting lattice addresses)

### 3.4 JSON Extension Full Schema

**Status: PARTIALLY COMPLETE.** Extension mechanism and example schema specified in EUDD v18 §7.14.

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
- Bootstrap catalog reference (what's in the initial database and why — derived from v18 §3.17 + §3.18)
- Discovery engine guide (how patterns are found, what the engine looks for — derived from v18 §3.16)

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

## 6. Design Decisions

**Status: ALL RESOLVED.** See EUDD v18 §7.10, §3.1a, §7.1, §7.1b, §7.1c, §7.1d, §7.11, §3.1b.

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
| ET_Universal_Discovery_Database18.md | Master design specification (3498 lines) — the primary reference for all resolved items |
| ET_Sempaevum_Paper20.tex | Theoretical foundation — lossless bijection, LCM tower, four paths, triple backbone, gauge derivations |
| ET_Three_Tools_Complete_Reference.md | Methodology — Identification Principle, Descriptor Gap Principle, Subsumption Law |
| EUDD_Production_Roadmap_v3.md | Previous gap analysis (this document supersedes it) |
| constants.py / primitives.py | Foundation constants and {P,D,T} class structures |

---

**Three Tools verification of this roadmap (v4):**

- **Identification Principle:** P = the complete production system. D = the remaining gaps. T = the build process. Gaps are in D — missing specifications for API wire protocol, error catalog, testing, performance targets, and implementation-level discovery algorithms.
- **Descriptor Gap Principle:** Remaining gaps across 3 tiers. Each gap IS a Descriptor pointing to its own resolution. The critical path is clear: API spec → error catalog → test cases → implementation.
- **Subsumption Law:** EUDD v18 subsumes all Tier 1 structural content except the API wire protocol and error catalog. This roadmap v4 subsumes v3 without remainder.

> *gap(model) = D_missing*
> *P ∘ D ∘ T = E*

---

**Document Version:** Production Roadmap v4.0
**Source Design:** ET_Universal_Discovery_Database18.md (3498 lines)
**Previous Version:** EUDD_Production_Roadmap_v3.md (superseded)
**Date:** May 2026

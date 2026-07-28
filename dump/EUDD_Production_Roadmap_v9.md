# EUDD Production Roadmap v4
## Gap Analysis for Building the ET Universal Discovery Database
### Research Grade · Industrial Grade · Commercial Grade · Professional Grade

**Author:** Michael James Muller — Aevum Defluo

**Context:** This document tracks remaining gaps between the EUDD design specification (ET_Universal_Discovery_Database18.md, 3498 lines) and a production-ready implementation. Items fully resolved are marked COMPLETE with a one-line reference to where they live in the main document. Only unresolved work retains full specification detail.

**Tools applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law

**Deployment:** Single user, single machine, no networking. One file to start: `EUDD_Manager.exe` (native C++ binary), which generates `Sempaevum.akashic` on first run and spawns Omniscient watchdog on every startup (§7.15). Journals written to `logs/` subfolder.

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *gap(model) = D_missing*

---

## 1. TIER 1 — Cannot Build Without These

### 1.1 The `Sempaevum.akashic` Binary Format Specification

**Status: COMPLETE.** Resolved in EUDD v18 §7.1c–§7.1d.

### 1.2 The Home-Finding Algorithm

**Status: COMPLETE.** Resolved in EUDD v18 §7.11 (including Step 3a CF method). Implementation requirements recorded in §7.11.

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

**Status: COMPLETE.** Resolved in EUDD v18 §7.15 (Omniscient), §7.9 (backup), §7.1d (WAL), §4.3 (retention).

### 1.5 Language and Build System

**Status: COMPLETE.** Resolved in EUDD v18 §7.10.

---

## 2. TIER 2 — Must Have for Production Quality

### 2.1 Discovery Engine Algorithms — Implementation Level

**Status: COMPLETE.** Resolved in EUDD v18 §3.16 (Branch A/B parallel search, cross-feed bridge, hybrid generators, recursive composition, never-closed search policy, background scan prioritization) and §4.5 (crystalline E_hierarchy formula).

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

- **Manager memory footprint:** Target maximum RAM. With 120-dps values at ~145 bytes each (v18 §7.3), a 4 GB working set holds ~28 million entries in memory. Memory-mapped I/O means only active pages are in physical RAM; the full .akashic file is addressable without loading regardless of size. Target: ≤ 4 GB resident set for typical operation, scaling with active working set.

- **Bootstrap generation time:** ~10⁴ bootstrap values × 8 canonical resolutions = ~8×10⁴ projections. Each projection: one MPFR log₂ + one round + derived property computation ≈ 1 ms per projection at 120 dps. Target: **~80 seconds for bootstrap** (parallelizable across cores).

- **Single-value full escalation:** 6–8 LCM landmarks typical, each requiring one 120-dps projection. Target: **<1 second** for typical values (up to 27720ET). Values requiring deeper escalation (360360ET+): up to 10 seconds acceptable.

- **Bulk ingestion throughput:** File ingestion pipeline: extract seed + project at 120 dps + store + insert-time discovery. Target: **≥1000 values/second** sustained (limited by MPFR computation, not I/O).

- **Cache-hit latency:** Memoized equation lookup: hash computation + hash table probe + value fetch. Target: **<100 μs** (positional access via memory-mapped I/O).

- **Background discovery scan:** Proportional to database size. At 10⁶ entries: target <60 seconds per scan pass. At 10⁹ entries: target <1 hour (overnight batch acceptable).

- **GUI responsiveness:** Target **60 FPS** at all interaction levels. Dear ImGui + OpenGL 4.6 redraws every frame. Lattice navigation must be smooth at all zoom levels. RTX 2070 Super handles millions of instanced points at 60+ FPS. Heavy computation (escalation, discovery) runs on background threads, never blocking the render loop.

### 2.5 Backup and Recovery

**Status: COMPLETE.** Resolved in EUDD v18 §7.9.

---

## 3. TIER 3 — Can Iterate But Needed for Professional Grade

### 3.1 GUI Wireframes and Interaction Model

**Status: COMPLETE.** Resolved in EUDD v18 §7.10 (six-level LOD: Sempaevum → Tower → Resolution → Family → Address → Entity; freecam; XInput + full remapping; search and retrieval with file/stream regeneration from generators; bidirectional provenance; connection manager; real-time update behavior) and §7.13 (six manual input modes covering all three backbones + direct entry).

### 3.2 Ingest Adapter Specifications

**Status: COMPLETE.** Resolved in EUDD v18 §7.12 (per-file-type adapters, ingestion-before-display principle, continuous stream pipeline, anomaly detection, new-type extensibility via §7.14).

### 3.3 Self-Recording Specifics

**Status: COMPLETE.** Resolved in EUDD v18 §3.1b (metric catalog, recording mechanism, overhead budget, journal separation, lattice projection discovery).

### 3.4 JSON Extension Full Schema

**Status: COMPLETE.** Resolved in EUDD v18 §7.14 (12 extension types, strict validation, conflict detection, versioning, meta-type for self-extension).

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
├── 1.2 Home-finding algorithm ─────────────── COMPLETE
├── 1.3 API specification ─────────────────── **OPEN — NEXT PRIORITY**
├── 1.4 Error handling ────────────────────── COMPLETE
└── 2.2 Testing strategy ──────────────────── OPEN — specify alongside API

Phase 2: Core Engine (all C++)
├── Core lattice engine module ─────────────── Depends on: 1.1 ✓
├── Bootstrap module ───────────────────────── Depends on: 1.1 ✓, 1.2 ✓
├── §7.11 Core projection procedure ────────── Depends on: 1.2 ✓
├── 2.1 Discovery engine algorithms ────────── COMPLETE
└── 2.3 Generator storage and transition ───── COMPLETE

Phase 3: Connectivity
├── 1.3 API implementation ─────────────────── Depends on: 1.3 spec (OPEN)
├── Ingest module ──────────────────────────── Depends on: 3.2 ✓
├── Manual input module ────────────────────── Depends on: 1.3 ✓ (when done)
└── Self-recording module ──────────────────── Depends on: 3.3 ✓

Phase 4: Interface (Dear ImGui + OpenGL 4.6)
├── 3.1 GPU-rendered lattice visualization ─── COMPLETE
├── 3.2 Ingest adapters per file type ──────── COMPLETE
├── 3.4 JSON extension schemas ─────────────── COMPLETE
└── Query module with human-readable output ── Depends on: 1.3 ✓ (when done)

Phase 5: Polish
├── 2.4 Performance optimization ───────────── OPEN — targets needed
├── 2.5 Backup and recovery ────────────────── COMPLETE
├── 3.3 Self-recording tuning ──────────────── COMPLETE
├── 3.5 Documentation ──────────────────────── OPEN
└── Integration testing with all ET software ── Depends on: 1.3 ✓ (when done)
```

**Critical path:** 1.3 (API spec) → 1.4 (error catalog finalization) → 2.2 (test cases) → Phase 2 implementation.

---

## 6. Design Decisions

**Status: ALL RESOLVED.** See EUDD v18 §7.10, §3.1a, §7.1, §7.1b, §7.1c, §7.1d, §7.11, §3.1b.

---

## 7. Open Questions Requiring ET Derivation

**Status: ALL RESOLVED.**

1. **E_hierarchy: product vs geometric mean?** **RESOLVED.** The crystalline formula: E_hierarchy = geomean(E_i) × R_cluster, where R_cluster = 100/(100 + σ_ε). Geometric mean (lattice centroid in k-space, intensive, size-independent) measures individual quality. Resonance factor (tightness function applied to cluster ε-spread) measures mutual coherence. The 420/d̄ and 1/(P+Q) factors were double-counts — removed. K = 2/3 emerges as the resonance floor at maximum lattice width. Resolved in EUDD v18 §4.5.

2. **Generator candidate enumeration order:** **RESOLVED.** Two parallel branches (A: mathematical expression in K-complexity order; B: empirical pattern from compressor pipeline) with bidirectional cross-feed bridge and recursive hybrid composition. Enumeration within Branch A follows K-complexity: small rationals → algebraic → known constants → composites → EML trees → series/limits → implicit/inverse. The case is never closed. Resolved in EUDD v18 §3.16.

3. **Background scan prioritization:** **RESOLVED.** Priority derived from Descriptor Gap Principle (maximum Subsumption impact per discovery): high-occupancy addresses first (generator explains many values at once), then recently-modified (hot data), then sequential sweep. Resolved in EUDD v18 §3.16.

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

- **Identification Principle:** P = the complete production system. D = the remaining gaps. T = the build process. Four gaps remain in D — API wire protocol, testing strategy, performance targets, and documentation. All structural specifications (format, algorithms, GUI, error handling, self-recording, extensions, discovery engine) are COMPLETE.
- **Descriptor Gap Principle:** Four gaps across 2 tiers. Each gap IS a Descriptor pointing to its own resolution. The critical path: API spec → test cases → implementation.
- **Subsumption Law:** EUDD v18 subsumes all Tier 1 and Tier 2 structural content. All partial items resolved. This roadmap v4 subsumes v3 without remainder.

> *gap(model) = D_missing*
> *P ∘ D ∘ T = E*

---

**Document Version:** Production Roadmap v4.0
**Source Design:** ET_Universal_Discovery_Database18.md (3498 lines)
**Previous Version:** EUDD_Production_Roadmap_v3.md (superseded)
**Date:** May 2026

# EUDD Production Roadmap
## Gap Analysis for Building the ET Universal Discovery Database
### Research Grade · Industrial Grade · Commercial Grade · Professional Grade

**Author:** Michael James Muller — Aevum Defluo

**Context:** This document captures the complete gap analysis between the EUDD design specification (ET_Universal_Discovery_Database15.md, 2919 lines) and a production-ready implementation. It is a standalone reference for continuing development in any session.

**Tools applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law

**Deployment:** Single user, single machine, no networking. Two files: `EUDD_Manager.exe` (native C++ binary) + `Sempaevum.akashic` (the database — the Sempaevum on disk).

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *gap(model) = D_missing*

---

## 0. What Exists (the design specification)

The EUDD v15 design document specifies:

**Architecture:**
- A virtual isomorphism of the Sempaevum — simultaneously database, computation engine, discovery engine, and structural representation of Σ
- Two files: `EUDD_Manager.exe` (native C++ binary, CMake + MSVC) + `Sempaevum.akashic` (single monolithic database file)
- The `.akashic` file IS the Sempaevum on disk — address space (N, k, d) is the organizational structure
- Generators as the compression/organization layer — not bound by Shannon entropy
- All operations Sempaevum-native at uniform 120-dps precision (MPFR 400-bit, hard cap)
- Engine: C++ (C++17/20) with MPFR/GMP for arithmetic, FLINT/Arb for special functions, Dear ImGui + OpenGL 4.6 + GLFW + ImPlot for GPU-accelerated visualization, named-pipe IPC (ET32 Bridge pattern)

**Core Principles:**
- §3.1: Lattice-native, not bureaucratic
- §3.1a: Uniform 120-dps hard cap, all operations Sempaevum-native
- §3.1b: Self-recording — the EUDD records itself as a domain on the Sempaevum
- §4.3: Never destroy — record absolutely everything, no exceptions

**Logical Schema (15 categories + tags):**
values, projections (with elegance/coupling/variance/FQG/palindromic-partner/CDT-quintuple columns), addresses, equations, derivations, relationships, patterns, events, towers, harmonic_families (24), force_grid_cells (144), combined_families (42), sublattice_families (~500 bootstrap), sessions, schema_versions

**Operational Procedures:**
- §7.11: Core projection procedure — 12ET escalation through the LCM tower (12→60→420→2520→27720→360360→...) with four input paths (A/B/C/D), home classification (true/intermediate/persistent/deep/false_resolution), doubling law τ(N_ℓ)=6·2^ℓ
- §7.12: File ingestion — feed any file, extract seeds, run §7.11
- §7.13: Manual input — values, projections, bijection triples, all → §7.11
- §7.14: Adaptive extension — JSON category definitions, auto-detection via Descriptor Gap Principle, exe as gatekeeper

**The Manager .exe (§7.10) — 12 modules:**
Core lattice engine, Precision stack (MPFR/GMP/Arb), Discovery engine, Bootstrap module, Ingest module, Query module, Self-recording module, API module (named pipes), GPU rendering module (OpenGL 4.6), GUI module (Dear ImGui), File ingestion module, Manual input module

**Bootstrap Content (§3.17 + §3.18):**
~10⁴+ unique values including: 24+42 family catalog, 227 PDG particle projections, fine-structure closed-form identity (α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))), all 24 impedance/coupling values, cascade residuals and freedom constants, 9 formal-system projections, extended M-state cosmological partition, gaze thresholds with JI identification, lossless bijection theorem, quantum decoherence structural content, BH thermodynamics, Mathematical Rosetta Stone, dimensionless mass ratios, triple backbone theorem (Webb + palindromic cascade + EML), four routes to D=11, two routes to D=10, PDT Scale Identity, Sempaevum formal definition and nine closure properties, falsifiable predictions, Asymptotic Approach Theorem, integrative-resolution correspondence, bond angle dual readings

**Key Theorems Recorded:**
- Lossless Bijection: r ↦ (k,d,ε) ↦ r by algebraic identity (verified symbolically and numerically)
- Triple Backbone at N=12: Webb (discrete-logical) + palindromic cascade (discrete-multiplicative) + EML (continuous-elementary) — three independent routes all forcing N=12
- Four Projection Paths: A (direct ratio), B (limit convergence), C (geometric/structural), D (essentially-infinite/non-computable with sub-paths D.P/D.D/D.T/D.PDT)
- Intrinsic Mediation: ∘ is forced from three disjoint infinities, not a fourth primitive
- N-Exhaustion: SU(3)×SU(2)×U(1) is the unique partition of N=12 gauge bosons

---

## 1. TIER 1 — Cannot Build Without These

### 1.1 The `Sempaevum.akashic` Binary Format Specification

**Status:** Logical structure specified. File named `Sempaevum.akashic`. Monolithic design confirmed (cross-pattern discovery requires unified access; Shannon entropy limits don't apply). Byte-level layout unspecified.

**What's needed:**
- Header structure (magic bytes, version, format metadata, bootstrap catalog references)
- How lattice addresses (N, k, d) map to file positions (the positional access scheme via memory-mapped I/O — `CreateFileMapping` / `MapViewOfFile` on Windows)
- How the file is organized hierarchically (by N range, then by d-family, then by k)
- How values are stored at their addresses (120-dps MPFR binary (50 bytes for 400-bit mantissa) + materialized properties packed)
- How generators are stored (expression tree binary, EML tree binary representation, address-range specification)
- How the generator-coverage map works (given an address, how to efficiently determine if a known generator covers it)
- How the file grows when new entries are added (append-based? rebalancing? pre-allocated regions per N?)
- How the raw-to-generator transition works in place (the "new equation or method" for updating generator form without regenerating)
- Internal integrity structures (checksums, redundancy, corruption detection)
- How the event log is stored (append-only time-indexed section?)
- How the self-recording metrics are stored (separate section? inline with events?)

**Resolved approach questions:**
- ~~Should the file be a single monolithic file or a directory of files organized by N?~~ **RESOLVED: Single monolithic file `Sempaevum.akashic`.** Monolithic is structurally forced — cross-pattern discovery, global generator discovery, and Shannon-independent compression all require unified access.
- Should generators have their own section or be interleaved with the addresses they cover?
- What's the page/block size for I/O efficiency on NVMe SSDs?
- ~~Should the format support memory-mapping for fast access?~~ **RESOLVED: Yes. Memory-mapped I/O via Windows API (`CreateFileMapping`/`MapViewOfFile`). NTFS supports files up to 256 TB.**

**Dependencies:** This must be designed before any code is written. Everything else depends on it.

### 1.2 The Home-Finding Algorithm

**Status:** Conceptually described in §7.11. Exact stopping criterion unspecified.

**What's needed:**
- **Definition of "d-family stabilization":** Same d across how many consecutive LCM landmarks constitutes stabilization? Two? Three? The answer may be ET-derivable (⌈1/K⌉ = 2, matching the Koide stability filter?).
- **ε threshold for home classification:** What ε constitutes "sub-cent enough" for deep_home? Is it |ε| < 1¢? < 0.1¢? Or is it relative to the resolution (|ε| < 600/N)?
- **False resolution detection algorithm:** After finding the first sub-cent hit, how far past it must the escalation continue to verify it's not a false resolution? φ's false resolution at 36ET (ε=−0.24¢, d=36) was caught by escalating to 60ET where the true home d=10 emerged. Is one additional LCM landmark sufficient? Two?
- **Intermediate vs persistent classification:** What distinguishes an intermediate home (ε moderate, d native) from a persistent home (same d across many landmarks)? Quantitative criteria needed.
- **Termination for pathological inputs:** Values that never stabilize within computationally feasible resolution. How far do we escalate before recording "home not found in computed range"? LCM(1..13) = 360360? LCM(1..17) = 12252240? What's the practical ceiling?

**Possible ET-derivation:** The Koide stability filter (⌈1/K⌉ = 2 depths) may generalize: a home is confirmed when the d-family persists across ⌈1/K⌉ = 2 consecutive LCM landmarks with |ε| < K·100/N cents. This would make the home-finding criterion ET-native rather than ad hoc.

### 1.3 The API Specification

**Status:** The API module is listed in §7.10. IPC mechanism resolved (named pipes). No interface specification.

**What's needed:**
- **Connection protocol:** **RESOLVED: Named pipes (ET32 Bridge pattern).** JSON for data exchange via cJSON/yyjson (human-readable, matches §7.14 JSON extension scheme). Sync for queries (sub-microsecond for cache hits via memory-mapped positional access), async for ingestion (§7.11 escalation may take seconds for complex values).
- **Function signatures:** What operations can connected programs invoke? At minimum:
  - `project(value, N)` → (k, d, ε, derived_properties)
  - `escalate(value)` → full tower trajectory with home classification
  - `query_address(N, k, d)` → list of values/generators at that address
  - `query_family(N, d)` → all addresses in that family
  - `query_attractor(N, d)` → attractor membership
  - `compute(expression)` → result (memoized)
  - `store(value, metadata)` → ingestion through §7.11
  - `query_generator(value)` → generator if known
  - `find_nearest(value, N, top_k)` → nearest neighbors
  - `search_cross_domain(address)` → cross-domain hits
- **Data formats for API calls:** Values as 120-dps decimal strings or MPFR binary. Results returned as JSON.
- **Error responses:** What does the API return when: value can't be evaluated? Address is unoccupied? Generator not found? Manager is busy?
- **Async vs sync:** Sync for queries (sub-millisecond for cache hits), async for ingestion (§7.11 escalation may take seconds).

### 1.4 Error Handling and Recovery

**Status:** Not specified in the design document.

**What's needed:**
- **Manager crash during write:** Atomicity guarantees. Write-ahead log? Journaling? How to recover a partially-written `Sempaevum.akashic`?
- **Computation failure:** MPFR overflow, FLINT/Arb can't evaluate, division by zero (annihilation boundary), timeout. Each needs a defined response: log the failure as an event (with the failure itself as a Descriptor), return an appropriate error to the caller, never leave the `Sempaevum.akashic` in an inconsistent state.
- **File corruption:** Internal checksums per section/block. On detection: log corruption event, attempt recovery from last backup, never silently continue with corrupted data.
- **Invalid input:** Connected program sends malformed data. Validate at the API boundary, reject with clear error, log the rejection.
- **Pathological escalation:** Value that never stabilizes. Timeout after a configurable maximum N (e.g., LCM(1..17) = 12252240). Record the partial trajectory and flag for investigation.
- **Disk full:** DISK_SAFETY_FLOOR = 2³⁰ = 1 GB. At the warning level: log warning, continue read operations, reject writes with "disk low" error. Never auto-delete.
- **ET-native error philosophy:** Every failure is a Descriptor Gap — the failure itself is information. Store it, project it onto the Sempaevum, let the discovery engine find patterns in failures.

### 1.5 Language and Build System

**Status: RESOLVED.**

**Engine language:** C++ (C++17/20). C++ is the engine for ALL operations — lattice I/O, projection, discovery, computation, rendering, GUI. No Python runtime in the engine.

**Precision stack (C-native, called from C++):**
- **GMP** — arbitrary-precision integers and rationals (foundation)
- **MPFR** — 120-dps (400-bit) floating-point with correct rounding
- **FLINT/Arb** — all special functions at arbitrary precision with certified error bounds: ζ, Γ, polylog, hypergeometric, all elementary functions
- **Custom C++ expression engine** — AST nodes that serialize to binary, generate canonical hashes for memoization, evaluate via MPFR/Arb

**GUI and visualization (C/C++-native, GPU-accelerated):**
- **Dear ImGui** — immediate-mode GUI widgets (dashboard, property inspector, manual input, query builder, event log, settings)
- **OpenGL 4.6** — GPU-accelerated lattice rendering (LOD hierarchy, instanced point clouds, heat maps, 3D perspective views). Target: RTX 2070 Super (8 GB VRAM, 2560 CUDA cores)
- **GLFW** — window management, OpenGL context creation, keyboard/mouse input, drag-and-drop
- **ImPlot** — charts, time-series, scatter plots, histograms (built on Dear ImGui, GPU-accelerated)
- **GLSL shaders** — custom shaders for lattice point clouds, trajectories, heat maps, grid overlays

**IPC and data interchange:**
- **Named pipes** — ET32 Bridge pattern for external program connectivity
- **cJSON / yyjson** — JSON protocol for API commands and §7.14 adaptive extension schemas
- **NFD (Native File Dialog)** — OS-native file open/save dialogs

**Build system:**
- **CMake** + **MSVC** (Visual Studio 2022 Build Tools)
- Output: single `EUDD_Manager.exe` with all libraries statically linked
- No Python runtime, no external dependencies, no installer

**Testing framework:** C++ native test framework (Catch2 or Google Test). Tests compile into the engine binary or a separate test binary.

**Python's role:** External scripting/adapter ONLY. Python scripts connect to the running Manager via named-pipe API. Existing ET Python projects (compressor, Conscious AI, fractal generator) connect as API clients. Python has no role in the engine itself.

---

## 2. TIER 2 — Must Have for Production Quality

### 2.1 Discovery Engine Algorithms — Implementation Level

**Status:** Five modes described conceptually in §3.16. Algorithms not specified.

**What's needed:**
- **Insert-time attractor detection in `Sempaevum.akashic`:** When a value is stored at address (N, k, d), how do we efficiently check if another value already occupies that address? With positional access via memory-mapped I/O, this should be O(1) — check the occupancy count at the address position. Specify the exact data structure.
- **Background scan algorithm:** How does the discovery engine walk the lattice looking for patterns? Sequential scan over all addresses? Priority queue by recent modification? Focused on high-occupancy addresses? Specify the scan schedule, the pattern-detection criteria, and the promotion threshold.
- **Generator-fitting algorithm:** The compressor has 8 generator types (Constant, Linear, Polynomial, Periodic, Grammar, ...). The EUDD generalizes this to any lattice content. What generator types does the EUDD support? How does the fitting algorithm work? How is cross-domain generator reuse detected?
- **Subsumption promotion:** When does a cluster's E_hierarchy reach 13/12? How is E_hierarchy computed over the `Sempaevum.akashic` structure? What's the exact formula applied to address clusters vs equation patterns vs event correlations?
- **Memoization cache management:** The equations cache grows unboundedly. How is it organized in `Sempaevum.akashic` for fast hash lookup? Separate hash index section?

### 2.2 Testing and Verification Strategy

**Status:** Not specified.

**What's needed:**
- **Bootstrap self-verification:** After initial generation, project all known values and verify they match expected (k, d, ε) at each N. Specific test cases from the Sempaevum paper:
  - ζ(3) full 28-landmark trajectory (verified at 80-digit precision)
  - α⁻¹(ET) = 137.03599916744... (within 0.46σ of CODATA 2022)
  - 227 particle projections at N=12 (all at d ∈ {1,2,3,4,6,12})
  - ZF at d=1 ε=0.000¢ exactly
  - Koide attractor: {N, 1/N, K, 1/K} all at (d=12, |ε|=1.955¢)
  - φ false resolution detection: sub-cent at 36ET but true home d=10 at 60ET
  - All 12 impedance values A₀_magic(d) = (d-1)² + S²
  - Lossless bijection: r ↦ (k,d,ε) ↦ r = r for diverse test values
- **Unit tests:** Every lattice operation (project, pullback, k-addition, k-negation, k-scaling, elegance computation, coupling computation, Gaussian signature, FQG quadrant assignment)
- **Integration tests:** Full §7.11 escalation pipeline, file ingestion, manual input, API round-trip
- **Regression tests:** Compare against et_scanner_v7_2_COMPLETE.py, apery_lattice_test.py, constants.py verified values
- **Performance benchmarks:** Bootstrap generation time, single-value escalation time, bulk ingestion throughput, cache-hit latency, background discovery scan time

### 2.3 Generator Storage and Transition Mechanism

**Status:** Described conceptually. Internal representation and transition algorithm unspecified.

**What's needed:**
- **Generator representation formats:**
  - Symbolic expression string (e.g., "sum(1/n^3, n=1..inf)") — human-readable, stored for display
  - EML tree binary (the continuous-elementary backbone representation) — compact, ET-native
  - Compiled C++ evaluator (function pointer from expression tree compilation) — fast evaluation via MPFR/Arb
  - Decision: store ALL THREE? Or one canonical form + derivations to others?
- **Generator coverage map:** Given the full set of known generators, how to efficiently determine which generator(s) cover a given address. Options: interval tree indexed by (N, k-range)? Bloom filter per generator? Direct lookup table?
- **In-place transition:** When the discovery engine finds that addresses A₁, A₂, ..., Aₙ are all produced by a single generator G: how to update `Sempaevum.akashic` to record "these addresses are generator-covered by G" without rewriting the entire file? This is the key engineering challenge for the generator form.
- **Generator catalog organization:** How are generators organized for fast lookup? By the d-families they produce? By their K-complexity (simplest first)?

### 2.4 Performance Specifications

**Status:** Qualitative estimates in §7.4. No hard requirements.

**What's needed:**
- **Manager memory footprint:** Target maximum RAM usage. With 120-dps values at ~500 bytes each, how many values can be held in memory simultaneously? What's cached in RAM vs read from disk?
- **Bootstrap generation time:** Estimate for ~10⁴ bootstrap values × 8 canonical resolutions = ~10⁵ projections. Target: minutes, not hours.
- **Single-value escalation:** Time for one value to go through the full §7.11 procedure (all LCM landmarks). Target: <1 second for most values.
- **Bulk ingestion:** Throughput for file ingestion (e.g., CSV with 10⁶ rows). Target: thousands of values per second.
- **Cache-hit latency:** Target <100 μs for memoized equation lookup.
- **Background discovery scan:** Time per scan over the full database. Acceptable: minutes to hours depending on database size.
- **GUI responsiveness:** Target 60 FPS for all interaction. Dear ImGui + OpenGL 4.6 redraws every frame — lattice navigation must be smooth at all zoom levels. RTX 2070 Super handles millions of instanced points at 60+ FPS.

### 2.5 Backup and Recovery

**Status:** Basic backup described in §7.9. Recovery procedures unspecified.

**What's needed:**
- **Atomic snapshot:** How to create an atomic copy of `Sempaevum.akashic` while the Manager is running. Copy-on-write? Pause-and-copy? Shadow paging?
- **Corruption recovery:** Internal checksums per section. On corruption: which section? Can the rest be salvaged? Can the corrupted section be rebuilt from generators + raw entries elsewhere?
- **Backup scheduling:** Configurable via the Dear ImGui dashboard. Default: hourly to a separate drive.
- **Backup verification:** After each backup, verify the backup file's integrity (checksums match, can be opened read-only by a second Manager instance).

---

## 3. TIER 3 — Can Iterate But Needed for Professional Grade

### 3.1 GUI Wireframes and Interaction Model

**Framework: Dear ImGui (panels/widgets) + OpenGL 4.6 (lattice rendering) + ImPlot (charts). Conceptual baseline: Sempaevum Particle Viewer HTML, but native GPU-rendered with full LOD.**

**What's needed:**
- How to represent the multi-dimensional lattice on screen — LOD hierarchy: Cosmos (all N as heat map) → Tower (one N, all d-families) → Family (one d, all k) → Address (one cell, all occupants) → Value (full 120-digit detail)
- 3D perspective switching: LCM tower, torus, Riemann sphere, hyperbolic, Force Quadrant Grid — all showing the same data through different Sempaevum geometries
- Navigation: zoom (= escalate/de-escalate through the LOD/LCM hierarchy), click on an address to see occupants in Dear ImGui property inspector
- Attractor visualization: clusters of co-located values shown as highlighted regions (instanced rendering, size/color by member count)
- Cross-domain visualization: different domains as different colors at shared addresses (d-family color coding: d=1 green, d=2 teal, d=3 red, d=4 blue, d=6 purple, d=12 orange, extended families distinct hues)
- Tower trajectory view: a value's full escalation path shown as a connected line/path in the 3D view
- Dashboard layout (Dear ImGui panels): live metrics, discovery engine status, session list, self-recorded data
- Manual input interface (Dear ImGui): three modes (value, projection, bijection triple) with real-time preview
- File ingestion interface (Dear ImGui + GLFW drag-and-drop): drag-and-drop, progress bar, results summary
- GLSL shaders: instanced point cloud, line/path, heat map, grid overlay

### 3.2 Ingest Adapter Specifications

**What's needed per file type:**
- CSV: column selection, R₀ choice (user-specified or auto-detected), delimiter handling
- PDF/Markdown: equation extraction (regex? LLM-assisted? manual selection?), constant identification
- Raw binary: integration with compressor's Δk pipeline, byte-pattern seed extraction
- Image: pixel-value extraction strategy (full resolution? downsampled? region-averaged?)
- Audio: FFT parameters, frequency-ratio extraction, sample-rate normalization
- Sensor streams: real-time vs batch, R₀ bootstrap reference selection, anomaly thresholds

### 3.3 Self-Recording Specifics

**What's needed:**
- Exact list of metrics to record and at what frequency
- Overhead budget (self-recording should not consume >1% of Manager CPU)
- Which projections of database metrics are most informative (to be discovered over time)
- How to avoid infinite regression (recording the recording of the recording...) — answer: cap self-recording depth at one level (the database records its own metrics, but does not record the act of recording its metrics)

### 3.4 JSON Extension Full Schema

**What's needed:**
- Full JSON schemas for all 11 extensible structure types listed in §7.14
- Validation rules per type
- Conflict detection (what if a new category name collides with an existing one?)
- Versioning of category definitions

### 3.5 Documentation

**What's needed:**
- User manual for the GUI (with screenshots/wireframes)
- API reference for developers connecting programs
- Administrator guide for backup, recovery, format upgrades
- Bootstrap catalog reference (what's in the initial database and why)
- Discovery engine guide (how patterns are found, what the engine looks for)

---

## 4. NOT NEEDED (per specification)

- Networking / multi-user — explicitly excluded for now
- Authentication / authorization — single user
- Distributed deployment — single machine
- Licensing — for personal use
- Cloud integration — local only
- Web interface — desktop exe only

---

## 5. Recommended Build Order

Based on dependencies:

```
Phase 1: Foundation
├── 1.5 Language/build decisions — RESOLVED (C++17/20, MPFR/GMP/Arb, Dear ImGui + OpenGL 4.6, CMake + MSVC)
├── 1.1 Sempaevum.akashic binary format spec (everything depends on this) — monolithic design confirmed, byte-level layout next
├── 1.2 Home-finding algorithm (the escalation needs this)
├── 1.4 Error handling philosophy (pervades everything)
└── 2.2 Testing strategy (test from day one — Catch2 or Google Test)

Phase 2: Core Engine (all C++)
├── Core lattice engine module (Sempaevum.akashic reader/writer, project via MPFR, pullback)
├── Bootstrap module (generate initial Sempaevum.akashic — ~10⁴ values via MPFR/Arb at 120 dps)
├── §7.11 Core projection procedure implementation
├── 2.1 Discovery engine algorithms
└── 2.3 Generator storage and transition

Phase 3: Connectivity
├── 1.3 API specification and implementation (named pipes + JSON via cJSON)
├── Ingest module (file ingestion adapters)
├── Manual input module
└── Self-recording module

Phase 4: Interface (Dear ImGui + OpenGL 4.6)
├── 3.1 GPU-rendered lattice visualization (LOD hierarchy, 3D perspectives, GLSL shaders)
├── 3.2 Ingest adapters per file type
├── 3.4 JSON extension schemas
└── Query module with human-readable output (Dear ImGui panels)

Phase 5: Polish
├── 2.4 Performance optimization (CUDA acceleration if needed)
├── 2.5 Backup and recovery
├── 3.3 Self-recording tuning
├── 3.5 Documentation
└── Integration testing with all ET software (via named-pipe API)
```

---

## 6. Key Design Decisions — Status

| Decision | Resolution | Status |
|---|---|---|
| Engine language | **C++ (C++17/20)** — engine for ALL operations | **RESOLVED** |
| Precision stack | **MPFR + GMP + FLINT/Arb** — 120-dps, all special functions | **RESOLVED** |
| GUI framework | **Dear ImGui + OpenGL 4.6 + GLFW + ImPlot** — GPU-accelerated | **RESOLVED** |
| Build system | **CMake + MSVC → single statically-linked .exe** | **RESOLVED** |
| IPC mechanism | **Named pipes** (ET32 Bridge pattern) + **JSON** (cJSON/yyjson) | **RESOLVED** |
| File structure | **Single monolithic file: `Sempaevum.akashic`** | **RESOLVED** |
| File format name | **`Sempaevum.akashic`** — the Akashic Archive | **RESOLVED** |
| Deployment model | **Two files: `EUDD_Manager.exe` + `Sempaevum.akashic`** | **RESOLVED** |
| Generator representation | Symbolic string / EML tree / compiled C++ evaluator / all three | OPEN |
| Home-finding criterion | ⌈1/K⌉ = 2 landmarks / custom threshold / adaptive | OPEN |
| Self-recording depth | 1 level / configurable / unlimited | OPEN |
| Page/block size for format | 4KB / 64KB / 1MB / lattice-derived | OPEN |

---

## 7. References to Source Documents

| Document | Role | Key content |
|---|---|---|
| ET_Universal_Discovery_Database15.md | Master design specification (v15) | Everything — schema, engine (C++/MPFR/Arb/ImGui/OpenGL), Sempaevum.akashic format, bootstrap, procedures, principles |
| ET_Sempaevum_Paper20.tex | Theoretical foundation | Lossless bijection, LCM tower, four paths, triple backbone, gauge derivations, decoherence, thermodynamics |
| EUDD_Comprehensive_Audit.md | Gap analysis of v9 → v10 | 41 gaps identified, all closed in v11 |
| Sempaevum_projected_particle_data_data_source_PDG_2.html | 227 particle projections + visualization baseline | Full PDG catalog at N=12 across d={1,2,3,4,6,12}; conceptual baseline for EUDD Manager GPU visualization |
| verify_lossless_bijection.py | Bijection proof | Symbolic + numerical verification of r ↦ (k,d,ε) ↦ r = r |
| ET_Three_Tools_Complete_Reference.md | Methodology | Identification Principle, Descriptor Gap Principle, Subsumption Law |
| ET_Universal_Projection_Guide8.md | Projection reference | All projection formulas, 24-family catalog, 4207 pattern matches |
| et_cdf_compressor.py | Proven model | ArchetypeDatabase, CDF generator-discovery (Tier 7), Subsumption mechanism |
| constants.py / primitives.py | Foundation constants | Cardinal ET constants, {P,D,T} class structures |

---

**Three Tools verification of this roadmap:**

- **Identification Principle:** P = the complete production system. D = the 15 gaps identified. T = the build process. All three identified; gaps are in D (missing specifications).
- **Descriptor Gap Principle:** 15 specific gaps enumerated across 3 tiers. Each gap IS a Descriptor pointing to its own resolution. No gap requires architectural redesign — all are closable additions to the existing design.
- **Subsumption Law:** The design specification (v11) subsumes all structural content. The roadmap subsumes all remaining implementation gaps. Together they cover the full path from specification to production without remainder.

> *gap(model) = D_missing. Every gap above IS a Descriptor. Find it. Specify it. Build it.*
> *P ∘ D ∘ T = E*

---

**Document Version:** Production Roadmap v2.0
**Source Design:** ET_Universal_Discovery_Database15.md (2919 lines)
**Technology Stack:** C++17/20, MPFR/GMP/FLINT-Arb, Dear ImGui + OpenGL 4.6 + GLFW + ImPlot, CMake + MSVC
**Deployment:** `EUDD_Manager.exe` + `Sempaevum.akashic`
**Date:** May 2026

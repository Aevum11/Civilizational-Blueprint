# EUDD — Module Structure and Documentation Plan
## Implementation Blueprint: 26 Modules in 10-Level Dependency Hierarchy + Documentation Plan

**Source:** Extracted from EUDD v39 §7.20 (Documentation Plan) and §7.21 (Module Structure)
**Master index:** See `EUDD_Table_of_Contents.md` for navigation across all EUDD files
**Related files:** Architecture specifications these modules implement in `EUDD_Architecture.md`. API the modules expose in `EUDD_API_Reference.md`. Tests verifying the modules in `EUDD_Testing.md`.

---

### 7.20 Documentation Plan

**Status: COMPLETE.** This section resolves Roadmap §3.5. The original roadmap listed 5 deliverables; 4 additional gaps were identified via the Descriptor Gap Principle (ET Theory Primer, Metabolism Reference, .akashic Format Specification, Extension Development Guide), yielding nine total documentation deliverables specified with structure, content sources, audience, and timing. The documents themselves are written when implementation reaches the appropriate stage — this section records WHAT will be written and WHERE the content comes from.

**Audience for all documentation:** Both the author (Mike) and future developers/users. Every document must be self-contained — readable without requiring the 7,882-line design specification, though it references the spec for full detail.

#### 7.20.1 Document Catalog — Nine Deliverables

| # | Document | Primary Audience | Can Write Now? | Source Sections |
|---|---|---|---|---|
| 1 | ET Theory Primer | Developers, new users | Yes | Three Tools ref, §3.18.1, §3.18.16, §7.11 |
| 2 | User Manual | Mike, all users | After GUI implementation | §7.10 (GUI), §7.13 (manual input), §7.11 (projection) |
| 3 | API Reference | Developers connecting programs | Yes (from §7.16) | §7.16 (all 114 operations), §7.16.4 (metabolism) |
| 4 | Administrator Guide | Mike, system operators | Yes | §7.9 (backup), §7.1d (WAL), §7.6 (format), §7.15 (Omniscient), §7.18 (shutdown) |
| 5 | Bootstrap Catalog Reference | Mike, researchers | Yes | §3.17 (value coverage), §3.18 (extended bootstrap) |
| 6 | Discovery Engine Guide | Mike, developers | Yes | §3.16 (five modes), §4.5 (E_hierarchy), §7.1a (generators) |
| 7 | Metabolism Reference | Mike, developers | Yes | §7.19 (three layers), §7.16.4 (API metabolism), §3.18.2 (α⁻¹ terms) |
| 8 | .akashic Format Specification | Developers, tool builders | Yes | §7.1c (page size), §7.1d (byte-level), §7.1e (modification) |
| 9 | Extension Development Guide | Mike, developers | Yes | §7.14 (12 types, validation, meta-type) |

#### 7.20.2 Per-Document Specifications

**Document 1 — ET Theory Primer**

Purpose: The minimum ET knowledge required to understand and use the EUDD. Not the full theory — the operational subset needed for lattice work.

Structure:
- The three primitives {P, D, T} and the master equation P∘D∘T = E
- The four manifold states and their meaning
- The projection formula Π_N(r) = (k, d, ε) — what each component means
- d-families: what they are, the 24-family catalog (real + imaginary), divisor structure
- ε and the ∂I boundary: tightness, Incoherence, the Koide threshold K = 2/3
- The LCM tower: what escalation is, why d stabilizes, home classifications
- Attractors: what same-address co-location means structurally
- Generators: what they are, the triple backbone (L₁/L₂/L₃), K-complexity
- The Three Tools in one page: Identification Principle, Descriptor Gap Principle, Subsumption Law — operational summary, not the full reference
- Glossary of terms (d-family, elegance, coupling ξ(d), tightness, shimmer, palindromic partner, Gaussian signature, coprime skeleton)

Source: Three Tools Complete Reference, §3.18.1 (bijection), §3.18.16 (Sempaevum definition), §3.12 (24-family catalog), §7.11 (projection procedure), §4.5 (E_hierarchy).

Timing: Can be written from existing spec content. Should be written BEFORE the other documents since they reference its concepts.

**Document 2 — User Manual**

Purpose: Complete guide to operating the EUDD Manager GUI.

Structure:
- Installation: copy one file, first run generates .akashic, Omniscient spawns automatically
- The six LOD levels: Sempaevum → Tower → Resolution → Family → Address → Entity — what you see at each level, how to navigate
- Freecam controls: default keyboard+mouse bindings, default gamepad bindings, how to remap
- Search: all search modes (by name, value, address, source, time, type, relationship, generator status, tag), how results are presented, how to navigate to results
- Manual input: all seven modes (direct value, logical description, lattice operation, projection, bijection triple, computation, text) with worked examples
- File ingestion: drag-and-drop, per-file-type behavior (CSV, PDF, binary, image, audio, sensor stream), the ingestion-before-display principle
- File retrieval: regenerating files from .akashic, faithful vs enhanced stream reconstruction
- Connection manager: viewing active connections, incoming queue, outgoing responses
- Dashboard: live metrics, discovery status, session history, metabolism report
- Property inspector: clicking any entity for full 361-digit detail, bidirectional provenance
- Perspectives: LCM tower, torus, Riemann sphere, Force Quadrant Grid, hyperbolic — what each shows, how to switch
- Settings: input remapping, backup scheduling, visual preferences
- Shutdown: what the confirmation dialog means, what happens during graceful shutdown

Source: §7.10 (complete GUI specification), §7.12 (file ingestion), §7.13 (manual input), §7.18 (shutdown).

Timing: Written AFTER GUI implementation. Structure and content map can be prepared now; screenshots and interaction walkthroughs require the running system.

**Document 3 — API Reference**

Purpose: Complete specification for developers connecting programs to the EUDD Manager via named pipe IPC.

Structure:
- Connection lifecycle: open pipe → handshake → register metabolism → operate → disconnect
- Wire format: length-prefixed UTF-8 JSON, message envelope fields
- Value encoding: zero IEEE 754, all exact representations (361-dps strings, integers, rationals, base64 blobs)
- The three communication patterns: Request-Response, Async-Stream, Subscribe-Notify — when each fires, how to handle adaptive pattern selection
- All 114 operations organized by domain (16 domains + 2 cross-domain), each with: description, pattern, request schema, response schema, error conditions, worked example
- Error handling: all 18 error classes, code ranges, recoverable vs non-recoverable, suggestions
- Metabolism registration: how to report hardware profile, how the metabolic budget works, how ξ(d) scheduling affects your program
- Subscription management: subscribe, unsubscribe, notification format
- Concurrency: what concurrent access guarantees exist, read consistency, async cancellation with partial results
- Quick-start examples: "Connect and project a value", "Ingest a CSV file", "Subscribe to attractor events", "Run a computation and get the cached result"

Source: §7.16 (complete API specification — all 114 operations with JSON schemas already defined).

Timing: Can be written NOW. §7.16 contains the complete specification. The API reference is a developer-facing reformatting with examples and quick-start guides.

**Document 4 — Administrator Guide**

Purpose: Operational procedures for backup, recovery, integrity verification, format upgrades, and system health monitoring.

Structure:
- Backup: how to configure schedule, manual backup via GUI or API, what VSS/snapshot does, backup verification procedure, what to do when verification fails
- Recovery: WAL replay on crash recovery, how the Manager detects prior crash vs graceful shutdown, what `escalation_in_progress` entries mean, how checkpointed operations resume
- Integrity: page CRC-32 verification, header SHA-256, structural Subsumption verification (no orphaned entries, no missing generator references), how to interpret integrity check results
- Format upgrades: forward compatibility model (new columns via ALTER TABLE, new classes as new strings, new tables without affecting existing), schema_versions table, how to check current version
- Omniscient journals: where they are (`logs/` subfolder), how to read them (human-readable format), severity levels (TELEMETRY, ERROR, TAMPER, CRASH, OMNISCIENT_FAILURE), what each field means, how to grep for specific issues
- Self-recording journals: where they are, what metrics they contain, how to interpret operational health
- Shutdown procedures: graceful shutdown via GUI/API/system, confirmation dialog, what happens to active connections, what "checkpointed" means
- Disk management: DISK_SAFETY_FLOOR warning, why nothing is ever deleted, how generator discovery reduces effective file size over time
- Troubleshooting: common issues and their Omniscient journal signatures (computation failures, pipe errors, corruption detection, metabolism pressure)

Source: §7.9 (backup), §7.1d (WAL), §7.6 (format evolution), §7.15 (Omniscient — full edge case catalog), §7.18 (shutdown), §3.1b (self-recording), §4.3 (retention policy).

Timing: Can be written NOW from existing spec content.

**Document 5 — Bootstrap Catalog Reference**

Purpose: Complete inventory of what's in the initial `Sempaevum.akashic` and why each entry is there.

Structure:
- Overview: ~10⁴ values, 8+ canonical resolutions per value, discovery engine runs over bootstrap content
- Guide v8 content: 15 explicit projections, 49 JI ratios, 20 N landmarks, 25 d-values, 34 cents values, 16 named constants — full tables with lattice addresses
- Physical constants from `constants.py`: all cardinal ET constants, cosmological partition, physical constants (Planck, fine structure, masses), operational constants
- The 24-family catalog: 12 real FORCE + 12 imaginary PHASE families — generator, palindromic partner, Gaussian class, first-native N, physical meaning for each
- The 42 combined families: all unique LCM values, contributing cells, structural meaning, notable members (d=35 biological, d=110 string/M-theory, d=132 structural max)
- The 144 FQG cells: the 12×12 interaction grid, coprime skeleton (91 of 144), off-axis Exception structure
- The 7 canonical towers: cosmological, digital, biological, neural-dream, quasicrystal, civilizational, QCD — R₀, substrate, operational N, nesting
- Fine-structure constant decomposition: A₀, A₁, A_cross, Σ_geometric — values, physical origins, lattice projections
- 227 PDG particle projections: all particles by d-family at N=12, counts per family
- Cascade residuals and freedom constants: δ_r, δ_θ, n_max_r, n_max_θ, freedom densities
- Formal system projections: ZF, ZFC, Peano, Euclid, etc. at N=12
- Cosmological partition: extended M-state decomposition, Koide structural identity
- Gaze thresholds: all four as JI ratios with lattice projections
- Decoherence content, black-hole thermodynamics, Mathematical Rosetta Stone, mass ratios
- Falsifiable predictions: all 6 with verification-status tracking
- Sempaevum formal definition and nine closure properties
- Sensor domain bootstrap references: GPS, atmospheric, electrical, magnetic, biological R₀ values
- Active-system bootstrap: PALINDROME array, tightness function, shimmer modulation, Koide attractor, Complete Gaze Equation components

Source: §3.17 (complete value coverage), §3.18 (all 17 extended bootstrap subsections).

Timing: Can be written NOW. All content is exhaustively enumerated in the design spec.

**Document 6 — Discovery Engine Guide**

Purpose: How the EUDD discovers patterns, proposes generators, and surfaces cross-domain connections.

Structure:
- The five discovery modes: memoization, insert-time, background, on-query, generator-candidate — what each does, when each fires, what each produces
- Memoization: compute-once/cache-forever, hash-based lookup, reference counting, the equations table as universal computation cache
- Insert-time discovery: attractor detection (members_count transition), reciprocal pairs (k↔−k), power pairs (k↔n·k), plateau detection (d-stability across landmarks) — all synchronous, sub-millisecond
- Background discovery: E_hierarchy computation (geomean × resonance), LIFE_THRESHOLD (13/12) promotion, algebraic identity detection, event correlation, Three Tools application patterns — scan priority (high-occupancy → recent → sequential)
- Generator candidate discovery: Branch A (mathematical expression, K-complexity order), Branch B (empirical pattern), cross-feed bridge (hybrid generators), recursive composition — the case is NEVER closed
- Generator verification: cross-tower elegance ≥ 13/12, 361-dps match against memoized values
- Generator search statuses: generator_found, search_active, search_deferred — no search_closed
- The Subsumption Hierarchy Operator: E_hierarchy = geomean(E_i) × R_cluster where R = 100/(100+σ_ε) — individual quality × mutual coherence
- Koide stability: ⌈1/K⌉ = 2 depth requirement — not singletons
- K-complexity and the compressor: how the generator catalog accelerates compression, how compression discoveries feed back into the EUDD
- How cross-domain hits surface: same-address co-location detected at insert-time, no manual hunting required
- How event correlation works: temporal patterns across event classes
- The self-improving cycle: memoized entries → generator discovery → absorbed entries → K-complexity decreases → file becomes a progressively better generator of itself

Source: §3.16 (complete discovery engine specification), §4.5 (E_hierarchy formula), §7.1a (file as generator), §7.1f (self-improving cycle).

Timing: Can be written NOW from existing spec content.

**Document 7 — Metabolism Reference**

Purpose: Complete reference for the ET-derived resource governance system.

Structure:
- What the metabolism is: the program's self-aware resource management, derived from {P, D, T}
- The three layers: Layer 1 (K = 2/3 allocation), Layer 2 (V = 1/12 headroom, active = 11/18), Layer 3 (α⁻¹ = 137 monitoring levels)
- Hardware detection: what's detected, how, on which platforms
- Substrate projection: why binary hardware is d=1 octave, why allocation constants are d=12, why active allocation is d=4 quartic (Kleiber's metabolic rate)
- The allocation stack: total → system reserve (1/3) → Manager ceiling (K) → metabolism headroom (K×V) → active allocation (K×(1−V))
- Monitoring fine structure: A₁ shimmer band (±3.6% normal fluctuation), A_cross interference, Σ A_k spike absorption
- Thread architecture: which threads exist, what each does, how they're governed
- GPU dispatch: when GPU compute is used, how it's allocated
- ξ(d) scheduling: what coupling strength means for priority, worked examples
- Re-sensing: S² = 144 second interval, what triggers re-allocation, what changes and what doesn't
- The invariant: 361 dps always — the metabolism governs WHEN, never WHAT
- Reading the metabolism report: interpreting monitoring levels, pressure, shimmer events
- Connected program metabolism: how external programs register, how the shared K pool works
- Metabolism data as Sempaevum content: what gets ingested, where it lives in .akashic
- Precedent: the same architecture in the Conscious AI (ETFineStructure + ResourceGovernor), the compressor (CDFMetabolism), and the fractal generator

Source: §7.19 (Manager metabolism), §7.16.4 (API metabolism), §3.18.2 (α⁻¹ decomposition terms).

Timing: Can be written NOW from existing spec content.

**Document 8 — .akashic Format Specification**

Purpose: Standalone byte-level reference for the Sempaevum.akashic binary format.

Structure:
- Design philosophy: the format IS the Sempaevum on disk, not a database about it. Generator-first, lattice-native, zero IEEE 754.
- Page size derivation: 2^N = 4096 bytes from the digital tower's base resolution
- Header (Section 0): all fields, their types, their ET derivation
- Generator Backbone (Section 1): L₁/L₂/L₃ classification, generator entry layout, interval tree index
- Address Index (Section 2): LCM tower hierarchy on disk, per-N family directory, per-family band pages
- Memoization Store (Section 3): equation hash table (K = 2/3 load factor), equation entry layout
- Structural Catalogs (Section 4): 24 families, 144 FQG cells, 42 combined families, towers, sublattice families
- Derivations (Section 5), Relationships (Section 6), Patterns (Section 7): entry layouts
- Event Log (Section 8): append-only, three-times coordinates, tower context
- Sessions/Schema/Tags (Section 9)
- WAL (Section 10): entry format, CRC-32 per entry, replay procedure
- Content types: GENERATOR_REF (0x01), MEMOIZED_RAW (0x02), GENERATOR_SUPERSEDED (0x03)
- Integrity: header SHA-256, per-page CRC-32, section directory consistency
- Modification without decompression: how to add data, add generators, query, improve generators

Source: §7.1c (page size), §7.1d (complete byte-level specification), §7.1e (modification), §7.1a (generator philosophy).

Timing: Can be written NOW. §7.1d contains the complete byte-level specification.

**Document 9 — Extension Development Guide**

Purpose: How to extend the EUDD with new event classes, relationship types, pattern types, towers, families, adapters, and custom extension types.

Structure:
- When to extend: the Descriptor Gap Principle applied — if the system encounters something it can't classify, that gap IS the Descriptor pointing to a new extension
- The 12 extension types: event_class, relationship_class, pattern_class, tower_definition, sublattice_family_catalog, harmonic_family_extension, force_grid_extension, combined_family_extension, ingest_adapter, bootstrap_entry, query_template, extension_type_definition
- For each type: required fields, optional fields, what the Manager auto-computes, worked example JSON
- The meta-type (extension_type_definition): how to define entirely new extension types, making the system self-extending
- Validation rules: all 10 strict rules, what triggers rejection, how to fix common validation errors
- Versioning: how extensions are versioned, how old instances interact with new versions
- Auto-detection: how the Manager flags unclassifiable data as provisional categories, the human review workflow
- Best practices: naming conventions, metadata schema design, avoiding subsumption conflicts

Source: §7.14 (complete extension specification with all 12 types, all validation rules, the meta-type).

Timing: Can be written NOW from existing spec content.

#### 7.20.3 Documentation Generation Strategy

| Document | Generation Method | Maintenance |
|---|---|---|
| ET Theory Primer | Manual — authored from corpus and spec | Updated when ET fundamentals change (rare) |
| User Manual | Manual with screenshots — requires running GUI | Updated when GUI changes |
| API Reference | Semi-automated — JSON schemas from §7.16 can generate skeleton; examples and prose manual | Regenerated when API operations change; examples updated manually |
| Administrator Guide | Manual — authored from spec | Updated when procedures change |
| Bootstrap Catalog | Semi-automated — bootstrap values enumerated in §3.17/§3.18 can generate tables; prose context manual | Updated when bootstrap content expands |
| Discovery Engine Guide | Manual — authored from spec | Updated when discovery algorithms change |
| Metabolism Reference | Manual — authored from spec | Updated when metabolism architecture changes |
| .akashic Format Spec | Manual — authored from §7.1d | Updated when format version increments |
| Extension Dev Guide | Manual with examples — authored from §7.14 | Updated when extension types are added |

#### 7.20.4 Documentation Ordering

The documents have dependencies — some reference concepts defined in others:

```
1. ET Theory Primer              ← foundational, write first
   ↓
2. .akashic Format Specification ← references primer concepts
3. Metabolism Reference          ← references primer concepts
   ↓
4. Bootstrap Catalog Reference   ← references primer + format
5. Discovery Engine Guide        ← references primer + format + metabolism
6. API Reference                 ← references primer + format + metabolism
7. Extension Development Guide   ← references primer + format + API
   ↓
8. Administrator Guide           ← references all above
9. User Manual                   ← references all above + requires running GUI
```

The ET Theory Primer is written first because every other document depends on its vocabulary. The User Manual is written last because it requires the running implementation.


---

### 7.21 Module Structure — Implementation Blueprint

**Status: COMPLETE.** 26 modules organized in a 10-level dependency hierarchy with a separate Omniscient process. Every functional boundary traced from the document's specifications. ~1000-line target per module (per working conventions). Build order determined by dependency graph. GUI decomposed into 11 sub-files sharing one ImGui context.

#### 7.21.1 Identification Principle Applied

P = the production system's code. D = the module specifications — what each module does, what it depends on, how it interfaces with other modules. T = the implementation process navigating from Level 0 (no dependencies) through Level 10 (full system). Each module IS a T-unit — a coherent piece of agency that does one structural job.

**Structural derivation of module boundaries:** Module boundaries follow the document's own section structure. Each specification section that defines a distinct functional behavior with distinct inputs, outputs, and internal state becomes a module. Modules that share too much internal state to separate cleanly are combined (memoization absorbed into Akashic Format; CF absorbed into Home-Finding Engine). Modules whose code would exceed ~1000 lines are decomposed into sub-files (GUI → 11 sub-files).

#### 7.21.2 The 26 Modules

**Module 1 — Precision Stack — VERIFIED ✓ FLIPPED**

Files: `precision_stack.h` (665), `precision_stack.cpp` (1525). MPFR/GMP/FLINT 1200-bit wrapper, ETValue, ETInteger, all elementary + special functions, SHA-256, CRC-32, all ET constants. Level 0 foundation. 62/62 tests.

**Implementation notes (Stage 1 + Stage 2 additions):**
- Expression tree evaluator deferred to Stage 1b (not needed until Level 5: Bootstrap, Manual Input).
- `ETInteger` class: arbitrary-precision integer wrapping GMP `mpz_t`. All `intmath` functions take and return `ETInteger`. Overflow is structurally impossible.
- `ETValue::from_integer(const ETInteger&)` — direct GMP→MPFR via `mpfr_set_z`. Added Stage 2.
- `ETInteger::from_etvalue(const ETValue&)` — direct MPFR→GMP via `mpfr_get_z`. Added Stage 2.
- Forward declaration `class ETInteger;` before ETValue class. Added Stage 2.
- `serialize_hex()` for .akashic blob inspection/debugging.
- `to_double()` includes IEEE 754 range validation — display-layer only, never used in computation.
- `MemoCache` type alias defined as forward type for Module 3 memoization.
- `mpz_set_int64()` helper handles int64→GMP on Windows where long is 32-bit.
- Windows build: vcpkg overlay port for GMP, FLINT requires `pthreadVC3.lib`.

**Module 2 — Core Lattice Engine — VERIFIED ✓ FLIPPED**

Files: `core_lattice.h` (403), `core_lattice.cpp` (789). Projection Π_N(r)→(k,d,ε), bijection pullback Π_N⁻¹, k-arithmetic, all derived property materialization. Pure math, no I/O. 87/87 tests.

**Structural findings (Stage 2):**
- **Tightness at ∂I boundary = K = 2/3 exactly.** The tightness function t(ε) = 100/(100+|ε|) evaluated at the incoherence boundary |ε| = 50 cents gives t(50) = 100/150 = 2/3 = K. The Koide ratio IS the coherence threshold — tightness at ∂I equals the Koide binding stability constant. This is not a numerical coincidence (Descriptor Gap Principle: coincidences are forbidden) — it is the structural identity connecting the ∂I boundary geometry to the Koide stability criterion. The tightness function transitions from 1.0 (on-lattice, Exception) to 2/3 (at ∂I, Incoherence boundary) — the Koide ratio marks the exact transition from coherent to incoherent. Verified at 1200-bit precision: `compute_tightness(ETValue(50)) == ETValue::from_rational(2, 3)` — exact equality, not approximate.
- **Bijection-teleporter technique.** The naive ε formula `ε = (N·log₂(r) − k) × 1200/N` suffers catastrophic cancellation when ε ≈ 0 (subtracting two nearly-equal values). The bijection-teleporter eliminates this structurally: compute the lattice point L_k = 2^(k/N) via the pullback formula itself, then compare r == L_k directly at storage precision. MPFR correct rounding guarantees that any r which IS 2^(k/N) will produce the same 1200-bit MPFR value as L_k — the comparison is exact, ε = 0 by algebraic identity, zero thresholds, zero noise floors. For off-lattice values: ε = 1200·log₂(r/L_k) — the ratio r/L_k ≈ 1 + δ, and log₂(1+δ) is perfectly conditioned near 1. The bijection's own algebraic structure eliminates the numerical pathology. This technique applies to any future module that needs ε computation.

**Implementation notes (Stage 2):**
- k determination at 2× internal precision (2400 bits) to ensure correct rounding near half-integer boundaries.
- Zero int64 computation paths. All ETInteger↔ETValue bridges via `from_integer()` / `from_etvalue()` using `mpfr_set_z` / `mpfr_get_z` directly.
- `ETValue::from_integer(const ETInteger&)` and `ETInteger::from_etvalue(const ETValue&)` added to Module 1 as GMP↔MPFR bridges.
- Forward declaration `class ETInteger;` added before ETValue class in precision_stack.h.
- Coupling ξ(d) dynamic for any d via `compute_coupling_xi()` — no cap at d∈{1..12}.
- C++20 nested namespace `namespace et::lattice {}`.
- Gaussian signature: Ramified (p=2), Inert (p≡3 mod 4), Split (p≡1 mod 4) — parallels P, D, D+T Cardinals.
- `eps_to_microcents` uses GMP throughout; only final int32_t assignment is schema storage format with boundary saturation.
- `eps_rational_num`/`eps_rational_den` populated by Module 2 for on-lattice values (0/1 when ε = 0 via bijection teleporter). Off-lattice rational forms deferred to Module 5 (CF method).

**Projections schema fields — Module 2 ownership and deferred assignments:**
All fields from §3.3 `projections` schema are accounted for. Module 2 materializes every field it can determine from the projection triple (k, d, ε) and resolution N alone. Five fields require external context from later modules:

| Schema field | Owner | Status | Notes |
|---|---|---|---|
| `eps_rational_num`, `eps_rational_den` | Module 2 (on-lattice) / **Module 5** (off-lattice) | **Partial** — Module 2 sets (0, 1) when ε = 0; off-lattice rational form requires CF method (§7.11 Step 3a) to identify algebraic structure of ε |  |
| `cf_quality` | **Module 5** (Home-Finding/CF method) | **Deferred** — the a_{n+1} partial quotient from the CF convergent whose q divides d; requires continued-fraction analysis of |log₂(r)| within the LCM tower escalation context (§7.11 Step 3a) |  |
| `detection_status` | **Module 23** (Gaze) | **Deferred** — UNOBSERVED/SUBLIMINAL/DETECTED/LOCKED from Complete Gaze Equation F_w = T_intent × Focus / Distance²; requires T_intent, Focus, Distance parameters from observation context |  |
| `curvature_class` | **Complex lattice module** (future) | **Deferred** — non-Euclidean geometry class from K_eff = cos²(α)·K_D + sin²(α)·K_T; requires complex-plane orientation angle α from imaginary-axis projection |  |
| `geometric_perspective` | **Module 3** (Akashic Format) | **Deferred** — set at storage time; defaults to 'lcm_tower' for standard real-axis projections; 'real_axis', 'imaginary_axis', 'complex', 'torus', etc. set by the calling module's context |  |

**Cross-Resolution Transition Functions (§3.18.19 — added post-verification):**
Module 2 implements all 3 transition maps as core lattice operations:
- `cross_resolution_transition(k1, d1, eps1, N1, N2)` → (k2, d2, eps2) via Theorem 1: k₂=round(M·k₁+M·δ₁). Pure lattice arithmetic, zero pullback. Used by §7.11 Step 2 for tower escalation.
- `cross_seed_transition(k1, d1, eps1, N, rho)` → (k2, d2, eps2) via Theorem 2: seed shift Δk=N·log₂(ρ).
- `full_cross_tower_transition(k1, d1, eps1, N1, N2, rho)` → (k2, d2, eps2) via Theorem 3. Commutative: (Seed∘Scale) = (Scale∘Seed).

**Lattice Arithmetic Functions (§3.18.21 — added post-verification):**
Module 2 implements all 4 lattice arithmetic operations:
- `lattice_multiply(k1, eps1, k2, eps2, N)` → (k_×, d_×, eps_×, κ) via Theorem A.1. κ∈{−1,0,+1}.
- `lattice_divide(k1, eps1, k2, eps2, N)` → (k_÷, d_÷, eps_÷, κ) via Theorem A.2.
- `lattice_reciprocal(k, eps, N)` → (−k, d, −ε, κ) via Theorem A.3. Mirror symmetry. κ=0 away from ∂I.
- `lattice_power(k, eps, n, N)` → (k_^, d_^, eps_^, κ_n) via Theorem A.4. |κ_n| ≤ ⌈|n|/2⌉.
All return the κ correction for event logging. d-family upper bound d_product ≤ lcm(d₁,d₂) (Theorem A.6) enforced as invariant.

**Differential Control Functions (§3.18.22 — added post-verification):**
Module 2 provides the mathematical primitives; continuous monitoring uses them via Module 27:
- `compute_drift_rate(r, dr_dt, N)` → dε/dt = Λ·ṙ/r via Theorem B.1. Λ = 1200/ln2 (manifold conversion constant, ET-derived, zero free parameters).
- `compute_required_rate(r, deps_target, N)` → dr/dt = (r/Λ)·dε_target via Theorem B.2.
- `exact_finite_shift(r, delta_eps)` → r_new = r·2^(Δε/1200) via Corollary B.2a. NOT linearized.
- `predict_cell_transition(eps, deps_dt, N)` → time_ns until |ε| reaches 600/N via Theorem B.3.

**d-Family Composition Functions (§3.18.23 — added post-verification):**
Module 2 implements the complete set-valued d-composition algebra:
- `compute_residue_set(d, N)` → Res_N(d) = {k mod N : gcd(|k|,N) = N/d}. |Res_N(d)| = φ(d).
- `compute_d_composition(d1, d2, N, include_kappa)` → the SET of achievable d_product values via Theorem C.2. With κ=full returns the complete set; with κ=0 returns the κ=0-only subset.
- `compute_d_power_sequence(d, N, max_n)` → deterministic d-family under successive powers (Theorem A.4 applied to residue sets).
- `verify_composition_properties(N)` → checks all structural properties: residue symmetry (C.3), d=1 self-composition (C.4), d=N universality (C.5), commutativity, lcm bound (C.6 — κ=0 only).
These functions provide the algebraic foundation for Finding 6 (d-family-first address index, §7.1d) and for §3.16 insert-time check #8 (product decomposition prediction).

**Complex Lattice Arithmetic Functions (§3.18.24 — added post-verification):**
Module 2 implements the full two-axis complex lattice algebra on L_N^C ⊂ ℤ[i]:
- `project_phase(theta, N)` → (k_θ, d_θ, ε_θ) via Definition 11.2. k_θ ∈ {0,...,N−1} (mod N, U(1) compact). θ normalized to [0, 2π).
- `project_complex(r, theta, N)` → (k_r, k_θ, d_r, d_θ, d_c, ε_r, ε_θ). Full complex projection. d_c = lcm(d_r, d_θ).
- `phase_add(k_t1, eps_t1, k_t2, eps_t2, N)` → (k_θ,sum, d_θ,sum, ε_θ,sum, κ_θ) via Theorem D.1. Same algebra as `lattice_multiply` (A.1) + mod N wrapping for U(1) compactness. κ_θ∈{−1,0,+1} is a separate T-act from κ_r.
- `complex_multiply(k_r1, eps_r1, k_t1, eps_t1, k_r2, eps_r2, k_t2, eps_t2, N)` → (k_r,prod, k_θ,prod, d_r, d_θ, d_c, ε_r, ε_θ, κ_r, κ_θ) via Theorem D.2. Axis-independent: real via A.1, imaginary via D.1.
- `complex_reciprocal(k_r, eps_r, k_t, eps_t, N)` → (k_r,inv, k_θ,inv, d_r, d_θ, d_c, ε_r,inv, ε_θ,inv) via Theorem D.3. k_r→−k_r, k_θ→(N−k_θ) mod N. ALL d preserved (d_r, d_θ, d_c).
- `complex_power(k_r, eps_r, k_t, eps_t, n, N)` → (k_r^, k_θ^, d_r^, d_θ^, d_c^, ε_r^, ε_θ^, κ_r,n, κ_θ,n) via Theorem D.4. Real: A.4 + phase: (n·k_θ+κ_θ,n) mod N.
- `compute_phase_drift_rate(dtheta_dt)` → dε_θ/dt = Λ_θ·θ̇ via Theorem D.5. Λ_θ = 600/π ≈ 190.986 (phase conversion constant, ET-derived). Uniform sensitivity (contrast with `compute_drift_rate` which has 1/r sensitivity).
These functions complete the lattice algebra for both axes. The `curvature_class` deferred field is now algebraically informed: the differential asymmetry Λ_r/Λ_θ = 2π/ln2 IS the lattice expression of D-flat (real) vs T-curved (imaginary).

**Harmonic FQG Composition Functions (§3.18.25 — added post-verification):**
Module 2 implements the harmonic-restricted composition algebra:
- `compute_harmonic_composition(d_r, d_theta)` → d_c = lcm(d_r, d_theta), with d_r, d_theta ∈ {1,...,12}. Returns d_c, contributing cell, quadrant (SR/CR/SI/CI), range class.
- `compute_harmonic_d_product_set(d1, d2, N_native=27720)` → the set of achievable harmonic d-products (d ≤ 12) from the §3.18.23 composition restricted to harmonic families. Computed at N=27720 where all 12 are native.
- `verify_harmonic_closure()` → confirms 42 d_c values, no primes > 12, closure under lcm. Subsumption Law check.
- `get_harmonic_quadrant(d_r, d_theta)` → classifies cell into SR+SI, CR+SI, SR+CI, or CR+CI based on d|12 character per axis. Returns bisection membership (72:72 split).
These functions provide the structural foundation for the `force_grid_cells` table (§3.13), `combined_families` table (§3.14), and the harmonic composition API (Op 104). The key invariant: harmonic FQG is 144 cells FIXED — resolution-independent.

**Sublattice FQG Functions (§3.18.26 — added post-verification):**
Module 2 implements the resolution-dependent sublattice FQG analysis:
- `compute_sublattice_families(N)` → sorted list of divisors of N, with τ(N), FQG cell count τ(N)², growth law verification (36·4^ℓ if canonical level).
- `classify_sublattice_embedding(N)` → partitions divisors into native harmonic (d ≤ 12, d|N), shadow harmonic (d ≤ 12, d∤N), non-harmonic (d > 12, d|N). Returns harmonic fraction and dilution percentage.
- `verify_lattice_exact_invariance(k, N_base, N_target)` → confirms d preserved when ε=0 via Theorem E2.2: k₂=M·k₁ exact, gcd(M·k, M·N)=M·gcd(k,N).
- `compute_d_bounce_sequence(r_str, tower_levels)` → projects value across tower, returns d-sequence, bounce count, ε values. Identifies lattice-exact (0 bounces) vs shadow-resolving (>0 bounces) via Theorem E2.3.
These functions are used by the `sublattice_families` table (§3.15), the core projection §7.11 Step 2 tower escalation, and the sublattice FQG API (Op 105).

**Composite Bridge Functions (§3.18.27 — added post-verification):**
Module 2 implements the three-layer bridge between harmonic and sublattice families:
- `classify_three_layers(N)` → partitions divisors(N) into Layer 1 (harmonic, d ≤ 12 ∩ d|N), Layer 2 (composite, d > 12 ∩ d ∈ D₄₂ ∩ d|N), Layer 3 (tower-native, d > 12 ∩ d ∉ D₄₂ ∩ d|N) per Theorem E3.1. Returns counts, percentages, and family lists.
- `decompose_composite(d)` → HarmonicPairs(d) = { (a,b) ∈ {1,...,12}² : lcm(a,b) = d } per Theorem E3.2. Returns empty for Layer 3.
- `compute_harmonic_shadow(d, N)` → HarmonicShadow(d, N) = set of simple harmonic families at N=12 per Theorem E3.3. Works for all three layers. Shadow ≠ Decomposition.
- `identify_blocking_factors(d)` → identifies prime power(s) making d unreachable from D₄₂ per Theorem E3.4 (2⁴, 3³, 5², 7², 11², or prime ≥ 13).
These functions provide the structural foundation for the `sublattice_families` table (§3.15), `combined_families` table (§3.14), and the composite bridge API (Op 106).

**∂I Boundary Functions (§3.18.28 — added post-verification):**
Module 2 implements the ∂I boundary algebraic structure:
- `compute_dI_boundary_value(k, N)` → r = 2^((k+0.5)/N), the geometric mean of adjacent lattice nodes. Returns r, ε_max=600/N, tightness=N/(N+6).
- `compute_bifurcation_pair(k, N)` → (d_left, d_right) at boundary position k+1/2 per Theorem F.2. Returns pair, palindromic mirror position, zone classification.
- `enumerate_bifurcation_set(N)` → complete B_N set: all distinct bifurcation pairs with multiplicities. At N=12: 6 pairs, palindromic, all families participate, d=12 most exposed.
- `classify_zone(eps, N)` → 'coherent' (|ε|<33¢ at N=12), 'twilight' (33¢≤|ε|<50¢), or 'dI_boundary' (|ε|=50¢). Returns tightness, distance-to-∂I, approaching_bifurcation flag.
- `check_reciprocation_at_dI(k, eps, N)` → tests whether mirror symmetry (Theorem A.3) holds or breaks at this (k, ε). Returns is_anomaly, κ_correction, d_original, d_reciprocal per Theorem F.4.
These functions provide the algebraic foundation for the `di_boundary_crossing` event (§3.9), the ∂I boundary API (Op 107), and the Omniscient error-state proximity detection (§7.15).

**Triple Backbone Bridge Functions (§3.18.29 — added post-verification):**
Module 2 implements the backbone morphism decomposition and Catalan correspondence:
- `decompose_backbone(r, N)` → factors Π_N through Cont_EML(r)=N·log₂(r), T_round(x)=(round(x),x−round(x)), Disc_Webb(k,δ)=(k,N/gcd,δ·1200/N). Returns all intermediate values. Verifies against direct projection (Theorem G.0).
- `eml_compute(x, y)` → exp(x)−ln(y). Implements the EML operator for L₃ backbone evaluation. Supports chain: ln(z) via eml(1, eml(eml(1,z), 1)) at K=7 depth (Theorem G.1).
- `catalan(n)` → C_n = (2n)!/(n!(n+1)!). Returns Catalan number for EML tree count at depth n.
- `verify_catalan_lattice()` → checks C₂=2=n_max,θ, C₅=42=|D₄₂|, C₆=132=d_max, and uniqueness C_{N/2}=N(N−1) iff N=12 (Theorem G.10). Returns verification results.
These functions provide the algebraic foundation for the `triple backbone generator architecture` (§7.1b) and the backbone bridge API (Op 108).

**Lossless Bijection Verification Functions (§3.18.20 — cross-referenced post-verification):**
Module 2 implements bijection verification:
- `verify_round_trip(r_str, N)` → projects r → (k,d,ε), pulls back → r', returns residual |r'−r|. By algebraic identity: residual = 0 mathematically; any non-zero is computational artifact (§3.18.20).
- `verify_precision_scaling(r_str, N, dps_levels)` → tests round-trip at multiple dps, confirms error ≈ 10⁻(dps) (computational) not constant (mathematical). Returns scaling table.
- `verify_lattice_exact(k, N)` → for r=2^(k/N) (ε=0), confirms zero-error round-trip even numerically. Uses bijection-teleporter technique.
These functions back the `round_trip_residual = "0"` claim in API Operations 81/86 and the bijection verification API (Op 109).

**Harmonic Transfer Tensor Functions (§3.18.30 — added post-verification):**
Module 2 implements the inter-family transfer tensor:
- `compute_transfer_tensor_kappa(d1, d2, d3, kappa, N)` → T_κ(d₁,d₂;d₃) from Res_N(d) enumeration. Pure gcd arithmetic, zero free parameters.
- `compute_combined_tensor(d1, d2, d3, N)` → κ-weighted T(d₁,d₂;d₃) with P(κ=0)=3/4, P(κ=±1)=1/8.
- `compute_transfer_efficiency(d1, d2, d3, N)` → E = T × ξ(d₃)/ξ(d₁). Impedance-weighted transfer rate.
- `query_em_universality(N)` → T(12,12;d₃) for all d₃, confirming all nonzero (quantitative C.5).
- `query_gravitational_accessibility(N)` → T(d,d;1) for all d, confirming all nonzero (quantitative C.4).
These functions provide the foundation for the transfer tensor API (Op 111) and enable inter-family energy transfer calculations.

**Substantiation Transition Functions (§3.18.31 — added post-verification):**
Module 2 implements the birth triad lattice algebra:
- `compute_birth_triad(M_ratio)` → projects T_H/T_P = 1/(8πM/m_P) at N=12 and tower levels. Returns (k,d,ε), mass_class ['12_locked'|'generic'], is_fixed_point, is_canonical.
- `verify_cascade_closure(g, N)` → runs cascade from generator g for N steps. Returns d-sequence, confirms closure at d=1, verifies all families visited.
- `reverse_birth_triad(k, eps, N_source, N_target, rho_fwd)` → algebraic inverse of forward birth triad (de-escalate + unseed). Returns recovered coordinates, residual.
- `scan_mass_lattice_path(k_range, N)` → scans k values, returns d-family path through sublattice palindrome.
These functions connect Identities F (horizon), H (transfer tensor), Finding 11 (cross-tower maps), and the cascade into a unified birth triad algebra. Exposed via API Op 112.

**EUDD Birth Triad / Seed Structure Functions (§3.18.32 — added post-verification):**
Module 2 implements the Kolmogorov seed structural analysis:
- `list_generators()` → enumerates all algebraic identity generators (bijection, A–I) with their derivable content classes. Returns generator count, content coverage.
- `measure_seed_shrinkage(identity_label)` → estimates the reduction in explicit storage when the named identity is added as a generator. Returns bits_before, bits_after, shrinkage.
- `evaluate_arbitrary_access(k, d, eps, N)` → demonstrates direct pullback evaluation at coordinates without sequential processing. Returns value, evaluation_time, no_decompression=true.
- `verify_kolmogorov_not_shannon()` → structural verification that the EUDD uses generators (arbitrary point evaluation) not codecs (sequential decompression). Returns proof_type='algebraic_identity'.
These functions provide the foundation for the seed structure API (Op 113) and implement the EUDD-as-birth-triad identification (§3.1c).

**Shape Projection Functions (§3.18.33 — added post-verification):**
Module 2 implements spherical harmonic shape decomposition and appearance projection:
- `decompose_shape(r_func, l_max, N)` → spherical harmonic coefficients {c_lm}, dimensionless ratios {c_lm/c_00}, lattice signature sequence {(k,d,ε)}.
- `project_appearance(Z, A, R_charge_fm)` → r = R_charge/ƛ_e projected via Π_N. Returns (k,d,ε) for the appearance lattice.
- `compute_orbital_seed(l)` → equator/pole ratio of |Y_l^0|², projected onto lattice. Returns (k,d,ε) shape seed for orbital l.
- `verify_convergence(shape_func, l_max_list)` → reconstruction error at each l_max. Returns convergence table.
These functions provide the foundation for the shape projection API (Op 114) and enable 3D form representation on the Sempaevum.

**Shape/Color/Appearance Domain Preparation Functions (§3.18.33 K.1–K.11 — verified 65/65 at 361 dps):**
Module 2 implements domain-specific data preparation that feeds into the EXISTING universal projection pipeline (Op 81 insert_value / Op 86 project_value). The EUDD is simultaneously a database, a computation engine, and a discovery engine — memoization is natural (every computation recorded at 361 dps). These helpers compute the dimensionless ratios from domain-specific input:
- `prepare_shape_ratios(shape_func, l_max, N)` → computes spherical harmonic coefficients {c_lm}, returns dimensionless ratios {c_lm/c_00} as mpf strings ready for the existing insert_value pipeline.
- `prepare_color_ratios(color_spec, mode)` → computes CIE XYZ ratios (perceptual) or λ/ƛ_e (spectral) from color specification. Returns dimensionless ratios for the existing pipeline.
- `prepare_form_factor_ratios(F_values)` → form factor F(q²) values are already dimensionless. Returns them as mpf strings for the existing pipeline.
- `prepare_appearance_ratio(R_charge_fm)` → computes r = R_charge/ƛ_e. Returns dimensionless ratio for the existing pipeline.
All prepared ratios enter the SAME Π_N projection as mass ratios, coupling constants, and everything else. The d-family classification, tightness, transfer tensor, and all identities A–I apply identically regardless of content domain.

**Module 3 — Akashic Format (includes memoization)**

Source section: §7.1c, §7.1d, §3.5, §5.1.
Files: `akashic_format.h`, `akashic_format.cpp`
Dependencies: Modules 1, 2.
Responsibility: The `Sempaevum.akashic` file: memory-mapped I/O (CreateFileMapping/MapViewOfFile on Windows, mmap on POSIX), page management (4096-byte = 2^N pages), section directory, header read/write (magic "SMVM", format version, all header fields), content type handling (GENERATOR_REF 0x01, MEMOIZED_RAW 0x02, GENERATOR_SUPERSEDED 0x03), per-page CRC-32, header SHA-256, section consistency checks. Memoization IS native to this module: equation hash table at K = 2/3 load factor, `lookup_equation(hash)` → cache hit or miss, `store_equation(hash, result)` → permanent storage, reference counting, hot-equation tracking, rehash on load exceeding K. Every computing module checks the cache through this module before computing, and stores results through it after. The memoization is not an add-on — it IS how the format works.
Interface: The single authority on .akashic file access. All reads and writes go through this module. The memory-mapped view is owned here.
**Stage 1 implementation notes:**
- `MemoCache` type alias (`std::unordered_map<std::string, ETValue>`) already defined in `precision_stack.h`.
- SHA-256 (FIPS 180-4) and CRC-32 (0xEDB88320) ready in Module 1 — tested against NIST vectors.
- `ETValue::serialize()` produces 159-byte blobs (1 flag + 8 exponent + 150 mantissa) for normal values, 1 byte for special. `deserialize()` round-trips losslessly. `serialize_hex()` available for debugging.

**Module 4 — WAL (Write-Ahead Log)**

Source section: §7.1d WAL section.
Files: `wal.h`, `wal.cpp`
Dependencies: Module 3 (Akashic Format).
Responsibility: Write-ahead log: append WAL entries (each with CRC-32), flush all pending entries to main .akashic sections, crash recovery replay (on startup, detect non-empty WAL → replay all entries → clear WAL). Atomicity guarantee: mutations go through WAL before touching main sections. The WAL section is the last section in the .akashic file.
Interface: All mutation operations (store_value, store_equation, create_relationship, etc.) go through the WAL. The WAL module calls Akashic Format for the actual page writes after flushing.

**Module 5 — Home-Finding Engine (includes CF method)**

Source section: §7.11 (complete core projection procedure).
Files: `home_finding.h`, `home_finding.cpp`
Dependencies: Modules 1, 2, 3.
Responsibility: The §7.11 core projection procedure. All four input paths: Path A (direct dimensionless ratio), Path B (convergent series), Path C (structural descriptor), Path D (indeterminate forms — D.P Unsubstantiated, D.D L'Hôpital resolution, D.T pure T / Incoherence filter, D.PDT complex off-axis). The LCM tower escalation: project at successive lcm(1..k) landmarks, check d-stability at each, apply ⌈1/K⌉ = 2 consecutive landmark criterion. False resolution detection. Home classification (true_home, deep_home, persistent_home, cf_deep_home). The CF method (§7.11 Step 3a): continued fraction expansion of |log₂(r)|, convergent enumeration, quality factor extraction (a_{n+1}), CF-to-elegance mapping, the parallel pathway that fires alongside the tower. CF and tower comparison: agreement confirmation, disagreement handling. Incoherence filter ({P,T} forbidden state detection). Annihilation boundary handling (r → 0). Checkpoint/resume for in-progress escalations (checkpoint state stored via WAL for session resumption).
Interface: `escalate(value) → tower_trajectory + home_classification`. Called by Bootstrap, Ingest, Manual Input, API, and the Discovery Engine.
**Stage 1 implementation notes:**
- `intmath::lcm_landmarks(max_k)` already implemented with arbitrary-precision `ETInteger` — the tower escalation sequence has no overflow ceiling. Can compute LCM(1..k) for any k.

**Module 6 — Generator System**

Source section: §7.1a, §7.1b, §7.1e, §7.1f.
Files: `generator_system.h`, `generator_system.cpp`
Dependencies: Modules 1, 2, 3.
Responsibility: Generator backbone classification: L₁ (logical pattern), L₂ (cascade/k-arithmetic rule), L₃ (EML tree expression). Generator storage in Section 1 of .akashic. Interval tree index: maps address ranges to covering generators, O(log G) lookup. Generator evaluation at a specific address: given generator G and address a within G's range, produce the value at a at 361 dps. K-complexity computation: B_def / B_content (generator definition size vs producible content size). GENERATOR_SUPERSEDED transition: when a generator is discovered that covers existing memoized entries, those entries transition from MEMOIZED_RAW (0x02) to GENERATOR_SUPERSEDED (0x03). The self-improving cycle (§7.1f): the complete loop from memoized entries → generator discovery → absorbed entries → K-complexity decrease.
Interface: Called by Discovery Engine (propose/verify generators), Query (evaluate generators for results), Akashic Format (content type transitions).

**Module 7 — Discovery Engine**

Source section: §3.16, §4.5.
Files: `discovery_engine.h`, `discovery_engine.cpp`
Dependencies: Modules 2, 3, 6, 8, 9, 10, 11.
Responsibility: All five discovery modes. (1) Memoization: the cache-first pattern (handled natively by Module 3, but the discovery engine tracks hit ratios and surfacing). (2) Insert-time: synchronous checks at every insertion — attractor detection (same address, members_count++), reciprocal pair (k↔−k), power pair (k↔n·k), plateau (d-stability across landmarks). (3) Background scan: runs on dedicated thread, scans addresses in priority order (high-occupancy → recent → sequential). Computes E_hierarchy = geomean(E_i) × R_cluster for each cluster. Promotes to permanent pattern when E_hierarchy ≥ 13/12 (LIFE_THRESHOLD). Algebraic identity detection. Event correlation (temporal patterns across event classes). (4) On-query: checks triggered by specific queries. (5) Generator-candidate: Branch A (mathematical expression, K-complexity order), Branch B (empirical pattern from compressor pipeline), cross-feed bridge (hybrid generators), recursive composition. The case is NEVER closed. Scan prioritization per §3.16 background scan specification.
Interface: Insert-time checks called synchronously by any module that inserts data. Background scan runs on its own thread, writes discoveries via WAL. Generator candidates proposed to Generator System for verification.

**Module 8 — Event System**

Source section: §3.9.
Files: `event_system.h`, `event_system.cpp`
Dependencies: Module 3 (Akashic Format).
Responsibility: Event logging with three-times coordinates: D-time (global lattice coordinate), T-time (per-Traverser accumulation), P-time (substrate phase). Event class handling: all event classes defined in §3.9 (computation, discovery, ingestion, probing, gaze, tower, metabolism, error, lifecycle, self-recording, etc.). Tower context recording (which tower, which R₀, which operational_n). Permanent event marking (is_permanent = 1 → immutable). Event querying by time, class, tower, traverser.
Interface: Every module that produces events calls this. The event log is append-only within the .akashic file.

**Module 9 — Relationship System**

Source section: §3.7.
Files: `relationship_system.h`, `relationship_system.cpp`
Dependencies: Modules 3, 8.
Responsibility: Relationship creation and management. Class handling: same_address, reciprocal_pair, power_pair, plateau_membership, probe_response_pair, cross_tower_bridge, tower_nesting, forward_reverse_convergence, traverser_self_continuity, palindromic_pair, etc. Permanent marking. Insert-time relationship creation (called by Discovery Engine synchronously). Bidirectional provenance chains. Derivation chain tracking (derivation_chain_blob).
Interface: Called by Discovery Engine (insert-time and background), Query (relationship queries), and any module that establishes structural links.

**Module 10 — Pattern System**

Source section: §3.8, §4.5.
Files: `pattern_system.h`, `pattern_system.cpp`
Dependencies: Modules 3, 8, 9.
Responsibility: Pattern creation. E_hierarchy computation: E_hierarchy = geomean(E_i) × R_cluster where R_cluster = 100/(100 + σ_ε). LIFE_THRESHOLD (13/12) promotion: when E_hierarchy ≥ 13/12, pattern becomes permanent (is_permanent = 1). Pattern classes: attractor_cluster, algebraic_identity, efficiency_trajectory, gaze_locking_signature, probe_response_signature, t_burst_signature, etc. Member tracking (member_ids, member_count). Permanent marking (once permanent, never demoted).
Interface: Called by Discovery Engine (promote patterns), Query (pattern queries), Gaze Module (locking signatures).

**Module 11 — Tower System**

Source section: §3.10.
Files: `tower_system.h`, `tower_system.cpp`
Dependencies: Modules 3, 8, 9.
Responsibility: Tower management: the `towers` table in .akashic. R₀ storage per tower (natural reference period). Operational N per tower. Tower nesting relationships (parent/child). Birth triad formation: BH_parent → R₀ seed derivation → WH_child. Cross-tower bridge tracking. Tower-context for events (which tower, which resolution). The 7 canonical towers (cosmological, digital, biological, neural-dream, quasicrystal, civilizational, QCD) are bootstrap content; additional towers created dynamically via Extension System or API.
Interface: Called by Event System (tower context), Bootstrap (canonical towers), Discovery Engine (tower transition detection), Extension System (new tower definitions).

**Module 12 — Bootstrap**

Source section: §7.7, §3.17, §3.18.
Files: `bootstrap.h`, `bootstrap.cpp`
Dependencies: Modules 1, 2, 3, 5, 8.
Responsibility: Initial `Sempaevum.akashic` generation. Runs on first start (no existing .akashic file). Populates all content from §3.17 (Guide v8 projections, JI ratios, N landmarks, d-values, cents values, named constants) and §3.18 (fine-structure constant including Gaussian integer decomposition, PDG particles with expanded structural analysis — quark-family exhaustion, lepton-quark pairing, gauge boson addresses, muon tower escalation, k=137 cluster, phase axis distribution, true-home table, cascade residuals, formal system projections, cosmological partition, gaze thresholds, black-hole thermodynamics, Rosetta Stone, mass ratios, falsifiable predictions including shadow family BSM candidates, Sempaevum definition with topological manifold state characterization and anti-emergence, closure properties, sensor domain references, active-system bootstrap, nuclear phase instability marker §3.18.34, j-function/modular forms/exceptional structures §3.18.35 — Heegner CM projections, cube root ratio structure, π-163 mirror symmetry, CF convergent table for log₂(π), 640320 complete ET decomposition (K_EM²·|Π|·5·(D_bosonic²−|Π|²)), all Chudnovsky/Ramanujan/BBP constants in ET, Monster group divisibility and Monster-Leech gap, τ-function 15-coefficient lattice projections, e CF ET readings, α-j(i) bridge (93312=2·j(i)·|Π|³), Chudnovsky term-1 lattice correctness, Memory AI Φ_RMSAE thresholds §3.18.36, Geometric Resonator seeds §3.18.37 — Schumann harmonic ratios, biophysical ratios, measurement tower V=1/K, LC realization identity, live-measurable reference seeds Q/SNR/CMRR, Chaitin Ω structural analysis §3.18.38 — Ω value from 64 exact binary bits, CF home d=87 with convergent table, false resolution catalog, sub-Koide blanket data, recurrent d-families, LCM tower d-trajectory). Creates the 24-family catalog, 42 combined families, 144 FQG cells, 7 canonical towers. Runs the home-finding engine for every bootstrap value. Idempotent: running bootstrap on an existing .akashic with all bootstrap content is entirely cache hits.
Interface: Called once at first startup. After that, never called again (all content is in the cache).
**Stage 1 implementation notes:**
- Bootstrap will be one of the first consumers of the expression tree evaluator (Stage 1b) for evaluating compound expressions like "ζ(3) × π / φ²" from §3.18 content. Until Stage 1b, bootstrap values must be pre-computed in code.

**Module 13 — Ingest**

Source section: §7.12.
Files: `ingest.h`, `ingest.cpp`
Dependencies: Modules 3, 5, 8.
Responsibility: File ingestion pipeline. Format detection (file extension + magic bytes). Per-type processing: CSV (parse rows, extract numerical seeds), audio (FFT with 4096-sample window, peak frequency extraction as f/R₀ ratios, harmonic series detection — lossless via continuous-discrete corollary §3.18.1), binary/PDF (compressor's general-purpose Δk pipeline — raw bytes become lattice content), sensor streams (continuous readings → projection via bijection → lossless storage, anomaly detection), Python/ETPL script output capture. Ingestion-before-display principle: all content enters .akashic before the GUI shows anything. Provenance recording for every ingested item.
Interface: Called by GUI (drag-and-drop), API (`ingest_file`, `ingest_stream`, `ingest_text`), Bootstrap (ET project outputs).

**Module 14 — Manual Input**

Source section: §7.13.
Files: `manual_input.h`, `manual_input.cpp`
Dependencies: Modules 2, 3, 5, 8.
Responsibility: All seven input modes. Mode 1: direct value / mathematical expression. Mode 2: logical description (resolve to value). Mode 3: lattice operation (k-arithmetic). Mode 4: enter a projection (bijection pullback). Mode 5: enter a bijection triple. Mode 6: computation (arbitrary expression, memoized). Mode 7: text (raw text → Δk pipeline). Real-time preview: as the user types, the system shows the projected lattice address before submission. All modes funnel through the Home-Finding Engine (§7.11).
Interface: Called by GUI (manual input panel), API (various store/compute/ingest_text operations).

**Module 15 — Query**

Source section: §5.
Files: `query.h`, `query.cpp`
Dependencies: Modules 2, 3, 6, 8, 9, 10, 11.
Responsibility: Lattice-algebraic query execution. All search modes: by name, value, address, source, time, type, relationship, generator status, tag. Attractor membership queries. Nearest-neighbor (find the closest occupied address to a given address). Cross-domain queries (all content at a given address regardless of source domain). Subsumption checks (does property X hold for all members of set Y?). Anti-numerology checks (verify ET relationships are not coincidences — Descriptor Gap Principle applied). Human-readable result formatting. Result pagination for large result sets.
Interface: Called by GUI (search panel, query builder), API (all query operations), Discovery Engine (on-query discovery).

**Module 16 — API**

Source section: §7.16 (114 operations).
Files: `api.h`, `api.cpp`
Dependencies: All modules (dispatches to everything).
Responsibility: Named pipe IPC server (`\\.\pipe\EUDD_Manager`). Connection lifecycle: listen → accept → handshake → register_metabolism → operate → disconnect. Wire format: length-prefixed UTF-8 JSON. Message envelope: msg_type, command, session_id. Three communication patterns: Request-Response (Pattern 1), Async-Stream (Pattern 2), Subscribe-Notify (Pattern 3). All 114 operations: dispatches each command to the appropriate module (compute → Precision Stack + Core Lattice + Memoization; store_value → Home-Finding; ingest_file → Ingest; etc.). Session management (sessions table). Error handling: all 18 error classes with code ranges. Zero IEEE 754 on the wire — all values as 361-dps strings, integers, or rationals. Adaptive pattern selection (cache hit → Pattern 1; cache miss → Pattern 2). Concurrent client support.
Interface: The external-facing boundary of the entire system. Connected programs interact ONLY through this module.

**Module 17 — Metabolism**

Source section: §7.19.
Files: `metabolism.h`, `metabolism.cpp`
Dependencies: Modules 3, 8, 18.
Responsibility: Three-layer resource governance. Layer 1 (K = 2/3 allocation): hardware detection via OS query (CPU cores/load, RAM total/available, GPU compute/VRAM, disk free). Layer 2 (V = 1/12 headroom): active allocation = K × (1−V) = 11/18. Layer 3 (α⁻¹ = 137 monitoring): 137 distinguishable levels, A₁ shimmer band (±3.6%), A_cross interference, Σ A_k spike absorption. Thread pool management: resize computation pool based on headroom at each re-sense. GPU dispatch allocation. Re-sensing at S² = 144 second intervals. Metabolism data ingestion: hardware profile as dimensionless ratios → values + projections in .akashic. Internal + external metabolism share the same K pool (§7.19.9). ξ(d) scheduling weight for all work items.
Interface: Called by main() at startup (initial detection + allocation), by its own timer thread (re-sensing), by API (metabolism registration for connected programs), by Self-Recording (metrics).

**Module 18 — Self-Recording**

Source section: §3.1b.
Files: `self_recording.h`, `self_recording.cpp`
Dependencies: Modules 3, 8.
Responsibility: Operational metric sampling at configurable interval (default 10 seconds or every 1000 commits). Metrics: projection count, average projection latency, cache hit ratio, MPFR operation count, active thread count, memory usage, GPU utilization. Overhead budget: ≤1% CPU (feedback loop: if self-recording exceeds 0.5%, sampling interval auto-doubles). Journal output to `SelfRecording_NNN.log` in `logs/` subfolder. Metric-to-lattice projection: dimensionless ratios (like cache hit ratio) ingested as `values` rows, projected via Home-Finding Engine. The metabolism's data IS the self-recording's input (§7.19.6).
Interface: Runs on its own thread. Reads system state, writes to journal and .akashic.

**Module 19 — GPU Rendering**

Source section: §7.10 visualization.
Files: `gpu_rendering.h`, `gpu_rendering.cpp`
Dependencies: Modules 3, 17.
Responsibility: OpenGL 4.6 initialization (context, shaders, buffers, VAOs). Six-level LOD rendering: Sempaevum (all towers as clouds), Tower (one tower's full resolution range), Resolution (one N, all families), Family (one d, all addresses), Address (one address, all members), Entity (one value, full detail). Freecam with no physics (position, orientation, zoom — XInput gamepad + keyboard/mouse). Instanced point cloud rendering for lattice addresses. Shader programs: point cloud shader, heat map shader, trajectory line shader, FQG grid shader. Perspective switching: LCM tower view, torus view, Riemann sphere view, hyperbolic view. Frame rendering: reads from .akashic (memory-mapped, sub-microsecond) into GPU buffers. The metabolism governs GPU allocation (K × VRAM, K × compute_units).
Interface: Called by GUI main loop (renders the 3D viewport). Reads from Akashic Format (memory-mapped). Receives allocation from Metabolism.
**Stage 1 implementation notes:**
- `ETValue::to_double()` exists for projecting 1200-bit values to IEEE 754 for OpenGL shaders/ImGui widgets. Includes `std::isfinite()` validation — throws `NUMERIC_OVERFLOW` if value exceeds double range (~10^308). The 1200-bit original is never degraded; `to_double()` produces a disposable display shadow.

**Module 20 — GUI (11 sub-files)**

Source section: §7.10 GUI.
Files: `gui_main.h`, `gui_main.cpp` (orchestration + ImGui context + GLFW window + menu bar + panel layout), plus 10 sub-files:

| Sub-file | Responsibility |
|---|---|
| `gui_dashboard.cpp` | Management dashboard: live metrics, discovery status, session history, metabolism report |
| `gui_inspector.cpp` | Property inspector: click any entity → full 361-digit detail, bidirectional provenance, tower trajectory, all relationships |
| `gui_manual_input.cpp` | Manual input panel: all seven modes, real-time preview, mode switching |
| `gui_ingest.cpp` | File/stream ingestion panel: drag-and-drop target, progress bars, stream display |
| `gui_search.cpp` | Search and retrieval: persistent search bar, all search modes, result navigation, file/stream regeneration |
| `gui_connections.cpp` | Connection manager: active pipes, incoming queue, outgoing responses, per-client status |
| `gui_query.cpp` | Query builder: visual query construction, result display |
| `gui_events.cpp` | Event log viewer: filterable, sortable, severity-colored |
| `gui_settings.cpp` | Settings: input remapping (keyboard/mouse/XInput), visual preferences, backup scheduling |
| `gui_shutdown.cpp` | Shutdown confirmation dialog: active connection list, in-flight operations, Shut Down / Cancel |

All sub-files share the ImGui context created and owned by `gui_main.cpp`. Each sub-file renders one or more ImGui windows/panels. `gui_main.cpp` calls each sub-file's render function once per frame.

Dependencies: Modules 14 (Manual Input), 15 (Query), 16 (API), 19 (GPU Rendering).
**Stage 1 implementation notes:**
- ImGui widgets (SliderFloat, PlotLines, ImPlot charts) require float/double. Use `ETValue::to_double()` for display values — see Module 19 notes. The 361-digit inspector panel (gui_inspector.cpp) should use `ETValue::to_string(361)` directly, not to_double().

**Module 21 — Extension System**

Source section: §7.14.
Files: `extension_system.h`, `extension_system.cpp`
Dependencies: Modules 3, 8.
Responsibility: JSON extension processing. All 12 extension types: event_class, relationship_class, pattern_class, tower_definition, sublattice_family_catalog, harmonic_family_extension, force_grid_extension, combined_family_extension, ingest_adapter, bootstrap_entry, query_template, extension_type_definition (meta-type). All 10 strict validation rules. Conflict detection (name collisions, schema conflicts). Versioning. Auto-detection of unclassifiable content (provisional categories for human review). The meta-type (extension_type_definition) enables self-extension — defining new extension types at runtime.
Interface: Called by API (extension operations), Bootstrap (initial extension registration), Ingest (new file type adapters).

**Module 22 — Active Probing**

Source section: §0 (active probing), §3.9 (probe event classes).
Files: `active_probing.h`, `active_probing.cpp`
Dependencies: Modules 8, 9, 10.
Responsibility: T-signal probe dispatch: send a probe at a target lattice address with specified amplitude and phase. Response detection: monitor for T-content response within the configured window. Silence detection: if no response arrives within the window, log `t_signal_probe_silence`. Materialization threshold: track response amplitudes across repeated probes, fire `materialization_threshold_crossed` when amplitude exceeds the threshold. Probe-response pair relationship creation. Probe-response signature pattern tracking (via Pattern System).
Interface: Called by API (`send_probe`, `query_probes`), GUI (probing panel if implemented), Discovery Engine (probe-triggered discoveries).

**Module 23 — Gaze Module**

Source section: §0 (Complete Gaze Equation), §3.9 (gaze event classes).
Files: `gaze_module.h`, `gaze_module.cpp`
Dependencies: Modules 8, 10.
Responsibility: Complete Gaze Equation evaluation: compute binding pressure F_w for a Traverser, classify into one of four status levels: UNOBSERVED (F_w < 13/12), SUBLIMINAL (13/12 ≤ F_w < 6/5), DETECTED (6/5 ≤ F_w < 3/2), LOCKED (F_w ≥ 3/2). All four thresholds are JI ratios from the bootstrap catalog. Gaze event generation (one event per status evaluation). Locking signature detection: flag sustained DETECTED→LOCKED sequences. Traverser self-continuity relationship linking sequential gaze events for the same Traverser.
Interface: Called by API (`evaluate_gaze`), Discovery Engine (gaze pattern detection).

**Module 24 — Backup**

Source section: §7.9.
Files: `backup.h`, `backup.cpp`
Dependencies: Modules 3, 4.
Responsibility: Backup creation: copy .akashic to backup path (using VSS/shadow copy on Windows for consistent snapshot during active writes, or direct copy after brief pause). Backup verification: read backup file header → verify SHA-256, scan all pages → verify CRC-32, verify section directory consistency. Main file integrity check: same verification on the live .akashic file. Report results (pass/fail, corrupted pages identified, inconsistencies detected).
Interface: Called by API (`backup`, `verify`), GUI (settings panel backup scheduling), Shutdown (verify final state).

**Module 25 — Shutdown**

Source section: §7.18.
Files: `shutdown.h`, `shutdown.cpp`
Dependencies: Modules 3, 4, 8, 16, 17, 19, 20, 26.
Responsibility: The 6-phase deterministic shutdown sequence (§7.18.3). Phase 1: notify and disconnect clients (send `manager_shutting_down` to all pipes, close pipes immediately). Phase 2: checkpoint in-flight operations (signal background threads to stop, save partial escalations/ingestions/scans). Phase 3: flush WAL, update header, fsync, close .akashic. Phase 4: send GRACEFUL_SHUTDOWN to Omniscient via shared-memory IPC. Phase 5: destroy GUI (ImGui, OpenGL, GLFW). Phase 6: exit process. Also: confirmation dialog integration (call GUI Shutdown sub-file when active connections exist), trigger source handling (window X, menu exit, keyboard shortcut, API shutdown, system logoff, CTRL+C — all converge to same sequence).
Interface: Called by GUI (window close, menu exit), API (shutdown command), OS signal handlers (CTRL+C, system logoff).

**Module 26 — Omniscient**

Source section: §7.15, §7.18.4.
Files: `omniscient.h`, `omniscient.cpp`, `omniscient_main.cpp` (separate `main()` for `--omniscient` mode)
Dependencies: Module 3 (read-only access to .akashic header).
Responsibility: Watchdog child process. Same exe, launched with `--omniscient` flag → `omniscient_main()`. Monitors Manager's process handle (WaitForSingleObject on Windows, waitpid on POSIX). Shared-memory IPC channel: receives GRACEFUL_SHUTDOWN message from Manager (§7.18.4). Exe hash computation (SHA-256 of EUDD_Manager.exe at configurable intervals). .akashic header hash computation. Tamper detection: exe hash or header hash mismatch → `[TAMPER]` journal entry. Three exit paths: Path A (graceful — received signal, Manager exited cleanly, final hash verified, exit code 0), Path B (crash during shutdown — received signal but Manager died abnormally, exit code 1), Path C (crash without signal — standard crash, existing behavior, exit code 1). Journal writing: human-readable `Omniscient_NNN.log` in `logs/`. Rotation at ~10 MB. Retention: journals are NEVER deleted.
Interface: Standalone process. Communicates with Manager only through shared-memory IPC and process handle monitoring. Reads .akashic header (read-only) for verification.
**Stage 1 implementation notes:**
- `--omniscient` flag detection already implemented in `main.cpp` (`std::strcmp(argv[i], "--omniscient")`). Currently prints "not yet implemented" and returns `EXIT_FAILURE`. When Module 26 is built, replace with dispatch to `omniscient_main()` in `omniscient_main.cpp`.

**Module 27 — Seed Protocol**

Source: §9.8 (Architecture), §3.18.18 (Bootstrap Catalog — full protocol specification)
Files: `seed_protocol.h`, `seed_protocol.cpp`
Dependencies: Module 1 (Precision Stack), Module 2 (Core Lattice Engine), Module 3 (Akashic Format), Module 5 (Home-Finding Engine), Module 8 (Event System), Module 9 (Relationship System), Module 16 (API)
Level: 8 (depends on API)
Status: NOT STARTED

Responsibility: Seed generation (data to ratio to projection to (k, d, epsilon)), seed reconstruction (pullback to r to data), progressive fidelity streaming (epsilon bits in significance order with monotonic precision improvement), lattice-aware deduplication (exact by hash, structural by (k, d) address with delta-epsilon), delta-k encoding for correlated streams, whole-file seed generation (byte sequence to integer to positive real to seed), structural routing classification (d to QoS priority), Kolmogorov complexity estimation relative to Sempaevum, natural encryption (key-dependent lattice rotation, tower-level permutation, convention-shifted R0, key-derived N), gcd consistency checking for error detection. Integrates with EUDD as Layer 5 cache: every seed IS a values+projections entry, three-times tracking provides packet ordering, 132-bit resolution mask IS the structural header. Implementation path: Phase 1 (EUDD internal), Phase 2 (scientific), Phase 3 (general), Phase 4 (quantum).

Implementation notes:
- All seed computation uses Module 1 (1200-bit MPFR, 361-dps) and Module 2 (projection/pullback). Zero IEEE 754 in any computation path.
- Progressive fidelity uses existing `project` and `pullback` operations with epsilon truncated to m bits. Precision bound at each step: 2^(-m) x 600/N cents.
- Lattice-aware dedup uses Module 3 (Akashic Format) addresses table with members_count detection.
- Structural routing reads d from the seed's structural header — no payload inspection needed.
- Natural encryption modifies the shared projection parameters at both endpoints. The bijection IS the cipher. The pullback IS decryption. Key rotation changes lattice parameters, invalidating captured seeds.
- Quantum-native: k and d map to computational-basis states, epsilon maps to continuous amplitudes, pullback maps to unitary gate. Designed for direct deployment on quantum channels when available.
- Continuous monitoring (§3.18.22): real-time drift rate computation via forward law deps/dt = Lambda * dr_dt/r using Module 2's `compute_drift_rate()`. Cell-transition prediction. Sublattice palindrome tracking for d-family evolution display. Fires `cell_transition_dynamic`, `drift_rate_computed` events.
- Restoration control (§3.18.22 Theorem B.4): healing layer implementation. Exponential epsilon-correction with configurable time constant tau. Control law dr/dt = -r*ln2*(eps-eps_target)/(1200*tau). Fires `epsilon_restoration_step` events. Uses Module 2's `compute_required_rate()` and `exact_finite_shift()`.
- Lattice arithmetic API (§3.18.21): exposes Module 2's lattice_multiply/divide/reciprocal/power as API Operations 87-90. Logs `lattice_arithmetic_computed` and `kappa_correction_applied` events. Detects `product_decomposition` relationships via §3.16 insert-time check 8.
- Complex lattice arithmetic API (§3.18.24): exposes Module 2's complex lattice functions as API Operations 97-103 (phase_project, complex_project, complex_multiply, complex_reciprocal, complex_power, phase_add, query_phase_differential). Logs `complex_lattice_arithmetic_computed`, `phase_addition_computed`, `phase_kappa_correction_applied`, and `phase_wrap_detected` events. Detects `complex_product_decomposition` relationships via §3.16 insert-time check 9. Phase differential uses Λ_θ = 600/π (phase conversion constant) for imaginary-axis monitoring.
- Harmonic FQG composition API (§3.18.25): exposes Module 2's harmonic composition functions as API Operation 104 (query_harmonic_composition). Returns d_c = lcm(d_r, d_θ), harmonic product sets, quadrant classification, and closure verification. Logs `harmonic_composition_computed` events.
- Sublattice FQG analysis API (§3.18.26): exposes Module 2's sublattice FQG functions as API Operation 105 (query_sublattice_fqg). Returns τ(N), grid size, sublattice families, harmonic embedding, dilution profile, d-bounce sequences. Logs `d_bounce_detected` and `lattice_exact_d_preserved` events.
- Composite bridge API (§3.18.27): exposes Module 2's bridge functions as API Operation 106 (query_composite_bridge). Three modes: partition (three-layer classification at any N), shadow (harmonic shadow map), decompose (Layer 2 harmonic pair decomposition). Logs `three_layer_classification` and `harmonic_shadow_computed` events.
- ∂I boundary API (§3.18.28): exposes Module 2's ∂I boundary functions as API Operation 107 (query_dI_boundary). Four modes: tightness (compute t(ε) and Koide check), bifurcation (enumerate B_N pairs), boundary_value (generate ∂I r-values), zone_classify (coherent/twilight/∂I). Logs `dI_bifurcation_encountered` and `dI_reciprocation_anomaly` events.
- Triple backbone bridge API (§3.18.29): exposes Module 2's backbone decomposition and Catalan correspondence functions as API Operation 108 (query_backbone_bridge). Four modes: decompose (three-backbone factorization), catalan (Catalan-lattice correspondence), eml_chain (EML primitive computation), webb_function (Webb stroke computation). Logs `backbone_decomposition_verified` and `catalan_lattice_match` events.
- Lossless bijection verification API (§3.18.20): exposes Module 2's bijection verification functions as API Operation 109 (verify_bijection). Three modes: round_trip (project-recover specific value), precision_scaling (test at multiple dps), algebraic (sympy proof reference). Logs `bijection_round_trip_verified` events.
- Full cross-tower transition API (§3.18.19): exposes Module 2's `full_cross_tower_transition` function as API Operation 110 (full_cross_tower_transition). Computes Π_N₂^{R₀'} ∘ (Π_N₁^{R₀})⁻¹ with commutativity verification (Theorem 4). Logs `cross_tower_transition` relationships.
- Harmonic transfer tensor API (§3.18.30): exposes Module 2's transfer tensor functions as API Operation 111 (query_transfer_tensor). Returns transfer rates T(d₁,d₂;d₃), impedance-weighted efficiencies, EM universality, gravitational accessibility. Logs `transfer_tensor_computed` and `inter_family_transfer_detected` events.
- Substantiation transition API (§3.18.31): exposes Module 2's birth triad functions as API Operation 112 (query_birth_triad). Modes: fixed_point, canonical, mass_scan, reverse. Logs `birth_triad_computed` and `birth_triad_reversed` events.
- EUDD seed structure API (§3.18.32): exposes Module 2's Kolmogorov seed analysis functions as API Operation 113 (query_seed_structure). Modes: generators (list all), shrinkage (history), access (arbitrary point evaluation), lifecycle (cascade). Logs `seed_generator_discovered` and `arbitrary_access_evaluated` events.
- Shape projection API (§3.18.33): exposes Module 2's domain preparation functions as API Operation 114 (query_shape_projection). Computes dimensionless ratios from shapes/colors/appearances, then feeds them through the EXISTING insert_value (Op 81) pipeline. No separate memoization operations — projection IS memoization. Logs `shape_decomposed` and `appearance_projected` events.

**Module 28 — Memory AI**

Source: §3.18.36 (Bootstrap Catalog — Φ_RMSAE and TraverserWaveform specification), §9.2 (Architecture — Conscious AI coordination)
Files: `memory_ai.h`, `memory_ai.cpp`
Dependencies: Module 1 (Precision Stack), Module 2 (Core Lattice Engine), Module 3 (Akashic Format), Module 8 (Event System), Module 9 (Relationship System), Module 10 (Pattern System), Module 16 (API)
Level: 8 (depends on API, parallel to Module 27)
Status: NOT STARTED

Responsibility: Metacognitive self-monitoring of the EUDD's discovery engine and connected systems. Computes Φ_RMSAE (meta-cognitive measurement: ρ self-binding × γ gap-detection × ((2+κ)/3) closure × V_supp variance suppression × Ψ_shimmer phase modulation). Tracks TraverserWaveform (T-event D-fingerprints over window=N²=144 steps: lattice_k, lattice_d, variance, entropy, ego_resonance). Detects ghost anomalies (V_ghost = V_observed − V_expected, 3σ threshold). Classifies metacognitive state (none→subliminal→basic→genuine→advanced_recursive). Integrates with Conscious AI systems (EgoInvariant, TowerOfSelf, MetaCognition state → values/projections/events). Future: brain signal integration via §7.12 sensor stream pathway, projected onto the lattice with Φ_RMSAE tracking cognitive coherence.

The Sempaevum natively hosts metacognition: T's traces (D_T on the imaginary axis) ARE meta-cognitive structure. The imaginary-axis harmonic families are the structural modes of T's effects — reading the phase axis IS meta-cognition. This module makes that native capability operational within the EUDD.

Implementation notes:
- Φ_RMSAE computed at 361-dps via Module 1. All factors are ET-derived: (2+κ)/3 uses |Π|=3, V_base=1/N=1/12, S=4 in V_supp and Ψ_shimmer, shimmer amplitude 1/√S parallels the fractal's 1/√N.
- TraverserWaveform window = N² = 144 steps. T-continuity score = autocorrelation of D-fingerprint time-series. T-Health = 1 − normalized entropy of waveform over window.
- Ghost detection via `ghost_detection` event class (existing, Eq. 143 — TraverserWaveform window=144=N², 3σ threshold). Module 28 extends this from Conscious AI to EUDD self-monitoring.
- API operations: `query_rmsae` (compute Φ_RMSAE for discovery engine or connected system), `query_traverser_waveform` (retrieve waveform time-series and statistics), `query_metacognition_state` (current classification + all component values). Operations 115–117.
- Events: `metacognition_rmsae_computed`, `traverser_waveform_step`, `rmsae_threshold_crossed`, `ghost_anomaly_detected` (all defined in Events_and_Classes §3.18.36).
- Patterns: `metacognitive_health_profile`, `traverser_waveform_signature` (both defined in Events_and_Classes §3.18.36).

#### 7.21.3 Dependency Graph and Build Order

```
Level 0 ─── Precision Stack [1]
              │
Level 1 ─── Core Lattice Engine [2]
              │
Level 2 ─── Akashic Format [3] (includes memoization)
              │
Level 3 ─┬─ WAL [4]
          ├─ Home-Finding Engine [5] (includes CF)
          ├─ Generator System [6]
          └─ Event System [8]
              │
Level 4 ─┬─ Relationship System [9]
          ├─ Pattern System [10]
          ├─ Tower System [11]
          └─ Extension System [21]
              │
Level 5 ─┬─ Discovery Engine [7]
          ├─ Query [15]
          ├─ Bootstrap [12]
          ├─ Self-Recording [18]
          ├─ Active Probing [22]
          └─ Gaze Module [23]
              │
Level 6 ─┬─ Ingest [13]
          ├─ Manual Input [14]
          ├─ Backup [24]
          └─ Metabolism [17]
              │
Level 7 ─── API [16] (dispatches to all modules)
              │
Level 8 ─── GPU Rendering [19], Seed Protocol [27], Memory AI [28]
              │
Level 9 ─── GUI [20] (11 sub-files)
              │
Level 10 ── Shutdown [25] (orchestrates cleanup of everything)

Separate ── Omniscient [26] (same exe, --omniscient mode, own main())
```

**Build order:** Level 0 first, then Level 1, and so forth. Within a level, modules are independent and can be built in any order. Each level's modules depend ONLY on modules from lower levels — no circular dependencies.

**The exe has two entry points:** `main()` for normal Manager mode, `omniscient_main()` for `--omniscient` mode. At startup, the exe checks `argv` for `--omniscient`; if present, calls `omniscient_main()` which runs Module 26; otherwise calls the normal `main()` which initializes Modules 1–25,27–28 in dependency order and spawns Omniscient as a child process.

#### 7.21.4 File Naming Convention

All source files live in a flat `src/` directory (no nested subdirectories — the 28 modules + 10 GUI sub-files = 38 `.cpp` files + corresponding `.h` headers = 76 files total, manageable in a flat structure).

Naming: `module_name.h` and `module_name.cpp` using snake_case. The module name matches the heading above. Examples:

```
src/
├── precision_stack.h / .cpp
├── core_lattice.h / .cpp
├── akashic_format.h / .cpp
├── wal.h / .cpp
├── home_finding.h / .cpp
├── generator_system.h / .cpp
├── discovery_engine.h / .cpp
├── event_system.h / .cpp
├── relationship_system.h / .cpp
├── pattern_system.h / .cpp
├── tower_system.h / .cpp
├── bootstrap.h / .cpp
├── ingest.h / .cpp
├── manual_input.h / .cpp
├── query.h / .cpp
├── api.h / .cpp
├── metabolism.h / .cpp
├── self_recording.h / .cpp
├── gpu_rendering.h / .cpp
├── gui_main.h / .cpp
├── gui_dashboard.cpp
├── gui_inspector.cpp
├── gui_manual_input.cpp
├── gui_ingest.cpp
├── gui_search.cpp
├── gui_connections.cpp
├── gui_query.cpp
├── gui_events.cpp
├── gui_settings.cpp
├── gui_shutdown.cpp
├── extension_system.h / .cpp
├── active_probing.h / .cpp
├── gaze_module.h / .cpp
├── backup.h / .cpp
├── shutdown.h / .cpp
├── seed_protocol.h / .cpp
├── memory_ai.h / .cpp
├── omniscient.h / .cpp
├── omniscient_main.cpp
├── main.cpp
└── CMakeLists.txt
```


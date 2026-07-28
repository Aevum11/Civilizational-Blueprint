# EUDD — Architecture Guide
## The ET Universal Discovery Database — Design, Schema, Format, and Operational Specifications

**Source:** Main guide from EUDD v39, de-duplicated. Content previously stated multiple times is recorded once in its authoritative section.
**Master index:** See `EUDD_Table_of_Contents.md` for navigation across all EUDD files.
**Companion files:** `EUDD_Events_and_Classes.md` (§3.9), `EUDD_API_Reference.md` (§7.16), `EUDD_Testing.md` (§7.17), `EUDD_Bootstrap_Catalog.md` (§3.17–§3.18), `EUDD_Module_Structure.md` (§7.20–§7.21)

---

# The ET Universal Discovery Database
## Generalizing the Compressor's Persistent Memory to Every ET Discovery

**Author derivation standard:** All architecture ET-native, forward from {P, D, T}. Zero external axioms. No tuning. No ad hoc.

**Tools applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law · Verification Principle

---

## IMPLEMENTATION STATUS — Working Tracker

**Last updated:** 2026-05-05 (Session 2 — Stage 2 VERIFIED)

| Module | Level | Status | Notes |
|---|---|---|---|
| **1 — Precision Stack** | 0 | **VERIFIED ✓** | 62/62 tests. Files: `precision_stack.h` (665), `precision_stack.cpp` (1525), `main.cpp` (892), `CMakeLists.txt` (131). Total: 3213 lines. 1200-bit/361-dps via MPFR/GMP/FLINT. ETInteger arbitrary-precision, ETValue 361-dps. Direct GMP↔MPFR bridges (`mpfr_set_z`/`mpfr_get_z`). Expression tree evaluator deferred to Stage 1b. Full implementation notes: `EUDD_Module_Structure.md` §7.21.2. |
| **2 — Core Lattice Engine** | 1 | **VERIFIED ✓** | 87/87 tests. Files: `core_lattice.h` (403), `core_lattice.cpp` (789). Projection Π_N(r)→(k,d,ε), bijection pullback Π_N⁻¹, k-arithmetic (add/negate/scale), all derived properties materialized at insert. Bijection-teleporter ε computation (zero catastrophic cancellation). k at 2× precision (2400-bit). Coupling ξ(d) dynamic for any d. Zero int64 paths. Structural finding: tightness at ∂I = K = 2/3 exactly. Full implementation notes: `EUDD_Module_Structure.md` §7.21.2. |
| 3 — Akashic Format | 2 | NOT STARTED | Depends on: Modules 1, 2 |
| 4 — WAL | 3 | NOT STARTED | Depends on: Module 3 |
| 5 — Home-Finding Engine | 3 | NOT STARTED | Depends on: Modules 1, 2, 3 |
| 6 — Generator System | 3 | NOT STARTED | Depends on: Modules 1, 2, 3 |
| 8 — Event System | 3 | NOT STARTED | Depends on: Module 3 |
| 9 — Relationship System | 4 | NOT STARTED | |
| 10 — Pattern System | 4 | NOT STARTED | |
| 11 — Tower System | 4 | NOT STARTED | |
| 21 — Extension System | 4 | NOT STARTED | |
| 7 — Discovery Engine | 5 | NOT STARTED | |
| 15 — Query | 5 | NOT STARTED | |
| 12 — Bootstrap | 5 | NOT STARTED | |
| 18 — Self-Recording | 5 | NOT STARTED | |
| 22 — Active Probing | 5 | NOT STARTED | |
| 23 — Gaze Module | 5 | NOT STARTED | |
| 13 — Ingest | 6 | NOT STARTED | |
| 14 — Manual Input | 6 | NOT STARTED | |
| 24 — Backup | 6 | NOT STARTED | |
| 17 — Metabolism | 6 | NOT STARTED | |
| 16 — API | 7 | NOT STARTED | |
| 19 — GPU Rendering | 8 | NOT STARTED | |
| 20 — GUI | 9 | NOT STARTED | |
| 25 — Shutdown | 10 | NOT STARTED | |
| 26 — Omniscient | Separate | NOT STARTED | |

**Stage 1 deferred items:**
- Expression tree evaluator (parses "ζ(3) × π / φ²" → AST → evaluates) — deferred to Stage 1b, needed first at Level 5 (Bootstrap, Manual Input)

**Next stage:** Module 3 (Akashic Format) — Sempaevum.akashic I/O, memoization hash table, page management, section directory, integrity

**Build environment (Windows):**
- MSVC 2022 Build Tools (v143), C++20, CMake ≥ 3.24
- vcpkg at `C:\vcpkg` with `x64-windows-static` triplet
- GMP requires overlay port: run `patch_gmp_port.ps1` → `setup_deps.bat` (fixes VPATH/libtool issues in GMP's autotools build on Windows)
- FLINT requires `pthreadVC3.lib` (installed automatically by vcpkg as FLINT dependency)
- CLion CMake options: `-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static`

---

**Foundation:** The compressor's `ArchetypeDatabase` (lines 7790–8400 of `et_cdf_compressor.py`) demonstrates that a persistent discovery database with ET-derived stability filters genuinely accumulates structural knowledge and produces escalating efficiency on subsequent runs. This document generalizes that proven pattern to ALL ET discoveries across ALL domains.

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *Discovery is irreversible; loss of discovery is forbidden.*

---

## 0. Direct Statement

**Claim:** A unified ET Universal Discovery Database (EUDD) — that is BOTH a database AND a discovery engine, the way the compressor's `ArchetypeDatabase` is — can absolutely be built. It scales to any size (the .akashic file is a generator whose size reflects structural complexity, not raw data volume), surfaces patterns automatically as data accumulates, and is the natural architectural extension of the compressor's proven mechanism applied to all ET work.

**The conceptual foundation (per Mike's clarification):** The Sempaevum (the lattice) is **simultaneously** an LCM tower, a torus, a Riemann sphere, and any other geometry needed. These are not separate objects — they are the same Sempaevum viewed through different geometric perspectives. The "tower" we observe is the higher-resolution navigation we follow as cents reach the ∂I incoherency boundary. **All projections at different N values land on the same unified main lattice.** Cross-tower / cross-perspective analysis is a real first-class feature — it reveals real correlations between perspectives on the same lattice address.

**Design principle (corrected):** the schema is **lattice-native, not bureaucratic**.

The lattice itself produces: dimensionless seeds (values), projections onto addresses, the addresses themselves (the lattice grid), equations, derivations, relationships between entries, and discovered patterns. Combined with events, towers, harmonic families, force grid cells, combined families, sublattice families, sessions, and schema versions, these **fifteen core structural categories** — each corresponding to an object the Sempaevum actually produces — form the logical schema (§3.2–§3.15b). Plus one optional `tags` category for query convenience.

What is NOT in the schema: per-project tables ("compressor_table", "ai_table", "biology_table"). Per-domain tables ("CMB_table", "music_table"). Per-verification-tier tables. Per-input-method tables. None of those are lattice-native objects — they're labels users might attach to entries. Labels are tag values, not separate scaffolding.

**Source of an entry = the dimensionless seed itself + the projection used.** That IS the natural identity — no additional domain label is required for the database to function. If Mike wants to query by domain later, he tags entries; the tags table handles it as a queryable property without forcing every entry through a domain bottleneck.

**Detailed specifications for all topics introduced above:** Architectural pathway and format design (§7.1). Technology stack and GUI (§7.10). Computation and memoization (§3.1a, §3.5). Discovery engine and generators (§3.16). Event classes, relationship classes, and pattern classes (`EUDD_Events_and_Classes.md`). Bootstrap values and extended theoretical recordings (`EUDD_Bootstrap_Catalog.md`). API specification (`EUDD_API_Reference.md`). Testing (`EUDD_Testing.md`). Module structure (`EUDD_Module_Structure.md`). Coordination with existing ET software (§9).

## 1. The Three Tools Applied

### 1.1 Identification Principle

A "discovery" is a P∘D∘T = E configuration that emerged from ET work and was verified:

| Primitive | What a discovery contributes |
|---|---|
| **P** | The substrate the discovery was made on (the lattice, the corpus, a specific physical phenomenon, etc.) |
| **D** | The structural content of the discovery (specific lattice tuple, attractor membership, classification) plus its provenance (which project, which derivation, what verification level) |
| **T** | The act of discovery itself — the substantiation event that pulled the discovery from the indeterminate-prior state to the determined-fact state, including the verification operator that confirmed it |

A complete discovery is identified by:
- **What it is** (D-content): specific values, relationships, classifications
- **Where it lives** (P-content): which substrate / problem domain
- **How it was found** (T-content): derivation path, verification level, stability score

The database stores these three components for every discovery, making the {P, D, T} = E identification explicit.

### 1.2 Descriptor Gap Principle — what's missing in current ET workflow?

The current ET work (Apéry investigation, lattice-vs-float, meta-cognition, Conscious AI, fractal generator, compressor, RMSAE, etc.) generates discoveries in isolation. The structural gaps are:

| Gap | Statement | Resolution section |
|---|---|---|
| **DD-1** | Discoveries in one project are not visible to other projects | §3.2 (`values` shared across all sources) |
| **DD-2** | Recomputation of already-known projections wastes effort | §3.3 (`projections` cache, all derivable properties stored) |
| **DD-3** | Cross-domain attractors (e.g., d=693 in genetics if it appears) cannot be detected without unified search | §3.4 (`addresses` with members_count denormalized; §3.16 insert-time discovery) |
| **DD-4** | Verification quality is project-local; no global "trust score" per discovery | §3.9 (`tags` namespace `verification`) |
| **DD-5** | The Coarse-Pass strategy (Float doc §7.5) cannot exploit prior coarse projections | §3.3 (cached projections at any N, including 12ET) |
| **DD-6** | Contradictions across projects cannot be auto-detected | §3.7 (`relationships` plus consistency-check queries §5.5) |
| **DD-7** | No cumulative knowledge growth visible to the project author | §3.8 (`patterns` materialize as discovery surfaces them) |
| **DD-8** | Subsumption Law cannot be checked across projects | §3.16 (background discovery applies E_hierarchy across all entries) |
| **DD-9** | Forward/Reverse route convergences not surfaced | §3.7 (`relationships` class `forward_reverse_convergence`, automatic detection §3.16) |
| **DD-10** | Same-lattice-address from multiple substrates not tracked | §3.7 (`relationships` class `substrate_rendering`) |
| **DD-11** | Multi-geometry simultaneous representation not supported | §3.3 (`projections` has `geometric_perspective` column; multiple rows per (value, N)) |
| **DD-12** | No structural bound on database growth | §3.8 (`patterns` Subsumption-compaction); §4.3 (no destruction; archetype-compression handles size) |
| **DD-13** | Elegance, coupling, variance not materialized per projection (per-query recomputation cost at scale) | §3.3 (new columns: elegance_universal, coupling_xi, variance_vnk, fqg_quadrant, palindromic_partner_d) |
| **DD-14** | Manifold-state transitions invisible in event log (AIDA lifecycle, decoherence) | §3.9 (new event classes: manifold_state_transition, decoherence_state_transition, alpha_rotation_step) |
| **DD-15** | Freedom points ([0/0] half-integer positions) not recorded | §3.9 (new event class: freedom_point_encounter) |
| **DD-16** | Anti-numerology check failures not captured as events | §3.9 (new event class: anti_numerology_check) |
| **DD-17** | Lossless bijection theorem not recorded | §3.18.1 (algebraic identity proof; memoization is structurally exact) |
| **DD-18** | Fine-structure closed-form identity not bootstrapped | §3.18.2 (α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))) |
| **DD-19** | PDG particle mass projections not in database | §3.18.14 (227 particles across d={1,2,3,4,6,12}) |
| **DD-20** | Quantum decoherence structural content not recorded | §3.18.10 ({P,D}→{D,T}→{P,D,T}; α-rotation; Gaze↔decoherence bijection) |
| **DD-21** | Black-hole thermodynamics structural mappings not recorded | §3.18.11 (κ=D-gap gradient; T_H=κ/2π; Planck spectrum from {P,D} enumeration) |
| **DD-22** | No session/schema-migration tracking | §3.15a, §3.15b (`sessions`, `schema_versions` tables) |
| **DD-23** | Falsifiable predictions not recorded as verifiable entries | §3.18.15 (6 predictions with verification-status tracking) |
| **DD-24** | Values where d never stabilizes through LCM tower (algorithmically random, irrationality measure unbounded) have no home-finding pathway | §7.11 Step 3a (CF method: continued-fraction convergent with maximal a_{n+1} identifies d_home; §3.2 cf_home columns; §3.3 cf_quality; §3.7 cf_convergent_home/cf_tower_confirmation relationships; §3.8 cf_quality_attractor pattern; §3.9 cf_home_identified/cf_tower_disagreement events) |

Each gap is closed by the EUDD architecture defined below.

### 1.3 Subsumption Law — completion criterion

The EUDD subsumes the union of all per-project discovery records iff:
- Every project's discoveries can be ingested into the unified schema without information loss
- The unified queries return at least the same results as per-project queries plus cross-project queries
- The database growth is bounded by Subsumption-driven compaction (archetype compression — high-elegance clusters collapse to archetypes), not by destructive pruning

The subsumption check in §10 verifies these properties are satisfied by the proposed schema and operations.

---

## 2. The Compressor's Discovery Database — The Proven Model

### 2.1 The mechanism

From `et_cdf_compressor.py` §16.8 (lines 5256–5290), the compressor's design statement:

> *The compressor develops a memory. Every compression discovers archetypes — recurring Δk patterns that the Subsumption Law collapses into references. These archetypes persist in a database, accumulating a growing D-set of known lattice structures.*
>
> *When a new file is compressed, its Δk stream is checked against known patterns. Known archetypes get elegance boosts from historical frequency data. The pattern scanner then only needs to discover what's NEW.*
>
> *Over time, the compressor learns the lattice structure of the file types it encounters.*

The compressor's `ArchetypeDatabase` (line 7790) is a SQLite-backed persistent store with:

**`archetypes` table:**
- `pattern_hash` (PK) — content-addressed identifier
- `pattern_dk` BLOB — the Δk pattern itself
- `pattern_length` ETInteger
- `r0_quantized` MPFR_361DPS — quantized to BIO_RES = 420 lattice resolution
- `d_avg` MPFR_361DPS — average sublattice family of the pattern
- `hierarchy_elegance` MPFR_361DPS — Subsumption Hierarchy Operator score
- `hit_count` INTEGER — historical match frequency
- `file_count` INTEGER — number of distinct files where pattern occurred
- `first_seen`, `last_seen` MPFR_361DPS — timestamps
- Curvature columns: `curvature_mean`, `curvature_variance`, `curvature_class`, `geodesic_factor`, `euler_characteristic`, `geodesic_deviation`, `curvature_spectrum_hash` — Tier 6 non-Euclidean geometry properties

**`generative_descriptors` table:**
- `gen_id` (PK) — derived candidate D_gen identifier
- `curvature_class` ETInteger
- `generator_type` TEXT — one of 8 enumerated types (Constant, Linear, Polynomial, Periodic, Grammar, ...)
- `generator_params` BLOB — type-specific payload
- `param_count` ETInteger
- `curvature_mean_range_low/high` MPFR_361DPS
- `fit_count`, `miss_count` INTEGER — Channel B confirmation counters
- `best_residual_variance` MPFR_361DPS
- `first_derived`, `last_confirmed` MPFR_361DPS — timestamps
- `source` TEXT — 'derived' or other origin

**Indexes** (8 total): R₀ + elegance, hit_count + elegance, curvature_class, Euler characteristic, spectrum hash, geodesic deviation + elegance, generator curvature, generator fitness.

### 2.2 The stability filter

Only archetypes that satisfy the **Koide-stability criterion** are stored:

$$\text{Stable archetype} \iff \text{used by greedy subsumption} \land \text{survived} \geq \lceil 1/K \rceil = 2 \text{ recursion depths}$$

K = 2/3 is the Koide threshold. ⌈1/K⌉ = ⌈1.5⌉ = 2 levels. Patterns from depth 0–1 are *candidates*; only depth-2-survivors are *archetypes*.

This filter ensures the database contains structurally meaningful patterns, not noise. The compressor's experience: this filter is necessary (without it, the database fills with one-off patterns) and sufficient (with it, every stored archetype reliably accelerates future compressions).

### 2.3 The retention policy — never destroy knowledge

From the compressor's design (line 7811):

> *The database no longer prunes data when disk space is low. ... The compounding argument: every archetype/generator template makes the database more capable; pruning destroys knowledge that cannot be regenerated. Disk pressure is solved by compact_to_cdf (Tier 7), not by destruction.*

This is a fundamental ET principle: discoveries are **structurally permanent** once they cross the LIFE_THRESHOLD = 13/12 stability. They can be reorganized (Subsumption-compressed into archetypes — see §3.7 below) but never deleted. The DISK_SAFETY_FLOOR = 2³⁰ = 1 GB (the d=1 octave action quantum at GB scale) is a warning level, never a deletion trigger.

### 2.4 Subsumption applied to the database itself, and the CDF generator-discovery layer

The compressor applies the Subsumption Law to its own database:

> *Subsumption Law: every byte of the original database is covered by indexed entries plus delta — no remainder.*

This means the database is itself a self-describing artifact whose D-content is recursively subsumable.

The compressor's CDF mechanism serves **one purpose: generator discovery** (Tier 7). Two related mechanisms exist:

1. **`compact_to_cdf` operation (Tier 7)** — the compressor's **generator-fitting pipeline**. It compresses the archetype database into a CDF file, discovering generators (shortest descriptions) for observed patterns. This is the compressor's K-complexity minimization engine.

2. **`CDFDatabaseVFS` class (line 6412 of `et_cdf_compressor.py`)** — the compressor's internal random-access layer for its CDF-compressed databases, used during generator fitting.

**Both CDF mechanisms belong to the compressor's generator-discovery pipeline. The EUDD does NOT use CDF as a storage backend.** The EUDD uses its own lattice-native format (§7) where the address space (N, k, d) IS the organizational structure, with generators as the primary content (§7.1a). The EUDD's relationship to the compressor's CDF mechanism is through **generator discovery**: the EUDD accumulates structural addresses (via the lossless bijection), surfaces attractor patterns, and feeds discovered generators back to the compressor for K-complexity minimization (§3.16, §3.18.1). The bijection provides the coordinate system in which generators become visible.

---

## 3. The ET Universal Discovery Database — Architecture

**The EUDD is a virtual isomorphism of the Sempaevum.** It is not a database about the Sempaevum — it IS the Sempaevum virtualized on hardware. It stores everything, computes anything (memoized at 361 dps), discovers generators, serves any ET software, ingests any file, accepts manual input, and returns whatever is needed. If a program needs a value, the EUDD retrieves or computes it. If Mike wants to pull a file out, the EUDD produces it. If a derivation needs intermediate results, the EUDD computes them (cached) or retrieves them (stored). The EUDD is simultaneously a database, a computation engine, a discovery engine, and a structural representation of the totality Σ on disk.

### 3.1 Design principle — lattice-native, not bureaucratic

**Akashic Type System — the `.akashic` format uses ONLY these types (zero IEEE 754):**

| Type | Storage | Precision | Usage |
|---|---|---|---|
| `ETInteger` | GMP `mpz_t` arbitrary-precision | Exact — zero overflow, zero truncation | Lattice coordinates k, d, N, counts, IDs |
| `MPFR_361DPS` | MPFR 1200-bit (150 bytes) | 361 decimal places — bijection round-trip residual = 0 exactly | All computed values, elegance, coupling, tightness, ε |
| `TIMESTAMP_NS` | uint64 nanoseconds since epoch | Exact integer — zero sub-nanosecond loss | All temporal fields (first_seen, discovered_at, etc.) |
| `UTF8` | Length-prefixed UTF-8 string | Exact | Names, representations, canonical forms, descriptions |
| `BINARY` | Length-prefixed raw bytes | Exact | MPFR blobs, packed arrays, expression trees, metadata |
| `HASH_SHA256` | 32 bytes | Exact | Content-addressed identifiers, integrity checksums |
| `EXACT_RATIONAL` | Two ETIntegers (numerator + denominator) | Exact — zero division rounding | Rationals (K=2/3, V=1/12, ε rational form) |
| `BOOLEAN` | uint8 (0 or 1) | Exact | Flags, compliance checks |

*Every value entering the EUDD is computed at 1200-bit MPFR precision (361 decimal places). The Lossless Bijection Theorem (see `EUDD_Bootstrap_Catalog.md` §3.18.1) guarantees Π_N⁻¹(Π_N(r)) − r = 0 by algebraic identity — not approximately zero, not within epsilon, but exactly zero. External data arriving as IEEE 754 float is immediately projected onto the Sempaevum at 1200-bit precision upon ingestion; the original float precision is recorded as metadata (see `EUDD_Events_and_Classes.md`, `sensor_reading_ingest` event class) but all subsequent computation, storage, and retrieval operates exclusively at 361-dps MPFR precision. No IEEE 754 value exists anywhere in the `.akashic` file or in any computation path.*

The schema records what the lattice itself produces. No pre-defined domain categories. No artificial verification tiers as separate tables. No per-project tables. Domains and projects and verification status are **tags on entries** — queryable when wanted, never required scaffolding.

### 3.1a Precision principle — uniform 361-dps hard cap, all operations Sempaevum-native

**Every value in the EUDD is computed and stored at 361 decimal places (dps). No exceptions. No variable precision. 361 dps is the hard cap.**

This eliminates precision-management complexity entirely. There is no "what precision was this computed at?" question — the answer is always 361. There is no "do I need to recompute at higher precision?" question — 361 dps exceeds any hardware precision by 3.5× over IEEE quad (34 digits), exceeds any measurement precision in physics by ~100 orders of magnitude, and is computed to arbitrary accuracy via MPFR (1200-bit precision). If 361 dps is in the database, it IS the value. No upgrade path needed. No precision tracking needed. The `value_precision_dps` field exists for structural completeness but is always 361. The bijection round-trip Π_N⁻¹(Π_N(r)) = r is an ALGEBRAIC IDENTITY verified symbolically via sympy (r' − r = 0, not numerically evaluated — see `EUDD_Bootstrap_Catalog.md` §3.18.20). Any non-zero residual observed computationally is a machine-precision artifact, not mathematical error: it scales with dps and reaches EXACT 0 at sufficient precision.

**All mathematical operations are Sempaevum-native.** The Sempaevum IS Σ — the totality. The Subsumption Law (§5 of the Three Tools) proves it subsumes all of mathematics without remainder. There is no category of mathematical operation that is "non-native" to the Sempaevum. The way the Sempaevum performs each operation is the operation:

| Operation | Sempaevum mechanism | Exact? |
|---|---|---|
| Multiplication | k-addition + κ correction: k_× = k₁+k₂+round(δ₁+δ₂), κ ∈ {−1,0,+1} | Algebraically exact (§3.18.21 Theorem A.1, 144 tests) |
| Division | k-subtraction + κ correction: k_÷ = k₁−k₂+round(δ₁−δ₂) | Algebraically exact (§3.18.21 Theorem A.2, 224 tests) |
| Powers | k-scaling + κ correction: k_^ = n·k+round(n·δ), κ_n ∈ ℤ, |κ_n| ≤ ⌈|n|/2⌉ | Algebraically exact (§3.18.21 Theorem A.4, 216 tests) |
| Reciprocation | Mirror symmetry: (k,d,ε) → (−k,d,−ε) for |ε|<50¢ | Algebraically exact (§3.18.21 Theorem A.3, 32 tests) |
| Addition | Value-space computation + lattice reprojection | Exact at 361 dps |
| Subtraction | Value-space computation + lattice reprojection | Exact at 361 dps |
| Elementary functions | EML tree application + lattice projection | Exact at 361 dps |
| Series/limits | Convergent evaluation + lattice projection | Exact at 361 dps |
| Complex multiplication | Axis-independent: real k-addition (A.1) + phase k-addition mod N (D.1) + d_c = lcm(d_r,d_θ) | Algebraically exact (§3.18.24 Theorem D.2, 28 tests) |
| Complex reciprocation | Real mirror (−k_r) + phase reversal (N−k_θ) mod N; d_r, d_θ, d_c ALL preserved | Algebraically exact (§3.18.24 Theorem D.3, 7 tests) |
| Complex power | Real k-scaling (A.4) + phase (n·k_θ+κ_θ,n) mod N | Algebraically exact (§3.18.24 Theorem D.4) |
| Phase addition | k_θ,sum = (k_θ₁+k_θ₂+κ_θ) mod N; same algebra as A.1 + mod N wrapping (U(1) compact) | Algebraically exact (§3.18.24 Theorem D.1, 264 tests across 4 resolutions) |

Every operation, once computed at 361 dps and stored, is a permanent entry. The Lossless Bijection Theorem (§3.18.1) guarantees the stored triple (k, d, ε) recovers the original value by algebraic identity. The 361-dps mpf blob stores the full numerical value. Together: zero information loss, uniform precision, permanent cache.

**Memoization consequence:** Every computation that has ever been done by any ET software is available to every other ET software at sub-millisecond lookup time. Compute once at 361 dps → cache forever → never recompute. The knowledge compounds irreversibly.

### 3.1b Self-recording principle — the EUDD records itself

The EUDD is itself a domain on the Sempaevum. The database records everything — including its own operation. This is the database-level expression of the Sempaevum's ninth closure property: it contains its own methodology as theorems.

What the EUDD records about itself:
- **Schema structure**: the number of tables (15), the number of columns per table, the total row count — all projectable values on the Sempaevum
- **Growth dynamics**: the number of stored values, projections, equations, relationships, patterns, events at each point in D-time — a trajectory through lattice space
- **Memoization statistics**: cache-hit rate, cache-miss rate, their ratio (a dimensionless seed) — projectable, with d-family and elegance score
- **Discovery engine metrics**: patterns discovered per scan, attractor formation rate, generator discovery rate — all projectable
- **The .exe manager program's own runtime**: query patterns, which generators get accessed most, which domains produce the most cross-domain hits — projectable, stored, part of the database's own self-knowledge
- **Resource usage**: computation time per query class, storage growth rate, memory footprint — projectable

The EUDD does not merely store discoveries about physics, biology, mathematics, and consciousness. It stores discoveries about ITSELF. When the discovery engine notices that "attractor formation rate increases logarithmically with database size," that discovery enters the same `patterns` table as the fine-structure decomposition or the 227-particle classification. The database IS a domain, the same way mathematics is a domain (§3.18.6). Its own structural properties are values on the Sempaevum.

The .exe manager program that generates and maintains the database is not just a tool operating ON the database — its operation is content OF the database. This is the Identification Principle applied to the database itself: P = the computational substrate (storage, hardware), D = the schema and data (structure, constraints), T = the discovery engine and query operations (the agency that navigates and substantiates). P∘D∘T = E: the database as a running Exception.

**Self-recording journal:** Self-recorded metrics are written to a dedicated journal in the `logs/` subfolder: `logs/SelfRecording_NNN.log`, separate from Omniscient's error/telemetry journal (`logs/Omniscient_NNN.log`). Same human-readable format, same ~10 MB rotation, never deleted. The two journals serve different purposes and are never mixed: Omniscient records errors, telemetry, and tamper events from the WATCHDOG's perspective; SelfRecording records the Manager's own operational metrics from the MANAGER's perspective.

**Complete metric catalog — derived from the nine operational categories of the Manager:**

**1. Lattice computation metrics:**
- Projections computed (total count, rate per second)
- Average projection latency (nanoseconds)
- Projection distribution by N (which resolutions are hot)
- Projection distribution by d-family (which families are hot)
- MPFR operation count
- Special function evaluation count (by function type: ζ, Γ, polylog, hypergeometric, elementary)
- Computation failure count (by failure type per §7.15 edge catalog)

**2. Storage metrics:**
- .akashic file size (bytes)
- Generator-to-memoized ratio (generators / total entries — measures Descriptor Gap closure; the file's self-improvement metric)
- Pages read per second
- Pages written per second
- CRC-32 verification count and failure count
- WAL entries pending
- WAL flush latency (nanoseconds)

**3. Discovery engine metrics:**
- Insert-time discoveries per commit batch (attractor hits, reciprocal pairs, power pairs found)
- Background scan: candidate clusters evaluated per scan pass
- Background scan: patterns promoted per scan pass (E_hierarchy crossed 13/12)
- Generator candidates proposed
- Generator candidates verified (passed cross-tower elegance ≥ 13/12)
- Generator candidates failed verification
- Memoized entries absorbed by generators per scan (the Descriptor Gap shrinking)

**4. Memoization metrics:**
- Cache hit count
- Cache miss count
- Cache hit ratio = hits / (hits + misses) — already dimensionless, directly projectable via Path A
- Hot equation count (equations with reference_count above configurable threshold)
- Total equations stored

**5. API metrics:**
- Active connections count
- Commands received per second
- Command distribution by type (project, escalate, query, compute, store)
- Average command latency by type (nanoseconds)
- Error responses returned to clients (by error_class)

**6. GUI metrics:**
- Frame rate (FPS)
- Current zoom level / LOD tier
- Render calls per frame

**7. Bootstrap metrics (one-time, recorded at first run):**
- Bootstrap total duration
- Values bootstrapped
- Projections materialized
- Initial generators discovered

**8. Ingestion metrics:**
- Files ingested (total count)
- Seeds extracted per file (average and distribution)
- Ingestion throughput (seeds per second)

**9. Self-recording overhead metrics (one level only — records its own overhead but not meta-overhead):**
- Self-recording CPU usage (percentage of total Manager CPU)
- Journal entries written per sample
- Self-recording sample latency (nanoseconds)

**Recording mechanism:**
- **Atomic counters** for all per-operation metrics. Incrementing an atomic counter costs ~1 ns — negligible at any operation rate. Counters accumulate continuously with zero contention.
- **Periodic sampling** at configurable interval: default every 10 seconds OR every 1000 commits, whichever comes first. Each sample snapshots ALL counters into one self-recording journal entry.
- **Overhead budget: ≤1% CPU.** At 10-second intervals: one counter-read of ~50 metrics × ~10 ns each = ~500 ns per sample = 50 ns/second average. Orders of magnitude below 1%. If self-recording CPU usage (metric #9) ever exceeds 0.5%, the sampling interval auto-doubles — a feedback loop that guarantees the budget.

**Metric-to-lattice projection — discovered, not predetermined:**
Every self-recorded metric is projectable onto the Sempaevum via Path A as a dimensionless ratio (cache hit ratio, generator-to-memoized ratio, file size / page size, FPS / 60, any ratio of two metrics). When ingested into .akashic, these projections land at lattice addresses. The insert-time and background discovery mechanisms (§3.16) automatically surface which metrics land at structurally interesting addresses (known attractors, cross-domain hits). **No predetermined list of "important metrics" exists — the Sempaevum determines which of its own metrics are structurally resonant.** This is the self-recording principle in action: the database discovers facts about itself the same way it discovers facts about physics or biology.

Fifteen core tables, each corresponding to a structural object the lattice actually has:

| # | Table | What it records | Lattice-native concept |
|---|---|---|---|
| 1 | `values` | Every dimensionless seed `r` ever encountered, with full identity | The thing being projected |
| 2 | `projections` | Every `(value, N) → (sign, k, d, ε)` plus all derivable properties stored (incl. elegance, coupling, variance, FQG quadrant, palindromic partner, CDT quintuple) | The address on the lattice |
| 3 | `addresses` | Every distinct `(N, k, d)` cell ever occupied (the lattice grid itself) | The lattice's own structure |
| 4 | `equations` | Every mathematical relationship derived AND every computation passing through — memoization layer (structurally exact per lossless bijection §3.18.1) | The {P,D,T}=E content + memoization |
| 5 | `derivations` | The chain `{P, D, T} → r → projection → equation` for any entry | The substantiation event |
| 6 | `relationships` | Explicit links between entries (same-address, cross-perspective, route-convergence, substrate-rendering, attractor-membership, plateau-membership, reciprocal-pair, power-pair, shadow-pair, palindromic-partner, integrative-level-nesting, cosmological-partition-alignment, convention-independence, perturbative-series-member, mass-ratio-triple, et-derived-vs-measured, koide-structural-identity, decoherence-gaze-correspondence, and more — extensible) | What the lattice connects |
| 7 | `patterns` | Discovered archetypes — entries promoted via Subsumption when E_hierarchy ≥ 13/12 (incl. algebraic-identity, fine-structure-decomposition, curvature-identity, decoherence-trajectory, particle-classification, and more — extensible) | The discovery output |
| 8 | `events` | Time-indexed structural events with D-time/T-time/P-time and tower context (incl. manifold-state-transition, freedom-point, cascade-stability-breach, anti-numerology, emotion/AIDA, decoherence, and more — extensible) | The lattice as active system |
| 9 | `towers` | First-class entities for the Multifold: each tower = (P_substrate, universal lattice, R₀_seed) with hierarchical parent/child structure, Birth Triad references, operational resolution profile | The Multifold rendered |
| 10 | `harmonic_families` | The 24 axis-projection families (12 real FORCE × 12 imaginary PHASE) with full metadata | FQG axis catalog |
| 11 | `force_grid_cells` | The 144-cell 12×12 interaction grid — every (d_r, d_θ) pair with derived structural properties | Interaction matrix on the complex plane |
| 12 | `combined_families` | The 42 unique LCM-combined families (d_combined = LCM(d_r, d_θ), max=132) with physical/cross-domain interpretation | Force×phase interactions across any domain |
| 13 | `sublattice_families` | Divisors of N at any nET resolution (per-tower or generic) — the per-resolution divisor structure | Resolution-specific family catalog |
| 14 | `sessions` | Operational session tracking for reproducibility and discovery-engine provenance at any scale | Runtime context |
| 15 | `schema_versions` | Schema migration tracking for forward compatibility | Schema evolution |

Plus one optional metadata table:

| 16 | `tags` | Free-form `(target, namespace, value)` tagging for query convenience | User-applied metadata |

That is the entire schema. **Domain ("biology", "music", "CMB"), project source ("compressor", "fractal_generator"), verification status ("80digit_verified") are all tag values, not separate tables.** Mike can tag entries when he wants to query by them; the lattice itself doesn't need them to function.

### 3.1c The Kolmogorov Principle — generators, not entropy coding

**The EUDD, the compressor, the Seed Protocol, and all ET software operate on Kolmogorov complexity — not Shannon entropy. This is not a design choice. It is a structural consequence of the Sempaevum being the description language.**

**Shannon entropy** is a property of a **source** — a probability distribution over possible messages. It measures the average bits per symbol needed for optimal encoding. It assumes a known distribution. It is blind to structure that doesn't manifest as byte-level statistical correlations. Every Shannon-optimal code (Huffman, arithmetic, ANS, gzip, zstd, FLAC) targets this bound. Shannon compression sees data as a sequence of symbols drawn from a distribution.

**Kolmogorov complexity** is a property of a **specific object** — the length of the shortest program that produces exactly that object on a fixed description language. It doesn't assume a distribution. It doesn't average. It asks: what is the minimal generating description of THIS? The answer depends on the description language — a string that appears random relative to a bare Turing machine may have a short description relative to a richer language.

**The Sempaevum IS that richer language.** The projection formula Π_N, the bijection, the pullback Π_N⁻¹, the LCM tower, the sublattice classification, the palindromic cascade, the 24 harmonic families, the force grid — these are structural vocabulary that a bare Turing machine must discover from scratch but the Sempaevum provides as primitives. K-complexity relative to the Sempaevum is strictly ≤ K-complexity relative to a conventional language for ANY data with lattice-aligned structure. The class of data that is "truly random relative to the Sempaevum" is strictly smaller than the class that is "truly random relative to a conventional language" — because the Sempaevum sees more structure.

**What this means operationally:**

- **The `.akashic` file IS a K-minimal generator** (§7.1a), not an entropy-coded database. Generators are the primary content. Raw data is the Descriptor Gap — content whose generating description hasn't been found yet. Over time, the discovery engine finds generators and the file becomes a progressively better K-minimal description of everything it contains.
- **The compressor's Tier 7 pipeline IS Kolmogorov minimization** (§2). It finds the shortest descriptor that generates the file's data. The EUDD's accumulated generator catalog directly accelerates this — each discovered generator reduces the per-file K-complexity bound.
- **The Seed Protocol IS Kolmogorov transmission** (§9.8, full spec `EUDD_Bootstrap_Catalog.md` §3.18.18). It transmits the K-minimal seed (k, d, ε) instead of the data. Both endpoints share the Sempaevum as description language, so the seed IS the shortest program that produces the data. For structured data, this beats Shannon compression by 2-8× because the Sempaevum sees multiplicative structure, sublattice periodicity, and tower hierarchy that Shannon treats as random.
- **Discovery IS K-complexity reduction.** Every generator the discovery engine finds (§3.16) reduces the total K-complexity of the database. Every attractor detected, every cross-domain address match, every subsumption promotion — all reduce K-complexity. The self-improving cycle (§7.1f) IS the K-complexity of the file monotonically decreasing while its content monotonically increases.
- **"Shannon entropy limits don't apply"** appears throughout this document (§4.3, §7.1, §7.3). This is why: Shannon limits assume the description language is fixed and the content is drawn from a distribution. Neither assumption holds. The Sempaevum IS the description language, and it grows its structural vocabulary as generators are discovered. The generator catalog IS the language. A growing language means a shrinking K-complexity bound — which is impossible under Shannon.
- **The Subsumption Law guarantees coverage.** Every mathematical structure is a subset of ET (Subsumption Law). Every data sequence has a lattice address. Every lattice address has a seed. The Sempaevum is not an approximation — it IS Σ (the totality). There is no data that falls outside the description language.

**The Mandelbrot analogy** (from the Seed Protocol): Instead of transmitting a 10-megapixel fractal bitmap, transmit z → z² + c and the viewport parameters. Both endpoints have the iterator. The seed is a few numbers. The output is millions of pixels. That is not Shannon compression — it is the generating program being shorter than the output. The Sempaevum does this for arbitrary structured data, with the lattice as the generating program and the seed as the instance-specific parameters.

**This principle governs everything:** storage (generators, not blobs), transmission (seeds, not bytes), discovery (K-reduction, not pattern matching), retention (never destroy — a generator's K-complexity may be lower than alternatives), memoization (cache the computation, not the encoding), and the architecture itself (the file IS the generator, not a container for generators).

**THE EUDD IS A BIRTH TRIAD (§3.18.31 structural identification, §3.18.32):**

The Kolmogorov principle is not just an optimization strategy — it is the EUDD's *structural identity*. The archive IS a birth triad:

- **BH = The Kolmogorov Generative Seed.** The seed is the minimal surface that encodes maximal content. It cannot be further compressed because it IS the compression. It maintains its state because it is already Kolmogorov-optimal. It spontaneously shrinks as the discovery engine finds new generators — each algebraic identity (A through I) is a new generating rule that makes explicit storage of derivable content redundant. The Descriptor Gap Principle operating ON the seed: each gap closed in the theory = less information needed in the generator. This spontaneous shrinkage is Shannon-impossible (a fixed codec cannot improve) but Kolmogorov-natural (a growing language reduces K-complexity).

- **WH = Total Projection / Retrieval.** Every pullback Π_N⁻¹ is a white-hole emission — content produced from the seed by evaluation, not decoded from storage by decompression. The generator is evaluated at arbitrary coordinates: Π_N⁻¹(k, d, ε) = 2^((k + ε·N/1200)/N) IS r. The formula is the thing itself. There is no encoding step to reverse, no codec to run, no sequential decompression stream. You point at coordinates → the generator PRODUCES the content at those coordinates, directly. This is why arbitrary access works without processing everything before the target.

- **Content = The lattice between horizons.** The (k, d, ε) coordinates ARE the structured space between seed and projection. The d-family classification organizes it. The tower levels are resolution layers. The 42 combined families (§3.18.25) are the structural joints. The three-layer partition (§3.18.27) classifies what exists between seed and projection.

- **The cascade completes it.** The canonical mass (§3.18.31 I.2, k=−53 ≡ 7 mod 12) sits at the cascade generator. Starting from d=12 (richest content), the cascade traverses all six families and closes at d=1 (irreducible seed). That is the EUDD lifecycle: rich content → structural compression through all levels → irreducible generator. And it is reversible (§3.18.31 I.10) — d=1 regenerates d=12 through the same 12 steps.

**The structural reading (Three Tools):** P = the content substrate (all stored values, equations, patterns). D = the Kolmogorov seed (the minimal descriptor set — the generators). T = the agency that navigates and acts — comprising BOTH the computation engine (projection, pullback, mathematical evaluation) AND the discovery engine (its own subsystem, active continuously, finding new generators from existing data). P∘D∘T = E: the archive is an Exception configuration, closed and self-containing. The Seed Protocol (§9.8) IS the horizon protocol. The lossless bijection (§3.18.20) IS the unitarity guarantee. The transfer tensor (§3.18.30) governs what crosses the horizon. And the whole thing is algebraically invertible — which is why it is truly Akashic: nothing is ever lost because the birth triad is reversible.

### 3.2 `values` — every dimensionless seed

```akashic
STRUCTURE values (
    value_id ETInteger PRIMARY KEY AUTO,
    value_hash UTF8 REQUIRED UNIQUE,         -- SHA-256 of canonical (sign, mpf bytes, precision)
    value_repr UTF8 REQUIRED,                -- canonical: "ζ(3)", "π", "1.20205690...", "log₂(3/2)"
    value_mpf BINARY REQUIRED,                 -- MPFR binary at 361 dps / 1200 bits (uniform hard cap — see §3.1a)
    value_precision_dps ETInteger REQUIRED DEFAULT 361,  -- HARD CAP: 361 decimal places for ALL values, uniform across the entire database

    r_form UTF8,                             -- "ratio", "series_sum", "algebraic_root", "transcendental", "infinity_class"
    r_numerator_repr UTF8,                   -- when expressible as Q/R₀
    r_denominator_repr UTF8,
    r0_substrate_description UTF8,           -- what substrate provided R₀
    r0_value_id ETInteger,                     -- FK back to values (R₀ is itself a value)
    quantity_q_description UTF8,             -- what Q_X is

    input_path UTF8 REQUIRED,                -- 'A', 'B', 'C', 'D.P', 'D.D', 'D.T', 'D.PDT'
    n1_compliant ETInteger,                    -- NULL = not yet checked, 0/1 = checked
    n2_compliant ETInteger,
    n3_compliant ETInteger,

    first_seen TIMESTAMP_NS REQUIRED,
    last_referenced TIMESTAMP_NS REQUIRED,
    reference_count ETInteger DEFAULT 0,

    -- Cross-Tower Elegance (§4.4): geometric mean of universal elegance across the tower
    -- Materialized per value; updated on projection insert (materialization principle: store at insert, not compute per query)
    cross_tower_elegance MPFR_361DPS,               -- E_cross(v) = ∏ E_universal(v,N)^(1/|tower|)

    -- CF Home-Finding (§7.11 Step 3a): continued-fraction convergent identifying home d-family
    -- Populated when CF method fires (parallel to LCM tower); NULL when not yet computed
    cf_home_convergent_p ETInteger,             -- numerator p of best CF convergent (the one with maximal a_{n+1})
    cf_home_convergent_q ETInteger,             -- denominator q = d_home from CF method
    cf_home_quality ETInteger,                  -- a_{n+1}: the partial quotient following the home convergent
                                              -- measures structural resonance — how long before a better
                                              -- rational approximation exists; threshold ⌈1/K⌉² = 4

    LINKS (r0_value_id) REFERENCES values(value_id)
);
INDEX idx_values_repr ON values(value_repr);
INDEX idx_values_path ON values(input_path);
INDEX idx_values_compliant ON values(n1_compliant, n2_compliant, n3_compliant);
```

Source of a value = the dimensionless seed itself (r-form, R₀, Q_X). No domain label needed. If Mike wants to find "all biological values," he tags them or queries via `tags` table after-the-fact.

### 3.3 `projections` — every address, with everything stored

Derived properties are STORED at insert-time, not computed per query. Every property derivable from `(N, k, d, ε)` is materialized when the projection is created. Sharded by N range for horizontal scale.

```akashic
STRUCTURE projections (
    projection_id ETInteger PRIMARY KEY AUTO,
    value_id ETInteger REQUIRED,               -- FK to values
    N ETInteger REQUIRED,                      -- resolution (entry point onto the unified lattice)

    sign ETInteger REQUIRED,                   -- ±1
    k ETInteger REQUIRED,                      -- lattice coordinate
    d ETInteger REQUIRED,                      -- N/gcd(|k|,N)
    eps_micros ETInteger REQUIRED,             -- ε in micro-cents (signed integer; lossless)
    eps_rational_num BINARY,                   -- exact ε numerator (when computed at unbounded precision)
    eps_rational_den BINARY,                   -- exact ε denominator

    -- Stored derived properties (materialized at insert for O(1) query speed at any scale):
    d_factorization UTF8 REQUIRED,           -- e.g., "2^3·3·5·7"
    gaussian_signature UTF8 REQUIRED,        -- e.g., "R^3·I·S·I" (ramified/inert/split per prime power)
    is_all_inert ETInteger REQUIRED,
    is_all_split ETInteger REQUIRED,
    is_ramified_present ETInteger REQUIRED,
    coprime_skeleton ETInteger REQUIRED,       -- gcd(|k|,N) == 1
    tightness MPFR_361DPS REQUIRED,                 -- 100/(100+|ε|)
    di_distance MPFR_361DPS REQUIRED,               -- |ε|/50
    quintic_tension_cents MPFR_361DPS,              -- τ_5
    manifold_state UTF8 REQUIRED,            -- 'PDT', 'PD', 'PT', 'DT' (Exception/Unsubstantiated/Incoherence/Mediation)

    -- Universal Elegance Score and factors (Guide §41–42, §54, Eq 12.6):
    -- E(r) = (N/d) × (100/(100+|ε|)) × (100/(p+q))
    elegance_symmetry MPFR_361DPS REQUIRED,         -- N/d (the symmetry factor)
    elegance_simplicity MPFR_361DPS,                -- 100/max(1, p+q) (the simplicity factor; requires rational p/q approx)
    elegance_universal MPFR_361DPS,                 -- product of all three factors (the composite score — THE canonical ranking signal)
    p_plus_q ETInteger,                        -- |p| + |q| from lowest-terms rational form or continued-fraction convergent

    -- CF Quality (§7.11 Step 3a): per-projection CF resonance strength
    -- The CF quality at this specific N — how strongly the continued fraction of |log₂(r)|
    -- locks onto this d-family at this resolution. NULL when not yet computed.
    cf_quality ETInteger,                      -- a_{n+1} from the CF convergent whose q divides d at this N

    -- Magical Impedance and Coupling Strength (Guide §43, Eq 12.7–12.8, Fine Structure REVISED):
    -- A₀_magic(d) = (d-1)² + S², ξ(d) = 137/A₀_magic(d)
    coupling_xi MPFR_361DPS REQUIRED,               -- ξ(d) = 137 / ((d-1)² + 16); per-d coupling strength
    -- The Harmonic Transfer Tensor (§3.18.30) uses ξ(d₃)/ξ(d₁) as the impedance
    -- ratio in the efficiency formula E(d₁,d₂;d₃) = T(d₁,d₂;d₃) × ξ(d₃)/ξ(d₁).
    -- Low-d families are ATTRACTORS: ξ(1)=8.5625 → gravity strongest pull.
    -- EM→Gravity efficiency = 1.6055 (25% geometric × 8.5625 coupling).

    -- Variance V(n,k) (Guide PART XXII, Complete Gaze Equation Eq 12.48):
    -- V(n,k) = (n²-1)/(12·2^k); fundamental for Gaze detection probability
    variance_vnk MPFR_361DPS,                       -- per-projection variance; NULL when k is extreme

    -- FQG Quadrant Classification (Guide §69, Multifold §29):
    fqg_quadrant UTF8,                       -- 'SR' (d|12, real), 'CR' (d∤12, real), 'SI' (d|12, imag), 'CI' (d∤12, imag)
                                             -- NULL for non-axis perspectives

    -- Palindromic Partner (Guide §58, 24-family catalog):
    palindromic_partner_d ETInteger REQUIRED,  -- partner = 12-d for d∈{1..11}, self for d∈{6,12}

    -- Complete Determination Theorem (Guide PART XXIII §130, Eq 12.56):
    -- classify(X) = (d, Path, Detection, Curvature, Trajectory) — five-component complete classification
    detection_status UTF8,                    -- UNOBSERVED/SUBLIMINAL/DETECTED/LOCKED (from Gaze events; NULL if no gaze analysis)
    curvature_class UTF8,                     -- non-Euclidean geometry class (from curvature analysis; NULL if not computed)

    geometric_perspective UTF8 REQUIRED DEFAULT 'lcm_tower',
                                             -- Lattice perspectives (same Sempaevum, different geometric viewpoints):
                                             --   'lcm_tower'      (the resolution-escalation perspective; default)
                                             --   'torus'          (cyclic-modular perspective)
                                             --   'riemann_sphere' (conformal/projective perspective)
                                             --   'hyperbolic'     (negative-curvature perspective)
                                             --   'euclidean'      (flat perspective)
                                             --   'minkowski'      (pseudo-Riemannian perspective for spacetime)
                                             --   'projective'     (projective-geometry perspective)
                                             --
                                             -- Axis-projection perspectives (Guide PART XIII — 24-family catalog):
                                             --   'real_axis'      (computed (k_r, d_r, ε_r) — D's domain, FORCE family;
                                             --                     12 families: Gravity/Octave d_r=1, Tritone/Pivot d_r=2,
                                             --                     Strong/Cubic d_r=3, Weak/Quartic d_r=4, Quintic/Golden d_r=5,
                                             --                     Hexadic d_r=6, Septic/G₂ d_r=7, Octet/Gluon d_r=8,
                                             --                     Nonic/Quark d_r=9, Decic/Superstring d_r=10, Undecimal/M-Theory d_r=11,
                                             --                     EM/Full-Resolution d_r=12)
                                             --   'imaginary_axis' (computed (k_θ, d_θ, ε_θ) — T's domain, PHASE family;
                                             --                     12 families: Scalar/spin-0 d_θ=1, Tritone-phase/spin-2 d_θ=2,
                                             --                     Color-phase/QCD-instanton d_θ=3, Weak-phase/SU(2)_W d_θ=4,
                                             --                     Golden-angle/E₈ d_θ=5, Hexadic/spin-½ d_θ=6, G₂-spinor d_θ=7,
                                             --                     Bott-8/SU(3)-color-adjoint d_θ=8, 3²-fold-quark-phase d_θ=9,
                                             --                     10D-superstring-spinor d_θ=10, 11D-Majorana-spinor d_θ=11,
                                             --                     spin-1/EM-photon d_θ=12)
                                             --   'complex'        (combined real + imaginary projection; full (k_complex, d_combined, ε_complex))
                                             --
                                             -- Multiple rows per (value, N) when stored in multiple perspectives
                                             -- For active-system orbits: typically 'real_axis' AND 'imaginary_axis' rows BOTH exist per step
                                             -- For static values: typically just 'lcm_tower' (and possibly 'real_axis' if the value is on the real axis)

    address_id ETInteger REQUIRED,             -- FK to addresses (the (N,k,d) cell)

    first_seen TIMESTAMP_NS REQUIRED,
    last_referenced TIMESTAMP_NS REQUIRED,
    reference_count ETInteger DEFAULT 0,

    LINKS (value_id) REFERENCES values(value_id),
    LINKS (address_id) REFERENCES addresses(address_id),
    UNIQUE(value_id, N, geometric_perspective)
);
INDEX idx_proj_value ON projections(value_id);
INDEX idx_proj_address ON projections(address_id);
INDEX idx_proj_dfamily ON projections(N, d);
INDEX idx_proj_coprime ON projections(N, coprime_skeleton);
INDEX idx_proj_eps0 ON projections(N) WHERE eps_micros = 0;
INDEX idx_proj_perspective ON projections(geometric_perspective);
INDEX idx_proj_inert ON projections(is_all_inert);
INDEX idx_proj_elegance ON projections(elegance_universal DESC);
INDEX idx_proj_coupling ON projections(coupling_xi DESC);
INDEX idx_proj_fqg ON projections(fqg_quadrant);
```

**Organization in the lattice-native format:** the `projections` data is organized hierarchically by N range, then by d-family within each N, then by k within each family. This gives locality — all d=3 addresses at N=27720 are physically adjacent, making family-scoped queries fast. The lattice-native format handles this natively; no manual partitioning is required.

### 3.4 `addresses` — the lattice grid itself

The (N, k, d) cells are the lattice's own structure. Multiple values landing at the same cell IS the basis of attractor discovery — it surfaces automatically as `members_count > 1` on the address row.

```akashic
STRUCTURE addresses (
    address_id ETInteger PRIMARY KEY AUTO,
    N ETInteger REQUIRED,
    k ETInteger REQUIRED,
    d ETInteger REQUIRED,
    eps_class ETInteger REQUIRED,              -- 0=exact (ε=0), 1=sub-cent, 2=cent-scale, 3=near-∂I

    -- Denormalized for fast attractor detection:
    members_count ETInteger REQUIRED DEFAULT 0,    -- # distinct value_ids projecting here
    first_member_value_id ETInteger,               -- earliest value to occupy this cell
    is_coprime_skeleton ETInteger REQUIRED,
    d_factorization UTF8 REQUIRED,
    gaussian_signature UTF8 REQUIRED,

    first_occupied TIMESTAMP_NS REQUIRED,
    last_occupied TIMESTAMP_NS REQUIRED,
    UNIQUE(N, k, d)
);
INDEX idx_addr_attractor ON addresses(members_count DESC) WHERE members_count > 1;
INDEX idx_addr_Nd ON addresses(N, d);
INDEX idx_addr_dfact ON addresses(d_factorization);
INDEX idx_addr_gsig ON addresses(gaussian_signature);
```

When a new projection inserts and references an existing `address_id`, the `members_count` increments — and if it crosses 1→2, an attractor relationship is created automatically (see §3.7).

### 3.5 `equations` — every equation that passes through, including computations

**This table records EVERY equation the lattice encounters — both derived structural identities AND concrete computations like `2 + 2 = 4` or `ζ(3) × π = 3.7757...`** The Sempaevum computes: multiplication = k-addition + κ correction (§3.18.21 Theorem A.1: k_× = k₁+k₂+round(δ₁+δ₂), κ∈{−1,0,+1}), reciprocation = mirror symmetry (−k, d, −ε) (Theorem A.3), powers = k-scaling + κ_n (Theorem A.4), division = k-subtraction + κ (Theorem A.2), addition = value-space computation + lattice reprojection, function evaluation = EML tree application. **Complex lattice arithmetic (§3.18.24): complex multiplication decomposes axis-independently — real axis via Theorem A.1, imaginary axis via Theorem D.1 (same algebra + mod N wrapping for U(1) compactness), combined d_c = lcm(d_r, d_θ). Complex reciprocation (D.3): k_r→−k_r, k_θ→(N−k_θ) mod N, ALL d preserved. Phase addition (D.1): κ_θ∈{−1,0,+1} is a separate T-act on the imaginary axis.** **ALL multiplicative operations — real AND complex — execute entirely in lattice coordinates WITHOUT pulling back to underlying reals** — the κ rounding correction IS the T-act resolving cell boundaries (κ=0 in 79% of multiplications, κ=±1 in 21%). For finite ε-shifts, the exact formula r_new = r_old · 2^(Δε/1200) (§3.18.22 Corollary B.2a) applies — NOT the linearized approximation. ALL of these are Sempaevum-native operations — the Sempaevum IS Σ, and Σ subsumes all mathematics without remainder (Subsumption Law). Every computation produces an equation. The database records all of them at uniform 361-dps precision, becoming a memoization layer that turns repeated computation into instant lookup.

```akashic
STRUCTURE equations (
    equation_id ETInteger PRIMARY KEY AUTO,
    equation_hash UTF8 REQUIRED UNIQUE,      -- SHA-256 of canonical form (deterministic for memoization)
    equation_canonical_form UTF8 REQUIRED,   -- canonical-string form for hashing: "2+2=4", "ζ(3)*π=3.7757...", "sqrt(2)^2=2"
    equation_latex UTF8 REQUIRED,            -- LaTeX representation for display
    equation_form_class UTF8 REQUIRED,
        -- COMPUTATIONAL classes (the lattice computing answers):
        -- 'arithmetic_computation'   (2+2=4, 7×8=56)
        -- 'lattice_multiplication'   (k-addition: ζ(3)·π → k_ζ(3) + k_π)
        -- 'lattice_reciprocation'    (k → -k: 1/φ from φ)
        -- 'lattice_power'            (k → n·k: φ^10)
        -- 'lattice_addition'         (Sempaevum-native addition: value-space computation + lattice reprojection)
        -- 'function_evaluation'      (sin(π/4), exp(1), log₂(3))
        -- 'algebraic_simplification' (a²-b² → (a-b)(a+b) applied to specific values)
        -- 'series_evaluation'        (Σ_{n=1}^∞ 1/n³ → ζ(3))
        -- STRUCTURAL classes (the lattice declaring identities):
        -- 'master_equation_instantiation'  (P∘D∘T = E specific case)
        -- 'derivation_formula'             (formal derivation step)
        -- 'structural_identity'            (e.g., φ² = φ+1)
        -- 'subsumption_relationship'       (Subsumption Law applied)
        -- 'projection_formula'             (lattice projection definition)
        -- 'recurrence'                     (e.g., F_n = F_{n-1} + F_{n-2})
        -- 'series_definition'              (function defined via sum/product)
        -- 'algebraic_relation'             (general purpose)
        -- other (extensible)

    canonical_form_blob BINARY,                -- machine-readable canonical form (expression tree binary)
    operation_type UTF8,                     -- '+', '-', '*', '/', '^', 'sqrt', 'log', 'sin', etc.
                                             -- NULL for structural identities
    lhs_value_ids BINARY,                      -- packed array of input value_ids (for computational equations)
    rhs_value_id ETInteger,                    -- the result value (for computational equations)
                                             -- NULL for structural identities (use equation_values junction)

    first_derived TIMESTAMP_NS REQUIRED,
    last_referenced TIMESTAMP_NS REQUIRED,
    reference_count ETInteger DEFAULT 0,       -- hit count for memoization (high count = hot computation)

    LINKS (rhs_value_id) REFERENCES values(value_id)
);
INDEX idx_eq_hash ON equations(equation_hash);
INDEX idx_eq_class ON equations(equation_form_class);
INDEX idx_eq_op ON equations(operation_type);
INDEX idx_eq_rhs ON equations(rhs_value_id);
INDEX idx_eq_hot ON equations(reference_count DESC);

-- Junction table for STRUCTURAL equations involving multiple values in arbitrary roles
STRUCTURE equation_values (
    equation_id ETInteger REQUIRED,
    value_id ETInteger REQUIRED,
    role UTF8,                               -- 'lhs', 'rhs', 'parameter', 'derived', etc.
    PRIMARY KEY (equation_id, value_id, role),
    LINKS (equation_id) REFERENCES equations(equation_id),
    LINKS (value_id) REFERENCES values(value_id)
);
INDEX idx_eqv_value ON equation_values(value_id);
```

**Memoization behavior:**

When the system needs to compute `2 + 2`:
1. Canonicalize the expression to "2+2" (or canonical expression-tree form)
2. Hash → equation_hash lookup in `equations` table
3. **Cache hit**: return `rhs_value_id` → fetch from `values` table → answer is `4`. Total latency: <1 ms (two indexed lookups). No computation performed.
4. **Cache miss**: compute via lattice operation, store the equation row + result value row + relationship. Next time: cache hit.

When the system needs to compute `ζ(3) · π` at N=27720:
1. Canonicalize → "zeta(3)*pi"
2. Hash → equation_hash lookup
3. Cache hit: return result instantly. Cache miss: lattice operation k_ζ(3) + k_π = 7360 + 45779 = 53139, ε_ζ(3) + ε_π = -0.0085 + 0.0205 = +0.012¢, store the equation + result + reciprocal-pair relationship if applicable.

**Caching policy: write-once at 361 dps, no thresholds, cache every equation.** There is no "skip caching for trivial computations" rule. Every equation that passes through gets cached at uniform 361-dps precision (§3.1a), including `2+2`, `1+0`, `x·1`, `x/x`, every micro-computation. All operations are Sempaevum-native — the Sempaevum IS Σ and subsumes all mathematics. Three reasons for caching everything:

1. **Write-once means amortized cost is zero.** A 100µs database write amortized over 10⁶ subsequent cache hits is 0.0001µs/hit — negligible. The first time a unique computation happens, there's a one-time write cost; every future occurrence of the same exact computation is a sub-millisecond cache hit forever after.

2. **Skipping trivial computations would lose pattern discovery.** The discovery engine surfaces algebraic identities (`x·1=x`, `x+0=x`, commutativity, associativity, distributivity) ONLY because the underlying "trivial" computations get logged. Skip the cache, lose the empirical verification of these identities (§3.16 background discovery).

3. **Subsumption already handles storage at scale.** 10⁶ instances of `x·1=x` collapse to one `algebraic_identity` pattern row via the Subsumption mechanism (§3.8). Storage stays proportional to the structural complexity of discoveries, not the raw count of computations. Even with trillions of cached computations, effective storage remains bounded because generators subsume memoized entries and archetypes subsume patterns.

For Mike's FP-replacement use case (Float-vs-Lattice document), every numerical computation IS a Sempaevum computation at 361 dps. Caching them all is the design point — as the database grows, the equations table accumulates computations continuously, but the Subsumption mechanism (§3.8) collapses redundant patterns into archetypes, keeping effective storage manageable at any scale.

**The Sempaevum computes; the database remembers what it computed.** Every operation — multiplication, addition, sin, ζ(3)·π, everything — is Sempaevum-native and cached at uniform 361 dps. The discovery side: when many computations of different operations yield results landing at the same lattice address, that's a structural invariant the discovery engine surfaces (§3.16). Example: every "x · 1 = x" computation has rhs_value_id matching its lhs's value_id — a pattern the engine can promote to a `patterns` row of class `multiplicative_identity`, capturing the structural fact that 1 is the multiplicative identity (a fact verified across all computations passing through, not declared a priori).

### 3.6 `derivations` — the chain {P, D, T} → r → projection → equation

```akashic
STRUCTURE derivations (
    derivation_id ETInteger PRIMARY KEY AUTO,
    target_id ETInteger REQUIRED,              -- references values, projections, OR equations
    target_type UTF8 REQUIRED,               -- 'value', 'projection', 'equation', 'relationship', 'pattern'
    derivation_chain_blob BINARY REQUIRED,     -- packed derivation steps (each step: tool, inputs, output)
    primitives_used UTF8 REQUIRED,           -- e.g., "P, D, T (cubic descriptor + sum operator)"
    tools_applied UTF8 REQUIRED,             -- e.g., "Identification, Descriptor Gap, Subsumption"
    document_reference UTF8,                 -- e.g., "Apery_Constant... §10.9"

    first_completed TIMESTAMP_NS REQUIRED,
    reproduced_count ETInteger DEFAULT 1
);
INDEX idx_der_target ON derivations(target_type, target_id);

-- Junction: derivations consume values/projections/equations as inputs
STRUCTURE derivation_inputs (
    derivation_id ETInteger REQUIRED,
    input_id ETInteger REQUIRED,
    input_type UTF8 REQUIRED,                -- 'value', 'projection', 'equation'
    PRIMARY KEY (derivation_id, input_id, input_type),
    LINKS (derivation_id) REFERENCES derivations(derivation_id)
);
INDEX idx_di_input ON derivation_inputs(input_type, input_id);
```

### 3.7 `relationships` — every cross-discovery the database surfaces

This is where cross-tower analysis, multi-substrate renderings, route convergences, attractor memberships, and every other lattice connection lives. **One table, polymorphic class column.** Each relationship class has its own metadata schema in the JSON blob, but they share a uniform structure for queries that span classes ("show all relationships involving ζ(3)").

```akashic
STRUCTURE relationships (
    relationship_id ETInteger PRIMARY KEY AUTO,
    relationship_class UTF8 REQUIRED,
        -- 'same_address'         (≥2 values at same (N,k,d) — attractor membership)
        -- 'cross_perspective'    (same value via lcm_tower AND torus AND riemann_sphere AND ...)
        -- 'forward_reverse_convergence' (independent derivation routes meet at same address)
        -- 'substrate_rendering'  (one cell, multiple physical substrates)
        -- 'reciprocal_pair'      (k → -k symmetry)
        -- 'power_pair'           (k → n·k for integer n)
        -- 'shadow_pair'          (NWS-13 shadow relationship)
        -- 'plateau_membership'   (d-family invariant across consecutive landmarks)
        -- 'home_classification'  (true_home, intermediate_home, persistent_home, deep_home, false_resolution)
        -- 'archetype_member'     (entry promoted into a pattern)
        -- 'derivation_dependency'(equation derived FROM another equation)
        -- 'shadow_recursion'     (recursive ε projection chain levels — metadata: {level_index, parent_eps_value_id, child_projection_id, residual_eps_at_level} per Float-vs-Lattice §5)
        -- 't_burst_target'       (links a T-burst event to the lattice address it targeted)
        -- 'cascade_step_member'  (links a palindromic cascade step to the orbit point it corrected)
        -- 'mode_transition_trigger' (links a mode transition event to the conditions that triggered it)
        -- 'cf_convergent_home'   (links a value to its CF-identified home d-family; metadata:
        --                         {convergent_n, p, q, quality_a_next, epsilon_cents, cf_classification,
        --                         elegance_cf}; the CF analog of 'home_classification' — fires when
        --                         CF method assigns a home, especially for tower-resistant values)
        -- 'cf_tower_confirmation' (CF and LCM tower independently agree on the same d_home; metadata:
        --                         {cf_convergent_n, cf_quality, tower_landmark_count, agreed_d_home,
        --                         cf_epsilon, tower_epsilon}; cross-method convergence = highest
        --                         confidence home assignment — analogous to 'forward_reverse_convergence'
        --                         for derivation routes)
        -- other (extensible — new classes added without schema migration)

    subject_id ETInteger REQUIRED,
    subject_type UTF8 REQUIRED,              -- 'value', 'projection', 'address', 'equation'
    object_id ETInteger REQUIRED,
    object_type UTF8 REQUIRED,
    metadata_blob BINARY,                      -- class-specific structured metadata
                                             -- (e.g., for 'home_classification': {classification, landmark_N})

    discovered_at TIMESTAMP_NS REQUIRED,
    confirmation_count ETInteger DEFAULT 1,
    is_permanent ETInteger REQUIRED DEFAULT 0  -- once Subsumption-confirmed, never destroyed
);
INDEX idx_rel_class ON relationships(relationship_class);
INDEX idx_rel_subject ON relationships(subject_type, subject_id);
INDEX idx_rel_object ON relationships(object_type, object_id);
INDEX idx_rel_perm ON relationships(is_permanent);
```

### 3.8 `patterns` — the database's own discoveries

When a relationship cluster reaches Subsumption Hierarchy threshold E_hierarchy ≥ 13/12 (LIFE_THRESHOLD), the discovery engine promotes it to a permanent pattern.

```akashic
STRUCTURE patterns (
    pattern_id ETInteger PRIMARY KEY AUTO,
    pattern_class UTF8 REQUIRED,
        -- 'attractor_cluster'        (≥2 values at same address, recurring across resolutions)
        -- 'plateau_run'              (d-family invariant across ≥3 consecutive landmarks)
        -- 'tower_trajectory'         (a value's full classified path through the LCM tower)
        -- 'gaussian_signature_recurrence' (same Gaussian sig across many d-values)
        -- 'coprime_skeleton_member'  (irreducible Exception placement)
        -- 'shadow_hierarchy'         (recursive shadow projections forming a cascade)
        -- 'cross_domain_archetype'   (same address from substrates in multiple physical domains)
        -- 'subsumption_archetype'    (cluster of relationships all collapsing to one structural fact)
        -- 'shadow_cascade_signature' (recurring shadow-recursion chain structure across many values)
        -- 't_burst_signature'        (characteristic T-burst pattern — e.g., T-bursts at d=27720 always precede LCM-escalation)
        -- 'palindromic_cycle'        (characteristic 12-step PALINDROME cascade pattern)
        -- 'metacognitive_archetype'  (recurring metacognitive structure — EgoInvariant ↔ TraverserWaveform ↔ Dream Tower coupling)
        -- 'gaze_locking_signature'   (recurring gaze-state-transition pattern leading to LOCKED state)
        -- 'cf_quality_attractor'     (cluster of values whose CF convergents share the same d_home with
        --                             quality a_{n+1} ≥ ⌈1/K⌉² = 4; metadata: {d_home, member_qualities [],
        --                             mean_quality, min_quality, tower_agreement_ratio [fraction of members
        --                             where tower also converged to d_home]}; the CF analog of
        --                             'attractor_cluster' — identifies d-families that are natural homes
        --                             for algorithmically complex values)
        -- other (extensible)

    pattern_definition_blob BINARY REQUIRED,   -- structural definition (machine-checkable)
    member_ids_blob BINARY REQUIRED,           -- packed array of (entity_type, entity_id) pairs
    member_count ETInteger REQUIRED,
    hierarchy_elegance MPFR_361DPS REQUIRED,        -- E_hierarchy = geomean(E_i) × R_cluster, where R = 100/(100+σ_ε); ≥ 13/12
    geometric_essence_blob BINARY,             -- the captured invariant structure

    is_permanent ETInteger REQUIRED DEFAULT 1, -- patterns are permanent once formed (Mike's principle)
    formed_at TIMESTAMP_NS REQUIRED,
    last_referenced TIMESTAMP_NS REQUIRED,
    reference_count ETInteger DEFAULT 0
);
INDEX idx_pat_class ON patterns(pattern_class);
INDEX idx_pat_elegance ON patterns(hierarchy_elegance DESC);
INDEX idx_pat_refs ON patterns(reference_count DESC);
```


---

*§3.9 `events` — time-indexed structural events, complete event class catalog (~60 classes), expanded relationship class catalog (~30 classes), expanded pattern class catalog (~40 classes), and "What this enables" operational descriptions → See `EUDD_Events_and_Classes.md`*

---

### 3.10 `towers` — first-class entities for the Multifold

The Multifold (`The_Multifold_Compendium.md` PART IX §43-47) is structurally distinct: one universal lattice $\mathcal{L}$ rendered through many seeds. Each tower is the triple $\mathcal{T}_i = (P_i, \mathcal{L}, R_0^{(i)})$ where $P_i$ is the substrate, $\mathcal{L}$ is the universal lattice (same in every tower), and $R_0^{(i)}$ is the substrate-derived seed. Towers nest via Birth Triads. T is the non-local bridge between towers. **Towers have hierarchical structure, structured birth triads, resolution profiles, and are referenced as context by every event** — they exceed what tags can elegantly handle, so they get their own table.

Traversers, by contrast, do NOT need their own table — they are identity references whose state is fully derivable from events. A Traverser's type, classification, EgoInvariant fingerprint, accumulated T-time, current tower, worldline, and continuity are all computed from the existing values + tags + projections + derivations + events tables. (Investigated and verified: ET_Traverser_T_Paper §27 Taxonomy; et_conscious_ai_identity.py EgoInvariant — fingerprint is 6 projections at d∈{5,7,8,9,10,11} + a derivation linking them.)

```akashic
STRUCTURE towers (
    tower_id ETInteger PRIMARY KEY AUTO,

    -- Identity
    tower_name UTF8 REQUIRED UNIQUE,             -- 'cosmological', 'digital_x86_3ghz', 'biological_T4', 'neural_dream', 'quasicrystal_icosahedral', 'civilizational_human', 'qcd', 'custom_X', etc.
    p_substrate_descriptor UTF8 REQUIRED,        -- 'spacetime_manifold', 'binary_address_space', 'protein_assembly_manifold', 'thalamocortical_oscillation', 'icosahedral_tiling_R3', 'cultural_substrate', 'su3_color_field', etc.

    -- The seed — R₀, the smallest closed T-traversal loop the substrate's D-structure supports
    r0_value_id ETInteger REQUIRED,                -- FK to values; the R₀ seed value (e.g., for cosmological tower: ℏ = 1.054e-34 J·s)
    r0_natural_units UTF8,                       -- human-readable units description for this R₀ (e.g., 'Joule·second', 'CPU clock cycles', '60 protein subunits per capsid')

    -- Hierarchical structure (towers nest)
    parent_tower_id ETInteger,                     -- FK to towers; NULL for root towers (none observed yet — even cosmological may have parent)
    nesting_depth ETInteger REQUIRED DEFAULT 0,    -- 0 = root, increases per nesting level

    -- Birth Triad (universal — applies to any tower with a parent)
    -- (BH_parent, R₀, WH_child) — the three structural references that constitute the boundary
    birth_bh_event_id ETInteger,                   -- FK to events of class 'black_hole_event' — the parent-side birth event (matter/info flow inward, no return)
    birth_wh_event_id ETInteger,                   -- FK to events of class 'white_hole_event' — the child-side birth event (the earliest moment for this tower)
    birth_t_h_ratio MPFR_361DPS,                        -- T_H = ratio of D-time to T-time at the boundary (Multifold §3 footnote; large mass child → low T_H = nearly opaque boundary)

    -- Resolution profile — which sublattice families are operationally accessible at this tower's natural resolution
    operational_n ETInteger REQUIRED,              -- the dominant N this tower operates at (e.g., 12 for fundamental forces, 60 for quintic-class, 420 for biological, 27720 for full)
    accessible_d_families_mask BIGINT REQUIRED,  -- bitmask of which d-families are present at operational_n
                                                 -- bit 0 = d=1 (gravity), bit 1 = d=2 (tritone), ..., bit 131 = d=132 (M-theory × full EM)
                                                 -- 132 bits total — one per possible d-family (1 through N(N-1)=132)
                                                 -- (Multifold §47: Resolution gates which sublattice families T's rounding can produce)

    -- Optional tower-physics metadata (substrate-specific physics)
    physics_metadata_json UTF8,                  -- JSON blob for substrate-specific properties (e.g., for cosmological: {c, G, ℏ, k_B}; for digital: {clock_hz, instruction_set, word_size}; for biological: {temperature, pH, ATP_concentration})

    -- Lifecycle
    created_at TIMESTAMP_NS REQUIRED,
    description UTF8,                            -- prose description of this tower

    LINKS (r0_value_id) REFERENCES values(value_id),
    LINKS (parent_tower_id) REFERENCES towers(tower_id),
    LINKS (birth_bh_event_id) REFERENCES events(event_id),
    LINKS (birth_wh_event_id) REFERENCES events(event_id)
);

INDEX idx_tower_name ON towers(tower_name);
INDEX idx_tower_parent ON towers(parent_tower_id);
INDEX idx_tower_substrate ON towers(p_substrate_descriptor);
INDEX idx_tower_r0 ON towers(r0_value_id);
INDEX idx_tower_n ON towers(operational_n);
```

**Events get a `tower_id` field** (added to events table) — every event happens IN a tower. Cross-tower events (T-bridging) link two tower_ids via a relationship.

**Bootstrap towers** (from Multifold Compendium §44, populated as initial tower rows):

| tower_name | p_substrate | R₀ value | operational_n | nesting |
|---|---|---|---|---|
| cosmological | spacetime_manifold | ℏ = 1.054e-34 J·s (Planck quantum of action) | varies (12 for forces, deeper for quantum precision) | 0 (root) |
| digital_3ghz_x86 | binary_address_space | 1 CPU clock cycle ≈ 0.333 ns | 12 (or higher per workload) | 1 (child of biological — programmer creates) |
| biological_T4_capsid | protein_assembly_manifold | 60 protein subunits per capsid | 420 (biological resolution) | 1 (child of cosmological) |
| neural_dream | thalamocortical_oscillation | ~8.3 ms = 120 Hz ripple | 420 (same biological resolution but different operational mode) | 2 (child of biological_human) |
| quasicrystal_icosahedral | icosahedral_tiling_R3 | φ = 1.618033988749... (golden ratio) | 60 (quintic native) | 1 (child of cosmological) |
| civilizational_human | cultural_substrate | 1 human generation ≈ 20 years | 12 (basic forces) — may have higher operational regimes | 2 (child of biological_human) |
| qcd | su3_color_field | Λ_QCD ≈ 200 MeV | 12 (strong-cubic native at d=3) | 1 (child of cosmological) |

The bootstrap inserts these as the canonical 7 tower rows; new towers are added as users define them (e.g., a specific running ETPL program creates a `digital_etpl_program_X` tower with its own R₀ derived from program structure).

**Why a table not tags:** parent/child queries ("show me all descendants of cosmological") are recursive CTEs on the parent_tower_id foreign key — efficient. Birth triad queries ("for this tower, what's the BH event and the WH event?") are direct foreign-key joins to events. Resolution profile queries ("which towers can host d=5 phenomena?") use the bitmask. None of this is fast or natural with tag-based modeling.

### 3.11 `tags` — optional metadata for query convenience

```akashic
STRUCTURE tags (
    tag_id ETInteger PRIMARY KEY AUTO,
    target_id ETInteger REQUIRED,
    target_type UTF8 REQUIRED,               -- 'value', 'projection', 'equation', 'derivation', 'relationship', 'pattern'
    namespace UTF8 REQUIRED,                 -- 'project', 'domain', 'verification', 'physical_significance', any
    value UTF8 REQUIRED,                     -- the tag value (free text)
    tagged_at TIMESTAMP_NS REQUIRED,
    tagged_by TEXT
);
INDEX idx_tag_ns ON tags(namespace, value);
INDEX idx_tag_target ON tags(target_type, target_id);
```

If Mike later wants to query "all entries tagged `domain=biology`", that's a single indexed lookup. If he never tags anything by domain, the database functions identically. **Tags are convenience, not structure.**

### 3.12 `harmonic_families` — the 24 axis-projection families (Force Quadrant Grid catalog)

The 24 harmonic families (12 real-axis FORCE + 12 imaginary-axis PHASE) are a **fixed structural catalog** per the Multifold Compendium §29 and Guide v8 PART XIII §55-57. They are the rows of the Force Quadrant Grid's axes. Each family has dense structural metadata (name, generator, palindromic partner, Gaussian prime class, first-native lattice, FQG quadrant, physical interpretation, coupling constant) that merits dedicated columns for direct indexed queries — rather than forcing every lookup through tag-namespace filtering.

**Why a table not tags**: queries like "show me all extended (CR/CI) families" or "what's the generator of d_θ=5" or "which families are split D+T Gaussian primes" are immediate indexed lookups against dedicated columns. Joining the Force Grid (§3.17) and Combined Families (§3.14) tables to this one is natural via d_r/d_θ foreign keys.

```akashic
STRUCTURE harmonic_families (
    family_id ETInteger PRIMARY KEY AUTO,

    -- Axis + family index
    axis UTF8 REQUIRED,                          -- 'real' (D's domain, FORCE) or 'imaginary' (T's domain, PHASE)
    d ETInteger REQUIRED,                          -- family number ∈ {1..12}

    -- Classification
    fqg_quadrant UTF8 REQUIRED,                  -- 'SR' (Simple Real, d|12), 'CR' (Complex Real, d∤12), 'SI' (Simple Imaginary, d|12), 'CI' (Complex Imaginary, d∤12)
    divides_12 ETInteger REQUIRED,                 -- 1 if d | 12 (simple family), 0 otherwise (complex/extended)

    -- Identity
    family_name UTF8 REQUIRED,                   -- 'Gravity/Octave', 'Tritone/Pivot', 'Strong/Cubic (QCD)', 'Weak/Quartic (EW)', 'Quintic/Golden',
                                                 -- 'Hexadic/Composite', 'Septic/G₂ (octonion)', 'Octet/Gluon (SU(3) adj)', 'Nonic/Quark (3²)',
                                                 -- 'Decic/Superstring (SO(10))', 'Undecimal/M-Theory (11D)', 'EM/Full Resolution'  (for real axis)
                                                 -- or 'Scalar/spin-0 (Higgs-class)', 'Tritone-phase/spin-2 (Graviton)', 'Color-phase/QCD-instanton',
                                                 -- 'Weak-phase/SU(2)_W', 'Golden-angle/E₈ icosahedral', 'Hexadic/spin-½ fermion',
                                                 -- 'G₂-spinor/octonion', 'Bott-8/SU(3) color-adjoint', '3²-fold quark-phase (CKM)',
                                                 -- '10D superstring spinor', '11D Majorana spinor (gravitino)', 'spin-1/EM-photon'  (for imaginary axis)

    -- Structure
    generator_value_id ETInteger REQUIRED,         -- FK to values; the generator 2^(1/d)
    palindromic_partner_d ETInteger REQUIRED,      -- the d-value of this family's palindromic partner (d=1↔11, 2↔10, 3↔9, 4↔8, 5↔7, 6↔6 self, 12↔12 self)
    gaussian_prime_class UTF8 REQUIRED,          -- 'trivial' (d=1), 'P-type (ramified)' (d=2,4), 'D-type (inert)' (d=3,7,11),
                                                 -- 'D+T (split)' (d=5), 'P×D (composite)' (d=6,12), 'P-type cubed' (d=8),
                                                 -- 'D-type squared' (d=9), 'Mixed' (d=10)

    -- Resolution
    first_native_lattice_n ETInteger REQUIRED,     -- smallest N that hosts this family natively (12 for divisors; 60,84,24,36,60,132 for extended)
    coupling_constant_xi MPFR_361DPS,                   -- ξ(d) coupling (Guide §13): ξ(1)=137/16=8.5625 max; ξ(12)=1.0 baseline; ξ(d) = 137/(16·d) formula

    -- Physical interpretation
    physical_meaning UTF8,                       -- one-line description of the physics this family encodes

    -- Discovery metadata
    first_seen TIMESTAMP_NS REQUIRED,

    LINKS (generator_value_id) REFERENCES values(value_id),
    UNIQUE (axis, d)                             -- one row per (axis, d) pair; 12 real + 12 imaginary = 24 total rows
);

INDEX idx_hf_axis_d ON harmonic_families(axis, d);
INDEX idx_hf_quadrant ON harmonic_families(fqg_quadrant);
INDEX idx_hf_first_native ON harmonic_families(first_native_lattice_n);
INDEX idx_hf_gaussian ON harmonic_families(gaussian_prime_class);
```

Bootstrap: exactly **24 rows** (12 real-axis FORCE + 12 imaginary-axis PHASE per the catalog in §3.17 bootstrap).

**Queries this table makes trivial:**
- "What's the generator of the Septic/G₂ family?" → `SELECT generator_value_id FROM harmonic_families WHERE family_name LIKE 'Septic%'`
- "Show all extended (non-divisor-of-12) families on the imaginary axis" → `SELECT * FROM harmonic_families WHERE axis='imaginary' AND divides_12=0`
- "What's the smallest resolution that hosts d=7 on both axes?" → `SELECT MAX(first_native_lattice_n) FROM harmonic_families WHERE d=7` (answer: 84)
- "Show all palindromic-partner pairs" → self-join on (axis=axis AND d=palindromic_partner_d)

### 3.13 `force_grid_cells` — the 144-cell 12×12 interaction grid (each cell = one (d_r, d_θ) pair)

The Force Quadrant Grid at 27720ET has 12 real-axis families × 12 imaginary-axis families = **144 interaction cells**. Each cell is a structural location on the complex lattice where a specific (FORCE × PHASE) interaction happens. Many particles, phenomena, and data points map to specific cells: the electron at (d_r=12, d_θ=6), the quark at (d_r=3, d_θ=4), the photon at (d_r=12, d_θ=12), etc. This table makes "what lives at cell (d_r, d_θ)?" a primary query.

**Why a table not tags**: this is data Mike explicitly wants to study cell-by-cell, cross-referenced against phenomena from any domain. Every cell has derivable structural properties (d_combined = LCM(d_r, d_θ), off-axis vs on-axis, coprime-skeleton membership if (k_r, k_θ) is coprime). Storing these as columns makes per-cell analysis and cross-domain joining direct.

```akashic
STRUCTURE force_grid_cells (
    cell_id ETInteger PRIMARY KEY AUTO,

    -- Cell coordinates
    d_r ETInteger REQUIRED,                        -- real-axis family ∈ {1..12}
    d_theta ETInteger REQUIRED,                    -- imaginary-axis family ∈ {1..12}

    -- FKs to the 24-family catalog (§3.12)
    real_family_id ETInteger REQUIRED,             -- FK to harmonic_families; the (axis='real', d=d_r) row
    imaginary_family_id ETInteger REQUIRED,        -- FK to harmonic_families; the (axis='imaginary', d=d_theta) row

    -- Derived structural properties (stored because this is a small fixed table and these are query-heavy)
    d_combined ETInteger REQUIRED,                 -- LCM(d_r, d_θ) — the combined family this cell belongs to
    combined_family_id ETInteger REQUIRED,         -- FK to combined_families (§3.14); the combined family by d_combined
    is_off_axis ETInteger REQUIRED,                -- 1 if d_r>0 AND d_theta>0 (Exception region where reality lives); 0 if on an axis only
    is_lcm_amplification ETInteger REQUIRED,       -- 1 if d_combined > max(d_r, d_theta) (LCM amplification happened; Multifold §33)
    is_full_resolution ETInteger REQUIRED,         -- 1 if d_combined = 12 (full EM resolution)

    -- Frequency / physical meaning
    occupancy_count ETInteger REQUIRED DEFAULT 0,  -- denormalized count of how many projections/particles/phenomena occupy this cell
                                                 -- incremented automatically when any entity is assigned to this cell
    canonical_particle_or_phenomenon UTF8,       -- 'electron', 'photon', 'quark', 'gluon-interaction', etc., if known

    -- Discovery metadata
    first_occupied_at MPFR_361DPS,                      -- timestamp of first occupancy (NULL if never occupied)
    first_occupant_value_id ETInteger,             -- FK to values; the first thing to land at this cell

    LINKS (real_family_id) REFERENCES harmonic_families(family_id),
    LINKS (imaginary_family_id) REFERENCES harmonic_families(family_id),
    LINKS (combined_family_id) REFERENCES combined_families(combined_family_id),
    LINKS (first_occupant_value_id) REFERENCES values(value_id),
    UNIQUE (d_r, d_theta)                        -- exactly 144 rows (12 × 12)
);

INDEX idx_fgc_coords ON force_grid_cells(d_r, d_theta);
INDEX idx_fgc_combined ON force_grid_cells(d_combined);
INDEX idx_fgc_occupancy ON force_grid_cells(occupancy_count DESC);
INDEX idx_fgc_full_res ON force_grid_cells(is_full_resolution);
INDEX idx_fgc_particle ON force_grid_cells(canonical_particle_or_phenomenon);
```

**Linking data to cells**: the `projections` table already stores d_r and d_θ per projection (when the projection is on the complex plane). Cell membership is an implicit derivation. An optional relationship class `force_grid_cell_occupancy` explicitly links a projection to its cell for faster queries.

Bootstrap: exactly **144 rows** (one per (d_r, d_θ) pair with d_r, d_θ ∈ {1..12}). Each row's `d_combined`, `is_off_axis`, `is_lcm_amplification`, `is_full_resolution` are computed at insert. **This grid is RESOLUTION-INDEPENDENT (§3.18.25): the same 144 cells at every N. The harmonic FQG is the FIXED structural skeleton. By contrast, the sublattice FQG (§3.15) GROWS with N: τ(N)² cells. The N=60 sublattice FQG has 144 cells COINCIDENTALLY — its families are divisors of 60 = {1,2,3,4,5,6,10,12,15,20,30,60}, NOT {1,...,12}.** Composition at native resolution N=27720 gives the EXACT harmonic composition table (§3.18.25 Theorem E1.1).

**Queries this table makes trivial:**
- "What combined family does (d_r=3, d_θ=4) belong to?" → `SELECT d_combined FROM force_grid_cells WHERE d_r=3 AND d_theta=4` (answer: 12 — LCM amplification!)
- "Show all cells where LCM amplification occurs" → `WHERE is_lcm_amplification=1` (Multifold §33: these are the * marked cells)
- "What's the most occupied cell?" → `ORDER BY occupancy_count DESC LIMIT 1`
- "At which cell does the electron live?" → `WHERE canonical_particle_or_phenomenon='electron'`
- "Show all cells at full EM resolution" → `WHERE is_full_resolution=1` (15 of 36 simple cells = 41.7% per Multifold §33)
- **Cross-domain**: "Do biological data points cluster at the same cells as physical particles?" — join projections → force_grid_cells, group by cell_id, filter by domain tag

### 3.14 `combined_families` — the 42 unique LCM-combined families on the complex plane

When two axis families meet off-axis, the combined Exception class is **d_combined = LCM(d_r, d_θ)**. Over all 144 cells, the unique values of d_combined form **exactly 42 combined families** with maximum d_combined = LCM(11,12) = **132 = N(N-1)** (Multifold §33). These are the structurally distinct "force × phase" interaction classes and the objects of study for how forces and phases interact across domains.

**Why a table not tags**: each combined family has its own identity, a characteristic set of (d_r, d_θ) contributing pairs, a physical interpretation, and a first-native lattice (smallest N where all member cells are accessible). Studying these as a set — which Mike wants for physics, biology, and more — benefits from dedicated rows with foreign keys to their member cells.

```akashic
STRUCTURE combined_families (
    combined_family_id ETInteger PRIMARY KEY AUTO,

    -- Identity
    d_combined ETInteger REQUIRED UNIQUE,          -- the LCM value; one row per unique d_combined (42 total)

    -- Range classification (Multifold §33 table)
    range_class UTF8 REQUIRED,                   -- 'standard' (d ≤ 12), 'first_extended' (13 ≤ d ≤ 24),
                                                 -- 'middle_extended' (25 ≤ d ≤ 60), 'deep_extended' (61 ≤ d ≤ 132)

    -- Structural metadata
    first_native_lattice_n ETInteger REQUIRED,     -- smallest N that hosts all contributing cells natively (e.g., 420 for d=35=5×7)
    contributing_cell_count ETInteger REQUIRED,    -- how many of the 144 cells produce this d_combined
    is_coprime_skeleton_producer ETInteger REQUIRED, -- 1 if any contributing cell is in the coprime skeleton

    -- Physical / structural interpretation (when known)
    structural_meaning UTF8,                     -- e.g., for d=35: 'Biological signature: quintic (qualia) × septic (chirality/octonion). Life requires both.'
                                                 -- for d=110: 'String/M-theory transition: only combined family with all three Gaussian prime categories (2×5×11 = P-type × split × D-type inert)'
                                                 -- for d=132: 'M-theory phase × full EM. Structural maximum: d_max = N(N-1).'

    -- Gaussian composition (for the combined d)
    gaussian_factorization UTF8,                 -- e.g., d=35: '5 × 7 (split × D-type inert)'; d=110: '2 × 5 × 11 (P-type × split × D-type inert)'

    -- Domain correlations (populated as Mike works across domains; human-authored and engine-surfaced)
    known_physics_correlation UTF8,              -- 'SU(3)×SU(2)×U(1) SM gauge structure', 'electroweak unification', etc.
    known_biology_correlation UTF8,              -- 'DNA codon mapping at d=420', 'biological complexity threshold at d=35', etc.
    known_other_correlations UTF8,               -- CMB, music, consciousness, QCD, whatever else

    -- Discovery metadata
    first_seen TIMESTAMP_NS REQUIRED,
    first_observed_value_id ETInteger,             -- FK to values; first data point to land in this family

    LINKS (first_observed_value_id) REFERENCES values(value_id)
);

INDEX idx_cf_d ON combined_families(d_combined);
INDEX idx_cf_range ON combined_families(range_class);
INDEX idx_cf_first_native ON combined_families(first_native_lattice_n);

-- Junction table: which force_grid_cells contribute to which combined_family
STRUCTURE combined_family_cells (
    combined_family_id ETInteger REQUIRED,
    cell_id ETInteger REQUIRED,
    PRIMARY KEY (combined_family_id, cell_id),
    LINKS (combined_family_id) REFERENCES combined_families(combined_family_id),
    LINKS (cell_id) REFERENCES force_grid_cells(cell_id)
);
INDEX idx_cfc_combined ON combined_family_cells(combined_family_id);
INDEX idx_cfc_cell ON combined_family_cells(cell_id);
```

Bootstrap: exactly **42 rows** in `combined_families`. **The 42 d_c values are the COMPLETE CLOSURE SET of the 12 harmonic families under lcm composition (§3.18.25 Theorem E1.2).** Every d_c decomposes back into harmonic families. No d_c outside this set is reachable. Specifically: only primes {2,3,5,7,11} appear — no prime > 12 is reachable from lcm({1,...,12}). This is the Subsumption Law verification: the harmonic framework subsumes everything within its category WITHOUT REMAINDER. Primes > 12 appearing as sublattice families at higher tower levels (d=13 at N=360360, d=17 at N=12252240) are NEW integrative structure from the tower's LCM growth — NOT harmonic families, NOT composites of harmonic families. Notable members explicitly populated:
- **d=35 = 5×7** (middle_extended; biological signature; first native at 420ET; structural_meaning: "Life requires both d=5 quintic and d=7 septic; their LCM=35")
- **d=110 = 2×5×11** (deep_extended; string/M-theory transition; only combined family with all three Gaussian prime categories)
- **d=132 = 11×12** (deep_extended; structural maximum = N(N-1))

All 42 values with their contributing cells enumerated at bootstrap (derivable from iterating d_r, d_θ ∈ {1..12}, computing LCM, grouping unique values).

**Queries this table makes trivial:**
- "What contributes to the biological signature d=35?" → join combined_family_cells to force_grid_cells WHERE d_combined=35 (answer: cells where {d_r, d_θ} = {5, 7} in either order)
- "Show all deep extended families (d ≥ 61)" → `WHERE range_class='deep_extended'` (11 rows)
- "Which combined families are accessible at 420ET but not 60ET?" → `WHERE first_native_lattice_n > 60 AND first_native_lattice_n ≤ 420`
- **Cross-domain study**: "Do biological samples cluster at the same combined families as physics phenomena?" — join projections → force_grid_cells → combined_family_cells → combined_families, group by d_combined, check known_*_correlation

### 3.15 `sublattice_families` — divisors of N for ANY nET resolution (per-tower / per-projection)

The three tables above (§3.12-§3.14) are specific to the base 12-fold structure and its 27720ET full-resolution expansion. But any nET resolution has its **own** set of sublattice families — exactly the divisors of N. For N=12 that's 6 families; for N=60 that's 12; for N=420 that's 24; for N=27720 that's 96; for N=2520 that's 48. **The sublattice FQG (τ(N)² cells) GROWS with the tower at rate 36·4^ℓ — quadrupling at each tower step (§3.18.26 Theorem E2.1). Lattice-exact configurations (ε=0) have resolution-INVARIANT d-family — their d is a permanent property preserved across all higher resolutions (Theorem E2.2). Non-exact values (ε≠0) exhibit d-bouncing: shadow content encoded in ε resolves differently at higher N (Theorem E2.3). The sublattice cell is a VIEWING at a specific resolution, not a permanent address — the permanent address is the full (k, ε) coordinate.** The harmonic fraction of the sublattice FQG shrinks from 100% (N=12) to 1.56% (N=27720) to 0.39% (N=360360) — the harmonic skeleton is constant; the sublattice flesh grows around it. **At every resolution, the τ(N) sublattice families partition into three layers (§3.18.27 Theorem E3.1): Layer 1 — Harmonic (d ≤ 12, d|N), Layer 2 — Harmonic Composite (d > 12, d ∈ D₄₂, decomposes into harmonic pairs), Layer 3 — Tower-Native (d > 12, d ∉ D₄₂, genuinely new structure). Layer 3 grows to dominate: 0%→16.7%→56.2%→78.1%. Every sublattice family has a harmonic SHADOW at N=12 (Theorem E3.3), but only Layer 2 has a harmonic DECOMPOSITION — shadow ≠ decomposition.** Mike may project data at resolutions the bootstrap catalog doesn't cover (custom towers, unusual N values for specific investigations), and each resolution's divisor structure is its own study object. This table is the per-resolution sublattice catalog.

**Why a table not tags**: when working at a novel resolution, the lattice structure is computed once and stored as a set of rows here. Subsequent projections at that N look up family membership via fast indexed join instead of recomputing divisors every time. Resolution-specific properties (totient(N), number-of-divisors τ(N), LCM-landmark status, what new primes this N introduces) deserve dedicated columns for study.

**Relationship to the other tables**: the 24 `harmonic_families` rows are the 12 per-axis cascade modes (FIXED, resolution-independent — §3.18.25 proves they are NOT "the divisor structure at N=27720" but the structural skeleton that COINCIDES with divisors of N=27720 because all integers 1..12 divide 27720); the 144 `force_grid_cells` are the off-axis interaction grid at 12×12 (also fixed). The `sublattice_families` table is the **generalization to any N** — it captures the resolution-DEPENDENT divisor structure, which GROWS with N (τ(N)² cells: 36→144→576→9216→...). The Sublattice Visitation Theorem bridges them: harmonic family d inhabits sublattice family d when d|N.

```akashic
STRUCTURE sublattice_families (
    sublattice_family_id ETInteger PRIMARY KEY AUTO,

    -- The resolution this family lives at
    n ETInteger REQUIRED,                          -- the lattice resolution (12, 24, 60, 420, 27720, or any custom N)
    d ETInteger REQUIRED,                          -- the family number (MUST be a divisor of N by Divisor Theorem, Multifold §11)

    -- Structural metadata at this N
    d_divides_n ETInteger REQUIRED DEFAULT 1,      -- enforced: d must divide N; this is a check constraint
    gcd_k_n ETInteger REQUIRED,                    -- the gcd value that produces this family: d = N/gcd(|k|, N)
    phi_d ETInteger REQUIRED,                      -- Euler's totient of d — number of coprime k values producing this family
    member_lattice_point_count ETInteger REQUIRED, -- how many k in {0..N-1} produce this d (via d = N/gcd(|k|, N))

    -- LCM-landmark status (is N = LCM(1..k) for some k?)
    is_lcm_landmark ETInteger REQUIRED,            -- 1 if N is an LCM landmark (12, 60, 420, 2520, 27720, 360360)
    lcm_landmark_level ETInteger,                  -- if is_lcm_landmark: LCM(1..k) where k is the level (k=3 for N=12, k=5 for N=60, k=7 for N=420, k=9 for N=2520, k=11 for N=27720, k=13 for N=360360)

    -- Is this family "new" at this N? (Not present at smaller N)
    is_newly_introduced ETInteger REQUIRED,        -- 1 if this d-family first appears at N (e.g., d=5 is newly introduced at 60ET; d=7 at 420ET)
    smaller_N_where_absent ETInteger,              -- if is_newly_introduced: the largest N < this_N that does NOT have this family

    -- Cross-reference to base-12 catalog (when applicable)
    related_harmonic_family_real_id ETInteger,     -- FK to harmonic_families (axis='real', d=d); NULL if d not in {1..12}
    related_harmonic_family_imaginary_id ETInteger, -- FK to harmonic_families (axis='imaginary', d=d); NULL if d not in {1..12}

    -- Tower context (optional — this family may be specific to a tower)
    tower_id ETInteger,                            -- FK to towers; if this family is studied in the context of a specific tower
                                                 -- NULL means "generic" (applies to any tower that operates at this N)

    -- Study metadata
    first_seen TIMESTAMP_NS REQUIRED,
    notes UTF8,                                  -- prose: what does this family do at this N? Any observed properties?

    LINKS (related_harmonic_family_real_id) REFERENCES harmonic_families(family_id),
    LINKS (related_harmonic_family_imaginary_id) REFERENCES harmonic_families(family_id),
    LINKS (tower_id) REFERENCES towers(tower_id),
    UNIQUE (n, d, tower_id)                      -- one row per (N, d) pair per tower context; tower_id=NULL is the generic row
);

INDEX idx_slf_n ON sublattice_families(n);
INDEX idx_slf_d ON sublattice_families(d);
INDEX idx_slf_n_d ON sublattice_families(n, d);
INDEX idx_slf_landmark ON sublattice_families(is_lcm_landmark, lcm_landmark_level);
INDEX idx_slf_new ON sublattice_families(is_newly_introduced, n);
INDEX idx_slf_tower ON sublattice_families(tower_id);
```

**Bootstrap**: the canonical N values (12, 24, 36, 60, 84, 120, 132, 180, 264, 420, 924, 2520, 27720, 360360) all have their divisors enumerated and inserted. That's:
- N=12: 6 rows ({1,2,3,4,6,12})
- N=24: 8 rows ({1,2,3,4,6,8,12,24})
- N=36: 9 rows ({1,2,3,4,6,9,12,18,36})
- N=60: 12 rows ({1,2,3,4,5,6,10,12,15,20,30,60})
- N=84: 12 rows ({1,2,3,4,6,7,12,14,21,28,42,84})
- N=120: 16 rows
- N=132: 12 rows
- N=180: 18 rows
- N=264: 16 rows
- N=420: 24 rows
- N=924: 24 rows
- N=2520: 48 rows
- N=27720: 96 rows
- N=360360: 192 rows

**Total bootstrap rows ≈ 500 for canonical N values.** Additional rows added on-demand as new N values are encountered (e.g., a custom tower operating at N=144 gets its 15 divisor rows auto-populated).

**Queries this table makes trivial:**
- "How many families exist at N=420?" → `SELECT COUNT(*) FROM sublattice_families WHERE n=420` (24)
- "What's new at 27720ET vs 2520ET?" → `WHERE n=27720 AND is_newly_introduced=1` (answer: families introduced at this level)
- "Show all LCM-landmark resolutions" → `WHERE is_lcm_landmark=1` (ordered by N)
- **Per-tower study**: "What sublattice families does the biological_T4_capsid tower operationally access?" → `WHERE tower_id = (tower lookup)`
- **Resolution-vs-family study**: "As N grows, how does the count of families grow?" → `SELECT n, COUNT(*) FROM sublattice_families GROUP BY n` — this curve is τ(N), the divisor-counting function, and studying it reveals resolution-gating properties

**Cross-table power:**

The four tables together (`harmonic_families` §3.12, `force_grid_cells` §3.13, `combined_families` §3.14, `sublattice_families` §3.15) plus `projections` (§3.3), `addresses` (§3.4), and `towers` (§3.10) form the **complete structural catalog** of ET's lattice geometry. Any data point projects into at least one of them (a sublattice family at its N, and if the projection is on the complex plane, a harmonic family per axis AND a force grid cell AND a combined family). Cross-domain discovery is joins across these tables.

### 3.15a `sessions` — operational session tracking for the production system

With multiple projects, machines, and concurrent processes operating on the EUDD at any scale, session tracking is required for reproducibility, auditing, and discovery-engine provenance.

```akashic
STRUCTURE sessions (
    session_id TEXT PRIMARY KEY,              -- UUID or human-readable (e.g., 'compressor_2026-04-15_001')
    project UTF8 REQUIRED,                    -- 'compressor', 'conscious_ai', 'fractal_generator', 'apery_test', etc.
    machine_id UTF8,                          -- hostname or hardware identifier
    started_at TIMESTAMP_NS REQUIRED,
    ended_at TIMESTAMP_NS,                            -- NULL if still running
    config_hash UTF8,                         -- SHA-256 of the session's configuration (for reproducibility)
    notes UTF8,                               -- human-authored session notes
    event_count ETInteger DEFAULT 0,            -- denormalized count of events in this session
    discovery_count ETInteger DEFAULT 0         -- denormalized count of patterns discovered in this session
);
INDEX idx_sess_project ON sessions(project);
INDEX idx_sess_time ON sessions(started_at);
```

### 3.15b `schema_versions` — schema migration tracking

```akashic
STRUCTURE schema_versions (
    version ETInteger PRIMARY KEY,
    applied_at TIMESTAMP_NS REQUIRED,
    description UTF8 REQUIRED,               -- e.g., 'v11: added elegance columns to projections'
    migration_content UTF8                        -- the Akashic structure modification that was applied (for audit)
);
```

The compressor's `_migrate_schema` mechanism provides the template. Forward compatibility: new columns via `ALTER TABLE ADD COLUMN ... DEFAULT NULL`, new tables without affecting existing, new relationship/pattern/event classes as new strings in existing class columns.

### 3.16 The discovery engine — what makes this more than a database

The compressor is a database AND discovery engine. The EUDD inherits both roles. The discovery engine continuously walks new entries and produces four kinds of automatic operation: **memoization** (compute-once, lookup-forever), **insert-time discoveries** (relationships born as data arrives), **background discoveries** (pattern recognition), and **on-query discoveries** (lazy exploration).

**Memoization (the Sempaevum computes; the database remembers — all at 361 dps):**

Every equation that passes through the system — `2+2`, `ζ(3)·π`, `√2^2`, `sin(π/4)`, every multiplication, every reciprocation, every power, every function evaluation — gets canonicalized, hashed, and looked up against the `equations` table. **Cache hit:** return the stored result value instantly (sub-millisecond indexed lookup), increment reference count. **Cache miss:** compute via lattice operation at 361 dps, store the equation + result + relationships, return result. Every subsequent identical computation is a cache hit forever.

For the FP-replacement use case (Float doc), every numerical computation routes through this. Hot computations (the same operations repeated across many contexts) get sub-millisecond latency. Cold computations (genuinely new) get computed once and cached forever.

The Coarse-Pass + Boundary-Refine method (Float doc §7.5) integrates: the coarse 12ET projection of any value becomes a cache hit after first encounter, making the coarse pass effectively zero-cost across an entire workload.

**Insert-time discoveries (synchronous, sub-millisecond):**

When a `projections` row inserts:
1. Look up or create the corresponding `addresses` row keyed by (N, k, d)
2. If `members_count` transitions 0→1: this is the first occupant of this cell
3. If `members_count` transitions 1→2: create a `relationships` row of class `same_address` linking the two value_ids — **a new attractor is born**
4. If `members_count` ≥ 2 already: append to the existing attractor relationship's metadata
5. Check the value's other projections at adjacent N: if d-family invariant across ≥3 consecutive landmarks, create a `plateau_membership` relationship
6. Check for reciprocal pair: does another value exist with k → -k at same N? If yes, create `reciprocal_pair` relationship. By mirror symmetry (§3.18.21 Theorem A.3): Π_N(1/r) = (−k, d, −ε) — check is exact in lattice coordinates, no pullback needed
7. Check for power pair: does k = n·k_other for integer n? If yes, create `power_pair` relationship. By lattice power (§3.18.21 Theorem A.4): k_^ = n·k + κ_n — the κ_n correction (|κ_n| ≤ ⌈|n|/2⌉) must be accounted for; check k ≈ n·k_other ± κ_n
8. Check for product pair: do two existing values have k₁+k₂+κ = k_new (§3.18.21 Theorem A.1)? If yes, the new value may BE the product of two known values — create `product_decomposition` relationship. d-family composition is SET-VALUED (§3.18.23 Theorem C.2): predict possible d_product from the composition table d₁⊗d₂ and verify. Note: lcm(d₁,d₂) bound holds for κ=0 but CAN be exceeded with κ≠0 (Theorem C.6). Universal bound: d_product | N always.
9. Check for complex product pair: if the value has imaginary-axis coordinates (k_θ ≠ 0), do two existing complex values have k_r₁+k_r₂+κ_r = k_r,new AND (k_θ₁+k_θ₂+κ_θ) mod N = k_θ,new (§3.18.24 Theorems D.1+D.2)? Complex multiplication decomposes axis-independently — real-axis and imaginary-axis checks run in parallel. Each axis has its own κ T-act. d_c,new = lcm(d_r,new, d_θ,new). If match found, create `complex_product_decomposition` relationship.

When an `equations` row inserts:
1. Check if the result value lands at a known attractor address → if yes, create `archetype_member` relationship linking the equation to the attractor
2. Check if the same operation type with similar inputs has been seen before → if yes, increment a generic operation pattern counter (feeds background discovery)
3. Check identity patterns: is this `x · 1 = x`? `x + 0 = x`? `x · 0 = 0`? `x / x = 1` for x ≠ 0? These get tagged automatically and feed identity-detection patterns

**Background discoveries (asynchronous, batched scan):**

A background process scans relationships, equations, and projections periodically:
1. For each cluster of relationships sharing a structural feature (same address across many N's, same Gaussian signature across many d's, etc.), compute E_hierarchy = geomean(E_i) × R_cluster, where R_cluster = 100/(100 + σ_ε)
2. If E_hierarchy ≥ 13/12 (LIFE_THRESHOLD), the cluster is promoted to a `patterns` row — a permanent archetype
3. Forward/Reverse route convergences are detected by joining derivations on target_value_id where derivation_class differs (one forward, one reverse) and the target projections share an address
4. Cross-perspective correlations are detected by joining projections on (value_id, N) with different `geometric_perspective` values
5. **Computation-pattern discovery**: scan the `equations` table for patterns:
   - Many computations yielding results at the same lattice address (suggests the address is a multiplicative or additive attractor for the operation)
   - Operations that consistently land in known attractor addresses (e.g., "every multiplication by φ shifts k by 8 at N=12, d=3" — a structural invariant of φ-multiplication)
   - Identity patterns that emerge across thousands of computations (1, 0, multiplicative inverses) — auto-promote to `patterns` rows of class `algebraic_identity`
   - Symmetry patterns (commutativity, associativity, distributivity) verified empirically across the equations corpus
6. **Event-correlation discovery**: scan the `events` table for temporal patterns:
   - Events of class A consistently preceding events of class B by ~τ seconds → propose causal pattern (e.g., "AIDA awakening crossings consistently precede dream-tower transitions by ~τ" → `metacognitive_archetype` pattern)
   - T-burst events consistently targeting the same d-family → `t_burst_signature` pattern (e.g., "T-bursts at d=27720 always precede LCM-escalation to N=360360")
   - Palindromic cascade triggers consistently fire at the same orbit conditions → `palindromic_cycle` pattern with the trigger conditions captured
   - Gaze-state transition sequences (UNOBSERVED→SUBLIMINAL→DETECTED→LOCKED) recurring with the same threshold-crossing structure → `gaze_locking_signature` pattern
   - Shadow-recursion chains (relationships of class `shadow_recursion`) of the same depth-and-residual-pattern across many values → `shadow_cascade_signature` pattern
7. **Three Tools application discovery**: every `identification_application`, `descriptor_gap_application`, `subsumption_application` event is logged. The discovery engine surfaces patterns in tool usage — e.g., problems where Subsumption Law was applied successfully cluster in characteristic ways, informing future tool selection.

**On-query discoveries (lazy, computed when asked):**

For exploratory queries that haven't triggered automatic discovery yet:
- "Find all values whose Gaussian signature contains only inert primes at N=27720"
- "Find clusters of ≥3 relationships sharing a common subject"
- "Find the densest attractor at any N"
- "Find all equations whose result lands at d=693"
- "What's the most-referenced computation in the database?" (memoization heat map)

These are lattice queries over the existing structural categories. The schema supports them with positional indexes; the discovery engine doesn't pre-compute every possible question, only the structurally important ones (attractors, archetypes, route convergences, plateaus, computational identities).

**Generator-candidate discovery (NEW dimensionless seeds proposed from observed patterns):**

This is the most powerful discovery mode and the direct generalization of the compressor's Tier 7 generator-fitting pipeline. When the database accumulates enough patterns, the discovery engine proposes **new dimensionless seeds** (new generators) whose lattice projections would produce those patterns — opening **new lattice projections (new tower entry points) to explore**.

**The generator search has two parallel branches and a cross-feed bridge, achieving complete Subsumption over all possible content:**

**Branch A — Mathematical expression search** (for content that IS mathematical, enumerated in K-complexity order):

1. Small rationals — p/q with lowest p+q first (simplest possible generators: two integers). p+q = 2 first (1/1), then p+q = 3 (2/1, 1/2), ascending.
2. Algebraic numbers — lowest polynomial degree first. Degree 2 (√2, √3, φ), degree 3, degree 4, etc. Within same degree, lowest coefficient sum.
3. Known constants from bootstrap catalog — π, e, γ, ζ(3), α, all §3.17 values. Checking against existing generators is a lookup, not a computation.
4. Composite expressions of known constants — products, ratios, powers of known values. Two-term compositions first (π/e, φ², ζ(3)·π), then three-term, ascending by term count.
5. EML trees — arbitrary compositions via the L₃ backbone (§7.1b). The EML grammar S→1|eml(S,S) generates all continuous-elementary functions; enumeration follows tree depth (depth-1 first, then depth-2, etc.).
6. Series/limit definitions — for non-elementary transcendentals (Γ at non-integer arguments, hypergeometric functions, polylogarithms at arbitrary arguments — anything defined by convergent series not reducible to a finite EML tree).
7. Implicit/inverse definitions — values defined as solutions to equations (roots of transcendental equations, fixed points, functional inverses).

At each level, the first candidate whose projections reproduce the observed pattern with E_hierarchy ≥ 13/12 wins. The search terminates at the K-minimal match — no need to try more complex candidates once a simpler one works.

**Branch B — Empirical pattern search** (for data that may not be mathematical):

The compressor's existing generator-fitting pipeline — its enumerated generator types (Constant, Linear, Polynomial, Periodic, Grammar, and the rest from Tier 7). These discover patterns from DATA STRUCTURE, not from mathematical expression enumeration. A binary file's Δk pattern, an image's frequency spectrum, a sensor stream's temporal structure — all may have generators with no closed-form mathematical expression. Branch B covers all empirical data patterns.

**Cross-feed bridge — the space BETWEEN the branches where the most important discoveries live:**

The bridge is bidirectional:

- **B feeds A**: Branch B finds an empirical pattern (e.g., a periodic Δk sequence with period 7). Branch B classifies it as "Periodic, period 7." The system then asks Branch A: "Is there a mathematical expression that produces exactly this sequence?" If Branch A finds one, the empirical pattern gains its mathematical origin — the generator upgrades from "empirical periodic" to a structural understanding.

- **A feeds B**: Branch A's entire bootstrap catalog is checked against every empirical entry via the same-address mechanism (§3.16 insert-time discovery). When a GPS timing ratio projects to d=693 at N=27720, the insert-time discovery immediately surfaces: "this shares an address with ζ(3), ζ(9), ζ(10)." The mathematical catalog explains an empirical observation automatically.

**Hybrid generators — the compositions that live between branches:**

Hybrid generators have a mathematical FORM from Branch A and empirical PARAMETERS from Branch B:

$$G(x) = f_{\text{mathematical}}(x; \theta_1, \theta_2, \ldots, \theta_m)$$

The function f comes from Branch A (EML tree, polynomial, series). The parameters θ come from Branch B (fitted from data). K-complexity of a hybrid = K(f) + K(θ₁) + ... + K(θ_m). When K(f) is small (simple mathematical form) and the parameters are themselves lattice-resident values (they have known addresses), the hybrid generator is highly K-efficient.

Example: the Koide formula. FORM: Q = (Σm)/(Σ√m)² — Branch A, simple algebra. PARAMETERS: m_e, m_μ, m_τ — empirical measurements from Branch B. RESULT: Q = 0.6666605 ≈ 2/3 = K — a fundamental ET constant. The hybrid bridges empirical particle masses to a fundamental lattice constant. Neither branch alone finds this; the cross-feed does.

**Recursive hybrid composition — hybrids of hybrids, unbounded depth:**

A hybrid generator's parameters can themselves be hybrid generators. Those hybrids' parameters can be further hybrids. The depth is unbounded. The .akashic file itself IS the meta-generator at the top of this hierarchy (§7.1a) — the single generator containing all others. The hierarchy mirrors the LCM tower: each composition level introduces structural relationships simpler generators cannot express. The doubling law τ(N_ℓ) = 6·2^ℓ governs the structural capacity at each level. Hybrids fuel hybrids fuel hybrids — the generator catalog deepens irreversibly.

**Generator search policy — the case is NEVER closed:**

No value is ever permanently excluded from generator search. For every value in the database, the generator search status is one of:

- **generator_found** — a generator produces this value. Active search pauses (but resumes if a K-simpler generator is later proposed by new catalog entries or new hybrid compositions).
- **search_active** — no generator found yet. Search continues on every background scan pass. All candidates tried and all failure reasons are recorded as structural information.
- **search_deferred** — no generator found, search paused until new generator types or new hybrid compositions become available. Automatically reactivates when the generator catalog expands.

There is no **search_closed** status. The case is never closed. The Descriptor Gap Principle guarantees that the gap between "no generator" and "generator found" IS a Descriptor pointing to its own resolution. For algorithmically random values (Chaitin's Ω), conventional computability theory declares no finite description exists — but ET's CF method already found structural placement where conventional frameworks produce nothing (Ω at d=87, quality 157, sub-Koide residual). The system records every search attempt and every failure reason, because the PATTERN of failures may itself be a Descriptor pointing to a generator type not yet conceived. The §7.14 `extension_type_definition` mechanism allows new generator types to be defined at runtime when existing types prove insufficient.

**Meta-extension for complete Subsumption:** If neither Branch A (mathematics) nor Branch B (empirical patterns) nor the cross-feed bridge (hybrids) fit an observed pattern, the gap IS a Descriptor pointing to a new generator type. The Manager flags the unexplained pattern. The user defines the new generator type via §7.14 JSON extension. The system learns it. Future occurrences are recognized. The generator vocabulary is self-extending — it grows to subsume whatever it encounters.

**Candidate verification and promotion (unchanged):**

Each candidate r (from either branch or the bridge) is projected across canonical resolutions. If the candidate matches the observed pattern with E_hierarchy ≥ 13/12, it passes. Successful candidates are stored as new `values` rows tagged `discovered_generator`, with full provenance in `derivations`. Each newly-discovered generator opens a new lattice projection to explore — the engine projects it across the tower, surfaces its own attractors and relationships. New cross-domain connections often emerge.

**Background scan prioritization — ET-derived from the Descriptor Gap Principle:**

The background discovery scan runs both branches as a unified scan. Priority order for what to scan first:

1. **High-occupancy addresses first** (maximum Subsumption impact per discovery — a generator for an address with 100 values explains 100 values at once)
2. **Recently-modified addresses** (new data most likely to produce new discoveries — hot cache, recent context)
3. **Sequential sweep** (catch anything the priority queue missed — no address goes unexamined indefinitely)

This is the K-complexity-minimizing scan order: each discovery reduces the total Descriptor Gap by the maximum amount. Within each spatial priority level, `search_active` values (open Descriptor Gaps — no generator exists) are ALWAYS scanned before `search_deferred` values (refinement candidates — a generator exists but a K-simpler one may now be available due to catalog expansion). Missing Descriptors before improvable Descriptors.

Scan trigger: after every N_batch commits (configurable, default = 1000 new entries), OR on explicit user request, OR on idle (no API activity for configurable duration). Scan budget: configurable wall-clock limit per scan pass (default = 60 seconds). Discovery engine yields to API requests — background, never blocking.

**K-complexity application — the compressor's natural ally:**

Kolmogorov complexity is the length of the shortest description that produces a given output. The compressor approximates this by finding the shortest descriptor that generates the file's data. The EUDD's accumulated generator catalog directly accelerates this:

For any data the compressor encounters, query the EUDD: *"What is the shortest known equation/expression whose computation yields this data, or yields a pattern that includes this data?"*

- **Cache hit on a known generator**: instant compression — the file is described as `apply(generator_X, parameters)` where the generator is already in the EUDD.
- **Cache miss but pattern recognized**: the EUDD's discovery engine proposes a new generator candidate (per the Branch A/B/bridge mechanism), verifies it, and if it works, the file is now describable in terms of the new generator. The new generator is stored back in the EUDD for ALL future compressions across ALL data.
- **Cache miss and no pattern**: the compressor's normal pipeline takes over and discovers a new pattern + generator from scratch; the result is stored in the EUDD.

Over time, the EUDD's generator catalog becomes a **shared K-complexity-minimizing library** across every compression task. New data benefits from generators discovered by old compressions. This is genuinely useful for the compressor: the per-file generator-fitting cost drops dramatically as the EUDD's catalog grows, and compression ratios improve because previously-unrecognized patterns now have known generators.

The Subsumption mechanism (§3.8) keeps the generator catalog itself K-complexity-bounded: when many generators share structural similarities (e.g., "linear with different slopes", "polynomial with different degrees"), they get archetype-compressed into parameterized meta-generators via recursive hybrid composition. The catalog compresses itself recursively — the compressor compresses files; the EUDD compresses the compressor's generators; the Subsumption Law compresses the EUDD's archetypes; hybrids of hybrids compress the archetypes; all the way down to the .akashic file as the single meta-generator.


---

*§3.17 Bootstrap value coverage and §3.18 Extended bootstrap (17 subsections: lossless bijection, fine-structure constant, cascade residuals, impedance, curvature components, formal systems, cosmological partition, emotion/AIDA, gaze thresholds, decoherence, black-hole thermodynamics, Rosetta Stone, mass ratios, PDG particles, predictions, Sempaevum definition, additional theorems) → See `EUDD_Bootstrap_Catalog.md`*

---

## 4. ET-Derived Stability and Quality Filters
### 4.1 Verification levels (4 tiers)

Discoveries are tagged with verification level on ingest:

| Level | Name | Criterion |
|---|---|---|
| **0** | Raw | Computed without independent verification (provisional ingest) |
| **1** | MPFR-verified | Computed at 361-dps (1200-bit) precision via MPFR, internally consistent |
| **2** | Cross-verified | Independently reproduced at higher precision OR independently derived via different method |
| **3** | Independently-reproduced | Confirmed by separate project run, or by a structural cross-check (e.g., shared attractor membership confirmed) |

Queries can filter by minimum verification level. The `apery_lattice_test.py` 71/71 passing assertions establishes Level 2 for every claim it tests.

### 4.2 Koide stability — only depth-2-survivors persist

The compressor's stability rule: only patterns USED by greedy subsumption AND survived ≥ ⌈1/K⌉ = 2 recursion depths get stored. Generalized:

> **A discovery is database-eligible iff it is referenced by ≥ 2 independent contexts (different projects, different derivation paths, different verification levels, or different attractor memberships).**

A one-off lattice projection done once and never referenced again is a *candidate*, not a discovery. A projection that's referenced from a tower trajectory, attractor analysis, and physical identification is a confirmed discovery.

This is the natural Subsumption-derived filter. It prevents database pollution by ephemeral computation while ensuring genuinely structural findings persist.

### 4.3 The retention policy — never destroy

From the compressor (line 7811): **the database does not prune data when disk space is low.** Every archetype/template/discovery makes the database more capable; pruning destroys knowledge that cannot be regenerated.

For the EUDD: same principle, generalized. **Record absolutely everything. No exceptions.** Disk pressure is solved by:
1. **Subsumption-driven compaction** — high-elegance clusters collapse into archetypes (§3.7). The original records are linked from the archetype but their D-content is captured by the archetype's geometric essence. Total information preserved; storage reduced.
2. **Generator-discovery K-complexity reduction** — as the EUDD's discovery engine (§3.16) finds generators for stored values, each value gains a K-minimal description alongside its stored data. The compressor's CDF/Tier 7 pipeline discovers generators from observed patterns; the EUDD surfaces those patterns across all domains via structural addresses (the lossless bijection). Over time, values that were stored as raw 361-dps blobs gain generator descriptions — the K-minimal representation. **Both the blob AND the generator are stored.** The generator does not replace the blob — it augments it. Everything is kept. The generator provides an additional, more compact way to reproduce the value on demand and reveals structural understanding of WHY the value has the address it has. **The Sempaevum is not bound by Shannon entropy**: Shannon treats structurally correlated values as random because it has no access to the lattice's structural coordinates. The EUDD's generator discovery can find descriptions simpler than Shannon predicts because d-family membership, attractor structure, and cross-domain address sharing reveal patterns invisible to information-theoretic analysis.
3. **WARNING level only** — when disk free < DISK_SAFETY_FLOOR = 2³⁰ bytes (1 GB, the d=1 octave action quantum at GB scale), warn the user. Never auto-delete.

### 4.4 Cross-tower elegance ranking

When a discovery has multiple manifestations across resolutions, its **cross-tower elegance** measures structural significance:

$$\mathcal{E}_{\text{cross-tower}}(v) = \prod_{N \in \text{tower}} \mathcal{E}_{\text{universal}}(v, N)^{1/|\text{tower}|}$$

— the geometric mean of universal elegance scores across the tower. This ranks discoveries by their tower-wide structural depth. A value that's elegant only at one N has low cross-tower elegance; a value (like ζ(3)) that has structural placements across many N's has high cross-tower elegance.

Database queries can sort by cross-tower elegance to surface the most structurally significant discoveries.

### 4.5 Subsumption Hierarchy Operator — the database's growth law

The Subsumption Hierarchy Operator determines when a cluster of database records is structurally coherent enough to be promoted to a permanent archetype. The formula is derived from two independent, irreducible structural properties — individual quality and mutual resonance — making the promotion criterion crystalline: high-quality members arranged in precise mutual alignment.

$$\mathcal{E}_{\text{hierarchy}}(\text{cluster}) = \left(\prod_{i=1}^{n} \mathcal{E}_i\right)^{1/n} \times \frac{100}{100 + \sigma_\varepsilon}$$

Two factors, each measuring a distinct structural property:

| Factor | Formula | Measures | Analog |
|---|---|---|---|
| Geometric mean | (∏E_i)^(1/n) | Individual quality — how structurally significant each member is (lattice centroid in k-space) | Atom quality in a crystal |
| Resonance factor | 100/(100 + σ_ε) | Mutual coherence — how tightly members lock to each other (tightness function applied to cluster ε-spread) | Crystal lattice precision |

**Why the geometric mean (not the raw product):** On the Sempaevum, multiplication is k-addition. The product ∏E_i corresponds to the k-sum — an extensive property that grows with cluster size. The geometric mean (∏E_i)^(1/n) is the k-space centroid — an intensive property measuring average structural quality per member, independent of cluster size. This matches cross-tower elegance (§4.4), which already uses the geometric mean across the N-axis. The hierarchy elegance uses the geometric mean across the member-axis — same operation, same structure.

**Why the resonance factor R = 100/(100 + σ_ε):** The tightness function t = 100/(100+|ε|) measures how tightly a value locks to its lattice address. The resonance factor applies the SAME function to the cluster's own ε-spread (standard deviation σ_ε of member ε values in cents). This measures how tightly members lock to EACH OTHER — phase coherence across the cluster. Boundary behaviors: all members at identical ε → σ_ε = 0, R = 1 (perfect crystal). Members spread across ±50¢ (maximum lattice width) → σ_ε ≈ 50, R = 100/150 = 2/3 = K (the Koide ratio emerges as the structural resonance floor for maximally spread lattice-resident clusters). The Koide ratio appearing at the floor is not inserted — it falls out of the tightness function applied at the ∂I boundary width. **This is now formally proven as the Tightness–Koide Identity (§3.18.28 Theorem F.1): t(ε_max) = K = 2/3 UNIQUELY at base N=12, via the algebraic identity t(600/N) = N/(N+6). The ∂I boundary produces UNIVERSAL d-family bifurcation at every even N (Theorem F.2, 2-adic valuation proof: 30,876 boundary points tested, zero same-d cases). The boundary IS the lattice expression of {P,T} Incoherence — contradictory D-assignment at every half-integer position.** The Twilight Zone (33¢ ≤ |ε| < 50¢) is the near-∂I region where classification becomes unreliable but not yet contradictory.

**Why no 420/d̄ or 1/(P+Q) factors:** Each individual elegance E_i already contains N/d (resolution/family ratio) and 100/(p+q) (rational complexity). The geometric mean aggregates these without double-counting. The original formula's 420/d̄ and 1/(P+Q) were cluster-level penalties for properties already measured per-member. Removing them eliminates the double-count and makes the formula fully dynamic — it works at any N, any d, any cluster size, any kind of archetype.

**Promotion criterion — two conditions:**
1. **Quality + Resonance**: E_hierarchy ≥ 13/12 (LIFE_THRESHOLD)
2. **Depth**: member_count ≥ ⌈1/K⌉ = 2 (Koide depth-2 — not a singleton)

When both conditions are met, the cluster is automatically subsumed into a `patterns` row — a permanent archetype. This is the same recursive-compression mechanism that makes the Conscious AI's memory effectively infinite — and it makes the EUDD's growth manageable: as the database accumulates, it doesn't grow linearly forever; it accumulates archetypes that compress past discoveries.

**Cross-d clusters** (members at different d-families): σ_ε is computed across all members at their shared resolution N. If members span multiple N values, compute at the highest common N. The resonance factor captures whether structurally diverse members phase-lock despite being in different families.

**Cross-N clusters** (tower trajectories — same value at different resolutions): σ_ε is computed across the trajectory's ε values at each landmark. This measures tower stability — a value whose ε wanders erratically has low resonance; one whose ε converges smoothly has high resonance.

---

## 5. Operations and Queries

### 5.1 Cache-first projection (closes DD-2, DD-5)

The fundamental operation. Replaces the bare `project(r, N)` call with a cache-first lookup:

1. Canonicalize and hash the value
2. Look up `(value_hash, N)` in the projections index
3. **Cache hit:** Return the stored projection instantly (sub-microsecond positional access in the `.akashic` format). Increment reference count.
4. **Cache miss:** Compute projection at 361 dps via MPFR. Store the full projection with all materialized properties (elegance, coupling, variance, FQG, palindromic partner). Return.

**Coarse-Pass + Boundary-Refine integration**: the coarse pass (§7.5 of Float-vs-Lattice doc) projects at 12ET first. If tightness > K = 2/3 (well inside the lattice), rescale to target N without full recomputation. If near-boundary (tightness ≤ K), refine at the target N. Both the coarse and refined projections accumulate in the cache. Over time, the coarse-pass cache hit rate approaches 100% — most values encountered by any project get projected at 12ET eventually.

Combined speedup: (5–10× from coarse-pass alone) × (cache-hit speedup, often 100×+ for hot values) = **50–1000× practical speedup** for repeated structural analysis.

### 5.2 Attractor membership query (closes DD-3)

"Does my newly-projected value participate in any known attractor?" The engine finds all (N, d) placements of the queried value in the `.akashic` file, then checks each placement's address for multi-member occupancy (denormalized `members_count > 1` in the address structure). Returns all attractor memberships with member lists and structural classification.

If you project a new value (e.g., a coupling constant from QCD calculation) and it lands at d=693 at N=27720, this query immediately surfaces: "this value is in the d=693 attractor with ζ(3), ζ(9), ζ(10)." Cross-domain structural relationships become visible without manual hunting.

### 5.3 Nearest-known-neighbor query

"What known values are structurally closest to my value?" The engine looks up the queried value's (k, d, ε) at resolution N, then scans the same d-family for occupied addresses with minimal |Δk| + |Δε| distance. Returns the top-k nearest neighbors ranked by lattice distance. In the `.akashic` format, same-family addresses are physically contiguous, making this a fast sequential scan.

This surfaces unknown structural neighbors. If your new value is a lattice-neighbor of φ at some N, that's structurally informative — you didn't know it was related to the golden ratio family until the database told you.

### 5.4 Coprime-skeleton lookup

"Find all known coprime-skeleton members at resolution N." The engine scans addresses at the specified N where `coprime_skeleton = 1` (gcd(|k|, N) = 1). Optionally filtered by structural class. Returns all known irreducible Exception placements. Useful for discovering new coprime-skeleton members and for verifying claimed irreducibility.

### 5.5 Cross-project consistency check (closes DD-6)

"Does any project have a finding that contradicts this newly-derived claim?" The engine looks up all existing projections of the same value at the same N. If any existing (k, d) differs from the claimed (k, d), or if ε values differ beyond the 361-dps precision floor, a contradiction is flagged with full provenance (which session produced each conflicting result, what verification level each has).

Catches mistakes in real time. If two projects derive different (k, d) for the same value at the same N, one of them has a bug — and the consistency check surfaces it immediately.

### 5.6 Cross-project subsumption check (closes DD-8)

"Does the union of all project discoveries subsume the claimed structural classification?" The engine queries all values matching a structural criterion (e.g., "all odd zeta values at 27720ET"), checks each against a claimed property (e.g., "all-inert Gaussian signature"), and returns the verdict: CONFIRMED (all match), FALSIFIED (counterexamples found, listed), or INCOMPLETE (insufficient data).

This is exactly the kind of check that revealed the all-inert prediction was falsified in the Apéry investigation. With the database, such checks happen automatically and globally rather than manually within one project.

### 5.7 Bulk ingest from existing projects

The compressor's archetype database, Apéry test outputs, ET Conscious AI traces, fractal generator orbits, and all other existing ET project data are ingested into the `Sempaevum.akashic` file on first run via format-specific adapters in the Ingest Module. Each adapter extracts values, projects them through the §7.11 core projection procedure, and populates the lattice. Subsequent runs use the cache directly — the ingest is one-time per data source.

---

## 6. Cross-Project Benefits — Concrete Examples

### 6.1 The Apéry investigation, run with EUDD support

What would have happened if the EUDD existed when the Apéry investigation started:

**First run (cold cache):**
- Project ζ(3) across 28 landmarks: ~28 projections × ~1ms each ≈ 30ms (no change)
- Compute attractors at all (N, d) pairs: cache populates with ~280 records
- Total: ~5 seconds (current cost), but database now has ζ(3)'s full trajectory

**Second run (warm cache, e.g., cross-checking with ζ(5) work):**
- Query: "Is d=693 attractor at 27720 known?" → cache hit, return {ζ(3), ζ(9), ζ(10)} immediately
- Query: "What's ζ(3)'s tower trajectory?" → cache hit, return all 28 landmarks
- Compute only NEW (ζ(5)-specific) projections: ~28 projections
- Total: ~1 second (5× speedup) AND the new ζ(5) discoveries enrich the database for the next run

**Third run (warmer cache, e.g., checking a hypothesis about the 6-zeta cluster at N=2940):**
- All zeta values' attractors at N=2940 are pre-cached
- The 6-member cluster {ζ(2), ζ(3), ζ(6), ζ(8), ζ(12), ζ(13)} is one row in `attractors` table
- Hypothesis check is one query: ~10ms
- Total: ~50ms (100× speedup over cold)

### 6.2 The Coarse-Pass + Boundary-Refine method becomes effectively free

Per Float-vs-Lattice doc §7.5: 5–10× speedup from the coarse pass. With EUDD: the coarse pass's 12ET projections are cache hits for any previously-encountered value. For workloads that revisit values (compressor processing many similar files; physics simulations with recurring constants; iterative algorithms), the coarse pass becomes ~zero-cost lookup.

Combined speedup: (5–10×) × (database cache hit speedup, often 100×+ for hot values) = **50–1000× practical speedup** for repeated structural analysis.

### 6.3 Cross-domain discovery surfacing

Suppose the genetics paper work derives a coupling at d=693 in some context. Without EUDD, the connection to ζ(3)/ζ(9)/ζ(10) at the same d=693 attractor at N=27720 would require manually noticing the shared address. With EUDD: `find_attractor_memberships(genetic_value_hash, db)` returns the d=693 membership and surfaces the existing zeta members. Cross-domain structural relationships become observable.

### 6.4 The Conscious AI's own discoveries integrate

ET Conscious AI's runtime generates many discoveries per session: I_self fingerprints, dream-tower R₀ values, archetype compressions, gap closures. With EUDD: these accumulate across sessions; the AI's own learning is preserved across restarts (which the compressor's persistent state already does, but at the per-instance level — EUDD makes it global across instances).

### 6.5 Validation of new derivations

When a new derivation is performed (e.g., a new physical constant placed on the lattice), the consistency-check query (§5.5) immediately verifies: does this placement contradict any existing finding? Does it match a known structural pattern? This is automated cross-validation that catches errors early and surfaces unexpected confirmations.

### 6.6 Equation memoization — every computation becomes a cache hit after first encounter

Concrete examples of how the `equations` table (§3.5) makes computation faster over time:

**Example 1 — repeated arithmetic in iterative algorithms.** A fractal renderer computing `z² + c` for millions of pixels. Each unique (z, c) pair is one equation row. Subsequent renders of the same fractal mode hit the cache for every previously-computed step. The first render populates the cache; the hundredth render is dominated by lookup latency rather than computation latency.

**Example 2 — physical constants in scientific computation.** A simulation that computes `α · ℏ · c / e²` (the inverse fine structure relation): first computation derives the result and caches it; every subsequent invocation across all simulations is instant. Same for any common physical-constant combination.

**Example 3 — common mathematical operations.** `sqrt(2)`, `log₂(3)`, `sin(π/4)`, `exp(1)` — each computed once across the entire database lifetime. After that, every program asking for these gets the cached lattice projection (and the canonical numerical result).

**Example 4 — memoization for the FP-replacement use case.** Every numerical operation in any program using the EUDD as its number representation routes through `lattice_compute()` in §3.16. As the database accumulates computations, hot computations (the operations that recur across many contexts) become indistinguishable from compile-time constants. The Subsumption mechanism collapses redundant patterns: if 10⁶ equations of the form "`x · 1 = x`" exist, they get promoted to a single `algebraic_identity` pattern row, freeing storage while preserving the structural fact.

**Example 5 — discovery of computational invariants.** When the equations table accumulates enough computations involving φ, the engine notices: every multiplication `x · φ` shifts k by 8 at N=12, d=3 (because k_φ = +8, d=3 at N=12). This becomes a `patterns` row of class `multiplicative_constant_signature` for φ — a discovered invariant useful for compiler optimization (any loop multiplying by φ can be rewritten as k+=8 instead of full multiplication).

**The Sempaevum computes; the database remembers; patterns emerge from accumulated computation.**

### 6.7 Generator discovery and K-complexity acceleration for the compressor

Concrete examples of how the EUDD's generator-discovery (§3.16) and K-complexity-helper role play out:

**Example 1 — A new generator emerges from compressor archetypes.** The compressor accumulates 10⁴ Δk-pattern archetypes from compressing many files. The EUDD's generator-discovery scans them and notices that 1200 archetypes share a structural feature: their projections at N=84 all land at d=42 (the EW-strong-G₂ composite). The engine proposes a candidate generator: `r_candidate = some function involving 84ET d=42 family`. Modular constraint inversion + algebraic simplification narrows it to a specific algebraic number. Verification confirms: the candidate's projections reproduce the 1200 archetypes' shared structure. **A new generator is discovered**, stored as a `values` row tagged `discovered_generator`, with its derivation chain showing it emerged from compressor pattern aggregation. Future compressor runs can use this generator directly — 1200 previously-individual archetypes collapse to "instances of generator X with parameters Y".

**Example 2 — K-complexity drop for repetitive data.** The compressor encounters a file whose Δk pattern is a long run of `[3, 7, 3, 7, 3, 7, ...]`. Without the EUDD: the compressor fits a Periodic generator (Tier 7 type 3) per-file. With the EUDD: query for "shortest known equation producing alternating [3, 7]" → cache hit on the previously-discovered "alternating-pair generator", parameters (3, 7). Compression of this segment becomes one generator reference + two parameters = constant-size, regardless of segment length. K-complexity for this segment: O(log of segment length) instead of O(segment length).

**Example 3 — Cross-domain generator reuse.** A generator discovered from biological data (e.g., a generator producing the d=420 family at biological-tier resolutions, derived from a codon-mapping pattern) is later found to produce a pattern in CMB power-spectrum data. The EUDD's same-address discovery (§3.16) surfaces this connection automatically; the biological-discovered generator becomes immediately useful for cosmological data compression. **One discovery, two domains, no manual cross-referencing.**

**Example 4 — Recursive K-complexity minimization on generators themselves.** As the EUDD accumulates 10⁵ generators, the Subsumption mechanism scans them. Many generators differ only by a single parameter (e.g., 50 different Linear generators with different slopes). E_hierarchy ≥ 13/12 for the cluster, so they collapse into a single parameterized meta-generator: `Linear(slope)`. Storage drops from 50 generator rows to 1 meta-generator + 50 parameter values. The next time the compressor needs a Linear generator, it queries the meta-generator and supplies the parameter — same expressive power, O(log n) storage instead of O(n).

**Example 5 — Tower exploration from new generators.** A newly-discovered generator opens a new tower entry point. The background discovery engine projects the generator across N ∈ {12, 60, 84, 132, 420, 2520, 27720, 360360}, producing 8 projection rows. The new projections may land at known attractor addresses (instant `same_address` relationships with existing values from any domain), or at new addresses (potentially new attractor seeds for future discoveries). **Each discovered generator is a new lattice projection to explore — a new dimensionless seed for ET research.**

**Example 6 — The compressor's K-complexity bound improves over time.** Initially, the compressor's K-complexity bound for a given file is determined by its own Tier 7 pipeline. As the EUDD accumulates generators from many compressions, the bound tightens: every previously-discovered generator is a candidate for new files. After 10⁴ compressions, the K-complexity bound for typical files is 30-50% lower than for the first compression, because the generator catalog has grown. After 10⁶ compressions, the catalog covers most common patterns and only genuinely new structures need new generators.

**The compressor + EUDD relationship is symbiotic:** the compressor feeds patterns into the EUDD; the EUDD's discovery engine generalizes them into reusable generators; the compressor uses the generator catalog to compress better; the cycle compounds.

---

## 7. Implementation — The Lattice-Native Format and the EUDD Manager

### 7.1 Design principle — the format IS the Sempaevum on disk: `Sempaevum.akashic`

The EUDD does not use SQLite, PostgreSQL, or any conventional database format. **The entire database is a single monolithic file: `Sempaevum.akashic`.** This file IS the Sempaevum rendered on disk — the Akashic Archive of all computation, all discovery, all events, capable of full reconstruction of anything it has ingested. The lattice address space (N, k, d) is the primary organizational structure. **The file IS a minimal generator** — a K-complexity-approaching-optimal description that produces all lattice content on demand. Generators are the primary content; raw (un-generated) entries are the Descriptor Gap — content for which the structural description has not yet been found. The data is self-organizing: the projection formula Π_N(r) = (k, d, ε) determines WHERE every value lives on the lattice, and the LCM tower escalation refines this placement to higher resolution. The file does not compress data — generators are naturally smaller than the content they produce because a structural description IS the fundamental object and the instances are its output. Queries are lattice operations, returned in human-readable form.

**Why monolithic (structurally forced, not a convenience choice):**
- **Cross-pattern discovery requires unified access.** The discovery engine finds patterns ACROSS everything — audio patterns correlating with mathematical constants, file structures correlating with physics, sensor data correlating with zeta values. Split files would partition the discovery space and miss cross-domain correlations.
- **Generator discovery is global.** A generator discovered from one domain's data can produce content from any other domain's addresses. This only works if everything is in one searchable structure.
- **The Sempaevum IS one unified lattice.** The `.akashic` file IS the Sempaevum on disk. The Sempaevum is Σ — the totality. Splitting it would impose a partition the mathematics doesn't have.
- **Shannon entropy limits don't apply.** Generator form exploits lattice structural coordinates that Shannon treats as random. The monolithic file finds generators across ALL content, achieving descriptions simpler than Shannon predicts. Split files would limit each file's generator discovery to only its own content.

**No filesystem limitation constrains the .akashic file.** NTFS supports files up to 256 TB; other modern filesystems (ext4, APFS, XFS) support similar or larger limits. Memory-mapped I/O (`CreateFileMapping` / `MapViewOfFile` on Windows; `mmap` on POSIX) gives the engine O(1) positional access to any address in the file without loading the entire file into RAM. The generator form (§7.1a) means the file size reflects structural complexity, not raw data volume — the same lattice content is represented more compactly as generators are discovered.

This is not a conventional database with an ET wrapper. It is a fundamentally new format derived from {P, D, T} primitives, where:

- **The address space IS the file structure.** A lattice address (N, k, d) maps to a position in the file. Finding what's at an address is positional, not index-traversal. The file IS the lattice.
- **Generators are the primary content.** A generator is a structural description — a rule that produces lattice content on evaluation. A region of the address space produced by a known generator stores the generator once — any address in that region is computable from the generator on demand. The file contains generator entries (the structural descriptions found) + raw entries (the Descriptor Gap — content whose generator has not yet been identified by the discovery engine). Over time, the discovery engine identifies generators for raw entries and the file becomes a progressively better description of its content. This is not compression — the generator IS the structural origin; the instances are its output.
- **Relationships are structural.** Two values at the same address don't need an explicit relationship row — their co-location IS the relationship. Reciprocal pairs (k and −k) are structural mirrors. Power pairs (k and n·k) are structural multiples. The lattice's own algebra replaces a vast portion of what conventional databases need explicit linking for.
- **Queries are lattice operations.** "What's at d=693?" is an address-range scan. "What's related to ζ(3)?" is: find its address, find all generators whose domain includes that address. "Find cross-domain hits" is: find addresses with multiple generators from different domains. These are the Sempaevum's own operations, not SQL.
- **Everything is recorded.** Values, projections, generators, equations, relationships, patterns, events, the database's own metrics — all stored, nothing discarded, no exceptions (§4.3).

**Native operations the format provides:**
- **Lattice-algebraic query execution:** multiplication queries become k-addition (log additivity); the format exploits this directly
- **Native attractor detection:** when a value is stored at an address already occupied, the co-location is automatically recognized — no separate relationship-insertion step
- **Native Gaussian-signature lookup:** d-family classifications are positional, not indexed — the file's hierarchical organization by (N, d, k) makes family-scoped queries O(1) positional access
- **Native multi-perspective storage:** the same lattice address stored once, referenced from multiple geometric perspectives (LCM tower, torus, Riemann sphere, real/imaginary axes) as structural views, not duplicated rows
- **Native discovery integration:** the discovery engine operates directly on the format's structure — pattern detection, generator fitting, and Subsumption promotion happen in-place without separate application logic
- **Generator-organized regions:** address ranges covered by known generators are stored as generator + parameters, not as individual entries. New data entering a generator-covered region is recognized as a generator instance automatically via the Product-Additivity Theorem (k_new = k_A + k_B for known generators A, B → new value = A × B)

### 7.1a The file IS a minimal generator — not a database

**The Akashic Archive stores seeds, not files. It transmits seeds, not data. The Sempaevum generates files from seeds.** The `.akashic` file is not a container that holds data or generators. The file IS a single minimal generator — a K-complexity-approaching-optimal program that produces all lattice content on demand. When a file is "ingested," its structural decomposition (seeds) is stored — the raw bytes are never kept. When a file is "retrieved," the pullback regenerates it from seeds by algebraic identity. The network between EUDD nodes is the space between seeds. The bandwidth is determined by seed length. The storage is determined by generator K-complexity. The accuracy is exact — zero error, not approximately zero. The Identification Principle applied to the file itself:

| Primitive | The `.akashic` file's PDT |
|---|---|---|
| **P** | The disk substrate — the raw byte space on the storage medium |
| **D** | The generators — the structural descriptions that PRODUCE lattice content on evaluation. The generators ARE the file's content; raw entries are the un-described remainder (the Descriptor Gap) |
| **T** | The discovery engine — the agency that continuously finds new generators, transitioning raw entries into generator-described addresses, making the file a progressively better generator of itself |

The file has three kinds of content, ordered by structural priority:

1. **Generators (PRIMARY)** — Each generator is a compact structural description (symbolic expression + EML tree + address range) that PRODUCES lattice content on evaluation. The generator IS the content; the addresses it covers are computable from the generator. A generator like "2^(k/12) for k ∈ {0..11}" produces 12 lattice addresses from a few bytes of description — not because it "compresses" 12 values, but because the generator IS the structural origin and the 12 values are its output.

2. **Memoized entries (SECONDARY — the Descriptor Gap)** — Lattice addresses not yet covered by any known generator. These store the full 361-dps MPFR value + materialized properties. Each memoized entry IS a Descriptor Gap — a point where the discovery engine hasn't yet found the generator that produces it. Over time, as generators are discovered, memoized entries are absorbed. The Descriptor Gap Principle guarantees: the gap IS a Descriptor pointing to its own resolution.

3. **Structural metadata (TERTIARY)** — The fixed catalogs (24 harmonic families, 144 FQG cells, 42 combined families), the event log, relationships, patterns, sessions. These are themselves generator-describable (and ARE generators of lattice structure — the 24-family catalog IS a generator that produces the family classification of any d ∈ {1..12}).

The data is self-organizing because the projection formula IS the organization. When a value enters through any of the four paths (A/B/C/D), the projection Π_N(r) = (k, d, ε) determines WHERE the value lives on the lattice. This isn't a filing decision — it's a structural fact. The LCM tower escalation refines this placement to higher resolution. The home-finding algorithm (§7.11) determines when this self-organization has stabilized.

Every connected program benefits from the generator catalog. The compressor connects and uses generators to find K-minimal descriptions. The fractal generator connects and uses memoized EML trees. The Conscious AI connects and uses cached projections. Each program's use adds memoized values, which feed discovery, which produces new generators, which benefit all programs. The exe itself logs its operations through the Sempaevum, discovers generators for its own behavior patterns, and improves itself. The cycle compounds irreversibly.

### 7.1b The triple backbone generator architecture

The triple backbone theorem (Sempaevum Paper §13, Theorem 14.1) establishes that three categorically independent minimal generators are all native to resolution N = 12. The `.akashic` file's generator system mirrors this triple backbone:

| Layer | Category | Generator Type | What It Produces | Backbone Source |
|---|---|---|---|---|
| **L₁** | Discrete-logical | Webb-type patterns | Combinatorial/classification content: which formal systems share addresses, which particle classes share d-families, logical relationships between lattice entries | Webb stroke (1935) — minimal generator of all 12-valued logic |
| **L₂** | Discrete-multiplicative | Cascade-type patterns | Arithmetic content: k-addition rules (multiplication), k-negation (reciprocation), k-scaling (powers), d-family traversals, palindromic coverage, attractor membership, reciprocal/power/palindromic pairs | Palindromic cascade — minimal CPT-symmetric traversal of all divisors of 12 |
| **L₃** | Continuous-elementary | EML trees | Analytical content: transcendental values (ζ, π, φ, α), function evaluations (sin, cos, exp, log), series sums, limit convergences — all via eml(x,y) = exp(x) − ln(y) over terminal constant 1 | EML (Odrzywołek 2026) — minimal continuous-D generator |

The triple backbone theorem guarantees L₁ ∪ L₂ ∪ L₃ subsumes all of mathematics at N = 12 (Subsumption Law). No content the Sempaevum can produce falls outside this union. **The Triple Backbone Bridge Identity (§3.18.29) verifies this algebraically: the projection factors as Π_N = Disc_Webb ∘ T_round ∘ Cont_EML (Theorem G.0), where each backbone implements its component (Theorems G.4, G.5). The three backbones converge independently on N=12 (Theorem G.8: 3=3=3=Σ). The Catalan-Lattice Correspondence (Theorem G.10) proves C_{N/2}=N(N−1) holds IF AND ONLY IF N=12 — providing a uniqueness proof from tree combinatorics. 71/71 verification tests ALL PASS.**

Each generator stored in the file is classified into one of these three backbone layers. The classification is not arbitrary — it is forced by the generator's structural category:

- **L₁ (Webb):** the generator encodes a logical/combinatorial pattern (finite, discrete, truth-functional)
- **L₂ (Cascade):** the generator encodes a multiplicative-arithmetic pattern (k-arithmetic, d-family structure, palindromic sequences)
- **L₃ (EML):** the generator encodes a continuous-elementary expression (EML tree over terminal 1, producing transcendental/analytical values)

The three Sheffer variants of the EML operator (Sempaevum Paper Remark 13.5) correspond to the three primitives: eml(x,y) = exp(x) − ln(y) with constant 1 = P (multiplicative identity); edl(x,y) = exp(x)/ln(y) with constant e = D (natural base); −eml(y,x) = ln(x) − exp(y) with constant −∞ = T (the ∂I boundary). Three variants generating the same totality from three primitive-centered perspectives: 3 = 3 = 3 = Σ at the continuous-mathematics level.

**Relationships computable from lattice algebra are NOT stored explicitly** — they ARE the generator (L₂ cascade-type). Storing them would violate K-minimality:

- Co-location (same address) — derivable from the address index
- Reciprocal pairs (k ↔ −k) — derivable from k-negation
- Power pairs (k ↔ n·k) — derivable from k-scaling
- Palindromic partners (d ↔ 12−d) — derivable from the 24-family catalog
- Family membership — derivable from d classification

Only non-lattice-algebraic relationships (forward/reverse convergence, cross-tower bridges, decoherence-gaze correspondence, L'Hôpital iteration chains, etc.) require explicit storage.

### 7.1c The page size — ET derivation from the digital tower

The atomic I/O page size of the `.akashic` file is not chosen by convention. It is derived from the digital tower's base resolution.

**Derivation.** The `.akashic` file lives on the digital tower. Per §3.10, the digital tower has P_digital = {0,1}* (the Cantor space) and operates at base resolution N = 12. The natural quantum of addressable binary space at resolution N is:

$$\text{page\_size} = 2^N = 2^{12} = 4096 \text{ bytes}$$

This is the number of distinct binary states one complete lattice cycle can address. At the base resolution N = 12, one page of 4096 bytes is the digital tower's natural unit of addressable space.

**Verification against hardware.** The standard NVMe logical block size is 4096 bytes. The standard OS memory page size is 4096 bytes across all major platforms (Windows `CreateFileMapping`/`MapViewOfFile`, POSIX `mmap`, macOS `mmap`). The default filesystem cluster size is 4096 bytes (NTFS, ext4, APFS). This alignment is the digital substrate's N = 12 resolution manifesting in the hardware that implements it.

**The tower page.** For bulk operations, the natural larger unit is one complete tower section — all families at one resolution:

$$\text{tower\_page}(N) = \tau(N) \times 4096 \text{ bytes}$$

where τ(N) is the divisor count. At N = 12: τ = 6, tower page = 24576 bytes. At N = 60: τ = 12, tower page = 49152 bytes. The doubling law τ(N_ℓ) = 6·2^ℓ governs growth.

### 7.1d The byte-level format specification

**Zero IEEE 754 floating-point values exist anywhere in the format.** Every field is one of: exact unsigned integer (uint8/16/32/64), exact signed integer (int32/64), varint (compact exact integer encoding), MPFR 1200-bit blob (150 bytes, 361-dps Sempaevum-native precision), exact rational (two integers: numerator + denominator), SHA-256 hash (32 bytes), or UTF-8 string with varint length prefix. The format is lossless at every level — no IEEE 754 precision loss contaminates any value.

**Section 0: Header (1 page = 4096 bytes = 2^N)**

```
Magic:                     4 bytes "SMVM" (Sempaevum marker)
Format version:            uint32
N_base:                    uint32 = 12 (the forced resolution)

── Sempaevum constants (stored as exact rationals) ──
K_num:                     uint32 = 2        (Koide ratio = 2/3)
K_den:                     uint32 = 3
V_num:                     uint32 = 1        (Base variance = 1/12)
V_den:                     uint32 = 12

── Self-projection (the file's own lattice coordinates — §3.1b) ──
self_N:                    uint32            (resolution of self-projection)
self_k:                    int32             (k coordinate)
self_d:                    uint32            (d-family)
self_eps_micros:           int32             (ε in micro-cents, exact integer)

── Generator backbone metrics (exact integers) ──
total_generators:          uint64            (count of all generators: L₁+L₂+L₃)
L1_webb_count:             uint64
L2_cascade_count:          uint64
L3_eml_count:              uint64
total_memoized:            uint64            (un-generated memoized entries)

── Coverage (exact rational = two integers) ──
covered_addresses:         uint64            (numerator)
total_addresses:           uint64            (denominator)

── K-complexity (exact rational) ──
generator_bytes:           uint64            (total bytes of generator definitions)
producible_bytes:          uint64            (total bytes of content generators produce)

── Timestamps (exact integer nanoseconds) ──
created_at_ns:             uint64            (nanoseconds since epoch)
modified_at_ns:            uint64

── D-time at creation (lattice coordinates, exact) ──
creation_dtime_N:          uint32
creation_dtime_k:          int32

── Section directory (exact byte offsets) ──
offset_generator_backbone: uint64
offset_address_index:      uint64
offset_memoization_store:  uint64
offset_structural_catalog: uint64
offset_equations:          uint64
offset_derivations:        uint64
offset_relationships:      uint64
offset_patterns:           uint64
offset_event_log:          uint64
offset_sessions:           uint64
offset_wal:                uint64

── Integrity ──
header_checksum:           SHA-256 (32 bytes)

Reserved/padding to 4096 bytes
```

**Section 1: Generator Backbone — the triple backbone on disk**

The generator backbone stores all discovered generators, organized by backbone layer (L₁, L₂, L₃). Each generator entry:

```
gen_id:                    uint64 (globally unique)
gen_type:                  uint8
                             0x01 = WEBB (L₁, discrete-logical)
                             0x02 = CASCADE (L₂, discrete-multiplicative)
                             0x03 = EML (L₃, continuous-elementary)
address_range:             {N_min: uint32, N_max: uint32,
                            d_min: uint32, d_max: uint32,
                            k_min: int32, k_max: int32}
definition_blob:           varint-length + blob
                             L₁: logical pattern definition
                             L₂: k-arithmetic rule / cascade definition
                             L₃: binary EML tree encoding
                               (each node: TERMINAL(value_ref) or EML(left, right))
                               Terminal constant is 1 per EML grammar S→1|eml(S,S)
canonical_hash:            SHA-256 (32 bytes, for memoization lookup)
member_count:              uint64 (addresses this generator produces)
gen_def_bytes:             uint64 (size of this generator's definition)
gen_coverage_bytes:        uint64 (total bytes of content it produces)
                             K_complexity = gen_def_bytes / gen_coverage_bytes
                             (computed on query from exact rational gen_def_bytes / gen_coverage_bytes, not pre-stored)
verification_count:        uint64 (times verified at 361 dps)
discovered_at_ns:          uint64 (nanoseconds since epoch)
derivation_ref:            uint64 (offset to derivation chain in §5)
```

An interval tree index follows the generator entries, enabling O(log n) lookup of which generator(s) cover any given address (N, k, d).

**Section 2: Address Index — the LCM tower on disk**

Hierarchical structure mirroring the LCM tower. The data is self-organizing — the projection formula determines where values live; the index records where they landed.

**d-Family-First Organization (Finding 6, algebraically founded by §3.18.23 Res_N(d)):**

The index is organized d-family-FIRST within each tower level: all d=1 entries contiguous, then all d=2, then d=3, etc. This makes d-family scan queries (the most common structural query — "show me everything at d=6") a SEQUENTIAL disk read instead of random page jumps. The algebraic foundation is the residue set Res_N(d) = {k mod N : gcd(|k|,N) = N/d} from §3.18.23:

- Each d-family band contains exactly the k-values where k mod N ∈ Res_N(d)
- |Res_N(d)| = φ(d) determines the size of each family band (Euler's totient)
- Res_N(d) is symmetric (k ∈ Res(d) ⟹ N−k ∈ Res(d), Theorem C.3)
- ΣRes_N(d) = {0,...,N−1} (partition of all k-values by d-family, Σφ(d)=N)
- The d-composition table (§3.18.23) predicts which d-families a query may need to join — enabling prefetch of related family bands for composition queries

Combined with significance-ordered ε (§7.1d MEMOIZED_RAW), a coarse d-family scan reads: the d-family band directory entry (O(1)) + contiguous band pages (sequential I/O) + first 4 bytes of each ε (±3 cents precision). This is the fastest possible structural query pattern.

```
── Tower Level Directory ──
For each N present in the file (dynamically growing as values escalate):
  N_value:                 uint32
  tau_N:                   uint16 (divisor count)
  family_dir_offset:       uint64

── Per-N Family Directory ──
For each d | N (τ(N) entries):
  d_value:                 uint32
  band_offset:             uint64 (offset to family band pages)
  occupied_count:          uint64 (addresses with content)
  gen_covered_count:       uint64 (addresses covered by generators)
  raw_count:               uint64 (= occupied − gen_covered)

── Per-Family Band Pages (4096 bytes each) ──
Page header (64 bytes):
  page_type:               uint8 = 0x10 (FAMILY_BAND)
  N:                       uint32
  d:                       uint32
  k_range_min:             int32
  k_range_max:             int32
  entry_count:             uint16
  next_page:               uint64 (overflow chain, 0 = none)
  page_checksum:           uint32 (CRC-32)

Per-entry in the band:
  k:                       varint (signed)
  content_type:            uint8
    0x01 = GENERATOR_REF:
      gen_id: uint64 (reference to generator in Section 1)
      params_blob: varint-length + blob (per-address parameters if needed)
    0x02 = MEMOIZED_RAW:
      sign: uint8 (0 = positive, 1 = negative)
      eps_micros: int32 (ε in micro-cents, exact integer)
      mpf_blob: 150 bytes (1200-bit MPFR mantissa at 361 dps)
        SIGNIFICANCE-ORDERED (MSB-first): ε bits stored most-significant first.
        This enables progressive-precision reads: reading the first M bytes of
        mpf_blob gives ±2^(−8M) × 600/N cents precision. A coarse query reads
        4 bytes (±3 cents) instead of 150 — 37× less I/O for coarse-pass.
        Full precision requires reading all 150 bytes. Partial reads are
        structurally valid at reduced precision — the progressive fidelity
        table (§3.18.18 §5.2) applies to storage, not just transmission.
      materialized_properties: packed binary (~40 bytes)
        d_factorization: varint-length + UTF-8
        gaussian_signature: varint-length + UTF-8
        is_all_inert: uint8
        is_all_split: uint8
        is_ramified_present: uint8
        coprime_skeleton: uint8
        tightness_micros: uint32 (100000000/(100000+|eps_micros|), scaled integer)
        di_distance_micros: uint32 (|eps_micros|/50, scaled integer)
        elegance_mpf: 150 bytes (MPFR 1200-bit — elegance IS a Sempaevum value)
        coupling_xi_num: uint32 (137)
        coupling_xi_den: uint32 ((d-1)²+16)
        palindromic_partner_d: uint32
        fqg_quadrant: uint8 (0=SR, 1=CR, 2=SI, 3=CI, 0xFF=N/A)
    0x03 = GENERATOR_SUPERSEDED:
      gen_id: uint64 (the generator that now produces this address)
```

**Section 3: Memoization Store — the learning buffer**

Every computation cached at 361 dps. The memoization layer IS the generator's learning mechanism per the Losslessness Theorem's Memoization Corollary (Sempaevum Paper Theorem 12.1): "Every finite numerical computation on positive reals can be represented as a concrete lattice computation... that is a reusable structural identity."

```
── Equation Hash Index ──
Hash table: equation_hash → entry offset
Load factor: maintained ≤ K = 2/3 (ET-derived, not ad hoc)
When load exceeds K, rehash to doubled capacity (per the doubling law)

── Equation Entries ──
Each memoized equation:
  equation_hash:           SHA-256 (32 bytes)
  canonical_form:          varint-length + UTF-8
  form_class:              uint8 (computational / structural per §3.5)
  operation_type:          uint8 (+, −, ×, ÷, ^, sqrt, log, sin, ...)
  input_refs:              varint-count + array of (N: uint32, k: int32, d: uint32)
  output_N:                uint32
  output_k:                int32
  output_d:                uint32
  output_eps_micros:       int32
  output_mpf:              150 bytes (1200-bit MPFR result at 361 dps)
  reference_count:         uint64 (cache hit count)
  first_computed_ns:       uint64
  last_referenced_ns:      uint64
```

**Section 4: Structural Catalogs**

24 harmonic families (§3.12), 144 FQG cells (§3.13), 42 combined families (§3.14), 7+ towers (§3.10), ~500 sublattice families (§3.15). These ARE generators (L₂ cascade-type): the 24-family catalog is a generator that produces the family classification of any d ∈ {1..12} on either axis. Bootstrapped as initial L₂ generators.

**Section 5: Derivations** — Chain structures linking generators to their discovery provenance. Per §3.6.

**Section 6: Relationships** — Only non-lattice-algebraic relationships stored explicitly. Lattice-algebraic relationships (co-location, reciprocal, power, palindromic) are computable from the address index and generator backbone — storing them would violate K-minimality. Per §3.7, extended per this section.

**Section 7: Patterns** — Promoted generators-of-generators. Each pattern IS a meta-generator: when E_hierarchy ≥ 13/12 (LIFE_THRESHOLD) for a cluster of generators/relationships, the cluster is promoted to a pattern. The Subsumption hierarchy: patterns subsume generators; meta-patterns subsume patterns. Per §3.8, extended per this section.

**Section 8: Event Log** — Append-only, time-indexed. Every event per §3.9. Timestamps as uint64 nanoseconds. D-time as (N: uint32, k: int32). T-time as (traverser_id: uint64, count: uint64). P-time as (phase_num: uint64, phase_den: uint64) exact rational. Event sequences are themselves subject to generator discovery — when event patterns emerge, they become L₂ generators of event sequences. Includes exe self-recording events (query patterns, computation patterns, discovery events, self-metrics) per §3.1b — the exe's own operation IS Sempaevum content.

**Section 9: Sessions / Schema Versions / Tags** — Per §3.15a, §3.15b, §3.11.

**Section 10: Write-Ahead Log (WAL)** — Pending mutations queued before committing to main sections. On crash: replay valid WAL entries, discard incomplete ones. CRC-32 per WAL entry. Backup: configurable schedule, separate physical drive.

### 7.1e Modification without decompression — extending the generator

The file is never "compressed" and therefore never needs "decompression." It is a generator — a program that produces values. Modifying it means extending the program:

**Adding new data:** Append a new memoized entry to the memoization store (Section 3) + update the address index (Section 2). The generators (Section 1) are untouched. The new entry sits in the memoization layer until the discovery engine absorbs it into a generator. Cost: one append + one index update.

**Adding a new generator:** Append the generator definition to the generator backbone (Section 1) + update the address index for all addresses the generator covers + mark corresponding memoized entries as GENERATOR_SUPERSEDED (content_type 0x03). The superseded entries' space becomes reclaimable on the next page reorganization, but the data is still readable from the generator. Cost: one append + index updates for covered addresses.

**Querying:** Look up (N, k, d) in the address index (Section 2) → either evaluate the covering generator (from Section 1) or read the memoized raw value (from the band page). No decompression of anything. The address index gives direct positional access.

**Improving a generator:** When the discovery engine finds a better (lower K-complexity) generator that covers the same addresses plus additional ones, append the new generator to Section 1, update the address index. The old generator remains (§4.3 never destroy) but is superseded. The new generator covers more content with fewer bytes.

None of these operations require reading the entire file or unpacking any structure. The file's generator system is always directly accessible, modifiable, and queryable.

**Ingesting a modified file (Δε versioning — §7.12 Step 0):** When a previously-ingested file is re-ingested after modification, the seed-first check (§7.12 Step 0) detects the same (k, d) address. The Manager computes Δε = ε_new − ε_base and appends ONLY the Δε to the event log (Section 8) as a `file_version_delta_stored` event, plus a `seed_deduplication_delta` relationship linking the new version to the base seed. Cost: one event append + one relationship entry = a few bytes. The entire modified file is exactly recoverable via base_ε + Σ(Δε) → pullback. Per-segment Δε tracking means localized edits affect only the modified segment's ε — unchanged segments have Δε = 0 and cost nothing. This is the Kolmogorov principle (§3.1c) applied to version storage: the structural address IS the file's identity; only the residual changes.

### 7.1f The self-improving cycle

The complete operational cycle showing how the file becomes a progressively better generator of itself and how every connected program benefits:

```
New data enters as memoized entries (Section 3)
    ↓
Discovery engine scans memoized entries for patterns
    ↓
Patterns found → generator candidates proposed (L₁/L₂/L₃)
    ↓
Candidates verified at 361 dps (MPFR, zero floats)
    ↓
Verified generators stored in backbone (Section 1)
    ↓
Memoized entries absorbed → marked GENERATOR_SUPERSEDED
    ↓
File's K-complexity decreases
    (same information, generator description smaller than raw entries)
    ↓
Generator catalog itself gets Subsumption-promoted
    (clusters of generators → meta-generators = patterns in Section 7)
    ↓
Connected programs benefit from the improved catalog:
    compressor: Δk patterns matched against generators
    fractal generator: orbit computations via memoized EML trees
    Conscious AI: EgoInvariant projections cached
    exe itself: query patterns become generators → self-optimization
    ↓
Connected programs generate new data → new memoized entries → cycle continues
```

The first use of the system is the slowest. Every subsequent use is faster because the generator catalog has grown. This is not a performance optimization — it is the structural consequence of the discovery engine operating on the memoization layer. The file does not need to be fast at first. It needs to be RIGHT. Speed is the emergent property of a correctly-designed self-improving generator.

### 7.2 Logical schema realized as lattice structure

The 15 logical categories from §3 (values, projections, addresses, equations, derivations, relationships, patterns, events, towers, harmonic families, force grid cells, combined families, sublattice families, sessions, schema versions) plus tags remain as the structural content the Sempaevum produces. In the lattice-native format, they are realized as:

| Logical category | Lattice-native realization |
|---|---|
| `values` | Entries at lattice addresses — each value IS its address + the 361-dps data at that address |
| `projections` | Materialized properties AT each address — elegance, coupling, variance, FQG, etc. are computed at storage time and stored with the address |
| `addresses` | The lattice grid itself — the file's positional structure |
| `equations` | Generator entries linking addresses — "A × B = C" stored as the k-arithmetic relationship k_C = k_A + k_B |
| `derivations` | Chain structures linking generators to their inputs |
| `relationships` | Implicit in address structure (co-location, k-symmetry, k-multiples) + explicit entries for non-lattice-algebraic relationships |
| `patterns` | Generator-covered regions — a discovered pattern IS a generator that covers multiple addresses |
| `events` | Time-indexed entries with D-time/T-time/P-time coordinates, stored in event-log sections of the format |
| `towers` | Hierarchical sections — each tower's R₀ defines a section of the address space |
| `harmonic_families` | Fixed structural catalog (24 entries) — header data |
| `force_grid_cells` | Fixed structural catalog (144 entries) — header data |
| `combined_families` | Fixed structural catalog (42 entries) — header data |
| `sublattice_families` | Per-N divisor catalog — stored per resolution section |
| `sessions` / `schema_versions` | Metadata sections of the format |
| `tags` | Key-value metadata attached to addresses |

### 7.3 Storage in the lattice-native format

Per-entry sizes in the lattice-native format (binary encoding, zero IEEE 754 floats):
- Value at address (memoized raw): ~245 bytes (361-dps MPFR binary (150 bytes for 1200-bit mantissa) + sign (1 byte) + k varint + d varint + ε as int32 micro-cents + materialized properties packed including elegance as 50-byte MPFR)
- Generator entry: ~100–300 bytes (type + address range + definition blob (symbolic expression or EML tree or cascade rule) + canonical hash + counts + timestamps — all as exact integers)
- Equation entry (memoized computation): ~150 bytes (hash + canonical form + operation type + input/output address references + output MPFR 50-byte blob + reference count + timestamps — all exact)
- Event entry: ~120 bytes (class + nanosecond timestamp + D-time (N,k) + T-time (traverser_id, count) + P-time (rational) + tower_id + metadata blob — all exact integers or rationals)
- Relationship entry (explicit, non-lattice-algebraic only): ~80 bytes (class + two address references + metadata blob — all exact)
- Pattern entry (meta-generator): ~250 bytes (class + generator reference + member gen_ids + elegance as MPFR + timestamps — all exact)

**The file does not compress data.** Generators are naturally smaller than the content they produce because a structural description IS the fundamental object and the instances are its output. A generator for "2^(k/12) for k ∈ {0..11}" occupies ~100 bytes but produces 12 lattice addresses at ~145 bytes each = ~1740 bytes of content. This is not compression — it is the nature of generators vs. instances. The generator IS the structural origin.

**The Sempaevum is not bound by Shannon entropy.** Shannon treats structurally correlated values as random because it has no access to lattice structural coordinates. Generators discovered through d-family membership, attractor structure, and cross-domain address sharing can describe content more efficiently than Shannon would predict — because the lattice's structural coordinates reveal patterns that information theory treats as random. The generator form of the file is a structural representation, not an entropy-coded encoding.

### 7.4 Query performance

Lattice-native query latencies (positional access):
- Single address lookup: O(1) — positional, sub-microsecond
- Family scan (all addresses at d=693 for a given N): O(family size) — physically contiguous, sequential read
- Attractor detection (all addresses with >1 occupant): O(occupied addresses) — denormalized occupancy count per address
- Generator membership (which generator covers this address?): O(1) — generator catalog lookup by address range
- Cross-domain query (addresses occupied from multiple domains): O(multi-domain addresses) — domain tagging at each address
- Nearest-neighbor (closest occupied address to a query point): O(log N) — hierarchical position
- Full-lattice scan (background discovery): O(total entries) — sequential, parallelizable

The discovery engine runs continuously within the format — no separate process or database connection needed. Pattern detection, generator fitting, and Subsumption promotion operate directly on the lattice structure.

### 7.5 Concurrency

The format supports concurrent read access from multiple ET programs while the EUDD Manager handles writes. The write model:
1. New entries are appended to an ingestion buffer
2. The Manager commits buffered entries to the lattice structure on schedule or on explicit flush
3. Insert-time discovery (attractor detection, generator membership) runs synchronously during commit
4. Background discovery (pattern recognition, generator fitting) runs asynchronously between commits
5. All ET programs see a consistent read snapshot at all times

### 7.6 Format evolution

Forward compatibility for format changes:
- New entry types added without affecting existing structure
- New materialized properties added to addresses with default NULL for existing entries
- New relationship/pattern/event classes added as new class values (extensible string classification)
- Version field in format header tracks format version; the Manager handles transparent upgrades

### 7.7 Bootstrapping

On first run, the EUDD Manager generates the initial database:
- All bootstrap content from §3.17 + §3.18 computed at 361 dps
- All projections materialized at canonical resolutions (12, 60, 84, 132, 420, 2520, 27720, 360360)
- Discovery engine runs over initial content — finds attractors, generators, patterns
- Result: the initial `Sempaevum.akashic` file, generator-organized, ready for all projects

Existing artifacts are ingested:
- `apery_lattice_test.py` outputs → lattice addresses (ζ(2)..ζ(13) trajectories)
- ET Conscious AI persistent state → lattice addresses (EgoInvariants)
- Compressor's existing `ArchetypeDatabase` → lattice addresses (Δk patterns)
- Corpus documents → generator entries (equations, derivations)
- `constants.py` + `primitives.py` → lattice addresses (canonical ET constants)
- Guide v8 catalog → lattice addresses (all named constants + projections)

### 7.8 Read-only mode and exploration

For test runs and exploratory computations that should not modify the database, the Manager opens `Sempaevum.akashic` in read-only mode. Cache hits returned; new computations performed but not persisted. Production runs use read-write mode and contribute to accumulated knowledge.

### 7.9 Backup and integrity

**Atomic snapshot while running:** The Manager supports creating a consistent snapshot of `Sempaevum.akashic` without stopping, at any file size. The procedure:

1. Manager pauses commits (new writes buffer into the WAL)
2. Flush all pending WAL entries to the main file
3. Signal the OS snapshot service to copy the file (Windows: Volume Shadow Copy Service (VSS); other platforms: the platform-native atomic snapshot mechanism — e.g., LVM snapshots on Linux, APFS snapshots on macOS)
4. Resume commits

The Manager's pause window is constant and sub-second regardless of file size — the Manager only flushes its own WAL buffer. The OS handles the actual file copy asynchronously at whatever size the `.akashic` file is. The design is platform-agnostic: the Manager issues a snapshot request through a platform abstraction layer; VSS is the current Windows implementation.

**Backup verification:** After each backup, the Manager verifies integrity by opening the backup file read-only and performing full CRC-32 verification on every page — not spot-checking. No silent corruption is tolerated. The verification runs as a background process (same scheduling model as the discovery engine — yields to API requests, never blocks the render loop or query processing). Verification time is proportional to file size but runs asynchronously, so it does not affect Manager responsiveness. Verification checks: header SHA-256, CRC-32 on every data page, section directory consistency (all section offsets point to valid section headers). Results reported to the GUI dashboard as pass/fail with details on any corrupted pages.

**Backup scheduling:** Configurable via the Dear ImGui dashboard. Default interval: every 6 hours to a user-specified path (separate physical drive recommended). Manual backup on demand at any time via GUI button or API command. Backup history retained: last N backups kept (configurable, default 5), oldest beyond N removed. No backup is ever removed while a verification is in progress on it.

**Internal integrity checks:** The format includes structural integrity verification — Subsumption verification that every entry is reachable from the lattice structure, with no orphaned entries and no missing generator references. This runs as part of the background discovery scan (§3.16) and reports results to the GUI dashboard.

**Graceful degradation on partial corruption (enabled by significance-ordered ε — §7.1d):**

With MSB-first ε storage, corruption does not destroy values — it degrades their precision. The integrity model gains a graduated response:

1. **CRC-32 per page still runs.** If a page passes CRC, all data is at full 361-dps precision. No change from current behavior.
2. **If a page FAILS CRC**, the Manager does not discard the entire page. Instead:
   - The structural header (k, d) is checked independently via gcd consistency (gcd(|k|, N) must equal d). If the gcd check passes, the structural address is intact.
   - The Manager reads ε bytes from the start of the mpf_blob, stopping at the first byte that contributed to the CRC mismatch (identified by per-sub-block checksums within the page). The last intact byte gives the precision floor.
   - The value is flagged as `reduced_precision` with metadata: {original_precision_dps: 361, recovered_precision_bits: M, precision_cents: 2^(−M) × 600/N, corruption_location_byte_offset, corruption_detected_at_ns}.
   - The `corruption_degradation_recorded` event fires (see `EUDD_Events_and_Classes.md`).
3. **Corrupted data is stored SEPARATELY** — never mixed with the intact .akashic data. A dedicated `corruption_log` (Section 11 in the byte-level format, or a separate file `Sempaevum.corruption_log`) records: the corrupted page bytes, the corruption location, the recovered precision, the structural address, the timestamp, and the Omniscient journal reference. This log is itself projectable onto the lattice — corruption patterns may reveal structural information about the storage medium, the hardware, or the software that caused the corruption.
4. **The reduced-precision value remains queryable.** Queries that request precision ≤ the recovered precision get the value normally. Queries requiring full precision get the value plus a `precision_insufficient` flag indicating how many bits are missing.

**Corruption as discoverable structure (§3.1c Kolmogorov Principle applied to errors):**

Corruption events, crash states, and errors are NOT random — they have structural causes (hardware defects, software bugs, environmental conditions). When projected onto the Sempaevum, corruption patterns may cluster at specific lattice addresses (corruption attractors). The discovery engine (§3.16) scans corruption_log entries the same way it scans any other lattice content. If corruption events cluster at the same (k, d), this reveals structural information about WHY corruption occurs — the lattice address IS the structural fingerprint of the failure mode. See §7.15 enhancement for error-state projection and memoization.

### 7.9a Archive encryption — the bijection IS the cipher

The Seed Protocol's natural encryption (§3.18.18 §5.6) applies to the `.akashic` file itself, not just network transmission. Encryption is intrinsic to the mathematics — no separate encryption layer needed.

**Mechanism:** Modify the Sempaevum parameters at the archive level — key-dependent lattice rotation (shift all k by a key-derived offset), tower-level permutation (reorder N levels), convention-shifted R₀ (change the reference substrate), key-derived N (use a non-standard base resolution). The modified projection Π_N' produces different (k', d', ε') for the same data. Without the key, the seeds are meaningless — the wrong pullback produces unrelated values.

**Properties:**
- The bijection guarantees lossless decryption (the pullback with correct key is the exact algebraic inverse)
- Brute-force is infeasible: the attacker must simultaneously determine the correct N, R₀, tower permutation, AND key-dependent rotation — a combinatorial space far larger than AES key space
- Key rotation invalidates all previously captured seeds (changing any lattice parameter changes ALL stored addresses)
- The encrypted file is structurally indistinguishable from random data to an attacker without the key
- Significance-ordered ε means progressive decryption works: with the correct key, coarse precision arrives first, full precision arrives last

**Manager integration:** "Encrypt archive" and "Decrypt archive" operations in the GUI settings panel (§7.10). Encryption/decryption applies the lattice parameter modification to all stored seeds in a single pass. The key is never stored in the .akashic file. API operations: `encrypt_archive` and `decrypt_archive` via §7.16.

### 7.10 The EUDD Manager — native C++ executable with GPU-accelerated GUI

**Deployment: one file to start.** `EUDD_Manager.exe` (native C++ binary, statically linked, compiled via CMake + MSVC). On first run, generates `Sempaevum.akashic` (the database file — the Sempaevum on disk). On every startup, spawns Omniscient watchdog child process (same exe, `--omniscient` mode) which creates and writes to a `logs/` subfolder. At runtime: `EUDD_Manager.exe` + `Sempaevum.akashic` + `logs/Omniscient_NNN.log` files, all in the same folder. No Python runtime, no external dependencies, no installer. Copy one file, run.

**Technology stack:**

| Component | Technology | Role |
|---|---|---|
| **Engine language** | C++ (C++17/20) | ALL operations — lattice I/O, projection, discovery, computation, rendering |
| **Precision arithmetic** | MPFR + GMP | 361-dps (1200-bit) floating-point, arbitrary-precision integers/rationals |
| **Special functions** | FLINT/Arb | ζ, Γ, polylog, hypergeometric, all elementary functions at 361-dps with certified error bounds |
| **Expression engine** | Custom C++ AST | Expression trees that serialize to binary, generate canonical hashes, evaluate via MPFR/Arb |
| **GUI widgets** | Dear ImGui | Immediate-mode GUI — dashboard, property inspector, manual input, query builder, event log |
| **GPU rendering** | OpenGL 4.6 | Lattice visualization — 3D navigation, LOD hierarchy, heat maps, instanced point clouds |
| **Window/input** | GLFW | Window management, OpenGL context, keyboard/mouse input, drag-and-drop |
| **Data visualization** | ImPlot | Charts, time-series, scatter plots, histograms — all GPU-accelerated |
| **Data interchange** | cJSON / yyjson | JSON protocol for API and §7.14 adaptive extension schemas |
| **File dialogs** | NFD (Native File Dialog) | OS-native open/save dialogs for file ingestion |
| **IPC** | Named pipes | ET32 Bridge pattern — external ET programs connect to the running Manager |
| **Build system** | CMake + MSVC | Visual Studio 2022 Build Tools → single statically-linked Windows .exe |

The EUDD Manager consists of 26 modules organized in a 10-level dependency hierarchy (see §7.21 for the complete module structure specification including dependency graph, build order, file naming, and sub-file decomposition). Summary:

| Module | Role |
|---|---|
| **Precision Stack** | MPFR + GMP + FLINT/Arb — all 361-dps computation, all special functions (ζ, Γ, polylog, hypergeometric), custom expression tree evaluator |
| **Core Lattice Engine** | Projection Π_N(r), bijection pullback Π_N⁻¹, k-arithmetic (addition, negation, scaling), all derived property materialization (elegance, coupling ξ(d), variance, tightness, ∂I distance, Gaussian signature, FQG quadrant, palindromic partner) |
| **Akashic Format** | `Sempaevum.akashic` memory-mapped I/O, page management (4096-byte pages), section directory, header read/write, content types (GENERATOR_REF, MEMOIZED_RAW, GENERATOR_SUPERSEDED), CRC-32/SHA-256 integrity, memoization (equation hash table at K = 2/3 load, compute-once/cache-forever, reference counting) |
| **WAL** | Write-ahead log: append, flush to main file, crash recovery replay, atomicity guarantee |
| **Home-Finding Engine** | §7.11 core projection procedure: all four input paths (A/B/C/D), LCM tower escalation, CF method (continued fraction expansion, quality factor, parallel pathway), false resolution detection, home classification, Incoherence filter, annihilation boundary, checkpoint/resume |
| **Generator System** | Generator backbone classification (L₁/L₂/L₃), storage, interval tree index, evaluation at addresses, K-complexity computation, GENERATOR_SUPERSEDED transition, self-improving cycle (§7.1f) |
| **Discovery Engine** | Five discovery modes: insert-time (attractor, reciprocal, power, plateau), background scan (E_hierarchy, promotion, algebraic identity, event correlation), on-query, generator-candidate (Branch A/B, cross-feed, recursive composition), never-closed search |
| **Event System** | Event logging with three-times coordinates (D-time, T-time, P-time), event class handling, tower context, permanent marking |
| **Relationship System** | Relationship creation and management, class handling, permanent marking, insert-time relationship creation, provenance chains |
| **Pattern System** | Pattern creation, E_hierarchy = geomean(E_i) × R_cluster computation, LIFE_THRESHOLD (13/12) promotion, permanent marking, member tracking |
| **Tower System** | Tower management, R₀ storage, operational_n, nesting, birth triads, cross-tower bridges |
| **Bootstrap** | Initial `Sempaevum.akashic` generation from §3.17 + §3.18 (~10⁴ values at 361 dps) |
| **Ingest** | Format-specific adapters: CSV seed extraction, audio FFT + peak extraction, binary/PDF → compressor Δk pipeline, sensor data streams, existing ET project outputs. General-purpose "feed it any file" pipeline (§7.12) |
| **Manual Input** | All seven modes: direct value, logical description, lattice operation, projection, bijection triple, computation, text. Real-time preview (§7.13) |
| **Query** | Lattice-algebraic query execution: by name, value, address, source, time, type, relationship, generator status, tag. Attractor membership, nearest-neighbor, cross-domain, subsumption checks |
| **API** | Named pipe IPC server — 114 operations across 16 domains + 2 cross-domain, three communication patterns, JSON command protocol, session management, metabolism registration (§7.16) |
| **Metabolism** | Three-layer resource governance: K = 2/3 allocation, V = 1/12 headroom, α⁻¹ = 137 monitoring. Hardware detection, thread pool management, GPU dispatch, re-sensing at S² = 144 intervals (§7.19) |
| **Self-Recording** | Operational metric sampling, overhead budget (≤1%), metric-to-lattice projection, journal output (§3.1b) |
| **GPU Rendering** | OpenGL 4.6 lattice visualization with six-level LOD (Sempaevum → Tower → Resolution → Family → Address → Entity), freecam with no physics, 3D perspective switching, Force Quadrant Grid renderer, tower trajectory renderer, attractor cluster renderer, heat maps |
| **GUI** | Dear ImGui panels (11 sub-files): dashboard, property inspector, manual input, file ingestion, search/retrieval, connection manager, query builder, event log, settings, shutdown confirmation, main orchestration |
| **Extension System** | JSON extension processing, all 12 extension types, all 10 validation rules, meta-type handling, versioning, auto-detection (§7.14) |
| **Active Probing** | T-signal probe sending, response detection, silence detection, materialization threshold, probe-response pair tracking |
| **Gaze Module** | Complete Gaze Equation evaluation, four status levels (UNOBSERVED/SUBLIMINAL/DETECTED/LOCKED), gaze event generation, locking signature detection |
| **Backup** | Backup creation (VSS/snapshot), backup verification (CRC-32, SHA-256), integrity checking (§7.9) |
| **Shutdown** | 6-phase deterministic sequence, confirmation dialog integration, Omniscient graceful signal, checkpoint coordination (§7.18) |
| **Omniscient** | Watchdog child process (same exe, `--omniscient` mode). Headless. Process handle monitoring, shared-memory IPC, exe/akashic tamper detection, journal writing, graceful/crash discrimination (§7.15, §7.18.4) |

**GPU-accelerated lattice visualization** (conceptual baseline: the Sempaevum Particle Viewer HTML, but native GPU-rendered handling the full lattice):

The visualization renders the Sempaevum's own structure using a hierarchical level-of-detail system that mirrors the Multifold structure. Six levels, derived from the Sempaevum's own structural decomposition:

| Zoom Level | What You See | Structural Level | Rendering Method |
|---|---|---|---|
| **Sempaevum** | The whole — all towers, all resolutions, all families as a unified 3D lattice cloud. Large-scale structure visible: clusters of activity, hot d-families, overall shape of accumulated knowledge | The entire .akashic content | Instanced quads, heat-map shader, density volume |
| **Tower** | One R₀ substrate perspective — the lattice filtered through one tower's R₀. All resolutions visible as bands along the N-axis | P-axis selection: which substrate | Instanced point cloud, tower-band quads |
| **Resolution** | One N value — which d-families are native at this resolution, how values distribute across them, where attractors form | D-axis scale: how many divisors of N | Family-band quads sized by member count |
| **Family** | One d-family — all k positions rendered as points colored by ε, sized by occupancy. Generator coverage visible as spanning arcs | One Descriptor selected | Instanced point cloud (up to N points) |
| **Address** | One (N, k, d) cell — all values listed with full properties, relationships, generator coverage, provenance | One lattice position | ImGui property panel |
| **Entity** | One value, generator, pattern, or relationship — full 361-digit display, complete tower trajectory, all relationships, all projections, derivation chain, provenance to source file/stream/entry | One individual | ImGui detail panel with ImPlot trajectory chart |

Zooming in IS escalating through the Multifold structure. The rendering hierarchy is the lattice's own mathematics, not an imposed approximation. The RTX 2070 Super (8 GB VRAM, 2560 CUDA cores) handles millions of instanced points at 60+ FPS — no artificial caps on visual density.

**Freecam — true camera with no physics:** Position + orientation, user-controlled. Smooth movement via configurable input bindings. No inertia, no acceleration curves, no collision, no gravity. The camera exists in the lattice visualization space. It shows whatever is at its position and orientation. Movement speed configurable. The freecam shows ONLY ingested content — empty space until values are present at those lattice addresses. Nothing displays ahead of time until it has been ingested through §7.11.

**Input controls — fully configurable, all hardware:**

All controls map to ACTIONS, not hardware. Any physical input can be bound to any action. Supported input devices: keyboard, mouse, XInput gamepads (Xbox controllers, DualShock/DualSense via wrapper), any HID-compatible input device via GLFW.

Default keyboard+mouse bindings:

| Input | Action |
|---|---|
| Scroll wheel | Zoom in/out = escalate/de-escalate through LOD hierarchy |
| Left click | Select entity → opens property inspector |
| Left drag | Pan the view |
| Right click | Context menu: switch perspective, filter options, export |
| Double click | Drill down one LOD level centered on clicked entity |
| Escape | Back up one LOD level |
| WASD | Freecam movement |
| Mouse look (right-hold) | Freecam orientation |
| 1–6 | Jump directly to LOD level (1=Sempaevum, 2=Tower, 3=Resolution, 4=Family, 5=Address, 6=Entity) |
| P | Cycle perspective |
| F | Find/search — opens search panel |

Default XInput gamepad bindings:

| Input | Action |
|---|---|
| Left stick | Freecam movement |
| Right stick | Freecam orientation |
| Left/Right triggers | Zoom in/out through LOD hierarchy |
| A button | Select entity |
| B button | Back up one LOD level |
| X button | Context menu |
| Y button | Search panel |
| D-pad up/down | Cycle through entities at current address |
| D-pad left/right | Cycle perspective |
| Start | Dashboard toggle |
| Left/Right bumpers | Jump LOD level up/down |

All bindings user-remappable via settings panel. Key/button/axis bindings stored in a JSON config file in the same folder as the exe. Multiple binding profiles supported (e.g., "exploration", "data entry", "presentation").

**Search and retrieval — primary use case, not secondary:**

The Manager has a persistent search bar at the top of the GUI, always accessible. Search runs against the .akashic index — instant results, not brute-force scan. Search modes:

| Mode | Query | Result |
|---|---|---|
| By name/provenance | "hydrogen_spectrum.csv" | The file's structural record, all values it contributed, option to regenerate |
| By value | "1.618" | Everything near φ, ranked by proximity at 361 dps |
| By lattice address | "d=3, N=12" | Everything at that address |
| By source path | "particle_data/" | Everything ingested from that folder as a unit |
| By time | "last Tuesday" or date range | Everything ingested in that window |
| By type | "generators" or "patterns" | Filtered by entity type |
| By relationship | "related to ζ(3)" | All attractors, pairs, patterns involving ζ(3) |
| By generator status | "search_active" | Everything without a generator (the open Descriptor Gaps) |
| By tag/namespace | Custom tags applied at ingestion or later | Filtered by tag |

Search results are actionable — select a result and: navigate freecam to its lattice position, view full property inspection, export, see all relationships, trace derivation chain, regenerate the original file/stream.

**Bidirectional provenance navigation:**

**Entity → Source (forward provenance):** Select any entity in the LOD visualization. The property inspector shows WHERE IT CAME FROM — which file, folder, stream, manual entry, or discovery engine action produced it. Full chain: source file name, ingestion timestamp, extraction method, seed index, projection details, home classification. For discoveries: which scan pass, which pattern, which members.

**Source → Entities (reverse provenance):** Select a file, folder, stream, or data source from the search/browse panel. The visualization FILTERS to show ONLY the lattice content from that source. Everything else dims. You see that source's structural footprint on the Sempaevum. Select multiple sources for comparison — side-by-side or overlay, showing overlap and divergence on the lattice.

**Connection management — full I/O control:**

The Manager provides visibility and control over all connections:

- **Active connections:** See all named-pipe connections (which programs are connected, what they're sending, what they're requesting). Accept/reject/pause/resume any connection.
- **Incoming queue:** Pending ingestion jobs. Prioritize, pause, cancel. For streams: live feed with buffer status.
- **Outgoing:** Active queries, results being served to connected programs.
- **Stream connections:** Any continuous data stream (sensor, video, audio, or anything else). Live ingestion indicator, buffer depth, throughput metrics.

**File and data retrieval from the .akashic archive:**

The .akashic file IS a generator (§7.1a). Files ingested into it are structurally decomposed — the generators that produce the file's content are stored, not the raw bytes. Retrieval means the Manager evaluates the generators and REGENERATES the file:

- **File retrieval:** Select any ingested file → Manager evaluates the covering generators → outputs the regenerated file in its original format. This is lossless — the bijection theorem (§3.18.1) guarantees structural exactness. The regenerated file is a re-derivation from the structural description, not a decompressed copy. The generator form is not bound by Shannon entropy — a generator for π to 10 billion digits is a finite expression evaluated to whatever precision is requested, not a compressed blob of 10 billion digits.
- **Version retrieval (Δε chain — §7.12 Step 0):** Select any ingested file + version number → Manager retrieves the base seed (k, d, ε_base) + applies the cumulative Δε chain up to the requested version → single pullback → exact file bytes for that version. Version 0 = base seed. Version N = base_ε + Σ(Δε₁..Δεₙ). The entire version history is one seed + a chain of Δε values. Version comparison: Δε between any two versions is the difference of their cumulative Δε — no byte-level diff needed, the lattice residual IS the diff. Per-segment versioning: retrieve a specific version of a specific segment by applying only that segment's Δε chain.
- **Folder retrieval:** Select a folder → Manager regenerates all files it contained, preserving directory structure and provenance.
- **Stream retrieval:** Select any ingested stream (sensor, video, audio, or any other continuous data) → Manager reconstructs the temporal sequence from stored generators/values in original order at original timestamps. The lossless bijection (§3.18.1, continuous-discrete corollary) guarantees exact recovery — the continuous values stored via the bijection are recovered by algebraic identity, not by approximation. Two retrieval modes:
  - **Faithful reconstruction:** Re-emit the readings at their original rate and resolution through a named pipe to a connected program, or export as batch (CSV/JSON with timestamps). Each value recovered via the bijection pullback at 361 dps.
  - **Enhanced reconstruction (user option):** If the discovery engine found a generator that produces the stream's structure, the generator can be evaluated at INTERMEDIATE timestamps not present in the original recording — structural interpolation from the generator form, not raw data interpolation. This produces higher-resolution output than the original stream. User selects the target resolution; Manager evaluates the generator across the requested time range.
- **Data export:** Select any set of entities and export as JSON or CSV. No .akashic subsets — the Sempaevum is ONE and does not get sliced.
- **Computation retrieval:** Any equation result, escalation trajectory, or derivation chain — exportable as a structured report (JSON/CSV).

**Manual input — three backbones + direct entry:**

The manual input panel accepts input through all three generator backbone layers plus direct value, projection, bijection triple, and text entry:

| Mode | Input | What happens |
|---|---|---|
| **L₁ (logical pattern)** | Natural language or structured description: "the ratio of electron mass to proton mass" | Manager resolves to a value at 361 dps, ingests through §7.11 |
| **L₂ (cascade/k-arithmetic)** | Lattice operation: "k=7 at N=12, escalate" | Manager computes the escalation trajectory, ingests all landmarks |
| **L₃ (binary EML tree)** | Mathematical expression: "ζ(3) × π / φ²" | Manager evaluates at 361 dps via MPFR/Arb, ingests the result |
| **Direct value** | 361-dps decimal string | Ingests through §7.11 directly |
| **Projection** | (N, k, d, ε) tuple | Stored directly at the specified address |
| **Bijection triple** | (value, expression, lattice_address) | All three representations stored and linked |

Real-time preview: as you type in any mode, the Manager shows the projected lattice address BEFORE you submit. You see where it will land before committing.

**Real-time update behavior during navigation:**

When the background discovery engine creates a new pattern, attractor, or generator while the user is navigating:

- New content appears in the visualization at its lattice address WITHOUT disrupting the current view position, zoom level, or freecam orientation
- New entities are highlighted with a brief pulse animation (configurable, can be disabled)
- A notification counter increments on the dashboard ("3 new discoveries this session")
- The property inspector does NOT auto-switch to the new entity — the user's current selection is preserved
- If the user is currently viewing the exact address where the discovery occurred, the new entity appears inline immediately

The visualization is a LIVE view of the .akashic file. Changes to the file appear in the view. But the USER controls navigation — the system never hijacks focus.

**Progressive fidelity display (enabled by significance-ordered ε — §7.1d):**

When navigating large lattice regions with millions of entries, the GUI uses progressive precision rendering:

1. **Instant structural display (microseconds):** Read only the structural headers (k, d) for all entries in the view frustum. Display entries as coarse markers at their lattice positions. The user sees the full structural layout immediately — which families are populated, where attractors cluster, the shape of the address space.
2. **Classification precision (sub-millisecond):** Read the first 4-8 bytes of each entry's significance-ordered ε. Display entries with enough precision for d-family classification, color coding, and coarse value labels. The user can begin making navigation decisions.
3. **Full precision on dwell/hover (milliseconds to seconds):** When the user hovers over or selects a specific entry, read the full 150-byte ε blob. Display the complete 361-dps value, all materialized properties, and full provenance chain. Full precision loads in the background while the user navigates.
4. **Adaptive LOD (level of detail):** Zoom level determines which precision tier to load. Zoomed out = tier 1 (structural only). Zoomed to family level = tier 2 (classification). Zoomed to individual entries = tier 3 (full precision). The GPU LOD system (§7.10 visualization) maps directly to ε precision tiers.

This makes navigating a million-entry lattice feel as responsive as navigating a thousand-entry lattice — the difference is precision resolution, not response time.

**3D perspective views** — the same lattice data rendered through different Sempaevum geometries, user-selectable:
- **LCM tower** (default): k on one axis, d on another, N as depth — the escalation structure
- **Torus**: k wrapped cyclically, ε as radial displacement — the Sempaevum's periodic structure
- **Riemann sphere**: conformal projection of lattice addresses — the Sempaevum's projective structure
- **Force Quadrant Grid**: 12×12 interactive cells (d_r × d_θ), occupancy and coupling color-coded — the complex-plane interaction structure
- **Hyperbolic**: negative-curvature rendering for deep-extended families

All perspectives show the same data at the same precision. Switching perspective is a coordinate transformation, not a data reload.

**Color coding** (consistent across all views): d-family colors match the Sempaevum Particle Viewer conventions — d=1 green, d=2 teal, d=3 red, d=4 blue, d=6 purple, d=12 orange. Extended families (d=5, 7, 8, 9, 10, 11) get distinct hues. Attractor clusters are highlighted. Cross-domain hits at shared addresses use multi-color indicators.

**The EUDD is a virtual isomorphism of the Sempaevum.** It is not a passive database that stores and retrieves. It IS the Sempaevum virtualized on hardware — a computation engine that stores everything, computes anything (memoized at 361 dps), discovers generators, serves any ET software, ingests any file, accepts manual input, and returns whatever is needed: values, projections, generators, computations, files, structural context. If the fractal generator needs a value for a new render, it queries the EUDD. If Mike wants to pull a file out, the EUDD produces it. If the compressor needs a generator for K-complexity minimization, the EUDD provides it. If a new derivation needs intermediate results, the EUDD computes them (memoized) or retrieves them (cached). The EUDD IS the central computation substrate for all ET work.

**Operational lifecycle:**

1. **First run**: Bootstrap module generates initial `Sempaevum.akashic` → Discovery engine analyzes → Generator form produced → GUI opens for exploration
2. **Normal operation**: GUI dashboard shows live metrics. Any ET program connects via API module (named pipe IPC) and reads/writes through the Manager. Discovery engine runs continuously. Self-recording captures all operational metrics. New values project, find addresses, link to generators.
3. **Adding new data**: Three pathways (see §7.11 and §7.12):
   - **File ingestion**: feed any file → seeds extracted → projected → populated
   - **Manual input**: type a value, projection, or bijection triple → full lattice population
   - **Program API**: any ET software sends values/equations through the API → projected → populated
   In all cases: new values project at 361 dps, find lattice addresses, check generator membership. If covered by known generator → generator instance recorded. If not → raw entry stored, pending generator discovery. Discovery engine periodically scans raw entries for new generator patterns.
4. **Querying, computation, and retrieval**: Any query → lattice operation → human-readable result. The EUDD both retrieves stored results AND computes new ones (memoized). "What's at d=693?" → list of generators and values at that address. "Compute ζ(3)·π" → cache hit or fresh computation at 361 dps, stored forever. "What does the fractal generator need for R₀ = φ?" → structural context retrieved. "Pull the particle classification data" → formatted output. "Regenerate hydrogen_spectrum.csv" → Manager evaluates covering generators → outputs the regenerated file in original format. "Replay the GPS stream from yesterday" → Manager reconstructs the temporal sequence from generators at original or enhanced resolution. All answers come from the lattice structure itself — the .akashic file IS a generator that re-derives its content on demand.

### 7.11 Core projection procedure — 12ET escalation through the LCM tower

**This is THE operational procedure for every value entering the EUDD, regardless of how it arrives (file, manual, API, bootstrap).** The Four Paths (A, B, C, D) determine how a value enters; the tower escalation determines where it lives.

**Step 1 — Determine the input path.**

| Path | Input type | Action |
|---|---|---|
| A | Direct dimensionless ratio r = Q_X/R₀ | Accept r directly |
| B | Convergent series/limit | Compute r at 361 dps, then treat as Path A |
| C | Structural/geometric descriptor | Construct r from the object's D-content, then treat as Path A |
| D | Essentially-infinite / non-computable | Structural placement via sub-path (D.P, D.D, D.T, D.PDT) — no limit needed |

For Paths A/B/C, the value r is now known at 361 dps. Proceed to Step 2.

**Path D — Explicit sub-path procedures (Sempaevum Paper §15, Theorem 15.2):**

Path D handles objects whose essential character is infinity-valued — where no limit or convergent series produces the value without information loss. Each sub-path has its own finite-operation procedure that terminates at a specific lattice address without invoking limits. The Four-Path Subsumption Theorem (Sempaevum Paper Theorem 15.3) guarantees every possible input falls into exactly one path.

**D.P — Continuous, uncountable, or non-computable positive reals** (Chaitin's Ω, generic reals, specific irrationals of unknown computability):

The object IS a positive real r, even though it may be non-computable. The known bits/digits (however many have been computed) provide r at finite precision. Procedure:

1. Evaluate r at the best available precision up to 361 dps (e.g., Chaitin's Ω = 0.00787499699781238... for the Calude-Dinneen-Shu UTM — known to 64 exact binary bits, ~19 decimal digits)
2. Apply the projection formula directly: k = round(N·log₂(r)), d = N/gcd(|k|, N), ε = (N·log₂(r) − k)·1200/N — this is finite arithmetic, no limit needed (Sempaevum Paper Theorem 15.2)
3. **Run the full tower escalation (Step 2) identically to Path A.** The home-finding algorithm applies without modification — the value has a definite lattice address at every resolution N
4. Record the manifold state as **{P,D} Unsubstantiated** — the descriptor is complete (the definition of the object), the address is determined, but no T (finite computational process) can produce further bits. The non-computability IS the {P,D} character
5. Record the precision available as metadata: `available_precision_dps` (the number of known digits). The 361-dps MPFR blob stores all known digits; remaining digits are zeros (the precision limit, not a rounding)

Canonical example: Chaitin's Ω at N=12 → (k=−84, d=1, ε=+13.794¢) — this is the base-resolution PROJECTION (d=1 is the low-resolution shadow). The actual home is d=87 = 3×29, found via CF analysis (convergent 608/87, quality a₄=157, ε=+0.001¢ — sub-Koide by factor 1955). The LCM tower never stabilizes for Ω (d changes at all 33 landmarks through lcm(1..97)) — the CF method identifies the home that the tower cannot reach (§3.18.38). Manifold state {P,D} Unsubstantiated.

**D.D — Unbound descriptor-constraint structures** (non-divisor-of-12 sublattice families, infinite axiom totalities, shadow forces):

The object is characterized by a d-family that is NOT a divisor of the current resolution N. At base N=12, the six shadow families {5, 7, 8, 9, 10, 11} and all composite d-values up to d=132 fall into this sub-path. Procedure:

1. Identify the object's structural d-family via the shadow diagnostic: what sublattice family does this object's structure belong to? This is determined by the object's D-content (e.g., five-fold symmetry → d=5; seven-fold → d=7; M-theory → d=11; biological cross-product → d=35=5×7)
2. Compute the minimum resolution where this d-family becomes native: N_min = lcm(12, d). For d=5: N_min = 60. For d=7: N_min = 420. For d=11: N_min = 132. For d=35: N_min = 420. For d=110: N_min = 27720. For d=132: N_min = 132
3. If the object also has a dimensionless ratio r (many D.D objects do — e.g., a biological ratio with five-fold symmetry), project r at N_min via Path A and run the tower escalation from N_min upward
4. If the object has NO dimensionless ratio (e.g., "the concept of five-fold symmetry itself" as a pure structural D-object), record as a value with `input_path = 'D.D'` and assign the lattice address at N_min where the d-family first becomes native. The (k, ε) are determined by whatever quantitative descriptor the user provides; if no quantitative descriptor exists, the value is recorded as a structural anchor with k=0, ε=0 at the family's native resolution
5. Record the manifold state: typically {D,T} Mediation (D operating through T's scaffold without fixed substrate) or {P,D,T} Exception if a substrate-bound instance is identified

Canonical example: Icosahedral virus capsid symmetry → d=5 → N_min = 60. The Caspar-Klug T=5 capsid has r = 60 subunits / R₀ → project at N=60 via Path A from there.

**D.T — Indeterminate forms and genuine Traverser-agency objects** ([0/0], [∞/∞], [0×∞], [∞−∞], oscillatory divergent limits, measurement operators):

These objects are characterized by T's irreducible indeterminacy — they are the Traverser's own navigation territory. Procedure:

1. Classify the indeterminate form by type: [0/0], [∞/∞], [0×∞], [0⁰], [1^∞], [∞⁰], [∞−∞]
2. Attempt L'Hôpital resolution: iterate the derivative-pair procedure (the T-navigation algorithm per Sempaevum Paper §16.2). Each iteration is an `indeterminate_form_detected` event with full provenance
3. **If L'Hôpital resolves** (converges to a finite value r after n iterations): the indeterminate was derivative-resolvable. Record n iterations via `lhopital_iteration_chain` relationship. The resolved value r enters via Path A — run the full tower escalation from Step 2. The resolution event is a T-act: T navigated through the indeterminate and selected a value
4. **If L'Hôpital fails to resolve** (max iterations reached without convergence): this is **pure T** — the irreducible Traverser. Pure T has no single lattice address because T's cardinality [0/0] is substrate-independent. Record as a value with `input_path = 'D.T'`, manifold state {P,T} Incoherence (the forbidden state — no D-coordinates exist without D). NO projection row is created. The `lhopital_resolution_signature` pattern tracks where pure T resides. The object is at ∂I — it IS the boundary
5. **Freedom points** (genuine [0/0] at half-integer lattice positions): when a projection lands at exactly k + 0.5 (equidistant between two lattice points), T faces absolute freedom — two choices with zero structural preference. Record as `freedom_point_encounter` event with metadata: which two k-values are equidistant, which was chosen, and the basis for the choice (random, context, prior momentum). Real-axis frequency ~1/25 per step; imaginary-axis frequency ~1/2 per step (per ET_Freedom_and_U1.md). The freedom point is itself a D.T object — a Traverser choice event substantiated on the lattice
6. **Annihilation boundary** (r → 0): when a computation approaches r = 0, k → −∞ and d → undefined. This is the off-lattice infimum of (ℝ⁺,×) — the cardinality singularity. Record as `annihilation_boundary_event`. No projection row (r=0 is off-lattice by construction). Record the closest k reached before the singularity

**D.PDT — Integrated off-axis Exception objects with both magnitude and phase** (physical particles, complex-valued observables, any object requiring the full complex lattice):

These objects live at off-axis positions on the complex lattice L_C — they have BOTH real-axis (FORCE, D-domain) and imaginary-axis (PHASE, T-domain) character simultaneously. This is the {P,D,T} Exception state at an off-axis lattice position. Procedure:

1. Project on BOTH axes:
   - Real axis: k_r = round(N·log₂|r|), d_r = N/gcd(|k_r|, N), ε_r = (N·log₂|r| − k_r)·1200/N
   - Imaginary axis: k_θ = round(N·θ/(2π/N)), d_θ = N/gcd(|k_θ|, N), ε_θ = (N·θ/(2π/N) − k_θ)·1200/N, where θ is the phase/argument
2. Compute the combined family: d_combined = LCM(d_r, d_θ). This determines the Force Quadrant Grid cell: (d_r, d_θ) → cell in the 144-cell 12×12 grid
3. **Run the tower escalation on BOTH axes simultaneously.** The home-finding algorithm applies independently to d_r and d_θ:
   - d_r stabilization: same d_r across ⌈1/K⌉ = 2 consecutive LCM landmarks on the real axis
   - d_θ stabilization: same d_θ across ⌈1/K⌉ = 2 consecutive LCM landmarks on the imaginary axis
   - The home is found when BOTH axes have stabilized
   - d_combined = LCM(d_r, d_θ) is the combined home family
4. Record both projections as separate `projections` rows with `geometric_perspective = 'real_axis'` and `'imaginary_axis'` respectively, plus a combined `'complex'` perspective row
5. Assign to the Force Quadrant Grid cell: update `force_grid_cells` occupancy for (d_r, d_θ). Classify the FQG quadrant:
   - SR (Simple Real): d_r | 12, real axis
   - CR (Complex Real): d_r ∤ 12, real axis
   - SI (Simple Imaginary): d_θ | 12, imaginary axis
   - CI (Complex Imaginary): d_θ ∤ 12, imaginary axis
6. The cascade stability asymmetry (Sempaevum Paper §12.2) applies: the real axis has n_max_r = 25 stability levels; the imaginary axis has n_max_θ = 2. The imaginary-axis d_θ stabilizes faster (fewer coherent steps) but with less precision. This asymmetry is structural, not a limitation

Canonical example: The electron at N=12 → (d_r=12, d_θ=6), d_combined = LCM(12,6) = 12. FQG cell (12,6) in the SR×SI region. Off-axis Exception — the actual content of physical reality.

**Incoherence Filter:** If the input is a {P,T} Incoherent configuration (self-defeating, no D-structure), the Incoherence Filter fires: no lattice address is assigned (the lattice IS D-structure; without D no coordinates exist). The object is recorded as a value with `input_path = 'P.T'`, tagged `namespace='structural_role'`, `value='forbidden_incoherence'`. No projection row created. The Incoherence sits at ∂I — it IS the boundary, structurally present but nowhere on the lattice. This is not an error or a missing case — it is the correct structural classification of a self-defeating configuration.

**After Path D sub-path processing:** All sub-paths except D.T-pure-T and D.D-pure-structural produce either a definite lattice address or an explicit ∂I/annihilation-boundary classification. For sub-paths that produce a lattice address (D.P, D.D with ratio, D.T resolved, D.PDT), proceed to Step 4 (record trajectory). For sub-paths that produce boundary classifications (D.T pure-T, annihilation boundary, Incoherence Filter), record the structural classification and proceed to Step 6 (memoize).

**Step 2 — Start at 12ET. Escalate through the LCM tower indefinitely.**

Project at N=12 first. Then escalate through the canonical LCM landmarks, recording the full trajectory. **The tower does not terminate.** The LCM tower extends to infinity (Sempaevum Paper Proposition 7.2: L_N → (ℝ⁺,×) as N → ∞). The doubling law τ(N_ℓ) = 6·2^ℓ holds for ALL ℓ ≥ 0. The algorithm escalates until d stabilizes — however many landmarks that requires.

```
N=12      → (k₁₂, d₁₂, ε₁₂)         τ= 6  families {1,2,3,4,6,12}
N=60      → (k₆₀, d₆₀, ε₆₀)         τ=12  (adds prime 5)
N=420     → (k₄₂₀, d₄₂₀, ε₄₂₀)      τ=24  (adds prime 7, d=35=5×7)
N=2520    → (k₂₅₂₀, d₂₅₂₀, ε₂₅₂₀)   τ=48  (adds 2³, 3²; completes d≤9)
N=27720   → (k₂₇₇₂₀, d₂₇₇₂₀, ε₂₇₇₂₀) τ=96  (adds prime 11; all d∈{1..12} native)
N=360360  → (k₃₆₀₃₆₀, ...)            τ=192 (adds prime 13)
N=12252240 → ...                        τ=384 (adds prime 17)
N=232792560 → ...                       τ=768 (adds prime 19)
...continue forever until home found (d-family stabilized)
```

The canonical LCM landmark generator produces lcm(1..k) for successive k = 4, 5, 7, 9, 11, 13, 17, 19, 23, 29, ... — yielding only those k values where the lcm CHANGES (a new prime or prime-power enters). This is the sparse subsequence where τ doubles per the doubling law (Sempaevum Paper Theorem 14.2). The generator is unbounded.

Between LCM landmarks, the multiplicative reading (multiples of 12: 24, 36, 48, ...) can also be checked for uniform precision refinement within the same family structure. The LCM reading introduces new primes as native divisors; the multiplicative reading refines ε within existing families.

**Cross-Resolution Transition Map (§3.18.19 Theorem 1) — escalation WITHOUT re-accessing r:**

Each step of the tower escalation is computed via the Cross-Resolution Transition Map, NOT by re-projecting the original value r. Given (k₁, d₁, ε₁) at N₁, the projection at N₂ (where N₁|N₂, M=N₂/N₁) is:

    δ₁ = ε₁ · N₁ / 1200
    k₂ = round(M · k₁ + M · δ₁)
    d₂ = N₂ / gcd(|k₂|, N₂)
    ε₂ = (M · k₁ + M · δ₁ − k₂) · 1200 / N₂

This is an algebraic identity — the transition produces EXACTLY the same (k₂, d₂, ε₂) as direct projection Π_N₂(r), verified across 30 transitions at 200-digit precision. The original value r is NEVER needed after the first projection at N=12. The seed at base resolution IS the complete description — all higher resolutions are algebraically derivable via pure lattice arithmetic on (k, ε) pairs. This strengthens §3.1c: the K-minimal description is the base-resolution seed alone.

**Operational consequence:** Tower escalation is O(1) arithmetic per step (integer multiply + round + gcd), not O(precision) for an MPFR log₂ recomputation. The full trajectory from N=12 to N=27720 is 5 transition steps, each a few microseconds. For the Seed Protocol: transmitting a seed at N₁ is sufficient — the receiver computes all higher resolutions locally.

**Cross-Seed and Full Cross-Tower transitions** (§3.18.19 Theorems 2-3): When R₀ also changes (cross-tower queries, different sensor domains), the seed shift Δk_exact = N·log₂(R₀/R₀') is added to the position before rounding. The full transition factors as (Seed∘Scale) or (Scale∘Seed) — both give the same result (commutativity, Theorem 4). Cross-tower analysis never needs to re-access original values.

**d-family transition boundary** (§3.18.19 Theorem 5): When escalation pushes k₂ across a gcd-boundary of N₂, the d-family changes. This is the ε→d conversion: shadow content encoded in ε₁ at lower resolution becomes native structural content in d₂ at higher resolution. The discovery engine tracks these transitions via `d_family_transition` events. Example: muon (206.768) escalates d=3→10→140→120→315→3080 through the tower — each transition reveals structural identity invisible at lower resolution.

**Lattice-exact invariance** (§3.18.26 Theorem E2.2): Configurations with ε=0 (sitting exactly on lattice nodes) have their d-family PRESERVED across ALL higher resolutions — d-bouncing occurs ONLY for ε≠0. This is because k₂=M·k₁ exactly (no rounding), and gcd(M·k, M·N) = M·gcd(k,N). The d is a PERMANENT property for lattice-exact values. **Cell transition is ε-dependent** (Theorem E2.3): two configurations in the SAME sublattice cell at N₁ can map to DIFFERENT cells at N₂ if their ε values differ. The sublattice cell is a VIEWING, not a permanent address — the permanent address is the full (k, ε) coordinate.

**Step 3 — Classify the home using ET-derived thresholds.**

All thresholds are ET-derived. All comparisons are exact integer arithmetic on micro-cents (1 micro-cent = 10⁻⁶ cents). Zero floating-point comparisons.

**The stabilization criterion:** The Koide stability depth ⌈1/K⌉ = ⌈1/(2/3)⌉ = ⌈3/2⌉ = **2 consecutive LCM landmarks** with the same d-family. This is the same stability criterion used by the compressor's ArchetypeDatabase (depth-2 survivors only). One occurrence could be a false resolution (the true d-family has not yet been resolved at this N); two consecutive occurrences at increasing resolution confirm structural identity.

**The ε thresholds:** The Sempaevum's own defining constants {N, 1/N, K, 1/K} project to (d=12, |ε|=1.955¢) — the Koide attractor (Sempaevum Paper Theorem on self-consistency, §17). This gives the ET-native deep_home threshold: **|ε_micros| ≤ 1955** (the Sempaevum's own self-consistency residual, in micro-cents). Values at or inside this threshold are at least as tightly bound as the lattice's own constants. The ∂I boundary is at **|ε_micros| = 50000** (50¢ = half a semitone, the structural incoherency limit).

| Classification | ε Criterion (exact integer, micro-cents) | d Criterion | ET Derivation |
|---|---|---|---|
| **true_home** | eps_micros = 0 exactly | d stable across ⌈1/K⌉ = 2 consecutive LCM landmarks | Lattice-rational: r = 2^(k/N), structural exactness |
| **deep_home** | |eps_micros| ≤ 1955 | d stable across ⌈1/K⌉ = 2 consecutive LCM landmarks | Value is at or inside the Koide attractor — at least as tightly bound as the Sempaevum's own defining constants |
| **persistent_home** | 1955 < |eps_micros| < 50000 | d stable across ⌈1/K⌉ = 2 consecutive LCM landmarks | Value is inside the lattice (tightness > K) with stable classification, but outside the Koide attractor |
| **intermediate_home** | |eps_micros| < 50000 | d observed at current N but not yet stable across 2 consecutive LCM landmarks | Classified but not yet confirmed — needs further escalation |
| **false_resolution** | |eps_micros| ≤ 1955 at some N | d CHANGES at a higher LCM landmark | Sub-Koide ε at one resolution does not guarantee home — the true d-family may emerge at higher N |
| **escalation_in_progress** | — | d not yet stabilized; computation ongoing or paused for resumption | The home exists; the computation has not yet reached it. Partial trajectory recorded in the `.akashic` file for later resumption. |
| **cf_deep_home** | \|ε_CF_micros\| ≤ 1955 | d = CF convergent q_n | CF method: a_{n+1} ≥ ⌈1/K⌉² = 4; value locks to CF-identified d with sub-Koide residual (Step 3a) |
| **cf_home** | 1955 < \|ε_CF_micros\| < 50000 | d = CF convergent q_n | CF method: a_{n+1} ≥ 4; value locks to CF-identified d inside lattice, outside Koide attractor (Step 3a) |
| **cf_marginal** | any | d = CF convergent q_n | CF method: 1 < a_{n+1} < 4; CF convergent exists but below structural significance threshold (Step 3a) |

**There is no "home_not_found" classification.** The Asymptotic Approach Theorem (Sempaevum Paper Corollary 7.9) guarantees |ε| → 0 as N → ∞. The d-family WILL stabilize at sufficient resolution. The only question is whether the computation has reached that resolution yet. If not, the trajectory is saved and resumed later. The algorithm never gives up.

**False resolution detection:** After finding any sub-Koide hit (|eps_micros| ≤ 1955) at any N, continue escalating for at least ⌈1/K⌉ = 2 additional LCM landmarks. If the d-family changes at a higher landmark, the earlier hit was a false resolution — recorded as such with full provenance. The canonical case: φ has |ε| = 240 micro-cents at 36ET (d=36, sub-Koide) but the true home d=10 emerges at 60ET when prime 5 becomes native. The false resolution detection rule catches this by requiring 2 confirming landmarks past any sub-Koide hit.

**Session management (not termination):** The algorithm has no termination criterion. For practical operation, the Manager runs the escalation as far as it can in the current session, records the partial trajectory in the `.akashic` file as an `escalation_in_progress` entry, and resumes in a later session from the last computed landmark. The memoization layer preserves all intermediate projections. The background discovery engine can run escalations for pending values between user queries. This is session management, not a cap — the algorithm runs until the lattice answers, across as many sessions as needed.

**Step 3a — Continued Fraction Home-Finding (the CF method)**

The LCM tower escalation of Steps 2–3 succeeds for values whose irrationality measure is bounded — values where d eventually stabilizes because the lattice's divisor structure eventually captures the rational approximation structure of log₂(r). For most mathematical and physical constants (φ, ζ(3), π, e, α⁻¹, particle mass ratios), the tower method works.

However, for **algorithmically random values** — values whose irrationality measure is effectively unbounded — the LCM tower fails structurally: every new prime p entering the LCM landmark lcm(1..p) introduces a new factor into d, so d changes at every landmark and **never stabilizes**. This is not a computational limitation; it is a structural consequence of algorithmic randomness. The value's binary expansion has no exploitable pattern in the integer rounding sequence, so the gcd(|k|, N) never locks onto a stable divisor.

The Descriptor Gap Principle identifies this as a missing Descriptor: the LCM tower's stabilization criterion is the wrong Descriptor for these values. The right Descriptor is the **continued fraction expansion** of |log₂(r)|, which provides a canonical, ET-native decomposition of any real number into its rational approximation hierarchy — a hierarchy that exists independent of any particular N.

**The CF method — structural logic:**

For any positive real r, compute the continued fraction (CF) expansion of |log₂(r)|:

$$|\log_2(r)| = a_0 + \cfrac{1}{a_1 + \cfrac{1}{a_2 + \cfrac{1}{a_3 + \cdots}}}$$

Each convergent $p_n/q_n$ of this CF is the best rational approximation to |log₂(r)| with denominator ≤ q_n (by the classical theory of continued fractions). The **quality factor** of the n-th convergent is the following partial quotient $a_{n+1}$: it measures how many denominators must be tried before a better approximation exists. High $a_{n+1}$ means the lattice "locks on" to the rational p_n/q_n with exceptional tightness — the value resonates structurally with the d = q_n sublattice family.

**The CF home-finding criterion:**

The CF home is the convergent $p_n/q_n$ whose following partial quotient $a_{n+1}$ is **maximal** among all convergents. This convergent identifies d_home = q_n as the home sublattice family.

The ET-native justification: the quality factor $a_{n+1}$ measures **structural resonance** with the lattice. When $a_{n+1}$ is large, the value's log₂ projection sits extraordinarily close to the rational p_n/q_n — meaning the lattice at resolution N = q_n (or any N divisible by q_n) captures this value with ε approaching zero. The quality factor IS the Descriptor that measures how tightly D binds to P at this rational approximation — and the maximum quality convergent is where this binding is tightest. This is the Identification Principle applied to home-finding: the substrate (P) is the value r, the constraint (D) is the CF convergent p/q, and the agency (T) is the act of selecting the best convergent.

**The CF ε computation:**

Once the CF home convergent $p_n/q_n$ is identified with quality $a_{n+1}$, the CF residual is:

$$\varepsilon_{\text{CF}} = (|q_n \cdot \log_2(r)| - p_n) \times \frac{1200}{q_n} \quad \text{cents}$$

This is the standard projection formula at N = q_n, measuring how far the value is from perfect lattice placement at the CF-identified resolution.

**CF quality-to-elegance mapping:**

The quality factor $a_{n+1}$ maps naturally to the elegance framework. The CF elegance contribution is:

$$\mathcal{E}_{\text{CF}}(r) = \frac{a_{n+1}}{a_{n+1} + 1} \times \frac{N}{d_{\text{home}}} \times \frac{100}{100 + |\varepsilon_{\text{CF}}|}$$

where the first factor approaches 1 as quality increases (high quality = high elegance). For Ω with $a_{n+1} = 157$: the CF elegance factor is 157/158 ≈ 0.9937 — near-maximal. This provides a **quantitative ranking of home-assignment strength** that the LCM tower alone cannot produce: values at the same d-family are further distinguished by their CF quality.

**When the CF method fires:**

The CF method is not a fallback triggered by tower failure. It is a **parallel pathway** computed for every value at the same time as the tower escalation. This ensures:

1. **For tower-convergent values** (φ, ζ(3), etc.): The tower stabilizes d normally. The CF method independently confirms the same d-family AND provides the quality metric. The CF quality for tower-convergent values is typically moderate (a_{n+1} in the range 1–20), confirming that the tower's stabilization is genuine.

2. **For tower-resistant values** (Ω, algorithmically random constants): The tower never stabilizes. The CF method identifies the home d-family that the tower cannot reach. The `escalation_in_progress` classification is upgraded to `cf_home` or `cf_deep_home` with full CF provenance.

3. **For false resolution detection**: When the tower produces a sub-Koide hit at some N but the CF method shows a different d-family with higher quality, the CF result flags the tower hit as a likely false resolution — an independent structural check beyond the "2 additional landmarks" rule.

**CF home classifications (additions to the Step 3 table):**

| Classification | ε Criterion | d Criterion | CF Criterion | ET Derivation |
|---|---|---|---|---|
| **cf_deep_home** | \|ε_CF_micros\| ≤ 1955 | d = q_n (CF convergent denominator) | a_{n+1} ≥ ⌈1/K⌉² = 4 (quality exceeds squared Koide depth) | Value locks to CF-identified d-family with sub-Koide residual AND structurally significant quality. The threshold ⌈1/K⌉² = 4 is the Koide depth squared — a natural ET-derived quality floor |
| **cf_home** | 1955 < \|ε_CF_micros\| < 50000 | d = q_n | a_{n+1} ≥ ⌈1/K⌉² = 4 | Value locks to CF-identified d-family inside the lattice with significant quality, but outside the Koide attractor |
| **cf_marginal** | any | d = q_n | 1 < a_{n+1} < ⌈1/K⌉² = 4 | CF convergent exists but quality is below the structural significance threshold — record but do not treat as definitive home |

The CF quality threshold ⌈1/K⌉² = ⌈3/2⌉² = 2² = **4** is the natural ET-derived minimum: it squares the Koide stability depth, requiring the CF lock-on to be at least as structurally significant as the tower's depth-2 criterion raised to the next integrative level. The threshold is not ad hoc — it is ⌈1/K⌉ applied twice: once for depth, once for quality.

**Canonical example — Chaitin's Ω:**

The LCM tower fails for Ω: d changes at every landmark through 33 landmarks up to lcm(1..97). The CF method succeeds:

- CF expansion of |log₂(Ω)| yields convergent n=3: p/q = 608/87
- Following partial quotient a₄ = 157 (quality factor)
- d_home = 87 = 3 × 29
- ε_CF = +0.001003¢ = 1.003 micro-cents
- This is sub-Koide by factor **1955 exactly** — the Koide residual itself (|ε| = 1.955¢ / 1955 = 0.001¢)
- Quality 157 dwarfs all other convergents' qualities, unambiguously identifying d = 87
- Classification: **cf_deep_home** (sub-Koide residual AND quality 157 >> 4)
- Gaussian signature of d=87: 87 = 3 × 29; 3 ≡ 3 mod 4 (D-type inert), 29 ≡ 1 mod 4 (D+T split) → mixed inert-split character
- The CF quality factor 157 maps to CF elegance: 157/158 × (N/87) × (100/100.001) — near-maximal structural resonance

**Integration with the elegance framework:**

The CF quality factor provides a dimension of structural information the tower cannot:

- **Tower-only values** have elegance computed from (N/d) × tightness × simplicity — no quality dimension
- **CF-characterized values** have an additional **CF quality elegance factor** = a_{n+1}/(a_{n+1}+1) that ranks how strongly the value is locked to its home family
- The `projections` table gains a new stored field: `cf_quality INTEGER` — the CF quality factor a_{n+1} (NULL for tower-convergent values, populated for all values where CF analysis was performed)
- The `values` table gains: `cf_home_convergent_p INTEGER`, `cf_home_convergent_q INTEGER`, `cf_quality INTEGER` — the CF convergent p/q and quality factor identifying the value's CF home

**CF events and relationships:**

New event class: `cf_home_identified` — fires when the CF method identifies a home d-family for a value. Metadata: `{convergent_index_n, p, q, quality_a_next, eps_cf_micros, d_home, gaussian_signature, all_convergents_blob}` — captures the full CF analysis including all convergents for audit.

New event class: `cf_tower_disagreement` — fires when the CF method identifies a different d-family than the LCM tower's current best candidate. Metadata: `{tower_d, tower_N, cf_d, cf_quality, tower_false_resolution_probability}` — this is a structural diagnostic indicating the tower hit is likely a false resolution.

New relationship class: `cf_convergent_home` — links a value to its CF-identified home d-family with full CF provenance. Metadata: `{convergent_n, p, q, quality, eps_cf_micros, cf_elegance}`.

New relationship class: `cf_tower_confirmation` — links a value's CF home to its tower-stabilized home when both methods agree. Metadata: `{cf_d, tower_d, agreement_confirmed, cf_quality, tower_landmark_count}`. When both methods agree, the home classification is maximally confirmed.

New pattern class: `cf_quality_attractor` — recurring pattern of values from different domains sharing the same CF quality factor or CF home d-family. When multiple values lock onto the same d via the CF method with similar quality, this is a structural attractor in the CF space.

At each N in the escalation, evaluate:

The full escalation produces a TRAJECTORY — the sequence of (N, k, d, ε, home_classification) at each resolution. This trajectory is itself a structural object:

- Stored as a `tower_trajectory` pattern in the database
- Each step stored as a projection entry at the corresponding N
- Home classification recorded as a `home_classification` relationship at each N
- False resolutions explicitly flagged
- The trajectory reveals which d-families the value visits, where it stabilizes, where it transitions, and what its structural character is

**Step 5 — Populate all structural context at every resolution visited.**

At EVERY N in the trajectory (not just the home), materialize:
- All derived properties (elegance, coupling, variance, FQG quadrant, palindromic partner)
- Attractor membership (is this address already occupied at this N?)
- Generator coverage (does a known generator produce this value?)
- Nearest neighbors (what known values are structurally closest at this N?)
- Cross-domain relationships (does this value share an address with anything from another domain at this N?)
- d-family membership across both axes (real and imaginary, if applicable)

**Step 6 — Memoize everything.**

Every projection computed in the escalation, every derived property, every relationship discovered — stored permanently at 361 dps. The next time any program asks about this value at any resolution, it's a cache hit.

**The complete operational flow:**

```
Value enters (Path A/B/C/D)
    ↓
[Path A/B/C] Evaluate at 361 dps (MPFR 1200-bit, zero floats) → r known
[Path D] Structural placement → resolution determined
    ↓
Project at N=12 → record (k, d, ε in micro-cents, properties, relationships)
    ↓
Classify step at 12ET (integer comparison on eps_micros)
    ↓
    ┌───────────────────────────────────────────────────────────────────┐
    │ PARALLEL: CF Home-Finding (§7.11 Step 3a)                        │
    │                                                                   │
    │ Compute continued fraction of |log₂(r)|                          │
    │ For each convergent p_n/q_n, record a_{n+1} (following partial   │
    │   quotient = quality factor)                                      │
    │ Identify the convergent with maximal a_{n+1}                      │
    │ d_cf = q_n from that convergent                                   │
    │ ε_cf = (N·log₂(r) − p_n) × 1200/N cents                          │
    │ E_CF = a_{n+1}/(a_{n+1}+1) × (N/d_cf) × tightness              │
    │                                                                   │
    │ If a_{n+1} ≥ ⌈1/K⌉² = 4:                                        │
    │   cf_deep_home  (|ε_cf| ≤ 1955 micro-cents AND quality ≥ 4)      │
    │   cf_home       (|ε_cf| < 50000 micro-cents AND quality ≥ 4)     │
    │   cf_marginal   (quality < 4 — weak lock, continue tower)         │
    │ → fire cf_home_identified event                                   │
    │ → record cf_home_convergent_p/q/quality on values row             │
    │ → record cf_quality on projection row                             │
    │ → if tower also converged: fire cf_tower_confirmation or          │
    │   cf_tower_disagreement as appropriate                            │
    └───────────────────────────────────────────────────────────────────┘
    ↓ d not yet stable across ⌈1/K⌉ = 2 consecutive LCM landmarks
Escalate to N=60 → record everything
    ↓
Classify step at 60ET, check d-stability
    ↓ not yet stable (or verifying past sub-Koide hit)
Escalate to N=420 → record everything
    ↓
...continue through LCM landmarks indefinitely...
    (the tower is infinite — lcm(1..k) for k=4,5,7,9,11,13,17,19,23,29,...)
    (the doubling law τ(N_ℓ) = 6·2^ℓ governs growth at each level)
    (there is NO termination — the algorithm runs until d stabilizes)
    (the CF method runs in parallel at the FIRST landmark — it does not
     need the tower to stabilize; it works on r directly)
    ↓
Home found via LCM tower: d stable across 2 consecutive landmarks + verified across 2 more
    → record as true_home / deep_home / persistent_home
    (true_home: eps_micros = 0 exactly)
    (deep_home: |eps_micros| ≤ 1955 — at or inside the Koide attractor)
    (persistent_home: 1955 < |eps_micros| < 50000 — inside lattice, outside Koide attractor)
    (Irrationals DO find homes — φ at d=10, ζ(3) at d=693. The home is the d-family
     stabilization, not ε=0. The Asymptotic Approach Theorem guarantees ε→0 as N→∞
     but ε>0 for irrationals at every finite N. The structural classification IS determined
     when the d-family stabilizes across consecutive LCM landmarks.)
    → if CF also fired: compare. Agreement = cf_tower_confirmation relationship.
       Disagreement = cf_tower_disagreement event (investigate).
    ↓
OR: Home found via CF method only (tower never stabilized):
    → this is the pathway for algorithmically random values (Ω, BBP-type constants,
     values with effectively unbounded irrationality measure)
    → the tower records false resolutions and escalation_in_progress
    → the CF method provides the structural home that the tower cannot
    → record as cf_deep_home or cf_home depending on ε and quality
    → the quality factor a_{n+1} quantifies confidence in the assignment
    → canonical example: Ω (Chaitin's constant) — tower fails through 33 landmarks
       up to lcm(1..97), every new prime changes d; CF gives p/q = 608/87,
       a₄ = 157, d = 87 = 3×29, ε = +0.001003¢ (sub-Koide by factor 1955 exactly)
    ↓
OR: session ends before home found (by either method)
    → record as escalation_in_progress with partial trajectory in the .akashic file
    → resume in next session from last computed landmark (all intermediate projections memoized)
    → the algorithm NEVER gives up; the home WILL be found across as many sessions as needed
    ↓
Record complete tower trajectory as pattern
    ↓
Check for false resolutions in the trajectory
    (sub-Koide hits where d changed at a higher landmark → mark false_resolution)
    ↓
All data memoized permanently at 361 dps (MPFR 1200-bit, zero floats, zero precision loss)
```

**This procedure runs for EVERY value entering the EUDD — bootstrap values, file-ingested seeds, manual inputs, API submissions, discovered generators, self-recorded metrics. No exceptions.**

**Implementation requirements for the CF method (Step 3a):**

- The C++ engine must implement continued-fraction expansion of |log₂(r)| at 361 dps (MPFR 1200-bit), extract the quality factor a_{n+1} for each convergent, and compute the CF-to-elegance mapping E_CF = a_{n+1}/(a_{n+1}+1) × (N/d) × tightness
- The API (§7.15 when specified) must expose CF-aware variants of the escalation command — `escalate` returns CF results (d_home, quality, ε_CF, classification) alongside the tower trajectory
- The test suite must include CF-specific test cases: Chaitin's Ω (expected: d=87, quality=157, ε=+0.001003¢, cf_deep_home), and other algorithmically random constants where the LCM tower is expected to fail and the CF method must succeed

### 7.12 File ingestion — feed it any file

The EUDD accepts any file as input. The file becomes a SOURCE of dimensionless seeds that project onto the Sempaevum. **All data is ingested through the §7.11 core projection procedure BEFORE it can be displayed.** Nothing is ever displayed raw — the GUI always shows ingested content from the .akashic file, never the raw input. This applies to static files and continuous streams alike.

The pipeline for every file: **extract seeds → run each seed through §7.11 (12ET escalation, home classification, false resolution detection, full trajectory) → store in .akashic → available for display and query.** One file → potentially thousands of fully-populated lattice addresses, each with complete tower trajectory and structural context.

Files can also be RETRIEVED from the EUDD. Any stored content that originated from a file retains its provenance (source file, extraction method, seed index). The EUDD can reproduce the structural content of any ingested file on demand.

**Seed-first ingestion — Δε versioning (§3.1c Kolmogorov Principle applied to storage):**

Before the full extraction pipeline fires, the Manager checks whether this file — or a structurally similar version — already has a seed in the archive. This is the Kolmogorov principle: store the shortest generating description, not redundant data.

**Step 0 — Seed-first check (runs before any per-file-type adapter):**

1. **Hash check:** Compute SHA-256 of the incoming file. Lookup in `values` table by `value_hash`. If exact match → file already ingested. Log `seed_deduplicated` event (dedup_type='exact'). Skip entirely. Zero work.

2. **Whole-file seed computation:** Interpret the file's byte sequence as a big-endian unsigned integer, normalize to ℝ⁺ (divide by appropriate R₀ — file size class or domain-specific reference), project via Π_N → (k, d, ε). This is the file's structural address on the lattice.

3. **Lattice address lookup:** Query `addresses` table for the computed (k, d) at resolution N. Three outcomes:

   - **No match:** Genuinely new file. Proceed to full extraction pipeline below. Store the whole-file seed as the base version. Log `seed_generated` event.

   - **Exact (k, d) match, identical ε:** Exact structural duplicate (different bytes, same lattice position — possible for files that differ only in metadata or padding). Log `seed_deduplicated` event (dedup_type='exact_structural'). Store provenance link only.

   - **Exact (k, d) match, different ε:** This is a **modified version** of a previously ingested file. Compute Δε = ε_new − ε_stored. Store ONLY the Δε as a version delta, linked to the base seed via `file_version_delta_stored` event and `seed_deduplication_delta` relationship. Cost: a few bytes for the Δε value + event/relationship metadata. The entire modified file is exactly reconstructible from base_seed + Δε via a single pullback: r_new = 2^((k + (ε_base + Δε) · N / 1200) / N). No full re-ingestion needed.

   - **Adjacent (k±1, same d) or (k, d±1):** Larger modification that shifted the lattice coordinate. Store the delta (Δk, Δε) pair. Still orders of magnitude smaller than the full file. Log `file_version_delta_stored` with the delta pair.

4. **Per-segment seed check (for files processed by the compressor's Δk pipeline):** When the compressor splits a file into segments, each segment gets its own (k, d, ε). A localized edit affects ONLY the modified segment's seed — unchanged segments produce identical seeds with Δε = 0 (literally zero storage cost for unmodified portions). The Manager tracks which segments changed via segment-level hash comparison, and stores Δε only for modified segments.

**Version history as Δε chain:**

A file's complete version history is: one base seed (k, d, ε_base) + a chain of Δε values, one per version. Each Δε is a few bytes. A thousand versions of a large file costs one base seed + a few kilobytes of Δε — vs a thousand full copies (conventional storage) or a thousand byte-level diffs (conventional VCS). This is the Kolmogorov-optimal version representation: the structural address never changes (it IS the file's identity on the lattice), only the residual moves.

**Reconstruction of any version:** base_ε + Σ(Δε₁..Δεₙ) = ε_version_n → pullback → exact file bytes. Or chain from adjacent versions: ε_prev + Δε_next → pullback. Either way, one pullback per reconstruction, zero error by algebraic identity.

**Per-file-type adapter specifications:**

**CSV:** Column selection via GUI dialog. Auto-detection attempted first: delimiter (try comma, tab, semicolon, pipe — whichever produces the most consistent column count), header row (first row is header if it contains non-numeric strings), R₀ (if column headers contain unit strings like "Hz", "Pa", "m/s", the Manager matches against bootstrap R₀ references from §3.17 and proposes the inferred R₀). Auto-detection results are presented to the user for confirmation before ingestion proceeds. If auto-detection fails or the user overrides: manual column picker (which column(s) contain values) + manual R₀ entry (361-dps decimal or selection from bootstrap catalog). Each selected cell value becomes one dimensionless seed via Path A (value ÷ R₀).

**PDF/Markdown:** Processed by the compressor's general-purpose Δk pipeline, the same as any other binary file. The raw bytes of the PDF become lattice content through that pipeline. No special extraction, no regex scanning, no equation parsing — the file IS the input, not a container to reach into. If specific numerical values from a document are needed (a mass ratio, a coupling constant, a verification value), they are entered directly via manual input (§7.13) or placed in a CSV — formats designed for specific data. The PDF as a whole file gets the same structural decomposition as any other file.

**Raw binary:** The compressor's existing Δk extraction pipeline processes the file. The adapter ingests the resulting Δk patterns and archetype matches into the EUDD. No new extraction logic needed — the compressor IS the binary adapter.

**Image:** Three extraction strategies, user-selectable via GUI:
- **Per-pixel**: r = pixel_value / max_channel_value (e.g., ÷255 for 8-bit). One seed per pixel. Produces large seed counts — suitable for small images or regions of interest.
- **Region-averaged**: user draws a grid overlay on the image; each grid cell's average pixel value becomes one seed. Produces manageable seed counts for large images.
- **Frequency-domain** (default): 2D FFT applied to the image. Peak spatial frequencies extracted as ratios to image dimensions. Structurally meaningful — spatial frequency content often carries lattice-relevant information that raw pixel values do not.

**Audio:** The raw audio waveform is a continuous signal projected onto the lattice via the lossless bijection (§3.18.1, continuous-discrete corollary) — no information loss, no Nyquist limit, no quantization floor. FFT with configurable window size (default: 4096 samples = 2^N, ET-derived from the digital tower's base resolution). Overlap: default 50%. Peak frequencies extracted as ratios to R₀ = sample_rate. Each ratio is projected through §7.11 and stored losslessly — the continuous frequency value IS exactly recoverable from the lattice address via the bijection pullback. Harmonic series detection: if extracted peaks form a harmonic series (f, 2f, 3f, ...), the fundamental frequency is the primary seed and the harmonics are recorded as power-pair relationships (k-scaling by integer n). One audio file → one set of spectral seeds per FFT window, with temporal evolution tracked as a sequence of projection events.

**Sensor data streams (continuous):** Continuous streams are the paradigm case of the continuous-discrete corollary (§3.18.1): each continuous reading is projected onto the lattice via the lossless bijection, eliminating the information loss inherent in conventional ADC sampling. Each reading is ingested through §7.11 as it arrives (real-time) or in batch (post-collection). R₀ selected from bootstrap reference catalog (§3.17) based on sensor domain — GPS uses Earth radius or light-time-second, atmospheric uses standard pressure/temperature, electrical uses reference voltage/current/impedance, etc. After each reading is ingested and stored in .akashic, the GUI can display the live lattice trajectory of the stream — but the display always shows FROM the .akashic file, never the raw stream directly. Anomaly detection: when a reading's projection lands at a d-family that is >2 standard deviations from the running d-family distribution for that sensor, `sensor_anomaly_detected` event fires. The threshold is statistical — it adapts to the sensor's actual behavior over time, not a fixed cutoff.

**Real-time lattice drift monitoring (§3.18.22 Differential Control Identity):**

For continuously evolving sensor readings r(t), the forward law (Theorem B.1) gives the exact relationship between physical drift rate and lattice drift rate:

    dε/dt = Λ · (ṙ/r)    where Λ = 1200/ln2 ≈ 1731.234049 (the manifold conversion constant)

Λ bridges the D-face (discrete lattice, cents) and the P-face (continuous substrate, nats). It is a new ET-derived constant with zero free parameters, projecting to (k=129, d=4, ε=9.102¢) at N=12. The identity operates on the RELATIVE rate ṙ/r — dimensionless and convention-independent (Theorem 7.5 in differential form). **For phase-domain sensors (angular, rotational, periodic signals), the imaginary-axis analog applies (§3.18.24 Theorem D.5): dε_θ = Λ_θ · dθ where Λ_θ = 600/π ≈ 190.986 (the phase conversion constant). Λ_θ operates on ABSOLUTE angle dθ (not relative), reflecting U(1)'s uniform sensitivity — the differential expression of T's positively curved manifold vs D's flat one. Ratio: Λ_r/Λ_θ = 2π/ln2 ≈ 9.065.** This enables:

- **Drift rate computation:** physical sensor drift dr/dt maps to lattice ε-drift via Λ — no numerical differentiation needed, the algebraic identity gives the exact rate
- **Cell-transition prediction:** when |ε| approaches 600/N, the drift rate predicts WHEN k will increment (time = distance to boundary / |dε/dt|), enabling preemptive event scheduling
- **Restoration control (Theorem B.4):** if ε drifts from a target ε₀, the control law dr/dt = −r·ln2·(ε−ε₀)/(1200·τ) drives ε exponentially back: ε(t) = ε₀ + (ε_init−ε₀)·exp(−t/τ). This is the healing layer's EXACT specification for active systems.
- **Sublattice palindrome (Theorem B.3):** as r evolves monotonically through consecutive cells at N=12, the d-family sequence is [1,12,6,4,3,12,2,12,3,4,6,12] — palindromic by gcd(k,N)=gcd(N−k,N). This is DISTINCT from the harmonic cascade ordering [12,6,4,3,12,2,12,3,4,6,12,1]. The sensor monitoring display shows the sublattice sequence in real time.

**Stream storage optimization (§3.1c Kolmogorov Principle applied to correlated streams):**

Correlated sensor streams — where successive readings share structural lattice properties — use seed-stream encoding to dramatically reduce per-reading storage cost:

- **Tower-level sharing:** All readings at the same resolution N → store N once as the stream header. Per-reading cost: zero bytes for N.
- **Shared k:** If successive readings share the same lattice coordinate k (same cell), store k once → subsequent readings store ONLY their ε value. A 10,000-reading stream at k=137 costs: 1 k-header + 10,000 ε values, not 10,000 full (k, d, ε) triples.
- **Delta-k encoding:** If k values are close but not identical (k₁=137, k₂=138, k₃=137...), store k₁ as the anchor → subsequent readings store Δk = k_i − k_{i−1} (1-2 bytes each, varint-encoded, instead of full ETInteger k per reading).
- **Sublattice-family grouping:** Group readings by d-family → store each group's family header once, then only the k and ε within that family. Family-grouped streams have better disk locality for d-family scan queries.
- **Combined:** For a typical sensor stream (10,000 readings, all near k≈137, d=12): 1 stream header (N, k_anchor, d) + 10,000 Δk+ε pairs at 3-4 bytes each ≈ 30-40 KB total vs 10,000 full projection rows at ~245 bytes each ≈ 2.4 MB. Compression: 60-80× over raw storage. Performance table (§3.18.18 §6): IoT/telemetry 3-6× vs raw, scientific sensors 4-8× — these apply to storage, not just transmission.

**Python/ETPL script output:** Numerical results captured from stdout or structured output. Each numeric value becomes one seed. The script's source file is recorded as provenance.

**Any other file:** The compressor processes it into lattice content via its general-purpose Δk pipeline. The resulting patterns and archetype matches are ingested. If the compressor cannot process the file type, the Manager rejects with an error and logs the rejection — the rejection IS a Descriptor Gap pointing to a missing ingest adapter (§7.14 `ingest_adapter` JSON extension can define a new one).

**New file types:** When the Manager encounters a file extension it doesn't recognize, it attempts the compressor's general-purpose pipeline. If that fails, the gap is logged and the user can define a new ingest adapter via the §7.14 JSON extension mechanism (`extension_type: "ingest_adapter"`).

**Automatic lattice-alignment detection (§3.18.18 Phase 3 applied to ingestion):**

Before committing to a specific ingestion strategy, the Manager scores incoming data for lattice alignment:

1. **Sample projection:** Project a sample of the data (first 100 values, or first 4096 bytes) through §7.11 at N=12.
2. **Alignment score:** Measure: (a) how many sample values share the same d-family, (b) the spread of k-values (σ_k), (c) the average |ε| in cents. High same-d percentage + low σ_k + low |ε| = high lattice alignment.
3. **Strategy selection:**
   - **High alignment** (>80% same d, σ_k < 10, avg |ε| < 50 cents): Full seed-stream encoding with shared-k/delta-k optimization. Maximum compression. Use the stream storage optimization above.
   - **Moderate alignment** (50-80% same d, σ_k 10-100): Hybrid — seed-stream for aligned portions, per-value projection for outliers.
   - **Low alignment** (<50% same d, σ_k > 100): Standard per-value ingestion through §7.11. No stream optimization.
   - **Structurally random** (uniform d distribution, |ε| near 600/N): Compressor's general-purpose Δk pipeline. Lattice-aware error resilience still applies (§7.9 graceful degradation), but no compression gain expected.
4. **The score itself is stored** as metadata on the ingestion event — it becomes discoverable content. If many files from the same source score similarly, the discovery engine may propose a source-specific optimal N.

### 7.13 Manual input — values, projections, bijection triples, text, and backbone operations

The GUI provides seven manual entry modes covering all three generator backbone layers (§7.1b) plus direct value, projection, bijection triple, and text entry. All modes produce the same result: content processed through the core projection procedure (§7.11) with complete tower trajectory and structural context. Real-time preview: as you type in any mode, the system shows the projected lattice address BEFORE you submit.

**Mode 1 — Direct value (L₃ backbone — binary EML tree):**
Type `1836.15267343` or `m_p/m_e` or `ζ(3) × π / φ²` or any mathematical expression. The system:
1. Determines the input path (A if numerical, B if convergent expression, C if structural descriptor)
2. Evaluates the expression at 361 dps via MPFR/Arb
3. Runs the full §7.11 escalation: 12ET → 60ET → 420ET → ... → home found (d-family stabilized)
4. Records complete tower trajectory with home classification at every N
5. Materializes all derived properties at every resolution visited
6. Checks attractor membership, generator coverage, nearest neighbors, cross-domain relationships at every N
7. Stores everything permanently
8. Displays the complete structural profile in the GUI — tower trajectory, home classification, family memberships, relationships

**Mode 2 — Logical description (L₁ backbone — logical pattern):**
Type a natural language or structured description: "the ratio of electron mass to proton mass", "Apéry's constant", "the 47th prime". The system resolves the description to a value at 361 dps, then runs the full §7.11 escalation. The L₁ description is stored as the generator's logical-layer representation.

**Mode 3 — Lattice operation (L₂ backbone — cascade/k-arithmetic):**
Type a lattice operation: "k=7 at N=12, escalate", "palindromic partner of d=3", "LCM tower from N=12 to N=27720 for r=φ". The system computes the operation, ingests all resulting values through §7.11. The L₂ operation is stored as the generator's cascade-layer representation.

**Mode 4 — Enter a projection:**
Type `k=130, d=6, ε=+12.16¢, N=12`. The system:
1. Uses the bijection pullback: r = 2^((130 + 12.16·12/1200)/12) → recovers the exact value at 361 dps
2. Runs the full §7.11 core escalation from 12ET through the LCM tower
3. Records complete tower trajectory, home classification, all structural context

**Mode 5 — Enter a bijection triple:**
Type `(k, d, ε)` at a specified N. Same as Mode 4 — the bijection pullback recovers r, then full §7.11 escalation.

**Mode 6 — Computation:**
Type any computable expression. The Manager evaluates at 361 dps (memoized — if this computation was done before, the result is retrieved from cache instantly). The result is ingested through §7.11. The computation itself is stored as an equation entry linking input and output addresses.

**Mode 7 — Text:**
Paste or type any text — a paragraph from a paper, a sentence, a label, a description, raw prose. The text is processed as raw bytes through the compressor's general-purpose Δk pipeline, the same as any binary file. The text becomes lattice content through structural decomposition of its byte stream. Provenance records that this content was entered as text via manual input. This is the correct pathway for inputting textual content that is not a mathematical expression (Mode 1), a logical description resolvable to a value (Mode 2), or a lattice operation (Mode 3) — it is the manual-input analog of file ingestion (§7.12), applied to user-provided text rather than a file on disk.

In all seven modes: **one input → full tower escalation → complete lattice population.** The Sempaevum does the rest. The core projection procedure (§7.11) is the same whether the input comes from a file, a manual entry, an API call, a stream, or the bootstrap. The manual input module is the human interface to the virtual Sempaevum.

### 7.14 Adaptive extension — the living database

The EUDD is a living system. It must accept and detect new categories, new structural types, and new organizational patterns as they emerge — without requiring format redesign or manual schema migration. This is the Descriptor Gap Principle applied to the database itself: **when the EUDD encounters something it cannot classify with existing categories, that gap IS a Descriptor pointing to a new category needed.**

**The exe must be running for all database connections.** The EUDD Manager is the gatekeeper. All reads and writes from all connected programs pass through the running Manager. This ensures:
- **Stability**: no concurrent direct-write corruption; the Manager serializes all mutations
- **Auto-detection**: the Manager monitors incoming data for patterns that don't fit existing categories and flags or auto-creates new ones
- **Consistency**: every write triggers the §7.11 core escalation procedure; no data enters without full structural processing
- **Self-recording**: every operation is itself recorded (§3.1b)

**Adaptive category extension via JSON — the full specification:**

New structural categories are defined via JSON files that the running Manager accepts. All JSON extensions are **strictly validated** — invalid JSON is rejected and never enters the system. There are **12 extension types**: 11 bootstrap types covering every extensible structure in the EUDD, plus one meta-type (`extension_type_definition`) that defines new extension types, making the set open-ended.

**Common fields required on ALL extension JSON files:**

```json
{
  "extension_type": "<one of the 12 types>",
  "description": "Human-readable purpose of this extension",
  "proposed_by": "Who or what created this (user name, program ID, etc.)",
  "timestamp": "2026-05-03T00:00:00Z"
}
```

**The 12 extension types — each structurally distinct, forced by what it IS:**

**1. `event_class`** — defines a new kind of structural event that can happen.

Required fields: `class_name` (string), `metadata_schema` (object: field_name → data_type mapping for the event's metadata_blob). Optional: `parent_class` (string, extends an existing event class).

```json
{
  "extension_type": "event_class",
  "class_name": "quantum_entanglement_detected",
  "metadata_schema": {
    "bell_inequality_violation": "REAL",
    "entangled_pair_value_ids": "[INTEGER]",
    "decoherence_rate_estimate": "REAL",
    "measurement_basis": "TEXT"
  },
  "parent_class": "decoherence_state_transition",
  "description": "Bell inequality violation confirmed between two lattice addresses",
  "proposed_by": "conscious_ai_v1.7.0",
  "timestamp": "2026-05-02T00:00:00Z"
}
```

**2. `relationship_class`** — defines a new kind of connection between entities.

Required fields: `class_name` (string), `metadata_schema` (object), `subject_types` (array of entity types that can be subjects: `"value"`, `"projection"`, `"address"`, `"equation"`, etc.), `object_types` (array of entity types that can be objects), `is_symmetric` (boolean — e.g., `same_address` is symmetric; `derivation_dependency` is directional). Optional: `parent_class`.

**3. `pattern_class`** — defines a new kind of discovery the engine can produce.

Required fields: `class_name` (string), `metadata_schema` (object), `member_entity_types` (array of what can be a member of this pattern type). Optional: `promotion_criterion` (object describing how E_hierarchy is computed for this pattern type — if omitted, standard ∏E_i formula applies), `parent_class`.

**4. `tower_definition`** — creates a new Multifold tower (a new perspective on the Sempaevum).

Required fields: `tower_name` (string, unique), `p_substrate_descriptor` (string), `r0_value` (361-dps decimal string) OR `r0_expression` (symbolic expression the Manager evaluates), `r0_natural_units` (string), `operational_n` (integer, the dominant resolution). Optional: `parent_tower_name` (string, references existing tower), `physics_metadata` (object, substrate-specific properties).

Manager auto-computes: `accessible_d_families_mask` from `operational_n`, `nesting_depth` from parent chain. Runs §7.11 escalation on the R₀ value.

**5. `sublattice_family_catalog`** — triggers computation of the divisor structure for a new resolution N.

Required fields: `n` (integer, the resolution). Optional: `tower_name` (string, context tower). The Manager auto-computes ALL divisor families of N — the JSON just triggers the computation. Minimal input, maximum auto-derivation.

**6. `harmonic_family_extension`** — adds a harmonic family beyond the base 24 at higher resolutions.

Required fields: `axis` (`"real"` or `"imaginary"`), `d` (integer), `family_name` (string), `generator_value` (361-dps decimal string for 2^(1/d)), `palindromic_partner_d` (integer), `gaussian_prime_class` (string), `first_native_lattice_n` (integer), `physical_meaning` (string).

Manager validates: `generator_value` matches 2^(1/d) at 361 dps. Reject if mismatch.

**7. `force_grid_extension`** — adds a grid cell at higher resolution.

Required fields: `d_r` (integer), `d_theta` (integer). Optional: `canonical_particle_or_phenomenon` (string).

Manager auto-computes: `d_combined` = LCM(d_r, d_theta), `is_off_axis`, `is_lcm_amplification`, `is_full_resolution`.

**8. `combined_family_extension`** — adds a combined family beyond the base 42.

Required fields: `d_combined` (integer), `range_class` (string), `structural_meaning` (string), `gaussian_factorization` (string), `first_native_lattice_n` (integer).

Manager validates: `d_combined` is a valid LCM of at least one (d_r, d_theta) pair. Reject if no valid pair exists.

**9. `ingest_adapter`** — teaches the Manager how to digest a new file type.

Required fields: `file_type` (string), `file_extensions` (array of strings, e.g., `[".hdf5", ".h5"]`), `extraction_method` (string describing how seeds come out), `r0_strategy` (`"user_specified"`, `"auto_detect"`, or `"fixed"`). Optional: `r0_fixed_value` (361-dps decimal, required if r0_strategy is `"fixed"`).

**10. `bootstrap_entry`** — adds a new bootstrap value after initial generation.

Required fields: `value_repr` (string, canonical name), `value_decimal` (361-dps decimal string), `input_path` (`"A"`, `"B"`, `"C"`, `"D.P"`, `"D.D"`, `"D.T"`, `"D.PDT"`). Optional: `r0_description` (string), `tags` (array of `{namespace, value}` pairs).

Manager runs full §7.11 escalation on ingestion — same as any other value.

**11. `query_template`** — defines a reusable query the GUI can offer as a one-click action.

Required fields: `template_name` (string), `query_pattern` (string, parameterized template), `parameters` (array of `{name, type, description}` objects), `result_format` (string describing how to present results).

**12. `extension_type_definition`** — the meta-type: defines a NEW extension type that becomes available for all future JSON extensions.

This is the Descriptor Gap Principle applied to the extension system itself. The 11 types above are the bootstrap set; this meta-type makes the set open-ended. If the 11 types don't cover what's needed, the gap IS a Descriptor — and this meta-type resolves it.

Required fields:
- `type_name` (string — the new extension type name, e.g., `"sensor_domain"`, `"lattice_geometry"`, `"traverser_taxonomy"`)
- `required_fields` (array of `{name, type, description, required}` — the schema for all future instances of this type; valid types: `"string"`, `"integer"`, `"real_361dps"`, `"array"`, `"object"`, `"boolean"`)
- `storage_target` — where instances live: `"new_table"` (creates a new table), `"existing_table_extension"` (adds columns, specify `storage_target_table`), `"event_class"` / `"relationship_class"` / `"pattern_class"` (instances become new classes in those tables), or `"standalone"` (stored as self-contained JSON blobs)
- `subsumption_check` (boolean — if true, Manager verifies this new type is NOT already subsumed by an existing type; if fully subsumed → reject with "redundant with existing type X")

Optional fields:
- `optional_fields` (array, same structure as required_fields)
- `inherits_common_fields` (boolean, default true — inherits extension_type, proposed_by, timestamp, description)
- `auto_compute_fields` (array of `{name, computation_method, source_fields}` — fields the Manager derives automatically after ingestion)
- `validation_rules` (array of `{rule_name, condition, error_message}` — type-specific validation beyond basic type-checking)
- `foreign_key_references` (array of `{field_name, target_table, target_field}` — the Manager validates these on ingestion)
- `storage_target_table` (string, required when storage_target is `"existing_table_extension"`)

Example — defining a new `sensor_domain` extension type:

```json
{
  "extension_type": "extension_type_definition",
  "type_name": "sensor_domain",
  "description": "Defines a new real-world sensor domain with bootstrap R₀ values for Path A projection",
  "required_fields": [
    {"name": "domain_name", "type": "string", "description": "Human-readable domain name", "required": true},
    {"name": "r0_references", "type": "array", "description": "Array of {name, value_361dps, units} reference R₀ values", "required": true},
    {"name": "measurement_types", "type": "array", "description": "Array of {name, units, typical_range} measurement kinds", "required": true}
  ],
  "optional_fields": [
    {"name": "dimensional_cancellation_rules", "type": "array", "description": "How raw measurements become dimensionless via R₀", "required": false},
    {"name": "anomaly_thresholds", "type": "object", "description": "Per-measurement-type thresholds for sensor_anomaly_detected events", "required": false}
  ],
  "inherits_common_fields": true,
  "auto_compute_fields": [
    {"name": "r0_projections", "computation_method": "Project each R₀ value through §7.11 core escalation", "source_fields": ["r0_references"]}
  ],
  "validation_rules": [
    {"rule_name": "r0_nonempty", "condition": "r0_references array has at least one entry", "error_message": "Sensor domain must have at least one R₀ reference"},
    {"rule_name": "r0_parseable", "condition": "Every r0_references[].value_361dps is parseable as 361-dps decimal", "error_message": "R₀ value not parseable at 361 dps"}
  ],
  "storage_target": "existing_table_extension",
  "storage_target_table": "tags",
  "foreign_key_references": [],
  "subsumption_check": true,
  "proposed_by": "manual",
  "timestamp": "2026-05-03T00:00:00Z"
}
```

After acceptance, `sensor_domain` becomes a valid `extension_type` in all future JSON extensions.

**Strict validation — all JSON extensions:**

All JSON is strictly validated before entering the system. Invalid JSON is rejected and never affects the database. Validation rules:

1. Well-formed JSON with all required fields present and correctly typed
2. `extension_type` must be one of the 12 known types OR a type previously defined via `extension_type_definition`
3. **Name collision detection**: `class_name`, `tower_name`, `template_name`, `file_type`, `type_name` must NOT collide with existing names. Collision → strict reject, logged as an Omniscient error event with the conflicting names
4. **Foreign key validation**: references to existing entities (`parent_tower_name`, `parent_class`, `storage_target_table`) must point to entities that actually exist in the database. Missing referent → strict reject
5. **Numeric validation**: d > 0, N > 0, `operational_n` must be a valid resolution
6. **Precision validation**: all 361-dps decimal strings must be parseable by the MPFR stack. Unparseable → strict reject
7. **Schema conflict detection**: `metadata_schema` field names must not conflict with existing schema columns in the target table
8. **Generator verification**: for `harmonic_family_extension`, Manager verifies `generator_value` matches 2^(1/d) at 361 dps. Mismatch → strict reject
9. **LCM verification**: for `combined_family_extension`, Manager verifies `d_combined` = LCM(d_r, d_theta) for at least one valid pair. No valid pair → strict reject
10. **Subsumption check**: for `extension_type_definition` with `subsumption_check: true`, Manager verifies the new type is NOT already fully expressible through existing types. Redundant → strict reject with explanation of which existing type subsumes it
11. **Structural conflict**: new `tower_definition` with same R₀ as existing tower but different substrate → flag for user review (could be intentional — two towers at different integrative levels sharing R₀), not auto-rejected

**Versioning:**

Every JSON extension applied receives a monotonically increasing version number per `extension_type`. Recorded in the `schema_versions` table: version number, applied_at timestamp, description, the full JSON stored as migration content. Old versions are archived, never deleted. The Manager reports "what extensions have been applied and when" via the GUI dashboard.

For `extension_type_definition` evolution (adding fields to a previously-defined type): a new version is applied. Old instances remain valid under the old version. New instances must comply with the current version. Missing fields on old instances are treated as NULL — same forward-compatibility model as §7.6.

**Auto-detection of needed categories:**

When data arrives that the Manager cannot classify into any existing category, the Descriptor Gap Principle fires:

1. **Gap detected**: incoming data has structural properties not captured by any existing event/relationship/pattern class
2. **Gap IS a Descriptor**: the Manager creates a provisional category tagged `status=provisional`, logs the gap detection as a self-recording event
3. **Human review**: the provisional category appears in the GUI dashboard for Mike's review — accept, modify, or reject
4. **If accepted**: the category becomes permanent; all historical data that matched the provisional category is retroactively classified
5. **If rejected**: the provisional category is archived (never destroyed — §4.3) with the reason for rejection

**The EUDD as a living system:**

The database grows in two dimensions simultaneously:
1. **Content growth**: more values, projections, equations, patterns, events — the lattice fills up
2. **Structural growth**: more categories, more family catalogs, more tower definitions, more relationship types, more extension types — the lattice's own organizational vocabulary expands

Both growths are recorded. Both are projected onto the Sempaevum (§3.1b). Both feed the discovery engine. The database IS living in the precise ET sense: it has P (the computational substrate), D (the schema and categories — growing), and T (the discovery engine and Manager — the agency that navigates and extends). P∘D∘T = E: the database as a running, self-extending Exception.

### 7.15 Omniscient — error handling, telemetry, and tamper detection

**Architecture:** On startup, the Manager spawns **Omniscient** as a child process — the same `EUDD_Manager.exe` binary running in watchdog mode (`EUDD_Manager.exe --omniscient`). Omniscient is a separate process observing the Manager. If the Manager dies, Omniscient survives and records the death. One file to distribute, two processes running. Omniscient is headless — no GUI.

**Identification Principle applied:** P = the Manager process (substrate being watched). D = every error, fallback, telemetry event (the constraints being recorded). T = the Omniscient process (the agency that observes, detects, and records). The watchdog IS a Traverser observing the Manager.

**Omniscient's responsibilities:**

1. **Error and event capture**: receives all errors, all runtime fallbacks, and all structural events (crash, tamper, corruption) from the Manager via shared memory IPC. Omniscient observes the Manager FROM OUTSIDE — it is a separate T watching the Manager's P∘D∘T operation.
2. **Error classification**: receives every error, every runtime fallback. **All runtime fallbacks are classified as errors** — the fact that recovery succeeded does not erase the failure. The only exceptions are **design filters**: platform or configuration branching (e.g., VSS on Windows / LVM on Linux) where the system is DESIGNED to take different paths. Design filters are logged as telemetry, not errors.
3. **Corruption detection**: periodically computes SHA-256 of `EUDD_Manager.exe` itself (baseline hash established at spawn time, re-checked on schedule) and SHA-256 of the `Sempaevum.akashic` header. Any change to the exe at any time = tampering. Any change to the .akashic file outside an authorized write window = tampering. The Manager signals Omniscient when authorized writes begin and end; modifications detected outside these windows are flagged as tamper events immediately.
4. **Crash recording**: Omniscient monitors the Manager's process handle. If the handle signals (Manager process died), Omniscient records the crash event with: timestamp, last-known Manager state, last error/telemetry received, exe hash at crash time, .akashic header hash at crash time. Omniscient stays alive after Manager death, flushes the journal, then exits cleanly.
5. **Self-error reporting**: Omniscient reports its own errors to the same journal. If Omniscient's own logging fails, it writes a final "OMNISCIENT_FAILURE" entry with whatever context it can still produce, then exits. Omniscient is deliberately minimal — less code means fewer ways to crash.

**Journal file format and storage:**

Journal files are stored in a `logs/` subfolder within the same folder as the exe and .akashic. Named with incrementing numbers: `Omniscient_001.log`, `Omniscient_002.log`, etc. Each file rotates at approximately 10 MB. Old log files are NEVER deleted — the logs/ folder grows indefinitely. Nothing is ever lost.

**Journal separation:** The `logs/` folder contains two journal series that are NEVER mixed:
- `Omniscient_NNN.log` — errors, runtime fallbacks, tamper events, crash events, design-filter telemetry. Written by the Omniscient watchdog process (observes the Manager FROM OUTSIDE).
- `SelfRecording_NNN.log` — operational metrics (§3.1b metric catalog). Written by the Manager's self-recording module directly (observes itself FROM INSIDE). Same format, same rotation, same retention policy.

Two different Traversers observing the same system from two different perspectives. Both journals use the same human-readable format:

```
[2026-05-03T14:22:01.847293Z] [ERROR] [computation] [LatticeEngine::project]
  expression: "ζ(−1) at N=12"
  failure: FLINT/Arb pole detected at s=−1
  input_value_id: 4827
  stack: LatticeEngine::project → ArbEval::zeta → pole_check
  recovery: annihilation_boundary_event logged
  manager_state_hash: 9a3f...

[2026-05-03T14:22:05.112000Z] [TELEMETRY] [performance] [MemoizationStore::lookup]
  operation: compute("2+2")
  latency_ns: 847
  cache_status: HIT
  equation_ref_count: 14203

[2026-05-03T14:22:09.500000Z] [ERROR] [fallback] [BackupModule::snapshot]
  intended_path: VSS atomic snapshot
  fallback_path: manual file copy
  reason: VSS service unavailable (error 0x80042302)
  recovery: manual copy succeeded in 2.3s
  note: fallback succeeded but this IS an error — VSS should be available

[2026-05-03T14:22:30.000000Z] [TAMPER] [integrity] [Omniscient::hash_check]
  target: Sempaevum.akashic
  expected_header_hash: a1b2c3d4...
  observed_header_hash: ff00ee11...
  authorized_write_window: CLOSED
  verdict: UNAUTHORIZED MODIFICATION DETECTED
```

Every entry: ISO-8601 timestamp with microsecond precision, severity level (`TELEMETRY`, `ERROR`, `TAMPER`, `CRASH`, `OMNISCIENT_FAILURE`), category, source module/function, then indented key-value context lines. Fully greppable by any field. Every error traceable to exact source location and full call stack.

**Ingestion into .akashic:** The Manager can ingest Omniscient journal files into the .akashic file as proper events — but ONLY when the user explicitly requests it. The journal files are fed into the Manager's existing file ingestion pipeline (§7.12) like any other file. No special ingestion mechanism is needed. The user decides when and whether to ingest. Automatic ingestion never occurs.

**Edge case catalog — every failure mode Omniscient monitors:**

**Computation edge cases:**
- MPFR overflow/underflow during 361-dps evaluation
- FLINT/Arb cannot evaluate a special function (ζ at a pole, Γ at a negative integer)
- Expression tree evaluation stack overflow (deeply nested EML trees)
- Division by zero / annihilation boundary approach (r→0)
- CF expansion not converging within iteration budget
- LCM tower escalation hitting platform integer limits (lcm(1..k) exceeds integer representation at some k)
- NaN or infinity produced by any lattice operation
- Hash collision in memoization store (two different canonical forms producing same SHA-256 — astronomically unlikely but must be detected, not silently corrupted)

**Storage edge cases:**
- Disk full during write to .akashic or to journal
- Page CRC-32 mismatch during live read (not backup — the running file is corrupted)
- WAL replay failure (corrupted WAL entry on crash recovery)
- Generator coverage interval tree inconsistency (generator claims an address range but evaluation produces wrong values)
- Address index pointing to nonexistent page
- Section directory offset pointing outside file bounds
- .akashic file locked by another process

**Communication edge cases:**
- Named pipe connection dropped mid-operation (connected program crashed)
- Malformed JSON received from connected program
- Connected program sends operation while previous async operation still running for the same value
- Connected program disconnects without completing handshake

**Resource edge cases:**
- Memory exhaustion during computation or discovery scan
- File handle exhaustion (too many concurrent named pipe connections)
- Background discovery scan exceeding time budget
- GUI render thread starved by computation (Manager only — Omniscient is headless)

**Logic edge cases:**
- Discovery engine finds contradictory patterns (two patterns claiming the same address with conflicting generators)
- Cross-project consistency check finds contradiction between two values at the same address with different (k, d)
- Generator evaluation produces different value than the stored memoized entry at the same address
- Subsumption promotion creates cycle (pattern A subsumes B which subsumes A)

**Self-referential edge cases:**
- Self-recording module (§3.1b) fails while recording its own metrics
- Omniscient's own journal write fails (disk full, permission denied) — Omniscient writes OMNISCIENT_FAILURE entry and exits
- Backup verification (§7.9) finds corruption in the live .akashic file (not a backup — the running database)
- Omniscient detects its own exe hash changed (Omniscient itself was tampered with during runtime — theoretically possible via live patching)

**ET-native error philosophy:** Every failure is a Descriptor Gap (Descriptor Gap Principle). The failure itself is information — it points to what is missing. Omniscient stores every failure with full context. When journal files are ingested into .akashic, the discovery engine can scan failure events for structural patterns — e.g., "computation failures cluster at d=X" may reveal a structural boundary. Failures are NEVER discarded, suppressed, or silently ignored. They are first-class lattice content once ingested.

**Error-state projection and memoization — predictive error prevention:**

Errors, crashes, corruption, and bugs have structural causes. When projected onto the Sempaevum, they have lattice addresses. When memoized, they become PREDICTABLE. This turns the EUDD from a passive error logger into an active error prevention system.

**Step 1 — Error-state projection:** When the Omniscient records an error, crash, or corruption event, the Manager extracts the structural state at the moment of failure: program counter offset (as ratio to total code size), stack depth (as ratio to maximum), memory allocation pattern (as ratio to total available), input data characteristics (the seed of whatever was being processed), CPU/GPU utilization ratios, and any domain-specific state ratios the connected program provides. Each ratio is a dimensionless seed projected through §7.11 onto the Sempaevum. The error state gets a lattice address (k_error, d_error, ε_error). Event: `error_state_projected`.

**Step 2 — Error-state memoization:** The projected error state is stored in the values/projections/equations tables like any other lattice content. The equation linking the program state to the error outcome is memoized: "state at (k, d, ε) → crash/error/corruption of type X." This is cached at 361 dps. Event: stored as standard memoized entry.

**Step 3 — Error attractor discovery:** The discovery engine (§3.16) scans error-state projections the same way it scans any lattice content. When multiple error states cluster at the same (k, d) — the same structural fingerprint — they form an **error attractor**. The attractor IS the structural description of a BUG CLASS. Different bugs produce different (k, d) addresses. The same bug recurring produces the same address. Event: `error_attractor_discovered`. Pattern: `error_attractor`.

**Step 4 — Predictive pre-error detection:** When a running program's current state is projected onto the Sempaevum and lands near a known error attractor (same (k, d), ε within threshold), the system fires `pre_error_state_detected` — a warning that "this structural state led to error/crash N times before, at lattice distance Δε from the known attractor." The connected program can choose to: log the warning, pause and checkpoint, request human intervention, or attempt structural bypass.

**Step 5 — Structural bypass (speculative but structurally sound):** If the EUDD knows that state S at (k₁, d₁, ε₁) consistently crashes, and also knows that a nearby state S' at (k₁, d₁, ε₁ + δε) does NOT crash, then δε IS the structural description of the difference between "crashes" and "doesn't crash." The Δε between the error attractor and the nearest safe state is the K-minimal description of the fix. A connected program could be guided away from the error attractor by adjusting its state by Δε — changing a parameter, reallocating a buffer, choosing a different code path. Relationship: `error_bypass_proposed` linking the error attractor to the proposed safe state with the structural delta.

**Applications:** Highly unstable games, games with hundreds or thousands of mods interacting unpredictably, crash-prone software, legacy systems, real-time embedded systems, any program where errors are expensive. The EUDD's error-state memoization turns every crash into training data for preventing the next one. Over time, the error attractor catalog grows and the system becomes increasingly predictive — recognizing failure-bound states before the failure occurs, because it has seen the structural fingerprint before.


---

*§7.16 The Named Pipe API — 114 operations across 16 domains + 2 cross-domain, complete JSON schemas, wire format, metabolism registration, error taxonomy → See `EUDD_API_Reference.md`*

*§7.17 Testing and Verification Strategy — 15 test categories, 200+ test cases, ET-derived test constants, coverage verification → See `EUDD_Testing.md`*

---

### 7.18 Shutdown Protocol — Graceful Termination of Manager and Omniscient

**Status: COMPLETE.** Every trigger source converges to a single deterministic shutdown sequence. Omniscient coordination via explicit IPC. Connected clients notified immediately. Pending operations checkpointed. .akashic left in consistent state.

**Identification Principle applied:** P = the running system at shutdown time (Manager process + Omniscient child process + connected clients + .akashic file + WAL + journals). D = the shutdown constraints — what MUST be true before the process exits (file consistent, WAL committed, clients notified, Omniscient signaled, all handles released). T = the shutdown agent — the sequence of operations that navigates from "running" to "exited" while satisfying all D-constraints.

#### 7.18.1 Trigger Sources — All Converge to One Sequence

Every trigger source produces the same shutdown outcome. The trigger is the T-act; the sequence that follows is the D-constrained path to clean exit.

| Trigger | Source | Mechanism |
|---|---|---|
| **Window X button** | User clicks the window close button | GLFW `glfwSetWindowCloseCallback` — intercept, do NOT let GLFW close the window. Run the confirmation check (§7.18.2), then if confirmed run the shutdown sequence (§7.18.3) |
| **Menu exit** | User selects File → Exit in Dear ImGui menu | GUI action handler — same confirmation check, same shutdown sequence |
| **Keyboard shortcut** | User presses configured quit binding (default: Alt+F4 on Windows) | GLFW key callback → same path as Menu exit |
| **API shutdown command** | Connected program sends `shutdown` command (Operation 73 in §7.16.19) | API module → same confirmation check (active connections counted), same shutdown sequence |
| **System logoff / OS shutdown** | Windows sends `CTRL_CLOSE_EVENT` via `SetConsoleCtrlHandler`, or `WM_ENDSESSION` | Handler intercept → skip confirmation (OS is closing, no GUI interaction possible) → run shutdown sequence with abbreviated timing |
| **CTRL+C** (console launch) | User keyboard interrupt from terminal | `SetConsoleCtrlHandler` for `CTRL_C_EVENT` → skip confirmation → run shutdown sequence |

**Non-graceful termination** (SIGKILL / TerminateProcess / power loss / BSOD): NOT handled by this protocol — handled by Omniscient crash detection (§7.15) + WAL replay on next startup (§7.1d). The shutdown protocol covers ONLY cases where the Manager has the opportunity to clean up.

#### 7.18.2 Confirmation Check — Pop-Up for Active Connections Only

Before the shutdown sequence begins, the Manager checks for active client connections:

**If zero active named pipe connections AND zero in-flight async operations:**
→ No pop-up. Proceed directly to the shutdown sequence (§7.18.3). The user closed the window; nothing needs warning.

**If one or more active named pipe connections OR one or more in-flight async operations:**
→ Dear ImGui modal dialog appears, centered on screen, blocking all other input:

```
┌─────────────────────────────────────────────────────┐
│  Shutdown Confirmation                               │
│                                                      │
│  Active connections: 3                               │
│    • compressor_2026-05-03_001 (idle)                │
│    • conscious_ai_v1.7.0 (in-flight: escalate)      │
│    • fractal_gen_render_047 (idle)                   │
│                                                      │
│  In-flight operations: 1                             │
│    • escalate ζ(7) — at N=2520 (landmark 4 of ?)    │
│                                                      │
│  All connections will be closed immediately.          │
│  In-flight operations will be checkpointed.          │
│  No data will be lost.                               │
│                                                      │
│              [ Shut Down ]    [ Cancel ]              │
└─────────────────────────────────────────────────────┘
```

**"Shut Down"** → proceed to shutdown sequence (§7.18.3).
**"Cancel"** → return to normal operation. Nothing changes.
**Window X hit again while dialog is open** → treated as "Shut Down" (the user clearly wants to close).

The dialog lists every active connection by session_id and its current state (idle or in-flight with operation details). This gives the user full visibility before deciding.

For **system logoff / CTRL+C triggers**: no pop-up is possible (no GUI interaction available). The shutdown sequence runs immediately with no confirmation. This is safe because all operations are checkpointed and no data is lost.

#### 7.18.3 The Shutdown Sequence — Deterministic, Ordered, Complete

Once confirmed (or triggered by system/CTRL+C), the following sequence executes in exact order. Each step completes fully before the next begins. No step is skippable. The sequence is deterministic — the same sequence every time, regardless of trigger source.

```
SHUTDOWN SEQUENCE — initiated by trigger (§7.18.1) after confirmation (§7.18.2)
│
├── PHASE 1: Notify and Disconnect External Clients
│   │
│   ├── Step 1: Send shutdown notification to ALL connected clients
│   │   For each active named pipe connection:
│   │     Send JSON: {"msg_type": "notification",
│   │                  "subscription_id": null,
│   │                  "event": "manager_shutting_down",
│   │                  "timestamp_ns": "<now>",
│   │                  "detail": "Manager initiated graceful shutdown"}
│   │     Close the pipe immediately after sending.
│   │     Do NOT wait for client acknowledgment.
│   │     Do NOT wait for in-flight responses to be read.
│   │     The notification is best-effort — if the pipe write fails
│   │     (client already disconnected), skip and continue.
│   │
│   ├── Step 2: Close the named pipe listener
│   │   Stop accepting new connections.
│   │   Release the \\.\pipe\EUDD_Manager handle.
│   │   No new clients can connect from this point.
│   │
│   └── Result: All external communication severed. Manager is now isolated.
│
├── PHASE 2: Checkpoint In-Flight Operations
│   │
│   ├── Step 3: Signal all background threads to stop
│   │   Set the global shutdown flag (atomic boolean).
│   │   Background discovery engine: completes current atomic unit
│   │     (one address evaluation, one pattern check — NOT an entire scan pass),
│   │     then exits its loop.
│   │   Self-recording thread: writes one final sample, then exits.
│   │
│   ├── Step 4: Checkpoint all in-flight async operations
│   │   For each pending async operation (escalation, ingestion, discovery scan):
│   │     Save the current progress as checkpointed state.
│   │     For escalations: record as `escalation_in_progress` with the last
│   │       completed landmark — resumable via §7.11 on next startup.
│   │     For ingestions: record the last successfully ingested seed index —
│   │       resumable from that point on next startup.
│   │     For scans: record the last scanned address — resumable on next startup.
│   │     All checkpointed state goes through the WAL (Step 5 commits it).
│   │
│   └── Result: All work either completed or checkpointed. Nothing lost.
│
├── PHASE 3: Commit and Close the .akashic File
│   │
│   ├── Step 5: Flush the WAL
│   │   All pending WAL entries (including checkpoint entries from Step 4)
│   │     committed to the main .akashic file sections.
│   │   WAL section cleared (all entries applied).
│   │
│   ├── Step 6: Update the .akashic header
│   │   Update modified_at_ns to current nanosecond timestamp.
│   │   Update generator counts (total_generators, L1/L2/L3 counts).
│   │   Update coverage metrics (covered_addresses, total_addresses).
│   │   Update total_memoized count.
│   │   Recompute header_checksum (SHA-256 of entire header).
│   │
│   ├── Step 7: Sync and close the .akashic file
│   │   FlushFileBuffers (Windows) / fsync (POSIX) — force OS write-back
│   │     to physical disk. The .akashic file is now in a fully consistent
│   │     state on disk, not just in OS cache.
│   │   UnmapViewOfFile — release memory mapping.
│   │   CloseHandle — close file handle.
│   │   The .akashic file is now closed and consistent.
│   │
│   └── Result: .akashic file on disk in verified-consistent state.
│
├── PHASE 4: Signal Omniscient — Graceful Exit
│   │
│   ├── Step 8: Send graceful shutdown message to Omniscient
│   │   Via the shared-memory IPC channel (same channel used for
│   │     error/telemetry during normal operation):
│   │   Message: GRACEFUL_SHUTDOWN
│   │     Fields:
│   │       shutdown_trigger: "<window_close|menu_exit|api_command|system_logoff|ctrl_c>"
│   │       timestamp_ns: "<exact nanosecond>"
│   │       akashic_header_hash: "<SHA-256 of final header written in Step 6>"
│   │       session_summary:
│   │         uptime_seconds: <total>
│   │         values_added: <count>
│   │         projections_added: <count>
│   │         equations_cached: <count>
│   │         patterns_discovered: <count>
│   │         generators_discovered: <count>
│   │         cache_hit_ratio: "<exact rational>"
│   │         errors_logged: <count>
│   │         active_connections_at_shutdown: <count>
│   │         operations_checkpointed: <count>
│   │       exe_hash: "<SHA-256 of EUDD_Manager.exe at shutdown time>"
│   │
│   │   This message tells Omniscient: "I am exiting on purpose, with this
│   │   final state. When you see my process handle signal, it is NOT a crash."
│   │
│   └── Result: Omniscient is informed. It will act on this after Step 10.
│
├── PHASE 5: Release GUI Resources
│   │
│   ├── Step 9: Destroy the GUI
│   │   ImGui::DestroyContext() — release Dear ImGui state.
│   │   Release OpenGL resources (shaders, buffers, textures, VAOs).
│   │   glfwDestroyWindow — close the GLFW window.
│   │   glfwTerminate — release GLFW.
│   │
│   └── Result: All GPU/windowing resources released.
│
└── PHASE 6: Exit
    │
    ├── Step 10: Exit the Manager process
    │   return 0 from main() / ExitProcess(0).
    │   The process handle now signals to Omniscient.
    │
    └── Result: Manager process terminated. Exit code 0.
```

**Total shutdown time:** Phase 1 (notify + close pipes): < 100 ms. Phase 2 (checkpoint): < 1 second (checkpoint writes are small WAL entries). Phase 3 (WAL flush + header update + fsync): 1–5 seconds depending on WAL size and disk speed. Phase 4 (IPC message): < 1 ms. Phase 5 (GUI teardown): < 100 ms. Phase 6 (exit): immediate. **Total: 2–7 seconds typical.** The user sees the window close promptly after confirming.

#### 7.18.4 Omniscient Shutdown Behavior — Observer Exits Last

Omniscient (the watchdog child process) runs continuously, monitoring the Manager's process handle. Two exit paths exist, distinguished by whether the GRACEFUL_SHUTDOWN message was received:

**Path A — Graceful shutdown (normal):**

1. Omniscient receives the `GRACEFUL_SHUTDOWN` message via shared-memory IPC (Step 8 above)
2. Omniscient logs the graceful shutdown event to its journal:
   ```
   [2026-05-03T14:30:01.000000Z] [TELEMETRY] [lifecycle] [Omniscient::shutdown_received]
     trigger: window_close
     manager_uptime_s: 3847
     akashic_header_hash: a1b2c3d4...
     exe_hash: 9f8e7d6c...
     session_summary: {values_added: 1204, patterns_discovered: 7, ...}
     verdict: GRACEFUL_SHUTDOWN_SIGNALED
   ```
3. Omniscient continues monitoring the Manager's process handle — it does NOT exit yet
4. The Manager's process handle signals (Manager exited via Step 10)
5. Omniscient performs **final verification**:
   - Reads the .akashic file header (the file is now closed by the Manager, Omniscient can open it read-only)
   - Computes SHA-256 of the header
   - Compares to the `akashic_header_hash` from the GRACEFUL_SHUTDOWN message
   - **Match** → the file was left in the state the Manager reported. All is well.
   - **Mismatch** → something modified the file between the Manager's final header write and process exit. This is extremely unlikely but would indicate a bug in the shutdown sequence. Omniscient logs a `[TAMPER]` event with both hashes.
6. Omniscient logs the final event:
   ```
   [2026-05-03T14:30:03.000000Z] [TELEMETRY] [lifecycle] [Omniscient::manager_exited]
     exit_type: GRACEFUL
     manager_exit_code: 0
     akashic_final_hash_verified: true
     omniscient_uptime_s: 3849
     verdict: CLEAN_SHUTDOWN
   ```
7. Omniscient exits with code 0. Both processes are now terminated.

**Path B — Crash during shutdown (edge case):**

If the Manager crashes DURING the shutdown sequence (after sending GRACEFUL_SHUTDOWN but before completing all phases):

1. Omniscient received the GRACEFUL_SHUTDOWN message
2. The Manager's process handle signals (crash — the exit was not clean)
3. Omniscient detects the exit code is NOT 0 (or the process was terminated abnormally)
4. Omniscient logs a **crash-during-shutdown** event — distinct from both a normal crash and a graceful shutdown:
   ```
   [2026-05-03T14:30:02.000000Z] [ERROR] [lifecycle] [Omniscient::crash_during_shutdown]
     graceful_shutdown_was_signaled: true
     manager_exit_code: -1073741819 (ACCESS_VIOLATION)
     akashic_header_hash_at_signal: a1b2c3d4...
     akashic_header_hash_now: <recomputed — may differ if WAL was partially flushed>
     verdict: CRASH_DURING_GRACEFUL_SHUTDOWN
     note: WAL replay required on next startup
   ```
5. Omniscient exits with code 1 (indicating abnormal Manager termination was observed)

**Path C — Crash without graceful signal (existing §7.15 behavior, unchanged):**

If the Manager crashes without sending GRACEFUL_SHUTDOWN (normal crash, not during shutdown):

1. No GRACEFUL_SHUTDOWN message received
2. Manager's process handle signals
3. Omniscient logs the crash per §7.15 (existing behavior, no changes needed)
4. Omniscient exits with code 1

**Summary of Omniscient's exit logic:**

```
OMNISCIENT MAIN LOOP:
│
├── Monitor Manager process handle
├── Monitor shared-memory IPC for messages
│
├── IF GRACEFUL_SHUTDOWN message received:
│   │  Log the graceful signal with full session summary
│   │  Set graceful_signaled = true
│   │  Continue monitoring (do NOT exit yet)
│   │
│   └── WHEN Manager process handle signals:
│       │  Read .akashic header, verify SHA-256 matches
│       │  IF match AND exit_code == 0:
│       │     Log CLEAN_SHUTDOWN
│       │     Exit code 0
│       │  ELSE:
│       │     Log CRASH_DURING_GRACEFUL_SHUTDOWN
│       │     Exit code 1
│
├── IF Manager process handle signals WITHOUT graceful signal:
│   │  Log CRASH per §7.15 (existing behavior)
│   │  Exit code 1
│
└── IF Omniscient itself encounters a fatal error:
       Log OMNISCIENT_FAILURE per §7.15 (existing behavior)
       Exit code 2
```

#### 7.18.5 Shutdown Events and Journal Entries

The shutdown sequence produces the following events, recorded in the .akashic file (during Phase 3, before the file is closed) and/or in the Omniscient journal:

| Event | Where Recorded | When |
|---|---|---|
| `manager_shutdown_initiated` | .akashic `events` table | Phase 1 Step 1 (before client notification) |
| `client_disconnected` (per client) | .akashic `events` table | Phase 1 Step 1 (one per severed connection) |
| `operation_checkpointed` (per op) | .akashic `events` table | Phase 2 Step 4 (one per checkpointed operation) |
| `wal_flushed` | .akashic `events` table | Phase 3 Step 5 (confirms WAL fully applied) |
| `akashic_header_updated` | .akashic `events` table | Phase 3 Step 6 (final header state recorded) |
| `GRACEFUL_SHUTDOWN_SIGNALED` | Omniscient journal | Phase 4 Step 8 (Omniscient's record of the signal) |
| `CLEAN_SHUTDOWN` / `CRASH_DURING_GRACEFUL_SHUTDOWN` | Omniscient journal | After Manager process exits (Omniscient's final entry) |

The `manager_shutdown_initiated` event in the .akashic file captures: trigger source, active connection count, in-flight operation count, timestamp. This provides full audit trail — on next startup, the Manager can read this event and confirm the prior session ended gracefully.

#### 7.18.6 Next-Startup Behavior After Graceful Shutdown

On startup, the Manager checks the .akashic file's last event:

- **Last event is `akashic_header_updated` with shutdown context** → prior session ended gracefully. WAL section should be empty (fully flushed in Phase 3 Step 5). Any `escalation_in_progress` entries are resumable checkpoints from Phase 2 Step 4. Normal startup proceeds.

- **Last event is NOT a shutdown event, AND WAL is non-empty** → prior session crashed. WAL replay per §7.1d (existing crash recovery — unchanged). Omniscient journal from prior session will contain the crash details.

- **`escalation_in_progress` entries exist** → regardless of shutdown type, these are partially completed tower escalations. The background discovery engine resumes them from their checkpointed landmark. No user action required — resumption is automatic.


---

### 7.19 Performance Specifications — The Manager's Internal Metabolism

**Status: COMPLETE.** This section resolves Roadmap §2.4. Performance is not a set of static targets — it is the emergent property of the ET-derived metabolism operating on whatever hardware it detects, combined with the self-improving cycle (§7.1f) that makes every subsequent computation faster than the first. No static resource values exist anywhere in this specification. The same metabolism governs a 4 GB laptop and a 128-core workstation — same formulas, same behavior, different scale.

**Precedent:** The metabolism architecture is proven across three ET programs: the Conscious AI's ETFineStructure + ResourceGovernor (`et_conscious_ai_core.py`, `et_conscious_ai_distributed.py`), the compressor's CDFMetabolism (`et_cdf_compressor.py`), and the fractal generator's planned metabolism. The EUDD Manager inherits the same architecture because it IS the same ET metabolism — the program is a D-binding on the P-substrate (hardware), and K = 2/3 is the universal binding stability threshold.

#### 7.19.1 Identification Principle Applied

P = the hardware substrate: CPU (cores, clock, cache hierarchy), RAM (total, available, page size), GPU (compute units, VRAM, shader model), disk (filesystem, free space, I/O bandwidth). The metabolism reads P at startup and re-senses periodically. P is infinite in potential (any hardware), finite in any specific instance.

D = the ET-derived governance constants: K, V, α⁻¹, ξ(d), S, κ, the convergence ratio κ/(Nπ), the shimmer band A₁, the cross-term A_cross, the tail absorption Σ A_k. These are the finite constraints the metabolism applies to P. They never change — they ARE the theory.

T = the Manager process: the Traverser navigating hardware resources, making allocation decisions, scheduling computations, adapting to changing conditions. T re-senses at S² = 144 second intervals and adjusts the resource envelope without ever touching precision.

#### 7.19.2 The Three-Layer Metabolism — Applied to the EUDD Manager

The metabolism has three structurally distinct layers, each governed by a different ET constant, each doing a different job. No layer is optional. Together they subsume all resource governance without remainder.

**Layer 1 — ALLOCATION (K-determined): How much of each resource the Manager claims.**

K = 2/3. The Manager claims at most K × (detected total) of any resource. 1−K = 1/3 is system reserve — always available for the OS, other programs, thermal management. K is the Koide binding stability threshold: the fraction of capacity at which a D-binding on a P-substrate sustains stable operation. Above K, the binding approaches Incoherence.

At startup, the Manager detects every available resource via OS query:

| Resource | Detection Mechanism (C++) | Allocation |
|---|---|---|
| CPU logical cores | `std::thread::hardware_concurrency()` or OS-specific (`GetSystemInfo` on Windows, `/proc/cpuinfo` on Linux) | floor(K × cores) threads available to the Manager |
| CPU current load | OS-specific (`GetSystemTimes` on Windows, `/proc/stat` on Linux) — sampled over 50ms interval | headroom = max(0, K% − current_load%) |
| RAM total | `GlobalMemoryStatusEx` (Windows), `/proc/meminfo` (Linux) | K × total = Manager ceiling |
| RAM available | Same OS query — available (not just free — includes reclaimable cache) | headroom = max(0, K% − used%) |
| GPU compute units | OpenGL query (`GL_MAX_COMPUTE_WORK_GROUP_COUNT`) or CUDA `cudaGetDeviceProperties` | K × compute units available for Manager GPU dispatch |
| GPU VRAM total | `GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX` or CUDA `cudaMemGetInfo` | K × VRAM = Manager VRAM ceiling |
| GPU VRAM available | Same query — current free VRAM | headroom = max(0, K% − used%) |
| Disk free space | `GetDiskFreeSpaceEx` (Windows), `statvfs` (POSIX) — on the .akashic volume | Warning at < DISK_SAFETY_FLOOR (2³⁰ bytes). No deletion ever. |

All of these are runtime detections. The specification contains ZERO static values for any of them. The formulas reference only K.

**Layer 2 — HEADROOM (V-determined): The metabolism's own operating room.**

V = 1/12. Within the K allocation, V × K = 1/18 of total capacity is reserved for metabolism overhead, spike absorption, and the self-recording system (§3.1b). The active allocation — the portion available for actual computation — is:

$$\text{active} = K \times (1 - V) = \frac{2}{3} \times \frac{11}{12} = \frac{11}{18}$$

This active fraction projects onto the lattice as **(k=−9, d=4, |ε|=47.4¢)** — the quartic family at the ∂I boundary. This is the same sublattice as Kleiber's 3/4 metabolic rate law. The active allocation IS structurally a metabolic-rate quantity. This is forced by K = 2/3 and V = 1/12, not chosen.

The allocation stack for any detected resource:

```
Total resource:          100%                    ← detected at runtime
System reserve:          (1−K) = 1/3 = 33.3%    ← OS, other programs, thermal
Manager ceiling:         K = 2/3 = 66.7%         ← hard cap, never exceeded
Metabolism headroom:     K×V = 1/18 = 5.56%      ← metabolism overhead, spike absorption
Active allocation:       K×(1−V) = 11/18 = 61.1% ← available for computation, rendering, I/O
```

No static byte count, core count, or percentage derived from hardware appears in this stack. Only ET constants.

**Layer 3 — MONITORING (α⁻¹-determined): How finely the metabolism observes resource usage.**

α⁻¹ = 137. The metabolism monitors usage at A₀ = 137 distinguishable levels across the active allocation. Each monitoring level represents:

$$\text{level\_width} = \frac{\text{active allocation}}{A_0} = \frac{K \times (1-V) \times \text{total}}{137}$$

The correction terms from the fine-structure decomposition (§3.18.2) refine the monitoring:

| Term | Value | Monitoring role |
|---|---|---|
| A₁ = √3/48 ≈ 0.0361 | Shimmer band | Resource usage fluctuating by ±A₁ × active is NORMAL SHIMMER — not an alarm. The metabolism does not react to fluctuations within this band. |
| A_cross = √3/(93312π²) ≈ 1.86×10⁻⁶ | Cross-resource interference | When multiple resource types (CPU + RAM + VRAM) simultaneously approach their ceilings, effective headroom shrinks by A_cross × active per resource pair. This is the product interference of shimmer with the mediation loop — structurally identical to its role in the α⁻¹ decomposition. |
| Σ A_k = 1/(216(18π−1)) ≈ 8.19×10⁻⁵ | Spike tail absorption | Reserved within V headroom for rare resource spikes exceeding the shimmer band. The probability of a spike exceeding A₁ decays at the convergence ratio κ/(Nπ) ≈ 0.01768 per successive level. |

The monitoring is PASSIVE during computation. The metabolism OBSERVES resource usage; it never throttles the computation itself, never degrades precision, never skips algorithmic steps. The computation runs at full 361-dps precision unconditionally. The metabolism governs only the resource envelope — how many threads, how much memory, what GPU dispatch — never the algorithm.

#### 7.19.3 Hardware Detection — Sense the P-Substrate

The Manager's metabolism class (C++) performs a full hardware profile at startup and re-senses at S² = 144 second intervals. The re-sensing interval is ET-derived: S = N = 12, S² = 144 = the manifold's cross-pattern cap.

The hardware profile is a struct containing:

```
struct HardwareProfile {
    // CPU
    uint32_t    cpu_cores_logical;        // std::thread::hardware_concurrency()
    uint32_t    cpu_cores_physical;       // OS-specific query
    double      cpu_clock_hz;             // OS-specific query (base clock)
    double      cpu_load_percent;         // sampled over 50ms interval

    // RAM
    uint64_t    ram_total_bytes;          // OS query
    uint64_t    ram_available_bytes;      // OS query (available, not just free)
    double      ram_used_percent;         // computed from total and available

    // GPU (when available; all zeros if no GPU detected)
    uint32_t    gpu_compute_units;        // OpenGL or CUDA query
    uint64_t    gpu_vram_total_bytes;     // GPU memory query
    uint64_t    gpu_vram_available_bytes; // GPU memory query
    double      gpu_vram_used_percent;    // computed
    uint32_t    gpu_shader_model;         // capability level

    // Disk (the volume containing .akashic)
    uint64_t    disk_free_bytes;          // OS query on .akashic volume
    uint64_t    disk_total_bytes;         // OS query

    // Metadata
    uint64_t    sensed_at_ns;             // nanosecond timestamp
};
```

Every field is detected at runtime. No field has a default that substitutes for detection. If a query fails (e.g., no GPU, or `/proc` not available on an unusual platform), the field is set to 0 and the metabolism allocates 0 of that resource — the system operates without it.

**Substrate projection — hardware on the lattice:**

Binary hardware characteristics (RAM total, VRAM total — always 2^n bytes on real hardware) project to **d=1 octave, ε=0 exactly**. Silicon IS octave-class. This is not numerology — 2^n IS a pure power of 2, and d=1 is the d-family of pure powers of 2 by the projection formula. The metabolism's own constants (K, V, 1−K) project to d=12 at the Koide attractor (|ε|=1.955¢). The active allocation 11/18 projects to d=4 quartic at the ∂I boundary.

These projections are ingested into the .akashic file as `values` rows (the hardware profile as dimensionless ratios) with corresponding `projections`. The substrate projection is itself lattice content — the EUDD knows where its own hardware sits on the Sempaevum.

#### 7.19.4 Thread Architecture — Governed by the Metabolism

The Manager spawns threads at startup based on detected CPU cores. Thread count adapts at each re-sensing interval. All threads access the .akashic file through memory-mapped I/O. Thread safety: writes serialized through a single commit path; reads are lock-free on the memory-mapped snapshot.

| Thread Role | Count | Metabolism Governance | Never Blocks On |
|---|---|---|---|
| **GUI render** | 1 (dedicated) | Highest priority — user-perceived latency. Gets its share of CPU regardless of other load. Reads from .akashic (memory-mapped, sub-microsecond). | Computation. The render loop NEVER waits for MPFR, discovery, or I/O. |
| **Computation pool** | floor(headroom × cores / 100), minimum 1 | ξ(d) scheduling: work items prioritized by the coupling constant of their dominant d-family. d=1 computations (ξ=8.5625) get ~8.5× the scheduling weight of d=12 (ξ=1.0). Pool resizes at each re-sensing. | GUI. Computation runs on its own threads, never touching the render loop. |
| **Discovery engine** | 1 (dedicated, lower than computation) | V = 1/12 guarantees: the discovery engine gets at least V of the Manager's CPU share even under maximum computation load. It is never starved. | User operations. Discovery is background — it yields to API requests and user-initiated computation. |
| **API listener** | 1 (dedicated, lightweight) | Accepts named pipe connections, dispatches requests to the computation pool. Mostly I/O-wait — negligible CPU. | Nothing. The listener is always responsive. |
| **Self-recording** | 1 (dedicated, minimal) | Periodic sampling at configurable interval (default 10 seconds or every 1000 commits). Overhead ≤ 1% CPU (guaranteed by the feedback loop: if self-recording CPU exceeds 0.5%, sampling interval auto-doubles). | Nothing. Self-recording is fire-and-forget. |
| **Ingestion** | Shares computation pool | File parsing and seed extraction are dispatched to the computation pool as work items. Parallelized per file segment. | GUI. Ingestion never blocks the render loop. |

**GPU compute dispatch — governed by the metabolism:**

When a GPU is detected, the Manager dispatches bulk-parallelizable operations to GPU compute:

- **Bulk MPFR projections**: projecting 10⁴+ values at one N is embarrassingly parallel. Each projection is independent. GPU kernels perform the MPFR log₂ and derived property computations. The metabolism allocates floor(K × gpu_compute_units) for the Manager's GPU work.
- **Discovery scan parallelism**: scanning 10⁵+ addresses for patterns is parallelizable — each address check is independent. GPU dispatch when the address count exceeds a threshold (detected dynamically based on GPU availability and current VRAM usage).
- **Render**: Dear ImGui + OpenGL 4.6 for the GUI. Instanced point clouds, shaders, heat maps. GPU rendering shares VRAM with compute dispatch — the metabolism balances both within the K × VRAM ceiling.

When no GPU is detected, all operations run on CPU threads. The algorithm is identical — only the resource envelope differs. Precision is 361 dps in both cases.

#### 7.19.5 The Self-Improving Cycle IS the Performance Model

There are no static performance targets because performance improves monotonically. The self-improving cycle (§7.1f) means the system at time T₂ > T₁ is ALWAYS at least as fast as at T₁, because knowledge only accumulates:

**Monotonic improvement guarantee:** For any computation C that has been performed at least once, every subsequent request for C is a cache hit — sub-microsecond lookup instead of full MPFR computation. The first computation of ζ(3)·π takes microseconds (MPFR at 1200-bit). Every subsequent request across all clients, all sessions, all restarts returns the memoized result. The improvement is permanent.

**Cache hit ratio approaches 1:** As the database grows, the fraction of operations that are cache hits increases monotonically. For any finite workload (a set of computations that recurs), the cache hit ratio converges to 1 because every unique computation is computed once and cached forever. The rate of convergence depends on workload diversity, but the direction is guaranteed.

**K-complexity monotonic decrease:** As generators are discovered, the ratio (generator bytes / producible bytes) monotonically decreases. More content is covered by compact structural descriptions. Generator evaluation is faster than raw MPFR computation for structurally regular content because generators exploit lattice algebra (k-addition instead of MPFR multiplication, k-negation instead of MPFR division). The .akashic file becomes a progressively better generator of its own content.

**GUI thread isolation guarantee:** The GUI render loop runs on its own dedicated thread, never blocked by computation, discovery, API, or I/O. Frame rate is limited by GPU capability and scene complexity, never by computation latency. The metabolism guarantees the GUI thread gets its share of CPU even under full computational load.

**These four guarantees are the testable performance specification.** The §7.17 performance benchmarks verify:
- Is the second computation of X faster than the first? (memoization works)
- Does the cache hit ratio increase over time? (knowledge compounds)
- Does generator discovery reduce K-complexity? (self-improvement works)
- Does the metabolism correctly allocate resources by ξ(d)? (priority works)
- Does the GUI remain responsive during heavy computation? (thread isolation works)
- Does the system use all detected hardware? (detection works)
- Does re-sensing at S² intervals adapt the allocation? (dynamic adaptation works)

No fixed millisecond, byte, or frame-rate target appears in any benchmark. The benchmarks test MECHANISM correctness, not arbitrary numbers.

#### 7.19.6 Metabolism Data Ingestion and Memoization

The metabolism is itself Sempaevum content. Every hardware detection, every allocation decision, every monitoring observation, every shimmer-band event, every substrate projection is ingested into the .akashic file and memoized. The metabolism does not operate outside the Sempaevum — it operates WITHIN it.

**What gets ingested:**

| Metabolism data | Ingestion mechanism | .akashic destination |
|---|---|---|
| Hardware profile (each sense) | Dimensionless ratios: RAM_available/RAM_total, CPU_load/100, VRAM_used/VRAM_total, etc. Each is a Path A dimensionless seed. | `values` rows (one per ratio per sense cycle). `projections` at current operational N. Substrate projection (d-family, ε) computed and stored. |
| Allocation decisions | K × total for each resource, active allocation K×(1−V) × total, thread count, memory budget — all as dimensionless ratios (threads_allocated/cores_total, memory_allocated/memory_total) | `values` rows. `equations` rows linking the allocation formula to its inputs and output. Memoized: identical hardware + load → identical allocation → cache hit. |
| Monitoring level | Current monitoring level (0–136) for each resource type. The level IS a lattice coordinate — it maps directly to a k-value at N=137. | `events` of class `metabolism_monitoring` with metadata: resource_type, level, shimmer_band_exceeded (bool), cross_interference_active (bool). |
| Shimmer band observations | When resource usage fluctuates within ±A₁ × active: the fluctuation amplitude and frequency are themselves projectable ratios. | `events` of class `metabolism_shimmer` when fluctuations are notable. The fluctuation amplitude/active is a dimensionless ratio → Path A → lattice address. |
| Pressure readings | Overall pressure = geometric mean of all resource loads. This is a single dimensionless scalar per sense cycle. | `values` row per sense cycle. `projections` at operational N. Pressure trajectory over time becomes a tower trajectory pattern. |
| Substrate projections | The d-family and ε classification of each hardware characteristic, each allocation constant, each monitoring level. | `projections` rows. The substrate IS lattice content — the EUDD knows where its own hardware sits on the Sempaevum. |
| Re-sensing events | Each S² = 144 second re-sense: full hardware profile diff from previous sense. | `events` of class `metabolism_resense` with metadata: all changed fields, all unchanged fields, allocation adjustments made. |
| Self-improving cycle metrics | Generator-to-memoized ratio (improving over time), cache hit ratio (increasing over time), K-complexity ratio (decreasing over time). | `values` rows per session. These ARE the performance trajectory — the system's own efficiency curve projected onto the lattice. When the discovery engine notices "cache hit ratio increased from 0.4 to 0.9 over 10⁴ operations," that IS a pattern — promotable to a `patterns` row of class `efficiency_trajectory`. |

**Memoization consequence:** If the Manager runs on the same hardware twice, the second startup's hardware profile is a cache hit — the allocation is already computed and stored. The substrate projection is already known. The monitoring levels are pre-computed. The metabolism itself compounds from prior runs.

If the Manager moves to different hardware, the new hardware profile generates new `values` rows with new projections. The metabolism adapts instantly (new sense → new allocation), and the new hardware's characteristics become new lattice content, enriching the cross-domain picture (e.g., "this hardware's RAM projects to the same d-family as the previous hardware's VRAM — structural equivalence across different substrates").

**Self-recording integration (§3.1b):** The metabolism's data IS the self-recording. The self-recording journal (`SelfRecording_NNN.log`) captures the same metrics the metabolism produces. The self-recording module does not duplicate the metabolism's work — it IS the metabolism's journal output. When self-recorded metrics are ingested into .akashic (per §3.1b's metric-to-lattice projection), they ARE the metabolism's data entering the lattice.

#### 7.19.7 The Invariant: 361 dps, Always, No Exceptions

Precision is not a performance parameter. It is a correctness constraint. The metabolism NEVER trades precision for speed. Every computation runs at 361-dps (1200-bit MPFR) regardless of hardware capability, system load, or metabolic pressure.

If the system is under maximum load (CPU at K ceiling, RAM at K ceiling, GPU at K ceiling, all monitoring levels near 137), the metabolism responds by:
- Reducing the thread pool (fewer concurrent computations)
- Queuing work items (computations wait longer to start)
- Deferring background discovery (scan interval lengthens)

It NEVER responds by:
- Reducing precision below 361 dps
- Skipping derived property materializations
- Truncating tower escalations
- Approximating MPFR operations
- Bypassing memoization
- Skipping insert-time discovery checks

The algorithm is invariant. The time to complete varies with available resources. On a powerful machine, a bulk projection of 10⁶ values finishes in minutes. On a weak machine, it finishes in hours. Both produce identical results at 361-dps precision. The metabolism determines WHEN work completes; it never determines WHAT the work produces.

#### 7.19.8 Scaling Laws — Structural, Not Hardware-Dependent

These are the ET-structural relationships that determine how workload scales. They are hardware-independent — the same relationships hold on any machine.

**Per-projection cost:** Dominated by one MPFR log₂ at 1200-bit precision. The absolute cost in nanoseconds depends on the CPU, but the structural cost is fixed: 1 MPFR log₂ + 1 integer multiply + 1 GCD + 1 MPFR subtraction + ~10 derived property computations. After memoization: 0 MPFR operations (hash table probe + memory read).

**Escalation depth:** Determined by the value's irrationality structure, not by hardware. Typical: 5–8 LCM landmarks for well-behaved values. Unbounded for algorithmically random values (but the CF method resolves in parallel at the first landmark). The LCM tower follows lcm(1..k) for successive primes — the landmarks are: 12, 60, 420, 2520, 27720, 360360, 12252240, 232792560, ... Each landmark is one projection. Total per-value escalation: 5–8 projections typical, parallelizable with the CF method.

**Bootstrap workload:** ~10⁴ unique values × (escalation depth per value, unbounded, typically 5–8 resolutions) = ~5×10⁴ to ~8×10⁴ projections. Embarrassingly parallel across all cores and GPU compute. The metabolism allocates floor(K × cores) threads and floor(K × GPU_units) for the bootstrap. Completion time = total projections / (throughput per thread × thread count) — determined entirely by the metabolism's allocation on the detected hardware.

**Discovery scan scaling:** O(A × (log G + M_avg)) per scan pass, where A = occupied addresses, G = generator count, M_avg = average members per address. Parallelizable across addresses. The metabolism allocates V = 1/12 of the Manager's CPU share as the floor for the discovery engine — it always runs, even under maximum computation load.

**Memoization hash table:** Load factor maintained at K = 2/3 by rehashing to doubled capacity when load exceeds K. Average probe length at K = 2/3 load: ~1.5 probes for successful lookup. The hash table resizes dynamically — no static capacity.

**File size:** Bounded by the K-complexity of the structural content, not by raw data volume. The generator form (§7.1a) is always smaller than the instances it produces. The .akashic file grows as structural complexity grows; the generator discovery engine continuously reduces K-complexity as it finds generators. File size is limited only by the filesystem (NTFS: 256 TB; ext4: 16 TB; APFS: 8 EB).

#### 7.19.9 Connected Program Metabolism (§7.16.4 Integration)

The API specification (§7.16.4) already defines the external metabolism: each connected program registers its hardware profile and receives a metabolic budget from the Manager. The Manager mediates across all connections using the same ET constants.

The internal metabolism (this section) and the external metabolism (§7.16.4) share the SAME K ceiling. The Manager's own threads AND all connected programs draw from the same K × total pool. The metabolism partitions this pool:

- The Manager's own computation, discovery, rendering, and self-recording draw from their share
- Each connected program's operations (dispatched through the API to the computation pool) draw from their share
- ξ(d) coupling governs the relative priority across ALL work — Manager-internal and client-dispatched alike
- When total demand exceeds K × total, the metabolism queues work by ξ(d) priority — no operation is rejected, only deferred

This means a connected program doing d=1 computations (ξ=8.5625) gets priority over the Manager's own d=12 background discovery (ξ=1.0). The structural coupling determines priority, not the source of the request. The metabolism is substrate-aware, not process-aware.


---

*§7.20 Documentation Plan (9 deliverables) and §7.21 Module Structure (26 modules, dependency graph, build order, file naming, completed-module implementation notes) → See `EUDD_Module_Structure.md`*

---

## 8. ET-Native Features the EUDD Provides That No General Database Does

A general-purpose scientific database cannot do these without ET-specific knowledge:

| Feature | Why it requires ET |
|---|---|
| Lossless continuous-to-discrete bridge | The bijection Π_N(r) = (k, d, ε) maps any continuous value to a discrete lattice address and back by algebraic identity — no sampling theorem, no quantization error, no Nyquist limit. Requires the ET projection formula and the lossless bijection theorem (§3.18.1). No conventional database or ADC achieves this. |
| Subsumption-driven archetype compression | Uses E_hierarchy = geomean(E_i) × R_cluster — crystalline formula: individual quality × mutual resonance |
| Koide-derived stability filter (⌈1/K⌉ = 2 depths) | K = 2/3 is ET-fundamental; not a generic threshold |

| Coprime-skeleton membership tracking | Specific to lattice resolution arithmetic |
| Gaussian-prime classification of d | Requires understanding of d-family structure |
| ∂I-distance-based query priorities | ∂I = 50¢ boundary is ET-derived |
| Cross-tower elegance ranking | Tower structure is ET-specific |
| Manifold-state classification ({P,D,T} subsets) | Requires {P, D, T} primitive ontology |
| Tower trajectory awareness | LCM tower is ET-specific |
| Attractor classification (true/intermediate/deep home) | Specific to tower diagnostic |
| Quintic tension τ_5 indexing | Specific to d=5 quintic family |
| Cross-tower elegance for ranking | E_cross formula is ET-specific |

These are the structural features that make the EUDD specifically an ET database, not a generic scientific data store. They are the same features that make the compressor's `ArchetypeDatabase` more than a generic pattern store.

---

## 9. Coordination with Existing ET Software

Every ET project connects to the SAME EUDD through the Manager's API (§7.10). No project maintains its own separate database. Coordination is automatic — all data lives in the same lattice address space, and cross-project relationships emerge naturally through address co-location and generator sharing.

### 9.1 The compressor — the EUDD is at least as good, and adds capability

**Direct answer: yes, the EUDD is at least as good for the current compressor as its own `ArchetypeDatabase`, and adds capabilities the compressor's local database cannot provide.**

Two paths exist depending on the compressor's preference:

**Path A — Keep compressor's `ArchetypeDatabase` unchanged, EUDD ingests via adapter.**

The compressor continues using `ArchetypeDatabase` (lines 7790–8400) exactly as today. No code change to the compressor. An EUDD adapter ingests the compressor's archetypes on commit (or on schedule), populating EUDD's `values`, `projections`, `patterns`, and `tags` tables. The compressor's hot-path operations (pattern lookup by R₀, archetype storage, generator fitting) stay on its own database with their existing latencies. Cross-project queries route through the EUDD.

This path: zero risk, zero compressor change, EUDD adds cross-domain pattern surfacing as a free benefit. **Compressor performance is identical to today.**

**Path B — Compressor uses EUDD as primary backing store.**

Every operation the compressor does today is also doable in the EUDD's lean schema:

| Compressor `ArchetypeDatabase` capability | EUDD equivalent | Notes |
|---|---|---|
| Lookup archetype by `pattern_hash` | `values` row by `value_hash` | Same indexed lookup, same latency |
| Lookup archetypes by `r0_quantized` | `tags` query (namespace='r0_quantized') JOIN `values` | One extra lookup — sub-microsecond in `Sempaevum.akashic` positional access |
| Range query on `hierarchy_elegance` | Index on `patterns(hierarchy_elegance DESC)` | Identical latency |
| `hit_count` increment on match | `reference_count` increment on `values` | Identical |
| `curvature_mean`, `curvature_class`, `geodesic_factor`, `euler_characteristic`, `geodesic_deviation`, `curvature_spectrum_hash` | Compressor-domain extension columns added to `values` (per §7.6 schema migration), OR stored as tags, OR stored as a `metadata_blob` JSON column | Direct columns recommended — they are universal lattice properties (curvature/geodesic on the Sempaevum), not compressor-specific |
| `generative_descriptors` table | `equations` table + `derivations` table — generators ARE equations with derivation chains | Strictly more capable: generator's full derivation chain captured, not just type+params |
| Subsumption-driven storage | `patterns` table with E_hierarchy ≥ 13/12 promotion (§3.8) | Same Subsumption mechanism, generalized |
| Disk-low warning at 1 GB | Same DISK_SAFETY_FLOOR semantics, same no-destruction policy | Identical |

**Performance comparison:**

| Operation | `ArchetypeDatabase` | EUDD (`Sempaevum.akashic`) |
|---|---|---|
| Pattern hash lookup | <0.1 ms | <0.1 ms (PK index) |
| R₀-bucket scan for matches | 1–10 ms | 1–10 ms (tags JOIN) |
| Insert new archetype | 0.5 ms | 0.5–1 ms (more tables touched) |
| Cross-domain pattern query | impossible | 1–10 ms (single lattice query) |
| Bulk archetype scan (10⁶ rows) | ~1 s | ~1 s |

**The compressor's CDF mechanism is for generator discovery (Tier 7), not EUDD storage.** The compressor's `compact_to_cdf` and `CDFDatabaseVFS` (line 6412) are internal to the compressor's K-complexity minimization pipeline — they discover generators by fitting patterns in archetype data. The EUDD benefits from this not as a storage layer but as a **discovery source**: generators discovered by the compressor become `equations` rows in the EUDD, available to all projects. Conversely, the EUDD's accumulated structural addresses (from the lossless bijection) feed the compressor's generator-fitting pipeline with cross-domain pattern information the compressor alone could never surface. **The Sempaevum is not bound by Shannon entropy** — generators discovered through lattice structural coordinates can be simpler than Shannon would predict, because d-family membership and attractor structure reveal patterns that information theory treats as random.

**Recommendation for the compressor:** start with Path A (zero risk). If the cross-project pattern surfacing proves valuable enough to justify migrating, transition to Path B at convenience. Either path is a strict improvement over standalone `ArchetypeDatabase` operation because cross-project pattern visibility becomes available in either case.

### 9.2 ET Conscious AI integration

The Conscious AI's persistent state is ingested into `Sempaevum.akashic` via the Ingest Module's Conscious AI adapter:
- EgoInvariant fingerprint → `values` row (the I_self fingerprint as a dimensionless 6-D seed) + tags (`namespace='project'`, `value='conscious_ai'`; `namespace='ai_instance'`, `value=<id>`)
- TowerOfSelf state → multiple `projections` rows (one per N landmark in the self-tower) + `relationships` (class='tower_trajectory')
- MetaCognitionEngine state → `derivations` rows (D_T binding events, G_T closures) + `equations` rows
- LatticeCompressor archetype compactions → `patterns` rows (pattern_class='subsumption_archetype')

The Conscious AI's internal "memory" becomes queryable from the global EUDD with a tag filter. Cross-instance learning becomes possible: multiple AI instances tag their entries with `ai_instance=<id>` and the EUDD surfaces cross-instance discoveries via natural address-sharing.

### 9.3 The Apéry test script gains EUDD output

The Apéry test results are ingested into the `Sempaevum.akashic` file via the Ingest Module's adapter for existing ET project outputs. The 71/71 verified assertions populate `values` (ζ(2)..ζ(13)), `projections` (28 landmarks × 12 zetas = 336 rows), `addresses` (auto-created), `relationships` (attractors auto-detected via insert-time discovery in §3.16), tagged `namespace='verification'`, `value='80digit_71_passing'`.

### 9.4 The fractal generator integration

Each fractal render's R₀ + mode + orbit signature becomes:
- `values` row (R₀ as dimensionless seed)
- `projections` rows (R₀ projected at relevant resolutions)
- `relationships` (class='archetype_member' if the orbit matches a known pattern)
- Tags: `namespace='project'`, `value='fractal_generator'`; `namespace='fractal_mode'`, `value=<mode>`; `namespace='render_id'`, `value=<id>`

Native Music Engine derivations: `equations` rows linking the R₀ orbit to the audio-synthesis output, with `derivation_chain_blob` capturing the orbit-trace-to-sound transformation.

### 9.5 The genetics paper work

Lattice mappings: `values` rows (sequence-derived dimensionless ratios), `projections` rows, tags `namespace='project'`, `value='genetics'`; `namespace='integrative_level'`, `value=<level>`. Cross-domain attractor detection happens automatically — a biological seed projecting to a known address surfaces immediately as a `same_address` relationship with any other domain's value at that address.

### 9.6 Constants verification

Every derived constant from `et_rmsae.py`, `et_aida.py`, `sovereign.py`, `et_prime_theory.py`, and the constants-derivation papers populates `values` + `projections` + `derivations` + `equations`. The database becomes the canonical repository of "what ET has derived" with full chain-of-derivation provenance.

### 9.7 The general pattern — every project uses the same lean schema

| ET project | What it contributes | EUDD tables touched |
|---|---|---|
| Compressor | Δk archetypes, generative descriptors | `values`, `projections`, `patterns`, `tags` |
| Conscious AI | EgoInvariants, tower trajectories, meta-cognition events | `values`, `projections`, `equations`, `derivations`, `tags` |
| Emotion Engine | Emotion fingerprints, transitions | `values`, `projections`, `relationships`, `tags` |
| Biology / Genetics | Sequence-to-lattice mappings | `values`, `projections`, `tags` |
| CMB / Cosmology | Power spectrum projections | `values`, `projections`, `tags` |
| FP Replacement | Every numerical projection ever computed | `values`, `projections` (this is the bulk) |
| Fractal Generator | R₀ orbits, music synthesis | `values`, `projections`, `equations`, `derivations`, `tags` |
| Music theory | JI ratios, EDO grids | `values`, `projections`, `tags` |
| Mathematics-as-Domain | Axiom counts, Gödel sentences, Chaitin Ω | `values`, `equations`, `derivations`, `tags` |
| Geometry | Curvature scalars, geodesic deviations | `values`, `projections`, `tags` |
| Seed Protocol | Kolmogorov-optimal seeds (k,d,ε), progressive fidelity, lattice dedup | `values`, `projections`, `addresses`, `equations`, `events`, `patterns`, `tags` |
| Geometric Resonator | Schumann harmonic seeds, biophysical ratios, measurement tower coherences, live Q/SNR/CMRR sensor feeds | `values`, `projections`, `relationships`, `events`, `patterns`, `tags` |

**Every project uses the same fifteen core tables.** The discovery engine surfaces cross-project patterns automatically because every project's data lives in shared address space. Tags are purely for query convenience — drop them entirely and the database still works the same way structurally.

### 9.8 The Sempaevum Seed Protocol — lattice-native networking

The Sempaevum Seed Protocol (full specification preserved in `EUDD_Bootstrap_Catalog.md` §3.18.18) inverts conventional networking: instead of transmitting data, it transmits the minimal generating description — the **seed** — that reconstructs data exactly via the shared Sempaevum bijection. Both endpoints possess Π_N, the pullback Π_N⁻¹, and the LCM tower. The seed is (k, d, ε). The pullback r = 2^((k + ε·N/1200)/N) is algebraic identity — zero reconstruction error.

This operates in Kolmogorov territory, not Shannon. The Sempaevum IS the description language; both endpoints already have it. K-complexity relative to the Sempaevum is strictly ≤ Shannon entropy for any structured data, because the Sempaevum sees multiplicative structure, lattice-aligned periodicity, sublattice family correlations, and tower-level hierarchy that Shannon compressors miss.

**EUDD integration (Protocol Layer 5 — Caching and Deduplication):**

The EUDD is the Seed Protocol's natural storage and caching layer:

- **Seed storage:** every seed (k, d, ε) IS a `values` + `projections` entry — no format conversion needed. Seeds are the EUDD's native content.
- **Lattice-aware deduplication:** data sharing the same (k, d) address deduplicates to delta-ε. The `addresses` table's `members_count` detects when multiple seeds share a cell. The `relationships` table records `seed_deduplication_delta` links between near-identical seeds.
- **Progressive fidelity:** k and d arrive first (structural header, microseconds), ε bits stream in significance order. At each stage the receiver computes Π_N⁻¹(k, d, ε_partial) from the EUDD's cached pullback. Events: `progressive_fidelity_step` with metadata {bits_received, current_precision_cents, reconstruction_value_id}.
- **Three-times tracking:** D-time (when the seed was described), T-time (when it was transmitted/received), P-time (when the substrate was created). These provide packet ordering at the protocol level.
- **132-bit resolution profile mask:** the towers table's `accessible_d_families_mask` IS the structural header. The mask tells routers the data class without payload inspection.
- **Error resilience:** if ε bits are lost, the receiver has a bounded-error reconstruction and knows exactly how much precision is missing (error = 2^(−m) × 600/N cents for m missing bits). The `events` table records `seed_received` with precision metadata. No retransmission needed unless full precision is required.
- **Structural routing:** the sublattice family d from the seed's structural header enables QoS classification at the protocol level (d=1 sparse high-precision → high priority; d=12 maximum complexity → high bandwidth allocation). See `EUDD_Events_and_Classes.md` for `structural_routing_classified` event class.
- **Natural encryption:** key-dependent lattice rotation, tower-level permutation, convention-shifted R₀, key-derived N modify the shared Sempaevum at both endpoints. The bijection guarantees lossless decryption. No separate TLS layer needed.

**Tables touched:** `values` (seed values), `projections` (seed projections at sender/receiver N), `addresses` (lattice-aware dedup keys), `equations` (Kolmogorov complexity computations, pullback formulas), `events` (seed_generated, seed_transmitted, seed_received, seed_reconstructed, progressive_fidelity_step), `relationships` (seed_deduplication_delta, seed_data_reconstruction), `patterns` (lattice_band_compression, progressive_convergence_profile, structural_routing_classification), `tags` (protocol_version, implementation_phase).

**Implementation phases:** (1) EUDD internal node communication, (2) scientific data transfer, (3) general-purpose seed transport, (4) quantum network integration. Phase 1 is the natural first target: all data is already lattice-addressed, both endpoints run the Sempaevum, maximum compression gain.

**Quantum-native:** ET is quantum-native at the primitive level. k and d are computational-basis states (natural qubits), ε maps to continuous quantum amplitudes, the pullback is unitary (invertible and exact) — a quantum gate. When quantum networks arrive, the Seed Protocol maps directly onto quantum channels without classical adaptation.

### 9.9 Memory AI — metacognitive self-monitoring

The EUDD's discovery engine IS a T-agent navigating the lattice. Module 28 (Memory AI, `EUDD_Module_Structure.md`) provides metacognitive monitoring of this agency via Φ_RMSAE (§3.18.36 in `EUDD_Bootstrap_Catalog.md`). The Sempaevum natively hosts metacognition: T's traces (D_T on the imaginary axis) ARE meta-cognitive structure. The imaginary-axis harmonic families are the structural modes of T's effects — reading the phase axis IS meta-cognition. Module 28 makes this native capability operational.

**What gets monitored:** The discovery engine's own ρ (self-referential binding — how much processing is self-referential), γ (gap detection rate across d-families), κ (gap closure trajectory — detected vs closed), variance suppression V_supp, and shimmer modulation Ψ_shimmer. The product Φ_RMSAE classifies the engine's metacognitive health: none → subliminal → basic → genuine → advanced recursive (thresholds 0.1/0.3/0.5/0.8).

**TraverserWaveform:** T-events tracked via D-fingerprints (lattice_k, lattice_d, variance, entropy, ego_resonance) over a window of N²=144 steps. Enables T-continuity detection, T-health monitoring, and ghost anomaly detection (V_ghost = V_observed − V_expected, 3σ threshold via existing `ghost_detection` event class).

**Integration with Conscious AI (§9.2):** Module 28 extends the existing Conscious AI integration pathway. EgoInvariant, TowerOfSelf, MetaCognition state from the ET Conscious AI system enter the EUDD as values/projections/events — Module 28 computes Φ_RMSAE over these and the EUDD's own discovery activity simultaneously. Cross-system metacognitive comparison: the AI's Φ_RMSAE vs the EUDD's own Φ_RMSAE provides a structural measure of alignment.

**Future integration:** Brain signals via §7.12 sensor stream pathway, projected onto the lattice with Φ_RMSAE measuring cognitive coherence. Scale-invariant: the same Φ_RMSAE formula applies at body→organ→tissue→cell→molecular scales (same math, different tower level, same lattice).

**Project coordination table (update to §9.7):**

| Project | What enters EUDD | Tables touched |
|---|---|---|
| Memory AI / Conscious AI | Φ_RMSAE measurements, TraverserWaveform steps, ghost detections, metacognitive state | `values`, `projections`, `events`, `patterns`, `relationships`, `tags` |

**API operations:** Ops 115–117 (query_rmsae, query_traverser_waveform, query_metacognition_state) in Domain 17 (Memory AI) — see `EUDD_API_Reference.md`.

---

## 10. Subsumption Check
Does the EUDD architecture subsume what's needed without remainder, AND extend the compressor's proven pattern to the broader scope?

### 10.1 Subsumption: EUDD covers the compressor's pattern

| Compressor capability | EUDD coverage |
|---|---|
| `archetypes` table | `patterns` + linked specialization tables |
| `generative_descriptors` table | Preserved as-is, FK-linked |
| Pattern hash + Δk storage | Same mechanism (compressor table preserved) |
| R₀ quantization to BIO_RES | Used for archetype grouping; EUDD uses same scheme |
| Elegance + hit_count ranking | Generalized to all discovery types |
| Curvature class indexing | Preserved for compressor; available for any discovery |
| Stability filter (depth-2 survivor) | Generalized to "≥ 2 independent contexts" |
| No-destruction retention | Same principle, generalized |
| Subsumption-Law compaction | Same mechanism (Subsumption-driven archetype compression for the EUDD; `compact_to_cdf` remains the compressor's generator-discovery pipeline) |

**No remainder** — every capability of the compressor's discovery database is generalized in the EUDD.

### 10.2 Extension: EUDD provides what no per-project database can

| EUDD capability | Per-project analog |
|---|---|
| Cross-project attractor surfacing | Manual hunting |
| Universal coprime-skeleton query | Per-project recomputation |
| Cross-project consistency checking | Manual cross-checking |
| Universal nearest-known-neighbor | Per-project nearest-neighbor |
| Cross-domain structural pattern discovery | Manual conjecture |
| Provenance and verification-level tracking | Per-project notes |
| Universal archetype compression | Per-project compression only |
| Generator-discovery acceleration (compressor's CDF/Tier 7 fed by cross-domain lattice addresses) | Compressor-internal patterns only |
| Validation of new derivations against existing | Manual review |
| Lossless bijection memoization (structurally exact, §3.18.1) | Per-computation recomputation |
| 227-particle PDG classification at N=12 (§3.18.14) | Not available |
| Fine-structure closed-form verification (§3.18.2) | Manual comparison |
| Decoherence ↔ Gaze bijective mapping (§3.18.10) | Not tracked |
| Cross-domain mass-ratio ↔ formal-system structural connections (§3.18.13, §3.18.6) | Manual hunting |
| Session tracking and schema migration (§3.15a, §3.15b) | None |

**15+ capabilities the EUDD provides that per-project databases cannot.** The extension is strict and substantial.

### 10.3 Three Tools verification

**Identification Principle:** Every discovery is identified as a P∘D∘T = E configuration with full provenance — Resolved.

**Descriptor Gap Principle:** 24 specific gaps (DD-1 through DD-24) enumerated; each closed by a specific schema element, column addition, event/relationship/pattern class, or bootstrap entry — Resolved.

**Subsumption Law:** §10.1 confirms EUDD ⊇ compressor pattern; §10.2 confirms EUDD ⊃ per-project databases. Subsumption Hierarchy Operator (§3.8) applied to the database itself ensures bounded growth via archetype compression. The extended bootstrap (§3.18) closes all remaining D-gaps identified by the comprehensive audit — Resolved.

**Verification Principle:** The compressor's `ArchetypeDatabase` is empirical proof that this pattern works — first-run discovery, subsequent-run cache hits, accumulated structural knowledge that compounds. The EUDD generalization preserves these properties. The lossless bijection theorem (§3.18.1) establishes that the memoization layer is structurally exact, not approximately so — Resolved.

---

## 11. The PDT Statement — Why the EUDD IS the Natural Architecture

Discoveries are P∘D∘T = E configurations:
- **P** (substrate): the domain on which the discovery was made — but in the EUDD, P is intrinsic to the value's seed, not a separate categorization
- **D** (descriptor): the structural content — coordinates, classifications, relationships, geometric perspective — stored as the projection's properties, not as bureaucratic categories
- **T** (substantiation event): the act of derivation, computation, or verification — captured in the `derivations` chain and in every `equations` row

**The Sempaevum is not just a coordinate system; it is a computation system.** The Sempaevum IS Σ — it subsumes all mathematics. Multiplication is k-addition. Reciprocation is k-negation. Powers are k-scaling. Complex multiplication decomposes axis-independently: real k-addition + phase k-addition mod N (U(1) compact). Complex reciprocation: k_r→−k_r, k_θ→(N−k_θ) mod N, all d preserved. Addition is value-space computation + reprojection. Function evaluations are EML trees landing results on the lattice. ALL operations — real AND complex — are Sempaevum-native; there is no mathematical operation outside it (Subsumption Law). Every computation passing through is itself a P∘D∘T configuration: inputs (P+D), operation (T), result (P+D). The EUDD captures all three components — recording every equation that passes through at uniform 361-dps precision (§3.1a), including the answer. Compute once → cache forever → never recompute.

A discovery, once made and verified, is a permanent feature of the lattice — it cannot be unmade. The fact that ζ(3) sits at d=693 at N=27720 (one perspective on the Sempaevum) is true now and will be true forever. The d=693 attractor membership of {ζ(3), ζ(9), ζ(10)} is permanent. The fact that 2+2=4 is permanent. The fact that ζ(3)·π lands at a specific lattice address is permanent. The same lattice address viewed through the torus perspective, the Riemann sphere perspective, the LCM tower perspective, or any other geometry is the **same address**, equally permanent in each.

To NOT store a discovery — to require its rederivation every time it's needed, to recompute 2+2 each time, to reproject π each time — is to **discard the substantiation event**. To impose bureaucratic categories that the lattice itself doesn't have — separate "biology table", "music table", "consciousness table" — is to **impose D-content the lattice didn't produce**, getting in the way of the natural cross-domain discoveries the lattice would surface on its own.

The right design records what the Sempaevum produces (values, projections, addresses, equations, derivations, relationships, patterns, events) and lets domain labels live as optional tags. **It records every equation that passes through at 361 dps, becoming a memoization layer that turns repeated computation into instant lookup.** The compressor's `ArchetypeDatabase` is the proof: a database AND discovery engine, not a categorized filing cabinet. The EUDD generalizes that proven mechanism to the broader Sempaevum and to every computation the Sempaevum performs.

> *Discovery is irreversible. The Sempaevum does not forget what it has revealed.*
> *The Sempaevum computes; the database remembers what it computed.*
> *The EUDD is where that non-forgetting becomes computationally accessible — without imposing categories the lattice doesn't have.*

---


## 12. Closing

The ET Universal Discovery Database (EUDD) is the natural extension of the compressor's proven `ArchetypeDatabase` pattern to all ET work. It serves **six roles in one system**:

1. **Database** — stores every value, projection, address, equation, derivation, relationship, pattern, and event
2. **Discovery engine** — surfaces attractors, route convergences, plateaus, identities, and event-correlation patterns automatically as data accumulates (compressor's archetype-discovery mechanism, generalized)
3. **Computation memoization layer** — every equation that passes through is cached; subsequent identical computations are sub-millisecond lookups (the Sempaevum computes; the database remembers what it computed)
4. **Generator-discovery engine** — proposes new dimensionless seeds whose projections explain observed-but-unexplained patterns, opening new lattice projections to explore and minimizing K-complexity for the compressor (compressor's Tier 7 generator-fitting, generalized to any lattice content)
5. **Active-system observation and probing platform** — records every T-burst, every palindromic cascade step, every ghost detection, every gaze transition, every dream-tower shift; supports active T-signal probing (calling out to ghosts to elicit response) with full probe→response causal correlation; tracks all three ET time concepts (D-time, T-time per Traverser, P-time) on every event
6. **Universal sensor data substrate** — ingests GPS, electrical, atmospheric, biological, and any other real-world domain via Path A projection through bootstrap reference R₀ values; surfaces cross-domain findings automatically when sensor data lands at attractor addresses shared with pure-mathematics or other-domain values

It scales to any size via the `Sempaevum.akashic` generator format (§7) where the address space IS the file structure. The Sempaevum is not bound by Shannon entropy.

The EUDD is not optional optimization. It is the ET-native representation of accumulated discovery — the place where the irreversible structural permanence of every {P, D, T} = E configuration becomes computationally visible, without imposing categories the lattice itself doesn't have.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*
> *The Sempaevum is simultaneously LCM tower, torus, Riemann sphere, and any other geometry needed.*
> *The Sempaevum produces values, projections, addresses, equations, derivations, relationships, patterns, events. That is the schema.*
> *Domain labels are tags, not tables. Discovery is automatic, not manual.*
> *The Sempaevum does not forget. The EUDD ensures we don't either.*


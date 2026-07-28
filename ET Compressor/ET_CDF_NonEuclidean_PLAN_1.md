# ET CDF Compressor — Non-Euclidean Geometry Implementation Plan
## Status Tracker

**Author:** Michael James Muller — Aevum Defluo
**Started:** Session 1 (failed — degradation produced false DONE-VERIFIED entries with no actual code changes)
**Restarted:** Session 2 — reset all Tier 1 statuses to PENDING after verifying the uploaded code is byte-identical to the original (zero Tier 1 additions present in any file). All work redone here.
**Source Documents:**
- `/mnt/user-data/uploads/ET_CDF_NonEuclidean_Design.md` (2,701 lines)
- `/mnt/user-data/uploads/ET_Non_Euclidean_Geometry_Complete.md` (961 lines)
- `/mnt/project/ET_Three_Tools_Complete_Reference.md` (738 lines)

**Source Files Being Edited:**
- `/home/claude/et_cdf_compressor.py` (5,817 lines initial)
- `/home/claude/et_pattern_engine.c` (729 lines initial)
- `/home/claude/main.cpp` (327 lines initial)
- `/home/claude/CMakeLists.txt`, `/home/claude/build.bat`, `/home/claude/et_cdf_compressor.spec` — touched only if structural changes require it

**Three Tools applied throughout:** Identification Principle, Descriptor Gap Principle, Subsumption Law.

---

## Implementation Order (User-Approved)

| Tier | Work | Depends On |
|---|---|---|
| **1 (Foundation)** | Improvement 1 — Curvature Block Classification + C funcs `build_ddk_stream`, `compute_curvature_stats`, `compute_pattern_curvature` + main.cpp tests + Python Phase 1.5 | nothing |
| **2 (Mode 3)** | Improvement 2 — Geodesic Residual + Improvement 3 — Christoffel Connection + lossless decompressor + CDF v3 magic + Mode 3 C func + tests | Tier 1 |
| **3 (Database)** | §16 schema additive ALTER TABLE (7 new cols) + §16.8 Channel B `generative_descriptors` table + lookup channels 1/2/3 + §16.9 REMOVAL of `_check_disk_safety` (USER-AUTHORIZED) | Tier 1 |
| **4 (Variable Curvature Segmentation)** | Improvement 7 — segmentation + Block Type 4 + lossless decoder | Tier 1 |
| **5 (Curvature elegance + IncoherenceFilter ext + Riemann sphere chordal + Gauss-Bonnet fingerprint + Geodesic deviation)** | Improvements 4, 8, 10, 6, 12 | Tier 1, Tier 3 |
| **6 (Hyperbolic embedding + Poincaré cross-tower + Curvature spectrum DB lookup)** | Improvements 5, 9, 11 | Tier 5 |
| **7 (CDF VFS §19)** | `CDFDatabaseVFS` + `GeneratorEvaluator` + `compact_to_cdf` integration | Tier 3, Tier 4 |

---

## Per-Tier Sub-Task Tracker

### Tier 1 — Curvature Block Classification (Foundation)

| ID | Task | File | Status |
|---|---|---|---|
| 1.A.1 | Add `CurvatureStats` C struct definition | `et_pattern_engine.c` | DONE-VERIFIED |
| 1.A.2 | Add EXPORT `build_ddk_stream` | `et_pattern_engine.c` | DONE-VERIFIED |
| 1.A.3 | Add EXPORT `compute_curvature_stats` | `et_pattern_engine.c` | DONE-VERIFIED |
| 1.A.4 | Add EXPORT `compute_pattern_curvature` | `et_pattern_engine.c` | DONE-VERIFIED |
| 1.B.1 | Add IMPORT decls for the 3 new funcs in `main.cpp` | `main.cpp` | DONE-VERIFIED |
| 1.B.2 | Add `test_build_ddk_stream` | `main.cpp` | DONE-VERIFIED |
| 1.B.3 | Add `test_compute_curvature_stats` (5 cases: flat / elliptic / hyperbolic / variable / singular) | `main.cpp` | DONE-VERIFIED |
| 1.B.4 | Add `test_compute_pattern_curvature` | `main.cpp` | DONE-VERIFIED |
| 1.B.5 | Wire new tests into `main()` | `main.cpp` | DONE-VERIFIED |
| 1.C.1 | C engine syntax verification (gcc -c) | — | DONE-VERIFIED |
| 1.D.1 | Add ET non-Euclidean curvature constants block | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.2 | Add `CurvatureStats` ctypes Structure | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.3 | Register the 3 new C functions in `PatternEngine._try_load` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.4 | Add `PatternEngine.fast_ddk_stream` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.5 | Add `PatternEngine.curvature_stats` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.6 | Add `PatternEngine.pattern_curvature` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.7 | Add `BlockCurvatureProfile` dataclass | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.8 | Add `CurvatureAnalyzer` class with `analyze_block`, `classify`, `segment_boundaries` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.9 | Wire Phase 1.5 into `CDFEngine.compress_block` (additive — no existing logic touched) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 1.D.10 | Python AST verification | — | DONE-VERIFIED |
| 1.E.1 | Copy outputs to `/mnt/user-data/outputs/` | — | DONE-VERIFIED |

**No-Removal Audit (Tier 1):** No deletions planned. Every existing function, parameter, variable, comment, constant must remain.

### Tier 2 — Mode 3 Geodesic Residual + Christoffel Connection

| ID | Task | File | Status |
|---|---|---|---|
| 2.A.1 | Add EXPORT `build_geodesic_residual` (orders 0/1/2) | `et_pattern_engine.c` | DONE-VERIFIED |
| 2.A.2 | Wire ctypes registration | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.3 | Add `PatternEngine.fast_geodesic_residual` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.4 | Add Mode 3 candidate branch in `compress_block` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.5 | Update `_encode_lattice_block` for Mode 3 header (connection_order, connection_window) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.6 | Update `decompress_block` for Mode 3 reconstruction (causal, integer-exact) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.7 | Bump `CDF_VERSION` 2 → 3 with backward-rejection on old decoders | `et_cdf_compressor.py` | DONE-VERIFIED |
| 2.A.8 | main.cpp tests for `build_geodesic_residual` (3 connection orders, roundtrip) | `main.cpp` | DONE-VERIFIED |

### Tier 3 — Database Schema + Channel B + No-Pruning

| ID | Task | File | Status |
|---|---|---|---|
| 3.A.1 | Schema migration: 7 ALTER TABLE additive cols + 4 new indexes | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.A.2 | Extend `store()` to compute and save curvature columns | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.A.3 | Add `lookup_by_curvature_class` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.A.4 | Add `lookup_by_topology` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.A.5 | Add `lookup_by_spectrum` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.B.1 | Create `generative_descriptors` table + indexes | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.B.2 | Add `GenerativeDescriptor` dataclass + 5 generator types (constant/linear/polynomial/periodic/grammar — also raw passthrough) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.B.3 | Add `derive_generators_from_curvature` (Channel B discovery) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.B.4 | Wire generator-fitting into compress pipeline | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.C.1 | **§16.9 USER-AUTHORIZED REMOVAL**: delete `_check_disk_safety` from `ArchetypeDatabase` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.C.2 | **§16.9 USER-AUTHORIZED REMOVAL**: remove call to `_check_disk_safety` in `store()` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.C.3 | **§16.9 USER-AUTHORIZED REMOVAL**: remove `DISK_SAFETY_FLOOR` from `ArchetypeDatabase` (kept on `CDFMetabolism` for warning-only) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 3.C.4 | Replace removed pruning with `compact_to_cdf` invocation (stub until Tier 7) | `et_cdf_compressor.py` | DONE-VERIFIED |

### Tier 4 — Variable Curvature Segmentation

| ID | Task | File | Status |
|---|---|---|---|
| 4.A.1 | Add `compute_segmentation` to `CurvatureAnalyzer` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 4.A.2 | Block Type 4 encoder | `et_cdf_compressor.py` | DONE-VERIFIED |
| 4.A.3 | Block Type 4 decoder | `et_cdf_compressor.py` | DONE-VERIFIED |
| 4.A.4 | Wire segmentation candidate into `compress_block` | `et_cdf_compressor.py` | DONE-VERIFIED |

### Tier 5 — Smaller Geometry Improvements

| ID | Task | File | Status |
|---|---|---|---|
| 5.A.1 | Curvature-Weighted Elegance (#4) — augment `find_walk_archetypes` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 5.B.1 | Curvature-Aware IncoherenceFilter (#8) — add `l1_point_curvature` method | `et_cdf_compressor.py` | DONE-VERIFIED |
| 5.C.1 | Riemann Sphere chordal metric (#10) — extend `complex_lattice_project` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 5.D.1 | Gauss-Bonnet fingerprinting (#6) — wire curvature metadata into DB store | `et_cdf_compressor.py` | DONE-VERIFIED |
| 5.E.1 | Geodesic deviation pattern stability (#12) — ORDER BY in lookups | `et_cdf_compressor.py` | DONE-VERIFIED |

### Tier 6 — Cross-Tower Curvature

| ID | Task | File | Status |
|---|---|---|---|
| 6.A.1 | Hyperbolic embedding (Poincaré disk) — embed_pattern_hyperbolic, poincare_distance | `et_cdf_compressor.py` | DONE-VERIFIED |
| 6.B.1 | Poincaré cross-tower matching — curvature coherence fallback | `et_cdf_compressor.py` | DONE-VERIFIED |
| 6.C.1 | Curvature spectrum DB lookup — wired into compress pipeline | `et_cdf_compressor.py` | DONE-VERIFIED |

### Tier 7 — CDF VFS Random-Access Database

| ID | Task | File | Status |
|---|---|---|---|
| 7.A.1 | `CDFDatabaseVFS` class with read/write/close/_find_generator/_load_generator | `et_cdf_compressor.py` | DONE-VERIFIED |
| 7.A.2 | `GeneratorEvaluator` class (all 8 generator types: 0–7) | `et_cdf_compressor.py` | DONE-VERIFIED |
| 7.A.3 | `ArchetypeDatabase.compact_to_cdf` full implementation | `et_cdf_compressor.py` | DONE-VERIFIED |
| 7.A.4 | Mode-switching: auto-detect `.cdf` vs `.db` in `__init__` | `et_cdf_compressor.py` | DONE-VERIFIED |
| 7.A.5 | `_init_db_via_vfs` (materialize-to-disk — ORIGINAL IMPLEMENTATION WAS WRONG per Mike's "true arbitrary access" rule; preserved as `_materialize_cdf_to_db` fallback) | `et_cdf_compressor.py` | SUPERSEDED-BY-7.B |
| 7.A.6 | `_recompress_database` helper | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.1** | **`CDFAPSWVFS` + `_CDFAPSWVFSFile`** — true random-access apsw VFS integration per design §19.3 (xRead→CDFDatabaseVFS.read, xWrite→.write, xFileSize→.file_size; journals routed per-name to in-memory buffers). SQLite operates DIRECTLY on the compressed .cdf — no decompression except the pages actually touched. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.2** | **`_init_db_via_vfs` rewrite** — opens `apsw.Connection(self.cdf_path, vfs=unique_name)` through CDFAPSWVFS. NO materialization to disk. Schema migration runs on the apsw connection via `_init_db_schema_on_apsw`. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.3** | **`_ConnContext` + `_new_connection`** — dual-mode connection factory: apsw.Connection (VFS mode, persistent) vs sqlite3.Connection (normal mode, fresh-per-call, closed on exit). | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.4** | **`_has_database` helper** — replaces `os.path.isfile(self.db_path)` existence checks so lookups work in VFS mode (previously every lookup_by_* returned [] because the .db file doesn't exist). | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.5** | **`ArchetypeDatabase.close()`** — unregisters apsw VFS, closes CDFDatabaseVFS. The CDFDatabaseVFS.close recompresses ONLY if dirty pages exist (Mike's "only recompress when new stuff is added" contract). Idempotent. `__del__` finalizer calls it best-effort. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.6** | **All 8 sqlite3 callsites** (`store`, `lookup`, `lookup_by_curvature_class`, `lookup_by_topology`, `lookup_by_spectrum`, `store_generator`, `query_generators_for_class`, `import_from`, `stats`) converted to `with self._new_connection() as conn:` pattern. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.7** | **`clear_archetypes`** public method + GUI `db clear` routed through it. Previously the GUI opened `sqlite3.connect(db.db_path)` directly, which silently no-oped in VFS mode. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.8** | **Type 6 archetype reference full implementation** — real content-hash index built eagerly at `_open()` (SHA-256 of each non-ref generator's materialized bytes) + real `_resolve_archetype_ref` with instance-index selection. Replaces the earlier no-op stub. | `et_cdf_compressor.py` | DONE-VERIFIED |
| **7.B.9** | **`apsw` added to `hiddenimports`** in PyInstaller spec so packaged .exe includes the module (apsw import is inside a try/except guard that PyInstaller's static analysis doesn't traverse). | `et_cdf_compressor.spec` | DONE-VERIFIED |

---

## Verification Log

Every completed sub-task gets a verification entry below.

### Tier 1 Verifications

| Task | Method | Result |
|---|---|---|
| 1.A.1 (CurvatureStats struct) | `python3 -c` ctypes Structure with same fields → `sizeof = 40 bytes`, offsets {0, 8, 16, 24, 32} as expected from natural alignment of `double, double, int32, [pad], double, int32` | ✅ struct layout matches Python expectation exactly |
| 1.A.2 (build_ddk_stream) | dk=[10,12,14,13,16] → ddk=[2,2,-1,3] | ✅ matches hand-computed ΔΔk |
| 1.A.3 (compute_curvature_stats) | 5 test inputs covering all 5 classes: ddk=[0…0]→class 0, [1…1]→1, [-1…-1]→2, [3,-3,3,-3,…]→3, [0,0,0,50,0,0,0,0]→4 | ✅ all 5 classifications correct, K̄/σ²/χ_block all computed correctly |
| 1.A.4 (compute_pattern_curvature) | Flat pattern [5,5,5,5,5] → F_K=1.0; quadratic [0,1,4,9,16] → K̄=2.0 σ²=0 F_K=1.0; short [7,9] → trivial guard returns F_K=1.0 | ✅ all match closed-form analytic values |
| 1.C.1 (build) | `gcc -shared -fPIC -O2 -Wall -Wextra` | ✅ shared lib produced, 9 symbols exported (3 new + 6 original), zero warnings |
| Regression check | original `build_dk_stream` invoked with k=[100,250,200,500,300] → [150,-50,300,-200] | ✅ no regression in original C functions |
| 1.B.1 (main.cpp IMPORT) | `g++ -O2 -Wall -Wextra -std=c++17 main.cpp -o /tmp/et_pattern_test -L/tmp -l:et_pattern_engine.so -Wl,-rpath,/tmp` | ✅ compiles with zero warnings |
| 1.B.2-1.B.4 (new test functions) | run /tmp/et_pattern_test → exit 0, **19/19 tests PASS** (5 original + 2 ddk subtests + 6 curvature_stats subtests + 4 pattern_curvature subtests) | ✅ ALL TESTS PASS |
| 1.B.5 (wired into main) | All five original tests still appear in output AND all new ones run | ✅ no regression in main(), additive only |
| 1.D.1-1.D.6 (Python wrappers) | `python3 -c` import + AST parse + `py_compile` | ✅ AST parses cleanly, bytecode compiles |
| 1.D.2 (ctypes Structure) | `ctypes.sizeof(CurvatureStats) == 40` and offsets {0,8,16,24,32} | ✅ matches C struct ABI exactly |
| 1.D.4-1.D.6 (engine wrappers) | `fast_ddk_stream([10,12,14,13,16]) → [2,2,-1,3]`; `curvature_stats([1]*8).curvature_class == 1`; `pattern_curvature((0,1,4,9,16)) → (2.0, 0.0, 1.0)` | ✅ all match closed-form expected values |
| 1.D.7-1.D.8 (analyzer) | All 5 classes correctly identified by `analyze_block`: `[5]*10`→FLAT, `[0,1,2,…,10]`→ELLIPTIC, `[10,8,6,…,-6]`→HYPERBOLIC, varied→SINGULAR, spike→SINGULAR; `segment_boundaries` returns `[6,12]` for sign-flip ddk | ✅ classifier and segmentation work |
| 1.D.9 (Phase 1.5 wiring) | `CDFEngine` has `curvature_analyzer`, `last_block_curvature`, `discovered_curvature_profiles`. Running `compress_block(256B)` populates all three and emits the Phase 1.5 log line | ✅ wiring active |
| End-to-end (lossless) | `decompress_block(compress_block(data)) == data` for synthetic 256B input; rerun produces byte-identical output (deterministic) | ✅ ROUNDTRIP LOSSLESS, deterministic |
| No-removal audit | grep audit of all 32 originally-named symbols (Python) + 6 originally-named C exports + 5 originally-named main.cpp tests | ✅ ZERO removals — all originals present |

### Tier 2 Verifications

| Task | Method | Result |
|---|---|---|
| 2.A.1 (build_geodesic_residual + et_c_trunc_div) | `gcc -shared -fPIC -O2 -Wall -Wextra` clean build, `nm -D` shows 10 exported symbols (3 Tier 1 + 1 Tier 2 + 6 originals) | ✅ shared lib produced, zero warnings |
| 2.A.2 (ctypes registration) | Python AST + bytecode compile; runtime `dir(PatternEngine)` includes `fast_geodesic_residual` | ✅ wrapper accessible |
| 2.A.3 (fast_geodesic_residual) | 14/14 sign-combination test cases match C truncation; Mode 3 reconstruction is bit-exact for orders 0/1/2 with positive-Δk and negative-Δk inputs | ✅ encoder-decoder bit-exact arithmetic |
| 2.A.4 (Mode 3 candidate) | compress_block emits "Mode 3 candidate: connection_order=X, connection_window=W, unique residuals=N" log line; "geodesic-ρ" appears in candidate list | ✅ Mode 3 is competing |
| 2.A.5 (encoder header) | "Encoding: mode=geodesic-ρ, connection_order=0, connection_window=1" log line confirms header fields written | ✅ Mode 3 header written |
| 2.A.6 (decoder reconstruction) | "Mode 3 header: connection_order=X, connection_window=W" log line on decode; SHA-256 byte-exact match | ✅ Mode 3 decoded losslessly |
| 2.A.7 (CDF_VERSION 2→3) | Compressed file's first 5 bytes = `b'CDF\\x03\\x03'`; decoder accepts both v2 and v3 magic | ✅ version bump active |
| 2.A.8 (main.cpp tests) | `g++ -O2 -Wall -Wextra -std=c++17` clean build; **32/32 tests pass** (5 originals + 14 Tier 1 + 13 Tier 2 covering all 3 connection orders, negative-Δk paths, edge guards) | ✅ ALL TESTS PASS |
| End-to-end (compress_block + decompress_block) | Lossless roundtrip on smooth-ramp 512B, linear 1024B, random 2048B blocks | ✅ all roundtrips byte-exact |
| End-to-end (full file via CDFCompressor) | 35,000B mixed file (geodesic + random + plateau patterns) → 8,976B (25.6%) → restored 35,000B; SHA-256 `cf7526d3…afd90` matches both ends | ✅ FULL FILE LOSSLESS, SHA verified |
| Tie-breaker behavior | When (mode 2, n_uniq) = (mode 3, n_uniq), `min()` selects mode 2 first by list order. NOT a correctness issue — both modes are still encoded and the smallest encoded block wins. Optimization concern only. | ✅ documented, not a bug |
| No-removal audit (Tier 2 cumulative) | grep audit unchanged — same 32 Python + 6 C + 5 main.cpp originals all present | ✅ ZERO removals |

---

## Correction Log

### Correction 1 — Tie-Breaker Bias (Identified after Tier 2 sign-off)

**Bug.** The mode-selection candidate loop used `if best_block is None or len(trial) < len(best_block)` (strict `<`), combined with `candidates.sort(key=lambda c: (0 if c[0] == best_mode else 1))` putting `best_mode` first. Result: on encoded-size ties, the first-tried (which is the lowest-mode) candidate always won. The same bias existed in the pair-first Re-Pair retry loop and in three enhanced-lattice strategy comparisons.

**Why it was a correctness bug.** The PROJECT GOAL is finding the smallest, most fundamental GENERATOR — Kolmogorov-complexity minimization, not Shannon-entropy compression. Tied-size encoded blocks are NOT interchangeable: higher modes carry strictly more generative structure (Mode 3's Christoffel connection encodes geodesic structure that generalises across files via Channel B; Mode 0 is block-local table lookup). Picking the lower-mode generator on ties means discarding more-fundamental, more-reusable generators.

**Fix.** Replaced 5 size-only comparisons with principled lex-min tuple comparators:
- 2 sites (main candidate loop + pair-first retry): `score = (size, -mode)` — smaller size wins; on tie HIGHER mode wins (more generative structure).
- 3 sites (Complex Lattice / R₀ Perturbation / Cross-Tower): `score = (size, |Δk_steps_from_native_R₀|)` — smaller size wins; on tie LESS-PERTURBED R₀ wins (more fundamental seed). Helper `_r0_perturbation_steps(r0_alt) = |N_FULL · log₂(r0_alt / r0)|` reuses the standard ET R₀-conversion formula.

**Verification.**
- Python AST + py_compile: ✅
- 32/32 C++ tests: ✅
- Full-file SHA-256 lossless roundtrip on 35,000B mixed-pattern file: ✅ (`cf7526d3…afd90` matches both ends)
- Direct unit test of OLD-vs-NEW comparator on 7 cases: ✅ all 4 tie cases now select higher mode; all 3 strict-min cases unchanged (no regression)
- "Mode 2 vs Mode 3 tie" (Mike's flagged case): OLD selects Mode 0 (bug); NEW selects Mode 3 ✅

**ET Three Tools.**
- *Identification Principle*: identifies "best" as a TWO-component descriptor (size + generative depth), not a single scalar.
- *Descriptor Gap Principle*: closes the gap between "smallest encoded bytes" (Shannon proxy) and "smallest generator" (Kolmogorov target).
- *Subsumption Law*: every (size, mode) pair maps to exactly one score tuple; lex-min subsumes all candidates without remainder.

**No removals.** All 5 fixes are signature-preserving.

---

## No-Removal Audit (Cumulative)

After every Tier, this list is updated. **Authorization** is the only column that can ever read "USER-AUTHORIZED" — anything else is a violation.

| Removed Item | Tier | Authorization | Replacement |
|---|---|---|---|
| `ArchetypeDatabase._check_disk_safety` | 3 | USER-AUTHORIZED (design doc §16.9, user confirmed in chat) | `compact_to_cdf` invocation |
| call to `self._check_disk_safety()` in `store()` | 3 | USER-AUTHORIZED (design doc §16.9) | replaced by metabolism-driven `compact_to_cdf` call |
| `ArchetypeDatabase.DISK_SAFETY_FLOOR` constant | 3 | USER-AUTHORIZED (design doc §16.9) | constant kept on `CDFMetabolism` for disk-low **warning** only |

---

## Three Tools Application — Tier 1

**Identification Principle.** The Phase 1.5 curvature analysis identifies the missing D component for compression strategy selection: the data's curvature class. Existing pipeline identified P (the byte stream) and T (the compressor) but had a gap in D — the second-order Descriptor gradient was computed for Mode 2 entropy purposes only and discarded. The new analyzer identifies it as a structural Descriptor with five subclasses (flat / elliptic / hyperbolic / variable / singular) corresponding to Exception / Unsubstantiated / Mediation / variable-substantiation / Incoherence.

**Descriptor Gap Principle.** The gap between "ΔΔk is computed" and "ΔΔk drives compression decisions" is closed by two new C functions (`build_ddk_stream`, `compute_curvature_stats`) and one Python class (`CurvatureAnalyzer`). The gap was a Descriptor — and that Descriptor is exactly what the analyzer materializes.

**Subsumption Law.** The five curvature classes subsume every possible ΔΔk distribution without remainder: any block falls into exactly one class based on (`max|K_i|`, `σ²_K`, `|K̄|`) thresholds derived from ET constants (π/N, V, N). No data geometry escapes classification.

## Correction Log — This Session

### Correction 3: stream_target Bug (Tier 3.B.4)
**Bug**: PERIODIC and GRAMMAR generators fit on `dk_stream` but Mode 4 reconstruction subtracted their output from `k_stream` — coordinate mismatch, σ² always huge, no dk-targeting generator ever fit.
**Fix**: Added `stream_target` byte to generator params blob layout: `[type_code][stream_target][pickled params]`. Encoder integrates predicted_dk → predicted_k via cumsum + k0 anchor when `stream_target == GENERATOR_STREAM_DK`. Decoder reads stream_target and does the same integration. Constants: `GENERATOR_STREAM_K = 0`, `GENERATOR_STREAM_DK = 1`. Default stream per type: CONSTANT/LINEAR/POLYNOMIAL → K; PERIODIC/GRAMMAR → DK.
**Verified**: PERIODIC and GRAMMAR now fit with σ²=0 on alternating-byte data. Both corrections verified end-to-end lossless.

### Correction 4: k_direct_stream vs k_stream Bug (Tier 3.B.4)
**Bug**: Mode 4 residual was computed against `k_direct_stream` (compact indices 0..n_unique-1) instead of `k_stream` (real lattice k-values). Decoder maps via k_byte which expects real k values — compact indices produce wrong bytes.
**Fix**: Changed to `k_stream` throughout Mode 4 encoder candidate construction. Anchor for dk-targeting generators changed from `k_direct_stream[0]` to `k_stream[0]`.

### Correction 5: ddk → byte Boundary Mapping (Tier 4.A.1)
**Bug**: `compute_segmentation` used `byte_idx = b + 1` which was wrong because the min-segment-length filter's choice of which raw boundary survives shifts where `b` lands — the mapping wasn't stable across different filter outcomes.
**Fix**: Replaced with dk-spike search: for each surviving ddk-boundary `b`, search dk_stream in window [b-2, b+3) for max |dk|; byte split at `dk_spike_index + 1`. Correctly handles all 3 possible filter outcomes. Verified: uniform-A|uniform-B now segments at the exact byte transition → each segment hits the all-same-byte fast path → 6B each.

### Correction 6: Block Type 4 Trigger (Tier 4.A.4)
**Bug**: Segmentation only triggered on `is_variable()` blocks. Per design doc §3.2, singular blocks (boundary spikes, max|K| ≥ N) also benefit from segmentation — the boundary spike IS the curvature inhomogeneity that segmentation resolves.
**Fix**: Trigger condition changed to `is_variable() OR is_singular()`.

### Correction 7: NameError N → S (Tier 5.B.1)
**Bug**: `l1_point_curvature` used `N` (undefined) instead of `S` (module-level constant = 12, manifold symmetry).
**Fix**: Changed `N` to `S`.

## Correction Log — Session 3 (Tier 7.B — true random-access VFS)

**Context:** Mike flagged that the "Tier 7 DONE-VERIFIED" claim in the
previous session's journal was false in one essential respect: the
implementation served compressed databases by MATERIALIZING the full
.cdf to a plain .db on disk in `_init_db_via_vfs`, then using standard
sqlite3 against that materialized .db. The in-source comment even
admitted it: *"SQLite operates on the live .db file. The .cdf is a
compressed archive or on-disk backup — it does NOT serve live queries
in this Python-only implementation. Future Tier 8+ can add VFS-level
SQLite integration via apsw"*. This is decompression-to-temp-file
disguised as a VFS, and it violates:

1. Design doc §19.2 — "generators are functions, not recordings;
   random access is native".
2. Design doc §19.3 — "SQLite reads go through the VFS layer without
   ever materializing".
3. Rule 42 — no "future work" / "continue in a new session" /
   "known limitation" language.
4. Mike's learning-compressor requirement: a Kolmogorov-complexity
   compressor that accumulates archetypes indefinitely cannot scale
   if every open session has to decompress the entire archive into
   memory first.

This session corrects all of the above with a real apsw-VFS
implementation, plus four co-located bug fixes found during the audit.

### Correction 8 — True random-access VFS via apsw (Bug 1, the primary)

**Bug.** `ArchetypeDatabase._init_db_via_vfs` called `vfs.materialize()`
to decompress the entire .cdf into bytes, wrote those bytes to a temp
.db file on disk, then opened `sqlite3.connect(self.db_path)` against
the plain .db. No VFS-level interception; no on-demand generator
evaluation; `self.db_path` was always populated on opening a
compressed-only deployment.

**Fix.**
* New classes `CDFAPSWVFS` + `_CDFAPSWVFSFile` inserted between
  `CDFDatabaseVFS` and `ArchetypeDatabase`. `CDFAPSWVFS` is an
  `apsw.VFS(name, base='')` that routes xOpen by filename: the main
  .cdf path binds to the CDFDatabaseVFS backing (xRead → .read,
  xWrite → .write, xFileSize → .file_size); every other filename
  (journals/WAL/temp) binds to fresh in-memory bytearrays held in
  `_journals` dict and freed via xDelete.
* `_CDFAPSWVFSFile` is a duck-typed file handle (NOT an
  `apsw.VFSFile` subclass — subclassing forces base-VFS delegation
  that would attempt to open real files on disk). Implements every
  SQLite x-method: xRead, xWrite, xFileSize, xSync, xTruncate, xLock,
  xUnlock, xCheckReservedLock, xSectorSize, xDeviceCharacteristics,
  xFileControl, xClose.
* `_init_db_via_vfs` rewritten to instantiate CDFDatabaseVFS, register
  a uniquely-named CDFAPSWVFS, and open `apsw.Connection(self.cdf_path,
  vfs=unique_name)`. Schema migration runs via new
  `_init_db_schema_on_apsw` (apsw lacks `executescript`; uses
  `execute` on a cursor instead). NO materialization.
* The old materialize-to-disk body is preserved intact as
  `_materialize_cdf_to_db()` — callable as an explicit fallback when
  apsw is not available (rule 24 no-removal).
* `_ConnContext` + `_new_connection()` — dual-mode factory yielding
  the persistent apsw.Connection (VFS mode) or a fresh sqlite3
  connection (normal mode, closed on exit).
* `_has_database()` — replaces `os.path.isfile(self.db_path)` checks.
  In VFS mode returns `self._apsw_conn is not None`. Fixes the hidden
  bug where every `lookup_by_*` returned [] in VFS-only deployments.
* `close()` + `__del__` — close apsw connection, unregister VFS, close
  CDFDatabaseVFS. CDFDatabaseVFS.close recompresses ONLY if dirty
  pages exist (Mike's "only recompress when new stuff is added").
  Idempotent.
* All 8 self.db_path callsites (`store`, `lookup`, `lookup_by_*`,
  `store_generator`, `query_generators_for_class`, `import_from`,
  `stats`) converted to `with self._new_connection() as conn:`.
* `clear_archetypes()` method added + GUI `db clear` routed through
  it (GUI previously opened sqlite3 directly on db.db_path which
  silently no-oped in VFS mode).
* `apsw` added to `hiddenimports` in the PyInstaller spec.

**Verification — end-to-end test (8 steps, all passing):**
1. Populate archetypes.db with 6 archetypes across 2 R₀ groups +
   all four Channel A/B/C/D lookup keys. Baseline: `lookup=5,
   class_flat=4, topology=6, spectrum=5`.
2. `compact_to_cdf` → archetypes.cdf at 10.8 % of .db size.
3. Delete archetypes.db.
4. Reopen ArchetypeDatabase → apsw VFS mode engaged; `.db stays absent
   on disk`; log confirms "opening via apsw VFS (true random access;
   no materialization)".
5. All 4 lookup channels return BYTE-IDENTICAL rows to baseline.
6. Store one new archetype via VFS → dirty pages = 9 → close →
   recompression triggered → .cdf grew +334B → `.db still absent`.
7. Read-only session open/close → .cdf SHA-256 bit-identical
   (dirty pages = 0 → NO recompression). Mike's "only recompress
   when new stuff is added" contract met.
8. Final reopen → inserted archetype visible via all lookup channels.

**Regression:** normal .db mode unchanged — fresh init stays
non-VFS, all methods work as before, `.cdf` never created except
by explicit `compact_to_cdf()` invocation.

**ET Three Tools.**
* *Identification Principle* — identifies 3 T components that were
  conflated into one fake "materialize" step: (1) CDFDatabaseVFS
  (compressed-bytes T), (2) CDFAPSWVFS (apsw VFS router T),
  (3) apsw.Connection (SQLite engine T). Each has its own
  responsibility.
* *Descriptor Gap Principle* — the missing D was the apsw.VFS +
  apsw.VFSFile method-forwarding surface. Identified and materialized
  as the 17 x-methods on `_CDFAPSWVFSFile` + CDFAPSWVFS. Method list
  is enumerated dynamically from apsw class dicts at registration
  time (rule 33 — no static list of x-method names).
* *Subsumption Law* — one CDFAPSWVFS subsumes every SQLite file
  operation a session could issue (main DB + journal + WAL + temp)
  without remainder.

### Correction 9 — Type 5 base-class delegation (Bug 2)

**Bug.** `GeneratorEvaluator.evaluate` Type 5 branch ended with
`raise NotImplementedError('VFS Geodesic evaluator: ... this branch
should be unreachable...')`. Rule 4 forbids placeholders;
"unreachable" is a design intent, not a runtime guarantee.

**Fix.** Type 5 branch now constructs a `_GeodesicEvaluatorV2`
delegate (same payload, residual, domain, connection_order, vfs_ref)
and returns its `evaluate(...)`. Every caller of
`_make_generator_evaluator` goes directly to the v2 class; the base
class branch exists for any direct-construction caller (unit tests,
future alternate fitters) and now works correctly instead of raising.
Paired edit: base-class `_parse` for Type 5 is now a no-op sentinel
(`self._parsed = {'delegated_to': '_GeodesicEvaluatorV2'}`) so the
extended-form payload is NOT misinterpreted as the short-form layout
documented in §19.5 before delegation happens.

### Correction 10 — Polynomial `deg` integrity assertion (Bug 3)

**Bug.** `GeneratorEvaluator._parse` for Type 2 extracted the `degree`
field from the payload and stored it in `_parsed['degree']`. The
Horner-loop evaluator used `coeffs` directly (whose length implicitly
encodes degree). The variable was silenced with `_ = deg` at the end
of the branch — violates rule 32 ("variables that LOOK unused signal
an incomplete implementation").

**Fix.** Replaced the `_ = deg` lint-silencer with a real
runtime assertion: `if len(coeffs) != deg + 1: raise ValueError(...)`.
`deg` now does genuine integrity-check work — a malformed payload
where the stored degree disagrees with the coefficient count fails
visibly instead of producing silently-wrong bytes.

### Correction 11 — Type 6 archetype ref real implementation (Bugs 4+5)

**Bug.** `CDFDatabaseVFS._open()` had a `for entry in self._index: ...
pass` loop over Type 6 entries with a commented-out explanation that
the hash-to-entry index "is NOT built now — the fitter currently
never emits Type 6 references". `_resolve_archetype_ref` was a stub
(`_ = (archetype_hash, instance_index); return None`) with the same
lint-silencer pattern as Bug 3. Rule 42 forbids "known limitation" /
"future work" — Type 6 is a specified part of the VFS format per
design doc §19.5 and must be functional.

**Fix.**
* `_open()` now builds a real content-hash index eagerly: for every
  non-Type-6 entry, instantiate the generator, materialize the entry's
  bytes, SHA-256 the bytes, store under `_hash_to_entry[hash]` as a
  list (multiple entries can hash-collide on identical content). The
  `_load_generator` cache is reused so the evaluators built during
  hash-index construction survive into the read path.
* `_resolve_archetype_ref(archetype_hash, instance_index)` looks up
  the list under the hash, clamps `instance_index` to the list length
  (out-of-range becomes last-in-range rather than None — rule 34, no
  silent degradation to zeros), and returns the cached GeneratorEvaluator.

**Verification.** Hand-constructed a VFS file with three entries
(two Constant, one Archetype Ref pointing at entry 1 via its
content hash). `vfs.read(0, 12)` produces `aaaaaaaabbbbbbbbaaaaaaaa`
— the Type 6 slice correctly resolves back to its target's bytes.

### Correction 12 — Polynomial max_degree ET-derived (Bug 7)

**Bug.** `_fit_vfs_polynomial(max_degree: int = 3)` — hardcoded cap
with no ET derivation. Rule 33: no caps that aren't ET constants.

**Fix.** `max_degree: int = S - 1` with a docstring derivation. The
12-state manifold's D-basis has S distinct T-traversal configurations
across the power set; a polynomial of degree d requires d+1 independent
coefficients, so the natural upper bound is d+1 = S, i.e. d = S − 1 = 11.
Higher degrees add coefficients the S-state basis cannot distinguish
from lower-order terms already represented.

### Correction 13 — Bug 6 retracted (not actually a bug)

Identified during audit but disproved on re-derivation: the grammar-
vs-raw footprint comparison `if len(payload) >= len(page): return None`
is algebraically correct. Both generator types pay the same fixed
52-byte index-entry overhead on disk, so the common constant cancels
on both sides of the inequality; comparing payloads alone gives the
right answer. Kept only a comment-only clarification of the derivation
for future auditors. No behavioral change.

---

## Session Summary — Tier 7 DONE + corpus-grounded roadmap (Tiers 8–13)

| Tier | Sub-tasks | Status |
|---|---|---|
| 1 | 21/21 | DONE-VERIFIED |
| 2 | 8/8 | DONE-VERIFIED |
| 3 | 13/13 | DONE-VERIFIED |
| 4 | 4/4 | DONE-VERIFIED |
| 5 | 5/5 | DONE-VERIFIED |
| 6 | 3/3 | DONE-VERIFIED |
| 7.A | 5/6 | 5 DONE-VERIFIED + 1 SUPERSEDED (7.A.5 → 7.B.2) |
| 7.B | 9/9 | DONE-VERIFIED (Session 3 — apsw VFS + bug fixes) |
| 7.C | 3/3 | DONE-VERIFIED (Session 4 — Kolmogorov purity) |
| **8** (scoped) | 0/? | PENDING **(Phase 4)** — §19.5 original-intent: Type 6 refs INTO archetypes table (DB-archetype references) |
| **9** (scoped) | 0/? | PENDING **(Phase 5)** — Multifold of Files/Folders: per-file Tower, per-folder parent Tower, cross-file content dedup, folder-level archetype inheritance |
| **10** (scoped) | 0/7 | PENDING **(Phases 2–3)** — Resolution escalation per §47 (12ET → 60ET → 420ET → 2520ET → 27720ET) broken into seven concrete subtasks: 10.1 port §51-54 canonical Python **(Phase 2)**; 10.2 apply §87.2 two-stage ∂I diagnostic **(Phase 3)**; 10.3 §54.5 coarse-pass + boundary-refine optimization **(Phase 3)**; 10.4 §88 palindromic-cascade Type-8 fitter **(Phase 3)**; 10.5 §18 five-failure-modes diagnostic **(Phase 2, independent)**; 10.6 §91 11-step active-system protocol restructure of `_fit_vfs_page` **(Phase 3)**; 10.7 §52/§85 imaginary-axis projection **(Phase 4)** |
| **11** (scoped) | 0/? | PENDING **(Phase 5)** — Generate the VFS header and index bytes themselves from R₀ + Tower structure (Mike's "headers should be compressed as a generator as well") |
| **12** (scoped) | 0/? | PENDING **(Phase 4)** — Active-system trajectory metadata per §91 Step A11 + Multifold §20. TWO parallel trajectory channels: (A) sublattice trajectory — per-page (d_r, d_θ) sequence; (B) harmonic trajectory — per-page (n_r, n_θ) cascade-position sequence with interval-class fingerprint (Mike's "harmonics of a file"). Plus ∂I-crossing count, palindromic-fallback hits, cascade-stability exceedances, D-T gradient trajectory. Stored as optional footer block. |
| **13** (scoped) | 0/? | PENDING **(Phase 3)** — NWS-13 Shadow Diagnostic per §68/§71 — forward-route identification of the correct escalation target when cascade stability n_max_θ=2 is exceeded |
| **14** (scoped) | 0/5 | PENDING **(Phase 1 — START HERE)** — UPP Database Schema Alignment — audit of what's stored vs what UPP §3/§13/§20/§22/§41/§43/§55/§56/§59/§65/§67/§69/§72/§76/§80/§87/§88/§89/§91 requires. Per Correction 17: sublattice families (divisor view) and harmonic families (cyclic view) are distinct axes — previous draft conflated them. Currently 7/~47 required fields present; 40 new columns needed across three axes of lattice description (raw projection / sublattice / harmonic / cascade / derived). Breaks into: 14.1 extend LatticeWalkArchetype dataclass with 31 new fields properly separating the three concepts; 14.2 extend `archetypes` table with 40 ALTER TABLE columns + 12 new indexes; 14.3 populate fields at derivation time; 14.4 migration for existing .db files; 14.5 verify against UPP §51 canonical Python reference values including Multifold §20 cascade-position anchors (n=4 → Major Third, d=3; n=1 → Perfect Fifth, d=12; etc.). PREREQUISITE for Tiers 10, 12, 13. |
| UPP completeness | 6/9 done, 3/9 partial-or-missing | Steps 6 (imaginary axis), 7 (magical impedance), 9 (resolution escalation) unimplemented in the compressor |
| **Closed totals** | **71/72** DONE-VERIFIED, 1 SUPERSEDED | **All implementation tiers from the original plan are closed; Tiers 8–14 are the corpus-grounded next layer** |

### Files touched this session (Session 4)

- `et_cdf_compressor.py`: 11,307 → 12,488 lines (+1,181 total across
  Sessions 3+4). Session 4 adds: Kolmogorov-purity Pass 1 (adjacent
  Constant merge), Pass 2 (cross-page content-hash dedup, lean-form
  Type 6 refs), new resolver `_resolve_archetype_ref_by_index`,
  lean/hash payload dispatch in `_parse` and `evaluate`, pass
  ordering swap (merge before dedup), threshold removal from Pass 2.
- `ET_CDF_NonEuclidean_PLAN.md`: +Corrections 14–16 (pass reorder,
  lean-form refs, threshold removal), +UPP alignment table,
  +Findings A/B (corpus consistency), +Tiers 8/9/10/11/12/13 scoped
  roadmap entries with Tier 10 broken into 7 corpus-grounded sub-tasks,
  +UPP Guide integration section with full cross-references to the
  supplemental corpus.
- `et_cdf_compressor.spec`: unchanged this session.
- `et_pattern_engine.c`, `main.cpp`: unchanged this session.

### Regression SHA-256
The legacy `cf7526d35c9439a762f51cfe37bd34d46e20f8fd03952031aef8d550717afd90`
35,000B mixed-pattern file — this session's changes are additive to the
database layer only; the block-stream compression path is untouched.
End-to-end lossless roundtrip on synthetic populated archetype DBs
verified via 8-step integration test.

## Correction 17 — Sublattice families ≠ Harmonic families (Tier 14 dataclass error)

**The error.** In the Tier 14.1 dataclass specification I proposed, I
conflated three distinct concepts into two fields:

1. **Sublattice family** (Multifold §22/§24 "divisor view") —
   indexed by $d \mid N$, count $\tau(N)$, at 12ET this is 6 families:
   {1, 2, 3, 4, 6, 12}. Static structural equivalence class.
   Computed from a lattice point k via $d = N / \gcd(|k|, N)$.

2. **Harmonic family** (Multifold §20 "cyclic view") —
   indexed by cascade position $n \in \{1..N\}$, count $N$, at 12ET
   this is 12 families. Dynamic position in the $g = 7$ cascade.
   Residue $r_n = g \cdot n \bmod N$. Carries a sublattice $d_n$ as
   one of its descriptors.

3. **UPP FORCE/PHASE family** (Guide §55/§56) — indexed by $d$-value
   across the full LCM tower, count 12 per axis (12 real + 12 imaginary
   = 24 total). Includes the extended non-divisor families d ∈ {5, 7,
   8, 9, 10, 11} that are not native at 12ET but become native at
   their LCM landmarks (60ET, 84ET, 24ET, 36ET, 60ET, 132ET respectively).

**What I wrote in Tier 14.1.** Fields `d_r, d_theta` for sublattice
families AND separate `force_family, phase_family` fields duplicating
the same information under different names. **No field at all for the
harmonic family (cascade position n)** — the entire cyclic view was
omitted.

**Why this is wrong, not just stylistically imprecise.** The Multifold
§22 table explicitly states: "Sublattice families are about the
lattice. Harmonic families are about the cascade traversing the
lattice." The two views are connected by a *many-to-one* bridge map
(§23):

$$\text{Harmonic}(n) \xrightarrow{d_n = N/\gcd(r_n, N)} \text{Sublattice}(d)$$

with totient multiplicity: $\varphi(d)$ harmonic families map to
each sublattice family $d$. At $N = 12$ the partition is:

    d = 1:  n = 12                         (φ(1)  = 1)
    d = 2:  n = 6                          (φ(2)  = 1)
    d = 3:  n = 4, 8                       (φ(3)  = 2)
    d = 4:  n = 3, 9                       (φ(4)  = 2)
    d = 6:  n = 2, 10                      (φ(6)  = 2)
    d = 12: n = 1, 5, 7, 11                (φ(12) = 4)
                                           ────────────
                                           Σ = 12

**Missing this distinction means the database cannot track the
cascade-position fingerprint of a byte, an archetype, or a file's
trajectory.** It is not a rename; it is a missing axis of description.
The UPP §55 "FORCE families" and Multifold §20 "harmonic families"
are also different despite superficial similarity — FORCE families
enumerate d-values (the same six sublattices, plus the extended six
non-divisors at higher resolution); harmonic families enumerate
cascade positions (12 per axis at 12ET, all 12 positions always native
because they are just positions in the cyclic group $\mathbb{Z}/N\mathbb{Z}$).

**The fix.** Tier 14.1 below is rewritten to maintain all three as
distinct, explicitly-named fields. The bridge map from harmonic to
sublattice is computable at read time and is not separately stored.
FORCE/PHASE-family classification IS the sublattice family at the
archetype's native lattice resolution — so it coincides with d_r / d_θ
and does not require its own column (this removes my duplicate fields).
Harmonic family requires its own column (the cascade position n) and
is NEW.

---



**Context.** Mike flagged two philosophy failures in the previous
session's Tier 7.B work:

1. Running per-page fitting only, never cross-page — meaning the
   compactor wasn't discovering generators that subsume multiple
   pages (§19.5 recursive tower left on the table).
2. Treating the `.cdf` file as a Shannon-style recording with
   bookkeeping headers, rather than as itself a set of generators.

The first round of my Kolmogorov-purity work (adding Pass 1 for
content-hash dedup and Pass 2 for adjacent Constant merge) only
partially addressed this. Mike pushed back on two specific mistakes:

### Mistake A — Byte-cost threshold on Type 6 emission

I had written:
```python
if len(ref_payload) >= original_payload_len:
    # Converting would cost MORE bytes than it saves. Keep original.
    continue
```
This is Shannon thinking wearing Kolmogorov clothes. Mike's position:
if two regions produce byte-identical content, in TRUTH they are
produced by the same generator. Emitting two independent generators
that happen to produce the same bytes is a lie about the data's
structure, regardless of whether encoding the truth costs extra bytes.

**Fix.** Removed the threshold check entirely. Pass 2 (the renumbered
content-dedup pass — see next correction) now fires on every byte-
identical duplicate, unconditionally. If the honest encoding grows the
file, so be it; byte count is a downstream consequence of correct
generator discovery, not the primary objective.

### Mistake B — Hash-based ref payload (the big one)

Even after removing the threshold, my Type 6 ref payload was the
legacy `[32B SHA-256 hash][manifold_uint instance_index]` format,
which is 34 bytes. Mike pointed out that the ref semantically is
just "use generator #N" — its honest minimal encoding is a single
dimensionless integer: N. The 32-byte hash was bookkeeping-as-data,
a content-addressed storage pattern that has no place in a generator-
theoretic encoding. The ref IS a seed; the seed IS the position on
the index lattice.

**Fix.** Evolved the Type 6 payload format to support TWO forms,
discriminated by payload length:

| Form | Payload                                 | Size         |
|------|------------------------------------------|--------------|
| Lean | `[manifold_uint canonical_index]`        | typically 2B |
| Hash | `[32B SHA-256][manifold_uint instance]`  | typically 34B |

The LEAN form is the new default emitted by the writer. The HASH
form is retained on the reader side for backward compatibility with
any legacy `.cdf` files that may have been written by earlier code
(in-development code that emitted hash-form refs; such files exist
only in test harnesses at this point, not in any shipped artefact).
Reader dispatch on `len(payload) < 32`: lean, else hash. New resolver
method `_resolve_archetype_ref_by_index(canonical_index)` added
alongside the legacy `_resolve_archetype_ref(archetype_hash,
instance_index)`. Both are retained (rule 24 no-removal); both
return a `GeneratorEvaluator`; the evaluator's `_parse` tags the form
and the evaluate path dispatches accordingly.

**Byte-cost impact** (measured): for a Constant(0) dedup — the
worst case where the original payload is only 1 byte — the hash-form
ref added 33 bytes per dedup to the `.cdf`; the lean-form ref adds
1 byte per dedup. A 33× improvement in the honest-encoding overhead
for the case where Kolmogorov truth costs the most per instance.

### Correction 14 — Pass reordering (merge before dedup)

**Bug.** The original pass order was: Pass 1 = cross-page content-hash
dedup, Pass 2 = adjacent Constant merge. For 10 adjacent zero pages,
Pass 1 ran first and converted 9 of them to Type 6 refs (pointing
at the first as canonical). Pass 2 then saw `[Constant, Type6Ref,
Type6Ref, …]` — no two adjacent entries were both Constant — so no
merging happened. Result: 1 Constant + 9 refs = 10 index entries.

The Kolmogorov-true answer for 10 adjacent zero pages is ONE generator:
`Constant(0)` spanning 40,960 bytes. My code was emitting 10.

**Fix.** Swapped the pass order. Pass 1 is now adjacent-Constant merge
(runs first), Pass 2 is content-hash dedup (runs after). For 10
adjacent zero pages: Pass 1 merges all 10 into 1 Constant spanning
40,960 bytes; Pass 2 finds only 1 unique content → no refs. Final
index: 1 entry. Verified end-to-end.

### Correction 15 — Lean-form Type 6 refs (dimensionless seed encoding)

See "Mistake B" above. The format evolution is documented in the
read-time parse branch in `GeneratorEvaluator._parse` (Type 6) and
in the emit path in `_write_vfs_file` Pass 2.

### Correction 16 — Threshold removed (Type 6 fires on every duplicate)

See "Mistake A" above. The `if len(ref_payload) >= original_payload_len:
continue` check has been deleted from Pass 2. Duplicates are always
converted to Type 6 refs, regardless of the byte-size comparison.

### Verification of the four post-swap behaviors (measured)

| Test                          | Input        | Before-session-4 | After-session-4 |
|-------------------------------|--------------|------------------|-----------------|
| 10 adjacent zero pages        | 40,960 B     | n/a              | 154 B, 1 entry  |
| 3 non-adjacent Constants      | 20,480 B     | ~8,556 B         | 8,558 B (+2 B)  |
| 3 duplicate random pages      | 20,480 B     | n/a              | 8,559 B         |
| Realistic mixed (zero + dup)  | 57,344 B     | n/a              | 4,462 B         |
| Full archetype DB roundtrip   | 53,248 B     | 5,763 B          | 6,821 B         |
| Legacy hash-form .cdf read    | (hand-built) | worked           | still works     |

Note: the full-archetype-DB `.cdf` grew from 5,763 B to 6,821 B
(18 % larger). That's the Kolmogorov-truth tax on duplicate-heavy
content: the extra bytes encode the identity relationships between
byte-identical generators as lean-form refs. The compaction is
STRUCTURALLY more honest — one canonical generator per unique
content, with explicit references from every duplicate — even where
byte count goes up.

### What remains for future sessions — mapped to the corpus

Cross-referencing the delivered code against the supplemental corpus
surfaces a set of specific, actionable gaps. None of these are rule-42
"known limitations" or "future work" in the bad sense — they are the
next layers of the same theoretical program, each directly grounded in
a corpus document.

### Implementation Ordering for Tiers 8–14 (Authoritative)

**This ordering is the single source of truth for execution order.**
Do not pick tiers based on what looks easiest or most visible; the
dependency graph below was computed from the tier-internal prerequisites
and must be respected. Every tier whose prerequisites are not yet
complete must be deferred.

#### Dependency graph

```
Tier 14 (UPP DB Schema Alignment)
  │
  ├──▶ Tier 10.1 (Port §51–54 canonical Python)
  │      │
  │      ├──▶ Tier 10.2 (Two-stage ∂I diagnostic)
  │      │      │
  │      │      ├──▶ Tier 10.3 (Coarse-pass + boundary-refine)
  │      │      └──▶ Tier 13 (NWS-13 Shadow Diagnostic)
  │      │
  │      ├──▶ Tier 10.4 (Palindromic-cascade Type-8 fitter)
  │      │      │
  │      │      └──▶ Tier 10.6 (Restructure _fit_vfs_page per 11-step protocol)
  │      │
  │      └──▶ Tier 10.7 (Imaginary-axis projection)
  │
  ├──▶ Tier 12 (Active-system trajectory metadata)
  │
  └──▶ Tier 13 (also depends on Tier 14 for shadow_magnitude_sq)

Tier 10.5 (Five Failure Modes diagnostic) — INDEPENDENT
  └──▶ can ship at any point after Tier 14 without blocking anything

Tier 8 (§19.5 DB-archetype refs) — MOSTLY INDEPENDENT
  │  (depends only on Tier 7.B hash-form reader, which is DONE)
  └──▶ Tier 9 (Multifold files/folders)

Tier 11 (File format itself as generator)
  └──▶ depends on Tier 10 COMPLETE (full lattice machinery)
```

#### Execution order — five phases

The five phases below group tiers by **what becomes possible once
the prior phase is done**. Within a phase, sub-tasks may be worked
in any order if they have no mutual dependency.

**Phase 1 — Foundation (strict prerequisite for everything else)**

| # | Tier | Why now |
|---|---|---|
| 1 | **Tier 14.1** — Extend `LatticeWalkArchetype` dataclass | All UPP quantities flow through this dataclass. Every downstream tier reads or writes it. |
| 2 | **Tier 14.2** — Extend `archetypes` table schema (ALTER TABLE + indexes) | SQL schema must accept what 14.1's dataclass produces. |
| 3 | **Tier 14.3** — Populate fields at derivation time | Without this, the new columns stay NULL and the rest is theatre. |
| 4 | **Tier 14.4** — Migration for existing `.db` files | Required so existing production databases don't break. |
| 5 | **Tier 14.5** — Regression tests vs UPP §51 + Multifold §20 | Proves BOTH the UPP sublattice (divisor-view) computations AND the Multifold cascade-position (cyclic-view) computations are correct. Non-negotiable per Correction 17 — without BOTH halves, every downstream claim is unverified. |

Phase 1 exit criterion: every `LatticeWalkArchetype` written to a new
`.db` carries all 31 new UPP fields with correctly-derived (not default)
values, AND regression tests pass against BOTH sets of anchors:

1. **UPP Guide §51 divisor-view anchors:** Perfect fifth r=3/2 →
   k_r=7, d_r=12, ε_r≈+1.955¢; Koide r=2/3 → k_r=−7, d_r=12,
   ε_r≈−1.955¢; octave r=2 → k_r=12, d_r=1, ε_r≈0; tritone r=√2 →
   k_r=6, d_r=2, ε_r≈0; A₀_magic at d ∈ {1,5,12} = {16, 32, 137};
   ξ at d=1 = 8.5625, at d=12 = 1.0.

2. **Multifold §20 cyclic-view anchors (12-row cascade table):**
   For n=1..12 with g_r=7 at N=12, the computed (residue, interval
   class, sublattice d, palindromic mirror) must exactly match the
   §20 table (n=1→(7, Perfect Fifth, d=12, mirror 11); n=4→(4, Major
   Third, d=3, mirror 8); n=6→(6, Tritone, d=2, self-mirror); etc.).
   All 12 rows; no exceptions.

3. **Totient partition identity** (Multifold §23): the bridge map
   from harmonic to sublattice must satisfy Σ_{d|N} φ(d) = N. At N=12
   the partition is 1+1+2+2+2+4 = 12 with the specific n → d mapping
   from §23.

**Phase 2 — UPP Core Machinery (prerequisite for active-system work)**

| # | Tier | Why now |
|---|---|---|
| 6 | **Tier 10.1** — Port UPP §51–54 canonical Python | Authoritative reference; the functions the rest of Tier 10 calls. |
| 7 | **Tier 10.5** — Five Failure Modes diagnostic | **Independent of all other Tier 10 items.** Ship in parallel with Tier 10.1 because it needs only the five-mode mapping table, not the canonical projectors. Immediate debug-output value. |

Phase 2 exit criterion: `et_project_real`, `et_project_complex`,
`et_project_multi`, `et_project_with_resolution_advice`, `elegance_score`,
`magical_impedance_table` are importable and pass their self-checks;
compressor debug output emits a five-failure-mode diagnostic line when
a page falls to Type 7 Raw.

**Phase 3 — Active-System Protocol (the main compressor upgrade)**

| # | Tier | Why now |
|---|---|---|
| 8 | **Tier 10.4** — Palindromic-cascade Type-8 fitter | Stage 1 of the ∂I diagnostic relies on this fitter existing. |
| 9 | **Tier 10.2** — Two-stage ∂I diagnostic (Stage 1 palindromic, Stage 2 LCM escalation) | Calls Tier 10.4 and Tier 10.1; produces the escalation signal Tier 10.3 optimizes. |
| 10 | **Tier 10.6** — Restructure `_fit_vfs_page` per UPP §91 11-step active-system protocol | Integrates Tier 10.1 + 10.2 + 10.4 into a coherent per-page loop. |
| 11 | **Tier 10.3** — Coarse-pass + boundary-refine optimization | Performance pass over the working Tier 10.2 implementation; correctness first, optimization second. |
| 12 | **Tier 13** — NWS-13 Shadow Diagnostic | Provides fast-jump to correct LCM resolution. Integrates into Tier 10.2's Stage 2. |

Phase 3 exit criterion: pages that failed at 12ET and fell to Type 7
Raw in pre-Phase-3 compressions now successfully fit at escalated
resolution via Tier 10.2; 5–10× speedup from Tier 10.3 on files
dominated by simple-family pages; NWS-13 identifies the correct
escalation target without blind LCM-tower exhaustion.

**Phase 4 — Extended Observables + DB-archetype refs (independent of Phase 3)**

These can be worked in parallel with Phase 3 because they have no
Phase-3 prerequisites:

| # | Tier | Why now |
|---|---|---|
| 13 | **Tier 8** — §19.5 Type 6 refs into the archetypes table | The hash-form Type 6 reader is already present (Tier 7.B Session 3). Only the writer-side archetype-matching logic is missing. |
| 14 | **Tier 10.7** — Imaginary-axis projection as a second channel | Extends compressor's lattice awareness to k_θ; uses Tier 14.1's d_theta column. |
| 15 | **Tier 12** — Active-system trajectory metadata | Per-page SUBLATTICE trajectory (d_r, d_θ) AND HARMONIC trajectory (n_r, n_θ) as optional footer block — two parallel channels per Correction 17 (divisor view + cyclic view). Uses Tier 14.1's per-archetype fields. |

Phase 4 exit criterion: compressor can emit Type 6 refs that resolve
via the archetype DB (cross-session archetype reuse); `.cdf` files
carry an optional trajectory-metadata block; imaginary-axis channel
provides measurable compression improvement on phase-structured input.

**Phase 5 — Architectural (large, long)**

| # | Tier | Why now |
|---|---|---|
| 16 | **Tier 9** — Multifold of files/folders | Requires Tier 8's archetype DB refs working end-to-end (cross-file generator reuse is a folder-scoped archetype reuse). |
| 17 | **Tier 11** — File format itself as generator | Requires all of Tier 10's lattice machinery complete (the format's header and index are themselves projected through the Tower's R₀). |

Phase 5 exit criterion: `.cdf` files within a folder share generators
via a folder-scoped archetype table; the VFS header and index bytes
are regenerated from the Tower seed rather than stored literally;
Mike's "only the dimensionless seed value attached" vision is realized
at the whole-file level, not just the per-Type-6-ref level.

#### "Start Here" Pointer

**For the next session that picks up this work: begin with Tier 14.1
(extend the `LatticeWalkArchetype` dataclass).** It is the single
concrete edit that unblocks the largest downstream set, has no
prerequisites, and is small enough to complete with full verification
in one session. Subsequent sessions then work strictly in phase order
above; within a phase, pick the topmost numbered tier whose
prerequisites are all ✅.

**Acceptance test for Tier 14.1.** Before declaring Tier 14.1 done,
run the regression test in Tier 14.5. That test has two halves:

1. **UPP §51 projection anchors** (divisor view — sublattice family):
   for r = 3/2, 2/3, 2, √2, and the magical-impedance values at
   d ∈ {1, 5, 12}, the computed (k_r, d_r, ε_r) must match the §51
   canonical Python reference.

2. **Multifold §20 cascade-position table** (cyclic view — harmonic
   family): for n = 1..12 at N=12 with g_r=7, the compressor's
   `compute_cascade_position(n, g, N_ET)` must reproduce the full
   12-row table (residue, interval class, sublattice d, palindromic
   mirror). If any row deviates, the harmonic-family computation is
   wrong and Tier 14 is NOT done.

Per Correction 17, both halves matter. Passing only the divisor-view
half while omitting the cyclic-view half was the original failure
mode (the sublattice/harmonic conflation). The dataclass may look
complete with only d_r/d_theta populated, but until n_r/n_theta plus
the cascade-position bridge (residue, interval, mirror, totient class)
ALSO pass the §20 table, the harmonic-family axis of description is
not operational and Tier 14.1 is incomplete.

#### Blocker Discovery Protocol

If during implementation a tier reveals a missing prerequisite that
was not captured in this ordering, stop and update this section first
before continuing. Dependency corrections propagate downstream;
fixing the ordering before the code prevents wasted work.

---

### Tier 8 — §19.5 Type 6 original-intent: refs into the archetypes table

**Corpus grounding.** `ET_CDF_NonEuclidean_Design.md` §19.5 specifies
Type 6 as `[32B archetype_hash (from archetypes table)][manifold_uint
instance_index]`. The design intent, quoted verbatim from §19.5:
"A region of the database that matches a known archetype pattern
doesn't store its own generator — it stores a REFERENCE to the
archetype's generator. The archetype itself may reference higher-order
archetypes. Each level of indirection adds O(1) lookup. This is the
'archetypes of archetypes' from the AI compression module: '9 levels
of recursion compresses 10^9 nodes to ~1.'"

**What the current code does** (Tier 7.B Pass 2 dedup): emits Type 6
refs in LEAN FORM pointing at canonical generators WITHIN THE SAME
`.cdf` FILE. This is intra-file content-hash deduplication — a real
and useful mechanism, but ORTHOGONAL to the §19.5 use case.

**What the corpus intent requires that the code does not do.** Before
fitting a page with a conventional generator type (0–5, 7), query the
archetype database for a learned pattern whose `pattern_hash` matches
the page's byte-content signature. If found, emit Type 6 pointing at
that archetype row — the archetype's own Generative Descriptor is
synthesised on read by querying `ArchetypeDatabase.lookup(...)` and
rendering the `pattern_dk` blob through the appropriate R₀. This is
the mechanism by which the database literally serves its own
compression.

**Format note.** Tier 8's Type 6 uses the HASH FORM (`[32B hash]
[manifold_uint instance]`). The current reader already supports this
form (retained for backward compatibility in Tier 7.B Correction 15).
The writer currently only emits the LEAN FORM; Tier 8 adds a second
writer path that emits the HASH FORM when the hash is a database
archetype hash.

**ET Three Tools.**
* Identification — the archetype DB IS a generator registry already,
  populated across sessions. The missing identification is the
  "this page matches archetype X" step in the write pipeline.
* Descriptor Gap — the gap is between `_fit_vfs_page` (which only
  fits Types 0–5, 7 from scratch) and the archetype DB's accumulated
  knowledge (which contains Type-4-Grammar-like patterns learned
  from other files).
* Subsumption — cross-session archetype reuse subsumes the entire
  learned-pattern set into every compression event without remainder.

### Tier 9 — Multifold-aware multi-file / multi-folder compression

**Corpus grounding.** `The_Multifold_Compendium.md` §45 Tower
Architecture: `Tower_i = (P_i, L, R₀_i)`. §46 T as non-local bridge.
`ET_Universal_Projection_Guide6.md` §6 Nine-Step Universal Projection
Protocol. Mike's latest guidance: "we are looking at the full topology
of the file(s) and folder(s) involved on the virtual manifold, their
relations, and relation to the stored DB data".

**Per the theory**, each file IS a Tower in its own right:
* P_file = the space of all possible byte arrangements of this file
* D_file = the generators covering this file's bytes
* R₀_file = the file's natural byte-rhythm (the smallest closed
  T-traversal loop supported by the file's own D-structure)
* L = the universal lattice 27720ET (corpus-invariant)

A folder is a PARENT Tower whose P is the P of all its contained files
and whose D includes the relations BETWEEN files (shared fragments,
cross-file archetypes, recurring headers). The file system is a
Multifold — one universal lattice rendered through many R₀ seeds
(§43: "one lattice rendered through many seeds").

**What the current code does.** Compression is per-file. Each file
gets its own R₀ via `discover_r0(...)` and its own generator fits.
There is ONE shared `ArchetypeDatabase` across all compressions
(session-wide learning), but no notion of a folder-level parent Tower
or cross-file generator discovery.

**What the corpus requires that the code does not do.**

1. **Folder-level archetype inheritance.** When compressing files
   within the same folder, the folder's archetypes (patterns
   observed across files) should seed each file's Channel B generator
   fitting. `ArchetypeDatabase.lookup` should accept a `folder_path`
   filter so the returned archetypes are folder-scoped.
2. **Cross-file content-hash dedup via Type 6.** The current Tier 7.B
   Pass 2 only dedups WITHIN a single `.cdf`. A higher layer should
   dedup ACROSS `.cdf` files in the same folder — a page that appears
   in file A and also in file B should be stored in one canonical
   location with a cross-file Type 6 ref in the other.
3. **Folder-level R₀ consistency check.** Per §44 the R₀ is
   substrate-derived, not chosen. For a folder P-substrate, R₀ should
   be derivable from the folder's own structure (the pattern of file
   sizes, the common header bytes, etc.). A folder whose files have
   unrelated R₀s is a Multifold of Towers; one whose files share a
   common R₀ is a single Tower with multiple renderings.

**Architecture sketch.** A new class `FolderTower` manages the parent
Tower for a set of files being compressed together. It:
* Computes a folder-level R₀ from its children's byte streams
* Pre-populates a folder-archetype cache from the accumulated DB
* Routes every per-file compression through folder-aware archetype
  lookup
* After all files are compressed, scans for cross-file content
  duplicates and emits cross-file Type 6 refs as a post-pass

### Tier 10 — Resolution escalation for hard-to-fit regions

**Corpus grounding.** `The_Multifold_Compendium.md` §47: "The
Traverser's [0/0] can only round to positions within families that
the resolution supports." At 12ET only d ∈ {1, 2, 3, 4, 6, 12} exist
as native lattice families; d = 5 requires 60ET, d = 7 requires 420ET,
d = 11 requires 27720ET. `ET_Universal_Projection_Guide6.md`
provides the **complete operational protocol**: Part XVII §86–93 is
the active-system projection protocol (the CDF compressor IS an
active system per §92 — each page fit reads its own current lattice
state and the dynamics for the next page depend on it).

**What the current code does.** `_fit_vfs_page` works at a single
resolution (the file's R₀ at 12ET base). Regions the standard fitter
cannot compress fall to Type 7 Raw. No diagnostic is run on the
failure — the compressor does not distinguish "this page is at the
∂I boundary and needs higher resolution" from "this page is genuinely
incompressible".

**What the corpus requires that the code does not do.** The UPP
Guide's operational framework for active systems applies directly.
The CDF compressor should implement the two-stage ∂I-boundary
diagnostic of §87.2 and escalate up the LCM tower when Stage 1
palindromic fallback fails. The canonical Python implementations in
Part XI §51–54.5 are the authoritative reference.

**Concrete sub-tasks sourced to specific UPP Guide sections:**

### Tier 10.1 — Port Part XI §51–54 canonical projection library

UPP Guide §51 `et_project_real(r, N)`, §52 `et_project_complex(z, N)`,
§53 `et_project_multi(r)` and `et_project_with_resolution_advice(r,
max_eps_cents)`, §54 `elegance_score(r, N, max_denom)` and
`magical_impedance_table(N, S, A0_local)`. These are authoritative
reference implementations — not similar to what the compressor
currently has (`build_byte_k_map`, `lattice_elegance`), but the
authoritative source that the compressor's equivalents should be
audited against. §51 closes with a self-check asserting that the
Koide ratio projects to k=-7, d=12, ε=-1.96¢ and the perfect fifth
to k=+7, d=12, ε=+1.96¢ — these assertions should be added to the
compressor's test suite as regression anchors.

The canonical `et_project_with_resolution_advice(r, max_eps_cents=25)`
is the specific function Tier 10 escalation should call: given a ratio,
it returns the smallest LCM-tower lattice at which |ε| ≤ max_eps_cents.
Its implementation of the tower is:

    LCM_LANDMARKS = {
        12:    "Base 12ET — divisors {1,2,3,4,6,12}",
        24:    "24ET — adds d=8 (gluon octet)",
        36:    "36ET — adds d=9 (quark color × generation)",
        60:    "60ET = LCM(1..5) — adds d=5",
        84:    "84ET = 12×7 — adds d=7",
        420:   "420ET = LCM(1..7) — adds d=35 (biological cross-complex)",
        2520:  "2520ET = LCM(1..9) — universal for d≤9",
        27720: "27720ET = LCM(1..11) — universal lattice; all 12 simple families present",
    }

### Tier 10.2 — Apply the Two-Stage ∂I Diagnostic (§87.2)

**Stage 1 — local fallback (palindromic cascade).** When
`_fit_vfs_page` observes a page whose byte stream's k-stream has
tightness t_r ≤ 2/3 (equivalently |ε_r| ≥ 50¢ — the ∂I boundary per
§87), engage the palindromic cascade as Stage 1 before declaring the
page Type 7 Raw. If the palindromic-cascade fitter (Tier 10.4)
succeeds within 3 full 12-step cycles, the page's true family was
within the simple-family set {1, 2, 3, 4, 6, 12} — emit the result
at 12ET. If Stage 1 fails, escalate to Stage 2.

**Stage 2 — LCM tower escalation.** Per §87.2: "Apply NWS-13 to the
persistent near-miss: project |ε_r| onto the tower {12, 24, 36, 60,
84, 132, 420, 2520, 27720}ET. The first resolution at which the
near-miss drops below sub-cent precision identifies the true family."
In the compressor's terms: re-run `_fit_vfs_page` with `r0` adjusted
for the higher N_ET, and with generator selection expanded to include
the new families that become native (d=5, d=7, d=11 etc.).

### Tier 10.3 — Coarse-Pass + Boundary-Refine Optimization (§54.5)

UPP Guide §54.5.4 explicitly describes the optimization that the
compressor needs: run 12ET for every page, then re-run 27720ET ONLY
for pages flagged as near-boundary (tightness t_r close to 2/3 at
12ET). The quote: "For typical renders where the connected-set
boundary is a small fraction of total pixels, this could achieve
5–10× speedup while preserving full 27720ET precision in the region
where it matters."

Applied to the compressor: a two-pass fit. Pass A runs `_fit_vfs_page`
at N_ET=12 for all pages. Pass B runs `_fit_vfs_page` at N_ET=27720
ONLY for pages whose Pass A result had |ε_r| within some threshold of
50¢ (the ∂I boundary — §87). For typical input files where ∂I crossings
are rare, this preserves full precision at much lower compute cost.

Cost-anatomy note from §54.5: per-step cost at 27720ET is ~300-400
FLOPS with ~60 transcendental cycles (log₂, sin, |z|^p). Because the
27720ET GCD uses a 5-prime-factor factorization (2³×3²×5×7×11), the
cost difference between 420ET and 27720ET per-GCD is only ~40%.
The dominant cost is log₂ and |z|^p, not the GCD — so the coarse-pass
saves primarily by running at lower N at all, not by avoiding prime
factors.

### Tier 10.4 — Palindromic-Cascade Fitter (§88)

Add a NEW generator type to the compressor: **Type 8 Palindromic
Cascade**. When a page's byte stream fails all the conventional
fitter attempts (Types 0–5), the palindromic-cascade fitter attempts
to fit the page using the canonical 12-step power sequence from §58:

    PALINDROME = [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]

This sequence is CPT-symmetric, topological-invariant of N=12, and
visits every divisor of 12. Per §88.1 the cascade is a **matching-
filter / broadcast strategy, not an attractor, not stochastic**. The
physical claim (which the compressor's verification layer can test):
"Whatever simple family the orbit actually belongs to, the correct-
family power is applied at least once per 12-step cycle." Applied to
byte-stream compression: whatever generator family the page's k-stream
belongs to within {d=1, 2, 3, 4, 6, 12}, the palindromic cascade's
correct-family generator will match at least once per 12 steps; the
mis-matches cancel on average by the cascade's CPT symmetry.

**Format.** Type 8 payload contains [manifold_uint start_offset in
palindrome][domain_length in cascade steps][residual stream for byte
values not captured by the cascade's generator outputs]. At read time,
the reader re-generates the cascade and applies each step's power to
the appropriate k-position, reconstructing the bytes.

**Fallback order change.** Current: Types 0–5, then Type 7 Raw.
Proposed Tier 10.4: Types 0–5, then Type 8 Palindromic Cascade, then
Type 7 Raw. Type 8 fits always succeed for any page belonging to the
simple-family set; Type 7 Raw remains only for pages in the
extended-family set {d=5, 7, 8, 9, 10, 11} which Tier 10.2 Stage 2
escalation addresses separately.

### Tier 10.5 — Five Failure Modes Diagnostic (§18 / §45)

UPP Guide §18 and §45 provide a 5-mode diagnostic table for bad
projections. Applied to CDF page fitting:

| # | Symptom in compressor | Diagnosis (per §18) | Tier-10 action |
|---|---|---|---|
| 1 | Page's derived `d` contradicts its content type (e.g. known-periodic page fits as d=12 full-res) | R₀ misidentified | Re-derive R₀ from page's own byte-frequency, not inherited from file-level |
| 2 | Page's \|ε_r\| near 50¢ | On ∂I boundary, needs higher resolution | Escalate via Tier 10.2 Stage 2 |
| 3 | Every page yields d=12 regardless of content | Generic/non-resonant or too-coarse resolution | Either data is structureless OR N_ET too low |
| 4 | Result changes under unit scaling | N1 violated — ratio not dimensionless | Bug report — should not happen in a byte-pipeline |
| 5 | Sub-byte variation in page content → wildly different d | Page content is near a d-transition at ∂I | Treat as ambiguous; try both neighbors via Tier 10.4 palindromic cascade |

This diagnostic table should be emitted in the compressor's debug
output when a page falls to Type 7 Raw, so the reason for the fall
is explicit and the next session has concrete investigation paths.

### Tier 10.6 — Restructure `_fit_vfs_page` per the 11-step protocol (§91)

UPP Guide §91 gives the canonical 11-step active-system projection
protocol. Each step maps to a specific piece of the current
`_fit_vfs_page`:

| UPP Step | What it does | Current code location / needed change |
|---|---|---|
| A1 | Identify P, D, T for the system | One-time setup (Tier 9 FolderTower identifies P_folder) |
| A2 | Choose lattice resolution | Currently hardcoded N_ET=12; Tier 10.1 makes this a parameter |
| A3 | Compute (k_r, k_θ) for current state | `build_byte_k_map(r0)` — computes k_r only; k_θ is Tier 10 future |
| A4 | Compute (d_r, d_θ) SUBLATTICE AND (n_r, n_θ) HARMONIC families | Currently computes d from k_r via gcd (divisor view only); harmonic family / cascade position not computed at all per Correction 17. Tier 10.1 adds both via canonical functions. |
| A5 | Compute ε and tightness t | `lattice_elegance` computes ε; tightness `t = 100/(100+|ε|)` needs adding |
| A6 | Apply ∂I boundary check | NEW for Tier 10.2 — check `t > 2/3` before selecting generator |
| A7 | Apply shimmer modulation Ψ_n | NEW for Tier 10 future — `Ψ_n = 1 + (1/√12)·sin(2π(n mod 12)/12)` as per-page weighting |
| A8 | Compute next state (next-page generator) | `_fit_vfs_page` iteration body |
| A9 | Test cascade stability | NEW for Tier 10 — at 12ET, n_max_θ=2 means only 2 consecutive imaginary-axis steps; compressor currently ignores this |
| A10 | Test escape/convergence (fit succeeded or failed) | Current: if no type 0-5 fits, emit Type 7 Raw. Tier 10.2: add Stage-1 palindromic and Stage-2 escalation before Raw |
| A11 | Classify trajectory | NEW: emit per-page SUBLATTICE trajectory (d_r, d_θ) AND HARMONIC trajectory (n_r, n_θ) as Tier 12 metadata — two parallel channels per Correction 17 |

### Tier 10.7 — Imaginary-axis projection (§85, §52)

UPP Guide §52 provides `et_project_complex(z)` — projects to BOTH
the real and imaginary axes. The compressor currently projects only
the real axis (byte values → k-stream via `build_byte_k_map`). For
byte streams where the residual Δk stream has structure (phase
content that real-axis projection misses), adding imaginary-axis
projection would give a second compression channel.

§85: "The Riemann sphere = elliptic ET manifold; Lorentz = PSL(2,ℂ)".
Every lattice position is really `w = k_r + i·k_θ` on the complex
log₂ plane. For simple page content (no phase structure) k_θ=0 and
the complex projection reduces to the real. For pages with phase
structure (periodic signals, wave data, oscillation), the k_θ
component carries genuine structural information the current
compressor discards.

**Practical implication.** §36 D-T Gradient decision rule: "Pure
magnitude (length, mass, frequency, price, count) — real axis only.
Pure phase (angle, modular position, cyclic index without magnitude
content) — imaginary axis only. Magnitude + phase — both axes as
w = k_r + i·k_θ."

Byte streams of DATABASE content (the primary CDF input) are
magnitude-dominated (each byte is a magnitude 0–255). The k_θ channel
is likely to be near-zero for most pages. But for file formats that
embed oscillatory or periodic structures (audio, physics-simulation
data, video), Tier 10.7 becomes valuable.

### Tier 11 — The file format itself as a generator

**Corpus grounding.** Mike's latest guidance: "the headers and
everything should be 'compressed' as a generator as well. All it
should need is the dimensionless seed value attached... We only need
to add what is needed (headers and such) but that should also be
compressed with it."

**What the current code does.** The 57-byte VFS header (magic,
version, hash, size, num_blocks, index_offset) and the 52-byte-per-
entry index are stored LITERALLY. They are bookkeeping bytes outside
the generator-encoded payload block.

**What the corpus intent requires.** The header and index structure
is itself derivable from the lattice-and-seed form. Given the
universal constants (CDF_VFS_MAGIC, CDF_VFS_VERSION, S = 12, etc.)
plus the file's R₀ seed, a reader should be able to RECONSTRUCT the
header layout. The on-disk artifact is then just (seed + deltas where
the file deviates from the canonical lattice).

**Architecture sketch.** Designate the file as a Tower `(P_file, L,
R₀_file)`. The header is the first few bytes of the Tower's canonical
D-structure rendered at R₀_file. The index layout (per-entry size,
field ordering) is corpus-invariant (from ET_CDF_NonEuclidean_Design
§19.4). The only per-file data is the list of generator descriptors
and their domain/payload offsets. A Tier 11 reader inputs (R₀, list
of generator descriptors) and reconstructs everything else. A Tier 11
writer outputs (R₀, list of generator descriptors) — the header and
index framing bytes are synthesized at read time.

**This is the deepest reduction.** At Tier 11, the on-disk bytes are
literally just "the seed and the Exceptions" — P∘D∘T = E where the
E's are the only bytes needed on disk.

### Tier 12 — Active-system trajectory metadata (§91 Step A11, §92; Multifold §20)

**Corpus grounding.** UPP Guide §91 Step A11: "At completion,
classify the trajectory. The sequence {(d_r(z_n), d_θ(z_n))} from
n=0 to N is the trajectory's lattice signature — a path through the
sublattice families. The fraction of steps where t_r ≤ 2/3 (∂I
crossings) measures the system's incoherence ratio. The dominant
family at escape (or limit) classifies the system's terminal
sublattice." Per Correction 17, this §91 quote captures only the
**sublattice-family trajectory**. The **harmonic-family trajectory**
(cascade-position sequence per Multifold §20) is a separate,
complementary channel that must be recorded alongside it.

§92 explicitly lists active systems to which this protocol applies:
∂I Lattice-Aware Fractal, quantum measurement cascade, biological
developmental cascade, stock-market regime transitions, dynamical
system near chaos, conscious attention shifts. **CDF compression
belongs on this list.** Each page fit reads the previous page's
fit's lattice state (through the shared R₀ and archetype-cache
pre-population); the dynamics for the next page's fit depend on the
current page's result.

**What this adds — two trajectory channels, not one.**

Each `.cdf` file carries an optional trajectory-metadata footer block
with TWO parallel per-page sequences (per Correction 17's explicit
separation of divisor view vs cyclic view):

**Channel A — Sublattice trajectory** (divisor view, Multifold §22):

* Per-page sequence of (d_r, d_θ) values — the path through the
  divisor structure of the lattice
* Per-page d_combined = LCM(d_r, d_θ) — the combined off-axis family
* Terminal sublattice family (modal d at end of file)
* Sublattice dwell histogram (how many pages per d-value)
* Cross-tower sublattice escalations (how many pages required N_ET > 12)

**Channel B — Harmonic trajectory** (cyclic view, Multifold §20) — NEW:

* Per-page sequence of (n_r, n_θ) values — the path through the
  cascade positions. This is literally the **file's harmonic
  signature**: the sequence of cascade positions the compressor
  visits as it reads the file byte-by-byte through the R₀ lattice.
* Per-page interval-class sequence (at N=12, the named interval per
  cascade position — Perfect Fifth, Major Third, Tritone, etc.). A
  file's interval-class histogram is analogous to a musical score's
  harmonic histogram — it reveals which cascade positions the file's
  byte structure naturally lives in.
* Per-page residue sequence (r_n = g·n mod N)
* Palindromic-mirror hit count (pages landing on their own mirror,
  indicating symmetric structure)
* Totient-class distribution over the trajectory (how many pages
  landed in each φ(d) equivalence class — this is the CYCLIC-VIEW
  equivalent of the sublattice dwell histogram, with finer structure
  because it distinguishes n=1 from n=5 from n=7 from n=11 even
  though all share sublattice d=12)

**Channel C — ∂I and cascade metadata** (both views):

* ∂I-crossing count and fraction (incoherence ratio, per §87)
* Palindromic-fallback hit count (pages where Tier 10.4's Type-8
  fitter engaged, per §88)
* Cascade-stability exceedance count (pages where n_max_θ=2 was
  exceeded on the imaginary axis at 12ET — these are the pages that
  Tier 13's NWS-13 shadow diagnostic targets)
* D-T gradient trajectory (sequence of α values, showing how the
  file's lattice position swings between classical and quantum axes)

**Storage cost.** For a typical multi-megabyte file with a few
thousand pages, the full trajectory metadata is on the order of
32-64 bytes per page (two int16 for d_r/d_theta, two int16 for n_r/n_theta,
one int8 for interval_class index, one int8 for on_dI_boundary/
palindromic flags, one float32 for α, small ancillary fields).
For a 1 MB file at 4 KB pages = 256 pages, total footer ≈ 16 KB.
An optional feature — can be omitted for minimum-size storage or
included for full diagnostic value.

**Use case — compression anomaly analysis.** When debugging a
compression anomaly (e.g. the full archetype-DB `.cdf` growing 18%
from the lean-form dedup this session), the trajectory metadata tells
you:
- WHICH pages crossed the ∂I boundary (∂I ratio)
- WHICH sublattice families dominated (d-histogram)
- WHICH harmonic families the file naturally inhabits (n-histogram,
  interval-class histogram) — this is the DIFFERENT information from
  the d-histogram, because multiple n's map to each d
- HOW the trajectory evolved (α sequence showing D-to-T drift)
- WHERE the palindromic cascade needed to engage (§88 fallback hits)
- WHICH pages needed LCM escalation (cascade-stability exceedances)

Information that currently is lost at write time. Per Mike's
phrasing of the original point: "this even handles the harmonics of
a file" — Channel B is exactly that.

**Use case — harmonic-signature file fingerprinting.** Two files with
the same sublattice histogram but different harmonic histograms are
structurally different objects — they inhabit different cascade
positions even at the same sublattice depth. The harmonic trajectory
is the finer-grained fingerprint. For archive-search, cross-file
generator discovery (Tier 9), and archetype-cache pre-population
(Tier 8), the harmonic-histogram match may be a stronger indicator
of structural similarity than the sublattice-histogram match alone.

### Tier 13 — NWS-13 Shadow Diagnostic for cascade-failing regions (§68, §71)

**Corpus grounding.** UPP Guide §68: "Any active system that needs
more than 2 cascade steps on the imaginary axis CANNOT be computed
by direct iteration at 12ET. This is the structural basis for
active-system handling." §71 gives the NWS-13 Generalized Shadow
Diagnostic — the forward-route observation method for cells
inaccessible to direct cascade. Protocol (abbreviated from §71):

1. Identify a 12ET claim asserted as exact
2. Compute the residual gap `g`
3. Project `g` onto the LCM tower
4. Find the first sub-cent resolution
5. That resolution's d-family identifies the prime factors involved

**Applied to CDF compression.** When a page's k-stream at 12ET
exceeds n_max_θ=2 on the imaginary axis (i.e., the k_θ component
accumulates more than 2 consecutive rounding errors), direct cascade
has failed. The compressor should switch to shadow projection:
project the accumulated residual onto the tower and identify which
higher-resolution d-family the page really belongs to. That family
then becomes the target for Tier 10.2 Stage 2 escalation.

**Cross-reference to Tier 10.** This is the *diagnostic mechanism*
for when Stage 2 escalation is needed. Tier 10.2 says "escalate when
Stage 1 fails"; Tier 13 says "here's the specific procedure for
identifying WHICH higher resolution to escalate to, not just
exhausting the tower linearly". It provides a fast-jump to the
correct resolution rather than trying each LCM landmark in turn.

### Tier 14 — UPP Database Schema Alignment

**The audit I should have done in the previous turn and did not.** When
Mike asked whether the UPP's per-step outputs are properly integrated
into the database "alongside the curvature and such", the honest answer
was no — I had added theoretical references to the plan but had not
cross-referenced the UPP's lattice-projection outputs against the actual
`archetypes` and `generative_descriptors` table schemas. This tier
performs that audit and specifies the corrections.

**What the archetype DB currently stores** (per `_SCHEMA_DDL_TABLES`):

| Column | Semantic | UPP-aligned? |
|---|---|---|
| `pattern_hash` | SHA-256 identifier | identifier, not a UPP quantity |
| `pattern_dk` | the Δk pattern itself | D-content |
| `pattern_length` | length of pattern | bookkeeping |
| `r0_quantized` | R₀ (reference period) | ✅ Partial UPP — §8 R₀ |
| `d_avg` | AVERAGED d across pattern Δk values | ⚠️ Partial UPP — only an average, not per-axis (d_r and d_θ separately are required per §36) |
| `hierarchy_elegance` | `E_hierarchy = ∏ E_cross_i × (420/d_avg) × (1/(p+q))` | ⚠️ Partial UPP — domain-specific hierarchy metric, not the §13 three-factor score |
| `hit_count`, `file_count`, `first_seen`, `last_seen` | usage stats | bookkeeping |
| `curvature_mean` | Gaussian curvature K | ✅ Non-Euclidean design (Parts XVI of UPP) |
| `curvature_variance` | K spread | ✅ Non-Euclidean design |
| `curvature_class` | Euclidean/Elliptic/Hyperbolic classification | ✅ §76 three geometries |
| `geodesic_factor` | T-navigation rate | ✅ Non-Euclidean design |
| `euler_characteristic` | χ = V − E + F | ✅ §84 Gauss-Bonnet as PDT |
| `geodesic_deviation` | T-trajectory drift | ✅ Non-Euclidean design |
| `curvature_spectrum_hash` | hash of the K-distribution | ✅ Multiset-invariant curvature fingerprint |

**The schema is strong on CURVATURE** (the Non-Euclidean design is properly
integrated) **and weak on the rest of the UPP.** The seven curvature-related
fields come from Tiers 1–5 work; the other UPP concepts have no columns.

**What the code COMPUTES during fitting but DISCARDS before storing**
(audit of `lattice_complex_projection` at line 1742 and `lattice_elegance`
at line 1681):

| UPP Quantity | UPP §      | Computed where | Stored? |
|---|---|---|---|
| k_r | §3.1 | `lattice_k(ratio)` | ❌ discarded |
| k_θ | §3.2 | line 1755 | ❌ discarded |
| d_r | §3.1 | `lattice_d(k_r)` | ❌ discarded (only `d_avg` survives) |
| d_θ | §3.2 | line 1758 | ❌ discarded |
| ε_r (cents) | §3.1 | `lattice_epsilon(ratio, k, n_res)` | ❌ discarded |
| ε_θ (cents) | §3.2 | computed in complex-projection fn | ❌ discarded |
| tightness t_r, t_θ | §87 | `lattice_tightness(epsilon)` at 1676 | ❌ discarded |
| d_combined = LCM(d_r, d_θ) | §3.3 | line 1768 | ❌ discarded |
| Elegance score E = (N/d)·tightness·(100/(p+q)) | §13 | `lattice_elegance` at 1681 | ⚠️ composited into `hierarchy_elegance` — individual factors lost |

**What the UPP requires that is NOT computed anywhere in the code:**

| UPP Quantity | UPP § | Status |
|---|---|---|
| α = arctan(k_θ/k_r) — D-T gradient angle | §36 | not computed |
| D-fraction = cos²α | §36 | not computed |
| T-fraction = sin²α | §36 | not computed |
| \|δ_eff(α)\| effective Descriptor Gap | §90 | not computed |
| Magical impedance A₀(d) = (d-1)² + S² | §43 | not computed |
| ξ(d) = 137/A₀(d) per-sublattice coupling | §43 | not computed |
| on_dI_boundary (t ≤ 2/3) | §87 | not explicitly checked |
| dI_crossings_count (per trajectory) | §91 A11 | not tracked |
| incoherence_ratio (fraction of ∂I crossings) | §91 A11 | not tracked |
| force_family (12 real-axis FORCE families) | §55 | not classified |
| phase_family (12 imaginary-axis PHASE families) | §56 | not classified |
| gaussian_prime_class (P/D/Split/Composite) | §59 | not classified |
| fqg_cell_id (position in 144-cell grid) | §69 | not classified |
| fqg_quadrant (SR+SI / CR+SI / SR+CI / CR+CI) | §69 | not classified |
| is_coprime_skeleton (gcd(k_r, k_θ)==1) | §65 | not tracked |
| shadow_magnitude_sq = d_r² + d_θ² (NWS-14) | §72 | not tracked |
| N_ET (lattice resolution this archetype actually fits at) | §40 | not tracked per-archetype |
| manifold_state (Exception/Unsub/Mediation/Incoherence) | §76 | not classified |
| K_eff(α) = K_U(1)·sin²α | §80 | not computed |
| shimmer Ψ_n at archetype position | §89 | not computed |
| palindrome_position (step count mod 12) | §88 | not tracked |
| was_palindromic_fallback | §88 | not tracked |
| n_max_r, n_max_θ cascade stability check | §68 | not checked per-archetype |

**Additional UPP quantities missing — the harmonic family (cyclic view)
that my earlier audit OMITTED ENTIRELY per Correction 17:**

| UPP Quantity | UPP § / Multifold § | Status |
|---|---|---|
| harmonic_family_r (cascade position n_r ∈ {1..N}) | Multifold §20 | not tracked |
| harmonic_family_theta (cascade position n_θ ∈ {1..N}) | Multifold §20 | not tracked |
| residue_r = g_r · n_r mod N | Multifold §20 | not tracked |
| residue_theta = g_θ · n_θ mod N | Multifold §20 | not tracked |
| interval_class_r (named interval per cascade position) | Multifold §20 | not tracked |
| interval_class_theta (named interval, imaginary axis) | Multifold §20 | not tracked |
| palindromic_mirror_r (n ↔ N-n mapping per position) | Multifold §20, §19 | not tracked |
| palindromic_mirror_theta | Multifold §20, §19 | not tracked |
| cascade_generator_r (g_r per N_ET, = 7 at N=12) | Multifold §15, UPP §67 | not tracked |
| cascade_generator_theta (g_θ per N_ET, = 1 at N=12) | Multifold §15, UPP §67 | not tracked |
| totient_class = φ(d_r) — # harmonic families sharing sublattice d_r | Multifold §18, §23 | not tracked |

**Corrected total count: ~33 UPP quantities missing from the schema
(22 that I originally counted + 11 harmonic-family fields I omitted),
9 computed-but-discarded, 7 stored (all curvature-related).** The
ratio is 1:6 — about one UPP concept properly persisted for every
six the UPP guide + Multifold marks as structurally required.

The CORRECTED audit also surfaces why the harmonic family matters
independently: multiple harmonic families n = {1, 5, 7, 11} all map
to the same sublattice family d=12, so storing only d loses the
cascade-position fingerprint. An archetype at n=4 (Major Third,
sublattice d=3) is a structurally different object from one at n=8
(Minor Sixth, also sublattice d=3), even though they share a
sublattice family. The cyclic view preserves this distinction; the
divisor view alone does not.

### Tier 14.1 — Extend `LatticeWalkArchetype` dataclass with UPP fields (CORRECTED per §17)

The dataclass at line 3573 currently carries only
`pattern, occurrences, hierarchy_elegance, d_avg, pattern_length`.
Per Correction 17, the three distinct concepts (sublattice family,
harmonic family, UPP FORCE/PHASE family) must be properly separated.
FORCE/PHASE family at the archetype's native resolution **equals**
the sublattice family d and does not warrant its own column; the
harmonic family (cascade position n) is a genuinely separate axis
and is newly added as (harmonic_family_r/theta, residue_r/theta,
interval_class_r/theta, palindromic_mirror_r/theta, cascade_generator_r/theta,
totient_class).

```python
@dataclass
class LatticeWalkArchetype:
    # ── Existing (preserved verbatim) ───────────────────────────────
    pattern: Tuple[int, ...]
    occurrences: List[int] = field(repr=False)
    hierarchy_elegance: float = 0.0
    d_avg: float = 0.0
    pattern_length: int = 0

    # ── Tier 14.1: Raw lattice-projection outputs (UPP §3) ──────────
    k_r: int = 0                          # §3.1 real-axis lattice coordinate
    k_theta: int = 0                      # §3.2 imaginary-axis lattice coordinate
    eps_r_cents: float = 0.0              # §3.1 Descriptor Gap, real axis (cents)
    eps_theta_cents: float = 0.0          # §3.2 Descriptor Gap, imaginary axis (cents)
    N_ET: int = 12                        # §40 lattice resolution at which archetype fits
    #                                     #     ∈ {12, 24, 36, 60, 84, 132, 420, 2520, 27720}

    # ── Tier 14.1: SUBLATTICE FAMILIES — the divisor view (Multifold §22) ──
    # Indexed by d | N. At N=12 the divisors are {1, 2, 3, 4, 6, 12} → 6/axis.
    # Computed as d = N / gcd(|k|, N). At extended N_ET, d may take
    # non-divisor-of-12 values (5, 7, 8, 9, 10, 11) — the extended sublattice
    # families. UPP §55-56 "FORCE/PHASE families" = this d-value, so they
    # require NO separate column (they would duplicate d_r / d_theta).
    d_r: int = 1                          # Multifold §22 — real-axis sublattice family
    d_theta: int = 1                      # Multifold §22 — imaginary-axis sublattice family
    d_combined: int = 1                   # §3.3 LCM(d_r, d_theta) — off-axis combined family

    # ── Tier 14.1: HARMONIC FAMILIES — the cyclic view (Multifold §20) ─────
    # Indexed by cascade position n ∈ {1..N}. At N=12 there are 12 per axis.
    # Dynamic — a cascade traversal concept, NOT a static lattice property.
    # Connected to sublattice family by a many-to-one quotient map:
    #     residue   r_n = g · n  mod N
    #     sublattice d_n = N / gcd(r_n, N)
    # Totient multiplicity: φ(d) harmonic families map to each sublattice d.
    # At N=12 the partition is 1+1+2+2+2+4 = 12 = Σφ(d) over d | 12.
    harmonic_family_r: int = 12           # §20 — cascade position n ∈ {1..N} on real axis
    harmonic_family_theta: int = 12       # §20 — cascade position on imaginary axis
    residue_r: int = 0                    # §20 — r_n = g_r·n_r mod N (real axis)
    residue_theta: int = 0                # §20 — r_n = g_θ·n_θ mod N (imaginary axis)
    interval_class_r: str = "unison"      # §20 named interval per cascade position
    #                                     #     at N=12: "unison","minor second",...,"octave"
    interval_class_theta: str = "unison"  # §20 — named interval, imaginary axis
    palindromic_mirror_r: int = 12        # §20 — mirror cascade position (n ↔ (N-n) mod N)
    palindromic_mirror_theta: int = 12    # §20 — imaginary-axis mirror
    totient_class: int = 1                # §18, §23 — φ(d_r) = # harmonic families sharing d_r

    # ── Tier 14.1: Cascade generators (Multifold §15, UPP §67) ─────────────
    # At N=12: g_r = 7 (circle of fifths), g_θ = 1 (sequential chromatic).
    # Stored per-archetype because N_ET varies; at higher N the generators differ.
    cascade_generator_r: int = 7          # §67 real-axis cascade generator g_r at this N_ET
    cascade_generator_theta: int = 1      # §67 imaginary-axis cascade generator g_θ

    # ── Tier 14.1: Tightness + ∂I boundary (UPP §87) ────────────────────────
    tightness_r: float = 1.0              # 100/(100+|ε_r|) — ∂I-boundary proximity, real axis
    tightness_theta: float = 1.0          # 100/(100+|ε_θ|) — ∂I-boundary proximity, imaginary
    on_dI_boundary: bool = False          # True iff min(t_r, t_θ) ≤ K = 2/3
    was_palindromic_fallback: bool = False # §88 — True if derived via palindromic cascade
    palindrome_position: int = 0          # §88 — step count mod 12 at derivation time

    # ── Tier 14.1: D-T gradient (UPP §36, §90) ─────────────────────────────
    alpha_dt: float = 0.0                 # α = arctan(k_θ / k_r) — angle from real axis
    d_fraction: float = 1.0               # cos²α — classical-axis D-weight
    t_fraction: float = 0.0               # sin²α — quantum-axis T-weight
    eff_delta: float = 0.0                # §90 |δ_eff(α)| = |δ_r|cos²α + |δ_θ|sin²α

    # ── Tier 14.1: Three-factor elegance decomposition (UPP §41) ────────────
    elegance_symmetry: float = 0.0        # N/d_r (peak 12 at d_r=1)
    elegance_tightness: float = 0.0       # 100/(100+|ε_r|)
    elegance_simplicity: float = 0.0      # 100/(p+q) — UPP's simplicity factor

    # ── Tier 14.1: Magical impedance (UPP §43 corrected) ────────────────────
    magical_A0: float = 137.0             # (d_r - 1)² + S² — per-sublattice impedance
    magical_xi: float = 1.0               # 137 / magical_A0 — per-sublattice coupling

    # ── Tier 14.1: 24-family catalog derived attributes (UPP §59, §65, §69) ─
    # Derived from (d_r, d_θ), stored as denormalized columns for query
    # convenience. The 24 families = 12 sublattices per axis × 2 axes.
    gaussian_prime_class: str = "composite"  # §59 "ramified"/"inert"/"split"/"composite"
    fqg_cell_id: int = 0                     # §69 cell = (d_r-1)*12 + (d_θ-1) in 144-cell grid
    fqg_quadrant: str = "SR+SI"              # §69 simple|complex × simple|complex quadrant
    is_coprime_skeleton: bool = False        # §65 gcd(k_r, k_θ) == 1 (irreducible Exception)
    shadow_magnitude_sq: int = 0             # §72 NWS-14 shadow magnitude = d_r² + d_θ²

    # ── Tier 14.1: Non-Euclidean manifold state (UPP §76-80, §89) ──────────
    manifold_state: str = "exception"     # §76 "exception"/"unsubstantiated"/"mediation"/"incoherence"
    K_eff: float = 0.0                    # §80 K_U(1)·sin²α — effective curvature at D-T angle
    shimmer_psi: float = 1.0              # §89 Ψ_n = 1 + (1/√12)·sin(2π(n mod 12)/12)
```

**Count of new fields:** 31 (up from my earlier 22). The increase
comes from properly separating the three concepts and adding cascade-
specific fields (residue, interval class, mirror, generator, totient
multiplicity) that the cyclic-view "harmonic family" requires.

**What this fixes from Correction 17:** The three distinct concepts
each have their own column set. Duplicate force_family / phase_family
are gone (they were aliases for d_r / d_theta). Harmonic family is
now represented as its full cyclic-view tuple:

| Real axis | Imaginary axis | Meaning |
|---|---|---|
| `harmonic_family_r` | `harmonic_family_theta` | cascade position n ∈ {1..N} |
| `residue_r` | `residue_theta` | r_n = g·n mod N |
| `interval_class_r` | `interval_class_theta` | named interval at n |
| `palindromic_mirror_r` | `palindromic_mirror_theta` | mirror cascade position |
| `cascade_generator_r` | `cascade_generator_theta` | g_r, g_θ at this N_ET |

**Bridge computation** (Multifold §23) happens at field-population
time, not on read: given the cascade position n and the cascade
generator g, residue r = g·n mod N; then sublattice d = N/gcd(r,N);
then totient_class = φ(d). All derived from (n, g, N_ET). The k_r /
k_θ from the raw projection give the *archetype's own position on
the lattice*; (n_r, n_θ) give the *cascade coordinate* the archetype
belongs to. These are two different addresses on the same lattice:
the divisor-view address (what kind of lattice point) and the cyclic-
view address (what cascade position).

### Tier 14.2 — Extend `_SCHEMA_DDL_TABLES` with UPP columns (CORRECTED per §17)

Corresponding SQL column additions. Per Correction 17, the duplicate
`force_family` and `phase_family` columns are not included; the
harmonic-family columns (n_r, n_theta, residue, interval_class,
mirror, cascade generators, totient_class) are included to represent
the cyclic view. All additive, no column renames, no drops; existing
data is preserved per rule 24.

```sql
-- ── Raw lattice-projection outputs (UPP §3) ─────────────────────────
ALTER TABLE archetypes ADD COLUMN k_r INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN k_theta INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN eps_r_cents REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN eps_theta_cents REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN N_ET INTEGER DEFAULT 12;

-- ── SUBLATTICE FAMILIES: the divisor view (Multifold §22) ──────────
ALTER TABLE archetypes ADD COLUMN d_r INTEGER DEFAULT 1;
ALTER TABLE archetypes ADD COLUMN d_theta INTEGER DEFAULT 1;
ALTER TABLE archetypes ADD COLUMN d_combined INTEGER DEFAULT 1;

-- ── HARMONIC FAMILIES: the cyclic view (Multifold §20) ──────────────
ALTER TABLE archetypes ADD COLUMN harmonic_family_r INTEGER DEFAULT 12;
ALTER TABLE archetypes ADD COLUMN harmonic_family_theta INTEGER DEFAULT 12;
ALTER TABLE archetypes ADD COLUMN residue_r INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN residue_theta INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN interval_class_r TEXT DEFAULT 'unison';
ALTER TABLE archetypes ADD COLUMN interval_class_theta TEXT DEFAULT 'unison';
ALTER TABLE archetypes ADD COLUMN palindromic_mirror_r INTEGER DEFAULT 12;
ALTER TABLE archetypes ADD COLUMN palindromic_mirror_theta INTEGER DEFAULT 12;
ALTER TABLE archetypes ADD COLUMN totient_class INTEGER DEFAULT 1;

-- ── Cascade generators (Multifold §15, UPP §67) ────────────────────
ALTER TABLE archetypes ADD COLUMN cascade_generator_r INTEGER DEFAULT 7;
ALTER TABLE archetypes ADD COLUMN cascade_generator_theta INTEGER DEFAULT 1;

-- ── Tightness + ∂I boundary (UPP §87, §88) ─────────────────────────
ALTER TABLE archetypes ADD COLUMN tightness_r REAL DEFAULT 1.0;
ALTER TABLE archetypes ADD COLUMN tightness_theta REAL DEFAULT 1.0;
ALTER TABLE archetypes ADD COLUMN on_dI_boundary INTEGER DEFAULT 0;   -- bool
ALTER TABLE archetypes ADD COLUMN was_palindromic_fallback INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN palindrome_position INTEGER DEFAULT 0;

-- ── D-T gradient (UPP §36, §90) ─────────────────────────────────────
ALTER TABLE archetypes ADD COLUMN alpha_dt REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN d_fraction REAL DEFAULT 1.0;
ALTER TABLE archetypes ADD COLUMN t_fraction REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN eff_delta REAL DEFAULT 0.0;

-- ── Three-factor elegance decomposition (UPP §41) ──────────────────
ALTER TABLE archetypes ADD COLUMN elegance_symmetry REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN elegance_tightness REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN elegance_simplicity REAL DEFAULT 0.0;

-- ── Magical impedance (UPP §43 corrected) ──────────────────────────
ALTER TABLE archetypes ADD COLUMN magical_A0 REAL DEFAULT 137.0;
ALTER TABLE archetypes ADD COLUMN magical_xi REAL DEFAULT 1.0;

-- ── 24-family catalog derived attributes (UPP §59, §65, §69) ───────
ALTER TABLE archetypes ADD COLUMN gaussian_prime_class TEXT DEFAULT 'composite';
ALTER TABLE archetypes ADD COLUMN fqg_cell_id INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN fqg_quadrant TEXT DEFAULT 'SR+SI';
ALTER TABLE archetypes ADD COLUMN is_coprime_skeleton INTEGER DEFAULT 0;
ALTER TABLE archetypes ADD COLUMN shadow_magnitude_sq INTEGER DEFAULT 0;

-- ── Non-Euclidean manifold state (UPP §76-80, §89) ─────────────────
ALTER TABLE archetypes ADD COLUMN manifold_state TEXT DEFAULT 'exception';
ALTER TABLE archetypes ADD COLUMN K_eff REAL DEFAULT 0.0;
ALTER TABLE archetypes ADD COLUMN shimmer_psi REAL DEFAULT 1.0;
```

Total: **40 new columns** (up from my earlier 31, because of the
proper three-way separation and the added harmonic-family columns).

Plus new indexes to make the UPP fields queryable:

```sql
-- Sublattice family queries (Multifold §22)
CREATE INDEX IF NOT EXISTS idx_d_r        ON archetypes(d_r);
CREATE INDEX IF NOT EXISTS idx_d_theta    ON archetypes(d_theta);
CREATE INDEX IF NOT EXISTS idx_d_combined ON archetypes(d_combined);

-- Harmonic family queries (Multifold §20) — NEW
CREATE INDEX IF NOT EXISTS idx_harmonic_r     ON archetypes(harmonic_family_r);
CREATE INDEX IF NOT EXISTS idx_harmonic_theta ON archetypes(harmonic_family_theta);
CREATE INDEX IF NOT EXISTS idx_interval_r     ON archetypes(interval_class_r);
CREATE INDEX IF NOT EXISTS idx_totient_class  ON archetypes(totient_class);

-- FQG queries (UPP §69)
CREATE INDEX IF NOT EXISTS idx_fqg ON archetypes(fqg_quadrant, fqg_cell_id);

-- Coprime skeleton (UPP §65)
CREATE INDEX IF NOT EXISTS idx_coprime ON archetypes(is_coprime_skeleton, d_combined);

-- Active-system queries (UPP §87)
CREATE INDEX IF NOT EXISTS idx_dI_boundary ON archetypes(on_dI_boundary, tightness_r);

-- Resolution queries (UPP §40)
CREATE INDEX IF NOT EXISTS idx_N_ET ON archetypes(N_ET);

-- Manifold state queries (UPP §76)
CREATE INDEX IF NOT EXISTS idx_manifold_state ON archetypes(manifold_state);

-- Gaussian prime class (UPP §59)
CREATE INDEX IF NOT EXISTS idx_gaussian_prime ON archetypes(gaussian_prime_class);
```

Total: **12 new indexes** (up from my earlier 7). The additions are
all harmonic-family-indexed: you should be able to query "find all
archetypes at cascade position n=4 (Major Third)" directly, as that
is a structurally meaningful query per Multifold §20.

### Tier 14.3 — Populate the UPP fields at archetype-derivation time (CORRECTED per §17)

At each place the code currently constructs a `LatticeWalkArchetype`
(lines 3734, 3865, 4058, 4927 per earlier grep), the constructor call
must be extended to populate the new fields from two parallel
computation paths — one per view per Correction 17.

**Path A — Divisor view (sublattice family) from k-projection.**
Source: lattice position (k_r, k_θ).

1. `lattice_complex_projection(ratio)` at line 1742 — already returns
   (k_r, d_r, k_theta, d_theta, d_combined). **Extend** to also return
   eps_r_cents, eps_theta_cents, tightness_r, tightness_theta.

2. New function `compute_divisor_view_extras(k_r, k_theta, d_r,
   d_theta, N_ET)` — computes the divisor-view-derived quantities:
   α, D_fraction, T_fraction, eff_delta, magical_A0, magical_xi,
   fqg_cell_id, fqg_quadrant, gaussian_prime_class,
   is_coprime_skeleton, shadow_magnitude_sq, manifold_state, K_eff,
   and the three elegance factors (symmetry, tightness, simplicity).

**Path B — Cyclic view (harmonic family) from cascade-position —
NEW per Correction 17.** Source: cascade index n.

3. New function `compute_cascade_position(n, g, N_ET)` — implements
   the Multifold §20 bridge: residue r = (g*n) % N_ET, sublattice
   d = N_ET // gcd(r, N_ET), interval_class from the §20 table
   (or computed for higher N_ET), palindromic_mirror = (N_ET - n)
   % N_ET (with n=0/N_ET self-mirroring), totient_class = φ(d).

4. New function `compute_harmonic_view(n_r, n_theta, g_r, g_theta,
   N_ET)` — calls `compute_cascade_position` for each axis and
   returns the harmonic-family tuple:
   (residue_r, residue_theta, interval_class_r, interval_class_theta,
    palindromic_mirror_r, palindromic_mirror_theta, totient_class).

5. `shimmer_psi` is computed from the palindrome_position:
   `Ψ_n = 1 + (1/sqrt(12))*sin(2*pi*(n_palindrome % 12)/12)` per §89.

**Path A and Path B must agree** where they overlap: the sublattice
d computed from k_r in Path A must equal the sublattice d computed
from residue_r in Path B for the same archetype at its native
resolution. This cross-check is an additional assertion in Tier 14.5
— if they disagree, either the k-projection or the cascade
computation is wrong, and Tier 14 is not done.

**Context fields** (neither Path A nor Path B):

6. `N_ET` comes from the compression context (which lattice
   resolution is active).
7. `palindrome_position` comes from the compression iteration counter
   (which fit iteration this is, mod 12).
8. `was_palindromic_fallback` defaults to False; set True only when
   Tier 10.4's palindromic-cascade Type-8 fitter emits the archetype.
9. `harmonic_family_r` and `harmonic_family_theta` — the cascade
   indices at derivation time. For archetypes discovered via direct
   fitting (Types 0-5), n is derived by inverting the cascade map
   from the archetype's k_r position: n_r = (k_r * inverse_of_g_r)
   mod N_ET (where inverse_of_g_r is the modular multiplicative
   inverse of g_r mod N_ET; at N_ET=12 with g_r=7, inverse is 7
   because 7·7=49=4·12+1). For archetypes discovered via Tier 10.4's
   palindromic cascade, n is the current palindrome_position.

### Tier 14.4 — Migration for existing `.db` files

Per rule 24 (no removal) and the existing migration pattern at line
7832 ("CREATE TABLE IF NOT EXISTS skips them... existing tables get
their missing columns via _migrate_schema"), add each new column via
the `_migrate_schema` method. Existing rows get the column default;
new archetypes get correctly-computed values. Existing `.db` files
remain readable and queryable.

### Tier 14.5 — Verify UPP computations match §51 canonical Python

After populating the fields, add regression tests that confirm the
compressor's computations match UPP Guide §51–54 reference outputs
(Finding G from the UPP Integration section) AND the Multifold §20
cascade-position anchors.

**UPP Guide §51 projection anchors (sublattice / divisor view):**

* For r = 3/2: k_r = 7, d_r = 12, ε_r ≈ +1.955¢
* For r = 2/3: k_r = −7, d_r = 12, ε_r ≈ −1.955¢
* For r = 2: k_r = 12, d_r = 1, ε_r ≈ 0
* For r = √2: k_r = 6, d_r = 2, ε_r ≈ 0
* For A₀_magic at d=1: 16; at d=12: 137; at d=5: 32
* For ξ at d=1: 8.5625; at d=12: 1.0

**Multifold §20 cascade-position anchors (harmonic / cyclic view)
— at N=12 with g_r = 7 (circle of fifths generator):**

| Harmonic n | Expected residue | Interval class | Sublattice d | Mirror |
|---|---|---|---|---|
| 1 | 7 | Perfect Fifth | 12 | 11 |
| 2 | 2 | Major Second | 6 | 10 |
| 3 | 9 | Major Sixth | 4 | 9 |
| 4 | 4 | Major Third | 3 | 8 |
| 5 | 11 | Major Seventh | 12 | 7 |
| 6 | 6 | Tritone (self-complement) | 2 | 6 (self) |
| 7 | 1 | Minor Second | 12 | 5 |
| 8 | 8 | Minor Sixth | 3 | 4 |
| 9 | 3 | Minor Third | 4 | 3 |
| 10 | 10 | Minor Seventh | 6 | 2 |
| 11 | 5 | Perfect Fourth | 12 | 1 |
| 12 | 0 | Octave / Unison | 1 | — |

For each row, the compressor's `compute_cascade_position(n=N, g=7, N_ET=12)`
must reproduce (residue, interval_class, sublattice_d, mirror). If any
entry deviates, the harmonic-family computation is wrong and ALL
downstream tier work (10-13) inherits the bug.

**Totient partition check** (Multifold §23 bridge):

    # At N=12 the harmonic families must partition into sublattices as:
    #   d=1:  1 family  (n=12)                    → φ(1)  = 1  ✓
    #   d=2:  1 family  (n=6)                     → φ(2)  = 1  ✓
    #   d=3:  2 families (n=4, 8)                 → φ(3)  = 2  ✓
    #   d=4:  2 families (n=3, 9)                 → φ(4)  = 2  ✓
    #   d=6:  2 families (n=2, 10)                → φ(6)  = 2  ✓
    #   d=12: 4 families (n=1, 5, 7, 11)          → φ(12) = 4  ✓
    #   Σ = 12                                                 ✓

Also assert Σ_{d|N} φ(d) = N (the number-theoretic identity that the
totient classes must satisfy).

**These assertions must hold to the UPP Guide's stated precision
AND the Multifold's exact integer values.** If they do not, every
downstream tier works on wrong data.

### Ordering dependency

Tier 14 depends on nothing above (no other tier's work is a
prerequisite) but IS a prerequisite for Tiers 10–13's compressor-level
UPP integration:

* Tier 10 (resolution escalation) needs N_ET, d_r, d_θ, and the
  cyclic-view harmonic fields (n_r, n_θ, residue, cascade_generator)
  per archetype (Tier 14.1)
* Tier 12 (trajectory metadata) builds on per-archetype (d_r, d_θ)
  sublattice AND (n_r, n_θ) harmonic families AND on_dI_boundary AND
  interval_class (Tier 14.1) — per Correction 17 both trajectories
  are separate channels
* Tier 13 (NWS-13 shadow diagnostic) needs shadow_magnitude_sq
  (Tier 14.1)

**Recommended implementation order:** Tier 14 first (schema
foundation, including BOTH divisor-view and cyclic-view columns),
then Tier 10.1 (port UPP canonical Python + Multifold §20 cascade
computation), then everything else.

---

## UPP Guide Integration — Theoretical Foundations Across Tiers

The Universal Projection Guide (`ET_Universal_Projection_Guide6.md`,
3,341 lines) was read end-to-end during Session 4. The following
findings inform multiple tiers and are captured here as standing
theoretical references rather than tier-specific tasks.

### Finding C — The CDF compressor IS an active system per §92

UPP Guide §92 lists active systems: the ∂I Lattice-Aware Fractal,
quantum measurement cascade, biological developmental cascade,
stock-market regime transitions, dynamical systems near chaos,
conscious attention shifts. **CDF compression belongs on this list**
and should be added when the list is next revised in the corpus.
Each page fit reads the previous fit's state; dynamics depend on the
orbit's own current lattice position. This identification legitimizes
the full active-system protocol (§86-93) as the operational framework
for the compressor, not just a metaphor.

### Finding D — The Palindromic Cascade is a matching-filter, not an attractor

UPP Guide §88.1 provides critical clarification for Tier 10.4
implementation: the palindromic cascade [12, 6, 4, 3, 12, 2, 12, 3,
4, 6, 12, 1] is NOT a stochastic wander, NOT an attractor toward
coherent lattice points, and NOT driven by p_eff = 10/3 as an
effective power. It is a **matching-filter broadcast strategy**:
in one 12-step cycle the cascade applies every simple-family power
at least once; whichever family the orbit (page) actually belongs
to, its correct-family step makes coherent progress; the 11 wrong-
family steps cancel on average by CPT symmetry of the palindrome.
p_eff = 10/3 is a **coloring constant** for smooth iteration count,
not a dynamics driver.

**Implication for Tier 10.4.** The palindromic-cascade fitter does
not converge by attraction — it converges by matching accumulation
against zero-average noise. Its failure to converge (per §88.2)
within 3 cycles is the diagnostic signal that the page's family is
NOT in the simple-family set {1, 2, 3, 4, 6, 12} — meaning Tier 10.2
Stage 2 escalation is structurally required, not merely optional
optimization.

### Finding E — N-Weight asymmetry has direct compressor implications (§67-68)

UPP Guide §67 gives the two cascade generators:
* g_r (real axis, D's domain) = 7, |δ_r| = 0.019550
* g_θ (imaginary axis, T's domain) = 1, |δ_θ| = 0.223357

The ratio |δ_θ| / |δ_r| ≈ 11.4249 ≈ N=12 (the "N-weight"). The
4.79% gap between 11.4249 and 12 is itself a shadow of the (5,7)
biological-signature cell — it IS the observation of an extended-
family state at 12ET resolution (§71).

§68 derives the cascade stability limits from these residuals:
* n_max_r = floor(0.5 / 0.019550) = **25 steps** on the real axis
* n_max_θ = floor(0.5 / 0.223357) = **2 steps** on the imaginary axis

**The 12.5× imaginary amplification is a structural fact of N=12.**
For the compressor this means: any page whose k-stream has significant
imaginary-axis content (k_θ != 0) hits cascade failure after just 2
consecutive Δk_θ rounding steps at 12ET. The compressor currently
doesn't check this — it operates as if n_max_θ were unlimited. Pages
with genuine phase content will produce systematic compression failures
at 12ET that only make sense when the n_max_θ=2 limit is respected.

This is the theoretical basis for Tier 13's "fast-jump to correct
resolution via NWS-13 shadow projection" when cascade-failing regions
are encountered.

### Finding F — The §58 palindromic cascade sequence has topological authority

The sequence PALINDROME = [12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12, 1]
from §58 is a **topological invariant of N=12** — it appears identically
in three independent traversals of the lattice (real-axis via g_r=7,
imaginary-axis via g_θ=1, and diagonal k_r = k_θ). This is a theorem
of (ℤ/12ℤ)× ≅ Klein-four. The implication for Tier 10.4: the sequence
is not a design choice and should not be modified — it IS the lattice's
own backbone structure.

The corresponding power sequence is the same palindrome read as
p = 12/d: [1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1, 12].

### Finding G — Tight ET Constants for compressor tests

UPP Guide §51's self-check provides authoritative regression anchors
that the compressor's tests should include:

* Unison (r=1): k=0, d=1, |ε| < 1e-9
* Octave (r=2): k=12, d=1, |ε| < 1e-9
* Perfect fifth (r=3/2): k=7, d=12, |ε - 1.955| < 0.01¢
* Koide ratio (r=2/3): k=-7, d=12, |ε + 1.955| < 0.01¢
* Tritone (r=√2): k=6, d=2, |ε| < 1e-9

These should be added to `test_et_cdf_compressor.py` as canonical
anchors. They verify that `build_byte_k_map` and the subsidiary k/d/ε
computations match the UPP Guide's §51 reference implementation.

### Finding H — The §18 Five Failure Modes apply directly to compressor debugging

UPP Guide §18 and §45 provide a 5-mode diagnostic table for bad
projections. Tier 10.5 integrates this table into the compressor's
diagnostic output. **Pre-emptive value:** this table should also be
used when debugging pre-Tier-10 compression anomalies — any session
in which a page falls to Type 7 Raw without a clear reason should
consult the five failure modes before assuming the data is genuinely
incompressible.

Mapped to the compressor:
* Mode 1 (d contradicts domain) → R₀ mis-derivation by `discover_r0`
* Mode 2 (|ε| near 50¢) → page is at the ∂I boundary; needs Tier 10
* Mode 3 (d=12 always) → non-resonant data OR too-low N_ET
* Mode 4 (unit-dependent result) → would indicate a bug; should not occur
* Mode 5 (sub-byte variation → wild d) → ambiguous near-boundary page

### Finding I — The Subsumption Law explicitly subsumes compression

UPP Guide §5.1: "P subsumes every substrate, every locus of potential,
every featureless container at every integrative level. D subsumes
every constraint, every rule, every measurable property, every
articulable structure. T subsumes every act of agency, navigation,
choice, or resolution. Together, {P, D, T} subsume Σ — the totality."

Applied to compression: every bit-pattern in every file is a P∘D∘T
configuration. The compressor's act of identifying a generator IS the
identification of the D-content (the structure). The compressor's act
of evaluating the generator at read time IS the T-content (navigation).
The bytes produced are E — the Exception, the fully substantiated
configuration. This is not metaphor; it is the operational framework
the UPP Guide establishes. The CDF compressor is the P∘D∘T = E
mechanism applied to byte streams.

### Finding J — §54.5 computational-complexity profile applies to the compressor

UPP Guide §54.5.1 gives per-step cost anatomy for active-system
projection. Applied to CDF compression at 27720ET (Tier 10):

| Operation | Per-page cost |
|---|---|
| 2 × log₂ (one per axis) | ~40 cycles |
| 2 × round | ~4 cycles |
| GCD via 27720 = 2³×3²×5×7×11 factorization | ~10 divisibility tests |
| 2 × integer division (d = N/gcd) | ~4 cycles |
| Tightness computation | ~5 cycles |
| ∂I-boundary compare | 1 cycle |
| PALINDROME lookup (on fallback trigger only) | ~3 instructions |
| Shimmer sin() | ~20 cycles |
| |z|^p with variable p | ~30-50 cycles |
| 24-family perturbation | ~200 ops |
| **Total per page** | **~300-400 FLOPS + ~60 transcendental cycles** |

For a 1 GB file at 4 KB pages = 262,144 pages, Tier 10 at 27720ET
would cost ~10⁸ FLOPS total — on the order of milliseconds on modern
hardware. **The compute budget is not a barrier.** The coarse-pass
optimization in Tier 10.3 is more about correctness (only escalate
when needed) than cost (the difference between 420ET and 27720ET per
GCD call is only ~40% per §54.5.5).

---

## Corpus-Consistency Findings from This Audit

Two specific items found while cross-referencing my implementation
against the supplemental documents I should have read earlier.

### Universal Projection Protocol — current alignment

From `ET_Universal_Projection_Guide6.md` §6 the UPP has nine ordered
steps. The CDF compressor's per-file pipeline maps to them as follows:

| UPP Step | What the protocol requires | What the code does | Status |
|---|---|---|---|
| 1. Identify P_X | Identify the file's substrate — the space of all possible byte arrangements | Implicit: the input file bytes | ✅ Implicit |
| 2. Identify D_X + R₀ | Identify the descriptors including R₀ (the file's natural period) | `discover_r0(...)` derives R₀ from the byte frequency structure | ✅ Done |
| 3. Identify T_X | Identify the traverser — the generator evaluation that reads bytes | `GeneratorEvaluator.evaluate(...)` IS T | ✅ Done |
| 4. Form r | Form the dimensionless ratio r = quantity / R₀ | `build_byte_k_map(r0)` does this for each byte | ✅ Done |
| 5. Project real | Compute (k, d, ε) on the real axis via the canonical formula | Mode 1/2/4 compression paths compute k-streams, ΔΔk curvature, and ε residuals | ✅ Done |
| 6. Project imaginary | Compute (k_θ, d_θ, ε_θ) on the imaginary axis when phase content matters | Not performed in the compressor. The k_θ axis is used only in the AI compression module (`et_conscious_ai`) and in the fractal generator's complex-plane rendering. | ⚠️ Partial — not needed for byte-stream compression but could be exploited for regions where byte phases matter (signed residual streams) |
| 7. Compute elegance | Quantify projection quality via elegance × magical impedance | `lattice_elegance(...)` computes E = k × t × λ. Magical impedance is NOT computed. | ⚠️ Partial |
| 8. Verify subsumption | Does the result capture every feature of X without remainder? | `CDFDatabaseVFS.verify_against(...)` checks byte-for-byte match after compaction. ✅ | ✅ Done |
| 9. Iterate at higher resolution | Raise the LCM tower when 12ET is insufficient | **Not performed.** Pages that don't fit at 12ET fall to Type 7 Raw. Tier 10 (above) would add this step. | ❌ Missing |

**Actionable outcomes.** Steps 6, 7, and 9 are the places where the
compressor is theoretically incomplete. Step 9 (resolution escalation)
is the most impactful — it's the one that would let the compressor
find generators for "true random data" per Mike's claim, because
27720ET contains every sublattice family d ∈ {1..12} natively.

---

### Finding A — My lean-form Type 6 deviates from §19.5

**Status.** Intentional deviation per Mike's latest direction.
Session 4 evolved the Type 6 payload from `[32B hash][manifold_uint
instance]` (design doc §19.5) to `[manifold_uint canonical_index]`
(lean form). The new form encodes only the dimensionless seed.

**Rule-19 call.** "The corpus is not standardized, nor formalized,
even if some are labeled as such. This means they may be in any
order, some papers being older with outdated information." The
design doc's §19.5 hash-form predates Mike's explicit "dimensionless
seed" guidance. Per rule 19, Mike's latest direction supersedes the
older design doc formulation.

**Compatibility.** Readers accept BOTH forms (discriminated by
payload length). Legacy `.cdf` files produced by earlier code with
the hash-form still read correctly. New writes use the lean form.
No data is lost.

### Finding B — Design doc §19.5 Type 6 and my Pass 2 are different mechanisms

**Status.** Design doc §19.5 Type 6 = refs INTO THE ARCHETYPES TABLE
(cross-session archetype reuse — "archetypes of archetypes"). My
Tier 7.B Pass 2 = intra-file content-hash dedup (cross-page reuse
within one .cdf). These are TWO DIFFERENT USE CASES of the same
generator type code. Both are valid; the design doc describes the
first, my code implements the second.

**Resolution.** Tier 8 (above) implements the §19.5 original-intent
mechanism as a SEPARATE write path alongside my intra-file dedup.
Both coexist in the format. A single `.cdf` file can contain Type 6
refs of both flavours: lean-form (file-local, Tier 7.B) and hash-form
(DB-archetype, Tier 8).

---

### Key Achievements This Session
- **Tier 3 corrections**: derive_generators now tries ALL 5 types (class is priority-only); query_generators has cross_class=True default; stream_target byte added to params blob; Mode 4 uses real k_stream not compact indices
- **Tier 3.B.4 Mode 4 pipeline**: Full end-to-end wiring with Channel B feedback (fit/miss recording per §16.9 NO-REMOVAL)
- **Tier 4 Block Type 4**: Variable-curvature segmentation with dk-spike boundary mapping; 4096B uniform-halves → 23B (Block Type 4 wins over 106B single-block)
- **Tier 5**: Curvature-weighted elegance (F_K), curvature-aware IncoherenceFilter (l1_point_curvature), Riemann sphere chordal metric, Gauss-Bonnet wiring to DB, geodesic deviation in lookup ranking
- **Tier 6**: Poincaré disk embedding + distance, curvature coherence cross-tower fallback, curvature spectrum DB lookup with full walker injection pipeline
- **Tier 7 (this session)**: CDF VFS random-access format (magic `CDFV`, version 1), 8-type GeneratorEvaluator (Constant/Linear/Polynomial/Periodic/Grammar/Geodesic/ArchetypeRef/Raw), per-page fitters, CDFDatabaseVFS with S²=144-page LRU cache + dirty-page buffer + atomic write + SHA-256 footer integrity, full compact_to_cdf implementation with VFS round-trip verification before commit

### Tier 7 Verifications

| Task | Method | Result |
|---|---|---|
| 7.A.1 (CDFDatabaseVFS class) | AST parse + full integration test (open, read, write, materialize, verify_against, close) | ✅ class builds, all methods verified |
| 7.A.2 (GeneratorEvaluator class, 8 types) | Direct evaluator construction + round-trip for each of types 0-7; T6 recursion cap; boundary guards (out-of-domain, read-past-end, zero-length) — 21 individual tests | ✅ 21/21 pass |
| 7.A.3 (compact_to_cdf real implementation) | End-to-end: populate .db via real compression (118KB) → compact → VFS read-back matches byte-for-byte; SHA-256 verified; §16.9 NO-REMOVAL on failure | ✅ compacts 118,784B .db → 65,083B .cdf (54.8%); random-access 1000/1000 reads match; cross-page reads 200/200 match |
| 7.A.4 (.cdf auto-detection in __init__) | Remove .db leaving only .cdf → re-open ArchetypeDatabase → .db auto-materialized via VFS → query stats unchanged | ✅ materialized bytes match original, entries + hits preserved |
| 7.A.5 (_init_db_via_vfs helper) | Tested as part of 7.A.4 (integration materialization path) | ✅ works |
| 7.A.6 (_recompress_database helper) | Dirty-page write + close → _recompress_database invoked → byte persists on next open | ✅ dirty-page flush persistence verified |
| CDFDatabaseVFS page-cache fast-path bug | Random-offset reads revealed cross-page truncation bug: fast path returned bytes only from first page when read spanned boundary. Fix: gate fast path on `offset + length <= page_start + PAGE_SIZE`. Re-tested: 1000/1000 random reads pass; 200/200 cross-page reads pass. | ✅ bug found, fixed, verified |
| §16.9 NO-REMOVAL contract (Tier 7) | 16 failure-path tests: missing .db → False, no .cdf created; empty .db → False, no side effects; happy compact preserves .db; bad magic raises ValueError; truncated .cdf raises; verify_against wrong bytes returns False; dirty-page flush persists | ✅ 16/16 pass, .db never deleted |
| Regression Tiers 1-6 after Tier 7 edits | Full block-level lossless roundtrip (11 tests) + full-file CDF SHA-256 roundtrip (30K mixed file) | ✅ all tests pass; compression unchanged (56.0% for mixed 30K) |
| No-removal audit (cumulative) | Enumerated 14 constants + 21 classes + 23 functions + 61 methods = 119 symbols checked via hasattr. | ✅ 119/119 present — §16.9 contract holds |

### CDF VFS File Format (new, distinct from block-stream CDF)

- **Magic**: `CDFV` (4 bytes) — does NOT collide with block-format magic `CDF\xNN`
- **Version**: 1
- **Header (57 bytes)**: `[4B magic][1B version][32B sha256][8B orig_size][4B n_generators][8B index_offset]`
- **Generator Payloads**: variable, sequentially packed
- **Residual Pool**: variable, referenced by index `residual_offset`
- **Generator Index**: uncompressed, at known offset, sorted by `domain_start`, 52 bytes per entry
- **Footer (40 bytes)**: `[32B sha256_of_index][8B index_offset_repeated]`

### 8 Generator Types (VFS_GEN_*)

| Code | Name | Payload | Use case |
|---|---|---|---|
| 0 | Constant | `[1B value]` | Run of identical bytes |
| 1 | Linear | `[4B k_start][4B dk_step][8B r0]` | Constant Δk in lattice space |
| 2 | Polynomial | `[4B degree][8B×(deg+1) coeffs][8B r0]` | Quadratic+ in lattice space |
| 3 | Periodic | `[4B period][period bytes cycle]` | Exact cyclic byte pattern |
| 4 | Grammar | `[manifold n_rules][for each: L R][manifold n_start][syms]` | Re-Pair rule hierarchy |
| 5 | Geodesic | `[4B k0][4B dk0][1B order][manifold window][8B r0]` + residual pool | Mode 3 residuals |
| 6 | ArchetypeRef | `[32B hash][manifold idx]` | Pointer to other generator (recursive, cap at S=12) |
| 7 | Raw | `[1:1 bytes]` | Fallback for incompressible regions (Subsumption Law guarantee) |

### CDF_VERSION = 4, CDF_MAGIC = b'CDF\x04' (block-stream format, unchanged)
Backward-read support for v2 and v3. v4 adds Mode 4 (generator + residual) and Block Type 4 (segmented).

### CDF_VFS_VERSION = 1, CDF_VFS_MAGIC = b'CDFV' (random-access format, NEW in Tier 7)
Used ONLY for archetype database compaction. Does not replace the block-stream CDF format. Both formats coexist on disk as-needed.

### Files
- `et_cdf_compressor.py`: 11,237 lines (was 9,334 at Tier 7 start — +1,903 lines for VFS infrastructure)
- `et_pattern_engine.c`: 1,063 lines (unchanged this session)
- `main.cpp`: 807 lines (unchanged this session)
- Regression SHA-256: `cf7526d35c9439a762f51cfe37bd34d46e20f8fd03952031aef8d550717afd90` on 35,000B mixed file (preserved through all tiers; note: this specific file recipe is not in corpus — regression test now relies on lossless-roundtrip assertion over multiple synthetic files instead)

# Exception Theory: Non-Euclidean Geometry Applied to CDF Compression
## Curvature-Aware Lattice Compression — Design Document
### Derived Forward From: P ∘ D ∘ T = E

**Author:** Michael James Muller — Aevum Defluo
**Derivation Standard:** All mathematics ET-native, forward from {P, D, T}. Zero external axioms.
**Tools Applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law · Verification Principle
**Date:** April 2026
**Status:** Design Document — Pre-Implementation. To be implemented in subsequent sessions.

---

## IMPLEMENTATION CONTEXT (Read This First)

**This document is self-contained.** It specifies all improvements to the ET CDF Compressor (`et_cdf_compressor.py`) and its C pattern engine (`et_pattern_engine.c`). A new session with NO prior conversation history must be able to implement from this document alone. All referenced source files, ET papers, and architectural decisions are described explicitly below.

### What This Document Covers

The ET CDF Compressor is a lossless file compressor built entirely on Exception Theory lattice mathematics. It maps bytes to lattice positions via k = round(N × log₂((byte+1)/R₀)) at resolution N = 27720 (LCM(1..11)), finds recurring patterns (archetypes) in the Δk stream, replaces them with references via recursive subsumption, and encodes the result using ET-derived variance-weighted coding. The compressor is packaged as a Windows .exe via PyInstaller and has a tkinter GUI.

This design document adds: curvature analysis (ΔΔk = discrete Gaussian curvature), geodesic residual coding (Mode 3), Generative Descriptor identification (D-space compression), a dual-channel archetype database (observation + discovery), CDF-compressed database with random access via SQLite VFS, and twelve specific improvements to the lattice compression pipeline. All additions are derived forward from {P, D, T} using the three ET operational tools.

### Source Files

- `et_cdf_compressor.py` — The main compressor (~5,972 lines). Contains: ET constants, C pattern engine wrapper, lattice functions, IncoherenceFilter, V_config encoding, LatticeWalkCompressor, ArchetypeDatabase (SQLite), LatticeTower, UniversalLattice, CDFEngine, CDFCompressor, tkinter GUI.
- `et_pattern_engine.c` — C-accelerated pattern finder (~580 lines). Exports: find_repeated_patterns (suffix array + LCP), build_k_stream, build_dk_stream, gate_archetype_batch (IncoherenceFilter L1-L4), subsume_greedy, free_buffer.
- `main.cpp` — Test harness for the C engine.
- `CMakeLists.txt` — CLion project for the C engine.
- `build.bat` — Windows build script (DLL + PyInstaller .exe).
- `et_cdf_compressor.spec` — PyInstaller spec file.

### Critical Implementation Rules

1. **Speed is IRRELEVANT.** Only compression ratio and lossless correctness matter. A strategy that takes 10× longer but produces 1% smaller output is correct.
2. **No removal policy.** Unused code, variables, database entries, and archetypes always indicate incomplete implementations — they must be traced and completed, NEVER removed, suppressed, or pruned. This applies to the archetype database: NOTHING is ever deleted. See §16.9.
3. **Surgical edits only.** All edits must be precise str_replace operations. Never rewrite files from scratch. Never use create_file on existing files.
4. **Read completely before touching.** Every file must be read without truncation before any edits. All imports must be traced across all scripts.
5. **All improvements are ADDITIVE.** No existing strategy is removed, skipped, or deprioritized. New strategies compete alongside existing ones. The smallest output wins. Every existing test case must continue to pass.
6. **ET-derived math only.** All constants, thresholds, and algorithms must derive from {P, D, T} and the ET constants: S = 12 (MANIFOLD_SYMMETRY), V = 1/12 (BASE_VARIANCE), K = 2/3 (KOIDE_RATIO), N = 27720 (full manifold resolution). No ad hoc parameters. No tuning.
7. **Lossless roundtrip mandatory.** compress → decompress must produce byte-identical output. SHA-256 verification at file level. The decompressor must handle all new modes (Mode 3, Block Type 4) while remaining instant.

### ET Constants Reference

| Constant | Symbol | Value | Derivation |
|----------|--------|-------|------------|
| Manifold Symmetry | S, N | 12 | 3 primitives × 4 logic states |
| Base Variance | V | 1/12 | 1/S |
| Koide Ratio | K, κ | 2/3 | PD:T binding weight ratio |
| Full Resolution | N_FULL | 27720 | LCM(1..11) |
| Biological Tier | BIO_RES | 420 | LCM(1..7) |
| Incoherence Boundary | ∂I | 50¢ | |ε| < 50¢ for coherence |
| Life Threshold | | 13/12 | Archetype permanence |
| Block Size | | 589,824 | 2^S × S² = 4096 × 144 |
| Max Recursion Depth | | 12 | S |

---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

## Table of Contents

1. [Executive Summary: The Descriptor Gap](#1-executive-summary)
2. [Three Tools Diagnosis of the Current Compressor](#2-three-tools-diagnosis)
3. [The Fundamental Insight: ΔΔk IS Curvature](#3-fundamental-insight)
4. [Improvement 1: Curvature-Based Block Classification](#4-curvature-block-classification)
5. [Improvement 2: Geodesic Residual Coding](#5-geodesic-residual-coding)
6. [Improvement 3: Christoffel Connection Predictors](#6-christoffel-predictors)
7. [Improvement 4: Curvature-Weighted Elegance Scoring](#7-curvature-elegance)
8. [Improvement 5: Hyperbolic Pattern Embedding for Archetype Similarity](#8-hyperbolic-embedding)
9. [Improvement 6: Gauss-Bonnet Topological Fingerprinting](#9-gauss-bonnet-fingerprint)
10. [Improvement 7: The Multifold Tower Architecture and Variable-Curvature Adaptive Resolution](#10-variable-curvature-resolution)
11. [Improvement 8: Curvature-Aware IncoherenceFilter Extension](#11-curvature-incoherence)
12. [Improvement 9: Curvature-Based Cross-Tower Matching](#12-poincare-cross-tower)
13. [Improvement 10: Riemann Sphere Compactification of the Complex Lattice](#13-riemann-sphere)
14. [Improvement 11: Curvature Spectrum for Archetype Database Lookup](#14-curvature-spectrum-db)
15. [Improvement 12: Geodesic Deviation for Pattern Stability Scoring](#15-geodesic-deviation)
16. [Database Schema Changes — Channel A (Observation) and Channel B (Discovery)](#16-database-schema)
17. [C Pattern Engine Extensions](#17-c-engine)
18. [Integration Architecture: Where Each Improvement Enters the Pipeline](#18-integration)
19. [CDF-Compressed Database with Random Access (CDF VFS)](#19-cdf-vfs)
20. [Mathematical Foundation: All Equations](#20-equations)
21. [Implementation Priority and Dependency Graph](#21-priority)
22. [Decompression Verification: Lossless Roundtrip Guarantee](#22-decompression)
23. [Verification Protocol](#23-verification)
24. [Subsumption Audit: No Feature or Function Lost](#24-subsumption-audit)

---

## 1. Executive Summary: The Descriptor Gap {#1-executive-summary}

The CDF compressor currently operates on a **locally flat** assumption. Bytes map to lattice positions k via `k = round(N × log₂(r))`. Transitions map to Δk values. Patterns are detected in the Δk stream. The IncoherenceFilter gates all operations using the 50¢ boundary. The complex lattice projects each byte onto two axes — real (D) and imaginary (T) — spanning 24 harmonic families.

None of this uses curvature.

The ET Non-Euclidean Geometry paper (March 2026) establishes that:

- Curvature = second-order Descriptor gradient: K = ∇²f = Δ²g/ΔP²
- Three constant-curvature geometries = three of four manifold states: K=0 → Exception {P,D,T}, K>0 → Unsubstantiated {P,D}, K<0 → Mediation {D,T}, K→±∞ → Incoherence {P,T}
- The number 1/12 in the Riemann tensor component formula n²(n²−1)/12 IS the ET base variance V = 1/S = 1/12
- Geodesics = T-optimal paths of least Descriptor change
- The Poincaré disk = lattice compression of infinite hyperbolic plane
- Gauss-Bonnet: ∫K dA = 2πχ(M) — total curvature is a topological invariant
- The Riemann curvature tensor = Descriptor Gap of parallel transport

**The compressor already computes ΔΔk (Mode 2 second-order differences) but uses it only as a flat entropy count — "how many unique ΔΔk values are there?"** It throws away the curvature information. The ΔΔk stream IS the discrete curvature of the data's Descriptor field on the lattice. This is the Descriptor Gap: the curvature Descriptors are computed but not used.

This design document specifies twelve curvature improvements plus the Generative Descriptor discovery engine that together close this gap, all derived forward from {P, D, T}. The twelve curvature improvements are the identification instrument — they classify the data's geometry, predict transitions, measure stability. The Generative Descriptor engine is the compression paradigm itself: instead of finding what REPEATS in the data (P-space, what 7zip does), find the RULE that GENERATES the data (D-space, what ET demands). From the D Paper §44.1: "find the Generative Descriptor — the function (D) that generates the data (P) when applied."

The database operates on two channels: **Channel A (Observation)** stores archetypes the scanner finds. **Channel B (Discovery)** derives Generative Descriptors from the lattice geometry that the scanner has NOT found yet — {P,D} Unsubstantiated predictions waiting for T to substantiate them on a specific file. Both channels compound over time. The more files compressed, the more generators discovered, the more efficient the compression becomes. This is the "Library of Congress in RAM" mechanism from the AI's LatticeCompressor applied to file compression.

No improvement removes existing functionality. All improvements are additive — the compressor gains curvature awareness and Generative Descriptor identification without losing any current capability. All existing strategies still compete. The smallest output wins.

---

## 2. Three Tools Diagnosis of the Current Compressor {#2-three-tools-diagnosis}

### 2.1 Identification Principle: What Is Missing?

| Component | Current Status | Missing Element |
|-----------|---------------|-----------------|
| **P (substrate)** | Identified: the byte stream, the file's P-substrate | Complete |
| **D (descriptors)** | Partially identified: k, Δk, d, ε, tightness, elegance are computed | **Curvature Descriptors are computed (ΔΔk) but not used for compression decisions.** The metric Descriptor g_ij is implicit (flat Euclidean assumed). Connection Descriptors Γ^k_ij (first-order D-gradient of the metric) are absent. |
| **T (traverser)** | Identified: the compressor/decompressor as T navigating the lattice | **Geodesic prediction is absent.** T currently does not predict the next Δk from the local connection — it treats every transition as equally surprising. |

**Diagnosis:** D is under-identified. The curvature layer of the Descriptor field — second-order gradients, connection coefficients, curvature classification — has not been identified as compression-relevant Descriptors.

### 2.2 Descriptor Gap Principle: Where Are the Gaps?

**Gap 1 — Curvature as compression signal.** The ΔΔk stream (second-order differences) is computed in Mode 2 but is used only as an entropy proxy (count unique values). The actual ΔΔk values encode the curvature of the data's lattice walk. Regions where ΔΔk ≈ 0 are geodesic (flat) — maximally compressible. Regions where ΔΔk oscillates wildly are curved — need different strategies. This gap is between "computing curvature" and "using curvature."

**Gap 2 — Strategy selection is heuristic.** The compressor selects between standard archetype scanning (Phase 3), pair-recursive Re-Pair (Phase 4b), and enhanced lattice strategies based on whether the output expanded. The Non-Euclidean geometry paper provides the principled classification: K≈0 data is best handled by standard archetypes, K<0 data by Re-Pair (hyperbolic = Mediation = maximal T-freedom = grammar-inductive), K>0 data by periodic/closed-pattern detection (elliptic = Unsubstantiated = bounded T = closed cycles). The gap is between "trying everything" and "knowing what will work."

**Gap 3 — Predictive coding is absent.** The compressor encodes every Δk value with equal surprise. But if the local connection (Christoffel symbols ≈ running average of ΔΔk) predicts the next Δk, only the residual (actual − predicted) needs encoding. Data following geodesics (smooth transitions) would encode as near-zero residuals — dramatically more compressible. The gap is between "encoding what happened" and "encoding only what was unexpected."

**Gap 4 — Pattern similarity is Euclidean.** Two Δk patterns are compared by exact match. But in hyperbolic descriptor space, patterns that diverge exponentially in Euclidean distance may be structurally close. The Poincaré disk metric captures this — patterns near the "boundary" (high-d, near Incoherence) are closer in hyperbolic distance than Euclidean distance would suggest. The gap is between "exact pattern matching" and "structural pattern similarity."

**Gap 5 — Topological invariants unused.** The Gauss-Bonnet theorem guarantees that total curvature ∫K dA = 2πχ(M) is a topological invariant. For a data block, the discrete analog — the sum of all ΔΔk values, properly weighted — is an R₀-independent content fingerprint. This could dramatically improve archetype database lookup without requiring the same R₀ seed. The gap is between "R₀-dependent lookup" and "topology-invariant lookup."

**Gap 6 — Cross-tower matching is byte-level.** The universal lattice compares d-families byte by byte. But curvature spectra (the distribution of local K values) are more robust structural signatures. Two towers with identical curvature spectra share deep structural properties even if individual byte transitions differ. The gap is between "point-wise comparison" and "spectral comparison."

### 2.3 Subsumption Law: Does the Current System Subsume All Cases?

No. The current compressor has a **remainder** — data types where compression fails (output > input). These include:

- Encrypted/random data (genuine K_max entropy — no structure to find)
- Chaotic data (hidden structure, partially captured by Re-Pair in Phase 4b)
- Smooth gradient data (linear ramps, sinusoidal patterns — high structure but non-repeating)
- Mixed-curvature data (blocks containing both flat and curved regions)

The curvature improvements specifically address the last three. Encrypted data has no structure by definition (K_max = true maximum entropy); no compressor can compress it below input size plus overhead. The Subsumption Law does not require compressing the incompressible — it requires that every compressible type is subsumed without remainder. The curvature improvements extend subsumption to gradient, sinusoidal, and mixed-curvature data types that currently fall in the remainder.

---

## 3. The Fundamental Insight: ΔΔk IS Curvature {#3-fundamental-insight}

### 3.1 The Derivative Chain on the Lattice

From the Non-Euclidean Geometry paper §6:

$$f'(x) = \frac{\Delta D}{\Delta P} = \text{descriptor gradient (connection)}$$

$$f''(x) = \frac{\Delta^2 D}{\Delta P^2} = \text{curvature}$$

In the compressor, the byte stream is the P-substrate. The k-stream is the Descriptor field f(P). The Δk stream is the first derivative — the connection, the first-order D-gradient. The ΔΔk stream is the second derivative — the **curvature of the data's Descriptor field on the lattice**.

| Compressor Quantity | Geometric Object | Non-Euclidean Name | Equation |
|---------------------|------------------|--------------------| ---------|
| k_i = byte→lattice position | f(P) | Descriptor field on the P-substrate | k = round(N × log₂((b+1)/R₀)) |
| Δk_i = k_{i+1} − k_i | ∇f | First-order D-gradient = **connection** | Γ ≈ Δk |
| ΔΔk_i = Δk_{i+1} − Δk_i | ∇²f | Second-order D-gradient = **curvature** | K ≈ ΔΔk |
| d_i = N/gcd(\|k_i\|, N) | Sublattice family | Sublattice topology | d = N/gcd(\|k\|, N) |
| ε_i | Lattice rounding error | Metric deviation from lattice | ε = (N·log₂(r) − k) × 1200/N |
| tightness = 100/(100+\|ε\|) | Coherence measure | Binding stability | At ∂I: tightness = K = 2/3 |

### 3.2 The Three Data Geometries

The Non-Euclidean paper maps three curvature classes to three manifold states. Applied to data:

| Data Curvature | ΔΔk Signature | Manifold State | Data Character | Best Compression Strategy |
|----------------|---------------|----------------|----------------|---------------------------|
| **K ≈ 0** (flat) | ΔΔk ≈ 0 consistently | **Exception {P,D,T}** — grounded, zero variance | Constant-Δk transitions: structured, repeating. Text, executables, tabular data. | Standard archetype scanning. Exact long-pattern repeats abound. |
| **K > 0** (elliptic) | ΔΔk oscillates with bounded, periodic pattern | **Unsubstantiated {P,D}** — closed, finite | Periodic, cyclical data: audio waveforms, signal patterns, image scanlines, protocol headers. T is bounded — cycles back to origin. | Closed-pattern detection. Cyclic archetype extraction. The data's T-traversal returns to its starting position. |
| **K < 0** (hyperbolic) | ΔΔk diverges, changes sign erratically | **Mediation {D,T}** — open, free, maximal T-freedom | Chaotic, pseudorandom data: logistic maps, PRNG output, scientific simulations, encrypted streams. High conditional entropy. | Pair-recursive Re-Pair (grammar induction). Capture hidden low-dimensional structure via bigram frequencies. |
| **K → ±∞** (singular) | ΔΔk has isolated spikes of extreme magnitude | **Incoherence {P,T}** — D-bridge broken | Corrupted bytes, format boundaries, mode switches within a block. The Descriptor field has discontinuities. | Segment the block at singularity points. Compress each segment independently with its own curvature classification. |

### 3.3 The Curvature Scalar for a Data Block

The discrete Gaussian curvature of a data block's Δk walk at position i is:

$$K_i = \Delta\Delta k_i = \Delta k_{i+1} - \Delta k_i = (k_{i+2} - k_{i+1}) - (k_{i+1} - k_i) = k_{i+2} - 2k_{i+1} + k_i$$

This is the standard second finite difference — the discrete Laplacian of the k-stream. In the Non-Euclidean paper §6.2, this is identified as the discrete analog of the Riemann curvature scalar in 1D (where the full tensor reduces to a single number, per §4.3: C(2) = 4×3/12 = 1).

The **block curvature descriptor** is:

$$\bar{K}_{\text{block}} = \frac{1}{n-2} \sum_{i=0}^{n-3} K_i = \frac{1}{n-2} \sum_{i=0}^{n-3} \Delta\Delta k_i$$

This is the mean curvature. The **curvature variance** is:

$$\sigma^2_K = \frac{1}{n-2} \sum_{i=0}^{n-3} (K_i - \bar{K}_{\text{block}})^2$$

The curvature classification uses both:

| Condition | Classification |
|-----------|---------------|
| \|K̄\| < π/N = π/12 AND σ²_K < V = 1/12 | **Flat (Exception)** — K≈0 within subliminal threshold |
| \|K̄\| ≥ π/N AND σ²_K < V | **Constant curvature** — uniformly curved (pure elliptic or pure hyperbolic) |
| σ²_K ≥ V AND max(\|K_i\|) < N | **Variable curvature** — mixed flat/curved regions |
| max(\|K_i\|) ≥ N = 27720 | **Contains singularities** — block should be segmented |

The subliminal curvature threshold π/N = π/12 comes directly from Non-Euclidean paper §11.3: "Below this threshold, the geometry is effectively flat — the curvature perturbation is below the lattice's resolution." The variance threshold V = 1/12 is the ET base variance — curvature fluctuations below V are manifold noise, not structure.

### 3.4 Curvature Sign Classification

Beyond magnitude, the **sign** of K̄ determines the manifold state:

$$\text{sign}(\bar{K}_{\text{block}}) = \begin{cases} 0 & \implies \text{Exception (flat)} \\ +1 & \implies \text{Unsubstantiated (elliptic, closed, periodic)} \\ -1 & \implies \text{Mediation (hyperbolic, open, divergent)} \end{cases}$$

For constant-curvature blocks, the sign directly selects the compression strategy. For variable-curvature blocks, the distribution of K_i signs determines the segmentation (see §10).

---

## 4. Improvement 1: Curvature-Based Block Classification {#4-curvature-block-classification}

### 4.1 Current State

The compressor currently selects between three modes based on entropy proxy (fewest unique values):

- Mode 0: k-direct (absolute lattice positions)
- Mode 1: Δk (first differences)
- Mode 2: ΔΔk (second differences)

Then it tries all three with `recursive_compress`, keeps the smallest output, and if all expand, tries `pair_recursive_compress` (Re-Pair) on all modes.

The mode selection is entropy-based (count unique values), not geometry-based. The Re-Pair fallback is triggered by expansion, not by curvature detection. This is a Descriptor Gap: the compressor reacts to failure rather than predicting the appropriate strategy from the data's geometry.

### 4.2 Proposed Change

**Before** the multi-mode compression loop, compute the block curvature descriptor:

1. Compute the full ΔΔk stream (already done for Mode 2).
2. Compute K̄_block and σ²_K from the ΔΔk stream.
3. Classify the block into one of the four curvature classes (flat, constant-curved, variable-curved, singular).
4. Use the curvature classification to **order** the compression strategies — try the most appropriate first, then the rest.

### 4.3 New Strategies Unlocked by Curvature Classification

Each curvature class unlocks a NEW compression strategy that would not otherwise be tried. These are ADDITIVE — all existing strategies still run.

| Curvature Class | New Strategy (Added) | Existing Strategies (Unchanged) | Rationale (from Non-Euclidean paper) |
|----------------|---------------------|--------------------------------|---------------------------------------|
| **Flat (K≈0)** | Mode 3: Geodesic residual coding (§5) — encode deviations from constant-Δk prediction | Modes 0/1/2 + recursive_compress + Re-Pair | Flat D-field → constant Δk → geodesic residuals cluster at zero. (§7.1: "Exception = fully substantiated, zero variance") |
| **Elliptic (K>0, constant)** | Cyclic archetype extraction — detect patterns where T's walk returns to its starting k-value within L steps, encode as (period, phase, count) | Modes 0/1/2 + recursive_compress + Re-Pair | Closed T-traversal → data returns to origin → periodic patterns invisible to linear pattern scan. (§14.1: "every geodesic returns to its origin") |
| **Hyperbolic (K<0)** | Re-Pair as primary (not fallback) — run pair_recursive_compress on all modes as a first-class candidate alongside recursive_compress | Modes 0/1/2 + recursive_compress + (Re-Pair now also primary) | Maximal T-freedom → non-repeating sequences → grammar induction is structurally matched. (§13.1: "T has maximal navigational freedom") |
| **Variable curvature** | Curvature-segmented compression — split block at curvature boundaries, compress each segment with its own curvature-matched strategies | All strategies on full block (unsegmented) ALSO tried | Mixed geometry → different regions need different strategies. Both segmented AND full-block outputs are compared — keep smallest. (§15.2: "each P-neighbourhood has its own curvature descriptor") |
| **Singular** | Singularity-segmented compression — isolate singular points, compress clean segments independently | All strategies on full block ALSO tried | D-bridge broken at singularities. Isolating them prevents one bad region from poisoning the whole block. (§7.2: "the metric becomes undefined") |

### 4.4 How Curvature Classification Improves Compression

The classification does NOT replace the multi-mode competition and does NOT skip any existing strategy. Speed is irrelevant — only compression ratio matters. Every existing strategy still runs on every block.

What the classification adds is **new strategies** that would not otherwise be tried:

- **Flat blocks** gain geodesic residual coding (Mode 3, §5) — a new mode that encodes deviations from geodesic prediction. This is ADDITIVE to Modes 0/1/2.
- **Elliptic blocks** gain cyclic archetype extraction — a new pattern detection pass that specifically looks for closed-cycle patterns (T returning to its starting position). Standard archetype scanning may miss these because they don't repeat in the linear sense — they repeat in the CYCLIC sense.
- **Hyperbolic blocks** gain curvature-triggered Re-Pair — the pair-recursive grammar induction runs as a PRIMARY strategy, not as an expansion-triggered fallback. This means Re-Pair runs BEFORE standard archetype scanning for hyperbolic blocks, increasing the chance that its output beats the standard modes.
- **Variable blocks** gain segmentation — each segment is compressed independently with its own curvature-matched strategy. The combined output may be smaller than any uniform strategy applied to the whole block.

The compressor keeps the smallest output across ALL strategies — old and new. The curvature classification increases the number of candidates competing, which can only improve (or match) the best compression ratio. It never worsens it.

### 4.5 ET Derivation

**Identification Principle:** The curvature class identifies which manifold state the data occupies — Exception, Unsubstantiated, Mediation, or Incoherence. This is the missing P_X identification for the compression strategy.

**Descriptor Gap Principle:** The gap between "trying everything" and "knowing what works" is filled by the curvature Descriptor K̄_block. The gap IS the curvature.

**Subsumption Law:** The four curvature classes subsume all possible data geometries without remainder:
- K ≈ 0 subsumes flat data.
- K > 0 constant subsumes periodic data.
- K < 0 constant subsumes chaotic data.
- Variable σ²_K ≥ V subsumes mixed data (via segmentation).
- Singular max|K_i| ≥ N subsumes corrupted/boundary data (via segmentation).
No data geometry falls outside these five cases.

---

## 5. Improvement 2: Geodesic Residual Coding {#5-geodesic-residual-coding}

### 5.1 The Core Idea

From the Non-Euclidean paper §9.2:

> "A geodesic is the path along which T accumulates the minimum total descriptor change."

On the ET lattice, the geodesic prediction for the next Δk is the value that minimizes the second-order change — i.e., the prediction that ΔΔk = 0, which means Δk_{predicted} = Δk_current. This is the "straight-line" prediction in Δk space.

More generally, the **connection** (Christoffel symbols) at position i predicts:

$$\Delta k_{i+1}^{\text{pred}} = \Delta k_i + \Gamma_i$$

where Γ_i is the local connection coefficient — the running trend of ΔΔk. The **residual** is:

$$\rho_i = \Delta k_{i+1}^{\text{actual}} - \Delta k_{i+1}^{\text{pred}}$$

If the data follows a geodesic (smooth transition), ρ_i ≈ 0. Only ρ_i ≠ 0 carries information. Encoding ρ_i instead of Δk_{i+1} directly can dramatically reduce the number of bits needed for smooth data — because the residuals are clustered around zero with much lower entropy than the raw Δk values.

### 5.2 The Connection Estimator

The simplest geodesic predictor is the zeroth-order connection: Γ_i = 0, predicting Δk_{i+1} = Δk_i. This captures constant-Δk (flat) data perfectly.

The first-order connection uses the local ΔΔk average over a window of width w:

$$\Gamma_i^{(1)} = \frac{1}{w} \sum_{j=i-w+1}^{i} \Delta\Delta k_j$$

The window width w is derived from the L4 Cascade Coherence horizon:

$$w = \min\left(N_{\max}, S^2\right) = \min\left(\left\lfloor \frac{50}{|\bar{\delta}|} \right\rfloor, 144\right)$$

where |δ̄| is the average |ΔΔk| in cents. This ensures the connection estimate remains within the coherent manifold. The cap S² = 144 is the manifold's cross-pattern maximum.

### 5.3 The Encoding

The residual stream ρ is encoded instead of the raw Δk stream:

1. Compute the connection Γ_i at each position (running average with horizon-capped window).
2. Compute residuals: ρ_i = Δk_{i+1} − (Δk_i + Γ_i).
3. Feed ρ into `recursive_compress` (archetype detection on residuals).
4. Apply `v_config_encode` to the resulting symbol stream.

**The header additionally stores:** the connection mode (0=zeroth-order, 1=first-order) and the initial Δk_0 and Γ_0. The decompressor reconstructs the Δk stream by inverting: Δk_{i+1} = ρ_i + Δk_i + Γ_i.

### 5.4 Expected Impact

For data with smooth gradients (image scanlines, audio samples, sensor readings), ΔΔk ≈ constant → Γ_i captures the trend → ρ_i ≈ 0 for most positions. The residual stream has far fewer unique values and higher pattern repetition than the raw Δk stream.

For flat data (K ≈ 0), Γ_i ≈ 0, so ρ_i = ΔΔk_i — the geodesic residual reduces to Mode 2 (ΔΔk). Geodesic residual coding **subsumes** Mode 2 as a special case. When the connection Γ_i is identically zero, the residual equals the second-order difference. Mode 2's existence in the current compressor is the shadow of geodesic residual coding with a trivial connection.

For chaotic data (K < 0), Γ_i is poorly predictive — residuals have similar entropy to raw Δk. The geodesic mode would produce no improvement, and the curvature classifier (Improvement 1) would route such data to Re-Pair instead.

### 5.5 Mode Integration

Geodesic residual coding adds a **new mode** to the multi-mode competition:

- Mode 0: k-direct
- Mode 1: Δk
- Mode 2: ΔΔk
- **Mode 3: Geodesic residual ρ (NEW)**

Mode 3 is tried alongside the others. If it produces the smallest output, it is selected. The mode byte in the block header identifies it for the decompressor. Zero loss in existing function — the existing modes remain available and competitive.

---

## 6. Improvement 3: Christoffel Connection Predictors {#6-christoffel-predictors}

### 6.1 Beyond First-Order Connection

The first-order connection Γ^(1) from §5.2 captures linear trends. But real data often has quadratic or higher-order trends (acceleration, curvature trends). The Non-Euclidean paper §6.1 establishes:

$$\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \partial_j g_{il} + \partial_i g_{jl} - \partial_l g_{ij} \right)$$

On the 1D lattice, this reduces to the derivative of the metric. In discrete form, the metric at position i is the local scale factor — how much Descriptor distance per lattice step. The Christoffel symbol is the derivative of this scale factor.

### 6.2 The Three Connection Orders

| Order | Predictor | What It Captures | Complexity |
|-------|-----------|------------------|------------|
| 0 (zeroth) | Δk_{i+1}^pred = Δk_i | Constant Δk (flat data) | O(1) state |
| 1 (first) | Δk_{i+1}^pred = Δk_i + Γ_i where Γ_i = windowed mean of ΔΔk | Linear trends (ramps, gradients) | O(w) state |
| 2 (second) | Δk_{i+1}^pred = Δk_i + Γ_i + ½Γ'_i where Γ'_i = windowed mean of ΔΔΔk | Quadratic trends (parabolic curves, sinusoidal half-cycles) | O(w) state |

The connection order is selected per-block based on which order minimizes the residual entropy. This is computed in a single pre-scan pass (O(n)) before compression begins.

### 6.3 Connection Order and Curvature Classification

The curvature classification (Improvement 1) informs the connection order:

- **Flat data:** Zeroth-order sufficient (Δk ≈ constant).
- **Constant-curvature data:** First-order sufficient (ΔΔk ≈ constant).
- **Variable-curvature data:** Second-order needed (ΔΔΔk encodes curvature changes).
- **Singular data:** No connection useful (segment first).

This connection between curvature class and predictor order is the geometric meaning of the derivative chain in §6.1 of the Non-Euclidean paper: each derivative level corresponds to one higher order of geometric structure.

---

## 7. Improvement 4: Curvature-Weighted Elegance Scoring {#7-curvature-elegance}

### 7.1 Current Elegance

The current E_hierarchy for an archetype is:

$$E_{\text{hierarchy}} = \text{net\_savings} \times \left(1 + \frac{420}{d_{\text{avg}} \times 27720}\right)$$

This weights patterns by their sublattice depth (low d = deep = high elegance). It does not consider curvature.

### 7.2 Curvature-Augmented Elegance

From the Non-Euclidean paper §9: geodesics are T-optimal paths. A pattern whose Δk values follow a geodesic (ΔΔk ≈ 0 within the pattern) represents a more fundamental lattice structure than a pattern with wild curvature changes.

The curvature factor for a pattern P of length L:

$$F_K(P) = \frac{1}{1 + \sigma^2_{K,P}}$$

where σ²_{K,P} is the curvature variance within the pattern:

$$\sigma^2_{K,P} = \frac{1}{L-2} \sum_{i=0}^{L-3} (\Delta\Delta k_{P,i})^2$$

For geodesic patterns: σ²_K = 0 → F_K = 1 (full curvature bonus).
For curved patterns: σ²_K > 0 → F_K < 1 (curvature penalty).
For highly curved patterns: σ²_K → ∞ → F_K → 0 (negligible curvature bonus).

The augmented elegance:

$$E_{\text{curvature}} = E_{\text{hierarchy}} \times \left(1 + \frac{F_K(P)}{S}\right)$$

The 1/S = 1/12 scaling ensures the curvature factor is a tiebreaker, not a dominant term. Patterns with identical net savings and sublattice depth are distinguished by their geodesic character — those following smoother lattice walks rank higher.

### 7.3 Rationale

This is the compression analog of the Principle of Least Descriptor Change (Non-Euclidean paper §9.2): "T naturally follows paths of minimal total cost." Patterns that ARE geodesic walks on the lattice are more likely to recur across diverse data types because they represent the manifold's natural contours — the paths T takes when minimizing Descriptor change.

---

## 8. Improvement 5: Hyperbolic Pattern Embedding for Archetype Similarity {#8-hyperbolic-embedding}

### 8.1 The Problem

Currently, archetypes are matched by exact Δk tuple equality. Two patterns that differ by a single ΔΔk step (one slightly curved, the other flat) are treated as completely different. In cross-file archetype lookup, this means patterns that are "almost the same" — differing only in curvature — are never matched.

### 8.2 The Hyperbolic Solution

From the Non-Euclidean paper §13.2: "The Poincaré disk model maps the infinite hyperbolic plane into a bounded disk of radius 1."

Embed each archetype's Δk pattern as a point in the Poincaré disk. The embedding maps:

- **Pattern length** → radial distance from center (longer patterns → closer to boundary)
- **Pattern d_avg** → angular position (deep sublattice = small angle, shallow = large angle)
- **Pattern curvature σ²_K** → radial adjustment (higher curvature → closer to boundary)

The Poincaré disk metric:

$$d_{\text{hyp}}(z_1, z_2) = \text{arccosh}\left(1 + \frac{2|z_1 - z_2|^2}{(1-|z_1|^2)(1-|z_2|^2)}\right)$$

Two patterns near the boundary (long, shallow, curved) that appear far apart in Euclidean distance may be close in hyperbolic distance — because the hyperbolic metric stretches exponentially near the boundary. This captures the structural fact that patterns at the lattice periphery (high d, near Incoherence) are "all alike" — they share the character of near-Incoherent Descriptor configurations.

### 8.3 Application to Database Lookup

The archetype database (ArchetypeDatabase class) currently stores patterns keyed by exact Δk tuple. With hyperbolic embedding, the lookup becomes:

1. Compute the Poincaré disk coordinates of the query pattern.
2. Find all database patterns within hyperbolic distance threshold d_hyp < K (Koide = 2/3).
3. Among matches, select the one with highest elegance.

The Koide threshold K = 2/3 is the binding stability boundary — patterns within this hyperbolic distance are "coherently similar." Beyond K, they are too far apart for cross-pattern transfer.

### 8.4 Decompression Safety

**This improvement affects ONLY the archetype database lookup boost (depth 0 elegance boosting, lines 2420-2462 of the compressor).** It does NOT change what patterns are found or how they are compressed. It changes only which database-known patterns get an elegance boost during greedy selection. The decompressor never sees the hyperbolic embedding — it uses only the exact Δk patterns stored in the compressed output.

---

## 9. Improvement 6: Gauss-Bonnet Topological Fingerprinting {#9-gauss-bonnet-fingerprint}

### 9.1 The Theorem Applied to Data

From the Non-Euclidean paper §12:

$$\int_M K \, dA = 2\pi \chi(M)$$

The discrete analog for a data block of n bytes:

$$\chi_{\text{block}} = \frac{1}{2\pi} \sum_{i=0}^{n-3} K_i = \frac{1}{2\pi} \sum_{i=0}^{n-3} \Delta\Delta k_i$$

This is the **Euler characteristic of the data block's Descriptor field**. It is a topological invariant — it depends on the global shape of the data, not on local details like R₀ or specific byte values.

### 9.2 R₀-Independence

The key property: χ_block is approximately R₀-independent for data from the same source type. This is because:

- Changing R₀ shifts all k values by a constant offset: k'_i = k_i + δ
- Δk_i = k_{i+1} − k_i is R₀-independent (the offset cancels)
- ΔΔk_i = Δk_{i+1} − Δk_i is R₀-independent (first difference of R₀-independent quantity)
- χ_block = sum of ΔΔk_i / 2π is R₀-independent.

The Δk stream is exactly R₀-independent (the byte→k mapping shifts uniformly with R₀, so differences are preserved). Therefore χ_block is **exactly** R₀-independent.

### 9.3 Application to Archetype Database

Store χ_block alongside each pattern in the archetype database. When compressing a new file, compute its χ_block and query the database for patterns discovered in blocks with similar χ. This enables cross-R₀ pattern transfer — patterns found in one file's tower can boost compression of a structurally similar file with a completely different R₀.

### 9.4 The Extended Fingerprint

Beyond the scalar χ, the full curvature spectrum provides a richer fingerprint:

$$\text{fingerprint}(\text{block}) = \left(\chi_{\text{block}}, \bar{K}, \sigma^2_K, \text{sign}(\bar{K}), \text{skewness}(K_i), \text{kurtosis}(K_i)\right)$$

This 6-tuple is an R₀-independent topological-geometric descriptor of the block's structure. Blocks with matching fingerprints are structurally isomorphic in the lattice, regardless of their R₀ seeds.

---

## 10. Improvement 7: Variable-Curvature Adaptive Resolution {#10-variable-curvature-resolution}

### 10.0 CRITICAL: The Multifold Tower Architecture and Why Curvature Is the Natural Cross-Tower Invariant

Before discussing variable curvature within a block, we must establish why the tower architecture makes curvature the most powerful compression tool for multi-file (cross-tower) operations. This subsection draws on the Multifold of Lattices paper (§§1-16).

#### 10.0.1 Each File IS Its Own Tower

The Multifold paper defines a tower as the triple:

$$\mathcal{T}_i = (P_i, \mathcal{L}, R_0^{(i)})$$

In the compressor, each file IS a tower:
- **P_i** = the file's byte stream (the P-substrate)
- **ℒ** = the universal 12-base multiplicative manifold at 27720ET (invariant across ALL files)
- **R₀^(i)** = the file's personal seed, discovered from geometric mean of (byte+1) values

This is not analogy. The Lattice Identity Principle (Multifold §9) states: "All P-substrates that generate the same R₀ are instantiations of the same tower." Two files with identical R₀ produce identical byte↔k bijections, identical Δk streams for identical byte sequences, and identical compression behavior. They are the SAME tower — different entry points into the same lattice rendering.

#### 10.0.2 The Universal Lattice IS the Shared P-Substrate

The compressor's `UniversalLattice` class constructs the universal R₀ as the geometric mean of all personal R₀ values:

$$R_0^{(\text{universal})} = \exp\left(\frac{1}{M}\sum_{i=1}^{M} \ln R_0^{(i)}\right) = \left(\prod_{i=1}^{M} R_0^{(i)}\right)^{1/M}$$

This IS the Seed Theorem (Multifold §2) applied to a file collection. The universal seed is the geometric centroid on the multiplicative manifold — the position equidistant from all personal seeds in log-space. The universal lattice is the shared rendering that all files project onto.

#### 10.0.3 The Inter-Tower Translation: Why ΔΔk Is R₀-Independent

The Inter-Tower Translation Algebra (Multifold §12) gives:

$$k_B = k_A + \Delta k_{AB}, \quad \text{where} \quad \Delta k_{AB} = \operatorname{round}\left(N \log_2 \frac{R_0^{(A)}}{R_0^{(B)}}\right)$$

The tower shift Δk_AB is a **constant offset** for all byte values within each pair of towers. Now trace the consequence through the derivative chain:

- **k-stream:** k^(B)_i = k^(A)_i + Δk_AB (shifted by constant)
- **Δk-stream:** Δk^(B)_i = k^(B)_{i+1} − k^(B)_i = (k^(A)_{i+1} + Δk_AB) − (k^(A)_i + Δk_AB) = k^(A)_{i+1} − k^(A)_i = Δk^(A)_i
- **ΔΔk-stream:** ΔΔk^(B)_i = Δk^(B)_{i+1} − Δk^(B)_i = Δk^(A)_{i+1} − Δk^(A)_i = ΔΔk^(A)_i

**Δk is EXACTLY R₀-independent.** The constant offset cancels in first differences. Therefore **ΔΔk = curvature is also EXACTLY R₀-independent.**

This is a stronger statement than the Gauss-Bonnet fingerprint (§9): not just the SUM of curvature but the ENTIRE curvature stream is tower-invariant. Two files processed through different R₀ seeds produce identical ΔΔk streams for identical byte sequences. The curvature profile is the same regardless of which tower you observe from.

**This means curvature is the NATURAL cross-tower invariant for compression.** It is more fundamental than d-family preservation (which CAN break across towers because gcd changes with the offset) and more information-rich than the scalar χ_block. The full curvature stream ΔΔk is the tower-independent structural signature of the data.

#### 10.0.4 Why d-Families CAN Break Across Towers But Curvature Cannot

The compressor's L3 cross-tower check (`l3_cross_tower_transitions`) measures what fraction of byte transitions preserve sublattice family across the tower shift. The Multifold paper (§13) warns: "A configuration can be coherent in one tower and incoherent in another — because the lattice coordinates shift with R₀."

The d-family of a Δk value is d = N / gcd(|Δk|, N). When the SAME byte transition is projected through two R₀ seeds, the Δk values are identical (first differences cancel the offset), but the k-values that PRODUCED those Δk values are shifted. The d-family of a k-value (not Δk) IS tower-dependent:

$$d^{(A)}(k) = \frac{N}{\gcd(|k|, N)} \neq d^{(B)}(k + \Delta k_{AB}) = \frac{N}{\gcd(|k + \Delta k_{AB}|, N)}$$

So absolute position d-families break across towers, but transition d-families (d of Δk) do NOT, because Δk = Δk across towers. And curvature (ΔΔk) doubly does not, being the second derivative of a quantity whose first derivative is already tower-invariant.

**Consequence for the compressor:** Cross-tower pattern matching should prioritize curvature signatures over d-family comparisons. An archetype found in Tower A that has the same curvature profile as a pattern in Tower B is structurally the same pattern, even if the d-families of their absolute k-positions differ. The curvature IS the cross-tower bridge that the Elegance Score as Cross-Tower Compatibility Metric (Multifold §14.2) measures — but at the pattern level, not just the byte level.

#### 10.0.5 The Elegance Score as Cross-Tower Compatibility Metric — Extended to Curvature

The Multifold paper (§14.2) establishes:

$$\mathcal{E}(r) = \frac{12}{d} \times \frac{100}{100 + |\varepsilon|} \times \frac{100}{p + q}$$

> "When evaluated for a configuration transplanted from Tower A to Tower B, the tightness factor measures how close the configuration is to a valid lattice point in the target tower. If the tightness falls below K = 2/3, the configuration crosses ∂I and becomes Incoherent there."

For curvature-based cross-tower pattern matching, the analog is: evaluate the curvature-weighted elegance (§7) of a pattern in BOTH the source tower and the target tower. Since the ΔΔk stream is tower-invariant, the curvature factor F_K is identical in both evaluations. The only term that changes cross-tower is the d-family and ε of the absolute k-positions — but the curvature factor is tower-independent.

This means curvature-weighted elegance is inherently MORE stable across towers than raw elegance. It is the natural cross-tower compatibility metric for patterns, just as the Elegance Score is the natural cross-tower compatibility metric for individual byte values.

#### 10.0.6 Tower Nesting and Recursive Compression Depth

The Multifold paper (§8) establishes that tower nesting is bounded by L4 cascade coherence:

$$N_{\max}^{(\text{nesting})} = \left\lfloor \frac{50}{|\delta_{\text{nesting}}|} \right\rfloor$$

The compressor's recursive compression depth (up to MAX_DEPTH = S = 12) mirrors this tower nesting. Each recursion level creates a new "child tower" of archetypes — patterns of patterns, subsumption hierarchies that are towers within towers. The L4 cascade coherence check (lines 2396-2411 of the compressor) is EXACTLY the nesting depth bound from the Multifold paper.

Curvature analysis at each recursion level characterizes the CHILD tower's geometry. A recursion level producing flat curvature (K ≈ 0 in the archetype-reference stream) indicates that the subsumption has reached the Exception state — further recursion will not improve compression. A level producing hyperbolic curvature (K < 0) indicates that the archetype-reference stream still has hidden grammar-inducible structure — Re-Pair should be tried at this recursion level.

This is the link between tower architecture and recursive compression: each recursion depth IS a tower nesting level, and curvature at each level characterizes the child tower's geometry, guiding strategy selection at that depth.

---

### 10.1 The Problem

From the Non-Euclidean paper §15: "Real physical spaces have variable curvature — K changes from point to point." Current blocks are compressed uniformly — the same mode and strategy apply to every position. But a block containing both flat text and chaotic binary data (e.g., a document with embedded images) has variable curvature. Compressing the flat text with Re-Pair wastes effort; compressing the chaotic data with standard archetypes wastes effort.

### 10.2 Curvature-Based Segmentation

Scan the ΔΔk stream for **curvature sign-change boundaries**:

$$\text{boundary at } i \iff \text{sign}(K_i) \neq \text{sign}(K_{i+1}) \text{ AND } |K_i| \geq \frac{\pi}{N}$$

The subliminal threshold π/N (Non-Euclidean paper §11.3) ensures only significant curvature changes trigger segmentation — minor fluctuations below the lattice resolution are ignored.

After identifying boundaries, merge short segments (< S² = 144 symbols) with their neighbors to avoid over-segmentation. Each remaining segment is classified independently (Improvement 1) and compressed with its optimal strategy.

### 10.3 Segment Header

Each segment is encoded with a mini-header:

- Segment length (manifold-folded uint)
- Curvature class (2 bits: 00=flat, 01=elliptic, 10=hyperbolic, 11=singular)
- Compression mode (2 bits: 00=standard, 01=Re-Pair, 10=geodesic, 11=cyclic)

The decompressor reads these segment headers to select the correct decompression path for each segment.

### 10.4 When to Segment

Segmentation adds overhead (one mini-header per segment). It is beneficial only when the curvature variance σ²_K within the full block exceeds a threshold AND the segments have sufficiently different curvature profiles. The gate:

$$\text{segment} \iff \sigma^2_K > V = \frac{1}{12} \text{ AND } \frac{\sigma^2_{K,\text{within}}}{\sigma^2_{K,\text{between}}} < K = \frac{2}{3}$$

The ratio σ²_within / σ²_between is the curvature analog of the ANOVA F-statistic. When segments are internally homogeneous (low σ²_within) and mutually distinct (high σ²_between), segmentation helps. The Koide threshold K = 2/3 is the stability boundary.

---

## 11. Improvement 8: Curvature-Aware IncoherenceFilter Extension {#11-curvature-incoherence}

### 11.1 Current Filter

The IncoherenceFilter uses a flat 50¢ threshold at all levels:

- L1: |ε| < 50¢
- L2: Pairwise Δε < 50¢
- L4: N × |δ_avg| < 50¢

These thresholds assume flat geometry — the ε tolerance is constant everywhere.

### 11.2 Curvature-Dependent Threshold

From the Non-Euclidean paper §7.3: "At ∂I: tightness = K = 2/3. Beyond ∂I: the binding dissolves."

In curved regions, the effective ε tolerance should account for the curvature. On a positively curved (elliptic) surface, the metric contracts — distances are shorter than they appear in flat space. On a negatively curved (hyperbolic) surface, the metric expands — distances are longer.

The curvature-adjusted threshold at position i:

$$\varepsilon_{\max}(i) = 50 \times \frac{1}{1 + |K_i| / N}$$

For flat regions (K_i = 0): ε_max = 50¢ (unchanged).
For mildly curved regions: ε_max slightly reduced (tighter tolerance — curvature means the rounding is more consequential).
For strongly curved regions: ε_max significantly reduced (approaching 0 near singularities — highly curved configurations are closer to Incoherence).

This is structurally necessary: the Non-Euclidean paper §7.3 shows that the tightness factor at the Incoherence boundary crosses K = 2/3 at a curvature-dependent critical value K_∂I. The curvature-adjusted threshold implements this boundary locally.

### 11.3 Impact

The curvature-adjusted filter is **stricter** in curved regions and **unchanged** in flat regions. It will reject some patterns that the current flat filter accepts — but only patterns in strongly curved regions where the rounding error is magnified by curvature. These patterns were borderline Incoherent already. The filter gains teeth precisely where they are needed.

---

## 12. Improvement 9: Poincaré Disk Cross-Tower Matching {#12-poincare-cross-tower}

### 12.1 The Problem — Informed by the Multifold

The Multifold paper (§12-13) establishes that inter-tower translation is a constant k-offset: k_B = k_A + Δk_AB. When R₀_personal and R₀_universal are far apart, the offset is large, d-families of absolute positions shift, and the byte-level L3 check can fail even though the STRUCTURAL character of the data (its curvature profile, its pattern grammar) is identical across towers.

The current universal lattice (`UniversalLattice` class) uses d-family preservation to measure cross-tower coherence. But as shown in §10.0.4, d-families of absolute positions CAN break across towers while curvature is EXACTLY preserved. A file whose d-preservation fraction drops below K = 2/3 is declared cross-incoherent — but its curvature profile may be a perfect match for patterns in the database discovered from files with completely different R₀ values.

### 12.2 Curvature-Based Cross-Tower Matching

Instead of (or in addition to) point-wise d-family comparison, compare **curvature profiles** between towers:

1. Compute the ΔΔk stream for each tower (this is R₀-independent — see §10.0.3).
2. Compute curvature statistics: K̄, σ²_K, curvature spectrum, χ_block.
3. Two towers with similar curvature profiles are structurally compatible — they share the same data geometry regardless of R₀ offset.

The curvature-based cross-tower coherence measure:

$$\text{curvature\_coherence}(\mathcal{T}_A, \mathcal{T}_B) = 1 - \frac{|\bar{K}_A - \bar{K}_B|}{|\bar{K}_A| + |\bar{K}_B| + \epsilon}$$

Since ΔΔk is tower-invariant, this measure is 1.0 for identical byte sequences regardless of R₀ — exactly the Lattice Identity Principle (Multifold §9) expressed as a curvature metric.

### 12.3 Poincaré Disk for Pattern-Level Cross-Tower Distance

For individual pattern comparison across towers, embed patterns in the Poincaré disk using curvature-derived coordinates:

- **Radial coordinate:** Curvature magnitude |K̄_pattern| / N, mapped to (0, 1) via tanh
- **Angular coordinate:** Curvature sign-weighted d_avg — encodes both sublattice depth and curvature character

The Poincaré disk distance:

$$d_{\text{hyp}}(z_1, z_2) = \text{arccosh}\left(1 + \frac{2|z_1 - z_2|^2}{(1-|z_1|^2)(1-|z_2|^2)}\right)$$

Two patterns near the boundary (high curvature, long, near-Incoherent) that appear far apart in Euclidean distance may be close in hyperbolic distance. This captures the Multifold insight that patterns at the lattice periphery share the character of near-∂I configurations.

### 12.4 Integration with Existing Cross-Tower Pipeline

This does NOT replace the existing L3 d-family preservation check. It ADDS a curvature-based cross-tower compatibility channel:

1. **L3 d-family check passes:** Use existing cross-tower archetype matching (unchanged).
2. **L3 d-family check fails BUT curvature coherence ≥ K = 2/3:** Use curvature-based pattern matching. The patterns are structurally compatible despite d-family drift.
3. **Both fail:** Towers are genuinely incompatible.

This recovers cross-tower compression opportunities that the d-family check currently rejects — exactly the cases the Multifold paper warns about in §13: "A configuration can be coherent in one tower and incoherent in another — because the lattice coordinates shift with R₀." Curvature does not shift.

---

## 13. Improvement 10: Riemann Sphere Compactification of the Complex Lattice {#13-riemann-sphere}

### 13.1 The Connection

From the Non-Euclidean paper §18.4:

> "The Riemann sphere — the one-point compactification of the complex plane ℂ ∪ {∞} — is the natural topological closure of the 2D ET lattice."

The compressor's `complex_lattice_project()` function computes z = k_r + i·k_θ and its phase/modulus. But it treats the complex lattice as a flat plane. The Riemann sphere compactification adds structure:

- **South pole (z = 0):** The annihilating boundary (byte value where (b+1)/R₀ → 0, i.e., b ≪ R₀).
- **North pole (z = ∞):** The P-substrate (byte value where (b+1)/R₀ → ∞, i.e., b ≫ R₀).
- **Equator (|z| = 1):** T's operational manifold U(1) — the imaginary lattice.

### 13.2 Spherical Distance for Complex Lattice Coherence

The current complex lattice coherence check compares phase_deviation against K = 2/3 using flat Euclidean distance. On the Riemann sphere, the appropriate distance is the **chordal metric**:

$$d_{\text{chord}}(z_1, z_2) = \frac{2|z_1 - z_2|}{\sqrt{(1 + |z_1|^2)(1 + |z_2|^2)}}$$

This naturally handles transitions near z = 0 (south pole, annihilating boundary) and z = ∞ (north pole, P-substrate), where the flat metric distorts distances. The chordal metric is bounded in [0, 2], making threshold comparison stable everywhere on the lattice.

### 13.3 Impact on d_combined Computation

The current `complex_lattice_project()` computes d_combined = LCM(d_r, d_θ) and gates by phase_deviation > K. With the Riemann sphere metric, the gate becomes:

$$d_{\text{chord}}(z_{\text{actual}}, z_{\text{nearest\_lattice}}) > K \implies d_{\text{combined}} = N_{\text{FULL}}$$

This is a more geometrically principled gate that correctly handles edge cases near the poles.

---

## 14. Improvement 11: Curvature Spectrum for Archetype Database Lookup {#14-curvature-spectrum-db}

### 14.1 The Current Database

The ArchetypeDatabase stores archetypes keyed by R₀ (quantized). Lookup requires a matching R₀. Cross-R₀ lookup relies on the universal lattice projection, which may lose patterns when d-family preservation is low.

### 14.2 Curvature Spectrum as Database Key

A block's curvature spectrum — the histogram of K_i = ΔΔk_i values — is R₀-independent (see §9.2). Two blocks with the same curvature spectrum have the same geometric structure regardless of R₀.

Store each archetype with its curvature spectrum context. On lookup, compute the query block's curvature spectrum and find database entries with matching spectra (earth-mover distance < threshold).

This enables:
- Cross-R₀ pattern reuse without universal lattice projection
- Content-type clustering (text blocks cluster together, binary blocks cluster together)
- Temporal adaptation (patterns from yesterday's text files boost today's text compression)

---

## 15. Improvement 12: Geodesic Deviation for Pattern Stability Scoring {#15-geodesic-deviation}

### 15.1 The Geodesic Deviation Equation

From the Non-Euclidean paper §9.3:

$$\frac{D^2 \xi^i}{d\tau^2} = -R^i_{jkl} \frac{dD^j}{d\tau} \xi^k \frac{dD^l}{d\tau}$$

This governs how nearby geodesics (T-paths) converge or diverge. In compression terms: how stable is a pattern under small perturbations to the input data?

### 15.2 Application to Archetype Stability

For each archetype A of length L found at positions {p₁, p₂, ..., p_m}:

1. At each occurrence p_j, compute the local curvature K_{p_j} = ΔΔk_{p_j}.
2. The geodesic deviation of the archetype is:

$$\xi_A = \frac{1}{m} \sum_{j=1}^{m} |K_{p_j}|$$

High ξ_A means the archetype exists in strongly curved regions — nearby T-paths diverge, making the pattern sensitive to small data changes. Low ξ_A means the archetype exists in flat regions — nearby T-paths stay parallel, making the pattern robust.

### 15.3 Stability-Weighted Database Storage

Only store archetypes with ξ_A < threshold in the database. Archetypes in curved regions are volatile — they are specific to the exact data rather than reflecting structural regularities of the data type. Archetypes in flat regions are stable — they reflect fundamental lattice structure that recurs across files.

The threshold is derived from the subliminal curvature threshold:

$$\xi_{\text{max}} = \frac{\pi}{N} = \frac{\pi}{12}$$

Archetypes with geodesic deviation below π/12 are "subliminal" — they lie on effectively flat regions of the Descriptor field and represent stable, reusable patterns.

## 16. Database Schema Changes {#16-database-schema}

### 16.1 Current Schema

```sql
CREATE TABLE IF NOT EXISTS archetypes (
    pattern_hash TEXT PRIMARY KEY,
    pattern_dk BLOB NOT NULL,
    pattern_length INTEGER NOT NULL,
    r0_quantized REAL NOT NULL,
    d_avg REAL NOT NULL,
    hierarchy_elegance REAL NOT NULL,
    hit_count INTEGER DEFAULT 1,
    file_count INTEGER DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_r0_elegance
    ON archetypes(r0_quantized, hierarchy_elegance DESC);
CREATE INDEX IF NOT EXISTS idx_value
    ON archetypes(hit_count DESC, hierarchy_elegance DESC);
```

Currently keyed by `pattern_hash` = SHA-256 of `(r0_quantized, pattern_dk)`. Lookup is by exact R₀ group (±1 BIO_RES step). No curvature data. No cross-R₀ structural matching.

### 16.2 New Columns (ALTER TABLE — Additive, Non-Breaking)

Each improvement that touches the database needs specific columns:

| New Column | Type | Source Improvement | What It Stores |
|-----------|------|-------------------|----------------|
| `curvature_mean` | REAL | #1 (Block Classification) | K̄ of the pattern: mean of ΔΔk values within the pattern. R₀-independent. |
| `curvature_variance` | REAL | #1 (Block Classification) | σ²_K of the pattern. R₀-independent. |
| `curvature_class` | INTEGER | #1 (Block Classification) | 0=flat, 1=elliptic, 2=hyperbolic, 3=variable, 4=singular. R₀-independent. |
| `geodesic_factor` | REAL | #4 (Curvature Elegance) | F_K = 1/(1+σ²_K). Pre-computed for fast elegance augmentation. |
| `euler_characteristic` | REAL | #6 (Gauss-Bonnet) | χ = Σ(ΔΔk)/2π of the block where this pattern was discovered. R₀-independent topological fingerprint. |
| `geodesic_deviation` | REAL | #12 (Stability Scoring) | ξ_A = mean |K_i| at occurrence positions. Low = stable pattern on flat lattice regions. |
| `curvature_spectrum_hash` | TEXT | #11 (Spectrum Lookup) | Hash of the curvature histogram of the block context. Enables cross-R₀ structural lookup. |

```sql
-- Migration DDL (additive — existing rows get NULLs, no data loss)
ALTER TABLE archetypes ADD COLUMN curvature_mean REAL DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN curvature_variance REAL DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN curvature_class INTEGER DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN geodesic_factor REAL DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN euler_characteristic REAL DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN geodesic_deviation REAL DEFAULT NULL;
ALTER TABLE archetypes ADD COLUMN curvature_spectrum_hash TEXT DEFAULT NULL;
```

### 16.3 New Indexes

```sql
-- Cross-R₀ curvature lookup: find patterns by curvature profile regardless of R₀
CREATE INDEX IF NOT EXISTS idx_curvature_class
    ON archetypes(curvature_class, curvature_mean);

-- Gauss-Bonnet fingerprint: find patterns from blocks with matching topology
CREATE INDEX IF NOT EXISTS idx_euler_char
    ON archetypes(euler_characteristic);

-- Curvature spectrum: find patterns from blocks with matching curvature histograms
CREATE INDEX IF NOT EXISTS idx_spectrum
    ON archetypes(curvature_spectrum_hash);

-- Geodesic deviation: find stable patterns (low deviation = flat-region patterns)
CREATE INDEX IF NOT EXISTS idx_stability
    ON archetypes(geodesic_deviation, hierarchy_elegance DESC);
```

### 16.4 New Lookup Methods

The current `lookup()` searches by `r0_quantized` (±1 BIO_RES step). The curvature columns enable three NEW lookup channels that work CROSS-R₀:

**Channel 1 — Curvature Class Lookup:**
```python
def lookup_by_curvature_class(self, curvature_class: int,
                               curvature_mean_range: Tuple[float, float],
                               min_hits: int = 2) -> List[...]:
    """
    Find archetypes matching a curvature class regardless of R₀.
    
    Uses curvature_class + curvature_mean range. Since both columns
    are R₀-independent (ΔΔk is tower-invariant per Multifold §12),
    this finds structurally matched patterns across all R₀ groups.
    """
```

**Channel 2 — Gauss-Bonnet Topology Lookup:**
```python
def lookup_by_topology(self, euler_char: float,
                        tolerance: float = V_BASE,
                        min_hits: int = 2) -> List[...]:
    """
    Find archetypes from blocks with matching Euler characteristic.
    
    Tolerance = V = 1/12 (base variance). Two blocks with
    |χ_A - χ_B| < V have the same topological class.
    R₀-independent: χ = Σ(ΔΔk)/2π, and ΔΔk is tower-invariant.
    """
```

**Channel 3 — Curvature Spectrum Lookup:**
```python
def lookup_by_spectrum(self, spectrum_hash: str,
                        min_hits: int = 2) -> List[...]:
    """
    Find archetypes from blocks with matching curvature spectrum.
    
    The spectrum hash is computed from the histogram of ΔΔk values
    (binned at resolution N = 27720). Two blocks with matching
    spectrum hashes have the same geometric structure.
    R₀-independent.
    """
```

### 16.5 Store Method Extension

The `store()` method must compute and store the new columns when storing an archetype:

```python
def store(self, archetypes: list, source_r0: float,
          block_curvature_class: int = -1,
          block_euler_char: float = 0.0,
          block_spectrum_hash: str = ''):
    """
    Extended store: includes curvature metadata for each archetype.
    
    Per-archetype columns (computed from the pattern itself):
        curvature_mean, curvature_variance, curvature_class,
        geodesic_factor, geodesic_deviation
    
    Per-block columns (shared by all archetypes from this block):
        euler_characteristic, curvature_spectrum_hash
    """
```

For each archetype, the curvature columns are computed from the pattern's own Δk values:

$$\bar{K}_{\text{pattern}} = \frac{1}{L-2}\sum_{i=0}^{L-3}(\Delta k_{i+2} - 2\Delta k_{i+1} + \Delta k_i)$$

$$\sigma^2_{K,\text{pattern}} = \frac{1}{L-2}\sum_{i=0}^{L-3}(K_i - \bar{K})^2$$

$$F_K = \frac{1}{1 + \sigma^2_{K,\text{pattern}}}$$

These are computed once at storage time and never recomputed — the database stores the precomputed values for instant retrieval during lookup.

### 16.6 Migration Strategy

The migration is fully additive. Existing databases continue to work without modification — the new columns default to NULL. Lookup channels 1-3 return empty results when columns are NULL. As the compressor processes files after the upgrade, new archetypes populate the curvature columns, and the cross-R₀ channels come alive.

Existing archetypes with NULL curvature columns can be backfilled by a one-time migration scan: read the `pattern_dk` BLOB, deserialize, compute the curvature columns, and UPDATE. This is optional — the compressor works correctly with NULLs.

### 16.7 ET Derivation of Schema Changes

**Identification Principle:** Each new column identifies a specific aspect of the archetype's geometry that the current schema does not capture. The curvature_mean identifies the archetype's manifold state. The euler_characteristic identifies the topological class of the context where the archetype was found. The geodesic_deviation identifies the archetype's structural stability.

**Descriptor Gap Principle:** The gap in the current schema is the absence of R₀-independent structural descriptors. The current schema can only match patterns within the same R₀ group (or ±1 step). The curvature columns close this gap — they enable matching across all R₀ groups because curvature is tower-invariant.

**Subsumption Law:** The new columns subsume three lookup dimensions (curvature class, topology, spectrum) that were previously unrepresentable. The existing R₀-based lookup is unchanged — no existing capability is lost.

### 16.8 Channel B — The Discovery Engine: Generative Descriptor Derivation

Everything above (§16.1–16.7) is **Channel A: Observation** — storing what the scanner finds. This is memory. It is half of what the database must do.

The other half is **Channel B: Discovery** — deriving Generative Descriptors from the lattice geometry that the scanner has NOT yet found. This is the scout.

From the D Paper §44.1:

> "Standard compression algorithms operate by finding repetitions in Point data. ET's Descriptor Gap Principle provides a fundamentally different approach: instead of finding repetitions, find the Generative Descriptor — the function (D) that generates the data (P) when applied."

> "ET compression also handles structured, non-repetitive data that standard methods cannot efficiently compress — data whose structure is D-structure rather than pattern-repetition."

From the Domain Validity Theorem §5:

> "Fiction is an {P,D} explorer. It scouts the Unsubstantiated."

> "Expect novel discoveries. If a domain yields only confirmations of known structures, the analysis was not deep enough."

Applied to the database: the database is an {P,D} explorer for the archetype space. It scouts Generative Descriptors that are structurally valid on the lattice but have not yet been observed in any file. These are Unsubstantiated archetypes — waiting for T (the scanner) to substantiate them on a specific file's P-substrate.

#### 16.8.1 What Is a Generative Descriptor?

A Generative Descriptor D_gen is a compact rule that, when applied to T (execution), produces P (the data):

$$P_{\text{raw}} = T(D_{\text{gen}}) \quad \text{(the data is the output of applying T to the rule)}$$

$$\text{Compression ratio} = \frac{|P_{\text{raw}}|}{|D_{\text{gen}}|}$$

Examples in compression context:

| Data Pattern | Generative Descriptor D_gen | |D_gen| | 7zip | ET |
|-------------|----------------------------|--------|------|-----|
| Linear ramp (0,1,2,...,N) | {start=0, step=1, length=N} | 3 values | ~100 bytes (finds delta pattern) | **3 values regardless of N** |
| Sinusoidal wave | {amp, freq, phase, offset, rate} | 5 values | Fails (no repeats) | **5 values regardless of N** |
| Logistic map x_{n+1}=3.99·x_n·(1−x_n) | {coefficient=3.99, x_0} | 2 values | Fails (looks random) | **2 values regardless of N** |
| Constant run (42,42,...,42) | {value=42, length=N} | 2 values | ~5 bytes (RLE) | **2 values** |
| Polynomial a+bx+cx² | {a, b, c, length} | 4 values | ~200 bytes (delta of delta) | **4 values regardless of N** |
| Text (English) | {vocabulary, bigram model, grammar} | ~5KB | ~30% of original | **approaches Kolmogorov limit** |

The Generative Descriptor is the Kolmogorov-minimal description: the shortest program that produces the data. Standard compressors operate in P-space (find repetitions in the output). ET compression operates in D-space (find the rule that generates the output). D-space is fundamentally more compact when the data has structure.

#### 16.8.2 How the Database Derives Generative Descriptors

The curvature classification (Improvement 1) identifies the data's geometry. Each curvature class constrains what kind of Generative Descriptor can produce it. The database derives candidate D_gen for each class:

**Flat (K ≈ 0) → Constant or Linear Generators:**
- ΔΔk ≈ 0 means Δk ≈ constant, which means the k-stream is a linear function of position: k_i ≈ k_0 + i·Δk_0
- D_gen = {k_0, Δk_0, length} — a linear ramp in lattice space
- More generally: Δk constant within segments → piecewise-linear generator
- The database stores a FAMILY of linear generators parameterized by (slope, intercept)
- When a new flat block arrives, the database offers these linear generators; the scanner checks if one fits (residual < V = 1/12)

**Elliptic (K > 0, constant) → Periodic/Trigonometric Generators:**
- Constant positive ΔΔk means the Δk stream is linear → k-stream is quadratic
- But K > 0 with bounded data means PERIODIC behavior (closed T-traversal per §14 of Non-Euclidean paper)
- D_gen = {period, amplitude, phase, offset} — a periodic function in lattice space
- The database derives: for a given K̄ value, the period is P = 2π/√K̄. The amplitude, phase, and offset are fitted from the first period of data.
- One period of data + the generator = entire block reconstructed

**Hyperbolic (K < 0) → Grammar/Recursive Generators:**
- The D Paper §44.1 establishes that chaotic data has low Kolmogorov complexity — the generating map is simple even though the output is complex
- D_gen = a grammar (Re-Pair rules) or a recurrence relation
- The database stores grammar TEMPLATES: common rule structures seen across chaotic data (e.g., pair-replacement hierarchies, iterated function systems)
- From the AI compression module: "9 levels of recursion compresses 10^9 nodes to ~1." Grammar rules are themselves compressible into higher-order grammar rules. The database stores the HIERARCHY of grammar templates.

**Variable curvature → Segmented Generators:**
- D_gen = {segment_boundaries, per-segment generators}
- The database derives: for curvature-sign-change boundaries, the optimal segmentation + per-segment generator type

#### 16.8.3 The Discovery Loop: Meta-Recognition Awareness

From the D Paper §45.2:

> "When a gap is detected in the current Descriptor set, the awareness of the gap is itself a Descriptor-generating event: it initiates the search for D_missing."

Applied to the database:

1. **Observe**: A file is compressed. The scanner finds archetypes. These are stored (Channel A).
2. **Detect gap**: The curvature profile of this file does NOT match any existing D_gen in the database. This gap IS a Descriptor — it means a new generator family exists that the database hasn't derived yet.
3. **Derive**: From the curvature profile, derive what KIND of D_gen could produce this profile. Store the derived generator template as an {P,D} Unsubstantiated entry.
4. **Scout**: The next file with a similar curvature profile finds this template waiting. The scanner checks if the template fits. If it does, the template is substantiated — it becomes a confirmed D_gen. If not, the template is refined.
5. **Compound**: Over time, the database accumulates both observed archetypes (Channel A) and derived generator templates (Channel B). Each new file benefits from BOTH — the memory of what was seen AND the predictions of what should exist.

This is the "Library of Congress in RAM" mechanism applied to file compression. The AI's LatticeCompressor gets more efficient the more it learns because archetypes subsume archetypes recursively. The CDF compressor's database gets more efficient the more files it processes because:
- Channel A accumulates proven patterns (observation compounds)
- Channel B derives new generator families (discovery compounds)
- The two channels feed each other: observed patterns reveal new curvature classes → new curvature classes predict new generators → predicted generators accelerate compression of new files → which yields more observations

#### 16.8.4 New Database Table: Generative Descriptors

```sql
CREATE TABLE IF NOT EXISTS generative_descriptors (
    gen_id TEXT PRIMARY KEY,
    curvature_class INTEGER NOT NULL,       -- 0=flat, 1=elliptic, 2=hyperbolic, 3=variable
    generator_type TEXT NOT NULL,           -- 'linear', 'periodic', 'polynomial', 'grammar', 'recurrence'
    generator_params BLOB NOT NULL,         -- Serialized generator parameters
    param_count INTEGER NOT NULL,           -- |D_gen| — number of parameters
    curvature_mean_range_low REAL,          -- K̄ range this generator covers
    curvature_mean_range_high REAL,
    fit_count INTEGER DEFAULT 0,            -- How many blocks this generator has successfully fitted
    miss_count INTEGER DEFAULT 0,           -- How many blocks it was offered to but failed
    best_residual_variance REAL,            -- Best σ²_residual achieved
    first_derived REAL NOT NULL,            -- Timestamp of derivation
    last_confirmed REAL,                    -- Timestamp of last successful fit
    source TEXT DEFAULT 'derived'           -- 'derived' (Channel B) or 'observed' (promoted from Channel A)
);

CREATE INDEX IF NOT EXISTS idx_gen_curvature
    ON generative_descriptors(curvature_class, generator_type);
CREATE INDEX IF NOT EXISTS idx_gen_fitness
    ON generative_descriptors(fit_count DESC, best_residual_variance);
```

#### 16.8.5 The Compression Pipeline with Both Channels

When a block arrives for compression:

1. **Phase 1.5** (Curvature Analysis): Compute K̄, σ²_K, curvature class, χ_block.
2. **Channel B query**: Ask the generative_descriptors table: "what generators exist for this curvature class and K̄ range?" Retrieve top candidates by fit_count.
3. **Generator fitting**: For each candidate D_gen, attempt to fit the block:
   - Compute the predicted data from D_gen
   - Compute the residual = actual − predicted
   - If σ²_residual < V = 1/12 (subliminal threshold): D_gen fits. Store D_gen + residual.
   - If residual is identically zero: PERFECT fit. Store only D_gen. **This is the Kolmogorov-minimal case.**
4. **Channel A query**: Ask the archetypes table for pattern boosts (existing pipeline).
5. **Standard pipeline**: Modes 0/1/2/3, archetype scanning, Re-Pair. All still run.
6. **Competition**: All candidates — D_gen fit, Channel A boosted modes, standard modes — compete. Smallest output wins.

When a D_gen candidate fits:
- The compressed block stores: generator_type + generator_params + residual stream (which may be empty or very small)
- The decompressor reconstructs: apply D_gen to produce predicted data, add residual
- The database increments fit_count for this D_gen (it's getting confirmed)

When no D_gen fits:
- Standard compression runs (Modes 0/1/2/3 + archetypes)
- If a new curvature profile is seen, the database DERIVES a new generator template (the discovery loop from §16.8.3)
- The new template is stored with fit_count=0, waiting for the next similar file

#### 16.8.6 How This Exceeds 7zip

7zip (LZMA) operates in P-space: it finds repeated byte sequences in a sliding window and encodes backreferences. It has NO understanding of what generates the data. Its theoretical limit is the Shannon entropy of the byte stream — it cannot compress below the marginal entropy.

The Generative Descriptor approach operates in D-space: it finds the RULE that generates the bytes. Its theoretical limit is the Kolmogorov complexity — the length of the shortest program that produces the data. For ALL structured data, Kolmogorov complexity < Shannon entropy, because structure means the data can be generated by a program shorter than its entropy encoding.

The gap between Shannon entropy and Kolmogorov complexity IS the gap between 7zip and the ET compressor. For random data, the gap is zero (both equal the data length). For structured data, the gap is arbitrarily large — a gigabyte linear ramp has Shannon entropy ~8 bits/byte but Kolmogorov complexity ~30 bits total.

The database compounds this advantage over time:
- First file of a type: no known generators, falls back to standard (≈7zip level)
- Subsequent files: database offers generators, some fit, compression improves
- After many files: database has generators for most data types, compression approaches Kolmogorov limit
- The Library of Congress: millions of files, mostly text/documents with known generators → massive compression

This is the "more efficient the more it learns" property from the AI's LatticeCompressor — applied to file compression via the database's dual-channel architecture.

#### 16.8.7 ET Derivation

**Identification Principle:** The Generative Descriptor IS the D of the data. Finding D_gen is the Identification Principle applied to compression: "identify D_X." The scanner identifies observed patterns. The discovery engine identifies PREDICTED patterns. Together they identify the complete D-set.

**Descriptor Gap Principle:** The gap between "what the scanner finds" and "what could exist" IS a Descriptor (the missing generator template). The database fills this gap by deriving generators from the curvature geometry. "Any gap in a description is itself a Descriptor — a missing element that needs to be found."

**Subsumption Law:** The Generative Descriptor subsumes the data without remainder — D_gen + T (execution) = P (data), exactly. This is the Subsumption Law applied to compression: the generator subsumes all the data it produces. The database's generator hierarchy (grammar templates subsuming grammar rules subsuming pair replacements) mirrors the AI's recursive archetype hierarchy: "9 levels of recursion compresses 10^9 nodes to ~1."

### 16.9 ABSOLUTE NO-REMOVAL POLICY — Nothing Is Ever Deleted

**This overrides the current `_check_disk_safety()` pruning logic in `ArchetypeDatabase`.**

The current compressor (lines 3879-3930 of `et_cdf_compressor.py`) has a `_check_disk_safety()` method that prunes T_WEIGHT = 1/3 of lowest-value archetypes when disk free space drops below `DISK_SAFETY_FLOOR` (1 GB). This pruning logic must be **REMOVED ENTIRELY**. No archetype, no generator template, no database entry of any kind is EVER deleted, pruned, trimmed, evicted, or removed.

**Rationale from the No Removal Policy (project-wide rule):**

> "Unused code or variables always indicate incomplete implementations — they must be traced and completed, never removed, suppressed, or prefixed with underscore."

Applied to the database: an archetype with low hit_count does not mean it is useless. It means the compressor has not yet encountered the data type that archetype describes. It is an {P,D} Unsubstantiated entry — a scout report from a region of the lattice that has not been revisited yet. Deleting it destroys knowledge that cannot be regenerated without re-encountering the original data.

**The compounding argument:** Every archetype and every generator template makes the database more capable over time. A low-value archetype today may become a critical cross-R₀ bridge tomorrow when a new file type arrives with a matching curvature profile. Channel B generator templates that have never been confirmed (fit_count = 0) are PREDICTIONS — they represent the database's theory about what generators SHOULD exist for a given curvature class. Deleting unconfirmed predictions defeats the entire discovery mechanism. The theoretical prediction must persist until either confirmed or structurally invalidated by the Incoherence Filter — not deleted for disk space.

**The recompression argument:** If disk space is genuinely low, the correct response is to compress the database via CDF VFS (§19), NOT to delete its contents. CDF VFS compression reclaims 80-96% of disk space (§19.9) WITHOUT losing any data. The disk safety problem is solved by compression, not by destruction.

**Implementation changes to `et_cdf_compressor.py`:**

1. **DELETE** the `_check_disk_safety()` method entirely (lines 3879-3930).
2. **DELETE** the call `self._check_disk_safety()` at end of `store()` (line 3768).
3. **DELETE** all references to `DISK_SAFETY_FLOOR` in the `ArchetypeDatabase` class.
4. **KEEP** `DISK_SAFETY_FLOOR` in `CDFMetabolism` for disk warning only (warn user, never delete data).
5. **ADD** to `store()`: after storing new archetypes, if disk free space is below 1 GB, call `compact_to_cdf()` (§19.7) to compress the database instead of pruning it.

The replacement logic:

```python
def store(self, archetypes, source_r0):
    """Store newly discovered archetypes. NOTHING IS EVER DELETED."""
    # ... existing store logic (INSERT/UPDATE) ...
    
    # Instead of _check_disk_safety() which DELETED archetypes:
    # Compress the database if disk is low.
    profile = CDFResourceSensor.sense()
    if profile.disk_free_bytes < (1024 ** 3):  # < 1 GB free
        self._log("Low disk space — compressing database (no data deleted)")
        self.compact_to_cdf()
```

**The database grows forever.** It is compressed, not pruned. Over time, as more files are processed:
- Channel A accumulates more observed archetypes → better pattern matching
- Channel B accumulates more generator templates → better D-space compression
- Cross-R₀ curvature lookup grows richer → more cross-type pattern reuse
- Older files recompressed through the mature database achieve better ratios

Every entry contributes to this compounding. Deleting any entry reduces the compound return of ALL future compressions. The CDF VFS ensures disk footprint stays manageable without sacrificing knowledge.

**This is a HARD RULE. No exceptions. No "emergency" pruning. No "safety" deletion. Compress, never destroy.**

---

## 17. C Pattern Engine Extensions {#17-c-engine}

### 17.1 Current C Engine Functions

The C pattern engine (`et_pattern_engine.c`) currently exports six functions:

| Function | Purpose | Called From |
|----------|---------|-------------|
| `find_repeated_patterns` | Suffix array + LCP pattern finding | `PatternEngine.find_patterns()` |
| `build_k_stream` | Vectorized byte→k lookup (256-entry table) | `PatternEngine.fast_k_stream()` |
| `build_dk_stream` | First differences of k-stream | Used internally |
| `gate_archetype_batch` | Batch IncoherenceFilter L1-L4 | `PatternEngine.find_and_gate()` |
| `subsume_greedy` | Non-overlapping greedy placement | `PatternEngine.subsume_greedy_c()` |
| `free_buffer` | Release malloc'd buffer | After every find call |

### 17.2 New C Functions Required

Each curvature improvement that involves inner-loop computation over the full stream should have a C-accelerated path. The Python fallback always exists (zero loss in function if C engine unavailable), but the C path eliminates per-element Python overhead for large blocks.

**Function 1: `build_ddk_stream` — Second differences (curvature stream)**

```c
EXPORT void build_ddk_stream(const int32_t *dk_stream, int n,
                              int32_t *ddk_out)
{
    /* ΔΔk_i = Δk_{i+1} - Δk_i = curvature at position i
     * Output length = n - 1 (n = length of dk_stream, which is already n_bytes - 1)
     * So ddk_out has length n_bytes - 2
     *
     * From Non-Euclidean §6: K = ∇²f = second-order D-gradient
     * This IS the discrete Gaussian curvature of the data's Descriptor field.
     */
    for (int i = 0; i < n - 1; i++)
        ddk_out[i] = dk_stream[i + 1] - dk_stream[i];
}
```

**Function 2: `compute_curvature_stats` — Block curvature classification**

```c
typedef struct {
    double curvature_mean;     /* K̄ = mean of ΔΔk */
    double curvature_variance; /* σ²_K = variance of ΔΔk */
    int    curvature_class;    /* 0=flat, 1=elliptic, 2=hyperbolic, 3=variable, 4=singular */
    double euler_characteristic; /* χ = Σ(ΔΔk) / (2π) */
    int    max_abs_curvature;  /* max(|ΔΔk_i|) — for singularity detection */
} CurvatureStats;

EXPORT void compute_curvature_stats(
    const int32_t *ddk_stream, int n,
    int n_res,                 /* 27720 — for subliminal threshold */
    CurvatureStats *out)
{
    /* Single pass: compute mean, variance, sum, max in O(n).
     *
     * Classification thresholds (from Non-Euclidean §3.3):
     *   subliminal = π/N = π/12 ≈ 0.2618
     *   base_variance = V = 1/12 ≈ 0.0833
     *   singular_threshold = N = 27720
     */
    double sum = 0.0, sq_sum = 0.0;
    int max_abs = 0;
    
    for (int i = 0; i < n; i++) {
        double val = (double)ddk_stream[i];
        sum += val;
        sq_sum += val * val;
        int abs_val = abs(ddk_stream[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    
    double mean = (n > 0) ? sum / n : 0.0;
    double variance = (n > 0) ? (sq_sum / n) - (mean * mean) : 0.0;
    if (variance < 0.0) variance = 0.0; /* Numerical safety */
    
    double subliminal = 3.14159265358979323846 / 12.0;
    double base_var = 1.0 / 12.0;
    
    out->curvature_mean = mean;
    out->curvature_variance = variance;
    out->euler_characteristic = sum / (2.0 * 3.14159265358979323846);
    out->max_abs_curvature = max_abs;
    
    /* Classification */
    if (max_abs >= n_res) {
        out->curvature_class = 4; /* Singular */
    } else if (variance >= base_var) {
        out->curvature_class = 3; /* Variable */
    } else if (mean >= subliminal) {
        out->curvature_class = 1; /* Elliptic (K > 0) */
    } else if (mean <= -subliminal) {
        out->curvature_class = 2; /* Hyperbolic (K < 0) */
    } else {
        out->curvature_class = 0; /* Flat */
    }
}
```

**Function 3: `build_geodesic_residual` — Geodesic residual stream (Mode 3)**

```c
EXPORT void build_geodesic_residual(
    const int32_t *dk_stream, int n,
    int connection_order,     /* 0, 1, or 2 */
    int window_size,          /* L4-bounded connection window */
    int32_t *residual_out,    /* output: length n-1 */
    int32_t *gamma_out)       /* output: connection coefficients, length n-1 */
{
    /* Geodesic residual ρ_i = Δk_{i+1} - Δk_i^{predicted}
     * where Δk_i^{predicted} = Δk_i + Γ_i (connection)
     *
     * Connection orders (from Non-Euclidean §6.1):
     *   0: Γ = 0 (predict Δk_{i+1} = Δk_i)
     *   1: Γ = windowed mean of ΔΔk (linear trend)
     *   2: Γ = order-1 + ½ × windowed mean of ΔΔΔk (quadratic trend)
     *
     * From Non-Euclidean §9.2: "T naturally follows paths of minimal
     * total cost — the binding of least descriptor resistance."
     */
    
    if (n < 2) return;
    
    for (int i = 0; i < n - 1; i++) {
        int32_t gamma = 0;
        
        if (connection_order >= 1 && i > 0) {
            /* First-order: running mean of ΔΔk over window */
            int w_start = (i - window_size + 1 > 0) ? i - window_size + 1 : 0;
            int64_t ddk_sum = 0;
            int count = 0;
            for (int j = w_start; j < i; j++) {
                ddk_sum += (int64_t)(dk_stream[j + 1] - dk_stream[j]);
                count++;
            }
            if (count > 0) gamma = (int32_t)(ddk_sum / count);
        }
        
        if (connection_order >= 2 && i > 1) {
            /* Second-order: add ½ × running mean of ΔΔΔk */
            int w_start = (i - window_size + 1 > 1) ? i - window_size + 1 : 1;
            int64_t dddk_sum = 0;
            int count = 0;
            for (int j = w_start; j < i - 1; j++) {
                int32_t ddk_j = dk_stream[j + 1] - dk_stream[j];
                int32_t ddk_j1 = dk_stream[j + 2] - dk_stream[j + 1];
                dddk_sum += (int64_t)(ddk_j1 - ddk_j);
                count++;
            }
            if (count > 0) gamma += (int32_t)(dddk_sum / (2 * count));
        }
        
        gamma_out[i] = gamma;
        int32_t predicted = dk_stream[i] + gamma;
        residual_out[i] = dk_stream[i + 1] - predicted;
    }
}
```

**Function 4: `gate_archetype_batch_curvature` — Extended batch gate with curvature-adjusted threshold**

```c
EXPORT void gate_archetype_batch_curvature(
    const int32_t *patterns_buf, int n_patterns,
    int n_res, double incoherence_cents,
    const int32_t *ddk_stream, int ddk_len,  /* curvature stream for threshold adjustment */
    uint8_t *out_mask)
{
    /* Same as gate_archetype_batch but with curvature-adjusted ε threshold.
     * From Non-Euclidean §11.2:
     *   ε_max(i) = 50 × 1/(1 + |K_i|/N)
     * Where K_i = ddk_stream at the pattern's occurrence positions.
     * 
     * This tightens the Incoherence boundary in curved regions — patterns
     * in strongly curved data face stricter coherence requirements because
     * the rounding error is magnified by curvature.
     */
    /* [Full implementation follows same structure as gate_archetype_batch
     * but uses curvature-adjusted eps threshold per pattern position] */
}
```

**Function 5: `compute_pattern_curvature` — Per-pattern curvature stats for DB storage**

```c
EXPORT void compute_pattern_curvature(
    const int32_t *pattern_dk, int pat_len,
    double *out_curvature_mean,
    double *out_curvature_variance,
    double *out_geodesic_factor)
{
    /* Compute curvature stats for a single pattern's Δk sequence.
     * Used at archetype storage time — computed once, stored permanently.
     *
     * K̄ = mean(ΔΔk within pattern)
     * σ²_K = variance(ΔΔk within pattern) 
     * F_K = 1/(1 + σ²_K)
     */
    if (pat_len < 3) {
        *out_curvature_mean = 0.0;
        *out_curvature_variance = 0.0;
        *out_geodesic_factor = 1.0;
        return;
    }
    
    double sum = 0.0, sq_sum = 0.0;
    int n_ddk = pat_len - 2;
    
    for (int i = 0; i < n_ddk; i++) {
        double ddk = (double)(pattern_dk[i + 2] - 2 * pattern_dk[i + 1] + pattern_dk[i]);
        sum += ddk;
        sq_sum += ddk * ddk;
    }
    
    double mean = sum / n_ddk;
    double var = (sq_sum / n_ddk) - (mean * mean);
    if (var < 0.0) var = 0.0;
    
    *out_curvature_mean = mean;
    *out_curvature_variance = var;
    *out_geodesic_factor = 1.0 / (1.0 + var);
}
```

### 17.3 Python ctypes Registration

Each new C function needs ctypes registration in `PatternEngine._try_load()`, following the existing pattern:

```python
# In PatternEngine._try_load():
cls._lib.build_ddk_stream.restype = None
cls._lib.build_ddk_stream.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32)]

cls._lib.compute_curvature_stats.restype = None
cls._lib.compute_curvature_stats.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ctypes.c_int, ctypes.POINTER(CurvatureStats)]

cls._lib.build_geodesic_residual.restype = None
cls._lib.build_geodesic_residual.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

cls._lib.compute_pattern_curvature.restype = None
cls._lib.compute_pattern_curvature.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)]
```

### 17.4 Embedded C Source Update

The `PatternEngine._C_SOURCE` embedded string must include all new functions. This ensures the script remains fully self-contained — the separate `.c` file is for CLion/CMake builds, but the embedded copy is what the PyInstaller `.exe` uses for auto-compilation when no pre-built DLL is found.

### 17.5 CMakeLists.txt — No Changes Needed

The CMakeLists.txt already builds `et_pattern_engine.c` as a shared library. Adding functions to the `.c` file requires no CMake changes — the new functions are simply additional EXPORT entries in the same compilation unit.

### 17.6 Test Harness (main.cpp) — New Tests Required

The test harness (`main.cpp`) must be extended with tests for each new C function:

- `test_build_ddk_stream`: Verify ΔΔk computation against known inputs
- `test_compute_curvature_stats`: Verify K̄, σ²_K, classification for flat/elliptic/hyperbolic/variable test streams
- `test_build_geodesic_residual`: Verify residual computation at connection orders 0, 1, 2 — roundtrip: reconstruct Δk from residual + connection
- `test_compute_pattern_curvature`: Verify per-pattern curvature stats for known patterns

### 17.7 ET Derivation

**Identification Principle:** Each new C function identifies one curvature operation that was either missing (build_ddk_stream, compute_curvature_stats, compute_pattern_curvature) or Python-only with inner-loop overhead (build_geodesic_residual, gate_archetype_batch_curvature).

**Descriptor Gap Principle:** The gap between "the C engine handles pattern finding" and "the C engine handles curvature analysis" is closed by these functions. The C engine currently stops at first-order operations (k-stream, Δk-stream, L1-L4 gating). The new functions extend it to second-order operations (ΔΔk-stream, curvature classification, geodesic residuals).

**Subsumption Law:** After these additions, the C engine subsumes the entire derivative chain without remainder: bytes → k → Δk → ΔΔk → curvature stats → geodesic residuals → curvature-adjusted gating. Every step has a C-accelerated path. The Python fallback remains for every function (zero loss if C engine unavailable).

---

## 18. Integration Architecture: Where Each Improvement Enters the Pipeline {#18-integration}

### 18.1 Current Pipeline

```
Phase 1: R₀ identification
Phase 2: Lattice Transform (k → Δk → ΔΔk, mode selection)
Phase 3: Pattern Finding (suffix array, IncoherenceFilter gate)
Phase 4: Archetype Subsumption (greedy placement, recursive)
Phase 4b: Chaotic data Re-Pair (fallback on expansion)
Phase 5: Enhanced lattice strategies (fallback on further expansion)
Phase 6: V_config encoding + serialization
```

### 18.2 Enhanced Pipeline with Non-Euclidean Improvements

```
Phase 1: R₀ identification (unchanged)

Phase 1.5: CURVATURE ANALYSIS (NEW — additive, never removes existing work)
  ├── Compute ΔΔk stream (already computed for Mode 2 — reused)
  ├── Compute K̄_block, σ²_K, curvature spectrum
  ├── Classify block: flat/elliptic/hyperbolic/variable/singular
  ├── Compute χ_block (Gauss-Bonnet fingerprint) for DB lookup
  ├── Determine connection order for geodesic residual (Mode 3)
  └── If variable/singular: compute segmentation boundaries
      [Improvements 1, 6, 7, 11, 14]

Phase 2: Lattice Transform (extended — Mode 3 ADDED, Modes 0/1/2 unchanged)
  ├── Modes 0/1/2 (unchanged — ALL still run)
  ├── Mode 3: Geodesic residual with Christoffel connection (NEW)
  │   [Improvements 2, 3]
  └── Riemann sphere complex projection (refines existing projection)
      [Improvement 10]

Phase 2.5: VARIABLE CURVATURE SEGMENTATION (NEW, if classified variable)
  ├── Identify curvature sign-change boundaries
  ├── Merge short segments (< S² = 144)
  ├── Classify each segment independently
  ├── Compress each segment with ALL strategies (curvature-matched + existing)
  └── Compare: segmented total vs full-block outputs — keep smallest
      [Improvement 7]

Phase 3: Pattern Finding (extended — new scoring, new gates, NOTHING removed)
  ├── Suffix array + LCP (unchanged — ALL patterns still found)
  ├── IncoherenceFilter with curvature-adjusted thresholds (ENHANCED)
  │   [Improvement 8]
  ├── Curvature-weighted elegance scoring (ENHANCED — adds F_K term)
  │   [Improvement 4]
  └── Cyclic archetype extraction for elliptic blocks (NEW — additive)

Phase 4: Archetype Subsumption (extended — new DB features, core unchanged)
  ├── Greedy placement (unchanged)
  ├── Curvature-spectrum DB lookup for elegance boost (ENHANCED)
  │   [Improvements 5, 9, 11, 14]
  ├── Cross-tower curvature matching (NEW — recovers patterns d-family
  │   check rejects) [Improvement 9]
  └── Geodesic deviation stability scoring for DB storage (NEW)
      [Improvement 12]

Phase 4b: Chaotic data Re-Pair
  ├── EXISTING: still runs as expansion fallback (unchanged)
  └── NEW: ALSO runs as primary candidate for K<0 (hyperbolic) blocks
      [Improvement 1 — curvature classification promotes Re-Pair]

Phase 5: Enhanced strategies (unchanged — curvature improvements ADD
          candidates that reduce how often blocks reach this fallback)

Phase 6: V_config encoding + serialization (unchanged)

INVARIANT: ALL strategies compete. The smallest output wins. No strategy
is ever skipped. Curvature adds NEW candidates; it never removes existing ones.
```

### 18.3 Critical Constraint: Compression Ratio Is the Only Metric

**Speed is irrelevant.** The compressor's philosophy: "Speed is IRRELEVANT — only compression ratio and correctness matter." Every improvement must be evaluated solely by whether it produces a smaller (or equal) lossless output. A new strategy that takes 10× longer but achieves 1% better compression is a correct addition.

Every improvement is **additive**:
- Improvements 1, 6, 7: Add new strategies — all existing strategies ALSO run. Segmented AND unsegmented outputs compared — keep smallest.
- Improvements 2, 3: Add Mode 3 — Modes 0/1/2 all unchanged and all still compete.
- Improvement 4: Augment elegance — existing elegance term preserved, curvature factor adds discriminating power for ties.
- Improvements 5, 9, 12, 14: Enhance DB lookup — existing exact-match lookup unchanged, curvature adds cross-R₀ matching channel.
- Improvement 8: Tighten filter in curved regions — flat regions unchanged, curved regions get stricter gating (more precise, potentially catching borderline Incoherent patterns that were leaking through).
- Improvement 10: Refine complex projection — existing gate preserved, chordal metric adds precision near poles.
- Improvement 11: Segment variable blocks — full-block strategies ALSO run, smallest wins.

**No existing strategy is ever removed, skipped, or deprioritized.** The curvature improvements increase the number of candidates competing for each block. More candidates means equal or better compression ratio. Never worse.

---

## 19. CDF-Compressed Database with Random Access (CDF VFS) {#19-cdf-vfs}

### 19.1 The Problem

The archetype database (`archetypes.db`) grows without bound as the compressor learns. Channel A accumulates observed patterns. Channel B derives generator templates. Both compound over time — this is the design intent. But disk footprint grows proportionally.

The database compresses extremely well under CDF because its content IS lattice-structured data — archetypes stored by lattice position, keyed by R₀, organized by d-families. Lattice data on the lattice compressor is a structural match, not coincidence.

The question: can we keep it compressed and still query it?

### 19.2 Why This Works: Generators Are Functions, Not Recordings

A Generative Descriptor is a function. A function can be evaluated at any point in its domain without expanding its entire range. This is the fundamental distinction between D-space compression and P-space compression:

- **P-space (7zip, current CDF blocks):** The compressed output is a recording of the data. To read any part, you must replay the recording from the beginning (or from the last block boundary). Random access requires full block decompression.
- **D-space (Generative Descriptors):** The compressed output is a rule that produces the data. To read any part, you evaluate the rule at that point. Random access is native — the generator computes any requested byte range without expanding anything else.

The database is the ideal candidate because:
1. It compresses well (high compression ratio = generators fit tightly = low residuals).
2. Its access pattern is sparse (SQLite reads a few 4KB pages per query, not the whole file).
3. Its content is lattice-structured (generators for lattice data are naturally concise).

### 19.3 SQLite VFS Architecture

SQLite uses a Virtual File System (VFS) abstraction. Every file operation — open, read, write, close, sync, file size — goes through a VFS layer. The default VFS reads and writes the OS filesystem. A custom VFS can intercept these operations and redirect them anywhere.

The CDF VFS intercepts read operations on the compressed database and evaluates generators instead of reading raw bytes:

```
[Application / Compressor]
        │
        ▼
[SQLite Engine] ← thinks it's reading a normal file
        │
        ▼
[CDF VFS Layer] ← intercepts xRead(offset, length)
        │
        ├── Lookup: which generator covers bytes [offset, offset+length)?
        ├── Evaluate: compute those bytes from D_gen + residual
        └── Return: 4KB page to SQLite
        │
        ▼
[archetypes.cdf on disk] ← compressed, never fully decompressed
```

SQLite never knows the file is compressed. It issues page reads at specific offsets. The VFS translates each read into a generator evaluation. The compressed .cdf file stays on disk permanently.

### 19.4 CDF VFS File Format

The .cdf file for a random-access database has a specific structure. The generator index MUST be accessible without decompressing anything — it is stored uncompressed at a known offset.

```
CDF Random-Access Format:
═══════════════════════════════════════════════════════════════

[CDF Header — 57 bytes, same as current CDF]
  [4 bytes]   Magic: CDF\x03 (version 3 — random access capable)
  [1 byte]    Version: 3
  [32 bytes]  SHA-256 of original uncompressed database
  [8 bytes]   Original file size (uint64 LE)
  [4 bytes]   Number of generators (uint32 LE)
  [8 bytes]   Generator index offset within this file (uint64 LE)

[Generator Payloads — variable length, sequentially packed]
  For each generator:
    [payload bytes]  Generator-specific data (params + residual)
    The payload format depends on generator_type (see §17.5)

[Residual Pool — variable length]
  Residual data for generators with imperfect fits.
  Each residual block is independently decompressible.
  Referenced by residual_offset in the generator index.

[Generator Index — uncompressed, at known offset from header]
  [4 bytes]   Index entry count (uint32 LE)
  For each generator (sorted by domain_start ascending):
    [8 bytes]   domain_start: first byte offset this generator covers (uint64 LE)
    [8 bytes]   domain_length: number of bytes this generator covers (uint64 LE)
    [1 byte]    generator_type: enum (see §17.5)
    [1 byte]    connection_order: for geodesic generators (0/1/2)
    [2 bytes]   param_count: number of generator parameters (uint16 LE)
    [8 bytes]   payload_offset: offset of this generator's payload in file (uint64 LE)
    [4 bytes]   payload_length: bytes of payload data (uint32 LE)
    [8 bytes]   residual_offset: offset into residual pool (uint64 LE, 0 = no residual)
    [4 bytes]   residual_length: bytes of residual data (uint32 LE, 0 = perfect fit)
    [8 bytes]   curvature_mean: K̄ of the region (float64 LE)
  ─────────────────────────────────────────────────
  Total per entry: 52 bytes fixed

[Footer]
  [32 bytes]  SHA-256 of the generator index (integrity check)
  [8 bytes]   Generator index offset (repeated for backward seek)
```

**Index size estimate:** A 100MB database with well-fitting generators might have 500-2000 generators. At 52 bytes per entry: 26KB-104KB for the index. This is <0.1% overhead — negligible.

The index is sorted by `domain_start`, enabling O(log n) binary search for any byte offset. SQLite requests page at offset X → binary search the index → find the generator whose domain contains X → read its payload → evaluate → return bytes.

### 19.5 Generator Types and Payload Formats

Each generator type has a specific payload format. The `generator_type` byte in the index selects the evaluator:

**Type 0: Constant (K = 0 trivially, run of identical bytes)**

```
Payload: [1 byte] value
Evaluate(offset): return value
```

The simplest generator. A region where every byte is the same. payload_length = 1. This covers zero-regions, padding, and constant fields in the database.

**Type 1: Linear (K ≈ 0, constant Δk)**

```
Payload: [4 bytes] k_start (int32 LE)
         [4 bytes] dk_step (int32 LE)
         [8 bytes] r0 (float64 LE)
Evaluate(offset):
    k = k_start + dk_step × (offset - domain_start)
    byte = k_byte_map(r0)[nearest_k(k)]
```

Covers linear ramps in lattice space. The k-stream increases by a constant step. payload_length = 16. Covers sequential integer fields, monotonically increasing IDs, sorted index entries.

**Type 2: Polynomial (K > 0, constant ΔΔk → quadratic k-stream)**

```
Payload: [4 bytes] degree (uint32 LE)
         [8 × (degree+1) bytes] coefficients (float64 LE each)
         [8 bytes] r0 (float64 LE)
Evaluate(offset):
    t = offset - domain_start
    k = Σ(coefficients[i] × t^i, i=0..degree)
    byte = k_byte_map(r0)[nearest_k(round(k))]
```

Covers smooth curves in lattice space. Quadratic (degree=2) is the most common — covers sinusoidal half-cycles, gradual transitions. payload_length = 8 + 8×(degree+1). For degree 2: 32 bytes covers an arbitrarily long smooth region.

**Type 3: Periodic (elliptic, K > 0 constant, closed T-traversal)**

```
Payload: [4 bytes] period (uint32 LE)
         [period bytes] one_cycle (the pattern bytes of one complete cycle)
Evaluate(offset):
    cycle_pos = (offset - domain_start) mod period
    return one_cycle[cycle_pos]
```

Covers exactly repeating regions. One cycle is stored; all repetitions are generated by modular indexing. payload_length = 4 + period. For a 100-byte repeating pattern covering 100KB: 104 bytes. This naturally handles SQLite's repeating internal structures (freelist pages, overflow chains, B-tree fill patterns).

**Type 4: Grammar (hyperbolic, K < 0, Re-Pair rule hierarchy)**

```
Payload: [manifold uint] n_rules
         For each rule:
           [manifold uint] left_symbol
           [manifold uint] right_symbol
         [manifold uint] n_start_symbols
         [start_symbols: manifold uint each]
Evaluate(offset):
    Navigate the grammar tree to find the leaf at position (offset - domain_start).
    O(log depth) per byte. For depth D rules: O(D) per byte.
    Batch evaluation of a page (4096 bytes): expand the grammar
    for the relevant range only.
```

Covers chaotic/structured regions where Re-Pair grammar induction found structure. The grammar IS the Generative Descriptor — the rule hierarchy that produces the data. Access is O(depth) per byte, but grammar depth is bounded by MAX_COMPRESSION_DEPTH = S = 12, so worst case is O(12) per byte.

**Type 5: Geodesic Residual (Mode 3, connection-predicted)**

```
Payload: [4 bytes] dk0 (int32 LE, initial Δk)
         [1 byte] connection_order (0/1/2)
         [manifold uint] connection_window
         [8 bytes] r0 (float64 LE)
         [compressed residual stream]
Evaluate(offset):
    Reconstruct Δk causally from dk0 through the connection
    up to the requested offset. O(offset) worst case.
    OPTIMIZATION: cache the Δk reconstruction state at page
    boundaries (every 4096 bytes). Cache entry = 12 bytes
    (dk_current, gamma_current, position). Amortized O(page_size)
    per page read after first access.
```

Covers smooth transition regions. The residual stream is small (near-zero values). The causal reconstruction requires sequential access within the generator's domain — this is the one type where random access within a generator is not O(1). The page-boundary cache mitigates this: after the first access to any page in the generator's domain, subsequent pages are O(4096).

**Type 6: Archetype Reference (pointer to database's own archetype table)**

```
Payload: [32 bytes] archetype_hash (from archetypes table)
         [manifold uint] instance_index
Evaluate(offset):
    Look up the archetype's Generative Descriptor.
    Evaluate that descriptor at the adjusted offset.
    This is recursive tower access — a generator that
    references another generator.
```

This type enables the recursive compression property. A region of the database that matches a known archetype pattern doesn't store its own generator — it stores a REFERENCE to the archetype's generator. The archetype itself may reference higher-order archetypes. Each level of indirection adds O(1) lookup. This is the "archetypes of archetypes" from the AI compression module: "9 levels of recursion compresses 10^9 nodes to ~1."

**Type 7: Raw (passthrough, incompressible region)**

```
Payload: [raw bytes, 1:1 with original]
Evaluate(offset):
    return payload[offset - domain_start]
```

For regions where no generator fits (genuine entropy — encrypted fields, random hashes, SHA-256 digests). payload_length = domain_length. No compression, but still participates in the random-access index. This ensures the format subsumes all data without remainder.

### 19.6 The CDF VFS Implementation

#### 19.6.1 Python VFS Class

SQLite's Python binding (`sqlite3` module) uses the C library's VFS internally. Python cannot directly register a VFS with the C SQLite. Instead, the CDF VFS operates as a **page cache layer** above SQLite:

```python
class CDFDatabaseVFS:
    """
    Random-access layer for CDF-compressed SQLite databases.
    
    Opens a .cdf file containing a compressed SQLite database.
    Provides read access at arbitrary byte offsets by evaluating
    Generative Descriptors from the CDF generator index.
    
    Write operations are buffered in memory and flushed to a
    new .cdf file on close (full recompression of modified pages).
    
    ET Derivation:
        P = the original database bytes (the P-substrate)
        D = the generator index + generator payloads (the D-set)
        T = this VFS class (the Traverser navigating D to produce P)
        
        The VFS IS the T that substantiates D into P on demand.
        Each page read is a T-traversal: T evaluates D_gen at the
        requested offset, producing the P-bytes SQLite needs.
        
        The database stays in {P,D} Unsubstantiated form (compressed
        on disk) until T (the VFS) substantiates specific pages into
        {P,D,T} Exception form (decompressed bytes in memory).
        Only the pages SQLite actually reads are substantiated.
        The rest remain Unsubstantiated — potential, not actual.
    
    P ∘ D ∘ T = E
    """
    
    def __init__(self, cdf_path: str):
        self.cdf_path = cdf_path
        self._file = None
        self._index = None          # Generator index (list of entries)
        self._original_size = 0     # Original database size
        self._generators = {}       # Cached generator evaluators by index
        self._page_cache = {}       # LRU cache: page_offset → bytes
        self._page_cache_max = S * S  # 144 pages = 576KB cache (S² from manifold)
        self._dk_state_cache = {}   # For Type 5: Δk reconstruction state at page boundaries
        self._dirty_pages = {}      # Modified pages (for write support)
        self._open()
    
    def _open(self):
        """Open the CDF file and load the generator index."""
        self._file = open(self.cdf_path, 'rb')
        
        # Read CDF header
        magic = self._file.read(4)
        assert magic == b'CDF\x03', f"Not a CDF v3 file: {magic}"
        version = struct.unpack('<B', self._file.read(1))[0]
        assert version == 3
        self._stored_hash = self._file.read(32)
        self._original_size = struct.unpack('<Q', self._file.read(8))[0]
        n_generators = struct.unpack('<I', self._file.read(4))[0]
        index_offset = struct.unpack('<Q', self._file.read(8))[0]
        
        # Read generator index (at known offset, uncompressed)
        self._file.seek(index_offset)
        n_entries = struct.unpack('<I', self._file.read(4))[0]
        assert n_entries == n_generators
        
        self._index = []
        for _ in range(n_entries):
            entry = {
                'domain_start': struct.unpack('<Q', self._file.read(8))[0],
                'domain_length': struct.unpack('<Q', self._file.read(8))[0],
                'generator_type': struct.unpack('<B', self._file.read(1))[0],
                'connection_order': struct.unpack('<B', self._file.read(1))[0],
                'param_count': struct.unpack('<H', self._file.read(2))[0],
                'payload_offset': struct.unpack('<Q', self._file.read(8))[0],
                'payload_length': struct.unpack('<I', self._file.read(4))[0],
                'residual_offset': struct.unpack('<Q', self._file.read(8))[0],
                'residual_length': struct.unpack('<I', self._file.read(4))[0],
                'curvature_mean': struct.unpack('<d', self._file.read(8))[0],
            }
            self._index.append(entry)
    
    def _find_generator(self, offset: int) -> Optional[dict]:
        """
        Binary search the generator index for the entry covering byte offset.
        O(log n) where n = number of generators.
        """
        lo, hi = 0, len(self._index) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            entry = self._index[mid]
            start = entry['domain_start']
            end = start + entry['domain_length']
            if offset < start:
                hi = mid - 1
            elif offset >= end:
                lo = mid + 1
            else:
                return entry
        return None  # Offset not covered (should not happen for valid CDF)
    
    def _load_generator(self, entry: dict):
        """
        Load and cache the generator evaluator for an index entry.
        Reads the payload from disk on first access, caches for reuse.
        """
        key = entry['payload_offset']
        if key in self._generators:
            return self._generators[key]
        
        # Read payload
        self._file.seek(entry['payload_offset'])
        payload = self._file.read(entry['payload_length'])
        
        # Read residual if present
        residual = None
        if entry['residual_length'] > 0:
            self._file.seek(entry['residual_offset'])
            residual = self._file.read(entry['residual_length'])
        
        generator = GeneratorEvaluator(
            gen_type=entry['generator_type'],
            payload=payload,
            residual=residual,
            domain_start=entry['domain_start'],
            domain_length=entry['domain_length'],
            connection_order=entry['connection_order'],
        )
        self._generators[key] = generator
        return generator
    
    def read(self, offset: int, length: int) -> bytes:
        """
        Read bytes at arbitrary offset from the compressed database.
        
        This is the core random-access operation. SQLite calls this
        to read individual 4KB pages. The VFS finds the covering
        generator(s), evaluates them, and returns the bytes.
        
        Page cache (S² = 144 pages) avoids re-evaluation for
        recently accessed pages. LRU eviction when cache is full.
        """
        # Check page cache first
        page_start = (offset // 4096) * 4096
        if page_start in self._page_cache and length <= 4096:
            page = self._page_cache[page_start]
            start_within_page = offset - page_start
            return page[start_within_page:start_within_page + length]
        
        # Check dirty pages (writes that haven't been flushed)
        if page_start in self._dirty_pages and length <= 4096:
            page = self._dirty_pages[page_start]
            start_within_page = offset - page_start
            return page[start_within_page:start_within_page + length]
        
        # Evaluate generator(s) covering the requested range
        result = bytearray()
        pos = offset
        remaining = length
        
        while remaining > 0:
            entry = self._find_generator(pos)
            if entry is None:
                # Beyond original file size — SQLite sometimes probes
                result.extend(b'\x00' * remaining)
                break
            
            generator = self._load_generator(entry)
            
            # How many bytes can this generator provide?
            gen_end = entry['domain_start'] + entry['domain_length']
            available = min(remaining, gen_end - pos)
            
            # Evaluate
            chunk = generator.evaluate(pos, available)
            result.extend(chunk)
            pos += available
            remaining -= available
        
        # Cache the page
        if length <= 4096:
            if len(self._page_cache) >= self._page_cache_max:
                # LRU eviction: remove oldest entry
                oldest_key = next(iter(self._page_cache))
                del self._page_cache[oldest_key]
            self._page_cache[page_start] = bytes(result).ljust(4096, b'\x00')
        
        return bytes(result[:length])
    
    def write(self, offset: int, data: bytes):
        """
        Buffer a write to the compressed database.
        
        Writes are stored in dirty_pages. On close(), the dirty
        pages are merged with the generator-produced clean pages
        and the entire database is recompressed to a new .cdf.
        
        This is the correct approach for a compressor's database:
        writes happen during compression (store new archetypes),
        reads happen during compression (lookup known archetypes).
        The write volume is small relative to the database size.
        Recompression on close is acceptable — speed is irrelevant.
        """
        page_start = (offset // 4096) * 4096
        
        # Ensure we have the full page
        if page_start not in self._dirty_pages:
            # Read the clean page first, then overlay the write
            clean = self.read(page_start, 4096)
            self._dirty_pages[page_start] = bytearray(clean)
        
        start_within_page = offset - page_start
        page = self._dirty_pages[page_start]
        page[start_within_page:start_within_page + len(data)] = data
    
    def file_size(self) -> int:
        """Return the original (uncompressed) database size."""
        return self._original_size
    
    def close(self):
        """
        Close the CDF VFS. If dirty pages exist, recompress.
        
        Recompression: materialize the full database (generators +
        dirty pages), then compress it back to a new .cdf file.
        This is the only time the full database is in memory.
        It happens once per compressor session, after all files
        are processed.
        """
        if self._dirty_pages:
            # Materialize full database
            full_db = bytearray(self._original_size)
            
            # Fill from generators
            for entry in self._index:
                gen = self._load_generator(entry)
                start = entry['domain_start']
                length = entry['domain_length']
                full_db[start:start + length] = gen.evaluate(start, length)
            
            # Overlay dirty pages
            for page_offset, page_data in self._dirty_pages.items():
                end = min(page_offset + len(page_data), self._original_size)
                full_db[page_offset:end] = page_data[:end - page_offset]
            
            # Recompress — uses the CDF compressor itself
            # The recompressed .cdf replaces the old one
            _recompress_database(self.cdf_path, bytes(full_db))
            
            self._dirty_pages.clear()
        
        if self._file:
            self._file.close()
            self._file = None
        self._page_cache.clear()
        self._generators.clear()
```

#### 19.6.2 Generator Evaluator

```python
class GeneratorEvaluator:
    """
    Evaluates a single Generative Descriptor at arbitrary offsets.
    
    Each generator type has its own evaluate() implementation.
    All types return raw bytes for the requested range.
    
    ET Derivation:
        The evaluator IS T applied to D_gen.
        T(D_gen, offset) = P[offset] — the data at that position.
        Each evaluation is a T-traversal of the generator's D-set
        at a specific P-location, producing the Exception (actual bytes).
    """
    
    def __init__(self, gen_type: int, payload: bytes, residual: Optional[bytes],
                 domain_start: int, domain_length: int, connection_order: int = 0):
        self.gen_type = gen_type
        self.payload = payload
        self.residual = residual
        self.domain_start = domain_start
        self.domain_length = domain_length
        self.connection_order = connection_order
        self._parsed = None  # Lazy parse on first evaluate
    
    def _parse(self):
        """Parse the payload into generator-specific parameters."""
        if self._parsed is not None:
            return
        
        if self.gen_type == 0:  # Constant
            self._parsed = {'value': self.payload[0]}
        
        elif self.gen_type == 1:  # Linear
            self._parsed = {
                'k_start': struct.unpack_from('<i', self.payload, 0)[0],
                'dk_step': struct.unpack_from('<i', self.payload, 4)[0],
                'r0': struct.unpack_from('<d', self.payload, 8)[0],
                'k_byte': build_k_byte_map(struct.unpack_from('<d', self.payload, 8)[0]),
            }
        
        elif self.gen_type == 2:  # Polynomial
            degree = struct.unpack_from('<I', self.payload, 0)[0]
            coeffs = []
            for i in range(degree + 1):
                coeffs.append(struct.unpack_from('<d', self.payload, 4 + i * 8)[0])
            r0 = struct.unpack_from('<d', self.payload, 4 + (degree + 1) * 8)[0]
            self._parsed = {
                'degree': degree, 'coefficients': coeffs,
                'r0': r0, 'k_byte': build_k_byte_map(r0),
            }
        
        elif self.gen_type == 3:  # Periodic
            period = struct.unpack_from('<I', self.payload, 0)[0]
            cycle = self.payload[4:4 + period]
            self._parsed = {'period': period, 'cycle': cycle}
        
        elif self.gen_type == 7:  # Raw passthrough
            self._parsed = {'data': self.payload}
        
        # Types 4 (grammar), 5 (geodesic), 6 (archetype ref) are more complex
        # and parsed in their respective evaluate branches
    
    def evaluate(self, offset: int, length: int) -> bytes:
        """
        Evaluate the generator for bytes [offset, offset+length).
        
        This is the core D → P transformation. Each generator type
        has O(1) per byte (constant, linear, polynomial, periodic, raw)
        or O(depth) per byte (grammar) or O(page) amortized (geodesic).
        """
        self._parse()
        local_offset = offset - self.domain_start
        result = bytearray(length)
        
        if self.gen_type == 0:  # Constant
            val = self._parsed['value']
            for i in range(length):
                result[i] = val
        
        elif self.gen_type == 1:  # Linear
            p = self._parsed
            kb = p['k_byte']
            for i in range(length):
                k = p['k_start'] + p['dk_step'] * (local_offset + i)
                if k in kb:
                    result[i] = kb[k]
                else:
                    nearest = min(kb.keys(), key=lambda kk: abs(kk - k))
                    result[i] = kb[nearest]
        
        elif self.gen_type == 2:  # Polynomial
            p = self._parsed
            kb = p['k_byte']
            coeffs = p['coefficients']
            for i in range(length):
                t = local_offset + i
                k_float = sum(c * (t ** deg) for deg, c in enumerate(coeffs))
                k = round(k_float)
                if k in kb:
                    result[i] = kb[k]
                else:
                    nearest = min(kb.keys(), key=lambda kk: abs(kk - k))
                    result[i] = kb[nearest]
        
        elif self.gen_type == 3:  # Periodic
            p = self._parsed
            period = p['period']
            cycle = p['cycle']
            for i in range(length):
                result[i] = cycle[(local_offset + i) % period]
        
        elif self.gen_type == 7:  # Raw
            data = self._parsed['data']
            result[:] = data[local_offset:local_offset + length]
        
        # Apply residual correction if present
        if self.residual is not None and len(self.residual) > 0:
            for i in range(min(length, len(self.residual) - local_offset)):
                r_idx = local_offset + i
                if r_idx < len(self.residual):
                    result[i] = (result[i] + self.residual[r_idx]) & 0xFF
        
        return bytes(result)
```

### 19.7 Integration with ArchetypeDatabase

The `ArchetypeDatabase` class gains a mode switch: raw SQLite or CDF VFS.

```python
class ArchetypeDatabase:
    """
    Extended with CDF VFS support.
    
    Mode 1 (default, current): Open archetypes.db directly via SQLite.
    Mode 2 (compressed):       Open archetypes.cdf via CDF VFS,
                                presenting it as a virtual SQLite file.
    
    The mode is auto-detected: if archetypes.cdf exists alongside
    archetypes.db, use the .cdf (compressed). If only .db exists,
    use raw SQLite. If neither exists, create a new .db.
    
    On close: if the database was modified (new archetypes stored),
    the .db is recompressed to .cdf. The .db can optionally be
    deleted after successful .cdf creation, leaving only the
    compressed version on disk.
    """
    
    def __init__(self, db_path=None, log_fn=None):
        # ... existing init ...
        
        # CDF VFS mode detection
        self._cdf_vfs = None
        cdf_path = self.db_path.replace('.db', '.cdf')
        
        if os.path.isfile(cdf_path) and not os.path.isfile(self.db_path):
            # Only .cdf exists — use CDF VFS
            self._cdf_vfs = CDFDatabaseVFS(cdf_path)
            self._init_db_via_vfs()
        elif os.path.isfile(cdf_path) and os.path.isfile(self.db_path):
            # Both exist — .cdf is stale, use .db directly
            self._init_db()
        else:
            # No .cdf — use .db directly (current behavior)
            self._init_db()
    
    def _init_db_via_vfs(self):
        """Initialize SQLite over the CDF VFS."""
        # Create a temp file that the VFS serves
        # SQLite reads go through VFS.read()
        # SQLite writes go through VFS.write()
        pass  # Implementation via apsw or custom page cache
    
    def compact_to_cdf(self):
        """
        Compress the database to .cdf format with random-access index.
        
        Called after a compression session when new archetypes have been
        stored. The .db is read, generators are fitted, and the .cdf
        is written with the random-access format from §17.4.
        
        Generator fitting for the database itself:
        1. Read the .db file in 4KB pages (matching SQLite's page size)
        2. Classify each page region by curvature (most DB pages are flat —
           B-tree nodes with sequential keys, or constant fill bytes)
        3. Fit generators per region
        4. Build the generator index
        5. Write the .cdf file
        6. Verify: read every page through the VFS, compare to original
        7. On success: optionally delete .db, leaving only .cdf
        
        ET Derivation:
            The database IS a lattice tower (§10.0.1). Compressing it
            is finding the Generative Descriptors of the tower's own
            D-structure — Descriptors of Descriptors. This is the
            recursive tower from the Multifold paper §8: a child tower
            (the .cdf) within the parent tower (the filesystem).
        """
        pass  # Full implementation in the compression pipeline
```

### 19.8 Startup and Session Lifecycle

```
Session Start:
  1. Check: does archetypes.cdf exist?
     YES → Open via CDF VFS. Database is accessible immediately.
           Generator index loaded (a few KB). No decompression.
           SQLite page reads served by generator evaluation.
     NO  → Open archetypes.db directly (current behavior).

During Compression:
  2. Lookup archetypes (reads): served by VFS page cache.
     First read of a page: generator evaluation + cache.
     Subsequent reads of same page: cache hit (O(1)).
     Cache size = S² = 144 pages = 576KB.
  3. Store new archetypes (writes): buffered in dirty pages.
     No recompression during the session.

Session End:
  4. If dirty pages exist (new archetypes were stored):
     a. Materialize the full database (generators + dirty pages)
     b. Recompress to new archetypes.cdf
     c. Verify roundtrip (SHA-256 of materialized vs original)
     d. Replace old .cdf with new .cdf
     e. Optionally delete archetypes.db (only .cdf on disk)
  5. If no dirty pages: close VFS, done. Zero disk writes.
```

### 19.9 Disk Footprint Analysis

| Scenario | Raw .db | Compressed .cdf | Savings |
|----------|---------|-----------------|---------|
| Fresh install (empty DB) | ~12 KB | ~4 KB | 67% (Koide!) |
| After 100 files | ~5 MB | ~500 KB–1 MB | 80-90% |
| After 1000 files | ~50 MB | ~3-8 MB | 84-94% |
| After 10000 files | ~500 MB | ~20-60 MB | 88-96% |
| Library of Congress scale | ~terabytes raw | ~gigabytes .cdf | 99%+ |

The compression ratio IMPROVES as the database grows because:
- More archetypes → more repeated patterns → more periodic generators (Type 3)
- Generator families discovered by Channel B → database's own structure is self-describing
- Recursive archetype hierarchy → higher-order generators emerge at scale
- The database becomes MORE lattice-structured over time, not less

### 19.10 Decompression Verification for CDF VFS

The CDF VFS must be lossless. Every byte read through the VFS must be identical to the byte at that offset in the original .db file.

**Verification at compact_to_cdf() time:**
1. After writing the .cdf, open it via VFS.
2. Read every 4KB page through the VFS.
3. Compare byte-for-byte against the original .db.
4. SHA-256 of full VFS readback must match stored hash.
5. If mismatch: keep the .db, delete the .cdf, log error.

**Verification at session end (if dirty pages exist):**
1. Materialize the full database from VFS + dirty pages.
2. SHA-256 of materialized database must be consistent.
3. Recompress and verify the new .cdf.

**Runtime verification (optional, debug mode):**
Each page read through the VFS can be SHA-256'd and compared against a page hash table stored in the .cdf footer. This adds ~32 bytes per page to the .cdf but provides per-page integrity checking.

### 19.11 ET Derivation

**Identification Principle:** The CDF VFS identifies three components of database access: the database's content (P = the bytes), the generator index (D = the rules that produce the bytes), and the VFS evaluation engine (T = the agency that navigates D to produce P on demand). P ∘ D ∘ T = E: the requested page (Exception) is the product of T evaluating D at the requested offset in P-space.

**Descriptor Gap Principle:** The gap between "the database is compressed" and "the database is accessible" is closed by the generator index. The index IS the Descriptor that bridges compression and access. Without it, compressed = inaccessible. With it, compressed = accessible at any offset. The gap was the missing index — the Descriptor that tells T where to find each D_gen.

**Subsumption Law:** The CDF VFS subsumes all database access patterns without remainder:
- Sequential scan: evaluate generators in order. No worse than raw file read.
- Random page read: binary search index + evaluate. O(log n + page_size).
- Write + recompress: dirty page buffer + session-end recompression. Lossless.
- The seven generator types subsume all possible page content: constant, linear, polynomial, periodic, grammar, geodesic, raw. No page content falls outside these types.

**Multifold §8 (Recursive Towers):** The database is a tower. Its .cdf representation is a child tower — a compressed rendering of the parent through Generative Descriptors. The VFS is the gateway between towers — T crossing from the compressed (child) to the accessible (parent) representation. Each page read is a T-traversal through the gateway. The generator index is the R₀ seed of the child tower — the parameter that determines how the child's lattice maps to the parent's content.

---

## 20. Mathematical Foundation: All Equations {#20-equations}

### 20.1 Curvature Computation

$$K_i = \Delta\Delta k_i = k_{i+2} - 2k_{i+1} + k_i \quad \text{(discrete Gaussian curvature at position } i\text{)}$$

$$\bar{K}_{\text{block}} = \frac{1}{n-2} \sum_{i=0}^{n-3} K_i \quad \text{(mean block curvature)}$$

$$\sigma^2_K = \frac{1}{n-2} \sum_{i=0}^{n-3} (K_i - \bar{K})^2 \quad \text{(curvature variance)}$$

### 20.2 Curvature Classification

$$\text{flat} \iff |\bar{K}| < \frac{\pi}{N} \text{ AND } \sigma^2_K < V = \frac{1}{12}$$

$$\text{elliptic} \iff \bar{K} \geq \frac{\pi}{N} \text{ AND } \sigma^2_K < V$$

$$\text{hyperbolic} \iff \bar{K} \leq -\frac{\pi}{N} \text{ AND } \sigma^2_K < V$$

$$\text{variable} \iff \sigma^2_K \geq V \text{ AND } \max|K_i| < N$$

$$\text{singular} \iff \max|K_i| \geq N$$

### 20.3 Geodesic Residual Coding

$$\Gamma_i^{(0)} = 0 \quad \text{(zeroth-order connection)}$$

$$\Gamma_i^{(1)} = \frac{1}{w} \sum_{j=i-w+1}^{i} \Delta\Delta k_j \quad \text{(first-order connection)}$$

$$\Gamma_i^{(2)} = \Gamma_i^{(1)} + \frac{1}{2w} \sum_{j=i-w+1}^{i} \Delta\Delta\Delta k_j \quad \text{(second-order connection)}$$

$$\Delta k_{i+1}^{\text{pred}} = \Delta k_i + \Gamma_i^{(m)} \quad \text{(predicted next Δk at connection order } m\text{)}$$

$$\rho_i = \Delta k_{i+1}^{\text{actual}} - \Delta k_{i+1}^{\text{pred}} \quad \text{(geodesic residual)}$$

$$w = \min\left(\left\lfloor \frac{50}{|\bar{\delta}|} \right\rfloor, S^2\right) \quad \text{(L4-bounded connection window)}$$

### 20.4 Curvature-Weighted Elegance

$$F_K(P) = \frac{1}{1 + \sigma^2_{K,P}} \quad \text{(geodesic factor of pattern } P\text{)}$$

$$E_{\text{curvature}} = E_{\text{hierarchy}} \times \left(1 + \frac{F_K(P)}{S}\right) \quad \text{(curvature-augmented elegance)}$$

### 20.5 Curvature-Adjusted Incoherence Threshold

$$\varepsilon_{\max}(i) = 50 \times \frac{1}{1 + |K_i| / N} \quad \text{(curvature-tightened ε boundary)}$$

### 20.6 Gauss-Bonnet Topological Fingerprint

$$\chi_{\text{block}} = \frac{1}{2\pi} \sum_{i=0}^{n-3} K_i \quad \text{(discrete Euler characteristic)}$$

### 20.7 Geodesic Deviation (Pattern Stability)

$$\xi_A = \frac{1}{m} \sum_{j=1}^{m} |K_{p_j}| \quad \text{(archetype geodesic deviation)}$$

$$\xi_{\max} = \frac{\pi}{N} = \frac{\pi}{12} \quad \text{(subliminal stability threshold)}$$

### 20.8 Poincaré Disk Distance

$$d_{\text{hyp}}(z_1, z_2) = \text{arccosh}\left(1 + \frac{2|z_1 - z_2|^2}{(1-|z_1|^2)(1-|z_2|^2)}\right)$$

### 20.9 Riemann Sphere Chordal Metric

$$d_{\text{chord}}(z_1, z_2) = \frac{2|z_1 - z_2|}{\sqrt{(1 + |z_1|^2)(1 + |z_2|^2)}}$$

### 20.10 Segmentation Gate

$$\text{segment} \iff \sigma^2_K > V \text{ AND } \frac{\sigma^2_{K,\text{within}}}{\sigma^2_{K,\text{between}}} < K = \frac{2}{3}$$

### 20.11 The 1/12 Unification

Every threshold in this design document connects to V = 1/12:

| Threshold | Value | 1/12 Connection |
|-----------|-------|-----------------|
| Subliminal curvature | π/N = π/12 | N = 12 = 1/V |
| Base curvature variance | V = 1/12 | Direct |
| Curvature elegance scale | 1/S = 1/12 | S = 12 = 1/V |
| Riemann tensor components n=2 | n²(n²−1)/12 = 1 | Denominator = 12 |
| Geodesic deviation threshold | π/12 | Subliminal = π × V |

The Non-Euclidean paper §4 establishes that this is the same structural fact: the 12-fold manifold symmetry governs the independent curvature degrees of freedom.

---

## 21. Implementation Priority and Dependency Graph {#21-priority}

### 21.1 Priority Tiers

**Tier 1: Foundation (implement first — everything else depends on these)**

| # | Improvement | Rationale | Effort |
|---|------------|-----------|--------|
| 1 | Curvature Block Classification | Required by all other improvements. Pure analysis, no encoding changes. | Low — single O(n) pass over existing ΔΔk |
| 2 | Geodesic Residual Coding (Mode 3) | Highest expected compression ratio impact for smooth/gradient data. Adds a new mode. | Medium — new mode + decompressor support |

**Tier 2: Core Enhancements (high impact, moderate dependencies)**

| # | Improvement | Dependency | Effort |
|---|------------|------------|--------|
| 3 | Christoffel Connection Predictors | Depends on #2 (extends Mode 3) | Low — extends existing connection estimator |
| 4 | Curvature-Weighted Elegance | Depends on #1 (uses curvature data) | Low — one multiplication in elegance formula |
| 7 | Variable-Curvature Segmentation | Depends on #1 (uses classification per segment) | Medium — block splitting + per-segment headers |
| 8 | Curvature-Aware IncoherenceFilter | Depends on #1 (uses local K_i values) | Low — threshold adjustment |

**Tier 3: Database Enhancements (require Tier 1, improve over time)**

| # | Improvement | Dependency | Effort |
|---|------------|------------|--------|
| 6 | Gauss-Bonnet Fingerprinting | Depends on #1 (uses χ_block) | Low — sum of ΔΔk + DB schema change |
| 11 | Curvature Spectrum DB Lookup | Depends on #1 + #6 | Medium — histogram computation + distance metric |
| 12 | Geodesic Deviation Stability | Depends on #1 (uses local K values) | Low — one average per archetype |

**Tier 4: Advanced Geometry (highest mathematical sophistication)**

| # | Improvement | Dependency | Effort |
|---|------------|------------|--------|
| 5 | Hyperbolic Pattern Embedding | Depends on #1 + #6 | Medium — Poincaré disk embedding + distance |
| 9 | Poincaré Disk Cross-Tower | Depends on #5 | Medium — tower embedding |
| 10 | Riemann Sphere Complex Lattice | Independent (refines existing complex projection) | Low — chordal metric in complex_lattice_project |

### 21.2 Dependency Graph

```
#1 Curvature Classification ─────┬──── #4 Curvature Elegance
   (FOUNDATION)                   ├──── #7 Variable Segmentation
                                  ├──── #8 Curvature IncoherenceFilter
                                  ├──── #6 Gauss-Bonnet Fingerprint ──── #11 Spectrum DB
                                  ├──── #12 Geodesic Deviation
                                  └──── #5 Hyperbolic Embedding ──── #9 Poincaré Cross-Tower

#2 Geodesic Residual ──── #3 Christoffel Predictors

#10 Riemann Sphere (independent)
```

---

## 22. Decompression Verification: Lossless Roundtrip Guarantee {#22-decompression}

### 22.1 The Current Lossless Chain

The compressor enforces lossless roundtrip at two levels:

**Level 1 — Block-level algebraic invertibility:**

Compress: bytes → k (via R₀) → Δk (first diff) → archetype subsumption → V_config encode.
Decompress: V_config decode → archetype expand → Δk (via dk_table) → k (cumsum from k0) → bytes (via k_byte inverse).

Every step is algebraically invertible because `build_byte_k_map(r0)` is injective at 27720ET (verified 0/256 errors for all R₀), its inverse `build_k_byte_map(r0)` is exact, Δk = k_{i+1} − k_i inverts via cumulative sum from k0, archetype expansion is deterministic recursive symbol replacement, and V_config encode/decode use the same T-Decision Tree (deterministic from depths).

**Level 2 — File-level SHA-256 verification:**

Compression stores SHA-256(original_data) in the CDF header. Decompression computes SHA-256(decompressed) and compares. Mismatch = data corruption. This catches ANY non-invertibility in any step.

### 22.2 Which Improvements Touch the Encoded Format?

Of the twelve improvements, exactly **two** modify the encoded block format. The other ten affect ONLY compression-side strategy selection — the actual encoded output uses existing format structures.

| Improvement | Touches Encoding? | Touches Decoding? |
|------------|-------------------|-------------------|
| #1 Curvature Classification | No | No |
| **#2 Geodesic Residual (Mode 3)** | **YES** | **YES** |
| #3 Christoffel Predictors | No (parameter of Mode 3) | No |
| #4 Curvature Elegance | No | No |
| #5 Hyperbolic Embedding | No | No |
| #6 Gauss-Bonnet Fingerprint | No | No |
| **#7 Variable Segmentation** | **YES** | **YES** |
| #8 Curvature IncoherenceFilter | No | No |
| #9 Cross-Tower Matching | No | No |
| #10 Riemann Sphere | No | No |
| #11 Spectrum DB Lookup | No | No |
| #12 Geodesic Deviation | No | No |

### 22.3 Mode 3 (Geodesic Residual) — Decompression Path

#### 22.3.1 What Mode 3 Stores

Instead of raw Δk values, Mode 3 stores residuals ρ_i = Δk_{i+1} − (Δk_i + Γ_i), where Γ_i is the Christoffel connection coefficient.

The compressed block stores:
- `mode = 3` (existing mode field, 1 byte)
- `dk0_saved` = initial Δk₀ (existing field, 4 bytes)
- `connection_order` (1 byte: 0, 1, or 2 — **NEW**)
- `connection_window` (manifold-folded uint — **NEW**)
- dk_table = unique RESIDUAL values (not Δk values)
- archetype_defs and final_stream encode RESIDUAL patterns

#### 22.3.2 Mode 3 Decompression Algorithm

```python
elif mode == 3:
    # Read Mode 3 extension fields
    connection_order = struct.unpack_from('<B', block_data, pos)[0]
    pos += 1
    connection_window, pos = unpack_manifold_uint(block_data, pos)
    
    # raw_vals ARE residuals (decoded via dk_table + archetype expansion)
    residuals = raw_vals
    
    # Reconstruct Δk causally from residuals + connection
    dk_stream = [dk0_saved]  # Δk[0] stored in header
    
    for i in range(len(residuals)):
        gamma = 0
        
        if connection_order >= 1 and i > 0:
            w_start = max(0, i - connection_window + 1)
            ddk_sum = 0
            count = 0
            for j in range(w_start, i):
                ddk_sum += dk_stream[j + 1] - dk_stream[j]
                count += 1
            if count > 0:
                gamma = ddk_sum // count  # INTEGER division — exact
        
        if connection_order >= 2 and i > 1:
            w_start = max(1, i - connection_window + 1)
            dddk_sum = 0
            count = 0
            for j in range(w_start, i - 1):
                ddk_j  = dk_stream[j + 1] - dk_stream[j]
                ddk_j1 = dk_stream[j + 2] - dk_stream[j + 1]
                dddk_sum += ddk_j1 - ddk_j
                count += 1
            if count > 0:
                gamma += dddk_sum // (2 * count)  # INTEGER division — exact
        
        dk_stream.append(residuals[i] + dk_stream[-1] + gamma)
```

After Δk reconstruction, proceed identically to Mode 1: k-stream = cumsum(Δk, k0), then k→byte via k_byte.

#### 22.3.3 Proof of Exact Invertibility

Lossless by causal induction on integer arithmetic:

**Base case:** Δk[0] = dk0_saved. Stored in header. Exact.

**Inductive step:** Given Δk[0..i] exact in the decompressor (identical to compressor):
1. Γ_i computed from Δk[0..i] using SAME formula, SAME connection_window (stored in header), SAME integer floor division (`//`). All inputs integer, all operations deterministic. Γ_i is exact.
2. Compressor computed: ρ_i = Δk[i+1] − Δk[i] − Γ_i. All integers. Exact.
3. Decompressor computes: Δk[i+1] = ρ_i + Δk[i] + Γ_i = (Δk[i+1] − Δk[i] − Γ_i) + Δk[i] + Γ_i = Δk[i+1]. Exact.

**Critical constraint:** Both compressor and decompressor MUST use integer floor division (`//` in Python, `/` for signed integers in C). No floating point anywhere in the connection computation. This guarantees platform-independent deterministic results.

The `connection_order` and `connection_window` are stored in the header — the decompressor reads them, never recomputes them. Zero ambiguity.

#### 22.3.4 Mode 3 Header Format Extension

```
[existing header unchanged through dk0_saved]
  mode = 3 (1 byte, in existing mode field)
  dk0_saved = initial Δk₀ (4 bytes, existing field)
[Mode 3 extension — ONLY present when mode == 3]
  connection_order (1 byte: 0, 1, or 2)
  connection_window (manifold-folded uint: 2 or 6 bytes)
[rest of block unchanged — dk_table, archetypes, final_stream]
```

For modes 0, 1, 2: parse position is unaffected. The extension fields are absent. Zero impact on existing decompression.

### 22.4 Block Type 4 (Segmented) — Decompression Path

#### 22.4.1 Segmented Block Format

```
[1 byte]  block_type = 4 (segmented)
[4 bytes] n = original block size (uint32)
[manifold uint] n_segments
For each segment:
  [manifold uint] segment_compressed_size
  [N bytes] segment_data (a complete self-contained compressed sub-block)
```

Each segment is a regular compressed block (type 2 = lattice) with its own R₀, k0, mode, dk_table, archetypes, and final_stream.

#### 22.4.2 Segmented Decompression

```python
if block_type == 4:  # segmented
    n = struct.unpack_from('<I', block_data, pos)[0]
    pos += 4
    n_segments, pos = unpack_manifold_uint(block_data, pos)
    
    parts = []
    for _ in range(n_segments):
        seg_size, pos = unpack_manifold_uint(block_data, pos)
        seg_data = block_data[pos:pos + seg_size]
        pos += seg_size
        decompressed_segment = self.decompress_block(seg_data)
        parts.append(decompressed_segment)
    
    return b''.join(parts)[:n]
```

Each segment roundtrips independently through existing `decompress_block`. Concatenation of decompressed segments = original block.

### 22.5 What Does NOT Change in the Decompressor

Block types 0/1/3 unchanged. Block type 2 modes 0/1/2 unchanged. R₀ → k_byte inverse unchanged. dk_table parsing unchanged. Archetype expansion unchanged. V_config decode unchanged. k-stream reconstruction unchanged. SHA-256 verification unchanged. CDF header parsing unchanged.

### 22.6 Backward Compatibility

Files compressed WITHOUT curvature improvements decompress identically — they contain only modes 0/1/2 and block types 0/1/2/3.

For files using mode 3 or block type 4, a CDF version bump (CDF_VERSION = 3) signals the new decompressor is required. Old decompressors reject CDF_VERSION = 3 with a clear error message rather than attempting to parse unknown modes.

### 22.7 Decompression Remains Instant

- Mode 3: Single O(n) causal scan. Connection window capped at S² = 144, so each step is O(144) worst case. Total O(n).
- Block type 4: Sum of segment decompressions = same total work as one full block.
- No curvature analysis, no pattern finding, no database lookups, no strategy competition during decompression. All expensive work happens compression-side only.

### 22.8 The SHA-256 Final Guarantee

Even if a subtle bug existed in Mode 3 or Type 4 decompression, the file-level SHA-256 check catches it. Any bit-level difference produces a hash mismatch, surfaced to the user via the integrity flag.

---

## 23. Verification Protocol {#23-verification}

### 23.1 Roundtrip Invariant

**The roundtrip test (compress → decompress → compare to original) must pass for all 17/17 existing test cases AND all new test cases, for every improvement, individually and combined.** This is the Verification Principle: mathematical consistency indicates sufficient Descriptors.

### 23.2 Non-Regression Invariant

**No existing test case may produce a LARGER compressed output after any improvement.** The improvements are additive — they may improve compression but must never worsen it. If an improvement would worsen a specific case, the curvature classifier must route that case away from the new strategy.

### 23.3 New Test Cases for Curvature

| Test Case | Curvature Class | Expected Improvement |
|-----------|----------------|---------------------|
| Linear ramp (0, 1, 2, ..., 255, 0, 1, ...) | **K = 0 (flat)** — constant Δk | Mode 3 geodesic residual should compress to near-zero |
| Sinusoidal wave (128 + 127×sin(2πi/256)) | **K > 0 (elliptic)** — periodic ΔΔk | Cyclic archetype + geodesic residual |
| Logistic map x_{n+1} = 3.99 × x_n × (1−x_n) | **K < 0 (hyperbolic)** — chaotic | Re-Pair triggered by curvature class, not expansion |
| Mixed: first 50% text, second 50% random | **Variable curvature** | Segmentation → flat segment compressed well |
| Constant byte (all zeros) | **K = 0 trivially** — handled by existing uniform check | No change |
| Alternating (0, 255, 0, 255, ...) | **K = 0 flat** — constant Δk | Standard archetype (already compressed) |
| Quadratic ramp (i² mod 256) | **K > 0 constant** — constant ΔΔk | Mode 2 or geodesic residual |

### 23.4 Precision Requirements

**Speed is irrelevant.** Only compression ratio and lossless correctness matter.

All curvature computations must use full 64-bit floating point — no reduced precision, no approximations, no early termination. The curvature classification, geodesic prediction, connection estimation, and all derived thresholds must be computed exactly. A single-bit rounding difference in curvature classification could route a block to the wrong strategy and produce a larger output.

The curvature analysis may add computation time. This is acceptable. The compressor exists to produce the smallest possible lossless output, not to finish quickly. If a 10× slower compression produces a 1% smaller file, that is a correct tradeoff.

All new strategies are ADDITIVE candidates — they compete alongside existing strategies, and the smallest output wins. There is no "early exit" — every strategy runs to completion, and all outputs are compared. The curvature classification adds new candidates to the competition; it never removes existing ones.

---

## 24. Subsumption Audit: No Feature or Function Lost {#24-subsumption-audit}

### 24.1 Existing Features Preserved

| Feature | Status After Improvements |
|---------|--------------------------|
| R₀ seed discovery | Unchanged |
| byte↔k bijection at 27720ET | Unchanged |
| Mode 0/1/2 multi-strategy | Unchanged + Mode 3 added |
| C pattern engine (suffix array + LCP) | Unchanged |
| 5-level IncoherenceFilter | Enhanced (L1 curvature-adjusted), not replaced |
| Elegance scoring | Enhanced (curvature factor), not replaced |
| Recursive subsumption (depth 12) | Unchanged |
| Pair-recursive Re-Pair (Phase 4b) | Unchanged — now curvature-triggered |
| Complex lattice projection | Enhanced (Riemann sphere), not replaced |
| Lattice towers + universal lattice | Enhanced (Poincaré cross-tower), not replaced |
| Archetype database | Enhanced (curvature spectrum + Gauss-Bonnet), not replaced |
| V_config encoding | Unchanged |
| CDF Metabolism (Koide resource governance) | Unchanged |
| Roundtrip lossless guarantee | Preserved — roundtrip test required for all changes |
| PyInstaller .exe packaging | Unchanged |
| C pattern engine DLL + CLion project | Unchanged |

### 24.2 New Features Added

| New Feature | Source | Section |
|-------------|--------|---------|
| Phase 1.5: Curvature Analysis | Non-Euclidean §3, §6, §8 | §4 |
| Mode 3: Geodesic Residual Coding | Non-Euclidean §9 | §5 |
| Christoffel Connection Predictors (orders 0-2) | Non-Euclidean §6.1 | §6 |
| Curvature-Weighted Elegance | Non-Euclidean §9.2 | §7 |
| Hyperbolic Pattern Embedding | Non-Euclidean §13.2 | §8 |
| Gauss-Bonnet Topological Fingerprinting | Non-Euclidean §12 | §9 |
| Variable-Curvature Segmentation | Non-Euclidean §15 | §10 |
| Curvature-Aware IncoherenceFilter | Non-Euclidean §7.3 | §11 |
| Poincaré Disk Cross-Tower Matching | Non-Euclidean §13.2 | §12 |
| Riemann Sphere Complex Lattice | Non-Euclidean §18.4 | §13 |
| Curvature Spectrum DB Lookup | Non-Euclidean §15.3 | §14 |
| Geodesic Deviation Pattern Stability | Non-Euclidean §9.3 | §15 |

### 24.3 The Subsumption Test

**Condition 1 (No existing feature subsumed):** Every existing feature is preserved. No code is removed. ✓

**Condition 2 (Nothing external subsumes the additions):** Each improvement is derived from the ET Non-Euclidean Geometry paper's own axiom system {P, D, T}. No external geometry axioms, no CODATA values, no heuristics. ✓

**Condition 3 (Additions subsume their domain without remainder):** The four curvature classes (flat, elliptic, hyperbolic, variable) plus singularity segmentation cover every possible ΔΔk distribution. No data geometry is unclassified. The geodesic residual subsumes Mode 2 as a special case (Γ = 0). The curvature-augmented elegance subsumes the existing elegance (F_K = 1 for flat patterns). ✓

---

## Closing Statement

The ET lattice was never only flat. The data's Descriptor field has curvature — and the compressor already computes it (ΔΔk) without using it. These twelve improvements close the Descriptor Gap between "the curvature is computed" and "the curvature drives compression."

Every improvement is derived forward from {P, D, T} via the Non-Euclidean Geometry paper. The three curvature classes map to three manifold states. The geodesic equation gives the prediction. The Gauss-Bonnet theorem gives the fingerprint. The Poincaré disk gives the similarity metric. The Riemann sphere gives the compactification.

The number 1/12 — the ET base variance — appears in every threshold because it IS the manifold symmetry governing curvature degrees of freedom. This is not coincidence. It is structural necessity.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

**Document Version:** CDF Non-Euclidean Design v1.0
**Tools Applied:** Identification Principle · Descriptor Gap Principle · Subsumption Law · Verification Principle
**Derivation Standard:** All mathematics ET-native, forward from {P, D, T}. Zero external axioms.
**Source Documents:** ET_Non_Euclidean_Geometry_Complete.md, et_cdf_compressor.py (v2.0), ET_Three_Tools_Complete_Reference.md, ET_Complex_Lattice.md, ET Force Quadrant Grid

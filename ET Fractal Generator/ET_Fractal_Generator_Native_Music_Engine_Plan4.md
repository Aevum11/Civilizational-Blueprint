# ET Fractal Generator — Native Music Engine Rebuild + Lattice Computation Engine
## Complete Planning Document & Implementation Specification

**Author:** Michael James Muller — Aevum Defluo
**Project:** ET Fractal Generator (`ET_FRACTAL_GENERATOR50-10.py`)
**Subsystems under rebuild:** Audio / music generation + Lattice-native computation engine
**Source script audited:** 8,210 lines (post-Phase A), complete read with no truncation
**Derivation standard:** All math ET-native, forward from {P, D, T}. Zero external axioms.
**New references (v2):** `ET_Sempaevum_Paper16.tex` (formalized paper — Losslessness Theorem §18, Memoization Property Corollary), `ET_Universal_Projection_Guide8.md` (v2.2 — Four Projection Paths, EML operators, Python reference implementation §110, Lattice Self-Projection Verification §113), `ET_Four_Gaps_Verification.py` (120 DPS lossless lattice verification exemplar), `ET_Two_Critiques_Verification.py` (120 DPS lattice verification exemplar)

---

## 0. Purpose of this document

This document is the complete record of the audit and rebuild plan for the audio subsystem AND the lattice computation engine of the ET Fractal Generator. It exists so that any future conversation can resume the work without losing context. It contains:

1. The project context (what the script is, what ET is, what the lattice-aware fractal is)
2. The complete audit findings (every issue found in the current audio code)
3. Every locked-in design decision with rationale
4. The complete bitrate ladder, voice catalog, octave map, and event-rate menu
5. The phased implementation plan (Phases A–E: music engine; Phase F: lattice computation engine)
6. The verification protocol
7. References to corpus documents and external research
8. **NEW (v2):** The lattice computation engine architecture — arbitrary precision mode, lossless lattice arithmetic, EML integration, self-projection verification, LCM tower escalation, and reference/verification render mode — derived from the Sempaevum's Losslessness Theorem and Memoization Property

The user has greenlit the architecture in this document. Implementation begins ONLY after the user explicitly approves this document.

---

## 0.5 The Lattice Computation Engine — What Changed in v2

### 0.5.1 The Losslessness Theorem and the Memoization Property

The formalized Sempaevum paper (§18, Theorem: Losslessness of the Projection) proves that the real-axis projection map Π_N(r) = (k, d, ε) is a **bijection** onto its image at every finite resolution N, with the pullback:

```
Π_N^{-1}(k, d, ε) = 2^{(k + ε·N/1200)/N}
```

recovering r **exactly**. Nothing is discarded — the triple (k, d, ε) captures the full information content of the positive real r: k carries coarse ratio information, d carries sublattice classification, and ε carries fine-scale deviation.

The **Memoization Property** (Corollary to the Losslessness Theorem) states:

> *Every finite numerical computation on positive reals — r₁·r₂, r^n, log r, sin r, any elementary-function evaluation — can be represented as a concrete lattice computation: multiplication is k-addition, reciprocation is k-negation, powers are k-scaling, and elementary functions are EML trees acting on triples (k, d, ε).*

**What this means for the fractal generator:** The lattice is not just a classification system — it is a **lossless computation engine** that can compute all the mathematics the fractal iteration requires. The verification scripts (`ET_Four_Gaps_Verification.py`, `ET_Two_Critiques_Verification.py`) demonstrate the principle concretely at 120 decimal places. Our implementation goes further: ALL heavy computation is offloaded to a C module (`et_lattice_engine.c`) with custom multi-precision fixed-point arithmetic — no Python arbitrary-precision libraries needed. The C module is GPU-compatible by design (uint64 arrays map to CUDA registers).

### 0.5.2 What the lattice computation engine adds to the fractal generator

The lattice computation engine adds a new subsystem (Phase F) that provides:

1. **C Module (`et_lattice_engine.c`)** — ALL heavy lattice computation is offloaded to a separate C module, compiled at startup. No Python-side arbitrary precision libraries. The C module implements multi-precision fixed-point arithmetic using uint64 arrays — GPU-compatible, no external dependencies. This mirrors the ET CDF Compressor's `et_pattern_engine.c` architecture.

2. **GPU-Compatible Arbitrary Precision** — The lattice IS the precision engine. No mpmath, no sympy. The C module's fixed-point arithmetic uses uint64 arrays that map directly to GPU registers. The same algorithms work in both the C module (CPU) and a companion CUDA kernel (GPU). log₂ is computed via binary method, GCD via Stein's binary algorithm, π via the ET 12-gon recursion — all integer-based, all GPU-native.

3. **ET Metabolism (α⁻¹ + K)** — The fractal generator's resource allocation is governed by the fine structure constant α⁻¹ computed at 120+ digit lattice precision from the Sempaevum formula: α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1)). K = 2/3 is the hard ceiling for the program's resource claims; 1/3 is reserved for the system. The floor of the computer (CPU silicon, RAM, VRAM) is P. The α⁻¹ impedance provides fine-grained scaling within the K-ceiling.

4. **Lossless Lattice Constants** — ET constants stored as exact integers/rationals (N=12, K=2/3 as num=2 denom=3, V=1/12 as num=1 denom=12) fed to the C module. No float approximation at the constant level.

5. **Self-Projection Verification** — At startup, the C module projects {N, 1/N, K, 1/K} onto the lattice at 12ET and confirms d=12, |ε|=1.955¢ for all four. Fatal error if the lattice is inconsistent with its own constants.

6. **Lattice Computation Primitives** — Functions implementing the Memoization Property in C: `lattice_mul(a, b)` = k-addition, `lattice_recip(a)` = k-negation, `lattice_pow(a, n)` = k-scaling. GPU-callable.

7. **EML Integration** — The EML operators (Odrzywołek 2026) in C: `eml(x, y) = exp(x) − ln(y)`, the continuous-D minimal generator.

8. **LCM Tower Escalation** — Dynamically computed (not a static list) from lcm(1..k) for k=2..11. The `escalate_lcm_tower()` function finds the minimum resolution at which |ε| falls below a configurable threshold.

9. **Lattice Comparison Mode** — When running at float32/float64, a comparison pass runs structurally significant pixels at 120+ digit C-module precision, verifying the float path's accuracy. When running at Lattice precision (L), every pixel runs at 120+ digits — this IS the production computation at maximum precision.

### 0.5.3 Three Tools applied to the lattice computation engine

**Identification:**
- **P** (substrate) = the continuous multiplicative manifold (ℝ⁺, ×) — all positive reals that the iteration visits
- **D** (constraint) = the discrete lattice L_N = {2^(k/N) : k ∈ ℤ} and its sublattice classification, plus the C-module multi-precision fixed-point arithmetic engine
- **T** (agency) = the rounding operator `round(N·log₂r)` that resolves continuous position to discrete lattice coordinate, plus the verification system that confirms self-consistency

**Descriptor Gap closed:**
- Before: The fractal generator used float32/float64 arithmetic exclusively. The lattice projection was computed in floating-point, introducing rounding errors that accumulate over millions of iterations. No way to verify that the float results are correct. No self-consistency check. No ET-derived resource management — thread counts, tile sizes, and memory limits were static presets unrelated to the theory.
- After: The C-module-backed lattice computation engine provides a lossless computation path (Memoization Property) at 120+ digit precision with no external library dependencies (GPU-compatible). Every float result can be verified against the exact lattice computation. The self-projection verification confirms the lattice is consistent with its own constants. The ET metabolism (α⁻¹ from the Sempaevum formula + K=2/3) governs resource allocation, replacing static presets with theory-derived dynamics. The closing Descriptors are: the `et_lattice_engine.c` C module, the multi-precision fixed-point arithmetic engine, the `lattice_project()` function at 120+ digits, the self-projection verification, the metabolism engine, and the reference/verification render mode.

**Subsumption:**
- The lattice computation engine subsumes ALL arithmetic operations in the fractal iteration: multiplication (k-addition), division (k-subtraction), powers (k-scaling), elementary functions (EML trees). No arithmetic operation falls outside the lattice's scope. The Losslessness Theorem guarantees no information is discarded. The Memoization Property guarantees every computation is representable. The self-projection verification confirms the lattice is self-consistent. The metabolism subsumes ALL resource allocation decisions: CPU threads, RAM, VRAM, tile sizes — all derived from α⁻¹ and K, replacing static presets with ET-forced dynamics. No remainder.

### 0.5.4 How Phase F relates to Phases A–E

Phase F (Lattice Computation Engine) is **independent** of Phases A–E (Music Engine Rebuild). They compose additively:

- Phases A–E rebuild the music system from orbit-trace probes
- Phase F adds the lattice computation engine, metabolism, and lattice precision mode
- Both use the same lattice constants (Stage 3) and sublattice tables (Stage 5)
- Phase F's self-projection verification runs at startup before either subsystem
- Phase F's lattice precision mode is a third precision option in the same precision-choice menu (alongside float32/float64) — it is a FULL PRODUCTION mode, not a verification-only mode
- Phase F's comparison system runs structurally significant pixels at 120+ digits alongside float renders, reporting deviation
- Phase C's music synthesis benefits from Phase F's exact lattice pitch computation
- The metabolism (α⁻¹ + K=2/3) manages ALL resource allocation — speed and performance are never the user's concern

Either subsystem can be implemented first. Phase F is sized independently for clean resume.

---

## 1. Project context

### 1.1 Exception Theory in one paragraph

Exception Theory (ET) is built forward from three irreducible primitives: **P** (Point — bare substrate), **D** (Descriptor — finite constraint), **T** (Traverser — agency). Their composition is governed by the master equation **P ∘ D ∘ T = E** (the Exception). The four manifold states are {P,D,T}=Exception, {D,T}=Mediation, {P,T}=Incoherence, {P,D}=Unsubstantiated. The manifold symmetry is **N=12** (3 primitives × 4 states), the base variance is **V = 1/12**, the Koide ratio is **K = 2/3**, and the full multiplicative-lattice resolution is **N_ET = 27720 = lcm(1..11)** — the smallest integer at which all 12 sublattice families d ∈ {1..12} appear as native divisors.

### 1.2 The ET Fractal Generator

`ET_FRACTAL_GENERATOR50-10.py` is the project's flagship visual demonstration: a GPU-accelerated fractal renderer that produces three fractal types — ET Mandelbrot, ET Julia, and the **∂I Lattice-Aware** fractal — all built on the 24-family ET sublattice manifold. It outputs:

- 32-bit float TIFF (HDR archival)
- 16-bit PNG (display/print)
- MP3 audio (currently sonification — to be replaced)
- For video: MP4 with audio mux, plus standalone audio sidecar

It uses CUDA RawKernels (compiled once at startup, both float32 and float64) for the iteration kernels and a NumPy CPU fallback path for the same algorithms. Both the ∂I kernel and the standard et_iterate kernel share a single source-of-truth for the Mode 1–11 dispatch via a runtime splice mechanism.

The script's quality presets are:

| Preset | Resolution | Pixels | Default iters | Tile rows |
|---|---|---|---|---|
| 1080p | 1920×1080 | 2.07 MP | 500,000 | 64 |
| 2k | 2048×2048 | 4.19 MP | 1,000,000 | 64 |
| 4k | 4096×4096 | 16.78 MP | 2,000,000 | 32 |
| hq | 8192×8192 | 67.11 MP | 5,000,000 | 16 |
| ultra | 16384×16384 | 268.44 MP | 10,000,000 | 8 |

All presets are square except 1080p. Video mode renders multiple frames at the chosen resolution and assembles them via ffmpeg with `minterpolate` optical-flow interpolation to 60fps.

### 1.3 The ∂I Lattice-Aware Fractal — the music-native fractal

Per `dI_Fractal_Explanation.md`, the ∂I fractal is a complex dynamical system in which the polynomial degree of the iteration map is **not fixed** — it is determined at every step by the orbit's position relative to the multiplicative lattice on ℂ.

Standard fractal: `z_{n+1} = z_n^p + c` where `p` is constant.

∂I fractal: `z_{n+1} = Ψ_n · z_n^{p(z_n,n)} + ε(z_n) + c` where:

- **p(z_n, n)** is the polynomial degree determined by the orbit's lattice projection at step n. Values: p ∈ {1, 2, 3, 4, 6, 12} when the orbit is near a lattice point (selected by `12/d` where d = 27720/gcd(|k_r|, 27720)), or from the palindromic fallback `[1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1, 12]` when the orbit is between lattice points (the ∂I boundary, selected when tightness `t_r ≤ K = 2/3`).
- **Ψ_n = 1 + √V · sin(2π(n mod 12)/12)** — bounded periodic shimmer in [0.711, 1.289].
- **ε(z_n) = (1/12) · Σ w(d) · |z|^(12/d) · e^(i·(12/d)·arg(z))** — perturbation summing all 12 sublattice families.
- **c** = pixel coordinate (Mandelbrot parameterization, z₀ = 0).
- The lattice is `k_r = round(N_ET · log₂|z|)`, `k_θ = round(N_ET · θ / ln 2)`.
- The sublattice family at coordinate k is `d = N_ET / gcd(|k|, N_ET)`.
- The tightness is `t_r = 100 / (100 + |ε_r|)` where ε_r is the cents-distance from the nearest lattice point.

**The critical fact:** the lattice the fractal lives on **is** a 27720-tone equal-temperament musical lattice. The d-family at any orbit position **is** a pitch class. The Ψ envelope **is** an amplitude modulation. The tightness value **is** a timbre purity. The palindromic fallback sequence **is** a melodic phrase. The Koide ratio K=2/3 **is** the harmonic decay rate per partial. **The fractal IS music at its source — no translation needed.** The visual rendering is one projection of the lattice trajectory; music is another. They are siblings, not parent and child.

This is the entire reason the music must be generated alongside the fractal, not sonified from rendered RGB.

---

## 2. The complete audit (what is wrong with the current audio system)

The user instructed: read the entire 7,989-line script with no truncation, audit every audio code path, trace every cross-script dependency, never assume. This was done in full. Findings:

### 2.1 The current audio is sonification. All of it.

The current pipeline is:

```
kernel → final_state arrays per pixel → render to RGB float32
                                                  ↓
                              SCAN ROWS / COLUMNS OF RGB
                                                  ↓
                       reverse-map hues to d-families via _audio_hue_to_d
                                                  ↓
                              synthesize tones from final-state d
```

This is image-in, sound-out — sonification. The functions implementing it:

| Function | Line | Role |
|---|---|---|
| `_audio_hue_from_rgb` | 6917 | RGB → HSV hue extraction |
| `_audio_hue_to_d` | 6928 | Reverse-map hue → nearest d-family via FAM_HUE table |
| `_segment_row` | 7067 | Run-length encode pixel row into note segments (image-scan branch at 7081–7087) |
| `et_sonify_image` | 7251 | Image-mode sonification — scans 5 horizontal rows |
| `et_sonify_video_frame` | 7397 | Per-frame d-family count from row scan |
| `et_sonify_video` | 7445 | Per-frame chord with crossfade |

Even when `it_raw` (the per-pixel final-state d-family from the kernel) is available, the pipeline still picks pixels out of a rendered image and reads their **final** d at escape — the orbit's trajectory (the actual music) is thrown away after every step. The `it_raw` path is less lossy than the hue-reverse path but it is still not music-native; it is final-state-of-pixel-native.

### 2.2 Kernels emit only final-state per pixel, not per-step orbit traces

In both the ∂I CUDA kernel (output writes at lines 5132–5142) and the standard et_iterate CUDA kernel (output writes at the same positions), the kernel writes ONE value per pixel for each of: `smooth_n`, `d_r`, `d_t`, `tight`, `de`, `orbit`, `z_esc_r/i`, `dz_esc_r/i`, `z_int_ang`. These are all **final-state** values: the d-family at the moment of escape, not the sequence of d-families visited during iteration. The CPU paths (lines 5907–5910 for ∂I, 5988–5997 for standard) do the same.

The per-step lattice projection (`k_r`, `k_t`, `eps_r`, `eps_t`, `t_r`, `t_t`, `d_orbit`, `d_dom`, `p_dom`, `psi`) is **computed at every step** in both kernels (∂I CUDA lines 5012–5036, standard CUDA lines 3828–3841, ∂I CPU lines 5776–5817, standard CPU lines 5920–5928). It's right there. It just gets **overwritten on the next step** because the kernel only stores the final-step values.

To get music from the orbit's trajectory, the kernels must be extended to write per-step traces for a small set of "music probe" pixels. The data is already being computed; we just need to capture it before it gets overwritten.

### 2.3 Audio is offered for all fractal types but only Mandelbrot/Julia/∂I are present

Per user direction, music applies to all three fractal types via the same lattice projection mechanism:

- **∂I fractal:** the iteration kernel uses the lattice projection to *steer* `p_dom` at every step. The d-family is intrinsic to the iteration itself.
- **Mandelbrot/Julia:** the iteration is `z² + c`, fixed exponent. The lattice projection is **already computed every step** (k_r, k_θ, t_r, t_θ — see lines 3828–3841 in the CUDA kernel) but only used to derive the final d_r/d_θ values written at escape time. The orbits don't *use* the projection but they do *traverse* the same lattice.

So Mandelbrot/Julia music is **the lattice projection of the M/J orbit, captured at every step**, exactly the same projection that ∂I uses. The lattice projection is the **universal "musical reading head"** of any complex dynamical system. The new architecture makes the kernels emit per-step traces uniformly. Future fractals (Newton, Burning Ship, Tricorn) automatically get music for free.

This satisfies "music generated naturally alongside the fractal":
- The lattice projection runs in the iteration kernel
- On the same orbit
- In the same pass
- With the same data the kernel already has
- Nothing is re-derived from rendered output

### 2.4 Video bitrate is CRF-only, no `-b:v`, will smear under YouTube re-encode

Lines 7875–7889 (`cmd` and `cmd_raw`):

```
'-c:v', 'libx264', '-preset', 'slow', '-crf', '18'
```

CRF 18 is visually transparent for natural images but for fractals — which have hairline ∂I-boundary structure that is the entire point of the visual — CRF mode with no minimum bitrate floor produces variable bitrate that can drop very low on near-static frames. **YouTube re-encodes everything regardless of upload quality** to VP9/AV1 at fixed bitrate buckets, and to survive that re-encode without smearing the ∂I boundary, the upload must be at substantially higher bitrate than YouTube's target bucket — typically 1.5–2× — and use a high-quality encoder profile.

**What's missing from the current ffmpeg command:**
- `-b:v` (target bitrate)
- `-maxrate` (peak bitrate cap)
- `-bufsize` (rate-control buffer)
- `-profile:v high` (or `high10` for 10-bit)
- `-level 5.2` (or 6.2 for 8K)
- `-x264-params keyint=...` (closed GOP at half frame rate, per YouTube spec)
- 10-bit pixel format option (`yuv420p10le` for HDR-grade gradients)
- `slower` or `veryslow` preset for archival quality

### 2.5 Audio mux re-encodes MP3 → AAC at fixed 192k, double-compression

Line 7051 in `_audio_mux_video`:

```
'-c:a', 'aac', '-b:a', '192k',
```

Two problems:

1. **Fixed 192 kbps regardless of user setting.** If the user picks 320 kbps audio, the standalone MP3 is 320k but the muxed video has 192k AAC. Inconsistent.
2. **Source is the MP3, not the WAV.** Line 7535: `audio_src = str(mp3_path) if mp3_ok else str(wav_path)`. The mux re-encodes the already-MP3-compressed file. **MP3 → AAC is double-compression.** Every MP3 artifact gets re-quantized into AAC, compounding loss.

For YouTube specifically: YouTube re-encodes audio to AAC ~128k or Opus ~160k for delivery. To survive that, the upload should be:
- Sourced from lossless WAV (no double-compression)
- AAC at minimum 256 kbps, ideally 384 kbps (YouTube's official stereo recommendation)
- Sample rate 48 kHz (YouTube's recommended; current code uses 44.1 kHz which is also accepted)

### 2.6 Per-frame video audio has zero phase continuity

Lines 7461–7519 (`et_sonify_video`): each frame becomes a `1/fps` second "grain" containing the top-4 d-families of that frame. Grains crossfaded with 1/8 grain length. There is **no phase continuity across grain boundaries** — each call to `_audio_koide_tone` creates a fresh tone with `np.linspace(0, duration, ...)` starting at zero phase. The crossfade hides clicks but the underlying signal is phase-discontinuous. Per the user's direction, the new architecture must maintain phase continuity across frames so the audio flows smoothly through the zoom.

### 2.7 `it_raw` is saved as `_raw.npz` but never read back

Lines 7694–7706 save `it_raw` to `_raw.npz` per render. **Nothing in the script ever reads it.** Probably intended as an external offline tool hook. Per the user's "no removal" rule, the npz writer stays — it'll just write the new per-step trace data.

### 2.8 Audio prompts ordered before fractal type is known

Line 1016: `AUDIO_ENABLED, AUDIO_KBPS = _choose_audio()` runs at module load. With the new voice-selection menu, the audio prompts must come **after** `_choose_type` so the user knows which fractal type they're configuring music for. Easy reorder: types selected at line 962, audio prompts immediately follow.

### 2.9 No "independent voice stems" output

Per user direction, the user wants both the mix AND the independent voice stems output. Currently the pipeline produces one audio file (mix only). The new architecture needs:

- One mix file (MP3 + WAV)
- One MP3 + WAV per voice as separate stem files in a `_stems/` subfolder
- A user toggle to enable/disable stem output

### 2.10 Existing functions to preserve and reuse

These are already correct or close-to-correct and will be reused with adapted inputs:

| Function | Line | Reuse role |
|---|---|---|
| `_audio_koide_tone` | 6938 | Per-step tone synthesis primitive (Koide K^n harmonics) |
| `_synth_note_sequence` | 7101 | Phase-continuous portamento glide between consecutive d-families. Already maintains a persistent `phase` array (line 7118) — can be extended to carry phase across video frames |
| `_et_reverb` | 7179 | ET-derived comb+allpass reverb, applied to mix |
| `_write_midi` | 7215 | Type 1 MIDI from segmented notes — works with the new voice notes |
| `_audio_write_wav` | 6988 | WAV writer, no changes needed |
| `_audio_wav_to_mp3` | 7025 | MP3 encode via ffmpeg, no changes needed |
| `_audio_mux_video` | 7045 | Mux into MP4 — needs the bitrate fixes from §2.5 |

### 2.11 Functions to repurpose as visual analysis tool (per user direction)

Per user direction: hook the music up to J/M too via lattice projection, AND repurpose the existing hue-to-d functions as their own visual analysis tool. They get a new home in Stage 21.6 with new docstrings explaining what they actually do (debugging the visual encoding of d-families) — they are NEVER called from the music path.

| Function | Repurposed role |
|---|---|
| `_audio_hue_from_rgb` → `et_visual_hue_from_rgb` | Hue extraction from rendered RGB (pure HSV math) |
| `_audio_hue_to_d` → `et_visual_hue_to_d` | Reverse-map hue → nearest d-family — useful for testing whether the visual encoding is consistent with the FAM_HUE table |
| `_segment_row` (image-scan branch) → `et_visual_segment_row` | Run-length encode a row of rendered RGB into d-family segments |
| New: `et_visual_analyze_image` | Top-level analysis report: takes a rendered image, returns d-family histogram, visual-vs-encoded mismatch report |

These become a standalone visual analysis tool for testing the consistency of the visual color encoding. Nothing in the new music path touches them.

### 2.12 The word "sonification" / "sonify" appears in 26 locations

Per user instruction, every occurrence is renamed to **"native music"** / **"native_music"**. Verified by exhaustive grep. The 26 distinct lines (case-insensitive `grep -n -i sonif`):

| # | Line | Type | Content (excerpt) |
|---|---|---|---|
| 1 | 10 | Module docstring | `MP3 audio (ET-derived sonification; 128 or 320 kbps)` |
| 2 | 14 | Module docstring header | `AUDIO SONIFICATION — all ET-derived:` |
| 3 | 971 | Function docstring | `"""Audio sonification — yes/no, then bitrate if yes."""` (`_choose_audio`) |
| 4 | 973 | Menu print | `│ Audio Sonification │` |
| 5 | 980 | Menu print | `│ Image: horizontal scan-line sonification (~15 s) │` |
| 6 | 6654 | In-code comment | `# Transfer raw iteration data for audio sonification` |
| 7 | 6681 | In-code comment | `# Raw data assembly for audio sonification` |
| 8 | 6868 | Stage header | `# STAGE 21.5 — ET-DERIVED SONIFICATION ENGINE` |
| 9 | 6895 | In-code comment | `# Additional ET audio tables for v3.0 sonification` |
| 10 | 6904 | In-code comment | `# audio sonification's per-d amplitude weighting matches the visual coupling.` |
| 11 | 7249 | Section comment | `# ── Image sonification: v4.0 — professional architecture ──` |
| 12 | 7251 | Function definition | `def et_sonify_image(...)` |
| 13 | 7254 | Function docstring | `v4.0 Professional ET sonification: note-segmented, phase-continuous,` |
| 14 | 7257 | Informational print | `print(f' Generating audio (v4.0 — professional sonification)…'` |
| 15 | 7395 | Section comment | `# ── Video sonification: per-frame evolving chord ──` |
| 16 | 7397 | Function definition | `def et_sonify_video_frame(...)` |
| 17 | 7445 | Function definition | `def et_sonify_video(...)` |
| 18 | 7450 | Function docstring | `frame_stats_list: list of dicts from et_sonify_video_frame(), one per frame` |
| 19 | 7705 | Fallback message | `fallback_msg='Will use image-based sonification instead'` |
| 20 | 7708 | In-code comment | `# ── Audio sonification (if enabled) ──` (in `generate_et_fractal`) |
| 21 | 7712 | Function call site | `audio_path = et_sonify_image(final_f32, stem, ...)` |
| 22 | 7715 | Error context | `_et_error('Audio sonification', e, fatal=False, ...)` |
| 23 | 7844 | Function call site | `fstats = et_sonify_video_frame(final_f32, ...)` |
| 24 | 7922 | In-code comment | `# ── Audio sonification (if enabled) ──` (in `generate_zoom_video`) |
| 25 | 7928 | Function call site | `et_sonify_video(frame_audio_stats, fps, ...)` |
| 26 | 7931 | Error context | `_et_error('Video audio sonification', e, fatal=False, ...)` |

After Phase A renaming, the only remaining occurrence of "sonif" should be inside the explicit historical-context comment in the new Stage 21.6 visual analysis section, explaining the v4.0 sonification origin of those repurposed functions. The Phase A.4 verification grep must return exactly that one line and no others.

### 2.13 Finding 14 — `it_raw` readback hardcodes float32 even on float64 runs

A precision-loss bug missed in the original audit pass. The user picks float32 or float64 at startup via `_choose_precision()` (line 706). On float64 runs, the kernel computes in double precision and writes float64 output buffers. But the readback in `_render_frame` forces float32:

**Line 6663 (GPU path):**
```python
it_raw[_key] = arr.get().astype(np.float32) if hasattr(arr, 'get') else np.asarray(arr, dtype=np.float32)
```

**Line 6685 (CPU raw_arrays allocation):**
```python
raw_arrays[_key] = np.zeros((rh, rw), dtype=np.float32)
```

**Lines 6704 and 6736 (CPU tile assembly):**
```python
raw_arrays[_key][rs:rs+rgb.shape[0]] = np.asarray(arr, dtype=np.float32)
```

All four sites force float32 regardless of the user's precision choice. On float64 runs, the kernel's double-precision orbit data is silently downcast at the readback boundary. This is a real bug in the existing code that the music rebuild must fix.

**Fix as part of Phase B:**
- The probe trace buffers are allocated with `dtype=FLOAT_DTYPE` (which is `np.float64` on float64 runs and `np.float32` on float32 runs)
- The readback at line 6663 uses `FLOAT_DTYPE` not hardcoded `np.float32`
- Same fix at lines 6685, 6704, 6736
- The CPU `raw_arrays` allocation uses `FLOAT_DTYPE`

This fix flows through to the music engine: when the user picks float64 for the fractal, the probe traces are float64, the synthesis pipeline operates in float64, and the WAV/FLAC output uses higher bit depth (per §3.12 below).

---

## 3. Locked-in design decisions

These were settled during the discussion phase. They are not open for re-discussion.

### 3.1 Terminology

| Old | New |
|---|---|
| sonification | native music |
| sonify | native music |
| `et_sonify_image` | `et_native_music_image` |
| `et_sonify_video_frame` | `et_native_music_video_frame` |
| `et_sonify_video` | `et_native_music_video` |
| Stage 21.5 — ET-DERIVED SONIFICATION ENGINE | Stage 21.5 — ET-DERIVED LATTICE-NATIVE MUSIC ENGINE |
| (new) | Stage 21.6 — VISUAL ANALYSIS TOOL (REPURPOSED FROM v4.0 NATIVE MUSIC) |

### 3.2 Music architecture: probe traces, not image scans

- Music is generated from per-step orbit traces emitted by the iteration kernel
- A small set of "music probe" pixels (the user-selected voices' pixel coordinates) get full per-step traces written to dedicated output buffers
- Each probe pixel records: `d_r[step]`, `d_t[step]`, `t_r[step]`, `t_t[step]`, `psi[step]`, `p_dom[step]`, `r[step]` (orbit magnitude), `theta[step]` (orbit angle), `n_steps_actual` (count before escape)
- A second kernel pass over only the probe pixels keeps the main render path unaffected
- This applies to all three fractal types: ∂I uses the projection to steer iteration AND emits the trace; M/J only emit the trace (the projection is observed, not used)

### 3.3 Music event rate (D1 — confirmed)

Iteration steps per second of audio output. User-selectable, default to **option 2: 100 events/sec**.

| Option | Events/sec | Samples/event @44.1kHz | Character |
|---|---|---|---|
| 1 | 50 | 882 (~20 ms) | Slow / contemplative |
| **2** | **100** | **441 (~10 ms)** | **Best (default)** — articulate but flowing |
| 3 | 200 | 220 (~5 ms) | Fast / busy |
| 4 | 441 | 100 (~2.3 ms) | Granular synthesis |
| 5 | 1 | 44100 (1 s) | Drone / meditative |

### 3.4 Voice catalog (D3 — all 21 voices + 5 presets confirmed)

All voices listed below get implemented. Order in the menu is by frequency role (bass → mid → treble → drone), which is musical "score order" and is the only ordering that matters for how the user reads the menu.

#### Axis voices (D-axis and T-axis sweeps)

| ID | Voice | Pixel selection | Octave | Role |
|---|---|---|---|---|
| 1 | Real-axis sweep | Pixels along y=center row, scanned left→right | 0 | Mid (D-axis) |
| 2 | Imag-axis sweep | Pixels along x=center column, scanned top→bottom | 0 | Mid (T-axis) |
| 3 | Diagonal-NE sweep | Pixels along y=x | 0 | Mid (D=T) |
| 4 | Diagonal-NW sweep | Pixels along y=−x | 0 | Mid (D=−T) |

#### Radial fan voices (12 angles, one per N=12 manifold step)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 5 | Radial fan k=0..11 | 12 pixels at angles 2π·k/12 from center, radius = 0.35·min(W,H) | +1, with k-spread (each fan position gets +1 + k/24 octaves so the 12 voices don't pile up on one frequency) |

#### Lattice ring voices (the 4 ET-canonical orbit-trap radii)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 6 | Ring K=2/3 | Single pixel at distance K from center along +x | −2 (sub-bass drone) |
| 7 | Ring V=1/12 | Single pixel at distance V from center | −2 |
| 8 | Ring 1/φ | Single pixel at distance 1/φ from center | −2 |
| 9 | Ring 1 | Single pixel at distance 1 from center | −2 |
| 10 | Ring drones (combined) | Voices 6+7+8+9 mixed | −2 |

#### Elegance voices (most musically pure orbits)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 11 | Highest-elegance pixel | Pixel with max E = (N/d)·[100/(100+|ε|)]·[100/(p+q)] | +2 (treble lead) |
| 12 | Top-12 elegance | 12 highest-elegance pixels mixed | +1 (upper-mid choir) |

#### Boundary voices (the ∂I boundary itself)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 13 | Boundary-tightness pixel | Pixel whose `tight` is closest to K=2/3 | +1 |
| 14 | Boundary-tightness band | Pixels whose `tight` is within ±V of K, mixed | +1 |

#### Escape-time voices (orbits sorted by smooth_n)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 15 | Fast-escape pixel | Lowest smooth_n | +2 (treble percussive) |
| 16 | Mid-escape pixel | Median smooth_n | 0 |
| 17 | Slow-escape pixel | Highest smooth_n | −2 (sub-bass long evolution) |

#### Trap voices (closest to each lattice ring)

| ID | Voice | Pixel selection | Octave |
|---|---|---|---|
| 18 | Closest-to-K trap | Pixel whose `orbit` is closest to K ring | −1 (bass) |
| 19 | Closest-to-V trap | Closest to V ring | −1 |
| 20 | Closest-to-φ trap | Closest to 1/φ ring | −1 |
| 21 | Closest-to-1 trap | Closest to unison ring | −1 |

#### Meta-options

| Option | Meaning |
|---|---|
| R | Random — one voice picked by T-agency entropy |
| A | All — every voice mixed |
| S | Solo — pick exactly one |
| M | Mix — pick any subset |
| P | Preset — pre-bundled selection |

#### Presets (P)

| Name | Voices | Character |
|---|---|---|
| Stereo | 1+2 | Pure PDT axes (D left, T right) |
| Full | 1+2+5+10 | Axes + radial fan + ring drones |
| Pure | 11+12 | Elegance lead + choir |
| Boundary | 13+14 | ∂I boundary only |
| Orchestra | 1+2+5+10+11+13+15+17 | Full orchestral spread (axes + fan + drones + elegance + boundary + extreme escape times) |

### 3.5 Octave map and treble/bass handling (D4 — confirmed)

Frequencies are relative to middle C (C4 = 261.63 Hz). The d-family pitch comes from `_D_FREQ` (existing table, line 6892); the per-voice octave shift moves the entire table up or down by powers of 2.

| Octave | Hz reference | Voices in this register |
|---|---|---|
| −2 | C2 ≈ 65 Hz | Ring drones (6,7,8,9,10), Slow-escape (17) |
| −1 | C3 ≈ 130 Hz | Trap voices (18,19,20,21) |
| 0 | C4 = 261.63 Hz | Axes (1,2,3,4), Mid-escape (16) |
| +1 | C5 ≈ 523 Hz | Radial fan (5, with k-spread to +1.5), Top-12 elegance (12), Boundary (13,14) |
| +2 | C6 ≈ 1046 Hz | Highest-elegance (11), Fast-escape (15) |

#### Per-voice envelope and EQ shaping

To make the voices coexist as a professional-grade mix:

- **Bass voices (octave −2, −1):** longer release tail ((1/K)× standard release = `(3/2) × K·V × note_samples = V × note_samples`). 1/K = 3/2 is the Pythagorean fifth — the most stable extended interval. Sub-bass sits underneath the mix.
- **Treble voices (octave +2):** shorter release (K× standard) so they sparkle without smearing. K = 2/3 as the scaling factor — the Koide binding stability threshold applied to the release envelope.
- **Mid voices (octave 0, +1):** standard envelope (attack = `V × note_samples`, release = `K·V × note_samples`).
- **One-pole shelf EQ per voice** (ET-derived coefficients):
  - Bass voices: shelf gain = 1 + V = 13/12, crossover at C3 = C4/2 = 130.8 Hz (one octave below middle C — structurally derived octave boundary)
  - Treble voices: shelf gain = 1 + V = 13/12, crossover at C5 = C4×2 = 523.3 Hz (one octave above middle C — structurally derived octave boundary)
  - Mids: no shelf
  - **NOTE:** The shelf gain 1 + V = 13/12 ≈ 0.69 dB is ET-derived (one V-quantum above unity). The crossover frequencies are exact octave boundaries from C4. Prior versions used ad-hoc +2 dB at 200 Hz / 2 kHz — these are replaced with ET-derived values.
- **Final mix processing:**
  1. Sum all selected voices (additive)
  2. Apply existing `_et_reverb` (comb+allpass, ET-derived)
  3. Soft peak limiter at (N−1)/N = 11/12 of full scale (same as the normalization target — 1 − V headroom, ET-derived). The limiter uses one-pole envelope follower with K-derived attack and V-derived release.
  4. Normalize to peak (N−1)/N = 11/12 before WAV write — this is 1 − V, leaving exactly one base-variance quantum of headroom. NOT 0.92 (which is an ad-hoc approximation of this value).

### 3.6 Image-mode duration (D2 — confirmed)

User-selectable from this list:

| Option | Duration |
|---|---|
| 1 | Natural (whatever the orbits produce) |
| 2 | 30 seconds |
| 3 | 1 minute |
| 4 | 5 minutes |
| 5 | 10 minutes |
| 6 | 30 minutes |
| 7 | 1 hour |

#### Behavior

- The "natural" duration is computed first from the probe orbits' actual lifetimes:  
  `natural_duration = max(probe_n_steps_actual) × samples_per_step / sample_rate`
- If the user picks a longer duration, the natural cycle **repeats** seamlessly until approximate fill. **Phase-continuous loop:** the last sample of repeat *k* hands its phase to the first sample of repeat *k+1* via the same persistent `phase` array used for portamento glides between notes. No clicks at loop boundaries.
- If the natural duration is **longer** than the user's selection, the natural duration wins. We don't truncate orbit data.
- If a non-natural duration is chosen, **a fade-out is applied at the end**, and the fade length equals the natural duration (per user direction: "the fade is over the length of the natural duration"). So a 5-minute selection with 47-second natural duration gets a 47-second fade-out at the end. If natural duration is selected, no fade.
- "Approximate" is acceptable. If 5 minutes is chosen and natural is 47s, we play approximately 6.4 cycles. The cut happens at the end of the last partial cycle, with the natural-duration fade-out covering the abruptness.

#### Replaces existing `audio_duration` parameter

The current `et_sonify_image` function (line 7251) has an `audio_duration=20.0` parameter that hardcodes a default 20-second audio output. The new `et_native_music_image` function:
- Removes the `audio_duration` parameter (the value comes from the new `_choose_image_duration()` prompt)
- Computes `natural_duration` from probe traces internally
- Implements the repeat-fade logic from this section
- The old 20s default is gone — there is no fixed default duration anymore

### 3.7 Video mode behavior

- Each frame's probe pixels get a per-step trace
- Music state (oscillator phases, global step counter, partial buffer) carries between frames via the existing `frame_idx` plumbing extended with a `music_state` parameter
- `samples_per_step` is computed once at the start of the video render so the audio's total length approximates `n_frames / fps`:
  - `mean_steps_per_frame = mean(probe_n_steps_actual across pre-sampled probe set)`
  - `samples_per_step = round((n_frames / fps) × sample_rate / (n_frames × mean_steps_per_frame))`
- The actual audio duration may differ slightly from `n_frames / fps`. We tell the user the actual duration before encoding so they can verify.
- Per user direction: "as long as it is approximate to the frames selected, then yes." Acceptable tolerance.

### 3.8 Output quality combined tier (D5 — confirmed)

A single "Output Quality" prompt at startup sets BOTH video and audio bitrates together. No mismatched tiers (high video + low audio or vice versa).

#### The five tiers

The bitrate ladder below uses the **research-validated** numbers from YouTube's official upload encoding settings (Google support page, retrieved 2026-04-09) and the professional-grade upload practice of going **1.5–2× above YouTube's minimums** to survive their re-encode.

**Video bitrates per preset (60fps, libx264 H.264 — see §3.9 for HEVC/AV1 alternates):**

| Preset | Pixels | Draft (CRF) | Standard (CRF) | High (CBR) | YouTube-optimal (CBR) | Archival (CBR) |
|---|---|---|---|---|---|---|
| 1080p | 2.07 MP | CRF 23 | CRF 20 | 25 Mbps | 35 Mbps | 80 Mbps |
| 2k | 4.19 MP | CRF 23 | CRF 20 | 50 Mbps | 70 Mbps | 160 Mbps |
| 4k | 16.78 MP | CRF 23 | CRF 20 | 150 Mbps | 200 Mbps | 500 Mbps |
| hq | 67.11 MP | CRF 23 | CRF 20 | 600 Mbps | 800 Mbps | 2000 Mbps |
| ultra | 268.44 MP | CRF 23 | CRF 20 | 2400 Mbps | 3200 Mbps | 8000 Mbps |

**Frame format:** the per-frame intermediate files (between the per-frame render and ffmpeg assembly) can be either 16-bit PNG (default, smaller, current behavior) or 32-bit float TIFF (larger but precision-preserving end-to-end). User-selectable per video render — see §3.13 for the menu, file size table, and rationale. The frame format choice is independent of the quality tier.

**Per-tier ffmpeg flag set:**

| Setting | Draft | Standard | High | YouTube-optimal | Archival |
|---|---|---|---|---|---|
| `-c:v` | libx264 | libx264 | libx264 | libx264 | libx264 |
| `-preset` | medium | slow | slow | slower | veryslow |
| Rate control | `-crf 23` | `-crf 20` | `-b:v X -maxrate 1.5X -bufsize 2X` | `-b:v X -maxrate 1.5X -bufsize 2X` | `-b:v X -maxrate 1.5X -bufsize 2X` |
| `-profile:v` | high | high | high | high10 | high444 |
| `-level` | 5.2 (8K: 6.2) | 5.2 (8K: 6.2) | 5.2 (8K: 6.2) | 5.2 (8K: 6.2) | 5.2 (8K: 6.2) |
| `-pix_fmt` | yuv420p | yuv420p | yuv420p | yuv420p10le | yuv444p10le |
| `-x264-params` | (default) | `keyint=60:min-keyint=60` | `keyint=60:min-keyint=60:no-scenecut=1` | `keyint=30:min-keyint=30:no-scenecut=1` | `keyint=30:min-keyint=30:no-scenecut=1` |
| `-color_primaries` / `-color_trc` / `-colorspace` | bt709 | bt709 | bt709 | bt709 | bt709 |

**Audio per tier:**

| Tier | MP3 standalone | AAC mux source | AAC mux bitrate | FLAC sidecar | WAV bit depth |
|---|---|---|---|---|---|
| Draft | 128 kbps | WAV (16-bit) | 192 kbps | no | 16-bit PCM |
| Standard | 192 kbps | WAV (16/24) | 256 kbps | no | 16-bit PCM (f32) / 24-bit PCM (f64) |
| High | 256 kbps | WAV (24-bit) | 320 kbps | no | 24-bit PCM |
| YouTube-optimal | 320 kbps | WAV (24-bit) | 384 kbps | no | 24-bit PCM |
| Archival | 320 kbps | WAV (24-bit / 32-float) | 512 kbps | **yes (24-bit)** | 24-bit PCM (f32) / **32-bit float** (f64) |

**Note on the WAV bit depth column:** the depth depends on BOTH the chosen tier AND the user's float precision choice (`GPU_FLOAT_PRECISION`). Float64 fractals get higher bit depth at the same tier — see §3.12 for the full ladder and rationale. The AAC mux always sources from the WAV file at whatever bit depth that WAV is, so the AAC encoder gets the best possible source. **The mux NEVER sources from the MP3** — this fixes the double-compression bug from §2.5.

**Critical fixes already locked into this ladder:**
- **AAC mux always sources from WAV** (lossless), never from MP3. No double-compression.
- **AAC mux bitrate scales with the chosen tier**, not fixed 192k.
- **WAV bit depth scales with both tier AND fractal float precision** (per §3.12).
- **YouTube-optimal tier matches YouTube's official AAC-LC stereo recommendation of 384 kbps** at 48 kHz (per https://support.google.com/youtube/answer/1722171).
- **Archival tier writes a FLAC sidecar** (lossless audio) for true mastering use, in addition to the muxed AAC.
- **Float64 Archival additionally writes a 32-bit float WAV master** for bit-perfect mastering — see §3.12.
- **Sample rate becomes 48 kHz throughout** (currently 44.1 kHz). Per YouTube's 2025 spec: "48 kHz is the standard for digital video and is recommended for upload."

#### Sample rate change: 44.1 kHz → 48 kHz

Per YouTube's official audio spec: 48 kHz is the recommended sample rate for video upload (44.1 kHz is also accepted but causes a transcode to 48 kHz at YouTube which can introduce aliasing artifacts). The current script uses 44.1 kHz (`_AUDIO_SR = 44100`). The new code uses 48 kHz (`_AUDIO_SR = 48000`). All `samples_per_step` calculations scale accordingly.

### 3.9 Codec sub-options in the Archival tier (libx264 / libx265 / SVT-AV1)

The Archival tier supports three codec choices, all called via ffmpeg, all open-source, all well-supported by modern ffmpeg builds. Tiers below Archival use libx264 only (no codec choice prompt).

#### The three codecs

**libx264 (default, H.264/AVC).** The current behavior. Universal playback compatibility (every device made in the last 15 years), the highest-quality H.264 encoder ever made, the gold standard for offline batch encoding at high quality. Per the encoder research: x264 with slower/veryslow presets gives the best quality for a given bitrate of any H.264 encoder, including all commercial alternatives. For YouTube uploads at the script's bitrate ladder, libx264 produces the highest-fidelity ∂I-boundary preservation through YouTube's re-encode pipeline.

**libx265 (sub-option, H.265/HEVC).** Approximately 50% more efficient than H.264 at equivalent quality — a 200 Mbps libx264 file is roughly equivalent in quality to a 100 Mbps libx265 file. Encodes 2-5× slower than libx264 at equivalent quality presets. Playback compatibility is good but not universal: every modern browser, every recent iPhone/iPad/Mac, every Android device from the last 5 years, every smart TV from the last 5 years, but some older devices and older browsers don't decode it. YouTube accepts HEVC uploads but re-encodes everything to VP9/AV1 for delivery anyway.

**libsvtav1 (sub-option, AV1).** Approximately 30% more efficient than HEVC at equivalent quality, so a 100 Mbps HEVC ≈ 70 Mbps AV1 ≈ 200 Mbps H.264. SVT-AV1 (Scalable Video Technology AV1) is Intel/Netflix's open-source AV1 encoder, included in modern ffmpeg builds since 2020. Encoding speed at preset 4 is comparable to libx265 slow, at preset 2 comparable to libx265 veryslow. **Strategic advantage for YouTube uploads:** YouTube's delivery pipeline already converts everything to VP9/AV1 internally, so an AV1 upload requires the minimum re-encoding from upload to delivery — fewer re-encoding steps means less generational quality loss. This is the cleanest path through YouTube for fractal fine structure preservation, even though the visual difference at the script's high bitrate ladder is small. Playback compatibility: less universal than HEVC. Every modern browser supports AV1 decoding, recent smart TVs and streaming devices support it, but older devices and older browsers don't. For local playback on a wide range of devices, libx264 or libx265 is safer; for YouTube upload, AV1 is theoretically the best fit.

#### The Archival tier sub-option prompt

When the user picks Archival, a follow-up prompt asks:

```
  ┌──────────────────────────────────────────────────────────────┐
  │   Archival tier — codec choice                               │
  │                                                              │
  │   1  H.264 (libx264)  — universal playback, default          │
  │   2  H.265 (libx265)  — 50% smaller files, broad support     │
  │   3  AV1 (libsvtav1)  — 70% smaller files, YouTube-native    │
  │                                                              │
  │   AV1 survives YouTube re-encoding cleanest because YouTube  │
  │   already uses AV1 internally for delivery.                  │
  │   H.264 has the widest local playback compatibility.         │
  │   H.265 is the middle ground.                                │
  └──────────────────────────────────────────────────────────────┘
```

Default is `1` (libx264). The user explicitly opts in to the alternates.

#### Per-codec ffmpeg flag set at the Archival tier

| Setting | libx264 (default) | libx265 (HEVC) | libsvtav1 (AV1) |
|---|---|---|---|
| `-c:v` | libx264 | libx265 | libsvtav1 |
| `-preset` | veryslow | veryslow | (uses preset number, see below) |
| Rate control | `-b:v X -maxrate 1.5X -bufsize 2X` | `-b:v X -maxrate 1.5X -bufsize 2X` | `-b:v X` (SVT-AV1 uses target bitrate directly) |
| Profile | `-profile:v high444` | `-profile:v main444-12` | (SVT-AV1 has no profile flag; uses preset and tune) |
| Level | `-level 5.2` (or 6.2 for ≥8K) | `-level 5.2` (or 6.2 for ≥8K) | (SVT-AV1 auto-selects) |
| Pixel format | yuv444p10le | yuv444p10le | yuv420p10le (SVT-AV1's most-supported high-quality format) |
| Codec params | `-x264-params keyint=30:min-keyint=30:no-scenecut=1` | `-x265-params keyint=30:min-keyint=30:no-scenecut=1:profile=main444-12` | `-svtav1-params preset=2:keyint=30:tune=0` (preset 2 = veryslow equivalent, tune 0 = visual quality) |
| Color | `-color_primaries bt709 -color_trc bt709 -colorspace bt709` | (same) | (same) |

**SVT-AV1 preset notes:** SVT-AV1 uses numeric presets 0-13 (lower = slower/better quality). For Archival, preset 2 corresponds roughly to libx265 veryslow in encoding time and quality. Preset 4 ≈ libx265 slow. Preset 8 ≈ libx265 medium. The plan uses preset 2 for Archival to match the libx264 veryslow / libx265 veryslow philosophy (maximum quality, encoding time is not a constraint at this tier).

**SVT-AV1 pixel format notes:** SVT-AV1 supports yuv420p and yuv420p10le natively. yuv444p10le is theoretically supported but rarely used in practice and has worse decoder compatibility than the 4:2:0 variants. The plan uses yuv420p10le for AV1 (10-bit smooth gradients, broadest decoder support). For libx264 and libx265 Archival, yuv444p10le is used (full chroma resolution, since H.264/H.265 4:4:4 is well-supported by modern decoders).

#### Codec sub-option in HEVC + AV1 in lower tiers

The codec sub-option prompt appears **only at the Archival tier**. Tiers Draft, Standard, High, and YouTube-optimal use libx264 unconditionally. Rationale: at the lower tiers, the user is prioritizing one of {speed, file size, YouTube compatibility} where libx264 is already the right answer. The Archival tier is where users explicitly opt in to "I want the absolute best, I'll wait for the encode, I'll deal with potentially-narrower playback compatibility for the smaller-file alternatives."

If you ever want HEVC or AV1 at a lower tier in the future, the implementation just needs to remove the "Archival only" gate from the codec prompt — the per-codec ffmpeg flag set is already general enough to work at any bitrate.

### 3.10 Voice menu prompt placement (D6 confirmed)

New prompt order at startup:

1. Quality preset (existing — line 88, `_choose_preset`)
2. Mode selection (existing — line 120, `_choose_modes`)
3. Iterations (existing — line 671, `_choose_iterations`)
4. Float precision (existing — line 706, `_choose_precision`)
5. Fractal type (existing — line 733, `_choose_type`)
6. Tower (existing — line 764, `_choose_tower`)
7. Advanced mode (existing — line 791, `_choose_advanced`)
8. Output mode (existing — line 865, `_choose_output_mode`)
9. Video parameters (existing, video-only — line 889, `_choose_video_params`)
10. **Video frame format** (NEW — `_choose_frame_format`, only if output mode = video) — see §3.13
11. **Output quality tier** (NEW — `_choose_output_quality`)
12. **Archival codec sub-option** (NEW — `_choose_archival_codec`, only if tier = Archival) — see §3.9
13. Audio enable Y/N (modified from existing `_choose_audio`)
14. **Music event rate** (NEW — `_choose_event_rate`, only if audio enabled)
15. **Music voices** (NEW — `_choose_music_voices`, only if audio enabled)
16. **Output stems Y/N** (NEW — `_choose_output_stems`, only if audio enabled)
17. **Image audio duration** (NEW — `_choose_image_duration`, only if audio enabled and output mode = image)

This ordering puts all music decisions after the user has committed to fractal type and output mode, so they make sense. The video-specific prompts (frame format) appear only for video runs. The Archival codec sub-option appears only when the user picks the Archival tier.

### 3.11 Stage 21.5 / 21.6 placement (greenlit)

```
STAGE 21    — CORE RENDER ENGINE  (single image + video)
STAGE 21.5  — ET-DERIVED LATTICE-NATIVE MUSIC ENGINE  ← rebuilt
STAGE 21.6  — VISUAL ANALYSIS TOOL  (REPURPOSED FROM v4.0 NATIVE MUSIC)  ← new
STAGE 22    — SINGLE IMAGE PIPELINE
STAGE 23    — ZOOM VIDEO PIPELINE
```

Stage 21.6 contains the repurposed `et_visual_*` functions with their history-explanation comment block.

### 3.12 Float precision matching: float64 fractal → float64 audio

**Principle:** if the user picks float64 for the iteration, the audio pipeline uses float64 throughout — same as the image. The music and image are siblings of the same orbit data; one cannot be lower-precision than the other.

#### Where float64 enters the audio pipeline

**Probe trace buffers (Phase B):** allocated with `dtype=FLOAT_DTYPE` (which is `np.float64` on float64 runs, `np.float32` on float32 runs). Both GPU (`cp.float64`/`cp.float32`) and CPU (`np.float64`/`np.float32`) paths use the matching dtype.

**Synthesis primitives (Phase C):** `_audio_koide_tone`, `_synth_note_sequence`, the new `_native_music_synth_voice`, and `_et_reverb` all already use `np.float64` internally for working precision (this is correct for both float32 and float64 fractal runs — the synthesis benefits from extra headroom). On float64 fractal runs, the inputs to these functions are now actually float64 (no upcast), so the precision is end-to-end double.

**Mix bus and limiter:** float64 throughout regardless of fractal precision. The mix sums many voices and benefits from double-precision accumulation even on float32 runs.

**WAV output:** the WAV bit depth is selected per the ladder in §3.8.1 below, scaled by the fractal precision.

**FLAC sidecar (Archival tier):** matches the WAV bit depth where possible (FLAC supports up to 24-bit losslessly). For the float64 Archival case, FLAC at 24-bit is provided AND the 32-bit float WAV is preserved as the true bit-perfect master. The user gets two lossless masters, one universally compatible (FLAC 24-bit) and one bit-perfect (32-bit float WAV).

**MP3 output:** MP3 encoding is lossy regardless of input bit depth. The MP3 bitrate ladder in §3.8 still applies. Higher input bit depth gives the MP3 encoder a better source to work from, so float64 → MP3 is slightly better than float32 → MP3 even though the output is the same MP3 bitrate.

**AAC mux:** AAC is also lossy but accepts up to 24-bit input (and some encoders accept 32-bit float). The mux command sources from the WAV at the chosen bit depth. AAC at 384k from a 24-bit WAV is meaningfully better than AAC at 384k from a 16-bit WAV.

#### Per-tier WAV bit depth ladder

This is a new addition to §3.8. The bit depth scales with both the chosen quality tier AND the fractal's float precision.

| Tier | Bit depth (float32 fractal) | Bit depth (float64 fractal) |
|---|---|---|
| Draft | 16-bit PCM | 16-bit PCM |
| Standard | 16-bit PCM | 24-bit PCM |
| High | 24-bit PCM | 24-bit PCM |
| YouTube-optimal | 24-bit PCM | 24-bit PCM |
| Archival | 24-bit PCM | **32-bit float WAV** |

**Rationale:**

- Draft/Standard at 16-bit gives 96 dB dynamic range — enough for preview audio and matches the size budget of these tiers.
- High and YouTube-optimal at 24-bit gives 144 dB dynamic range — matches YouTube's official mastering recommendation ("Bit depth: 24 bit is recommended, 16 bit accepted" — YouTube music video encoding spec) and is the standard for professional audio work.
- Archival float64 jumps to 32-bit float WAV (IEEE 754 single-precision float, supported by ffmpeg, FLAC, Audacity, Reaper, Pro Tools, and all major DAWs) — preserves the full precision of the synthesis pipeline. This is the maximum WAV format that has near-universal tool support; 64-bit float WAV exists but is rarely supported.
- Float64 Archival also writes a parallel 24-bit FLAC for tools that don't read 32-bit float WAV.

#### Implementation in `_audio_write_wav`

The existing function (line 6988) hardcodes 16-bit PCM. It gets a new `bit_depth` parameter:

```python
def _audio_write_wav(filepath, data_L, data_R=None, sr=_AUDIO_SR, bit_depth=16):
    """Write PCM or float WAV at specified bit depth.
       bit_depth: 16 (PCM int16), 24 (PCM int24), 32 (IEEE 754 float32).
       For 64-bit float fractals, use bit_depth=32 in Archival tier.
    """
```

The WAV format:
- **16-bit:** existing path, format code 1 (PCM), bps=2
- **24-bit:** format code 1 (PCM), bps=3, packed little-endian 24-bit ints
- **32-bit float:** format code 3 (IEEE_FLOAT), bps=4, IEEE 754 single-precision

The format header for 24-bit and 32-bit float requires updating the `fmt ` chunk (currently hardcoded for PCM 16-bit). All three are well-documented WAV variants. No external library needed — pure struct.pack.

#### FLAC for the float64 Archival case

In float64 Archival, after the 32-bit float WAV is written, also write a 24-bit FLAC sidecar via:

```python
def _build_flac24_cmd(wav_path, flac_path):
    return ['ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(wav_path),
            '-c:a', 'flac',
            '-sample_fmt', 's32',          # input float32 → FLAC needs s32
            '-bits_per_raw_sample', '24',  # store as 24-bit
            '-compression_level', '8',
            '-ar', '48000',
            str(flac_path)]
```

The float32 WAV is downconverted to s32 (signed 32-bit) by ffmpeg internally then stored as 24-bit FLAC. Nothing in the float64 → 32-bit float WAV → 24-bit FLAC path is lossy at the perceptual level — 24 bits is 144 dB dynamic range, far below any audible threshold. The 32-bit float WAV remains as the true bit-perfect master.

#### Audio precision summary table

| Fractal precision | Tier | Synthesis | WAV format | FLAC sidecar | MP3 | AAC mux |
|---|---|---|---|---|---|---|
| float32 | Draft | float64 | 16-bit PCM | — | 128 kbps | 192 kbps |
| float32 | Standard | float64 | 16-bit PCM | — | 192 kbps | 256 kbps |
| float32 | High | float64 | 24-bit PCM | — | 256 kbps | 320 kbps |
| float32 | YouTube-opt | float64 | 24-bit PCM | — | 320 kbps | 384 kbps |
| float32 | Archival | float64 | 24-bit PCM | 24-bit FLAC | 320 kbps | 512 kbps |
| float64 | Draft | float64 | 16-bit PCM | — | 128 kbps | 192 kbps |
| float64 | Standard | float64 | 24-bit PCM | — | 192 kbps | 256 kbps |
| float64 | High | float64 | 24-bit PCM | — | 256 kbps | 320 kbps |
| float64 | YouTube-opt | float64 | 24-bit PCM | — | 320 kbps | 384 kbps |
| float64 | Archival | float64 | **32-bit float WAV** | **24-bit FLAC (parallel)** | 320 kbps | 512 kbps |

Sample rate is 48 kHz throughout, all tiers, both precisions.

This precision matching is **not optional**. Per the user: "if an image or video uses 64bit float, then the audio should too." The plan now reflects this principle end-to-end.

### 3.13 Video frame format choice (PNG vs TIFF)

For video mode only, the user picks the per-frame intermediate format. This is the format that the renderer writes to disk for each frame, and that ffmpeg reads as input during video assembly. The choice affects intermediate disk usage and the precision of the encoder's input data, but does NOT affect the final muxed video file's bit depth (which is capped by the encoder's pixel format — yuv420p10le for high tiers, yuv444p10le for Archival H.264/HEVC).

#### The two formats

**16-bit PNG (default).** Current behavior. Each frame is a 16-bit unsigned integer per channel PNG file (3 channels × 16 bits = 48 bits per pixel). Compressed via PNG's deflate. The 16-bit precision is the maximum the PNG format supports — there is no 32-bit or 64-bit per-channel mode in PNG (the format spec hardcodes the legal bit depths to 1, 2, 4, 8, 16). Modest file size, fast to write on most disks (compression is the bottleneck), universal compatibility, fast to read by ffmpeg.

**32-bit float TIFF (sub-option).** Each frame is a 32-bit IEEE 754 float per channel TIFF file (3 channels × 32 bits = 96 bits per pixel). Same format as the single-image TIFF master that the script already writes (`write_tiff_float32` at line 7680). Preserves the full float32 precision of `et_post`'s output through the entire pipeline up to the encoder. The precision matters most in smooth color gradients near the ∂I boundary, where 32-bit float input produces cleaner dithering at the encoder's quantization step than 16-bit integer input. Each TIFF frame is a standalone HDR still image at the same precision as the single-image master output.

#### File size comparison (per-frame intermediate, before video assembly)

| Preset | Pixels | 16-bit PNG per frame | 32-bit float TIFF per frame |
|---|---|---|---|
| 1080p | 2.07 MP | ~6 MB | ~25 MB |
| 2k | 4.19 MP | ~12 MB | ~50 MB |
| 4k | 16.78 MP | ~50 MB | ~200 MB |
| hq (8K) | 67.11 MP | ~200 MB | ~800 MB |
| ultra (16K) | 268.44 MP | ~800 MB | ~3.2 GB |

#### Total intermediate disk usage for typical video lengths

For a 240-frame video (8 seconds at 30fps, or 4 seconds at 60fps as the source frame rate before `minterpolate` upscales to 60fps output):

| Preset | PNG total | TIFF total |
|---|---|---|
| 1080p | ~1.4 GB | ~6 GB |
| 2k | ~2.9 GB | ~12 GB |
| 4k | ~12 GB | ~48 GB |
| hq (8K) | ~48 GB | ~192 GB |
| ultra (16K) | ~192 GB | ~768 GB |

The intermediate frames are deleted after video assembly by default (the script keeps them in the `et_video_<timestamp>_*` subfolder until the user explicitly removes them — current behavior preserved). Disk space consumed during the render is the full intermediate total plus the final muxed video file size.

#### Per-frame TIFFs as standalone deliverables

A genuine bonus of the TIFF frame format: **each per-frame TIFF is a standalone HDR still image at the same precision as the script's single-image TIFF master output.** If the user picks TIFF frames for a long zoom video, every frame in the `et_video_<timestamp>_*` subfolder is a usable archival still. Want frame 142 as a standalone master print? It's already there, in 32-bit float TIFF format, ready to load in Photoshop/GIMP/Affinity for editing. PNG frames are 16-bit and also usable as stills, but the TIFF case parallels the single-image TIFF output exactly.

#### The frame format choice prompt

A new prompt `_choose_frame_format()` is added for video mode only (not for image mode, where the format choice doesn't apply). It runs after the output mode selection at startup, alongside the other video-specific prompts.

```
  ┌──────────────────────────────────────────────────────────────┐
  │   Video frame format                                         │
  │                                                              │
  │   P  — 16-bit PNG    (default, smaller files, ~1/4 the size) │
  │   T  — 32-bit float TIFF  (larger, precision-preserving,     │
  │                            each frame is a standalone        │
  │                            HDR master image)                 │
  │                                                              │
  │   PNG is fine for all normal use cases. TIFF is for those    │
  │   who want every drop of precision through the encoder       │
  │   pipeline AND want each frame as its own archival still.    │
  │   See plan §3.13 for the file size comparison.               │
  └──────────────────────────────────────────────────────────────┘
```

Default is `P` (16-bit PNG). The TIFF choice is opt-in.

#### Implementation in `generate_zoom_video`

The current code at line 7836:
```python
write_png_16bit(arr_u16, fp, dpi=OUTPUT_DPI)
```

becomes a tier-driven branch:
```python
if FRAME_FORMAT == 'tiff':
    fp = frame_dir / f'frame_{fi:06d}.tiff'
    write_tiff_float32(final_f32, fp, dpi=OUTPUT_DPI, description=frame_meta)
else:
    fp = frame_dir / f'frame_{fi:06d}.png'
    write_png_16bit(arr_u16, fp, dpi=OUTPUT_DPI)
```

Both writers already exist in the script (`write_png_16bit` at the existing call site, `write_tiff_float32` at line 7680 in the single-image path). No new writer functions needed — just pick one based on `FRAME_FORMAT`.

#### Implementation in the ffmpeg assembly command

The current input pattern at line 7867:
```python
ffmpeg_in = str(frame_dir / 'frame_%06d.png')
```

becomes:
```python
if FRAME_FORMAT == 'tiff':
    ffmpeg_in = str(frame_dir / 'frame_%06d.tiff')
else:
    ffmpeg_in = str(frame_dir / 'frame_%06d.png')
```

ffmpeg auto-detects TIFF input via the `tiff` decoder — no extra command-line flag needed. The encoder side of the command (`-c:v libx264`, `-b:v`, `-pix_fmt`, etc.) is unchanged regardless of the input format, because the encoder reads decoded float frames and quantizes them to its working bit depth either way.

#### Disk space warning at the prompt

When the user picks TIFF frames AND a high-resolution preset (4K or higher), the prompt prints an estimate of the intermediate disk usage so the user can confirm they have enough free space:

```
  → 32-bit float TIFF
  Estimated intermediate disk usage at 8K, 240 frames: ~192 GB
  Make sure you have this much free space in the script's directory.
```

The estimate is computed from preset pixel count × 4 bytes per channel × 3 channels × frame count, with a 10% safety margin.

---

## 4. Per-step trace data structures

### 4.1 Probe pixel index set

Computed once at the start of `_render_frame` based on the user's voice selection:

```
probe_pixel_indices = sorted unique union of (pixel_index for each selected voice's source pixels)
n_probes = len(probe_pixel_indices)
voice_pixel_map[voice_id] = list of probe_pixel_indices belonging to this voice
```

For voices that depend on **post-iteration** properties (Highest-elegance, Boundary-tightness, Escape-time, Trap voices), the probe set is determined in **two passes**:

1. First pass: render the entire frame normally (final state per pixel via existing kernel)
2. Compute the post-iteration metric for each pixel (elegance, tightness-distance-from-K, escape rank, trap distance)
3. Identify the pixels matching each post-iteration voice
4. Second pass: re-iterate only those pixels (plus the structurally-determined voices' pixels) with the per-step trace probe kernel

The structurally-determined voices (Real-axis, Imag-axis, Diagonals, Radial fan, Ring radius pixels) don't need pass 1 — their pixel coordinates are known from W, H, and center. These can be probed in a single pass alongside the main render.

### 4.2 Trace buffers

For each probe pixel, allocate per-step arrays. The number of steps is determined by the orbit's actual lifetime — **no static cap.** The metabolism manages memory allocation:

```
MAX_MUSIC_STEPS = max_iter   (capture EVERY step — no data discarded)
```

The metabolism ensures this fits in available memory: `n_probes × MAX_MUSIC_STEPS × n_fields × dtype_size ≤ K × available_memory`. If the full allocation exceeds the metabolism's memory budget, the metabolism reports the constraint and the user decides whether to reduce max_iter, reduce probe count, or allocate more memory. There is NO silent cap — every iteration step is precious orbit data.

Where `target_audio_duration_seconds` is:
- For image mode: the user's chosen image duration (defaults to 30 seconds for the cap calculation; actual playback uses natural duration logic from §3.6)
- For video mode: `n_frames / fps` divided by `n_frames` to get per-frame allowance, then summed

The trace buffers per probe:

```
probe_d_r       : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]
probe_d_t       : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]
probe_t_r       : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]
probe_t_t       : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]
probe_psi       : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]
probe_p_dom     : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]   (for ∂I; constant 2.0 for M/J)
probe_r         : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]   (orbit magnitude)
probe_theta     : FLOAT_DTYPE[n_probes, MAX_MUSIC_STEPS]   (orbit angle)
probe_n_actual  : int32[n_probes]                       (actual step count before escape or cap)
```

### 4.3 Memory budget

Memory allocation for probe traces is managed by the metabolism (Phase F). The metabolism computes the available memory budget from K × total_RAM (or K × total_VRAM for GPU probes) and allocates probe buffers within that budget. No static cap is applied — every iteration step is captured.

For reference, typical probe buffer sizes at full max_iter:

| Configuration | n_probes | max_iter | Bytes (float64) |
|---|---|---|---|
| Image, all voices, 500K iters | 30 | 500,000 | 30 × 500K × 9 × 8 = 1.08 GB |
| Image, all voices, 10M iters | 30 | 10,000,000 | 30 × 10M × 9 × 8 = 21.6 GB |
| Video, 240 frames, all voices | 30 per frame | (per-frame) | accumulates incrementally |

The metabolism verifies that the required allocation fits within K × available_memory BEFORE the render begins. If it does not fit, the metabolism reports the constraint explicitly — it does NOT silently cap or truncate. The user decides how to proceed (reduce max_iter, reduce probe count, or confirm they have sufficient memory).

For the video case, probe data is processed per-frame and the probe buffers are reused — only one frame's worth of probes is in memory at any time, plus the accumulated music state. This makes video probe memory bounded by the single-frame budget.

---

## 5. Implementation phases

The user has authorized: "We do not care about tool or context limits, as we can just continue when you reach a limit." Each phase below is sized to allow clean resume between conversations. After each phase I report what was done and pause for the next signal.

### Phase A — Renaming + repurposing (mechanical, no logic changes)  ✅ **COMPLETE (2026-04-09)**

**A.1** ✅ **DONE.** Rename "sonification" → "native music" in all comments, docstrings, prints, error messages, and Stage 21.5 header. Use `str_replace` with exact multi-line matching, never `create_file`. Verify with grep.
*Result:* All 26 textual occurrences renamed via 19 distinct `str_replace` edit sites. Post-edit grep returned zero "sonif" matches outside the deliberate Stage 21.6 history-context comment (Phase A.4 verifies the single intentional survivor).

**A.2** ✅ **DONE.** Rename functions:
- `et_sonify_image` → `et_native_music_image`
- `et_sonify_video_frame` → `et_native_music_video_frame`
- `et_sonify_video` → `et_native_music_video`
- Update all call sites in `generate_et_fractal` (line 7712) and `generate_zoom_video` (line 7928)
*Result:* 3 `def` lines renamed (previously at lines 7251, 7397, 7445 — now at the same line numbers because the renames are in-place). The internal docstring cross-reference inside `et_native_music_video` that said "list of dicts from `et_sonify_video_frame()`" also updated to `et_native_music_video_frame()` (line 7450). Three call sites updated: `generate_et_fractal` at line 7712 (now 7933 due to the Stage 21.6 insertion shifting downstream lines), `generate_zoom_video` frame loop at 7844 (now 8065), and `generate_zoom_video` audio block at 7928 (now 8149). Post-edit `grep 'et_sonify'` returns zero matches; `grep 'et_native_music'` returns 7 matches (3 defs + 1 docstring reference + 3 call sites).

**A.3** ✅ **DONE.** Create Stage 21.6 section with the repurposed functions:
- `et_visual_hue_from_rgb` (was `_audio_hue_from_rgb`)
- `et_visual_hue_to_d` (was `_audio_hue_to_d`)
- `et_visual_segment_row` (extracted from the image-scan branch of `_segment_row`)
- `et_visual_analyze_image` (new top-level analyzer that combines the above)
- Include explicit history-context comment explaining v4.0 origin
*Result:* Stage 21.6 header now at line 7550 (between the end of `et_native_music_video`'s body at line 7546 and the Stage 22 header). 4 new public functions added: `et_visual_hue_from_rgb` (7607, wraps legacy `_audio_hue_from_rgb` so the implementation lives in one place), `et_visual_hue_to_d` (7621, wraps legacy `_audio_hue_to_d`), `et_visual_segment_row` (7635, pure-visual extraction of the image-scan branch of `_segment_row` — does NOT accept the legacy `raw_dr`/`raw_dt`/`raw_tight` params because by construction this diagnostic tool has no kernel data available), and `et_visual_analyze_image` (7688, top-level analyzer scanning `n_scan_rows=12` rows by default, matching `N=12` manifold symmetry, returning a dict with `visual_histogram`, `kernel_histogram`, `mismatch_count`, `mismatch_rate`, `rows_scanned`, `segments_per_row`, `pixels_scanned` — enables visual-vs-kernel d-family consistency checking when `it_raw['d_r']` is supplied). The history-context header comment concentrates the v4.0 "sonification" identifier into exactly one line (7553) per plan §2.12 literal spec. Header comment also contains the explicit Three Tools mapping (Identification: P = rendered RGB image, D = FAM_HUE encoding constraints, T = analysis pass; Descriptor Gap: "is the visual encoding faithful to the kernel's computed d?" closed by the `mismatch_count`/`mismatch_rate` fields; Subsumption: no remainder). The legacy private helpers `_audio_hue_from_rgb` (line 6917), `_audio_hue_to_d` (6928), and `_segment_row` (7067) are preserved untouched in Stage 21.5 per the no-removal rule — they are still called by the native-music image/video-frame fallback branches when raw kernel data is unavailable.

**A.4** ✅ **DONE.** Verify with grep: zero remaining occurrences of `sonif` outside the deliberate Stage 21.6 history comment.
*Result:* `grep -c -i 'sonif'` returns exactly 1 match: line 7553 inside the Stage 21.6 history-context comment block. `grep 'et_sonify'` returns 0 matches. Python AST parse of the edited 8210-line script (grew from 7989, +221 lines for Stage 21.6) confirms it is syntactically valid Python 3 with 100 function definitions total. All edits applied via surgical `str_replace` with exact multi-line matching — no `create_file` on the script, no removal of any existing function, variable, parameter, or comment.

### Phase B — Kernel infrastructure for per-step probe traces

**B.1** Add the probe pixel selection function `_compute_probe_pixels(voices_selected, W, H, cx, cy, zoom, fractal_type)` that returns:
- `probe_pixel_indices`: 1-D int array of pixel indices into the flat W*H array
- `voice_pixel_map`: dict mapping voice_id → list of indices in `probe_pixel_indices`

**B.2** Add the probe-pass CUDA kernel `et_iterate_probe_di` (companion to `et_iterate_di`):
- Same iteration logic as `et_iterate_di`
- Additional output buffers: `probe_d_r_steps`, `probe_d_t_steps`, `probe_t_r_steps`, `probe_t_t_steps`, `probe_psi_steps`, `probe_p_dom_steps`, `probe_r_steps`, `probe_theta_steps`, `probe_n_actual`
- All probe output buffers are typed `float` (which becomes `double` in the F64 version via the existing `_make_f64_di_kernel` regex)
- Writes one entry per step to the per-step buffers, indexed by `(thread_idx, n)`
- Uses `__global__` launch with `n_probes` threads instead of `n_pix` threads
- Receives `probe_pixel_indices` to map thread_idx → original pixel coordinate

**B.3** Add the probe-pass CUDA kernel `et_iterate_probe_std` for J/M:
- Same as B.2 but based on `et_iterate` (the standard kernel)
- The lattice projection is computed every step in the standard kernel anyway (it's already used for tightness); now it's also captured per step

**B.4** Build the F64 versions through the existing `_make_f64_kernel` and `_make_f64_di_kernel` functions. Verify both compile. Both use the same intrinsic-replacement regex (`__log2f` → `log2`, `sqrtf` → `sqrt`, etc.) so the probe kernels follow the same pattern automatically.

**B.5** Add the CPU probe paths inside `iterate_strip_v2`:
- After the main iteration completes, run a second pass over only the probe pixels
- Same NumPy logic as the main path, but writing per-step traces
- Both `IS_DI_TYPE` and standard branches
- All probe arrays allocated with `dtype=FLOAT_DTYPE` (matches user precision choice)

**B.6** **Fix Finding 14 (the existing float32 hardcode bug):**
- Line 6663: change `astype(np.float32)` to `astype(FLOAT_DTYPE)` for both branches of the GPU `it_raw` readback
- Line 6685: change `np.zeros((rh, rw), dtype=np.float32)` to `np.zeros((rh, rw), dtype=FLOAT_DTYPE)` for the CPU `raw_arrays` allocation
- Line 6704: change `np.asarray(arr, dtype=np.float32)` to `np.asarray(arr, dtype=FLOAT_DTYPE)` for the CPU serial path
- Line 6736: same fix for the CPU thread-pool path
- This is a pre-existing bug independent of the music rebuild but it MUST be fixed because the music engine depends on receiving full-precision probe data on float64 runs

**B.7** Wire the probe pass into `_render_frame`:
- After the main kernel/CPU iteration completes, if `AUDIO_ENABLED`, compute probe pixel indices, run the probe pass, and add results to `it_raw` under new keys: `probe_d_r`, `probe_d_t`, `probe_t_r`, `probe_t_t`, `probe_psi`, `probe_p_dom`, `probe_r`, `probe_theta`, `probe_n_actual`, plus `voice_pixel_map`
- All probe arrays are `FLOAT_DTYPE`-typed (float64 on float64 runs, float32 on float32 runs)
- Both GPU and CPU paths

**B.8** Verify by running a small render (1080p, low iters, audio enabled, both float32 and float64 runs) and dumping the `it_raw['probe_*']` keys to confirm:
- They have the expected shapes
- They contain plausible values
- Their dtype matches `FLOAT_DTYPE` (specifically: `np.float64` on float64 runs, not `np.float32`)
- The Finding 14 fix at lines 6663/6685/6704/6736 is verified by checking that on a float64 run, the existing `it_raw` keys (`smooth_n`, `d_r`, etc.) also come back as float64 not float32

### Phase C — Native music engine (orbit-trace driven)

**C.1** Replace `_AUDIO_SR = 44100` with `_AUDIO_SR = 48000`. Trace all dependents and confirm consistency.

**C.2** New function `_native_music_synth_voice(probe_traces_for_voice, voice_meta, music_state, samples_per_step)`:
- Takes one voice's probe traces (could be 1 pixel or multiple, e.g. 12 for radial fan)
- Synthesizes audio for this voice using the existing `_synth_note_sequence` adapted for orbit-trace input
- Maintains persistent oscillator phase across calls (for video frame continuity)
- Applies octave shift, envelope shaping (per §3.5), per-voice EQ shelf
- Returns L/R sample arrays + updated music_state

**C.3** New function `et_native_music_image(it_raw, voices_selected, kbps, output_stems, image_duration, stem, script_dir)`:
- For image mode
- For each selected voice, calls `_native_music_synth_voice` to get its stream
- Mixes all voice streams (additive)
- Applies natural-duration repeat logic per §3.6 (with phase-continuous loop and natural-length fade-out if non-natural duration chosen)
- Applies `_et_reverb` to mix
- Applies soft peak limiter
- Writes mix to `_audio.wav` then encodes to `_audio.mp3` at `kbps`
- If `output_stems`, writes per-voice WAV+MP3 to `_stems/` subfolder
- Writes MIDI sidecar via `_write_midi`
- Returns mix path

**C.4** New function `et_native_music_video(per_frame_probe_data, voices_selected, fps, n_frames, kbps, output_stems, stem, script_dir, video_path)`:
- For video mode
- Accumulates probe data across all frames
- Computes `samples_per_step` once at start so total audio duration ≈ `n_frames / fps`
- Iterates frame-by-frame, calling `_native_music_synth_voice` per voice with persistent music_state
- No frame-boundary clicks: phase carries through `music_state.phase`
- After all frames processed, applies `_et_reverb` to mix, applies limiter
- Writes mix to WAV → MP3 at `kbps`
- If `output_stems`, writes per-voice files
- Calls (rewritten) `_audio_mux_video` to mux into MP4 at the tier-appropriate AAC bitrate

**C.5** Modify `_render_frame` to optionally accept and return `music_state` for the video loop, so per-frame state carries through.

### Phase D — Output bitrate fixes + new prompts + WAV bit depth + frame format + codec sub-option

**D.1** Add `_choose_output_quality()` returning a dict like:
```python
{
    'tier': 'youtube_optimal',
    'video_b': 200_000_000,    # bits per second
    'video_maxrate': 300_000_000,
    'video_bufsize': 400_000_000,
    'video_preset': 'slower',
    'video_profile': 'high10',
    'video_level': '5.2',
    'video_pix_fmt': 'yuv420p10le',
    'video_keyint': 30,
    'video_codec': 'libx264',
    'audio_mp3_kbps': 320,
    'audio_aac_kbps': 384,
    'wav_bit_depth': 24,        # 16, 24, or 32 — set per §3.12 ladder
    'flac_sidecar': False,
    'flac_bit_depth': 24,       # only used if flac_sidecar=True
}
```

The `wav_bit_depth` field is computed from the chosen tier AND the global `FLOAT_DTYPE`. The function reads `FLOAT_DTYPE` (which was set at line 1019 from `GPU_FLOAT_PRECISION`) and applies the §3.12 ladder.

**D.2** Add `_choose_event_rate()` returning event rate (50, 100, 200, 441, 1) — defaults to 100.

**D.3** Add `_choose_music_voices()` returning list of voice IDs (1..21).

**D.4** Add `_choose_output_stems()` returning bool.

**D.5** Add `_choose_image_duration()` returning seconds (None for natural).

**D.6** Modify `_choose_audio` to drop the bitrate prompt (now part of output quality tier). Just Y/N.

**D.6.1** **NEW: Add `_choose_frame_format()`** for video mode only — returns `'png'` (default) or `'tiff'`. Per §3.13. Includes the disk-space warning when TIFF is picked at 4K or higher resolutions, computed from `IMG_W × IMG_H × 12 bytes × n_frames × 1.1` (4 bytes per channel × 3 channels × 1.1 safety margin).

**D.6.2** **NEW: Add `_choose_archival_codec()`** for the Archival tier only — returns `'libx264'` (default), `'libx265'`, or `'libsvtav1'`. Per §3.9. Only called when `_choose_output_quality()` returns `tier == 'archival'`. The function returns its result into the tier dict's `video_codec` field.

**D.7** Reorder prompts per §3.10.

**D.8** Rewrite `_audio_write_wav` to support 16-bit PCM, 24-bit PCM, and 32-bit float formats:
- New `bit_depth` parameter (default 16 for backward compatibility)
- 16-bit path: existing logic, preserved unchanged
- 24-bit path: format code 1 (PCM), bps=3, packed little-endian 24-bit ints (use `np.int32` then strip the high byte during write)
- 32-bit float path: format code 3 (IEEE_FLOAT), bps=4, IEEE 754 single-precision floats written directly
- All three paths share the RIFF/WAVE container code with the format-specific `fmt ` chunk
- All three handle stereo (2-channel interleaved) the same way
- The peak-normalize step (`L = L / peak * (N-1)/N`) runs in float64 then casts to the output format. The normalization target is (N−1)/N = 11/12 ≈ 0.9167 — exactly 1 − V, leaving one base-variance quantum of headroom. This is ET-derived, not the ad-hoc 0.92 of the prior code.

**D.9** Rewrite `_audio_mux_video` to:
- Take the tier dict (or just `aac_kbps`)
- Use WAV as source, never MP3 (eliminates double-compression bug from §2.5)
- Encode AAC at the tier bitrate
- 48 kHz sample rate (`-ar 48000`)
- Use `-c:a aac -b:a {kbps}k -ar 48000`

**D.10** Rewrite the ffmpeg video assembly command in `generate_zoom_video` (lines 7875–7889):
- Build the command from the tier dict
- All the flags from §3.8 table
- Branch on `tier['video_codec']`:
  - `libx264` (default for non-Archival, sub-option for Archival): existing flag set
  - `libx265` (Archival sub-option): per-codec flag set from §3.9 table
  - `libsvtav1` (Archival sub-option): per-codec flag set from §3.9 table, including the SVT-AV1-specific `-svtav1-params preset=2:keyint=30:tune=0` and the `yuv420p10le` pixel format
- Branch on `FRAME_FORMAT` for the `ffmpeg_in` pattern:
  - `'png'`: `frame_%06d.png` (existing)
  - `'tiff'`: `frame_%06d.tiff` (new)
- Keep the `minterpolate` step for smoothness (it's a video filter, not a codec setting)
- The existing `cmd_raw` path (assembly without optical-flow `minterpolate`) is preserved as an error-reported path per convention 6 — it fires only when the `minterpolate` filter is unavailable, reports via `_et_error`, and applies the same tier bitrate, codec, and frame format. This is the ONLY accepted fallback in the entire encode pipeline — all other encode failures are fatal.

**D.10.1** **NEW: Branch the per-frame writer in `generate_zoom_video`** at line 7836:
```python
if FRAME_FORMAT == 'tiff':
    fp = frame_dir / f'frame_{fi:06d}.tiff'
    write_tiff_float32(final_f32, fp, dpi=OUTPUT_DPI, description=frame_meta)
else:
    fp = frame_dir / f'frame_{fi:06d}.png'
    write_png_16bit(arr_u16, fp, dpi=OUTPUT_DPI)
```
The `frame_meta` for the TIFF path is a per-frame ImageDescription tag containing the frame index, zoom level, and the same ET provenance fields as the single-image TIFF (mode, tower, center, fractal type). This makes each TIFF frame self-describing — the user can identify any extracted frame from its metadata alone.

**D.11** Add FLAC sidecar writing in the Archival tier path:
- For float32 Archival: write 24-bit FLAC from the 24-bit WAV via ffmpeg (`-c:a flac -compression_level 8 -ar 48000`)
- For float64 Archival: write 24-bit FLAC from the 32-bit float WAV via ffmpeg with `-sample_fmt s32 -bits_per_raw_sample 24` so the float32 source is properly converted to 24-bit signed PCM inside the FLAC container, AND keep the 32-bit float WAV alongside as the bit-perfect master
- Save as `_audio.flac` alongside the MP3
- The 32-bit float WAV (float64 case only) stays as `_audio_master.wav` to distinguish it from the standard `_audio.wav` from lower tiers

### Phase E — Verification

**E.1** Re-grep for `sonif` — only the historical comment in Stage 21.6 should match. The grep should return exactly 1 line.

**E.2** Trace every call to the new music functions: confirm they receive `probe_*` data, never `final_f32`. Use grep on `et_native_music_*` call sites.

**E.3** Trace every kernel output binding: confirm probe arrays are allocated, passed to kernel calls, and read back correctly for both GPU and CPU, both ∂I and standard, both F32 and F64. Use grep on kernel call sites to verify symmetry.

**E.4** Confirm `_render_frame` returns probe data in `it_raw` when audio enabled, both for image and video.

**E.5** **Float precision verification (covers Finding 14 fix):**
- Run a small image render at float32, audio enabled, all voices. Confirm `it_raw['probe_d_r'].dtype == np.float32`, `it_raw['smooth_n'].dtype == np.float32`. Confirm WAV bit depth = 24 (High tier).
- Run the same render at float64. Confirm `it_raw['probe_d_r'].dtype == np.float64`, `it_raw['smooth_n'].dtype == np.float64` (this is the Finding 14 fix in action — currently the existing code would return float32 here, which is the bug). Confirm WAV bit depth still 24 (High tier).
- Run float64 + Archival tier. Confirm WAV bit depth = 32-bit float (format code 3 IEEE_FLOAT). Confirm both `_audio_master.wav` (32-bit float) and `_audio.flac` (24-bit) exist alongside the MP3.
- Verify the 32-bit float WAV opens correctly in Audacity / Reaper / other DAW and reports 32-bit float format.

**E.6** Run a small image render with audio enabled, all voices selected, stems on. Confirm files are produced:
- `et_fractal_*.tiff`
- `et_fractal_*.png`
- `et_fractal_*_raw.npz` (now contains probe traces)
- `et_fractal_*_audio.wav` (mix at the tier-appropriate bit depth)
- `et_fractal_*_audio.mp3`
- `et_fractal_*_audio.mid`
- `et_fractal_*_stems/voice_1.wav` ... `voice_21.wav`
- `et_fractal_*_stems/voice_1.mp3` ... `voice_21.mp3`

**E.7** Run a small video render (10 frames, 1080p, low iters) with audio enabled, voices = preset "Stereo", stems off, tier = High. Confirm:
- Per-frame PNG files in subfolder
- `et_video_*.mp4` (60fps interpolated)
- `et_video_*_with_audio.mp4` (audio muxed at 320k AAC sourced from 24-bit WAV)
- `et_video_*_audio.mp3` (256 kbps standalone for High tier)
- `et_video_*_audio.wav` (24-bit master)
- No clicks at frame boundaries (audible test)

**E.8** Inspect the muxed video with `ffprobe`:
- Confirm video bitrate matches the tier
- Confirm AAC audio bitrate matches the tier
- Confirm sample rate is 48000
- Confirm pixel format matches the tier
- Confirm profile and level match
- Confirm WAV source bit depth via `mediainfo` or `ffprobe` on the `_audio.wav` file

**E.9** Run an Archival tier video render at float64 (smallest possible — 5 frames, 1080p, minimal iters) and confirm:
- FLAC sidecar exists (`_audio.flac`, 24-bit)
- 32-bit float master WAV exists (`_audio_master.wav`)
- Standard `_audio.wav` also exists (24-bit copy for cases where 32-bit float isn't supported)
- Muxed video uses AAC at 512 kbps
- HEVC option works if selected (libx265 instead of libx264)

**E.10** Run an Archival tier video render at float32 and confirm:
- FLAC sidecar exists (`_audio.flac`, 24-bit)
- WAV is 24-bit (no 32-bit float master needed for float32 fractals)
- Muxed video uses AAC at 512 kbps

**E.11** **Frame format verification:**
- Run a short video render with `_choose_frame_format()` returning `'png'`. Confirm intermediate frames in `et_video_*` subfolder are `.png` files. Confirm ffmpeg input pattern is `frame_%06d.png`. Confirm video assembly succeeds. Verify final muxed video plays.
- Run the same render with `_choose_frame_format()` returning `'tiff'`. Confirm intermediate frames are `.tiff` files. Confirm each TIFF is 32-bit float per channel via `tiffinfo` or similar (or by re-loading one in PIL and checking `mode == 'F'` for each channel). Confirm ffmpeg input pattern is `frame_%06d.tiff`. Confirm video assembly succeeds. Verify the final muxed video is byte-different from the PNG-source version (subtle dithering differences from the encoder's 32-float input vs 16-int input).
- Confirm the disk-space warning prints when TIFF + 4K or higher is selected.
- Confirm each TIFF frame opens as a standalone HDR still image in GIMP/Photoshop and shows the 32-bit float depth.

**E.12** **Archival codec sub-option verification:**
- Run an Archival tier render with `_choose_archival_codec()` returning `'libx264'`. Confirm ffmpeg command uses `-c:v libx264 -profile:v high444 -pix_fmt yuv444p10le -x264-params keyint=30:...`. Confirm muxed video plays in VLC, browsers, and Windows Media Player.
- Run the same render with `'libx265'`. Confirm ffmpeg command uses `-c:v libx265 -profile:v main444-12 -pix_fmt yuv444p10le -x265-params keyint=30:...`. Confirm muxed video plays in VLC, modern browsers, and recent media players. File size should be roughly 50% smaller than the libx264 version at the same target bitrate.
- Run the same render with `'libsvtav1'`. Confirm ffmpeg command uses `-c:v libsvtav1 -pix_fmt yuv420p10le -svtav1-params preset=2:keyint=30:tune=0` with NO `-preset`, `-profile:v`, or `-level` flags (those are libx264/libx265 specific). Confirm muxed video plays in VLC, modern browsers, and modern media players. Note that libsvtav1 encoding will be substantially slower than libx264/libx265 at preset 2 — this is expected for the Archival tier.
- Confirm the codec sub-option prompt only appears for the Archival tier and not for other tiers.

### Phase F — Lattice Computation Engine + ET Metabolism (NEW — Sempaevum Losslessness Theorem + Guide v2.2)

This phase adds two major subsystems: (1) a C-module-backed lattice computation engine with GPU-compatible arbitrary precision, and (2) an ET-derived metabolism that uses the fine structure constant α⁻¹ from the Sempaevum formula to govern the fractal generator's resource allocation. Both are independent of Phases A–E (Music Engine) and compose additively.

**Sources for Phase F (all read in full):**
- `ET_Sempaevum_Paper16.tex` — Losslessness Theorem (§18), Memoization Property (Corollary), α⁻¹ closed-form identity (Equation eq:alpha, §subsec:alpha), ∂I formalization (§12)
- `ET_Universal_Projection_Guide8.md` — Four Projection Paths (Part XIX), EML operators (Part XX §107), Python reference implementation (§110), Lattice Self-Projection Verification (§113)
- `ET_Four_Gaps_Verification.py` — Lossless lattice verification exemplar (120 DPS)
- `ET_Two_Critiques_Verification.py` — Full α⁻¹ verification at 120 DPS, closed-form identity checks
- `ET_Three_Tools_Complete_Reference.md` — Descriptor Gap Principle (§4), Subsumption Law (§5)

**CRITICAL CORRECTIONS FROM v1:**
1. **ALL heavy computation offloaded to a separate C module** (`et_lattice_engine.c`). Python is the orchestrator; C is the workhorse. This mirrors the ET CDF Compressor's `et_pattern_engine.c` architecture.
2. **mpmath is NOT used.** The lattice IS the precision engine. We build our own arbitrary-precision arithmetic from ET math, GPU-compatible. The C module implements multi-precision integer arithmetic using uint64 arrays. For GPU: a companion CUDA kernel provides the same algorithms. The lattice projection formula itself requires only log₂ (computable via binary method), round (integer), and GCD (integer) — no floating-point library needed at the core.
3. **α⁻¹ from the Sempaevum formula is the metabolism.** The program's resource allocation is governed by the ET-derived fine structure constant computed at 120+ lattice-precision digits, with K=2/3 as the program's hard ceiling and 1/3 reserved for the system. The floor of the computer (CPU silicon) is P.

---

#### F.A — The C Module: `et_lattice_engine.c`

**F.A.1** Create the C module with the following architecture:

```
et_lattice_engine.c
├── Multi-Precision Integer Arithmetic (ET-native, no external libs)
│   ├── mp_uint: uint64 array representation (configurable limbs)
│   ├── mp_add, mp_sub, mp_mul, mp_divmod: basic arithmetic
│   ├── mp_gcd: binary GCD (Stein's algorithm — no division needed)
│   ├── mp_cmp: comparison
│   └── mp_from_fraction, mp_to_string: conversion
│
├── Multi-Precision Fixed-Point Arithmetic
│   ├── mpfx: fixed-point with configurable fractional bits (400+ for 120 digits)
│   ├── mpfx_log2: binary method for log₂(r) at arbitrary precision
│   │   Algorithm: log₂(r) = integer_part + Σ (bit_k · 2^{-k})
│   │   where bit_k is determined by squaring and comparing to 2
│   │   This is the ET-native recursion: each step is a T-act of rounding
│   ├── mpfx_sqrt: Newton's method using mpfx arithmetic
│   ├── mpfx_pi: Machin-like formula or AGM (both are T-recursions)
│   │   π = lim N·2^k·sin(π/N·2^{-k}) where N=12 (ET 12-gon recursion,
│   │   as already noted in the fractal generator code at line 564)
│   └── mpfx_sin, mpfx_cos: Taylor series in mpfx (needed for π computation)
│
├── Lattice Projection Engine
│   ├── lattice_project(r, N_lat): (k, d, g, eps) at arbitrary precision
│   │   Uses mpfx_log2 for the log₂(r) computation
│   │   Uses mp_gcd for gcd(|k|, N_lat)
│   │   Everything else is integer arithmetic
│   ├── lattice_pullback(k, eps, N_lat): recover r from (k, eps)
│   │   r = 2^{(k + eps·N/1200)/N} using mpfx arithmetic
│   ├── lattice_mul, lattice_recip, lattice_pow: Memoization Property
│   │   mul = k-addition, recip = k-negation, pow = k-scaling
│   ├── lattice_self_verify(): project {N, 1/N, K, 1/K} → verify d=12, |ε|=1.955¢
│   └── equation_lattice_verify(lhs, rhs, tol): verify A=B as lattice identity
│
├── Fine Structure Constant Engine (Sempaevum eq:alpha)
│   ├── compute_alpha_inv(dps): α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))
│   │   Uses mpfx_sqrt(3), mpfx_pi, mpfx arithmetic
│   │   Computes at 120+ lattice-precision digits (configurable)
│   │   Verifies closed-form identities:
│   │     A₁ = √3/48 = σ/K_EM
│   │     A_cross = √3/(93312π²) = (2/π)·A₁·A₂
│   │     Σ A_k = 1/(216(18π−1)) = κ²/(N²(Nπ−κ))
│   │   Cross-verifies against 50-term partial sum convergence
│   ├── verify_alpha_codata(): compare ET α⁻¹ against CODATA 2022 value
│   └── alpha_digits(n): return first n digits of α⁻¹(ET) as string
│
├── EML Operators (Guide §107, Odrzywołek 2026)
│   ├── eml(x, y) = exp(x) − ln(y) in mpfx
│   ├── eml_exp(x) = eml(x, 1)
│   ├── eml_ln(x) = eml(1, eml(eml(1, x), 1))
│   ├── eml_mul(x, y) = eml_exp(eml_ln(x) + eml_ln(y))
│   └── eml_div(x, y) = eml_exp(eml_ln(x) − eml_ln(y))
│
├── LCM Tower (dynamically computed, not static list)
│   ├── compute_lcm_tower(): lcm(1..k) for k=2..11 → (2,6,12,60,420,2520,27720)
│   │   Plus intermediate refinements: 24=lcm(12,8), 36=lcm(12,9), etc.
│   │   Computed via mp_gcd: lcm(a,b) = a·b/gcd(a,b)
│   └── escalate_lcm_tower(r, eps_threshold): find min N where |ε|<threshold
│
└── Metabolism Engine (see F.C below)
    ├── compute_metabolism(): derive resource limits from α⁻¹ + K
    ├── metabolism_cpu_threads(): floor(K × N_CPU)
    ├── metabolism_memory_limit(): floor(K × total_RAM)
    ├── metabolism_vram_limit(): floor(K × total_VRAM)
    ├── metabolism_tile_size(): derived from α⁻¹ correction terms
    └── metabolism_report(): print resource allocation summary
```

**F.A.2** Compile the C module:
- On Windows: `cl.exe /O2 /GL /LD et_lattice_engine.c /Fe:et_lattice_engine.dll`
- On Linux: `gcc -O3 -shared -fPIC et_lattice_engine.c -o et_lattice_engine.so -lm`
- No external library dependencies. Pure C99 + POSIX (for sysinfo).
- The build happens at startup if the .dll/.so is missing, using the same pattern as the ET CDF Compressor.

**F.A.3** Python bindings via ctypes:
```python
# Pattern: getattr(dll, 'FunctionName')() per project rule
_dll = ctypes.CDLL(str(_dll_path))
_lattice_project = getattr(_dll, 'lattice_project')
_lattice_project.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.POINTER(LatticeResult)]
_lattice_project.restype = ctypes.c_int
```

**F.A.4** GPU companion: `et_lattice_engine_kernel.cu`
- Implements the SAME lattice projection at the SAME 120+ digit precision as the C module, using CUDA
- For arbitrary precision on GPU: implements the same fixed-point uint64-array representation with the same algorithms as the C module — no precision compromise
- The GPU kernel provides: `lattice_project_gpu()` at 120+ digits, `lattice_gcd_27720_gpu()` (extends the existing CUDA GCD), the metabolism's VRAM monitoring, and the full mpfx arithmetic suite
- The GPU path and CPU path produce IDENTICAL results — the hardware path never affects precision
- The existing float32/float64 CUDA kernels (`et_iterate`, `et_iterate_di`) are preserved unchanged for the float precision modes

---

#### F.B — The Fine Structure Constant as Metabolism

**F.B.1** The Sempaevum formula (Equation eq:alpha, verified at 120+ digits):

```
α⁻¹(ET) = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1))
```

Structural decomposition (from Sempaevum §subsec:alpha):
- **A₀ = (N−1)² + S² = 137** — integer, exact. Sum of squares of non-trivial semitone count (N−1=11) and manifold state count (S=4). This is the INTEGER IMPEDANCE FLOOR — the coarse structure of the EM coupling.
- **A₁ = √3/48 = σ/K_EM ≈ 0.0361** — open shimmer over 8 EM channels. σ = √V = √(1/12). K_EM = N·K = 8. Lattice projection: A₁ → (k=−58, d=6, |ε|=49.0¢) — hexadic, at the ∂I boundary. **Structural reading:** the shimmer correction IS a hexadic (composite resource allocation) quantity at the ∂I boundary. This is not numerology — it's forced by √V/K_EM.
- **A_cross = √3/(93312π²) = (2/π)·A₁·A₂** — product interference of shimmer (open) with closed bilateral-mediation loop. A₂ = κ²/(N³π). The geometric factor 2/π is the bilateral-to-circumferential phase conversion.
- **Σ A_k = 1/(216(18π−1)) = κ²/(N²(Nπ−κ))** — closed-form geometric series of higher closed-mediation loops, with k-th term carrying k Koide vertices, manifold-volume suppression N^{-(k+1)}, and (k−1) phase integrations π^{-(k-1)}.

The C module computes this at 120+ digits of precision using:
- `mpfx_sqrt(3)` for √3 (Newton's method in fixed-point)
- `mpfx_pi` for π (ET 12-gon recursion: π = lim 12·2^k·sin(π/(12·2^k)))
- Pure fixed-point arithmetic for all divisions and multiplications

**CRITICAL: The existing code (lines 563–572) uses an OUTDATED formula that must be updated.**

The existing code has:
```python
_DELTA_FS = ((1-_SIGMA)*K*V/A0_EM)*(1+K/(N*S_STATES))     # NOT in Sempaevum
_A1_5 = _SIGMA*K*(1+_DELTA_FS)/(S_STATES*_K_EM*N**3*math.sqrt(_PI))  # WRONG cross term
_A2 = K**2/(N**3*_PI)
_A3 = K**3/(N**4*_PI**2)
ALPHA_INV_ET = A0_EM + _A1 - _A1_5 - _A2 - _A3    # 5 terms, truncated series
```

Three discrepancies with the Sempaevum formalization:
1. **Cross term:** Code has A₁.₅ with an ad-hoc DELTA_FS factor and √π divisor. Sempaevum has A_cross = (2/π)·A₁·A₂ with π² divisor. These differ by ~4.25% (ratio 0.9575). The code's A₁.₅ predates the formalization.
2. **Series sum:** Code has A₂ + A₃ (two explicit terms, truncated). Sempaevum has the CLOSED FORM: Σ A_k = κ²/(N²(Nπ−κ)) = 1/(216(18π−1)). The closed form is exact; the two-term truncation misses the tail.
3. **Accuracy:** Code α⁻¹ = 137.035999110... (3.2σ from CODATA). Sempaevum α⁻¹ = 137.035999167... (0.46σ from CODATA). The Sempaevum formula is 7× closer to experiment.

The code MUST be updated to match the Sempaevum formula:
```python
# Sempaevum eq:alpha — the formalized, canonical form
_A1      = _SIGMA / _K_EM                                    # √3/48 = σ/K_EM
_A_CROSS = (2.0/_PI) * _A1 * (K**2 / (N**3 * _PI))          # (2/π)·A₁·A₂ = √3/(93312π²)
_SUM_AK  = K**2 / (N**2 * (N*_PI - K))                       # κ²/(N²(Nπ−κ)) = 1/(216(18π−1))
ALPHA_INV_ET = A0_EM + _A1 - _A_CROSS - _SUM_AK              # 4 terms, closed-form series
```

The old variables `_DELTA_FS`, `_A1_5`, `_A2`, `_A3` are NOT removed (per project rules) but are relabeled as `_DELTA_FS_LEGACY`, `_A1_5_LEGACY`, `_A2_LEGACY`, `_A3_LEGACY` with a comment explaining they are from the pre-Sempaevum derivation and retained for historical reference.

**F.B.2** The metabolism architecture — properly derived from ET:

The metabolism is the program's self-aware resource management system. It has THREE structurally distinct layers, each governed by a different ET constant:

**Layer 1 — ALLOCATION (K-determined):**
K = 2/3 determines HOW MUCH of each hardware resource the program claims. This is the Koide binding stability threshold — the program is bound to the hardware with two of three primitives aligned.

For each resource type:
```
Program allocation = floor(K × total)
System reserve     = total − program allocation = ceil((1−K) × total)
```

This is the HARD CEILING. The program never exceeds K × total for any resource. 1−K = 1/3 is reserved for the OS, other processes, and thermal headroom.

**Layer 2 — HEADROOM (V-determined):**
V = 1/12 determines the MINIMUM HEADROOM within the program's K allocation. This is the base variance — the irreducible quantum of descriptor uncertainty applied to resource management.

The allocation stack:
```
Total resource:      100%
System reserve:      (1−K) = 1/3 = 33.3%         ← OS and system
Program ceiling:     K = 2/3 = 66.7%              ← maximum program claim
Metabolism headroom: K × V = 1/18 = 5.56%         ← lattice engine overhead + spike absorption
Active allocation:   K × (1−V) = 11/18 = 61.1%    ← available for the actual computation
```

The active allocation K×(1−V) = 11/18 projects onto the lattice as **(k=−9, d=4, |ε|=47.4¢)** — **quartic**, near the **∂I boundary**. This is the same sublattice family as Kleiber's 3/4 law (the universal metabolic rate exponent, k=9, d=4). The active allocation IS structurally a metabolic-rate quantity. This is forced by K=2/3 and V=1/12.

**Layer 3 — MONITORING (α⁻¹-determined):**
α⁻¹ = 137.036... determines the RESOLUTION at which the metabolism monitors resource usage. This is the impedance of the program-hardware coupling.

The monitoring works as follows:
- The metabolism tracks resource usage at **A₀ = 137 distinguishable levels** between 0 and the active allocation K×(1−V)×total
- Each monitoring level represents **(K × (1−V) × total) / A₀** units of the resource
- For a 32 GB system: level width = (11/18 × 32 GB) / 137 = **143 MB per level**
- The metabolism reports which of the 137 levels the current usage occupies

The correction terms refine the monitoring:
- **A₁ (shimmer band ≈ 3.6%):** Resource usage naturally oscillates by ±A₁ around the mean within each monitoring cycle. The metabolism considers fluctuations within ±A₁ × active_allocation as NORMAL SHIMMER — not an alarm condition. For a 32 GB system: shimmer band = ±A₁ × 19.56 GB ≈ **±706 MB**. Usage drifting within this band is structural oscillation, not a resource leak.
- **A_cross (interference ≈ 1.9×10⁻⁶):** When multiple resource types (CPU, RAM, VRAM) are simultaneously near their limits, the effective headroom shrinks by A_cross × active. This is the cross-term between resource dimensions — the product interference of shimmer with the mediation loop.
- **Σ A_k (tail ≈ 8.4×10⁻⁵):** The probability of rare resource spikes exceeding the shimmer band. The metabolism reserves Σ A_k × active as additional absorption capacity within the V headroom.

**F.B.3** Substrate projection — hardware on the lattice:

The metabolism reads the P-substrate (hardware) and projects each characteristic onto the lattice. The d-family classification reveals the hardware's structural type:

| Hardware characteristic | Example | k | d | ε | Structural reading |
|---|---|---|---|---|---|
| K = 2/3 (allocation fraction) | 0.667 | −7 | **12** | −1.955¢ | Koide attractor — the allocation lives at EM/full-resolution |
| 1−K = 1/3 (system reserve) | 0.333 | −19 | **12** | −1.955¢ | Same Koide attractor — system share is structurally paired |
| V = 1/12 (headroom fraction) | 0.083 | −43 | **12** | −1.955¢ | Koide attractor — headroom is at the same structural position |
| K×(1−V) = 11/18 (active fraction) | 0.611 | −9 | **4** | +47.4¢ | **Quartic at ∂I boundary** — metabolic rate class |
| RAM = 2^n bytes (any power of 2) | 2^35 | +420 | **1** | 0.000¢ | **Octave exact** — binary hardware IS d=1 |
| VRAM = 2^n bytes (any power of 2) | 2^33 | +396 | **1** | 0.000¢ | **Octave exact** — same structural class |
| N cores = 12 | 12 | +43 | **12** | +1.955¢ | N = manifold symmetry → d=12 Koide attractor |
| floor(K × 24) = 16 = 2⁴ threads | 16 | +48 | **1** | 0.000¢ | **Octave exact** — allocated threads are d=1 |

**Structural finding:** Binary hardware resources (RAM, VRAM, powers-of-2 thread counts) are ALWAYS d=1 octave with ε=0 — perfectly lattice-aligned. This is structural: silicon is octave-class. The metabolism's allocation constants (K, V) are d=12 at the Koide attractor. The active allocation (K×(1−V)) is d=4 quartic — the same sublattice as Kleiber's metabolic rate. The lattice classifies the metabolism's own structure.

**F.B.4** Per-sublattice coupling — ξ(d) for computational scheduling:

The per-sublattice impedance ξ(d) = A₀ / ((d−1)² + S²) determines the coupling strength between the program and hardware for each sublattice family:

| d | A₀_magic | ξ(d) | Coupling class |
|---|---|---|---|
| 1 | 16 | 8.563× | Maximum — octave/gravity, pure integer ops |
| 2 | 17 | 8.059× | Near-maximum — tritone/boundary |
| 3 | 20 | 6.850× | Strong — cubic/strong force |
| 4 | 25 | 5.480× | Quartic — metabolic rate class |
| 6 | 41 | 3.341× | Hexadic — composite resource class |
| 12 | 137 | 1.000× | Baseline — EM/full-resolution |

The ξ(d) values provide scheduling weights. When the metabolism classifies a computational region by its dominant d-family (from the fractal's lattice projection), it assigns scheduling weight ξ(d) to that region's processing. Regions dominated by d=1 (strongest coupling) get priority because they benefit most from tight hardware coupling. Regions at d=12 (baseline) get standard scheduling.

**F.B.5** The metabolism as a continuous control loop:

The metabolism runs CONTINUOUSLY during the render, not just at startup:

1. **Before each tile:** Project available resources onto the lattice. Classify current usage level (which of the 137 monitoring levels). If usage exceeds the active allocation K×(1−V)×total, report and wait for resources to free.
2. **During each tile:** The metabolism is passive — the computation runs at full precision without interference. The metabolism only OBSERVES, never throttles the computation itself.
3. **After each tile:** Report actual peak usage, the monitoring level, whether shimmer band was exceeded, and the d-family classification of the tile's dominant computation.
4. **Between frames (video mode):** Aggregate per-tile metabolism reports into a per-frame summary. Adjust tile geometry if the V-headroom was consistently exceeded (indicating the tile size is too large for the available resources).

**F.B.6** The metabolism report at startup:

```
═══════════════════════════════════════════════════════════════════════
  ET METABOLISM — Sempaevum α⁻¹ Resource Governance
═══════════════════════════════════════════════════════════════════════
  α⁻¹(ET) = 137.03599916744... (120 digits computed from Sempaevum eq:alpha)
  CODATA 2022: 137.035999177(21) — ET sits 0.46σ from center of CODATA interval

  ALLOCATION (K-determined):
    K = 2/3 (program ceiling)   1−K = 1/3 (system reserve)

  P (substrate floor):
    CPU: AMD Ryzen 9 5900X — 12 cores / 24 threads
    RAM: 32.00 GB total          → d=1 octave (binary = octave-class)
    GPU: NVIDIA RTX 2070 SUPER — 8.00 GB VRAM  → d=1 octave

  HEADROOM (V-determined):
    Active allocation:  K×(1−V) = 11/18 = 61.1%  → d=4 quartic (metabolic rate class)
    Metabolism headroom: K×V = 1/18 = 5.56%

  ALLOCATION TABLE:
    CPU threads: floor(K × 24) = 16 program  |  8 system   → d=1 octave (exact)
    RAM:         K × 32.00 = 21.33 GB program |  10.67 GB system
    VRAM:        K × 8.00 = 5.33 GB program   |  2.67 GB system

  MONITORING (α⁻¹-determined):
    Resolution: A₀ = 137 levels across active allocation
    Level width: 143 MB per level
    Shimmer band: ±706 MB (±A₁ = ±3.6% normal fluctuation)

  SUBSTRATE PROJECTION:
    K=2/3 → (k=−7, d=12, ε=−1.955¢) Koide attractor ✓
    Active 11/18 → (k=−9, d=4, ε=+47.4¢) Quartic/∂I boundary ✓
    16 threads → (k=48, d=1, ε=0.000¢) Octave exact ✓

  VERIFICATION:
    Self-projection: {N,1/N,K,1/K} → d=12, |ε|=1.955¢ ✓
    α⁻¹ identity:    closed-form matches 50-term series ✓
    CODATA check:     |α⁻¹(ET) − α⁻¹(CODATA)| < 0.46σ ✓
═══════════════════════════════════════════════════════════════════════
```

**F.B.7** Wire the metabolism into the existing resource management:

The existing code at line 519 has `N_CPU = max(1, os.cpu_count() or 1)` and at line 1027 has `N_THREADS = 1 if USE_GPU else N_CPU`. The metabolism replaces these with:

```python
# ── Metabolism (K + α⁻¹ + V) ──────────────────────────────────────
N_CPU = max(1, os.cpu_count() or 1)
_metabolism = getattr(_dll, 'compute_metabolism')
# ... ctypes call to C module ...

# Layer 1: K-allocation
N_THREADS_MAX = _metabolism_result.cpu_threads   # = floor(K × N_CPU)
N_THREADS = 1 if USE_GPU else N_THREADS_MAX
VRAM_CEILING = _metabolism_result.vram_ceiling   # = floor(K × total_VRAM)
RAM_CEILING  = _metabolism_result.ram_ceiling    # = floor(K × total_RAM)

# Layer 2: V-headroom
VRAM_ACTIVE = _metabolism_result.vram_active     # = floor(K × (1-V) × total_VRAM)
RAM_ACTIVE  = _metabolism_result.ram_active      # = floor(K × (1-V) × total_RAM)
VRAM_HEADROOM = VRAM_CEILING - VRAM_ACTIVE       # = floor(K × V × total_VRAM)

# Layer 3: α⁻¹ monitoring
MONITOR_LEVELS = A0_EM   # = 137
MONITOR_LEVEL_WIDTH = RAM_ACTIVE / MONITOR_LEVELS
SHIMMER_BAND = A1 * RAM_ACTIVE   # ±A₁ × active = normal fluctuation
```

The tile size calculation becomes metabolism-governed:
```python
# Tile geometry: metabolism-aware, lattice-classified
_tile_memory_per_row = RENDER_W * N_CHANNELS * FLOAT_SIZE * 2  # input+output per row
TILE_ROWS_MAX = int(VRAM_ACTIVE / _tile_memory_per_row) if USE_GPU else int(RAM_ACTIVE / _tile_memory_per_row)
TILE_ROWS = min(_P['tile'], TILE_ROWS_MAX)  # never exceed the active allocation
# Project tile count onto lattice for structural classification
_n_tiles = math.ceil(RENDER_H / TILE_ROWS)
_tile_proj = lattice_project(_n_tiles, 12)
# Report: tile count d-family tells you the tiling's structural type
```

**F.B.8** Complete resource detection — works on ANY system:

The C module detects all hardware resources at startup via platform-native system calls. No external libraries (no psutil, no platform-specific Python packages). Pure C99 with platform `#ifdef` blocks:

```c
/* et_lattice_engine.c — resource detection */

typedef struct {
    uint64_t cpu_cores_physical;   /* physical cores */
    uint64_t cpu_cores_logical;    /* logical threads (hyperthreading) */
    uint64_t ram_total_bytes;      /* total physical RAM */
    uint64_t ram_available_bytes;  /* currently available RAM */
    uint64_t disk_free_bytes;      /* free disk space in working directory */
    uint64_t disk_total_bytes;     /* total disk space */
    uint64_t vram_total_bytes;     /* GPU VRAM total (0 if no GPU) */
    uint64_t vram_free_bytes;      /* GPU VRAM free (0 if no GPU) */
    uint64_t page_size;            /* OS page size */
    int      platform;             /* 0=Linux, 1=Windows, 2=macOS, 3=other */
    int      gpu_detected;         /* 1 if CUDA GPU detected, 0 otherwise */
} et_system_resources_t;

int et_detect_resources(et_system_resources_t* out) {
#ifdef _WIN32
    /* Windows: kernel32.dll APIs */
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    out->cpu_cores_logical = si.dwNumberOfProcessors;
    out->page_size = si.dwPageSize;
    
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    out->ram_total_bytes = ms.ullTotalPhys;
    out->ram_available_bytes = ms.ullAvailPhys;
    
    /* Physical cores via GetLogicalProcessorInformation */
    DWORD buflen = 0;
    GetLogicalProcessorInformation(NULL, &buflen);
    /* ... enumerate PROCESSOR_RELATIONSHIP to count physical cores ... */
    
    /* Disk: GetDiskFreeSpaceExW on working directory */
    ULARGE_INTEGER free_avail, total, total_free;
    GetDiskFreeSpaceExW(L".", &free_avail, &total, &total_free);
    out->disk_free_bytes = free_avail.QuadPart;
    out->disk_total_bytes = total.QuadPart;
    
    out->platform = 1;

#elif defined(__linux__)
    /* Linux: sysconf + sysinfo + statvfs */
    out->cpu_cores_logical = sysconf(_SC_NPROCESSORS_ONLN);
    out->page_size = sysconf(_SC_PAGESIZE);
    
    /* Physical cores: parse /sys/devices/system/cpu/cpu*/topology/core_id */
    /* ... count unique core_id values ... */
    
    struct sysinfo si;
    sysinfo(&si);
    out->ram_total_bytes = (uint64_t)si.totalram * si.mem_unit;
    out->ram_available_bytes = (uint64_t)(si.freeram + si.bufferram) * si.mem_unit;
    /* Also read /proc/meminfo for MemAvailable (more accurate) */
    
    struct statvfs sv;
    statvfs(".", &sv);
    out->disk_free_bytes = (uint64_t)sv.f_bavail * sv.f_bsize;
    out->disk_total_bytes = (uint64_t)sv.f_blocks * sv.f_bsize;
    
    out->platform = 0;

#elif defined(__APPLE__)
    /* macOS: sysctl + statvfs */
    int mib[2] = {CTL_HW, HW_NCPU};
    int ncpu;
    size_t len = sizeof(ncpu);
    sysctl(mib, 2, &ncpu, &len, NULL, 0);
    out->cpu_cores_logical = ncpu;
    
    /* Physical cores */
    size_t pcount;
    len = sizeof(pcount);
    sysctlbyname("hw.physicalcpu", &pcount, &len, NULL, 0);
    out->cpu_cores_physical = pcount;
    
    /* RAM */
    uint64_t memsize;
    len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    out->ram_total_bytes = memsize;
    
    /* Available RAM via vm_statistics64 */
    /* ... mach_host_self() + host_statistics64() ... */
    
    out->page_size = sysconf(_SC_PAGESIZE);
    
    struct statvfs sv;
    statvfs(".", &sv);
    out->disk_free_bytes = (uint64_t)sv.f_bavail * sv.f_bsize;
    out->disk_total_bytes = (uint64_t)sv.f_blocks * sv.f_bsize;
    
    out->platform = 2;
#else
    out->platform = 3;
    /* Fallback: report what sysconf can give */
    out->cpu_cores_logical = sysconf(_SC_NPROCESSORS_ONLN);
    out->page_size = sysconf(_SC_PAGESIZE);
    out->ram_total_bytes = (uint64_t)sysconf(_SC_PHYS_PAGES) * out->page_size;
#endif

    /* GPU/VRAM: passed from Python via separate call (CUDA detected in Python) */
    out->gpu_detected = 0;  /* set by Python caller if CUDA available */
    out->vram_total_bytes = 0;
    out->vram_free_bytes = 0;
    
    return 0;
}
```

The metabolism then computes:
```c
typedef struct {
    /* Layer 1: K-allocation */
    uint64_t cpu_threads_program;     /* floor(K × logical cores) */
    uint64_t ram_ceiling_bytes;       /* floor(K × total RAM) */
    uint64_t vram_ceiling_bytes;      /* floor(K × total VRAM), 0 if no GPU */
    uint64_t disk_ceiling_bytes;      /* floor(K × free disk) */
    
    /* Layer 2: V-headroom */
    uint64_t ram_active_bytes;        /* floor(K × (1-V) × total RAM) */
    uint64_t vram_active_bytes;       /* floor(K × (1-V) × total VRAM) */
    uint64_t ram_headroom_bytes;      /* ram_ceiling - ram_active */
    uint64_t vram_headroom_bytes;     /* vram_ceiling - vram_active */
    
    /* Layer 3: α⁻¹ monitoring */
    uint64_t monitor_levels;          /* A₀ = 137 */
    uint64_t ram_level_width_bytes;   /* ram_active / 137 */
    uint64_t shimmer_band_bytes;      /* A₁ × ram_active */
    
    /* Substrate projection */
    int      ram_lattice_d;           /* d-family of total RAM (always 1 for 2^n) */
    int      cores_lattice_d;         /* d-family of core count */
    int      active_lattice_d;        /* d-family of K×(1-V) = 11/18 → always d=4 */
    
    /* α⁻¹ at 120+ digits (stored as string) */
    char     alpha_inv_digits[256];   /* "137.035999167441337483..." */
} et_metabolism_t;
```

**Complete resource coverage checklist:**
- CPU physical cores: ✓ (Linux: /sys topology, Windows: GetLogicalProcessorInformation, macOS: sysctlbyname hw.physicalcpu)
- CPU logical threads: ✓ (all platforms: sysconf/_SC_NPROCESSORS_ONLN or GetSystemInfo)
- Total RAM: ✓ (Linux: sysinfo, Windows: GlobalMemoryStatusEx, macOS: sysctl hw.memsize)
- Available RAM: ✓ (Linux: /proc/meminfo MemAvailable, Windows: GlobalMemoryStatusEx, macOS: vm_statistics64)
- Disk free/total: ✓ (POSIX: statvfs, Windows: GetDiskFreeSpaceExW)
- Page size: ✓ (all platforms: sysconf/_SC_PAGESIZE or GetSystemInfo)
- GPU VRAM: ✓ (passed from Python CUDA detection — already in the fractal generator at line 513)
- Platform identification: ✓ (compile-time #ifdef)

**Edge cases handled:**
- No GPU: `gpu_detected=0`, `vram_*=0`, metabolism uses RAM-only allocations
- Very low RAM (< 4 GB): metabolism still works, active allocation scales with K×(1−V)
- Very high RAM (> 1 TB): all uint64, no overflow up to 16 exabytes
- Single-core CPU: `cpu_threads_program = max(1, floor(K × 1)) = 1` — still functional
- Multiple GPUs: Python-side detection enumerates all CUDA devices, passes the selected device's VRAM
- 32-bit system: NOT SUPPORTED — the fractal generator requires 64-bit for NumPy/CuPy. The C module asserts `sizeof(void*) >= 8` at compile time.
- Containers/VMs: sysconf and sysinfo report the CONTAINER's limits, which is correct behavior — the metabolism operates within whatever resources the container exposes.

---

#### F.C — Lossless Lattice Constants (exact forms alongside floats)

**F.C.1** Add exact constant representations to Stage 3:

```python
# === LOSSLESS LATTICE CONSTANTS (Sempaevum §18) ===
# Exact integer forms (zero loss, used by C module)
N_EXACT      = 12           # already exact
N_ET_EXACT   = 27720        # already exact
S_EXACT      = 4            # already exact
A0_EXACT     = 137          # (N-1)² + S² = 137 exact

# Exact rational forms (numerator, denominator pairs for C module)
K_NUM, K_DEN = 2, 3         # K = 2/3
V_NUM, V_DEN = 1, 12        # V = 1/12

# These feed the C module's mpfx arithmetic for arbitrary-precision computation.
# The float forms (K=2.0/3.0, V=1.0/12, etc.) remain for the float iteration paths.
# The C module computes its own mpfx versions at startup from these exact integers.
```

**F.C.2** The C module receives these as integer parameters and constructs its own arbitrary-precision representations. No Python-side arbitrary precision library is needed.

---

#### F.D — Self-Projection Verification

**F.D.1** At startup, the C module runs `lattice_self_verify()`:
- Projects {N=12, 1/N=1/12, K=2/3, 1/K=3/2} onto the lattice at N_lat=12
- Verifies all four land at d=12, |ε|=1.955¢ (the Koide attractor)
- This is the Sempaevum §18 / Guide §113 self-consistency check
- Failure = fatal error (the lattice constants are corrupted)

**F.D.2** At startup, the C module runs `verify_alpha_codata()`:
- Computes α⁻¹(ET) at 120+ digits
- Verifies: A₁ = √3/48 (exact identity)
- Verifies: A_cross = √3/(93312π²) (exact identity)
- Verifies: Σ A_k = 1/(216(18π−1)) matches 50-term partial sum
- Verifies: |α⁻¹(ET) − 137.035999177| < 0.000000021 (CODATA 2022 uncertainty)
- Non-fatal if the last check fails (CODATA may be updated), but prints warning

---

#### F.E — Lattice Precision Mode + Comparison System

**F.E.1** A comparison mode that runs alongside the main render at lattice precision (120+ digits in the C module), verifying that the float path produces results consistent with the exact computation:

- Every pixel computed at float32/float64 can also be computed at 120+ digits for comparison
- At minimum, N=12 structurally significant pixels are always compared (center, axes, radial fan)
- Reports per-pixel deviation in cents between float and exact lattice positions
- Reports sublattice family distribution across the full render
- Validates that the float path is within acceptable tolerance
- In Lattice precision mode (L), this IS the primary computation — every pixel runs at 120+ digits

**F.E.2** Precision option in `_choose_precision()`:

The existing float32/float64 choice is extended:
```
  ┌──────────────────────────────────────────────────────────────┐
  │   Float Precision                                            │
  │                                                              │
  │   32  — float32 (GPU, good for exploration)                  │
  │   64  — float64 (GPU/CPU, publication quality)               │
  │   L   — Lattice (C engine + GPU, 120+ digits)                │
  │         Computes every lattice projection at 120+ digit      │
  │         precision using the ET lattice engine C module and   │
  │         companion CUDA kernel. Maximum possible precision.   │
  │         The metabolism manages all resource allocation.      │
  └──────────────────────────────────────────────────────────────┘
```

When Lattice mode is selected:
- `USE_LATTICE_PRECISION = True` flag is set
- The C module and companion CUDA kernel handle all lattice projections at 120+ digit precision
- GPU path uses the companion CUDA kernel with uint64-array mpfx arithmetic
- CPU path uses the C module's mpfx arithmetic
- Both paths produce identical results — precision is the same regardless of hardware path
- Final results are cast to float64 for display/output
- The metabolism manages all resource allocation — the user does not need to worry about anything other than precision and quality

---

#### F.F — Phase F Verification

- **F.V1:** C module compiles on Windows (MSVC) and Linux (GCC) without external library dependencies
- **F.V2:** `lattice_project(2/3, 12)` returns d=12, |ε|=1.955¢ via C module
- **F.V3:** `lattice_pullback(k, eps)` recovers original r for 100 test rationals via C module
- **F.V4:** `lattice_mul` satisfies `pullback(mul(proj(a), proj(b))) ≈ a·b` for 100 random pairs
- **F.V5:** `compute_alpha_inv(120)` matches CODATA 2022 within ±0.46σ
- **F.V6:** α⁻¹ closed-form identities all verified at 120 digits
- **F.V7:** Self-projection passes: {N, 1/N, K, 1/K} → d=12, |ε|=1.955¢
- **F.V8:** Metabolism correctly computes floor(K × N_CPU) threads
- **F.V9:** Metabolism correctly limits VRAM to K × total_VRAM
- **F.V10:** LCM tower is dynamically computed (not a static list), matches known values
- **F.V11:** EML operators produce correct results: eml_exp(0) = e, eml_ln(e) ≈ 1
- **F.V12:** Lattice comparison mode runs on all structurally significant pixels, deviation between float64 and 120+ digit results < 0.01¢ for well-resolved orbits
- **F.V13:** C module's mpfx_pi converges to π at 120+ digits using the 12-gon recursion
- **F.V14:** C module's mpfx_log2 matches known values (log₂(3), log₂(5), log₂(7)) at 120 digits
- **F.V15:** Tile size is dynamically adjusted when VRAM is constrained
- **F.V16:** No mpmath or sympy import anywhere in the codebase — all precision is C-module-backed

---

## 6. Files involved

### 6.1 Primary file (the only one being modified)

- `/mnt/user-data/uploads/ET_FRACTAL_GENERATOR50-10.py` — 8,210 lines (post-Phase A), all changes are surgical edits via `str_replace` with exact multi-line matching. No `create_file` on this file.

### 6.2 Reference documents (read-only, in project knowledge)

- `dI_Fractal_Explanation.md` — the lattice-aware fractal specification (uploaded this session)
- `/mnt/project/ET_Three_Tools_Complete_Reference.md` — Identification, Descriptor Gap, Subsumption (required reading per project rules)
- `/mnt/project/ET_Complex_Lattice.md` — 2D Gaussian-integer lattice on ℂ
- `/mnt/project/ET_Fine_Structure_Constant_REVISED.md` — A₀ = (N−1)² + S² = 137
- `/mnt/project/ET_Fantastical_Configurations.md` — magical impedance §3.3 Table 2
- `/mnt/project/ET_Semitone_Cascade_Complete.md` — palindromic cascade derivation
- `/mnt/project/ET_Multifold_of_Lattices_Investigation_3_.md` — inter-tower Δk table §12.2
- `/mnt/project/ET_Weak_Sector_Open_Directions_Closed.md` — Route A/B closure theorems
- `/mnt/project/ET_Weak_Sector_Four_Open_Questions.md` — Route A/B canonical sequences
- `/mnt/project/ET_Quintic_Shadow_d5_Complete_Investigation.md` — Quintic Shadow α₅
- `/mnt/project/ET_Lagrangian_Field_Theory.md` — Mexican-hat Lagrangian
- `/mnt/project/ET_Traverser_T_Paper.md` — T-density / Scopaesthesia
- `/mnt/project/ET_Descriptor_D_Paper.md` — Descriptor primitive
- `/mnt/project/ET_Point_P_Paper.md` — Point primitive
- `/mnt/project/ET_Incoherence_Paper.md` — {P,T} state
- `/mnt/project/ET_Domain_Validity_Theorem.md` — domain applicability proofs
- `/mnt/project/ET_Programming_Math_Compendium.md` — programming math reference

### 6.3 New reference documents for Phase F (uploaded this session — read in full)

- `ET_Sempaevum_Paper16.tex` — Formalized paper. Losslessness Theorem (§18), Memoization Property (Corollary), ∂I Lattice-Aware Fractal formalization (§12), Self-Projection Identity (§18), Shimmer-α⁻¹ connection (Remark after Proposition: shimmer-V), Four Projection Paths (§14), EML operators (§13.2)
- `ET_Universal_Projection_Guide8.md` — v2.2 operational reference. Four Projection Paths (Part XIX §94–104), EML operators (Part XX §105–110), Python Reference Implementation (§110), Lattice Self-Projection Verification (§113), Axiom-Count Projections (§112), Complete Gaze Equation (Part XXII), Secret 26 Fully Generalized (Part XXIII), LCM Tower Escalation (§110)
- `ET_Four_Gaps_Verification.py` — 120 DPS lossless lattice verification exemplar. Demonstrates: mpmath + sympy Rational computation pattern, `project()` function as fundamental computation, cross-verification between symbolic and numeric forms, N-Exhaustion Theorem verification, Division Algebra route, Riemann C(n) identity
- `ET_Two_Critiques_Verification.py` — 120 DPS lattice verification exemplar. Demonstrates: full α⁻¹ computation at 120 DPS, closed-form identity verification (A₁ = √3/48, A_cross = √3/(93312π²), Σ A_k = 1/[216(18π−1)]), partial-sum convergence verification, V-threshold significance across LCM tower, lattice self-projection at multiple resolutions

---

## 7. External research data (verified 2026-04-09)

Captured here so a future conversation has the validated YouTube specs without needing to re-search.

### 7.1 YouTube official upload specs

**Source:** https://support.google.com/youtube/answer/1722171 — YouTube recommended upload encoding settings (current as of 2026)
**Source:** https://support.google.com/youtube/answer/6039860 — Encoding specifications for music videos

**Container:** MP4
**Video codec:** H.264 (high profile, CABAC, closed GOP at half frame rate)
**Audio codec:** AAC-LC (Opus and Eclipsa Audio also accepted)
**Sample rate:** 48 kHz recommended (44.1 kHz accepted but causes transcode + possible aliasing)
**Audio channels:** Stereo (5.1 surround also supported)
**Audio bitrate (stereo, official YouTube recommendation):** 384 kbps for AAC-LC
**Audio bitrate (music videos, YouTube recommendation):** 320 kbps minimum, "higher is always better"
**Color space:** BT.709 for SDR
**Chroma subsampling:** 4:2:0 (4:4:4 also accepted)
**moov atom placement:** front of file (Fast Start)
**For mastering uploads:** PCM/WAV/FLAC at 24-bit accepted (YouTube re-encodes anyway)

### 7.2 YouTube official video bitrate recommendations (H.264 SDR)

| Resolution | 30 fps | 60 fps |
|---|---|---|
| 1080p | 8 Mbps | 12 Mbps |
| 1440p | 16 Mbps | 24 Mbps |
| 2160p (4K) | 35–45 Mbps | 53–68 Mbps |
| 4320p (8K) | ~80 Mbps (extrapolated) | ~120 Mbps (extrapolated; YouTube doesn't publish 8K official numbers) |

**Industry consensus for surviving YouTube re-encoding:** Multiply YouTube's minimums by 1.5–2× as the upload floor. For "archival" / mastering: multiply by 2.5–4×.

### 7.3 The script's resolution presets vs YouTube's per-resolution bitrates

The script's presets are square (except 1080p), so per-pixel scaling matters more than horizontal-resolution scaling. The bitrate ladder in §3.8 was computed by:

1. Taking YouTube's official 4K-60 recommendation (60 Mbps midpoint)
2. Computing per-pixel bits: 60 Mbps / 8.29 MP ≈ 7.2 Mbps per megapixel
3. Scaling per the script's actual pixel counts
4. For YouTube-optimal tier, multiplying by 1.5×
5. For Archival tier, multiplying by 4×
6. For High tier, by 1.2×
7. For Standard / Draft tiers, falling back to CRF mode (variable bitrate)

This gives the ladder in §3.8.

### 7.4 ffmpeg flags reference (verified syntactically)

**Tier flag construction (Python f-string template), with codec branching for Archival sub-options:**

```python
def _build_video_ffmpeg_cmd(input_pattern, output_path, fps, tier):
    """
    input_pattern: 'frame_%06d.png' or 'frame_%06d.tiff' (per FRAME_FORMAT)
    tier: dict from _choose_output_quality() + _choose_archival_codec() if archival
    """
    codec = tier['video_codec']
    cmd = ['ffmpeg', '-y', '-r', str(fps),
           '-i', input_pattern,
           '-c:v', codec,
           '-pix_fmt', tier['video_pix_fmt'],
           '-color_primaries', 'bt709',
           '-color_trc', 'bt709',
           '-colorspace', 'bt709',
           '-r', '60']

    # Preset / rate-control branching by codec
    if codec == 'libsvtav1':
        # SVT-AV1 uses numeric preset (0..13, lower = slower/better) and tune
        # SVT-AV1 doesn't accept the libx264-style -preset names
        # Bitrate is set via -b:v alone (SVT-AV1 handles VBV internally)
        cmd.extend(['-b:v', f"{tier['video_b']}",
                    '-svtav1-params',
                    f"preset=2:keyint={tier['video_keyint']}:tune=0"])
    else:
        # libx264 and libx265 share the named-preset interface
        cmd.extend(['-preset', tier['video_preset']])

        if tier['tier'] in ('draft', 'standard'):
            crf = '23' if tier['tier'] == 'draft' else '20'
            cmd.extend(['-crf', crf])
        else:
            cmd.extend(['-b:v', f"{tier['video_b']}",
                        '-maxrate', f"{tier['video_maxrate']}",
                        '-bufsize', f"{tier['video_bufsize']}"])

        cmd.extend(['-profile:v', tier['video_profile'],
                    '-level', tier['video_level']])

        if tier['tier'] != 'draft':
            keyint = tier['video_keyint']
            params_key = '-x265-params' if codec == 'libx265' else '-x264-params'
            cmd.extend([params_key,
                        f'keyint={keyint}:min-keyint={keyint}:no-scenecut=1'])

    # Smooth zoom interpolation (preserved from current code)
    cmd.extend(['-vf',
                f'scale=trunc(iw/2)*2:trunc(ih/2)*2,'
                f'minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:'
                f'me_mode=bidir:vsbmc=1'])

    cmd.append(str(output_path))
    return cmd
```

**Mux command (Python):**

```python
def _build_mux_cmd(video_path, wav_path, output_path, aac_kbps):
    return ['ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(video_path),
            '-i', str(wav_path),                # WAV, not MP3 — no double compression
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', f'{aac_kbps}k',
            '-ar', '48000',
            '-shortest',
            str(output_path)]
```

**FLAC sidecar (Archival tier):**

```python
def _build_flac_cmd(wav_path, flac_path):
    return ['ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(wav_path),
            '-c:a', 'flac',
            '-compression_level', '8',
            '-ar', '48000',
            str(flac_path)]
```

**Frame format input pattern (in `generate_zoom_video`):**

```python
if FRAME_FORMAT == 'tiff':
    ffmpeg_in = str(frame_dir / 'frame_%06d.tiff')
else:
    ffmpeg_in = str(frame_dir / 'frame_%06d.png')
```

ffmpeg auto-detects the frame format from the file extension via its built-in `tiff` and `png` decoders. No `-f image2` or other input-format flag is required because the file extension is unambiguous.

---

## 8. Open items and risks

### 8.1 Items still to confirm during implementation

None — all design decisions are locked. The user has greenlit:
- D1: event rate menu with default 100 ev/s ✓
- D2: image duration menu with natural-duration repeat + natural-length fade ✓
- D3: full 21-voice catalog + 5 presets ✓
- D4: octave map + treble/bass shaping ✓
- D5: combined output quality tiers + HEVC option in Archival + FLAC sidecar ✓
- D6: Stage 21.5/21.6 placement ✓

### 8.2 Risks to watch for during implementation

**R1: Probe pass for post-iteration voices requires two-pass rendering.**
The voices that depend on final-state metrics (Highest-elegance, Boundary-tightness, Escape-time, Trap voices) can't have their pixels selected until the main render completes. So if those voices are selected, the render does:
1. Main pass over all pixels (existing behavior)
2. Compute the post-iteration metrics
3. Identify probe pixels
4. Probe pass over only those pixels
5. Music synthesis from the probe traces

For structurally-determined voices (Real-axis, Imag-axis, Diagonals, Radial fan, Ring radius), the probe pixels are known before the render starts, so they can be probed in pass 2 only. The structurally-determined voices' probe pass and the post-iteration voices' probe pass can be merged into a single pass 2 by computing the post-iteration metrics first.

Risk: doubles the iteration time for probe pixels (maybe 30 pixels — negligible compared to a million-pixel render).

**R2: Memory budget for very long iterations.**
At max_iter = 10,000,000 (ultra preset), per-step traces for 30 probes = 30 × 10M × 9 × 8 = ~21.6 GB at float64. The metabolism manages this: it computes K × available_RAM before the render begins and verifies the probe allocation fits. If it does not fit, the metabolism reports the shortfall explicitly and halts — it does NOT silently cap or truncate the probe data. Every iteration step is precious orbit data and must not be discarded. The user can then reduce max_iter or confirm they have sufficient memory.

**R3: GPU memory for probe trace buffers.**
The metabolism computes K × total_VRAM and verifies the probe allocation fits before the render begins. If VRAM is insufficient for full probe capture at the chosen max_iter, the metabolism reports this explicitly and halts — it does NOT silently cap or reduce precision. The user can then choose CPU path (which uses system RAM, typically much larger than VRAM) or reduce probe count. Precision is never compromised by VRAM constraints.

**R4: Mode dispatch in the probe kernel.**
The ∂I kernel has 11 modes spliced in via `_extract_main_kernel_mode_blocks`. The probe-pass kernel needs the same splicing to be consistent with the main render. **Solution:** the probe kernel is generated by the same splice mechanism — extract once, splice into both main and probe kernels. Don't duplicate logic.

**R5: F64 generation pipeline.**
Both main kernels have F64 versions auto-generated by `_make_f64_kernel` and `_make_f64_di_kernel`. The probe kernels need the same. **Solution:** add `et_iterate_probe_di_f64` and `et_iterate_probe_std_f64` generated by the same functions. Verify both compile.

**R6: ffmpeg version compatibility.**
The new flags (`yuv420p10le`, `high10`, `level 5.2`, `x264-params`) require recent ffmpeg builds. If any tier-specific flag causes ffmpeg to fail, this is a **fatal error** — the script halts and prints the exact ffmpeg command, its output, and the specific flag that failed, with instructions to update ffmpeg. There is NO quality-compromising fallback. The `cmd_raw` path (existing code — assembly without `minterpolate` optical-flow) is preserved ONLY as an error-reported path per convention 6 (`_et_error` with `fatal=False` and `fallback_msg`), and only for the `minterpolate` filter which is a post-processing step, not a quality-determining encode setting.

**R7: Voice 5 (Radial fan) is 12 individual probe pixels but produces ONE voice.**
The mix logic needs to know that voice 5's 12 probe pixels are summed into one voice stream, not 12. **Solution:** `voice_pixel_map` is a dict where each value is a list; the mix iterates over voices, not over probes.

**R8: Float64 propagation through the entire audio stack.**
Per §3.12, float64 fractal runs must produce float64 audio throughout the synthesis pipeline. The risks are:
- The probe kernel must declare its output buffers as `float` in F32 source and have them auto-converted to `double` by the existing `_make_f64_di_kernel` regex. **Mitigation:** the probe kernel uses the same idioms as the main kernels — `float`/`double` keyword, `f` suffix on literals, `f` suffix on intrinsics. The auto-converter will handle them identically.
- The CPU probe path must allocate with `FLOAT_DTYPE` not hardcoded `np.float32`. **Mitigation:** explicit code review during Phase B implementation; grep for `np.float32` in the probe path and verify each occurrence is correct (some constants legitimately stay float32 — e.g. `_D_LUT` is float32 by design).
- The synthesis primitives (`_audio_koide_tone`, `_synth_note_sequence`) already use `np.float64` internally; this is fine for both precisions but I should not accidentally cast inputs to float32 anywhere.
- The WAV writer must support all three formats (16/24/32-float) without falling back to 16-bit silently. **Mitigation:** explicit assertion at the top of `_audio_write_wav` that checks `bit_depth in (16, 24, 32)` and raises if not.
- The mux command must read the right WAV file (the higher-bit-depth one for higher tiers). **Mitigation:** the mux source path is explicitly set from the tier dict; there's no magic file-discovery.

**R9: SVT-AV1 availability in installed ffmpeg builds.**
SVT-AV1 (`libsvtav1`) is included in modern ffmpeg builds (since ffmpeg 4.3, released in 2020) and is present in essentially every recent Windows build (gyan.dev, BtbN, official binaries). However, very old ffmpeg installations or stripped builds may lack it. **Mitigation:** before running the SVT-AV1 command, the script runs `ffmpeg -h encoder=libsvtav1` and checks the return code. If libsvtav1 is unavailable, the script prints an explicit error explaining how to upgrade ffmpeg, and asks the user to either pick a different codec (libx264 or libx265) or cancel the render. No silent fallback — per the project rule, every fallback must be explicit and user-confirmed. The codec availability check runs once at startup.

**R10: TIFF intermediate frame disk space exhaustion.**
At 8K and 16K presets, TIFF intermediate frames can consume hundreds of GB or low TB of disk space. **Mitigation:** the disk-space warning printed at the `_choose_frame_format` prompt (per §3.13) computes the estimated total upfront and shows it to the user before they commit. Additionally, the script checks free disk space in the script directory via `shutil.disk_usage()` immediately after the prompt. If free space is less than `estimated × 1.2` (20% safety margin), the script prints a warning and asks the user to confirm they want to proceed. If the user is on Windows and the estimated total exceeds the free space on the drive containing the script, this is an explicit prompt — no silent failure mid-render.

**R11: Lattice precision mode operates at 120+ digit fixed-point. (Phase F)**
At 120+ digit fixed-point, each arithmetic operation involves ~7 uint64 limbs. The metabolism (K=2/3 + α⁻¹ impedance) manages all resource allocation — the user does not need to worry about speed or performance. **The lattice precision mode is a FULL PRODUCTION precision option**, not a verification-only mode. It generates the most precise fractals and music possible. If a render takes weeks or months, that is acceptable — precision and quality are the only constraints that matter. The C module is optimized for throughput (SIMD, loop unrolling, cache-line alignment) to make the best use of the K=2/3 resource allocation, but speed never trumps precision.

**R12: C module compilation at startup. (Phase F)**
The C module must compile on first run (or when missing). The compilation uses the same pattern as the ET CDF Compressor's `et_pattern_engine.c` — detect MSVC on Windows, GCC on Linux, compile with optimizations, cache the .dll/.so. **If compilation fails, that is a FATAL error** — print the compiler command and output, then halt. The lattice engine is not optional; it is the metabolism. No external library dependencies — pure C99 with POSIX sysinfo for memory detection.

**R13: EML operators are complex-domain internally. (Phase F)**
The EML operator `eml(x, y) = exp(x) − ln(y)` can produce complex results. For real-valued lattice arithmetic, the imaginary part should be negligible. **Mitigation:** the C module's EML functions assert `|im(result)| < 10^{-100}` for real-valued inputs and raise a diagnostic error if this fails.

**R14: Self-projection verification could fail on corrupted installs. (Phase F)**
If the ET constants in Stage 3 are accidentally changed (e.g. N set to 13), the self-projection verification will correctly catch this and halt with a fatal error. **Mitigation:** this is desired behavior — the self-projection IS the canary. The error message explains which constant is wrong and what the expected value is.

**R15: LCM tower resolution array is static. (Phase F)**
The tower `(12, 24, 36, 60, 84, 132, 420, 2520, 27720)` is a fixed list of the canonical refinements from ET. Per project rules, static lists should be dynamic. **Mitigation:** the tower is computed dynamically from the LCM of consecutive integers — `lcm(1..2)=2, lcm(1..3)=6, lcm(1..4)=12, ...` — and filtered to include only those LCMs that are ≥ 12 and ≤ 27720. The C module computes this at load time via `mp_gcd: lcm(a,b) = a·b/gcd(a,b)`. The resulting set is a structural consequence of N=12, not a hand-curated list.

**R16: Substrate projection requires hardware detection. (Phase F)**
The metabolism projects hardware characteristics (CPU cores, RAM, VRAM) onto the lattice. Hardware detection uses `os.cpu_count()` for CPU, platform-specific APIs for RAM (`psutil` or POSIX `sysinfo`), and CUDA APIs for VRAM. If hardware detection fails for any resource type, the metabolism reports the failure and uses a conservative default (the minimum structurally valid allocation: V × total for the detected resources). This is reported via `_et_error` as a non-fatal error with specific fallback explanation.

**R17: GPU arbitrary precision implementation. (Phase F)**
The C module's mpfx arithmetic uses uint64 arrays that are GPU-native. The companion CUDA kernel (`et_lattice_engine_kernel.cu`) implements the same algorithms in CUDA. The uint64-array representation maps directly to CUDA's `unsigned long long`. The binary GCD (Stein's algorithm) uses only bit shifts and comparisons — fully GPU-native. The binary log₂ method uses only squaring and comparison — fully GPU-native. The GPU path provides the same 120+ digit precision as the CPU path with no compromises. Both paths produce identical results — precision is never traded for speed.

### 8.3 Things explicitly NOT being done

- The script's iteration math is not being changed (except in Lattice precision mode where the same math is re-implemented in the C module at 120+ digit fixed-point). Only the output capture is extended.
- The rendering pipeline (escape coloring, normal-map lighting, orbit traps, interior coloring, ACES tone-map) is unchanged.
- The mode dispatch (Modes 0–11) is unchanged.
- The fractal type selection logic is unchanged.
- The `_resolve_run_params` function and tower system are unchanged.
- Anything in the Three Tools reference, the lattice corpus, the ET constants, or the corpus papers is treated as authoritative and not modified.
- The CPU/GPU dispatch, NVRTC bootstrap, GPU detection, or fallback logic is not modified except where probe pass plumbing is added and where Lattice precision mode disables GPU.
- No removal of any existing function, variable, or comment. Per user rule, even unused code gets traced and completed, never removed.
- Phase F does NOT replace the existing float computation paths — it ADDS a parallel lossless C-module-backed path. The float paths remain the primary computation paths for production renders.
- **No mpmath or sympy anywhere.** All arbitrary precision is implemented in the `et_lattice_engine.c` C module using custom multi-precision fixed-point arithmetic. No external Python numerical libraries beyond NumPy/CuPy (which are already used).

---

## 9. Convention reminders (project-wide rules in force during implementation)

These are non-negotiable per the user's instructions:

1. **No removal.** Unused code means incomplete implementation to be traced and completed, never deleted. Parameters, variables, and imports flagged as unused indicate missing functionality.
2. **Dynamic, not static.** Lists are forbidden where dynamic discovery is possible. No hardcoded counts, no hardcoded length checks against named constants. Use `getattr()`, `glob`, `__subclasses__()`, etc.
3. **Surgical edits only.** `str_replace` with exact multi-line matching. `create_file` is forbidden on the existing script.
4. **Read full files before editing.** Truncation is a cardinal error. The 7,989 lines of `ET_FRACTAL_GENERATOR50-10.py` were read in full this session.
5. **Trace cross-script before changing.** Every import, parameter, and call site must be traced before any modification.
6. **No silent failures — and a fallback is itself considered an error.** Every feature and every function SHALL do exactly what it is specified to do. If code reaches a fallback path, that fallback is treated as an error condition and MUST be reported via `_et_error(..., fatal=False, fallback_msg=...)` or `_et_fallback(context, reason, fallback_msg)` — never silently. The existing `_et_error` and `_et_fallback` infrastructure is the required logging mechanism; new code reuses it and does not invent its own silent paths. **Narrow exception for structural branches only:** code that selects between platform-equivalent or capability-equivalent alternatives as a normal, expected structural decision is NOT a fallback and does NOT require error logging. Examples of allowed structural branches: `if sys.platform == 'win32'` vs Linux/macOS paths, `if USE_GPU` vs CPU path (both are first-class supported code paths), `if FLOAT_DTYPE == np.float64` vs float32 selection (both are user-selected precisions), `if FRAME_FORMAT == 'tiff'` vs PNG (both are user-selected frame formats), `if codec == 'libsvtav1'` vs libx264/libx265 branches at the Archival tier. These are not fallbacks — they are design-level branches where both sides are equally valid primary paths. Everything else IS a fallback and IS an error: a CUDA kernel failing and dropping to CPU, a raw probe path returning None and forcing the image-scan fallback, an ffmpeg optical-flow step failing and dropping to raw assembly, an MP3 encode failing and leaving only WAV, a FLAC sidecar step failing — all of these MUST print through `_et_error` or `_et_fallback` with the specific reason and the specific fallback action being taken. The rule: the code SHOULD and SHALL do exactly what it should; if it doesn't, that's an error, and the user is told.
7. **All math ET-derived.** All thresholds, weights, durations, EQ coefficients derive from K, V, N, N_ET, K^n, √V, etc. No tuning, no ad-hoc, no external axioms (CODATA imports etc. are forbidden except as comparison).
8. **Three Tools applied.** Identification first (P, D, T of the new music engine), then Descriptor Gap (what's missing in the current code), then Subsumption (does the new code subsume the music-native semantic without remainder).

### 9.1 Three Tools applied to this rebuild

**Identification:**
- **P** (substrate) = the orbit's complex position `z_n` at every step n, for every probe pixel
- **D** (constraint) = the 27720ET lattice projection `(k_r, k_θ, ε_r, ε_θ, t_r, t_θ, d_r, d_θ, p_dom, Ψ_n)` computed at every step
- **T** (agency) = the per-step iteration that advances the orbit and selects voices, plus the music synthesis that turns the trace into audio samples

**Descriptor Gap closed:**
- Before: the kernel computed (D) every step but discarded it after each step → no music could be derived from the orbit's trajectory → all music came from re-deriving (D) from the rendered RGB, lossily.
- After: the kernel writes (D) per step for probe pixels → music is the direct synthesis of (D) → no re-derivation, no loss, no sonification.
- The closing Descriptor is the **per-step probe trace buffer** plus the **probe pixel selection function** plus the **per-voice synthesis function** plus the **phase-continuous frame-boundary handoff**.

**Subsumption:**
- Every voice in §3.4 is a different way of selecting (sub-P) from the manifold's pixel space; each sub-P has its own (D) trace; each (D) trace becomes audio via the same per-step synthesis. The voices subsume "all the ways an orbit can be musically meaningful in this manifold." 
- The video frame-boundary handoff subsumes the phase-continuity requirement without remainder: the music_state object carries everything across frames so nothing is lost at boundaries.
- The output quality tier subsumes the YouTube-survival requirement without remainder: bitrate, codec, profile, level, pixel format, sample rate, and audio bitrate are all set together so the upload meets YouTube's published specs at the chosen tier.
- No remainder, no leftover fudge factors, no manual tuning. Everything traceable to ET constants or to YouTube's published spec.

---

## 10. Resumption protocol for future conversations

If the implementation is interrupted (context limit, session change, etc.), the next conversation can resume by:

1. **Reading this document in full.** It contains every decision, every number, every code structure.
2. **Re-reading `ET_FRACTAL_GENERATOR50-10.py` in full** (no truncation, per the user's standing rule).
3. **Re-reading `dI_Fractal_Explanation.md`** (the lattice-aware fractal specification).
4. **Re-reading `/mnt/project/ET_Three_Tools_Complete_Reference.md`** (required per the user's project-wide rule for any problem-solving conversation).
5. **Checking git/file diff or grep for `sonif`** to see how far Phase A got.
6. **Checking grep for `et_native_music` and `_native_music_synth_voice`** to see how far Phase B/C got.
7. **Checking grep for `lattice_project` and `lattice_self_verify`** to see how far Phase F got.
8. **Asking the user for the latest script version** if there is any chance the file has been modified between sessions.
9. **Picking up at the next un-completed phase step.**

The user has stated: "we do not care about tool or context limits, as we can just continue when you reach a limit. We need accuracy and precision."



---

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

**End of plan document v2 — Lattice Computation Engine Extension.**

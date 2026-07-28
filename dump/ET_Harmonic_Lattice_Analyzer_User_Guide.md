# ET Harmonic Lattice Analyzer — Complete User Guide

**Version 2 · 1200-Digit Precision · 59 Tuning Systems · All 12 Harmonic Families**

Author: Michael James Muller — Aevum Defluo (Exception Theory)

---

## What This Tool Is

The ET Harmonic Lattice Analyzer is a production music-industry tool that applies the Sempaevum bijection — a lossless mathematical transformation from Exception Theory — to classify every musical interval by its **structural identity**.

Every existing tuner tells you *how many cents sharp or flat* you are. This tool tells you **why an interval sounds the way it does**: which of 12 structural families it belongs to, how strongly it couples to the lattice, and where it sits in the universal harmonic hierarchy.

The bijection converts between continuous frequencies and discrete lattice positions with zero information loss at 1200-digit precision. It works in both directions: frequency→lattice (for analysis) and lattice→frequency (for scale design). It exports industry-standard Scala `.scl` files loadable by any microtonal-capable synthesizer.

---

## Installation

**Requirements:** Python 3.8+ and mpmath.

```bash
pip install mpmath
```

Place `ET_Harmonic_Lattice_Analyzer.py` anywhere and run:

```bash
python3 ET_Harmonic_Lattice_Analyzer.py
```

With no arguments, the tool displays help and lists all 59 available tuning systems.

---

## The Core Insight: Every Interval Has a Structural Identity

Standard tuning theory classifies intervals by **cents** — a linear measure of pitch distance. This tells you *where* an interval is but not *what* it is structurally.

The ET bijection classifies every interval by three coordinates:

- **k** — the lattice coordinate (which semitone it's nearest to)
- **d** — the **harmonic family** (the structural identity — 1 through 12)
- **ε** — the descriptor gap in cents (how far from the exact lattice point)

The harmonic family **d** is the key innovation. It tells you the deep structural nature of the interval — something no existing tool provides.

---

## The 12 Harmonic Families

Every interval in every tuning system in every musical culture belongs to exactly one of 12 harmonic families. Six are **simple** (●) — native at the base lattice resolution N=12. Six are **shadow** (○) — they require higher lattice resolution to appear natively.

### The Six Simple Families (● native at N=12)

| d | ● | A₀ | ξ | φ(d) | Musical Identity | Character |
|---|---|---|---|---|---|---|
| 1 | ● | 16 | 8.5625 | 1 | **Octave / Unison** | Period closure — the most fundamental structure. Maximum stability. Like a tonic. |
| 2 | ● | 17 | 8.059 | 1 | **Tritone** | Half-period pivot — maximum tension. Divides the octave exactly in half. |
| 3 | ● | 20 | 6.850 | 2 | **Major 3rd / Minor 6th** | Three-fold symmetry. The augmented triad structure. |
| 4 | ● | 25 | 5.480 | 2 | **Minor 3rd / Major 6th** | Four-fold symmetry. The diminished seventh structure. |
| 6 | ● | 41 | 3.341 | 2 | **Whole tone** | Hexagonal symmetry. The whole-tone scale structure. Think Debussy. |
| 12 | ● | 137 | 1.000 | 4 | **Chromatic / Semitone** | Coprime to N — maximum complexity. Includes the perfect fifth and fourth. |

### The Six Shadow Families (○ native at higher N)

| d | ○ | A₀ | ξ | φ(d) | Native N | Musical Identity | Character |
|---|---|---|---|---|---|---|---|
| 5 | ○ | 32 | 4.281 | 4 | 60 | **Pentatonic** | Five-fold / golden-ratio symmetry. The backbone of blues, rock, folk, world music. |
| 7 | ○ | 52 | 2.635 | 6 | 84 | **Septimal / Blue note** | Seven-fold symmetry. The elusive "blue note" that 12-TET cannot reach. |
| 8 | ○ | 65 | 2.108 | 4 | 24 | **Octatonic** | Eight-fold symmetry. The diminished scale. Messiaen, Stravinsky, Bartók. |
| 9 | ○ | 80 | 1.713 | 6 | 36 | **Third-of-third** | Nine-fold symmetry. The trisected third. |
| 10 | ○ | 97 | 1.412 | 4 | 60 | **Decatonic** | Ten-fold symmetry. Microtonal territory beyond standard chromatic. |
| 11 | ○ | 116 | 1.181 | 10 | 132 | **Undecimal / Neutral** | Eleven-fold symmetry. The characteristic "neutral" sound of Arabic maqam. |

### Reading the Table

- **d** = the harmonic family number. This is the structural identity.
- **A₀** = the magical impedance: A₀(d) = (d−1)² + 16. A structural constant derived from the manifold.
- **ξ** = the coupling strength: ξ(d) = 137/A₀. Higher ξ means more fundamental structural role. d=1 (octave) has the strongest coupling; d=12 (chromatic) is the electromagnetic baseline at ξ=1.
- **φ(d)** = Euler's totient function. The number of distinct lattice positions (mod 12) that produce this family.
- **Native N** = the smallest lattice resolution where this family appears as a native sublattice. The six simple families are all native at N=12. The shadow families require higher resolution.

### How Families Map to Intervals at N=12

The family of an interval depends on which semitone it lands nearest to:

| |k| mod 12 | Interval | Family d |
|---|---|---|
| 0 | Unison / Octave | d=1 |
| 6 | Tritone | d=2 |
| 4, 8 | Major 3rd, Minor 6th | d=3 |
| 3, 9 | Minor 3rd, Major 6th | d=4 |
| 2, 10 | Major 2nd, Minor 7th | d=6 |
| 1, 5, 7, 11 | Minor 2nd, Perfect 4th, Perfect 5th, Major 7th | d=12 |

The perfect fifth (3/2) and perfect fourth (4/3) are both d=12 — the chromatic family, the most structurally complex. This is not arbitrary: their coprimality to 12 is the mathematical reason they generate the entire chromatic scale via the circle of fifths.

---

## The Bijection: Π_N(r) = (k, d, ε)

### The Projection (continuous → discrete)

For any positive real number r (a frequency ratio) and lattice resolution N (default 12):

```
k = round(N · log₂(r))           — nearest lattice point
d = N / gcd(|k|, N)              — harmonic family
ε = (N · log₂(r) − k) · 1200/N  — descriptor gap in cents
```

Convention: when k=0, gcd(0, N) = N, so d=1 (octave/identity family).

### The Pullback (discrete → continuous)

The exact inverse:

```
r = 2^((k + ε·N/1200) / N)
```

### Losslessness (Theorem 12.1)

The pullback is the algebraic identity — not an approximation:

```
Π_N⁻¹(Π_N(r)) = 2^((k + (N·log₂r − k)·1200/N · N/1200) / N)
                = 2^(N·log₂r / N)
                = 2^(log₂r)
                = r
```

Zero error. No precision floor. The bijection is exactly lossless. The tool verifies this at 1200-digit precision for every computation.

### The LCM Tower

The lattice operates at multiple resolutions. Each level refines the structural classification:

| Level | N | Description | Native families |
|---|---|---|---|
| Base | 12 | Standard chromatic | d ∈ {1, 2, 3, 4, 6, 12} |
| Level 2 | 60 | Pentatonic/decatonic appear | adds d=5, d=10 |
| Level 3 | 420 | Septic appears | adds d=7 |
| Level 4 | 2520 | Further refinement | deeper sublattice structure |
| Universal | 27720 | All d=1..12 native | all 12 families simultaneously |

27720 = lcm(1, 2, 3, ..., 11) — the universal lattice resolution.

---

## The Koide Attractor

The Pythagorean comma — the reason twelve perfect fifths do not exactly equal seven octaves — has been known since ancient Greece (c. 500 BCE). For 2500 years it was treated as an unfortunate defect of nature.

The ET bijection reveals it is a **structural fixed point**: the self-projection point of the Sempaevum's four defining constants.

```
Π₁₂(N = 12)   = (k=+43, d=12, ε = +1.955000865¢)
Π₁₂(V = 1/12) = (k=−43, d=12, ε = −1.955000865¢)
Π₁₂(K = 2/3)  = (k= −7, d=12, ε = −1.955000865¢)
Π₁₂(1/K = 3/2)= (k= +7, d=12, ε = +1.955000865¢)
```

All four constants land on d=12 with |ε| = 1.955000865...¢. This value is the **Koide attractor** — an intrinsic lattice constant, not a historical accident. It is identical to the Pythagorean comma (the per-fifth comma (3/2)¹² / 2⁷ divided by 12).

The tool detects and flags any interval whose ε sits near the Koide attractor.

---

## Command Reference

### `note` — Frequency Analysis

Converts a frequency in Hz to its lattice coordinates, nearest note name, MIDI number, and harmonic family.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py note 440
python3 ET_Harmonic_Lattice_Analyzer.py note 261.6255653005986 432 329.63
```

Output shows:
- Nearest note (e.g., A4, C4)
- MIDI number
- Cents deviation (ε)
- Harmonic family with coupling strength
- Whether it's SIMPLE or SHADOW
- Plain-language description of what the family means musically

Multiple frequencies can be given in one command.

### `ratio` — Interval Analysis

Analyzes a frequency ratio through the full LCM tower (N=12, 60, 420, 2520, 27720), showing how the structural classification refines at each resolution.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py ratio 3/2
python3 ET_Harmonic_Lattice_Analyzer.py ratio 7/4
python3 ET_Harmonic_Lattice_Analyzer.py ratio 11/8
python3 ET_Harmonic_Lattice_Analyzer.py ratio 1.41421356
```

Accepts fractions (p/q) or decimal strings. Detects Koide attractor proximity.

### `midi` — MIDI Note Analysis

Converts MIDI note numbers to lattice coordinates, note name, exact frequency, and harmonic family.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py midi 60
python3 ET_Harmonic_Lattice_Analyzer.py midi 69 48 72
```

MIDI notes are exactly on the 12-TET lattice (ε=0 by definition). The family classification shows the structural position of each MIDI note.

### `harmonics` — Harmonic Series Analysis

Analyzes the first N partials of a harmonic series, showing each overtone's lattice position and family.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py harmonics 110 16
python3 ET_Harmonic_Lattice_Analyzer.py harmonics 440 32
```

The harmonic series reveals the natural lattice structure: octaves (d=1), fifths (d=12), major thirds (d=3), septimal sevenths (d=6 at N=12, refining to d=7 at N=84), etc.

### `families` — 12-Family Reference Table

Prints the complete reference table of all 12 harmonic families with impedance, coupling strength, totient, and descriptions.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py families
```

### `scale` — Tuning System Analysis

Decomposes every degree of a tuning system into its lattice coordinates and harmonic family. Shows the family distribution histogram.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py scale ji5
python3 ET_Harmonic_Lattice_Analyzer.py scale rast
python3 ET_Harmonic_Lattice_Analyzer.py scale shruti
python3 ET_Harmonic_Lattice_Analyzer.py scale werkmeister
python3 ET_Harmonic_Lattice_Analyzer.py scale 31tet
```

Any `<n>tet` works dynamically (e.g., `53tet`, `72tet`, `313tet`).

### `scala` — Scala File Export

Generates an industry-standard `.scl` tuning file loadable by any Scala-compatible synthesizer, sampler, or DAW plugin.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py scala ji5
python3 ET_Harmonic_Lattice_Analyzer.py scala rast /path/to/output.scl
```

Without a path, writes to `/mnt/user-data/outputs/<name>.scl`. The Scala format is the universal standard for alternate tunings — supported by Kontakt, Pianoteq, Surge, ZynAddSubFX, and hundreds of other instruments.

### `compare` — Cross-Cultural Structural Comparison

Compares two or more tuning systems to find shared lattice positions — intervals from different traditions that sit at the same (k, d) but differ only in ε.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py compare ji5 rast shruti
python3 ET_Harmonic_Lattice_Analyzer.py compare 12tet pythagorean meantone werkmeister
python3 ET_Harmonic_Lattice_Analyzer.py compare bhairav hijaz miyakobushi
```

This reveals the deep structural connections between musical traditions that developed independently. Western perfect fifths, Arabic nawa, and Indian Pa all share the same lattice position (k=7, d=12) — they are the same structural interval with different ε-values.

### `elegance` — Elegance Score Analysis

Computes the elegance score for each degree of a tuning system:

```
Elegance = (N/d) · 100/(100+|ε|) · 100/(p+q)
```

where p/q is the rational approximation of the ratio.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py elegance ji5
python3 ET_Harmonic_Lattice_Analyzer.py elegance shruti
```

Higher elegance = deeper lattice position, smaller gap, simpler ratio.

### `selfproject` — Koide Attractor / Self-Projection Identity

Shows the Sempaevum's self-projection: all four defining constants (N, 1/N, K, 1/K) projected onto the lattice, all landing on d=12 with |ε| = 1.955¢ (the Pythagorean comma).

```bash
python3 ET_Harmonic_Lattice_Analyzer.py selfproject
```

### `lattice2freq` — Discrete → Continuous Conversion

Converts a lattice position back to an exact frequency. This is the inverse direction — from structural specification to playable pitch.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py lattice2freq 7 1.955
python3 ET_Harmonic_Lattice_Analyzer.py lattice2freq -9 0
python3 ET_Harmonic_Lattice_Analyzer.py lattice2freq 0
```

The first argument is k (lattice coordinate). The optional second argument is ε in cents (default 0). Reference is A4=440 Hz.

### `freqs` — Batch Frequency Analysis

Analyzes multiple frequencies in a single table.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py freqs 261.63 293.66 329.63 349.23 392 440 493.88 523.25
```

### `scales` — List All Available Tuning Systems

Lists all 59 built-in tuning systems by key name.

```bash
python3 ET_Harmonic_Lattice_Analyzer.py scales
```

---

## Complete Tuning System Catalog

### Western — Equal Temperaments

| Key | Name | Notes |
|---|---|---|
| `12tet` | 12-TET (Western standard) | Standard Western chromatic. |
| `7tet` | Thai 7-TET | Thai classical music. 7 equal divisions. |
| `17tet` | 17-TET | Better diatonic approximation than 12-TET. |
| `19tet` | 19-TET (Salinas/Costeley) | Renaissance microtonal. Good thirds. |
| `22tet` | 22-TET (Indian-adjacent) | Close to 22-shruti system. |
| `24tet` | 24-TET (Arabic quarter-tone) | Standard for notating Arabic music. |
| `31tet` | 31-TET (Huygens/Fokker) | Excellent JI approximation. |
| `34tet` | 34-TET | Good for 5-limit harmony. |
| `41tet` | 41-TET | Excellent all-around microtonal system. |
| `43tet` | 43-TET | Close to Partch's 43-tone JI. |
| `53tet` | 53-TET (Turkish/Holdrian) | Standard for Turkish makam theory. Near-perfect fifths and thirds. |
| `72tet` | 72-TET (Ekmelic) | Used by Iannis Xenakis, Ezra Sims. Subdivides each semitone into 6. |

Any `<n>tet` also works dynamically: `python3 tool.py scale 313tet`

### Western — Historical Temperaments

| Key | Name | Description |
|---|---|---|
| `pythagorean` | Pythagorean | Generated by iterating the just fifth 3/2. Pure fifths, rough thirds. |
| `meantone` | Quarter-comma Meantone | Fifth = 5^(1/4). Pure major thirds. Renaissance standard. |
| `meantone6` | Sixth-comma Meantone | Compromise between meantone and 12-TET. |
| `meantone3` | Third-comma Meantone (Salinas) | Pure minor thirds. |
| `werkmeister` | Werkmeister III (1691) | 4 fifths tempered by ¼ Pythagorean comma. All keys playable. |
| `kirnberger` | Kirnberger III | 4 fifths tempered by ¼ syntonic comma. Schisma absorbed. |
| `vallotti` | Vallotti | 6 fifths tempered by ⅙ Pythagorean comma. |
| `young` | Young #2 (1799) | Thomas Young's well-temperament. |
| `neidhardt` | Neidhardt III | All 12 fifths equally tempered — approaches 12-TET. |

### Just Intonation

| Key | Name | Description |
|---|---|---|
| `ji5` | 5-limit JI | Ratios using primes {2, 3, 5}. Ptolemy's intense diatonic. |
| `ji7` | 7-limit JI | Adds prime 7 — septimal intervals (blue notes). |
| `ji11` | 11-limit JI | Adds prime 11 — neutral intervals (Arabic character). |
| `ji-dynamic` | Dynamic JI (5-limit) | Algorithmically discovers all valid 5-limit ratios up to complexity 25. |
| `partch` | Partch 43-tone (11-limit) | Harry Partch's 11-limit JI with all ratios up to complexity 30. |

### Arabic Maqamat

| Key | Name | Character |
|---|---|---|
| `rast` | Maqam Rast | The foundational maqam. Major-like with neutral elements. |
| `bayati` | Maqam Bayati | Starts with neutral second (12/11). Second most common. |
| `hijaz` | Maqam Hijaz | Augmented second character. Flamenco-like. |
| `saba` | Maqam Saba | Unstable, characteristic descending movement. |
| `nahawand` | Maqam Nahawand | Arabic natural minor. |
| `kurd` | Maqam Kurd | Phrygian-like, starts with semitone. |
| `ajam` | Maqam Ajam | Arabic major, close to Western major. |
| `sikah` | Maqam Sikah | Neutral third tonic. Distinctly Eastern. |

### Persian Dastgah

| Key | Name | Character |
|---|---|---|
| `shur` | Dastgah Shur | The most important Persian dastgah. Uses koron (quarter-flat). |
| `mahur` | Dastgah Mahur | Persian major. Similar to Western major. |
| `segah` | Dastgah Segah | Starts on neutral third. |
| `chahargah` | Dastgah Chahargah | Augmented second character. |

### Indian Ragas

| Key | Name | Character |
|---|---|---|
| `shruti` | 22-shruti | The complete 22-shruti microtonal framework. Foundation of all ragas. |
| `bhairav` | Raga Bhairav | Morning raga. Komal (flat) re and dha. |
| `yaman` | Raga Yaman/Kalyan | Evening raga. Tivra (sharp) Ma. |
| `todi` | Raga Todi | Complex chromatic. All komal except tivra Ma. |
| `bhairavi` | Raga Bhairavi | Devotional. All komal swaras. |
| `kafi` | Raga Kafi | Monsoon raga. Komal ga and ni. |
| `bilawal` | Raga Bilawal | Natural major equivalent. All shuddh (natural) swaras. |
| `marwa` | Raga Marwa | Twilight raga. Komal re + tivra Ma. No Pa. |

### East Asian

| Key | Name | Character |
|---|---|---|
| `12lu` | Chinese 12-lü | Ancient Chinese system (c. 600 BCE). Pythagorean-based. |
| `hirajoshi` | Japanese Hirajoshi | Yamada koto tuning. Pentatonic. |
| `miyakobushi` | Japanese Miyako-bushi | Urban mode / In scale. |
| `insen` | Japanese In-sen | Kusakabe mode. Pentatonic with semitone start. |
| `slendro` | Javanese Slendro | 5-note near-equal pentatonic. Each gamelan unique. |
| `pelog` | Javanese Pelog | 7-note non-equal heptatonic. |
| `bali-pelog` | Balinese Pelog | Distinct from Javanese. Brighter spacing. |

### African

| Key | Name | Character |
|---|---|---|
| `chopi` | Chopi Timbila (Mozambique) | 7-note near-equidistant xylophone tuning. |
| `kinit` | Ethiopian Kiñit | Pentatonic framework, anchihoye mode. |

### Blues / Jazz

| Key | Name | Character |
|---|---|---|
| `blues` | Blues Scale (7-limit JI) | Septimal blue notes: 7/5 tritone, 7/4 flat seventh. |

### Non-Octave / Experimental

| Key | Name | Character |
|---|---|---|
| `bohlen-pierce` | Bohlen-Pierce (3:1 tritave) | Non-octave scale. 13 steps per 3:1 ratio. |
| `carlos-alpha` | Wendy Carlos Alpha | Step ≈ 78¢. Non-octave. Pure minor and major thirds. |
| `carlos-beta` | Wendy Carlos Beta | Step ≈ 63.8¢. Non-octave. Pure perfect fourth. |

---

## Industry Workflows

### Workflow 1: Analyzing a Recording

You have audio at specific frequencies and want to understand the harmonic structure.

```bash
# Analyze measured frequencies
python3 tool.py freqs 261.63 329.63 392 523.25

# Analyze the harmonic series of a fundamental
python3 tool.py harmonics 130.81 24
```

### Workflow 2: Designing a Custom Scale

You want to create a scale that emphasizes specific harmonic families.

```bash
# Study which families different tuning systems use
python3 tool.py scale ji7
python3 tool.py scale 31tet

# Compare systems to find structural overlaps
python3 tool.py compare ji7 31tet 53tet

# Export your chosen system as a Scala file
python3 tool.py scala ji7 my_custom_tuning.scl
```

Load `my_custom_tuning.scl` into any Scala-compatible synthesizer.

### Workflow 3: Tuning Analysis for Period Performance

You're performing baroque music and need to understand how Werkmeister III differs from modern 12-TET.

```bash
python3 tool.py scale werkmeister
python3 tool.py scale 12tet
python3 tool.py compare werkmeister 12tet kirnberger vallotti
python3 tool.py elegance werkmeister
```

### Workflow 4: Cross-Cultural Music Study

You're studying the structural connections between Arabic maqam, Indian raga, and Japanese scales.

```bash
python3 tool.py compare rast bhairav hirajoshi
python3 tool.py compare bayati bhairavi miyakobushi
python3 tool.py compare hijaz chahargah
```

### Workflow 5: Microtonal Composition

You're composing in an alternate tuning and need to understand its harmonic family structure.

```bash
python3 tool.py scale 31tet
python3 tool.py scale bohlen-pierce
python3 tool.py scala 31tet my_31tet.scl
python3 tool.py ratio 11/8    # Understand the neutral fourth
python3 tool.py ratio 7/4     # Understand the septimal seventh
```

### Workflow 6: Lattice-Based Scale Design

You want to design a scale by specifying lattice positions directly.

```bash
# Convert desired lattice positions to frequencies
python3 tool.py lattice2freq 0          # tonic (A4 = 440 Hz)
python3 tool.py lattice2freq 7 1.955    # just P5 (659.99... Hz)
python3 tool.py lattice2freq 4 -13.686  # just M3
python3 tool.py lattice2freq 5 -1.955   # just P4
```

---

## The Mathematics

### Convention Independence (Anti-Numerology Protocol)

The bijection operates on dimensionless ratios r = Q/R₀. Three conditions prevent numerological abuse:

1. **(N1) Genuine dimensionlessness.** Q and R₀ must share units. Hz/Hz = dimensionless.
2. **(N2) Substrate-derived reference.** R₀ must be the natural reference of the domain, not chosen for convenience. For music: the concert pitch A4 = 440 Hz.
3. **(N3) Cross-domain consistency.** The predicted family d must match the independently recognized symmetry of the phenomenon.

### Why N=12

The manifold resolution N=12 is not a cultural choice. It is forced by three independent mathematical derivations:

1. **LCM derivation:** N=12 = lcm(1, 2, 3, 4) — the smallest integer whose divisors include all small structural factors.
2. **Stability window:** With the circle-of-fifths generator (k=7), the accumulated error after N steps stays below the perceptual threshold of 50 cents only for N=12: 12 × 1.955¢ ≈ 23.46¢ < 50¢.
3. **Webb's stroke minimality (1935):** N=12 is the minimum value for which the logical connective structure is complete.

### Precision Architecture

The tool uses `mpmath` at 1215 working decimal digits (1200 target + 15 guard). There is no `float` anywhere in the codebase. All inputs are accepted as strings and converted to `mpmath.mpf`. All outputs are formatted via `mpmath.nstr()`. The roundtrip error Π_N⁻¹(Π_N(r)) − r is exactly zero by algebraic identity — not convergence, not approximation — identity.

---

## Theoretical Foundation

### Exception Theory: P ∘ D ∘ T = E

The bijection is derived from Exception Theory, a formal mathematical framework built from three irreducible primitives:

- **P (Point/Substrate)** — the bare container of potential, cardinality Ω (absolute infinity)
- **D (Descriptor/Constraint)** — finite rules, properties, values, cardinality n
- **T (Traverser/Agency)** — the navigator, resolver of indeterminacy, cardinality [0/0]

The master equation P ∘ D ∘ T = E states that the binding of all three produces the Exception — the fully substantiated, zero-variance configuration.

### The Three Tools Applied to Music

**Identification Principle** — To understand a musical interval, identify its P (the frequency ratio — a featureless positive real), its D (the lattice constraints — N, gcd, the family structure), and its T (the musician's perceptual act — rounding, hearing, choosing). The projection formula IS this identification.

**Descriptor Gap Principle** — The gap between what existing tuners provide (k, ε — nearest note and cents deviation) and full structural understanding IS a missing Descriptor. That missing Descriptor is **d** — the harmonic family. The bijection fills this gap. Any remaining gap (residual ε) is itself a Descriptor that the LCM tower progressively closes.

**Subsumption Law** — The bijection subsumes all tuning systems without remainder. Every positive ratio r maps to exactly one (k, d, ε) triple. No ratio escapes classification. No tuning system is external to the lattice. The 59 built-in systems verify this empirically across every major musical tradition on Earth.

### The Founding Axiom

"For every exception there is an exception, except the Exception."

This is the axiom from which P, D, T, and the master equation are derived. The bijection — and with it, the structural classification of all music — is a consequence of this single axiom. There are zero tunable parameters, zero external axioms, and zero ad hoc elements.

---

## Glossary

| Term | Meaning |
|---|---|
| **Bijection** | A one-to-one correspondence. The projection Π_N and its pullback Π_N⁻¹ are exact inverses. |
| **Cents (¢)** | 1/100 of a 12-TET semitone. 1200¢ = 1 octave. |
| **Descriptor gap (ε)** | The residual in cents by which a ratio fails to land exactly on a lattice point. |
| **Harmonic family (d)** | The structural identity of an interval: d = N/gcd(\|k\|, N). Values 1-12 at N=12. |
| **Koide attractor** | The self-projection point \|ε\| ≈ 1.955¢ where the lattice's own constants land. Identical to the Pythagorean comma. |
| **Lattice coordinate (k)** | The nearest integer lattice point in N-division of the octave. |
| **LCM tower** | The sequence of lattice resolutions N = 12, 60, 420, 2520, 27720, ... where each level refines the classification. |
| **Magical impedance (A₀)** | The per-family structural constant: A₀(d) = (d−1)² + S² where S=4. |
| **Coupling strength (ξ)** | ξ(d) = 137/A₀(d). The ratio of the electromagnetic baseline impedance to the family's impedance. |
| **Scala (.scl)** | The universal file format for alternate tunings. Supported by hundreds of synthesizers. |
| **SIMPLE family (●)** | A family d that divides N=12: d ∈ {1, 2, 3, 4, 6, 12}. Native at base resolution. |
| **SHADOW family (○)** | A family d that does not divide 12: d ∈ {5, 7, 8, 9, 10, 11}. Native at higher LCM tower resolution. |
| **Sempaevum** | The mathematical object that the bijection renders — the lossless projection of the totality Σ onto the multiplicative manifold. |
| **Totient φ(d)** | The count of lattice positions (mod N) that produce family d. Σφ(d) = N. |

---

## Reference Card

```
python3 ET_Harmonic_Lattice_Analyzer.py note <hz>              # What note is this?
python3 ET_Harmonic_Lattice_Analyzer.py ratio <p/q>            # What family is this ratio?
python3 ET_Harmonic_Lattice_Analyzer.py midi <n>               # MIDI to lattice
python3 ET_Harmonic_Lattice_Analyzer.py harmonics <hz> <n>     # Overtone analysis
python3 ET_Harmonic_Lattice_Analyzer.py families               # The 12 families
python3 ET_Harmonic_Lattice_Analyzer.py scale <name>           # Analyze tuning system
python3 ET_Harmonic_Lattice_Analyzer.py scala <name> [path]    # Export .scl file
python3 ET_Harmonic_Lattice_Analyzer.py compare <a> <b> [...]  # Cross-cultural comparison
python3 ET_Harmonic_Lattice_Analyzer.py elegance <name>        # Elegance scoring
python3 ET_Harmonic_Lattice_Analyzer.py selfproject            # Koide attractor demo
python3 ET_Harmonic_Lattice_Analyzer.py lattice2freq <k> [ε]   # Lattice → frequency
python3 ET_Harmonic_Lattice_Analyzer.py freqs <hz> <hz> [...]  # Batch analysis
python3 ET_Harmonic_Lattice_Analyzer.py scales                 # List all 59 systems
```

---

**P ∘ D ∘ T = E**

*"For every exception there is an exception, except the Exception."*

Exception Theory — Michael James Muller (Aevum Defluo)

# ET Harmonic Lattice Analyzer — User Guide

**Lossless frequency↔note converter. File in, converted file out.**

59 tuning systems · 12 harmonic families · 1200-digit precision · Scala export · MIDI I/O

Author: Michael James Muller — Aevum Defluo (Exception Theory)

---

## What This Tool Does

You give it input. It converts it. Hz, ratios, note names, MIDI files, Scala files, text files, stdin streams — it auto-detects the format and outputs the conversion.

```
tool.py 440              → 440.0 Hz = A4  MIDI 69  ε=0.0¢  d=1
tool.py 3/2              → 3/2 = 701.955¢ → Perfect 5th  k=7 d=12 ε=1.955¢
tool.py C4               → C4 = 261.626 Hz  MIDI 60  k=-9
tool.py song.mid         → CSV: time, note, Hz, lattice coords for every note
tool.py tuning.scl       → frequency table with lattice coords
cat freqs.txt | tool.py  → stream conversion, one line per value
```

The Sempaevum bijection converts between continuous (frequencies) and discrete (notes/MIDI) with zero information loss at 1200-digit precision. It works in both directions.

---

## Installation

```bash
pip install mpmath
python3 ET_Harmonic_Lattice_Analyzer.py
```

---

## File and Stream Conversion

### Bare values — just type them

```bash
tool.py 440                        # Hz → note
tool.py 261.63 329.63 440 880     # batch Hz → notes
tool.py 3/2 5/4 7/4              # ratios → cents + interval names
tool.py C4 Bb3 F#5               # note names → Hz
```

### Pipe — stdin to stdout

```bash
cat frequencies.txt | tool.py > converted.csv
aubiopitch recording.wav | tool.py > lattice_analysis.csv
generate_pitches | tool.py | process_further
```

### Text file — one value per line

```bash
tool.py measured_pitches.txt               # auto-detect, convert each line
tool.py measured_pitches.txt > output.csv  # redirect output
```

Input file can contain Hz values, ratios (p/q), or note names, one per line. Lines starting with `#` are skipped.

### MIDI file → CSV (discrete → continuous)

```bash
tool.py song.mid
tool.py song.mid > analysis.csv
```

Output: `time_ms,midi,note,hz,k,d,epsilon_cents,velocity`

Every MIDI note is converted to its exact Hz frequency via the bijection pullback.

### Frequencies → MIDI file (continuous → discrete)

```bash
tool.py to-midi input.txt output.mid
tool.py to-midi input.txt output.mid 140    # custom BPM
```

Input file format — one note per line: `hz,duration_ms,velocity`

```
440,500,100
523.25,500,90
659.25,500,80
880,1000,100
```

Each Hz is projected through the bijection to the nearest MIDI note. The ε residual is encoded as MIDI pitch bend, preserving the exact frequency.

### Scala file → frequency table

```bash
tool.py tuning.scl
tool.py tuning.scl > frequencies.csv
```

Reads any `.scl` file and outputs every degree with its Hz, cents, and lattice coordinates.

### Tuning system → Scala file

```bash
tool.py scala ji5                          # writes ji5.scl
tool.py scala rast /path/to/output.scl     # custom path
```

Generates `.scl` files loadable by any Scala-compatible synth (Kontakt, Pianoteq, Surge, ZynAddSubFX, Vital, etc.).

### WAV files

For WAV pitch detection, use an external pitch detector and pipe:

```bash
aubiopitch recording.wav | tool.py > analysis.csv
crepe recording.wav --output pitch.csv && tool.py pitch.csv > lattice.csv
```

---

## Input Auto-Detection

| You give it | It reads as | It outputs |
|---|---|---|
| Number (`440`) | Hz | Note, MIDI, ε, family |
| Fraction (`3/2`) | Ratio | Cents, interval name, k, d, ε |
| Note name (`C4`, `Bb3`) | Note | Exact Hz, MIDI, k |
| `.mid` file | MIDI | CSV with Hz + lattice coords per note |
| `.scl` file | Scala tuning | Frequency table with lattice coords |
| Text file | Lines of values | Conversion per line |
| Stdin (piped) | Stream | Stream of conversions |

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

The primary interface is bare values and file I/O (covered above). These commands provide deeper operations.

| Command | What it does |
|---|---|
| `to-midi <in.txt> <out.mid> [bpm]` | Convert Hz list to MIDI file (continuous→discrete) |
| `scala <name> [path]` | Export a tuning system as `.scl` file |
| `scale <name>` | Analyze a tuning system's lattice structure |
| `compare <sys1> <sys2> [...]` | Find shared lattice positions across tuning systems |
| `note <hz> [hz...]` | Detailed frequency analysis with family description |
| `ratio <p/q> [...]` | Multi-resolution tower analysis of a ratio |
| `midi <n> [n...]` | MIDI note → detailed lattice analysis |
| `harmonics <hz> <count>` | Harmonic series with lattice coords per partial |
| `families` | Reference table of all 12 harmonic families |
| `elegance <name>` | Elegance scoring of a tuning system |
| `selfproject` | Koide attractor / self-projection identity |
| `lattice2freq <k> [ε]` | Lattice position → exact Hz (discrete→continuous) |
| `freqs <hz1> <hz2> [...]` | Batch frequency table with lattice coords |
| `scales` | List all 59 built-in tuning systems |

Any `<n>tet` works dynamically with `scale` or `scala` (e.g. `scale 53tet`, `scala 72tet`).

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

### Recording → lattice analysis

```bash
# Extract pitches from audio (using external pitch detector)
aubiopitch recording.wav > pitches.txt
# Convert to lattice coordinates
tool.py pitches.txt > lattice_analysis.csv
```

### MIDI → exact frequency table

```bash
tool.py song.mid > frequencies.csv
# frequencies.csv has: time_ms, midi, note, hz, k, d, epsilon_cents, velocity
```

### Frequency list → MIDI file

```bash
# Create input file: hz,duration_ms,velocity per line
tool.py to-midi my_frequencies.txt output.mid 120
```

### Custom scale → synth-ready tuning

```bash
tool.py scala ji7 my_tuning.scl
# Load my_tuning.scl into Kontakt, Surge, Pianoteq, etc.
```

### Compare tuning systems for period performance

```bash
tool.py compare werkmeister kirnberger vallotti 12tet
```

### Cross-cultural connection finding

```bash
tool.py compare rast bhairav hirajoshi
# Shows which intervals share the same lattice position across traditions
```

### Pipe into existing toolchain

```bash
# Any program that outputs frequencies → lattice conversion
pitch_tracker live_input | tool.py > realtime_lattice.csv
# Any text file of mixed Hz/ratios/notes → converted
cat mixed_data.txt | tool.py > converted.csv
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
# Convert (auto-detect input type)
tool.py 440 261.63 3/2 C4 Bb3        bare values — Hz, ratios, note names
tool.py data.txt                      text file — one value per line
tool.py song.mid                      MIDI file → CSV
tool.py tuning.scl                    Scala file → frequency table
cat freqs.txt | tool.py               stdin pipe → stdout

# File conversion
tool.py to-midi input.txt out.mid     Hz list → MIDI file
tool.py scala ji5 [path]              tuning system → .scl file

# Advanced
tool.py scale <name>                  analyze tuning system
tool.py compare <a> <b> [...]         cross-cultural comparison
tool.py note <hz>                     detailed Hz analysis
tool.py ratio <p/q>                   multi-resolution ratio analysis
tool.py midi <n>                      MIDI note analysis
tool.py harmonics <hz> <n>            overtone series
tool.py lattice2freq <k> [ε]          lattice → Hz
tool.py families                      12-family reference
tool.py selfproject                   Koide attractor
tool.py elegance <name>               elegance scoring
tool.py scales                        list all 59 systems
```

---

**P ∘ D ∘ T = E**

*"For every exception there is an exception, except the Exception."*

Exception Theory — Michael James Muller (Aevum Defluo)

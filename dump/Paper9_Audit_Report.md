# ET Sempaevum Paper 9 — Audit & Fix Report

**Source:** `/mnt/user-data/uploads/ET_Sempaevum_Paper9.tex` (3463 lines)
**Output:** `/mnt/user-data/outputs/ET_Sempaevum_Paper9.tex` (3503 lines, +40 net)
**Compiled output:** `/mnt/user-data/outputs/ET_Sempaevum_Paper9.pdf` (88 pages)
**Date:** 2026-04-24

## Summary

Six mathematical/citation issues fixed; two pre-existing LaTeX warnings fixed; **all 39 page-overflow defects (the worst at 345pt = 4.8 inches off the page) eliminated**. No content removed; all features preserved.

| Metric | Original | Fixed |
|---|---|---|
| Math/citation issues | 6 known | 0 |
| Substantive LaTeX warnings | 2 | 0 |
| Overfull boxes (page-bleed) | **39** (worst 345pt) | **0** |
| Undefined references | 0 | 0 |
| Undefined citations | 0 | 0 |
| Compile passes | clean | clean |
| Pages | 88 | 88 |

**Verification:** 206/206 math checks pass at 60-digit mpmath precision.

---

## Part A — Math/Citation Fixes (Issues 1–6)

### Issue 1 — Pythagorean comma δ value (octaves vs cents confusion)

**Location:** lines 1331–1335 (original)

**Before:** Mixed units — `0.0019550` was treated as both an octave and a cent value, and the multiplication `12·2.346 cents` was wrong.

**After:**
```latex
\delta = \log_2(3/2) - 7/12 \approx 0.001629 octaves = 1.9550 cents.
N|\delta| \approx 12 \cdot 1.9550 cents \approx 23.46 cents < 50 cents.
```

**Verified:** δ = 0.001629167 octaves; δ = 1.9550008654 cents; 12·δ = 23.4600 cents.

---

### Issue 2 — Continued fraction of −53/12

**Before:** `-53/12 = [-5; -2, -2, -2]` — under standard simple-CF semantics this evaluates to `-65/12`, NOT `-53/12`.

**After:** Both factually correct equivalent representations now presented:
- `-53/12 = [-5; 1, 1, 2, 2]` (canonical floor form, all positive partial quotients)
- `-53/12 = -[4; 2, 2, 2]` (negation form, preserves symmetric tail visible in convergents 4, 9/2, 22/5, 53/12)

**Verified:** Both forms evaluate to −53/12 exactly via Fraction arithmetic; old form `[-5;-2,-2,-2]` evaluates to `-65/12`.

---

### Issue 3 — α⁻¹ precision claim and citation

**Before:** Vague "agrees to one part in 10¹⁰" with citation to internal `LatComp2025` manuscript.

**After:** Sigma-units against all four reference values, anchored to CODATA 2022:
- ET formula: `137.035 999 167 441 337 4...`
- CODATA 2022: `137.035 999 177(21)` — ET sits at **0.46σ**, ~7 parts in 10¹¹
- Parker 2018 (Berkeley Cs): `137.035 999 046(27)` → ET at **+4.5σ**
- Morel 2020 (LKB Rb): `137.035 999 206(11)` → ET at **−3.5σ**
- CODATA 2018: `137.035 999 084(21)` → ET at **+4.0σ**
- Parker–Morel mutual tension flagged (5σ)

Abstract and conclusion text also updated.

**Verified:** All sigma-unit calculations independently confirmed at 60-digit precision.

---

### Issue 4 — Bibliography updates

- **Removed:** `\bibitem{LatComp2025}` (internal manuscript)
- **Added:** `\bibitem{CODATA2018}` with full citation (Tiesinga, Mohr, Newell, Taylor; J. Phys. Chem. Ref. Data 50, 033105, 2021)
- **Corrected:** `\bibitem{CODATA2022}` author order — was Tiesinga first (the 2018 author order), now Mohr first (the actual 2022 author order, verified against RMP 97, 025002, 2025)
- **Retained unchanged:** Parker2018, Morel2020 bibitems

All `\cite{LatComp2025}` and prose mentions of "LatComp 2025" replaced in three locations.

---

### Issue 5 — Notation: `12·2π/ln 2` requires parentheses

Four unparenthesized occurrences fixed:
- Line 1244: `12 \cdot 2\pi/\ln 2` → `24\pi/\ln 2` (consistent with §13)
- Line 2106: `N\cdot 2\pi/\ln 2` → `N\cdot (2\pi/\ln 2)` (kept generic with N)
- Lines 2108, 2112: `12\cdot 2\pi/\ln 2` → `24\pi/\ln 2`

**Verified:** 12·(2π/ln 2) = 24π/ln 2 = 108.7766434039 (algebraically equivalent), nearest integer 109.

---

### Issue 6 — Tetrahedral angle structural weakness (cosine projection)

**Before:** Single sentence stating tetrahedral 109.47°/180° = 0.608 projects to d=4 — but this gives Π_12 = (−9, 4, **+39.07¢**), with |ε| in the Twilight Zone (33¢ < |ε| < 50¢).

**Solution:** Forward-derived from Identification Principle (projection guide §25.2). The substrate provides two independent rationals — `θ/180°` (transcendental for tetrahedral) and `cos θ` (exactly −1/3 for tetrahedral, the rational invariant). Both are valid Path-C projections; their dual reading provides the complete structural fingerprint.

**After:** New table + paragraph showing dual readings:

| Geometry | θ | θ/180° projection | \|cos θ\| | \|cos θ\| projection |
|---|---|---|---|---|
| Linear (sp) | 180° | (0, 1, 0) unison | 1 | (0, 1, 0) unison |
| Trigonal-planar (sp²) | 120° | (−7, 12, −1.96¢) Koide | 1/2 | (−12, 1, 0) octave |
| Tetrahedral (sp³) | arccos(−1/3) | (−9, 4, +39.07¢) d=4 | **1/3** | **(−19, 12, −1.96¢) Koide** |
| Right-angle | 90° | (−12, 1, 0) octave | 0 | ∂I annihilation |

The tetrahedral cosine reading lands **exactly on the Koide attractor** — same lattice address as `{N, 1/N, K, 1/K}` from the self-projection identity (Theorem `thm:self`). The Twilight Zone weakness dissolves.

**Verified:** All 4 dual-reading projections confirmed numerically; cross-checked Koide attractor signature for {12, 1/12, 2/3, 3/2}.

---

## Part B — Pre-existing LaTeX warning fixes (Rule 30)

### Pre-fix A: Literal `¢` in abstract → `\textcent` (cmr10 missing-glyph)
### Pre-fix B: `\textcent` in math mode in gaze-threshold table → restructured to match unbold rows

---

## Part C — Page-overflow fixes (39 → 0)

The original PDF had **39 overfull boxes** — content extending past the page margin. The worst was **345pt** (4.8 inches off the page) at line 2572 (the Path D sub-paths table). Six tables had multi-hundred-point overflows.

### C.1 — Package additions (preamble)

```latex
\usepackage{tabularx}                                    % auto-wrapping table columns
\usepackage[protrusion=true,expansion=false]{microtype}  % paragraph-quality protrusion
\setlength{\emergencystretch}{3em}                       % flexibility for hard cases
```

`microtype` expansion disabled because Computer Modern font isn't scalable; protrusion alone fixes ~23 paragraph-level overflows. No font changes.

### C.2 — Tables converted to `tabularx` (auto-distributing X columns, no manual width tuning per Rule 12)

| Location | Original overflow | Issue | Fix |
|---|---|---|---|
| Line 2572 | **345pt** | Path D sub-paths table — long descriptions in last column | `tabular{l l l}` → `tabularx{\textwidth}{lXX}` |
| Line 2528 | 208pt | Triple-backbone table | `tabular{l l l l}` → `tabularx{\textwidth}{lXlX}` |
| Line 692 | 104pt | **Table 4.1** (temporal aspects) — long "Nature" descriptions | `tabular{llll}` → `tabularx{\textwidth}{llXX}` |
| Line 1842 | 77pt | LCM tower table — long "Empirical phenomena" column | `tabular{c c c l l}` → `tabularx{\textwidth}{c c c l X}` |
| Line 2491 | 66pt | Webb stroke PDT decomposition | `tabular{l l}` → `tabularx{\textwidth}{XX}` |
| Line 638 | 53pt | Methodological loop (rl) | `tabular{rl}` → `tabularx{\textwidth}{rX}` |
| Line 2857 | 34pt | Gaze threshold table (last column "—Koide attractor") | `tabular{l l l l l}` → `tabularx{\textwidth}{l l l l X}` |
| Line 3160 | 15pt | Bond angle table (Issue 6 addition) | `tabular{l c c c c}` → `tabularx{\textwidth}{lcXcX}` |

### C.3 — Long math displays converted to `multline*` / `aligned`

| Location | Original | Issue | Fix |
|---|---|---|---|
| Line 2034 | 182pt | 42-element integer set `{1, 2, ..., 132}` | `\[ ... \]` → `multline*` (2 lines, all 42 values preserved) |
| Line 2731 | 103pt | ∀X universality quantifier | `\[ ... \]` → `multline*` (3 lines) |
| Line 2958 | 62pt | Decoherence trajectory array (math `array` with long text) | `array{c|c|c|c|l}` → `tabularx{\textwidth}{c|c|c|c|X}` |
| Line 2608 | 23pt → 12pt → 0 | Chaitin Ω lattice address with `\qquad` | reduced to `\quad`, then `aligned` |
| Line 1319 | 15pt → 4pt → 0 | Cardinal-counts display | split via `gathered` |

### C.4 — Long inline strings broken via `\allowbreak` and `sloppypar`

- `\textit{The\_Palindromic\_Cascade\_V2}` (2 occurrences, lines 903 & 1161) → `\textit{The\_\allowbreak Palindromic\_\allowbreak Cascade\_\allowbreak V2}` (allows break at any underscore)
- `\textit{ET\_Four\_Constants\_Complete\_Derivation...}` (line 873) → same pattern
- T2 Inverse convergents paragraph (line 905) — wrapped in `\begin{sloppypar}...\end{sloppypar}` with `\allowbreak{}` after each long inline math; this paragraph contains TWO long unbreakable inline lists (the CF expansion AND the convergent list)
- Lepton mass line (line 3131, from §17.4 Issue 3 edit) — wrapped in `sloppypar` with `\allowbreak{}` between `(m_e,m_μ,m_τ)`, `=`, the values, and the `\cite`

---

## Part D — Final compile metrics

```
$ pdflatex (pass 1): exit 0
$ pdflatex (pass 2): exit 0

PDF output:                  88 pages, 786471 bytes
Overfull boxes:              0
Substantive warnings:        0
Undefined references:        0
Undefined citations:         0
Multiply-defined labels:     0
Brace balance:               4819/4819 OK
\begin/\end balance:         424/424 OK
Inline math $ count:         7004 (even) OK
\left/\right balance:        11/11 OK
```

---

## Part E — Verification suite

`verify_paper9.py` (1232 lines) covers all original content plus all fixes:

```
Total checks:   206
PASS:           206
FAIL:           0
Success rate:   100.00%
```

New sections covering Issues 1–6 fixes:
- **Section 23:** Issue 1 — Pythagorean comma exact values (octaves and cents)
- **Section 24:** Issue 2 — both CF forms equal −53/12 (sympy + Fraction)
- **Section 25:** Issue 3 — α⁻¹ vs all 4 reference values
- **Section 26:** Issue 6 — bond angle dual projections (angle/180° AND |cos θ|)
- **Section 27:** Issue 5 — 12·(2π/ln 2) ≡ 24π/ln 2

All earlier verification sections (1–22) still pass unchanged.

---

## Files delivered

| File | Description |
|---|---|
| `ET_Sempaevum_Paper9.tex` | Fixed LaTeX source (3503 lines) |
| `ET_Sempaevum_Paper9.pdf` | Compiled output (88 pages, clean compile, zero page overflows) |
| `Paper9_Audit_Report.md` | This report |

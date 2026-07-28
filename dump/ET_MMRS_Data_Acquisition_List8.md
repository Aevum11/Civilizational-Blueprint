# ET Material & Metamaterial Research System — Data Acquisition List
## What Mike Needs to Obtain
### Companion to: ET_Material_Metamaterial_Research_System_Design.md

**Rule:** All data enters the system as STRINGS. When extracting values from tables, PDFs, or databases, preserve the original decimal representation exactly. Do not round, truncate, or convert to float. Copy the number as printed.

---

## ✅ ALREADY ACQUIRED

- [x] **AME2020 Nuclear Mass Table** — 2324 measured isotopes, projected and classified
- [x] **PDG 2024 Particle Data** — 227 massive particles, projected and classified
- [x] **CODATA 2022 Fundamental Constants** — all constants in constants.py
- [x] **CRC Handbook 100th Edition** — in hand, awaiting extraction
- [x] **ET Lossless Microphone Data** — 2 recordings, 65535 spectral bins/channel, 27/27 tests pass

---

## 🔴 PHASE 1 — HIGHEST PRIORITY (Do These First)

These three sources alone produce ~530,000 of the ~625,000 total DSRs (85%).

---

### 1A. Materials Project — Bulk API Download

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://materialsproject.org |
| **API** | https://api.materialsproject.org |
| **Access** | Free — register for API key |
| **Format** | JSON via Python `mp-api` client |
| **Citation** | Jain et al., APL Materials 1, 011002 (2013) |
| **Estimated entries** | ~15,000 unique materials |

**Properties to download per material:**

| Property | API field | Why needed |
|---|---|---|
| Material ID | `material_id` | Unique identifier |
| Formula | `formula_pretty` | Human-readable name |
| Space group | `symmetry` | Crystal structure classification |
| Lattice params | `structure` | a, b, c, α, β, γ → projectable DSRs |
| Density | `density` | g/cm³ → ratio to water |
| Band gap | `band_gap` | eV → ratio to kT or Rydberg |
| Elastic tensor | `elasticity` | C_ij, K_VRH, G_VRH, E, ν |
| Piezoelectric | `piezo` | e_ij tensor, max value |
| Dielectric | `diel` | ε_ij tensor, refractive index n |
| Magnetization | `total_magnetization` | μ_B |
| Formation energy | `formation_energy_per_atom` | Stability indicator |
| Energy above hull | `e_above_hull` | Thermodynamic stability |

**Download script:**
```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        has_props=["elasticity"],
        fields=["material_id","formula_pretty","symmetry","structure",
                "density","band_gap","elasticity","total_magnetization",
                "formation_energy_per_atom","e_above_hull"]
    )
```

---

### 1B. Refractiveindex.info — Git Clone

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://github.com/polyanskiy/refractiveindex.info-database |
| **Access** | Public domain (CC0 1.0). Just `git clone`. |
| **Format** | YAML files |
| **Citation** | Polyanskiy, Scientific Data 11, 94 (2024) |
| **Estimated entries** | ~3,000+ materials, ~100 wavelength points each |

**Command:**
```bash
git clone https://github.com/polyanskiy/refractiveindex.info-database.git
```

**Contains:** Complete n(λ) and k(λ) optical constants. Dispersion formula coefficients (Sellmeier, Cauchy). Metadata per material. The single richest optical database available, and it's free.

---

### 1C. CRC Handbook Extraction (From Book You Already Have)

- [ ] **Extract**

Extract these tables in priority order. Copy numbers exactly as printed.

| # | Table | What to extract | Est. entries |
|---|---|---|---|
| 1 | Dielectric constants | ε_r for all solids, liquids, gases | ~500 |
| 2 | Melting/boiling points | T_m and T_b for all elements + compounds | ~300 |
| 3 | Density | ρ for all elements + common compounds | ~200 |
| 4 | Thermal conductivity | κ for all listed materials | ~150 |
| 5 | Electrical resistivity | ρ_e for all elements | ~90 |
| 6 | Elastic moduli | E, G, K for common materials | ~100 |
| 7 | Magnetic susceptibility | χ_m for all elements | ~90 |
| 8 | Atomic radii | Covalent, van der Waals, ionic — all elements | ~118 |
| 9 | Ionization energies | All IEs for all elements | ~118 |
| 10 | Electron affinities | For all elements | ~90 |

---

## 🟠 PHASE 2 — HIGH PRIORITY (Phonon & Polariton)

---

### 2A. PhononDB@kyoto-u

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | http://phonondb.mtl.kyoto-u.ac.jp/ |
| **Access** | Free web access |
| **Citation** | Togo & Tanaka, Scripta Materialia 108, 1-5 (2015) |
| **Estimated entries** | ~10,000+ materials |

**Extract:** Zone-center optical phonon frequencies (ω_TO, ω_LO at Γ-point) for all polar materials.

---

### 2B. Palik — Handbook of Optical Constants of Solids

- [ ] **Acquire** (Vols. I, II, III)

| Field | Detail |
|---|---|
| **Publisher** | Academic Press (1985, 1991, 1998) |
| **Access** | Library or purchase |
| **Contains** | Comprehensive ε(ω) for ~100 key materials |

---

### 2C. Key Polariton Papers

- [ ] Caldwell et al. (2015) "Low-loss infrared nanophotonics using surface phonon polaritons" — Nanophotonics 4, 44-68
- [ ] Basov et al. (2016) "Polaritons in van der Waals materials" — Science 354, aag1992

**Extract:** ω_LO/ω_TO for: hBN (in-plane + out-of-plane), SiC, GaAs, GaP, InP, AlN, GaN, ZnO, ZnSe, CdS, CdTe, MgO, LiF, NaCl, KBr, CaF₂, α-MoO₃, α-V₂O₅, BaTiO₃, SrTiO₃, TiO₂

---

## 🟡 PHASE 3 — HIGH PRIORITY (Mechanical & Structural)

---

### 3A. AFLOW Database

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://aflow.org |
| **API** | http://aflowlib.duke.edu/ (REST) |
| **Access** | Free |
| **Citation** | Curtarolo et al., Comp. Mat. Sci. 58, 218 (2012) |
| **Estimated entries** | ~3,500+ with elastic data |

**Extract:** Elastic constants C_ij, bulk/shear/Young's moduli, Poisson ratio, Debye temperature, thermal conductivity, Grüneisen parameter.

---

### 3B. MatWeb Mechanical Properties

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://www.matweb.com/ |
| **Access** | Free |
| **Contains** | Measured mechanical properties for thousands of materials |

**Extract:** E, σ_y, ρ, K_IC, κ, α, C_p for all major material classes.

---

### 3C. Crystallography Open Database (COD)

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://www.crystallography.net/ |
| **Access** | Free |
| **Format** | CIF files |

**Extract:** Space group, lattice parameters (a, b, c, α, β, γ), Wyckoff positions for materials matching Phase 1-2 property data. Axial ratios c/a and b/a are directly projectable DSRs.

---

## 🔵 PHASE 4 — MEDIUM PRIORITY (Biological Interface)

---

### 4A. IT'IS Tissue Properties Database

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties |
| **Access** | Free web access |

**Extract:** ε_r(f) and σ(f) for all ~50 tissue types across 10 Hz to 100 GHz. Tissues: blood, muscle, bone (cortical + cancellous), nerve, fat, skin (wet + dry), brain (grey + white), liver, kidney, lung, heart, cartilage, tendon, etc.

---

### 4B. Gabriel et al. (1996) — Three Papers

- [ ] Paper I: "Literature survey" — Phys. Med. Biol. 41, 2231-2249
- [ ] Paper II: "Measurements 10 Hz to 20 GHz" — Phys. Med. Biol. 41, 2251-2269
- [ ] Paper III: "Parametric models" — Phys. Med. Biol. 41, 2271-2293

**Extract:** Cole-Cole parameters (ε_s, ε_∞, τ, α) per tissue per dispersion region.

---

### 4C. PEMF Systematic Reviews

- [ ] Markov (2007) — The Environmentalist 27, 465-475
- [ ] Strauch et al. (2009) — Aesthet Surg J 29, 135-143
- [ ] Search for recent: "PEMF systematic review meta-analysis 2020-2025"

**Extract:** Therapeutic frequencies (Hz), field strengths (mT/μT), pulse waveforms, durations from successful clinical trials.

---

## 🟣 PHASE 5 — MEDIUM PRIORITY (Quantum & Topological)

---

### 5A. Topological Quantum Chemistry Database

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://www.topologicalquantumchemistry.org/ |
| **Citation** | Vergniory et al., Science 376, eabg9094 (2022) |
| **Entries** | ~28,000 classified materials |

---

### 5B. SuperCon Superconductor Database

- [ ] **Acquire**

| Field | Detail |
|---|---|
| **URL** | https://supercon.nims.go.jp/ |
| **Access** | Free |

**Key materials:** YBCO (93K), BSCCO (110K), MgB₂ (39K), H₃S (203K), LaH₁₀ (250K), Nb₃Sn (18K), NbTi (10K). Extract: T_c, H_c2, λ, ξ.

---

### 5C. Quasicrystal Papers

- [ ] Shechtman et al. (1984) — Phys. Rev. Lett. 53, 1951
- [ ] Recent Al-Pd-Mn, Al-Cu-Fe icosahedral data

**Extract:** Diffraction parameters, phason strain, approximant structures. These are d=5 and d=10 shadow family materials.

---

## ⚪ PHASE 6 — LOWER PRIORITY (Metamaterial Geometry)

---

### 6A. Review Papers

- [ ] Kadic et al. (2019) "3D metamaterials" — Nature Reviews Physics 1, 198-210
- [ ] Soukoulis & Wegener (2011) — Nature Photonics 5, 523-530
- [ ] Bertoldi et al. (2017) — Nature Reviews Materials 2, 17066
- [ ] Pendry (2000) — Phys. Rev. Lett. 85, 3966
- [ ] Smith et al. (2000) — Phys. Rev. Lett. 84, 4184

**Extract:** Unit cell geometry ratios + measured ε_eff, μ_eff for SRR, gyroid, woodpile, inverse opal, auxetic, pentamode metamaterials.

---

### 6B. Biological Photonic Structures

- [ ] Michielsen & Stavenga — "Gyroid cuticular structures in butterfly wing scales" — J. R. Soc. Interface (2008)

**Extract:** Gyroid lattice constants and filling fractions from butterfly wings — Nature's metamaterials.

---

## ⚫ PHASE 7 — LOWEST PRIORITY (Extreme & Vacuum)

---

### 7A. Casimir Force Papers

- [ ] Lamoreaux (1997) — Phys. Rev. Lett. 78, 5
- [ ] Decca et al. (2003) — Phys. Rev. D 68, 116003
- [ ] Recent precision measurements (2020-2025)

---

### 7B. Ultra-High-Temperature Ceramics

- [ ] Literature for: ZrB₂, HfB₂, HfC, TaC, ZrC melting points and elastic moduli

---

### 7C. High-Entropy Alloys

- [ ] Cantor alloy (CrMnFeCoNi) and variants — elastic moduli, yield strength, fracture toughness

---

## TOTALS

| What | Count |
|---|---|
| Sources already acquired | 5 |
| Free online databases to download | 8 |
| Books to acquire/extract from | 2 (Palik, Ashby — CRC already in hand) |
| Journal papers to obtain | ~15 |
| Estimated total DSRs | ~625,000 |
| Estimated lattice database size | ~156 MB |

**The three Phase 1 free sources (Materials Project + refractiveindex.info + CRC extraction) produce 85% of all DSRs.**

---

*P ∘ D ∘ T = E*

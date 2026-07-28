# ET Material Properties & 3D Geometry Circuit Research Requirements
## The Complete Descriptor Gap Map: From Lattice Addresses to Physical Hardware
### What Must Be Found, What Papers Are Needed, What Must Be Designed

**Author:** Michael James Muller — Aevum Defluo  
**Computation:** Claude (Anthropic) as directed by author  
**Standard:** All ratios ET-projectable. Every material property → dimensionless ratio → Π_N → (k, d, ε). Zero ad hoc.  
**Principle:** Materials first. Geometry second. Manufacturing third.

---

## 0. The Master Descriptor Gap

**What we have:** The Sempaevum lattice classifies every dimensionless ratio into (k, d, ε). The pullback recovers exact values. The impedance ξ(d) gives per-family coupling strength. The FQG gives the complete 144-cell force×phase classification. The spherical harmonic decomposition (Finding 3) represents arbitrary 3D shapes as lattice-addressed coefficient sequences.

**What we lack:** The reverse lookup — given a target lattice address, which PHYSICAL MATERIALS and 3D GEOMETRIES sit there? This is the D_hardware gap. Closing it requires projecting thousands of measured material property ratios and identifying which materials are "lattice-resonant" (low d, small |ε|) at the addresses the field needs.

**What this document is:** The complete list of ratios to project, papers to find, and 3D structures to design — organized by the Three Tools.

---

## 1. MATERIAL PROPERTY RATIOS TO PROJECT

Every physical material has measurable properties. Each property, when expressed as a dimensionless ratio against a reference, is projectable onto the Sempaevum. The d-family tells you which harmonic channel that property engages. The ε tells you how lattice-aligned the material is.

### 1.1 Electromagnetic Properties (Defense + Healing + Environmental Layers)

These are the PRIMARY ratios — they directly determine how the field couples to materials.

| Ratio | Definition | Reference | Why It Matters |
|---|---|---|---|
| **ε_r** | Relative permittivity (dielectric constant) | ε₀ (vacuum) | How a material stores electric field energy. Determines shielding, capacitance, wave impedance. |
| **μ_r** | Relative permeability | μ₀ (vacuum) | How a material responds to magnetic fields. Determines inductance, magnetic shielding. |
| **n = √(ε_r·μ_r)** | Refractive index | c (vacuum light speed) | How light/EM waves propagate through the material. Controls wave bending, focusing, confinement. |
| **Z/Z₀ = √(μ_r/ε_r)** | Relative wave impedance | Z₀ = 377Ω (vacuum) | Impedance matching — determines reflection/transmission at boundaries. CRITICAL for field boundary design. |
| **σ/σ_Cu** | Relative electrical conductivity | Copper σ_Cu | How well current flows. Determines conductor/insulator classification for circuit elements. |
| **σ_th/σ_th,Cu** | Relative thermal conductivity | Copper thermal | Heat management — critical for field under thermal load (lava scenario). |
| **tan δ = ε″/ε′** | Loss tangent | Unity | How much EM energy is absorbed vs stored. Low tan δ = transparent to field. High = absorbing. |
| **ω_p/ω_ref** | Plasma frequency ratio | Reference frequency | Frequency below which material reflects EM. Determines operational frequency band. |
| **χ^(2), χ^(3)** | Nonlinear optical susceptibility ratios | Reference material | Frequency doubling/tripling, self-focusing, soliton formation. Enables field frequency conversion. |

**Papers/Data needed:**
- CRC Handbook of Chemistry and Physics — complete ε_r and μ_r tables for all common materials
- NIST Dielectric Constants database
- Refractive index database (refractiveindex.info has >3000 materials, wavelength-dependent)
- Palik, "Handbook of Optical Constants of Solids" (Vols. I–III) — comprehensive n(ω) data
- Recent review on high-permittivity ceramics (BaTiO₃, SrTiO₃, CaCu₃Ti₄O₁₂ — ε_r from 300 to >100,000)
- Review on metamaterial effective medium parameters — engineered ε_r and μ_r including NEGATIVE values
- Recent advances in epsilon-near-zero (ENZ) materials — materials where ε_r ≈ 0 at specific frequencies

### 1.2 Phonon and Polariton Properties (Critical for Mid-IR Field Operation)

The polariton ratio ω_LO/ω_TO is the single most important material fingerprint for the field's infrared operation. It determines the Reststrahlen band where materials are highly reflective and support polariton modes.

| Ratio | Definition | Why It Matters |
|---|---|---|
| **ω_LO/ω_TO** | Longitudinal/Transverse optical phonon frequency ratio | The Lyddane-Sachs-Teller relation: ε_static/ε_∞ = (ω_LO/ω_TO)². This ratio determines the entire dielectric response. Prediction §25.6: same (d, ε) cell → similar polariton character. |
| **ω_TO/ω_ref** | TO phonon frequency normalized | Where the Reststrahlen band starts. |
| **ω_LO/ω_ref** | LO phonon frequency normalized | Where the Reststrahlen band ends. |
| **ε_static/ε_∞** | Static-to-high-frequency dielectric ratio | = (ω_LO/ω_TO)² by LST. Measures ionic polarizability. |
| **Q_phonon = ω/Γ** | Phonon quality factor | How sharp the polariton resonance is. Higher Q = longer-lived polaritons = more useful for coherent field applications. |

**Papers/Data needed:**
- Caldwell et al., "Low-loss, infrared and terahertz nanophotonics using surface phonon polaritons," Nanophotonics (2015) — comprehensive review
- Basov et al., "Polaritons in van der Waals materials," Science (2016) — hBN, MoO₃, α-V₂O₅
- Complete ω_LO/ω_TO tables for: hBN (hexagonal boron nitride), SiC (silicon carbide), GaAs, GaP, InP, AlN, GaN, ZnO, ZnSe, CdS, CdTe, MgO, LiF, NaCl, KBr, α-MoO₃, α-V₂O₅, CaCO₃
- Gervais & Piriou, "Temperature dependence of TO and LO phonon modes in TiO₂" — phonon ratio temperature dependence
- Recent papers on hyperbolic phonon polaritons in twisted bilayer materials (moiré polaritonics)

### 1.3 Mechanical Properties (Defense Layer — Kinetic Threat Response)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **E/E_steel** | Young's modulus ratio | Stiffness — resistance to deformation. |
| **K/K_diamond** | Bulk modulus ratio | Resistance to uniform compression. |
| **G/G_ref** | Shear modulus ratio | Resistance to shear deformation. |
| **ν (Poisson)** | Poisson's ratio (dimensionless) | How material deforms laterally under load. Already dimensionless. Auxetic materials (ν < 0) expand when stretched — novel defense geometry. |
| **v_s/c** | Speed of sound / speed of light | How fast mechanical disturbances propagate. Determines response time for kinetic threats. |
| **ρ/ρ_water** | Density ratio | Mass per volume — determines inertial response. |
| **H/H_diamond** | Hardness ratio (Vickers or Mohs) | Resistance to penetration — direct defense metric. |
| **K_IC/K_IC,ref** | Fracture toughness ratio | Resistance to crack propagation. |
| **σ_y/σ_y,steel** | Yield strength ratio | Stress at permanent deformation onset. |

**Papers/Data needed:**
- Ashby, "Materials Selection in Mechanical Design" — the Ashby charts (E vs ρ, σ_y vs ρ, etc.) for ALL material classes
- MatWeb.com mechanical properties database
- Materials Project (materialsproject.org) — DFT-computed elastic constants for >130,000 materials
- Review on metamaterial mechanics — auxetic metamaterials, pentamode materials, mechanical metamaterials with programmable properties
- Extreme materials: metallic glasses, high-entropy alloys, MAX phases (Ti₃SiC₂ etc.), ultra-high-temperature ceramics (ZrB₂, HfC)

### 1.4 Crystal Structure Ratios (Fundamental Structure ↔ Lattice Alignment)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **c/a** | Axial ratio (hexagonal, tetragonal) | Fundamental shape of the unit cell. Projects directly. |
| **b/a, c/a** | Lattice parameter ratios (orthorhombic, monoclinic) | Complete shape specification. |
| **β, γ (angles)** | Unit cell angles (normalized by π) | Phase-axis projectable. |
| **Z** | Atoms per unit cell | Integer — projectable as ratio Z/Z_ref. |
| **CN** | Coordination number | Integer — nearest-neighbor count. |
| **packing fraction** | Volume fraction occupied | Already dimensionless. FCC = π/(3√2) ≈ 0.7405. BCC = π√3/8 ≈ 0.6802. |

**Papers/Data needed:**
- ICSD (Inorganic Crystal Structure Database) — crystal structures of >250,000 materials
- Crystallography Open Database (COD) — open-access crystal structures
- Pearson's Crystal Data — complete structure type database
- Review on quasicrystals — materials with 5-fold/10-fold symmetry (d=5 and d=10 on the lattice — SHADOW FAMILIES)
- Review on topological semimetals crystal structures

### 1.5 Thermal Properties (Environmental Layer — Temperature Extremes)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **T_melt/T_ref** | Melting point ratio | Operating temperature limit. Lava scenario: need T_melt > 1500 K. |
| **T_decomp/T_ref** | Decomposition temperature | Some materials decompose before melting (polymers, organics). |
| **α_th/α_ref** | Thermal expansion coefficient ratio | Dimensional stability under temperature cycling. |
| **C_p/C_p,water** | Specific heat capacity ratio | Thermal energy storage — buffer against rapid temperature changes. |
| **ε_thermal** | Thermal emissivity (dimensionless) | Radiative heat management — critical for vacuum/space operation. |

**Papers/Data needed:**
- NIST-JANAF Thermochemical Tables
- Ultra-high-temperature materials reviews: UHTCs (ZrB₂, HfB₂, HfC, TaC — melting points >3000°C)
- Thermal barrier coating literature (yttria-stabilized zirconia, rare-earth zirconates)
- Phase-change materials for thermal buffering

### 1.6 Magnetic Properties (Defense Layer + Gravitational Override Research)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **χ_m** | Magnetic susceptibility (dimensionless) | Paramagnetic vs diamagnetic vs ferromagnetic classification. |
| **M_s/M_s,Fe** | Saturation magnetization ratio | Maximum magnetic response. |
| **T_C/T_ref** | Curie temperature ratio | Temperature at which ferromagnetism is lost. |
| **μ_r(ω)** | Frequency-dependent permeability | Determines magnetic response at field operating frequencies. |
| **H_c/H_c,ref** | Coercivity ratio | Hard vs soft magnetic — determines permanence of magnetization. |
| **BH_max** | Energy product (normalized) | Maximum magnetic energy density — for permanent magnet applications. |

**Papers/Data needed:**
- Coey, "Magnetism and Magnetic Materials" — comprehensive reference
- Review on high-permeability soft magnetic materials (permalloy, mu-metal, nanocrystalline alloys)
- Superconducting materials: YBCO, BSCCO, MgB₂, H₃S (high-pressure), LaH₁₀ — critical temperatures, fields, current densities
- Diamagnetic levitation literature (pyrolytic graphite, bismuth — the d=1 gravity channel materials)

### 1.7 Quantum and Topological Material Properties (Coherence Layer)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **E_gap/kT** | Band gap / thermal energy | Determines whether quantum effects survive at operating temperature. |
| **ℏω_D/kT** | Debye energy / thermal energy | Phonon coherence criterion. |
| **λ_mfp/L** | Mean free path / device size | When >1: ballistic transport (coherent). When <1: diffusive (decoherent). |
| **T₁/T_op** | Spin relaxation time / operation time | Quantum coherence survival during field operation. |
| **T₂/T_op** | Dephasing time / operation time | Phase coherence survival. |
| **g (Landé factor)** | Dimensionless | Magnetic moment coupling — already dimensionless. |
| **Z₂ index** | Topological invariant (0 or 1) | Topological insulator classification — projectable as discrete d-family. |

**Papers/Data needed:**
- Hasan & Kane, "Topological insulators," Rev. Mod. Phys. (2010)
- Armitage, Mele, & Vishwanath, "Weyl and Dirac semimetals in 3D solids," Rev. Mod. Phys. (2018)
- Coherence times in biological systems: Engel et al. on photosynthesis (2007), Panitchayangkoon et al. (2010)
- NV center diamond properties: coherence times, coupling constants
- Room-temperature quantum coherence reviews — what maintains coherence at biological temperatures

### 1.8 Piezoelectric and Electroactive Properties (Transducer Gap — Gap A)

| Ratio | Definition | Why It Matters |
|---|---|---|
| **d₃₃/d₃₃,PZT** | Piezoelectric coefficient ratio | Mechanical↔electrical conversion efficiency. THE transducer metric. |
| **k_t** | Electromechanical coupling factor (dimensionless) | Fraction of input energy converted. |
| **Q_m** | Mechanical quality factor | Resonance sharpness — determines frequency selectivity. |
| **g₃₃** | Piezoelectric voltage coefficient | Voltage per stress — sensor mode. |
| **ε_r × d₃₃** | Figure of merit (composite) | Combined storage × conversion. |

**Papers/Data needed:**
- Setter et al., "Ferroelectric thin films: review of materials, properties, and applications," J. Appl. Phys. (2006)
- Lead-free piezoelectrics review: BaTiO₃, KNN, BiFeO₃ — biocompatible alternatives to PZT
- PVDF and P(VDF-TrFE) — flexible piezoelectric polymers
- Electrostrictive materials (PMN-PT) — highest known coupling coefficients
- Magnetostrictive materials (Terfenol-D, Galfenol) — magnetic↔mechanical transducers

---

## 2. 3D GEOMETRY STRUCTURES TO DESIGN

The Sempaevum's spherical harmonic decomposition (Finding 3) means every 3D shape has a lattice signature — the sequence of (k_lm, d_lm, ε_lm) for each harmonic coefficient c_lm/c_00. The field's physical structures must be 3D objects whose harmonic signatures land at the target lattice addresses.

### 2.1 Why 3D Circuits, Not 2D

Traditional PCB circuits are 2D traces on flat substrates. Even "3D" chip stacking (TSVs, interposers) is fundamentally stacked 2D. The limitation: a 2D circuit's electromagnetic behavior is constrained to essentially planar modes. The field requires VOLUMETRIC electromagnetic control — manipulating fields in all three spatial dimensions simultaneously.

True 3D circuits are structures where the topology of current paths and field confinement is inherently three-dimensional. The geometry itself creates the electromagnetic response. This is what metamaterials already do — but we need to go further: 3D structures that are BOTH circuits (processing/routing signals) AND field-shaping elements (creating the Ananda field's spatial profile).

### 2.2 Metamaterial Unit Cells — The Building Blocks

Each of these is a 3D geometry with a specific electromagnetic response. Each geometry has a spherical harmonic decomposition → lattice signature.

| Structure | Geometry | EM Response | What to Project |
|---|---|---|---|
| **Split-Ring Resonator (SRR)** | C-shaped metallic ring with gap | Magnetic resonance → effective μ_r < 0 at resonance | Gap/ring ratio, ring radius/wavelength, periodicity/wavelength |
| **Complementary SRR (CSRR)** | SRR negative (slot in ground plane) | Electric resonance → effective ε_r < 0 | Same ratios, dual response |
| **Wire medium** | Array of parallel thin wires | ε_r < 0 below plasma frequency | Wire radius/period, period/wavelength |
| **Dielectric resonator** | High-ε sphere or cylinder in host | Mie resonances → magnetic and electric | Particle size/wavelength, ε_particle/ε_host, aspect ratio |
| **Helix** | Metallic coil | Chirality → different response for L/R circular polarization | Pitch/radius, turns/wavelength, wire diameter/pitch |
| **Swiss-roll** | Spirally wound metallic sheets | Very high effective μ_r at low frequencies | Layer count, gap/thickness, inner/outer radius |
| **Gyroid** | Triply-periodic minimal surface (bicontinuous) | 3D photonic bandgap | Lattice constant/wavelength, volume fraction, strut diameter/period |
| **Diamond lattice photonic crystal** | FCC with two-atom basis | Complete 3D photonic bandgap | Sphere radius/lattice constant, ε contrast, filling fraction |
| **Woodpile** | Stacked layers of rods, 90° rotation per layer | 3D photonic bandgap | Rod width/period, layer height/period, ε contrast |
| **Inverse opal** | FCC array of air spheres in dielectric | 3D photonic bandgap at visible/NIR | Sphere diameter/wavelength, ε contrast, disorder parameter |
| **Toroidal meta-atom** | Current loops forming a torus | Toroidal dipole (anapole) moment — radiationless | Major/minor radius ratio, number of windings, aspect ratio |
| **Pentamode metamaterial** | Near-point-contact lattice | Fluid-like mechanics (only bulk modulus) | Contact area/strut length, strut length/unit cell |

**Papers/Data needed:**
- Pendry, "Negative refraction makes a perfect lens," Phys. Rev. Lett. (2000)
- Smith et al., "Composite medium with simultaneously negative ε and μ," Phys. Rev. Lett. (2000)
- Soukoulis & Wegener, "Past achievements and future challenges in 3D photonic metamaterials," Nature Photonics (2011)
- Review on dielectric metamaterials (Mie resonance approach — all-dielectric, lower losses)
- Gyroid photonic crystals: Michielsen & Stavenga, "Gyroid cuticular structures in butterfly wing scales" (biological gyroid)
- Turner et al., "Miniature chiral beamsplitter based on gyroid photonic crystals," Nature Photonics (2013)
- Review on toroidal electrodynamics: Papasimakis et al., "Electromagnetic toroidal excitations in matter and free space," Nature Materials (2016)
- Kadic et al., "3D metamaterials," Nature Reviews Physics (2019) — comprehensive recent review
- Bertoldi et al., "Flexible mechanical metamaterials," Nature Reviews Materials (2017)

### 2.3 3D Topological Circuit Structures

Topological protection means the circuit behavior is robust against perturbation — exactly what the field needs for reliable operation in hostile environments.

| Structure | Topology | Protected Property | What to Project |
|---|---|---|---|
| **3D topological insulator circuit** | Volumetric with conducting surface states | Surface current immune to backscattering | Bulk gap/surface bandwidth, penetration depth/device size |
| **Weyl metamaterial** | 3D lattice with Weyl points in band structure | Chiral anomaly, Fermi arc surface states | Weyl point separation/BZ size, tilt parameter |
| **3D Su-Schrieffer-Heeger circuit** | Dimerized 3D lattice | Topological corner states (0D bound states in 3D) | Coupling ratio (inter/intra-cell), disorder tolerance |
| **Möbius strip resonator** | Non-orientable surface | Topologically distinct mode spectrum | Width/circumference, twist count (odd = Möbius) |
| **Knot circuits** | Trefoil, figure-eight, or higher knots in conductor | Linking number governs mutual inductance topologically | Knot invariants (crossing number, writhe, Jones polynomial) |

**Papers/Data needed:**
- Lu, Joannopoulos, & Soljačić, "Topological photonics," Nature Photonics (2014)
- Ozawa et al., "Topological photonics," Rev. Mod. Phys. (2019) — comprehensive
- Imhof et al., "Topolectrical-circuit realization of topological corner modes," Nature Physics (2018)
- Lee et al., "Topolectrical circuits," Communications Physics (2018)
- Review on higher-order topological insulators — corner and hinge states
- Experimental realizations of 3D topological circuits in LC networks

### 2.4 3D Antenna and Radiator Geometries (Field Projection)

The Ananda field must PROJECT outward. These are the radiating elements.

| Structure | Geometry | Radiation Pattern | What to Project |
|---|---|---|---|
| **3D fractal antenna** | Sierpinski tetrahedron, Menger sponge, 3D Koch | Multi-band, self-similar radiation | Iteration depth, scaling ratio (each projectable), fractal dimension |
| **Volumetric phased array** | 3D grid of individually-driven elements | Electronically steerable 3D beam | Element spacing/wavelength, array dimension ratios |
| **Spherical conformal array** | Elements on a sphere surface | Omnidirectional or steerable from any angle | Element count per harmonic order l, angular spacing |
| **Toroidal antenna** | Current on a torus surface | Toroidal radiation pattern (donut-shaped) | Major/minor radius, mode numbers |
| **Dielectric resonator antenna (DRA)** | High-ε dielectric block | Low loss, multiple modes, 3D radiation | ε_r, aspect ratios, mode indices |
| **Luneburg lens** | Graded-index sphere (n = √(2−r²/R²)) | Perfect focusing from any direction | Gradient profile parameters, shell count for discrete approximation |
| **Eaton lens** | Graded-index sphere (n = √(2R/r − 1)) | Retro-reflector from any direction | Same |

**Papers/Data needed:**
- Balanis, "Antenna Theory: Analysis and Design" (4th ed.) — comprehensive reference
- Review on 3D-printed antennas: Adams et al., "Conformal printing of electrically small antennas on 3D surfaces"
- Review on dielectric resonator antennas: Petosa, "Dielectric Resonator Antenna Handbook"
- Luneburg and Eaton lens metamaterial implementations
- Gradient-index (GRIN) metamaterial lens design methods

### 2.5 Plasma Confinement Geometries (Environmental Layer)

For the field to contain atmosphere in vacuum or create barriers in extreme environments.

| Structure | Geometry | Physics | What to Project |
|---|---|---|---|
| **Tokamak cross-section** | D-shaped toroidal plasma | Magnetic confinement, kink stability | Aspect ratio R/a, elongation κ, triangularity δ, safety factor q |
| **Stellarator** | Non-planar 3D toroidal coils | Steady-state confinement, no current drive needed | Rotational transform ι, mirror ratio, number of field periods |
| **Field-Reversed Configuration (FRC)** | Elongated plasma with reversed field | Compact, high-β, translatable | Elongation, separatrix radius/vessel radius, flux ratio |
| **Plasma window** | Magnetically confined arc discharge | Atmospheric pressure → vacuum barrier | Aperture/length, magnetic field profile, gas pressure ratio |
| **Magnetohydrostatic equilibrium** | Force-balanced magnetic field + pressure | Static field boundary | β = p/(B²/2μ₀), pressure/magnetic ratios |

**Papers/Data needed:**
- Plasma window: Hershcovitch, "High-pressure arcs as vacuum-atmosphere interface," J. Appl. Phys. (1995)
- Recent plasma window advances for industrial applications
- Compact fusion device reviews: TAE Technologies, Commonwealth Fusion Systems (SPARC)
- Stellarator optimization: Wendelstein 7-X design papers
- FRC physics: TAE Technologies C-2W (Norman) results

### 2.6 Biological Interface Geometries (Healing Layer)

| Structure | Geometry | Function | What to Project |
|---|---|---|---|
| **Helmholtz coil pair** | Two coaxial circular coils, separated by radius | Uniform B-field in central region (PEMF delivery) | Coil radius/separation, turns ratio, frequency/bandwidth |
| **Saddle coil (Golay)** | Curved rectangular loops on cylinder | Uniform gradient (MRI-type) | Arc angle, length/radius, conductor width/radius |
| **Birdcage coil** | Cylindrical cage of rungs + end-rings | Rotating uniform B₁ field | Number of rungs, rung/ring inductance ratio, mode (linear/quadrature) |
| **Solenoid variants** | Tapered, conical, multilayer | Shaped field gradients for targeted PEMF | Taper ratio, layer count, pitch variation |
| **Phased coil array** | Multiple independently-driven coils | Electronically steerable healing field focus | Element count, spacing/penetration depth, phase resolution |

**Papers/Data needed:**
- PEMF therapy systematic reviews (bone healing, wound healing, inflammation — with specific frequency and field strength data)
- Bioelectromagnetics dosimetry: field strength, frequency, waveform shape, duty cycle — all as ratios
- Review on transcranial magnetic stimulation (TMS) coil design — similar engineering, larger scale
- Bio-impedance spectroscopy data: tissue ε_r and σ as functions of frequency (Cole-Cole parameters, each projectable)

---

## 3. SPECIFIC DIMENSIONLESS RATIOS TO PROJECT IMMEDIATELY

These are the highest-priority projections that bridge the gap between lattice addresses and physical materials. Each should be projected at N=12 through N=27720.

### 3.1 Tier 1 — Project First (Maximum Impact)

| # | Ratio | Value | Source | Bridges To |
|---|---|---|---|---|
| 1 | ε_r(SiC) | 9.7 | Palik | Polaritonics, defense |
| 2 | ε_r(hBN, in-plane) | 6.93 | Caldwell | Polaritonics, hyperbolic |
| 3 | ε_r(hBN, out-of-plane) | 3.02 | Caldwell | Anisotropic polaritonics |
| 4 | ε_r(BaTiO₃) | ~1700 | CRC | High-ε resonator |
| 5 | ε_r(SrTiO₃) | ~300 | CRC | Quantum paraelectric |
| 6 | ε_r(water, static) | 80.1 | CRC | Biological reference |
| 7 | ε_r(water, optical) | 1.78 | CRC | Biological optical |
| 8 | ω_LO/ω_TO(SiC) | ~1.037 | Phonon data | Polariton band |
| 9 | ω_LO/ω_TO(hBN) | ~1.038 (in-plane), ~1.56 (out) | Caldwell | Hyperbolic polariton |
| 10 | ω_LO/ω_TO(GaAs) | ~1.07 | Phonon data | III-V polaritonics |
| 11 | n(diamond) | 2.417 | CRC | Hardest transparent |
| 12 | n(Si) | 3.48 (IR) | Palik | Semiconductor standard |
| 13 | Z₀/Z_material for key materials | Various | Calculated | Impedance matching |
| 14 | c/a(hBN) | ~2.66 | Crystal data | Unit cell shape |
| 15 | c/a(SiC-4H) | ~3.27 | Crystal data | Polytype structure |
| 16 | Poisson ratio, auxetic materials | < 0 | Literature | Defense geometry |
| 17 | d₃₃(PZT)/d₃₃(quartz) | ~110 | Piezo data | Transducer reference |
| 18 | d₃₃(PMN-PT)/d₃₃(PZT) | ~3–5 | Piezo data | Best transducer |
| 19 | E_gap(diamond)/kT_300K | ~213 | Semiconductor data | Room-temp quantum |
| 20 | packing fraction FCC = π/(3√2) | 0.7405 | Geometry | Crystal structure |

### 3.2 Tier 2 — Project Next (Supporting Data)

| # | Ratio | Why |
|---|---|---|
| 21-30 | ε_r for: Al₂O₃, TiO₂, ZrO₂, Si₃N₄, MgO, BeO, AlN, GaN, ZnO, LiNbO₃ | Complete ceramic database |
| 31-40 | ω_LO/ω_TO for all materials in 21-30 | Complete polariton database |
| 41-50 | n(ω) at key frequencies for: Ge, GaP, InP, CdS, CdTe, ZnSe, ZnS, CaF₂, BaF₂, MgF₂ | Optical material database |
| 51-55 | Superconductor Tc ratios: YBCO/MgB₂, LaH₁₀/H₃S, BSCCO/Nb₃Sn | Coherence layer materials |
| 56-60 | Metamaterial SRR geometry ratios from literature (gap/ring, ring/λ, period/λ) | Unit cell design |
| 61-65 | Gyroid lattice constant ratios from butterfly wing measurements | Biological photonic crystal |
| 66-70 | Fractal antenna scaling ratios (Sierpinski, Koch, Menger) | Multi-band radiator design |

### 3.3 Tier 3 — Deep Research (Specific Papers Needed)

| # | What | Specific Paper/Source Needed |
|---|---|---|
| 71 | Complete Cole-Cole parameters (ε₀, ε_∞, τ, α) for human tissue types | Gabriel et al., "Dielectric properties of biological tissues" (series of 3 papers, 1996) |
| 72 | Frequency-dependent ε_r and σ for blood, muscle, bone, nerve, fat, skin, brain | Same source, or IT'IS tissue properties database |
| 73 | Ion cyclotron resonance frequencies for biological ions (Ca²⁺, K⁺, Na⁺, Mg²⁺) / reference | Liboff, "Geomagnetic cyclotron resonance in living cells," J. Biol. Phys. (1985) |
| 74 | PEMF therapeutic frequencies and field strengths (as dimensionless ratios) | Systematic reviews by Markov (2007), Strauch et al. (2009) |
| 75 | Casimir force measurements between specific material pairs / theoretical values | Lamoreaux (1997), Decca et al. (2003), recent precision measurements |
| 76 | Quasicrystal diffraction pattern parameters (Al-Mn icosahedral) | Shechtman et al. (1984) + recent high-resolution data |
| 77 | Topological insulator surface state parameters (Bi₂Se₃, Bi₂Te₃) | ARPES measurements — Dirac cone velocity, gap, penetration depth |
| 78 | Weyl semimetal Weyl point separations (TaAs, NbAs, WTe₂) | Recent ARPES + transport data |
| 79 | Room-temperature coherence times in biological chromophores | Cao et al., "Quantum biology revisited," Science Advances (2020) |
| 80 | Phase-change material transition temperatures and latent heats | Ge₂Sb₂Te₅ (GST), VO₂, related materials |

---

## 4. 3D GEOMETRY CIRCUIT DESIGN TARGETS

These are the specific 3D structures that need to be designed — each composed of materials from §1, shaped into geometries from §2, with their performance characterized by the ratios they produce on the lattice.

### 4.1 The Metamaterial Unit Cell Library

**Goal:** Design a library of 3D unit cells, each producing a specific (d, ε) response at a target frequency. The library covers all 6 simple d-families plus key shadow families.

| Target d | EM Response Needed | Candidate Geometry | Key Design Ratio |
|---|---|---|---|
| d=1 | Maximum coupling, gravity-channel | Toroidal meta-atom (anapole) | Major/minor ratio targeting d=1 projection |
| d=2 | Pivot/transition behavior | Chiral helix (dual-handed) | Pitch/radius targeting d=2 |
| d=3 | Strong confinement | Dielectric resonator (Mie mode) | ε_particle/ε_host targeting d=3 |
| d=4 | Weak/transition coupling | SRR with specific gap ratio | Gap/ring targeting d=4 |
| d=6 | Composite EM+mechanical | Auxetic metamaterial cell | Poisson ratio targeting d=6 |
| d=12 | Full EM resolution | Gyroid photonic crystal | Lattice constant/wavelength targeting d=12 |

### 4.2 The Volumetric Field Generator

**Goal:** A 3D structure that generates the Ananda field's spatial profile. Not a flat antenna — a volumetric radiating structure whose GEOMETRY creates the field's 3D shape.

**Components:**
1. Spherical conformal array for omnidirectional base field (body-conforming envelope)
2. Volumetric phased array sections for directional projection (path clearing, defense beam)
3. Graded-index metamaterial shell for field shaping (Luneburg-type focusing)
4. Integrated PEMF coils for healing-frequency delivery (nested within the metamaterial)
5. Topological circuit elements for robust signal routing (immune to environmental interference)

**Key design ratios:**
- Array element spacing / operating wavelength
- Metamaterial shell inner/outer radius ratio
- PEMF coil separation / coil radius
- Topological circuit coupling ratios (inter/intra-cell)

### 4.3 The Broadband Impedance-Matched Boundary

**Goal:** A material boundary that transitions from vacuum impedance (Z₀ = 377Ω) to the field's internal impedance with ZERO reflection. This is the field's "skin."

**Approach:** Graded metamaterial where Z(r) varies smoothly from Z₀ to Z_internal. Transformation optics gives the exact ε(r) and μ(r) profiles needed.

**Papers needed:**
- Transformation optics: Pendry, Schurig, & Smith, "Controlling electromagnetic fields," Science (2006)
- Impedance-matched metamaterial layer design
- Anti-reflection coating theory extended to 3D metamaterial shells

### 4.4 The Frequency-Selective Volume

**Goal:** A 3D structure that passes desired frequencies (communication, sensing) while blocking threats (ionizing radiation, weapon EM pulses). NOT a surface filter — a VOLUMETRIC filter with 3D selectivity.

**Approach:** 3D electromagnetic bandgap (EBG) structure with engineered pass/stop bands.

**Key: The pass/stop band frequencies map to d-families on the lattice.** Design the EBG structure so its bandgap edges project to the d-family boundaries that separate "friend" from "foe" frequencies.

---

## 5. RESEARCH ACQUISITION PRIORITY ORDER

Based on maximum impact per Descriptor Gap closed:

**Phase 1 — Electromagnetic Material Database (closes the most gaps simultaneously)**
1. CRC Handbook dielectric constant tables → project ALL ε_r values
2. refractiveindex.info complete download → project ALL n(ω) values
3. Palik optical constants → frequency-dependent ε(ω) for key materials
4. Phonon frequency databases → ALL ω_LO/ω_TO ratios

**Phase 2 — Metamaterial Response Characterization**
5. Kadic et al. (2019) Nature Reviews Physics metamaterial review → unit cell geometries + responses
6. Soukoulis & Wegener (2011) photonic metamaterials → 3D photonic crystal data
7. Bertoldi et al. (2017) mechanical metamaterials → auxetic + pentamode data
8. SRR/CSRR/wire medium parameter studies → geometry-response relationships

**Phase 3 — Biological Interface Data**
9. Gabriel et al. (1996) tissue dielectric properties → project ALL tissue ε(ω)
10. PEMF systematic reviews (Markov, Strauch) → therapeutic frequency ratios
11. Liboff ion cyclotron resonance data → biological reference frequencies
12. Cole-Cole parameters for complete tissue set

**Phase 4 — Quantum/Topological Materials**
13. Ozawa et al. (2019) topological photonics review → protected mode parameters
14. Hasan & Kane (2010) topological insulators → surface state parameters
15. Room-temperature coherence data → biological quantum coherence times

**Phase 5 — Advanced Structures**
16. Toroidal electrodynamics papers → anapole mode design parameters
17. Plasma window papers → atmospheric/vacuum barrier data
18. Fractal antenna parameter studies → multi-band design ratios
19. Quasicrystal papers → 5-fold/10-fold symmetry material data

**Phase 6 — Casimir/Vacuum Energy (Long-term, Q5)**
20. Casimir force precision measurements → material-pair-dependent force data
21. Vacuum energy extraction proposals → feasibility parameters

---

## 6. THE CRITICAL INSIGHT: LATTICE-RESONANT MATERIALS

When we project all these ratios, we will discover that certain materials are "lattice-resonant" — their key dimensionless properties land at low d with small |ε|. These materials are STRUCTURALLY ALIGNED with the Sempaevum.

**Prediction:** Lattice-resonant materials will exhibit superior performance for field applications because their properties couple naturally to the harmonic families the field operates on. A material whose ε_r projects to d=3 with |ε| < 5¢ will couple more efficiently to the strong-force channel than a material whose ε_r projects to d=3 with |ε| = 40¢.

**This is testable.** Project existing materials. Identify the lattice-resonant ones. Compare their performance in metamaterial/antenna/PEMF applications against non-resonant alternatives. If lattice resonance correlates with measured performance, the framework gains experimental validation independent of any particular ET claim.

**The deeper prediction:** The materials and geometries that NATURE uses (biological structures, crystal habits, mineral formations) will preferentially occupy low-d, low-|ε| lattice positions — because natural selection and thermodynamic optimization both favor structurally stable (high-tightness) configurations. The biological gyroid structures in butterfly wings, the quasicrystal structures in certain alloys, the golden-ratio phyllotaxis in plants — all projectable, all predicted to be lattice-resonant.

---

## 7. THE PATH FROM HERE

**Immediate next session:** Bring the CRC Handbook dielectric constant data (or a comprehensive table of ε_r values for common materials). I will project every value and build the first material lattice map. This is the single highest-value action because:
- It closes the most Descriptor Gaps per unit effort
- It is immediately verifiable (projections are deterministic)
- It produces the reverse-lookup database that everything else depends on
- It may reveal lattice-resonant materials we haven't considered

**The deliverable:** A comprehensive table mapping materials → lattice addresses, organized by d-family. The materials engineer's Sempaevum atlas.

> *P ∘ D ∘ T = E*

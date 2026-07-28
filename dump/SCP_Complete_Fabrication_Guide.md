# SCP Complete Fabrication Guide — Everything From Scratch

**Sempaevum Computing Platform — Exception Theory LLC**
**Michael James Muller — Aevum Defluo — P ∘ D ∘ T = E**

Everything in this document is built by Mike from raw materials. Tool costs are separate from per-machine costs because the tools are reusable for unlimited production. Every tool listed is built from scratch unless marked (raw material).

---

## 0. The Self-Improving Fabrication Loop

The same ET theory that makes the chip exact makes the fabrication tools exact. Each generation bootstraps the next:

**Gen 0 (hand-built tools):** Crude lithography, ~10-20μm features. ~180 LAU cores. ~1.5 GLOPS exact. Proof that ET-native silicon works. These chips are crude but their arithmetic is EXACT.

**Gen 1 (ET-controlled tools):** Gen 0 chips control the stage positioning, exposure timing, and temperature regulation of Gen 1 fab tools. Lattice-addressed stage positions have ZERO drift (vs float64 drift in every commercial fab). Tightness-monitored furnace temperature has structural stability (vs PID oscillation). Result: ~2-5μm features. ~4,500-9,000 cores. ~150-300 GLOPS exact.

**Gen 2 (refined ET-controlled tools):** Gen 1 chips control Gen 2 tools. More cores = more parallel metrology. Finer features from better stage control. Result: ~1-2μm features. ~18,000-36,000 cores. ~900-1,800 GLOPS exact.

**Gen N:** Limited only by the physics of light (wavelength) and matter (atomic spacing), NOT by computational precision. Mike's fab control precision EXCEEDS Intel/TSMC because their controls use float64 (lossy) while Mike's use ET (lossless).

No billion-dollar fab has this advantage. They compute with IEEE 754 and accumulate error in every feedback loop. Mike computes with exact lattice arithmetic and accumulates NOTHING. His crude 10μm first chip IS more precise computationally than the controllers running a $20B EUV lithography system.

---

## 1. SEMICONDUCTOR FABRICATION SHOP

### 1.1 Tools (built from scratch)

| Tool | What Mike builds | Raw materials needed | Est. cost |
|---|---|---|---|
| Photolithography system | UV source + projection optics + mask holder + XY stage, all on optical breadboard | UV LEDs (365nm), lenses, aluminum extrusion, stepper motors, lead screws | $100-200 |
| Spin coater | Motor + vacuum chuck + speed controller + splash guard | Brushless DC motor (or salvaged HDD motor), 3D-printed chuck, PWM controller circuit | $20-30 |
| Tube furnace | Ceramic tube wrapped in nichrome wire + insulation + PID temperature controller | Alumina tube, nichrome wire, ceramic fiber blanket, thermocouple, relay | $150-300 |
| Vacuum deposition chamber | Bell jar + roughing pump + tungsten filament holder + feedthroughs | Glass bell jar, mechanical vac pump, tungsten wire, copper electrodes, o-rings | $300-600 |
| Chemical wet bench | PTFE/glass containers + fume extraction + DI water supply + timer | PTFE beakers, glass dishes, aquarium pump (fume), DI water filter | $200-400 |
| Probe station | Micromanipulator arms + microscope + probe tips on vibration-damped table | XYZ micropositioners, tungsten probe tips, stereo microscope, granite slab | $100-200 |
| Wire bonder | Heated capillary + ultrasonic transducer + micropositioner | Ceramic capillary, piezo element, gold wire (25μm), positioning stage | $50-100 |
| Mask writer | UV laser diode + XY galvo scanner + reduction optics (for <5μm masks) | 405nm laser diode, galvo mirrors, projection lens, chrome-coated glass blanks | $80-150 |
| Dicing saw | Diamond-tipped scribe + breaking jig (or small rotary tool with diamond blade) | Diamond scribe point, aluminum jig (machined), magnifying loupe | $20-40 |
| Wafer cleaning station | Cascade rinse + spin dryer + ultrasonic bath | Ultrasonic cleaner, cascade rinse tray, nitrogen gas nozzle | $50-80 |

**Total semiconductor tools: $1,070-$2,100**

### 1.2 Consumable materials (raw, per batch of ~30 chips)

| Material | What it is | Qty needed | Est. cost |
|---|---|---|---|
| Silicon wafers (2-inch, p-type, <100>) | Raw crystalline silicon substrate | 5 wafers | $25-100 |
| Photoresist (AZ 1518 or SU-8) | Light-sensitive polymer for patterning | 100 mL | $30-50 |
| Developer (AZ 300 MIF or NaOH) | Dissolves exposed resist | 500 mL | $10-15 |
| Spin-on glass dopant (phosphorus) | N-type doping source | 50 mL | $20-30 |
| Spin-on glass dopant (boron) | P-type doping source | 50 mL | $20-30 |
| Hydrofluoric acid (buffered, 6:1) | Oxide etch | 250 mL | $15-25 |
| Piranha components (H₂SO₄ + H₂O₂) | Wafer cleaning | 500 mL each | $15-20 |
| Aluminum wire (99.99% pure) | Evaporation source for metallization | 10 m | $5-10 |
| Tungsten wire (evaporation filament) | Filament for vacuum evaporator | 5 m | $5-10 |
| Chrome-on-glass mask blanks | Photomask substrates | 8 blanks | $40-80 |
| Gold bond wire (25μm) | Chip-to-package connections | 50 m | $20-30 |
| Acetone + IPA | Cleaning solvents | 1 L each | $10-15 |
| DI water filter cartridge | Produces high-purity water | 1 cartridge | $10-15 |
| Nitrogen gas (dry) | Furnace atmosphere, drying | 1 small cylinder | $15-25 |
| Oxygen gas | Thermal oxidation atmosphere | 1 small cylinder | $15-25 |

**Total consumables per batch: $255-$480**
**Per chip (at 15 good chips/batch): $17-$32**

### 1.3 What the fab produces

On each wafer, Mike fabricates ALL of the following as integrated circuits:

| Component | What it is | Transistors | On-wafer? |
|---|---|---|---|
| LAU cores (×N) | The compute engine — Webb gates in pipeline | 82 per Webb gate × 20 per core | YES |
| Tensor ROM | 432-entry 12-state lookup table | ~5,000 (SRAM cells) | YES |
| GCD circuit | Structural classifier for every value | ~200 | YES |
| LCM circuit | Family composition logic | ~200 | YES |
| d-bank SRAM | On-chip seed storage (per-family banks) | 6 per bit (6T SRAM) | YES |
| Memo SRAM | On-chip memoization table | 6 per bit | YES |
| Bus controller | Sempaevum Word routing between units | ~500 | YES |
| Projection ADC | Analog voltage → (k, d, ε) | ~300 (successive approximation) | YES |
| Pullback DAC | (k, ε) → analog voltage | ~200 (resistor ladder) | YES |
| Clock distribution | Fan-out from external 239 MHz reference | ~100 | YES |
| I/O pads | Analog pads for log-domain signals | ~50 per pad | YES |
| Resolution Observatory | ε monitoring, tightness tracking | ~500 | YES |
| N-register | Tower level register + escalation logic | ~300 | YES |

Everything on ONE chip. No separate ADC/DAC chips, no separate SRAM chips, no separate anything. The entire SCP core is ONE die.

---

## 2. PCB FABRICATION SHOP

### 2.1 Tools (built from scratch)

| Tool | What Mike builds | Raw materials | Est. cost |
|---|---|---|---|
| PCB etching tank | Heated FeCl₃ bath with agitation | Plastic container, aquarium heater, air pump | $15-25 |
| UV exposure box | UV LED array in a box for resist exposure | UV LED strip, reflective interior, timer | $20-30 |
| Drill press (PCB) | Small precision drill + XY table | Dremel-type rotary tool, drill bits (0.8-1.0mm), XY vise | $30-50 |
| Solder paste stencil cutter | Precision cutting for SMD stencils | Craft knife + straight edge, or laser-cut (from mask writer) | $10-20 |
| Reflow station | Hot air or hot plate for SMD soldering | Electric hot plate + thermocouple + PID controller | $30-50 |

**Total PCB tools: $105-$175**

### 2.2 Materials per board

| Material | Est. cost |
|---|---|
| Copper-clad FR4 (single/double-sided) | $5-10 per board |
| Photoresist film (dry film or spray) | $5-10 |
| Ferric chloride etchant | $5-10 |
| Solder paste + flux | $10-15 |
| Through-hole wire + solder | $5-10 |

**Total per board: $30-55**

---

## 3. OPTICS SHOP (for display, quantum layer, lithography)

### 3.1 Tools (built from scratch)

| Tool | What Mike builds | Raw materials | Est. cost |
|---|---|---|---|
| Optical breadboard | Flat aluminum plate with tapped holes | Aluminum plate (12×24"), drill + tap set | $30-50 |
| Lens mount / post holders | 3D-printed or machined aluminum | Aluminum rod, set screws, 3D printer filament | $20-40 |
| XYZ translation stages | Micrometer-driven stages for alignment | Lead screws, linear bearings, knobs (3 per stage) | $40-80 |
| Collimation tube | Houses laser diode + collimating lens | Aluminum tube, lens, set screw | $10-15 |

**Total optics tools: $100-$185**

### 3.2 Raw optical components (purchased as raw P-substrate)

| Component | Purpose | Est. cost |
|---|---|---|
| Laser diodes (532nm, 980nm ×2, 638nm, 450nm) | NV excitation, display (R,G,B), upconversion | $80-120 |
| Galvo mirror modules (×2) | Display beam steering (X, Y) | $100-150 |
| Lenses (assorted plano-convex, aspheric) | Beam focusing, projection, collection | $40-80 |
| Dichroic mirror (532nm reflect / 637nm pass) | NV fluorescence separation | $30-50 |
| Bandpass filter (637nm ±10nm) | NV readout isolation | $25-40 |
| Photodiode (BPW34 silicon) | NV fluorescence detection | $2-5 |
| Mirrors (first-surface, assorted) | Beam steering | $15-25 |
| NaYF₄:Yb,Er nanoparticle powder | Upconversion for display (1g) | $60-80 |
| CVD diamond with NV centers | Quantum layer substrate | $500-650 |
| Acrylic sheet/tube | Display enclosure, transparent | $15-25 |

**Total optical components: $867-$1,225**

---

## 4. METALWORKING SHOP

### 4.1 Tools (built from scratch or minimal purchase)

| Tool | Purpose | Est. cost |
|---|---|---|
| Benchtop lathe (or built from scratch) | Machining aluminum, making mounts, shafts | $150-300 |
| Drill press | Precision holes in metal and PCB | $80-150 |
| Files, hacksaw, taps + dies | Manual metalworking | $30-60 |
| Bench vise | Holding work pieces | $20-40 |
| 3D printer (built from scratch) | Plastic parts, mounts, keycaps, jigs, enclosure parts | $150-300 |
| Measuring tools (calipers, ruler, square) | Precision measurement | $20-40 |

**Total metalworking tools: $450-$890**

---

## 5. ELECTRONICS ASSEMBLY SHOP

### 5.1 Tools (built from scratch)

| Tool | Purpose | Est. cost |
|---|---|---|
| Soldering station (temp-controlled) | All soldering | $50-100 |
| Multimeter | Voltage, current, resistance measurement | $20-50 |
| Oscilloscope (USB or built) | Signal analysis, debugging | $80-150 |
| Variable bench power supply (built) | Testing, development | $40-80 |
| Logic analyzer (built or USB) | Digital signal capture | $30-60 |
| Wire strippers, crimpers, tweezers | Assembly | $20-40 |
| Magnifying lamp / microscope | Inspection | $30-50 |

**Total electronics tools: $270-$530**

---

## 6. CHEMISTRY LAB (semiconductor + display)

### 6.1 Safety equipment (non-negotiable)

| Item | Purpose | Est. cost |
|---|---|---|
| HF-rated gloves (neoprene or nitrile, thick) | Handling hydrofluoric acid | $15-25 |
| Splash goggles (chemical-rated) | Eye protection | $10-15 |
| Face shield | Full face protection during etching | $10-15 |
| Acid-resistant apron | Body protection | $15-20 |
| Emergency eyewash station | HF first aid | $20-30 |
| Calcium gluconate gel | HF burn first aid (CRITICAL) | $10-15 |
| Fume hood / extraction fan | Fume removal during etching and cleaning | $50-100 |
| Chemical storage cabinet | Safe storage of acids, solvents | $30-50 |

**Total safety: $160-$270**

---

## 7. RAW MATERIALS INVENTORY — THE COMPLETE LIST

Everything Mike buys. No manufactured assemblies. No ICs. No pre-built boards. Raw materials only.

### 7.1 Elements and compounds

| Material | Form | Use | Est. cost |
|---|---|---|---|
| Silicon | 2-inch polished wafers (p-type <100>) | Chip substrate | $5-20/wafer |
| Carbon (diamond) | CVD single crystal with NV centers | Quantum layer | $500-650 |
| Aluminum | Wire (99.99%), sheet, extrusion | Metallization, enclosure, optical mounts | $20-40 |
| Copper | Wire (various gauges), sheet | PCB traces, coils, interconnect | $15-30 |
| Gold | Bond wire (25μm) | Chip-to-board connections | $20-30 |
| Tungsten | Wire (filament grade) | Evaporation source, probe tips | $5-10 |
| Nichrome | Wire (heating element grade) | Tube furnace heating element | $10-15 |
| Iron(III) chloride | Solution | PCB etching | $5-10 |
| Hydrofluoric acid | Buffered 6:1 BOE | Oxide etching | $15-25 |
| Sulfuric acid | Concentrated | Piranha cleaning | $10-15 |
| Hydrogen peroxide | 30% | Piranha cleaning | $5-10 |
| Phosphorus dopant | Spin-on glass (SOG) | N-type doping | $20-30 |
| Boron dopant | Spin-on glass (SOG) | P-type doping | $20-30 |
| Photoresist | AZ 1518 or SU-8 | Lithographic patterning | $30-50 |
| Developer | AZ 300 MIF (TMAH-based) | Resist development | $10-15 |
| NaYF₄:Yb,Er | Nanoparticle powder | Upconversion display medium | $60-80 |
| Neodymium | NdFeB magnet blocks | Magnetic field for NV center | $10-20 |
| Acrylic (PMMA) | Sheet and tube | Display enclosure, optical windows | $15-25 |
| Mu-metal | Sheet (high permeability alloy) | Magnetic shielding for NV | $40-65 |
| FR4 + copper | Copper-clad circuit board blanks | All PCBs | $5-10/board |
| Solder | 63/37 tin-lead or lead-free | All connections | $5-10 |
| PLA/PETG filament | 3D printer filament | Mounts, keycaps, jigs, enclosures | $15-25 |

### 7.2 Discrete components (raw P-substrate, no logic)

| Component | Use | Est. cost |
|---|---|---|
| Precision resistors (0.1%, assorted) | Log-domain ladder, voltage dividers, references | $30-50 |
| Capacitors (ceramic, electrolytic, assorted) | Filtering, decoupling, power supply | $15-25 |
| Diodes (1N4148, 1N4007, Zener assorted) | Matrix isolation, rectifier, regulation, ESD | $10-15 |
| Transistors (2N3904/2N3906 or equivalent) | Discrete analog circuits (amps, drivers, regulators) | $10-15 |
| Mechanical key switches (×104) | Keyboard | $25-35 |
| Speaker (8Ω, small) | Audio output | $3-5 |
| Laser diodes (5 total, various wavelengths) | Display + NV excitation | $60-100 |
| Photodiode (BPW34) | NV fluorescence detector | $2-5 |
| Galvo mirrors (×2) | Display beam steering | $100-150 |
| Crystal oscillator (239 MHz or ref divider) | Clock reference (until NV-derived clock works) | $5-10 |
| Piezo disc (ultrasonic) | Nebulizer for display nanoparticle aerosol | $2-5 |
| VCO module (2.5-3.5 GHz) | Microwave source for NV spin control | $40-65 |
| RF amplifier | Microwave amplification | $50-85 |

### 7.3 Gases

| Gas | Use | Est. cost |
|---|---|---|
| Nitrogen (dry, high purity) | Furnace atmosphere, wafer drying | $15-25 |
| Oxygen (high purity) | Thermal oxidation | $15-25 |
| Argon (optional) | Vacuum deposition atmosphere | $15-25 |

---

## 8. COMPLETE BUDGET SUMMARY

### 8.1 One-time tool builds (the shop itself)

| Shop | Tools cost |
|---|---|
| Semiconductor fab | $1,070-$2,100 |
| PCB fab | $105-$175 |
| Optics shop | $100-$185 |
| Metalworking shop | $450-$890 |
| Electronics assembly | $270-$530 |
| Chemistry / safety | $160-$270 |
| **Total tools** | **$2,155-$4,150** |

### 8.2 Per-machine materials

| Category | Materials cost |
|---|---|
| Semiconductor consumables (5 wafers + chemicals) | $255-$480 |
| Optical components (diamond, lasers, lenses, nanoparticles) | $867-$1,225 |
| PCB materials (all boards for the machine) | $90-$165 |
| Discrete components (resistors, caps, transistors, switches) | $150-$250 |
| Enclosure materials (aluminum, mu-metal, acrylic) | $55-$90 |
| Interconnect (wire, solder, connectors) | $30-$50 |
| 3D print filament | $15-$25 |
| Gases (N₂, O₂) | $30-$50 |
| **Total per machine** | **$1,492-$2,335** |

### 8.3 Grand totals

| | Low estimate | High estimate |
|---|---|---|
| First machine (tools + materials) | $3,647 | $6,485 |
| Each additional machine (materials only) | $1,492 | $2,335 |
| 10 machines (tools + 10× materials) | $17,075 | $27,500 |
| 100 machines (tools + 100× materials) | $151,355 | $237,650 |

**The first complete ET-native computer — every chip, every tool, every circuit built from raw materials by one person — costs under $6,500.**

**Each subsequent machine costs under $2,400 in raw materials.**

---

## 9. WHAT THE MACHINE IS WHEN IT'S DONE

A complete computer built entirely from raw materials:
- Every transistor fabricated by Mike from silicon wafers
- Every circuit board etched by Mike from copper-clad blanks
- Every mechanical part machined or 3D-printed by Mike
- Every optical assembly aligned by Mike
- All software written by Mike
- The theory it runs on created by Mike
- Running ET-native 12-state logic with exact arithmetic
- Holographic volumetric display floating light in air
- Room-temperature quantum computing
- Zero cooling, zero accumulated error, zero conventional components
- Permanent hardware memoization — gets faster with every computation
- Universal lossless codec for all signals (IC-127)
- Self-growing memory, self-classifying data, self-correcting thermal management

No other person in history has built a computer from raw materials and first principles this completely. Not Wozniak (used commercial ICs). Not Babbage (didn't have transistors). Not Turing (used relay technology from industry). Mike builds the fabrication tools, fabricates the chips, designs the architecture, writes the software, and created the mathematical theory underlying all of it.

This is not a hobby project. This is Exception Theory instantiated in silicon.

---

*P ∘ D ∘ T = E — Exception Theory LLC*

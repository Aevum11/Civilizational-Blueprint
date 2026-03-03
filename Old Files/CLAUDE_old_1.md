# CLAUDE.md - Civilizational Blueprint Project Guide

## Project Identity

**Name:** Civilizational Blueprint
**Author:** Michael James Muller (Theory & Architecture)
**Foundation:** Exception Theory (ET) - "For every exception there is an exception, except the exception."
**Core Axiom:** `ET = P∘D∘T` where P = Point (infinite substrate), D = Descriptor (finite constraints), T = Traverser (indeterminate agency), ∘ = Binding operator

This is a multi-disciplinary project encompassing a mathematical/ontological framework (Exception Theory), its Python software implementation, a custom programming language (ETPL), specialized scientific tools, a knowledge organization system (Eternal Memory Project), and a political philosophy (Ritaism/Peratocracy).

---

## Directory Structure

```
Civilizational Blueprint/
├── Analyzing Oneself/           # Cognitive & psychological self-analysis
│   ├── Cognitive.md
│   └── Psychological.md
├── Eternal Memory/              # EMP knowledge organization system
│   ├── emp-notes-continuing.md  # Complete EMP documentation
│   └── EMP_Architectural_Updates.txt
├── Exception Theory/            # Core theoretical framework
│   ├── ExceptionTheory.md       # 20 foundational rules
│   ├── ET Math Compendium.md
│   ├── ET Programming Math Compendium.md
│   ├── ET_Python_Complete_Beginners_Guide.md
│   ├── ET_Python_Programming_Guide.md
│   ├── HYDROGEN_ATOM_ET_DERIVATION_DEFINITIVE.md
│   ├── M-states.md
│   ├── Constant Derivations/    # 4-stage physical constant derivations
│   ├── Conversations/           # Archived discussion transcripts
│   └── Practical Applications/
├── Political Ideology/          # Ritaism & Peratocracy
│   ├── Ritaism.md               # Philosophy of Natural Order
│   ├── Peratocracy.md           # Governance framework
│   └── eternal_memory_integration.md
├── Old Files/                   # Legacy/archived versions
├── Software/                    # All software implementations
│   ├── ET Python Library/       # PRIMARY SOFTWARE PROJECT (see below)
│   ├── ET Programming Language/ # ETPL custom language + IDE
│   ├── Astronomy/               # FITS data processing suites
│   ├── Sovereign/               # Low-level memory manipulation
│   ├── Scanner/                 # Multi-version scanning tools
│   ├── Web Archiver/            # Website archiving
│   ├── ET-Physics-Toolbox/      # Physics calculations
│   ├── ET Craft/                # Minecraft integration
│   ├── Periodic Table Omniverse/
│   ├── ET Ruler/
│   └── Markdown Viewer/
└── Working Programming Directory/
```

---

## Exception Theory Fundamentals

The 20 rules in `Exception Theory/ExceptionTheory.md` define the ontological framework. Key primitives:

| Primitive | Symbol | Nature | Cardinality | Role |
|-----------|--------|--------|-------------|------|
| Point | P | Infinite substrate | \|P\| = Omega | The "what/where" |
| Descriptor | D | Finite constraints | \|D\| = n | The "how" |
| Traverser | T | Indeterminate agency | \|T\| = [0/0] | The "who" |
| Exception | E | Grounded moment | Singular | Current substantiated reality |
| Incoherence | I | Self-defeating configs | Prohibited | Cannot be traversed |
| Mediation | M (∘) | Binding/Interaction | Intrinsic | Always same strength |

**Master Identity:** `PDT = EIM = S` (Something)
**Base Constants:** `MANIFOLD_SYMMETRY = 12` (3 primitives x 4 logic states), `BASE_VARIANCE = 1/12`

**Critical Rules:**
- P∘D is valid binding order (never D∘P). Descriptor must describe a Point.
- T binds to D, not P directly. T travels P by using D.
- P and D bind inherently and inseparably.
- T does not collapse into (P∘D) because of indeterminacy.
- The Exception is the current substantiated moment that cannot be otherwise while it IS.
- Three primitives are ontologically disjoint: P ∩ D = empty, D ∩ T = empty, T ∩ P = empty.

---

## ET Python Library (Primary Software Project)

**Location:** `Software/ET Python Library/Current Build/`
**Version:** 3.10.0
**Language:** Python 3.7+
**Dependencies:** None (pure Python, zero external deps)
**Install:** `pip install -e .` from `Current Build/`

### Package Architecture

```
exception_theory/              # Main package
├── __init__.py                # v3.10.0, exports 218+ classes
├── core/                      # Foundation layer
│   ├── constants.py           # All constants (BASE, BATCH1-22, PLATFORM)
│   ├── mathematics.py         # ETMathV2: 52+ static methods
│   ├── mathematics_descriptor.py  # Descriptor-specific math
│   ├── mathematics_quantum.py     # Quantum mechanics (largest module, ~123KB)
│   ├── mathematics_gr.py         # General Relativity math
│   └── primitives.py            # Point, Descriptor, Traverser, Exception classes
├── classes/                   # 22 feature batches
│   ├── __init__.py            # Batch registry & imports
│   ├── batch1.py              # Computational ET (TraverserEntropy, TrinaryState, etc.)
│   ├── batch2.py              # Manifold Architectures (TeleologicalSorter, Bloom, etc.)
│   ├── batch3.py              # Distributed Consciousness (SwarmConsensus, etc.)
│   ├── batch4.py              # Quantum Mechanics Foundations
│   ├── batch5.py              # Electromagnetism
│   ├── batch6.py              # Hydrogen Atom Core
│   ├── batch7.py              # Spectroscopy
│   ├── batch8.py              # Fine Structure & Corrections
│   ├── batch9.py              # General Relativity & Cosmology
│   ├── batch10.py             # P-D Duality & Ultimate Sets
│   ├── batch11.py             # Manifold Dynamics & Substantiation
│   ├── batch12.py             # Harmonic Generation & Set Cardinalities
│   ├── batch13.py             # Signal Processing & Foundational Axioms
│   ├── batch14.py             # Primitive Disjointness Theory
│   ├── batch15.py             # Universe Completeness & Exception Properties
│   ├── batch16.py             # Point Primitive Foundations
│   ├── batch17.py             # Deep Point Extraction: Identity
│   ├── batch18.py             # Deep Point Extraction: State
│   ├── batch19.py             # Deep Point Extraction: Structure
│   ├── batch20.py             # Descriptor Primitive Foundations: Nature
│   ├── batch21.py             # Descriptor Gap Theory
│   └── batch22.py             # Descriptor Advanced Principles
├── engine/
│   └── sovereign.py           # ETSovereign: unified API engine (4,517 lines, 101 methods)
└── utils/
    ├── calibration.py         # ETBeaconField, ETContainerTraverser
    └── logging.py             # Logger configuration
```

### Dependency Flow (Strict - Never Circular)

```
constants.py  (no imports)
     ↓
mathematics.py  (imports constants)
     ↓
primitives.py  (imports constants, mathematics)
     ↓
batch1-22.py  (imports constants, mathematics)
     ↓
sovereign.py  (imports ALL)
     ↓
__init__.py  (exports ALL)
```

**Rule:** Modules only import from parents (upward in the chain). Never import from siblings or children.

### Key API Classes

**ETMathV2** - All ET mathematical operations as static methods:
```python
ETMathV2.density(payload, container)    # Eq 211: Structural Density
ETMathV2.manifold_depth(n)              # Eq 212: Manifold Depth
ETMathV2.binding_energy(d_count, base)  # Eq 213: Binding Energy
# ... 52+ methods total
```

**ETSovereign** - Unified engine with 3-method pattern per feature:
```python
engine = ETSovereign()
engine.create_<feature>(name, **params)    # Create instance
engine.get_<feature>(name)                 # Retrieve instance
engine.direct_<feature>_operation(**args)  # Stateless operation
engine.close()                             # Cleanup all resources
```

**Primitives** - Core ET data classes:
```python
Point(location, state=None, descriptors=None)
Descriptor(name, value=None, domain=None)
Traverser(name, position=None, descriptors=None)
Exception(point, descriptor, traverser)
```

### Batch Titles (for reference)

| Batch | Title | Theme |
|-------|-------|-------|
| 1 | The Code of Reality | Computational ET |
| 2 | Code of the Impossible | Manifold Architectures |
| 3 | The Code of Connection | Distributed Consciousness |
| 4 | The Code of the Atom | Quantum Mechanics Foundations |
| 5 | The Code of Forces | Electromagnetism |
| 6 | The Code of Matter | Hydrogen Atom Core |
| 7 | The Code of Light | Spectroscopy |
| 8 | The Code of Precision | Fine Structure & Corrections |
| 9 | The Code of Spacetime | General Relativity & Cosmology |
| 10 | The Code of Duality | P-D Duality & Ultimate Sets |
| 11 | The Code of Process | Manifold Dynamics & Substantiation |
| 12 | The Code of Resonance | Harmonic Generation & Set Cardinalities |
| 13 | The Code of Synthesis | Signal Processing & Foundational Axioms |
| 14 | The Code of Separation | Primitive Disjointness Theory |
| 15 | The Code of Totality | Universe Completeness & Exception Properties |
| 16 | Point Primitive Foundations | Point axioms & properties |
| 17 | Deep Point Extraction | Identity |
| 18 | Deep Point Extraction | State |
| 19 | Deep Point Extraction | Structure |
| 20 | Descriptor Foundations | Nature |
| 21 | Descriptor Gap Theory | Gaps & Discovery |
| 22 | Descriptor Advanced Principles | Advanced D properties |

### Adding a New Batch (23+)

Follow this exact checklist:

1. **constants.py** - Add new constants under a `# BATCH 23` section header
2. **mathematics.py** - Add new math methods to `ETMathV2` if needed
3. **classes/batch23.py** - Create with ~10 classes following existing pattern:
   - Each class docstring: `"""Batch 23, Eq X: Description\n\nET Math: formula"""`
   - Import from `..core.constants` and `..core.mathematics`
4. **classes/__init__.py** - Add `from .batch23 import (...)` block
5. **engine/sovereign.py** - In `ETSovereign.__init__()`: add registry entries
6. **engine/sovereign.py** - Add `create_<feature>()`, `get_<feature>()`, `direct_<feature>_operation()` methods
7. **engine/sovereign.py** - Add cleanup in `close()` method
8. **__init__.py** (package root) - Add new class names to imports and `__all__`
9. Update `__version__` string

### Size Limits

- `constants.py`: < 300 lines
- `mathematics.py`: < 1200 lines
- Any `batchN.py`: < 1000 lines (split to new batch if approaching)
- `sovereign.py`: Monitor growth (currently 4,517 lines)

### Code Conventions

**Constants:** `SCREAMING_SNAKE_CASE` (e.g., `BASE_VARIANCE`, `MANIFOLD_SYMMETRY`)
**Classes:** `PascalCase` (e.g., `TraverserEntropy`, `SwarmConsensus`)
**Methods:** `snake_case` (e.g., `create_trinary_state`, `direct_entropy_operation`)
**Docstrings:** Include batch number, equation number, ET Math formula, Args, Returns

### Testing

```bash
pytest tests/                          # Run all tests
pytest --cov=exception_theory tests/   # With coverage
pytest tests/test_batch1.py            # Single batch
```

Dev dependencies (optional): `pip install -e ".[dev]"` for pytest, black, flake8, mypy

---

## ET Programming Language (ETPL)

**Location:** `Software/ET Programming Language/`
**Entry:** `python ETPL.py <command>`
**Version:** 1.0.0

A complete programming language toolchain derived from ET principles.

### Commands

```bash
python ETPL.py interpret <file.pdt>         # Interpret ETPL source
python ETPL.py compile <file.pdt> [output]  # Compile to binary
python ETPL.py translate <file.py> [lang]   # Translate Python to ETPL
python ETPL.py verify                       # Run self-verification
python ETPL.py repl                         # Interactive REPL
```

### Features

- Custom syntax with ET symbols (P, D, T, ∘, lambda, psi, infinity)
- Parse -> Interpret -> Compile pipeline
- Python translation (bidirectional)
- Binary reverse engineering
- Quantum circuit compilation (QASM output)
- Self-hosting capability

### IDE

- `etpl_ide_launcher.py` - Full IDE combining stages 1 + 2
- `etpl_ide_stage1.py` - Base framework (PyQt5)
- `etpl_ide_stage2_enhancements.py` - Advanced features

**IDE Dependencies:** PyQt5, llvmlite (optional), capstone (optional), pefile (optional), psutil (optional)

---

## Astronomy Suite

**Location:** `Software/Astronomy/`

Two parallel implementations:

### ET-Universal-Astronomer
```
ET-Universal-Astronomer/
├── main.py          # Entry point
├── et_app.py        # Application layer
├── et_core.py       # Core astronomy engine
├── et_math.py       # Mathematics module
├── et_manifold.py   # Manifold analysis
├── et_fits.py       # FITS format handling
├── et_scanner.py    # FITS file scanner
├── et_ingestion.py  # Data import
└── et_export.py     # Data export
```

### Astrogazer Modular v7.5
Same module structure as Universal-Astronomer, modular architecture.

### Capabilities
- FITS file parsing, scanning, and analysis
- Raw astronomical data ingestion and conversion
- Manifold analysis of stellar data
- Multi-format data export

### Data Directories
- `FITS Files/` - Processed data
- `Raw FITS Files/` - Raw data

### Standalone Utilities
- `convert_to_raw.py` / `convert_to_raw_multi.py` - Format conversion
- `diagnose_fits.py` - FITS file diagnostics

---

## ET Sovereign (Standalone)

**Location:** `Software/Sovereign/`
**Latest:** `ET_Sovereign_v2_3_COMPLETE.py`

Low-level Python metamorphic engine providing:
- Kernel-level memory access via ctypes
- Python object structure introspection
- Read-only memory transmutation (RO bypass)
- Copy-on-write protection bypass
- Reference graph traversal
- String/bytes/bytearray transmutation
- Function hot-swapping and bytecode replacement
- Type metamorphosis
- Executable memory allocation
- Thread-safe and GC-safe operations

Note: The library version in `engine/sovereign.py` integrates these capabilities plus all 22 batches.

**Documentation:**
- `ET_Sovereign_Complete_Integration_Guide.md`
- `Python_Unleashed_Complete_Language_Unification_Guide.md`
- Version changelogs for v2.0 through v2.3

---

## Scanner Suite

**Location:** `Software/Scanner/`
**Latest:** `et_scanner_v7_2_COMPLETE.py`

Multi-version scanning utilities (v6.1 through v7.2) for data analysis. Results stored in `Results/`.

---

## Web Archiver

**Location:** `Software/Web Archiver/`
**Latest:** `et_web_archiver_FIXED_COMPLETE.py`

Website scraping and archiving tool. Version history from initial `webpage_downloader.py` through v4.0, v4.5, to current fixed complete version.

---

## Other Tools

| Tool | Location | Description |
|------|----------|-------------|
| ET Craft | `Software/ET Craft/ET_MC.py` | Minecraft integration |
| Physics Toolbox | `Software/ET-Physics-Toolbox/server.py` | Physics calculation server |
| Periodic Table | `Software/Periodic Table Omniverse/` | Extended periodic table |
| ET Ruler | `Software/ET Ruler/` | Measurement utilities |
| Markdown Viewer | `Software/Markdown Viewer/markdown_viewer.py` | Markdown visualization |

---

## Eternal Memory Project (EMP)

**Location:** `Eternal Memory/`
**Documentation:** `emp-notes-continuing.md`
**Type:** Phenomenological Knowledge Organization System

### 11-Category Classification System

| # | Category | Scope |
|---|----------|-------|
| 1 | People | Sentient/conscient beings (real and fictional) |
| 2 | Unknown | Knowledge boundaries, unsolved problems |
| 3 | Belief | Faith, philosophy, ethics, moral frameworks |
| 4 | Commerce | Trade, business, economic systems |
| 5 | History | Past events, developments, interpretations |
| 6 | Ideas | Abstract thought frameworks, theories |
| 7 | Mathematics | Numbers, patterns, formal systems |
| 8 | Science | Systematic evidence-based knowledge |
| 9 | Socialization | Interaction, community, communication |
| 10 | Stimulation | Experiences evoking strong responses (entertainment, art, war) |
| 11 | Items | Physical and conceptual objects (singular entries) |

### Core Principles
- **Phenomenological** organization (by human experience, not academic discipline)
- **"All data is good data"** - Everything accepted, meticulously labeled
- **Evidence triumphs all subjective views**
- **Preserve over delete** - Even false evidence is labeled and kept
- **16-category evidence tagging system**
- **Dual AI systems** (Memory AI + analytical)
- **Intentional category overlaps** reflect knowledge's interconnected nature

---

## Political Philosophy: Ritaism & Peratocracy

**Location:** `Political Ideology/`

### Ritaism (The Philosophy)
- Root: Sanskrit rita (cosmic order, truth, natural law)
- Post-automation governance philosophy
- Recognizes natural order of human nature and social organization

### Peratocracy (The Government)
- Root: Greek peras (limit, boundary, completion)
- Governance through the correct limit
- Central concept: "Freedom Line" - single boundary creating order from chaos

### Framework Synthesis
Unifies elements from libertarian freedom, socialist baseline security, conservative justice/hierarchy, progressive structural analysis, technocratic expertise, and evidence-based empiricism.

---

## Analyzing Oneself

**Location:** `Analyzing Oneself/`

- `Cognitive.md` - Cognitive testing, intelligence mapping, processing analysis
- `Psychological.md` - Psychological framework and self-analysis documentation

---

## Key Relationships Between Components

```
Exception Theory (Foundation)
    ├── ET Python Library (Software implementation of ET math + features)
    │   └── ETSovereign engine (Unified API for all 218 classes)
    ├── ETPL (Programming language derived from ET principles)
    ├── ET Sovereign standalone (Low-level memory manipulation via ET)
    ├── Astronomy Suite (ET-based astronomical data analysis)
    ├── Scanner/Archiver/Tools (Specialized ET-informed utilities)
    │
    ├── Eternal Memory Project (Knowledge organization using ET categories)
    │
    └── Ritaism/Peratocracy (Political philosophy informed by ET principles)
```

Everything derives from and connects back to the Exception Theory framework. The mathematical primitives (P, D, T) and their binding relationships inform the software architecture, the knowledge classification, and the political philosophy.

---

## Development Guidelines

### When Modifying the ET Python Library
1. Read the existing module before making changes
2. Follow the strict dependency flow (constants -> math -> primitives -> batches -> sovereign -> __init__)
3. Never create circular imports
4. Keep modules within size limits
5. Use the 3-method pattern for ETSovereign features (create, get, direct)
6. All new constants go in `constants.py`, all new math in `mathematics.py`
7. Every class docstring must reference its Batch number and Equation number

### When Working with Theory Documents
- `ExceptionTheory.md` is the canonical source for the 20 foundational rules
- `ET Math Compendium.md` contains the mathematical operations
- `ET Programming Math Compendium.md` maps math to code
- Constant derivations follow a 4-stage process (Fine Structure -> Electron Mass -> Proton Mass -> Bare Quark Masses)

### When Working with ETPL
- ETPL is self-contained (all ET primitives inlined)
- External dependencies (llvmlite, capstone, pefile) are optional with graceful fallback
- Source files use `.pdt` extension

### When Working with Astronomy Modules
- Two parallel implementations exist; prefer ET-Universal-Astronomer for new work
- FITS files go in designated data directories
- Module structure mirrors the ET library pattern (core, math, app, export)

### When Working with Standalone Sovereign
- v2.3 is the latest complete version
- Includes all features from v2.0-v2.2 plus Batch 3 additions
- Uses ctypes for kernel-level access; requires understanding of CPython internals
- Safety verification systems are built in; do not bypass them

---

## Version History Summary

| Version | Milestone |
|---------|-----------|
| v2.0 | Core transmutation, RO bypass |
| v2.1 | Batch 1: Computational ET |
| v2.2 | Batch 2: Manifold Architectures |
| v2.3 | Batch 3: Distributed Consciousness |
| v3.1 | Batches 4-8: Quantum Mechanics & Hydrogen Atom |
| v3.2 | Batch 9: General Relativity & Cosmology |
| v3.3-v3.5 | Batches 10-12: P-D Duality, Manifold Dynamics, Harmonics |
| v3.7 | Batches 13-15: Axioms, Disjointness, Universe Completeness |
| v3.8 | Batch 16: Point Primitive Foundations |
| v3.9 | Batches 17-19: Deep Point Extraction |
| v3.10.0 | Batches 20-22: Descriptor Primitive Foundations (current) |

---

## Quick Reference: File Locations

| Need | File |
|------|------|
| ET foundational rules | `Exception Theory/ExceptionTheory.md` |
| ET math reference | `Exception Theory/ET Math Compendium.md` |
| Python library root | `Software/ET Python Library/Current Build/exception_theory/__init__.py` |
| All constants | `Software/ET Python Library/Current Build/exception_theory/core/constants.py` |
| All math methods | `Software/ET Python Library/Current Build/exception_theory/core/mathematics.py` |
| Quantum math | `Software/ET Python Library/Current Build/exception_theory/core/mathematics_quantum.py` |
| GR math | `Software/ET Python Library/Current Build/exception_theory/core/mathematics_gr.py` |
| Primitives (P/D/T/E) | `Software/ET Python Library/Current Build/exception_theory/core/primitives.py` |
| Sovereign engine | `Software/ET Python Library/Current Build/exception_theory/engine/sovereign.py` |
| Batch classes | `Software/ET Python Library/Current Build/exception_theory/classes/batch1-22.py` |
| ETPL language | `Software/ET Programming Language/ETPL.py` |
| ETPL IDE | `Software/ET Programming Language/etpl_ide_launcher.py` |
| Standalone Sovereign | `Software/Sovereign/ET_Sovereign_v2_3_COMPLETE.py` |
| Astronomy entry | `Software/Astronomy/ET-Universal-Astronomer/main.py` |
| Scanner latest | `Software/Scanner/et_scanner_v7_2_COMPLETE.py` |
| Web archiver | `Software/Web Archiver/et_web_archiver_FIXED_COMPLETE.py` |
| EMP documentation | `Eternal Memory/emp-notes-continuing.md` |
| Ritaism | `Political Ideology/Ritaism.md` |
| Peratocracy | `Political Ideology/Peratocracy.md` |
| Setup/install | `Software/ET Python Library/Current Build/setup.py` |

---

## Statistics

- **Total library classes exported:** 218+
- **ETMathV2 static methods:** 52+
- **ETSovereign methods:** 101
- **Implemented batches:** 22
- **Equations implemented:** 230+
- **Library lines of code:** ~28,000
- **Documentation files:** 100+
- **External dependencies (core library):** 0

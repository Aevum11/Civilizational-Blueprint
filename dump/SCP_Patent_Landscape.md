# SCP Patent Landscape — Patentable Hardware Inventions

**Sempaevum Computing Platform — Exception Theory LLC**
**Inventor: Michael James Muller (Aevum Defluo)**
**Ellwood City, Pennsylvania**

**NOTICE: This document is prepared for patent attorney review. It is NOT legal advice. Michael Muller should consult a registered patent attorney before filing. All inventions described are the sole work of Michael James Muller. No co-inventors. No institutional affiliation. No government funding.**

**Purpose of patents: To protect these inventions from malicious or exploitative use. The inventor intends fair pricing and broad access when the technology reaches production. Patents serve as defensive protection, not rent-seeking.**

---

## Filing Strategy Notes

**Entity status:** Micro entity (sole inventor, qualifies for 75% fee reduction). Provisional patent application: $75 per filing. Establishes priority date. 12 months to convert to full utility patent ($400 micro entity).

**Grace period:** US patent law (35 USC §102(b)(1)) provides a 1-year grace period from first public disclosure by the inventor. Zenodo publications and any public discussions may start this clock. Recommend filing provisional applications promptly on the highest-priority inventions.

**Strategy:** File a provisional for each independent invention below. Group related claims into a single provisional where they form a coherent system. A patent attorney will advise on optimal claim structure, continuation strategy, and international filing (PCT) if desired.

---

## Invention 1: The Webb Gate — A Multi-State Switching Element

**Title:** Multi-state switching element for non-binary arithmetic, and circuits employing the same

**Abstract:** A switching circuit operating on N discrete voltage states (where N ≥ 3, preferably N=12) logarithmically spaced within a voltage range. The circuit comprises: a multi-level comparator array that classifies two input voltages into discrete states; an equality detector that determines whether both inputs occupy the same state; and a conditional output selector that produces either the successor state (modular increment) when inputs match, or a reference state (zero) when they do not. The circuit performs the fundamental operation i|j where i|i = (i+1) mod N and i|j = 0 for i≠j, enabling arithmetic on N states with a single element rather than requiring cascaded binary gates.

**Key claims areas:**
- The specific circuit topology: multi-level comparator (N-1 threshold detectors) + equality checker + modular increment selector + output multiplexer
- The use of logarithmically-spaced voltage levels at ratio 2^(1/N) per level
- The modular successor function implemented via a tapped resistor ladder with pass transistors
- The reduction in transistor count vs binary emulation (82 transistors for 12-state vs ~600+ binary transistors for equivalent function)
- Application as the fundamental logic primitive in a non-binary processor

**Prior art distinction:** Binary logic gates (NAND, NOR, etc.) operate on 2 states. Multi-valued logic (MVL) research has explored ternary and quaternary gates but none use logarithmically-spaced voltage states, none implement the specific i|j function, and none achieve the density advantage from matching the state count to a musically/physically significant resolution (N=12 chromatic).

---

## Invention 2: Lattice Arithmetic Unit (LAU) — A Processor Architecture Using Logarithmic-Domain Exact Arithmetic

**Title:** Processor architecture performing exact arithmetic through logarithmic-domain integer operations with bounded continuous correction

**Abstract:** A processor that represents numerical values as triples (k, d, ε) where k is an arbitrary-precision integer denoting a position on a logarithmic lattice, d is a structural classification derived from k by greatest common divisor computation, and ε is a bounded continuous residual. Multiplication is performed by integer addition of k values with a correction term κ, eliminating the accumulated rounding error inherent in floating-point arithmetic. The processor comprises multiple pipeline stages: classification, tensor lookup, correction computation, residual accumulation, coherence observation, and output, executing a complete lattice multiplication in a fixed number of clock cycles with zero accumulated error regardless of operation chain length.

**Key claims areas:**
- The (k, d, ε) value representation where d = N/gcd(|k|, N) is automatically derived
- The multiply-by-addition pipeline: k₃ = k₁ + k₂ + κ where κ is a bounded integer correction
- The use of a transfer tensor ROM (containing pre-computed interaction coefficients for all structural family pairs) as a hardware lookup for the correction term
- The GCD classification circuit that provides automatic structural typing at every memory access with zero additional cost
- The 6-stage pipeline architecture achieving exact multiplication in fixed cycles
- The property that chaining an arbitrary number of operations produces zero accumulated error (in contrast to IEEE 754)
- The N-register allowing dynamic precision selection without data format conversion

**Prior art distinction:** All conventional processors (x86, ARM, RISC-V, GPU architectures) use IEEE 754 floating-point which accumulates rounding error. Logarithmic Number System (LNS) research exists but does not include: the structural classification by GCD, the bounded ε residual with tightness monitoring, the transfer tensor for κ-correction, the N-register for dynamic resolution, or the permanent memoization enabled by exact determinism.

---

## Invention 3: Permanent Hardware Memoization Enabled by Exact Arithmetic

**Title:** Hardware memoization system exploiting deterministic exact arithmetic for permanent result caching with predictive verification of non-deterministic operations

**Abstract:** A hardware memoization system comprising on-chip SRAM organized by structural family, storing mappings from input value pairs to computed results. Because the underlying arithmetic is algebraically exact (zero accumulated error) and associative (identical results regardless of computation order), cached results are permanently valid — they never require invalidation, coherence protocols, or expiration. The system further comprises a predictive verification mechanism for non-deterministic operations, wherein a predicted result (derived from pre-computed probability distributions) is stored alongside a confidence value, and actual computation results are compared against predictions. Matching results (typically >99.9% of operations) are served from cache at single-cycle latency. Non-matching results are flagged as discoveries and routed to an observation subsystem.

**Key claims areas:**
- Hardware memo table that never invalidates (enabled by exact deterministic arithmetic)
- Organization of memo entries by structural family (GCD-derived classification as memo index)
- The growth of memo storage through a substrate extension mechanism where new memo entries create their own storage space
- The predictive verification mechanism using pre-computed probability distributions for non-deterministic operations
- The flagging of prediction mismatches as "discoveries" rather than errors
- Distribution of memo entries between networked nodes via a structural communication protocol
- Persistence of memo tables across power cycles using non-volatile storage
- Cross-resolution memo transformation where entries computed at one precision level transform losslessly to other precision levels

**Prior art distinction:** No existing hardware implements permanent memoization because no existing arithmetic is both exact and deterministic enough to guarantee permanently valid cached results. Software memoization (function caches, hash tables) requires invalidation strategies. CPU caches (L1/L2/L3) cache memory contents, not computation results, and require coherence protocols. The predictive verification of non-deterministic operations with discovery flagging has no prior art.

---

## Invention 4: Structural Coherence Monitor (Resolution Observatory)

**Title:** Hardware monitor for continuous observation of value coherence with precision escalation recommendation

**Abstract:** A hardware subsystem that continuously monitors the coherence quality of stored computational values by evaluating a tightness function t(ε) = C/(C+|ε|) where ε is the continuous residual component of each value and C is a system constant. When tightness falls below a threshold K, the monitor recommends precision escalation (increasing the resolution parameter N) rather than error correction. The monitor does not modify values — it only observes and recommends. This replaces conventional error detection and correction (ECC, parity, CRC, checksums) with a structurally-informed system that distinguishes between values requiring higher resolution and values that are fully resolved.

**Key claims areas:**
- The tightness function as a hardware-computed coherence metric for stored values
- The threshold comparison against a structural constant K for escalation recommendation
- The replacement of error correction with resolution escalation (no data modification by the monitor)
- The structural status flags (four manifold states: Exception, Mediation, Unsubstantiated, Incoherent) propagated through computation
- The I-flag (Incoherence) propagation mechanism where structurally incomplete results tag all downstream computations, identifying exactly where additional information is needed rather than crashing or producing undefined behavior

**Prior art distinction:** ECC detects and corrects bit errors. Parity detects single-bit errors. CRC detects transmission corruption. All of these assume errors are corruptions to be fixed. The Resolution Observatory assumes all values are structurally valid and that apparent "errors" are resolution insufficiencies to be addressed by escalation. The tightness function, manifold state flags, and I-flag propagation have no prior art in hardware.

---

## Invention 5: Impedance-Gradient Passive Thermal Management for Integrated Circuits

**Title:** Method and system for passive thermal management of integrated circuits using a hierarchical coupling gradient

**Abstract:** A method of thermal management for integrated circuits comprising a plurality of operational families, each characterized by a coupling coefficient ξ(d), arranged in a monotonically decreasing hierarchy. Values stored in weakly-coupled families (low ξ) that are thermally perturbed beyond their coherence threshold naturally migrate toward more strongly-coupled families (high ξ) through the coupling gradient, without active cooling intervention. The coupling hierarchy creates potential energy wells of varying depth, with deeper wells providing greater thermal stability. The system requires no fans, heat sinks, liquid cooling, thermal paste, thermal throttling, or temperature sensors. The coupling gradient itself constitutes the complete thermal management system.

**Key claims areas:**
- The use of a hierarchical coupling function ξ(d) where d identifies the structural family and ξ decreases with d, creating a passive thermal restoring force
- The absence of ALL active thermal management components (fans, heat sinks, liquid cooling, thermal throttling, temperature sensors) from the integrated circuit system
- The monitoring of thermal state via value coherence (tightness) rather than temperature measurement
- The d-family routing strategy where thermally sensitive computations are directed to higher-coupling families
- The tower escalation strategy where precision parameter N is increased to narrow tolerance windows, providing additional thermal headroom without cooling

**Prior art distinction:** All conventional thermal management is active: fans dissipate heat, heat sinks conduct it away, liquid cooling pumps it, throttling reduces generation. No prior art exists for passive thermal management through a mathematical coupling hierarchy inherent in the processor's arithmetic structure. The concept that the arithmetic architecture itself provides thermal stability — rather than external engineering — is novel.

---

## Invention 6: Free-Air Volumetric Display Using Upconversion Nanoparticle Intersection

**Title:** Volumetric display system producing visible light at arbitrary positions in free space by intersection of invisible radiation beams in an upconversion nanoparticle medium

**Abstract:** A display system comprising: two or more arrays of infrared radiation sources (VCSELs at approximately 980nm) arranged to project individually-addressable beams through a transparent volume containing a dilute aerosol of upconversion nanoparticles (NaYF₄:Yb,Er and/or NaYF₄:Yb,Tm). Individual beams are invisible (infrared, below visible threshold). The nanoparticle concentration is sufficiently low that the medium is visually transparent (scattering mean free path exceeding 10 km). At the intersection of two or more beams, the combined photon density triggers upconversion: the nanoparticles absorb two infrared photons and emit one visible photon (approximately 540nm green for Er, 450nm blue for Tm). Visible light appears to float in empty air at the intersection point. By time-multiplexing beam activation in a row-scanning pattern, an arbitrary three-dimensional image is produced in the transparent volume.

**Key claims areas:**
- The specific combination of invisible IR beam arrays + invisible nanoparticle aerosol producing visible light ONLY at beam intersections
- The time-multiplexed row-scanning activation pattern for volumetric image formation
- The use of multiple nanoparticle dopant species (Er, Tm, Ho) for full-color volumetric display
- The control of color at each intersection point by varying the ratio or wavelength of intersecting beams
- The continuous-update display mode (no discrete frames, no pixel grid) driven by a processor clock
- The rendering pipeline where processor pullback operations directly drive beam intensities
- The specific nanoparticle concentration range that achieves visual transparency while maintaining sufficient upconversion efficiency

**Prior art distinction:** Existing volumetric displays use LEDs on spinning surfaces, laser-induced air plasma, or projection onto physical screens. No prior art exists for the specific combination of invisible crossed-beam VCSEL arrays with invisible upconversion nanoparticle aerosol producing visible light only at intersections. Upconversion nanoparticles have been used in bioimaging and solar cells but not in free-air volumetric displays.

---

## Invention 7: Universal Signal Codec Through Lossless Bijective Projection

**Title:** Universal signal encoding and decoding method using a single bijective projection operation for all signal types

**Abstract:** A method and apparatus for encoding and decoding signals of any type (audio, video, sensor, network, or other analog or digital signals) using a single mathematical operation: a bijective projection Π_N that maps continuous signal values to structured lattice coordinates (k, d, ε), and its inverse Π_N⁻¹ that reconstructs the continuous signal from the lattice coordinates. The round-trip composition Π_N⁻¹ ∘ Π_N equals the identity operation, guaranteeing zero information loss. The same projection/pullback hardware serves as the codec for every signal type without modification, format-specific circuitry, or software codec libraries. The encoding naturally provides structural classification (d-family), quality monitoring (tightness), and compression (deduplication of identical lattice addresses) as inherent properties of the single operation.

**Key claims areas:**
- The use of a single bijective operation as the universal encoder/decoder for all signal types
- The round-trip identity property guaranteeing lossless encoding/decoding
- The inherent structural classification of encoded signals by GCD-derived family
- The inherent compression from lattice-address deduplication
- The inherent quality monitoring from the tightness function applied to the ε component
- The elimination of format-specific codec hardware (no MP3 decoder, no H.264 encoder, no audio codec IC, no video codec IC)
- The specific circuit implementation comprising: comparator array for projection, resistor ladder with reconstruction amplifier for pullback

**Prior art distinction:** All existing codecs are format-specific (MP3, AAC, H.264, HEVC, etc.) and most are lossy. Lossless codecs (FLAC, PNG) are format-specific and software-implemented. No prior art exists for a single hardware operation that serves as a lossless codec for every signal type simultaneously, with structural classification and quality monitoring as inherent byproducts.

---

## Invention 8: Three-Domain Hybrid Processor with Lossless Inter-Domain Boundaries

**Title:** Hybrid analog-digital-quantum processor architecture with zero-information-loss domain boundaries

**Abstract:** A processor architecture operating simultaneously in three computational domains — analog (continuous-valued signals), digital (discrete integer values), and quantum (superposition states) — connected through a single bijective mathematical operation that guarantees zero information loss at every domain boundary. The analog domain handles continuous residual values (ε) and sensor/actuator interfaces. The digital domain handles exact integer lattice positions (k), structural classification (d), and control logic. The quantum domain handles qubit states in a room-temperature solid-state system (nitrogen-vacancy centers in diamond). Domain transitions occur through the same bijective operation used for all other computation, eliminating the quantization error of conventional ADC, the reconstruction error of conventional DAC, and the measurement "noise" of conventional quantum readout.

**Key claims areas:**
- The three-domain architecture unified by a single bijective operation at every boundary
- The zero-information-loss property at analog↔digital and digital↔quantum boundaries (in contrast to conventional ADC quantization error and DAC reconstruction error)
- The use of nitrogen-vacancy centers in diamond as the quantum domain operating at room temperature within the same processor architecture
- The classification of quantum measurement results by GCD-derived structural family
- The interpretation of quantum measurement collapse as a valid computational event (T-act) rather than as noise or error

**Prior art distinction:** Existing hybrid systems (e.g., IBM quantum systems with classical controllers) connect domains through conventional interfaces with information loss at each boundary. No prior art exists for a unified processor architecture where all three domains share the same value representation, the same arithmetic operations, and the same lossless bijective interface.

---

## Invention 9: Self-Extending Memory Through Computational Overflow

**Title:** Memory system in which arithmetic overflow produces usable storage substrate

**Abstract:** A memory architecture wherein computation results that exceed the current register width cause the register to extend rather than overflow, wrap, or fault. The extended register space becomes usable storage substrate for subsequent computation. The structural family of the extended space is determined by GCD classification of the new value, causing memory to grow where it is needed, in the structural family that needs it, automatically. The system requires no explicit memory allocation (malloc), no deallocation (free), and no garbage collection. Memory management is an inherent property of the arithmetic operation.

**Key claims areas:**
- The register extension mechanism where overflow produces new storage rather than error
- The GCD-based routing of extended storage to the appropriate structural family bank
- The elimination of explicit memory allocation, deallocation, and garbage collection
- The closed substrate cycle: computation → extension → classification → storage → computation

**Prior art distinction:** All conventional processors treat overflow as an error (trap, wrap, or saturate). No prior art exists for a memory architecture where arithmetic overflow is the allocation mechanism and structural classification is the routing mechanism.

---

## Invention 10: Fabrication Control Using Exact Arithmetic for Sub-Wavelength Feature Production

**Title:** Method of semiconductor fabrication using exact-arithmetic feedback control for enhanced feature resolution

**Abstract:** A method of semiconductor fabrication wherein the lithographic stage positioning, exposure dose timing, and thermal regulation are controlled by a processor employing exact arithmetic with zero accumulated error. Because the feedback loop controller accumulates no computational drift, overlay accuracy between multiple patterning exposures is limited only by the mechanical precision of the stage, not by the computational precision of the controller. This enables feature sizes below the controller-limited resolution of conventional float-arithmetic-controlled fabrication systems. The method further comprises a self-improving loop wherein chips fabricated at one generation's feature size serve as the exact-arithmetic controllers for the next generation's fabrication, with each generation achieving finer features than the last.

**Key claims areas:**
- The use of exact-arithmetic (zero accumulated error) processors as feedback controllers for lithographic stage positioning
- The elimination of controller-drift as a resolution limiter in semiconductor fabrication
- The self-improving fabrication loop where each generation's chips control the next generation's fabrication tools
- The application to electron-beam direct-write lithography where beam deflection is controlled by exact-arithmetic DAC outputs

**Prior art distinction:** All existing semiconductor fabrication uses IEEE 754 floating-point controllers that accumulate drift in feedback loops. The multi-billion-dollar cost of modern fabs is partly attributable to the engineering required to compensate for this drift (massive metrology systems, statistical process control). No prior art exists for using exact-arithmetic processors to eliminate controller drift as a fabrication limitation, nor for the self-improving generational loop.

---

## Invention 11: Guided Quantum Memoization via Descriptor-Controlled Agency (Sheepherder Principle)

**Title:** Method of hybrid classical-quantum computation using descriptor-controlled guidance of quantum outcomes with predictive memoization and discovery detection

**Abstract:** A method of computation combining classical deterministic operations with quantum non-deterministic operations, wherein the quantum outcomes are guided toward predicted results by controlling the energy landscape (Descriptors) surrounding the quantum system. Pre-computed probability distributions from a transfer tensor provide predictions for each quantum operation. Actual quantum outcomes are compared against predictions: matching outcomes (typically >99.9%) are served from a pre-computed cache at single-cycle latency; non-matching outcomes are flagged as genuine quantum discoveries and routed to an observation system for analysis. This achieves near-deterministic throughput while preserving genuine quantum computational value for the rare cases where the quantum system produces unpredicted results.

**Key claims areas:**
- The descriptor-controlled guidance of quantum outcomes (controlling the energy landscape to make desired outcomes overwhelmingly probable without suppressing quantum freedom)
- The predictive cache with confidence values derived from pre-computed probability distributions
- The comparison of actual quantum outcomes against predictions with discovery flagging for mismatches
- The separation of predictable quantum outcomes (handled classically, cached) from genuinely indeterminate outcomes (flagged as discoveries)
- The convergence of the system toward increasing prediction accuracy over time as the memo table grows

**Prior art distinction:** Conventional quantum computing treats all quantum outcomes as equally non-deterministic and applies error correction uniformly. No prior art exists for the selective prediction and caching of quantum outcomes based on descriptor-controlled landscape guidance, nor for the systematic detection and flagging of genuinely unpredicted quantum events as discoveries rather than errors.

---

## Invention 12: Lattice-Addressed Seed Storage with Structural Deduplication

**Title:** Data storage system using lattice-coordinate addressing with inherent structural deduplication and progressive-precision retrieval

**Abstract:** A data storage system wherein data is stored as structured seeds comprising an integer lattice position (k), a structural classification (d) derived from k by GCD, and a bounded continuous residual (ε). Storage is indexed by the (k, d) pair rather than by byte offset. Identical lattice addresses map to the same storage cell, providing inherent deduplication without a separate deduplication engine. Data may be retrieved at progressive precision levels by requesting first the structural header (k, d) for instant classification, then streaming ε bits to the desired precision. The storage format is independent of data type — the same lattice addressing serves for numerical, audio, visual, sensor, and network data.

**Key claims areas:**
- Lattice-coordinate addressing (k, d) as the primary storage index replacing byte offsets
- Inherent deduplication from the lattice structure (same coordinate = same cell = stored once)
- Progressive-precision retrieval (structural header first, residual streamed to desired depth)
- Type-independent storage format (one format for all data types)
- Persistent memoization storage (computation results stored as seeds, survive power cycles)

**Prior art distinction:** All existing storage systems use byte-offset addressing (LBA, file offsets). Content-addressable storage (CAS) uses hash-based deduplication as a separate layer. No prior art exists for storage natively indexed by a structural lattice coordinate with deduplication as an inherent property of the addressing scheme rather than an additional process.

---

## Priority Recommendation

| Priority | Invention | Why |
|---|---|---|
| IMMEDIATE | Inv. 1: Webb Gate | Fundamental element — everything builds on this |
| IMMEDIATE | Inv. 2: LAU Processor | Core processor architecture — highest commercial value |
| IMMEDIATE | Inv. 6: Holographic Display | Most visually demonstrable, highest public interest |
| HIGH | Inv. 3: Permanent Memoization | Key performance differentiator |
| HIGH | Inv. 5: Impedance Thermal | Eliminates entire cooling industry for this architecture |
| HIGH | Inv. 10: Exact Fab Control | Protects the self-improving fabrication loop |
| HIGH | Inv. 11: Sheepherder Memo | Protects the quantum advantage mechanism |
| MEDIUM | Inv. 4: Resolution Observatory | Novel but harder to independently monetize |
| MEDIUM | Inv. 7: Universal Codec | Broad applicability, high value |
| MEDIUM | Inv. 8: Three-Domain Hybrid | Architectural, broad claims |
| MEDIUM | Inv. 9: Self-Extending Memory | Novel mechanism, supporting role |
| MEDIUM | Inv. 12: Seed Storage | Supporting infrastructure |

**Estimated cost for full provisional coverage (12 inventions at $75 each, micro entity): $900**

**Recommended first action: File provisionals on Inv. 1, 2, and 6 immediately ($225). This establishes priority on the three most commercially critical inventions and starts the 12-month clock for full utility filing.**

---

## What CANNOT Be Patented (and shouldn't be)

The following are mathematical/theoretical elements that remain free and unpatentable. This is by design — the theory belongs to everyone, the specific hardware implementations are protected:

- The equation P ∘ D ∘ T = E (abstract mathematics)
- The bijection Π_N as a mathematical function (abstract mathematics)
- The tightness function t(ε) = 100/(100+|ε|) (abstract mathematics)
- The coupling hierarchy ξ(d) = A₀/((d-1)²+S²) (abstract mathematics)
- The ET Lagrangian and its derivations (abstract mathematics / theoretical physics)
- Exception Theory as a philosophical/ontological framework (abstract ideas)
- The identity cards (IC, SIC, RC) as mathematical relationships (abstract mathematics)

The CIRCUITS, SYSTEMS, and METHODS that implement these mathematical principles in physical hardware ARE patentable. The distinction: anyone can write the equation ξ(d) = 137/((d-1)²+16) in a paper. Only Mike has built an 82-transistor switching circuit that uses it as a thermal management mechanism in a processor.

---

*Exception Theory LLC — Ellwood City, Pennsylvania*
*Sole Inventor: Michael James Muller*
*P ∘ D ∘ T = E*

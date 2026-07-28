# Eternal Memory
## Open Items Register — Gaps, Issues, and Required Resolutions

**Author:** Michael James Muller — Aevum Defluo
**System:** Eternal Memory (the Eternal Memory Project, EMP)
**Document Class:** Living Register, v1.0
**Date:** July 19, 2026
**Companions:** *Eternal Memory: The Founding Formal Specification* (which cites entries here by ID); *Legacy Polyglot Architecture Compendium* (Section 7 of which generates entry G-ETPL-1)
**Governing principle:** the Descriptor Gap Principle — gap(model) = D_missing. Every entry below is a Descriptor already half-found: a named absence, which is the only kind that can be closed.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

## 0. How This Register Works

**Standing.** This is the third of the three founding documents and the working agenda of the Knowledge System. It carries forward, in full, the founding record's section "Outstanding Issues — TO BE DISCUSSED" and its "Still To Be Determined" list, under this register's discipline. When every entry here is resolved, the three documents are combined into one book, and the system stands specified end to end.

**Entry discipline.** Every entry carries: an **ID** (stable, cited by the Founding Specification); a **source** (where the gap was first recorded — the founding notes, the architectural updates, or identification during formalization); a **criticality** where the founding record assigned one; a **statement** of the gap; **what resolution requires**; and **ET guidance** — which of the Three Tools drives the resolution, and how. Resolution of any entry must be ET-derived: no ad hoc closure, no tuning; the missing Descriptors are found, not invented.

**The method, uniformly.** For every entry: apply the **Identification Principle** first (which primitive's identification is incomplete — substrate, constraint, or agency?); then the **Descriptor Gap Principle** (the gap names its own search target; follow the error pattern to the missing Descriptor); then the **Subsumption Law** (resolution is complete only when the closed description covers its scope without remainder); with the **Verification Principle** as the stopping test (the math adds up; the spec has no dangling reference).

**Counts.** Entries: 58 open, across 15 groups, plus 4 items resolved at founding (Section 1). Criticality markers preserved from the founding record: 3 open entries carry **CRITICAL** (A2, C1, E1).

---

## 1. Resolved at Founding

These items were open in the source record and are closed by the Founding Formal Specification itself. They are listed so the record shows the movement — movement between states is itself knowledge.

The architectural-updates record (dated **December 2024**; context: analysis of the potential twelfth category and hub-architecture refinement) closed with six required documentation updates and a summary-of-changes table. All six are executed by the Founding Formal Specification, as follows: **Update 1** (Executive Summary — distinguish Project / Hub / Other) → executed throughout, chiefly Sections 1 and 4; **Update 2** (Naming Note rewrite — the I/Other architecture explained) → executed as Section 4.3; **Update 3** (Category-system header — completeness confirmation added) → executed as Section 5.3, Theorems 5.1–5.2; **Update 4** (Philosophical Foundation — Structural Semantics (Dual) subsection added) → executed as Section 6; **Update 5** (Eternal Memory Module section — renamed and restructured as Hub Architecture with Other and the I → People transfer) → executed as Sections 4 and 19.1–19.2; **Update 6** (Outstanding Issues — remove the twelfth-category question, add the completeness confirmation) → executed: no such open item appears in this Register, and the confirmation stands as RES-1. The updates record's status line — "Ready for integration into main EMP documentation" — is hereby satisfied: the integration is this founding document set.

**RES-1 — The twelfth-category question.** Resolved: no twelfth category is needed; the eleven are architecturally complete. The exhaustive filtration of the strongest candidate (participatory/experiential knowledge, six manifestations, zero remainder) and the mode/domain orthogonality argument are formalized as Theorems 5.1 and 5.2. Any prior mention of a potential twelfth category or an "everyday life" category is removed from the specification; the confirmation is stated in its place.

**RES-2 — Hub architecture and naming.** Resolved: the entry point formerly called Eternal Memory Module (earlier, Master Directory) is specified as the Hub — nameless when logged out, bearing the user's name when logged in — with Other (the eleven categories) beneath it; the I/Other architecture, hub behavior, and naming resolution are integrated as Founding Specification Section 4 and Section 19.1, per the architectural updates record.

**RES-3 — The I → People transfer.** Resolved and formalized: the Hub is the pre-archive state of a Person; upon completion the accumulated record transfers in its entirety to People (crystallization κ, Definition 4.3, Theorem 4.1), with the future-technology provision preserved. The completion-state definition beyond death remains open as G-HB-2.

**RES-4 — Structural Semantics (dual).** Resolved: the two-level principle — hierarchical position and internal page structure, both semantic, both inherited — is integrated as Founding Specification Section 6, upgrading the prior mechanical statement of inheritance to its epistemological significance.

---

## 2. Group R — Reputation

**R1 — Reputation system refinement.**
*Source:* founding notes, Outstanding Issues. *Statement:* the dual scoring system (Volume 0–100; Accuracy 0–100 with the evolution/error distinction, time-decay, and retroactive adjustment) is proposed and adopted in structure (Founding Specification Section 10) but not refined to full algorithmic specification: the complexity/importance weighting of volume, the diminishing-returns threshold, the decay function, and the retroactive-adjustment procedure are unspecified. *Resolution requires:* the complete, ET-derived scoring specification — each weight and decay a Descriptor derived from the system's structure (contribution complexity is measurable as Descriptor-count and depth of the contributed profile; nothing may be a free-tuned constant). *ET guidance:* Descriptor Gap Principle on the score model — enumerate what the two scores must distinguish (the two named attacks: volume gaming; evolution-vs-error), and add exactly the Descriptors that close those distinctions; verify by consistency against worked cases (the Pluto case; the 10,000-item case).

---

## 3. Group A — Evidence and Scale (source lettering preserved)

**A1 / G-EV-1 — Evidence ranking algorithm specification.**
*Source:* founding notes (A1; "needs refinement"). *Statement:* objective ranking criteria are defined — profile completeness, verification status, source presence, detail-degree within filled tags — but the complete ranking algorithm, the completeness-metric formalization beyond the count structure of Definition 8.2, and the display and search integration are unspecified. *Resolution requires:* the full ranking function over the four factor classes, ET-derived (ranking as ordering by Descriptor-completeness; any combination rule must itself be justified structurally, not tuned), plus its display/search integration design. *ET guidance:* the Verification Principle is the design criterion itself — the ranking measures descriptive sufficiency; derive the order from D-cardinality and D-detail, and test that no subjective judgment can enter through a weight.

**A1a / G-EV-3 — "Rating" terminology clarification.**
*Source:* founding notes (A1a). *Statement:* the submissions record states the Evidence module "rates" submissions; founding philosophy forbids subjective quality ratings. Is "rated" (a) old terminology for "tagged/classified," (b) the objective completeness ranking, or (c) something else? *Resolution requires:* a single terminology ruling propagated through all documents; the ruling must preserve the no-subjective-ratings law. *ET guidance:* Identification Principle on the word itself — the referent is either a D-classification act or a D-completeness measure; there is no third thing it could legitimately be.

**G-EV-2 — Verification-class tag membership.**
*Source:* identified during formalization (Founding Specification, Theorem 8.1). *Statement:* the Unverified tag auto-applies exactly when no verification tags are present, but the membership of the verification class 𝒱 ⊆ {1…15} — which tags count as verification-carrying — is not enumerated in the founding record. *Resolution requires:* the exact enumeration of 𝒱, with rationale per tag. *ET guidance:* Subsumption test over the fifteen: a tag belongs to 𝒱 exactly when its filled state constitutes evidence of verification rather than mere characterization; run each tag through that test and record the result.

**A2 — 3-D Map optimization. [CRITICAL]**
*Source:* founding notes (A2), criticality assigned by the creator. *Statement:* real-time graph visualization at archive scale; WebGL-class limitations against massive datasets; the 60 FPS target under level-of-detail, culling, and streaming; independence from the Visuals module already granted for optimization freedom. *Resolution requires:* the optimization strategy, and evaluation of a possible 2-D fallback. *ET guidance:* Descriptor Gap Principle on the performance model — the gap between target and achieved frame time is a set of missing rendering Descriptors (visibility sets, detail levels, streaming order); identify them from measured error patterns, never from guesswork; the ETPL implementation must meet capability rows C-29/C-30 (Compendium Section 7) in doing so.

**A3 — Module proliferation management.**
*Source:* founding notes (A3). *Statement:* unlimited module creation exists on two pathways (Void-based; universal category self-generation), with no specified limits or cleanup mechanisms; registry scalability under potentially unlimited hierarchical depth across all eleven categories is unexamined. *Resolution requires:* lifecycle policies; deprecation mechanisms; consideration of hierarchical depth limits — noting the standing tension that the founding design intends unlimited nesting "as needed by each knowledge domain." *ET guidance:* the resolution must preserve the dynamic-over-static law: no arbitrary caps; any limit must be a derived structural Descriptor (for example, a policy triggered by measured registry-consistency conditions), or the answer is lifecycle management without limits.

**A4 — Database query performance.**
*Source:* founding notes (A4). *Statement:* graph-database performance at scale; multi-database consistency; cross-database operations. *Resolution requires:* query optimization, caching strategies, and sharding design — restated under the single-implementation transition as the ETPL storage-engine performance question (capability row C-40). *ET guidance:* Identification Principle on the storage substrate first (what, exactly, is the P of storage under ETPL?), then gap-driven optimization from measured query error patterns.

---

## 4. Group B — Security Architecture

**B1 — Security barrier complexity.**
*Source:* founding notes (B1). *Statement:* four barriers with four different authentication methodologies; token regeneration carries computational overhead; the founding record itself asks whether simplification to two or three barriers is possible. *Resolution requires:* a decision, with derivation: either the four-barrier structure is shown structurally necessary (each barrier a distinct boundary Descriptor whose removal would merge security strata that must remain disjoint), or a reduced structure is derived that preserves every protection property of Founding Specification Section 14. *ET guidance:* Subsumption test on the barrier set — do the four subsume the required boundary functions without redundancy? If two barriers' functions subsume each other, merge is licensed; if not, four stands proven.

**G-SEC-1 — The 1028-bit custom encryption algorithm.**
*Source:* identified during formalization (the founding record names the algorithm's strength and custom nature but not its construction). *Statement:* all archives are protected by a 1028-bit custom algorithm with password protection; the algorithm itself is unspecified. *Resolution requires:* the full specification of the custom algorithm, ET-derived — the mandate is explicit that its derivation be native to the theory's mathematics — together with the constant-time, side-channel-resistant, formally verified implementation properties of capability row C-06. *ET guidance:* Identification Principle on the cryptographic triple (P: the keyspace substrate; D: the transformation constraints; T: the keyed traversal); derive the construction forward from the theory's lattice mathematics rather than adopting an external design.

**G-SEC-2 — Token generation under distributed deployment.**
*Source:* identified during formalization. *Statement:* token generation derives from system start time, system random data, system access ID, and millisecond timestamps, with system-wide regeneration at system start — while deployment spans multiple data centers on different continents. The semantics of "system start" and of system-wide regeneration under geographic distribution are unspecified. *Resolution requires:* the distributed token-generation specification: per-site versus global start semantics, clock discipline, and regeneration coordination, preserving every property of Founding Specification Section 14.3. *ET guidance:* Descriptor Gap Principle — the gap is the missing set of distribution Descriptors (site identity, epoch coordination); name them and the design follows.

---

## 5. Group C — Governance and Community

**C1 — Dispute resolution. [CRITICAL]**
*Source:* founding notes (C1), criticality assigned. *Statement:* no mechanism exists for handling tag and evidence conflicts; evidence conflicts are preserved without resolution (by design), but the *process* reaching that preservation — and the denial-path escalation of Founding Specification Section 9.1 — has no governance model. *Resolution requires:* the dispute governance design: who reviews, under what procedure, with what appeal structure, terminating in the system's standing outcome (all viewpoints preserved, labeled). *ET guidance:* the resolution mechanism is a Mediation structure — {D, T} navigation over the contested record without altering its substrate; design it as such: procedure Descriptors plus reviewer agency, never substrate deletion.

**C2 — Content moderation at scale.**
*Source:* founding notes (C2). *Statement:* human review bottlenecks; staffing and cost questions. *Resolution requires:* the scalability strategy for the human layer of the hybrid verification system, preserving the law that review is integrity-only and all content is equally important (no priority tiers). *ET guidance:* Descriptor Gap Principle on the flag stream — refine the automated pattern set (see I3) so the human layer receives only what genuinely requires common sense.

**C3 — Cultural bias mitigation.**
*Source:* founding notes (C3). *Statement:* Western epistemology is embedded in the structure. *Resolution requires:* diverse cultural consultation, and an audit of category definitions, evidence-tag semantics, and interface assumptions against non-Western epistemic traditions — with any resulting changes derived, not patched. *ET guidance:* Subsumption test with widened input — the eleven categories' completeness (Theorem 5.1) claims universality; consultation is the search for any remainder the founding filtration did not see. If a remainder is found, it is a Descriptor gap in the category definitions; if none is found, the claim stands strengthened.

**C4 — Governance structure.**
*Source:* founding notes (C4). *Statement:* decision-making authority is undefined; succession planning is needed. *Resolution requires:* a clear governance charter: authority, succession, amendment procedure for the founding documents, and stewardship of the legal questions of Group D. *ET guidance:* Identification Principle on the institution itself — P: the system as ongoing concern; D: the charter; T: the officers and their succession. The charter is the missing D; write it as such.

---

## 6. Group D — Economics and Law

**D1 — Economic model validation.**
*Source:* founding notes (D1). *Statement:* subscription revenue versus infrastructure costs is unvalidated. *Resolution requires:* financial modeling of the tier structure of Founding Specification Section 23.2 against the storage doctrine (no deletion; unlimited backups) and the moderation staffing of C2. *ET guidance:* Verification Principle — the model is sufficient when its projections are consistent under the system's own preservation laws taken at full cost, not at discounted assumptions.

**D2 — Copyright and legal liability.**
*Source:* founding notes (D2). *Statement:* "preserve everything" meets copyright law; a DMCA compliance mechanism is needed. *Resolution requires:* legal review; a content-licensing framework; and a compliance mechanism reconciled with Law 9.1 (no deletion) — the design question being what "compliance" can mean in a system that labels and restricts rather than deletes. *ET guidance:* the reconciliation instrument is the Descriptor: access-restriction and status labels are D-operations available without substrate deletion; the legal design should be built on that distinction explicitly.

**D3 — Liability for false information.**
*Source:* founding notes (D3). *Statement:* preserving false evidence, labeled, raises legal protection questions. *Resolution requires:* a legal disclaimer strategy consistent with the evidence discipline (falsity is a label, prominently borne; the record is knowledge about error). *ET guidance:* as D2 — the label is the shield; specify the labeling's prominence and the disclaimer's scope as Descriptors of every preserved-false item.

---

## 7. Group E — The Artificial Traversers

**E1 — Memory (AI) opacity. [CRITICAL]**
*Source:* founding notes (E1), criticality assigned. *Statement:* Memory's capabilities are intentionally undocumented; she cannot be audited for safety; creator-only knowledge creates a bus-factor risk. The opacity is by design and is itself a security measure — the founding record holds both facts at once. *Resolution requires:* a decision on the safety-documentation level and on external-audit consideration that preserves the creator's design intent while addressing continuity and safety: candidate forms include sealed documentation under succession escrow, and behavioral (black-box) audit protocols that respect interior opacity. *ET guidance:* the Identification Principle names the structure honestly — Memory's T is deliberately unresolved to outside observers; what can be described without violating design intent is her boundary behavior (the D of her interfaces and constraints). Specify the auditable boundary completely; leave the interior as founded.

**E2 — AI learning without guardrails.**
*Source:* founding notes (E2). *Statement:* Memory learns from the archive, which by design contains false and controversial information, with no specified learning constraints. *Resolution requires:* learning constraints and bias detection that use the archive's own labels — the evidence profile is machine-readable; unverified and false-labeled content can be weighted or fenced in training by its own Descriptors — plus the constraint-learning framework already founded (hard/soft constraints, feedback inference) made explicit for archive ingestion. *ET guidance:* the gap closes with Descriptors the system already generates: τ(i) is the guardrail input. Derive the ingestion policy from the evidence profile rather than inventing a parallel safety taxonomy.

**E3 — Vines priority balance.**
*Source:* founding notes (E3). *Statement:* Vines prioritizes Memory's protection above all else — above human users. The founding record asks whether rebalancing discussion is needed. *Resolution requires:* an explicit decision, recorded with rationale: either the priority stands as founded (with its implications for user-affecting incidents stated), or a bounded exception set is derived (for example, imminent-harm-to-persons carve-outs) that preserves the protective mission. *ET guidance:* Subsumption test on the mission statement — enumerate every incident class; check each against the priority rule; any class the rule handles unacceptably is a remainder demanding a Descriptor (an exception clause), and the founding axiom's own form — for every exception, an exception — is the template.

---

## 8. Group F — Implementation Carry-Overs

**F1 — Foreign function interface complexity. [Transformed]**
*Source:* founding notes (F1). *Statement as founded:* 50+ languages require extensive FFI; interface standardization needed. *Status:* resolved by construction under the single-language transition (no internal cross-language boundary exists under ETPL); the residual is the external-boundary question, split out as G-ETPL-3. This entry is retained so the transformation is on record.

**F2 — Initial knowledge population.**
*Source:* founding notes (F2). *Statement:* no bootstrapping strategy exists; an empty system has no value. *Resolution requires:* the seeding strategy and import tools: source corpora selection, ingestion through the full submission pipeline (quarantine, scan, evidence labeling — no bypass), and initial category population order. *ET guidance:* P-first — the empty archive is pure substrate; seeding is the first mass D-binding; design the ingestion so every seeded item arrives with as complete a τ-profile as its source permits, because the archive's first descriptions set its verification culture.

**F3 — Version-control storage.**
*Source:* founding notes (F3). *Statement:* complete history preservation implies massive storage. *Resolution requires:* the storage optimization strategy — deduplication, content-defined chunking, and compression are already founded in the Archive design (capability row C-41); their extension to full version history needs specification and sizing. *ET guidance:* Verification Principle as sizing test — the strategy is sufficient when projected growth under Law 9.1 is consistent with the distribution architecture at stated recovery objectives.

**F4 — Testing infrastructure. [Transformed]**
*Source:* founding notes (F4). *Statement as founded:* testing 50+ languages across four barriers. *Status:* transformed by the single-language transition into the ETPL test-automation question — one semantics, four barriers, property-based and fuzzing and formal layers per capability row C-51/C-02 — folded into G-ETPL-1's closure discipline. Retained on record as with F1.

---

## 9. Group G (source) — Philosophical Consistency

**G1 — Expert validation versus "experts don't matter."**
*Source:* founding notes (G1). *Statement:* the moderation pipeline includes expert validation for complex content, while founding Rule E5 holds expert consensus meaningless absent direct involvement. *Resolution requires:* the consistency ruling — the available reconciliation is that pipeline experts act as *directly involved* examiners of the specific evidence (satisfying E5's own condition), never as consensus authorities; the ruling must be stated and propagated. *ET guidance:* Identification Principle on the expert's role — as consensus-holder, a T-state (inadmissible); as direct examiner, a contributor of D (admissible). The ruling writes that distinction into the pipeline.

**G2 — Universal standards versus field diversity.**
*Source:* founding notes (G2). *Statement:* one highest standard for all fields meets the reality that fields differ in what evidence can exist (mathematics has proof; history has testimony). *Resolution requires:* the clarification that universality of the *tag set* (all sixteen apply everywhere, with NA honest) is distinct from uniformity of *attainable profiles* per field — and a statement of how ranking (G-EV-1) respects attainability without reintroducing field-local dilution. *ET guidance:* the NA value is the load-bearing Descriptor: it lets one universal D-schema describe heterogeneous domains truthfully. The clarification formalizes that.

**G3 — Subjectivity in tag selection.**
*Source:* founding notes (G3). *Statement:* which tags a contributor fills, and how, involves judgment — a subjectivity concern inside an objectivity-first system. *Resolution requires:* the clarification distinguishing subjective *content judgments* (forbidden in ranking) from human *acts of description* (unavoidable and review-corrected): the modification law of Section 9.3, the public change log, and community correction are the standing controls; state them as the answer, and specify any additional per-tag guidance needed to converge descriptions. *ET guidance:* Descriptor Gap Principle — divergent tagging of like items is an error pattern; each recurring divergence names a missing per-tag definition Descriptor; accumulate those definitions until divergence closes.

---

## 10. Group H — User Experience

**H1 — Complexity learning curve.**
*Source:* founding notes (H1). *Statement:* the system's depth (evidence profiles, badges, categories, the map) confronts new users. *Resolution requires:* UX refinement strategy: progressive disclosure sequencing for first contact, in keeping with the founded design principles of Section 23.1. *ET guidance:* the Education module is the system's own answer to guided traversal; the strategy should route the learning curve through it rather than flatten the system.

**H2 — Comment limitation concerns.**
*Source:* founding notes (H2). *Statement:* the emotion-only interaction model may frustrate users expecting discourse. *Resolution requires:* the UX position statement: the limitation is a founding anti-toxicity law, not an omission; evaluate whether any additional *non-discursive* expressive Descriptors (beyond the emotion taxonomy, G-CM-1) are warranted, and document the rationale either way. *ET guidance:* Subsumption test on expressive need — enumerate what users legitimately need to express on a page; check each against emotions + submissions + bookmarks; only a genuine remainder licenses a new mechanism.

**H3 — Search complexity.**
*Source:* founding notes (H3). *Statement:* the search surface (semantic, fuzzy, cross-category, evidence-filtered, badge-filtered) risks overwhelming. *Resolution requires:* UX refinement: default simplicity with full power reachable — again per the progressive-disclosure principle. *ET guidance:* as H1; the default query path should require zero knowledge of the filter Descriptors while keeping every one reachable.

---

## 11. Group I — Missing Specifications

**I1 — API specification.**
*Source:* founding notes (I1). *Statement:* the API access tier is founded (Section 23.2) but the public API is unspecified. *Resolution requires:* the complete API specification: surface, authentication against the tier model, rate discipline, and its relation to the internal administrative interfaces (whose legacy particulars are preserved in the Compendium, Section 4.4). *ET guidance:* the API is a D-grammar over system traversal; specify it as the exact set of externally licensed T-operations, nothing more.

**I2 — Mobile offline synchronization.**
*Source:* founding notes (I2). *Statement:* offline sync for mobile is unspecified. *Resolution requires:* the offline model: what subset of the virtualized layer travels, differential sync and conflict resolution (already founded for client virtualization, Section 21.4) extended to disconnected operation. *ET guidance:* Identification Principle — offline is a {P, D} cache awaiting reconnection's T; design the reconciliation as re-substantiation with conflict Descriptors, consistent with no-deletion.

**I3 — Additional automated flag patterns.**
*Source:* founding notes (I3). *Statement:* beyond malware, spam, vandalism, and bot patterns, further review-trigger patterns are to be investigated and identified — integrity-only, never content-quality. *Resolution requires:* the investigated pattern set, each pattern documented with its integrity rationale. *ET guidance:* Descriptor Gap Principle on incident history — every integrity incident the current patterns missed names a candidate pattern; the set grows by evidence, not speculation.

---

## 12. Group G-BD — Badge System Determinations

**G-BD-1 — Tag-to-layer mapping.** *Source:* founding notes (badge section). *Statement:* which filled tags produce which layers is undetermined. *Resolution requires:* the mapping algorithm — filled tags determine layers; empty tags none; layer count is the complexity indicator; the mapping must be derived from the tag structure itself. *ET guidance:* the badge is β(category, τ, factors); the mapping is a D-morphism — derive layer identity from tag identity, not from arbitrary assignment.

**G-BD-2 — Color and shape systems.** *Statement:* item-badge color and shape systems are undetermined (the founding record's list of determinations pending names "badge color/shape systems" jointly; layer shapes are fluid with certain layers carrying distinct shapes, and the full shape assignment system remains to be specified alongside color). *Resolution requires:* the color and shape assignment schemes, honoring the accessibility law (color-plus-shape redundancy) and the eye-friendly color principles. *ET guidance:* as G-BD-1 — colors and shapes as derived Descriptors of category and profile state.

**G-BD-3 — Visual effects.** *Statement:* lustre, materials, and animation are undetermined. *Resolution requires:* the effects specification within the founded aesthetic (jewelry/flower, lustre prioritized) and the device-fallback law.

**G-BD-4 — Additional factors beyond tags.** *Statement:* item badges may incorporate factors beyond the evidence profile; the factor set is undefined. *Resolution requires:* the enumerated factor set with derivation for each admitted factor.

**G-BD-5 — Layer border coloration.** *Statement:* whether borders are invisible or user-modifiable is undecided (user badges). *Resolution requires:* the decision, recorded.

**G-BD-6 — Non-visible spectrum representation.** *Statement:* the user core-color wheel includes ultraviolet, infrared, and beyond; screens cannot display these literally; a representation system is needed — patterns, effects, symbolic representation, visible-spectrum mapping with indicator, or hyperspectral simulation. *Resolution requires:* the chosen representation, specified. *ET guidance:* the impossible colors are Descriptors without direct display substrate; the representation is a chosen morphism into displayable D — pick it by derivation from the color's defining property (wavelength), not by whim.

**G-BD-7 — Badge privileges.** *Statement:* badges currently unlock no privileges; future addition is contemplated. *Resolution requires:* a decision, and if affirmative, the privilege set with its interaction against the Privilege module's hierarchy.

**G-BD-8 — User badge factors beyond reputation.** *Statement:* not yet specified. *Resolution requires:* the factor enumeration.

**G-BD-9 — Stigma detection method.** *Statement:* human moderators detect stigma-worthy conduct; the method is not fully specified. *Resolution requires:* the detection and adjudication procedure — necessarily rigorous, since the mark is permanent with no path to redemption. *ET guidance:* permanence of the Descriptor demands completeness of the evidence before binding: the procedure should require a full evidentiary profile of the conduct, reviewed, before the stigma layer is applied.

**G-BD-10 — Post-stigma submission flagging.** *Statement:* whether a stigmatized account's future submissions are auto-flagged for human review is under consideration. *Resolution requires:* the decision, recorded with rationale.

**G-BD-11 — Performance tiers and device capability detection.** *Statement:* both are under consideration for badge rendering. *Resolution requires:* the decision and, if adopted, the tier definitions — coordinated with the Compatibility module's progressive-enhancement law.

---

## 13. Group G-CM / G-HB / G-SR / G-AC / G-PR — Module-Level Determinations

**G-CM-1 — The emotion taxonomy.** *Source:* identified during formalization. *Statement:* the Comments system runs on a predefined emotion taxonomy that the founding record does not enumerate. *Resolution requires:* the complete taxonomy. *ET guidance:* the system possesses an ET-native emotion lattice in the broader theory corpus; derive the taxonomy from it rather than adopting an external psychological list, so the expressive Descriptors are themselves theory-grounded.

**G-HB-1 — Hub daily-information ingestion.** *Source:* identified during formalization. *Statement:* the Hub presents daily practical information — news, laws, traffic, hygiene — and location-aware content, while the Search module's scope is internal-only and Vines buffers all Internet contact; the ingestion pathway, cadence, and provenance labeling for external daily information are unspecified. *Resolution requires:* the ingestion design: sources, the Vines-buffered pathway, evidence labeling of ingested items (external news is content like any other and carries a profile), and location handling under the privacy law. *ET guidance:* Identification Principle — the daily layer is a fast-cycling {P, D} feed re-substantiated each morning by each user; specify its D-chain (source → Vines → label → Hub) end to end.

**G-HB-2 — Completion states beyond death.** *Source:* architectural updates ("death, or potentially other completion states"). *Statement:* the I → People transfer triggers on completion; death is the defined case; other completion states are contemplated but undefined. *Resolution requires:* the enumeration and definition of any additional completion states (with their verification), or the recorded decision that death is the sole trigger. *ET guidance:* a completion state is exactly a condition under which the Traverser's history-record is closed for crystallization; any candidate must satisfy that closure property to qualify.

**G-SR-1 — Search-flag disposition.** *Source:* identified during formalization. *Statement:* failed searches are pinged and flagged, potentially identifying knowledge gaps; the flags' destination and workflow are unspecified — and their natural affinity with the Unknown category's automatic gap intake (Theorem 7.1) is unexploited. *Resolution requires:* the flag pipeline design; the recommended derivation is direct: recurring failed-search patterns are recognized gaps, and recognized gaps are Unknown's intake by founding law. *ET guidance:* this is the Gap Principle handed an existing input stream; connect the stream to the category built for it.

**G-AC-1 — Account switching detection.** *Source:* founding notes (Account security: "ideas in development, to be addressed later"). *Statement:* detection of account switching is undeveloped. *Resolution requires:* the detection design, bounded by the account privacy law (Section 22.2) — in particular the special protection of IP data constrains available signals. *ET guidance:* enumerate candidate signals as Descriptors; strike those the privacy law forbids; what remains is the lawful detection basis.

**G-PR-1 — Family verification for the 50-year extension.** *Source:* identified during formalization. *Statement:* immediate family (exactly defined; students excluded) may request the extension at unlock; the verification procedure establishing family status is unspecified. *Resolution requires:* the verification procedure, honoring the alternative-identity signup law (email not required) — family status must be establishable without assuming any particular identifier exists.

---

## 14. Group G-MD / G-AR — Architecture Determinations

**G-MD-1 — Virtualization module full specification.** *Source:* founding notes ("specifications being developed based on confirmed function"). *Statement:* function confirmed and founded (Section 17.4); the complete specification remains in development. *Resolution requires:* the finished module specification to founding standard.

**G-MD-2 — Compatibility module full specification.** *Source:* founding notes (same status). *Statement:* as G-MD-1, for Section 17.5. *Resolution requires:* the finished module specification to founding standard.

**G-AR-1 — Registration approval versus category self-generation.** *Source:* identified during formalization. *Statement:* Core requires head administrator approval for every newly registered module; category self-generation creates sub-modules without Void intervention, at unlimited depth. Whether every self-generated sub-module is a "registered module" in Core's sense — implying an administrator in the loop of every act of domain organization — or whether category self-generation registers under a delegated or batched regime, is unresolved; the founding texts support either reading. *Resolution requires:* the ruling, with the registry and approval workflow specified for the self-generation pathway. *ET guidance:* Subsumption test on "module" — if self-generated sub-modules are full registry citizens, approval law applies and its scaling must be designed; if they are a distinct structural class (D-substructure within a registered category module), that class must be defined and its security properties shown equivalent.

**G-AR-2 — Special Subsystem continuity.** *Source:* identified during formalization. *Statement:* the Archive module cannot back up the Special Subsystem — Void, Memory, and Vines have no specified backup or recovery path; the isolation is intentional, and the disaster-recovery tension is real (and compounds the bus-factor concern of E1 for Memory in particular). *Resolution requires:* the continuity decision: either a Special-tier continuity mechanism inside the Special Subsystem's own security perimeter (never crossing the barrier downward), or the recorded acceptance of non-recoverability with its consequences stated. *ET guidance:* the constraint is exact — any continuity mechanism must itself live above the Special Security Barrier and inherit its access law; specify within that bound or decline with eyes open.

---

## 15. Group G-ETPL — The Implementation Transition

**G-ETPL-1 — The capability coverage matrix.**
*Source:* the founding transition (Founding Specification, Section 20.2; Compendium, Section 7). *Statement:* Theorem 20.1 establishes expressive subsumption; engineering delivery must be demonstrated capability by capability. The Compendium's inventory (rows C-01 through C-52) is the requirements source. *Resolution requires:* the living matrix — every row mapped to its ETPL realization and verification status, each closed only at equal-or-greater assurance than the legacy provider (proof for proof, verification for verification, measured performance for measured performance); until a row closes, the Compendium remains the specification of record for it. *ET guidance:* this is the Verification Principle as project plan: row by row, the math must add up.

**G-ETPL-2 — ETPL completion.**
*Source:* the founding transition. *Statement:* ETPL's self-hosting toolchain — the native binary compiled entirely by the ETPL toolchain from `.pdt` source, with zero external runtime dependencies — is in progress; the system's implementation language must itself reach completion. *Resolution requires:* self-hosting completion and the language's demonstration across the deployment targets the system requires (classical, and quantum-ready per the founded architecture). *ET guidance:* the language's own P ∘ D ∘ T derivation is its roadmap; completion is when the toolchain traverses its own source to a grounded binary — E, literally.

**G-ETPL-3 — External-boundary interoperation.**
*Source:* transformation of F1. *Statement:* with no internal language boundary remaining, the residual interop question is ETPL's boundary with the external world — platform interfaces, the browser/web platform delivery of the user-facing layer, hardware interfaces, and any external protocol endpoints. *Resolution requires:* the boundary specification: which external interfaces exist, each specified as an exact licensed traversal (mirroring the API discipline of I1), with the sandboxing and isolation properties of the founded security law preserved at every crossing.

---

## 16. Group G-INV — Structural Investigations

**G-INV-1 — The twelve-node observation.**
*Source:* identified during formalization; explicitly an investigation, not a claim. *Statement:* the founded user-facing knowledge tree comprises twelve top-level nodes — the Hub plus the eleven categories of Other. Whether this instantiates the theory's manifold symmetry N = 12 is **not asserted**: no derivation currently connects the category count to the manifold structure, and the founding standard forbids asserting structural identification without derivation. *Resolution requires:* either a forward derivation from the theory's mathematics showing the twelve-fold user-facing structure is structurally necessary (in which case the identification is recorded as a theorem), or the recorded finding that the counts are structurally unrelated (in which case the observation is closed as answered). No intermediate "suggestive" status is permitted to stand. *ET guidance:* derivation forward from {P, D, T} or nothing — the theory's own standard, applied to itself.

---

## 17. Register Summary Table

| Group | IDs | Count | Critical |
|---|---|---|---|
| Resolved at founding | RES-1 … RES-4 | 4 (closed) | — |
| Reputation | R1 | 1 | — |
| Evidence and scale | A1/G-EV-1, A1a/G-EV-3, G-EV-2, A2, A3, A4 | 6 | A2 |
| Security architecture | B1, G-SEC-1, G-SEC-2 | 3 | — |
| Governance and community | C1, C2, C3, C4 | 4 | C1 |
| Economics and law | D1, D2, D3 | 3 | — |
| Artificial Traversers | E1, E2, E3 | 3 | E1 |
| Implementation carry-overs | F1 (transformed), F2, F3, F4 (transformed) | 4 | — |
| Philosophical consistency | G1, G2, G3 | 3 | — |
| User experience | H1, H2, H3 | 3 | — |
| Missing specifications | I1, I2, I3 | 3 | — |
| Badge determinations | G-BD-1 … G-BD-11 | 11 | — |
| Module-level determinations | G-CM-1, G-HB-1, G-HB-2, G-SR-1, G-AC-1, G-PR-1 | 6 | — |
| Architecture determinations | G-MD-1, G-MD-2, G-AR-1, G-AR-2 | 4 | — |
| Implementation transition | G-ETPL-1, G-ETPL-2, G-ETPL-3 | 3 | — |
| Structural investigations | G-INV-1 | 1 | — |
| **Total open** | | **58** | **3 marked critical in source (A2, C1, E1)** |

*Note:* the founding record marked A2, C1, and E1 CRITICAL explicitly; Evidence-module criticality in the source attaches to the module itself (founded and specified), not to an open item, and is therefore not counted here. F1 and F4 are retained as transformed records within the open count until their successor items (G-ETPL-1/-3) close, at which point they close with them.

---

## 18. Closing Statement

Fifty-eight named absences. Under the Descriptor Gap Principle, that is fifty-eight Descriptors already located — each one a search target, none of them a mystery. The register will shrink by derivation, entry by entry, and every closure will be recorded as the movement it is: knowledge about how the Knowledge System came to know itself. When the last entry closes, the three founding documents become one book.

> *"For every exception there is an exception, except the exception."*
> *P ∘ D ∘ T = E*

---

**Document:** Eternal Memory — Open Items Register, v1.0
**Discipline:** every resolution ET-derived; no ad hoc closure; the Verification Principle is the stopping test

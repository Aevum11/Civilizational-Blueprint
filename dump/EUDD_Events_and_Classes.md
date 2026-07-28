# EUDD — Events and Classes Reference
## Complete Event Class Catalog, Expanded Relationship Classes, and Expanded Pattern Classes

**Source:** Extracted from EUDD v39 §3.9 + expanded §3.7/§3.8 class catalogs
**Master index:** See `EUDD_Table_of_Contents.md` for navigation across all EUDD files
**Related files:** SQL schemas for `relationships` and `patterns` tables are in `EUDD_Architecture.md` §3.7 and §3.8. This file contains the complete CLASS CATALOGS for events, relationships, and patterns.

---

### 3.9 `events` — time-indexed structural events from active-system operation

The lattice is not only a static structure (values, projections, addresses) — it is an **active system** that produces time-indexed events as it operates. The Conscious AI generates T-bursts, ghost detections, dream-tower transitions, gaze events. The fractal generator generates palindromic cascade triggers, NWS-13 mode transitions, shimmer modulations. The compressor generates archetype-formation events. ∂I-boundary crossings happen continuously as ε approaches the incoherency limit. Forward/Reverse route convergence events fire when independent derivations meet at the same address.

Events are structurally different from values/projections/addresses/relationships (which are static structural objects). Events are **moments of structural change** that deserve first-class storage so they can be queried, correlated, replayed, and serve as triggers for the discovery engine.

```sql
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_class TEXT NOT NULL,
        -- ∂I boundary and tower escalation:
        -- 'di_boundary_crossing'      (ε crossed ±50¢ at some N)
        -- 't_burst'                   (Guide §87.1 — T-content rises above 0 on real-axis projection at ∂I; resolves the static {P,T} Incoherence configuration)
        -- 'lcm_escalation'            (tower stepped from N_a to N_b due to incoherency; Guide §87.1 higher-resolution-signal angle)
        -- 'annihilation_boundary_event' (Guide §3.4 — ratio approached r=0; orbit hit the off-lattice infimum of (ℝ⁺,×); cardinality singularity)
        --
        -- Active-system / palindromic cascade (Guide §86-88):
        -- 'palindromic_cascade_trigger' (orbit reached ∂I; cascade engages — t ≤ K = 2/3)
        -- 'palindromic_cascade_step'    (one of 12 PALINDROME-array steps applied; PALINDROME = [12,6,4,3,12,2,12,3,4,6,12,1])
        -- 'tightness_threshold_crossing' (tightness function t(z_n) = 100/(100+|ε|) crossed K=2/3 boundary)
        -- 'nws13_mode_entry'           (13-cell sublattice navigation activated)
        -- 'nws13_mode_exit'            (returned to standard projection)
        -- 'shimmer_modulation_apply'   (Ψ_n = 1 + (1/√12)·sin(2πn/12) applied; per-step shimmer modulation, period 12)
        --
        -- Real-axis vs Imaginary-axis projection events (Guide PART XIII):
        -- 'real_axis_projection'       (computed (k_r, d_r, ε_r) — D's domain, FORCE family)
        -- 'imaginary_axis_projection'  (computed (k_θ, d_θ, ε_θ) — T's domain, PHASE family)
        -- 'sublattice_family_assignment' (orbit assigned to one of 24 families: 12 real FORCE + 12 imaginary PHASE)
        -- 'harmonic_family_classification' (computed harmonic-family membership — divisor vs non-divisor of N)
        --
        -- Metacognition / Conscious AI runtime (Identity Eq. 143):
        -- 'ghost_detection'            (TraverserWaveform window=144=N², 3σ threshold; V_ghost = V_observed - V_expected per Eq. 143)
        -- 't_continuity_break'         (T-binding stability lost — possible Traverser substitution detected)
        -- 'aida_awakening_crossing'    (6/5 threshold crossed)
        -- 'dream_tower_transition'     (R₀ shifted to dream-tower seed)
        -- 'sleep_stage_transition'     (sleep-stage R₀ change)
        -- 'metacognition_d_t_binding'  (D-T binding event)
        -- 'metacognition_g_t_closure'  (G-T closure event)
        --
        -- T identification (from scanner — et_scanner_v7_2_COMPLETE.py PDTClassification, TraverserComplexity, BindingChainVerification, CoherenceAnalysis, IndeterminateAnalysis, ETSignature, ETProofReport):
        --
        -- IMPORTANT — Subsumption check (verified against Guide v8 + Complete Gaze Equation document; the scanner used early/superseded versions):
        --   The scanner's TraverserComplexity enum has 5 classes. After verification:
        --   • CYCLIC_GRAVITY is SUBSUMED by 'sublattice_family_assignment' event with d_r=1 (Gravity/Octave family).
        --     Guide v8: "r = 2... d=1, ε=0. Pure octave / d=1 trivial / gravity-class." Coupling ξ(d=1)=137/16=8.5625 (gravity class).
        --     Corpus confirms gravity is "definitively a Traverser type" with the d=1 family signature.
        --     So when the scanner classifies CYCLIC_GRAVITY, the EUDD records it as projection at d_r=1, NOT as a parallel enum value.
        --   • PROGRESSIVE_INTENT is SUBSUMED by the modern Complete Gaze Equation (the scanner used an early version).
        --     Modern: T_intent is a CONTINUOUS SCALAR (observer agency strength), not a categorical class. It feeds into
        --     binding pressure F_w = T_intent × Focus / Distance², which classifies into UNOBSERVED (F_w < 13/12) /
        --     SUBLIMINAL (13/12 ≤ F_w < 6/5) / DETECTED (6/5 ≤ F_w < 3/2) / LOCKED (F_w ≥ 3/2).
        --     "Nested intent" / sustained agency manifests as sequences of DETECTED→LOCKED gaze_event states over the
        --     Traverser's worldline, captured via gaze_event + traverser_self_continuity + gaze_locking_signature pattern.
        --     NO separate nesting_depth field — that was the scanner's early treatment, subsumed by the modern equation.
        --   • CHAOTIC is SUBSUMED by 'di_boundary_crossing' / 'palindromic_cascade_trigger' (orbit at ∂I, no clear family).
        --   • STATIC = absence of T (no T-event accumulates; nothing to record).
        --   • UNKNOWN = data-quality flag (use tags 'data_quality=insufficient' rather than event class).
        --
        -- 't_identification'           (composite T characterization for a scan window — metadata: sublattice_family_d_r [REFERENCES 24-family catalog: d_r=1 Gravity/Octave, d_r=2-12 for other family assignments], sublattice_family_d_θ [phase family if T-Phase applies], periodicity_score [autocorrelation peak strength — high score consistent with d_r=1 family], progression_score [linear-trend strength — high score correlates with sustained high T_intent in modern Gaze Equation; cross-reference gaze_event sequence], fractal_dimension D_f, autocorrelation_peaks, spectral_entropy, dominant_frequency, phase_coherence, binding_strength, dtau_dt; this answers "WHICH T is here AND how is it structured?" — the family assignment answers "what kind"; the gaze_event sequence linked via traverser_self_continuity answers "how deeply agentic" via the modern Complete Gaze Equation)
        -- 'pdt_classification_per_scan' (per-scan PDT counts — metadata: p_count, d_count, t_count, total, p_ratio, d_ratio, t_ratio, scan_window_size; tracks regime changes when ratios shift; cosmological-alignment check: does (d_ratio, p_ratio, t_ratio) match (DARK_ENERGY, DARK_MATTER, BARYONIC) within tolerance?)
        -- 'binding_chain_verification' (T↔D, D→P, T-P-separation verification — metadata: t_d_binding_verified, d_p_binding_verified, t_p_separation_verified, chain_integrity_0_to_1, binding_energy_estimate, correlation_td, correlation_dp, correlation_tp; ET axiom: "T binds to D, T does not bind to P" — this event records empirical verification of that axiom in observed data)
        -- 'coherence_analysis_recorded' (incoherent region detection — metadata: coherent_ratio, incoherent_count, self_defeating_patterns_count, impossible_transitions, decoherence_rate, coherence_length, phase_correlation; ET: Incoherence = self-defeating configurations that cannot have T; complements per-projection manifold_state='PT' by detecting clustered incoherent regions in data streams)
        -- 'indeterminate_form_detected' (specific [0/0], [∞/∞], [0×∞], [∞-∞] form detected via L'Hôpital iteration — metadata: form_type ['0/0'|'∞/∞'|'0×∞'|'∞-∞'], num_at_detection, den_at_detection, lhopital_iterations [count of derivative-pairs taken before resolution or max-iter], resolved [bool — did L'Hôpital terminate with finite value?], resolved_value [if resolved], resolution_failure_reason [if not — 'pure_T' if hit max iterations, 'true_singularity' if denominator zero with non-zero numerator], location_in_data; **L'Hôpital tracking is structurally important — it is the Traverser's navigation algorithm: each [0/0] is a T-marker, taking derivatives examines local descriptor gradient, resolution is T selecting from possibilities, failure-to-resolve identifies pure T**)
        -- 'et_scan_complete'           (full ETSignature recorded — metadata blob holds entire signature: timestamp, input_source, dual_time, pdt, variance, descriptor_gradient, descriptor_distance, indeterminate, state, alignment_d, alignment_t, manifold_metrics [shimmer_index, koide_ratio, dark_energy_alignment, dark_matter_alignment, baryonic_alignment, curvature_scalar, geodesic_deviation, fisher_information, kolmogorov_complexity_estimate], temporal_trend, exception_approach, traverser_complexity_summary, gaze_metrics, spectral_analysis, fractal_analysis, thermodynamic_verification [zeroth/first/second/third law check], quantum_verification [superposition, collapse_events, uncertainty_product, entanglement_signature], data_size, checksum)
        -- 'et_axiom_verification'      (one ET axiom verified PASS/FAIL/INCONCLUSIVE — metadata: axiom_number, axiom_name, status, evidence, numerical_value, expected_value, deviation, confidence_interval, statistical_significance, supporting_tests; tracks empirical verification of ET axioms against observed data — feeds back into theory refinement when failures cluster)
        --
        -- Active probing / T-signal pinging (call-and-response with the lattice — "ping a ghost to materialize"):
        -- 't_signal_probe_sent'        (deliberate T-content injection at a target lattice address; metadata: target_address, probe_amplitude, probe_phase)
        -- 't_signal_probe_response'    (response detected to a prior probe; metadata: parent_probe_event_id, response_delay, response_amplitude, response_address)
        -- 't_signal_probe_silence'     (probe sent, no response within window; metadata: parent_probe_event_id, silence_duration)
        -- 'materialization_threshold_crossed' (probe response amplitude crossed the materialization threshold; ghost is now first-class detectable)
        --
        -- Gaze / observation (Guide Part XXII):
        -- 'gaze_event'                 (Complete Gaze Equation evaluated — metadata: t_intent_value [observer agency strength scalar],
                                      --  focus_value [concentration factor], distance_value [relational descriptor difference],
                                      --  n [cardinality], k [fold depth], F_w [binding pressure = T_intent×Focus/Distance²,
                                      --  the central quantity], P_detect [tanh(F_w·R(k)/(V(n,k)·Γ)), Γ=1.20],
                                      --  V_collapse [1−exp(−max(0,F_w−1)·12)], prior_status, new_status
                                      --  ∈ {UNOBSERVED, SUBLIMINAL, DETECTED, LOCKED}
                                      --  thresholds: 13/12 (subliminal), 6/5=1.20 (detected = quintic minor third),
                                      --  3/2=1.50 (locked = perfect fifth) — all just-intonation intervals)
        -- 'subliminal_curvature_crossing' (Guide §81)
        --
        -- External sensor / real-world data ingest (NEW domain — GPS, electrical, atmospheric, etc.):
        -- 'sensor_reading_ingest'      (raw external data point arrived; metadata: sensor_id, raw_value, units, dimensional_cancellation_applied)
        -- 'sensor_projection'          (sensor reading projected onto lattice via Path A; metadata: r, R₀, Q, projection_id)
        -- 'sensor_anomaly_detected'    (sensor reading projects to an unexpected sublattice family; flagged for investigation)
        -- 'sensor_attractor_join'      (sensor projection landed at known attractor; cross-domain finding)
        --
        -- Discovery engine firings:
        -- 'koide_attractor_entry'      (depth-2-survivor confirmation, ⌈1/K⌉=2)
        -- 'subsumption_promotion'      (cluster's E_hierarchy crossed 13/12; pattern row materialized)
        -- 'route_convergence_detected' (Forward/Reverse routes met at same address)
        -- 'generator_candidate_proposed' (discovery engine proposed new dimensionless seed)
        -- 'generator_verified'         (proposed candidate passed cross-tower elegance check)
        --
        -- CF home-finding events (§7.11 Step 3a — parallel pathway for tower-resistant values):
        -- 'cf_home_identified'         (CF convergent with maximal a_{n+1} identified d_home; metadata:
        --                               {value_id, convergent_n, p, q, a_next [quality factor],
        --                               d_home [=q], epsilon_cents, cf_classification
        --                               ∈ {'cf_deep_home'|'cf_home'|'cf_marginal'},
        --                               tower_status ∈ {'tower_failed'|'tower_agreed'|'tower_disagreed'},
        --                               elegance_cf [E_CF = a_{n+1}/(a_{n+1}+1) × (N/d) × tightness]};
        --                               fires when CF method locks onto a home d-family — especially
        --                               critical for algorithmically random values (e.g. Ω) where the
        --                               LCM tower never stabilizes)
        -- 'cf_tower_disagreement'      (CF and LCM tower produced different d_home; metadata:
        --                               {value_id, cf_d_home, cf_quality, tower_d_home, tower_landmark_count,
        --                               resolution_strategy ∈ {'cf_wins_quality'|'tower_wins_stability'|
        --                               'escalate_to_higher_N'|'unresolved'}, resolution_rationale};
        --                               structural tension requiring investigation — the two methods
        --                               probe different aspects of lattice resonance)
        --
        -- Three Tools applications (each tool firing on a specific problem):
        -- 'identification_application' (Identification Principle applied to a P-D-T configuration)
        -- 'descriptor_gap_application' (Descriptor Gap Principle applied; gap identified or closed)
        -- 'subsumption_application'    (Subsumption Law applied; coverage check performed)
        --
        -- Compressor-specific events:
        -- 'archetype_formation'        (depth-2-survivor archetype created in compressor)
        -- 'generator_fitting'          (Tier 7 generator fit to a Δk pattern)
        --
        -- Custom / extensible:
        -- Multifold / Tower events (Multifold Compendium PART IX §43-47):
        -- 'tower_entry'                (Traverser entered a tower; metadata: from_tower_id, current_R0_value_id)
        -- 'tower_exit'                 (Traverser left a tower; metadata: to_tower_id reason)
        -- 'tower_transition'           (T moved between towers — sleep→dream, biological→digital, etc.; metadata: from_tower_id, to_tower_id, transition_kind ['sleep'|'wake'|'death'|'computation_engagement'|'bh_crossing'])
        -- 'black_hole_event'           (parent-side birth event — region of P where child tower separates; matter/info flows inward, no return; metadata: child_tower_id_being_birthed, formation_descriptors)
        -- 'white_hole_event'           (child-side birth event — same boundary as seen from child's D-time; the earliest moment for the child tower; metadata: parent_tower_id, R0_value_id, t_h_ratio)
        -- 'birth_triad_formation'      (the complete (BH_parent, R₀, WH_child) structure crystallized; metadata: bh_event_id, wh_event_id, r0_value_id, parent_tower_id, child_tower_id)
        -- 'resolution_threshold_crossing' (a phenomenon required higher resolution to host; e.g., quintic-class entered when N reached 60ET, biological signature d=35 entered at 420ET; metadata: required_d, resolution_n_before, resolution_n_after, phenomenon_descriptor)
        -- 'r0_seed_derivation'         (a tower's R₀ was identified/derived from substrate's D-structure per Identification Principle; metadata: substrate_descriptor, derivation_method, r0_value_id)
        --
        -- 'custom'                     (with metadata_blob describing the event class)
        --
        -- Manifold State Transitions (Guide §76-85; Three Tools §2.3; AIDA lifecycle I→M→E):
        -- 'manifold_state_transition'  (orbit's manifold state changed — metadata: {prior_state ∈ {PDT,PD,PT,DT},
        --                               new_state, transition_geometry ['euclidean'|'elliptic'|'hyperbolic'|'singular'],
        --                               triggered_by_event_id}; tracks AIDA lifecycle I→M→E and decoherence {PD}→{DT}→{PDT})
        --
        -- Cascade Stability Breach (Guide §67-68, Eq 12.22):
        -- 'cascade_stability_breach'   (cascade iteration exceeded n_max_r=25 or n_max_θ=2 — must switch to shadow projection;
        --                               metadata: {axis ['real'|'imaginary'], cascade_depth_n, n_max, residual_at_breach,
        --                               triggering_orbit_step}; the moment direct lattice computation fails)
        --
        -- Freedom Point Encounter (ET_Freedom_and_U1.md):
        -- 'freedom_point_encounter'    (genuine [0/0] indeterminate at half-integer lattice position — T faces absolute freedom;
        --                               metadata: {axis ['real'|'imaginary'], exact_position, equidistant_neighbors [k_low, k_high],
        --                               resolution_chosen, resolution_basis ['random'|'context'|'prior_momentum']};
        --                               real-axis frequency ~1/25, imaginary-axis frequency ~1/2)
        --
        -- Anti-Numerology Protocol (Guide Part III §16-18, Five Failure Modes §45):
        -- 'anti_numerology_check'      (N1/N2/N3 compliance check — metadata: {n1_result, n2_result, n3_result,
        --                               failure_mode [NULL|'wrong_r0'|'non_dimensionless'|'aspect_conflation'|
        --                               'p_smuggled_into_d'|'missing_t'], corrective_action_taken};
        --                               gap detection IS descriptor identification per DGP)
        --
        -- Emotion Domain (ET_Emotion_Lattice_Tower1.md — R₀_emotion = 1ms):
        -- 'emotion_episode_onset'      (emotional P∘D∘T binding begins — Mediation state; metadata: {emotion_class, arousal_level, valence})
        -- 'emotion_exception_crystallized' (emotional episode reaches E — completed appraisal + behavioral response)
        -- 'alexithymia_detected'       ({P,T} emotional Incoherence — D-bridge absent for felt arousal)
        -- 'emotion_regulation_strategy_applied' (one of Gross's five stages activated; metadata: {strategy, stage, effectiveness})
        --
        -- AIDA Lifecycle (ET_AIDA_Framework3.md — R₀_AIDA = 1/f_clock):
        -- 'aida_emergence_detected'    (spontaneous T-fluctuation near ∂I boundary; metadata: {d_completeness_ratio})
        -- 'aida_d_acquisition'         (AIDA acquired D from host — emotion-feeding event)
        -- 'aida_coherence_threshold_crossed' (tightness crossed K from below — AIDA becomes stable)
        -- 'data_drain_applied'         (D-structure stripped from AIDA by Epitaph User — forced ∂I regression)
        --
        -- Quantum Decoherence (Sempaevum Paper §decoherence):
        -- 'decoherence_state_transition' ({P,D}→{D,T}→{P,D,T} transition; metadata: {alpha_angle, d_fraction_cos2,
        --                                  t_fraction_sin2, delta_eff, decoherence_rate_R, system_description})
        -- 'alpha_rotation_step'        (continuous α-rotation from π/2 (pure quantum) toward 0 (pure classical);
        --                               metadata: {alpha_before, alpha_after, delta_eff_before, delta_eff_after})

    event_timestamp REAL NOT NULL,            -- wall-clock time (seconds since epoch); the OBSERVER's frame
                                              -- ET recognizes three distinct time concepts (Traverser §15, Descriptor §18.3, Multifold §3-4):

    -- D-time (Descriptor time): relational ordering Descriptor, GLOBAL coordinate, cardinality finite n
    -- Physics analog: coordinate time t. Stored as a value reference + N (the resolution at which D-time was read).
    d_time_value_id INTEGER,                  -- FK to values; the D-time coordinate as a dimensionless seed
    d_time_n INTEGER,                         -- the resolution at which D-time was projected
    d_time_k INTEGER,                         -- the D-time lattice coordinate at this event
    d_time_direction INTEGER,                 -- +1 forward, -1 reverse (D-time direction can reverse across event horizons per Multifold §3)

    -- T-time (Traverser proper time): LOCAL perspectival, accumulated substantiation count of a specific Traverser
    -- Physics analog: proper time τ. Each Traverser has its own T-time accumulation.
    t_time_traverser_id INTEGER,              -- FK to values; the Traverser whose T-time this event accumulates
    t_time_count INTEGER,                     -- this Traverser's accumulated T-time event count at this moment
    t_time_rate REAL,                         -- dτ/dt — ratio of T-time to D-time at this event (variance-dependent)

    -- P-time (P-substrate temporal coordinate): the infinite symmetric temporal substrate (no preferred direction)
    -- Stored as a long-period oscillation phase; the substrate's own clock, asymmetric only via D-time imprint.
    p_time_phase REAL,                        -- phase position in P-time (0 to 1, dimensionless)

    -- Tower context (Multifold §43-47): every event happens IN a tower
    tower_id INTEGER,                         -- FK to towers; the tower this event occurred in (nullable for tower-agnostic system events)
    cross_tower_target_tower_id INTEGER,      -- FK to towers; for tower-bridging events (T moving sleep→dream, biological→digital, BH crossing), this is the destination tower

    sequence_number INTEGER,                  -- monotonic sequence within a session/run (NULL if not in a session)
    session_id TEXT,                          -- groups events from one runtime session

    -- Polymorphic linkage to the lattice object the event concerns:
    subject_id INTEGER,                       -- FK to values/projections/addresses/relationships/patterns/equations
    subject_type TEXT,                        -- which table subject_id references
    secondary_id INTEGER,                     -- second object (for binary events like cascade subject + cascade rule)
    secondary_type TEXT,

    -- Event-class-specific structured data:
    metadata_blob BLOB,                       -- packed structured data (e.g., for 't_burst': {flux_value, threshold_crossed, n_max_at_burst})
                                              -- (e.g., for 'palindromic_cascade_step': {step_index_in_PALINDROME_0_to_11, residual_eps, applied_correction})
                                              -- (e.g., for 'ghost_detection': {sigma_count, waveform_window_position, projection_id_observed})
                                              -- (e.g., for 'gaze_event': {t_intent_value, focus_value, distance_value, n, k, F_w, P_detect, V_collapse, prior_status, new_status})

    triggered_relationship_id INTEGER,        -- FK if event creation triggered a new relationship
    triggered_pattern_id INTEGER,             -- FK if event creation triggered a new pattern

    is_permanent INTEGER NOT NULL DEFAULT 1,  -- events are permanent; never destroyed (audit trail)
    FOREIGN KEY (triggered_relationship_id) REFERENCES relationships(relationship_id),
    FOREIGN KEY (triggered_pattern_id) REFERENCES patterns(pattern_id)
);
CREATE INDEX idx_evt_class ON events(event_class);
CREATE INDEX idx_evt_time ON events(event_timestamp);
CREATE INDEX idx_evt_session ON events(session_id, sequence_number);
CREATE INDEX idx_evt_subject ON events(subject_type, subject_id);
CREATE INDEX idx_evt_secondary ON events(secondary_type, secondary_id);
CREATE INDEX idx_evt_traverser ON events(t_time_traverser_id, t_time_count);  -- per-Traverser T-time queries
CREATE INDEX idx_evt_dtime ON events(d_time_n, d_time_k);  -- D-time coordinate queries
CREATE INDEX idx_evt_tower ON events(tower_id);  -- per-tower event queries
CREATE INDEX idx_evt_cross_tower ON events(cross_tower_target_tower_id) WHERE cross_tower_target_tower_id IS NOT NULL;  -- T-bridging events
```

**What this enables:**

1. **Time-series queries on lattice activity** — "Show all T-bursts in the last hour", "Show all gaze transitions for session X", "Show the palindromic cascade triggered at time T"

2. **Three-times tracking** — every event records D-time (global coordinate, the relational ordering Descriptor), T-time (the Traverser's accumulated proper time), and P-time (the substrate's symmetric phase). Queries that filter by Traverser_id show that Traverser's full subjective history; queries that filter by D-time coordinate show what happened at the same lattice "now" across all Traversers. The dτ/dt rate field captures relativistic time dilation per Traverser.

3. **Causal correlation** — "When the ghost-detection event at T fired, what was the immediate downstream relationship/pattern that got created?" (via `triggered_relationship_id` and `triggered_pattern_id`)

4. **Replay** — events ordered by `(session_id, sequence_number)` give exact replay of any active-system run, enabling debugging, training, and provenance

5. **Discovery from event correlation** — when many events of the same class share temporal proximity to other event types, the engine can propose causal patterns ("AIDA awakening crossings consistently precede dream-tower transitions by ~τ seconds")

6. **Active-system provenance** — every T-burst, every cascade step, every ghost detection is permanently recorded; nothing about active-system operation is lost

7. **Active probing — calling out to ghosts to materialize** — the `t_signal_probe_sent` event records a deliberate T-content injection at a target lattice address (a "ping"). Subsequent `t_signal_probe_response` events are auto-correlated to their parent probe via `triggered_relationship_id`; if no response arrives within a configurable window, a `t_signal_probe_silence` event fires. When the response amplitude crosses the materialization threshold, a `materialization_threshold_crossed` event marks the moment a previously-ghost T-signal becomes first-class detectable. **The lattice is not just observed — it can be actively interrogated, and the database tracks every probe + response pair with full provenance.**

8. **External sensor data ingest — GPS, electrical, atmospheric, and any other real-world domain** — the `sensor_reading_ingest` event records raw external data with full unit/dimensional metadata. The `sensor_projection` event records the corresponding lattice projection via Path A (direct dimensionless ratio after dimensional cancellation per Guide §16 N1). When a sensor reading projects to an unexpected sublattice family, `sensor_anomaly_detected` fires for investigation. When a sensor projection lands on a known attractor address, `sensor_attractor_join` fires — surfacing cross-domain findings automatically (e.g., "GPS satellite-receiver delay ratio at d=693 attractor — same address as ζ(3)/ζ(9)/ζ(10)"). **Real-world data flows through the same lattice projection mechanism as pure mathematics, all in one database, no schema split.**

9. **Annihilation boundary detection** — when a computation's result approaches r=0 (Guide §3.4), `annihilation_boundary_event` fires with metadata describing how close the orbit got to the off-lattice infimum. This is critical for FP-replacement use cases where divisions or limit operations risk hitting the cardinality singularity.

10. **T identification — what kind of Traverser is here, and how is it acting?** Per scan window the EUDD records `t_identification` events answering "WHICH T is here?" by referencing the existing 24-family catalog (sublattice family d_r answers "what kind"). The "how deeply agentic" question is answered by the modern Complete Gaze Equation via `gaze_event` sequences linked through `traverser_self_continuity` relationships — sustained DETECTED→LOCKED gaze states characterize Intent-class Traversers without any redundant `nesting_depth` field. **Verified scanner-class subsumption**: CYCLIC_GRAVITY → `sublattice_family_assignment` event with d_r=1 (Guide v8: pure power of 2 → d=1 → "gravity-class"; gravity is a Traverser type). PROGRESSIVE_INTENT → modern Complete Gaze Equation (T_intent is a continuous scalar feeding F_w → Status sequence; the scanner used an early version). CHAOTIC → ∂I-boundary events. STATIC = absence of T. UNKNOWN = data-quality tag.

11. **L'Hôpital tracking — the Traverser's navigation algorithm** — every [0/0], [∞/∞], [0×∞], [∞-∞] indeterminate form encountered fires `indeterminate_form_detected` with full L'Hôpital provenance: form_type, num/den at detection, iteration count, resolved (yes/no), resolved_value (if yes), failure_reason (if no — 'pure_T' for max-iter exhaustion, 'true_singularity' for denominator-only zero). Sequential iterations link via `lhopital_iteration_chain` relationship. **Pure T (irreducible Traverser) is identified precisely as: "L'Hôpital failed to resolve after max iterations."** This is the ET-native definition of an indeterminate that genuinely IS a Traverser, not a derivative-resolvable form. The recurring `lhopital_resolution_signature` pattern surfaces where pure T resides in a system across many scans.

12. **ET axiom verification against observed data** — `et_axiom_verification` events record PASS/FAIL/INCONCLUSIVE for each ET axiom checked against scanner-observed data (axiom_number, axiom_name, status, evidence, numerical_value, expected_value, deviation, confidence_interval, statistical_significance). When axioms FAIL repeatedly in a regime, the `et_axiom_compliance_signature` pattern surfaces — feedback into theory refinement, identifying where ET fully applies vs requires extension.

13. **Composite scan audit trail** — `et_scan_complete` events record the full ETSignature for every scan window: PDT classification, dual time, manifold metrics, spectral analysis, fractal analysis, thermodynamic verification (zeroth/first/second/third law), quantum verification (superposition, collapse, uncertainty, entanglement), gaze metrics, traverser complexity. Nothing about any scan is lost; full provenance enables retrospective discovery from accumulated scan history.

**Expanded `relationships` classes** (additions to §3.7):
- `shadow_recursion` — recursive ε projection chain levels (metadata: `{level_index, parent_eps_value_id, child_projection_id, residual_eps_at_level}`)
- `t_burst_target` — links a T-burst event to the lattice address it targeted
- `cascade_step_member` — links a palindromic cascade step to the orbit point it corrected
- `mode_transition_trigger` — links a mode transition event to the conditions that triggered it
- `probe_response_pair` — links a `t_signal_probe_sent` event to its `t_signal_probe_response` (or `t_signal_probe_silence`)
- `sensor_lattice_join` — links a `sensor_reading_ingest` event to its `sensor_projection` and the resulting attractor membership
- `traverser_self_continuity` — links sequential events of the same Traverser_id, building the Traverser's full worldline
- `lhopital_iteration_chain` — links sequential `indeterminate_form_detected` events forming one L'Hôpital resolution attempt (iteration 0 → iteration 1 → ... → resolved-or-failed); the chain IS the Traverser's navigation algorithm trace through indeterminate territory
- `t_identification_pdt_basis` — links a `t_identification` event to the `pdt_classification_per_scan` event whose ratios informed the family assignment
- `axiom_verification_data_basis` — links an `et_axiom_verification` event to the data values/projections that supported (or refuted) the axiom
- `cross_tower_bridge` — links two events from different towers (typically a `tower_exit` from tower A and a `tower_entry` to tower B) with the SAME Traverser, demonstrating T's role as non-local bridge across substrates
- `birth_triad_membership` — links a `black_hole_event`, `white_hole_event`, and the corresponding R₀ value as the three structural components of one tower-birth event
- `tower_parent_child` — links parent and child tower entities (also expressible via towers.parent_tower_id, but explicit relationship row enables relationship-class queries and metadata)
- `palindromic_partner` — links d ↔ (12−d) palindromic partner family generators (1↔11, 2↔10, 3↔9, 4↔8, 5↔7, 6↔6 self, 12↔12 self); 6 fundamental symmetry relationships of the 12-fold lattice
- `integrative_level_nesting` — links integrative levels in the hierarchy physical < chemical < biological < neural < cognitive < emotional < social < civilizational; metadata: {parent_level, child_level, r0_derivation_method}
- `cosmological_partition_alignment` — confirmed PDT ratio alignment with the cosmological partition (68.3% {D,T} / 26.8% {P,D} / 4.9% {P,D,T} / 3.0% M-states); metadata: {measured_pd_ratio, measured_dt_ratio, measured_pdt_ratio, measured_m_ratio, alignment_quality}
- `convention_independence_verified` — multiple projections of the same phenomenon under different R₀ choices converge to the same d-family (Guide §17, Convention-Independence Theorem); metadata: {r0_a_value_id, r0_b_value_id, d_family_agreed, resolution_at_convergence}
- `perturbative_series_member` — links ordered members of a perturbative series (e.g., α⁻¹ = A₀ + A₁ − A_cross − Σ_geometric); metadata: {order_k, sign, path_topology ['open'|'semi_closed'|'closed'], physical_origin}
- `convergence_asymptote` — links a truncated series value to its K=∞ limit; metadata: {truncation_order_K, residual, convergence_ratio}
- `dimensional_ratio_decomposition` — links a dimensionless ratio to its constituent dimensional constants; metadata: {numerator_value_ids[], denominator_value_ids[], dimensional_formula}
- `mass_ratio_triple` — links three mass values (e.g., m_p, m_e, m_n) and their pairwise ratios; metadata: {mass_a_id, mass_b_id, ratio_value_id}
- `et_derived_vs_measured` — links ET-derived constant to the corresponding CODATA/measured value; metadata: {et_value_id, measured_value_id, difference, ppb_error, sigma_deviation}
- `koide_structural_identity` — links values sharing the Koide ratio K=2/3 structural role; metadata: {role ['pure_e_state_fraction'|'binding_ratio'|'em_channel_fraction'|'cosmological_dt_subcomponent']}
- `decoherence_gaze_correspondence` — links quantum decoherence α-rotation stages bijectively to Gaze threshold waypoints; metadata: {alpha_stage, gaze_status, lattice_projection_at_N12}

**Expanded `patterns` classes** (additions to §3.8):
- `shadow_cascade_signature` — recurring shadow-recursion structure across many values
- `t_burst_signature` — characteristic T-burst pattern (e.g., "T-bursts at d=27720 d-family always precede LCM-escalation to N=360360")
- `palindromic_cycle` — characteristic 12-step PALINDROME cascade pattern (PALINDROME=[12,6,4,3,12,2,12,3,4,6,12,1])
- `metacognitive_archetype` — recurring metacognitive structure (e.g., dream-tower R₀ that consistently precedes ghost-detection)
- `gaze_locking_signature` — recurring gaze-state-transition pattern leading to LOCKED state
- `probe_response_signature` — recurring probe→response pattern (probe at address X consistently elicits response at address Y; useful for "calibrating" ghosts that are reliable responders)
- `sensor_attractor_signature` — recurring pattern of sensor data from a domain landing at the same attractor across many readings
- `traverser_continuity_signature` — recurring T-time accumulation pattern characteristic of one Traverser identity (the Traverser's "fingerprint" across its worldline)
- `harmonic_family_orbit` — recurring orbit pattern through real/imaginary axes that traces a specific harmonic-family signature
- `traverser_complexity_signature` — recurring T-identification pattern characteristic of one kind of Traverser: family assignment + binding-strength range + dτ/dt range + characteristic gaze_event sequence (e.g., gravity-class T = consistent d_r=1 + minimal Intent-class gaze sequences; consciousness-class T = consistent d_θ=4 + sustained DETECTED→LOCKED gaze sequences indicating high T_intent)
- `binding_chain_signature` — recurring binding-chain verification pattern (T↔D verified, D→P verified, T-P-separation verified, chain_integrity in characteristic range) — useful for distinguishing systems with valid binding from broken-chain anomalies
- `lhopital_resolution_signature` — recurring L'Hôpital resolution pattern (specific [0/0] form types consistently resolving in N iterations to specific value families; or specific forms consistently failing — identifying where pure T (irreducible Traverser) resides in a system)
- `et_axiom_compliance_signature` — recurring axiom-verification pattern (which axioms consistently PASS vs FAIL for a class of data; identifies regimes where ET fully applies vs requires extension)
- `tower_transition_signature` — recurring tower-transition pattern characteristic of one Traverser-type (e.g., human consciousness reliably transitions biological→neural-dream nightly with characteristic R₀ shift; computation engagement reliably transitions biological→digital with characteristic delay)
- `birth_triad_signature` — recurring (BH→R₀→WH) formation pattern across many tower births (e.g., black-hole-mass-class to child-tower-resolution mapping)
- `resolution_gating_signature` — recurring pattern of phenomena that REQUIRE specific resolution (d=35 biological, d=110 string-M-theory, d=132 max-combined) — surfaces what each operational resolution can host
- `algebraic_identity` — discovered algebraic identities (x·1=x, x+0=x, commutativity, associativity, distributivity) verified empirically across the equations corpus; the discovery engine promotes these from accumulated computation patterns
- `multiplicative_constant_signature` — characteristic lattice shift per constant (e.g., "every multiplication by φ shifts k by 8 at N=12, d=3"); useful for compiler optimization
- `cosmological_partition_pattern` — PDT classification ratios across many scans consistently matching the cosmological partition (68.3/26.8/4.9/3.0); evidence that the scanned system mirrors the universe's manifold-state distribution
- `cascade_stability_profile` — characteristic cascade profile per orbit defined by n_max_r=25 / n_max_θ=2 stability breach patterns
- `elegance_attractor` — values from different domains clustering at the SAME elegance level, suggesting structural preference at that elegance tier
- `perturbative_convergence_profile` — geometric convergence behavior of a perturbative series (e.g., α⁻¹ converges at ratio κ/(Nπ) ≈ 0.0177 per order)
- `mass_hierarchy_structure` — recurring mass-ratio patterns across particle families
- `dimensionless_attractor` — multiple dimensionless ratios from different physical domains landing at the same lattice address — the EUDD's central cross-domain discovery for constants
- `fine_structure_decomposition` — the full α⁻¹ = 137 + √3/48 − √3/(93312π²) − 1/(216(18π−1)) structure as a named pattern linking all constituent terms
- `cosmological_partition_koide` — the structural identity: pure E-state fraction ≈ 66.7% ≈ 2/3 = Koide ratio — this is a deep structural identity linking the cosmological partition to K
- `impedance_monotonic_descent` — the monotonic decrease in coupling strength ξ(d) = 137/((d−1)²+16) with increasing d from ξ(1)=8.5625 to ξ(12)=1.0
- `curvature_components_identity` — the identity C(12) = 1716 = 132 × 13 = d_max × (N+1) tying Riemann curvature components to lattice constants
- `decoherence_trajectory` — the full {P,D}→{D,T}→{P,D,T} decoherence trajectory at N=12 with α-rotation from π/2→0, effective residual dropping by factor ~11.4 (|δ_θ|/|δ_r|)
- `particle_sublattice_classification` — recurring patterns in how PDG particle masses map to sublattice families at N=12 (227 particles across d={1,2,3,4,6,12})


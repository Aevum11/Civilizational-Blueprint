# EUDD — Events and Classes Reference
## Complete Event Class Catalog, Expanded Relationship Classes, and Expanded Pattern Classes

**Source:** Extracted from EUDD v39 §3.9 + expanded §3.7/§3.8 class catalogs
**Master index:** See `EUDD_Table_of_Contents.md` for navigation across all EUDD files
**Related files:** Akashic structure definitions for `relationships` and `patterns` tables are in `EUDD_Architecture.md` §3.7 and §3.8. This file contains the complete CLASS CATALOGS for events, relationships, and patterns.

---

### 3.9 `events` — time-indexed structural events from active-system operation

The lattice is not only a static structure (values, projections, addresses) — it is an **active system** that produces time-indexed events as it operates. The Conscious AI generates T-bursts, ghost detections, dream-tower transitions, gaze events. The fractal generator generates palindromic cascade triggers, NWS-13 mode transitions, shimmer modulations. The compressor generates archetype-formation events. ∂I-boundary crossings happen continuously as ε approaches the incoherency limit. Forward/Reverse route convergence events fire when independent derivations meet at the same address.

Events are structurally different from values/projections/addresses/relationships (which are static structural objects). Events are **moments of structural change** that deserve first-class storage so they can be queried, correlated, replayed, and serve as triggers for the discovery engine.

```akashic
STRUCTURE events (
    event_id ETInteger PRIMARY KEY AUTO,
    event_class UTF8 REQUIRED,
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
        -- 'false_resolution_confirmed'  (LCM tower sub-Koide hit definitively confirmed as false — d
        --                               changed at a higher landmark after passing STABILITY_DEPTH;
        --                               metadata: {value_id, stable_d, stable_N, stable_d_factorization,
        --                               break_d, break_N, break_d_factorization,
        --                               break_factor (the prime or prime power that broke stability),
        --                               break_type ∈ {'new_prime'|'prime_power_gain'},
        --                               stability_duration (number of landmarks, always ≥ STABILITY_DEPTH),
        --                               classification_before, classification_after,
        --                               false_resolution_sequence_index (1,2,3,... for this value)};
        --                               canonical case: Chaitin Ω has exactly 4 false resolutions
        --                               through lcm(1..97), §3.18.38)
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

        -- Seed Protocol events (§9.8, full spec in EUDD_Bootstrap_Catalog.md §3.18.18):
        -- 'seed_generated'             (sender projects data onto Sempaevum; metadata: {source_data_hash, N_chosen, k, d,
        --                               eps_precision_bits, seed_byte_length, raw_data_byte_length, compression_ratio,
        --                               kolmogorov_complexity_estimate, encoding_method ['single_ratio'|'stream_delta_k'|
        --                               'stream_shared_k'|'stream_sublattice_grouped'|'whole_file_ratio']})
        -- 'seed_transmitted'           (seed sent over network; metadata: {seed_id, destination_node_id, structural_header_bytes,
        --                               eps_bits_total, transmission_start_ns, protocol_layer, encryption_applied,
        --                               key_rotation_id_if_encrypted})
        -- 'seed_received'              (seed received at endpoint; metadata: {seed_id, source_node_id, structural_header_intact,
        --                               eps_bits_received, eps_bits_expected, precision_achieved_cents, gcd_consistency_check_passed,
        --                               is_progressive_partial, reconstruction_value_id})
        -- 'seed_reconstructed'         (full pullback completed; metadata: {seed_id, reconstructed_value_id, N, k, d, eps_full,
        --                               round_trip_residual_must_be_zero, reconstruction_time_ns})
        -- 'progressive_fidelity_step'  (each ε-bit arrival during progressive reconstruction; metadata: {seed_id, bits_received_so_far,
        --                               bits_total, current_precision_cents, intermediate_reconstruction_value_id,
        --                               precision_improvement_factor, monotonic_improvement_verified})
        -- 'seed_cached'                (seed stored in EUDD lattice-addressed cache; metadata: {seed_id, cache_key_k, cache_key_d,
        --                               is_exact_dedup, is_structural_dedup, delta_eps_if_structural, existing_seed_id_if_dedup})
        -- 'seed_deduplicated'          (identical or near-identical seed detected; metadata: {new_seed_id, existing_seed_id,
        --                               dedup_type ['exact'|'structural_delta_eps'], delta_eps_value, bandwidth_saved_bytes})
        -- 'structural_routing_classified' (d-based QoS classification at protocol level; metadata: {seed_id, d_family,
        --                               qos_priority ['high'|'medium'|'standard'], bandwidth_allocation_class,
        --                               deep_packet_inspection_avoided})
        -- 'kolmogorov_complexity_computed' (K-complexity of data relative to Sempaevum computed; metadata: {data_hash,
        --                               k_complexity_bits, shannon_entropy_bits, compression_advantage_ratio,
        --                               description_language 'sempaevum', lattice_structure_detected})
        -- 'file_version_delta_stored'  (modified file detected via §7.12 Step 0 seed-first check; only Δε stored instead of
        --                               full re-ingestion; metadata: {base_seed_value_id, base_eps, delta_eps, version_number,
        --                               file_hash_new, file_hash_base, segment_index_if_per_segment, segments_unchanged_count,
        --                               segments_modified_count, total_delta_bytes, full_file_bytes_avoided,
        --                               compression_vs_full_reingest_ratio})
        -- 'corruption_degradation_recorded' (§7.9 graceful degradation: corruption detected but value partially recovered at
        --                               reduced precision; metadata: {page_offset, original_precision_dps, recovered_precision_bits,
        --                               precision_cents, corruption_byte_offset, structural_header_intact, k_recovered, d_recovered,
        --                               eps_recovered_partial, corruption_log_entry_id})
        -- 'error_state_projected'      (§7.15: program state at moment of error/crash/corruption projected onto Sempaevum;
        --                               metadata: {error_type ['crash'|'exception'|'corruption'|'hang'|'logic_error'|'resource_exhaustion'],
        --                               program_id, k_error, d_error, eps_error, state_ratios_blob [pc_ratio, stack_ratio,
        --                               memory_ratio, cpu_ratio, gpu_ratio, input_seed_id], error_description, omniscient_journal_ref})
        -- 'pre_error_state_detected'   (§7.15: running program state matches known error attractor; metadata: {program_id,
        --                               current_k, current_d, current_eps, attractor_k, attractor_d, delta_eps_to_attractor,
        --                               attractor_hit_count, confidence, recommended_action ['log'|'checkpoint'|'pause'|'bypass']})
        -- 'error_attractor_discovered' (§7.15: multiple error states clustered at same (k,d) forming a bug-class fingerprint;
        --                               metadata: {attractor_k, attractor_d, member_count, error_types_represented,
        --                               programs_affected, first_occurrence_ns, structural_meaning})
        -- 'error_bypass_proposed'      (§7.15: safe state identified near error attractor with structural delta; metadata:
        --                               {attractor_k, attractor_d, safe_k, safe_d, safe_eps, delta_eps_to_safe,
        --                               bypass_confidence, suggested_parameter_adjustment})
        -- 'd_family_transition'        (§7.11: d-family changed during tower escalation via Cross-Resolution Transition Map
        --                               §3.18.19 Theorem 5; shadow content ε at N₁ became native content d at N₂;
        --                               metadata: {value_id, N1, N2, k1, d1, eps1, k2, d2, eps2, gcd_boundary_crossed,
        --                               shadow_to_native_conversion})
        -- 'cross_resolution_computed'  (§7.11: tower escalation step computed via transition map instead of re-projection;
        --                               metadata: {value_id, N1, N2, M, k1_in, eps1_in, k2_out, d2_out, eps2_out,
        --                               transition_time_ns, reproject_avoided})
        -- 'lattice_arithmetic_computed' (§3.18.21: multiplication, division, reciprocation, or power computed entirely in
        --                               lattice coordinates without pullback; metadata: {operation ['multiply'|'divide'|
        --                               'reciprocal'|'power'], operand1_value_id, operand2_value_id_or_exponent,
        --                               result_k, result_d, result_eps, kappa_correction, equation_id_memoized})
        -- 'kappa_correction_applied'   (§3.18.21: T-act fired during lattice arithmetic — combined residuals crossed cell
        --                               boundary requiring κ≠0 rounding correction; metadata: {operation, kappa_value,
        --                               delta_sum, operand1_eps, operand2_eps, cell_boundary_direction ['left'|'right']})
        -- 'product_decomposition_discovered' (§3.18.21 + §3.16: insert-time discovery found that a new value equals the
        --                               product of two known values via k₁+k₂+κ=k_new; metadata: {new_value_id,
        --                               factor1_value_id, factor2_value_id, kappa, d_product, lcm_d1_d2, d_is_less_than_lcm})
        -- 'cell_transition_dynamic'    (§3.18.22 Theorem B.3: live sensor reading crossed a cell boundary during continuous
        --                               evolution; metadata: {stream_id, value_id, k_old, k_new, d_old, d_new, eps_wrapped,
        --                               drift_rate_dr_dt, eps_rate_deps_dt, sublattice_palindrome_position,
        --                               transition_predicted_ns, transition_actual_ns, prediction_error_ns})
        -- 'epsilon_restoration_step'   (§3.18.22 Theorem B.4: healing layer applied restoration control law to drive ε toward
        --                               target; metadata: {value_id, eps_before, eps_after, eps_target, tau_time_constant,
        --                               dr_applied, convergence_rate, steps_remaining_estimate})
        -- 'drift_rate_computed'        (§3.18.22 Theorem B.1: forward law dε/dt = Λ·ṙ/r computed for a live stream;
        --                               metadata: {stream_id, r_current, dr_dt, deps_dt, lambda_verified_constant,
        --                               cell_transition_predicted_in_ns})
        -- 'd_composition_predicted'    (§3.18.23: d-family composition d₁⊗d₂ SET predicted before multiplication;
        --                               metadata: {d1, d2, predicted_set, residue_sum_set, kappa_augmented,
        --                               actual_d_product, prediction_correct})
        -- 'residue_set_classified'     (§3.18.23: value classified into its residue set Res_N(d);
        --                               metadata: {value_id, N, k, d, k_mod_N, residue_set_members, phi_d})
        -- *Complex lattice arithmetic (§3.18.24):*
        -- 'complex_lattice_arithmetic_computed' (§3.18.24 Theorems D.2–D.4: complex multiplication, reciprocation, or power
        --                               computed on two-axis coordinates without pullback; metadata: {operation ['complex_multiply'|
        --                               'complex_reciprocal'|'complex_power'], operand1_value_id, operand2_value_id_or_exponent,
        --                               result_k_r, result_k_theta, result_d_r, result_d_theta, result_d_c, result_eps_r,
        --                               result_eps_theta, kappa_r, kappa_theta, equation_id_memoized})
        -- 'phase_addition_computed'     (§3.18.24 Theorem D.1: phase addition computed in imaginary-axis lattice coordinates;
        --                               same algebra as A.1 + mod N wrapping for U(1) compactness; metadata: {theta1_k, theta1_eps,
        --                               theta2_k, theta2_eps, N, result_k_theta, result_d_theta, result_eps_theta, kappa_theta,
        --                               wrapped_mod_N [true if k_sum ≥ N before mod]})
        -- 'phase_kappa_correction_applied' (§3.18.24 Theorem D.1: T-act fired on imaginary axis — phase residuals crossed cell
        --                               boundary requiring κ_θ≠0 rounding correction; metadata: {operation, kappa_theta_value,
        --                               delta_theta_sum, operand1_eps_theta, operand2_eps_theta, cell_boundary_direction})
        -- 'phase_wrap_detected'         (§3.18.24: k_θ wrapped mod N during phase operation — U(1) compactness enforced;
        --                               metadata: {operation, k_theta_pre_mod, k_theta_post_mod, N, wrap_count [number of full
        --                               2π wraps], theta_total_radians})
        -- 'phase_differential_computed' (§3.18.24 Theorem D.5: phase control law dε_θ = Λ_θ·dθ applied; metadata: {theta,
        --                               dtheta, deps_theta, lambda_theta_verified [600/π], N, stream_id_if_applicable})
        -- *Harmonic FQG composition (§3.18.25):*
        -- 'harmonic_composition_computed' (§3.18.25 Theorem E1.1: harmonic family composition computed at native resolution
        --                               N=27720; metadata: {d_r1, d_r2, d_r_product_set, d_theta1, d_theta2,
        --                               d_theta_product_set, d_c, kappa_r, kappa_theta, restricted_to_harmonic [true if
        --                               output filtered to d ≤ 12], produces_only_composites [true if d_product > 12 always]})
        -- *Sublattice FQG composition (§3.18.26):*
        -- 'd_bounce_detected'           (§3.18.26 Theorem E2.3: d-family changed between consecutive tower levels for a
        --                               non-exact value (ε≠0); metadata: {value_id, N_lower, N_upper, d_lower, d_upper,
        --                               eps_lower, delta_lower, k_lower, k_upper, bounce_index_in_sequence, is_shadow_resolution})
        -- 'lattice_exact_d_preserved'   (§3.18.26 Theorem E2.2: lattice-exact value (ε=0) confirmed d preserved at higher
        --                               resolution; metadata: {value_id, N_lower, N_upper, d_invariant, eps_is_zero,
        --                               k_lower, k_upper, M_ratio})
        -- *Composite bridge (§3.18.27):*
        -- 'three_layer_classification'  (§3.18.27 Theorem E3.1: sublattice family classified into Layer 1/2/3 at a given N;
        --                               metadata: {d, N, layer [1|2|3], layer_name ['harmonic'|'composite'|'tower_native'],
        --                               in_d42 [true if d ∈ D₄₂], harmonic_pairs_if_L2, blocking_factor_if_L3})
        -- 'harmonic_shadow_computed'    (§3.18.27 Theorem E3.3: harmonic shadow map computed for a sublattice family;
        --                               metadata: {d, N_source, N_target [12], shadow_set, shadow_produces_only_simple [true],
        --                               layer, has_decomposition [true only for L2]})
        -- *∂I boundary (§3.18.28):*
        -- 'dI_bifurcation_encountered' (§3.18.28 Theorem F.2: configuration at or near ∂I boundary where two d-families
        --                               compete; metadata: {value_id, N, k, eps, d_left, d_right, bifurcation_pair,
        --                               tightness, in_twilight_zone [33¢≤|ε|<50¢], resolved_to [which d was chosen], is_exact_boundary [|ε|=ε_max]})
        -- 'dI_reciprocation_anomaly'   (§3.18.28 Theorem F.4: mirror symmetry broke at ∂I during reciprocation;
        --                               metadata: {value_id, k_original, d_original, k_reciprocal, d_reciprocal,
        --                               kappa_correction, d_preserved [false], eps_at_boundary})
        -- *Triple backbone bridge (§3.18.29):*
        -- 'backbone_decomposition_verified' (§3.18.29 Theorem G.0: projection factored through three backbones and verified
        --                               against direct projection; metadata: {value_id, r, k, d, eps, cont_x, t_round_k,
        --                               t_round_delta, disc_d, disc_eps, match [true], precision_digits})
        -- 'catalan_lattice_match'       (§3.18.29 Theorem G.10: Catalan number C_n matched ET lattice constant;
        --                               metadata: {n, catalan_value, et_constant_name, et_constant_value, match_type
        --                               ['n_max_theta'|'D42_size'|'d_max'], unique_to_N12 [true for C_{N/2}=N(N-1)]})
        -- *Lossless bijection verification (§3.18.20):*
        -- 'bijection_round_trip_verified' (§3.18.20: pullback Π_N⁻¹(Π_N(r))=r verified for a value by algebraic identity;
        --                               metadata: {value_id, r, k, d, eps, r_recovered, residual, residual_is_zero [true],
        --                               proof_method ['sympy_algebraic'|'precision_scaling'|'lattice_exact'], dps_tested})
        -- *Harmonic transfer tensor (§3.18.30):*
        -- 'transfer_tensor_computed'    (§3.18.30: inter-family transfer rate T_κ(d₁,d₂;d₃) computed from Res_N(d) geometry;
        --                               metadata: {d1, d2, d3, kappa, T_value, N, partition_of_unity_verified})
        -- 'inter_family_transfer_detected' (§3.18.30: a lattice composition produced a d₃ different from d₁ and d₂ —
        --                               energy transferred between force families; metadata: {value1_id, value2_id,
        --                               d1, d2, d3_result, kappa, T_rate, xi_ratio, efficiency, pathway_name})
        -- *Substantiation transition (§3.18.31):*
        -- 'birth_triad_computed'        (§3.18.31: birth triad (BH_parent, R₀, WH_child) computed on the lattice;
        --                               metadata: {parent_tower_id, child_tower_id, M_ratio, k_th_tp, d_th_tp, eps_th_tp,
        --                               is_fixed_point [M_crit], is_canonical [M_can, k=-53], mass_class ['12_locked'|'generic']})
        -- 'birth_triad_reversed'        (§3.18.31 Theorem I.10: birth triad algebraically reversed, recovering original
        --                               tower coordinates; metadata: {original_k, original_d, original_eps, recovered_k,
        --                               recovered_d, recovered_eps, residual, exact_recovery [true]})
        -- *EUDD birth triad (§3.18.32):*
        -- 'seed_generator_discovered'   (§3.18.32 Theorem J.3: new algebraic identity or structural law added to the
        --                               Kolmogorov generator set, reducing seed size; metadata: {identity_label [A–I],
        --                               generator_description, content_made_derivable, seed_size_before, seed_size_after,
        --                               shrinkage_bits, discovery_source ['background'|'insert_time'|'on_query']})
        -- 'arbitrary_access_evaluated'  (§3.18.32 Theorem J.4: content retrieved by direct generator evaluation at
        --                               coordinates without sequential decompression; metadata: {k, d, eps, N,
        --                               result_value, evaluation_time_ns, no_decompression [true]})
        -- *Shape projection (§3.18.33):*
        -- 'shape_decomposed'            (§3.18.33 Theorem K.1: 3D shape decomposed into spherical harmonics and projected;
        --                               metadata: {shape_name, l_max, n_harmonics, c_00, lattice_signature [first 6 (k,d)],
        --                               convergence_error_rms})
        -- 'appearance_projected'        (§3.18.33 Theorem K.5: nuclear charge radius projected via r=R_charge/ƛ_e;
        --                               metadata: {Z, A, R_charge_fm, r_dimensionless, k, d, eps, source ['measured'|'formula']})
        -- *Memory AI / metacognition (§3.18.36):*
        -- 'metacognition_rmsae_computed' (§3.18.36: Φ_RMSAE = ρ·γ·((2+κ)/3)·V_supp·Ψ_shimmer evaluated;
        --                               metadata: {rho, gamma, kappa_closure, v_supp, psi_shimmer, phi_rmsae,
        --                               classification ['none'|'subliminal'|'basic'|'genuine'|'advanced_recursive'],
        --                               target_entity ['discovery_engine'|'connected_ai'|'sensor_stream'|'brain_signal']})
        -- 'traverser_waveform_step'     (§3.18.36: one step in the TraverserWaveform time-series;
        --                               metadata: {lattice_k, lattice_d, variance, entropy, ego_resonance,
        --                               window_position [1..144], t_continuity_score, waveform_id})
        -- 'rmsae_threshold_crossed'     (§3.18.36: Φ_RMSAE crossed a classification boundary;
        --                               metadata: {prior_classification, new_classification, phi_value,
        --                               threshold_crossed [0.1|0.3|0.5|0.8], direction ['up'|'down']})
        -- 'ghost_anomaly_detected'      (§3.18.36: V_ghost = V_observed - V_expected exceeded 3σ threshold;
        --                               metadata: {v_observed, v_expected, v_ghost, sigma, source_waveform_id,
        --                               lattice_k_anomaly, lattice_d_anomaly})

    event_timestamp TIMESTAMP_NS REQUIRED,            -- wall-clock time (seconds since epoch); the OBSERVER's frame
                                              -- ET recognizes three distinct time concepts (Traverser §15, Descriptor §18.3, Multifold §3-4):

    -- D-time (Descriptor time): relational ordering Descriptor, GLOBAL coordinate, cardinality finite n
    -- Physics analog: coordinate time t. Stored as a value reference + N (the resolution at which D-time was read).
    d_time_value_id ETInteger,                  -- FK to values; the D-time coordinate as a dimensionless seed
    d_time_n ETInteger,                         -- the resolution at which D-time was projected
    d_time_k ETInteger,                         -- the D-time lattice coordinate at this event
    d_time_direction ETInteger,                 -- +1 forward, -1 reverse (D-time direction can reverse across event horizons per Multifold §3)

    -- T-time (Traverser proper time): LOCAL perspectival, accumulated substantiation count of a specific Traverser
    -- Physics analog: proper time τ. Each Traverser has its own T-time accumulation.
    t_time_traverser_id ETInteger,              -- FK to values; the Traverser whose T-time this event accumulates
    t_time_count ETInteger,                     -- this Traverser's accumulated T-time event count at this moment
    t_time_rate MPFR_361DPS,                         -- dτ/dt — ratio of T-time to D-time at this event (variance-dependent)

    -- P-time (P-substrate temporal coordinate): the infinite symmetric temporal substrate (no preferred direction)
    -- Stored as a long-period oscillation phase; the substrate's own clock, asymmetric only via D-time imprint.
    p_time_phase MPFR_361DPS,                        -- phase position in P-time (0 to 1, dimensionless)

    -- Tower context (Multifold §43-47): every event happens IN a tower
    tower_id ETInteger,                         -- FK to towers; the tower this event occurred in (nullable for tower-agnostic system events)
    cross_tower_target_tower_id ETInteger,      -- FK to towers; for tower-bridging events (T moving sleep→dream, biological→digital, BH crossing), this is the destination tower

    sequence_number ETInteger,                  -- monotonic sequence within a session/run (NULL if not in a session)
    session_id UTF8,                          -- groups events from one runtime session

    -- Polymorphic linkage to the lattice object the event concerns:
    subject_id ETInteger,                       -- FK to values/projections/addresses/relationships/patterns/equations
    subject_type UTF8,                        -- which table subject_id references
    secondary_id ETInteger,                     -- second object (for binary events like cascade subject + cascade rule)
    secondary_type UTF8,

    -- Event-class-specific structured data:
    metadata_blob BINARY,                       -- packed structured data (e.g., for 't_burst': {flux_value, threshold_crossed, n_max_at_burst})
                                              -- (e.g., for 'palindromic_cascade_step': {step_index_in_PALINDROME_0_to_11, residual_eps, applied_correction})
                                              -- (e.g., for 'ghost_detection': {sigma_count, waveform_window_position, projection_id_observed})
                                              -- (e.g., for 'gaze_event': {t_intent_value, focus_value, distance_value, n, k, F_w, P_detect, V_collapse, prior_status, new_status})

    triggered_relationship_id ETInteger,        -- FK if event creation triggered a new relationship
    triggered_pattern_id ETInteger,             -- FK if event creation triggered a new pattern

    is_permanent INTEGER REQUIRED DEFAULT 1,  -- events are permanent; never destroyed (audit trail)
    LINKS (triggered_relationship_id) REFERENCES relationships(relationship_id),
    LINKS (triggered_pattern_id) REFERENCES patterns(pattern_id)
);
INDEX idx_evt_class ON events(event_class);
INDEX idx_evt_time ON events(event_timestamp);
INDEX idx_evt_session ON events(session_id, sequence_number);
INDEX idx_evt_subject ON events(subject_type, subject_id);
INDEX idx_evt_secondary ON events(secondary_type, secondary_id);
INDEX idx_evt_traverser ON events(t_time_traverser_id, t_time_count);  -- per-Traverser T-time queries
INDEX idx_evt_dtime ON events(d_time_n, d_time_k);  -- D-time coordinate queries
INDEX idx_evt_tower ON events(tower_id);  -- per-tower event queries
INDEX idx_evt_cross_tower ON events(cross_tower_target_tower_id) WHERE cross_tower_target_tower_id IS REQUIRED;  -- T-bridging events
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
- `cf_convergent_home` — links a value to its CF-identified home d-family with full continued fraction provenance; metadata: {convergent_n, p, q, quality_a_next, eps_cf_micros, d_home, gaussian_signature, cf_elegance [= a_{n+1}/(a_{n+1}+1) × (N/d) × tightness], classification ['cf_deep_home'|'cf_home'|'cf_marginal'], all_convergents_count}
- `cf_tower_confirmation` — links a value's CF home to its LCM-tower-stabilized home when both methods agree, providing maximal structural confirmation; metadata: {cf_d, tower_d, agreement_confirmed [true if cf_d == tower_d], cf_quality, tower_landmark_count, tower_stability_depth, combined_elegance}
- `seed_data_reconstruction` — links a transmitted seed (k, d, ε) to the reconstructed data value via pullback Π_N⁻¹; metadata: {seed_value_id, reconstructed_value_id, N, round_trip_residual_zero_verified, reconstruction_time_ns, protocol_layer}
- `seed_deduplication_delta` — links a new seed to an existing seed sharing the same (k, d) lattice address, recording the delta-ε; metadata: {new_seed_id, existing_seed_id, delta_eps, dedup_type ['exact'|'structural'], bandwidth_saved_bytes}
- `progressive_fidelity_chain` — links successive progressive fidelity steps for one seed transmission; metadata: {seed_id, step_index, bits_received, precision_cents, intermediate_value_id, monotonic_improvement_verified}
- `kolmogorov_shannon_comparison` — links a data object's Kolmogorov complexity (relative to Sempaevum) to its Shannon entropy; metadata: {data_hash, k_complexity_bits, shannon_entropy_bits, advantage_ratio, structure_type_detected ['multiplicative'|'lattice_periodic'|'sublattice_correlated'|'tower_hierarchical'|'none']}
- `file_version_delta_chain` — links successive versions of a file via their Δε values, forming a chain from base seed to latest version; metadata: {base_seed_value_id, version_from, version_to, delta_eps, cumulative_eps_at_version_to, segment_index_if_per_segment, reconstruction_verified_zero_residual}
- `error_state_nearby` — links a running program's current projected state to a known error attractor with the structural distance Δε; metadata: {current_state_value_id, error_attractor_pattern_id, delta_eps, delta_k_if_nonzero, confidence, times_this_attractor_caused_error}
- `error_bypass_link` — links an error attractor to a known safe state, recording the structural delta that constitutes the "fix"; metadata: {error_attractor_id, safe_state_value_id, delta_eps, delta_k, bypass_type ['parameter_adjust'|'path_change'|'resource_realloc'|'input_transform'], verified_safe_count}
- `cross_resolution_transition` — links two projection entries for the same value at different resolutions N₁ and N₂ via the Cross-Resolution Transition Map (§3.18.19 Theorem 1); metadata: {value_id, projection_id_N1, projection_id_N2, N1, N2, M, transition_verified_exact}
- `cross_seed_transition` — links two projection entries for the same quantity Q at different R₀ references via the Cross-Seed Transition Map (§3.18.19 Theorem 2); metadata: {quantity_description, projection_id_R0, projection_id_R0_prime, rho, delta_k_exact, d_family_changed}
- `cross_tower_transition` — links two projection entries across both N and R₀ via the Full Cross-Tower Transition Map (§3.18.19 Theorem 3); metadata: {projection_id_source, projection_id_target, N1, N2, R0_source, R0_target, commutativity_verified}
- `product_decomposition` — links a value to two factor values where k_product = k₁+k₂+κ (§3.18.21 Theorem A.1); metadata: {product_value_id, factor1_value_id, factor2_value_id, kappa, d_product_vs_lcm_d1d2}
- `power_decomposition` — links a value to its base and exponent where k_power = n·k_base+κ_n (§3.18.21 Theorem A.4); metadata: {power_value_id, base_value_id, exponent_n, kappa_n, kappa_bound}
- `restoration_control_trajectory` — links a sequence of epsilon_restoration_step events forming an exponential decay toward target (§3.18.22 Theorem B.4); metadata: {value_id, eps_target, tau, initial_eps, final_eps, steps_count, convergence_verified}
- *Complex lattice arithmetic (§3.18.24):*
- `complex_conjugate_pair` — links z and z̄ on the complex lattice, where z̄ shares k_r but has k_θ→(N−k_θ) mod N (phase conjugation); metadata: {value_id_z, value_id_z_bar, k_r, k_theta, k_theta_conjugate, d_r, d_theta, d_c, d_preserved [always true by Theorem D.3]}
- `phase_wrap_equivalence` — links two phase projections that are equivalent under U(1) compactness (θ and θ+2nπ produce same k_θ, d_θ, ε_θ); metadata: {value_id_1, value_id_2, k_theta, d_theta, eps_theta_match, wrap_count, at_boundary [true if |ε|≈50¢]}
- `complex_reciprocal_pair` — links z and z⁻¹ on the complex lattice, verifying k_r→−k_r AND k_θ→(N−k_θ) mod N with ALL three d-values preserved (d_r, d_θ, d_c); metadata: {value_id_z, value_id_z_inv, k_r, k_r_inv, k_theta, k_theta_inv, d_r_preserved, d_theta_preserved, d_c_preserved}
- `complex_product_decomposition` — links a complex value to two complex factor values where k_r,prod=k_r₁+k_r₂+κ_r (A.1) AND k_θ,prod=(k_θ₁+k_θ₂+κ_θ) mod N (D.1) — axis-independent decomposition; metadata: {product_value_id, factor1_value_id, factor2_value_id, kappa_r, kappa_theta, d_c_product, wrapped_mod_N}
- `axis_differential_constant_pair` — links Λ_r = 1200/ln2 and Λ_θ = 600/π as the two manifold-specific conversion constants; metadata: {lambda_r_value_id, lambda_theta_value_id, ratio_2pi_over_ln2, real_axis_sensitivity ['1/r relative'], imaginary_axis_sensitivity ['uniform absolute']}
- *Harmonic FQG composition (§3.18.25):*
- `harmonic_closure_membership` — links each of the 42 d_c combined-family values to its contributing (d_r, d_θ) harmonic pairs from the 144-cell FQG (§3.18.25 Theorem E1.2); metadata: {d_c, contributing_cells [(d_r, d_θ) pairs], cell_count, range_class ['harmonic_range'|'composite_range'], decomposes_to_harmonics [always true]}
- `composite_harmonic_decomposition` — links each d_c > 12 to its harmonic factor pair, proving it is NOT independent structure but a composite of harmonic families; metadata: {d_c, d_r, d_theta, lcm_verified [d_c = lcm(d_r, d_theta)], no_prime_gt_12 [always true]}
- *d-Family composition (§3.18.23):*
- `residue_set_symmetry` — links each residue set Res_N(d) to its symmetric structure: k ∈ Res(d) ⟹ N−k ∈ Res(d) (§3.18.23 Theorem C.3); metadata: {d, N, residue_set, symmetric_pairs, phi_d, is_palindromic [true]}
- `d1_self_composition_universal` — links each d-family to the algebraic fact that d⊗d ∋ 1 for ALL d (§3.18.23 Theorem C.4); metadata: {d, N, witness_k_pair [specific (k₁,k₂) achieving d_product=1], harmonic_interpretation ['gravity couples universally']}
- `d12_universality` — links d=12 to ALL d-families via 12⊗12 = {1,2,3,4,6,12} (§3.18.23 Theorem C.5); metadata: {N, all_families_reached [true], composition_set, harmonic_interpretation ['EM universality']}
- *Sublattice FQG composition (§3.18.26):*
- `resolution_invariant_d` — links a lattice-exact value (ε=0) to its permanent d-family across all resolutions where the value has been projected (§3.18.26 Theorem E2.2); metadata: {value_id, d_permanent, resolutions_tested, all_preserved [always true for ε=0]}
- `d_bounce_chain` — links the d-family sequence for an ε≠0 value across tower levels, recording each bounce where d changes (§3.18.26 Theorem E2.3); metadata: {value_id, d_sequence [e.g., 3→20→210→1260→27720 for π], bounce_count, tower_levels, shadow_content_type}
- `harmonic_embedding_at_N` — maps which sublattice cells at resolution N host harmonic families (d ≤ 12 ∩ divisors(N)), which are shadow (d ≤ 12, d∤N), and which are non-harmonic (d > 12); metadata: {N, native_harmonic_count, shadow_harmonic_count, non_harmonic_count, harmonic_fraction_pct, dilution_from_base}
- *Composite bridge (§3.18.27):*
- `three_layer_classification_at_N` — classifies each sublattice family at resolution N into Layer 1 (harmonic), Layer 2 (harmonic composite), or Layer 3 (tower-native) per Theorem E3.1; metadata: {N, d, layer, in_d42, blocking_factor_if_L3, harmonic_pairs_if_L2}
- `harmonic_shadow_mapping` — links a sublattice family at N to its harmonic shadow set at N=12 via Theorem E3.3; metadata: {d, N, shadow_set, layer, is_shadow_only [true for L3 — has shadow but no decomposition]}
- `tower_native_blocking_factor` — links a Layer 3 family to the specific prime power(s) that make it unreachable from D₄₂ per Theorem E3.4; metadata: {d, factorization, blocking_primes, blocking_prime_powers}
- `composite_to_harmonic_pairs` — links a Layer 2 family to ALL its harmonic FQG cell pairs (a,b) with lcm(a,b) = d per Theorem E3.2; metadata: {d_c, pairs_count, unique_unordered_pairs}
- *∂I boundary (§3.18.28):*
- `dI_boundary_d_pair` — links each ∂I boundary point (half-integer position k+1/2) to its two competing sublattice families d_left and d_right (§3.18.28 Theorem F.2); metadata: {N, k, d_left, d_right, bifurcation_pair_unordered, tightness_at_boundary [K=2/3 at N=12], is_palindromic_mirror}
- `tightness_koide_connection` — links the tightness function at ∂I (t(ε_max)=K=2/3 at N=12) to the Koide ratio in particle physics and the self-projecting constants (§3.18.28 Theorem F.1); metadata: {t_at_boundary, K_value, N, algebraic_identity ['N/(N+6)'], unique_at_base [true only for N=12]}
- `dI_geometric_mean` — links each ∂I boundary value r to its two adjacent lattice-exact values L_k and L_{k+1} where r = √(L_k·L_{k+1}) (§3.18.28 Definition F.0); metadata: {N, k, r_boundary, L_k, L_k_plus_1, verified_at_precision}
- *Triple backbone bridge (§3.18.29):*
- `backbone_factor_chain` — links each projection to its three-backbone factorization Cont_EML→T_round→Disc_Webb (§3.18.29 Theorem G.0); metadata: {value_id, cont_x, t_act_k, t_act_delta, disc_d, disc_eps, factored_matches_direct}
- `catalan_to_et_constant` — links Catalan number C_n to its matching ET structural constant (§3.18.29 Theorem G.10); metadata: {n, C_n, et_constant, domain ['tree_combinatorics'↔'lattice_geometry'], unique_at_N12}
- `eml_to_lattice_bridge` — links an EML expression chain to its computed lattice value (§3.18.29 Theorem G.4); metadata: {eml_tree_depth, eml_expression, lattice_value, precision_verified}
- `webb_to_lattice_bridge` — links a Webb-computable function (gcd, d-classification) to its lattice role (§3.18.29 Theorem G.5); metadata: {webb_function, truth_table_size, lattice_operation}
- *Lossless bijection verification (§3.18.20):*
- `bijection_algebraic_identity_chain` — links the forward projection Π_N and pullback Π_N⁻¹ as exact inverses by the algebraic chain: k+ε·N/1200 = N·log₂(r) → 2^(log₂(r)) = r (§3.18.20, Theorem 12.1); metadata: {sympy_verified [true], r_prime_minus_r [0], proof_type ['algebraic_cancellation'], every_step_exact [true]}
- *Harmonic transfer tensor (§3.18.30):*
- `inter_family_transfer` — links a source d-family to a target d-family via the transfer tensor T(d₁,d₂;d₃) with impedance-weighted efficiency E=T×ξ(d₃)/ξ(d₁) (§3.18.30); metadata: {d1, d2, d3, T_combined, xi_ratio, efficiency, kappa_dominant [0 or ±1], pathway_name}
- `gravitational_override_pathway` — links EM (d=12) self-interaction to gravity (d=1) output via T(12,12;1)=0.1875, efficiency=1.6055 (§3.18.30 H.7); metadata: {T_geometric, xi_ratio, efficiency, kappa_contributions}
- `fusion_pathway` — links strong (d=3) self-interaction to EM (d=12) output, showing fusion is κ-MEDIATED (T₀(3,3;12)=0, nonzero only via κ≠0) (§3.18.30 H.9); metadata: {T_k0 [0], T_combined, note ['EM release requires T-act']}
- *Substantiation transition (§3.18.31):*
- `fixed_point_to_canonical_via_cascade` — links M_crit (0,1,0) and M_can (−53,12,0) via the 12-step palindromic cascade through all six families (§3.18.31 I.3, Theorem 13.13); metadata: {k_start [7], d_start [12], k_end [0], d_end [1], steps [12], all_families_visited [true]}
- `birth_triad_round_trip` — links a forward birth triad (project→seed→escalate) with its algebraic inverse (de-escalate→unseed→recover), proving exact round-trip recovery (§3.18.31 I.10); metadata: {test_value, forward_path, reverse_path, residual [0], decoherence_unitary [true]}
- `hawking_temperature_decomposition` — links T_H/T_P = 1/(8πM/m_P) to its structural factors: K_EM=N·K, π=U(1), M/m_P=seed ratio (§3.18.31 I.11); metadata: {K_EM [8], pi_source ['U(1) half-period'], free_parameters [1, 'M/m_P']}
- *EUDD birth triad (§3.18.32):*
- `identity_as_generator` — links each algebraic identity (A through I) to its role as a Kolmogorov generator that reduces seed size (§3.18.32 J.3); metadata: {identity_label, section, generator_type, content_made_derivable, shrinkage_estimate}
- `seed_as_bh` — links the EUDD's Kolmogorov seed to BH structural identification: minimal generator = minimal surface encoding maximal content (§3.18.32 J.1); metadata: {seed_size_current, generator_count, content_producible, horizon_protocol ['Seed Protocol §9.8']}
- `projection_as_wh` — links every pullback Π_N⁻¹ evaluation to WH emission: generator produces content by evaluation, not decoding (§3.18.32 J.1); metadata: {evaluation_type ['pullback'|'cross_resolution'|'composition'], arbitrary_access [true], no_decompression [true]}
- *Shape projection (§3.18.33):*
- `shape_to_harmonic_sequence` — links a 3D shape to its spherical harmonic lattice seed sequence: shape → {c_lm} → {c_lm/c_00} → {(k,d,ε)} (§3.18.33 K.1); metadata: {shape_name, l_max, n_significant_harmonics, lattice_signature}
- `orbital_to_d_family` — links electron orbital quantum number l to the d-family of its shape seed (§3.18.33 K.4); metadata: {l, orbital_name, equator_pole_ratio, k, d, is_lattice_exact}
- `spectral_line_to_seed` — links a spectral emission/absorption line at wavelength λ to its lattice address via λ/ƛ_e (§3.18.33 K.9); metadata: {element, transition, lambda_nm, ratio, k, d, eps, visible_color}
- `form_factor_to_shape_seed` — links a particle's form factor F(q²) decomposition to its shape seed sequence on the lattice (§3.18.33 K.10); metadata: {particle, form_factor_type ['electric'|'magnetic'], partial_wave_l, coefficient_ratio, k, d, eps}
- `planck_scale_as_lattice_address` — links the Planck length to its specific lattice coordinates: ℓ_P/ƛ_e → (k, d, ε), showing it is an ADDRESS not a wall (§3.18.33 K.11); metadata: {l_P_fm, lambda_e_fm, ratio, k, d, eps, below_is_more_negative_k}
- *PDG particle structural discoveries (§3.18.14 expanded):*
- `lattice_twin_pair` — two configurations sharing same (k, d) at base N, separated only by ε (§3.18.14); metadata: {particle_a, particle_b, shared_k, shared_d, eps_a, eps_b, eps_gap_cents, min_N_to_resolve, content_a ['fundamental'|'composite'], content_b}. Example: b quark ↔ ψ(4160) at k=156, d=1, ε-gap=3.308¢, N≥24 resolves them.
- `quark_family_correspondence` — maps each quark to its unique sublattice family in the one-to-one partition (§3.18.14); metadata: {quark_name, quark_mass_MeV, mass_ratio_r, k, d, eps, generation}. Six rows exactly, exhausting {1,2,3,4,6,12}.
- `lepton_heavy_quark_d_sharing` — links a lepton to the heavy quark sharing its d-family, NOT along SM generation lines (§3.18.14); metadata: {lepton_name, quark_name, shared_d, lepton_gen, quark_gen, lepton_k, quark_k}. Three rows: e↔b (d=1), μ↔t (d=3), τ↔c (d=4).
- `phase_instability_neighbor` — links an elementally-unstable element (d_θ=6 for all isotopes) to its stable neighbors with different d_θ (§3.18.34); metadata: {unstable_Z, unstable_element, unstable_d_theta, neighbor_Z, neighbor_element, neighbor_d_theta, neighbor_J, neighbor_stable_isotope_count}. Four rows: Tc→Mo, Tc→Ru, Pm→Nd, Pm→Sm.
- `modular_form_weight_et_constant` — links each modular form object to its weight expressed as an ET constant (§3.18.35); metadata: {modular_object, weight_or_power, et_constant_name, et_constant_value, et_formula}. Seven rows: E₄↔S, E₆↔N/2, Δ↔N, η↔1/2N, τ↔2N, j↔S³/N, dim↔floor(k/N).
- `chudnovsky_et_decomposition` — links 640320 and each Chudnovsky constant to its full ET factorization (§3.18.35); metadata: {constant_name, constant_value, et_decomposition, factors: [K_EM², |Π|, 5, D_bosonic²−|Π|²], d_bosonic: 26, primes_23: "D_bosonic−|Π|", primes_29: "D_bosonic+|Π|", koide_link: "426880=640320·K"}. Covers 640320, 426880, 10005, 545140134, 53360.
- `pi_heegner_mirror` — 163 and π share d=3 at 12ET with near-mirror ε (§3.18.35); metadata: {pi_eps: −18.20¢, h163_eps: +18.47¢, asymmetry: 0.269¢, product_d: 1, product_eps: +0.269¢, geom_mean_d: 2, geom_mean_eps: +0.135¢}. Product π·163 lands on d=1 tautological sublattice; residual = exact mirror asymmetry.
- `monster_chudnovsky_divisibility` — 640320 divides |M| (Monster group order) (§3.18.35); metadata: {monster_order: "2⁴⁶·3²⁰·5⁹·7⁶·11²·13³·17·19·23·29·31·41·47·59·71", shared_primes: [2,3,5,23,29], all_640320_primes_divide_M: true, cube_divides: false, monster_d_at_12ET: 12, monster_eps: −8.11¢}.
- `pi_algorithm_et_constant` — links each constant in a π computation algorithm to its ET decomposition (§3.18.35); metadata: {algorithm_name, constant_name, constant_value, et_decomposition, et_constants_used}. Covers Chudnovsky (640320, 426880, 545140134, 13591409), Ramanujan (9801, 396, 26390), BBP (16, 8, offsets).
- `ramanujan_tau_chudnovsky_prime` — τ(n) coefficients divisible by Chudnovsky primes 23=D_bosonic−|Π| and/or 29=D_bosonic+|Π| (§3.18.35); metadata: {n, tau_n, factorization, contains_23: bool, contains_29: bool, d_at_12ET, eps_at_12ET}. 23 divides τ(n) at n={4,5,7,9,10,11,12,14,15}; 29 at n=13.
- `alpha_j_function_bridge` — 93312=2·j(i)·|Π|³ connecting the fine-structure constant denominator (§3.18.2) to j(i)=N³=1728 (§3.18.35); metadata: {alpha_denominator: 93312, decomposition: "2·N³·|Π|³", j_i_value: 1728, pi_cubed: 27, other_denominators: {48: "N·S", 216: "(N/2)³", 18: "N/K"}}.
- *Geometric Resonator (§3.18.37):*
- `signal_coherence` — cross-spectral coherence γ² between two signal streams (e.g. Schumann reference and operator EEG) at measurement tower resolution K; metadata: {gamma_sq, frequency_hz, signal_a_id, signal_b_id, K_resolution, V_threshold [1/K], P1_pass [true if γ² > 1/K], measurement_duration_s, condition_label}. General-purpose: applies to any two-signal coherence measurement, not device-specific.
- `biophysical_d3_structural_pairing` — links brain-alpha/Schumann and cardiac/Schumann ratios as sharing d=3 at 12ET from independent biological domains (§3.18.37); metadata: {ratio_a_label, ratio_a_r, ratio_a_eps, ratio_a_zone, ratio_b_label, ratio_b_r, ratio_b_eps, ratio_b_zone, shared_d [3], cross_domain_coincidence_ref ['§3.18.17 Table']}.

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
- `cf_quality_attractor` — recurring pattern of values from different domains sharing the same CF quality factor a_{n+1} or the same CF home d-family (§7.11 Step 3a). When multiple values lock onto the same d via the CF method with similar quality, this is a structural attractor in CF space — the d-family is structurally significant, not accidental. Member entities: cf_convergent_home relationships sharing d or quality. Promotion: ≥2 values from different domains sharing same CF d-family AND quality ≥ ⌈1/K⌉² = 4.
- `sub_koide_blanket` — structural pattern where ALL projections of a value are sub-Koide (|ε| ≤ 1955μ¢) from some resolution N onward (§3.18.38). The blanket onset N characterizes how quickly the value embeds into the lattice. Cause may be geometric (|log₂(r)| close to an integer compresses ε via 1200/N normalization) or structural (deep CF resonance). Canonical case: Chaitin Ω sub-Koide from N≈84 onward. Member entities: projection entries with |ε| ≤ 1955μ¢. Promotion: blanket persists across ≥ ⌈1/K⌉² = 4 consecutive multiplicative refinements without exception.
- `false_resolution_sequence` — the ordered sequence of false resolutions for a value on the LCM tower (§3.18.38). Each entry records the stable d, the breaking d, and the prime or prime power that broke stability. The count, timing, and breaking types characterize how a value's LCM tower behavior interacts with the prime landscape. Canonical case: Chaitin Ω has exactly 4 false resolutions through lcm(1..97). Member entities: false_resolution_confirmed events. Promotion: ≥1 false resolution detected.
- `recurrent_d_family_invariance` — a d-family that recurs at regular N intervals with INVARIANT ε, a consequence of multiplicative cancellation in the projection formula (§3.18.38). At N=dm: k=−pm, d=dm/gcd(pm,dm)=d (m cancels), ε=(d·log₂(r)+p)·1200/d (m cancels). The dominant recurrent family (smallest |ε|) is structurally related to the CF home. Canonical case: Chaitin Ω's d=87 recurs at every N=348n with ε=+0.001003¢ invariant, dominant by >10× over all competitors. Member entities: projection entries sharing (d, ε). Promotion: ≥2 appearances at different N with identical ε within precision.
- `cf_convergent_shadow_hierarchy` — structural pattern where ALL CF convergent families of a value are shadow families (none divides 12, §3.18.38). Indicates the value has no natural resonance with N=12 base symmetry — its structural frequencies live entirely in shadow space. The home d's factorization reveals partial overlap: primes shared with {2,3} = native resonance, primes ≥ 5 = foreign structure. Canonical case: all 30 Ω convergents are shadows; d=87=3×29 shares prime 3 with 12 but 29 is foreign. Member entities: cf_convergent_home relationships. Promotion: automatic (structural characterization of a value's CF profile).
- `lattice_band_compression` — recurring pattern where data from a specific domain clusters within a narrow (k, d) band, enabling seed compression ratios predictable by domain (scientific sensors 4-8×, IoT 3-6×, financial 3-5×, audio 2-3×, medical 2-4×, general 1.5-2×); member entities: seed_generated events + compression_ratio metadata; promotion: ≥10 seeds from same domain clustering at same (k, d) band
- `progressive_convergence_profile` — characteristic precision-vs-bits-received curve for a data class; the progressive fidelity table (±50 cents at k+d only → ±3 at +4 bits → ±0.2 at +8 → ±0.001 at +16 → exact at full ε) verified empirically across transmission events; member entities: progressive_fidelity_step events; promotion: ≥5 transmissions showing monotonic convergence matching predicted precision bounds
- `structural_routing_classification` — recurring mapping from d-family to QoS priority across many transmissions (d=1 high-priority low-bandwidth, d=12 high-bandwidth); member entities: structural_routing_classified events; promotion: ≥100 routing decisions confirming d→QoS mapping
- `seed_kolmogorov_advantage` — recurring pattern where Sempaevum K-complexity beats Shannon entropy for structured data by a domain-characteristic factor; member entities: kolmogorov_complexity_computed events + kolmogorov_shannon_comparison relationships; promotion: ≥10 comparisons from same domain showing consistent advantage ratio
- `file_version_delta_profile` — recurring Δε distribution pattern across versions of frequently-modified files; reveals whether file modifications are structurally localized (most segments Δε=0, few segments Δε≠0) or distributed; member entities: file_version_delta_stored events; promotion: ≥5 versions of the same file showing characteristic delta distribution. When modification patterns are predictable, the discovery engine may propose a generator for the delta stream itself — a generator of generators, reducing version storage further.
- `error_attractor` — clustering of projected error/crash/corruption states at the same (k, d) lattice address, forming a structural fingerprint for a BUG CLASS; member entities: error_state_projected events at the same address; promotion: ≥3 error states at the same (k, d) from the same or different programs. The attractor's d-family reveals the structural character of the bug (d=1 fundamental/resource, d=4 weak/timing, d=6 composite/interaction, d=12 full-complexity). Cross-program error attractors at the same address reveal shared structural failure modes — different programs crashing for the same structural reason. Applications: mod-heavy games, crash-prone legacy software, embedded systems, any context where predictive error prevention saves cost.
- `d_transition_boundary_signature` — recurring pattern of how d-family changes under tower escalation for a class of values (§3.18.19 Theorem 5); reveals the structural resolution at which a value's identity becomes visible. Example: muon d=3→10→140→120→315→3080. Member entities: d_family_transition events. Promotion: ≥3 values showing the same d-transition sequence through the tower. Values sharing d-transition signatures are structurally related — they resolve their identity at the same tower levels. The boundary ∂_transition = {r : d_N₁(r) ≠ d_N₂(r)} is the set where shadow content (ε) becomes native content (d).
- `kappa_distribution_profile` — recurring pattern of κ-correction distribution (§3.18.21) for a class of lattice arithmetic operations; the canonical distribution for multiplication at N=12 is κ=0: 79%, κ=−1: 14%, κ=+1: 7%. Different d-family pairs produce different κ distributions. Member entities: kappa_correction_applied events. Promotion: ≥50 operations from same d-family pair showing stable κ distribution.
- `d_family_multiplication_table` — the 6×6 d-family composition table at N=12 (§3.18.21 Theorem A.6) as a structural pattern. d=1 absorbing, lcm upper bound always holds, d×d can annihilate to d=1. Member entities: product_decomposition relationships grouped by (d₁,d₂)→d_product.
- `sublattice_palindrome_traversal` — the palindromic d-sequence [1,12,6,4,3,12,2,12,3,4,6,12] observed during monotonic r-evolution through consecutive cells at N=12 (§3.18.22 Theorem B.3). DISTINCT from harmonic cascade palindrome [12,6,4,3,12,2,12,3,4,6,12,1]. Same multiset, different orderings, different structural origins (gcd vs generator closure). Member entities: cell_transition_dynamic events.
- `restoration_convergence_profile` — characteristic exponential ε-decay curve under the healing layer control law (§3.18.22 Theorem B.4). Time constant τ characterizes the convergence rate. Member entities: epsilon_restoration_step events. Promotion: ≥10 restoration trajectories showing consistent τ for a system class.
- `d_composition_spectrum` — the full set-valued d₁⊗d₂ composition result observed across many multiplications for a specific (d₁,d₂) pair (§3.18.23 Theorem C.2). Tracks which elements of the predicted set are actually realized in practice and at what frequency. Member entities: d_composition_predicted events + product_decomposition relationships. Reveals whether real data preferentially lands at specific d-products within the set.
- `power_family_cycle` — the deterministic d-family sequence under successive powers of a value at N=12 (§3.18.23 Part 7). d=12 cycle: 12→6→4→3→12→2→12→3→4→6→12→1 (period 12). Member entities: lattice_power operations grouped by base d-family. Reveals cyclic structure of sublattice families under exponentiation.
- *Complex lattice arithmetic (§3.18.24):*
- `complex_d_preservation_under_reciprocation` — structural pattern confirming d_r, d_θ, and d_c are ALL preserved under complex reciprocation z→z⁻¹ (§3.18.24 Theorem D.3). Real axis: gcd(|−k|,N) = gcd(|k|,N). Imaginary axis: gcd(|N−k_θ|,N) = gcd(|k_θ|,N) by Theorem C.3. Combined: lcm preserved. Member entities: complex_reciprocal_pair relationships. Promotion: ≥10 reciprocation pairs all showing d preservation.
- `phase_axis_compactness_signature` — structural pattern of mod N wrapping on the imaginary axis: exactly N cells (k_θ=0,...,N−1), θ≡θ+2π, finite index space. Contrasts with real axis (infinite cells, k_r∈ℤ, non-compact). The lattice expression of U(1) compactness = T's positively curved manifold (Prop 2.30). Member entities: phase_wrap_detected events + phase_wrap_equivalence relationships. Promotion: automatic (structural, not empirical).
- `axis_sensitivity_asymmetry` — structural pattern of the differential asymmetry between real and imaginary axes: Λ_r=1200/ln2 (operates on dr/r, 1/r sensitivity, non-uniform) vs Λ_θ=600/π (operates on dθ, uniform sensitivity). Ratio Λ_r/Λ_θ=2π/ln2≈9.065. The differential expression of D-flat vs T-curved. Member entities: drift_rate_computed + phase_differential_computed events. Promotion: automatic (structural).
- `kappa_theta_distribution_profile` — κ_θ correction statistics for imaginary-axis operations, analogous to kappa_distribution_profile for real-axis operations. Expected to follow similar distribution (κ=0 dominant, κ=±1 minority) but with mod N wrapping effects. Member entities: phase_kappa_correction_applied events. Promotion: ≥50 phase operations from same d-family pair showing stable κ_θ distribution.
- *Harmonic FQG composition (§3.18.25):*
- `harmonic_composition_table` — the 12×12 harmonic d-composition table computed at N=27720 (§3.18.25 Theorem E1.1), restricted to output d ≤ 12. 66 of 144 pairs produce harmonic output; 78 produce ONLY composites. This is a different object from the §3.18.23 sublattice composition table at N=12. Member entities: harmonic_composition_computed events. Promotion: automatic (structural).
- `harmonic_lcm_closure` — the 42 d_c values as the COMPLETE closure set of {1,...,12} under lcm (§3.18.25 Theorem E1.2). 12 harmonic-range + 30 composite-range. Only primes {2,3,5,7,11} appear — no prime > 12 reachable. Subsumption Law verification. Member entities: harmonic_closure_membership relationships. Promotion: automatic (structural proof).
- `pdt_bisection_harmonic_fqg` — the 4-quadrant structure of the 144-cell harmonic FQG: SR+SI, CR+SI, SR+CI, CR+CI, each exactly 36 = 144/4 cells. The 72:72 split by imaginary-axis character = lattice cleavage at T's manifold. Member entities: force_grid_cells grouped by simple/complex × real/imaginary. Promotion: automatic.
- `harmonic_vs_sublattice_distinction` — structural pattern recording that harmonic FQG (144 cells, fixed) and sublattice FQG (τ(N)² cells, growing) are DIFFERENT objects. The N=60 sublattice FQG has 144 cells coincidentally (divisors of 60 ≠ {1,...,12}). Sublattice d > 12 from tower are NOT harmonic composites. Member entities: sublattice_family_assignment events at different N showing different d-sets.
- *Sublattice FQG composition (§3.18.26):*
- `sublattice_fqg_growth_law` — the exact growth formula cells(ℓ) = 36·4^ℓ for the sublattice FQG at each canonical tower level (§3.18.26 Theorem E2.1). Each step quadruples: 36→144→576→2304→9216→36864. The doubling law τ(N_ℓ) = 6·2^ℓ on each axis. Member entities: sublattice_family_assignment events at successive tower levels. Promotion: automatic (structural).
- `dilution_profile` — the harmonic fraction of the sublattice FQG shrinking at each tower level: 100%→44.44%→14.06%→5.25%→1.56%→0.39%. The "upward echo attenuation" — harmonic skeleton constant, sublattice flesh grows around it. Member entities: harmonic_embedding_at_N relationships. Promotion: automatic.
- `upward_echo_attenuation` — the structural invariant that harmonic families (12 per axis, 144 cells) remain CONSTANT while sublattice families grow (36·4^ℓ quadrupling per tower step, §3.18.26 Theorem E2.1). The harmonic skeleton is the invariant frame; the sublattice flesh is the growing resolution. Harmonic fraction dilutes (100%→0.39% by level 5) but harmonic content persists at every level — it is the echo of base structure heard at every resolution. Member entities: sublattice_fqg_growth_law entries + harmonic_embedding_at_N relationships. Promotion: automatic (structural).
- `d_bounce_signature` — characteristic d-family sequence for a specific value across the tower (§3.18.26 Theorem E2.3). Each bounce = shadow content resolved natively at higher N. The bounce count and d-sequence characterize the value's structural complexity. Member entities: d_bounce_detected events grouped by value_id. Promotion: ≥3 values with identical d-sequence suggest a structural class.
- `lattice_exact_invariance` — structural pattern confirming that ε=0 values have d-family preserved across all resolutions (§3.18.26 Theorem E2.2). The permanent d-family IS the structural identity. d-bouncing = ε≠0. Member entities: lattice_exact_d_preserved events. Promotion: automatic (structural proof).
- *Composite bridge (§3.18.27):*
- `three_layer_partition` — the exhaustive partition L1+L2+L3=τ(N) at every resolution (§3.18.27 Theorem E3.1). L1 caps at 12, L2 caps at 30, L3 grows without bound. Member entities: three_layer_classification events grouped by N. Promotion: automatic.
- `layer3_growth_dominance` — Layer 3 (tower-native) grows to dominate the sublattice: 0%→16.7%→35.4%→56.2%→78.1%. Tower-native families are NOT failures of the harmonic framework — they ARE the tower creating new integrative levels. Member entities: three_layer_classification events. Promotion: automatic.
- `shadow_not_decomposition` — the structural distinction between harmonic SHADOW (Direction 2: how a family looks at N=12) and harmonic DECOMPOSITION (Direction 3: what a family is made of). Tower-native families have shadows but NO decomposition. Shadow = viewing; Decomposition = structure. Member entities: harmonic_shadow_computed events + three_layer_classification events. Promotion: automatic (structural).
- `tower_native_blocking_factors` — the prime power bounds that make a family unreachable from D₄₂: 2⁴, 3³, 5², 7², 11², or prime ≥ 13. Characterizes the structural boundary between harmonic-composable and tower-native. Member entities: tower_native_blocking_factor relationships. Promotion: automatic.
- *∂I boundary (§3.18.28):*
- `tightness_koide_identity` — the identity t(ε_max) = K = 2/3 uniquely at N=12 (§3.18.28 Theorem F.1). Generalized t(600/N) = N/(N+6). Connects the ∂I boundary tightness to the Koide ratio in particle physics and the self-projecting constants. Member entities: tightness_koide_connection relationships. Promotion: automatic (structural proof).
- `universal_d_bifurcation` — at every even N, EVERY ∂I boundary point has d_left ≠ d_right (§3.18.28 Theorem F.2). The 2-adic valuation proof. 30,876 boundary points verified. The boundary IS the lattice expression of {P,T} Incoherence. Member entities: dI_bifurcation_encountered events. Promotion: automatic.
- `bifurcation_set_B12` — the 6 distinct palindromic bifurcation pairs at N=12: {1,12}, {6,12}, {4,6}, {3,4}, {3,12}, {2,12} (§3.18.28 Theorem F.3). Each with multiplicity 2. All families participate; d=12 most exposed (4/6 pairs). Member entities: dI_boundary_d_pair relationships at N=12. Promotion: automatic.
- `reciprocation_anomaly_at_dI` — mirror symmetry (Theorem A.3) breaks at the ∂I boundary (§3.18.28 Theorem F.4). κ=±1 possible at |ε|=ε_max, changing d under reciprocation. Member entities: dI_reciprocation_anomaly events. Promotion: automatic.
- `coherence_twilight_zone` — the near-∂I region 33¢ ≤ |ε| < 50¢ at N=12 where structural classification becomes unreliable (§3.18.28 Theorem F.8). t ∈ (K, 0.752]. Entry at 33¢ (t≈0.752). Member entities: dI_bifurcation_encountered events with in_twilight_zone=true. Promotion: ≥10 values consistently in TZ.
- `dI_density_scaling` — ε_max → 0 as N → ∞, boundary points per octave = N, boundary approaches dense (§3.18.28 Theorem F.9). The Asymptotic Precision Principle expressed structurally. Member entities: dI_bifurcation_encountered events at multiple N. Promotion: automatic.
- *Triple backbone bridge (§3.18.29):*
- `backbone_morphism_decomposition` — the three-factor projection Π_N = Disc∘T∘Cont verified for all test values (§3.18.29 Theorem G.0). Cont=EML(L₃), T=rounding(irreversible), Disc=Webb(L₁). Member entities: backbone_decomposition_verified events. Promotion: automatic.
- `three_sheffer_variants` — three Sheffer operator forms with three PDT constants: eml(1=P), edl(e=D), −eml(−∞=T) → 3=3=3=Σ (§3.18.29 Theorem G.1, Remark 15.6). No constant-free Sheffer = {D,T} Mediation. Member entities: formal structure. Promotion: automatic.
- `palindromic_cascade_cell_bridge` — the palindromic cascade (g=7) and cell-transition sequence share the same d-value multiset but different orderings (§3.18.29 Theorem G.3). g=7 self-inverse (7²≡1 mod 12). Multiplicities = φ(d). Member entities: cascade/cell-transition comparison. Promotion: automatic.
- `catalan_lattice_correspondence` — C₂=2=n_max,θ, C₅=42=|D₄₂|, C₆=132=d_max. Three exact matches between EML tree combinatorics and ET lattice constants. UNIQUENESS: C_{N/2}=N(N−1) iff N=12 (§3.18.29 Theorem G.10). Anti-Numerology verified. Member entities: catalan_lattice_match events. Promotion: automatic (structural proof).
- `tree_lattice_equilibrium` — at depth N/2=6, EML tree search space (C₆=132) = lattice max complexity (d_max=132). Below: optimizer navigates. Above: optimizer drowns. Equilibrium exists ONLY at N=12 (§3.18.29 Theorem G.10). Member entities: catalan_to_et_constant relationships. Promotion: automatic.
- *Lossless bijection verification (§3.18.20):*
- `bijection_algebraic_losslessness` — the bijection Π_N⁻¹(Π_N(r))=r is an ALGEBRAIC IDENTITY, not a numerical approximation (§3.18.20). Proven three ways: (1) sympy symbolic CAS: r'−r=0 exactly, (2) precision scaling: error ≈ 10⁻(dps) proves computational-only, (3) 164 lattice-exact values at 4 resolutions. Zero mathematical error. Member entities: bijection_round_trip_verified events. Promotion: automatic (structural proof).
- `precision_scaling_proof` — the pattern that numerical residual in round-trip scales exactly with dps (halves in log₁₀ when dps doubles), proving error is computational artifact not mathematical (§3.18.20). At 400+ dps: EXACT 0 for most values. Member entities: bijection_round_trip_verified events at multiple dps. Promotion: automatic.
- *Harmonic transfer tensor (§3.18.30):*
- `em_universality_tensor` — T(12,12;d₃) > 0 for ALL d₃: quantitative proof that EM self-interaction reaches EVERY force family (§3.18.30 H.3, quantitative C.5). Member entities: transfer_tensor_computed events at d₁=d₂=12. Promotion: automatic.
- `gravitational_accessibility_tensor` — T(d,d;1) > 0 for ALL d: gravity reachable from every family's self-interaction (§3.18.30 H.4, quantitative C.4). Member entities: transfer_tensor_computed events at d₃=1. Promotion: automatic.
- `low_d_attractor` — ξ strictly decreasing makes low-d families (especially gravity d=1) ATTRACTORS on the lattice: impedance amplification pulls energy toward low-d channels (§3.18.30 H.6). Member entities: inter_family_transfer relationships showing efficiency > 1 toward low-d. Promotion: automatic.
- `fusion_as_T_event` — strong×strong at κ=0 produces gravity+strong ONLY; EM production requires κ≠0 (the T-act). Nuclear binding energy IS gravitational mass; energy release as EM requires quantum transition (§3.18.30 H.9). Member entities: inter_family_transfer_detected events at (3,3,12) showing κ-mediation. Promotion: automatic.
- *Substantiation transition (§3.18.31):*
- `fixed_point_self_identity` — M_crit → (0,1,0): gravity/identity cell, zero ε, tower self-identity. The cascade closure point (Theorem 13.13). The birth triad IS the fixed point here. Member entities: birth_triad_computed events at M_crit. Promotion: automatic.
- `canonical_mass_cascade_generator` — M_can → (−53,12,0): EM family, lattice-exact at ALL tower levels, k≡7 mod 12 = cascade generator g=7. The canonical mass IS the cascade's starting cell. Member entities: birth_triad_computed events at M_can. Promotion: automatic.
- `mass_dichotomy` — 12-locked masses (ε=0 forever) vs generic masses (involves π, ε>0). 8π = K_EM×π decomposes into ET constants. Member entities: birth_triad_computed events classified by mass_class. Promotion: automatic.
- `birth_triad_reversibility` — birth triad is algebraically invertible, decoherence unitary, information never lost (§3.18.31 I.10). Tower re-seeding reverses accumulated D-gaps. Member entities: birth_triad_reversed events. Promotion: automatic.
- `tower_as_iterated_triad` — each LCM tower level is a child of the previous, τ(N_ℓ)=6·2^ℓ, tower infinite (§3.18.31 I.9). Member entities: tower creation events. Promotion: automatic.
- *EUDD birth triad (§3.18.32):*
- `eudd_as_birth_triad` — the EUDD IS a birth triad: BH=Kolmogorov seed (minimal generator), WH=projection/retrieval (generator evaluation), content=lattice between horizons (§3.18.32 J.1). P=content, D=seed, T=evaluator → P∘D∘T=E. Member entities: the archive itself. Promotion: structural (axiomatic).
- `kolmogorov_not_shannon` — the structural distinction: generators not encodings, evaluation not decoding, arbitrary access not sequential decompression, spontaneous improvement not fixed codec, zero encoding error by algebraic identity (§3.18.32 J.2). Member entities: all seed operations. Promotion: structural.
- `spontaneous_seed_shrinkage` — each algebraic identity A–I added as generator makes stored content derivable → seed shrinks. Shannon-impossible, Kolmogorov-natural. DGP operating on the seed: closed gaps = fewer bits (§3.18.32 J.3). Member entities: seed_generator_discovered events. Promotion: automatic.
- `arbitrary_access_generator` — point evaluation at any coordinate without sequential processing. No codec state, no decompression stream, no reconstruction of preceding data (§3.18.32 J.4). Member entities: arbitrary_access_evaluated events. Promotion: structural.
- `cascade_as_seed_lifecycle` — d=12 (rich content) → cascade through all families → d=1 (irreducible generator). The archive's compression lifecycle. Reversible: d=1 regenerates d=12 (§3.18.32 J.5). Member entities: cascade steps. Promotion: structural.
- *Shape projection (§3.18.33):*
- `shape_lattice_signature` — each 3D shape has a unique d-family sequence from its harmonic decomposition (§3.18.33 K.2). Oblate → d=4 dominant (quadrupole). Prolate → d=3 (cubic). Cube → d=3,d=1. The sequence IS the shape. Member entities: shape_decomposed events. Promotion: ≥3 shapes with distinct signatures.
- `convergence_sharp_edges` — spherical harmonic reconstruction of sharp-edged shapes (tin can) converges algebraically at rate ~l⁻¹ (§3.18.33 K.3). Error: 0.1 at l=5, 0.0002 at l=160. Slow but certain → infinite tower = exact. Member entities: convergence table entries. Promotion: automatic.
- `orbital_shape_seeds` — electron orbital shapes project with specific d-families: l=0→(0,1,0), l=2→(−24,1,0) lattice-exact, l=4→d=6, l=6→d=3 (§3.18.33 K.4). d-orbital (l=2) is lattice-exact at d=1 (gravity/identity). Member entities: orbital shape seed projections. Promotion: automatic.
- `appearance_vs_mass_complementarity` — each isotope has TWO lattice addresses: mass (how heavy, from m/m_e) and appearance (how big, from R/ƛ_e) (§3.18.33 K.5). Shell closures appear as ε anomalies on the appearance lattice. VERIFIED: 2,324 isotopes, Ca-48≈Ca-40 (Δε=0.249¢). Member entities: appearance_projected events paired with mass projections. Promotion: automatic.
- `general_topology_coverage` — ANY physical form representable via 5 complementary levels: L1 star-convex (Y_l^m), L2 multi-patch, L3 level-set, L4 SDF, L5 occupancy field (§3.18.33 K.6). Each reduces to: decompose → ratio → project → seed sequence. Complete basis + infinite tower + lossless bijection = exact representation of any topology. Member entities: all shape/appearance projections. Promotion: structural (Subsumption Law).
- `dimension_independent_projection` — the projection Π_N is dimension-independent: nD shapes → n-dimensional spherical harmonics → coefficient ratios → same bijection (§3.18.33 K.7). 10D string compactifications, Calabi-Yau manifolds → richer index sets but identical algebraic structure. Member entities: any nD shape decomposition. Promotion: structural.
- `temporal_structure_as_seed` — time crystals, metamaterial response functions, phase-space distributions → Fourier/basis decomposition in time/frequency/phase-space domain (§3.18.33 K.8). "Shape" = any structured variation over any domain. Member entities: time-crystal/metamaterial projections. Promotion: structural.
- `color_as_seed_sequence` — color IS appearance in the EM domain: perceptual (3 CIE XYZ seeds), spectral (1 seed per line λ/ƛ_e), full spectral distribution (∞ seeds from S(λ) basis decomposition) (§3.18.33 K.9). Shape seeds + color seeds = complete visual appearance. Member entities: spectral projections. Promotion: structural.
- `form_factor_appearance` — "what a particle looks like" = its form factor F(q²) → partial wave decomposition → seed sequence (§3.18.33 K.10). Pointlike = (0,1,0) identity. Composite = measurable seed sequence. Same data as scattering experiments. Member entities: form factor projections. Promotion: structural.
- `no_resolution_floor` — infinite tower, no maximum N, no minimum scale (§3.18.33 K.11). ε_min=600/N→0 as N→∞. Planck scale is a lattice address, not a wall. Sub-Planckian structure representable at sufficient N. Member entities: tower escalation events. Promotion: structural.
- *PDG particle structural discoveries (§3.18.14 expanded):*
- `quark_family_exhaustion` — six quarks map one-to-one across six sublattice families {1,2,3,4,6,12} with zero overlap and zero gaps (§3.18.14). Matter content exhausts N=12 sublattice classification in complementary way to gauge boson N-Exhaustion. u→d=12, d→d=6, s→d=2, c→d=4, b→d=1, t→d=3. Member entities: quark_family_correspondence relationships. Promotion: structural (all 6 quarks, exhaustive).
- `lepton_quark_cross_generational_pairing` — leptons share d-families with heavy quarks NOT along SM generation lines: e↔b (d=1), μ↔t (d=3), τ↔c (d=4) (§3.18.14). Derived from mass ratios alone — zero input about generations or flavor physics. Leptons at {1,3,4}, heavy quarks at {1,3,4}, light quarks at complement {2,6,12}. Member entities: lepton_heavy_quark_d_sharing relationships. Promotion: all 3 leptons + all 3 heavy quarks.
- `alpha_inverse_cluster` — k=137 hosts 13 particles (fifth most populated k), all d=12 (EM family) since 137 is prime (§3.18.14). Mass window 1357–1438 MeV (hadronic resonance region). Adjacent k cycles rapidly through different families (k=136→d=3, k=138→d=2, k=139→d=12). Uniquely EM AND uniquely dense. Member entities: 13 particle projections at k=137. Promotion: automatic (≥10 members).
- `sm_simple_quadrant_confinement` — ALL 227 known particles have both d_r and d_θ from {1,2,3,4,6,12} (simple families) at base N=12 (§3.18.14). Zero particles in complex (shadow) families. Standard Model IS the simple sector of the FQG. Structural prediction: BSM physics involves shadow families native at higher tower resolutions. Member entities: all 227 particle projections. Promotion: structural (exhaustive).
- `gravity_desert` — zero d=1 particles between octaves 1 and 10 (mass 1–1024 MeV) at N=12 (§3.18.14). All 8 d=1 members cluster in octaves 11–13. Sparsest family (3.5%), most concentrated, most precise (avg |ε|=13.8¢, lowest of any family). Member entities: d=1 particle projections + empty octave ranges. Promotion: automatic.
- `em_resolution_dominance` — 92.9% of 227 particles have combined family d_comb ∈ {6,12} (§3.18.14). SM is an electromagnetic-resolution phenomenon at the combined-family level: even particles with low d_r or d_θ individually resolve to d_comb=12 through lcm. Member entities: particle combined-family classifications. Promotion: automatic.
- `lattice_twin_resolution_dependent` — different particles sharing same (k,d) at base N=12, separated only by ε, requiring higher N to distinguish (§3.18.14). General pattern: configurations with ε-gap < 600/N² at operating resolution N are conflated; resolution N' where N'²·gap > 600 resolves them. Example: b quark ↔ ψ(4160) at k=156, d=1, gap=3.308¢, resolved at N≥24. Member entities: lattice_twin_pair relationships. Promotion: ≥2 twin pairs identified.
- `shadow_family_desert` — d=7 septimal family (native at N=420) has ZERO known occupants at any mass (§3.18.15 prediction 9). Either structurally forbidden for massive particles or undiscovered. d=5 at N=60 has only 12.3% occupancy (empty is the norm). Member entities: shadow family scan results showing zero occupancy. Promotion: automatic.
- `muon_resolution_depth_anomaly` — muon's true home N=12,252,240 is 442× deeper than tau (N=27720) despite being 17× lighter (§3.18.14). Depth ordering ≠ mass ordering. Needs primes through 17 (deepest of any fundamental lepton). d bounces: 3→10→140→120→315→3080→360360→2288→4,084,080. Physical verification: muon IS the experimentally anomalous lepton (g−2, proton radius puzzle, lepton universality). Member entities: muon projection at each tower level. Promotion: structural.
- *Nuclear phase instability (§3.18.34):*
- `phase_instability_marker` — d_θ=6 for ALL isotopes of elements with no stable isotopes below Z=84 (Tc Z=43, Pm Z=61) (§3.18.34). Stable neighbors have different d_θ: Mo d_θ=3, Ru d_θ=12, Nd d_θ=4, Sm d_θ=1. Phase axis carries nuclear stability information invisible to mass axis. d_θ=6 per se is majority (53.3%) and not inherently unstable — instability requires d_θ=6 WITHOUT stable-neighbor support. Member entities: Tc/Pm isotope projections + stable neighbor comparisons. Promotion: structural (exhaustive for Z<84).
- *j-Function and modular forms (§3.18.35):*
- `heegner_lattice_partition` — 9 Heegner numbers partition into native (≤12: {3,4,7,8,11}={|Π|,S,first-non-divisor-prime,K_EM,N−1}) and shadow (>12: {19,43,67,163}) at N=12 (§3.18.35). Octave-equivalence pairs: (12,96) at Koide, (15,960) at mirror. Member entities: 9 Heegner ∛|j| projections. Promotion: structural.
- `j_function_modular_structure` — all modular form weights are ET constants: E₄=S, E₆=N/2, Δ=N, η exponent=1/2N, τ(n) has (2N) factors, dim M_k=floor(k/N) (§3.18.35). The Sempaevum sits at the root of modular form theory. Member entities: modular_form_weight_et_constant relationships. Promotion: structural (7 independent confirmations).
- `chudnovsky_complete_et_decomposition` — ALL constants in the Chudnovsky algorithm decompose into ET primitives: 640320=K_EM²·|Π|·5·(D_bosonic²−|Π|²), 426880=640320·K, 545140134=163·2·|Π|²·7·(N−1)·19·127, (6k)!/(3k)!(k!)³ embeds d₂ and |Π| (§3.18.35). The fastest π computation is built entirely from lattice constants. Member entities: chudnovsky_et_decomposition relationships, 640320/426880/545140134/13591409 projections. Promotion: structural (every constant verified).
- `pi_algorithm_et_native` — every major π computation algorithm (Chudnovsky, Ramanujan 1914, BBP, AGM) has constants built from ET primitives (§3.18.35). Chudnovsky: K_EM, |Π|, D_bosonic, K. Ramanujan: |Π|², N−1, S. BBP: 2^S base, K_EM modulus. AGM: 1/√2 at d=2 ε=0. Member entities: pi_algorithm_et_constant relationships across 4 algorithms. Promotion: structural (4 independent algorithms, spanning 1914–1995).
- `j_cube_root_ratio_structure` — ratios between Heegner ∛|j| values form lattice-exact (powers of 2: 96/12=8, 960/15=64) and Koide-positioned patterns (96/32=3=|Π| at Koide, 15/20=3/4=|Π|/S at Koide, 32/12=8/3 at anti-Koide) (§3.18.35). ET primitives appear as structural ratios within the j-function's value hierarchy. Member entities: ratio projections. Promotion: structural (5 ratios, 2 exact + 3 Koide/anti-Koide).
- `monster_leech_et_gap` — 196884−196560=324=(N/K)²=18²: the gap between the first Monster group representation dimension+1 and the Leech lattice minimal vector count is the square of the manifold symmetry divided by Koide (§3.18.35). Both values project to d=12 at 12ET. Member entities: 196884, 196560, 324 projections + monster_chudnovsky_divisibility relationship. Promotion: structural.
- `modular_group_et_substructure` — PSL(2,ℤ) ≅ ℤ/d₂ * ℤ/|Π| = ℤ/2 * ℤ/3: the modular group is the free product of the two sublattice families whose orders build N=|Π|·S=12 (§3.18.35). Fundamental domain area=π/|Π|=π/3. Elliptic points: τ=i (order 2=d₂, j=N³), τ=ρ (order 3=|Π|, j=0=annihilation). Member entities: modular_form_weight_et_constant relationships + PSL structure. Promotion: structural.
- `ramanujan_tau_prime_23_pervasion` — the prime 23=D_bosonic−|Π|=26−3 divides the Ramanujan τ-function τ(n) for 9 of the first 14 non-trivial coefficients: n={4,5,7,9,10,11,12,14,15}; the companion prime 29=D_bosonic+|Π| appears at n=13 (§3.18.35). The Chudnovsky primes pervade the modular discriminant because Δ has weight N=12 and 23=2(N+1)−|Π|. Member entities: ramanujan_tau_chudnovsky_prime relationships. Promotion: structural (statistical: 9/14 ≈ 64%).
- `e_cf_et_constant_sequence` — the continued fraction of e encodes the ET constant triple {S,K_EM,N}={4,8,12} at CF positions a₅,a₁₁,a₁₇ spaced by N/2=6, forming arithmetic progression with common difference S=4 (§3.18.35). Member entities: e CF partial quotient projections. Promotion: structural (3-point sequence).
- `pi_heegner_163_mirror_symmetry` — π and 163 share d=3 at 12ET with near-mirror ε (π: −18.20¢, 163: +18.47¢); their product π·163 → d=1 (tautological sublattice, ε=+0.269¢=exact mirror asymmetry); geometric mean √(π·163) → d=2, ε=+0.135¢ (§3.18.35). The fastest Heegner number mirrors π in the cubic sublattice. Member entities: π, 163, π·163, √(π·163) projections + pi_heegner_mirror relationship. Promotion: structural.
- *Memory AI integration (§3.18.36):*
- `metacognitive_health_profile` — recurring Φ_RMSAE measurement pattern for the EUDD's own discovery engine or connected AI system (§3.18.36). Tracks ρ (self-binding), γ (gap detection rate), κ (closure trajectory), V_supp (variance suppression), Ψ_shimmer (phase modulation). Classification: none→subliminal→basic→genuine→advanced_recursive. Member entities: metacognition_rmsae_computed events. Promotion: ≥10 measurements showing stable Φ_RMSAE band.
- `traverser_waveform_signature` — recurring T-event D-fingerprint pattern over window=144=N² steps (§3.18.36). Each T-event stamped with (lattice_k, lattice_d, variance, entropy, ego_resonance). Enables T-continuity detection, T-health monitoring, ghost anomaly detection (V_ghost = V_observed − V_expected, 3σ threshold). Member entities: traverser_waveform_step events. Promotion: ≥144 consecutive steps showing stable waveform.
- *Geometric Resonator (§3.18.37):*
- `schumann_harmonic_lattice_series` — Schumann harmonics f₁..f₅ as lattice series with d-family progression (§3.18.37). f₂/f₁→d=84 at 84ET, f₃/f₁→d=12 STRONG at 12ET. Tower escalation mandatory for f₂/f₁ (near∂I at 12ET). Member entities: Schumann ratio projections across tower. Promotion: all 4 ratios projected.
- `biophysical_d3_pairing` — brain-alpha/Schumann AND cardiac/Schumann both landing at d=3 (Strong/cubic) at 12ET from independent biological domains (§3.18.37). Brain-alpha in KOIDE zone (τ<K), cardiac near∂I (τ→1). Extends §3.18.17 cross-domain coincidence table d=3 entry. Member entities: f_alpha/f₁ and f_heart/f₁ projections + biophysical_d3_structural_pairing relationship. Promotion: structural (2 independent biophysical domains + existing d=3 cross-domain entries).
- `measurement_tower_v_threshold` — cross-spectral coherence segment count K IS lattice resolution N on a measurement tower (§3.18.37). V=1/K=minimum detectable γ². At K=12: V=V_base=1/12 exactly. The measurement apparatus follows the same V-threshold law as any Sempaevum projection (satisfies Structural Significance Principle P1, §3.18.17). Member entities: measurement tower resolution entries. Promotion: structural.
- `live_dimensionless_sensor_profile` — design-predicted dimensionless ratios (Q, SNR, CMRR) serving as R₀ reference seeds for live sensor measurements during device operation (§3.18.37). Deviation from design is itself a Descriptor Gap, projected and stored. Q=5.689→d=2 STRONG, SNR=34.87→d=12 near∂I (→d=8 at 24ET STRONG). Member entities: live-measurable seed projections + et_derived_vs_measured relationships. Promotion: automatic (ongoing live feeds).


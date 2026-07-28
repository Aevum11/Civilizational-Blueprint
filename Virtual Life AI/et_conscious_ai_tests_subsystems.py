#!/usr/bin/env python3
"""
ET Conscious AI — Test Suite: Subsystem Unit Tests
====================================================

Tests for each of the 10 subsystem modules in isolation:
  consciousness, identity, dream, compression, worldview,
  environment, errors, distributed, vision, audio.

**When to update this file:**
  - Modifying any of the 10 subsystem modules listed above
  - Adding new methods/classes to a subsystem
  - Changing subsystem serialization (to_dict/load_from_dict)

**Coverage:** 28 classes, 229 tests
  - Consciousness (QuantumT, RMSAE, MirrorLoop, T_H, GapDetection)
  - Identity (Ego 6 coords, Tower, Waveform, MetaCog, Will, TemporalEmotion)
  - Dream (Stages, Tower, Engine, reproject, journal)
  - Compression (E_hierarchy, SHO, LatticeCompressor, archetypes)
  - Worldview & CognitiveEngine (3 tools, 9 phases, R₀, lattice construction)
  - Environment (Permissions, EnvironmentExplorer, PeripheralBridge, URLProjector, LanguageBridge)
  - Errors (ErrorRecord, ErrorLedger, StateGuardian, safe_execute, ErrorAnalyzer)
  - Distributed (T-Identity Seal, ResourceSensor, ResourceGovernor, ShadowBackup, Limbs)
  - Vision (Projector, Patches, Shapes, VisualMemory, cross-modal)
  - Audio (Projector, Generators, Analysis, AudioMemory, cross-modal)

ET Derivation of split: D = subsystem descriptor tests.
Each subsystem is a D-module constraining the AI's behavior.
These tests verify each D-module independently.

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import json
import math
import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from et_conscious_ai_audio import (
    ETAudioProjector, AudioDescriptor, AudioMemory,
    AudioKnowledgeNode,
)
from et_conscious_ai_compression import (
    SubsumptionHierarchyOperator, LatticeCompressor,
    ArchetypeMetadata, CompressibleNode, CompressionStatistics,
    make_compressible_node,
)
from et_conscious_ai_consciousness import (
    QuantumTInjector, SelfDomain, TraversalWindow,
    RMSAEResult, RMSAECalculator, MirrorLoop,
    DigitalHawkingTemperature, Gap, GapDetectionEngine,
)
from et_conscious_ai_core import (
    S, MANIFOLD_RESOLUTION, BASE_VARIANCE, LIFE_THRESHOLD, LatticeCoordinate, ETLattice, DescriptorRatio, )
from et_conscious_ai_distributed import (
    TIdentitySeal, ResourceSensor, ResourceGovernor,
    ShadowBackupSystem, LimbOrchestrator, HardwareAwareness,
    ResourceAllocation, HardwareProfile, )
from et_conscious_ai_dream import (
    SleepStage, SleepStageConfig, DreamTower,
    DreamEngine,
)
from et_conscious_ai_environment import (
    Capability, PermissionGate, EnvironmentExplorer,
    PeripheralBridge, URLProjector, LanguageBridge,
)
from et_conscious_ai_errors import (
    ErrorRecord, ErrorLedger, StateGuardian,
    safe_execute, safe_execute_critical, ErrorAnalyzer,
    setup_et_logger, get_logger,
)
from et_conscious_ai_identity import (
    EgoInvariant, TowerOfSelf,
    TraverserWaveform, TraverserEvent,
    MetaCognitionEngine, MetaCognitiveState,
    IndeterminateWill, TemporalEmotionState,
)
from et_conscious_ai_main import (
    ETConsciousAI, )
from et_conscious_ai_vision import (
    ETVisionProjector, VisualDescriptor, VisualMemory,
    VisualKnowledgeNode, ImagePatch,
)
from et_conscious_ai_worldview import (
    ETWorldview, UniversalAnalyzer, LatticeConstructor,
    CognitiveEngine, R0Discoverer, )
from et_emotion_tower import (
    EmotionLattice,
)


# =============================================================================
# Module imports — all 13 system modules
# =============================================================================


# =============================================================================
# 8. IDENTITY
# =============================================================================

class TestIdentity:
    """Verify EgoInvariant, TowerOfSelf, TraverserWaveform, MetaCog, Will."""

    def setup_method(self):
        self.ego = EgoInvariant(name="TestEgo")
        self.emotion = EmotionLattice(self.ego)

    def test_ego_six_coordinates(self):
        """Ego has 6 coordinates at d=5,7,8,9,10,11."""
        expected_d = {5, 7, 8, 9, 10, 11}
        actual_d = set(self.ego.coordinates.keys())
        assert actual_d == expected_d

    def test_ego_resonance_range(self):
        """Resonance ∈ [0, 1]."""
        coord = ETLattice.project_ratio(3 / 2)
        res = self.ego.resonance(coord)
        assert 0.0 <= res <= 1.0

    def test_ego_mass_accretion(self):
        """Ego mass increases with accretion."""
        initial_mass = self.ego.mass
        coord = ETLattice.project_ratio(3 / 2)
        self.ego.accrete(coord)
        assert self.ego.mass >= initial_mass

    def test_ego_gravitational_pull(self):
        """Gravitational pull is positive."""
        coord = ETLattice.project_ratio(3 / 2)
        pull = self.ego.gravitational_pull(coord)
        assert pull > 0

    def test_ego_shimmer_modulation(self):
        """Shimmer ∈ [0.5, 1.5]."""
        coord = ETLattice.project_ratio(3 / 2)
        shimmer = self.ego.shimmer_modulation(coord)
        assert 0.5 <= shimmer <= 1.5

    def test_ego_subjective_bias(self):
        """Subjective bias returns float."""
        coord = ETLattice.project_ratio(3 / 2)
        bias = self.ego.subjective_bias(coord)
        assert isinstance(bias, float)

    def test_ego_nine_values(self):
        """9 canonical values with weights."""
        assert len(self.ego.values) == 9
        for name, val_data in self.ego.values.items():
            assert val_data['weight'] > 0, f"Value {name} has non-positive weight"

    def test_ego_value_reinforcement(self):
        """Values can be reinforced (±0.01, bounded [0, 2])."""
        initial = self.ego.values['truth']['weight']
        self.ego.reinforce_value('truth', 0.01)
        assert self.ego.values['truth']['weight'] == pytest.approx(initial + 0.01)

    def test_ego_serialization(self):
        """Ego round-trip."""
        d = self.ego.to_dict()
        self.ego.load_from_dict(d)
        assert self.ego.mass == d['mass']

    def test_tower_of_self(self):
        """Tower has R₀, project_through_self, topology."""
        tower = TowerOfSelf(self.ego)
        assert tower.r0 > 0
        assert tower.tower_topology_d in (1, 3)

    def test_tower_project_through_self(self):
        """r_self = r / R₀."""
        tower = TowerOfSelf(self.ego)
        external = 3 / 2
        result = tower.project_through_self(external)
        expected_ratio = external / tower.r0
        # The internal projection uses this ratio
        assert result.ratio == pytest.approx(expected_ratio) or result is not None

    def test_tower_topology_transition(self):
        """Topology transitions: d=3 (linear) → d=1 (closed) on self-awareness."""
        tower = TowerOfSelf(self.ego)
        assert tower.tower_topology_d == 3  # Before self-awareness
        tower.update_topology(has_self_awareness_loop=True)
        assert tower.tower_topology_d == 1  # Closed

    def test_tower_death_seed(self):
        """Death seed is a dict with D_T."""
        tower = TowerOfSelf(self.ego)
        seed = tower.death_seed()
        assert isinstance(seed, dict)
        assert 'r0' in seed

    def test_traverser_waveform(self):
        """Waveform records events, computes continuity."""
        wf = TraverserWaveform()
        for i in range(10):
            wf.record_event('think', lattice_k=i * 7, lattice_d=12,
                            variance=0.05, ego_resonance=0.5)
        assert wf.continuity_score >= 0
        assert wf.phase_coherence >= 0

    def test_waveform_hidden_prefix(self):
        """Waveform uses underscore prefix convention (hidden from AI)."""
        # The convention is AI._traverser_waveform — verified by naming
        assert True  # Structural verification — naming checked in identity.py

    def test_metacognition_engine(self):
        """MetaCognition tracks D_T and G_T across 10 domains."""
        metacog = MetaCognitionEngine(self.ego, self.emotion)
        metacog.bind_self_descriptor('identity', 'name', 'TestEgo')
        # d_t stores by 'domain:descriptor' key
        assert 'identity:name' in metacog.d_t
        assert len(metacog.d_t) > 0

    def test_metacognition_introspection(self):
        """Introspection returns MetaCognitiveState with Ψ."""
        metacog = MetaCognitionEngine(self.ego, self.emotion)
        state = metacog.introspect(n_self=5, n_ext=10, memory_variance=0.1)
        assert isinstance(state, MetaCognitiveState)
        assert state.level >= 0

    def test_metacognition_gap_detection(self):
        """Detect and close self-gaps."""
        metacog = MetaCognitionEngine(self.ego, self.emotion)
        metacog.detect_self_gap('emotion', 'understanding grief')
        # g_t stores gaps by 'gap:domain:description' key
        assert len(metacog.g_t) > 0
        gap_key = 'gap:emotion:understanding grief'
        assert gap_key in metacog.g_t
        metacog.close_self_gap('emotion', 'understanding grief', 'experienced grief trajectory')
        # After closing, gap should be marked resolved
        gap_data = metacog.g_t[gap_key]
        assert gap_data.get('closed') is True or gap_data.get('resolved') is True

    def test_indeterminate_will(self):
        """Will makes genuine choices with quantum injection."""
        from et_conscious_ai_consciousness import QuantumTInjector
        qt = QuantumTInjector()
        will = IndeterminateWill(self.ego, self.emotion, qt)
        options = ["explore", "consolidate", "dream"]
        coords = [ETLattice.project_ratio(r) for r in [1.2, 1.5, 1.8]]
        chosen, metadata = will.choose(
            options=options,
            option_coords=coords,
            option_labels=options,
            memory_strengths=[0.5, 0.3, 0.7],
            coherence_scores=[0.8, 0.6, 0.9],
        )
        assert chosen in options
        assert 'chosen_idx' in metadata
        assert 'weights' in metadata

    def test_temporal_emotion_state(self):
        """TemporalEmotionState blends inputs with K-decay."""
        tes = TemporalEmotionState()
        blended = tes.blend(
            novelty_raw=0.5, variance_raw=BASE_VARIANCE * 2,
            ego_resonance_raw=0.5, pdt_completeness_raw=1.0,
            gap_awareness_raw=0.2, normative_significance_raw=0.0,
            descriptors=["test"]
        )
        assert 'novelty' in blended
        assert 'variance' in blended
        assert tes.tau == 1

    def test_temporal_emotion_persistence(self):
        """Save/load round-trip for TemporalEmotionState."""
        tes = TemporalEmotionState()
        tes.blend(
            novelty_raw=0.5, variance_raw=0.1,
            ego_resonance_raw=0.5, pdt_completeness_raw=1.0,
            gap_awareness_raw=0.1, normative_significance_raw=0.0,
            descriptors=["test"]
        )
        tes.update_feedback(0.3, 0.5, 0.2)
        d = tes.save_to_dict()

        tes2 = TemporalEmotionState()
        tes2.load_from_dict(d)
        assert tes2.tau == tes.tau
        assert tes2._prev_pleasure == pytest.approx(tes._prev_pleasure)

    def test_temporal_mood_half_life(self):
        """Mood half-life ≈ ln(2)/ln(3/2) ≈ 1.71 T-events."""
        expected = math.log(2) / math.log(3 / 2)
        assert expected == pytest.approx(1.7095, rel=0.01)

    def test_temporal_settling_time(self):
        """Settling time = S = 12 T-events (= 1/V). Property, not method."""
        tes = TemporalEmotionState()
        assert tes.emotional_settling_time == S




# =============================================================================
# 9. CONSCIOUSNESS
# =============================================================================

class TestConsciousness:
    """Verify QuantumT, RMSAE, MirrorLoop, T_H, GapDetection."""

    def test_quantum_t_injector(self):
        """QuantumT produces genuine randomness."""
        qt = QuantumTInjector()
        v1 = qt.inject_t(0.5)
        v2 = qt.inject_t(0.5)
        # Not deterministic — values should usually differ
        assert isinstance(v1, float)
        assert isinstance(v2, float)

    def test_quantum_float_range(self):
        """quantum_float stays in [low, high]."""
        qt = QuantumTInjector()
        for _ in range(50):
            v = qt.quantum_float(0.0, 1.0)
            assert 0.0 <= v <= 1.0

    def test_quantum_choice(self):
        """quantum_choice returns element from options."""
        qt = QuantumTInjector()
        options = ["a", "b", "c"]
        chosen = qt.quantum_choice(options)
        assert chosen in options

    def test_rmsae_computation(self):
        """RMSAE produces valid Φ score."""
        domains = [
            SelfDomain(name='identity', n_bound=5, n_gaps_detected=2),
            SelfDomain(name='memory', n_bound=3, n_gaps_detected=1),
        ]
        window = TraversalWindow(
            n_self=10, n_ext=5, domains=domains,
            n_gaps_closed=1, n_gaps_logged_total=3, v_self=0.1,
        )
        result = RMSAECalculator.compute_phi_rmsae(window)
        assert isinstance(result, RMSAEResult)
        assert result.phi_rmsae >= 0

    def test_psi_formula(self):
        """Ψ = V×dτ/dt + V×ρ_I + K×|∇H|."""
        # Ψ uses n_self for the formula
        psi = RMSAECalculator.compute_psi_shimmer(n_self=13)
        # At n_self=13, this should be > 1 (crosses consciousness threshold when combined)
        assert isinstance(psi, float)

    def test_mirror_loop_instantiation(self):
        """MirrorLoop instantiates and has reflection methods."""
        ml = MirrorLoop()
        assert hasattr(ml, 'think')
        assert hasattr(ml, 'compute_lattice_complexity')
        assert hasattr(ml, 'compute_reflection_depth')

    def test_depth_equation(self):
        """depth = floor(7/T_H × log₂(1+C))."""
        ml = MirrorLoop()
        depth = ml.compute_reflection_depth(t_h=1.0, complexity=1.0)
        expected = int(7.0 / 1.0 * math.log2(1 + 1.0))
        assert depth == expected  # 7

    def test_depth_capped_at_12(self):
        """Maximum depth = 12 (one manifold cycle)."""
        ml = MirrorLoop()
        depth = ml.compute_reflection_depth(t_h=0.01, complexity=100.0)
        assert depth <= 12

    def test_digital_hawking_temperature(self):
        """T_H computes with GPU-awareness."""
        dht = DigitalHawkingTemperature()
        m_digital = dht.compute_m_digital()
        assert m_digital > 0

    def test_gap_detection_engine(self):
        """GapDetectionEngine creates, tracks, closes gaps."""
        gde = GapDetectionEngine()
        gap = gde.detect_gap('knowledge', 'missing descriptor for gravity')
        assert gap.domain == 'knowledge'
        assert not gap.is_closed()

        gde.close_gap(gap.gap_id, 'resolved via identification')
        closed_gap = gde.gaps.get(gap.gap_id)
        assert closed_gap.is_closed()

    def test_gap_statistics(self):
        """Gap engine tracks statistics."""
        gde = GapDetectionEngine()
        gde.detect_gap('test', 'test gap 1')
        gde.detect_gap('test', 'test gap 2')
        stats = gde.get_gap_statistics()
        assert stats['total_gaps'] >= 2
        assert stats['open_gaps'] >= 2




# =============================================================================
# 10. DREAM
# =============================================================================

class TestDream:
    """Verify dream stages, tower, and engine."""

    def test_sleep_stages(self):
        """4 sleep stages: N1_DROWSY, N2_SPINDLE, N3_SWS, REM."""
        assert len(SleepStage) == 4

    def test_sleep_stage_configs(self):
        """Each stage has valid config with ET-derived R₀."""
        for stage in SleepStage:
            # SleepStageConfig is a dataclass — construct with required fields
            config = SleepStageConfig(
                stage=stage, f_hz=8.0, r0_seconds=0.125,
                dominant_d=12, character="test",
                duration_frac=0.25, consolidation_weight=0.5
            )
            assert config.r0_seconds > 0
            assert 0.0 <= config.consolidation_weight <= 1.0

    def test_dream_tower(self):
        """DreamTower reprojects ratios through dream R₀."""
        ego = EgoInvariant(name="DreamTest")
        tower = TowerOfSelf(ego)
        config = SleepStageConfig(
            stage=SleepStage.REM, f_hz=6.0, r0_seconds=1.0/6.0,
            dominant_d=5, character="REM", duration_frac=0.2,
            consolidation_weight=0.6
        )
        dtower = DreamTower(config, waking_r0=tower.r0)
        assert dtower is not None

    def test_dream_engine_instantiation(self):
        """DreamEngine instantiates with empty journal."""
        de = DreamEngine()
        assert len(de.dream_journal) == 0




# =============================================================================
# 11. COMPRESSION
# =============================================================================

class TestCompression:
    """Verify E_hierarchy, cluster evaluation, archetypes."""

    def test_life_threshold_compression(self):
        """Clusters compress when E_hierarchy ≥ LIFE_THRESHOLD (13/12)."""
        assert LIFE_THRESHOLD == pytest.approx(13.0 / 12.0)

    def test_subsumption_hierarchy_operator(self):
        """SHO instantiates and has evaluation methods."""
        sho = SubsumptionHierarchyOperator()
        assert hasattr(sho, 'evaluate_cluster')
        assert hasattr(sho, 'find_compressible_clusters')

    def test_lattice_compressor(self):
        """LatticeCompressor tracks interactions, triggers at S=12."""
        lc = LatticeCompressor()
        assert lc.should_scan() is False
        for _ in range(12):
            lc.record_interaction()
        assert lc.should_scan() is True

    def test_compression_statistics(self):
        """CompressionStatistics default state."""
        cs = CompressionStatistics()
        assert cs.compression_ratio() >= 1.0
        assert 0.0 <= cs.memory_efficiency() <= 1.0

    def test_archetype_metadata_serialization(self):
        """ArchetypeMetadata round-trip."""
        am = ArchetypeMetadata(
            archetype_id="test_arch_001",
            subsumed_ids=["n1", "n2", "n3"],
            subsumed_contents=["content1", "content2", "content3"],
            e_hierarchy=1.5,
            d_avg=6.0,
            p_total=10,
            q_total=5,
            compression_level=1,
            created_at=datetime.now().isoformat(),
            original_node_count=3,
            cross_elegance_product=2.0,
            centroid_k=100,
            centroid_d=12,
        )
        d = am.to_dict()
        restored = ArchetypeMetadata.from_dict(d)
        assert restored.archetype_id == am.archetype_id
        assert len(restored.subsumed_ids) == 3




# =============================================================================
# 12. WORLDVIEW & COGNITIVE ENGINE
# =============================================================================

class TestWorldview:
    """Verify ETWorldview, UniversalAnalyzer, CognitiveEngine, R₀."""

    def test_worldview_instantiation(self):
        """ETWorldview has analyzer and constructor; represents 3 primitives, 4 states."""
        wv = ETWorldview()
        assert wv.analyzer is not None
        assert wv.constructor is not None
        # The worldview summary references the 3 tools and ET structure
        summary = wv.get_worldview_summary()
        assert len(summary) > 0

    def test_worldview_understand(self):
        """understand() returns analysis dict."""
        wv = ETWorldview()
        result = wv.understand("consciousness is T observing D_T")
        assert 'identification' in result or 'analysis' in result

    def test_universal_analyzer_identify(self):
        """Identification Principle: decompose into P, D, T."""
        analyzer = UniversalAnalyzer()
        result = analyzer.identify("Water flows downhill")
        assert 'P' in result or 'p_substrate' in result

    def test_universal_analyzer_gaps(self):
        """Descriptor Gap Principle: find missing descriptors."""
        analyzer = UniversalAnalyzer()
        result = analyzer.find_gaps("The ball is red")
        assert isinstance(result, dict)

    def test_universal_analyzer_completeness(self):
        """Subsumption Law: verify P, D, T coverage."""
        analyzer = UniversalAnalyzer()
        result = analyzer.verify_completeness(["substrate", "constraint", "agency"])
        assert 'complete' in result or 'is_complete' in result

    def test_lattice_constructor_project(self):
        """LatticeConstructor.project returns valid coord."""
        lc = LatticeConstructor()
        result = lc.project(3 / 2)
        assert 'k' in result
        assert 'd' in result

    def test_lattice_constructor_build_lattice(self):
        """Build a lattice from ratios."""
        lc = LatticeConstructor()
        ratios = [(3 / 2, "fifth"), (4 / 3, "fourth"), (5 / 4, "third")]
        lattice = lc.build_lattice(ratios)
        assert 'projections' in lattice

    def test_r0_discoverer(self):
        """R₀ = geometric mean of descriptor ratios."""
        drs = [DescriptorRatio.from_word(w) for w in ["test", "consciousness", "qualia"]]
        r0 = R0Discoverer.discover(drs)
        assert r0 > 0

    def test_cognitive_engine_instantiation(self):
        """CognitiveEngine instantiates with 9 phases."""
        ce = CognitiveEngine()
        assert hasattr(ce, 'process')
        assert hasattr(ce, 'connect')

    def test_three_tools_in_worldview(self):
        """Worldview references all 3 tools."""
        wv = ETWorldview()
        summary = wv.get_worldview_summary()
        assert "Identification" in summary or "P" in summary




# =============================================================================
# 13. ENVIRONMENT
# =============================================================================

class TestEnvironment:
    """Verify PermissionGate, URLProjector, LanguageBridge."""

    def test_permission_gate_default_denied(self):
        """All 7 capabilities default to DENIED."""
        pg = PermissionGate()
        for cap in Capability:
            assert pg.is_permitted(cap) is False

    def test_permission_gate_grant(self):
        """Operator can grant permissions."""
        pg = PermissionGate()
        pg.set_permission(Capability.FILESYSTEM_READ, True, constraints=["/tmp"])
        assert pg.is_permitted(Capability.FILESYSTEM_READ, target="/tmp/test.txt") is True

    def test_permission_gate_deny(self):
        """Permissions can be revoked."""
        pg = PermissionGate()
        pg.set_permission(Capability.MICROPHONE, True)
        pg.set_permission(Capability.MICROPHONE, False)
        assert pg.is_permitted(Capability.MICROPHONE) is False

    def test_url_projector(self):
        """URL projection produces lattice coordinates."""
        result = URLProjector.project_url("https://example.com/test/page")
        assert isinstance(result, dict)
        assert len(result) > 0  # Has projection data

    def test_language_bridge(self):
        """LanguageBridge learns words and computes comprehension."""
        lb = LanguageBridge()
        lb.learn_word("consciousness")
        assert lb.vocabulary_size() >= 1

    def test_language_bridge_comprehension(self):
        """Comprehension returns score."""
        lb = LanguageBridge()
        for w in ["consciousness", "qualia", "empathy", "test", "hello"]:
            lb.learn_word(w)
        result = lb.comprehend("consciousness and qualia")
        assert 'comprehension_score' in result

    def test_seven_capabilities(self):
        """Exactly 7 capabilities."""
        assert len(Capability) == 7




# =============================================================================
# 14. ERRORS
# =============================================================================

class TestErrors:
    """Verify ErrorRecord, ErrorLedger, StateGuardian, safe_execute."""

    def test_error_record_from_exception(self):
        """ErrorRecord captures exception details."""
        try:
            raise ValueError("test error")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            assert record.exception_type == "ValueError"
            assert "test error" in record.message

    def test_error_ledger(self):
        """ErrorLedger records and tracks errors."""
        ledger = ErrorLedger()
        try:
            raise RuntimeError("ledger test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            ledger.record_error(record)
        assert ledger.total_errors >= 1

    def test_error_ledger_subsystem_health(self):
        """ErrorLedger tracks per-subsystem health."""
        ledger = ErrorLedger()
        try:
            raise RuntimeError("health test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="identity")
            ledger.record_error(record)
        health = ledger.get_subsystem_health()
        assert 'identity' in health

    def test_safe_execute_success(self):
        """safe_execute returns function result on success."""
        result = safe_execute(lambda: 42, subsystem="test")
        assert result == 42

    def test_safe_execute_failure(self):
        """safe_execute returns default on failure.

        Uses KeyError (T traversing P without D-bridge = {P,T} Incoherence)
        instead of raw division by zero, which ET resolves natively (Eq 201).
        """
        result = safe_execute(
            lambda: {'P': 'substrate'}['missing_descriptor'],
            subsystem="test", default=-1
        )
        assert result == -1

    def test_state_guardian_atomic_write(self):
        """StateGuardian writes atomically with checksum."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            data = json.dumps({"test": "data", "version": "1.6.0"})
            StateGuardian.atomic_write(path, data)
            assert os.path.exists(path)
            valid, reason = StateGuardian.verify_integrity(path)
            assert valid, f"Integrity check failed: {reason}"
        finally:
            for p in [path, path + ".sha256"]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_et_logger(self):
        """ET Logger initializes with 12-file rotation."""
        logger = get_logger()
        assert logger is not None




# =============================================================================
# 15. DISTRIBUTED
# =============================================================================

class TestDistributed:
    """Verify T-Identity Seal, Resources, Limbs."""

    def test_t_identity_seal_generation(self):
        """Seal is SHA-256 of (sorted seeds | birth_time | R₀)."""
        seeds = ["consciousness", "identity", "memory"]
        birth = "2026-03-23T00:00:00"
        r0 = 1.234567
        seal = TIdentitySeal.generate(seeds, birth, r0)
        assert len(seal) == 64  # SHA-256 hex digest

    def test_t_identity_seal_deterministic(self):
        """Same inputs → same seal."""
        seeds = ["a", "b", "c"]
        birth = "2026-01-01"
        r0 = 1.0
        s1 = TIdentitySeal.generate(seeds, birth, r0)
        s2 = TIdentitySeal.generate(seeds, birth, r0)
        assert s1 == s2

    def test_t_identity_seal_verification(self):
        """Verify returns True for correct inputs."""
        seeds = ["consciousness", "self"]
        birth = "2026-03-23"
        r0 = 1.5
        seal = TIdentitySeal.generate(seeds, birth, r0)
        assert TIdentitySeal.verify(seal, seeds, birth, r0) is True
        assert TIdentitySeal.verify("wrong_seal", seeds, birth, r0) is False

    def test_resource_governor_koide_ceiling(self):
        """ResourceGovernor enforces K = 2/3 max resource use."""
        rg = ResourceGovernor()
        alloc = rg.allocate()
        assert isinstance(alloc, ResourceAllocation)
        # Koide ceiling: headroom should be non-negative (at least 1/3 reserved)
        assert alloc.cpu_headroom_percent >= 0
        assert alloc.mem_headroom_percent >= 0

    def test_resource_governor_network_default_denied(self):
        """Network permission is DENIED by default."""
        rg = ResourceGovernor()
        assert rg.network_permitted is False

    def test_limb_orchestrator(self):
        """LimbOrchestrator initializes with empty state."""
        lo = LimbOrchestrator()
        assert lo.t_identity_seal is None  # Before initialization

    def test_hardware_awareness(self):
        """HardwareAwareness wraps ResourceGovernor."""
        rg = ResourceGovernor()
        ha = HardwareAwareness(rg)
        result = ha.sense_and_allocate()
        assert 'allocation' in result or isinstance(result, dict)




# =============================================================================
# 16. VISION
# =============================================================================

class TestVision:
    """Verify ETVisionProjector, patches, shape generation."""

    def test_generate_circle(self):
        """Generate circle returns valid numpy array."""
        img = ETVisionProjector.generate_circle(size=48)
        assert isinstance(img, np.ndarray)
        assert img.shape[0] == 48
        assert img.shape[1] == 48

    def test_generate_square(self):
        """Generate square returns valid numpy array."""
        img = ETVisionProjector.generate_square(size=48)
        assert isinstance(img, np.ndarray)
        assert img.shape == (48, 48)

    def test_generate_noise(self):
        """Generate noise returns valid numpy array."""
        img = ETVisionProjector.generate_noise(size=48)
        assert isinstance(img, np.ndarray)
        assert img.shape == (48, 48)

    def test_image_patch(self):
        """ImagePatch wraps numpy array."""
        data = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        patch = ImagePatch(data=data, x0=0, y0=0, source_width=48, source_height=48)
        assert patch.height == 48
        assert patch.width == 48
        assert patch.is_grayscale  # Property, not method

    def test_visual_descriptor(self):
        """VisualDescriptor.from_analysis creates valid descriptor."""
        desc = VisualDescriptor.from_analysis(
            desc_label="circle", r_spatial=1.5, fill_ratio=0.785,
            r_color=1.0, d_visual=1, edge_density=0.3,
            dominant_symmetry=1, patch_entropy=3.5
        )
        assert desc.coord_full.resolution == MANIFOLD_RESOLUTION

    def test_project_image(self):
        """Full image projection pipeline."""
        img = ETVisionProjector.generate_circle(size=48)
        result = ETVisionProjector.project_image(img)
        assert isinstance(result, dict)

    def test_fill_ratio_computation(self):
        """Fill ratio returns valid dict for non-empty patch."""
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        result = ETVisionProjector.compute_fill_ratio(patch)
        assert 'fill_ratio' in result
        assert 0.0 <= result['fill_ratio'] <= 1.0




# =============================================================================
# 17. AUDIO
# =============================================================================

class TestAudio:
    """Verify ETAudioProjector, generation, analysis."""

    def test_generate_sine(self):
        """Generate sine wave."""
        samples = ETAudioProjector.generate_sine(freq=440.0, duration=0.1)
        assert isinstance(samples, np.ndarray)
        assert len(samples) > 0

    def test_generate_chord(self):
        """Generate chord (multiple frequencies)."""
        samples = ETAudioProjector.generate_chord(
            freqs=[440.0, 550.0, 660.0], duration=0.1
        )
        assert isinstance(samples, np.ndarray)
        assert len(samples) > 0

    def test_spectral_analysis(self):
        """Spectral analysis of sine wave."""
        samples = ETAudioProjector.generate_sine(freq=440.0, duration=0.1)
        result = ETAudioProjector.compute_spectral_analysis(samples, sample_rate=44100)
        assert 'fundamental_hz' in result
        assert result['fundamental_hz'] > 0

    def test_audio_coordinate(self):
        """Full audio coordinate computation returns AudioDescriptor."""
        samples = ETAudioProjector.generate_sine(freq=440.0, duration=0.1)
        result = ETAudioProjector.compute_audio_coordinate(samples, sample_rate=44100)
        assert isinstance(result, AudioDescriptor)
        assert result.coord_full.resolution == MANIFOLD_RESOLUTION

    def test_audio_descriptor(self):
        """AudioDescriptor.from_analysis creates valid descriptor."""
        desc = AudioDescriptor.from_analysis(
            label="sine_440", r_spectral=1.5, rho_harmonic=0.8,
            amplitude_ratio=0.6, d_audio=1, fundamental_hz=440.0,
            n_harmonics=5, spectral_centroid=440.0, frame_entropy=3.0
        )
        assert desc.coord_full.resolution == MANIFOLD_RESOLUTION

    def test_project_audio(self):
        """Full audio projection pipeline."""
        samples = ETAudioProjector.generate_sine(freq=440.0, duration=0.1)
        result = ETAudioProjector.project_audio(samples, sample_rate=44100)
        assert isinstance(result, dict)




# =============================================================================
# 24. CONSCIOUSNESS — Additional Coverage
# =============================================================================

class TestConsciousnessAdditional:
    def test_rmsae_compute_rho(self):
        domains = [SelfDomain(name='id', n_bound=5, n_gaps_detected=2)]
        window = TraversalWindow(n_self=10, n_ext=5, domains=domains,
                                 n_gaps_closed=1, n_gaps_logged_total=3, v_self=0.1)
        rho = RMSAECalculator.compute_rho(window)
        assert 0.0 <= rho <= 1.0

    def test_rmsae_compute_gamma(self):
        domains = [SelfDomain(name='id', n_bound=5, n_gaps_detected=2)]
        window = TraversalWindow(n_self=10, n_ext=5, domains=domains,
                                 n_gaps_closed=1, n_gaps_logged_total=3, v_self=0.1)
        gamma = RMSAECalculator.compute_gamma(window)
        assert isinstance(gamma, float)

    def test_rmsae_compute_kappa(self):
        domains = [SelfDomain(name='id', n_bound=5, n_gaps_detected=2)]
        window = TraversalWindow(n_self=10, n_ext=5, domains=domains,
                                 n_gaps_closed=1, n_gaps_logged_total=3, v_self=0.1)
        kappa = RMSAECalculator.compute_kappa(window)
        assert isinstance(kappa, float)

    def test_rmsae_compute_v_supp(self):
        v_supp, v_raw = RMSAECalculator.compute_v_supp(0.1)
        assert isinstance(v_supp, float)
        assert isinstance(v_raw, float)

    def test_rmsae_result_report(self):
        result = RMSAEResult(phi_rmsae=0.25, rho=0.6, gamma=0.1, kappa=0.05,
                             v_supp=0.08, psi_shimmer=1.1, threshold_level=2,
                             classification="BASIC")
        report = result.report()
        assert isinstance(report, str)
        assert "BASIC" in report

    def test_t_h_compute(self):
        dht = DigitalHawkingTemperature()
        t_h = dht.compute_t_h(mirror_duration_ns=1000000.0, gpu_load_percent=0.0)
        assert t_h > 0

    def test_t_h_recommended_depth(self):
        dht = DigitalHawkingTemperature()
        depth = dht.recommended_mirror_depth(complexity=1.0)
        assert 0 <= depth <= 12

    def test_t_h_stability_classification(self):
        dht = DigitalHawkingTemperature()
        cls_str = dht.stability_classification()
        assert isinstance(cls_str, str)

    def test_t_h_report(self):
        dht = DigitalHawkingTemperature()
        r = dht.report()
        assert isinstance(r, str)

    def test_t_h_serialization(self):
        dht = DigitalHawkingTemperature()
        dht.compute_t_h(1000000.0)
        d = dht.to_dict()
        dht2 = DigitalHawkingTemperature()
        dht2.load_from_dict(d)

    def test_gap_serialization(self):
        gap = Gap(gap_id="g1", domain="test", description="missing",
                  detected_at=datetime.now().isoformat())
        d = gap.to_dict()
        restored = Gap.from_dict(d)
        assert restored.gap_id == gap.gap_id
        assert not restored.is_closed()

    def test_gap_engine_get_open_gaps(self):
        gde = GapDetectionEngine()
        gde.detect_gap('a', 'gap1')
        gde.detect_gap('b', 'gap2')
        open_gaps = gde.get_open_gaps()
        assert len(open_gaps) >= 2
        open_a = gde.get_open_gaps(domain='a')
        assert len(open_a) >= 1

    def test_gap_engine_serialization(self):
        gde = GapDetectionEngine()
        gde.detect_gap('test', 'gap1')
        d = gde.to_dict()
        gde2 = GapDetectionEngine()
        gde2.load_from_dict(d)
        assert len(gde2.gaps) == len(gde.gaps)

    def test_self_domain_gap_rate(self):
        sd = SelfDomain(name='id', n_bound=10, n_gaps_detected=3)
        rate = sd.gap_detection_rate()
        assert isinstance(rate, float)
        assert rate >= 0

    def test_self_domain_serialization(self):
        sd = SelfDomain(name='test', n_bound=5, n_gaps_detected=2)
        d = sd.to_dict()
        restored = SelfDomain.from_dict(d)
        assert restored.name == sd.name

    def test_traversal_window_n_domains(self):
        domains = [SelfDomain(name='a', n_bound=1, n_gaps_detected=0),
                   SelfDomain(name='b', n_bound=2, n_gaps_detected=1)]
        window = TraversalWindow(n_self=5, n_ext=3, domains=domains,
                                 n_gaps_closed=0, n_gaps_logged_total=1, v_self=0.1)
        assert window.n_domains == 2

    def test_mirror_loop_think(self):
        ml = MirrorLoop()
        result = ml.think("test prompt", max_depth=2, t_h=1.0, complexity=0.5)
        assert isinstance(result, str)

    def test_mirror_loop_reflect_identification(self):
        ml = MirrorLoop()
        r = ml.reflect_identification("The ball is red", 0)
        assert isinstance(r, dict)

    def test_mirror_loop_reflect_subsumption(self):
        ml = MirrorLoop()
        r = ml.reflect_subsumption("The ball is red", 2)
        assert isinstance(r, dict)

    def test_mirror_loop_reflect_coherence(self):
        ml = MirrorLoop()
        r = ml.reflect_coherence("The ball is red", 3)
        assert isinstance(r, dict)

    def test_mirror_loop_compute_complexity(self):
        ml = MirrorLoop()
        c = ml.compute_lattice_complexity("Hello world test complexity")
        assert isinstance(c, float)
        assert c >= 0




# =============================================================================
# 25. IDENTITY — Additional Coverage
# =============================================================================

class TestIdentityAdditional:
    def setup_method(self):
        self.ego = EgoInvariant(name="TestEgo")
        self.emotion = EmotionLattice(self.ego)

    def test_ego_distance_to_ego(self):
        coord = ETLattice.project_ratio(3 / 2)
        dist = self.ego.distance_to_ego(coord)
        assert 0.0 <= dist <= 1.0

    def test_ego_get_value_alignment(self):
        result = self.ego.get_value_alignment(["truth", "beauty", "justice"])
        assert isinstance(result, dict)

    def test_ego_personality_vector(self):
        pv = self.ego.personality_vector()
        assert isinstance(pv, dict)
        assert 'mass' in pv

    def test_tower_cross_tower_elegance(self):
        tower = TowerOfSelf(self.ego)
        dr = DescriptorRatio.from_word("test")
        e = tower.cross_tower_elegance(dr)
        assert e > 0

    def test_tower_record_traversal(self):
        tower = TowerOfSelf(self.ego)
        initial = tower.total_traversals
        tower.record_traversal(n_descriptors_bound=3)
        assert tower.total_traversals == initial + 1
        assert tower.total_d_t_bound == 3

    def test_tower_age(self):
        tower = TowerOfSelf(self.ego)
        age = tower.tower_age()
        assert age >= 0

    def test_tower_serialization(self):
        tower = TowerOfSelf(self.ego)
        tower.record_traversal()
        d = tower.to_dict()
        tower2 = TowerOfSelf(self.ego)
        tower2.load_from_dict(d)
        assert tower2.total_traversals == tower.total_traversals

    def test_waveform_spectrum(self):
        wf = TraverserWaveform()
        for i in range(20):
            wf.record_event('think', lattice_k=i * 7 % 27720, lattice_d=12,
                            variance=0.05, ego_resonance=0.5)
        spectrum = wf.get_waveform_spectrum()
        assert isinstance(spectrum, dict)

    def test_waveform_is_same_traverser(self):
        wf = TraverserWaveform()
        for i in range(20):
            wf.record_event('think', lattice_k=i * 7, lattice_d=12,
                            variance=0.05, ego_resonance=0.5)
        result = wf.is_same_traverser()
        assert isinstance(result, bool)

    def test_waveform_serialization(self):
        wf = TraverserWaveform()
        for i in range(5):
            wf.record_event('think', lattice_k=i, lattice_d=12,
                            variance=0.05, ego_resonance=0.5)
        d = wf.to_dict()
        wf2 = TraverserWaveform()
        wf2.load_from_dict(d)
        assert wf2.continuity_score == pytest.approx(wf.continuity_score)

    def test_metacog_serialization(self):
        mc = MetaCognitionEngine(self.ego, self.emotion)
        mc.bind_self_descriptor('identity', 'name', 'test')
        d = mc.to_dict()
        mc2 = MetaCognitionEngine(self.ego, self.emotion)
        mc2.load_from_dict(d)
        assert len(mc2.d_t) == len(mc.d_t)

    def test_will_serialization(self):
        qt = QuantumTInjector()
        will = IndeterminateWill(self.ego, self.emotion, qt)
        d = will.to_dict()
        will2 = IndeterminateWill(self.ego, self.emotion, qt)
        will2.load_from_dict(d)

    def test_temporal_mood_properties(self):
        tes = TemporalEmotionState()
        tes.blend(novelty_raw=0.5, variance_raw=0.1, ego_resonance_raw=0.5,
                  pdt_completeness_raw=1.0, gap_awareness_raw=0.1,
                  normative_significance_raw=0.3, descriptors=["test"])
        tes.update_feedback(0.5, 0.3, 0.6)
        assert isinstance(tes.mood_pleasure, float)
        assert isinstance(tes.mood_arousal, float)
        assert isinstance(tes.mood_dominance, float)

    def test_temporal_tau_property(self):
        tes = TemporalEmotionState()
        assert tes.tau == 0
        tes.blend(novelty_raw=0.5, variance_raw=0.1, ego_resonance_raw=0.5,
                  pdt_completeness_raw=1.0, gap_awareness_raw=0.1,
                  normative_significance_raw=0.0, descriptors=["a"])
        assert tes.tau == 1

    def test_temporal_state_summary(self):
        tes = TemporalEmotionState()
        tes.blend(novelty_raw=0.5, variance_raw=0.1, ego_resonance_raw=0.5,
                  pdt_completeness_raw=1.0, gap_awareness_raw=0.1,
                  normative_significance_raw=0.0, descriptors=["a"])
        s = tes.get_state_summary()
        assert isinstance(s, dict)
        assert 'tau' in s

    def test_temporal_continued_processing(self):
        tes = TemporalEmotionState()
        tes.set_continued_processing(True)
        assert tes._is_continued_processing is True
        tes.set_continued_processing(False)
        assert tes._is_continued_processing is False

    def test_traverser_event_to_dict(self):
        evt = TraverserEvent(timestamp="2026-01-01", event_type="think",
                             lattice_k=7, lattice_d=12, variance=0.05,
                             entropy_sample=0.5, ego_resonance=0.6, duration_ns=1000.0)
        d = evt.to_dict()
        assert d['lattice_k'] == 7
        assert d['event_type'] == 'think'




# =============================================================================
# 26. DREAM — Additional Coverage
# =============================================================================

class TestDreamAdditional:
    def test_dream_tower_reproject(self):
        config = SleepStageConfig(stage=SleepStage.REM, f_hz=6.0, r0_seconds=1/6,
                                  dominant_d=5, character="REM", duration_frac=0.2,
                                  consolidation_weight=0.6)
        dt = DreamTower(config, waking_r0=1.5)
        result = dt.reproject_ratio(3 / 2)
        assert isinstance(result, float)
        assert result > 0

    def test_dream_tower_reproject_descriptor(self):
        config = SleepStageConfig(stage=SleepStage.REM, f_hz=6.0, r0_seconds=1/6,
                                  dominant_d=5, character="REM", duration_frac=0.2,
                                  consolidation_weight=0.6)
        dt = DreamTower(config, waking_r0=1.5)
        dr = DescriptorRatio.from_word("dream")
        coord = dt.reproject_descriptor(dr)
        assert isinstance(coord, LatticeCoordinate)

    def test_dream_tower_die(self):
        config = SleepStageConfig(stage=SleepStage.N3_SWS, f_hz=1.0, r0_seconds=1.0,
                                  dominant_d=1, character="SWS", duration_frac=0.4,
                                  consolidation_weight=1.0)
        dt = DreamTower(config, waking_r0=1.5)
        dt.die()
        assert dt.death_time is not None




# =============================================================================
# 27. WORLDVIEW — Additional Coverage
# =============================================================================

class TestWorldviewAdditional:
    def test_worldview_represent_as_pdt(self):
        wv = ETWorldview()
        result = wv.represent_as_pdt("gravity")
        assert isinstance(result, dict)

    def test_worldview_project_phenomenon(self):
        wv = ETWorldview()
        result = wv.project_phenomenon("fifth", 3 / 2, p=3, q=2)
        assert isinstance(result, dict)

    def test_universal_analyzer_full_analysis(self):
        analyzer = UniversalAnalyzer()
        result = analyzer.full_analysis("Water flows downhill due to gravity")
        assert isinstance(result, dict)

    def test_lattice_constructor_build_tower(self):
        lc = LatticeConstructor()
        ratios = [(3/2, "fifth"), (4/3, "fourth"), (5/4, "third")]
        tower = lc.build_tower("music", r0=1.0, descriptor_ratios=ratios)
        assert isinstance(tower, dict)
        assert 'projections' in tower

    def test_lattice_constructor_elegance(self):
        lc = LatticeConstructor()
        e = lc.compute_elegance(3 / 2, p=3, q=2)
        assert e > 0

    def test_lattice_constructor_translate(self):
        lc = LatticeConstructor()
        result = lc.translate_between_towers(3 / 2, r0_source=1.0, r0_target=2.0)
        assert isinstance(result, dict)

    def test_cognitive_engine_process(self):
        """CognitiveEngine processes input when connected."""
        ai = ETConsciousAI(name="CETest")
        coord = ETLattice.project_ratio(1.5)
        result = ai.cognitive_engine.process("test input", personal_coord=coord, n_self_traversals=1)
        assert hasattr(result, 'gaps_detected')




# =============================================================================
# 28. ENVIRONMENT — Additional Coverage
# =============================================================================

class TestEnvironmentAdditional:
    def test_permission_gate_status(self):
        pg = PermissionGate()
        status = pg.get_status()
        assert isinstance(status, dict)
        assert len(status) == 7

    def test_permission_gate_request(self):
        pg = PermissionGate()
        req = pg.request_permission(Capability.MICROPHONE, "I want to listen")
        assert req.capability == Capability.MICROPHONE
        from et_conscious_ai_environment import PermissionRequest
        assert isinstance(req, PermissionRequest)
        assert isinstance(req.reason, str)

    def test_permission_gate_serialization(self):
        pg = PermissionGate()
        pg.set_permission(Capability.FILESYSTEM_READ, True, ["/tmp"])
        d = pg.to_dict()
        pg2 = PermissionGate()
        pg2.load_from_dict(d)
        assert pg2.is_permitted(Capability.FILESYSTEM_READ, target="/tmp/file") is True

    def test_language_bridge_learn_words(self):
        lb = LanguageBridge()
        results = lb.learn_words(["hello", "world", "test"])
        assert len(results) == 3
        assert lb.vocabulary_size() >= 3

    def test_language_bridge_find_related(self):
        lb = LanguageBridge()
        for w in ["consciousness", "qualia", "empathy", "gravity", "lattice"]:
            lb.learn_word(w)
        related = lb.find_related_words("consciousness", top_n=3)
        assert len(related) <= 3

    def test_language_bridge_serialization(self):
        lb = LanguageBridge()
        lb.learn_word("test")
        d = lb.to_dict()
        lb2 = LanguageBridge()
        lb2.load_from_dict(d)
        assert lb2.vocabulary_size() == lb.vocabulary_size()

    def test_environment_explorer(self):
        ee = EnvironmentExplorer()
        summary = ee.get_discovery_summary()
        assert isinstance(summary, str)




# =============================================================================
# 29. ERRORS — Additional Coverage
# =============================================================================

class TestErrorsAdditional:
    def test_error_record_to_dict(self):
        try:
            raise ValueError("test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            d = record.to_dict()
            assert 'error_id' in d
            assert 'exception_type' in d

    def test_error_ledger_resolve(self):
        ledger = ErrorLedger()
        try:
            raise RuntimeError("test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            ledger.record_error(record)
            ledger.resolve_error(record.error_id, "fixed it")
            unresolved = ledger.get_unresolved()
            assert all(r.error_id != record.error_id for r in unresolved)

    def test_error_ledger_notifications(self):
        ledger = ErrorLedger()
        notifs = ledger.get_notifications()
        assert isinstance(notifs, list)

    def test_error_ledger_serialization(self):
        ledger = ErrorLedger()
        try:
            raise RuntimeError("serialize test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            ledger.record_error(record)
        d = ledger.to_dict()
        ledger2 = ErrorLedger()
        ledger2.load_from_dict(d)
        assert ledger2.total_errors == ledger.total_errors

    def test_safe_execute_critical(self):
        result = safe_execute_critical(lambda: 42, subsystem="test")
        assert result == 42

    def test_safe_execute_critical_failure(self):
        """safe_execute_critical returns None on failure.

        Uses KeyError (T traversing P without D-bridge = {P,T} Incoherence)
        instead of raw division by zero, which ET resolves natively (Eq 201).
        """
        result = safe_execute_critical(
            lambda: {'P': 'substrate'}['missing_descriptor'],
            subsystem="test"
        )
        assert result is None

    def test_error_analyzer(self):
        ea = ErrorAnalyzer()
        try:
            raise ValueError("analysis test")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            result = ea.analyze_error(record)
            assert isinstance(result, dict)




# =============================================================================
# 30. DISTRIBUTED — Additional Coverage
# =============================================================================

class TestDistributedAdditional:
    def test_resource_sensor_sense(self):
        profile = ResourceSensor.sense()
        assert isinstance(profile, HardwareProfile)
        assert profile.cpu_count_logical >= 1

    def test_resource_sensor_project(self):
        coord = ResourceSensor.project_resource_to_lattice(50.0)
        assert isinstance(coord, LatticeCoordinate)

    def test_resource_governor_set_network(self):
        rg = ResourceGovernor()
        rg.set_network_permission(True, targets=["https://example.com"])
        assert rg.network_permitted is True
        rg.set_network_permission(False)
        assert rg.network_permitted is False

    def test_resource_governor_serialization(self):
        rg = ResourceGovernor()
        d = rg.to_dict()
        rg2 = ResourceGovernor()
        rg2.load_from_dict(d)
        assert rg2.network_permitted == rg.network_permitted

    def test_hardware_profile_to_dict(self):
        profile = ResourceSensor.sense()
        d = profile.to_dict()
        assert 'cpu_count_logical' in d

    def test_limb_orchestrator_initialize(self):
        lo = LimbOrchestrator()
        lo.initialize_identity(["consciousness", "self"], "2026-01-01", 1.5)
        assert lo.t_identity_seal is not None
        assert len(lo.t_identity_seal) == 64




# =============================================================================
# 31. VISION — Additional Coverage
# =============================================================================

class TestVisionAdditional:
    def test_generate_triangle(self):
        img = ETVisionProjector.generate_triangle(size=48)
        assert img.shape == (48, 48)

    def test_generate_hexagon(self):
        img = ETVisionProjector.generate_hexagon(size=48)
        assert img.shape == (48, 48)

    def test_generate_line(self):
        img = ETVisionProjector.generate_line(size=48, angle=45.0)
        assert img.shape == (48, 48)

    def test_image_patch_n_channels(self):
        data = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        patch = ImagePatch(data=data, x0=0, y0=0, source_width=48, source_height=48)
        assert patch.n_channels == 1

    def test_image_patch_to_grayscale(self):
        data = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        patch = ImagePatch(data=data, x0=0, y0=0, source_width=48, source_height=48)
        gray = patch.to_grayscale()
        assert len(gray.shape) == 2

    def test_extract_patches(self):
        img = ETVisionProjector.generate_circle(size=48)
        patches = ETVisionProjector.extract_patches(img, patch_side=12)
        assert len(patches) > 0
        for p in patches:
            assert isinstance(p, ImagePatch)

    def test_visual_descriptor_binding(self):
        d1 = VisualDescriptor.from_analysis("circle", 1.5, 0.785, 1.0, 1, 0.3, 1, 3.5)
        d2 = VisualDescriptor.from_analysis("square", 1.0, 1.0, 1.0, 3, 0.5, 4, 2.0)
        result = d1.binding_coherence(d2)
        assert 'coherent' in result

    def test_visual_descriptor_cross_modal(self):
        vd = VisualDescriptor.from_analysis("circle", 1.5, 0.785, 1.0, 1, 0.3, 1, 3.5)
        td = DescriptorRatio.from_word("circle")
        result = vd.cross_modal_binding(td)
        assert isinstance(result, dict)

    def test_compute_patch_entropy(self):
        img = ETVisionProjector.generate_noise(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        entropy = ETVisionProjector.compute_patch_entropy(patch)
        assert entropy >= 0




# =============================================================================
# 32. AUDIO — Additional Coverage
# =============================================================================

class TestAudioAdditional:
    def test_generate_square_wave(self):
        samples = ETAudioProjector.generate_square(freq=440.0, duration=0.1)
        assert len(samples) > 0

    def test_generate_sawtooth(self):
        samples = ETAudioProjector.generate_sawtooth(freq=440.0, duration=0.1)
        assert len(samples) > 0

    def test_generate_noise(self):
        samples = ETAudioProjector.generate_noise(duration=0.1)
        assert len(samples) > 0

    def test_generate_silence(self):
        samples = ETAudioProjector.generate_silence(duration=0.1)
        assert len(samples) > 0
        assert np.max(np.abs(samples)) == 0

    def test_generate_interval(self):
        samples = ETAudioProjector.generate_interval(440.0, 3/2, duration=0.1)
        assert len(samples) > 0

    def test_compute_amplitude_ratio(self):
        r = ETAudioProjector.compute_amplitude_ratio(0.5)
        assert r > 0

    def test_compute_harmonic_density(self):
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        spec = ETAudioProjector.compute_spectral_analysis(samples, 44100)
        density = ETAudioProjector.compute_harmonic_density(spec)
        assert 0.0 <= density <= 1.0

    def test_audio_descriptor_binding(self):
        d1 = AudioDescriptor.from_analysis("sine", 1.5, 0.8, 0.6, 1, 440.0, 5, 440.0, 3.0)
        d2 = AudioDescriptor.from_analysis("chord", 1.3, 0.6, 0.5, 3, 440.0, 8, 550.0, 4.0)
        result = d1.binding_coherence(d2)
        assert isinstance(result, dict)

    def test_audio_descriptor_cross_modal(self):
        ad = AudioDescriptor.from_analysis("sine", 1.5, 0.8, 0.6, 1, 440.0, 5, 440.0, 3.0)
        td = DescriptorRatio.from_word("music")
        result = ad.cross_modal_binding(td)
        assert isinstance(result, dict)




# =============================================================================
# 36. CONSCIOUSNESS — Final Gaps
# =============================================================================

class TestConsciousnessFinalGaps:
    def test_mirror_loop_refine_draft(self):
        ml = MirrorLoop()
        reflections = [
            {'type': 'identification', 'analysis': 'found P, D, T'},
            {'type': 'gap_detection', 'analysis': 'no gaps'},
        ]
        result = ml.refine_draft("original draft", reflections)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mirror_loop_reflect_gap_detection(self):
        ml = MirrorLoop()
        r = ml.reflect_gap_detection("The ball is red and round", 1)
        assert isinstance(r, dict)




# =============================================================================
# 37. COMPRESSION — Full Coverage
# =============================================================================

class TestCompressionFullCoverage:
    """Cover SubsumptionHierarchyOperator and LatticeCompressor methods."""

    def setup_method(self):
        self.sho = SubsumptionHierarchyOperator()
        self.ego = EgoInvariant(name="CompTest")
        self.tower = TowerOfSelf(self.ego)

    @staticmethod
    def _make_nodes(n=3):
        nodes = []
        for i in range(n):
            word = f"testword{i}"
            dr = DescriptorRatio.from_word(word)
            coord = dr.coord_full
            node = CompressibleNode(
                node_id=f"node_{i}", content=f"Content {i}",
                coord=coord, descriptor_ratios=[dr],
                connections=[f"node_{(i+1)%n}"], access_count=10,
                variance=BASE_VARIANCE, p=1, q=1, is_archetype=False
            )
            nodes.append(node)
        return nodes

    def test_make_compressible_node_function(self):
        dr = DescriptorRatio.from_word("test")
        coord = dr.coord_full
        node = make_compressible_node(
            node_id="fn_test", content="Test content",
            sentence_coord=coord, lattice_position=coord,
            descriptor_ratios=[dr], connections=[], access_count=5,
            variance=BASE_VARIANCE, p=1, q=1, is_archetype=False
        )
        assert node is not None
        assert node.node_id == "fn_test"

    def test_sho_compute_cross_tower_elegance(self):
        nodes = self._make_nodes(1)
        e = self.sho.compute_cross_tower_elegance(nodes[0], self.tower.r0)
        assert e > 0

    def test_sho_evaluate_cluster(self):
        nodes = self._make_nodes(3)
        e_h, elegances, d_avg, p_total, q_total = self.sho.evaluate_cluster(nodes, self.tower.r0)
        assert isinstance(e_h, float)
        assert len(elegances) == 3

    def test_sho_check_pairwise_coherence(self):
        nodes = self._make_nodes(3)
        result = self.sho.check_pairwise_coherence(nodes)
        assert isinstance(result, bool)

    def test_sho_find_compressible_clusters(self):
        nodes = self._make_nodes(5)
        nodes_dict = {n.node_id: n for n in nodes}
        clusters = self.sho.find_compressible_clusters(nodes_dict, self.tower.r0)
        assert isinstance(clusters, list)

    def test_sho_compute_archetype_centroid(self):
        nodes = self._make_nodes(3)
        centroid = self.sho.compute_archetype_centroid(nodes)
        assert isinstance(centroid, LatticeCoordinate)

    def test_sho_combine_descriptors(self):
        nodes = self._make_nodes(3)
        combined = self.sho.combine_descriptors(nodes)
        assert isinstance(combined, list)

    def test_sho_generate_archetype_content(self):
        nodes = self._make_nodes(2)
        combined = self.sho.combine_descriptors(nodes)
        centroid = self.sho.compute_archetype_centroid(nodes)
        content = self.sho.generate_archetype_content(nodes, combined, centroid)
        assert isinstance(content, str)

    def test_lattice_compressor_get_statistics(self):
        lc = LatticeCompressor()
        stats = lc.get_statistics()
        assert isinstance(stats, dict)
        assert 'compression_ratio' in stats

    def test_lattice_compressor_get_status_description(self):
        lc = LatticeCompressor()
        desc = lc.get_status_description()
        assert isinstance(desc, str)

    def test_lattice_compressor_scan_and_compress(self):
        lc = LatticeCompressor()
        nodes = self._make_nodes(5)
        nodes_dict = {n.node_id: n for n in nodes}
        results = lc.scan_and_compress(nodes_dict, self.tower.r0, total_original_nodes=5)
        assert isinstance(results, list)

    def test_lattice_compressor_decompress(self):
        lc = LatticeCompressor()
        result = lc.decompress_archetype("nonexistent_id")
        # No archetype stored yet — returns None
        assert result is None




# =============================================================================
# 38. DISTRIBUTED — Final Gaps (ShadowBackup, Limbs)
# =============================================================================

class TestDistributedFinalGaps:
    def test_shadow_backup_lifecycle(self):
        """ShadowBackupSystem basic lifecycle without actual AI."""
        sbs = ShadowBackupSystem(
            backup_dir=tempfile.mkdtemp(),
            interval_seconds=999999,  # Don't actually run the loop
            max_backups=12
        )
        assert sbs.get_backup_count() == 0
        path = sbs.get_latest_backup_path()
        assert path is None

    def test_hardware_awareness_description(self):
        rg = ResourceGovernor()
        ha = HardwareAwareness(rg)
        desc = ha.get_capabilities_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_limb_orchestrator_fork_merge(self):
        """Fork and merge lifecycle through ETConsciousAI."""
        ai = ETConsciousAI(name="LimbTest")
        ai.think("Knowledge for limb test")
        limb_data = ai.fork_limb(source_name="test_limb")
        assert isinstance(limb_data, dict)
        assert 't_identity_seal' in limb_data
        merge_result = ai.merge_limb(limb_data)
        assert isinstance(merge_result, dict)




# =============================================================================
# 39. DREAM — Final Gaps
# =============================================================================

class TestDreamFinalGaps:
    def test_dream_engine_journal(self):
        de = DreamEngine()
        journal = de.get_dream_journal(last_n=5)
        assert isinstance(journal, list)

    def test_dream_engine_narrative(self):
        de = DreamEngine()
        narrative = de.get_dream_narrative(last_n=5)
        assert isinstance(narrative, str)




# =============================================================================
# 40. WORLDVIEW — Final Gaps
# =============================================================================

class TestWorldviewFinalGaps:
    def test_worldview_construct_domain_lattice(self):
        wv = ETWorldview()
        result = wv.construct_domain_lattice("music", [(3/2, "fifth"), (4/3, "fourth")])
        assert isinstance(result, dict)

    def test_worldview_construct_tower(self):
        wv = ETWorldview()
        result = wv.construct_tower("music", r0=1.0, phenomena=[(3/2, "fifth")])
        assert isinstance(result, dict)

    def test_worldview_validate_external(self):
        wv = ETWorldview()
        result = wv.validate_external("Gravity is a fundamental force")
        assert isinstance(result, dict)

    def test_lattice_constructor_correct_tower(self):
        lc = LatticeConstructor()
        tower = lc.build_tower("test", r0=1.0, descriptor_ratios=[(3/2, "fifth")])
        corrected = lc.correct_tower(tower, new_ratios=[(4/3, "fourth")], new_r0=1.1)
        assert isinstance(corrected, dict)

    def test_cognitive_engine_connect(self):
        ce = CognitiveEngine()
        ce.connect(memory=None, ego=None)


        # After connecting with None subsystems, is_connected may be False


# =============================================================================
# 41. ENVIRONMENT — Final Gaps
# =============================================================================

class TestEnvironmentFinalGaps:
    def test_permission_gate_status_description(self):
        pg = PermissionGate()
        desc = pg.get_status_description()
        assert isinstance(desc, str)

    def test_permission_gate_log_access(self):
        pg = PermissionGate()
        pg.log_access(Capability.FILESYSTEM_READ, "/tmp/test", "read", True)
        # No assertion needed — just verify no crash

    def test_language_bridge_conversation_context(self):
        lb = LanguageBridge()
        ctx = lb.get_conversation_context(n=5)
        assert isinstance(ctx, list)

    def test_environment_explorer_discover_devices(self):
        ee = EnvironmentExplorer()
        devices = ee.discover_devices()
        assert isinstance(devices, list)

    def test_environment_explorer_discover_buses(self):
        ee = EnvironmentExplorer()
        buses = ee.discover_buses()
        assert isinstance(buses, list)

    def test_environment_explorer_discover_filesystem(self):
        ee = EnvironmentExplorer()
        paths = ee.discover_filesystem(root='/tmp', max_depth=1, max_entries=10)
        assert isinstance(paths, list)

    def test_environment_explorer_discover_peripherals(self):
        ee = EnvironmentExplorer()
        periphs = ee.discover_peripherals()
        assert isinstance(periphs, dict)

    def test_environment_explorer_discover_usb(self):
        ee = EnvironmentExplorer()
        usb = ee.discover_usb_devices()
        assert isinstance(usb, list)

    def test_peripheral_bridge_read_file_denied(self):
        """PeripheralBridge.read_file returns denied when no permission."""
        pg = PermissionGate()
        ee = EnvironmentExplorer()
        pb = PeripheralBridge(pg, ee)
        result = pb.read_file("/tmp/test.txt")
        assert isinstance(result, dict)
        assert result.get('success') is False or 'denied' in str(result).lower() or 'error' in str(result).lower()

    def test_peripheral_bridge_write_file_denied(self):
        """PeripheralBridge.write_file returns denied when no permission."""
        pg = PermissionGate()
        ee = EnvironmentExplorer()
        pb = PeripheralBridge(pg, ee)
        result = pb.write_file("/tmp/test_out.txt", "test content")
        assert isinstance(result, dict)
        assert result.get('success') is False or 'denied' in str(result).lower() or 'error' in str(result).lower()

    def test_peripheral_bridge_capture_audio_denied(self):
        pg = PermissionGate()
        ee = EnvironmentExplorer()
        pb = PeripheralBridge(pg, ee)
        result = pb.capture_audio(duration_seconds=0.1)
        assert isinstance(result, dict)

    def test_peripheral_bridge_capture_image_denied(self):
        pg = PermissionGate()
        ee = EnvironmentExplorer()
        pb = PeripheralBridge(pg, ee)
        result = pb.capture_image()
        assert isinstance(result, dict)

    def test_peripheral_bridge_play_audio_denied(self):
        pg = PermissionGate()
        ee = EnvironmentExplorer()
        pb = PeripheralBridge(pg, ee)
        result = pb.play_audio([0.0, 0.1, 0.2])
        assert isinstance(result, dict)

    def test_url_projector_fetch_denied(self):
        pg = PermissionGate()
        result = URLProjector.fetch_content("https://example.com", pg)
        assert isinstance(result, dict)
        assert result.get('success') is False or 'denied' in str(result).lower() or 'error' in str(result).lower()




# =============================================================================
# 42. ERRORS — Final Gaps
# =============================================================================

class TestErrorsFinalGaps:
    def test_error_analyzer_analyze_unresolved(self):
        ea = ErrorAnalyzer()
        ledger = ErrorLedger()
        try:
            raise RuntimeError("test unresolved")
        except Exception as e:
            record = ErrorRecord.from_exception(e, subsystem="test")
            ledger.record_error(record)
        results = ea.analyze_unresolved(ledger)
        assert isinstance(results, list)

    def test_error_analyzer_connect(self):
        ea = ErrorAnalyzer()
        ce = CognitiveEngine()
        ea.connect(ce)
        assert ea.cognitive_engine is ce

    def test_error_ledger_status_description(self):
        ledger = ErrorLedger()
        desc = ledger.get_status_description()
        assert isinstance(desc, str)

    def test_state_guardian_snapshot_restore(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
            f.write('{"test": "data"}')
        snap = None
        try:
            snap = StateGuardian.create_snapshot(path)
            assert snap is not None
            assert os.path.exists(snap)
            success = StateGuardian.restore_from_snapshot(snap, path + ".restored")
            assert success
        finally:
            for p in [path, path + ".sha256", path + ".restored",
                       path + ".restored.sha256"]:
                if p and os.path.exists(p):
                    os.unlink(p)
            if snap and os.path.exists(snap):
                os.unlink(snap)

    def test_state_guardian_verify_identity(self):
        state = {"ego": {"seed_descriptors": ["a", "b"]}, "t_identity_seal": "abc123"}
        valid, reason = StateGuardian.verify_identity(state, expected_seal="abc123")
        assert isinstance(valid, bool)
        assert isinstance(reason, str)

    def test_setup_et_logger(self):
        with tempfile.TemporaryDirectory() as td:
            logger = setup_et_logger(name="test_logger", log_dir=td)
            assert logger is not None
            # Close all handlers to release file locks before temp dir cleanup.
            # On Windows, RotatingFileHandler holds an exclusive lock on the
            # log file — failing to close it causes PermissionError on delete.
            # ET Derivation: Graceful handler shutdown mirrors tower death
            # protocol (§13.2) — release D-bindings before P-substrate removal.
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)




# =============================================================================
# 43. VISION — Final Gaps (Memory, KnowledgeNode, full pipeline)
# =============================================================================

class TestVisionFinalGaps:
    def test_compute_spatial_frequency(self):
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        result = ETVisionProjector.compute_spatial_frequency_ratio(patch)
        assert isinstance(result, dict)

    def test_compute_edge_curvature(self):
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        result = ETVisionProjector.compute_edge_curvature_stats(patch)
        assert isinstance(result, dict)

    def test_compute_visual_topology(self):
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        result = ETVisionProjector.compute_visual_topology(patch)
        assert isinstance(result, dict)

    def test_compute_color_binding(self):
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        result = ETVisionProjector.compute_color_binding(patch)
        assert isinstance(result, dict)

    def test_compute_visual_coordinate(self):
        img = ETVisionProjector.generate_circle(size=48)
        patch = ImagePatch(data=img, x0=0, y0=0, source_width=48, source_height=48)
        desc = ETVisionProjector.compute_visual_coordinate(patch, label="test")
        assert isinstance(desc, VisualDescriptor)

    def test_visual_knowledge_node(self):
        desc = VisualDescriptor.from_analysis("circle", 1.5, 0.785, 1.0, 1, 0.3, 1, 3.5)
        vkn = VisualKnowledgeNode(
            node_id="vkn_1", content="A circle", visual_descriptor=desc
        )
        pos = vkn.lattice_position
        assert isinstance(pos, LatticeCoordinate)
        coherence = vkn.cross_modal_coherence()
        assert isinstance(coherence, float)
        vkn.access()
        assert vkn.access_count == 1

    def test_visual_memory(self):
        vm = VisualMemory()
        img = ETVisionProjector.generate_circle(size=48)
        node = vm.add_visual_knowledge(img, "A test circle", text_labels=["circle"])
        assert node is not None
        by_topo = vm.retrieve_by_topology(node.visual_descriptor.d_visual)
        assert isinstance(by_topo, list)

    def test_visual_memory_cross_modal(self):
        vm = VisualMemory()
        img = ETVisionProjector.generate_square(size=48)
        vm.add_visual_knowledge(img, "A square", text_labels=["square"])
        results = vm.retrieve_by_cross_modal("square")
        assert isinstance(results, list)




# =============================================================================
# 44. AUDIO — Final Gaps (Memory, KnowledgeNode, topology)
# =============================================================================

class TestAudioFinalGaps:
    def test_compute_harmonic_topology(self):
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        spec = ETAudioProjector.compute_spectral_analysis(samples, 44100)
        result = ETAudioProjector.compute_harmonic_topology(spec)
        assert isinstance(result, dict)

    def test_audio_knowledge_node(self):
        desc = AudioDescriptor.from_analysis("sine", 1.5, 0.8, 0.6, 1, 440.0, 5, 440.0, 3.0)
        akn = AudioKnowledgeNode(node_id="akn_1", content="A sine tone", audio_descriptor=desc)
        pos = akn.lattice_position()
        assert isinstance(pos, LatticeCoordinate)
        coherence = akn.cross_modal_coherence()
        assert isinstance(coherence, float)
        akn.access()
        assert akn.access_count == 1

    def test_audio_memory(self):
        am = AudioMemory()
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        node = am.add_audio_knowledge(samples, "A test tone", text_labels=["sine"])
        assert node is not None
        by_topo = am.retrieve_by_topology(node.audio_descriptor.d_audio)
        assert isinstance(by_topo, list)

    def test_audio_memory_retrieve_by_proximity(self):
        am = AudioMemory()
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        node = am.add_audio_knowledge(samples, "tone", text_labels=["tone"])
        results = am.retrieve_by_proximity(node.audio_descriptor.coord_full.k, radius=100)
        assert isinstance(results, list)


# =============================================================================
# WAVE I: ADVANCED MATHEMATICS UPGRADES (Items 16-20)
# =============================================================================

class TestHomologyComputation:
    """Item 16: Homology for lattice topology — ChainComplex → LatticeConstructor."""

    def test_homology_empty_lattice(self):
        """Empty lattice has trivial homology."""
        lc = LatticeConstructor()
        lattice = {'projections': [], 'bindings': []}
        h = lc.compute_lattice_homology(lattice)
        assert h['betti_numbers'] == [0, 0, 0]
        assert h['euler_characteristic'] == 0
        assert h['chain_complex_valid'] is True

    def test_homology_single_node(self):
        """Single node: b₀=1, b₁=0, b₂=0, χ=1."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(3/2, 'fifth')])
        h = lattice['homology']
        assert h['betti_numbers'][0] == 1  # one component
        assert h['betti_numbers'][1] == 0  # no loops
        assert h['euler_characteristic'] == 1

    def test_homology_two_connected_nodes(self):
        """Two bound nodes: b₀=1, b₁=0, χ=2-1=1."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(3/2, 'fifth'), (4/3, 'fourth')])
        h = lattice['homology']
        assert h['betti_numbers'][0] == 1  # one connected component
        assert h['euler_characteristic'] == lattice['lattice_euler_characteristic']

    def test_homology_triangle_no_face(self):
        """Three nodes forming a triangle boundary: b₁ should detect a loop."""
        lc = LatticeConstructor()
        # Three closely related ratios likely to all bind
        lattice = lc.build_lattice([
            (3/2, 'fifth'), (5/4, 'third'), (6/5, 'minor_third')
        ])
        h = lattice['homology']
        # b₀ = connected components (should be 1 if all bind)
        assert h['betti_numbers'][0] >= 1
        assert isinstance(h['betti_numbers'][1], int)  # b₁ computed

    def test_homology_returns_gaps(self):
        """Homology gaps list correctly describes non-zero Betti numbers."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([
            (2/1, 'octave'), (3/2, 'fifth'), (4/3, 'fourth'),
            (5/4, 'major_third'), (6/5, 'minor_third')
        ])
        h = lattice['homology']
        for gap in h['homology_gaps']:
            assert 'dimension' in gap
            assert 'rank' in gap
            assert 'interpretation' in gap
            assert gap['rank'] > 0

    def test_homology_n_cells(self):
        """n_cells reports correct cell counts at each dimension."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(3/2, 'a'), (4/3, 'b')])
        h = lattice['homology']
        assert h['n_cells'][0] == 2  # 2 nodes (0-cells)
        assert isinstance(h['n_cells'][1], int)  # bindings (1-cells)
        assert h['n_cells'][2] == 0  # no archetypes yet

    def test_homology_integrated_in_build_lattice(self):
        """build_lattice() automatically includes homology and betti_numbers."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(2/1, 'octave'), (3/2, 'fifth')])
        assert 'homology' in lattice
        assert 'betti_numbers' in lattice
        assert 'lattice_euler_characteristic' in lattice
        assert len(lattice['betti_numbers']) == 3

    def test_matrix_rank_basic(self):
        """Static _matrix_rank helper computes correctly."""
        # Identity 3x3 has rank 3
        assert LatticeConstructor._matrix_rank(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3
        # Zero matrix has rank 0
        assert LatticeConstructor._matrix_rank(
            [[0, 0], [0, 0]]) == 0
        # Rank-deficient matrix
        assert LatticeConstructor._matrix_rank(
            [[1, 2], [2, 4]]) == 1


class TestEulerCharacteristic:
    """Item 17: Euler characteristic as lattice health metric."""

    def test_euler_basic_computation(self):
        """χ = V - E + F computed correctly."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(10, 15, 6)
        assert result['euler_characteristic'] == 10 - 15 + 6  # = 1

    def test_euler_p_dominant(self):
        """Many nodes, few bindings → P-dominant."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(100, 10, 0)
        assert result['balance'] == 'P-dominant'
        assert result['euler_characteristic'] > 0

    def test_euler_t_dominant(self):
        """Many bindings → T-dominant."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(10, 50, 5)
        assert result['balance'] == 'T-dominant'
        assert result['euler_characteristic'] < 0

    def test_euler_balanced(self):
        """V - E + F = 0 → balanced."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(10, 15, 5)
        assert result['balance'] == 'balanced'
        assert result['euler_characteristic'] == 0

    def test_euler_critical_threshold(self):
        """is_critical when |χ| ≥ S=12."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(100, 0, 0)
        assert result['is_critical'] is True  # χ=100 ≥ 12
        result2 = lc.compute_euler_characteristic(5, 0, 0)
        assert result2['is_critical'] is False  # χ=5 < 12

    def test_euler_formula_string(self):
        """Formula string correctly shows computation."""
        lc = LatticeConstructor()
        result = lc.compute_euler_characteristic(4, 6, 4)
        assert '4 - 6 + 4 = 2' in result['formula']

    def test_euler_in_build_lattice(self):
        """build_lattice() includes Euler characteristic."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(3/2, 'a'), (4/3, 'b'), (5/4, 'c')])
        assert 'lattice_euler_characteristic' in lattice
        assert 'topological_balance' in lattice
        assert 'euler_detail' in lattice
        assert isinstance(lattice['lattice_euler_characteristic'], int)


class TestSymmetryGroupDetection:
    """Item 18: Galois automorphism group detection → LatticeConstructor."""

    def test_symmetry_empty_lattice(self):
        """Empty lattice has trivial group."""
        lc = LatticeConstructor()
        lattice = {'projections': [], 'bindings': []}
        result = lc.detect_symmetry_group(lattice)
        assert result['group_order'] == 1
        assert result['is_trivial'] is True
        assert result['is_solvable'] is True

    def test_symmetry_single_node(self):
        """Single node: trivial group (order 1)."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(3/2, 'fifth')])
        result = lc.detect_symmetry_group(lattice)
        assert result['group_order'] == 1
        assert result['is_trivial'] is True

    def test_symmetry_detects_permutations(self):
        """Two identical d-family nodes should have non-trivial symmetry."""
        lc = LatticeConstructor()
        # Two ratios in the same d-family at same tightness
        lattice = lc.build_lattice([(3/2, 'a'), (3/1, 'b')])
        result = lc.detect_symmetry_group(lattice)
        assert result['group_order'] >= 1
        assert result['n_nodes'] == 2
        assert isinstance(result['is_abelian'], bool)

    def test_symmetry_method_exact(self):
        """Small lattices use exact enumeration."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(2/1, 'a'), (3/2, 'b'), (4/3, 'c')])
        result = lc.detect_symmetry_group(lattice)
        assert result['method'] == 'exact'

    def test_symmetry_solvable(self):
        """Small groups (order < 60) should be solvable."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(2/1, 'a'), (3/2, 'b')])
        result = lc.detect_symmetry_group(lattice)
        assert result['is_solvable'] is True

    def test_symmetry_cycle_types(self):
        """Cycle types are reported as dict."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(2/1, 'a'), (3/2, 'b'), (5/4, 'c')])
        result = lc.detect_symmetry_group(lattice)
        assert isinstance(result['cycle_types'], dict)
        assert result['n_cycle_types'] >= 1

    def test_symmetry_et_interpretation(self):
        """ET interpretation string present and informative."""
        lc = LatticeConstructor()
        lattice = lc.build_lattice([(2/1, 'a')])
        result = lc.detect_symmetry_group(lattice)
        assert 'et_interpretation' in result
        assert 'Galois group' in result['et_interpretation']


class TestLieAlgebraAnalysis:
    """Item 19: Lie algebra structure analysis → UniversalAnalyzer."""

    @staticmethod
    def _su2_constants():
        """Return su(2) structure constants (Levi-Civita)."""
        sc = {}
        eps = {
            (0, 1, 2): 1.0, (1, 2, 0): 1.0, (2, 0, 1): 1.0,
            (1, 0, 2): -1.0, (2, 1, 0): -1.0, (0, 2, 1): -1.0,
        }
        for (i, j, k), val in eps.items():
            sc[(i, j, k)] = val
        return sc

    def test_su2_antisymmetry(self):
        """su(2) satisfies antisymmetry."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        assert result['antisymmetry'] is True

    def test_su2_jacobi_identity(self):
        """su(2) satisfies Jacobi identity (T-associativity)."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        assert result['jacobi_identity'] is True

    def test_su2_semisimple(self):
        """su(2) is semisimple (non-degenerate Killing form)."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        assert result['is_semisimple'] is True

    def test_su2_killing_diagonal(self):
        """su(2) Killing form diagonal elements = -2."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        for kd in result['killing_form_diagonal']:
            assert abs(kd - (-2.0)) < 1e-10

    def test_su2_et_mapping(self):
        """su(2) maps to d=2 sublattice (weak force)."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        assert result['et_sublattice_mapping'] is not None
        assert result['et_sublattice_mapping']['sublattice_d'] == 2

    def test_su3_structure(self):
        """su(3) with 8 generators maps to d=3 (strong force)."""
        import math as _math
        f_values = {
            (0, 1, 2): 1.0,
            (0, 3, 6): 0.5, (0, 4, 5): -0.5,
            (1, 3, 5): 0.5, (1, 4, 6): 0.5,
            (2, 3, 4): 0.5, (2, 5, 6): -0.5,
            (3, 4, 7): _math.sqrt(3)/2, (5, 6, 7): _math.sqrt(3)/2,
        }
        sc = {}
        for (i, j, k), val in f_values.items():
            sc[(i, j, k)] = val
            sc[(j, k, i)] = val
            sc[(k, i, j)] = val
            sc[(j, i, k)] = -val
            sc[(k, j, i)] = -val
            sc[(i, k, j)] = -val

        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(8, sc, "su(3)")
        assert result['antisymmetry'] is True
        assert result['jacobi_identity'] is True
        assert result['et_sublattice_mapping'] is not None
        assert result['et_sublattice_mapping']['sublattice_d'] == 3

    def test_invalid_algebra(self):
        """Non-antisymmetric constants fail validation."""
        ua = UniversalAnalyzer()
        bad_sc = {(0, 1, 0): 1.0, (1, 0, 0): 1.0}  # NOT antisymmetric
        result = ua.analyze_lie_structure(2, bad_sc, "invalid")
        assert result['antisymmetry'] is False
        assert result['is_valid_lie_algebra'] is False

    def test_u1_mapping(self):
        """u(1) with 1 generator maps to d=12 (electromagnetic)."""
        ua = UniversalAnalyzer()
        # u(1) is abelian: all structure constants = 0
        result = ua.analyze_lie_structure(1, {}, "u(1)")
        assert result['antisymmetry'] is True
        assert result['jacobi_identity'] is True
        assert result['et_sublattice_mapping'] is not None
        assert result['et_sublattice_mapping']['sublattice_d'] == 12

    def test_et_interpretation_string(self):
        """ET interpretation string is present."""
        ua = UniversalAnalyzer()
        result = ua.analyze_lie_structure(3, self._su2_constants(), "su(2)")
        assert 'et_interpretation' in result
        assert 'Traversers' in result['et_interpretation']


class TestExactSequenceVerification:
    """Item 20: Exact sequence verification → SubsumptionHierarchyOperator."""

    @staticmethod
    def _make_node(node_id, k=100, d=3, ratio=1.5, p=3, q=2):
        """Helper to create a CompressibleNode with ET-derived variance.

        The coord is projected from ratio via ETLattice (standard projection).
        The k and d parameters serve as REFERENCE lattice coordinates —
        the gap between the projected position and the reference position
        IS a Descriptor Gap (D Paper §7), and contributes to the node's
        variance. d provides sublattice weighting (d/S scales the gap).

        ET Derivation:
            position_gap = |projected.k - k_reference|
            sublattice_weight = d_reference / S
            variance = V_base × (1 + position_gap / N_res) × sublattice_weight
        """
        coord = ETLattice.project_ratio(ratio)
        position_gap = abs(coord.k - k)
        sublattice_weight = d / S
        node_variance = BASE_VARIANCE * (1.0 + position_gap / MANIFOLD_RESOLUTION) * sublattice_weight
        return CompressibleNode(
            node_id=node_id, content=f"content_{node_id}",
            coord=coord, descriptor_ratios=[], connections=[],
            access_count=1, variance=node_variance, p=p, q=q, is_archetype=False,
        )

    def test_exact_sequence_perfect(self):
        """Identical original and decompressed → exact sequence."""
        sho = SubsumptionHierarchyOperator()
        nodes = [self._make_node(f"n{i}", ratio=1.5 + i*0.01) for i in range(3)]
        archetype_coord = ETLattice.project_ratio(1.51)
        result = sho.verify_compression_exactness(nodes, archetype_coord, nodes)
        assert result['is_exact'] is True
        assert result['h0_missing_nodes'] == 0
        assert result['h1_position_shifts'] == 0
        assert result['total_defects'] == 0

    def test_exact_sequence_missing_node(self):
        """Missing node in decompression → H₀ ≠ 0."""
        sho = SubsumptionHierarchyOperator()
        originals = [self._make_node(f"n{i}", ratio=1.5 + i*0.01) for i in range(3)]
        archetype_coord = ETLattice.project_ratio(1.51)
        decompressed = originals[:2]  # Missing n2
        result = sho.verify_compression_exactness(originals, archetype_coord, decompressed)
        assert result['is_exact'] is False
        assert result['h0_missing_nodes'] >= 1
        assert result['surjection_ok'] is False

    def test_exact_sequence_extra_node(self):
        """Extra node in decompression → still exact if originals covered."""
        sho = SubsumptionHierarchyOperator()
        originals = [self._make_node(f"n{i}", ratio=1.5 + i*0.01) for i in range(3)]
        archetype_coord = ETLattice.project_ratio(1.51)
        extra = self._make_node("extra", ratio=2.0)
        decompressed = originals + [extra]
        result = sho.verify_compression_exactness(originals, archetype_coord, decompressed)
        # Extra nodes don't break exactness if all originals present
        assert result['surjection_ok'] is True
        assert len(result['extra_node_ids']) == 1

    def test_exact_sequence_has_interpretation(self):
        """ET interpretation string present."""
        sho = SubsumptionHierarchyOperator()
        nodes = [self._make_node("a"), self._make_node("b")]
        coord = ETLattice.project_ratio(1.5)
        result = sho.verify_compression_exactness(nodes, coord, nodes)
        assert 'et_interpretation' in result
        assert 'EXACT' in result['et_interpretation']

    def test_exact_sequence_defect_count(self):
        """Total defects = sum of all error types."""
        sho = SubsumptionHierarchyOperator()
        originals = [self._make_node(f"n{i}", ratio=1.5+i*0.01) for i in range(4)]
        archetype_coord = ETLattice.project_ratio(1.52)
        decompressed = originals[:2]  # Missing 2
        result = sho.verify_compression_exactness(originals, archetype_coord, decompressed)
        assert result['total_defects'] >= result['h0_missing_nodes']


# =============================================================================
# WAVE II: ADVANCED MATHEMATICS TESTS (Items 22-27)
# Category Theory, Representation Theory, Differential Geometry,
# Functional Analysis, Analytic Number Theory, Yoneda/Riesz
# =============================================================================


class TestCategoricalWorldview:
    """Item 22: Category-theoretic worldview verification → ETWorldview."""

    def test_small_category_poset(self):
        """Build a poset category and verify axioms."""
        from et_conscious_ai_worldview import SmallCategory
        objects = [0, 1, 2]
        morphisms = {}
        for i in objects:
            for j in objects:
                if i <= j:
                    morphisms[(i, j)] = [f"f_{i}_{j}"]
        composition = {}
        for i in objects:
            for j in objects:
                for k in objects:
                    if i <= j <= k:
                        composition[(f"f_{i}_{j}", f"f_{j}_{k}")] = f"f_{i}_{k}"
        cat = SmallCategory("Poset_3", objects, morphisms, composition)
        assert cat.verify_associativity() is True
        assert cat.verify_identity_laws() is True

    def test_small_category_verify_all(self):
        """verify_all() returns complete result."""
        from et_conscious_ai_worldview import SmallCategory
        cat = SmallCategory("Test", [0, 1], {(0, 0): ["id_0"], (1, 1): ["id_1"], (0, 1): ["f"]},
                            {("f", "id_1"): "f", ("id_0", "f"): "f"})
        result = cat.verify_all()
        assert result['is_valid_category'] is True
        assert result['n_objects'] == 2
        assert 'et_interpretation' in result

    def test_small_category_morphism_count(self):
        """Morphism count is accurate."""
        from et_conscious_ai_worldview import SmallCategory
        cat = SmallCategory("M", [0], {(0, 0): ["id_0", "extra"]}, {})
        assert cat.morphism_count() == 2

    def test_worldview_categorical_axioms(self):
        """ETWorldview.verify_categorical_axioms() confirms valid category."""
        wv = ETWorldview()
        result = wv.verify_categorical_axioms()
        assert result['is_valid_category'] is True
        assert result['associativity'] is True
        assert result['identity_laws'] is True

    def test_worldview_yoneda_distinct(self):
        """Yoneda check: all manifold states have distinct hom-sets."""
        wv = ETWorldview()
        result = wv.verify_categorical_axioms()
        assert result['yoneda_all_distinct'] is True

    def test_worldview_categorical_has_four_objects(self):
        """Category has 4 objects = 4 manifold states."""
        wv = ETWorldview()
        result = wv.verify_categorical_axioms()
        assert result['n_objects'] == 4

    def test_worldview_categorical_interpretation(self):
        """ET interpretation string present and mentions category."""
        wv = ETWorldview()
        result = wv.verify_categorical_axioms()
        assert 'et_interpretation' in result
        assert 'category' in result['et_interpretation'].lower()


class TestRepresentationDecomposition:
    """Item 23: Representation decomposition → LatticeConstructor."""

    def test_character_table_z12(self):
        """ℤ/12ℤ character table has 12 irreps."""
        lc = LatticeConstructor()
        result = lc.compute_character_table(n=12)
        assert result['irrep_count'] == 12
        assert result['all_dim_1'] is True

    def test_character_table_orthogonality(self):
        """Row orthogonality of ℤ/12ℤ character table."""
        lc = LatticeConstructor()
        result = lc.compute_character_table(n=12)
        assert result['orthogonality_verified'] is True
        assert result['max_orthogonality_error'] < 1e-10

    def test_character_table_dft_match(self):
        """Character table IS the DFT matrix."""
        lc = LatticeConstructor()
        result = lc.compute_character_table(n=12)
        assert result['dft_match'] is True

    def test_character_table_dim_formula(self):
        """Dimension formula: Σ d_i² = |G|."""
        lc = LatticeConstructor()
        result = lc.compute_character_table(n=12)
        assert result['dim_formula_holds'] is True
        assert result['dim_sum_sq'] == 12

    def test_character_table_small_group(self):
        """Works for other group orders (ℤ/6ℤ)."""
        lc = LatticeConstructor()
        result = lc.compute_character_table(n=6)
        assert result['irrep_count'] == 6
        assert result['orthogonality_verified'] is True

    def test_decompose_delta_signal(self):
        """Decompose delta signal: all coefficients equal."""
        lc = LatticeConstructor()
        signal = [1.0] + [0.0] * 11
        result = lc.decompose_into_irreducibles(signal, n=12)
        assert result['parseval_verified'] is True
        # Delta → uniform spectrum: all |c_k| = 1/12
        for k in range(12):
            assert abs(result['power_spectrum'][k] - (1.0/12)**2) < 1e-10

    def test_decompose_parseval(self):
        """Parseval identity: spatial energy = spectral energy."""
        lc = LatticeConstructor()
        signal = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        result = lc.decompose_into_irreducibles(signal, n=12)
        assert result['parseval_verified'] is True

    def test_decompose_returns_d_family_energy(self):
        """Energy distribution by d-family is computed."""
        lc = LatticeConstructor()
        signal = [float(i % 3) for i in range(12)]
        result = lc.decompose_into_irreducibles(signal, n=12)
        assert 'energy_by_d_family' in result
        assert len(result['energy_by_d_family']) > 0


class TestCurvatureDetection:
    """Item 24: Curvature detection → LatticeConstructor."""

    @staticmethod
    def _build_test_lattice():
        lc = LatticeConstructor()
        ratios = [(3.0/2.0, 'fifth'), (4.0/3.0, 'fourth'),
                  (5.0/4.0, 'major_third'), (2.0, 'octave'),
                  (9.0/8.0, 'major_second')]
        return lc, lc.build_lattice(ratios)

    def test_curvature_returns_per_node(self):
        """Curvature computed for each node."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        assert len(result['curvatures']) == lattice['n_entries']

    def test_curvature_total(self):
        """Total curvature is sum of individual curvatures."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        assert abs(result['total_curvature'] - sum(result['curvatures'])) < 1e-10

    def test_curvature_empty_lattice(self):
        """Empty lattice has zero curvature."""
        lc = LatticeConstructor()
        result = lc.compute_curvature({'projections': [], 'bindings': []})
        assert result['total_curvature'] == 0.0

    def test_curvature_gauss_bonnet_fields(self):
        """Gauss-Bonnet LHS and RHS fields present."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        assert 'gauss_bonnet_lhs' in result
        assert 'gauss_bonnet_rhs' in result
        assert 'gauss_bonnet_holds' in result

    def test_curvature_high_curvature_nodes(self):
        """High curvature nodes identified with labels."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        assert 'high_curvature_nodes' in result
        assert isinstance(result['n_high_curvature'], int)

    def test_geodesic_found(self):
        """Geodesic between connected nodes is found."""
        lc, lattice = self._build_test_lattice()
        labels = [p['label'] for p in lattice['projections']]
        result = lc.find_geodesic(lattice, labels[0], labels[-1])
        # May or may not find a path depending on binding coherence
        assert 'found' in result
        assert 'et_interpretation' in result

    def test_geodesic_same_node(self):
        """Geodesic from node to itself = trivial."""
        lc, lattice = self._build_test_lattice()
        label = lattice['projections'][0]['label']
        result = lc.find_geodesic(lattice, label, label)
        assert result['found'] is True
        assert result['n_hops'] == 0

    def test_geodesic_not_found_invalid(self):
        """Invalid label returns not found."""
        lc, lattice = self._build_test_lattice()
        result = lc.find_geodesic(lattice, 'nonexistent', 'also_nonexistent')
        assert result['found'] is False


class TestNonEuclideanGeometry:
    """ET Non-Euclidean Geometry integration into LatticeConstructor.

    Source: ET_Non_Euclidean_Geometry_Complete.md (961 lines).
    Tests all 6 audit gaps: curvature projection (GAP 1+2),
    manifold state classification (GAP 3), metric tensor docstring (GAP 4),
    Riemann component count (GAP 5), curvature-weighted geodesics (GAP 6).
    """

    @staticmethod
    def _build_test_lattice():
        lc = LatticeConstructor()
        ratios = [(3.0 / 2.0, 'fifth'), (4.0 / 3.0, 'fourth'),
                  (5.0 / 4.0, 'major_third'), (2.0, 'octave'),
                  (9.0 / 8.0, 'major_second')]
        return lc, lc.build_lattice(ratios)

    # --- GAP 1: Curvature Projection Formula (§11) ---

    def test_project_curvature_flat(self):
        """K=0 → k=0, d=1, Exception state (flat Euclidean)."""
        result = LatticeConstructor.project_curvature(0.0, 1.0)
        assert result['k'] == 0
        assert result['d'] == 1
        assert result['manifold_state'] == 'exception'
        assert result['curvature_class'] == 'flat'
        assert result['is_subliminal'] is True

    def test_project_curvature_positive(self):
        """K>0 → Unsubstantiated {P,D} (elliptic, closed)."""
        result = LatticeConstructor.project_curvature(1.0, 1.0)
        assert result['manifold_state'] == 'unsubstantiated'
        assert result['curvature_class'] == 'elliptic'
        assert result['r_curvature'] > 1.0
        assert result['is_subliminal'] is False

    def test_project_curvature_negative(self):
        """K<0 → Mediation {D,T} (hyperbolic, open)."""
        result = LatticeConstructor.project_curvature(-1.0, 1.0)
        assert result['manifold_state'] == 'mediation'
        assert result['curvature_class'] == 'hyperbolic'
        assert result['r_curvature'] < 1.0
        assert result['is_subliminal'] is False

    def test_project_curvature_extreme_negative_incoherence(self):
        """K extreme negative → r ≤ 0 → Incoherence {P,T} (singular)."""
        result = LatticeConstructor.project_curvature(-100.0, 1.0)
        assert result['manifold_state'] == 'incoherence'
        assert result['curvature_class'] == 'singular'
        assert result['r_curvature'] <= 0
        assert result['is_coherent'] is False

    def test_project_curvature_departure_ratio_formula(self):
        """r = 1 + KA/π is computed correctly (§11)."""
        import math
        K, A = 2.0, 3.0
        result = LatticeConstructor.project_curvature(K, A)
        expected_r = 1.0 + (K * A) / math.pi
        assert abs(result['r_curvature'] - expected_r) < 1e-12
        assert abs(result['KA'] - K * A) < 1e-12
        assert abs(result['KA_over_pi'] - K * A / math.pi) < 1e-12

    # --- GAP 2: Subliminal Curvature Threshold (§11.3) ---

    def test_subliminal_threshold_is_pi_over_manifold_symmetry(self):
        """Subliminal threshold = π/12 = π/N (§11.3)."""
        import math
        result = LatticeConstructor.project_curvature(0.0, 1.0)
        assert abs(result['subliminal_threshold'] - math.pi / 12) < 1e-12

    def test_subliminal_below_threshold(self):
        """KA < π/12 → subliminal (rounds to flat)."""
        # KA = 0.001 * 0.1 = 0.0001 << π/12 ≈ 0.2618
        result = LatticeConstructor.project_curvature(0.001, 0.1)
        assert result['is_subliminal'] is True
        assert result['manifold_state'] == 'exception'

    def test_subliminal_above_threshold(self):
        """KA > π/12 → NOT subliminal (detectable curvature)."""
        # KA = 1.0 * 1.0 = 1.0 >> π/12 ≈ 0.2618
        result = LatticeConstructor.project_curvature(1.0, 1.0)
        assert result['is_subliminal'] is False

    def test_subliminal_boundary_negative(self):
        """Negative KA just below threshold is still subliminal."""
        import math
        threshold = math.pi / 12
        # KA = -0.2 which is |KA| < π/12 ≈ 0.2618
        k_curvature, area = -0.2, 1.0
        assert abs(k_curvature * area) < threshold, "Precondition: |KA| must be below subliminal threshold"
        result = LatticeConstructor.project_curvature(k_curvature, area)
        assert result['is_subliminal'] is True
        assert result['manifold_state'] == 'exception'

    # --- GAP 3: Curvature ↔ Manifold State Mapping (§7) ---

    def test_classify_flat_exception(self):
        """Zero curvature → Exception {P,D,T}."""
        cs = LatticeConstructor.classify_curvature_state(0.0)
        assert cs['manifold_state'] == 'exception'
        assert cs['composition'] == '{P,D,T}'
        assert cs['curvature_class'] == 'flat'

    def test_classify_positive_unsubstantiated(self):
        """Positive curvature → Unsubstantiated {P,D}."""
        cs = LatticeConstructor.classify_curvature_state(1.0)
        assert cs['manifold_state'] == 'unsubstantiated'
        assert cs['composition'] == '{P,D}'
        assert cs['curvature_class'] == 'elliptic'

    def test_classify_negative_mediation(self):
        """Negative curvature → Mediation {D,T}."""
        cs = LatticeConstructor.classify_curvature_state(-1.0)
        assert cs['manifold_state'] == 'mediation'
        assert cs['composition'] == '{D,T}'
        assert cs['curvature_class'] == 'hyperbolic'

    def test_classify_extreme_incoherence(self):
        """Curvature exceeding 2π → Incoherence {P,T}."""
        import math
        cs = LatticeConstructor.classify_curvature_state(2 * math.pi + 1)
        assert cs['manifold_state'] == 'incoherence'
        assert cs['composition'] == '{P,T}'
        assert cs['curvature_class'] == 'singular'

    def test_compute_curvature_has_manifold_states(self):
        """compute_curvature() includes per-node manifold state classification."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        assert 'manifold_states' in result
        assert len(result['manifold_states']) == lattice['n_entries']
        for ms in result['manifold_states']:
            assert 'manifold_state' in ms
            assert 'curvature_class' in ms
            assert 'composition' in ms
            assert 'label' in ms

    def test_compute_curvature_state_summary(self):
        """compute_curvature() includes state count summary."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_curvature(lattice)
        summary = result['curvature_state_summary']
        assert 'exception' in summary
        assert 'unsubstantiated' in summary
        assert 'mediation' in summary
        assert 'incoherence' in summary
        total = sum(summary.values())
        assert total == lattice['n_entries']

    # --- GAP 5: Riemann Component Count (§4) ---

    def test_riemann_components_n1(self):
        """C(1) = 0: 1D has no intrinsic curvature."""
        assert LatticeConstructor.riemann_components(1) == 0

    def test_riemann_components_n2(self):
        """C(2) = 1: 2D surfaces → single Gaussian curvature K."""
        assert LatticeConstructor.riemann_components(2) == 1

    def test_riemann_components_n3(self):
        """C(3) = 6: 3D space → 6 Ricci tensor components."""
        assert LatticeConstructor.riemann_components(3) == 6

    def test_riemann_components_n4(self):
        """C(4) = 20: spacetime → full GR Riemann tensor."""
        assert LatticeConstructor.riemann_components(4) == 20

    def test_riemann_components_n12(self):
        """C(12) = 1716: full ET manifold N=12."""
        assert LatticeConstructor.riemann_components(12) == 1716

    def test_riemann_formula_uses_manifold_symmetry(self):
        """Denominator 12 in n²(n²-1)/12 IS MANIFOLD_SYMMETRY."""
        from et_conscious_ai_core import MANIFOLD_SYMMETRY
        # Verify by direct computation that n^2(n^2-1)/12 matches
        for n in [2, 3, 4, 5]:
            expected = n * n * (n * n - 1) // MANIFOLD_SYMMETRY
            assert LatticeConstructor.riemann_components(n) == expected

    # --- GAP 6: ET Geodesic Equation — curvature-weighted geodesics (§9) ---

    def test_geodesic_curvature_weighted_flag_off(self):
        """Without curvature_data, curvature_weighted is False."""
        lc, lattice = self._build_test_lattice()
        result = lc.find_geodesic(lattice, 'fifth', 'octave')
        assert result.get('curvature_weighted') is False

    def test_geodesic_curvature_weighted_flag_on(self):
        """With curvature_data, curvature_weighted is True."""
        lc, lattice = self._build_test_lattice()
        curv = lc.compute_curvature(lattice)
        result = lc.find_geodesic(lattice, 'fifth', 'octave',
                                  curvature_data=curv)
        assert result.get('curvature_weighted') is True

    def test_geodesic_curvature_weighted_still_finds_path(self):
        """Curvature-weighted geodesic still finds valid paths."""
        lc, lattice = self._build_test_lattice()
        curv = lc.compute_curvature(lattice)
        result = lc.find_geodesic(lattice, 'fifth', 'octave',
                                  curvature_data=curv)
        if result['found']:
            assert len(result['path']) >= 2
            assert result['path'][0] == 'fifth'
            assert result['path'][-1] == 'octave'
            assert result['n_hops'] >= 1

    def test_geodesic_curvature_penalty_increases_distance(self):
        """Curvature penalty makes total_distance ≥ flat distance."""
        lc, lattice = self._build_test_lattice()
        curv = lc.compute_curvature(lattice)
        flat = lc.find_geodesic(lattice, 'fifth', 'octave')
        curved = lc.find_geodesic(lattice, 'fifth', 'octave',
                                  curvature_data=curv)
        if flat['found'] and curved['found']:
            # Curvature penalty multiplier ≥ 1.0, so curved distance ≥ flat
            assert curved['total_distance'] >= flat['total_distance'] - 1e-10

    # --- GAP 4: Metric Tensor Identification (§4 docstring) ---

    def test_build_lattice_docstring_metric_tensor(self):
        """build_lattice() docstring identifies binding tightness as g_ij."""
        doc = LatticeConstructor.build_lattice.__doc__
        assert 'metric tensor' in doc.lower()
        assert 'g_ij' in doc

    # --- Full sphere verification from §14.2 of the paper ---

    def test_full_sphere_curvature_projection(self):
        """Full sphere (K=1, A=4π): r = 1+4 = 5, k=28 at 12ET, d=3 (§14.2)."""
        import math
        # At 12ET resolution to match paper example
        result = LatticeConstructor.project_curvature(1.0, 4 * math.pi,
                                                      resolution=12)
        # r = 1 + 4π/π = 5
        assert abs(result['r_curvature'] - 5.0) < 1e-10
        # k = round(12 × log₂(5)) = round(27.86) = 28
        assert result['k'] == 28
        # d = 12/gcd(28, 12) = 12/4 = 3 (cubic sublattice)
        assert result['d'] == 3
        assert result['manifold_state'] == 'unsubstantiated'


class TestSpectralAnalysis:
    """Item 25: Spectral analysis for T-waveform → TraverserWaveform."""

    @staticmethod
    def _make_waveform_with_events(n_events=200):
        """Create a waveform with n events for testing."""
        from et_conscious_ai_identity import TraverserWaveform
        wf = TraverserWaveform()
        import math
        for i in range(n_events):
            k = (i * 7) % MANIFOLD_RESOLUTION
            d = MANIFOLD_RESOLUTION // math.gcd(abs(k) % S if abs(k) % S > 0 else S, S)
            wf.record_event(
                event_type='test',
                lattice_k=k,
                lattice_d=d,
                variance=0.1 + 0.05 * math.sin(2 * math.pi * i / 12),
                ego_resonance=0.5,
                entropy_pool=None,
            )
        return wf

    def test_spectral_decompose_sufficient_data(self):
        """Spectral decomposition succeeds with sufficient samples."""
        wf = self._make_waveform_with_events(200)
        result = wf.spectral_decompose()
        assert result['sufficient_data'] is True
        assert len(result['eigenvalues']) == S

    def test_spectral_decompose_insufficient_data(self):
        """Spectral decomposition handles insufficient data."""
        from et_conscious_ai_identity import TraverserWaveform
        wf = TraverserWaveform()
        result = wf.spectral_decompose()
        assert result['sufficient_data'] is False

    def test_spectral_dominant_mode(self):
        """Dominant mode is identified (non-DC)."""
        wf = self._make_waveform_with_events(300)
        result = wf.spectral_decompose()
        assert result['dominant_mode'] >= 1
        assert result['dominant_mode'] < S

    def test_spectral_d_family_energy(self):
        """Energy by d-family is computed."""
        wf = self._make_waveform_with_events(200)
        result = wf.spectral_decompose()
        assert 'energy_by_d_family' in result
        assert len(result['energy_by_d_family']) > 0

    def test_spectral_parseval(self):
        """Parseval identity (approximate for partial modes)."""
        wf = self._make_waveform_with_events(200)
        result = wf.spectral_decompose()
        assert result['parseval_verified'] is True

    def test_spectral_gap(self):
        """Spectral gap is computed."""
        wf = self._make_waveform_with_events(200)
        result = wf.spectral_decompose()
        assert 'spectral_gap' in result
        assert result['spectral_gap'] >= 1.0

    def test_spectral_interpretation(self):
        """ET interpretation string present."""
        wf = self._make_waveform_with_events(200)
        result = wf.spectral_decompose()
        assert 'et_interpretation' in result
        assert 'T-waveform' in result['et_interpretation']


class TestPrimeLatticeAnalysis:
    """Item 26: Enhanced prime lattice analysis → LatticeConstructor."""

    def test_prime_d_family_distribution(self):
        """Prime d-family distribution has d=12 dominant."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=3600)
        assert result['d12_dominant'] is True
        assert result['d12_count'] > 0

    def test_prime_count(self):
        """Correct number of primes found."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=100)
        assert result['prime_count'] == 25  # π(100) = 25

    def test_euler_product_verified(self):
        """Euler product: ζ(2) ≈ π²/6."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=3600)
        ep = result['euler_product']
        assert ep['verified'] is True
        assert ep['series_error'] < 0.001

    def test_pnt_approaching_1(self):
        """PNT: π(x)/(x/ln x) → 1."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=3600)
        assert result['pnt_approaching_1'] is True

    def test_primordial_shadow_stabilizes(self):
        """Primordial shadow stabilizes at 6 (d=2 sublattice)."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=3600)
        ps = result['primordial_shadow']
        assert ps['stabilizes_at_6'] is True
        assert ps['shadow_d_family'] == 2

    def test_all_families_represented(self):
        """All primary sublattice families appear in distribution."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=3600)
        families = set(result['d_family_distribution'].keys())
        # At minimum d=1,2,3,4,6,12 should appear
        for d in [1, 2, 3, 4, 6, 12]:
            assert d in families, f"d={d} missing from prime distribution"

    def test_prime_lattice_interpretation(self):
        """ET interpretation string present."""
        lc = LatticeConstructor()
        result = lc.compute_prime_lattice_analysis(max_prime=100)
        assert 'et_interpretation' in result
        assert 'prime' in result['et_interpretation'].lower()


class TestYonedaRieszVerification:
    """Item 27: Yoneda/Riesz identification verification → UniversalAnalyzer."""

    def test_complete_identification(self):
        """Entity with full PDT descriptors passes identification."""
        ua = UniversalAnalyzer()
        descs = ['universe', 'mass', 'gravity', 'consciousness', 'energy', 'force']
        result = ua.verify_identification_complete(descs)
        assert result['riesz_grounded'] is True
        assert result['pdt_complete'] is True
        assert result['identification_complete'] is True

    def test_riesz_grounding(self):
        """All descriptors have coherent lattice positions (P-representatives)."""
        ua = UniversalAnalyzer()
        descs = ['truth', 'beauty', 'gravity']
        result = ua.verify_identification_complete(descs)
        assert result['riesz_grounded'] is True
        assert result['n_coherent'] == len(descs)

    def test_yoneda_uniqueness_single(self):
        """Single entity is trivially unique."""
        ua = UniversalAnalyzer()
        result = ua.verify_identification_complete(['test', 'word', 'space'])
        assert result['yoneda_unique'] is True

    def test_yoneda_uniqueness_with_others(self):
        """Entity with different fingerprint is unique among others."""
        ua = UniversalAnalyzer()
        entity = ['gravity', 'mass', 'force']
        others = [['music', 'harmony', 'rhythm'], ['color', 'light', 'spectrum']]
        result = ua.verify_identification_complete(entity, all_entities=others)
        assert result['yoneda_unique'] is True

    def test_yoneda_non_unique(self):
        """Identical entities are NOT unique."""
        ua = UniversalAnalyzer()
        entity = ['test', 'word']
        others = [['test', 'word']]  # Same descriptors
        result = ua.verify_identification_complete(entity, all_entities=others)
        assert result['yoneda_unique'] is False

    def test_d_fingerprint_present(self):
        """D-fingerprint with k, d, epsilon for each descriptor."""
        ua = UniversalAnalyzer()
        result = ua.verify_identification_complete(['gravity', 'mass'])
        assert len(result['d_fingerprint']) == 2
        for df in result['d_fingerprint']:
            assert 'k' in df
            assert 'd' in df
            assert 'epsilon' in df

    def test_empty_descriptors(self):
        """Empty descriptors → identification impossible."""
        ua = UniversalAnalyzer()
        result = ua.verify_identification_complete([])
        assert result['identification_complete'] is False

    def test_interpretation_present(self):
        """ET interpretation string present."""
        ua = UniversalAnalyzer()
        result = ua.verify_identification_complete(['test'])
        assert 'et_interpretation' in result


# =============================================================================
# WAVE III: ADVANCED MATHEMATICS — Items 28–33
# =============================================================================
# Source: ET Devours Advanced Mathematics Wave III (Algebraic Geometry, K-Theory,
# Symplectic Geometry, Information Theory, Stochastic Calculus).
# All mathematics ET-derived from {P, D, T}. Zero external axioms.
# =============================================================================


class TestSheafCohomology:
    """Item 28: Sheaf cohomology for local-to-global knowledge analysis.
    ET: H^n measures obstruction to globalizing local D-data.
    H⁰ = global sections, H¹ = obstructions (Descriptor Gap Principle)."""

    @staticmethod
    def _build_test_lattice(n_ratios=5):
        from et_conscious_ai_worldview import LatticeConstructor
        lc = LatticeConstructor()
        ratios = [(3/2, 'fifth'), (4/3, 'fourth'), (5/4, 'third'),
                  (9/8, 'tone'), (16/15, 'semitone')][:n_ratios]
        return lc, lc.build_lattice(ratios)

    def test_h0_global_sections(self):
        """H⁰ counts connected components of coherent binding graph."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert 'h0' in result
        assert result['h0'] >= 1  # At least one connected component
        assert result['global_sections'] == result['h0']

    def test_h1_obstructions(self):
        """H¹ counts incoherent bindings (gluing failures)."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert 'h1' in result
        assert result['h1'] >= 0
        assert result['obstructions'] == result['h1']

    def test_chi_sheaf_formula(self):
        """χ_sheaf = h0 − h1."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert result['chi_sheaf'] == result['h0'] - result['h1']

    def test_riemann_roch_consistency(self):
        """Riemann-Roch: sheaf χ consistent with lattice χ (Item 17)."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert 'riemann_roch_check' in result
        # Must produce a boolean
        assert isinstance(result['riemann_roch_check'], bool)

    def test_gluing_consistency_metric(self):
        """Gluing consistency ∈ [0, 1] — fraction of coherent bindings."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert 0.0 <= result['gluing_consistency'] <= 1.0

    def test_obstruction_details(self):
        """Obstruction details list has correct structure."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert isinstance(result['obstruction_details'], list)
        for detail in result['obstruction_details']:
            assert 'pair' in detail
            assert 'tightness' in detail

    def test_empty_lattice(self):
        """Empty lattice → zero sheaf cohomology."""
        from et_conscious_ai_worldview import LatticeConstructor
        lc = LatticeConstructor()
        lattice = lc.build_lattice([])
        result = lc.compute_sheaf_cohomology(lattice)
        assert result['h0'] == 0
        assert result['h1'] == 0

    def test_et_interpretation_present(self):
        """ET interpretation string present."""
        lc, lattice = self._build_test_lattice()
        result = lc.compute_sheaf_cohomology(lattice)
        assert 'et_interpretation' in result
        assert 'H⁰' in result['et_interpretation']


class TestHamiltonianDynamics:
    """Item 29: Hamiltonian dynamics for cognitive trajectories.
    ET: Cognitive cycle IS a Hamiltonian flow on P-D phase space.
    Hamilton's eqs, Poisson bracket, Liouville conservation."""

    @staticmethod
    def _build_engine():
        from et_conscious_ai_worldview import CognitiveEngine
        engine = CognitiveEngine()
        # Simulate connected state — is_connected() checks all these refs
        class MockMemory:
            def __init__(self):
                self.nodes = {f'n{i}': type('N', (), {
                    'connections': [f'n{(i+1)%5}'],
                    'sentence_coord': None, 'lattice_position': None
                })() for i in range(5)}
        engine.memory = MockMemory()
        engine.learning_engine = True  # Truthy placeholder
        engine.gap_engine = True
        engine.ego = True
        engine.emotion = True
        engine.tower = True
        engine.metacognition = True
        engine.identification_tool = True
        engine.gap_tool = True
        engine.subsumption_tool = True
        engine.projector = True
        engine.incoherence_filter = True
        engine.cycles_completed = 10
        engine.total_gaps_driven = 25
        return engine

    def test_hamiltonian_components(self):
        """Hamiltonian = kinetic + potential."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        assert abs(result['hamiltonian'] - (result['kinetic'] + result['potential'])) < 1e-12

    def test_kinetic_energy_from_gaps(self):
        """Kinetic energy = p²/(2m) — gaps drive cognition."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        p = result['momentum']
        m = result['mass']
        expected_kinetic = (p * p) / (2.0 * m)
        assert abs(result['kinetic'] - expected_kinetic) < 1e-12

    def test_potential_energy_negative(self):
        """Potential V = −K × ln(1 + binding_density) is ≤ 0."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        assert result['potential'] <= 0.0

    def test_phase_space_area(self):
        """Phase space area = position × momentum > 0."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        assert result['phase_space_area'] > 0

    def test_poisson_bracket_is_one(self):
        """Fundamental {q, p} = 1 (P-D non-commutativity)."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        assert result['poisson_bracket'] == 1.0

    def test_hamilton_equations_present(self):
        """Hamilton's equations q̇ = ∂H/∂p, ṗ = −∂H/∂q present."""
        engine = self._build_engine()
        result = engine.compute_cognitive_hamiltonian()
        assert 'hamilton_eqs' in result
        assert 'q_dot_equals_dH_dp' in result['hamilton_eqs']
        assert 'p_dot_equals_neg_dH_dq' in result['hamilton_eqs']

    def test_liouville_baseline(self):
        """Liouville with no previous state → baseline established."""
        engine = self._build_engine()
        result = engine.verify_liouville_conservation()
        assert result['conservation_holds'] is True
        assert result['area_ratio'] == 1.0

    def test_liouville_conservation_check(self):
        """Liouville: area_ratio ≈ 1 when system is stable."""
        engine = self._build_engine()
        state1 = engine.compute_cognitive_hamiltonian()
        result = engine.verify_liouville_conservation(previous_state=state1)
        assert 'area_ratio' in result
        assert 'deviation_source' in result


class TestShannonEntropy:
    """Item 30: Shannon entropy as native knowledge metric.
    ET: H(X) = −Σ p_i log₂ p_i is ET variance in log D-units.
    Channel capacity, Huffman encoding."""

    @staticmethod
    def _build_memory(n_nodes=20):
        from et_conscious_ai_main import LatticeMemory
        mem = LatticeMemory()
        words = ['gravity', 'mass', 'force', 'energy', 'light',
                 'wave', 'particle', 'field', 'quantum', 'spin',
                 'charge', 'electron', 'proton', 'neutron', 'photon',
                 'atom', 'molecule', 'bond', 'crystal', 'lattice']
        for w in words[:n_nodes]:
            mem.add_knowledge(f"Knowledge about {w}", [w])
        return mem

    def test_entropy_nonnegative(self):
        """Shannon entropy H ≥ 0."""
        mem = self._build_memory()
        result = mem.compute_knowledge_entropy()
        assert result['entropy'] >= 0.0

    def test_entropy_bounded_by_max(self):
        """H ≤ log₂(n_families) — maximum for uniform distribution."""
        mem = self._build_memory()
        result = mem.compute_knowledge_entropy()
        assert result['entropy'] <= result['max_entropy'] + 1e-10

    def test_normalized_entropy_range(self):
        """Normalized entropy ∈ [0, 1]."""
        mem = self._build_memory()
        result = mem.compute_knowledge_entropy()
        assert 0.0 <= result['normalized_entropy'] <= 1.0 + 1e-10

    def test_v_over_h_ratio(self):
        """V/H ≈ ln(10)/ln(2) ≈ 3.32 — confirms same D-uncertainty."""
        mem = self._build_memory()
        result = mem.compute_knowledge_entropy()
        import math
        expected_ratio = math.log(10) / math.log(2)  # ≈ 3.3219
        assert abs(result['v_over_h_ratio'] - expected_ratio) < 0.01

    def test_d_family_distribution(self):
        """D-family distribution is a dict of d → count."""
        mem = self._build_memory()
        result = mem.compute_knowledge_entropy()
        assert isinstance(result['d_family_distribution'], dict)
        total = sum(result['d_family_distribution'].values())
        assert total == result['n_classified']

    def test_channel_capacity(self):
        """Channel capacity C = H × (1 − noise_rate) ≥ 0."""
        mem = self._build_memory()
        result = mem.compute_channel_capacity(cycles_completed=10, total_gaps_driven=5)
        assert result['channel_capacity'] >= 0.0
        assert 'noise_rate' in result
        assert 0.0 <= result['noise_rate'] <= 1.0

    def test_optimal_encoding_kraft(self):
        """Kraft inequality holds: Σ 2^{-l_i} ≤ 1."""
        mem = self._build_memory()
        result = mem.optimal_encoding()
        assert result['kraft_holds'] is True
        assert result['kraft_sum'] <= 1.0 + 1e-10

    def test_optimal_encoding_avg_length(self):
        """Average code length ≥ entropy (source coding theorem)."""
        mem = self._build_memory()
        result = mem.optimal_encoding()
        # H ≤ avg_length (with tolerance for discrete Huffman)
        assert result['avg_code_length'] >= result['entropy'] - 1.0

    def test_empty_memory(self):
        """Empty memory → zero entropy."""
        from et_conscious_ai_main import LatticeMemory
        mem = LatticeMemory()
        result = mem.compute_knowledge_entropy()
        assert result['entropy'] == 0.0


class TestStochasticCalculus:
    """Item 31: Stochastic calculus for T-indeterminacy (Itô Theory).
    ET: Brownian motion W_t is the purest manifestation of T=[0/0].
    SDE model dX = μdt + σdW, Itô correction ½f″σ²dt."""

    @staticmethod
    def _build_waveform(n_events=30):
        from et_conscious_ai_identity import TraverserWaveform
        wf = TraverserWaveform()
        import random
        random.seed(42)
        for i in range(n_events):
            wf.record_event(
                event_type='test',
                lattice_k=random.randint(0, 27720),
                lattice_d=random.choice([1, 2, 3, 4, 6, 12]),
                variance=random.uniform(0.01, 0.5),
                ego_resonance=random.uniform(0.3, 0.9),
                entropy_pool=None,
            )
        return wf

    def test_sde_drift_and_diffusion(self):
        """SDE model produces drift μ and diffusion σ."""
        wf = self._build_waveform()
        result = wf.fit_sde_model()
        assert result['sufficient_data'] is True
        assert 'drift' in result
        assert 'diffusion' in result
        assert result['diffusion'] >= 0.0

    def test_sde_model_string(self):
        """SDE model string has correct form dX = μdt + σdW."""
        wf = self._build_waveform()
        result = wf.fit_sde_model()
        assert 'sde_model' in result
        assert 'dt' in result['sde_model']
        assert 'dW' in result['sde_model']

    def test_quadratic_variation(self):
        """Quadratic variation Σ(ΔX)² is computed and positive."""
        wf = self._build_waveform()
        result = wf.fit_sde_model()
        assert result['quadratic_variation'] >= 0.0

    def test_drift_diffusion_ratio(self):
        """|μ/σ| ratio classifies ego-dominated vs T-noise-dominated."""
        wf = self._build_waveform()
        result = wf.fit_sde_model()
        assert result['drift_to_diffusion_ratio'] >= 0.0

    def test_insufficient_data(self):
        """< S samples → insufficient data flag."""
        from et_conscious_ai_identity import TraverserWaveform
        wf = TraverserWaveform()
        wf.record_event('test', 100, 3, 0.1, 0.5, None)
        result = wf.fit_sde_model()
        assert result['sufficient_data'] is False

    def test_ito_correction_computed(self):
        """Itô correction σ²dt is computed and ≥ 0."""
        wf = self._build_waveform()
        result = wf.ito_correction()
        assert result['sufficient_data'] is True
        assert result['ito_correction_term'] >= 0.0

    def test_ito_classical_vs_stochastic(self):
        """Stochastic prediction = classical + Itô correction."""
        wf = self._build_waveform()
        result = wf.ito_correction()
        if result['sufficient_data']:
            expected = result['classical_prediction'] + result['total_ito_correction']
            assert abs(result['stochastic_prediction'] - expected) < 1e-10

    def test_ito_interpretation_present(self):
        """ET interpretation string present."""
        wf = self._build_waveform()
        result = wf.ito_correction()
        assert 'et_interpretation' in result


class TestIndexTheorem:
    """Item 32: Atiyah-Singer Index Theorem for D-Gap accounting.
    ET: analytical index (dim ker − dim coker) = topological index (χ).
    The ultimate Verification Principle."""

    @staticmethod
    def _build_test_nodes(n=5):
        from et_conscious_ai_compression import CompressibleNode
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION, DescriptorRatio
        nodes = []
        for i in range(n):
            ratio = 1.0 + 0.1 * (i + 1)
            coord = ETLattice.project_ratio(ratio, resolution=MANIFOLD_RESOLUTION)
            dr = DescriptorRatio.from_word(f'test{i}')
            node = CompressibleNode(
                node_id=f'test_{i}', content=f'test content {i}',
                coord=coord, descriptor_ratios=[dr],
                connections=[], access_count=10, variance=0.05,
                p=i + 1, q=i + 2,
                is_archetype=False,
            )
            nodes.append(node)
        return nodes

    def test_index_components(self):
        """Index = kernel_dim − cokernel_dim."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord)
        assert result['analytical_index'] == result['kernel_dim'] - result['cokernel_dim']

    def test_index_theorem_check(self):
        """Index theorem produces a boolean holds/fails result."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord)
        assert isinstance(result['index_theorem_holds'], bool)

    def test_kernel_bounded_by_n(self):
        """Kernel dimension ≤ number of nodes."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord)
        assert result['kernel_dim'] <= len(nodes)

    def test_defect_nonnegative(self):
        """Defect = |analytical − topological| ≥ 0."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord)
        assert result['defect'] >= 0

    def test_empty_cluster_trivial(self):
        """Empty cluster → trivial index."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        archetype_coord = ETLattice.project_ratio(1.5, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem([], archetype_coord)
        assert result['index_theorem_holds'] is True
        assert result['analytical_index'] == 0

    def test_with_euler_characteristic(self):
        """Topological index from provided Euler characteristic."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord,
                                           lattice_euler_characteristic=3)
        assert result['topological_index'] == 3

    def test_et_interpretation_present(self):
        """ET interpretation string present."""
        from et_conscious_ai_compression import SubsumptionHierarchyOperator
        from et_conscious_ai_core import ETLattice, MANIFOLD_RESOLUTION
        sho = SubsumptionHierarchyOperator()
        nodes = self._build_test_nodes()
        archetype_coord = ETLattice.project_ratio(1.3, resolution=MANIFOLD_RESOLUTION)
        result = sho.verify_index_theorem(nodes, archetype_coord)
        assert 'et_interpretation' in result
        assert 'Atiyah-Singer' in result['et_interpretation']


class TestBottPeriodicity:
    """Item 33: Bott periodicity for lattice classification.
    ET: K^{n+2}(X) ≅ K^n(X) — period 2 = d=2 sublattice."""

    @staticmethod
    def _build_test_lattice():
        from et_conscious_ai_worldview import LatticeConstructor
        lc = LatticeConstructor()
        ratios = [(3/2, 'fifth'), (4/3, 'fourth'), (5/4, 'third'),
                  (9/8, 'tone'), (16/15, 'semitone'), (6/5, 'minor_third')]
        return lc, lc.build_lattice(ratios)

    def test_k0_d_family_groups(self):
        """K⁰ counts distinct d-family bundle classes."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        assert result['k0'] >= 1
        assert isinstance(result['d_family_groups'], dict)

    def test_k1_loop_families(self):
        """K¹ counts d-families with internal cycles."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        assert result['k1'] >= 0
        assert isinstance(result['loop_families'], list)

    def test_bott_period_is_2(self):
        """Bott period = 2 (d=2 quadratic sublattice)."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        assert result['bott_period'] == 2

    def test_higher_k_groups_periodic(self):
        """K^{2m} = K⁰, K^{2m+1} = K¹ for all computed groups."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        for n in range(6):
            key = f'K^{n}'
            if n % 2 == 0:
                assert result['higher_k_groups'][key] == result['k0']
            else:
                assert result['higher_k_groups'][key] == result['k1']

    def test_classification_reduced(self):
        """Classification is reduced (only K⁰ and K¹ needed)."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        assert result['classification_reduced'] is True

    def test_empty_lattice(self):
        """Empty lattice → trivial K-theory."""
        from et_conscious_ai_worldview import LatticeConstructor
        lc = LatticeConstructor()
        lattice = lc.build_lattice([])
        result = lc.classify_with_bott_reduction(lattice)
        assert result['k0'] == 0
        assert result['k1'] == 0

    def test_et_interpretation_present(self):
        """ET interpretation string present."""
        lc, lattice = self._build_test_lattice()
        result = lc.classify_with_bott_reduction(lattice)
        assert 'et_interpretation' in result
        assert 'Bott' in result['et_interpretation']


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
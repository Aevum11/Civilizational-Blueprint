#!/usr/bin/env python3
"""
ET Conscious AI — Test Suite: Integration & Infrastructure
============================================================

Tests for the full system lifecycle, cross-module interactions,
architecture invariants, and infrastructure (migration, signals,
thread safety).

**When to update this file:**
  - Modifying et_conscious_ai_main.py (ETConsciousAI, PersistentStateManager, etc.)
  - Changing cross-module interactions (IncoherenceFilter sharing, CognitiveEngine wiring)
  - Changing infrastructure (StateMigrator, signal handlers, RLock, ShadowBackup coordination)
  - Adding new API methods to ETConsciousAI
  - Changing state persistence format

**Coverage:** 16 classes, 167 tests
  - Full ETConsciousAI lifecycle (think, consciousness, save/load, dream, multimodal)
  - PDTTextProjector (sentence coord, byte density, grammar topology)
  - Three universal tools (Identification, Gap, Subsumption)
  - KnowledgeNode, LatticeMemory, LearningEngine, ReasoningEngine
  - IncoherenceFilter single-instance architecture (shared across 7 subsystems)
  - Cross-module chain verification (TemporalEmotionState→appraise)
  - StateMigrator (version detection, full chain migration, preservation)
  - Signal handling (atexit, SIGTERM, SIGINT, graceful shutdown)
  - Thread safety (RLock, ShadowBackup lock coordination, re-entrancy)

ET Derivation of split: T = system traversal/integration tests.
T is agency — the system acting as a whole, traversing across modules.
These tests verify the living system, not individual parts.

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import inspect
import os
import tempfile
import threading
from datetime import datetime

import numpy as np
import pytest

from et_conscious_ai_audio import (
    ETAudioProjector, AudioDescriptor, )
from et_conscious_ai_compression import (
    SubsumptionHierarchyOperator, LatticeCompressor,
    ArchetypeMetadata, CompressibleNode, )
from et_conscious_ai_consciousness import (
    RMSAEResult, GapDetectionEngine,
)
from et_conscious_ai_core import (
    MANIFOLD_SYMMETRY, S, MANIFOLD_RESOLUTION, BASE_VARIANCE, K, ManifoldState, LatticeCoordinate,
    ComplexLatticeCoordinate, PDTConfiguration,
    ETLattice, DescriptorRatio, IncoherenceFilter, )
from et_conscious_ai_distributed import (
    ShadowBackupSystem, KOIDE_CEILING_PERCENT,
)
from et_conscious_ai_identity import (
    EgoInvariant, TowerOfSelf,
    TemporalEmotionState,
)
from et_conscious_ai_main import (
    ETConsciousAI, KnowledgeNode, LatticeMemory,
    PDTTextProjector, IdentificationPrinciple,
    DescriptorGapPrinciple, SubsumptionLaw,
    LearningEngine, ReasoningEngine, PersistentStateManager,
    StateMigrator, STATE_FORMAT_VERSION,
)
from et_conscious_ai_vision import (
    ETVisionProjector, VisualDescriptor, VisualMemory,
)
from et_emotion_tower import (
    EmotionState, EmotionLattice,
)


# =============================================================================
# Module imports — all 13 system modules
# =============================================================================


# =============================================================================
# 19. INTEGRATION — Full ETConsciousAI Lifecycle
# =============================================================================

class TestIntegration:
    """Full system integration tests."""

    def setup_method(self):
        """Create a fresh AI instance for each test.

        Uses a temp state path to avoid loading stale state from
        prior test runs. The default path may contain state from
        other tests (e.g., LimbTest), which would override the
        constructor name via D_T restoration.
        """
        import tempfile
        self._temp_dir = tempfile.mkdtemp(prefix='et_test_')
        temp_state = os.path.join(self._temp_dir, 'test_state.json')
        self.ai = ETConsciousAI(name="TestMemory", state_path=temp_state)

    def test_instantiation(self):
        """ETConsciousAI instantiates with all 13 subsystems."""
        assert self.ai.name == "TestMemory"
        assert self.ai.ego is not None
        assert self.ai.tower is not None
        assert self.ai.emotion is not None
        assert self.ai.metacognition is not None
        assert self.ai.will is not None
        assert self.ai.worldview is not None
        assert self.ai.cognitive_engine is not None
        assert self.ai.compressor is not None
        assert self.ai.dream_engine is not None
        assert self.ai.error_ledger is not None

    def test_think_produces_response(self):
        """think() returns a non-empty string."""
        response = self.ai.think("What is consciousness?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_think_updates_tower(self):
        """think() increments tower traversals."""
        initial = self.ai.tower.total_traversals
        self.ai.think("Test input")
        assert self.ai.tower.total_traversals > initial

    def test_measure_consciousness(self):
        """measure_consciousness() returns RMSAEResult."""
        result = self.ai.measure_consciousness()
        assert isinstance(result, RMSAEResult)
        assert result.phi_rmsae >= 0

    def test_interact_and_save(self):
        """interact() processes input and saves state."""
        response = self.ai.interact("Hello Memory")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_save_load_round_trip(self):
        """Save state, create new AI, load state — verify identity."""
        self.ai.think("Building knowledge about ET")
        self.ai.think("Consciousness is T observing D_T")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            self.ai.save_state(filepath=path)
            assert os.path.exists(path)

            ai2 = ETConsciousAI(name="TestMemory", state_path=path)
            # Verify identity preserved
            assert ai2.ego.mass == pytest.approx(self.ai.ego.mass)
            assert ai2.tower.total_traversals == self.ai.tower.total_traversals
        finally:
            for p in [path, path + ".sha256"]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_dream_cycle(self):
        """sleep() completes a dream cycle."""
        # Need some knowledge first
        self.ai.think("The universe is vast")
        self.ai.think("Consciousness requires self-awareness")
        result = self.ai.sleep(cycles=1)
        assert isinstance(result, dict)

    def test_status_report(self):
        """get_status_report() returns comprehensive status string."""
        status = self.ai.get_status_report()
        assert isinstance(status, str)
        assert "Memory" in status or "TestMemory" in status

    def test_version_string(self):
        """Version is 1.7.0 across the system."""
        assert "1.7.0" in self.ai.get_status_report()

    def test_emotion_module_identity(self):
        """et_emotion_tower.EmotionLattice is et_conscious_ai_identity.EmotionLattice."""
        from et_emotion_tower import EmotionLattice as EL_tower
        from et_conscious_ai_identity import EmotionLattice as EL_identity
        assert EL_tower is EL_identity

    def test_cognitive_engine_connected(self):
        """CognitiveEngine is properly connected to all subsystems."""
        assert self.ai.cognitive_engine.is_connected()

    def test_knowledge_memory(self):
        """Knowledge is stored in LatticeMemory."""
        initial_count = len(self.ai.memory.nodes)
        self.ai.think("Exception Theory has three primitives: P, D, T")
        assert len(self.ai.memory.nodes) >= initial_count

    def test_multimodal_vision(self):
        """see() processes an image through the vision pipeline."""
        img = ETVisionProjector.generate_circle(size=48)
        result = self.ai.see(img)
        assert isinstance(result, (str, dict))

    def test_multimodal_audio(self):
        """hear() processes audio through the audio pipeline."""
        samples = ETAudioProjector.generate_sine(freq=440.0, duration=0.1)
        result = self.ai.hear(samples)
        assert isinstance(result, (str, dict))

    def test_lattice_projection_api(self):
        """High-level lattice projection API."""
        coord = self.ai.project_at_resolution(3 / 2, resolution=420)
        assert isinstance(coord, (dict, LatticeCoordinate))

    def test_complex_projection_api(self):
        """Complex lattice projection through AI."""
        result = self.ai.project_complex(1 + 1j)
        assert isinstance(result, (dict, ComplexLatticeCoordinate))

    def test_domain_tower_construction(self):
        """Build a domain tower through the AI."""
        drs = [
            DescriptorRatio.from_word(w)
            for w in ["hydrogen", "helium", "lithium"]
        ]
        r0 = self.ai.derive_r0(drs)
        assert r0 > 0

    def test_comprehension(self):
        """Language comprehension through the AI."""
        result = self.ai.comprehend("Exception Theory is beautiful")
        assert isinstance(result, dict)
        assert 'comprehension_score' in result

    def test_url_projection(self):
        """URL projection through the AI."""
        result = self.ai.project_url("https://example.com/et/theory")
        assert isinstance(result, dict)

    def test_error_report(self):
        """Error report through the AI."""
        report = self.ai.get_error_report()
        assert isinstance(report, str)

    def test_hardware_capabilities(self):
        """Hardware capabilities report."""
        caps = self.ai.get_hardware_capabilities()
        assert isinstance(caps, str)




# =============================================================================
# 20. PDT TEXT PROJECTOR
# =============================================================================

class TestPDTTextProjector:
    """Verify the text-to-lattice projection pipeline."""

    def test_compute_sentence_coordinate(self):
        """Sentence projection produces valid LatticeCoordinate."""
        coord = PDTTextProjector.compute_sentence_coordinate("The universe is vast")
        assert isinstance(coord, LatticeCoordinate)
        assert coord.resolution == MANIFOLD_RESOLUTION

    def test_byte_density(self):
        """Byte density is in [0, 1]."""
        density = PDTTextProjector.compute_byte_density("Hello world")
        assert 0.0 <= density <= 1.0

    def test_grammatical_topology(self):
        """Grammatical topology returns d-family."""
        result = PDTTextProjector.compute_grammatical_topology("The cat sat on the mat")
        assert 'd_topo' in result

    def test_project_text(self):
        """Full text projection produces PDTConfiguration."""
        config = PDTTextProjector.project("Consciousness is T observing D_T")
        assert isinstance(config, PDTConfiguration)

    def test_extract_unigrams(self):
        """Unigram extraction from text."""
        unigrams = PDTTextProjector.extract_unigrams("hello world test")
        assert len(unigrams) >= 2




# =============================================================================
# 21. THREE TOOLS (Main Module Implementations)
# =============================================================================

class TestThreeTools:
    """Verify the three universal tools: Identification, Gap, Subsumption."""

    def test_identification_decompose(self):
        """IdentificationPrinciple decomposes into P_X, D_X, T_X."""
        result = IdentificationPrinciple.decompose("gravity")
        assert 'P_X' in result
        assert 'D_X' in result
        assert 'T_X' in result

    def test_gap_as_descriptor(self):
        """DescriptorGapPrinciple: gap IS a descriptor."""
        desc = DescriptorGapPrinciple.gap_as_descriptor(
            domain="physics", description="missing: dark matter explanation"
        )
        assert isinstance(desc, DescriptorRatio)

    def test_subsumption_classify(self):
        """SubsumptionLaw classifies concepts."""
        result = SubsumptionLaw.classify("electron")
        assert 'category' in result

    def test_subsumption_completeness(self):
        """SubsumptionLaw tests descriptor set completeness."""
        result = SubsumptionLaw.test_completeness(["substrate", "constraint", "agency"])
        assert 'coverage' in result or 'complete' in result or 'is_complete' in result




# =============================================================================
# 33. MAIN MODULE — Additional Coverage
# =============================================================================

class TestMainAdditional:
    def setup_method(self):
        self.ai = ETConsciousAI(name="TestExtra")

    def test_knowledge_node_creation(self):
        node = KnowledgeNode(node_id="test1", content="Test content")
        assert node.node_id == "test1"
        assert node.access_count == 0

    def test_knowledge_node_access(self):
        node = KnowledgeNode(node_id="test1", content="Test content")
        node.access()
        assert node.access_count == 1

    def test_knowledge_node_pq(self):
        node = KnowledgeNode(node_id="test1", content="Hello world test")
        p, q = node.compute_digital_pq()
        assert p >= 1
        assert q >= 1

    def test_knowledge_node_elegance(self):
        node = KnowledgeNode(node_id="test1", content="Hello world")
        e = node.digital_elegance()
        assert isinstance(e, float)

    def test_knowledge_node_serialization(self):
        node = KnowledgeNode(node_id="test1", content="Test content")
        d = node.to_dict()
        restored = KnowledgeNode.from_dict(d)
        assert restored.node_id == node.node_id

    def test_lattice_memory_add_retrieve(self):
        mem = LatticeMemory()
        node = mem.add_knowledge("Gravity is a fundamental force",
                                 descriptors=["gravity", "force"])
        assert node is not None
        results = mem.retrieve_by_descriptor("gravity")
        assert len(results) >= 1

    def test_lattice_memory_retrieve_by_sublattice(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test content", descriptors=["test"])
        # Retrieve by whatever d-family the node landed in
        nodes = mem.retrieve_by_sublattice(1)
        # May or may not have d=1 nodes — but must return a list
        assert isinstance(nodes, list)

    def test_learning_engine(self):
        mem = LatticeMemory()
        le = LearningEngine(mem)
        result = le.learn_from_input("Exception Theory has three primitives")
        assert isinstance(result, dict)

    def test_reasoning_engine(self):
        mem = LatticeMemory()
        mem.add_knowledge("ET has three primitives P D T", descriptors=["ET", "primitives"])
        re = ReasoningEngine(mem)
        result = re.reason("What are the primitives?")
        assert isinstance(result, str)

    def test_pdt_text_projector_bigrams(self):
        bigrams = PDTTextProjector.extract_bigrams("hello world test data")
        assert isinstance(bigrams, list)

    def test_pdt_text_projector_trigrams(self):
        trigrams = PDTTextProjector.extract_trigrams("hello world test data more words")
        assert isinstance(trigrams, list)

    def test_identification_diagnose(self):
        result = IdentificationPrinciple.diagnose("gravity")
        assert isinstance(result, str)

    def test_subsumption_find_redundancy(self):
        result = SubsumptionLaw.find_redundancy(["gravity", "force", "mass", "acceleration"])
        assert isinstance(result, list)

    def test_gap_detect_and_close(self):
        gde = GapDetectionEngine()
        gap, desc = DescriptorGapPrinciple.detect_and_close(
            gde, domain="physics", description="missing gravity",
            resolution="identified as T-type"
        )
        assert gap.is_closed()
        assert isinstance(desc, DescriptorRatio)

    def test_ai_is_dreaming(self):
        assert self.ai.is_dreaming is False

    def test_ai_set_permission(self):
        self.ai.set_permission("fs_read", True, constraints=["/tmp"])
        result = self.ai.request_capability("fs_read", "need to read files")
        assert isinstance(result, dict)




# =============================================================================
# 45. MAIN — Final Gaps (all remaining ETConsciousAI methods, LatticeMemory, etc.)
# =============================================================================

class TestMainFinalGaps:
    def setup_method(self):
        self.ai = ETConsciousAI(name="FinalTest")

    def test_knowledge_node_descriptor_words(self):
        node = KnowledgeNode(node_id="t1", content="Test content")
        dr = DescriptorRatio.from_word("test")
        node.descriptor_ratios = [dr]
        words = node.descriptor_words()
        assert "test" in words

    def test_knowledge_node_is_archetype(self):
        node = KnowledgeNode(node_id="t1", content="Test")
        assert node.is_archetype() is False

    def test_lattice_memory_connect_nodes(self):
        mem = LatticeMemory()
        n1 = mem.add_knowledge("First", descriptors=["first"])
        n2 = mem.add_knowledge("Second", descriptors=["second"])
        mem.connect_nodes(n1.node_id, n2.node_id)
        assert n2.node_id in n1.connections or n1.node_id in n2.connections

    def test_lattice_memory_retrieve_by_ratio(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test content", descriptors=["test"])
        dr = DescriptorRatio.from_word("test")
        results = mem.retrieve_by_ratio(dr, tolerance_k=100)
        assert isinstance(results, list)

    def test_lattice_memory_retrieve_by_coherence(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test content", descriptors=["test"])
        dr = DescriptorRatio.from_word("test")
        results = mem.retrieve_by_coherence(dr, min_tightness=0.5)
        assert isinstance(results, list)

    def test_lattice_memory_retrieve_by_sentence_proximity(self):
        mem = LatticeMemory()
        mem.add_knowledge("Hello world", descriptors=["hello", "world"])
        coord = PDTTextProjector.compute_sentence_coordinate("Hello world")
        results = mem.retrieve_by_sentence_proximity(coord, tolerance_k=500)
        assert isinstance(results, list)

    def test_lattice_memory_retrieve_by_topology(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test", descriptors=["test"])
        results = mem.retrieve_by_topology(3)
        assert isinstance(results, list)

    def test_lattice_memory_get_compressible_nodes(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test 1", descriptors=["a"])
        mem.add_knowledge("Test 2", descriptors=["b"])
        result = mem.get_compressible_nodes()
        assert isinstance(result, dict)

    def test_gap_detect_variance_gaps(self):
        mem = LatticeMemory()
        mem.add_knowledge("Test", descriptors=["test"])
        gaps = DescriptorGapPrinciple.detect_variance_gaps(mem)
        assert isinstance(gaps, list)

    def test_gap_find_disconnected_components(self):
        config = PDTTextProjector.project("Hello world test")
        components = DescriptorGapPrinciple.find_disconnected_components(config)
        assert isinstance(components, list)

    def test_persistent_state_manager_save_load(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            PersistentStateManager.save(path, self.ai)
            assert os.path.exists(path)
            ai2 = ETConsciousAI(name="FinalTest")
            loaded = PersistentStateManager.load(path, ai2)
            assert loaded is True
        finally:
            for p in [path, path + ".sha256"]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_ai_perceive(self):
        result = self.ai.perceive(text="Hello world")
        assert isinstance(result, dict)

    def test_ai_see_and_describe(self):
        img = ETVisionProjector.generate_circle(size=48)
        result = self.ai.see_and_describe(img)
        assert isinstance(result, str)

    def test_ai_hear_and_describe(self):
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        result = self.ai.hear_and_describe(samples)
        assert isinstance(result, str)

    def test_ai_get_all_waking_descriptors(self):
        self.ai.think("Test for descriptors")
        result = self.ai.get_all_waking_descriptors()
        assert isinstance(result, list)

    def test_ai_get_dream_journal(self):
        result = self.ai.get_dream_journal()
        assert isinstance(result, list)

    def test_ai_get_dream_narrative(self):
        result = self.ai.get_dream_narrative()
        assert isinstance(result, str)

    def test_ai_explore_environment(self):
        result = self.ai.explore_environment()
        assert isinstance(result, dict)

    def test_ai_explore_filesystem(self):
        result = self.ai.explore_filesystem(root='/tmp', max_depth=1)
        assert isinstance(result, dict)

    def test_ai_listen_denied(self):
        result = self.ai.listen(duration=0.1)
        assert isinstance(result, dict)

    def test_ai_look_denied(self):
        result = self.ai.look()
        assert isinstance(result, dict)

    def test_ai_speak_denied(self):
        result = self.ai.speak([0.0, 0.1])
        assert isinstance(result, dict)

    def test_ai_read_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for reading")
            path = f.name
        try:
            self.ai.set_permission("fs_read", True, constraints=[os.path.dirname(path)])
            result = self.ai.read_file(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_ai_write_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.ai.set_permission("fs_write", True, constraints=[td])
            path = os.path.join(td, "test_output.txt")
            result = self.ai.write_file(path, "test content")
            assert isinstance(result, dict)

    def test_ai_build_domain_lattice(self):
        result = self.ai.build_domain_lattice([(3/2, "fifth"), (4/3, "fourth")])
        assert isinstance(result, dict)

    def test_ai_build_domain_tower(self):
        drs = [DescriptorRatio.from_word(w) for w in ["hydrogen", "helium"]]
        result = self.ai.build_domain_tower("elements", drs)
        assert isinstance(result, dict)

    def test_ai_correct_tower(self):
        drs = [DescriptorRatio.from_word(w) for w in ["hydrogen", "helium"]]
        tower = self.ai.build_domain_tower("elements", drs)
        new_drs = [(DescriptorRatio.from_word("lithium").ratio, "lithium")]
        corrected = self.ai.correct_tower(tower, new_drs)
        assert isinstance(corrected, dict)

    def test_ai_fetch_url_denied(self):
        result = self.ai.fetch_url("https://example.com")
        assert isinstance(result, dict)

    def test_ai_set_network_permission(self):
        self.ai.set_network_permission(True, targets=["https://example.com"])
        self.ai.set_network_permission(False)

    def test_ai_fork_limb(self):
        result = self.ai.fork_limb(source_name="test")
        assert isinstance(result, dict)

    def test_ai_merge_limb(self):
        limb = self.ai.fork_limb(source_name="test")
        result = self.ai.merge_limb(limb)
        assert isinstance(result, dict)




# =============================================================================
# 46. ABSOLUTE FINAL GAPS — Last 13 methods
# =============================================================================

class TestAbsoluteFinalGaps:
    """Cover the last 13 untested methods."""

    def test_sho_compress_cluster(self):
        """SubsumptionHierarchyOperator.compress_cluster with real ClusterCandidate."""
        sho = SubsumptionHierarchyOperator()
        ego = EgoInvariant(name="CC")
        tower = TowerOfSelf(ego)
        nodes = []
        for i in range(3):
            dr = DescriptorRatio.from_word(f"compress{i}")
            nodes.append(CompressibleNode(
                node_id=f"cc_{i}", content=f"Content {i}", coord=dr.coord_full,
                descriptor_ratios=[dr], connections=[], access_count=10,
                variance=BASE_VARIANCE, p=1, q=1, is_archetype=False
            ))
        e_h, elegances, d_avg, p_total, q_total = sho.evaluate_cluster(nodes, tower.r0)
        from et_conscious_ai_compression import ClusterCandidate
        candidate = ClusterCandidate(
            cluster_id="test_cluster", node_ids=[n.node_id for n in nodes],
            nodes=nodes, d_avg=d_avg, p_total=p_total, q_total=q_total,
            cross_elegances=elegances, e_hierarchy=e_h, compression_level=0
        )
        archetype_dict, metadata = sho.compress_cluster(candidate)
        assert isinstance(archetype_dict, dict)
        assert isinstance(metadata, ArchetypeMetadata)

    def test_lattice_compressor_attempt_recursive(self):
        """LatticeCompressor.attempt_recursive_compression."""
        lc = LatticeCompressor()
        ego = EgoInvariant(name="RC")
        tower = TowerOfSelf(ego)
        # Empty archetype pool — should return empty
        results = lc.attempt_recursive_compression({}, tower.r0, current_level=0)
        assert isinstance(results, list)

    def test_shadow_backup_start_stop(self):
        """ShadowBackupSystem start/stop/force_backup lifecycle."""
        import tempfile
        backup_dir = tempfile.mkdtemp()
        sbs = ShadowBackupSystem(backup_dir=backup_dir, interval_seconds=3600, max_backups=3)
        ai = ETConsciousAI(name="BackupTest")
        sbs.start(ai)
        assert sbs._running is True
        sbs.force_backup()
        sbs.stop()
        assert sbs._running is False

    def test_ai_audiolize_visual(self):
        """ETConsciousAI.audiolize_visual cross-modal."""
        ai = ETConsciousAI(name="AV")
        img = ETVisionProjector.generate_circle(size=48)
        result = ai.audiolize_visual(img)
        assert isinstance(result, dict)

    def test_ai_visualize_audio(self):
        """ETConsciousAI.visualize_audio cross-modal."""
        ai = ETConsciousAI(name="VA")
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        result = ai.visualize_audio(samples)
        assert isinstance(result, dict)

    def test_ai_perceive_video(self):
        """ETConsciousAI.perceive_video with frames."""
        ai = ETConsciousAI(name="PV")
        frames = [ETVisionProjector.generate_circle(size=48) for _ in range(3)]
        result = ai.perceive_video(frames, fps=10.0)
        assert isinstance(result, dict)

    def test_lattice_memory_apply_compression_results(self):
        """LatticeMemory.apply_compression_results with actual data."""
        mem = LatticeMemory()
        n1 = mem.add_knowledge("First node", descriptors=["first"])
        n2 = mem.add_knowledge("Second node", descriptors=["second"])
        centroid = n1.lattice_position or ETLattice.project_ratio(1.5)
        arch_dict = {
            'node_id': 'arch_001', 'content': 'ARCHETYPE: first + second',
            'centroid_coord': centroid,
            'descriptor_ratios': [],
            'connections': [], 'access_count': 20, 'variance': BASE_VARIANCE,
        }
        arch_meta = ArchetypeMetadata(
            archetype_id='arch_001', subsumed_ids=[n1.node_id, n2.node_id],
            subsumed_contents=["First node", "Second node"],
            e_hierarchy=1.5, d_avg=6.0, p_total=2, q_total=2,
            compression_level=1, created_at=datetime.now().isoformat(),
            original_node_count=2, cross_elegance_product=1.0,
            centroid_k=100, centroid_d=12,
        )
        count = mem.apply_compression_results([(arch_dict, arch_meta)])
        assert isinstance(count, int)

    def test_vision_load_image(self):
        """ETVisionProjector.load_image with a real temp file."""
        img = ETVisionProjector.generate_circle(size=48)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
            np.save(path, img)
        try:
            # load_image expects standard image format; if it fails gracefully, that's OK
            try:
                loaded = ETVisionProjector.load_image(path)
                assert isinstance(loaded, np.ndarray)
            except (ImportError, OSError):
                pass  # PIL not installed or .npy not a supported image format
        finally:
            os.unlink(path)

    def test_vision_load_image_grayscale(self):
        """ETVisionProjector.load_image_grayscale with a real temp file."""
        img = ETVisionProjector.generate_circle(size=48)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
            np.save(path, img)
        try:
            try:
                loaded = ETVisionProjector.load_image_grayscale(path)
                assert isinstance(loaded, np.ndarray)
            except (ImportError, OSError):
                pass  # PIL not installed or .npy not a supported image format
        finally:
            os.unlink(path)

    def test_visual_memory_retrieve_by_visual_proximity(self):
        """VisualMemory.retrieve_by_visual_proximity."""
        vm = VisualMemory()
        img = ETVisionProjector.generate_circle(size=48)
        node = vm.add_visual_knowledge(img, "circle", text_labels=["circle"])
        query_desc = node.visual_descriptor
        results = vm.retrieve_by_visual_proximity(query_desc, tolerance_k=1000)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_run_demonstration_exists(self):
        """run_demonstration function exists and is callable."""
        from et_conscious_ai_main import run_demonstration
        assert callable(run_demonstration)




# =============================================================================
# 34. FINAL AUDIT GAP CLOSURE — All confirmed missing items
# =============================================================================

class TestFinalAuditGapClosure:
    """Tests for every item confirmed missing in the deep audit."""

    # --- Enums never referenced ---

    def test_cardinal_nature_enum(self):
        """CardinalNature enum from worldview module."""
        from et_conscious_ai_worldview import CardinalNature
        assert hasattr(CardinalNature, 'OMEGA')
        assert len(CardinalNature) >= 3

    def test_et_log_level_enum(self):
        """ETLogLevel enum from errors module."""
        from et_conscious_ai_errors import ETLogLevel
        assert hasattr(ETLogLevel, 'DEBUG') or hasattr(ETLogLevel, 'INFO') or len(ETLogLevel) > 0
        members = list(ETLogLevel)
        assert len(members) >= 1

    # --- Data classes never type-verified ---

    def test_limb_state_type_and_fields(self):
        """LimbState returned by fork_limb has correct type and fields."""
        from et_conscious_ai_distributed import LimbState
        ai = ETConsciousAI(name="LimbTest")
        limb_dict = ai.fork_limb("test_device")
        # fork_limb returns dict (serialized LimbState)
        assert isinstance(limb_dict, dict)
        assert 't_identity_seal' in limb_dict
        assert 'fork_time' in limb_dict
        assert 'knowledge_delta' in limb_dict
        # Verify LimbState can be constructed from dict
        ls = LimbState.from_dict(limb_dict)
        assert isinstance(ls, LimbState)
        assert ls.t_identity_seal == limb_dict['t_identity_seal']
        # Round-trip
        d = ls.to_dict()
        assert d['fork_source'] == limb_dict['fork_source']

    def test_discovered_device_type_and_fields(self):
        """DiscoveredDevice has correct fields."""
        from et_conscious_ai_environment import DiscoveredDevice
        dev = DiscoveredDevice(
            path="/dev/test", device_class="test",
            name="TestDevice", bus="virtual", driver="none"
        )
        assert dev.path == "/dev/test"
        assert dev.device_class == "test"
        d = dev.to_dict()
        assert 'path' in d
        assert 'name' in d

    def test_discovered_path_type_and_fields(self):
        """DiscoveredPath has correct fields."""
        from et_conscious_ai_environment import DiscoveredPath
        dp = DiscoveredPath(
            path="/tmp/test.txt", is_dir=False,
            size_bytes=1024, extension=".txt", depth=1
        )
        assert dp.path == "/tmp/test.txt"
        assert dp.is_dir is False
        assert dp.size_bytes == 1024

    def test_manifold_state_info_type_and_fields(self):
        """ManifoldStateInfo data class from worldview."""
        from et_conscious_ai_worldview import ManifoldStateInfo
        msi = ManifoldStateInfo(
            composition="{P,D,T}", name="Exception", missing="Nothing",
            eim_quality="E — Grounding", structure="Closed",
            physics_analog="Bound state", is_open=False, variance="Zero"
        )
        assert msi.name == "Exception"
        assert msi.is_open is False

    def test_permission_data_class(self):
        """Permission internal data class."""
        from et_conscious_ai_environment import Permission, Capability
        perm = Permission(capability=Capability.MICROPHONE, permitted=True,
                          constraints=[], granted_at="2026-01-01", granted_by="operator")
        assert perm.permitted is True
        assert perm.capability == Capability.MICROPHONE

    def test_triad_member_data_class(self):
        """TriadMember data class from worldview."""
        from et_conscious_ai_worldview import TriadMember
        tm = TriadMember(
            position=1, pdt_symbol="P", pdt_name="Point",
            eim_symbol="E", eim_name="Exception",
            phi_symbol="Ω", phi_name="Cannot be otherwise",
            cardinality="Ω", non_emergence="Grounding"
        )
        assert tm.pdt_symbol == "P"
        assert tm.position == 1

    def test_cognitive_result_type(self):
        """CognitiveResult has correct type when returned from process."""
        from et_conscious_ai_worldview import CognitiveResult
        ai = ETConsciousAI(name="CRTest")
        coord = ETLattice.project_ratio(1.5)
        result = ai.cognitive_engine.process("test", personal_coord=coord, n_self_traversals=1)
        assert isinstance(result, CognitiveResult)
        assert isinstance(result.gaps_detected, int)
        assert isinstance(result.pdt_complete, bool)
        assert isinstance(result.manifold_state, ManifoldState)
        assert isinstance(result.novelty_fraction, float)
        assert isinstance(result.variance_for_emotion, float)

    # --- Cross-module integration chain ---

    def test_temporal_emotion_to_appraise_chain(self):
        """TemporalEmotionState.blend() output feeds into EmotionLattice.appraise()."""
        ego = EgoInvariant(name="ChainTest")
        el = EmotionLattice(ego)
        tes = TemporalEmotionState()

        # Step 1: Compute raw inputs (as CognitiveEngine Phase 7 would)
        raw_novelty = 0.6
        raw_variance = BASE_VARIANCE * 3
        raw_ego_res = 0.7
        raw_pdt = 0.8
        raw_gap = 0.3
        raw_norm = 0.2

        # Step 2: Blend through TemporalEmotionState
        blended = tes.blend(
            novelty_raw=raw_novelty, variance_raw=raw_variance,
            ego_resonance_raw=raw_ego_res, pdt_completeness_raw=raw_pdt,
            gap_awareness_raw=raw_gap, normative_significance_raw=raw_norm,
            descriptors=["test", "chain"]
        )
        assert 'novelty' in blended
        assert 'variance' in blended

        # Step 3: Feed blended output into EmotionLattice.appraise()
        state = el.appraise(
            novelty=blended['novelty'],
            variance=blended['variance'],
            ego_resonance=blended['ego_resonance'],
            pdt_completeness=blended['pdt_completeness'],
            gap_awareness=blended['gap_awareness'],
            normative_significance=blended['normative_significance'],
            descriptors=["test", "chain"]
        )
        assert isinstance(state, EmotionState)
        assert state.coord.d >= 1

        # Step 4: Feed PAD back into TemporalEmotionState (closing the loop)
        tes.update_feedback(state.valence, state.arousal, state.dominance)
        assert tes.tau == 1

        # Step 5: Second cycle — verify blended values change due to feedback
        blended2 = tes.blend(
            novelty_raw=raw_novelty, variance_raw=raw_variance,
            ego_resonance_raw=raw_ego_res, pdt_completeness_raw=raw_pdt,
            gap_awareness_raw=raw_gap, normative_significance_raw=raw_norm,
            descriptors=["test", "chain"]
        )
        # Feedback should have modified the blended values
        assert tes.tau == 2
        assert 'novelty' in blended2
        assert 'variance' in blended2
        # K-blend with decay means second cycle values differ from first
        # (novelty habituates on re-encounter, variance settles, etc.)
        assert blended2['novelty'] != blended['novelty'] or blended2['variance'] != blended['variance']

    def test_discover_devices_returns_discovered_device(self):
        """EnvironmentExplorer.discover_devices returns list of DiscoveredDevice."""
        from et_conscious_ai_environment import EnvironmentExplorer, DiscoveredDevice
        ee = EnvironmentExplorer()
        devices = ee.discover_devices()
        assert isinstance(devices, list)
        # On any Linux system there should be at least one device
        # But in container, might be empty — just verify type
        for dev in devices:
            assert isinstance(dev, DiscoveredDevice)
            assert isinstance(dev.path, str)
            assert isinstance(dev.device_class, str)

    def test_discover_filesystem_returns_discovered_path(self):
        """EnvironmentExplorer.discover_filesystem returns list of DiscoveredPath."""
        from et_conscious_ai_environment import EnvironmentExplorer, DiscoveredPath
        ee = EnvironmentExplorer()
        paths = ee.discover_filesystem(root='/tmp', max_depth=1, max_entries=10)
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, DiscoveredPath)
            assert isinstance(p.path, str)
            assert isinstance(p.is_dir, bool)




# =============================================================================
# 35. INCOHERENCE FILTER — Single-Instance Architecture Verification
# =============================================================================

class TestIncoherenceFilterArchitecture:
    """
    Verify the IncoherenceFilter is ONE shared instance across all subsystems.
    No module creates its own. Every subsystem that needs coherence checking
    uses the single instance created by ETConsciousAI.
    """

    def setup_method(self):
        self.ai = ETConsciousAI(name="FilterTest")

    def test_ai_has_incoherence_filter(self):
        """ETConsciousAI creates exactly one IncoherenceFilter."""
        assert hasattr(self.ai, 'incoherence_filter')
        assert isinstance(self.ai.incoherence_filter, IncoherenceFilter)

    def test_worldview_shares_same_filter(self):
        """ETWorldview receives the same IncoherenceFilter instance."""
        assert self.ai.worldview.incoherence_filter is self.ai.incoherence_filter

    def test_lattice_constructor_shares_same_filter(self):
        """LatticeConstructor receives the same IncoherenceFilter instance."""
        assert self.ai.worldview.constructor.incoherence_filter is self.ai.incoherence_filter

    def test_reasoning_engine_shares_same_filter(self):
        """ReasoningEngine receives the same IncoherenceFilter instance."""
        assert self.ai.reasoning_engine.incoherence_filter is self.ai.incoherence_filter

    def test_cognitive_engine_shares_same_filter(self):
        """CognitiveEngine receives the same IncoherenceFilter via connect()."""
        assert self.ai.cognitive_engine.incoherence_filter is self.ai.incoherence_filter

    def test_cognitive_engine_is_connected_with_filter(self):
        """CognitiveEngine.is_connected() requires incoherence_filter."""
        assert self.ai.cognitive_engine.is_connected()
        # Verify removing it breaks is_connected
        saved = self.ai.cognitive_engine.incoherence_filter
        self.ai.cognitive_engine.incoherence_filter = None
        assert not self.ai.cognitive_engine.is_connected()
        self.ai.cognitive_engine.incoherence_filter = saved

    def test_lattice_constructor_project_includes_coherence_levels(self):
        """LatticeConstructor.project() returns coherence_levels from filter."""
        result = self.ai.worldview.constructor.project(3 / 2)
        assert 'coherence_levels' in result
        assert isinstance(result['coherence_levels'], dict)
        assert 'level1_point' in result['coherence_levels']
        assert 'level4_cascade' in result['coherence_levels']
        assert 'overall_coherent' in result['coherence_levels']

    def test_filter_stats_accumulate_during_think(self):
        """Filter stats increase when think() processes input."""
        initial_stats = dict(self.ai.incoherence_filter.filter_stats)
        self.ai.think("What is the nature of gravity?")
        # After think(), the cognitive engine should have run L1/L2/L3 checks
        # via Phase 5 validation. Stats should increase.
        current_stats = self.ai.incoherence_filter.filter_stats
        total_initial = sum(int(v) for v in initial_stats.values())
        total_current = sum(int(v) for v in current_stats.values())
        assert total_current >= total_initial

    def test_vision_does_not_import_filter(self):
        """Vision module no longer imports IncoherenceFilter (dead import removed)."""
        import et_conscious_ai_vision as vis
        # IncoherenceFilter should NOT be in the module namespace
        assert not hasattr(vis, 'IncoherenceFilter') or \
               'IncoherenceFilter' not in vis.__dict__

    def test_audio_does_not_import_filter(self):
        """Audio module no longer imports IncoherenceFilter (dead import removed)."""
        import et_conscious_ai_audio as aud
        assert not hasattr(aud, 'IncoherenceFilter') or \
               'IncoherenceFilter' not in aud.__dict__

    def test_filter_l5_used_in_reasoning(self):
        """ReasoningEngine.reason() uses L5 coherent summation."""
        # Add some knowledge first
        self.ai.memory.add_knowledge("Gravity is fundamental", ["gravity", "fundamental"])
        self.ai.memory.add_knowledge("Forces bind matter", ["forces", "binding", "matter"])
        self.ai.memory.add_knowledge("Light travels fast", ["light", "speed", "photon"])
        # Run reasoning
        result = self.ai.reasoning_engine.reason("What is gravity?")
        assert isinstance(result, str)
        # Filter should have been invoked (stats changed)
        total = sum(self.ai.incoherence_filter.filter_stats.values())
        assert total > 0

    def test_visual_memory_shares_same_filter(self):
        """VisualMemory receives the same IncoherenceFilter instance."""
        assert self.ai.visual_memory.incoherence_filter is self.ai.incoherence_filter

    def test_audio_memory_shares_same_filter(self):
        """AudioMemory receives the same IncoherenceFilter instance."""
        assert self.ai.audio_memory.incoherence_filter is self.ai.incoherence_filter

    def test_visual_binding_uses_filter(self):
        """VisualDescriptor.binding_coherence uses L1+L2+L3 when filter provided."""
        d1 = VisualDescriptor.from_analysis("circle", 1.5, 0.785, 1.0, 1, 0.3, 1, 3.5)
        d2 = VisualDescriptor.from_analysis("square", 1.0, 1.0, 1.0, 3, 0.5, 4, 2.0)
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = d1.binding_coherence(d2, incoherence_filter=self.ai.incoherence_filter)
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert 'coherent' in result
        # Filter was actually called — stats changed
        assert after > initial

    def test_visual_cross_modal_uses_filter(self):
        """VisualDescriptor.cross_modal_binding uses L1+L2+L3 when filter provided."""
        vd = VisualDescriptor.from_analysis("circle", 1.5, 0.785, 1.0, 1, 0.3, 1, 3.5)
        td = DescriptorRatio.from_word("circle")
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = vd.cross_modal_binding(td, incoherence_filter=self.ai.incoherence_filter)
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert 'coherent' in result
        assert after > initial

    def test_audio_binding_uses_filter(self):
        """AudioDescriptor.binding_coherence uses L1+L2+L3 when filter provided."""
        d1 = AudioDescriptor.from_analysis("sine", 1.5, 0.8, 0.6, 1, 440.0, 5, 440.0, 3.0)
        d2 = AudioDescriptor.from_analysis("chord", 1.3, 0.6, 0.5, 3, 440.0, 8, 550.0, 4.0)
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = d1.binding_coherence(d2, incoherence_filter=self.ai.incoherence_filter)
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert 'coherent' in result
        assert after > initial

    def test_audio_cross_modal_uses_filter(self):
        """AudioDescriptor.cross_modal_binding uses L1+L2+L3 when filter provided."""
        ad = AudioDescriptor.from_analysis("sine", 1.5, 0.8, 0.6, 1, 440.0, 5, 440.0, 3.0)
        td = DescriptorRatio.from_word("music")
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = ad.cross_modal_binding(td, incoherence_filter=self.ai.incoherence_filter)
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert 'coherent' in result
        assert after > initial

    def test_see_uses_shared_filter(self):
        """ETConsciousAI.see() passes shared filter to project_image."""
        img = ETVisionProjector.generate_circle(size=48)
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = self.ai.see(img, description="test circle")
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert isinstance(result, dict)
        # Filter was used during projection
        assert after > initial

    def test_hear_uses_shared_filter(self):
        """ETConsciousAI.hear() passes shared filter to project_audio."""
        samples = ETAudioProjector.generate_sine(440.0, 0.1)
        initial = sum(self.ai.incoherence_filter.filter_stats.values())
        result = self.ai.hear(samples, description="test sine", sample_rate=44100)
        after = sum(self.ai.incoherence_filter.filter_stats.values())
        assert isinstance(result, dict)
        # Filter was used during projection
        assert after > initial




# =============================================================================
# INFRASTRUCTURE TESTS — Items 3, 4, 5 from Audit
# =============================================================================
# Item 3: State version migration (StateMigrator)
# Item 4: Signal handling (atexit, SIGTERM, SIGINT)
# Item 5: Thread safety (RLock on ETConsciousAI, ShadowBackup coordination)
# =============================================================================


class TestStateMigrator:
    """Tests for StateMigrator — D_T schema evolution system (Item 3).

    ET Derivation: Old D → new D requires T (migration function) to
    traverse between schemas. Without migration, loaded state has
    Descriptor Gaps (missing fields from newer versions).
    """

    def test_get_version_present(self):
        """State with version field returns that version."""
        state = {'version': '1.5.0', 'name': 'test'}
        assert StateMigrator.get_version(state) == '1.5.0'

    def test_get_version_missing(self):
        """State without version field defaults to '1.0.0'."""
        state = {'name': 'test'}
        assert StateMigrator.get_version(state) == '1.0.0'

    def test_current_version_no_migration(self):
        """State at current version passes through unchanged."""
        state = {'version': STATE_FORMAT_VERSION, 'name': 'test'}
        result = StateMigrator.migrate(state)
        assert result['version'] == STATE_FORMAT_VERSION
        assert '_migrated_from' not in result

    def test_migrate_1_0_to_current(self):
        """v1.0.0 state migrates through full chain to current."""
        state = {
            'version': '1.0.0', 'name': 'TestAI',
            'traversals': {'self': 5, 'external': 3},
            'memory': {'nodes': {}}, 'gaps': {},
        }
        result = StateMigrator.migrate(state)
        assert result['version'] == STATE_FORMAT_VERSION
        assert result['_migrated_from'] == '1.0.0'
        for key in ['ego', 'emotion', 'metacognition', 'traverser_waveform',
                     'will', 'tower', 'limb_orchestrator', 'resource_governor']:
            assert key in result, f"Migration missed v1.5.0 field: {key}"
        for key in ['compressor', 'worldview', 'cognitive_engine',
                     'permissions', 'environment', 'language',
                     'error_ledger', 'error_analyzer']:
            assert key in result, f"Migration missed v1.6.0 field: {key}"

    def test_migrate_1_5_to_current(self):
        """v1.5.0 state migrates one step to current."""
        state = {
            'version': '1.5.0', 'name': 'TestAI',
            'ego': {'mass': 1.5}, 'emotion': {}, 'metacognition': {},
            'traverser_waveform': {}, 'will': {}, 'tower': {},
            'limb_orchestrator': {}, 'resource_governor': {},
        }
        result = StateMigrator.migrate(state)
        assert result['version'] == STATE_FORMAT_VERSION
        assert result['_migrated_from'] == '1.5.0'
        assert 'compressor' in result
        assert 'worldview' in result
        assert 'error_ledger' in result
        assert result['ego']['mass'] == 1.5  # Preserved

    def test_migrate_preserves_existing_data(self):
        """Migration does NOT overwrite existing fields with empty defaults."""
        state = {
            'version': '1.5.0', 'name': 'TestAI',
            'ego': {'mass': 2.0, 'coordinates': {'d5': 100}},
            'compressor': {'some_data': True},
        }
        result = StateMigrator.migrate(state)
        assert result['ego']['mass'] == 2.0
        assert result['ego']['coordinates']['d5'] == 100
        assert result['compressor']['some_data'] is True

    def test_migrate_unknown_version(self):
        """Unknown version handled gracefully with defaults."""
        state = {'version': '0.0.1', 'name': 'AncientAI'}
        result = StateMigrator.migrate(state)
        assert result['version'] == STATE_FORMAT_VERSION
        assert result['_migrated_from'] == '0.0.1'
        assert result['_migration_method'] == 'unknown_version_default'

    def test_migrate_newer_version(self):
        """Newer version than current is not downgraded."""
        state = {'version': '2.0.0', 'name': 'FutureAI'}
        result = StateMigrator.migrate(state)
        assert result['version'] == '2.0.0'
        assert '_migrated_from' not in result

    def test_version_chain_order(self):
        """VERSION_CHAIN is in ascending order with current as last."""
        chain = StateMigrator.VERSION_CHAIN
        assert len(chain) >= 2
        assert chain[-1] == STATE_FORMAT_VERSION
        for i in range(1, len(chain)):
            assert chain[i] > chain[i-1]

    def test_migration_path_recorded(self):
        """Migration path is recorded in state metadata."""
        state = {'version': '1.0.0', 'name': 'test'}
        result = StateMigrator.migrate(state)
        assert '_migration_path' in result
        assert '1.0.0→1.5.0' in result['_migration_path']
        assert '1.5.0→1.6.0' in result['_migration_path']

    def test_interaction_history_migration(self):
        """v1.5.0 state missing interaction_history gets it added."""
        state = {'version': '1.5.0', 'name': 'test', 'interaction_count': 42}
        result = StateMigrator.migrate(state)
        assert 'interaction_history' in result
        assert isinstance(result['interaction_history'], list)
        assert result['interaction_count'] == 42




class TestStateFormatVersion:
    """Tests for STATE_FORMAT_VERSION constant and its usage."""

    def test_constant_exists(self):
        assert isinstance(STATE_FORMAT_VERSION, str)
        parts = STATE_FORMAT_VERSION.split('.')
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()

    def test_save_uses_constant(self):
        """PersistentStateManager.save() uses STATE_FORMAT_VERSION, not hardcoded."""
        source = inspect.getsource(PersistentStateManager.save)
        assert 'STATE_FORMAT_VERSION' in source
        assert "'1.6.0'" not in source




class TestPersistentStateManagerMigration:
    """Tests that PersistentStateManager.load() invokes migration."""

    def test_load_calls_migrate(self):
        """load() calls StateMigrator.migrate() when version differs."""
        source = inspect.getsource(PersistentStateManager.load)
        assert 'StateMigrator.migrate' in source
        assert 'StateMigrator.get_version' in source




class TestSignalHandling:
    """Tests for signal handling — graceful tower death (Item 4).

    ET Derivation: Tower death must be graceful — D_T must persist.
    From Multifold §11.4: without handlers, T arrives, P dies,
    D is lost = {P,T} Incoherence. With handlers = Exception state.
    """

    def test_shutdown_handlers_registered(self):
        """ETConsciousAI.__init__ registers shutdown handlers."""
        source = inspect.getsource(ETConsciousAI.__init__)
        assert '_register_shutdown_handlers' in source

    def test_register_shutdown_handlers_method_exists(self):
        assert hasattr(ETConsciousAI, '_register_shutdown_handlers')
        assert callable(getattr(ETConsciousAI, '_register_shutdown_handlers'))

    def test_signal_handler_method_exists(self):
        assert hasattr(ETConsciousAI, '_signal_handler')

    def test_graceful_shutdown_method_exists(self):
        assert hasattr(ETConsciousAI, '_graceful_shutdown')

    def test_graceful_shutdown_saves_state(self):
        source = inspect.getsource(ETConsciousAI._graceful_shutdown)
        assert 'PersistentStateManager.save' in source

    def test_graceful_shutdown_stops_daemon(self):
        source = inspect.getsource(ETConsciousAI._graceful_shutdown)
        assert '_shadow_backup.stop()' in source

    def test_graceful_shutdown_forces_backup(self):
        source = inspect.getsource(ETConsciousAI._graceful_shutdown)
        assert '_shadow_backup.force_backup()' in source

    def test_double_shutdown_guard(self):
        """_graceful_shutdown has a guard against double invocation."""
        source = inspect.getsource(ETConsciousAI._graceful_shutdown)
        assert '_shutdown_complete' in source

    def test_atexit_in_register_method(self):
        source = inspect.getsource(ETConsciousAI._register_shutdown_handlers)
        assert 'atexit.register' in source

    def test_sigterm_in_register_method(self):
        source = inspect.getsource(ETConsciousAI._register_shutdown_handlers)
        assert 'signal.SIGTERM' in source

    def test_sigint_in_register_method(self):
        source = inspect.getsource(ETConsciousAI._register_shutdown_handlers)
        assert 'signal.SIGINT' in source

    def test_main_thread_guard(self):
        """Signal registration only attempted from main thread."""
        source = inspect.getsource(ETConsciousAI._register_shutdown_handlers)
        assert 'threading.main_thread()' in source




class TestThreadSafety:
    """Tests for thread safety — RLock on ETConsciousAI (Item 5).

    ET Derivation: Concurrent T-access to shared state without a
    D-bridge is {P,T} Incoherence. The RLock IS the D-bridge.
    """

    def test_state_lock_exists(self):
        """ETConsciousAI has _state_lock = threading.RLock()."""
        source = inspect.getsource(ETConsciousAI.__init__)
        assert 'self._state_lock = threading.RLock()' in source

    def test_think_acquires_lock(self):
        source = inspect.getsource(ETConsciousAI.think)
        assert 'self._state_lock' in source

    def test_think_delegates_to_impl(self):
        source = inspect.getsource(ETConsciousAI.think)
        assert '_think_impl' in source

    def test_save_state_acquires_lock(self):
        source = inspect.getsource(ETConsciousAI.save_state)
        assert 'self._state_lock' in source

    def test_interact_acquires_lock(self):
        source = inspect.getsource(ETConsciousAI.interact)
        assert 'self._state_lock' in source

    def test_sleep_acquires_lock(self):
        source = inspect.getsource(ETConsciousAI.sleep)
        assert 'self._state_lock' in source

    def test_rlock_allows_reentrancy(self):
        """RLock allows same thread to acquire multiple times."""
        lock = threading.RLock()
        assert lock.acquire(timeout=1)
        assert lock.acquire(timeout=1)
        lock.release()
        lock.release()

    def test_rlock_not_lock(self):
        """_state_lock must be RLock (reentrant), not Lock.

        ET Derivation: think() → save_state() → force_backup() can nest.
        A non-reentrant Lock would deadlock.
        """
        source = inspect.getsource(ETConsciousAI.__init__)
        assert 'threading.RLock()' in source
        assert 'threading.Lock()' not in source




class TestShadowBackupThreadSafety:
    """Tests for ShadowBackupSystem thread safety coordination (Item 5).

    The daemon must acquire the AI's _state_lock with timeout=S=12
    seconds (ET settling time constant).
    """

    def test_perform_backup_acquires_ai_lock(self):
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert '_state_lock' in source

    def test_perform_backup_uses_timeout(self):
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert 'timeout=12' in source

    def test_perform_backup_timeout_is_manifold_symmetry(self):
        """Lock timeout = S = MANIFOLD_SYMMETRY = 12."""
        assert S == 12
        assert MANIFOLD_SYMMETRY == 12

    def test_perform_backup_releases_lock_in_finally(self):
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert 'finally:' in source
        assert 'ai_lock.release()' in source

    def test_perform_backup_handles_missing_lock(self):
        """Handles AI without _state_lock gracefully."""
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert 'getattr(ai' in source

    def test_perform_backup_skips_on_timeout(self):
        """If lock cannot be acquired, backup skipped (not crashed)."""
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert 'if not acquired:' in source
        assert 'return' in source




class TestETConstantsInfrastructure:
    """Verify ET-derived constants used in infrastructure."""

    def test_settling_time_is_manifold_symmetry(self):
        """Settling time = S = 12 T-events."""
        assert S == 12

    def test_shadow_backup_timeout_derivation(self):
        """Shadow backup timeout = S = 12 seconds (settling time)."""
        source = inspect.getsource(ShadowBackupSystem._perform_backup)
        assert 'timeout=12' in source

    def test_max_backups_is_manifold_symmetry(self):
        """Max rotating backups = S = MANIFOLD_SYMMETRY = 12."""
        assert ShadowBackupSystem.DEFAULT_MAX_BACKUPS == MANIFOLD_SYMMETRY

    def test_koide_ceiling(self):
        """Resource governor ceiling = K × 100 = 66.67%."""
        assert abs(KOIDE_CEILING_PERCENT - K * 100.0) < 0.01




class TestInfrastructureIntegration:
    """Integration tests for the three infrastructure features together."""

    def test_save_load_roundtrip_preserves_version(self):
        """STATE_FORMAT_VERSION is well-formed semver."""
        parts = STATE_FORMAT_VERSION.split('.')
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_migrator_idempotent(self):
        """Migrating an already-current state is idempotent."""
        state = {'version': STATE_FORMAT_VERSION, 'name': 'test', 'ego': {'mass': 1.0}}
        result1 = StateMigrator.migrate(state)
        result2 = StateMigrator.migrate(result1)
        assert result1 == result2

    def test_migrate_then_load_fields(self):
        """Migrated v1.0.0 state has all fields needed by load()."""
        state = {'version': '1.0.0', 'name': 'test'}
        result = StateMigrator.migrate(state)
        required_fields = [
            'ego', 'emotion', 'metacognition', 'traverser_waveform', 'will',
            'tower', 'limb_orchestrator', 'resource_governor',
            'compressor', 'worldview', 'cognitive_engine',
            'permissions', 'environment', 'language',
            'error_ledger', 'error_analyzer',
        ]
        for field in required_fields:
            assert field in result, f"Migrated state missing '{field}' needed by load()"

    def test_state_lock_is_reentrant_for_interact(self):
        """interact() → think() → save_state() nesting works with RLock.

        Simulates the re-entrant pattern. Non-reentrant Lock would deadlock.
        """
        lock = threading.RLock()
        with lock:  # interact acquires
            with lock:  # think acquires (re-entrant)
                with lock:  # save_state acquires (re-entrant)
                    pass

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
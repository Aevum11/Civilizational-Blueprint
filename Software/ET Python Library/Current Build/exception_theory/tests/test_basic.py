"""
Exception Theory - Basic Test Suite

Tests core functionality of the Exception Theory library.

From: "For every exception there is an exception, except the exception."

Author: ET Development Team
"""

import pytest
from exception_theory import (
    ETSovereign,
    ETMathV2,
    Point,
    Descriptor,
    Traverser,
    bind_pdt,
    TraverserEntropy,
    TrinaryState,
    SwarmConsensus,
    SemanticManifold,
    TimeTraveler,
    FractalReality,
)


def test_version_import():
    """Test that version info is accessible."""
    from exception_theory import __version__
    assert __version__ == "3.0.0"


def test_et_math_basic():
    """Test basic ET mathematics."""
    # Density calculation
    density = ETMathV2.density(64, 100)
    assert 0.63 < density < 0.65
    
    # Effort calculation
    effort = ETMathV2.effort(3, 4)
    assert 4.9 < effort < 5.1  # sqrt(9 + 16) = 5
    
    # Phase transition
    status = ETMathV2.phase_transition(0.0)
    assert 0.49 < status < 0.51  # Should be ~0.5 at threshold


def test_primitives():
    """Test ET primitive types."""
    # Create point
    p = Point(location="origin", state=5)
    assert p.location == "origin"
    assert p.state == 5
    
    # Create descriptor
    d = Descriptor(name="positive", constraint=lambda x: x > 0)
    assert d.name == "positive"
    assert d.apply(p) is True  # 5 > 0
    
    # Create traverser
    t = Traverser(identity="test")
    assert t.identity == "test"
    
    # Bind into exception
    e = bind_pdt(p, d, t)
    assert e.is_coherent() is True


def test_traverser_entropy():
    """Test true entropy generation."""
    entropy_gen = TraverserEntropy()
    
    # Generate entropy strings
    val1 = entropy_gen.substantiate(32)
    val2 = entropy_gen.substantiate(32)
    
    assert len(val1) == 32
    assert len(val2) == 32
    assert val1 != val2  # Should be different (true random)
    
    # Generate entropy bytes
    bytes_val = entropy_gen.substantiate_bytes(16)
    assert len(bytes_val) == 16


def test_trinary_state():
    """Test trinary logic."""
    # Create superposition
    state = TrinaryState(2, bias=0.7)
    assert state._state == 2
    assert state._bias == 0.7
    
    # Collapse
    collapsed = state.collapse()
    assert collapsed in [0, 1]
    
    # Logic operations
    true_state = TrinaryState(1)
    false_state = TrinaryState(0)
    
    result = true_state & false_state
    assert result._state == 0  # TRUE AND FALSE = FALSE


def test_swarm_consensus():
    """Test distributed consensus."""
    # Create swarm nodes
    nodes = [
        SwarmConsensus(f"node_{i}", "StateA" if i < 3 else "StateB")
        for i in range(7)
    ]
    
    # Majority is StateB (4 nodes)
    # Run gossip rounds
    for _ in range(5):
        for node in nodes:
            import random
            neighbors = random.sample(nodes, min(3, len(nodes)))
            node.gossip(neighbors)
    
    # Check convergence (most nodes should align to StateB)
    state_b_count = sum(1 for n in nodes if n.data == "StateB")
    assert state_b_count >= 5  # At least 5/7 should converge


def test_semantic_manifold():
    """Test semantic similarity."""
    manifold = SemanticManifold("test")
    
    # Bind embeddings
    manifold.bind_batch({
        "king": [0.9, 0.8, 0.1],
        "queen": [0.9, 0.9, 0.1],
        "apple": [0.1, 0.1, 0.9],
    })
    
    # Search for similar
    results = manifold.search("king", top_k=2)
    assert len(results) == 2
    assert results[0][0] == "queen"  # Queen should be most similar to king


def test_time_traveler():
    """Test event sourcing with undo/redo."""
    tt = TimeTraveler("test")
    
    # Record changes
    tt.commit("score", 100)
    assert tt.state["score"] == 100
    
    tt.commit("score", 200)
    assert tt.state["score"] == 200
    
    # Undo
    success = tt.undo()
    assert success is True
    assert tt.state["score"] == 100
    
    # Redo
    success = tt.redo()
    assert success is True
    assert tt.state["score"] == 200


def test_fractal_reality():
    """Test procedural generation."""
    world = FractalReality("test_world", seed=42, octaves=3)
    
    # Same coordinates should give same result (deterministic)
    h1 = world.get_elevation(10.5, 20.7)
    h2 = world.get_elevation(10.5, 20.7)
    assert h1 == h2
    
    # Different coordinates should give different results
    h3 = world.get_elevation(100.0, 200.0)
    assert h1 != h3
    
    # Render chunk
    chunk = world.render_chunk(0, 0, size=5)
    assert len(chunk) == 5
    assert len(chunk[0]) == 5


def test_et_sovereign_basic():
    """Test main engine initialization."""
    engine = ETSovereign()
    
    # Should initialize without errors
    assert engine is not None
    
    # Test calibration
    cal_result = engine.calibrate()
    assert cal_result['calibrated'] is True
    assert 'char_width' in cal_result
    assert 'platform' in cal_result


def test_et_sovereign_entropy():
    """Test engine entropy generation."""
    engine = ETSovereign()
    
    entropy1 = engine.generate_true_entropy(32)
    entropy2 = engine.generate_true_entropy(32)
    
    assert len(entropy1) == 32
    assert len(entropy2) == 32
    assert entropy1 != entropy2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

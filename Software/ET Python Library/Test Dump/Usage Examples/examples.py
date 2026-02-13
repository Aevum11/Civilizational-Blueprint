#!/usr/bin/env python3
"""
Exception Theory - Comprehensive Example

Demonstrates the major capabilities of the Exception Theory library.

From: "For every exception there is an exception, except the exception."

Author: ET Development Team
"""

from exception_theory import (
    ETSovereign,
    ETMathV2,
    Point,
    Descriptor,
    Traverser,
    bind_pdt,
    SemanticManifold,
    TimeTraveler,
    FractalReality,
    SwarmConsensus,
)


def example_primitives():
    """Example: Working with ET primitives."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: ET Primitives")
    print("=" * 60)
    
    # Create a Point (substrate)
    p = Point(location="quantum_state", state=5)
    print(f"Point: {p.location} = {p.state}")
    
    # Create a Descriptor (constraint)
    d = Descriptor(
        name="positive_only",
        constraint=lambda x: x > 0,
        metadata={"description": "Values must be positive"}
    )
    print(f"Descriptor: {d.name}")
    
    # Create a Traverser (agency)
    t = Traverser(identity="observer_1")
    print(f"Traverser: {t.identity}")
    
    # Bind into an Exception
    exception = bind_pdt(p, d, t)
    print(f"Exception coherent? {exception.is_coherent()}")  # True (5 > 0)
    
    # Change Point state to violate constraint
    p.substantiate(-5)
    print(f"After changing to -5, coherent? {exception.is_coherent()}")  # False


def example_mathematics():
    """Example: ET Mathematics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: ET Mathematics")
    print("=" * 60)
    
    # Structural density
    density = ETMathV2.density(payload=64, container=100)
    print(f"Density: {density:.3f}")
    
    # Traverser effort
    effort = ETMathV2.effort(observers=3, byte_delta=4)
    print(f"Effort: {effort:.3f}")
    
    # Phase transition
    transition = ETMathV2.phase_transition(gradient_input=2.0)
    print(f"Phase transition at gradient=2.0: {transition:.3f}")
    
    # Variance minimization
    current_var = 10.0
    target_var = 5.0
    next_var = ETMathV2.variance_gradient(current_var, target_var, step_size=0.5)
    print(f"Variance: {current_var} → {next_var:.3f} (target: {target_var})")
    
    # Universal resolution (handles singularities)
    result = ETMathV2.universal_resolution(1.0, 0.0)  # 1/0 singularity
    print(f"Universal resolution of 1/0: {result}")


def example_sovereign_engine():
    """Example: Main ET Sovereign Engine."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: ET Sovereign Engine")
    print("=" * 60)
    
    engine = ETSovereign()
    
    # Calibrate
    cal = engine.calibrate()
    print(f"Calibrated: {cal['calibrated']}")
    print(f"Platform: {cal['platform']}")
    print(f"Character width: {cal['char_width']}")
    
    # Generate true entropy
    entropy = engine.generate_true_entropy(32)
    print(f"True entropy (32 chars): {entropy}")
    
    # Create trinary state
    state = engine.create_trinary_state(2, bias=0.8)
    print(f"Trinary state: {state} (bias toward TRUE: 0.8)")
    collapsed = state.collapse()
    print(f"Collapsed to: {collapsed}")


def example_semantic_manifold():
    """Example: Semantic similarity and analogies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Semantic Manifold")
    print("=" * 60)
    
    manifold = SemanticManifold("concepts")
    
    # Simple embeddings for demonstration
    embeddings = {
        "king": [0.9, 0.8, 0.1, 0.2],
        "queen": [0.9, 0.9, 0.1, 0.2],
        "man": [0.8, 0.2, 0.2, 0.1],
        "woman": [0.8, 0.3, 0.2, 0.1],
        "prince": [0.85, 0.75, 0.15, 0.2],
    }
    
    manifold.bind_batch(embeddings)
    print(f"Vocabulary: {list(embeddings.keys())}")
    
    # Search for similar concepts
    results = manifold.search("king", top_k=3)
    print(f"\nMost similar to 'king':")
    for word, similarity in results:
        print(f"  {word}: {similarity:.3f}")
    
    # Solve analogy: king:queen :: man:?
    analogy = manifold.analogy("king", "queen", "man", top_k=1)
    print(f"\nAnalogy king:queen :: man:? → {analogy[0][0]}")


def example_time_travel():
    """Example: Event sourcing with undo/redo."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Time Travel (Event Sourcing)")
    print("=" * 60)
    
    tt = TimeTraveler("game_state")
    
    # Record state changes
    print("Recording changes:")
    tt.commit("player_health", 100)
    print(f"  health = 100: {tt.state}")
    
    tt.commit("player_health", 75)
    print(f"  health = 75: {tt.state}")
    
    tt.commit("player_health", 50)
    print(f"  health = 50: {tt.state}")
    
    # Travel backward in time
    print("\nTime travel - UNDO:")
    tt.undo()
    print(f"  After undo: {tt.state}")
    
    tt.undo()
    print(f"  After undo: {tt.state}")
    
    # Travel forward in time
    print("\nTime travel - REDO:")
    tt.redo()
    print(f"  After redo: {tt.state}")


def example_fractal_reality():
    """Example: Procedural world generation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Fractal Reality (Procedural Generation)")
    print("=" * 60)
    
    # Create deterministic world
    world = FractalReality("demo_world", seed=42, octaves=3, persistence=0.5)
    print(f"World seed: 42")
    print(f"Octaves: 3, Persistence: 0.5")
    
    # Get elevation at specific coordinates (always same for same seed)
    h1 = world.get_elevation(100.0, 200.0)
    h2 = world.get_elevation(100.0, 200.0)
    print(f"\nElevation at (100, 200): {h1:.3f}")
    print(f"Second query (same coords): {h2:.3f}")
    print(f"Deterministic? {h1 == h2}")
    
    # Render ASCII terrain
    print("\nTerrain chunk (10x10) at origin:")
    chunk = world.render_chunk_string(0, 0, size=10)
    print(chunk)


def example_swarm_consensus():
    """Example: Distributed consensus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Swarm Consensus")
    print("=" * 60)
    
    # Create cluster with split-brain
    nodes = [
        SwarmConsensus(f"node_{i}", "StateA" if i < 3 else "StateB")
        for i in range(7)
    ]
    
    print("Initial state: 3 nodes=StateA, 4 nodes=StateB")
    
    # Run gossip rounds
    print("\nGossip rounds (variance minimization):")
    for round_num in range(3):
        alignments = 0
        for node in nodes:
            import random
            neighbors = random.sample(nodes, min(3, len(nodes)))
            result = node.gossip(neighbors)
            if result['aligned']:
                alignments += 1
        print(f"  Round {round_num + 1}: {alignments} alignments")
    
    # Check final state
    state_counts = {}
    for node in nodes:
        state = str(node.data)
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print(f"\nFinal state distribution: {state_counts}")
    print("(Nodes converge to majority via data gravity)")


def main():
    """Run all examples."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              EXCEPTION THEORY - COMPREHENSIVE EXAMPLES           ║
    ║                                                                  ║
    ║  "For every exception there is an exception, except exception"  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    example_primitives()
    example_mathematics()
    example_sovereign_engine()
    example_semantic_manifold()
    example_time_travel()
    example_fractal_reality()
    example_swarm_consensus()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

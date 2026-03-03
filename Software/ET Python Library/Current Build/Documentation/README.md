# Exception Theory - Python Library

> "For every exception there is an exception, except the exception."

A comprehensive mathematical and computational framework built on three fundamental primitives that describe all of existence.

## Overview

Exception Theory (ET) is a complete ontological framework that posits three irreducible primitives:

- **P (Point)**: The substrate of existence - pure potential, location without property
- **D (Descriptor)**: Constraints and properties - what limits and defines what can exist
- **T (Traverser)**: Agency and navigation - the principle of choice and movement

These combine via the master equation: **PDT = EIM = S**
- Exception = Incoherence = Mediation = Something (total existence)

## Installation

```bash
pip install exception-theory
```

Or install from source:

```bash
git clone <repository-url>
cd exception-theory
pip install -e .
```

## Quick Start

```python
from exception_theory import ETSovereign, ETMathV2

# Initialize the main engine
engine = ETSovereign()

# Generate true entropy from T-singularities
entropy = engine.generate_true_entropy(32)
print(f"True random: {entropy}")

# Create trinary logic states (superposition computing)
state = engine.create_trinary_state(2, bias=0.7)  # 70% toward TRUE
collapsed = state.collapse()

# Navigate manifolds with T-path optimization
path = engine.navigate_manifold('Start', 'End', graph_descriptor)

# Use ET mathematics directly
density = ETMathV2.density(payload=64, container=100)  # 0.64
variance = ETMathV2.variance_gradient(current=10.0, target=5.0)
```

## Core Capabilities

### Computational Exception Theory (Batch 1)
- **TraverserEntropy**: True entropy from T-singularities (race conditions)
- **TrinaryState**: Superposition computing with bias propagation
- **ChameleonObject**: Polymorphic contextual binding (pure relativism)
- **TraverserMonitor**: Halting heuristic via state recurrence detection
- **RealityGrounding**: Exception handler for system coherence
- **TemporalCoherenceFilter**: Kalman filtering for variance minimization
- **EvolutionarySolver**: Genetic algorithms via manifold optimization
- **PNumber**: Infinite precision arithmetic

### Advanced Manifold Architectures (Batch 2)
- **TeleologicalSorter**: O(n) sorting via direct coordinate mapping
- **ProbabilisticManifold**: Bloom filter for existence testing
- **HolographicValidator**: Merkle tree for integrity verification
- **ZeroKnowledgeProtocol**: Prove knowledge without revealing secrets
- **ContentAddressableStorage**: Identity-based addressing (CAS)
- **ReactivePoint**: Observer pattern for manifold consistency
- **GhostSwitch**: Dead man's trigger for inactivity detection
- **UniversalAdapter**: Aggressive type transmutation

### Distributed Consciousness (Batch 3)
- **SwarmConsensus**: Byzantine consensus via variance minimization
- **PrecognitiveCache**: Trajectory extrapolation for negative latency
- **ImmortalSupervisor**: Homeostatic crash recovery
- **SemanticManifold**: Meaning as geometric proximity
- **VarianceLimiter**: Entropy-based adaptive rate limiting
- **ProofOfTraversal**: Anti-spam hashcash protocol
- **EphemeralVault**: Perfect forward secrecy encryption
- **ConsistentHashingRing**: Sharded DHT topology
- **TimeTraveler**: Event sourcing with undo/redo
- **FractalReality**: Procedural world generation

## Mathematical Foundations

Exception Theory includes 40+ equations across 10 batches:

```python
from exception_theory import ETMathV2

# Structural density (Eq 211)
S = ETMathV2.density(payload, container)

# Traverser effort (Eq 212)
effort = ETMathV2.effort(observers, byte_delta)

# Phase transition (Eq 30)
status = ETMathV2.phase_transition(gradient_input)

# Variance minimization (Eq 83)
D_next = ETMathV2.variance_gradient(D_current, D_target)

# Kolmogorov complexity (Eq 77)
min_descriptors = ETMathV2.kolmogorov_complexity(descriptor_set)

# Universal Resolution Function (handles singularities)
result = ETMathV2.universal_resolution(numerator, denominator)

# Swarm consensus (Eq 21)
weight = ETMathV2.swarm_consensus_weight(state_hash, all_hashes)

# Semantic similarity (Eq 24)
similarity = ETMathV2.cosine_similarity(vec_a, vec_b)

# Fractal noise (Eq 30)
elevation = ETMathV2.fractal_noise(x, y, seed, octaves, persistence)
```

## Advanced Examples

### Distributed Consensus

```python
from exception_theory import ETSovereign

engine = ETSovereign()

# Create a swarm cluster
nodes = engine.create_swarm_cluster("cluster", count=10, initial_data="V1")

# Introduce split-brain
for i in range(7):
    engine.get_swarm_node(f"cluster_{i}").data = "V2"

# Execute gossip rounds - nodes drift to consensus via variance minimization
for round in range(5):
    result = engine.swarm_gossip_round(peer_count=3)
    print(f"Round {round}: {result['alignments']} alignments")
```

### Semantic Search

```python
from exception_theory import SemanticManifold

# Create meaning space
manifold = SemanticManifold("concepts")

# Bind words to descriptor vectors
manifold.bind_batch({
    "king": [0.9, 0.8, 0.1, 0.2],
    "queen": [0.9, 0.9, 0.1, 0.2],
    "man": [0.8, 0.2, 0.2, 0.1],
    "woman": [0.8, 0.3, 0.2, 0.1],
})

# Search for similar concepts
results = manifold.search("king", top_k=3)
# [('queen', 0.998), ('man', 0.879), ...]

# Solve analogies: king:queen :: man:?
analogy = manifold.analogy("king", "queen", "man")
# Returns candidates for "woman"
```

### Procedural World Generation

```python
from exception_theory import FractalReality

# Create deterministic world from seed
world = FractalReality("overworld", seed=42, octaves=3, persistence=0.5)

# Get elevation at any coordinate (always same for same seed)
height = world.get_elevation(1000.5, 2000.7)

# Render terrain chunk as ASCII
chunk = world.render_chunk_string(0, 0, size=20)
print(chunk)
# ~ ~ . . , , # # ^ ^
# ~ . . , , # # ^ ^ ^
# ...
```

### Time Travel (Event Sourcing)

```python
from exception_theory import TimeTraveler

tt = TimeTraveler("game_state")

# Record state changes
tt.commit("health", 100)
tt.commit("health", 75)
tt.commit("health", 50)

print(tt.state)  # {'health': 50}

# Travel back in time
tt.undo()
print(tt.state)  # {'health': 75}

tt.undo()
print(tt.state)  # {'health': 100}

# Travel forward
tt.redo()
print(tt.state)  # {'health': 75}
```

## Primitives

Work directly with ET's fundamental primitives:

```python
from exception_theory import Point, Descriptor, Traverser, bind_pdt

# Create primitives
p = Point(location="origin", state=0)
d = Descriptor(name="positive", constraint=lambda x: x > 0)
t = Traverser(identity="navigator")

# Bind them into an Exception
exception = bind_pdt(p, d, t)

# Check coherence
is_coherent = exception.is_coherent()  # False (0 is not > 0)

# Substantiate the point
p.substantiate(10)
is_coherent = exception.is_coherent()  # True (10 > 0)
```

## Constants

ET derives fundamental constants from first principles:

```python
from exception_theory import (
    BASE_VARIANCE,        # 1/12 - fundamental variance
    MANIFOLD_SYMMETRY,    # 12 - symmetry count
    KOIDE_RATIO,          # 2/3 - Koide formula constant
    DARK_ENERGY_RATIO,    # 0.683 - predicted cosmological ratio
    DARK_MATTER_RATIO,    # 0.268 - predicted cosmological ratio
)
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=exception_theory tests/

# Run specific test
pytest tests/test_batch1.py::test_traverser_entropy
```

## Documentation

Full documentation available at: [Add documentation URL]

## Contributing

Contributions welcome! Please ensure:
- All code derives from ET mathematics (P, D, T primitives)
- No placeholders or dummy implementations
- Comprehensive test coverage
- Documentation for new features

## License

MIT License - see LICENSE file

## Citation

If you use Exception Theory in research, please cite:

```
Exception Theory: A Complete Ontological Framework
M.J.M., 2024-2026
```

## Version History

- **v3.0.0** (2026): Library architecture with full modularization
- **v2.3** (5,799 lines): Batch 3 - Distributed Consciousness
- **v2.2** (4,313 lines): Batch 2 - Advanced Manifold Architectures  
- **v2.1** (3,119 lines): Batch 1 - Computational Exception Theory
- **v2.0** (2,586 lines): Core transmutation and calibration

## Contact

Exception Theory Mathematics: M.J.M.  
Implementation: ET Development Team

---

**"For every exception there is an exception, except the exception."**

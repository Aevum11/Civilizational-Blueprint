# Exception Theory - Quick Start Guide

## Installation (3 Steps)

### Step 1: Navigate to the library directory
```bash
cd /path/to/outputs/
```

### Step 2: Install the library
```bash
pip install -e .
```

That's it! The library is now installed and ready to use.

### Step 3 (Optional): Install development tools
```bash
pip install -e ".[dev]"
```

## Verify Installation

### Test the import:
```bash
python -c "from exception_theory import ETSovereign; print('SUCCESS!')"
```

### Run the examples:
```bash
python examples.py
```

### Run the test suite:
```bash
pytest exception_theory/tests/ -v
```

## Your First Program

Create a file called `my_first_et.py`:

```python
from exception_theory import ETSovereign, ETMathV2

# Initialize the engine
engine = ETSovereign()

# Calibrate (detect Python's memory geometry)
result = engine.calibrate()
print(f"Calibrated: {result['calibrated']}")
print(f"Platform: {result['platform']}")

# Generate true entropy from T-singularities
entropy = engine.generate_true_entropy(32)
print(f"True random: {entropy}")

# Use ET mathematics
density = ETMathV2.density(payload=64, container=100)
print(f"Structural density: {density}")

# Create a trinary logic state
state = engine.create_trinary_state(2, bias=0.8)  # 80% toward TRUE
print(f"Trinary state: {state}")
collapsed = state.collapse()
print(f"Collapsed to: {collapsed}")

print("\n✅ Exception Theory is working perfectly!")
```

Run it:
```bash
python my_first_et.py
```

## Common Import Patterns

### Import the main engine:
```python
from exception_theory import ETSovereign
engine = ETSovereign()
```

### Import specific classes:
```python
from exception_theory import (
    SwarmConsensus,
    SemanticManifold,
    TimeTraveler,
    FractalReality,
)
```

### Import mathematics:
```python
from exception_theory import ETMathV2

# Use any of the 40+ ET equations
density = ETMathV2.density(payload, container)
effort = ETMathV2.effort(observers, byte_delta)
transition = ETMathV2.phase_transition(gradient_input)
```

### Import primitives:
```python
from exception_theory import (
    Point,
    Descriptor,
    Traverser,
    bind_pdt,
)

# Create and bind primitives
p = Point(location="quantum", state=5)
d = Descriptor(name="positive", constraint=lambda x: x > 0)
t = Traverser(identity="observer")
exception = bind_pdt(p, d, t)
```

### Import everything:
```python
from exception_theory import *

# Now you have access to all classes, functions, and constants
```

## Quick Examples

### Example 1: True Entropy
```python
from exception_theory import TraverserEntropy

entropy = TraverserEntropy()
random_hex = entropy.substantiate(32)
random_bytes = entropy.substantiate_bytes(16)
random_int = entropy.substantiate_int(64)
random_float = entropy.substantiate_float()
```

### Example 2: Semantic Search
```python
from exception_theory import SemanticManifold

manifold = SemanticManifold("concepts")
manifold.bind_batch({
    "king": [0.9, 0.8, 0.1],
    "queen": [0.9, 0.9, 0.1],
    "man": [0.8, 0.2, 0.2],
})

results = manifold.search("king", top_k=2)
# Returns: [('queen', 0.998), ('man', 0.879)]
```

### Example 3: Procedural World
```python
from exception_theory import FractalReality

world = FractalReality("my_world", seed=42)
elevation = world.get_elevation(100, 200)
terrain = world.render_chunk_string(0, 0, size=10)
print(terrain)
```

### Example 4: Time Travel
```python
from exception_theory import TimeTraveler

tt = TimeTraveler()
tt.commit("health", 100)
tt.commit("health", 75)
tt.undo()  # Back to 100
tt.redo()  # Forward to 75
```

### Example 5: Swarm Consensus
```python
from exception_theory import SwarmConsensus

nodes = [SwarmConsensus(f"node_{i}", "data") for i in range(10)]
# Nodes automatically converge via variance minimization
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'exception_theory'"
- Make sure you ran `pip install -e .` from the correct directory
- Check that you're using the right Python environment

### "ImportError" when importing classes
- All classes are available from the main package
- Use: `from exception_theory import ClassName`
- Or: `from exception_theory.classes import ClassName`

### Tests failing
- Make sure all dependencies are installed
- Some tests require specific Python versions (3.7+)
- Check that you have write permissions for temp files

## Getting Help

1. **Read the documentation**: Check `README.md` and `LIBRARY_STRUCTURE.md`
2. **Run examples**: `python examples.py` shows all features
3. **Read the code**: All modules have comprehensive docstrings
4. **Check tests**: `exception_theory/tests/` shows usage patterns

## Next Steps

1. ✅ Install the library
2. ✅ Run examples.py
3. ✅ Run tests
4. 📚 Read README.md for comprehensive documentation
5. 💻 Start building with Exception Theory!

---

**"For every exception there is an exception, except the exception."**

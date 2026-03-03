# QUICK_REFERENCE.md - One-Page Cheat Sheet

## WHERE DOES CODE GO?

```
Math formula          → core/mathematics.py (ETMathV2 static method)
Constant value        → core/constants.py (SCREAMING_SNAKE)
Feature class         → classes/batchN.py (new or existing)
Integration wrapper   → engine/sovereign.py (create/get/direct)
Helper utility        → utils/calibration.py or utils/logging.py
```

## FILE SIZE LIMITS

```
constants.py:    < 300L  (currently 210L)
mathematics.py:  < 1200L (currently 908L)
batchN.py:       < 1000L (each batch)
sovereign.py:    < 2500L (currently 1879L)
```

## ADDING 10 ITEMS - CHECKLIST

```
☐ Parse: Identify types (math/class/const/integration)
☐ Constants: Add to constants.py if needed
☐ Math: Add @staticmethod to ETMathV2 if needed
☐ Classes: Create/update batchN.py
☐ Imports: Update classes/__init__.py
☐ Registries: Add self._X_registry = {} to sovereign.__init__
☐ Methods: Add create_X(), get_X() to sovereign.py
☐ Direct: Add direct_X() if math operation
☐ Cleanup: Add registry.clear() to sovereign.close()
☐ Docs: Update docstrings, version
```

## TEMPLATE: ETMathV2 Method

```python
@staticmethod
def operation(p, d):
    """Batch N, Eq X: Brief
    ET Math: formula"""
    # Derive from P, D, T
    return result
```

## TEMPLATE: Feature Class

```python
class Feature:
    """Batch N, Eq X: Brief
    ET Math: formula"""
    def __init__(self, param):
        self.data = param
    def operate(self):
        return ETMathV2.op(self.data)
```

## TEMPLATE: Integration (3 methods)

```python
def create_X(self, name, param):
    obj = XClass(param)
    self._x_registry[name] = obj
    return obj

def get_X(self, name):
    return self._x_registry.get(name)

def direct_X(self, param):  # Optional
    return ETMathV2.x_operation(param)
```

## CURRENT STATE

```
Batches: 1 (Eq 1-10), 2 (Eq 11-20), 3 (Eq 21-30)
Next: Batch 4 (Eq 31-40)
ETMathV2: 52 methods
ETSovereign: 101 methods
Classes: 26
Version: v2.3
```

## DEPENDENCY FLOW (NEVER BREAK)

```
constants → mathematics → primitives → batchN → sovereign → __init__
```

## FORBIDDEN

```
❌ Circular imports
❌ External algorithms (must derive from ET)
❌ Hardcoded values
❌ Missing ET Math docstrings
❌ Forgetting cleanup
❌ Placeholders
```

## VERSION INCREMENT

```
Add to existing batch: v2.X (patch)
New batch: v2.X+1 (minor)
Architecture change: v3.X (major)
```

## TOKEN ESTIMATE

```
Full codebase read: ~50,000 tokens
CLAUDE.md read: ~2,500 tokens
This card: ~500 tokens

Savings: 95% reduction
```

**Read CLAUDE.md for complete architecture.**
**Read CLAUDE_EXAMPLE.md for worked example.**
**Use this for quick lookups.**

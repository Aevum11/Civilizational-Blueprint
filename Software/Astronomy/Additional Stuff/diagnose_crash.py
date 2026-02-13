#!/usr/bin/env python3
"""
ET Scanner - Windows Crash Diagnostic Tool
==========================================
Run this FIRST to identify why Python is crashing.

Usage: python diagnose_crash.py
"""

import sys
import os

print("=" * 60)
print("ET Scanner - Windows Crash Diagnostic")
print("=" * 60)
print()

# Step 1: Basic Python info
print("[1] Python Information:")
print(f"    Version: {sys.version}")
print(f"    Platform: {sys.platform}")
print(f"    Executable: {sys.executable}")
print()

# Step 2: Check multiprocessing FIRST (critical for Windows)
print("[2] Testing multiprocessing.freeze_support()...")
try:
    import multiprocessing
    multiprocessing.freeze_support()
    print("    [OK] multiprocessing.freeze_support() succeeded")
except Exception as e:
    print(f"    [FAIL] multiprocessing error: {e}")
print()

# Step 3: Test signal module
print("[3] Testing signal module...")
try:
    import signal
    print("    [OK] signal module imported")
except Exception as e:
    print(f"    [FAIL] signal import error: {e}")
print()

# Step 4: Test numpy (most common crash source)
print("[4] Testing numpy import...")
print("    (This is where most Windows crashes occur)")
try:
    import numpy as np
    print(f"    [OK] numpy version: {np.__version__}")
    # Test basic operation
    arr = np.array([1, 2, 3])
    print(f"    [OK] numpy basic test passed: {arr}")
except ImportError as e:
    print(f"    [FAIL] numpy not installed: {e}")
except Exception as e:
    print(f"    [FAIL] numpy crash/error: {e}")
    print()
    print("    LIKELY CAUSE: DLL conflicts or corrupted numpy installation")
    print("    TRY: pip uninstall numpy && pip install numpy")
print()

# Step 5: Test scipy (second most common crash source)
print("[5] Testing scipy import...")
try:
    import scipy
    print(f"    [OK] scipy version: {scipy.__version__}")
except ImportError as e:
    print(f"    [FAIL] scipy not installed: {e}")
except Exception as e:
    print(f"    [FAIL] scipy crash/error: {e}")
    print()
    print("    LIKELY CAUSE: DLL conflicts or corrupted scipy installation")
    print("    TRY: pip uninstall scipy && pip install scipy")
print()

# Step 6: Test scipy.signal specifically
print("[6] Testing scipy.signal import...")
try:
    from scipy import signal as sp_signal
    print("    [OK] scipy.signal imported")
except ImportError as e:
    print(f"    [FAIL] scipy.signal not available: {e}")
except Exception as e:
    print(f"    [FAIL] scipy.signal crash/error: {e}")
print()

# Step 7: Test scipy.stats
print("[7] Testing scipy.stats import...")
try:
    from scipy.stats import entropy as sp_entropy
    print("    [OK] scipy.stats.entropy imported")
except ImportError as e:
    print(f"    [FAIL] scipy.stats not available: {e}")
except Exception as e:
    print(f"    [FAIL] scipy.stats crash/error: {e}")
print()

# Step 8: Test tkinter
print("[8] Testing tkinter...")
try:
    import tkinter as tk
    print("    [OK] tkinter available")
except ImportError:
    print("    [SKIP] tkinter not available (optional)")
except Exception as e:
    print(f"    [WARN] tkinter error: {e}")
print()

# Step 9: Test psutil
print("[9] Testing psutil...")
try:
    import psutil
    print(f"    [OK] psutil version: {psutil.__version__}")
except ImportError:
    print("    [SKIP] psutil not installed (optional)")
except Exception as e:
    print(f"    [WARN] psutil error: {e}")
print()

# Step 10: Environment info
print("[10] Environment Information:")
print(f"    PATH contains python: {'python' in os.environ.get('PATH', '').lower()}")
print(f"    PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
print(f"    PYTHONHOME: {os.environ.get('PYTHONHOME', '(not set)')}")
print()

# Summary
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print()
print("If you see [FAIL] for numpy or scipy above, try:")
print()
print("  1. Uninstall and reinstall:")
print("     pip uninstall numpy scipy")
print("     pip install numpy scipy")
print()
print("  2. Or use specific versions known to work:")
print("     pip install numpy==1.24.3 scipy==1.11.4")
print()
print("  3. Or try in a fresh virtual environment:")
print("     python -m venv fresh_env")
print("     fresh_env\\Scripts\\activate")
print("     pip install numpy scipy")
print()
print("If everything shows [OK] above, the issue may be")
print("in the scanner script itself.")
print()
input("Press Enter to exit...")

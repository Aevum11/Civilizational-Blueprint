# -*- mode: python ; coding: utf-8 -*-
"""
Exception Theory — CDF Compressor PyInstaller Build Specification
=================================================================

Build specification for packaging et_cdf_compressor.py into a standalone
single-file executable (.exe) with the bundled C pattern engine DLL.

ET Derivation (Three Tools):
  Identification Principle: This spec identifies the complete D-set required
  to build the executable — the Python source (P-substrate), the DLL binary
  (D-constraint on pattern engine availability), and PyInstaller's build
  engine (T-agency that traverses the spec to produce the .exe).

  Descriptor Gap Principle: The DLL is detected dynamically at build time.
  If present, it is bundled into the .exe. If absent, the .exe will attempt
  to auto-compile from the external .c file at runtime. The C pattern engine
  is REQUIRED for operation — the DLL should always be built before packaging.

  Subsumption Law: The spec subsumes ALL build configurations without remainder:
  DLL present → bundled; DLL absent → omitted (runtime auto-compile from .c).
  No manual intervention needed.

P ∘ D ∘ T = E
Author: Michael James Muller — Aevum_Defluo
"""

import os
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE

# ── Dynamic DLL Detection ──────────────────────────────────────────────
# Descriptor Gap Principle: the gap between "DLL exists on disk" and "DLL
# bundled into .exe" is closed by runtime detection. No static list —
# the DLL is discovered dynamically via os.path.isfile. If the DLL was
# compiled (by build.bat or CMake), it gets bundled. If not, the .exe
# will attempt to auto-compile from the external .c file at runtime.
# The C pattern engine is REQUIRED — always build the DLL before packaging.
_dll_name = 'et_pattern_engine.dll'
_binaries = [(_dll_name, '.')] if os.path.isfile(_dll_name) else []

a = Analysis(
    ['et_cdf_compressor.py'],
    pathex=[],
    binaries=_binaries,
    datas=[],
    # apsw is the Another Python SQLite Wrapper package required for
    # Tier 7 true random-access VFS mode. It is imported inside a
    # try/except guard (optional-availability pattern) so PyInstaller's
    # static analysis does NOT discover it automatically. Listing it
    # here as a hidden import forces inclusion in the packaged .exe.
    # Without this, a packaged build silently falls back to the
    # materialize-to-disk path when opening a compressed-only .cdf —
    # which breaks the learning Kolmogorov-complexity compressor at
    # scale per Mike's "only recompress when new stuff is added" rule.
    hiddenimports=['apsw'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='et_cdf_compressor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir='.',
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

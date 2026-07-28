# -*- mode: python ; coding: utf-8 -*-
"""
et32_bridge_helper.spec — ET32 Bridge PyInstaller Build Specification (32-bit Helper).

Derived from P ∘ D ∘ T = E.

P = et32_bridge_helper.py + the ET bridge modules it requires.
D = this spec — the 32-bit Python constraint is the Descriptor of 32-bit space.
T = 32-bit PyInstaller traversal.
E = ET32_Bridge_Helper.exe — a 32-bit companion that runs inside target processes.

Copyright (c) 2026 Michael James Muller (Aevum Defluo) — Exception Theory.
SPDX-License-Identifier: MIT
"""
#
# et32_bridge_helper.spec
# ET32 Bridge — PyInstaller Build Specification (32-bit Companion Helper)
#
# Derived from P ∘ D ∘ T = E.
#
# PDT of this build:
#   P = et32_bridge_helper.py + et_bridge modules
#   D = this spec (32-bit Python constraint = the Descriptor of 32-bit space)
#   T = 32-bit PyInstaller traversal
#   E = ET32_Bridge_Helper.exe — a 32-bit companion that runs inside targets
#
# Usage:
#   C:\Python32\Scripts\pyinstaller.exe et32_bridge_helper.spec
#   (Must use a 32-bit Python installation.)
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Michael James Muller (Aevum Defluo) — Exception Theory

import sys
import os
from pathlib import Path

# PyInstaller build-API — imported for static-analysis resolution.
# When PyInstaller executes this spec it injects these as globals;
# these imports are harmlessly shadowed at runtime.
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.api import PYZ, EXE, COLLECT

# ---------------------------------------------------------------------------
# Architecture gate — 32-bit Python required (uses sys)
# ---------------------------------------------------------------------------
assert sys.maxsize <= 2**32, (
    "ET32 Bridge Helper requires 32-bit Python. "
    "Current interpreter is 64-bit. Set PYTHON32 to a 32-bit interpreter."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# SPECPATH is injected by PyInstaller at runtime.  Compute equivalent from
# os.path so the spec is also valid under static analysis and direct execution.
SPECPATH = globals().get('SPECPATH', os.path.dirname(os.path.abspath(__file__)))
HERE     = Path(SPECPATH)
SRC_MAIN = str(HERE / "et32_bridge_helper.py")

a = Analysis(
    [SRC_MAIN],
    pathex=[str(HERE)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "et_math",
        "et_handle",
        "et_config",
        "et_logger",
        "et_ipc",
        "et_api",
        "et_errors",
        "win32api",
        "win32con",
        "pywintypes",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "numpy", "scipy", "matplotlib", "pandas",
        "tkinter", "unittest", "email", "html", "http",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ET32_Bridge_Helper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,              # must be x86 (32-bit Python)
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="ET32_Bridge_Helper",
)

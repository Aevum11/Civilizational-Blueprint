# -*- mode: python ; coding: utf-8 -*-
"""
et32_bridge.spec — ET32 Bridge PyInstaller Build Specification (64-bit Broker).

Derived from P ∘ D ∘ T = E.

P = the Python source tree (all et_bridge modules + entry point).
D = this spec — the Descriptor set that constrains the distribution.
T = PyInstaller — the Traverser that compiles P through D into E.
E = ET32_Bridge.exe — a fully self-contained 64-bit broker executable.

ET constants reflected in the build:
    S  = 12  → et_*.py modules dynamically discovered from flat directory.
    hd = 4096 → digital action quantum (DLL buffer unit).
    IPC_BUFFER_SIZE = 49152 → S × hd, runtime constant embedded via Analysis.

Copyright (c) 2026 Michael James Muller (Aevum Defluo) — Exception Theory.
SPDX-License-Identifier: MIT
"""
#
# et32_bridge.spec
# ET32 Bridge — PyInstaller Build Specification (64-bit Broker)
#
# Derived from P ∘ D ∘ T = E.
#
# PDT of this build:
#   P = the Python source tree (all et_bridge modules + entry point)
#   D = PyInstaller spec (the Descriptor set of the distribution)
#   T = PyInstaller itself (the Traverser that compiles P into a binary)
#   E = ET32_Bridge.exe — a fully self-contained 64-bit broker executable
#
# ET constants reflected in the build:
#   S = 12 → 12+ et_*.py modules collected from flat directory
#   IPC_BUFFER_SIZE = 49152 → runtime constant embedded via Analysis
#
# Usage:
#   pyinstaller et32_bridge.spec
#
# Prerequisites (64-bit Python ≥ 3.9):
#   pip install pyinstaller pystray pillow pywin32
#
# The output is placed in dist/ET32_Bridge/ (one-folder mode for easier
# distribution alongside et_bridge32.dll and config_template.json).
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
# Architecture gate — 64-bit Python required (uses sys)
# ---------------------------------------------------------------------------
assert sys.maxsize > 2**32, (
    "ET32 Bridge broker requires 64-bit Python. "
    "Current interpreter is 32-bit. Set PYTHON64 to a 64-bit interpreter."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# SPECPATH is injected by PyInstaller at runtime.  Compute equivalent from
# os.path so the spec is also valid under static analysis and direct execution.
SPECPATH     = globals().get('SPECPATH', os.path.dirname(os.path.abspath(__file__)))
HERE         = Path(SPECPATH)          # directory containing this .spec file
SRC_MAIN    = str(HERE / "et32_bridge_main.py")
SRC_DIR     = str(HERE)              # all modules live flat alongside the entry point
DIST_NAME   = "ET32_Bridge"

# ---------------------------------------------------------------------------
# Dynamic module discovery — ET Identification Principle:
# every et_*.py file in the project directory IS a bridge module.
# No static list — modules are identified by their Descriptor (et_ prefix).
# ---------------------------------------------------------------------------
_ET_ENTRY_POINTS = {"et32_bridge_main", "et32_bridge_helper"}
_ET_MODULES = sorted(
    p.stem for p in HERE.glob("et_*.py")
    if p.stem not in _ET_ENTRY_POINTS
)

# ---------------------------------------------------------------------------
# Dynamic data-file discovery — Descriptor Gap Principle:
# every distributable artifact in the project directory is discovered by its
# file-type Descriptor, not by a static name list.  Nothing is missed.
# ---------------------------------------------------------------------------
_DATAS = []
# Configuration template
for _cfg in sorted(HERE.glob("config_template*.json")):
    _DATAS.append((str(_cfg), "."))
# 32-bit DLLs (compiled separately by build.bat)
for _dll in sorted(HERE.glob("et_bridge32*.dll")):
    _DATAS.append((str(_dll), "."))

# UPX must not compress 32-bit DLLs — collect their names dynamically
_DLL_NAMES = sorted(p.name for p in HERE.glob("et_bridge32*.dll"))

# ---------------------------------------------------------------------------
# Analysis — collect all source modules
# ---------------------------------------------------------------------------
a = Analysis(
    [SRC_MAIN],
    pathex=[SRC_DIR],
    binaries=[],
    datas=_DATAS,
    hiddenimports=_ET_MODULES + [
        # Win32 extensions used by the bridge
        "win32api",
        "win32con",
        "win32process",
        "win32security",
        "pywintypes",
        # System tray (optional — present if installed)
        "pystray",
        "PIL",
        "PIL.Image",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not used by the bridge
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "tkinter",
        "unittest",
        "email",
        "html",
        "http",
        "xmlrpc",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# PYZ — compressed Python bytecode archive
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ---------------------------------------------------------------------------
# EXE — the broker executable (64-bit, Windows subsystem = console for log output)
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,         # one-folder mode: binaries stay in dist/
    name=DIST_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                  # console mode: log output visible in terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,              # inherits host arch (must be x64)
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version=None,
)

# ---------------------------------------------------------------------------
# COLLECT — assemble the one-folder distribution
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=_DLL_NAMES + [
        # Do not UPX-compress 32-bit DLLs — it breaks the PE structure.
        # _DLL_NAMES is dynamically populated from et_bridge32*.dll glob.
    ],
    name=DIST_NAME,
)

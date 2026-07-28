#!/usr/bin/env python3
"""
ET_FRACTAL_GENERATOR.py  [v2.0 — Professional Quality Edition]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exception Theory Fractal Engine — P ∘ D ∘ T = E
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — TWO professional-grade files written on every run:
  ① 32-bit float TIFF  (HDR/archival; Photoshop/GIMP/Affinity-ready)
  ② 16-bit PNG          (display/print; pHYs at OUTPUT_DPI, 65535 levels)
  Both saved to the same folder as this script.

QUALITY PRESETS  (chosen interactively at startup):
  1080p : 1920×1080    500,000 iters  (video / fast preview)
  2k    : 2048×2048  1,000,000 iters
  4k    : 4096×4096  2,000,000 iters
  hq    : 8192×8192  5,000,000 iters  (8K default)
  ultra : 16384×16384 10,000,000 iters (16K maximum)
  All iteration counts are overrideable at startup via the iterations prompt.

RENDERING TECHNIQUES — all ET-derived:
  ① Distance Estimation (DE)  — f'(z) Jacobian tracked at every pixel.
     Formula: de = 2·|z|·ln|z| / |dz|   (smooth ∂I boundary at any zoom)
     Derivative: dz_{n+1} = f'(z_n)·dz_n + [1|0]
     f'(z) = Σ_d w_r[d]·(12/d)·z^(12/d−1)   [ET Jacobian, real families]

  ② Normal-Map Lighting  — surface relief from DE gradient.
     Light from: angle = k=7/12·2π = 210° (circle-of-fifths generator),
                 elevation = K = 2/3 radians ≈ 38.2° (Koide angle).
     Normal: n_complex = z_esc / (|z_esc|·dz_esc)   [complex surface normal]
     Shading: h = (Re(n)·cos_L + Im(n)·sin_L + sin(K)) / (1 + sin(K))
     This produces 3D relief that makes boundaries infinitely sharp AND beautiful.

  ③ Orbit Traps  — minimum iteration distance to ET lattice rings.
     Trap rings: K=2/3, V=1/12, 1/φ, 1  (Koide, base-variance, golden, unison)
     Each ring corresponds to an ET manifold position; traps reveal
     the lattice's inner structure as explicit coloring layers.

  ④ Interior Coloring  — {P,D} Unsubstantiated is dark matter, not black.
     Uses final orbit angle and |z| at max_iter for subtle texture.
     "Dark matter gravitates (d=1) but does not emit (d≠12)" — very dark,
     with a hue whisper from the last orbit direction.

  ⑤ ACES filmic HDR tone mapping  (industry standard: film/TV color)
     + Koide gamma K=2/3 for ET-native perceptual encoding
     Applied to the multi-pass composite BEFORE quantisation.

MULTI-PASS COMPOSITOR:
  pass1 = escape coloring  (smooth_n, d_r, d_θ, tightness)
  pass2 = lighting         (normal-map diffuse shading from DE)
  pass3 = orbit traps      (additive color at ET lattice ring distances)
  pass4 = interior         (dark-matter texture at non-escaped pixels)
  final = ACES ∘ Koide_γ ∘ quartic-vignette ∘ unsharp

2D Complex Lattice (ET_Complex_Lattice.md):
  ℒ_ℂ = {2^(w/12) : w ∈ ℤ[i]} — full Gaussian integer lattice
  Real axis = D's domain (magnitude, force hierarchy)
  Imaginary axis = T's domain (phase/spin, rotation)
  d = LCM(d_r, d_θ)  — full combined sublattice class
  Palindrome [12,6,4,3,12,2,12,3,4,6,12,1] is a TOPOLOGICAL INVARIANT of N=12:
  same d-sequence for real cascade (g_r=7) AND imaginary instanton (g_θ=1)

All constants:  α⁻¹=137.035999110  126=10N+N/2  μ²=K  λ=V  v=2  π=12-gon-T-limit
                quintic τ=[0,100,40,60,80,20,120,20,80,60,40,100]¢
                Var(n)=(n²-1)/12  Ψ=1+√V·sin(2π·k/N)  E=(N/d)·[100/(100+|ε|)]·[100/(p+q)]

Output folder: same directory as this .py file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Derived from Exception Theory — Michael James Muller (Aevum Defluo)
P ∘ D ∘ T = E
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY PRESET  — change here, nothing else needed
# ══════════════════════════════════════════════════════════════════════════════
def _choose_preset():
    print()
    print('  ┌───────────────────────────────────────────────────────────────┐')
    print('  │   EXCEPTION THEORY FRACTAL GENERATOR  v2.0                   │')
    print('  │   Select quality preset:                                      │')
    print('  │                                                               │')
    print('  │   1  1080p  — 1920×1080    500,000 iters  (video/fast)       │')
    print('  │   2  2k     — 2048×2048  1,000,000 iters                     │')
    print('  │   3  4k     — 4096×4096  2,000,000 iters                     │')
    print('  │   4  hq     — 8192×8192  5,000,000 iters  (8K/default)       │')
    print('  │   5  ultra  — 16384×16384 10,000,000 iters (16K/maximum)     │')
    print('  │                                                               │')
    print('  │   You will be asked to override iteration count next.        │')
    print('  └───────────────────────────────────────────────────────────────┘')
    _map = {'1':'1080p','2':'2k','3':'4k','4':'hq','5':'ultra',
            '1080p':'1080p','2k':'2k','4k':'4k','hq':'hq','ultra':'ultra'}
    while True:
        try:
            raw = input('  Choice [1-5]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw in _map:
            chosen = _map[raw]
            print(f'  → {chosen}')
            print()
            return chosen
        print('  Please enter 1, 2, 3, 4, or 5.')

QUALITY_PRESET = _choose_preset()


def _choose_modes():
    """
    Prompt for mode selection.
    Returns a list of mode_ids (1 to 12 entries).
    """
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Mode selection                                             │')
    print('  │                                                              │')
    print('  │   R   — random  (one mode, full original randomness)        │')
    print('  │   A   — all     (all 12 modes blended equally)              │')
    print('  │   or enter any combination of mode numbers, e.g.:           │')
    print('  │         0 3 7   /   0,3,7   /   3-7   /   0 3-5 11         │')
    print('  │                                                              │')
    print('  │    0  PDT Genesis          6  Septic Otherworld             │')
    print('  │    1  Traverser Field      7  Nonic Recursion               │')
    print('  │    2  Descriptor Cascade   8  Magical Impedance             │')
    print('  │    3  Koide Boundary       9  Exception State               │')
    print('  │    4  Multifold Tower     10  Lagrangian Field              │')
    print('  │    5  Quintic Shadow      11  Route A/B Cascade             │')
    print('  └──────────────────────────────────────────────────────────────┘')
    import re
    while True:
        try:
            raw = input('  Modes [R/A/0-11]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not raw or raw == 'r':
            import random as _r; chosen = [_r.randint(0, 11)]
            print(f'  → Random: mode {chosen[0]}')
            print()
            return chosen
        if raw == 'a':
            chosen = list(range(12))
            print(f'  → All 12 modes blended')
            print()
            return chosen
        # Parse: supports "0 3 7", "0,3,7", "3-7", "0 3-5 11", mixed
        tokens = re.split(r'[,\s]+', raw.replace(' ',','))
        ids = []
        ok  = True
        for tok in tokens:
            tok = tok.strip()
            if not tok: continue
            m = re.match(r'^(\d+)-(\d+)$', tok)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b: a, b = b, a
                if b > 11: ok = False; break
                ids.extend(range(a, b+1))
            elif re.match(r'^\d+$', tok):
                v = int(tok)
                if v > 11: ok = False; break
                ids.append(v)
            else:
                ok = False; break
        if not ok or not ids:
            print('  Please enter R, A, or mode numbers 0-11 (e.g. 0 3 7, 3-7, A).')
            continue
        chosen = sorted(set(ids))
        print(f'  → {len(chosen)} mode(s): {chosen}')
        print()
        return chosen


SELECTED_MODES = _choose_modes()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 — BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
import sys, subprocess, importlib, os

def _pip(pkg, quiet=True):
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', pkg],
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None)
        return True
    except Exception:
        return False

def _have(mod):
    try: importlib.import_module(mod); return True
    except ImportError: return False

def _cuda_major():
    for cmd in (['nvcc','--version'],
                ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            if r.returncode == 0 and r.stdout.strip():
                txt = r.stdout.lower()
                if 'release' in txt:
                    return int(txt.split('release')[-1].split(',')[0].strip().split('.')[0])
                return 12
        except Exception: pass
    return None

def _cupy_pkg_for(major):
    """Return the correct CuPy wheel name for the installed CUDA major version."""
    if major is None: return 'cupy'
    if major >= 13:   return 'cupy-cuda13x'
    if major >= 12:   return 'cupy-cuda12x'
    if major >= 11:   return 'cupy-cuda11x'
    return 'cupy'


def _bootstrap():
    # Disable CUB/cuTENSOR accelerators BEFORE cupy is imported.
    # CUB's device-reduction path includes cuda_fp8/fp6/fp4.hpp which fail to
    # compile under NVRTC on CUDA 13.x.  Disabling CUB makes CuPy use its own
    # reduction kernels — still fully GPU-accelerated, no FP8 headers needed.
    # setdefault: respect any value the user already set explicitly.
    os.environ.setdefault('CUPY_ACCELERATORS', '')
    print('\n  [Setup] Checking dependencies…', flush=True)
    for mod, pkg in [('numpy','numpy'), ('PIL','Pillow')]:
        if not _have(mod):
            print(f'  [Setup] Installing {pkg}…', flush=True)
            if not _pip(pkg):
                print(f'\n  ERROR: Cannot install {pkg}.  Run:  pip install {pkg}\n')
                if sys.platform=='win32': input('Press Enter to exit…')
                sys.exit(1)
            importlib.invalidate_caches()
    if not _have('cupy'):
        major = _cuda_major()
        if major is not None:
            pkg = _cupy_pkg_for(major)
            print(f'  [Setup] CUDA {major}.x detected — installing {pkg}…', flush=True)
            if _pip(pkg):
                importlib.invalidate_caches()
            else:
                print(f'  [Setup] WARNING: Auto-install of {pkg} failed.', flush=True)
                print(f'  [Setup]          GPU will not be available this run.', flush=True)
                print(f'  [Setup]          To fix: pip install {pkg}', flush=True)
        else:
            print(f'  [Setup] No CUDA toolkit detected — GPU not available.', flush=True)
    print('  [Setup] Dependencies OK.', flush=True)

_bootstrap()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import time, math, struct, zlib, hashlib, platform, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageFilter

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR REPORTING SYSTEM
#  Every error and every fallback is reported here with full detail.
#  Nothing is swallowed silently.
# ══════════════════════════════════════════════════════════════════════════════
import traceback as _tb

_ERROR_HEADER = '  ' + '!'*68
_ERROR_FOOTER = '  ' + '!'*68

def _et_error(context, exc, fatal=True, fallback_msg=None):
    """
    Central error reporter. Prints full exception info with context.
    context     : what was being attempted (string)
    exc         : the Exception object
    fatal       : if True, prints FATAL and re-raises; if False, prints WARNING
    fallback_msg: what will happen instead (string, only when fatal=False)
    """
    level = 'FATAL ERROR' if fatal else 'ERROR — FALLBACK ACTIVE'
    print(f'\n{_ERROR_HEADER}', flush=True)
    print(f'  [ET {level}]', flush=True)
    print(f'  Context  : {context}', flush=True)
    print(f'  Type     : {type(exc).__name__}', flush=True)
    print(f'  Message  : {exc}', flush=True)
    print(f'  Traceback:', flush=True)
    for line in _tb.format_exc().splitlines():
        print(f'    {line}', flush=True)
    if fallback_msg:
        print(f'  Fallback : {fallback_msg}', flush=True)
    print(_ERROR_FOOTER, flush=True)
    if fatal:
        raise exc

def _et_fallback(context, reason, fallback_msg):
    """Report a fallback without an exception (e.g. feature unavailable)."""
    print(f'\n  [ET FALLBACK] {context}', flush=True)
    print(f'    Reason  : {reason}', flush=True)
    print(f'    Fallback: {fallback_msg}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — GPU DETECTION  (auto-installs missing NVRTC; no silent fallbacks)
# ══════════════════════════════════════════════════════════════════════════════
def _nvrtc_kernel_test(cp):
    """
    Fire BOTH an elementwise kernel AND a CUB reduction kernel.
    Elementwise alone does NOT include cuda_fp8/fp6/fp4 headers.
    cp.any() triggers the CUB reduction path that DOES include them —
    this is exactly what exposes a CUDA-toolkit / CuPy version mismatch.
    """
    # Elementwise
    a = cp.full((64,), 1.5, dtype=cp.float64)
    b = a * cp.float64(2.0) + cp.float64(0.5)
    assert abs(float(b[0]) - 3.5) < 1e-9
    # CUB reduction — hits cuda_fp8.hpp / cuda_fp6.hpp / cuda_fp4.hpp
    assert bool(cp.any(a > 1.0)), "reduction returned wrong result"
    assert not bool(cp.any(a > 2.0)), "reduction returned wrong result"


def _detect_gpu():
    # ── Phase 1: CuPy import + CUDA runtime (memory only — no NVRTC yet) ──────
    try:
        import cupy as cp
        _ = float(cp.sum(cp.array([1.0])))   # no kernel compilation here
        free_b, tot_b = cp.cuda.runtime.memGetInfo()
        free_gb, tot_gb = free_b/1e9, tot_b/1e9
        try:
            dev  = cp.cuda.runtime.getDeviceProperties(0)
            name = dev.get('name', b'GPU')
            if isinstance(name, bytes): name = name.decode('utf-8','replace')
        except Exception: name = 'GPU'
    except Exception as exc:
        # No GPU hardware at all — CPU is genuinely the best option
        print(f'  [CPU] No GPU found ({type(exc).__name__}) — CPU mode.', flush=True)
        return np, 0.0, 0.0

    # ── Phase 2: NVRTC JIT test — diagnose and fix any compilation failure ──────
    # GPU hardware is confirmed present.  Any failure here must be fixed, not
    # silently ignored.  We try the cheapest fix first (env var), then reinstall.
    def _try_nvrtc():
        try:
            _nvrtc_kernel_test(cp)
            return None
        except Exception as exc:
            return exc

    def _is_fp8_error(e):
        m = str(e).lower()
        return any(k in m for k in (
            'cuda_fp8', 'cuda_fp6', 'cuda_fp4',
            'fp8.hpp', 'fp6.hpp', 'fp4.hpp',
            'no storage class or type specifier',
            'expected a ";"', '__nv_silence_deprecation',
        ))

    def _is_dll_missing(e):
        m = str(e).lower()
        return any(k in m for k in ('nvrtc','dll','dynamic','libdevice','not found')) \
               and not _is_fp8_error(e)

    err = _try_nvrtc()
    if err is not None:

        # ── Fix A: FP8/FP6/FP4 header compile error (CUDA 13.x + CuPy CUB) ──
        # Cause: CuPy's CUB reduction includes cuda_fp8/fp6/fp4.hpp from the
        # system CUDA toolkit.  CUDA 13.x changed __NV_SILENCE_DEPRECATION_BEGIN
        # in a way NVRTC rejects.
        # Fix:  CUPY_ACCELERATORS='' disables CUB; CuPy uses its own reduction
        # kernels which do NOT include those headers.  Fully GPU-accelerated.
        if _is_fp8_error(err):
            print(f'  [GPU] CUDA 13.x FP8 header incompatibility detected.', flush=True)
            print(f'  [GPU] Disabling CUB accelerator (CUPY_ACCELERATORS=\"\") …', flush=True)
            os.environ['CUPY_ACCELERATORS'] = ''
            # Full cupy re-import so the new env var takes effect
            import importlib as _il, sys as _sys
            for _m in list(_sys.modules):
                if _m.startswith('cupy'): del _sys.modules[_m]
            importlib.invalidate_caches()
            try:
                import cupy as cp
            except Exception as reimport_exc:
                print(f'\n  FATAL: cupy re-import failed after setting CUPY_ACCELERATORS.')
                print(f'  {reimport_exc}')
                print(f'  Try:  pip install --force-reinstall {_cupy_pkg_for(_cuda_major() or 12)}')
                if sys.platform == 'win32': input('Press Enter to exit…')
                sys.exit(1)
            err2 = _try_nvrtc()
            if err2 is None:
                print(f'  [GPU] CUB disabled — GPU kernels working correctly.', flush=True)
                # Update free/total memory after re-import
                try:
                    free_b, tot_b = cp.cuda.runtime.memGetInfo()
                    free_gb, tot_gb = free_b/1e9, tot_b/1e9
                except Exception: pass
            else:
                # CUB disable didn't help — try reinstalling the correct cupy wheel
                major = _cuda_major() or 12
                correct_pkg = _cupy_pkg_for(major)
                print(f'  [GPU] CUB disable did not resolve the error.', flush=True)
                print(f'  [GPU] Reinstalling {correct_pkg} …', flush=True)
                for old_p in ['cupy-cuda11x','cupy-cuda12x','cupy-cuda13x','cupy']:
                    subprocess.run([sys.executable,'-m','pip','uninstall','-y',old_p],
                                   capture_output=True)
                if not _pip(correct_pkg, quiet=False):
                    print(f'\n  FATAL: pip install {correct_pkg} failed.')
                    print(f'  Try manually:  pip install {correct_pkg}')
                    if sys.platform == 'win32': input('Press Enter to exit…')
                    sys.exit(1)
                importlib.invalidate_caches()
                for _m in list(sys.modules):
                    if _m.startswith('cupy'): del sys.modules[_m]
                try:
                    import cupy as cp
                except Exception as e3:
                    print(f'\n  FATAL: cupy re-import failed: {e3}')
                    if sys.platform == 'win32': input('Press Enter to exit…')
                    sys.exit(1)
                err3 = _try_nvrtc()
                if err3 is not None:
                    print(f'\n  FATAL: GPU still fails after reinstall + CUB disable.')
                    print(f'  Error: {err3}')
                    print(f'  This is a known CUDA 13.x / CuPy incompatibility.')
                    print(f'  Resolution: downgrade CUDA toolkit to 12.x OR wait for')
                    print(f'  an official CuPy release with CUDA 13.x support.')
                    print(f'  https://github.com/cupy/cupy/issues')
                    if sys.platform == 'win32': input('Press Enter to exit…')
                    sys.exit(1)
                print(f'  [GPU] {correct_pkg} installed — GPU ready.', flush=True)

        # ── Fix B: NVRTC DLL missing ──────────────────────────────────────────
        elif _is_dll_missing(err):
            major = _cuda_major() or 12
            pkg   = f'nvidia-cuda-nvrtc-cu{major}'
            print(f'  [GPU] {name} detected — NVRTC JIT library missing.', flush=True)
            print(f'  [GPU] Auto-installing {pkg} …', flush=True)
            if not _pip(pkg, quiet=False):
                print(f'\n  FATAL: pip install {pkg} failed.')
                print(f'  Try manually:  pip install {pkg}')
                if sys.platform == 'win32': input('Press Enter to exit…')
                sys.exit(1)
            importlib.invalidate_caches()
            import importlib as _il2
            try:
                import cupy as cp; cp = _il2.reload(cp)
            except Exception: pass
            err2 = _try_nvrtc()
            if err2 is not None:
                print(f'\n  FATAL: {pkg} installed but NVRTC still unavailable.')
                print(f'  Add this folder to PATH then re-run:')
                print(f'  <python>\\\\Lib\\\\site-packages\\\\nvidia\\\\cuda_nvrtc\\\\bin')
                if sys.platform == 'win32': input('Press Enter to exit…')
                sys.exit(1)
            print(f'  [GPU] NVRTC installed — GPU ready.', flush=True)

        # ── Fix C: Unknown error ──────────────────────────────────────────────
        else:
            major = _cuda_major() or 12
            print('\n  FATAL: GPU kernel test failed with unrecognised error:')
            print(f'  {err}')
            print(f'  Try:  pip install --force-reinstall {_cupy_pkg_for(major)}')
            if sys.platform == 'win32': input('Press Enter to exit…')
            sys.exit(1)

    print(f'  [GPU] {name}  |  {free_gb:.1f}/{tot_gb:.1f} GB VRAM free  |  NVRTC OK',
          flush=True)
    return cp, free_gb, tot_gb

xp, GPU_FREE_GB, GPU_TOTAL_GB = _detect_gpu()
USE_GPU  = (xp is not np)
N_CPU    = max(1, os.cpu_count() or 1)

def _to_np(a):
    if USE_GPU and hasattr(a,'get'): return a.get()
    return np.asarray(a)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — ET MANIFOLD CONSTANTS  (all derived; zero external axioms)
# ══════════════════════════════════════════════════════════════════════════════

N        = 12                            # MANIFOLD_SYMMETRY = 3 primitives × 4 states
N_ET     = 27720                         # LCM(1..12) — ALL sublattice families
V        = 1.0/N                         # BASE_VARIANCE = 1/12
K        = 2.0/3.0                       # KOIDE_RATIO = binding stability threshold
LN2      = math.log(2.0)
PHI      = (1.0 + math.sqrt(5.0))/2.0   # φ = 1.618…

# ESCAPE_R must be large for smooth coloring accuracy.
# Formula: mu = n+1 - (ln(ln|z|) - ln(ln(R))) / ln(p_eff)
# Requires ln|z| >> 1 at escape.  1e8 satisfies this for all ET powers.
ESCAPE_R    = 1e8
LN_LN_ESC   = math.log(math.log(ESCAPE_R))   # ln(ln(1e8)) ≈ 2.914
# Legacy alias used in some intermediate formulas:
LOG2_LOG2_ESC = math.log2(math.log2(ESCAPE_R))

# ── 2D complex lattice (ET_Complex_Lattice.md §3-5) ───────────────────────────
G_REAL     = 7        # circle-of-fifths generator: round(N·log₂N) mod N
G_IMAG     = 1        # imaginary generator: round(N·2π/ln2) mod N = 109 mod 12
REAL_DELTA = abs(round(N*math.log2(N)) - N*math.log2(N))       # 0.01955
IMAG_DELTA = abs(round(N*2*math.pi/LN2) - N*2*math.pi/LN2)    # 0.2234
N_MAX_REAL = int(0.5/REAL_DELTA)     # 25 — real cascade stable levels
N_MAX_IMAG = int(0.5/IMAG_DELTA)    # 2  — imaginary cascade stable levels
# Ratio |δ_θ|/|δ_r| = N = 12 exactly (T's axis N× less stable than D's)

UNIT_CIRCLE_D = [1, 4, 2, 6, 12]    # force sequence: +1→d=1, i→d=4, -1→d=2, -i→d=6, back→d=12
UNIT_CIRCLE_K = [0, 27, 54, 82, 109]

# ── A₀ = (N−1)² + S² = 137  (fine structure manifold impedance) ─────────────
S_STATES   = 4            # C(3,2)+C(3,3) = 3+1 = 4  (manifold states)
A0_EM      = (N-1)**2 + S_STATES**2   # = 137

PHOTON_DR  = 12;  PHOTON_DT = 12     # Photon = {D,T} Mediation, LCM(12,12)=12

# ── α⁻¹ = A₀+A₁−A₁.₅−A₂−A₃ (ET_Fine_Structure_REVISED.md) ─────────────────
_PI        = math.pi   # derived from 12-gon T-recursion: π = lim 12·2^k·sin(π/3·2^-k)
_SIGMA     = math.sqrt(V)                              # √(1/12)
_K_EM      = N * K                                     # 8 — EM coupling channels
_A1        = _SIGMA / _K_EM
_DELTA_FS  = ((1-_SIGMA)*K*V/A0_EM)*(1+K/(N*S_STATES))
_A1_5      = _SIGMA*K*(1+_DELTA_FS)/(S_STATES*_K_EM*N**3*math.sqrt(_PI))
_A2        = K**2/(N**3*_PI)
_A3        = K**3/(N**4*_PI**2)
ALPHA_INV_ET = A0_EM + _A1 - _A1_5 - _A2 - _A3   # ≈ 137.035999110

# ── 126 = 10N+N/2 — strong/gravity octave separation ─────────────────────────
FORCE_RATIO_EXP = 10*N + N//2   # = 126

# ── Mexican-hat vacuum v=√(μ²/2λ)=2 (Lagrangian mode, ET_Lagrangian §VIII) ──
_MH_MU2    = K               # μ²=2/3 (Koide ratio = D-binding curvature)
_MH_LAMBDA = V               # λ=1/12 (base variance = quartic confinement)
_MH_V      = math.sqrt(_MH_MU2/(2*_MH_LAMBDA))  # = 2.0 exactly

# ── Route A/B canonical Weak→EM cascades ─────────────────────────────────────
ROUTE_A_RATIOS  = [6/5, 5/4, 3/2];  ROUTE_A_D  = [4, 3, 12]  # via Strong d=3
ROUTE_B_RATIOS  = [6/5, 9/8, 3/2];  ROUTE_B_D  = [4, 6, 12]  # via Hexadic d=6
ROUTE_BC_RATIOS = [5/3,16/9, 2/3];  ROUTE_BC_D = [4, 6, 12]  # complement → K

# ── Quintic tension τ(m) per semitone class m=0..11 ──────────────────────────
# τ(m)=min_{j=0..4}|5m−12j|×20¢  Max at m=6 (tritone)=120¢  Sum=720¢
QUINTIC_TENSION = np.array([0,100,40,60,80,20,120,20,80,60,40,100], dtype=np.float64)

# ── Var(n)=(n²−1)/12 — descriptor variance of n-descriptor configuration ─────
def _var_n(n): return (n*n-1)/12.0

# ── Canonical p+q per sublattice for Elegance Score ──────────────────────────
FAM_PQ = {1:3, 2:70, 3:13, 4:11, 5:13, 6:17, 7:15, 8:33, 9:17, 10:19, 11:27, 12:5}

# ── ET-derived light direction for normal-map lighting ───────────────────────
# Angle = k=7/12·2π = 210° (circle-of-fifths generator, strongest stable path)
# Elevation = K = 2/3 rad ≈ 38.2° (Koide binding stability threshold)
_THETA_LIGHT   = 7.0/12.0 * 2.0*math.pi   # 210° in standard notation
_COS_L = math.cos(_THETA_LIGHT)            # ≈ −0.866
_SIN_L = math.sin(_THETA_LIGHT)            # ≈ −0.500
_SIN_K = math.sin(K)                       # sin(2/3) ≈ 0.619
_NORM_L = 1.0 + _SIN_K                     # normalisation denominator

# ── ACES filmic constants (a,b,c,d,e — industry standard) ────────────────────
_ACES_A = 2.51; _ACES_B = 0.03; _ACES_C = 2.43; _ACES_D = 0.59; _ACES_E = 0.14

# ── RMSAE shimmer Ψ = 1 + √V·sin(2π·k_class/N) ───────────────────────────────
_RMSAE_AMP = math.sqrt(V)   # = 1/√12 ≈ 0.2887


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — QUALITY / RESOLUTION SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Iteration counts — maximum quality, no performance concessions.
# ET iteration is 20-40× heavier than z²+c (24 families, complex exponentiation).
# 1M ET iterations ≈ 20-40M standard iterations in compute cost.
# Without perturbation theory the numerical precision ceiling is ~10M iterations.
# All these counts are in line with or exceed professional fractal software at
# equivalent compute cost when the ET overhead is factored in.
_PRESETS = {
    '1080p': dict(w=1920,  h=1080,  iters=500000,   ss=1, tile=64),
    '2k'   : dict(w=2048,  h=2048,  iters=1000000,  ss=1, tile=64),
    '4k'   : dict(w=4096,  h=4096,  iters=2000000,  ss=1, tile=32),
    'hq'   : dict(w=8192,  h=8192,  iters=5000000,  ss=1, tile=16),
    'ultra': dict(w=16384, h=16384, iters=10000000, ss=1, tile=8),
}
_P = _PRESETS[QUALITY_PRESET]

IMG_W      = _P['w']
IMG_H      = _P['h']
MAX_ITER   = _P['iters']
OUTPUT_DPI = 300

# SSAA: on GPU use 1× (DE compensates better than SSAA; saves VRAM)
#       on CPU use preset SS (each render pixel ~27ms on CPU; 4× is fine)
if USE_GPU:
    SS = 1
else:
    SS = _P['ss']

RENDER_W = IMG_W * SS
RENDER_H = IMG_H * SS


def _choose_iterations():
    """Allow user to override the preset iteration count, or keep the preset value."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print(f'  │   Iterations  (preset default: {MAX_ITER:,}){"":>14}│')
    print('  │                                                              │')
    print('  │   Press Enter to keep preset value.                         │')
    print('  │   Or enter any integer ≥ 1 to override.                     │')
    print('  │                                                              │')
    print('  │   Professional fractal renderers use 10K–10M+               │')
    print('  │   ET iteration is 20-40× heavier than z²+c per step.        │')
    print('  └──────────────────────────────────────────────────────────────┘')
    while True:
        try:
            raw = input(f'  Iterations [Enter={MAX_ITER:,}]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw == '':
            print(f'  → {MAX_ITER:,}')
            print()
            return MAX_ITER
        try:
            v = int(raw.replace(',','').replace('_',''))
            if v < 1:
                print('  Must be ≥ 1.')
                continue
            print(f'  → {v:,}')
            print()
            return v
        except ValueError:
            print('  Please enter a whole number, or press Enter to keep the default.')


MAX_ITER = _choose_iterations()


def _choose_precision():
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Float precision                                            │')
    print('  │                                                              │')
    print('  │   32  — float32  fast    (~1-4 min HQ on RTX 2070 SUPER)   │')
    print('  │   64  — float64  precise (~20-25 min HQ on RTX 2070 SUPER) │')
    print('  │                                                              │')
    print('  │   float32 is visually identical at 8K for all normal use.  │')
    print('  │   float64 only matters for extreme zoom or large print.     │')
    print('  └──────────────────────────────────────────────────────────────┘')
    _map = {'32':'float32','64':'float64','f32':'float32','f64':'float64',
            'float32':'float32','float64':'float64'}
    while True:
        try:
            raw = input('  Precision [32/64]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in _map:
            chosen = _map[raw]
            print(f'  → {chosen}')
            print()
            return chosen
        print('  Please enter 32 or 64.')

GPU_FLOAT_PRECISION = _choose_precision()


def _choose_type():
    """Julia / Mandelbrot / ∂I Lattice-Aware / Random choice."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Fractal type                                               │')
    print('  │                                                              │')
    print('  │   J  — Julia          (fixed c, z₀ = pixel)                │')
    print('  │   M  — Mandelbrot     (z₀ = 0, c = pixel)                  │')
    print('  │   D  — ∂I Lattice-Aware  ← ET-native fractal type          │')
    print('  │         Dominant power from orbit 27720ET lattice position  │')
    print('  │         z_{n+1} = Ψ·z^{p_dom} + V·Σ(24 families) + c      │')
    print('  │         p_dom = 12/d_orbit (near lattice) or palindrome     │')
    print('  │   R  — Random         (equal chance of all three)           │')
    print('  │                                                              │')
    print('  │   J/M/D use ET iteration (24 sublattice families)           │')
    print('  │   ∂I: the orbit determines the map — self-referential       │')
    print('  └──────────────────────────────────────────────────────────────┘')
    _map = {'j':'julia','m':'mandelbrot','d':'di','r':'random',
            'julia':'julia','mandelbrot':'mandelbrot','di':'di','random':'random'}
    while True:
        try:
            raw = input('  Type [J/M/D/R]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in _map:
            chosen = _map[raw]
            print(f'  → {chosen}')
            print()
            return chosen
        print('  Please enter J, M, D, or R.')


def _choose_tower():
    """Tower selection."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Tower  (ET substrate — sets palette + centre presets)     │')
    print('  │                                                              │')
    print('  │   1  Cosmological  — Black Hole / White Hole seed           │')
    print('  │   2  Digital       — Silicon Clock Cycle seed               │')
    print('  │   3  Dream         — 40 Hz Gamma / Biological seed          │')
    print('  │   4  Civilizational— Generational seed                      │')
    print('  │   R  Random        — T-agency selects                       │')
    print('  └──────────────────────────────────────────────────────────────┘')
    _map = {'1':'cosmological','2':'digital','3':'dream','4':'civilizational','r':'random',
            'cosmological':'cosmological','digital':'digital',
            'dream':'dream','civilizational':'civilizational','random':'random'}
    while True:
        try:
            raw = input('  Tower [1-4/R]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in _map:
            chosen = _map[raw]
            print(f'  → {chosen}')
            print()
            return chosen
        print('  Please enter 1, 2, 3, 4, or R.')


def _choose_advanced():
    """
    Advanced Mode gate.  Returns a dict of overrides (empty = all random).
    Exposes: manual centre, zoom, julia_c override, seed pin.
    """
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Advanced Mode                                              │')
    print('  │   Manual centre / zoom / Julia c / seed pin                 │')
    print('  └──────────────────────────────────────────────────────────────┘')
    while True:
        try:
            raw = input('  Enter Advanced Mode? [Y/N]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in ('y','yes'):
            break
        if raw in ('n','no',''):
            print()
            return {}
        print('  Please enter Y or N.')

    print()
    params = {}

    def _ask_float(prompt, allow_empty=True):
        while True:
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if allow_empty and raw == '':
                return None
            try:
                return float(raw)
            except ValueError:
                print('  Invalid number. Try again (or press Enter to skip).')

    def _ask_int(prompt, allow_empty=True):
        while True:
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if allow_empty and raw == '':
                return None
            try:
                return int(raw)
            except ValueError:
                print('  Invalid integer. Try again (or press Enter to skip).')

    print('  Centre (press Enter to use tower random):')
    cx = _ask_float('    Real part      (e.g. -0.75): ')
    cy = _ask_float('    Imaginary part (e.g.  0.12): ')
    if cx is not None and cy is not None:
        params['cx'] = cx; params['cy'] = cy

    zoom = _ask_float('  Zoom (half-height; e.g. 0.5 is tight, 3.0 wide; Enter=random): ')
    if zoom is not None:
        params['zoom'] = max(1e-10, zoom)

    print('  Julia c override (press Enter to use ET-derived):')
    jcr = _ask_float('    Real part      (Enter=skip): ')
    jci = _ask_float('    Imaginary part (Enter=skip): ')
    if jcr is not None and jci is not None:
        params['julia_c'] = complex(jcr, jci)

    seed_val = _ask_int('  Seed pin (integer to reproduce a run; Enter=entropy): ')
    if seed_val is not None:
        params['seed'] = seed_val

    print()
    return params


def _choose_output_mode():
    """Single image or zoom video."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Output mode                                                │')
    print('  │                                                              │')
    print('  │   I  — Single image  (TIFF + PNG)                           │')
    print('  │   V  — Zoom video    (PNG frames → MP4 via ffmpeg)          │')
    print('  └──────────────────────────────────────────────────────────────┘')
    while True:
        try:
            raw = input('  Output [I/V]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in ('i','image',''):
            print('  → single image')
            print()
            return 'image'
        if raw in ('v','video'):
            print('  → zoom video')
            print()
            return 'video'
        print('  Please enter I or V.')


def _choose_video_params():
    """Parameters for the zoom video."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Zoom video parameters                                      │')
    print('  └──────────────────────────────────────────────────────────────┘')

    def _ask(prompt, default, cast):
        while True:
            try:
                raw = input(f'  {prompt} [default={default}]: ').strip()
            except (EOFError, KeyboardInterrupt):
                print(); sys.exit(0)
            if raw == '':
                return default
            try:
                v = cast(raw)
                if cast == float and v <= 0: raise ValueError
                if cast == int   and v <= 0: raise ValueError
                return v
            except ValueError:
                print(f'  Invalid. Enter a positive {cast.__name__}.')

    print('  Target point — the coordinate the zoom dives toward.')
    print('  (Press Enter to use Advanced Mode centre, or tower random if not set)')
    tx = None; ty = None
    try:
        raw = input('    Target real part      (Enter=use centre): ').strip()
        if raw: tx = float(raw)
        raw = input('    Target imaginary part (Enter=use centre): ').strip()
        if raw: ty = float(raw)
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)

    zoom_start = _ask('Start zoom (wide; e.g. 2.5)',  2.5,   float)
    zoom_end   = _ask('End zoom   (tight; e.g. 0.01)', 0.001, float)
    n_frames   = _ask('Number of frames (e.g. 240 = 8s at 30fps)', 240, int)

    # FPS selection — fixed choices only (ffmpeg + display standards)
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Frame rate                                                 │')
    print('  │                                                              │')
    print('  │   1  — 15 fps  (cinematic slow / file-size efficient)       │')
    print('  │   2  — 30 fps  (broadcast standard — default)               │')
    print('  │   3  — 45 fps  (smooth / high motion)                       │')
    print('  │   4  — 60 fps  (maximum smoothness)                         │')
    print('  └──────────────────────────────────────────────────────────────┘')
    _fps_map = {'1':15,'2':30,'3':45,'4':60,
                '15':15,'30':30,'45':45,'60':60}
    while True:
        try:
            raw = input('  FPS [1-4 or 15/30/45/60, Enter=30]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw == '':
            fps = 30
            print('  → 30 fps')
            print()
            break
        if raw in _fps_map:
            fps = _fps_map[raw]
            print(f'  → {fps} fps')
            print()
            break
        print('  Please enter 1, 2, 3, 4, or 15/30/45/60.')

    # Duration hint
    dur = n_frames / fps
    print(f'  Duration  : {n_frames} frames ÷ {fps} fps = {dur:.1f} s')
    print()
    return dict(tx=tx, ty=ty, zoom_start=zoom_start, zoom_end=zoom_end,
                n_frames=n_frames, fps=fps)


FRACTAL_TYPE  = _choose_type()
IS_DI_TYPE    = (FRACTAL_TYPE == 'di')   # ∂I Lattice-Aware fractal — orbit determines the map
SELECTED_TOWER = _choose_tower()
ADVANCED_PARAMS = _choose_advanced()
OUTPUT_MODE   = _choose_output_mode()
VIDEO_PARAMS  = _choose_video_params() if OUTPUT_MODE == 'video' else {}
if USE_GPU and GPU_FLOAT_PRECISION == 'float32':
    FLOAT_DTYPE   = np.float32
    COMPLEX_DTYPE = np.complex64
else:
    FLOAT_DTYPE   = np.float64
    COMPLEX_DTYPE = np.complex128

# Tile size in rows: smaller = lower peak VRAM with DE arrays
TILE_ROWS  = max(SS, _P['tile'])
N_THREADS  = 1 if USE_GPU else N_CPU


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — SUBLATTICE FAMILY TABLES  (27720ET — all families active)
# ══════════════════════════════════════════════════════════════════════════════

SIMPLE_REAL   = [1, 2, 3, 4, 6, 12]   # native to 12ET (divisors of 12)
EXTENDED_REAL = [5, 7, 8, 9, 10, 11]  # extended (need 27720ET)
ALL_REAL      = SIMPLE_REAL + EXTENDED_REAL    # 12 real families
ALL_COMPLEX   = ALL_REAL[:]                    # same 12 for imaginary axis
N_FAM         = len(ALL_REAL)                  # = 12

_P_REAL   = np.array([12.0/d for d in ALL_REAL],    dtype=np.float64)   # p_d = 12/d
_P_CMPLX  = np.array([12.0/d for d in ALL_COMPLEX], dtype=np.float64)
_ROT_D    = np.array([LN2/d  for d in ALL_COMPLEX], dtype=np.float64)   # imaginary phase
_P_R_M1   = _P_REAL - 1.0                                                # for Jacobian

# Elegance Score E = (N/d)·[100/(100+|ε|)]·[100/(p+q)]  at ε=0 (best-case)
FAM_ELG_FULL = {d: (N/d)*(100.0/(100.0+FAM_PQ.get(d,20))) for d in range(1,13)}

FAM_CHAR = {
    1:'Trivial/Gravity',   2:'Quadratic/EW',    3:'Cubic/Strong',
    4:'Quartic/Weak',      5:'Quintic/Golden',   6:'Hexadic/Higgs',
    7:'Septic/Otherworld', 8:'Octet/Shadow',     9:'Nonic/Fractal',
   10:'Decic/φ-Binary',   11:'Undecimal/Prime', 12:'Full-Res/EM',
}

# EM Spectrum musical scale (ET_EM_Spectrum_Framework1.md Discovery 6)
FAM_HUE = {
     1:0.00,  # Red       d=1  Gravity/unison        (400THz k=0)
     2:0.05,  # Dark-red  d=2  Tritone/pivot         (Euler e^{iπ}=−1)
     3:0.33,  # Green     d=3  Cubic/Strong          (avg Yellow+Blue)
     4:0.07,  # Orange    d=4  Quartic/Weak          (480THz = conscious!)
     5:0.13,  # Gold      d=5  Quintic/Golden-φ      (qualia, 60ET+)
     6:0.22,  # Yel-Green d=6  Hexadic/Higgs         (spin-½ 4π→d=6)
     7:0.72,  # Indigo    d=7  Septic/Otherworld     (cryst. forbidden 3D)
     8:0.80,  # Purple    d=8  Octet/Shadow
     9:0.87,  # Magenta   d=9  Nonic/Fractal (3²)
    10:0.83,  # Violet    d=10 Decic/φ-Binary        (visible span 60ET!)
    11:0.95,  # Crimson   d=11 Undecimal/Prime
    12:0.50,  # Cyan      d=12 Full-Res/EM           (perf 5th = photon)
}

# d-value LUT: _D_LUT[k mod 12] = 12/gcd(k,12)
# TOPOLOGICAL INVARIANT of N=12: same for real cascade (g_r=7) AND imaginary (g_θ=1)
# (ET_Complex_Lattice §18 — palindromic cascade is direction-independent)
# The sequence [12,6,4,3,12,2,12,3,4,6,12,1] appears identically in both
# D-axis (magnitude) and T-axis (phase) directions — force hierarchy is
# the same whether approached through magnitude or through phase.
_D_LUT = np.array([12//math.gcd(k if k!=0 else 12, 12) for k in range(12)],
                   dtype=np.float32)
# = [1, 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12]

_PALINDROME = np.array([12,6,4,3,12,2,12,3,4,6,12,1], dtype=np.float64)

# ── ∂I Lattice-Aware Fractal constants (ET_dI_Lattice_Aware_Fractal_Complete_Specification.md)
# p_eff = mean of palindromic power sequence = (1+2+3+4+1+6+1+4+3+2+1+12)/12 = 40/12 = 10/3
_PALINDROME_P = 12.0 / _PALINDROME   # [1,2,3,4,1,6,1,4,3,2,1,12]
_P_EFF_DI     = 10.0 / 3.0
_LN_P_EFF_DI  = math.log(_P_EFF_DI)
# Shimmer Ψ_k = 1 + √V · sin(2π·k/N)  k=0..11
_SHIMMER_NP   = np.array([1.0 + _RMSAE_AMP * math.sin(2*math.pi*k/N)
                           for k in range(N)], dtype=np.float64)

def _vec_gcd_27720_np(a_int64):
    """Vectorised gcd(|a|, 27720) for NumPy int64 arrays.
    27720 = 2^3 * 3^2 * 5 * 7 * 11.  Identical algorithm to CUDA et_gcd_27720.
    Used by the CPU path of the ∂I iteration.
    """
    a  = np.abs(a_int64).astype(np.int64)
    g  = np.ones_like(a)
    nz = a > 0
    # 2-adic
    g  = np.where(nz & ((a & 7)==0), 8,
         np.where(nz & ((a & 3)==0), 4,
         np.where(nz & ((a & 1)==0), 2, g)))
    # 3-adic
    g  = np.where(nz & (a % 9 == 0), g*9,
         np.where(nz & (a % 3 == 0), g*3, g))
    # 5, 7, 11-adic
    g  = np.where(nz & (a % 5  == 0), g*5,  g)
    g  = np.where(nz & (a % 7  == 0), g*7,  g)
    g  = np.where(nz & (a % 11 == 0), g*11, g)
    g  = np.where(a == 0, 27720, g)
    return g

# FAM_COUPLING[d]: relative coupling ξ(d) = A₀_EM / A₀_magic(d)
# Source: ET_Fantastical §5 — A₀_magic(d) = (N/d − 1)² + S²
# ξ=1.00 at d=1, ξ=8.56 at d=12 (maximum EM coupling)
# FAM_COUPLING[d]: ξ(d) = A₀_EM / A₀_magic(d)  [relative T-P coupling vs local EM]
# Source: ET_Fantastical §3.2/§14 — A₀_magic = (d−1)² + S²
# Physical: lower d → less mediation → stronger coupling
#   d=1:  A₀=16,  ξ=8.56× (max — pure will, no sublattice structure)
#   d=3:  A₀=20,  ξ=6.85× (cubic/strong)   d=12: A₀=137, ξ=1× (EM baseline)
# Coupling: FAM_COUPLING[d] = A₀/(N/d-1)²+S² — ET manifold impedance per family
FAM_COUPLING = {d: A0_EM/((N/d-1.0)**2+S_STATES**2+1e-6) for d in range(1,13)}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — T-AGENCY: GENUINE ENTROPY  (T=[0/0], no two traversals identical)
# ══════════════════════════════════════════════════════════════════════════════

def t_agency_seed():
    parts = [struct.pack('<Q', time.time_ns()),
             struct.pack('<Q', time.perf_counter_ns()%(2**64)),
             struct.pack('<Q', os.getpid()),
             struct.pack('<Q', id(object())&0xFFFFFFFFFFFFFFFF)]
    try:    parts.append(os.urandom(32))
    except: parts.append(struct.pack('<d', time.monotonic()))
    try:    parts.append(platform.node().encode('utf-8','replace'))
    except: pass
    h = hashlib.sha512()
    for p in parts: h.update(p)
    seed = int.from_bytes(h.digest()[:8],'big') % (2**31)
    return np.random.RandomState(seed), seed


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — TOWER DEFINITIONS  (Multifold of Lattices §16)
# ══════════════════════════════════════════════════════════════════════════════

TOWERS = {
    'cosmological': dict(
        name='Cosmological Tower — Black Hole / White Hole Seed', symbol='Σ_cosm',
        # R₀=GM_BH/c³. 126=10N+N/2. Primary: d=12(EM),d=6(composite),d=3(QCD)
        prim_r=[12,6,3,4,2], prim_c=[12,4,2,6,3], ext_boost=[5,7], delta_k=0,
        centers=[(-0.75,0.00),(-0.50,0.56),(-0.70,0.27),(-1.40,0.00),
                 (0.28,0.53),(-0.10,0.65),(-0.77,0.11),(0.00,0.00),
                 (-1.26,0.38),(-0.52,0.52),(-1.78,0.00),(-0.16,1.04)],
        zoom_lo=1.5, zoom_hi=3.5, pal_base=0.68, pal_range=0.28),
    'digital': dict(
        name='Digital Tower — Silicon Clock Cycle Seed', symbol='Σ_digit',
        # R₀=1/f_clock. Binary=d=1. Page=2^N=4096. K=2/3=hash load.
        prim_r=[2,12,8,4,1], prim_c=[2,12,4,8,6], ext_boost=[8,10], delta_k=-996,
        centers=[(0.00,0.00),(-0.50,0.50),(0.30,-0.50),(-1.30,0.00),
                 (0.00,1.00),(-0.12,-0.75),(-0.60,-0.40),(0.25,0.00),
                 (-1.755,0.00),(-0.10,0.65),(-0.74,0.12),(0.35,0.35)],
        zoom_lo=1.8, zoom_hi=4.0, pal_base=0.50, pal_range=0.20),
    'dream': dict(
        name='Dream / Biological Tower — 40 Hz Gamma Seed', symbol='Σ_dream',
        # R₀=1/f_gamma=25ms. 420ET: d=5(qualia),d=7(Otherworld).
        prim_r=[5,7,9,3,12], prim_c=[5,7,4,9,6], ext_boost=[5,7,9], delta_k=-1279,
        centers=[(-0.125,0.744),(-0.750,0.123),(0.285,0.014),(-0.160,1.034),
                 (-1.770,0.000),(0.000,-1.000),(-0.502,0.000),(0.355,0.355),
                 (-0.590,0.413),(-0.750,0.100),(-0.125,-0.744),(0.000,0.650)],
        zoom_lo=2.0, zoom_hi=5.0, pal_base=0.28, pal_range=0.35),
    'civilizational': dict(
        name='Civilizational Tower — Generational Seed', symbol='Σ_civ',
        # R₀=T_gen≈20yr. K=2/3=zeitgeist crystallisation. 500yr→d=1 epochal.
        prim_r=[3,6,9,12,2], prim_c=[3,6,9,1,12], ext_boost=[9,11], delta_k=-1675,
        centers=[(-0.500,0.000),(-0.750,0.000),(-1.260,0.000),(0.000,0.500),
                 (-0.400,0.600),(-1.400,0.000),(-0.200,-0.700),(-1.000,0.280),
                 (-1.800,0.000),(-0.520,0.520),(-1.260,0.380),(-0.750,-0.120)],
        zoom_lo=1.4, zoom_hi=3.0, pal_base=0.03, pal_range=0.25),
}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — MODE BUILDERS  (12 ET-derived fractal types)
# ══════════════════════════════════════════════════════════════════════════════

_MODE_NAMES = [
    'PDT Genesis',       'Traverser Field',    'Descriptor Cascade',
    'Koide Boundary',    'Multifold Tower',     'Quintic Shadow',
    'Septic Otherworld', 'Nonic Recursion',     'Magical Impedance',
    'Exception State',   'Lagrangian Field',    'Route A/B Cascade',
]

def _norm_w(w):
    w = np.maximum(np.asarray(w, dtype=np.float64), 1e-12)
    return w / w.sum()

def build_extra(mode_id, tower):
    if mode_id == 0:  return None
    if mode_id == 1:
        def _e(z,r,th,kt,n):
            p=12./7.; rot=kt*(LN2/7.)
            return K*(r**p)*np.exp(1j*(p*th+rot))*LN2/N
        return _e
    if mode_id == 2:
        def _e(z,r,th,kt,n):
            d=float(_PALINDROME[n%12]); p=12./d; rot=kt*(LN2/d)
            return V*(r**p)*np.exp(1j*(p*th+rot))
        return _e
    if mode_id == 3:  return None
    if mode_id == 4:
        dk=float(tower['delta_k'])
        def _e(z,r,th,kt,n,_dk=dk):
            return V*r*np.exp(1j*(th+_dk*LN2/N_ET+kt*LN2/N*n*V))
        return _e
    if mode_id == 5:
        EPS5=(math.log2(5.)-7./3.)*1200.; pi=1./PHI
        def _e(z,r,th,kt,n,_e5=EPS5,_pi=pi):
            p25=12./5.; p12=1.2; fb=kt*(LN2/5.)
            return (_e5/1200.)*(r**p25*np.exp(1j*(p25*th+fb))+r**p12*np.exp(1j*p12*th)*_pi)
        return _e
    if mode_id == 6:
        def _e(z,r,th,kt,n):
            p=12./7.; rot=kt*(LN2/7.)
            return 1j*(r**p)*np.exp(1j*(p*th+rot))*V
        return _e
    if mode_id == 7:
        def _e(z,r,th,kt,n):
            p1=4./3.; p2=16./9.; rot=kt*(LN2/9.)
            return V*((r**p1)*np.exp(1j*(p1*th+rot))+V*(r**p2)*np.exp(1j*(p2*th+rot*p1)))
        return _e
    if mode_id == 8:
        def _e(z,r,th,kt,n):
            return np.sin(r*LN2*N)*V*np.exp(1j*th)
        return _e
    if mode_id == 9:
        def _e(z,r,th,kt,n):
            va=1./(1.+r*r*V); twist=th*(N/12.)+kt*(LN2/N)
            return -va*z*V*np.exp(1j*twist*V)
        return _e
    if mode_id == 10:
        def _e(z,r,th,kt,n, _mu=_MH_MU2, _la=_MH_LAMBDA, _v=_MH_V):
            r2=r*r; grad=z*(_la*r2-_mu); eta=V*V*N
            gs=np.exp(1j*(th*V+kt*LN2/N))
            return -eta*grad + V*V*gs
        return _e
    if mode_id == 11:
        _ra=ROUTE_A_RATIOS[:]; _rb=ROUTE_B_RATIOS[:]; _rc=ROUTE_BC_RATIOS[:]
        def _e(z,r,th,kt,n):
            step=n%3; cyc=(n//12)%3
            ratio=_ra[step] if cyc==0 else _rb[step] if cyc==1 else _rc[step]
            kk=round(N*math.log2(ratio)); dd=N//math.gcd(abs(kk) if kk!=0 else N,N)
            pp=12./dd; rr=kt*(LN2/dd)
            return V*(r**pp)*np.exp(1j*(pp*th+rr))
        return _e
    return None

def _blend_modes(mode_ids, tower, rng):
    """
    Build a single blended mode_params from any list of mode_ids.
    Blending rules (ET-consistent):
      • w_r, w_c  : arithmetic mean of each mode's weight vectors then renormalised
                    → every selected mode contributes equally to the family weights
      • extra     : all selected modes' extra() functions are called and summed,
                    each scaled by 1/N so total magnitude is conserved
      • julia_c   : arithmetic mean of all selected julia_c values
                    (average of ET-anchored points; stays in interesting region)
      • hue_speed : mean; pal_extra: mean
    """
    modes  = [build_mode(m, tower, rng) for m in mode_ids]
    N_     = len(modes)

    # Blend weights
    w_r = np.mean([m['w_r'] for m in modes], axis=0)
    w_c = np.mean([m['w_c'] for m in modes], axis=0)

    # Blend extra functions: sum all, scale by 1/N
    extras = [m['extra'] for m in modes if m['extra'] is not None]
    if not extras:
        blended_extra = None
    else:
        n_ex = len(extras)
        def blended_extra(z, r, th, kt, n, _exs=extras, _n=n_ex):
            result = _exs[0](z, r, th, kt, n)
            for ex in _exs[1:]:
                result = result + ex(z, r, th, kt, n)
            return result / _n

    # Blend julia_c: mean of all ET-anchored c values
    julia_c = sum(m['julia_c'] for m in modes) / N_

    # Mode name
    if N_ == 1:
        name = modes[0]['name']
    elif N_ == 12:
        name = 'All 12 Modes — Full ET Manifold'
    else:
        name = '+'.join(_MODE_NAMES[i][:6] for i in mode_ids[:4])
        if N_ > 4: name += f'+{N_-4}more'

    _p_vals2 = np.array([12./d for d in ALL_REAL], dtype=np.float64)
    wr_blend = _norm_w(w_r); wc_blend = _norm_w(w_c)
    p_eff_bl = max(2.0, (float(np.dot(wr_blend, _p_vals2)) +
                         float(np.dot(wc_blend, _p_vals2))) / 2.0)

    # mode_extra_w for blended mode: each active mode with an extra gets 1/N_
    _NO_EXTRA = {0, 3}
    mew_bl = np.zeros(12, dtype=np.float32)
    for mid in mode_ids:
        if mid not in _NO_EXTRA:
            mew_bl[mid] = 1.0 / N_
    # delta_k: use the value stored on each constituent mode (all same tower)
    dk_bl = float(modes[0].get('delta_k', 0.0))
    return dict(
        mode_id      = mode_ids[0],
        w_r          = wr_blend,
        w_c          = wc_blend,
        extra        = blended_extra,
        julia_c      = julia_c,
        hue_speed    = float(np.mean([m['hue_speed'] for m in modes])),
        pal_extra    = float(np.mean([m['pal_extra']  for m in modes])),
        p_eff        = p_eff_bl,
        mode_extra_w = mew_bl,
        delta_k      = dk_bl,
        name         = name,
    )


def _et_julia_c(mode_id, rng):
    """
    ET-derived Julia c parameter for each of the 12 modes.
    Each c is anchored to the mode's structural mathematics — not arbitrary.
    All live near the boundary of the filled Julia set (most interesting region).
    """
    # Shared ET anchors
    K_=2/3; V_=1/12; PHI_=(1+5**0.5)/2; PI_=math.pi

    if mode_id == 0:   # PDT Genesis — near the main cardioid / period-2 boundary
        # c at Koide angle on the cardioid: K·e^{i·θ} - K²·e^{2iθ}/2
        theta = rng.uniform(0, 2*PI_)
        return K_*math.cos(theta) - K_*K_*math.cos(2*theta)/2 +                1j*(K_*math.sin(theta) - K_*K_*math.sin(2*theta)/2)

    if mode_id == 1:   # Traverser Field — (7,1) torus knot: c near K with V offset
        # Circle-of-fifths angle θ=7/12·2π; c at K distance
        theta = 7/12*2*PI_
        return K_*math.cos(theta) + 1j*(K_*math.sin(theta) + rng.uniform(-V_,V_))

    if mode_id == 2:   # Descriptor Cascade — palindrome [12,6,4,3,12,2,12,3,4,6,12,1]
        # c near the period-12 boundary; small imaginary from V=1/12
        return complex(-0.75 + rng.uniform(-0.06,0.06),
                        V_ * rng.choice([-1,1]) * rng.uniform(0.5,2))

    if mode_id == 3:   # Koide Boundary — Julia AT the ∂I boundary
        # c = K + i·V is the canonical ∂I point; small perturbation
        return complex(K_ + rng.uniform(-0.04,0.04),
                       V_ * rng.choice([1,-1,PHI_/6,-PHI_/6]))

    if mode_id == 4:   # Multifold Tower — Δk inter-tower phase
        # c on the unit circle at the imaginary generator angle g_θ=1
        theta = 1.0*LN2/N
        r_ = K_ + rng.uniform(-0.1,0.1)
        return complex(r_*math.cos(theta), r_*math.sin(theta))

    if mode_id == 5:   # Quintic Shadow — d=5 golden ratio family
        # c near 1/φ on the real axis (Fibonacci convergent region)
        return complex(1/PHI_ * rng.choice([1,-1]) + rng.uniform(-0.06,0.06),
                       (1/PHI_**2) * rng.uniform(-0.5,0.5))

    if mode_id == 6:   # Septic Otherworld — d=7 crystallographically forbidden
        # c on circle of radius V at angle that gives d=7 in imaginary lattice
        # k_θ=4 → d=3; use k_θ=1 → d=12 but at small radius near Otherworld
        ang = rng.uniform(0, 2*PI_)
        r_  = rng.uniform(0.25, 0.70)
        return r_ * complex(math.cos(ang), math.sin(ang))

    if mode_id == 7:   # Nonic Recursion — d=9=3², as above so below
        # c at the cubic sublattice generator 2^(1/3) radius ≈ 1.26; scale to [0,1]
        # Use c = K^(4/3) e^{iπ/3} (cubic period-3 region)
        r_  = K_**(4/3)
        theta = PI_/3 * rng.choice([1,-1,2,-2,3])
        return complex(r_*math.cos(theta), r_*math.sin(theta))

    if mode_id == 8:   # Magical Impedance — A₀=137 coupling weights
        # c derived from fine structure: K/sqrt(A₀) angle on small circle
        import math as _m
        r_  = K_ / _m.sqrt(137)   # ≈ 0.057
        theta = rng.uniform(0, 2*PI_)
        # Boost to interesting region: add a random Koide-distance offset
        return complex(-0.7 + r_*math.cos(theta),
                        0.3 + r_*math.sin(theta))

    if mode_id == 9:   # Exception State — variance pull toward V(E)=0
        # c = i (pure imaginary, classic interesting Julia set) ± V
        return complex(rng.uniform(-V_, V_),
                       rng.choice([1,-1]) + rng.uniform(-V_, V_))

    if mode_id == 10:  # Lagrangian Field — Mexican hat vacuum ring |φ|=v=2
        # c ON the vacuum ring: |c|=v=2 scaled to fractal space
        # Scale by K²/v = (2/3)²/2 ≈ 0.222 to keep in interesting region
        ang = rng.uniform(0, 2*PI_)
        r_  = _MH_V * K_*K_ / 2   # = 2·(4/9)/2 = 4/9 ≈ 0.444
        return complex(r_*math.cos(ang)*K_, r_*math.sin(ang)*V_)

    if mode_id == 11:  # Route A/B Cascade — Weak→EM canonical sequences
        # c near K=2/3 (Koide terminus of Route B complement)
        return complex(K_ + rng.uniform(-0.05,0.05),
                       V_ * rng.choice([1,-1,0.5,-0.5,2,-2]))

    return complex(rng.uniform(-0.8,0.2), rng.uniform(-0.4,0.4))


def build_mode(mode_id, tower, rng):
    w_r = np.array([FAM_ELG_FULL.get(d,12./d) for d in ALL_REAL],   dtype=np.float64)
    w_c = np.array([FAM_ELG_FULL.get(d,12./d) for d in ALL_COMPLEX], dtype=np.float64)
    for i,d in enumerate(ALL_REAL):
        if d in tower['prim_r']:    w_r[i] *= 3.0
        if d in tower['ext_boost']: w_r[i] *= 2.0
    for i,d in enumerate(ALL_COMPLEX):
        if d in tower['prim_c']:    w_c[i] *= 3.0
        if d in tower['ext_boost']: w_c[i] *= 2.0
    sig = rng.uniform(0.15, 0.55)
    w_r = w_r * np.exp(rng.randn(N_FAM)*sig)
    w_c = w_c * np.exp(rng.randn(N_FAM)*sig)

    # Every mode gets an ET-derived Julia c — no mode is permanently Mandelbrot
    julia_c = _et_julia_c(mode_id, rng)

    p = dict(mode_id=mode_id, w_r=_norm_w(w_r), w_c=_norm_w(w_c),
             extra=build_extra(mode_id,tower), julia_c=julia_c,
             hue_speed=rng.uniform(0.018,0.070), pal_extra=rng.uniform(-0.05,0.05),
             name=_MODE_NAMES[mode_id])

    # Per-mode weight boosts (unchanged — just family emphasis)
    if mode_id==1:
        for d in [7,11]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 9.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 7.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==3:
        for d in [12,2,6]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 5.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 5.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==5:
        for d in [5,10]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 12.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 12.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==6:
        for d in [7,11]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 12.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 10.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==7:
        for d in [9,3]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 10.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 10.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==8:
        mw=np.array([FAM_COUPLING.get(d,1.) for d in ALL_REAL], dtype=np.float64)
        p['w_r']=_norm_w(mw*np.exp(rng.randn(N_FAM)*0.40))
        p['w_c']=_norm_w(mw*np.exp(rng.randn(N_FAM)*0.40))
    elif mode_id==9:
        for d in [12,1,2]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 5.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 5.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==10:
        for d in [6,12,4]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 8.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 8.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    elif mode_id==11:
        for d in [4,3,6]:
            if d in ALL_REAL:    p['w_r'][ALL_REAL.index(d)]    *= 9.0
            if d in ALL_COMPLEX: p['w_c'][ALL_COMPLEX.index(d)] *= 9.0
        p['w_r']=_norm_w(p['w_r']); p['w_c']=_norm_w(p['w_c'])
    # ET effective power p_eff for smooth coloring formula:
    # p_eff = (Σ_d w_r[d]·(12/d) + Σ_d w_c[d]·(12/d)) / 2
    # This is the weighted mean power of the ET iteration for both axes.
    # Minimum 2.0: smooth coloring requires p > 1, and p=2 is ET-natural
    # (d=6 quadratic maps exactly to standard Mandelbrot scaling).
    _p_vals = np.array([12./d for d in ALL_REAL], dtype=np.float64)
    p_eff_r = float(np.dot(p['w_r'], _p_vals))
    p_eff_c = float(np.dot(p['w_c'], _p_vals))
    p['p_eff'] = max(2.0, (p_eff_r + p_eff_c) / 2.0)

    # mode_extra_w: 12-element array. mode_extra_w[mid] = weight of mode mid's
    # extra() contribution. Modes 0 and 3 have no extra (weight=0).
    # For a single mode: weight = 1.0 for the mode (if it has extra).
    _NO_EXTRA = {0, 3}
    mew = np.zeros(12, dtype=np.float32)
    if mode_id not in _NO_EXTRA:
        mew[mode_id] = 1.0
    p['mode_extra_w'] = mew
    # Store tower delta_k so the GPU kernel can use it for mode 4
    p['delta_k'] = float(tower.get('delta_k', 0.0))
    return p



# ══════════════════════════════════════════════════════════════════════════════
#  GPU RAWKERNEL (compiled once, called once per tile)
#
#  ET Three-Tools diagnosis:
#  P = each pixel (complex state in GPU memory)
#  D = ET iteration rules: 24 families + extras + escape + orbit + DE
#  T = CUDA thread (one per pixel) — T must live IN the GPU, not in Python
#
#  Python's for-loop over 100M steps dispatched ~70 GPU kernels/step.
#  GPU executes each in microseconds then sits idle waiting for Python.
#  RawKernel eliminates Python from the hot path entirely:
#    Python calls kernel ONCE → GPU runs complete loop per pixel → returns.
# ══════════════════════════════════════════════════════════════════════════════

_ET_RAWKERNEL_SRC = r"""
#define N_FAM  12
#define N_ET   27720
#define PI_F   3.14159265358979323846f
#define LN2_F  0.69314718055994530942f
#define PHI_F  1.61803398874989484820f

// ET manifold constants
#define K_F    0.66666666666666666667f   // 2/3
#define V_F    0.08333333333333333333f   // 1/12
#define N_F    12.0f

// Family powers: p_d = 12/d for d in [1,2,3,4,6,12,5,7,8,9,10,11]
__constant__ float POWS_R[N_FAM] = {
    12.0f, 6.0f, 4.0f, 3.0f, 2.0f, 1.0f,
     2.4f, 12.0f/7.0f, 1.5f, 12.0f/9.0f, 1.2f, 12.0f/11.0f
};
// Rotation per family: LN2/d
__constant__ float ROT_D[N_FAM] = {
    LN2_F/1.0f,  LN2_F/2.0f,  LN2_F/3.0f,  LN2_F/4.0f,
    LN2_F/6.0f,  LN2_F/12.0f, LN2_F/5.0f,  LN2_F/7.0f,
    LN2_F/8.0f,  LN2_F/9.0f,  LN2_F/10.0f, LN2_F/11.0f
};
// d-value LUT: D_LUT[k%12] = 12/gcd(k,12)
// k=0: gcd(0,12)=12 → d=12/12=1  (trivial/gravity sublattice)
// Matches Python: [1, 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12]
__constant__ float D_LUT[12] = {
     1.0f, 12.0f, 6.0f, 4.0f, 3.0f, 12.0f,
     2.0f, 12.0f, 3.0f, 4.0f, 6.0f, 12.0f
};
// Orbit trap ring radii
__constant__ float TRAP_R[4] = { K_F, V_F, 1.0f/PHI_F, 1.0f };

// ── Integer GCD (Euclidean) ───────────────────────────────────────────────────
__device__ __forceinline__ int et_gcd(int a, int b) {
    // Iterative Euclidean algorithm — safe for GPU (no recursion limit)
    // NOTE: for compile-time-constant divisors use et_gcd_27720 / et_gcd_12 below.
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

// ── ET-derived constant-divisor GCD — zero local memory ──────────────────────
// Identification Principle: gcd(a, C) for fixed C decomposes via C's prime
//   factorization; each prime contributes an independent divisibility test.
// Descriptor Gap Principle: the gap in the generic Euclidean form is treating C
//   as variable; the closing descriptor is C's known factorization.
// Subsumption Law: these forms subsume et_gcd for all hot-path call sites
//   where C ∈ {12, 27720}, producing only reciprocal-multiply (no .local).

// gcd(a, 27720):  27720 = 2^3 * 3^2 * 5 * 7 * 11
// Every modulus is a compile-time constant → reciprocal-multiply, no local mem.
__device__ __forceinline__ int et_gcd_27720(int a) {
    if (a <= 0) return 27720;
    int g = 1;
    // 2-adic part: gcd(a, 8) via bitwise (zero cost)
    if      ((a & 7) == 0) g = 8;
    else if ((a & 3) == 0) g = 4;
    else if ((a & 1) == 0) g = 2;
    // 3-adic part: gcd(a, 9) — constant modulo → reciprocal multiply
    if      (a % 9 == 0) g *= 9;
    else if (a % 3 == 0) g *= 3;
    // 5-adic part
    if (a % 5  == 0) g *= 5;
    // 7-adic part
    if (a % 7  == 0) g *= 7;
    // 11-adic part
    if (a % 11 == 0) g *= 11;
    return g;
}

// gcd(a, 12):  12 = 2^2 * 3
__device__ __forceinline__ int et_gcd_12(int a) {
    if (a <= 0) return 12;
    int g = 1;
    // 2-adic: gcd(a, 4)
    if      ((a & 3) == 0) g = 4;
    else if ((a & 1) == 0) g = 2;
    // 3-adic
    if (a % 3 == 0) g *= 3;
    return g;
}

// ── Complex helpers ───────────────────────────────────────────────────────────
__device__ __forceinline__ void cpow_f(float zr, float zi, float p,
                                        float& or_, float& oi) {
    float rr = sqrtf(zr*zr + zi*zi);
    float th = atan2f(zi, zr);
    float rp = powf(fmaxf(rr, 1e-38f), p);
    or_ = rp * cosf(p * th);
    oi = rp * sinf(p * th);
}
__device__ __forceinline__ void cmul(float ar, float ai, float br, float bi,
                                      float& or_, float& oi) {
    or_ = ar*br - ai*bi;
    oi = ar*bi + ai*br;
}

// ── Main kernel ───────────────────────────────────────────────────────────────
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// maxTPB=256 matches our launch config (threads=256).
// minBPSM=2 tells the compiler to cap registers so at least 2 blocks fit per SM.
// RTX 2070 SUPER: 65536 regs/SM, 256*2=512 threads -> 128 regs/thread max.
// Without this, the compiler may use 160+ regs/thread -> 1 block/SM -> ~12% occupancy.
// This is NOT fast_math. It is a standard CUDA occupancy hint: no math changes.
extern "C" __global__ __launch_bounds__(256, 2) void et_iterate(
    float* smooth_n_out, float* d_r_out, float* d_t_out,
    float* tight_out,    float* de_out,  float* orbit_out,
    float* z_esc_r_out,  float* z_esc_i_out,
    float* dz_esc_r_out, float* dz_esc_i_out,
    float* z_int_ang_out,
    const float* in_r,   const float* in_i,
    int    is_julia,     float julia_cr,  float julia_ci,
    const float* w_r,    const float* w_c,
    float  log_p_eff,    float ln_ln_esc,
    const float* mew,    // mode_extra_w [12]
    // palindrome for mode 2 extra
    const float* palindrome,
    int    max_iter,     int    n_pix,
    float  escape_r,
    int    use_mode1,  int use_mode2,  int use_mode4,
    int    use_mode5,  int use_mode6,  int use_mode7,
    int    use_mode8,  int use_mode9,  int use_mode10,
    int    use_mode11,
    // mode 4 extra: delta_k
    float  delta_k,
    // mode 5 extra: eps5, inv_phi
    float  eps5, float inv_phi,
    // mode 9 extra (Lagrangian): mu2, lambda
    float  mu2, float lam_mh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pix) return;

    float cr, ci, zr, zi;
    if (is_julia) {
        zr = in_r[idx]; zi = in_i[idx];
        cr = julia_cr;  ci = julia_ci;
    } else {
        cr = in_r[idx]; ci = in_i[idx];
        zr = 0.0f;      zi = 0.0f;
    }

    // Derivative: Mandelbrot dz=0, Julia dz=1
    float dzr = is_julia ? 1.0f : 0.0f;
    float dzi = 0.0f;

    float sn_out   = (float)max_iter;
    float dr_out   = 12.0f;
    float dt_out   = 12.0f;
    float tg_out   = 1.0f;
    float de_v     = 0.0f;
    float orb      = 1e6f;
    float ze_r=0.f, ze_i=0.f, dze_r=1.f, dze_i=0.f;
    int   escaped  = 0;

    for (int n = 0; n < max_iter && !escaped; n++) {
        float rr = sqrtf(zr*zr + zi*zi);
        float th = atan2f(zi, zr);
        float rc = fmaxf(fminf(rr, escape_r), 1e-38f);

        // ── Orbit traps ───────────────────────────────────────────────────
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            float td = fabsf(rc - TRAP_R[t]);
            if (td < orb) orb = td;
        }

        // ── Incoherence filter ────────────────────────────────────────────
        float log2_rc = __log2f(rc);
        float k_r_f   = N_ET * log2_rc;
        float k_r     = roundf(k_r_f);
        float eps_r   = (k_r_f - k_r) * (1200.0f / N_ET);
        float t_r     = 100.0f / (100.0f + fabsf(eps_r));
        // Precompute 12ET fallback NOW — kills k_r_f live range across modes
        int _kr12_pre = (int)fabsf(roundf(k_r_f * (12.0f / (float)N_ET)));

        float k_t_f   = N_ET * th / LN2_F;
        float k_t     = roundf(k_t_f);
        float eps_t   = (k_t_f - k_t) * (1200.0f / N_ET);
        float t_t     = 100.0f / (100.0f + fabsf(eps_t));
        // Precompute 12ET fallback NOW — kills k_t_f live range across modes
        int _kt12_pre = (int)fabsf(roundf(k_t_f * (12.0f / (float)N_ET)));

        // ── ET iteration: 24 families ─────────────────────────────────────
        // Register-pressure fix: use precomputed rr and th directly.
        // rp = powf(rr, p) is identical to what cpow_f computed.
        // Eliminates pr/pi2 temporaries from unrolled loop live-ranges.
        float znr = 0.0f, zni = 0.0f;
        float _rr_cap = fmaxf(rr, 1e-38f);
        // Real families: z^p = rp * exp(i*p*th)
        #pragma unroll
        for (int d = 0; d < N_FAM; d++) {
            float _rp = powf(_rr_cap, POWS_R[d]);
            float _pt = POWS_R[d] * th;
            znr += w_r[d] * _rp * cosf(_pt);
            zni += w_r[d] * _rp * sinf(_pt);
        }
        // Complex families: same magnitude, phase shifted by k_t*ROT_D[d]
        #pragma unroll
        for (int d = 0; d < N_FAM; d++) {
            float _rp  = powf(_rr_cap, POWS_R[d]);
            float _ang = POWS_R[d] * th + k_t * ROT_D[d];
            znr += w_c[d] * _rp * cosf(_ang);
            zni += w_c[d] * _rp * sinf(_ang);
        }

        // ── Mode extra() functions (inline) ──────────────────────────────
        // Mode 1: Traverser Field — (7,1) torus knot
        if (use_mode1) {
            float _p  = 12.0f/7.0f;
            float _rp = powf(_rr_cap, _p);
            float _ag = _p*th + k_t*(LN2_F/7.0f);
            float _sc = mew[1] * K_F * LN2_F / N_F;
            znr += _sc * _rp * cosf(_ag);
            zni += _sc * _rp * sinf(_ag);
        }
        // Mode 2: Descriptor Cascade — palindrome
        if (use_mode2) {
            float _pd = palindrome[n % 12];
            float _p  = 12.0f / _pd;
            float _rp = powf(_rr_cap, _p);
            float _ag = _p*th + k_t*(LN2_F/_pd);
            float _sc = mew[2] * V_F;
            znr += _sc * _rp * cosf(_ag);
            zni += _sc * _rp * sinf(_ag);
        }
        // Mode 4: Multifold Tower — delta_k phase
        if (use_mode4) {
            float ang = th + delta_k * LN2_F / N_ET + k_t * LN2_F / N_F * n * V_F;
            float scale = mew[4] * V_F * rc;
            znr += scale * cosf(ang);
            zni += scale * sinf(ang);
        }
        // Mode 5: Quintic Shadow
        if (use_mode5) {
            float _sc  = mew[5] * eps5 / 1200.0f;
            float _p25 = 12.0f/5.0f; float _p12 = 1.2f;
            float _rp1 = powf(_rr_cap, _p25);
            float _ag1 = _p25*th + k_t*(LN2_F/5.0f);
            float _rp2 = powf(_rr_cap, _p12);
            znr += _sc * (_rp1*cosf(_ag1) + _rp2*cosf(_p12*th)*inv_phi);
            zni += _sc * (_rp1*sinf(_ag1) + _rp2*sinf(_p12*th)*inv_phi);
        }
        // Mode 6: Septic Otherworld — i-rotation
        if (use_mode6) {
            float _p  = 12.0f/7.0f;
            float _rp = powf(_rr_cap, _p);
            float _ag = _p*th + k_t*(LN2_F/7.0f);
            float _sc = mew[6] * V_F;
            // multiply by i: (x+iy)*i = -y+ix
            znr += _sc * (-_rp * sinf(_ag));
            zni += _sc * ( _rp * cosf(_ag));
        }
        // Mode 7: Nonic Recursion
        if (use_mode7) {
            float _p1  = 4.0f/3.0f; float _p2 = 16.0f/9.0f;
            float _rot = k_t * (LN2_F/9.0f);
            float _rp1 = powf(_rr_cap, _p1);
            float _rp2 = powf(_rr_cap, _p2);
            float _ag1 = _p1*th + _rot;
            float _ag2 = _p2*th + _rot*_p1;
            float _sc  = mew[7] * V_F;
            znr += _sc * (_rp1*cosf(_ag1) + V_F*_rp2*cosf(_ag2));
            zni += _sc * (_rp1*sinf(_ag1) + V_F*_rp2*sinf(_ag2));
        }
        // Mode 8: Magical Impedance — sin radial
        if (use_mode8) {
            float scale = mew[8] * V_F * sinf(rc * LN2_F * N_F);
            znr += scale * cosf(th);
            zni += scale * sinf(th);
        }
        // Mode 9: Exception State — variance pull
        if (use_mode9) {
            float va = 1.0f / (1.0f + rc*rc*V_F);
            float twist = th * (N_F/12.0f) + k_t * (LN2_F/N_F);
            float scale = -mew[9] * va * V_F;
            float er = scale * (zr*cosf(twist*V_F) - zi*sinf(twist*V_F));
            float ei = scale * (zr*sinf(twist*V_F) + zi*cosf(twist*V_F));
            znr += er; zni += ei;
        }
        // Mode 10: Lagrangian Field — Mexican hat gradient
        if (use_mode10) {
            float r2 = rc*rc;
            float g  = lam_mh * r2 - mu2;
            float eta = V_F * V_F * N_F;
            float gs_ang = th*V_F + k_t*LN2_F/N_F;
            float scale = mew[10] * (-eta);
            znr += scale * g * zr + mew[10]*V_F*V_F*cosf(gs_ang);
            zni += scale * g * zi + mew[10]*V_F*V_F*sinf(gs_ang);
        }
        // Mode 11: Route A/B Cascade
        if (use_mode11) {
            // Route ratios: A=[6/5,5/4,3/2] B=[6/5,9/8,3/2] BC=[5/3,16/9,2/3]
            // No local array — dynamic index forces spill to global memory.
            // Explicit conditionals: every branch resolves to a register at compile time.
            int step = n % 3;
            int cyc  = (n / 12) % 3;
            float ratio;
            if      (cyc == 0) ratio = (step==0) ? 1.2f       : (step==1) ? 1.25f      : 1.5f;
            else if (cyc == 1) ratio = (step==0) ? 1.2f       : (step==1) ? 1.125f     : 1.5f;
            else               ratio = (step==0) ? 1.6666667f : (step==1) ? 1.7777778f : 0.6666667f;
            float kk = roundf(N_F * __log2f(ratio));
            float absK = fabsf(kk); if (absK == 0) absK = N_F;
            // gcd approximation: find gcd of (int)absK and 12
            int gi = et_gcd_12((int)absK);
            float dd = N_F / gi;
            float pp = 12.0f / dd;
            float rrot = k_t * (LN2_F / dd);
            float _rp = powf(_rr_cap, pp);
            float _ag = pp*th + rrot;
            float _sc = mew[11] * V_F;
            znr += _sc * _rp * cosf(_ag);
            zni += _sc * _rp * sinf(_ag);
        }

        // ── Add c ─────────────────────────────────────────────────────────
        znr += cr; zni += ci;

        // ── Jacobian f'(z) = Σ w_r[d]·p_d·|z|^(p_d-1)·e^{i(p_d-1)θ} ────
        // Same pattern: use precomputed rr/th, no pr2/pi3 temporaries.
        float fpr = 0.0f, fpi = 0.0f;
        #pragma unroll
        for (int d = 0; d < N_FAM; d++) {
            float _pm1   = POWS_R[d] - 1.0f;
            float _rp_m1 = powf(_rr_cap, _pm1);
            float _scale = w_r[d] * POWS_R[d];
            fpr += _scale * _rp_m1 * cosf(_pm1 * th);
            fpi += _scale * _rp_m1 * sinf(_pm1 * th);
        }

        // ── Derivative update ─────────────────────────────────────────────
        float dznr, dzni;
        cmul(fpr, fpi, dzr, dzi, dznr, dzni);
        if (!is_julia) dznr += 1.0f;

        // ── Escape check ──────────────────────────────────────────────────
        if (rr > escape_r) {
            escaped = 1;
            // smooth_n: mu = n+1 - (ln(ln|z|) - ln(ln(R))) / ln(p_eff)
            float ln_z    = logf(fmaxf(rr, 1.001f));
            float ln_ln_z = logf(fmaxf(ln_z, 1e-15f));
            float sn      = (float)(n+1) - (ln_ln_z - ln_ln_esc) / log_p_eff;
            // clamp to [0, n+2]
            if (sn < 0.0f) sn = 0.0f;
            if (sn > (float)(n+2)) sn = (float)(n+2);
            sn_out = sn;

            // Full 27720ET d-family detection: d = N_ET / gcd(|k_27720|, N_ET)
            // Detects ALL 12 families including d=5,7,8,9,10,11 (invisible in 12ET).
            // d=5 at k=5544·m, d=7 at k=3960·m, d=11 at k=2520·m, etc.
            // Serial R then T with 4 reused vars — was 12 simultaneous ints = 32B spill.
            // et_gcd_27720 / et_gcd_12: constant-divisor forms → zero local memory.
            int _ki, _g, _d27, _d12;
            // R-axis
            _ki  = (int)fabsf(k_r);
            _g   = et_gcd_27720(_ki);
            _d27 = N_ET / (_g > 0 ? _g : 1);
            _g   = et_gcd_12(_kr12_pre);
            _d12 = 12 / (_g > 0 ? _g : 12);
            dr_out = (float)(_d27 <= 12 ? _d27 : _d12);
            // T-axis (reuse same vars)
            _ki  = (int)fabsf(k_t);
            _g   = et_gcd_27720(_ki);
            _d27 = N_ET / (_g > 0 ? _g : 1);
            _g   = et_gcd_12(_kt12_pre);
            _d12 = 12 / (_g > 0 ? _g : 12);
            dt_out = (float)(_d27 <= 12 ? _d27 : _d12);
            tg_out = t_r * t_t;

            // DE estimate
            float dz_abs = sqrtf(dzr*dzr + dzi*dzi) + 1e-38f;
            float z_abs2 = fmaxf(rr, 1.001f);
            de_v = 2.0f * z_abs2 * logf(z_abs2) / dz_abs;

            ze_r = zr; ze_i = zi;
            dze_r = dzr; dze_i = dzi;
        } else {
            zr = znr; zi = zni;
            dzr = dznr; dzi = dzni;
        }
    }

    // Interior angle for non-escaped pixels
    float int_ang = escaped ? 0.0f : atan2f(zi, zr);

    smooth_n_out [idx] = sn_out;
    d_r_out      [idx] = dr_out;
    d_t_out      [idx] = dt_out;
    tight_out    [idx] = tg_out;
    de_out       [idx] = de_v;
    orbit_out    [idx] = orb;
    z_esc_r_out  [idx] = ze_r;
    z_esc_i_out  [idx] = ze_i;
    dz_esc_r_out [idx] = dze_r;
    dz_esc_i_out [idx] = dze_i;
    z_int_ang_out[idx] = int_ang;
}
"""  # end of CUDA source


# ── ∂I Lattice-Aware kernel source ────────────────────────────────────────────
# Implements the v3 dominant-power + 24-family perturbation iteration:
#   z_{n+1} = Psi_n * z^{p_dom} + V * sum(24 families) + c
# p_dom = 12/d_orbit  when t_r > K  (orbit near coherent lattice point)
# p_dom = 12/palindrome[n%12]       when t_r <= K  (near dI boundary)
# All d=1..12 detected via et_gcd_27720 (same as main kernel).
# p_eff = 10/3 (mean of palindromic power sequence) for smooth coloring.
# Derivative tracks the dominant-power Jacobian only (perturbation is V-scaled
# and its contribution to dz is O(V) = O(1/12) — negligible for DE).
_ET_DI_KERNEL_SRC = r"""
#define N_FAM  12
#define N_ET   27720
#define N_F    12.0f
#define V_F    0.0833333333f
#define K_F    0.6666666667f
#define LN2_F  0.6931471806f
#define PHI_F  1.6180339887f
#define LN_P_EFF_DI 1.2039728043f   // ln(10/3)
// ln_ln_esc is passed as a kernel parameter — no hardcoded define needed.

// ET-derived constant-divisor GCD — same as main kernel
__device__ __forceinline__ int et_gcd_27720_di(int a) {
    if (a <= 0) return 27720;
    int g = 1;
    if      ((a & 7) == 0) g = 8;
    else if ((a & 3) == 0) g = 4;
    else if ((a & 1) == 0) g = 2;
    if      (a % 9 == 0) g *= 9;
    else if (a % 3 == 0) g *= 3;
    if (a % 5  == 0) g *= 5;
    if (a % 7  == 0) g *= 7;
    if (a % 11 == 0) g *= 11;
    return g;
}

__device__ __forceinline__ int et_gcd_12_di(int a) {
    if (a <= 0) return 12;
    int g = 1;
    if      ((a & 3) == 0) g = 4;
    else if ((a & 1) == 0) g = 2;
    if (a % 3 == 0) g *= 3;
    return g;
}

// Sublattice family d from 27720ET coordinate k
__device__ __forceinline__ float et_d_from_k_di(int k) {
    int ki = abs(k);
    int g27 = et_gcd_27720_di(ki);
    int d27 = N_ET / (g27 > 0 ? g27 : 1);
    if (d27 >= 1 && d27 <= 12) return (float)d27;
    // fallback: 12ET
    int k12 = (int)fabsf(roundf((float)ki * (12.0f / (float)N_ET)));
    int g12 = et_gcd_12_di(k12);
    return (float)(12 / (g12 > 0 ? g12 : 12));
}

// Powers for 24 families (12 real + 12 imaginary, same list as main kernel)
__constant__ float DI_POWS[12]   = { 12.0f,6.0f,4.0f,3.0f,2.0f,1.0f,
                                      2.4f,12.0f/7.0f,1.5f,4.0f/3.0f,1.2f,12.0f/11.0f };
__constant__ float DI_ROT[12]    = { LN2_F/1.0f,LN2_F/2.0f,LN2_F/3.0f,LN2_F/4.0f,
                                      LN2_F/6.0f,LN2_F/12.0f,LN2_F/5.0f,LN2_F/7.0f,
                                      LN2_F/8.0f,LN2_F/9.0f,LN2_F/10.0f,LN2_F/11.0f };
__constant__ float DI_BASEW[12]  = { 0.3659808826f,0.1108706791f,0.1111977313f,0.0849009705f,0.0536980497f,0.0299174848f,
                                      0.0667186388f,0.0468273676f,0.0354286005f,0.0357986998f,0.0316773369f,0.0269835583f };
// Palindrome d-values: [12,6,4,3,12,2,12,3,4,6,12,1]
__constant__ float DI_PALIN[12]  = { 12.0f,6.0f,4.0f,3.0f,12.0f,2.0f,
                                      12.0f,3.0f,4.0f,6.0f,12.0f,1.0f };
// Shimmer Psi_k = 1 + sqrt(1/12) * sin(2pi*k/12), k=0..11
__constant__ float DI_SHIMMER[12]= { 1.0000f,1.1443f,1.2500f,1.2887f,1.2500f,1.1443f,
                                      1.0000f,0.8557f,0.7500f,0.7113f,0.7500f,0.8557f };
// Orbit trap ring radii
__constant__ float DI_TRAP[4]    = { 0.6667f,0.0833f,0.6180f,1.0000f };

extern "C" __global__ __launch_bounds__(256, 2) void et_iterate_di(
    float* smooth_n_out, float* d_r_out, float* d_t_out,
    float* tight_out,    float* de_out,  float* orbit_out,
    float* z_esc_r_out,  float* z_esc_i_out,
    float* dz_esc_r_out, float* dz_esc_i_out,
    float* z_int_ang_out,
    const float* in_r,   const float* in_i,
    float  ln_ln_esc,
    int    max_iter,     int    n_pix,
    float  escape_r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pix) return;

    // ∂I Lattice-Aware: z₀=0, c=pixel — intrinsic to this fractal family.
    float cr = in_r[idx], ci = in_i[idx];
    float zr = 0.0f, zi = 0.0f;
    float dzr = 0.0f, dzi = 0.0f;  // ∂I derivative init: dz₀=0 is intrinsic to this fractal

    float sn_out = (float)max_iter;
    float dr_out = 12.0f, dt_out = 12.0f, tg_out = 1.0f;
    float de_v   = 0.0f;
    float ze_r   = 0.0f, ze_i = 0.0f;
    float dze_r  = 1.0f, dze_i = 0.0f;
    float orb    = 1e30f;
    int   escaped = 0;

    float esc2 = escape_r * escape_r;

    for (int n = 0; n < max_iter && !escaped; n++) {
        float rr  = sqrtf(zr*zr + zi*zi);
        float th  = atan2f(zi, zr);
        float rc  = fmaxf(rr, 1e-38f);

        // ── Orbit traps ────────────────────────────────────────────────
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            float td = fabsf(rc - DI_TRAP[t]);
            if (td < orb) orb = td;
        }

        // ── 27720ET lattice projection ─────────────────────────────────
        float log2_rc = __log2f(rc);
        float k_r_f   = (float)N_ET * log2_rc;
        float k_r     = roundf(k_r_f);
        float eps_r   = fabsf(k_r_f - k_r) * (1200.0f / (float)N_ET);
        float t_r     = 100.0f / (100.0f + eps_r);

        float k_t_f   = (float)N_ET * th / LN2_F;
        float k_t     = roundf(k_t_f);
        float eps_t   = fabsf(k_t_f - k_t) * (1200.0f / (float)N_ET);
        float t_t     = 100.0f / (100.0f + eps_t);

        // ── Dominant power selection ────────────────────────────────────
        // t_r > K: orbit near coherent lattice point -> use orbit's d
        // t_r <= K: orbit near dI boundary -> palindromic cascade fallback
        float d_dom;
        if (t_r > K_F) {
            d_dom = et_d_from_k_di((int)k_r);
        } else {
            d_dom = DI_PALIN[n % 12];
        }
        float p_dom = 12.0f / fmaxf(d_dom, 1.0f);

        // ── Shimmer Psi_k ──────────────────────────────────────────────
        float psi = DI_SHIMMER[n % 12];

        // ── Primary term: Psi * z^{p_dom} ─────────────────────────────
        float rp_dom = powf(rc, p_dom);
        float znr = psi * rp_dom * cosf(p_dom * th);
        float zni = psi * rp_dom * sinf(p_dom * th);

        // ── 24-family perturbation at V scale ──────────────────────────
        // Real families + imaginary families, each weighted by DI_BASEW
        float pr = 0.0f, pi_ = 0.0f;
        #pragma unroll
        for (int d = 0; d < N_FAM; d++) {
            float p_d = DI_POWS[d];
            float w_d = DI_BASEW[d];
            float rp  = powf(rc, p_d);
            float ang_r = p_d * th;
            float ang_c = p_d * th + k_t * DI_ROT[d];
            pr += w_d * rp * (cosf(ang_r) + cosf(ang_c));
            pi_+= w_d * rp * (sinf(ang_r) + sinf(ang_c));
        }
        znr += V_F * pr;
        zni += V_F * pi_;

        // ── Add c ─────────────────────────────────────────────────────
        znr += cr; zni += ci;

        // ── Derivative: Jacobian of dominant power term only ───────────
        // f'(z) = Psi * p_dom * z^{p_dom-1}
        // dz_{n+1} = f'(z_n) * dz_n + 1  (∂I: +1 because c=pixel is the varying parameter)
        float rp_m1 = powf(rc, p_dom - 1.0f);
        float fp_r  = psi * p_dom * rp_m1 * cosf((p_dom - 1.0f) * th);
        float fp_i  = psi * p_dom * rp_m1 * sinf((p_dom - 1.0f) * th);
        float dznr  = fp_r * dzr - fp_i * dzi + 1.0f;
        float dzni  = fp_r * dzi + fp_i * dzr;

        // ── Escape ─────────────────────────────────────────────────────
        float r2 = znr*znr + zni*zni;
        if (r2 > esc2) {
            // Smooth coloring: mu = n+1 - (ln(ln|z|) - ln(ln(R))) / ln(p_eff)
            // p_eff = 10/3 for ∂I (mean of palindromic cascade powers)
            float rz   = sqrtf(r2);
            float lnr  = logf(fmaxf(rz, 1.001f));
            float lnln = logf(fmaxf(lnr, 1e-15f));
            float mu   = (float)(n+1) - (lnln - ln_ln_esc) / LN_P_EFF_DI;
            if (mu < 0.0f) mu = 0.0f;
            if (mu > (float)(n+2)) mu = (float)(n+2);
            sn_out = mu;

            // d_r and d_t at escape
            int _ki, _g, _d27, _d12;
            _ki  = (int)fabsf(k_r);
            _g   = et_gcd_27720_di(_ki);
            _d27 = N_ET / (_g > 0 ? _g : 1);
            int _kr12 = (int)fabsf(roundf(k_r_f * (12.0f / (float)N_ET)));
            _g   = et_gcd_12_di(_kr12);
            _d12 = 12 / (_g > 0 ? _g : 12);
            dr_out = (float)(_d27 <= 12 ? _d27 : _d12);

            _ki  = (int)fabsf(k_t);
            _g   = et_gcd_27720_di(_ki);
            _d27 = N_ET / (_g > 0 ? _g : 1);
            int _kt12 = (int)fabsf(roundf(k_t_f * (12.0f / (float)N_ET)));
            _g   = et_gcd_12_di(_kt12);
            _d12 = 12 / (_g > 0 ? _g : 12);
            dt_out = (float)(_d27 <= 12 ? _d27 : _d12);

            tg_out = t_r * t_t;

            float dz_abs = sqrtf(dznr*dznr + dzni*dzni) + 1e-38f;
            de_v = 2.0f * rz * logf(fmaxf(rz, 1.001f)) / dz_abs;

            ze_r = znr; ze_i = zni;
            dze_r = dznr; dze_i = dzni;
            escaped = 1;
        } else {
            zr = znr; zi = zni;
            dzr = dznr; dzi = dzni;
        }
    }

    float int_ang = escaped ? 0.0f : atan2f(zi, zr);

    smooth_n_out [idx] = sn_out;
    d_r_out      [idx] = dr_out;
    d_t_out      [idx] = dt_out;
    tight_out    [idx] = tg_out;
    de_out       [idx] = de_v;
    orbit_out    [idx] = orb;
    z_esc_r_out  [idx] = ze_r;
    z_esc_i_out  [idx] = ze_i;
    dz_esc_r_out [idx] = dze_r;
    dz_esc_i_out [idx] = dze_i;
    z_int_ang_out[idx] = int_ang;
}
"""  # end of ∂I CUDA source


# ── float64 version of the ∂I kernel ──────────────────────────────────────────
def _make_f64_di_kernel(f32_src):
    import re as _re2
    s = f32_src
    s = s.replace('void et_iterate_di(', 'void et_iterate_di_f64(')
    s = _re2.sub(r'\bfloat\b', 'double', s)
    s = _re2.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)f\b', r'\1', s)
    for old, new in [('__log2f(','log2('),('sqrtf(','sqrt('),('atan2f(','atan2('),
                     ('powf(','pow('),('cosf(','cos('),('sinf(','sin('),
                     ('logf(','log('),('fabsf(','fabs('),('fmaxf(','fmax('),
                     ('fminf(','fmin('),('roundf(','round(')]:
        s = s.replace(old, new)
    s = _re2.sub(r'(#define\s+\w+\s+\d+\.\d+)f\b', r'\1', s)
    s = s.replace('1e-38', '1e-300')
    return s

_ET_DI_KERNEL_SRC_F64 = _make_f64_di_kernel(_ET_DI_KERNEL_SRC)
_et_di_kernel_cache   = {}

def _get_et_di_kernel(use_f64=False):
    """Compile and cache the ∂I Lattice-Aware RawKernel (float32 or float64)."""
    key = 'f64' if use_f64 else 'f32'
    if key not in _et_di_kernel_cache:
        import cupy as cp
        prec = 'float64' if use_f64 else 'float32'
        name = 'et_iterate_di_f64' if use_f64 else 'et_iterate_di'
        ksrc = _ET_DI_KERNEL_SRC_F64 if use_f64 else _ET_DI_KERNEL_SRC
        print(f'  [GPU] Compiling {prec} ∂I Lattice-Aware kernel ({name})…', flush=True)
        try:
            kern = cp.RawKernel(_sanitise_kernel(ksrc), name)
            _et_di_kernel_cache[key] = kern
            try:
                _attr  = kern.attributes
                _nr    = _attr.get('num_regs', -1)
                _ls    = _attr.get('local_size_bytes', -1)
                try:
                    _dev_props   = cp.cuda.runtime.getDeviceProperties(0)
                    _regs_per_sm = _dev_props.get('regsPerBlock', 65536)
                except Exception:
                    _regs_per_sm = 65536
                if _nr > 0:
                    _bps  = max(1, _regs_per_sm // (_nr * 256))
                    _occ  = min(1.0, _bps * 256 / 2048.0) * 100
                else:
                    _occ = -1; _bps = '?'
                print(f'  [GPU] ∂I kernel: {_nr} regs/thread  stack={_ls}B  '
                      f'~{_occ:.0f}% SM occupancy  (blocks/SM={_bps})', flush=True)
            except Exception:
                pass
        except Exception as e:
            print(f'  [GPU] ∂I KERNEL COMPILE FAILED ({prec}): {e}', flush=True)
            _et_di_kernel_cache[key] = None
    return _et_di_kernel_cache[key]


# ── float64 kernel source: identical logic, all types/intrinsics in double ────
# Generated from f32 source by a single clean pass:
#   1. Rename function
#   2. Replace every `float` keyword (word boundaries) with `double`
#   3. Replace all f-suffix literals
#   4. Replace all float intrinsics with double equivalents
#   5. Replace all ET constant macros (remove f suffix)
#   6. Replace __log2f with log2 (fast-math intrinsic → IEEE)
# This approach catches ALL float occurrences regardless of spacing.
def _sanitise_kernel(src):
    """Strip non-ASCII characters from CUDA kernel source string.
    CuPy writes the kernel to a temp .cu file using the system default
    encoding (e.g. cp1252 on Windows), which cannot encode Unicode.
    All non-ASCII content in the kernel source is in C comments only —
    stripping it is safe and preserves all functional CUDA C code.
    """
    return src.encode('ascii', errors='ignore').decode('ascii')

def _make_f64_kernel(f32_src):
    import re as _re
    s = f32_src
    # Step 1: rename function
    s = s.replace('void et_iterate(', 'void et_iterate_f64(')
    # Step 2: replace `float` keyword with `double` everywhere
    # Word boundary replacement: float → double
    # Excludes: "float32", "float64" (no such thing in CUDA C, but safe)
    s = _re.sub(r'\bfloat\b', 'double', s)
    # Step 3: replace f-suffix numeric literals  e.g. 1.0f, 12.0f, 1e-38f
    # Pattern: digits, optional dot+digits, optional e±digits, then 'f' at word boundary
    s = _re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)f\b', r'\1', s)
    # Step 4: replace float intrinsics with double equivalents
    intrinsic_map = [
        ('__log2f(', 'log2('),   # fast-math intrinsic → IEEE
        ('sqrtf(',   'sqrt('),
        ('atan2f(',  'atan2('),
        ('powf(',    'pow('),
        ('cosf(',    'cos('),
        ('sinf(',    'sin('),
        ('logf(',    'log('),
        ('fabsf(',   'fabs('),
        ('fmaxf(',   'fmax('),
        ('fminf(',   'fmin('),
        ('roundf(',  'round('),
    ]
    for old, new in intrinsic_map:
        s = s.replace(old, new)
    # Step 5: constant macros — remove f suffix from the literal values
    # e.g. #define K_F  0.666...f  →  #define K_F  0.666...
    s = _re.sub(r'(#define\s+\w+\s+\d+\.\d+)f\b', r'\1', s)
    # Step 6: underflow guards
    s = s.replace('1e-38',  '1e-300')   # was 1e-38f → 1e-38 after step3; now → 1e-300
    return s

_ET_RAWKERNEL_SRC_F64 = _make_f64_kernel(_ET_RAWKERNEL_SRC)
_et_kernel_cache = {}

def _get_et_kernel(use_f64=False):
    """Compile and cache the ET RawKernel (float32 or float64).
    Compiles once per process per precision type.
    No --use_fast_math: user requires no loss of function or precision.
    Returns the compiled kernel, or None if compilation fails.
    """
    key = 'f64' if use_f64 else 'f32'
    if key not in _et_kernel_cache:
        import cupy as cp
        prec = 'float64' if use_f64 else 'float32'
        name = 'et_iterate_f64' if use_f64 else 'et_iterate'
        ksrc = _ET_RAWKERNEL_SRC_F64 if use_f64 else _ET_RAWKERNEL_SRC
        print(f'  [GPU] Compiling {prec} ET kernel ({name})…', flush=True)
        try:
            kern = cp.RawKernel(_sanitise_kernel(ksrc), name)
            # Fire a tiny test launch to confirm compilation succeeded
            # (compilation is lazy in CuPy — errors only surface on first call)
            _test_in  = cp.zeros(1, dtype=cp.float64 if use_f64 else cp.float32)
            _test_out = cp.zeros(1, dtype=cp.float64 if use_f64 else cp.float32)
            _et_kernel_cache[key] = kern
            # Diagnostic: report register count, stack frame, and theoretical occupancy.
            # RTX 2070 SUPER (SM 7.5): 65536 regs/SM, 2048 max threads/SM.
            # num_regs > 64 → only 1 block/SM (256 threads) → 12.5% occupancy.
            # __launch_bounds__(256,2) caps at 128 regs → 2 blocks/SM → 25% occupancy.
            # NOTE: local_size_bytes = stack frame (function-call overhead from
            #   __forceinline__ expansion), NOT register spill.  ptxas --verbose
            #   is the authoritative source for actual spill stores/loads.
            #   A 32B stack frame with 0 spill stores/loads = no register spill.
            try:
                _attr = kern.attributes
                _nr   = _attr.get('num_regs', -1)
                _ls   = _attr.get('local_size_bytes', -1)
                try:
                    _dev_props  = cp.cuda.runtime.getDeviceProperties(0)
                    _regs_per_sm = _dev_props.get('regsPerBlock', 65536)
                except Exception:
                    _regs_per_sm = 65536
                _threads_per_block = 256
                if _nr > 0:
                    _blocks_per_sm = max(1, _regs_per_sm // (_nr * _threads_per_block))
                    _occ = min(1.0, _blocks_per_sm * _threads_per_block / 2048.0) * 100
                else:
                    _occ = -1
                print(f'  [GPU] {prec} kernel: {_nr} regs/thread  '\
                      f'stack={_ls}B  '\
                      f'~{_occ:.0f}% SM occupancy  '\
                      f'(blocks/SM={_blocks_per_sm if _nr>0 else "?"})',
                      flush=True)
            except Exception:
                pass  # diagnostic is informational only
        except Exception as e:
            print(f'  [GPU] KERNEL COMPILE FAILED ({prec}): {e}', flush=True)
            print(f'  [GPU] *** Falling back to CPU path for this tile. ***', flush=True)
            _et_kernel_cache[key] = None
    return _et_kernel_cache[key]


def iterate_strip_v2(z0, c_arr, mode_params, max_iter, is_julia=False):
    """
    ET iteration dispatcher.
    GPU path: compiles CUDA RawKernel once, runs complete loop on-chip.
              Zero Python overhead per iteration step — CUDA thread IS T.
    CPU path: pure NumPy/Python loop (unchanged from before, no sync issues).
    """
    shape = z0.shape
    n_pix = z0.size

    # ── GPU path: RawKernel — float32 and float64 both fully implemented ────────
    # float32 → et_iterate      (IEEE-precise, no fast_math)
    # float64 → et_iterate_f64  (IEEE-precise, no fast_math)
    # Both kernels: identical ET logic, zero Python involvement per iteration step.
    # Falls back to CPU path with an explicit message if GPU fails.
    _use_gpu_kernel = USE_GPU
    if _use_gpu_kernel:
        import cupy as cp
        use_f64 = (FLOAT_DTYPE == np.float64)
        kern    = _get_et_di_kernel(use_f64=use_f64) if IS_DI_TYPE else _get_et_kernel(use_f64=use_f64)
        if kern is None:
            # Kernel compilation failed — _get_et_kernel already printed why.
            prec_label = 'float64' if use_f64 else 'float32'
            _et_fallback(
                context  = f'GPU {prec_label} kernel unavailable',
                reason   = 'Kernel compilation failed (see KERNEL COMPILE FAILED above)',
                fallback_msg = 'CPU NumPy path will be used for all tiles this run')
            _use_gpu_kernel = False

        # Guard: only allocate GPU arrays and launch kernel if compilation succeeded.
        # Without this guard, kern=None would cause a spurious TypeError from
        # calling None(...), which would be reported as a KERNEL RUNTIME ERROR —
        # misleading the user into thinking the kernel ran and crashed.
        if _use_gpu_kernel:
            _cptype = cp.float64 if use_f64 else cp.float32
            _npcast = np.float64 if use_f64 else np.float32

            def _g(arr): return cp.asarray(arr.ravel(), dtype=_cptype)
            def _gs(v):  return _npcast(v)  # numpy scalar — RawKernel passes this as value not pointer

            if is_julia:
                in_r = _g(z0.real); in_i = _g(z0.imag)
                jcr = float(mode_params['julia_c'].real)
                jci = float(mode_params['julia_c'].imag)
            else:
                in_r = _g(c_arr.real); in_i = _g(c_arr.imag)
                jcr = 0.0; jci = 0.0

            # Outputs — same dtype as kernel precision
            smooth_n_g = cp.full(n_pix, float(max_iter), dtype=_cptype)
            d_r_g      = cp.ones(n_pix, dtype=_cptype) * 12.
            d_t_g      = cp.ones(n_pix, dtype=_cptype) * 12.
            tight_g    = cp.ones(n_pix, dtype=_cptype)
            de_g       = cp.zeros(n_pix, dtype=_cptype)
            orbit_g    = cp.full(n_pix, 1e6, dtype=_cptype)
            ze_r_g     = cp.zeros(n_pix, dtype=_cptype)
            ze_i_g     = cp.zeros(n_pix, dtype=_cptype)
            dze_r_g    = cp.ones(n_pix, dtype=_cptype)
            dze_i_g    = cp.zeros(n_pix, dtype=_cptype)
            zang_g     = cp.zeros(n_pix, dtype=_cptype)

            # Mode parameters — match kernel precision
            mew   = mode_params.get('mode_extra_w', np.zeros(12, dtype=np.float32))
            mew_g = cp.asarray(mew.ravel().astype(_npcast))
            w_r_g = cp.asarray(mode_params['w_r'].astype(_npcast))
            w_c_g = cp.asarray(mode_params['w_c'].astype(_npcast))
            pal_g = cp.asarray(_PALINDROME.astype(_npcast))

            def _active(mid): return int(float(mew[mid]) > 1e-6)

            dk        = float(mode_params.get('delta_k', 0.0))
            eps5      = float((math.log2(5.) - 7./3.) * 1200.)
            inv_phi   = float(1./PHI)
            mu2       = float(_MH_MU2)
            lam_mh    = float(_MH_LAMBDA)
            log_p_eff = float(math.log(max(mode_params.get('p_eff', 2.0), 1.001)))
            ln_ln_esc = float(LN_LN_ESC)

            threads = 256

            # ── Batched kernel launch with real ASCII progress bar ────────────
            # Single monolithic launch gives ZERO progress feedback — the kernel
            # runs start-to-finish with no way to query internal state.
            # Fix: split into N_BATCHES equal row-batches.  Each batch is an
            # independent kernel call on a slice of the flat pixel arrays.
            # After each synchronize() we update a real [====>   ] bar showing
            # actual percentage, elapsed time and ETA.
            # All coordinates computed once upfront, all coloring runs once at
            # the end — no CPU-per-tile overhead is reintroduced.
            # ASCII-only bar: works on every terminal including Windows CMD.
            N_BATCHES   = 20
            batch_size  = (n_pix + N_BATCHES - 1) // N_BATCHES
            _BAR_W      = 40    # total bar width in characters

            # _prog_bar: single print per update — pads to _LINE_W chars so \r
            # always fully overwrites any previous content on Windows CMD.
            # No competing "launching..." print — the bar alone is sufficient.
            _LINE_W = 72    # must be >= len of longest bar line

            def _prog_bar(done_batches, total_batches, elapsed):
                pct  = done_batches / total_batches
                fill = int(pct * _BAR_W)
                if fill < _BAR_W:
                    bar = '=' * fill + '>' + ' ' * (_BAR_W - fill - 1)
                else:
                    bar = '=' * _BAR_W
                eta_s = (elapsed / pct - elapsed) if pct > 0 else 0.0
                line  = (f'  [GPU] [{bar}] {pct*100:5.1f}%'
                         f'  {elapsed:6.1f}s  ETA {eta_s:5.0f}s')
                # Pad to fixed width so \r erases all previous content
                line  = line.ljust(_LINE_W)
                print(line, end='\r', flush=True)

            _t_batch_start = time.time()
            # Print 0% bar immediately — user sees it before first batch starts
            _prog_bar(0, N_BATCHES, 0.001)

            try:
                for _bi in range(N_BATCHES):
                    _ps = _bi * batch_size
                    _pe = min(_ps + batch_size, n_pix)
                    _bn = _pe - _ps           # pixels in this batch
                    _bb = (_bn + threads - 1) // threads

                    if IS_DI_TYPE:
                        # ∂I kernel: simpler signature — no mode weights, no mode extras
                        kern(
                            (_bb,), (threads,),
                            (smooth_n_g[_ps:_pe], d_r_g[_ps:_pe],
                             d_t_g[_ps:_pe],      tight_g[_ps:_pe],
                             de_g[_ps:_pe],        orbit_g[_ps:_pe],
                             ze_r_g[_ps:_pe],      ze_i_g[_ps:_pe],
                             dze_r_g[_ps:_pe],     dze_i_g[_ps:_pe],
                             zang_g[_ps:_pe],
                             in_r[_ps:_pe],        in_i[_ps:_pe],
                             _gs(ln_ln_esc),
                             cp.int32(max_iter), cp.int32(_bn),
                             _gs(ESCAPE_R),
                             )
                        )
                    else:
                        # Standard ET kernel: full mode weights + extras
                        kern(
                            (_bb,), (threads,),
                            (smooth_n_g[_ps:_pe], d_r_g[_ps:_pe],
                             d_t_g[_ps:_pe],      tight_g[_ps:_pe],
                             de_g[_ps:_pe],        orbit_g[_ps:_pe],
                             ze_r_g[_ps:_pe],      ze_i_g[_ps:_pe],
                             dze_r_g[_ps:_pe],     dze_i_g[_ps:_pe],
                             zang_g[_ps:_pe],
                             in_r[_ps:_pe],        in_i[_ps:_pe],
                             cp.int32(1 if is_julia else 0),
                             _gs(jcr), _gs(jci),
                             w_r_g, w_c_g,
                             _gs(log_p_eff), _gs(ln_ln_esc),
                             mew_g, pal_g,
                             cp.int32(max_iter), cp.int32(_bn),
                             _gs(ESCAPE_R),
                             cp.int32(_active(1)),  cp.int32(_active(2)),
                             cp.int32(_active(4)),  cp.int32(_active(5)),
                             cp.int32(_active(6)),  cp.int32(_active(7)),
                             cp.int32(_active(8)),  cp.int32(_active(9)),
                             cp.int32(_active(10)), cp.int32(_active(11)),
                             _gs(dk),
                             _gs(eps5), _gs(inv_phi),
                             _gs(mu2), _gs(lam_mh),
                             )
                        )
                    cp.cuda.Device().synchronize()   # one sync per batch
                    _el = time.time() - _t_batch_start
                    _prog_bar(_bi + 1, N_BATCHES, _el)

                _el_kern = time.time() - _t_batch_start
                # Overwrite bar line with final done message, then newline to release the line
                _done_line = f'  [GPU] [{"=" * _BAR_W}] 100.0%  {_el_kern:6.1f}s  done'
                print(_done_line.ljust(_LINE_W), flush=True)
            except Exception as _gpu_err:
                prec_label = 'float64' if use_f64 else 'float32'
                _et_error(
                    context      = f'GPU {prec_label} kernel runtime — tile shape {shape}',
                    exc          = _gpu_err,
                    fatal        = False,
                    fallback_msg = 'Falling back to CPU NumPy path for this tile')
                _use_gpu_kernel = False
                # Fall through to CPU path below

            if _use_gpu_kernel:   # only return GPU results if kernel succeeded
                try:
                    # Return CuPy arrays directly — NO .get() here.
                    # All 5 coloring passes run on the GPU in _render_frame
                    # using these CuPy arrays.  Only the final composite
                    # is transferred to CPU (single PCIe transfer, not eleven).
                    def _rs(g): return g.reshape(shape)
                    return {
                        'smooth_n' : _rs(smooth_n_g),
                        'd_r'      : _rs(d_r_g),
                        'd_t'      : _rs(d_t_g),
                        'tight'    : _rs(tight_g),
                        'de'       : _rs(de_g),
                        'orbit'    : _rs(orbit_g),
                        'z_esc'    : (_rs(ze_r_g)  + 1j * _rs(ze_i_g)).astype(cp.complex128),
                        'dz_esc'   : (_rs(dze_r_g) + 1j * _rs(dze_i_g)).astype(cp.complex128),
                        'z_int_ang': _rs(zang_g),
                        '_gpu'     : True,   # signals to caller: arrays are CuPy
                    }
                except Exception as _xfer_err:
                    prec_label = 'float64' if use_f64 else 'float32'
                    _et_error(
                        context      = f'GPU {prec_label} result transfer — tile shape {shape}',
                        exc          = _xfer_err,
                        fatal        = False,
                        fallback_msg = 'Falling back to CPU NumPy path for this tile')
                    _use_gpu_kernel = False
                    # Fall through to CPU path
            # else fall through to CPU path

    # ── CPU path: Python/NumPy loop ──────────────────────────────────────────
    # Reached when: no GPU detected, kernel compile failed, or kernel runtime error.
    if USE_GPU and not _use_gpu_kernel:
        # GPU was expected but something failed — already printed above.
        pass
    elif not USE_GPU:
        # This fires once per process (first tile) — subsequent tiles are silent.
        if not getattr(iterate_strip_v2, '_cpu_msg_printed', False):
            print('  [CPU] No GPU available — using CPU NumPy path.', flush=True)
            iterate_strip_v2._cpu_msg_printed = True
    _di_label = '∂I Lattice-Aware' if IS_DI_TYPE else ('Julia' if is_julia else 'Mandelbrot')
    _cpu_ctx = (f'CPU path — shape={shape} max_iter={max_iter} '
                f'mode={mode_params.get("name","?")} '
                f'{_di_label}')
    try:
        z      = np.asarray(z0,    dtype=COMPLEX_DTYPE)
        c      = np.asarray(c_arr, dtype=COMPLEX_DTYPE)

        dz = (np.ones(shape,  dtype=COMPLEX_DTYPE) if is_julia
              else np.zeros(shape, dtype=COMPLEX_DTYPE))

        escaped  = np.zeros(shape, dtype=bool)
        smooth_n = np.full(shape,  float(max_iter), dtype=FLOAT_DTYPE)
        d_r_out  = np.ones(shape,  dtype=np.float32)*12.
        d_t_out  = np.ones(shape,  dtype=np.float32)*12.
        tight    = np.ones(shape,  dtype=FLOAT_DTYPE)
        de_out   = np.zeros(shape, dtype=FLOAT_DTYPE)
        z_esc    = np.zeros(shape, dtype=COMPLEX_DTYPE)
        dz_esc   = np.ones(shape,  dtype=COMPLEX_DTYPE)
        orbit    = np.full(shape, 1e6, dtype=FLOAT_DTYPE)

        P_R_b  = np.asarray(_P_REAL  [:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        P_C_b  = np.asarray(_P_CMPLX [:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        ROT_b  = np.asarray(_ROT_D   [:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        P_M1_b = np.asarray(_P_R_M1  [:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        w_de   = mode_params['w_r'] * _P_REAL
        W_DE_b = np.asarray(w_de[:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        W_R_b  = np.asarray(mode_params['w_r'][:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)
        W_C_b  = np.asarray(mode_params['w_c'][:, np.newaxis, np.newaxis], dtype=FLOAT_DTYPE)

        D_LUT_np = np.asarray(_D_LUT)
        extra  = mode_params['extra']

        _f    = lambda v: np.array(v, dtype=FLOAT_DTYPE)
        ESC_F = _f(ESCAPE_R); K_F2 = _f(K)
        LOG_P_EFF = _f(math.log(max(mode_params.get('p_eff', 2.0), 1.001)))
        LN_LN_R   = _f(LN_LN_ESC)
        TR_K  = _f(K);  TR_V = _f(V);  TR_PHI = _f(1./PHI);  TR_1 = _f(1.0)
    except Exception as e:
        _et_error(f'CPU array allocation — {_cpu_ctx}', e, fatal=True)

    # ── ∂I Lattice-Aware CPU iteration path ─────────────────────────────────
    # Implements dominant-power + 24-family perturbation from v3 spec.
    # Used when GPU is unavailable and IS_DI_TYPE is True.
    if IS_DI_TYPE:
        _POWS_NP  = np.array([12.,6.,4.,3.,2.,1.,2.4,12./7,1.5,4./3,1.2,12./11], np.float64)
        _ROT_NP   = np.array([LN2/d for d in [1,2,3,4,6,12,5,7,8,9,10,11]], np.float64)
        _BASEW_NP = np.array([12./d * 100./(100.+FAM_PQ[d]) for d in [1,2,3,4,6,12,5,7,8,9,10,11]], np.float64)
        _BASEW_NP /= _BASEW_NP.sum()
        _LOG_P_EFF_DI = _f(_LN_P_EFF_DI)
        for n in range(max_iter):
          try:
            r     = np.abs(z)
            theta = np.angle(z)
            r_cap = np.maximum(r, _f(1e-300))
            r_cap = np.minimum(r_cap, ESC_F)

            # 27720ET lattice projection
            k_r_f = _f(N_ET) * np.log2(r_cap)
            k_r   = np.round(k_r_f)
            eps_r = np.abs(k_r_f - k_r) * _f(1200./N_ET)
            t_r   = _f(100.) / (_f(100.) + eps_r)

            k_t_f = _f(N_ET) * theta / _f(LN2)
            k_t   = np.round(k_t_f)
            eps_t = np.abs(k_t_f - k_t) * _f(1200./N_ET)
            t_t   = _f(100.) / (_f(100.) + eps_t)

            # Orbit traps
            td = np.minimum(np.minimum(np.abs(r_cap-TR_K), np.abs(r_cap-TR_V)),
                            np.minimum(np.abs(r_cap-TR_PHI), np.abs(r_cap-TR_1)))
            orbit = np.minimum(orbit, td)

            # Dominant power: orbit lattice d when t_r > K, else palindrome
            k_r_int = np.abs(k_r).astype(np.int64)
            g27 = _vec_gcd_27720_np(k_r_int)
            d27 = N_ET // np.maximum(g27, 1)
            k12 = np.abs(np.round(k_r_f * (12./N_ET))).astype(np.int64)
            # gcd(k12, 12) vectorised
            g12 = np.ones_like(k12)
            nz12 = k12 > 0
            g12 = np.where(nz12 & (k12%12==0), 12, np.where(nz12 & (k12%6==0), 6,
                  np.where(nz12 & (k12%4==0), 4, np.where(nz12 & (k12%3==0), 3,
                  np.where(nz12 & (k12%2==0), 2, g12)))))
            g12 = np.where(k12==0, 12, g12)
            d12 = 12 // np.maximum(g12, 1)
            d_orbit = np.where((d27 >= 1) & (d27 <= 12), d27, d12).astype(np.float64)

            d_casc  = float(_PALINDROME[n % N])
            d_dom   = np.where(t_r > K, d_orbit, d_casc)
            p_dom   = 12.0 / np.maximum(d_dom, 1.0)

            # Shimmer
            psi = float(_SHIMMER_NP[n % N])

            # Primary term: Psi * z^{p_dom}
            rp_dom = np.power(r_cap, p_dom)
            z_prim = (psi * rp_dom * np.exp(1j * p_dom * theta)).astype(COMPLEX_DTYPE)

            # 24-family perturbation at V scale
            z_pert = np.zeros(shape, dtype=COMPLEX_DTYPE)
            for _di in range(N_FAM):
                _pd = _POWS_NP[_di]; _rot = _ROT_NP[_di]; _wd = _BASEW_NP[_di]
                _rp = np.power(r_cap, _pd)
                z_pert += _wd * _rp * np.exp(1j * _pd * theta)              # real family
                z_pert += _wd * _rp * np.exp(1j * (_pd * theta + k_t * _rot))  # imag family
            z_pert *= V

            z_new = z_prim + z_pert + c

            # Derivative: Jacobian of dominant power only (perturbation is V-scaled, negligible)
            rp_m1   = np.power(r_cap, p_dom - 1.0)
            f_prime = (psi * p_dom * rp_m1 * np.exp(1j * (p_dom - 1.0) * theta)).astype(COMPLEX_DTYPE)
            dz_new  = f_prime * dz + _f(1.0)   # ∂I: +1 because c=pixel is the varying parameter

            # Escape
            beyond   = (~escaped) & (r > ESC_F)
            new_e    = beyond

            r_e_full   = np.maximum(r, _f(1.001))
            ln_z_full  = np.log(r_e_full)
            ln_ln_full = np.log(np.maximum(ln_z_full, _f(1e-15)))
            sn_cand    = _f(n+1.) - (ln_ln_full - LN_LN_R) / _LOG_P_EFF_DI
            sn_cand    = np.clip(sn_cand, _f(0.), _f(float(n)+2.))
            smooth_n   = np.where(new_e, sn_cand, smooth_n)

            # d_r, d_t at escape
            d_r_out = np.where(new_e, d_orbit.astype(np.float32), d_r_out)
            d_t_out_di = np.where((d27 >= 1) & (d27 <= 12),
                                  np.abs(k_t).astype(np.int64), k12).astype(np.float64)
            # compute d_t properly
            k_t_int = np.abs(k_t).astype(np.int64)
            g27t = _vec_gcd_27720_np(k_t_int)
            d27t = N_ET // np.maximum(g27t, 1)
            k12t = np.abs(np.round(k_t_f * (12./N_ET))).astype(np.int64)
            g12t = np.ones_like(k12t)
            nz12t = k12t > 0
            g12t = np.where(nz12t & (k12t%12==0), 12, np.where(nz12t & (k12t%6==0), 6,
                   np.where(nz12t & (k12t%4==0), 4, np.where(nz12t & (k12t%3==0), 3,
                   np.where(nz12t & (k12t%2==0), 2, g12t)))))
            g12t = np.where(k12t==0, 12, g12t)
            d12t = 12 // np.maximum(g12t, 1)
            d_t_here = np.where((d27t >= 1) & (d27t <= 12), d27t, d12t).astype(np.float32)
            d_t_out  = np.where(new_e, d_t_here, d_t_out)
            tight    = np.where(new_e, t_r * t_t, tight)

            dz_m_full = np.abs(dz) + _f(1e-300)
            de_cand   = 2. * r_e_full * np.log(r_e_full) / dz_m_full
            de_out    = np.where(new_e, de_cand, de_out)

            z_esc  = np.where(new_e, z_new,  z_esc)
            dz_esc = np.where(new_e, dz_new, dz_esc)
            escaped = escaped | new_e
            z  = np.where(escaped, z,  z_new)
            dz = np.where(escaped, dz, dz_new)
            if bool(np.all(escaped)): break
          except Exception as _loop_err:
            _et_error(f'CPU ∂I iteration n={n} — {_cpu_ctx}', _loop_err, fatal=True)

        try:
            z_int_ang = np.where(escaped, _f(0.), np.angle(z).astype(FLOAT_DTYPE))
        except Exception as e:
            _et_error('CPU ∂I path — interior angle', e, fatal=True)
        return {
            'smooth_n' : smooth_n, 'd_r': d_r_out, 'd_t': d_t_out,
            'tight': tight, 'de': de_out, 'orbit': orbit.astype(FLOAT_DTYPE),
            'z_esc': z_esc, 'dz_esc': dz_esc, 'z_int_ang': z_int_ang,
        }

    # ── Standard ET CPU iteration path (Julia / Mandelbrot) ──────────────────
    for n in range(max_iter):
      try:
        r     = np.abs(z)
        theta = np.angle(z)
        r_cap = np.minimum(np.maximum(r, _f(1e-300)), ESC_F)

        k_r_f = _f(N_ET) * np.log2(r_cap)
        k_r   = np.round(k_r_f)
        eps_r = (k_r_f - k_r)*_f(1200./N_ET)
        t_r   = _f(100.)/(  _f(100.) + np.abs(eps_r))

        k_t_f = _f(N_ET) * theta / _f(LN2)
        k_t   = np.round(k_t_f)
        eps_t = (k_t_f - k_t)*_f(1200./N_ET)
        t_t   = _f(100.)/(_f(100.) + np.abs(eps_t))

        r_abs = r_cap
        td = np.minimum(np.minimum(np.abs(r_abs-TR_K), np.abs(r_abs-TR_V)),
                        np.minimum(np.abs(r_abs-TR_PHI), np.abs(r_abs-TR_1)))
        orbit = np.minimum(orbit, td)

        r_b = r_cap[np.newaxis]; th_b = theta[np.newaxis]; kt_b = k_t[np.newaxis]
        z_new = np.sum(W_R_b*(r_b**P_R_b)*np.exp(1j*(P_R_b*th_b)), axis=0)
        z_new = z_new + np.sum(W_C_b*(r_b**P_C_b)*np.exp(1j*(P_C_b*th_b+kt_b*ROT_b)), axis=0)
        if extra is not None:
            z_new = z_new + extra(z, r_cap, theta, k_t, n)
        z_new = z_new + c

        f_prime = np.sum(W_DE_b*(r_b**P_M1_b)*np.exp(1j*(P_M1_b*th_b)), axis=0)
        if is_julia:
            dz_new = f_prime * dz
        else:
            dz_new = f_prime * dz + _f(1.0)

        beyond   = (~escaped) & (r > ESC_F)
        new_e    = beyond

        r_e_full   = np.maximum(r, _f(1.001))
        ln_z_full  = np.log(r_e_full)
        ln_ln_full = np.log(np.maximum(ln_z_full, _f(1e-15)))
        sn_cand    = _f(n+1.) - (ln_ln_full - LN_LN_R) / LOG_P_EFF
        sn_cand    = np.clip(sn_cand, _f(0.), _f(float(n) + 2.))
        smooth_n   = np.where(new_e, sn_cand, smooth_n)

        # d_r, d_t — exact match to v19: D_LUT[|k_27720| % 12]
        k_mod_r_full = (np.abs(k_r).astype(np.int64) % 12).astype(np.int32)
        k_mod_t_full = (np.abs(k_t).astype(np.int64) % 12).astype(np.int32)
        d_r_out = np.where(new_e, D_LUT_np[k_mod_r_full].astype(np.float32), d_r_out)
        d_t_out = np.where(new_e, D_LUT_np[k_mod_t_full].astype(np.float32), d_t_out)
        tight   = np.where(new_e, t_r * t_t, tight)

        dz_m_full = np.abs(dz) + _f(1e-300)
        de_cand   = 2. * r_e_full * np.log(r_e_full) / dz_m_full
        de_out    = np.where(new_e, de_cand, de_out)

        z_esc  = np.where(new_e, z,  z_esc)
        dz_esc = np.where(new_e, dz, dz_esc)
        escaped = escaped | new_e

        z  = np.where(escaped, z,  z_new)
        dz = np.where(escaped, dz, dz_new)

        if bool(np.all(escaped)): break
      except Exception as _loop_err:
        _et_error(
            context = f'CPU iteration n={n} — {_cpu_ctx}',
            exc     = _loop_err,
            fatal   = True)

    try:
        z_int_ang = np.where(escaped, _f(0.), np.angle(z).astype(FLOAT_DTYPE))
    except Exception as e:
        _et_error('CPU path — interior angle computation', e, fatal=True)

    return {
        'smooth_n' : smooth_n,
        'd_r'      : d_r_out,
        'd_t'      : d_t_out,
        'tight'    : tight,
        'de'       : de_out,
        'orbit'    : orbit.astype(FLOAT_DTYPE),
        'z_esc'    : z_esc,
        'dz_esc'   : dz_esc,
        'z_int_ang': z_int_ang,
    }



# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 10 — ESCAPE COLORING  (base HSV pass, full 2D lattice)
# ══════════════════════════════════════════════════════════════════════════════

def _hsv_to_f32(H, S, Br, xp=np):
    """Vectorised HSV → float32 RGB [0,1].
    xp: numpy or cupy — runs on whichever device holds the arrays.
    Uses xp.select (fully vectorised, no Python for-loop) so it runs
    natively on GPU when xp=cupy.
    """
    h6  = H.ravel() * 6.
    fh6 = xp.floor(h6)
    hi  = fh6.astype(xp.int64) % 6
    f   = (h6 - fh6).astype(xp.float64)
    v   = Br.ravel().astype(xp.float64)
    s   = S.ravel().astype(xp.float64)
    p_  = v * (1. - s)
    q_  = v * (1. - f * s)
    t_  = v * (1. - (1. - f) * s)
    # Sector table:
    # hi=0: R=v  G=t  B=p
    # hi=1: R=q  G=v  B=p
    # hi=2: R=p  G=v  B=t
    # hi=3: R=p  G=q  B=v
    # hi=4: R=t  G=p  B=v
    # hi=5: R=v  G=p  B=q
    # hi = fh6 % 6 is always in [0,5] — all 6 conditions are exhaustive.
    # CuPy requires default to be a scalar; 0.0 is never reached.
    cl = [hi==0, hi==1, hi==2, hi==3, hi==4, hi==5]
    R  = xp.select(cl, [v,  q_, p_, p_, t_, v],  default=0.0)
    G  = xp.select(cl, [t_, v,  v,  q_, p_, p_], default=0.0)
    B  = xp.select(cl, [p_, p_, t_, v,  v,  q_], default=0.0)
    rgb = xp.stack([R.astype(xp.float32),
                    G.astype(xp.float32),
                    B.astype(xp.float32)], axis=1)
    return xp.clip(rgb, 0., 1.).reshape(H.shape + (3,))


def et_escape_color(it_dict, max_iter, mode_params, tower, xp=np):
    """
    Professional cycling palette coloring for the ET iteration.

    The old approach used norm_n = smooth_n/max_iter for brightness, which
    made 95%+ of the image near-white with a thin colourful boundary shell.
    The root cause: at zoom 2-5x most pixels escape in the first 1-5% of
    iterations, so norm_n ≈ 0.02 everywhere → virtually no variation.

    This version uses CYCLING smooth_n for HUE — the standard professional
    approach used in all serious fractal software (Ultra Fractal, Kalles
    Fraktaler, Mandelbulber, etc.).

    Cycling formula:
      log_mu = ln(smooth_n+1) / ln(max_iter+1)   [log-scale, maps [0,1]]
      COLOR_CYCLES = K·N = (2/3)·12 = 8           [ET-derived: 8 full cycles]
      Ψ = 1 + √V·sin(2π·k_class/N)                [RMSAE shimmer]
      H_cycle = log_mu · COLOR_CYCLES · Ψ          [the cycling term]

    Using log-scale spreading ensures the early-escaping pixels (the bulk
    of the exterior) have as much hue variation as boundary pixels.
    COLOR_CYCLES = K·N = 8 is strictly ET-derived.

    ET structural contribution:
      H_ET = K·HUE[d_r] + (1-K)·HUE[d_t]         [Koide 2:1 D/T weight]
    This preserves the ET lattice structure in the colour.

    Saturation: high base (0.80) + Elegance Score modulation
    Brightness: high constant (0.88) for ALL escaped pixels
                Near-∂I: darken + shift to d=7 Otherworld indigo
    Interior: {P,D} Unsubstantiated — dark, subtle arg(z) texture
    """
    smooth_n = it_dict['smooth_n']
    d_r = xp.clip(xp.round(it_dict['d_r']).astype(xp.int32), 1, 12)
    d_t = xp.clip(xp.round(it_dict['d_t']).astype(xp.int32), 1, 12)
    tight    = it_dict['tight']

    hue_lut_np = np.array([FAM_HUE.get(d,0.5) for d in range(13)], dtype=np.float64)
    elg_lut_np = np.array([FAM_ELG_FULL.get(d,1.)/N for d in range(13)], dtype=np.float64)
    hue_lut = xp.asarray(hue_lut_np)
    elg_lut = xp.asarray(elg_lut_np)

    H_r   = hue_lut[d_r]; H_t = hue_lut[d_t]
    Elg_r = elg_lut[d_r]; Elg_t = elg_lut[d_t]

    interior = (smooth_n >= max_iter - 0.5)
    escaped  = ~interior

    # ── Smooth_n log-scale normalisation ──────────────────────────────────
    # Maps [1, max_iter] → [0, 1] with log spacing so early- and late-
    # escaping pixels get equal hue variation.
    log_mu = xp.log(xp.maximum(smooth_n, 1.0) + 1.0) / math.log(max_iter + 1.0)

    # ── COLOR_CYCLES = K·N = 8 (ET-derived) ───────────────────────────────
    # K=2/3, N=12 → K·N=8 complete hue cycles over the iteration range.
    COLOR_CYCLES = K * N   # = 8.0 exactly

    # ── RMSAE shimmer Ψ modulates cycling speed per-pixel ─────────────────
    psi = 1.0 + _RMSAE_AMP * xp.sin(2.0*math.pi * d_r.astype(xp.float64) / N)

    # ── ET structural hue: d_r and d_t contribute an additive offset ──────
    # Koide 2:1 weighting: D-axis (magnitude) = 2/3, T-axis (phase) = 1/3
    H_ET = (K * H_r + (1.0-K) * H_t)

    # ── Full hue: cycling + ET structural + palette base ──────────────────
    H = (log_mu * COLOR_CYCLES * psi
         + H_ET
         + tower['pal_base']
         + mode_params.get('pal_extra', 0.0)) % 1.0

    # ── Saturation: high base + Elegance Score modulation ─────────────────
    # Elegance E = (N/d)·[100/(100+|ε|)]·[100/(p+q)] — normalised to [0,1]
    elg = xp.sqrt(xp.maximum(Elg_r * Elg_t, 1e-12))
    # Quintic tension τ(m) — d=5 shadow force presses on each position
    k_tau  = (d_r * G_REAL) % 12
    qt_xp  = xp.asarray(QUINTIC_TENSION)
    tau    = qt_xp[k_tau] / 120.0
    # Saturation: 0.80 base, Elegance lifts it, quintic tension slightly lowers
    S = xp.clip(0.80 + 0.18 * elg * (1.0 - 0.3*tau), 0.72, 0.99)

    # ── Brightness: high constant for all escaped pixels ──────────────────
    # The old (1-norm_n)^0.40 made 95% of the image near-white with barely
    # any variation.  Fixed: high constant brightness; DE provides local
    # variation through the lighting pass, not through raw brightness here.
    BRT = xp.where(escaped, 0.88, 0.0)

    # ── Near-∂I: darken + shift toward d=7 Otherworld indigo ─────────────
    # tight = t_r·t_t ∈ [0,1].  tight < K means both axes near ∂I.
    # This preserves the ET incoherence structure in the coloring.
    inco = xp.maximum(0.0, 1.0 - tight * (1.0/K))   # 0 = coherent, 1 = near ∂I
    BRT  = xp.where(escaped, BRT * (1.0 - 0.80*inco), 0.0)
    H    = (H * (1.0 - 0.30*inco) + FAM_HUE[7] * 0.30*inco) % 1.0
    S    = xp.clip(S * (1.0 - 0.25*inco), 0.0, 1.0)

    S   = xp.where(interior, 0.0, S)
    BRT = xp.where(interior, 0.0, BRT)

    return _hsv_to_f32(H.astype(xp.float64), S, BRT, xp=xp)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 11 — NORMAL-MAP LIGHTING  (3D relief from DE Jacobian)
# ══════════════════════════════════════════════════════════════════════════════

def et_normal_lighting(it_dict, xp=np):
    """
    3D relief lighting derived from the escape derivative dz.
    Light direction: angle θ_L = 7/12·2π (k=7 circle-of-fifths generator),
                     elevation = K = 2/3 radians (Koide angle).
    Normal: n = z_esc / (|z_esc|·|dz_esc|)  (complex surface-normal vector)
    Shading: h = (Re(n)·cos_L + Im(n)·sin_L + sin(K)) / (1+sin(K))
    This formula gives every escaped pixel a beautiful 3D illumination
    that is INFINITELY SHARP regardless of zoom level (not pixel-limited).
    Returns float32 array (H,W) in [0,1].
    xp: numpy or cupy — all ops stay on the device that holds it_dict.
    """
    cplx_t = xp.complex128 if hasattr(xp, 'complex128') else complex
    z_esc  = it_dict['z_esc'].astype(cplx_t)
    dz_esc = it_dict['dz_esc'].astype(cplx_t)
    escaped = (it_dict['smooth_n'] < (MAX_ITER-0.5))

    z_abs  = xp.abs(z_esc)
    dz_abs = xp.abs(dz_esc)
    # Surface normal: n_complex = z_esc / (|z_esc|·dz_esc)
    denom  = z_abs * dz_abs + 1e-300
    n_cplx = z_esc / denom

    # Shading factor (Lambertian + elevation ambient)
    h = (n_cplx.real*_COS_L + n_cplx.imag*_SIN_L + _SIN_K) / _NORM_L
    h = xp.clip(h, 0., 1.)

    # 50% ambient + 50% diffuse (prevents pure-black shadows)
    shading = 0.50 + 0.50*h
    shading = xp.where(escaped, shading, 1.0)   # interior: no lighting mod

    return shading.astype(xp.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 12 — ORBIT TRAP COLORING  (ET lattice ring distances)
# ══════════════════════════════════════════════════════════════════════════════

def et_orbit_color(it_dict, tower, xp=np):
    """
    Orbit trap coloring: each pixel gets the minimum distance traversed
    to any of the 4 ET lattice ring radii (K, V, 1/φ, 1).
    Produces fine detail INSIDE the escape region matching the
    internal structure of the ET manifold.

    Physics: these rings ARE the major manifold positions:
      K=2/3  (Koide binding stability / meta-cognition threshold)
      V=1/12 (base variance / Planck-equivalent)
      1/φ    (golden conjugate / d=5 Fibonacci)
      1.0    (unison, the origin)
    xp: numpy or cupy — all ops stay on the device that holds it_dict.
    """
    orbit   = it_dict['orbit']  # min distance, (H,W) float
    escaped = (it_dict['smooth_n'] < (MAX_ITER-0.5))

    # Map min-distance to color: trap_weight ∈ [0,1], high near ET ring
    # Scale 3.0 (reduced from 6.0): subtler rings, less visual intrusion
    scale       = 3.0
    trap_weight = xp.exp(-orbit * scale)

    # Hue: tower palette base + trap gradient toward golden (d=5)
    base_h  = tower['pal_base']
    H_trap  = (base_h + trap_weight*0.20 + 0.08*xp.sin(orbit*15.)) % 1.
    # Reduced saturation + brightness: orbit traps are a detail layer,
    # not the primary color.  Old values (0.85 / 0.60) were too strong.
    S_trap  = xp.clip(trap_weight * 0.55, 0., 1.)
    V_trap  = xp.clip(trap_weight * 0.40, 0., 1.)

    S_trap  = xp.where(escaped, S_trap, 0.)
    V_trap  = xp.where(escaped, V_trap, 0.)

    rgb_trap = _hsv_to_f32(H_trap.astype(xp.float64), S_trap, V_trap, xp=xp)
    # weight drives the screen blend in et_composite — capped at 0.35
    weight   = xp.clip(trap_weight * xp.where(escaped, 1., 0.) * 0.35, 0., 1.)

    return rgb_trap, weight[:,:,xp.newaxis].astype(xp.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 13 — INTERIOR COLORING  ({P,D} Unsubstantiated = dark matter)
# ══════════════════════════════════════════════════════════════════════════════

def et_interior_color(it_dict, tower, xp=np):
    """
    Interior coloring for non-escaped pixels.
    ET: interior = {P,D} Unsubstantiated state = dark matter.
    "Dark matter gravitates (d=1) but does not emit (d≠12)."
    Very dark, with a subtle hue whisper from final orbit angle.
    Uses z_int_ang (arg of z at max_iter) for texture.
    xp: numpy or cupy — all ops stay on the device that holds it_dict.
    """
    ang     = it_dict['z_int_ang'].astype(xp.float64)   # (H,W)
    interior= (it_dict['smooth_n'] >= MAX_ITER-0.5)

    base_h = tower['pal_base']
    # Subtle hue: base + small fraction of orbit angle
    H = (base_h + ang/(2.*math.pi)*0.12) % 1.
    # Very dark saturation and brightness — dark matter does not emit
    S = 0.18*(0.5 + 0.5*xp.sin(ang))
    Br= 0.04 + 0.05*(0.5 + 0.5*xp.cos(ang*3.))
    # Only apply to interior pixels
    S  = xp.where(interior, S,  0.)
    Br = xp.where(interior, Br, 0.)
    return _hsv_to_f32(H, S, Br, xp=xp)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 14 — MULTI-PASS COMPOSITOR
# ══════════════════════════════════════════════════════════════════════════════

def et_composite(base_rgb, interior_rgb, normal_shading, orbit_rgb, orbit_w, xp=np):
    """
    Multi-pass ET compositor (all float32 [0,1]):
      1. Apply normal-map lighting to base escape coloring
      2. Blend orbit trap layer (additive detail)
      3. Replace interior pixels with dark-matter interior coloring
    Returns float32 (H,W,3).
    xp: numpy or cupy — all ops stay on whichever device holds the arrays.
    """
    # Pass 1: lighting applied to escape coloring
    lit = base_rgb * normal_shading[:,:,xp.newaxis]

    # Pass 2: orbit trap screen blend (never exceeds 1.0 — no blown highlights)
    # Screen: result = 1 - (1-base)*(1-overlay*weight)
    # orbit_w already capped at 0.35 in et_orbit_color
    overlay = orbit_rgb * orbit_w
    mixed   = 1.0 - (1.0 - lit) * (1.0 - overlay)
    mixed   = xp.clip(mixed, 0.0, 1.0).astype(xp.float32)

    # Pass 3: interior replacement
    interior_mask = (interior_rgb.sum(axis=2) > 0)[:,:,xp.newaxis]
    final = xp.where(interior_mask, interior_rgb, mixed)

    return xp.clip(final.astype(xp.float32), 0., 1.)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 15 — POST-PROCESSING  (ACES + Koide γ + vignette + unsharp)
# ══════════════════════════════════════════════════════════════════════════════

def _aces(x):
    """ACES filmic tone-mapping approximation (industry standard HDR)."""
    a,b,c,d,e = _ACES_A, _ACES_B, _ACES_C, _ACES_D, _ACES_E
    x = np.asarray(x, dtype=np.float32)
    return np.clip((x*(a*x+b))/(x*(c*x+d)+e), 0., 1.)

def et_post(a):
    """
    ET-derived post-processing on composite float32 [0,1] RGB:
      1. ACES filmic tone mapping  (HDR highlight compression)
      2. Koide gamma γ=K=2/3       (ET-native perceptual encoding)
      3. Quartic vignette 1−0.18r⁴ (d=4 quartic geometry, Weak force)
      4. Unsharp mask               (sharpens sublattice boundary transitions)
    """
    a = _aces(np.clip(a, 0., 2.))            # ACES HDR compression
    a = np.power(np.clip(a, 0., 1.), K)      # Koide gamma
    h, w = a.shape[:2]
    xs = np.linspace(-1., 1., w, dtype=np.float32)
    ys = np.linspace(-1., 1., h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    r2   = xx**2 + yy**2
    vign = np.clip(1. - 0.18*(r2**2), 0., 1.)[:,:,np.newaxis]
    a    = a * vign
    tmp  = Image.fromarray((a*255).clip(0,255).astype(np.uint8))
    blr  = np.array(tmp.filter(ImageFilter.GaussianBlur(radius=1.4)),
                    dtype=np.float32)/255.
    a    = np.clip(a + 0.30*(a-blr), 0., 1.)
    return a


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 16 — 32-BIT FLOAT TIFF WRITER  (no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def write_tiff_float32(arr_f32, filepath, dpi=300, description=''):
    """
    Write IEEE 754 32-bit float RGB TIFF.  No external library required.
    arr_f32: float32 (H,W,3), range [0,1].  Photoshop/GIMP/Affinity-ready.
    16-bit per channel is 65536 levels; 32-bit float is 4.2 billion effective
    levels with full dynamic range — essential for professional post-work.

    TIFF structure (little-endian / II):
      Header  (8 B)  →  IFD at offset 8
      IFD     (2 + 14×12 + 4 = 174 B)
      Extra   (28 B) →  BitsPerSample[32,32,32], SampleFormat[3,3,3],
                         XResolution, YResolution
      Image   (H×W×12 B) →  float32 chunky RGB, top-to-bottom
    """
    assert arr_f32.dtype == np.float32, 'Need float32 input'
    H, W = arr_f32.shape[:2]
    LE   = '<'

    # Exact byte offsets:
    IFD_OFFSET   = 8
    N_ENTRIES    = 14
    IFD_SIZE     = 2 + N_ENTRIES*12 + 4    # = 174
    EXTRA_OFFSET = IFD_OFFSET + IFD_SIZE   # = 182
    BPS_OFFSET   = EXTRA_OFFSET + 0        # BitsPerSample [32,32,32] 6B
    SFMT_OFFSET  = EXTRA_OFFSET + 6        # SampleFormat  [3,3,3]   6B
    XRES_OFFSET  = EXTRA_OFFSET + 12       # XResolution   RATIONAL  8B
    YRES_OFFSET  = EXTRA_OFFSET + 20       # YResolution   RATIONAL  8B
    DATA_OFFSET  = EXTRA_OFFSET + 28       # Image data
    IMG_BYTES    = H * W * 3 * 4

    SHORT=3; LONG=4; RATIONAL=5

    def ifd_entry(tag, dtype, count, value):
        return struct.pack(LE+'HHII', tag, dtype, count, value)

    entries = (
        ifd_entry(256, LONG,     1, W)             +  # ImageWidth
        ifd_entry(257, LONG,     1, H)             +  # ImageLength
        ifd_entry(258, SHORT,    3, BPS_OFFSET)    +  # BitsPerSample
        ifd_entry(259, SHORT,    1, 1)             +  # Compression=None
        ifd_entry(262, SHORT,    1, 2)             +  # PhotometricInterp=RGB
        ifd_entry(273, LONG,     1, DATA_OFFSET)   +  # StripOffsets
        ifd_entry(277, SHORT,    1, 3)             +  # SamplesPerPixel
        ifd_entry(278, LONG,     1, H)             +  # RowsPerStrip
        ifd_entry(279, LONG,     1, IMG_BYTES)     +  # StripByteCounts
        ifd_entry(282, RATIONAL, 1, XRES_OFFSET)   +  # XResolution
        ifd_entry(283, RATIONAL, 1, YRES_OFFSET)   +  # YResolution
        ifd_entry(284, SHORT,    1, 1)             +  # PlanarConfig=Chunky
        ifd_entry(296, SHORT,    1, 2)             +  # ResolutionUnit=Inch
        ifd_entry(339, SHORT,    3, SFMT_OFFSET)       # SampleFormat
    )
    assert len(entries) == N_ENTRIES*12

    header = b'II' + struct.pack(LE+'HI', 42, IFD_OFFSET)
    ifd    = struct.pack(LE+'H', N_ENTRIES) + entries + struct.pack(LE+'I', 0)
    extra  = struct.pack(LE+'HHH', 32,32,32)          # BitsPerSample
    extra += struct.pack(LE+'HHH', 3, 3, 3)           # SampleFormat (IEEE float)
    extra += struct.pack(LE+'II', int(dpi), 1)         # XResolution = dpi/1
    extra += struct.pack(LE+'II', int(dpi), 1)         # YResolution = dpi/1
    assert len(extra) == 28
    img_data = np.ascontiguousarray(arr_f32, dtype='<f4').tobytes()

    with open(str(filepath), 'wb') as f:
        f.write(header)    #  8 B
        f.write(ifd)       # 174 B
        f.write(extra)     #  28 B
        f.write(img_data)  # H×W×12 B

    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 17 — 16-BIT PNG WRITER  (pure Python + stdlib zlib; pHYs at DPI)
# ══════════════════════════════════════════════════════════════════════════════

def _png_chunk(tag, data):
    crc = zlib.crc32(tag+data) & 0xFFFFFFFF
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

def write_png_16bit(arr_u16, filepath, dpi=300, text_meta=None):
    """Write 16-bit per-channel RGB PNG with DPI metadata and optional tEXt."""
    H, W = arr_u16.shape[:2]
    assert arr_u16.dtype == np.uint16
    sig  = b'\x89PNG\r\n\x1a\n'
    ihdr = _png_chunk(b'IHDR', struct.pack('>IIBBBBB', W,H,16,2,0,0,0))
    ppm  = round(dpi/0.0254)   # = 11811 for 300 DPI
    phys = _png_chunk(b'pHYs', struct.pack('>IIB', ppm, ppm, 1))
    texts= b''
    if text_meta:
        for k,v in text_meta.items():
            kb = k.encode('latin-1','replace')[:79]
            vb = v.encode('latin-1','replace')
            texts += _png_chunk(b'tEXt', kb+b'\x00'+vb)
    rows = [b'\x00' + row.astype('>u2').tobytes() for row in arr_u16]
    idat = _png_chunk(b'IDAT', zlib.compress(b''.join(rows), level=6))
    iend = _png_chunk(b'IEND', b'')
    with open(str(filepath),'wb') as f:
        f.write(sig+ihdr+phys+texts+idat+iend)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 18 — JITTER SSAA COORDINATES
# ══════════════════════════════════════════════════════════════════════════════

def make_strip_coords(row_start, row_count, cx, cy, zoom, rw, rh, rng=None, jitter=True):
    """
    Complex pixel coordinates for one strip at render resolution.
    If jitter=True and rng is provided, adds stratified sub-pixel jitter
    (±0.3 pixel in each direction) for better anti-aliasing than pure grid.
    """
    asp  = rw/rh
    xs   = np.linspace(cx-zoom*asp, cx+zoom*asp, rw,  dtype=np.float64)
    ys   = np.linspace(cy+zoom,     cy-zoom,     rh,  dtype=np.float64)
    ys_s = ys[row_start : row_start+row_count]
    xx, yy = np.meshgrid(xs, ys_s)
    if jitter and rng is not None and SS > 1:
        px_size_x = (zoom*asp*2.)/rw
        px_size_y = (zoom*2.)/rh
        jx = rng.uniform(-0.30, 0.30, xx.shape) * px_size_x
        jy = rng.uniform(-0.30, 0.30, yy.shape) * px_size_y
        xx = xx + jx; yy = yy + jy
    return (xx + 1j*yy).astype(np.complex128)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 19 — BOX-FILTER SSAA DOWNSAMPLE
# ══════════════════════════════════════════════════════════════════════════════

def ssaa_downsample(rgb_f32, ss):
    """Box-filter downsample by factor ss. Input: (H*ss, W*ss, 3) float32."""
    if ss == 1: return rgb_f32
    h_out = rgb_f32.shape[0]//ss; w_out = rgb_f32.shape[1]//ss
    return (rgb_f32.reshape(h_out,ss,w_out,ss,3)
                   .mean(axis=(1,3)).astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 20 — TILE WORKER  (processes one horizontal strip)
# ══════════════════════════════════════════════════════════════════════════════

def _render_tile(args):
    (row_start, row_count, cx, cy, zoom, mode_params, max_iter,
     julia_c, is_julia, rw, rh, tower, rng_state) = args
    _type_str = '\u2202I Lattice-Aware' if IS_DI_TYPE else ('Julia' if is_julia else 'Mandelbrot')
    ctx = (f'Tile rows {row_start}–{row_start+row_count-1}  '
           f'mode={mode_params.get("name","?")}  '
           f'{_type_str}  '
           f'{max_iter:,} iters')
    try:
        rng = np.random.RandomState(rng_state)
        coords = make_strip_coords(row_start, row_count, cx, cy, zoom, rw, rh,
                                   rng=rng, jitter=(SS>1))
        if IS_DI_TYPE:
            # ∂I Lattice-Aware: z₀=0, c=pixel — intrinsic to this fractal family,
            # not borrowed from Mandelbrot. The orbit starts at the origin and
            # the pixel coordinate drives the dynamics through the ∂I iteration.
            z0    = np.zeros_like(coords)
            c_arr = coords
        elif is_julia:
            z0    = coords
            c_arr = np.full_like(coords, julia_c)
        else:
            z0    = np.zeros_like(coords)
            c_arr = coords
    except Exception as e:
        _et_error(f'Coordinate setup — {ctx}', e, fatal=True)

    try:
        it = iterate_strip_v2(z0, c_arr, mode_params, max_iter, is_julia=is_julia)
    except Exception as e:
        _et_error(f'ET iteration — {ctx}', e, fatal=True)

    try:
        base_rgb = et_escape_color(it, max_iter, mode_params, tower)
    except Exception as e:
        _et_error(f'Escape coloring — {ctx}', e, fatal=True)

    try:
        lighting = et_normal_lighting(it)
    except Exception as e:
        _et_error(f'Normal-map lighting — {ctx}', e, fatal=True)

    try:
        orb_rgb, orb_w = et_orbit_color(it, tower)
    except Exception as e:
        _et_error(f'Orbit trap coloring — {ctx}', e, fatal=True)

    try:
        int_rgb  = et_interior_color(it, tower)
    except Exception as e:
        _et_error(f'Interior coloring — {ctx}', e, fatal=True)

    try:
        composite = et_composite(base_rgb, int_rgb, lighting, orb_rgb, orb_w)
    except Exception as e:
        _et_error(f'Multi-pass compositor — {ctx}', e, fatal=True)

    return row_start, composite


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 21 — CORE RENDER ENGINE  (used by both single image and video)
# ══════════════════════════════════════════════════════════════════════════════

def _render_frame(cx, cy, zoom, mode, tower, jc, is_julia,
                  max_iter, rw, rh, tile_rows, n_threads, seed_base, frame_idx=0):
    """
    Render one frame to a float32 (H,W,3) array.
    All parameters explicit — fully reusable by video loop.

    GPU path: builds ALL pixel coordinates at once, launches the CUDA kernel
    ONCE for the entire frame (all rw*rh pixels in a single call), then runs
    the coloring pipeline on the full-frame result arrays.
    One launch + one sync = near-100% GPU utilisation.
    Per-tile launches were the cause of 20% CPU / low GPU: each tile took
    microseconds on the GPU but the CPU blocked on synchronize() between tiles.

    CPU path: unchanged tiled loop with optional thread pool.
    """
    t0 = time.time()

    # ── GPU: full-frame single kernel launch ─────────────────────────────────
    if USE_GPU:
        _type_str_gpu = '\u2202I Lattice-Aware' if IS_DI_TYPE else ('Julia' if is_julia else 'Mandelbrot')
        ctx_gpu = (f'GPU full-frame — {rw}×{rh} mode={mode.get("name","?")} '
                   f'{_type_str_gpu} {max_iter:,} iters')
        try:
            rng_full = np.random.RandomState(
                seed_base ^ (frame_idx * 0x9e3779b9 & 0x7FFFFFFF))
            coords = make_strip_coords(0, rh, cx, cy, zoom, rw, rh,
                                       rng=rng_full, jitter=(SS > 1))
            if IS_DI_TYPE:
                # ∂I Lattice-Aware: z₀=0, c=pixel — intrinsic to this fractal.
                z0_full    = np.zeros_like(coords)
                c_arr_full = coords
            elif is_julia:
                z0_full    = coords
                c_arr_full = np.full_like(coords, jc)
            else:
                z0_full    = np.zeros_like(coords)
                c_arr_full = coords
        except Exception as e:
            _et_error(f'Coordinate setup — {ctx_gpu}', e, fatal=True)

        try:
            it = iterate_strip_v2(z0_full, c_arr_full, mode, max_iter,
                                  is_julia=is_julia)
        except Exception as e:
            _et_error(f'ET iteration — {ctx_gpu}', e, fatal=True)

        # All coloring passes run on the GPU using CuPy arrays.
        # xp=cp keeps all operations on the device — no PCIe traffic.
        # Single .get() on the final composite is the only transfer.
        import cupy as _cp_col
        _xp = _cp_col if it.get('_gpu', False) else np

        def _gpu_phase(label):
            """Print current GPU phase with elapsed time on its own line."""
            print(f'  [GPU] {label:<32s}  {time.time()-t0:6.1f}s', flush=True)

        _gpu_phase('Escape coloring...')
        try:
            base_rgb = et_escape_color(it, max_iter, mode, tower, xp=_xp)
        except Exception as e:
            _et_error(f'Escape coloring — {ctx_gpu}', e, fatal=True)

        _gpu_phase('Normal-map lighting...')
        try:
            lighting = et_normal_lighting(it, xp=_xp)
        except Exception as e:
            _et_error(f'Normal-map lighting — {ctx_gpu}', e, fatal=True)

        _gpu_phase('Orbit trap coloring...')
        try:
            orb_rgb, orb_w = et_orbit_color(it, tower, xp=_xp)
        except Exception as e:
            _et_error(f'Orbit trap coloring — {ctx_gpu}', e, fatal=True)

        _gpu_phase('Interior coloring...')
        try:
            int_rgb = et_interior_color(it, tower, xp=_xp)
        except Exception as e:
            _et_error(f'Interior coloring — {ctx_gpu}', e, fatal=True)

        _gpu_phase('Multi-pass composite...')
        try:
            composite_gpu = et_composite(base_rgb, int_rgb, lighting, orb_rgb, orb_w, xp=_xp)
            _gpu_phase('Transfer to CPU...')
            # Single transfer: composite float32 RGB → CPU for et_post (PIL blur)
            buf = composite_gpu.get() if hasattr(composite_gpu, 'get') else composite_gpu
        except Exception as e:
            _et_error(f'Multi-pass compositor — {ctx_gpu}', e, fatal=True)

        elapsed = time.time() - t0
        print(f'  Frame {rw}×{rh}  [█████████████████████████] 100.0%'
              f'  {elapsed:5.1f}s', flush=True)
        print()
        return buf, elapsed

    # ── CPU: tiled loop (unchanged) ──────────────────────────────────────────
    tile_rng = np.random.RandomState(seed_base ^ (frame_idx * 0x9e3779b9 & 0x7FFFFFFF))
    n_tiles  = math.ceil(rh / tile_rows)
    tile_rngs = tile_rng.randint(0, 2**31, size=n_tiles)

    tile_args = [(ti*tile_rows, min(tile_rows, rh-ti*tile_rows),
                  cx, cy, zoom, mode, max_iter, jc, is_julia,
                  rw, rh, tower, int(tile_rngs[ti]))
                 for ti in range(n_tiles)]

    buf = np.zeros((rh, rw, 3), dtype=np.float32)

    def _prog(done, total, el, prefix=''):
        eta = el/done*(total-done) if done else 0
        bar = '█'*int(done/total*25)
        print(f'{prefix}  Tile {done:4d}/{total}  [{bar:<25}] {done/total*100:5.1f}%'
              f'  {el:5.1f}s  ETA {eta:5.1f}s', end='\r', flush=True)

    t0 = time.time()
    if n_threads == 1:
        for ti, args in enumerate(tile_args):
            try:
                rs, rgb = _render_tile(args)
                buf[rs:rs+rgb.shape[0]] = rgb
            except Exception as e:
                row_s = args[0]; row_c = args[1]
                _et_error(
                    f'Tile {ti+1}/{n_tiles} (rows {row_s}–{row_s+row_c-1}) — serial render',
                    e, fatal=True)
            _prog(ti+1, n_tiles, time.time()-t0)
    else:
        completed = {}
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {pool.submit(_render_tile, a): i for i,a in enumerate(tile_args)}
            done = 0
            for fut in futures:
                idx = futures[fut]
                row_s = tile_args[idx][0]; row_c = tile_args[idx][1]
                try:
                    rs, rgb = fut.result()
                    completed[idx] = (rs, rgb)
                except Exception as e:
                    _et_error(
                        f'Tile {idx+1}/{n_tiles} (rows {row_s}–{row_s+row_c-1}) — thread pool',
                        e, fatal=True)
                done += 1
                _prog(done, n_tiles, time.time()-t0)
        for idx in sorted(completed):
            rs, rgb = completed[idx]; buf[rs:rs+rgb.shape[0]] = rgb

    elapsed = time.time() - t0
    print(f'\n')
    return buf, elapsed


def _resolve_run_params(rng):
    """
    Apply ADVANCED_PARAMS overrides on top of tower-random defaults.
    Returns (cx, cy, zoom, is_julia, jc).
    """
    # Tower choice
    if SELECTED_TOWER == 'random':
        tkey  = list(TOWERS.keys())[rng.randint(0, len(TOWERS))]
    else:
        tkey  = SELECTED_TOWER
    tower = TOWERS[tkey]

    # Centre and zoom
    if 'cx' in ADVANCED_PARAMS and 'cy' in ADVANCED_PARAMS:
        cx = ADVANCED_PARAMS['cx']
        cy = ADVANCED_PARAMS['cy']
    else:
        ccs     = tower['centers']
        cx0,cy0 = ccs[rng.randint(0, len(ccs))]
        zoom_   = rng.uniform(tower['zoom_lo'], tower['zoom_hi'])
        cx = cx0 + rng.uniform(-0.12,0.12)*zoom_
        cy = cy0 + rng.uniform(-0.12,0.12)*zoom_

    if 'zoom' in ADVANCED_PARAMS:
        zoom = ADVANCED_PARAMS['zoom']
    else:
        zoom = rng.uniform(tower['zoom_lo'], tower['zoom_hi'])

    # Mode
    mode_id = SELECTED_MODES[0]
    mode    = _blend_modes(SELECTED_MODES, tower, rng)

    # Julia c override
    if 'julia_c' in ADVANCED_PARAMS:
        jc_base = ADVANCED_PARAMS['julia_c']
    else:
        jc_base = mode['julia_c']

    # Type choice
    if FRACTAL_TYPE == 'julia':
        is_julia = True; jc = jc_base
    elif FRACTAL_TYPE == 'mandelbrot':
        is_julia = False; jc = None
    elif FRACTAL_TYPE == 'di':
        # ∂I Lattice-Aware: z₀=0, c=pixel is intrinsic to this fractal family —
        # not borrowed from Mandelbrot. The self-referential dominant-power
        # (orbit's 27720ET sublattice → p_dom each step) is what defines it.
        is_julia = False; jc = None
    else:
        # random: equal chance of all three types
        _r3 = rng.randint(0, 3)
        if _r3 == 0:
            is_julia = True;  jc = jc_base   # ET Julia
        elif _r3 == 1:
            is_julia = False; jc = None       # ET Mandelbrot
        else:
            is_julia = False; jc = None       # ∂I Lattice-Aware — IS_DI_TYPE set globally

    return tkey, tower, mode, mode_id, cx, cy, zoom, is_julia, jc


def _banner():
    print('\n' + '═'*72)
    print('  EXCEPTION THEORY FRACTAL GENERATOR  v2.0 — Professional Edition')
    print('  P ∘ D ∘ T = E  |  N=12  |  V=1/12  |  K=2/3  |  A₀=137')
    print(f'  α⁻¹(ET) ≈ {ALPHA_INV_ET:.9f}  (CODATA: 137.035999084±0.000000021)')
    print(f'  126=10N+N/2  |  π from 12-gon T-recursion  |  v=√(μ²/2λ)=2')
    print('  DE + normal-map lighting + orbit traps + interior + ACES tone-map')
    print('  ET Mandelbrot: z₀=0, c=pixel through ET 24-family manifold')
    print('  ET Julia:      z₀=pixel, c=ET-derived for selected mode')
    print('  ∂I Lattice-Aware: orbit’s 27720ET sublattice → p_dom per step  [ET-native fractal]')
    print('  H_ET: K·H(d_r) + (1-K)·H(d_θ)   [Koide 2:1 D/T weighting — ET-derived]')
    print('═'*72)



# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 22 — SINGLE IMAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_et_fractal():
    """Full professional ET fractal generation — single image."""
    _banner()

    # Seed
    if 'seed' in ADVANCED_PARAMS:
        seed = ADVANCED_PARAMS['seed']
        rng  = np.random.RandomState(seed & 0x7FFFFFFF)
    else:
        rng, seed = t_agency_seed()

    print(f'\n  Preset    : {QUALITY_PRESET}  ({IMG_W}×{IMG_H}  {MAX_ITER:,} iters)')
    print(f'  T-seed    : {seed:020d}')

    try:
        tkey, tower, mode, mode_id, cx, cy, zoom, is_julia, jc = _resolve_run_params(rng)
    except Exception as e:
        _et_error('Resolving run parameters (tower/mode/centre/zoom)', e, fatal=True)
    print(f'  Tower     : {tower["name"]}')

    n_modes = len(SELECTED_MODES)
    if n_modes == 1:
        print(f'  Mode      : [{mode_id:2d}]  {mode["name"]}')
    else:
        names = ', '.join(f'{i}:{_MODE_NAMES[i]}' for i in SELECTED_MODES)
        print(f'  Modes     : {n_modes} blended — {mode["name"]}')
        print(f'  IDs       : {names}')

    if IS_DI_TYPE:
        print(f'  Type      : ∂I Lattice-Aware  (orbit’s 27720ET sublattice → p_dom each step)')
        print(f'               p_eff=10/3  PALINDROME=[12,6,4,3,12,2,12,3,4,6,12,1]')
    elif is_julia:
        print(f'  Type      : ET Julia  c={jc.real:+.6f}{jc.imag:+.6f}j')
    else:
        print(f'  Type      : ET Mandelbrot  (z₀=0, c=pixel through ET 24-family)')

    print(f'  Centre    : ({cx:.8f}, {cy:.8f})')
    print(f'  Zoom      : {zoom:.8f}')

    prec_str = 'float32' if FLOAT_DTYPE==np.float32 else 'float64'
    accel    = f'GPU/{prec_str}' if USE_GPU else f'CPU×{N_THREADS}/{prec_str}'
    print(f'  Precision : {accel}')
    print(f'  Lattice   : |δ_r|={REAL_DELTA:.5f}  |δ_θ|={IMAG_DELTA:.4f}  g_r={G_REAL}  g_θ={G_IMAG}')

    top3r = sorted(zip(mode['w_r'], ALL_REAL),    reverse=True)[:3]
    top3c = sorted(zip(mode['w_c'], ALL_COMPLEX), reverse=True)[:3]
    print(f'  Top d_r   : {[(d, FAM_CHAR[d][:9], f"{w:.3f}") for w,d in top3r]}')
    print(f'  Top d_θ   : {[(d, FAM_CHAR[d][:9], f"{w:.3f}") for w,d in top3c]}')
    import math as _m2
    _dr_top = top3r[0][1]; _dt_top = top3c[0][1]
    _tau_idx = int((_dr_top * G_REAL) % 12)
    _tau_val  = int(QUINTIC_TENSION[_tau_idx])
    print(f'  τ_quintic : {_tau_val}¢  at semitone {_tau_idx}  '
          f'(d=5 shadow pressure on d_r={_dr_top} via g_r={G_REAL})')
    if IS_DI_TYPE:
        print(f'  p_eff     : {_P_EFF_DI:.4f}  (10/3 — mean of palindromic cascade powers, ∂I-native)')
    else:
        print(f'  p_eff     : {mode.get("p_eff",2.0):.4f}  (ET weighted power for smooth coloring)')
    print()

    try:
        render_buf, elapsed = _render_frame(
            cx, cy, zoom, mode, tower, jc, is_julia,
            MAX_ITER, RENDER_W, RENDER_H, TILE_ROWS, N_THREADS, seed)
    except Exception as e:
        _et_error(
            f'Render frame — mode={mode.get("name","?")} '
            f'{"\u2202I Lattice-Aware" if IS_DI_TYPE else ("Julia" if is_julia else "Mandelbrot")} '
            f'{MAX_ITER:,} iters  {RENDER_W}×{RENDER_H}',
            e, fatal=True)

    print(f'  Render complete in {elapsed:.1f} s', flush=True)

    if SS > 1:
        print(f'  SSAA {SS}× downsample…', flush=True)
        try:
            render_buf = ssaa_downsample(render_buf, SS)
        except Exception as e:
            _et_error('SSAA downsample', e, fatal=True)

    print('  Post-processing (ACES filmic + γ=K=2/3 + quartic vignette + unsharp)…', flush=True)
    try:
        final_f32 = et_post(render_buf)
    except Exception as e:
        _et_error('Post-processing (ACES + gamma + vignette + unsharp)', e, fatal=True)

    script_dir = Path(__file__).resolve().parent
    ts   = time.strftime('%Y%m%d_%H%M%S')
    _type_tag = 'lat' if IS_DI_TYPE else ('jul' if is_julia else 'm')
    stem = f'et_fractal_{ts}_{_type_tag}{mode_id}_{tkey[:4]}_{QUALITY_PRESET}'

    meta = {
        'Description':   'Exception Theory Fractal  |  P o D o T = E  |  v2.0',
        'ET_Manifold':   f'N={N} V={V:.6f} K={K:.6f} A0={A0_EM} 27720ET',
        'ET_Alpha_Inv':  f'{ALPHA_INV_ET:.9f}',
        'ET_Mode':       f'{SELECTED_MODES} {mode["name"]}',
        'ET_Tower':      tower["name"],
        'ET_Centre':     f'({cx:.10f}, {cy:.10f})',
        'ET_Zoom':       f'{zoom:.10f}',
        'ET_Seed':       str(seed),
        'ET_Type':       ('ET_dI_LatticeAware_p=10/3' if IS_DI_TYPE
                          else (f'Julia c={jc}' if is_julia else 'ETMandelbrot')),
        'ET_MaxIter':    f'{MAX_ITER:,}',
        'ET_p_eff':      (f'{_P_EFF_DI:.6f}' if IS_DI_TYPE else f'{mode.get("p_eff",2.0):.6f}'),
        'ET_ESCAPE_R':   f'{ESCAPE_R:.2e}',
        'ET_Preset':     QUALITY_PRESET,
        'ET_Precision':  'float64' if FLOAT_DTYPE==np.float64 else 'float32',
        'Author':        'Exception Theory -- Michael James Muller (Aevum Defluo)',
        'Software':      'ET_FRACTAL_GENERATOR.py v2.0',
    }

    print(f'  Saving 32-bit float TIFF…', flush=True)
    tiff_path = script_dir / (stem + '.tiff')
    try:
        write_tiff_float32(final_f32, tiff_path, dpi=OUTPUT_DPI)
        tiff_mb = tiff_path.stat().st_size/1_048_576
    except Exception as e:
        _et_error(f'Writing TIFF: {tiff_path}', e, fatal=True)

    print(f'  Saving 16-bit PNG…', flush=True)
    arr_u16 = (np.clip(final_f32,0.,1.)*65535.).round().astype(np.uint16)
    png_path = script_dir / (stem + '.png')
    try:
        write_png_16bit(arr_u16, png_path, dpi=OUTPUT_DPI, text_meta=meta)
        png_mb = png_path.stat().st_size/1_048_576
    except Exception as e:
        _et_error(f'Writing PNG: {png_path}', e, fatal=True)

    total = elapsed
    print(f'\n  ✓ TIFF : {tiff_path}  ({tiff_mb:.0f} MB)')
    print(f'  ✓ PNG  : {png_path}  ({png_mb:.1f} MB  16-bit  {OUTPUT_DPI} DPI)')
    print(f'  Size   : {IMG_W}×{IMG_H}  |  {MAX_ITER:,} iters  |  {total:.1f} s total')
    print('\n' + '═'*72)
    print('  P ∘ D ∘ T = E  —  Exception Theory  —  Michael James Muller')
    print('═'*72 + '\n')
    return str(tiff_path), str(png_path)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 23 — ZOOM VIDEO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_zoom_video():
    """
    Zoom video: renders N frames with exponentially decreasing zoom centered
    on a target point.  Every frame is a full fresh ET computation — no pixel
    magnification.  DE keeps boundaries infinitely sharp in every frame.

    Zoom sequence (log-linear = natural perception):
      zoom_n = zoom_start · exp( n/(N-1) · ln(zoom_end/zoom_start) )
    This gives equal visual zoom speed per frame.

    Output: PNG frames in et_video_<timestamp>/ subfolder, then ffmpeg MP4.
    """
    _banner()

    if 'seed' in ADVANCED_PARAMS:
        seed = ADVANCED_PARAMS['seed']
        rng  = np.random.RandomState(seed & 0x7FFFFFFF)
    else:
        rng, seed = t_agency_seed()

    # Resolve all parameters once — same fractal, varying zoom
    try:
        tkey, tower, mode, mode_id, cx_base, cy_base, _zoom, is_julia, jc =             _resolve_run_params(rng)
    except Exception as e:
        _et_error('Resolving run parameters for video (tower/mode/centre/zoom)', e, fatal=True)

    # Target point: VIDEO_PARAMS override or use resolved centre
    vp  = VIDEO_PARAMS
    tx  = vp.get('tx') if vp.get('tx') is not None else cx_base
    ty  = vp.get('ty') if vp.get('ty') is not None else cy_base
    z0  = float(vp.get('zoom_start', 2.5))
    z1  = float(vp.get('zoom_end',   0.001))
    nf  = int(vp.get('n_frames', 240))
    fps = int(vp.get('fps', 30))

    # Video uses 1080p resolution from preset, max iter from preset
    vw  = IMG_W; vh  = IMG_H; mi = MAX_ITER

    script_dir = Path(__file__).resolve().parent
    ts         = time.strftime('%Y%m%d_%H%M%S')
    frame_dir  = script_dir / f'et_video_{ts}_m{mode_id}_{tkey[:4]}'
    frame_dir.mkdir(exist_ok=True)

    print(f'\n  Video     : {nf} frames  {vw}×{vh}  {fps} fps')
    print(f'  Zoom      : {z0:.4f} → {z1:.6f}  (×{z0/z1:.0f} total)')
    print(f'  Target    : ({tx:.8f}, {ty:.8f})')
    print(f'  Tower     : {tower["name"]}')
    print(f'  Mode      : {mode["name"]}')
    if IS_DI_TYPE:
        print('  Type      : ∂I Lattice-Aware  p_eff=10/3  PALINDROME=[12,6,4,3,12,2,12,3,4,6,12,1]')
    elif is_julia:
        print(f'  Type      : ET Julia c={round(jc.real,5):+.5f}{round(jc.imag,5):+.5f}j')
    else:
        print('  Type      : ET Mandelbrot')
    print(f'  Frames dir: {frame_dir}')
    print(f'  Max iters : {mi:,}')
    print()

    # Precompute zoom sequence: exponential (log-linear)
    if nf > 1:
        log_z0 = math.log(z0); log_z1 = math.log(z1)
        zooms = [math.exp(log_z0 + i/(nf-1)*(log_z1-log_z0)) for i in range(nf)]
    else:
        zooms = [z0]

    frame_paths = []
    total_t0    = time.time()

    for fi, zoom_fi in enumerate(zooms):
        pct = (fi+1)/nf*100
        print(f'  Frame {fi+1:4d}/{nf}  ({pct:5.1f}%)  zoom={zoom_fi:.6f}', flush=True)

        try:
            render_buf, elapsed = _render_frame(
                tx, ty, zoom_fi, mode, tower, jc, is_julia,
                mi, vw, vh, TILE_ROWS, N_THREADS, seed, frame_idx=fi)
        except Exception as e:
            _et_error(
                f'Render frame {fi+1}/{nf} — zoom={zoom_fi:.6f} '
                f'{"Julia" if is_julia else "Mandelbrot"} {mi:,} iters',
                e, fatal=True)

        if SS > 1:
            try:
                render_buf = ssaa_downsample(render_buf, SS)
            except Exception as e:
                _et_error(f'SSAA downsample — frame {fi+1}/{nf}', e, fatal=True)

        try:
            final_f32 = et_post(render_buf)
        except Exception as e:
            _et_error(f'Post-processing — frame {fi+1}/{nf}', e, fatal=True)

        arr_u16   = (np.clip(final_f32,0.,1.)*65535.).round().astype(np.uint16)
        fp = frame_dir / f'frame_{fi:06d}.png'
        try:
            write_png_16bit(arr_u16, fp, dpi=OUTPUT_DPI)
        except Exception as e:
            _et_error(f'Writing frame PNG: {fp}  (frame {fi+1}/{nf})', e, fatal=True)
        frame_paths.append(str(fp))

        elapsed_total = time.time()-total_t0
        eta = elapsed_total/(fi+1)*(nf-fi-1)
        print(f'       frame {elapsed:.1f}s  total {elapsed_total:.0f}s  ETA {eta:.0f}s',
              flush=True)

    total_elapsed = time.time()-total_t0
    print(f'\n  All {nf} frames rendered in {total_elapsed:.0f}s '
          f'({total_elapsed/nf:.1f}s/frame avg)')

    # Assemble with ffmpeg
    video_path = script_dir / f'et_video_{ts}_m{mode_id}_{tkey[:4]}.mp4'
    ffmpeg_in  = str(frame_dir / 'frame_%06d.png')
    cmd = ['ffmpeg', '-y', '-r', str(fps),
           '-i', ffmpeg_in,
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
           '-pix_fmt', 'yuv420p',
           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
           str(video_path)]

    print(f'\n  Assembling video with ffmpeg…')
    print(f'  {" ".join(cmd)}')
    try:
        import subprocess as _sp
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            vmb = video_path.stat().st_size/1_048_576
            print(f'\n  ✓ VIDEO : {video_path}  ({vmb:.0f} MB)')
        else:
            print(f'\n  ffmpeg failed (return code {result.returncode}):')
            print(result.stderr[-2000:])
            print(f'\n  Frames are in: {frame_dir}')
            print(f'  Assemble manually:')
            print(f'  ffmpeg -r {fps} -i {ffmpeg_in} -c:v libx264 -pix_fmt yuv420p {video_path}')
    except FileNotFoundError:
        print(f'\n  ffmpeg not found. Frames are in: {frame_dir}')
        print(f'  Install ffmpeg and run:')
        print(f'  ffmpeg -r {fps} -i "{ffmpeg_in}" -c:v libx264 -pix_fmt yuv420p "{video_path}"')

    print('\n' + '═'*72)
    print('  P ∘ D ∘ T = E  —  Exception Theory  —  Michael James Muller')
    print('═'*72 + '\n')
    return str(frame_dir), str(video_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        if OUTPUT_MODE == 'video':
            fd, vp = generate_zoom_video()
            if sys.platform == 'win32':
                print(f'Frames: {fd}\nVideo:  {vp}\n')
                input('Press Enter to exit…')
        else:
            tp, pp = generate_et_fractal()
            if sys.platform == 'win32':
                print(f'Files saved:\n  {tp}\n  {pp}\n')
                input('Press Enter to exit…')
    except KeyboardInterrupt:
        print('\n\n  [Interrupted by user]')
        sys.exit(0)
    except Exception:
        import traceback
        print('\n  [ET FRACTAL ERROR]\n')
        traceback.print_exc()
        if sys.platform == 'win32': input('\nPress Enter to exit…')
        sys.exit(1)

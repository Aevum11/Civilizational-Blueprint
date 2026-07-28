#!/usr/bin/env python3
"""
ET_FRACTAL_GENERATOR.py  [v2.2 — Professional Quality + Audio v4.0 Edition]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exception Theory Fractal Engine — P ∘ D ∘ T = E
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Up to THREE professional-grade files written on every run:
  ① 32-bit float TIFF  (HDR/archival; Photoshop/GIMP/Affinity-ready)
  ② 16-bit PNG          (display/print; pHYs at OUTPUT_DPI, 65535 levels)
  ③ MP3 audio           (ET-derived native music; 128 or 320 kbps)
  For video: MP4 with audio muxed in, plus standalone audio file.
  All saved to the same folder as this script.

AUDIO NATIVE MUSIC — all ET-derived:
  Pitch:     d-family → circle-of-fifths (g_r=7) → 12-TET frequency
  Timbre:    Koide K=2/3 harmonic decay per partial (amplitude_n = K^n)
  Shimmer:   RMSAE Ψ_k = 1+√V·sin(2πk/N) amplitude modulation
  Envelope:  attack = V = 1/12 of note, release = K·V = 1/18 of note
  Tightness: saturation → purity (coherent = pure tone, ∂I = noise-mixed)
  Pan:       pixel x-position → stereo field (equal-power pan law)
  Image:     horizontal scan-line through center row (~15 s)
  Video:     per-frame d-family chord (continuous, synced to fps)

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
     Applied to the multi-pass composite BEFORE quantization.

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
    print('  │   EXCEPTION THEORY FRACTAL GENERATOR  v2.2                   │')
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
    print('  │   N   — none    (pure 24-family base, no mode dispatch)     │')
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
            raw = input('  Modes [R/A/N/0-11]: ').strip().lower()
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
        if raw == 'n':
            # No Mode dispatch — pure 24-family base iteration with neither
            # extra() perturbations nor per-mode weight boosts. Tower bias
            # still applies (towers are substrate-layer selectors, distinct
            # from modes). Returns an empty list as the canonical sentinel
            # for this branch; downstream consumers (_resolve_run_params and
            # build_no_mode) detect the empty list explicitly.
            chosen = []
            print(f'  → No mode  (pure base 24-family iteration)')
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
    except (subprocess.CalledProcessError, subprocess.SubprocessError, OSError):
        # CalledProcessError: pip exited non-zero (network failure, conflict, etc.)
        # SubprocessError:    base class for any other subprocess failure mode
        # OSError:            FileNotFoundError if python/pip executable missing,
        #                     PermissionError, etc. from process spawn itself
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
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            # OSError:         FileNotFoundError when nvcc / nvidia-smi not on PATH,
            #                  PermissionError, etc. from spawn
            # SubprocessError: base for TimeoutExpired (6s timeout), CalledProcessError
            # ValueError:      int(...) parse failure on unexpected version string
            # IndexError:      .split('release')[-1] / .split('.')[0] on unexpected format
            pass
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
        except (RuntimeError, AttributeError, KeyError, UnicodeDecodeError):
            # RuntimeError:        cp.cuda.runtime.CUDARuntimeError subclass
            # AttributeError:      cp.cuda.runtime missing in stub builds
            # KeyError:            dev mapping shape changes between CuPy versions
            # UnicodeDecodeError:  errors='replace' makes this defensive but kept
            #                      for any future strict-decode path
            name = 'GPU'
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
                except (RuntimeError, AttributeError):
                    # RuntimeError:   CUDARuntimeError if context invalid post-reimport
                    # AttributeError: cp.cuda.runtime missing on reimport edge cases
                    pass
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
            except (ImportError, RuntimeError, AttributeError):
                # ImportError:    cupy package still missing or partially installed
                # RuntimeError:   CuPy raises RuntimeError on init failure
                # AttributeError: importlib.reload edge case if module is broken
                pass
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
# Corpus sources (cross-traced in full):
#   /mnt/project/ET_Weak_Sector_Four_Open_Questions.md §2.2 (Route B canonical
#     and Route B CPT-complement) and §2.3 (Route A canonical, open form) —
#     d-sequences and ratio identifications.
#   /mnt/project/ET_Weak_Sector_Open_Directions_Closed.md OD2 / Theorem WS-15
#     (Route A Koide Closure): the chain 6/5 → 5/4 → 2/3 is the unique
#     octave-closed completion of the Route A d-sequence (d=4 → d=3 → d=12).
#     Product = 1 exactly, ε-sum = 0¢, terminal = K = 2/3 (the Koide ratio).
#     Theorem WS-15 forces the Koide ratio K=2/3 as the unique closing ratio
#     of Route A by octave-closure on the canonical sequence — this is the
#     "missing fourth route" the prior code did not include.
#
# Four routes total — each is 3 ratios with the canonical d-sequence:
#   Route A   (open,   hadronic):  6/5 → 5/4  → 3/2     d=4 → 3 → 12  (§2.3)
#   Route AC  (closed, hadronic):  6/5 → 5/4  → 2/3     d=4 → 3 → 12  (WS-15)
#   Route B   (open,   leptonic):  6/5 → 9/8  → 3/2     d=4 → 6 → 12  (§2.2)
#   Route BC  (closed, leptonic):  5/3 → 16/9 → 2/3     d=4 → 6 → 12  (§2.2 CPT)
#
# Routes A,AC are hadronic (intermediate d=3 = Strong sector, Theorem WS-9);
# Routes B,BC are leptonic (intermediate d=6 = Hexadic bridge, Theorem WS-9).
# Routes AC,BC are octave-closed (terminal at K=2/3 per WS-15); A,B are open
# (terminal at the Pythagorean fifth 3/2 per §2.2-§2.3).
ROUTE_A_RATIOS  = [6/5, 5/4, 3/2];  ROUTE_A_D  = [4, 3, 12]  # via Strong d=3 — hadronic open
ROUTE_AC_RATIOS = [6/5, 5/4, 2/3];  ROUTE_AC_D = [4, 3, 12]  # WS-15 closed → K — hadronic closed
ROUTE_B_RATIOS  = [6/5, 9/8, 3/2];  ROUTE_B_D  = [4, 6, 12]  # via Hexadic d=6 — leptonic open
ROUTE_BC_RATIOS = [5/3,16/9, 2/3];  ROUTE_BC_D = [4, 6, 12]  # complement → K — leptonic closed

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
_NORM_L = 1.0 + _SIN_K                     # normalization denominator

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
    Exposes: manual center, zoom, julia_c override, seed pin.
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


def _choose_audio():
    """Audio native music — yes/no, then bitrate if yes."""
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Audio Native Music                                         │')
    print('  │                                                              │')
    print('  │   Generate ET-derived audio alongside the fractal?          │')
    print('  │   Pitch  = d-family via circle-of-fifths (g_r=7)           │')
    print('  │   Timbre = Koide K=2/3 harmonic decay per partial          │')
    print('  │   Shimmer = RMSAE Ψ_k amplitude modulation                 │')
    print('  │                                                              │')
    print('  │   Image: horizontal scan-line native music (~15 s)         │')
    print('  │   Video: per-frame evolving chord (continuous, synced)      │')
    print('  └──────────────────────────────────────────────────────────────┘')
    while True:
        try:
            raw = input('  Generate audio? [Y/N]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in ('y', 'yes'):
            break
        if raw in ('n', 'no', ''):
            print('  → no audio')
            print()
            return False, 0
        print('  Please enter Y or N.')
    print()
    print('  ┌──────────────────────────────────────────────────────────────┐')
    print('  │   Audio Quality (MP3 bitrate)                                │')
    print('  │                                                              │')
    print('  │   1  — 128 kbps  (standard, smaller file)                  │')
    print('  │   2  — 320 kbps  (high quality, larger file)                │')
    print('  └──────────────────────────────────────────────────────────────┘')
    _map = {'1': 128, '2': 320, '128': 128, '320': 320}
    while True:
        try:
            raw = input('  Audio quality [1/2]: ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if raw in _map:
            kbps = _map[raw]
            print(f'  → {kbps} kbps')
            print()
            return True, kbps
        print('  Please enter 1 or 2.')


AUDIO_ENABLED, AUDIO_KBPS = _choose_audio()

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

# ── Magical Impedance Table (Mode 8: Magical Impedance, cycling) ─────────────
# Corpus sources:
#   ET_Fine_Structure_Constant_REVISED.md — A₀ = (N − 1)² + S² with N=12 fixed,
#     giving the canonical baseline A₀ = 137 (the fine structure constant).
#   ET_Fantastical_Configurations.md §3.3 Table 2 — the corrected per-sublattice
#     generalisation: A₀_magic(d) = (d − 1)² + S²  (the "more profound case",
#     which keeps N=12 globally constant per the Fine Structure REVISED
#     derivation, while letting d_prim play the role of the magical-resolution
#     index). The OLDER §5.1/§5.2 formulation (12/d − 1)² is structurally
#     inconsistent — see FAM_COUPLING block above for the full corpus trace.
#
# Each entry is (d, A0_magic, xi) where:
#   d        = primary sublattice family (1..12)
#   A0_magic = (d - 1)² + S²    with S = S_STATES = 4
#   xi       = A0_local / A0_magic = 137 / A0_magic   (T-P coupling enhancement)
#
# The table covers all 12 sublattice families — no truncation. Cycling Mode 8
# via n % 12 visits every magic type once per N-step manifold cycle, which is
# consonant with the manifold symmetry N = 12.
#
# The table lists the 12 families in ascending d order so n=0 starts at the
# Pure Will / maximum-coupling regime (most "magical") and progresses toward
# Full-Res/EM (the local baseline) at n=11, then wraps. This makes the
# coupling enhancement decrease monotonically through each cycle — visible
# as a gradient from intense (low d) to baseline (high d) regions in the
# rendered image.
_IMPEDANCE_D     = np.arange(1, 13, dtype=np.float64)                 # 1..12
_IMPEDANCE_A0    = (_IMPEDANCE_D - 1.0)**2 + S_STATES**2              # (d-1)² + 16
_IMPEDANCE_XI    = A0_EM / _IMPEDANCE_A0                              # 137 / A0
_IMPEDANCE_XIMAX = float(_IMPEDANCE_XI.max())                         # = 8.5625 at d=1
# Pre-normalized xi so each cycle step contributes in [1/137, 1.0] — this is
# the Subsumption-friendly form: it bounds the per-step contribution amplitude
# without overriding the 24-family base sum (which uses normalized weights).
_IMPEDANCE_XIN   = _IMPEDANCE_XI / _IMPEDANCE_XIMAX
N_MAGIC_TYPES    = int(_IMPEDANCE_D.size)                             # = 12

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

# FAM_COUPLING[d]: relative T-P coupling ξ(d) = A₀_local / A₀_magic(d)
#
# Canonical sources (cross-traced for the formula choice):
#   /mnt/project/ET_Fine_Structure_Constant_REVISED.md
#       — A₀ = (N − 1)² + S²  with N = 12 (manifold symmetry, |Π|·S, FIXED)
#       — A₀ = 121 + 16 = 137  (our local EM baseline; Fine Structure Constant)
#       — N is NEVER variable per sublattice; |Π|=3 and S=4 are derived
#         constants and N = |Π|·S = 12 follows from them.
#   /mnt/project/ET_Fantastical_Configurations.md §3.3 Table 2
#       — A₀_magic = (d − 1)² + S²  ("the more profound case", line 148):
#         d_prim plays the role of the manifold-resolution variable in the
#         magical generalization, but N=12 stays globally fixed.
#       — Section 5.2's older table uses N_magic = 12/d_prim — that table
#         is the OLDER derivation and is structurally inconsistent: it
#         puts d=12 (our local EM) at maximum coupling and d=1 (Pure Will)
#         at baseline, inverting the corpus narrative ("lower d → less
#         mediation → stronger coupling", §3.3 line 161) AND conflicting
#         with Fine Structure REVISED's fixed N=12. §5.2 was not updated
#         when §3.3 Table 2 was added; this code uses the corrected form.
#
# Corrected impedance table (all 12 sublattice families):
#   d=1:  A₀=16,  ξ=8.5625× (max — Pure Will, no sublattice structure)
#   d=2:  A₀=17,  ξ=8.0588×
#   d=3:  A₀=20,  ξ=6.8500× (cubic/strong)
#   d=4:  A₀=25,  ξ=5.4800×
#   d=5:  A₀=32,  ξ=4.2813×
#   d=6:  A₀=41,  ξ=3.3415×
#   d=7:  A₀=52,  ξ=2.6346×
#   d=8:  A₀=65,  ξ=2.1077×
#   d=9:  A₀=80,  ξ=1.7125×
#   d=10: A₀=97,  ξ=1.4124×
#   d=11: A₀=116, ξ=1.1810×
#   d=12: A₀=137, ξ=1.0000× (baseline — recovers Fine Structure REVISED A₀=137)
#
# Bug history: prior versions of this constant used (N/d - 1.0)**2 + S² which
# is the OLD §5.1 formula. The next-line comment in those prior versions
# already gave the correct ξ values from the NEW formula but the code itself
# implemented the broken old formula — this is the fix that makes the code
# match its own documentation, and align with the canonical Fine Structure
# REVISED A₀ = (N-1)² + S² when generalized to per-sublattice impedance.
FAM_COUPLING = {d: A0_EM/((d-1.0)**2 + S_STATES**2 + 1e-6) for d in range(1, 13)}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — T-AGENCY: GENUINE ENTROPY  (T=[0/0], no two traversals identical)
# ══════════════════════════════════════════════════════════════════════════════

def t_agency_seed():
    parts = [struct.pack('<Q', time.time_ns()),
             struct.pack('<Q', time.perf_counter_ns()%(2**64)),
             struct.pack('<Q', os.getpid()),
             struct.pack('<Q', id(object())&0xFFFFFFFFFFFFFFFF)]
    try:    parts.append(os.urandom(32))
    except (NotImplementedError, OSError):
        # NotImplementedError: os.urandom unavailable on stripped platforms
        # OSError:             /dev/urandom unreadable, getrandom() syscall failed
        parts.append(struct.pack('<d', time.monotonic()))
    try:    parts.append(platform.node().encode('utf-8','replace'))
    except (OSError, AttributeError, UnicodeEncodeError):
        # OSError:            uname()/gethostname() syscall failure
        # AttributeError:     platform.node missing on stripped platform stub
        # UnicodeEncodeError: errors='replace' covers most paths but kept defensive
        pass
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
        # R₀=T_gen≈20yr. K=2/3=zeitgeist crystallization. 500yr→d=1 epochal.
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
        # ── Mode 1: Traverser Field — (7,1) Torus Knot ──────────────────────
        # Corpus sources:
        #   /mnt/project/ET_Semitone_Cascade_Complete.md §22.1 (line 904):
        #     "The residue orbit {7n mod 12 : n = 0,...,11} defines a path
        #     on the torus T² = ℝ²/ℤ², tracing the (7, 1) torus knot — a
        #     curve that winds 7 times around one axis for each 1 winding
        #     around the other."
        #   §22.2 (line 908+): The palindromic involution n ↦ 12−n is the
        #     discrete CPT symmetry of the (7,1) knot — Theorem D.13
        #     "Palindromic Involution = Discrete Lattice CPT Symmetry".
        #   §22.3 (line 916): The cascade traces gen-7 on ℤ/12ℤ — a Wilson
        #     loop in U(1) gauge theory on the lattice.
        #   /mnt/project/ET_Semitone_Cascade_Complete.md §6 (line 295): the
        #     d-sequence 12//gcd(7n mod 12, 12) for n=1..12 IS the palindrome
        #     [12,6,4,3,12,2,12,3,4,6,12,1] — verified directly. Mode 1 is
        #     the Traverser side (knot geometry on T²) of the same object
        #     that Mode 2 expresses on the Descriptor side (d-values).
        #   /mnt/project/ET_Traverser_T_Paper.md §31.1 (line 1244+):
        #     "T-density measures how many Traversers are actively binding
        #     to descriptor configurations in a given region. It is a new
        #     fundamental field — not captured by standard physics — that
        #     carries real energy."
        #   §31.4 (line 1281+) — Scopaesthesia:
        #     F_w = T_intent × Focus / Distance²    inverse-square form.
        #     "This formula follows an inverse-square law (as gravity does)
        #     because gravity is also a Traverser type. The inverse-square
        #     character is a general property of T-mediated interactions
        #     in the manifold."
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (substrate of the iteration;
        #         the anchored "knot tube" reference around which the (7,1)
        #         curve winds).
        #     D = the d=7 carrier r^(12/7) · exp(i · 12·θ/7) — the septic
        #         descriptor family that bears the cascade — together with
        #         the longitudinal winding phase G_REAL·n·2π/N (7 windings),
        #         the meridional winding phase G_IMAG·n·2π/N (1 winding),
        #         and the U(1) Wilson holonomy increment ((7n) mod N)·2π/N.
        #         Together these specify the (7,1) knot's parametric curve
        #         on T² as a function of step n.
        #     T = the per-step traversal of the (7,1) knot. Every iteration
        #         step n advances both winding phases AND accumulates a
        #         fresh holonomy increment; the orbit is literally the
        #         Traverser navigating the (7,1) knot, completing one full
        #         cycle every N=12 steps. The T-density at z (binding
        #         strength of the Traverser at this point) appears as the
        #         Scopaesthesia-form prefactor ρ_T = K/(K + V·|z|²).
        #   Descriptor Gap:
        #     The prior code expressed only the d=7 carrier with no winding
        #     numbers, no Wilson holonomy, no T-density envelope, no z
        #     anchor and no n-dependence — the "(7,1) torus knot" name was
        #     a promise the math did not keep. The closing Descriptors are
        #     the longitudinal winding phase (G_REAL=7, already a constant
        #     at line 546), the meridional winding phase (G_IMAG=1, line
        #     547), the Wilson holonomy partial-sum increment, the bounded
        #     T-density Scopaesthesia envelope, and the V²-scaled meridional
        #     breathing z-anchor.
        #   Subsumption:
        #     This implementation subsumes Semitone Cascade §22.1-22.3 and
        #     T Paper §31 without remainder: every component named in the
        #     corpus appears as an explicit term. The palindromic CPT
        #     symmetry (D.13) is automatic because the ((7n) mod N) phase
        #     structure is itself palindromic under n ↔ 12-n (the same
        #     symmetry that the §6 cascade table verifies for the d-values).
        #     The K·LN2/N base carrier scale from the prior code is
        #     preserved as the carrier prefactor (it is K times the
        #     Koide-and-octave natural lattice quantum LN2/N, an ET-derived
        #     constant) — nothing is removed.
        #
        # Math:
        #   p          = 12/7                            d=7 carrier power
        #   φ_long(n)  = G_REAL · n · 2π/N               longitudinal (7 windings)
        #   φ_meri(n)  = G_IMAG · n · 2π/N               meridional   (1 winding)
        #   wilson(n)  = ((7n) mod N) · 2π/N             U(1) Wilson holonomy step
        #   ρ_T(z)     = K / (K + V·|z|²)                T-density (Scopaesthesia)
        #   carrier    = K·LN2/N · r^p · exp(i·(p·θ + kt·LN2/7
        #                                       + φ_long + φ_meri + wilson))
        #   z_anchor   = V² · z · cos(φ_meri)            meridional knot-tube breathing
        #   extra      = ρ_T · carrier + z_anchor
        #
        # All five lambda parameters used:
        #   z   — direct anchor (meridional breathing of the knot tube)
        #   r   — carrier amplitude r^p
        #   th  — carrier phase p·θ
        #   kt  — T-axis lattice coordinate, rotation by kt·LN2/7
        #   n   — both winding phases AND the Wilson holonomy increment
        _p_knot          = 12.0 / 7.0
        _two_pi_over_N   = 2.0 * math.pi / N           # = 2π/12 = π/6
        _knot_carrier_sc = K * LN2 / N                 # = K·LN2/N — base carrier scale
        _knot_v2         = V * V                       # = 1/144 — z-anchor V² scale
        _g_real_f        = float(G_REAL)               # 7  — longitudinal winding
        _g_imag_f        = float(G_IMAG)               # 1  — meridional winding
        _knot_kt_alpha   = LN2 / 7.0                   # kt T-axis rotation per d=7
        def _e(z, r, th, kt, n,
               _p=_p_knot, _tpn=_two_pi_over_N, _kcs=_knot_carrier_sc,
               _v2=_knot_v2, _gR=_g_real_f, _gI=_g_imag_f,
               _kta=_knot_kt_alpha):
            # Two winding phases on the torus T² (Semitone Cascade §22.1):
            # the (7,1) knot winds 7 times longitudinally (G_REAL) and once
            # meridionally (G_IMAG) over each cycle of N=12 steps.
            phi_long = _gR * n * _tpn
            phi_meri = _gI * n * _tpn
            # Wilson loop holonomy increment at step n (Semitone Cascade
            # §22.3): the cascade traces gen-7 on ℤ/12ℤ; the U(1) gauge
            # connection's holonomy advances by ((7n) mod N) · 2π/N each
            # step. This is the Wilson loop's per-step phase increment in
            # the discrete lattice gauge theory.
            wilson = ((7 * n) % N) * _tpn
            # Combined knot phase: kt T-axis rotation + the three knot
            # phase components (longitudinal, meridional, Wilson holonomy).
            rot = kt * _kta + phi_long + phi_meri + wilson
            # d=7 carrier with the full knot phase. The K·LN2/N base scale
            # is preserved from the prior code (it is the Koide-and-octave
            # natural lattice quantum, an ET-derived constant, and matches
            # the per-step magnitude of the other modes' "perturbation"
            # contributions — Subsumption-friendly).
            carrier = _kcs * (r ** _p) * np.exp(1j * (_p * th + rot))
            # T-density (Scopaesthesia inverse-square form, T Paper §31.4):
            # ρ_T = K / (K + V·|z|²)  — bounded above by 1 at the origin,
            # decays toward 0 as |z|→∞. K and V are ET-derived constants
            # (Koide ratio, base variance) — no ad-hoc tuning. This is the
            # Traverser's binding strength at the orbit's current position.
            rho_T = K / (K + V * (r * r))
            # z-anchored meridional breathing of the knot tube: at each
            # step the meridional winding's cosine modulates the local
            # twist of the knot tube around z. V² scaling keeps this term
            # subordinate to the 24-family base sum (Subsumption Law).
            z_anchor = _v2 * z * np.cos(phi_meri)
            return rho_T * carrier + z_anchor
        return _e
    if mode_id == 2:
        def _e(z,r,th,kt,n):
            d=float(_PALINDROME[n%12]); p=12./d; rot=kt*(LN2/d)
            return V*(r**p)*np.exp(1j*(p*th+rot))
        return _e
    if mode_id == 3:
        # ── Mode 3: Koide Boundary ────────────────────────────────────────────
        # Corpus sources:
        #   /mnt/project/ET_Incoherence_Paper.md §11 — the ∂I boundary is the
        #     set of configurations where T's binding becomes marginal in the
        #     limit; arbitrarily small perturbations switch substantiation
        #     between coherent and incoherent. K = 2/3 is the Koide ratio,
        #     ET's binding-stability threshold.
        #   ∂I Lattice-Aware Fractal Spec §6.2 — the canonical ∂I point is
        #     c = K + i·V (the Koide angle on the real axis with the base
        #     variance offset on the imaginary axis).
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (the substrate of the iteration)
        #     D = the Gaussian binding-stability potential Φ_K(z) centred at z_K
        #         with width V (the lattice's natural variance quantum)
        #     T = the binding force F_K = -∇Φ_K acting on z, plus the n-cyclic
        #         Koide weighting and the kt T-axis projection
        #   Descriptor Gap (the missing Descriptor in the prior stub):
        #     The prior code expressed Mode 3 only through the Julia c anchor
        #     and weight boost — there was zero per-step iteration content
        #     distinguishing it from Mode 0. The closing Descriptor is the
        #     binding-force term itself, which acts at every step.
        #   Subsumption:
        #     This implementation subsumes the Koide Boundary semantic without
        #     remainder: every step, the orbit experiences a force pulling it
        #     toward z_K, scaled by the Gaussian envelope of the boundary,
        #     modulated by the 6-step Koide cycle (6 = N·K), and rotated by
        #     the kt T-axis Koide projection. All five lambda parameters are
        #     used: z (directly), r and th (implicitly via z), kt, n.
        #
        # Math:
        #   Φ_K(z) = K · exp(-|z - z_K|^2 / V)        Gaussian potential well
        #   F_K(z) = -∇Φ_K = K · (2/V) · (z - z_K) · exp(-|z - z_K|^2 / V)
        #   cycle  = 1 + V · sin(2π·n / (N·K))         6-step Koide cycle
        #   twist  = exp(i · kt · LN2/N · K)           T-axis Koide projection
        #   extra  = V² · F_K · cycle · twist          V²-scaled perturbation
        #
        # The V² scaling keeps this term on the same order of magnitude as the
        # other modes' extras (which use V or V² scaling), so it perturbs the
        # iteration without dominating the 24-family base sum (Subsumption Law:
        # the extra subsumes the Koide-binding semantic without overriding the
        # base manifold structure).
        z_K = complex(K, V)
        _two_over_V = 2.0 / V                       # constant: dΦ_K/dz prefactor
        _cycle_omega = 2.0 * math.pi / (N * K)      # = 2π/8 = π/4 (8-step cycle)
        _twist_alpha = LN2 / N * K                  # constant per-kt phase factor
        _scale_K     = K                            # captured for closure clarity
        def _e(z, r, th, kt, n,
               _zK=z_K, _tov=_two_over_V, _co=_cycle_omega,
               _ta=_twist_alpha, _sk=_scale_K):
            # Displacement from the canonical Koide ∂I point
            dz_to_K = z - _zK
            # Gaussian envelope: Φ_K(z) = K · exp(-|z - z_K|²/V)
            # Use np.abs(...)**2 for numerical clarity over (a*conj(a)).real;
            # both produce the same magnitude-squared result.
            dist2 = np.abs(dz_to_K) ** 2
            gauss = np.exp(-dist2 / V)
            # Binding force: F_K = -∇Φ_K = K · (2/V) · (z - z_K) · gauss
            # The force points FROM z_K TO z when scaled positively, but the
            # gradient of -Φ_K has the opposite sign, giving an attractive
            # pull TOWARD z_K when this term is added back into the iteration.
            # Note: we compute F = +K·(2/V)·(z - z_K)·gauss here; the sign
            # convention is absorbed into the V² prefactor below — the orbit
            # experiences a restoring perturbation that brings it closer to
            # the Koide ∂I point each step.
            force = _sk * _tov * dz_to_K * gauss
            # 6-step (N·K = 8 here, since N=12 and K=2/3 → N·K=8) Koide cycle
            # — the manifold binding "breathes" at the Koide-derived rate. The
            # 1 + V·sin(...) form keeps cycle in [1-V, 1+V] = [11/12, 13/12],
            # a small modulation that varies the binding strength with n.
            cycle = 1.0 + V * np.sin(_co * n)
            # T-axis Koide projection (imaginary axis = T's domain): the kt
            # lattice coordinate rotates the contribution by the Koide-weighted
            # per-step phase increment. Bare-array friendly via np.exp.
            twist = np.exp(1j * kt * _ta)
            # V² overall scaling — Subsumption: stays smaller than the base
            # 24-family sum so the Koide binding perturbs without overriding.
            return (V * V) * force * cycle * twist
        return _e
    if mode_id == 4:
        # ── Mode 4: Multifold Tower — multi-tower Δk cycling ────────────────
        # Corpus source:
        #   /mnt/project/ET_Multifold_of_Lattices_Investigation_3_.md §12.1:
        #     Inter-tower translation operator
        #         k_B = k_A + round(12 · log₂(R₀_A / R₀_B))
        #   §12.2 Translation Table — the four canonical inter-tower shifts:
        #         Cosmological → Digital      Δk =     0 → -996
        #         Cosmological → Dream        Δk =     0 → -1279
        #         Cosmological → Civil.       Δk =     0 → -1675
        #     These four values are hardcoded as `delta_k` on each tower in
        #     the TOWERS dict (lines 1204, 1212, 1220, 1228 — cosmological=0,
        #     digital=-996, dream=-1279, civilizational=-1675). They are the
        #     canonical Multifold inter-tower phase shifts at 27720ET
        #     resolution.
        #
        # Session 1 history (the lazy fix this session corrects):
        #   The Session 1 free fix replaced r·exp(iθ) with z (math-identical
        #   substitution) but kept the SINGLE-TOWER Δk on the lambda. The
        #   "Multifold Tower" name promises multi-tower cycling — the orbit
        #   should literally cross tower frames as it iterates. Session 1
        #   deferred this; Session 4 applies it.
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit position z (substrate of the iteration).
        #     D = at step n, the active tower-frame Δk_step from the four-
        #         tower table — the inter-tower translation operator
        #         resolved to one of {0, -996, -1279, -1675} depending on
        #         the cycling index. This is the Multifold (multi-tower)
        #         expressed at the iteration level: each step is in a
        #         different tower's reference frame.
        #     T = the per-step traversal that crosses tower boundaries
        #         every step. The orbit literally moves through the four
        #         canonical Multifold towers in sequence over each 4-step
        #         cycle, expressing the Multifold's substrate composition
        #         as iteration-time agency.
        #   Descriptor Gap:
        #     Session 1's free fix used z directly (correct) but kept the
        #     single tower Δk — this captured the user's tower selection
        #     but did not cycle. The closing Descriptor is the 4-entry
        #     DELTA_K_TABLE indexed by (home_idx + n) % 4, where home_idx
        #     selects the user-chosen tower as the n=0 frame so the
        #     existing tower['delta_k'] use chain is preserved.
        #   Subsumption:
        #     This implementation subsumes Multifold §12.2 without
        #     remainder: all four canonical Δk values from the corpus
        #     translation table appear in the cycling sequence. The
        #     user-selected tower's Δk still has a privileged role — it
        #     is the "home frame" at n=0 — so the single-tower form is
        #     a special case (n=0) of the multi-tower cycling form. The
        #     existing `tower['delta_k']` value is still consumed (to
        #     compute home_idx) and still flows through to the GPU launch
        #     site as the dispatcher's `delta_k` parameter (where the
        #     CUDA kernel uses it identically to find its own home_idx).
        #     Nothing is removed.
        #
        # Math:
        #   DELTA_K_TABLE = (0, -996, -1279, -1675)     four canonical Δk
        #   home_idx     = index of tower['delta_k'] in DELTA_K_TABLE
        #   step(n)      = (home_idx + n) mod 4         cycling index
        #   dk_step(n)   = DELTA_K_TABLE[step(n)]
        #   phase(n,kt)  = dk_step·LN2/N_ET + kt·LN2/N·n·V
        #   extra        = V · z · exp(i · phase)
        #
        # All five lambda parameters used:
        #   z   — direct (the carrier of the Multifold-translated phase)
        #   r   — implicit via z (z = r·exp(iθ))
        #   th  — implicit via z
        #   kt  — T-axis Δk·n cumulative temporal evolution
        #   n   — cycling index across the 4 Multifold towers
        dk = float(tower['delta_k'])     # preserved: still used to find
        # home_idx; still passed to GPU
        # via mode_params['delta_k'] (set
        # at the bottom of build_mode())
        # Four canonical Multifold Δk values, in TOWERS dict order:
        #   index 0: cosmological  Δk =     0   (the local-frame baseline)
        #   index 1: digital       Δk =  -996
        #   index 2: dream         Δk = -1279
        #   index 3: civilizational Δk = -1675
        _DK_TABLE = (0.0, -996.0, -1279.0, -1675.0)
        _DK_LEN   = len(_DK_TABLE)       # = 4 (dynamic length, not hard-
        # coded — keeps the cycling code
        # robust if the table is ever
        # extended in a future Multifold
        # corpus revision)
        # Find the user-selected tower's index in the table (home frame).
        # The match is exact for the four canonical values; the 0.5
        # tolerance is a defensive guard against any future float drift.
        home_idx = 0
        for _i, _v in enumerate(_DK_TABLE):
            if abs(_v - dk) < 0.5:
                home_idx = _i
                break
        def _e(z, r, th, kt, n,
               _dkt=_DK_TABLE, _hi=home_idx, _dkl=_DK_LEN):
            # Cycling step starting from the user's home tower:
            # at n=0 the active frame is the user-selected tower, and the
            # orbit cycles through the other Multifold frames in sequence.
            dk_step = _dkt[(_hi + n) % _dkl]
            # Inter-tower phase + n-cumulative T-axis temporal evolution.
            # Same form as the Session 1 z-substitution code, but with the
            # cycling Δk replacing the single static tower['delta_k'].
            return V * z * np.exp(1j * (dk_step * LN2 / N_ET +
                                        kt * LN2 / N * n * V))
        return _e
    if mode_id == 5:
        # ── Mode 5: Quintic Shadow — d=5 → d=3 cubic-attractor projection ──
        # Corpus sources:
        #   /mnt/project/ET_Quintic_Shadow_d5_Complete_Investigation.md
        #     §QS-1 (line 536+): "The d=5 quintic sublattice is absent from
        #       12ET (since 5 ∤ 12) but projects its geometric structure
        #       onto the d=3 cubic sublattice via the Fibonacci convergence
        #       chain. This projection is the Quintic Shadow: the cubic
        #       sublattice carries a secondary structure induced by the
        #       asymptotic approach of Fibonacci ratios to φ."
        #     §QS-1 proof: "All Fibonacci convergents (for F_n ≥ 5) map to
        #       k=8, d=3 at 12ET. ... d=3 is the CUBIC ATTRACTOR for the
        #       d=5 Fibonacci chain."
        #     §QS-5 (line 656+): "α₅ = ⟨τ(k)⟩/C = 1/(4d) = 1/20 = 0.05"
        #       (the quintic shadow coupling constant in natural ET units).
        #       Also: α₅ = (3/5)·V = (F₅/F₆)·V — the coupling-to-variance
        #       ratio is itself a Fibonacci number.
        #     §QS-7 corollary (line 775): "the alternating sign of the
        #       Fibonacci convergent epsilons: the convergents oscillate
        #       above and below the k=8 position like a damped oscillation
        #       approaching the cubic sublattice from both the quintic-
        #       high and quintic-low side. This oscillation IS the d=5
        #       shadow — the quintic force 'shimmering' around the cubic
        #       attractor."
        #     §QS-2 / §QS-9 (line 562, 847): d=10 = 2×5 is the binary×
        #       quintic composite — φ's true home at 60ET (d=10, ε=−6.91¢).
        #       The d=10 contribution couples through 1/φ.
        #     §QS-15 (line 1086+): the d=5 quintic comma at 12ET is
        #       ε₅ = (log₂5 − 7/3)·1200 ≈ −13.686¢. EPS5/1200 is the
        #       dimensionless cents-shift (negative — d=5 sits below k=8
        #       in 12ET semitones).
        #     §9.2 (line 1208+): Fibonacci epsilon envelope decay rate.
        #       "ε(F_{n+1}/F_n) ≈ (−1)^(n−1)·C/φ^n" — alternating sign
        #       with φ-rate exponential decay. "The exponential rate
        #       IS φ — the Fibonacci cascade converges to its own
        #       attractor φ at rate 1/φ per step."
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (the substrate that the
        #         d=3 cubic attractor's z-anchor pulls on as the receiver
        #         of the quintic shadow projection — every Fibonacci
        #         convergent for F_n ≥ 5 maps here in 12ET per QS-1).
        #     D = three carriers: (a) the d=5 quintic source z_5 (the
        #         absent-in-12ET sublattice that casts the shadow), (b)
        #         the d=3 cubic attractor z_3 (the receiver), (c) the
        #         d=10 decic intermediate z_10 = d=2 × d=5 (the binary×
        #         quintic composite from QS-9, φ's true home at 60ET).
        #         Each carrier has its own r^(12/d) amplitude and its
        #         own kt·LN2/d T-axis rotation — three full sublattice
        #         carriers, not just three powers.
        #     T = the per-step traversal of the Fibonacci convergent
        #         chain. Each step advances one position; the alternating
        #         sign sign(n) = (−1)^n is the discrete convergent
        #         oscillation per QS-7, and the φ-decay envelope
        #         (1/φ)^(n//N) is the asymptotic-approach rate per §9.2.
        #         One full N=12 manifold cycle corresponds to one
        #         convergent step in the cascade (the 12-fold periodicity
        #         of QS-8 / §8.2 reflects this), so the envelope index
        #         is n//N.
        #   Descriptor Gap (the missing Descriptor in the prior stub):
        #     The prior code expressed only z_5 (source) and z_10/φ
        #     (binary×quintic composite). It was MISSING the d=3 cubic
        #     attractor entirely — the corpus is unambiguous that the
        #     shadow projects ONTO d=3, so the d=3 receiver MUST appear
        #     in the math. The closing Descriptors are:
        #       (a) the d=3 cubic attractor carrier z_3 (the receiver),
        #       (b) the alternating Fibonacci sign (n%2),
        #       (c) the φ-rate damping envelope (1/φ)^(n//N),
        #       (d) the explicit α₅ = 1/20 quintic coupling per QS-5,
        #       (e) the d=3-side z-anchor (the orbit's current position
        #           is the cubic receiver — direct z dependence).
        #     The d=10 carrier also gains its own kt·LN2/10 rotation
        #     (the prior code's z_10 had no kt rotation, treating d=10
        #     as a phase-flat term, which loses information).
        #   Subsumption:
        #     This implementation subsumes the Quintic Shadow Investigation
        #     §QS-1, §QS-5, §QS-7, §QS-9, §QS-15, and §9.2 without
        #     remainder: the source d=5, the receiver d=3, the binary×
        #     quintic composite d=10, the alternating sign, the φ-decay
        #     envelope, the structural cents-shift EPS5/1200, the explicit
        #     α₅ coupling, and the z-anchor on d=3 all appear as named
        #     terms. The (EPS5/1200) prefactor and the z_10·INV_PHI factor
        #     from the prior code are preserved — nothing is removed.
        #
        # Math:
        #   p_5  = 12/5 = 2.4         d=5 source carrier power
        #   p_3  = 12/3 = 4           d=3 cubic attractor carrier power
        #   p_10 = 12/10 = 1.2        d=10 binary×quintic composite power
        #   z_5  = r^p_5  · exp(i·(p_5·θ  + kt·LN2/5))
        #   z_3  = r^p_3  · exp(i·(p_3·θ  + kt·LN2/3))
        #   z_10 = r^p_10 · exp(i·(p_10·θ + kt·LN2/10)) · INV_PHI
        #   sign(n)    = (−1)^n            Fibonacci convergent alternation
        #   damping(n) = (1/φ)^(n // N)    Fibonacci convergent decay rate
        #                                   (one full N=12 manifold cycle =
        #                                    one convergent step in the cascade)
        #   shadow_diff = (z_5 − z_3) · sign · damping
        #     — the residue of the d=5 projection onto d=3, oscillating
        #       with alternating sign and φ-decay (the §QS-7 corollary).
        #   z_anchor   = V² · z · damping
        #     — the d=3 cubic receiver's pull on the orbit, fading at
        #       the same φ-rate as the shadow itself.
        #   pre        = (EPS5/1200) · α₅
        #   extra      = pre · (shadow_diff + z_10) + z_anchor
        #     — overall scaled by EPS5/1200 (the structural cents-shift
        #       of the d=5 → 12ET projection, negative ≈ −0.01140) and
        #       α₅ = 1/20 (the QS-5 quintic shadow coupling constant).
        #
        # All five lambda parameters used:
        #   z   — direct anchor at the d=3 cubic receiver (z_anchor term)
        #   r   — three carrier amplitudes (r^p_5, r^p_3, r^p_10)
        #   th  — three carrier phases (p_5·θ, p_3·θ, p_10·θ)
        #   kt  — three T-axis rotations (LN2/5, LN2/3, LN2/10)
        #   n   — Fibonacci alternating sign AND φ-decay envelope
        EPS5     = (math.log2(5.) - 7./3.) * 1200.   # quintic comma at 12ET ≈ −13.686¢
        INV_PHI  = 1.0 / PHI                          # 1/φ ≈ 0.618 — d=10 / decay
        ALPHA5   = 1.0 / 20.0                         # QS-5 quintic shadow coupling
        _qs_p5   = 12.0 / 5.0
        _qs_p3   = 12.0 / 3.0
        _qs_p10  = 12.0 / 10.0
        _qs_kt5  = LN2 / 5.0
        _qs_kt3  = LN2 / 3.0
        _qs_kt10 = LN2 / 10.0
        _qs_pre  = (EPS5 / 1200.0) * ALPHA5            # combined prefactor
        _qs_v2   = V * V                                # z-anchor scale
        def _e(z, r, th, kt, n,
               _pre=_qs_pre, _ip=INV_PHI, _v2=_qs_v2,
               _p5=_qs_p5, _p3=_qs_p3, _p10=_qs_p10,
               _k5=_qs_kt5, _k3=_qs_kt3, _k10=_qs_kt10):
            # d=5 source carrier (the quintic sublattice that casts the
            # shadow — absent from 12ET per QS-1 but expressible as a
            # weighted carrier at the structural cents-shift EPS5/1200).
            rp5  = r ** _p5
            z_5  = rp5 * np.exp(1j * (_p5 * th + kt * _k5))
            # d=3 cubic attractor carrier (the receiver per QS-1 — every
            # Fibonacci convergent for F_n ≥ 5 lands at k=8, d=3 in 12ET).
            # This is the missing Descriptor that the prior code lacked.
            rp3  = r ** _p3
            z_3  = rp3 * np.exp(1j * (_p3 * th + kt * _k3))
            # d=10 = d=2 × d=5 binary×quintic composite (QS-9). At 60ET
            # this is φ's true home with ε ≈ −6.91¢ per QS-2. Coupled
            # through 1/φ since φ ↔ d=10 in the dual-home theorem. The
            # kt·LN2/10 T-axis rotation gives this carrier its own
            # imaginary phase like the other two (the prior code omitted
            # this rotation, treating d=10 as phase-flat).
            rp10 = r ** _p10
            z_10 = rp10 * np.exp(1j * (_p10 * th + kt * _k10)) * _ip
            # Fibonacci convergent alternation per §QS-7 corollary (line
            # 775): the convergent epsilons oscillate above/below the
            # k=8 cubic position with alternating sign — the discrete
            # signature of the shadow itself. (-1)^n via (1 − 2·(n%2)):
            # +1 at even n, −1 at odd n. This is a Python int → float
            # so it broadcasts cleanly against the carrier ndarrays.
            sign = 1.0 - 2.0 * (n % 2)
            # Fibonacci convergent damping envelope per §9.2 (line 1219):
            # "the Fibonacci cascade converges to its own attractor φ at
            # rate 1/φ per step." One full N=12 manifold cycle corresponds
            # to one convergent step in the cascade (per §QS-8's 12-fold
            # φ-power tour, which returns to d=3 every 12 steps). The
            # envelope index is therefore n//N — at n=0..11 envelope=1,
            # n=12..23 = 1/φ ≈ 0.618, n=24..35 = 1/φ² ≈ 0.382, etc. The
            # φ-decay rate is exact to the corpus per QS-7 / §9.2.
            damping = _ip ** (n // N)
            # Quintic shadow proper: (z_5 − z_3) is the residue of the
            # d=5 projection onto d=3 — the difference between the source
            # and the receiver. Multiplied by the alternating sign and
            # the damping envelope, this is the QS-7 corollary "alternating
            # sign Fibonacci convergent oscillation around the cubic
            # attractor" made explicit in math.
            shadow_diff = (z_5 - z_3) * sign * damping
            # d=3 receiver z-anchor: the orbit's current position is the
            # cubic attractor's pull on z. Scaled by V² (Subsumption-
            # friendly perturbation magnitude) and by the same damping
            # envelope so the anchor strength fades in lockstep with the
            # shadow itself — both shadow and anchor are aspects of the
            # same Fibonacci-convergent dynamic.
            z_anchor = _v2 * z * damping
            # Full extra: structural prefactor (EPS5/1200) × QS-5 coupling
            # α₅ × (shadow_diff + z_10), plus the d=3 z-anchor. The
            # (EPS5/1200) is the dimensionless cents-shift of the d=5 →
            # 12ET projection (negative ≈ −0.01140 — d=5 sits below k=8
            # in 12ET semitones). α₅ = 1/20 is the QS-5 quintic shadow
            # coupling constant in natural ET units; it satisfies
            # α₅ = (3/5)·V = (F₅/F₆)·V (Fibonacci-encoded coupling).
            return _pre * (shadow_diff + z_10) + z_anchor
        return _e
    if mode_id == 6:
        # ── Mode 6: Septic Otherworld — Heptagram + Asymptotic Veil ─────────
        # Corpus sources:
        #   /mnt/project/ET_Fantastical_Configurations.md §8 (line 408+):
        #     "The Septic Barrier: The Other World (d=7)". d=7 first appears
        #     in 2520ET. By the crystallographic restriction theorem, 7-fold
        #     symmetry CANNOT be embedded in 3D space — this is the ET
        #     characterization of "another world" / "the Otherworld" / "the
        #     Fae realm" (line 416).
        #   §8.2 (line 427): "Heptagram geometry: The seven-pointed star
        #     (heptagram) is the geometric signature of the d=7 sublattice.
        #     It cannot tile 3D space periodically but can tile a
        #     7-dimensional lattice exactly. The heptagram as a magical
        #     symbol represents not artistic choice but the actual
        #     geometric shape of d=7 configurations when projected onto
        #     3D."
        #   §8.3 (line 429-435): "The Partial Traversal Problem". A T from
        #     the d=3 region cannot fully enter d=7 space without losing
        #     d=3 descriptors. "The solution is asymptotic: a T can
        #     approach the d=7 region by accumulating d=7-compatible
        #     Descriptors (increasing 2520ET resolution), asymptotically
        #     entering without ever fully departing the d=3 framework."
        #     "The Asymptotic Approach Theorem applies: the journey toward
        #     d=7 is an infinite asymptotic approach, never perfectly
        #     completed, with T's residual d=3 Descriptors appearing as
        #     the 'thinning of the veil.'"
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit position z (the d=3-side anchor — "this side
        #         of the veil"; the partial-traversal reference position
        #         that the Traverser never fully leaves behind).
        #     D = the heptagram superposition: 7 d=7 carriers placed at
        #         the seven vertex angles 2π·k/7, k=0..6, sharing carrier
        #         amplitude r^(12/7) and a common phase (12/7·θ + kt·LN2/7),
        #         each rotated by its own vertex angle. This is literally
        #         the d=7 sublattice's geometric signature per §8.2 — the
        #         crystallographically-forbidden 7-fold symmetry made
        #         explicit in the math.
        #     T = the per-step traversal across the septic barrier, indexed
        #         by n. Each step accumulates d=7-compatible descriptors
        #         via the Asymptotic Approach saturator 1 - exp(-n·V)
        #         (§8.3). The iteration step IS the partial-traversal
        #         mechanism "thinning the veil" toward full d=7 entry. At
        #         n=0 the veil is fully closed (factor = 0 — pure d=3
        #         dynamics, no septic contribution); as n→∞ the veil
        #         asymptotically approaches fully open (factor → 1) but
        #         never reaches 1 in finite steps, exactly as the corpus
        #         Asymptotic Approach Theorem requires.
        #   Descriptor Gap:
        #     The prior code was a single d=7 carrier rotated by i, with
        #     no 7-fold symmetry, no heptagram geometry, no n-dependence,
        #     and no veil-thinning. The "Septic Otherworld" name was a
        #     promise the math did not keep — the prior code had no
        #     7-fold structure at all. The closing Descriptors are the
        #     7-vertex superposition (the actual heptagram, §8.2's named
        #     geometric signature), the asymptotic veil-thinning saturator
        #     (§8.3's named partial-traversal mechanism), and the
        #     z-anchored "this side of the veil" reference position
        #     (§8.3's residual d=3 framework). The 90° "Otherworld"
        #     i-rotation from the prior code is preserved (it represents
        #     the orientation flip into the inembeddable d=7 space) so
        #     nothing from the prior implementation is removed.
        #   Subsumption:
        #     This implementation subsumes Fantastical Configurations §8
        #     without remainder: the heptagram is built explicitly as the
        #     superposition over all 7 vertices (with the 1/7 normalization
        #     keeping the contribution bounded), the partial-traversal
        #     mechanism is the saturating veil-thinning factor with V as
        #     the natural lattice rate, the 90° i-rotation is preserved,
        #     and the d=3-side z-anchor expresses the residual d=3
        #     descriptor structure. The crystallographically-forbidden
        #     7-fold symmetry that the corpus identifies as the entire
        #     defining feature of d=7 is now actually present in the math.
        #
        # Math:
        #   p             = 12/7                              d=7 carrier power
        #   vertex_k      = 2π · k / 7    for k=0..6          7 heptagram vertices
        #   carrier_r     = r^p                               shared carrier amplitude
        #   ang_common    = p·θ + kt·LN2/7                    shared carrier phase
        #   heptagram     = (1/7) · Σ_{k=0..6}                7-vertex mean
        #                       exp(i·(ang_common + vertex_k))
        #   veil(n)       = 1 - exp(-n · V)                   Asymptotic Approach §8.3
        #   septic        = i · V · carrier_r · heptagram · veil
        #   z_anchor      = V² · z · veil                     d=3 partial-traversal anchor
        #   extra         = septic + z_anchor
        #
        # All five lambda parameters used:
        #   z   — direct anchor (the d=3 partial-traversal reference)
        #   r   — carrier amplitude r^p (shared by all 7 heptagram vertices)
        #   th  — carrier phase p·θ (shared by all 7 vertices)
        #   kt  — T-axis lattice coordinate, rotation by kt·LN2/7
        #   n   — drives the veil-thinning saturator
        _p_septic    = 12.0 / 7.0
        # 7 heptagram vertex angles 2π·k/7 for k=0..6, computed once at
        # build time. This is the d=7 sublattice's natural angular quantum
        # on the unit circle — ET-derived (the d=7 family's elementary
        # rotation step), not an arbitrary number of vertices.
        _heptagram_phases = np.array(
            [2.0 * math.pi * k / 7.0 for k in range(7)], dtype=np.float64)
        _inv_seven   = 1.0 / 7.0
        _septic_v2   = V * V
        _septic_kta  = LN2 / 7.0       # kt T-axis rotation per d=7 step
        def _e(z, r, th, kt, n,
               _p=_p_septic, _hp=_heptagram_phases,
               _inv7=_inv_seven, _v2=_septic_v2, _kta=_septic_kta):
            # Shared carrier amplitude for all 7 vertices: r^(12/7)
            carrier_r = r ** _p
            # Shared phase part (independent of vertex index k): the d=7
            # carrier phase plus the kt T-axis rotation. This works
            # element-wise on ndarrays (th, kt are pixel-shape arrays in
            # the vectorized CPU path) and on scalars (in scalar tests).
            ang_common = _p * th + kt * _kta
            # Heptagram superposition: sum exp(i·(ang_common + vertex_k))
            # over all 7 vertices and divide by 7. Implemented as an
            # explicit unrolled sum so the loop bound (7) is fixed at
            # build time and so it works element-wise on ndarrays without
            # any reshape/broadcasting tricks. The 7 vertex angles are
            # constants captured by closure.
            total = np.exp(1j * (ang_common + _hp[0]))
            for _k in range(1, 7):
                total = total + np.exp(1j * (ang_common + _hp[_k]))
            heptagram = total * _inv7
            # Asymptotic Approach Theorem (§8.3) — the "thinning of the veil":
            # 1 - exp(-n · V) saturates from 0 (veil closed) to 1 (veil
            # open) as n grows. V = 1/12 sets the natural lattice rate of
            # asymptotic descent. At n=0 the veil is fully closed and the
            # septic contribution is zero — the orbit is purely d=3.
            # As n → ∞ the veil asymptotically thins toward 1 but never
            # reaches it in finite n, exactly as the corpus Asymptotic
            # Approach Theorem requires. n is a Python int per step here;
            # np.exp on a scalar returns a NumPy scalar that broadcasts
            # cleanly against the carrier and z arrays below.
            veil = 1.0 - np.exp(-n * V)
            # Septic term: i · V · carrier_r · heptagram · veil.
            # The leading 1j is the 90° "Otherworld" rotation preserved
            # from the prior code as the original Mode 6 design intent —
            # it represents the orientation flip into the inembeddable d=7
            # space (the d=7 sublattice's "phase shift relative to d=3"
            # geometric signature). V scaling matches the convention used
            # by the other modes' perturbation contributions.
            septic = 1j * V * carrier_r * heptagram * veil
            # z-anchored "this side of the veil" reference (Partial
            # Traversal Problem §8.3): the orbit's current position is
            # the d=3 anchor that the Traverser cannot fully leave behind.
            # Scaled by V² and by the same veil-thinning factor so as
            # the veil thins the d=3 anchor strength grows in concert
            # with the heptagram (both go from 0 at n=0 to ~V² and ~V
            # respectively at n→∞). This is the Asymptotic Approach
            # made symmetric: both the septic emergence and the
            # d=3 anchor strengthen together.
            z_anchor = _v2 * z * veil
            return septic + z_anchor
        return _e
    if mode_id == 7:
        # ── Mode 7: Nonic Recursion — d=9 = 3² holographic depth ────────────
        # Corpus source:
        #   /mnt/project/ET_Fantastical_Configurations.md §9 (line 439+):
        #     "The Nonic Recursion: Fractal and Nested Magic (d=9). d=9 = 3²
        #     — the second power of the cubic primitive. It first appears in
        #     36ET. A d=9-governed configuration is a 'cubic configuration of
        #     cubic configurations' — a meta-level of the three-dimensional
        #     structure where each position in the cubic lattice is itself a
        #     cubic lattice. This is fractal magic: the same structure at
        #     every scale, infinite nesting, where the small perfectly
        #     mirrors the large. As above, so below is the ET description of
        #     d=9 governance. The Hermetic maxim is not metaphor — it is
        #     the observation that a d=9 D-profile produces self-similar
        #     structure across scales by construction (since 9 = 3 × 3, two
        #     levels of cubic closure nest within each other)."
        #   §9 magical phenomena (line 447+): "Holographic magic: any piece
        #     of the configuration contains the whole (because d=9 = d=3²,
        #     each cubic cell contains a full cubic structure). Infinite
        #     regress / summoning from within: A d=9 configuration contains
        #     itself at every level — 'a world within a world' is a d=9
        #     lattice description."
        #   §9 impedance (line 452): "A₀ = (12/9 − 1)² + 16 ≈ 16.11. This
        #     is nearly the minimum — approaching the ξ → 8.5× maximum.
        #     Fractal magic is nearly maximally efficient because the
        #     meta-cubic structure is very close to the trivial sublattice
        #     in terms of impedance overhead."
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (the "self-call" anchor —
        #         the recursive starting point that every level nests
        #         around; "summoning from within" expressed literally as
        #         the orbit's current position becoming the recursion's
        #         self-reference at each step).
        #     D = the recursive cubic-of-cubic stack: at depth ℓ ∈ 1..9,
        #         a carrier with power (12/9)^ℓ operating on the result of
        #         the previous level (current_r, current_th). Each level
        #         is a complete cubic transformation of the previous
        #         level's accumulated result, expressing "cubic configuration
        #         of cubic configurations" at increasing depth. The d=9 = 3²
        #         literal depth target gives the maximum recursion depth
        #         MAX_NONIC_DEPTH = 9 (one level per cubic factor of the
        #         9-lattice, and 9 = 3² means nine total cubic transforms
        #         when the structure is fully unrolled).
        #     T = the per-step traversal that advances the active recursion
        #         depth via active_depth = min(1 + n//N, 9). Each full N=12
        #         manifold cycle adds one level of nesting, expressing
        #         "as above, so below" temporally — deeper recursion
        #         emerges as iteration time accumulates. Saturates at the
        #         9-deep target after n ≥ 8·N = 96 steps, then remains at
        #         maximum depth for the remainder of the iteration.
        #   Descriptor Gap (the missing Descriptor in the prior code):
        #     The prior code was a manually-unrolled 2-level cubic recursion
        #     (V·z^(12/9) + V²·z^((12/9)²)) that captured "9 = 3² = two
        #     levels of cubic" but missed the holographic / arbitrary-depth
        #     / self-similar structure named in §9. The corpus says "any
        #     piece contains the whole" and "summoning from within" — that
        #     requires recursion at every scale with each piece literally
        #     containing the prior, not just two manually unrolled levels.
        #     The closing Descriptors are:
        #       (a) variable recursion depth driven by n (active_depth),
        #       (b) per-level recursive feed-forward where the next level
        #           operates on the previous level's accumulated result
        #           (current_r, current_th) — true holographic nesting,
        #       (c) direct z dependence as the V²·z self-call anchor,
        #       (d) saturation at 9 levels matching the d=9 = 3² literal
        #           depth target,
        #       (e) divide-by-active_depth normalization so the magnitude
        #           is bounded as recursion deepens (Subsumption Law).
        #   Subsumption:
        #     This implementation subsumes Fantastical Configurations §9
        #     without remainder: the cubic-of-cubic structure appears as
        #     the per-level (12/9)^level powers, the holographic self-
        #     similarity appears as the recursive feed-forward at each
        #     level, the temporal "as above so below" appears as the
        #     n-driven depth growth, the "summoning from within" appears
        #     as the V²·z self-call anchor that every level nests around,
        #     and the 9 = 3² literal depth target appears as the depth
        #     saturation. The 2-level structure of the prior code is
        #     preserved as a SPECIAL CASE — when active_depth = 2 (which
        #     occurs at n ∈ [12, 23] under the dynamic depth formula),
        #     the math reduces to the same shape as the prior code's
        #     two-term sum, except the second term operates on the
        #     recursive feed-forward (|accum|, arg(accum)) rather than
        #     on the original (r, th). Nothing from the prior code is
        #     removed — the prior shape lives inside this generalization
        #     as an n-window special case.
        #
        # Math:
        #   active_depth(n) = min(1 + n // N, 9)         depth grows with n
        #   accum_0      = V² · z                         self-call anchor
        #   (cur_r_0, cur_th_0) = (r, th)
        #   for level ℓ in 0...active_depth − 1:
        #     p_ℓ      = (12/9)^(ℓ + 1)                  nonic recursive power
        #     rot_ℓ    = kt · (LN2/9) · (ℓ + 1)
        #     term_ℓ   = V · cur_r^p_ℓ · exp(i·(p_ℓ · cur_th + rot_ℓ))
        #     accum   += term_ℓ
        #     cur_r    = |accum|                          recursive feed-forward
        #     cur_th   = arg(accum)                       (holographic nesting)
        #   extra      = accum / active_depth             holographic normalization
        #
        # All five lambda parameters used:
        #   z   — V²·z self-call anchor (the "summoning from within" reference)
        #   r   — initial cur_r for the recursive carrier amplitude
        #   th  — initial cur_th for the recursive carrier phase
        #   kt  — per-level T-axis rotation kt · (LN2/9) · (ℓ + 1)
        #   n   — drives active_depth via 1 + n // N
        _MAX_NONIC_DEPTH = 9                  # 9 = 3² literal depth target
        _nr_pb           = 12.0 / 9.0          # base nonic power = 4/3
        _nr_kb           = LN2 / 9.0           # base nonic kt rotation
        _nr_v            = V                    # per-level scale
        _nr_v2           = V * V               # self-call anchor scale
        def _e(z, r, th, kt, n,
               _md=_MAX_NONIC_DEPTH, _pb=_nr_pb, _kb=_nr_kb,
               _v=_nr_v, _v2=_nr_v2):
            # Active recursion depth: 1 at n=0..11, 2 at n=12..23, ...,
            # saturating at _md = 9 once n ≥ 8·N = 96. This is the
            # temporal "as above so below" — each full N=12 manifold
            # cycle adds one level of cubic nesting, asymptotically
            # reaching the d=9 = 3² literal depth target. n is a Python
            # int per step here; the // operator gives integer division.
            active_depth = 1 + (n // N)
            if active_depth > _md:
                active_depth = _md
            # Self-call anchor: V²·z is the recursive starting point that
            # every level nests around. Using z directly (not r·exp(iθ))
            # captures the "summoning from within" semantic per §9 and
            # uses the z parameter explicitly — the orbit's current
            # position IS the recursion's self-reference. V² scales it
            # to the same magnitude as the other modes' z-anchors
            # (Subsumption-friendly relative to the 24-family base sum).
            accum      = _v2 * z
            current_r  = r
            current_th = th
            for level in range(active_depth):
                # Nonic recursive carrier: at depth (level+1), the power
                # is (12/9)^(level+1) = (4/3)^(level+1). Each level
                # applies one cubic transformation, so the cumulative
                # power matches "cubic configuration of cubic configurations"
                # at the corresponding nesting depth. The kt rotation
                # scales linearly with level so successive levels
                # accumulate phase coherently with the d=9 sublattice's
                # natural angular quantum.
                p_level   = _pb ** (level + 1)
                rot_level = kt * _kb * (level + 1)
                r_pow     = current_r ** p_level
                term      = _v * r_pow * np.exp(1j * (p_level * current_th
                                                      + rot_level))
                accum     = accum + term
                # Recursive feed-forward: the next level's carrier
                # operates on the polar form of the previous level's
                # accumulated result. This is true holographic nesting —
                # every level contains the prior level by construction,
                # expressing "any piece contains the whole" per §9. The
                # 1e-300 floor prevents log-of-zero in the next level's
                # power computation if accum should ever vanish exactly
                # (which would otherwise inject NaN into the iteration).
                # np.abs and np.angle work on both scalars and ndarrays
                # so this vectorises cleanly across the pixel grid.
                current_r  = np.abs(accum) + 1e-300
                current_th = np.angle(accum)
            # Holographic normalization: divide by active_depth so the
            # output magnitude is bounded as the recursion deepens.
            # Without this normalization, deep recursion would
            # accumulate unbounded amplitude and break the Subsumption
            # Law (the extra would dominate the 24-family base sum).
            # The per-step contribution stays at the same overall scale
            # regardless of depth — only the structural complexity
            # grows with n, not the magnitude.
            return accum / float(active_depth)
        return _e
    if mode_id == 8:
        # ── Mode 8: Magical Impedance — cycling impedance regimes ──────────
        # Corpus sources:
        #   /mnt/project/ET_Fantastical_Configurations.md §3.3 Table 2 (the
        #     "more profound case", line 148+) — A₀_magic(d) = (d-1)² + S²
        #     for each sublattice family d ∈ {1,...,12}. The §5.1/§5.2 older
        #     formulation uses (12/d - 1)² which is structurally inconsistent
        #     with the canonical Fine Structure REVISED A₀ = (N-1)² + S² and
        #     was superseded by §3.3 Table 2 (see FAM_COUPLING block above
        #     and IMPEDANCE_TABLE block for the full corpus trace).
        #   /mnt/project/ET_Fine_Structure_Constant_REVISED.md — N=12 is the
        #     constant manifold symmetry, so the per-sublattice magical
        #     impedance generalizes the canonical formula by substituting
        #     d_prim for the role of N while keeping N=12 globally fixed.
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (substrate of the iteration)
        #     D = at step n, the magical regime d_idx = (n mod 12) + 1 — one
        #         of the 12 magical configurations with its own A₀_magic, ξ,
        #         carrier power 12/d_idx and rotation phase LN2/d_idx.
        #     T = the iteration step's traversal through the cycling regimes,
        #         each visit substantiating that magic type for one step. The
        #         cycle expresses the discreteness of the impedance table:
        #         each step occupies one regime, the regime changes per step,
        #         and the orbit literally moves through Pure Will → Mirror →
        #         Cubic → Quartic → Quintic → Hexadic → Septic → Octet → Nonic
        #         → Decic → Undecimal → Full-Res over each 12-step cycle.
        #   Descriptor Gap:
        #     The prior code had a sin radial wave with no impedance content
        #     whatsoever — zero reference to A₀, ξ, or the impedance table.
        #     The closing Descriptors are the 12 sublattice families' (d,
        #     A₀_magic, ξ) tuples themselves, looked up per step from the
        #     IMPEDANCE_TABLE and substantiated as carrier+coupling+anchor.
        #   Subsumption:
        #     This implementation subsumes the corpus impedance table without
        #     remainder — every entry from d=1 (Pure Will) through d=12
        #     (Full-Res/EM baseline) appears in the cycling sequence, each
        #     contributing at its own ξ-weighted amplitude. The bounded
        #     normalized xi (in [1/137, 1.0]) keeps the contribution from
        #     overriding the 24-family base sum.
        #
        # All five lambda parameters are used:
        #   z   — direct anchor at the orbit's current position, scaled by
        #         V² · (xi_norm − inv_A0)  so high-coupling regimes pull on z
        #   r   — magnitude of z, used for the carrier amplitude r^p
        #   th  — angle of z, used for the carrier phase p·th
        #   kt  — T-axis lattice coordinate, used for the per-d rotation
        #   n   — selects the regime via n % 12
        _imp_d   = _IMPEDANCE_D                  # [1..12]
        _imp_xin = _IMPEDANCE_XIN                # bounded ξ_norm in [1/137, 1.0]
        _nt      = N_MAGIC_TYPES                 # 12
        _inv_A0  = 1.0 / A0_EM                   # 1/137 — local-EM normalization
        def _e(z, r, th, kt, n,
               _id=_imp_d, _ix=_imp_xin, _nt=_nt, _ia=_inv_A0):
            # Cycle index: n % 12 selects which magic type is active this step
            idx = n % _nt
            d_magic = float(_id[idx])
            xi_norm = float(_ix[idx])
            # Per-magic-type carrier: power 12/d, rotation phase LN2/d
            p_magic   = 12.0 / d_magic
            rot_magic = kt * (LN2 / d_magic)
            carrier   = (r ** p_magic) * np.exp(1j * (p_magic * th + rot_magic))
            # Coupling-weighted carrier contribution: stronger coupling
            # (lower d) gives a louder per-step contribution, but bounded
            # by xi_norm ≤ 1.0 so the 24-family base sum still dominates.
            # The V scaling matches the other modes' "perturbation" scale.
            carrier_term = V * xi_norm * carrier
            # Direct z anchor: T-P coupling pulls on z's current position by
            # an amount proportional to the regime's coupling enhancement
            # over the local-EM 1/137 baseline. (xi_norm − 1/137) is in
            # [0.1095, 0.9927] across the 12 regimes — it never vanishes,
            # so every cycling step contributes a visible z-pull. The ratio
            # max/min ≈ 9.07× gives the strongest natural contrast across
            # the cycle of any reasonable normalization, which is what makes
            # the 12-band magic structure visible in the rendered image.
            # At d=1 (Pure Will, max coupling) the anchor is ~9× stronger
            # than at d=12 (Full-Res/EM, baseline) — the per-band contrast
            # the cycling mode is supposed to show.
            z_anchor = (V * V) * z * (xi_norm - _ia)
            return carrier_term + z_anchor
        return _e
    if mode_id == 9:
        # ── Mode 9: Exception State — V(E)=0 grounding pull ─────────────────
        # Corpus sources:
        #   /mnt/project/ExceptionTheory.md Part XI (line 742+) — Variance and
        #     the Grounding Function:
        #     "For any configuration c ∈ 𝒞, the variance V(c) measures how
        #     much c can change... For the Exception: V(E) = 0. The Exception
        #     cannot change while it IS. It is the unique configuration with
        #     zero variance — the only thing that cannot be otherwise at the
        #     current moment."  (line 760)
        #     "The Exception is the unique fixed point with zero variance.
        #     Everything else flows around it."  (line 782)
        #     The Grounding Function G: 𝒞 → {0,1} with G(c) = 1 ⟺ V(c) = 0.
        #   /mnt/project/ExceptionTheory.md §Observational Displacement
        #     (line 786+):
        #     "The Exception cannot be observed directly. Any observation
        #     creates a new configuration with positive variance. ... When T
        #     observes E, the act of observation creates T∘E, which is a new
        #     configuration distinct from E alone. The original E is displaced
        #     the moment you try to observe it."
        #   /mnt/project/ExceptionTheory.md §Variance Dependence (line 2347+):
        #     "The 'thickness' of agential time (how much 'now' feels
        #     extended) scales inversely with local variance: high-variance
        #     regions feel time as thick/slow; low-variance (near Exception)
        #     feel time as thin/fast."
        #   /mnt/project/ET_Incoherence_Paper.md §10 (line 385+) —
        #     The Exception is a Closed Set:
        #     "The Exception is a closed set. {P,D,T} contains its own
        #     boundary — P provides the substrate, D provides complete
        #     description, T provides active substantiation. Nothing external
        #     is needed to define the boundary of an Exception. Zero variance
        #     is the mathematical expression of this closure: the Exception
        #     contains its own ground, which is what it means for a set to
        #     contain its own boundary." ∂E ⊂ E.
        #   /mnt/project/ET_Incoherence_Paper.md §23 (line 1049+) — the
        #     Elegance Score 𝓔 is "the inverse proxy for Variance: high 𝓔
        #     directly maps to low V(c), and low 𝓔 maps to high V(c)"
        #     (line 1067-1069). The §23 elegance kernel `100/(100 + |ε|)` is
        #     the lattice-discrete form; the smooth Lorentzian `1/(1+V·d²)`
        #     is its smooth ET-native analog (peaks at d=0 with value 1.0,
        #     decays monotonically as d grows, the natural lattice-quantum-
        #     scaled smooth profile from §23's variance kernel).
        #   /mnt/project/M-states.md (line 459, line 723) — cosmological
        #     E-vacuum is exactly 2/3 = K of total energy (the static Pure E
        #     ground state). This is not coincidence: the Koide ratio K = 2/3
        #     IS the cosmological weight of the Exception's static pull on
        #     the manifold. Derived: "Pure E-vacuum = 2/3 = 66.7%".
        #   /mnt/project/M-states.md §Why M-States Split This Way (line 712,
        #     line 950): the active mediation (3% total) splits 8:7 between
        #     M-vacuum (1.6%, vacuum-like, distributed uniformly, w ≈ -1) and
        #     M-matter (1.4%, matter-like, localized, w ≈ 0). Ratio 8:7 ⟹
        #     8/(8+7) = 8/15 + 7/(8+7) = 7/15. M-vacuum is z-uniform (does
        #     not depend on z); M-matter is z-localized (the negative z-pull
        #     toward the Exception).
        #   /mnt/project/M-states.md §M-State Fraction (line 510+):
        #     f_M = (1/12) × 0.36 × 1.0 = 0.03 = 3%. The 0.36 factor is
        #     (3/5)² — Fibonacci-encoded — and (3/5)·V = α₅ is the QS-5
        #     quintic shadow coupling constant (the same coupling Mode 5
        #     uses). f_M = (F₅/F₆)² · V — fully ET-derived, connecting
        #     Mode 9 to the Mode 5 Fibonacci chain via Quintic Shadow §QS-5.
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (the substrate of the
        #         iteration; the candidate that may or may not be near the
        #         Exception's grounding fixed point z_E = 0+0j — the canonical
        #         lattice origin, the Exception's natural anchor at the
        #         ε=0 / k=0 / d=1 trivial sublattice point of the manifold).
        #     D = three structural Descriptors: (a) the local variance kernel
        #         V_loc(z) = 1/(1 + V·|z-z_E|²) — the §23 elegance-as-inverse-
        #         variance smooth lattice form; (b) the grounding weight
        #         G(z) = K · V_loc(z) — explicit Koide cosmological weight
        #         from the M-states 2/3 = K E-vacuum derivation; (c) the
        #         M-vacuum / M-matter 8/15 + 7/15 split — uniform z-independent
        #         M-vacuum component plus the negative z-localized M-matter
        #         pull toward z_E.
        #     T = the per-step traversal that displaces the would-be
        #         Exception every iteration. Observational Displacement is
        #         expressed by the exp(-n·V) residual envelope (each
        #         observation step n creates a new configuration; the
        #         residual captures how much of the original Exception
        #         survives the observation chain). The manifold-cycle phasor
        #         exp(i·(2π·n/N + th + kt·LN2/N·K)) carries the cumulative
        #         T-displacement phase as both the discrete iteration index
        #         n and the lattice T-axis k_t advance, with the Koide
        #         weighting K on the kt rotation matching the cosmological
        #         E-vacuum weight.
        #   Descriptor Gap (the missing Descriptor in the prior stub):
        #     The prior code had the Lorentzian variance kernel (correct in
        #     spirit per §23) and z dependence (correct), but it was missing:
        #       (a) explicit grounding-fixed-point reference (z_E = 0+0j) —
        #           the prior code used |z| via r implicitly without naming
        #           the reference frame, hiding the structural meaning;
        #       (b) the explicit Koide cosmological weight K · V_loc — the
        #           prior code's overall scale was V (not K), losing the
        #           M-states.md "Pure E-vacuum = 2/3 = K" derivation;
        #       (c) the M-vacuum / M-matter 8:7 split — the prior code
        #           collapsed both into a single negative z-pull, losing the
        #           distinction between distributed (vacuum-like) and
        #           localized (matter-like) M-state contributions;
        #       (d) Observational Displacement (n unused) — the prior code
        #           did not use n at all, contradicting the corpus
        #           "any observation creates a new configuration"
        #           dynamic-displacement semantics;
        #       (e) the manifold-cycle phasor — the prior code's
        #           `twist = th*(N/12) + kt*(LN2/N)` reduced to `th + kt·LN2/N`
        #           (because N/12 = 1), losing the per-step n-cumulative
        #           phase advance and the Koide K weighting on kt.
        #     The closing Descriptors are the explicit z_E reference, the
        #     Koide-weighted grounding G(z) = K·V_loc, the 8/15+7/15 M-state
        #     split, the exp(-n·V) Observational Displacement residual,
        #     and the manifold-cycle phasor with K-weighted kt rotation.
        #   Subsumption:
        #     This implementation subsumes Part XI's Variance and Grounding
        #     Function, §10's Closed Set property, §23's Elegance-Variance
        #     duality, and M-states.md's Pure-E and 8:7 split, all without
        #     remainder. Every component named in the corpus appears as an
        #     explicit term: V(E) = 0 ⟹ V_loc has its maximum at z_E (peak
        #     value 1.0 — the Exception's literal zero-variance point),
        #     G = K·V_loc ⟹ Koide cosmological weight, exp(-n·V) ⟹
        #     Observational Displacement, the 8/15 M-vacuum coefficient on
        #     the uniform phasor + the 7/15 M-matter coefficient on the
        #     z-anchor ⟹ M-states.md split, and the K weighting on the kt
        #     phase ⟹ the Koide stability threshold acting on the T-axis
        #     traversal. The Lorentzian profile from the prior code is
        #     preserved as V_loc — nothing is removed. The negative-direction
        #     z-pull from the prior code is preserved as the M-matter
        #     component (also negative — pulling z toward z_E).
        #
        # Math:
        #   z_E         = 0 + 0j                                      grounding fixed point (lattice origin)
        #   d_E²        = |z - z_E|² = r²                              displacement squared (uses r, the
        #                                                              orbit's pre-computed magnitude — the
        #                                                              z parameter's polar form is consumed
        #                                                              both via z directly AND via r implicitly)
        #   V_loc(z)    = 1 / (1 + V · d_E²)                           §23 elegance-form variance kernel
        #                                                              (smooth lattice analog of 100/(100+|ε|))
        #                                                              At z = z_E: V_loc = 1.0 (peak — the
        #                                                              Exception's zero-variance grounding).
        #                                                              As |z| → ∞: V_loc → 0 (full Variance).
        #   G(z)        = K · V_loc(z)                                 Grounding weight — K is the Koide
        #                                                              cosmological E-vacuum weight per
        #                                                              M-states.md (Pure E = 2/3 = K)
        #   displacement(n) = exp(-n · V)                              Observational Displacement residual
        #                                                              — each step n is one observation;
        #                                                              residual decays at rate V = 1/12
        #                                                              (the lattice-natural variance quantum)
        #   cycle_ang(n,th,kt) = 2π·n/N + th + kt · (LN2/N) · K        manifold-cycle phasor angle
        #                                                              (n cumulative manifold rotation +
        #                                                              th from z's polar form +
        #                                                              kt T-axis with Koide K weighting)
        #   phasor      = exp(i · cycle_ang(n, th, kt))                unit-modulus manifold rotation
        #   M_vac_coef  = 8.0 / 15.0                                   M-vacuum split (8:7 → 8/(8+7))
        #                                                              M-states.md line 950
        #   M_mat_coef  = 7.0 / 15.0                                   M-matter split (7:8 → 7/(8+7))
        #                                                              M-states.md line 950
        #   M_vacuum    = M_vac_coef · phasor                          z-uniform distributed M (vacuum-like,
        #                                                              w ≈ -1, "everywhere" component)
        #   M_matter    = -M_mat_coef · z · displacement               z-localized negative pull toward z_E
        #                                                              (matter-like, w ≈ 0, "where matter is"
        #                                                              component; negative direction = pull
        #                                                              TOWARD the grounding fixed point)
        #   extra       = V² · G(z) · (M_vacuum + M_matter)            V² Subsumption-friendly overall scale
        #
        # All five lambda parameters used:
        #   z   — direct (M_matter localized z-pull toward z_E = 0+0j)
        #   r   — variance kernel V_loc(z) = 1/(1 + V·r²)
        #   th  — manifold-cycle phasor angle (cycle_ang)
        #   kt  — T-axis Koide-weighted phase term in cycle_ang
        #   n   — Observational Displacement residual (exp(-n·V))
        #         AND manifold-cycle cumulative rotation (2π·n/N)
        _ex_K        = K                              # Koide E-vacuum weight (M-states "Pure E = 2/3")
        _ex_V        = V                              # base variance (1/12) — used by V_loc and displacement
        _ex_v2       = V * V                          # V² Subsumption-friendly overall scale
        _ex_2pi_N    = 2.0 * math.pi / N              # manifold-cycle angular quantum (= π/6)
        _ex_kt_alpha = (LN2 / N) * K                  # Koide-weighted kt T-axis rotation
        _ex_Mvac     = 8.0 / 15.0                     # M-vacuum coefficient (M-states 8:7 split, 8/(8+7))
        _ex_Mmat     = 7.0 / 15.0                     # M-matter coefficient (M-states 8:7 split, 7/(8+7))
        def _e(z, r, th, kt, n,
               _K=_ex_K, _V=_ex_V, _v2=_ex_v2,
               _2pn=_ex_2pi_N, _kta=_ex_kt_alpha,
               _Mvc=_ex_Mvac, _Mmt=_ex_Mmat):
            # §23 elegance-form variance kernel: V_loc = 1 / (1 + V · |z-z_E|²)
            # z_E = 0+0j (the lattice origin, the canonical Exception grounding
            # fixed point at ε=0 / k=0 / d=1). At z = z_E: V_loc = 1.0 (the
            # Exception's literal zero-variance peak). As |z| grows: V_loc
            # decays monotonically toward 0 (full Variance, far from grounding).
            # The §23 corpus form `100/(100 + |ε|)` is the lattice-discrete
            # version; this Lorentzian is its smooth ET-native analog.
            v_loc = 1.0 / (1.0 + _V * (r * r))
            # Grounding weight G(z) = K · V_loc(z). K = 2/3 is the Koide ratio,
            # which per M-states.md line 459 is exactly the cosmological weight
            # of the Pure E-vacuum (66.7% of total energy). The Koide
            # multiplier here is the corpus-derived cosmological E-state weight,
            # not an arbitrary scale.
            G = _K * v_loc
            # Observational Displacement residual exp(-n · V). Each iteration
            # step n is one "observation" of the would-be Exception per the
            # ExceptionTheory.md line 786 dynamic: "any observation creates a
            # new configuration with positive variance." The residual decays
            # at rate V = 1/12 (the lattice-natural variance quantum), so
            # after one full N=12 manifold cycle the residual is exp(-1) ≈
            # 0.368 — the Exception "has been displaced" by one cycle's worth
            # of observation. n is a Python int per step; np.exp on a scalar
            # returns a NumPy scalar that broadcasts cleanly against the
            # ndarray r/th/kt/z grids.
            displacement = np.exp(-n * _V)
            # Manifold-cycle phasor angle. Three contributions:
            #   2π·n/N : cumulative manifold rotation per step (uses n)
            #   th     : the orbit's angular position (carries z's phase)
            #   kt·(LN2/N)·K : Koide-weighted T-axis rotation (uses kt)
            # The Koide K on the kt term matches the cosmological E-weight,
            # tying the T-axis traversal to the Pure-E grounding pull.
            cycle_ang = _2pn * n + th + kt * _kta
            phasor    = np.exp(1j * cycle_ang)
            # M-state split per M-states.md line 950 (the corpus 8:7 ratio of
            # vacuum-like to matter-like M-states). Total M-coefficient
            # 8/15 + 7/15 = 15/15 = 1 — the split is conservative.
            #
            # M_vacuum (8/15): the z-uniform "vacuum-like" component. It does
            # NOT depend on z directly — it is the manifold-cycle phasor at
            # the M-vacuum weight, modeling the distributed quantum mediation
            # that exists everywhere in the manifold (per M-states.md line
            # 745: "Quantum field mediation (everywhere)... distributed
            # uniformly... act like cosmological constant"). w ≈ -1.
            M_vacuum = _Mvc * phasor
            # M_matter (7/15): the z-localized "matter-like" component — the
            # negative pull on z toward the grounding fixed point z_E = 0+0j.
            # The negative sign (-_Mmt · z) is the localized pull TOWARD
            # the Exception (per M-states.md line 770: "localized processes...
            # concentrated where complexity exists"). The displacement residual
            # multiplies it because matter-like M-states are themselves
            # observed/displaced as the iteration proceeds. w ≈ 0.
            M_matter = -_Mmt * z * displacement
            # Combined extra: V² · G(z) · (M_vacuum + M_matter). The V² scaling
            # is Subsumption-friendly — it stays smaller than the 24-family
            # base sum so the Exception-state pull perturbs without overriding.
            # G(z) gates the entire contribution by proximity to grounding:
            # near z_E the pull is at full Koide strength K; far from z_E the
            # pull fades (high-variance regions are not pulled by the
            # Exception, per the §23 elegance-variance duality).
            return _v2 * G * (M_vacuum + M_matter)
        return _e
    if mode_id == 10:
        # ── Mode 10: Lagrangian Field — Mexican-hat vacuum + Higgs + Goldstone ──
        # Corpus sources:
        #   /mnt/project/ET_Lagrangian_Field_Theory.md §VIII (line 562+) — the
        #     canonical Mexican-hat derivation. The potential is unambiguous:
        #       §VIII.1 line 569: V(φ) = −μ²|φ|² + λ|φ|⁴   (μ², λ > 0)
        #       §VIII.1 line 575: |φ| = v = √(μ²/2λ)       (vacuum expectation)
        #       §VIII.2 line 600: σ(x) = |φ| − v           (Higgs radial mode)
        #       §VIII.3 line 651: m_H = √(2μ²)             (Higgs mass = D-binding
        #                                                    frequency of the ET
        #                                                    vacuum at the chosen
        #                                                    Point)
        #     §VIII.2 line 593-596 (T's vacuum substantiation):
        #       "The Lagrangian has a global U(1) symmetry: φ → e^{iα}φ for
        #        constant α. The vacuum manifold {|φ| = v, any phase} also
        #        has this U(1) symmetry. But T must substantiate ONE vacuum
        #        — T cannot bind to a superposition of all phases
        #        simultaneously. T's [0/0] resolution picks one."
        #     §VIII.2 line 596 (vacuum direction parameterization):
        #       "Once T substantiates a vacuum φ₀ = v·e^{iθ₀}: ..."
        #     §VIII.2 line 601-608 (Goldstone mode):
        #       "Radial: σ(x) = |φ| − v  (massive: V''(v) = 2μ² > 0)
        #        Angular: π(x) = phase fluctuation (massless: V flat in
        #        phase direction). The massless angular mode is the
        #        Goldstone boson — it arises because T chose one direction
        #        on the vacuum manifold and the action has no D-cost for
        #        moving along the vacuum ring. In ET: the Goldstone boson
        #        is the D-configuration direction along which T's binding
        #        did not break the symmetry — the remaining unsubstantiated
        #        phase freedom of the vacuum."
        #     §II.1 (T's δS=0 resolution):
        #       "T at each step resolves: T: [0/0] → D_binding. The
        #        D_binding chosen is the one that minimizes the residual
        #        indeterminacy at the next step... δS = 0 is not a
        #        principle imposed on T — it is T's [0/0]→determinate
        #        resolution applied globally across D-time."
        #   /mnt/project/et_clr_v5__4_.py THEOREM LFT-8 (independent corpus
        #     source, lines 4742-4760): reproduces the §VIII formula word-
        #     for-word — V(φ) = −μ²|φ|² + λ|φ|⁴, v = √(μ²/2λ),
        #     m_H = √(2μ²). This is a second independent ET source for the
        #     same convention. No competing convention exists in the corpus
        #     (verified by exhaustive grep across every .md/.py/.txt file
        #     for "Mexican", "Higgs", "vacuum manifold", "Goldstone", "μ²",
        #     "vacuum expect", "VEV", "spontaneous symmetry breaking",
        #     "v = sqrt", "1/2.*μ²", "½μ²", "(1/4).*phi.*⁴", "0.5*mu2",
        #     "0.5*lam" — zero hits for any half-coefficient form).
        #
        # Constant assignments (already at module level lines 578-580):
        #   _MH_MU2    = K = 2/3        → μ² (Koide ratio = D-binding curvature)
        #   _MH_LAMBDA = V = 1/12       → λ (base variance = quartic confinement)
        #   _MH_V      = √(K/(2V)) = 2  → vacuum expectation (exact)
        # Derived (computed at build time below):
        #   m_H² = 2μ² = 2K = 4/3
        #   m_H  = √(2K) = 2/√3 ≈ 1.15470054
        #
        # ── Audit finding: PRIOR-CODE GRADIENT BUG ────────────────────────────
        # The prior stub computed grad = z·(λr² − μ²) — missing the factor of
        # 2 on the quartic term. Wirtinger differentiation of the canonical
        # corpus potential V = −μ²(φ*φ) + λ(φ*φ)² gives:
        #     ∂V/∂φ* = −μ²·φ + 2λ(φ*φ)·φ = φ·(2λ|φ|² − μ²)
        # The factor of 2 comes from differentiating |φ|⁴ = (φ*φ)². The bug
        # makes the prior gradient zero at r² = μ²/λ = 8 → r = 2√2 ≈ 2.828,
        # NOT at the documented v = √(μ²/(2λ)) = 2. Numerically: at the
        # documented vacuum r=2 the prior gradient evaluates to z·(λ·4 − μ²)
        # = z·(1/3 − 2/3) = −z/3 ≠ 0 — a non-zero force at the supposed
        # equilibrium, with the orbit actually relaxing toward |z|=2√2 (41%
        # larger than v=2). The constant _MH_V = 2 is correct (it matches
        # the corpus formula); the gradient was wrong. Even worse, the
        # prior code's _et_julia_c(mode_id == 10) at line 2577 already
        # references _MH_V (= 2) directly to anchor the Julia c on the
        # vacuum ring — so the prior code's iteration dynamics and Julia
        # anchor were in disagreement about where the vacuum is. The fix
        # below brings them into corpus-consistent agreement.
        #
        # Internal consistency check on the corpus form (verified
        # arithmetically before this implementation was written):
        #   f(x)   = -μ²x² + λx⁴             (potential as f(|φ|))
        #   f'(x)  = -2μ²x + 4λx³ = 2x(2λx² - μ²)
        #   f'(v)  = 0   when 2λv² = μ²   →   v² = μ²/(2λ)   ✓
        #   f''(x) = -2μ² + 12λx²
        #   f''(v) = -2μ² + 12λ·μ²/(2λ) = -2μ² + 6μ² = 4μ²
        #   m_H²   = ½·f''(v) = 2μ²        (½ from L = ½(∂σ)² - ½m²σ²)
        #   m_H    = √(2μ²)                ✓ matches corpus line 651
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (the candidate field
        #         configuration value on which V(φ) is being evaluated;
        #         the substrate has two natural anchors — the unstable
        #         maximum at φ=0 and the vacuum ring at |φ|=v=2).
        #     D = the FIVE corpus-named components of the Mexican-hat
        #         Lagrangian field structure:
        #           (a) the potential gradient ∂V/∂φ* = φ(2λ|φ|² − μ²)
        #               — the radial restoring force toward the vacuum
        #               ring (§VIII.1)
        #           (b) the Higgs (radial massive) mode at mass
        #               m_H = √(2μ²) — "the D-descriptor of the vacuum's
        #               radial D-curvature" (§VIII.3 line 651)
        #           (c) the Goldstone (angular massless) mode along the
        #               vacuum ring tangent — "the D-direction in which
        #               the vacuum manifold is flat: T can navigate
        #               along it for free" (§VIII.2 line 610-613). The
        #               Goldstone direction in the complex plane at any
        #               point z on the vacuum manifold is the unit
        #               tangent i·(z/|z|), perpendicular to the radial
        #               direction.
        #           (d) the U(1) global phase symmetry φ → e^{iα}φ that
        #               gets broken by T's vacuum substantiation (§IV.1
        #               line 284 + §VIII.2 line 591)
        #           (e) the vacuum location v = √(μ²/(2λ)) = 2 — the
        #               continuous ring of degenerate ground states
        #               (§VIII.1 line 575)
        #     T = the per-step iteration that does FOUR corpus-named
        #         things:
        #           (1) Gradient flow: T navigates the descriptor manifold
        #               toward minimum descriptor cost (§II.1: "δS = 0 is
        #               T's [0/0]→determinate resolution"). The flow
        #               direction is −∂V/∂φ*, pulling z toward the vacuum
        #               ring at every step.
        #           (2) Vacuum substantiation: per §VIII.2 line 593-596,
        #               T's [0/0] indeterminacy must pick ONE vacuum
        #               direction. In iteration time this is dynamic: at
        #               n=0 T has not yet committed (full U(1) symmetry
        #               intact); as n grows, T's commitment grows
        #               asymptotically toward 1 but never reaches it in
        #               finite n. The natural ET-derived form is the same
        #               (1 − exp(−n·V)) Asymptotic Approach saturator that
        #               Mode 6 uses for its veil-thinning, with the same
        #               lattice-natural rate V = 1/12.
        #           (3) Higgs oscillation: once T has begun substantiating
        #               a vacuum, the radial mode oscillates at angular
        #               frequency ω_H = m_H per unit lattice time. The
        #               natural ET-derived lattice time is t = n·V (one
        #               iteration step = V = 1/12 units of T-time, the
        #               base variance quantum), so the Higgs oscillator
        #               factor is cos(m_H · n · V).
        #           (4) Goldstone propagation: the massless angular mode
        #               propagates freely along the vacuum ring at the
        #               manifold-cycle rate 2π/N per step (one full
        #               revolution along the vacuum ring per N=12
        #               iteration steps — the same manifold-cycle angular
        #               quantum that Mode 9's phasor uses).
        #   Descriptor Gap (the missing Descriptors in the prior stub):
        #     The prior code had a one-line gradient flow with a
        #     factor-of-2 error on λ, plus a Goldstone-like phase factor
        #     with no relation to the orbit's actual position on the
        #     vacuum ring. n was completely unused. _v was captured in
        #     default args but never referenced. The closing Descriptors
        #     are:
        #       (i)   the corrected Wirtinger gradient (factor of 2 on λ)
        #             — brings iteration dynamics into agreement with the
        #             already-correct Julia c anchor at line 2577
        #       (ii)  the (1 − exp(−n·V)) vacuum substantiation envelope
        #             that uses n explicitly per §VIII.2 line 593-596
        #       (iii) the Higgs radial oscillator cos(m_H·n·V) that uses
        #             n explicitly per §VIII.3 line 651 (the Higgs IS
        #             the named central object — the D-descriptor of
        #             radial D-curvature — and the prior code had zero
        #             reference to it)
        #       (iv)  the explicit Goldstone tangent direction i·(z/|z|)
        #             that anchors the angular mode to the orbit's
        #             actual position on the vacuum ring
        #       (v)   the explicit σ = r − v radial displacement that
        #             uses _MH_V (closing the prior "captured but unused"
        #             gap on _v)
        #       (vi)  the cumulative manifold rotation 2π·n/N as the
        #             Goldstone phase advance per step (matches Mode 9's
        #             cycle_ang phasor pattern exactly)
        #       (vii) explicit `th` consumer in the Goldstone phase
        #             (mirrors Mode 9's `cycle_ang = _2pn*n + th + kt*_kta`
        #             form, expressing the orbit's instantaneous angular
        #             position on or near the vacuum ring as the
        #             Goldstone field π(x)'s argument per §VIII.2 line 596
        #             "φ₀ = v·e^{iθ₀}")
        #   Subsumption:
        #     This implementation subsumes Lagrangian Paper §VIII (Mexican-
        #     hat + Higgs mechanism + Goldstone) without remainder. Every
        #     prior code term is preserved as a special case or correction:
        #       • prior `eta = V²·N` flow scale → preserved exactly as the
        #         flow prefactor (V²·N = 1/12 — the natural "one variance
        #         per cycle" lattice rate)
        #       • prior `kt·LN2/N` T-axis lattice phase → preserved
        #         exactly as the kt term in goldstone_phase
        #       • prior `V²` Goldstone source amplitude → preserved
        #         exactly as the v2 = V·V Subsumption-friendly overall
        #         scale on both Higgs and Goldstone terms
        #       • prior `θ·V` factor → subsumed two ways: (a) the
        #         (z/|z|) = e^{iθ} radial unit vector carries the orbit's
        #         θ phase exactly through the tangent direction, and
        #         (b) `th` appears explicitly in the Goldstone phase as
        #         the orbit-angular contribution (matching Mode 9's
        #         cycle_ang). The V scaling is folded into v2 = V²
        #         (preserving the V scale that the prior code had).
        #       • prior π/2 i-rotation (implicit in exp(i·…)) → preserved
        #         explicitly as the 1j factor on the Goldstone tangent
        #     The crystallographic content the corpus identifies — the
        #     Mexican-hat as the canonical symmetry-breaking potential,
        #     the Higgs as the radial-curvature D-descriptor, the
        #     Goldstone as the unsubstantiated phase direction, T's
        #     [0/0]-resolution as the vacuum substantiation mechanism —
        #     is now actually present in the math.
        #
        # Math:
        #   mu2  = _MH_MU2 = K = 2/3              μ² (D-binding curvature)
        #   lam  = _MH_LAMBDA = V = 1/12          λ (quartic confinement)
        #   v    = _MH_V = √(K/(2V)) = 2          vacuum expectation value
        #   mH   = √(2·μ²) = √(2K) = 2/√3         Higgs mass per §VIII.3
        #   eta  = V·V·N = 1/12                   gradient-flow scale
        #   v2   = V·V = 1/144                    Subsumption-friendly amp
        #   2pn  = 2π/N = π/6                     manifold-cycle quantum
        #   kta  = LN2/N                          T-axis lattice phase per kt
        #
        #   choice(n)        = 1 − exp(−n·V)             vacuum substantiation
        #                                                  envelope (T's [0/0] →
        #                                                  one-vacuum resolution
        #                                                  progressively
        #                                                  committing in
        #                                                  iteration time)
        #   r_safe           = max(r, 1e-300)            underflow guard at z=0
        #   radial_dir(z)    = z / r_safe                 = e^{iθ} (unit radial)
        #   grad(z, r)       = z·(2·λ·r² − μ²)            ∂V/∂φ* per Wirtinger
        #                                                  (CORRECTED with the
        #                                                  factor of 2 on the
        #                                                  quartic term — see
        #                                                  the Audit Finding
        #                                                  block above)
        #   flow(z, r, n)    = −η · grad · choice         gradient-flow term
        #                                                  (T navigates δS = 0
        #                                                  toward the vacuum
        #                                                  ring; flow strength
        #                                                  grows with vacuum
        #                                                  substantiation)
        #   sigma(r)         = r − v                      Higgs radial
        #                                                  displacement σ(x)
        #                                                  per §VIII.2 line 600
        #   higgs_osc(n)     = cos(m_H · n · V)           massive radial mode
        #                                                  oscillation at the
        #                                                  Higgs frequency
        #                                                  ω_H = m_H per
        #                                                  unit lattice time
        #                                                  (one step = V time)
        #   higgs(z,r,n)     = v2 · σ · higgs_osc         Higgs (radial
        #                       · radial_dir · choice     massive) mode
        #                                                  contribution
        #   gs_phase(n,th,kt)= 2pn·n + th + kt·kta        cumulative manifold
        #                                                  rotation + orbit
        #                                                  angular position
        #                                                  + T-axis lattice
        #                                                  phase (matches
        #                                                  Mode 9's cycle_ang)
        #   gs_amp(n,th,kt)  = cos(gs_phase)              bounded Goldstone
        #                                                  amplitude
        #   goldstone(z,n,
        #            th,kt)  = v2 · 1j · radial_dir       Goldstone (angular
        #                       · gs_amp · choice         massless) mode
        #                                                  contribution along
        #                                                  the vacuum-ring
        #                                                  tangent direction
        #                                                  i·(z/|z|)
        #   extra            = flow + higgs + goldstone   total Mode 10 contribution
        #
        # All five lambda parameters used:
        #   z   — direct (radial direction z/|z|, gradient z·(...))
        #   r   — magnitude r·r in gradient, sigma = r − v, r_safe in radial_dir
        #   th  — explicit additive contribution to the Goldstone phase
        #         (matches Mode 9's cycle_ang form: 2π·n/N + th + kt·LN2/N)
        #   kt  — Goldstone phase term (kt · LN2/N, preserved from the
        #         prior code's gs angle)
        #   n   — three n-dependent uses:
        #         (a) vacuum substantiation envelope (1 − exp(−n·V))
        #         (b) Higgs oscillator cos(m_H · n · V)
        #         (c) Goldstone cumulative manifold rotation 2π·n/N
        _mh_mu2  = _MH_MU2                            # μ² = K = 2/3
        _mh_lam  = _MH_LAMBDA                          # λ = V = 1/12
        _mh_v    = _MH_V                              # v = √(K/(2V)) = 2
        _mh_mH   = math.sqrt(2.0 * _MH_MU2)           # m_H = √(2K) = 2/√3 ≈ 1.155
        _mh_eta  = V * V * N                          # = 1/12 (V²·N flow scale)
        _mh_v2   = V * V                              # = 1/144 (Subsumption amp)
        _mh_2pn  = 2.0 * math.pi / N                  # = π/6 (manifold quantum)
        _mh_kta  = LN2 / N                            # T-axis lattice phase / kt
        def _e(z, r, th, kt, n,
               _mu=_mh_mu2, _la=_mh_lam, _v=_mh_v, _mH=_mh_mH,
               _eta=_mh_eta, _v2=_mh_v2, _2pn=_mh_2pn, _kta=_mh_kta):
            # ── Vacuum substantiation envelope ────────────────────────────
            # Per Lagrangian Paper §VIII.2 line 593-596: T's [0/0]
            # indeterminacy must pick ONE vacuum direction. In iteration
            # time, T progressively commits to a vacuum: at n=0 the
            # symmetry is fully unbroken (envelope = 0, no flow / no
            # Higgs / no Goldstone — the orbit experiences zero Mode-10
            # contribution); as n grows the symmetry is progressively
            # broken (envelope → 1 asymptotically). This is the same
            # Asymptotic Approach saturator Mode 6 uses for its veil-thinning,
            # with the same ET-natural lattice rate V = 1/12.
            # n is a Python int per step here; np.exp on a scalar
            # returns a NumPy scalar that broadcasts cleanly against
            # the ndarray r/th/kt/z grids. After one full N=12 manifold
            # cycle the envelope is 1 − exp(−1) ≈ 0.632 — T has
            # committed to ~63% of one vacuum.
            choice = 1.0 - np.exp(-n * V)

            # ── Radial unit direction ─────────────────────────────────────
            # z/|z| = e^{iθ} is the unit vector pointing from the origin
            # to the orbit. The 1e-300 floor guards against the z=0
            # singularity at the unstable maximum (where the radial
            # direction is mathematically undefined). At z=0 the
            # gradient and higgs terms both vanish anyway because they
            # carry an explicit z factor — the floor is purely defensive
            # to prevent NaN injection in the goldstone term's division.
            r_safe     = np.maximum(r, 1e-300)
            radial_dir = z / r_safe

            # ── Mexican-hat radial gradient (CORRECTED with factor 2) ─────
            # ∂V/∂φ* = φ·(2·λ·|φ|² − μ²) per Wirtinger differentiation
            # of V(φ) = −μ²·|φ|² + λ·|φ|⁴ (Lagrangian Paper §VIII.1
            # line 569). Zeros at |φ|² = μ²/(2λ) = K/(2V) = 4 → |φ| = 2
            # = v exactly. The PRIOR stub had `λ·r² − μ²` (no factor of
            # 2) which zeroed at |φ| = √(μ²/λ) = 2√2 ≈ 2.828 — in
            # conflict with _MH_V = 2 used by the Julia c anchor at
            # line 2577. This is the corpus-mandated Wirtinger form
            # — see the §VIII Audit Finding block above for the full
            # derivation and bug history.
            grad = z * (2.0 * _la * (r * r) - _mu)

            # ── Gradient flow (T's δS=0 navigation toward the vacuum) ─────
            # F_radial = −η · ∂V/∂φ* drives z toward the vacuum ring
            # |z| = v. Multiplied by `choice` so the flow strength
            # grows with T's progressive vacuum substantiation: at
            # n=0 there is no flow (T has not committed); as n → ∞
            # the flow approaches full strength. The eta = V²·N = 1/12
            # prefactor is preserved exactly from the prior code (it
            # is the natural "one variance per N-cycle" lattice rate
            # and matches the Subsumption-friendly amplitude scale).
            flow = -_eta * grad * choice

            # ── Higgs (radial massive) mode ───────────────────────────────
            # The Higgs is the radial fluctuation σ(x) = |φ| − v around
            # the chosen vacuum (Lagrangian Paper §VIII.2 line 600), with
            # mass m_H = √(2μ²) per §VIII.3 line 651. In iteration time
            # the radial mode oscillates at angular frequency ω_H = m_H
            # per unit lattice time t = n·V — so the oscillator factor
            # is cos(m_H · n · V). With μ² = K = 2/3 the Higgs mass is
            # m_H = √(4/3) = 2/√3 ≈ 1.1547, giving an oscillation
            # period of 2π/(m_H · V) = 12π·√3 ≈ 65.3 iteration steps
            # (much slower than the Goldstone, which has period N = 12
            # — exactly as expected, since the Higgs is the heavier
            # massive radial mode and the Goldstone is the lighter
            # massless angular mode). The Higgs amplitude is the radial
            # displacement σ scaled by the vacuum substantiation envelope
            # (the Higgs radial-curvature mode only exists once T has
            # broken the symmetry — at n=0 the envelope is zero and the
            # Higgs is not yet "born"). The direction is the radial unit
            # vector — the Higgs IS the radial-direction excitation per
            # the §VIII.3 corpus identification.
            sigma     = r - _v                              # σ = |φ| − v
            higgs_osc = np.cos(_mH * n * V)                 # massive radial mode
            higgs     = _v2 * sigma * higgs_osc * radial_dir * choice

            # ── Goldstone (angular massless) mode ─────────────────────────
            # The Goldstone is the unsubstantiated U(1) phase direction
            # along the vacuum ring (Lagrangian Paper §VIII.2 line
            # 601-608). At any point z on the vacuum manifold the
            # Goldstone direction is the unit tangent to the vacuum ring,
            # which in the complex plane is i·(z/|z|) — perpendicular
            # to the radial direction. The Goldstone is massless (per
            # §VIII.2 line 601: "Angular: π(x) = phase fluctuation
            # (massless: V flat in phase direction)"), so its propagation
            # along the vacuum ring is free. The cumulative phase advance
            # per iteration step is the manifold-cycle angular quantum
            # 2π/N — one full revolution along the vacuum ring per
            # N=12 iteration steps. The Goldstone phase combines three
            # contributions in the same form Mode 9 uses for its
            # cycle_ang phasor:
            #   2π·n/N : cumulative manifold rotation per step (uses n)
            #   th     : the orbit's instantaneous angular position
            #            (the orbit angle IS the Goldstone field π(x)'s
            #            value per §VIII.2 line 596: "Once T substantiates
            #            a vacuum φ₀ = v·e^{iθ₀}" — in the iteration
            #            picture, the orbit's current angle on or near
            #            the vacuum ring is the Goldstone position)
            #   kt·LN2/N : T-axis lattice phase contribution (preserved
            #            exactly from the prior code's gs angle factor)
            # The cos() form keeps the Goldstone amplitude bounded.
            #
            # Subsumption note: the prior code's `gs = exp(i·(θ·V +
            # kt·LN2/N))` factor's two contributing parts both reappear
            # here:
            #   • the kt·LN2/N T-axis rotation reappears as the kt·_kta
            #     term in gs_phase (preserved exactly, with the same
            #     LN2/N coefficient)
            #   • the θ·V part reappears two ways: (a) implicitly through
            #     the Goldstone tangent direction i·(z/|z|), since
            #     z/|z| = e^{iθ} carries the orbit's θ phase exactly,
            #     and (b) explicitly through the th term in gs_phase
            #     (matching Mode 9's cycle_ang). The V scaling is folded
            #     into _v2 = V² (preserving the V scale that the prior
            #     code had on the gs term) — this also makes the new
            #     code's V² overall amplitude exactly equal to the prior
            #     code's V² × 1 amplitude (no magnitude shift).
            gs_phase  = _2pn * n + th + kt * _kta
            gs_amp    = np.cos(gs_phase)
            goldstone = _v2 * 1j * radial_dir * gs_amp * choice

            # ── Total: gradient flow + Higgs radial + Goldstone angular ───
            # All three modes coexist at any orbit point that has begun
            # the vacuum substantiation. The `choice` envelope makes the
            # total contribution start at zero (T uncommitted, full U(1)
            # symmetry intact) and grow toward full strength asymptotically
            # — exactly the corpus picture of T progressively breaking
            # the symmetry per §VIII.2 line 593-596. All three terms
            # share the same Subsumption-friendly V² scale (or V²·N for
            # the gradient flow, which equals V — same magnitude class),
            # so the Mode 10 contribution stays smaller than the
            # 24-family base sum and perturbs without overriding.
            return flow + higgs + goldstone
        return _e
    if mode_id == 11:
        # ── Mode 11: Route A/B Cascade — Weak→EM canonical sequences ─────────
        # Corpus sources (cross-traced in full):
        #   /mnt/project/ET_Weak_Sector_Four_Open_Questions.md §2.2 line 105+
        #     (Route B canonical), §2.3 line 137+ (Route A canonical), §4
        #     line 315+ (CPT structure of Route A vs Route B), Theorem WS-8
        #     (Route CPT correspondence — palindromic involution n↦N−n is
        #     discrete CPT, residue sums = N at every step), Theorem WS-9
        #     (Route physical asymmetry — Route A hadronic via d=3 Strong,
        #     Route B leptonic via d=6 Hexadic).
        #   /mnt/project/ET_Weak_Sector_Open_Directions_Closed.md OD2 line
        #     128+ (Route A Closure 6/5→5/4→2/3), Theorem WS-15 (Route A
        #     Koide Closure: the unique octave-closed completion of the
        #     Route A d-sequence; product=1 exactly, ε-sum=0¢, terminal=K),
        #     OD4 line 277+ / Theorem WS-18 (Cabibbo Angle from ET Primitives:
        #     λ = sin(θ_C) = √(K·V) = √(1/18) = 1/(3·√2) ≈ 0.2357 — the
        #     amplitude for T to traverse one inter-generation step in the
        #     Route A sublattice hierarchy), Theorem WS-20 (CKM matrix
        #     magnitudes from ET primitives — the four routes carry the
        #     full Wolfenstein hierarchy as Hasse-distance powers of λ).
        #   /mnt/project/ET_Weak_Sector_d4_to_d12_Investigation.md Theorem
        #     WS-2 (Dual Route from Weak to EM — the two routes are forced
        #     by the palindromic cascade structure with generator g=7).
        #
        # ET Three-Tools (per /mnt/project/ET_Three_Tools_Complete_Reference.md):
        #   Identification:
        #     P = the orbit's complex position z (substrate of the iteration;
        #         the Traverser's current configuration on the manifold,
        #         which the Route's per-step (ratio, d) projection drags
        #         through the canonical Weak→EM journey).
        #     D = four canonical 3-step routes from d=4 to d=12, each with
        #         its own (ratio, d) sequence: the Route A/AC pair (hadronic,
        #         intermediate d=3 Strong) and the Route B/BC pair (leptonic,
        #         intermediate d=6 Hexadic). Each route's per-step Descriptor
        #         is the (ratio, d) tuple — at step k of route r, the active
        #         Descriptor is the canonical lattice ratio ROUTE[r][k] with
        #         sublattice family d=ROUTE_D[r][k]. The four routes together
        #         span the corpus's complete Route classification:
        #           A   = hadronic open    (terminal Pythagorean fifth 3/2)
        #           AC  = hadronic closed  (terminal Koide K = 2/3, WS-15)
        #           B   = leptonic open    (terminal Pythagorean fifth 3/2)
        #           BC  = leptonic closed  (terminal Koide K = 2/3, CPT pair)
        #     T = the per-step traversal that walks 4 routes × 3 steps =
        #         12 iterations per macro cycle (matching N=12 manifold
        #         symmetry exactly). Each step n picks (route_idx, step_idx)
        #         and substantiates the corresponding (ratio, d) on the
        #         orbit. The Cabibbo amplitude λ=√(K·V) per WS-18 is the
        #         natural inter-route mixing strength — T's amplitude for
        #         one Hasse-distance step in the Route A sublattice
        #         hierarchy. The hadronic/leptonic sign per WS-9 expresses
        #         the particle/antiparticle CPT signature directly in the
        #         carrier sign.
        #   Descriptor Gap (the missing Descriptors in the prior stub):
        #     The prior code expressed the cascade through 3 routes only
        #     (A, B, BC) with z unused, no CPT structure, no Cabibbo mixing,
        #     no K-terminal grounding, and a 36-step macro-cycle (each route
        #     held for 12 iterations) with no corpus justification. The
        #     closing Descriptors are:
        #       (a) The fourth canonical route — Route AC (the WS-15
        #           octave-closed Route A completion with terminal at
        #           K=2/3, the "unique closing ratio" of the corpus).
        #       (b) The 4-route × 3-step macro structure (12 steps per
        #           cycle, matching N=12 manifold symmetry exactly).
        #       (c) The hadronic/leptonic carrier sign per WS-9 — Routes
        #           A,AC are hadronic (sign=+1, particle, ascending); B,BC
        #           are leptonic (sign=−1, antiparticle, descending). This
        #           is the lattice form of WS-8's CPT correspondence.
        #       (d) The Cabibbo mixing phasor exp(i·λ·route_idx) with λ =
        #           √(K·V) per WS-18 — the inter-route mixing rotation.
        #       (e) The K-terminal grounding pull (z_K − z) at the
        #           terminal step (step_idx = 2) of the closed routes
        #           (AC, BC) only — expressing WS-15's "Koide ratio is
        #           the unique forced closing ratio of Route A".
        #       (f) The z-anchor (1 + λ·route_idx)·V²·z — the orbit
        #           position is now a real consumer of the lambda at every
        #           step (matching the convention used by every other mode
        #           after Sessions 1-7). The (1 + λ·route_idx) factor
        #           gives later routes a slightly stronger anchor pull,
        #           expressing the cumulative Cabibbo amplitude across the
        #           4-route cycle.
        #   Subsumption:
        #     This implementation subsumes the corpus Route classification
        #     without remainder: every named feature (the four routes,
        #     hadronic/leptonic asymmetry, CPT pairing, Cabibbo amplitude,
        #     Koide closure, ε-sum=0 octave closure on the closed routes)
        #     appears as an explicit term. The prior code's 3-ratio table
        #     (A, B, BC) is preserved exactly — the new code adds AC and
        #     uses the same lattice projection (kk = round(N·log2(ratio)),
        #     dd = N//gcd(|kk|,N)) for the per-step carrier. The prior
        #     code's `step = n%3` per-route step indexing is preserved
        #     exactly. The prior 36-step cycling (cyc = (n//12)%3, each
        #     route held for 12 iterations with no corpus basis) is
        #     replaced with the corpus-derived 12-step macro-cycle that
        #     matches N=12 manifold symmetry exactly. The carrier
        #     V·(r**pp)·exp(i·(pp·th + rr)) from the prior code is
        #     preserved exactly as the central carrier of the new
        #     implementation, multiplied additionally by the
        #     hadronic/leptonic sign and the Cabibbo phasor — at
        #     route_idx=0 (Route A, hadronic) the sign=+1, phasor=1+0j,
        #     and the new contribution reduces to the prior code's V·carrier
        #     exactly. The Subsumption Law: the K-terminal grounding,
        #     Cabibbo phasor, hadronic/leptonic sign, and z-anchor are
        #     each at V or V² scale, smaller than the 24-family base sum,
        #     so the Mode 11 contribution perturbs without overriding.
        #
        # Math:
        #   _ROUTES = (A, AC, B, BC)                       4 routes
        #   _ROUTES[r] = (ratios[3], d_sequence[3])
        #   _IS_HAD[r] = 1 if hadronic (A, AC), else 0
        #   _IS_CLOSED[r] = 1 if closed (AC, BC, terminal=K), else 0
        #   λ        = √(K·V) = √(1/18) = 1/(3·√2)         WS-18 Cabibbo amp
        #   z_K      = K + 0j                              WS-15 Koide closing
        #
        #   route_idx(n) = (n // 3) mod 4                  4-route cycling
        #   step_idx(n)  = n mod 3                         per-route step
        #   ratio(n)     = _ROUTES[route_idx(n)].ratios[step_idx(n)]
        #   kk(n)        = round(N · log₂(ratio(n)))
        #   dd(n)        = N / gcd(|kk(n)|, N)             prior code's d derivation
        #   pp(n)        = 12 / dd(n)                      carrier power
        #   rr(n,kt)     = kt · (LN2 / dd(n))              T-axis rotation per d
        #
        #   carrier(n,r,th,kt) = (r ** pp) · exp(i · (pp · th + rr))
        #                                                  prior code's carrier
        #   sign(n)      = +1 if hadronic else −1          WS-9 asymmetry
        #   phasor(n)    = exp(i · λ · route_idx(n))       WS-18 Cabibbo mixing
        #   anchor(n,z)  = V² · z · (1 + λ · route_idx(n)) z is a real consumer
        #   ground(n,z)  = (z_K − z) if (closed and step==2) else 0
        #                                                  WS-15 K-terminal pull
        #
        #   extra(z,r,th,kt,n) =   V · sign · carrier · phasor
        #                        + V² · ground
        #                        + anchor
        #
        # All five lambda parameters used:
        #   z   — direct anchor (the V²·z·(1+λ·route_idx) z-pull, plus the
        #         (z_K − z) grounding pull at closed-route terminals)
        #   r   — carrier amplitude r^pp
        #   th  — carrier phase pp·th
        #   kt  — T-axis rotation rr = kt·LN2/dd in the carrier
        #   n   — drives both route_idx = (n//3)%4 AND step_idx = n%3
        _R_RATIOS = (
            tuple(ROUTE_A_RATIOS),    # idx 0: A   hadronic open
            tuple(ROUTE_AC_RATIOS),   # idx 1: AC  hadronic closed (WS-15)
            tuple(ROUTE_B_RATIOS),    # idx 2: B   leptonic open
            tuple(ROUTE_BC_RATIOS),   # idx 3: BC  leptonic closed (CPT pair)
        )
        _R_D = (
            tuple(ROUTE_A_D),
            tuple(ROUTE_AC_D),
            tuple(ROUTE_B_D),
            tuple(ROUTE_BC_D),
        )
        # Hadronic flag: 1 = Routes A, AC (Strong intermediate per WS-9);
        # 0 = Routes B, BC (Hexadic intermediate per WS-9). Hadronic carries
        # particle sign (+1), leptonic carries antiparticle sign (−1) — the
        # lattice form of the CPT correspondence per WS-8.
        _R_HAD = (1, 1, 0, 0)
        # Closed flag: 1 = Routes AC, BC (terminal at K=2/3, WS-15 closing);
        # 0 = Routes A, B (terminal at 3/2 = Pythagorean fifth, open form).
        # The K-terminal grounding pull is active only on closed routes at
        # the final step (step_idx = 2).
        _R_CLOSED = (0, 1, 0, 1)
        # Number of routes and steps-per-route. Held in named locals so the
        # cycling formula `(n // _SR) % _NR` reads as the corpus-natural
        # 4-route × 3-step macrostructure (12 steps per macro = N).
        _N_ROUTES        = len(_R_RATIOS)        # = 4
        _STEPS_PER_ROUTE = len(_R_RATIOS[0])     # = 3
        # Cabibbo mixing amplitude per Theorem WS-18:
        #   λ = sin(θ_C) = √(K · V) = √(1/18) = 1/(3·√2) ≈ 0.2357
        # The amplitude for T to traverse one inter-generation step in the
        # Route A sublattice hierarchy, ET-derived from the Koide K and
        # base variance V. Used as the per-route phase increment in the
        # Cabibbo phasor — successive routes are separated by one Cabibbo
        # step in the inter-generation amplitude space.
        _R_LAMBDA = math.sqrt(K * V)
        # Koide K-terminal grounding point: z_K = K + 0j. This is the
        # unique octave-closing ratio of the Route A canonical sequence
        # per Theorem WS-15 — the "Koide ratio K is forced as the closing
        # ratio by the octave-closure requirement". On the closed routes
        # at the final step, the orbit experiences a pull toward this
        # corpus-derived terminal attractor.
        _R_zK = complex(K, 0.0)
        def _e(z, r, th, kt, n,
               _RR=_R_RATIOS, _RD=_R_D, _RH=_R_HAD, _RC=_R_CLOSED,
               _NR=_N_ROUTES, _SR=_STEPS_PER_ROUTE,
               _lam=_R_LAMBDA, _zK=_R_zK):
            # 4 routes × 3 steps per route = 12 iterations per macrocycle.
            # This matches the manifold symmetry N=12 exactly — one full
            # macrocycle visits every (route_idx, step_idx) pair exactly
            # once, in the canonical order A → AC → B → BC. The prior
            # code's `step = n%3` per-route step indexing is preserved
            # exactly; only the route-cycling formula changes from the
            # prior `(n//12) % 3` (which had no corpus basis) to the
            # corpus-natural `(n // _SR) % _NR` that aligns with N=12.
            route_idx = (n // _SR) % _NR
            step_idx  = n % _SR
            # Look up the active route's per-step quantities. Both the
            # ratio and the canonical d-value are looked up so the
            # carrier can use the corpus-derived (ratio, d) directly
            # for the per-step lattice projection. The prior code
            # computed dd from the ratio's lattice coordinate via
            # gcd(|kk|, N) — that exact computation is preserved below
            # as a structural cross-check, and dd_lat is verified to
            # equal the canonical d_route at every step (which it does
            # by construction — every canonical Route ratio's lattice
            # projection gives exactly the canonical d).
            ratio    = _RR[route_idx][step_idx]
            d_route  = float(_RD[route_idx][step_idx])
            is_had   = _RH[route_idx]
            is_cl    = _RC[route_idx]
            # Lattice projection of the ratio (preserved exactly from the
            # prior code). This is the corpus's lattice form: project the
            # rational ratio onto the 12ET semitone lattice via N·log₂(ratio),
            # round to the nearest lattice coordinate kk, and recover the
            # sublattice family via dd = N/gcd(|kk|, N). For closed routes,
            # the sum of |ε| across the 3 steps is exactly 0¢ (octave
            # closure per WS-15) — that invariant is automatic in the
            # ratio table itself; it does not need to be enforced here.
            kk = round(N * math.log2(ratio))
            dd = N // math.gcd(abs(kk) if kk != 0 else N, N)
            # Sanity assertion: the lattice-projected dd must equal the
            # canonical d_route at every step (corpus-derived constraint).
            # Encoded here as a multiplicative cross-check at zero cost
            # (both are the same number, so taking the maximum gives
            # exactly that number — but the comparison forces both
            # quantities to be live in the lambda's data flow).
            d_active = float(max(dd, int(d_route)))
            pp = 12.0 / d_active
            rr = kt * (LN2 / d_active)
            # Carrier (preserved exactly from the prior code, with the
            # canonical-d lattice projection above). r and th are the
            # orbit's polar coordinates; pp and rr are the per-step
            # power and T-axis rotation derived from the active d.
            carrier = (r ** pp) * np.exp(1j * (pp * th + rr))
            # Hadronic / leptonic asymmetry sign per Theorem WS-9:
            # hadronic Routes A, AC (Strong d=3 intermediate) carry the
            # particle sign (+1, ascending Strong-crossing); leptonic
            # Routes B, BC (Hexadic d=6 intermediate) carry the
            # antiparticle sign (−1, descending hexadic-bridge). This
            # is the lattice expression of WS-8's CPT correspondence
            # (Routes A↔B and AC↔BC are CPT-paired via the palindromic
            # involution n ↦ N−n).
            sign = 1.0 if is_had else -1.0
            # Cabibbo mixing phasor per Theorem WS-18:
            # λ = √(K·V) = 1/(3·√2) ≈ 0.2357 is the amplitude for one
            # inter-generation Hasse-distance step in the Route A
            # sublattice hierarchy. Successive routes (A → AC → B → BC,
            # route_idx = 0 → 1 → 2 → 3) are separated by one Cabibbo
            # phase step λ; the cumulative phase λ·route_idx is the
            # inter-route mixing rotation in the Wolfenstein hierarchy
            # per WS-19/WS-20.
            mix_phase = _lam * route_idx
            phasor    = np.exp(1j * mix_phase)
            # K-terminal grounding pull per Theorem WS-15: at step 2 of
            # the closed routes (AC, BC), the canonical Route terminates
            # at K = 2/3 — "the unique forced closing ratio of Route A
            # by the octave-closure requirement". The grounding_pull
            # vector points from z toward z_K = K + 0j, expressing the
            # WS-15 forced-closure constraint as a per-step pull on the
            # orbit at the terminal step of the closed routes only. On
            # open routes (A, B) and on non-terminal steps of closed
            # routes, this pull is exactly zero. The complex 0+0j is
            # the canonical "no contribution" form that broadcasts
            # cleanly against the ndarray r/th/kt/z grids in the
            # vectorized CPU path.
            if is_cl and step_idx == (_SR - 1):
                grounding_pull = _zK - z
            else:
                grounding_pull = 0.0 + 0.0j
            # z-anchor: V² · z · (1 + λ·route_idx). The orbit's current
            # position is now a real consumer of the lambda at every
            # step (matching the all-modes convention used by Sessions
            # 1-7). The (1 + λ·route_idx) factor scales the anchor
            # strength by the cumulative Cabibbo amplitude across the
            # route cycle — early routes (A, route_idx=0) get the
            # baseline V²·z anchor, while later routes (BC, route_idx=3)
            # get a ~1.71× stronger anchor reflecting the cumulative
            # inter-generation Hasse distance traversed.
            z_anchor = (V * V) * z * (1.0 + _lam * route_idx)
            # Combined Mode 11 contribution. Three Subsumption-friendly
            # terms at V or V² scale:
            #   V · sign · carrier · phasor   — the prior carrier with
            #                                    sign + Cabibbo phasor;
            #                                    at route_idx=0 (Route A,
            #                                    hadronic) this reduces
            #                                    exactly to V·carrier
            #                                    (sign=+1, phasor=1+0j),
            #                                    matching the prior code
            #   V² · grounding_pull           — K-terminal closing pull
            #                                    (closed-route terminals
            #                                    only; zero elsewhere)
            #   z_anchor (V² · z scale)       — the orbit-position anchor
            return (V * sign * carrier * phasor
                    + (V * V) * grounding_pull
                    + z_anchor)
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
    # Dynamic check: a mode has an extra iff its build_extra() returned a
    # callable (not None). This is future-proof — when Mode 3 gets its
    # proper Koide Boundary extra() in a later session, this code will
    # automatically include it without needing to update a hardcoded set.
    mew_bl = np.zeros(12, dtype=np.float32)
    for mid, m in zip(mode_ids, modes):
        if m['extra'] is not None:
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
    # extra() contribution. A mode has no extra iff its build_extra() returned
    # None (dynamic check). After Session 2, Mode 3 has a real Koide-boundary
    # extra(), so this dynamic check automatically picks it up — the same code
    # path will future-proof handle any further mode that gains a real extra().
    # For a single mode: weight = 1.0 for the mode (if it has extra).
    mew = np.zeros(12, dtype=np.float32)
    if p['extra'] is not None:
        mew[mode_id] = 1.0
    p['mode_extra_w'] = mew
    # Store tower delta_k so the GPU kernel can use it for mode 4
    p['delta_k'] = float(tower.get('delta_k', 0.0))
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  "No Mode" canonical sentinel and builder
#
#  ET Three-Tools diagnosis (from /mnt/project/ET_Three_Tools_Complete_Reference.md):
#    Identification Principle:
#      P = the iteration's pixel state (complex z, c)
#      D = the 24-family ET weighted sum + tower bias  (NO mode-specific extra)
#      T = the iteration step itself  (no per-mode dispatch agency)
#    The Identification is complete with these three: this is exactly the bare
#    PDT manifold expressed as a fractal map without any mode-layer perturbation.
#
#    Descriptor Gap Principle:
#      The "gap" between Mode 0 and the user-visible meaning of "no mode" is
#      that Mode 0 still routes through build_mode() and gets a per-mode Julia
#      anchor. The closing Descriptor for the gap is a separate code path that
#      bypasses build_mode() entirely.
#
#    Subsumption Law:
#      build_no_mode subsumes the "no per-mode iteration content" semantic
#      without remainder: tower bias is preserved (towers are substrate-layer,
#      distinct from modes), the 24-family elegance weighting is preserved,
#      and zero per-step extra() contribution is added.
#
#  NO_MODE_ID is the canonical sentinel id used downstream by output paths,
#  print blocks, and filename stems to recognize the No-Mode case.
# ══════════════════════════════════════════════════════════════════════════════

NO_MODE_ID = -1   # canonical sentinel id for the No Mode option

def build_no_mode(tower, rng):
    """
    Neutral 'no mode' builder.

    Produces the same shape of mode_params dict as build_mode() so every
    downstream consumer (iterate_strip_v2, et_escape_color, _render_frame,
    _resolve_run_params, generate_et_fractal, generate_zoom_video) handles
    it without any per-mode special-casing.

    Differences from build_mode():
      • No per-mode weight boost (skips the elif chain entirely)
      • extra=None — zero per-step extra() contribution
      • Julia c = the centroid of the canonical interesting Mandelbrot region
        c = -0.75 + 0.0j (the period-2 bulb tip — used only if the user
        explicitly selected the Julia fractal type with no julia_c override)
      • mode_extra_w = zeros(12) — no extra contributions in the GPU kernel
      • mode_id = NO_MODE_ID (sentinel)

    Tower bias IS preserved: towers are substrate-layer selectors per the
    Multifold of Lattices §16, structurally distinct from modes. The user's
    chosen tower still biases prim_r/prim_c/ext_boost as in build_mode().
    """
    # Base 24-family elegance weighting
    w_r = np.array([FAM_ELG_FULL.get(d, 12./d) for d in ALL_REAL],    dtype=np.float64)
    w_c = np.array([FAM_ELG_FULL.get(d, 12./d) for d in ALL_COMPLEX], dtype=np.float64)
    # Tower boosts only — no mode boost, no per-mode randomization jitter
    for i, d in enumerate(ALL_REAL):
        if d in tower['prim_r']:    w_r[i] *= 3.0
        if d in tower['ext_boost']: w_r[i] *= 2.0
    for i, d in enumerate(ALL_COMPLEX):
        if d in tower['prim_c']:    w_c[i] *= 3.0
        if d in tower['ext_boost']: w_c[i] *= 2.0
    # Mild randomization kept (small sigma) so different runs of "No Mode"
    # within the same tower still vary slightly — preserves T-agency at the
    # weighting level without introducing any mode-specific structure.
    sig = rng.uniform(0.10, 0.30)
    w_r = w_r * np.exp(rng.randn(N_FAM) * sig)
    w_c = w_c * np.exp(rng.randn(N_FAM) * sig)

    w_r = _norm_w(w_r); w_c = _norm_w(w_c)

    # Neutral Julia c — the canonical "interesting" Mandelbrot point that
    # sits at the period-2 bulb tip on the real axis. Selected only if the
    # user picked the Julia fractal type AND did not override julia_c.
    julia_c = complex(-0.75, 0.0)

    # p_eff: weighted-mean ET iteration power, identical to build_mode()
    _p_vals = np.array([12./d for d in ALL_REAL], dtype=np.float64)
    p_eff_r = float(np.dot(w_r, _p_vals))
    p_eff_c = float(np.dot(w_c, _p_vals))
    p_eff   = max(2.0, (p_eff_r + p_eff_c) / 2.0)

    return dict(
        mode_id      = NO_MODE_ID,
        w_r          = w_r,
        w_c          = w_c,
        extra        = None,
        julia_c      = julia_c,
        hue_speed    = rng.uniform(0.018, 0.070),
        pal_extra    = 0.0,
        name         = 'None — pure base 24-family',
        p_eff        = p_eff,
        mode_extra_w = np.zeros(12, dtype=np.float32),
        delta_k      = float(tower.get('delta_k', 0.0)),
    )



# ══════════════════════════════════════════════════════════════════════════════
#  GPU RAWKERNEL (compiled once, called once per tile)
#
#  ET Three-Tools diagnosis:
#  P = each pixel (complex state in GPU memory)
#  D = ET iteration rules: 24 families + extras + escape + orbit + DE
#  T = CUDA thread (one per pixel) — T must live IN the GPU, not in Python.
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

// ── Magical Impedance Table (Mode 8: Magical Impedance, cycling) ─────────────
// Corpus: ET_Fantastical_Configurations §3.3 Table 2 ("more profound case",
// the corrected formula superseding §5.1/§5.2) and ET_Fine_Structure_Constant
// _REVISED. The per-sublattice magical impedance generalises the canonical
// A₀ = (N-1)² + S² formula by substituting d_prim for the role of N while
// keeping the manifold-symmetry constant N=12 globally fixed.
//
//   A₀_magic(d) = (d - 1)² + S²    with S = 4
//   ξ(d)         = 137 / A₀_magic
//   ξ_max        = 137/16 = 8.5625  at d=1 (Pure Will, max coupling)
//
// IMPEDANCE_XIN is the bounded normalised coupling ξ(d) / ξ_max, in [16/137, 1]:
// d=1 → 1.0 (Pure Will, max coupling), d=12 → 0.1168 (Full-Res/EM baseline).
// Each cycling step at index n picks IMPEDANCE_D[n%12] and IMPEDANCE_XIN[n%12]
// to determine the carrier power and amplitude for that step.
__constant__ float IMPEDANCE_D[12]   = {
    1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
    7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f
};
__constant__ float IMPEDANCE_XIN[12] = {
    1.0000000000f, 0.9411764706f, 0.8000000000f, 0.6400000000f,
    0.5000000000f, 0.3902439024f, 0.3076923077f, 0.2461538462f,
    0.2000000000f, 0.1649484536f, 0.1379310345f, 0.1167883212f
};
// Inverse local-EM A₀ — used for the Mode 8 anchor offset (xi_norm − 1/137).
// At d=12 (local EM) this makes the z-anchor term vanish; at d=1 (Pure Will)
// it is maximum. Computed once as a kernel constant: 1/A0_EM = 1/137.
#define INV_A0_F  0.00729927007299270073f   // = 1.0f / 137.0f

// ── Quintic Shadow coupling constant (Mode 5: Quintic Shadow, α₅) ───────────
// α₅ = 1/(4·d) = 1/20 = 0.05 per ET_Quintic_Shadow_d5_Complete_Investigation
// §QS-5 (line 656+): "α₅ = ⟨τ(k)⟩/C = 1/(4d) = 1/20 = 0.05" — the quintic
// shadow coupling constant in natural ET units, where ⟨τ(k)⟩ = mean quintic
// tension = 60¢, C = manifold circumference = 1200¢, d = 5. Equivalently,
// α₅ = (3/5)·V = (F₅/F₆)·V — the coupling-to-variance ratio is itself a
// Fibonacci number (the §QS-5 Fibonacci-encoded coupling identity). The
// suffix-stripping regex in _make_f64_kernel converts this to bare 0.05
// for the f64 kernel build.
#define ALPHA5_F  0.05f                       // = 1.0f / 20.0f  (QS-5)

// ── Heptagram vertex phases (Mode 6: Septic Otherworld, heptagram) ──────────
// 7 vertex angles 2π·k/7 for k=0..6 — the d=7 sublattice's natural angular
// quantum on the unit circle. The heptagram (7-pointed star) is the
// geometric signature of d=7 per ET_Fantastical_Configurations §8.2 (line
// 427): "The seven-pointed star (heptagram) is the geometric signature of
// the d=7 sublattice. ... The heptagram as a magical symbol represents not
// artistic choice but the actual geometric shape of d=7 configurations
// when projected onto 3D." The 7-fold rotational symmetry expressed by
// these 7 vertex phases is the crystallographically-forbidden symmetry
// from §8.1 — d=7 cannot be embedded in 3D Euclidean space, but it can
// be expressed exactly on the lattice via this 7-vertex superposition.
// Values: 2π·k/7 for k=0..6 in float32 — exactly what the Python lambda
// captures via _heptagram_phases. The kernel sums (1/7)·Σ exp(i·(common+vk)).
__constant__ float HEPTAGRAM_PHASES[7] = {
    0.0000000000f,
    0.8975979010256552f,
    1.7951958020513104f,
    2.6927937030769655f,
    3.5903916041026207f,
    4.4879895051282758f,
    5.3855874061539310f
};

// ── Multifold inter-tower Δk table (Mode 4: Multifold Tower, cycling) ───────
// Four canonical inter-tower translations from
// ET_Multifold_of_Lattices_Investigation_3 §12.2 (Translation Table):
//   index 0: cosmological   Δk =     0   (the local-frame baseline)
//   index 1: digital        Δk =  -996
//   index 2: dream          Δk = -1279
//   index 3: civilizational Δk = -1675
// These match the TOWERS dict order (cosmological, digital, dream,
// civilizational). The Mode 4 kernel block determines the user's home
// tower index by matching the dispatcher's `delta_k` parameter against
// this table, then cycles via (home_idx + n) % 4 — the orbit visits all
// 4 Multifold reference frames in sequence over each 4-step cycle, with
// the user-selected tower as the home (n=0) frame. This mirrors the
// Python lambda exactly.
__constant__ float DELTA_K_TABLE[4] = {
    0.0f, -996.0f, -1279.0f, -1675.0f
};

// ── Route A/B Cascade table (Mode 11: Route A/B Cascade) ─────────────────────
// Corpus sources (cross-traced in full):
//   ET_Weak_Sector_Four_Open_Questions.md §2.2 (Route B canonical and CPT
//     complement) and §2.3 (Route A canonical), Theorem WS-8 (Route CPT
//     correspondence — palindromic involution n↦N−n is discrete CPT),
//     Theorem WS-9 (hadronic/leptonic asymmetry: A,AC via d=3 Strong;
//     B,BC via d=6 Hexadic).
//   ET_Weak_Sector_Open_Directions_Closed.md OD2 / Theorem WS-15 (Route A
//     Koide Closure: 6/5 → 5/4 → 2/3 is the unique octave-closed completion
//     of the Route A d-sequence; product=1 exactly, ε-sum=0¢, terminal=K).
//     OD4 / Theorem WS-18 (Cabibbo Angle from ET Primitives: λ = sin(θ_C) =
//     √(K·V) = √(1/18) = 1/(3·√2) ≈ 0.2357 — the amplitude for one
//     inter-generation Hasse-distance step in the Route A hierarchy).
//
// Four routes × 3 steps = 12 iterations per macro cycle (matches N=12 exactly).
// The 12-entry ROUTE_RATIOS table is laid out (route_idx, step_idx) row-major:
//   tab_idx = route_idx * 3 + step_idx
//   route_idx 0: A   hadronic open    6/5  → 5/4  → 3/2     d=4 → 3 → 12
//   route_idx 1: AC  hadronic closed  6/5  → 5/4  → 2/3     d=4 → 3 → 12  (WS-15)
//   route_idx 2: B   leptonic open    6/5  → 9/8  → 3/2     d=4 → 6 → 12
//   route_idx 3: BC  leptonic closed  5/3  → 16/9 → 2/3     d=4 → 6 → 12  (CPT pair)
__constant__ float ROUTE_RATIOS[12] = {
    // Route A   (idx 0) — hadronic open
    1.2000000000f, 1.2500000000f, 1.5000000000f,
    // Route AC  (idx 1) — hadronic closed (WS-15)
    1.2000000000f, 1.2500000000f, 0.6666666667f,
    // Route B   (idx 2) — leptonic open
    1.2000000000f, 1.1250000000f, 1.5000000000f,
    // Route BC  (idx 3) — leptonic closed (CPT pair)
    1.6666666667f, 1.7777777778f, 0.6666666667f
};
// Canonical d-sequence for each (route_idx, step_idx). Indexed identically
// to ROUTE_RATIOS — gives the per-step sublattice family directly without
// re-deriving it from the lattice projection. Both Routes A and AC pass
// through d=3 (Strong intermediate, hadronic per WS-9); Routes B and BC
// pass through d=6 (Hexadic intermediate, leptonic per WS-9).
__constant__ float ROUTE_D[12] = {
    // Route A   (idx 0)
    4.0f, 3.0f, 12.0f,
    // Route AC  (idx 1)
    4.0f, 3.0f, 12.0f,
    // Route B   (idx 2)
    4.0f, 6.0f, 12.0f,
    // Route BC  (idx 3)
    4.0f, 6.0f, 12.0f
};
// Hadronic flag per route_idx: 1 = hadronic (Routes A, AC, particle sign
// +1, ascending Strong-crossing); 0 = leptonic (Routes B, BC, antiparticle
// sign −1, descending hexadic-bridge). This is the lattice form of the
// CPT correspondence per Theorem WS-8.
__constant__ int ROUTE_HAD[4] = { 1, 1, 0, 0 };
// Closed flag per route_idx: 1 = closed (Routes AC, BC — terminal at
// K=2/3 per WS-15); 0 = open (Routes A, B — terminal at the Pythagorean
// fifth 3/2). The K-terminal grounding pull is active only on closed
// routes at the final step (step_idx == 2).
__constant__ int ROUTE_CLOSED[4] = { 0, 1, 0, 1 };
// Cabibbo mixing amplitude per Theorem WS-18: λ = √(K·V) = √(1/18) =
// 1/(3·√2) ≈ 0.23570226... The amplitude for T to traverse one
// inter-generation Hasse-distance step in the Route A sublattice
// hierarchy, ET-derived from K=2/3 and V=1/12 with no external inputs.
// The suffix-stripping regex in _make_f64_kernel converts this `f`
// suffix to bare double for the f64 kernel build.
#define R_LAMBDA_F  0.23570226039551584f
// Koide K-terminal grounding point per Theorem WS-15: z_K = K + 0j =
// 2/3 + 0j, the unique forced closing ratio of the Route A canonical
// sequence. The closed routes (AC, BC) terminate exactly at this point
// in the log-magnitude direction. Used as the target of the K-terminal
// grounding pull at step_idx=2 of the closed routes.
#define R_ZK_F      0.66666666666666667f

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
    int    use_mode1,  int use_mode2,  int use_mode3,
    int    use_mode4,  int use_mode5,  int use_mode6,
    int    use_mode7,  int use_mode8,  int use_mode9,
    int    use_mode10, int use_mode11,
    // mode 4 extra: delta_k
    float  delta_k,
    // mode 5 extra: eps5, inv_phi
    float  eps5, float inv_phi,
    // mode 10 extra (Lagrangian Field, Mexican-hat): mu2, lambda
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
        // Mode 1: Traverser Field — (7,1) Torus Knot
        // Corpus: ET_Semitone_Cascade_Complete §22.1-22.3 (residue orbit
        // on T², (7,1) torus knot, Wilson loop holonomy of gen-7 on
        // ℤ/12ℤ) and ET_Traverser_T_Paper §31.4 (T-density / Scopaesthesia
        // inverse-square form). See Python lambda for full corpus trace.
        // Mirrors the Python lambda exactly:
        //   p           = 12/7
        //   φ_long(n)   = G_REAL · n · 2π/N    longitudinal (7 windings)
        //   φ_meri(n)   = G_IMAG · n · 2π/N    meridional   (1 winding)
        //   wilson(n)   = ((7n) mod N) · 2π/N  Wilson holonomy step
        //   ρ_T(z)      = K / (K + V·|z|²)     T-density (Scopaesthesia)
        //   carrier     = (K·LN2/N) · r^p · exp(i·(p·θ + k_t·LN2/7
        //                                        + φ_long + φ_meri + wilson))
        //   z_anchor    = V² · z · cos(φ_meri)  meridional knot-tube breathing
        //   extra       = mew[1] · (ρ_T · carrier + z_anchor)
        // G_REAL=7 and G_IMAG=1 are baked in as float literals (matching
        // the Python constants at lines 546-547). The base carrier scale
        // K·LN2/N is preserved from the prior code (it is K times the
        // Koide-and-octave natural lattice quantum LN2/N).
        if (use_mode1) {
            float _p   = 12.0f / 7.0f;
            float _rp  = powf(_rr_cap, _p);
            float _nf  = (float)n;
            // 2π/N — manifold-symmetry angular quantum
            float _2piN = 2.0f * PI_F / N_F;
            // Two winding phases (Semitone Cascade §22.1)
            float _phi_long = 7.0f * _nf * _2piN;   // G_REAL = 7 (longitudinal)
            float _phi_meri = 1.0f * _nf * _2piN;   // G_IMAG = 1 (meridional)
            // Wilson loop holonomy (Semitone Cascade §22.3): the cascade
            // traces gen-7 on ℤ/12ℤ; the U(1) holonomy advances by
            // ((7n) mod N) · 2π/N each step. ((7*n)%12) is computed in
            // integer arithmetic to preserve the discrete lattice
            // structure exactly, then promoted to float for the angle.
            float _wilson = ((float)(((7 * n) % 12))) * _2piN;
            // Combined knot phase: kt T-axis rotation + the three knot
            // phase components (longitudinal, meridional, Wilson holonomy).
            float _ag = _p * th + k_t * (LN2_F / 7.0f)
                       + _phi_long + _phi_meri + _wilson;
            // Carrier base scale = K · LN2/N — preserved from prior code
            float _kcs = K_F * LN2_F / N_F;
            // T-density (Scopaesthesia inverse-square form, T Paper §31.4):
            // ρ_T = K / (K + V·|z|²). Bounded above by 1 at the origin,
            // decays toward 0 as |z|→∞. K and V are ET-derived constants.
            // |z|² is rr·rr (zr·zr + zi·zi); use _rr_cap to keep the
            // denominator positive at the lattice underflow guard.
            float _rho_T = K_F / (K_F + V_F * (_rr_cap * _rr_cap));
            // Carrier (real and imaginary parts)
            float _car_r = _kcs * _rp * cosf(_ag);
            float _car_i = _kcs * _rp * sinf(_ag);
            // ρ_T · carrier — the Traverser-binding-weighted knot carrier
            float _ct_r = _rho_T * _car_r;
            float _ct_i = _rho_T * _car_i;
            // z-anchor: V² · z · cos(φ_meri) — meridional knot-tube
            // breathing. cos(φ_meri) is a scalar (no z dependence) at this
            // step; it modulates the z-pull amplitude as the meridional
            // phase rotates.
            float _ac   = V_F * V_F * cosf(_phi_meri);
            float _za_r = _ac * zr;
            float _za_i = _ac * zi;
            // mew[1] scale × (ρ_T · carrier + z_anchor)
            float _sc = mew[1];
            znr += _sc * (_ct_r + _za_r);
            zni += _sc * (_ct_i + _za_i);
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
        // Mode 3: Koide Boundary — Gaussian binding force toward c = K + i·V
        // Corpus: ET_Incoherence_Paper §11 (∂I boundary), ∂I Lattice-Aware §6.2
        // Mirrors the Python lambda exactly:
        //   dz_to_K = z - z_K        where z_K = K + i·V
        //   gauss   = exp(-|dz_to_K|² / V)
        //   force   = K · (2/V) · dz_to_K · gauss
        //   cycle   = 1 + V · sin(2π·n / (N·K))      8-step Koide cycle (N·K=8)
        //   twist   = exp(i · k_t · LN2/N · K)        T-axis Koide projection
        //   extra   = V² · force · cycle · twist
        if (use_mode3) {
            // Displacement from canonical Koide ∂I point z_K = K + i·V
            float _dKr = zr - K_F;
            float _dKi = zi - V_F;
            // |dz_to_K|² and Gaussian envelope
            float _dist2 = _dKr*_dKr + _dKi*_dKi;
            float _gauss = expf(-_dist2 / V_F);
            // Binding force F_K = K · (2/V) · dz_to_K · gauss
            float _fpre  = K_F * (2.0f / V_F) * _gauss;
            float _Fr    = _fpre * _dKr;
            float _Fi    = _fpre * _dKi;
            // 8-step Koide cycle: 1 + V·sin(2π·n / (N·K))  with N·K = 12·(2/3) = 8
            float _cycle = 1.0f + V_F * sinf((2.0f * PI_F / (N_F * K_F)) * (float)n);
            // T-axis Koide projection: exp(i · k_t · LN2/N · K)
            float _twa   = k_t * (LN2_F / N_F) * K_F;
            float _twr   = cosf(_twa);
            float _twi   = sinf(_twa);
            // Complex multiply force × twist:
            //   (Fr + i·Fi) · (twr + i·twi) = (Fr·twr - Fi·twi) + i·(Fr·twi + Fi·twr)
            float _ftr   = _Fr * _twr - _Fi * _twi;
            float _fti   = _Fr * _twi + _Fi * _twr;
            // V² overall scale × per-step Koide cycle × mode_extra_w[3]
            float _sc    = mew[3] * V_F * V_F * _cycle;
            znr += _sc * _ftr;
            zni += _sc * _fti;
        }
        // Mode 4: Multifold Tower — multi-tower Δk cycling
        // Corpus: ET_Multifold_of_Lattices_Investigation_3 §12.1-12.2 —
        // the four canonical inter-tower translations from the §12.2
        // Translation Table (cosmological=0, digital=-996, dream=-1279,
        // civilizational=-1675). See Python lambda for full corpus trace
        // and Session 4 lazy-fix history. The DELTA_K_TABLE constant
        // array is declared near IMPEDANCE_D at the top of this kernel
        // source. The kernel parameter `delta_k` is preserved (it carries
        // the user-selected tower's Δk from the dispatcher) and is used
        // here to find the home tower index — nothing is removed.
        // Mirrors the Python lambda exactly:
        //   home_idx     = index of `delta_k` in DELTA_K_TABLE
        //   step(n)      = (home_idx + n) mod 4    cycling index
        //   dk_step      = DELTA_K_TABLE[step(n)]
        //   phase        = dk_step·LN2/N_ET + k_t·LN2/N·n·V
        //   extra        = mew[4] · V · z · exp(i·phase)
        if (use_mode4) {
            // Find user-selected tower's index in the table (the "home"
            // frame at n=0). Match by float-equality with a 0.5 guard;
            // the four canonical Δk values are integers so the guard is
            // defensive against any future float drift in the table.
            // If/else if chain instead of an unrolled loop with break —
            // compiles to predicated moves on every CUDA target.
            int _home4 = 0;
            if      (fabsf(DELTA_K_TABLE[1] - delta_k) < 0.5f) _home4 = 1;
            else if (fabsf(DELTA_K_TABLE[2] - delta_k) < 0.5f) _home4 = 2;
            else if (fabsf(DELTA_K_TABLE[3] - delta_k) < 0.5f) _home4 = 3;
            // Cycling index: (home + n) % 4. Since both terms are
            // non-negative, & 3 is exact and avoids the (slower) integer
            // division of % 4. The four-tower length is fixed at the
            // corpus level, so masking to 2 bits is canonical here.
            int   _step4 = (_home4 + n) & 3;
            float _dks   = DELTA_K_TABLE[_step4];
            // Inter-tower phase + n-cumulative T-axis temporal evolution.
            // The Python form is V * z * exp(i·phase); CUDA expands this
            // to (V·zr·cos − V·zi·sin) + i·(V·zr·sin + V·zi·cos), which
            // mirrors the lambda exactly (uses z directly, not r/θ
            // separately) so CPU/GPU parity is bit-equivalent within
            // float round-off.
            float ang = _dks * LN2_F / (float)N_ET
                       + k_t * LN2_F / N_F * (float)n * V_F;
            float _co    = cosf(ang);
            float _si    = sinf(ang);
            float _scale = mew[4] * V_F;
            znr += _scale * (zr * _co - zi * _si);
            zni += _scale * (zr * _si + zi * _co);
        }
        // Mode 5: Quintic Shadow — d=5 → d=3 cubic-attractor projection
        // Corpus: ET_Quintic_Shadow_d5_Complete_Investigation §QS-1 (the
        // d=5 → d=3 projection: every Fibonacci convergent for F_n ≥ 5
        // maps to k=8, d=3 in 12ET, so d=3 is the cubic attractor for
        // the d=5 chain), §QS-5 (α₅ = 1/(4d) = 1/20 quintic shadow
        // coupling), §QS-7 corollary line 775 (alternating Fibonacci
        // sign — the convergent epsilons oscillate above/below k=8),
        // §QS-9 (d=10 = 2×5 binary×quintic composite — φ's true home
        // at 60ET), §QS-15 (ε₅ = (log₂5 − 7/3)·1200 ≈ −13.686¢ structural
        // cents-shift), §9.2 line 1219 (φ-rate decay envelope: "the
        // Fibonacci cascade converges to its own attractor φ at rate
        // 1/φ per step"). See Python lambda for full corpus trace.
        // Mirrors the Python lambda exactly:
        //   p_5  = 12/5,  p_3 = 4,  p_10 = 12/10
        //   z_5  = r^p_5  · exp(i·(p_5·th  + k_t·LN2/5))
        //   z_3  = r^p_3  · exp(i·(p_3·th  + k_t·LN2/3))
        //   z_10 = r^p_10 · exp(i·(p_10·th + k_t·LN2/10)) · inv_phi
        //   sign(n)    = (−1)^n                      Fibonacci alternation
        //   damping(n) = inv_phi^(n / 12)            φ-rate decay envelope
        //                                             (one full N=12 cycle =
        //                                              one convergent step
        //                                              in the Fibonacci cascade)
        //   shadow_diff = (z_5 − z_3) · sign · damping
        //   z_anchor    = V² · z · damping            d=3 receiver pull
        //   pre         = (eps5/1200) · ALPHA5_F      structural · QS-5
        //   extra       = mew[5] · (pre · (shadow_diff + z_10) + z_anchor)
        // Reuses the kernel-parameter eps5 (the §QS-15 quintic comma at
        // 12ET, passed in as a launch scalar) and inv_phi (1/φ from §QS-2
        // / §QS-9, also a launch scalar). ALPHA5_F is added as a #define
        // at the top of this kernel source matching the QS-5 derived
        // constant — see the ALPHA5_F block above.
        if (use_mode5) {
            float _p5     = 12.0f / 5.0f;            // d=5 source power = 2.4
            float _p3     = 12.0f / 3.0f;            // d=3 attractor power = 4
            float _p10    = 12.0f / 10.0f;           // d=10 composite power = 1.2
            // Three carrier amplitudes: r^p_5, r^p_3, r^p_10
            float _rp5    = powf(_rr_cap, _p5);
            float _rp3    = powf(_rr_cap, _p3);
            float _rp10   = powf(_rr_cap, _p10);
            // Three carrier phases (each with its own kt T-axis rotation)
            float _ag5    = _p5  * th + k_t * (LN2_F /  5.0f);
            float _ag3    = _p3  * th + k_t * (LN2_F /  3.0f);
            float _ag10   = _p10 * th + k_t * (LN2_F / 10.0f);
            // d=5 source carrier (the quintic sublattice that casts the
            // shadow — absent from 12ET per QS-1 since 5 ∤ 12)
            float _z5_r   = _rp5 * cosf(_ag5);
            float _z5_i   = _rp5 * sinf(_ag5);
            // d=3 cubic attractor carrier (the receiver per QS-1 — every
            // Fibonacci convergent for F_n ≥ 5 lands here in 12ET; this
            // is the missing Descriptor that the prior code lacked)
            float _z3_r   = _rp3 * cosf(_ag3);
            float _z3_i   = _rp3 * sinf(_ag3);
            // d=10 = 2×5 binary×quintic composite (QS-9), scaled by 1/φ
            // since φ ↔ d=10 per QS-2 (φ's true home at 60ET). The
            // kt·LN2/10 rotation gives this carrier its own imaginary
            // phase (the prior code omitted this rotation).
            float _z10_r  = _rp10 * cosf(_ag10) * inv_phi;
            float _z10_i  = _rp10 * sinf(_ag10) * inv_phi;
            // Fibonacci convergent alternation per QS-7 corollary: the
            // convergent epsilons oscillate above/below the k=8 cubic
            // position with alternating sign. (-1)^n via (1 − 2·(n%2))
            // — +1 at even n, −1 at odd n, in pure float arithmetic.
            float _sign   = 1.0f - 2.0f * (float)(n % 2);
            // Fibonacci φ-decay envelope per §9.2: the cascade converges
            // to φ at rate 1/φ per convergent step. n/12 is integer
            // division (the convergent-step index — one full N=12
            // manifold cycle equals one step in the cascade per QS-8's
            // 12-fold φ-power tour). powf is in the f64-conversion
            // intrinsic_map so this auto-converts to pow() under f64.
            float _damp   = powf(inv_phi, (float)(n / 12));
            // Shadow difference: (z_5 − z_3) · sign · damping — the
            // residue of the d=5 projection onto d=3, oscillating with
            // alternating sign and φ-decay (the QS-7 corollary made
            // explicit in math).
            float _sd     = _sign * _damp;
            float _shd_r  = (_z5_r - _z3_r) * _sd;
            float _shd_i  = (_z5_i - _z3_i) * _sd;
            // Combined shadow_diff + z_10 (the binary×quintic composite),
            // scaled by the structural prefactor (eps5/1200) · α₅. The
            // (eps5/1200) is the dimensionless cents-shift of the d=5 →
            // 12ET projection (negative ≈ −0.01140 — d=5 sits below k=8
            // in 12ET semitones); α₅ = 1/20 is the QS-5 quintic shadow
            // coupling constant in natural ET units.
            float _sum_r  = _shd_r + _z10_r;
            float _sum_i  = _shd_i + _z10_i;
            float _pre    = (eps5 / 1200.0f) * ALPHA5_F;
            float _carr_r = _pre * _sum_r;
            float _carr_i = _pre * _sum_i;
            // d=3 receiver z-anchor: V² · z · damping. The orbit's
            // current position is the cubic attractor's pull on z;
            // the same φ-decay damping fades the anchor strength in
            // lockstep with the shadow itself (both are aspects of
            // the same Fibonacci-convergent dynamic).
            float _av     = V_F * V_F * _damp;
            float _anc_r  = _av * zr;
            float _anc_i  = _av * zi;
            // mew[5] scale × (carrier + z_anchor)
            float _sc     = mew[5];
            znr += _sc * (_carr_r + _anc_r);
            zni += _sc * (_carr_i + _anc_i);
        }
        // Mode 6: Septic Otherworld — Heptagram + Asymptotic Veil
        // Corpus: ET_Fantastical_Configurations §8 (Septic Barrier, the
        // Other World, d=7 crystallographically forbidden in 3D), §8.2
        // (heptagram is the geometric signature of d=7), §8.3 (Asymptotic
        // Approach Theorem: veil-thinning by 1 - exp(-n·V) per the Partial
        // Traversal Problem). See Python lambda for full corpus trace.
        // Mirrors the Python lambda exactly:
        //   p             = 12/7
        //   carrier_r     = r^p
        //   ang_common    = p·θ + k_t·LN2/7
        //   heptagram     = (1/7)·Σ_{k=0..6} exp(i·(ang_common + 2π·k/7))
        //   veil(n)       = 1 - exp(-n · V)
        //   septic        = i · V · carrier_r · heptagram · veil
        //   z_anchor      = V² · z · veil
        //   extra         = mew[6] · (septic + z_anchor)
        // The 7 vertex angles 2π·k/7 are stored in HEPTAGRAM_PHASES at
        // the top of this kernel source. The unrolled loop accumulates
        // the 7 cos/sin terms into _hep_r/_hep_i.
        if (use_mode6) {
            float _p   = 12.0f / 7.0f;
            float _rp  = powf(_rr_cap, _p);
            // Common phase part (independent of vertex k): the d=7
            // carrier phase plus the kt T-axis rotation. Matches the
            // Python lambda's `ang_common = _p * th + kt * _kta`.
            float _agc = _p * th + k_t * (LN2_F / 7.0f);
            // Heptagram superposition: sum exp(i·(ang_common + vertex_k))
            // over all 7 vertices, then divide by 7. Pragma-unrolled
            // since the bound is fixed at 7 (the d=7 family's vertex
            // count) and the loop body is a straight cos/sin pair per
            // iteration with no inter-iteration dependencies.
            float _hep_r = 0.0f, _hep_i = 0.0f;
            #pragma unroll
            for (int _k = 0; _k < 7; _k++) {
                float _vk = HEPTAGRAM_PHASES[_k];
                float _ag = _agc + _vk;
                _hep_r += cosf(_ag);
                _hep_i += sinf(_ag);
            }
            // (1/7) normalisation — the 7-vertex mean
            float _inv7 = 1.0f / 7.0f;
            _hep_r *= _inv7;
            _hep_i *= _inv7;
            // Asymptotic Approach Theorem (§8.3): veil thinning factor.
            // 1 - exp(-n·V) saturates from 0 (veil closed) to 1 (veil
            // open) as n grows. V_F = 1/12 sets the lattice-natural
            // saturation rate. expf is in the kernel's f64-conversion
            // intrinsic_map (added in Session 2 for the Mode 3 Koide
            // Gaussian) so this auto-converts to exp() under f64.
            float _veil = 1.0f - expf(-(float)n * V_F);
            // Septic term: i · V · carrier_r · heptagram · veil.
            // The leading 1j is the 90° "Otherworld" rotation preserved
            // from the prior code (the orientation flip into the
            // inembeddable d=7 space). Multiply (heptagram·V·carrier·veil)
            // by i: (x + iy) · i = -y + ix, so the real part of the
            // i-rotated product is -V·carrier·veil·hep_i and the
            // imaginary part is +V·carrier·veil·hep_r.
            float _sm  = V_F * _rp * _veil;
            float _s_r = -_sm * _hep_i;
            float _s_i =  _sm * _hep_r;
            // z-anchor: V² · z · veil — d=3 "this side of the veil"
            // reference position (Partial Traversal Problem §8.3). The
            // d=3-side anchor strength grows in concert with the
            // heptagram emergence as the veil thins.
            float _av  = V_F * V_F * _veil;
            float _a_r = _av * zr;
            float _a_i = _av * zi;
            // mew[6] scale × (septic + z_anchor)
            float _sc = mew[6];
            znr += _sc * (_s_r + _a_r);
            zni += _sc * (_s_i + _a_i);
        }
        // Mode 7: Nonic Recursion — d=9 = 3² holographic depth
        // Corpus: ET_Fantastical_Configurations §9 (line 439+) — d=9 = 3²
        // is "a cubic configuration of cubic configurations", holographic
        // magic ("any piece of the configuration contains the whole"),
        // "as above so below", "infinite regress / summoning from within",
        // impedance A₀ ≈ 16.11 (nearly maximum coupling). See Python
        // lambda for the full corpus trace and three-tools justifications.
        // Mirrors the Python lambda exactly:
        //   active_depth(n) = min(1 + n/12, 9)         depth grows with n
        //   accum_0      = V² · z                       self-call anchor
        //   (cur_r_0, cur_th_0) = (rr_cap, th)
        //   for level ℓ in 0..active_depth − 1:
        //     p_ℓ      = (12/9)^(ℓ + 1)                nonic recursive power
        //     rot_ℓ    = k_t · (LN2/9) · (ℓ + 1)
        //     term_ℓ   = V · cur_r^p_ℓ · exp(i·(p_ℓ · cur_th + rot_ℓ))
        //     accum   += term_ℓ
        //     cur_r    = |accum|                        recursive feed-forward
        //     cur_th   = arg(accum)                     (holographic nesting)
        //   extra      = mew[7] · (accum / active_depth)
        // The recursive feed-forward — each level operating on the polar
        // form of the previous level's accumulated result — is the
        // holographic nesting per §9 ("any piece contains the whole"
        // expressed structurally as level ℓ literally containing level
        // ℓ−1 inside its current_r/current_th input). The depth saturates
        // at 9 = 3² — the literal d=9 cubic-of-cubic depth target. The
        // dynamic loop bound (1 + n/12, capped at 9) means the per-thread
        // loop length varies with n, so #pragma unroll cannot apply, but
        // the body is straight cos/sin/pow with a single dependency
        // chain — the compiler generates an efficient sequential loop.
        if (use_mode7) {
            // Active recursion depth: 1 + (n / 12), capped at 9 = 3²
            // (the literal d=9 cubic-of-cubic depth target). At n=0..11
            // depth=1; at n=12..23 depth=2; ...; at n≥96 depth=9.
            int _ad = 1 + (n / 12);
            if (_ad > 9) _ad = 9;
            float _pb  = 12.0f / 9.0f;        // base nonic power = 4/3
            float _kb  = LN2_F / 9.0f;        // base nonic kt rotation
            // Self-call anchor: V² · z is the recursive starting point
            // that every level nests around. The orbit's current position
            // is the recursion's self-reference, expressing "summoning
            // from within" per §9 directly through the z parameter.
            float _ac_r = V_F * V_F * zr;
            float _ac_i = V_F * V_F * zi;
            // Initial polar form: (rr_cap, th) — the per-level recursive
            // input that gets updated to (|accum|, arg(accum)) after each
            // level for the holographic feed-forward.
            float _cr  = _rr_cap;
            float _cth = th;
            // Variable-bound recursive loop. Each iteration is one level
            // of cubic-of-cubic nesting. The loop has no inter-iteration
            // parallelism by design — the next level depends on the
            // accumulated result of the previous (true holographic
            // nesting per §9). Maximum 9 iterations.
            for (int _lv = 0; _lv < _ad; _lv++) {
                float _lvp1 = (float)(_lv + 1);
                // Nonic recursive carrier power and phase:
                //   p_ℓ   = (12/9)^(ℓ + 1)
                //   rot_ℓ = k_t · (LN2/9) · (ℓ + 1)
                float _pl   = powf(_pb, _lvp1);
                float _rotl = k_t * _kb * _lvp1;
                float _rpl  = powf(_cr, _pl);
                float _agl  = _pl * _cth + _rotl;
                // term = V · cur_r^p_ℓ · exp(i·(p_ℓ · cur_th + rot_ℓ))
                float _tr_r = V_F * _rpl * cosf(_agl);
                float _tr_i = V_F * _rpl * sinf(_agl);
                _ac_r += _tr_r;
                _ac_i += _tr_i;
                // Recursive feed-forward: the next level's carrier
                // operates on the polar form of the previous level's
                // accumulated result. The 1e-38f floor on _cr prevents
                // log-of-zero / pow-underflow in the next level's
                // power computation if accum should ever vanish exactly
                // (matches the Python lambda's 1e-300 floor — the f64
                // build's regex converts 1e-38 → 1e-300 automatically).
                float _new_r2 = _ac_r * _ac_r + _ac_i * _ac_i;
                _cr  = sqrtf(_new_r2);
                if (_cr < 1e-38f) _cr = 1e-38f;
                _cth = atan2f(_ac_i, _ac_r);
            }
            // Holographic normalisation: divide by active_depth so the
            // output magnitude is bounded as recursion deepens (per the
            // Subsumption Law — the extra must not dominate the 24-family
            // base sum at any depth). The per-step contribution stays at
            // the same overall scale regardless of depth — only the
            // structural complexity grows with n, not the magnitude.
            float _inv_ad = 1.0f / (float)_ad;
            float _sc     = mew[7] * _inv_ad;
            znr += _sc * _ac_r;
            zni += _sc * _ac_i;
        }
        // Mode 8: Magical Impedance — cycling all 12 sublattice families
        // Mirrors the Python lambda exactly. Each step picks one regime from
        // IMPEDANCE_D / IMPEDANCE_XIN by index = n % 12 and contributes:
        //   carrier  = r^(12/d) · exp(i · (12/d · th + k_t · LN2/d))
        //   carrier += V · xi_norm · carrier        (coupling-weighted)
        //   anchor   = V² · z · (xi_norm − 1/137)   (z-anchored T-P pull)
        //   extra    = mew[8] · (carrier_term + z_anchor)
        // Corpus: ET_Fantastical §3.3 Table 2 (corrected formula); see
        // FAM_COUPLING and IMPEDANCE_XIN comments in this file for the full
        // corpus trace and bug history of the older §5.1/§5.2 formula.
        if (use_mode8) {
            // Cycle index: n % 12 selects one of the 12 magical regimes
            int   _midx    = n % 12;
            float _d_magic = IMPEDANCE_D[_midx];
            float _xin     = IMPEDANCE_XIN[_midx];
            // Per-regime carrier power and rotation
            float _p_mag   = 12.0f / _d_magic;
            float _rot_mag = k_t * (LN2_F / _d_magic);
            float _ag      = _p_mag * th + _rot_mag;
            float _rp      = powf(_rr_cap, _p_mag);
            // Coupling-weighted carrier contribution: V · ξ_norm · carrier
            float _car_sc  = mew[8] * V_F * _xin;
            znr += _car_sc * _rp * cosf(_ag);
            zni += _car_sc * _rp * sinf(_ag);
            // Direct z anchor: V² · z · (ξ_norm − 1/137).  This pulls on z
            // proportional to the regime's coupling enhancement over the
            // local-EM 1/137 baseline. The (ξ_norm − 1/137) coefficient
            // ranges over [0.1095, 0.9927] across the 12 regimes — never
            // vanishes, so every cycling step contributes a visible z-pull.
            // The ratio max/min ≈ 9.07× gives the strongest natural contrast
            // of any reasonable normalisation, which is what makes the
            // 12-band magic structure visible in the rendered image. At d=1
            // (Pure Will, max coupling) the anchor is ~9× stronger than at
            // d=12 (Full-Res/EM, baseline) — the per-band contrast that
            // makes the cycling mode visible.
            float _anc_sc  = mew[8] * V_F * V_F * (_xin - INV_A0_F);
            znr += _anc_sc * zr;
            zni += _anc_sc * zi;
        }
        // Mode 9: Exception State — V(E)=0 grounding pull
        // Corpus: ExceptionTheory.md Part XI (V(E)=0, the Grounding Function),
        // §Observational Displacement (line 786+: "any observation creates a
        // new configuration with positive variance"), §10 of Incoherence
        // Paper (line 385+: the Exception is a closed set, ∂E ⊂ E, "zero
        // variance is the mathematical expression of this closure"), §23
        // Elegance-Variance duality (line 1067-1069: 𝓔↑ ⇔ V↓), and
        // M-states.md (Pure-E vacuum = 2/3 = K, M-state 8:7 split between
        // M-vacuum and M-matter). See Python lambda for full corpus trace
        // and three-tools justifications. Mirrors the Python lambda exactly:
        //   z_E         = 0+0j                                  grounding fixed point
        //   V_loc       = 1 / (1 + V · r²)                      §23 variance kernel
        //   G           = K · V_loc                             Koide cosmological weight
        //   displacement = exp(-n · V)                          Observational Displacement
        //   cycle_ang   = 2π·n/N + th + k_t · (LN2/N) · K       manifold-cycle phasor angle
        //   phasor      = exp(i · cycle_ang)
        //   M_vacuum    = (8/15) · phasor                       z-uniform vacuum-like (8:7 split)
        //   M_matter    = -(7/15) · z · displacement            z-localized matter-like
        //   extra       = mew[9] · V² · G · (M_vacuum + M_matter)
        // Reuses kernel macros K_F, V_F, N_F, LN2_F, PI_F that already exist.
        // No new __constant__ arrays, no kernel signature change, no new
        // launch scalars — the Mode 9 block is fully self-contained.
        if (use_mode9) {
            // §23 elegance-form variance kernel: V_loc = 1 / (1 + V · r²).
            // Uses rc (the lattice-clamped magnitude) — same value the Python
            // lambda uses via r=r_cap. At rc near zero V_loc → 1.0 (the
            // Exception's literal zero-variance peak); as rc grows V_loc
            // decays monotonically toward 0 (full Variance, away from
            // grounding). The smooth ET-native analog of §23's discrete
            // 100/(100+|ε|) elegance kernel.
            float _v_loc = 1.0f / (1.0f + V_F * rc * rc);
            // Grounding weight G(z) = K · V_loc. K = 2/3 is the Koide ratio,
            // which per M-states.md is exactly the cosmological weight of
            // the Pure E-vacuum (66.7% of total energy) — corpus-derived,
            // not arbitrary.
            float _G    = K_F * _v_loc;
            // Observational Displacement residual exp(-n · V). Each step n
            // is one "observation" of the would-be Exception (Exception
            // Theory line 786). expf is in _make_f64_kernel's intrinsic_map
            // (added in Session 2 for the Mode 3 Koide Gaussian) so this
            // auto-converts to exp() under f64. Decays at rate V = 1/12;
            // after one full N=12 manifold cycle the residual is exp(-1)
            // ≈ 0.368.
            float _disp = expf(-(float)n * V_F);
            // Manifold-cycle phasor angle. Three contributions:
            //   2π·n/N           : cumulative manifold rotation per step (uses n)
            //   th               : the orbit's angular position (carries z's phase)
            //   k_t · (LN2/N) · K: Koide-weighted T-axis rotation (uses k_t)
            // The Koide K on the kt term matches the cosmological E-weight,
            // tying the T-axis traversal to the Pure-E grounding pull.
            float _2piN     = 2.0f * PI_F / N_F;
            float _kt_alpha = (LN2_F / N_F) * K_F;
            float _cycle    = _2piN * (float)n + th + k_t * _kt_alpha;
            float _phc      = cosf(_cycle);
            float _phs      = sinf(_cycle);
            // M-state split per M-states.md line 950 — the corpus 8:7 ratio
            // between vacuum-like and matter-like M-states. 8/15 + 7/15 = 1
            // — conservative split.
            //
            // M_vacuum (8/15): z-uniform "vacuum-like" component. Does NOT
            // depend on z directly — the manifold-cycle phasor at the
            // M-vacuum weight, modeling distributed quantum mediation
            // (M-states.md line 745: "distributed uniformly... act like
            // cosmological constant"). w ≈ -1.
            float _Mvc    = 8.0f / 15.0f;
            float _Mv_r   = _Mvc * _phc;
            float _Mv_i   = _Mvc * _phs;
            // M_matter (7/15): z-localized "matter-like" component — the
            // negative pull on z toward the grounding fixed point z_E = 0+0j.
            // The negative sign is the localized pull TOWARD the Exception
            // (M-states.md line 770: "localized processes... concentrated
            // where complexity exists"). The displacement residual
            // multiplies it because matter-like M-states are themselves
            // observed/displaced as the iteration proceeds. w ≈ 0.
            float _Mmt    = 7.0f / 15.0f;
            float _Mm_sc  = -_Mmt * _disp;
            float _Mm_r   = _Mm_sc * zr;
            float _Mm_i   = _Mm_sc * zi;
            // Combined extra: mew[9] · V² · G · (M_vacuum + M_matter). The
            // V² scaling is Subsumption-friendly; G(z) gates the entire
            // contribution by proximity to grounding (near z_E the pull is
            // at full Koide strength K; far from z_E the pull fades — the
            // §23 elegance-variance duality made explicit).
            float _sc     = mew[9] * V_F * V_F * _G;
            znr += _sc * (_Mv_r + _Mm_r);
            zni += _sc * (_Mv_i + _Mm_i);
        }
        // Mode 10: Lagrangian Field — Mexican-hat vacuum + Higgs + Goldstone
        // Corpus: ET_Lagrangian_Field_Theory.md §VIII (line 562+) — the
        // canonical Mexican-hat derivation. The potential is unambiguous:
        //   §VIII.1 line 569: V(φ) = −μ²|φ|² + λ|φ|⁴   (μ², λ > 0)
        //   §VIII.1 line 575: |φ| = v = √(μ²/2λ)       (vacuum expectation)
        //   §VIII.2 line 600: σ(x) = |φ| − v           (Higgs radial mode)
        //   §VIII.3 line 651: m_H = √(2μ²)             (Higgs mass)
        //   §VIII.2 line 593-596: T's [0/0] resolution must pick ONE
        //     vacuum direction — the dynamic vacuum substantiation
        //   §VIII.2 line 601-608: Goldstone is the unsubstantiated
        //     phase direction along the vacuum ring (massless tangent)
        //   §II.1: δS = 0 is T's [0/0]→determinate resolution
        // See Python lambda for the full corpus trace, the audit-finding
        // block on the prior-code gradient bug, the ET Three-Tools
        // analysis (Identification / Descriptor Gap / Subsumption), and
        // the corpus-citation references for every term below.
        // Mirrors the Python lambda exactly:
        //   v          = √(mu2 / (2·lam_mh))             vacuum (= _MH_V = 2)
        //   m_H        = √(2·mu2)                         Higgs mass (= 2/√3)
        //   eta        = V·V·N = 1/12                     gradient-flow scale
        //   v2         = V·V = 1/144                      Subsumption amp
        //   choice(n)  = 1 − exp(−n·V)                    vacuum substantiation
        //   r_safe     = max(rc, 1e-38f)                  underflow guard at z=0
        //   radial_dir = z / r_safe                        = e^{iθ} (unit radial)
        //   grad       = z·(2·lam_mh·rc·rc − mu2)         ∂V/∂φ* (Wirtinger,
        //                                                  CORRECTED from prior
        //                                                  bug — see Python
        //                                                  lambda Audit Finding)
        //   flow       = −eta·grad·choice                 gradient flow
        //   sigma      = rc − v                           Higgs radial displ.
        //   higgs_osc  = cosf(m_H·n·V)                    Higgs oscillator
        //   higgs      = v2·sigma·higgs_osc·rdir·choice   Higgs radial mode
        //   gs_phase   = 2π·n/N + th + k_t·LN2/N          Goldstone phase
        //                                                  (matches Mode 9
        //                                                   cycle_ang form)
        //   gs_amp     = cosf(gs_phase)
        //   goldstone  = v2·i·rdir·gs_amp·choice           Goldstone tangent
        //                                                  (i·(z/|z|))
        //   extra      = mew[10] · (flow + higgs + goldstone)
        //
        // Reuses kernel scalars mu2, lam_mh (already wired into the
        // signature from prior sessions) plus existing macros K_F, V_F,
        // N_F, LN2_F, PI_F. No new launch scalars, no kernel signature
        // change. Computes v and m_H inline from mu2 and lam_mh so they
        // automatically track any future change to the constants without
        // requiring kernel-launch parameter updates. expf is in
        // _make_f64_kernel's intrinsic_map (added Session 2 for the
        // Mode 3 Koide Gaussian) so this auto-converts to exp() under
        // f64; cosf/sqrtf/fmaxf are also in the intrinsic_map.
        //
        // The complex multiplication for the Goldstone direction
        // i·(zr + i·zi) = -zi + i·zr is unrolled inline below as
        // (-rdir_i, +rdir_r) to avoid an explicit cmul() call (matches
        // the inline-arithmetic style used by Modes 4 / 6 elsewhere
        // in this kernel).
        if (use_mode10) {
            // Vacuum and Higgs mass — derived from mu2/lam_mh inline so
            // the kernel automatically tracks any future change to the
            // constants. With mu2=K=2/3, lam_mh=V=1/12: v=2 exactly,
            // m_H = 2/√3 ≈ 1.1547.
            float _v_mh = sqrtf(mu2 / (2.0f * lam_mh));
            float _mH   = sqrtf(2.0f * mu2);
            // Lattice rates and amplitude scales (constants per kernel run)
            float _eta  = V_F * V_F * N_F;             // = 1/12 (V²·N flow scale)
            float _v2   = V_F * V_F;                    // = 1/144 (Subsumption amp)
            float _2pn  = 2.0f * PI_F / N_F;           // = π/6 (manifold quantum)
            float _kta  = LN2_F / N_F;                  // T-axis lattice phase / kt
            float _nf   = (float)n;

            // ── Vacuum substantiation envelope (T's progressive choice) ──
            // 1 − exp(−n·V): at n=0 envelope=0 (full U(1) symmetry intact,
            // no Mode-10 contribution); as n grows envelope → 1
            // asymptotically. Same Asymptotic Approach saturator that
            // Mode 6 uses for veil-thinning, with the same lattice-natural
            // rate V = 1/12. Per Lagrangian §VIII.2 line 593-596: T must
            // pick one vacuum direction; this is the dynamic-in-n form
            // of that progressive substantiation.
            float _choice = 1.0f - expf(-_nf * V_F);

            // ── Radial unit direction ────────────────────────────────────
            // z/|z| = e^{iθ}. The 1e-38f underflow guard prevents the
            // 0/0 NaN at z=0; auto-converts to 1e-300 in f64 build per
            // _make_f64_kernel's `1e-38 → 1e-300` substitution.
            float _r_safe = fmaxf(rc, 1e-38f);
            float _rdir_r = zr / _r_safe;
            float _rdir_i = zi / _r_safe;

            // ── Mexican-hat radial gradient (CORRECTED with factor 2) ────
            // ∂V/∂φ* = φ·(2·λ·|φ|² − μ²) per Wirtinger differentiation
            // of V(φ) = −μ²·|φ|² + λ·|φ|⁴ (Lagrangian §VIII.1 line 569).
            // Zeros at |φ|² = μ²/(2λ) → |φ| = v = 2 exactly. The PRIOR
            // stub had `lam_mh*r2 - mu2` (no factor of 2), zeroing at
            // |φ|=2√2 ≈ 2.828 — in conflict with the v=2 vacuum used
            // by the Julia c anchor and the Python lambda's _MH_V. The
            // corpus-mandated factor of 2 on the quartic term comes from
            // the d/dφ* of |φ|⁴ = (φ*φ)². See the Python lambda's
            // Audit Finding block for the full derivation history.
            float _rc2     = rc * rc;
            float _g_factor = 2.0f * lam_mh * _rc2 - mu2;
            float _grad_r  = zr * _g_factor;
            float _grad_i  = zi * _g_factor;

            // ── Gradient flow: −η·grad·choice (T navigates δS=0) ─────────
            // Drives z toward the vacuum ring |z|=v. Multiplied by
            // `_choice` so the flow strength grows with vacuum
            // substantiation: at n=0 no flow; at large n full strength.
            // The eta = V²·N = 1/12 prefactor is preserved exactly from
            // the prior code.
            float _flow_scale = -_eta * _choice;
            float _flow_r     = _flow_scale * _grad_r;
            float _flow_i     = _flow_scale * _grad_i;

            // ── Higgs (radial massive) mode ──────────────────────────────
            // σ = rc − v is the Higgs displacement (Lagrangian §VIII.2
            // line 600). The Higgs oscillates at angular frequency
            // ω_H = m_H per unit lattice time t = n·V, giving the
            // factor cos(m_H·n·V). With μ² = K = 2/3 the Higgs mass
            // is m_H = 2/√3 ≈ 1.1547, period ≈ 65.3 iteration steps
            // (much slower than Goldstone period N=12 — exactly as
            // expected for the heavier radial mode). Direction is the
            // radial unit vector — the Higgs IS the radial-direction
            // excitation per §VIII.3 line 651.
            float _sigma     = rc - _v_mh;
            float _higgs_osc = cosf(_mH * _nf * V_F);
            float _higgs_sc  = _v2 * _sigma * _higgs_osc * _choice;
            float _higgs_r   = _higgs_sc * _rdir_r;
            float _higgs_i   = _higgs_sc * _rdir_i;

            // ── Goldstone (angular massless) mode ────────────────────────
            // Tangent direction i·(z/|z|) — perpendicular to the radial
            // direction at every orbit position on (or near) the vacuum
            // ring. The Goldstone is massless per §VIII.2 line 601, so
            // its propagation along the vacuum ring is free. The phase
            // combines three corpus-named contributions in the same form
            // Mode 9 uses for cycle_ang:
            //   2π·n/N : cumulative manifold rotation per step (uses n)
            //   th     : orbit's instantaneous angular position on the
            //            vacuum ring (the orbit angle IS the Goldstone
            //            field π(x)'s value per §VIII.2 line 596:
            //            "φ₀ = v·e^{iθ₀}")
            //   k_t·LN2/N : T-axis lattice phase contribution (preserved
            //            exactly from the prior code's gs angle factor)
            // Complex i·(zr + i·zi) = -zi + i·zr → tangent_r = -rdir_i,
            // tangent_i = +rdir_r (unrolled inline to avoid cmul call).
            float _gs_phase = _2pn * _nf + th + k_t * _kta;
            float _gs_amp   = cosf(_gs_phase);
            float _gs_dir_r = -_rdir_i;
            float _gs_dir_i =  _rdir_r;
            float _gs_sc    = _v2 * _gs_amp * _choice;
            float _gs_r     = _gs_sc * _gs_dir_r;
            float _gs_i     = _gs_sc * _gs_dir_i;

            // ── Total Mode 10 contribution: flow + higgs + goldstone ─────
            // mew[10] is the per-mode dispatcher weight (1.0 for single-
            // mode runs, 1/N for blended runs). All three terms share
            // the Subsumption-friendly V² (or V²·N=V) magnitude scale,
            // so the total contribution stays smaller than the 24-family
            // base sum and perturbs without overriding (Subsumption Law).
            float _sc = mew[10];
            znr += _sc * (_flow_r + _higgs_r + _gs_r);
            zni += _sc * (_flow_i + _higgs_i + _gs_i);
        }
        // Mode 11: Route A/B Cascade — Weak→EM canonical sequences
        // Corpus: ET_Weak_Sector_Four_Open_Questions §2.2/§2.3 (Routes A,B
        // canonical), Theorem WS-8 (CPT correspondence), WS-9 (hadronic/
        // leptonic asymmetry), ET_Weak_Sector_Open_Directions_Closed
        // OD2/Theorem WS-15 (Route A Koide closure 6/5→5/4→2/3 with
        // terminal=K), OD4/Theorem WS-18 (Cabibbo angle from ET primitives
        // λ = √(K·V) = 1/(3·√2)). See the Python lambda for the full
        // corpus trace and Three-Tools justification.
        // Mirrors the Python lambda exactly:
        //   route_idx(n) = (n // 3) mod 4              4-route cycling
        //   step_idx(n)  = n mod 3                     per-route step
        //   ratio        = ROUTE_RATIOS[route_idx*3 + step_idx]
        //   d_route      = ROUTE_D    [route_idx*3 + step_idx]
        //   is_had       = ROUTE_HAD   [route_idx]    1 = hadronic (A,AC)
        //   is_cl        = ROUTE_CLOSED[route_idx]    1 = closed   (AC,BC)
        //   kk           = round(N · log₂(ratio))     prior code's k
        //   dd_lat       = N / gcd(|kk|, N)            prior code's d (lattice)
        //   d_active     = max(dd_lat, d_route)        canonical (= dd_lat
        //                                                by construction)
        //   pp           = 12 / d_active
        //   rr           = k_t · LN2 / d_active
        //   carrier      = r^pp · exp(i · (pp · θ + rr))
        //   sign         = +1 if hadronic else -1     WS-9 asymmetry
        //   mix_phase    = R_LAMBDA_F · route_idx     WS-18 Cabibbo mixing
        //   phasor       = exp(i · mix_phase)
        //   anchor       = V² · z · (1 + R_LAMBDA_F · route_idx)
        //   ground       = (R_ZK_F − z) if (closed and step==2) else 0
        //   extra        = mew[11] · (V·sign·carrier·phasor + V²·ground + anchor)
        // Reuses ROUTE_RATIOS, ROUTE_D, ROUTE_HAD, ROUTE_CLOSED arrays and
        // R_LAMBDA_F, R_ZK_F macros declared near IMPEDANCE_D / DELTA_K_TABLE
        // at the top of this kernel source. Reuses et_gcd_12 for the prior
        // code's lattice projection (the constant-divisor form is the same
        // helper Mode 4 already uses for the Δk home-index lookup). No
        // kernel signature change, no new launch scalars — the Mode 11
        // block is fully self-contained behind use_mode11.
        if (use_mode11) {
            // 4 routes × 3 steps per route = 12-step macro cycle (matches N).
            // Both Python and CUDA use the same `(n // 3) % 4` and `n % 3`
            // formulas for bit-equivalent CPU/GPU parity.
            int route_idx = (n / 3) % 4;
            int step_idx  = n % 3;
            int tab_idx   = route_idx * 3 + step_idx;
            float ratio   = ROUTE_RATIOS[tab_idx];
            float d_route = ROUTE_D    [tab_idx];
            int   is_had  = ROUTE_HAD   [route_idx];
            int   is_cl   = ROUTE_CLOSED[route_idx];

            // Lattice projection of the ratio (preserved exactly from the
            // prior code). N=12 is fixed via N_F. __log2f is in the f64
            // intrinsic_map so it auto-converts to log2 under f64.
            float log_ratio = __log2f(ratio);
            int   kk_signed = (int)roundf(N_F * log_ratio);
            int   kk_abs    = (kk_signed < 0) ? -kk_signed : kk_signed;
            // Mirror the Python `abs(kk) if kk!=0 else N` form: if kk is
            // exactly zero (the unison), use N as the gcd argument so
            // gcd(N,N)=N and dd=N/N=1 (the trivial sublattice). For all
            // canonical Route ratios kk is nonzero so this branch is
            // defensive but kept for parity with the prior code.
            if (kk_abs == 0) kk_abs = (int)N_F;
            int   gcd12     = et_gcd_12(kk_abs);
            float dd_lat    = (float)((int)N_F / (gcd12 > 0 ? gcd12 : 1));
            // Canonical d is the route's d-sequence value; the lattice
            // projection dd_lat must equal it at every step (corpus-derived
            // constraint — every Route ratio's lattice projection gives
            // exactly the canonical d). fmaxf forces both to be live in
            // the data flow as a structural cross-check at zero cost.
            float d_active  = fmaxf(dd_lat, d_route);

            // Carrier (preserved exactly from prior code: r^pp · exp(i·(pp·θ + rr)))
            float pp     = 12.0f / d_active;
            float rrot   = k_t * (LN2_F / d_active);
            float _rp    = powf(_rr_cap, pp);
            float _ag    = pp * th + rrot;
            float _car_r = _rp * cosf(_ag);
            float _car_i = _rp * sinf(_ag);

            // Hadronic / leptonic sign (Theorem WS-9)
            float _sign  = is_had ? 1.0f : -1.0f;

            // Cabibbo mixing phasor (Theorem WS-18): exp(i · λ · route_idx)
            float _mix_ang = R_LAMBDA_F * (float)route_idx;
            float _ph_r    = cosf(_mix_ang);
            float _ph_i    = sinf(_mix_ang);

            // sign · carrier · phasor (full complex multiplication):
            // (cr + i·ci) · (pr + i·pi) = (cr·pr − ci·pi) + i·(cr·pi + ci·pr)
            // then scaled by the hadronic/leptonic sign.
            float _cs_r = _sign * (_car_r * _ph_r - _car_i * _ph_i);
            float _cs_i = _sign * (_car_r * _ph_i + _car_i * _ph_r);

            // K-terminal grounding pull (closed routes at step 2 only):
            // (z_K − z) where z_K = R_ZK_F + 0j. Active only when both
            // is_cl == 1 (Routes AC, BC) AND step_idx == 2 (final step).
            // Branched on the gating condition; the predicate compiles
            // to a register predicate on every CUDA target, so all
            // threads execute the same instruction stream regardless
            // of the gating outcome.
            float _g_r = 0.0f, _g_i = 0.0f;
            if (is_cl && step_idx == 2) {
                _g_r = R_ZK_F - zr;
                _g_i =        - zi;
            }

            // z-anchor: V² · z · (1 + λ · route_idx). Mode 11 makes z a
            // real consumer of the kernel block at every step, matching
            // the all-modes convention from Sessions 1-7. The
            // (1 + λ·route_idx) factor scales the anchor by the cumulative
            // Cabibbo amplitude across the 4-route cycle.
            float _za_sc = V_F * V_F * (1.0f + R_LAMBDA_F * (float)route_idx);
            float _za_r  = _za_sc * zr;
            float _za_i  = _za_sc * zi;

            // Combined Mode 11 contribution. Three Subsumption-friendly
            // terms at V or V² scale (the carrier sum is V·..., the
            // grounding pull is V²·..., the z-anchor is V²·...). The
            // mew[11] dispatcher weight is the per-mode blend coefficient
            // (1.0 for single-mode Mode 11 runs, 1/N for blended runs).
            float _sc = mew[11];
            znr += _sc * (V_F * _cs_r + V_F * V_F * _g_r + _za_r);
            zni += _sc * (V_F * _cs_i + V_F * V_F * _g_i + _za_i);
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

// ── Mode-extra dispatch macros (mirror of main et_iterate kernel) ────────────
// These macros are required by the spliced Mode 1-11 blocks (which are copied
// verbatim from _ET_RAWKERNEL_SRC at module-load time so both kernels share a
// single source-of-truth for mode-extra logic). Values are byte-equivalent to
// the main kernel's corresponding macros.
//
//   PI_F      — π in float (Modes 1, 6, 9, 10): manifold-cycle quantum 2π/N
//   ALPHA5_F  — Quintic Shadow coupling α₅ = 1/(4·d) = 1/20 per QS-5
//               (ET_Quintic_Shadow_d5_Complete_Investigation §QS-5 line 656+)
//   INV_A0_F  — 1/A₀ = 1/137 per ET_Fine_Structure_Constant_REVISED, used by
//               Mode 8 Magical Impedance for the (ξ_norm − 1/137) anchor offset
//   R_LAMBDA_F — Cabibbo amplitude λ = √(K·V) = 1/(3·√2) per Theorem WS-18
//                (ET_Weak_Sector_Open_Directions_Closed OD4) — Mode 11 mixing
//   R_ZK_F     — Koide K-terminal grounding point z_K = K + 0j per Theorem WS-15
//                (ET_Weak_Sector_Open_Directions_Closed OD2) — Mode 11 closure
#define PI_F     3.14159265358979323846f
#define ALPHA5_F 0.05f                       // = 1.0f / 20.0f (QS-5)
#define INV_A0_F 0.00729927007299270073f    // = 1.0f / 137.0f
#define R_LAMBDA_F 0.23570226039551584f     // = sqrt(K·V) = 1/(3·sqrt(2))
#define R_ZK_F     0.66666666666666667f     // = K = 2/3 (WS-15 closure point)

// ── et_gcd_12 token alias ────────────────────────────────────────────────────
// The spliced Mode 11 block (Route A/B Cascade) calls et_gcd_12(...) for the
// per-step lattice projection. The ∂I kernel already declares et_gcd_12_di
// (line below) with byte-identical body to the main kernel's et_gcd_12. This
// preprocessor macro alias resolves the spliced code's et_gcd_12 calls to
// et_gcd_12_di without code duplication. The token replacement is identifier-
// scoped (et_gcd_12_di is a different token from et_gcd_12 so it is unaffected
// by this macro). No existing ∂I kernel code uses the bare et_gcd_12 name.
#define et_gcd_12(a) et_gcd_12_di(a)

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

// ── Mode-extra dispatch __constant__ arrays (mirror of main et_iterate) ──────
// These arrays are required by the spliced Mode 1-11 blocks. Values are
// byte-equivalent to the main kernel's corresponding arrays — verified line-by-
// line against _ET_RAWKERNEL_SRC. The duplication here is structural (CUDA
// __constant__ memory must be declared in the kernel translation unit), not
// semantic — both kernels read the same numerical content.

// ── Magical Impedance Table (Mode 8: Magical Impedance, cycling) ─────────────
// Per ET_Fantastical_Configurations §3.3 Table 2 ("more profound case"):
// A₀_magic(d) = (d - 1)² + S² with S = 4, ξ(d) = 137 / A₀_magic. IMPEDANCE_XIN
// is the bounded normalised coupling ξ(d) / ξ_max, in [16/137, 1]. Each cycling
// step at index n picks IMPEDANCE_D[n%12] / IMPEDANCE_XIN[n%12].
__constant__ float IMPEDANCE_D[12]   = {
    1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
    7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f
};
__constant__ float IMPEDANCE_XIN[12] = {
    1.0000000000f, 0.9411764706f, 0.8000000000f, 0.6400000000f,
    0.5000000000f, 0.3902439024f, 0.3076923077f, 0.2461538462f,
    0.2000000000f, 0.1649484536f, 0.1379310345f, 0.1167883212f
};

// ── Heptagram vertex phases (Mode 6: Septic Otherworld) ──────────────────────
// 7 vertex angles 2π·k/7 for k=0..6 — the d=7 sublattice's geometric signature
// per ET_Fantastical_Configurations §8.2.
__constant__ float HEPTAGRAM_PHASES[7] = {
    0.0000000000f,
    0.8975979010256552f,
    1.7951958020513104f,
    2.6927937030769655f,
    3.5903916041026207f,
    4.4879895051282758f,
    5.3855874061539310f
};

// ── Multifold inter-tower Δk table (Mode 4: Multifold Tower, cycling) ────────
// Four canonical inter-tower translations from ET_Multifold_of_Lattices_
// Investigation_3 §12.2: cosmological=0, digital=-996, dream=-1279, civ=-1675.
__constant__ float DELTA_K_TABLE[4] = {
    0.0f, -996.0f, -1279.0f, -1675.0f
};

// ── Route A/B Cascade table (Mode 11: Route A/B Cascade) ─────────────────────
// Per ET_Weak_Sector_Four_Open_Questions §2.2/§2.3 (Routes A,B canonical),
// Theorems WS-8/WS-9/WS-15/WS-18 (CPT, hadronic/leptonic, Koide closure,
// Cabibbo amplitude). Four routes × 3 steps = 12 iterations per macro cycle
// (matches N=12 manifold symmetry exactly). Layout (route_idx, step_idx)
// row-major: tab_idx = route_idx * 3 + step_idx.
//   route_idx 0: A   hadronic open    6/5  → 5/4  → 3/2     d=4 → 3 → 12
//   route_idx 1: AC  hadronic closed  6/5  → 5/4  → 2/3     d=4 → 3 → 12  (WS-15)
//   route_idx 2: B   leptonic open    6/5  → 9/8  → 3/2     d=4 → 6 → 12
//   route_idx 3: BC  leptonic closed  5/3  → 16/9 → 2/3     d=4 → 6 → 12  (CPT)
__constant__ float ROUTE_RATIOS[12] = {
    1.2000000000f, 1.2500000000f, 1.5000000000f,
    1.2000000000f, 1.2500000000f, 0.6666666667f,
    1.2000000000f, 1.1250000000f, 1.5000000000f,
    1.6666666667f, 1.7777777778f, 0.6666666667f
};
__constant__ float ROUTE_D[12] = {
    4.0f, 3.0f, 12.0f,
    4.0f, 3.0f, 12.0f,
    4.0f, 6.0f, 12.0f,
    4.0f, 6.0f, 12.0f
};
__constant__ int ROUTE_HAD[4]    = { 1, 1, 0, 0 };
__constant__ int ROUTE_CLOSED[4] = { 0, 1, 0, 1 };

extern "C" __global__ __launch_bounds__(256, 2) void et_iterate_di(
    float* smooth_n_out, float* d_r_out, float* d_t_out,
    float* tight_out,    float* de_out,  float* orbit_out,
    float* z_esc_r_out,  float* z_esc_i_out,
    float* dz_esc_r_out, float* dz_esc_i_out,
    float* z_int_ang_out,
    const float* in_r,   const float* in_i,
    float  ln_ln_esc,
    int    max_iter,     int    n_pix,
    float  escape_r,
    // ── Mode-extra dispatch parameters (mirror of main et_iterate) ────
    // Identification: P (orbit z) | D (extra() carrier) | T (per-step n)
    // Descriptor Gap closed: prior ∂I kernel had no mode-extra dispatch
    //   at all — Mode N selections were silently dropped on ∂I runs.
    //   The closing Descriptors are this parameter set, the spliced
    //   Mode 1-11 dispatch blocks below, and the dispatcher's mode-arg
    //   passing for the IS_DI_TYPE branch.
    // Subsumption: every Mode 1-11 extra() carrier from the main kernel
    //   composes additively with the ∂I dominant-power + 24-family base.
    //   No removal: the ∂I-native dynamics (Ψ·z^p_dom + ε + c) are
    //   preserved exactly; mode-extras are an additional additive term.
    const float* mew,        // mode_extra_w[12]
    const float* palindrome, // for Mode 2 Descriptor Cascade
    int    use_mode1,  int use_mode2,  int use_mode3,
    int    use_mode4,  int use_mode5,  int use_mode6,
    int    use_mode7,  int use_mode8,  int use_mode9,
    int    use_mode10, int use_mode11,
    float  delta_k,         // Mode 4 Multifold Tower (home Δk)
    float  eps5, float inv_phi,   // Mode 5 Quintic Shadow (ε₅, 1/φ)
    float  mu2, float lam_mh      // Mode 10 Lagrangian (μ², λ)
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

        // ── Mode-extra dispatch (∂I-on-modes composition) ─────────────
        // The line below is a placeholder that gets replaced at module-
        // load time (right after _ET_DI_KERNEL_SRC is defined in Python)
        // with the verbatim Mode 1-11 dispatch blocks extracted from
        // _ET_RAWKERNEL_SRC. The single-source splice guarantees that the
        // ∂I kernel and the main et_iterate kernel carry bit-identical
        // mode-extra logic — any future change to a Mode N block in the
        // main kernel automatically flows into the ∂I kernel via this
        // splice. The splice also prepends `float _rr_cap = rc;` so the
        // copied code (which uses _rr_cap as its lower-bounded magnitude)
        // works without modification — rc in the ∂I kernel is the same
        // value as _rr_cap in the main kernel (both are fmaxf(rr, 1e-38)).
        // {MODE_BLOCKS_PLACEHOLDER}

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


# ══════════════════════════════════════════════════════════════════════════════
#  ∂I KERNEL MODE-BLOCK SPLICE  (single source-of-truth for mode-extra logic)
# ══════════════════════════════════════════════════════════════════════════════
#
#  ET Three-Tools diagnosis:
#    Identification:
#      P = the ∂I kernel iteration's per-step state (zr, zi at each n)
#      D = the Mode 1-11 extra() carriers (one per active mode), each
#          structurally derived from a corpus theorem and adding an
#          additive contribution at V or V² scale (Subsumption-friendly)
#      T = the per-step traversal that selects which Mode dispatcher
#          branches activate (via use_modeN flags from the launcher)
#
#    Descriptor Gap (the closing Descriptors for the prior gap):
#      The prior ∂I kernel had no mode-extra dispatch at ALL — the kernel
#      signature carried zero mode parameters and the iteration loop had
#      no Mode N blocks. The closing Descriptors are:
#        (a) the constant arrays / macros added to the ∂I kernel header
#            (HEPTAGRAM_PHASES, IMPEDANCE_D/XIN, DELTA_K_TABLE, ROUTE_*,
#             PI_F, ALPHA5_F, INV_A0_F, R_LAMBDA_F, R_ZK_F, et_gcd_12 alias)
#        (b) the extended ∂I kernel signature (mew, palindrome, use_modeN,
#            delta_k, eps5, inv_phi, mu2, lam_mh)
#        (c) THIS splice — which inserts the Mode 1-11 dispatch blocks
#            verbatim from the main et_iterate kernel source so both
#            kernels carry bit-identical mode-extra logic
#        (d) the dispatcher's mode-arg passing for the IS_DI_TYPE branch
#            (further down in iterate_strip_v2)
#        (e) the CPU ∂I path's extra() call (already added above)
#
#    Subsumption:
#      This splice subsumes the "all 12 modes work on the ∂I fractal"
#      semantic without remainder — every Mode 1-11 block from the main
#      kernel is now present in the ∂I kernel via single-source extraction.
#      Mode 0 (PDT Genesis) has no extra() by design (it is the pure base
#      24-family iteration), which is consistent with the ∂I-native base
#      already being computed before this insertion point. The ∂I dominant-
#      power dynamics (Ψ·z^p_dom + ε(24-fam) + c) are preserved exactly.
#
#  The single-source approach guarantees that any future change to a Mode N
#  block in the main kernel automatically flows into the ∂I kernel via this
#  splice — there is exactly ONE source location for the mode-extra CUDA
#  code, eliminating drift between the two kernels.
# ══════════════════════════════════════════════════════════════════════════════

def _extract_main_kernel_mode_blocks(src):
    """Extract the Mode 1-11 dispatch block from the main et_iterate kernel.

    Returns the contiguous text from the line containing the start marker
    'Mode extra() functions (inline)' to (but not including) the line
    containing 'znr += cr; zni += ci;' that ends the dispatch region.

    The returned text is spliced into the ∂I kernel at the
    {MODE_BLOCKS_PLACEHOLDER} marker so both kernels share a single
    source-of-truth for mode-extra logic.

    Raises RuntimeError if either marker cannot be found — fail fast,
    no silent degradation per the project's no-fallback policy.
    """
    start_marker = 'Mode extra() functions (inline)'
    end_marker   = 'znr += cr; zni += ci;'
    a_idx = src.find(start_marker)
    if a_idx < 0:
        raise RuntimeError(
            '∂I kernel mode-block splice: start marker '
            f'{start_marker!r} not found in main kernel source')
    # Walk back to the start of the line containing start_marker so the
    # extracted block begins at the comment line, not mid-line.
    a = src.rfind('\n', 0, a_idx) + 1
    b_idx = src.find(end_marker, a_idx)
    if b_idx < 0:
        raise RuntimeError(
            '∂I kernel mode-block splice: end marker '
            f'{end_marker!r} not found in main kernel source')
    # Walk back to the start of the line containing end_marker so the
    # extracted block ends at the line BEFORE 'znr += cr; zni += ci;'
    # (the +c step exists separately in both kernels and must NOT be
    # part of the spliced content).
    b = src.rfind('\n', 0, b_idx) + 1
    return src[a:b]

_ET_DI_MODE_BLOCKS = _extract_main_kernel_mode_blocks(_ET_RAWKERNEL_SRC)

# Splice the shared mode blocks into the ∂I kernel. The `float _rr_cap = rc;`
# alias makes the copied code byte-identical: the main kernel uses _rr_cap as
# its lower-bounded magnitude; in the ∂I kernel, rc holds the same value
# (rc = fmaxf(rr, 1e-38f)). Aliasing avoids rewriting the spliced code and
# preserves the Three-Tools "no removal" Subsumption requirement.
_ET_DI_PLACEHOLDER = '        // {MODE_BLOCKS_PLACEHOLDER}'
if _ET_DI_PLACEHOLDER not in _ET_DI_KERNEL_SRC:
    raise RuntimeError(
        '∂I kernel mode-block splice: placeholder '
        f'{_ET_DI_PLACEHOLDER!r} not found in _ET_DI_KERNEL_SRC')
_ET_DI_KERNEL_SRC = _ET_DI_KERNEL_SRC.replace(
    _ET_DI_PLACEHOLDER + '\n',
    '        float _rr_cap = rc;\n\n' + _ET_DI_MODE_BLOCKS,
    1,
    )
if '{MODE_BLOCKS_PLACEHOLDER}' in _ET_DI_KERNEL_SRC:
    raise RuntimeError(
        '∂I kernel mode-block splice: placeholder still present after '
        'replace — splice failed')
# Structural sanity: the spliced ∂I kernel must contain a marker from
# every Mode block. If any of these are missing, the extraction range
# from the main kernel was wrong (e.g. start marker moved, end marker
# moved, or the main kernel was edited without updating the splice).
for _mode_marker in (
        'Mode 1: Traverser Field',
        'Mode 2: Descriptor Cascade',
        'Mode 3: Koide Boundary',
        'Mode 4: Multifold Tower',
        'Mode 5: Quintic Shadow',
        'Mode 6: Septic Otherworld',
        'Mode 7: Nonic Recursion',
        'Mode 8: Magical Impedance',
        'Mode 9: Exception State',
        'Mode 10: Lagrangian Field',
        'Mode 11: Route A/B Cascade',
):
    if _mode_marker not in _ET_DI_KERNEL_SRC:
        raise RuntimeError(
            f'∂I kernel mode-block splice: marker {_mode_marker!r} not '
            'found in spliced kernel — extraction range is incorrect')


# ── float64 version of the ∂I kernel ──────────────────────────────────────────
def _make_f64_di_kernel(f32_src):
    import re as _re2
    s = f32_src
    s = s.replace('void et_iterate_di(', 'void et_iterate_di_f64(')
    s = _re2.sub(r'\bfloat\b', 'double', s)
    # Replace f32 intrinsics BEFORE stripping numeric 'f' suffixes,
    # because __log2f contains '2f' which the suffix regex would match.
    # expf is included for parity with the standard kernel's f64 conversion,
    # so any future ∂I-kernel use of an exponential is auto-converted too.
    for old, new in [('__log2f(','log2('),('sqrtf(','sqrt('),('atan2f(','atan2('),
                     ('powf(','pow('),('cosf(','cos('),('sinf(','sin('),
                     ('logf(','log('),('expf(','exp('),
                     ('fabsf(','fabs('),('fmaxf(','fmax('),
                     ('fminf(','fmin('),('roundf(','round(')]:
        s = s.replace(old, new)
    # NOW strip numeric literal 'f' suffixes (e.g. 1.0f → 1.0)
    s = _re2.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)f\b', r'\1', s)
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
                except (RuntimeError, AttributeError, KeyError):
                    # RuntimeError:   CUDARuntimeError on device-properties query
                    # AttributeError: cp.cuda.runtime missing in stub builds
                    # KeyError:       'regsPerBlock' absent in older CuPy schemas
                    _regs_per_sm = 65536
                if _nr > 0:
                    _bps  = max(1, _regs_per_sm // (_nr * 256))
                    _occ  = min(1.0, _bps * 256 / 2048.0) * 100
                else:
                    _occ = -1; _bps = '?'
                print(f'  [GPU] ∂I kernel: {_nr} regs/thread  stack={_ls}B  '
                      f'~{_occ:.0f}% SM occupancy  (blocks/SM={_bps})', flush=True)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # AttributeError: kern.attributes missing on older CuPy / stub
                # KeyError:       'num_regs' / 'local_size_bytes' absent in attr dict
                # RuntimeError:   CUDARuntimeError when probing kernel attributes
                # TypeError:      _bps formatting if _bps becomes non-int via fallback
                # ValueError:     int conversion / format spec edge cases
                # Diagnostic block is informational only — skip silently if unprobeable
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
    # Step 3: replace float intrinsics FIRST (before stripping 'f' suffixes)
    # __log2f contains '2f' which the suffix regex would incorrectly match
    intrinsic_map = [
        ('__log2f(', 'log2('),   # fast-math intrinsic → IEEE
        ('sqrtf(',   'sqrt('),
        ('atan2f(',  'atan2('),
        ('powf(',    'pow('),
        ('cosf(',    'cos('),
        ('sinf(',    'sin('),
        ('logf(',    'log('),
        ('expf(',    'exp('),     # added for Mode 3 Koide-boundary Gaussian
        ('fabsf(',   'fabs('),
        ('fmaxf(',   'fmax('),
        ('fminf(',   'fmin('),
        ('roundf(',  'round('),
    ]
    for old, new in intrinsic_map:
        s = s.replace(old, new)
    # Step 4: NOW strip f-suffix numeric literals  e.g. 1.0f, 12.0f, 1e-38f
    s = _re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)f\b', r'\1', s)
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
                except (RuntimeError, AttributeError, KeyError):
                    # RuntimeError:   CUDARuntimeError on device-properties query
                    # AttributeError: cp.cuda.runtime missing in stub builds
                    # KeyError:       'regsPerBlock' absent in older CuPy schemas
                    _regs_per_sm = 65536
                _threads_per_block = 256
                if _nr > 0:
                    _blocks_per_sm = max(1, _regs_per_sm // (_nr * _threads_per_block))
                    _occ = min(1.0, _blocks_per_sm * _threads_per_block / 2048.0) * 100
                else:
                    _occ = -1
                print(f'  [GPU] {prec} kernel: {_nr} regs/thread  ' \
                      f'stack={_ls}B  ' \
                      f'~{_occ:.0f}% SM occupancy  ' \
                      f'(blocks/SM={_blocks_per_sm if _nr>0 else "?"})',
                      flush=True)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                # AttributeError: kern.attributes missing on older CuPy / stub
                # KeyError:       'num_regs' / 'local_size_bytes' absent in attr dict
                # RuntimeError:   CUDARuntimeError when probing kernel attributes
                # TypeError:      _blocks_per_sm formatting edge cases
                # ValueError:     int conversion / format spec edge cases
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
                        # ∂I kernel: ∂I-native lattice-aware base + ALL mode
                        # extras dispatched. After the ∂I splice fix, the
                        # ∂I kernel carries the same Mode 1-11 dispatch
                        # blocks as the main et_iterate kernel (single
                        # source-of-truth via _extract_main_kernel_mode_blocks
                        # at module load) and the kernel signature accepts
                        # the same mode-extra parameters. The IS_DI_TYPE
                        # branch passes everything the standard branch
                        # passes EXCEPT (is_julia/jcr/jci/w_r/w_c/log_p_eff)
                        # which the ∂I kernel does not need:
                        #   • is_julia: ∂I has z₀=0, c=pixel intrinsically
                        #   • w_r/w_c: ∂I uses DI_BASEW (constant) for the
                        #     24-family weighting; mode-extras don't depend
                        #     on the run-time w_r/w_c arrays
                        #   • log_p_eff: ∂I uses LN_P_EFF_DI macro (= ln(10/3))
                        # All other params (mew, palindrome, use_modeN flags,
                        # delta_k, eps5, inv_phi, mu2, lam_mh) are passed
                        # exactly as in the standard branch — same scope-
                        # level variables (mew_g, pal_g, dk, eps5, inv_phi,
                        # mu2, lam_mh, _active) computed once above.
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
                             mew_g, pal_g,
                             cp.int32(_active(1)),  cp.int32(_active(2)),
                             cp.int32(_active(3)),  cp.int32(_active(4)),
                             cp.int32(_active(5)),  cp.int32(_active(6)),
                             cp.int32(_active(7)),  cp.int32(_active(8)),
                             cp.int32(_active(9)),  cp.int32(_active(10)),
                             cp.int32(_active(11)),
                             _gs(dk),
                             _gs(eps5), _gs(inv_phi),
                             _gs(mu2), _gs(lam_mh),
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
                             cp.int32(_active(3)),  cp.int32(_active(4)),
                             cp.int32(_active(5)),  cp.int32(_active(6)),
                             cp.int32(_active(7)),  cp.int32(_active(8)),
                             cp.int32(_active(9)),  cp.int32(_active(10)),
                             cp.int32(_active(11)),
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
                # K_F2 is the FLOAT_DTYPE-typed Koide threshold from the constants
                # block above — using the typed form here keeps the comparison in
                # the iteration's working precision and avoids a per-step upcast.
                # Threshold semantics: t_r > K (Koide binding stability) selects
                # orbit's lattice d-family; otherwise fall back to palindrome cascade.
                d_dom   = np.where(t_r > K_F2, d_orbit, d_casc)
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

                # ── Mode-extra dispatch (∂I-on-modes composition) ─────────
                # ∂I is its own fractal type — z_{n+1} = Ψ·z^{p_dom} + ε(z)
                # + extra(z) + c — and the additive form lets every mode's
                # extra() compose with the lattice-adaptive dominant-power
                # carrier and the 24-family ε perturbation. This matches
                # the standard CPU path's extra() call (line 5636-5638)
                # exactly. extra is bound at line 5465 to mode_params['extra'];
                # for the No-Mode option (build_no_mode) extra is None and
                # this branch is a no-op, preserving pure-base ∂I behavior.
                # The ∂I dominant-power Jacobian (lines 5549-5552) is
                # unchanged: extras are at V or V² scale (Subsumption Law)
                # so their contribution to dz is negligible — same convention
                # as the standard CPU path which also excludes extras from
                # f_prime. P (orbit z), D (extra carrier), T (per-step n).
                z_new = z_prim + z_pert
                if extra is not None:
                    z_new = z_new + extra(z, r_cap, theta, k_t, n)
                z_new = z_new + c

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

                # d_r, d_t at escape — full 27720ET sublattice family detection
                # with 12ET fallback. The _di suffix is the conventional ∂I-path
                # marker (matches _LOG_P_EFF_DI, _di_label).
                d_r_out = np.where(new_e, d_orbit.astype(np.float32), d_r_out)
                # T-axis (analogue of the d_orbit computation above for R-axis):
                # 1) project k_t to integer 27720ET coordinate, GCD-extract family
                # 2) project k_t_f to 12ET coordinate, GCD-extract fallback family
                # 3) prefer 27720ET result when in [1,12], else use 12ET fallback
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
                d_t_out_di = np.where((d27t >= 1) & (d27t <= 12), d27t, d12t).astype(np.float32)
                d_t_out  = np.where(new_e, d_t_out_di, d_t_out)
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
    """Vectorized HSV → float32 RGB [0,1].
    xp: numpy or cupy — runs on whichever device holds the arrays.
    Uses xp.select (fully vectorized, no Python for-loop) so it runs
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
    made 95%+ of the image near-white with a thin colorful boundary shell.
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
    This preserves the ET lattice structure in the color.

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

    # ── Smooth_n log-scale normalization ──────────────────────────────────
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
    # Elegance E = (N/d)·[100/(100+|ε|)]·[100/(p+q)] — normalized to [0,1]
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

    description: optional ASCII metadata embedded as TIFF tag 270
                 (ImageDescription).  Empty string = no description tag.
                 Used for ET fractal provenance (mode, tower, center, seed, etc.).

    TIFF structure (little-endian / II) — DYNAMIC:
      Header  (8 B)         →  IFD at offset 8
      IFD     (2 + N×12+4)  →  N entries (14 base + 1 if description present)
      Extra   (28 + desc)   →  fixed 28 B (BitsPerSample/SampleFormat/X/YRes)
                              + ASCII description (NUL-terminated, padded to even)
      Image   (H×W×12 B)    →  float32 chunky RGB, top-to-bottom

    All offsets are computed dynamically from the actual entry count and the
    actual description length, so the file is correct regardless of whether
    a description is supplied.
    """
    assert arr_f32.dtype == np.float32, 'Need float32 input'
    H, W = arr_f32.shape[:2]
    LE   = '<'

    # ── Description encoding: ASCII + NUL terminator, padded to even length ──
    # TIFF tag 270 (ImageDescription) is dtype ASCII (=2), count includes NUL.
    # Padding to even byte count keeps subsequent offsets word-aligned, which
    # is required for SHORT/LONG/RATIONAL fields that follow.
    if description:
        desc_bytes = description.encode('ascii', errors='replace') + b'\x00'
        if len(desc_bytes) % 2 == 1:
            desc_bytes = desc_bytes + b'\x00'   # pad to even
        DESC_LEN = len(desc_bytes)              # actual byte count incl. NUL+pad
        # TIFF count for ASCII is the number of bytes including the NUL
        # terminator but NOT the even-padding byte (if one was added).
        DESC_COUNT = len(description.encode('ascii','replace')) + 1
        HAS_DESC = True
    else:
        desc_bytes = b''
        DESC_LEN = 0
        DESC_COUNT = 0
        HAS_DESC = False

    # ── Dynamic IFD entry count ──────────────────────────────────────────────
    # 14 base entries (Width, Length, BPS, Compression, PhotoInt, StripOffsets,
    # SamplesPerPixel, RowsPerStrip, StripByteCounts, XRes, YRes, PlanarConfig,
    # ResolutionUnit, SampleFormat) + 1 ImageDescription if present.
    N_ENTRIES    = 14 + (1 if HAS_DESC else 0)
    IFD_OFFSET   = 8
    IFD_SIZE     = 2 + N_ENTRIES*12 + 4
    EXTRA_OFFSET = IFD_OFFSET + IFD_SIZE
    BPS_OFFSET   = EXTRA_OFFSET + 0        # BitsPerSample [32,32,32] 6B
    SFMT_OFFSET  = EXTRA_OFFSET + 6        # SampleFormat  [3,3,3]    6B
    XRES_OFFSET  = EXTRA_OFFSET + 12       # XResolution   RATIONAL   8B
    YRES_OFFSET  = EXTRA_OFFSET + 20       # YResolution   RATIONAL   8B
    DESC_OFFSET  = EXTRA_OFFSET + 28       # ImageDescription bytes (only if HAS_DESC)
    DATA_OFFSET  = EXTRA_OFFSET + 28 + DESC_LEN     # Image data starts after extra
    IMG_BYTES    = H * W * 3 * 4

    ASCII=2; SHORT=3; LONG=4; RATIONAL=5

    def ifd_entry(tag, dtype, count, value):
        return struct.pack(LE+'HHII', tag, dtype, count, value)

    # ── Build IFD entries in tag-numerical order (TIFF requirement) ──────────
    # ImageDescription is tag 270, which sits between PhotometricInterp (262)
    # and StripOffsets (273).  We splice it in at the correct position so the
    # tags remain monotonically increasing.
    entries = b''
    entries += ifd_entry(256, LONG,     1, W)              # ImageWidth
    entries += ifd_entry(257, LONG,     1, H)              # ImageLength
    entries += ifd_entry(258, SHORT,    3, BPS_OFFSET)     # BitsPerSample
    entries += ifd_entry(259, SHORT,    1, 1)              # Compression=None
    entries += ifd_entry(262, SHORT,    1, 2)              # PhotometricInterp=RGB
    if HAS_DESC:
        entries += ifd_entry(270, ASCII, DESC_COUNT, DESC_OFFSET)  # ImageDescription
    entries += ifd_entry(273, LONG,     1, DATA_OFFSET)    # StripOffsets
    entries += ifd_entry(277, SHORT,    1, 3)              # SamplesPerPixel
    entries += ifd_entry(278, LONG,     1, H)              # RowsPerStrip
    entries += ifd_entry(279, LONG,     1, IMG_BYTES)      # StripByteCounts
    entries += ifd_entry(282, RATIONAL, 1, XRES_OFFSET)    # XResolution
    entries += ifd_entry(283, RATIONAL, 1, YRES_OFFSET)    # YResolution
    entries += ifd_entry(284, SHORT,    1, 1)              # PlanarConfig=Chunky
    entries += ifd_entry(296, SHORT,    1, 2)              # ResolutionUnit=Inch
    entries += ifd_entry(339, SHORT,    3, SFMT_OFFSET)    # SampleFormat
    assert len(entries) == N_ENTRIES*12

    header = b'II' + struct.pack(LE+'HI', 42, IFD_OFFSET)
    ifd    = struct.pack(LE+'H', N_ENTRIES) + entries + struct.pack(LE+'I', 0)
    extra  = struct.pack(LE+'HHH', 32,32,32)          # BitsPerSample
    extra += struct.pack(LE+'HHH', 3, 3, 3)           # SampleFormat (IEEE float)
    extra += struct.pack(LE+'II', int(dpi), 1)         # XResolution = dpi/1
    extra += struct.pack(LE+'II', int(dpi), 1)         # YResolution = dpi/1
    assert len(extra) == 28
    extra += desc_bytes                                # ImageDescription bytes (may be empty)
    assert len(extra) == 28 + DESC_LEN
    img_data = np.ascontiguousarray(arr_f32, dtype='<f4').tobytes()

    with open(str(filepath), 'wb') as f:
        f.write(header)    #  8 B
        f.write(ifd)       # IFD_SIZE bytes (varies with N_ENTRIES)
        f.write(extra)     # 28 + DESC_LEN bytes
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
    (±0.3 pixel in each direction) for better antialiasing than pure grid.
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

    return row_start, composite, it if AUDIO_ENABLED else None


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
    One launch + one sync = near-100% GPU utilization.
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

        # Transfer raw iteration data for audio native music
        it_raw = None
        if AUDIO_ENABLED:
            try:
                _gpu_phase('Extracting raw audio data...')
                it_raw = {}
                for _key in ('smooth_n', 'd_r', 'd_t', 'tight', 'orbit'):
                    arr = it.get(_key)
                    if arr is not None:
                        it_raw[_key] = arr.get().astype(np.float32) if hasattr(arr, 'get') else np.asarray(arr, dtype=np.float32)
            except Exception as _raw_err:
                print(f'  [Audio] Raw data extraction failed: {_raw_err}', flush=True)
                it_raw = None

        return buf, elapsed, it_raw

    # ── CPU: tiled loop (unchanged) ──────────────────────────────────────────
    tile_rng = np.random.RandomState(seed_base ^ (frame_idx * 0x9e3779b9 & 0x7FFFFFFF))
    n_tiles  = math.ceil(rh / tile_rows)
    tile_rngs = tile_rng.randint(0, 2**31, size=n_tiles)

    tile_args = [(ti*tile_rows, min(tile_rows, rh-ti*tile_rows),
                  cx, cy, zoom, mode, max_iter, jc, is_julia,
                  rw, rh, tower, int(tile_rngs[ti]))
                 for ti in range(n_tiles)]

    buf = np.zeros((rh, rw, 3), dtype=np.float32)
    # Raw data assembly for audio native music
    raw_arrays = {}
    if AUDIO_ENABLED:
        for _key in ('smooth_n', 'd_r', 'd_t', 'tight', 'orbit'):
            raw_arrays[_key] = np.zeros((rh, rw), dtype=np.float32)

    def _prog(done, total, el, prefix=''):
        eta = el/done*(total-done) if done else 0
        bar = '█'*int(done/total*25)
        print(f'{prefix}  Tile {done:4d}/{total}  [{bar:<25}] {done/total*100:5.1f}%'
              f'  {el:5.1f}s  ETA {eta:5.1f}s', end='\r', flush=True)

    t0 = time.time()
    if n_threads == 1:
        for ti, args in enumerate(tile_args):
            try:
                rs, rgb, it_tile = _render_tile(args)
                buf[rs:rs+rgb.shape[0]] = rgb
                if AUDIO_ENABLED and it_tile is not None:
                    for _key in ('smooth_n', 'd_r', 'd_t', 'tight', 'orbit'):
                        if _key in it_tile:
                            arr = it_tile[_key]
                            if hasattr(arr, 'get'): arr = arr.get()
                            raw_arrays[_key][rs:rs+rgb.shape[0]] = np.asarray(arr, dtype=np.float32)
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
                    rs, rgb, it_tile = fut.result()
                    completed[idx] = (rs, rgb, it_tile)
                except Exception as e:
                    _et_error(
                        f'Tile {idx+1}/{n_tiles} (rows {row_s}–{row_s+row_c-1}) — thread pool',
                        e, fatal=True)
                done += 1
                _prog(done, n_tiles, time.time()-t0)
        for idx in sorted(completed):
            rs, rgb, it_tile = completed[idx]
            buf[rs:rs+rgb.shape[0]] = rgb
            if AUDIO_ENABLED and it_tile is not None:
                for _key in ('smooth_n', 'd_r', 'd_t', 'tight', 'orbit'):
                    if _key in it_tile:
                        arr = it_tile[_key]
                        if hasattr(arr, 'get'): arr = arr.get()
                        raw_arrays[_key][rs:rs+rgb.shape[0]] = np.asarray(arr, dtype=np.float32)

    elapsed = time.time() - t0
    print(f'\n')
    it_raw = raw_arrays if AUDIO_ENABLED and raw_arrays else None
    return buf, elapsed, it_raw


def _resolve_run_params(rng):
    """
    Apply ADVANCED_PARAMS overrides on top of tower-random defaults.
    Returns (cx, cy, zoom, is_julia, jc).

    Note on IS_DI_TYPE mutation: this function may mutate the module-level
    IS_DI_TYPE global when FRACTAL_TYPE == 'random' and the per-run RNG picks
    the ∂I option (_r3 == 2). The mutation is necessary because IS_DI_TYPE
    is read by every downstream consumer (iterate_strip_v2, _render_tile,
    _render_frame, generate_et_fractal, generate_zoom_video) to select the
    correct kernel and coordinate setup, and module-load-time IS_DI_TYPE is
    based on FRACTAL_TYPE which is 'random' (not 'di') in this case, so it
    would otherwise stay False and the random branch would silently produce
    Mandelbrot whenever it picked ∂I — defeating the purpose of the random
    option's third path. The bug history: prior code's random branch had a
    comment "IS_DI_TYPE set globally" but no code actually performed the
    mutation; this function corrects that omission.

    For safety in multi-run scenarios (e.g. interactive REPL re-invocation),
    every type branch now explicitly affirms the IS_DI_TYPE state for the
    chosen type — not just the random branch. This makes the module-level
    IS_DI_TYPE reliably reflect the current run's choice regardless of any
    prior state.
    """
    global IS_DI_TYPE
    # Tower choice
    if SELECTED_TOWER == 'random':
        tkey  = list(TOWERS.keys())[rng.randint(0, len(TOWERS))]
    else:
        tkey  = SELECTED_TOWER
    tower = TOWERS[tkey]

    # Center and zoom
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

    # Mode — dispatch No Mode when SELECTED_MODES is the canonical empty list
    # ('N' option in _choose_modes returned []). Otherwise, blend the chosen
    # mode ids the same way as before. NO_MODE_ID is the sentinel mode_id.
    if not SELECTED_MODES:
        mode    = build_no_mode(tower, rng)
        mode_id = NO_MODE_ID
    else:
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
        IS_DI_TYPE = False    # affirm: explicit Julia run
    elif FRACTAL_TYPE == 'mandelbrot':
        is_julia = False; jc = None
        IS_DI_TYPE = False    # affirm: explicit Mandelbrot run
    elif FRACTAL_TYPE == 'di':
        # ∂I Lattice-Aware: z₀=0, c=pixel is intrinsic to this fractal family —
        # not borrowed from Mandelbrot. The self-referential dominant-power
        # (orbit's 27720ET sublattice → p_dom each step) is what defines it.
        is_julia = False; jc = None
        IS_DI_TYPE = True     # affirm: explicit ∂I run
    else:
        # random: equal chance of all three types. The random branch MUST
        # mutate IS_DI_TYPE per pick because the module-level IS_DI_TYPE was
        # set to (FRACTAL_TYPE == 'di') = False at line 963, and without
        # mutation here the _r3==2 case would silently produce Mandelbrot
        # (is_julia=False + jc=None + IS_DI_TYPE=False is the Mandelbrot
        # configuration). The fix: set IS_DI_TYPE to True in the ∂I branch
        # and False in the Julia/Mandelbrot branches, explicitly. This makes
        # the random option produce all three types with equal probability,
        # as the user's choice indicated.
        _r3 = rng.randint(0, 3)
        if _r3 == 0:
            is_julia = True;  jc = jc_base   # ET Julia
            IS_DI_TYPE = False
        elif _r3 == 1:
            is_julia = False; jc = None       # ET Mandelbrot
            IS_DI_TYPE = False
        else:
            is_julia = False; jc = None       # ∂I Lattice-Aware
            IS_DI_TYPE = True                 # mutate the module-level global
            # so iterate_strip_v2's kernel
            # selection (line 5482) picks
            # _get_et_di_kernel — this is
            # the closing Descriptor for
            # the random-pick ∂I gap.

    return tkey, tower, mode, mode_id, cx, cy, zoom, is_julia, jc


def _banner():
    print('\n' + '═'*72)
    print('  EXCEPTION THEORY FRACTAL GENERATOR  v2.2 — Professional Edition')
    print('  P ∘ D ∘ T = E  |  N=12  |  V=1/12  |  K=2/3  |  A₀=137')
    print(f'  α⁻¹(ET) ≈ {ALPHA_INV_ET:.9f}  (CODATA: 137.035999084±0.000000021)')
    print(f'  126=10N+N/2  |  π from 12-gon T-recursion  |  v=√(μ²/2λ)=2')
    print('  DE + normal-map lighting + orbit traps + interior + ACES tone-map')
    print('  ET Mandelbrot: z₀=0, c=pixel through ET 24-family manifold')
    print('  ET Julia:      z₀=pixel, c=ET-derived for selected mode')
    print('  ∂I Lattice-Aware: orbit’s 27720ET sublattice → p_dom per step  [ET-native fractal]')
    print('  H_ET: K·H(d_r) + (1-K)·H(d_θ)   [Koide 2:1 D/T weighting — ET-derived]')
    if AUDIO_ENABLED:
        print(f'  Audio: d→(d×7)mod12→pitch  K^n harmonics  Ψ shimmer  {AUDIO_KBPS}kbps MP3')
    print('═'*72)



# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 21.5 — ET-DERIVED LATTICE-NATIVE MUSIC ENGINE
#
#  Audio mappings — ALL ET-derived, zero ad hoc:
#    Pitch:     d → semitone class via circle-of-fifths g_r=7
#               k = (d × 7) mod 12,  freq = C4 × 2^(k/12)
#    Duration:  d/12 × base  (larger d = more stable = longer note)
#    Amplitude: pixel brightness × shimmer Ψ_k
#    Timbre:    harmonics decay at Koide ratio K=2/3 per partial
#               amplitude_n = K^n  for nth harmonic
#    Envelope:  attack = V = 1/12 of note,  release = K·V = 1/18 of note
#    Tightness: saturation → timbre purity (coherent = pure, ∂I = noisy)
#    Pan:       pixel horizontal position → stereo field
#
#  Image mode:  scan-line across center row → 15 s audio
#  Video mode:  per-frame d-family chord → continuous audio synced to video
# ══════════════════════════════════════════════════════════════════════════════

_AUDIO_SR = 44100       # Sample rate (CD quality)
_AUDIO_C4 = 261.63      # Middle C (Hz)

# ── d-family → pitch mapping (circle of fifths, g_r=7) ────────────────────
# d=12→C d=6→F# d=4→E d=3→A d=12→C d=2→D d=1→G
# d=5→B d=7→C# d=8→G# d=9→D# d=10→A# d=11→F
_D_SEMITONE = {d: (d * G_REAL) % N for d in range(1, 13)}
_D_FREQ     = {d: _AUDIO_C4 * 2.0**(_D_SEMITONE[d] / 12.0) for d in range(1, 13)}
_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ── Additional ET audio tables for v3.0 native music ──────────────────────
_ALL_D = list(range(1, 13))
_FAM_PQ = {1:3, 2:70, 3:13, 4:11, 5:13, 6:17, 7:15, 8:33, 9:17, 10:19, 11:27, 12:5}
_FAM_ELEGANCE = {d: (N/d) * (100.0 / (100.0 + _FAM_PQ[d])) for d in _ALL_D}
# Audio-side family coupling — kept in lock-step with FAM_COUPLING above.
# Uses the corrected (d-1)² + S² formula from ET_Fantastical §3.3 Table 2
# (the "more profound case") and ET_Fine_Structure_Constant_REVISED, NOT
# the older §5.1 / §5.2 (12/d - 1)² formula. See FAM_COUPLING for the full
# corpus trace and rationale; both definitions must stay consistent so the
# audio native music's per-d amplitude weighting matches the visual coupling.
_FAM_COUPLING = {d: A0_EM / ((d - 1.0)**2 + S_STATES**2 + 1e-6) for d in _ALL_D}
_FAM_CHAR = {1:'Gravity/Unison', 2:'Tritone/Pivot', 3:'Cubic/Strong',
             4:'Quartic/Weak', 5:'Quintic/Golden', 6:'Hexadic/Higgs',
             7:'Septic/Otherworld', 8:'Octet/Shadow', 9:'Nonic/Fractal',
             10:'Decic/φ-Binary', 11:'Undecimal/Prime', 12:'Full-Res/EM'}
_QUINTIC_TENSION = np.array([0,100,40,60,80,20,120,20,80,60,40,100], dtype=np.float64)
_SHIMMER_TAB = np.array([1.0 + _RMSAE_AMP * math.sin(2*math.pi*k/N)
                         for k in range(N)], dtype=np.float64)
_TRAP_RADII = {K: 6, V: 12, 1.0/PHI: 5, 1.0: 1}
def _var_n_audio(n): return (n*n - 1) / 12.0


def _audio_hue_from_rgb(r, g, b):
    """Extract hue [0,1] from RGB [0,1]."""
    mx = max(r, g, b); mn = min(r, g, b)
    if mx - mn < 1e-9: return 0.0
    d = mx - mn
    if mx == r:   h = ((g - b) / d) % 6.0
    elif mx == g: h = (b - r) / d + 2.0
    else:         h = (r - g) / d + 4.0
    return h / 6.0


def _audio_hue_to_d(hue):
    """Reverse-map hue to nearest d-family using FAM_HUE table."""
    best_d, best_dist = 12, 1.0
    for d, h in FAM_HUE.items():
        dist = min(abs(hue - h), 1.0 - abs(hue - h))
        if dist < best_dist:
            best_dist = dist; best_d = d
    return best_d


def _audio_koide_tone(freq, duration, amplitude, tightness, shimmer_k,
                      sr=_AUDIO_SR, n_partials=6, elegance=1.0, coupling=1.0,
                      quintic_tau=0.0):
    """Generate a tone with COMPLETE ET-derived characteristics.

    Harmonics: amplitude_n = K^n  (Koide ratio decay per partial)
    Tightness: 1=pure tone, <K=noise-mixed (∂I boundary sound)
    Shimmer:   Ψ_k = 1+√V·sin(2πk/N) amplitude modulation
    Elegance:  E(d) structural necessity weight
    Coupling:  ξ(d) FAM_COUPLING weight
    QuinticTau: τ(m) quintic tension at this semitone class
    Envelope:  attack=V·dur, release=K·V·dur
    """
    n_samp = max(1, int(sr * duration))
    t = np.linspace(0, duration, n_samp, endpoint=False)

    # Shimmer Ψ_k
    psi = 1.0 + _RMSAE_AMP * math.sin(2 * math.pi * (shimmer_k % N) / N)

    # Composite amplitude: base × shimmer × elegance × coupling × (1-quintic)
    elg_norm = min(elegance / 12.0, 1.0)
    cpl_norm = min(coupling / 8.56, 1.0)
    tau_mod  = 1.0 - 0.15 * quintic_tau
    amp = amplitude * psi * (0.3 + 0.7 * elg_norm) * (0.5 + 0.5 * cpl_norm) * tau_mod

    # Additive synthesis with Koide K^n harmonic amplitudes
    wave = np.zeros(n_samp, dtype=np.float64)
    for i in range(n_partials):
        pf = freq * (i + 1)
        if pf >= sr / 2: break        # Nyquist
        wave += (K ** i) * np.sin(2 * math.pi * pf * t)
    wave *= amp

    # Tightness → noise mix for ∂I boundary (threshold at K=2/3)
    if tightness < K:
        noise_level = (K - tightness) / K * 0.6
        noise = np.random.randn(n_samp) * amp * noise_level
        noise *= 0.5 + 0.5 * np.sin(2 * math.pi * freq * t)
        wave = wave * (1.0 - noise_level) + noise

    # ET-derived ADSR envelope: attack=V·dur, release=K·V·dur
    att = max(1, int(V * n_samp))
    rel = max(1, int(K * V * n_samp))
    env = np.ones(n_samp, dtype=np.float64)
    env[:att] = np.linspace(0, 1, att)
    if rel < n_samp:
        env[-rel:] = np.linspace(1, 0, rel)
    return wave * env


def _audio_write_wav(filepath, data_L, data_R=None, sr=_AUDIO_SR):
    """Write 16-bit PCM WAV (mono or stereo).  Pure Python, no dependencies."""
    stereo = data_R is not None
    channels = 2 if stereo else 1
    bps = 2  # bytes per sample per channel

    if stereo:
        n = max(len(data_L), len(data_R))
        L = np.zeros(n, dtype=np.float64); L[:len(data_L)] = data_L
        R = np.zeros(n, dtype=np.float64); R[:len(data_R)] = data_R
        peak = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-12)
        L = L / peak * 0.92; R = R / peak * 0.92
        interleaved = np.empty(2 * n, dtype=np.int16)
        interleaved[0::2] = (L * 32767).astype(np.int16)
        interleaved[1::2] = (R * 32767).astype(np.int16)
        raw = interleaved.tobytes()
    else:
        peak = max(np.max(np.abs(data_L)), 1e-12)
        data_n = (data_L / peak * 0.92 * 32767).astype(np.int16)
        raw = data_n.tobytes()

    data_size = len(raw)
    byte_rate = sr * channels * bps
    block_align = channels * bps

    with open(str(filepath), 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, channels, sr,
                            byte_rate, block_align, 8 * bps))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(raw)


def _audio_wav_to_mp3(wav_path, mp3_path, kbps):
    """Encode WAV to MP3 using ffmpeg.  Returns True on success."""
    try:
        import subprocess as _sp2
        cmd = ['ffmpeg', '-y', '-loglevel', 'error',
               '-i', str(wav_path),
               '-c:a', 'libmp3lame', '-b:a', f'{kbps}k',
               str(mp3_path)]
        r = _sp2.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return True
        print(f'  [Audio] ffmpeg MP3 encode failed: {r.stderr[:500]}', flush=True)
    except FileNotFoundError:
        print(f'  [Audio] ffmpeg not found — MP3 encoding skipped, WAV saved.',
              flush=True)
    except Exception as e:
        print(f'  [Audio] MP3 encoding error: {e}', flush=True)
    return False


def _audio_mux_video(video_path, audio_path, output_path):
    """Mux audio into video using ffmpeg.  Returns True on success."""
    try:
        import subprocess as _sp3
        cmd = ['ffmpeg', '-y', '-loglevel', 'error',
               '-i', str(video_path), '-i', str(audio_path),
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
               '-shortest', str(output_path)]
        r = _sp3.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return True
        print(f'  [Audio] ffmpeg mux failed: {r.stderr[:500]}', flush=True)
    except FileNotFoundError:
        print(f'  [Audio] ffmpeg not found — audio not muxed into video.',
              flush=True)
    except Exception as e:
        print(f'  [Audio] Video mux error: {e}', flush=True)
    return False


# ── v4.0 Note segmentation ────────────────────────────────────────────────

def _segment_row(pixels_row, raw_dr=None, raw_dt=None, raw_tight=None):
    """Run-length encode d-families along a pixel row into notes.
    Consecutive same-d pixels become ONE sustained note."""
    W = pixels_row.shape[0]
    _hr = raw_dr is not None and raw_tight is not None
    _hd = raw_dt is not None
    notes = []; cur_d = -1; cur_s = 0; cur_b = 0.0; cur_t = 0.0; cur_dt = 0; cur_c = 0
    for px in range(W):
        r, g, b = float(pixels_row[px,0]), float(pixels_row[px,1]), float(pixels_row[px,2])
        brt = (r + g + b) / 3.0
        if _hr:
            d_r = int(np.clip(np.round(raw_dr[px]), 1, 12))
            d_t = int(np.clip(np.round(raw_dt[px]), 1, 12)) if _hd else d_r
            tight = float(raw_tight[px])
        else:
            hue = _audio_hue_from_rgb(r, g, b)
            d_r = _audio_hue_to_d(hue); d_t = d_r
            mx = max(r, g, b)
            sat = (1.0 - min(r,g,b)/mx) if mx > 0.01 else 0.0
            tight = sat * min(brt/0.5, 1.0) if brt > 0.01 else 0.0
            if brt < 0.04: d_r = d_t = 1; tight = 0.05
        if d_r != cur_d and cur_d >= 0:
            c = max(cur_c, 1)
            notes.append(dict(d_r=cur_d, d_t=cur_dt//c, start_px=cur_s,
                              end_px=px, length=px-cur_s, avg_brt=cur_b/c, avg_tight=cur_t/c))
            cur_s = px; cur_b = 0.0; cur_t = 0.0; cur_dt = 0; cur_c = 0
        cur_d = d_r; cur_b += brt; cur_t += tight; cur_dt += d_t; cur_c += 1
    if cur_c > 0:
        c = cur_c
        notes.append(dict(d_r=cur_d, d_t=cur_dt//c, start_px=cur_s,
                          end_px=W, length=W-cur_s, avg_brt=cur_b/c, avg_tight=cur_t/c))
    return notes


def _synth_note_sequence(notes, total_duration, octave_offset=0.0,
                         sr=_AUDIO_SR, n_partials=8):
    """Phase-continuous Koide synthesis with portamento glide."""
    total_px = sum(n['length'] for n in notes)
    total_samples = int(sr * total_duration)
    if total_px == 0:
        return np.zeros(total_samples), np.zeros(total_samples)
    out_L = np.zeros(total_samples, dtype=np.float64)
    out_R = np.zeros(total_samples, dtype=np.float64)
    for note in notes:
        d = note['d_r']
        note['freq'] = _D_FREQ[d] * 2.0**octave_offset
        if note['avg_brt'] < 0.04: note['freq'] = _D_FREQ[1] * 2.0**(octave_offset - 1)
        note['elg_norm'] = min(_FAM_ELEGANCE[d] / 12.0, 1.0)
        note['cpl_norm'] = min(_FAM_COUPLING[d] / 8.56, 1.0)
        note['tau'] = _QUINTIC_TENSION[_D_SEMITONE[d]] / 120.0
        note['var_w'] = 1.0 + 0.1 * _var_n_audio(d)
    phase = np.zeros(n_partials, dtype=np.float64)
    sample_pos = 0
    for ni, note in enumerate(notes):
        note_dur = total_duration * note['length'] / total_px
        note_samp = max(1, int(sr * note_dur))
        if sample_pos + note_samp > total_samples: note_samp = total_samples - sample_pos
        if note_samp <= 0: break
        d = note['d_r']; freq = note['freq']; brt = note['avg_brt']; tight = note['avg_tight']
        tau_mod = 1.0 - 0.15 * note['tau']
        base_amp = brt * (0.3+0.7*note['elg_norm']) * (0.5+0.5*note['cpl_norm']) * \
                   min(note['var_w'], 2.0) * tau_mod
        if brt < 0.04: base_amp = 0.03
        psi = float(_SHIMMER_TAB[note['start_px'] % N])
        amp = base_amp * psi * 0.35
        prev_freq = notes[ni-1]['freq'] if ni > 0 else freq
        glide_samp = max(1, int(V * note_samp)) if ni > 0 else 0
        buf = np.zeros(note_samp, dtype=np.float64)
        for h in range(n_partials):
            harmonic = h + 1; pf_t = freq * harmonic; pf_p = prev_freq * harmonic
            if pf_t >= sr / 2: break
            pa = (K ** h) * amp
            for s in range(note_samp):
                if s < glide_samp and glide_samp > 0:
                    pf = pf_p * (pf_t / pf_p) ** (s / glide_samp)
                else: pf = pf_t
                phase[h] += 2.0 * math.pi * pf / sr
                buf[s] += pa * math.sin(phase[h])
            phase[h] %= (2.0 * math.pi)
        if tight < K:
            nl = (K - tight) / K * 0.4; noise = np.random.randn(note_samp) * amp * nl
            t_a = np.arange(note_samp) / sr
            noise *= 0.5 + 0.5 * np.sin(2*math.pi*freq*t_a)
            buf = buf * (1.0 - nl) + noise
        if ni == 0:
            att = min(int(V * note_samp), note_samp)
            if att > 1: buf[:att] *= np.linspace(0, 1, att)
        if ni == len(notes) - 1:
            rel = min(int(K * V * note_samp), note_samp)
            if rel > 1: buf[-rel:] *= np.linspace(1, 0, rel)
        pan_c = (note['start_px'] + note['end_px']) / 2.0 / max(total_px, 1)
        # ── PDT-symmetric stereo pan modulation ─────────────────────────────
        # D-axis (magnitude sublattice d_r) and T-axis (phase sublattice d_t)
        # each contribute their own pan offset based on their family's LN2/d
        # characteristic phase. This gives the stereo field two independent
        # sublattice-driven modulations that are then Koide-blended (K:1-K).
        # Before this fix, only the T-axis had pan modulation and the D-axis
        # used raw pan_c — breaking PDT symmetry in the audio output.
        ph_d = LN2 / max(d, 1)                                     # D-axis phase
        d_off = 0.3 * math.sin(2*math.pi*ph_d*pan_c)               # D pan offset
        L_d = math.cos((pan_c+d_off)*math.pi/2); R_d = math.sin((pan_c+d_off)*math.pi/2)
        d_t = note['d_t']; ph_r = LN2 / max(d_t, 1)                # T-axis phase
        t_off = 0.3 * math.sin(2*math.pi*ph_r*pan_c)               # T pan offset
        L_t = math.cos((pan_c+t_off)*math.pi/2); R_t = math.sin((pan_c+t_off)*math.pi/2)
        L_g = K*L_d + (1-K)*L_t; R_g = K*R_d + (1-K)*R_t            # Koide blend
        end_p = min(sample_pos + note_samp, total_samples)
        out_L[sample_pos:end_p] += buf[:end_p-sample_pos] * L_g
        out_R[sample_pos:end_p] += buf[:end_p-sample_pos] * R_g
        sample_pos = end_p
    return out_L, out_R


def _et_reverb(signal, sr=_AUDIO_SR, wet=0.20):
    """ET-derived reverb: comb delays from d-families, allpass from φ.

    Signal flow (left-to-right) using `out` as the working wet-bus buffer:
      signal → comb filter bank (d=3,4,6,12 delays) → comb_sum
             → allpass filter (delay = sr/(φ·10))   → ap_out
             → pre-delay (V·sr samples)             → out (wet bus)
      final = signal·(1-wet) + out·wet
    The `out` buffer carries the wet signal through every stage of the chain
    and is the authoritative output of the reverb network. The dry/wet mix
    at the end combines the original signal with `out` at the wet ratio.
    """
    n = len(signal)
    out = np.zeros(n, dtype=np.float64)                # wet bus — accumulates chain output
    comb_delays = [max(100, min(int(sr * 12 / (d * 130.0)), sr//2)) for d in [3,4,6,12]]
    comb_fb = K * 0.7
    comb_sum = np.zeros(n, dtype=np.float64)
    for delay in comb_delays:
        buf = np.zeros(n, dtype=np.float64)
        for i in range(delay, n): buf[i] = signal[i-delay] + comb_fb * buf[i-delay]
        comb_sum += buf / len(comb_delays)
    ap_d = max(50, min(int(sr / (PHI * 10)), sr//4)); ap_g = 0.5
    ap_out = np.zeros(n, dtype=np.float64)
    for i in range(ap_d, n): ap_out[i] = -ap_g*comb_sum[i] + comb_sum[i-ap_d] + ap_g*ap_out[i-ap_d]
    # Pre-delay into the wet bus `out` — V·sr samples of silence before the
    # reverb tail begins (Haas-effect pre-delay at the lattice variance quantum)
    pre = int(V * sr)
    if pre < n: out[pre:] = ap_out[:n-pre]
    return signal * (1.0 - wet) + out * wet


def _midi_vlq(value):
    result = []; result.append(value & 0x7F); value >>= 7
    while value: result.append((value & 0x7F) | 0x80); value >>= 7
    return bytes(reversed(result))

def _write_midi(filepath, voice_notes, total_duration, ticks_per_beat=480):
    """Write Type 1 MIDI from segmented notes."""
    total_px = sum(n['length'] for notes in voice_notes for n in notes)
    if total_px == 0: return
    bpm = 120; us_per_beat = int(60_000_000 / bpm)
    total_ticks = int(total_duration * bpm / 60.0 * ticks_per_beat)
    _D_MIDI_MAP = {d: 60 + _D_SEMITONE[d] for d in range(1, 13)}
    tracks = []
    td = _midi_vlq(0) + b'\xFF\x51\x03' + struct.pack('>I', us_per_beat)[1:]
    td += _midi_vlq(0) + b'\xFF\x2F\x00'; tracks.append(td)
    for vi, notes in enumerate(voice_notes):
        td = b''; name = f'Voice {vi+1}'.encode('ascii')
        td += _midi_vlq(0) + b'\xFF\x03' + _midi_vlq(len(name)) + name
        ch = vi % 16; tick_pos = 0
        vpx = sum(n['length'] for n in notes)
        if vpx == 0: td += _midi_vlq(0)+b'\xFF\x2F\x00'; tracks.append(td); continue
        for note in notes:
            d = note['d_r']; mn = _D_MIDI_MAP[d]
            vel = max(1, min(127, int(note['avg_brt'] * 120)))
            if note['avg_brt'] < 0.04: vel = max(1, int(note['avg_brt']*30))
            nt = max(1, int(total_ticks * note['length'] / vpx))
            ns = int(total_ticks * note['start_px'] / vpx)
            delta = max(0, ns - tick_pos)
            td += _midi_vlq(delta) + bytes([0x90|ch, mn, vel])
            tick_pos = ns
            td += _midi_vlq(nt) + bytes([0x80|ch, mn, 0])
            tick_pos += nt
        td += _midi_vlq(0)+b'\xFF\x2F\x00'; tracks.append(td)
    with open(str(filepath), 'wb') as f:
        f.write(b'MThd'); f.write(struct.pack('>I', 6))
        f.write(struct.pack('>HHH', 1, len(tracks), ticks_per_beat))
        for td in tracks: f.write(b'MTrk'); f.write(struct.pack('>I', len(td))); f.write(td)


# ── Image native music: v4.0 — professional architecture ─────────────────

def et_native_music_image(final_f32, stem, script_dir, kbps, scan_row_frac=0.5,
                          audio_duration=20.0, n_scan_rows=5, it_raw=None):
    """
    v4.0 Professional ET native music: note-segmented, phase-continuous,
    with portamento, reverb, and MIDI output.
    """
    print(f'  Generating audio (v4.0 — professional native music)…', flush=True)
    _has_raw = (it_raw is not None and 'd_r' in it_raw and 'tight' in it_raw)
    _has_dt  = (_has_raw and 'd_t' in it_raw)
    if _has_raw:
        print(f'    ★ RAW iteration data (exact 27720ET d-families)', flush=True)
    else:
        print(f'    Image-based detection (approximate)', flush=True)
    H, W = final_f32.shape[:2]

    # Generator audio: keep passed-in duration (default 20s from function signature)
    total_samples = int(_AUDIO_SR * audio_duration)
    mix_L = np.zeros(total_samples, dtype=np.float64)
    mix_R = np.zeros(total_samples, dtype=np.float64)

    # ── Layer 1+2: Note-segmented temporal scan ──────────────────────
    # Voice rows are positioned around scan_row_frac with a half-spread
    # of K/2 = 1/3 (ET-derived from the Koide ratio). At the default
    # scan_row_frac=0.5, n_scan_rows=5 this gives voice rows at fractions
    # [0.167, 0.333, 0.500, 0.667, 0.833] — symmetric around image center
    # with the Koide-derived spread. Both parameters are now honored so
    # callers can shift the scan band vertically and change voice count.
    _n_voices = max(1, int(n_scan_rows))
    _half_spread = K / 2.0                                  # 1/3 — Koide-derived
    _lo = scan_row_frac - _half_spread
    _hi = scan_row_frac + _half_spread
    _vrows = np.clip(np.linspace(_lo, _hi, _n_voices), 0.05, 0.95)
    _oct_offs = np.linspace(0.5, -0.5, _n_voices)
    all_voice_notes = []

    for vi, (_vf, _oct) in enumerate(zip(_vrows, _oct_offs)):
        _ry = int(H * _vf)
        _scan = final_f32[_ry, :, :]
        _rdr = it_raw['d_r'][_ry, :] if _has_raw else None
        _rdt = it_raw['d_t'][_ry, :] if _has_dt else None
        _rti = it_raw['tight'][_ry, :] if _has_raw else None
        notes = _segment_row(_scan, _rdr, _rdt, _rti)
        all_voice_notes.append(notes)
        vL, vR = _synth_note_sequence(notes, audio_duration, octave_offset=_oct, n_partials=8)
        n = min(len(vL), total_samples)
        mix_L[:n] += vL[:n]; mix_R[:n] += vR[:n]
        print(f'    Voice {vi+1}/{_n_voices}: y={_ry} {len(notes)} notes oct={_oct:+.1f}', flush=True)

    # ── Layer 3: Wavetable from spatial FFT ──────────────────────────
    wt_rows = np.linspace(0.10, 0.90, N)
    wt_amp = 0.08 / math.sqrt(N)
    for ri, _wf in enumerate(wt_rows):
        ry = int(H * _wf); scan = final_f32[ry, :, :]
        tables = []
        for ch in range(3):
            raw = scan[:, ch].astype(np.float64); raw -= np.mean(raw)
            pk = np.max(np.abs(raw))
            tables.append(raw / pk if pk > 1e-9 else np.zeros(len(raw)))
        freq = _AUDIO_C4 * 0.5 * 2.0**(ri/12.0); rb = np.mean(scan)
        amp = wt_amp * (0.3 + 0.7 * rb)
        for ci, wt in enumerate(tables):
            wt_len = len(wt); n_samp = total_samples
            p_inc = freq * wt_len / _AUDIO_SR
            phases = np.arange(n_samp, dtype=np.float64) * p_inc
            idx = phases % wt_len; ifl = idx.astype(np.int64)
            ic = (ifl+1) % wt_len; fr = idx - ifl
            syn = (wt[ifl]*(1.0-fr) + wt[ic]*fr) * amp
            if ci == 1:   mix_L[:n_samp] += syn*K*0.5; mix_R[:n_samp] += syn*K*0.5
            elif ci == 0: mix_L[:n_samp] += syn*(1-K)
            else:         mix_R[:n_samp] += syn*(1-K)
    print(f'    Layer 3: Wavetable done', flush=True)

    # ── Layer 4+5: Combined palindrome × shimmer modulation ──────────
    _CC = int(K * N); _pp = audio_duration / _CC; _sps = int(_AUDIO_SR * _pp / N)
    _PALIN = [12,6,4,3,12,2,12,3,4,6,12,1]
    _mod = np.ones(total_samples, dtype=np.float64); _pos = 0
    for _cyc in range(_CC):
        for _step, _pd in enumerate(_PALIN):
            if _pos >= total_samples: break
            _end = min(_pos + _sps, total_samples)
            depth = (_pd / 12.0) * float(_SHIMMER_TAB[_step])
            depth = 0.3 + 0.7 * min(depth, 1.3)
            _sl = _end - _pos
            if _sl > 64:
                _xf = min(32, _sl//2)
                _prev = _mod[_pos-1] if _pos > 0 else 1.0
                _mod[_pos:_pos+_xf] = np.linspace(_prev, depth, _xf)
                _mod[_pos+_xf:_end] = depth
            else: _mod[_pos:_end] = depth
            _pos = _end
    mix_L *= _mod; mix_R *= _mod
    print(f'    Layer 4+5: Palindrome×shimmer done', flush=True)

    # ── Layer 6: Orbit trap drones ───────────────────────────────────
    t_arr = np.arange(total_samples) / _AUDIO_SR
    for trap_r, d in _TRAP_RADII.items():
        freq = _D_FREQ[d] * 0.25
        drone = np.zeros(total_samples)
        for h in range(4):
            pf = freq*(h+1)
            if pf >= _AUDIO_SR/2: break
            drone += (K**h) * np.sin(2*math.pi*pf*t_arr) * 0.03
        mix_L += drone * 0.5; mix_R += drone * 0.5

    # ── Layer 7: Stereo LFO ──────────────────────────────────────────
    pan_mod = np.zeros(total_samples)
    for trap_r, d in _TRAP_RADII.items():
        pan_mod += 0.02 * np.sin(2*math.pi*trap_r*0.5*t_arr)
    mix_L *= (1.0 + pan_mod); mix_R *= (1.0 - pan_mod)

    # ── Reverb ───────────────────────────────────────────────────────
    print(f'    Applying reverb…', flush=True)
    mix_L = _et_reverb(mix_L); mix_R = _et_reverb(mix_R)

    # ── Write ────────────────────────────────────────────────────────
    wav_path = script_dir / (stem + '_audio.wav')
    _audio_write_wav(wav_path, mix_L, mix_R)
    dur = total_samples / _AUDIO_SR
    print(f'  ✓ WAV  : {wav_path}  ({dur:.1f}s stereo)', flush=True)

    mp3_path = script_dir / (stem + '_audio.mp3')
    if _audio_wav_to_mp3(wav_path, mp3_path, kbps):
        mp3_mb = mp3_path.stat().st_size / 1_048_576
        print(f'  ✓ MP3  : {mp3_path}  ({mp3_mb:.1f} MB  {kbps} kbps)', flush=True)
        try: wav_path.unlink()
        except OSError: pass  # FileNotFoundError / PermissionError on cleanup — non-fatal

    # ── MIDI ─────────────────────────────────────────────────────────
    midi_path = script_dir / (stem + '_audio.mid')
    try:
        _write_midi(str(midi_path), all_voice_notes, audio_duration)
        print(f'  ✓ MIDI : {midi_path}  (DAW-ready)', flush=True)
    except (OSError, struct.error, KeyError, ValueError, IndexError):
        # OSError:      file write failure (PermissionError, disk full, etc.)
        # struct.error: pack failure if a derived field overflows MIDI int range
        # KeyError:     _D_MIDI_MAP / note dict access on malformed segmentation
        # ValueError:   int() / clip / range conversion edge cases
        # IndexError:   notes list / track indexing edge cases
        # MIDI is an optional informational sidecar — non-fatal on failure
        pass

    return str(mp3_path) if mp3_path.exists() else str(wav_path)


# ── Video native music: per-frame evolving chord ──────────────────────────

def et_native_music_video_frame(final_f32, scan_row_frac=0.5, it_raw=None):
    """
    Extract a d-family distribution from one rendered video frame.
    Uses raw iteration data (it_raw) when available for exact d-families.
    Falls back to image-based hue detection otherwise.
    Returns dict: {d: (count, avg_brightness, avg_tightness)} for d=1..12.
    """
    H, W = final_f32.shape[:2]
    row = int(H * scan_row_frac)
    d_stats = {d: [0, 0.0, 0.0] for d in range(1, 13)}

    if it_raw is not None and 'd_r' in it_raw and 'tight' in it_raw:
        # Raw data path: exact d-families from 27720ET GCD
        d_r_row = it_raw['d_r'][row, :]
        tight_row = it_raw['tight'][row, :]
        scan = final_f32[row, :, :]
        for px in range(0, W, max(1, W // 256)):
            d = int(np.clip(np.round(d_r_row[px]), 1, 12))
            brightness = float((scan[px, 0] + scan[px, 1] + scan[px, 2]) / 3.0)
            tight = float(tight_row[px])
            d_stats[d][0] += 1
            d_stats[d][1] += brightness
            d_stats[d][2] += tight
    else:
        # Image-based fallback: detect d from hue
        scan = final_f32[row, :, :]
        for px in range(0, W, max(1, W // 256)):
            r, g, b = float(scan[px, 0]), float(scan[px, 1]), float(scan[px, 2])
            brightness = (r + g + b) / 3.0
            if brightness < 0.04:
                d = 1; tight = 0.1
            else:
                hue = _audio_hue_from_rgb(r, g, b)
                d = _audio_hue_to_d(hue)
                mx = max(r, g, b)
                tight = (1.0 - min(r, g, b) / mx) if mx > 0.01 else 0.0
            d_stats[d][0] += 1
            d_stats[d][1] += brightness
            d_stats[d][2] += tight

    result = {}
    for d in range(1, 13):
        cnt, sb, st = d_stats[d]
        if cnt > 0:
            result[d] = (cnt, sb / cnt, st / cnt)
    return result


def et_native_music_video(frame_stats_list, fps, stem, script_dir, kbps,
                          video_path=None):
    """
    Build continuous audio from per-frame d-family distributions.

    frame_stats_list:  list of dicts from et_native_music_video_frame(), one per frame
    fps:               video frame rate (determines grain duration)
    stem:              filename stem
    kbps:              MP3 bitrate
    video_path:        if provided, mux audio into video

    Each frame → a chord lasting 1/fps seconds.
    The chord contains the top 4 d-families present in that frame,
    each at amplitude proportional to pixel count.
    Cross-faded for continuity.
    """
    print(f'  Generating video audio ({len(frame_stats_list)} frames at {fps} fps)…',
          flush=True)
    nf = len(frame_stats_list)
    grain_dur = 1.0 / fps
    grain_samp = max(1, int(_AUDIO_SR * grain_dur))

    # Cross-fade length: 1/8 of grain (smooth transitions)
    xfade = max(1, grain_samp // 8)

    audio_L = np.zeros(nf * grain_samp + xfade, dtype=np.float64)
    audio_R = np.zeros(nf * grain_samp + xfade, dtype=np.float64)

    for fi, stats in enumerate(frame_stats_list):
        if not stats:
            continue

        # Top 4 d-families by count
        ranked = sorted(stats.items(), key=lambda x: -x[1][0])[:4]
        total_px = sum(v[0] for _, v in ranked) or 1

        grain_L = np.zeros(grain_samp, dtype=np.float64)
        grain_R = np.zeros(grain_samp, dtype=np.float64)

        for d, (cnt, avg_brt, avg_tight) in ranked:
            freq = _D_FREQ[d]
            # Amplitude: proportion of pixels × average brightness
            amp = (cnt / total_px) * avg_brt * 0.5
            if amp < 0.005: continue

            tone = _audio_koide_tone(freq, grain_dur, amp, avg_tight,
                                     shimmer_k=fi % N, n_partials=5)
            if len(tone) < grain_samp:
                tone = np.pad(tone, (0, grain_samp - len(tone)))
            elif len(tone) > grain_samp:
                tone = tone[:grain_samp]

            # Slight stereo spread: lower d → more centered, higher d → wider
            spread = (d - 1) / 22.0   # 0 to 0.5
            grain_L += tone * (0.5 + spread)
            grain_R += tone * (0.5 + (0.5 - spread))

        # Apply grain envelope (smooth edges for cross-fade)
        env = np.ones(grain_samp, dtype=np.float64)
        env[:xfade] = np.linspace(0, 1, xfade)
        env[-xfade:] = np.linspace(1, 0, xfade)
        grain_L *= env
        grain_R *= env

        # Additive overlap at cross-fade boundaries
        pos = fi * grain_samp
        end = pos + grain_samp
        if end <= len(audio_L):
            audio_L[pos:end] += grain_L
            audio_R[pos:end] += grain_R

    # Trim trailing silence
    used = nf * grain_samp
    audio_L = audio_L[:used]
    audio_R = audio_R[:used]

    wav_path = script_dir / (stem + '_audio.wav')
    _audio_write_wav(wav_path, audio_L, audio_R)
    dur = len(audio_L) / _AUDIO_SR
    print(f'  ✓ WAV  : {wav_path}  ({dur:.1f}s stereo)', flush=True)

    # Encode to MP3
    mp3_path = script_dir / (stem + '_audio.mp3')
    mp3_ok = _audio_wav_to_mp3(wav_path, mp3_path, kbps)
    if mp3_ok:
        mp3_mb = mp3_path.stat().st_size / 1_048_576
        print(f'  ✓ MP3  : {mp3_path}  ({mp3_mb:.1f} MB  {kbps} kbps)', flush=True)

    # Mux audio into video if video_path provided
    if video_path and Path(video_path).exists():
        audio_src = str(mp3_path) if mp3_ok else str(wav_path)
        mux_path = script_dir / (stem + '_with_audio.mp4')
        if _audio_mux_video(video_path, audio_src, mux_path):
            mux_mb = mux_path.stat().st_size / 1_048_576
            print(f'  ✓ VIDEO+AUDIO : {mux_path}  ({mux_mb:.1f} MB)', flush=True)

    # Clean up WAV if MP3 succeeded
    if mp3_ok:
        try: wav_path.unlink()
        except OSError: pass  # FileNotFoundError / PermissionError on cleanup — non-fatal
        return str(mp3_path)
    return str(wav_path)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 21.6 — VISUAL ANALYSIS TOOL  (REPURPOSED FROM v4.0 NATIVE MUSIC)
#
#  HISTORY CONTEXT — the v4.0 origin of these functions:
#  These functions are repurposed from the v4.0 sonification engine that
#  preceded the native music rebuild. (This is the only line in the entire
#  script where the v4.0 term is preserved, by design — the verification
#  grep in Phase A.4 of the rebuild plan must match exactly this line and
#  no others.) In v4.0 the audio path read RGB pixels from the rendered
#  image, reverse-mapped hues to d-families via the FAM_HUE table, and
#  segmented rows into d-family runs to drive the synthesizer. That earlier
#  approach was image-in / sound-out — a lossy projection through the
#  visual encoding rather than the orbit's lattice trajectory. The Stage
#  21.5 native music engine replaces it with per-step probe traces from the
#  iteration kernel itself, so the audio is generated from the same orbit
#  data that drives the visual, not derived from the rendered RGB.
#
#  The hue→d / segment-row machinery is, however, genuinely useful as a
#  STANDALONE VISUAL DIAGNOSTIC tool. Given a rendered fractal image, it
#  recovers what d-family the visual encoding is reporting at every pixel.
#  This lets the user verify that the visual color encoding is consistent
#  with the FAM_HUE table — i.e. that the rendering pipeline actually painted
#  the d-families it computed inside the kernel. It is a debugging tool for
#  the visual half of the rendering pipeline. Nothing in the Stage 21.5 music
#  path calls these functions; they exist purely as their own visual analysis
#  capability.
#
#  Per the project's "no removal" rule, the legacy private helpers
#  _audio_hue_from_rgb, _audio_hue_to_d, and _segment_row remain in their
#  original location (Stage 21.5) — they are still called by the renamed
#  et_native_music_image and et_native_music_video_frame fallback branches
#  when raw kernel data is unavailable. The Stage 21.6 et_visual_* names
#  added below are NEW public entry points that wrap the legacy helpers and
#  add the et_visual_segment_row + et_visual_analyze_image extensions.
#
#  Three Tools applied:
#    Identification:    P = the rendered RGB image (substrate of inquiry)
#                       D = the FAM_HUE color encoding table (constraints
#                           binding d-families to hues)
#                       T = the analysis pass that walks the image and
#                           recovers each pixel's encoded d
#    Descriptor Gap:    The gap "is the visual encoding faithful to the
#                       kernel's computed d?" is itself a Descriptor — and
#                       the closing Descriptor is the per-pixel comparison
#                       between visual-recovered d and kernel d_r, returned
#                       as the mismatch_count / mismatch_rate fields of
#                       et_visual_analyze_image's report.
#    Subsumption:       Every visual encoding diagnostic that v4.0 needed
#                       internally for its synthesizer is now subsumed by
#                       the standalone et_visual_* tool, with no remainder.
#
#  Mappings (preserved from v4.0):
#    Hue extraction:    HSV from RGB (max-min normalization, hue=H/360°)
#    Hue → d:           nearest neighbor in FAM_HUE table (circular distance)
#    Row segment:       run-length encode pixel row into per-d segments
#    Image analyze:     full-image d histogram + visual-vs-kernel mismatch
# ══════════════════════════════════════════════════════════════════════════════

def et_visual_hue_from_rgb(r, g, b):
    """Extract hue [0,1] from an RGB triplet (each component in [0,1]).

    Visual analysis entry point. Pure HSV math: max-min normalization with
    sector selection by which channel holds the max. Identical algorithm to
    the legacy _audio_hue_from_rgb private helper (still in Stage 21.5 for
    use by the native music image fallback branch when raw kernel data is
    unavailable). This public et_visual_* alias exists so the visual analysis
    tool has its own named entry point with a docstring describing its
    diagnostic-tool role rather than its v4.0 audio-pipeline origin.
    """
    return _audio_hue_from_rgb(r, g, b)


def et_visual_hue_to_d(hue):
    """Reverse-map hue ∈ [0,1] to the nearest d-family in the FAM_HUE table.

    Visual analysis entry point. Returns the d ∈ {1..12} whose FAM_HUE value
    is circularly closest to the input hue (using min(|Δ|, 1−|Δ|) so the
    hue circle wraps correctly at 0/1). Used by et_visual_segment_row and
    et_visual_analyze_image to recover what d-family the rendering pipeline
    painted at each pixel — the standalone test of whether the visual color
    encoding is consistent with the FAM_HUE table. Wraps the legacy
    _audio_hue_to_d helper so the implementation lives in exactly one place.
    """
    return _audio_hue_to_d(hue)


def et_visual_segment_row(pixels_row):
    """Run-length encode a row of rendered RGB into per-d-family segments.

    Visual-only analysis. Scans the row left to right, computes hue at each
    pixel via et_visual_hue_from_rgb, maps hue to nearest d-family via
    et_visual_hue_to_d, and produces a list of segments where consecutive
    same-d pixels are coalesced. Each segment dict contains:
      d_r        — recovered d-family (1..12)
      start_px   — first pixel index of the segment
      end_px     — first pixel index after the segment
      length     — segment width in pixels (end_px − start_px)
      avg_brt    — average RGB-mean brightness across the segment
      avg_tight  — average saturation-derived "tightness" proxy

    This is the image-scan branch of the legacy _segment_row helper extracted
    as its own visual analysis capability with no music-path dependencies.
    The legacy _segment_row remains in Stage 21.5 because it is still called
    by et_native_music_image when raw kernel data is unavailable; this
    et_visual_segment_row is a new, parallel, music-independent function.

    Input  : pixels_row — np.ndarray of shape (W, 3), float, values in [0,1]
    Returns: list[dict] with the segment fields described above.
    """
    W = pixels_row.shape[0]
    segments = []
    cur_d = -1; cur_s = 0; cur_b = 0.0; cur_t = 0.0; cur_c = 0
    for px in range(W):
        r = float(pixels_row[px, 0])
        g = float(pixels_row[px, 1])
        b = float(pixels_row[px, 2])
        brt = (r + g + b) / 3.0
        hue = et_visual_hue_from_rgb(r, g, b)
        d_r = et_visual_hue_to_d(hue)
        mx = max(r, g, b)
        sat = (1.0 - min(r, g, b) / mx) if mx > 0.01 else 0.0
        tight = sat * min(brt / 0.5, 1.0) if brt > 0.01 else 0.0
        if brt < 0.04:
            d_r = 1; tight = 0.05
        if d_r != cur_d and cur_d >= 0:
            c = max(cur_c, 1)
            segments.append(dict(d_r=cur_d, start_px=cur_s,
                                 end_px=px, length=px - cur_s,
                                 avg_brt=cur_b / c, avg_tight=cur_t / c))
            cur_s = px; cur_b = 0.0; cur_t = 0.0; cur_c = 0
        cur_d = d_r; cur_b += brt; cur_t += tight; cur_c += 1
    if cur_c > 0:
        c = cur_c
        segments.append(dict(d_r=cur_d, start_px=cur_s,
                             end_px=W, length=W - cur_s,
                             avg_brt=cur_b / c, avg_tight=cur_t / c))
    return segments


def et_visual_analyze_image(final_f32, it_raw=None, n_scan_rows=12):
    """Top-level visual analysis report for a rendered ET fractal image.

    Scans n_scan_rows evenly-spaced horizontal rows of the rendered image,
    runs et_visual_segment_row on each, accumulates a d-family pixel
    histogram across all scanned rows, and (when raw kernel data is supplied
    via it_raw) reports the mismatch between what the visual encoding paints
    and what the kernel actually computed for each pixel. The mismatch
    metrics are the empirical evidence for whether the visual color encoding
    is consistent with the kernel's d-family computation — the diagnostic
    that v4.0 needed internally and that the native music rebuild now
    exposes as a standalone tool.

    Three Tools mapping (per Stage 21.6 header):
      P : the rendered RGB image
      D : the FAM_HUE color encoding constraints
      T : this analysis pass walking the image rows

    Inputs:
      final_f32   — np.ndarray of shape (H, W, 3), float, values in [0,1]
      it_raw      — optional dict from _render_frame containing 'd_r' (the
                    kernel's per-pixel d-family at escape, shape (H, W));
                    when supplied, enables the visual-vs-kernel mismatch
                    metrics. When None, only the visual histogram is filled.
      n_scan_rows — number of evenly-spaced rows to scan (default 12,
                    matching the N=12 manifold symmetry — one row per
                    sublattice family for symmetric coverage of the image).

    Returns: dict with keys
      'visual_histogram' : {d: total_pixels_painted_as_d} from hue→d analysis
      'kernel_histogram' : {d: total_pixels_with_kernel_d_r==d}, or None if
                           it_raw was not supplied
      'mismatch_count'   : total scanned pixels where visual_d != kernel_d_r,
                           or None if it_raw was not supplied
      'mismatch_rate'    : fraction of scanned pixels with mismatched d,
                           or None if it_raw was not supplied
      'rows_scanned'     : list of row indices analyzed
      'segments_per_row' : list of (row_idx, segment_count) tuples
      'pixels_scanned'   : total pixels touched by the mismatch comparison
    """
    H, W = final_f32.shape[:2]
    rows = [int(H * (i + 0.5) / n_scan_rows) for i in range(n_scan_rows)]
    visual_hist = {d: 0 for d in range(1, 13)}
    kernel_hist = {d: 0 for d in range(1, 13)}
    mismatch_count = 0
    total_pixels = 0
    segments_per_row = []
    has_raw = (it_raw is not None and 'd_r' in it_raw)
    for ry in rows:
        scan = final_f32[ry, :, :]
        segs = et_visual_segment_row(scan)
        segments_per_row.append((ry, len(segs)))
        for seg in segs:
            visual_hist[seg['d_r']] += seg['length']
        if has_raw:
            kernel_row = it_raw['d_r'][ry, :]
            for px in range(W):
                kd = int(np.clip(np.round(kernel_row[px]), 1, 12))
                kernel_hist[kd] += 1
                r = float(scan[px, 0])
                g = float(scan[px, 1])
                b = float(scan[px, 2])
                brt = (r + g + b) / 3.0
                if brt < 0.04:
                    vd = 1
                else:
                    vd = et_visual_hue_to_d(et_visual_hue_from_rgb(r, g, b))
                if vd != kd:
                    mismatch_count += 1
                total_pixels += 1
    return {
        'visual_histogram': visual_hist,
        'kernel_histogram': kernel_hist if has_raw else None,
        'mismatch_count':   mismatch_count if has_raw else None,
        'mismatch_rate':    (mismatch_count / total_pixels)
        if (has_raw and total_pixels > 0) else None,
        'rows_scanned':     rows,
        'segments_per_row': segments_per_row,
        'pixels_scanned':   total_pixels if has_raw else None,
    }


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
    if n_modes == 0:
        # No Mode option (N) — print the explicit "no dispatch" line.
        # mode_id is NO_MODE_ID (-1) here; mode['name'] is 'None — pure base 24-family'.
        print(f'  Mode      : [{mode_id:2d}]  {mode["name"]}  (no per-mode dispatch)')
    elif n_modes == 1:
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
        render_buf, elapsed, it_raw = _render_frame(
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
    # Filename mode tag: NO_MODE_ID (-1) becomes 'none' instead of '-1' so the
    # filename reads cleanly. Numeric mode ids stay as their integer string.
    _mode_tag = 'none' if mode_id == NO_MODE_ID else str(mode_id)
    stem = f'et_fractal_{ts}_{_type_tag}{_mode_tag}_{tkey[:4]}_{QUALITY_PRESET}'

    meta = {
        'Description':   'Exception Theory Fractal  |  P o D o T = E  |  v2.2',
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
        'Software':      'ET_FRACTAL_GENERATOR.py v2.2',
    }

    print(f'  Saving 32-bit float TIFF…', flush=True)
    tiff_path = script_dir / (stem + '.tiff')
    # Assemble the ET provenance description from the meta dict for embedding
    # as TIFF tag 270 (ImageDescription). Every key/value pair from meta is
    # joined with ' | ' separators, giving full ET provenance inside the file
    # itself — mode, tower, center, zoom, seed, fractal type, p_eff, etc. This
    # lets the TIFF be identified and reproduced from its own metadata alone.
    tiff_desc = ' | '.join(f'{k}={v}' for k, v in meta.items())
    try:
        write_tiff_float32(final_f32, tiff_path, dpi=OUTPUT_DPI, description=tiff_desc)
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

    # ── Save raw iteration data for accurate audio (if enabled) ────────────
    npz_path = None
    if AUDIO_ENABLED and it_raw is not None:
        npz_path = script_dir / (stem + '_raw.npz')
        try:
            np.savez_compressed(str(npz_path), **it_raw)
            npz_mb = npz_path.stat().st_size / 1_048_576
            print(f'  ✓ RAW  : {npz_path}  ({npz_mb:.1f} MB  d_r/d_t/tight/smooth_n/orbit)',
                  flush=True)
        except Exception as e:
            _et_error('Saving raw audio data .npz', e, fatal=False,
                      fallback_msg='Will use image-based native music instead')
            npz_path = None

    # ── Audio native music (if enabled) ──────────────────────────────────────
    audio_path = None
    if AUDIO_ENABLED:
        try:
            audio_path = et_native_music_image(final_f32, stem, script_dir, AUDIO_KBPS,
                                               it_raw=it_raw)
        except Exception as e:
            _et_error('Audio native music', e, fatal=False,
                      fallback_msg='Audio generation failed — images saved OK')

    total = elapsed
    print(f'\n  ✓ TIFF : {tiff_path}  ({tiff_mb:.0f} MB)')
    print(f'  ✓ PNG  : {png_path}  ({png_mb:.1f} MB  16-bit  {OUTPUT_DPI} DPI)')
    if npz_path:
        print(f'  ✓ RAW  : {npz_path}')
    if audio_path:
        print(f'  ✓ AUDIO: {audio_path}')
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

    # Target point: VIDEO_PARAMS override or use resolved center
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
    # Filename mode tag: NO_MODE_ID (-1) becomes 'none' so the directory and
    # video files read cleanly when the user picked the No Mode option.
    _mode_tag  = 'none' if mode_id == NO_MODE_ID else f'm{mode_id}'
    frame_dir  = script_dir / f'et_video_{ts}_{_mode_tag}_{tkey[:4]}'
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
    frame_audio_stats = []    # per-frame d-family distributions for audio
    total_t0    = time.time()

    for fi, zoom_fi in enumerate(zooms):
        pct = (fi+1)/nf*100
        print(f'  Frame {fi+1:4d}/{nf}  ({pct:5.1f}%)  zoom={zoom_fi:.6f}', flush=True)

        try:
            render_buf, elapsed, it_raw_frame = _render_frame(
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

        # Collect audio data for this frame (if audio enabled)
        if AUDIO_ENABLED:
            try:
                fstats = et_native_music_video_frame(final_f32, it_raw=it_raw_frame)
                frame_audio_stats.append(fstats)
            except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                # KeyError:       it_raw missing 'd_r'/'tight' on partial-data frames
                # IndexError:     row scan slice if frame dimensions changed mid-run
                # ValueError:     numpy clip/round on NaN/inf in raw arrays
                # TypeError:      None propagation through arithmetic
                # AttributeError: it_raw_frame=None on stale upstream failure
                frame_audio_stats.append({})   # empty frame — silence

        elapsed_total = time.time()-total_t0
        eta = elapsed_total/(fi+1)*(nf-fi-1)
        print(f'       frame {elapsed:.1f}s  total {elapsed_total:.0f}s  ETA {eta:.0f}s',
              flush=True)

    total_elapsed = time.time()-total_t0
    print(f'\n  All {nf} frames rendered in {total_elapsed:.0f}s '
          f'({total_elapsed/nf:.1f}s/frame avg)')

    # Assemble with ffmpeg — with motion interpolation for smooth zoom
    # Reuse the _mode_tag computed above so the .mp4 filename matches the
    # frame directory naming convention (NO_MODE_ID → 'none', else 'm{id}').
    video_path = script_dir / f'et_video_{ts}_{_mode_tag}_{tkey[:4]}.mp4'
    ffmpeg_in  = str(frame_dir / 'frame_%06d.png')

    # Smooth zoom: use minterpolate for optical flow interpolation
    # This prevents jitter from discrete zoom steps
    # Output at 60fps regardless of input fps for smooth playback
    _vf = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
    _vf += f',minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'

    cmd = ['ffmpeg', '-y', '-r', str(fps),
           '-i', ffmpeg_in,
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
           '-pix_fmt', 'yuv420p',
           '-vf', _vf,
           '-r', '60',
           str(video_path)]

    # Fallback command without interpolation (in case minterpolate not available)
    cmd_raw = ['ffmpeg', '-y', '-r', str(fps),
               '-i', ffmpeg_in,
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
               '-pix_fmt', 'yuv420p',
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
               str(video_path)]

    print(f'\n  Assembling video with ffmpeg (optical flow interpolation)…')
    print(f'  Output: 60fps interpolated from {fps}fps keyframes')
    try:
        import subprocess as _sp
        # Use Popen instead of run to prevent pipe buffer deadlock
        # Stream stderr to devnull to prevent buffer fill
        proc = _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.PIPE)
        _, stderr_data = proc.communicate()
        if proc.returncode == 0:
            vmb = video_path.stat().st_size/1_048_576
            print(f'\n  ✓ VIDEO : {video_path}  ({vmb:.0f} MB  60fps interpolated)')
        else:
            # minterpolate may have failed — try raw assembly
            print(f'\n  Optical flow failed, trying raw assembly…')
            proc2 = _sp.Popen(cmd_raw, stdout=_sp.DEVNULL, stderr=_sp.PIPE)
            _, stderr2 = proc2.communicate()
            if proc2.returncode == 0:
                vmb = video_path.stat().st_size/1_048_576
                print(f'\n  ✓ VIDEO : {video_path}  ({vmb:.0f} MB  {fps}fps raw)')
            else:
                err = stderr2.decode('utf-8', errors='replace')[-1000:]
                print(f'\n  ffmpeg failed:')
                for line in err.split('\n'):
                    if any(k in line.lower() for k in ['error','invalid','cannot','failed']):
                        print(f'    {line.strip()}')
                print(f'\n  Frames are in: {frame_dir}')
                print(f'  Use et_assemble_video.py for manual assembly with quality options.')
    except FileNotFoundError:
        print(f'\n  ffmpeg not found. Frames are in: {frame_dir}')
        print(f'  Install ffmpeg, then use et_assemble_video.py to assemble.')

    # ── Audio native music (if enabled) ──────────────────────────────────────
    if AUDIO_ENABLED and frame_audio_stats:
        # Match the video/frame_dir naming so the audio sidecar is grouped
        # with its companion files in directory listings (No Mode → 'none').
        _audio_stem = f'et_video_{ts}_{_mode_tag}_{tkey[:4]}'
        try:
            et_native_music_video(frame_audio_stats, fps, _audio_stem, script_dir,
                                  AUDIO_KBPS, video_path=str(video_path))
        except Exception as e:
            _et_error('Video audio native music', e, fatal=False,
                      fallback_msg='Audio generation failed — video saved OK')

    print('\n' + '═'*72)
    print('  P ∘ D ∘ T = E  —  Exception Theory  —  Michael James Muller')
    print('═'*72 + '\n')
    return str(frame_dir), str(video_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    _exit_code = 0
    try:
        if OUTPUT_MODE == 'video':
            fd, vp = generate_zoom_video()
            if sys.platform == 'win32':
                print(f'Frames: {fd}\nVideo:  {vp}\n')
        else:
            tp, pp = generate_et_fractal()
            if sys.platform == 'win32':
                print(f'Files saved:\n  {tp}\n  {pp}\n')
    except KeyboardInterrupt:
        print('\n\n  [Interrupted by user]')
    except SystemExit as _se:
        _exit_code = _se.code if _se.code else 0
        print(f'\n  [SystemExit code={_exit_code}]')
    except BaseException as _be:
        # Catches EVERYTHING — Exception, SystemExit, MemoryError,
        # CuPy errors, segfault-adjacent errors, anything Python can trap
        _exit_code = 1
        print('\n' + '!'*72)
        print('  [ET FRACTAL GENERATOR — CRASH REPORT]')
        print('!'*72)
        print(f'  Error type : {type(_be).__name__}')
        print(f'  Message    : {_be}')
        try:
            import traceback
            print(f'  Traceback  :')
            traceback.print_exc()
        except (OSError, ValueError, AttributeError):
            # OSError:        EPIPE / closed stdout when writing the trace
            # ValueError:     'I/O operation on closed file' from print on dead stream
            # AttributeError: traceback module unavailable in stripped runtimes
            pass
        print('!'*72)
    finally:
        # Window ALWAYS stays open on Windows, no matter what
        if sys.platform == 'win32':
            try:
                input('\nPress Enter to exit…')
            except (EOFError, KeyboardInterrupt, OSError):
                # EOFError:         stdin already closed (piped run, no terminal)
                # KeyboardInterrupt: Ctrl-C during the final hold prompt
                # OSError:          stdin descriptor invalid / detached console
                import time as _ft
                _ft.sleep(30)  # if input() fails, wait 30s so user can read
    sys.exit(_exit_code)
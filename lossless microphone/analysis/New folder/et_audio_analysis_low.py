#!/usr/bin/env python3
"""
Exception Theory — Ultra-Resolution Audio Analysis Engine
=========================================================
Author: Michael James Muller (Aevum Defluo) — Exception Theory
Engine: ET Lattice-Aware Multi-Resolution Analysis

From: P ∘ D ∘ T = E
"For every exception there is an exception, except the exception."

This script performs the highest-quality analysis possible of a lossless
WAV file generated via the ET bijection. It uses:

  1. Full-bandwidth linear STFT (no perceptual weighting, no frequency caps)
  2. Constant-Q Transform (logarithmically spaced = semitone-lattice-native)
  3. Synchrosqueezed Continuous Wavelet Transform (highest resolution)
  4. Ultrasonic band isolation and analysis (20kHz–Nyquist)
  5. ET Lattice projection at multiple LCM tower levels (N=12,60,2520,27720)
  6. Cross-channel (stereo) phase and correlation analysis
  7. Harmonic series detection and fundamental frequency tracking
  8. Spectral statistics (centroid, bandwidth, rolloff, flatness, contrast)
  9. Instantaneous frequency analysis via analytic signal
 10. Dynamic range and amplitude statistics per channel

All mathematics ET-native. No Mel weighting. No frequency caps.
No resampling. Native sample rate preserved throughout.

ET Constants:
  N_base = 12 (3 primitives × 4 logic states)
  V_base = 1/12
  s = 2^(1/12) (semitone generator)
  f_ref = 440 Hz (A440 concert pitch — lattice reference)
  K = 2/3 (Koide ratio — triadic stability threshold)
"""

import os
import sys
import gc
import json
import time
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib

# Use Agg backend if no display available (headless server/container)
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import librosa
import soundfile as sf
from math import gcd, log2, log10, pi, sqrt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows (default cp1252 can't encode ET symbols: ∘ Π ε τ μ ℓ →)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================================
# ET CONSTANTS — Forward from P ∘ D ∘ T = E
# ============================================================================

N_BASE = 12  # Manifold symmetry: 3 × 4
V_BASE = 1.0 / N_BASE  # Base variance = 1/12
KOIDE = 2.0 / 3.0  # Koide ratio (triadic stability)
SEMITONE = 2.0 ** (1.0 / N_BASE)  # Primitive lattice generator
F_REF = 440.0  # Concert pitch A4 (lattice reference)
PHI = (1.0 + sqrt(5.0)) / 2.0  # Golden ratio (manifold resonant constant)

# LCM Tower — THE COMPLETE ET RESOLUTION TOWER
# Every level where a new harmonic family activates, where multiple complex
# families first coexist, or where the Doubling Law (τ = 6·2^ℓ) is satisfied.
#
# Two categorically distinct structures tracked simultaneously:
#   24 Harmonic Families: 12 per axis (6 Simple + 6 Complex), d ∈ {1..12}
#     Simple (d|12): {1,2,3,4,6,12} — native at 12ET
#     Complex (d∤12): {5,7,8,9,10,11} — shadow forces at extended nET
#   τ(N) Sublattice Families: divisors of N, grows with resolution
#
# Doubling Law (Theorem thm:doubling):
#   τ(N_ℓ) = 6·2^ℓ at canonical landmarks ℓ=0,1,2,3,4,...
#   Each canonical step exactly doubles the sublattice family count.
#
# Force Quadrant Grid activation thresholds:
#   SR+SI (Standard Model): 12ET
#   CR+SI (CKM mixing): 24ET
#   SR+CI (PMNS mixing): 36ET
#   CR+CI (E₈/M-theory): 420ET
#
# Palindromic Mirror Law: d ↔ 12−d
#   d=1↔d=11, d=2↔d=10, d=3↔d=9, d=4↔d=8, d=5↔d=7, d=6↔d=6

LCM_TOWER = {
    # ── Individual harmonic family activations ──────────────────────────
    12: "ℓ=0 Base ET | d={1,2,3,4,6,12} | τ=6 | 12 harmonic families (6/axis)",
    24: "d=8 octet activates (12×2) | τ=8 | 14 harmonic families | CR+SI threshold",
    36: "d=9 nonic activates (12×3) | τ=9 | 16 harmonic families | SR+CI threshold",
    60: "ℓ=1 LCM(1..5) | d=5,10 quintic+decic activate | τ=12 | 20 harmonic families",
    84: "d=7 septic activates (12×7) | τ=12 | 22 harmonic families",
    132: "d=11 undecimal activates (12×11) | τ=12 | ALL 24 activated (not co-present)",
    # ── Simultaneous co-activation milestones ──────────────────────────
    120: "d=5,8,10 simultaneously native | LCM(24,60) | τ=16",
    168: "d=7,8 simultaneously native | LCM(24,84) | τ=16",
    180: "d=5,9,10 simultaneously native | LCM(36,60) | τ=18",
    252: "d=7,9 simultaneously native | LCM(36,84) | τ=18",
    360: "d=5,8,9,10 simultaneously native | LCM(24,36,60) | τ=24",
    # ── LCM landmarks (Doubling Law satisfied) ─────────────────────────
    420: "ℓ=2 LCM(1..7) | d=5,7 BOTH native | BIOLOGICAL THRESHOLD | τ=24 | CR+CI threshold",
    840: "LCM(1..8) | d=5,7,8 simultaneously native | τ=32",
    2520: "ℓ=3 LCM(1..9) | ALL d=1..9 native | UNIVERSAL HARMONIC | τ=48",
    27720: "ℓ=4 LCM(1..11) | ALL d=1..12 native | COMPLETE ET LATTICE | τ=96",
}

# Sublattice family descriptors for N=12
SUBLATTICE_NAMES_12 = {
    1: "Trivial (Octave)",
    2: "Quadratic (Tritone)",
    3: "Cubic (Strong Force)",
    4: "Quartic (Hypercubic)",
    6: "Hexadic (Composite)",
    12: "Full Resolution (EM/Ambient)",
}

# Output directory — always the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


# ============================================================================
# ET CORE MATHEMATICS — The Sempaevum Bijection
# ============================================================================

def et_project(r, N=12):
    """
    Project positive ratio r onto the N-ET lattice.
    Π_N(r) = (k, d, ε)

    This is the Sempaevum bijection — lossless by algebraic identity:
    r = 2^((k + εN/1200)/N) exactly recovers r at every finite N.

    Parameters:
        r: positive real ratio (f / f_ref)
        N: lattice resolution (default 12, the base ET)

    Returns:
        dict with k (lattice coordinate), d (sublattice family),
        g (shared factor), epsilon (error in cents), r_ET (lattice value)
    """
    if r <= 0:
        return {'k': 0, 'd': N, 'g': 1, 'epsilon': 0.0, 'r_ET': 1.0,
                'cents_exact': 0.0}

    cents_exact = 1200.0 * log2(r)
    lattice_pos = N * log2(r)  # continuous position on N-lattice
    k = round(lattice_pos)  # T acts as rounding operator
    g = gcd(abs(k) if k != 0 else N, N)
    d = N // g  # sublattice family
    epsilon = (lattice_pos - k) * (1200.0 / N)  # error in cents
    r_ET = 2.0 ** (k / N)  # lattice-projected value

    return {
        'k': k,
        'd': d,
        'g': g,
        'epsilon': epsilon,
        'r_ET': r_ET,
        'cents_exact': cents_exact,
    }


def et_project_freq(f, N=12, f_ref=F_REF):
    """Project a frequency onto the ET lattice."""
    r = f / f_ref
    result = et_project(r, N)
    result['freq'] = f
    result['f_ref'] = f_ref
    result['ratio'] = r
    result['f_ET'] = result['r_ET'] * f_ref  # lattice-projected frequency
    return result


def sublattice_name(d, N=12):
    """Get human-readable sublattice family name."""
    if N == 12:
        return SUBLATTICE_NAMES_12.get(d, f"d={d}")
    return f"d={d} (N={N})"


def _khz_formatter(x, pos):
    """FuncFormatter callback: display frequency in kHz when >= 1000 Hz.
    pos = tick position index (int during rendering, None if called manually)."""
    if pos is not None and x >= 1000:
        return f'{x / 1000:.1f}k'
    elif x >= 1000:
        return f'{x / 1000:.1f} kHz'
    return f'{int(x)}' if pos is not None else f'{x:.0f} Hz'


# Reusable axis formatters for frequency displays
KHZ_FORMATTER = FuncFormatter(_khz_formatter)


def _apply_et_axis_style(ax):
    """Apply consistent ET dark theme to an axis."""
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#00FFAA')
    # Apply clean fixed-point notation (no scientific notation on axes)
    for axis in [ax.xaxis, ax.yaxis]:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _make_et_figure(n_rows, n_cols=1, height_per_row=5, width=24, use_gridspec=False):
    """
    Create a figure with ET styling. Uses gridspec for complex layouts.

    Args:
        n_rows: Number of subplot rows
        n_cols: Number of subplot columns
        height_per_row: Height per row in inches
        width: Total figure width
        use_gridspec: If True, returns (fig, gridspec) for custom sub-layouts
    """
    fig = plt.figure(figsize=(width, height_per_row * n_rows), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    if use_gridspec:
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        return fig, gs
    else:
        axes = fig.subplots(n_rows, n_cols)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_1d(axes)
        # Convert to list for correct type inference (PyCharm sees .flat as ndarray scalars)
        axes_list = axes if isinstance(axes, list) else np.atleast_1d(axes).ravel().tolist()
        for ax_item in axes_list:
            _apply_et_axis_style(ax_item)
        return fig, axes


# ============================================================================
# AUDIO LOADING — Native sample rate, no resampling, full precision
# ============================================================================

def load_wav_native(path):
    """
    Load WAV at native sample rate with full precision.
    CRITICAL: No resampling. No frequency cap. No perceptual weighting.

    Primary: soundfile (supports float64, all bit depths).
    Fallback: scipy.io.wavfile (for edge cases where soundfile fails).
    """
    try:
        data, sr = sf.read(path, dtype='float64')
    except Exception as sf_err:
        print(f"  soundfile failed ({sf_err}), falling back to scipy.io.wavfile...")
        sr, raw_data = wavfile.read(path)
        # Normalize integer formats to float64 [-1.0, 1.0]
        if raw_data.dtype == np.int16:
            data = raw_data.astype(np.float64) / 32768.0
        elif raw_data.dtype == np.int32:
            data = raw_data.astype(np.float64) / 2147483648.0
        elif raw_data.dtype == np.float32:
            data = raw_data.astype(np.float64)
        else:
            data = raw_data.astype(np.float64)

    info = {
        'sample_rate': sr,
        'nyquist': sr / 2.0,
        'duration': len(data) / sr,
        'num_samples': len(data),
        'channels': data.shape[1] if data.ndim > 1 else 1,
        'dtype': str(data.dtype),
        'file_size_bytes': os.path.getsize(path),
    }

    if data.ndim == 1:
        channels = {'mono': data}
    else:
        channels = {f'ch{i}': data[:, i] for i in range(data.shape[1])}
        # Also compute mid/side decomposition
        if data.shape[1] == 2:
            channels['mid'] = (data[:, 0] + data[:, 1]) / 2.0
            channels['side'] = (data[:, 0] - data[:, 1]) / 2.0

    return channels, info


# ============================================================================
# SIGNAL STATISTICS
# ============================================================================

def compute_signal_stats(channels, info):
    """Comprehensive amplitude and dynamic range statistics per channel."""
    stats = {}
    for name, sig in channels.items():
        peak = np.max(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        dc_offset = np.mean(sig)

        # Dynamic range (dB)
        peak_db = 20.0 * log10(peak) if peak > 0 else -np.inf
        rms_db = 20.0 * log10(rms) if rms > 0 else -np.inf
        crest_factor_db = peak_db - rms_db

        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        zc_rate = zero_crossings / info['duration']

        # Amplitude distribution
        percentiles = np.percentile(np.abs(sig), [25, 50, 75, 90, 95, 99])

        stats[name] = {
            'peak': peak,
            'peak_dBFS': peak_db,
            'rms': rms,
            'rms_dBFS': rms_db,
            'crest_factor_dB': crest_factor_db,
            'dc_offset': dc_offset,
            'zero_crossings': zero_crossings,
            'zc_rate_hz': zc_rate,
            'percentiles_abs': {
                'p25': percentiles[0], 'p50': percentiles[1],
                'p75': percentiles[2], 'p90': percentiles[3],
                'p95': percentiles[4], 'p99': percentiles[5],
            },
            'min': np.min(sig),
            'max': np.max(sig),
            'std': np.std(sig),
        }
    return stats


# ============================================================================
# ANALYSIS 1: Full-Bandwidth Linear STFT
# ============================================================================

def compute_full_bandwidth_stft(sig, sr, n_fft=16384, hop_length=512):
    """
    Full-bandwidth linear STFT. NO Mel weighting. NO frequency cap.
    Uses massive FFT for maximum frequency resolution.

    Frequency resolution = sr / n_fft
    For 48kHz, n_fft=16384 → resolution = 2.93 Hz per bin
    For 48kHz, n_fft=32768 → resolution = 1.46 Hz per bin

    This covers the FULL bandwidth: 0 Hz to Nyquist (24kHz for 48kHz sr).
    """
    f, t, Zxx = signal.stft(sig, fs=sr, nperseg=n_fft,
                            noverlap=n_fft - hop_length,
                            window='hann', return_onesided=True)

    magnitude = np.abs(Zxx)
    power = magnitude ** 2
    power_db = 10.0 * np.log10(power + 1e-300)  # avoid log(0)

    return {
        'frequencies': f,
        'times': t,
        'magnitude': magnitude,
        'power_db': power_db,
        'n_fft': n_fft,
        'freq_resolution': sr / n_fft,
        'time_resolution': hop_length / sr,
        'num_freq_bins': len(f),
        'freq_range': (f[0], f[-1]),
    }


# ============================================================================
# ANALYSIS 2: Constant-Q Transform (Semitone-Lattice-Native)
# ============================================================================

def compute_cqt(sig, sr, fmin=20.0, n_bins_per_octave=36):
    """
    Constant-Q Transform — logarithmically spaced frequency bins.
    This is NATIVELY aligned with the ET semitone lattice.

    n_bins_per_octave=36 gives 3 bins per semitone (sub-semitone resolution).
    fmin=20Hz covers human hearing floor.

    The CQT extends to Nyquist automatically.
    """
    # Compute number of octaves from fmin to Nyquist
    nyquist = sr / 2.0
    n_octaves = log2(nyquist / fmin)
    n_bins = int(n_octaves * n_bins_per_octave)

    C = librosa.cqt(y=sig, sr=sr, fmin=fmin, n_bins=n_bins,
                    bins_per_octave=n_bins_per_octave,
                    hop_length=512)

    # CQT frequencies
    freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=fmin,
                                    bins_per_octave=n_bins_per_octave)

    magnitude = np.abs(C)
    power_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Map each CQT frequency to ET lattice
    et_mapping = [et_project_freq(f) for f in freqs]

    return {
        'frequencies': freqs,
        'magnitude': magnitude,
        'power_db': power_db,
        'et_mapping': et_mapping,
        'n_bins': n_bins,
        'n_bins_per_octave': n_bins_per_octave,
        'fmin': fmin,
        'fmax': freqs[-1],
    }


# ============================================================================
# ANALYSIS 3: Synchrosqueezed CWT — Highest Resolution
# ============================================================================

def compute_ssq_cwt(sig, sr):
    """
    Synchrosqueezed Continuous Wavelet Transform.
    This gives the SHARPEST possible time-frequency resolution —
    beating STFT and standard CWT by focusing energy via reassignment.

    Processes the FULL signal. Memory requirement is estimated and reported.
    SSQ-CWT is O(N × num_scales) in memory.
    """
    from ssqueezepy import ssq_cwt

    n_samples = len(sig)
    # Estimate memory: ~286 scales × n_samples × 16 bytes (complex128) × 2 (Tx + Wx)
    est_scales = 286  # typical for Morlet mu=13.4
    est_memory_gb = (est_scales * n_samples * 16 * 2) / (1024 ** 3)
    print(f"    SSQ-CWT memory estimate: {est_memory_gb:.2f} GB for {n_samples:,} samples")

    # Perform SSQ-CWT on the full signal
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(
        sig,
        wavelet=('morlet', {'mu': 13.4}),  # high mu for better freq resolution
        fs=sr,
    )

    return {
        'Tx': Tx,
        'Wx': Wx,
        'ssq_freqs': ssq_freqs,
        'scales': scales,
        'analysis_duration': n_samples / sr,
        'sr': sr,
        'num_samples': n_samples,
        'num_scales': Tx.shape[0],
        'memory_gb': est_memory_gb,
    }


# ============================================================================
# ANALYSIS 4: Ultrasonic Band Isolation (20kHz–Nyquist)
# ============================================================================

def analyze_ultrasonic_band(sig, sr, stft_result):
    """
    Isolate and analyze the ultrasonic band (20kHz to Nyquist).
    This is where standard tools completely fail.

    For 48kHz audio, the ultrasonic band is 20kHz–24kHz.
    Regular Mel spectrograms cap at ~8-11kHz. Even linear spectrograms
    in most tools clip at 20kHz. We analyze the full range.
    """
    nyquist = sr / 2.0
    freqs = stft_result['frequencies']
    power_db = stft_result['power_db']
    magnitude = stft_result['magnitude']

    # Ultrasonic mask: 20kHz to Nyquist
    ultra_mask = freqs >= 20000.0
    sub_ultra_mask = (freqs >= 15000.0) & (freqs < 20000.0)

    # High-audible band for comparison: 10-20kHz
    high_audible_mask = (freqs >= 10000.0) & (freqs < 20000.0)

    # Full audible band: 20Hz-20kHz
    audible_mask = (freqs >= 20.0) & (freqs < 20000.0)

    # Band energies over time
    ultra_energy = np.mean(magnitude[ultra_mask, :] ** 2, axis=0) if np.any(ultra_mask) else np.zeros(
        magnitude.shape[1])
    audible_energy = np.mean(magnitude[audible_mask, :] ** 2, axis=0)
    high_audible_energy = np.mean(magnitude[high_audible_mask, :] ** 2, axis=0)
    sub_ultra_energy = np.mean(magnitude[sub_ultra_mask, :] ** 2, axis=0) if np.any(sub_ultra_mask) else np.zeros(
        magnitude.shape[1])

    # Time-domain ultrasonic RMS via bandpass filter on the raw signal
    if sr > 40000:
        sos_ultra = signal.butter(8, 20000.0, btype='highpass', fs=sr, output='sos')
        sig_ultra = signal.sosfilt(sos_ultra, sig)
        ultra_rms_time_domain = np.sqrt(np.mean(sig_ultra ** 2))
    else:
        ultra_rms_time_domain = 0.0

    # Ultrasonic spectral detail
    ultra_freqs = freqs[ultra_mask]
    ultra_power = power_db[ultra_mask, :]
    ultra_magnitude = magnitude[ultra_mask, :]

    # Peak frequencies in ultrasonic band per time frame
    if np.any(ultra_mask) and ultra_magnitude.shape[0] > 0:
        ultra_peak_idx = np.argmax(ultra_magnitude, axis=0)
        ultra_peak_freqs = ultra_freqs[ultra_peak_idx]
        ultra_peak_power = np.max(ultra_magnitude, axis=0)
    else:
        ultra_peak_freqs = np.zeros(magnitude.shape[1])
        ultra_peak_power = np.zeros(magnitude.shape[1])

    # Ultrasonic vs audible energy ratio
    total_ultra_energy = np.sum(magnitude[ultra_mask, :] ** 2) if np.any(ultra_mask) else 0
    total_audible_energy = np.sum(magnitude[audible_mask, :] ** 2)
    ultra_ratio = total_ultra_energy / (total_audible_energy + 1e-300)

    # ET lattice mapping of ultrasonic peaks
    unique_ultra_peaks = np.unique(np.round(ultra_peak_freqs, 1))
    unique_ultra_peaks = unique_ultra_peaks[unique_ultra_peaks > 0]
    ultra_et_map = [et_project_freq(f) for f in unique_ultra_peaks]  # ALL peaks, no cap

    return {
        'ultra_freqs': ultra_freqs,
        'ultra_power_db': ultra_power,
        'ultra_magnitude': ultra_magnitude,
        'ultra_energy_over_time': ultra_energy,
        'audible_energy_over_time': audible_energy,
        'high_audible_energy_over_time': high_audible_energy,
        'sub_ultra_energy_over_time': sub_ultra_energy,
        'ultra_peak_freqs': ultra_peak_freqs,
        'ultra_peak_power': ultra_peak_power,
        'ultra_to_audible_ratio': ultra_ratio,
        'ultra_to_audible_ratio_dB': 10.0 * log10(ultra_ratio + 1e-300),
        'total_ultra_energy': total_ultra_energy,
        'total_audible_energy': total_audible_energy,
        'ultra_rms_time_domain': ultra_rms_time_domain,
        'ultra_et_map': ultra_et_map,
        'num_ultra_bins': np.sum(ultra_mask),
        'ultra_band_hz': (20000.0, nyquist),
    }


# ============================================================================
# ANALYSIS 5: Spectral Statistics Over Time
# ============================================================================

def compute_spectral_stats(sig, sr):
    """
    Spectral statistics over time — all computed from the raw signal,
    no perceptual weighting.
    """
    hop = 512
    n_fft = 4096

    # Spectral centroid (center of mass of spectrum)
    centroid = librosa.feature.spectral_centroid(y=sig, sr=sr, n_fft=n_fft,
                                                 hop_length=hop)[0]

    # Spectral bandwidth (variance around centroid)
    bandwidth = librosa.feature.spectral_bandwidth(y=sig, sr=sr, n_fft=n_fft,
                                                   hop_length=hop)[0]

    # Spectral rolloff (frequency below which X% of energy is concentrated)
    rolloff_85 = librosa.feature.spectral_rolloff(y=sig, sr=sr, n_fft=n_fft,
                                                  hop_length=hop,
                                                  roll_percent=0.85)[0]
    rolloff_95 = librosa.feature.spectral_rolloff(y=sig, sr=sr, n_fft=n_fft,
                                                  hop_length=hop,
                                                  roll_percent=0.95)[0]
    rolloff_99 = librosa.feature.spectral_rolloff(y=sig, sr=sr, n_fft=n_fft,
                                                  hop_length=hop,
                                                  roll_percent=0.99)[0]

    # Spectral flatness (Wiener entropy — how noise-like the spectrum is)
    flatness = librosa.feature.spectral_flatness(y=sig, n_fft=n_fft,
                                                 hop_length=hop)[0]

    # Spectral contrast (valley-to-peak energy ratio per sub-band)
    contrast = librosa.feature.spectral_contrast(y=sig, sr=sr, n_fft=n_fft,
                                                 hop_length=hop)

    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr,
                                   hop_length=hop)

    return {
        'times': times,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff_85': rolloff_85,
        'rolloff_95': rolloff_95,
        'rolloff_99': rolloff_99,
        'flatness': flatness,
        'contrast': contrast,
    }


# ============================================================================
# ANALYSIS 6: Cross-Channel Analysis (Stereo)
# ============================================================================

def analyze_stereo(channels, sr):
    """
    Cross-channel analysis for stereo files.
    Correlation, phase difference, mid/side energy ratio.
    """
    if 'ch0' not in channels or 'ch1' not in channels:
        return None

    left = channels['ch0']
    right = channels['ch1']
    mid = channels['mid']
    side = channels['side']

    # Short-time correlation
    frame_len = 2048
    hop = 512
    n_frames = (len(left) - frame_len) // hop

    correlation = np.zeros(n_frames)
    mid_energy = np.zeros(n_frames)
    side_energy = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        l_frame = left[start:end]
        r_frame = right[start:end]
        m_frame = mid[start:end]
        s_frame = side[start:end]

        # Pearson correlation
        l_norm = l_frame - np.mean(l_frame)
        r_norm = r_frame - np.mean(r_frame)
        denom = np.sqrt(np.sum(l_norm ** 2) * np.sum(r_norm ** 2))
        if denom > 0:
            correlation[i] = np.sum(l_norm * r_norm) / denom

        mid_energy[i] = np.mean(m_frame ** 2)
        side_energy[i] = np.mean(s_frame ** 2)

    times = np.arange(n_frames) * hop / sr

    # Frequency-domain phase difference
    n_fft = 8192
    L = np.fft.rfft(left[:n_fft * (len(left) // n_fft)].reshape(-1, n_fft), axis=1)
    R = np.fft.rfft(right[:n_fft * (len(right) // n_fft)].reshape(-1, n_fft), axis=1)

    # Cross-spectral phase
    cross_spectrum = L * np.conj(R)
    phase_diff = np.angle(cross_spectrum)
    mean_phase_diff = np.mean(phase_diff, axis=0)

    phase_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    return {
        'times': times,
        'correlation': correlation,
        'mean_correlation': np.mean(correlation),
        'mid_energy': mid_energy,
        'side_energy': side_energy,
        'mid_side_ratio': np.mean(mid_energy) / (np.mean(side_energy) + 1e-300),
        'phase_freqs': phase_freqs,
        'mean_phase_diff': mean_phase_diff,
    }


# ============================================================================
# ANALYSIS 7: Harmonic Analysis & Fundamental Frequency Tracking
# ============================================================================

def analyze_harmonics(sig, sr):
    """
    Fundamental frequency (F0) tracking and harmonic series detection.
    Uses the pYIN algorithm for robust F0 estimation.
    """
    # F0 tracking via pYIN
    f0, voiced_flag, voiced_prob = librosa.pyin(
        sig, fmin=20, fmax=sr / 2,
        sr=sr, hop_length=512,
        fill_na=0.0,
    )

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

    # ET lattice mapping of detected F0 values
    f0_valid = f0[f0 > 0]
    f0_et_map = [et_project_freq(f) for f in f0_valid]  # ALL values, no cap

    # Dominant frequency histogram
    if len(f0_valid) > 0:
        # Dynamic bin count: Freedman-Diaconis rule or sqrt(n), whichever gives more
        n_f0 = len(f0_valid)
        iqr = np.percentile(f0_valid, 75) - np.percentile(f0_valid, 25)
        if iqr > 0 and n_f0 > 1:
            fd_bins = int(np.ceil((np.max(f0_valid) - np.min(f0_valid)) / (2 * iqr * n_f0 ** (-1 / 3))))
            sqrt_bins = int(np.ceil(np.sqrt(n_f0)))
            n_bins = max(fd_bins, sqrt_bins, 10)
        else:
            n_bins = max(int(np.ceil(np.sqrt(n_f0))), 10)
        f0_hist, f0_bins = np.histogram(f0_valid, bins=n_bins)
    else:
        f0_hist, f0_bins = np.array([]), np.array([])

    return {
        'f0': f0,
        'times': times,
        'voiced_flag': voiced_flag,
        'voiced_prob': voiced_prob,
        'f0_valid': f0_valid,
        'f0_et_map': f0_et_map,
        'f0_hist': f0_hist,
        'f0_bins': f0_bins,
        'f0_mean': np.mean(f0_valid) if len(f0_valid) > 0 else 0,
        'f0_std': np.std(f0_valid) if len(f0_valid) > 0 else 0,
    }


# ============================================================================
# ANALYSIS 8: ET Lattice Frequency Distribution
# ============================================================================

def compute_et_lattice_distribution(stft_result, sr, N_values=None):
    """
    Map the entire frequency content onto the ET lattice at multiple
    LCM tower levels. This reveals the sublattice family distribution
    of the audio's spectral energy.

    For each N in the LCM tower, we project every STFT frequency bin
    and weight by its average power. This gives an energy-weighted
    sublattice family histogram.
    """
    if N_values is None:
        N_values = sorted(LCM_TOWER.keys())

    freqs = stft_result['frequencies']
    magnitude = stft_result['magnitude']
    nyquist = sr / 2.0  # Use sr for frequency range validation

    # Average power per frequency bin across all time frames
    avg_power = np.mean(magnitude ** 2, axis=1)

    results = {}

    for N in N_values:
        d_histogram = defaultdict(float)
        k_histogram = defaultdict(float)
        epsilon_values = []

        for i, f in enumerate(freqs):
            if f <= 0:
                continue
            proj = et_project_freq(f, N=N)
            d = proj['d']
            k = proj['k']
            eps = proj['epsilon']
            power = avg_power[i]

            d_histogram[d] += power
            k_mod = k % N
            k_histogram[k_mod] += power
            epsilon_values.append((f, eps, power, d, k))

        # Sort d_histogram by d
        d_sorted = dict(sorted(d_histogram.items()))

        # Weighted mean epsilon (how well the content aligns to lattice)
        total_power = sum(p for _, _, p, _, _ in epsilon_values)
        if total_power > 0:
            weighted_abs_eps = sum(abs(e) * p for _, e, p, _, _ in epsilon_values) / total_power
        else:
            weighted_abs_eps = 0

        results[N] = {
            'd_histogram': d_sorted,
            'k_histogram': dict(sorted(k_histogram.items())),
            'weighted_mean_abs_epsilon': weighted_abs_eps,
            'epsilon_times_N': weighted_abs_eps * N,  # Should be ~300 for uniform (Descriptor Gap convergence)
            'nyquist_hz': nyquist,
            'epsilon_values': epsilon_values,  # ALL values, no cap
            'N': N,
            'description': LCM_TOWER.get(N, f"N={N}"),
        }

    return results


# ============================================================================
# ANALYSIS 9: Instantaneous Frequency via Analytic Signal
# ============================================================================

def compute_instantaneous_frequency(sig, sr):
    """
    Compute instantaneous frequency via the Hilbert transform (analytic signal).
    This reveals the moment-by-moment frequency content without windowing artifacts.
    """
    analytic = signal.hilbert(sig)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * pi) * sr

    # Clip to valid range
    inst_freq = np.clip(inst_freq, 0, sr / 2)

    times = np.arange(len(inst_freq)) / sr

    return {
        'inst_freq': inst_freq,
        'times': times,
        'envelope': np.abs(analytic),
        'phase': inst_phase,
        'mean_inst_freq': np.mean(inst_freq),
        'std_inst_freq': np.std(inst_freq),
    }


# ============================================================================
# VISUALIZATION — Comprehensive Multi-Panel Output
# ============================================================================

def plot_full_bandwidth_stft(stft_result, info, filename):
    """Plot 1: Full-bandwidth linear STFT spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(24, 14), dpi=150)
    fig.suptitle('ET Audio Analysis — Full-Bandwidth Linear STFT\n'
                 f'No Mel Weighting | No Frequency Cap | Resolution: '
                 f'{stft_result["freq_resolution"]:.2f} Hz × '
                 f'{stft_result["time_resolution"] * 1000:.1f} ms | '
                 f'n_fft={stft_result["n_fft"]}',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    # Full bandwidth
    im0 = axes[0].pcolormesh(stft_result['times'], stft_result['frequencies'],
                             stft_result['power_db'],
                             shading='gouraud', cmap='inferno',
                             vmin=np.max(stft_result['power_db']) - 120)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_title(f'Full Bandwidth: 0–{info["nyquist"] / 1000:.0f} kHz (LINEAR scale)', fontsize=12)
    axes[0].axhline(y=20000, color='#00FF00', linestyle='--', alpha=0.7, label='20 kHz (hearing limit)')
    axes[0].legend(loc='upper right', fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc')
    fig.colorbar(im0, ax=axes[0], label='Power (dB)', pad=0.01)

    # Ultrasonic zoom: 15–24 kHz
    ultra_mask = stft_result['frequencies'] >= 15000
    im1 = None
    if np.any(ultra_mask):
        im1 = axes[1].pcolormesh(stft_result['times'],
                                 stft_result['frequencies'][ultra_mask],
                                 stft_result['power_db'][ultra_mask, :],
                                 shading='gouraud', cmap='magma',
                                 vmin=np.max(stft_result['power_db'][ultra_mask, :]) - 90)
        axes[1].axhline(y=20000, color='#00FF00', linestyle='--', alpha=0.7, label='20 kHz')
        axes[1].legend(loc='upper right', fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                       labelcolor='#cccccc')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title(f'Ultrasonic Band Zoom: 15–{info["nyquist"] / 1000:.0f} kHz', fontsize=12)
    if im1 is not None:
        fig.colorbar(im1, ax=axes[1], label='Power (dB)', pad=0.01)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_cqt(cqt_result, info, filename):
    """Plot 2: Constant-Q Transform (semitone-lattice-native)."""
    fig, axes = plt.subplots(2, 1, figsize=(24, 14), dpi=150)
    fig.suptitle('ET Audio Analysis — Constant-Q Transform (Semitone-Lattice-Native)\n'
                 f'{cqt_result["n_bins_per_octave"]} bins/octave | '
                 f'{cqt_result["fmin"]:.0f} Hz – {cqt_result["fmax"]:.0f} Hz | '
                 f'{cqt_result["n_bins"]} total bins',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    times = librosa.frames_to_time(np.arange(cqt_result['magnitude'].shape[1]),
                                   sr=info['sample_rate'], hop_length=512)

    # Full CQT
    im0 = axes[0].pcolormesh(times, cqt_result['frequencies'],
                             cqt_result['power_db'],
                             shading='gouraud', cmap='viridis')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Frequency (Hz) — LOG scale', fontsize=12)
    axes[0].set_title('Full CQT — Logarithmic frequency (= ET lattice native)', fontsize=12)
    axes[0].axhline(y=20000, color='#FF4444', linestyle='--', alpha=0.7, label='20 kHz')
    axes[0].axhline(y=440, color='#FFAA00', linestyle='--', alpha=0.5, label='A440 (lattice ref)')
    axes[0].legend(loc='upper right', fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc')
    fig.colorbar(im0, ax=axes[0], label='Power (dB)', pad=0.01)

    # CQT high-frequency zoom (5kHz–Nyquist)
    high_mask = cqt_result['frequencies'] >= 5000
    im1 = None
    if np.any(high_mask):
        high_freqs = cqt_result['frequencies'][high_mask]
        high_db = cqt_result['power_db'][high_mask, :]
        im1 = axes[1].pcolormesh(times, high_freqs, high_db,
                                 shading='gouraud', cmap='plasma')
        axes[1].set_yscale('log')
        axes[1].axhline(y=20000, color='#FF4444', linestyle='--', alpha=0.7, label='20 kHz')
        axes[1].legend(loc='upper right', fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                       labelcolor='#cccccc')
    axes[1].set_ylabel('Frequency (Hz) — LOG scale', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title('CQT High-Frequency Zoom: 5 kHz – Nyquist', fontsize=12)
    if im1 is not None:
        fig.colorbar(im1, ax=axes[1], label='Power (dB)', pad=0.01)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_ssq_cwt(ssq_result, info, filename):
    """Plot 3: Synchrosqueezed CWT (highest resolution)."""
    fig, axes = plt.subplots(2, 1, figsize=(24, 14), dpi=150)
    fig.suptitle('ET Audio Analysis — Synchrosqueezed CWT (Maximum Resolution)\n'
                 f'Morlet wavelet μ=13.4 | {info["sample_rate"]} Hz | Full duration: '
                 f'{ssq_result["analysis_duration"]:.2f}s | '
                 f'{ssq_result["num_scales"]} scales × {ssq_result["num_samples"]:,} samples',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    Tx = ssq_result['Tx']
    Wx = ssq_result['Wx']
    ssq_freqs = ssq_result['ssq_freqs']
    sr = ssq_result['sr']

    t_axis = np.linspace(0, ssq_result['analysis_duration'], Tx.shape[1])

    # PROBED: ssq_freqs is 1D (254,) float64, monotonically DECREASING from
    # Nyquist (24000 Hz) to ~11.72 Hz. All positive. Each element maps directly
    # to one row of Tx/Wx. Decreasing order + pcolormesh = correct spectrogram
    # orientation (high freq at top, low freq at bottom).
    freq_axis = ssq_freqs  # Already in Hz, already positive, no transformation needed
    nyquist_hz = sr / 2.0
    freq_min = np.min(freq_axis)
    freq_max = np.max(freq_axis)
    # Validate: ssq_freqs max should equal Nyquist from sr
    freq_coverage_pct = 100.0 * (freq_max - freq_min) / nyquist_hz

    # SSQ-CWT (synchrosqueezed — focused energy)
    Tx_db = 20.0 * np.log10(np.abs(Tx) + 1e-300)
    vmax_tx = np.percentile(Tx_db[np.isfinite(Tx_db)], 99)
    vmin_tx = vmax_tx - 80

    im0 = axes[0].pcolormesh(t_axis, freq_axis,
                             Tx_db, shading='auto', cmap='inferno',
                             vmin=vmin_tx, vmax=vmax_tx)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].set_ylim(freq_min, freq_max)
    axes[0].axhline(y=20000, color='#00FF00', linestyle='--', alpha=0.5, linewidth=0.8,
                    label='20 kHz (hearing limit)')
    axes[0].axhline(y=F_REF, color='#FFAA00', linestyle='--', alpha=0.3, linewidth=0.8,
                    label=f'A{int(F_REF)} (lattice ref)')
    axes[0].legend(loc='lower right', fontsize=8, facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc')
    axes[0].set_title(f'Synchrosqueezed CWT — {freq_min:.1f} Hz to {freq_max / 1000:.0f} kHz '
                      f'({freq_coverage_pct:.0f}% of Nyquist, highest resolution)', fontsize=12)
    fig.colorbar(im0, ax=axes[0], label='Magnitude (dB)', pad=0.01)

    # Raw CWT (before squeezing — for comparison)
    Wx_db = 20.0 * np.log10(np.abs(Wx) + 1e-300)
    vmax_wx = np.percentile(Wx_db[np.isfinite(Wx_db)], 99)
    vmin_wx = vmax_wx - 80

    im1 = axes[1].pcolormesh(t_axis, freq_axis,
                             Wx_db, shading='auto', cmap='magma',
                             vmin=vmin_wx, vmax=vmax_wx)
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].set_ylim(freq_min, freq_max)
    axes[1].axhline(y=20000, color='#00FF00', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title('Raw CWT (before synchrosqueezing) — for comparison', fontsize=12)
    fig.colorbar(im1, ax=axes[1], label='Magnitude (dB)', pad=0.01)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_ultrasonic_analysis(ultra_result, stft_result, info, filename):
    """Plot 4: Ultrasonic band detail."""
    fig, axes = plt.subplots(3, 1, figsize=(24, 18), dpi=150)
    fig.suptitle('ET Audio Analysis — Ultrasonic Band Detail\n'
                 f'{info["sample_rate"]} Hz | '
                 f'Band: {ultra_result["ultra_band_hz"][0] / 1000:.0f}–'
                 f'{ultra_result["ultra_band_hz"][1] / 1000:.0f} kHz | '
                 f'{ultra_result["num_ultra_bins"]} frequency bins | '
                 f'Ultra/Audible ratio: {ultra_result["ultra_to_audible_ratio_dB"]:.1f} dB',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    times = stft_result['times']

    # Ultrasonic spectrogram
    if ultra_result['ultra_freqs'].size > 0:
        im0 = axes[0].pcolormesh(times, ultra_result['ultra_freqs'],
                                 ultra_result['ultra_power_db'],
                                 shading='gouraud', cmap='hot',
                                 vmin=np.max(ultra_result['ultra_power_db']) - 80)
        axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
        axes[0].set_title('Ultrasonic Spectrogram (20kHz–Nyquist)', fontsize=12)
        fig.colorbar(im0, ax=axes[0], label='Power (dB)', pad=0.01)

    # Band energy comparison
    axes[1].semilogy(times, ultra_result['audible_energy_over_time'] + 1e-300,
                     color='#00AAFF', alpha=0.8, label='Audible (20Hz–20kHz)', linewidth=0.5)
    axes[1].semilogy(times, ultra_result['high_audible_energy_over_time'] + 1e-300,
                     color='#FFAA00', alpha=0.8, label='High-audible (10–20kHz)', linewidth=0.5)
    axes[1].semilogy(times, ultra_result['ultra_energy_over_time'] + 1e-300,
                     color='#FF4444', alpha=0.8, label='Ultrasonic (20kHz+)', linewidth=0.5)
    axes[1].set_ylabel('Mean Band Energy', fontsize=12)
    axes[1].set_title('Band Energy Comparison Over Time', fontsize=12)
    axes[1].legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc')
    axes[1].grid(True, alpha=0.15)

    # Ultrasonic peak frequency tracking with LogNorm for power color scaling
    peak_power = ultra_result['ultra_peak_power']
    if np.any(peak_power > 0):
        power_min = np.min(peak_power[peak_power > 0])
        power_max = np.max(peak_power)
        color_norm = LogNorm(vmin=max(power_min, 1e-20), vmax=max(power_max, 1e-19))
    else:
        color_norm = Normalize(vmin=0, vmax=1)
    axes[2].scatter(times, ultra_result['ultra_peak_freqs'],
                    c=peak_power, cmap='plasma',
                    s=1, alpha=0.6, norm=color_norm)
    axes[2].set_ylabel('Peak Ultrasonic Freq (Hz)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title('Ultrasonic Peak Frequency Tracking', fontsize=12)
    axes[2].axhline(y=20000, color='#00FF00', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_et_lattice_distribution(lattice_result, filename):
    """Plot 5: ET lattice energy distribution at multiple N."""
    N_values = sorted(lattice_result.keys())
    n_plots = len(N_values)

    fig, axes = plt.subplots(n_plots, 2, figsize=(24, 6 * n_plots), dpi=150)
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('ET Audio Analysis — Sempaevum Lattice Energy Distribution\n'
                 'Π_N(f/f_ref) = (k, d, ε) at LCM Tower Levels',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for row in range(n_plots):
        for col in range(2):
            _apply_et_axis_style(axes[row, col])

    cmap = plt.cm.Set3

    for idx, N in enumerate(N_values):
        data = lattice_result[N]
        d_hist = data['d_histogram']

        # Left: d-family energy bar chart
        ax_d = axes[idx, 0]
        d_vals = list(d_hist.keys())
        d_energies = list(d_hist.values())
        total = sum(d_energies) if d_energies else 1
        d_pcts = [100.0 * e / total for e in d_energies]

        colors = [cmap(i / max(len(d_vals), 1)) for i in range(len(d_vals))]
        ax_d.bar([str(d) for d in d_vals], d_pcts, color=colors,
                 edgecolor='#333333', linewidth=0.5)
        ax_d.set_xlabel('Sublattice Family d', fontsize=11)
        ax_d.set_ylabel('Energy (%)', fontsize=11)
        ax_d.set_title(f'N={N} — {data["description"]}\n'
                       f'Mean |ε| = {data["weighted_mean_abs_epsilon"]:.3f} cents',
                       fontsize=11)
        ax_d.grid(True, alpha=0.15, axis='y')

        # Rotate labels if many bars
        if len(d_vals) > 15:
            ax_d.tick_params(axis='x', rotation=90, labelsize=6)

        # Right: epsilon distribution
        ax_e = axes[idx, 1]
        eps_vals = [e[1] for e in data['epsilon_values'] if abs(e[1]) < 50]
        if eps_vals:
            ax_e.hist(eps_vals, bins=100, color='#00AAFF', alpha=0.7,
                      edgecolor='#0066AA', linewidth=0.3)
        ax_e.set_xlabel('ε (cents — lattice alignment error)', fontsize=11)
        ax_e.set_ylabel('Count', fontsize=11)
        ax_e.set_title(f'N={N} — Lattice Alignment Distribution', fontsize=11)
        ax_e.axvline(x=0, color='#FF4444', linestyle='--', alpha=0.7, label='Perfect alignment')
        ax_e.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                    labelcolor='#cccccc')
        ax_e.grid(True, alpha=0.15)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_spectral_stats_and_stereo(spec_stats, stereo_result, info, filename):
    """Plot 6: Spectral statistics and stereo analysis."""
    n_rows = 4 if stereo_result else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(24, 5 * n_rows), dpi=150)
    fig.suptitle(f'ET Audio Analysis — Spectral Statistics & Cross-Channel Analysis\n'
                 f'{info["sample_rate"]} Hz | {info["channels"]} ch | {info["duration"]:.2f}s',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    t = spec_stats['times']

    # Spectral centroid + rolloffs
    axes[0].plot(t, spec_stats['centroid'], color='#00FFAA', linewidth=0.8,
                 label='Centroid')
    axes[0].plot(t, spec_stats['rolloff_85'], color='#FFAA00', linewidth=0.5,
                 alpha=0.7, label='85% rolloff')
    axes[0].plot(t, spec_stats['rolloff_95'], color='#FF6600', linewidth=0.5,
                 alpha=0.7, label='95% rolloff')
    axes[0].plot(t, spec_stats['rolloff_99'], color='#FF2200', linewidth=0.5,
                 alpha=0.7, label='99% rolloff')
    axes[0].axhline(y=20000, color='#666666', linestyle=':', alpha=0.5, label='20 kHz')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[0].set_title('Spectral Centroid & Rolloff Points', fontsize=11)
    axes[0].legend(fontsize=9, loc='upper right', facecolor='#1a1a1a',
                   edgecolor='#333333', labelcolor='#cccccc')
    axes[0].grid(True, alpha=0.15)

    # Spectral flatness
    axes[1].plot(t, spec_stats['flatness'], color='#AA88FF', linewidth=0.8)
    axes[1].set_ylabel('Flatness (0=tonal, 1=noise)', fontsize=11)
    axes[1].set_title('Spectral Flatness (Wiener Entropy)', fontsize=11)
    axes[1].axhline(y=KOIDE, color='#FF4444', linestyle='--', alpha=0.5,
                    label=f'Koide ratio (2/3 ≈ {KOIDE:.4f})')
    axes[1].legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                   labelcolor='#cccccc')
    axes[1].grid(True, alpha=0.15)

    # Spectral contrast
    im = axes[2].imshow(spec_stats['contrast'], aspect='auto', origin='lower',
                        cmap='coolwarm', extent=[t[0], t[-1], 0, spec_stats['contrast'].shape[0]])
    axes[2].set_ylabel('Sub-band index', fontsize=11)
    axes[2].set_title('Spectral Contrast (valley-to-peak per sub-band)', fontsize=11)
    fig.colorbar(im, ax=axes[2], label='Contrast (dB)', pad=0.01)

    # Stereo correlation
    if stereo_result and n_rows > 3:
        ax_s = axes[3]
        ax_s.plot(stereo_result['times'], stereo_result['correlation'],
                  color='#00DDFF', linewidth=0.5, alpha=0.8)
        ax_s.axhline(y=1.0, color='#00FF00', linestyle=':', alpha=0.3, label='Perfect correlation')
        ax_s.axhline(y=0.0, color='#FFFF00', linestyle=':', alpha=0.3, label='Uncorrelated')
        ax_s.axhline(y=-1.0, color='#FF0000', linestyle=':', alpha=0.3, label='Anti-correlated')
        ax_s.axhline(y=stereo_result['mean_correlation'], color='#FFFFFF',
                     linestyle='--', alpha=0.5,
                     label=f'Mean: {stereo_result["mean_correlation"]:.4f}')
        ax_s.set_ylabel('L/R Correlation', fontsize=11)
        ax_s.set_xlabel('Time (s)', fontsize=11)
        ax_s.set_title(f'Stereo Cross-Correlation | Mid/Side Ratio: '
                       f'{stereo_result["mid_side_ratio"]:.2f}', fontsize=11)
        ax_s.set_ylim(-1.1, 1.1)
        ax_s.legend(fontsize=9, loc='lower right', facecolor='#1a1a1a',
                    edgecolor='#333333', labelcolor='#cccccc')
        ax_s.grid(True, alpha=0.15)

    axes[-1].set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


def plot_waveform_and_stats(channels, info, sig_stats, filename):
    """Plot 7: Waveform overview and amplitude statistics."""
    ch_names = [n for n in channels.keys() if n not in ('mid', 'side')]
    n_ch = len(ch_names)

    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(24, 4 * (n_ch + 1)), dpi=150)
    if n_ch + 1 == 1:
        axes = [axes]
    fig.suptitle(f'ET Audio Analysis — Waveform & Amplitude Statistics\n'
                 f'{info["sample_rate"]} Hz / 32-bit / {info["channels"]} ch / '
                 f'{info["duration"]:.3f}s / {info["file_size_bytes"]:,} bytes',
                 fontsize=14, fontweight='bold', color='#00FFAA')
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        _apply_et_axis_style(ax)

    t = np.arange(info['num_samples']) / info['sample_rate']

    for idx, name in enumerate(ch_names):
        sig = channels[name]
        stats = sig_stats[name]
        axes[idx].plot(t, sig, color='#00AAFF', linewidth=0.1, alpha=0.7)
        axes[idx].set_ylabel('Amplitude', fontsize=11)
        axes[idx].set_title(f'{name.upper()} | Peak: {stats["peak_dBFS"]:.1f} dBFS | '
                            f'RMS: {stats["rms_dBFS"]:.1f} dBFS | '
                            f'Crest: {stats["crest_factor_dB"]:.1f} dB | '
                            f'DC: {stats["dc_offset"]:.6f} | '
                            f'ZC rate: {stats["zc_rate_hz"]:.0f} Hz',
                            fontsize=10)
        axes[idx].grid(True, alpha=0.1)

    # Amplitude histogram (last subplot) — dynamic bin count
    for name in ch_names:
        sig = channels[name]
        n_amp = len(sig)
        amp_bins = max(int(np.ceil(np.sqrt(n_amp))), 100)
        axes[-1].hist(sig, bins=amp_bins, alpha=0.5, density=True,
                      label=name.upper(), linewidth=0)
    axes[-1].set_xlabel('Amplitude', fontsize=11)
    axes[-1].set_ylabel('Density', fontsize=11)
    axes[-1].set_title('Amplitude Distribution', fontsize=10)
    axes[-1].legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333333',
                    labelcolor='#cccccc')
    axes[-1].grid(True, alpha=0.15)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return path


# ============================================================================
# MASTER ANALYSIS FUNCTION
# ============================================================================

def run_full_analysis(wav_path):
    """
    Execute the complete ET-lattice-aware audio analysis.
    Every analysis preserves full bandwidth and native sample rate.
    Output filenames are prefixed with the WAV filename (minus extension)
    to support multiple WAV files in the same directory.
    """
    # Derive output prefix from input filename
    wav_basename = os.path.splitext(os.path.basename(wav_path))[0]
    prefix = f"{wav_basename}_et_"

    print("=" * 80)
    print("EXCEPTION THEORY — ULTRA-RESOLUTION AUDIO ANALYSIS ENGINE")
    print("P ∘ D ∘ T = E")
    print(f"Input: {wav_path}")
    print(f"Output prefix: {prefix}")
    print("=" * 80)

    t_start = time.time()
    output_files = []

    # ── STEP 1: Load at native sample rate ──────────────────────────────
    print("\n[1/10] Loading WAV at native sample rate (NO resampling)...")
    channels, info = load_wav_native(wav_path)
    print(f"  Sample rate: {info['sample_rate']} Hz")
    print(f"  Nyquist:     {info['nyquist']} Hz")
    print(f"  Duration:    {info['duration']:.6f} s")
    print(f"  Channels:    {info['channels']}")
    print(f"  Samples:     {info['num_samples']:,}")
    print(f"  File size:   {info['file_size_bytes']:,} bytes")

    # Use first channel for mono analyses, or mid for stereo
    if 'mid' in channels:
        analysis_sig = channels['mid']
        print("  Analysis signal: MID (L+R)/2")
    elif 'mono' in channels:
        analysis_sig = channels['mono']
        print("  Analysis signal: MONO")
    else:
        analysis_sig = channels['ch0']
        print("  Analysis signal: CH0 (Left)")

    sr = info['sample_rate']

    # ── STEP 2: Signal statistics ───────────────────────────────────────
    print("\n[2/10] Computing signal statistics...")
    sig_stats = compute_signal_stats(channels, info)
    for name, stats in sig_stats.items():
        if name in ('mid', 'side'):
            continue
        print(f"  {name.upper()}: Peak={stats['peak_dBFS']:.1f} dBFS, "
              f"RMS={stats['rms_dBFS']:.1f} dBFS, "
              f"Crest={stats['crest_factor_dB']:.1f} dB, "
              f"DC={stats['dc_offset']:.8f}")

    # ── STEP 3: Full-bandwidth STFT ────────────────────────────────────
    print("\n[3/10] Computing full-bandwidth linear STFT (n_fft=16384)...")
    stft_result = compute_full_bandwidth_stft(analysis_sig, sr, n_fft=16384, hop_length=512)
    print(f"  Frequency resolution: {stft_result['freq_resolution']:.3f} Hz/bin")
    print(f"  Time resolution:      {stft_result['time_resolution'] * 1000:.2f} ms")
    print(f"  Frequency bins:       {stft_result['num_freq_bins']:,}")
    print(f"  Frequency range:      {stft_result['freq_range'][0]:.0f} – "
          f"{stft_result['freq_range'][1]:.0f} Hz")

    # ── STEP 4: CQT (semitone-lattice-native) ─────────────────────────
    print("\n[4/10] Computing Constant-Q Transform (36 bins/octave)...")
    cqt_result = compute_cqt(analysis_sig, sr, fmin=20.0, n_bins_per_octave=36)
    print(f"  Total bins:     {cqt_result['n_bins']}")
    print(f"  Frequency range: {cqt_result['fmin']:.0f} – {cqt_result['fmax']:.0f} Hz")

    # ── STEP 5: Synchrosqueezed CWT ───────────────────────────────────
    print("\n[5/10] Computing Synchrosqueezed CWT (Morlet μ=13.4)...")
    ssq_result = compute_ssq_cwt(analysis_sig, sr)
    print(f"  Analysis duration: {ssq_result['analysis_duration']:.2f}s (full signal)")
    print(f"  Tx shape: {ssq_result['Tx'].shape} ({ssq_result['num_scales']} scales × "
          f"{ssq_result['num_samples']:,} samples)")

    # ── STEP 6: Ultrasonic band analysis ──────────────────────────────
    print("\n[6/10] Analyzing ultrasonic band (20kHz–Nyquist)...")
    ultra_result = analyze_ultrasonic_band(analysis_sig, sr, stft_result)
    print(f"  Ultrasonic bins:         {ultra_result['num_ultra_bins']}")
    print(f"  Ultra/Audible ratio:     {ultra_result['ultra_to_audible_ratio_dB']:.1f} dB")
    print(f"  Total ultrasonic energy: {ultra_result['total_ultra_energy']:.6e}")
    print(f"  Total audible energy:    {ultra_result['total_audible_energy']:.6e}")

    # ── STEP 7: Spectral statistics ───────────────────────────────────
    print("\n[7/10] Computing spectral statistics...")
    spec_stats = compute_spectral_stats(analysis_sig, sr)
    print(f"  Mean centroid:  {np.mean(spec_stats['centroid']):.0f} Hz")
    print(f"  Mean flatness:  {np.mean(spec_stats['flatness']):.6f}")
    print(f"  Mean rolloff99: {np.mean(spec_stats['rolloff_99']):.0f} Hz")

    # ── STEP 8: Stereo analysis ───────────────────────────────────────
    print("\n[8/10] Analyzing stereo cross-channel properties...")
    stereo_result = analyze_stereo(channels, sr)
    if stereo_result:
        print(f"  Mean L/R correlation: {stereo_result['mean_correlation']:.6f}")
        print(f"  Mid/Side energy ratio: {stereo_result['mid_side_ratio']:.4f}")
    else:
        print("  (Mono file — stereo analysis skipped)")

    # ── STEP 9: ET lattice distribution ───────────────────────────────
    print("\n[9/10] Computing ET lattice distribution at LCM tower levels...")
    lattice_result = compute_et_lattice_distribution(stft_result, sr,
                                                     N_values=sorted(LCM_TOWER.keys()))
    for N, data in sorted(lattice_result.items()):
        n_families = len(data['d_histogram'])
        print(f"  N={N:>5}: {n_families} sublattice families | "
              f"Mean |ε| = {data['weighted_mean_abs_epsilon']:.4f} cents | "
              f"{data['description']}")

    # ── Save report data before visualization (which deletes large objects) ──
    report_data = {
        'stft_n_fft': stft_result['n_fft'],
        'stft_freq_res': stft_result['freq_resolution'],
        'stft_time_res': stft_result['time_resolution'],
        'stft_freq_bins': stft_result['num_freq_bins'],
        'stft_freq_range': stft_result['freq_range'],
        'ultra_band': ultra_result['ultra_band_hz'],
        'ultra_bins': ultra_result['num_ultra_bins'],
        'ultra_ratio_db': ultra_result['ultra_to_audible_ratio_dB'],
        'ultra_total': ultra_result['total_ultra_energy'],
        'audible_total': ultra_result['total_audible_energy'],
        'lattice_summary': {},
    }
    for N, data in sorted(lattice_result.items()):
        total_e = sum(data['d_histogram'].values())
        report_data['lattice_summary'][N] = {
            'desc': data['description'],
            'mean_eps': data['weighted_mean_abs_epsilon'],
            'd_pcts': {d: 100.0 * e / total_e if total_e > 0 else 0
                       for d, e in sorted(data['d_histogram'].items())},
        }

    # ── STEP 10: Generate visualizations ──────────────────────────────
    print("\n[10/10] Generating visualizations...")

    f1 = plot_waveform_and_stats(channels, info, sig_stats,
                                 f"{prefix}01_waveform.png")
    output_files.append(f1)
    print(f"  [1/7] Waveform & Stats → {f1}")
    plt.close('all')
    gc.collect()

    f2 = plot_full_bandwidth_stft(stft_result, info,
                                  f"{prefix}02_stft_full_bandwidth.png")
    output_files.append(f2)
    print(f"  [2/7] Full-Bandwidth STFT → {f2}")
    plt.close('all')
    gc.collect()

    f3 = plot_cqt(cqt_result, info,
                  f"{prefix}03_cqt_semitone_native.png")
    output_files.append(f3)
    print(f"  [3/7] CQT (Semitone-Native) → {f3}")
    plt.close('all')
    gc.collect()
    del cqt_result  # free memory

    f4 = plot_ssq_cwt(ssq_result, info,
                      f"{prefix}04_ssq_cwt.png")
    output_files.append(f4)
    print(f"  [4/7] Synchrosqueezed CWT → {f4}")
    plt.close('all')
    gc.collect()
    del ssq_result  # free memory

    f5 = plot_ultrasonic_analysis(ultra_result, stft_result, info,
                                  f"{prefix}05_ultrasonic.png")
    output_files.append(f5)
    print(f"  [5/7] Ultrasonic Analysis → {f5}")
    plt.close('all')
    gc.collect()
    del ultra_result  # free memory

    f6 = plot_et_lattice_distribution(lattice_result,
                                      f"{prefix}06_et_lattice.png")
    output_files.append(f6)
    print(f"  [6/7] ET Lattice Distribution → {f6}")
    plt.close('all')
    gc.collect()
    del lattice_result, stft_result  # free memory

    f7 = plot_spectral_stats_and_stereo(spec_stats, stereo_result, info,
                                        f"{prefix}07_spectral_stereo.png")
    output_files.append(f7)
    print(f"  [7/7] Spectral Stats & Stereo → {f7}")
    plt.close('all')
    gc.collect()

    # ── WRITE TEXT REPORT ──────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, f"{prefix}report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXCEPTION THEORY — ULTRA-RESOLUTION AUDIO ANALYSIS REPORT\n")
        f.write("P ∘ D ∘ T = E\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"File: {wav_path}\n")
        f.write(f"Sample Rate: {info['sample_rate']} Hz\n")
        f.write(f"Nyquist: {info['nyquist']} Hz\n")
        f.write(f"Duration: {info['duration']:.6f} s\n")
        f.write(f"Channels: {info['channels']}\n")
        f.write(f"Samples: {info['num_samples']:,}\n")
        f.write(f"File Size: {info['file_size_bytes']:,} bytes\n\n")

        f.write("--- SIGNAL STATISTICS ---\n")
        for name, stats in sig_stats.items():
            if name in ('mid', 'side'):
                continue
            f.write(f"\n  {name.upper()}:\n")
            for key, val in stats.items():
                if isinstance(val, dict):
                    f.write(f"    {key}:\n")
                    for k2, v2 in val.items():
                        f.write(f"      {k2}: {v2}\n")
                else:
                    f.write(f"    {key}: {val}\n")

        f.write("\n--- STFT ANALYSIS ---\n")
        f.write(f"  n_fft: {report_data['stft_n_fft']}\n")
        f.write(f"  Freq resolution: {report_data['stft_freq_res']:.4f} Hz/bin\n")
        f.write(f"  Time resolution: {report_data['stft_time_res'] * 1000:.3f} ms\n")
        f.write(f"  Freq bins: {report_data['stft_freq_bins']}\n")
        f.write(f"  Freq range: {report_data['stft_freq_range'][0]:.0f} – "
                f"{report_data['stft_freq_range'][1]:.0f} Hz\n")

        f.write("\n--- ULTRASONIC ANALYSIS ---\n")
        f.write(f"  Band: {report_data['ultra_band'][0] / 1000:.0f}–"
                f"{report_data['ultra_band'][1] / 1000:.0f} kHz\n")
        f.write(f"  Bins: {report_data['ultra_bins']}\n")
        f.write(f"  Ultra/Audible ratio: {report_data['ultra_ratio_db']:.2f} dB\n")
        f.write(f"  Total ultrasonic energy: {report_data['ultra_total']:.10e}\n")
        f.write(f"  Total audible energy: {report_data['audible_total']:.10e}\n")

        f.write("\n--- ET LATTICE DISTRIBUTION ---\n")
        for N, ldata in sorted(report_data['lattice_summary'].items()):
            f.write(f"\n  N={N} ({ldata['desc']}):\n")
            f.write(f"    Weighted mean |ε|: {ldata['mean_eps']:.6f} cents\n")
            f.write(f"    d-family energy distribution:\n")
            for d, pct in sorted(ldata['d_pcts'].items()):
                name = sublattice_name(d, N)
                f.write(f"      d={d:>5} ({name}): {pct:.2f}%\n")

        if stereo_result:
            f.write("\n--- STEREO ANALYSIS ---\n")
            f.write(f"  Mean L/R correlation: {stereo_result['mean_correlation']:.8f}\n")
            f.write(f"  Mid/Side energy ratio: {stereo_result['mid_side_ratio']:.6f}\n")

        f.write("\n--- SPECTRAL STATISTICS (MEANS) ---\n")
        f.write(f"  Centroid: {np.mean(spec_stats['centroid']):.2f} Hz\n")
        f.write(f"  Bandwidth: {np.mean(spec_stats['bandwidth']):.2f} Hz\n")
        f.write(f"  Flatness: {np.mean(spec_stats['flatness']):.8f}\n")
        f.write(f"  Rolloff 85%: {np.mean(spec_stats['rolloff_85']):.2f} Hz\n")
        f.write(f"  Rolloff 95%: {np.mean(spec_stats['rolloff_95']):.2f} Hz\n")
        f.write(f"  Rolloff 99%: {np.mean(spec_stats['rolloff_99']):.2f} Hz\n")

    output_files.append(report_path)

    # ── WRITE JSON STRUCTURED REPORT ───────────────────────────────────
    json_report_path = os.path.join(OUTPUT_DIR, f"{prefix}report.json")
    json_data = {
        'et_version': 'P ∘ D ∘ T = E',
        'file': wav_path,
        'info': {k: v for k, v in info.items() if not isinstance(v, np.generic)},
        'signal_stats': {},
        'stft': {
            'n_fft': report_data['stft_n_fft'],
            'freq_resolution_hz': report_data['stft_freq_res'],
            'time_resolution_ms': report_data['stft_time_res'] * 1000,
            'freq_bins': report_data['stft_freq_bins'],
            'freq_range_hz': list(report_data['stft_freq_range']),
        },
        'ultrasonic': {
            'band_hz': list(report_data['ultra_band']),
            'bins': int(report_data['ultra_bins']),
            'ultra_to_audible_ratio_dB': report_data['ultra_ratio_db'],
            'total_ultra_energy': report_data['ultra_total'],
            'total_audible_energy': report_data['audible_total'],
        },
        'lattice_distribution': {},
        'spectral_stats_means': {
            'centroid_hz': float(np.mean(spec_stats['centroid'])),
            'bandwidth_hz': float(np.mean(spec_stats['bandwidth'])),
            'flatness': float(np.mean(spec_stats['flatness'])),
            'rolloff_85_hz': float(np.mean(spec_stats['rolloff_85'])),
            'rolloff_95_hz': float(np.mean(spec_stats['rolloff_95'])),
            'rolloff_99_hz': float(np.mean(spec_stats['rolloff_99'])),
        },
    }
    # Add signal stats (convert numpy types for JSON serialization)
    signal_stats_json = {}
    for ch_name, ch_stats in sig_stats.items():
        if ch_name in ('mid', 'side'):
            continue
        ch_serialized = {}
        for k, v in ch_stats.items():
            if isinstance(v, (np.floating, float)):
                ch_serialized[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                ch_serialized[k] = int(v)
            elif isinstance(v, dict):
                ch_serialized[k] = {str(k2): float(v2) for k2, v2 in v.items()}
            else:
                ch_serialized[k] = str(v)
        signal_stats_json[str(ch_name)] = ch_serialized
    json_data['signal_stats'] = signal_stats_json
    # Add lattice data
    lattice_json = {}
    for N, ldata in sorted(report_data['lattice_summary'].items()):
        lattice_json[str(N)] = {
            'description': str(ldata['desc']),
            'weighted_mean_abs_epsilon_cents': float(ldata['mean_eps']),
            'd_family_energy_pct': {str(d): float(pct) for d, pct in ldata['d_pcts'].items()},
        }
    json_data['lattice_distribution'] = lattice_json
    if stereo_result:
        json_data['stereo'] = {
            'mean_lr_correlation': stereo_result['mean_correlation'],
            'mid_side_energy_ratio': stereo_result['mid_side_ratio'],
        }
    with open(json_report_path, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, indent=2, default=str)
    output_files.append(json_report_path)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS COMPLETE — {elapsed:.1f}s elapsed")
    print(f"Generated {len(output_files)} output files")
    print(f"{'=' * 80}")

    return output_files


# ============================================================================
# ENTRY POINT — Windows double-click compatible
# ============================================================================
# Usage:
#   Double-click: processes ALL .wav files in the same folder as this script
#   Command line: python et_audio_analysis.py [file1.wav] [file2.wav] ...
#   If no args and no WAVs found, prints help and waits.

if __name__ == '__main__':
    try:
        # Discover WAV files
        if len(sys.argv) > 1:
            # Command-line: use provided paths
            wav_files = [p for p in sys.argv[1:] if p.lower().endswith('.wav')]
            if not wav_files:
                wav_files = sys.argv[1:]  # user might have omitted .wav extension
        else:
            # Double-click: scan script's own directory for ALL .wav files
            wav_files = sorted([
                os.path.join(SCRIPT_DIR, f)
                for f in os.listdir(SCRIPT_DIR)
                if f.lower().endswith('.wav')
            ])

        if not wav_files:
            print("=" * 80)
            print("EXCEPTION THEORY — ULTRA-RESOLUTION AUDIO ANALYSIS ENGINE")
            print("P ∘ D ∘ T = E")
            print("=" * 80)
            print()
            print("No .wav files found.")
            print()
            print("Place one or more .wav files in the same folder as this script,")
            print(f"then double-click it again.")
            print()
            print(f"Script location: {SCRIPT_DIR}")
            print()
            print("Or run from command line:")
            print(f"  python {os.path.basename(__file__)} <file.wav>")
            print()
            input("Press Enter to exit...")
            sys.exit(0)

        # Validate all files exist
        valid_files = []
        for wav_path in wav_files:
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(SCRIPT_DIR, wav_path)
            if os.path.isfile(wav_path):
                valid_files.append(wav_path)
            else:
                print(f"WARNING: File not found, skipping: {wav_path}")

        if not valid_files:
            print("ERROR: No valid .wav files found.")
            input("Press Enter to exit...")
            sys.exit(1)

        # Process each WAV file
        print(f"Found {len(valid_files)} WAV file(s) to analyze:")
        for i, f in enumerate(valid_files):
            print(f"  [{i + 1}] {os.path.basename(f)}")
        print(f"Output directory: {OUTPUT_DIR}")
        print()

        all_outputs = []
        for i, wav_path in enumerate(valid_files):
            if len(valid_files) > 1:
                print(f"\n{'#' * 80}")
                print(f"# FILE {i + 1}/{len(valid_files)}: {os.path.basename(wav_path)}")
                print(f"{'#' * 80}")

            outputs = run_full_analysis(wav_path)
            all_outputs.extend(outputs)

        # Final summary
        print(f"\n{'=' * 80}")
        print(f"ALL ANALYSIS COMPLETE — {len(valid_files)} file(s), "
              f"{len(all_outputs)} output files")
        print(f"{'=' * 80}")
        for f in all_outputs:
            print(f"  → {os.path.basename(f)}")

    except Exception as e:
        print(f"\n{'!' * 80}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"{'!' * 80}")
        import traceback

        traceback.print_exc()

    finally:
        # Always pause on Windows so the console window stays open
        print()
        input("Press Enter to exit...")
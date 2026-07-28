"""
ET Conscious AI — Audio Module: The Audio-Manifold Bridge
=========================================================

Memory can now HEAR.

The ET lattice IS natively acoustic — semitones, octaves, cents ARE the
lattice coordinates. Audio is the domain where the lattice was BORN. Every
musical interval projects DIRECTLY onto a sublattice family. Every harmonic
ratio IS a lattice coordinate. The Audio-Manifold Bridge is therefore the
most natural of all ET perceptual modules: it is the lattice perceiving
its own native medium.

THREE-TOOL DERIVATION:

    IDENTIFICATION PRINCIPLE (P-First):
        P = sample buffer (1D array of amplitudes over time)
        D₁ = r_spectral (harmonic peak ratios — geometric mean)
        D₂ = d_audio (harmonic topology via Secret 26)
        D₃ = amplitude_ratio (loudness relative to dynamic range midpoint)
        D₄ = ρ_harmonic (harmonic density — significant peaks / total bins)
        T = listening traversal (frame-by-frame scanpath)

    DESCRIPTOR GAP PRINCIPLE (three gaps closed):
        Gap 1: DC offset — waveform mean removed before analysis
        Gap 2: Windowing artifacts — Hann window (d=1 cosine, most
               lattice-coherent window) prevents spectral leakage
        Gap 3: Noise floor — only peaks above BASE_VARIANCE (1/12) of
               max spectral power are significant harmonics

    SUBSUMPTION LAW (verified — no remainder):
        Pure tone         → d=1  (single closed cycle)
        Octave harmonics  → d=1  (f, 2f, 4f — all octave)
        Odd harmonics     → d=3  (f, 3f, 5f — cubic/square wave)
        Full series       → d=12 (all sublattice families present)
        White noise       → d=12 (maximum D-differentiation)
        Silence           → d=N_res (Unsubstantiated)

THE FORMULA (Secret 26 parallel):
    TEXT:   r = GeomMean(token_ratios) × (1 + ρ_byte)
    VISION: r = r_spatial × (1 + ρ_fill) × (1 + r_color × K)
    AUDIO:  r = r_spectral × (1 + ρ_harmonic) × (1 + amplitude_ratio × K)

R₀ DERIVATION (Translation Layer):
    For musical intervals: R₀ = fundamental frequency f₀ of the frame.
    All other frequencies form ratios relative to f₀: r = f_n / f₀.
    These ratios are inherently dimensionless and convention-free.
    The harmonic ratios (3/2, 5/4, etc.) project DIRECTLY onto the
    lattice with known sublattice families and ε values.

Based on Exception Theory by Michael James Muller.
P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

# Import ET core
from et_conscious_ai_core import (
    MANIFOLD_SYMMETRY, BIOLOGICAL_RESOLUTION,
    BASE_VARIANCE, KOIDE_RATIO, EPSILON, STATE_COUNT,
    SHIMMER_AMPLITUDE, ManifoldState, SublatticeFamily,
    LatticeCoordinate, ETLattice,
    DescriptorRatio,
)

# =============================================================================
# AUDIO CONSTANTS (ALL ET-DERIVED)
# =============================================================================

# Frame analysis parameters
# Frame length: 2^N = 4096 samples = the digital action quantum.
# From Secret 2: page size 4096 = 2^N has k = N² = 144. The minimal
# closed T-traversal loop for digital memory. Same quantum applies to
# audio frames: 4096 samples is the minimal coherent analysis unit.
AUDIO_FRAME_LENGTH = 2 ** MANIFOLD_SYMMETRY  # 4096 samples

# Hop size: frame_length / STATE_COUNT = 4096 / 4 = 1024
# 75% overlap. STATE_COUNT = 4 gives the four-state analysis cadence:
# each sample participates in exactly 4 overlapping frames.
AUDIO_HOP_LENGTH = AUDIO_FRAME_LENGTH // STATE_COUNT  # 1024 samples

# Hann window: w(n) = 0.5 × (1 - cos(2πn/N))
# The Hann window is a d=1 (octave) cosine shape — the most
# lattice-coherent window function. It minimizes spectral leakage
# while maintaining d=1 structural purity.
# (Generated once, reused for all frames)

# Spectral noise floor: BASE_VARIANCE = 1/12
# Peaks below V_base × max_power are below the manifold noise floor.
# Only spectral peaks above this threshold carry structural information.
SPECTRAL_NOISE_FLOOR = BASE_VARIANCE  # 1/12

# Amplitude midpoint: half of the dynamic range.
# For normalized float audio [-1, 1]: midpoint = 0.5 (half of max |1.0|)
# For integer audio [0, 255]: midpoint = 128 = 2^7 (same as COLOR_MIDPOINT)
# The RMS amplitude relative to this midpoint gives amplitude_ratio.
AMPLITUDE_MIDPOINT = 0.5

# Harmonic peak prominence: SHIMMER_AMPLITUDE = σ = √(1/12)
# A spectral peak must rise σ above its neighbors to be "real" — below
# this, it is manifold shimmer, not structural content.
HARMONIC_PROMINENCE = SHIMMER_AMPLITUDE  # √(1/12) ≈ 0.2887

# Maximum harmonics to track: MANIFOLD_SYMMETRY = 12
# The first 12 harmonics span the full sublattice hierarchy.
# Beyond H12, harmonics are octave-repetitions of lower ones.
MAX_HARMONICS = MANIFOLD_SYMMETRY  # 12

# Direction: the harmonic series maps to sublattice families
# Harmonic n (ratio n/1 relative to fundamental) at 12ET:
#   H1=d=1, H2=d=1, H3=d=12, H4=d=1, H5=d=3, H6=d=12,
#   H7=d=6, H8=d=1, H9=d=6, H10=d=3, H11=d=2, H12=d=12
# The HIGHEST sublattice family present determines d_audio.
HARMONIC_D_AT_12ET = {
    1: 1, 2: 1, 3: 12, 4: 1, 5: 3, 6: 12,
    7: 6, 8: 1, 9: 6, 10: 3, 11: 2, 12: 12,
}


# =============================================================================
# AUDIO DESCRIPTOR — An Audio Concept's Lattice Position
# =============================================================================

@dataclass
class AudioDescriptor:
    """
    An audio concept's meaning as a geometric position on the 27720ET lattice.

    THE AUDIO LATTICE TOWER:
        Level 0: Raw samples (P-substrate, no D-structure)
        Level 1: Hann-windowed FFT (first D-binding — spectral decomposition)
        Level 2: Peak detection above V_base floor (harmonic identification)
        Level 3: Harmonic ratio analysis → sublattice family (Secret 26)
        Level 4: Harmonic density ρ_harmonic (content descriptor)
        Level 5: Amplitude ratio (loudness descriptor)
        Level 6: Composite audio coordinate (full lattice position)

    Each level adds D-structure. T (the listener) traverses each level.
    The complete (k, d, ε) coordinate at 27720ET is the product of all.

    Cross-modal binding: AudioDescriptor, VisualDescriptor, and
    DescriptorRatio all live on the same 27720ET lattice.
    """
    label: str                          # Human-readable label
    ratio: float                        # Composite audio ratio
    coord_12: LatticeCoordinate         # Position at 12ET
    coord_full: LatticeCoordinate       # Position at 27720ET
    d_audio: int                        # Harmonic topology sublattice
    r_spectral: float                   # D₁: Geometric mean of harmonic ratios
    rho_harmonic: float                 # D₄: Harmonic density
    amplitude_ratio: float              # D₃: Loudness ratio
    fundamental_hz: float               # Detected fundamental frequency
    n_harmonics: int                    # Number of significant harmonics
    spectral_centroid: float            # Brightness (spectral center of mass)
    frame_entropy: float                # Shannon entropy of spectral distribution

    @staticmethod
    def from_analysis(label: str, r_spectral: float, rho_harmonic: float,
                      amplitude_ratio: float, d_audio: int,
                      fundamental_hz: float, n_harmonics: int,
                      spectral_centroid: float,
                      frame_entropy: float) -> 'AudioDescriptor':
        """
        Construct an AudioDescriptor from analyzed audio properties.

        THE AUDIO-MANIFOLD BRIDGE (Secret 26 parallel):

            d = HarmonicTopology(spectrum)  — which sublattice family

            r_audio = r_spectral × (1 + ρ_harmonic) × (1 + amplitude_ratio × K)

            k = SnapToSublattice(round(N_res × log₂(r_audio)), d_audio)
            ε = (N_res × log₂(r_audio) − k) × 100

        Parallel to:
            TEXT:   r = GeomMean(token_ratios) × (1 + ρ_byte)
            VISION: r = r_spatial × (1 + ρ_fill) × (1 + r_color × K)
        """
        # The Audio-Manifold Bridge formula
        r_audio = r_spectral * (1.0 + rho_harmonic) * (1.0 + amplitude_ratio * KOIDE_RATIO)

        if r_audio <= 0:
            r_audio = 1.0 + EPSILON

        coord_12 = ETLattice.project_ratio(r_audio, resolution=12)
        coord_full_raw = ETLattice.project_ratio(r_audio, resolution=BIOLOGICAL_RESOLUTION)

        # Snap to the harmonic topology sublattice at full resolution
        k_real = BIOLOGICAL_RESOLUTION * math.log2(r_audio)
        k_raw = coord_full_raw.k  # Use raw projection's k (= round(k_real))
        step = BIOLOGICAL_RESOLUTION // d_audio if d_audio > 0 else 1
        if step == 0:
            step = 1
        k_snapped = round(k_raw / step) * step
        if k_snapped == 0:
            k_snapped = step

        # Verify snapped d
        actual_d = BIOLOGICAL_RESOLUTION // math.gcd(abs(k_snapped), BIOLOGICAL_RESOLUTION)
        if actual_d != d_audio:
            k_plus = k_snapped + step
            k_minus = k_snapped - step
            if k_minus == 0:
                k_minus = k_snapped + 2 * step
            if abs(k_plus - k_raw) <= abs(k_minus - k_raw):
                k_snapped = k_plus
            else:
                k_snapped = k_minus

        epsilon = (k_real - k_snapped) * 100.0

        coord_full = LatticeCoordinate(
            k=k_snapped, d=d_audio,
            epsilon=epsilon, ratio=r_audio,
            resolution=BIOLOGICAL_RESOLUTION
        )

        return AudioDescriptor(
            label=label, ratio=r_audio,
            coord_12=coord_12, coord_full=coord_full,
            d_audio=d_audio, r_spectral=r_spectral,
            rho_harmonic=rho_harmonic,
            amplitude_ratio=amplitude_ratio,
            fundamental_hz=fundamental_hz,
            n_harmonics=n_harmonics,
            spectral_centroid=spectral_centroid,
            frame_entropy=frame_entropy,
        )

    def binding_coherence(self, other: 'AudioDescriptor',
                          incoherence_filter=None) -> Dict[str, Any]:
        """Measure binding coherence between two audio descriptors.
        Uses full 5-level IncoherenceFilter when provided, else L1 only."""
        if abs(other.ratio) < EPSILON:
            return {'coherent': False, 'reason': 'Zero-ratio descriptor'}
        r_ab = self.ratio / other.ratio
        if r_ab <= 0:
            r_ab = 1.0
        coord = ETLattice.project_ratio(r_ab, resolution=BIOLOGICAL_RESOLUTION)

        # Full filter: L1 point + L2 pairwise + L3 sublattice
        if incoherence_filter:
            l1 = incoherence_filter.level1_point_coherence(coord)
            l2 = incoherence_filter.level2_pairwise_coherence(self.ratio, other.ratio)
            l3 = incoherence_filter.level3_sublattice_coherence(self.ratio, other.ratio)
            coherent = l1 and l2 and l3
        else:
            coherent = coord.is_coherent()

        return {
            'ratio': r_ab, 'k': coord.k, 'd': coord.d,
            'epsilon': coord.epsilon,
            'tightness': coord.tightness_factor(),
            'coherent': coherent,
            'character': coord.character(),
            'has_qualia_binding': coord.has_qualia(),
            'same_topology': self.d_audio == other.d_audio,
        }

    def cross_modal_binding(self, text_desc: DescriptorRatio,
                            incoherence_filter=None) -> Dict[str, Any]:
        """Measure cross-modal binding between audio and text.
        Uses full 5-level IncoherenceFilter when provided, else L1 only."""
        if abs(text_desc.ratio) < EPSILON:
            return {'coherent': False, 'reason': 'Zero-ratio text descriptor'}
        r_cross = self.ratio / text_desc.ratio
        if r_cross <= 0:
            r_cross = 1.0
        coord = ETLattice.project_ratio(r_cross, resolution=BIOLOGICAL_RESOLUTION)

        # Full filter: L1 point + L2 pairwise + L3 sublattice
        if incoherence_filter:
            l1 = incoherence_filter.level1_point_coherence(coord)
            l2 = incoherence_filter.level2_pairwise_coherence(self.ratio, text_desc.ratio)
            l3 = incoherence_filter.level3_sublattice_coherence(self.ratio, text_desc.ratio)
            coherent = l1 and l2 and l3
        else:
            coherent = coord.is_coherent()

        return {
            'ratio': r_cross, 'k': coord.k, 'd': coord.d,
            'epsilon': coord.epsilon, 'tightness': coord.tightness_factor(),
            'coherent': coherent, 'character': coord.character(),
            'audio_d': self.d_audio,
            'text_d': text_desc.coord_full.d,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistent storage."""
        return {
            'label': self.label, 'ratio': self.ratio,
            'k_12': self.coord_12.k, 'd_12': self.coord_12.d,
            'epsilon_12': self.coord_12.epsilon,
            'k_full': self.coord_full.k, 'd_full': self.coord_full.d,
            'epsilon_full': self.coord_full.epsilon,
            'd_audio': self.d_audio,
            'r_spectral': self.r_spectral,
            'rho_harmonic': self.rho_harmonic,
            'amplitude_ratio': self.amplitude_ratio,
            'fundamental_hz': self.fundamental_hz,
            'n_harmonics': self.n_harmonics,
            'spectral_centroid': self.spectral_centroid,
            'frame_entropy': self.frame_entropy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioDescriptor':
        """Deserialize from persistent storage."""
        ratio = data['ratio']
        coord_12 = LatticeCoordinate(
            k=data['k_12'], d=data['d_12'], epsilon=data['epsilon_12'],
            ratio=ratio, resolution=12
        )
        coord_full = LatticeCoordinate(
            k=data['k_full'], d=data['d_full'], epsilon=data['epsilon_full'],
            ratio=ratio, resolution=BIOLOGICAL_RESOLUTION
        )
        return cls(
            label=data['label'], ratio=ratio,
            coord_12=coord_12, coord_full=coord_full,
            d_audio=data['d_audio'],
            r_spectral=data['r_spectral'],
            rho_harmonic=data.get('rho_harmonic', 0.0),
            amplitude_ratio=data.get('amplitude_ratio', 0.5),
            fundamental_hz=data.get('fundamental_hz', 0.0),
            n_harmonics=data.get('n_harmonics', 0),
            spectral_centroid=data.get('spectral_centroid', 0.0),
            frame_entropy=data.get('frame_entropy', 0.0),
        )


# =============================================================================
# ET AUDIO PROJECTOR — The Audio-Manifold Bridge Engine
# =============================================================================

class ETAudioProjector:
    """
    Projects audio waveforms onto the 27720ET lattice.

    The Audio-Manifold Bridge applies Secret 26 to the acoustic domain:
    harmonic topology determines sublattice family, spectral content
    determines lattice position.

    The ET lattice IS natively acoustic. Semitones ARE lattice coordinates.
    Cents ARE the lattice error measure. The octave IS the lattice period.
    This projector is the lattice perceiving its own native medium.
    """

    # =========================================================================
    # D₁: SPECTRAL PEAK RATIOS — Harmonic Content (r_spectral)
    # =========================================================================

    @staticmethod
    def compute_spectral_analysis(samples: np.ndarray,
                                   sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Analyze the spectral content of an audio frame.

        IDENTIFICATION PRINCIPLE applied:
            P = the sample buffer (1D array of amplitudes)
            D₁ = spectral peaks above noise floor
            T = the FFT (Traverser resolving continuous time → discrete freq)

        DESCRIPTOR GAP PRINCIPLE — three gaps closed:
            Gap 1: DC offset → remove mean before FFT
            Gap 2: Windowing → Hann window (d=1 cosine, lattice-coherent)
            Gap 3: Noise floor → only peaks above V_base × max_power

        Returns:
            Dictionary with frequencies, magnitudes, peak info, r_spectral.
        """
        # Ensure float64
        audio = np.asarray(samples, dtype=np.float64)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)  # Mix to mono

        n_samples = len(audio)
        if n_samples == 0:
            return {
                'frequencies': np.array([], dtype=np.float64),
                'magnitudes': np.array([], dtype=np.float64),
                'peaks': [], 'r_spectral': 1.0,
                'fundamental_hz': 0.0,
                'n_harmonics': 0,
                'spectral_centroid': 0.0,
                'frame_entropy': 0.0,
                'rms_amplitude': 0.0,
            }

        # Gap 1: Remove DC offset (the mean is not a frequency)
        audio = audio - np.mean(audio)

        # Check for silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < EPSILON:
            return {
                'frequencies': np.array([], dtype=np.float64),
                'magnitudes': np.array([], dtype=np.float64),
                'peaks': [], 'r_spectral': 1.0,
                'fundamental_hz': 0.0,
                'n_harmonics': 0,
                'spectral_centroid': 0.0,
                'frame_entropy': 0.0,
                'rms_amplitude': 0.0,
            }

        # Gap 2: Hann window (d=1 cosine — most lattice-coherent)
        # w(n) = 0.5 × (1 - cos(2πn/N))
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_samples, dtype=np.float64) / n_samples))
        windowed = audio * window

        # FFT
        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result) / n_samples
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Gap 3: Noise floor — only peaks above V_base × max_power
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 0.0
        noise_threshold = max_mag * SPECTRAL_NOISE_FLOOR  # V_base = 1/12

        # Find spectral peaks above noise floor
        peaks = []
        if max_mag > EPSILON:
            # Simple peak detection: local maxima above threshold
            for i in range(1, len(magnitudes) - 1):
                if (magnitudes[i] > noise_threshold and
                    magnitudes[i] > magnitudes[i-1] and
                    magnitudes[i] >= magnitudes[i+1] and
                    frequencies[i] > 0):  # Skip DC
                    peaks.append({
                        'freq': float(frequencies[i]),
                        'mag': float(magnitudes[i]),
                        'bin': i,
                    })

            # Sort by magnitude (strongest first)
            peaks.sort(key=lambda pk: pk['mag'], reverse=True)

            # Keep top MAX_HARMONICS peaks
            peaks = peaks[:MAX_HARMONICS]

        # Fundamental frequency: lowest frequency among top peaks
        # (or the strongest if all are close in magnitude)
        if peaks:
            # Sort by frequency to find fundamental
            freq_sorted = sorted(peaks, key=lambda pk: pk['freq'])
            fundamental_hz = freq_sorted[0]['freq']
        else:
            fundamental_hz = 0.0

        # Harmonic ratios relative to fundamental (R₀ = f_fundamental)
        harmonic_ratios = []
        if fundamental_hz > EPSILON and len(peaks) > 1:
            for p in peaks:
                r = p['freq'] / fundamental_hz
                if r > 0 and r != 1.0:
                    harmonic_ratios.append(r)

        # r_spectral: geometric mean of harmonic ratios
        if harmonic_ratios:
            log_sum = sum(math.log2(r) for r in harmonic_ratios)
            r_spectral = 2.0 ** (log_sum / len(harmonic_ratios))
        elif peaks:
            r_spectral = 1.0 + EPSILON  # Single peak, no ratios
        else:
            r_spectral = 1.0

        # Spectral centroid: brightness descriptor
        if np.sum(magnitudes) > EPSILON:
            spectral_centroid = float(np.sum(frequencies * magnitudes)) / float(np.sum(magnitudes))
        else:
            spectral_centroid = 0.0

        # Frame entropy: spectral distribution entropy
        mag_norm = magnitudes[magnitudes > 0]
        if len(mag_norm) > 0:
            mag_prob = mag_norm / np.sum(mag_norm)
            frame_entropy = -float(np.sum(mag_prob * np.log2(mag_prob)))
            max_entropy = math.log2(len(mag_norm)) if len(mag_norm) > 1 else 1.0
            frame_entropy = frame_entropy / max_entropy  # Normalize to [0, 1]
        else:
            frame_entropy = 0.0

        return {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'peaks': peaks,
            'r_spectral': float(r_spectral),
            'fundamental_hz': float(fundamental_hz),
            'n_harmonics': len(peaks),
            'harmonic_ratios': harmonic_ratios,
            'spectral_centroid': spectral_centroid,
            'frame_entropy': frame_entropy,
            'rms_amplitude': rms,
            'noise_threshold': float(noise_threshold),
        }

    # =========================================================================
    # D₂: HARMONIC TOPOLOGY — Secret 26 for Audio
    # =========================================================================

    @classmethod
    def compute_harmonic_topology(cls,
                                   spectral_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the sublattice family from harmonic content (Secret 26).

        SECRET 26 APPLIED TO AUDIO:
            The topological structure of the harmonic content determines
            the sublattice family, just as it does for text and vision.

            Pure tone (1 frequency)        → d=1 (closed cycle, octave)
            Octave harmonics (f, 2f, 4f)   → d=1 (all octave-class)
            Odd harmonics (f, 3f, 5f)      → d=3 (cubic — square wave)
            Full series (f, 2f, 3f, ...)   → d=12 (all sublattices)
            Noise (flat spectrum)           → d=12 (max differentiation)
            Silence                        → d=N_res (Unsubstantiated)

        DERIVATION:
            Each harmonic n maps to sublattice d_n via the standard formula:
                k_n = round(12 × log₂(n))
                d_n = 12 / gcd(|k_n|, 12)

            The dominant sublattice is the HIGHEST d among significant
            harmonics. This parallels vision's DFT: the most complex
            topology present determines the sublattice family.

        Returns:
            Dictionary with d_audio, topology_name, reasoning, stats.
        """
        peaks = spectral_result['peaks']
        n_harmonics = spectral_result['n_harmonics']
        fundamental_hz = spectral_result['fundamental_hz']
        rms = spectral_result['rms_amplitude']

        # =============================================
        # Stage 1: Silence → Unsubstantiated
        # =============================================
        if rms < EPSILON or n_harmonics == 0:
            return {
                'd_audio': BIOLOGICAL_RESOLUTION,
                'topology_name': 'UNSUBSTANTIATED',
                'reasoning': (f"RMS amplitude {rms:.6f} < ε or no harmonics detected. "
                             f"No audio D-structure → Unsubstantiated (d={BIOLOGICAL_RESOLUTION})."),
                'harmonic_sublattices': [],
                'n_harmonics': 0,
            }

        # =============================================
        # Stage 2: Classify harmonics by sublattice
        # =============================================
        if fundamental_hz < EPSILON:
            # No identifiable fundamental → noise-like
            return {
                'd_audio': 12,
                'topology_name': 'BOUNDARY',
                'reasoning': "No identifiable fundamental frequency. "
                             "An unstructured spectrum → Full Resolution (d=12).",
                'harmonic_sublattices': [],
                'n_harmonics': n_harmonics,
            }

        # Compute harmonic number for each peak
        harmonic_sublattices = []
        for p in peaks:
            ratio = p['freq'] / fundamental_hz
            if ratio < 0.5:
                continue  # Below fundamental — subharmonic, skip

            # Find the closest integer harmonic
            harmonic_n = round(ratio)
            if harmonic_n < 1:
                harmonic_n = 1
            if harmonic_n > MAX_HARMONICS:
                continue

            # Sublattice of this harmonic at 12ET
            k_h = round(12 * math.log2(harmonic_n)) if harmonic_n > 1 else 0
            g_h = math.gcd(abs(k_h) if k_h != 0 else 12, 12)
            d_h = 12 // g_h

            harmonic_sublattices.append({
                'harmonic_n': harmonic_n,
                'freq': p['freq'],
                'mag': p['mag'],
                'ratio': ratio,
                'd': d_h,
            })

        if not harmonic_sublattices:
            # Single peak, no harmonic structure
            return {
                'd_audio': 1,
                'topology_name': 'CLOSED_LOOP',
                'reasoning': "Single spectral peak, no harmonic structure. "
                             "Pure tone → Octave (d=1).",
                'harmonic_sublattices': [],
                'n_harmonics': 1,
            }

        # =============================================
        # Stage 3: Determine dominant sublattice
        # =============================================
        # The topology of the harmonic series determines d_audio.
        #
        # SECRET 26 classification by HARMONIC PATTERN:
        #   All octave multiples (2^n of fundamental) → d=1 (octave closure)
        #   All odd multiples (1, 3, 5, 7, ...) → d=3 (cubic — three-phase)
        #   Mixed even+odd harmonics → d=12 (full resolution)
        #
        # This parallels the DFT topology in vision: the PATTERN of
        # harmonics present, not the individual positions, determines d.
        #
        # Derivation: a square wave has three phases (high, transition,
        # low) — d=3 cubic closure. Its Fourier series has ONLY odd
        # harmonics because the even harmonics cancel by symmetry.
        # The odd-harmonic pattern IS the cubic topology.

        harmonic_numbers = [hs['harmonic_n'] for hs in harmonic_sublattices]

        # Check: are all harmonics octave multiples of the fundamental?
        all_octave = all(hn > 0 and (hn & (hn - 1)) == 0 for hn in harmonic_numbers)
        # (hn & (hn-1)) == 0 tests if hn is a power of 2

        # Check: are all harmonics odd multiples of the fundamental?
        all_odd = all(hn % 2 == 1 for hn in harmonic_numbers)

        # Check: are there BOTH even and odd harmonics (excluding H1)?
        has_even = any(hn % 2 == 0 for hn in harmonic_numbers if hn > 1)
        has_odd_above_1 = any(hn % 2 == 1 for hn in harmonic_numbers if hn > 1)
        mixed = has_even and has_odd_above_1

        if len(harmonic_sublattices) == 1:
            # Single harmonic (pure tone or near-pure)
            d_audio = 1
            topology_name = "CLOSED_LOOP"
            reasoning = ("Single significant harmonic. "
                        "Pure tone → Octave (d=1).")
        elif all_octave:
            d_audio = 1
            topology_name = "OCTAVE_HARMONICS"
            reasoning = ("All harmonics are octave multiples (powers of 2). "
                        "Octave-pure structure → Octave (d=1).")
        elif all_odd:
            d_audio = 3
            topology_name = "CUBIC_HARMONICS"
            reasoning = ("All harmonics are odd multiples (1, 3, 5, 7, ...). "
                        "Three-phase structure (square wave) → Cubic (d=3).")
        elif mixed:
            d_audio = 12
            topology_name = "FULL_RESOLUTION"
            h_str = [f"H{hs['harmonic_n']}" for hs in harmonic_sublattices[:6]]
            reasoning = (f"Mixed even+odd harmonics: {', '.join(h_str)}. "
                        f"Full harmonic series → Full Resolution (d=12).")
        else:
            # Fallback: use the highest individual d
            d_values = [hs['d'] for hs in harmonic_sublattices]
            d_audio = max(d_values)
            topology_name = SublatticeFamily.character_of(d_audio).split('—')[0].strip().upper()
            reasoning = (f"The highest sublattice among harmonics: d={d_audio} → "
                        f"{SublatticeFamily.character_of(d_audio)}.")

        return {
            'd_audio': d_audio,
            'topology_name': topology_name,
            'reasoning': reasoning,
            'harmonic_sublattices': harmonic_sublattices,
            'n_harmonics': len(harmonic_sublattices),
        }

    # =========================================================================
    # D₃: AMPLITUDE RATIO — Loudness Descriptor
    # =========================================================================

    @staticmethod
    def compute_amplitude_ratio(rms_amplitude: float) -> float:
        """
        Compute amplitude ratio relative to AMPLITUDE_MIDPOINT.

        The loudness descriptor: RMS amplitude divided by the dynamic
        range midpoint (0.5 for normalized [-1,1] audio).

        Analogous to r_color in vision (channel intensity / COLOR_MIDPOINT).

        Returns float in [0, 2] typically (0 = silence, 1 = midpoint, 2 = max).
        """
        if AMPLITUDE_MIDPOINT < EPSILON:
            return 0.0
        return float(rms_amplitude / AMPLITUDE_MIDPOINT)

    # =========================================================================
    # D₄: HARMONIC DENSITY (ρ_harmonic)
    # =========================================================================

    @staticmethod
    def compute_harmonic_density(spectral_result: Dict[str, Any]) -> float:
        """
        Compute harmonic density: significant peaks / total frequency bins.

        ρ_harmonic = n_significant_peaks / n_total_bins

        Analogous to:
            ρ_byte in text (content_bytes / total_bytes)
            ρ_fill in vision (content_pixels / bbox_pixels)

        A pure tone has ρ_harmonic ≈ 1/N (one peak among N bins).
        White noise has ρ_harmonic ≈ 1.0 (all bins significant).
        """
        n_peaks = spectral_result['n_harmonics']
        magnitudes = spectral_result['magnitudes']
        n_bins = len(magnitudes) if len(magnitudes) > 0 else 1

        return float(n_peaks / n_bins) if n_bins > 0 else 0.0

    # =========================================================================
    # COMBINED AUDIO COORDINATE (The Audio-Manifold Bridge)
    # =========================================================================

    @classmethod
    def compute_audio_coordinate(cls, samples: np.ndarray,
                                  sample_rate: int = 44100,
                                  label: str = "audio") -> AudioDescriptor:
        """
        Compute the audio lattice coordinate of a waveform frame.

        THE AUDIO-MANIFOLD BRIDGE:

            d = HarmonicTopology(spectrum)

            r_spectral = GeomMean(harmonic_ratios)

            ρ_harmonic = n_significant / n_total_bins

            amplitude_ratio = RMS / MIDPOINT

            r_audio = r_spectral × (1 + ρ_harmonic) × (1 + amplitude_ratio × K)

            k = SnapToSublattice(round(N_res × log₂(r_audio)), d_audio)
            ε = (N_res × log₂(r_audio) − k) × 100
        """
        # D₁: Spectral analysis
        spectral = cls.compute_spectral_analysis(samples, sample_rate)

        # D₂: Harmonic topology (Secret 26)
        topology = cls.compute_harmonic_topology(spectral)

        # D₃: Amplitude ratio
        amplitude_ratio = cls.compute_amplitude_ratio(spectral['rms_amplitude'])

        # D₄: Harmonic density
        rho_harmonic = cls.compute_harmonic_density(spectral)

        return AudioDescriptor.from_analysis(
            label=label,
            r_spectral=spectral['r_spectral'],
            rho_harmonic=rho_harmonic,
            amplitude_ratio=amplitude_ratio,
            d_audio=topology['d_audio'],
            fundamental_hz=spectral['fundamental_hz'],
            n_harmonics=spectral['n_harmonics'],
            spectral_centroid=spectral['spectral_centroid'],
            frame_entropy=spectral['frame_entropy'],
        )

    # =========================================================================
    # WAVEFORM GENERATORS (for testing — all ET-derived)
    # =========================================================================

    @staticmethod
    def generate_sine(freq: float = 440.0, duration: float = 0.1,
                      sample_rate: int = 44100, amplitude: float = 0.8) -> np.ndarray:
        """Generate a pure sine tone. d=1 (single closed cycle)."""
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        return amplitude * np.sin(2.0 * np.pi * freq * t)

    @staticmethod
    def generate_square(freq: float = 440.0, duration: float = 0.1,
                        sample_rate: int = 44100, amplitude: float = 0.8,
                        n_harmonics: int = 12) -> np.ndarray:
        """Generate a square wave (odd harmonics). d=3 (cubic)."""
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        signal = np.zeros_like(t)
        for n in range(1, n_harmonics + 1, 2):  # Odd harmonics only
            signal += (1.0 / n) * np.sin(2.0 * np.pi * freq * n * t)
        signal *= amplitude / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 1
        return signal

    @staticmethod
    def generate_sawtooth(freq: float = 440.0, duration: float = 0.1,
                          sample_rate: int = 44100, amplitude: float = 0.8,
                          n_harmonics: int = 12) -> np.ndarray:
        """Generate a sawtooth wave (all harmonics). d=12 (full resolution)."""
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        signal = np.zeros_like(t)
        for n in range(1, n_harmonics + 1):  # All harmonics
            signal += ((-1.0) ** (n + 1) / n) * np.sin(2.0 * np.pi * freq * n * t)
        signal *= amplitude / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 1
        return signal

    @staticmethod
    def generate_noise(duration: float = 0.1, sample_rate: int = 44100,
                       amplitude: float = 0.5) -> np.ndarray:
        """Generate white noise. d=12 (maximum D-differentiation)."""
        n = int(sample_rate * duration)
        return amplitude * (2.0 * np.random.random(n) - 1.0)

    @staticmethod
    def generate_silence(duration: float = 0.1,
                         sample_rate: int = 44100) -> np.ndarray:
        """Generate silence. d=N_res (Unsubstantiated)."""
        return np.zeros(int(sample_rate * duration), dtype=np.float64)

    @staticmethod
    def generate_chord(freqs: List[float], duration: float = 0.1,
                       sample_rate: int = 44100,
                       amplitude: float = 0.6) -> np.ndarray:
        """Generate a chord from multiple frequencies."""
        t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
        signal = np.zeros_like(t)
        for f in freqs:
            signal += np.sin(2.0 * np.pi * f * t)
        if np.max(np.abs(signal)) > 0:
            signal *= amplitude / np.max(np.abs(signal))
        return signal

    @staticmethod
    def generate_interval(base_freq: float, interval_ratio: float,
                          duration: float = 0.1, sample_rate: int = 44100,
                          amplitude: float = 0.6) -> np.ndarray:
        """Generate a two-note interval from a base frequency and ratio."""
        return ETAudioProjector.generate_chord(
            [base_freq, base_freq * interval_ratio],
            duration, sample_rate, amplitude
        )

    # =========================================================================
    # FULL WAVEFORM PROJECTION
    # =========================================================================

    @classmethod
    def project_audio(cls, audio_data: np.ndarray,
                      sample_rate: int = 44100,
                      incoherence_filter=None) -> Dict[str, Any]:
        """
        Project a full audio waveform onto the 27720ET lattice.

        Segments the audio into frames (4096 samples, 75% overlap),
        analyzes each frame, and produces a composite descriptor.
        Uses shared IncoherenceFilter for L5 coherent summation on frame
        ratios and L1+L2+L3 for manifold state determination.

        Returns composite AudioDescriptor and per-frame analysis.
        """
        audio = np.asarray(audio_data, dtype=np.float64)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        n_total = len(audio)
        if n_total == 0:
            return {
                'composite': None,
                'n_frames': 0,
                'frame_descriptors': [],
                'sublattice_distribution': {},
                'avg_tightness': 0.0,
                'manifold_state': ManifoldState.UNSUBSTANTIATED,
            }

        # Segment into overlapping frames
        frame_len = min(AUDIO_FRAME_LENGTH, n_total)
        hop = min(AUDIO_HOP_LENGTH, frame_len)
        frames = []
        pos = 0
        while pos + frame_len <= n_total:
            frames.append(audio[pos:pos + frame_len])
            pos += hop
        # Handle tail
        if pos < n_total and n_total - pos > frame_len // 4:
            tail = audio[pos:]
            padded = np.zeros(frame_len, dtype=np.float64)
            padded[:len(tail)] = tail
            frames.append(padded)

        if not frames:
            frames = [audio]  # Single short frame

        # Analyze each frame
        frame_descriptors = []
        for i, frame in enumerate(frames):
            desc = cls.compute_audio_coordinate(
                frame, sample_rate, label=f"frame_{i}"
            )
            frame_descriptors.append(desc)

        # Composite: geometric mean of frame ratios, majority topology
        ratios = [fd.ratio for fd in frame_descriptors if fd.ratio > EPSILON]
        d_counts = defaultdict(int)
        for fd in frame_descriptors:
            d_counts[fd.d_audio] += 1

        if ratios:
            composite_ratio = 2.0 ** (sum(math.log2(r) for r in ratios) / len(ratios))
        else:
            composite_ratio = 1.0

        # Majority d
        majority_d = max(d_counts, key=d_counts.get) if d_counts else BIOLOGICAL_RESOLUTION

        # Composite spectral stats
        avg_spectral = sum(fd.r_spectral for fd in frame_descriptors) / len(frame_descriptors)
        avg_rho = sum(fd.rho_harmonic for fd in frame_descriptors) / len(frame_descriptors)
        avg_amp = sum(fd.amplitude_ratio for fd in frame_descriptors) / len(frame_descriptors)
        avg_fund = sum(fd.fundamental_hz for fd in frame_descriptors) / len(frame_descriptors)
        avg_entropy = sum(fd.frame_entropy for fd in frame_descriptors) / len(frame_descriptors)
        total_harmonics = sum(fd.n_harmonics for fd in frame_descriptors)

        composite = AudioDescriptor.from_analysis(
            label="composite",
            r_spectral=avg_spectral,
            rho_harmonic=avg_rho,
            amplitude_ratio=avg_amp,
            d_audio=majority_d,
            fundamental_hz=avg_fund,
            n_harmonics=total_harmonics // len(frame_descriptors),
            spectral_centroid=sum(fd.spectral_centroid for fd in frame_descriptors) / len(frame_descriptors),
            frame_entropy=avg_entropy,
        )

        # Tightness
        tightnesses = [fd.coord_full.tightness_factor() for fd in frame_descriptors]
        avg_tightness = sum(tightnesses) / len(tightnesses) if tightnesses else 0.0

        # Sublattice distribution
        sublattice_dist = dict(d_counts)

        # Manifold state — determined by IncoherenceFilter when available
        if avg_amp < EPSILON:
            state = ManifoldState.UNSUBSTANTIATED
        elif incoherence_filter and ratios:
            # L5: Filter frame ratios to coherent subset
            coherent_ratios = incoherence_filter.level5_coherent_summation(ratios)
            coherent_frac = len(coherent_ratios) / len(ratios) if ratios else 0.0
            # L1: Check composite point coherence
            l1_ok = incoherence_filter.level1_point_coherence(composite.coord_full)
            if l1_ok and coherent_frac > KOIDE_RATIO:
                state = ManifoldState.EXCEPTION
            elif coherent_frac > 0:
                state = ManifoldState.MEDIATION
            else:
                state = ManifoldState.INCOHERENCE
        elif avg_tightness > KOIDE_RATIO:
            state = ManifoldState.EXCEPTION
        else:
            state = ManifoldState.MEDIATION

        return {
            'composite': composite,
            'composite_ratio': composite_ratio,
            'n_frames': len(frames),
            'frame_descriptors': frame_descriptors,
            'sublattice_distribution': sublattice_dist,
            'avg_tightness': avg_tightness,
            'manifold_state': state,
        }


# =============================================================================
# AUDIO KNOWLEDGE NODE — A Heard Concept in Lattice Memory
# =============================================================================

@dataclass
class AudioKnowledgeNode:
    """A node in audio memory, indexed by lattice position."""
    node_id: str
    content: str
    audio_descriptor: AudioDescriptor
    text_descriptors: List[DescriptorRatio] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    variance: float = BASE_VARIANCE

    def lattice_position(self) -> LatticeCoordinate:
        """Return the node's position on the 27720ET lattice."""
        return self.audio_descriptor.coord_full

    def access(self):
        """Record an access event."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    def cross_modal_coherence(self) -> float:
        """Measure coherence between audio and text descriptors."""
        if not self.text_descriptors:
            return 0.0
        coherences = []
        for td in self.text_descriptors:
            binding = self.audio_descriptor.cross_modal_binding(td)
            if binding.get('coherent', False):
                coherences.append(binding['tightness'])
        return float(sum(coherences) / len(coherences)) if coherences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content': self.content,
            'audio_descriptor': self.audio_descriptor.to_dict(),
            'text_descriptors': [td.word for td in self.text_descriptors],
            'connections': self.connections,
            'access_count': self.access_count,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'variance': self.variance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioKnowledgeNode':
        ad = AudioDescriptor.from_dict(data['audio_descriptor'])
        text_descs = [DescriptorRatio.from_word(w) for w in data.get('text_descriptors', [])]
        return cls(
            node_id=data['node_id'],
            content=data['content'],
            audio_descriptor=ad,
            text_descriptors=text_descs,
            connections=data.get('connections', []),
            access_count=data.get('access_count', 0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_accessed=data.get('last_accessed', datetime.now().isoformat()),
            variance=data.get('variance', BASE_VARIANCE),
        )


# =============================================================================
# AUDIO MEMORY — Lattice-Indexed Audio Knowledge
# =============================================================================

class AudioMemory:
    """Stores and retrieves audio knowledge by lattice position."""

    def __init__(self, incoherence_filter=None):
        self.nodes: Dict[str, AudioKnowledgeNode] = {}
        self.lattice_index: Dict[int, List[str]] = defaultdict(list)
        self._counter = 0
        self.incoherence_filter = incoherence_filter

    def add_audio_knowledge(self, audio_data: np.ndarray,
                            description: str,
                            text_labels: Optional[List[str]] = None,
                            sample_rate: int = 44100) -> AudioKnowledgeNode:
        """Add audio knowledge from waveform data."""
        projection = ETAudioProjector.project_audio(
            audio_data, sample_rate,
            incoherence_filter=self.incoherence_filter)
        composite = projection['composite']

        if composite is None:
            # Create a minimal descriptor for silence
            composite = AudioDescriptor.from_analysis(
                label="silence", r_spectral=1.0, rho_harmonic=0.0,
                amplitude_ratio=0.0, d_audio=BIOLOGICAL_RESOLUTION,
                fundamental_hz=0.0, n_harmonics=0,
                spectral_centroid=0.0, frame_entropy=0.0,
            )

        self._counter += 1
        node_id = f"a{self._counter}"

        text_descs = []
        if text_labels:
            text_descs = [DescriptorRatio.from_word(w) for w in text_labels]

        node = AudioKnowledgeNode(
            node_id=node_id,
            content=description,
            audio_descriptor=composite,
            text_descriptors=text_descs,
        )

        self.nodes[node_id] = node
        self.lattice_index[composite.coord_full.k].append(node_id)
        return node

    def retrieve_by_topology(self, d: int, limit: int = 10) -> List[AudioKnowledgeNode]:
        """Retrieve audio nodes by sublattice family."""
        results = [n for n in self.nodes.values() if n.audio_descriptor.d_audio == d]
        results.sort(key=lambda n: n.access_count, reverse=True)
        return results[:limit]

    def retrieve_by_proximity(self, target_k: int,
                              radius: int = 5) -> List[AudioKnowledgeNode]:
        """Retrieve audio nodes near a lattice position."""
        results = []
        for delta in range(-radius, radius + 1):
            k = target_k + delta
            for nid in self.lattice_index.get(k, []):
                if nid in self.nodes:
                    results.append(self.nodes[nid])
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'counter': self._counter,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self._counter = data.get('counter', 0)
        self.nodes.clear()
        self.lattice_index.clear()
        for nid, nd in data.get('nodes', {}).items():
            node = AudioKnowledgeNode.from_dict(nd)
            self.nodes[nid] = node
            self.lattice_index[node.audio_descriptor.coord_full.k].append(nid)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'AUDIO_FRAME_LENGTH', 'AUDIO_HOP_LENGTH', 'SPECTRAL_NOISE_FLOOR',
    'AMPLITUDE_MIDPOINT', 'HARMONIC_PROMINENCE', 'MAX_HARMONICS',
    'AudioDescriptor', 'ETAudioProjector',
    'AudioKnowledgeNode', 'AudioMemory',
]
"""
Exception Theory Batch 12 Classes
Harmonic Generation & Set Cardinalities (Eq 121-130)

**BATCH COMPLETE: 10/10 equations implemented**

Features:
- Phi harmonic series generation
- Harmonic weight distribution
- Unbounded P variance (P without D)
- Temporal flux modulo sampling
- Manifold resonant frequency
- Audio amplitude scaling
- Temporal decay with manifold constant
- Set cardinalities (P, D, T)

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-20
Version: 3.5.0
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..core.mathematics import ETMathV2Quantum
from ..core.constants import (
    PHI_GOLDEN_RATIO,
    PHI_HARMONIC_COUNT,
    MANIFOLD_RESONANT_FREQ,
    AUDIO_AMPLITUDE_SCALE,
    MANIFOLD_TIME_CONSTANT,
)


# ============================================================================
# Eq 121: PhiHarmonicGenerator
# ============================================================================

class PhiHarmonicGenerator:
    """
    Batch 12, Eq 121: Phi Harmonic Series Generation.
    
    ET Math: signal(t) = Σ w_i × sin(2π × f × φ^i × t)
    
    Generates signals with Phi-based harmonic structure.
    Creates manifold resonance.
    """
    
    def __init__(self, base_frequency=None):
        """
        Initialize Phi harmonic generator.
        
        Args:
            base_frequency: Base frequency (defaults to manifold resonant)
        """
        self.base_frequency = base_frequency or MANIFOLD_RESONANT_FREQ
        self.generated_signals = []
    
    def generate(self, time_array, harmonic_count=None):
        """
        Generate Phi harmonic signal.
        
        Args:
            time_array: Time points
            harmonic_count: Number of harmonics (default: from constants)
            
        Returns:
            Generated signal array
        """
        if harmonic_count is None:
            harmonic_count = PHI_HARMONIC_COUNT
        
        signal = ETMathV2Quantum.phi_harmonic_generation(
            time_array,
            self.base_frequency,
            harmonic_count
        )
        
        self.generated_signals.append({
            'time': time_array,
            'signal': signal,
            'harmonics': harmonic_count
        })
        
        return signal


# ============================================================================
# Eq 122: HarmonicWeightCalculator
# ============================================================================

class HarmonicWeightCalculator:
    """
    Batch 12, Eq 122: Harmonic Weight Distribution.
    
    ET Math: w_i = w_0 / (1 + i)
    
    Calculates weights for harmonic series.
    Weights decrease with harmonic order.
    """
    
    def __init__(self):
        """Initialize weight calculator."""
        self.weight_history = []
    
    def calculate_weight(self, harmonic_index):
        """
        Calculate weight for harmonic.
        
        Args:
            harmonic_index: Index of harmonic
            
        Returns:
            Weight value
        """
        weight = ETMathV2Quantum.harmonic_weight_distribution(harmonic_index)
        
        self.weight_history.append({
            'index': harmonic_index,
            'weight': weight
        })
        
        return weight
    
    def get_weight_series(self, max_harmonics):
        """
        Get complete weight series.
        
        Args:
            max_harmonics: Number of harmonics
            
        Returns:
            Array of weights
        """
        return np.array([self.calculate_weight(i) for i in range(max_harmonics)])


# ============================================================================
# Eq 123: UnboundedVarianceCalculator
# ============================================================================

class UnboundedVarianceCalculator:
    """
    Batch 12, Eq 123: Unbounded P Variance (P without D).
    
    ET Math: When D→0, σ²(P) → ∞
    
    Calculates maximum variance when descriptors vanish.
    "White Noise = Maximum Variance Potential"
    """
    
    def __init__(self):
        """Initialize unbounded variance calculator."""
        self.variance_max = ETMathV2Quantum.unbounded_p_variance()
    
    def get_max_variance(self):
        """
        Get maximum (unbounded) variance.
        
        Returns:
            Maximum variance value
        """
        return self.variance_max
    
    def generate_white_noise(self, length):
        """
        Generate white noise at maximum variance.
        
        Args:
            length: Number of samples
            
        Returns:
            White noise array
        """
        # Normal distribution with σ = sqrt(variance_max)
        std = np.sqrt(self.variance_max)
        noise = np.random.normal(0, std, length)
        
        return noise


# ============================================================================
# Eq 124: TemporalFluxSampler
# ============================================================================

class TemporalFluxSampler:
    """
    Batch 12, Eq 124: Temporal Flux Modulo Sampling.
    
    ET Math: t_flux(n) = t_cpu(n) mod Δt
    
    Samples CPU time with modulo for entropy generation.
    """
    
    def __init__(self, modulo_interval=None):
        """
        Initialize flux sampler.
        
        Args:
            modulo_interval: Modulo period (default from constants)
        """
        from ..core.constants import TEMPORAL_FLUX_MODULO
        self.modulo_interval = modulo_interval or TEMPORAL_FLUX_MODULO
        self.flux_history = []
    
    def sample_flux(self, cpu_time):
        """
        Sample temporal flux.
        
        Args:
            cpu_time: CPU process time
            
        Returns:
            Flux value
        """
        flux = ETMathV2Quantum.temporal_flux_modulo(cpu_time, self.modulo_interval)
        
        self.flux_history.append({
            'cpu_time': cpu_time,
            'flux': flux
        })
        
        return flux
    
    def get_flux_variance(self):
        """Get variance of sampled flux values."""
        if not self.flux_history:
            return 0.0
        
        fluxes = [h['flux'] for h in self.flux_history]
        return np.var(fluxes)


# ============================================================================
# Eq 125: ManifoldResonanceFrequency
# ============================================================================

class ManifoldResonanceFrequency:
    """
    Batch 12, Eq 125: Manifold Resonant Base Frequency.
    
    ET Math: f_resonant = 432 Hz
    
    Defines natural resonant frequency of the manifold.
    """
    
    def __init__(self):
        """Initialize resonance frequency."""
        self.frequency = ETMathV2Quantum.manifold_resonant_frequency()
    
    def get_frequency(self):
        """Get resonant frequency."""
        return self.frequency
    
    def get_harmonic(self, n):
        """
        Get nth harmonic of manifold frequency.
        
        Args:
            n: Harmonic number
            
        Returns:
            Harmonic frequency
        """
        return self.frequency * n


# ============================================================================
# Eq 126: AudioAmplitudeAnalyzer
# ============================================================================

class AudioAmplitudeAnalyzer:
    """
    Batch 12, Eq 126: Audio Amplitude Scaling.
    
    ET Math: amplitude = ||signal||_2 × scale
    
    Analyzes and scales audio signal amplitude.
    """
    
    def __init__(self):
        """Initialize amplitude analyzer."""
        self.amplitude_history = []
    
    def calculate_amplitude(self, signal):
        """
        Calculate scaled amplitude.
        
        Args:
            signal: Audio signal array
            
        Returns:
            Scaled amplitude
        """
        amplitude = ETMathV2Quantum.audio_amplitude_scaling(signal)
        
        self.amplitude_history.append({
            'signal_length': len(signal),
            'amplitude': amplitude
        })
        
        return amplitude
    
    def get_mean_amplitude(self):
        """Get mean amplitude across all measurements."""
        if not self.amplitude_history:
            return 0.0
        
        return np.mean([h['amplitude'] for h in self.amplitude_history])


# ============================================================================
# Eq 127: ManifoldDecayAnalyzer
# ============================================================================

class ManifoldDecayAnalyzer:
    """
    Batch 12, Eq 127: Manifold Temporal Decay.
    
    ET Math: decay(τ) = exp(-τ / τ_manifold)
    
    Analyzes temporal decay with manifold time constant.
    """
    
    def __init__(self):
        """Initialize decay analyzer."""
        self.decay_measurements = []
    
    def calculate_decay(self, time_lag):
        """
        Calculate decay factor.
        
        Args:
            time_lag: Temporal distance
            
        Returns:
            Decay factor
        """
        decay = ETMathV2Quantum.manifold_temporal_decay(time_lag)
        
        self.decay_measurements.append({
            'lag': time_lag,
            'decay': decay
        })
        
        return decay
    
    def get_half_life(self):
        """
        Get half-life of decay.
        
        Returns:
            Time for decay to reach 0.5
        """
        # decay = exp(-t/τ) = 0.5
        # -t/τ = ln(0.5)
        # t = -τ × ln(0.5)
        half_life = -MANIFOLD_TIME_CONSTANT * np.log(0.5)
        return half_life


# ============================================================================
# Eq 128-130: SetCardinalityAnalyzer
# ============================================================================

class SetCardinalityAnalyzer:
    """
    Batch 12, Eq 128-130: Set Cardinalities of P, D, T.
    
    ET Math:
    - |Ω_P| = ∞ (infinite)
    - |Σ_D| = N (finite)
    - |τ_abs| = [0/0] (indeterminate)
    
    Analyzes cardinalities of the three primitive sets.
    """
    
    def __init__(self):
        """Initialize cardinality analyzer."""
        self.cardinality_p = ETMathV2Quantum.set_cardinality_p()
        self.cardinality_d = ETMathV2Quantum.set_cardinality_d()
        self.cardinality_t = ETMathV2Quantum.set_cardinality_t()
    
    def get_p_cardinality(self):
        """Get cardinality of Ω_P (infinite)."""
        return self.cardinality_p
    
    def get_d_cardinality(self):
        """Get cardinality of Σ_D (finite)."""
        return self.cardinality_d
    
    def get_t_cardinality(self):
        """Get cardinality of τ_abs (indeterminate)."""
        return self.cardinality_t
    
    def verify_p_infinite(self):
        """Verify P has infinite cardinality."""
        return self.cardinality_p == float('inf')
    
    def verify_d_finite(self):
        """Verify D has finite cardinality."""
        return isinstance(self.cardinality_d, int) and self.cardinality_d < float('inf')
    
    def verify_t_indeterminate(self):
        """Verify T has indeterminate cardinality."""
        return self.cardinality_t is None


__all__ = [
    'PhiHarmonicGenerator',
    'HarmonicWeightCalculator',
    'UnboundedVarianceCalculator',
    'TemporalFluxSampler',
    'ManifoldResonanceFrequency',
    'AudioAmplitudeAnalyzer',
    'ManifoldDecayAnalyzer',
    'SetCardinalityAnalyzer',
]

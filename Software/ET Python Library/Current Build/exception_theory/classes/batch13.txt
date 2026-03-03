"""
Exception Theory Batch 13 Classes
Signal Processing & Foundational Axioms (Eq 131-140)

**BATCH COMPLETE: 10/10 equations implemented**

Features:
- Amplitude modulation product (Eq 131)
- Output signal scaling/gain (Eq 132)
- Correlation window constraint (Eq 133)
- Cross-correlation product formula (Eq 134)
- Threshold-based state decision (Eq 135)
- Audio sampling rate (Nyquist) (Eq 136)
- Axiom self-validation (Eq 137)
- Exception singularity count (Eq 138)
- Universal exception confirmation (Eq 139)
- Complete categorical disjointness (Eq 140)

From: "For every exception there is an exception, except the exception."

Author: Derived from Michael James Muller's Exception Theory
Date: 2026-01-20
Version: 3.7.0
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from ..core.mathematics import ETMathV2Quantum
from ..core.constants import (
    OUTPUT_GAIN_FACTOR,
    CORRELATION_WINDOW_SIZE,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
    AUDIO_SAMPLE_RATE,
    AXIOM_SELF_GROUNDING,
    GROUNDING_EXCEPTION_COUNT,
    EXCEPTION_CONFIRMATION_RATE,
    CATEGORICAL_INTERSECTION,
)


# ============================================================================
# Eq 131: AmplitudeModulator
# ============================================================================

class AmplitudeModulator:
    """
    Batch 13, Eq 131: Amplitude Modulation Product.
    
    ET Math: output(t) = signal(t) × modulation(t)
    
    Applies amplitude modulation to signals.
    """
    
    def __init__(self):
        """Initialize amplitude modulator."""
        self.modulation_history = []
    
    def modulate(self, signal, modulation):
        """
        Apply amplitude modulation.
        
        Args:
            signal: Base signal
            modulation: Modulation envelope
            
        Returns:
            Modulated signal
        """
        modulated = ETMathV2Quantum.amplitude_modulation_product(signal, modulation)
        
        self.modulation_history.append({
            'signal_length': len(signal),
            'mod_length': len(modulation),
            'output_peak': np.max(np.abs(modulated))
        })
        
        return modulated


# ============================================================================
# Eq 132: SignalScaler
# ============================================================================

class SignalScaler:
    """
    Batch 13, Eq 132: Output Signal Scaling/Gain.
    
    ET Math: output = signal × gain
    
    Scales signal amplitude by gain factor.
    """
    
    def __init__(self, default_gain=None):
        """
        Initialize signal scaler.
        
        Args:
            default_gain: Default gain factor
        """
        self.default_gain = default_gain or OUTPUT_GAIN_FACTOR
        self.scaling_history = []
    
    def scale(self, signal, gain_factor=None):
        """
        Scale signal amplitude.
        
        Args:
            signal: Input signal
            gain_factor: Scaling factor (uses default if None)
            
        Returns:
            Scaled signal
        """
        gain = gain_factor if gain_factor is not None else self.default_gain
        
        scaled = ETMathV2Quantum.output_signal_scaling(signal, gain)
        
        self.scaling_history.append({
            'gain': gain,
            'input_peak': np.max(np.abs(signal)),
            'output_peak': np.max(np.abs(scaled))
        })
        
        return scaled


# ============================================================================
# Eq 133: CorrelationWindowManager
# ============================================================================

class CorrelationWindowManager:
    """
    Batch 13, Eq 133: Correlation Window Size Constraint.
    
    ET Math: N_max = window_size
    
    Manages rolling window for correlation analysis.
    """
    
    def __init__(self, max_size=None):
        """
        Initialize window manager.
        
        Args:
            max_size: Maximum window size
        """
        self.max_size = max_size or CORRELATION_WINDOW_SIZE
        self.window = []
    
    def add_sample(self, value):
        """
        Add sample to window, enforcing size constraint.
        
        Args:
            value: Sample value to add
        """
        self.window.append(value)
        
        # Enforce constraint
        if not ETMathV2Quantum.correlation_window_constraint(len(self.window)):
            self.window.pop(0)  # Remove oldest
    
    def get_window(self):
        """Get current window."""
        return np.array(self.window)
    
    def is_full(self):
        """Check if window is at maximum size."""
        return len(self.window) >= self.max_size


# ============================================================================
# Eq 134: CrossCorrelationAnalyzer
# ============================================================================

class CrossCorrelationAnalyzer:
    """
    Batch 13, Eq 134: Cross-Correlation Product Formula.
    
    ET Math: ρ(a,b) = E[a·b] = mean(a × b)
    
    Calculates Pearson correlation between signals.
    """
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.correlation_history = []
    
    def calculate_correlation(self, signal_a, signal_b):
        """
        Calculate cross-correlation.
        
        Args:
            signal_a: First signal (normalized)
            signal_b: Second signal (normalized)
            
        Returns:
            Correlation coefficient
        """
        correlation = ETMathV2Quantum.cross_correlation_product(signal_a, signal_b)
        
        self.correlation_history.append({
            'correlation': correlation,
            'length': len(signal_a)
        })
        
        return correlation
    
    def get_mean_correlation(self):
        """Get mean correlation across all measurements."""
        if not self.correlation_history:
            return 0.0
        
        return np.mean([h['correlation'] for h in self.correlation_history])


# ============================================================================
# Eq 135: ThresholdDecisionMaker
# ============================================================================

class ThresholdDecisionMaker:
    """
    Batch 13, Eq 135: Threshold-Based State Decision.
    
    ET Math: state = {HIGH, MID, LOW} based on thresholds
    
    Makes tri-state decisions based on dual thresholds.
    """
    
    def __init__(self, high_threshold=None, low_threshold=None):
        """
        Initialize decision maker.
        
        Args:
            high_threshold: Upper threshold
            low_threshold: Lower threshold
        """
        self.high_threshold = high_threshold or THRESHOLD_HIGH
        self.low_threshold = low_threshold or THRESHOLD_LOW
        self.decision_history = []
    
    def make_decision(self, score):
        """
        Make threshold-based decision.
        
        Args:
            score: Input score
            
        Returns:
            State string: 'HIGH', 'MID', or 'LOW'
        """
        state = ETMathV2Quantum.threshold_state_decision(score)
        
        self.decision_history.append({
            'score': score,
            'state': state
        })
        
        return state
    
    def get_state_distribution(self):
        """
        Get distribution of states.
        
        Returns:
            Dict with counts for each state
        """
        if not self.decision_history:
            return {'HIGH': 0, 'MID': 0, 'LOW': 0}
        
        states = [h['state'] for h in self.decision_history]
        return {
            'HIGH': states.count('HIGH'),
            'MID': states.count('MID'),
            'LOW': states.count('LOW')
        }


# ============================================================================
# Eq 136: AudioSamplingRateManager
# ============================================================================

class AudioSamplingRateManager:
    """
    Batch 13, Eq 136: Audio Sampling Rate (Nyquist).
    
    ET Math: f_s = 44100 Hz
    
    Manages audio sampling rate and Nyquist frequency.
    """
    
    def __init__(self):
        """Initialize sampling rate manager."""
        self.sample_rate = ETMathV2Quantum.audio_sampling_rate()
    
    def get_sample_rate(self):
        """Get sampling rate in Hz."""
        return self.sample_rate
    
    def get_nyquist_frequency(self):
        """
        Get Nyquist frequency (max representable frequency).
        
        Returns:
            Nyquist frequency (f_s / 2)
        """
        return self.sample_rate / 2.0
    
    def validate_frequency(self, frequency):
        """
        Validate if frequency can be represented.
        
        Args:
            frequency: Frequency to validate
            
        Returns:
            True if frequency < Nyquist limit
        """
        return frequency < self.get_nyquist_frequency()


# ============================================================================
# Eq 137: AxiomSelfValidator
# ============================================================================

class AxiomSelfValidator:
    """
    Batch 13, Eq 137: Axiom Self-Validation (Reflexive Grounding).
    
    ET Math: validate_axiom(axiom) → True
    
    Validates the foundational axiom's reflexive property.
    The axiom "For every exception there is an exception" applies
    to itself, establishing self-grounding completeness.
    """
    
    def __init__(self):
        """Initialize axiom validator."""
        self.validation_count = 0
        self.self_grounding = AXIOM_SELF_GROUNDING
    
    def validate(self):
        """
        Validate axiom self-grounding property.
        
        Returns:
            True (axiom always validates itself)
        """
        result = ETMathV2Quantum.axiom_self_validation()
        self.validation_count += 1
        return result
    
    def is_self_grounding(self):
        """Check if axiom is self-grounding."""
        return self.self_grounding
    
    def get_validation_count(self):
        """Get number of times validation was performed."""
        return self.validation_count


# ============================================================================
# Eq 138: ExceptionSingularityCounter
# ============================================================================

class ExceptionSingularityCounter:
    """
    Batch 13, Eq 138: Exception Singularity Count.
    
    ET Math: |{grounding_exceptions}| = 1
    
    Tracks and verifies the singular grounding exception.
    Exactly ONE exception (THE Exception) cannot be otherwise.
    """
    
    def __init__(self):
        """Initialize singularity counter."""
        self.grounding_count = GROUNDING_EXCEPTION_COUNT
        self.verification_history = []
    
    def count_grounding_exceptions(self):
        """
        Count grounding exceptions.
        
        Returns:
            1 (always exactly one grounding exception)
        """
        count = ETMathV2Quantum.exception_singularity()
        
        self.verification_history.append({
            'count': count,
            'verified': (count == 1)
        })
        
        return count
    
    def verify_singularity(self):
        """
        Verify singularity property.
        
        Returns:
            True if exactly one grounding exception exists
        """
        count = self.count_grounding_exceptions()
        return count == 1
    
    def get_verification_history(self):
        """Get history of singularity verifications."""
        return self.verification_history


# ============================================================================
# Eq 139: UniversalExceptionConfirmer
# ============================================================================

class UniversalExceptionConfirmer:
    """
    Batch 13, Eq 139: Universal Exception Confirmation.
    
    ET Math: ∀x ∈ exceptions: confirms(x, axiom) = True
    
    Verifies that all exceptions confirm the foundational axiom.
    Any proposed exception validates the axiom by existing.
    """
    
    def __init__(self):
        """Initialize universal confirmer."""
        self.confirmation_rate = EXCEPTION_CONFIRMATION_RATE
        self.exception_count = 0
        self.confirmations = 0
    
    def confirm_exception(self, exception_id=None):
        """
        Confirm that an exception validates the axiom.
        
        Args:
            exception_id: ID of exception (optional)
        
        Returns:
            True (all exceptions confirm axiom)
        """
        self.exception_count += 1
        self.confirmations += 1
        
        # Any exception confirms the axiom
        return True
    
    def get_confirmation_rate(self):
        """
        Get confirmation rate.
        
        Returns:
            1.0 (100% confirmation - all exceptions confirm axiom)
        """
        rate = ETMathV2Quantum.universal_exception_confirmation(self.exception_count)
        return rate
    
    def verify_universal_confirmation(self):
        """
        Verify universal confirmation property.
        
        Returns:
            True if confirmation rate is 1.0 (100%)
        """
        rate = self.get_confirmation_rate()
        return rate == 1.0


# ============================================================================
# Eq 140: CategoricalDisjointnessChecker
# ============================================================================

class CategoricalDisjointnessChecker:
    """
    Batch 13, Eq 140: Complete Categorical Disjointness.
    
    ET Math: P ∩ D ∩ T = ∅
    
    Verifies ontological disjointness of the three primitives.
    Point, Descriptor, and Traverser share no common elements.
    """
    
    def __init__(self):
        """Initialize disjointness checker."""
        self.categorical_intersection = CATEGORICAL_INTERSECTION
        self.checks_performed = 0
        self.disjoint_verified = True
    
    def check_disjointness(self, p_set=None, d_set=None, t_set=None):
        """
        Check complete categorical disjointness.
        
        Args:
            p_set: Point set (optional)
            d_set: Descriptor set (optional)
            t_set: Traverser set (optional)
        
        Returns:
            0 (intersection cardinality always 0)
        """
        result = ETMathV2Quantum.complete_categorical_disjointness(p_set, d_set, t_set)
        
        self.checks_performed += 1
        self.disjoint_verified = (result == 0)
        
        return result
    
    def verify_disjointness(self):
        """
        Verify disjointness property.
        
        Returns:
            True if P, D, T are completely disjoint
        """
        return self.disjoint_verified
    
    def get_intersection_cardinality(self):
        """
        Get expected intersection cardinality.
        
        Returns:
            0 (P ∩ D ∩ T is always empty)
        """
        return self.categorical_intersection


__all__ = [
    'AmplitudeModulator',
    'SignalScaler',
    'CorrelationWindowManager',
    'CrossCorrelationAnalyzer',
    'ThresholdDecisionMaker',
    'AudioSamplingRateManager',
    'AxiomSelfValidator',
    'ExceptionSingularityCounter',
    'UniversalExceptionConfirmer',
    'CategoricalDisjointnessChecker',
]

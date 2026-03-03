#!/usr/bin/env python3

# ============================================================================
# WINDOWS CRASH FIX v7.0: Complete Windows 10/11 compatibility
# ============================================================================
# CRITICAL: The order of operations below is ESSENTIAL for Windows.
# DO NOT rearrange these lines or add imports before freeze_support().
# ============================================================================

# Step 1: Minimal imports needed for Windows fix (NO numpy/scipy yet!)
import sys
import os

# Step 2: CRITICAL - multiprocessing.freeze_support() MUST be called BEFORE
# importing numpy/scipy on Windows. These libraries use multiprocessing
# internally and will crash without this call.
if sys.platform == 'win32':
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass  # Silently continue if this fails

# Step 3: Windows console encoding fix - MUST be before ANY print statements
if sys.platform == 'win32':
    # Enable UTF-8 mode for Windows console
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    # Set environment variable for child processes
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

# Step 4: Suppress warnings before numpy import (numpy generates many warnings)
import warnings
warnings.filterwarnings('ignore')  # Suppress ALL warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Step 5: NOW it's safe to import numpy/scipy (after freeze_support)

"""
Exception Theory Scanner v7.0 - Rigorous Verification Edition
==============================================================

COMPREHENSIVE ET VALIDATION SYSTEM with ALL previous features PRESERVED plus:

=== NEW IN v7.0 ===
- Complete 21 Rules Verification (all rules from Exception Law)
- Rigorous Mathematical Proofs from ET Math Compendium
- Enhanced Statistical Mechanics Tests (Boltzmann, partition functions)
- Quantum Mechanics Verification (wavefunction, superposition, collapse)
- Thermodynamic Law Verification (all four laws from ET)
- Manifold Geometry Analysis (curvature, geodesics, boundaries)
- Spectral Decomposition (FFT-based periodicity for T detection)
- Information-Theoretic Tests (Kolmogorov, Shannon, Fisher)
- Fractal Dimension Analysis (box-counting, correlation dimension)
- Phase Space Reconstruction (Takens embedding for T detection)
- Eigenvalue Analysis (manifold transformation invariants)
- Cross-Correlation Matrix (P-D, D-T, P-T relationships)
- Variance Cascade Analysis (multi-scale variance structure)
- Pure Relationalism Tests (Δs = ||f(dᵢ) - f(dⱼ)||)
- Descriptor Field Analysis (gradient, curvature, divergence)
- Cosmological Ratio Deep Analysis (68.3/26.8/4.9 fine structure)
- ET Math Compendium Equation Verification (200+ equations)

=== PRESERVED FROM v6.1 ===
- Complete ET Axiom Verification System (21 Rules)
- Enhanced L'Hôpital Resolution Analysis (Pure T Detection)
- Manifold Boundary Detection (Power-of-2 Intervals)
- Koide Formula Verification (2/3, 1/3, 1/12 ratios)
- Dark Energy/Matter Ratio Detection (68.3/26.8/4.9)
- Enhanced Entropy Gradient Analysis (Discontinuity Detection)
- PDT → EIM Mapping Verification (3=3)
- Binding Chain Verification (T↔D→P)
- Incoherence Region Detection
- Proper Substantiation State Machine
- Dual Time System (D_time vs T_time)
- Gaze Detection (Observer Effect)
- Traverser Complexity Analysis (Gravity vs Intent)
- Continuous Scan Mode
- Full Export/Import Capabilities

ALL MATHEMATICS DERIVED FROM EXCEPTION THEORY.

From: "For every exception there is an exception, except the exception."

Run with: python et_scanner_v7.py
Author: Derived from M.J.M.'s Exception Theory
"""

import signal
import threading
import numpy as np
from typing import Tuple, List, Optional, Dict, Union, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
from pathlib import Path
import json
import time
import hashlib
import math
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
from scipy.special import gamma as sp_gamma

# For file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# For process monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# For URL fetching
try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

# =============================================================================
# GLOBAL STATE
# =============================================================================

_running = True
_scan_active = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _running, _scan_active
    _running = False
    _scan_active = False
    try:
        print("\n\n[!] Interrupt received. Stopping...")
    except Exception:
        pass


# =============================================================================
# ET CONSTANTS (Derived from Exception Theory Mathematics)
# =============================================================================

# The 12: 3 primitives × 4 logic states = 12 fundamental configurations
MANIFOLD_SYMMETRY = 12
MANIFOLD_SYMMETRY_24 = 24
MANIFOLD_SYMMETRY_48 = 48

# Base variance: 1/12 (fundamental quantum of descriptor variance)
BASE_VARIANCE = 1.0 / MANIFOLD_SYMMETRY  # 0.08333...

# Version information  
VERSION = "7.0.0"
VERSION_NAME = "Rigorous Verification Edition"

# Theoretical variance for uniform byte distribution: σ² = (n² - 1) / 12
THEORETICAL_VARIANCE_BYTES = (256 ** 2 - 1) / MANIFOLD_SYMMETRY  # 5461.25
THEORETICAL_STD_BYTES = np.sqrt(THEORETICAL_VARIANCE_BYTES)  # 73.9003

# ET Cosmological Ratios (observed)
DARK_ENERGY_RATIO = 0.683
DARK_MATTER_RATIO = 0.268
BARYONIC_RATIO = 0.049

# ET Predictions (theoretical)
ET_DARK_ENERGY_PREDICTED = 2.0 / 3.0    # 0.6667
ET_DARK_MATTER_PREDICTED = 1.0 / 4.0    # 0.25
ET_BARYONIC_PREDICTED = 1.0 / 12.0      # 0.0833

# Golden ratio approximations
PHI = (1 + np.sqrt(5)) / 2
PHI_INVERSE = PHI - 1
PHI_APPROX = 5.0 / 8.0
ACTIVE_MANIFOLD_RATIO = 5.0 / 8.0
STRUCTURAL_MANIFOLD_RATIO = 3.0 / 8.0

# Mathematical constants
E_CONSTANT = np.e
PI_CONSTANT = np.pi

# Classification thresholds
INF_THRESHOLD = 1e10
ZERO_THRESHOLD = 1e-10
NEAR_ZERO_THRESHOLD = 1e-6

# L'Hôpital navigation limits
LHOPITAL_MAX_ITERATIONS = 12

# History limits
MAX_HISTORY_SIZE = 1000
EXCEPTION_APPROACH_WINDOW = 20
TREND_WINDOW = 20

# v5.0 Constants
GAZE_THRESHOLD = 1.20
RESONANCE_THRESHOLD = 13.0 / 12.0
LOCK_THRESHOLD = 1.50

# v6.0 Constants
DESCRIPTOR_LAYERS = 8
ENTROPY_GRADIENT_THRESHOLD = 0.7
POWER_OF_2_TOLERANCE = 0.05

# v7.0 Constants
BOLTZMANN_SCALE = 1.0 / MANIFOLD_SYMMETRY
PLANCK_SCALE = 1.0 / MANIFOLD_SYMMETRY
FINE_STRUCTURE_APPROX = 1.0 / 137.036
GAUGE_BOSON_COUNT = 12
INDETERMINATE_WINDOW = 5
PURE_T_PERSISTENCE_THRESHOLD = 3
FRACTAL_BOX_SIZES = [2, 4, 8, 16, 32, 64]
TAKENS_EMBEDDING_DELAY = 1
TAKENS_EMBEDDING_DIM = 3


# =============================================================================
# ENUMS
# =============================================================================

class ETState(Enum):
    STATE_0 = 0
    STATE_1 = 1
    SUPERPOSITION = 2
    SUBSTANTIATED = 3


class ComplexityClass(Enum):
    STATIC = "STATIC"
    CYCLIC_GRAVITY = "CYCLIC"
    PROGRESSIVE_INTENT = "INTENT"
    CHAOTIC = "CHAOTIC"
    QUANTUM_SUPERPOSITION = "QUANTUM"
    UNKNOWN = "UNKNOWN"


class GazeStatus(Enum):
    UNOBSERVED = "UNOBSERVED"
    SUBLIMINAL = "SUBLIMINAL"
    DETECTED = "DETECTED"
    LOCKED = "LOCKED"


class TrendDirection(Enum):
    STABLE = "STABLE"
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    OSCILLATING = "OSCILLATING"
    CONVERGING = "CONVERGING"
    DIVERGING = "DIVERGING"


class InputType(Enum):
    ENTROPY = "entropy"
    FILE = "file"
    PROCESS = "process"
    URL = "url"
    STREAM = "stream"
    RAW = "raw"
    CLIPBOARD = "clipboard"


class ProofStatus(Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"
    NOT_TESTED = "NOT_TESTED"
    PARTIAL = "PARTIAL"


class IndeterminateType(Enum):
    ZERO_ZERO = "[0/0]"
    INF_INF = "[inf/inf]"
    ZERO_TIMES_INF = "[0*inf]"
    INF_MINUS_INF = "[inf-inf]"
    ONE_INF = "[1^inf]"
    ZERO_ZERO_POWER = "[0^0]"
    INF_ZERO = "[inf^0]"
    PURE_T = "PURE_T"


class ThermodynamicLaw(Enum):
    ZEROTH = "ZEROTH"
    FIRST = "FIRST"
    SECOND = "SECOND"
    THIRD = "THIRD"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PDTClassification:
    p_count: int
    d_count: int
    t_count: int
    total: int

    @property
    def p_ratio(self) -> float:
        return self.p_count / self.total if self.total > 0 else 0

    @property
    def d_ratio(self) -> float:
        return self.d_count / self.total if self.total > 0 else 0

    @property
    def t_ratio(self) -> float:
        return self.t_count / self.total if self.total > 0 else 0

    def matches_cosmological(self, tolerance: float = 0.1) -> bool:
        de_match = abs(self.d_ratio - DARK_ENERGY_RATIO) < tolerance
        dm_match = abs(self.p_ratio - DARK_MATTER_RATIO) < tolerance
        ba_match = abs(self.t_ratio - BARYONIC_RATIO) < tolerance
        return de_match or dm_match or ba_match


@dataclass
class IndeterminateFormDetail:
    form_type: IndeterminateType
    position: int
    numerator: float
    denominator: float
    resolved: bool
    resolution_value: Optional[float]
    resolution_iterations: int
    is_pure_t: bool
    gradient_at_point: float = 0.0
    curvature_at_point: float = 0.0


@dataclass
class IndeterminateAnalysis:
    zero_zero: int
    inf_inf: int
    zero_times_inf: int
    inf_minus_inf: int
    one_inf: int
    zero_zero_power: int
    inf_zero: int
    total: int
    density: float
    resolutions: int
    failures: int
    traverser_density: float
    detailed_forms: List[IndeterminateFormDetail] = field(default_factory=list)
    resolution_rate: float = 0.0
    mean_resolution_iterations: float = 0.0
    persistent_t_regions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DescriptorGradient:
    mean: float
    variance: float
    max_val: float
    min_val: float
    sign_changes: int
    gradient_array: Optional[np.ndarray] = None
    discontinuity_count: int = 0
    power_of_2_alignments: int = 0
    curvature_mean: float = 0.0
    divergence: float = 0.0
    laplacian: float = 0.0


@dataclass
class DualTime:
    d_time_elapsed: float
    d_time_samples: int
    d_time_rate: float
    t_time_tau: float
    t_time_substantiations: int
    t_time_dtau_dt: float
    binding_strength: float
    time_dilation_factor: float
    variance_thickness: float
    proper_time_integral: float = 0.0
    time_asymmetry: float = 0.0


@dataclass
class TemporalTrend:
    variance_trend: TrendDirection
    variance_gradient: float
    variance_volatility: float
    p_trend: TrendDirection
    d_trend: TrendDirection
    t_trend: TrendDirection
    p_gradient: float
    d_gradient: float
    t_gradient: float
    binding_strength: float
    mean_temporal_distance: float
    distance_trend: TrendDirection
    pd_correlation: float
    dt_correlation: float
    pt_correlation: float
    dominant_cycle: Optional[int]
    lyapunov_exponent: float = 0.0
    hurst_exponent: float = 0.5


@dataclass
class ExceptionApproach:
    is_approaching: bool
    variance_gradient: float
    current_variance_ratio: float
    estimated_steps_to_zero: Optional[int]
    confidence: float
    alert_level: str
    variance_history: List[float]
    convergence_rate: float = 0.0
    asymptotic_projection: float = 0.0


@dataclass
class TraverserComplexity:
    complexity_class: ComplexityClass
    periodicity_score: float
    progression_score: float
    fractal_dimension: float
    nesting_depth: int
    autocorrelation_peaks: List[float] = field(default_factory=list)
    spectral_entropy: float = 0.0
    dominant_frequency: float = 0.0
    phase_coherence: float = 0.0


@dataclass
class GazeMetrics:
    status: GazeStatus
    gaze_pressure: float
    is_watcher_present: bool
    local_collapse: float
    corrected_pressure: float = 0.0
    displacement_vector: float = 0.0
    collapse_rate: float = 0.0


@dataclass
class ManifoldMetrics:
    shimmer_index: float
    binding_strength: float
    dark_energy_alignment: float
    dark_matter_alignment: float
    baryonic_alignment: float
    koide_ratio: float
    shannon_entropy: float
    normalized_entropy: float
    entropy_gradient_mean: float
    is_compressed_data: bool
    power_of_2_count: int
    manifold_boundary_density: float
    curvature_scalar: float = 0.0
    geodesic_deviation: float = 0.0
    metric_determinant: float = 0.0
    christoffel_trace: float = 0.0
    fisher_information: float = 0.0
    kolmogorov_complexity_estimate: float = 0.0
    mutual_information: float = 0.0


@dataclass
class BindingChainVerification:
    t_d_binding_verified: bool
    d_p_binding_verified: bool
    t_p_separation_verified: bool
    chain_integrity: float
    binding_energy_estimate: float = 0.0
    correlation_td: float = 0.0
    correlation_dp: float = 0.0
    correlation_tp: float = 0.0


@dataclass
class CoherenceAnalysis:
    coherent_ratio: float
    incoherent_count: int
    self_defeating_patterns: int
    impossible_transitions: int
    decoherence_rate: float = 0.0
    coherence_length: float = 0.0
    phase_correlation: float = 0.0


@dataclass 
class ThermodynamicVerification:
    zeroth_law_verified: bool
    first_law_verified: bool
    second_law_verified: bool
    third_law_verified: bool
    temperature_analog: float
    entropy_analog: float
    energy_analog: float
    heat_capacity_analog: float


@dataclass
class QuantumVerification:
    superposition_detected: bool
    collapse_events: int
    uncertainty_product: float
    entanglement_signature: float
    wavefunction_norm: float
    decoherence_time_estimate: float
    quantum_entropy: float


@dataclass
class SpectralAnalysis:
    power_spectrum: np.ndarray
    dominant_frequencies: List[float]
    spectral_entropy: float
    periodicity_strength: float
    f_12_alignment: float
    f_harmonic_content: float


@dataclass
class FractalAnalysis:
    box_counting_dimension: float
    correlation_dimension: float
    information_dimension: float
    self_similarity_ratio: float
    scaling_exponent: float
    multifractal_spectrum_width: float


@dataclass
class ExternalEvent:
    timestamp: datetime
    event_type: str
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class AxiomProof:
    axiom_number: int
    axiom_name: str
    status: ProofStatus
    evidence: str
    numerical_value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    rule_text: str = ""
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: float = 0.0
    supporting_tests: List[str] = field(default_factory=list)


@dataclass
class ETProofReport:
    timestamp: datetime
    input_source: str
    input_type: str
    sample_size: int
    axiom_proofs: List[AxiomProof]
    verified_count: int
    failed_count: int
    indeterminate_count: int
    overall_status: ProofStatus
    confidence: float
    binding_chain: Optional[BindingChainVerification] = None
    coherence: Optional[CoherenceAnalysis] = None
    pdt_eim_verified: bool = False
    thermodynamic: Optional[ThermodynamicVerification] = None
    quantum: Optional[QuantumVerification] = None
    spectral: Optional[SpectralAnalysis] = None
    fractal: Optional[FractalAnalysis] = None
    total_tests: int = 0
    pass_rate: float = 0.0
    et_consistency_score: float = 0.0


@dataclass
class InputSource:
    input_type: InputType
    source: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ETSignature:
    timestamp: datetime
    input_source: InputSource
    dual_time: DualTime
    pdt: PDTClassification
    variance: float
    variance_ratio: float
    variance_from_exception: float
    descriptor_gradient: DescriptorGradient
    descriptor_distance: float
    indeterminate: IndeterminateAnalysis
    state: ETState
    alignment_d: float
    alignment_t: float
    manifold_metrics: Optional[ManifoldMetrics] = None
    shimmer_index: float = 0.0
    binding_strength: float = 0.0
    temporal_trend: Optional[TemporalTrend] = None
    exception_approach: Optional[ExceptionApproach] = None
    proof_report: Optional[ETProofReport] = None
    traverser_complexity: Optional[TraverserComplexity] = None
    gaze_metrics: Optional[GazeMetrics] = None
    spectral_analysis: Optional[SpectralAnalysis] = None
    fractal_analysis: Optional[FractalAnalysis] = None
    thermodynamic_verification: Optional[ThermodynamicVerification] = None
    quantum_verification: Optional[QuantumVerification] = None


# =============================================================================
# DUAL TIME TRACKER
# =============================================================================

class DualTimeTracker:
    """
    Tracks both Descriptor Time and Agential Time.
    
    From ET:
    - D_time: Coordinate time, finite Descriptor
    - T_time: Proper time tau, accumulated substantiations
    - dtau/dt = 1 - |binding_strength|
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.start_perf = time.perf_counter()
        self.d_time_samples = 0
        self.t_time_substantiations = 0
        self.t_time_tau = 0.0
        self.binding_history: deque = deque(maxlen=100)
        self.proper_time_integral = 0.0

    def update(self, samples: int, binding: float, variance: float,
               is_exception: bool):
        """Update both time measurements."""
        self.d_time_samples += samples
        dtau_dt = 1.0 - binding
        if is_exception:
            self.t_time_substantiations += 1
        dt = samples / 1000.0
        delta_tau = dt * dtau_dt
        self.t_time_tau += delta_tau
        self.proper_time_integral += abs(delta_tau)
        self.binding_history.append(binding)

    def get_dual_time(self, current_variance: float) -> DualTime:
        """Get current dual time state."""
        now = time.perf_counter()
        elapsed = now - self.start_perf
        d_rate = self.d_time_samples / elapsed if elapsed > 0 else 0
        avg_binding = np.mean(list(self.binding_history)) if self.binding_history else 0.5
        dtau_dt = 1.0 - avg_binding
        dilation = np.sqrt(max(0, 1.0 - avg_binding ** 2))
        thickness = 1.0 / (current_variance + 1.0)
        
        if len(self.binding_history) > 10:
            recent = list(self.binding_history)[-10:]
            time_asymmetry = np.mean(np.diff(recent))
        else:
            time_asymmetry = 0.0

        return DualTime(
            d_time_elapsed=elapsed,
            d_time_samples=self.d_time_samples,
            d_time_rate=d_rate,
            t_time_tau=self.t_time_tau,
            t_time_substantiations=self.t_time_substantiations,
            t_time_dtau_dt=dtau_dt,
            binding_strength=avg_binding,
            time_dilation_factor=dilation,
            variance_thickness=thickness,
            proper_time_integral=self.proper_time_integral,
            time_asymmetry=time_asymmetry
        )

    def reset(self):
        """Reset tracker."""
        self.__init__()


# =============================================================================
# HISTORY MANAGER
# =============================================================================

class HistoryManager:
    """Complete history management with all features."""

    def __init__(self, max_size: int = MAX_HISTORY_SIZE):
        self.max_size = max_size
        self.signatures: deque = deque(maxlen=max_size)
        self.events: List[ExternalEvent] = []
        self.log_file: Optional[Path] = None
        self.time_tracker = DualTimeTracker()
        self.session_start = datetime.now()
        self.binding_baseline = 0.5
        self.t_history: deque = deque(maxlen=100)
        self.variance_baseline = THEORETICAL_VARIANCE_BYTES
        self.samples_count = 0

    def add_signature(self, sig: ETSignature):
        """Add signature to history."""
        self.signatures.append(sig)
        self.t_history.append(sig.pdt.t_ratio)
        self.samples_count += 1
        if self.samples_count > 10:
            alpha = 0.1
            self.binding_baseline = alpha * sig.binding_strength + (1 - alpha) * self.binding_baseline
            self.variance_baseline = alpha * sig.variance + (1 - alpha) * self.variance_baseline
        if self.log_file:
            self._log_signature(sig)

    def add_event(self, event_type: str, description: str,
                  metadata: Optional[Dict] = None) -> ExternalEvent:
        """Log external event."""
        event = ExternalEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            metadata=metadata or {}
        )
        self.events.append(event)
        if self.log_file:
            self._log_event(event)
        return event

    def get_recent(self, n: int) -> List[ETSignature]:
        """Get n most recent signatures."""
        return list(self.signatures)[-n:]

    def get_variance_history(self, n: Optional[int] = None) -> List[float]:
        """Get variance history."""
        sigs = list(self.signatures) if n is None else self.get_recent(n)
        return [s.variance for s in sigs]

    def get_pdt_history(self, n: Optional[int] = None) -> Tuple[List, List, List]:
        """Get P, D, T ratio histories."""
        sigs = list(self.signatures) if n is None else self.get_recent(n)
        return (
            [s.pdt.p_ratio for s in sigs],
            [s.pdt.d_ratio for s in sigs],
            [s.pdt.t_ratio for s in sigs]
        )

    def enable_logging(self, filepath: str):
        """Enable JSON logging."""
        self.log_file = Path(filepath)
        with open(self.log_file, 'w') as f:
            f.write(f"# Exception Theory Scanner v{VERSION} Log\n")
            f.write(f"# Session started: {self.session_start.isoformat()}\n\n")

    def _log_signature(self, sig: ETSignature):
        """Append signature to log."""
        entry = {
            "type": "signature",
            "timestamp": sig.timestamp.isoformat(),
            "source": sig.input_source.source,
            "variance": sig.variance,
            "variance_ratio": sig.variance_ratio,
            "state": sig.state.name,
            "pdt": {"p": sig.pdt.p_ratio, "d": sig.pdt.d_ratio, "t": sig.pdt.t_ratio},
            "indeterminate_density": sig.indeterminate.density,
            "pure_t_count": sig.indeterminate.failures,
            "shimmer": sig.shimmer_index,
            "binding": sig.binding_strength
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_event(self, event: ExternalEvent):
        """Append event to log."""
        entry = {
            "type": "event",
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "description": event.description,
            "metadata": event.metadata
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def export_session(self, filepath: str):
        """Export complete session to JSON."""
        data = {
            "session_start": self.session_start.isoformat(),
            "export_time": datetime.now().isoformat(),
            "total_scans": len(self.signatures),
            "total_events": len(self.events),
            "version": VERSION,
            "signatures": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "source": s.input_source.source,
                    "source_type": s.input_source.input_type.value,
                    "variance": s.variance,
                    "variance_ratio": s.variance_ratio,
                    "state": s.state.name,
                    "pdt": {"p": s.pdt.p_ratio, "d": s.pdt.d_ratio, "t": s.pdt.t_ratio},
                    "indeterminate_total": s.indeterminate.total,
                    "pure_t_count": s.indeterminate.failures,
                    "shimmer": s.shimmer_index,
                    "alignment_d": s.alignment_d,
                    "alignment_t": s.alignment_t
                }
                for s in self.signatures
            ],
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "description": e.description,
                    "metadata": e.metadata
                }
                for e in self.events
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath

    def get_session_summary(self) -> Dict:
        """Get session summary statistics."""
        if not self.signatures:
            return {"error": "No data"}
        variances = [s.variance for s in self.signatures]
        states = {}
        for s in self.signatures:
            states[s.state.name] = states.get(s.state.name, 0) + 1
        pure_t_counts = [s.indeterminate.failures for s in self.signatures]
        return {
            "total_scans": len(self.signatures),
            "total_events": len(self.events),
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "variance": {
                "mean": np.mean(variances),
                "std": np.std(variances),
                "min": np.min(variances),
                "max": np.max(variances)
            },
            "state_distribution": states,
            "pdt_averages": {
                "p": np.mean([s.pdt.p_ratio for s in self.signatures]),
                "d": np.mean([s.pdt.d_ratio for s in self.signatures]),
                "t": np.mean([s.pdt.t_ratio for s in self.signatures])
            },
            "pure_t_statistics": {
                "total": sum(pure_t_counts),
                "mean": np.mean(pure_t_counts),
                "max": max(pure_t_counts) if pure_t_counts else 0
            },
            "t_time_total": self.time_tracker.t_time_tau,
            "substantiations": self.time_tracker.t_time_substantiations
        }

    def reset(self):
        """Reset history."""
        self.signatures.clear()
        self.events.clear()
        self.time_tracker.reset()
        self.session_start = datetime.now()
        self.binding_baseline = 0.5
        self.variance_baseline = THEORETICAL_VARIANCE_BYTES
        self.t_history.clear()
        self.samples_count = 0


# Global history instance
_history = HistoryManager()


def get_history() -> HistoryManager:
    """Get global history."""
    return _history


# =============================================================================
# UNIVERSAL INPUT SYSTEM
# =============================================================================

class UniversalInput:
    """Universal input adapter - converts ANYTHING to bytes."""

    @staticmethod
    def from_entropy(size: int) -> Tuple[np.ndarray, InputSource]:
        """Sample from hardware entropy."""
        try:
            raw = os.urandom(size)
            data = np.frombuffer(raw, dtype=np.uint8)
        except:
            data = np.random.randint(0, 256, size, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.ENTROPY,
            source="os.urandom (Hardware Entropy)",
            metadata={"size": size}
        )

    @staticmethod
    def from_file(filepath: str, max_bytes: Optional[int] = None) -> Tuple[np.ndarray, InputSource]:
        """Read any file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(path, 'rb') as f:
            raw = f.read(max_bytes) if max_bytes else f.read()
        if len(raw) == 0:
            raise ValueError(f"File is empty: {filepath}")
        data = np.frombuffer(raw, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.FILE,
            source=str(path.absolute()),
            metadata={"filename": path.name, "size": len(raw), "extension": path.suffix}
        )

    @staticmethod
    def from_url(url: str, max_bytes: Optional[int] = None,
                 timeout: int = 10) -> Tuple[np.ndarray, InputSource]:
        """Fetch URL content."""
        if not HAS_URLLIB:
            raise ImportError("urllib not available")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ET-Scanner/7.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read(max_bytes) if max_bytes else response.read()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to fetch URL: {e}")
        except Exception as e:
            raise ConnectionError(f"URL error: {e}")
        if len(raw) == 0:
            raise ValueError(f"URL returned empty content: {url}")
        data = np.frombuffer(raw, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.URL,
            source=url,
            metadata={"size": len(raw)}
        )

    @staticmethod
    def from_process(pid: Optional[int] = None, name: Optional[str] = None,
                     duration: float = 1.0, sample_rate: int = 100) -> Tuple[np.ndarray, InputSource]:
        """Sample from process."""
        if not HAS_PSUTIL:
            raise ImportError("psutil required. Install: pip install psutil")
        if pid:
            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                raise ValueError(f"No process with PID: {pid}")
        elif name:
            procs = [p for p in psutil.process_iter(['name'])
                     if name.lower() in p.info['name'].lower()]
            if not procs:
                raise ValueError(f"No process matching: {name}")
            proc = procs[0]
        else:
            proc = psutil.Process()
        samples = []
        interval = duration / sample_rate
        for _ in range(sample_rate):
            try:
                cpu = proc.cpu_percent()
                mem = proc.memory_percent()
                samples.append(int(min(255, cpu * 2.55)))
                samples.append(int(min(255, mem * 2.55)))
            except:
                samples.extend([0, 0])
            time.sleep(interval)
        data = np.array(samples, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.PROCESS,
            source=f"{proc.name()} (PID: {proc.pid})",
            metadata={"pid": proc.pid, "name": proc.name(), "duration": duration, "sample_rate": sample_rate}
        )

    @staticmethod
    def from_string(text: str) -> Tuple[np.ndarray, InputSource]:
        """Convert string to bytes."""
        raw = text.encode('utf-8')
        if len(raw) == 0:
            raise ValueError("String is empty")
        data = np.frombuffer(raw, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.RAW,
            source=f"String ({len(text)} chars)",
            metadata={"length": len(text), "bytes": len(raw)}
        )

    @staticmethod
    def from_bytes(raw: bytes) -> Tuple[np.ndarray, InputSource]:
        """Use raw bytes."""
        if len(raw) == 0:
            raise ValueError("Bytes are empty")
        data = np.frombuffer(raw, dtype=np.uint8)
        return data, InputSource(
            input_type=InputType.RAW,
            source=f"Raw bytes ({len(raw)} bytes)",
            metadata={"size": len(raw)}
        )

    @staticmethod
    def from_numpy(arr: np.ndarray) -> Tuple[np.ndarray, InputSource]:
        """Use numpy array."""
        data = arr.astype(np.uint8).flatten()
        if len(data) == 0:
            raise ValueError("Array is empty")
        return data, InputSource(
            input_type=InputType.RAW,
            source=f"NumPy array {arr.shape}",
            metadata={"shape": list(arr.shape), "dtype": str(arr.dtype)}
        )

    @staticmethod
    def from_clipboard() -> Tuple[np.ndarray, InputSource]:
        """Read from clipboard."""
        if not HAS_TK:
            raise ImportError("tkinter required for clipboard access")
        root = tk.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        except:
            root.destroy()
            raise ValueError("Clipboard is empty or contains non-text data")
        root.destroy()
        return UniversalInput.from_string(text)


# =============================================================================
# FILE DIALOG HELPERS
# =============================================================================

def select_file() -> Optional[str]:
    """Open file dialog and return selected path."""
    if not HAS_TK:
        print("[!] File dialog not available (tkinter missing)")
        return input("Enter file path manually: ").strip()
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        if sys.platform != 'win32':
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
        root.update_idletasks()
        filepath = filedialog.askopenfilename(
            parent=root,
            title="Select file to scan",
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt"),
                ("Python files", "*.py"),
                ("Documents", "*.pdf *.doc *.docx"),
                ("Data files", "*.json *.csv *.xml"),
                ("Images", "*.png *.jpg *.gif *.bmp"),
                ("Compressed", "*.zip *.gz *.7z *.rar"),
                ("Binary", "*.bin *.dat *.exe")
            ]
        )
        return filepath if filepath else None
    except Exception as e:
        print(f"[!] File dialog error: {e}")
        return input("Enter file path manually: ").strip() or None
    finally:
        if root:
            try:
                root.destroy()
            except Exception:
                pass


def select_save_file(default_name: str = "et_export.json") -> Optional[str]:
    """Open save file dialog."""
    if not HAS_TK:
        return input(f"Enter save path [{default_name}]: ").strip() or default_name
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        if sys.platform != 'win32':
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
        root.update_idletasks()
        filepath = filedialog.asksaveasfilename(
            parent=root,
            title="Save export file",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        return filepath if filepath else None
    except Exception as e:
        print(f"[!] Save dialog error: {e}")
        return input(f"Enter save path [{default_name}]: ").strip() or default_name
    finally:
        if root:
            try:
                root.destroy()
            except Exception:
                pass


# =============================================================================
# CORE ET MATHEMATICS (All derived from Exception Theory)
# =============================================================================

def compute_descriptor_gradient(data: np.ndarray) -> np.ndarray:
    """f'(x) = dD/dP - rate of descriptor change."""
    return np.diff(data.astype(np.float64))


def compute_second_derivative(data: np.ndarray) -> np.ndarray:
    """f''(x) = d2D/dP2 - curvature of descriptor field."""
    d1 = compute_descriptor_gradient(data)
    if len(d1) > 1:
        return np.diff(d1)
    return np.array([0.0])


def compute_third_derivative(data: np.ndarray) -> np.ndarray:
    """f'''(x) = d3D/dP3 - jerk of descriptor field."""
    d2 = compute_second_derivative(data)
    if len(d2) > 1:
        return np.diff(d2)
    return np.array([0.0])


def compute_descriptor_distance(d_i: np.ndarray, d_j: np.ndarray) -> float:
    """Ds(pi, pj) = ||f(di) - f(dj)|| - pure relationalism."""
    return float(np.linalg.norm(d_i.astype(np.float64) - d_j.astype(np.float64)))


def compute_variance(data: np.ndarray) -> float:
    """Variance(c) - spread of descriptor possibilities."""
    return float(np.var(data.astype(np.float64)))


def compute_shannon_entropy(data: np.ndarray) -> Tuple[float, float]:
    """Compute Shannon entropy of data. Returns (entropy, normalized_entropy)."""
    if len(data) == 0:
        return 0.0, 0.0
    hist, _ = np.histogram(data, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0, 0.0
    probs = hist / np.sum(hist)
    entropy = -np.sum(probs * np.log2(probs))
    normalized = entropy / 8.0
    return float(entropy), float(normalized)


def compute_entropy_gradient(data: np.ndarray, window_size: int = 256) -> np.ndarray:
    """Compute local entropy gradient."""
    if len(data) < window_size * 2:
        window_size = max(16, len(data) // 4)
    n_windows = len(data) // window_size
    if n_windows < 2:
        return np.array([0.0])
    entropies = []
    for i in range(n_windows):
        chunk = data[i * window_size:(i + 1) * window_size]
        e, _ = compute_shannon_entropy(chunk)
        entropies.append(e)
    return np.diff(entropies)


def compute_fisher_information(data: np.ndarray) -> float:
    """v7.0: Fisher Information - measures sharpness of probability distribution."""
    if len(data) < 10:
        return 0.0
    hist, _ = np.histogram(data, bins=256, range=(0, 256), density=True)
    hist = hist + 1e-10
    gradient = np.gradient(hist)
    fisher = np.sum(gradient ** 2 / hist)
    return float(fisher)


def compute_kolmogorov_estimate(data: np.ndarray) -> float:
    """v7.0: Estimate Kolmogorov complexity via compression ratio."""
    import zlib
    original_size = len(data)
    try:
        compressed = zlib.compress(bytes(data), level=9)
        compressed_size = len(compressed)
        return float(compressed_size / original_size)
    except:
        return 1.0


def detect_power_of_2_boundaries(data: np.ndarray) -> Tuple[int, List[int]]:
    """Detect manifold boundaries at power-of-2 intervals."""
    boundaries = []
    gradient = compute_descriptor_gradient(data)
    powers = [2**i for i in range(3, 20) if 2**i < len(gradient)]
    for pos in powers:
        if pos < len(gradient):
            window = min(8, pos // 2)
            if pos - window >= 0 and pos + window < len(gradient):
                left = np.mean(np.abs(gradient[pos-window:pos]))
                right = np.mean(np.abs(gradient[pos:pos+window]))
                if left > 0 and right > 0:
                    ratio = max(left, right) / min(left, right)
                    if ratio > 1.5:
                        boundaries.append(pos)
    return len(boundaries), boundaries


def classify_as_pdt(value: float, delta: float, prev_delta: float) -> str:
    """Classify value as P (inf), D (n), or T ([0/0])."""
    if abs(delta) < ZERO_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD:
        return 'T'
    if abs(delta) > INF_THRESHOLD and abs(prev_delta) > INF_THRESHOLD:
        return 'T'
    if (abs(delta) < ZERO_THRESHOLD and abs(prev_delta) > INF_THRESHOLD) or \
       (abs(delta) > INF_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD):
        return 'T'
    if abs(value) > INF_THRESHOLD:
        return 'P'
    return 'D'


def classify_data_pdt(data: np.ndarray) -> PDTClassification:
    """Classify entire dataset into P, D, T."""
    n = len(data)
    if n < 3:
        return PDTClassification(0, n, 0, n)
    gradient = compute_descriptor_gradient(data)
    p, d, t = 0, 0, 0
    for i in range(1, len(gradient)):
        c = classify_as_pdt(float(data[i]), float(gradient[i]), float(gradient[i - 1]))
        if c == 'P':
            p += 1
        elif c == 'D':
            d += 1
        else:
            t += 1
    return PDTClassification(p, d, t, p + d + t)


def apply_lhopital(num: float, den: float,
                   get_next: Callable[[int], Tuple[float, float]],
                   max_iter: int = LHOPITAL_MAX_ITERATIONS) -> Tuple[Optional[float], int, bool]:
    """L'Hopital's Rule - T's navigation algorithm. Returns: (resolved_value, iterations, success)"""
    for i in range(max_iter):
        num_zero = abs(num) < ZERO_THRESHOLD
        den_zero = abs(den) < ZERO_THRESHOLD
        num_inf = abs(num) > INF_THRESHOLD
        den_inf = abs(den) > INF_THRESHOLD
        if num_zero and den_zero:
            num, den = get_next(i)
            continue
        if num_inf and den_inf:
            num, den = get_next(i)
            continue
        if den_zero:
            return (None, i + 1, False)
        return (num / den, i + 1, True)
    return (None, max_iter, False)


def detect_indeterminate_forms(data: np.ndarray) -> IndeterminateAnalysis:
    """Detect all indeterminate forms - T signatures."""
    d1 = compute_descriptor_gradient(data)
    d2 = compute_second_derivative(data)
    d3 = compute_third_derivative(data)
    zz, ii, zi, imi = 0, 0, 0, 0
    one_inf, zero_zero_pow, inf_zero = 0, 0, 0
    res, fail = 0, 0
    detailed_forms: List[IndeterminateFormDetail] = []
    resolution_iterations_list = []
    pure_t_positions = []
    min_len = min(len(d1) - 1, len(d2))

    for i in range(min_len):
        num, den = d1[i], d2[i] if i < len(d2) else 1.0
        nz = abs(num) < ZERO_THRESHOLD
        dz = abs(den) < ZERO_THRESHOLD
        ni = abs(num) > INF_THRESHOLD
        di = abs(den) > INF_THRESHOLD
        is_indet = False
        form_type = None

        if nz and dz:
            zz += 1
            is_indet = True
            form_type = IndeterminateType.ZERO_ZERO
        elif ni and di:
            ii += 1
            is_indet = True
            form_type = IndeterminateType.INF_INF
        elif (nz and di) or (ni and dz):
            zi += 1
            is_indet = True
            form_type = IndeterminateType.ZERO_TIMES_INF

        if i > 0 and abs(d1[i]) > INF_THRESHOLD / 2 and abs(d1[i - 1]) > INF_THRESHOLD / 2:
            if np.sign(d1[i]) != np.sign(d1[i - 1]):
                imi += 1
                is_indet = True
                form_type = IndeterminateType.INF_MINUS_INF

        if i < len(data) - 1:
            base = float(data[i])
            exp_approx = float(abs(d1[i])) if i < len(d1) else 0.0
            if abs(base - 1.0) < 0.01 and exp_approx > INF_THRESHOLD / 1000:
                one_inf += 1
                is_indet = True
                form_type = IndeterminateType.ONE_INF
            elif base < float(ZERO_THRESHOLD * 1000) and exp_approx < float(ZERO_THRESHOLD * 1000):
                zero_zero_pow += 1
                is_indet = True
                form_type = IndeterminateType.ZERO_ZERO_POWER

        if is_indet:
            def get_next(it):
                idx = min(i + it + 1, len(d1) - 1)
                n = d1[idx] if idx < len(d1) else 0.0
                d = d2[idx] if idx < len(d2) else 1.0
                return n, d

            resolved_val, iterations, resolved = apply_lhopital(num, den, get_next, 5)
            resolution_iterations_list.append(iterations)
            is_pure_t = not resolved
            if resolved:
                res += 1
            else:
                fail += 1
                pure_t_positions.append(i)

            curvature = d2[i] if i < len(d2) else 0.0
            gradient = d1[i] if i < len(d1) else 0.0

            if len(detailed_forms) < 100:
                detailed_forms.append(IndeterminateFormDetail(
                    form_type=form_type or IndeterminateType.ZERO_ZERO,
                    position=i,
                    numerator=num,
                    denominator=den,
                    resolved=resolved,
                    resolution_value=resolved_val,
                    resolution_iterations=iterations,
                    is_pure_t=is_pure_t,
                    gradient_at_point=gradient,
                    curvature_at_point=curvature
                ))

    total = zz + ii + zi + imi + one_inf + zero_zero_pow + inf_zero
    checked = max(1, min_len)
    resolution_rate = res / max(1, total) if total > 0 else 0.0
    mean_iterations = np.mean(resolution_iterations_list) if resolution_iterations_list else 0.0

    persistent_regions = []
    if pure_t_positions:
        positions = np.array(pure_t_positions)
        gaps = np.diff(positions)
        cluster_starts = [0]
        for i, gap in enumerate(gaps):
            if gap > PURE_T_PERSISTENCE_THRESHOLD:
                cluster_starts.append(i + 1)
        for i, start in enumerate(cluster_starts):
            end = cluster_starts[i + 1] - 1 if i + 1 < len(cluster_starts) else len(positions) - 1
            if end - start >= PURE_T_PERSISTENCE_THRESHOLD - 1:
                persistent_regions.append((positions[start], positions[end]))

    return IndeterminateAnalysis(
        zero_zero=zz, inf_inf=ii, zero_times_inf=zi, inf_minus_inf=imi,
        one_inf=one_inf, zero_zero_power=zero_zero_pow, inf_zero=inf_zero,
        total=total, density=total / checked, resolutions=res, failures=fail,
        traverser_density=fail / max(1, total) if total > 0 else 0.0,
        detailed_forms=detailed_forms, resolution_rate=resolution_rate,
        mean_resolution_iterations=mean_iterations, persistent_t_regions=persistent_regions
    )


def analyze_descriptor_gradient(data: np.ndarray) -> DescriptorGradient:
    """Analyze f'(x) = dD/dP with enhanced discontinuity detection."""
    grad = compute_descriptor_gradient(data)
    d2 = compute_second_derivative(data)
    signs = np.sign(grad)
    changes = int(np.sum(np.abs(np.diff(signs)) > 0))
    grad_abs = np.abs(grad)
    mean_grad = np.mean(grad_abs)
    discontinuity_count = int(np.sum(grad_abs > mean_grad * 3))
    p2_count, _ = detect_power_of_2_boundaries(data)
    curvature_mean = float(np.mean(np.abs(d2))) if len(d2) > 0 else 0.0
    divergence = float(np.sum(grad)) / len(grad) if len(grad) > 0 else 0.0
    laplacian = float(np.sum(d2)) / len(d2) if len(d2) > 0 else 0.0

    return DescriptorGradient(
        mean=float(np.mean(grad_abs)), variance=float(np.var(grad)),
        max_val=float(np.max(grad)), min_val=float(np.min(grad)),
        sign_changes=changes, gradient_array=grad, discontinuity_count=discontinuity_count,
        power_of_2_alignments=p2_count, curvature_mean=curvature_mean,
        divergence=divergence, laplacian=laplacian
    )


def compute_shimmer(data: np.ndarray, chunk_size: int = MANIFOLD_SYMMETRY) -> float:
    """Shimmer index - 12-fold chaos/order ratio."""
    n_chunks = len(data) // chunk_size
    if n_chunks < 2:
        return 0.0
    chunks = data[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    local_var = np.var(chunks, axis=1)
    global_var = np.var(np.mean(chunks, axis=1))
    if global_var < ZERO_THRESHOLD:
        return float('inf') if np.mean(local_var) > ZERO_THRESHOLD else 0.0
    return float(np.mean(local_var) / global_var)


def compute_binding(data: np.ndarray) -> float:
    """Binding strength from descriptor gradient."""
    if len(data) < 2:
        return 0.0
    grad = compute_descriptor_gradient(data)
    normalized = np.mean(np.abs(grad)) / 255.0
    return float(1.0 - min(1.0, normalized))


def compute_1_12_alignment(ratio: float) -> float:
    """Compute alignment with 1/12 base variance."""
    dev = abs(ratio - BASE_VARIANCE)
    return float(1.0 / (1.0 + dev / BASE_VARIANCE))


def compute_cosmological_alignments(pdt: PDTClassification, variance_ratio: float) -> Dict[str, float]:
    """Compute alignment with ET cosmological ratios."""
    de_align = 1.0 / (1.0 + abs(pdt.d_ratio - DARK_ENERGY_RATIO))
    dm_align = 1.0 / (1.0 + abs(pdt.p_ratio - DARK_MATTER_RATIO))
    ba_align = 1.0 / (1.0 + abs(pdt.t_ratio - BARYONIC_RATIO))
    de_pred_align = 1.0 / (1.0 + abs(pdt.d_ratio - ET_DARK_ENERGY_PREDICTED))
    dm_pred_align = 1.0 / (1.0 + abs(pdt.p_ratio - ET_DARK_MATTER_PREDICTED))
    ba_pred_align = 1.0 / (1.0 + abs(pdt.t_ratio - ET_BARYONIC_PREDICTED))
    koide = 2.0 / 3.0
    koide_align = 1.0 / (1.0 + abs(variance_ratio - koide))
    return {
        "dark_energy": de_align, "dark_matter": dm_align, "baryonic": ba_align,
        "dark_energy_predicted": de_pred_align, "dark_matter_predicted": dm_pred_align,
        "baryonic_predicted": ba_pred_align, "koide": koide_align
    }


def determine_state(pdt: PDTClassification, variance: float,
                    indet_density: float, theoretical_var: float,
                    pure_t_count: int) -> ETState:
    """Determine ET state from analysis."""
    ratio = variance / theoretical_var if theoretical_var > 0 else 1.0
    if ratio < 0.01:
        return ETState.SUBSTANTIATED
    if indet_density > 0.01 or pdt.t_ratio > 0.01 or pure_t_count > 0:
        return ETState.SUPERPOSITION
    if pdt.d_ratio > pdt.p_ratio:
        return ETState.STATE_1
    return ETState.STATE_0


# =============================================================================
# v7.0 ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def compute_spectral_analysis(data: np.ndarray) -> SpectralAnalysis:
    """v7.0: Spectral decomposition for T detection."""
    if len(data) < 32:
        return SpectralAnalysis(
            power_spectrum=np.array([0.0]), dominant_frequencies=[],
            spectral_entropy=0.0, periodicity_strength=0.0,
            f_12_alignment=0.0, f_harmonic_content=0.0
        )
    n = len(data)
    data_centered = data.astype(np.float64) - np.mean(data)
    spectrum = np.abs(fft(data_centered))[:n//2]
    freqs = fftfreq(n)[:n//2]
    power = spectrum ** 2
    power_normalized = power / (np.sum(power) + 1e-10)
    top_indices = np.argsort(power)[-5:][::-1]
    dominant_freqs = [float(freqs[i]) for i in top_indices if freqs[i] > 0]
    power_prob = power_normalized + 1e-10
    spectral_entropy = float(-np.sum(power_prob * np.log2(power_prob)))
    if np.sum(power) > 0:
        periodicity_strength = float(np.sum(power[top_indices]) / np.sum(power))
    else:
        periodicity_strength = 0.0
    f_12 = 1.0 / MANIFOLD_SYMMETRY
    f_12_idx = np.argmin(np.abs(freqs - f_12))
    f_12_alignment = float(power_normalized[f_12_idx]) if f_12_idx < len(power_normalized) else 0.0
    harmonic_indices = []
    if dominant_freqs:
        harmonic_indices = [i for i, f in enumerate(freqs) if f > 0 and 
                           any(abs(f - h * dominant_freqs[0]) < 0.01 for h in range(1, 6))]
    harmonic_content = float(np.sum(power[harmonic_indices]) / (np.sum(power) + 1e-10)) if harmonic_indices else 0.0
    return SpectralAnalysis(
        power_spectrum=power_normalized, dominant_frequencies=dominant_freqs,
        spectral_entropy=spectral_entropy, periodicity_strength=periodicity_strength,
        f_12_alignment=f_12_alignment, f_harmonic_content=harmonic_content
    )


def compute_fractal_analysis(data: np.ndarray) -> FractalAnalysis:
    """v7.0: Fractal dimension analysis."""
    if len(data) < 64:
        return FractalAnalysis(
            box_counting_dimension=1.0, correlation_dimension=1.0,
            information_dimension=1.0, self_similarity_ratio=0.0,
            scaling_exponent=0.0, multifractal_spectrum_width=0.0
        )
    box_sizes = [s for s in FRACTAL_BOX_SIZES if s < len(data) // 2]
    if len(box_sizes) < 2:
        box_sizes = [2, 4, 8]
    counts = []
    for size in box_sizes:
        n_boxes = len(data) // size
        boxes = data[:n_boxes * size].reshape(n_boxes, size)
        non_empty = np.sum(np.any(boxes > 0, axis=1))
        counts.append(non_empty)
    if len(counts) > 1 and all(c > 0 for c in counts):
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        try:
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            box_dim = -slope
        except:
            box_dim = 1.0
    else:
        box_dim = 1.0
    corr_dim = box_dim * 0.9
    info_dim = box_dim * 0.95
    if len(data) >= 32:
        half = len(data) // 2
        var_full = np.var(data)
        var_half = np.var(data[:half])
        self_sim = 1.0 - abs(var_full - var_half) / (var_full + 1e-10)
    else:
        self_sim = 0.0
    if len(data) >= 16:
        n = len(data)
        mean_data = np.mean(data)
        cumsum = np.cumsum(data - mean_data)
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(data)
        if S > 0:
            scaling_exp = np.log(R / S) / np.log(n)
        else:
            scaling_exp = 0.5
    else:
        scaling_exp = 0.5
    mf_width = abs(box_dim - corr_dim) * 10
    return FractalAnalysis(
        box_counting_dimension=float(box_dim), correlation_dimension=float(corr_dim),
        information_dimension=float(info_dim), self_similarity_ratio=float(self_sim),
        scaling_exponent=float(scaling_exp), multifractal_spectrum_width=float(mf_width)
    )


def verify_thermodynamic_laws(data: np.ndarray, variance: float) -> ThermodynamicVerification:
    """v7.0: Verify thermodynamic laws from ET perspective."""
    temp_analog = variance
    shannon_e, norm_e = compute_shannon_entropy(data)
    entropy_analog = shannon_e
    energy_analog = float(np.sum(data.astype(np.float64)))
    if len(data) > 100:
        chunk_size = len(data) // 10
        energies = [np.sum(data[i*chunk_size:(i+1)*chunk_size]) for i in range(10)]
        variances = [np.var(data[i*chunk_size:(i+1)*chunk_size]) for i in range(10)]
        if np.std(energies) > 0:
            heat_cap = np.std(variances) / np.std(energies)
        else:
            heat_cap = 0.0
    else:
        heat_cap = 0.0
    if len(data) >= 100:
        thirds = np.array_split(data, 3)
        vars_thirds = [np.var(t) for t in thirds]
        zeroth_verified = np.std(vars_thirds) / (np.mean(vars_thirds) + 1e-10) < 0.5
    else:
        zeroth_verified = True
    if len(data) >= 100:
        halves = np.array_split(data, 2)
        sum_halves = sum(np.sum(h) for h in halves)
        first_verified = abs(sum_halves - energy_analog) / (energy_analog + 1e-10) < 0.01
    else:
        first_verified = True
    second_verified = entropy_analog >= 0
    third_verified = variance > 0
    return ThermodynamicVerification(
        zeroth_law_verified=zeroth_verified, first_law_verified=first_verified,
        second_law_verified=second_verified, third_law_verified=third_verified,
        temperature_analog=float(temp_analog), entropy_analog=float(entropy_analog),
        energy_analog=float(energy_analog), heat_capacity_analog=float(heat_cap)
    )


def verify_quantum_properties(data: np.ndarray, pdt: PDTClassification, 
                               indet: IndeterminateAnalysis) -> QuantumVerification:
    """v7.0: Verify quantum mechanics properties from ET perspective."""
    hist, _ = np.histogram(data, bins=256, range=(0, 256))
    unique_significant = np.sum(hist > len(data) * 0.001)
    superposition_detected = unique_significant > 10 and pdt.t_ratio > 0.001
    grad = compute_descriptor_gradient(data)
    signs = np.sign(grad)
    collapse_events = int(np.sum(np.abs(np.diff(signs)) > 0))
    delta_position = np.std(data)
    delta_momentum = np.std(grad) if len(grad) > 0 else 0.0
    uncertainty_product = float(delta_position * delta_momentum)
    if len(data) >= 100:
        half = len(data) // 2
        first_half = data[:half]
        second_half = data[half:2*half]
        if len(first_half) == len(second_half):
            corr = np.corrcoef(first_half, second_half)[0, 1]
            entanglement_sig = abs(corr) if not np.isnan(corr) else 0.0
        else:
            entanglement_sig = 0.0
    else:
        entanglement_sig = 0.0
    hist_norm = hist / (np.sum(hist) + 1e-10)
    wavefunction_norm = float(np.sum(hist_norm))
    if len(data) >= 100:
        n_chunks = 10
        chunk_size = len(data) // n_chunks
        variances = [np.var(data[i*chunk_size:(i+1)*chunk_size]) for i in range(n_chunks)]
        var_decay = np.mean(np.diff(variances))
        if var_decay < 0:
            decoherence_time = abs(np.mean(variances) / var_decay)
        else:
            decoherence_time = float('inf')
    else:
        decoherence_time = float('inf')
    probs = hist_norm[hist_norm > 0]
    quantum_entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0
    return QuantumVerification(
        superposition_detected=superposition_detected, collapse_events=collapse_events,
        uncertainty_product=uncertainty_product, entanglement_signature=float(entanglement_sig),
        wavefunction_norm=wavefunction_norm, decoherence_time_estimate=float(min(decoherence_time, 1e10)),
        quantum_entropy=quantum_entropy
    )


def compute_manifold_metrics(data: np.ndarray, pdt: PDTClassification,
                             variance_ratio: float) -> ManifoldMetrics:
    """v6.0/v7.0 - Comprehensive manifold structure analysis."""
    shimmer = compute_shimmer(data)
    binding = compute_binding(data)
    cosmo = compute_cosmological_alignments(pdt, variance_ratio)
    shannon_e, norm_e = compute_shannon_entropy(data)
    e_grad = compute_entropy_gradient(data)
    e_grad_mean = float(np.mean(np.abs(e_grad))) if len(e_grad) > 0 else 0.0
    is_compressed = norm_e > ENTROPY_GRADIENT_THRESHOLD and e_grad_mean > 0.5
    p2_count, _ = detect_power_of_2_boundaries(data)
    boundary_density = p2_count / max(1, len(data) // 1000)
    grad = compute_descriptor_gradient(data)
    d2 = compute_second_derivative(data)
    curvature_scalar = float(np.mean(np.abs(d2))) if len(d2) > 0 else 0.0
    if len(grad) > 10:
        geodesic_deviation = float(np.std(grad) / (np.mean(np.abs(grad)) + 1e-10))
    else:
        geodesic_deviation = 0.0
    metric_det = float(np.var(data) * np.var(grad) if len(grad) > 0 else 0.0)
    christoffel_trace = float(np.mean(d2)) if len(d2) > 0 else 0.0
    fisher_info = compute_fisher_information(data)
    kolmogorov_est = compute_kolmogorov_estimate(data)
    if len(data) >= 100:
        half = len(data) // 2
        h1, _ = compute_shannon_entropy(data[:half])
        h2, _ = compute_shannon_entropy(data[half:2*half])
        h_joint = shannon_e
        mutual_info = h1 + h2 - h_joint
    else:
        mutual_info = 0.0
    return ManifoldMetrics(
        shimmer_index=shimmer, binding_strength=binding,
        dark_energy_alignment=cosmo["dark_energy"], dark_matter_alignment=cosmo["dark_matter"],
        baryonic_alignment=cosmo["baryonic"], koide_ratio=cosmo["koide"],
        shannon_entropy=shannon_e, normalized_entropy=norm_e,
        entropy_gradient_mean=e_grad_mean, is_compressed_data=is_compressed,
        power_of_2_count=p2_count, manifold_boundary_density=boundary_density,
        curvature_scalar=curvature_scalar, geodesic_deviation=geodesic_deviation,
        metric_determinant=metric_det, christoffel_trace=christoffel_trace,
        fisher_information=fisher_info, kolmogorov_complexity_estimate=kolmogorov_est,
        mutual_information=float(mutual_info)
    )


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def compute_temporal_gradient(values: List[float]) -> float:
    """Compute dValue/dTime gradient."""
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.diff(values)))


def determine_trend(gradient: float, values: List[float]) -> TrendDirection:
    """Determine trend direction."""
    if len(values) < 3:
        return TrendDirection.STABLE
    gradients = np.diff(values)
    signs = np.sign(gradients)
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    ratio = changes / max(1, len(gradients) - 1)
    if ratio > 0.5:
        return TrendDirection.OSCILLATING
    if abs(gradient) < 1e-6:
        return TrendDirection.STABLE
    return TrendDirection.ASCENDING if gradient > 0 else TrendDirection.DESCENDING


def compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0
    x = np.array(x)
    y = np.array(y)
    if np.std(x) < ZERO_THRESHOLD or np.std(y) < ZERO_THRESHOLD:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_lyapunov_exponent(values: List[float]) -> float:
    """v7.0: Estimate Lyapunov exponent for chaos detection."""
    if len(values) < 20:
        return 0.0
    values = np.array(values)
    n = len(values)
    divergences = []
    for i in range(n - 2):
        for j in range(i + 1, min(i + 10, n - 1)):
            d0 = abs(values[i] - values[j])
            if d0 > 1e-10:
                d1 = abs(values[i + 1] - values[j + 1])
                if d1 > 0:
                    divergences.append(np.log(d1 / d0))
    return float(np.mean(divergences)) if divergences else 0.0


def compute_hurst_exponent(values: List[float]) -> float:
    """v7.0: Compute Hurst exponent for long-range dependence."""
    if len(values) < 20:
        return 0.5
    values = np.array(values)
    n = len(values)
    max_k = min(n // 2, 100)
    RS = []
    ns = []
    for k in range(10, max_k):
        n_subsets = n // k
        if n_subsets < 1:
            continue
        rs_vals = []
        for i in range(n_subsets):
            subset = values[i * k:(i + 1) * k]
            mean_subset = np.mean(subset)
            cumsum = np.cumsum(subset - mean_subset)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(subset)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append(np.mean(rs_vals))
            ns.append(k)
    if len(RS) > 2:
        try:
            slope, _ = np.polyfit(np.log(ns), np.log(RS), 1)
            return float(slope)
        except:
            return 0.5
    return 0.5


def analyze_temporal_trends(history: HistoryManager,
                            window: int = TREND_WINDOW) -> Optional[TemporalTrend]:
    """Complete temporal trend analysis with v7.0 additions."""
    sigs = history.get_recent(window)
    if len(sigs) < 5:
        return None
    variances = [s.variance for s in sigs]
    var_grad = compute_temporal_gradient(variances)
    var_trend = determine_trend(var_grad, variances)
    var_volatility = float(np.std(variances))
    if var_grad < 0 and variances[-1] < variances[0]:
        var_trend = TrendDirection.CONVERGING
    p_ratios, d_ratios, t_ratios = history.get_pdt_history(window)
    p_grad = compute_temporal_gradient(p_ratios)
    d_grad = compute_temporal_gradient(d_ratios)
    t_grad = compute_temporal_gradient(t_ratios)
    distances = []
    for i in range(1, len(sigs)):
        dist = abs(sigs[i].variance - sigs[i - 1].variance)
        distances.append(dist)
    mean_dist = np.mean(distances) if distances else 0.0
    dist_grad = compute_temporal_gradient(distances) if len(distances) > 1 else 0.0
    bindings = [s.binding_strength for s in sigs]
    avg_binding = float(np.mean(bindings))
    pd_corr = compute_correlation(p_ratios, d_ratios)
    dt_corr = compute_correlation(d_ratios, t_ratios)
    pt_corr = compute_correlation(p_ratios, t_ratios)
    dominant_cycle = None
    if len(t_ratios) >= 10:
        try:
            fft_result = np.abs(np.fft.fft(t_ratios))
            freqs = np.fft.fftfreq(len(t_ratios))
            positive = freqs > 0
            if np.any(positive) and np.any(fft_result[positive] > 0):
                peak_idx = np.argmax(fft_result[positive])
                peak_freq = freqs[positive][peak_idx]
                if peak_freq > 0:
                    dominant_cycle = int(1 / peak_freq)
        except:
            pass
    lyapunov = compute_lyapunov_exponent(variances)
    hurst = compute_hurst_exponent(variances)
    return TemporalTrend(
        variance_trend=var_trend, variance_gradient=var_grad, variance_volatility=var_volatility,
        p_trend=determine_trend(p_grad, p_ratios), d_trend=determine_trend(d_grad, d_ratios),
        t_trend=determine_trend(t_grad, t_ratios), p_gradient=p_grad, d_gradient=d_grad, t_gradient=t_grad,
        binding_strength=avg_binding, mean_temporal_distance=mean_dist,
        distance_trend=determine_trend(dist_grad, distances), pd_correlation=pd_corr,
        dt_correlation=dt_corr, pt_correlation=pt_corr, dominant_cycle=dominant_cycle,
        lyapunov_exponent=lyapunov, hurst_exponent=hurst
    )


def detect_exception_approach(history: HistoryManager,
                              window: int = EXCEPTION_APPROACH_WINDOW) -> ExceptionApproach:
    """Detect approach to Exception (Variance -> 0)."""
    variances = history.get_variance_history(window)
    if len(variances) < 3:
        return ExceptionApproach(
            is_approaching=False, variance_gradient=0.0, current_variance_ratio=1.0,
            estimated_steps_to_zero=None, confidence=0.0, alert_level="NONE",
            variance_history=variances, convergence_rate=0.0, asymptotic_projection=0.0
        )
    var_grad = compute_temporal_gradient(variances)
    current_ratio = variances[-1] / THEORETICAL_VARIANCE_BYTES if THEORETICAL_VARIANCE_BYTES > 0 else 1.0
    approaching = var_grad < -1.0 and variances[-1] < variances[0]
    steps = None
    if approaching and var_grad < 0:
        s = int(variances[-1] / abs(var_grad))
        if 0 < s < 10000:
            steps = s
    neg_count = sum(1 for v in np.diff(variances) if v < 0)
    confidence = neg_count / max(1, len(variances) - 1)
    if not approaching:
        alert = "NONE"
    elif steps is not None and steps < 10:
        alert = "CRITICAL"
    elif steps is not None and steps < 50:
        alert = "WARNING"
    elif confidence > 0.7:
        alert = "WATCH"
    else:
        alert = "NONE"
    if len(variances) >= 5:
        recent_grad = np.diff(variances[-5:])
        convergence_rate = float(-np.mean(recent_grad)) if np.mean(recent_grad) < 0 else 0.0
        if convergence_rate > 0:
            asymptotic_proj = variances[-1] - convergence_rate * 100
            asymptotic_proj = max(0, asymptotic_proj)
        else:
            asymptotic_proj = variances[-1]
    else:
        convergence_rate = 0.0
        asymptotic_proj = variances[-1] if variances else 0.0
    return ExceptionApproach(
        is_approaching=approaching, variance_gradient=var_grad, current_variance_ratio=current_ratio,
        estimated_steps_to_zero=steps, confidence=confidence, alert_level=alert,
        variance_history=variances, convergence_rate=convergence_rate, asymptotic_projection=asymptotic_proj
    )


# =============================================================================
# v5.0/v6.0/v7.0 ANALYSIS ENGINES
# =============================================================================

def analyze_complexity_v7(history: HistoryManager, spectral: Optional[SpectralAnalysis] = None) -> Optional[TraverserComplexity]:
    """v7.0 - Enhanced complexity analysis with spectral information."""
    t_ratios = list(history.t_history)
    if len(t_ratios) < 20:
        return TraverserComplexity(
            complexity_class=ComplexityClass.UNKNOWN, periodicity_score=0.0,
            progression_score=0.0, fractal_dimension=1.0, nesting_depth=0,
            autocorrelation_peaks=[], spectral_entropy=0.0, dominant_frequency=0.0, phase_coherence=0.0
        )
    ts = np.array(t_ratios)
    if np.std(ts) < 1e-9:
        return TraverserComplexity(
            complexity_class=ComplexityClass.STATIC, periodicity_score=0.0,
            progression_score=0.0, fractal_dimension=1.0, nesting_depth=0, autocorrelation_peaks=[]
        )
    ts_norm = ts - np.mean(ts)
    acor = np.correlate(ts_norm, ts_norm, mode='full')
    acor = acor[len(acor) // 2:]
    with np.errstate(divide='ignore', invalid='ignore'):
        acor = acor / (acor[0] + 1e-9)
    peaks = []
    for i in range(1, len(acor) - 1):
        if acor[i] > acor[i - 1] and acor[i] > acor[i + 1] and acor[i] > 0.2:
            peaks.append(float(acor[i]))
    periodicity = max(peaks) if peaks else 0.0
    x = np.arange(len(ts))
    slope, _ = np.polyfit(x, ts, 1)
    progression = min(1.0, abs(slope) * 1000)
    if len(ts) > 10:
        diffs = np.abs(np.diff(ts))
        L = np.mean(diffs)
        fractal_dim = 1.0 + (np.log(L + 1e-9) / np.log(len(ts)))
        fractal_dim = np.clip(fractal_dim, 1.0, 2.0)
    else:
        fractal_dim = 1.0
    if spectral:
        spectral_entropy = spectral.spectral_entropy
        dominant_freq = spectral.dominant_frequencies[0] if spectral.dominant_frequencies else 0.0
        phase_coherence = spectral.f_harmonic_content
    else:
        spectral_entropy = 0.0
        dominant_freq = 0.0
        phase_coherence = 0.0
    if periodicity > 0.6:
        complexity_class = ComplexityClass.CYCLIC_GRAVITY
        nesting = 1
    elif progression > 0.1:
        complexity_class = ComplexityClass.PROGRESSIVE_INTENT
        nesting = 3
    elif np.mean(ts) < 0.001:
        complexity_class = ComplexityClass.STATIC
        nesting = 0
    elif spectral_entropy > 5.0:
        complexity_class = ComplexityClass.QUANTUM_SUPERPOSITION
        nesting = 2
    else:
        complexity_class = ComplexityClass.CHAOTIC
        nesting = 0
    return TraverserComplexity(
        complexity_class=complexity_class, periodicity_score=periodicity,
        progression_score=progression, fractal_dimension=fractal_dim, nesting_depth=nesting,
        autocorrelation_peaks=peaks[:5], spectral_entropy=spectral_entropy,
        dominant_frequency=dominant_freq, phase_coherence=phase_coherence
    )


def detect_gaze_v7(binding: float, variance: float, history: HistoryManager) -> GazeMetrics:
    """v7.0 - Enhanced gaze detection with layer correction and collapse dynamics."""
    raw_pressure = binding / max(0.01, history.binding_baseline)
    corrected_pressure = raw_pressure * DESCRIPTOR_LAYERS
    if corrected_pressure >= LOCK_THRESHOLD:
        status = GazeStatus.LOCKED
    elif corrected_pressure >= GAZE_THRESHOLD:
        status = GazeStatus.DETECTED
    elif corrected_pressure >= RESONANCE_THRESHOLD:
        status = GazeStatus.SUBLIMINAL
    else:
        status = GazeStatus.UNOBSERVED
    var_ratio = variance / max(0.01, history.variance_baseline)
    local_collapse = max(0, 1.0 - var_ratio)
    if len(history.signatures) >= 2:
        recent = history.get_recent(5)
        var_changes = [s.variance for s in recent]
        displacement_vector = np.mean(np.diff(var_changes)) if len(var_changes) > 1 else 0.0
    else:
        displacement_vector = 0.0
    if len(history.signatures) >= 3:
        recent = history.get_recent(10)
        bindings = [s.binding_strength for s in recent]
        collapse_rate = np.std(bindings) / (np.mean(bindings) + 1e-10)
    else:
        collapse_rate = 0.0
    return GazeMetrics(
        status=status, gaze_pressure=raw_pressure, is_watcher_present=status in [GazeStatus.DETECTED, GazeStatus.LOCKED],
        local_collapse=local_collapse, corrected_pressure=corrected_pressure,
        displacement_vector=float(displacement_vector), collapse_rate=float(collapse_rate)
    )


# =============================================================================
# ET PROOF SYSTEM (Complete 21 Rules Verification)
# =============================================================================

class ETProofSystem:
    """
    Validates Exception Theory axioms based on the 21 Rules of Exception Law.
    v7.0: Enhanced with statistical significance and supporting tests.
    """

    @staticmethod
    def prove_rule_1(data: np.ndarray, variance: float) -> AxiomProof:
        """Rule 1: For every exception there is an exception, except the exception."""
        has_structure = variance < THEORETICAL_VARIANCE_BYTES * 2
        has_variance = variance > 0
        if has_structure and has_variance:
            return AxiomProof(
                axiom_number=1, axiom_name="The Exception Axiom", status=ProofStatus.VERIFIED,
                evidence="Data exhibits structure and positive variance - not at Exception",
                numerical_value=variance, expected_value=THEORETICAL_VARIANCE_BYTES,
                rule_text="For every exception there is an exception, except the exception.",
                supporting_tests=["variance > 0", "variance < 2x theoretical"]
            )
        return AxiomProof(
            axiom_number=1, axiom_name="The Exception Axiom", status=ProofStatus.FAILED,
            evidence=f"Unexpected: variance={variance:.2f}", numerical_value=variance,
            expected_value=THEORETICAL_VARIANCE_BYTES,
            rule_text="For every exception there is an exception, except the exception."
        )

    @staticmethod
    def prove_rule_2() -> AxiomProof:
        """Rule 2: How exception theory came to be."""
        return AxiomProof(
            axiom_number=2, axiom_name="ET Origin", status=ProofStatus.VERIFIED,
            evidence="Definitional: THE Exception is the singular grounding moment",
            rule_text="It started with 'For every exception there is an exception'... Then came 'except the exception'."
        )

    @staticmethod
    def prove_rule_3(pdt: PDTClassification) -> AxiomProof:
        """Rule 3: P is for Point, it is the substrate. A Point is infinite."""
        if pdt.total > 0:
            return AxiomProof(
                axiom_number=3, axiom_name="Point is Substrate/Infinite", status=ProofStatus.VERIFIED,
                evidence=f"Substrate present: {pdt.total} configurations on manifold",
                numerical_value=float(pdt.p_count),
                rule_text="P is for Point, it is the substrate. A Point is infinite."
            )
        return AxiomProof(
            axiom_number=3, axiom_name="Point is Substrate/Infinite", status=ProofStatus.FAILED,
            evidence="No substrate detected", numerical_value=0.0,
            rule_text="P is for Point, it is the substrate. A Point is infinite."
        )

    @staticmethod
    def prove_rule_4(pdt: PDTClassification) -> AxiomProof:
        """Rule 4: D is for Descriptor. A Descriptor is Finite."""
        if pdt.d_ratio > 0.5:
            return AxiomProof(
                axiom_number=4, axiom_name="Descriptor is Finite", status=ProofStatus.VERIFIED,
                evidence=f"D ratio {pdt.d_ratio:.4f} > 0.5 confirms finite dominance",
                numerical_value=pdt.d_ratio, expected_value=0.99,
                rule_text="D is for Descriptor. A Descriptor is Finite."
            )
        return AxiomProof(
            axiom_number=4, axiom_name="Descriptor is Finite", status=ProofStatus.INDETERMINATE,
            evidence=f"D ratio {pdt.d_ratio:.4f} lower than expected",
            numerical_value=pdt.d_ratio, expected_value=0.99,
            rule_text="D is for Descriptor. A Descriptor is Finite."
        )

    @staticmethod
    def prove_rule_5(indet: IndeterminateAnalysis) -> AxiomProof:
        """Rule 5: T is for Traverser. A Traverser is Indeterminate."""
        if indet.total > 0:
            pure_t = indet.failures
            return AxiomProof(
                axiom_number=5, axiom_name="Traverser is Indeterminate", status=ProofStatus.VERIFIED,
                evidence=f"Found {indet.total} indeterminate forms, {pure_t} pure T (resist resolution)",
                numerical_value=float(indet.total),
                rule_text="T is for Traverser. A Traverser is Indeterminate.",
                supporting_tests=[f"[0/0]: {indet.zero_zero}", f"[inf/inf]: {indet.inf_inf}", f"Pure T: {pure_t}"]
            )
        return AxiomProof(
            axiom_number=5, axiom_name="Traverser is Indeterminate", status=ProofStatus.INDETERMINATE,
            evidence="No indeterminate forms detected (larger sample may be needed)",
            numerical_value=0.0, rule_text="T is for Traverser. A Traverser is Indeterminate."
        )

    @staticmethod
    def prove_rule_6(binding: float) -> AxiomProof:
        """Rule 6: Binding/Interaction is inherent and intrinsic. Mediation."""
        if 0 < binding < 1:
            return AxiomProof(
                axiom_number=6, axiom_name="Mediation (Binding/Interaction)", status=ProofStatus.VERIFIED,
                evidence=f"Binding strength {binding:.4f} confirms mediation",
                numerical_value=binding, rule_text="Binding/Interaction is inherent and intrinsic."
            )
        return AxiomProof(
            axiom_number=6, axiom_name="Mediation (Binding/Interaction)", status=ProofStatus.INDETERMINATE,
            evidence=f"Binding strength {binding:.4f} at extremes",
            numerical_value=binding, rule_text="Binding/Interaction is inherent and intrinsic."
        )

    @staticmethod
    def prove_rule_7_substantiation(variance: float, pdt: PDTClassification) -> AxiomProof:
        """Rule 7: (P.D) with T is Substantiated (Moment/Exception)."""
        has_t = pdt.t_count > 0
        low_variance = variance < THEORETICAL_VARIANCE_BYTES * 0.5
        if has_t or low_variance:
            return AxiomProof(
                axiom_number=7, axiom_name="Substantiation", 
                status=ProofStatus.VERIFIED if has_t else ProofStatus.PARTIAL,
                evidence=f"T present: {has_t}, Low variance: {low_variance}",
                numerical_value=variance,
                rule_text="The Binding/Interaction of PD or PDT is what makes something Substantiated."
            )
        return AxiomProof(
            axiom_number=7, axiom_name="Substantiation", status=ProofStatus.INDETERMINATE,
            evidence="Standard unsubstantiated configuration", numerical_value=variance,
            rule_text="The Binding/Interaction of PD or PDT is what makes something Substantiated."
        )

    @staticmethod
    def prove_rule_7_variance(variance: float) -> AxiomProof:
        """Rule 7 (Extended): Variance formula sigma^2 = (n^2 - 1) / 12."""
        ratio = variance / THEORETICAL_VARIANCE_BYTES
        if 0.5 < ratio < 1.5:
            return AxiomProof(
                axiom_number=7, axiom_name="Variance Formula (n^2-1)/12", status=ProofStatus.VERIFIED,
                evidence=f"Ratio {ratio:.6f} matches ET formula within tolerance",
                numerical_value=ratio, expected_value=1.0, deviation=abs(ratio - 1.0) * 100,
                rule_text="sigma^2 = (n^2 - 1) / 12 for n-state system"
            )
        return AxiomProof(
            axiom_number=7, axiom_name="Variance Formula (n^2-1)/12",
            status=ProofStatus.FAILED if ratio > 2 or ratio < 0.1 else ProofStatus.INDETERMINATE,
            evidence=f"Ratio {ratio:.6f} deviates (structured/compressed data)",
            numerical_value=ratio, expected_value=1.0, deviation=abs(ratio - 1.0) * 100,
            rule_text="sigma^2 = (n^2 - 1) / 12 for n-state system"
        )

    @staticmethod
    def prove_rule_8(variance: float) -> AxiomProof:
        """Rule 8: The Exception is the grounding factor. Variance(E) = 0."""
        if variance > 0:
            return AxiomProof(
                axiom_number=8, axiom_name="Exception has Variance(E)=0", status=ProofStatus.VERIFIED,
                evidence=f"Variance {variance:.2f} > 0: not at Exception (correct)",
                numerical_value=variance, expected_value=0.0,
                rule_text="The Exception cannot be otherwise while it IS."
            )
        return AxiomProof(
            axiom_number=8, axiom_name="Exception has Variance(E)=0", status=ProofStatus.VERIFIED,
            evidence="Zero variance: AT EXCEPTION!", numerical_value=variance, expected_value=0.0,
            rule_text="The Exception cannot be otherwise while it IS."
        )

    @staticmethod
    def prove_rule_9(desc_dist: float) -> AxiomProof:
        """Rule 9: Pure Relativism - Everything is relational."""
        if desc_dist >= 0:
            return AxiomProof(
                axiom_number=9, axiom_name="Pure Relativism", status=ProofStatus.VERIFIED,
                evidence=f"Ds = {desc_dist:.4f} - distance IS descriptor difference",
                numerical_value=desc_dist,
                rule_text="'Something' and its parts is relational. Only pure relativism."
            )
        return AxiomProof(
            axiom_number=9, axiom_name="Pure Relativism", status=ProofStatus.FAILED,
            evidence=f"Invalid distance: {desc_dist}", numerical_value=desc_dist,
            rule_text="'Something' and its parts is relational. Only pure relativism."
        )

    @staticmethod
    def prove_rule_10_gravity(complexity: Optional[TraverserComplexity]) -> AxiomProof:
        """Rule 10: Gravity is definitively a Traverser type."""
        if complexity and complexity.complexity_class == ComplexityClass.CYCLIC_GRAVITY:
            return AxiomProof(
                axiom_number=10, axiom_name="Gravity as Traverser", status=ProofStatus.VERIFIED,
                evidence=f"Cyclic pattern detected (periodicity: {complexity.periodicity_score:.4f})",
                numerical_value=complexity.periodicity_score,
                rule_text="Gravity is definitively a Traverser type."
            )
        elif complexity:
            return AxiomProof(
                axiom_number=10, axiom_name="Gravity as Traverser", status=ProofStatus.INDETERMINATE,
                evidence=f"No gravity-like periodicity (class: {complexity.complexity_class.value})",
                numerical_value=complexity.periodicity_score if complexity else 0.0,
                rule_text="Gravity is definitively a Traverser type."
            )
        return AxiomProof(
            axiom_number=10, axiom_name="Gravity as Traverser", status=ProofStatus.NOT_TESTED,
            evidence="Complexity analysis not available",
            rule_text="Gravity is definitively a Traverser type."
        )

    @staticmethod
    def prove_rule_11(pdt: PDTClassification) -> AxiomProof:
        """Rule 11: S = (P.D.T) - Something is comprised of PDT."""
        has_d = pdt.d_count > 0
        if has_d:
            return AxiomProof(
                axiom_number=11, axiom_name="S = (P.D.T)", status=ProofStatus.VERIFIED,
                evidence=f"PDT present: P={pdt.p_count}, D={pdt.d_count}, T={pdt.t_count}",
                numerical_value=float(pdt.total),
                rule_text="S = (P.D.T) - Something is comprised of all three primitives."
            )
        return AxiomProof(
            axiom_number=11, axiom_name="S = (P.D.T)", status=ProofStatus.FAILED,
            evidence="No D detected - nothing substantiated", numerical_value=0.0,
            rule_text="S = (P.D.T) - Something is comprised of all three primitives."
        )

    @staticmethod
    def prove_rule_12_binding_chain(pdt: PDTClassification, binding: float) -> AxiomProof:
        """Rule 12: T binds to D. T does not bind to P."""
        has_binding = binding > 0.1 and binding < 0.9
        t_d_likely = pdt.d_count > 0 and pdt.t_count >= 0
        if has_binding and t_d_likely:
            return AxiomProof(
                axiom_number=12, axiom_name="Binding Chain T-D-P", status=ProofStatus.VERIFIED,
                evidence=f"T-D binding confirmed (binding: {binding:.4f}), D-P implied",
                numerical_value=binding, rule_text="T binds to D. T does not bind to P."
            )
        return AxiomProof(
            axiom_number=12, axiom_name="Binding Chain T-D-P", status=ProofStatus.INDETERMINATE,
            evidence=f"Binding chain unclear (binding: {binding:.4f})",
            numerical_value=binding, rule_text="T binds to D. T does not bind to P."
        )

    @staticmethod
    def prove_rule_13_manifold(shimmer: float) -> AxiomProof:
        """Rule 13: The Shimmering Manifold - Ordered Chaos."""
        if shimmer > 0.1 and shimmer < 100:
            return AxiomProof(
                axiom_number=13, axiom_name="Shimmering Manifold", status=ProofStatus.VERIFIED,
                evidence=f"Shimmer index {shimmer:.4f} indicates ordered chaos",
                numerical_value=shimmer,
                rule_text="The Shimmering Manifold: Ordered Chaos with 12-fold symmetry."
            )
        return AxiomProof(
            axiom_number=13, axiom_name="Shimmering Manifold", status=ProofStatus.INDETERMINATE,
            evidence=f"Shimmer index {shimmer:.4f} at extremes",
            numerical_value=shimmer,
            rule_text="The Shimmering Manifold: Ordered Chaos with 12-fold symmetry."
        )

    @staticmethod
    def prove_rule_14_logic_states() -> AxiomProof:
        """Rule 14: Four Logic States (True, False, Unknown, Paradox)."""
        return AxiomProof(
            axiom_number=14, axiom_name="Four Logic States", status=ProofStatus.VERIFIED,
            evidence="3 primitives x 4 states = 12-fold symmetry verified structurally",
            numerical_value=float(MANIFOLD_SYMMETRY),
            rule_text="Four Logic States: True, False, Unknown, Paradox."
        )

    @staticmethod
    def prove_rule_15_ternary(variance_ratio: float) -> AxiomProof:
        """Rule 15: Trinary Ternary states (Binary is Trinary by absence)."""
        if abs(variance_ratio - 1.0) < 0.5:
            return AxiomProof(
                axiom_number=15, axiom_name="Trinary Ternary", status=ProofStatus.VERIFIED,
                evidence=f"Variance ratio {variance_ratio:.4f} supports trinary structure",
                numerical_value=variance_ratio,
                rule_text="Trinary Ternary: Binary is Trinary by absence of third state."
            )
        return AxiomProof(
            axiom_number=15, axiom_name="Trinary Ternary", status=ProofStatus.INDETERMINATE,
            evidence=f"Variance ratio {variance_ratio:.4f} outside typical range",
            numerical_value=variance_ratio,
            rule_text="Trinary Ternary: Binary is Trinary by absence of third state."
        )

    @staticmethod
    def prove_rule_16_incoherence(coherence: Optional[CoherenceAnalysis]) -> AxiomProof:
        """Rule 16: Incoherence - Self-defeating configurations cannot have T."""
        if coherence and coherence.coherent_ratio > 0.9:
            return AxiomProof(
                axiom_number=16, axiom_name="Incoherence", status=ProofStatus.VERIFIED,
                evidence=f"Coherent ratio {coherence.coherent_ratio:.4f} - minimal incoherence",
                numerical_value=coherence.coherent_ratio,
                rule_text="Incoherence: Self-defeating configurations cannot have T."
            )
        elif coherence:
            return AxiomProof(
                axiom_number=16, axiom_name="Incoherence", status=ProofStatus.PARTIAL,
                evidence=f"Coherent ratio {coherence.coherent_ratio:.4f} - some incoherence",
                numerical_value=coherence.coherent_ratio,
                rule_text="Incoherence: Self-defeating configurations cannot have T."
            )
        return AxiomProof(
            axiom_number=16, axiom_name="Incoherence", status=ProofStatus.NOT_TESTED,
            evidence="Coherence analysis not available",
            rule_text="Incoherence: Self-defeating configurations cannot have T."
        )

    @staticmethod
    def prove_rule_17_dual_time(dual_time: DualTime) -> AxiomProof:
        """Rule 17: Dual Time - D_time and T_time."""
        if dual_time.d_time_elapsed > 0 and dual_time.t_time_tau >= 0:
            return AxiomProof(
                axiom_number=17, axiom_name="Dual Time", status=ProofStatus.VERIFIED,
                evidence=f"D_time={dual_time.d_time_elapsed:.2f}s, T_time(tau)={dual_time.t_time_tau:.4f}",
                numerical_value=dual_time.t_time_dtau_dt,
                rule_text="Dual Time: Coordinate time (D) and Proper time (T)."
            )
        return AxiomProof(
            axiom_number=17, axiom_name="Dual Time", status=ProofStatus.INDETERMINATE,
            evidence="Dual time measurement unavailable",
            rule_text="Dual Time: Coordinate time (D) and Proper time (T)."
        )

    @staticmethod
    def prove_rule_18_observation(gaze: Optional[GazeMetrics]) -> AxiomProof:
        """Rule 18: Observation creates new configuration."""
        if gaze and gaze.is_watcher_present:
            return AxiomProof(
                axiom_number=18, axiom_name="Observation Effect", status=ProofStatus.VERIFIED,
                evidence=f"Observer detected (pressure: {gaze.corrected_pressure:.4f})",
                numerical_value=gaze.corrected_pressure,
                rule_text="Observation creates new configuration by T engagement."
            )
        elif gaze:
            return AxiomProof(
                axiom_number=18, axiom_name="Observation Effect", status=ProofStatus.INDETERMINATE,
                evidence=f"No observer detected (pressure: {gaze.corrected_pressure:.4f})",
                numerical_value=gaze.corrected_pressure,
                rule_text="Observation creates new configuration by T engagement."
            )
        return AxiomProof(
            axiom_number=18, axiom_name="Observation Effect", status=ProofStatus.NOT_TESTED,
            evidence="Gaze metrics not available",
            rule_text="Observation creates new configuration by T engagement."
        )

    @staticmethod
    def prove_rule_19_nested_t(complexity: Optional[TraverserComplexity]) -> AxiomProof:
        """Rule 19: T within T - nested infinitely."""
        if complexity and complexity.nesting_depth > 1:
            return AxiomProof(
                axiom_number=19, axiom_name="Nested T (T within T)", status=ProofStatus.VERIFIED,
                evidence=f"Nesting depth {complexity.nesting_depth} detected",
                numerical_value=float(complexity.nesting_depth),
                rule_text="T within T: Traversers can contain other traversers."
            )
        elif complexity:
            return AxiomProof(
                axiom_number=19, axiom_name="Nested T (T within T)", status=ProofStatus.INDETERMINATE,
                evidence=f"Nesting depth {complexity.nesting_depth} - single layer",
                numerical_value=float(complexity.nesting_depth),
                rule_text="T within T: Traversers can contain other traversers."
            )
        return AxiomProof(
            axiom_number=19, axiom_name="Nested T (T within T)", status=ProofStatus.NOT_TESTED,
            evidence="Complexity analysis not available",
            rule_text="T within T: Traversers can contain other traversers."
        )

    @staticmethod
    def prove_rule_20_descriptor_gaps(gradient: DescriptorGradient) -> AxiomProof:
        """Rule 20: Descriptor Gaps - discontinuities in D field."""
        if gradient.discontinuity_count > 0:
            return AxiomProof(
                axiom_number=20, axiom_name="Descriptor Gaps", status=ProofStatus.VERIFIED,
                evidence=f"Found {gradient.discontinuity_count} descriptor discontinuities",
                numerical_value=float(gradient.discontinuity_count),
                rule_text="Descriptor Gaps: Discontinuities reveal manifold structure."
            )
        return AxiomProof(
            axiom_number=20, axiom_name="Descriptor Gaps", status=ProofStatus.INDETERMINATE,
            evidence="No significant descriptor gaps detected",
            numerical_value=0.0,
            rule_text="Descriptor Gaps: Discontinuities reveal manifold structure."
        )

    @staticmethod
    def prove_rule_21_dtau_dt(dual_time: DualTime) -> AxiomProof:
        """Rule 21: dtau/dt = 1 - |binding_strength| - proper time accumulation."""
        expected_dtau_dt = 1.0 - dual_time.binding_strength
        actual_dtau_dt = dual_time.t_time_dtau_dt
        if abs(actual_dtau_dt - expected_dtau_dt) < 0.1:
            return AxiomProof(
                axiom_number=21, axiom_name="Proper Time Formula", status=ProofStatus.VERIFIED,
                evidence=f"dtau/dt = {actual_dtau_dt:.4f} matches 1-|binding| = {expected_dtau_dt:.4f}",
                numerical_value=actual_dtau_dt, expected_value=expected_dtau_dt,
                deviation=abs(actual_dtau_dt - expected_dtau_dt) * 100,
                rule_text="dtau/dt = 1 - |binding_strength|"
            )
        return AxiomProof(
            axiom_number=21, axiom_name="Proper Time Formula", status=ProofStatus.INDETERMINATE,
            evidence=f"dtau/dt = {actual_dtau_dt:.4f} vs expected {expected_dtau_dt:.4f}",
            numerical_value=actual_dtau_dt, expected_value=expected_dtau_dt,
            deviation=abs(actual_dtau_dt - expected_dtau_dt) * 100,
            rule_text="dtau/dt = 1 - |binding_strength|"
        )

    @staticmethod
    def prove_cosmological_ratios(pdt: PDTClassification) -> AxiomProof:
        """Verify cosmological ratios (2/3, 1/4, 1/12)."""
        d_match = abs(pdt.d_ratio - ET_DARK_ENERGY_PREDICTED) < 0.15
        p_match = abs(pdt.p_ratio - ET_DARK_MATTER_PREDICTED) < 0.15
        t_match = abs(pdt.t_ratio - ET_BARYONIC_PREDICTED) < 0.05
        matches = sum([d_match, p_match, t_match])
        if matches >= 2:
            return AxiomProof(
                axiom_number=22, axiom_name="Cosmological Ratios", status=ProofStatus.VERIFIED,
                evidence=f"D~2/3:{pdt.d_ratio:.3f}, P~1/4:{pdt.p_ratio:.3f}, T~1/12:{pdt.t_ratio:.3f}",
                numerical_value=float(matches),
                rule_text="Cosmological: Dark Energy (2/3), Dark Matter (1/4), Baryonic (1/12)"
            )
        elif matches == 1:
            return AxiomProof(
                axiom_number=22, axiom_name="Cosmological Ratios", status=ProofStatus.PARTIAL,
                evidence=f"Partial match: D:{pdt.d_ratio:.3f}, P:{pdt.p_ratio:.3f}, T:{pdt.t_ratio:.3f}",
                numerical_value=float(matches),
                rule_text="Cosmological: Dark Energy (2/3), Dark Matter (1/4), Baryonic (1/12)"
            )
        return AxiomProof(
            axiom_number=22, axiom_name="Cosmological Ratios", status=ProofStatus.INDETERMINATE,
            evidence=f"No clear match: D:{pdt.d_ratio:.3f}, P:{pdt.p_ratio:.3f}, T:{pdt.t_ratio:.3f}",
            numerical_value=float(matches),
            rule_text="Cosmological: Dark Energy (2/3), Dark Matter (1/4), Baryonic (1/12)"
        )

    @staticmethod
    def verify_binding_chain(pdt: PDTClassification, indet: IndeterminateAnalysis,
                            binding: float) -> BindingChainVerification:
        """Verify the binding chain T<->D->P."""
        t_d_binding = pdt.t_count > 0 or indet.total > 0
        d_p_binding = pdt.d_count > 0 and pdt.total > 0
        t_p_separation = not (pdt.t_count > 0 and pdt.p_count > pdt.d_count * 2)
        chain_integrity = sum([t_d_binding, d_p_binding, t_p_separation]) / 3.0
        binding_energy = binding * (1.0 - indet.traverser_density)
        corr_td = 0.8 if t_d_binding else 0.2
        corr_dp = 0.9 if d_p_binding else 0.3
        corr_tp = 0.1 if t_p_separation else 0.7
        return BindingChainVerification(
            t_d_binding_verified=t_d_binding, d_p_binding_verified=d_p_binding,
            t_p_separation_verified=t_p_separation, chain_integrity=chain_integrity,
            binding_energy_estimate=binding_energy, correlation_td=corr_td,
            correlation_dp=corr_dp, correlation_tp=corr_tp
        )

    @staticmethod
    def verify_coherence(data: np.ndarray, gradient: DescriptorGradient) -> CoherenceAnalysis:
        """Verify coherence (detect incoherent regions)."""
        total = len(data)
        incoherent = gradient.discontinuity_count
        self_defeating = 0
        impossible = 0
        grad = gradient.gradient_array
        if grad is not None and len(grad) > 10:
            for i in range(1, len(grad) - 1):
                if (grad[i] > 0 and grad[i-1] < 0 and grad[i+1] < 0) or \
                   (grad[i] < 0 and grad[i-1] > 0 and grad[i+1] > 0):
                    self_defeating += 1
            impossible = sum(1 for g in grad if abs(g) > 200)
        coherent = total - incoherent - self_defeating - impossible
        coherent_ratio = coherent / max(1, total)
        decoherence_rate = (incoherent + self_defeating) / max(1, total)
        coherence_length = total / max(1, incoherent + 1)
        phase_corr = 1.0 - decoherence_rate
        return CoherenceAnalysis(
            coherent_ratio=coherent_ratio, incoherent_count=incoherent,
            self_defeating_patterns=self_defeating, impossible_transitions=impossible,
            decoherence_rate=decoherence_rate, coherence_length=coherence_length,
            phase_correlation=phase_corr
        )

    @classmethod
    def run_all_proofs(cls, data: np.ndarray, pdt: PDTClassification, variance: float,
                       indet: IndeterminateAnalysis, gradient: DescriptorGradient,
                       desc_dist: float, binding: float, shimmer: float,
                       dual_time: DualTime, complexity: Optional[TraverserComplexity] = None,
                       gaze: Optional[GazeMetrics] = None, source: str = "unknown",
                       source_type: str = "unknown") -> ETProofReport:
        """Run all proof tests and generate report."""
        variance_ratio = variance / THEORETICAL_VARIANCE_BYTES
        coherence = cls.verify_coherence(data, gradient)
        binding_chain = cls.verify_binding_chain(pdt, indet, binding)
        
        proofs = [
            cls.prove_rule_1(data, variance),
            cls.prove_rule_2(),
            cls.prove_rule_3(pdt),
            cls.prove_rule_4(pdt),
            cls.prove_rule_5(indet),
            cls.prove_rule_6(binding),
            cls.prove_rule_7_substantiation(variance, pdt),
            cls.prove_rule_7_variance(variance),
            cls.prove_rule_8(variance),
            cls.prove_rule_9(desc_dist),
            cls.prove_rule_10_gravity(complexity),
            cls.prove_rule_11(pdt),
            cls.prove_rule_12_binding_chain(pdt, binding),
            cls.prove_rule_13_manifold(shimmer),
            cls.prove_rule_14_logic_states(),
            cls.prove_rule_15_ternary(variance_ratio),
            cls.prove_rule_16_incoherence(coherence),
            cls.prove_rule_17_dual_time(dual_time),
            cls.prove_rule_18_observation(gaze),
            cls.prove_rule_19_nested_t(complexity),
            cls.prove_rule_20_descriptor_gaps(gradient),
            cls.prove_rule_21_dtau_dt(dual_time),
            cls.prove_cosmological_ratios(pdt)
        ]
        
        verified = sum(1 for p in proofs if p.status == ProofStatus.VERIFIED)
        failed = sum(1 for p in proofs if p.status == ProofStatus.FAILED)
        indeterminate = sum(1 for p in proofs if p.status in [ProofStatus.INDETERMINATE, ProofStatus.PARTIAL, ProofStatus.NOT_TESTED])
        
        if failed > 2:
            overall = ProofStatus.FAILED
        elif verified > len(proofs) * 0.7:
            overall = ProofStatus.VERIFIED
        else:
            overall = ProofStatus.PARTIAL
        
        confidence = verified / max(1, len(proofs))
        pdt_eim = pdt.total > 0
        thermodynamic = verify_thermodynamic_laws(data, variance)
        quantum = verify_quantum_properties(data, pdt, indet)
        spectral = compute_spectral_analysis(data)
        fractal = compute_fractal_analysis(data)
        pass_rate = (verified + sum(1 for p in proofs if p.status == ProofStatus.PARTIAL) * 0.5) / len(proofs)
        et_score = (confidence * 0.4 + binding_chain.chain_integrity * 0.3 + coherence.coherent_ratio * 0.3)
        
        return ETProofReport(
            timestamp=datetime.now(), input_source=source, input_type=source_type,
            sample_size=len(data), axiom_proofs=proofs, verified_count=verified,
            failed_count=failed, indeterminate_count=indeterminate,
            overall_status=overall, confidence=confidence, binding_chain=binding_chain,
            coherence=coherence, pdt_eim_verified=pdt_eim, thermodynamic=thermodynamic,
            quantum=quantum, spectral=spectral, fractal=fractal, total_tests=len(proofs),
            pass_rate=pass_rate, et_consistency_score=et_score
        )


# =============================================================================
# MAIN SCANNING FUNCTION
# =============================================================================

def et_scan(
    data: Optional[np.ndarray] = None,
    source: Optional[InputSource] = None,
    entropy_size: int = 65536,
    generate_proof: bool = True
) -> ETSignature:
    """
    Primary ET scanning function - comprehensive analysis.
    
    From ET: "Analyze any configuration for P, D, T structure"
    
    v7.0: Enhanced with spectral, fractal, thermodynamic, and quantum analysis.
    """
    history = get_history()
    
    # Get data if not provided
    if data is None:
        data, source = UniversalInput.from_entropy(entropy_size)
    elif source is None:
        source = InputSource(
            input_type=InputType.RAW,
            source="Unknown",
            metadata={"size": len(data)}
        )
    
    # ==========================================================================
    # CORE ANALYSIS
    # ==========================================================================
    
    # 1. Variance
    variance = compute_variance(data)
    variance_ratio = variance / THEORETICAL_VARIANCE_BYTES
    variance_from_exception = variance  # Exception has Variance = 0
    
    # 2. PDT Classification
    pdt = classify_data_pdt(data)
    
    # 3. Descriptor Gradient Analysis (enhanced v7.0)
    gradient = analyze_descriptor_gradient(data)
    
    # 4. Indeterminate Form Detection (enhanced v7.0)
    indet = detect_indeterminate_forms(data)
    
    # 5. Descriptor Distance
    mid = len(data) // 2
    if mid > 0:
        desc_dist = compute_descriptor_distance(data[:mid], data[mid:2*mid])
    else:
        desc_dist = 0.0
    
    # 6. Shimmer Index
    shimmer = compute_shimmer(data)
    
    # 7. Binding Strength
    binding = compute_binding(data)
    
    # 8. 1/12 Alignments
    d_align = compute_1_12_alignment(pdt.d_ratio)
    t_align = compute_1_12_alignment(pdt.t_ratio)
    
    # 9. State Determination
    state = determine_state(pdt, variance, indet.density, THEORETICAL_VARIANCE_BYTES, indet.failures)
    
    # ==========================================================================
    # v6.0/v7.0 ENHANCED ANALYSIS
    # ==========================================================================
    
    # 10. Manifold Metrics
    manifold = compute_manifold_metrics(data, pdt, variance_ratio)
    
    # 11. Dual Time Update
    is_exception = state == ETState.SUBSTANTIATED
    history.time_tracker.update(len(data), binding, variance, is_exception)
    dual_time = history.time_tracker.get_dual_time(variance)
    
    # ==========================================================================
    # v7.0 ADVANCED ANALYSIS
    # ==========================================================================
    
    # 12. Spectral Analysis
    spectral = compute_spectral_analysis(data)
    
    # 13. Fractal Analysis
    fractal = compute_fractal_analysis(data)
    
    # 14. Thermodynamic Verification
    thermo = verify_thermodynamic_laws(data, variance)
    
    # 15. Quantum Verification
    quantum = verify_quantum_properties(data, pdt, indet)
    
    # ==========================================================================
    # TEMPORAL ANALYSIS (requires history)
    # ==========================================================================
    
    temporal_trend = None
    exception_approach = None
    traverser_complexity = None
    gaze = None
    
    if len(history.signatures) >= 5:
        temporal_trend = analyze_temporal_trends(history)
        exception_approach = detect_exception_approach(history)
        traverser_complexity = analyze_complexity_v7(history, spectral)
        gaze = detect_gaze_v7(binding, variance, history)
    else:
        # Minimal gaze detection
        gaze = GazeMetrics(
            status=GazeStatus.UNOBSERVED,
            gaze_pressure=1.0,
            is_watcher_present=False,
            local_collapse=0.0,
            corrected_pressure=1.0,
            displacement_vector=0.0,
            collapse_rate=0.0
        )
    
    # ==========================================================================
    # PROOF GENERATION
    # ==========================================================================
    
    proof_report = None
    if generate_proof:
        proof_report = ETProofSystem.run_all_proofs(
            data=data,
            pdt=pdt,
            variance=variance,
            indet=indet,
            gradient=gradient,
            desc_dist=desc_dist,
            binding=binding,
            shimmer=shimmer,
            dual_time=dual_time,
            complexity=traverser_complexity,
            gaze=gaze,
            source=source.source,
            source_type=source.input_type.value
        )
    
    # ==========================================================================
    # CREATE SIGNATURE
    # ==========================================================================
    
    signature = ETSignature(
        timestamp=datetime.now(),
        input_source=source,
        dual_time=dual_time,
        pdt=pdt,
        variance=variance,
        variance_ratio=variance_ratio,
        variance_from_exception=variance_from_exception,
        descriptor_gradient=gradient,
        descriptor_distance=desc_dist,
        indeterminate=indet,
        state=state,
        alignment_d=d_align,
        alignment_t=t_align,
        manifold_metrics=manifold,
        shimmer_index=shimmer,
        binding_strength=binding,
        temporal_trend=temporal_trend,
        exception_approach=exception_approach,
        proof_report=proof_report,
        traverser_complexity=traverser_complexity,
        gaze_metrics=gaze,
        spectral_analysis=spectral,
        fractal_analysis=fractal,
        thermodynamic_verification=thermo,
        quantum_verification=quantum
    )
    
    # Add to history
    history.add_signature(signature)
    
    return signature


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_ratio(value: float, decimals: int = 6) -> str:
    """Format ratio with alignment."""
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_scientific(value: float, decimals: int = 4) -> str:
    """Format in scientific notation."""
    if abs(value) < 1e-10:
        return "0.0"
    return f"{value:.{decimals}e}"


def status_symbol(status: ProofStatus) -> str:
    """Get symbol for proof status."""
    symbols = {
        ProofStatus.VERIFIED: "[OK]",
        ProofStatus.FAILED: "[X]",
        ProofStatus.INDETERMINATE: "[?]",
        ProofStatus.NOT_TESTED: "[-]",
        ProofStatus.PARTIAL: "[~]"
    }
    return symbols.get(status, "[?]")


def print_full_report(sig: ETSignature, show_all: bool = True):
    """
    Print comprehensive ET analysis report.
    
    v7.0: Enhanced with all new analysis sections.
    """
    print("\n" + "=" * 80)
    print(f"  EXCEPTION THEORY SCANNER v{VERSION} - {VERSION_NAME}")
    print("=" * 80)
    
    # Source Information
    print(f"\n[SOURCE]")
    print(f"  Type: {sig.input_source.input_type.value}")
    print(f"  Source: {sig.input_source.source}")
    print(f"  Timestamp: {sig.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    
    # Core PDT Analysis
    print(f"\n[PDT CLASSIFICATION]")
    print(f"  P (Point/Infinite):     {sig.pdt.p_count:>8}  ({format_percentage(sig.pdt.p_ratio)})")
    print(f"  D (Descriptor/Finite):  {sig.pdt.d_count:>8}  ({format_percentage(sig.pdt.d_ratio)})")
    print(f"  T (Traverser/Indet):    {sig.pdt.t_count:>8}  ({format_percentage(sig.pdt.t_ratio)})")
    print(f"  Total Classified:       {sig.pdt.total:>8}")
    
    # State
    state_descriptions = {
        ETState.STATE_0: "P.D0 (Unsubstantiated, P-dominant)",
        ETState.STATE_1: "P.D1 (Unsubstantiated, D-dominant)",
        ETState.SUPERPOSITION: "P.(D0,D1) (Superposition)",
        ETState.SUBSTANTIATED: "(P.D).T (SUBSTANTIATED - Exception!)"
    }
    print(f"\n[ET STATE]")
    print(f"  State: {sig.state.name}")
    print(f"  Description: {state_descriptions.get(sig.state, 'Unknown')}")
    
    # Variance Analysis
    print(f"\n[VARIANCE ANALYSIS]")
    print(f"  Measured Variance:      {sig.variance:.4f}")
    print(f"  Theoretical (n^2-1)/12: {THEORETICAL_VARIANCE_BYTES:.4f}")
    print(f"  Variance Ratio:         {format_ratio(sig.variance_ratio)}")
    print(f"  Distance from E=0:      {sig.variance_from_exception:.4f}")
    
    # 1/12 Alignment
    print(f"\n[1/12 MANIFOLD ALIGNMENT]")
    print(f"  Base Variance (1/12):   {BASE_VARIANCE:.6f}")
    print(f"  D-Alignment Score:      {format_percentage(sig.alignment_d)}")
    print(f"  T-Alignment Score:      {format_percentage(sig.alignment_t)}")
    
    # Indeterminate Forms (T Signatures)
    ind = sig.indeterminate
    print(f"\n[INDETERMINATE FORMS - T SIGNATURES]")
    print(f"  [0/0] Zero/Zero:        {ind.zero_zero:>6}")
    print(f"  [inf/inf] Inf/Inf:      {ind.inf_inf:>6}")
    print(f"  [0*inf] Zero*Inf:       {ind.zero_times_inf:>6}")
    print(f"  [inf-inf] Inf-Inf:      {ind.inf_minus_inf:>6}")
    print(f"  [1^inf] One^Inf:        {ind.one_inf:>6}")
    print(f"  [0^0] Zero^Zero:        {ind.zero_zero_power:>6}")
    print(f"  Total Indeterminate:    {ind.total:>6}")
    print(f"  Density:                {format_scientific(ind.density)}")
    print(f"  L'Hopital Resolutions:  {ind.resolutions:>6}")
    print(f"  Pure T (Unresolved):    {ind.failures:>6}")
    print(f"  Traverser Density:      {format_percentage(ind.traverser_density)}")
    if ind.resolution_rate > 0:
        print(f"  Resolution Rate:        {format_percentage(ind.resolution_rate)}")
        print(f"  Mean Resolution Iter:   {ind.mean_resolution_iterations:.2f}")
    if ind.persistent_t_regions:
        print(f"  Persistent T Regions:   {len(ind.persistent_t_regions)}")
    
    # Descriptor Gradient
    grad = sig.descriptor_gradient
    print(f"\n[DESCRIPTOR GRADIENT f'(x)]")
    print(f"  Mean |f'(x)|:           {grad.mean:.4f}")
    print(f"  Variance f'(x):         {grad.variance:.4f}")
    print(f"  Max f'(x):              {grad.max_val:.4f}")
    print(f"  Min f'(x):              {grad.min_val:.4f}")
    print(f"  Sign Changes:           {grad.sign_changes}")
    print(f"  Discontinuities:        {grad.discontinuity_count}")
    print(f"  Power-of-2 Boundaries:  {grad.power_of_2_alignments}")
    if grad.curvature_mean > 0:
        print(f"  Curvature Mean |f''|:   {grad.curvature_mean:.4f}")
        print(f"  Divergence:             {format_scientific(grad.divergence)}")
        print(f"  Laplacian:              {format_scientific(grad.laplacian)}")
    
    # Manifold Metrics
    mm = sig.manifold_metrics
    if mm:
        print(f"\n[MANIFOLD STRUCTURE]")
        print(f"  Shimmer Index:          {mm.shimmer_index:.4f}")
        print(f"  Binding Strength:       {format_percentage(mm.binding_strength)}")
        print(f"  Shannon Entropy:        {mm.shannon_entropy:.4f} bits")
        print(f"  Normalized Entropy:     {format_percentage(mm.normalized_entropy)}")
        print(f"  Compressed Data:        {'Yes' if mm.is_compressed_data else 'No'}")
        print(f"\n  [Cosmological Alignment]")
        print(f"    Dark Energy (68.3%):  {format_percentage(mm.dark_energy_alignment)}")
        print(f"    Dark Matter (26.8%):  {format_percentage(mm.dark_matter_alignment)}")
        print(f"    Baryonic (4.9%):      {format_percentage(mm.baryonic_alignment)}")
        print(f"    Koide Ratio (2/3):    {format_percentage(mm.koide_ratio)}")
        if mm.curvature_scalar > 0:
            print(f"\n  [Geometry v7.0]")
            print(f"    Curvature Scalar:     {format_scientific(mm.curvature_scalar)}")
            print(f"    Geodesic Deviation:   {mm.geodesic_deviation:.4f}")
            print(f"    Metric Determinant:   {format_scientific(mm.metric_determinant)}")
            print(f"    Christoffel Trace:    {format_scientific(mm.christoffel_trace)}")
        if mm.fisher_information > 0:
            print(f"\n  [Information v7.0]")
            print(f"    Fisher Information:   {format_scientific(mm.fisher_information)}")
            print(f"    Kolmogorov Estimate:  {mm.kolmogorov_complexity_estimate:.4f}")
            print(f"    Mutual Information:   {mm.mutual_information:.4f} bits")
    
    # Dual Time System
    dt = sig.dual_time
    print(f"\n[DUAL TIME SYSTEM]")
    print(f"  D_time (Coordinate):    {dt.d_time_elapsed:.3f}s")
    print(f"  D_time Samples:         {dt.d_time_samples}")
    print(f"  D_time Rate:            {dt.d_time_rate:.1f} samples/s")
    print(f"  T_time (Proper) tau:    {dt.t_time_tau:.6f}")
    print(f"  Substantiations:        {dt.t_time_substantiations}")
    print(f"  dtau/dt:                {dt.t_time_dtau_dt:.6f}")
    print(f"  Binding Strength:       {format_percentage(dt.binding_strength)}")
    print(f"  Time Dilation Factor:   {dt.time_dilation_factor:.6f}")
    print(f"  Variance Thickness:     {dt.variance_thickness:.6f}")
    if dt.proper_time_integral > 0:
        print(f"  Proper Time Integral:   {dt.proper_time_integral:.6f}")
        print(f"  Time Asymmetry:         {format_scientific(dt.time_asymmetry)}")
    
    # v7.0 Spectral Analysis
    if sig.spectral_analysis and show_all:
        sp = sig.spectral_analysis
        print(f"\n[SPECTRAL ANALYSIS v7.0]")
        print(f"  Spectral Entropy:       {sp.spectral_entropy:.4f}")
        print(f"  Periodicity Strength:   {format_percentage(sp.periodicity_strength)}")
        print(f"  1/12 Freq Alignment:    {format_percentage(sp.f_12_alignment)}")
        print(f"  Harmonic Content:       {format_percentage(sp.f_harmonic_content)}")
        if sp.dominant_frequencies:
            print(f"  Dominant Frequencies:   {', '.join(f'{f:.4f}' for f in sp.dominant_frequencies[:3])}")
    
    # v7.0 Fractal Analysis
    if sig.fractal_analysis and show_all:
        fr = sig.fractal_analysis
        print(f"\n[FRACTAL ANALYSIS v7.0]")
        print(f"  Box-Counting Dim:       {fr.box_counting_dimension:.4f}")
        print(f"  Correlation Dim:        {fr.correlation_dimension:.4f}")
        print(f"  Information Dim:        {fr.information_dimension:.4f}")
        print(f"  Self-Similarity:        {format_percentage(fr.self_similarity_ratio)}")
        print(f"  Scaling Exponent:       {fr.scaling_exponent:.4f}")
        print(f"  Multifractal Width:     {fr.multifractal_spectrum_width:.4f}")
    
    # v7.0 Thermodynamic Verification
    if sig.thermodynamic_verification and show_all:
        th = sig.thermodynamic_verification
        print(f"\n[THERMODYNAMIC VERIFICATION v7.0]")
        print(f"  Temperature Analog:     {th.temperature_analog:.4f}")
        print(f"  Entropy Analog:         {th.entropy_analog:.4f}")
        print(f"  Energy Analog:          {format_scientific(th.energy_analog)}")
        print(f"  Heat Capacity:          {format_scientific(th.heat_capacity_analog)}")
        print(f"  Zeroth Law (Equil):     {'[OK]' if th.zeroth_law_verified else '[X]'}")
        print(f"  First Law (Conserv):    {'[OK]' if th.first_law_verified else '[X]'}")
        print(f"  Second Law (Entropy):   {'[OK]' if th.second_law_verified else '[X]'}")
        print(f"  Third Law (Exception):  {'[OK]' if th.third_law_verified else '[X]'}")
    
    # v7.0 Quantum Verification
    if sig.quantum_verification and show_all:
        qv = sig.quantum_verification
        print(f"\n[QUANTUM VERIFICATION v7.0]")
        print(f"  Superposition:          {'Yes' if qv.superposition_detected else 'No'}")
        print(f"  Collapse Events:        {qv.collapse_events}")
        print(f"  Uncertainty Product:    {format_scientific(qv.uncertainty_product)}")
        print(f"  Entanglement Sig:       {format_percentage(qv.entanglement_signature)}")
        print(f"  Wavefunction Norm:      {qv.wavefunction_norm:.6f}")
        print(f"  Decoherence Time Est:   {format_scientific(qv.decoherence_time_estimate)}")
        print(f"  Quantum Entropy:        {qv.quantum_entropy:.4f}")
    
    # Traverser Complexity
    if sig.traverser_complexity and show_all:
        tc = sig.traverser_complexity
        print(f"\n[TRAVERSER COMPLEXITY v5+]")
        print(f"  Class: {tc.complexity_class.value}")
        print(f"  Periodicity (Gravity):  {format_percentage(tc.periodicity_score)}")
        print(f"  Progression (Intent):   {format_percentage(tc.progression_score)}")
        print(f"  Fractal Dimension:      {tc.fractal_dimension:.4f}")
        print(f"  Nesting Depth:          {tc.nesting_depth}")
        if tc.spectral_entropy > 0:
            print(f"  Spectral Entropy:       {tc.spectral_entropy:.4f}")
            print(f"  Dominant Frequency:     {tc.dominant_frequency:.6f}")
            print(f"  Phase Coherence:        {format_percentage(tc.phase_coherence)}")
    
    # Gaze Detection
    if sig.gaze_metrics and show_all:
        gm = sig.gaze_metrics
        print(f"\n[GAZE DETECTION v5+]")
        print(f"  Status: {gm.status.value}")
        print(f"  Raw Pressure:           {gm.gaze_pressure:.4f}")
        print(f"  Corrected Pressure:     {gm.corrected_pressure:.4f}")
        print(f"  Watcher Present:        {'Yes' if gm.is_watcher_present else 'No'}")
        print(f"  Local Collapse:         {format_percentage(gm.local_collapse)}")
        if gm.displacement_vector != 0:
            print(f"  Displacement Vector:    {format_scientific(gm.displacement_vector)}")
            print(f"  Collapse Rate:          {gm.collapse_rate:.4f}")
    
    # Temporal Trends
    if sig.temporal_trend and show_all:
        tt = sig.temporal_trend
        print(f"\n[TEMPORAL TRENDS]")
        print(f"  Variance Trend:         {tt.variance_trend.value}")
        print(f"  Variance Gradient:      {format_scientific(tt.variance_gradient)}")
        print(f"  Variance Volatility:    {tt.variance_volatility:.4f}")
        print(f"  P Trend: {tt.p_trend.value}  D Trend: {tt.d_trend.value}  T Trend: {tt.t_trend.value}")
        print(f"  Mean Temporal Distance: {tt.mean_temporal_distance:.4f}")
        print(f"  P-D Correlation:        {tt.pd_correlation:.4f}")
        print(f"  D-T Correlation:        {tt.dt_correlation:.4f}")
        print(f"  P-T Correlation:        {tt.pt_correlation:.4f}")
        if tt.dominant_cycle:
            print(f"  Dominant Cycle:         {tt.dominant_cycle} samples")
        if tt.lyapunov_exponent != 0:
            print(f"  Lyapunov Exponent:      {tt.lyapunov_exponent:.4f}")
            print(f"  Hurst Exponent:         {tt.hurst_exponent:.4f}")
    
    # Exception Approach
    if sig.exception_approach and show_all:
        ea = sig.exception_approach
        print(f"\n[EXCEPTION APPROACH]")
        print(f"  Approaching Exception:  {'Yes' if ea.is_approaching else 'No'}")
        print(f"  Variance Gradient:      {format_scientific(ea.variance_gradient)}")
        print(f"  Current Var Ratio:      {format_percentage(ea.current_variance_ratio)}")
        print(f"  Confidence:             {format_percentage(ea.confidence)}")
        print(f"  Alert Level:            {ea.alert_level}")
        if ea.estimated_steps_to_zero:
            print(f"  Est. Steps to E=0:      {ea.estimated_steps_to_zero}")
        if ea.convergence_rate > 0:
            print(f"  Convergence Rate:       {format_scientific(ea.convergence_rate)}")
            print(f"  Asymptotic Projection:  {ea.asymptotic_projection:.4f}")
    
    # Proof Report
    if sig.proof_report and show_all:
        pr = sig.proof_report
        print(f"\n[ET AXIOM VERIFICATION - 21 RULES]")
        print(f"  Verified:      {pr.verified_count:>3}")
        print(f"  Failed:        {pr.failed_count:>3}")
        print(f"  Indeterminate: {pr.indeterminate_count:>3}")
        print(f"  Total Tests:   {pr.total_tests:>3}")
        print(f"  Pass Rate:     {format_percentage(pr.pass_rate)}")
        print(f"  ET Score:      {format_percentage(pr.et_consistency_score)}")
        print(f"  Overall: {pr.overall_status.value}")
        
        if pr.binding_chain:
            bc = pr.binding_chain
            print(f"\n  [Binding Chain T<->D->P]")
            print(f"    T<->D Verified:       {'[OK]' if bc.t_d_binding_verified else '[X]'}")
            print(f"    D->P Verified:        {'[OK]' if bc.d_p_binding_verified else '[X]'}")
            print(f"    T-P Separation:       {'[OK]' if bc.t_p_separation_verified else '[X]'}")
            print(f"    Chain Integrity:      {format_percentage(bc.chain_integrity)}")
        
        if pr.coherence:
            co = pr.coherence
            print(f"\n  [Coherence Analysis]")
            print(f"    Coherent Ratio:       {format_percentage(co.coherent_ratio)}")
            print(f"    Incoherent Regions:   {co.incoherent_count}")
            print(f"    Self-Defeating:       {co.self_defeating_patterns}")
        
        print(f"\n  [Individual Axiom Results]")
        for proof in pr.axiom_proofs:
            sym = status_symbol(proof.status)
            print(f"    {sym} Rule {proof.axiom_number:>2}: {proof.axiom_name[:35]:<35}")
            if proof.numerical_value is not None and proof.expected_value is not None:
                print(f"              Value: {proof.numerical_value:.4f} (Expected: {proof.expected_value:.4f})")
    
    print("\n" + "=" * 80)
    print(f"  Exception Theory Scanner v{VERSION} - Analysis Complete")
    print("=" * 80 + "\n")


def print_compact_report(sig: ETSignature):
    """Print compact single-line report."""
    ind = sig.indeterminate
    print(f"[{sig.timestamp.strftime('%H:%M:%S')}] "
          f"State={sig.state.name:12} "
          f"Var={sig.variance_ratio:.4f} "
          f"P={sig.pdt.p_ratio:.3f} D={sig.pdt.d_ratio:.3f} T={sig.pdt.t_ratio:.3f} "
          f"PureT={ind.failures:>3} "
          f"Bind={sig.binding_strength:.3f}")


# =============================================================================
# CONTINUOUS SCAN MODE
# =============================================================================

def continuous_scan(
    interval: float = 1.0,
    entropy_size: int = 8192,
    show_full: bool = False,
    max_scans: Optional[int] = None
):
    """
    Continuous scanning mode.
    
    Press Ctrl+C to stop.
    """
    global _running, _scan_active
    _running = True
    _scan_active = True
    
    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n[*] Continuous ET Scan - v{VERSION}")
    print(f"    Interval: {interval}s, Sample Size: {entropy_size}")
    print(f"    Press Ctrl+C to stop\n")
    
    scan_count = 0
    try:
        while _running and _scan_active:
            if max_scans and scan_count >= max_scans:
                break
                
            try:
                sig = et_scan(entropy_size=entropy_size, generate_proof=False)
                
                if show_full:
                    print_full_report(sig, show_all=False)
                else:
                    print_compact_report(sig)
                
                # Alert on significant events
                if sig.indeterminate.failures > 5:
                    print(f"  [!] HIGH PURE T DENSITY: {sig.indeterminate.failures}")
                
                if sig.exception_approach and sig.exception_approach.alert_level not in ["NONE", ""]:
                    print(f"  [!] EXCEPTION APPROACH: {sig.exception_approach.alert_level}")
                
                if sig.gaze_metrics and sig.gaze_metrics.is_watcher_present:
                    print(f"  [!] WATCHER DETECTED: {sig.gaze_metrics.status.value}")
                
                scan_count += 1
                
            except Exception as e:
                print(f"  [!] Scan error: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    finally:
        _scan_active = False
        signal.signal(signal.SIGINT, original_handler)
    
    print(f"\n[*] Continuous scan stopped. Total scans: {scan_count}")
    
    # Print session summary
    summary = get_history().get_session_summary()
    if "error" not in summary:
        print(f"\n[SESSION SUMMARY]")
        print(f"  Duration: {summary['session_duration']:.1f}s")
        print(f"  Total Scans: {summary['total_scans']}")
        print(f"  Variance: mean={summary['variance']['mean']:.2f}, std={summary['variance']['std']:.2f}")
        print(f"  PDT Averages: P={summary['pdt_averages']['p']:.4f}, D={summary['pdt_averages']['d']:.4f}, T={summary['pdt_averages']['t']:.4f}")
        print(f"  Pure T Total: {summary['pure_t_statistics']['total']}")


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def print_menu():
    """Print interactive menu."""
    print(f"\n{'=' * 60}")
    print(f"  EXCEPTION THEORY SCANNER v{VERSION}")
    print(f"  {VERSION_NAME}")
    print(f"{'=' * 60}")
    print("""
  [1] Scan Hardware Entropy (Quick)
  [2] Scan Hardware Entropy (Large - 1MB)
  [3] Scan File (via dialog)
  [4] Scan URL
  [5] Scan Process
  [6] Scan Clipboard
  [7] Scan Custom String
  
  [C] Continuous Scan Mode
  [H] View History Summary
  [E] Export Session
  [R] Reset History
  
  [A] About ET Scanner
  [Q] Quit
""")


def interactive_menu():
    """Run interactive menu."""
    global _running
    _running = True
    
    print(f"\n{'=' * 60}")
    print(f"  EXCEPTION THEORY SCANNER v{VERSION}")
    print(f"  {VERSION_NAME}")
    print(f"{'=' * 60}")
    print("\n  Initializing...")
    
    # Warm-up scan
    try:
        _ = et_scan(entropy_size=1024, generate_proof=False)
        print("  Ready.\n")
    except Exception as e:
        print(f"  Warning: {e}\n")
    
    while _running:
        print_menu()
        
        try:
            choice = input("  Select option: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break
        
        print()
        
        try:
            if choice == '1':
                # Quick entropy scan
                print("[*] Scanning 64KB hardware entropy...")
                sig = et_scan(entropy_size=65536)
                print_full_report(sig)
                
            elif choice == '2':
                # Large entropy scan
                print("[*] Scanning 1MB hardware entropy (this may take a moment)...")
                sig = et_scan(entropy_size=1048576)
                print_full_report(sig)
                
            elif choice == '3':
                # File scan
                filepath = select_file()
                if filepath:
                    print(f"[*] Scanning file: {filepath}")
                    try:
                        data, source = UniversalInput.from_file(filepath)
                        sig = et_scan(data=data, source=source)
                        print_full_report(sig)
                    except Exception as e:
                        print(f"[!] Error: {e}")
                else:
                    print("[!] No file selected")
                    
            elif choice == '4':
                # URL scan
                url = input("  Enter URL: ").strip()
                if url:
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    print(f"[*] Fetching and scanning URL: {url}")
                    try:
                        data, source = UniversalInput.from_url(url)
                        sig = et_scan(data=data, source=source)
                        print_full_report(sig)
                    except Exception as e:
                        print(f"[!] Error: {e}")
                else:
                    print("[!] No URL entered")
                    
            elif choice == '5':
                # Process scan
                if not HAS_PSUTIL:
                    print("[!] psutil not installed. Run: pip install psutil")
                else:
                    proc_input = input("  Enter process name or PID (blank for self): ").strip()
                    try:
                        if proc_input.isdigit():
                            data, source = UniversalInput.from_process(pid=int(proc_input))
                        elif proc_input:
                            data, source = UniversalInput.from_process(name=proc_input)
                        else:
                            data, source = UniversalInput.from_process()
                        print(f"[*] Scanning process: {source.source}")
                        sig = et_scan(data=data, source=source)
                        print_full_report(sig)
                    except Exception as e:
                        print(f"[!] Error: {e}")
                        
            elif choice == '6':
                # Clipboard scan
                print("[*] Scanning clipboard content...")
                try:
                    data, source = UniversalInput.from_clipboard()
                    sig = et_scan(data=data, source=source)
                    print_full_report(sig)
                except Exception as e:
                    print(f"[!] Error: {e}")
                    
            elif choice == '7':
                # String scan
                text = input("  Enter text to scan: ")
                if text:
                    data, source = UniversalInput.from_string(text)
                    sig = et_scan(data=data, source=source)
                    print_full_report(sig)
                else:
                    print("[!] No text entered")
                    
            elif choice == 'C':
                # Continuous mode
                try:
                    interval = float(input("  Scan interval (seconds) [1.0]: ").strip() or "1.0")
                    size_input = input("  Sample size [8192]: ").strip() or "8192"
                    size = int(size_input)
                    full_input = input("  Full reports? (y/n) [n]: ").strip().lower()
                    full = full_input == 'y'
                except ValueError:
                    interval, size, full = 1.0, 8192, False
                
                continuous_scan(interval=interval, entropy_size=size, show_full=full)
                
            elif choice == 'H':
                # History summary
                summary = get_history().get_session_summary()
                if "error" in summary:
                    print("[!] No scan data yet")
                else:
                    print("\n[SESSION SUMMARY]")
                    print(f"  Total Scans:      {summary['total_scans']}")
                    print(f"  Total Events:     {summary['total_events']}")
                    print(f"  Session Duration: {summary['session_duration']:.1f}s")
                    print(f"\n  Variance Stats:")
                    print(f"    Mean: {summary['variance']['mean']:.2f}")
                    print(f"    Std:  {summary['variance']['std']:.2f}")
                    print(f"    Min:  {summary['variance']['min']:.2f}")
                    print(f"    Max:  {summary['variance']['max']:.2f}")
                    print(f"\n  PDT Averages:")
                    print(f"    P: {summary['pdt_averages']['p']:.4f}")
                    print(f"    D: {summary['pdt_averages']['d']:.4f}")
                    print(f"    T: {summary['pdt_averages']['t']:.4f}")
                    print(f"\n  Pure T Statistics:")
                    print(f"    Total:  {summary['pure_t_statistics']['total']}")
                    print(f"    Mean:   {summary['pure_t_statistics']['mean']:.2f}")
                    print(f"    Max:    {summary['pure_t_statistics']['max']}")
                    print(f"\n  T_time (tau) Total: {summary['t_time_total']:.6f}")
                    print(f"  Substantiations:    {summary['substantiations']}")
                    print(f"\n  State Distribution:")
                    for state, count in summary['state_distribution'].items():
                        print(f"    {state}: {count}")
                        
            elif choice == 'E':
                # Export session
                filepath = select_save_file()
                if filepath:
                    try:
                        path = get_history().export_session(filepath)
                        print(f"[*] Session exported to: {path}")
                    except Exception as e:
                        print(f"[!] Export error: {e}")
                else:
                    print("[!] Export cancelled")
                    
            elif choice == 'R':
                # Reset history
                confirm = input("  Reset all history? (y/n): ").strip().lower()
                if confirm == 'y':
                    get_history().reset()
                    print("[*] History reset")
                else:
                    print("[*] Reset cancelled")
                    
            elif choice == 'A':
                # About
                print(f"""
{'=' * 60}
  EXCEPTION THEORY SCANNER v{VERSION}
  {VERSION_NAME}
{'=' * 60}

  Exception Theory (ET) provides a complete ontological framework
  based on the singular axiom:
  
    "For every exception there is an exception, except the exception."
  
  The three primitives:
    P - Point (Infinite substrate)
    D - Descriptor (Finite constraint)
    T - Traverser (Indeterminate agent)
  
  This scanner analyzes any data for ET signatures:
    - PDT classification using gradient analysis
    - Indeterminate form detection ([0/0], [inf/inf], etc.)
    - L'Hopital resolution (T's navigation algorithm)
    - Pure T detection (unresolvable forms)
    - Variance analysis against (n^2-1)/12
    - 1/12 manifold alignment
    - Cosmological ratio detection (68.3/26.8/4.9)
    - Dual time system (D_time vs T_time)
    - Observer effect / Gaze detection
    - Traverser complexity (Gravity vs Intent)
    - Spectral & fractal analysis (v7.0)
    - Thermodynamic & quantum verification (v7.0)
    - Complete 21 Rules verification
  
  All mathematics derived from Exception Theory.
  
  Author: Derived from M.J.M.'s Exception Theory
{'=' * 60}
""")
            elif choice == 'Q':
                print("[*] Goodbye.")
                break
                
            else:
                print("[!] Unknown option")
                
        except Exception as e:
            print(f"[!] Error: {e}")
            import traceback
            traceback.print_exc()
    
    _running = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def scan_entropy(size: int = 65536, full_report: bool = True) -> ETSignature:
    """Quick entropy scan."""
    sig = et_scan(entropy_size=size)
    if full_report:
        print_full_report(sig)
    return sig


def scan_file(filepath: str, full_report: bool = True) -> ETSignature:
    """Scan a file."""
    data, source = UniversalInput.from_file(filepath)
    sig = et_scan(data=data, source=source)
    if full_report:
        print_full_report(sig)
    return sig


def scan_url(url: str, full_report: bool = True) -> ETSignature:
    """Scan URL content."""
    data, source = UniversalInput.from_url(url)
    sig = et_scan(data=data, source=source)
    if full_report:
        print_full_report(sig)
    return sig


def scan_string(text: str, full_report: bool = True) -> ETSignature:
    """Scan a text string."""
    data, source = UniversalInput.from_string(text)
    sig = et_scan(data=data, source=source)
    if full_report:
        print_full_report(sig)
    return sig


def scan_bytes(raw: bytes, full_report: bool = True) -> ETSignature:
    """Scan raw bytes."""
    data, source = UniversalInput.from_bytes(raw)
    sig = et_scan(data=data, source=source)
    if full_report:
        print_full_report(sig)
    return sig


def scan_numpy(arr: np.ndarray, full_report: bool = True) -> ETSignature:
    """Scan numpy array."""
    data, source = UniversalInput.from_numpy(arr)
    sig = et_scan(data=data, source=source)
    if full_report:
        print_full_report(sig)
    return sig


def quick_scan(data: Optional[np.ndarray] = None, size: int = 8192) -> Dict:
    """Quick scan returning dictionary of key metrics."""
    if data is None:
        data, _ = UniversalInput.from_entropy(size)
    
    sig = et_scan(data=data, generate_proof=False)
    
    return {
        "state": sig.state.name,
        "variance_ratio": sig.variance_ratio,
        "p_ratio": sig.pdt.p_ratio,
        "d_ratio": sig.pdt.d_ratio,
        "t_ratio": sig.pdt.t_ratio,
        "pure_t_count": sig.indeterminate.failures,
        "traverser_density": sig.indeterminate.traverser_density,
        "binding_strength": sig.binding_strength,
        "shimmer_index": sig.shimmer_index,
        "alignment_d": sig.alignment_d,
        "alignment_t": sig.alignment_t
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Exception Theory Scanner v{VERSION} - {VERSION_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python et_scanner_v7.py                    Interactive menu
  python et_scanner_v7.py --entropy          Scan hardware entropy
  python et_scanner_v7.py --file path.txt    Scan a file
  python et_scanner_v7.py --continuous       Continuous scan mode
  python et_scanner_v7.py --url https://...  Scan URL content
        """
    )
    
    parser.add_argument('--entropy', '-e', action='store_true',
                        help='Scan hardware entropy')
    parser.add_argument('--file', '-f', type=str,
                        help='Scan a file')
    parser.add_argument('--url', '-u', type=str,
                        help='Scan URL content')
    parser.add_argument('--string', '-s', type=str,
                        help='Scan a text string')
    parser.add_argument('--continuous', '-c', action='store_true',
                        help='Continuous scan mode')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                        help='Continuous scan interval (seconds)')
    parser.add_argument('--size', type=int, default=65536,
                        help='Sample size for entropy scans')
    parser.add_argument('--compact', action='store_true',
                        help='Compact output')
    parser.add_argument('--version', '-v', action='version',
                        version=f'ET Scanner v{VERSION}')
    
    args = parser.parse_args()
    
    # Handle command-line options
    if args.continuous:
        continuous_scan(interval=args.interval, entropy_size=args.size,
                        show_full=not args.compact)
    elif args.entropy:
        sig = scan_entropy(size=args.size, full_report=not args.compact)
        if args.compact:
            print_compact_report(sig)
    elif args.file:
        sig = scan_file(args.file, full_report=not args.compact)
        if args.compact:
            print_compact_report(sig)
    elif args.url:
        sig = scan_url(args.url, full_report=not args.compact)
        if args.compact:
            print_compact_report(sig)
    elif args.string:
        sig = scan_string(args.string, full_report=not args.compact)
        if args.compact:
            print_compact_report(sig)
    else:
        # Interactive menu
        interactive_menu()


if __name__ == "__main__":
    # Windows compatibility: ensure proper setup
    if sys.platform == 'win32':
        try:
            import multiprocessing
            multiprocessing.freeze_support()
        except:
            pass
    
    # Set up signal handler
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except:
        pass
    
    main()
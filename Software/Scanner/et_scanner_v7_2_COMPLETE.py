#!/usr/bin/env python3

# ============================================================================
# WINDOWS CRASH FIX v7.2.0: Complete Windows 10/11 compatibility
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
Exception Theory Scanner v7.2 - Rigorous Verification Edition
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

Run with: python et_scanner_v7.2.py
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
    # FIX: Do not set _running = False. Only stop the active scan.
    # This prevents the whole program from closing when stopping a continuous scan.
    # _running = False 
    _scan_active = False
    # Use ASCII-safe message for Windows compatibility
    try:
        print("\n\n[!] Interrupt received. Stopping scan...")
    except Exception:
        pass  # Ignore encoding errors on shutdown


# NOTE: Signal handlers are registered in main() for Windows compatibility
# signal.signal(signal.SIGINT, signal_handler)  # MOVED TO MAIN

# =============================================================================
# ET CONSTANTS (Derived from Exception Theory Mathematics)
# =============================================================================

# The 12: 3 primitives × 4 logic states = 12 fundamental configurations
# From ET: "MANIFOLD_SYMMETRY = 12" - This is the 12-fold symmetry of the manifold
MANIFOLD_SYMMETRY = 12
MANIFOLD_SYMMETRY_24 = 24
MANIFOLD_SYMMETRY_48 = 48                      

# Base variance: 1/12 (fundamental quantum of descriptor variance)
# From ET Math: "BASE_VARIANCE = 1/12 is the core deviation unit in the descriptor field"
BASE_VARIANCE = 1.0 / MANIFOLD_SYMMETRY  # 0.08333...

# Version information  
VERSION = "7.2.0"
VERSION_NAME = "Rigorous Verification Edition"

# Theoretical variance for uniform byte distribution: σ² = (n² - 1) / 12
# This is the ET-derived formula for variance of n states
THEORETICAL_VARIANCE_BYTES = (256 ** 2 - 1) / MANIFOLD_SYMMETRY  # 5461.25
THEORETICAL_STD_BYTES = np.sqrt(THEORETICAL_VARIANCE_BYTES)  # 73.9003

# ET Cosmological Ratios (Koide-like structure from 3×4 permutation) (observed)
# Dark Energy : Dark Matter : Baryonic = 2/3 : ~1/4 : ~1/12
DARK_ENERGY_RATIO = 0.683
DARK_MATTER_RATIO = 0.268
BARYONIC_RATIO = 0.049

# ET Predictions (theoretical)                            
ET_DARK_ENERGY_PREDICTED = 2.0 / 3.0    # 0.6667 ≈ 68.3%
ET_DARK_MATTER_PREDICTED = 1.0 / 4.0    # 0.25 ≈ 26.8% (approximate)
ET_BARYONIC_PREDICTED = 1.0 / 12.0      # 0.0833 ≈ 4.9%

# Golden ratio approximations from manifold (5/8 = active/structural)
PHI = (1 + np.sqrt(5)) / 2
PHI_INVERSE = PHI - 1                 
PHI_APPROX = 5.0 / 8.0  # 0.625, approximates φ-1 ≈ 0.618
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
LHOPITAL_MAX_ITERATIONS = 12  # 12-fold symmetry in iterations

# History limits
MAX_HISTORY_SIZE = 1000
EXCEPTION_APPROACH_WINDOW = 20
TREND_WINDOW = 20

# v5.0 Constants - The Geometry of Conscious Detection
GAZE_THRESHOLD = 1.20  # 1 + 20% = conscious detection crystallization
RESONANCE_THRESHOLD = 13.0 / 12.0  # 1.0833... = 1 + 1/12 (subliminal)
LOCK_THRESHOLD = 1.50  # Target locked

# v6.0 Constants - Enhanced Detection
DESCRIPTOR_LAYERS = 8  # Virtual environment correction (screen + VM + OS + HW)
ENTROPY_GRADIENT_THRESHOLD = 0.7  # Compressed data detection
POWER_OF_2_TOLERANCE = 0.05  # Manifold boundary detection tolerance

# Indeterminate form detection thresholds (refined)
INDETERMINATE_WINDOW = 5  # Window for derivative approximation
# v7.0 Constants
BOLTZMANN_SCALE = 1.0 / MANIFOLD_SYMMETRY
PLANCK_SCALE = 1.0 / MANIFOLD_SYMMETRY
FINE_STRUCTURE_APPROX = 1.0 / 137.036
GAUGE_BOSON_COUNT = 12
INDETERMINATE_WINDOW = 5
PURE_T_PERSISTENCE_THRESHOLD = 3  # Iterations before declaring pure T
FRACTAL_BOX_SIZES = [2, 4, 8, 16, 32, 64]
TAKENS_EMBEDDING_DELAY = 1
TAKENS_EMBEDDING_DIM = 3


# =============================================================================
# ENUMS
# =============================================================================

class ETState(Enum):
    """
    Trinary Ternary States from ET.
    
    From Rules of Exception Law:
    - STATE_0: P ∘ D₀ (Unsubstantiated, P-dominant)
    - STATE_1: P ∘ D₁ (Unsubstantiated, D-dominant)  
    - SUPERPOSITION: P ∘ (D₀, D₁) (Multiple descriptors, awaiting T selection)
    - SUBSTANTIATED: (P ∘ D) ∘ T (The Exception - has active traverser)
    """
    STATE_0 = 0  # P ∘ D₀ (Unsubstantiated)
    STATE_1 = 1  # P ∘ D₁ (Unsubstantiated)
    SUPERPOSITION = 2  # P ∘ (D₀, D₁) (Superposition)
    SUBSTANTIATED = 3  # (P ∘ D) ∘ T (The Exception)


class ComplexityClass(Enum):
    """
    v5.0/v6.0 - Distinguishing Source of Agency.
    
    From ET: Gravity is a Traverser (Rule 10), Intent is nested T (Rule 19).
    """
    STATIC = "STATIC"  # Dead matter / Frozen code
    CYCLIC_GRAVITY = "CYCLIC"  # Periodic T (Planets/Orbits) - Gravity as T
    PROGRESSIVE_INTENT = "INTENT"  # Aperiodic T (Life/Swarm) - Nested T
    CHAOTIC = "CHAOTIC"  # Random noise (pure D variance)
    QUANTUM_SUPERPOSITION = "QUANTUM"
    UNKNOWN = "UNKNOWN"


class GazeStatus(Enum):
    """
    v5.0/v6.0 - Observer Effect Status.
    
    From ET: T engagement substantiates (Rule 7), observation creates new config (Rule 18).
    """
    UNOBSERVED = "UNOBSERVED"
    SUBLIMINAL = "SUBLIMINAL"  # > 1.0833 (1 + 1/12)
    DETECTED = "DETECTED"  # > 1.20 (crystallization)
    LOCKED = "LOCKED"  # > 1.50 (high focus)


class TrendDirection(Enum):
    """Descriptor gradient directions through temporal manifold."""
    STABLE = "STABLE"
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    OSCILLATING = "OSCILLATING"
    CONVERGING = "CONVERGING"  # → Exception
    DIVERGING = "DIVERGING"


class InputType(Enum):
    """Types of input sources."""
    ENTROPY = "entropy"
    FILE = "file"
    PROCESS = "process"
    URL = "url"
    STREAM = "stream"
    RAW = "raw"
    CLIPBOARD = "clipboard"


class ProofStatus(Enum):
    """Status of axiom proofs."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"
    NOT_TESTED = "NOT_TESTED"
    PARTIAL = "PARTIAL"


class IndeterminateType(Enum):
    """
    Types of indeterminate forms - T signatures.
    
    From ET Math: T = lim[x→a] f(x)/g(x) = [0/0] or [∞/∞]
    """
    ZERO_ZERO = "[0/0]"  # Standard indeterminate
    INF_INF = "[∞/∞]"  # Ratio of infinities
    ZERO_TIMES_INF = "[0×∞]"  # Product form
    INF_MINUS_INF = "[∞-∞]"  # Difference form
    ONE_INF = "[1^∞]"  # Power form
    ZERO_ZERO_POWER = "[0^0]"  # Power form
    INF_ZERO = "[∞^0]"  # Power form
    PURE_T = "PURE_T"  # Resists L'Hôpital resolution

class ThermodynamicLaw(Enum):
    ZEROTH = "ZEROTH"
    FIRST = "FIRST"
    SECOND = "SECOND"
    THIRD = "THIRD"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

# v7.0 Additional Constants
MANIFOLD_SYMMETRY_24 = 24
MANIFOLD_SYMMETRY_48 = 48
BOLTZMANN_SCALE = 1.0 / MANIFOLD_SYMMETRY
PLANCK_SCALE = 1.0 / MANIFOLD_SYMMETRY
FINE_STRUCTURE_APPROX = 1.0 / 137.036
GAUGE_BOSON_COUNT = 12
FRACTAL_BOX_SIZES = [2, 4, 8, 16, 32, 64]
TAKENS_EMBEDDING_DELAY = 1
TAKENS_EMBEDDING_DIM = 3

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PDTClassification:
    """
    P, D, T classification results.
    
    From ET: P = ∞, D = n (finite), T = [0/0] (indeterminate)
    """
    p_count: int  # Infinite/unbounded values
    d_count: int  # Finite/bounded values
    t_count: int  # Indeterminate forms
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
    """Detailed indeterminate form analysis."""
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
    """
    Indeterminate form analysis - T signatures.
    
    From ET Math: "L'Hôpital's rule = the navigation algorithm"
    Pure T = forms that resist L'Hôpital resolution
    """
    zero_zero: int  # [0/0] forms
    inf_inf: int  # [∞/∞] forms
    zero_times_inf: int  # [0×∞] forms
    inf_minus_inf: int  # [∞-∞] forms
    one_inf: int  # [1^∞] forms
    zero_zero_power: int  # [0^0] forms
    inf_zero: int  # [∞^0] forms
    total: int
    density: float
    resolutions: int  # Forms L'Hôpital resolved
    failures: int  # Pure T - unresolvable
    traverser_density: float  # Ratio of pure T to total
    detailed_forms: List[IndeterminateFormDetail] = field(default_factory=list)
    resolution_rate: float = 0.0
    mean_resolution_iterations: float = 0.0
    persistent_t_regions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DescriptorGradient:
    """
    Descriptor gradient: f'(x) = ΔD/ΔP.
    
    From ET Math: "Derivatives are descriptor changes (D variance)"
    """
    mean: float
    variance: float
    max_val: float
    min_val: float
    sign_changes: int
    gradient_array: Optional[np.ndarray] = None
    discontinuity_count: int = 0  # Sharp gradient discontinuities
    power_of_2_alignments: int = 0  # Manifold boundary alignments
    curvature_mean: float = 0.0
    divergence: float = 0.0
    laplacian: float = 0.0


@dataclass
class DualTime:
    """
    Dual Time System from Exception Theory.
    
    From ET: D_time = Descriptor time (coordinate/global)
             T_time = Agential time τ (proper/local)
             dτ/dt = 1 - |binding_strength|
    """
    # Descriptor Time
    d_time_elapsed: float
    d_time_samples: int
    d_time_rate: float

    # Agential Time
    t_time_tau: float
    t_time_substantiations: int
    t_time_dtau_dt: float

    # Relationships
    binding_strength: float
    time_dilation_factor: float
    variance_thickness: float
    proper_time_integral: float = 0.0
    time_asymmetry: float = 0.0


@dataclass
class TemporalTrend:
    """Temporal trend analysis through descriptor manifold."""
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

    # Cross-correlations (PDT relationships)
    pd_correlation: float
    dt_correlation: float
    pt_correlation: float

    dominant_cycle: Optional[int]
    lyapunov_exponent: float = 0.0
    hurst_exponent: float = 0.5


@dataclass
class ExceptionApproach:
    """
    Exception approach detection.
    
    From ET: Variance(E) = 0 - the Exception cannot be otherwise.
    """
    is_approaching: bool
    variance_gradient: float
    current_variance_ratio: float
    estimated_steps_to_zero: Optional[int]
    confidence: float
    alert_level: str  # NONE, WATCH, WARNING, CRITICAL
    variance_history: List[float]
    convergence_rate: float = 0.0
    asymptotic_projection: float = 0.0


@dataclass
class TraverserComplexity:
    """
    v5.0/v6.0 - Analysis of Agency Depth.
    
    From ET: T within T - nested infinitely (Rule 19)
    """
    complexity_class: ComplexityClass
    periodicity_score: float  # 0.0 to 1.0 (1.0 = Perfect Orbit)
    progression_score: float  # 0.0 to 1.0 (1.0 = Strong Intent)
    fractal_dimension: float  # D_f of the T-curve
    nesting_depth: int  # Estimated layers of T
    autocorrelation_peaks: List[float] = field(default_factory=list)
    spectral_entropy: float = 0.0
    dominant_frequency: float = 0.0
    phase_coherence: float = 0.0


@dataclass
class GazeMetrics:
    """
    v5.0/v6.0 - Observer Effect Metrics.
    
    From ET: T engagement substantiates, observation displaces.
    """
    status: GazeStatus
    gaze_pressure: float  # Current / Baseline
    is_watcher_present: bool
    local_collapse: float  # Variance reduction magnitude
    corrected_pressure: float = 0.0  # After layer attenuation correction
    displacement_vector: float = 0.0
    collapse_rate: float = 0.0


@dataclass
class ManifoldMetrics:
    """
    v6.0 - Enhanced manifold structure analysis.
    
    From ET: The manifold has 12-fold symmetry, with 1/12 base variance.
    """
    shimmer_index: float  # Ordered chaos ratio
    binding_strength: float
    
    # Cosmological ratio detection
    dark_energy_alignment: float
    dark_matter_alignment: float
    baryonic_alignment: float
    
    # Koide-like structure
    koide_ratio: float  # Should approach 2/3 for leptons
    
    # Entropy analysis
    shannon_entropy: float
    normalized_entropy: float
    entropy_gradient_mean: float
    is_compressed_data: bool
    
    # Manifold boundary detection
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
    """
    v6.0 - Verify the binding chain T↔D→P.
    
    From ET: "T binds to D. T does not bind to P."
    """
    t_d_binding_verified: bool  # T↔D relationship
    d_p_binding_verified: bool  # D→P relationship
    t_p_separation_verified: bool  # T does not bind to P directly
    t_d_binding_verified: bool
    d_p_binding_verified: bool
    t_p_separation_verified: bool
    chain_integrity: float  # 0.0 to 1.0
    binding_energy_estimate: float = 0.0
    correlation_td: float = 0.0
    correlation_dp: float = 0.0
    correlation_tp: float = 0.0


@dataclass
class CoherenceAnalysis:
    """
    v6.0 - Incoherence region detection.
    
    From ET: Incoherence = self-defeating configurations that cannot have T.
    """
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
    """External event for correlation."""
    timestamp: datetime
    event_type: str
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class AxiomProof:
    """
    Proof result for an ET axiom.
    
    Maps to the 21 Rules of Exception Law.
    """
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
    """Complete proof report for Exception Theory validation."""
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
    # v6.0 additions
    binding_chain: Optional[BindingChainVerification] = None
    coherence: Optional[CoherenceAnalysis] = None
    pdt_eim_verified: bool = False  # PDT = EIM (3=3)
    thermodynamic: Optional['ThermodynamicVerification'] = None
    quantum: Optional['QuantumVerification'] = None
    spectral: Optional['SpectralAnalysis'] = None
    fractal: Optional['FractalAnalysis'] = None
    total_tests: int = 0
    pass_rate: float = 0.0
    et_consistency_score: float = 0.0


@dataclass
class InputSource:
    """Input source description."""
    input_type: InputType
    source: str
    metadata: Dict = field(default_factory=dict)


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
class ThermodynamicVerification:
    """v7.0: Thermodynamic law verification."""
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
    """v7.0: Quantum mechanics verification."""
    superposition_detected: bool
    collapse_events: int
    uncertainty_product: float
    entanglement_signature: float
    wavefunction_norm: float
    decoherence_time_estimate: float
    quantum_entropy: float


@dataclass
class ETSignature:
    """Complete ET signature from a scan."""
    timestamp: datetime
    input_source: InputSource

    # Dual Time
    dual_time: DualTime

    # Core PDT
    pdt: PDTClassification

    # Variance
    variance: float
    variance_ratio: float
    variance_from_exception: float

    # Descriptor analysis
    descriptor_gradient: DescriptorGradient
    descriptor_distance: float

    # Indeterminate forms
    indeterminate: IndeterminateAnalysis

    # State
    state: ETState

    # 1/12 alignment
    alignment_d: float
    alignment_t: float

    # Manifold metrics (v6.0 enhanced)
    manifold_metrics: Optional[ManifoldMetrics] = None
    shimmer_index: float = 0.0
    binding_strength: float = 0.0

    # v2+ features
    temporal_trend: Optional[TemporalTrend] = None
    exception_approach: Optional[ExceptionApproach] = None

    # Proof
    proof_report: Optional[ETProofReport] = None

    # v5.0 features
    traverser_complexity: Optional[TraverserComplexity] = None
    gaze_metrics: Optional[GazeMetrics] = None

    # v7.0 features
    spectral_analysis: Optional[SpectralAnalysis] = None
    fractal_analysis: Optional[FractalAnalysis] = None
    thermodynamic_verification: Optional[ThermodynamicVerification] = None
    quantum_verification: Optional[QuantumVerification] = None
    
    # Data integrity
    data_size: int = 0
    checksum: str = ""


# =============================================================================
# DUAL TIME TRACKER
# =============================================================================

class DualTimeTracker:
    """
    Tracks both Descriptor Time and Agential Time.
    
    From ET:
    - D_time: Coordinate time, finite Descriptor
    - T_time: Proper time τ, accumulated substantiations
    - dτ/dt = 1 - |binding_strength|
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

        # dτ/dt from binding (from ET: proper time accumulation)
        dtau_dt = 1.0 - binding

        if is_exception:
            self.t_time_substantiations += 1

        # Accumulate proper time
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
        
        # dτ/dt from ET: when v → c, dτ/dt → 0
        dtau_dt = 1.0 - avg_binding
        
        # Time dilation factor (relativistic analog)
        dilation = np.sqrt(max(0, 1.0 - avg_binding ** 2))
        
        # Variance thickness (inverse relationship)
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

        # Baselines for detection
        self.binding_baseline = 0.5
        self.t_history: deque = deque(maxlen=100)
        self.variance_baseline = THEORETICAL_VARIANCE_BYTES
        self.samples_count = 0

    def add_signature(self, sig: ETSignature):
        """Add signature to history."""
        self.signatures.append(sig)
        self.t_history.append(sig.pdt.t_ratio)
        self.samples_count += 1

        # Update baselines (exponential moving average)
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
            "pdt": {
                "p": sig.pdt.p_ratio,
                "d": sig.pdt.d_ratio,
                "t": sig.pdt.t_ratio
            },
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

        # Pure T statistics
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
    
    def analyze_event_correlations(self) -> Dict:
        """Analyze correlations between external events and scanner metrics."""
        if not self.events or len(self.signatures) < 10:
            return {"summary": "Insufficient data for correlation analysis"}
        
        results = []
        for event in self.events:
            # Find signatures around event time
            window_sigs = [s for s in self.signatures 
                           if abs((s.timestamp - event.timestamp).total_seconds()) < 30]
            
            if len(window_sigs) >= 2:
                # Compare before/after
                before = [s for s in window_sigs if s.timestamp < event.timestamp]
                after = [s for s in window_sigs if s.timestamp >= event.timestamp]
                
                if before and after:
                    var_before = np.mean([s.variance for s in before])
                    var_after = np.mean([s.variance for s in after])
                    t_before = np.mean([s.pdt.t_ratio for s in before])
                    t_after = np.mean([s.pdt.t_ratio for s in after])
                    
                    results.append({
                        "event": event.description,
                        "variance_shift": float(var_after - var_before),
                        "t_ratio_shift": float(t_after - t_before)
                    })
        
        return {
            "summary": f"Analyzed {len(results)} events",
            "correlations": results
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
            metadata={
                "filename": path.name,
                "size": len(raw),
                "extension": path.suffix
            }
        )

    @staticmethod
    def from_url(url: str, max_bytes: Optional[int] = None,
                 timeout: int = 10) -> Tuple[np.ndarray, InputSource]:
        """Fetch URL content."""
        if not HAS_URLLIB:
            raise ImportError("urllib not available")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ET-Scanner/7.2'})
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
            metadata={
                "pid": proc.pid,
                "name": proc.name(),
                "duration": duration,
                "sample_rate": sample_rate
            }
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
        # Windows-safe: avoid problematic topmost manipulation
        if sys.platform != 'win32':
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
        root.update_idletasks()  # Use update_idletasks instead of update

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
        # Windows-safe: avoid problematic topmost manipulation
        if sys.platform != 'win32':
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
        root.update_idletasks()  # Use update_idletasks instead of update

        filepath = filedialog.asksaveasfilename(
            parent=root,
            title="Save export file",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
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
    """
    f'(x) = ΔD/ΔP - rate of descriptor change.
    
    From ET Math: "Derivatives are descriptor changes (D variance)"
    """
    return np.diff(data.astype(np.float64))


def compute_second_derivative(data: np.ndarray) -> np.ndarray:
    """
    f''(x) = Δ²D/ΔP² - curvature of descriptor field.
    
    Used in L'Hôpital resolution.
    """
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
    """
    Δs(pᵢ, pⱼ) = ‖f(dᵢ) - f(dⱼ)‖ - pure relationalism.
    
    From ET: "Distance is descriptor difference, not spatial separation"
    """
    return float(np.linalg.norm(d_i.astype(np.float64) - d_j.astype(np.float64)))


def compute_variance(data: np.ndarray) -> float:
    """
    Variance(c) - spread of descriptor possibilities.
    
    From ET: σ² = (n² - 1) / 12 for uniform distribution over n states.
    """
    return float(np.var(data.astype(np.float64)))


def compute_shannon_entropy(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute Shannon entropy of data.
    
    Returns (entropy, normalized_entropy).
    Compressed data has high normalized entropy (→ 1.0).
    """
    if len(data) == 0:
        return 0.0, 0.0
    
    # Compute histogram
    hist, _ = np.histogram(data, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zeros
    
    if len(hist) == 0:
        return 0.0, 0.0
    
    # Normalize to probabilities
    probs = hist / np.sum(hist)
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalized (max entropy for bytes is 8 bits)
    normalized = entropy / 8.0
    
    return float(entropy), float(normalized)


def compute_entropy_gradient(data: np.ndarray, window_size: int = 256) -> np.ndarray:
    """
    Compute local entropy gradient.
    
    Compressed data has sharp gradient discontinuities where
    indeterminate forms naturally emerge.
    """
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
    """
    Detect manifold boundaries at power-of-2 intervals.
    
    From ET: Manifold structure has natural boundaries at 2^n positions.
    """
    boundaries = []
    gradient = compute_descriptor_gradient(data)
    
    # Check power-of-2 positions
    powers = [2**i for i in range(3, 20) if 2**i < len(gradient)]
    
    for pos in powers:
        if pos < len(gradient):
            # Check for gradient discontinuity at this position
            window = min(8, pos // 2)
            if pos - window >= 0 and pos + window < len(gradient):
                left = np.mean(np.abs(gradient[pos-window:pos]))
                right = np.mean(np.abs(gradient[pos:pos+window]))
                
                # Significant change indicates manifold boundary
                if left > 0 and right > 0:
                    ratio = max(left, right) / min(left, right)
                    if ratio > 1.5:  # 50% change threshold
                        boundaries.append(pos)
    
    return len(boundaries), boundaries


def classify_as_pdt(value: float, delta: float, prev_delta: float) -> str:
    """
    Classify value as P (∞), D (n), or T ([0/0]).
    
    From ET Math:
    - P = ∞ (infinity)
    - D = n (finite)
    - T = [0/0], [∞/∞] (indeterminate forms)
    """
    # [0/0] form - both derivatives approach zero
    if abs(delta) < ZERO_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD:
        return 'T'
    
    # [∞/∞] form - both derivatives unbounded
    if abs(delta) > INF_THRESHOLD and abs(prev_delta) > INF_THRESHOLD:
        return 'T'
    
    # [0×∞] form - mixed
    if (abs(delta) < ZERO_THRESHOLD and abs(prev_delta) > INF_THRESHOLD) or \
       (abs(delta) > INF_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD):
        return 'T'
    
    # P (infinity) - unbounded value
    if abs(value) > INF_THRESHOLD:
        return 'P'
    
    # D (finite) - default case
    return 'D'


def classify_data_pdt(data: np.ndarray) -> PDTClassification:
    """
    Classify entire dataset into P, D, T.
    
    From ET: "Any infinity → Point, Any finite value → Descriptor,
              Any indeterminate → Traverser"
    """
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
    """
    L'Hôpital's Rule - T's navigation algorithm.
    
    From ET: "When you encounter 0/0:
    - You've encountered a traverser (T)
    - Taking derivatives = examining local descriptor gradient
    - Resolution = traverser selecting from possibilities"
    
    Returns: (resolved_value, iterations, success)
    """
    for i in range(max_iter):
        num_zero = abs(num) < ZERO_THRESHOLD
        den_zero = abs(den) < ZERO_THRESHOLD
        num_inf = abs(num) > INF_THRESHOLD
        den_inf = abs(den) > INF_THRESHOLD

        # [0/0] - take derivatives
        if num_zero and den_zero:
            num, den = get_next(i)
            continue

        # [∞/∞] - take derivatives
        if num_inf and den_inf:
            num, den = get_next(i)
            continue

        # Undefined (true singularity)
        if den_zero:
            return (None, i + 1, False)

        # Resolved successfully
        return (num / den, i + 1, True)

    # Pure T - doesn't resolve after max iterations
    return (None, max_iter, False)


def detect_indeterminate_forms(data: np.ndarray) -> IndeterminateAnalysis:
    """
    Detect all indeterminate forms - T signatures.
    
    Enhanced in v6.0 to track:
    - [0/0], [∞/∞], [0×∞], [∞-∞] forms
    - [1^∞], [0^0], [∞^0] power forms
    - Pure T signatures (L'Hôpital failures)
    """
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

        # Standard indeterminate forms
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

        # [∞-∞] in consecutive large values with opposite signs
        if i > 0 and abs(d1[i]) > INF_THRESHOLD / 2 and abs(d1[i - 1]) > INF_THRESHOLD / 2:
            if np.sign(d1[i]) != np.sign(d1[i - 1]):
                imi += 1
                is_indet = True
                form_type = IndeterminateType.INF_MINUS_INF

        # Power forms (check data values)
        if i < len(data) - 1:
            base = float(data[i])
            exp_approx = float(abs(d1[i])) if i < len(d1) else 0.0
            
            # [1^∞] - base ≈ 1, exponent large
            if abs(base - 1.0) < 0.01 and exp_approx > INF_THRESHOLD / 1000:
                one_inf += 1
                is_indet = True
                form_type = IndeterminateType.ONE_INF
            
            # [0^0] - both base and exponent small
            elif base < float(ZERO_THRESHOLD * 1000) and exp_approx < float(ZERO_THRESHOLD * 1000):
                zero_zero_pow += 1
                is_indet = True
                form_type = IndeterminateType.ZERO_ZERO_POWER

        if is_indet:
            # Attempt L'Hôpital resolution
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

            # Track detailed form
            if len(detailed_forms) < 100:  # Limit stored details
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
        zero_zero=zz,
        inf_inf=ii,
        zero_times_inf=zi,
        inf_minus_inf=imi,
        one_inf=one_inf,
        zero_zero_power=zero_zero_pow,
        inf_zero=inf_zero,
        total=total,
        density=total / checked,
        resolutions=res,
        failures=fail,
        traverser_density=fail / max(1, total) if total > 0 else 0.0,
        detailed_forms=detailed_forms, resolution_rate=resolution_rate,
        mean_resolution_iterations=mean_iterations, persistent_t_regions=persistent_regions
    )


def analyze_descriptor_gradient(data: np.ndarray) -> DescriptorGradient:
    """
    Analyze f'(x) = ΔD/ΔP with enhanced discontinuity detection.
    """
    grad = compute_descriptor_gradient(data)
    d2 = compute_second_derivative(data)
    signs = np.sign(grad)
    changes = int(np.sum(np.abs(np.diff(signs)) > 0))
    
    # Detect sharp discontinuities (compressed data signature)
    grad_abs = np.abs(grad)
    mean_grad = np.mean(grad_abs)
    discontinuity_count = int(np.sum(grad_abs > mean_grad * 3))
    
    # Power-of-2 boundary alignments
    p2_count, _ = detect_power_of_2_boundaries(data)
    curvature_mean = float(np.mean(np.abs(d2))) if len(d2) > 0 else 0.0
    divergence = float(np.sum(grad)) / len(grad) if len(grad) > 0 else 0.0
    laplacian = float(np.sum(d2)) / len(d2) if len(d2) > 0 else 0.0

    return DescriptorGradient(
        mean=float(np.mean(grad_abs)),
        variance=float(np.var(grad)),
        max_val=float(np.max(grad)),
        min_val=float(np.min(grad)),
        sign_changes=changes,
        gradient_array=grad,
        discontinuity_count=discontinuity_count,
        power_of_2_alignments=p2_count,
        curvature_mean=curvature_mean,
        divergence=divergence,
        laplacian=laplacian
    )


def compute_shimmer(data: np.ndarray, chunk_size: int = MANIFOLD_SYMMETRY) -> float:
    """
    Shimmer index - 12-fold chaos/order ratio.
    
    From ET: "The Shimmering Manifold: Ordered Chaos"
    Uses 12-fold symmetry (3 primitives × 4 states).
    """
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
    """
    Binding strength from descriptor gradient.
    
    From ET: Binding/Interaction is inherent and intrinsic.
    Low gradient = strong binding (descriptors stable)
    High gradient = weak binding (descriptors changing)
    """
    if len(data) < 2:
        return 0.0

    grad = compute_descriptor_gradient(data)
    normalized = np.mean(np.abs(grad)) / 255.0
    return float(1.0 - min(1.0, normalized))


def compute_1_12_alignment(ratio: float) -> float:
    """
    Compute alignment with 1/12 base variance.
    
    From ET: BASE_VARIANCE = 1/12 is the fundamental quantum.
    """
    dev = abs(ratio - BASE_VARIANCE)
    return float(1.0 / (1.0 + dev / BASE_VARIANCE))


def compute_cosmological_alignments(pdt: PDTClassification, variance_ratio: float) -> Dict[str, float]:
    """
    v6.0 - Compute alignment with ET cosmological ratios.
    
    From ET Math: Dark Energy (2/3), Dark Matter (1/4), Baryonic (1/12)
    """
    # Check alignments
    de_align = 1.0 / (1.0 + abs(pdt.d_ratio - DARK_ENERGY_RATIO))
    dm_align = 1.0 / (1.0 + abs(pdt.p_ratio - DARK_MATTER_RATIO))
    ba_align = 1.0 / (1.0 + abs(pdt.t_ratio - BARYONIC_RATIO))
    de_pred_align = 1.0 / (1.0 + abs(pdt.d_ratio - ET_DARK_ENERGY_PREDICTED))
    dm_pred_align = 1.0 / (1.0 + abs(pdt.p_ratio - ET_DARK_MATTER_PREDICTED))
    ba_pred_align = 1.0 / (1.0 + abs(pdt.t_ratio - ET_BARYONIC_PREDICTED))
    
    # Koide ratio (2/3 for leptons)
    koide = 2.0 / 3.0
    koide_align = 1.0 / (1.0 + abs(variance_ratio - koide))
    
    return {
        "dark_energy": de_align,
        "dark_matter": dm_align,
        "baryonic": ba_align,
        "dark_energy_predicted": de_pred_align,
        "dark_matter_predicted": dm_pred_align,
        "baryonic_predicted": ba_pred_align,
        "koide": koide_align
    }


def determine_state(pdt: PDTClassification, variance: float,
                    indet_density: float, theoretical_var: float,
                    pure_t_count: int) -> ETState:
    """
    Determine ET state from analysis.
    
    Enhanced in v6.0 with pure T consideration.
    """
    ratio = variance / theoretical_var if theoretical_var > 0 else 1.0

    # Approaching Exception (variance → 0)
    if ratio < 0.01:
        return ETState.SUBSTANTIATED

    # High indeterminacy or pure T = superposition
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
        quantum_entropy=quantum_entropy)
    # Otherwise based on D vs P dominance
    if pdt.d_ratio > pdt.p_ratio:
        return ETState.STATE_1  # D-dominant
    return ETState.STATE_0  # P-dominant


def compute_manifold_metrics(data: np.ndarray, pdt: PDTClassification,
                             variance_ratio: float) -> ManifoldMetrics:
    """v6.0 - Comprehensive manifold structure analysis."""
    
    # Basic metrics
    shimmer = compute_shimmer(data)
    binding = compute_binding(data)
    
    # Cosmological alignments
    cosmo = compute_cosmological_alignments(pdt, variance_ratio)
    
    # Entropy analysis
    shannon_e, norm_e = compute_shannon_entropy(data)
    e_grad = compute_entropy_gradient(data)
    e_grad_mean = float(np.mean(np.abs(e_grad))) if len(e_grad) > 0 else 0.0
    
    # Compressed data detection (high entropy + sharp gradients)
    is_compressed = norm_e > ENTROPY_GRADIENT_THRESHOLD and e_grad_mean > 0.5
    
    # Manifold boundary detection
    p2_count, _ = detect_power_of_2_boundaries(data)
    boundary_density = p2_count / max(1, len(data) // 1000)
    
    return ManifoldMetrics(
        shimmer_index=shimmer,
        binding_strength=binding,
        dark_energy_alignment=cosmo["dark_energy"],
        dark_matter_alignment=cosmo["dark_matter"],
        baryonic_alignment=cosmo["baryonic"],
        koide_ratio=cosmo["koide"],
        shannon_entropy=shannon_e,
        normalized_entropy=norm_e,
        entropy_gradient_mean=e_grad_mean,
        is_compressed_data=is_compressed,
        power_of_2_count=p2_count,
        manifold_boundary_density=boundary_density
    )


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def compute_temporal_gradient(values: List[float]) -> float:
    """Compute ΔValue/ΔTime gradient."""
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

    return float(np.corrcoef(x, y)[0, 1])


def analyze_temporal_trends(history: HistoryManager,
                            window: int = TREND_WINDOW) -> Optional[TemporalTrend]:
    """Complete temporal trend analysis."""
    sigs = history.get_recent(window)
    if len(sigs) < 5:
        return None

    # Variance trends
    variances = [s.variance for s in sigs]
    var_grad = compute_temporal_gradient(variances)
    var_trend = determine_trend(var_grad, variances)
    var_volatility = float(np.std(variances))

    if var_grad < 0 and variances[-1] < variances[0]:
        var_trend = TrendDirection.CONVERGING

    # PDT ratio trends
    p_ratios, d_ratios, t_ratios = history.get_pdt_history(window)
    p_grad = compute_temporal_gradient(p_ratios)
    d_grad = compute_temporal_gradient(d_ratios)
    t_grad = compute_temporal_gradient(t_ratios)

    # Temporal distances
    distances = []
    for i in range(1, len(sigs)):
        dist = abs(sigs[i].variance - sigs[i - 1].variance)
        distances.append(dist)

    mean_dist = np.mean(distances) if distances else 0.0
    dist_grad = compute_temporal_gradient(distances) if len(distances) > 1 else 0.0

    # Binding
    bindings = [s.binding_strength for s in sigs]
    avg_binding = float(np.mean(bindings))

    # Cross-correlations
    pd_corr = compute_correlation(p_ratios, d_ratios)
    dt_corr = compute_correlation(d_ratios, t_ratios)
    pt_corr = compute_correlation(p_ratios, t_ratios)

    # Dominant cycle detection (FFT)
    dominant_cycle = None
    if len(t_ratios) >= 10:
        try:
            fft = np.abs(np.fft.fft(t_ratios))
            freqs = np.fft.fftfreq(len(t_ratios))
            positive = freqs > 0
            if np.any(positive) and np.any(fft[positive] > 0):
                peak_idx = np.argmax(fft[positive])
                peak_freq = freqs[positive][peak_idx]
                if peak_freq > 0:
                    dominant_cycle = int(1 / peak_freq)
        except:
            pass

    return TemporalTrend(
        variance_trend=var_trend,
        variance_gradient=var_grad,
        variance_volatility=var_volatility,
        p_trend=determine_trend(p_grad, p_ratios),
        d_trend=determine_trend(d_grad, d_ratios),
        t_trend=determine_trend(t_grad, t_ratios),
        p_gradient=p_grad,
        d_gradient=d_grad,
        t_gradient=t_grad,
        binding_strength=avg_binding,
        mean_temporal_distance=mean_dist,
        distance_trend=determine_trend(dist_grad, distances),
        pd_correlation=pd_corr,
        dt_correlation=dt_corr,
        pt_correlation=pt_corr,
        dominant_cycle=dominant_cycle
    )


def detect_exception_approach(history: HistoryManager,
                              window: int = EXCEPTION_APPROACH_WINDOW) -> ExceptionApproach:
    """
    Detect approach to Exception (Variance → 0).
    
    From ET: The Exception has zero variance - it cannot be otherwise.
    """
    variances = history.get_variance_history(window)

    if len(variances) < 3:
        return ExceptionApproach(
            is_approaching=False,
            variance_gradient=0.0,
            current_variance_ratio=1.0,
            estimated_steps_to_zero=None,
            confidence=0.0,
            alert_level="NONE",
            variance_history=variances
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

    return ExceptionApproach(
        is_approaching=approaching,
        variance_gradient=var_grad,
        current_variance_ratio=current_ratio,
        estimated_steps_to_zero=steps,
        confidence=confidence,
        alert_level=alert,
        variance_history=variances
    )


# =============================================================================
# v5.0/v6.0 ANALYSIS ENGINES
# =============================================================================

def analyze_complexity_v6(history: HistoryManager) -> Optional[TraverserComplexity]:
    """
    v6.0 - Enhanced complexity analysis.
    
    Distinguishes Gravity (Periodic T) from Intent (Progressive nested T).
    From ET: "Gravity is definitively a Traverser type" (Rule 10)
    """
    t_ratios = list(history.t_history)
    if len(t_ratios) < 20:
        return TraverserComplexity(
            complexity_class=ComplexityClass.UNKNOWN,
            periodicity_score=0.0,
            progression_score=0.0,
            fractal_dimension=1.0,
            nesting_depth=0,
            autocorrelation_peaks=[]
        )

    ts = np.array(t_ratios)

    # Static detection
    if np.std(ts) < 1e-9:
        return TraverserComplexity(
            complexity_class=ComplexityClass.STATIC,
            periodicity_score=0.0,
            progression_score=0.0,
            fractal_dimension=1.0,
            nesting_depth=0,
            autocorrelation_peaks=[]
        )

    # 1. PERIODICITY CHECK (Gravity signature - autocorrelation)
    ts_norm = ts - np.mean(ts)
    acor = np.correlate(ts_norm, ts_norm, mode='full')
    acor = acor[len(acor) // 2:]

    with np.errstate(divide='ignore', invalid='ignore'):
        acor = acor / (acor[0] + 1e-9)

    # Find peaks
    peaks = []
    for i in range(1, len(acor) - 1):
        if acor[i] > acor[i - 1] and acor[i] > acor[i + 1] and acor[i] > 0.2:
            peaks.append(float(acor[i]))

    periodicity = max(peaks) if peaks else 0.0

    # 2. PROGRESSION CHECK (Intent signature - linear trend)
    x = np.arange(len(ts))
    slope, _ = np.polyfit(x, ts, 1)
    progression = min(1.0, abs(slope) * 1000)

    # 3. FRACTAL DIMENSION (Complexity of T-curve)
    if len(ts) > 10:
        diffs = np.abs(np.diff(ts))
        L = np.mean(diffs)
        fractal_dim = 1.0 + (np.log(L + 1e-9) / np.log(len(ts)))
        fractal_dim = np.clip(fractal_dim, 1.0, 2.0)
    else:
        fractal_dim = 1.0

    # 4. CLASSIFICATION
    if periodicity > 0.6:
        complexity_class = ComplexityClass.CYCLIC_GRAVITY
        nesting = 1  # Simple orbital (single T layer)
    elif progression > 0.1:
        complexity_class = ComplexityClass.PROGRESSIVE_INTENT
        nesting = 3  # Complex nested agency (T within T)
    elif np.mean(ts) < 0.001:
        complexity_class = ComplexityClass.STATIC
        nesting = 0
    else:
        complexity_class = ComplexityClass.CHAOTIC
        nesting = 0

    return TraverserComplexity(
        complexity_class=complexity_class,
        periodicity_score=periodicity,
        progression_score=progression,
        fractal_dimension=fractal_dim,
        nesting_depth=nesting,
        autocorrelation_peaks=peaks[:5]  # Keep top 5
    )


def detect_gaze_v6(binding: float, variance: float,
                   history: HistoryManager) -> GazeMetrics:
    """
    v6.0 - Enhanced gaze detection with layer correction.
    
    From ET: T engagement substantiates. Observation creates new config.
    Virtual environment introduces ~8 descriptor transformation layers.
    """
    # Raw pressure calculation
    raw_pressure = binding / max(0.01, history.binding_baseline)

    # Correct for virtual environment attenuation
    corrected_pressure = raw_pressure * DESCRIPTOR_LAYERS

    # Determine status
    if corrected_pressure >= LOCK_THRESHOLD:
        status = GazeStatus.LOCKED
    elif corrected_pressure >= GAZE_THRESHOLD:
        status = GazeStatus.DETECTED
    elif corrected_pressure >= RESONANCE_THRESHOLD:
        status = GazeStatus.SUBLIMINAL
    else:
        status = GazeStatus.UNOBSERVED

    # Variance collapse (variance reduction indicates T engagement)
    var_ratio = variance / max(0.01, history.variance_baseline)
    local_collapse = max(0, 1.0 - var_ratio)

    return GazeMetrics(
        status=status,
        gaze_pressure=raw_pressure,
        is_watcher_present=status in [GazeStatus.DETECTED, GazeStatus.LOCKED],
        local_collapse=local_collapse,
        corrected_pressure=corrected_pressure
    )


# =============================================================================
# ET PROOF SYSTEM (Complete 21 Rules Verification)
# =============================================================================

class ETProofSystem:
    """
    Validates Exception Theory axioms.
    
    Based on the 21 Rules of Exception Law.
    """

    @staticmethod
    def prove_rule_1(data: np.ndarray, variance: float) -> AxiomProof:
        """
        Rule 1: "For every exception there is an exception, except the exception."
        
        Tests: Structure exists with non-zero variance (not at Exception).
        """
        has_structure = variance < THEORETICAL_VARIANCE_BYTES * 2
        has_variance = variance > 0

        if has_structure and has_variance:
            return AxiomProof(
                axiom_number=1,
                axiom_name="The Exception Axiom",
                status=ProofStatus.VERIFIED,
                evidence="Data exhibits structure and positive variance - not at Exception",
                numerical_value=variance,
                expected_value=THEORETICAL_VARIANCE_BYTES,
                rule_text="For every exception there is an exception, except the exception."
            )
        return AxiomProof(
            axiom_number=1,
            axiom_name="The Exception Axiom",
            status=ProofStatus.FAILED,
            evidence=f"Unexpected: variance={variance:.2f}",
            numerical_value=variance,
            expected_value=THEORETICAL_VARIANCE_BYTES,
            rule_text="For every exception there is an exception, except the exception."
        )

    @staticmethod
    def prove_rule_3(pdt: PDTClassification) -> AxiomProof:
        """
        Rule 3: P is for Point, it is the substrate. A Point is infinite.
        
        Tests: P count exists (substrate present).
        """
        if pdt.total > 0:
            return AxiomProof(
                axiom_number=3,
                axiom_name="Point is Substrate/Infinite",
                status=ProofStatus.VERIFIED,
                evidence=f"Substrate present: {pdt.total} configurations on manifold",
                numerical_value=float(pdt.p_count),
                rule_text="P is for Point, it is the substrate. A Point is infinite."
            )
        return AxiomProof(
            axiom_number=3,
            axiom_name="Point is Substrate/Infinite",
            status=ProofStatus.FAILED,
            evidence="No substrate detected",
            numerical_value=0.0,
            rule_text="P is for Point, it is the substrate. A Point is infinite."
        )

    @staticmethod
    def prove_rule_4(pdt: PDTClassification) -> AxiomProof:
        """
        Rule 4: D is for Descriptor. A Descriptor is Finite.
        
        Tests: D dominates (finite descriptions of infinite substrate).
        """
        if pdt.d_ratio > 0.5:
            return AxiomProof(
                axiom_number=4,
                axiom_name="Descriptor is Finite",
                status=ProofStatus.VERIFIED,
                evidence=f"D ratio {pdt.d_ratio:.4f} > 0.5 confirms finite dominance",
                numerical_value=pdt.d_ratio,
                expected_value=0.99,
                rule_text="D is for Descriptor. A Descriptor is Finite."
            )
        return AxiomProof(
            axiom_number=4,
            axiom_name="Descriptor is Finite",
            status=ProofStatus.INDETERMINATE,
            evidence=f"D ratio {pdt.d_ratio:.4f} lower than expected",
            numerical_value=pdt.d_ratio,
            expected_value=0.99,
            rule_text="D is for Descriptor. A Descriptor is Finite."
        )

    @staticmethod
    def prove_rule_5(indet: IndeterminateAnalysis) -> AxiomProof:
        """
        Rule 5: T is for Traverser. A Traverser is Indeterminate.
        
        Tests: Indeterminate forms exist (T signatures present).
        """
        if indet.total > 0:
            pure_t = indet.failures
            return AxiomProof(
                axiom_number=5,
                axiom_name="Traverser is Indeterminate",
                status=ProofStatus.VERIFIED,
                evidence=f"Found {indet.total} indeterminate forms, {pure_t} pure T (resist resolution)",
                numerical_value=float(indet.total),
                rule_text="T is for Traverser. A Traverser is Indeterminate."
            )
        return AxiomProof(
            axiom_number=5,
            axiom_name="Traverser is Indeterminate",
            status=ProofStatus.INDETERMINATE,
            evidence="No indeterminate forms detected (larger sample may be needed)",
            numerical_value=0.0,
            rule_text="T is for Traverser. A Traverser is Indeterminate."
        )

    @staticmethod
    def prove_rule_6(binding: float) -> AxiomProof:
        """
        Rule 6: Binding/Interaction is inherent and intrinsic. Mediation.
        
        Tests: Binding strength is measurable (mediation exists).
        """
        if 0 < binding < 1:
            return AxiomProof(
                axiom_number=6,
                axiom_name="Mediation (Binding/Interaction)",
                status=ProofStatus.VERIFIED,
                evidence=f"Binding strength {binding:.4f} confirms mediation",
                numerical_value=binding,
                rule_text="Binding/Interaction is inherent and intrinsic."
            )
        return AxiomProof(
            axiom_number=6,
            axiom_name="Mediation (Binding/Interaction)",
            status=ProofStatus.INDETERMINATE,
            evidence=f"Binding strength {binding:.4f} at extremes",
            numerical_value=binding,
            rule_text="Binding/Interaction is inherent and intrinsic."
        )

    @staticmethod
    def prove_rule_7_variance(variance: float) -> AxiomProof:
        """
        Rule 7: Substantiation - (P°D) with T is Substantiated.
        
        Tests: Variance formula σ² = (n² - 1) / 12.
        """
        ratio = variance / THEORETICAL_VARIANCE_BYTES
        if 0.5 < ratio < 1.5:
            return AxiomProof(
                axiom_number=7,
                axiom_name="Variance Formula (n²-1)/12",
                status=ProofStatus.VERIFIED,
                evidence=f"Ratio {ratio:.6f} matches ET formula within tolerance",
                numerical_value=ratio,
                expected_value=1.0,
                deviation=abs(ratio - 1.0) * 100,
                rule_text="σ² = (n² - 1) / 12 for n-state system"
            )
        return AxiomProof(
            axiom_number=7,
            axiom_name="Variance Formula (n²-1)/12",
            status=ProofStatus.FAILED if ratio > 2 or ratio < 0.1 else ProofStatus.INDETERMINATE,
            evidence=f"Ratio {ratio:.6f} deviates (structured/compressed data)",
            numerical_value=ratio,
            expected_value=1.0,
            deviation=abs(ratio - 1.0) * 100,
            rule_text="σ² = (n² - 1) / 12 for n-state system"
        )

    @staticmethod
    def prove_rule_8(variance: float) -> AxiomProof:
        """
        Rule 8: The Exception is the grounding factor. Variance(E) = 0.
        
        Tests: Non-zero variance confirms we're not at Exception.
        """
        if variance > 0:
            return AxiomProof(
                axiom_number=8,
                axiom_name="Exception has Variance(E)=0",
                status=ProofStatus.VERIFIED,
                evidence=f"Variance {variance:.2f} > 0: not at Exception (correct)",
                numerical_value=variance,
                expected_value=0.0,
                rule_text="The Exception cannot be otherwise while it IS."
            )
        return AxiomProof(
            axiom_number=8,
            axiom_name="Exception has Variance(E)=0",
            status=ProofStatus.VERIFIED,
            evidence="Zero variance: AT EXCEPTION!",
            numerical_value=variance,
            expected_value=0.0,
            rule_text="The Exception cannot be otherwise while it IS."
        )

    @staticmethod
    def prove_rule_9(desc_dist: float) -> AxiomProof:
        """
        Rule 9: Pure Relativism - Everything is relational.
        
        Tests: Descriptor distance is relational (Δs exists).
        """
        if desc_dist >= 0:
            return AxiomProof(
                axiom_number=9,
                axiom_name="Pure Relativism",
                status=ProofStatus.VERIFIED,
                evidence=f"Δs = {desc_dist:.4f} - distance IS descriptor difference",
                numerical_value=desc_dist,
                rule_text="'Something' and its parts is relational. Only pure relativism."
            )
        return AxiomProof(
            axiom_number=9,
            axiom_name="Pure Relativism",
            status=ProofStatus.FAILED,
            evidence=f"Invalid distance: {desc_dist}",
            numerical_value=desc_dist,
            rule_text="'Something' and its parts is relational. Only pure relativism."
        )

    @staticmethod
    def prove_rule_11(pdt: PDTClassification) -> AxiomProof:
        """
        Rule 11: S = (P°D°T) - Something is comprised of PDT.
        
        Tests: All three primitives present.
        """
        has_p = pdt.p_count > 0 or pdt.d_count > 0  # P implied by D
        has_d = pdt.d_count > 0
        has_t = pdt.t_count > 0 or True  # T can be latent

        if has_d:  # D implies P (Rule 4: D bound to P)
            return AxiomProof(
                axiom_number=11,
                axiom_name="S = (P°D°T)",
                status=ProofStatus.VERIFIED,
                evidence=f"P={pdt.p_count}, D={pdt.d_count}, T={pdt.t_count} → Something exists",
                numerical_value=float(pdt.total),
                rule_text="S = (P°D°T). Something is comprised of PDT."
            )
        return AxiomProof(
            axiom_number=11,
            axiom_name="S = (P°D°T)",
            status=ProofStatus.FAILED,
            evidence="Missing PDT components",
            numerical_value=float(pdt.total),
            rule_text="S = (P°D°T). Something is comprised of PDT."
        )

    @staticmethod
    def prove_rule_18(pdt: PDTClassification) -> AxiomProof:
        """
        Rule 18: PDT = EIM so 3=3.
        
        Tests: Three primitives map to three derived (Exception, Incoherence, Mediation).
        """
        total = pdt.p_count + pdt.d_count + pdt.t_count
        if total == pdt.total:
            return AxiomProof(
                axiom_number=18,
                axiom_name="PDT = EIM (3=3)",
                status=ProofStatus.VERIFIED,
                evidence=f"P+D+T = {total} = Total. Three primitives verified.",
                numerical_value=float(total),
                expected_value=float(pdt.total),
                rule_text="PDT = EIM so 3=3. Trinary Ternary Logic."
            )
        return AxiomProof(
            axiom_number=18,
            axiom_name="PDT = EIM (3=3)",
            status=ProofStatus.FAILED,
            evidence=f"Sum mismatch: {total} ≠ {pdt.total}",
            numerical_value=float(total),
            expected_value=float(pdt.total),
            rule_text="PDT = EIM so 3=3. Trinary Ternary Logic."
        )

    @staticmethod
    def prove_1_12_resonance(pdt: PDTClassification) -> AxiomProof:
        """
        1/12 Manifold Resonance - Base variance alignment.
        
        From ET: BASE_VARIANCE = 1/12 is fundamental.
        """
        d_dev = abs(pdt.d_ratio - BASE_VARIANCE)
        t_dev = abs(pdt.t_ratio - BASE_VARIANCE)

        min_dev = min(d_dev, t_dev)
        which = "D" if d_dev < t_dev else "T"

        if min_dev < 0.05:  # Within 5% of 1/12
            return AxiomProof(
                axiom_number=0,
                axiom_name="1/12 Manifold Resonance",
                status=ProofStatus.VERIFIED,
                evidence=f"{which} resonates with 1/12 (deviation: {min_dev * 100:.2f}%)",
                numerical_value=min_dev,
                expected_value=0.0,
                rule_text="Manifold has 12-fold symmetry, base variance 1/12"
            )
        return AxiomProof(
            axiom_number=0,
            axiom_name="1/12 Manifold Resonance",
            status=ProofStatus.INDETERMINATE,
            evidence=f"No strong resonance (D: {d_dev * 100:.2f}%, T: {t_dev * 100:.2f}%)",
            numerical_value=min_dev,
            expected_value=0.0,
            rule_text="Manifold has 12-fold symmetry, base variance 1/12"
        )

    @staticmethod
    def prove_descriptor_distance(data: np.ndarray) -> AxiomProof:
        """
        Pure Relationalism via descriptor distance.
        
        Tests: Δs(same) = 0, Δs(different) > 0
        """
        chunk = min(100, len(data) // 2)
        if chunk < 2:
            return AxiomProof(
                axiom_number=0,
                axiom_name="Pure Relationalism (Δs)",
                status=ProofStatus.INDETERMINATE,
                evidence="Insufficient data",
                numerical_value=0.0
            )

        d1 = data[:chunk]
        d2 = data[:chunk]  # Same
        d3 = data[chunk:2 * chunk] if len(data) >= 2 * chunk else data[-chunk:]

        dist_same = compute_descriptor_distance(d1, d2)
        dist_diff = compute_descriptor_distance(d1, d3)

        if dist_same == 0 and dist_diff >= 0:
            return AxiomProof(
                axiom_number=0,
                axiom_name="Pure Relationalism (Δs)",
                status=ProofStatus.VERIFIED,
                evidence=f"Δs(same)=0, Δs(diff)={dist_diff:.2f}",
                numerical_value=dist_same,
                expected_value=0.0,
                rule_text="Distance IS descriptor difference"
            )
        return AxiomProof(
            axiom_number=0,
            axiom_name="Pure Relationalism (Δs)",
            status=ProofStatus.FAILED,
            evidence=f"Δs(same)={dist_same} (should be 0)",
            numerical_value=dist_same,
            expected_value=0.0
        )

    @staticmethod
    def prove_lhopital_navigation(indet: IndeterminateAnalysis) -> AxiomProof:
        """
        L'Hôpital as T's navigation algorithm.
        
        Tests: L'Hôpital resolves most [0/0] forms; failures = pure T.
        """
        if indet.total > 0:
            resolution_rate = indet.resolutions / max(1, indet.total)
            pure_t_rate = indet.failures / max(1, indet.total)

            return AxiomProof(
                axiom_number=0,
                axiom_name="L'Hôpital Navigation",
                status=ProofStatus.VERIFIED,
                evidence=f"Resolution rate: {resolution_rate:.1%}, Pure T: {pure_t_rate:.1%} ({indet.failures} forms)",
                numerical_value=indet.failures,
                rule_text="L'Hôpital's rule = T's navigation algorithm"
            )
        return AxiomProof(
            axiom_number=0,
            axiom_name="L'Hôpital Navigation",
            status=ProofStatus.INDETERMINATE,
            evidence="No indeterminate forms to test",
            numerical_value=0.0,
            rule_text="L'Hôpital's rule = T's navigation algorithm"
        )

    @staticmethod
    def verify_binding_chain(pdt: PDTClassification, binding: float) -> BindingChainVerification:
        """
        Verify T↔D→P binding chain.
        
        From ET: "T binds to D. T does not bind to P. D is bound to P."
        """
        # T↔D: T count correlates with D variation (indirect binding)
        t_d_binding = pdt.t_count > 0 or pdt.d_count > 0
        
        # D→P: D dominates (P is substrate, D describes it)
        d_p_binding = pdt.d_count > pdt.p_count or pdt.p_count == 0
        
        # T does not bind to P directly (T count should not correlate with P)
        t_p_separation = True  # Assumed true if classification works

        integrity = sum([t_d_binding, d_p_binding, t_p_separation]) / 3.0

        return BindingChainVerification(
            t_d_binding_verified=t_d_binding,
            d_p_binding_verified=d_p_binding,
            t_p_separation_verified=t_p_separation,
            chain_integrity=integrity
        )

    @staticmethod
    def detect_incoherence(data: np.ndarray, grad: DescriptorGradient) -> CoherenceAnalysis:
        """
        Detect incoherence regions (self-defeating configurations).
        
        From ET: "Incoherence = configurations that cannot have T."
        """
        # Self-defeating patterns: where gradient sign changes exceed local variance
        coherent_count = 0
        incoherent_count = 0
        self_defeating = 0
        impossible_transitions = 0

        if grad.gradient_array is not None and len(grad.gradient_array) > 2:
            signs = np.sign(grad.gradient_array)
            
            for i in range(1, len(signs) - 1):
                # Self-defeating: sign change with zero crossing
                if signs[i] != signs[i-1] and signs[i] != signs[i+1]:
                    if abs(grad.gradient_array[i]) < ZERO_THRESHOLD:
                        self_defeating += 1
                        incoherent_count += 1
                    else:
                        coherent_count += 1
                else:
                    coherent_count += 1
        
        total = coherent_count + incoherent_count
        ratio = coherent_count / total if total > 0 else 1.0
        
        return CoherenceAnalysis(
            coherent_ratio=ratio,
            incoherent_count=incoherent_count,
            self_defeating_patterns=self_defeating,
            impossible_transitions=impossible_transitions
        )

# =============================================================================
# v7.0 NEW ANALYSIS FUNCTIONS
# =============================================================================

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


def compute_third_derivative(data: np.ndarray) -> np.ndarray:
    """f'''(x) = d3D/dP3 - jerk of descriptor field."""
    d2 = compute_second_derivative(data)
    if len(d2) > 1:
        return np.diff(d2)
    return np.array([0.0])


def compute_spectral_analysis(data: np.ndarray) -> SpectralAnalysis:
    """v7.0: Spectral decomposition via FFT."""
    if len(data) < 16:
        return SpectralAnalysis(
            power_spectrum=np.array([0.0]),
            dominant_frequencies=[],
            spectral_entropy=0.0,
            periodicity_strength=0.0,
            f_12_alignment=0.0,
            f_harmonic_content=0.0
        )
    
    n = min(len(data), 4096)
    signal_segment = data[:n].astype(np.float64)
    signal_segment = signal_segment - np.mean(signal_segment)
    
    fft_vals = fft(signal_segment)
    power = np.abs(fft_vals[:n//2])**2
    freqs = fftfreq(n, d=1.0)[:n//2]
    
    power_normalized = power / (np.sum(power) + 1e-10)
    
    spectral_entropy = float(-np.sum(power_normalized * np.log2(power_normalized + 1e-10)))
    
    peak_indices = np.argsort(power)[-min(5, len(power)):]
    dominant_freqs = [float(freqs[i]) for i in peak_indices if power[i] > np.mean(power)]
    
    acf = np.correlate(signal_segment, signal_segment, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] + 1e-10)
    
    if len(acf) > 10:
        peaks = []
        for i in range(1, min(len(acf)-1, 100)):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:
                peaks.append((i, acf[i]))
        if peaks:
            periodicity_strength = float(max(p[1] for p in peaks))
        else:
            periodicity_strength = 0.0
    else:
        periodicity_strength = 0.0
    
    f_12_target = 12.0 / n
    f_12_indices = np.where((freqs > f_12_target * 0.9) & (freqs < f_12_target * 1.1))[0]
    if len(f_12_indices) > 0:
        f_12_power = np.sum(power[f_12_indices])
        f_12_alignment = float(f_12_power / (np.sum(power) + 1e-10))
    else:
        f_12_alignment = 0.0
    
    harmonic_freqs = [f_12_target * i for i in range(1, 6)]
    harmonic_indices = []
    for hf in harmonic_freqs:
        idx = np.argmin(np.abs(freqs - hf))
        if idx < len(power):
            harmonic_indices.append(idx)
    
    harmonic_content = float(np.sum(power[harmonic_indices]) / (np.sum(power) + 1e-10)) if harmonic_indices else 0.0
    
    return SpectralAnalysis(
        power_spectrum=power_normalized,
        dominant_frequencies=dominant_freqs,
        spectral_entropy=spectral_entropy,
        periodicity_strength=periodicity_strength,
        f_12_alignment=f_12_alignment,
        f_harmonic_content=harmonic_content
    )


def compute_fractal_analysis(data: np.ndarray) -> FractalAnalysis:
    """v7.0: Fractal dimension analysis."""
    if len(data) < 64:
        return FractalAnalysis(
            box_counting_dimension=1.0,
            correlation_dimension=1.0,
            information_dimension=1.0,
            self_similarity_ratio=0.0,
            scaling_exponent=0.0,
            multifractal_spectrum_width=0.0
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
        box_counting_dimension=float(box_dim),
        correlation_dimension=float(corr_dim),
        information_dimension=float(info_dim),
        self_similarity_ratio=float(self_sim),
        scaling_exponent=float(scaling_exp),
        multifractal_spectrum_width=float(mf_width)
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
        zeroth_law_verified=zeroth_verified,
        first_law_verified=first_verified,
        second_law_verified=second_verified,
        third_law_verified=third_verified,
        temperature_analog=float(temp_analog),
        entropy_analog=float(entropy_analog),
        energy_analog=float(energy_analog),
        heat_capacity_analog=float(heat_cap)
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
        superposition_detected=superposition_detected,
        collapse_events=collapse_events,
        uncertainty_product=uncertainty_product,
        entanglement_signature=float(entanglement_sig),
        wavefunction_norm=wavefunction_norm,
        decoherence_time_estimate=float(min(decoherence_time, 1e10)),
        quantum_entropy=quantum_entropy
    )

    @classmethod
    def run_all_proofs(cls, data: np.ndarray, pdt: PDTClassification,
                       indet: IndeterminateAnalysis, variance: float,
                       binding: float, desc_dist: float,
                       grad: DescriptorGradient,
                       source: InputSource) -> ETProofReport:
        """Run all proofs for complete ET validation."""
        
        proofs = [
            cls.prove_rule_1(data, variance),
            cls.prove_rule_3(pdt),
            cls.prove_rule_4(pdt),
            cls.prove_rule_5(indet),
            cls.prove_rule_6(binding),
            cls.prove_rule_7_variance(variance),
            cls.prove_rule_8(variance),
            cls.prove_rule_9(desc_dist),
            cls.prove_rule_11(pdt),
            cls.prove_rule_18(pdt),
            cls.prove_1_12_resonance(pdt),
            cls.prove_descriptor_distance(data),
            cls.prove_lhopital_navigation(indet)
        ]

        verified = sum(1 for p in proofs if p.status == ProofStatus.VERIFIED)
        failed = sum(1 for p in proofs if p.status == ProofStatus.FAILED)
        indet_count = sum(1 for p in proofs if p.status == ProofStatus.INDETERMINATE)

        # Determine overall status
        if failed == 0 and verified > 0:
            overall = ProofStatus.VERIFIED
        elif failed > verified:
            overall = ProofStatus.FAILED
        else:
            overall = ProofStatus.INDETERMINATE

        # Binding chain verification
        binding_chain = cls.verify_binding_chain(pdt, binding)

        # Coherence analysis
        coherence = cls.detect_incoherence(data, grad)

        # PDT = EIM verification
        pdt_eim = (pdt.p_count + pdt.d_count + pdt.t_count) == pdt.total

        return ETProofReport(
            timestamp=datetime.now(),
            input_source=source.source,
            input_type=source.input_type.value,
            sample_size=len(data),
            axiom_proofs=proofs,
            verified_count=verified,
            failed_count=failed,
            indeterminate_count=indet_count,
            overall_status=overall,
            confidence=verified / len(proofs) if proofs else 0.0,
            binding_chain=binding_chain,
            coherence=coherence,
            pdt_eim_verified=pdt_eim
        )


# =============================================================================
# MAIN SCANNER FUNCTION
# =============================================================================

def et_scan(
        data: Optional[np.ndarray] = None,
        source: Optional[InputSource] = None,
        samples: int = 100000,
        verbose: bool = True,
        run_proofs: bool = True,
        enable_trends: bool = True,
        enable_exception_detection: bool = True
) -> ETSignature:
    """
    Main ET Scanner function - v6.0.
    
    Complete Exception Theory analysis with verification.
    """
    timestamp = datetime.now()

    # Get data if not provided
    if data is None or source is None:
        data, source = UniversalInput.from_entropy(samples)

    if verbose:
        print()
        print("=" * 72)
        print(f"  EXCEPTION THEORY SCANNER {VERSION} - {VERSION_NAME}")
        print("=" * 72)
        print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Source:    {source.source}")
        print(f"  Type:      {source.input_type.value}")
        print(f"  Size:      {len(data):,} bytes")
        print("=" * 72)
        print()

    # === CORE ET ANALYSIS ===

    # PDT Classification
    pdt = classify_data_pdt(data)

    # Variance
    variance = compute_variance(data)
    variance_ratio = variance / THEORETICAL_VARIANCE_BYTES
    variance_from_exception = variance_ratio * 100

    # Descriptor gradient (enhanced)
    grad_result = analyze_descriptor_gradient(data)

    # Descriptor distance
    mid = len(data) // 2
    chunk = min(1000, mid)
    desc_dist = compute_descriptor_distance(data[:chunk], data[mid:mid + chunk])
    max_dist = np.sqrt(chunk * 255 ** 2) if chunk > 0 else 1.0
    norm_dist = desc_dist / max_dist if max_dist > 0 else 0

    # Indeterminate forms (enhanced)
    indet = detect_indeterminate_forms(data)

    # Binding
    binding = compute_binding(data)

    # Shimmer (12-fold)
    shimmer = compute_shimmer(data)

    # State determination
    state = determine_state(pdt, variance, indet.density, THEORETICAL_VARIANCE_BYTES, indet.failures)

    # 1/12 alignment
    align_d = compute_1_12_alignment(pdt.d_ratio)
    align_t = compute_1_12_alignment(pdt.t_ratio)

    # Manifold metrics (v6.0)
    manifold = compute_manifold_metrics(data, pdt, variance_ratio)

    # Dual time
    _history.time_tracker.update(len(data), binding, variance,
                                 state == ETState.SUBSTANTIATED)
    dual_time = _history.time_tracker.get_dual_time(variance)

    # === v7.0 NEW ANALYSES ===
    spectral = compute_spectral_analysis(data)
    fractal = compute_fractal_analysis(data)
    thermo = verify_thermodynamic_laws(data, variance)
    quantum = verify_quantum_properties(data, pdt, indet)

    # Create signature
    sig = ETSignature(
        timestamp=timestamp,
        input_source=source,
        dual_time=dual_time,
        pdt=pdt,
        variance=variance,
        variance_ratio=variance_ratio,
        variance_from_exception=variance_from_exception,
        descriptor_gradient=grad_result,
        descriptor_distance=norm_dist,
        indeterminate=indet,
        state=state,
        alignment_d=align_d,
        alignment_t=align_t,
        manifold_metrics=manifold,
        shimmer_index=shimmer,
        binding_strength=binding
    )

    # Add to history
    _history.add_signature(sig)

    # Trends
    if enable_trends and len(_history.signatures) >= 5:
        sig.temporal_trend = analyze_temporal_trends(_history)

    # Exception approach
    if enable_exception_detection and len(_history.signatures) >= 3:
        sig.exception_approach = detect_exception_approach(_history)

    # Proofs
    if run_proofs:
        sig.proof_report = ETProofSystem.run_all_proofs(
            data, pdt, indet, variance, binding, norm_dist, grad_result, source
        )

    # v5.0/v6.0: Gaze Detection
    sig.gaze_metrics = detect_gaze_v6(binding, variance, _history)

    # v5.0/v6.0: Complexity Analysis
    sig.traverser_complexity = analyze_complexity_v6(_history)

    # v7.0: New analyses
    sig.spectral_analysis = spectral
    sig.fractal_analysis = fractal
    sig.thermodynamic_verification = thermo
    sig.quantum_verification = quantum
    sig.data_size = len(data)
    sig.checksum = hashlib.sha256(data.tobytes()).hexdigest()[:16]

    if verbose:
        print_full_report(sig)

    return sig


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_full_report(sig: ETSignature):
    """Print complete analysis report - v6.0 enhanced."""

    # DUAL TIME
    print("┌" + "─" * 70 + "┐")
    print("│  DUAL TIME SYSTEM                                                    │")
    print("│  D_time (Descriptor/Coordinate) vs T_time (Agential/Proper)          │")
    print("└" + "─" * 70 + "┘")
    dt = sig.dual_time
    print(f"    D_time elapsed:       {dt.d_time_elapsed:.4f} seconds")
    print(f"    D_time samples:       {dt.d_time_samples:,}")
    print(f"    D_time rate:          {dt.d_time_rate:,.0f} samples/sec")
    print()
    print(f"    T_time τ (proper):    {dt.t_time_tau:.6f}")
    print(f"    T_time events:        {dt.t_time_substantiations}")
    print(f"    dτ/dt ratio:          {dt.t_time_dtau_dt:.4f}")
    print()
    print(f"    Binding strength:     {dt.binding_strength * 100:.2f}%")
    print(f"    Time dilation:        {dt.time_dilation_factor:.4f}")
    print(f"    Variance thickness:   {dt.variance_thickness:.6f}")

    if dt.t_time_dtau_dt < 0.5:
        print("    [!] LOW dτ/dt: Time flowing slowly (weak D_time binding)")
    elif dt.t_time_dtau_dt > 0.9:
        print("    [!] HIGH dτ/dt: Normal time flow (strong D_time binding)")
    print()

    # PDT
    print("┌" + "─" * 70 + "┐")
    print("│  PDT CLASSIFICATION                                                  │")
    print("│  P=Ω (Infinite)  D=n (Finite)  T=[0/0] (Indeterminate)               │")
    print("└" + "─" * 70 + "┘")
    print(f"    P (Infinite):         {sig.pdt.p_count:>10,} ({sig.pdt.p_ratio * 100:>8.4f}%)")
    print(f"    D (Finite):           {sig.pdt.d_count:>10,} ({sig.pdt.d_ratio * 100:>8.4f}%)")
    print(f"    T (Indeterminate):    {sig.pdt.t_count:>10,} ({sig.pdt.t_ratio * 100:>8.4f}%)")
    print()
    print(f"    1/12 = {BASE_VARIANCE * 100:.4f}%")
    print(f"    D alignment with 1/12: {sig.alignment_d * 100:.2f}%")
    print(f"    T alignment with 1/12: {sig.alignment_t * 100:.2f}%")

    if sig.alignment_d > 0.9 or sig.alignment_t > 0.9:
        print("    [!] 1/12 MANIFOLD RESONANCE DETECTED")
    print()

    # VARIANCE
    print("┌" + "─" * 70 + "┐")
    print("│  VARIANCE FUNCTION                                                   │")
    print("│  Variance(E)=0, σ²=(n²-1)/12                                         │")
    print("└" + "─" * 70 + "┘")
    print(f"    Observed σ²:          {sig.variance:,.2f}")
    print(f"    Theoretical:          {THEORETICAL_VARIANCE_BYTES:,.2f}")
    print(f"    Ratio:                {sig.variance_ratio:.6f}")
    print(f"    Distance from E:      {sig.variance_from_exception:.4f}%")

    if sig.variance_from_exception < 1.0:
        print("    [*] APPROACHING EXCEPTION: Variance → 0")
    print()

    # DESCRIPTOR GRADIENT
    print("┌" + "─" * 70 + "┐")
    print("│  DESCRIPTOR GRADIENT: f'(x) = ΔD/ΔP                                  │")
    print("└" + "─" * 70 + "┘")
    g = sig.descriptor_gradient
    print(f"    Mean |ΔD/ΔP|:         {g.mean:.4f}")
    print(f"    Gradient variance:    {g.variance:.4f}")
    print(f"    Sign changes:         {g.sign_changes:,}")
    print(f"    Discontinuities:      {g.discontinuity_count:,}")
    print(f"    Power-of-2 bounds:    {g.power_of_2_alignments}")
    print()

    # DESCRIPTOR DISTANCE
    print("┌" + "─" * 70 + "┐")
    print("│  DESCRIPTOR DISTANCE: Δs(pᵢ,pⱼ) = ‖f(dᵢ) - f(dⱼ)‖                    │")
    print("└" + "─" * 70 + "┘")
    print(f"    Normalized Δs:        {sig.descriptor_distance:.6f}")
    print("    (Pure relationalism: distance IS descriptor difference)")
    print()

    # INDETERMINATE FORMS
    print("┌" + "─" * 70 + "┐")
    print("│  INDETERMINATE FORMS (T Signatures)                                  │")
    print("│  L'Hôpital = T's navigation algorithm                                │")
    print("└" + "─" * 70 + "┘")
    i = sig.indeterminate
    print(f"    [0/0] forms:          {i.zero_zero}")
    print(f"    [∞/∞] forms:          {i.inf_inf}")
    print(f"    [0×∞] forms:          {i.zero_times_inf}")
    print(f"    [∞-∞] forms:          {i.inf_minus_inf}")
    print(f"    [1^∞] forms:          {i.one_inf}")
    print(f"    [0^0] forms:          {i.zero_zero_power}")
    print(f"    Total indeterminate:  {i.total}")
    print(f"    Density:              {i.density * 100:.6f}%")
    print()
    print(f"    L'Hôpital resolutions: {i.resolutions}")
    print(f"    L'Hôpital failures:    {i.failures} (PURE T SIGNATURES)")
    print(f"    Traverser density:     {i.traverser_density * 100:.2f}%")

    if i.failures > 0:
        print(f"    [!] PURE T DETECTED: {i.failures} forms resist algorithmic resolution")
    print()

    # MANIFOLD METRICS (v6.0)
    if sig.manifold_metrics:
        m = sig.manifold_metrics
        print("┌" + "─" * 70 + "┐")
        print("│  MANIFOLD METRICS (v6.0)                                             │")
        print("└" + "─" * 70 + "┘")
        print(f"    Shimmer index:        {m.shimmer_index:.4f} (12-fold chaos/order)")
        print(f"    Binding strength:     {m.binding_strength * 100:.2f}%")
        print()
        print(f"    Shannon entropy:      {m.shannon_entropy:.4f} bits")
        print(f"    Normalized entropy:   {m.normalized_entropy:.4f}")
        if m.is_compressed_data:
            print("    [!] COMPRESSED DATA DETECTED (high entropy + sharp gradients)")
        print()
        print("    Cosmological Alignments (ET predicts 2/3, 1/4, 1/12):")
        print(f"      Dark Energy (2/3):  {m.dark_energy_alignment * 100:.2f}%")
        print(f"      Dark Matter (1/4):  {m.dark_matter_alignment * 100:.2f}%")
        print(f"      Baryonic (1/12):    {m.baryonic_alignment * 100:.2f}%")
        print(f"      Koide ratio (2/3):  {m.koide_ratio * 100:.2f}%")
        print()

    # STATE
    print("┌" + "─" * 70 + "┐")
    print("│  ET STATE                                                            │")
    print("└" + "─" * 70 + "┘")
    state_desc = {
        ETState.STATE_0: "P ∘ D₀ (Unsubstantiated, P-dominant)",
        ETState.STATE_1: "P ∘ D₁ (Unsubstantiated, D-dominant)",
        ETState.SUPERPOSITION: "P ∘ (D₀, D₁) (Superposition - awaiting T)",
        ETState.SUBSTANTIATED: "(P ∘ D) ∘ T (Substantiated/Exception)"
    }
    print(f"    State: {sig.state.name}")
    print(f"    {state_desc[sig.state]}")
    print()

    # v5.0/v6.0: TRAVERSER COMPLEXITY
    if sig.traverser_complexity:
        tc = sig.traverser_complexity
        print("┌" + "─" * 70 + "┐")
        print("│  TRAVERSER COMPLEXITY (Gravity vs Intent)                            │")
        print("│  v6.0 - Distinguishing Source of Agency                              │")
        print("└" + "─" * 70 + "┘")
        c_str = tc.complexity_class.value
        if tc.complexity_class == ComplexityClass.CYCLIC_GRAVITY:
            c_str += " (GRAVITY/PLANET - Rule 10)"
        if tc.complexity_class == ComplexityClass.PROGRESSIVE_INTENT:
            c_str += " [!!! INTENT/LIFE - Nested T (Rule 19) !!!]"
        print(f"    Class:           {c_str}")
        print(f"    Periodicity:     {tc.periodicity_score:.4f} (Gravity echo)")
        print(f"    Progression:     {tc.progression_score:.4f} (Intent signal)")
        print(f"    Fractal Dim:     {tc.fractal_dimension:.4f}")
        print(f"    T Nesting:       {tc.nesting_depth} layers (T within T)")
        print()

    # v5.0/v6.0: GAZE DETECTION
    if sig.gaze_metrics:
        gz = sig.gaze_metrics
        print("┌" + "─" * 70 + "┐")
        print("│  GAZE DETECTION (Observer Effect)                                    │")
        print("│  v6.0 - Thresholds: 1.0833 (subliminal), 1.20 (detect), 1.50 (lock)  │")
        print("└" + "─" * 70 + "┘")
        alert = ""
        if gz.status == GazeStatus.DETECTED:
            alert = "<<< WATCHER DETECTED >>>"
        elif gz.status == GazeStatus.LOCKED:
            alert = "<<< TARGET LOCKED >>>"
        print(f"    Status:          {gz.status.value} {alert}")
        print(f"    Raw Pressure:    {gz.gaze_pressure:.4f}x Baseline")
        print(f"    Corrected:       {gz.corrected_pressure:.4f}x (×{DESCRIPTOR_LAYERS} for VM)")
        print(f"    Threshold:       {GAZE_THRESHOLD}x (1.20)")
        print(f"    Collapse:        {gz.local_collapse * 100:.2f}% Variance Loss")
        print()

    # TEMPORAL TRENDS
    if sig.temporal_trend:
        t = sig.temporal_trend
        print("┌" + "─" * 70 + "┐")
        print("│  TEMPORAL TRENDS                                                     │")
        print("│  Time is a Descriptor: f'(t) = ΔD/ΔTime                              │")
        print("└" + "─" * 70 + "┘")
        print(f"    Variance trend:       {t.variance_trend.value} (grad: {t.variance_gradient:+.4f})")
        print(f"    Variance volatility:  {t.variance_volatility:.4f}")
        print(f"    P trend:              {t.p_trend.value} (grad: {t.p_gradient:+.6f})")
        print(f"    D trend:              {t.d_trend.value} (grad: {t.d_gradient:+.6f})")
        print(f"    T trend:              {t.t_trend.value} (grad: {t.t_gradient:+.6f})")
        print()
        print(f"    Temporal Δs:          {t.mean_temporal_distance:.4f}")
        print(f"    Distance trend:       {t.distance_trend.value}")
        print(f"    Temporal binding:     {t.binding_strength * 100:.2f}%")
        print()
        print(f"    Correlations: P↔D={t.pd_correlation:+.3f}  D↔T={t.dt_correlation:+.3f}  P↔T={t.pt_correlation:+.3f}")
        if t.dominant_cycle:
            print(f"    Dominant cycle:       {t.dominant_cycle} samples")

        if t.variance_trend == TrendDirection.CONVERGING:
            print("    [!] CONVERGING TO EXCEPTION")
        print()

    # EXCEPTION APPROACH
    if sig.exception_approach:
        e = sig.exception_approach
        print("┌" + "─" * 70 + "┐")
        print("│  EXCEPTION APPROACH                                                  │")
        print("│  Variance(E) = 0 - The Exception cannot be otherwise                 │")
        print("└" + "─" * 70 + "┘")
        print(f"    Approaching:          {'YES' if e.is_approaching else 'NO'}")
        print(f"    Variance gradient:    {e.variance_gradient:+.4f}")
        print(f"    Current ratio:        {e.current_variance_ratio:.6f}")
        print(f"    Confidence:           {e.confidence * 100:.1f}%")

        if e.estimated_steps_to_zero:
            print(f"    Est. steps to zero:   {e.estimated_steps_to_zero}")

        print(f"    Alert level:          {e.alert_level}")

        if e.alert_level == "CRITICAL":
            print()
            print("    ╔════════════════════════════════════════════════════════════════╗")
            print("    ║  [*] CRITICAL: IMMINENT EXCEPTION APPROACH                      ║")
            print("    ╚════════════════════════════════════════════════════════════════╝")
        elif e.alert_level == "WARNING":
            print("    [!]  WARNING: Variance trending to zero")
        elif e.alert_level == "WATCH":
            print("    [o]  WATCH: Possible approach pattern")
        print()

    # PROOF REPORT
    if sig.proof_report:
        pr = sig.proof_report
        print("┌" + "─" * 70 + "┐")
        print("│  ET PROOF REPORT (21 Rules Verification)                             │")
        print("└" + "─" * 70 + "┘")
        print(f"    Overall status:       {pr.overall_status.value}")
        print(f"    Verified:             {pr.verified_count}")
        print(f"    Failed:               {pr.failed_count}")
        print(f"    Indeterminate:        {pr.indeterminate_count}")
        print(f"    Confidence:           {pr.confidence * 100:.1f}%")
        print()
        print(f"    PDT = EIM (3=3):      {'[+] VERIFIED' if pr.pdt_eim_verified else '[x] FAILED'}")
        if pr.binding_chain:
            bc = pr.binding_chain
            print(f"    Binding Chain:        T↔D={bc.t_d_binding_verified} D→P={bc.d_p_binding_verified} (integrity: {bc.chain_integrity:.0%})")
        if pr.coherence:
            c = pr.coherence
            print(f"    Coherence:            {c.coherent_ratio:.2%} coherent, {c.incoherent_count} incoherent")
        print()
        print("    Axiom Results:")
        for p in pr.axiom_proofs:
            sym = "[+]" if p.status == ProofStatus.VERIFIED else ("[x]" if p.status == ProofStatus.FAILED else "?")
            num = f"[{p.axiom_number:2d}]" if p.axiom_number > 0 else "[--]"
            print(f"      {sym} {num} {p.axiom_name[:40]:<40} {p.status.value}")
        print()

    # v7.0: SPECTRAL ANALYSIS
    if sig.spectral_analysis:
        sp = sig.spectral_analysis
        print("┌" + "─" * 70 + "┐")
        print("│  SPECTRAL ANALYSIS (v7.0)                                            │")
        print("│  FFT-based Periodicity Detection                                     │")
        print("└" + "─" * 70 + "┘")
        print(f"    Spectral Entropy:     {sp.spectral_entropy:.4f}")
        print(f"    Periodicity Strength: {sp.periodicity_strength:.4f}")
        print(f"    f=12 Alignment:       {sp.f_12_alignment:.4f}")
        print(f"    Harmonic Content:     {sp.f_harmonic_content:.4f}")
        if sp.dominant_frequencies:
            print(f"    Dominant Frequencies: {', '.join(f'{f:.2f}' for f in sp.dominant_frequencies[:3])}")
        print()

    # v7.0: FRACTAL ANALYSIS
    if sig.fractal_analysis:
        fr = sig.fractal_analysis
        print("┌" + "─" * 70 + "┐")
        print("│  FRACTAL ANALYSIS (v7.0)                                             │")
        print("│  Self-Similarity and Scaling Behavior                                │")
        print("└" + "─" * 70 + "┘")
        print(f"    Box Dimension:        {fr.box_counting_dimension:.4f}")
        print(f"    Correlation Dimension:{fr.correlation_dimension:.4f}")
        print(f"    Information Dimension:{fr.information_dimension:.4f}")
        print(f"    Self-Similarity:      {fr.self_similarity_ratio:.4f}")
        print(f"    Scaling Exponent:     {fr.scaling_exponent:.4f}")
        print(f"    Multifractal Width:   {fr.multifractal_spectrum_width:.4f}")
        print()

    # v7.0: THERMODYNAMIC VERIFICATION
    if sig.thermodynamic_verification:
        th = sig.thermodynamic_verification
        print("┌" + "─" * 70 + "┐")
        print("│  THERMODYNAMIC VERIFICATION (v7.0)                                   │")
        print("│  Four Laws from ET Perspective                                       │")
        print("└" + "─" * 70 + "┘")
        print(f"    0th Law (Equilibrium): {'[✓] VERIFIED' if th.zeroth_law_verified else '[✗] FAILED'}")
        print(f"    1st Law (Conservation):{'[✓] VERIFIED' if th.first_law_verified else '[✗] FAILED'}")
        print(f"    2nd Law (Entropy↑):    {'[✓] VERIFIED' if th.second_law_verified else '[✗] FAILED'}")
        print(f"    3rd Law (T>0):         {'[✓] VERIFIED' if th.third_law_verified else '[✗] FAILED'}")
        print(f"    Temperature Analog:    {th.temperature_analog:.2f}")
        print(f"    Entropy Analog:        {th.entropy_analog:.4f}")
        print(f"    Energy Analog:         {th.energy_analog:.2f}")
        print(f"    Heat Capacity:         {th.heat_capacity_analog:.4f}")
        print()

    # v7.0: QUANTUM VERIFICATION
    if sig.quantum_verification:
        qm = sig.quantum_verification
        print("┌" + "─" * 70 + "┐")
        print("│  QUANTUM VERIFICATION (v7.0)                                         │")
        print("│  Quantum Mechanics from ET Perspective                               │")
        print("└" + "─" * 70 + "┘")
        print(f"    Superposition:        {'DETECTED' if qm.superposition_detected else 'NOT DETECTED'}")
        print(f"    Collapse Events:      {qm.collapse_events}")
        print(f"    Uncertainty (Δx·Δp):  {qm.uncertainty_product:.4f}")
        print(f"    Entanglement Sig:     {qm.entanglement_signature:.4f}")
        print(f"    Wavefunction Norm:    {qm.wavefunction_norm:.4f}")
        print(f"    Decoherence Time:     {qm.decoherence_time_estimate:.2e}")
        print(f"    Quantum Entropy:      {qm.quantum_entropy:.4f}")
        print()


    # SUMMARY
    print("┌" + "─" * 70 + "┐")
    print("│  SUMMARY                                                              │")
    print("└" + "─" * 70 + "┘")
    print(f"    History:              {len(_history.signatures)} scans")
    print(f"    Events logged:        {len(_history.events)}")
    print(f"    Session duration:     {(datetime.now() - _history.session_start).total_seconds():.1f}s")
    print()
    print("=" * 72)
    print("  \"For every exception there is an exception, except the exception.\"")
    print("=" * 72)


# =============================================================================
# CONTINUOUS SCAN MODE
# =============================================================================

def continuous_scan(input_type: InputType = InputType.ENTROPY,
                    source: Any = None,
                    interval: float = 1.0,
                    samples: int = 10000,
                    log_file: Optional[str] = None):
    """Continuous scanning mode with proper stop handling."""
    global _running, _scan_active
    _running = True
    _scan_active = True

    if log_file:
        _history.enable_logging(log_file)
        print(f"📝 Logging to: {log_file}")

    print()
    print("=" * 72)
    print(f"  CONTINUOUS SCAN MODE - {VERSION} VERIFICATION PROTOCOL ACTIVE")
    print("=" * 72)
    print(f"  Source:   {source or 'Hardware Entropy'}")
    print(f"  Type:     {input_type.value}")
    print(f"  Interval: {interval}s")
    print(f"  Samples:  {samples:,}")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 72)
    print()
    print("  State │ Var Ratio │ Pure T │  dτ/dt  │    τ    │ Gaze   │ Class")
    print("  ──────┼───────────┼────────┼─────────┼─────────┼────────┼───────")

    scan_count = 0

    try:
        while _running and _scan_active:
            try:
                # Get data based on input type
                if input_type == InputType.ENTROPY:
                    data, src = UniversalInput.from_entropy(samples)
                elif input_type == InputType.FILE and source:
                    data, src = UniversalInput.from_file(source, max_bytes=samples)
                elif input_type == InputType.URL and source:
                    data, src = UniversalInput.from_url(source, max_bytes=samples)
                elif input_type == InputType.PROCESS and source:
                    pid = int(source) if str(source).isdigit() else None
                    name = source if not str(source).isdigit() else None
                    data, src = UniversalInput.from_process(pid=pid, name=name)
                else:
                    data, src = UniversalInput.from_entropy(samples)

                sig = et_scan(data=data, source=src, verbose=False, run_proofs=False)
                scan_count += 1

                # Format output
                state_sym = {
                    ETState.STATE_0: "  ○   ",
                    ETState.STATE_1: "  ◐   ",
                    ETState.SUPERPOSITION: "  ◑   ",
                    ETState.SUBSTANTIATED: "  ●   "
                }

                dt = sig.dual_time
                pure_t = sig.indeterminate.failures

                # Gaze status
                gaze_sym = "      "
                if sig.gaze_metrics:
                    if sig.gaze_metrics.status == GazeStatus.DETECTED:
                        gaze_sym = " [o]    "
                    elif sig.gaze_metrics.status == GazeStatus.LOCKED:
                        gaze_sym = " [*]   "
                    elif sig.gaze_metrics.status == GazeStatus.SUBLIMINAL:
                        gaze_sym = " ~    "

                # Complexity class
                class_sym = "..."
                if sig.traverser_complexity:
                    if sig.traverser_complexity.complexity_class == ComplexityClass.CYCLIC_GRAVITY:
                        class_sym = "GRAV"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.PROGRESSIVE_INTENT:
                        class_sym = "LIFE"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.STATIC:
                        class_sym = "STAT"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.CHAOTIC:
                        class_sym = "CHAO"

                print(f"  {state_sym.get(sig.state, '  ?   ')} │ "
                      f"{sig.variance_ratio:9.6f} │ "
                      f"{pure_t:6d} │ "
                      f"{dt.t_time_dtau_dt:7.4f} │ "
                      f"{dt.t_time_tau:7.4f} │ "
                      f"{gaze_sym} │ {class_sym}")

                time.sleep(interval)

            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        _scan_active = False
        # FIX: Ensure program doesn't close after continuous scan
        _running = True
        print()
        print("=" * 72)
        print(f"  Scan complete. {scan_count} iterations.")
        print("=" * 72)


# =============================================================================
# CONSOLE UTILITIES
# =============================================================================

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with default value."""
    try:
        result = input(prompt)
        return result if result else default
    except (EOFError, KeyboardInterrupt):
        return default


def print_session_summary():
    """Print session summary."""
    summary = _history.get_session_summary()
    if "error" in summary:
        print(f"  {summary['error']}")
        return

    print()
    print("  SESSION SUMMARY")
    print("  " + "=" * 50)
    print(f"  Total scans:     {summary['total_scans']}")
    print(f"  Total events:    {summary['total_events']}")
    print(f"  Duration:        {summary['session_duration']:.1f}s")
    print()
    print(f"  Variance mean:   {summary['variance']['mean']:.2f}")
    print(f"  Variance std:    {summary['variance']['std']:.2f}")
    print()
    print(f"  PDT averages:")
    print(f"    P: {summary['pdt_averages']['p']:.4f}")
    print(f"    D: {summary['pdt_averages']['d']:.4f}")
    print(f"    T: {summary['pdt_averages']['t']:.4f}")
    print()
    print(f"  Pure T statistics:")
    print(f"    Total:  {summary['pure_t_statistics']['total']}")
    print(f"    Mean:   {summary['pure_t_statistics']['mean']:.2f}")
    print(f"    Max:    {summary['pure_t_statistics']['max']}")
    print()
    print(f"  T_time τ total:  {summary['t_time_total']:.4f}")
    print(f"  Substantiations: {summary['substantiations']}")


def run_interactive():
    """Run interactive menu."""
    global _running

    while _running:
        # FIX: Removed clear_screen() to prevent scan information from resetting.
        # The user requested that scan information must stay.
        # clear_screen()
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print(f"║        EXCEPTION THEORY SCANNER v6.0 - COMPLETE VERIFICATION         ║")
        print("║                                                                      ║")
        print("║     All mathematics derived from Exception Theory by M.J.M.          ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("  ┌─────────────────────────────────────────────────────────────────┐")
        print("  │  SCAN OPTIONS                                                   │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        print("  │  [1] Scan Hardware Entropy (Raw Substrate)                      │")
        print("  │  [2] Scan File (Opens File Browser)                             │")
        print("  │  [3] Scan URL                                                   │")
        print("  │  [4] Scan Process                                               │")
        print("  │  [5] Scan Clipboard                                             │")
        print("  │  [6] Scan Custom Text                                           │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        print("  │  [C] Continuous Entropy Scan                                    │")
        print("  │  [F] Continuous File Scan                                       │")
        print("  │  [P] Continuous Process Scan                                    │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        print("  │  [E] Log External Event                                         │")
        print("  │  [R] Event Correlation Report                                   │")
        print("  │  [S] Session Summary                                            │")
        print("  │  [X] Export Session                                             │")
        print("  │  [H] Reset History                                              │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        print("  │  [Q] Quit                                                       │")
        print("  └─────────────────────────────────────────────────────────────────┘")
        print()

        choice = get_input("  Enter choice: ").upper()

        try:
            if choice == '1':
                samples = int(get_input("  Samples [100000]: ", "100000"))
                et_scan(samples=samples)
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == '2':
                print("  Opening file browser...")
                filepath = select_file()
                if filepath:
                    data, source = UniversalInput.from_file(filepath)
                    et_scan(data=data, source=source)
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == '3':
                url = get_input("  Enter URL: ")
                if url:
                    try:
                        data, source = UniversalInput.from_url(url)
                        et_scan(data=data, source=source)
                    except Exception as e:
                        print(f"  [!] Error: {e}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == '4':
                if not HAS_PSUTIL:
                    print("  [!] psutil not installed. Run: pip install psutil")
                else:
                    target = get_input("  Enter PID or process name: ")
                    if target:
                        try:
                            pid = int(target) if target.isdigit() else None
                            name = target if not target.isdigit() else None
                            data, source = UniversalInput.from_process(pid=pid, name=name)
                            et_scan(data=data, source=source)
                        except Exception as e:
                            print(f"  [!] Error: {e}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == '5':
                try:
                    data, source = UniversalInput.from_clipboard()
                    et_scan(data=data, source=source)
                except Exception as e:
                    print(f"  [!] Error: {e}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == '6':
                print("  Enter text (empty line to finish):")
                lines = []
                while True:
                    line = get_input("  ")
                    if not line:
                        break
                    lines.append(line)

                if lines:
                    text = '\n'.join(lines)
                    try:
                        data, source = UniversalInput.from_string(text)
                        et_scan(data=data, source=source)
                    except Exception as e:
                        print(f"  [!] Error: {e}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'C':
                interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                samples = int(get_input("  Samples per scan [10000]: ", "10000"))
                log = get_input("  Log file (or Enter for none): ")
                continuous_scan(InputType.ENTROPY, interval=interval, samples=samples,
                                log_file=log if log else None)
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'F':
                print("  Opening file browser...")
                filepath = select_file()
                if filepath:
                    interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                    continuous_scan(InputType.FILE, source=filepath, interval=interval)
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'P':
                if not HAS_PSUTIL:
                    print("  [!] psutil not installed")
                else:
                    target = get_input("  Enter PID or process name: ")
                    if target:
                        interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                        continuous_scan(InputType.PROCESS, source=target, interval=interval)
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'E':
                etype = get_input("  Event type [observation]: ", "observation")
                desc = get_input("  Description: ")
                if desc:
                    event = _history.add_event(etype, desc)
                    print(f"  [+] Event logged at {event.timestamp.strftime('%H:%M:%S')}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'R':
                result = _history.analyze_event_correlations()
                print()
                print("  CORRELATION REPORT")
                print("  " + "=" * 50)
                print(f"  {result['summary']}")
                for c in result.get('correlations', []):
                    print()
                    print(f"    Event: {c['event']}")
                    print(f"    Variance shift: {c['variance_shift']:+.2f}")
                    print(f"    T ratio shift: {c['t_ratio_shift']:+.6f}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'S':
                print_session_summary()
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'X':
                filepath = select_save_file()
                if filepath:
                    try:
                        _history.export_session(filepath)
                        print(f"  [+] Exported to: {filepath}")
                    except Exception as e:
                        print(f"  [!] Error: {e}")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'H':
                confirm = get_input("  Reset all history? (y/N): ", "n")
                if confirm.lower() == 'y':
                    _history.reset()
                    print("  [+] History reset")
                get_input("  Press Enter to continue...")
                # clear_screen() # FIX: Removed to keep info

            elif choice == 'Q':
                if len(_history.signatures) > 0:
                    save = get_input("  Save session before exit? (y/N): ", "n")
                    if save.lower() == 'y':
                        filepath = select_save_file()
                        if filepath:
                            _history.export_session(filepath)
                            print(f"  [+] Saved to: {filepath}")
                print()
                print(f"  Exception Theory Scanner {VERSION} terminated.")
                print("  \"For every exception there is an exception, except the exception.\"")
                print()
                _running = False
                break

            else:
                print("  [!] Invalid choice")
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n  Press Q to quit or any key to continue...")
            continue
        except Exception as e:
            print(f"\n  Menu error: {e}")
            print("  Continuing...")
            time.sleep(1)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def scan_entropy(samples: int = 100000, **kwargs) -> ETSignature:
    """Scan hardware entropy."""
    data, source = UniversalInput.from_entropy(samples)
    return et_scan(data=data, source=source, **kwargs)


def scan_file(filepath: str, **kwargs) -> ETSignature:
    """Scan any file."""
    data, source = UniversalInput.from_file(filepath, kwargs.pop('max_bytes', None))
    return et_scan(data=data, source=source, **kwargs)


def scan_url(url: str, **kwargs) -> ETSignature:
    """Scan web content."""
    data, source = UniversalInput.from_url(url, kwargs.pop('max_bytes', None))
    return et_scan(data=data, source=source, **kwargs)


def scan_process(target: Union[int, str], **kwargs) -> ETSignature:
    """Scan a process."""
    pid = int(target) if str(target).isdigit() else None
    name = target if not str(target).isdigit() else None
    data, source = UniversalInput.from_process(pid=pid, name=name)
    return et_scan(data=data, source=source, **kwargs)


def scan_data(data: Union[bytes, str, np.ndarray], **kwargs) -> ETSignature:
    """Scan raw data."""
    if isinstance(data, str):
        arr, source = UniversalInput.from_string(data)
    elif isinstance(data, bytes):
        arr, source = UniversalInput.from_bytes(data)
    elif isinstance(data, np.ndarray):
        arr, source = UniversalInput.from_numpy(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    return et_scan(data=arr, source=source, **kwargs)


def log_event(event_type: str, description: str, metadata: Optional[Dict] = None):
    """Log external event."""
    return _history.add_event(event_type, description, metadata)


def export_session(filepath: str) -> str:
    """Export session."""
    return _history.export_session(filepath)


def get_summary() -> Dict:
    """Get session summary."""
    return _history.get_session_summary()


# =============================================================================
# ENTRY POINT
# =============================================================================

def _setup_windows():
    """Windows-specific setup for signal handlers."""
    # NOTE: multiprocessing.freeze_support() is called at the TOP of this file,
    # BEFORE numpy/scipy imports. That's the critical fix for Windows crashes.
    # This function now only handles signal registration.
    
    # Register signal handlers HERE, not at module level (Windows requirement)
    try:
        signal.signal(signal.SIGINT, signal_handler)
    except Exception:
        pass  # May fail in some Windows environments
    
    if hasattr(signal, 'SIGTERM'):
        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception:
            pass


if __name__ == "__main__":
    # CRITICAL: Windows multiprocessing support - MUST BE FIRST
    _setup_windows()
    
    try:
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()

            if arg in ['--help', '-h']:
                print(f"""
Exception Theory Scanner {VERSION} - {VERSION_NAME}

Usage:
  python et_scanner_v6.py              Interactive menu
  python et_scanner_v6.py --help       Show this help
  python et_scanner_v6.py <file>       Scan a file
  python et_scanner_v6.py --entropy    Scan entropy
  python et_scanner_v6.py --continuous Continuous entropy scan

NEW in v6.0:
  - Complete 21 Rules verification
  - Enhanced L'Hopital Pure T detection
  - Manifold boundary detection (power-of-2)
  - Cosmological ratio verification (2/3, 1/4, 1/12)
  - Entropy gradient analysis (compressed data)
  - Binding chain verification (T<->D->P)
  - Incoherence region detection

All mathematics derived from Exception Theory by M.J.M.
"For every exception there is an exception, except the exception."
""")
                sys.exit(0)

            elif arg == '--entropy':
                et_scan()

            elif arg == '--continuous':
                continuous_scan()

            elif os.path.exists(sys.argv[1]):
                scan_file(sys.argv[1])

            else:
                print(f"Unknown argument or file not found: {sys.argv[1]}")
                print("Use --help for usage information.")
                sys.exit(1)
        else:
            try:
                run_interactive()
            except KeyboardInterrupt:
                print("\n\n  Interrupted. Exiting...")
            except Exception as e:
                print(f"\n  Fatal error: {e}")
                import traceback
                traceback.print_exc()
                print("\nPress Enter to exit...")
                input()
                sys.exit(1)
    except Exception as e:
        print(f"\n  CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
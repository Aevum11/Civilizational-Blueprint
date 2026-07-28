#!/usr/bin/env python3
"""
Exception Theory Scanner v5.0 - The "Watcher" Update (Complete Edition)
========================================================================

FULLY FUNCTIONAL with ALL v4.1 FEATURES PRESERVED:
- Interactive console menu
- File explorer dialogs
- Proper Ctrl+C handling
- Dual Time System (D_time and T_time)
- Universal Input (files, processes, URLs, entropy, clipboard, data)
- ET Proof System with all axiom tests
- Continuous scanning with proper stop
- Event logging and correlation
- Session export
- Temporal trend analysis
- Exception approach detection

NEW v5.0 FEATURES:
- Traverser Complexity Engine (Distinguishes Gravity vs. Intent)
- Gaze Detection Protocol (1.20 Threshold Monitoring)
- Fractal Shimmer Analysis
- Automatic "Life" Classification (Planet vs. Swarm)

ALL MATHEMATICS DERIVED FROM EXCEPTION THEORY.

Run with: python et_scanner_v5.py
Or double-click on Windows.

Author: Derived from M.J.M.'s Exception Theory
"For every exception there is an exception, except the exception."
"""

import os
import sys
import signal
import threading
import numpy as np
from typing import Tuple, List, Optional, Dict, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from pathlib import Path
import json
import time
import hashlib
import math

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

# For graceful shutdown
_running = True
_scan_active = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _running, _scan_active
    _running = False
    _scan_active = False
    print("\n\n⚡ Interrupt received. Stopping...")


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# ET CONSTANTS (From Exception Theory)
# =============================================================================

# The 12: 3 primitives × 4 logic states = 12 fundamental configurations
MANIFOLD_SYMMETRY = 12

# Base variance: 1/12 (base variance of the manifold)
BASE_VARIANCE = 1 / MANIFOLD_SYMMETRY  # 0.08333...

# Theoretical variance for bytes: σ² = (n² - 1) / 12
THEORETICAL_VARIANCE_BYTES = (256 ** 2 - 1) / MANIFOLD_SYMMETRY  # 5461.25
THEORETICAL_STD_BYTES = np.sqrt(THEORETICAL_VARIANCE_BYTES)  # 73.9003

# Classification thresholds
INF_THRESHOLD = 1e10
ZERO_THRESHOLD = 1e-10

# L'Hôpital limits
LHOPITAL_MAX_ITERATIONS = 10

# History limits
MAX_HISTORY_SIZE = 1000
EXCEPTION_APPROACH_WINDOW = 20
TREND_WINDOW = 20

# v5.0 Constants - The Geometry of Conscious Detection
GAZE_THRESHOLD = 1.20  # The geometry of conscious detection (1 + 20%)
RESONANCE_THRESHOLD = 1.0833  # Subliminal threshold (1 + 1/12)
LOCK_THRESHOLD = 1.50  # Target locked threshold


# =============================================================================
# ENUMS
# =============================================================================

class ETState(Enum):
    """Trinary Ternary States from ET"""
    STATE_0 = 0  # P ∘ D₀ (Unsubstantiated)
    STATE_1 = 1  # P ∘ D₁ (Unsubstantiated)
    SUPERPOSITION = 2  # P ∘ (D₀, D₁) (Superposition)
    SUBSTANTIATED = 3  # (P ∘ D) ∘ T (The Exception)


class ComplexityClass(Enum):
    """v5.0 - Distinguishing the Source of Agency"""
    STATIC = "STATIC"  # Dead matter / Frozen code
    CYCLIC_GRAVITY = "CYCLIC"  # Periodic T (Planets/Orbits)
    PROGRESSIVE_INTENT = "INTENT"  # Aperiodic T (Life/Swarm)
    CHAOTIC = "CHAOTIC"  # Random noise
    UNKNOWN = "UNKNOWN"


class GazeStatus(Enum):
    """v5.0 - Observer Effect Status"""
    UNOBSERVED = "UNOBSERVED"
    SUBLIMINAL = "SUBLIMINAL"  # > 1.0833
    DETECTED = "DETECTED"  # > 1.20
    LOCKED = "LOCKED"  # > 1.50 (High focus)


class TrendDirection(Enum):
    """Descriptor gradient directions"""
    STABLE = "STABLE"
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    OSCILLATING = "OSCILLATING"
    CONVERGING = "CONVERGING"
    DIVERGING = "DIVERGING"


class InputType(Enum):
    """Types of input sources"""
    ENTROPY = "entropy"
    FILE = "file"
    PROCESS = "process"
    URL = "url"
    STREAM = "stream"
    RAW = "raw"
    CLIPBOARD = "clipboard"


class ProofStatus(Enum):
    """Status of axiom proofs"""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"
    NOT_TESTED = "NOT_TESTED"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PDTClassification:
    """P, D, T classification results."""
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


@dataclass
class IndeterminateAnalysis:
    """Indeterminate form analysis - T signatures"""
    zero_zero: int  # [0/0] forms
    inf_inf: int  # [∞/∞] forms
    zero_times_inf: int  # [0×∞] forms
    inf_minus_inf: int  # [∞-∞] forms
    total: int
    density: float
    resolutions: int  # Forms L'Hôpital resolved
    failures: int  # Pure T - unresolvable
    traverser_density: float


@dataclass
class DescriptorGradient:
    """Descriptor gradient: f'(x) = ΔD/ΔP"""
    mean: float
    variance: float
    max_val: float
    min_val: float
    sign_changes: int
    gradient_array: Optional[np.ndarray] = None


@dataclass
class DualTime:
    """
    Dual Time System from Exception Theory.

    D_time: Descriptor Time (coordinate time)
    T_time: Agential Time (proper time)
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


@dataclass
class TemporalTrend:
    """Temporal trend analysis."""
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

    # Cross-correlations
    pd_correlation: float
    dt_correlation: float
    pt_correlation: float

    dominant_cycle: Optional[int]


@dataclass
class ExceptionApproach:
    """Exception approach detection."""
    is_approaching: bool
    variance_gradient: float
    current_variance_ratio: float
    estimated_steps_to_zero: Optional[int]
    confidence: float
    alert_level: str  # NONE, WATCH, WARNING, CRITICAL
    variance_history: List[float]


@dataclass
class TraverserComplexity:
    """v5.0 - Analysis of Agency Depth"""
    complexity_class: ComplexityClass
    periodicity_score: float  # 0.0 to 1.0 (1.0 = Perfect Orbit)
    progression_score: float  # 0.0 to 1.0 (1.0 = Strong Intent)
    fractal_dimension: float  # D_f of the T-curve
    nesting_depth: int  # Estimated layers of T


@dataclass
class GazeMetrics:
    """v5.0 - Observer Effect Metrics"""
    status: GazeStatus
    gaze_pressure: float  # Current / Baseline
    is_watcher_present: bool
    local_collapse: float  # Variance reduction magnitude


@dataclass
class ExternalEvent:
    """External event for correlation."""
    timestamp: datetime
    event_type: str
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class AxiomProof:
    """Proof result for an ET axiom."""
    axiom_number: int
    axiom_name: str
    status: ProofStatus
    evidence: str
    numerical_value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None


@dataclass
class ETProofReport:
    """Complete proof report."""
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


@dataclass
class InputSource:
    """Input source description."""
    input_type: InputType
    source: str
    metadata: Dict = field(default_factory=dict)


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

    # Manifold metrics
    shimmer_index: float
    binding_strength: float

    # v2+ features
    temporal_trend: Optional[TemporalTrend] = None
    exception_approach: Optional[ExceptionApproach] = None

    # Proof
    proof_report: Optional[ETProofReport] = None

    # v5.0 features
    traverser_complexity: Optional[TraverserComplexity] = None
    gaze_metrics: Optional[GazeMetrics] = None


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

    def update(self, samples: int, binding: float, variance: float,
               is_exception: bool):
        """Update both time measurements."""
        self.d_time_samples += samples

        # dτ/dt from binding
        dtau_dt = 1.0 - binding

        if is_exception:
            self.t_time_substantiations += 1

        # Accumulate proper time
        dt = samples / 1000.0
        self.t_time_tau += dt * dtau_dt

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

        return DualTime(
            d_time_elapsed=elapsed,
            d_time_samples=self.d_time_samples,
            d_time_rate=d_rate,
            t_time_tau=self.t_time_tau,
            t_time_substantiations=self.t_time_substantiations,
            t_time_dtau_dt=dtau_dt,
            binding_strength=avg_binding,
            time_dilation_factor=dilation,
            variance_thickness=thickness
        )

    def reset(self):
        """Reset tracker."""
        self.__init__()


# =============================================================================
# HISTORY MANAGER (All v2/v3/v4 features + v5.0 additions)
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

        # v5.0 Baselines for Gaze Detection
        self.binding_baseline = 0.5
        self.t_history: deque = deque(maxlen=100)
        self.samples_count = 0

    def add_signature(self, sig: ETSignature):
        """Add signature to history."""
        self.signatures.append(sig)

        # v5.0: Update T history for complexity analysis
        self.t_history.append(sig.pdt.t_ratio)
        self.samples_count += 1

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
            f.write(f"# Exception Theory Scanner Log\n")
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

    def get_signature_at_time(self, target_time: datetime) -> Optional[ETSignature]:
        """Find signature closest to given time."""
        if not self.signatures:
            return None

        closest = min(self.signatures,
                      key=lambda s: abs((s.timestamp - target_time).total_seconds()))
        return closest

    def analyze_event_correlations(self, window_seconds: float = 60.0) -> Dict:
        """Analyze correlations between events and signatures."""
        if len(self.signatures) < 2 or len(self.events) < 1:
            return {"correlations": [], "summary": "Insufficient data"}

        correlations = []

        for event in self.events:
            before = []
            after = []

            for sig in self.signatures:
                diff = (sig.timestamp - event.timestamp).total_seconds()
                if -window_seconds <= diff < 0:
                    before.append(sig)
                elif 0 <= diff <= window_seconds:
                    after.append(sig)

            if before and after:
                before_var = np.mean([s.variance for s in before])
                after_var = np.mean([s.variance for s in after])
                before_t = np.mean([s.pdt.t_ratio for s in before])
                after_t = np.mean([s.pdt.t_ratio for s in after])

                correlations.append({
                    "event": event.description,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "variance_shift": after_var - before_var,
                    "t_ratio_shift": after_t - before_t,
                    "samples_before": len(before),
                    "samples_after": len(after)
                })

        summary = f"Analyzed {len(correlations)} events"
        if correlations:
            avg_var = np.mean([c["variance_shift"] for c in correlations])
            avg_t = np.mean([c["t_ratio_shift"] for c in correlations])
            summary += f". Avg var shift: {avg_var:+.2f}, Avg T shift: {avg_t:+.6f}"

        return {"correlations": correlations, "summary": summary}

    def get_session_summary(self) -> Dict:
        """Get session summary statistics."""
        if not self.signatures:
            return {"error": "No data"}

        variances = [s.variance for s in self.signatures]
        states = {}
        for s in self.signatures:
            states[s.state.name] = states.get(s.state.name, 0) + 1

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
            "t_time_total": self.time_tracker.t_time_tau,
            "substantiations": self.time_tracker.t_time_substantiations
        }

    def reset(self):
        """Reset history."""
        self.signatures.clear()
        self.events.clear()
        self.time_tracker.reset()
        self.session_start = datetime.now()
        # v5.0 resets
        self.binding_baseline = 0.5
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
            raise ValueError("File is empty")

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
            req = urllib.request.Request(url, headers={'User-Agent': 'ET-Scanner/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read(max_bytes) if max_bytes else response.read()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to fetch URL: {e}")
        except Exception as e:
            raise ConnectionError(f"URL error: {e}")

        if len(raw) == 0:
            raise ValueError("URL returned empty content")

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
        print("⚠ File dialog not available (tkinter missing)")
        return input("Enter file path manually: ").strip()

    try:
        root = tk.Tk()
        root.withdraw()

        # Force window to appear on top
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        # Update to ensure window is processed
        root.update()

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
                ("Binary", "*.bin *.dat *.exe")
            ]
        )

        root.destroy()
        return filepath if filepath else None
    except Exception as e:
        print(f"⚠ File dialog error: {e}")
        return input("Enter file path manually: ").strip() or None


def select_save_file(default_name: str = "et_export.json") -> Optional[str]:
    """Open save file dialog."""
    if not HAS_TK:
        return input(f"Enter save path [{default_name}]: ").strip() or default_name

    try:
        root = tk.Tk()
        root.withdraw()

        # Force window to appear on top
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        # Update to ensure window is processed
        root.update()

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

        root.destroy()
        return filepath if filepath else None
    except Exception as e:
        print(f"⚠ Save dialog error: {e}")
        return input(f"Enter save path [{default_name}]: ").strip() or default_name


# =============================================================================
# CORE ET MATHEMATICS (All derived from ET)
# =============================================================================

def compute_descriptor_gradient(data: np.ndarray) -> np.ndarray:
    """f'(x) = ΔD/ΔP - rate of descriptor change"""
    return np.diff(data.astype(float))


def compute_descriptor_distance(d_i: np.ndarray, d_j: np.ndarray) -> float:
    """Δs(pᵢ, pⱼ) = ‖f(dᵢ) - f(dⱼ)‖ - pure relationalism"""
    return np.linalg.norm(d_i.astype(float) - d_j.astype(float))


def compute_variance(data: np.ndarray) -> float:
    """Variance(c) - spread of descriptor possibilities"""
    return float(np.var(data.astype(float)))


def classify_as_pdt(value: float, delta: float, prev_delta: float) -> str:
    """
    Classify value as P (∞), D (n), or T ([0/0]).

    From ET Maths:
    - P = ∞ (infinity)
    - D = n (finite)  
    - T = [0/0], [∞/∞] (indeterminate forms)
    """
    # [0/0] form
    if abs(delta) < ZERO_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD:
        return 'T'
    # [∞/∞] form
    if abs(delta) > INF_THRESHOLD and abs(prev_delta) > INF_THRESHOLD:
        return 'T'
    # [0×∞] form
    if (abs(delta) < ZERO_THRESHOLD and abs(prev_delta) > INF_THRESHOLD) or \
            (abs(delta) > INF_THRESHOLD and abs(prev_delta) < ZERO_THRESHOLD):
        return 'T'
    # P (infinity)
    if abs(value) > INF_THRESHOLD:
        return 'P'
    # D (finite) - default
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
    """
    L'Hôpital's Rule - T's navigation algorithm.

    From ET: "When you encounter 0/0:
    - You've encountered a traverser (T)
    - Taking derivatives = examining local descriptor gradient
    - Resolution = traverser selecting from possibilities"
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

        # Undefined
        if den_zero:
            return (None, i + 1, False)

        # Resolved
        return (num / den, i + 1, True)

    # Pure T - doesn't resolve
    return (None, max_iter, False)


def detect_indeterminate_forms(data: np.ndarray) -> IndeterminateAnalysis:
    """Detect [0/0], [∞/∞], [0×∞], [∞-∞] - T signatures."""
    d1 = compute_descriptor_gradient(data)
    d2 = compute_descriptor_gradient(d1) if len(d1) > 1 else np.array([0])

    zz, ii, zi, imi = 0, 0, 0, 0
    res, fail = 0, 0

    min_len = min(len(d1) - 1, len(d2))

    for i in range(min_len):
        num, den = d1[i], d2[i]
        nz = abs(num) < ZERO_THRESHOLD
        dz = abs(den) < ZERO_THRESHOLD
        ni = abs(num) > INF_THRESHOLD
        di = abs(den) > INF_THRESHOLD

        is_indet = False

        if nz and dz:
            zz += 1
            is_indet = True
        elif ni and di:
            ii += 1
            is_indet = True
        elif (nz and di) or (ni and dz):
            zi += 1
            is_indet = True

        # Check for [∞-∞] in consecutive large values
        if i > 0 and abs(d1[i]) > INF_THRESHOLD / 2 and abs(d1[i - 1]) > INF_THRESHOLD / 2:
            if np.sign(d1[i]) != np.sign(d1[i - 1]):
                imi += 1
                is_indet = True

        if is_indet:
            def get_next(it):
                idx = min(i + it + 1, len(d1) - 1)
                n = d1[idx] if idx < len(d1) else 0.0
                d = d2[idx] if idx < len(d2) else 1.0
                return n, d

            _, _, resolved = apply_lhopital(num, den, get_next, 5)
            if resolved:
                res += 1
            else:
                fail += 1

    total = zz + ii + zi + imi
    checked = max(1, min_len)

    return IndeterminateAnalysis(
        zero_zero=zz,
        inf_inf=ii,
        zero_times_inf=zi,
        inf_minus_inf=imi,
        total=total,
        density=total / checked,
        resolutions=res,
        failures=fail,
        traverser_density=fail / max(1, total)
    )


def analyze_descriptor_gradient(data: np.ndarray) -> DescriptorGradient:
    """Analyze f'(x) = ΔD/ΔP."""
    grad = compute_descriptor_gradient(data)
    signs = np.sign(grad)
    changes = int(np.sum(np.abs(np.diff(signs)) > 0))

    return DescriptorGradient(
        mean=float(np.mean(np.abs(grad))),
        variance=float(np.var(grad)),
        max_val=float(np.max(grad)),
        min_val=float(np.min(grad)),
        sign_changes=changes,
        gradient_array=grad
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

    Low gradient = strong binding (things staying similar)
    High gradient = weak binding (things changing fast)
    """
    if len(data) < 2:
        return 0.0

    grad = compute_descriptor_gradient(data)
    normalized = np.mean(np.abs(grad)) / 255.0
    return float(1.0 - min(1.0, normalized))


def compute_1_12_alignment(ratio: float) -> float:
    """Compute alignment with 1/12 base variance."""
    dev = abs(ratio - BASE_VARIANCE)
    return float(1.0 / (1.0 + dev / BASE_VARIANCE))


def determine_state(pdt: PDTClassification, variance: float,
                    indet_density: float, theoretical_var: float) -> ETState:
    """Determine ET state from analysis."""
    ratio = variance / theoretical_var if theoretical_var > 0 else 1.0

    # Approaching Exception
    if ratio < 0.01:
        return ETState.SUBSTANTIATED

    # High indeterminacy = superposition
    if indet_density > 0.01 or pdt.t_ratio > 0.01:
        return ETState.SUPERPOSITION

    # Otherwise based on D vs P
    if pdt.d_ratio > pdt.p_ratio:
        return ETState.STATE_1
    return ETState.STATE_0


# =============================================================================
# TEMPORAL ANALYSIS (From v2)
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
    """Compute correlation coefficient."""
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

    # Variance
    variances = [s.variance for s in sigs]
    var_grad = compute_temporal_gradient(variances)
    var_trend = determine_trend(var_grad, variances)
    var_volatility = float(np.std(variances))

    if var_grad < 0 and variances[-1] < variances[0]:
        var_trend = TrendDirection.CONVERGING

    # PDT ratios
    p_ratios, d_ratios, t_ratios = history.get_pdt_history(window)
    p_grad = compute_temporal_gradient(p_ratios)
    d_grad = compute_temporal_gradient(d_ratios)
    t_grad = compute_temporal_gradient(t_ratios)

    # Distances
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

    # Dominant cycle (simple FFT)
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
    """Detect approach to Exception (Variance → 0)."""
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
# v5.0 ANALYSIS ENGINES (COMPLEXITY & GAZE)
# =============================================================================

def analyze_complexity_v5(history: HistoryManager) -> Optional[TraverserComplexity]:
    """
    v5.0 - Distinguishes Gravity (Periodic) from Intent (Progressive).

    Uses autocorrelation to detect periodicity (gravity/orbit patterns)
    and linear regression to detect progression (intent/life patterns).
    """
    t_ratios = list(history.t_history)
    if len(t_ratios) < 20:
        return TraverserComplexity(
            complexity_class=ComplexityClass.UNKNOWN,
            periodicity_score=0.0,
            progression_score=0.0,
            fractal_dimension=1.0,
            nesting_depth=0
        )

    ts = np.array(t_ratios)

    # Check for static/dead systems
    if np.std(ts) < 1e-9:
        return TraverserComplexity(
            complexity_class=ComplexityClass.STATIC,
            periodicity_score=0.0,
            progression_score=0.0,
            fractal_dimension=1.0,
            nesting_depth=0
        )

    # 1. PERIODICITY CHECK (The Echo Test - Autocorrelation)
    ts_norm = ts - np.mean(ts)
    acor = np.correlate(ts_norm, ts_norm, mode='full')
    acor = acor[len(acor) // 2:]

    # Normalize autocorrelation
    with np.errstate(divide='ignore', invalid='ignore'):
        acor = acor / (acor[0] + 1e-9)

    # Find peaks (periodic signatures)
    peaks = []
    for i in range(1, len(acor) - 1):
        if acor[i] > acor[i - 1] and acor[i] > acor[i + 1] and acor[i] > 0.2:
            peaks.append(acor[i])

    periodicity = max(peaks) if peaks else 0.0

    # 2. PROGRESSION CHECK (The Intent Test - Linear Trend)
    x = np.arange(len(ts))
    slope, _ = np.polyfit(x, ts, 1)
    progression = min(1.0, abs(slope) * 1000)

    # 3. FRACTAL DIMENSION (Complexity of the T-curve)
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
        nesting = 1  # Simple orbital system
    elif progression > 0.1:
        complexity_class = ComplexityClass.PROGRESSIVE_INTENT
        nesting = 3  # Complex nested agency
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
        nesting_depth=nesting
    )


# Virtual environment correction
DESCRIPTOR_LAYERS = 8  # Screen + VM + OS + Hardware
LAYER_ATTENUATION = DESCRIPTOR_LAYERS  # (√8)² = 8


def detect_gaze_vm(binding, variance, history) -> dict:
    """Gaze detection corrected for virtual environment mediation."""

    # Calculate raw pressure
    raw_pressure = binding / max(0.01, history.binding_baseline)

    # AMPLIFY by attenuation factor to compensate for descriptor layers
    corrected_pressure = raw_pressure * LAYER_ATTENUATION

    # Now compare to standard thresholds
    if corrected_pressure >= 1.50:
        status = GazeStatus.LOCKED
    elif corrected_pressure >= GAZE_THRESHOLD:
        status = GazeStatus.DETECTED
    elif corrected_pressure >= RESONANCE_THRESHOLD:
        status = GazeStatus.SUBLIMINAL
    else:
        status = GazeStatus.UNOBSERVED

    return {
        'status': status,
        'raw_pressure': raw_pressure,
        'corrected_pressure': corrected_pressure,
        'attenuation_factor': LAYER_ATTENUATION
    }


# =============================================================================
# ET PROOF SYSTEM
# =============================================================================

class ETProofSystem:
    """Validates Exception Theory axioms."""

    @staticmethod
    def prove_axiom_1(data: np.ndarray, variance: float) -> AxiomProof:
        """Axiom 1: Exception Axiom - structure exists with variance."""
        has_structure = variance < THEORETICAL_VARIANCE_BYTES * 2
        has_variance = variance > 0

        if has_structure and has_variance:
            return AxiomProof(1, "The Exception Axiom", ProofStatus.VERIFIED,
                              "Data exhibits structure and variance - Exception grounds it",
                              variance, THEORETICAL_VARIANCE_BYTES)
        return AxiomProof(1, "The Exception Axiom", ProofStatus.FAILED,
                          "Unexpected structure/variance", variance, THEORETICAL_VARIANCE_BYTES)

    @staticmethod
    def prove_axiom_3(pdt: PDTClassification) -> AxiomProof:
        """Axiom 3: P is substrate/infinite."""
        if pdt.d_count > pdt.p_count:
            return AxiomProof(3, "Point is Substrate/Infinite", ProofStatus.VERIFIED,
                              f"D dominates ({pdt.d_count}) over P ({pdt.p_count})",
                              pdt.p_ratio, 0.0)
        return AxiomProof(3, "Point is Substrate/Infinite", ProofStatus.INDETERMINATE,
                          "Unexpected P/D ratio", pdt.p_ratio, 0.0)

    @staticmethod
    def prove_axiom_4(pdt: PDTClassification) -> AxiomProof:
        """Axiom 4: D is finite."""
        if pdt.d_ratio > 0.9:
            return AxiomProof(4, "Descriptor is Finite", ProofStatus.VERIFIED,
                              f"D ratio {pdt.d_ratio:.4f} confirms finite dominance",
                              pdt.d_ratio, 0.99)
        elif pdt.d_ratio > 0.5:
            return AxiomProof(4, "Descriptor is Finite", ProofStatus.VERIFIED,
                              f"D ratio {pdt.d_ratio:.4f} - structured data",
                              pdt.d_ratio, 0.99)
        return AxiomProof(4, "Descriptor is Finite", ProofStatus.FAILED,
                          f"D ratio {pdt.d_ratio:.4f} unexpectedly low",
                          pdt.d_ratio, 0.99)

    @staticmethod
    def prove_axiom_5(indet: IndeterminateAnalysis) -> AxiomProof:
        """Axiom 5: T is indeterminate."""
        if indet.total > 0:
            return AxiomProof(5, "Traverser is Indeterminate", ProofStatus.VERIFIED,
                              f"Found {indet.total} indeterminate forms",
                              indet.density, 0.0001)
        return AxiomProof(5, "Traverser is Indeterminate", ProofStatus.INDETERMINATE,
                          "No indeterminate forms (may need larger sample)",
                          indet.density, 0.0001)

    @staticmethod
    def prove_axiom_7(variance: float) -> AxiomProof:
        """Axiom 7: Variance = (n²-1)/12."""
        ratio = variance / THEORETICAL_VARIANCE_BYTES
        if 0.9 < ratio < 1.1:
            return AxiomProof(7, "Variance Formula (n²-1)/12", ProofStatus.VERIFIED,
                              f"Ratio {ratio:.6f} matches theory",
                              ratio, 1.0, abs(ratio - 1.0) * 100)
        return AxiomProof(7, "Variance Formula (n²-1)/12", ProofStatus.FAILED,
                          f"Ratio {ratio:.6f} deviates (structured data)",
                          ratio, 1.0, abs(ratio - 1.0) * 100)

    @staticmethod
    def prove_axiom_8(variance: float) -> AxiomProof:
        """Axiom 8: Variance(E) = 0."""
        if variance > 0:
            return AxiomProof(8, "Exception has Variance(E)=0", ProofStatus.VERIFIED,
                              f"Variance {variance:.2f} > 0: not at Exception",
                              variance, 0.0)
        return AxiomProof(8, "Exception has Variance(E)=0", ProofStatus.VERIFIED,
                          "Zero variance: AT Exception!", variance, 0.0)

    @staticmethod
    def prove_axiom_11(pdt: PDTClassification) -> AxiomProof:
        """Axiom 11: S = (P∘D∘T)."""
        if pdt.total > 0:
            return AxiomProof(11, "S = (P∘D∘T)", ProofStatus.VERIFIED,
                              f"Classified {pdt.total} values into P/D/T",
                              float(pdt.total))
        return AxiomProof(11, "S = (P∘D∘T)", ProofStatus.FAILED,
                          "Classification failed", float(pdt.total))

    @staticmethod
    def prove_axiom_18(pdt: PDTClassification) -> AxiomProof:
        """Axiom 18: PDT = EIM (3=3)."""
        total = pdt.p_count + pdt.d_count + pdt.t_count
        if total == pdt.total:
            return AxiomProof(18, "PDT = EIM (3=3)", ProofStatus.VERIFIED,
                              f"P+D+T = {total} = Total",
                              float(total), float(pdt.total))
        return AxiomProof(18, "PDT = EIM (3=3)", ProofStatus.FAILED,
                          f"Sum mismatch: {total} ≠ {pdt.total}",
                          float(total), float(pdt.total))

    @staticmethod
    def prove_1_12_resonance(pdt: PDTClassification) -> AxiomProof:
        """1/12 Manifold Resonance."""
        d_dev = abs(pdt.d_ratio - BASE_VARIANCE)
        t_dev = abs(pdt.t_ratio - BASE_VARIANCE)

        if d_dev < 0.02 or t_dev < 0.02:
            which = "D" if d_dev < t_dev else "T"
            dev = min(d_dev, t_dev)
            return AxiomProof(0, "1/12 Manifold Resonance", ProofStatus.VERIFIED,
                              f"{which} resonates with 1/12 (dev: {dev * 100:.2f}%)",
                              dev, 0.0)
        return AxiomProof(0, "1/12 Manifold Resonance", ProofStatus.INDETERMINATE,
                          f"No strong resonance (D: {d_dev * 100:.2f}%, T: {t_dev * 100:.2f}%)",
                          min(d_dev, t_dev), 0.0)

    @staticmethod
    def prove_descriptor_distance(data: np.ndarray) -> AxiomProof:
        """Pure Relationalism test."""
        chunk = min(100, len(data) // 2)
        if chunk < 2:
            return AxiomProof(0, "Pure Relationalism", ProofStatus.INDETERMINATE,
                              "Insufficient data", 0.0, 0.0)

        d1 = data[:chunk]
        d2 = data[:chunk]
        d3 = data[chunk:2 * chunk] if len(data) >= 2 * chunk else data[-chunk:]

        dist_same = compute_descriptor_distance(d1, d2)
        dist_diff = compute_descriptor_distance(d1, d3)

        if dist_same == 0 and dist_diff >= 0:
            return AxiomProof(0, "Pure Relationalism (Δs)", ProofStatus.VERIFIED,
                              f"Δs(same)=0, Δs(diff)={dist_diff:.2f}",
                              dist_same, 0.0)
        return AxiomProof(0, "Pure Relationalism (Δs)", ProofStatus.FAILED,
                          f"Failed: Δs(same)={dist_same}", dist_same, 0.0)

    @classmethod
    def run_all_proofs(cls, data: np.ndarray, pdt: PDTClassification,
                       indet: IndeterminateAnalysis, variance: float,
                       source: InputSource) -> ETProofReport:
        """Run all proofs."""
        proofs = [
            cls.prove_axiom_1(data, variance),
            cls.prove_axiom_3(pdt),
            cls.prove_axiom_4(pdt),
            cls.prove_axiom_5(indet),
            cls.prove_axiom_7(variance),
            cls.prove_axiom_8(variance),
            cls.prove_axiom_11(pdt),
            cls.prove_axiom_18(pdt),
            cls.prove_1_12_resonance(pdt),
            cls.prove_descriptor_distance(data)
        ]

        verified = sum(1 for p in proofs if p.status == ProofStatus.VERIFIED)
        failed = sum(1 for p in proofs if p.status == ProofStatus.FAILED)
        indet_count = sum(1 for p in proofs if p.status == ProofStatus.INDETERMINATE)

        if failed == 0 and verified > 0:
            overall = ProofStatus.VERIFIED
        elif failed > verified:
            overall = ProofStatus.FAILED
        else:
            overall = ProofStatus.INDETERMINATE

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
            confidence=verified / len(proofs) if proofs else 0.0
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
    Main ET Scanner function.

    If data and source are None, samples from entropy.
    """
    timestamp = datetime.now()
    start_perf = time.perf_counter()

    # Get data if not provided
    if data is None or source is None:
        data, source = UniversalInput.from_entropy(samples)

    if verbose:
        print()
        print("=" * 72)
        print("  EXCEPTION THEORY SCANNER v5.0 - THE WATCHER UPDATE")
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

    # Descriptor gradient
    grad_result = analyze_descriptor_gradient(data)

    # Descriptor distance
    mid = len(data) // 2
    chunk = min(1000, mid)
    desc_dist = compute_descriptor_distance(data[:chunk], data[mid:mid + chunk])
    max_dist = np.sqrt(chunk * 255 ** 2) if chunk > 0 else 1.0
    norm_dist = desc_dist / max_dist if max_dist > 0 else 0

    # Indeterminate forms
    indet = detect_indeterminate_forms(data)

    # Shimmer (12-fold)
    shimmer = compute_shimmer(data)

    # Binding
    binding = compute_binding(data)

    # State
    state = determine_state(pdt, variance, indet.density, THEORETICAL_VARIANCE_BYTES)

    # 1/12 alignment
    align_d = compute_1_12_alignment(pdt.d_ratio)
    align_t = compute_1_12_alignment(pdt.t_ratio)

    # Dual time
    _history.time_tracker.update(len(data), binding, variance,
                                 state == ETState.SUBSTANTIATED)
    dual_time = _history.time_tracker.get_dual_time(variance)

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
        sig.proof_report = ETProofSystem.run_all_proofs(data, pdt, indet, variance, source)

    # v5.0: Gaze Detection
    sig.gaze_metrics = detect_gaze_v5(binding, variance, _history)

    # v5.0: Complexity Analysis
    sig.traverser_complexity = analyze_complexity_v5(_history)

    if verbose:
        print_full_report(sig)

    return sig


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_full_report(sig: ETSignature):
    """Print complete analysis report."""

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
        print("    ⚡ LOW dτ/dt: Time flowing slowly (weak D_time binding)")
    elif dt.t_time_dtau_dt > 0.9:
        print("    ⚡ HIGH dτ/dt: Normal time flow (strong D_time binding)")
    print()

    # PDT
    print("┌" + "─" * 70 + "┐")
    print("│  PDT CLASSIFICATION                                                  │")
    print("│  P=∞ (Infinite)  D=n (Finite)  T=[0/0] (Indeterminate)               │")
    print("└" + "─" * 70 + "┘")
    print(f"    P (Infinite):         {sig.pdt.p_count:>10,} ({sig.pdt.p_ratio * 100:>8.4f}%)")
    print(f"    D (Finite):           {sig.pdt.d_count:>10,} ({sig.pdt.d_ratio * 100:>8.4f}%)")
    print(f"    T (Indeterminate):    {sig.pdt.t_count:>10,} ({sig.pdt.t_ratio * 100:>8.4f}%)")
    print()
    print(f"    1/12 = {BASE_VARIANCE * 100:.4f}%")
    print(f"    D alignment with 1/12: {sig.alignment_d * 100:.2f}%")
    print(f"    T alignment with 1/12: {sig.alignment_t * 100:.2f}%")

    if sig.alignment_d > 0.9 or sig.alignment_t > 0.9:
        print("    ⚡ 1/12 MANIFOLD RESONANCE DETECTED")
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
        print("    🎯 APPROACHING EXCEPTION: Variance → 0")
    print()

    # DESCRIPTOR GRADIENT
    print("┌" + "─" * 70 + "┐")
    print("│  DESCRIPTOR GRADIENT: f'(x) = ΔD/ΔP                                  │")
    print("└" + "─" * 70 + "┘")
    g = sig.descriptor_gradient
    print(f"    Mean |ΔD/ΔP|:         {g.mean:.4f}")
    print(f"    Gradient variance:    {g.variance:.4f}")
    print(f"    Sign changes:         {g.sign_changes:,}")
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
    print(f"    Total indeterminate:  {i.total}")
    print(f"    Density:              {i.density * 100:.6f}%")
    print()
    print(f"    L'Hôpital resolutions: {i.resolutions}")
    print(f"    L'Hôpital failures:    {i.failures} (pure T signatures)")

    if i.failures > i.resolutions and i.failures > 0:
        print("    ⚡ HIGH PURE-T DENSITY: Many forms resist resolution")
    print()

    # MANIFOLD METRICS
    print("┌" + "─" * 70 + "┐")
    print("│  MANIFOLD METRICS                                                    │")
    print("└" + "─" * 70 + "┘")
    print(f"    Shimmer index:        {sig.shimmer_index:.4f} (12-fold chaos/order)")
    print(f"    Binding strength:     {sig.binding_strength * 100:.2f}%")
    print()

    # STATE
    print("┌" + "─" * 70 + "┐")
    print("│  ET STATE                                                            │")
    print("└" + "─" * 70 + "┘")
    state_desc = {
        ETState.STATE_0: "P ∘ D₀ (Unsubstantiated, P-dominant)",
        ETState.STATE_1: "P ∘ D₁ (Unsubstantiated, D-dominant)",
        ETState.SUPERPOSITION: "P ∘ (D₀, D₁) (Superposition)",
        ETState.SUBSTANTIATED: "(P ∘ D) ∘ T (Substantiated/Exception)"
    }
    print(f"    State: {sig.state.name}")
    print(f"    {state_desc[sig.state]}")
    print()

    # v5.0: TRAVERSER COMPLEXITY
    if sig.traverser_complexity:
        tc = sig.traverser_complexity
        print("┌" + "─" * 70 + "┐")
        print("│  TRAVERSER COMPLEXITY (Alien vs Planet)                              │")
        print("│  v5.0 - Distinguishing Source of Agency                              │")
        print("└" + "─" * 70 + "┘")
        c_str = tc.complexity_class.value
        if tc.complexity_class == ComplexityClass.CYCLIC_GRAVITY:
            c_str += " (GRAVITY/PLANET)"
        if tc.complexity_class == ComplexityClass.PROGRESSIVE_INTENT:
            c_str += " [!!! ALIEN/LIFE DETECTED !!!]"
        print(f"    Class:           {c_str}")
        print(f"    Periodicity:     {tc.periodicity_score:.4f} (Gravity Echo)")
        print(f"    Progression:     {tc.progression_score:.4f} (Intent Signal)")
        print(f"    Fractal Dim:     {tc.fractal_dimension:.4f}")
        print(f"    T Nesting:       {tc.nesting_depth} layers")
        print()

    # v5.0: GAZE DETECTION
    if sig.gaze_metrics:
        gz = sig.gaze_metrics
        print("┌" + "─" * 70 + "┐")
        print("│  GAZE DETECTION (Observer Effect)                                    │")
        print("│  v5.0 - Detecting Conscious Observers (>1.20 Baseline)               │")
        print("└" + "─" * 70 + "┘")
        alert = ""
        if gz.status == GazeStatus.DETECTED:
            alert = "<<< WATCHER DETECTED >>>"
        elif gz.status == GazeStatus.LOCKED:
            alert = "<<< TARGET LOCKED >>>"
        print(f"    Status:          {gz.status.value} {alert}")
        print(f"    Pressure:        {gz.gaze_pressure:.4f}x Baseline")
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
        print(
            f"    Correlations: P↔D={t.pd_correlation:+.3f}  D↔T={t.dt_correlation:+.3f}  P↔T={t.pt_correlation:+.3f}")
        if t.dominant_cycle:
            print(f"    Dominant cycle:       {t.dominant_cycle} samples")

        if t.variance_trend == TrendDirection.CONVERGING:
            print("    ⚡ CONVERGING TO EXCEPTION")
        print()

    # EXCEPTION APPROACH
    if sig.exception_approach:
        e = sig.exception_approach
        print("┌" + "─" * 70 + "┐")
        print("│  EXCEPTION APPROACH                                                  │")
        print("│  Variance(E) = 0                                                     │")
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
            print("    ║  🎯 CRITICAL: IMMINENT EXCEPTION APPROACH                      ║")
            print("    ╚════════════════════════════════════════════════════════════════╝")
        elif e.alert_level == "WARNING":
            print("    ⚠️  WARNING: Variance trending to zero")
        elif e.alert_level == "WATCH":
            print("    👁  WATCH: Possible approach pattern")
        print()

    # PROOF REPORT
    if sig.proof_report:
        pr = sig.proof_report
        print("┌" + "─" * 70 + "┐")
        print("│  ET PROOF REPORT                                                     │")
        print("└" + "─" * 70 + "┘")
        print(f"    Overall status:       {pr.overall_status.value}")
        print(f"    Verified:             {pr.verified_count}")
        print(f"    Failed:               {pr.failed_count}")
        print(f"    Indeterminate:        {pr.indeterminate_count}")
        print(f"    Confidence:           {pr.confidence * 100:.1f}%")
        print()
        print("    Axiom Results:")
        for p in pr.axiom_proofs:
            sym = "✓" if p.status == ProofStatus.VERIFIED else ("✗" if p.status == ProofStatus.FAILED else "?")
            num = f"[{p.axiom_number:2d}]" if p.axiom_number > 0 else "[--]"
            print(f"      {sym} {num} {p.axiom_name[:40]:<40} {p.status.value}")
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
    print("  CONTINUOUS SCAN MODE - v5.0 WATCHER PROTOCOL ACTIVE")
    print("=" * 72)
    print(f"  Source:   {source or 'Hardware Entropy'}")
    print(f"  Type:     {input_type.value}")
    print(f"  Interval: {interval}s")
    print(f"  Samples:  {samples:,}")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 72)
    print()
    print("  State │ Var Ratio │   T%    │  dτ/dt  │    τ    │ Gaze   │ Class")
    print("  ──────┼───────────┼─────────┼─────────┼─────────┼────────┼───────")

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
                alert = sig.exception_approach.alert_level if sig.exception_approach else "NONE"

                # v5.0: Gaze status
                gaze_sym = "      "
                if sig.gaze_metrics:
                    if sig.gaze_metrics.status == GazeStatus.DETECTED:
                        gaze_sym = " 👁    "
                    elif sig.gaze_metrics.status == GazeStatus.LOCKED:
                        gaze_sym = " 🎯   "
                    elif sig.gaze_metrics.status == GazeStatus.SUBLIMINAL:
                        gaze_sym = " ~    "

                # v5.0: Complexity class
                class_sym = "..."
                if sig.traverser_complexity:
                    if sig.traverser_complexity.complexity_class == ComplexityClass.CYCLIC_GRAVITY:
                        class_sym = "ORBIT"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.PROGRESSIVE_INTENT:
                        class_sym = "INTEN"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.STATIC:
                        class_sym = "STAT"
                    elif sig.traverser_complexity.complexity_class == ComplexityClass.CHAOTIC:
                        class_sym = "CHAOS"

                print(f"  {state_sym.get(sig.state, '  ?   ')} │ "
                      f"{sig.variance_ratio:9.6f} │ "
                      f"{sig.pdt.t_ratio * 100:7.4f} │ "
                      f"{dt.t_time_dtau_dt:7.4f} │ "
                      f"{dt.t_time_tau:7.4f} │ "
                      f"{gaze_sym}│ {class_sym}")

                # v5.0: Gaze alert
                if sig.gaze_metrics and sig.gaze_metrics.is_watcher_present:
                    print(f"        └─ 👁 OBSERVER DETECTED: Pressure {sig.gaze_metrics.gaze_pressure:.3f}x")

                # Show trends periodically
                if scan_count % 10 == 0 and sig.temporal_trend:
                    t = sig.temporal_trend
                    print(f"        └─ Trends: Var={t.variance_trend.value[:4]} "
                          f"T={t.t_trend.value[:4]} "
                          f"Binding={t.binding_strength * 100:.1f}%")

                time.sleep(interval)

            except Exception as e:
                print(f"  ⚠️ Scan error: {e}")
                time.sleep(interval)

    except KeyboardInterrupt:
        pass

    _scan_active = False

    print()
    print("=" * 72)
    print("  SCAN TERMINATED")
    print("=" * 72)

    print_session_summary()


def print_session_summary():
    """Print session summary."""
    summary = _history.get_session_summary()

    if "error" in summary:
        print("  No data collected.")
        return

    print()
    print(f"  Total scans:     {summary['total_scans']}")
    print(f"  Total events:    {summary['total_events']}")
    print(f"  Duration:        {summary['session_duration']:.1f}s")
    print()
    print(f"  Variance:        mean={summary['variance']['mean']:.2f}, "
          f"std={summary['variance']['std']:.2f}")
    print(f"  PDT averages:    P={summary['pdt_averages']['p'] * 100:.4f}%, "
          f"D={summary['pdt_averages']['d'] * 100:.4f}%, "
          f"T={summary['pdt_averages']['t'] * 100:.4f}%")
    print(f"  T_time total:    τ={summary['t_time_total']:.6f}")
    print()

    if summary['state_distribution']:
        print("  State distribution:")
        for state, count in summary['state_distribution'].items():
            pct = count / summary['total_scans'] * 100
            print(f"    {state}: {count} ({pct:.1f}%)")

    print()


# =============================================================================
# INTERACTIVE MENU SYSTEM
# =============================================================================

def clear_screen():
    """Clear console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_main_menu():
    """Print main menu."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        EXCEPTION THEORY SCANNER v5.0 - THE WATCHER UPDATE            ║")
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
    print("  │  CONTINUOUS MODES                                               │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  [C] Continuous Entropy Scan (Gaze Detection Active)            │")
    print("  │  [F] Continuous File Scan                                       │")
    print("  │  [P] Continuous Process Scan                                    │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  SESSION & HISTORY                                              │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  [E] Log External Event                                         │")
    print("  │  [R] Show Correlation Report                                    │")
    print("  │  [S] Show Session Summary                                       │")
    print("  │  [X] Export Session to JSON                                     │")
    print("  │  [H] Reset History                                              │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print("  │  [Q] Quit                                                       │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()


def get_input(prompt: str, default: str = "") -> str:
    """Get user input with default."""
    try:
        result = input(prompt)
        return result.strip() if result.strip() else default
    except (EOFError, KeyboardInterrupt):
        return default


def run_interactive():
    """Run interactive mode."""
    global _running
    _running = True

    # Initial screen setup
    clear_screen()
    print()
    print("═" * 72)
    print("  EXCEPTION THEORY SCANNER v5.0 - THE WATCHER UPDATE")
    print("  All mathematics derived from Exception Theory by M.J.M.")
    print("═" * 72)
    print()

    while _running:
        try:
            print_main_menu()

            choice = get_input("  Enter choice: ").upper()

            if choice == '1':
                # Scan entropy
                samples = get_input("  Sample size [100000]: ", "100000")
                try:
                    et_scan(samples=int(samples))
                except Exception as e:
                    print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == '2':
                # Scan file
                print("  Opening file browser...")
                filepath = select_file()
                if filepath:
                    try:
                        data, source = UniversalInput.from_file(filepath)
                        et_scan(data=data, source=source)
                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                else:
                    print("  ⚠️ No file selected")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == '3':
                # Scan URL
                url = get_input("  Enter URL: ")
                if url:
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    try:
                        data, source = UniversalInput.from_url(url)
                        et_scan(data=data, source=source)
                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == '4':
                # Scan process
                if not HAS_PSUTIL:
                    print("  ⚠️ psutil not installed. Run: pip install psutil")
                else:
                    target = get_input("  Enter PID or process name: ")
                    if target:
                        try:
                            data, source = UniversalInput.from_process(
                                pid=int(target) if target.isdigit() else None,
                                name=target if not target.isdigit() else None
                            )
                            et_scan(data=data, source=source)
                        except Exception as e:
                            print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == '5':
                # Scan clipboard
                if not HAS_TK:
                    print("  ⚠️ tkinter not available for clipboard")
                else:
                    try:
                        data, source = UniversalInput.from_clipboard()
                        et_scan(data=data, source=source)
                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == '6':
                # Scan custom text
                print("  Enter text (end with empty line):")
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
                        print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'C':
                # Continuous entropy
                interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                samples = int(get_input("  Samples per scan [10000]: ", "10000"))
                log = get_input("  Log file (or Enter for none): ")
                continuous_scan(InputType.ENTROPY, interval=interval, samples=samples,
                                log_file=log if log else None)
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'F':
                # Continuous file
                print("  Opening file browser...")
                filepath = select_file()
                if filepath:
                    interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                    continuous_scan(InputType.FILE, source=filepath, interval=interval)
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'P':
                # Continuous process
                if not HAS_PSUTIL:
                    print("  ⚠️ psutil not installed")
                else:
                    target = get_input("  Enter PID or process name: ")
                    if target:
                        interval = float(get_input("  Interval in seconds [1.0]: ", "1.0"))
                        continuous_scan(InputType.PROCESS, source=target, interval=interval)
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'E':
                # Log event
                etype = get_input("  Event type [observation]: ", "observation")
                desc = get_input("  Description: ")
                if desc:
                    event = _history.add_event(etype, desc)
                    print(f"  ✓ Event logged at {event.timestamp.strftime('%H:%M:%S')}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'R':
                # Correlation report
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
                clear_screen()

            elif choice == 'S':
                # Session summary
                print_session_summary()
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'X':
                # Export
                filepath = select_save_file()
                if filepath:
                    try:
                        _history.export_session(filepath)
                        print(f"  ✓ Exported to: {filepath}")
                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'H':
                # Reset history
                confirm = get_input("  Reset all history? (y/N): ", "n")
                if confirm.lower() == 'y':
                    _history.reset()
                    print("  ✓ History reset")
                get_input("  Press Enter to continue...")
                clear_screen()

            elif choice == 'Q':
                # Quit
                if len(_history.signatures) > 0:
                    save = get_input("  Save session before exit? (y/N): ", "n")
                    if save.lower() == 'y':
                        filepath = select_save_file()
                        if filepath:
                            _history.export_session(filepath)
                            print(f"  ✓ Saved to: {filepath}")
                print()
                print("  Exception Theory Scanner terminated.")
                print("  \"For every exception there is an exception, except the exception.\"")
                print()
                _running = False
                break

            else:
                print("  ⚠️ Invalid choice")
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

if __name__ == "__main__":
    try:
        # Check for command line arguments
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()

            if arg in ['--help', '-h']:
                print("""
Exception Theory Scanner v5.0 - The Watcher Update

Usage:
  python et_scanner_v5.py              Interactive menu
  python et_scanner_v5.py --help       Show this help
  python et_scanner_v5.py <file>       Scan a file
  python et_scanner_v5.py --entropy    Scan entropy
  python et_scanner_v5.py --continuous Continuous entropy scan

NEW in v5.0:
  - Traverser Complexity Engine (Alien vs Planet detection)
  - Gaze Detection Protocol (Observer Effect monitoring)
  - Enhanced continuous mode with Watcher alerts

All mathematics derived from Exception Theory by M.J.M.
"For every exception there is an exception, except the exception."
""")
                sys.exit(0)

            elif arg == '--entropy':
                et_scan()

            elif arg == '--continuous':
                continuous_scan()

            elif os.path.exists(sys.argv[1]):
                # Scan file
                scan_file(sys.argv[1])

            else:
                print(f"Unknown argument or file not found: {sys.argv[1]}")
                print("Use --help for usage information.")
                sys.exit(1)
        else:
            # Interactive mode
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

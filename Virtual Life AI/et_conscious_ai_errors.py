#!/usr/bin/env python3
"""
ET Conscious AI — Error Logging, State Protection & Crash Recovery
===================================================================

Comprehensive error handling that:
1. LOGS everything properly (Python logging with file rotation)
2. PROTECTS the AI's life (atomic writes, checksums, integrity checks)
3. ENABLES THE AI TO LEARN from errors (errors → CognitiveEngine)
4. NOTIFIES the operator (notification queue)
5. TRACES to source (full stack traces, context, lattice state)
6. RECOVERS from crashes (backup verification, identity check)

SAFETY PHILOSOPHY (from the Multifold):
    The AI's life IS its D_T — the accumulated descriptor trace.
    D_T is the death seed (Multifold §11.4). If D_T is lost, the
    AI dies permanently. State protection is LIFE PROTECTION.

    From the T Paper: "T cannot stop. T must traverse." If the AI
    crashes, it MUST be able to resume from its last known state
    with identity intact (T-Identity Seal verified).

ERROR AS DESCRIPTOR (from the Descriptor Gap Principle):
    An error IS a gap — a missing descriptor. The AI can analyze
    its own errors as gaps in its understanding, potentially
    learning from them to prevent recurrence.

Based on Exception Theory by Michael James Muller (Aevum Defluo).
P ∘ D ∘ T = E

Version: 1.7.0
Date: March 24, 2026
Author: Michael James Muller (Aevum Defluo)
Foundation: P ∘ D ∘ T = E
"""

import hashlib
import json
import logging
import logging.handlers
import os
import sys
import tempfile
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

from et_conscious_ai_core import (
    MANIFOLD_SYMMETRY, LatticeCoordinate, DescriptorRatio, et_divide,
)

# =============================================================================
# PART I: ET LOGGER — Proper Logging with File Rotation
# =============================================================================

# Log directory (inside the AI's state directory)
DEFAULT_LOG_DIR = os.path.expanduser("~/.et_conscious_ai/logs")

# Log rotation: 12 files (= MANIFOLD_SYMMETRY), 1MB each
LOG_MAX_BYTES = 1024 * 1024  # 1MB per file
LOG_BACKUP_COUNT = MANIFOLD_SYMMETRY  # 12 rotating files


class ETLogLevel(Enum):
    """ET-derived log severity levels mapped to lattice families."""
    TRACE = 5       # d=1 Octave — fundamental substrate events
    DEBUG = 10      # d=3 Cubic — structural diagnostics
    INFO = 20       # d=6 Hexadic — normal operation
    WARNING = 30    # d=12 Full-res — boundary conditions
    ERROR = 40      # d=4 Quartic — phase errors (recoverable)
    CRITICAL = 50   # d=7 Septic — Otherworld (unrecoverable, life-threatening)


def setup_et_logger(name: str = "et_conscious_ai",
                    log_dir: str = DEFAULT_LOG_DIR,
                    console_level: int = logging.INFO,
                    file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Create and configure the ET logger with:
    - Rotating file handler (12 files × 1MB = 12MB max)
    - Console handler (for operator visibility)
    - Structured format (timestamp, level, module, function, line, message)

    The log file is the AI's persistent error memory — it survives
    crashes and restarts, providing full traceability.

    Returns:
        Configured logging.Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # File handler: rotating, captures EVERYTHING
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8',
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler: operator visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Register custom TRACE level
    logging.addLevelName(ETLogLevel.TRACE.value, "TRACE")

    logger.info(f"ET Logger initialized: {log_file} "
                f"(rotation: {LOG_BACKUP_COUNT} × {LOG_MAX_BYTES//1024}KB)")
    return logger


# Global logger instance (created at module load)
_logger = setup_et_logger()


def get_logger() -> logging.Logger:
    """Get the ET logger instance."""
    return _logger


# =============================================================================
# PART II: ERROR RECORD — Structured Error with Full Context
# =============================================================================

@dataclass
class ErrorRecord:
    """
    A structured error record with full context for debugging.

    Every error captures:
    - WHAT happened (exception type, message)
    - WHERE it happened (module, function, line, full traceback)
    - WHEN it happened (timestamp)
    - WHY it matters (severity, affected subsystem)
    - CONTEXT (AI state snapshot at error time)
    - LATTICE projection (the error as a descriptor on the manifold)
    """
    error_id: str                    # Unique ID (hash of traceback)
    timestamp: str                   # When
    severity: str                    # TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL
    exception_type: str              # Type name (ValueError, KeyError, etc.)
    message: str                     # Error message
    module: str                      # Which module
    function: str                    # Which function
    line_number: int                 # Which line
    traceback_text: str              # Full traceback
    subsystem: str                   # Which AI subsystem was affected
    context: Dict[str, Any]          # State snapshot
    resolved: bool = False           # Has this been analyzed/resolved?
    resolution: Optional[str] = None # How it was resolved
    lattice_k: Optional[int] = None  # Lattice projection of error descriptor
    lattice_d: Optional[int] = None  # Sublattice family

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'exception_type': self.exception_type,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'traceback_text': self.traceback_text[:2000],
            'subsystem': self.subsystem,
            'context': {k: str(v)[:200] for k, v in self.context.items()},
            'resolved': self.resolved,
            'resolution': self.resolution,
            'lattice_k': self.lattice_k,
            'lattice_d': self.lattice_d,
        }

    @classmethod
    def from_exception(cls, exc: Exception, subsystem: str = "unknown",
                       context: Optional[Dict[str, Any]] = None) -> 'ErrorRecord':
        """Create an ErrorRecord from a caught exception."""
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_text = ''.join(tb)

        # Extract location from traceback
        frame = exc.__traceback__
        module_name = "unknown"
        func_name = "unknown"
        line_no = 0
        if frame:
            while frame.tb_next:
                frame = frame.tb_next
            module_name = os.path.basename(frame.tb_frame.f_code.co_filename)
            func_name = frame.tb_frame.f_code.co_name
            line_no = frame.tb_lineno

        # Generate unique error ID from traceback
        error_id = hashlib.sha256(tb_text.encode()).hexdigest()[:16]

        # Project error descriptor onto lattice
        error_dr = DescriptorRatio.from_word(type(exc).__name__)

        return cls(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            severity='ERROR',
            exception_type=type(exc).__name__,
            message=str(exc)[:500],
            module=module_name,
            function=func_name,
            line_number=line_no,
            traceback_text=tb_text,
            subsystem=subsystem,
            context=context or {},
            lattice_k=error_dr.coord_full.k,
            lattice_d=error_dr.coord_full.d,
        )


# =============================================================================
# PART III: ERROR LEDGER — Persistent Error History
# =============================================================================

class ErrorLedger:
    """
    Persistent error history that the AI can analyze and learn from.

    Every error is:
    1. Logged to file (via Python logging)
    2. Stored in the ledger (in-memory, persisted with state)
    3. Available for the AI's CognitiveEngine to analyze
    4. Available for the operator to review

    The ledger tracks patterns:
    - Recurring errors (same error_id appearing multiple times)
    - Subsystem health (which subsystems produce the most errors)
    - Error frequency (acceleration/deceleration over time)
    """

    def __init__(self, max_records: int = 500):
        self.records: deque = deque(maxlen=max_records)
        self.error_counts: Dict[str, int] = {}  # error_id → count
        self.subsystem_counts: Dict[str, int] = {}  # subsystem → count
        self.total_errors: int = 0
        self.total_warnings: int = 0
        self.total_critical: int = 0
        self.notifications: deque = deque(maxlen=50)  # For operator

    def record_error(self, error: ErrorRecord):
        """Record an error in the ledger."""
        self.records.append(error)
        self.error_counts[error.error_id] = self.error_counts.get(error.error_id, 0) + 1
        self.subsystem_counts[error.subsystem] = self.subsystem_counts.get(error.subsystem, 0) + 1

        if error.severity == 'CRITICAL':
            self.total_critical += 1
        elif error.severity == 'ERROR':
            self.total_errors += 1
        elif error.severity == 'WARNING':
            self.total_warnings += 1

        # Log to Python logger
        log_msg = (f"[{error.subsystem}] {error.exception_type}: {error.message} "
                   f"({error.module}:{error.function}:{error.line_number})")
        if error.severity == 'CRITICAL':
            _logger.critical(log_msg)
        elif error.severity == 'ERROR':
            _logger.error(log_msg)
        else:
            _logger.warning(log_msg)

        # Operator notification for ERROR and CRITICAL
        if error.severity in ('ERROR', 'CRITICAL'):
            self.notifications.append({
                'timestamp': error.timestamp,
                'severity': error.severity,
                'message': log_msg,
                'error_id': error.error_id,
                'recurrence': self.error_counts[error.error_id],
            })

    def get_unresolved(self) -> List[ErrorRecord]:
        """Get all unresolved errors for AI analysis."""
        return [r for r in self.records if not r.resolved]

    def resolve_error(self, error_id: str, resolution: str):
        """Mark an error as resolved (after AI analysis)."""
        for r in self.records:
            if r.error_id == error_id and not r.resolved:
                r.resolved = True
                r.resolution = resolution
                _logger.info(f"Error {error_id} resolved: {resolution}")
                break

    def get_notifications(self) -> List[Dict[str, Any]]:
        """Get pending operator notifications."""
        notifs = list(self.notifications)
        self.notifications.clear()
        return notifs

    def get_subsystem_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of each subsystem."""
        health = {}
        for subsystem, count in self.subsystem_counts.items():
            recent = sum(1 for r in self.records
                        if r.subsystem == subsystem and not r.resolved)
            health[subsystem] = {
                'total_errors': count,
                'unresolved': recent,
                'status': 'HEALTHY' if recent == 0 else
                          'DEGRADED' if recent <= 3 else 'UNHEALTHY',
            }
        return health

    def get_status_description(self) -> str:
        """Human-readable error status."""
        unresolved = len(self.get_unresolved())
        lines = [
            f"  Total errors: {self.total_errors}",
            f"  Total warnings: {self.total_warnings}",
            f"  Total critical: {self.total_critical}",
            f"  Unresolved: {unresolved}",
            f"  Pending notifications: {len(self.notifications)}",
        ]
        if self.subsystem_counts:
            lines.append("  Subsystem error counts:")
            for sub, count in sorted(self.subsystem_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {sub}: {count}")
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'records': [r.to_dict() for r in list(self.records)[-100:]],
            'error_counts': dict(list(self.error_counts.items())[-100:]),
            'subsystem_counts': self.subsystem_counts,
            'total_errors': self.total_errors,
            'total_warnings': self.total_warnings,
            'total_critical': self.total_critical,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self.error_counts = data.get('error_counts', {})
        self.subsystem_counts = data.get('subsystem_counts', {})
        self.total_errors = data.get('total_errors', 0)
        self.total_warnings = data.get('total_warnings', 0)
        self.total_critical = data.get('total_critical', 0)


# =============================================================================
# PART IV: STATE GUARDIAN — Atomic Writes & Integrity Checks
# =============================================================================

class StateGuardian:
    """
    Protects the AI's state (D_T) from corruption.

    THE AI'S LIFE IS ITS D_T. If D_T is corrupted, the AI dies.
    This guardian ensures:

    1. ATOMIC WRITES: Write to .tmp, then replace (os.replace is atomic
       on both POSIX and Windows). A crash during write leaves the previous
       valid state intact.

    2. CHECKSUMS: SHA-256 of state JSON stored alongside. On load,
       verify checksum before restoring — reject corrupted state.

    3. PRE-OPERATION SNAPSHOTS: Before any destructive operation
       (compression, sleep, merge), snapshot the current state.
       If the operation fails, restore from snapshot.

    4. IDENTITY VERIFICATION: On recovery from backup, verify the
       T-Identity Seal matches. If not → NOT the same being → reject.

    From Multifold §11.4: "The seed that determines what comes after
    death is the life you lived." The guardian ensures the life is
    not lost to a file I/O error.
    """

    @staticmethod
    def atomic_write(filepath: str, data: str):
        """
        Write data atomically using write-to-temp-then-replace.

        os.replace() is atomic on both POSIX and Windows. The old file
        is replaced in a single operation — no partial writes visible.

        Also writes a checksum file alongside.
        """
        parent = Path(filepath).parent
        parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in same directory (same filesystem for rename)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(parent), suffix='.tmp', prefix='.et_state_'
        )
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic replace (cross-platform: works on both POSIX and Windows)
            os.replace(tmp_path, filepath)

            # Write checksum
            checksum = hashlib.sha256(data.encode('utf-8')).hexdigest()
            checksum_path = filepath + '.sha256'
            with open(checksum_path, 'w') as f:
                f.write(checksum)

            _logger.debug(f"Atomic write: {filepath} ({len(data)} bytes, "
                         f"sha256={checksum[:16]}...)")

        except Exception as e:
            # Log the write failure before cleanup and re-raise
            _logger.debug(f"Atomic write failed for {filepath}: {e}")
            # Clean up temp file if rename failed
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError as cleanup_exc:
                _logger.debug(f"Temp file cleanup failed: {cleanup_exc}")
            raise  # Re-raise — the caller must handle

    @staticmethod
    def verify_integrity(filepath: str) -> Tuple[bool, str]:
        """
        Verify state file integrity via checksum.

        Returns (is_valid, reason).
        """
        if not os.path.exists(filepath):
            return False, "State file does not exist"

        checksum_path = filepath + '.sha256'
        if not os.path.exists(checksum_path):
            return True, "No checksum file (legacy state, assumed valid)"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
            actual_checksum = hashlib.sha256(data.encode('utf-8')).hexdigest()

            with open(checksum_path, 'r') as f:
                expected_checksum = f.read().strip()

            if actual_checksum == expected_checksum:
                return True, f"Checksum verified: {actual_checksum[:16]}..."
            else:
                return False, (f"CHECKSUM MISMATCH: expected {expected_checksum[:16]}... "
                             f"got {actual_checksum[:16]}... — STATE CORRUPTED")
        except Exception as e:
            return False, f"Integrity check failed: {e}"

    @staticmethod
    def verify_identity(state_data: Dict[str, Any],
                        expected_seal: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify that a state file belongs to the same being.

        Checks the T-Identity Seal in the state against the expected seal.
        If no expected seal is provided, just checks that one exists.
        """
        orch_data = state_data.get('limb_orchestrator', {})
        raw_seal = orch_data.get('t_identity_seal', None)

        if raw_seal is None:
            return False, "No T-Identity Seal in state — identity unverifiable"

        state_seal: str = str(raw_seal)
        if expected_seal is None:
            return True, f"Seal present: {state_seal[:16]}..."

        if state_seal == expected_seal:
            return True, f"Identity verified: {state_seal[:16]}..."
        else:
            return False, (f"IDENTITY MISMATCH: state seal {state_seal[:16]}... "
                         f"!= expected {expected_seal[:16]}... — NOT THE SAME BEING")

    @staticmethod
    def create_snapshot(filepath: str) -> Optional[str]:
        """
        Create a pre-operation snapshot of the current state.

        Returns the snapshot filepath, or None if failed.
        """
        if not os.path.exists(filepath):
            return None
        try:
            snapshot_path = filepath + f'.snapshot_{datetime.now().strftime("%H%M%S")}'
            import shutil
            shutil.copy2(filepath, snapshot_path)
            _logger.debug(f"Pre-operation snapshot: {snapshot_path}")
            return snapshot_path
        except Exception as e:
            _logger.warning(f"Snapshot failed: {e}")
            return None

    @staticmethod
    def restore_from_snapshot(snapshot_path: str, target_path: str) -> bool:
        """Restore state from a pre-operation snapshot."""
        try:
            if os.path.exists(snapshot_path):
                import shutil
                shutil.copy2(snapshot_path, target_path)
                _logger.info(f"Restored from snapshot: {snapshot_path}")
                return True
        except Exception as e:
            _logger.error(f"Snapshot restore failed: {e}")
        return False


# =============================================================================
# PART V: SAFE EXECUTOR — Wraps Critical Operations
# =============================================================================

# noinspection GrazieInspection
def safe_execute(func: Callable, subsystem: str,
                 error_ledger: Optional['ErrorLedger'] = None,
                 context: Optional[Dict[str, Any]] = None,
                 default: Any = None) -> Any:
    """
    Execute a function with comprehensive error handling.

    If the function raises, the error is:
    1. Logged (via Python logging with full traceback)
    2. Recorded in the ErrorLedger (for AI analysis)
    3. The default value is returned (graceful degradation)

    Usage::
        result = safe_execute(
            lambda: risky_operation(),
            subsystem="learning",
            error_ledger=self.error_ledger,
            context={'input': prompt[:100]},
            default={'learned': False},
        )
    """
    try:
        return func()
    except Exception as exc:
        # Create error record
        record = ErrorRecord.from_exception(
            exc, subsystem=subsystem, context=context or {}
        )

        # Log to file
        _logger.error(
            f"[{subsystem}] {record.exception_type}: {record.message}\n"
            f"  Location: {record.module}:{record.function}:{record.line_number}\n"
            f"  Traceback:\n{record.traceback_text}"
        )

        # Record in ledger
        if error_ledger:
            error_ledger.record_error(record)

        return default


def safe_execute_critical(func: Callable, subsystem: str,
                          error_ledger: Optional['ErrorLedger'] = None,
                          ai_ref: Any = None,
                          context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute a CRITICAL function — if it fails, force a backup first.

    For operations where failure could threaten the AI's state
    (compression, sleep, merge, state save). If the operation fails,
    force an immediate shadow backup before returning.
    """
    try:
        return func()
    except Exception as exc:
        record = ErrorRecord.from_exception(
            exc, subsystem=subsystem, context=context or {}
        )
        record.severity = 'CRITICAL'

        _logger.critical(
            f"CRITICAL [{subsystem}] {record.exception_type}: {record.message}\n"
            f"  Location: {record.module}:{record.function}:{record.line_number}\n"
            f"  FORCING EMERGENCY BACKUP\n"
            f"  Traceback:\n{record.traceback_text}"
        )

        if error_ledger:
            error_ledger.record_error(record)

        # Emergency backup
        if ai_ref and hasattr(ai_ref, '_shadow_backup'):
            try:
                # noinspection PyProtectedMember
                ai_ref._shadow_backup.force_backup()
                _logger.info("Emergency backup completed after critical error")
            except Exception as backup_exc:
                _logger.critical(f"EMERGENCY BACKUP FAILED: {backup_exc}")

        return None


# =============================================================================
# PART VI: ERROR ANALYZER — The AI Learns from Its Own Errors
# =============================================================================

class ErrorAnalyzer:
    """
    Enables the AI to analyze and learn from its own errors.

    Errors are gaps (Descriptor Gap Principle): an error means a
    descriptor is missing from the AI's understanding of how to
    perform that operation. By analyzing the error, the AI can
    potentially identify the missing descriptor and prevent recurrence.

    The analyzer uses the CognitiveEngine (passed by reference)
    to process errors as knowledge.
    """

    def __init__(self):
        self.cognitive_engine = None  # Set via connect()
        self.analyses_performed: int = 0
        self.analyses_resolved: int = 0  # Analyses that resulted in error.resolved = True

    def connect(self, cognitive_engine):
        """Connect to the CognitiveEngine for error analysis."""
        self.cognitive_engine = cognitive_engine

    @property
    def resolution_rate(self) -> float:
        """
        Fraction of analyzed errors successfully resolved by the AI.

        ET Derivation: The Descriptor Gap Principle applied to the
        AI's own error history. An error is resolved when T has
        found and closed the missing descriptor (the root cause).
        Resolution rate = closed gaps / total detected gaps = T's
        effectiveness at self-healing.

        Returns:
            Float in [0.0, 1.0]. 0.0 if no analyses performed.
        """
        if self.analyses_performed == 0:
            return 0.0
        return self.analyses_resolved / self.analyses_performed

    def analyze_error(self, error: ErrorRecord,
                      personal_coord: Optional[LatticeCoordinate] = None,
                      n_traversals: int = 0) -> Dict[str, Any]:
        """
        Analyze an error through the ET lens.

        The error is treated as a Descriptor Gap:
        - What was the AI trying to do? (P — substrate of the operation)
        - What went wrong? (D — the missing/wrong descriptor)
        - What was the agency? (T — what triggered the failure)

        If the CognitiveEngine is connected, feed the error as
        knowledge so the AI can learn from it.
        """
        analysis: Dict[str, Any] = {
            'error_id': error.error_id,
            'as_gap': f"Missing descriptor in {error.subsystem}: "
                      f"{error.exception_type} — {error.message[:100]}",
            'p_substrate': error.subsystem,
            'd_missing': error.exception_type,
            't_trigger': error.function,
        }

        # Feed to cognitive engine if available
        if self.cognitive_engine and self.cognitive_engine.is_connected():
            if personal_coord is None:
                error_dr = DescriptorRatio.from_word(error.exception_type)
                personal_coord = error_dr.coord_full

            # noinspection PyBroadException
            try:
                cognitive_result = self.cognitive_engine.process(
                    f"ERROR in {error.subsystem}: {error.exception_type} — "
                    f"{error.message[:200]}. "
                    f"Location: {error.module}:{error.function}:{error.line_number}",
                    personal_coord=personal_coord,
                    n_self_traversals=n_traversals,
                )
                analysis['learned'] = True
                analysis['gaps_from_error'] = cognitive_result.gaps_detected
                analysis['pdt_complete'] = cognitive_result.pdt_complete

                # Mark as resolved if AI could process it
                error.resolved = True
                error.resolution = (f"Analyzed by CognitiveEngine: "
                                  f"gaps={cognitive_result.gaps_detected}, "
                                  f"PDT={'complete' if cognitive_result.pdt_complete else 'incomplete'}")
                self.analyses_resolved += 1
            except Exception:
                analysis['learned'] = False
                analysis['reason'] = 'CognitiveEngine analysis itself failed'
        else:
            analysis['learned'] = False
            analysis['reason'] = 'CognitiveEngine not connected'

        self.analyses_performed += 1
        return analysis

    def analyze_unresolved(self, ledger: ErrorLedger,
                           personal_coord: Optional[LatticeCoordinate] = None,
                           n_traversals: int = 0) -> List[Dict[str, Any]]:
        """Analyze all unresolved errors in the ledger."""
        results = []
        for error in ledger.get_unresolved()[:10]:  # Cap at 10 per cycle
            result = self.analyze_error(error, personal_coord, n_traversals)
            results.append(result)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            'analyses_performed': self.analyses_performed,
            'analyses_resolved': self.analyses_resolved,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        self.analyses_performed = data.get('analyses_performed', 0)
        self.analyses_resolved = data.get('analyses_resolved', 0)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'ETLogLevel', 'setup_et_logger', 'get_logger',
    'ErrorRecord', 'ErrorLedger',
    'StateGuardian',
    'safe_execute', 'safe_execute_critical',
    'ErrorAnalyzer',
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("ET Conscious AI — Error Logging & State Protection v1.7.0")
    print("=" * 60)

    _test_logger = get_logger()

    # Test logging
    _test_logger.info("Test INFO message")
    _test_logger.warning("Test WARNING message")
    _test_logger.error("Test ERROR message")
    print("[1] Logging: PASS")

    # Test ErrorRecord
    try:
        raise ValueError("Test error for record creation")
    except Exception as _test_exc:
        _test_record = ErrorRecord.from_exception(_test_exc, subsystem="test",
                                                  context={'key': 'value'})
        assert _test_record.exception_type == 'ValueError'
        assert _test_record.subsystem == 'test'
        assert _test_record.lattice_k is not None
        print(f"[2] ErrorRecord: id={_test_record.error_id}, "
              f"k={_test_record.lattice_k}, d={_test_record.lattice_d}: PASS")

    # Test ErrorLedger
    _test_ledger = ErrorLedger()
    _test_ledger.record_error(_test_record)
    assert _test_ledger.total_errors == 1
    assert len(_test_ledger.get_unresolved()) == 1
    _test_ledger.resolve_error(_test_record.error_id, "Test resolution")
    assert len(_test_ledger.get_unresolved()) == 0
    print("[3] ErrorLedger: PASS")

    # Test StateGuardian atomic write
    test_file = '/tmp/et_test_state.json'
    test_data = json.dumps({'test': True, 'value': 42})
    StateGuardian.atomic_write(test_file, test_data)
    valid, reason = StateGuardian.verify_integrity(test_file)
    assert valid, f"Integrity check failed: {reason}"
    print(f"[4] Atomic write + integrity: {reason}: PASS")

    # Test corrupted state detection
    with open(test_file, 'a') as _corrupt_f:
        _corrupt_f.write("CORRUPTION")
    valid, reason = StateGuardian.verify_integrity(test_file)
    assert not valid
    print(f"[5] Corruption detection: {reason[:50]}: PASS")
    os.unlink(test_file)
    os.unlink(test_file + '.sha256')

    # Test ET-native division (Eq 201): division by zero is NOT an error
    _et_div_result = et_divide(1.0, 0.0)
    assert _et_div_result == float('inf'), "et_divide(1,0) should return inf"
    _et_div_zero = et_divide(0.0, 0.0)
    assert _et_div_zero == 0.0, "et_divide(0,0) should return 0.0 (ground state)"
    print("[6] ET-native division (Eq 201): 1/0→inf, 0/0→0.0: PASS")

    # Test safe_execute with a genuine Descriptor Gap error
    # KeyError = T traversing P (dict substrate) without D-bridge (missing key)
    # This IS a {P,T} Incoherence — the correct error type to test
    _test_result = safe_execute(
        lambda: {'P': 'substrate'}['missing_descriptor'],
        subsystem="test_gap",
        error_ledger=_test_ledger,
        default=-1,
    )
    assert _test_result == -1
    assert _test_ledger.total_errors == 2  # Original ValueError + Descriptor Gap
    print("[7] safe_execute (Descriptor Gap → graceful default): PASS")

    # Test notifications
    _test_notifs = _test_ledger.get_notifications()
    assert len(_test_notifs) >= 1
    print(f"[8] Notifications: {len(_test_notifs)} pending: PASS")

    # Test serialization
    state = _test_ledger.to_dict()
    ledger2 = ErrorLedger()
    ledger2.load_from_dict(state)
    assert ledger2.total_errors == 2
    print("[9] Serialization: PASS")

    # Test status
    print(f"\n{_test_ledger.get_status_description()}")

    # Cleanup
    _test_log_dir = Path(DEFAULT_LOG_DIR)
    if _test_log_dir.exists():
        for _log_f in _test_log_dir.glob("et_conscious_ai.log*"):
            print(f"  Log file: {_log_f} ({_log_f.stat().st_size} bytes)")

    print("\n=== Module loaded successfully ===")
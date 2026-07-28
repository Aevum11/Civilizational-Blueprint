"""
et_bridge/et_errors.py
ET32 Bridge — Complete Error Handling System

Derived from P ∘ D ∘ T = E.

ET derivation of error taxonomy:
  Every error is a failure to complete P ∘ D ∘ T = E.
  The variance V(error) measures how far from grounded the failure is:

    V = 0.0            → No error — E state (grounded, complete)
    V = V_BASE = 1/12  → Minor error — M state (mediated, recoverable)
    V = K = 2/3        → Warning error — ∂I state (near incoherence)
    V > K              → Critical error — I state (incoherent, irrecoverable)

  These map to ETErrorSeverity.TRACE / MEDIATION / BOUNDARY / INCOHERENT.

Every ETOperationError carries:
  - module, function, line  (exact source location — from inspect + traceback)
  - operation               (human-readable description of what was attempted)
  - et_pid                  (source_pid from ETPacket — which process)
  - et_family               (CmdFamily lattice position 1..12)
  - et_code                 (CmdCode within family)
  - os_error                (GetLastError() / errno at point of failure)
  - os_error_message        (FormatMessageW — Windows text for the error code)
  - severity                (ETErrorSeverity — maps to ET manifold state)
  - cause                   (chained ETOperationError or base exception)
  - timestamp               (monotonic time of error)
  - variance                (V(error) derived from severity)
  - context_vars            (arbitrary key-value pairs for extra context)

Design principles:
  - Zero silent failures: every failure point raises or logs with full context
  - No swallowed exceptions: `except Exception: pass` is forbidden
  - Exact location: file:function:line always captured via traceback
  - OS-level detail: GetLastError() captured before it can be clobbered
  - Chain preservation: cause exceptions are always chained
  - ET state: every error knows which lattice operation it came from

Author: Derived from Michael James Muller's Exception Theory
"""

import ctypes
import ctypes.wintypes as wintypes
import enum
import inspect
import os
import sys
import time
import traceback
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from et_math import V_BASE, K, S, QUEUE_DEPTH
from et_math import (
    tightness, coherence_depth, incoherence_filter,
    n_max_cascade, et_variance, COHERENCE_N_MAX,
)

# =============================================================================
# ET ERROR SEVERITY — MAPS TO ET MANIFOLD STATES
# =============================================================================

class ETErrorSeverity(enum.IntEnum):
    """
    Error severity derived from ET variance V:

    TRACE       V = 0.0        Informational — no error, but worth recording
    MEDIATION   V = V_BASE     Recoverable — system continues with degraded state
    BOUNDARY    V = K          Near-incoherence — requires attention, may recover
    INCOHERENT  V > K          System incoherent — cannot continue this operation
    FATAL       V = 1.0        System-level failure — broker must shut down

    ET derivation: V = fraction of K exceeded:
      MEDIATION:  V/K = V_BASE/K = (1/12)/(2/3) = 1/8  — within stable zone
      BOUNDARY:   V/K = 1.0                              — at the incoherence wall
      INCOHERENT: V/K > 1.0                              — beyond the wall
    """
    TRACE       = 0   # V = 0.0
    MEDIATION   = 1   # V = V_BASE = 1/12
    BOUNDARY    = 2   # V = K = 2/3
    INCOHERENT  = 3   # V > K
    FATAL       = 4   # V = 1.0 (system must stop)

    @property
    def variance(self) -> float:
        """ET variance V for this severity level, derived from the manifold state map.

        TRACE=0.0 (E state), MEDIATION=V_BASE=1/12, BOUNDARY=K=2/3,
        INCOHERENT=K+V_BASE=3/4, FATAL=1.0 (full incoherence).
        """
        return {
            self.TRACE:      0.0,
            self.MEDIATION:  V_BASE,
            self.BOUNDARY:   K,
            self.INCOHERENT: K + V_BASE,
            self.FATAL:      1.0,
        }[self]

    @property
    def python_log_level(self) -> int:
        """Map to Python logging levels."""
        import logging
        return {
            self.TRACE:      logging.DEBUG,
            self.MEDIATION:  logging.WARNING,
            self.BOUNDARY:   logging.ERROR,
            self.INCOHERENT: logging.CRITICAL,
            self.FATAL:      logging.CRITICAL,
        }[self]


# =============================================================================
# WINDOWS ERROR CODE RESOLUTION
# =============================================================================

def _format_win32_error(error_code: int) -> str:
    """
    Convert a Windows error code to a human-readable string via FormatMessageW.
    No static table — uses the OS itself as the authoritative source.

    Dynamic DLL access via getattr() ensures PyCharm resolution and
    handles non-Windows platforms gracefully.
    argtypes/restype set from ctypes.wintypes for correct 64-bit marshaling.
    """
    if error_code == 0:
        return "ERROR_SUCCESS (0)"
    try:
        # Dynamic DLL access — getattr() for windll resolution (no static _DLLT)
        _kernel32 = getattr(ctypes.windll, 'kernel32')
        _FormatMessageW = getattr(_kernel32, 'FormatMessageW')

        # Set proper argtypes/restype via wintypes for correct marshaling
        _FormatMessageW.argtypes = [
            wintypes.DWORD,     # dwFlags
            ctypes.c_void_p,    # lpSource (LPCVOID)
            wintypes.DWORD,     # dwMessageId
            wintypes.DWORD,     # dwLanguageId
            ctypes.c_wchar_p,   # lpBuffer
            wintypes.DWORD,     # nSize
            ctypes.c_void_p,    # Arguments (va_list*)
        ]
        _FormatMessageW.restype = wintypes.DWORD

        buf = ctypes.create_unicode_buffer(512)
        result = _FormatMessageW(
            0x00001000,  # FORMAT_MESSAGE_FROM_SYSTEM
            None,
            error_code,
            0,
            buf,
            512,
            None
        )
        if result > 0:
            msg = buf.value.strip().rstrip(".")
            return f"{msg} (0x{error_code:08X})"
    except (OSError, ValueError, ctypes.ArgumentError, AttributeError):
        # FormatMessageW itself failed — this is the error formatter's own fallback.
        # OSError: ctypes call failure; ValueError: buffer issue;
        # ctypes.ArgumentError: wrong types; AttributeError: no windll on non-Windows.
        # We cannot log here (would cause infinite recursion).
        # Fall through to return the numeric code as a string.
        pass
    return f"Unknown error (0x{error_code:08X})"


def _format_ntstatus(ntstatus: int) -> str:
    """Format an NTSTATUS code."""
    if ntstatus == 0:
        return "STATUS_SUCCESS (0x00000000)"
    # Severity bits: [31:30] — 00=success, 01=info, 10=warning, 11=error
    severity = (ntstatus >> 30) & 0x3
    sev_names = {0: "SUCCESS", 1: "INFORMATIONAL", 2: "WARNING", 3: "ERROR"}
    return f"NTSTATUS 0x{ntstatus:08X} ({sev_names.get(severity, 'UNKNOWN')})"


# =============================================================================
# LOCATION CAPTURE
# =============================================================================

@dataclass
class ErrorLocation:
    """Exact source location of an error."""
    module:   str  # filename, shortened
    function: str  # function name
    line:     int  # line number
    qualname: str = field(default="")  # qualified name if available (e.g. ClassName.method_name)

    def __str__(self) -> str:
        short_mod = os.path.basename(self.module).replace(".py", "")
        return f"{short_mod}.{self.qualname or self.function}:{self.line}"

    @classmethod
    def capture(cls, depth: int = 2) -> "ErrorLocation":
        """Capture location at `depth` frames up the call stack."""
        try:
            frame = sys._getframe(depth)
            code  = frame.f_code
            # Try to get qualname from the frame's local `self` or `cls`
            qualname = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
            return cls(
                module   = code.co_filename,
                function = code.co_name,
                line     = frame.f_lineno,
                qualname = qualname,
            )
        except (AttributeError, ValueError):
            return cls("unknown", "unknown", 0, "")


# =============================================================================
# ET OPERATION ERROR — THE CENTRAL ERROR CLASS
# =============================================================================

class ETOperationError(Exception):
    """
    Complete error record for any failure in the ET32 Bridge.

    PDT of this error object:
      P = the failed operation (what we were trying to do)
      D = the error context (all diagnostic information)
      T = this exception object (the traverser carrying the error upward)
      E = precise diagnosis of what failed and why

    Every ETOperationError carries enough information to:
      1. Find the exact source line that failed
      2. Know which 32-bit process was being served
      3. Know which ET lattice operation was in progress
      4. Know the Windows OS-level error code with human-readable text
      5. Reproduce the failure scenario for debugging
    """

    def __init__(
        self,
        operation:    str,
        *,
        severity:     ETErrorSeverity = ETErrorSeverity.BOUNDARY,
        et_pid:       int = 0,
        et_family:    int = 0,
        et_code:      int = 0,
        os_error:     int = 0,
        ntstatus:     int = 0,
        cause:        Optional[BaseException] = None,
        location:     Optional[ErrorLocation] = None,
        depth:        int = 2,
        **context_vars: Any,
    ):
        self.operation        = operation
        self.severity         = severity
        self.et_pid           = et_pid
        self.et_family        = et_family
        self.et_code          = et_code
        self.os_error         = os_error
        self.ntstatus         = ntstatus
        self.cause            = cause
        self.timestamp        = time.monotonic()
        self.context_vars     = context_vars
        self.location         = location or ErrorLocation.capture(depth + 1)

        # Resolve OS error message immediately (before GetLastError() is clobbered)
        self.os_error_message = _format_win32_error(os_error) if os_error else ""
        self.ntstatus_message = _format_ntstatus(ntstatus)    if ntstatus else ""

        # Full traceback of the cause
        self.cause_traceback: str = ""
        if cause is not None:
            self.cause_traceback = "".join(
                traceback.format_exception(type(cause), cause, cause.__traceback__)
            )

        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Build the complete error message with all diagnostic context."""
        parts = [f"[ET32 ERROR] {self.operation}"]

        # ET state
        if self.et_pid or self.et_family or self.et_code:
            parts.append(
                f"  ET state:  PID={self.et_pid}  family=d{self.et_family}"
                f"  code=0x{self.et_code:02X}"
            )

        # Source location
        parts.append(f"  Location:  {self.location}")

        # Severity / variance
        parts.append(
            f"  Severity:  {self.severity.name}  V={self.severity.variance:.4f}"
        )

        # OS error
        if self.os_error:
            parts.append(f"  OS error:  {self.os_error_message}")

        # NTSTATUS
        if self.ntstatus:
            parts.append(f"  NTSTATUS:  {self.ntstatus_message}")

        # Extra context
        for k, v in self.context_vars.items():
            parts.append(f"  {k}: {v!r}")

        # Cause chain
        if self.cause is not None:
            cause_type = type(self.cause).__name__
            cause_msg  = str(self.cause)[:200]
            parts.append(f"  Caused by: {cause_type}: {cause_msg}")

        return "\n".join(parts)

    @property
    def variance(self) -> float:
        """ET variance V of this error, delegated from its severity level."""
        return self.severity.variance

    @property
    def et_state(self) -> Tuple[int, int, int]:
        """ET lattice state triple (pid, family, code) for this error.

        Returns the PDT-derived classification:
          pid    — which 32-bit process (P substrate)
          family — which of the S=12 command families (D lattice position)
          code   — specific operation code within family (T traversal point)
        """
        return self.et_pid, self.et_family, self.et_code

    def is_recoverable(self) -> bool:
        """Whether this error is below the incoherence threshold (V < K + V_BASE).

        Returns True for TRACE, MEDIATION, and BOUNDARY severities.
        Returns False for INCOHERENT and FATAL (system cannot continue this operation).
        """
        return self.severity < ETErrorSeverity.INCOHERENT

    def chain(self, outer_operation: str,
              severity: Union[ETErrorSeverity, None] = None,
              **kwargs) -> "ETOperationError":
        """
        Wrap this error in a higher-level context.
        Preserves the full cause chain for precise tracing.
        """
        return ETOperationError(
            outer_operation,
            severity   = severity or self.severity,
            et_pid     = kwargs.pop("et_pid",    self.et_pid),
            et_family  = kwargs.pop("et_family", self.et_family),
            et_code    = kwargs.pop("et_code",   self.et_code),
            os_error   = kwargs.pop("os_error",  self.os_error),
            cause      = self,
            depth      = 2,
            **kwargs,
        )

    @property
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict for JSON logging / IPC error reports."""
        return {
            "operation":     self.operation,
            "severity":      self.severity.name,
            "variance":      self.severity.variance,
            "et_pid":        self.et_pid,
            "et_family":     self.et_family,
            "et_code":       f"0x{self.et_code:02X}",
            "os_error":      self.os_error,
            "os_error_msg":  self.os_error_message,
            "ntstatus":      f"0x{self.ntstatus:08X}" if self.ntstatus else None,
            "location":      str(self.location),
            "timestamp":     self.timestamp,
            "context":       {k: repr(v) for k, v in self.context_vars.items()},
            "cause":         str(self.cause) if self.cause else None,
        }


# =============================================================================
# SPECIFIC ERROR SUBCLASSES — each pinpoints a category of failure
# =============================================================================

class ETWindowsAPIError(ETOperationError):
    """A Windows API call returned FALSE or NULL. Captures GetLastError()."""
    def __init__(self, api_name: str, **kwargs):
        os_err = kwargs.pop("os_error", ctypes.GetLastError())
        super().__init__(
            f"Windows API failed: {api_name}",
            os_error  = os_err,
            severity  = kwargs.pop("severity", ETErrorSeverity.BOUNDARY),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETInjectionError(ETOperationError):
    """Error during DLL or hook injection into a target process."""
    def __init__(self, operation: str, pid: int, **kwargs):
        super().__init__(
            f"Injection error in PID {pid}: {operation}",
            et_pid    = pid,
            severity  = kwargs.pop("severity", ETErrorSeverity.BOUNDARY),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETIPCError(ETOperationError):
    """Error on the IPC named pipe (read, write, connect, timeout)."""
    def __init__(self, operation: str, pid: int = 0, **kwargs):
        super().__init__(
            f"IPC error (PID {pid}): {operation}",
            et_pid    = pid,
            severity  = kwargs.pop("severity", ETErrorSeverity.MEDIATION),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETPacketError(ETOperationError):
    """ETPacket serialisation, deserialisation, or checksum failure."""
    def __init__(self, operation: str, **kwargs):
        super().__init__(
            f"Packet error: {operation}",
            severity  = kwargs.pop("severity", ETErrorSeverity.MEDIATION),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETAWEError(ETOperationError):
    """AWE (Address Windowing Extensions) bookshelf operation failure."""
    def __init__(self, operation: str, pid: int = 0, **kwargs):
        super().__init__(
            f"AWE error (PID {pid}): {operation}",
            et_pid    = pid,
            severity  = kwargs.pop("severity", ETErrorSeverity.BOUNDARY),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETHookError(ETOperationError):
    """KiFastSystemCall or IAT hook installation/removal failure."""
    def __init__(self, operation: str, pid: int = 0, **kwargs):
        super().__init__(
            f"Hook error (PID {pid}): {operation}",
            et_pid    = pid,
            severity  = kwargs.pop("severity", ETErrorSeverity.BOUNDARY),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETDispatchError(ETOperationError):
    """Error dispatching a 64-bit operation in et_host64."""
    def __init__(self, operation: str, et_pid: int = 0,
                 et_family: int = 0, et_code: int = 0, **kwargs):
        super().__init__(
            f"Dispatch error (d={et_family} code=0x{et_code:02X}): {operation}",
            et_pid    = et_pid,
            et_family = et_family,
            et_code   = et_code,
            severity  = kwargs.pop("severity", ETErrorSeverity.MEDIATION),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETConfigError(ETOperationError):
    """Configuration parsing or validation error."""
    def __init__(self, operation: str, **kwargs):
        super().__init__(
            f"Config error: {operation}",
            severity  = kwargs.pop("severity", ETErrorSeverity.BOUNDARY),
            depth     = kwargs.pop("depth", 3),
            **kwargs,
        )


class ETHandleError(ETOperationError):
    """Handle table allocation, resolution, or translation failure."""
    def __init__(self, operation: str, handle: int = 0, **kwargs):
        super().__init__(
            f"Handle error (handle=0x{handle:08X}): {operation}",
            severity  = kwargs.pop("severity", ETErrorSeverity.MEDIATION),
            depth     = kwargs.pop("depth", 3),
            handle    = f"0x{handle:08X}",
            **kwargs,
        )


# =============================================================================
# WIN32 CHECK UTILITIES — call these after every Windows API call
# =============================================================================

def win32_check(
    result:    Any,
    api_name:  str,
    *,
    et_pid:    int = 0,
    et_family: int = 0,
    et_code:   int = 0,
    severity:  ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    **context,
) -> Any:
    """
    Check a Windows API BOOL/HANDLE result.
    Captures GetLastError() immediately before anything else can clobber it.
    Raises ETWindowsAPIError with full context on failure.
    Returns result unchanged on success.

    Usage::

        ptr = win32_check(
            kernel32.VirtualAllocEx(h, None, size, MEM_COMMIT, PAGE_RW),
            "VirtualAllocEx for hook stub",
            et_pid=pid, size=size
        )
    """
    # Capture error code NOW before any other call
    if not result:
        os_err = ctypes.GetLastError()
        raise ETWindowsAPIError(
            api_name,
            os_error  = os_err,
            et_pid    = et_pid,
            et_family = et_family,
            et_code   = et_code,
            severity  = severity,
            depth     = 3,
            **context,
        )
    return result


def win32_check_handle(
    handle:    int,
    api_name:  str,
    *,
    invalid:   int = 0,
    et_pid:    int = 0,
    severity:  ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    **context,
) -> int:
    """
    Check a Windows HANDLE result.
    invalid: the value that indicates failure (default 0; use -1 for INVALID_HANDLE_VALUE).
    """
    if handle == invalid or handle is None:
        os_err = ctypes.GetLastError()
        raise ETWindowsAPIError(
            api_name,
            os_error = os_err,
            et_pid   = et_pid,
            severity = severity,
            depth    = 3,
            **context,
        )
    return handle


def ntstatus_check(
    status:    int,
    operation: str,
    *,
    et_pid:    int = 0,
    et_family: int = 0,
    et_code:   int = 0,
    severity:  ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    **context,
) -> int:
    """
    Check an NTSTATUS value. Raises ETOperationError if status indicates failure.
    NTSTATUS severity bits [31:30]: 11 = error, 10 = warning, 01 = info, 00 = success.
    """
    status_unsigned = status & 0xFFFFFFFF
    sev_bits = (status_unsigned >> 30) & 0x3
    if sev_bits >= 2:  # warning or error
        # sev_bits == 3 (error): always INCOHERENT — beyond ∂I
        # sev_bits == 2 (warning): use caller-supplied severity (default BOUNDARY)
        effective_severity = ETErrorSeverity.INCOHERENT if sev_bits == 3 else severity
        raise ETOperationError(
            operation,
            severity  = effective_severity,
            et_pid    = et_pid,
            et_family = et_family,
            et_code   = et_code,
            ntstatus  = status_unsigned,
            depth     = 3,
            **context,
        )
    return status


# =============================================================================
# CONTEXT MANAGER — et_context()
# =============================================================================

@contextmanager
def et_context(
    operation:  str,
    *,
    et_pid:     int = 0,
    et_family:  int = 0,
    et_code:    int = 0,
    severity:   ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    reraise:    bool = True,
    log_fn:     Union[Callable[[str], None], None] = None,
    **context_vars,
):
    """
    Context manager that wraps a block with complete error context.

    Any exception raised inside is caught, wrapped in ETOperationError with
    full context, logged (if log_fn provided), then re-raised or suppressed.

    Usage::

        with et_context("allocating hook stubs", et_pid=pid, size=size):
            stub_va = win32_check(
                VirtualAllocEx(h, None, size, MEM_COMMIT, PAGE_RW),
                "VirtualAllocEx"
            )

    On error: logs precise location, ET state, OS error, and re-raises.
    """
    try:
        yield
    except ETOperationError:
        raise  # already has full context, pass through
    except Exception as exc:
        os_err = ctypes.GetLastError()
        wrapped = ETOperationError(
            operation,
            severity   = severity,
            et_pid     = et_pid,
            et_family  = et_family,
            et_code    = et_code,
            os_error   = os_err,
            cause      = exc,
            depth      = 3,
            **context_vars,
        )
        if log_fn is not None:
            log_fn(str(wrapped))
        if reraise:
            raise wrapped from exc


# =============================================================================
# DECORATOR — @et_traced
# =============================================================================

def et_traced(
    severity:   ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    reraise:    bool = True,
    log_fn:     Union[Callable[[str], None], None] = None,
):
    """
    Decorator that adds complete error tracing to any function.

    Wraps every unhandled exception in ETOperationError with:
      - function name and module as operation description
      - full traceback of the original exception
      - exact file:function:line (via inspect + traceback)
      - call arguments (if they are simple types)

    Usage::

        @et_traced(severity=ETErrorSeverity.INCOHERENT, reraise=False)
        def _inject_dll(self, pid, h_process):
            ...
    """
    import functools

    def decorator(fn: Callable) -> Callable:
        """ET tracing decorator — wraps fn with full PDT error context."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """ET-traced wrapper — captures exceptions with inspect-derived location."""
            try:
                return fn(*args, **kwargs)
            except ETOperationError:
                raise
            except Exception as exc:
                os_err = ctypes.GetLastError()
                qualname = getattr(fn, "__qualname__", fn.__name__)
                # Use inspect for robust module/file resolution (docstring contract)
                try:
                    module = inspect.getfile(fn)
                except (TypeError, OSError):
                    # Built-in or dynamically created functions have no file
                    module = getattr(
                        inspect.getmodule(fn), "__file__",
                        getattr(fn, '__module__', '?')
                    )
                loc = ErrorLocation(
                    module   = module or "?",
                    function = fn.__name__,
                    line     = exc.__traceback__.tb_lineno if exc.__traceback__ else 0,
                    qualname = qualname,
                )
                wrapped = ETOperationError(
                    f"{qualname}() raised {type(exc).__name__}",
                    severity = severity,
                    os_error = os_err,
                    cause    = exc,
                    location = loc,
                    depth    = 2,
                )
                if log_fn is not None:
                    log_fn(str(wrapped))
                if reraise:
                    raise wrapped from exc
                return None
        return wrapper
    return decorator


# =============================================================================
# CENTRALIZED ERROR REGISTRY
# =============================================================================

class ETErrorRegistry:
    """
    Centralised error log — aggregates all ETOperationErrors across the bridge.

    ET derivation:
      P = the set of all errors that have occurred
      D = their classification (severity, ET state, location)
      T = this registry (traverses errors to produce diagnostics)
      E = a complete picture of system health — V(registry) measures overall health

    V(registry) = weighted sum of error variances / total operations:
      V → 0 when few/low-severity errors
      V → K when many BOUNDARY errors
      V → 1 when INCOHERENT errors present
    """

    def __init__(self, max_history: int = QUEUE_DEPTH * S) -> None:
        self._errors: List[ETOperationError] = []
        self._lock   = threading.Lock()
        self._max    = max_history  # S² × S = 1728 entries max
        self._total_operations: int = 0

    def record(self, error: ETOperationError) -> None:
        """Record an error. Evicts oldest if at capacity (LRU)."""
        with self._lock:
            if len(self._errors) >= self._max:
                self._errors.pop(0)
            self._errors.append(error)

    def increment_ops(self, n: int = 1) -> None:
        """Increment total operation count (for V calculation)."""
        with self._lock:
            self._total_operations += n

    def variance(self) -> float:
        """
        V(registry) — overall system health measured as ET variance.
        V = sum(error.variance) / max(1, total_operations).
        Clipped to [0.0, 1.0].
        """
        with self._lock:
            if not self._errors:
                return 0.0
            total_v = sum(e.variance for e in self._errors)
            ops     = max(1, self._total_operations)
            return min(1.0, total_v / ops)

    def coherent_variance(self) -> float:
        """
        Level 5 𝒜_I — Coherent Summation (incoherence_filter_-_lattice.txt §Level 5).

        Σ_physical = Σ_{r ∈ C_coherent} f(r)  where  C_coherent = {r : 𝒜_I(r) = 0}

        For the error registry: instead of summing ALL error variances
        (which inflates the health metric by including incoherent entries),
        sum ONLY errors whose tightness > K (𝒜_I = 0).

        An error e is coherent if tightness(e.variance × 50¢) > K.
        Incoherent errors (INCOHERENT/FATAL severity) are excluded from
        the coherent sum — they are the ∂I slice, not the physical sum.

        Returns the coherent variance as a fraction of V_theory.
        """
        with self._lock:
            if not self._errors:
                return 0.0
            # Map each error's variance to epsilon_cents: v ∈ [0,1] → ε ∈ [0,50¢]
            coherent_v = 0.0
            for e in self._errors:
                eps = e.variance * 50.0            # scale to cents
                if incoherence_filter(eps) == 0:   # 𝒜_I = 0: coherent
                    coherent_v += e.variance
            ops = max(1, self._total_operations)
            return min(1.0, coherent_v / ops)

    def ai_tightness(self) -> float:
        """
        Tightness of the error registry as a whole.

        Maps V_normalized → ε → tightness:
          ε = V_normalized × 50¢  (maps [0,1] health to the [0,50¢] cent scale)
          tightness = 100/(100+|ε|)
          At V_normalized=0: tightness=1.0 (perfect lattice point, maximally coherent)
          At V_normalized=1: tightness=100/150=2/3=K (∂I boundary, minimally coherent)

        𝒜_I = 0 (coherent)   iff tightness > K
        𝒜_I = 1 (incoherent) iff tightness ≤ K
        """
        eps = self._registry_epsilon()
        return tightness(eps)

    def ai_state(self) -> int:
        """
        Global 𝒜_I state of the error registry: 0 = coherent, 1 = incoherent.
        𝒜_I = 1 iff tightness ≤ K (system is at or past ∂I).
        """
        t = self.ai_tightness()
        return 0 if t > float(K) else 1

    def _registry_epsilon(self) -> float:
        """Compute the registry's ε in cents: V_normalized × 50¢.

        Shared computation for ai_tightness, ai_coherence_depth, and cascade_n_max.
        Maps the normalized variance [0, 1] → [0, 50¢] cent scale.
        """
        v_theory = et_variance(S)
        v_actual = self.variance()
        v_norm   = v_actual / v_theory if v_theory > 0 else 0.0
        return v_norm * 50.0

    def ai_coherence_depth(self) -> float:
        """
        Coherence depth Δ∂I of the error registry.

        Uses the ET coherence_depth function:
          Δ∂I = tightness(ε) − K
          = 1/3 at perfect lattice point (no errors)
          = 0   at ∂I boundary (system at incoherence wall)
          < 0   beyond ∂I (system incoherent)

        This measures how much margin remains before the system crosses ∂I.
        """
        eps = self._registry_epsilon()
        return coherence_depth(eps)

    def cascade_n_max(self) -> int:
        """
        Level 4 — Dynamic cascade coherence horizon for the error registry.

        Uses the ET n_max_cascade function:
          N_max = ⌊50¢/|δ|⌋

        where δ is the average per-error deviation in cents.
        Computes the maximum number of consecutive errors the system can
        sustain before the accumulated deviation crosses ∂I.

        When N_max > total_errors: system is within coherence horizon.
        When N_max ≤ total_errors: system has exceeded cascade limit.
        """
        with self._lock:
            if not self._errors:
                return n_max_cascade(0.0)  # no errors: infinite horizon
            # Average error variance → map to ε in cents
            avg_v = sum(e.variance for e in self._errors) / len(self._errors)
            v_theory = et_variance(S)
            # Each error contributes avg_v / v_theory fraction → δ in cents
            delta_cents = (avg_v / v_theory * 50.0) if v_theory > 0 else 0.0
            return n_max_cascade(delta_cents)

    def recent(self, n: int = S) -> List[ETOperationError]:
        """Return the n most recent errors."""
        with self._lock:
            return list(self._errors[-n:])

    @property
    def error_count(self) -> int:
        """Total number of errors currently in the registry."""
        with self._lock:
            return len(self._errors)

    def severity_summary(self) -> Dict[str, int]:
        """
        Return a dict of {severity_name: count} for all error severities.
        Dynamically enumerates ETErrorSeverity — no static list.
        """
        with self._lock:
            counts = {s.name: 0 for s in ETErrorSeverity}
            for e in self._errors:
                counts[e.severity.name] = counts.get(e.severity.name, 0) + 1
            return counts

    def by_severity(
        self, severity: ETErrorSeverity
    ) -> List[ETOperationError]:
        """Return all errors matching a specific ET severity level (manifold state filter)."""
        with self._lock:
            return [e for e in self._errors if e.severity == severity]

    def by_pid(self, pid: int) -> List[ETOperationError]:
        """Return all errors from a specific 32-bit process (P-substrate filter)."""
        with self._lock:
            return [e for e in self._errors if e.et_pid == pid]

    def clear_pid(self, pid: int) -> int:
        """Remove all errors for a process that has exited."""
        with self._lock:
            before = len(self._errors)
            self._errors = [e for e in self._errors if e.et_pid != pid]
            return before - len(self._errors)

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary dict for logging.

        Includes theoretical variance baseline from Measure Theory §4:
          V_theory = et_variance(S) = (S²-1)/S = (144-1)/12 = 143/12 ≈ 11.917
          This is the MAXIMUM possible variance for the full 12-family manifold.
          V_normalized = V_actual / V_theory is the normalized health metric [0, 1].
          V_normalized → 0: healthy (few/minor errors)
          V_normalized → 1: incoherent (approaching theoretical maximum)
        """
        with self._lock:
            counts = {s.name: 0 for s in ETErrorSeverity}
            for e in self._errors:
                counts[e.severity.name] = counts.get(e.severity.name, 0) + 1
            v_actual = self.variance()
            # Theoretical maximum: V(S) = (S²-1)/S from ET Measure Theory
            # et_variance and S already imported at module top level
            v_theory = et_variance(S)  # = (144-1)/12 ≈ 11.917
            v_normalized = v_actual / v_theory if v_theory > 0 else 0.0
            return {
                "total_errors":  len(self._errors),
                "total_ops":     self._total_operations,
                "variance":      round(v_actual,     4),
                "variance_theory": round(v_theory,   4),
                "variance_normalized": round(v_normalized, 4),
                "health":        "STABLE" if v_normalized < 0.05 else (
                                 "DEGRADED" if v_normalized < float(2/3) else "INCOHERENT"),
                # Level 5 𝒜_I tightness-based coherence metrics
                "ai_state":         self.ai_state(),
                "ai_tightness":     round(self.ai_tightness(), 4),
                "coherence_depth":  round(self.ai_coherence_depth(), 4),
                "coherent_variance":round(self.coherent_variance(), 4),
                "coherence_n_max":  COHERENCE_N_MAX,
                # Level 4 — dynamic cascade coherence horizon
                "cascade_n_max":    self.cascade_n_max(),
                "by_severity":   counts,
                "most_recent":   str(self._errors[-1]) if self._errors else None,
            }

    def format_report(self) -> str:
        """Format a human-readable diagnostic report."""
        s = self.summary()
        lines = [
            "=" * 72,
            "ET32 Bridge — Error Registry Report",
            f"  V(system) = {s['variance']:.4f}  "
            f"({'STABLE' if s['variance'] < K else 'NEAR-INCOHERENT' if s['variance'] < 1 else 'INCOHERENT'})",
            f"  𝒜_I state  = {s.get('ai_state', '?')}" f" (tightness={s.get('ai_tightness', 0):.4f}, depth={s.get('coherence_depth', 0):.4f})",
            f"  Coherent V = {s.get('coherent_variance', 0):.4f}" f"  (N_max={s.get('coherence_n_max', 25)}, cascade={s.get('cascade_n_max', '?')})",
            f"  Total errors:     {s['total_errors']}",
            f"  Total operations: {s['total_ops']}",
            "  By severity:",
        ]
        for sev, count in s["by_severity"].items():
            if count:
                lines.append(f"    {sev:12s}: {count}")
        if s["most_recent"]:
            lines.append("  Most recent error:")
            for ln in s["most_recent"].split("\n"):
                lines.append(f"    {ln}")
        lines.append("=" * 72)
        return "\n".join(lines)


# =============================================================================
# MODULE-LEVEL SINGLETONS
# =============================================================================

# Lattice classification imports — used by registry lattice analysis functions below
from et_math import family_sublattice, lattice_coords

# Global error registry — shared across all modules
_global_registry: Optional[ETErrorRegistry] = None
_registry_lock   = threading.Lock()


def get_registry() -> ETErrorRegistry:
    """Return the global error registry (created on first call)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = ETErrorRegistry()
    return _global_registry


def record_error(error: ETOperationError) -> None:
    """Record an error in the global registry."""
    get_registry().record(error)


def record_op() -> None:
    """Increment the global operation counter."""
    get_registry().increment_ops()


def classify_error_lattice(error: ETOperationError) -> Tuple[int, int, float]:
    """Classify an error by its ET lattice position using its variance.

    Maps error.variance → lattice coordinates (k, d, ε):
      k = semitone class (lattice position)
      d = sublattice family (1, 2, 3, 4, 6, 12)
      ε = rounding error in cents

    The variance is treated as a ratio: V/V_BASE gives the ratio-space
    position, which lattice_coords maps to (k, d, ε).
    """
    v = error.variance
    ratio = v / V_BASE if V_BASE > 0 and v > 0 else 1.0
    k, d, eps = lattice_coords(ratio)
    return k, d, eps


def error_family_sublattice(error: ETOperationError) -> int:
    """Return the ET sublattice family d for an error's command family.

    Uses family_sublattice(et_family) to map the CmdFamily (1..12) to its
    canonical sublattice d. This determines the resolution/complexity class
    of the operation that failed.
    """
    return family_sublattice(error.et_family) if error.et_family else S


# =============================================================================
# CONVENIENCE: safe_call() — wraps a callable with full error handling
# =============================================================================

def safe_call(
    fn: Callable,
    *args: Any,
    operation:   str = "",
    et_pid:      int = 0,
    et_family:   int = 0,
    et_code:     int = 0,
    severity:    ETErrorSeverity = ETErrorSeverity.BOUNDARY,
    error_cls:   Type[ETOperationError] = ETOperationError,
    default:     Any = None,
    log_fn:      Union[Callable[[str], None], None] = None,
    **kwargs: Any,
) -> Any:
    """
    Call fn(*args, **kwargs) with complete error handling.
    On any exception: logs, records in registry, returns default.
    Never raises. For use in fire-and-forget situations.

    Unlike et_context() which re-raises, safe_call() absorbs and returns default.

    Args:
        fn:         The callable to invoke.
        *args:      Positional arguments forwarded to fn.
        operation:  Human-readable description of what fn does (for error messages).
                    Defaults to fn's qualname if empty.
        et_pid:     Source PID from ETPacket — which 32-bit process this call serves.
        et_family:  CmdFamily lattice position (1..12) of the operation.
        et_code:    CmdCode within the family.
        severity:   ET severity to assign if fn raises a non-ET exception.
        error_cls:  ETOperationError subclass to wrap non-ET exceptions in.
                    Default is ETOperationError itself. Callers can pass e.g.
                    ETIPCError, ETDispatchError for domain-specific wrapping.
        default:    Value to return if fn raises. Defaults to None.
        log_fn:     Optional callback for logging error strings. Called with str(error).
        **kwargs:   Keyword arguments forwarded to fn.
    """
    try:
        record_op()
        return fn(*args, **kwargs)
    except ETOperationError as exc:
        record_error(exc)
        if log_fn:
            log_fn(str(exc))
        return default
    except Exception as exc:
        os_err = ctypes.GetLastError()
        op_name = operation or getattr(fn, "__qualname__", str(fn))
        wrapped = error_cls(
            op_name,
            severity  = severity,
            et_pid    = et_pid,
            et_family = et_family,
            et_code   = et_code,
            os_error  = os_err,
            cause     = exc,
            depth     = 2,
        )
        record_error(wrapped)
        if log_fn:
            log_fn(str(wrapped))
        return default
"""
et_bridge/et_logger.py
ET32 Bridge — Structured Logger with ET Metrics and Error Integration

All log entries carry ET variance metadata.
V(E) = 0 in log entries means the operation completed as an Exception.

Updated to integrate with et_errors.ETErrorRegistry and ETOperationError:
  - Every error-level log also records in the global registry
  - ETLog.error_op() logs an ETOperationError with full formatting
  - ETLog.set_context() attaches persistent PID/family context to log entries
"""

import logging
import sys
import os
import time
import traceback
from typing import Optional, Any, Dict, TYPE_CHECKING
from et_math import V_BASE, K, S

# TYPE_CHECKING-only import: ETOperationError is used as a forward reference
# in error_op() type annotation. Runtime imports happen inside method bodies
# to avoid circular dependency (et_errors imports from et_math, et_logger
# also imports from et_math — no cycle, but defensive).
if TYPE_CHECKING:
    from et_errors import ETOperationError

# =============================================================================
# LOG PATH RESOLUTION — exe-relative by default, config-overridable
# =============================================================================

def resolve_log_path(configured_path: str = None) -> str:
    """
    Resolve the log file path.

    ET derivation:
      P = the filesystem (infinite substrate of possible paths)
      D = {configured_path, exe_dir, cwd} (the Descriptor set constraining location)
      T = this function (the traverser that picks the correct path)
      E = an absolute path where the log file will be written

    Resolution order (first that yields a writable directory wins):
      1. configured_path if absolute → use as-is
      2. configured_path if relative → resolve relative to exe directory
      3. No config → exe_dir / "et32_bridge.log"

    Exe directory detection:
      - PyInstaller frozen: os.path.dirname(sys.executable)
      - Running from source: project root (two levels up from this file)
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller .exe: log alongside the executable
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # Source: project root (et_bridge/ is one level below project root)
        exe_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    if configured_path:
        if os.path.isabs(configured_path):
            return configured_path
        # Relative: resolve from exe_dir
        return os.path.join(exe_dir, configured_path)

    return os.path.join(exe_dir, "et32_bridge.log")


# =============================================================================
# IMMEDIATE-FLUSH FILE HANDLER
# =============================================================================

class ETFlushingFileHandler(logging.FileHandler):
    """
    FileHandler that flushes to disk after EVERY write.

    ET derivation:
      Standard FileHandler buffers writes in OS page cache.
      A crash before flush loses everything buffered.
      flush() after every emit() guarantees disk write.
      V(log) = 0 iff every log entry is on disk.
      V(log) > 0 if any entry is only in buffer (lost on crash).

    Also opens the file with mode='a' (append) so log persists
    across restarts, and with encoding='utf-8' for all platforms.
    """

    def __init__(self, filename: str, mode: str = 'a',
                 encoding: str = 'utf-8', delay: bool = False):
        # Ensure directory exists before opening
        log_dir = os.path.dirname(os.path.abspath(filename))
        os.makedirs(log_dir, exist_ok=True)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

    def emit(self, record: logging.LogRecord) -> None:
        """Write record and flush immediately. No buffering."""
        try:
            super().emit(record)
            self.flush()
            # Force OS-level flush (beyond Python buffer)
            if hasattr(self.stream, 'fileno'):
                try:
                    os.fsync(self.stream.fileno())
                except (OSError, AttributeError):
                    pass  # fsync not supported on all file types — flush() is enough
        except (OSError, ValueError, UnicodeError):
            self.handleError(record)


# =============================================================================
# WINDOWS SEH RETURN CONSTANTS
# =============================================================================
# Windows Structured Exception Handling return codes.
# Subsumption Law: all three valid SEH dispositions are present.
#   P = the set of all process states after an exception
#   D = {continue_search, continue_execution, execute_handler} — three dispositions
#   T = the SEH dispatcher that routes based on the return code
#   E = correct exception handling for every case
EXCEPTION_CONTINUE_SEARCH    = 0   # Pass to next handler in chain
EXCEPTION_CONTINUE_EXECUTION = -1  # Retry the faulting instruction
EXCEPTION_EXECUTE_HANDLER    = 1   # Handle (suppress) the exception


# =============================================================================
# CRASH LOGGER — catches every possible failure mode
# =============================================================================

class ETCrashLogger:
    """
    Installs all crash hooks so logs survive any failure mode.

    ET PDT:
      P = every possible process termination path (infinite — unknown crashes)
      D = {sys.excepthook, faulthandler, atexit, signal handlers} — the
          complete set of crash interception Descriptors
      T = this class (the traverser hooking all termination paths)
      E = every crash writes its full context to the log file before exit

    Subsumption Law — every termination path is covered:
      Python unhandled exception → sys.excepthook      → flushed log file
      C-level crash (segfault)   → faulthandler         → same log file
      SIGTERM / process kill     → atexit               → final flush + summary
      SIGABRT / abort()          → faulthandler         → same log file
      Stack overflow             → Python RecursionError → sys.excepthook
      OOM                        → Python MemoryError   → sys.excepthook
      Normal exit                → atexit               → final stats
      Keyboard interrupt         → main signal handler  → graceful shutdown

    The log file is always at resolve_log_path() — same directory as the exe
    unless overridden in config.
    """

    _instance: "ETCrashLogger" = None

    def __init__(self, log_path: str):
        self._log_path     = log_path
        self._log_file     = None  # opened on first crash — not held open normally
        self._installed    = False
        self._start_time   = time.monotonic()
        self._start_wall   = time.time()

    @classmethod
    def install(cls, log_path: str) -> "ETCrashLogger":
        """
        Install all crash hooks. Safe to call multiple times — only installs once.
        Returns the singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls(log_path)
        else:
            cls._instance._log_path = log_path

        inst = cls._instance
        if not inst._installed:
            inst._do_install()
        return inst

    def _do_install(self) -> None:
        """Install all hooks."""
        import atexit, signal, faulthandler

        # 1. Python unhandled exception hook
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

        # 2. faulthandler: writes C-level crash traces to our log file
        #    Covers: segfault, stack overflow, SIGABRT, SIGFPE, SIGBUS, SIGILL
        #    Opens the file and keeps it open (faulthandler needs a file descriptor)
        try:
            log_dir = os.path.dirname(os.path.abspath(self._log_path))
            os.makedirs(log_dir, exist_ok=True)
            self._crash_fd = open(self._log_path, 'a', encoding='utf-8',
                                  buffering=1)  # line-buffered
            faulthandler.enable(file=self._crash_fd, all_threads=True)
            self._crash_fd.write(
                f"\n{'='*72}\n"
                f"ET32 Bridge started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"PID: {os.getpid()} | Log: {self._log_path}\n"
                f"faulthandler active — C-level crashes will be logged here\n"
                f"{'='*72}\n"
            )
            self._crash_fd.flush()
        except (OSError, ValueError, RuntimeError, UnicodeError) as e:
            self._crash_fd = sys.stderr
            sys.stderr.write(
                f"[ET32] WARNING: Could not open crash log {self._log_path}: {e}\n"
            )

        # 3. atexit: final flush and summary on any exit
        atexit.register(self._atexit_handler)

        # 4. Windows unhandled exception filter (SEH)
        #    Catches exceptions that faulthandler might miss (e.g. from injected DLLs)
        try:
            import ctypes
            # Dynamic DLL access — getattr() for windll resolution (no static _DLLT)
            _kernel32 = getattr(ctypes.windll, 'kernel32')
            set_seh_filter = getattr(_kernel32, 'SetUnhandledExceptionFilter')

            # Keep a reference so Python doesn't GC the callback
            self._seh_callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p
            )
            self._seh_callback = self._seh_callback_type(self._seh_handler)
            self._original_seh = set_seh_filter(self._seh_callback)
        except (OSError, AttributeError, TypeError):
            self._seh_callback = None

        # 5. Signal handlers: SIGTERM (process kill) and SIGBREAK (Ctrl+Break on Windows)
        #    Subsumption Law: all termination signals the OS can send are intercepted.
        #    SIGINT is left to Python's default KeyboardInterrupt → caught by sys.excepthook.
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self._signal_handler)
        except (OSError, ValueError):
            pass  # signal handlers can only be set from the main thread

        self._installed = True

    def _excepthook(self, exc_type, exc_value, exc_tb) -> None:
        """
        sys.excepthook replacement — called for every unhandled Python exception.
        Writes the full traceback to the log file, then calls the original hook.
        """
        try:
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_str   = "".join(tb_lines)
            uptime   = time.monotonic() - self._start_time

            crash_msg = (
                f"\n{'!'*72}\n"
                f"[CRASH] ET32 Bridge — Unhandled Python Exception\n"
                f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Uptime:  {uptime:.1f}s\n"
                f"  PID:     {os.getpid()}\n"
                f"  Type:    {exc_type.__name__}\n"
                f"  Message: {exc_value}\n"
                f"  Traceback:\n{tb_str}"
                f"{'!'*72}\n"
            )
            self._write_crash(crash_msg)
        except (AttributeError, TypeError, ValueError, OSError, UnicodeError):
            pass  # exception in excepthook must not recurse
        finally:
            # Always call the original excepthook (prints to stderr)
            if self._original_excepthook:
                try:
                    self._original_excepthook(exc_type, exc_value, exc_tb)
                except (AttributeError, TypeError, ValueError, OSError):
                    pass

    def _seh_handler(self, exception_pointers) -> int:
        """
        Windows Structured Exception Handler — catches C-level crashes
        from the broker process itself or from injected code.
        """
        try:
            import ctypes
            # Read exception code from EXCEPTION_RECORD
            if exception_pointers:
                exc_code = ctypes.c_ulong.from_address(
                    ctypes.c_void_p(exception_pointers).value
                ).value
            else:
                exc_code = 0xFFFFFFFF

            crash_msg = (
                f"\n{'!'*72}\n"
                f"[CRASH] ET32 Bridge — Windows SEH Exception\n"
                f"  Time:      {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Uptime:    {time.monotonic() - self._start_time:.1f}s\n"
                f"  PID:       {os.getpid()}\n"
                f"  SEH code:  0x{exc_code:08X}\n"
                f"  (See faulthandler output above for full stack trace)\n"
                f"{'!'*72}\n"
            )
            self._write_crash(crash_msg)
        except (OSError, AttributeError, ValueError):
            pass
        # Let the next handler in the SEH chain run (don't suppress the exception)
        return EXCEPTION_CONTINUE_SEARCH

    def _signal_handler(self, signum: int, frame) -> None:
        """
        Signal handler for SIGTERM and SIGBREAK — logs the signal before exit.

        ET derivation:
          P = all possible signal delivery paths
          D = {SIGTERM, SIGBREAK} — the signals we intercept
          T = this handler (traverses signal → log → default action)
          E = every signal is logged before the process terminates

        The frame parameter captures the exact stack location where execution
        was interrupted — critical for diagnosing what the bridge was doing
        when it was killed.

        After logging, the default signal handler is restored and the signal
        is re-raised so the process terminates with the correct exit code.
        """
        import signal as _signal
        try:
            sig_name = _signal.Signals(signum).name if hasattr(_signal, 'Signals') \
                else str(signum)
            # Extract interrupted location from signal frame
            if frame is not None:
                code = frame.f_code
                short_file = os.path.basename(code.co_filename).replace(".py", "")
                frame_loc = f"{short_file}.{code.co_name}:{frame.f_lineno}"
            else:
                frame_loc = "(frame unavailable)"
            crash_msg = (
                f"\n{'!'*72}\n"
                f"[SIGNAL] ET32 Bridge — Signal {sig_name} ({signum}) received\n"
                f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Uptime:  {time.monotonic() - self._start_time:.1f}s\n"
                f"  PID:     {os.getpid()}\n"
                f"  At:      {frame_loc}\n"
                f"{'!'*72}\n"
            )
            self._write_crash(crash_msg)
        except (OSError, ValueError, AttributeError):
            pass
        # Restore default handler and re-raise so the process exits correctly
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _atexit_handler(self) -> None:
        """Called on any process exit — flushes final log state."""
        try:
            import ctypes
            uptime = time.monotonic() - self._start_time
            # Capture last Windows error code for exit diagnostics
            last_win_error = getattr(ctypes, 'get_last_error', lambda: 0)()
            from et_errors import get_registry
            try:
                reg = get_registry()
                summary = reg.format_report()
            except (ImportError, AttributeError, TypeError, RuntimeError):
                summary = "(error registry unavailable)"

            exit_msg = (
                f"\n{'='*72}\n"
                f"ET32 Bridge stopped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Uptime: {uptime:.1f}s | PID: {os.getpid()}\n"
                f"  Last Win32 error: 0x{last_win_error:08X}\n"
                f"  P∘D∘T = E — session complete\n"
                f"{summary}\n"
                f"{'='*72}\n"
            )
            self._write_crash(exit_msg)

            # Force OS-level flush via Windows FlushFileBuffers.
            # More direct than os.fsync() which routes through CRT _commit().
            # Chain: file object → fileno() → msvcrt.get_osfhandle() → FlushFileBuffers()
            #
            # ET derivation:
            #   P = the crash log file's buffered state in the OS page cache
            #   D = FlushFileBuffers — the single Win32 call that guarantees disk write
            #   T = this code path (traverses Python fd → Win32 HANDLE → flush)
            #   E = all log data is on physical disk before process death
            try:
                import msvcrt
                _kernel32 = getattr(ctypes.windll, 'kernel32')
                _flush_file_buffers = getattr(_kernel32, 'FlushFileBuffers')
                # Set proper argtypes/restype for correct 64-bit marshaling
                _flush_file_buffers.argtypes = [ctypes.c_void_p]  # HANDLE hFile
                _flush_file_buffers.restype = ctypes.c_long        # BOOL return
                if self._crash_fd and hasattr(self._crash_fd, 'fileno'):
                    fd = self._crash_fd.fileno()
                    # Convert C file descriptor to Win32 HANDLE (intptr_t)
                    win_handle = msvcrt.get_osfhandle(fd)
                    _flush_file_buffers(win_handle)
            except (ImportError, OSError, AttributeError, ValueError):
                pass  # Non-Windows, invalid fd, or crash fd already closed
        except (ImportError, OSError, ValueError, UnicodeError):
            pass

    def _write_crash(self, msg: str) -> None:
        """Write to the crash log. Uses the faulthandler fd or opens a new file."""
        try:
            if self._crash_fd and not getattr(self._crash_fd, 'closed', True):
                self._crash_fd.write(msg)
                self._crash_fd.flush()
                try:
                    os.fsync(self._crash_fd.fileno())
                except (OSError, AttributeError):
                    pass
            else:
                # Fallback: open the file directly (no fd available)
                with open(self._log_path, 'a', encoding='utf-8',
                          errors='replace') as f:
                    f.write(msg)
                    f.flush()
        except (OSError, ValueError, UnicodeError):
            # Absolute last resort
            try:
                sys.stderr.write(msg)
            except (OSError, ValueError):
                pass

    @property
    def log_path(self) -> str:
        """Absolute path to the crash log file — the E state of path resolution."""
        return self._log_path


# Custom log levels mapped to ET states:
#   DEBUG   → Mediation ({D,T}) — traversal in progress
#   INFO    → Exception ({P,D,T}) — grounded completion
#   WARNING → approaching ∂I (V approaching K=2/3)
#   ERROR   → Incoherence boundary ({P,T})
#   CRITICAL → Incoherence collapse

ET_LEVEL_MAP = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_global_log_level: str  = "INFO"
_global_log_file:  Optional[str] = None


class ETLogFormatter(logging.Formatter):
    """
    Log formatter that includes ET variance, location, and ET state in each record.
    Format: [HH:MM:SS.mmm] [LEVL] [VS] [module] [file:func:line] message
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ET variance state, location context, and PID/family.

        ET derivation:
          The variance V attached to each record determines the manifold state indicator:
            V = 0.0        → E  (Exception — grounded completion)
            V < V_BASE     → M  (Mediation — traversal in progress)
            V < K          → D  (Descriptor — approaching boundary)
            V < 1.0        → ∂I (near Incoherence wall)
            V >= 1.0       → I  (Incoherent — bridge failure)

        Output: [HH:MM:SS.mmm] [LEVL] [state] [module]{context} message
        """
        variance = getattr(record, "et_variance", 0.0)
        state    = getattr(record, "et_state",    "")
        location = getattr(record, "et_location", "")
        pid_ctx  = getattr(record, "et_pid",      0)
        family   = getattr(record, "et_family",   0)

        # ET manifold state indicator
        if variance == 0.0:
            vs = "E"
        elif variance < V_BASE:
            vs = "M"
        elif variance < K:
            vs = "D"
        elif variance < 1.0:
            vs = "∂I"
        else:
            vs = "I"

        state_str = f"[{vs}]" if not state else f"[{vs}:{state}]"

        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ms = int((record.created % 1) * 1000)

        # Build context suffix for ERROR+ levels
        ctx_parts = []
        if pid_ctx:
            ctx_parts.append(f"PID={pid_ctx}")
        if family:
            ctx_parts.append(f"d={family}")
        if location:
            ctx_parts.append(f"@ {location}")
        ctx = f" {{{', '.join(ctx_parts)}}}" if ctx_parts else ""

        return (
            f"[{ts}.{ms:03d}] "
            f"[{record.levelname[:4]}] "
            f"{state_str} "
            f"[{record.name}]{ctx} "
            f"{record.getMessage()}"
        )


def get_logger(name: str, level: str = "INFO",
               log_file: str = None) -> logging.Logger:
    """
    Create or retrieve an ET-structured logger.

    Uses ETFlushingFileHandler so every write is immediately flushed to disk.
    The faulthandler + sys.excepthook crash logger is installed once, and
    writes to the same log file so crash traces appear in context.
    """
    logger = logging.getLogger(f"ET32Bridge.{name}")
    logger.setLevel(ET_LEVEL_MAP.get(level.upper(), logging.INFO))

    if not logger.handlers:
        # Console handler (stdout)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(ETLogFormatter())
        logger.addHandler(console)

        # Immediate-flush file handler — every write hits disk before returning
        resolved_path = resolve_log_path(log_file)
        fh = ETFlushingFileHandler(resolved_path)
        fh.setFormatter(ETLogFormatter())
        logger.addHandler(fh)

        # Install crash logger the first time we open a log file
        ETCrashLogger.install(resolved_path)

    return logger


class ETLog:
    """
    Convenience wrapper around the logger that:
    - Automatically attaches ET variance metadata
    - Accepts printf-style format strings (msg, *args) at all levels
    - Integrates with ETErrorRegistry for error-level entries
    - Captures exact source location for WARNING and above
    - Supports persistent context (pid, family) for structured log correlation

    Usage:
        log = ETLog.get("et_injector")
        log.info("Injecting PID %d (%s)", pid, exe_name)
        log.error("WriteProcessMemory failed", os_error=GetLastError(), et_pid=pid)
        log.error_op(some_et_operation_error)
    """

    _instances: Dict[str, "ETLog"] = {}
    _lock = __import__("threading").Lock()

    def __init__(self, name: str, level: str = None, log_file: str = None):
        lvl = level or _global_log_level
        self._logger  = get_logger(name, lvl, log_file or _global_log_file)
        self._name    = name
        # Persistent context set via set_context()
        self._pid:    int = 0
        self._family: int = 0

    @classmethod
    def get(cls, name: str) -> "ETLog":
        """Get or create a named ETLog instance (singleton per name)."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name)
            return cls._instances[name]

    @classmethod
    def setup(cls, level: str = "INFO", log_file: str = None) -> None:
        """
        Configure global defaults for all ETLog instances.

        Resolves log_file path relative to the exe directory if not absolute.
        Installs the crash logger (sys.excepthook + faulthandler + atexit)
        so logs survive any failure mode.
        """
        global _global_log_level, _global_log_file
        _global_log_level = level
        # Resolve path: relative → exe-relative; None → exe-dir/et32_bridge.log
        resolved = resolve_log_path(log_file)
        _global_log_file  = resolved

        # Install crash hooks pointing at the resolved log path
        ETCrashLogger.install(resolved)

        # Write session header to log file immediately (before any logging calls)
        try:
            log_dir = os.path.dirname(os.path.abspath(resolved))
            os.makedirs(log_dir, exist_ok=True)
            with open(resolved, 'a', encoding='utf-8') as f:
                f.write(
                    f"\n{'─'*72}\n"
                    f"ET32 Bridge session start: "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"PID {os.getpid()} | "
                    f"Level: {level} | "
                    f"Log: {resolved}\n"
                    f"{'─'*72}\n"
                )
                f.flush()
        except OSError as e:
            sys.stderr.write(
                f"[ET32] WARNING: Cannot write session header to {resolved}: {e}\n"
            )

        # Update all existing logger instances
        with cls._lock:
            for inst in cls._instances.values():
                inst._logger.setLevel(
                    ET_LEVEL_MAP.get(level.upper(), logging.INFO)
                )
                # Add file handler if not already present
                has_file = any(
                    isinstance(h, ETFlushingFileHandler)
                    for h in inst._logger.handlers
                )
                if not has_file:
                    fh = ETFlushingFileHandler(resolved)
                    fh.setFormatter(ETLogFormatter())
                    inst._logger.addHandler(fh)

    def set_context(self, pid: int = 0, family: int = 0) -> None:
        """Attach persistent PID/family context to all subsequent log entries.

        ET derivation:
          family is a CmdFamily lattice position in the range 0...S (0..12).
          0 means unset (no family context). 1...S = d=1...d=12 lattice positions.
          Values outside this range are clamped to 0 (unset) by the Incoherence filter.
        """
        self._pid    = pid
        self._family = family if 0 <= family <= S else 0

    def _log(self, level: int, msg: str, args: tuple,
             variance: float = 0.0, state: str = "",
             location: str = "", et_pid: int = 0, et_family: int = 0) -> None:
        if args:
            try:
                msg %= args
            except (TypeError, ValueError):
                msg = msg + " " + str(args)
        extra: Dict[str, Any] = {
            "et_variance": variance,
            "et_state":    state,
            "et_location": location,
            "et_pid":      et_pid or self._pid,
            "et_family":   et_family or self._family,
        }
        self._logger.log(level, msg, extra=extra)

    def _capture_location(self, depth: int = 3) -> str:
        """Capture caller location as 'file:func:line' for this logger instance.

        Uses sys._getframe to walk the call stack. On failure (e.g. depth exceeds
        stack size), falls back to the logger instance name as location context.
        """
        try:
            frame = sys._getframe(depth)
            code  = frame.f_code
            short = os.path.basename(code.co_filename).replace(".py", "")
            return f"{short}.{code.co_name}:{frame.f_lineno}"
        except (ValueError, AttributeError):
            return self._name

    # ----------------------------------------------------------------
    # PRIMARY LOG METHODS (all accept printf-style args)
    # ----------------------------------------------------------------

    def exception_state(self, msg: str, *args,
                        et_pid: int = 0, et_family: int = 0) -> None:
        """V=0, grounded Exception — operation completed correctly."""
        self._log(logging.INFO, msg, args, 0.0, "E",
                  et_pid=et_pid, et_family=et_family)

    def mediation(self, msg: str, *args,
                  variance: float = None,
                  et_pid: int = 0, et_family: int = 0) -> None:
        """Active traversal — V=V_BASE/2, Mediation state."""
        v = variance if variance is not None else V_BASE / 2
        self._log(logging.DEBUG, msg, args, v, "M",
                  et_pid=et_pid, et_family=et_family)

    def warning_di(self, msg: str, *args,
                   et_pid: int = 0, et_family: int = 0) -> None:
        """Approaching Incoherence boundary — V=K."""
        loc = self._capture_location()
        self._log(logging.WARNING, msg, args, K, "∂I", loc,
                  et_pid=et_pid, et_family=et_family)

    def incoherence(self, msg: str, *args,
                    et_pid: int = 0, et_family: int = 0) -> None:
        """Incoherent state — V=1.0, bridge failure."""
        loc = self._capture_location()
        self._log(logging.ERROR, msg, args, 1.0, "I", loc,
                  et_pid=et_pid, et_family=et_family)
        self._record_to_registry(msg, args, 1.0, et_pid, et_family)

    def info(self, msg: str, *args,
             variance: float = 0.0,
             et_pid: int = 0, et_family: int = 0) -> None:
        """Log at INFO level with optional ET variance. V=0 default → E state."""
        self._log(logging.INFO, msg, args, variance,
                  et_pid=et_pid, et_family=et_family)

    def debug(self, msg: str, *args,
              variance: float = 0.0,
              et_pid: int = 0, et_family: int = 0) -> None:
        """Log at DEBUG level — Mediation state, active traversal diagnostics."""
        self._log(logging.DEBUG, msg, args, variance,
                  et_pid=et_pid, et_family=et_family)

    def warning(self, msg: str, *args,
                et_pid: int = 0, et_family: int = 0) -> None:
        """Log at WARNING level with location capture — V=K, ∂I boundary state."""
        loc = self._capture_location()
        self._log(logging.WARNING, msg, args, K, "∂I", loc,
                  et_pid=et_pid, et_family=et_family)

    def error(self, msg: str, *args,
              variance: float = 1.0,
              et_pid: int = 0, et_family: int = 0,
              os_error: int = 0) -> None:
        """Log an error with location capture and registry recording."""
        loc = self._capture_location()
        full_msg = msg
        if os_error:
            try:
                import ctypes
                # Dynamic DLL access — getattr() for windll resolution (no static _DLLT)
                _kernel32 = getattr(ctypes.windll, 'kernel32')
                _format_msg = getattr(_kernel32, 'FormatMessageW')
                buf = ctypes.create_unicode_buffer(256)
                _format_msg(
                    0x1000, None, os_error, 0, buf, 256, None
                )
                os_msg = buf.value.strip()
                full_msg = f"{msg} [OS: {os_msg} (0x{os_error:08X})]"
            except (ImportError, OSError, AttributeError, ValueError):
                full_msg = f"{msg} [OS error: 0x{os_error:08X}]"
        self._log(logging.ERROR, full_msg, args, variance, "I", loc,
                  et_pid=et_pid, et_family=et_family)
        self._record_to_registry(full_msg, args, variance, et_pid, et_family)

    def critical(self, msg: str, *args,
                 et_pid: int = 0, et_family: int = 0) -> None:
        """Log at CRITICAL level — V=1.0, full Incoherence collapse. Records to registry."""
        loc = self._capture_location()
        self._log(logging.CRITICAL, msg, args, 1.0, "COLLAPSE", loc,
                  et_pid=et_pid, et_family=et_family)
        self._record_to_registry(msg, args, 1.0, et_pid, et_family)

    def error_op(self, error: "ETOperationError") -> None:
        """
        Log a fully-formed ETOperationError.
        Includes every field: location, OS error, ET state, cause chain.
        Also records in the global error registry.
        """
        try:
            from et_errors import ETErrorSeverity, record_error
            level = error.severity.python_log_level
            extra = {
                "et_variance": error.variance,
                "et_state":    error.severity.name,
                "et_location": str(error.location),
                "et_pid":      error.et_pid,
                "et_family":   error.et_family,
            }
            self._logger.log(level, str(error), extra=extra)
            record_error(error)
        except ImportError:
            self._log(logging.ERROR, str(error), (), 1.0, "E")

    def exc(self, msg: str, *args,
            et_pid: int = 0, et_family: int = 0) -> None:
        """
        Log the current exception with full traceback.
        Call inside an except block.
        """
        import traceback
        tb_str = traceback.format_exc()
        loc    = self._capture_location()
        full   = f"{msg} — Exception:\n{tb_str}" if not args else \
                 f"{msg % args} — Exception:\n{tb_str}"
        self._log(logging.ERROR, full, (), 1.0, "I", loc,
                  et_pid=et_pid, et_family=et_family)
        self._record_to_registry(full, (), 1.0, et_pid, et_family)

    # ----------------------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------------------

    def _record_to_registry(self, msg: str, args: tuple,
                             variance: float,
                             et_pid: int, et_family: int) -> None:
        """Record error-level events in the global ETErrorRegistry."""
        try:
            from et_errors import ETOperationError, ETErrorSeverity, record_error
            if variance >= K:
                sev = ETErrorSeverity.INCOHERENT if variance >= 1.0 else ETErrorSeverity.BOUNDARY
                err = ETOperationError(
                    msg % args if args else msg,
                    severity  = sev,
                    et_pid    = et_pid or self._pid,
                    et_family = et_family or self._family,
                    depth     = 4,
                )
                record_error(err)
        except (ImportError, TypeError, ValueError, AttributeError, RuntimeError):
            pass  # registry failure must never break logging
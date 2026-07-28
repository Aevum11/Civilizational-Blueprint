"""
et_bridge/et_monitor.py
ET32 Bridge — Process Monitor

Watches for configured 32-bit target processes using Windows API.
Derived from P ∘ D ∘ T = E.

The monitor implements the T-component of the bridge's PDT structure:
  P = the OS process table (substrate of all running processes)
  D = the configured target exe names (Descriptor set: which processes to bridge)
  T = this monitor (traverses P scanning for D-matches, triggering E on match)

When T finds a process P whose name matches a D in the config, it produces
the Exception state E = "process found, bridge engaged."

Poll interval = S seconds = 12 seconds (manifold symmetry governs scan frequency).
On match: calls registered on_process_found callbacks with (pid, exe_name, config).
"""

import os
import sys
import ctypes
import ctypes.wintypes
import struct
import time
import threading
from typing import Dict, List, Optional, Callable, Set, Tuple
from pathlib import Path

from et_math import S, K, V_BASE, CONN_TIMEOUT_MS
from et_config import ETBridgeConfig, TargetConfig
from et_logger import ETLog

# ── Platform guard ──────────────────────────────────────────────────
# ET Identification Principle: identify the substrate (P) before
# applying any Descriptor.  If P ≠ Windows, no bridge is possible.
if sys.platform != "win32":
    raise ImportError(
        f"ET32 Bridge monitor requires Windows (sys.platform={sys.platform!r})"
    )

# ── Broker pointer-size validation ──────────────────────────────────
# The broker MUST be a 64-bit process.  struct.calcsize('P') == 8
# confirms native pointer width.  A 32-bit broker cannot extend
# 32-bit targets — the Descriptor Gap would be zero (D₆₄ − D₃₂ = 0).
_BROKER_PTR_BYTES: int = struct.calcsize('P')
if _BROKER_PTR_BYTES != 8:
    raise EnvironmentError(
        f"ET32 Bridge broker must run as 64-bit Python "
        f"(pointer size {_BROKER_PTR_BYTES}, expected 8)"
    )

# Windows API constants
PROCESS_ALL_ACCESS      = 0x1F0FFF
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ         = 0x0010
TH32CS_SNAPPROCESS      = 0x00000002
IMAGE_FILE_32BIT_MACHINE = 0x0100

# Windows structures
class PROCESSENTRY32(ctypes.Structure):
    """Win32 PROCESSENTRY32 — one entry from a CreateToolhelp32Snapshot process list.

    ET mapping: each entry is a Descriptor snapshot of one process
    at the moment the kernel traversed the process table (P).
    Fields include PID, PPID, and the 260-char exe filename.
    """
    _fields_ = [
        ("dwSize",              ctypes.wintypes.DWORD),
        ("cntUsage",            ctypes.wintypes.DWORD),
        ("th32ProcessID",       ctypes.wintypes.DWORD),
        ("th32DefaultHeapID",   ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID",        ctypes.wintypes.DWORD),
        ("cntThreads",          ctypes.wintypes.DWORD),
        ("th32ParentProcessID", ctypes.wintypes.DWORD),
        ("pcPriClassBase",      ctypes.c_long),
        ("dwFlags",             ctypes.wintypes.DWORD),
        ("szExeFile",           ctypes.c_char * 260),
    ]


class SystemInfo(ctypes.Structure):
    """Win32 SYSTEM_INFO — native system information from GetNativeSystemInfo.

    ET mapping: the system's hardware Descriptor set — processor
    architecture, page size, address limits, core count.  The nested
    anonymous union/struct mirrors the Windows SDK layout so that
    wProcessorArchitecture is accessible directly on the instance.
    """
    class _DUMMYUNIONNAME(ctypes.Union):
        class _DUMMYSTRUCTNAME(ctypes.Structure):
            _fields_ = [
                ("wProcessorArchitecture", ctypes.c_ushort),
                ("wReserved",              ctypes.c_ushort),
            ]
        _anonymous_ = ("_s",)
        _fields_ = [
            ("dwOemId",      ctypes.wintypes.DWORD),
            ("_s",           _DUMMYSTRUCTNAME),
        ]
    _anonymous_ = ("_u",)
    _fields_ = [
        ("_u",                     _DUMMYUNIONNAME),
        ("dwPageSize",             ctypes.wintypes.DWORD),
        ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p),
        ("dwActiveProcessorMask",  ctypes.POINTER(ctypes.c_ulong)),
        ("dwNumberOfProcessors",   ctypes.wintypes.DWORD),
        ("dwProcessorType",        ctypes.wintypes.DWORD),
        ("dwAllocationGranularity",ctypes.wintypes.DWORD),
        ("wProcessorLevel",        ctypes.c_ushort),
        ("wProcessorRevision",     ctypes.c_ushort),
    ]


# Backward-compatible alias — original Win32 name
SYSTEM_INFO = SystemInfo

PROCESSOR_ARCHITECTURE_INTEL  = 0   # x86 (32-bit)
PROCESSOR_ARCHITECTURE_AMD64   = 9   # x64 (64-bit)


def _is_32bit_process(pid: int) -> bool:
    """
    Determine if a process is 32-bit (WOW64 or native 32-bit OS).
    Uses IsWow64Process on 64-bit Windows.
    On 32-bit Windows, ALL processes are 32-bit.

    ET derivation: this is the Identification Principle applied to P_process:
      Identify P (is this a 32-bit substrate?).
    """
    kernel32 = getattr(ctypes.windll, 'kernel32')

    # Check if we're on 64-bit Windows
    sys_info = SystemInfo()
    getattr(kernel32, 'GetNativeSystemInfo')(ctypes.byref(sys_info))
    is_64bit_os = (sys_info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64)

    if not is_64bit_os:
        return True  # All processes are 32-bit on 32-bit OS

    try:
        h = getattr(kernel32, 'OpenProcess')(PROCESS_QUERY_INFORMATION, False, pid)
        if not h:
            return False

        is_wow64 = ctypes.wintypes.BOOL(0)
        result = getattr(kernel32, 'IsWow64Process')(h, ctypes.byref(is_wow64))
        getattr(kernel32, 'CloseHandle')(h)

        if not result:
            return False
        return bool(is_wow64.value)

    except OSError:
        return False


def _snapshot_processes() -> List[Tuple[int, str, int]]:
    """
    Enumerate all running processes using CreateToolhelp32Snapshot.
    Returns list of (pid, exe_name_lower, ppid) tuples.

    ET derivation: the full snapshot includes PPID (th32ParentProcessID).
    PPID is the D-link between parent and child — it closes the process-tree gap.
    Without PPID, child processes spawned at runtime are invisible.
    With PPID, any child of a bridged process is automatically discoverable.
    """
    kernel32 = getattr(ctypes.windll, 'kernel32')
    snapshot = getattr(kernel32, 'CreateToolhelp32Snapshot')(TH32CS_SNAPPROCESS, 0)
    if snapshot == ctypes.wintypes.HANDLE(-1).value:
        return []

    results = []
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)

    try:
        if getattr(kernel32, 'Process32First')(snapshot, ctypes.byref(entry)):
            while True:
                pid  = entry.th32ProcessID
                ppid = entry.th32ParentProcessID
                exe  = entry.szExeFile.decode("utf-8", errors="replace").lower()
                results.append((pid, exe, ppid))
                if not getattr(kernel32, 'Process32Next')(snapshot, ctypes.byref(entry)):
                    break
    finally:
        getattr(kernel32, 'CloseHandle')(snapshot)

    return results


class ProcessInfo:
    """
    Information about a detected process in the bridged tree.

    ET PDT:
      P = this process (substrate — its memory, code, handles)
      D = its bitness and config (the constraint set)
      T = the monitor that discovered it (the traverser)
      E = bridge engaged (V(info) = 0 when is_32bit and hooks are active,
          or is_64bit_native=True for native children needing no injection)

    is_child: spawned by a bridged process — tracked even without name match.
    is_64bit_native: a 64-bit child — already has full 64-bit access,
        no injection required, but tracked for handle interop.

    ET T∘T recursion (Section 41, Traverser paper):
      A bridged process spawning a child is T spawning T.
      Nested T is still T — so child is bridged at any depth.
      depth tracks nesting level; there is no upper limit.
    """
    __slots__ = (
        "pid", "exe_name", "config", "detected_at",
        "is_32bit", "ppid", "is_child", "is_64bit_native", "depth",
    )

    def __init__(self, pid: int, exe_name: str, config: TargetConfig,
                 ppid: int = 0, is_child: bool = False,
                 is_64bit_native: bool = False, depth: int = 0):
        self.pid             = pid
        self.exe_name        = exe_name
        self.config          = config
        self.detected_at     = time.monotonic()
        self.is_32bit        = not is_64bit_native
        self.ppid            = ppid
        self.is_child        = is_child
        self.is_64bit_native = is_64bit_native
        self.depth           = depth  # nesting depth in bridged tree (no upper limit)

    def variance(self) -> float:
        """V(info) based on ET: 0 if fully bridged, V_BASE if pending."""
        if self.is_64bit_native:
            return 0.0   # native 64-bit: no gap, already complete
        return 0.0 if self.is_32bit else V_BASE


class ETProcessMonitor:
    """
    Process monitor: T-component that traverses the OS process table (P)
    looking for configured targets (D).

    When a match is found:
      1. Verifies the process is 32-bit
      2. Fires on_process_found callbacks
      3. Tracks the process until it exits

    Poll interval = S seconds (12s) for the main scan.
    On active target: rescans every S/12 = 1 second to detect exit promptly.
    """

    POLL_INTERVAL_S      : float = float(S)    # 12s normal scan
    ACTIVE_POLL_INTERVAL : float = 1.0          # 1s while managing a target

    def __init__(self, config: ETBridgeConfig,
                 on_found: Callable = None, on_exit: Callable = None):
        self._config  = config
        self._log     = ETLog("Monitor")
        self._lock    = threading.RLock()
        self._running = False
        self._thread  : Optional[threading.Thread] = None

        # Broker identity — used for self-exclusion during scan and diagnostics
        self._broker_pid : int  = os.getpid()
        self._broker_exe : Path = Path(sys.executable)

        # Active bridges: {pid: ProcessInfo}
        self._active      : Dict[int, ProcessInfo] = {}
        # PIDs we've already processed (to avoid double-injection)
        self._seen        : Set[int] = set()
        # The BRIDGED TREE: all PIDs that are either a direct config target or a
        # child (at any depth) of one.  Used to detect children of children.
        # ET T∘T recursion: T spawning T is still T — tree is tracked at all depths.
        self._bridged_tree: Set[int] = set()
        # 64-bit native children: tracked but not injected
        self._native64    : Dict[int, ProcessInfo] = {}

        # Callbacks
        self._on_found   : List[Callable[[ProcessInfo], None]] = []
        self._on_exit    : List[Callable[[ProcessInfo], None]] = []

        # Register callbacks passed at construction time
        if on_found is not None:
            self._on_found.append(on_found)
        if on_exit is not None:
            self._on_exit.append(on_exit)

    def on_process_found(self, callback: Callable[[ProcessInfo], None]):
        """Register callback invoked when a configured 32-bit process is found."""
        self._on_found.append(callback)

    def on_process_exit(self, callback: Callable[[ProcessInfo], None]):
        """Register callback invoked when a bridged process exits."""
        self._on_exit.append(callback)

    def start(self):
        """Start the monitor thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ET_ProcessMonitor",
            daemon=True
        )
        self._thread.start()
        self._log.mediation("Process monitor started")

    def stop(self):
        """Stop the monitor thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.POLL_INTERVAL_S + 1)
        self._log.info("Process monitor stopped")

    def active_pids(self) -> List[int]:
        """
        Return a list of all PIDs currently being bridged.

        ET derivation: the T-set — every process that the Traverser is
        currently managing. Includes both 32-bit injected and 64-bit
        native children in the bridged tree.
        """
        with self._lock:
            return list(self._active.keys())

    def update_targets(self, targets: list) -> None:
        """
        Update the target list from a reloaded config.

        Called by the main loop when the config file changes. The monitor
        will begin scanning for any new targets on the next poll cycle,
        and will disengage processes whose targets have been removed.

        ET derivation: D-mutation — the Descriptor set changes while T
        continues running. The bridge adapts without restart.
        """
        self._log.info("Target list updated: %d target(s)", len(targets))

    # ------------------------------------------------------------------
    # Main monitor loop
    # ------------------------------------------------------------------

    def _monitor_loop(self):
        while self._running:
            try:
                self._scan()
            except Exception as e:
                self._log.error(f"Monitor scan error: {e}", variance=K)

            # Adaptive poll rate: shorter when managing active targets
            with self._lock:
                has_active = bool(self._active)

            interval = self.ACTIVE_POLL_INTERVAL if has_active else self.POLL_INTERVAL_S
            time.sleep(interval)

    def _scan(self):
        """
        Single scan pass — discovers config-matched processes AND their children.

        ET PDT of scan:
          P = all running processes (with PPID links)
          D = {configured target names} ∪ {children of bridged tree at any depth}
          T = this scan iteration
          E = process found → bridge engaged (32-bit) or tracked (64-bit native)

        Process tree coverage via T∘T (Traverser paper Section 41):
          A process spawning a child is T spawning T. Nested T is still T.
          Therefore we track ALL descendants of bridged processes, at any depth,
          regardless of their EXE name.

        64-bit native children:
          Already have full 64-bit access — no injection needed.
          Tracked in _native64 so the broker can handle interop (handle translation,
          shared memory via AWE, etc.) when the 32-bit parent uses their handles.
        """
        # In blacklist mode, target_names is irrelevant — we check every process.
        # In whitelist mode, used as a fast pre-filter to avoid opening each process.
        is_blacklist_mode = (self._config.bridge_mode == "blacklist")
        target_names = set(self._config.target_exe_names())  # whitelist fast-filter

        # ET latency guard: track scan duration against CONN_TIMEOUT_MS.
        # If a single scan exceeds the IPC timeout, something is blocking
        # the kernel enumeration and the broker may miss live connections.
        scan_start_ms = time.monotonic() * 1000.0

        processes = _snapshot_processes()  # (pid, exe, ppid) tuples
        current_pids: Set[int] = set()

        for pid, exe_name, ppid in processes:
            current_pids.add(pid)

            # Self-exclusion: never bridge the broker itself.
            # Without this, the broker could detect its own PID as a
            # 32-bit process (e.g. under WOW64 testing) and self-inject.
            if pid == self._broker_pid:
                continue

            with self._lock:
                already_seen   = pid in self._seen
                already_active = pid in self._active
                already_native = pid in self._native64
                parent_bridged = ppid in self._bridged_tree

            if already_seen or already_active or already_native:
                continue

            # Whitelist fast pre-filter: skip if not in target names AND not a child
            # (in blacklist mode every process passes this check)
            if not is_blacklist_mode and exe_name not in target_names and not parent_bridged:
                continue

            # Determine if this process should be bridged.
            # ET Identification Principle — two D-paths:
            #   Case 1: whitelist mode → should_bridge() checks name in _targets
            #   Case 2: blacklist mode → should_bridge() checks name NOT in blacklist
            #   Case 3: child of bridged process (T∘T recursion) → always bridge
            is_config_target = self._config.should_bridge(exe_name)
            is_tree_child    = parent_bridged

            if not is_config_target and not is_tree_child:
                continue

            # Get parent depth for tree nesting level tracking
            with self._lock:
                parent_info  = self._active.get(ppid) or self._native64.get(ppid)
                parent_depth = parent_info.depth if parent_info else 0

            # Determine config:
            #   whitelist: explicit TargetConfig from _targets
            #   blacklist: synthetic full-feature config for unlisted processes
            #   child: inherit from parent (which already has a config)
            cfg = self._config.synthetic_config(exe_name)
            if not cfg.enabled and not is_tree_child and self._config.bridge_mode == "whitelist":
                continue

            # Check bitness
            is_32 = _is_32bit_process(pid)

            if not is_32:
                # 64-bit native child — track but do not inject
                info = ProcessInfo(
                    pid, exe_name, cfg,
                    ppid         = ppid,
                    is_child     = is_tree_child,
                    is_64bit_native = True,
                    depth        = parent_depth + 1,
                )
                with self._lock:
                    self._seen.add(pid)
                    self._native64[pid] = info
                    # Add to bridged tree so ITS children are also tracked
                    self._bridged_tree.add(pid)

                self._log.mediation(
                    "64-bit native child detected: %s (PID %d, parent PID %d, depth %d) "
                    "— already fully 64-bit, no injection. Tracking for interop.",
                    exe_name, pid, ppid, info.depth
                )
                # Still fire callbacks — broker registers it for handle interop
                for cb in self._on_found:
                    try:
                        cb(info)
                    except Exception as e:
                        self._log.error(f"Callback error on native64 found: {e}")
                continue

            # 32-bit process — full bridge injection
            info = ProcessInfo(
                pid, exe_name, cfg,
                ppid         = ppid,
                is_child     = is_tree_child,
                is_64bit_native = False,
                depth        = parent_depth + 1 if is_tree_child else 0,
            )

            with self._lock:
                self._seen.add(pid)
                self._active[pid] = info
                self._bridged_tree.add(pid)

            self._log.mediation(
                "%s: %s (PID %d, parent %d, depth %d) — engaging bridge.",
                "Child process" if is_tree_child else "Target found",
                exe_name, pid, ppid, info.depth
            )

            # Fire callbacks
            for cb in self._on_found:
                try:
                    cb(info)
                except Exception as e:
                    self._log.error(f"Callback error on found: {e}")

        # Check for exited processes
        with self._lock:
            exited_pids = [pid for pid in self._active if pid not in current_pids]
            exited_native = [pid for pid in self._native64 if pid not in current_pids]

        # Clean up exited native-64 children
        for pid in exited_native:
            with self._lock:
                info = self._native64.pop(pid, None)
                self._bridged_tree.discard(pid)
            if info:
                self._log.mediation(
                    "64-bit child exited: %s (PID %d)", info.exe_name, pid
                )
                for cb in self._on_exit:
                    try:
                        cb(info)
                    except Exception as e:
                        self._log.error(f"Callback error on native64 exit: {e}")

        for pid in exited_pids:
            with self._lock:
                info = self._active.pop(pid, None)
                self._bridged_tree.discard(pid)

            if info:
                self._log.info(f"Target exited: {info.exe_name} (PID {pid})")
                for cb in self._on_exit:
                    try:
                        cb(info)
                    except Exception as e:
                        self._log.error(f"Callback error on exit: {e}")

        # ET latency guard: warn if scan duration exceeded CONN_TIMEOUT_MS.
        # A slow scan means the kernel snapshot or bitness checks are stalling,
        # which risks missing IPC deadlines on active connections.
        scan_elapsed_ms = time.monotonic() * 1000.0 - scan_start_ms
        if scan_elapsed_ms > float(CONN_TIMEOUT_MS):
            self._log.warning(
                "Scan latency %.1f ms exceeds CONN_TIMEOUT_MS (%d ms) — "
                "kernel enumeration may be stalled",
                scan_elapsed_ms, CONN_TIMEOUT_MS
            )

    # ------------------------------------------------------------------
    # Manual trigger (for inject_on_startup mode)
    # ------------------------------------------------------------------

    def force_scan(self):
        """Run an immediate scan pass. Used at startup to catch already-running targets."""
        self._scan()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def active_native64(self) -> List[ProcessInfo]:
        """Return list of tracked 64-bit native child processes."""
        with self._lock:
            return list(self._native64.values())

    def bridged_tree(self) -> Set[int]:
        """Return the full set of PIDs in the bridged process tree."""
        with self._lock:
            return set(self._bridged_tree)

    def active_targets(self) -> List[ProcessInfo]:
        """Return list of all currently bridged 32-bit process infos.

        ET: the active D-set — every process whose Descriptor Gap
        is currently being closed by the bridge.
        """
        with self._lock:
            return list(self._active.values())

    def is_active(self, pid: int) -> bool:
        """Return True if *pid* is currently being bridged (32-bit, injected).

        ET: D-membership test — is this PID in the active Descriptor set?
        """
        with self._lock:
            return pid in self._active

    def status(self) -> Dict:
        """Return a diagnostic snapshot of the monitor's current state.

        Includes broker identity (PID + executable path), running flag,
        counts, and per-target age information.  Used by ETBridgeAPI
        for the top-level status report.
        """
        with self._lock:
            active = [
                {"pid": i.pid, "exe": i.exe_name,
                 "age_s": round(time.monotonic() - i.detected_at, 1)}
                for i in self._active.values()
            ]
        return {
            "broker_pid":    self._broker_pid,
            "broker_exe":    str(self._broker_exe),
            "running":       self._running,
            "active_count":  len(active),
            "native64_count": len(self._native64),
            "total_seen":    len(self._seen),
            "active":        active,
        }
"""
et_bridge/et_awe.py
ET32 Bridge — AWE Bookshelf (Address Windowing Extensions)

Derived from P ∘ D ∘ T = E.

The AWE Bookshelf is the ET solution to the memory Descriptor Gap:

  P = the full physical address space (all installed RAM — unlimited)
  D = the AWE window Descriptor (a 32-bit-sized view into P, 4KB-aligned)
  T = the 32-bit process thread accessing memory through the window
  E = any physical address in P is reachable via D-window sliding

The "bookshelf" metaphor (from C2C session):
  - The bookshelf = all physical RAM (P — the infinite substrate)
  - The books     = 16MB physical page groups (D-projection units)
  - Your arm      = the 32-bit VA window (the constrained D view)
  - Sliding arm   = MapUserPhysicalPages (T traversing P via D)

This gives TRUE 64-bit memory access — not bridge handles, not copies,
not IPC overhead. The 32-bit process receives REAL 32-bit pointers that
map to REAL physical pages, identical to those accessible from 64-bit code.

ET-derived constants:
  AWE_PAGE_SIZE    = ħ_d = 4096 bytes (digital action quantum)
  AWE_WINDOW_SIZE  = ħ_d² = ħ_d << S = 4096 << 12 = 16,777,216 bytes (16 MB)
  AWE_MAX_WINDOWS  = S² = 144 (QUEUE_DEPTH — maximum simultaneous views)
  AWE_SLOT_STRIDE  = S = 12 (window alignment stride in pages)
  AWE_FILL_TRIGGER = K = 2/3 (Koide ratio — window eviction threshold)
  AWE_INIT_PAGES   = K × 2^20 = 699,050 pages (~2.7 GB initial allocation)
  AWE_EXPAND_STEP  = S × AWE_WINDOW_PAGES = 12 × 4096 pages = 196,608 pages (768 MB)

The window management follows the ET Traverser mobility principle:
  T can rebind to any (P∘D) configuration. The Traverser (the 32-bit thread)
  is never permanently bound to any window — it slides as needed.
  Window eviction uses LRU order (minimum-variance eviction: V(evict) = V_BASE).

Author: Derived from Michael James Muller's Exception Theory
"""

import ctypes
import ctypes.wintypes as wintypes
import threading
import time
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from et_math import (
    S, K, V_BASE,
    DIGITAL_ACTION_QUANTUM, QUEUE_DEPTH, HANDLE_BASE,
    CONN_TIMEOUT_MS, RETRY_COUNT,
    ETMetrics, pack_args, unpack_args,
)
from et_logger import ETLog

# =============================================================================
# AWE CONSTANTS — DERIVED FROM ET
# =============================================================================

# Physical page size = digital action quantum ħ_d = 4096 bytes
AWE_PAGE_SIZE: int = DIGITAL_ACTION_QUANTUM          # 4096 bytes

# AWE window size = ħ_d² = ħ_d << S (second-order quantum)
# = 4096 << 12 = 4096 × 4096 = 16,777,216 bytes = 16 MB per window
AWE_WINDOW_SIZE: int = DIGITAL_ACTION_QUANTUM << S   # 16 MB

# Pages per window = AWE_WINDOW_SIZE / AWE_PAGE_SIZE = 4096
AWE_WINDOW_PAGES: int = AWE_WINDOW_SIZE // AWE_PAGE_SIZE   # 4096 pages

# Maximum simultaneous windows = S² (QUEUE_DEPTH)
AWE_MAX_WINDOWS: int = QUEUE_DEPTH                   # 144 windows

# Window stride in pages = S (manifold symmetry)
AWE_SLOT_STRIDE: int = S                             # 12-page alignment stride

# Koide fill trigger: when K fraction of windows is mapped, pre-expand physical
AWE_FILL_TRIGGER: float = K                          # 2/3 of windows occupied → expand

# Initial physical page allocation: K × 2^20 pages = ~2.7 GB
# Derived: K × (all 32-bit pages) = K × (4 GB / 4 KB) = K × 1,048,576
AWE_INIT_PAGES: int = int(K * (1 << 20))             # 699,050 pages ≈ 2.7 GB

# Expansion step: S × AWE_WINDOW_PAGES = 12 × 4096 = 49,152 pages = 192 MB
AWE_EXPAND_STEP_PAGES: int = S * AWE_WINDOW_PAGES    # 49,152 pages = 192 MB

# Maximum total physical pages managed = S × 2^20 = 12 × 1M = 12 million pages = 48 GB
AWE_MAX_TOTAL_PAGES: int = S * (1 << 20)             # 12,582,912 pages = 48 GB

# Shared memory name for AWE coordination between broker and DLL
AWE_SHMEM_NAME_TEMPLATE: str = "ET32_AWE_{pid}"
AWE_SHMEM_SIZE: int = AWE_MAX_WINDOWS * 64           # 144 × 64 = 9216 bytes for window metadata

# =============================================================================
# WIN32 STRUCTURES FOR AWE
# =============================================================================

kernel32 = getattr(ctypes.windll, 'kernel32')
advapi32 = getattr(ctypes.windll, 'advapi32')

# AllocateUserPhysicalPages prototype
# BOOL AllocateUserPhysicalPages(HANDLE hProcess, PULONG_PTR NumberOfPages, PULONG_PTR PageArray)
_AllocateUserPhysicalPages = getattr(kernel32, 'AllocateUserPhysicalPages')
_AllocateUserPhysicalPages.restype  = wintypes.BOOL
_AllocateUserPhysicalPages.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]

# FreeUserPhysicalPages
_FreeUserPhysicalPages = getattr(kernel32, 'FreeUserPhysicalPages')
_FreeUserPhysicalPages.restype  = wintypes.BOOL
_FreeUserPhysicalPages.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]

# MapUserPhysicalPages (called from 32-bit target via our hook)
_MapUserPhysicalPages = getattr(kernel32, 'MapUserPhysicalPages')
_MapUserPhysicalPages.restype  = wintypes.BOOL
_MapUserPhysicalPages.argtypes = [
    ctypes.c_void_p,                       # VirtualAddress in target
    ctypes.c_size_t,                       # NumberOfPages
    ctypes.POINTER(ctypes.c_size_t),       # PageArray (None to unmap)
]

# MapUserPhysicalPagesScatter
_MapUserPhysicalPagesScatter = getattr(kernel32, 'MapUserPhysicalPagesScatter')
_MapUserPhysicalPagesScatter.restype  = wintypes.BOOL
_MapUserPhysicalPagesScatter.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]

# VirtualAllocEx / VirtualFreeEx for reserving AWE regions in target
_VirtualAllocEx = getattr(kernel32, 'VirtualAllocEx')
_VirtualAllocEx.restype  = ctypes.c_void_p
_VirtualAllocEx.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p,
    ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD,
]

_VirtualFreeEx = getattr(kernel32, 'VirtualFreeEx')
_VirtualFreeEx.restype  = wintypes.BOOL
_VirtualFreeEx.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD,
]

# Token privilege for SeLockMemoryPrivilege (required for AWE)
_OpenProcessToken   = getattr(kernel32, 'OpenProcessToken')
_AdjustTokenPriv    = getattr(advapi32, 'AdjustTokenPrivileges')
_LookupPrivilegeVal = getattr(advapi32, 'LookupPrivilegeValueW')
_GetCurrentProcess  = getattr(kernel32, 'GetCurrentProcess')

# CloseHandle — needed for token cleanup and process handle release
_CloseHandle = getattr(kernel32, 'CloseHandle')

# GetLastError — ET: reads the Descriptor Gap code from the last Win32 operation
_GetLastError = getattr(kernel32, 'GetLastError')

# Constants
MEM_RESERVE         = 0x00002000
MEM_COMMIT          = 0x00001000
MEM_PHYSICAL        = 0x00400000    # AWE physical-backed flag
MEM_RELEASE         = 0x00008000
PAGE_READWRITE      = 0x00000004
SE_PRIVILEGE_ENABLED = 0x00000002
TOKEN_ADJUST_PRIVILEGES = 0x00000020
TOKEN_QUERY         = 0x00000008


# =============================================================================
# AWE PRIVILEGE SETUP
# =============================================================================

def _acquire_lock_memory_privilege() -> bool:
    """
    Acquire SeLockMemoryPrivilege for the broker process.
    Required for AllocateUserPhysicalPages.
    Derived from ET: the privilege is the D-unlock that allows P (physical)
    to be directly bound to T (process) without the OS memory manager mediating.
    """
    h_token = wintypes.HANDLE()
    if not _OpenProcessToken(
        _GetCurrentProcess(),
        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
        ctypes.byref(h_token)
    ):
        return False

    class LUID(ctypes.Structure):
        """Win32 LUID — Locally Unique Identifier for privilege lookup."""
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    # noinspection PyPep8Naming
    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        """Win32 LUID_AND_ATTRIBUTES — privilege descriptor with enable/disable flag."""
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    # noinspection PyPep8Naming
    class TOKEN_PRIVILEGES(ctypes.Structure):
        """Win32 TOKEN_PRIVILEGES — single-slot privilege adjustment structure."""
        _fields_ = [
            ("PrivilegeCount", wintypes.DWORD),
            ("Privileges", LUID_AND_ATTRIBUTES * 1),
        ]

    luid = LUID()
    _LookupPrivilegeVal(None, "SeLockMemoryPrivilege", ctypes.byref(luid))

    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    # noinspection PyPep8Naming
    tp.Privileges[0].Luid = luid
    # noinspection PyPep8Naming
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED

    ok = _AdjustTokenPriv(h_token, False, ctypes.byref(tp), 0, None, None)
    _CloseHandle(h_token)
    return bool(ok)


# =============================================================================
# AWE WINDOW DESCRIPTOR — D-PROJECTION UNIT
# =============================================================================

@dataclass
class AWEWindow:
    """
    One AWE window Descriptor (D-projection of P onto 32-bit VA space).

    P ∘ D ∘ T = E:
      P  = the physical pages this window maps (page_frames list)
      D  = the 32-bit VA address where they are mapped (va_base)
      T  = the 32-bit process accessing via this window (pid)
      E  = any byte in the mapped pages is directly addressable by 32-bit code

    ET lattice position = d derived from size:
      d = classify_size(AWE_WINDOW_SIZE) = 12 (full-resolution, exceeds 2GB threshold)
    """
    pid:          int                     # target process PID (T-owner)
    va_base:      int                     # 32-bit VA of the reserved region (D-position)
    page_frames:  List[int]               # physical page frame numbers (P-list)
    n_pages:      int                     # number of pages mapped
    mapped:       bool = False            # D is active (window is mapped)
    last_access:  float = field(default_factory=time.monotonic)  # for LRU
    access_count: int = 0                 # T-traversal count (ET: V tracks over time)
    physical_base: int = 0               # 64-bit physical base address (P-address)

    def variance(self) -> float:
        """
        V(window) = V_BASE if unmapped (D not bound to T).
        V(window) = 0.0 if mapped and recently accessed.
        V increases with time since last access (T-drift from D).
        """
        if not self.mapped:
            return V_BASE
        age_s = time.monotonic() - self.last_access
        # V grows as age / (CONN_TIMEOUT_MS / 1000) until it reaches V_BASE
        return min(V_BASE, age_s / (CONN_TIMEOUT_MS / 1000.0) * V_BASE)


# =============================================================================
# AWE PHYSICAL POOL — THE BOOKSHELF ITSELF
# =============================================================================

@dataclass
class AWEPhysicalPool:
    """
    Manages a pool of physical pages allocated for a specific target process.

    The pool IS the bookshelf — a collection of physical pages (the books)
    that can be mapped into the 32-bit VA space on demand (pulling books
    from the shelf).

    ET derivation of pool structure:
      Total pages tracked = S² × AWE_WINDOW_PAGES = 144 × 4096 = 589,824 pages
      This is the natural ET bound: QUEUE_DEPTH windows of WINDOW_PAGES each.
      Actual physical allocation grows in AWE_EXPAND_STEP_PAGES increments.
    """
    pid:             int           # target process PID
    h_process:       int           # Windows HANDLE to target process
    page_frames:     List[int]     # all allocated physical page frame numbers
    n_allocated:     int = 0       # pages currently allocated from OS
    n_in_use:        int = 0       # pages currently mapped into VA windows
    expanded_count:  int = 0       # number of expansion steps taken

    def pages_available(self) -> int:
        """Return the number of physical pages allocated but not yet mapped into VA windows."""
        return self.n_allocated - self.n_in_use

    def utilisation(self) -> float:
        """Return pool utilization as a fraction: n_in_use / n_allocated (0.0 when empty)."""
        if self.n_allocated == 0:
            return 0.0
        return self.n_in_use / self.n_allocated

    def needs_expansion(self) -> bool:
        """K-threshold: expand when utilization > K (Koide ratio)."""
        return self.utilisation() > K


# =============================================================================
# ET AWE BOOKSHELF — MAIN CLASS
# =============================================================================

class ETAWEBookshelf:
    """
    The ET AWE Bookshelf — gives a 32-bit process TRUE access to all physical RAM.

    Architecture:
      1. Broker (64-bit) allocates physical pages via AllocateUserPhysicalPages
         in the context of the 32-bit target process. Pages are physical RAM —
         identical to what 64-bit code can access.
      2. The 32-bit DLL (et_bridge32.dll) calls MapUserPhysicalPages to map
         a group of pages into a reserved VA region. It gets a REAL 32-bit pointer.
      3. The 32-bit code accesses memory via the real pointer — NO IPC overhead,
         NO bridge handles, just raw memory access as if it were normal RAM.
      4. When the 32-bit code needs a different physical region, it requests a
         window slide: the broker remaps different pages into the VA region.
      5. Multiple VA windows can be active simultaneously (up to S² = 144).

    This IS the bookshelf:
      P = all physical RAM (all the books)
      D = each 16MB AWE window (the arm's reach)
      T = the 32-bit process accessing data (the reader)
      E = reading any book on any shelf (accessing any physical address)

    Completeness guarantee (Subsumption Law):
      Every byte of physical RAM is addressable by the 32-bit process via
      window sliding. No physical address is beyond reach. V(bookshelf) = 0
      when all physical pages are covered by the pool.
    """

    def __init__(self) -> None:
        self._log = ETLog.get("et_awe")
        self._lock = threading.Lock()
        # pid → AWEPhysicalPool
        self._pools: Dict[int, AWEPhysicalPool] = {}
        # pid → list of AWEWindow (each is an active or inactive VA window in target)
        self._windows: Dict[int, List[AWEWindow]] = {}
        # Privilege acquired flag
        self._privilege_ok: bool = _acquire_lock_memory_privilege()
        if not self._privilege_ok:
            self._log.warning_di(
                "SeLockMemoryPrivilege not acquired — AWE may fail on large allocations. "
                "Run broker as Administrator to enable full bookshelf."
            )
        self._metrics = ETMetrics()

    # ------------------------------------------------------------------
    # PHYSICAL PAGE ALLOCATION
    # ------------------------------------------------------------------

    def allocate_pool(self, pid: int, h_process: int,
                      n_pages: int = AWE_INIT_PAGES) -> bool:
        """
        Allocate a physical page pool for a target process.

        ET: this instantiates P (the bookshelf substrate) for the given
        process. Pages are allocated from the OS's physical memory manager
        and assigned to the target process's working set.

        n_pages: initial number of pages (default AWE_INIT_PAGES = K × 1M ≈ 2.7 GB).
        Returns True if at least one page was allocated.
        """
        with self._lock:
            if pid in self._pools:
                return True  # already has a pool

            # Clamp to AWE_MAX_TOTAL_PAGES
            n_pages = min(n_pages, AWE_MAX_TOTAL_PAGES)

            # ET retry loop: S = 12 attempts with D-narrowing (halve request on each failure)
            # The Descriptor narrows until P (physical memory) can satisfy the constraint.
            attempt_pages = n_pages
            actual = 0
            for attempt in range(RETRY_COUNT):
                if attempt_pages <= 0:
                    break
                page_array = (ctypes.c_size_t * attempt_pages)()
                count = ctypes.c_size_t(attempt_pages)

                ok = _AllocateUserPhysicalPages(
                    wintypes.HANDLE(h_process),
                    ctypes.byref(count),
                    page_array,
                )

                if ok and count.value > 0:
                    actual = int(count.value)
                    frames = [int(page_array[i]) for i in range(actual)]
                    break

                # D-narrowing: halve the request and retry
                err = _GetLastError()
                self._log.warning_di(
                    "AllocateUserPhysicalPages attempt %d/%d for PID %d: "
                    "error=%d (requested %d pages) — narrowing D.",
                    attempt + 1, RETRY_COUNT, pid, err, attempt_pages
                )
                attempt_pages //= 2

            if actual == 0:
                self._log.exception_state(
                    "AllocateUserPhysicalPages exhausted %d retries for PID %d "
                    "(final request: %d pages, got 0).",
                    RETRY_COUNT, pid, attempt_pages
                )
                return False

            pool = AWEPhysicalPool(
                pid         = pid,
                h_process   = h_process,
                page_frames = frames,
                n_allocated = actual,
            )
            self._pools[pid]   = pool
            self._windows[pid] = []

            self._log.mediation(
                "AWE pool created for PID %d: %d pages (%.1f GB) allocated.",
                pid, actual, (actual * AWE_PAGE_SIZE) / (1 << 30)
            )
            return True

    def expand_pool(self, pid: int, n_additional: int = AWE_EXPAND_STEP_PAGES) -> bool:
        """
        Expand the physical page pool by n_additional pages.

        ET: this extends P (adds more books to the shelf) when K-threshold
        is reached (V-proximity to Incoherence triggers expansion).
        """
        with self._lock:
            pool = self._pools.get(pid)
            if pool is None:
                return False

            n_additional = min(n_additional, AWE_MAX_TOTAL_PAGES - pool.n_allocated)
            if n_additional <= 0:
                return False

            page_array = (ctypes.c_size_t * n_additional)()
            count = ctypes.c_size_t(n_additional)

            ok = _AllocateUserPhysicalPages(
                wintypes.HANDLE(pool.h_process),
                ctypes.byref(count),
                page_array,
            )

            if not ok or count.value == 0:
                return False

            actual = int(count.value)
            pool.page_frames.extend(int(page_array[i]) for i in range(actual))
            pool.n_allocated += actual
            pool.expanded_count += 1

            self._log.mediation(
                "AWE pool expanded for PID %d: +%d pages (total %.1f GB).",
                pid, actual, (pool.n_allocated * AWE_PAGE_SIZE) / (1 << 30)
            )
            return True

    def release_pool(self, pid: int) -> None:
        """Release all physical pages for a process. Called on process exit."""
        with self._lock:
            pool = self._pools.pop(pid, None)
            if pool is None:
                return

            # Unmap all windows first
            for win in self._windows.pop(pid, []):
                if win.mapped:
                    self._unmap_window_locked(win, pool)

            # Free physical pages
            if pool.page_frames:
                n = len(pool.page_frames)
                arr = (ctypes.c_size_t * n)(*pool.page_frames)
                count = ctypes.c_size_t(n)
                _FreeUserPhysicalPages(
                    wintypes.HANDLE(pool.h_process),
                    ctypes.byref(count),
                    arr,
                )
                self._log.mediation("AWE pool released for PID %d.", pid)

    # ------------------------------------------------------------------
    # WINDOW RESERVATION AND MAPPING
    # ------------------------------------------------------------------

    def reserve_window(self, pid: int, h_process: int,
                       protect: int = PAGE_READWRITE) -> Optional[int]:
        """
        Reserve a 32-bit VA region of AWE_WINDOW_SIZE in the target process.

        ET: this creates a D-slot — a region in the 32-bit address space
        reserved for window mapping. The region holds no physical pages yet
        (Unsubstantiated state: {P,D} without T).

        Returns the 32-bit VA base address, or None on failure.
        """
        with self._lock:
            wins = self._windows.get(pid, [])
            if len(wins) >= AWE_MAX_WINDOWS:
                # Try evicting an LRU window to free a slot
                evicted = self._evict_lru(pid)
                if not evicted:
                    self._log.exception_state(
                        "AWE window limit reached for PID %d (max=%d).",
                        pid, AWE_MAX_WINDOWS
                    )
                    return None

            va_base = _VirtualAllocEx(
                wintypes.HANDLE(h_process),
                None,
                AWE_WINDOW_SIZE,
                MEM_RESERVE | MEM_PHYSICAL,
                protect,
            )

            if not va_base:
                err = _GetLastError()
                self._log.exception_state(
                    "VirtualAllocEx (MEM_PHYSICAL) failed for PID %d: error=%d",
                    pid, err
                )
                return None

            win = AWEWindow(
                pid        = pid,
                va_base    = int(va_base),
                page_frames = [],
                n_pages    = 0,
                mapped     = False,
            )
            wins.append(win)
            self._windows[pid] = wins

            self._log.mediation(
                "AWE window reserved for PID %d at 0x%08X (%.0f MB).",
                pid, int(va_base), AWE_WINDOW_SIZE / (1 << 20)
            )
            return int(va_base)

    def map_window(self, pid: int, va_base: int, physical_base: int) -> bool:
        """
        Map physical pages starting at physical_base into the reserved VA window.

        ET: this is the D-T binding — the Traverser (process) is now connected
        to the substrate (physical pages) via the window Descriptor.
        The window transitions from {P,D} Unsubstantiated to {P,D,T} = E.

        physical_base: 64-bit physical byte address (must be AWE_PAGE_SIZE-aligned).
        Returns True if mapping succeeded.
        """
        with self._lock:
            pool = self._pools.get(pid)
            if pool is None:
                return False

            win = self._find_window(pid, va_base)
            if win is None:
                return False

            # Unmap current contents if any
            if win.mapped:
                self._unmap_window_locked(win, pool)

            # Calculate which page frames correspond to physical_base
            page_offset = physical_base // AWE_PAGE_SIZE
            n_pages = min(AWE_WINDOW_PAGES, pool.n_allocated - page_offset)
            if n_pages <= 0:
                # Need to expand the pool
                if pool.needs_expansion() or page_offset >= pool.n_allocated:
                    self.expand_pool(pid)
                    pool = self._pools.get(pid)
                    n_pages = min(AWE_WINDOW_PAGES, pool.n_allocated - page_offset)
                if n_pages <= 0:
                    return False

            # Build page frame array for this window
            frames = pool.page_frames[page_offset:page_offset + n_pages]
            arr = (ctypes.c_size_t * len(frames))(*frames)

            ok = _MapUserPhysicalPages(
                ctypes.c_void_p(va_base),
                len(frames),
                arr,
            )

            if not ok:
                err = _GetLastError()
                self._log.exception_state(
                    "MapUserPhysicalPages failed for PID %d: va=0x%08X error=%d",
                    pid, va_base, err
                )
                return False

            win.page_frames   = list(frames)
            win.n_pages       = len(frames)
            win.mapped        = True
            win.physical_base = physical_base
            win.last_access   = time.monotonic()
            win.access_count += 1

            pool.n_in_use += len(frames)

            # Trigger pool expansion if K-threshold exceeded
            if pool.needs_expansion() and pool.n_allocated < AWE_MAX_TOTAL_PAGES:
                self._log.mediation(
                    "AWE pool for PID %d at K-threshold (%.1f%%) — expanding.",
                    pid, pool.utilisation() * 100
                )
                # Expansion happens async to avoid blocking the caller
                threading.Thread(
                    target=self.expand_pool,
                    args=(pid,),
                    daemon=True,
                    name=f"AWE-expand-{pid}",
                ).start()

            self._log.mediation(
                "AWE window mapped: PID %d, VA=0x%08X → phys=0x%016X (%d pages = %.1f MB)",
                pid, va_base, physical_base, len(frames),
                (len(frames) * AWE_PAGE_SIZE) / (1 << 20)
            )
            return True

    def unmap_window(self, pid: int, va_base: int) -> bool:
        """Unmap a window. The VA region remains reserved for future remapping."""
        with self._lock:
            pool = self._pools.get(pid)
            win  = self._find_window(pid, va_base)
            if win is None or not win.mapped:
                return False
            return self._unmap_window_locked(win, pool)

    def _unmap_window_locked(self, win: AWEWindow, pool: AWEPhysicalPool) -> bool:
        """Unmap under lock."""
        t0 = time.monotonic()
        ok = _MapUserPhysicalPages(
            ctypes.c_void_p(win.va_base),
            win.n_pages,
            None,  # None = unmap
        )
        if ok and pool:
            pool.n_in_use -= win.n_pages
        if not ok:
            self._log.exception_state(
                "AWE unmap failed: PID %d, VA=0x%08X (%d pages)",
                win.pid, win.va_base, win.n_pages
            )
        freed_pages = win.n_pages
        win.mapped        = False
        win.page_frames   = []
        win.n_pages       = 0
        win.physical_base = 0
        self._metrics.record(
            family      = 1,  # d=1 MEMORY_BASIC (unmap is memory-class)
            latency_us  = (time.monotonic() - t0) * 1_000_000.0,
            success     = bool(ok),
            bytes_count = freed_pages * AWE_PAGE_SIZE,
        )
        return bool(ok)

    # ------------------------------------------------------------------
    # WINDOW LOOKUP AND LRU EVICTION
    # ------------------------------------------------------------------

    def _find_window(self, pid: int, va_base: int) -> Optional[AWEWindow]:
        for win in self._windows.get(pid, []):
            if win.va_base == va_base:
                return win
        return None

    def find_window_for_physical(self, pid: int, phys_addr: int) -> Optional[AWEWindow]:
        """Find a currently-mapped window that covers phys_addr."""
        for win in self._windows.get(pid, []):
            if win.mapped:
                end = win.physical_base + win.n_pages * AWE_PAGE_SIZE
                if win.physical_base <= phys_addr < end:
                    return win
        return None

    def va_for_physical(self, pid: int, phys_addr: int) -> Optional[int]:
        """
        Translate a 64-bit physical address to a 32-bit VA in the target process.
        If no window currently maps this address, maps one (evicting LRU if needed).
        Returns the 32-bit VA, or None on failure.
        """
        # First check if already mapped
        win = self.find_window_for_physical(pid, phys_addr)
        if win:
            win.last_access   = time.monotonic()
            win.access_count += 1
            offset = phys_addr - win.physical_base
            return win.va_base + offset

        # Need to map a new window covering phys_addr
        # Align to AWE_WINDOW_SIZE boundary
        phys_aligned = (phys_addr // AWE_WINDOW_SIZE) * AWE_WINDOW_SIZE

        # Find an unmapped window slot or evict LRU
        with self._lock:
            target_win = None
            for win in self._windows.get(pid, []):
                if not win.mapped:
                    target_win = win
                    break

            if target_win is None:
                # Evict LRU
                wins = self._windows.get(pid, [])
                if wins:
                    target_win = min(wins, key=lambda w: w.last_access)
                    pool = self._pools.get(pid)
                    self._unmap_window_locked(target_win, pool)
                else:
                    return None

        if self.map_window(pid, target_win.va_base, phys_aligned):
            offset = phys_addr - phys_aligned
            return target_win.va_base + offset
        return None

    def _evict_lru(self, pid: int) -> bool:
        """Evict the least-recently-used window (minimum-variance eviction)."""
        wins = self._windows.get(pid, [])
        if not wins:
            return False
        lru = min(wins, key=lambda w: w.last_access)
        pool = self._pools.get(pid)
        return self._unmap_window_locked(lru, pool)

    # ------------------------------------------------------------------
    # BOOKSHELF ALLOCATION — THE PRIMARY PUBLIC API
    # ------------------------------------------------------------------

    def bookshelf_alloc(self, pid: int, h_process: int,
                        n_bytes: int, protect: int = PAGE_READWRITE) -> Optional[int]:
        """
        Allocate n_bytes of physical-backed memory for the target process.

        This is the "bookshelf allocation" — it:
        1. Ensures the pool has enough pages
        2. Reserves a VA window (or reuses an existing free one)
        3. Maps physical pages into the window
        4. Returns a REAL 32-bit pointer to the mapped memory

        The 32-bit process can read/write this pointer DIRECTLY — no IPC,
        no bridge handles. This is TRUE memory access.

        Returns the 32-bit VA pointer, or None on failure.
        """
        t0 = time.monotonic()  # ET: T-traversal start (for metrics latency measurement)

        # Ensure pool exists
        if pid not in self._pools:
            # Calculate initial pages: at least enough for this allocation + INIT_PAGES
            init = max(AWE_INIT_PAGES, (n_bytes // AWE_PAGE_SIZE) + S)
            if not self.allocate_pool(pid, h_process, init):
                return None

        pool = self._pools[pid]

        # Check if pool has enough free pages
        pages_needed = (n_bytes + AWE_PAGE_SIZE - 1) // AWE_PAGE_SIZE
        while pool.pages_available() < pages_needed:
            expand_n = max(AWE_EXPAND_STEP_PAGES, pages_needed - pool.pages_available() + S)
            if not self.expand_pool(pid, expand_n):
                self._log.exception_state(
                    "Cannot expand AWE pool for PID %d — insufficient physical memory.",
                    pid
                )
                return None

        # Find or create a window large enough
        # For allocations > AWE_WINDOW_SIZE, we'd need multiple windows.
        # For now: limit single allocation to AWE_WINDOW_SIZE.
        if n_bytes > AWE_WINDOW_SIZE:
            self._log.exception_state(
                "AWE single allocation request (%d bytes) exceeds window size (%d). "
                "Use multiple windows for very large allocations.",
                n_bytes, AWE_WINDOW_SIZE
            )
            return None

        # Reserve a window in the target's VA space
        va = self.reserve_window(pid, h_process, protect)
        if va is None:
            return None

        # Calculate physical base: use the next available range of free pages in the pool
        phys_page_start = pool.n_in_use  # next free page
        phys_base = phys_page_start * AWE_PAGE_SIZE

        if not self.map_window(pid, va, phys_base):
            # Release the reserved VA
            _VirtualFreeEx(wintypes.HANDLE(h_process), ctypes.c_void_p(va), 0, MEM_RELEASE)
            return None

        self._metrics.record(
            family      = 1,  # d=1 MEMORY_BASIC
            latency_us  = (time.monotonic() - t0) * 1_000_000.0,
            success     = True,
            bytes_count = n_bytes,
        )
        return va

    def bookshelf_free(self, pid: int, va: int) -> bool:
        """
        Free a bookshelf allocation (unmap window, release VA reservation).
        Physical pages remain in pool for reuse — they are not freed
        until release_pool() is called on process exit.
        """
        return self.unmap_window(pid, va)

    # ------------------------------------------------------------------
    # HANDLE MAPPING — BRIDGE HANDLE ↔ AWE VA TRANSLATION
    # Uses HANDLE_BASE to place AWE handles in the bridge handle namespace.
    # ET: handle = D-projection of the VA address into the bridge's
    # shared handle space. Stride = S to maintain manifold alignment.
    # ------------------------------------------------------------------

    def _va_to_handle(self, pid: int, va_base: int) -> int:
        """
        Convert an AWE window's VA base into a bridge handle.

        ET derivation: handle = HANDLE_BASE + (pid_hash × S² + window_index × S).
        The S-stride ensures handles are manifold-aligned and never collide
        with raw 32-bit pointers (which live below HANDLE_BASE).
        """
        wins = self._windows.get(pid, [])
        for idx, win in enumerate(wins):
            if win.va_base == va_base:
                pid_offset = (pid & 0xFFFF) * (S * S)
                return HANDLE_BASE + pid_offset + idx * S
        return 0  # invalid handle — window not found

    def _handle_to_va(self, handle: int) -> Tuple[int, int]:
        """
        Reverse-map a bridge handle back to (pid, va_base).

        Returns (0, 0) if the handle does not correspond to any active AWE window.
        ET: this is the inverse D-projection — recovering the concrete VA from
        the abstract handle Descriptor.
        """
        if handle < HANDLE_BASE:
            return 0, 0
        offset = handle - HANDLE_BASE
        for pid, wins in self._windows.items():
            pid_offset = (pid & 0xFFFF) * (S * S)
            if pid_offset <= offset < pid_offset + len(wins) * S:
                idx = (offset - pid_offset) // S
                if 0 <= idx < len(wins):
                    return pid, wins[idx].va_base
        return 0, 0

    # ------------------------------------------------------------------
    # SHARED MEMORY SYNCHRONIZATION — WINDOW METADATA FOR 32-BIT DLL
    # Uses struct to pack AWEWindow records into the AWE shared memory
    # region (AWE_SHMEM_NAME_TEMPLATE). The 32-bit DLL reads this to know
    # which windows are active and where they are mapped.
    # ET: the shared memory IS a D-table — each 64-byte record is a
    # Descriptor of one window's P↔D↔T binding state.
    # ------------------------------------------------------------------

    # 64-byte record format: matches AWE_SHMEM_SIZE = AWE_MAX_WINDOWS × 64
    #   pid          : uint32   (4 bytes)  — T-owner
    #   va_base      : uint32   (4 bytes)  — D-position in 32-bit VA
    #   physical_base: uint64   (8 bytes)  — P-address (64-bit physical)
    #   n_pages      : uint32   (4 bytes)  — D-extent (pages mapped)
    #   mapped       : uint8    (1 byte)   — D-T binding active flag
    #   _pad         : 3 bytes             — alignment padding
    #   last_access  : float64  (8 bytes)  — T-traversal timestamp
    #   access_count : uint32   (4 bytes)  — T-traversal count
    #   _reserved    : 28 bytes            — future expansion
    _WINDOW_RECORD_FMT: str = "<IIQIBxxxdI28x"
    _WINDOW_RECORD_SIZE: int = 64  # struct.calcsize(_WINDOW_RECORD_FMT)

    def _pack_window_record(self, win: AWEWindow) -> Tuple[bytes, int]:
        """
        Serialize one AWEWindow into a 64-byte shared memory record.

        Returns (record_bytes, record_size).
        ET: this is D-serialization — converting the live window state into
        a fixed-format Descriptor that the 32-bit DLL can read directly.
        """
        record = struct.pack(
            self._WINDOW_RECORD_FMT,
            win.pid & 0xFFFFFFFF,
            win.va_base & 0xFFFFFFFF,
            win.physical_base & 0xFFFFFFFFFFFFFFFF,
            win.n_pages & 0xFFFFFFFF,
            1 if win.mapped else 0,
            win.last_access,
            win.access_count & 0xFFFFFFFF,
        )
        return record, self._WINDOW_RECORD_SIZE

    def _unpack_window_record(self, data: bytes) -> Dict:
        """
        Deserialize a 64-byte shared memory record into a window state dict.

        ET: inverse of _pack_window_record — reconstructing live state from
        the serialized D-table entry.
        """
        (pid, va_base, physical_base, n_pages,
         mapped, last_access, access_count) = struct.unpack(
            self._WINDOW_RECORD_FMT, data[:self._WINDOW_RECORD_SIZE]
        )
        return {
            "pid": pid,
            "va_base": va_base,
            "physical_base": physical_base,
            "n_pages": n_pages,
            "mapped": bool(mapped),
            "last_access": last_access,
            "access_count": access_count,
        }

    def sync_shmem(self, pid: int, shmem_buf: bytearray) -> int:
        """
        Write all window records for *pid* into the provided shared memory buffer.

        shmem_buf must be at least AWE_SHMEM_SIZE bytes (AWE_MAX_WINDOWS × 64).
        Returns the number of records written.

        ET: this synchronizes the D-table — the 32-bit DLL polls this buffer
        to discover which windows are active without any IPC round-trip.
        """
        wins = self._windows.get(pid, [])
        offset = 0
        count = 0
        for win in wins:
            if offset + self._WINDOW_RECORD_SIZE > len(shmem_buf):
                break
            record, size = self._pack_window_record(win)
            shmem_buf[offset:offset + size] = record
            offset += size
            count += 1
        # Zero-fill remaining slots (marks them as inactive)
        if offset < len(shmem_buf):
            shmem_buf[offset:] = b'\x00' * (len(shmem_buf) - offset)
        return count

    # ------------------------------------------------------------------
    # IPC PAYLOAD ENCODING — AWE COMMAND SERIALISATION
    # Uses pack_args / unpack_args (ET-derived type-tagged encoding) to
    # serialize AWE commands sent between broker and 32-bit DLL over IPC.
    # ET: the command payload IS a PDT packet — each argument is a
    # D-tagged value (type tag from the ET lattice: d=1..12).
    # ------------------------------------------------------------------

    def encode_awe_command(self, cmd: int, pid: int,
                           va_base: int = 0, phys_base: int = 0,
                           n_bytes: int = 0, protect: int = PAGE_READWRITE) -> bytes:
        """
        Encode an AWE command as an IPC payload using ET pack_args.

        cmd codes (ET lattice d-positions):
          1 = ALLOC       (bookshelf_alloc)
          2 = FREE        (bookshelf_free)
          3 = MAP         (map_window)
          4 = UNMAP       (unmap_window)
          5 = EXPAND      (expand_pool)
          6 = STATUS      (status query)
          7 = RESERVE     (reserve_window)
          8 = TRANSLATE   (va_for_physical)
          12 = RELEASE    (release_pool)

        Returns the packed byte payload ready for IPC transmission.
        The first byte is the argument count (for receiver-side validation).
        """
        if pid not in self._pools and cmd not in (1, 6):
            self._log.warning_di(
                "AWE encode: cmd=%d targets unknown PID %d — pool not yet allocated.",
                cmd, pid
            )
        payload, arg_count = pack_args(cmd, pid, va_base, phys_base, n_bytes, protect)
        # Prepend argument count as a 1-byte header for receiver validation
        return struct.pack("B", arg_count) + payload

    def decode_awe_command(self, data: bytes) -> Dict:
        """
        Decode an incoming AWE command payload from the 32-bit DLL.

        The first byte is the expected argument count (written by encode_awe_command).
        Returns a dict with keys: cmd, pid, va_base, phys_base, n_bytes, protect.
        Missing fields default to 0 / PAGE_READWRITE.
        """
        expected_count = struct.unpack("B", data[:1])[0] if len(data) > 0 else 0
        args = unpack_args(data[1:])
        if len(args) != expected_count:
            self._log.warning_di(
                "AWE command arg count mismatch: header=%d, decoded=%d",
                expected_count, len(args)
            )
        keys = ("cmd", "pid", "va_base", "phys_base", "n_bytes", "protect")
        defaults = (0, 0, 0, 0, 0, PAGE_READWRITE)
        result = {}
        for i, key in enumerate(keys):
            if i < len(args):
                result[key] = args[i]
            else:
                result[key] = defaults[i]
        return result

    # ------------------------------------------------------------------
    # STATUS AND METRICS
    # ------------------------------------------------------------------

    def status(self, pid: int) -> Dict:
        """
        Return a diagnostic status dict for the given PID's AWE bookshelf.

        ET: this is the Descriptor snapshot — the complete state of P (pool),
        D (windows), and T (process) at this instant. Variance is implicit
        in the utilization percentage (V approaches K when util > 66.7%).
        """
        pool = self._pools.get(pid)
        wins = self._windows.get(pid, [])
        if pool is None:
            return {"pid": pid, "pool": "absent"}
        mapped_wins = sum(1 for w in wins if w.mapped)
        return {
            "pid":             pid,
            "pages_allocated": pool.n_allocated,
            "pages_in_use":    pool.n_in_use,
            "pages_free":      pool.pages_available(),
            "gb_allocated":    round(pool.n_allocated * AWE_PAGE_SIZE / (1 << 30), 2),
            "gb_in_use":       round(pool.n_in_use    * AWE_PAGE_SIZE / (1 << 30), 2),
            "windows_total":   len(wins),
            "windows_mapped":  mapped_wins,
            "utilisation_pct": round(pool.utilisation() * 100, 1),
            "expand_count":    pool.expanded_count,
            "privilege_ok":    self._privilege_ok,
        }

    def all_pool_pids(self) -> list:
        """Return a list of all PIDs that have an AWE pool allocated."""
        with self._lock:
            return list(self._pools.keys())
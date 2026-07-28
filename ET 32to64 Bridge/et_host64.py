"""
et_bridge/et_host64.py
ET32 Bridge — 64-bit Operation Host

Executes all 12 command families on behalf of the 32-bit client.
Derived from P ∘ D ∘ T = E.

This is the right-hand side of the bridge: the 64-bit broker receives an
ETPacket, dispatches to the correct family handler, performs the real 64-bit
OS operation, and returns an ETPacket response.

ET derivation of the host:
  P = the 64-bit OS (substrate of all 64-bit resources)
  D = the command families (Descriptors of what is requested)
  T = this host process (traverser through the 64-bit OS)
  E = the completed operation (Exception: {P,D,T} grounded)

Handler dispatch table:
  family 1  → _handle_memory_basic    (VirtualAlloc, HeapAlloc, ReadMem, WriteMem)
  family 2  → _handle_memory_map      (CreateFileMapping, MapViewOfFile)
  family 3  → _handle_thread_ops      (CreateThread, SuspendThread, ResumeThread)
  family 4  → _handle_dll_ops         (LoadLibrary, GetProcAddress, DLL call)
  family 5  → _handle_process_ops     (CreateProcess, OpenProcess, GetSystemInfo)
  family 6  → _handle_registry_ops    (RegOpenKey 64-bit, RegQuery, RegSet)
  family 7  → _handle_graphics_ops    (D3D12, Vulkan, VRAM alloc)
  family 8  → _handle_file_ops        (Large file >4GB, mmap, seek/read/write)
  family 9  → _handle_sync_ops        (CreateEvent, WaitForSingleObject, Mutex)
  family 10 → _handle_net_ops         (Socket with 64-bit buffers)
  family 11 → _handle_python_ops      (64-bit Python embedding)
  family 12 → _handle_compound_ops    (Batch / Atomic / Rollback)
  family 0  → _handle_control         (PING, HANDSHAKE, SHUTDOWN, STATUS)

All memory allocated through this host is tracked in the HandleTable so
32-bit handles can be reconstructed from 64-bit addresses.

ET Constants applied per handler:
  S = 12 — used for batch sizing and recursion depth in COMPOUND_OPS
  K = 2/3 — stability threshold for operation retry logic
  V = 1/12 — minimum granularity for latency measurement
  ħ_d = 4096 — page size for memory operations (digital action quantum)
"""

import ctypes
import ctypes.util
import ctypes.wintypes
import os
import sys
import struct
import threading
import time
import subprocess
import winreg
import socket
import importlib
from typing import Dict, Optional, Any, List, Tuple, Callable

from et_awe import (
    ETAWEBookshelf,
    AWE_PAGE_SIZE, AWE_WINDOW_SIZE, AWE_WINDOW_PAGES,
    AWE_MAX_WINDOWS, AWE_INIT_PAGES,
)

from et_math import (
    S, K, V_BASE, DIGITAL_ACTION_QUANTUM, IPC_BUFFER_SIZE,
    PDT_HEADER_SIZE, CONN_TIMEOUT_MS, QUEUE_DEPTH,
    ADDR64_BASE, HANDLE_BASE, HANDLE_MAX,
    ETPacket, CmdFamily, CmdCode, ETMetrics, ETHandleMath,
    pack_args, unpack_args,
    sublattice_incoherence, combined_sublattice, gaussian_prime_type,
    tightness, coherence_depth,
    SUBLATTICE_LCM_TABLE, D_MAX, N_MAX_IMAGINARY,
)
from et_handle import HandleTable, HandleEntry
from et_logger import ETLog
from et_heaven import ETHeavenGate
from et_wow64 import ETWow64Hook
from et_errors import (
    ETOperationError, ETWindowsAPIError, ETInjectionError,
    ETIPCError, ETPacketError, ETAWEError, ETHookError,
    ETDispatchError, ETConfigError, ETHandleError,
    ETErrorSeverity, win32_check, win32_check_handle,
    ntstatus_check, et_context, safe_call,
    record_error, record_op, get_registry,
)


# ============================================================================
# WINDOWS API CONSTANTS
# ============================================================================

MEM_COMMIT              = 0x1000
MEM_RESERVE             = 0x2000
MEM_RELEASE             = 0x8000
MEM_DECOMMIT            = 0x4000
PAGE_READWRITE          = 0x04
PAGE_EXECUTE_READWRITE  = 0x40
PAGE_EXECUTE_READ       = 0x20
PAGE_NOACCESS           = 0x01
PROCESS_ALL_ACCESS      = 0x1F0FFF
PROCESS_VM_READ         = 0x0010
PROCESS_VM_WRITE        = 0x0020
PROCESS_VM_OPERATION    = 0x0008
THREAD_ALL_ACCESS       = 0x1F03FF
KEY_READ                = 0x20019
KEY_WRITE               = 0x20006
KEY_ALL_ACCESS          = 0xF003F
KEY_WOW64_64KEY         = 0x0100   # Force 64-bit registry access
REG_SZ                  = 1
REG_DWORD               = 4
REG_QWORD               = 11
REG_BINARY              = 3
SYNCHRONIZE             = 0x00100000
FILE_MAP_ALL_ACCESS     = 0x000F001F
PAGE_READWRITE_MAP      = 0x04
SOCKET_ERROR            = -1
SOCK_STREAM             = 1
AF_INET                 = 2
IPPROTO_TCP             = 6

# Large page flag (requires SeLockMemoryPrivilege — attempt, fall back gracefully)
MEM_LARGE_PAGES         = 0x20000000
FILE_ATTRIBUTE_NORMAL   = 0x00000080

kernel32  = getattr(ctypes.windll, 'kernel32')
ntdll     = getattr(ctypes.windll, 'ntdll')
advapi32  = getattr(ctypes.windll, 'advapi32')
ws2_32    = getattr(ctypes.windll, 'ws2_32')

# ============================================================================
# ET ERROR CODES — derived from V-space
# ============================================================================

ET_ERR_OK               = 0x00000000
ET_ERR_INVALID_ARGS     = 0xE0000001
ET_ERR_ALLOC_FAIL       = 0xE0000002
ET_ERR_ACCESS_DENIED    = 0xE0000003
ET_ERR_NOT_FOUND        = 0xE0000004
ET_ERR_HANDLE_FULL      = 0xE0000005
ET_ERR_INCOHERENT       = 0xE0000006
ET_ERR_TIMEOUT          = 0xE0000007
ET_ERR_OS_ERROR         = 0xE0000008
ET_ERR_PYTHON_ERROR     = 0xE0000009
ET_ERR_UNSUPPORTED      = 0xE000000A


# ============================================================================
# WINDOWS STRUCTURES FOR OPERATIONS
# ============================================================================

# noinspection PyPep8Naming
class SECURITY_ATTRIBUTES(ctypes.Structure):
    """Win32 SECURITY_ATTRIBUTES — D-container for object security descriptors."""
    _fields_ = [
        ("nLength",              ctypes.wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle",       ctypes.wintypes.BOOL),
    ]


# noinspection PyPep8Naming
class SYSTEM_INFO(ctypes.Structure):
    """Win32 SYSTEM_INFO — P-substrate descriptor for processor and memory layout."""
    class _DUMMYUNIONNAME(ctypes.Union):
        class _DUMMYSTRUCTNAME(ctypes.Structure):
            _fields_ = [
                ("wProcessorArchitecture", ctypes.wintypes.WORD),
                ("wReserved",              ctypes.wintypes.WORD),
            ]
        _fields_ = [
            ("dwOemId",       ctypes.wintypes.DWORD),
            ("_DUMMYSTRUCTNAME", _DUMMYSTRUCTNAME),
        ]
    _fields_ = [
        ("_DUMMYUNIONNAME",         _DUMMYUNIONNAME),
        ("dwPageSize",              ctypes.wintypes.DWORD),
        ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p),
        ("dwActiveProcessorMask",   ctypes.POINTER(ctypes.c_ulong)),
        ("dwNumberOfProcessors",    ctypes.wintypes.DWORD),
        ("dwProcessorType",         ctypes.wintypes.DWORD),
        ("dwAllocationGranularity", ctypes.wintypes.DWORD),
        ("wProcessorLevel",         ctypes.wintypes.WORD),
        ("wProcessorRevision",      ctypes.wintypes.WORD),
    ]


# noinspection PyPep8Naming
class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    """Win32 MEMORY_BASIC_INFORMATION — D-descriptor for a virtual memory region's state."""
    _fields_ = [
        ("BaseAddress",       ctypes.c_void_p),
        ("AllocationBase",    ctypes.c_void_p),
        ("AllocationProtect", ctypes.wintypes.DWORD),
        ("PartitionId",       ctypes.wintypes.WORD),
        ("RegionSize",        ctypes.c_size_t),
        ("State",             ctypes.wintypes.DWORD),
        ("Protect",           ctypes.wintypes.DWORD),
        ("Type",              ctypes.wintypes.DWORD),
    ]


# noinspection PyPep8Naming
class PROCESS_INFORMATION(ctypes.Structure):
    """Win32 PROCESS_INFORMATION — T-traverser identifiers returned by CreateProcess."""
    _fields_ = [
        ("hProcess",    ctypes.wintypes.HANDLE),
        ("hThread",     ctypes.wintypes.HANDLE),
        ("dwProcessId", ctypes.wintypes.DWORD),
        ("dwThreadId",  ctypes.wintypes.DWORD),
    ]


class STARTUPINFOW(ctypes.Structure):
    """Win32 STARTUPINFOW — D-descriptor for new process startup configuration."""
    _fields_ = [
        ("cb",              ctypes.wintypes.DWORD),
        ("lpReserved",      ctypes.wintypes.LPWSTR),
        ("lpDesktop",       ctypes.wintypes.LPWSTR),
        ("lpTitle",         ctypes.wintypes.LPWSTR),
        ("dwX",             ctypes.wintypes.DWORD),
        ("dwY",             ctypes.wintypes.DWORD),
        ("dwXSize",         ctypes.wintypes.DWORD),
        ("dwYSize",         ctypes.wintypes.DWORD),
        ("dwXCountChars",   ctypes.wintypes.DWORD),
        ("dwYCountChars",   ctypes.wintypes.DWORD),
        ("dwFillAttribute", ctypes.wintypes.DWORD),
        ("dwFlags",         ctypes.wintypes.DWORD),
        ("wShowWindow",     ctypes.wintypes.WORD),
        ("cbReserved2",     ctypes.wintypes.WORD),
        ("lpReserved2",     ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput",       ctypes.wintypes.HANDLE),
        ("hStdOutput",      ctypes.wintypes.HANDLE),
        ("hStdError",       ctypes.wintypes.HANDLE),
    ]


# =============================================================================
# MODULE-LEVEL CTYPES STRUCTURES — shared across ETHost64 methods
# =============================================================================

# _FindFILETIME and _FindWin32FindData are defined at module level so that:
#   (a) _FindWin32FindData._fields_ can reference _FindFILETIME by name in
#       module scope — PyCharm resolves it correctly (inner-class sibling names
#       are not tracked by PyCharm's static analyzer, causing false unresolved
#       reference errors on the _fields_ list entries).
#   (b) Both find methods share one definition — Issue 10 (duplicate ctypes
#       structs) resolved correctly: one Descriptor, two usages.
#   (c) PEP 8 CapWords satisfied: _FindWin32FindData replaces the prior
#       _FindWIN32_FIND_DATA (screaming-case violation).
#   _Find prefix avoids collision with other FILETIME/WIN32 structs in this
#   module (e.g. _handle_file_getattr defines its own local WIN32_FILE_ATTRIBUTE_DATA).


class _FindFILETIME(ctypes.Structure):
    """Win32 FILETIME — 64-bit timestamp as low/high DWORD pair (directory enumeration)."""
    _fields_ = [
        ("dwLow",  ctypes.wintypes.DWORD),
        ("dwHigh", ctypes.wintypes.DWORD),
    ]


class _FindWin32FindData(ctypes.Structure):
    """Win32 WIN32_FIND_DATAW — D-descriptor for a directory enumeration entry."""
    _fields_ = [
        ("dwFileAttributes",   ctypes.wintypes.DWORD),
        ("ftCreationTime",     _FindFILETIME),
        ("ftLastAccessTime",   _FindFILETIME),
        ("ftLastWriteTime",    _FindFILETIME),
        ("nFileSizeHigh",      ctypes.wintypes.DWORD),
        ("nFileSizeLow",       ctypes.wintypes.DWORD),
        ("dwReserved0",        ctypes.wintypes.DWORD),
        ("dwReserved1",        ctypes.wintypes.DWORD),
        ("cFileName",          ctypes.c_wchar * 260),
        ("cAlternateFileName", ctypes.c_wchar * 14),
    ]


# ============================================================================
# ET HOST 64 — main dispatcher
# ============================================================================

class ETHost64:
    """
    64-bit operation host. Receives ETPacket requests, executes the 64-bit
    OS calls, and returns ETPacket responses.

    This is the P-component of the broker process: P = 64-bit OS substrate,
    D = command families, T = this host instance.

    Thread safety: all family handlers acquire per-family locks where needed.
    The handle table is already thread-safe internally.

    Usage:
        host = ETHost64(handle_table, metrics)
        response = host.dispatch(request_packet)
    """

    def __init__(self, handle_table: HandleTable, metrics: Optional[ETMetrics] = None):
        self._table   = handle_table
        self._awe: Optional[ETAWEBookshelf] = None   # set by ETBridgeAPI after construction
        self._metrics = metrics or ETMetrics()
        self._log     = ETLog.get("et_host64")
        self._heaven  = ETHeavenGate()
        self._broker_pid = os.getpid()

        # Per-family locks (one per lattice position = S = 12)
        self._family_locks: Dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(1, S + 1)
        }

        # Python interpreter state for PY_OPS
        self._py_initialized = False
        self._py_lock = threading.Lock()
        self._py_globals: Dict[int, Any] = {}

        # Winsock init
        self._winsock_initialized = False
        self._init_winsock()

        # WOW64 universal hook — set by ETBridgeAPI after construction
        self._wow64: Optional[ETWow64Hook] = None

        # GPU device tracking — maps PID → (device_ptr, context_ptr, device_type)
        # device_type: 0=D3D9, 1=D3D11
        # Populated by GPU_CREATE_DEVICE, consumed by GPU_ALLOC_VRAM / GPU_MAP_VRAM
        self._gpu_devices: Dict[int, tuple] = {}
        self._gpu_lock = threading.Lock()

        # Dispatch table: family → handler method
        self._dispatch_table: Dict[int, Callable[[ETPacket], ETPacket]] = {
            CmdFamily.MEMORY_BASIC : self._handle_memory_basic,
            CmdFamily.MEMORY_MAP   : self._handle_memory_map,
            CmdFamily.THREAD_OPS   : self._handle_thread_ops,
            CmdFamily.DLL_OPS      : self._handle_dll_ops,
            CmdFamily.PROCESS_OPS  : self._handle_process_ops,
            CmdFamily.REGISTRY_OPS : self._handle_registry_ops,
            CmdFamily.GRAPHICS_OPS : self._handle_graphics_ops,
            CmdFamily.FILE_OPS     : self._handle_file_ops,
            CmdFamily.SYNC_OPS     : self._handle_sync_ops,
            CmdFamily.NET_OPS      : self._handle_net_ops,
            CmdFamily.PYTHON_OPS   : self._handle_python_ops,
            CmdFamily.COMPOUND_OPS : self._handle_compound_ops,
            0                      : self._handle_control,
        }

        self._log.info("ETHost64 initialised: broker PID=%d", self._broker_pid)

    # -------------------------------------------------------------------------
    # PUBLIC ACCESSORS — for CLI and diagnostics (no protected-member access)
    # -------------------------------------------------------------------------

    @property
    def heaven(self) -> Optional[ETHeavenGate]:
        """Public accessor for the Heaven's Gate instance (or None)."""
        return self._heaven

    @property
    def wow64(self) -> Optional[ETWow64Hook]:
        """Public accessor for the WOW64 hook instance (or None)."""
        return self._wow64

    @property
    def awe(self) -> Optional[ETAWEBookshelf]:
        """Public accessor for the AWE Bookshelf instance (or None)."""
        return self._awe

    @property
    def metrics(self) -> ETMetrics:
        """Public accessor for the metrics instance."""
        return self._metrics

    @property
    def handle_table(self) -> HandleTable:
        """Public accessor for the handle table."""
        return self._table

    # -------------------------------------------------------------------------
    # MAIN DISPATCH ENTRY POINT
    # -------------------------------------------------------------------------

    def dispatch(self, pkt: ETPacket) -> ETPacket:
        """
        Dispatch an incoming ETPacket to the correct family handler.
        Returns a response ETPacket.

        This is the ET Exception production:
          P∘D∘T = E
          P = 64-bit OS
          D = pkt.cmd_family (Descriptor)
          T = this method call (Traverser)
          E = returned response packet
        """
        t0 = time.monotonic()
        family = pkt.cmd_family

        # Handle control codes (family=0 or specific control codes)
        if pkt.cmd_code in (
            CmdCode.CTRL_PING, CmdCode.CTRL_HANDSHAKE,
            CmdCode.CTRL_SHUTDOWN, CmdCode.CTRL_STATUS, CmdCode.CTRL_ACK
        ):
            family = 0

        handler = self._dispatch_table.get(family)
        if handler is None:
            return self._make_error(pkt, ET_ERR_UNSUPPORTED,
                                    f"Unknown family {family}")

        try:
            with et_context("dispatch", et_pid=pkt.source_pid, et_family=family,
                            et_code=pkt.cmd_code, reraise=True,
                            log_fn=self._log.incoherence):
                response = handler(pkt)
            latency  = (time.monotonic() - t0) * 1_000_000
            # ET variance of this operation: V = latency / max_latency_us
            # V < V_BASE (1/12) → excellent; V ≥ K (2/3) → approaching ∂I
            v_op = latency / (CONN_TIMEOUT_MS * 1000) if CONN_TIMEOUT_MS > 0 else 0.0
            # Level 1 𝒜_I — single-value tightness evaluation
            # Map V(op) to effective cents: ε_eff = (V/K) × 50¢
            # At V=0: ε=0¢ → tightness=1.0 (perfect)
            # At V=K: ε=50¢ → tightness=K (exactly at ∂I)
            eps_eff = (v_op / K) * 50.0 if K > 0 else 0.0
            t_op = tightness(eps_eff)
            cd_op = coherence_depth(eps_eff)
            if v_op >= K:
                self._log.warning_di(
                    "dispatch: V(op)=%.4f ≥ K=%.4f (approaching ∂I) "
                    "tightness=%.4f coherence_depth=%.4f ε_eff=%.2f¢ "
                    "family=%d code=0x%02X latency=%.1fμs PID=%d",
                    v_op, K, t_op, cd_op, eps_eff,
                    family, pkt.cmd_code, latency, pkt.source_pid)
            elif v_op >= V_BASE:
                self._log.mediation(
                    "dispatch: V(op)=%.4f ≥ V_BASE=%.4f "
                    "tightness=%.4f coherence_depth=%.4f "
                    "family=%d code=0x%02X latency=%.1fμs PID=%d",
                    v_op, V_BASE, t_op, cd_op,
                    family, pkt.cmd_code, latency, pkt.source_pid)
            self._metrics.record(family, latency, True)
            record_op()
            return response
        except Exception as exc:
            self._log.incoherence("Handler exception family=%d code=0x%02X: %s",
                                   family, pkt.cmd_code, exc)
            # Record as ETIPCError — from the client's perspective, dispatch failure
            # means the IPC round-trip failed to produce a valid response
            record_error(ETIPCError(
                f"dispatch failed: family={family} code=0x{pkt.cmd_code:02X} — {exc}",
                pid=pkt.source_pid,
                et_family=family,
                et_code=pkt.cmd_code,
                severity=ETErrorSeverity.BOUNDARY,
            ))
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    # -------------------------------------------------------------------------
    # CONTROL HANDLER (family = 0)
    # -------------------------------------------------------------------------

    def _handle_control(self, pkt: ETPacket) -> ETPacket:
        code = pkt.cmd_code

        if code == CmdCode.CTRL_PING:
            return self._make_ok(pkt, pkt.source_pid, int(time.monotonic() * 1e6))

        elif code == CmdCode.CTRL_HANDSHAKE:
            # Validate handshake source — PID 0 indicates misconfigured client
            if not pkt.source_pid:
                record_error(ETConfigError(
                    "HANDSHAKE from PID 0 — client must report a valid PID",
                    et_pid=0, et_family=0, et_code=CmdCode.CTRL_HANDSHAKE,
                ))
            # Respond with bridge version and broker PID
            return self._make_ok(pkt, self._broker_pid, S, int(K * 1000))

        elif code == CmdCode.CTRL_STATUS:
            summary = self._metrics.summary()
            # Include error registry health in the status report
            registry = get_registry()
            summary['error_registry'] = registry.summary()
            # AWE bookshelf configuration — ET-derived constants
            summary['awe_config'] = {
                'window_size': AWE_WINDOW_SIZE,         # ħ_d² = 16 MB per window
                'window_pages': AWE_WINDOW_PAGES,       # 4096 pages per window
                'max_windows': AWE_MAX_WINDOWS,         # S² = 144 simultaneous windows
                'init_pages': AWE_INIT_PAGES,           # K × 2²⁰ ≈ 699K initial pages
                'bookshelf_active': self._awe is not None,
            }
            info_str = str(summary)
            payload, count = pack_args(info_str)
            return self._make_response(
                pkt, CmdCode.CTRL_ACK, ETPacket.FLAG_RESPONSE, count, payload
            )

        elif code == CmdCode.CTRL_SHUTDOWN:
            self._log.info("Shutdown requested by PID %d", pkt.source_pid)
            return self._make_ok(pkt, 0)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, "Unknown control code")

    # -------------------------------------------------------------------------
    # FAMILY 1: MEMORY BASIC
    # -------------------------------------------------------------------------

    def _handle_memory_basic(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.MEMORY_BASIC]:

            if code == CmdCode.GLOBAL_MEM_STATUS:
                return self._handle_global_mem_status(pkt)
            elif code == CmdCode.NATIVE_SYS_INFO:
                return self._handle_native_sys_info(pkt)
            elif code == CmdCode.CLOSE_HANDLE64:
                return self._handle_close_handle64(pkt)
            elif code == CmdCode.DUPLICATE_HANDLE64:
                return self._handle_duplicate_handle64(pkt)

            if code == CmdCode.VIRT_ALLOC:
                # args: (address_hint: uint64, size: uint64, alloc_type: uint32, protect: uint32)
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "VIRT_ALLOC needs size")
                addr_hint  = args[0] if len(args) > 0 else 0
                size       = args[1] if len(args) > 1 else DIGITAL_ACTION_QUANTUM
                alloc_type = args[2] if len(args) > 2 else (MEM_COMMIT | MEM_RESERVE)
                protect    = args[3] if len(args) > 3 else PAGE_READWRITE

                # Round size up to ħ_d multiple (digital action quantum)
                size = ((size + DIGITAL_ACTION_QUANTUM - 1) // DIGITAL_ACTION_QUANTUM) * DIGITAL_ACTION_QUANTUM

                ptr = getattr(kernel32, 'VirtualAlloc')(
                    ctypes.c_void_p(addr_hint),
                    ctypes.c_size_t(size),
                    ctypes.wintypes.DWORD(alloc_type),
                    ctypes.wintypes.DWORD(protect)
                )
                if not ptr:
                    return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                            f"VirtualAlloc failed: {getattr(kernel32, 'GetLastError')()}")

                addr64  = ctypes.cast(ptr, ctypes.c_void_p).value
                # Classify allocation: addr64 ≥ ADDR64_BASE (4GB) means true 64-bit
                # This is the D-gap the bridge was built to close
                if addr64 >= ADDR64_BASE:
                    self._log.info(
                        "VIRT_ALLOC: true 64-bit allocation at 0x%016X (%d bytes) PID=%d",
                        addr64, size, pkt.source_pid)
                handle  = self._table.allocate(
                    addr64, size, protect, CmdFamily.MEMORY_BASIC,
                    tag=f"VirtAlloc_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, addr64, size)

            elif code == CmdCode.VIRT_FREE:
                # args: (handle: uint32,)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "VIRT_FREE needs handle")
                handle = args[0]
                addr64 = self._table.resolve(handle)
                if addr64 is None:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} unknown")
                ok = getattr(kernel32, 'VirtualFree')(
                    ctypes.c_void_p(addr64),
                    ctypes.c_size_t(0),
                    ctypes.wintypes.DWORD(MEM_RELEASE)
                )
                if ok:
                    self._table.release(handle)
                    return self._make_ok(pkt, 1)
                return self._make_error(pkt, ET_ERR_OS_ERROR,
                                        f"VirtualFree failed: {getattr(kernel32, 'GetLastError')()}")

            elif code == CmdCode.VIRT_PROTECT:
                # args: (handle, new_protect)
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "VIRT_PROTECT needs handle+protect")
                handle      = args[0]
                new_protect = args[1]
                addr64      = self._table.resolve(handle)
                if addr64 is None:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} unknown")
                entry: Optional[HandleEntry] = self._table.get_entry(handle)
                size        = entry.size if entry else DIGITAL_ACTION_QUANTUM
                old_protect = ctypes.wintypes.DWORD(0)
                ok = getattr(kernel32, 'VirtualProtect')(
                    ctypes.c_void_p(addr64),
                    ctypes.c_size_t(size),
                    ctypes.wintypes.DWORD(new_protect),
                    ctypes.byref(old_protect)
                )
                if ok:
                    return self._make_ok(pkt, old_protect.value)
                return self._make_error(pkt, ET_ERR_OS_ERROR,
                                        f"VirtualProtect failed: {getattr(kernel32, 'GetLastError')()}")

            elif code == CmdCode.VIRT_QUERY:
                # args: (handle or addr64, [fault_addr for AWE remap])
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "VIRT_QUERY needs address")
                addr_or_handle = args[0]
                fault_va = args[1] if len(args) > 1 else 0
                pid = pkt.source_pid
                # AWE remap request from VEH handler:
                # args = (va_base_of_window, fault_va) — remap physical pages
                if self._awe is not None and fault_va and int(fault_va) < 0x100000000:
                    win = self._awe.find_window_for_physical(pid, int(addr_or_handle))
                    if win is not None:
                        h_proc = getattr(kernel32, 'OpenProcess')(0x1FFFFF, False, pid)
                        if h_proc:
                            remap_ok = self._awe.map_window(pid, int(addr_or_handle),
                                                             win.physical_base)
                            getattr(kernel32, 'CloseHandle')(h_proc)
                            if remap_ok:
                                return self._make_ok(pkt, 0)  # 0 = pass-through
                addr64 = self._table.resolve(addr_or_handle) or addr_or_handle
                mbi = MEMORY_BASIC_INFORMATION()
                sz  = getattr(kernel32, 'VirtualQuery')(
                    ctypes.c_void_p(addr64),
                    ctypes.byref(mbi),
                    ctypes.c_size_t(ctypes.sizeof(mbi))
                )
                if sz == 0:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"VirtualQuery failed: {getattr(kernel32, 'GetLastError')()}")
                return self._make_ok(pkt,
                    mbi.BaseAddress or 0,
                    mbi.RegionSize,
                    mbi.State,
                    mbi.Protect,
                    mbi.Type
                )

            elif code in (CmdCode.HEAP_ALLOC, CmdCode.HEAP_FREE):
                pid = pkt.source_pid

                if code == CmdCode.HEAP_ALLOC:
                    # Detect AWE bookshelf init signal:
                    # DLL sends HEAP_ALLOC with args=(target_pid,) after initializing.
                    # args[0] == pid AND size == 0 OR size == pid → AWE init signal.
                    is_awe_init = (len(args) == 1 and int(args[0]) == pid)
                    if is_awe_init and self._awe is not None:
                        # AWE init signal — broker side: ensure bookshelf pool exists.
                        # The ETBridgeAPI already called allocate_pool; this is a
                        # confirmation round-trip.  Just ACK.
                        return self._make_ok(pkt, 1)

                    # Regular HEAP_ALLOC — attempt AWE bookshelf first.
                    size    = args[0] if args else DIGITAL_ACTION_QUANTUM
                    protect = args[1] if len(args) > 1 else PAGE_READWRITE
                    awe_flag = args[2] if len(args) > 2 else 0
                    size    = ((int(size) + AWE_PAGE_SIZE - 1) // AWE_PAGE_SIZE) * AWE_PAGE_SIZE

                    # AWE window validation: single allocation should fit within
                    # AWE_WINDOW_SIZE (ħ_d² = 16 MB) — larger allocations span
                    # multiple windows (up to AWE_MAX_WINDOWS = S² = 144)
                    if int(size) > AWE_WINDOW_SIZE:
                        pages_needed = int(size) // AWE_PAGE_SIZE
                        windows_needed = (pages_needed + AWE_WINDOW_PAGES - 1) // AWE_WINDOW_PAGES
                        self._log.mediation(
                            "HEAP_ALLOC: size=%d exceeds AWE_WINDOW_SIZE=%d, "
                            "needs %d windows (max %d) PID=%d",
                            int(size), AWE_WINDOW_SIZE,
                            windows_needed, AWE_MAX_WINDOWS, pid)

                    if self._awe is not None and int(awe_flag) & 0x00400000:  # MEM_PHYSICAL flag
                        # AWE bookshelf allocation — gives REAL 32-bit pointer.
                        h_proc = getattr(kernel32, 'OpenProcess')(0x1FFFFF, False, pid)
                        if h_proc:
                            va = self._awe.bookshelf_alloc(pid, int(h_proc), size, int(protect))
                            getattr(kernel32, 'CloseHandle')(h_proc)
                            if va:
                                # Return the real 32-bit VA — no handle translation needed!
                                return self._make_ok(pkt, va)
                            else:
                                # AWE bookshelf_alloc failed — record ETAWEError
                                record_error(ETAWEError(
                                    f"bookshelf_alloc failed: size={size} protect=0x{int(protect):X}",
                                    pid=pid,
                                    et_family=CmdFamily.MEMORY_BASIC,
                                    et_code=CmdCode.HEAP_ALLOC,
                                ))
                        else:
                            # OpenProcess failed for AWE — record ETAWEError
                            record_error(ETAWEError(
                                f"OpenProcess for AWE bookshelf failed: PID={pid}",
                                pid=pid,
                                et_family=CmdFamily.MEMORY_BASIC,
                                et_code=CmdCode.HEAP_ALLOC,
                                os_error=getattr(kernel32, 'GetLastError')(),
                            ))
                        # Fall through to standard HeapAlloc

                    # Standard HeapAlloc in broker process
                    h_heap  = getattr(kernel32, 'GetProcessHeap')()
                    ptr     = getattr(kernel32, 'HeapAlloc')(h_heap, 0, ctypes.c_size_t(size))
                    if not ptr:
                        return self._make_error(pkt, ET_ERR_ALLOC_FAIL, "HeapAlloc failed")
                    addr64  = ctypes.cast(ptr, ctypes.c_void_p).value
                    handle  = self._table.allocate(
                        addr64, size, PAGE_READWRITE, CmdFamily.MEMORY_BASIC,
                        tag=f"HeapAlloc_{pid}"
                    )
                    return self._make_ok(pkt, handle, addr64, size)

                else:  # HEAP_FREE
                    handle = args[0] if args else 0
                    # Check if it's an AWE VA (real 32-bit pointer, not a bridge handle)
                    if self._awe is not None and int(handle) < 0x80000000:
                        if self._awe.bookshelf_free(pid, int(handle)):
                            return self._make_ok(pkt, 1)
                    addr64 = self._table.resolve(handle)
                    if addr64 is None:
                        return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} unknown")
                    h_heap = getattr(kernel32, 'GetProcessHeap')()
                    ok     = getattr(kernel32, 'HeapFree')(h_heap, 0, ctypes.c_void_p(addr64))
                    if ok:
                        self._table.release(handle)
                        return self._make_ok(pkt, 1)
                    return self._make_error(pkt, ET_ERR_OS_ERROR, "HeapFree failed")

            elif code == CmdCode.READ_MEM:
                # args: (src_addr64, size) — read from 64-bit address into response payload
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "READ_MEM needs addr+size")
                src   = self._table.resolve(args[0]) or args[0]
                size  = min(int(args[1]), IPC_BUFFER_SIZE - PDT_HEADER_SIZE - 64)
                buf   = ctypes.create_string_buffer(size)
                read  = ctypes.c_size_t(0)
                ok    = getattr(kernel32, 'ReadProcessMemory')(
                    getattr(kernel32, 'GetCurrentProcess')(),
                    ctypes.c_void_p(src),
                    buf,
                    size,
                    ctypes.byref(read)
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, f"ReadProcessMemory failed: {getattr(kernel32, 'GetLastError')()}")
                raw_data   = bytes(buf[:read.value])
                payload, c = pack_args(raw_data)
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

            elif code == CmdCode.WRITE_MEM:
                # args: (dst_addr64, data_bytes)
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "WRITE_MEM needs addr+data")
                dst  = self._table.resolve(args[0]) or args[0]
                data = args[1] if isinstance(args[1], bytes) else str(args[1]).encode()
                written = ctypes.c_size_t(0)
                ok = getattr(kernel32, 'WriteProcessMemory')(
                    getattr(kernel32, 'GetCurrentProcess')(),
                    ctypes.c_void_p(dst),
                    data,
                    len(data),
                    ctypes.byref(written)
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, f"WriteProcessMemory failed: {getattr(kernel32, 'GetLastError')()}")
                return self._make_ok(pkt, written.value)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown MEMORY_BASIC code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 2: MEMORY MAP
    # -------------------------------------------------------------------------

    def _handle_memory_map(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.MEMORY_MAP]:

            if code == CmdCode.FILE_MAP_CREATE:
                # args: (h_file or -1 for pagefile, protect, size_high, size_low, name_or_none)
                h_file    = args[0] if len(args) > 0 else 0xFFFFFFFF  # INVALID_HANDLE_VALUE → page file
                protect   = args[1] if len(args) > 1 else PAGE_READWRITE_MAP
                size      = args[2] if len(args) > 2 else 0  # 0 = same as file
                name      = args[3] if len(args) > 3 else None

                # Resolve file handle if it's a bridge handle
                real_h_file = self._table.resolve(h_file) or h_file
                # INVALID_HANDLE_VALUE encoding
                if h_file in (0xFFFFFFFF, 0):
                    real_h_file = ctypes.wintypes.HANDLE(-1).value

                name_str = ctypes.c_wchar_p(name) if name else None
                h_map = getattr(kernel32, 'CreateFileMappingW')(
                    ctypes.wintypes.HANDLE(real_h_file),
                    None,
                    ctypes.wintypes.DWORD(protect),
                    ctypes.wintypes.DWORD(size >> 32),
                    ctypes.wintypes.DWORD(size & 0xFFFFFFFF),
                    name_str
                )
                if not h_map:
                    return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                            f"CreateFileMappingW failed: {getattr(kernel32, 'GetLastError')()}")
                map_addr64 = ctypes.cast(ctypes.c_void_p(h_map), ctypes.c_void_p).value
                handle = self._table.allocate(
                    map_addr64, size, protect, CmdFamily.MEMORY_MAP,
                    tag=f"FileMap_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, map_addr64)

            elif code == CmdCode.FILE_MAP_VIEW:
                # args: (map_handle, access, offset, size)
                map_handle = args[0] if args else 0
                access     = args[1] if len(args) > 1 else FILE_MAP_ALL_ACCESS
                offset     = args[2] if len(args) > 2 else 0
                map_size   = args[3] if len(args) > 3 else 0
                real_h_map = self._table.resolve(map_handle) or map_handle
                view_ptr = getattr(kernel32, 'MapViewOfFile')(
                    ctypes.wintypes.HANDLE(real_h_map),
                    ctypes.wintypes.DWORD(access),
                    ctypes.wintypes.DWORD(offset >> 32),
                    ctypes.wintypes.DWORD(offset & 0xFFFFFFFF),
                    ctypes.c_size_t(map_size)
                )
                if not view_ptr:
                    return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                            f"MapViewOfFile failed: {getattr(kernel32, 'GetLastError')()}")
                addr64  = ctypes.cast(view_ptr, ctypes.c_void_p).value
                handle  = self._table.allocate(
                    addr64, map_size, access, CmdFamily.MEMORY_MAP,
                    tag=f"MapView_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, addr64)

            elif code == CmdCode.FILE_MAP_CLOSE:
                handle = args[0] if args else 0
                addr64 = self._table.resolve(handle)
                if addr64 is None:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} unknown")
                # Determine if this is a view or a mapping handle
                entry = self._table.get_entry(handle)
                if entry and entry.tag and "MapView" in entry.tag:
                    ok = getattr(kernel32, 'UnmapViewOfFile')(ctypes.c_void_p(addr64))
                else:
                    ok = getattr(kernel32, 'CloseHandle')(ctypes.wintypes.HANDLE(addr64))
                if ok:
                    self._table.release(handle)
                return self._make_ok(pkt, int(bool(ok)))

            elif code == CmdCode.FILE_MAP_FLUSH:
                handle = args[0] if args else 0
                addr64 = self._table.resolve(handle) or handle
                size   = args[1] if len(args) > 1 else 0
                ok     = getattr(kernel32, 'FlushViewOfFile')(ctypes.c_void_p(addr64), ctypes.c_size_t(size))
                return self._make_ok(pkt, int(bool(ok)))

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown MEMORY_MAP code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 3: THREAD OPS
    # -------------------------------------------------------------------------

    def _handle_thread_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.THREAD_OPS]:

            if code == CmdCode.THREAD_CREATE:
                # args: (func_addr64, param_addr64, stack_size, flags)
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "THREAD_CREATE needs func+param")
                func_handle  = args[0]
                param_handle = args[1]
                stack_size   = args[2] if len(args) > 2 else 0
                flags        = args[3] if len(args) > 3 else 0

                func_addr64  = self._table.resolve(func_handle) or func_handle
                param_addr64 = self._table.resolve(param_handle) or param_handle

                # Use LPTHREAD_START_ROUTINE prototype
                # noinspection PyPep8Naming
                THREAD_PROC = ctypes.WINFUNCTYPE(ctypes.wintypes.DWORD, ctypes.c_void_p)
                func_ptr    = THREAD_PROC(func_addr64)

                h_thread = getattr(kernel32, 'CreateThread')(
                    None,                              # default security
                    ctypes.c_size_t(stack_size),       # stack size
                    func_ptr,                          # function
                    ctypes.c_void_p(param_addr64),     # parameter
                    ctypes.wintypes.DWORD(flags),      # creation flags
                    None                               # thread ID (don't care)
                )
                if not h_thread:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"CreateThread failed: {getattr(kernel32, 'GetLastError')()}")
                handle = self._table.allocate(
                    h_thread, 0, THREAD_ALL_ACCESS, CmdFamily.THREAD_OPS,
                    tag=f"Thread_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, h_thread)

            elif code in (CmdCode.THREAD_SUSPEND, CmdCode.THREAD_RESUME):
                handle   = args[0] if args else 0
                h_thread = self._table.resolve(handle) or handle
                if code == CmdCode.THREAD_SUSPEND:
                    prev = getattr(kernel32, 'SuspendThread')(ctypes.wintypes.HANDLE(h_thread))
                else:
                    prev = getattr(kernel32, 'ResumeThread')(ctypes.wintypes.HANDLE(h_thread))
                return self._make_ok(pkt, prev)

            elif code == CmdCode.THREAD_TERMINATE:
                handle   = args[0] if args else 0
                exit_code = args[1] if len(args) > 1 else 0
                h_thread  = self._table.resolve(handle) or handle
                ok        = getattr(kernel32, 'TerminateThread')(
                    ctypes.wintypes.HANDLE(h_thread),
                    ctypes.wintypes.DWORD(exit_code)
                )
                if ok:
                    self._table.release(handle)
                return self._make_ok(pkt, int(bool(ok)))

            elif code == CmdCode.THREAD_CONTEXT:
                # Return thread context as serialized bytes
                handle   = args[0] if args else 0
                h_thread = self._table.resolve(handle) or handle
                # CONTEXT structure is 1232 bytes on x64
                context_buf = ctypes.create_string_buffer(1232)
                # Set ContextFlags to CONTEXT_ALL = 0x10003F
                struct.pack_into("<I", context_buf, 48, 0x10003F)
                ok = getattr(kernel32, 'GetThreadContext')(
                    ctypes.wintypes.HANDLE(h_thread),
                    context_buf
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"GetThreadContext failed: {getattr(kernel32, 'GetLastError')()}")
                payload, c = pack_args(bytes(context_buf))
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

            elif code == CmdCode.THREAD_SET_CONTEXT:
                return self._handle_thread_set_context(pkt)
            elif code == CmdCode.THREAD_EXIT_CODE:
                return self._handle_thread_exit_code(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown THREAD_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 4: DLL OPS
    # -------------------------------------------------------------------------

    def _handle_dll_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.DLL_OPS]:

            if code == CmdCode.DLL_LOAD:
                # args: (dll_path: str, flags: uint32)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "DLL_LOAD needs path")
                dll_path = str(args[0])
                flags    = args[1] if len(args) > 1 else 0
                h_module = getattr(kernel32, 'LoadLibraryExW')(
                    dll_path,
                    None,
                    ctypes.wintypes.DWORD(flags)
                )
                if not h_module:
                    os_err = getattr(kernel32, 'GetLastError')()
                    record_error(ETInjectionError(
                        f"LoadLibraryExW('{dll_path}') failed",
                        pid=pkt.source_pid,
                        et_family=CmdFamily.DLL_OPS,
                        et_code=CmdCode.DLL_LOAD,
                        os_error=os_err,
                    ))
                    return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                            f"LoadLibraryExW('{dll_path}') failed: {os_err}")
                module_addr = ctypes.cast(h_module, ctypes.c_void_p).value
                handle = self._table.allocate(
                    module_addr, 0, 0, CmdFamily.DLL_OPS,
                    tag=f"Module_{os.path.basename(dll_path)}_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, module_addr)

            elif code == CmdCode.DLL_FREE:
                handle    = args[0] if args else 0
                mod_addr  = self._table.resolve(handle) or handle
                ok        = getattr(kernel32, 'FreeLibrary')(ctypes.wintypes.HMODULE(mod_addr))
                if ok:
                    self._table.release(handle)
                return self._make_ok(pkt, int(bool(ok)))

            elif code == CmdCode.DLL_GETPROC:
                # args: (module_handle, proc_name: str)
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "DLL_GETPROC needs module+name")
                mod_handle = args[0]
                proc_name  = str(args[1]).encode("ascii")
                mod_addr   = self._table.resolve(mod_handle) or mod_handle
                proc_ptr   = getattr(kernel32, 'GetProcAddress')(
                    ctypes.wintypes.HMODULE(mod_addr),
                    proc_name
                )
                if not proc_ptr:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                            f"GetProcAddress('{proc_name.decode()}') failed")
                proc_addr64 = ctypes.cast(proc_ptr, ctypes.c_void_p).value
                handle = self._table.allocate(
                    proc_addr64, 0, 0, CmdFamily.DLL_OPS,
                    tag=f"ProcAddr_{proc_name.decode()}_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, proc_addr64)

            elif code == CmdCode.DLL_CALL:
                # args: (func_handle, arg1, arg2, ..., argN)
                # Calls a 64-bit DLL function through ETHeavenGate
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "DLL_CALL needs func handle")
                func_handle = args[0]
                call_args   = [self._table.resolve(a) or a if isinstance(a, int) else a
                               for a in args[1:]]
                func_addr64 = self._table.resolve(func_handle) or func_handle
                result = self._heaven.call_64bit_function(func_addr64, call_args)
                if result is None:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"DLL_CALL failed for func 0x{func_addr64:016X}")
                return self._make_ok(pkt, result)

            elif code == CmdCode.DLL_LIST:
                # Returns list of loaded 64-bit module names and base addresses
                modules: List[str] = []
                # Walk the PEB LDR list to enumerate loaded modules
                # GetModuleHandleExW with NULL gives current module; snapshot approach simpler
                # Use CreateToolhelp32Snapshot for module enumeration
                # noinspection PyPep8Naming
                TH32CS_SNAPMODULE   = 0x00000008
                # noinspection PyPep8Naming
                TH32CS_SNAPMODULE32 = 0x00000010

                class MODULEENTRY32W(ctypes.Structure):
                    """Win32 MODULEENTRY32W — D-descriptor for a loaded module snapshot entry."""
                    _fields_ = [
                        ("dwSize",        ctypes.wintypes.DWORD),
                        ("th32ModuleID",  ctypes.wintypes.DWORD),
                        ("th32ProcessID", ctypes.wintypes.DWORD),
                        ("GlblcntUsage",  ctypes.wintypes.DWORD),
                        ("ProccntUsage",  ctypes.wintypes.DWORD),
                        ("modBaseAddr",   ctypes.POINTER(ctypes.c_byte)),
                        ("modBaseSize",   ctypes.wintypes.DWORD),
                        ("hModule",       ctypes.wintypes.HMODULE),
                        ("szModule",      ctypes.c_wchar * 256),
                        ("szExePath",     ctypes.c_wchar * 260),
                    ]

                h_snap = getattr(kernel32, 'CreateToolhelp32Snapshot')(
                    TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32,
                    ctypes.wintypes.DWORD(os.getpid())
                )
                if h_snap and h_snap != ctypes.wintypes.HANDLE(-1).value:
                    entry = MODULEENTRY32W()
                    entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
                    if getattr(kernel32, 'Module32FirstW')(h_snap, ctypes.byref(entry)):
                        while True:
                            base = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                            modules.append(f"{entry.szModule}=0x{base:016X}")
                            if not getattr(kernel32, 'Module32NextW')(h_snap, ctypes.byref(entry)):
                                break
                    getattr(kernel32, 'CloseHandle')(h_snap)
                module_list_str = "\n".join(modules)
                payload, c = pack_args(module_list_str)
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown DLL_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 5: PROCESS OPS
    # -------------------------------------------------------------------------

    def _handle_process_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.PROCESS_OPS]:

            if code == CmdCode.PROC_CREATE:
                # args: (command_line: str, flags: uint32, working_dir_or_None)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "PROC_CREATE needs cmdline")
                cmd_line   = str(args[0])
                flags      = int(args[1]) if len(args) > 1 else 0
                working_dir = str(args[2]) if len(args) > 2 else None

                si = STARTUPINFOW()
                si.cb = ctypes.sizeof(STARTUPINFOW)
                pi = PROCESS_INFORMATION()

                ok = getattr(kernel32, 'CreateProcessW')(
                    None,
                    cmd_line,
                    None, None,   # process/thread security
                    False,        # no handle inheritance
                    ctypes.wintypes.DWORD(flags),
                    None,         # inherit environment
                    ctypes.c_wchar_p(working_dir),
                    ctypes.byref(si),
                    ctypes.byref(pi)
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"CreateProcessW failed: {getattr(kernel32, 'GetLastError')()}")
                proc_handle = self._table.allocate(
                    pi.hProcess, 0, PROCESS_ALL_ACCESS, CmdFamily.PROCESS_OPS,
                    tag=f"Process_{pi.dwProcessId}"
                )
                getattr(kernel32, 'CloseHandle')(pi.hThread)  # we don't track the main thread handle
                return self._make_ok(pkt, proc_handle, pi.dwProcessId)

            elif code == CmdCode.PROC_OPEN:
                # args: (pid: uint32, access: uint32)
                pid    = int(args[0]) if args else pkt.source_pid
                access = int(args[1]) if len(args) > 1 else PROCESS_ALL_ACCESS
                h_proc = getattr(kernel32, 'OpenProcess')(
                    ctypes.wintypes.DWORD(access),
                    False,
                    ctypes.wintypes.DWORD(pid)
                )
                if not h_proc:
                    return self._make_error(pkt, ET_ERR_ACCESS_DENIED,
                                            f"OpenProcess({pid}) failed: {getattr(kernel32, 'GetLastError')()}")
                handle = self._table.allocate(
                    h_proc, 0, access, CmdFamily.PROCESS_OPS,
                    tag=f"ProcHandle_{pid}"
                )
                return self._make_ok(pkt, handle, h_proc)

            elif code == CmdCode.PROC_INFO:
                # GetSystemInfo — return 64-bit system info
                si = SYSTEM_INFO()
                getattr(kernel32, 'GetNativeSystemInfo')(ctypes.byref(si))
                max_addr = ctypes.cast(
                    si.lpMaximumApplicationAddress, ctypes.c_void_p
                ).value or 0
                return self._make_ok(pkt,
                    si.dwPageSize,
                    max_addr,                       # 64-bit max address
                    si.dwNumberOfProcessors,
                    si.dwAllocationGranularity
                )

            elif code == CmdCode.PROC_EXIT_CODE:
                return self._handle_proc_exit_code(pkt)
            elif code == CmdCode.PROC_TERMINATE:
                return self._handle_proc_terminate(pkt)
            elif code == CmdCode.PROC_ENUM:
                return self._handle_proc_enum(pkt)
            elif code == CmdCode.PROC_MODULES:
                return self._handle_proc_modules(pkt)
            elif code == CmdCode.PROC_WOW64_FS:
                return self._handle_proc_wow64_fs(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown PROCESS_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 6: REGISTRY OPS (64-bit bypass)
    # -------------------------------------------------------------------------

    def _handle_registry_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.REGISTRY_OPS]:
            # All registry operations use KEY_WOW64_64KEY to bypass WOW64 redirection

            # Dynamic hive map — discover ALL HKEY_* constants from winreg at runtime
            # ET: D-discovery via getattr() ensures no hive is missed (vs static list)
            # noinspection PyPep8Naming
            HIVE_MAP = {}
            for _hive_name in dir(winreg):
                if _hive_name.startswith('HKEY_'):
                    _hive_val = getattr(winreg, _hive_name, None)
                    if isinstance(_hive_val, int):
                        HIVE_MAP[_hive_val] = _hive_val

            if code == CmdCode.REG_OPEN64:
                # args: (hive: uint32, subkey: str, access: uint32)
                hive_id  = int(args[0]) if args else 0x80000002  # HKLM
                subkey   = str(args[1]) if len(args) > 1 else ""
                access   = int(args[2]) if len(args) > 2 else KEY_READ
                access  |= KEY_WOW64_64KEY  # force 64-bit access

                hive = HIVE_MAP.get(hive_id, winreg.HKEY_LOCAL_MACHINE)
                try:
                    hkey = winreg.OpenKey(hive, subkey, 0, access)
                    # Store the hkey handle value
                    hkey_val = hkey.handle
                    handle   = self._table.allocate(
                        int(hkey_val), 0, access, CmdFamily.REGISTRY_OPS,
                        tag=f"RegKey_{subkey[:32]}_{pkt.source_pid}"
                    )
                    # Keep hkey open — stored in tag via handle table; close on REG_CLOSE
                    # We detach the handle so Python doesn't close it
                    hkey.Detach()
                    return self._make_ok(pkt, handle)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, str(exc))

            elif code == CmdCode.REG_QUERY64:
                # args: (key_handle, value_name: str)
                key_handle  = int(args[0]) if args else 0
                value_name  = str(args[1]) if len(args) > 1 else ""
                hkey_val    = self._table.resolve(key_handle) or key_handle
                try:
                    # Pass raw HKEY integer — winreg accepts int as key handle
                    data, reg_type = winreg.QueryValueEx(hkey_val, value_name)
                    # Serialise: reg_type + data
                    if isinstance(data, int):
                        payload, c = pack_args(reg_type, data)
                    elif isinstance(data, str):
                        payload, c = pack_args(reg_type, data)
                    else:
                        payload, c = pack_args(reg_type, str(data))
                    return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, str(exc))

            elif code == CmdCode.REG_SET64:
                # args: (key_handle, value_name: str, reg_type: uint32, data)
                if len(args) < 4:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "REG_SET64 needs 4 args")
                key_handle = int(args[0])
                value_name = str(args[1])
                reg_type   = int(args[2])
                data       = args[3]
                hkey_val   = self._table.resolve(key_handle) or key_handle
                try:
                    # Pass raw HKEY integer — winreg accepts int as key handle
                    winreg.SetValueEx(hkey_val, value_name, 0, reg_type, data)
                    return self._make_ok(pkt, 1)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_ACCESS_DENIED, str(exc))

            elif code == CmdCode.REG_ENUM64:
                # args: (key_handle,) — enumerate subkeys
                key_handle = int(args[0]) if args else 0
                hkey_val   = self._table.resolve(key_handle) or key_handle
                try:
                    # Pass raw HKEY integer — winreg accepts int as key handle
                    subkeys = []
                    i = 0
                    while True:
                        try:
                            subkeys.append(winreg.EnumKey(hkey_val, i))
                            i += 1
                        except OSError:
                            break
                    payload, c = pack_args("\n".join(subkeys))
                    return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, str(exc))

            elif code == CmdCode.REG_CREATE64:
                return self._handle_reg_create(pkt)
            elif code == CmdCode.REG_DELETE_KEY64:
                return self._handle_reg_delete_key(pkt)
            elif code == CmdCode.REG_DELETE_VAL64:
                return self._handle_reg_delete_value(pkt)
            elif code == CmdCode.REG_CLOSE64:
                return self._handle_reg_close(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown REGISTRY_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 7: GRAPHICS OPS
    # -------------------------------------------------------------------------

    def _handle_graphics_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.GRAPHICS_OPS]:

            if code == CmdCode.GPU_QUERY_INFO:
                # Query GPU/adapter info via DXGI (no render target needed)
                # Uses ctypes to call DXGI factory
                info_parts: List[str] = []
                try:
                    dxgi = ctypes.windll.dxgi
                    # IDXGIFactory interface
                    # IID_IDXGIFactory = {7b7166ec-21c7-44ae-b21a-c9ae321ae369}
                    # noinspection PyPep8Naming
                    IID_IDXGIFactory = (ctypes.c_byte * 16)(
                        0xEC, 0x66, 0x71, 0x7B, 0xC7, 0x21, 0xAE, 0x44,
                        0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69
                    )
                    factory_ptr = ctypes.c_void_p(0)
                    hr = getattr(dxgi, 'CreateDXGIFactory')(
                        ctypes.byref(IID_IDXGIFactory),
                        ctypes.byref(factory_ptr)
                    )
                    if hr == 0 and factory_ptr.value:
                        info_parts.append(f"DXGI_FACTORY=0x{factory_ptr.value:016X}")
                        # Store factory as handle
                        handle = self._table.allocate(
                            factory_ptr.value, 0, 0, CmdFamily.GRAPHICS_OPS,
                            tag=f"DXGIFactory_{pkt.source_pid}"
                        )
                        info_parts.append(f"FACTORY_HANDLE=0x{handle:08X}")
                    else:
                        info_parts.append(f"DXGI_HR=0x{hr & 0xFFFFFFFF:08X}")
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    info_parts.append(f"DXGI_ERROR={exc}")

                # Also report system GPU info via WMI-lite (registry path)
                try:
                    import winreg as _wr
                    gpu_key = _wr.OpenKey(
                        _wr.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Video",
                        0,
                        _wr.KEY_READ | KEY_WOW64_64KEY
                    )
                    info_parts.append(f"GPU_KEY=HKLM\\SYSTEM\\CurrentControlSet\\Control\\Video")
                    gpu_key.Close()
                except Exception as _et_exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", _et_exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {_et_exc}",
                        cause=_et_exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    pass

                payload, c = pack_args("\n".join(info_parts))
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

            elif code == CmdCode.GPU_ALLOC_VRAM:
                # Allocate GPU VRAM via D3D11 CreateBuffer.
                #
                # ET derivation (Identification Principle):
                #   P = GPU VRAM (the substrate — actual video memory)
                #   D = size + usage flags (the constraints on allocation)
                #   T = ID3D11Device::CreateBuffer (the API traversal)
                #   E = allocated buffer with real GPU backing
                #
                # If a D3D11 device exists for this PID (created by GPU_CREATE_DEVICE),
                # we use it to allocate a real GPU buffer. Otherwise, we auto-create
                # a D3D11 device first. VirtualAlloc fallback only if D3D11 is completely
                # unavailable (no GPU, no DXGI).
                size    = int(args[0]) if args else DIGITAL_ACTION_QUANTUM
                protect = int(args[1]) if len(args) > 1 else PAGE_READWRITE

                # Round size to ħ_d boundary
                size = ((size + DIGITAL_ACTION_QUANTUM - 1) // DIGITAL_ACTION_QUANTUM) * DIGITAL_ACTION_QUANTUM

                # Attempt D3D11 GPU buffer allocation
                gpu_alloc_ok = False
                buffer_ptr = 0
                with self._gpu_lock:
                    gpu_info = self._gpu_devices.get(pkt.source_pid)

                if gpu_info and gpu_info[2] == 1 and gpu_info[0]:
                    # D3D11 device available — use ID3D11Device::CreateBuffer
                    device_ptr = gpu_info[0]
                    try:
                        # D3D11_BUFFER_DESC structure (24 bytes):
                        #   ByteWidth:          UINT  [4]
                        #   Usage:              UINT  [4]  (0=DEFAULT, 3=STAGING)
                        #   BindFlags:          UINT  [4]  (0x80=UNORDERED_ACCESS for GPU RW)
                        #   CPUAccessFlags:     UINT  [4]  (0x30000=READ|WRITE for staging)
                        #   MiscFlags:          UINT  [4]
                        #   StructureByteStride:UINT  [4]
                        class D3D11BufferDesc(ctypes.Structure):
                            """D3D11_BUFFER_DESC — D-container for GPU buffer creation."""
                            _fields_ = [
                                ("ByteWidth",           ctypes.wintypes.UINT),
                                ("Usage",               ctypes.wintypes.UINT),
                                ("BindFlags",           ctypes.wintypes.UINT),
                                ("CPUAccessFlags",      ctypes.wintypes.UINT),
                                ("MiscFlags",           ctypes.wintypes.UINT),
                                ("StructureByteStride", ctypes.wintypes.UINT),
                            ]

                        # D3D11_USAGE_DEFAULT=0 for GPU VRAM, D3D11_BIND_UNORDERED_ACCESS=0x80
                        # for GPU read/write. Staging copy created at Map time.
                        desc = D3D11BufferDesc()
                        desc.ByteWidth           = size if size <= 0xFFFFFFFF else 0xFFFFFFFF
                        desc.Usage               = 0   # D3D11_USAGE_DEFAULT (GPU VRAM)
                        desc.BindFlags           = 0x80  # D3D11_BIND_UNORDERED_ACCESS
                        desc.CPUAccessFlags      = 0
                        desc.MiscFlags           = 0
                        desc.StructureByteStride = 0

                        # ID3D11Device::CreateBuffer is vtable[3]
                        device_vtable = ctypes.cast(
                            ctypes.c_void_p(
                                ctypes.cast(ctypes.c_void_p(device_ptr),
                                            ctypes.POINTER(ctypes.c_void_p))[0]
                            ),
                            ctypes.POINTER(ctypes.c_void_p)
                        )
                        # CreateBuffer(this, pDesc, pInitialData, ppBuffer)
                        # noinspection PyPep8Naming
                        CreateBuffer = ctypes.WINFUNCTYPE(
                            ctypes.c_long,                      # HRESULT
                            ctypes.c_void_p,                    # this
                            ctypes.POINTER(D3D11BufferDesc),    # pDesc
                            ctypes.c_void_p,                    # pInitialData (NULL)
                            ctypes.POINTER(ctypes.c_void_p),    # ppBuffer (OUT)
                        )(device_vtable[3])

                        buffer_out = ctypes.c_void_p(0)
                        hr = CreateBuffer(
                            ctypes.c_void_p(device_ptr),
                            ctypes.byref(desc),
                            None,  # no initial data
                            ctypes.byref(buffer_out)
                        )
                        if hr == 0 and buffer_out.value:
                            buffer_ptr = buffer_out.value
                            gpu_alloc_ok = True
                            self._log.mediation(
                                "GPU_ALLOC_VRAM: D3D11 CreateBuffer %d bytes → 0x%016X PID=%d",
                                size, buffer_ptr, pkt.source_pid)
                        else:
                            self._log.warning_di(
                                "GPU_ALLOC_VRAM: D3D11 CreateBuffer failed HR=0x%08X, "
                                "falling back to VirtualAlloc PID=%d",
                                hr & 0xFFFFFFFF, pkt.source_pid)
                    except Exception as _gpu_exc:
                        self._log.exc(
                            "[et_host64] GPU_ALLOC_VRAM D3D11 exception: %s", _gpu_exc)
                        record_error(ETOperationError(
                            f"GPU_ALLOC_VRAM D3D11 CreateBuffer: {_gpu_exc}",
                            cause=_gpu_exc,
                            os_error=ctypes.GetLastError(),
                            severity=ETErrorSeverity.BOUNDARY,
                            depth=2,
                        ))

                if not gpu_alloc_ok:
                    # Fallback: VirtualAlloc in 64-bit address space.
                    # This provides >4GB addressability but NOT GPU VRAM.
                    # Logged as MEDIATION (not silent) so operator knows.
                    self._log.mediation(
                        "GPU_ALLOC_VRAM: no D3D11 device for PID %d — VirtualAlloc fallback "
                        "(%d bytes, CPU-only, not true GPU VRAM)", pkt.source_pid, size)
                    ptr = getattr(kernel32, 'VirtualAlloc')(
                        None,
                        ctypes.c_size_t(size),
                        ctypes.wintypes.DWORD(MEM_COMMIT | MEM_RESERVE),
                        ctypes.wintypes.DWORD(protect)
                    )
                    if not ptr:
                        return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                                f"GPU VirtualAlloc fallback failed: "
                                                f"{getattr(kernel32, 'GetLastError')()}")
                    buffer_ptr = ctypes.cast(ptr, ctypes.c_void_p).value

                handle = self._table.allocate(
                    buffer_ptr, size, protect, CmdFamily.GRAPHICS_OPS,
                    tag=f"VRAM_{'D3D11' if gpu_alloc_ok else 'CPU'}_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, buffer_ptr, size)

            elif code == CmdCode.GPU_FREE_VRAM:
                handle = int(args[0]) if args else 0
                addr64 = self._table.resolve(handle)
                if addr64 is None:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} unknown")
                ok = getattr(kernel32, 'VirtualFree')(
                    ctypes.c_void_p(addr64),
                    ctypes.c_size_t(0),
                    ctypes.wintypes.DWORD(MEM_RELEASE)
                )
                if ok:
                    self._table.release(handle)
                return self._make_ok(pkt, int(bool(ok)))

            elif code == CmdCode.GPU_MAP_VRAM:
                # Map a VRAM allocation for CPU access.
                #
                # ET derivation (Descriptor Gap Principle):
                #   The gap between "GPU buffer allocated" and "CPU can read/write it"
                #   IS the mapping operation. For D3D11 buffers, this requires creating
                #   a staging buffer, copying the GPU data to it, then mapping the staging
                #   buffer. For VirtualAlloc fallback buffers, the address is already
                #   CPU-accessible.
                #
                # Returns: (handle, mapped_cpu_addr64, size)
                handle = int(args[0]) if args else 0
                entry = self._table.get_entry(handle)
                if entry is None:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                            f"Handle 0x{handle:08X} unknown")

                addr64 = entry.addr64
                tag = entry.tag if hasattr(entry, 'tag') else ""

                if "VRAM_D3D11" in tag:
                    # Real D3D11 buffer — create staging copy and map it.
                    # The GPU buffer (D3D11_USAGE_DEFAULT) cannot be mapped directly.
                    # We create a staging buffer, CopyResource from GPU → staging,
                    # then Map the staging buffer for CPU access.
                    with self._gpu_lock:
                        gpu_info = self._gpu_devices.get(pkt.source_pid)

                    if gpu_info and gpu_info[2] == 1 and gpu_info[0] and gpu_info[1]:
                        device_ptr  = gpu_info[0]
                        context_ptr = gpu_info[1]
                        try:
                            # Create staging buffer with same size
                            class D3D11BufferDesc(ctypes.Structure):
                                """D3D11_BUFFER_DESC for staging buffer."""
                                _fields_ = [
                                    ("ByteWidth",           ctypes.wintypes.UINT),
                                    ("Usage",               ctypes.wintypes.UINT),
                                    ("BindFlags",           ctypes.wintypes.UINT),
                                    ("CPUAccessFlags",      ctypes.wintypes.UINT),
                                    ("MiscFlags",           ctypes.wintypes.UINT),
                                    ("StructureByteStride", ctypes.wintypes.UINT),
                                ]

                            class D3D11MappedSubresource(ctypes.Structure):
                                """D3D11_MAPPED_SUBRESOURCE — output of Map()."""
                                _fields_ = [
                                    ("pData",      ctypes.c_void_p),
                                    ("RowPitch",   ctypes.wintypes.UINT),
                                    ("DepthPitch", ctypes.wintypes.UINT),
                                ]

                            desc = D3D11BufferDesc()
                            desc.ByteWidth           = entry.size if entry.size <= 0xFFFFFFFF else 0xFFFFFFFF
                            desc.Usage               = 3      # D3D11_USAGE_STAGING
                            desc.BindFlags           = 0
                            desc.CPUAccessFlags      = 0x30000  # READ | WRITE
                            desc.MiscFlags           = 0
                            desc.StructureByteStride = 0

                            # ID3D11Device::CreateBuffer (vtable[3])
                            d_vtable = ctypes.cast(
                                ctypes.c_void_p(
                                    ctypes.cast(ctypes.c_void_p(device_ptr),
                                                ctypes.POINTER(ctypes.c_void_p))[0]
                                ),
                                ctypes.POINTER(ctypes.c_void_p)
                            )
                            # noinspection PyPep8Naming
                            CreateBuffer = ctypes.WINFUNCTYPE(
                                ctypes.c_long, ctypes.c_void_p,
                                ctypes.POINTER(D3D11BufferDesc),
                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
                            )(d_vtable[3])

                            staging_out = ctypes.c_void_p(0)
                            hr_stg = CreateBuffer(
                                ctypes.c_void_p(device_ptr),
                                ctypes.byref(desc), None,
                                ctypes.byref(staging_out)
                            )
                            if hr_stg != 0 or not staging_out.value:
                                return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                    f"GPU_MAP_VRAM: staging buffer creation failed "
                                    f"HR=0x{hr_stg & 0xFFFFFFFF:08X}")

                            # ID3D11DeviceContext::CopyResource (vtable[47])
                            # Copies entire GPU buffer → staging buffer
                            c_vtable = ctypes.cast(
                                ctypes.c_void_p(
                                    ctypes.cast(ctypes.c_void_p(context_ptr),
                                                ctypes.POINTER(ctypes.c_void_p))[0]
                                ),
                                ctypes.POINTER(ctypes.c_void_p)
                            )
                            # noinspection PyPep8Naming
                            CopyResource = ctypes.WINFUNCTYPE(
                                None, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p
                            )(c_vtable[47])
                            CopyResource(
                                ctypes.c_void_p(context_ptr),
                                ctypes.c_void_p(staging_out.value),
                                ctypes.c_void_p(addr64)
                            )

                            # ID3D11DeviceContext::Map (vtable[14])
                            # Map(this, pResource, Subresource, MapType, MapFlags, pMappedResource)
                            # noinspection PyPep8Naming
                            MapFn = ctypes.WINFUNCTYPE(
                                ctypes.c_long, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.wintypes.UINT,
                                ctypes.wintypes.UINT, ctypes.wintypes.UINT,
                                ctypes.POINTER(D3D11MappedSubresource)
                            )(c_vtable[14])

                            mapped = D3D11MappedSubresource()
                            # D3D11_MAP_READ_WRITE = 3
                            hr_map = MapFn(
                                ctypes.c_void_p(context_ptr),
                                ctypes.c_void_p(staging_out.value),
                                0, 3, 0, ctypes.byref(mapped)
                            )
                            if hr_map == 0 and mapped.pData:
                                # Store staging buffer handle for later Unmap
                                staging_handle = self._table.allocate(
                                    staging_out.value, entry.size, 0,
                                    CmdFamily.GRAPHICS_OPS,
                                    tag=f"VRAM_STAGING_{pkt.source_pid}"
                                )
                                self._log.mediation(
                                    "GPU_MAP_VRAM: D3D11 Map → CPU addr 0x%016X "
                                    "staging=0x%08X PID=%d",
                                    mapped.pData, staging_handle, pkt.source_pid)
                                return self._make_ok(pkt, handle, mapped.pData, entry.size)
                            else:
                                # Release staging buffer on Map failure
                                # IUnknown::Release (vtable[2])
                                s_vtable = ctypes.cast(
                                    ctypes.c_void_p(
                                        ctypes.cast(ctypes.c_void_p(staging_out.value),
                                                    ctypes.POINTER(ctypes.c_void_p))[0]
                                    ),
                                    ctypes.POINTER(ctypes.c_void_p)
                                )
                                # noinspection PyPep8Naming
                                Release = ctypes.WINFUNCTYPE(
                                    ctypes.c_ulong, ctypes.c_void_p
                                )(s_vtable[2])
                                Release(ctypes.c_void_p(staging_out.value))
                                return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GPU_MAP_VRAM: D3D11 Map failed "
                                    f"HR=0x{hr_map & 0xFFFFFFFF:08X}")
                        except Exception as _map_exc:
                            self._log.exc(
                                "[et_host64] GPU_MAP_VRAM D3D11 exception: %s", _map_exc)
                            record_error(ETOperationError(
                                f"GPU_MAP_VRAM D3D11: {_map_exc}",
                                cause=_map_exc,
                                os_error=ctypes.GetLastError(),
                                severity=ETErrorSeverity.BOUNDARY,
                                depth=2,
                            ))
                            return self._make_error(pkt, ET_ERR_OS_ERROR, str(_map_exc))
                    else:
                        # No D3D11 context — cannot map
                        return self._make_error(pkt, ET_ERR_UNSUPPORTED,
                            f"GPU_MAP_VRAM: no D3D11 context for PID {pkt.source_pid} "
                            f"— call GPU_CREATE_DEVICE first")

                # VirtualAlloc fallback buffer — already CPU-accessible
                return self._make_ok(pkt, handle, addr64, entry.size)

            elif code == CmdCode.GPU_SUBMIT:
                # Submit a command buffer through Heaven's Gate.
                # C2C uses DirectX 9/11 heavily — this routes D3D calls through
                # the broker's 64-bit address space via Heaven's Gate.
                #
                # ET derivation (Identification Principle):
                #   P = GPU device state (substrate)
                #   D = command buffer contents (Descriptors)
                #   T = Heaven's Gate call (Traverser — crosses 32→64 boundary)
                #   E = GPU operation completed (Exception)
                #
                # args: (device_handle, func_addr_handle, [arg0..argN])
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                            "GPU_SUBMIT needs device_handle + func_addr")
                device_handle = int(args[0])
                func_handle   = int(args[1])
                call_args     = [int(a) if isinstance(a, (int, float)) else a
                                 for a in args[2:]]

                # Resolve handles to 64-bit addresses
                device_addr = self._table.resolve(device_handle) or device_handle
                func_addr   = self._table.resolve(func_handle) or func_handle

                # Route through Heaven's Gate for the actual 64-bit D3D call
                if func_addr and self._heaven is not None:
                    # Prepend the device pointer as first arg (COM this-pointer convention)
                    full_args = [device_addr] + call_args
                    result = self._heaven.call_64bit_function(func_addr, full_args)
                    if result is not None:
                        return self._make_ok(pkt, result)
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"Heaven's Gate GPU_SUBMIT failed for "
                                            f"device=0x{device_addr:016X} func=0x{func_addr:016X}")
                return self._make_error(pkt, ET_ERR_UNSUPPORTED,
                                        "GPU_SUBMIT: Heaven's Gate not available")

            elif code == CmdCode.GPU_ENUM_ADAPTERS:
                # Enumerate all DXGI adapters with full info.
                # C2C needs this to discover GPU capabilities for DirectX 9/11.
                #
                # ET derivation (Subsumption Law):
                #   Each adapter is a P-substrate with its own D-set (VRAM, vendor, caps).
                #   Enumeration must subsume ALL adapters without remainder.
                #   We use dynamic enumeration (not static lists) per the _DLLT principle.
                #
                # Returns: adapter_count, then a string with all adapter info
                adapter_info_parts: List[str] = []
                adapter_count = 0
                try:
                    dxgi = ctypes.windll.dxgi
                    # IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387}
                    # noinspection PyPep8Naming
                    IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
                        0x78, 0xAE, 0x0A, 0x77, 0x6F, 0xF2, 0xBA, 0x4D,
                        0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87
                    )
                    factory_ptr = ctypes.c_void_p(0)
                    hr = getattr(dxgi, 'CreateDXGIFactory1')(
                        ctypes.byref(IID_IDXGIFactory1),
                        ctypes.byref(factory_ptr)
                    )
                    if hr == 0 and factory_ptr.value:
                        # IDXGIFactory1 vtable: EnumAdapters1 is at index 12
                        vtable = ctypes.cast(
                            ctypes.c_void_p(
                                ctypes.cast(factory_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
                            ),
                            ctypes.POINTER(ctypes.c_void_p)
                        )
                        # DXGI_ADAPTER_DESC1 structure (packed)
                        class DxgiAdapterDesc1(ctypes.Structure):
                            """DXGI adapter descriptor — D-container for GPU hardware identity."""
                            _fields_ = [
                                ("Description",           ctypes.c_wchar * 128),
                                ("VendorId",              ctypes.wintypes.UINT),
                                ("DeviceId",              ctypes.wintypes.UINT),
                                ("SubSysId",              ctypes.wintypes.UINT),
                                ("Revision",              ctypes.wintypes.UINT),
                                ("DedicatedVideoMemory",  ctypes.c_size_t),
                                ("DedicatedSystemMemory", ctypes.c_size_t),
                                ("SharedSystemMemory",    ctypes.c_size_t),
                                ("AdapterLuid",           ctypes.c_uint64),
                                ("Flags",                 ctypes.wintypes.UINT),
                            ]

                        # EnumAdapters1 prototype: HRESULT (this, UINT, IDXGIAdapter1**)
                        # noinspection PyPep8Naming
                        EnumAdapters1 = ctypes.WINFUNCTYPE(
                            ctypes.c_long,          # HRESULT
                            ctypes.c_void_p,        # this
                            ctypes.wintypes.UINT,   # adapter index
                            ctypes.POINTER(ctypes.c_void_p)  # ppAdapter
                        )(vtable[12])

                        # GetDesc1 is at vtable index 10 of IDXGIAdapter1
                        adapter_idx = 0
                        while True:
                            adapter_ptr = ctypes.c_void_p(0)
                            hr_enum = EnumAdapters1(
                                factory_ptr, adapter_idx, ctypes.byref(adapter_ptr)
                            )
                            if hr_enum != 0:  # DXGI_ERROR_NOT_FOUND = 0x887A0002
                                break
                            if adapter_ptr.value:
                                # Read adapter descriptor via GetDesc1 (vtable[10])
                                a_vtable = ctypes.cast(
                                    ctypes.c_void_p(
                                        ctypes.cast(adapter_ptr,
                                                    ctypes.POINTER(ctypes.c_void_p))[0]
                                    ),
                                    ctypes.POINTER(ctypes.c_void_p)
                                )
                                # noinspection PyPep8Naming
                                GetDesc1 = ctypes.WINFUNCTYPE(
                                    ctypes.c_long,
                                    ctypes.c_void_p,
                                    ctypes.POINTER(DxgiAdapterDesc1)
                                )(a_vtable[10])
                                desc = DxgiAdapterDesc1()
                                hr_desc = GetDesc1(adapter_ptr, ctypes.byref(desc))
                                if hr_desc == 0:
                                    vram_mb = desc.DedicatedVideoMemory // (1024 * 1024)
                                    adapter_info_parts.append(
                                        f"ADAPTER[{adapter_idx}]="
                                        f"{desc.Description.rstrip(chr(0))}|"
                                        f"VID=0x{desc.VendorId:04X}|"
                                        f"DID=0x{desc.DeviceId:04X}|"
                                        f"VRAM={vram_mb}MB|"
                                        f"FLAGS=0x{desc.Flags:X}"
                                    )
                                # Release adapter (IUnknown::Release = vtable[2])
                                # noinspection PyPep8Naming
                                Release = ctypes.WINFUNCTYPE(
                                    ctypes.c_ulong, ctypes.c_void_p
                                )(a_vtable[2])
                                Release(adapter_ptr)
                                adapter_count += 1
                            adapter_idx += 1

                        # Release factory
                        f_vtable = ctypes.cast(
                            ctypes.c_void_p(
                                ctypes.cast(factory_ptr,
                                            ctypes.POINTER(ctypes.c_void_p))[0]
                            ),
                            ctypes.POINTER(ctypes.c_void_p)
                        )
                        # noinspection PyPep8Naming
                        FactoryRelease = ctypes.WINFUNCTYPE(
                            ctypes.c_ulong, ctypes.c_void_p
                        )(f_vtable[2])
                        FactoryRelease(factory_ptr)
                    else:
                        adapter_info_parts.append(f"DXGI_HR=0x{hr & 0xFFFFFFFF:08X}")
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] GPU_ENUM_ADAPTERS exception: %s", exc)
                    record_error(ETOperationError(
                        f"GPU_ENUM_ADAPTERS exception: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    adapter_info_parts.append(f"ERROR={exc}")
                info_str = "\n".join(adapter_info_parts)
                payload, c = pack_args(adapter_count, info_str)
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

            elif code == CmdCode.GPU_CREATE_DEVICE:
                # Create a D3D9 or D3D11 device via broker's 64-bit address space.
                # C2C uses DirectX 9 primarily (d3d9.dll → Direct3DCreate9).
                #
                # ET derivation (Descriptor Gap Principle):
                #   The gap between C2C's 32-bit D3D9 device and 64-bit GPU features
                #   (>4GB VRAM, large textures, full shader model) is closed by creating
                #   the device in the broker and bridging calls through Heaven's Gate.
                #
                # args: (device_type: 0=D3D9/1=D3D11, adapter_index: uint32)
                device_type   = int(args[0]) if args else 0
                adapter_index = int(args[1]) if len(args) > 1 else 0

                try:
                    device_addr = 0
                    if device_type == 0:
                        # D3D9: Load d3d9.dll and call Direct3DCreate9(D3D_SDK_VERSION=32)
                        # Direct3DCreate9 returns IDirect3D9* — the adapter_index is needed
                        # for the subsequent IDirect3D9::CreateDevice(Adapter, ...) call.
                        # We store adapter_index in the handle tag so GPU_HEAVEN_CALL can
                        # retrieve it when the client calls CreateDevice.
                        d3d9_base = self._heaven.load_library_64("d3d9.dll")
                        if d3d9_base:
                            create_addr = self._heaven.get_proc_from_loaded_dll(
                                d3d9_base, "Direct3DCreate9")
                            if create_addr:
                                # noinspection PyPep8Naming
                                D3D_SDK_VERSION = 32
                                device_addr = self._heaven.call_64bit_function(
                                    create_addr, [D3D_SDK_VERSION])
                    elif device_type == 1:
                        # D3D11: Load d3d11.dll and call D3D11CreateDevice.
                        # D3D11CreateDevice signature:
                        #   HRESULT D3D11CreateDevice(
                        #     IDXGIAdapter* pAdapter,   ← selected by adapter_index
                        #     D3D_DRIVER_TYPE,          ← D3D_DRIVER_TYPE_HARDWARE = 1
                        #     HMODULE Software,         ← NULL
                        #     UINT Flags,               ← 0
                        #     const D3D_FEATURE_LEVEL*, ← NULL (use default)
                        #     UINT FeatureLevels,       ← 0
                        #     UINT SDKVersion,          ← D3D11_SDK_VERSION = 7
                        #     ID3D11Device** ppDevice,  ← output
                        #     D3D_FEATURE_LEVEL*,       ← NULL
                        #     ID3D11DeviceContext**      ← NULL
                        #   )
                        # Use adapter_index to select the DXGI adapter via factory enum.
                        d3d11_base = self._heaven.load_library_64("d3d11.dll")
                        if d3d11_base:
                            create_addr = self._heaven.get_proc_from_loaded_dll(
                                d3d11_base, "D3D11CreateDevice")
                            if create_addr:
                                # Resolve the specific DXGI adapter for adapter_index
                                dxgi_adapter_ptr = 0
                                try:
                                    dxgi = ctypes.windll.dxgi
                                    # noinspection PyPep8Naming
                                    IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
                                        0x78, 0xAE, 0x0A, 0x77, 0x6F, 0xF2, 0xBA, 0x4D,
                                        0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87
                                    )
                                    factory_ptr = ctypes.c_void_p(0)
                                    hr = getattr(dxgi, 'CreateDXGIFactory1')(
                                        ctypes.byref(IID_IDXGIFactory1),
                                        ctypes.byref(factory_ptr)
                                    )
                                    if hr == 0 and factory_ptr.value:
                                        # EnumAdapters1 at vtable[12]
                                        f_vtable = ctypes.cast(
                                            ctypes.c_void_p(
                                                ctypes.cast(factory_ptr,
                                                            ctypes.POINTER(ctypes.c_void_p))[0]
                                            ),
                                            ctypes.POINTER(ctypes.c_void_p)
                                        )
                                        # noinspection PyPep8Naming
                                        EnumAdapters1 = ctypes.WINFUNCTYPE(
                                            ctypes.c_long, ctypes.c_void_p,
                                            ctypes.wintypes.UINT,
                                            ctypes.POINTER(ctypes.c_void_p)
                                        )(f_vtable[12])
                                        adapter_out = ctypes.c_void_p(0)
                                        hr_enum = EnumAdapters1(
                                            factory_ptr, adapter_index,
                                            ctypes.byref(adapter_out)
                                        )
                                        if hr_enum == 0 and adapter_out.value:
                                            dxgi_adapter_ptr = adapter_out.value
                                        # Release factory (IUnknown::Release = vtable[2])
                                        # noinspection PyPep8Naming
                                        FactoryRelease = ctypes.WINFUNCTYPE(
                                            ctypes.c_ulong, ctypes.c_void_p
                                        )(f_vtable[2])
                                        FactoryRelease(factory_ptr)
                                except Exception as dxgi_exc:
                                    self._log.mediation(
                                        "GPU_CREATE_DEVICE: DXGI adapter %d lookup failed: %s "
                                        "(falling back to NULL adapter) PID=%d",
                                        adapter_index, dxgi_exc, pkt.source_pid)

                                # Call D3D11CreateDevice with resolved adapter (or NULL=default)
                                # D3D11CreateDevice takes 10 params:
                                #   1: pAdapter, 2: DriverType, 3: Software, 4: Flags,
                                #   5: pFeatureLevels, 6: FeatureLevels, 7: SDKVersion,
                                #   8: ppDevice (OUT), 9: pFeatureLevel (OUT), 10: ppContext (OUT)
                                # noinspection PyPep8Naming
                                D3D11_SDK_VERSION = 7
                                device_out  = ctypes.c_void_p(0)
                                feature_out = ctypes.c_uint32(0)
                                context_out = ctypes.c_void_p(0)
                                hr_create = self._heaven.call_64bit_function(
                                    create_addr, [
                                        dxgi_adapter_ptr,  # pAdapter (0 = default)
                                        1 if not dxgi_adapter_ptr else 0,  # DriverType: HARDWARE=1 if no adapter, UNKNOWN=0 if adapter
                                        0,                  # Software HMODULE
                                        0,                  # Flags
                                        0,                  # pFeatureLevels (NULL)
                                        0,                  # FeatureLevels count
                                        D3D11_SDK_VERSION,  # SDKVersion
                                        ctypes.addressof(device_out),   # ppDevice (OUT)
                                        ctypes.addressof(feature_out),  # pFeatureLevel (OUT)
                                        ctypes.addressof(context_out),  # ppImmediateContext (OUT)
                                    ]
                                )
                                if hr_create is not None and hr_create == 0 and device_out.value:
                                    device_addr = device_out.value
                                    # Store context for GPU_ALLOC_VRAM / GPU_MAP_VRAM
                                    with self._gpu_lock:
                                        self._gpu_devices[pkt.source_pid] = (
                                            device_out.value,
                                            context_out.value if context_out.value else 0,
                                            1  # device_type = D3D11
                                        )

                    if device_addr:
                        handle = self._table.allocate(
                            device_addr, 0, 0, CmdFamily.GRAPHICS_OPS,
                            tag=f"D3DDevice_type{device_type}_adapter{adapter_index}_{pkt.source_pid}"
                        )
                        # Also register device in _gpu_devices for D3D9 path
                        if device_type == 0:
                            with self._gpu_lock:
                                self._gpu_devices[pkt.source_pid] = (
                                    device_addr, 0, 0  # D3D9: no separate context
                                )
                        return self._make_ok(pkt, handle, device_addr, adapter_index)
                    return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                            f"GPU_CREATE_DEVICE: D3D{'9' if device_type == 0 else '11'} "
                                            f"creation failed")
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] GPU_CREATE_DEVICE exception: %s", exc)
                    record_error(ETOperationError(
                        f"GPU_CREATE_DEVICE exception: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

            elif code == CmdCode.GPU_HEAVEN_CALL:
                # Execute an arbitrary D3D/GPU function through Heaven's Gate.
                # This is the general-purpose GPU call path for C2C's DirectX usage.
                #
                # ET derivation (Subsumption Law):
                #   One handler subsumes ALL D3D function calls. No static list of
                #   D3D functions needed — the func_handle identifies the target,
                #   and Heaven's Gate provides the 64-bit execution context.
                #
                # args: (func_handle, [arg0, arg1, ...])
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                            "GPU_HEAVEN_CALL needs func handle")
                func_handle = int(args[0])
                call_args   = [int(a) if isinstance(a, (int, float)) else a
                               for a in args[1:]]
                func_addr64 = self._table.resolve(func_handle) or func_handle

                if self._heaven is not None:
                    result = self._heaven.call_64bit_function(func_addr64, call_args)
                    if result is not None:
                        return self._make_ok(pkt, result)
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"GPU_HEAVEN_CALL at 0x{func_addr64:016X} failed")
                return self._make_error(pkt, ET_ERR_UNSUPPORTED,
                                        "GPU_HEAVEN_CALL: Heaven's Gate not available")

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown GRAPHICS_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 8: FILE OPS (large files, >4GB)
    # -------------------------------------------------------------------------

    def _handle_file_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        # noinspection PyPep8Naming
        GENERIC_READ_W  = 0x80000000
        # noinspection PyPep8Naming
        GENERIC_WRITE_W = 0x40000000
        # noinspection PyPep8Naming
        OPEN_EXISTING_W = 3
        # noinspection PyPep8Naming
        CREATE_ALWAYS_W = 2
        # noinspection PyPep8Naming
        FILE_SHARE_READ = 0x00000001
        # noinspection PyPep8Naming
        FILE_SHARE_WRITE = 0x00000002
        # noinspection PyPep8Naming
        FILE_FLAG_NO_BUFFERING = 0x20000000
        # noinspection PyPep8Naming
        FILE_FLAG_OVERLAPPED_W = 0x40000000

        with self._family_locks[CmdFamily.FILE_OPS]:

            if code == CmdCode.FILE_OPEN_LARGE:
                # args: (path: str, access: uint32, share: uint32, disposition: uint32, flags: uint32)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "FILE_OPEN_LARGE needs path")
                path        = str(args[0])
                access      = int(args[1]) if len(args) > 1 else GENERIC_READ_W | GENERIC_WRITE_W
                share       = int(args[2]) if len(args) > 2 else FILE_SHARE_READ | FILE_SHARE_WRITE
                disposition = int(args[3]) if len(args) > 3 else OPEN_EXISTING_W
                flags       = int(args[4]) if len(args) > 4 else FILE_ATTRIBUTE_NORMAL

                # Dynamic flag detection — ET: Descriptors identify the operation mode
                if disposition == CREATE_ALWAYS_W:
                    # Creating new file — ensure write access is granted
                    if not (access & GENERIC_WRITE_W):
                        access |= GENERIC_WRITE_W
                if flags & FILE_FLAG_NO_BUFFERING:
                    # Direct I/O — log for alignment awareness (ħ_d coherence)
                    self._log.mediation(
                        "FILE_OPEN_LARGE: direct I/O (NO_BUFFERING) for '%s' PID=%d",
                        path, pkt.source_pid)
                if flags & FILE_FLAG_OVERLAPPED_W:
                    # Overlapped (async) I/O — log for dispatch tracking
                    self._log.mediation(
                        "FILE_OPEN_LARGE: overlapped I/O for '%s' PID=%d",
                        path, pkt.source_pid)

                h_file = getattr(kernel32, 'CreateFileW')(
                    path,
                    ctypes.wintypes.DWORD(access),
                    ctypes.wintypes.DWORD(share),
                    None,
                    ctypes.wintypes.DWORD(disposition),
                    ctypes.wintypes.DWORD(flags),
                    None
                )
                if h_file == ctypes.wintypes.HANDLE(-1).value:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                            f"CreateFileW('{path}') failed: {getattr(kernel32, 'GetLastError')()}")
                # Get file size
                file_size_high = ctypes.wintypes.DWORD(0)
                file_size_low  = getattr(kernel32, 'GetFileSize')(h_file, ctypes.byref(file_size_high))
                file_size      = (file_size_high.value << 32) | file_size_low

                handle = self._table.allocate(
                    h_file, file_size, access, CmdFamily.FILE_OPS,
                    tag=f"File_{os.path.basename(path)}_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, h_file, file_size)

            elif code == CmdCode.FILE_SEEK_LARGE:
                # args: (file_handle, offset: int64, whence: uint32)
                file_handle = int(args[0]) if args else 0
                offset      = int(args[1]) if len(args) > 1 else 0
                whence      = int(args[2]) if len(args) > 2 else 0  # FILE_BEGIN=0
                h_file      = self._table.resolve(file_handle) or file_handle
                new_pos_high = ctypes.wintypes.LONG(offset >> 32)
                new_pos_low  = getattr(kernel32, 'SetFilePointer')(
                    ctypes.wintypes.HANDLE(h_file),
                    ctypes.wintypes.LONG(offset & 0xFFFFFFFF),
                    ctypes.byref(new_pos_high),
                    ctypes.wintypes.DWORD(whence)
                )
                if new_pos_low == 0xFFFFFFFF and getattr(kernel32, 'GetLastError')() != 0:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"SetFilePointer failed: {getattr(kernel32, 'GetLastError')()}")
                new_pos = (new_pos_high.value << 32) | new_pos_low
                return self._make_ok(pkt, new_pos)

            elif code == CmdCode.FILE_READ_LARGE:
                # args: (file_handle, size)
                file_handle = int(args[0]) if args else 0
                size        = min(int(args[1]) if len(args) > 1 else 4096,
                                  IPC_BUFFER_SIZE - PDT_HEADER_SIZE - 64)
                h_file      = self._table.resolve(file_handle) or file_handle
                buf         = ctypes.create_string_buffer(size)
                read        = ctypes.wintypes.DWORD(0)
                ok          = getattr(kernel32, 'ReadFile')(
                    ctypes.wintypes.HANDLE(h_file), buf, size,
                    ctypes.byref(read), None
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"ReadFile failed: {getattr(kernel32, 'GetLastError')()}")
                payload, c = pack_args(bytes(buf[:read.value]))
                return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)

            elif code == CmdCode.FILE_WRITE_LARGE:
                # args: (file_handle, data: bytes)
                file_handle = int(args[0]) if args else 0
                data        = args[1] if len(args) > 1 and isinstance(args[1], bytes) else b""
                h_file      = self._table.resolve(file_handle) or file_handle
                written     = ctypes.wintypes.DWORD(0)
                ok          = getattr(kernel32, 'WriteFile')(
                    ctypes.wintypes.HANDLE(h_file), data, len(data),
                    ctypes.byref(written), None
                )
                if not ok:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"WriteFile failed: {getattr(kernel32, 'GetLastError')()}")
                return self._make_ok(pkt, written.value)

            elif code == CmdCode.FILE_MAP_LARGE:
                # Map a large file region: args = (file_handle, offset, size, protect)
                file_handle = int(args[0]) if args else 0
                offset      = int(args[1]) if len(args) > 1 else 0
                size        = int(args[2]) if len(args) > 2 else 0
                protect     = int(args[3]) if len(args) > 3 else PAGE_READWRITE
                h_file      = self._table.resolve(file_handle) or file_handle

                h_map = getattr(kernel32, 'CreateFileMappingW')(
                    ctypes.wintypes.HANDLE(h_file),
                    None,
                    ctypes.wintypes.DWORD(protect),
                    ctypes.wintypes.DWORD(size >> 32),
                    ctypes.wintypes.DWORD(size & 0xFFFFFFFF),
                    None
                )
                if not h_map:
                    return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                            f"CreateFileMappingW failed: {getattr(kernel32, 'GetLastError')()}")

                view = getattr(kernel32, 'MapViewOfFile')(
                    h_map,
                    ctypes.wintypes.DWORD(FILE_MAP_ALL_ACCESS),
                    ctypes.wintypes.DWORD(offset >> 32),
                    ctypes.wintypes.DWORD(offset & 0xFFFFFFFF),
                    ctypes.c_size_t(size)
                )
                getattr(kernel32, 'CloseHandle')(h_map)  # view keeps the mapping alive
                if not view:
                    return self._make_error(pkt, ET_ERR_ALLOC_FAIL,
                                            f"MapViewOfFile failed: {getattr(kernel32, 'GetLastError')()}")
                addr64  = ctypes.cast(view, ctypes.c_void_p).value
                handle  = self._table.allocate(
                    addr64, size, protect, CmdFamily.FILE_OPS,
                    tag=f"LargeMap_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, addr64, size)

            elif code == CmdCode.FILE_CLOSE_LARGE:
                return self._handle_file_close(pkt)
            elif code == CmdCode.FILE_GETSIZE_LARGE:
                return self._handle_file_getsize(pkt)
            elif code == CmdCode.FILE_GETATTR_LARGE:
                return self._handle_file_getattr(pkt)
            elif code == CmdCode.FILE_SETATTR_LARGE:
                return self._handle_file_setattr(pkt)
            elif code == CmdCode.FILE_SETEOF_LARGE:
                return self._handle_file_seteof(pkt)
            elif code == CmdCode.FILE_FLUSH_LARGE:
                return self._handle_file_flush(pkt)
            elif code == CmdCode.FILE_GETTIME_LARGE:
                return self._handle_file_gettime(pkt)
            elif code == CmdCode.FILE_SETTIME_LARGE:
                return self._handle_file_settime(pkt)
            elif code == CmdCode.FILE_FIND_FIRST:
                return self._handle_file_find_first(pkt)
            elif code == CmdCode.FILE_FIND_NEXT:
                return self._handle_file_find_next(pkt)
            elif code == CmdCode.FILE_FIND_CLOSE:
                return self._handle_file_find_close(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown FILE_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 9: SYNC OPS
    # -------------------------------------------------------------------------

    def _handle_sync_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        # noinspection PyPep8Naming
        EVENT_ALL_ACCESS = 0x1F0003
        # noinspection PyPep8Naming
        MUTEX_ALL_ACCESS = 0x1F0001
        # noinspection PyPep8Naming
        SEMAPHORE_ALL_ACCESS = 0x1F0003

        with self._family_locks[CmdFamily.SYNC_OPS]:

            if code == CmdCode.SYNC_CREATE_EVENT:
                # args: (manual_reset: bool, initial_state: bool, name_or_None)
                manual   = bool(args[0]) if args else False
                initial  = bool(args[1]) if len(args) > 1 else False
                name     = ctypes.c_wchar_p(str(args[2])) if len(args) > 2 and args[2] else None
                h_evt    = getattr(kernel32, 'CreateEventW')(None, manual, initial, name)
                if not h_evt:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"CreateEventW failed: {getattr(kernel32, 'GetLastError')()}")
                handle = self._table.allocate(
                    h_evt, 0, EVENT_ALL_ACCESS, CmdFamily.SYNC_OPS,
                    tag=f"Event_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, h_evt)

            elif code == CmdCode.SYNC_SIGNAL:
                handle = int(args[0]) if args else 0
                h_evt  = self._table.resolve(handle) or handle
                ok     = getattr(kernel32, 'SetEvent')(ctypes.wintypes.HANDLE(h_evt))
                return self._make_ok(pkt, int(bool(ok)))

            elif code == CmdCode.SYNC_WAIT:
                # args: (handle, timeout_ms)
                handle     = int(args[0]) if args else 0
                timeout_ms = int(args[1]) if len(args) > 1 else CONN_TIMEOUT_MS
                h_obj      = self._table.resolve(handle) or handle
                result     = getattr(kernel32, 'WaitForSingleObject')(
                    ctypes.wintypes.HANDLE(h_obj),
                    ctypes.wintypes.DWORD(timeout_ms)
                )
                # WAIT_OBJECT_0=0, WAIT_TIMEOUT=0x102, WAIT_ABANDONED=0x80, WAIT_FAILED=0xFFFFFFFF
                return self._make_ok(pkt, result)

            elif code == CmdCode.SYNC_MUTEX:
                # args: (initial_owner: bool, name_or_None)
                initial_owner = bool(args[0]) if args else False
                name          = ctypes.c_wchar_p(str(args[1])) if len(args) > 1 and args[1] else None
                h_mutex       = getattr(kernel32, 'CreateMutexW')(None, initial_owner, name)
                if not h_mutex:
                    return self._make_error(pkt, ET_ERR_OS_ERROR,
                                            f"CreateMutexW failed: {getattr(kernel32, 'GetLastError')()}")
                handle = self._table.allocate(
                    h_mutex, 0, MUTEX_ALL_ACCESS, CmdFamily.SYNC_OPS,
                    tag=f"Mutex_{pkt.source_pid}"
                )
                return self._make_ok(pkt, handle, h_mutex)

            elif code == CmdCode.SYNC_SEMAPHORE:
                return self._handle_sync_semaphore(pkt)
            elif code == CmdCode.SYNC_RELEASE_SEM:
                # Validate semaphore access before release
                sem_args = unpack_args(pkt.payload)
                if sem_args:
                    sem_entry = self._table.get_entry(int(sem_args[0]))
                    if sem_entry and sem_entry.flags != SEMAPHORE_ALL_ACCESS:
                        self._log.mediation(
                            "SYNC_RELEASE_SEM: handle 0x%08X access=0x%X expected=0x%X PID=%d",
                            int(sem_args[0]), sem_entry.flags,
                            SEMAPHORE_ALL_ACCESS, pkt.source_pid)
                return self._handle_sync_release_sem(pkt)
            elif code == CmdCode.SYNC_WAIT_MULTIPLE:
                return self._handle_sync_wait_multiple(pkt)
            elif code == CmdCode.SYNC_RESET_EVENT:
                return self._handle_sync_reset_event(pkt)
            elif code == CmdCode.SYNC_CLOSE:
                return self._handle_sync_close(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown SYNC_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 10: NET OPS
    # -------------------------------------------------------------------------

    def _handle_net_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._family_locks[CmdFamily.NET_OPS]:

            if code == CmdCode.NET_SOCKET64:
                # args: (af: int, sock_type: int, protocol: int, recv_buf_size: int)
                af        = int(args[0]) if args else AF_INET
                sock_type = int(args[1]) if len(args) > 1 else SOCK_STREAM
                proto     = int(args[2]) if len(args) > 2 else IPPROTO_TCP
                # recv buf size: max of IPC_BUFFER_SIZE or user-specified (64-bit can handle large)
                recv_size = int(args[3]) if len(args) > 3 else IPC_BUFFER_SIZE

                # Use Python socket for clean 64-bit socket management
                try:
                    sock = socket.socket(af, sock_type, proto)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_size)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, recv_size)
                    # Get the raw fd/handle
                    raw_fd = sock.fileno()
                    handle = self._table.allocate(
                        raw_fd, recv_size, 0, CmdFamily.NET_OPS,
                        tag=f"Socket_{pkt.source_pid}"
                    )
                    # Detach so we track lifetime manually
                    sock.detach()
                    return self._make_ok(pkt, handle, raw_fd, recv_size)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

            elif code == CmdCode.NET_BIND64:
                # args: (socket_handle, addr_str: str, port: int)
                sock_handle = int(args[0]) if args else 0
                addr_str    = str(args[1]) if len(args) > 1 else "0.0.0.0"
                port        = int(args[2]) if len(args) > 2 else 0
                raw_fd      = self._table.resolve(sock_handle) or sock_handle
                try:
                    sock = socket.fromfd(raw_fd, AF_INET, SOCK_STREAM)
                    sock.bind((addr_str, port))
                    bound_addr, bound_port = sock.getsockname()
                    sock.detach()
                    return self._make_ok(pkt, bound_port)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

            elif code == CmdCode.NET_SEND64:
                # args: (socket_handle, data: bytes)
                sock_handle = int(args[0]) if args else 0
                data        = args[1] if len(args) > 1 and isinstance(args[1], bytes) else b""
                raw_fd      = self._table.resolve(sock_handle) or sock_handle
                try:
                    sock = socket.fromfd(raw_fd, AF_INET, SOCK_STREAM)
                    sent = sock.send(data)
                    sock.detach()
                    return self._make_ok(pkt, sent)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

            elif code == CmdCode.NET_RECV64:
                # args: (socket_handle, max_size: int)
                sock_handle = int(args[0]) if args else 0
                max_size    = min(int(args[1]) if len(args) > 1 else IPC_BUFFER_SIZE,
                                  IPC_BUFFER_SIZE - PDT_HEADER_SIZE - 64)
                raw_fd      = self._table.resolve(sock_handle) or sock_handle
                try:
                    sock = socket.fromfd(raw_fd, AF_INET, SOCK_STREAM)
                    data = sock.recv(max_size)
                    sock.detach()
                    payload, c = pack_args(data)
                    return self._make_response(pkt, code, ETPacket.FLAG_RESPONSE, c, payload)
                except OSError as exc:
                    return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

            elif code == CmdCode.NET_CONNECT64:
                return self._handle_net_connect(pkt)
            elif code == CmdCode.NET_LISTEN64:
                return self._handle_net_listen(pkt)
            elif code == CmdCode.NET_ACCEPT64:
                return self._handle_net_accept(pkt)
            elif code == CmdCode.NET_CLOSE64:
                return self._handle_net_close(pkt)
            elif code == CmdCode.NET_SELECT64:
                return self._handle_net_select(pkt)
            elif code == CmdCode.NET_SOCKOPT64:
                return self._handle_net_sockopt(pkt)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown NET_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 11: PYTHON OPS
    # -------------------------------------------------------------------------

    def _handle_python_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)
        code = pkt.cmd_code

        with self._py_lock:

            if code == CmdCode.PY_INIT:
                # Initialize the 64-bit Python execution environment
                if self._py_initialized:
                    return self._make_ok(pkt, 1, sys.version)
                try:
                    # Set up isolated globals for the target PID
                    pid  = pkt.source_pid
                    genv = {
                        "__builtins__": __builtins__,
                        "__name__":     f"et32_py_{pid}",
                        "__file__":     "<ET32Bridge>",
                    }
                    self._py_globals[pid] = genv
                    self._py_initialized  = True
                    # PY_OPS is family d=11 (near-full, imaginary axis dominant)
                    # T-axis cascade coherence horizon: N_MAX_IMAGINARY = 2
                    # Sequential PY_EXEC calls should not exceed this without re-grounding
                    self._log.info(
                        "Python 64-bit env initialized for PID %d: %s "
                        "(T-axis coherence horizon: N_max_θ=%d)",
                        pid, sys.version, N_MAX_IMAGINARY)
                    # Return version, executable, and T-coherence horizon to client
                    return self._make_ok(pkt, 1, sys.version, sys.executable,
                                         N_MAX_IMAGINARY)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_PYTHON_ERROR, str(exc))

            elif code == CmdCode.PY_EXEC:
                # Execute a Python code string in the 64-bit interpreter
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "PY_EXEC needs code string")
                code_str = str(args[0])
                pid      = pkt.source_pid
                genv     = self._py_globals.get(pid, {})

                # Shell command execution: "!" prefix runs via subprocess
                # This gives the 32-bit client access to native 64-bit system commands
                if code_str.startswith("!"):
                    shell_cmd = code_str[1:].strip()
                    if not shell_cmd:
                        return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                                "PY_EXEC shell command is empty")
                    try:
                        proc_result = subprocess.run(
                            shell_cmd, shell=True, capture_output=True,
                            text=True, timeout=CONN_TIMEOUT_MS / 1000.0,
                        )
                        output = proc_result.stdout
                        if proc_result.stderr:
                            output += "\n[stderr]\n" + proc_result.stderr
                        output += f"\n[exit_code={proc_result.returncode}]"
                        payload, c = pack_args(output)
                        return self._make_response(pkt, CmdCode.PY_EXEC,
                                                   ETPacket.FLAG_RESPONSE, c, payload)
                    except subprocess.TimeoutExpired:
                        return self._make_error(pkt, ET_ERR_TIMEOUT,
                                                f"Shell command timed out: {shell_cmd[:64]}")
                    except Exception as exc:
                        return self._make_error(pkt, ET_ERR_PYTHON_ERROR,
                                                f"Shell command failed: {exc}")

                # Standard Python code execution path
                old_stdout = sys.stdout
                try:
                    import io
                    sys.stdout = io.StringIO()
                    exec(compile(code_str, "<ET32Bridge>", "exec"), genv)
                    output = sys.stdout.getvalue()
                    payload, c = pack_args(output)
                    return self._make_response(pkt, CmdCode.PY_EXEC,
                                               ETPacket.FLAG_RESPONSE, c, payload)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_PYTHON_ERROR, str(exc))
                finally:
                    sys.stdout = old_stdout

            elif code == CmdCode.PY_IMPORT:
                # Import a 64-bit module
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "PY_IMPORT needs module name")
                module_name = str(args[0])
                try:
                    mod = importlib.import_module(module_name)
                    # Store in globals for the PID
                    pid  = pkt.source_pid
                    genv: Dict[str, Any] = self._py_globals.setdefault(pid, {"__builtins__": __builtins__})
                    genv[module_name.split(".")[-1]] = mod
                    version = getattr(mod, "__version__", "unknown")
                    return self._make_ok(pkt, 1, module_name, version)
                except ImportError as exc:
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, str(exc))

            elif code == CmdCode.PY_CALL:
                # Call a Python function by name: args = (func_name: str, *func_args)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "PY_CALL needs func name")
                func_name = str(args[0])
                func_args = args[1:]
                pid       = pkt.source_pid
                genv      = self._py_globals.get(pid, {})
                # Resolve dotted names (e.g. "numpy.zeros")
                # First part: dict lookup from genv → returns Any (not dict)
                # Remaining parts: getattr chain → remains Any
                parts = func_name.split(".")
                try:
                    obj: Any = genv[parts[0]]
                    for part in parts[1:]:
                        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
                    if not callable(obj):
                        return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                                f"'{func_name}' resolved to non-callable {type(obj).__name__}")
                    result = obj(*func_args)
                    payload, c = pack_args(str(result))
                    return self._make_response(pkt, CmdCode.PY_CALL,
                                               ETPacket.FLAG_RESPONSE, c, payload)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_PYTHON_ERROR, str(exc))

            elif code == CmdCode.PY_GETOBJ:
                # Get a Python object's repr/value: args = (obj_name: str,)
                obj_name = str(args[0]) if args else ""
                pid      = pkt.source_pid
                genv     = self._py_globals.get(pid, {})
                # Resolve dotted names — first part from dict, rest via getattr
                parts = obj_name.split(".")
                try:
                    obj: Any = genv[parts[0]]
                    for part in parts[1:]:
                        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
                    payload, c = pack_args(repr(obj))
                    return self._make_response(pkt, CmdCode.PY_GETOBJ,
                                               ETPacket.FLAG_RESPONSE, c, payload)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] Unhandled exception: %s", exc)
                    record_error(ETOperationError(
                        f"Unhandled exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_NOT_FOUND, str(exc))

            elif code == CmdCode.PY_EVAL:
                # Evaluate a Python expression and return the result value.
                # Unlike PY_EXEC (which uses exec()), PY_EVAL uses eval() which
                # returns a value. C2C UI callbacks (CyInterface queries,
                # CvGameUtils evaluations) need expression results.
                #
                # ET derivation (Identification Principle):
                #   P = the Python globals namespace (substrate)
                #   D = the expression string (Descriptor — constrains what is computed)
                #   T = eval() (Traverser — navigates D through P to produce E)
                #   E = the result value (Exception — grounded)
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                            "PY_EVAL needs expression string")
                expr_str = str(args[0])
                pid      = pkt.source_pid
                genv     = self._py_globals.get(pid, {})
                if not genv:
                    # Auto-initialize if PY_INIT wasn't called explicitly
                    genv = {
                        "__builtins__": __builtins__,
                        "__name__":     f"et32_py_{pid}",
                        "__file__":     "<ET32Bridge>",
                    }
                    self._py_globals[pid] = genv
                    self._py_initialized  = True

                try:
                    result = eval(compile(expr_str, "<ET32Bridge_eval>", "eval"), genv)
                    # Serialize the result: try numeric first (C2C expects DWORD often),
                    # fall back to string representation
                    if isinstance(result, (int, float)):
                        payload, c = pack_args(result, repr(result))
                    elif isinstance(result, bool):
                        payload, c = pack_args(int(result), repr(result))
                    elif result is None:
                        payload, c = pack_args(0, "None")
                    else:
                        payload, c = pack_args(repr(result))
                    return self._make_response(pkt, CmdCode.PY_EVAL,
                                               ETPacket.FLAG_RESPONSE, c, payload)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] PY_EVAL exception: %s", exc)
                    record_error(ETOperationError(
                        f"PY_EVAL exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_PYTHON_ERROR, str(exc))

            elif code == CmdCode.PY_SETOBJ:
                # Set a variable in the 64-bit Python globals from C side.
                # C2C event system needs C++→Python variable injection:
                # gc.getPlayer(), gc.getGame(), gc.getMap() etc. are set by the
                # engine before calling Python callbacks.
                #
                # ET derivation (Descriptor Gap Principle):
                #   The gap between C++ game state and Python script visibility
                #   is itself a Descriptor. PY_SETOBJ closes this gap by injecting
                #   the C++ value into the Python D-space (globals).
                #
                # args: (obj_name: str, value_repr: str)
                # value_repr is eval()'d to reconstruct the object
                if len(args) < 2:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                            "PY_SETOBJ needs name and value")
                obj_name  = str(args[0])
                value_str = str(args[1])
                pid       = pkt.source_pid
                genv      = self._py_globals.setdefault(pid, {
                    "__builtins__": __builtins__,
                    "__name__":     f"et32_py_{pid}",
                    "__file__":     "<ET32Bridge>",
                })
                try:
                    # Attempt to reconstruct the value via eval.
                    # Simple types (int, float, str, list, dict, bool, None) are safe.
                    # Complex types: the C side should serialize as repr().
                    value = eval(compile(value_str, "<ET32Bridge_setobj>", "eval"), genv)
                    # Support dotted names: "gc.player" sets genv["gc"]["player"]
                    parts = obj_name.split(".")
                    if len(parts) == 1:
                        genv[parts[0]] = value
                    else:
                        # Navigate to parent, set final attr
                        parent: Any = genv[parts[0]]
                        for part in parts[1:-1]:
                            parent = parent[part] if isinstance(parent, dict) else getattr(parent, part)
                        if isinstance(parent, dict):
                            parent[parts[-1]] = value
                        else:
                            setattr(parent, parts[-1], value)
                    self._log.debug(
                        "PY_SETOBJ: %s = %s (PID=%d)",
                        obj_name, repr(value)[:64], pid)
                    return self._make_ok(pkt, 1)
                except Exception as exc:
                    self._log.exc(
                        "[et_host64] PY_SETOBJ exception: %s", exc)
                    record_error(ETOperationError(
                        f"PY_SETOBJ exception in et_host64: {exc}",
                        cause=exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))
                    return self._make_error(pkt, ET_ERR_PYTHON_ERROR, str(exc))

            elif code == CmdCode.PY_SYSPATH:
                # Append or prepend a directory to sys.path.
                # C2C mods expect these on the Python path:
                #   Assets/Python/
                #   Assets/Python/Screens/
                #   Assets/Python/Contrib/
                #   Assets/Python/EntryPoints/
                #
                # args: (path: str, mode: uint32)
                # mode 0 = append (sys.path.append), mode 1 = prepend (sys.path.insert(0, ...))
                #
                # ET derivation (Subsumption Law):
                #   sys.path is the D-set that constrains Python's module resolution.
                #   Adding a path adds a Descriptor. The module search now subsumes
                #   the new directory without remainder.
                if not args:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                            "PY_SYSPATH needs path string")
                path_str = str(args[0])
                mode     = int(args[1]) if len(args) > 1 else 0
                # Normalize and validate path
                norm_path = os.path.normpath(path_str)
                if norm_path not in sys.path:
                    if mode == 1:
                        sys.path.insert(0, norm_path)
                    else:
                        sys.path.append(norm_path)
                    self._log.info(
                        "PY_SYSPATH: %s '%s' (PID=%d, total paths=%d)",
                        "prepended" if mode == 1 else "appended",
                        norm_path, pkt.source_pid, len(sys.path))
                else:
                    self._log.debug(
                        "PY_SYSPATH: '%s' already in sys.path (PID=%d)",
                        norm_path, pkt.source_pid)
                return self._make_ok(pkt, len(sys.path), norm_path)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown PYTHON_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # FAMILY 12: COMPOUND OPS
    # -------------------------------------------------------------------------

    def _handle_compound_ops(self, pkt: ETPacket) -> ETPacket:
        args = unpack_args(pkt.payload)


        # CMD_CTRL_ERR — structured error report from et_bridge32.dll
        # The DLL sends this when any Windows API call fails in the target process.
        # args: (file:line, function, operation, os_error, et_pid)
        if pkt.cmd_code == CmdCode.CTRL_ERR:
            location   = str(args[0]) if len(args) > 0 else "unknown"
            func_name  = str(args[1]) if len(args) > 1 else "unknown"
            operation  = str(args[2]) if len(args) > 2 else "unknown"
            os_error   = int(args[3]) if len(args) > 3 else 0
            src_pid    = pkt.source_pid

            # Build a full ETOperationError from the DLL report
            err = ETWindowsAPIError(
                f"{func_name}() in target PID {src_pid}",
                os_error  = os_error,
                et_pid    = src_pid,
                et_family = pkt.cmd_family,
                et_code   = pkt.cmd_code,
                severity  = ETErrorSeverity.BOUNDARY,
                location_str = location,
                operation = operation,
            )
            self._log.error_op(err)
            # ACK the error report so the DLL's et_call() returns cleanly
            return self._make_ok(pkt, 1)

        # DYNAMIC SYSCALL — dedicated command code for dynamic forwarding
        if pkt.cmd_code == CmdCode.DYNAMIC_SYSCALL:
            if not args:
                return self._make_error(pkt, ET_ERR_INVALID_ARGS,
                                        'DYNAMIC_SYSCALL needs service_number')
            service_number = int(args[0]) & 0x0FFF
            raw_args = [int(a) & 0xFFFFFFFF for a in args[1:13]]
            while len(raw_args) < 12: raw_args.append(0)
            if self._wow64 is not None:
                result = self._wow64.dispatch_service(
                    pkt.source_pid, service_number, raw_args
                )
                if result is not None:
                    # Validate NTSTATUS — if error status, record but still return to client
                    try:
                        ntstatus_check(result, f"dispatch_service(0x{service_number:04X})",
                                       et_pid=pkt.source_pid,
                                       et_family=CmdFamily.COMPOUND_OPS,
                                       et_code=CmdCode.DYNAMIC_SYSCALL)
                    except ETOperationError as _nt_err:
                        record_error(_nt_err)
                    return self._make_ok(pkt, result)

            # WOW64 hook did not handle this service number.
            # ET derivation (Descriptor Gap Principle): the gap between "WOW64 can't
            # handle it" and "pass-through to original" is itself a Descriptor.
            # Heaven's Gate can close this gap by executing the 64-bit syscall directly
            # from the broker process, for cases where the 32-bit WOW64 thunk is
            # insufficient (e.g., D3D calls, large-buffer I/O, extended NT APIs).
            if self._heaven is not None:
                # Attempt Heaven's Gate resolution: look up the 64-bit ntdll
                # syscall address for this service number from the runtime-built
                # service table and call it directly via Heaven's Gate.
                # This is the d=12 path (manifold-complete: direct NT system call).
                #
                # The service table (ETServiceTable) was built by PE reflection of
                # ntdll64.dll at startup. dispatch_service may have returned None
                # for two reasons: (a) lookup miss, or (b) ctypes call exception.
                # Heaven's Gate uses a different calling mechanism and may succeed
                # where the standard ctypes path failed.
                try:
                    # Look up the 64-bit function address from the service table
                    # (the same table ETDynamic64Caller uses, but we retry via
                    # Heaven's Gate calling convention instead of ctypes WINFUNCTYPE).
                    nt_func_addr = None
                    if self._wow64 is not None:
                        nt_func_addr = self._wow64.lookup_service(service_number)

                    if not nt_func_addr:
                        # Service number truly absent from ntdll64 PE reflection —
                        # no function exists with this service number in the running
                        # system's ntdll. Genuine pass-through.
                        self._log.mediation(
                            "DYNAMIC_SYSCALL 0x%04X: not in service table (%d entries), "
                            "Heaven's Gate cannot resolve — pass-through PID=%d",
                            service_number,
                            self._wow64.service_table_size() if self._wow64 else 0,
                            pkt.source_pid)
                    else:
                        # Address found — retry via Heaven's Gate direct call.
                        # NT syscalls use the Windows x64 calling convention:
                        # RCX, RDX, R8, R9 for the first 4 args, rest on stack.
                        # Heaven's Gate handles this translation.
                        hg_result = self._heaven.call_64bit_function(
                            nt_func_addr, raw_args[:S]  # pass all S=12 captured args
                        )
                        if hg_result is not None and hg_result != 0xC000001C:
                            self._log.info(
                                "DYNAMIC_SYSCALL 0x%04X: Heaven's Gate resolved → 0x%08X PID=%d",
                                service_number, hg_result, pkt.source_pid)
                            return self._make_ok(pkt, hg_result)
                except Exception as _hg_exc:
                    self._log.exc(
                        "[et_host64] Heaven's Gate fallback exception: %s", _hg_exc)
                    record_error(ETOperationError(
                        f"Heaven's Gate fallback for syscall 0x{service_number:04X}: {_hg_exc}",
                        cause=_hg_exc,
                        os_error=ctypes.GetLastError(),
                        severity=ETErrorSeverity.BOUNDARY,
                        depth=2,
                    ))

            # Service not handled by WOW64 or Heaven's Gate — record ETHookError for diagnostics
            record_error(ETHookError(
                f"Dynamic syscall 0x{service_number:04X} unhandled",
                pid=pkt.source_pid,
                et_family=CmdFamily.COMPOUND_OPS,
                et_code=CmdCode.DYNAMIC_SYSCALL,
            ))
            return self._make_ok(pkt, 0xC000001C)

        code = pkt.cmd_code

        with self._family_locks[CmdFamily.COMPOUND_OPS]:

            if code == CmdCode.COMPOUND_BATCH:
                """
                Execute a batch of sub-packets.
                Payload format: uint32 count, then N concatenated serialised ETPackets.
                Each sub-packet is dispatched independently.
                Returns a batch response: uint32 count, then N response payloads.

                Max batch size = S = 12 (manifold symmetry governs compound depth).
                """
                if len(args) < 1:
                    return self._make_error(pkt, ET_ERR_INVALID_ARGS, "COMPOUND_BATCH needs sub-packets")

                # The payload is a raw bytes blob of concatenated ETPackets
                sub_data    = pkt.payload
                offset      = 0
                results     = []
                count       = 0
                max_batch   = S  # 12 sub-operations per batch

                # Level 3 𝒜_I — Sublattice coherence pre-scan
                # (incoherence_filter_-_lattice.txt Level 3)
                # Before dispatching, scan all sub-packet pairs for sublattice
                # compatibility: d(ri·rj) = 12/gcd(ki+kj, 12).
                # Treat cmd_family as the lattice k-proxy for each sub-operation.
                # If any pair is incoherent (lcm(di,dj) > 12), reject the batch.
                sub_pkts_preview = []
                _scan_offset = 0
                while _scan_offset < len(sub_data):
                    _sp = ETPacket.deserialise(sub_data[_scan_offset:])
                    if _sp is None:
                        record_error(ETPacketError(
                            f"COMPOUND_BATCH pre-scan: sub-packet deserialization failed "
                            f"at offset {_scan_offset}/{len(sub_data)}",
                            et_pid=pkt.source_pid,
                            et_family=CmdFamily.COMPOUND_OPS,
                            et_code=CmdCode.COMPOUND_BATCH,
                        ))
                        break
                    sub_pkts_preview.append(_sp)
                    _scan_offset += PDT_HEADER_SIZE + (
                        len(_sp.payload) if _sp.payload else 0)
                    if len(sub_pkts_preview) >= max_batch:
                        break

                _ai3_violated = False
                for _i in range(len(sub_pkts_preview)):
                    for _j in range(_i + 1, len(sub_pkts_preview)):
                        _ki = sub_pkts_preview[_i].cmd_family
                        _kj = sub_pkts_preview[_j].cmd_family
                        _ai3, _d_combined = sublattice_incoherence(_ki, _kj)
                        # combined_sublattice gives the LCM(d_r, d_θ) resolution class
                        # (ET_Where_Does_Zero_Over_Zero_Come_In §15, §42).
                        # d=12 in 41.7% of base pairs — LCM amplification is normal.
                        # Coprime theorem: gcd(ki,kj)=1 → d=12 always (no incoherence).
                        # At 12ET base: 𝒜_I=1 never occurs (all LCMs ≤ 12).
                        # Use SUBLATTICE_LCM_TABLE for pre-computed base-family lookup
                        _lcm_key = (_ki, _kj)
                        _d_comb_table = SUBLATTICE_LCM_TABLE.get(_lcm_key)
                        _d_comb_new = combined_sublattice(_ki, _kj)
                        # Cross-validate: pre-computed table vs dynamic computation
                        if _d_comb_table is not None and _d_comb_table != _d_comb_new:
                            record_error(ETDispatchError(
                                f"Level 3 LCM table mismatch: table({_ki},{_kj})="
                                f"{_d_comb_table} vs dynamic={_d_comb_new}",
                                et_pid=pkt.source_pid,
                                et_family=CmdFamily.COMPOUND_OPS,
                                et_code=CmdCode.COMPOUND_BATCH,
                            ))
                        # Validate d_combined ≤ D_MAX (132 = N(N-1))
                        if _d_comb_new > D_MAX:
                            self._log.incoherence(
                                "Level 3: d_combined=%d exceeds D_MAX=%d "
                                "— beyond the 27720ET lattice ceiling. PID=%d",
                                _d_comb_new, D_MAX, pkt.source_pid)
                        self._log.mediation(
                            "Level 3: sub-ops d%d ⊕ d%d → d_combined=%d (%s) PID=%d",
                            _ki, _kj, _d_comb_new,
                            "full-res EM" if _d_comb_new == 12 else
                            "D+T mixed" if gaussian_prime_type(_d_comb_new) == "D+T" else
                            gaussian_prime_type(_d_comb_new),
                            pkt.source_pid,
                        )
                        if _ai3 == 1:
                            # This only happens if lcm > S (extended families, not at 12ET base)
                            self._log.warning_di(
                                "Level 3 𝒜_I: COMPOUND_BATCH sublattice mismatch "
                                "— sub-ops d%d and d%d, d_combined=%d > N=%d. PID=%d",
                                _ki, _kj, _d_combined, S, pkt.source_pid,
                            )
                            record_error(ETDispatchError(
                                f"Level 3 sublattice mismatch COMPOUND_BATCH "
                                f"(d{_ki} ⊕ d{_kj}→{_d_combined}, LCM>{S})",
                                et_pid    = pkt.source_pid,
                                et_family = CmdFamily.COMPOUND_OPS,
                                et_code   = CmdCode.COMPOUND_BATCH,
                                severity  = ETErrorSeverity.MEDIATION,
                            ))
                            _ai3_violated = True

                if _ai3_violated:
                    return self._make_error(
                        pkt, ET_ERR_INVALID_ARGS,
                        "COMPOUND_BATCH: sublattice incoherence — sub-ops incompatible"
                    )

                while offset < len(sub_data) and count < max_batch:
                    sub_pkt = ETPacket.deserialise(sub_data[offset:])
                    if sub_pkt is None:
                        # Packet deserialization failure — record ETPacketError
                        record_error(ETPacketError(
                            f"COMPOUND_BATCH sub-packet #{count} deserialization failed "
                            f"at offset {offset}/{len(sub_data)}",
                            et_pid=pkt.source_pid,
                            et_family=CmdFamily.COMPOUND_OPS,
                            et_code=CmdCode.COMPOUND_BATCH,
                        ))
                        break
                    sub_response = self.dispatch(sub_pkt)
                    results.append(sub_response.serialise())
                    offset += PDT_HEADER_SIZE + (len(sub_pkt.payload) if sub_pkt.payload else 0)
                    count  += 1

                # Concatenate results
                combined   = b"".join(results)
                # Log the combined resolution class of the batch
                if sub_pkts_preview:
                    batch_families = [sp.cmd_family for sp in sub_pkts_preview[:count]]
                    if batch_families:
                        d_batch = batch_families[0]
                        for _df in batch_families[1:]:
                            d_batch = combined_sublattice(d_batch, _df)
                        self._log.mediation(
                            "COMPOUND_BATCH(%d ops) d_combined=%d PID=%d",
                            count, d_batch, pkt.source_pid,
                        )
                batch_hdr  = struct.pack("<I", count)
                payload    = batch_hdr + combined
                return self._make_response(
                    pkt, CmdCode.COMPOUND_BATCH,
                    ETPacket.FLAG_RESPONSE, count, payload
                )

            elif code == CmdCode.COMPOUND_ATOMIC:
                """
                Execute sub-packets atomically: if any fails, roll back previous ones.
                Roll-back is best-effort (VirtualFree on allocations, CloseHandle on handles).
                """
                sub_data   = pkt.payload
                offset     = 0
                executed   = []  # (response, sub_pkt)
                count      = 0
                max_atomic = S

                while offset < len(sub_data) and count < max_atomic:
                    sub_pkt = ETPacket.deserialise(sub_data[offset:])
                    if sub_pkt is None:
                        # Packet deserialization failure — record ETPacketError
                        record_error(ETPacketError(
                            f"COMPOUND_ATOMIC sub-packet #{count} deserialization failed "
                            f"at offset {offset}/{len(sub_data)}",
                            et_pid=pkt.source_pid,
                            et_family=CmdFamily.COMPOUND_OPS,
                            et_code=CmdCode.COMPOUND_ATOMIC,
                        ))
                        break
                    sub_response = self.dispatch(sub_pkt)
                    if sub_response.cmd_code == CmdCode.CTRL_ERR:
                        # Failure — rollback
                        self._rollback_executed(executed)
                        return self._make_error(pkt, ET_ERR_OS_ERROR,
                                                f"COMPOUND_ATOMIC failed at step {count}")
                    executed.append((sub_response, sub_pkt))
                    offset += PDT_HEADER_SIZE + len(sub_pkt.payload)
                    count  += 1

                results  = [r.serialise() for r, _ in executed]
                combined = b"".join(results)
                hdr      = struct.pack("<I", count)
                return self._make_response(
                    pkt, CmdCode.COMPOUND_ATOMIC,
                    ETPacket.FLAG_RESPONSE, count, hdr + combined
                )

            elif code == CmdCode.COMPOUND_ROLLBACK:
                # Explicit rollback: args = (handle1, handle2, ...) to release
                released = 0
                for handle in args:
                    # ETHandleMath.is_bridge_handle validates HANDLE_BASE ≤ h ≤ HANDLE_MAX
                    if isinstance(handle, int) and ETHandleMath.is_bridge_handle(handle):
                        lattice_pos = ETHandleMath.handle_lattice_position(handle)
                        addr64 = self._table.resolve(handle)
                        if addr64:
                            self._log.mediation(
                                "COMPOUND_ROLLBACK: releasing handle 0x%08X "
                                "lattice_pos=%d addr64=0x%016X PID=%d",
                                handle, lattice_pos, addr64, pkt.source_pid)
                            getattr(kernel32, 'VirtualFree')(
                                ctypes.c_void_p(addr64),
                                ctypes.c_size_t(0),
                                ctypes.wintypes.DWORD(MEM_RELEASE)
                            )
                            self._table.release(handle)
                            released += 1
                    elif isinstance(handle, int) and handle > HANDLE_MAX:
                        # Handle exceeds HANDLE_MAX — invalid, record the anomaly
                        record_error(ETHandleError(
                            f"COMPOUND_ROLLBACK: handle 0x{handle:08X} exceeds "
                            f"HANDLE_MAX 0x{HANDLE_MAX:08X}",
                            handle=handle, et_pid=pkt.source_pid,
                            et_family=CmdFamily.COMPOUND_OPS,
                            et_code=CmdCode.COMPOUND_ROLLBACK,
                        ))
                return self._make_ok(pkt, released)

        return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown COMPOUND_OPS code 0x{pkt.cmd_code:02X}")

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _rollback_executed(self, executed: List[Tuple[ETPacket, ETPacket]]):
        """Best-effort rollback of already-executed steps in a COMPOUND_ATOMIC."""
        for resp, sub_pkt in reversed(executed):
            args = unpack_args(resp.payload)
            if args and isinstance(args[0], int) and args[0] >= HANDLE_BASE:
                handle = args[0]
                addr64 = self._table.resolve(handle)
                if addr64:
                    try:
                        getattr(kernel32, 'VirtualFree')(
                            ctypes.c_void_p(addr64),
                            ctypes.c_size_t(0),
                            ctypes.wintypes.DWORD(MEM_RELEASE)
                        )
                    except Exception as _et_exc:
                        self._log.exc(
                            "[et_host64] Unhandled exception: %s", _et_exc)
                        record_error(ETOperationError(
                            f"Unhandled exception in et_host64: {_et_exc}",
                            cause=_et_exc,
                            os_error=ctypes.GetLastError(),
                            severity=ETErrorSeverity.BOUNDARY,
                            depth=2,
                        ))
                        pass
                    self._table.release(handle)
        # Fire-and-forget metrics update for the rollback — safe_call absorbs any error
        safe_call(
            self._metrics.record, CmdFamily.COMPOUND_OPS, 0.0, False,
            operation="metrics.record after COMPOUND_ATOMIC rollback",
            log_fn=self._log.incoherence,
        )

    def _make_ok(self, request: ETPacket, *result_args) -> ETPacket:
        """Build a success response packet."""
        payload, count = pack_args(*result_args)
        return ETPacket(
            source_pid  = self._broker_pid,
            dest_pid    = request.source_pid,
            space_token = request.space_token,
            cmd_family  = request.cmd_family,
            cmd_code    = CmdCode.CTRL_ACK,
            flags       = ETPacket.FLAG_RESPONSE,
            arg_count   = count,
            payload     = payload,
            sequence    = request.sequence,
        )

    def _make_error(self, request: ETPacket, error_code: int, message: str = "") -> ETPacket:
        """Build an error response packet."""
        payload, count = pack_args(error_code, message)
        return ETPacket(
            source_pid  = self._broker_pid,
            dest_pid    = request.source_pid,
            space_token = request.space_token,
            cmd_family  = request.cmd_family,
            cmd_code    = CmdCode.CTRL_ERR,
            flags       = ETPacket.FLAG_RESPONSE | ETPacket.FLAG_ERROR,
            arg_count   = count,
            payload     = payload,
            sequence    = request.sequence,
        )

    def _make_response(
        self,
        request: ETPacket,
        cmd_code: int,
        flags: int,
        arg_count: int,
        payload: bytes
    ) -> ETPacket:
        """Build a response with explicit code and payload."""
        return ETPacket(
            source_pid  = self._broker_pid,
            dest_pid    = request.source_pid,
            space_token = request.space_token,
            cmd_family  = request.cmd_family,
            cmd_code    = cmd_code,
            flags       = flags,
            arg_count   = arg_count,
            payload     = payload,
            sequence    = request.sequence,
        )

    def _init_winsock(self):
        """Initialise Winsock 2.2 for NET_OPS."""
        try:
            class WSADATA(ctypes.Structure):
                """Win32 WSADATA — D-descriptor for Winsock 2.2 initialization state."""
                _fields_ = [
                    ("wVersion",      ctypes.c_ushort),
                    ("wHighVersion",  ctypes.c_ushort),
                    ("iMaxSockets",   ctypes.c_ushort),
                    ("iMaxUdpDg",     ctypes.c_ushort),
                    ("lpVendorInfo",  ctypes.c_char_p),
                    ("szDescription", ctypes.c_char * 257),
                    ("szSystemStatus", ctypes.c_char * 129),
                ]
            wsa_data = WSADATA()
            ret = getattr(ws2_32, 'WSAStartup')(0x0202, ctypes.byref(wsa_data))
            if ret == 0:
                self._winsock_initialized = True
        except Exception as _et_exc:
            self._log.exc(
                "[et_host64] Unhandled exception: %s", _et_exc)
            record_error(ETOperationError(
                f"Unhandled exception in et_host64: {_et_exc}",
                cause=_et_exc,
                os_error=ctypes.GetLastError(),
                severity=ETErrorSeverity.BOUNDARY,
                depth=2,
            ))
            pass  # Winsock already initialized by Python's socket module

    @property
    def broker_pid(self) -> int:
        """Return the broker process ID — the T-traverser's OS identity."""
        return self._broker_pid

    # =========================================================================
    # COMPLETENESS ADDITIONS — All 36 gaps closed
    # =========================================================================

    # ── MEMORY_BASIC additions ─────────────────────────────────────────────

    def _handle_global_mem_status(self, pkt: ETPacket) -> ETPacket:
        """GlobalMemoryStatusEx — true 64-bit memory status. 32-bit gets capped 4GB view."""
        class MEMORYSTATUSEX(ctypes.Structure):
            """Win32 MEMORYSTATUSEX — 64-bit memory status beyond the 4 GB P₃₂ ceiling."""
            _fields_ = [
                ("dwLength",                ctypes.wintypes.DWORD),
                ("dwMemoryLoad",            ctypes.wintypes.DWORD),
                ("ullTotalPhys",            ctypes.c_uint64),
                ("ullAvailPhys",            ctypes.c_uint64),
                ("ullTotalPageFile",        ctypes.c_uint64),
                ("ullAvailPageFile",        ctypes.c_uint64),
                ("ullTotalVirtual",         ctypes.c_uint64),
                ("ullAvailVirtual",         ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = getattr(kernel32, 'GlobalMemoryStatusEx')(ctypes.byref(ms))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GlobalMemoryStatusEx failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt,
            ms.dwMemoryLoad, ms.ullTotalPhys, ms.ullAvailPhys,
            ms.ullTotalPageFile, ms.ullAvailPageFile,
            ms.ullTotalVirtual, ms.ullAvailVirtual)

    def _handle_native_sys_info(self, pkt: ETPacket) -> ETPacket:
        """GetNativeSystemInfo — true 64-bit system info, not filtered by WoW64."""
        # noinspection PyPep8Naming
        class NATIVE_SYSTEM_INFO(ctypes.Structure):
            """GetNativeSystemInfo structure — flat layout for true 64-bit system info."""
            _fields_ = [
                ("wProcessorArchitecture",      ctypes.c_uint16),
                ("wReserved",                   ctypes.c_uint16),
                ("dwPageSize",                  ctypes.wintypes.DWORD),
                ("lpMinimumApplicationAddress", ctypes.c_void_p),
                ("lpMaximumApplicationAddress", ctypes.c_void_p),
                ("dwActiveProcessorMask",       ctypes.c_uint64),
                ("dwNumberOfProcessors",        ctypes.wintypes.DWORD),
                ("dwProcessorType",             ctypes.wintypes.DWORD),
                ("dwAllocationGranularity",     ctypes.wintypes.DWORD),
                ("wProcessorLevel",             ctypes.c_uint16),
                ("wProcessorRevision",          ctypes.c_uint16),
            ]
        si = NATIVE_SYSTEM_INFO()
        getattr(kernel32, 'GetNativeSystemInfo')(ctypes.byref(si))
        return self._make_ok(pkt,
            si.wProcessorArchitecture, si.dwPageSize,
            si.lpMinimumApplicationAddress or 0,
            si.lpMaximumApplicationAddress or 0,
            si.dwNumberOfProcessors, si.dwAllocationGranularity)

    def _handle_close_handle64(self, pkt: ETPacket) -> ETPacket:
        """CloseHandle for any 64-bit handle in the bridge handle table."""
        args = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        addr64 = self._table.resolve(handle)
        if addr64 is None:
            record_error(ETHandleError(
                "resolve failed in CLOSE_HANDLE64",
                handle=handle,
                et_pid=pkt.source_pid, et_family=CmdFamily.MEMORY_BASIC,
            ))
            return self._make_error(pkt, ET_ERR_NOT_FOUND, f"Handle 0x{handle:08X} not found")
        ok = getattr(kernel32, 'CloseHandle')(ctypes.wintypes.HANDLE(addr64))
        try:
            win32_check(ok, "CloseHandle", et_pid=pkt.source_pid,
                        et_family=CmdFamily.MEMORY_BASIC)
        except ETWindowsAPIError as exc:
            record_error(exc)
        self._table.release(handle)
        return self._make_ok(pkt, int(bool(ok)))

    def _handle_duplicate_handle64(self, pkt: ETPacket) -> ETPacket:
        """DuplicateHandle — copy a handle from one process to another."""
        args = unpack_args(pkt.payload)
        src_proc_handle  = int(args[0]) if len(args) > 0 else 0
        src_handle_val   = int(args[1]) if len(args) > 1 else 0
        tgt_proc_handle  = int(args[2]) if len(args) > 2 else 0
        desired_access   = int(args[3]) if len(args) > 3 else 0
        inherit          = bool(args[4]) if len(args) > 4 else False
        options          = int(args[5]) if len(args) > 5 else 2  # DUPLICATE_SAME_ACCESS

        src_h = self._table.resolve(src_proc_handle) or src_proc_handle
        tgt_h = self._table.resolve(tgt_proc_handle) or tgt_proc_handle
        new_handle = ctypes.wintypes.HANDLE(0)
        ok = getattr(kernel32, 'DuplicateHandle')(
            ctypes.wintypes.HANDLE(src_h),
            ctypes.wintypes.HANDLE(src_handle_val),
            ctypes.wintypes.HANDLE(tgt_h),
            ctypes.byref(new_handle),
            ctypes.wintypes.DWORD(desired_access),
            ctypes.wintypes.BOOL(inherit),
            ctypes.wintypes.DWORD(options)
        )
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"DuplicateHandle failed: {getattr(kernel32, 'GetLastError')()}")
        new_h_val = new_handle.value
        # Validate the output handle — NULL means DuplicateHandle silently failed
        try:
            win32_check_handle(new_h_val, "DuplicateHandle output",
                               invalid=0, et_pid=pkt.source_pid)
        except ETWindowsAPIError as exc:
            record_error(exc)
        bridge_handle = self._table.allocate(
            new_h_val, 0, desired_access, CmdFamily.MEMORY_BASIC,
            tag=f"Dup_{pkt.source_pid}")
        return self._make_ok(pkt, bridge_handle, new_h_val)

    # ── THREAD_OPS additions ───────────────────────────────────────────────

    def _handle_thread_set_context(self, pkt: ETPacket) -> ETPacket:
        """SetThreadContext — write 64-bit thread context."""
        args = unpack_args(pkt.payload)
        handle     = int(args[0]) if args else 0
        context_bytes = args[1] if len(args) > 1 and isinstance(args[1], bytes) else b""
        h_thread = self._table.resolve(handle) or handle
        # noinspection PyPep8Naming
        CONTEXT_ALL = 0x10003F
        ctx_buf = ctypes.create_string_buffer(len(context_bytes))
        ctypes.memmove(ctx_buf, context_bytes, len(context_bytes))
        # Ensure ContextFlags is set to CONTEXT_ALL for full context write
        # (offset 48 matches the x64 CONTEXT structure layout)
        if len(ctx_buf) > 52:
            struct.pack_into("<I", ctx_buf, 48, CONTEXT_ALL)
        ok = getattr(kernel32, 'SetThreadContext')(
            ctypes.wintypes.HANDLE(h_thread),
            ctx_buf
        )
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"SetThreadContext failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, 1)

    def _handle_thread_exit_code(self, pkt: ETPacket) -> ETPacket:
        """GetExitCodeThread."""
        args = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_thread = self._table.resolve(handle) or handle
        exit_code = ctypes.wintypes.DWORD(0)
        ok = getattr(kernel32, 'GetExitCodeThread')(
            ctypes.wintypes.HANDLE(h_thread), ctypes.byref(exit_code))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GetExitCodeThread failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, exit_code.value)  # 0x103 = STILL_ACTIVE

    # ── PROCESS_OPS additions ──────────────────────────────────────────────

    def _handle_proc_exit_code(self, pkt: ETPacket) -> ETPacket:
        """GetExitCodeProcess."""
        args = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_proc = self._table.resolve(handle) or handle
        exit_code = ctypes.wintypes.DWORD(0)
        ok = getattr(kernel32, 'GetExitCodeProcess')(
            ctypes.wintypes.HANDLE(h_proc), ctypes.byref(exit_code))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GetExitCodeProcess failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, exit_code.value)  # 0x103 = STILL_ACTIVE

    def _handle_proc_terminate(self, pkt: ETPacket) -> ETPacket:
        """TerminateProcess."""
        args = unpack_args(pkt.payload)
        handle    = int(args[0]) if args else 0
        exit_code = int(args[1]) if len(args) > 1 else 1
        h_proc = self._table.resolve(handle) or handle
        ok = getattr(kernel32, 'TerminateProcess')(
            ctypes.wintypes.HANDLE(h_proc),
            ctypes.wintypes.UINT(exit_code))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"TerminateProcess failed: {getattr(kernel32, 'GetLastError')()}")
        self._table.release(handle)
        return self._make_ok(pkt, 1)

    def _handle_proc_enum(self, pkt: ETPacket) -> ETPacket:
        """EnumProcesses — returns list of all 64-bit process IDs."""
        # noinspection PyPep8Naming
        MAX_PIDS = 4096
        pid_arr  = (ctypes.wintypes.DWORD * MAX_PIDS)()
        cb_needed = ctypes.wintypes.DWORD(0)
        ok = getattr(ctypes.windll.psapi, 'EnumProcesses')(
            ctypes.byref(pid_arr),
            ctypes.wintypes.DWORD(ctypes.sizeof(pid_arr)),
            ctypes.byref(cb_needed)
        )
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"EnumProcesses failed: {getattr(kernel32, 'GetLastError')()}")
        count = cb_needed.value // ctypes.sizeof(ctypes.wintypes.DWORD)
        pids  = [pid_arr[i] for i in range(count)]
        payload, c = pack_args(pids)
        return self._make_response(pkt, pkt.cmd_code, ETPacket.FLAG_RESPONSE, c, payload)

    def _handle_proc_modules(self, pkt: ETPacket) -> ETPacket:
        """EnumProcessModules — list DLLs loaded in a 64-bit process."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_proc = self._table.resolve(handle) or handle
        # noinspection PyPep8Naming
        MAX_MODS = 1024
        mod_arr  = (ctypes.c_void_p * MAX_MODS)()
        cb_needed = ctypes.wintypes.DWORD(0)
        # noinspection PyPep8Naming
        LIST_MODULES_ALL = 0x03
        ok = getattr(ctypes.windll.psapi, 'EnumProcessModulesEx')(
            ctypes.wintypes.HANDLE(h_proc),
            ctypes.byref(mod_arr),
            ctypes.wintypes.DWORD(ctypes.sizeof(mod_arr)),
            ctypes.byref(cb_needed),
            ctypes.wintypes.DWORD(LIST_MODULES_ALL)
        )
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"EnumProcessModulesEx failed: {getattr(kernel32, 'GetLastError')()}")
        count = cb_needed.value // ctypes.sizeof(ctypes.c_void_p)
        mod_names = []
        buf = ctypes.create_unicode_buffer(512)
        for i in range(min(count, MAX_MODS)):
            if mod_arr[i]:
                getattr(ctypes.windll.psapi, 'GetModuleFileNameExW')(
                    ctypes.wintypes.HANDLE(h_proc),
                    ctypes.c_void_p(mod_arr[i]),
                    buf, 512)
                mod_names.append((mod_arr[i], buf.value))
        payload, c = pack_args(mod_names)
        return self._make_response(pkt, pkt.cmd_code, ETPacket.FLAG_RESPONSE, c, payload)

    def _handle_proc_wow64_fs(self, pkt: ETPacket) -> ETPacket:
        """Wow64DisableWow64FsRedirection / Wow64RevertWow64FsRedirection."""
        args   = unpack_args(pkt.payload)
        enable = bool(args[0]) if args else False  # False=disable, True=revert
        if not enable:
            # Disable: Wow64DisableWow64FsRedirection(&old_value)
            old_val = ctypes.c_void_p(0)
            ok = getattr(kernel32, 'Wow64DisableWow64FsRedirection')(ctypes.byref(old_val))
            if not ok:
                return self._make_error(pkt, ET_ERR_OS_ERROR,
                                        f"Wow64DisableWow64FsRedirection failed: {getattr(kernel32, 'GetLastError')()}")
            return self._make_ok(pkt, 1, old_val.value or 0)
        else:
            # Revert: Wow64RevertWow64FsRedirection(old_value)
            old_val = int(args[1]) if len(args) > 1 else 0
            ok = getattr(kernel32, 'Wow64RevertWow64FsRedirection')(ctypes.c_void_p(old_val))
            if not ok:
                return self._make_error(pkt, ET_ERR_OS_ERROR,
                                        f"Wow64RevertWow64FsRedirection failed: {getattr(kernel32, 'GetLastError')()}")
            return self._make_ok(pkt, 1)

    # ── REGISTRY_OPS additions ─────────────────────────────────────────────

    def _handle_reg_create(self, pkt: ETPacket) -> ETPacket:
        """RegCreateKeyExW — create or open a 64-bit registry key."""
        import winreg as _wr
        args   = unpack_args(pkt.payload)
        if not args:
            return self._make_error(pkt, ET_ERR_INVALID_ARGS, "REG_CREATE64 needs hive+subkey")
        hive_val = int(args[0])
        subkey   = str(args[1]) if len(args) > 1 else ""
        access   = int(args[2]) if len(args) > 2 else _wr.KEY_READ  # KEY_WOW64_64KEY added below
        options  = int(args[3]) if len(args) > 3 else _wr.REG_OPTION_NON_VOLATILE

        hive = ctypes.wintypes.HKEY(hive_val)
        new_key  = ctypes.wintypes.HKEY(0)
        disp     = ctypes.wintypes.DWORD(0)
        rc = getattr(advapi32, 'RegCreateKeyExW')(
            hive, subkey, 0, None,
            ctypes.wintypes.DWORD(options),
            ctypes.wintypes.DWORD(access | KEY_WOW64_64KEY),
            None, ctypes.byref(new_key), ctypes.byref(disp)
        )
        if rc != 0:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"RegCreateKeyExW('{subkey}') failed: rc={rc}")
        handle = self._table.allocate(
            new_key.value, 0, access, CmdFamily.REGISTRY_OPS,
            tag=f"RegKey_{subkey[:32]}_{pkt.source_pid}")
        return self._make_ok(pkt, handle, new_key.value, disp.value)  # disp: 1=created 2=opened

    def _handle_reg_delete_key(self, pkt: ETPacket) -> ETPacket:
        """RegDeleteKeyExW — delete a 64-bit registry key."""
        args     = unpack_args(pkt.payload)
        hive_val = int(args[0]) if args else 0
        subkey   = str(args[1]) if len(args) > 1 else ""
        rc = getattr(advapi32, 'RegDeleteKeyExW')(
            ctypes.wintypes.HKEY(hive_val), subkey,
            ctypes.wintypes.DWORD(KEY_WOW64_64KEY), 0)
        if rc != 0:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"RegDeleteKeyExW('{subkey}') failed: rc={rc}")
        return self._make_ok(pkt, 0)

    def _handle_reg_delete_value(self, pkt: ETPacket) -> ETPacket:
        """RegDeleteValueW — delete a 64-bit registry value."""
        args       = unpack_args(pkt.payload)
        handle     = int(args[0]) if args else 0
        value_name = str(args[1]) if len(args) > 1 else ""
        hkey = self._table.resolve(handle) or handle
        rc   = getattr(advapi32, 'RegDeleteValueW')(
            ctypes.wintypes.HKEY(hkey),
            ctypes.c_wchar_p(value_name) if value_name else None)
        if rc != 0:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"RegDeleteValueW('{value_name}') failed: rc={rc}")
        return self._make_ok(pkt, 0)

    def _handle_reg_close(self, pkt: ETPacket) -> ETPacket:
        """RegCloseKey — close a 64-bit registry key handle."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        hkey   = self._table.resolve(handle) or handle
        rc     = getattr(advapi32, 'RegCloseKey')(ctypes.wintypes.HKEY(hkey))
        self._table.release(handle)
        return self._make_ok(pkt, rc)

    # ── SYNC_OPS additions ─────────────────────────────────────────────────

    def _handle_sync_semaphore(self, pkt: ETPacket) -> ETPacket:
        """CreateSemaphoreW."""
        args        = unpack_args(pkt.payload)
        initial     = int(args[0]) if args else 0
        maximum     = int(args[1]) if len(args) > 1 else S  # max = N = 12
        name        = ctypes.c_wchar_p(str(args[2])) if len(args) > 2 and args[2] else None
        # noinspection PyPep8Naming
        SEMAPHORE_ALL_ACCESS = 0x1F0003
        h_sem = getattr(kernel32, 'CreateSemaphoreW')(None,
            ctypes.wintypes.LONG(initial),
            ctypes.wintypes.LONG(maximum), name)
        if not h_sem:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"CreateSemaphoreW failed: {getattr(kernel32, 'GetLastError')()}")
        handle = self._table.allocate(h_sem, maximum, SEMAPHORE_ALL_ACCESS,
            CmdFamily.SYNC_OPS, tag=f"Sem_{pkt.source_pid}")
        return self._make_ok(pkt, handle, h_sem)

    def _handle_sync_release_sem(self, pkt: ETPacket) -> ETPacket:
        """ReleaseSemaphore — increment semaphore count."""
        args       = unpack_args(pkt.payload)
        handle     = int(args[0]) if args else 0
        release_ct = int(args[1]) if len(args) > 1 else 1
        h_sem      = self._table.resolve(handle) or handle
        prev_count = ctypes.wintypes.LONG(0)
        ok = getattr(kernel32, 'ReleaseSemaphore')(
            ctypes.wintypes.HANDLE(h_sem),
            ctypes.wintypes.LONG(release_ct),
            ctypes.byref(prev_count))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"ReleaseSemaphore failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, prev_count.value)

    def _handle_sync_wait_multiple(self, pkt: ETPacket) -> ETPacket:
        """WaitForMultipleObjects(Ex) — wait on up to QUEUE_DEPTH handles."""
        args       = unpack_args(pkt.payload)
        handles    = list(args[0]) if args and isinstance(args[0], (list, tuple)) else []
        wait_all   = bool(args[1]) if len(args) > 1 else False
        timeout_ms = int(args[2]) if len(args) > 2 else CONN_TIMEOUT_MS

        if not handles:
            return self._make_error(pkt, ET_ERR_INVALID_ARGS, "SYNC_WAIT_MULTIPLE needs handle list")

        # Resolve bridge handles to native handles
        native = []
        for bh in handles[:QUEUE_DEPTH]:  # cap at QUEUE_DEPTH
            h = self._table.resolve(int(bh)) or int(bh)
            native.append(h)

        n         = len(native)
        arr_type  = ctypes.wintypes.HANDLE * n
        h_arr     = arr_type(*native)
        result    = getattr(kernel32, 'WaitForMultipleObjects')(
            ctypes.wintypes.DWORD(n),
            h_arr,
            ctypes.wintypes.BOOL(wait_all),
            ctypes.wintypes.DWORD(timeout_ms))
        # WAIT_OBJECT_0=0 to WAIT_OBJECT_0+n-1, WAIT_TIMEOUT=0x102, WAIT_FAILED=0xFFFFFFFF
        return self._make_ok(pkt, result)

    def _handle_sync_reset_event(self, pkt: ETPacket) -> ETPacket:
        """ResetEvent — put event into non-signaled state."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_evt  = self._table.resolve(handle) or handle
        ok     = getattr(kernel32, 'ResetEvent')(ctypes.wintypes.HANDLE(h_evt))
        return self._make_ok(pkt, int(bool(ok)))

    def _handle_sync_close(self, pkt: ETPacket) -> ETPacket:
        """Close a sync object handle."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_obj  = self._table.resolve(handle) or handle
        ok     = getattr(kernel32, 'CloseHandle')(ctypes.wintypes.HANDLE(h_obj))
        self._table.release(handle)
        return self._make_ok(pkt, int(bool(ok)))

    # ── NET_OPS additions ──────────────────────────────────────────────────

    def _handle_net_connect(self, pkt: ETPacket) -> ETPacket:
        """connect() — establish TCP client connection."""
        import socket as _sock
        args        = unpack_args(pkt.payload)
        sock_handle = int(args[0]) if args else 0
        addr_str    = str(args[1]) if len(args) > 1 else "127.0.0.1"
        port        = int(args[2]) if len(args) > 2 else 80
        raw_fd      = self._table.resolve(sock_handle) or sock_handle
        try:
            s = _sock.fromfd(raw_fd, _sock.AF_INET, _sock.SOCK_STREAM)
            s.connect((addr_str, port))
            s.detach()
            return self._make_ok(pkt, 0)
        except OSError as exc:
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    def _handle_net_listen(self, pkt: ETPacket) -> ETPacket:
        """listen() — enable incoming connections on a bound socket."""
        import socket as _sock
        args        = unpack_args(pkt.payload)
        sock_handle = int(args[0]) if args else 0
        backlog     = int(args[1]) if len(args) > 1 else S  # ET: backlog = N = 12
        raw_fd      = self._table.resolve(sock_handle) or sock_handle
        try:
            s = _sock.fromfd(raw_fd, _sock.AF_INET, _sock.SOCK_STREAM)
            s.listen(backlog)
            s.detach()
            return self._make_ok(pkt, 0)
        except OSError as exc:
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    def _handle_net_accept(self, pkt: ETPacket) -> ETPacket:
        """accept() — accept an incoming connection."""
        import socket as _sock
        args        = unpack_args(pkt.payload)
        sock_handle = int(args[0]) if args else 0
        raw_fd      = self._table.resolve(sock_handle) or sock_handle
        try:
            s       = _sock.fromfd(raw_fd, _sock.AF_INET, _sock.SOCK_STREAM)
            conn, addr = s.accept()
            s.detach()
            new_fd  = conn.fileno()
            handle  = self._table.allocate(new_fd, 0, 0, CmdFamily.NET_OPS,
                          tag=f"AcceptSock_{pkt.source_pid}_{addr[1]}")
            conn.detach()
            payload, c = pack_args(handle, new_fd, addr[0], addr[1])
            return self._make_response(pkt, pkt.cmd_code, ETPacket.FLAG_RESPONSE, c, payload)
        except OSError as exc:
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    def _handle_net_close(self, pkt: ETPacket) -> ETPacket:
        """closesocket() — close a socket and release the bridge handle."""
        import socket as _sock
        args        = unpack_args(pkt.payload)
        sock_handle = int(args[0]) if args else 0
        raw_fd      = self._table.resolve(sock_handle) or sock_handle
        try:
            s = _sock.fromfd(raw_fd, _sock.AF_INET, _sock.SOCK_STREAM)
            s.close()
        except OSError as _close_exc:
            # Socket may already be closed — record but proceed with handle release
            self._log.mediation(
                "NET_CLOSE64: socket close OSError (fd=%d PID=%d): %s",
                raw_fd, pkt.source_pid, _close_exc)
        self._table.release(sock_handle)
        return self._make_ok(pkt, 0)

    def _handle_net_select(self, pkt: ETPacket) -> ETPacket:
        """select() — check socket readiness (read/write/error sets)."""
        import socket as _sock, select as _sel
        args         = unpack_args(pkt.payload)
        read_handles = list(args[0]) if args and isinstance(args[0], (list,tuple)) else []
        write_handles= list(args[1]) if len(args)>1 and isinstance(args[1], (list,tuple)) else []
        timeout_s    = float(args[2]) if len(args) > 2 else 0.0

        # Validate timeout — negative means use system default (ET: D must be bounded)
        if timeout_s < 0:
            timeout_s = _sock.getdefaulttimeout() or 0.0

        def resolve_fds(hs):
            """D-expand bridge socket handles to native file descriptors."""
            fds = []
            for bh in hs:
                fd = self._table.resolve(int(bh)) or int(bh)
                fds.append(fd)
            return fds

        r_fds = resolve_fds(read_handles)
        w_fds = resolve_fds(write_handles)
        try:
            rr, ww, ee = _sel.select(r_fds, w_fds, [], timeout_s)
            payload, c = pack_args(rr, ww, ee)
            return self._make_response(pkt, pkt.cmd_code, ETPacket.FLAG_RESPONSE, c, payload)
        except (_sock.error, OSError) as exc:
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    def _handle_net_sockopt(self, pkt: ETPacket) -> ETPacket:
        """getsockopt / setsockopt."""
        import socket as _sock
        args        = unpack_args(pkt.payload)
        sock_handle = int(args[0]) if args else 0
        level       = int(args[1]) if len(args) > 1 else _sock.SOL_SOCKET
        optname     = int(args[2]) if len(args) > 2 else 0
        value       = args[3] if len(args) > 3 else None  # None = get, else set
        raw_fd      = self._table.resolve(sock_handle) or sock_handle
        try:
            s = _sock.fromfd(raw_fd, _sock.AF_INET, _sock.SOCK_STREAM)
            if value is None:
                result = s.getsockopt(level, optname)
                s.detach()
                return self._make_ok(pkt, result)
            else:
                s.setsockopt(level, optname, int(value))
                s.detach()
                return self._make_ok(pkt, 0)
        except OSError as exc:
            return self._make_error(pkt, ET_ERR_OS_ERROR, str(exc))

    # ── FILE_OPS additions ─────────────────────────────────────────────────

    def _handle_file_close(self, pkt: ETPacket) -> ETPacket:
        """CloseHandle for a bridged file handle."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        ok          = getattr(kernel32, 'CloseHandle')(ctypes.wintypes.HANDLE(h_file))
        self._table.release(file_handle)
        return self._make_ok(pkt, int(bool(ok)))

    def _handle_file_getsize(self, pkt: ETPacket) -> ETPacket:
        """GetFileSizeEx — 64-bit file size."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        size        = ctypes.c_int64(0)
        ok          = getattr(kernel32, 'GetFileSizeEx')(
            ctypes.wintypes.HANDLE(h_file), ctypes.byref(size))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GetFileSizeEx failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, size.value)

    def _handle_file_getattr(self, pkt: ETPacket) -> ETPacket:
        """GetFileAttributesExW — file attributes, timestamps, and sizes."""
        args = unpack_args(pkt.payload)
        path = str(args[0]) if args else ""

        # noinspection PyPep8Naming
        class WIN32_FILE_ATTRIBUTE_DATA(ctypes.Structure):
            """Win32 WIN32_FILE_ATTRIBUTE_DATA — D-descriptor for file attributes and timestamps."""
            _fields_ = [
                ("dwFileAttributes",  ctypes.wintypes.DWORD),
                ("ftCreationTime",    ctypes.c_uint64),
                ("ftLastAccessTime",  ctypes.c_uint64),
                ("ftLastWriteTime",   ctypes.c_uint64),
                ("nFileSizeHigh",     ctypes.wintypes.DWORD),
                ("nFileSizeLow",      ctypes.wintypes.DWORD),
            ]

        data = WIN32_FILE_ATTRIBUTE_DATA()
        # noinspection PyPep8Naming
        GetFileExInfoStandard = 0
        ok = getattr(kernel32, 'GetFileAttributesExW')(
            path, GetFileExInfoStandard, ctypes.byref(data))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GetFileAttributesExW('{path}') failed: {getattr(kernel32, 'GetLastError')()}")
        file_size = (data.nFileSizeHigh << 32) | data.nFileSizeLow
        return self._make_ok(pkt, data.dwFileAttributes, file_size,
                             data.ftCreationTime, data.ftLastAccessTime, data.ftLastWriteTime)

    def _handle_file_setattr(self, pkt: ETPacket) -> ETPacket:
        """SetFileAttributesW."""
        args  = unpack_args(pkt.payload)
        path  = str(args[0]) if args else ""
        attrs = int(args[1]) if len(args) > 1 else 0x20  # FILE_ATTRIBUTE_NORMAL
        ok    = getattr(kernel32, 'SetFileAttributesW')(path, ctypes.wintypes.DWORD(attrs))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"SetFileAttributesW('{path}') failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, 1)

    def _handle_file_seteof(self, pkt: ETPacket) -> ETPacket:
        """SetEndOfFile — truncate/extend file to current 64-bit pointer position."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        offset      = int(args[1]) if len(args) > 1 else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        # First seek to position
        pos_high = ctypes.wintypes.LONG(offset >> 32)
        getattr(kernel32, 'SetFilePointer')(ctypes.wintypes.HANDLE(h_file),
            ctypes.wintypes.LONG(offset & 0xFFFFFFFF),
            ctypes.byref(pos_high), 0)
        ok = getattr(kernel32, 'SetEndOfFile')(ctypes.wintypes.HANDLE(h_file))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"SetEndOfFile failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, 1)

    def _handle_file_flush(self, pkt: ETPacket) -> ETPacket:
        """FlushFileBuffers — force OS write-back."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        ok          = getattr(kernel32, 'FlushFileBuffers')(ctypes.wintypes.HANDLE(h_file))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"FlushFileBuffers failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, 1)

    def _handle_file_gettime(self, pkt: ETPacket) -> ETPacket:
        """GetFileTime — created/accessed/written FILETIME values."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        t_create = ctypes.c_uint64(0)
        t_access = ctypes.c_uint64(0)
        t_write  = ctypes.c_uint64(0)
        ok       = getattr(kernel32, 'GetFileTime')(ctypes.wintypes.HANDLE(h_file),
            ctypes.byref(t_create), ctypes.byref(t_access), ctypes.byref(t_write))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"GetFileTime failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, t_create.value, t_access.value, t_write.value)

    def _handle_file_settime(self, pkt: ETPacket) -> ETPacket:
        """SetFileTime — set created/accessed/written timestamps."""
        args        = unpack_args(pkt.payload)
        file_handle = int(args[0]) if args else 0
        t_create    = int(args[1]) if len(args) > 1 else 0
        t_access    = int(args[2]) if len(args) > 2 else 0
        t_write     = int(args[3]) if len(args) > 3 else 0
        h_file      = self._table.resolve(file_handle) or file_handle
        ok          = getattr(kernel32, 'SetFileTime')(ctypes.wintypes.HANDLE(h_file),
            ctypes.byref(ctypes.c_uint64(t_create)),
            ctypes.byref(ctypes.c_uint64(t_access)),
            ctypes.byref(ctypes.c_uint64(t_write)))
        if not ok:
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"SetFileTime failed: {getattr(kernel32, 'GetLastError')()}")
        return self._make_ok(pkt, 1)

    def _handle_file_find_first(self, pkt: ETPacket) -> ETPacket:
        """FindFirstFileW — begin directory enumeration."""
        args    = unpack_args(pkt.payload)
        pattern = str(args[0]) if args else "*"

        find_data = _FindWin32FindData()
        # noinspection PyPep8Naming
        INVALID_HANDLE_VALUE = ctypes.wintypes.HANDLE(-1).value
        h_find = getattr(kernel32, 'FindFirstFileW')(pattern, ctypes.byref(find_data))
        if h_find == INVALID_HANDLE_VALUE:
            err = getattr(kernel32, 'GetLastError')()
            if err == 2:  # ERROR_FILE_NOT_FOUND
                return self._make_ok(pkt, 0, "", 0, 0)  # empty result
            return self._make_error(pkt, ET_ERR_NOT_FOUND,
                                    f"FindFirstFileW('{pattern}') failed: {err}")
        handle = self._table.allocate(h_find, 0, 0, CmdFamily.FILE_OPS,
            tag=f"Find_{pkt.source_pid}")
        file_size = (find_data.nFileSizeHigh << 32) | find_data.nFileSizeLow
        return self._make_ok(pkt, handle, find_data.cFileName,
                             find_data.dwFileAttributes, file_size)

    def _handle_file_find_next(self, pkt: ETPacket) -> ETPacket:
        """FindNextFileW — continue directory enumeration."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_find = self._table.resolve(handle) or handle

        find_data = _FindWin32FindData()
        ok = getattr(kernel32, 'FindNextFileW')(ctypes.wintypes.HANDLE(h_find), ctypes.byref(find_data))
        if not ok:
            err = getattr(kernel32, 'GetLastError')()
            if err == 18:  # ERROR_NO_MORE_FILES
                return self._make_ok(pkt, 0, "", 0, 0)  # end of enumeration
            return self._make_error(pkt, ET_ERR_OS_ERROR,
                                    f"FindNextFileW failed: {err}")
        file_size = (find_data.nFileSizeHigh << 32) | find_data.nFileSizeLow
        return self._make_ok(pkt, 1, find_data.cFileName,
                             find_data.dwFileAttributes, file_size)

    def _handle_file_find_close(self, pkt: ETPacket) -> ETPacket:
        """FindClose — end directory enumeration and release handle."""
        args   = unpack_args(pkt.payload)
        handle = int(args[0]) if args else 0
        h_find = self._table.resolve(handle) or handle
        ok     = getattr(kernel32, 'FindClose')(ctypes.wintypes.HANDLE(h_find))
        self._table.release(handle)
        return self._make_ok(pkt, int(bool(ok)))
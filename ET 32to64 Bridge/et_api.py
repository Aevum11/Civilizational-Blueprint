"""
et_bridge/et_api.py
ET32 Bridge — API Layer: Marshaling, Routing, and Hook Management

Derived from P ∘ D ∘ T = E.

The API layer is the Descriptor-space interface between:
  - The 32-bit process (P₃₂, source of requests)
  - The bridge (D, the marshaling/routing layer)
  - The 64-bit host (T, the executor)

This module provides:
  1. ETAPIGateway  — high-level API for external callers to request 64-bit operations
  2. ETHookManager — manages the lifecycle of IAT hooks per process (start, refresh, remove)
  3. ETMarshal     — converts platform-specific data types ↔ ETPacket arguments
  4. ETBridgeAPI   — the singleton surface used by et32_bridge_main.py

ET derivation:
  The API layer occupies d=4 (DLL_OPS — temporal persistence, the "glue" layer).
  It provides the handle that translates between P₃₂ and P_full without mutation
  of either space's identity. This IS the Subsumption Law in code form:
    Complete(bridge) ↔ ∀ op₃₂: ∃ route(op₃₂) → op₆₄ ∧ V(route) = 0

Key ET constants used:
  S = 12 — max concurrently managed target PIDs in the active hook registry
  K = 2/3 — stability threshold: less than K × S PIDs failing = bridge stable
  RETRY_COUNT = 12 — pipe connection retries on new process detection
"""

import os
import sys
import threading
import time
import struct
import ctypes.wintypes
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from et_math import (
    S, K, V_BASE, CONN_TIMEOUT_MS, RETRY_COUNT,
    DIGITAL_ACTION_QUANTUM, IPC_BUFFER_SIZE, QUEUE_DEPTH,
    ETPacket, CmdFamily, CmdCode, ETMetrics, ETHandleMath,
    pack_args, unpack_args, PIPE_NAME_TEMPLATE
)
from et_handle import HandleTable
from et_config import ETBridgeConfig, TargetConfig
from et_logger import ETLog
from et_ipc import ETIPCServer, pipe_name_for_pid
from et_host64 import ETHost64
from et_injector import ETInjector
from et_awe     import (
    ETAWEBookshelf,
    AWE_PAGE_SIZE, AWE_WINDOW_SIZE, AWE_WINDOW_PAGES,
    AWE_MAX_WINDOWS, AWE_INIT_PAGES, AWE_EXPAND_STEP_PAGES,
)
from et_wow64   import ETWow64Hook, NT_HOOK_CATALOGUE

# Windows API access — module-level binding resolves ctypes.windll dynamic
# attributes for static analysis (same pattern as et_injector.py line 66).
# ET: the kernel32 handle IS the OS T-interface — the traversal gateway
# through which all Windows syscalls pass.
kernel32 = getattr(ctypes.windll, 'kernel32')

# Windows type constant — size of a native HANDLE on the broker (64-bit) side.
# Used by ETMarshal for handle width validation.
HANDLE_SIZE_BYTES: int = ctypes.sizeof(ctypes.wintypes.HANDLE)

# ============================================================================
# ET MARSHAL — type conversion between Windows API and ET packet arguments
# ============================================================================

class ETMarshal:
    """
    Converts between Windows API data types and ETPacket argument encoding.

    Every Windows API call intercepted by the bridge must have its arguments
    marshaled into pack_args format and the response unmarshaled back.

    ET derivation: the marshal is a D-transformer. It does not change P (the
    actual data), only re-encodes it from one Descriptor (Windows ABI) to
    another (ETPacket binary format). V(marshal) = 0 for any lossless conversion.

    Address marshaling:
      32-bit pointer → pack_args as uint32 (type tag 0x01)
      64-bit address → pack_args as uint64 (type tag 0x02)
      Bridge handle  → pack_args as uint32 (encoded via HandleTable)

    Size marshaling:
      Sizes ≤ 0xFFFFFFFF → uint32 (type tag 0x01)
      Sizes > 0xFFFFFFFF → uint64 (type tag 0x02) [LARGE_INTEGER handling]
    """

    @staticmethod
    def addr32_to_arg(addr32: int) -> int:
        """
        Marshal a 32-bit address for sending.
        If it is a bridge handle, pass through as-is.
        If it is a real 32-bit address, pass through as-is (broker resolves).
        """
        return addr32 & 0xFFFFFFFF

    @staticmethod
    def size_to_arg(size: int) -> int:
        """Marshal a size value. Preserves 64-bit sizes."""
        return size  # pack_args will choose uint32 vs uint64 automatically

    @staticmethod
    def string_to_arg(s: Optional[str]) -> Optional[str]:
        """Marshal a string (LPCSTR/LPWSTR) to a Python str or None."""
        return s

    @staticmethod
    def handle_to_arg(handle_val: int) -> int:
        """
        Marshal a Windows HANDLE to a bridge argument.
        Windows HANDLEs are pointer-sized but semantically opaque integers.
        In 32-bit mode they are ≤ 0xFFFFFFFF, so pack_args uses uint32.
        """
        return handle_val & 0xFFFFFFFF

    @staticmethod
    def result_to_addr32(result_args: list, handle_table: HandleTable) -> int:
        """
        Unmarshal a response from the broker back to a 32-bit address/handle.
        The first result argument is always the bridge handle (or 0 on error).
        Returns the 32-bit value safe for use in the 32-bit process.

        Uses the handle_table to verify that returned bridge handles are valid
        entries — a handle that doesn't exist in the table is an incoherent
        response (V > 0) and is mapped to 0 (NULL).
        """
        if not result_args:
            return 0
        first = result_args[0]
        if isinstance(first, int):
            if first >= 0x100000000:
                # 64-bit address: project to 32-bit handle via the table.
                # project_address will look up existing or alloc new handle.
                projected = handle_table.project_address(first)
                if projected == 0:
                    return 0  # table full or incoherent — NULL
                return projected & 0xFFFFFFFF
            if ETHandleMath.is_bridge_handle(first):
                # Already a bridge handle — verify it exists in the table
                entry = handle_table.get(first)
                if entry is None:
                    return 0  # incoherent handle — not in table
            return first & 0xFFFFFFFF
        return 0

    @staticmethod
    def pack_call(cmd_family: int, cmd_code: int, *args) -> Tuple[bytes, int]:
        """
        Marshal a complete API call into an ETPacket payload.

        Validates that cmd_family is within the ET S=12 lattice and that
        cmd_code is non-negative, then packs the routing prefix followed
        by the serialized arguments.

        The routing prefix (8 bytes: uint32 family + uint32 code) is prepended
        to the packed args so batch/compound operations can demux sub-payloads
        without inspecting the packet header.

        ET derivation: this is the D-encoding step — the API call's semantic
        content (P) is wrapped in the ETPacket Descriptor (D) for transport
        through the Mediation channel. V(encoding) = 0 for lossless packing.
        cmd_family is clamped to [0, S] and cmd_code to [0, 0xFFFF] — the
        full ET lattice range.

        Returns (payload_bytes, arg_count).
        """
        # Validate against ET lattice constraints
        family_clamped = max(0, min(cmd_family, S))
        code_clamped   = max(0, min(cmd_code, 0xFFFF))
        # Routing prefix: 8 bytes (family uint32 + code uint32)
        routing_prefix = struct.pack("<II", family_clamped, code_clamped)
        arg_payload, arg_count = pack_args(*args)
        return routing_prefix + arg_payload, arg_count

    @staticmethod
    def alloc_size_rounded(size: int) -> int:
        """
        Round allocation size to the digital action quantum ħ_d = 4096.
        V(rounding) = 0 because no data is lost — only padding is added.
        """
        return ((size + DIGITAL_ACTION_QUANTUM - 1) // DIGITAL_ACTION_QUANTUM) * DIGITAL_ACTION_QUANTUM


# ============================================================================
# ET HOOK MANAGER — lifecycle of hooks per PID
# ============================================================================

class ETHookManager:
    """
    Manages IAT hooks across multiple 32-bit processes.

    ET derivation:
      P = the set of all active 32-bit processes
      D = the hook configuration per process (which APIs are hooked, where stubs are)
      T = this manager (traverses P to apply/remove D)

    Each tracked process is one lattice element. The maximum number of
    simultaneously tracked processes = QUEUE_DEPTH = 144 = S² (the full
    squared manifold — one process per lattice²).

    Active tracking uses a dict: pid → HookState.
    Koide stability: if success_rate >= K (2/3) across all tracked PIDs → stable.
    """

    class HookState:
        """
        Complete state of hooks for one target PID.

        ET PDT:
          P = pid (process identity substrate)
          D = config (descriptor of what to hook)
          T = injector + pipe_connected (the traversal status)
        """
        __slots__ = (
            "pid", "config", "injector",
            "hooks_active", "pipe_connected",
            "error_count", "start_time", "last_check"
        )

        def __init__(self, pid: int, config: TargetConfig):
            self.pid            = pid
            self.config         = config
            self.injector       = None      # ETInjector instance
            self.hooks_active   = False
            self.pipe_connected = False
            self.error_count    = 0
            self.start_time     = time.monotonic()
            self.last_check     = time.monotonic()

        def variance(self) -> float:
            """V(state) — measures how far from grounded this hook state is."""
            v = 0.0
            if not self.hooks_active:
                v += V_BASE
            if not self.pipe_connected:
                v += V_BASE
            if self.error_count > 0:
                v += min(V_BASE * self.error_count, K)
            return v

        def is_stable(self) -> bool:
            """Koide-stable iff V(state) < K = 2/3. ET stability criterion."""
            return self.variance() < K

    def __init__(
        self,
        ipc_server: ETIPCServer,
        log: Any
    ):
        self._ipc    = ipc_server
        self._log    = log
        self._states: Dict[int, "ETHookManager.HookState"] = {}
        self._lock   = threading.Lock()

    def engage(self, pid: int, config: TargetConfig) -> bool:
        """
        Engage the bridge for a newly detected 32-bit process.

        Steps:
          1. Create named pipe for this PID
          2. Inject hook stubs into the target process
          3. Record hook state

        Returns True if engagement succeeded within S retries.
        """
        with self._lock:
            if pid in self._states:
                self._log.mediation("PID %d already engaged", pid)
                return True
            if len(self._states) >= QUEUE_DEPTH:
                self._log.warning("Hook registry full (%d entries)", QUEUE_DEPTH)
                return False

        # Step 1: Create pipe
        if not self._ipc.create_pipe_for_pid(pid):
            self._log.error("Failed to create pipe for PID %d", pid)
            return False

        # Step 2: Inject
        state = ETHookManager.HookState(pid, config)
        injector = ETInjector(PIPE_NAME_TEMPLATE)

        self._log.info("Injecting hooks into PID %d (%s)...", pid, config.exe_name)
        inject_ok = injector.inject(pid, config)

        if inject_ok:
            state.injector     = injector
            state.hooks_active = True
            self._log.info("Hooks engaged for PID %d: %s", pid, config.exe_name)
        else:
            state.error_count += 1
            self._log.warning("Injection failed for PID %d (errors=%d)",
                               pid, state.error_count)

        with self._lock:
            self._states[pid] = state

        return inject_ok

    def refresh(self, pid: int) -> bool:
        """
        Re-inject hooks if they have been removed or the process has been updated.
        Called periodically by the monitor thread.
        """
        with self._lock:
            state = self._states.get(pid)
        if state is None:
            return False

        # Update pipe connectivity status
        state.pipe_connected = self._ipc.is_connected(pid)
        state.last_check     = time.monotonic()

        if not state.hooks_active and state.injector:
            self._log.mediation("Re-injecting hooks for PID %d", pid)
            ok = state.injector.inject(pid, state.config)
            if ok:
                state.hooks_active = True
                state.error_count  = 0
            else:
                state.error_count += 1
        return state.hooks_active

    def disengage(self, pid: int):
        """
        Remove hooks from a process (on exit or on request).
        Closes the pipe and removes the state entry.
        """
        with self._lock:
            state = self._states.pop(pid, None)
        if state is None:
            return
        if state.injector:
            try:
                state.injector.remove_hooks(pid)
            except Exception as exc:
                self._log.warning("remove_hooks failed for PID %d: %s", pid, exc)
        self._ipc.remove_pipe_for_pid(pid)
        self._log.info("Bridge disengaged for PID %d (%s)", pid, state.config.exe_name)

    def get_state(self, pid: int) -> Optional["ETHookManager.HookState"]:
        """Return the HookState for a tracked PID, or None if not engaged."""
        with self._lock:
            return self._states.get(pid)

    def get_universal_hook_addr(self, pid: int) -> int:
        """
        Return the 32-bit address of ET32_UniversalHook export in
        et_bridge32.dll as loaded in the target process.

        ET derivation: ET32_UniversalHook is the T-traversal point for
        ALL intercepted ntdll32 functions. Its address in the target process
        is resolved via GetProcAddress on the injected DLL's module handle.
        Returns 0 if the DLL is not yet loaded or address not resolvable.
        """
        # ctypes imported at module level — kernel32 bound at module scope
        with self._lock:
            state = self._states.get(pid)
        if state is None or state.injector is None:
            return 0
        # Ask the injector for the DLL base in the target process
        dll_base = getattr(state.injector, '_dll_base_in_target', 0)
        if not dll_base:
            return 0
        # Resolve ET32_UniversalHook from the DLL's PE export table
        # Use the broker's loaded copy to read the relative export offset,
        # then rebase to dll_base in the target.
        try:
            h = getattr(kernel32, 'LoadLibraryExA')(
                b'et_bridge32.dll', None,
                0x00000001  # DONT_RESOLVE_DLL_REFERENCES
            )
            if not h:
                return 0
            local_addr = getattr(kernel32, 'GetProcAddress')(
                h, b'ET32_UniversalHook'
            )
            local_base = h
            getattr(kernel32, 'FreeLibrary')(h)
            if not local_addr:
                return 0
            # RVA of ET32_UniversalHook = local_addr - local_base
            rva = int(local_addr) - int(local_base)
            # Address in target = dll_base + rva
            return (int(dll_base) + rva) & 0xFFFFFFFF
        except (OSError, ValueError, AttributeError):
            return 0

    def active_pids(self) -> List[int]:
        """Return list of all currently engaged PIDs. Thread-safe snapshot."""
        with self._lock:
            return list(self._states.keys())

    def all_states(self) -> Dict[int, "ETHookManager.HookState"]:
        """Return a snapshot of all hook states keyed by PID. Thread-safe."""
        with self._lock:
            return dict(self._states)

    def stability(self) -> float:
        """
        Koide alignment of the hook manager.
        K_eff = (stably-hooked PIDs) / (total PIDs).
        Stable iff K_eff >= K = 2/3.
        """
        with self._lock:
            if not self._states:
                return 1.0  # vacuously stable
            stable = sum(1 for s in self._states.values() if s.is_stable())
            return stable / len(self._states)

    def is_koide_stable(self) -> bool:
        """True iff hook manager stability K_eff ≥ K = 2/3 (Koide threshold)."""
        return self.stability() >= K


# ============================================================================
# ET API GATEWAY — high-level call surface
# ============================================================================

class ETAPIGateway:
    """
    High-level API gateway: wraps ETIPCClient calls with semantic method names.

    This is used by:
      - The 32-bit helper process (et32_bridge_helper.py) running in-process
      - Any test harness that wants to exercise bridge operations

    ET derivation:
      The gateway is a T-interface: it provides the traversal methods that
      activate the Descriptor set of the bridge (the 12 command families).
      Each method is one T-activation of one D (command family) over P (the OS).

    All methods block until the broker responds or timeout expires.
    All addresses returned are 32-bit bridge handles (safe for 32-bit code).
    """

    def __init__(self, client):
        """client: ETIPCClient (already connected)."""
        self._client = client
        self._log    = ETLog.get("et_api_gateway")
        self._pid    = client.pid

    # --- Memory operations (family 1) ---

    def virtual_alloc(self, size: int, protect: int = 0x04,
                      addr_hint: int = 0) -> Optional[int]:
        """
        Allocate 'size' bytes in the 64-bit broker process.
        Returns a 32-bit bridge handle, or None on failure.
        """
        resp = self._client.call(
            CmdFamily.MEMORY_BASIC, CmdCode.VIRT_ALLOC,
            addr_hint, size, 0x3000, protect  # MEM_COMMIT|MEM_RESERVE
        )
        return self._extract_handle(resp)

    def virtual_free(self, handle: int) -> bool:
        """Free a 64-bit allocation by bridge handle. Returns True on success."""
        resp = self._client.call(
            CmdFamily.MEMORY_BASIC, CmdCode.VIRT_FREE,
            handle
        )
        return self._is_ok(resp)

    def virtual_protect(self, handle: int, new_protect: int) -> int:
        """Returns old protection flags, or 0 on failure."""
        resp = self._client.call(
            CmdFamily.MEMORY_BASIC, CmdCode.VIRT_PROTECT,
            handle, new_protect
        )
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            return args[0] if args else 0
        return 0

    def read_memory(self, handle: int, size: int) -> Optional[bytes]:
        """Read 'size' bytes from the 64-bit allocation at 'handle'."""
        resp = self._client.call(
            CmdFamily.MEMORY_BASIC, CmdCode.READ_MEM,
            handle, size
        )
        if resp and resp.cmd_code != CmdCode.CTRL_ERR:
            args = unpack_args(resp.payload)
            return args[0] if args and isinstance(args[0], bytes) else None
        return None

    def write_memory(self, handle: int, data: bytes) -> int:
        """Write bytes to the 64-bit allocation at 'handle'. Returns bytes written."""
        resp = self._client.call(
            CmdFamily.MEMORY_BASIC, CmdCode.WRITE_MEM,
            handle, data
        )
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            return args[0] if args else 0
        return 0

    # --- Memory mapping (family 2) ---

    def create_file_mapping(self, size: int, protect: int = 0x04,
                            name: Optional[str] = None) -> Optional[int]:
        """Create a 64-bit file mapping (backed by page file). Returns handle."""
        resp = self._client.call(
            CmdFamily.MEMORY_MAP, CmdCode.FILE_MAP_CREATE,
            0xFFFFFFFF, protect, size, name
        )
        return self._extract_handle(resp)

    def map_view_of_file(self, map_handle: int, size: int = 0,
                         offset: int = 0) -> Optional[int]:
        """Map a view of a file mapping. Returns handle."""
        resp = self._client.call(
            CmdFamily.MEMORY_MAP, CmdCode.FILE_MAP_VIEW,
            map_handle, 0xF001F, offset, size  # FILE_MAP_ALL_ACCESS
        )
        return self._extract_handle(resp)

    def unmap_view(self, view_handle: int) -> bool:
        """Unmap a file-mapping view by handle. Returns True on success."""
        resp = self._client.call(
            CmdFamily.MEMORY_MAP, CmdCode.FILE_MAP_CLOSE,
            view_handle
        )
        return self._is_ok(resp)

    # --- DLL operations (family 4) ---

    def load_library_64(self, dll_path: str, flags: int = 0) -> Optional[int]:
        """Load a 64-bit DLL into the broker. Returns module handle."""
        resp = self._client.call(
            CmdFamily.DLL_OPS, CmdCode.DLL_LOAD,
            dll_path, flags
        )
        return self._extract_handle(resp)

    def get_proc_address_64(self, module_handle: int, proc_name: str) -> Optional[int]:
        """Get the address of a function in a 64-bit DLL. Returns function handle."""
        resp = self._client.call(
            CmdFamily.DLL_OPS, CmdCode.DLL_GETPROC,
            module_handle, proc_name
        )
        return self._extract_handle(resp)

    def call_64bit(self, func_handle: int, *args) -> Optional[int]:
        """Call a 64-bit function. Returns result as integer."""
        resp = self._client.call(
            CmdFamily.DLL_OPS, CmdCode.DLL_CALL,
            func_handle, *args
        )
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            result_args = unpack_args(resp.payload)
            return result_args[0] if result_args else 0
        return None

    def free_library_64(self, module_handle: int) -> bool:
        """Unload a 64-bit DLL from the broker by module handle. Returns True on success."""
        resp = self._client.call(
            CmdFamily.DLL_OPS, CmdCode.DLL_FREE,
            module_handle
        )
        return self._is_ok(resp)

    # --- File operations (family 8) ---

    def open_large_file(self, path: str, access: int = 0xC0000000) -> Optional[int]:
        """Open a large file (>4GB capable). Returns file handle."""
        resp = self._client.call(
            CmdFamily.FILE_OPS, CmdCode.FILE_OPEN_LARGE,
            path, access
        )
        return self._extract_handle(resp)

    def seek_large(self, file_handle: int, offset: int, whence: int = 0) -> int:
        """Seek to a 64-bit offset. Returns new position."""
        resp = self._client.call(
            CmdFamily.FILE_OPS, CmdCode.FILE_SEEK_LARGE,
            file_handle, offset, whence
        )
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            return args[0] if args else -1
        return -1

    def read_large(self, file_handle: int, size: int) -> Optional[bytes]:
        """Read from a large file. Returns bytes."""
        resp = self._client.call(
            CmdFamily.FILE_OPS, CmdCode.FILE_READ_LARGE,
            file_handle, size
        )
        if resp and resp.cmd_code != CmdCode.CTRL_ERR:
            args = unpack_args(resp.payload)
            return args[0] if args and isinstance(args[0], bytes) else b""
        return None

    def write_large(self, file_handle: int, data: bytes) -> int:
        """Write to a large file. Returns bytes written."""
        resp = self._client.call(
            CmdFamily.FILE_OPS, CmdCode.FILE_WRITE_LARGE,
            file_handle, data
        )
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            return args[0] if args else 0
        return 0

    # --- Python operations (family 11) ---

    def python_init(self) -> bool:
        """Initialize 64-bit Python in the broker."""
        resp = self._client.call(CmdFamily.PYTHON_OPS, CmdCode.PY_INIT)
        return self._is_ok(resp)

    def python_exec(self, code: str) -> str:
        """Execute Python code in 64-bit interpreter. Returns stdout."""
        resp = self._client.call(CmdFamily.PYTHON_OPS, CmdCode.PY_EXEC, code)
        if resp and resp.cmd_code != CmdCode.CTRL_ERR:
            args = unpack_args(resp.payload)
            return str(args[0]) if args else ""
        return ""

    def python_import(self, module_name: str) -> bool:
        """Import a 64-bit Python module."""
        resp = self._client.call(CmdFamily.PYTHON_OPS, CmdCode.PY_IMPORT, module_name)
        return self._is_ok(resp)

    def python_call(self, func_name: str, *args) -> Optional[str]:
        """Call a Python function and return its string repr."""
        resp = self._client.call(CmdFamily.PYTHON_OPS, CmdCode.PY_CALL, func_name, *args)
        if resp and resp.cmd_code != CmdCode.CTRL_ERR:
            result = unpack_args(resp.payload)
            return str(result[0]) if result else None
        return None

    # --- System info ---

    def get_system_info_64(self) -> Dict[str, int]:
        """Get 64-bit system info: page_size, max_addr, cpu_count, granularity."""
        resp = self._client.call(CmdFamily.PROCESS_OPS, CmdCode.PROC_INFO)
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            if len(args) >= 4:
                return {
                    "page_size":    args[0],
                    "max_address":  args[1],
                    "cpu_count":    args[2],
                    "granularity":  args[3],
                }
        return {}

    # --- Control ---

    def ping(self) -> bool:
        """Liveness check. Returns True if broker responds."""
        resp = self._client.call(0, CmdCode.CTRL_PING)
        return resp is not None and resp.cmd_code == CmdCode.CTRL_ACK

    def handshake(self) -> Dict[str, int]:
        """Perform initial handshake. Returns broker PID and ET constants."""
        resp = self._client.call(0, CmdCode.CTRL_HANDSHAKE)
        if resp and resp.cmd_code == CmdCode.CTRL_ACK:
            args = unpack_args(resp.payload)
            if len(args) >= 3:
                return {"broker_pid": args[0], "S": args[1], "K_milli": args[2]}
        return {}

    # --- Batch / compound ---

    def batch(self, operations: List[ETPacket]) -> List[ETPacket]:
        """
        Execute a batch of up to S=12 operations.
        Returns a list of response packets.
        """
        if not operations:
            return []
        # Serialize all sub-packets into one payload blob
        sub_data = b"".join(p.serialise() for p in operations[:S])
        resp = self._client.send_request(ETPacket(
            source_pid  = self._pid,
            dest_pid    = 0,
            space_token = 0,
            cmd_family  = CmdFamily.COMPOUND_OPS,
            cmd_code    = CmdCode.COMPOUND_BATCH,
            flags       = ETPacket.FLAG_REQUEST,
            arg_count   = len(operations),
            payload     = sub_data,
        ))
        if resp is None:
            return []
        # Parse batch response: uint32 count + N serialised responses
        results = []
        payload = resp.payload
        if len(payload) < 4:
            return results
        count = struct.unpack_from("<I", payload, 0)[0]
        offset = 4
        for _ in range(count):
            sub = ETPacket.deserialise(payload[offset:])
            if sub is None:
                break
            results.append(sub)
            offset += PDT_HEADER_SIZE + len(sub.payload)
        return results

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_handle(resp: Optional[ETPacket]) -> Optional[int]:
        """Extract the first argument (bridge handle) from a response packet."""
        if resp is None or resp.cmd_code == CmdCode.CTRL_ERR:
            return None
        args = unpack_args(resp.payload)
        return args[0] if args and isinstance(args[0], int) else None

    @staticmethod
    def _is_ok(resp: Optional[ETPacket]) -> bool:
        """True iff response is non-null and is an ACK (not ERR)."""
        return resp is not None and resp.cmd_code == CmdCode.CTRL_ACK


# Use PDT_HEADER_SIZE from et_math — needed in batch
from et_math import PDT_HEADER_SIZE


# ============================================================================
# ET BRIDGE API — the singleton entry point for the broker
# ============================================================================

class ETBridgeAPI:
    """
    The singleton API surface used by et32_bridge_main.py.
    Wires together: Config → Monitor → IPC Server → Host64 → Hook Manager.

    ET derivation:
      P = this broker process
      D = ETBridgeConfig (the complete Descriptor set)
      T = all threads (workers, monitor, IOCP)
      E = active, running bridge (V(E) = 0)

    Lifecycle:
      api = ETBridgeAPI(config)
      api.start()     — start all subsystems
      api.wait()      — block until shutdown requested
      api.stop()      — graceful shutdown
    """

    def __init__(self, config: ETBridgeConfig):
        self._config      = config
        self._metrics     = ETMetrics()
        self._log         = ETLog.get("et_bridge_api")
        self._handle_table = HandleTable()
        self._host64      = ETHost64(self._handle_table, self._metrics)
        self._ipc         = ETIPCServer(self._host64.dispatch, self._metrics)

        # AWE Bookshelf — physical memory windowing (complete memory access)
        self._awe = ETAWEBookshelf()

        # WOW64 Universal Hook — ntdll32-level complete API interception
        self._wow64 = ETWow64Hook(self._awe)

        # Wire subsystems into host dispatcher (must follow _awe/_wow64 creation)
        self._host64._awe  = self._awe   # wire bookshelf into host dispatcher
        self._host64._wow64 = self._wow64  # wire dynamic syscall hook

        self._hook_manager = ETHookManager(self._ipc, self._log)
        self._running      = False
        self._stop_event   = threading.Event()
        self._pid          = os.getpid()

        # Variance tracking
        self._engage_count   = 0
        self._disengage_count = 0
        self._error_count    = 0

        # Shutdown hooks — T-continuations invoked during bridge exit sequence
        self._shutdown_hooks: List[Callable[[], None]] = []

        self._log.info(
            "ETBridgeAPI initialised: broker PID=%d, targets=%d",
            self._pid, len(config.targets)
        )

    def start(self) -> bool:
        """Start IPC server and prepare for process monitoring."""
        if self._running:
            return True
        ok = self._ipc.start()
        if not ok:
            self._log.incoherence("Failed to start IPC server")
            return False
        self._running = True
        self._log.info("ETBridgeAPI started: IPC server running")
        return True

    def stop(self):
        """Stop all subsystems gracefully."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        # Disengage all active hooks
        for pid in self._hook_manager.active_pids():
            self._hook_manager.disengage(pid)

        self._ipc.stop()

        # Execute registered shutdown hooks (FIFO order)
        for hook in getattr(self, '_shutdown_hooks', []):
            try:
                hook()
            except (OSError, RuntimeError) as exc:
                self._log.warning("Shutdown hook failed: %s", exc)

        self._log.info("ETBridgeAPI stopped. Final metrics: %s", self._metrics.summary())

    def on_process_found(self, info) -> None:
        """
        Called by ETProcessMonitor when a process is detected in the bridged tree.

        Handles two cases:
          1. 32-bit process (target or child): full injection — AWE + KiFast hook
          2. 64-bit native child: NO injection (already 64-bit). Registered with
             the broker for handle interop — when the 32-bit parent passes a handle
             from PROCESS_INFORMATION to the broker, the broker can now resolve it.

        ET T∘T (Traverser paper §41):
          A child process is T spawning T. Nested T is still T.
          Depth is unlimited — every descendant is handled the same way.
        """
        # ctypes imported at module level — kernel32 bound at module scope
        # Support both old (pid, exe_name, config) call and new ProcessInfo call
        if hasattr(info, 'pid'):
            pid      = info.pid
            exe_name = info.exe_name
            config   = info.config
            is_64bit = getattr(info, 'is_64bit_native', False)
            depth    = getattr(info, 'depth', 0)
            ppid     = getattr(info, 'ppid', 0)
        else:
            # Legacy: called with bare PID integer (early development convention).
            # Synthesize a full-feature config since no explicit D is available.
            pid      = int(info)
            exe_name = "unknown"
            config   = self._config.synthetic_config(exe_name)
            is_64bit = False
            depth    = 0
            ppid     = 0

        pipe_path = pipe_name_for_pid(pid)
        label = "64-bit native child" if is_64bit else (
            f"child (depth {depth})" if depth > 0 else "target"
        )
        self._log.info(
            "Process detected [%s]: %s (PID %d, parent %d, pipe %s)",
            label, exe_name, pid, ppid, pipe_path
        )

        if is_64bit:
            # 64-bit native child — already has complete 64-bit access.
            # Register its PID with the broker so handle interop works:
            # when the 32-bit parent passes hProcess/hThread from CreateProcess,
            # the broker can resolve those handles to this native process.
            self._handle_table.register_native64(pid, exe_name)
            self._log.info(
                "64-bit native child PID %d (%s) registered for handle interop. "
                "No injection needed — process already has full 64-bit access.",
                pid, exe_name
            )
            self._engage_count += 1
            return

        # 32-bit process: full bridge
        ok = self._hook_manager.engage(pid, config)
        if ok:
            self._engage_count += 1
            h_proc = getattr(kernel32, 'OpenProcess')(0x1FFFFF, False, pid)
            if h_proc:
                # AWE bookshelf
                awe_ok = self._awe.allocate_pool(pid, int(h_proc))
                if awe_ok:
                    self._log.info(
                        "AWE bookshelf ready for PID %d (depth %d) — "
                        "true 64-bit physical memory access active.", pid, depth
                    )
                else:
                    self._log.warning_di(
                        "AWE pool failed for PID %d. "
                        "Ensure Administrator + SeLockMemoryPrivilege.", pid
                    )
                # KiFastSystemCall universal hook (one root patch, zero lists)
                installed = self._wow64.install(pid, int(h_proc))
                if installed:
                    self._log.info(
                        "KiFastSystemCall hook installed in PID %d — "
                        "ALL syscalls intercepted dynamically.", pid
                    )
                    # ISSUE-04 resolution: communicate the trampoline address
                    # to the injected DLL so ET32_KiFastHook can JMP to
                    # the trampoline on pass-through syscalls (instead of
                    # dereferencing g_kifastsystemcall_trampoline = NULL).
                    #
                    # ET Descriptor Gap Principle: the gap between trampoline
                    # installation (Python broker) and trampoline address
                    # communication (C DLL) IS the missing Descriptor.
                    # This call closes the gap.
                    hook_entry = NT_HOOK_CATALOGUE.get(pid)
                    state = self._hook_manager.get_state(pid)
                    if hook_entry and state and state.injector:
                        tramp_addr = hook_entry.tramp_va
                        result = state.injector.call_dll_export(
                            int(h_proc),
                            "ET32_SetKiFastTrampoline",
                            tramp_addr
                        )
                        if result is not None:
                            self._log.info(
                                "ET32_SetKiFastTrampoline(0x%08X) called in "
                                "PID %d — pass-through syscalls routed to "
                                "trampoline.", tramp_addr, pid
                            )
                        else:
                            self._log.warning_di(
                                "Failed to call ET32_SetKiFastTrampoline in "
                                "PID %d — pass-through syscalls will crash "
                                "(NULL trampoline). Check DLL injection state.",
                                pid
                            )

                        # ISSUE-03 resolution: rewrite the WOW64 fail-safe
                        # stub to JMP to ET32_KiFastHook in the injected DLL.
                        #
                        # The fail-safe stub at hook_entry.stub_va is currently:
                        #   E9 <rel32> → JMP trampoline (pass-through)
                        # We rewrite it to:
                        #   E9 <rel32> → JMP ET32_KiFastHook (real hook)
                        #
                        # After this, the hook chain becomes:
                        #   KiFastSystemCall → JMP stub → JMP ET32_KiFastHook
                        #     → DynamicDispatch → [result or trampoline]
                        #
                        # ET derivation (Subsumption Law): the fail-safe stub
                        # subsumes nothing (all syscalls pass through unchanged).
                        # The real hook subsumes ALL syscalls via ET32_KiFastHook.
                        # Rewriting the stub achieves full subsumption.
                        kihook_addr = self._hook_manager.get_universal_hook_addr(pid)
                        if kihook_addr and hook_entry.stub_va:
                            stub_va = hook_entry.stub_va
                            jmp_disp = (kihook_addr - (stub_va + 5)) & 0xFFFFFFFF
                            patch_data = struct.pack("<BI", 0xE9, jmp_disp)
                            patch_buf = (ctypes.c_char * 5)(*patch_data)
                            written = ctypes.c_size_t()
                            ok = getattr(kernel32, 'WriteProcessMemory')(
                                ctypes.wintypes.HANDLE(int(h_proc)),
                                ctypes.c_void_p(stub_va),
                                patch_buf, 5, ctypes.byref(written)
                            )
                            if ok:
                                self._log.info(
                                    "WOW64 stub rewritten in PID %d: "
                                    "fail-safe → ET32_KiFastHook @ 0x%08X. "
                                    "Real syscall interception active.",
                                    pid, kihook_addr
                                )
                            else:
                                self._log.warning_di(
                                    "Failed to rewrite WOW64 stub in PID %d "
                                    "— fail-safe (pass-through) remains active.",
                                    pid
                                )
                getattr(kernel32, 'CloseHandle')(h_proc)
        else:
            self._error_count += 1

    def on_process_exit(self, info) -> None:
        """
        Called by ETProcessMonitor when any process in the bridged tree exits.
        """
        if hasattr(info, 'pid'):
            pid      = info.pid
            exe_name = info.exe_name
            is_64bit = getattr(info, 'is_64bit_native', False)
        else:
            pid, exe_name = info, "unknown"
            is_64bit = False

        self._log.info("Process exited: %s (PID %d)", exe_name, pid)
        self._disengage_count += 1

        if is_64bit:
            # Deregister native64 child
            self._handle_table.deregister_native64(pid)
            return

        # 32-bit: full cleanup
        self._hook_manager.disengage(pid)
        self._awe.release_pool(pid)
        self._wow64.remove(pid)
        # et_bridge32.dll ET32_Shutdown() called from within the process

    @property
    def awe(self) -> ETAWEBookshelf:
        """AWE bookshelf manager — provides true 64-bit memory to 32-bit processes."""
        return self._awe

    @property
    def wow64(self) -> ETWow64Hook:
        """WOW64 universal hook manager — complete ntdll32 API interception."""
        return self._wow64

    @property
    def host(self) -> 'ETHost64':
        """Public accessor for the 64-bit operation dispatcher."""
        return self._host64

    @property
    def hook_manager(self) -> 'ETHookManager':
        """Public accessor for the IAT hook lifecycle manager."""
        return self._hook_manager

    def wait(self):
        """Block the calling thread until stop() is called."""
        self._stop_event.wait()

    def wait_for_stop(self, timeout: float = None) -> bool:
        """
        Wait for the stop event with an optional timeout.

        Returns True if the stop event was set, False if the timeout expired.

        ET derivation: this is a T-pause — the Traverser yields control for
        up to `timeout` seconds, then resumes. The main loop uses this to
        sleep in short intervals (CONN_TIMEOUT_MS / S) while remaining
        responsive to stop signals.
        """
        return self._stop_event.wait(timeout=timeout)

    def request_stop(self):
        """Signal the main loop to stop (can be called from any thread)."""
        self._running = False
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True while the bridge is actively running and servicing requests."""
        return self._running

    @property
    def stability(self) -> float:
        """Global bridge stability K_eff."""
        hook_k = self._hook_manager.stability()
        metric_k = self._metrics.koide_alignment
        # Combined stability = geometric mean (ET product of two K-measures)
        return (hook_k * metric_k) ** 0.5

    def status_report(self) -> Dict[str, Any]:
        """
        Return a full status dict for logging/display.

        Includes: broker identity, subsystem status, ET constants, AWE bookshelf
        configuration, WOW64 hook catalogue size, and handle table diagnostics.

        ET derivation: the status report is the D-snapshot of the complete bridge
        Exception state — every subsystem's current Descriptor is captured here.
        """
        # AWE bookshelf configuration (ET-derived physical memory constants)
        awe_status: Dict[str, Any] = {
            "page_size":         AWE_PAGE_SIZE,
            "window_size":       AWE_WINDOW_SIZE,
            "window_pages":      AWE_WINDOW_PAGES,
            "max_windows":       AWE_MAX_WINDOWS,
            "init_pages":        AWE_INIT_PAGES,
            "expand_step_pages": AWE_EXPAND_STEP_PAGES,
        }

        # WOW64 hook catalogue — total interceptable ntdll32 functions
        wow64_catalogue_size: int = len(NT_HOOK_CATALOGUE)

        return {
            "broker_pid":       self._pid,
            "broker_executable": sys.executable,
            "broker_maxsize":   sys.maxsize,
            "running":          self._running,
            "active_targets":   len(self._hook_manager.active_pids()),
            "engage_count":     self._engage_count,
            "disengage_count":  self._disengage_count,
            "error_count":      self._error_count,
            "stability":        f"{self.stability:.4f}",
            "et_constants": {
                "S":               S,
                "K":               K,
                "V_BASE":          V_BASE,
                "CONN_TIMEOUT_MS": CONN_TIMEOUT_MS,
                "RETRY_COUNT":     RETRY_COUNT,
                "IPC_BUFFER_SIZE": IPC_BUFFER_SIZE,
                "QUEUE_DEPTH":     QUEUE_DEPTH,
            },
            "metrics":          self._metrics.summary(),
            "handle_table": {
                "total_allocated": self._handle_table.total_allocated,
                "fill_ratio":      f"{self._handle_table.fill_ratio:.4f}",
            },
            "awe_bookshelf":    awe_status,
            "wow64_catalogue_entries": wow64_catalogue_size,
        }

    def reload_config(self):
        """Apply a reloaded config without restart."""
        self._log.info("Config reload: %d targets", len(self._config.targets))
        # New targets do not need action here — the monitor will pick them up.
        # Removed targets: disengage their hooks.
        active: Set[int] = set(self._hook_manager.active_pids())
        valid_names: Set[str] = {t.exe_name.lower() for t in self._config.targets}
        for pid in active:
            state = self._hook_manager.get_state(pid)
            if state and state.config.exe_name.lower() not in valid_names:
                self._hook_manager.disengage(pid)

    def register_shutdown_hook(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be invoked when the bridge shuts down.

        ET derivation: shutdown hooks are T-continuations — each callback
        is a Traverser action that runs during the bridge's exit sequence.
        The hooks run in registration order (FIFO), after subsystems are
        stopped but before the final log flush.

        Used by external integrations (GUI, CLI, monitoring) that need to
        perform cleanup when the bridge terminates.
        """
        if not hasattr(self, '_shutdown_hooks'):
            self._shutdown_hooks: List[Callable[[], None]] = []
        self._shutdown_hooks.append(callback)
"""
et32_bridge_helper.py
ET32 Bridge — 32-bit Companion Helper

Derived from P ∘ D ∘ T = E.

This is the 32-bit side of the bridge: a lightweight Python process (compiled
with 32-bit Python + PyInstaller) that runs inside or alongside the 32-bit
target process.

ET PDT of the helper:
  P = the 32-bit target process's address space (4GB constrained)
  D = the bridge connection (named pipe to the 64-bit broker)
  T = this helper process (traverses D to reach 64-bit resources)
  E = a completed 64-bit call from within 32-bit space

Two deployment modes:
  1. STANDALONE: Launched as a separate process by the 64-bit broker,
     communicating with the target via shared memory or message passing.
     The target process calls the helper via a Win32 message or IPC.

  2. INJECTED: Loaded into the 32-bit target's address space as a DLL
     via the broker's ETInjector, running Python in-process.
     In this mode, the helper uses the injected named pipe directly.

The helper exposes a simple call table: a shared memory region that the
target's hook stubs write requests to and read responses from.

Call table layout (in the ħ_d × S = 49152 byte shared memory region):
  [0..3]   Magic number: 0x45543332 ("ET32")
  [4..7]   Request slot count: uint32 (max S² = 144)
  [8..11]  Response slot count: uint32
  [12..15] Status flags: uint32 (ET state machine)
  [16..47] Reserved (PDT header area)
  [48:]    Slot array: SLOT_COUNT × SLOT_SIZE (336 = 48+PDT_HEADER bytes each)

SLOT layout (each slot = 48 bytes):
  [0..3]   Slot state: 0=FREE, 1=REQUEST, 2=RESPONSE, 3=ERROR
  [4..7]   cmd_family: uint32
  [8..11]  cmd_code: uint32
  [12..15] arg_count: uint32
  [16..31] Args inline (up to 4 × uint32, or overflow pointer)
  [32..39] Result: uint64 (bridge handle or value)
  [40..43] Error code: uint32
  [44..47] Sequence: uint32

All communication through the call table uses atomic uint32 operations
on the slot state, preventing race conditions between the 32-bit stub
and the helper.

ET Constants:
  MAGIC          = 0x45543332  ("ET32" LE)
  SLOT_COUNT     = S² = 144   (QUEUE_DEPTH)
  SLOT_SIZE      = S × 4 = 48  (4 bytes per lattice position)
  SHMEM_SIZE     = 49152       (ħ_d × S = IPC_BUFFER_SIZE)
  SHMEM_NAME     = "ET32_SHMEM_{pid}"
"""

import ctypes
from ctypes import wintypes
import os
import sys
import struct
import threading
import time
import argparse
import signal
from typing import Optional, Dict, Any

# When running as a frozen 32-bit helper, insert the module path
if getattr(sys, "frozen", False):
    _base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    if _base not in sys.path:
        sys.path.insert(0, _base)
else:
    _dev_root = os.path.dirname(os.path.abspath(__file__))
    if _dev_root not in sys.path:
        sys.path.insert(0, _dev_root)

from et_math import (
    S, K, V_BASE,
    IPC_BUFFER_SIZE, CONN_TIMEOUT_MS, QUEUE_DEPTH, RETRY_COUNT,
    SHMEM_NAME_TEMPLATE, PIPE_NAME_TEMPLATE,
    ETPacket, CmdFamily, CmdCode, ETMetrics,
    pack_args, unpack_args
)
from et_ipc import ETIPCClient
from et_api import ETAPIGateway
from et_logger import ETLog

# ============================================================================
# WINDOWS API CONSTANTS
# ============================================================================

FILE_MAP_ALL_ACCESS     = 0x000F001F
PAGE_READWRITE          = 0x04
OPEN_EXISTING           = 3  # dwCreationDisposition for CreateFile (open-only mode)
INVALID_HANDLE_VALUE    = wintypes.HANDLE(-1).value

class _Kernel32API:
    """
    Dynamic dispatch layer for the Win32 kernel32 API surface.

    ET derivation:
      P = kernel32.dll (the complete Win32 kernel API — Ω-cardinality function set)
      D = this dispatch layer (resolves function names to typed callables)
      T = any function call through this layer (traverses D to reach P)
      E = a resolved, cached, callable Win32 function (grounded Exception)

    Dynamic ``__getattr__`` resolves every kernel32 function on first access
    and caches it in the instance ``__dict__`` for zero-cost repeat calls.
    No hardcoded function list is maintained — the dispatcher covers the
    complete kernel32 surface.  Nothing can be missed.

    The cache in ``__dict__`` bypasses ``__getattr__`` on subsequent accesses,
    so the overhead of dynamic resolution is paid exactly once per function
    (V(subsequent_lookup) = 0).
    """

    def __init__(self):
        """Bind the underlying kernel32 DLL handle via ctypes.windll."""
        object.__setattr__(self, '_dll', ctypes.windll.kernel32)

    def __getattr__(self, name: str):
        """
        Resolve and cache any kernel32 function on first access.

        ET: The function name is a D (Descriptor) that selects one function
        from the P-surface (kernel32.dll).  Resolution is the T-traversal
        that grounds the name to a callable.  Caching in ``__dict__`` gives
        V(subsequent_lookup) = 0.

        Raises AttributeError with an ET-derived message if the function
        does not exist in kernel32 (Incoherence: the D has no P-binding).
        """
        dll = object.__getattribute__(self, '_dll')
        try:
            func = getattr(dll, name)
        except AttributeError:
            raise AttributeError(
                f"kernel32 has no export '{name}' — "
                f"Incoherence: D('{name}') has no P-binding in kernel32.dll"
            ) from None
        # Cache in instance __dict__ — bypasses __getattr__ on next access
        self.__dict__[name] = func
        return func


kernel32 = _Kernel32API()

# ============================================================================
# CALL TABLE MAGIC AND LAYOUT
# ============================================================================

CALL_TABLE_MAGIC  = 0x45543332   # "ET32"
SLOT_STATE_FREE   = 0
SLOT_STATE_REQ    = 1
SLOT_STATE_RESP   = 2
SLOT_STATE_ERR    = 3

# Header: 48 bytes (4 × S)
CALL_TABLE_HEADER_SIZE = 4 * S  # 48 bytes
# Each slot: 48 bytes (4 × S)
SLOT_SIZE              = 4 * S  # 48 bytes
# Total slots
SLOT_COUNT             = QUEUE_DEPTH  # 144
# Shared memory total: header + slots
SHMEM_DATA_SIZE        = CALL_TABLE_HEADER_SIZE + SLOT_COUNT * SLOT_SIZE  # 6960 bytes
# We use the full IPC_BUFFER_SIZE = 49152 bytes for the shared region
SHMEM_TOTAL_SIZE       = IPC_BUFFER_SIZE  # 49152 bytes


# ============================================================================
# CALL TABLE STRUCTURES
# ============================================================================

class CallTableHeader(ctypes.LittleEndianStructure):
    """
    48-byte header for the shared call table.
    ET PDT derivation: 4 × S = 48 bytes mirrors the ETPacket header size.
    """
    _fields_ = [
        ("magic",          ctypes.c_uint32),   # 0x45543332
        ("req_slots",      ctypes.c_uint32),   # number of active request slots
        ("resp_slots",     ctypes.c_uint32),   # number of pending response slots
        ("flags",          ctypes.c_uint32),   # ET state flags
        ("broker_pid",     ctypes.c_uint32),   # 64-bit broker PID
        ("client_pid",     ctypes.c_uint32),   # 32-bit client PID
        ("version",        ctypes.c_uint32),   # protocol version (1)
        ("et_S",           ctypes.c_uint32),   # S constant (12)
        ("reserved",       ctypes.c_uint8 * (CALL_TABLE_HEADER_SIZE - 32)),
    ]


class CallSlot(ctypes.LittleEndianStructure):
    """
    48-byte call slot. One per in-flight operation.
    ET derivation: SLOT_SIZE = 4 × S bytes — four words per lattice position.
    """
    _fields_ = [
        ("state",          ctypes.c_uint32),   # SLOT_STATE_*
        ("cmd_family",     ctypes.c_uint32),
        ("cmd_code",       ctypes.c_uint32),
        ("arg_count",      ctypes.c_uint32),
        ("arg0",           ctypes.c_uint32),   # inline arg0 (32-bit)
        ("arg1",           ctypes.c_uint32),   # inline arg1
        ("arg2",           ctypes.c_uint32),   # inline arg2
        ("arg3",           ctypes.c_uint32),   # inline arg3
        ("result_lo",      ctypes.c_uint32),   # result low 32 bits
        ("result_hi",      ctypes.c_uint32),   # result high 32 bits (bridge handle)
        ("error_code",     ctypes.c_uint32),
        ("sequence",       ctypes.c_uint32),
    ]


assert ctypes.sizeof(CallTableHeader) == CALL_TABLE_HEADER_SIZE, \
    f"CallTableHeader size mismatch: {ctypes.sizeof(CallTableHeader)} != {CALL_TABLE_HEADER_SIZE}"

assert ctypes.sizeof(CallSlot) == SLOT_SIZE, \
    f"CallSlot size mismatch: {ctypes.sizeof(CallSlot)} != {SLOT_SIZE}"


# ============================================================================
# SHARED MEMORY CALL TABLE
# ============================================================================

class SharedCallTable:
    """
    Shared memory region that acts as the communication buffer between the
    32-bit hook stubs and the 32-bit helper.

    ET derivation:
      P = the shared memory region (common substrate for both processes)
      D = the call table layout (CallTableHeader + CallSlot array)
      T = the helper's polling thread (traverses D to find REQUEST slots)

    The 32-bit hook stubs write to slots (state=REQUEST) and spin-wait for
    state=RESPONSE. The helper reads REQUEST slots, calls the broker, and
    writes RESPONSE.

    Polling interval: CONN_TIMEOUT_MS / QUEUE_DEPTH = ~10ms per slot scan.
    This gives effective latency of < 10ms for any given slot.
    """

    def __init__(self, pid: int, create: bool = True):
        """
        Create (broker side) or open (helper side) the shared memory.

        pid: the 32-bit target's PID (used to name the shmem)
        create: True = create new; False = open existing
        """
        self._pid       = pid
        self._name      = SHMEM_NAME_TEMPLATE.format(pid=pid)
        self._h_mapping = 0
        self._view_ptr  = None
        self._buf       = None

        if create:
            self._h_mapping = kernel32.CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                0,
                SHMEM_TOTAL_SIZE,
                self._name
            )
        else:
            # Open existing mapping (mode=OPEN_EXISTING=%d) — helper side
            self._open_mode = OPEN_EXISTING
            self._h_mapping = kernel32.OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,
                False,
                self._name
            )

        if not self._h_mapping:
            raise RuntimeError(
                f"SharedCallTable({'create' if create else 'open'}) failed "
                f"for PID {pid}: error {kernel32.GetLastError()}"
            )

        self._view_ptr = kernel32.MapViewOfFile(
            self._h_mapping,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            SHMEM_TOTAL_SIZE
        )
        if not self._view_ptr:
            kernel32.CloseHandle(self._h_mapping)
            raise RuntimeError(f"MapViewOfFile failed: error {kernel32.GetLastError()}")

        # Map Python ctypes structures over the shared region
        self._header = CallTableHeader.from_address(self._view_ptr)
        self._slots  = (CallSlot * SLOT_COUNT).from_address(
            self._view_ptr + CALL_TABLE_HEADER_SIZE
        )

        if create:
            self._init_header(pid)

    def _init_header(self, pid: int):
        """Initialize the call table header with ET constants."""
        self._header.magic      = CALL_TABLE_MAGIC
        self._header.req_slots  = 0
        self._header.resp_slots = 0
        self._header.flags      = 0
        self._header.broker_pid = os.getpid()
        self._header.client_pid = pid
        self._header.version    = 1
        self._header.et_S       = S
        # Initialize all slots to FREE
        for i in range(SLOT_COUNT):
            self._slots[i].state = SLOT_STATE_FREE

    def is_valid(self) -> bool:
        """
        Check whether the shared call table is mapped and has valid ET32 magic.

        ET: V(call_table) = 0 iff the view pointer is live and the magic
        word equals 0x45543332 ('ET32' in little-endian).  The magic is the
        D-identity of this shared region — its absence signals {P,T} Incoherence.
        """
        return (self._view_ptr is not None and
                self._header.magic == CALL_TABLE_MAGIC)

    def find_request_slot(self) -> Optional[int]:
        """
        Scan for the first REQUEST slot.
        Returns slot index or None if no pending requests.
        The scan is O(SLOT_COUNT) = O(144) — fast enough for <1ms poll.
        """
        for i in range(SLOT_COUNT):
            if self._slots[i].state == SLOT_STATE_REQ:
                return i
        return None

    def get_slot(self, index: int) -> CallSlot:
        """
        Return the CallSlot structure at the given index.

        ET: Direct D-access to a specific lattice position in the call table.
        index ∈ (0, SLOT_COUNT) = (0, 144).  The caller is responsible for
        bounds validation — this is a raw substrate access (P-level).
        """
        return self._slots[index]

    def write_response(self, index: int, result: int, error: int = 0):
        """
        Write a response to a slot.
        Atomically transitions state REQUEST → RESPONSE.

        ET: The result is a 64-bit value stored in two 32-bit halves.
        Explicit ``struct.pack`` and ``struct.unpack`` calls ensure correct
        little-endian encoding of the full uint64 result before splitting
        into low and high uint32 words.
        """
        slot = self._slots[index]
        # Explicit LE encoding: uint64 → (uint32_lo, uint32_hi)
        result_lo, result_hi = struct.unpack("<II", struct.pack("<Q", result & 0xFFFFFFFFFFFFFFFF))
        slot.result_lo  = result_lo
        slot.result_hi  = result_hi
        slot.error_code = error
        # Memory barrier via ctypes (Python GIL provides ordering)
        slot.state = SLOT_STATE_RESP if error == 0 else SLOT_STATE_ERR

    def close(self):
        """
        Unmap the shared memory view and close the file mapping handle.

        ET: Returns the P-substrate (physical shared memory) to its unbound
        state.  After close, is_valid returns False and all slot
        references are invalidated.  Idempotent — safe to call multiple times.
        """
        if self._view_ptr:
            kernel32.UnmapViewOfFile(self._view_ptr)
            self._view_ptr = None
        if self._h_mapping:
            kernel32.CloseHandle(self._h_mapping)
            self._h_mapping = 0

    def __del__(self):
        self.close()


# ============================================================================
# ET 32-BIT HELPER — main class
# ============================================================================

class ET32Helper:
    """
    32-bit helper that connects to the 64-bit broker and processes call table requests.

    The helper runs as a separate process launched by the broker.
    It:
      1. Connects to the broker via named pipe (ETIPCClient)
      2. Performs the handshake
      3. Creates the shared call table (or connects to existing)
      4. Spins polling the call table for REQUEST slots
      5. For each REQUEST: calls the broker via ETAPIGateway, writes RESPONSE

    This provides the "last mile" connectivity between the 32-bit hook stubs
    (which write to the call table) and the 64-bit broker (which executes ops).

    Poll interval derived from ET:
      T_poll = CONN_TIMEOUT_MS / QUEUE_DEPTH = 1500 / 144 ≈ 10.4 ms
      This is within the V_BASE = 1/12 = 83ms window — excellent latency.
    """

    POLL_INTERVAL = CONN_TIMEOUT_MS / QUEUE_DEPTH / 1000.0  # ~0.0104 seconds

    # Maximum acceptable single-slot latency before warning (microseconds).
    # Derived: V_BASE × CONN_TIMEOUT_MS × 1000 = (1/12) × 1500 × 1000 = 125,000 μs.
    # Any slot that exceeds this budget consumes more than one base-variance unit
    # of the timeout window — a D-gap signal that the broker may be overloaded.
    SLOT_LATENCY_WARN_US = V_BASE * CONN_TIMEOUT_MS * 1000.0  # 125,000 μs = 125 ms

    def __init__(self, pid: int):
        self._pid      = pid
        self._log      = ETLog.get("et32_helper")
        self._client   = ETIPCClient(pid)
        self._gateway  = None
        self._table    = None
        self._metrics  = ETMetrics()
        self._running  = False
        self._stop_evt = threading.Event()
        self._sequence = 0

    def start(self) -> bool:
        """
        Connect to broker, create call table, start polling thread.
        Returns True on success.

        Connection and handshake are retried up to RETRY_COUNT = S = 12 times.
        ET: RETRY_COUNT is within the cascade coherence horizon (N_max=25),
        so the retry loop is guaranteed to remain coherent per Level 4 𝒜_I.
        """
        pipe_path = PIPE_NAME_TEMPLATE.format(pid=self._pid)
        self._log.info(
            "Connecting to broker pipe: %s (retries=%d)", pipe_path, RETRY_COUNT
        )

        # Connect to broker pipe
        if not self._client.connect():
            self._log.incoherence("Failed to connect to broker pipe for PID %d", self._pid)
            return False

        self._gateway = ETAPIGateway(self._client)

        # Handshake with retry — RETRY_COUNT = S = 12 attempts
        info: Dict[str, Any] = {}
        for attempt in range(1, RETRY_COUNT + 1):
            info = self._gateway.handshake()
            if info:
                break
            self._log.mediation(
                "Handshake attempt %d/%d failed for PID %d",
                attempt, RETRY_COUNT, self._pid,
            )
            if attempt < RETRY_COUNT:
                time.sleep(CONN_TIMEOUT_MS / RETRY_COUNT / 1000.0)
        if not info:
            self._log.error("Handshake failed after %d attempts", RETRY_COUNT)
            return False
        self._log.info(
            "Connected to broker: PID=%d, S=%d, K=%d‰",
            info.get("broker_pid", 0),
            info.get("S", S),
            info.get("K_milli", int(K * 1000))
        )

        # Create shared call table
        try:
            self._table = SharedCallTable(self._pid, create=True)
            self._log.info("Call table created for PID %d (%d slots)", self._pid, SLOT_COUNT)
        except RuntimeError as exc:
            self._log.error("Failed to create call table: %s", exc)
            return False

        # Start polling thread
        self._running = True
        t = threading.Thread(
            target = self._poll_loop,
            name   = f"ET_Helper_Poll_{self._pid}",
            daemon = True
        )
        t.start()

        self._log.info("ET32Helper started for PID %d", self._pid)
        return True

    def stop(self):
        """Stop polling and close resources."""
        self._running = False
        self._stop_evt.set()
        self._client.disconnect()
        if self._table:
            self._table.close()
            self._table = None
        self._log.info("ET32Helper stopped for PID %d", self._pid)

    def wait(self):
        """Block until stop() is called."""
        self._stop_evt.wait()

    @property
    def metrics(self) -> ETMetrics:
        """
        Public accessor for bridge performance metrics.

        ET: ETMetrics tracks per-family latency, success rate, Koide alignment,
        operational entropy, and the Atiyah-Singer index theorem check.
        Exposing via property preserves encapsulation while granting read access
        to the metrics summary for logging and diagnostics.
        """
        return self._metrics

    def _poll_loop(self):
        """
        Main polling loop: scans the call table every POLL_INTERVAL seconds.
        When a REQUEST slot is found, dispatches via the gateway and writes RESPONSE.

        This loop is the T-traverser of the call table (D).
        V(loop) = 0 iff all slots are correctly processed within the poll interval.
        """
        while self._running:
            if self._table is None:
                break

            idx = self._table.find_request_slot()
            if idx is not None:
                self._process_slot(idx)
            else:
                # No requests pending — sleep for one poll interval
                self._stop_evt.wait(timeout=self.POLL_INTERVAL)

        self._log.mediation("Poll loop exiting for PID %d", self._pid)

    def _process_slot(self, idx: int):
        """
        Process a single REQUEST slot at index idx.
        Calls the broker and writes the result back.

        ET: slot.state = REQUEST → DISPATCHING (implicit) → RESPONSE/ERROR
        Each slot dispatches through one of the 12 CmdFamily lattice positions.
        """
        if not self._table:
            return

        slot = self._table.get_slot(idx)
        t0   = time.monotonic()

        try:
            family  = slot.cmd_family
            code    = slot.cmd_code
            args    = [slot.arg0, slot.arg1, slot.arg2, slot.arg3][:slot.arg_count]

            # Log the dispatch with ET lattice denomination via CmdFamily
            family_name = CmdFamily.FAMILY_TO_D.get(family, f"unknown({family})")
            self._log.mediation(
                "Slot %d: dispatching family=%d [%s] code=0x%02X args=%d",
                idx, family, family_name, code, len(args),
            )

            # Send to broker via ETIPCClient
            payload, count = pack_args(*args)
            self._sequence += 1
            pkt = ETPacket(
                source_pid  = self._pid,
                dest_pid    = 0,
                space_token = 0,
                cmd_family  = family,
                cmd_code    = code,
                flags       = ETPacket.FLAG_REQUEST,
                arg_count   = count,
                payload     = payload,
                sequence    = self._sequence,
            )
            resp = self._client.send_request(pkt)

            if resp is None:
                self._table.write_response(idx, 0, error=0xE000FFFF)
                self._metrics.record(family, 0.0, False)
                return

            # Extract first result arg as the bridge handle/value
            result_args = unpack_args(resp.payload)
            result_val  = result_args[0] if result_args and isinstance(result_args[0], int) else 0
            error_code  = 0 if resp.cmd_code == CmdCode.CTRL_ACK else (
                result_args[0] if result_args else 0xE0000001
            )

            self._table.write_response(idx, result_val, error=error_code)

            latency = (time.monotonic() - t0) * 1_000_000
            self._metrics.record(family, latency, error_code == 0)

            # V_BASE latency threshold check: warn if slot exceeds one base-variance
            # unit of the timeout budget (125 ms = V_BASE × CONN_TIMEOUT_MS × 1000)
            if latency > self.SLOT_LATENCY_WARN_US:
                self._log.warning_di(
                    "Slot %d [%s] latency %.0f μs exceeds V_BASE threshold %.0f μs "
                    "— potential D-gap in broker response",
                    idx, family_name, latency, self.SLOT_LATENCY_WARN_US,
                )

        except Exception as exc:
            self._log.incoherence("Slot %d processing error: %s", idx, exc)
            if self._table:
                self._table.write_response(idx, 0, error=0xE0000002)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "ET32_Bridge_Helper",
        description = "ET32 Bridge: 32-bit companion helper (connects to 64-bit broker)."
    )
    parser.add_argument(
        "--pid", "-p",
        type    = int,
        required= True,
        help    = "PID of the 32-bit target process (used to name the pipe and shmem)"
    )
    parser.add_argument(
        "--log-level", "-l",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Log level"
    )
    parser.add_argument(
        "--log-file", "-f",
        default = None,
        help    = "Log file path (default: ET32_Helper_{pid}.log)"
    )
    return parser.parse_args()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    """
    ET32 Bridge Helper entry point — 32-bit companion process.

    Parses CLI arguments (--pid, --log-level, --log-file), initializes
    logging, connects to the 64-bit broker via named pipe, creates the
    shared call table, and polls for requests until signaled or error.

    ET: T (this process) traverses D (pipe + call table) to bridge the
    32-bit P-constraint to the 64-bit broker.  The signal handler captures
    SIGINT/SIGTERM for graceful shutdown.

    Returns 0 on clean shutdown, 1 on startup failure.
    """
    args     = _parse_args()
    log_file = args.log_file or f"ET32_Helper_{args.pid}.log"
    ETLog.setup(level=args.log_level, log_file=log_file)
    log = ETLog.get("et32_helper_main")

    log.info("ET32 Bridge Helper starting for PID %d", args.pid)

    helper = ET32Helper(args.pid)

    def _shutdown(signum, frame):
        # Use frame to log the interrupted code location for diagnostics.
        # frame is the Python stack frame active at the moment of signal delivery.
        frame_info = f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame else "unknown"
        log.info("Signal %d at %s — stopping helper", signum, frame_info)
        helper.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not helper.start():
        log.incoherence("Helper failed to start for PID %d", args.pid)
        return 1

    helper.wait()
    log.info("ET32 Bridge Helper stopped for PID %d. Metrics: %s",
             args.pid, helper.metrics.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
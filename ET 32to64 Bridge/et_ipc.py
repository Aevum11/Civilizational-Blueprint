"""
et_bridge/et_ipc.py
ET32 Bridge — IPC Server (Named Pipe, Overlapped I/O)

Derived from P ∘ D ∘ T = E.

The IPC layer is the Mediation ({D,T}) state of the bridge:
  P = the shared named pipe (substrate of communication)
  D = the protocol (ETPacket structure, IPC_BUFFER_SIZE, PIPE_NAME_TEMPLATE)
  T = the server/client threads (traversers through the communication channel)
  E = a delivered ETPacket that has been acted upon and a response returned

Named pipe model (ET-derived):
  PIPE instances = S² = 144 (QUEUE_DEPTH) — one pipe slot per queue position.
  Worker threads  = S   = 12  — one worker per lattice position (manifold symmetry).
  Buffer size     = ħ_d × S = 4096 × 12 = 49152 bytes (IPC_BUFFER_SIZE).
  Timeout         = 1/K × 1000 = 1500 ms (CONN_TIMEOUT_MS).
  Flush at        = K × queue = 2/3 full (Koide fill ratio).

Each 32-bit target process connects via its own named pipe:
  \\\\.\\pipe\\ET32_PDT_{pid}

The server creates a new pipe instance for each PID it manages.
Multiple requests from the same process share one pipe (sequential per-connection).

Overlapped I/O is used so the server never blocks a worker thread on a single client.
The I/O completion port (IOCP) model is used:
  - CreateIoCompletionPort
  - GetQueuedCompletionStatus
  - ReadFile / WriteFile with OVERLAPPED
  - One IOCP handles all pipes and all workers

ET lattice position of IPC: d=9 (SYNC_OPS family — synchronization and event-driven
coordination). The IOCP IS the ET Event mechanism.
"""

import ctypes
import ctypes.wintypes
import struct
import threading
import queue
import time
import os
import sys
from typing import Dict, Optional, Callable, Any, List, Tuple
from dataclasses import dataclass, field

from et_math import (
    S, K, V_BASE,
    IPC_BUFFER_SIZE, PDT_HEADER_SIZE, CONN_TIMEOUT_MS,
    QUEUE_DEPTH, PIPE_NAME_TEMPLATE, RETRY_COUNT,
    ETPacket, ETMetrics, CmdCode, CmdFamily,
    pack_args, unpack_args,
    pairwise_incoherence, n_max_cascade, tightness,
    coherence_depth, COHERENCE_N_MAX,
)
from et_logger import ETLog
from et_errors import (
    ETOperationError, ETWindowsAPIError, ETInjectionError,
    ETIPCError, ETPacketError, ETAWEError, ETHookError,
    ETDispatchError, ETConfigError, ETHandleError,
    ETErrorSeverity, win32_check, win32_check_handle,
    ntstatus_check, et_context, safe_call,
    record_error, record_op, get_registry,
)


# ============================================================================
# WINDOWS API CONSTANTS AND STRUCTURES
# ============================================================================

INVALID_HANDLE_VALUE    = ctypes.wintypes.HANDLE(-1).value
PIPE_ACCESS_DUPLEX      = 0x00000003
PIPE_TYPE_BYTE          = 0x00000000
PIPE_READMODE_BYTE      = 0x00000000
PIPE_WAIT               = 0x00000000
PIPE_UNLIMITED_INSTANCES = 255
FILE_FLAG_OVERLAPPED    = 0x40000000
FILE_FLAG_WRITE_THROUGH = 0x80000000

GENERIC_READ            = 0x80000000
GENERIC_WRITE           = 0x40000000
OPEN_EXISTING           = 3
FILE_ATTRIBUTE_NORMAL   = 0x80

ERROR_IO_PENDING        = 997
ERROR_BROKEN_PIPE       = 109
ERROR_PIPE_CONNECTED    = 535
ERROR_NO_DATA           = 232
ERROR_PIPE_BUSY         = 231
ERROR_MORE_DATA         = 234
WAIT_OBJECT_0           = 0x00000000
WAIT_TIMEOUT            = 0x00000102
WAIT_FAILED             = 0xFFFFFFFF
INFINITE                = 0xFFFFFFFF

IOCP_KEY_SHUTDOWN       = 0xDEADBEEF  # sentinel completion key for shutdown


class OVERLAPPED(ctypes.Structure):
    """
    Win32 OVERLAPPED structure for asynchronous I/O.

    ET derivation: the OVERLAPPED is the D-state of an in-flight I/O request.
    hEvent is the T-signal: it fires when the I/O completes (T reaches E).
    Internal/InternalHigh carry the P-result (bytes transferred, status).
    """
    class _DUMMYUNION(ctypes.Union):
        class _DUMMYSTRUCT(ctypes.Structure):
            _fields_ = [
                ("Offset",     ctypes.wintypes.DWORD),
                ("OffsetHigh", ctypes.wintypes.DWORD),
            ]
        _fields_ = [
            ("DUMMYSTRUCT", _DUMMYSTRUCT),
            ("Pointer",     ctypes.c_void_p),
        ]
    _fields_ = [
        ("Internal",     ctypes.POINTER(ctypes.c_ulong)),
        ("InternalHigh", ctypes.POINTER(ctypes.c_ulong)),
        ("DUMMYUNION",   _DUMMYUNION),
        ("hEvent",       ctypes.wintypes.HANDLE),
    ]


kernel32 = getattr(ctypes.windll, 'kernel32')

# ── Platform validation (uses sys, os) ──────────────────────────────────────
# ET derivation: the bridge operates in a specific P-domain (Windows x64).
# Verifying platform is the D-constraint that ensures P∘D coherence before
# any T (thread) begins traversing the pipe infrastructure.
if sys.platform != "win32":
    raise ETConfigError(
        f"ET32 Bridge IPC requires Windows (sys.platform={sys.platform!r})",
        severity=ETErrorSeverity.FATAL,
    )
if sys.maxsize <= 2 ** 32:
    raise ETConfigError(
        "ET32 Bridge IPC server must run as a 64-bit Python process "
        f"(sys.maxsize={sys.maxsize})",
        severity=ETErrorSeverity.FATAL,
    )

# Broker PID — used for logging and self-identity validation
_BROKER_PID: int = os.getpid()


def _check_handle(h) -> bool:
    """True if handle is valid (not INVALID_HANDLE_VALUE and not None/0)."""
    if h is None:
        return False
    val = int(h)
    return val != 0 and val != INVALID_HANDLE_VALUE


# ============================================================================
# CONNECTION STATE — one per accepted pipe client
# ============================================================================

@dataclass
class PipeConnection:
    """
    Represents one accepted named-pipe connection from a 32-bit client.

    ET PDT decomposition:
      P = h_pipe (the actual pipe handle — the substrate connection)
      D = pid, pipe_name, buffers (the descriptor of this connection)
      T = state machine: CONNECTING → READING → DISPATCHING → WRITING → READING

    V(conn) = 0 iff the connection is in a stable reading or writing state.
    V(conn) = V_BASE iff state is CONNECTING (unsubstantiated — T not yet arrived).
    """
    pid:       int
    pipe_name: str
    h_pipe:    int                    # Windows HANDLE
    recv_buf:  bytearray = field(default_factory=lambda: bytearray(IPC_BUFFER_SIZE))
    send_buf:  bytearray = field(default_factory=lambda: bytearray(IPC_BUFFER_SIZE))
    bytes_read: int = 0
    state:     str = "CONNECTING"     # CONNECTING | READING | DISPATCHING | WRITING | CLOSED
    ol_read:   Any = None             # OVERLAPPED for read
    ol_write:  Any = None             # OVERLAPPED for write
    last_activity: float = field(default_factory=time.monotonic)
    sequence:  int = 0
    errors:    int = 0

    def is_alive(self) -> bool:
        """True if the connection is not CLOSED and the pipe handle is valid (E-state reachable)."""
        return self.state not in ("CLOSED",) and _check_handle(self.h_pipe)

    def variance(self) -> float:
        """
        ET variance V(conn) — measures distance from grounded E-state.

        V = 0.0      if stable (READING/WRITING/DISPATCHING, no errors)
        V = V_BASE   if CONNECTING (Unsubstantiated — T not yet arrived)
        V = V_BASE×n if n errors accumulated (progressing toward incoherence)
        V = 1.0      if CLOSED (full incoherence — P∘D∘T collapsed)
        """
        if self.state == "CONNECTING":
            return V_BASE
        if self.state == "CLOSED":
            return 1.0  # full incoherence
        if self.errors > 0:
            return min(1.0, V_BASE * self.errors)
        return 0.0

    def idle_seconds(self) -> float:
        """Seconds since last activity — T-traversal idle time on this connection."""
        return time.monotonic() - self.last_activity

    def conn_tightness(self) -> float:
        """
        Tightness of this connection, derived from its variance.

        Maps V(conn) to the ε-domain: V ∈ [0, 1] → ε ∈ [0, 50¢],
        then applies tightness(ε) = 100/(100 + |ε|).
        Result ∈ [2/3, 1]: 1.0 = perfect, K = at ∂I boundary.
        """
        eps_cents = self.variance() * 50.0
        return tightness(eps_cents)

    def conn_coherence_depth(self) -> float:
        """
        Coherence depth Δ∂I of this connection — distance from the ∂I boundary.

        Δ∂I = tightness(ε) − K.
        = 1/3 at perfect lattice point (V = 0).
        = 0   at ∂I (V maps to |ε| = 50¢).
        < 0   beyond ∂I (incoherent connection).
        """
        eps_cents = self.variance() * 50.0
        return coherence_depth(eps_cents)


# ============================================================================
# ET IPC SERVER — 64-bit broker side
# ============================================================================

class ETIPCServer:
    """
    64-bit named-pipe IPC server. Manages all connections from 32-bit clients.

    Architecture (ET-derived):
      - One IOCP handles all pipe instances (single P-substrate for all I/O)
      - S=12 worker threads drain the IOCP (T-traversers, one per lattice position)
      - QUEUE_DEPTH=144 in-flight requests in the work queue (S² internal queue)
      - Per-PID pipe name: \\\\.\\pipe\\ET32_PDT_{pid}
      - Buffer = IPC_BUFFER_SIZE = 49152 bytes per connection

    The dispatcher callback receives an ETPacket and returns an ETPacket.
    This callback is provided by et_host64.ETHost64.dispatch().

    Lifecycle:
      ETIPCServer.start()     — creates IOCP, starts S worker threads
      ETIPCServer.create_pipe_for_pid(pid) — opens a new pipe instance for a PID
      ETIPCServer.stop()      — graceful shutdown
    """

    def __init__(
        self,
        dispatcher: Callable[[ETPacket], ETPacket],
        metrics: Optional[ETMetrics] = None
    ):
        self._dispatcher = dispatcher
        self._metrics    = metrics or ETMetrics()
        self._log        = ETLog.get("et_ipc")

        # IOCP handle
        self._h_iocp: int = 0

        # Active connections: pipe_handle (int) → PipeConnection
        self._connections: Dict[int, PipeConnection] = {}
        self._conn_lock   = threading.Lock()

        # Worker thread pool: S = 12 workers
        self._workers: List[threading.Thread] = []
        self._n_workers: int = S  # 12

        # Work queue for dispatching (S² = 144 depth)
        self._work_queue: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)

        # Dispatcher threads: S/2 = 6 (dispatch while rest handle I/O)
        # ET derivation: S/2 = 6 fulfils d=6 (hexadic mediation)
        self._dispatch_threads: List[threading.Thread] = []
        self._n_dispatch: int = S // 2  # 6

        # Pipe instances: pid → pipe handle
        self._pipe_handles: Dict[int, int] = {}
        self._pipe_lock     = threading.Lock()

        self._running = False
        self._stop_event = threading.Event()

        # Per-connection overlapped structures kept alive (ctypes GC prevention)
        self._overlapped_pool: Dict[int, OVERLAPPED] = {}

        self._log.info("ETIPCServer initialised: %d workers, %d dispatchers, buf=%d",
                       self._n_workers, self._n_dispatch, IPC_BUFFER_SIZE)

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        """Start the IPC server: create IOCP and spawn workers."""
        try:
            # Create the I/O Completion Port
            with et_context("creating IOCP for IPC server", log_fn=self._log.incoherence):
                h_iocp = win32_check_handle(
                    getattr(kernel32, 'CreateIoCompletionPort')(
                        INVALID_HANDLE_VALUE,   # no file handle yet
                        None,                   # create new IOCP
                        0,                      # completion key (unused at creation)
                        self._n_workers         # max concurrent threads = S
                    ),
                    "CreateIoCompletionPort",
                    invalid=0,
                )
            self._h_iocp = h_iocp
            self._running = True

            # Spawn IOCP worker threads
            for i in range(self._n_workers):
                t = threading.Thread(
                    target=self._iocp_worker,
                    name=f"ET_IPC_Worker_{i+1}",
                    daemon=True
                )
                t.start()
                self._workers.append(t)

            # Spawn dispatch worker threads
            for i in range(self._n_dispatch):
                t = threading.Thread(
                    target=self._dispatch_worker,
                    name=f"ET_IPC_Dispatch_{i+1}",
                    daemon=True
                )
                t.start()
                self._dispatch_threads.append(t)

            record_op()
            self._log.info("ETIPCServer started: IOCP=0x%X, %d workers, %d dispatchers",
                           self._h_iocp, self._n_workers, self._n_dispatch)
            return True

        except ETOperationError as exc:
            self._log.incoherence("ETIPCServer.start failed: %s", exc)
            record_error(exc)
            return False
        except Exception as exc:
            self._log.incoherence("ETIPCServer.start failed: %s", exc)
            record_error(ETWindowsAPIError(
                f"start() raised {type(exc).__name__}: {exc}",
                cause=exc,
                os_error=ctypes.GetLastError(),
                severity=ETErrorSeverity.BOUNDARY,
                depth=2,
            ))
            return False

    def stop(self):
        """Graceful shutdown: drain queue, signal workers, close all pipes."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        # Post S+S_DISPATCH shutdown keys to IOCP so all workers wake
        total_workers = self._n_workers + self._n_dispatch
        for _ in range(total_workers):
            safe_call(
                getattr(kernel32, 'PostQueuedCompletionStatus'),
                self._h_iocp, 0, IOCP_KEY_SHUTDOWN, None,
                operation="PostQueuedCompletionStatus(shutdown)",
                log_fn=self._log.warning,
            )

        # Poison pills for dispatch queue
        for _ in range(self._n_dispatch):
            self._work_queue.put(None)

        # Wait for workers
        for t in self._workers + self._dispatch_threads:
            t.join(timeout=float(CONN_TIMEOUT_MS) / 1000.0 * 2)

        # Close all pipe handles
        with self._conn_lock:
            for conn in self._connections.values():
                self._close_pipe(conn)
            self._connections.clear()

        if _check_handle(self._h_iocp):
            safe_call(
                getattr(kernel32, 'CloseHandle'),
                self._h_iocp,
                operation="CloseHandle(IOCP)",
                log_fn=self._log.warning,
            )
            self._h_iocp = 0

        self._log.info("ETIPCServer stopped. Metrics: %s", self._metrics.summary())

    def create_pipe_for_pid(self, pid: int) -> bool:
        """
        Create a named pipe instance for a specific 32-bit target PID.
        The pipe name encodes the PID so the injected stub can find it.

        Returns True if the pipe was created and is waiting for a connection.
        """
        # Validate PID — D-constraint coherence: never bridge our own process
        if pid == _BROKER_PID:
            record_error(ETConfigError(
                f"Refusing to create pipe for broker's own PID {pid}",
                severity=ETErrorSeverity.BOUNDARY,
            ))
            self._log.error("Cannot bridge our own process (PID %d)", pid)
            return False

        pipe_name = PIPE_NAME_TEMPLATE.format(pid=pid)
        with self._pipe_lock:
            if pid in self._pipe_handles:
                self._log.mediation("Pipe already exists for PID %d", pid)
                return True

        h_pipe = getattr(kernel32, 'CreateNamedPipeW')(
            pipe_name,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_WRITE_THROUGH,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,   # allow re-connect after disconnect
            IPC_BUFFER_SIZE,            # out-buffer = full IPC buffer
            IPC_BUFFER_SIZE,            # in-buffer = full IPC buffer
            CONN_TIMEOUT_MS,
            None                        # default security
        )

        if not _check_handle(h_pipe):
            err = getattr(kernel32, 'GetLastError')()
            self._log.error("CreateNamedPipeW failed for PID %d: error %d", pid, err)
            record_error(ETWindowsAPIError(
                "CreateNamedPipeW",
                os_error=err, et_pid=pid,
                severity=ETErrorSeverity.BOUNDARY,
            ))
            return False

        # Associate with IOCP using pid as completion key
        result = getattr(kernel32, 'CreateIoCompletionPort')(
            h_pipe,
            self._h_iocp,
            pid,   # completion key = pid
            0
        )
        if not _check_handle(result):
            err = getattr(kernel32, 'GetLastError')()
            self._log.error("CreateIoCompletionPort(assoc) failed for PID %d: error %d", pid, err)
            record_error(ETHandleError(
                "CreateIoCompletionPort(associate pipe)",
                handle=h_pipe,
                severity=ETErrorSeverity.BOUNDARY,
            ))
            getattr(kernel32, 'CloseHandle')(h_pipe)
            return False

        conn = PipeConnection(
            pid=pid,
            pipe_name=pipe_name,
            h_pipe=h_pipe,
            state="CONNECTING"
        )
        # Allocate overlapped structures
        ol = OVERLAPPED()
        ol.hEvent = getattr(kernel32, 'CreateEventW')(None, True, False, None)
        conn.ol_read  = ol
        conn.ol_write = OVERLAPPED()
        conn.ol_write.hEvent = getattr(kernel32, 'CreateEventW')(None, True, False, None)
        self._overlapped_pool[h_pipe] = ol

        with self._conn_lock:
            self._connections[h_pipe] = conn

        with self._pipe_lock:
            self._pipe_handles[pid] = h_pipe

        # Post async ConnectNamedPipe
        self._begin_connect(conn)
        record_op()
        self._log.info("Pipe created for PID %d: %s", pid, pipe_name)
        return True

    def remove_pipe_for_pid(self, pid: int):
        """Remove and close the pipe for a specific PID."""
        with self._pipe_lock:
            h = self._pipe_handles.pop(pid, None)
        if h is None:
            return
        with self._conn_lock:
            conn = self._connections.pop(h, None)
        if conn:
            self._close_pipe(conn)
        self._log.info("Pipe removed for PID %d", pid)

    def is_connected(self, pid: int) -> bool:
        """True iff the pipe for this PID has an established connection."""
        with self._pipe_lock:
            h = self._pipe_handles.get(pid)
        if h is None:
            return False
        with self._conn_lock:
            conn = self._connections.get(h)
        return conn is not None and conn.state in ("READING", "WRITING", "DISPATCHING")

    def metrics(self) -> ETMetrics:
        """
        Return the bridge IPC performance metrics (ETMetrics).

        ET derivation: metrics capture the V(system) state across all
        command families. Combined with get_registry() error data,
        this provides the complete E-state health picture.
        """
        return self._metrics

    def health_summary(self) -> Dict[str, Any]:
        """
        Return combined IPC health: metrics + error registry + connection states.

        Uses get_registry() for the centralized error picture and
        per-connection tightness/coherence_depth for individual health.
        """
        registry = get_registry()
        with self._conn_lock:
            conn_states = {
                pid: {
                    "state": conn.state,
                    "variance": conn.variance(),
                    "tightness": conn.conn_tightness(),
                    "coherence_depth": conn.conn_coherence_depth(),
                    "errors": conn.errors,
                    "idle_s": round(conn.idle_seconds(), 2),
                }
                for h, conn in self._connections.items()
                for pid in [conn.pid]
            }
        return {
            "metrics": self._metrics.summary(),
            "registry": registry.summary(),
            "connections": conn_states,
            "broker_pid": _BROKER_PID,
        }

    # -------------------------------------------------------------------------
    # INTERNAL: OVERLAPPED CONNECT
    # -------------------------------------------------------------------------

    def _begin_connect(self, conn: PipeConnection):
        """Issue an overlapped ConnectNamedPipe call."""
        conn.state = "CONNECTING"
        ol = conn.ol_read
        getattr(kernel32, 'ResetEvent')(ol.hEvent)
        ok = getattr(kernel32, 'ConnectNamedPipe')(conn.h_pipe, ctypes.byref(ol))
        if not ok:
            err = getattr(kernel32, 'GetLastError')()
            if err == ERROR_IO_PENDING:
                # Normal: async connection pending — IOCP will notify
                pass
            elif err == ERROR_PIPE_CONNECTED:
                # Client connected before we called ConnectNamedPipe
                conn.state = "READING"
                self._begin_read(conn)
            else:
                self._log.warning("ConnectNamedPipe error %d for PID %d", err, conn.pid)
                self._reconnect(conn)

    # -------------------------------------------------------------------------
    # INTERNAL: OVERLAPPED READ
    # -------------------------------------------------------------------------

    def _begin_read(self, conn: PipeConnection):
        """Issue an overlapped ReadFile call for the next packet header."""
        conn.state = "READING"
        conn.bytes_read = 0
        buf = (ctypes.c_byte * IPC_BUFFER_SIZE).from_buffer(conn.recv_buf)
        ol  = conn.ol_read
        getattr(kernel32, 'ResetEvent')(ol.hEvent)
        read_bytes = ctypes.wintypes.DWORD(0)
        ok = getattr(kernel32, 'ReadFile')(
            conn.h_pipe,
            buf,
            IPC_BUFFER_SIZE,
            ctypes.byref(read_bytes),
            ctypes.byref(ol)
        )
        if not ok:
            err = getattr(kernel32, 'GetLastError')()
            if err != ERROR_IO_PENDING:
                self._log.warning("ReadFile error %d for PID %d", err, conn.pid)
                self._reconnect(conn)

    # -------------------------------------------------------------------------
    # INTERNAL: OVERLAPPED WRITE
    # -------------------------------------------------------------------------

    def _begin_write(self, conn: PipeConnection, data: bytes):
        """Write a response back to the 32-bit client."""
        conn.state = "WRITING"
        # Copy data into the send buffer
        n = min(len(data), IPC_BUFFER_SIZE)
        conn.send_buf[:n] = data[:n]
        buf = (ctypes.c_byte * n).from_buffer(conn.send_buf)
        ol  = conn.ol_write
        getattr(kernel32, 'ResetEvent')(ol.hEvent)
        written = ctypes.wintypes.DWORD(0)
        ok = getattr(kernel32, 'WriteFile')(
            conn.h_pipe,
            buf,
            n,
            ctypes.byref(written),
            ctypes.byref(ol)
        )
        if not ok:
            err = getattr(kernel32, 'GetLastError')()
            if err != ERROR_IO_PENDING:
                self._log.warning("WriteFile error %d for PID %d", err, conn.pid)
                self._reconnect(conn)

    # -------------------------------------------------------------------------
    # INTERNAL: IOCP WORKER THREAD
    # -------------------------------------------------------------------------

    def _iocp_worker(self):
        """
        IOCP completion drain thread.
        Calls GetQueuedCompletionStatus, dispatches completed I/O.
        S=12 of these run concurrently.
        """
        while self._running or not self._stop_event.is_set():
            bytes_transferred = ctypes.wintypes.DWORD(0)
            completion_key    = ctypes.POINTER(ctypes.c_ulong)()
            p_ol              = ctypes.POINTER(OVERLAPPED)()

            ok = getattr(kernel32, 'GetQueuedCompletionStatus')(
                self._h_iocp,
                ctypes.byref(bytes_transferred),
                ctypes.byref(completion_key),
                ctypes.byref(p_ol),
                CONN_TIMEOUT_MS * 2   # wait up to 2× timeout per iteration
            )

            if not ok:
                err = getattr(kernel32, 'GetLastError')()
                if err == WAIT_TIMEOUT:
                    continue
                if not self._running:
                    break
                # Broken pipe or other error — find the connection
                if p_ol:
                    self._handle_iocp_error(p_ol, err)
                continue

            # Check for shutdown sentinel
            key_val = ctypes.cast(completion_key, ctypes.c_void_p).value or 0
            if key_val == IOCP_KEY_SHUTDOWN:
                break

            # Find the connection by matching the OVERLAPPED pointer
            if not p_ol:
                continue
            ol_addr = ctypes.addressof(p_ol.contents)
            conn = self._find_connection_by_overlapped(ol_addr)
            if conn is None:
                continue

            conn.last_activity = time.monotonic()

            if conn.state == "CONNECTING":
                # Connection established
                conn.state = "READING"
                self._log.info("PID %d connected to pipe", conn.pid)
                record_op()
                self._begin_read(conn)

            elif conn.state == "READING":
                n = bytes_transferred.value
                if n == 0:
                    # Client disconnected
                    self._reconnect(conn)
                    continue
                conn.bytes_read = n
                # Deserialise packet
                try:
                    raw = bytes(conn.recv_buf[:n])

                    # Level 0 — struct header sanity check before full deserialise
                    # PDT_HEADER_SIZE = 48 bytes; first 4 bytes are cmd_family (uint32)
                    if n < PDT_HEADER_SIZE:
                        record_error(ETPacketError(
                            f"Short packet from PID {conn.pid}: {n} bytes "
                            f"(minimum {PDT_HEADER_SIZE})",
                            severity=ETErrorSeverity.MEDIATION,
                        ))
                        conn.errors += 1
                        self._begin_read(conn)
                        continue

                    # Peek at cmd_family via struct to validate before full deserialize
                    raw_family = struct.unpack_from('<I', raw, 16)[0]
                    if not (CmdFamily.MEMORY_BASIC <= raw_family <= CmdFamily.COMPOUND_OPS):
                        record_error(ETPacketError(
                            f"Invalid cmd_family d={raw_family} from PID {conn.pid} "
                            f"(valid range: {CmdFamily.MEMORY_BASIC}..{CmdFamily.COMPOUND_OPS})",
                            severity=ETErrorSeverity.MEDIATION,
                        ))
                        conn.errors += 1
                        self._begin_read(conn)
                        continue

                    pkt = ETPacket.deserialise(raw)
                    if pkt is None:
                        self._log.warning("Invalid packet from PID %d (%d bytes)", conn.pid, n)
                        record_error(ETPacketError(
                            f"Deserialisation returned None for PID {conn.pid} ({n} bytes)",
                            severity=ETErrorSeverity.MEDIATION,
                        ))
                        conn.errors += 1
                        self._begin_read(conn)
                        continue
                    conn.state = "DISPATCHING"
                    self._metrics.record(pkt.cmd_family, 0.0, True, n)
                    record_op()

                    # Koide fill-ratio early warning: K × QUEUE_DEPTH = 2/3 × 144 = 96
                    # When queue exceeds K-fill, log a mediation warning (back-pressure imminent)
                    koide_threshold = int(K * QUEUE_DEPTH)
                    current_depth = self._work_queue.qsize()
                    if current_depth >= koide_threshold:
                        self._log.warning_di(
                            "Work queue at Koide threshold: %d/%d (K=%.4f) for PID %d",
                            current_depth, QUEUE_DEPTH, float(K), conn.pid,
                        )

                    # Push to dispatch queue
                    try:
                        self._work_queue.put_nowait((conn, pkt))
                    except queue.Full:
                        # Queue at full capacity — push back-pressure response
                        self._log.warning_di("Work queue full (%d) for PID %d",
                                             QUEUE_DEPTH, conn.pid)
                        resp = self._make_error_response(pkt, 0xE000001)  # ET error: queue full
                        self._begin_write(conn, resp)
                except Exception as exc:
                    self._log.incoherence("Packet processing error PID %d: %s", conn.pid, exc)
                    record_error(ETIPCError(
                        f"Packet processing: {exc}",
                        pid=conn.pid,
                        severity=ETErrorSeverity.MEDIATION,
                    ))
                    self._begin_read(conn)

            elif conn.state == "WRITING":
                # Write complete — loop back to reading
                conn.state = "READING"
                record_op()
                self._begin_read(conn)

        self._log.mediation("IOCP worker thread exiting")

    # -------------------------------------------------------------------------
    # INTERNAL: DISPATCH WORKER THREAD
    # -------------------------------------------------------------------------

    def _dispatch_worker(self):
        """
        Dispatch thread: dequeues (conn, pkt) pairs, calls dispatcher, writes response.
        S/2 = 6 of these run concurrently (d=6 hexadic mediation).
        """
        while True:
            item: Optional[Tuple[PipeConnection, ETPacket]] = self._work_queue.get()
            if item is None:
                break  # shutdown poison pill
            conn, pkt = item
            try:
                # Log dispatched packet with unpack_args for argument visibility
                if pkt.payload and pkt.arg_count > 0:
                    arg_values = unpack_args(pkt.payload)
                    self._log.mediation(
                        "Dispatching d=%d code=0x%02X PID=%d args=%s",
                        pkt.cmd_family, pkt.cmd_code, conn.pid, arg_values,
                    )

                t0 = time.monotonic()
                response_pkt = self._dispatcher(pkt)
                latency_us   = (time.monotonic() - t0) * 1_000_000
                self._metrics.record(pkt.cmd_family, latency_us, True)
                record_op()

                resp_bytes = response_pkt.serialise()
                self._begin_write(conn, resp_bytes)
            except Exception as exc:
                self._log.incoherence("Dispatch error PID %d: %s", conn.pid, exc)
                # Write error packet — use specific ETDispatchError
                record_error(ETDispatchError(
                    f"Dispatch raised {type(exc).__name__}: {exc}",
                    et_pid=conn.pid,
                    et_family=pkt.cmd_family,
                    et_code=pkt.cmd_code,
                    cause=exc,
                    os_error=ctypes.GetLastError(),
                    severity=ETErrorSeverity.BOUNDARY,
                    depth=2,
                ))
                resp = self._make_error_response(pkt, 0xE000000)
                self._begin_write(conn, resp)
            finally:
                self._work_queue.task_done()

        self._log.mediation("Dispatch worker thread exiting")

    # -------------------------------------------------------------------------
    # INTERNAL: HELPERS
    # -------------------------------------------------------------------------

    def _find_connection_by_overlapped(self, ol_addr: int) -> Optional[PipeConnection]:
        """Find the PipeConnection whose ol_read or ol_write is at ol_addr."""
        with self._conn_lock:
            for conn in self._connections.values():
                if conn.ol_read and ctypes.addressof(conn.ol_read) == ol_addr:
                    return conn
                if conn.ol_write and ctypes.addressof(conn.ol_write) == ol_addr:
                    return conn
        return None

    def _classify_and_record_error(
        self, conn: PipeConnection, error_code: int, context: str
    ):
        """
        Classify an IPC error by its Windows error code and record it using the
        appropriate domain-specific ET error type.

        ET derivation: the error code's nature determines the D-classification.
        The IPC layer is the Mediation ({D,T}) state — it sees errors from ALL
        domains flowing through the pipe. The error code acts as D-constraint
        that selects the correct error family:
          ERROR_NO_DATA / ERROR_MORE_DATA → ETAWEError  (memory/AWE domain)
          ERROR_BROKEN_PIPE              → ETHookError  (hook detach / sync)
          ERROR_PIPE_BUSY                → ETInjectionError (injection contention)
          other                          → ETIPCError   (generic transport)
        """
        pid = conn.pid
        if error_code in (ERROR_NO_DATA, ERROR_MORE_DATA):
            self._log.mediation("AWE-domain error %d for PID %d: %s", error_code, pid, context)
            record_error(ETAWEError(
                f"{context}: error {error_code}",
                pid=pid,
                severity=ETErrorSeverity.MEDIATION,
            ))
        elif error_code == ERROR_BROKEN_PIPE:
            self._log.warning("Hook-domain error (broken pipe) for PID %d: %s", pid, context)
            record_error(ETHookError(
                f"{context}: broken pipe (hook detach?)",
                pid=pid,
                severity=ETErrorSeverity.BOUNDARY,
            ))
        elif error_code == ERROR_PIPE_BUSY:
            self._log.warning("Injection-domain error (pipe busy) for PID %d: %s", pid, context)
            record_error(ETInjectionError(
                f"{context}: pipe busy (injection contention?)",
                pid=pid,
                severity=ETErrorSeverity.MEDIATION,
            ))
        else:
            self._log.mediation("IPC transport error %d for PID %d: %s", error_code, pid, context)
            record_error(ETIPCError(
                f"{context}: error {error_code}",
                pid=pid,
                severity=ETErrorSeverity.MEDIATION,
            ))

    def _handle_iocp_error(self, p_ol, error_code: int):
        """Handle an IOCP error by finding and reconnecting the affected pipe."""
        ol_addr = ctypes.addressof(p_ol.contents)
        conn = self._find_connection_by_overlapped(ol_addr)
        if conn:
            self._log.warning("IOCP error %d for PID %d, reconnecting", error_code, conn.pid)
            # Record via win32_check for full OS-level error capture
            try:
                win32_check(
                    False,  # force error path — we already know this failed
                    f"IOCP completion for PID {conn.pid}",
                    et_pid=conn.pid,
                    severity=ETErrorSeverity.MEDIATION,
                )
            except ETWindowsAPIError as api_err:
                record_error(api_err)
            self._classify_and_record_error(conn, error_code, "IOCP completion")
            self._reconnect(conn)

    def _reconnect(self, conn: PipeConnection):
        """Disconnect and re-listen on the same pipe handle."""
        if conn.state == "CLOSED":
            return
        conn.state = "CONNECTING"
        conn.errors += 1
        # Disconnect without closing — reuse the pipe handle
        getattr(kernel32, 'DisconnectNamedPipe')(conn.h_pipe)
        # Re-listen for a new connection
        self._begin_connect(conn)

    def _close_pipe(self, conn: PipeConnection):
        """Fully close a pipe connection and release all associated handles."""
        if conn.state == "CLOSED":
            return
        conn.state = "CLOSED"
        self._log.mediation("Closing pipe for PID %d (handle=0x%X)", conn.pid, conn.h_pipe)
        if conn.ol_read and conn.ol_read.hEvent:
            safe_call(
                getattr(kernel32, 'CloseHandle'), conn.ol_read.hEvent,
                operation="CloseHandle(ol_read.hEvent)",
                et_pid=conn.pid, log_fn=self._log.warning,
            )
        if conn.ol_write and conn.ol_write.hEvent:
            safe_call(
                getattr(kernel32, 'CloseHandle'), conn.ol_write.hEvent,
                operation="CloseHandle(ol_write.hEvent)",
                et_pid=conn.pid, log_fn=self._log.warning,
            )
        if _check_handle(conn.h_pipe):
            safe_call(
                getattr(kernel32, 'DisconnectNamedPipe'), conn.h_pipe,
                operation="DisconnectNamedPipe",
                et_pid=conn.pid, log_fn=self._log.warning,
            )
            safe_call(
                getattr(kernel32, 'CloseHandle'), conn.h_pipe,
                operation="CloseHandle(pipe)",
                et_pid=conn.pid, log_fn=self._log.warning,
            )

    def _make_error_response(self, request: ETPacket, error_code: int) -> bytes:
        """
        Build an error response ETPacket.
        The payload carries the error code as an unsigned 32-bit integer.
        """
        payload, count = pack_args(error_code)
        self._metrics.record(request.cmd_family, 0.0, False)
        resp = ETPacket(
            source_pid  = request.dest_pid,
            dest_pid    = request.source_pid,
            space_token = request.space_token,
            cmd_family  = request.cmd_family,
            cmd_code    = CmdCode.CTRL_ERR,
            flags       = ETPacket.FLAG_RESPONSE | ETPacket.FLAG_ERROR,
            arg_count   = count,
            payload     = payload,
            sequence    = request.sequence,
        )
        return resp.serialise()


# ============================================================================
# ET IPC CLIENT — 32-bit side (called from injected Python stub if used)
# ============================================================================

class ETIPCClient:
    """
    32-bit IPC client that connects to the 64-bit broker's named pipe.

    Used by et32_bridge_helper.py which runs as a 32-bit Python process
    co-located with the target, or by the injected Python stub.

    ET derivation:
      P = the named pipe (shared substrate)
      D = the PID-encoded pipe name (Descriptor of this client's identity)
      T = this client process (traverser through the pipe)

    Connection retry = RETRY_COUNT = S = 12 attempts.
    Retry interval   = CONN_TIMEOUT_MS / S = 125 ms per attempt.
    """

    def __init__(self, pid: int):
        self._pid        = pid
        self._pipe_name  = PIPE_NAME_TEMPLATE.format(pid=pid)
        self._h_pipe: int = 0
        self._lock        = threading.RLock()  # RLock: reentrant for nested sequence/send
        self._sequence    = 0
        self._log         = ETLog.get("et_ipc_client")

    def connect(self) -> bool:
        """
        Connect to the broker's named pipe.
        Retries RETRY_COUNT = S = 12 times with WaitNamedPipe.
        Returns True on success.
        """
        retry_interval_ms = CONN_TIMEOUT_MS // RETRY_COUNT  # 125 ms

        # Level 4 Cascade Coherence (incoherence_filter_-_lattice.txt):
        # N_max = ⌊50¢/|δ|⌋ where δ = deviation per retry step.
        # For canonical ET generators (K=2/3, 1/S): |δ|=1.955¢ → N_max=25.
        # effective_retries = min(RETRY_COUNT, COHERENCE_N_MAX) = min(12, 25) = 12.
        # RETRY_COUNT=12 is strictly within the coherence window of 25.
        # If observed timing deviation is larger than canonical, N_max shrinks;
        # we cap retries at the tighter limit.
        _t0_connect = time.monotonic()
        effective_retries = min(RETRY_COUNT, COHERENCE_N_MAX)

        for attempt in range(effective_retries):
            # WaitNamedPipe blocks until a pipe instance is available
            ok = getattr(kernel32, 'WaitNamedPipeW')(self._pipe_name, retry_interval_ms)
            if not ok:
                err = getattr(kernel32, 'GetLastError')()
                if err == WAIT_TIMEOUT:
                    self._log.mediation(
                        "Connect attempt %d/%d: pipe not yet available for PID %d",
                        attempt + 1, effective_retries, self._pid,
                    )
                    # Level 4: compute timing tightness for this attempt
                    _elapsed = time.monotonic() - _t0_connect
                    _expected = (attempt + 1) * retry_interval_ms / 1000.0
                    if _elapsed > 0 and _expected > 0:
                        import math as _m
                        _ratio = _elapsed / _expected
                        _delta = (12.0 * _m.log2(_ratio) - round(12.0 * _m.log2(_ratio))) * 100.0
                        _n_actual = n_max_cascade(_delta)
                        if attempt + 1 >= _n_actual:
                            self._log.warning_di(
                                "Level 4 𝒜_I: cascade coherence horizon N_max=%d "
                                "reached at attempt %d — aborting retries",
                                _n_actual, attempt + 1,
                            )
                            break
                    continue
                # Non-timeout error
                self._log.warning("WaitNamedPipe error %d (attempt %d)", err, attempt + 1)
                time.sleep(retry_interval_ms / 1000.0)
                continue

            h = getattr(kernel32, 'CreateFileW')(
                self._pipe_name,
                GENERIC_READ | GENERIC_WRITE,
                0,            # no sharing
                None,         # default security
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
                None
            )
            if _check_handle(h):
                self._h_pipe = h
                record_op()
                self._log.info("Connected to broker pipe for PID %d", self._pid)
                return True
            err = getattr(kernel32, 'GetLastError')()
            self._log.warning("CreateFileW error %d (attempt %d)", err, attempt + 1)
            time.sleep(retry_interval_ms / 1000.0)

        self._log.incoherence("Failed to connect to broker pipe after %d attempts (N_max=%d)", effective_retries, COHERENCE_N_MAX)
        return False

    def disconnect(self):
        """Close the pipe connection."""
        if _check_handle(self._h_pipe):
            getattr(kernel32, 'CloseHandle')(self._h_pipe)
            self._h_pipe = 0

    def send_request(self, pkt: ETPacket) -> Optional[ETPacket]:
        """
        Send an ETPacket to the broker and wait for a response.
        Thread-safe (uses internal RLock).

        Returns the response ETPacket, or None on I/O failure.
        Timeout = CONN_TIMEOUT_MS = 1500 ms.
        """
        if not _check_handle(self._h_pipe):
            self._log.error("Not connected — call connect() first")
            return None

        with self._lock:
            # RLock is reentrant, so a nested acquisition would not deadlock,
            # but it is redundant and misleading.  The outer lock already serializes
            # the entire send_request body — sequence increment and serialize are
            # already protected.  The redundant inner lock was removed (double-lock fix):
            # it added zero mutual-exclusion benefit and obscured the intended scope.
            # V(redundant_lock) = V_BASE — a Descriptor that adds no constraint is
            # an Unsubstantiated D that should not exist.
            self._sequence += 1
            pkt.sequence = self._sequence

            data = pkt.serialise()

            # Write request
            t0_send = time.monotonic()
            written = ctypes.wintypes.DWORD(0)
            ok = getattr(kernel32, 'WriteFile')(
                self._h_pipe,
                data,
                len(data),
                ctypes.byref(written),
                None
            )
            if not ok or written.value != len(data):
                err = getattr(kernel32, 'GetLastError')()
                self._log.error("WriteFile failed: error %d", err)
                record_error(ETIPCError(
                    "WriteFile failed",
                    pid=self._pid,
                    severity=ETErrorSeverity.MEDIATION,
                    os_error=err,
                ))
                return None

            # Read response (header first, then payload)
            resp_buf = ctypes.create_string_buffer(IPC_BUFFER_SIZE)
            read_bytes = ctypes.wintypes.DWORD(0)
            ok = getattr(kernel32, 'ReadFile')(
                self._h_pipe,
                resp_buf,
                IPC_BUFFER_SIZE,
                ctypes.byref(read_bytes),
                None
            )
            if not ok:
                err = getattr(kernel32, 'GetLastError')()
                self._log.error("ReadFile failed: error %d", err)
                record_error(ETIPCError(
                    "ReadFile failed",
                    pid=self._pid,
                    severity=ETErrorSeverity.MEDIATION,
                    os_error=err,
                ))
                return None

            t1_recv = time.monotonic()

            n = read_bytes.value
            if n < PDT_HEADER_SIZE:
                self._log.warning("Short response: %d bytes (expected >= %d)", n, PDT_HEADER_SIZE)
                record_error(ETPacketError(
                    f"Short response from broker: {n} bytes (min {PDT_HEADER_SIZE})",
                    severity=ETErrorSeverity.MEDIATION,
                ))
                return None

            # Level 0 — struct header peek: validate cmd_family before full deserialize
            raw_resp = bytes(resp_buf[:n])
            raw_family = struct.unpack_from('<I', raw_resp, 16)[0]
            if not (CmdFamily.MEMORY_BASIC <= raw_family <= CmdFamily.COMPOUND_OPS):
                record_error(ETPacketError(
                    f"Response has invalid cmd_family d={raw_family}",
                    severity=ETErrorSeverity.MEDIATION,
                ))
                return None

            resp = ETPacket.deserialise(raw_resp)

            # ── Level 2 𝒜_I — Pairwise coherence check (request, response) ───────
            # The (request, response) pair is coherent iff:
            #   1. Sequence numbers match (D-bridge: request's k == response's k)
            #   2. Timing ratio does not introduce a rounding-flip contradiction
            #   3. Response flags with NTSTATUS error are validated
            # Source: incoherence_filter_-_lattice.txt Level 2
            if resp is not None:
                # Check 1: sequence number match — primary Level 2 𝒜_I
                if resp.sequence != pkt.sequence:
                    self._log.warning_di(
                        "Level 2 𝒜_I: sequence mismatch "
                        "(request seq=%d, response seq=%d) PID=%d — incoherent pair",
                        pkt.sequence, resp.sequence, self._pid,
                    )
                    record_error(ETIPCError(
                        "Level 2 pairwise 𝒜_I: sequence number mismatch",
                        pid       = self._pid,
                        severity  = ETErrorSeverity.MEDIATION,
                        req_seq   = pkt.sequence,
                        resp_seq  = resp.sequence,
                    ))
                    return None

                # Check 2: cmd_family coherence — response family must match request
                if resp.cmd_family != pkt.cmd_family:
                    self._log.warning_di(
                        "Level 2 𝒜_I: cmd_family mismatch "
                        "(request d%d, response d%d) PID=%d",
                        pkt.cmd_family, resp.cmd_family, self._pid,
                    )
                    record_error(ETIPCError(
                        "Level 2 pairwise 𝒜_I: cmd_family mismatch",
                        pid        = self._pid,
                        severity   = ETErrorSeverity.MEDIATION,
                        req_family = pkt.cmd_family,
                        resp_family= resp.cmd_family,
                    ))
                    return None

                # Check 3: Level 2 timing-based pairwise_incoherence
                # The round-trip latency ratio should not cause a rounding flip
                # relative to the expected retry_interval timing unit.
                roundtrip_s = t1_recv - t0_send
                expected_s  = CONN_TIMEOUT_MS / 1000.0
                if roundtrip_s > 0 and expected_s > 0:
                    timing_ratio = roundtrip_s / expected_s
                    ai_flag, delta_eps = pairwise_incoherence(timing_ratio, 1.0)
                    if ai_flag:
                        self._log.warning_di(
                            "Level 2 𝒜_I: timing rounding flip "
                            "(ratio=%.4f, Δε=%.2f¢) PID=%d — incoherent timing pair",
                            timing_ratio, delta_eps, self._pid,
                        )
                        # Timing incoherence is a MEDIATION warning, not a hard reject:
                        # the data may still be valid even if the timing is anomalous
                        record_error(ETIPCError(
                            f"Level 2 timing 𝒜_I: rounding flip Δε={delta_eps:.2f}¢",
                            pid=self._pid,
                            severity=ETErrorSeverity.MEDIATION,
                            timing_ratio=timing_ratio,
                            delta_eps=delta_eps,
                        ))

                # Check 4: if response carries an NTSTATUS error, validate it
                if resp.flags & ETPacket.FLAG_ERROR and resp.payload:
                    error_val = struct.unpack_from('<I', resp.payload, 0)[0]
                    # If high 2 bits indicate NT error/warning, check via ntstatus_check
                    if (error_val >> 30) & 0x3 >= 2:
                        try:
                            ntstatus_check(
                                error_val,
                                f"broker response NTSTATUS for d={resp.cmd_family}",
                                et_pid=self._pid,
                                et_family=resp.cmd_family,
                                et_code=resp.cmd_code,
                                severity=ETErrorSeverity.MEDIATION,
                            )
                        except ETOperationError as nt_exc:
                            self._log.warning(
                                "Response carries NTSTATUS error 0x%08X for PID %d",
                                error_val, self._pid,
                            )
                            record_error(nt_exc)
                            # Still return the response — caller decides what to do

                record_op()

            return resp

    def call(
        self,
        cmd_family: int,
        cmd_code:   int,
        *args,
        flags: int = ETPacket.FLAG_REQUEST
    ) -> Optional["ETPacket"]:
        """
        High-level call: build a request packet from args, send, return response.
        """
        payload, arg_count = pack_args(*args)
        pkt = ETPacket(
            source_pid  = self._pid,
            dest_pid    = 0,  # broker does not have a fixed PID from client's perspective
            space_token = 0,
            cmd_family  = cmd_family,
            cmd_code    = cmd_code,
            flags       = flags,
            arg_count   = arg_count,
            payload     = payload,
        )
        return self.send_request(pkt)

    @property
    def connected(self) -> bool:
        """True if the pipe handle is valid — client is connected to the broker (E-state)."""
        return _check_handle(self._h_pipe)

    @property
    def pid(self) -> int:
        """
        The PID this client is bound to.

        ET derivation: the D-identity of this client's communication channel.
        Exposed as a read-only property so external code (ETAPIGateway) can
        read the PID without accessing the protected _pid attribute directly.
        """
        return self._pid


# ============================================================================
# PIPE NAME HELPER (used by injector and stubs)
# ============================================================================

def pipe_name_for_pid(pid: int) -> str:
    """Return the named pipe path for a given target PID."""
    return PIPE_NAME_TEMPLATE.format(pid=pid)
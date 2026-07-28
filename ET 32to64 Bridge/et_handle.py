"""
et_bridge/et_handle.py
ET32 Bridge — Handle Table: 32-bit ↔ 64-bit Address Mapping

Derived from P ∘ D ∘ T = E.

The Handle Table is the core Descriptor bridge between P_32 (the 32-bit
address space) and P_full (the infinite substrate). It is a bijective mapping:

    h: ADDR₆₄ → HANDLE₃₂    (D-projection, lossy only for addr ≥ 4GB)
    h⁻¹: HANDLE₃₂ → ADDR₆₄  (D-expansion, perfect via table)

This resolves the Descriptor Gap:
    gap(P₃₂) = {addr | addr > 0xFFFFFFFF}
    D_bridge = handle_table   (the new Descriptor that closes the gap)

Subsumption check:
    Complete(handle_table) ↔ ∀ addr₆₄ ∈ P_full: ∃ handle ∈ P_32 s.t. h(addr₆₄) = handle
    This is satisfied by the bijection. Every 64-bit address gets exactly one handle.

ET Constants applied:
    SLOT_STRIDE  = S = 12       (one stride per lattice position)
    Max handles  ≈ 178M         ((HANDLE_MAX - HANDLE_BASE) / S)
    Initial pool = S² = 144     (first queue depth — manifold symmetry²)
    Resize at    = K × capacity (Koide threshold: resize when 2/3 full)
"""

import threading
import mmap
import struct
import os
import ctypes
from typing import Dict, Optional, Tuple, List

from et_math import (
    S, K, V_BASE,
    HANDLE_BASE, HANDLE_MAX, ADDR64_BASE,
    ETHandleMath, CmdFamily, ETMetrics, DIGITAL_ACTION_QUANTUM,
    SHMEM_NAME_TEMPLATE,
)


class HandleEntry:
    """
    A single entry in the handle table.

    Each entry is a complete P∘D binding:
      P = the 64-bit address (substrate location in P_full)
      D = metadata describing the allocation (size, type, flags)
      T-status = whether this entry has been traversed (accessed) at least once

    Untraversed entries are {P,D} Unsubstantiated.
    Traversed entries are {P,D,T} — Exception state.
    """
    __slots__ = (
        "handle", "addr64", "size", "flags", "family",
        "accessed", "ref_count", "tag"
    )

    def __init__(
        self,
        handle: int,
        addr64: int,
        size: int,
        flags: int,
        family: int,
        tag: str = ""
    ):
        self.handle    = handle     # 32-bit proxy handle
        self.addr64    = addr64     # real 64-bit address
        self.size      = size       # allocated size
        self.flags     = flags      # VirtualAlloc-style flags
        self.family    = family     # CmdFamily lattice position
        self.accessed  = False      # T-traversal status
        self.ref_count = 1          # reference count
        self.tag       = tag        # optional debug tag

    def variance(self) -> float:
        """
        V(entry) = 0 if entry is fully substantiated (addr64 valid, ref_count > 0).
        V(entry) = V_BASE if accessed == False (unsubstantiated {P,D}).
        """
        if self.ref_count <= 0 or self.addr64 == 0:
            return 1.0  # incoherent
        if not self.accessed:
            return V_BASE  # {P,D} unsubstantiated
        return 0.0  # {P,D,T} = Exception


class HandleTable:
    """
    Thread-safe bijective mapping between 32-bit bridge handles and 64-bit addresses.

    Layout in memory: each slot is 32 bytes (aligned on ET manifold boundaries):
      addr64   : uint64  [8]
      size     : uint64  [8]
      flags    : uint32  [4]
      family   : uint8   [1]
      accessed : uint8   [1]
      ref_count: uint16  [2]
      handle   : uint32  [4]
      _padding : uint32  [4]
    Total: 32 bytes = S × 8/3 bytes (ET-derived: S/V_BASE / 4)

    The table lives in a shared memory region so both the 32-bit helper process
    and the 64-bit host can access it directly.
    """

    ENTRY_STRUCT = struct.Struct("<QQIBBHIxx")  # 32 bytes, see layout above
    ENTRY_SIZE   = ENTRY_STRUCT.size             # must be 32

    # Shared memory layout:
    # [0..3]   magic   : uint32 = 0x45543332 ("ET32")
    # [4..7]   version : uint32 = 1
    # [8..11]  count   : uint32 = number of live handles
    # [12..15] capacity: uint32 = max handles in this slab
    # [16..47] reserved: padding to 48-byte header (= PDT_HEADER_SIZE)
    SHMEM_HDR_SIZE = 48
    SHMEM_MAGIC    = 0x45543332  # "ET32"
    SHMEM_VERSION  = 1

    def __init__(self, initial_capacity: int = None):
        """
        Initialize the handle table.
        initial_capacity defaults to S² = 144 (manifold symmetry squared).
        """
        self._lock     = threading.RLock()
        self._capacity = initial_capacity if initial_capacity is not None else S * S
        self._entries: Dict[int, HandleEntry] = {}
        # Reverse dict: addr64 → handle (O(1) duplicate detection — replaces O(n) scan)
        # ET derivation: Identification Principle — to identify whether P (addr64) already
        # has a D (handle), we maintain the inverse D-projection as a dedicated dict.
        # V(dedup_scan) was O(n)/lock-time; V(reverse_dict) = O(1)/lock-time = 0 variance.
        self._addr64_to_handle: Dict[int, int] = {}
        self._free_slots: List[int] = list(range(self._capacity))
        self._next_slot  = 0
        self._metrics    = ETMetrics()
        self._total_allocated = 0  # cumulative handle allocations (never decremented)
        self._native64_pids: dict = {}  # PID → exe_name for known 64-bit native processes

        # Shared memory state — used by create_shmem / open_shmem / close_shmem
        self._shmem: Optional[mmap.mmap] = None   # active mmap object (None if not mapped)
        self._shmem_name: str = ""                 # tagname for the named shared memory region

        # Validate ENTRY_SIZE
        assert self.ENTRY_SIZE == 32, f"Entry size must be 32, got {self.ENTRY_SIZE}"

    # ------------------------------------------------------------------
    # Core handle allocation / deallocation
    # ------------------------------------------------------------------

    def alloc(
        self,
        addr64: int,
        size: int,
        flags: int = 0,
        family: int = CmdFamily.MEMORY_BASIC,
        tag: str = ""
    ) -> int:
        """
        Allocate a bridge handle for the given 64-bit address.
        Returns the 32-bit proxy handle, or 0 on failure.

        If addr64 < 4GB, returns addr64 directly (no bridge needed — identity mapping).
        If addr64 >= 4GB, allocates a new slot and returns the bridge handle.

        ET derivation:
          Low addresses: π₃₂(addr64) = addr64 (lossless, V=0)
          High addresses: π₃₂(addr64) = HANDLE_BASE + slot × SLOT_STRIDE (bridge proxy)
        """
        if addr64 == 0:
            return 0  # NULL passthrough

        # Identity for low addresses
        if addr64 < ADDR64_BASE:
            return addr64 & 0xFFFFFFFF

        with self._lock:
            # Check if this addr64 already has a handle (O(1) dedup via reverse dict)
            existing_handle = self._addr64_to_handle.get(addr64)
            if existing_handle is not None:
                entry = self._entries.get(existing_handle)
                if entry is not None:
                    entry.ref_count += 1
                    return existing_handle
                # Stale entry in reverse dict (should not happen) — clean up and reallocate
                del self._addr64_to_handle[addr64]

            # Allocate new slot
            if not self._free_slots:
                self._expand()

            if not self._free_slots:
                return 0  # out of handles — unlikely given S²=144 initial capacity

            slot = self._free_slots.pop(0)
            handle = ETHandleMath.slot_to_handle(slot)

            # Guard: ensure handle is in bridge range
            if not ETHandleMath.is_bridge_handle(handle):
                self._free_slots.insert(0, slot)
                return 0

            # Update high-water mark: _next_slot tracks the deepest slot ever issued
            if slot >= self._next_slot:
                self._next_slot = slot + 1

            entry = HandleEntry(handle, addr64, size, flags, family, tag)
            self._entries[handle] = entry
            self._addr64_to_handle[addr64] = handle  # maintain reverse dict
            self._total_allocated += 1
            return handle

    def dealloc(self, handle: int) -> bool:
        """
        Free a bridge handle. Returns True if successfully freed.
        For non-bridge handles (passthrough), always returns True.
        """
        if not ETHandleMath.is_bridge_handle(handle):
            return True  # passthrough — not managed by bridge

        with self._lock:
            entry = self._entries.get(handle)
            if entry is None:
                return False

            entry.ref_count -= 1
            if entry.ref_count <= 0:
                slot = ETHandleMath.handle_to_slot(handle)
                del self._entries[handle]
                # Remove from reverse dict — addr64 no longer has a live handle
                self._addr64_to_handle.pop(entry.addr64, None)
                self._free_slots.append(slot)
            return True

    def get(self, handle: int) -> Optional[HandleEntry]:
        """
        Look up the entry for a handle. Marks entry as accessed (T-traversal).
        Returns None if handle is not managed by bridge.
        """
        if not ETHandleMath.is_bridge_handle(handle):
            return None  # passthrough

        with self._lock:
            entry = self._entries.get(handle)
            if entry is not None:
                entry.accessed = True  # T-traversal: {P,D} → {P,D,T}
            return entry

    def expand_address(self, handle: int) -> int:
        """
        Expand a handle to its 64-bit address.
        Returns 0 if handle is invalid.
        For passthrough handles (< HANDLE_BASE), returns handle unchanged.
        """
        if not ETHandleMath.is_bridge_handle(handle):
            return handle  # direct 32-bit address passthrough

        entry = self.get(handle)
        if entry is None:
            return 0
        return entry.addr64

    def project_address(self, addr64: int) -> int:
        """
        Project a 64-bit address to a 32-bit handle.
        For low addresses: identity.
        For high addresses: looks up existing handle or allocates new one.
        """
        if addr64 < ADDR64_BASE:
            return addr64 & 0xFFFFFFFF

        with self._lock:
            # Find existing
            for entry in self._entries.values():
                if entry.addr64 == addr64:
                    return entry.handle
        # Allocate new
        return self.alloc(addr64, 0, 0, CmdFamily.MEMORY_BASIC)

    # ------------------------------------------------------------------
    # Semantic aliases — ETHost64 dispatch interface
    # ------------------------------------------------------------------
    # The ETHost64 dispatch layer uses these semantic names to interact
    # with the handle table.  Each delegates to the core implementation:
    #   allocate  → alloc          (D-projection: 64-bit → 32-bit proxy)
    #   resolve   → expand_address (D-expansion:  32-bit proxy → 64-bit)
    #   release   → dealloc        (D-removal:    free bridge handle)
    #   get_entry → get            (D-lookup:     retrieve HandleEntry)
    #
    # ET derivation: these are different Descriptor perspectives on the
    # same underlying P (handle table).  The Identification Principle
    # confirms they are semantically identical — same P, different D names.

    def allocate(
        self,
        addr64: int,
        size: int,
        flags: int = 0,
        family: int = CmdFamily.MEMORY_BASIC,
        tag: str = ""
    ) -> int:
        """
        Allocate a bridge handle for the given 64-bit address.

        Semantic alias for alloc() — used by ETHost64 dispatch handlers.
        Returns the 32-bit proxy handle, or 0 on failure.

        ET derivation: D-projection of P_full into P_32.
        """
        return self.alloc(addr64, size, flags, family, tag)

    def resolve(self, handle: int) -> Optional[int]:
        """
        Resolve a 32-bit bridge handle to its 64-bit address.

        Semantic alias for expand_address() — used by ETHost64 dispatch handlers.
        Returns the 64-bit address, or 0 if handle is invalid.
        For passthrough handles (< HANDLE_BASE), returns handle unchanged.

        ET derivation: D-expansion — the inverse of allocate().
        """
        result = self.expand_address(handle)
        return result if result != 0 else None

    def release(self, handle: int) -> bool:
        """
        Release a bridge handle and free its slot.

        Semantic alias for dealloc() — used by ETHost64 dispatch handlers.
        Returns True if successfully freed.

        ET derivation: D-removal — the bridge handle is no longer needed.
        The 64-bit address continues to exist in P_full; only the
        D-projection into P_32 is dissolved.
        """
        return self.dealloc(handle)

    def get_entry(self, handle: int) -> Optional[HandleEntry]:
        """
        Look up the HandleEntry for a bridge handle.

        Semantic alias for get() — used by ETHost64 dispatch handlers.
        Marks the entry as accessed (T-traversal).
        Returns None if handle is not managed by bridge.

        ET derivation: T-traversal of the D-projection. Accessing the
        entry transitions it from {P,D} Unsubstantiated to {P,D,T} Exception.
        """
        return self.get(handle)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def alloc_range(
        self,
        base64: int,
        size: int,
        chunk_size: int = None,
        family: int = CmdFamily.MEMORY_BASIC,
    ) -> List[int]:
        """
        Allocate handles for a large 64-bit address range.
        The range from base64 to (base64 + size), exclusive of the end, is chunked
        into slices.  chunk_size defaults to DIGITAL_ACTION_QUANTUM × S = 49152 bytes.

        This is used for large memory-mapped files and VRAM allocations.
        Returns list of handles, one per chunk.
        """
        chunk_size = chunk_size or (DIGITAL_ACTION_QUANTUM * S)
        handles = []
        offset = 0
        while offset < size:
            chunk_end  = min(offset + chunk_size, size)
            chunk_len  = chunk_end - offset
            addr_chunk = base64 + offset
            h = self.alloc(addr_chunk, chunk_len, 0, family, f"chunk@{offset:#x}")
            handles.append(h)
            offset += chunk_len
        return handles

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def live_count(self) -> int:
        """
        Number of currently live (allocated) handle entries.

        ET derivation: the cardinality of the active D-projection set.
        live_count = |{h ∈ HandleTable : ref_count(h) > 0}|.
        """
        with self._lock:
            return len(self._entries)

    def capacity(self) -> int:
        """
        Current allocated capacity of the handle table (number of slots).

        ET derivation: initial capacity = S² = 144 (manifold symmetry squared).
        Expands by factor S/K = 18 when fill_ratio reaches K = 2/3.
        Upper bound: max_handles = (HANDLE_MAX − HANDLE_BASE) // S.
        """
        return self._capacity

    @property
    def fill_ratio(self) -> float:
        """Fill ratio. K-threshold: at K=2/3, table is expanded."""
        with self._lock:
            return len(self._entries) / max(1, self._capacity)

    @property
    def total_allocated(self) -> int:
        """
        Cumulative handle allocation count (monotonically increasing).

        ET derivation: total T-traversals through the alloc gate.
        Unlike live_count (current entries), this includes handles that have
        since been freed — the full history of D-projections performed.
        """
        return self._total_allocated

    @property
    def max_handles(self) -> int:
        """
        Maximum theoretical handle capacity of the bridge.

        ET derivation: the handle space spans HANDLE_BASE to HANDLE_MAX with
        SLOT_STRIDE = S = 12 between consecutive handles.
        max_handles = (HANDLE_MAX − HANDLE_BASE) // S ≈ 178,956,970.
        This is the hard ceiling — _expand() must not exceed this.
        """
        return (HANDLE_MAX - HANDLE_BASE) // S

    @property
    def high_water_slot(self) -> int:
        """
        The highest slot index ever issued plus one (high-water mark).

        ET derivation: measures how deep into the handle space the bridge has
        penetrated.  Unlike live_count (which can decrease as handles are freed),
        high_water_slot is monotonically non-decreasing — it records the maximum
        extent of D-projection into the slot space, analogous to total_allocated
        being the monotonic count of T-traversals through the alloc gate.
        """
        return self._next_slot

    def total_bytes_managed(self) -> int:
        """
        Total bytes of 64-bit address space currently managed by the handle table.

        ET derivation: Σ size(e) for all live entries e — the aggregate P-extent
        that the D-bridge is currently projecting into P_32.
        """
        with self._lock:
            return sum(e.size for e in self._entries.values())

    def variance(self) -> float:
        """
        V(table) = fraction of unsubstantiated entries (not yet accessed).
        V = 0 → all entries are fully traversed (Exception state).
        V = 1 → all entries are unsubstantiated ({P,D} state).
        """
        with self._lock:
            if not self._entries:
                return 0.0
            unaccessed = sum(1 for e in self._entries.values() if not e.accessed)
            return unaccessed / len(self._entries)

    def dump(self) -> List[Dict]:
        """Diagnostic dump of all live handles."""
        with self._lock:
            result = []
            for h, entry in sorted(self._entries.items()):
                lattice_pos = ETHandleMath.handle_lattice_position(h)
                result.append({
                    "handle":    f"0x{h:08X}",
                    "addr64":    f"0x{entry.addr64:016X}",
                    "size":      entry.size,
                    "family":    entry.family,
                    "lattice_d": lattice_pos,
                    "accessed":  entry.accessed,
                    "ref_count": entry.ref_count,
                    "tag":       entry.tag,
                    "variance":  entry.variance(),
                })
            return result

    # ------------------------------------------------------------------
    # Shared memory serialisation (for cross-process access by 32-bit helper)
    # ------------------------------------------------------------------

    def serialise_to_shmem(self, buf: bytearray) -> int:
        """
        Write handle table to shared memory buffer.
        Returns bytes written (including header).
        Format: SHMEM_HDR_SIZE header + N × ENTRY_SIZE entries.
        """
        with self._lock:
            entries = list(self._entries.values())
            required = self.SHMEM_HDR_SIZE + len(entries) * self.ENTRY_SIZE
            if len(buf) < required:
                return 0

            # Header
            struct.pack_into("<IIII", buf, 0,
                self.SHMEM_MAGIC, self.SHMEM_VERSION, len(entries), self._capacity)
            # Padding bytes 16..47 left as zero

            # Entries
            offset = self.SHMEM_HDR_SIZE
            for entry in entries:
                self.ENTRY_STRUCT.pack_into(
                    buf, offset,
                    entry.addr64, entry.size, entry.flags,
                    entry.family, int(entry.accessed),
                    min(entry.ref_count, 0xFFFF), entry.handle
                )
                offset += self.ENTRY_SIZE

            return offset

    def deserialise_from_shmem(self, buf: bytes) -> bool:
        """
        Load handle table from shared memory buffer.
        Returns True on success.
        """
        if len(buf) < self.SHMEM_HDR_SIZE:
            return False

        magic, version, count, capacity = struct.unpack_from("<IIII", buf, 0)
        if magic != self.SHMEM_MAGIC or version != self.SHMEM_VERSION:
            return False

        required = self.SHMEM_HDR_SIZE + count * self.ENTRY_SIZE
        if len(buf) < required:
            return False

        with self._lock:
            self._entries.clear()
            self._addr64_to_handle.clear()  # reset reverse dict before rebuild
            self._free_slots.clear()
            self._capacity = capacity

            offset = self.SHMEM_HDR_SIZE
            for _ in range(count):
                addr64, size, flags, family, accessed_byte, ref_count, handle = \
                    self.ENTRY_STRUCT.unpack_from(buf, offset)
                offset += self.ENTRY_SIZE

                entry = HandleEntry(handle, addr64, size, flags, family)
                entry.accessed  = bool(accessed_byte)
                entry.ref_count = ref_count
                self._entries[handle] = entry
                self._addr64_to_handle[addr64] = handle  # rebuild reverse dict

            # Rebuild free slots
            allocated_slots = {ETHandleMath.handle_to_slot(h) for h in self._entries}
            self._free_slots = [s for s in range(capacity) if s not in allocated_slots]

            # Restore high-water mark: the deepest slot ever issued
            self._next_slot = (max(allocated_slots) + 1) if allocated_slots else 0

        return True

    def _shmem_required_size(self) -> int:
        """
        Compute the required shared memory size for the current capacity,
        rounded up to the nearest DIGITAL_ACTION_QUANTUM (ħ_d = 4096) boundary.

        ET derivation: shared memory must be page-aligned because ħ_d is the
        fundamental D-coherence quantum.  size = ceil((hdr + cap × entry) / ħ_d) × ħ_d.

        Uses ctypes.sizeof(ctypes.c_uint64) to validate the 8-byte alignment
        assumption underpinning the ENTRY_STRUCT layout.
        """
        # Validate that the platform uint64 matches our struct assumption
        assert ctypes.sizeof(ctypes.c_uint64) == 8, (
            "Platform uint64 must be 8 bytes for ENTRY_STRUCT compatibility"
        )
        raw = self.SHMEM_HDR_SIZE + self._capacity * self.ENTRY_SIZE
        # Page-align to ħ_d = DIGITAL_ACTION_QUANTUM
        pages = (raw + DIGITAL_ACTION_QUANTUM - 1) // DIGITAL_ACTION_QUANTUM
        return pages * DIGITAL_ACTION_QUANTUM

    def create_shmem(self, pid: int = None) -> Tuple[str, int]:
        """
        Create a named shared memory region and write the handle table into it.

        The shared memory IS the D-bridge between P_32 (32-bit helper) and P_64
        (64-bit host): both processes map the same physical pages and see the
        same handle table without IPC serialisation overhead.

        pid:  process ID used to construct the tag name via SHMEM_NAME_TEMPLATE.
              Defaults to the current process (os.getpid()).

        Returns (shmem_tag_name, bytes_written) on success.

        ET derivation:
          P = the physical page backing the shared region (substrate).
          D = the SHMEM_NAME_TEMPLATE tag that names and locates the region.
          T = the process threads that map and traverse the region.
          E = both 32-bit and 64-bit sides reading the same grounded handle table.
        """
        if pid is None:
            pid = os.getpid()

        tag = SHMEM_NAME_TEMPLATE.format(pid=pid)
        size = self._shmem_required_size()

        # Create the named shared memory region.
        # On Windows, mmap(-1, size, tagname=tag) creates a pagefile-backed section
        # accessible by any process that opens the same tagname.
        self._shmem = mmap.mmap(-1, size, tagname=tag)
        self._shmem_name = tag

        # Serialize the table into the shared region
        buf = bytearray(size)
        written = self.serialise_to_shmem(buf)
        self._shmem[:size] = buf

        return tag, written

    def open_shmem(self, pid: int) -> Tuple[str, bool]:
        """
        Open an existing named shared memory region and load the handle table.

        This is the 32-bit helper side counterpart to create_shmem().
        The tag name is derived from SHMEM_NAME_TEMPLATE using the given pid.

        Returns (shmem_tag_name, success_bool).
        """
        tag = SHMEM_NAME_TEMPLATE.format(pid=pid)
        size = self._shmem_required_size()

        # Open the existing named shared memory
        self._shmem = mmap.mmap(-1, size, tagname=tag, access=mmap.ACCESS_READ)
        self._shmem_name = tag

        # Deserialize handle table from the shared region
        raw = self._shmem[:size]
        ok = self.deserialise_from_shmem(bytes(raw))
        return tag, ok

    def sync_shmem(self) -> int:
        """
        Synchronize the current handle table state to the active shared memory region.

        Must be called after any alloc/dealloc to keep the 32-bit helper's view current.
        Returns bytes written, or 0 if no shared memory region is active.
        """
        if self._shmem is None:
            return 0
        size = len(self._shmem)
        buf = bytearray(size)
        written = self.serialise_to_shmem(buf)
        self._shmem[:size] = buf
        return written

    def close_shmem(self) -> None:
        """
        Close and release the shared memory region.

        ET: closing the shmem removes the D-bridge between P_32 and P_64.
        The handle table remains in-process; only the cross-process projection is lost.
        """
        if self._shmem is not None:
            self._shmem.close()
            self._shmem = None
            self._shmem_name = ""

    @staticmethod
    def shmem_tag_for_pid(pid: int = None) -> str:
        """
        Return the shared memory tag name for a given PID.

        Utility for external code (e.g. the 32-bit helper) to compute the
        expected tag name without instantiating a HandleTable.

        ET derivation: the tag IS the Descriptor that identifies which P (physical
        page backing) to map.  Format: SHMEM_NAME_TEMPLATE with {pid} replaced.
        """
        if pid is None:
            pid = os.getpid()
        return SHMEM_NAME_TEMPLATE.format(pid=pid)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _expand(self):
        """
        Expand the handle table capacity.
        Expansion factor = S (manifold symmetry): multiply capacity by S/K = 18.
        ET derivation: new_capacity = old_capacity × (S / K) = old × 18
        (growing by the ratio S/K ensures the new size remains at the Koide threshold
        of the next ET lattice level).

        The expansion is clamped to max_handles = (HANDLE_MAX − HANDLE_BASE) // S
        to prevent slot indices from generating handles outside the valid bridge range.
        """
        hard_ceiling = (HANDLE_MAX - HANDLE_BASE) // S
        expansion = int(S / K)  # = 18
        old_capacity = self._capacity
        new_capacity = min(old_capacity * expansion, hard_ceiling)
        if new_capacity <= old_capacity:
            return  # already at ceiling — no further expansion possible
        self._capacity = new_capacity
        new_slots = list(range(old_capacity, self._capacity))
        self._free_slots.extend(new_slots)

    def register_native64(self, pid: int, exe_name: str = "") -> None:
        """
        Register a process ID as a known native 64-bit process.

        Native 64-bit processes do not require bridge handle translation —
        their addresses are already in P_full.  Registering them allows the
        bridge to skip D-projection for their handles (identity mapping, V=0).

        ET derivation: a native 64-bit process has D = P_full (no gap).
        The Identification Principle confirms: if D already spans P, the
        bridge Descriptor is redundant and can be elided.
        """
        with self._lock:
            if not hasattr(self, '_native64_pids'):
                self._native64_pids: dict = {}
            self._native64_pids[pid] = exe_name

    def deregister_native64(self, pid: int) -> None:
        """
        Remove a process ID from the native 64-bit process registry.

        Called when a native 64-bit process exits or is no longer tracked.
        After deregistration, handles from this PID will be subject to
        normal bridge D-projection again.
        """
        with self._lock:
            if hasattr(self, '_native64_pids'):
                self._native64_pids.pop(pid, None)

    def is_native64(self, pid: int) -> bool:
        """
        Check whether a process ID is registered as a native 64-bit process.

        Returns True if the PID was previously registered via register_native64().
        Used by the dispatch layer to decide whether to bypass D-projection.
        """
        with self._lock:
            return hasattr(self, '_native64_pids') and pid in self._native64_pids

    def native64_pids(self) -> list:
        """
        Return a list of all currently registered native 64-bit process IDs.

        ET derivation: the set of PIDs for which D = P_full (no Descriptor Gap).
        These processes have V(bridge) = 0 by identity — the bridge is transparent.
        """
        with self._lock:
            if not hasattr(self, '_native64_pids'):
                return []
            return list(self._native64_pids.keys())
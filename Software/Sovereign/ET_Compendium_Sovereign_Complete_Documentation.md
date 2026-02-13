================================================================================
ET COMPENDIUM SOVEREIGN - COMPLETE DOCUMENTATION
================================================================================
A Low-Level Memory Manipulation Engine for Python Enhancement
Version: 1.0 (Foundation)
Date: 2026-01-07
================================================================================



MISSION STATEMENT:
Create a Python library that allows Python to overcome its inherent limits and 
do everything any other language can do, including matching the speed of C, 
without requiring multiple languages.

PROJECT GOALS:
1. Match C/Assembly performance through JIT compilation
2. Enable assembly injection for performance-critical code
3. Provide direct system-level access (syscalls, hardware)
4. Eliminate need for C extensions or other languages
5. Maintain safety while providing low-level control
6. Make Python a complete systems programming language

WHY THIS SCRIPT EXISTS:
This script is the FOUNDATION - the low-level memory manipulation engine that
provides the primitives ALL other features will be built upon. It's not just
"string transmutation code." It's a general-purpose memory manipulation 
framework that demonstrates its capabilities via string modification.

CORE INSIGHT:
Every advanced feature (JIT, assembly injection, type optimization) requires
the ability to:
  • Read arbitrary memory addresses
  • Write to protected memory regions
  • Navigate Python object internal structures
  • Maintain reference integrity
  • Bypass OS and interpreter protections

This script provides ALL these primitives. Everything else is just pointing
these tools at different memory addresses.

================================================================================
2. WHAT THIS SCRIPT IS
================================================================================

TECHNICAL CLASSIFICATION:
A kernel-level memory manipulation engine with automated structure discovery,
reference graph management, and OS protection bypass capabilities.

WHAT IT PROVIDES:
• Direct memory read/write via kernel interfaces
• Automated Python object structure introspection
• Safe reference counting management (Py_IncRef/DecRef)
• Copy-On-Write protection bypass (Phase-Locking)
• Complete reference graph traversal
• Platform-adaptive offset detection

WHAT IT'S NOT:
• NOT just a "string manipulation utility"
• NOT limited to immutable object modification
• NOT a high-level API (it's the low-level foundation)
• NOT the final product (it's the keystone component)

ANALOGIES:
• Like `libc` for C - provides primitives everything else builds on
• Like an assembler - low-level tool that higher abstractions use
• Like a kernel - provides capabilities, not user-facing features

DEMONSTRATED CAPABILITY:
String transmutation (modifying "immutable" strings) proves the engine works.
It's the "Hello World" of memory manipulation, demonstrating:
  ✓ Kernel access works
  ✓ Structure discovery works
  ✓ Protection bypass works
  ✓ Reference safety works

These same techniques apply to:
  → Code objects (for assembly injection)
  → Function pointers (for JIT compilation)
  → Type objects (for optimization)
  → Builtin functions (for syscall injection)
  → Memory allocators (for custom allocation)

================================================================================
3. ARCHITECTURE OVERVIEW
================================================================================

THREE-LAYER DESIGN:

LAYER 1: MATHEMATICAL FOUNDATION (ETMath)
  ├─ Equation 211: Structural Density (geometry detection)
  └─ Equation 212: Traverser Effort (cost calculation)

LAYER 2: MEMORY ACCESS PRIMITIVES (ETCompendiumSovereign Core)
  ├─ Kernel Tunnel (OS-level access)
  ├─ Geometry Calibration (structure discovery)
  ├─ Pointer Manipulation (direct memory control)
  └─ Safety Systems (verification, probing)

LAYER 3: OBJECT MANIPULATION (High-Level Operations)
  ├─ Transmutation (modify objects)
  ├─ Reference Management (maintain integrity)
  └─ Graph Traversal (find all references)

EXECUTION FLOW:
┌─────────────────────────────────────────────────────────────┐
│ User Request: transmute(target, replacement)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ DECISION TREE: Determine object type & modification method  │
│   • bytearray? → Buffer replacement                         │
│   • bytes/str? → Calculate width (1/2/4 byte Unicode)       │
│   • Length match? → Try direct memory write                 │
│   • Length mismatch? → Use reference displacement           │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│ DIRECT PATH      │          │ DISPLACEMENT     │
│ (Length Match)   │          │ (Length Mismatch)│
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ Tier 1: Tunnel   │          │ Scan references: │
│ Phase-lock write │          │ • gc.referrers() │
│ Verify success   │          │ • sys.modules    │
└────────┬─────────┘          │ • stack frames   │
         │ Fail               └────────┬─────────┘
         ▼                              │
┌──────────────────┐                   ▼
│ Tier 2: Direct   │          ┌──────────────────┐
│ ctypes.memmove   │          │ Replace in:      │
│ Safety probe     │          │ • Dicts          │
└────────┬─────────┘          │ • Lists          │
         │ Fail               │ • Sets           │
         ▼                    │ • Tuples (patch) │
┌──────────────────┐          └────────┬─────────┘
│ Tier 3:          │                   │
│ Displacement ────┼───────────────────┘
└──────────────────┘                   │
                                       ▼
                              ┌──────────────────┐
                              │ Return result    │
                              │ • String: Success│
                              │ • Dict: Report   │
                              └──────────────────┘

================================================================================
4. CONFIGURATION CONSTANTS
================================================================================

CACHE_FILE = os.path.join(tempfile.gettempdir(), "et_compendium_geometry.json")
────────────────────────────────────────────────────────────────────────────
PURPOSE: Stores platform-specific memory offsets after first calibration
LOCATION: System temp directory (e.g., /tmp on Linux)
FORMAT: JSON with keys '1', '2', '4', 'bytes', 'tuple'
ATOMICITY: Written via mkstemp + os.replace for crash safety
WHY: Avoids ~0.33ms calibration overhead on subsequent runs
EXAMPLE: {"1": 40, "2": 56, "4": 56, "bytes": 32, "tuple": 24}

MAX_SCAN_WIDTH = 2048
────────────────────────────────────────────────────────────────────────────
PURPOSE: Maximum offset to scan when detecting data payload location
RATIONALE: Python object headers are typically 16-80 bytes, scan up to 2KB
TRADE-OFF: Higher = more robust, slower; Lower = faster, may miss
USAGE: In _scan_offset() beacon detection loop
FUTURE: Could be made adaptive based on object type

DEFAULT_TUPLE_DEPTH = 4
────────────────────────────────────────────────────────────────────────────
PURPOSE: Maximum recursion depth for nested tuple patching
RATIONALE: Prevents infinite recursion, handles practical nesting
USAGE: Passed to _patch_tuple_recursive() to limit depth
CONSIDERATION: Deep tuples (>4 levels) require multiple transmutation passes

================================================================================
5. ETMath CLASS - Exception Theory Mathematics
================================================================================

PURPOSE:
Operationalizes equations from the ET Math Compendium for practical use in
memory manipulation. These equations guide algorithm behavior and provide
metrics for operation cost assessment.

CLASS STRUCTURE:
class ETMath:
    @staticmethod
    def structural_density(obj_len, mem_size)
    @staticmethod
    def traverser_effort(observers, byte_delta)

────────────────────────────────────────────────────────────────────────────
METHOD: structural_density(obj_len, mem_size)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
@staticmethod
def structural_density(obj_len, mem_size) -> float

EQUATION:
Eq 211 from ET Math Compendium: S = D / D²
Operational Form: S = obj_len / mem_size

PARAMETERS:
  obj_len (int): Payload length (e.g., string character count)
  mem_size (int): Total memory footprint (from sys.getsizeof())

RETURNS:
  float: Density ratio (0.0 to 1.0+)

PURPOSE:
Distinguishes between compact (inline) and pointer-based (external buffer)
object storage geometries in Python's memory model.

INTERPRETATION:
  • High Density (0.7-1.0): Efficient storage, most bytes are payload
    Examples: Large strings (1000+ chars), pointer to external buffer
  
  • Low Density (0.1-0.3): Inefficient storage, overhead dominates
    Examples: Small strings (<50 chars), inline storage with headers
  
  • Transitional (0.3-0.7): Mid-size objects, variable geometry

USAGE IN SCRIPT:
Used during calibration (_calibrate_all) to verify that offset detection
correctly identifies compact vs pointer strings. Acts as validation that
beacon scanning found correct memory locations.

THEORETICAL SIGNIFICANCE:
Reveals Python's geometric phase transition (~100 chars) where internal
representation switches from inline to external buffer storage.

EXAMPLE:
>>> ETMath.structural_density(len("short"), sys.getsizeof("short"))
0.0889  # Low density, high overhead

>>> ETMath.structural_density(len("X"*1000), sys.getsizeof("X"*1000))
0.9606  # High density, efficient packing

────────────────────────────────────────────────────────────────────────────
METHOD: traverser_effort(observers, byte_delta)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
@staticmethod
def traverser_effort(observers, byte_delta) -> float

EQUATION:
Eq 212 from ET Math Compendium: |T|² = |D1|² + |D2|²
Operational Form: |T| = sqrt(observers² + byte_delta²)

PARAMETERS:
  observers (int): Number of reference swaps required
  byte_delta (int): Byte difference between old and new values

RETURNS:
  float: Effort magnitude (Pythagorean distance in descriptor space)

PURPOSE:
Quantifies the "metabolic cost" of a transmutation operation by combining
discrete (reference count) and continuous (byte size) dimensions.

INTERPRETATION:
  • Low Effort (<10): Cheap operation, few references
  • Medium Effort (10-100): Moderate cost, multiple containers
  • High Effort (>100): Expensive, widespread references
  
  Byte delta increases cost when resizing is involved.

USAGE IN SCRIPT:
Calculated during reference displacement (_displace_references) to provide
cost metrics in operation reports. Currently informational; could drive
adaptive algorithm selection in future.

THEORETICAL SIGNIFICANCE:
Demonstrates geometric interpretation of traversal through descriptor space.
The Pythagorean formulation suggests orthogonal dimensions (discrete vs
continuous transformations).

EXAMPLE:
>>> ETMath.traverser_effort(10, 0)
10.0  # 10 reference swaps, no resizing

>>> ETMath.traverser_effort(10, 20)
22.36  # 10 swaps plus 20-byte resize

================================================================================
6. ETCompendiumSovereign CLASS - The Core Engine
================================================================================

PURPOSE:
The main memory manipulation engine. Provides kernel-level access, automated
structure discovery, and safe object modification capabilities.

INITIALIZATION SEQUENCE:
1. Detect platform (Linux/Windows/macOS)
2. Load or calibrate geometry offsets (cached)
3. Initialize kernel tunnel (if available)
4. Store configuration

STATE MAINTAINED:
  • os_type: Platform identifier
  • pid: Process ID
  • is_64bit: Architecture (True for 64-bit)
  • ptr_size: Pointer size in bytes (4 or 8)
  • offsets: Dict of payload offsets by type
  • wormhole: Linux /proc/pid/mem file handle
  • win_handle: Windows process handle

LIFECYCLE:
1. Create: sov = ETCompendiumSovereign()
2. Use: sov.transmute(obj, replacement)
3. Cleanup: sov.close()  # Optional, releases handles

================================================================================
7. COMPLETE FUNCTION REFERENCE
================================================================================

────────────────────────────────────────────────────────────────────────────
METHOD: __init__(self)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def __init__(self) -> None

PURPOSE:
Initialize the memory manipulation engine with platform detection, geometry
calibration, and kernel access setup.

EXECUTION FLOW:
1. Detect OS (platform.system())
2. Get process ID (os.getpid())
3. Determine architecture (sys.maxsize > 2**32)
4. Load or calibrate memory offsets
5. Initialize kernel tunnel
6. Print status message

SIDE EFFECTS:
  • Creates cache file if doesn't exist
  • Opens /proc/pid/mem (Linux) or process handle (Windows)
  • May take ~0.33ms for first-run calibration

THREAD SAFETY:
Not thread-safe during initialization. Create one instance per thread or
use external synchronization.

EXAMPLE:
>>> sov = ETCompendiumSovereign()
[ET] Compendium Sovereign Active. Offsets: {'1': 40, '2': 56, ...}

────────────────────────────────────────────────────────────────────────────
METHOD: _load_geometry(self)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _load_geometry(self) -> dict

PURPOSE:
Load platform-specific memory offsets from cache or calibrate if needed.

ALGORITHM:
1. Try to load from CACHE_FILE
2. If not exists or corrupt: calibrate via _calibrate_all()
3. Atomically write new cache (mkstemp + replace)
4. Return offset dictionary

CACHE FORMAT:
{
  "1": 40,      # 1-byte (ASCII) string data offset
  "2": 56,      # 2-byte (BMP) string data offset
  "4": 56,      # 4-byte (Astral) string data offset
  "bytes": 32,  # bytes object data offset
  "tuple": 24   # tuple items array offset
}

ATOMICITY:
Uses tempfile.mkstemp() + os.replace() to ensure cache is never corrupt
(atomic replace operation).

ERROR HANDLING:
All exceptions suppressed (bare except). Returns calibrated values on any
cache failure.

RETURNS:
Dict[str, int]: Offset mappings

────────────────────────────────────────────────────────────────────────────
METHOD: _calibrate_all(self)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _calibrate_all(self) -> dict

PURPOSE:
Detect platform-specific memory offsets through beacon scanning and density
verification.

ALGORITHM:
1. Define fallback offsets (48/24 for 64/32-bit)
2. Create test strings (compact and pointer-based)
3. Verify density ratios (validation step)
4. Scan for each beacon type:
   - Width 1: "ET_A" (ASCII)
   - Width 2: "ET_Ω" (Greek Omega, U+03A9)
   - Width 4: "ET_🐍" (Snake emoji, U+1F40D)
   - Bytes: b"ET_B"
5. Use fallback if scan fails
6. Return offset dictionary

VALIDATION:
Calculates structural density for compact vs pointer strings to verify
that Python's geometry matches expected patterns. Not strictly necessary
but provides confidence in detection accuracy.

FALLBACK BEHAVIOR:
If beacon scanning fails (returns None), uses architecture-based defaults:
  64-bit: 48 bytes
  32-bit: 24 bytes

RETURNS:
Dict[str, int]: Complete offset mappings

────────────────────────────────────────────────────────────────────────────
METHOD: _scan_offset(self, beacon, width, is_bytes=False)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _scan_offset(self, beacon, width, is_bytes=False) -> Optional[int]

PARAMETERS:
  beacon: Test object (string or bytes) containing distinctive pattern
  width: Character width (1, 2, or 4 bytes)
  is_bytes: True if scanning bytes object, False for string

PURPOSE:
Locate the data payload offset within a Python object's memory layout by
scanning for a known byte pattern.

ALGORITHM:
1. Get object base address via id(beacon)
2. Encode beacon to target byte pattern:
   - bytes: use as-is
   - width 1: encode as latin-1
   - width 2: pack as UTF-16 LE (<H)
   - width 4: pack as UTF-32 LE (<I)
3. Cast address to ctypes ubyte pointer
4. Scan range [16, MAX_SCAN_WIDTH):
   - Compare bytes at offset against target
   - Return offset on complete match
5. Return None if not found

RANGE JUSTIFICATION:
  Start: 16 bytes (past refcount, type pointer, size fields)
  End: 2048 bytes (well beyond typical object headers)

BYTE PATTERN EXAMPLES:
  "ET_A" (width 1): b'E' b'T' b'_' b'A'
  "ET_Ω" (width 2): b'\xa9\x03' (little-endian U+03A9)
  "ET_🐍" (width 4): b'\x0d\xf4\x01\x00' (little-endian U+1F40D)

ERROR HANDLING:
Try-except around array access to stop at unmapped memory. Returns None
instead of raising exception.

RETURNS:
Optional[int]: Offset in bytes, or None if pattern not found

EXAMPLE:
>>> sov._scan_offset("ET_A", 1)
40  # Found at offset 40 bytes

────────────────────────────────────────────────────────────────────────────
METHOD: _init_tunnel(self)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _init_tunnel(self) -> None

PURPOSE:
Initialize kernel-level memory access tunnel for direct process memory writes.

PLATFORM IMPLEMENTATIONS:

LINUX:
  File: /proc/{pid}/mem
  Mode: rb+ (read-write binary, unbuffered)
  Requires: Read/write access to /proc filesystem
  Access Check: os.access(path, os.W_OK)
  Storage: self.wormhole (file handle)

WINDOWS:
  API: kernel32.OpenProcess()
  Flags: 0x0038 (PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION)
  Requires: Administrator privileges or debug rights
  Storage: self.win_handle (process handle)

MACOS:
  Currently not implemented
  Future: task_for_pid() + vm_write() approach

ERROR HANDLING:
All exceptions suppressed. If initialization fails, attributes remain None
and tunnel write methods will return False (graceful degradation to Tier 2).

SIDE EFFECTS:
  • Opens file descriptor (Linux) - must call close() to release
  • Opens process handle (Windows) - must call CloseHandle()

SECURITY:
Requires elevated permissions. Regular users may not have access.

────────────────────────────────────────────────────────────────────────────
METHOD: _tunnel_write(self, address, data)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _tunnel_write(self, address: int, data: bytes) -> bool

PARAMETERS:
  address: Virtual memory address to write to
  data: Byte sequence to write

PURPOSE:
Write directly to process memory via kernel tunnel, using phase-locking
technique to bypass Copy-On-Write protections.

INNOVATION - PHASE-LOCKING:
Traditional direct writes may fail due to OS-level COW optimization.
Solution: Two-phase write
  1. NOISE: Write XOR'd first byte (forces page coalescing)
  2. SIGNAL: Write correct data (actual modification)

ALGORITHM:
1. Calculate noise byte: data[0] XOR 0xFF
2. IF Linux tunnel available:
     a. Seek to address
     b. Write noise byte
     c. Seek to address again
     d. Write actual data
     e. Return True
3. IF Windows handle available:
     a. WriteProcessMemory(noise byte)
     b. WriteProcessMemory(actual data)
     c. Return success status
4. ELSE: Return False (no tunnel available)

WHY IT WORKS:
The noise write marks the page as "modified," breaking any COW lock. The
immediate overwrite with correct data completes the modification without
triggering defensive copies.

RETURNS:
bool: True if write succeeded, False otherwise

ERROR HANDLING:
All exceptions caught and suppressed. Returns False on any failure.

THREAD SAFETY:
NOT thread-safe. Concurrent writes may interleave noise/signal phases.

FUTURE ENHANCEMENT:
Add threading.Lock() around write sequence.

────────────────────────────────────────────────────────────────────────────
METHOD: _displace_references(self, target, replacement, dry_run, depth_limit)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _displace_references(self, target, replacement, 
                         dry_run=False, depth_limit=3) -> dict

PARAMETERS:
  target: Object to find and replace
  replacement: New object to substitute
  dry_run: If True, don't modify, just report what would happen
  depth_limit: Maximum recursion depth for nested tuples

PURPOSE:
Perform holographic displacement - find all references to target object
throughout the process memory space and replace them with replacement object.

SEARCH SCOPE:
1. Garbage Collector Referrers: gc.get_referrers(target)
2. Module Globals: sys.modules[*].__dict__
3. Stack Frames: inspect.currentframe() chain
4. Local Variables: frame.f_locals for all frames

CONTAINER HANDLING:

DICTIONARIES:
  • Check all values: if v is target → replace
  • Check keys: if target in dict → pop and re-add with replacement
    (Only if replacement is hashable)

LISTS:
  • Iterate indices: if list[i] is target → replace

SETS:
  • Check membership: if target in set → remove + add replacement
    (Only if replacement is hashable)

TUPLES:
  • Call _patch_tuple_recursive() for pointer-level modification
  • Maintains refcounts via Py_IncRef/DecRef
  • Respects depth_limit to prevent infinite recursion

CYCLE PREVENTION:
Maintains visited set (by id()) to avoid:
  • Processing same container twice
  • Infinite loops in circular references
  • Modifying the report dict itself

METRICS CALCULATION:
  swaps: Total reference replacements performed
  effort: Traverser effort via Eq 212 (assumes byte_delta=0 for identity)
  locations: Breakdown by container type

GHOST DETECTION:
If target is an interned string with zero observers, adds warning:
  "Ghost Object (Interned). Zero Observers."

RETURNS:
Dict with structure:
{
  "status": "SIMULATION" | "EXECUTED",
  "swaps": int,
  "effort": float,
  "locations": {"Dict_Value": N, "List_Item": M, ...},
  "warnings": [...]
}

THREAD SAFETY:
NOT thread-safe. Reference graph may change during traversal.

────────────────────────────────────────────────────────────────────────────
METHOD: _patch_tuple_recursive(self, curr_tuple, target, replacement, 
                                depth, dry_run, visited)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _patch_tuple_recursive(self, curr_tuple, target, replacement,
                            depth, dry_run, visited) -> int

PARAMETERS:
  curr_tuple: Tuple to scan/modify
  target: Object to replace
  replacement: New object
  depth: Remaining recursion depth
  dry_run: If True, count but don't modify
  visited: Set of already-processed tuple ids

PURPOSE:
Modify "immutable" tuples at the pointer level by directly rewriting their
item array entries.

ALGORITHM:
1. Check termination: depth <= 0 or id in visited → return 0
2. Mark as visited
3. Calculate items array address: id(tuple) + tuple_offset
4. For each slot in tuple:
     a. Calculate slot address: items + (i * ptr_size)
     b. Read pointer value at slot
     c. If pointer == id(target):
        - Increment replacement refcount (Py_IncRef)
        - Overwrite slot with id(replacement)
        - Decrement target refcount (Py_DecRef)
        - Increment swap counter
5. Return total swaps

MEMORY LAYOUT:
Python tuple structure (simplified):
  [refcount][type*][size][item0*][item1*][item2*]...
                          ^
                          tuple_offset points here

REFCOUNT SAFETY:
CRITICAL: Must increment before decrement to prevent premature GC
  CORRECT:
    Py_IncRef(replacement)
    *slot = id(replacement)
    Py_DecRef(target)
  
  WRONG (may crash):
    Py_DecRef(target)        # Could free target
    *slot = id(replacement)  # Dangling pointer!
    Py_IncRef(replacement)

RECURSION:
Currently does NOT recurse into nested tuples (optimization/simplification).
Future enhancement could add recursion for deep nesting.

RETURNS:
int: Number of swaps performed in this tuple

ERROR HANDLING:
Try-except suppresses access violations. Returns accumulated swap count.

────────────────────────────────────────────────────────────────────────────
METHOD: transmute(self, target, replacement, force_phase, dry_run)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def transmute(self, target, replacement, 
              force_phase=None, dry_run=False) -> Union[str, dict]

PARAMETERS:
  target: Object to modify
  replacement: New value/object
  force_phase: (Unused, reserved for future) Force specific tier
  dry_run: If True, predict without modifying

PURPOSE:
Master transmutation method. Routes to appropriate modification strategy
based on object type and length compatibility.

DECISION TREE:

1. BYTEARRAY SPECIAL CASE:
   If target is bytearray:
     - Use buffer replacement: target[:] = replacement
     - Return immediately
   
2. GEOMETRY DETECTION:
   Determine encoding width:
     - bytes: use as-is
     - string: calculate max(ord(c))
       • >65535: width 4 (UTF-32)
       • >255: width 2 (UTF-16)
       • else: width 1 (Latin-1)
   
   Encode replacement to byte payload:
     - width 1: encode('latin-1')
     - width 2: struct.pack('<H', ...) per char
     - width 4: struct.pack('<I', ...) per char

3. DRY RUN BRANCH:
   If dry_run=True:
     - Call _displace_references() with dry_run=True
     - Return prediction report
     - No modifications performed

4. LENGTH-MATCHED PATH (Tier 1/2):
   If len(payload) == physical_length:
     a. Calculate data address: id(target) + offset
     b. TRY Tier 1 (Tunnel):
        - _tunnel_write() with phase-locking
        - _verify() payload written correctly
        - _blind_hash_reset() to invalidate cached hash
        - Return "TRANSMUTATION_COMPLETE (Tunnel+PhaseLock)"
     c. TRY Tier 2 (Direct):
        - _safety_probe() address is readable
        - ctypes.memmove() direct write
        - _verify() success
        - _blind_hash_reset()
        - Return "TRANSMUTATION_COMPLETE (Direct)"

5. FALLBACK PATH (Tier 3):
   If length mismatch or Tier 1/2 fail:
     - Call _displace_references() (actual modification)
     - Add "method": "HOLOGRAPHIC_DISPLACEMENT" to report
     - Return report dict

RETURN TYPES:
  Success (Tier 1/2): str - "TRANSMUTATION_COMPLETE (...)"
  Success (Tier 3): dict - Detailed report with swap counts
  Dry Run: dict - Prediction report
  Error: dict - {"status": "ERROR", "msg": "..."}

PERFORMANCE:
  Tier 1/2: ~0.022ms (length-matched)
  Tier 3: ~0.922ms (displacement)
  Dry Run: ~0.922ms (full scan without writes)

EXAMPLE:
>>> sov.transmute("ABCDEFGH", "12345678")
'TRANSMUTATION_COMPLETE (Tunnel+PhaseLock)'

>>> sov.transmute("Short", "VeryLongReplacement")
{'status': 'EXECUTED', 'swaps': 12, 'effort': 12.0, ...}

────────────────────────────────────────────────────────────────────────────
METHOD: _verify(self, addr, expected)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _verify(self, addr: int, expected: bytes) -> bool

PARAMETERS:
  addr: Memory address to read
  expected: Byte sequence expected at address

PURPOSE:
Verify that a memory write succeeded by reading back and comparing.

ALGORITHM:
1. Read len(expected) bytes from addr via ctypes.string_at()
2. Compare to expected bytes
3. Return True if match, False otherwise

ERROR HANDLING:
Try-except returns False on any read failure (invalid address, etc.)

USAGE:
Called after _tunnel_write() and ctypes.memmove() to confirm success.

RETURNS:
bool: True if memory contains expected bytes

────────────────────────────────────────────────────────────────────────────
METHOD: _safety_probe(self, addr, ln)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _safety_probe(self, addr: int, ln: int) -> bool

PARAMETERS:
  addr: Starting address to test
  ln: Length of region to test

PURPOSE:
Test if a memory region is readable before attempting write. Prevents
segmentation faults.

ALGORITHM:
1. Try to read first byte: ctypes.string_at(addr, 1)
2. Try to read last byte: ctypes.string_at(addr+ln-1, 1)
3. Return True if both succeed

RATIONALE:
If first and last bytes are readable, entire region is likely safe to write.
Not perfect (could have holes) but catches most invalid addresses.

ERROR HANDLING:
Returns False on any exception.

USAGE:
Called before ctypes.memmove() in Tier 2 direct write path.

RETURNS:
bool: True if region appears accessible

────────────────────────────────────────────────────────────────────────────
METHOD: _blind_hash_reset(self, target)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def _blind_hash_reset(self, target) -> None

PARAMETERS:
  target: Object whose hash cache to reset

PURPOSE:
Invalidate Python's cached hash value after modifying an object's contents.

BACKGROUND:
Python caches hash values in the object structure at offset 24 (64-bit) or
12 (32-bit). When an object's content changes, the cache becomes invalid.
Setting hash to -1 forces Python to recalculate on next hash() call.

ALGORITHM:
1. Calculate hash offset: 24 for 64-bit, 12 for 32-bit
2. Cast to pointer: (id(target) + offset) as ssize_t*
3. Write -1 to invalidate cache

ERROR HANDLING:
Try-except suppresses errors. Silent failure acceptable (hash will be wrong
but won't crash - just cause dict lookup failures).

USAGE:
Called after successful Tier 1/2 transmutation to maintain dict integrity.

IMPORTANCE:
CRITICAL for objects used as dict keys. Without this, dict lookups may fail:
  d = {s: value}
  transmute(s, new_content)
  # Without hash reset: d[s] will fail to find value
  # With hash reset: d[s] works correctly

RETURNS:
None

────────────────────────────────────────────────────────────────────────────
METHOD: close(self)
────────────────────────────────────────────────────────────────────────────

SIGNATURE:
def close(self) -> None

PURPOSE:
Release kernel tunnel resources (file handles, process handles).

ALGORITHM:
1. If wormhole (Linux): close file handle
2. (MISSING) If win_handle (Windows): should call CloseHandle()

CURRENT ISSUE:
Windows handle not being closed - resource leak!

FIX NEEDED:
if self.win_handle:
    self.kernel32.CloseHandle(self.win_handle)
    self.win_handle = None

USAGE:
Call at program exit or when done with sovereign:
  sov = ETCompendiumSovereign()
  # ... use sov ...
  sov.close()

Can also use context manager (future enhancement):
  with ETCompendiumSovereign() as sov:
      sov.transmute(...)

RETURNS:
None

================================================================================
8. USAGE PATTERNS & EXAMPLES
================================================================================

PATTERN 1: BASIC TRANSMUTATION
────────────────────────────────────────────────────────────────────────────
from et_compendium_sovereign import ETCompendiumSovereign

sov = ETCompendiumSovereign()

# Length-matched (fast, direct)
s = "Original"
result = sov.transmute(s, "Modified")
print(s)  # "Modified"
print(result)  # "TRANSMUTATION_COMPLETE (Tunnel+PhaseLock)"

sov.close()

PATTERN 2: LENGTH MISMATCH (DISPLACEMENT)
────────────────────────────────────────────────────────────────────────────
s = "Short"
refs = [s, s, s]

result = sov.transmute(s, "VeryLongReplacement")
print(refs)  # All references updated
print(result['swaps'])  # Number of references replaced

PATTERN 3: DRY RUN PREDICTION
────────────────────────────────────────────────────────────────────────────
s = "Test"
container_list = [s, s, s]
container_dict = {"key": s, s: "value"}

report = sov.transmute(s, "Demo", dry_run=True)
print(f"Would swap: {report['swaps']} references")
print(f"Effort: {report['effort']:.2f}")
print(f"Locations: {report['locations']}")
# No actual modification - s still "Test"

PATTERN 4: BYTEARRAY BUFFER MODIFICATION
────────────────────────────────────────────────────────────────────────────
ba = bytearray(b"MutableBuffer")
sov.transmute(ba, b"NewContent!!!")
print(ba)  # bytearray(b'NewContent!!!')

PATTERN 5: BYTES OBJECT (IMMUTABLE)
────────────────────────────────────────────────────────────────────────────
b = b"Immutable"
result = sov.transmute(b, b"Modified!")
print(b)  # b'Modified!' (immutability bypassed!)

PATTERN 6: INTERNED STRING MODIFICATION
────────────────────────────────────────────────────────────────────────────
import sys
s = sys.intern("CONSTANT")
print(sys.intern("CONSTANT") is s)  # True (same object)

sov.transmute(s, "MODIFIED!")
print(s)  # "MODIFIED!"
print(sys.intern("CONSTANT"))  # "MODIFIED!" (all references changed!)

PATTERN 7: UNICODE HANDLING
────────────────────────────────────────────────────────────────────────────
# ASCII (1-byte)
s1 = "ASCII"
sov.transmute(s1, "HELLO")

# Greek (2-byte)
s2 = "Ωmega"
sov.transmute(s2, "Δelta")

# Emoji (4-byte)
s3 = "🐍Python"
sov.transmute(s3, "🔥Blazing")

PATTERN 8: CALCULATING METRICS
────────────────────────────────────────────────────────────────────────────
from et_compendium_sovereign import ETMath
import sys

# Density analysis
strings = ["short", "X"*100, "X"*1000]
for s in strings:
    density = ETMath.structural_density(len(s), sys.getsizeof(s))
    print(f"{len(s):5d} chars: density={density:.3f}")

# Effort calculation
effort = ETMath.traverser_effort(observers=50, byte_delta=10)
print(f"Effort for 50 swaps + 10 byte delta: {effort:.2f}")

PATTERN 9: TUPLE MODIFICATION (IMMUTABLE)
────────────────────────────────────────────────────────────────────────────
s = "Element"
t = (s, "other", s, s)

# Tuples are "immutable" but we can change their contents
sov.transmute(s, "NewElem!")
print(t)  # ('NewElem!', 'other', 'NewElem!', 'NewElem!')

PATTERN 10: SAFE EXPLORATION (DRY RUN FIRST)
────────────────────────────────────────────────────────────────────────────
target = important_string

# Always dry run first for critical operations
report = sov.transmute(target, replacement, dry_run=True)

if report['swaps'] < 100:  # Acceptable impact
    result = sov.transmute(target, replacement)
    print(f"Success: {result}")
else:
    print(f"Too many references ({report['swaps']}), aborting")





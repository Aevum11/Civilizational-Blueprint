"""
et_bridge/et_injector.py
ET32 Bridge — Injection Engine

Injects the ET bridge hook stubs into 32-bit target processes.
Derived from P ∘ D ∘ T = E.

The injection process:
  P = the 32-bit process's address space (target P-substrate)
  D = the hook stubs and IAT patches (new Descriptors being installed)
  T = CreateRemoteThread / QueueUserAPC (the Traverser performing injection)
  E = the injection is complete, bridge engaged

Injection modes:
  "iat"      — Import Address Table patching (safest, most reliable)
  "debug"    — Windows Debug API (WaitForDebugEvent loop)
  "shellcode" — Direct shellcode injection (fastest, most invasive)

For "iat" mode (default):
  1. Open target process with PROCESS_ALL_ACCESS
  2. Locate the IAT of the target's main module
  3. Patch entries for bridged APIs to point to our stubs
  4. Each stub communicates with the 64-bit broker via named pipe

For "shellcode" mode:
  1. Allocate a code cave in the target
  2. Write 32-bit shellcode stubs for each hooked function
  3. Use CreateRemoteThread to run an init stub that:
     a. Connects to the broker via named pipe
     b. Sets up the hook table in the target

IAT patching is the preferred method (ET d=3 — structural identity with
the compiler pipeline: both patch addresses at link/load time).
"""

import ctypes
import ctypes.wintypes
import struct
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from et_math import (
    S, K, V_BASE, DIGITAL_ACTION_QUANTUM, IPC_BUFFER_SIZE,
    CONN_TIMEOUT_MS,
    CmdFamily, CmdCode, ETLatticeRouter
)
from et_config import TargetConfig
from et_logger import ETLog
from et_heaven import ETHeavenGate, MEM_RESERVE, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE

# Windows constants
PROCESS_ALL_ACCESS       = 0x1F0FFF
PROCESS_VM_OPERATION     = 0x0008
PROCESS_VM_READ          = 0x0010
PROCESS_VM_WRITE         = 0x0020
PROCESS_CREATE_THREAD    = 0x0002
PROCESS_QUERY_INFORMATION = 0x0400
PAGE_EXECUTE_READ        = 0x20

# Toolhelp32 snapshot flags (Windows constants)
TH32CS_SNAPMODULE        = 0x00000008
TH32CS_SNAPMODULE32      = 0x00000010  # for 32-bit modules in WOW64

IMAGE_DOS_SIGNATURE      = 0x5A4D  # "MZ"
IMAGE_NT_SIGNATURE       = 0x00004550  # "PE\0\0"
IMAGE_FILE_32BIT_MACHINE = 0x0100

kernel32 = getattr(ctypes.windll, 'kernel32')

# =============================================================================
# PE PARSING UTILITIES
# =============================================================================

def _read_process_memory(h_process, addr: int, size: int) -> Optional[bytes]:
    """Read 'size' bytes from 'addr' in the target process."""
    buf = ctypes.create_string_buffer(size)
    read = ctypes.c_size_t(0)
    ok = getattr(kernel32, 'ReadProcessMemory')(
        h_process,
        ctypes.c_void_p(addr),
        buf,
        size,
        ctypes.byref(read)
    )
    if not ok or read.value == 0:
        return None
    return bytes(buf[:read.value])


def _write_process_memory(h_process, addr: int, data: bytes) -> bool:
    """Write bytes to 'addr' in the target process."""
    written = ctypes.c_size_t(0)
    old_protect = ctypes.wintypes.DWORD(0)

    # Make writable
    getattr(kernel32, 'VirtualProtectEx')(
        h_process, ctypes.c_void_p(addr), len(data),
        PAGE_EXECUTE_READWRITE,
        ctypes.byref(old_protect)
    )

    ok = getattr(kernel32, 'WriteProcessMemory')(
        h_process,
        ctypes.c_void_p(addr),
        ctypes.c_char_p(data),
        len(data),
        ctypes.byref(written)
    )

    # Restore protection
    getattr(kernel32, 'VirtualProtectEx')(
        h_process, ctypes.c_void_p(addr), len(data),
        old_protect,
        ctypes.byref(old_protect)
    )

    return bool(ok) and written.value == len(data)


def _get_module_base(h_process, module_name: str) -> int:
    """
    Find the base address of a module in a 32-bit WOW64 process.
    Uses CreateToolhelp32Snapshot + Module32First/Next.
    Returns 0 if not found.
    """
    snap_flags = TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32

    pid = ctypes.wintypes.DWORD(0)
    get_pid = getattr(kernel32, 'GetProcessId')
    get_pid.restype  = ctypes.wintypes.DWORD
    get_pid.argtypes = [ctypes.wintypes.HANDLE]
    pid.value = get_pid(h_process)
    pid_val = pid.value

    snap = getattr(kernel32, 'CreateToolhelp32Snapshot')(
        snap_flags, pid_val
    )
    if snap == ctypes.wintypes.HANDLE(-1).value:
        return 0

    class MODULEENTRY32(ctypes.Structure):
        """Win32 MODULEENTRY32 structure for Toolhelp32 module enumeration.

        ET mapping: Each module in the snapshot is a Descriptor (D) within the
        target process's substrate (P). The enumeration traverses (T) the full
        D-set of loaded modules to locate the bridge's injection target.
        """
        _fields_ = [
            ("dwSize",         ctypes.wintypes.DWORD),
            ("th32ModuleID",   ctypes.wintypes.DWORD),
            ("th32ProcessID",  ctypes.wintypes.DWORD),
            ("GlblcntUsage",   ctypes.wintypes.DWORD),
            ("ProccntUsage",   ctypes.wintypes.DWORD),
            ("modBaseAddr",    ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",    ctypes.wintypes.DWORD),
            ("hModule",        ctypes.wintypes.HMODULE),
            ("szModule",       ctypes.c_char * 256),
            ("szExePath",      ctypes.c_char * 260),
        ]

    entry = MODULEENTRY32()
    entry.dwSize = ctypes.sizeof(MODULEENTRY32)
    result = 0
    module_lower = module_name.lower().encode()

    try:
        if getattr(kernel32, 'Module32First')(snap, ctypes.byref(entry)):
            while True:
                name = entry.szModule.lower()
                if module_lower in name:
                    result = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                    break
                if not getattr(kernel32, 'Module32Next')(snap, ctypes.byref(entry)):
                    break
    finally:
        getattr(kernel32, 'CloseHandle')(snap)

    return result


def _parse_iat(h_process, module_base: int) -> Dict[str, int]:
    """
    Parse the Import Address Table of a module in the target process.
    Returns {function_name: iat_entry_address}.

    The IAT lives at the start of the .idata section, or is pointed to
    by the IMAGE_IMPORT_DESCRIPTOR array in the PE data directory.

    ET mapping: The IAT IS the D-set of the module's external dependencies.
    Patching an IAT entry = replacing a D with a new D.
    """
    result: Dict[str, int] = {}

    # Read DOS header
    dos_hdr = _read_process_memory(h_process, module_base, 64)
    if not dos_hdr or len(dos_hdr) < 64:
        return result

    magic, = struct.unpack_from("<H", dos_hdr, 0)
    if magic != IMAGE_DOS_SIGNATURE:
        return result

    e_lfanew, = struct.unpack_from("<I", dos_hdr, 60)

    # Read NT headers (PE32: 248 bytes)
    nt_hdr = _read_process_memory(h_process, module_base + e_lfanew, 256)
    if not nt_hdr or len(nt_hdr) < 120:
        return result

    sig, = struct.unpack_from("<I", nt_hdr, 0)
    if sig != IMAGE_NT_SIGNATURE:
        return result

    # Optional header starts at offset 24 (FILE_HEADER is 20 bytes, sig is 4)
    # For PE32: magic=0x10B, at offset 24
    opt_magic, = struct.unpack_from("<H", nt_hdr, 24)
    if opt_magic not in (0x10B, 0x20B):  # PE32 or PE32+
        return result

    # Import directory entry:
    # PE32:  data directory starts at offset 24+96 = 120 (index 1 = import = at 120+8=128)
    # PE32+: data directory starts at offset 24+112 = 136 (index 1 = at 144)
    if opt_magic == 0x10B:  # PE32
        import_rva_off  = 24 + 96 + 8   # OptHdr + 96 bytes + ImportDir offset
        import_size_off = import_rva_off + 4
    else:  # PE32+
        import_rva_off  = 24 + 112 + 8
        import_size_off = import_rva_off + 4

    if import_rva_off + 8 > len(nt_hdr):
        return result

    import_rva,  = struct.unpack_from("<I", nt_hdr, import_rva_off)
    import_size, = struct.unpack_from("<I", nt_hdr, import_size_off)

    if import_rva == 0 or import_size == 0:
        return result

    # Read IMAGE_IMPORT_DESCRIPTORs (each 20 bytes)
    descriptors_data = _read_process_memory(
        h_process, module_base + import_rva, import_size
    )
    if not descriptors_data:
        return result

    desc_size = 20  # sizeof(IMAGE_IMPORT_DESCRIPTOR)
    offset = 0
    while offset + desc_size <= len(descriptors_data):
        orig_thunk_rva, ts, fwd, name_rva, thunk_rva = struct.unpack_from(
            "<IIIII", descriptors_data, offset
        )
        offset += desc_size

        if name_rva == 0 and thunk_rva == 0:
            break  # null terminator

        if name_rva == 0 or thunk_rva == 0:
            continue

        # Read DLL name
        dll_name_data = _read_process_memory(h_process, module_base + name_rva, 256)
        if not dll_name_data:
            continue
        dll_name = dll_name_data.split(b"\x00")[0].decode("ascii", errors="replace").lower()
        if not dll_name:
            continue  # Skip import descriptor with empty DLL name

        # Walk the INT (Import Name Table) and IAT in parallel
        # Use original first thunk (INT) for names, thunk for IAT addresses
        int_rva = orig_thunk_rva if orig_thunk_rva else thunk_rva
        ptr_size = 4  # PE32 uses 4-byte pointers

        iat_entry_addr = module_base + thunk_rva
        int_entry_addr = module_base + int_rva

        slot = 0
        while True:
            int_data = _read_process_memory(h_process, int_entry_addr + slot * ptr_size, ptr_size)
            if not int_data:
                break
            thunk_val, = struct.unpack_from("<I", int_data)
            if thunk_val == 0:
                break

            # Check if import by ordinal
            if thunk_val & 0x80000000:
                # Import by ordinal — skip for now
                slot += 1
                iat_entry_addr += ptr_size
                continue

            # Import by name: thunk_val is RVA to IMAGE_IMPORT_BY_NAME
            ibn_data = _read_process_memory(h_process, module_base + thunk_val, 258)
            if not ibn_data or len(ibn_data) < 3:
                slot += 1
                iat_entry_addr += ptr_size
                continue

            # Skip 2-byte Hint, then read null-terminated name
            func_name = ibn_data[2:].split(b"\x00")[0].decode("ascii", errors="replace")
            if func_name:
                # iat_entry_addr is the ADDRESS of the IAT slot (the pointer-to-pointer)
                current_iat_slot_addr = module_base + thunk_rva + slot * ptr_size
                result[func_name] = current_iat_slot_addr

            slot += 1

    return result


# =============================================================================
# IAT HOOK TABLE
# =============================================================================

class IATHook:
    """A single IAT hook entry."""
    __slots__ = ("func_name", "iat_addr", "original_ptr", "stub_addr", "active")

    def __init__(self, func_name: str, iat_addr: int, original_ptr: int, stub_addr: int):
        self.func_name    = func_name
        self.iat_addr     = iat_addr      # address of the IAT slot in target
        self.original_ptr = original_ptr  # original function pointer (saved)
        self.stub_addr    = stub_addr     # our hook stub address (in target)
        self.active       = False


# =============================================================================
# STUB GENERATOR — generates 32-bit x86 stubs that forward to the broker
# =============================================================================

class ETStubGenerator:
    """
    Generates 32-bit x86 hook stubs for IAT patching.

    Each stub:
    1. Saves all registers (pushad + pushfd)
    2. Pushes stub ID (which API is being hooked)
    3. Calls a shared dispatcher at a known address in the target
    4. The dispatcher writes to the named pipe (broker comm)
    5. Reads the response
    6. If 64-bit result: returns bridge handle; else returns original result
    7. Restores registers and returns to caller

    The dispatcher is a shared shellcode blob written once per process.
    All per-API stubs call into the same dispatcher.

    Stub layout (each ~40 bytes):
      60                 pushad
      9C                 pushfd
      68 <id3 id2 id1 id0>  push <stub_id>        ; which API
      E8 <d3  d2  d1  d0>   call <dispatcher>     ; call the shared dispatcher
      9D                 popfd
      61                 popad
      C2 <lo hi>         ret <args×4>          ; __stdcall cleanup
    """

    def __init__(self, dispatcher_addr: int, pipe_name_addr: int):
        """
        dispatcher_addr: 32-bit address of the shared dispatcher shellcode
        pipe_name_addr:  32-bit address of the null-terminated pipe name string
        """
        self.dispatcher_addr = dispatcher_addr
        self.pipe_name_addr  = pipe_name_addr

    def make_stub(
        self,
        stub_id: int,
        arg_count: int,
        cmd_family: int,
        cmd_code: int,
        stub_addr: int = 0
    ) -> bytes:
        """
        Generate a 32-bit x86 IAT hook stub.

        stub_id:   unique identifier for this hook (1..255)
        arg_count: number of DWORD arguments the hooked function takes
        cmd_family: ET lattice family (1..12)
        cmd_code:   specific ET command code
        stub_addr:  the 32-bit address where this stub will be placed in the
                    target process (required for computing E8 rel32 displacement)

        The stub uses __stdcall convention: callee pops args.
        Stack on entry: [esp+4 … esp+4+arg_count*4] = args, [esp] = return addr

        We DON'T pop the args inside the stub because we forward them through
        the pipe to the broker. The broker returns either:
          - 0: call the original function (no bridge needed)
          - N: bridge handled, N is the 32-bit result (or bridge handle)

        ISSUE-01B RESOLVED: Uses 0xE8 <rel32> (relative call to dispatcher code)
        instead of 0xFF 0x15 <addr32> (indirect call through memory pointer).
        The old 0xFF 0x15 read 4 bytes from the dispatcher CODE as a function
        pointer → garbage address → guaranteed crash.

        ET derivation (Descriptor Gap Principle): the gap between "call to
        dispatcher code" and "CALL DWORD PTR [code]" IS a Descriptor.
        0xE8 calls the code directly; 0xFF 0x15 dereferences it as a pointer.
        The correct opcode for calling a known address is 0xE8 <rel32>.
        """
        # Calculate call displacement to dispatcher
        # E8 rel32: displacement = target - (instruction_address + 5)
        # The E8 instruction will be at stub_addr + offset_of_E8_byte.

        # The full_id encodes stub_id, cmd_family, and cmd_code into a single uint32:
        # bits [31:24] = stub_id (0..255 — sequential hook index for debugging)
        # bits [23:16] = cmd_family (1..12 — ET lattice position)
        # bits [15:0]  = cmd_code (specific operation within family)
        full_id = ((stub_id & 0xFF) << 24) | ((cmd_family & 0xFF) << 16) | (cmd_code & 0xFFFF)
        ret_bytes = arg_count * 4  # __stdcall: callee cleans args

        # Ensure ret_bytes fits in 2-byte imm16
        ret_word = min(ret_bytes, 0xFFFF)

        stub = bytearray()
        stub += bytes([0x60])            # pushad
        stub += bytes([0x9C])            # pushfd

        # push the full_id (dword immediate)
        stub += bytes([0x68]) + struct.pack("<I", full_id)

        # call dispatcher (relative call via E8 rel32)
        # E8 displacement is: target - (addr_of_E8_insn + 5)
        e8_offset = len(stub)  # offset of the E8 byte within this stub
        e8_addr = (stub_addr + e8_offset) & 0xFFFFFFFF
        rel32 = (self.dispatcher_addr - (e8_addr + 5)) & 0xFFFFFFFF
        stub += bytes([0xE8]) + struct.pack("<I", rel32)

        # Add esp, 4 (clean up our push of full_id)
        stub += bytes([0x83, 0xC4, 0x04])

        # EAX now contains the result from dispatcher:
        # 0 = call original, else = bridge result
        # Test EAX
        stub += bytes([0x85, 0xC0])     # test eax, eax

        # We always return EAX as result (either original passthrough or bridge result)
        stub += bytes([0x9D])            # popfd
        stub += bytes([0x61])            # popad

        # Return with stack cleanup (__stdcall)
        if ret_word == 0:
            stub += bytes([0xC3])        # ret (no args)
        else:
            stub += bytes([0xC2]) + struct.pack("<H", ret_word)

        return bytes(stub)


# =============================================================================
# DISPATCHER SHELLCODE — the shared communication bridge in the 32-bit process
# =============================================================================

def make_dispatcher_shellcode(
    pipe_name_bytes_addr: int,  # 32-bit addr of hook_data base (pipe name, API addrs, buffers)
    result_buffer_addr:   int,  # 32-bit addr of response buffer
    buffer_size:          int = IPC_BUFFER_SIZE
) -> bytes:
    """
    The shared dispatcher shellcode (32-bit x86).
    All per-API stubs call this dispatcher, which:
    1. Reads the full argument frame from the stack
    2. Serializes it into a PDT packet
    3. Writes to the named pipe (synchronous)
    4. Reads the response
    5. Returns result in EAX

    This is the T-component of the bridge's IPC:
      T = this dispatcher traversing the Mediation channel {D,T} (the pipe)

    Input (on stack when called from a stub):
      [esp+4] = stub_id (uint32) — which API

    The full arg frame is reconstructed from the known arg counts per API.

    Returns in EAX: the bridge result (0 = let original proceed).

    Implementation note: for maximum portability, we use the Windows API
    approach of calling CreateFile + WriteFile + ReadFile on the named pipe.
    These addresses are resolved from the 32-bit kernel32.dll at dispatcher init.

    Since writing self-contained position-independent 32-bit shellcode that calls
    kernel32 without knowing addresses in advance is complex, our actual approach
    uses a LOOKUP TABLE approach:

    The 64-bit broker (our Python exe) writes the addresses of CreateFileW,
    WriteFile, ReadFile, and CloseHandle into a known memory location in the
    target process (the hook_data region). The dispatcher reads these addresses
    from the hook_data region at startup.

    Layout of hook_data region (256 bytes):
      [0..3]   = magic     (0x45543332 = "ET32")
      [4..7]   = CreateFileW address (32-bit)
      [8..11]  = WriteFile address (32-bit)
      [12..15] = ReadFile address (32-bit)
      [16..19] = CloseHandle address (32-bit)
      [20..23] = result_buffer_addr
      [24..27] = buffer_size
      [28..31] = pipe_name_addr (32-bit)
      [32..255] = pipe_name string (wchar_t, L"\\\\.\\pipe\\ET32_PDT_XXXX")

    ISSUE-01 RESOLVED: Real pipe-communicating dispatcher replaces xor eax,eax;ret.

    ET derivation (Identification Principle):
      P = the 32-bit process, D = the dispatcher code + hook_data addresses,
      T = the executing thread calling Win32 pipe APIs.
      All three are now present — E (bridged IPC) is achieved.
      The Descriptor Gap (missing pipe communication) IS closed.

    x86 machine code layout (143 bytes executable + 12 bytes data trailer):
      Prologue → Load hook_data base into ESI → Check pipe connected →
      [if not: CreateFileW to open pipe] → Write full_id via WriteFile →
      ReadFile response → Return result_buffer[0] in EAX → Epilogue.
      On failure: set status=disconnected, return 0 (pass-through).
    """
    # hook_data_base is passed as pipe_name_bytes_addr by the caller
    # (et_injector.py:_do_inject line 729 passes hook_addr as first arg)
    hook_data_base = pipe_name_bytes_addr & 0xFFFFFFFF

    # Hook_data layout offsets (must match HookDataLayout class):
    #   +0x04 = CreateFileW addr
    #   +0x08 = WriteFile addr
    #   +0x0C = ReadFile addr
    #   +0x20 = pipe_handle (HANDLE, initially 0)
    #   +0x2C = status (0=disconnected, 1=connected)  [offset 44]
    #   +0x30 = pipe_name wchar string                 [offset 48]
    #   +0x130 = arg_buffer                            [offset 304]
    #   +0x230 = result_buffer                         [offset 560]

    code = bytearray()

    # ---- Prologue (6 bytes) ----
    code += bytes([0x55])                            # push ebp
    code += bytes([0x89, 0xE5])                      # mov ebp, esp
    code += bytes([0x53])                            # push ebx
    code += bytes([0x56])                            # push esi
    code += bytes([0x57])                            # push edi

    # ---- Load hook_data base into ESI (5 bytes) ----
    code += bytes([0xBE]) + struct.pack("<I", hook_data_base)

    # ---- Check pipe status (6 bytes) ----
    code += bytes([0x83, 0x7E, 0x2C, 0x01])         # cmp dword [esi+0x2C], 1
    code += bytes([0x74, 0x25])                      # je .connected (+37 bytes)

    # ---- CreateFileW: open the named pipe (37 bytes) ----
    code += bytes([0x6A, 0x00])                      # push 0      (hTemplateFile)
    code += bytes([0x6A, 0x00])                      # push 0      (dwFlagsAndAttributes)
    code += bytes([0x6A, 0x03])                      # push 3      (OPEN_EXISTING)
    code += bytes([0x6A, 0x00])                      # push 0      (lpSecurityAttributes)
    code += bytes([0x6A, 0x00])                      # push 0      (dwShareMode)
    code += bytes([0x68])                            # push imm32  (GENERIC_READ|GENERIC_WRITE)
    code += struct.pack("<I", 0xC0000000)
    code += bytes([0x8D, 0x46, 0x30])                # lea eax, [esi+0x30]  (pipe name)
    code += bytes([0x50])                            # push eax
    code += bytes([0xFF, 0x56, 0x04])                # call [esi+4]  (CreateFileW)
    code += bytes([0x83, 0xF8, 0xFF])                # cmp eax, -1  (INVALID_HANDLE_VALUE)
    code += bytes([0x74, 0x58])                      # je .fail  (+88 bytes)
    code += bytes([0x89, 0x46, 0x20])                # mov [esi+0x20], eax  (store handle)
    code += bytes([0xC7, 0x46, 0x2C])                # mov dword [esi+0x2C], 1
    code += struct.pack("<I", 0x00000001)            # (status = connected)

    # ---- .connected: build request packet (11 bytes) ----
    code += bytes([0x8B, 0x45, 0x08])                # mov eax, [ebp+8]  (full_id)
    code += bytes([0x8D, 0xBE])                      # lea edi, [esi+0x130]
    code += struct.pack("<I", 0x00000130)            # (arg_buffer offset 304)
    code += bytes([0x89, 0x07])                      # mov [edi], eax

    # ---- WriteFile(pipe, arg_buf, 4, &written, NULL) (28 bytes) ----
    code += bytes([0x6A, 0x00])                      # push 0
    code += bytes([0x8D, 0x86])                      # lea eax, [esi+0x230]
    code += struct.pack("<I", 0x00000230)            # (result_buffer for scratch)
    code += bytes([0x50])                            # push eax
    code += bytes([0x6A, 0x04])                      # push 4
    code += bytes([0x8D, 0x86])                      # lea eax, [esi+0x130]
    code += struct.pack("<I", 0x00000130)            # (arg_buffer)
    code += bytes([0x50])                            # push eax
    code += bytes([0xFF, 0x76, 0x20])                # push [esi+0x20]
    code += bytes([0xFF, 0x56, 0x08])                # call [esi+8]  (WriteFile)
    code += bytes([0x85, 0xC0])                      # test eax, eax
    code += bytes([0x74, 0x27])                      # jz .fail  (+39 bytes)

    # ---- ReadFile(pipe, result_buf, 16, &read, NULL) (28 bytes) ----
    code += bytes([0x6A, 0x00])                      # push 0
    code += bytes([0x8D, 0x86])                      # lea eax, [esi+0x130]
    code += struct.pack("<I", 0x00000130)            # (reuse arg_buffer for bytesRead)
    code += bytes([0x50])                            # push eax
    code += bytes([0x6A, 0x10])                      # push 16
    code += bytes([0x8D, 0x86])                      # lea eax, [esi+0x230]
    code += struct.pack("<I", 0x00000230)            # (result_buffer)
    code += bytes([0x50])                            # push eax
    code += bytes([0xFF, 0x76, 0x20])                # push [esi+0x20]
    code += bytes([0xFF, 0x56, 0x0C])                # call [esi+0x0C]  (ReadFile)
    code += bytes([0x85, 0xC0])                      # test eax, eax
    code += bytes([0x74, 0x0B])                      # jz .fail  (+11 bytes)

    # ---- Success: return first DWORD of result_buffer (6 bytes) ----
    code += bytes([0x8B, 0x86])                      # mov eax, [esi+0x230]
    code += struct.pack("<I", 0x00000230)

    # ---- .done: epilogue (5 bytes) ----
    code += bytes([0x5F])                            # pop edi
    code += bytes([0x5E])                            # pop esi
    code += bytes([0x5B])                            # pop ebx
    code += bytes([0x5D])                            # pop ebp
    code += bytes([0xC3])                            # ret

    # ---- .fail: disconnect and return 0 (11 bytes) ----
    code += bytes([0xC7, 0x46, 0x2C])                # mov dword [esi+0x2C], 0
    code += struct.pack("<I", 0x00000000)            # (status = disconnected)
    code += bytes([0x31, 0xC0])                      # xor eax, eax
    code += bytes([0xEB, 0xF0])                      # jmp .done (-16)

    # Data trailer: Descriptor fields for the dispatcher's IPC configuration
    # (preserved for backward compatibility and data discovery by the broker)
    code += struct.pack("<I", pipe_name_bytes_addr)   # pipe name / hook_data address
    code += struct.pack("<I", result_buffer_addr)      # result buffer address
    code += struct.pack("<I", buffer_size)              # IPC buffer size

    return bytes(code)


# =============================================================================
# HOOK DATA REGION
# =============================================================================

HOOK_DATA_MAGIC   = 0x45543332  # "ET32"
HOOK_DATA_SIZE    = DIGITAL_ACTION_QUANTUM  # 4096 bytes

class HookDataLayout:
    """
    Layout of the shared hook_data region written into the target process.

    Offsets (all uint32 unless noted):
      0     magic              = 0x45543332
      4     createfilew_addr   = 32-bit address of CreateFileW in target
      8     writefile_addr     = 32-bit address of WriteFile in target
      12    readfile_addr      = 32-bit address of ReadFile in target
      16    closehandle_addr   = 32-bit address of CloseHandle in target
      20    result_buf_addr    = 32-bit address of 1KB result buffer (below)
      24    buffer_size        = size of result buffer
      28    pipe_name_addr     = 32-bit address of wchar_t pipe name
      32    pipe_handle        = HANDLE to open pipe (filled after CreateFile)
      36    event_req          = HANDLE to "request ready" event
      40    event_resp         = HANDLE to "response ready" event
      44    status             = bridge status: 0=disconnected, 1=connected
      48..303 = pipe_name_wchar (256 bytes = 128 wchar_t)
      304..559 = arg_buffer (256 bytes for passing args)
      560..575 = result_buffer (16 bytes = 4 uint32 return values)
      576..4095 = general IPC buffer
    """
    OFFSET_MAGIC           = 0
    OFFSET_CREATEFILEW     = 4
    OFFSET_WRITEFILE       = 8
    OFFSET_READFILE        = 12
    OFFSET_CLOSEHANDLE     = 16
    OFFSET_RESULT_BUF_ADDR = 20
    OFFSET_BUFFER_SIZE     = 24
    OFFSET_PIPE_NAME_ADDR  = 28
    OFFSET_PIPE_HANDLE     = 32
    OFFSET_EVENT_REQ       = 36
    OFFSET_EVENT_RESP      = 40
    OFFSET_STATUS          = 44
    OFFSET_PIPE_NAME_WCHAR = 48
    PIPE_NAME_WCHAR_SIZE   = 256   # bytes (128 wchar_t)
    OFFSET_ARG_BUFFER      = 304
    ARG_BUFFER_SIZE        = 256
    OFFSET_RESULT_BUFFER   = 560
    RESULT_BUFFER_SIZE     = 16
    OFFSET_IPC_BUFFER      = 576
    IPC_BUFFER_DATA_SIZE   = HOOK_DATA_SIZE - 576

    @classmethod
    def build(cls, pipe_name: str) -> bytes:
        """Build the initial hook_data block (4096 bytes)."""
        buf = bytearray(HOOK_DATA_SIZE)
        struct.pack_into("<I", buf, cls.OFFSET_MAGIC, HOOK_DATA_MAGIC)
        struct.pack_into("<I", buf, cls.OFFSET_BUFFER_SIZE, cls.IPC_BUFFER_DATA_SIZE)
        struct.pack_into("<I", buf, cls.OFFSET_STATUS, 0)  # disconnected

        # Pipe name (wchar_t)
        pipe_wide = pipe_name.encode("utf-16-le")[:cls.PIPE_NAME_WCHAR_SIZE]
        buf[cls.OFFSET_PIPE_NAME_WCHAR:cls.OFFSET_PIPE_NAME_WCHAR + len(pipe_wide)] = pipe_wide

        return bytes(buf)


# =============================================================================
# MAIN INJECTOR CLASS
# =============================================================================

class ETInjector:
    """
    Injects the ET bridge into a 32-bit target process.

    Steps:
    1. Open the target process
    2. Locate main module IAT
    3. Resolve 32-bit kernel32 addresses (for hook_data)
    4. Allocate hook_data region in target
    5. Write hook_data block into target
    6. Write dispatcher shellcode into target
    7. Write per-API stubs into target
    8. Patch IAT entries to point to stubs
    9. Report injection complete

    ET derivation:
      The injection IS the Descriptor installation: we're adding D_bridge to the
      32-bit process's Descriptor set. Each IAT patch = replacing one D with a
      richer D that includes the 64-bit capability.
    """

    def __init__(self, pipe_name_template: str):
        self.pipe_name_template = pipe_name_template
        self._log    = ETLog("Injector")
        self._heaven = ETHeavenGate()
        self._hooks: Dict[int, List[IATHook]] = {}  # pid → hooks
        self._hooks_lock = threading.Lock()  # thread-safe access to _hooks

    def inject(self, pid: int, config: TargetConfig) -> bool:
        """
        Perform full injection into target process.

        Primary path: inject et_bridge32.dll via CreateRemoteThread(LoadLibraryA).
        The DLL's ET32_Init handles pipe connection, handshake, VEH, IAT patching.
        Fallback path: shellcode stubs + IAT patching from the broker side.

        Returns True on success.

        ET derivation (Identification Principle):
          P = 32-bit target, D = et_bridge32.dll, T = CreateRemoteThread.
          All three are required for E (bridged execution).
        """
        pipe_name = self.pipe_name_template.format(pid=pid)

        self._log.mediation(f"Starting injection into PID {pid} ({config.exe_name})")

        h = getattr(kernel32, 'OpenProcess')(PROCESS_ALL_ACCESS, False, pid)
        if not h:
            self._log.error(
                f"OpenProcess({pid}) failed: {ctypes.GetLastError()}"
            )
            return False

        try:
            # Primary path: DLL injection (activates the full C bridge)
            if self._inject_dll(h, pid, config, pipe_name):
                self._log.exception_state(
                    f"DLL injection succeeded for PID {pid} — "
                    f"et_bridge32.dll loaded, ET32_Init complete, "
                    f"pipe connected, IAT patched from C side"
                )
                return True

            # Fallback: shellcode stubs + broker-side IAT patching
            self._log.warning_di(
                f"DLL injection failed for PID {pid}, "
                f"falling back to shellcode stub injection"
            )
            return self._do_inject(h, pid, config, pipe_name)
        finally:
            getattr(kernel32, 'CloseHandle')(h)

    # =========================================================================
    # PRIMARY INJECTION PATH — et_bridge32.dll via CreateRemoteThread
    # =========================================================================

    def _resolve_dll_path(self, config: TargetConfig) -> Optional[str]:
        """
        Resolve the full filesystem path to et_bridge32.dll.

        Search order (ET Descriptor Gap Principle — follow the search path):
          1. Config-specified dll_path (if absolute and exists)
          2. Config-specified dll_path relative to the broker executable directory
          3. 'et_bridge32.dll' in the broker executable directory
          4. 'et_bridge32.dll' in the current working directory

        Returns the resolved absolute path, or None if the DLL cannot be found.
        """
        # Determine the broker's base directory
        if getattr(sys, 'frozen', False):
            # PyInstaller: _MEIPASS or the directory of the executable
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path(__file__).parent

        # Candidate: config-specified path
        cfg_path = getattr(config, 'dll_path', None) or "et_bridge32.dll"

        candidates = []
        p = Path(cfg_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(base_dir / p)
            candidates.append(Path.cwd() / p)

        # Always include the default name in the broker directory
        candidates.append(base_dir / "et_bridge32.dll")
        candidates.append(Path.cwd() / "et_bridge32.dll")

        for candidate in candidates:
            if candidate.is_file():
                resolved = str(candidate.resolve())
                self._log.debug(f"DLL resolved: {resolved}")
                return resolved

        self._log.error(
            f"et_bridge32.dll not found. Searched: "
            f"{', '.join(str(c) for c in candidates)}"
        )
        return None

    def _inject_dll(
        self,
        h_process,
        pid: int,
        config: TargetConfig,
        pipe_name: str
    ) -> bool:
        """
        Inject et_bridge32.dll into the target 32-bit process.

        Steps (ET derivation — each step is a Descriptor in the injection chain):
          D₁: Resolve DLL path on disk
          D₂: Find LoadLibraryA address in target's kernel32.dll
          D₃: Allocate + write DLL path string into target memory
          D₄: CreateRemoteThread(LoadLibraryA, dll_path_addr) → loads the DLL
          D₅: Wait for DLL load, verify module presence
          D₆: Resolve ET32_Init export in target's loaded DLL
          D₇: CreateRemoteThread(ET32_Init, broker_pid) → connects pipe + patches IAT
          D₈: Verify init result

        The Descriptor Gap Principle: any failure in D₁–D₈ identifies exactly
        which Descriptor is missing. The gap IS the diagnosis.

        Returns True if injection and init both succeeded.
        """
        # D₁: Resolve DLL path
        dll_path = self._resolve_dll_path(config)
        if not dll_path:
            self._log.error(f"DLL path resolution failed for PID {pid}")
            return False

        self._log.debug(f"DLL path for PID {pid}: {dll_path}")

        # D₂: Find LoadLibraryA in target's 32-bit kernel32.dll
        k32_base = _get_module_base(h_process, "kernel32.dll")
        if not k32_base:
            self._log.error(f"kernel32.dll not found in PID {pid}")
            return False

        load_library_addr = self._get_func_addr_in_target(
            h_process, k32_base, "LoadLibraryA"
        )
        if not load_library_addr:
            self._log.error(f"LoadLibraryA not found in PID {pid} kernel32")
            return False

        self._log.debug(
            f"PID {pid}: kernel32 @ 0x{k32_base:08X}, "
            f"LoadLibraryA @ 0x{load_library_addr:08X}"
        )

        # D₃: Write DLL path string into target memory
        dll_path_bytes = dll_path.encode("ascii") + b"\x00"

        virt_alloc_ex = getattr(kernel32, 'VirtualAllocEx')
        virt_alloc_ex.restype = ctypes.c_void_p
        virt_alloc_ex.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD
        ]

        path_va = virt_alloc_ex(
            h_process, None, len(dll_path_bytes),
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        )
        if not path_va:
            self._log.error(
                f"VirtualAllocEx for DLL path failed in PID {pid}: "
                f"{ctypes.GetLastError()}"
            )
            return False

        path_addr = ctypes.c_void_p(path_va).value or 0
        if not _write_process_memory(h_process, path_addr, dll_path_bytes):
            self._log.error(f"WriteProcessMemory for DLL path failed in PID {pid}")
            return False

        self._log.debug(
            f"PID {pid}: DLL path written to 0x{path_addr:08X} "
            f"({len(dll_path_bytes)} bytes)"
        )

        # D₄: CreateRemoteThread(LoadLibraryA, dll_path_addr)
        thread_id = ctypes.wintypes.DWORD(0)

        create_remote_thread = getattr(kernel32, 'CreateRemoteThread')
        create_remote_thread.restype = ctypes.wintypes.HANDLE
        create_remote_thread.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.wintypes.DWORD)
        ]

        h_thread = create_remote_thread(
            h_process, None, 0,
            ctypes.c_void_p(load_library_addr),
            ctypes.c_void_p(path_addr),
            0, ctypes.byref(thread_id)
        )
        if not h_thread:
            self._log.error(
                f"CreateRemoteThread(LoadLibraryA) failed for PID {pid}: "
                f"{ctypes.GetLastError()}"
            )
            return False

        self._log.debug(
            f"PID {pid}: LoadLibraryA thread created (TID {thread_id.value}), "
            f"waiting for DLL load..."
        )

        # Wait for DLL load (ET timeout: CONN_TIMEOUT_MS)
        wait_result = getattr(kernel32, 'WaitForSingleObject')(
            h_thread, CONN_TIMEOUT_MS
        )
        if wait_result != 0:  # WAIT_OBJECT_0 = 0
            self._log.error(
                f"WaitForSingleObject on LoadLibraryA thread timed out "
                f"for PID {pid} (wait_result={wait_result})"
            )
            getattr(kernel32, 'CloseHandle')(h_thread)
            return False

        # Get exit code — this is the HMODULE returned by LoadLibraryA
        exit_code = ctypes.wintypes.DWORD(0)
        getattr(kernel32, 'GetExitCodeThread')(
            h_thread, ctypes.byref(exit_code)
        )
        getattr(kernel32, 'CloseHandle')(h_thread)

        # D₅: Verify DLL loaded
        # Note: GetExitCodeThread returns DWORD (32-bit), so for 32-bit targets
        # the HMODULE fits. If exit_code is 0, LoadLibraryA failed.
        if exit_code.value == 0:
            self._log.error(
                f"LoadLibraryA returned NULL for PID {pid} — "
                f"DLL load failed (check DLL path and 32-bit compatibility)"
            )
            # Free the path string allocation
            getattr(kernel32, 'VirtualFreeEx')(
                h_process, ctypes.c_void_p(path_addr), 0, 0x8000  # MEM_RELEASE
            )
            return False

        # Confirm via module enumeration (more reliable than exit code for base)
        dll_base = _get_module_base(h_process, "et_bridge32.dll")
        if not dll_base:
            # Fall back to exit code as base address
            dll_base = exit_code.value
            self._log.warning_di(
                f"Module enum did not find et_bridge32.dll in PID {pid}, "
                f"using LoadLibraryA return value 0x{dll_base:08X} as base"
            )
        else:
            self._log.debug(
                f"PID {pid}: et_bridge32.dll loaded at 0x{dll_base:08X}"
            )

        # Store for later use (get_universal_hook_addr, call_dll_export, etc.)
        self._dll_base_in_target = dll_base

        # D₅½: Write hook_data block into target — provides the broker's
        # canonical pipe_name and kernel32 API addresses in the target's
        # address space. Used by:
        #   - The DLL (pipe name verification / reconnect fallback)
        #   - The shellcode dispatcher (if DLL init fails and stubs are used)
        #   - Diagnostic tools reading the target's memory
        #
        # ET derivation: the hook_data IS the Descriptor set for IPC.
        # Writing it into P (target memory) before T (ET32_Init) executes
        # ensures the full D-set is available when T traverses.
        hook_data_bytes = HookDataLayout.build(pipe_name)
        hook_region = virt_alloc_ex(
            h_process, None, HOOK_DATA_SIZE,
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        )
        if hook_region:
            hook_addr = ctypes.c_void_p(hook_region).value or 0
            # Fill in pipe_name_addr and kernel32 function addresses
            hook_data_final = bytearray(hook_data_bytes)
            pipe_name_addr_32 = (hook_addr + HookDataLayout.OFFSET_PIPE_NAME_WCHAR) & 0xFFFFFFFF
            struct.pack_into("<I", hook_data_final,
                             HookDataLayout.OFFSET_PIPE_NAME_ADDR, pipe_name_addr_32)
            # Resolve kernel32 API addresses for the dispatcher lookup table
            for func_name, hook_offset in [
                ("CreateFileW",  HookDataLayout.OFFSET_CREATEFILEW),
                ("WriteFile",    HookDataLayout.OFFSET_WRITEFILE),
                ("ReadFile",     HookDataLayout.OFFSET_READFILE),
                ("CloseHandle",  HookDataLayout.OFFSET_CLOSEHANDLE),
            ]:
                addr_32 = self._get_func_addr_in_target(
                    h_process, k32_base, func_name
                )
                if addr_32:
                    struct.pack_into("<I", hook_data_final, hook_offset, addr_32)
            _write_process_memory(h_process, hook_addr, bytes(hook_data_final))
            self._log.debug(
                f"PID {pid}: hook_data written @ 0x{hook_addr:08X}, "
                f"pipe={pipe_name}"
            )
        else:
            self._log.warning_di(
                f"PID {pid}: hook_data allocation failed — "
                f"DLL will use internal pipe name construction. "
                f"Broker pipe: {pipe_name}"
            )

        # D₆: Resolve ET32_Init in the loaded DLL
        init_addr = self._get_func_addr_in_target(
            h_process, dll_base, "ET32_Init"
        )
        if not init_addr:
            self._log.error(
                f"ET32_Init export not found in et_bridge32.dll @ "
                f"0x{dll_base:08X} in PID {pid}"
            )
            return False

        self._log.debug(
            f"PID {pid}: ET32_Init @ 0x{init_addr:08X}"
        )

        # D₇: Call ET32_Init(broker_pid) via CreateRemoteThread
        broker_pid = os.getpid()
        h_thread_init = create_remote_thread(
            h_process, None, 0,
            ctypes.c_void_p(init_addr),
            ctypes.c_void_p(broker_pid),
            0, ctypes.byref(thread_id)
        )
        if not h_thread_init:
            self._log.error(
                f"CreateRemoteThread(ET32_Init) failed for PID {pid}: "
                f"{ctypes.GetLastError()}"
            )
            return False

        self._log.debug(
            f"PID {pid}: ET32_Init thread created (TID {thread_id.value}), "
            f"broker_pid={broker_pid}, waiting for handshake..."
        )

        # Wait for init (includes pipe connect + handshake + IAT patch)
        wait_result = getattr(kernel32, 'WaitForSingleObject')(
            h_thread_init, CONN_TIMEOUT_MS
        )
        if wait_result != 0:
            self._log.error(
                f"ET32_Init timed out for PID {pid}"
            )
            getattr(kernel32, 'CloseHandle')(h_thread_init)
            return False

        # D₈: Verify init result (TRUE=1, FALSE=0)
        init_result = ctypes.wintypes.DWORD(0)
        getattr(kernel32, 'GetExitCodeThread')(
            h_thread_init, ctypes.byref(init_result)
        )
        getattr(kernel32, 'CloseHandle')(h_thread_init)

        if not init_result.value:
            self._log.error(
                f"ET32_Init returned FALSE for PID {pid} — "
                f"pipe connection or handshake failed"
            )
            return False

        self._log.info(
            f"PID {pid}: ET32_Init succeeded — pipe connected, "
            f"IAT patched, VEH installed, AWE signalled"
        )

        # Free the path string allocation (no longer needed)
        getattr(kernel32, 'VirtualFreeEx')(
            h_process, ctypes.c_void_p(path_addr), 0, 0x8000  # MEM_RELEASE
        )

        # Record hooks (DLL-side IAT hooks are managed by the DLL;
        # we record an empty list here as the broker tracks DLL-managed hooks
        # through the pipe protocol, not through Python IATHook objects)
        with self._hooks_lock:
            self._hooks[pid] = []

        return True

    def call_dll_export(
        self,
        h_process,
        func_name: str,
        param: int = 0
    ) -> Optional[int]:
        """
        Call an exported function in the injected et_bridge32.dll via
        CreateRemoteThread.

        Used by the broker to invoke DLL functions post-injection, such as
        ET32_SetKiFastTrampoline(trampoline_addr) after the WOW64 hook
        installs the trampoline.

        ET derivation: T = CreateRemoteThread, D = the export function,
        P = the target process. E = the export function executes.

        Returns the function's return value (thread exit code), or None on failure.
        """
        dll_base = getattr(self, '_dll_base_in_target', 0)
        if not dll_base:
            self._log.error(
                f"call_dll_export({func_name}): no DLL base — "
                f"DLL not injected"
            )
            return None

        func_addr = self._get_func_addr_in_target(
            h_process, dll_base, func_name
        )
        if not func_addr:
            self._log.error(
                f"call_dll_export: {func_name} not found in "
                f"et_bridge32.dll @ 0x{dll_base:08X}"
            )
            return None

        thread_id = ctypes.wintypes.DWORD(0)
        create_remote_thread = getattr(kernel32, 'CreateRemoteThread')
        create_remote_thread.restype = ctypes.wintypes.HANDLE
        create_remote_thread.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.wintypes.DWORD)
        ]

        h_thread = create_remote_thread(
            h_process, None, 0,
            ctypes.c_void_p(func_addr),
            ctypes.c_void_p(param),
            0, ctypes.byref(thread_id)
        )
        if not h_thread:
            self._log.error(
                f"call_dll_export: CreateRemoteThread({func_name}) failed: "
                f"{ctypes.GetLastError()}"
            )
            return None

        wait_result = getattr(kernel32, 'WaitForSingleObject')(
            h_thread, CONN_TIMEOUT_MS
        )
        if wait_result != 0:
            self._log.error(
                f"call_dll_export: {func_name} timed out"
            )
            getattr(kernel32, 'CloseHandle')(h_thread)
            return None

        exit_code = ctypes.wintypes.DWORD(0)
        getattr(kernel32, 'GetExitCodeThread')(
            h_thread, ctypes.byref(exit_code)
        )
        getattr(kernel32, 'CloseHandle')(h_thread)

        self._log.debug(
            f"call_dll_export: {func_name}(0x{param:X}) → 0x{exit_code.value:X}"
        )
        return exit_code.value

    def _do_inject(
        self,
        h_process,
        pid: int,
        config: TargetConfig,
        pipe_name: str
    ) -> bool:

        # Step 1: Find main module base
        exe_lower = config.exe_name.lower()
        mod_base = _get_module_base(h_process, exe_lower)
        if not mod_base:
            # Try kernel32 as fallback to at least find the module list
            self._log.warning_di(
                f"Could not find module base for {exe_lower}, using PEB walk fallback"
            )
            # Use GetModuleHandle in target via shellcode — not implemented here.
            # Fall through to IAT-less mode.
            return self._inject_no_module(h_process, pid, config, pipe_name)

        # Step 2: Parse IAT
        iat = _parse_iat(h_process, mod_base)
        if not iat:
            self._log.warning_di(f"IAT parse failed for PID {pid}")
            return False

        self._log.debug(f"IAT has {len(iat)} entries for PID {pid}")

        # Step 3: Allocate hook_data region in target (4096 bytes)
        hook_data_bytes = HookDataLayout.build(pipe_name)
        virt_alloc_ex = getattr(kernel32, 'VirtualAllocEx')
        virt_alloc_ex.restype  = ctypes.c_void_p
        virt_alloc_ex.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD
        ]

        # hook_data is a pure data region — PAGE_READWRITE (no execute needed)
        hook_region = virt_alloc_ex(
            h_process, None, HOOK_DATA_SIZE,
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        )
        if not hook_region:
            self._log.error(
                f"Failed to allocate hook_data region in PID {pid}: "
                f"{ctypes.GetLastError()}"
            )
            return False

        hook_addr = ctypes.c_void_p(hook_region).value or 0

        # Fill in pipe_name_addr in hook_data
        pipe_name_offset = HookDataLayout.OFFSET_PIPE_NAME_WCHAR
        pipe_name_addr_32 = (hook_addr + pipe_name_offset) & 0xFFFFFFFF
        struct.pack_into("<I",
            bytearray(hook_data_bytes),
            HookDataLayout.OFFSET_PIPE_NAME_ADDR,
            pipe_name_addr_32
        )
        hook_data_bytes_final = bytearray(hook_data_bytes)
        struct.pack_into("<I", hook_data_bytes_final, HookDataLayout.OFFSET_PIPE_NAME_ADDR, pipe_name_addr_32)

        # Step 4: Resolve 32-bit kernel32 addresses in target
        k32_base_32 = _get_module_base(h_process, "kernel32.dll")
        if k32_base_32:
            # Get CreateFileW, WriteFile, ReadFile, CloseHandle addresses
            # We compute them from the 64-bit broker's knowledge of offsets
            # (they have the same offsets in 32-bit kernel32 from SysWOW64)
            for func_name, hook_offset in [
                ("CreateFileW",  HookDataLayout.OFFSET_CREATEFILEW),
                ("WriteFile",    HookDataLayout.OFFSET_WRITEFILE),
                ("ReadFile",     HookDataLayout.OFFSET_READFILE),
                ("CloseHandle",  HookDataLayout.OFFSET_CLOSEHANDLE),
            ]:
                # Find the function in the target's 32-bit IAT or module
                addr_32 = self._get_func_addr_in_target(h_process, k32_base_32, func_name)
                if addr_32:
                    struct.pack_into(
                        "<I", hook_data_bytes_final, hook_offset, addr_32
                    )

        # Step 5: Write hook_data to target
        if not _write_process_memory(h_process, hook_addr, bytes(hook_data_bytes_final)):
            self._log.error(f"Failed to write hook_data to PID {pid}")
            return False

        # Step 6: Write dispatcher stub
        dispatcher_code = make_dispatcher_shellcode(hook_addr, hook_addr + HookDataLayout.OFFSET_RESULT_BUFFER)
        disp_region = virt_alloc_ex(
            h_process, None, max(len(dispatcher_code), DIGITAL_ACTION_QUANTUM),
            MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE
        )
        if not disp_region:
            self._log.error("Failed to allocate dispatcher region")
            return False

        disp_addr = ctypes.c_void_p(disp_region).value or 0
        if not _write_process_memory(h_process, disp_addr, dispatcher_code):
            self._log.error("Failed to write dispatcher code")
            return False

        # Step 7: Generate and write per-API stubs
        # Allocate one big code region for all stubs (S × 64 bytes = 768 bytes)
        stub_region_size = S * 64
        stub_region = virt_alloc_ex(
            h_process, None, stub_region_size,
            MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE
        )
        if not stub_region:
            self._log.error("Failed to allocate stub region")
            return False

        stub_base = ctypes.c_void_p(stub_region).value or 0
        gen = ETStubGenerator(
            dispatcher_addr=disp_addr & 0xFFFFFFFF,
            pipe_name_addr=pipe_name_addr_32
        )

        hooks = []
        stub_offset = 0
        all_stubs_data = bytearray()

        # Only hook APIs that the target needs based on config features
        # Build CmdCode reverse lookup dynamically for validation
        cmd_code_names: Dict[int, str] = {
            v: k for k, v in CmdCode.__dict__.items()
            if not k.startswith('_') and isinstance(v, int)
        }

        for api_name, iat_entry_addr in iat.items():
            route: Optional[Tuple[int, int]] = ETLatticeRouter.route(api_name)
            if route is None:
                continue

            family, code = route
            if not config.has_feature(family):
                continue

            # Log the ET lattice routing with CmdFamily and CmdCode resolution
            lattice_desc = CmdFamily.FAMILY_TO_D.get(family, f"d={family}")
            code_name = cmd_code_names.get(code, f"0x{code:02X}")
            self._log.debug(
                f"  Routing {api_name} → {lattice_desc}, cmd={code_name}"
            )

            # Determine arg count for this API
            arg_count = self._api_arg_count(api_name)
            stub_id   = len(hooks) + 1

            stub = gen.make_stub(stub_id, arg_count, family, code,
                                stub_addr=(stub_base + stub_offset) & 0xFFFFFFFF)
            stub_addr = stub_base + stub_offset

            all_stubs_data += stub
            stub_offset += len(stub)
            if stub_offset % 4 != 0:
                # Align to 4 bytes
                pad = 4 - (stub_offset % 4)
                all_stubs_data += bytes(pad)
                stub_offset += pad

            # Read original IAT value
            orig_data = _read_process_memory(h_process, iat_entry_addr, 4)
            orig_ptr = struct.unpack_from("<I", orig_data)[0] if orig_data else 0

            hook = IATHook(api_name, iat_entry_addr, orig_ptr, stub_addr & 0xFFFFFFFF)
            hooks.append(hook)

        # Write all stubs at once
        if all_stubs_data:
            if not _write_process_memory(h_process, stub_base, bytes(all_stubs_data)):
                self._log.error("Failed to write stubs")
                return False

        # Step 8: Patch IAT entries
        patched = 0
        for hook in hooks:
            new_ptr = struct.pack("<I", hook.stub_addr)
            if _write_process_memory(h_process, hook.iat_addr, new_ptr):
                hook.active = True
                patched += 1
                self._log.debug(
                    f"  Patched [{hook.func_name}] IAT @ 0x{hook.iat_addr:08X} "
                    f"→ stub @ 0x{hook.stub_addr:08X}"
                )
            else:
                self._log.warning_di(
                    f"  Failed to patch [{hook.func_name}] IAT"
                )

        with self._hooks_lock:
            self._hooks[pid] = hooks
        self._log.exception_state(
            f"Injection complete: PID {pid}, "
            f"patched {patched}/{len(hooks)} hooks, "
            f"hook_data @ 0x{hook_addr:08X}"
        )
        return patched > 0

    def _inject_no_module(
        self, h_process, pid: int, config: TargetConfig, pipe_name: str
    ) -> bool:
        """
        Fallback injection without module IAT parsing.

        When the main module base cannot be located (e.g. packed executables
        or unusual PE layouts), we still allocate the hook_data region and
        write the pipe connection descriptor into the target. This enables
        later connection via the pipe when the target calls bridgeable APIs.

        ET derivation: This is pure Descriptor installation without Traverser
        activation — {P,D} Unsubstantiated state. The hook_data region is
        a D waiting for T (the target's execution) to discover and traverse it.
        """
        self._log.info(f"Using no-module injection for PID {pid}")

        # Validate configuration: Koide threshold — at least K × S features should
        # be enabled for a meaningful bridge (K = 2/3, so 8 of 12 families)
        active_features = sum(1 for d in range(1, S + 1) if config.has_feature(d))
        koide_minimum = int(K * S)
        if active_features < koide_minimum:
            self._log.warning_di(
                f"PID {pid}: only {active_features}/{S} features active "
                f"(Koide minimum = {koide_minimum})"
            )

        # Log DLL search paths from config if present, resolving via Path
        for dll_dir in config.dll_search_paths:
            dll_path = Path(dll_dir)
            if dll_path.exists():
                self._log.debug(f"DLL search path validated: {dll_path}")
            else:
                self._log.warning_di(f"DLL search path not found: {dll_path}")

        # Resolve broker PID for pipe name construction
        broker_pid = os.getpid()
        self._log.debug(f"Broker PID={broker_pid}, target PID={pid}")

        # Detect frozen (PyInstaller) environment for path context
        frozen = getattr(sys, 'frozen', False)
        if frozen:
            self._log.debug("Running in frozen (PyInstaller) environment")

        # Build and write hook_data with pipe connection info
        hook_data_bytes = HookDataLayout.build(pipe_name)

        virt_alloc_ex = getattr(kernel32, 'VirtualAllocEx')
        virt_alloc_ex.restype  = ctypes.c_void_p
        virt_alloc_ex.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD
        ]

        hook_region = virt_alloc_ex(
            h_process, None, HOOK_DATA_SIZE,
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        )
        if not hook_region:
            self._log.error(
                f"Fallback inject: failed to allocate hook_data in PID {pid}: "
                f"{ctypes.GetLastError()}"
            )
            return False

        hook_addr = ctypes.c_void_p(hook_region).value or 0

        # Validate alignment: hook_addr should fall on a V_BASE-aligned boundary
        # V_BASE = 1/S, so the address modulo DIGITAL_ACTION_QUANTUM should be 0
        # (all allocations from VirtualAllocEx are page-aligned by Windows,
        #  but we verify ET invariant: addr mod ħ_d == 0)
        alignment_remainder = hook_addr % DIGITAL_ACTION_QUANTUM
        if alignment_remainder != 0:
            self._log.warning_di(
                f"hook_data address 0x{hook_addr:08X} not ħ_d-aligned "
                f"(remainder={alignment_remainder}, V_BASE={V_BASE})"
            )

        # Fill in pipe_name_addr field
        pipe_name_offset = HookDataLayout.OFFSET_PIPE_NAME_WCHAR
        pipe_name_addr_32 = (hook_addr + pipe_name_offset) & 0xFFFFFFFF
        hook_data_final = bytearray(hook_data_bytes)
        struct.pack_into("<I", hook_data_final, HookDataLayout.OFFSET_PIPE_NAME_ADDR, pipe_name_addr_32)

        if not _write_process_memory(h_process, hook_addr, bytes(hook_data_final)):
            self._log.error(f"Fallback inject: failed to write hook_data to PID {pid}")
            return False

        # Log feature set from config using CmdFamily lattice descriptions
        enabled_descs = [
            CmdFamily.FAMILY_TO_D.get(d, f"d={d}")
            for d in sorted(config.features)
        ]
        self._log.info(
            f"Fallback inject complete for PID {pid}: "
            f"hook_data @ 0x{hook_addr:08X}, "
            f"features: {', '.join(enabled_descs)}"
        )
        return True  # partial success — hook_data written, pipe ready, IAT hooks not installed

    def _get_func_addr_in_target(
        self, h_process, module_base: int, func_name: str
    ) -> int:
        """
        Find the address of 'func_name' in the target's loaded module.
        Parses the module's export table.
        """
        # Read DOS + NT headers
        hdr = _read_process_memory(h_process, module_base, 256)
        if not hdr:
            self._log.debug(f"Export resolve: cannot read headers at 0x{module_base:08X}")
            return 0

        sig, = struct.unpack_from("<H", hdr, 0)
        if sig != IMAGE_DOS_SIGNATURE:
            self._log.debug(f"Export resolve: invalid DOS signature at 0x{module_base:08X}")
            return 0

        e_lfanew, = struct.unpack_from("<I", hdr, 60)
        nt = _read_process_memory(h_process, module_base + e_lfanew, 256)
        if not nt:
            return 0

        # Export directory: PE32 offset = 24 + 112 bytes into optional header...
        # Actually: FILE_HEADER size=20, optional magic at offset 24
        opt_magic, = struct.unpack_from("<H", nt, 24)
        if opt_magic == 0x10B:  # PE32
            exp_rva_off = 24 + 96  # offset of export dir in PE32 optional header
        else:
            exp_rva_off = 24 + 112

        if exp_rva_off + 8 > len(nt):
            return 0

        exp_rva,  = struct.unpack_from("<I", nt, exp_rva_off)
        exp_size, = struct.unpack_from("<I", nt, exp_rva_off + 4)

        if exp_rva == 0:
            return 0

        exp_data = _read_process_memory(h_process, module_base + exp_rva, exp_size)
        if not exp_data or len(exp_data) < 40:
            return 0

        # IMAGE_EXPORT_DIRECTORY (40 bytes):
        # [0]  Characteristics
        # [4]  TimeDateStamp
        # [8]  MajorVersion / MinorVersion
        # [12] Name RVA
        # [16] Base
        # [20] NumberOfFunctions
        # [24] NumberOfNames
        # [28] AddressOfFunctions (RVA)
        # [32] AddressOfNames (RVA)
        # [36] AddressOfNameOrdinals (RVA)
        _, _, _, _, _, num_funcs, num_names, funcs_rva, names_rva, ords_rva = \
            struct.unpack_from("<IIIIIIIIII", exp_data, 0)

        names_data   = _read_process_memory(h_process, module_base + names_rva, num_names * 4)
        ords_data    = _read_process_memory(h_process, module_base + ords_rva,  num_names * 2)
        funcs_data   = _read_process_memory(h_process, module_base + funcs_rva, num_funcs * 4)

        if not (names_data and ords_data and funcs_data):
            return 0

        func_name_enc = func_name.encode("ascii")
        for i in range(num_names):
            name_rva_i, = struct.unpack_from("<I", names_data, i * 4)
            name_data   = _read_process_memory(h_process, module_base + name_rva_i, 128)
            if not name_data:
                continue
            name = name_data.split(b"\x00")[0]
            if name == func_name_enc:
                ordinal, = struct.unpack_from("<H", ords_data, i * 2)
                func_rva_i, = struct.unpack_from("<I", funcs_data, ordinal * 4)
                resolved_addr = (module_base + func_rva_i) & 0xFFFFFFFF
                self._log.debug(
                    f"Export resolve: {func_name} → 0x{resolved_addr:08X} "
                    f"(ordinal={ordinal})"
                )
                return resolved_addr

        self._log.debug(f"Export resolve: {func_name} not found in module @ 0x{module_base:08X}")
        return 0

    @staticmethod
    def _api_arg_count(api_name: str) -> int:
        """Return the number of DWORD arguments for a known API."""
        arg_counts = {
            "VirtualAlloc":        4,
            "VirtualAllocEx":      5,
            "VirtualFree":         3,
            "VirtualFreeEx":       4,
            "VirtualProtect":      4,
            "VirtualQuery":        3,
            "HeapAlloc":           3,
            "HeapFree":            3,
            "HeapReAlloc":         4,
            "GlobalAlloc":         2,
            "LocalAlloc":          2,
            "malloc":              1,
            "calloc":              2,
            "realloc":             2,
            "CreateFileMappingA":  6,
            "CreateFileMappingW":  6,
            "MapViewOfFile":       5,
            "MapViewOfFileEx":     6,
            "UnmapViewOfFile":     1,
            "FlushViewOfFile":     2,
            "CreateThread":        6,
            "CreateRemoteThread":  7,
            "SuspendThread":       1,
            "ResumeThread":        1,
            "TerminateThread":     2,
            "GetThreadContext":    2,
            "LoadLibraryA":        1,
            "LoadLibraryW":        1,
            "LoadLibraryExA":      3,
            "LoadLibraryExW":      3,
            "FreeLibrary":         1,
            "GetProcAddress":      2,
            "CreateProcessA":      10,
            "CreateProcessW":      10,
            "OpenProcess":         3,
            "GetSystemInfo":       1,
            "CreateFileA":         7,
            "CreateFileW":         7,
        }
        return arg_counts.get(api_name, 4)  # default 4 args

    def remove_hooks(self, pid: int) -> bool:
        """Restore all IAT entries to their original values."""
        with self._hooks_lock:
            hooks = self._hooks.get(pid, [])
        if not hooks:
            return True

        h = getattr(kernel32, 'OpenProcess')(PROCESS_ALL_ACCESS, False, pid)
        if not h:
            return False

        restored = 0
        try:
            for hook in hooks:
                if hook.active and hook.original_ptr:
                    orig_bytes = struct.pack("<I", hook.original_ptr)
                    if _write_process_memory(h, hook.iat_addr, orig_bytes):
                        hook.active = False
                        restored += 1
        finally:
            getattr(kernel32, 'CloseHandle')(h)

        with self._hooks_lock:
            del self._hooks[pid]
        self._log.info(f"Restored {restored}/{len(hooks)} IAT entries for PID {pid}")
        return True

    def active_hook_count(self, pid: int) -> int:
        """Return the count of currently active IAT hooks for a given process.

        ET derivation: counts the cardinality of the active D-set (Descriptors
        successfully installed in the target's Import Address Table). Each active
        hook is a Descriptor that has been traversed from Unsubstantiated {P,D}
        to Exception state E — the bridge is engaged for that API.
        """
        with self._hooks_lock:
            return sum(1 for h in self._hooks.get(pid, []) if h.active)

    def get_hooks_summary(self, pid: int) -> Tuple[int, int]:
        """Return (total_hooks, active_hooks) for a given process.

        ET derivation: returns the full D-set cardinality and the active
        (traversed) subset cardinality. The ratio active/total approximates
        the Koide binding stability K = 2/3 for a healthy injection.
        """
        with self._hooks_lock:
            hooks = self._hooks.get(pid, [])
            total = len(hooks)
            active = sum(1 for h in hooks if h.active)
        return total, active
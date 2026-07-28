"""
et_bridge/et_heaven.py
ET32 Bridge — Heaven's Gate: 32-bit → 64-bit Mode Transition

Derived from P ∘ D ∘ T = E.

Heaven's Gate is the physical instantiation of the ET Mediation state {D,T}:
  - D = GDT entry 0x33 (the Descriptor that enables 64-bit execution)
  - T = the executing thread (the Traverser that switches through the gate)

The transition: CS = 0x23 (32-bit mode) → CS = 0x33 (64-bit mode)
From ET Digital Manifold: The D-bit in the GDT entry IS the Descriptor change.
Switching from CS=0x23 to CS=0x33 changes exactly ONE Descriptor:
  D_32: GDT entry with L-bit = 0 (32-bit execution)
  D_64: GDT entry with L-bit = 1 (64-bit execution)

The GDT segment selectors:
  0x23 = 00100011₂ = TI=0, RPL=3, Index=4 → GDT[4] → 32-bit code (D/B=1, L=0)
  0x33 = 00110011₂ = TI=0, RPL=3, Index=6 → GDT[6] → 64-bit code (L=1, D/B=0)

Both descriptors point to base=0x00000000 with limit=0xFFFFFFFF (flat segment).
The ONLY difference is the L-bit — a single Descriptor toggle.

ET derivation:
  P_32 (32-bit register file) → D_transition → T (executing thread) → P_64 (64-bit registers)
  The Heaven's Gate IS the ET Mediation layer: {D,T} without fixed P.
  After the gate: new P = 64-bit register/address space. D and T unchanged.

Heaven's Gate shellcode patterns:
  (1) CALL FAR pattern:
      push 0x33
      call $+5       ; pushes EIP+5 onto stack
      add [esp], 5   ; advance to x64 code
      retf           ; pops EIP+5 and CS=0x33, entering 64-bit mode

  (2) JMP FAR pattern:
      mov EDX, ESP
      sub ESP, 8
      mov [ESP+4], 0x33
      mov [ESP+0], <target_addr>
      jmp far [ESP]  ; jumps to target in 64-bit mode

  Return from 64-bit: retf with CS=0x23 pushes on stack before the retf.

This module provides:
  1. Pre-built shellcode stubs for common bridge operations
  2. A 64-bit ntdll function resolver (reads the 64-bit PEB via FS:[0xC0])
  3. A VirtualAlloc64 stub that allocates memory in the high 64-bit address space
  4. A GetProcAddress64 stub for 64-bit DLL functions
"""

import ctypes
import ctypes.wintypes
import struct
import os
import sys
from typing import Optional, Dict, Tuple
from et_math import (
    S, K, V_BASE, ADDR64_BASE, DIGITAL_ACTION_QUANTUM,
    CmdFamily, CmdCode
)
from et_logger import ETLog

# Segment selector constants (ET-derived: CS register values)
CS_32BIT = 0x23   # GDT[4], L=0: 32-bit compatibility mode
CS_64BIT = 0x33   # GDT[6], L=1: 64-bit long mode

# Windows API constants
MEM_COMMIT   = 0x1000
MEM_RESERVE  = 0x2000
MEM_RELEASE  = 0x8000
MEM_DECOMMIT = 0x4000
PAGE_EXECUTE_READWRITE = 0x40
PAGE_READWRITE = 0x04
PROCESS_ALL_ACCESS = 0x1F0FFF

kernel32 = getattr(ctypes.windll, 'kernel32')
ntdll    = getattr(ctypes.windll, 'ntdll')


# =============================================================================
# SHELLCODE STUBS — x86 assembly as bytes
# =============================================================================
# All stubs are position-independent. Addresses are patched at injection time.
# These run INSIDE the 32-bit WOW64 process.
#
# ET derivation of why these work:
#   The 32-bit process's P-space (registers, stack) is a SUBSET of P_64.
#   EAX in 32-bit mode IS the low 32 bits of RAX in 64-bit mode.
#   ESP in 32-bit mode IS the low 32 bits of RSP in 64-bit mode.
#   The memory IS the same memory — only the D-constraint (CS L-bit) changes.
#
# After entering 64-bit mode through CS=0x33:
#   - All 16 64-bit registers (RAX through R15) are accessible
#   - The address space is still the same (WOW64 shares the 4GB low space)
#   - The 64-bit ntdll is already mapped into the process (at address > 2GB)
# =============================================================================


def make_heaven_gate_switch_stub() -> bytes:
    """
    The core Heaven's Gate transition stub (x86, ~20 bytes).
    This is the base mechanism for all other stubs.

    Input:  EAX = address of 64-bit code to execute
    Effect: switches to CS=0x33 (64-bit mode) and JMPs to EAX
    Note:   this stub is a fragment — it's embedded in larger stubs

    x86 bytes:
      60           pushad          ; save all 32-bit registers
      9C           pushfd          ; save EFLAGS
      54           push esp        ; save 32-bit ESP for later restoration
      83 C4 F8     add esp, -8     ; allocate 8 bytes for the far pointer
      89 04 24     mov [esp+0], eax; store target address (low 32)
      C7 44 24 04  mov [esp+4], 0x33  ; store segment 0x33
      33 00 00 00
      FF 2C 24     jmp far [esp]   ; ENTER 64-bit mode
    """
    stub = bytes([
        0x60,               # pushad
        0x9C,               # pushfd
        0x54,               # push esp
        0x83, 0xC4, 0xF8,  # add esp, -8
        0x89, 0x04, 0x24,  # mov [esp+0], eax
        0xC7, 0x44, 0x24, 0x04, 0x33, 0x00, 0x00, 0x00,  # mov [esp+4], 0x33
        0xFF, 0x2C, 0x24,  # jmp far [esp]
    ])
    return stub


def make_return_to_32bit_stub() -> bytes:
    """
    Return from 64-bit to 32-bit mode stub (x64, ~10 bytes).
    Must be placed in executable memory within the 32-bit WOW64 process.
    Restores the 32-bit ESP and returns through CS=0x23.

    x64 bytes:
      4C 89 C4     mov rsp, r8     ; restore 32-bit ESP (passed in R8)
      48 83 C4 08  add rsp, 8      ; skip the saved ESP itself
      9D           popfd           ; restore EFLAGS (32-bit)
      61           popad           ; restore 32-bit registers
      CB           retf            ; far return to CS=0x23 (32-bit)
    """
    stub = bytes([
        0x4C, 0x89, 0xC4,       # mov rsp, r8
        0x48, 0x83, 0xC4, 0x08, # add rsp, 8
        0x9D,                   # popfd
        0x61,                   # popad
        0xCB,                   # retf
    ])
    return stub


def make_virtualalloc64_stub(
    ntdll64_ntalloc_addr: int,
    return_32bit_code_addr: int,
    use_nt_syscall: bool = False
) -> bytes:
    """
    64-bit VirtualAlloc stub for high memory allocation.
    Runs in 64-bit mode after Heaven's Gate transition.

    Two implementation paths (ET Identification Principle: same P, different D):
      use_nt_syscall=False (default): uses kernel32!VirtualAlloc (simpler, d=1)
      use_nt_syscall=True:            uses ntdll!NtAllocateVirtualMemory (direct, d=12)

    Calling convention (x64): RCX, RDX, R8, R9 for first 4 args.
    We want to call: NtAllocateVirtualMemory(
        ProcessHandle = -1 (current process),
        BaseAddress   = &base,
        ZeroBits      = 0,
        RegionSize    = &size,
        AllocationType = MEM_RESERVE | MEM_COMMIT,
        Protect       = PAGE_READWRITE
    )

    Args passed from 32-bit side:
      EAX = requested_size (in bytes)
      ECX = allocation_type flags (MEM_RESERVE | MEM_COMMIT)
      EDX = protect flags (PAGE_READWRITE)

    Returns in EAX: low 32 bits of allocated address (or bridge handle)
    Returns in EDX: high 32 bits of allocated address

    This stub is position-independent except for the two absolute addresses
    passed as parameters (ntdll64 NtAllocateVirtualMemory and return stub).
    """
    # In x64, RCX = process handle, RDX = &baseAddr, R8 = zeroBits,
    # R9 = &regionSize, stack+0x20 = allocType, stack+0x28 = protect
    # We'll use a simplified VirtualAllocEx-equivalent call pattern.

    # Patch addresses into the stub
    alloc_addr_bytes   = struct.pack("<Q", ntdll64_ntalloc_addr)
    return_addr_bytes  = struct.pack("<Q", return_32bit_code_addr)

    # =========================================================================
    # PATH 1: NT direct syscall stub (NtAllocateVirtualMemory)
    # Used when use_nt_syscall=True — lower-level, bypasses kernel32 layer.
    # ET derivation: d=12 (manifold-complete) — direct NT system call.
    # =========================================================================

    # x64 NT stub preamble: save non-volatile registers
    stub_nt_preamble = bytearray([
        # Save x64 non-volatile registers we'll clobber
        0x41, 0x57,             # push r15
        0x41, 0x56,             # push r14
        0x57,                   # push rdi
        0x56,                   # push rsi

        # EAX contains requested_size from 32-bit caller.
        # In x64, writing a 32-bit register zero-extends to 64-bit automatically.
        # Explicit zero-extension: mov eax, eax (ensures RAX = ZeroExtend(EAX))
        0x89, 0xC0,             # mov eax, eax  (zero-extends EAX → RAX)
        0x90, 0x90,             # nop; nop (alignment padding)
    ])
    # Continue NT preamble: allocate stack space and store region size
    stub_nt_preamble += bytes([
        # Allocate shadow space + region_size storage on stack
        0x48, 0x83, 0xEC, 0x48,  # sub rsp, 0x48 (shadow + local vars)

        # Store requested size in local var at [rsp+0x28]
        0x48, 0x89, 0x44, 0x24, 0x28,  # mov [rsp+0x28], rax (region size)

        # Setup NtAllocateVirtualMemory args:
        #   RCX = handle (-1 = current process)
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF,  # mov rcx, -1
    ])

    # Full x64 NtAllocateVirtualMemory call setup:
    # This is the complete NT-level stub body:
    stub_nt_body = bytes([
        # Sub RSP for shadow space (required by x64 calling convention)
        0x48, 0x83, 0xEC, 0x58,  # sub rsp, 0x58

        # RCX = NtCurrentProcess = (HANDLE)-1 = 0xFFFFFFFFFFFFFFFF
        0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF,  # mov rcx, -1

        # RDX = &BaseAddress (pointer to pointer, we use RSP+0x40 as scratch)
        0x48, 0x31, 0xC0,              # xor rax, rax
        0x48, 0x89, 0x44, 0x24, 0x40,  # mov [rsp+0x40], rax  (BaseAddress hint = 0)
        0x48, 0x8D, 0x54, 0x24, 0x40,  # lea rdx, [rsp+0x40]

        # R8 = ZeroBits = 0
        0x4D, 0x31, 0xC0,              # xor r8, r8

        # R9 = &RegionSize (we use rsp+0x48 as scratch, value passed in EAX from 32-bit)
        # EAX was set by the 32-bit caller. In 64-bit mode, writing to a 32-bit
        # register zero-extends to the full 64-bit register automatically.
        # So RAX already = ZeroExtend(EAX). Explicit: mov eax, eax
        0x89, 0xC0,                    # mov eax, eax (zero-extend EAX → RAX)
        0x48, 0x89, 0x44, 0x24, 0x48,  # mov [rsp+0x48], rax
        # lea r9, [rsp+0x48] — REX.WR prefix 0x4C for R9 destination
        0x4C, 0x8D, 0x4C, 0x24, 0x48,  # lea r9, [rsp+0x48]
    ])

    # =========================================================================
    # PATH 2: Simple VirtualAlloc stub (kernel32 level)
    # Used when use_nt_syscall=False (default) — higher-level, simpler.
    # ET derivation: d=1 (octave, fundamental) — standard allocation.
    # =========================================================================

    # SIMPLEST working approach: call the 64-bit kernel32!VirtualAlloc
    # which IS loaded in WOW64 processes (at high address).
    # Stub: receives (size: EAX, allocType: ECX, protect: EDX)
    # Calls VirtualAlloc64(0, size, allocType, protect)
    # Returns RAX = 64-bit base address

    stub_simple = bytes([
        0x48, 0x83, 0xEC, 0x28,         # sub rsp, 0x28 (shadow space)

        # arg1 RCX = lpAddress = 0 (let system choose)
        0x48, 0x31, 0xC9,               # xor rcx, rcx

        # arg2 RDX = dwSize = RAX (zero-extended from EAX, set by 32-bit caller)
        0x48, 0x89, 0xC2,               # mov rdx, rax

        # arg3 R8 = dwAllocationType (from ECX, zero-extended)
        0x4C, 0x8B, 0xC1,               # mov r8, rcx ... WAIT ecx conflicts

        # Better: use different registers. The 32-bit side passes:
        #   [ESP+4] = size, [ESP+8] = allocType, [ESP+12] = protect
        # In 64-bit mode after Heaven's Gate, RSP points to same stack.
        # [RSP+4] = size, [RSP+8] = allocType, [RSP+12] = protect (32-bit values)

        # arg1 RCX = 0
        0x48, 0x31, 0xC9,               # xor rcx, rcx
        # arg2 RDX = size (from [RSP+4])
        0x8B, 0x54, 0x24, 0x04,         # mov edx, [rsp+4] (zero extends to RDX)
        # arg3 R8 = allocType (from [RSP+8])
        0x44, 0x8B, 0x44, 0x24, 0x08,  # mov r8d, [rsp+8]
        # arg4 R9 = protect (from [RSP+12])
        0x44, 0x8B, 0x4C, 0x24, 0x0C,  # mov r9d, [rsp+12]

        # CALL VirtualAlloc (address patched below)
        0xFF, 0x15, 0x02, 0x00, 0x00, 0x00,  # call [rip+2]
        0xEB, 0x08,                           # jmp over the address
    ])
    # Append VirtualAlloc64 address (8 bytes)
    stub_simple += alloc_addr_bytes

    # Return: RAX = allocated 64-bit address.
    # Store result back in [RSP+0] so 32-bit caller can read it.
    stub_simple += bytes([
        0x48, 0x89, 0x04, 0x24,         # mov [rsp], rax   (store for 32-bit side)
        0x48, 0x83, 0xC4, 0x28,         # add rsp, 0x28   (cleanup shadow)

        # Return to 32-bit mode: push CS=0x23 and return address, then retf
        0x68, 0x23, 0x00, 0x00, 0x00,   # push 0x23 (32-bit CS)
        # push the return address (patched by the calling stub)
        0xFF, 0x35, 0x02, 0x00, 0x00, 0x00,  # push [rip+2]
        0xEB, 0x08,                           # jmp over
    ])
    stub_simple += return_addr_bytes

    stub_simple += bytes([
        0xCB,                            # retf (far return to 32-bit mode)
    ])

    # Select stub path: NT syscall (d=12, manifold-complete) or VirtualAlloc (d=1, fundamental)
    if use_nt_syscall:
        # NT path: use complete preamble + body for direct NtAllocateVirtualMemory
        return bytes(stub_nt_preamble) + stub_nt_body
    return stub_simple


# =============================================================================
# HIGH-LEVEL HEAVEN'S GATE INTERFACE (runs in the 64-bit BROKER process)
# =============================================================================

class ETHeavenGate:
    """
    Manages Heaven's Gate transitions on behalf of 32-bit processes.

    This class runs in our 64-bit broker. It provides:
    1. Resolution of 64-bit ntdll function addresses
    2. Writing and executing Heaven's Gate shellcode in 32-bit WOW64 processes
    3. 64-bit VirtualAlloc for high-memory allocations
    4. 64-bit DLL loading and GetProcAddress

    ET derivation: The broker IS the T-component that traverses through the
    Heaven's Gate (D = GDT entry 0x33) to provide 64-bit capabilities.
    """

    def __init__(self):
        self._log = ETLog("HeavenGate")
        # Validate platform: Heaven's Gate is Windows-only (GDT segments)
        if sys.platform != 'win32':
            raise OSError("ETHeavenGate requires Windows (GDT CS=0x33 transition)")
        # Validate 64-bit broker: sys.maxsize > 2^32 only in 64-bit Python
        if sys.maxsize <= 0xFFFFFFFF:
            self._log.warning(
                "HeavenGate running in 32-bit Python — "
                "direct 64-bit operations will be limited"
            )
        self._broker_pid: int = os.getpid()
        self._ntdll64_base: int = 0
        self._k32_64_base: int = 0
        self._func_cache: Dict[str, int] = {}
        self._init_64bit_modules()

    @property
    def module_bases(self) -> Optional[Tuple[int, int]]:
        """
        Return the (ntdll64_base, kernel32_64_base) pair, or None if not resolved.
        ET derivation: the two bases form a {D,T} Mediation pair —
        ntdll is the D (system call Descriptor layer),
        kernel32 is the T (user-mode Traverser layer).
        """
        if self._ntdll64_base and self._k32_64_base:
            return self._ntdll64_base, self._k32_64_base
        return None

    def _init_64bit_modules(self):
        """
        Locate the 64-bit ntdll and kernel32 in the current (64-bit) process.
        We ARE the 64-bit broker, so we directly use ctypes.
        """
        try:
            self._ntdll64_base = ctypes.cast(
                getattr(ntdll, '_handle'),    # _handle is the ctypes HMODULE — access via getattr per project pattern
                ctypes.c_void_p
            ).value or 0
            self._k32_64_base = ctypes.cast(
                getattr(kernel32, '_handle'),  # getattr avoids direct protected-member access
                ctypes.c_void_p
            ).value or 0
            self._log.info(
                f"HeavenGate broker PID={self._broker_pid}: "
                f"ntdll64=0x{self._ntdll64_base:016X}, "
                f"kernel32_64=0x{self._k32_64_base:016X}"
            )
        except Exception as e:
            self._log.error(f"Failed to locate 64-bit modules: {e}")

    @property
    def ntdll64_base(self) -> int:
        """Public accessor for the resolved ntdll64.dll base address (0 if unresolved)."""
        return self._ntdll64_base

    def get_proc_address_64(self, module_name: str, proc_name: str) -> int:
        """
        Get the 64-bit address of a function in a 64-bit DLL.
        Returns 0 if not found.

        This runs in our 64-bit process, so ctypes.windll works directly.
        """
        cache_key = f"{module_name}:{proc_name}"
        if cache_key in self._func_cache:
            return self._func_cache[cache_key]

        try:
            mod = ctypes.windll.LoadLibrary(module_name)
            addr = ctypes.cast(
                getattr(mod, proc_name),
                ctypes.c_void_p
            ).value or 0
            if addr:
                self._func_cache[cache_key] = addr
            return addr
        except (AttributeError, OSError) as e:
            self._log.error(f"GetProcAddress64({module_name}, {proc_name}): {e}")
            return 0

    def virtual_alloc_64(
        self,
        size: int,
        alloc_type: int = MEM_RESERVE | MEM_COMMIT,
        protect: int = PAGE_READWRITE
    ) -> int:
        """
        Allocate memory in the 64-bit address space.
        Called from the 64-bit broker process.
        Returns 64-bit base address, or 0 on failure.

        Size is aligned up to DIGITAL_ACTION_QUANTUM (page boundary = 4096).
        Preferred base address is ADDR64_BASE (above 4GB) to guarantee
        no collision with any 32-bit pointer space.

        ET derivation:
          P = the full 64-bit address space (P_full, Ω-cardinality)
          D = (size, alloc_type, protect) — the allocation Descriptors
          T = VirtualAlloc call — the Traverser that actualizes the allocation
          E = the allocated region — Exception (grounded, V=0)
          CmdFamily.MEMORY_BASIC (d=1), CmdCode.VIRT_ALLOC (0x01)
        """
        # Align size up to digital action quantum (page boundary, ET-derived)
        aligned_size = (
            (size + DIGITAL_ACTION_QUANTUM - 1)
            // DIGITAL_ACTION_QUANTUM
            * DIGITAL_ACTION_QUANTUM
        )
        try:
            virtual_alloc = getattr(kernel32, 'VirtualAlloc')
            virtual_alloc.restype  = ctypes.c_void_p
            virtual_alloc.argtypes = [
                ctypes.c_void_p,   # lpAddress
                ctypes.c_size_t,   # dwSize
                ctypes.wintypes.DWORD,  # flAllocationType
                ctypes.wintypes.DWORD,  # flProtect
            ]
            # First attempt: prefer allocation above 4GB (ADDR64_BASE)
            # This guarantees the returned address is unreachable by 32-bit code,
            # making bridge handle distinction trivial.
            ptr = virtual_alloc(
                ctypes.c_void_p(ADDR64_BASE), aligned_size, alloc_type, protect
            )
            if ptr is None:
                # Fallback: let the OS choose any available address
                ptr = virtual_alloc(None, aligned_size, alloc_type, protect)
            if ptr is None:
                err = ctypes.GetLastError()
                self._log.error(
                    f"VirtualAlloc64 failed: size={size} "
                    f"(aligned={aligned_size}), error={err}",
                    variance=1.0, et_family=CmdFamily.MEMORY_BASIC
                )
                return 0
            addr = ctypes.c_void_p(ptr).value or 0
            self._log.debug(
                f"VirtualAlloc64 [d={CmdFamily.MEMORY_BASIC},"
                f"c=0x{CmdCode.VIRT_ALLOC:02X}]: "
                f"addr=0x{addr:016X}, size=0x{aligned_size:X}",
                variance=V_BASE, et_family=CmdFamily.MEMORY_BASIC
            )
            return addr
        except Exception as e:
            self._log.error(
                f"VirtualAlloc64 exception: {e}",
                et_family=CmdFamily.MEMORY_BASIC
            )
            return 0

    def virtual_free_64(self, addr: int, size: int = 0, free_type: int = MEM_RELEASE) -> bool:
        """
        Free a 64-bit allocation.
        free_type: MEM_RELEASE (0x8000) or MEM_DECOMMIT (0x4000).
        CmdFamily.MEMORY_BASIC (d=1), CmdCode.VIRT_FREE (0x02).
        """
        try:
            virtual_free = getattr(kernel32, 'VirtualFree')
            virtual_free.restype  = ctypes.wintypes.BOOL
            virtual_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.wintypes.DWORD]
            result = virtual_free(ctypes.c_void_p(addr), size, free_type)
            if result:
                self._log.debug(
                    f"VirtualFree64 [d={CmdFamily.MEMORY_BASIC},"
                    f"c=0x{CmdCode.VIRT_FREE:02X}]: "
                    f"addr=0x{addr:016X}",
                    variance=V_BASE, et_family=CmdFamily.MEMORY_BASIC
                )
            return bool(result)
        except Exception as e:
            self._log.error(
                f"VirtualFree64 exception: {e}",
                et_family=CmdFamily.MEMORY_BASIC
            )
            return False

    def load_library_64(self, dll_path: str) -> int:
        """
        Load a 64-bit DLL into the broker process and return its base address.
        64-bit DLLs can ONLY be loaded into 64-bit processes (us, the broker).
        The 32-bit target accesses them through the bridge.
        Relative paths are resolved to absolute via os.path.abspath.
        CmdFamily.DLL_OPS (d=4), CmdCode.DLL_LOAD (0x31).
        """
        # Resolve relative DLL paths to absolute (ET: remove ambiguity from D-set)
        if not os.path.isabs(dll_path):
            dll_path = os.path.abspath(dll_path)
        try:
            load_library_w = getattr(kernel32, 'LoadLibraryW')
            load_library_w.restype  = ctypes.c_void_p
            load_library_w.argtypes = [ctypes.c_wchar_p]
            handle = load_library_w(dll_path)
            if not handle:
                err = ctypes.GetLastError()
                self._log.error(
                    f"LoadLibrary64({dll_path}) failed: error={err}",
                    et_family=CmdFamily.DLL_OPS
                )
                return 0
            base = ctypes.c_void_p(handle).value or 0
            self._log.info(
                f"LoadLibrary64 [d={CmdFamily.DLL_OPS},"
                f"c=0x{CmdCode.DLL_LOAD:02X}]: "
                f"{dll_path} → base=0x{base:016X}",
                et_family=CmdFamily.DLL_OPS
            )
            return base
        except Exception as e:
            self._log.error(
                f"LoadLibrary64 exception: {e}",
                et_family=CmdFamily.DLL_OPS
            )
            return 0

    def get_proc_from_loaded_dll(self, dll_base: int, proc_name: str) -> int:
        """
        Get a function address from an already-loaded 64-bit DLL.
        dll_base is the HMODULE (LoadLibrary result).
        CmdFamily.DLL_OPS (d=4), CmdCode.DLL_GETPROC (0x33).
        """
        try:
            get_proc_address = getattr(kernel32, 'GetProcAddress')
            get_proc_address.restype  = ctypes.c_void_p
            get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            addr = get_proc_address(
                ctypes.c_void_p(dll_base),
                proc_name.encode("ascii")
            )
            resolved = ctypes.c_void_p(addr).value or 0 if addr else 0
            if resolved:
                self._log.debug(
                    f"GetProcFromDLL [d={CmdFamily.DLL_OPS},"
                    f"c=0x{CmdCode.DLL_GETPROC:02X}]: "
                    f"{proc_name} → 0x{resolved:016X}",
                    variance=V_BASE, et_family=CmdFamily.DLL_OPS
                )
            return resolved
        except Exception as e:
            self._log.error(
                f"GetProcFromDLL exception: {e}",
                et_family=CmdFamily.DLL_OPS
            )
            return 0

    def call_64bit_function(
        self,
        func_addr: int,
        args: list,
        restype=ctypes.c_uint64
    ) -> int:
        """
        Call an arbitrary 64-bit function by address from the broker process.
        Supports up to 12 arguments (S = 12 = manifold symmetry).
        Arguments beyond K×S (Koide-stable limit = 8) are logged as a warning.

        This is the core of the d=4 (DLL_CALL) command family.

        ET derivation:
          func_addr is a P-address in P_full (the infinite address space)
          args are the D-set (the arguments = Descriptors for this call)
          The calling mechanism is T (the traverser that actualizes the function)
          CmdFamily.DLL_OPS (d=4), CmdCode.DLL_CALL (0x34)
        """
        if not func_addr or len(args) > S:
            return 0

        # Koide stability check: more than K×S args approaches instability
        koide_stable_limit = int(S * K)
        if len(args) > koide_stable_limit:
            self._log.warning(
                f"Call64 at 0x{func_addr:016X}: {len(args)} args exceeds "
                f"Koide-stable limit ({koide_stable_limit})",
                et_family=CmdFamily.DLL_OPS
            )

        # Build a ctypes function prototype dynamically
        arg_types = []
        for arg in args:
            if isinstance(arg, int):
                if 0 <= arg <= 0xFFFFFFFF:
                    arg_types.append(ctypes.c_uint32)
                else:
                    arg_types.append(ctypes.c_uint64)
            elif isinstance(arg, float):
                arg_types.append(ctypes.c_double)
            elif isinstance(arg, bytes):
                arg_types.append(ctypes.c_char_p)
            elif isinstance(arg, str):
                arg_types.append(ctypes.c_wchar_p)
            else:
                arg_types.append(ctypes.c_uint64)

        try:
            func_type = ctypes.CFUNCTYPE(restype, *arg_types)
            func_ptr  = func_type(func_addr)
            result = func_ptr(*args)
            ret_val = int(result) if result is not None else 0
            self._log.debug(
                f"Call64 [d={CmdFamily.DLL_OPS},c=0x{CmdCode.DLL_CALL:02X}] "
                f"at 0x{func_addr:016X} → 0x{ret_val:X} ({len(args)} args)",
                variance=V_BASE, et_family=CmdFamily.DLL_OPS
            )
            return ret_val
        except Exception as e:
            self._log.error(
                f"Call64 at 0x{func_addr:016X} exception: {e}",
                et_family=CmdFamily.DLL_OPS
            )
            return 0

    def write_executable_memory(
        self,
        target_pid: int,
        code: bytes
    ) -> int:
        """
        Write executable shellcode into a target process.
        Returns the 32-bit address within the target where code was written.
        Uses VirtualAllocEx + WriteProcessMemory.
        CmdFamily.PROCESS_OPS (d=5), CmdCode.PROC_INJECT (0x43).
        """
        # Resolve kernel32 functions dynamically (ET: D-access through getattr)
        open_process = getattr(kernel32, 'OpenProcess')
        close_handle = getattr(kernel32, 'CloseHandle')
        try:
            h = open_process(PROCESS_ALL_ACCESS, False, target_pid)
            if not h:
                self._log.error(
                    f"OpenProcess({target_pid}) failed: {ctypes.GetLastError()}",
                    et_family=CmdFamily.PROCESS_OPS
                )
                return 0

            virtual_alloc_ex = getattr(kernel32, 'VirtualAllocEx')
            virtual_alloc_ex.restype  = ctypes.c_void_p
            virtual_alloc_ex.argtypes = [
                ctypes.wintypes.HANDLE,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.wintypes.DWORD,
                ctypes.wintypes.DWORD,
            ]

            # Allocate within low 4GB (accessible from 32-bit code)
            # For WOW64 processes, allocations by default go into the low 4GB.
            code_addr = virtual_alloc_ex(
                h, None, len(code),
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE
            )

            if not code_addr:
                close_handle(h)
                self._log.error(
                    f"VirtualAllocEx in PID {target_pid} failed: "
                    f"{ctypes.GetLastError()}",
                    et_family=CmdFamily.PROCESS_OPS
                )
                return 0

            addr_val = ctypes.c_void_p(code_addr).value or 0

            written = ctypes.c_size_t(0)
            write_process_memory = getattr(kernel32, 'WriteProcessMemory')
            ok = write_process_memory(
                h,
                ctypes.c_void_p(addr_val),
                ctypes.c_char_p(code),
                len(code),
                ctypes.byref(written)
            )

            close_handle(h)

            if not ok or written.value != len(code):
                self._log.error(
                    f"WriteProcessMemory failed: wrote {written.value}/{len(code)}",
                    et_family=CmdFamily.PROCESS_OPS
                )
                return 0

            self._log.debug(
                f"Wrote {len(code)}B to PID {target_pid} at 0x{addr_val:08X} "
                f"[d={CmdFamily.PROCESS_OPS},c=0x{CmdCode.PROC_INJECT:02X}]",
                variance=V_BASE, et_family=CmdFamily.PROCESS_OPS
            )
            return addr_val & 0xFFFFFFFF

        except Exception as e:
            self._log.error(
                f"write_executable_memory exception: {e}",
                et_family=CmdFamily.PROCESS_OPS
            )
            return 0

# Module-level assertion: verify PROCESS_ALL_ACCESS is consistent
# (canonical definition is in the constants section above)
assert PROCESS_ALL_ACCESS == 0x1F0FFF, "PROCESS_ALL_ACCESS constant mismatch"
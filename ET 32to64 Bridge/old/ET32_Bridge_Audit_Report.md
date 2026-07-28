# ET32 Bridge — Comprehensive Audit Report
### P ∘ D ∘ T = E | Michael James Muller / Aevum Defluo
### Audit Standard: Professional | ET Tools Applied: Identification Principle, Descriptor Gap Principle, Subsumption Law

---

## Audit Scope

All 24 files audited in full. Every issue below has been fully traced from source to consequence, verified against the codebase, and logged individually. No assumptions made — every finding is proven by reading.

**Files Audited:**
`et_bridge32.c` (3,777 lines) · `et_host64.py` (3,553) · `et_ipc.py` (1,287) · `et_api.py` (1,049) · `et_injector.py` (1,118) · `et_wow64.py` (1,229) · `et_awe.py` (1,044) · `et_math.py` (1,757) · `et_errors.py` (1,085) · `et_config.py` (976) · `et_logger.py` (814) · `et_handle.py` (762) · `et_monitor.py` (598) · `et_heaven.py` (735) · `et32_bridge_main.py` (486) · `et32_bridge_helper.py` (689) · `__init__.py` · `main.py` · `et32_bridge.spec` · `et32_bridge_helper.spec` · `CMakeLists.txt` · `main.cpp` · `build.bat` · `config_template.json`

**Audit Criteria Applied:**
1. Race conditions
2. Anything not fully implemented
3. Placeholders, dummies, mocks, temporaries
4. Mission completeness: does every 32-bit process (including children) gain true 64-bit capability?
5. Error logging completeness: all errors from attached programs captured
6. Console CLI completeness: persistent console, full CLI, help menu, attachment feedback, live metrics

---

## Issue Index

| ID | Severity | Category | File(s) | Short Title |
|----|----------|----------|---------|-------------|
| ISSUE-01 | **CRITICAL** | Placeholder | `et_injector.py:441–521` | Dispatcher shellcode is `xor eax, eax; ret` — no pipe communication |
| ISSUE-01B | **CRITICAL** | Bug | `et_injector.py:411–414` | `ETStubGenerator` uses wrong call opcode — guaranteed crash |
| ISSUE-02 | **CRITICAL** | Not Implemented | `et_injector.py`, `et_api.py` | `et_bridge32.dll` is never injected into any target process |
| ISSUE-03 | **CRITICAL** | Consequence of ISSUE-02 | `et_wow64.py:1009–1013` | WOW64 hook always installs fail-safe stub — never the real hook |
| ISSUE-04 | **CRITICAL** | Crash | `et_bridge32.c:2909, 2970, 3008` | `g_kifastsystemcall_trampoline` NULL dereference |
| ISSUE-05 | **CRITICAL** | Dead Code | `et_host64.py:2438, 2521` | `COMPOUND_BATCH` unreachable — cmd code 0xB1 consumed by `DYNAMIC_SYSCALL` early return |
| ISSUE-06 | **HIGH** | Not Implemented | `et32_bridge_main.py` | No interactive console CLI — audit point 6 unmet |
| ISSUE-07 | **HIGH** | Not Implemented | `et_host64.py:2473–2475` | Heaven's Gate dynamic syscall fallback searches non-existent function names |
| ISSUE-08 | **HIGH** | Incomplete / False Success | `et_injector.py:938` | `_inject_shellcode_fallback()` returns `True` when IAT hooks are NOT installed |
| ISSUE-09 | **HIGH** | Wrong Build Target | `CMakeLists.txt`, `main.cpp` | CMake builds an EXE placeholder, not `et_bridge32.dll` |
| ISSUE-10 | **MEDIUM** | Incomplete | `et_bridge32.c:3257–3265` | `ET32_GetNativeSystemInfo` discards broker response — `lpSystemInfo` never populated |
| ISSUE-11 | **MEDIUM** | Incomplete | `et_bridge32.c:3527–3535` | `ET32_accept` zeroes peer sockaddr — callers cannot retrieve peer address |
| ISSUE-12 | **MEDIUM** | Incomplete | `et_api.py:444–850` | `ETAPIGateway` missing handlers for most command families |
| ISSUE-13 | **MEDIUM** | Race Condition | `et_ipc.py:404–488` | `create_pipe_for_pid()` TOCTOU — duplicate pipe creation possible |
| ISSUE-14 | **MEDIUM** | Race Condition | `et_api.py:259–302` | `ETHookManager.engage()` TOCTOU — double-injection possible |
| ISSUE-15 | **MEDIUM** | Bug / Protocol | `et_bridge32.c:1040`, `et_ipc.py:74` | Pipe mode mismatch — broker creates BYTE pipe, DLL requests MESSAGE mode |
| ISSUE-16 | **LOW** | Performance Bug | `et_handle.py:264–279` | `project_address()` O(n) linear scan ignores the O(1) reverse dict |
| ISSUE-17 | **LOW** | Unresolved Warning | `et_bridge32.c:2293, 3083` | `g_awe_windows` CLion/Clangd warning unresolved — ReSharper comment ineffective for Clangd |
| ISSUE-18 | **MEDIUM** | Error Logging Gap | All files | Attached process stdout/stderr/Win32 exceptions not captured — audit point 5 partially unmet |
| ISSUE-19 | **LOW** | Inconsistency | `build.bat:170–177`, `et_bridge32.c:27` | `-lntdll` in DLL source comment but absent from `build.bat` MinGW command |

---

## Detailed Issue Reports

---

## ISSUE-01 — CRITICAL: Dispatcher Shellcode Is a Placeholder (`xor eax, eax; ret`)

**File:** `et_injector.py`
**Lines:** 441–521 (`make_dispatcher_shellcode`), invoked at line 729 (`_do_inject`)
**Category:** Placeholder / Not Implemented
**Audit Point:** 3 (placeholders), 4 (bridge mission)

### Full Trace

1. `ETHookManager.engage()` (et_api.py:287) calls `ETInjector.inject(pid, config)`.
2. `ETInjector.inject()` (et_injector.py:621) calls `_do_inject(h, pid, config, pipe_name)`.
3. `_do_inject()` (et_injector.py:729) calls `make_dispatcher_shellcode(hook_addr, hook_addr + OFFSET_RESULT_BUFFER)`.
4. `make_dispatcher_shellcode()` (et_injector.py:441–521) is the function in question.

### Evidence — Lines 492–521

```python
# The dispatcher itself is minimal — we rely on the stub calling our
# Python-side named pipe bridge. For simplicity, the actual communication
# happens through shared memory + event objects that are set up by the broker.
#
# The dispatcher writes to shared memory and signals an event.
# The broker reads from shared memory, processes, writes response, signals back.
#
# This avoids the need for the dispatcher to call CreateFile/WriteFile by itself.

# For the initial implementation, the executable portion returns 0 (passthrough).
# The pipe_name_bytes_addr, result_buffer_addr, and buffer_size are embedded as
# a data trailer (the dispatcher's own Descriptor set) at offset 4 after the
# executable code. A future pipe-connected dispatcher will read these fields
# to establish full IPC communication.

# Executable portion: xor eax, eax; ret
code = bytearray([
    0x31, 0xC0,   # xor eax, eax  (EAX = 0)
    0xC3          # ret
])
```

### Verification

The comment explicitly states:
- "For the **initial implementation**" — it is a placeholder.
- "A **future** pipe-connected dispatcher will read these fields."

The generated code is `0x31 0xC0 0xC3` = `xor eax, eax; ret` — 3 bytes that return 0 immediately with no pipe communication whatsoever.

### Consequence

Every IAT hook installed in a 32-bit target produces a stub that calls this dispatcher. The dispatcher returns 0 (EAX = 0 = "let original proceed") on every call. The bridge intercepts nothing. The 32-bit target's behaviour is entirely unmodified. The entire IAT hook layer is a no-op.

### ET Derivation

Identification Principle: P = the 32-bit target process, D = the dispatcher shellcode, T = the thread executing the hook stub. The D is incomplete (it has no traversal to the broker), so T cannot produce E (bridged execution). The gap IS the missing pipe-communication logic.

---

## ISSUE-01B — CRITICAL: `ETStubGenerator.make_stub()` Uses Wrong Call Opcode — Guaranteed Crash

**File:** `et_injector.py`
**Lines:** 411–414 (`make_stub`), 729–756 (`_do_inject` invocation)
**Category:** Bug
**Audit Point:** 2 (not fully implemented), 3 (broken implementation)

### Full Trace

1. `_do_inject()` creates `ETStubGenerator(dispatcher_addr=disp_addr & 0xFFFFFFFF, ...)` (line 755–757).
2. For each hooked IAT entry, it calls `gen.make_stub(stub_id, arg_count, family, code)` (line 791).
3. `make_stub()` emits:

```python
# call dispatcher (absolute indirect through pointer)
# FF 15 addr32 = call [addr32] — absolute indirect via memory
stub += bytes([0xFF, 0x15]) + struct.pack("<I", self.dispatcher_addr)
```

### Problem

`0xFF 0x15 <addr32>` is the x86 opcode for **`CALL DWORD PTR [addr32]`** — it reads 4 bytes from `addr32` in memory and calls *that address* as a function pointer.

`self.dispatcher_addr` = `disp_addr`, which is the start of the dispatcher CODE region. The first 4 bytes of the dispatcher code are `0x31, 0xC0, 0xC3, <data_byte>`. Interpreted as a little-endian 32-bit function pointer this is `0x??C3C031` — a garbage address pointing to unmapped memory.

### Verification

Step-by-step at runtime:
1. IAT hook fires → stub executes.
2. Stub executes `CALL DWORD PTR [disp_addr]`.
3. CPU reads 4 bytes at `disp_addr` = `{0x31, 0xC0, 0xC3, data}` = some arbitrary value like `0x12C3C031`.
4. CPU attempts to `CALL 0x12C3C031`.
5. That address is unmapped → **access violation** → target process crashes.

### Correct Pattern

The opcode should be `0xE8 <rel32>` (relative call to a known address), or the dispatcher address should be stored as a 4-byte pointer at a known writable location and the stub should do `CALL DWORD PTR [pointer_location]`.

### ET Derivation

Descriptor Gap Principle: The gap between "call to dispatcher code" and "CALL DWORD PTR [code]" IS a Descriptor. The missing Descriptor is the pointer-indirection level. `0xFF 0x15` requires a POINTER to the function, not the function address itself. The correct opcode for calling a known absolute address is `0xE8 <rel32>` (relative call).

---

## ISSUE-02 — CRITICAL: `et_bridge32.dll` Is Never Injected Into Any Target Process

**File:** `et_injector.py`, `et_api.py`, `et_host64.py`
**Category:** Not Implemented
**Audit Point:** 2, 4

### Full Trace

The bridge has two complete and separate implementations:

**Path A — C DLL (`et_bridge32.c`):**
- Fully functional: connects to broker pipe, implements all 12 command families, patches IAT with `ET32_*` wrappers, hooks `KiFastSystemCall` with `ET32_KiFastHook`, communicates via named pipe.
- Entry point: `ET32_Init(broker_pid)` (et_bridge32.c:2309) must be called after DLL load.
- This path is **never activated** from the Python broker.

**Path B — Python shellcode injection (`et_injector.py`):**
- Writes dispatcher shellcode (placeholder, ISSUE-01) and per-API stubs into target memory.
- Patches IAT to call stubs.
- Never loads `et_bridge32.dll`.

### Evidence

The only reference to `et_bridge32.dll` in the Python broker is in `get_universal_hook_addr()` (et_api.py:370–391):

```python
h = getattr(kernel32, 'LoadLibraryExA')(
    b'et_bridge32.dll', None,
    0x00000001  # DONT_RESOLVE_DLL_REFERENCES ← broker process only, read RVA
)
```

This loads the DLL into the **broker** (64-bit) process with `DONT_RESOLVE_DLL_REFERENCES`, reads one export RVA, and immediately calls `FreeLibrary`. It does not inject the DLL into any target.

There is no `CreateRemoteThread + LoadLibraryA` call anywhere in the 24 files. The `_dll_base_in_target` attribute accessed at et_api.py:366 is never set by the injector — `getattr(state.injector, '_dll_base_in_target', 0)` always returns 0.

### Consequence

Without DLL injection:
- `ET32_Init()` is never called → no pipe connection from inside the target.
- `ET32_SetKiFastTrampoline()` is never called → see ISSUE-04.
- The real `ET32_KiFastHook` is never installed → see ISSUE-03.
- All `ET32_*` IAT wrappers in the DLL are unreachable.
- The entire `et_bridge32.c` codebase (3,777 lines) is effectively dead code at runtime.

### ET Derivation

Identification Principle: P = the 32-bit target process, D = the bridge DLL providing capability extensions, T = the injection mechanism (CreateRemoteThread). T is entirely absent. Without T, P and D never combine — E (bridged execution) is impossible.

---

## ISSUE-03 — CRITICAL: WOW64 Hook Always Installs Fail-Safe (Pass-Through) Stub

**File:** `et_wow64.py`
**Lines:** 1009–1013 (`install()`)
**Category:** Incomplete — consequence of ISSUE-02
**Audit Point:** 2, 4

### Full Trace

`ETKiFastHook.install()` (et_wow64.py:916) writes to the target process:
1. A trampoline at `tramp_addr`: original 5 bytes of `KiFastSystemCall` + `JMP` back.
2. A "hook stub" at `stub_addr`:

```python
# Build hook stub: captures EAX, S args, calls ET32_UniversalHook in DLL
# ET32_UniversalHook address is resolved from the injected et_bridge32.dll
# For now, stub falls through to trampoline on any problem (fail-safe)
# Full stub installed by et_bridge32.dll's ET32_Init after injection
stub = self._make_failsafe_stub(stub_addr, tramp_addr)
```

`_make_failsafe_stub()` (et_wow64.py:1100–1108):

```python
@staticmethod
def _make_failsafe_stub(stub_addr: int, tramp_addr: int) -> bytes:
    """
    Minimal x86 fail-safe stub: just JMP to trampoline.
    The real hook logic is in et_bridge32.dll's ET32_KiFastHook
    (installed by ET32_Init after DLL injection).
    This fail-safe ensures nothing breaks if the DLL isn't loaded yet.
    """
    jmp_rel = (tramp_addr - (stub_addr + 5)) & 0xFFFFFFFF
    return struct.pack("<BI", JMP_REL32, jmp_rel)
```

3. Patches `KiFastSystemCall` prologue: `JMP → stub_addr`.

### What Actually Happens

`KiFastSystemCall` → `JMP stub` → `stub: JMP trampoline` → `trampoline: original bytes + JMP KiFastSystemCall+5`

This is a complete round-trip that changes nothing. Every syscall executes normally.

### Verification

The code comment states explicitly: "Full stub installed by `et_bridge32.dll's ET32_Init` after injection." Since `et_bridge32.dll` is never injected (ISSUE-02), the full stub is never installed. The fail-safe is permanently the only stub present.

---

## ISSUE-04 — CRITICAL: `g_kifastsystemcall_trampoline` NULL Dereference

**File:** `et_bridge32.c`
**Lines:** 2909 (init), 2970 (MSVC path), 3008 (GCC path)
**Category:** Crash
**Audit Point:** 2, 4

### Full Trace

Declaration (et_bridge32.c:2909):
```c
/* Saved original KiFastSystemCall bytes + JMP-back trampoline address */
static FARPROC g_kifastsystemcall_trampoline = NULL;
```

`ET32_SetKiFastTrampoline()` (et_bridge32.c:3029–3033) is the only way to set this:
```c
void WINAPI ET32_SetKiFastTrampoline(DWORD trampoline_addr)
{
    g_kifastsystemcall_trampoline = (FARPROC)(UINT_PTR)trampoline_addr;
}
```

`ET32_SetKiFastTrampoline` is listed as an export in `build.bat` but is **never called** from the Python broker. There is no call to it in `et_wow64.py`, `et_api.py`, `et_host64.py`, `et_injector.py`, or any other Python file.

### MSVC Pass-Through Path (et_bridge32.c:2970)

```asm
_hg_passthrough:
    popfd
    popad
    pop  ebp
    jmp  dword ptr [g_kifastsystemcall_trampoline]  ← NULL dereference
```

### GCC Pass-Through Path (et_bridge32.c:3008)

```asm
1:
    popfd
    popad
    pop  %%ebp
    jmp  *_g_kifastsystemcall_trampoline             ← NULL dereference
```

### Consequence

If the KiFastSystemCall hook fires and `ET32_DynamicDispatch` returns `0xC000001C` (pass-through), the hook jumps to NULL → access violation → target process crashes. This affects every non-routing syscall — the pass-through path is the MAJORITY of all syscalls.

### ET Derivation

Descriptor Gap Principle: The gap between the trampoline installation (broker Python side) and the trampoline address communication to the DLL (C side) is the missing Descriptor. `ET32_SetKiFastTrampoline` exists but is never called — the D-chain is broken at the handshake boundary.

---

## ISSUE-05 — CRITICAL: `COMPOUND_BATCH` Handler Is Unreachable Dead Code

**File:** `et_host64.py`
**Lines:** 2438 (early return), 2521 (dead handler)
**Category:** Dead Code / Command Code Collision
**Audit Point:** 2, 4

### Full Trace

In `_handle_compound_ops()`, the structure is:

```python
def _handle_compound_ops(self, pkt: ETPacket) -> ETPacket:
    args = unpack_args(pkt.payload)

    # ... CTRL_ERR handler (line ~2420) ...

    # DYNAMIC SYSCALL — CMD_COMPOUND_BATCH used for dynamic forwarding
    if pkt.cmd_code == CmdCode.COMPOUND_BATCH:          # ← line 2438, check cmd=0xB1
        ...
        return self._make_ok(pkt, result)                # ← returns here for ALL cmd=0xB1

    code = pkt.cmd_code                                  # ← line 2517, never reached if cmd=0xB1
    with self._family_locks[CmdFamily.COMPOUND_OPS]:
        if code == CmdCode.COMPOUND_BATCH:               # ← line 2521, also cmd=0xB1
            # Execute a batch of sub-packets               ← DEAD CODE
```

### Verification

`CmdCode.COMPOUND_BATCH = 0xB1` (et_math.py:964).
`CMD_DYNAMIC_SYSCALL = CMD_COMPOUND_BATCH = 0xB1` (et_bridge32.c:2801).

The check at line 2438 matches **all** packets with `cmd_code == 0xB1`. It always returns before reaching line 2517. Therefore the batch handler at line 2521 is permanently unreachable.

### Consequence

- `CmdCode.COMPOUND_BATCH` (batch multiple operations) → dead
- `CmdCode.COMPOUND_ATOMIC` (atomic multi-step) → the handler at line 2521 would also need to dispatch to COMPOUND_ATOMIC (0xB2) and COMPOUND_ROLLBACK (0xB3), but since the entire `with self._family_locks` block is unreachable, those too are dead
- Every batch/compound/atomic/rollback operation silently routes to the dynamic syscall dispatch path, receives `0xC000001C` (service not found), and is treated as pass-through

### ET Derivation

Identification Principle: Two distinct T-paths (DYNAMIC_SYSCALL dispatch and COMPOUND_BATCH execution) have been assigned the identical D (cmd code 0xB1). They cannot coexist. Subsumption Law: neither subsumes the other without remainder. The fix is to assign distinct command codes.

---

## ISSUE-06 — HIGH: No Interactive Console CLI (Audit Point 6 Entirely Unmet)

**File:** `et32_bridge_main.py`
**Lines:** All of `_main_loop()` (249–324), `main()` (331–482)
**Category:** Not Implemented
**Audit Point:** 6

### Full Trace

`_main_loop()` (et32_bridge_main.py:249):
- Runs a `while bridge_api.is_running` loop.
- Performs stability reporting, config reloads, and health checks.
- Blocks on `bridge_api.wait_for_stop(timeout=CONN_TIMEOUT_MS / 1000.0 / S)`.
- **There is no `input()` call, no readline loop, no command parser, no help command, no CLI thread anywhere in the file.**

`main()` (et32_bridge_main.py:331):
- Parses arguments, loads config, starts IPC, starts monitor.
- Prints the banner to stdout with `print(_BANNER)`.
- Calls `_main_loop()`.
- **There is no `AllocConsole()`, no `SetConsoleTitle()`, no `AttachConsole(ATTACH_PARENT_PROCESS)` call.**

### What Is Required (Audit Point 6)

The specification requires:
1. Console opened and kept open (does not close on Windows)
2. Full CLI with command parsing
3. `help` command listing all features and functions
4. Default dynamic/automatic behaviour linking to whitelist/blacklist items
5. Step-by-step attachment confirmation shown in console
6. Any errors or fallbacks shown in console
7. Live metrics (AWE status, Heaven's Gate status, hook status)
8. Manual commands for debugging

### What Is Present

A non-interactive loop that logs to file (and console handler streams). No user input is ever read. No command parser exists. No help text is available at runtime. The attachment process logs to the logger (file + stdout), but there is no structured console confirmation sequence.

### ET Derivation

Identification Principle: P = the operator, D = the CLI interface, T = the running bridge process. D (the CLI) is entirely absent. Without D, the operator (P) has no T to traverse — no way to interact with the running bridge. The gap IS the entire CLI subsystem.

---

## ISSUE-07 — HIGH: Heaven's Gate Dynamic Syscall Fallback Searches Non-Existent Function

**File:** `et_host64.py`
**Lines:** 2466–2496
**Category:** Not Implemented / Logic Error
**Audit Point:** 2

### Full Trace

In `_handle_compound_ops()`, after `ETWow64Hook.dispatch_service()` returns `None` (service not found in the NT service table), the broker attempts a Heaven's Gate fallback:

```python
if self._heaven is not None:
    try:
        nt_func_addr = self._heaven.get_proc_address_64(
            "ntdll.dll", f"__syscall_{service_number:04X}"   # ← line 2474
        )
```

### Verification

`get_proc_address_64(module, proc_name)` (et_heaven.py:395–417) calls `getattr(mod, proc_name)`. ntdll.dll exports no function named `__syscall_XXXX`. This will always raise `AttributeError`, which is caught at line 415 `except (AttributeError, OSError)` and returns 0.

Therefore `nt_func_addr` is always 0. The `if not nt_func_addr:` branch at line 2478 always fires, logging "Heaven's Gate fallback not resolvable" and falling through to pass-through.

The `else` branch at line 2488 (the actual Heaven's Gate call) is permanently unreachable.

### Consequence

The Heaven's Gate fallback for dynamic syscalls is a dead code path. Any syscall that the WOW64 service table misses is permanently a pass-through. The broker cannot use Heaven's Gate to fill gaps in the syscall coverage.

### Correct Approach

The broker already has `ETServiceTable` which maps service numbers to ntdll64 function addresses discovered by PE reflection. The correct fallback is to look up `service_number` in `self._wow64._svc_tbl` and call the ntdll64 function via `ETDynamic64Caller`. The `__syscall_XXXX` lookup is not a valid ntdll API surface.

---

## ISSUE-08 — HIGH: `_inject_shellcode_fallback()` Returns `True` When IAT Hooks Are NOT Installed

**File:** `et_injector.py`
**Lines:** 840–938
**Category:** Incomplete / False Success Return
**Audit Point:** 2, 3

### Full Trace

1. `_do_inject()` (et_injector.py:652) calls `_inject_no_module()` when the target module base cannot be found.
2. `_inject_no_module()` (et_injector.py:840) calls `_inject_shellcode_fallback()`.
3. `_inject_shellcode_fallback()` (et_injector.py:905+):

```python
return True  # partial success — hook_data written, pipe ready, IAT hooks not installed
```

4. Back in `ETHookManager.engage()` (et_api.py:289–293):

```python
inject_ok = injector.inject(pid, config)
if inject_ok:
    state.injector     = injector
    state.hooks_active = True       # ← WRONG: hooks are NOT active
```

### Verification

The comment at et_injector.py:938 explicitly states "IAT hooks not installed." Despite this, the method returns `True` (success). `ETHookManager.engage()` unconditionally sets `state.hooks_active = True` on any `True` return. Subsequent refresh calls check `state.hooks_active` and do not re-inject.

### Consequence

The broker believes injection succeeded and marks the process as fully hooked. No re-injection is attempted. The 32-bit process runs with hook_data written to its memory but zero IAT entries patched — completely unhooked. The operator sees no error.

---

## ISSUE-09 — HIGH: CMakeLists.txt Builds Wrong Target — EXE Placeholder, Not DLL

**Files:** `CMakeLists.txt` (9 lines), `main.cpp` (5 lines)
**Category:** Wrong Build Target / Placeholder
**Audit Point:** 2, 3

### Evidence

`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 4.2)
project(ET_32to64_Bridge)

set(CMAKE_CXX_STANDARD 20)

add_executable(ET_32to64_Bridge main.cpp
        et_bridge32.c)
```

`main.cpp`:
```cpp
#include <iostream>

int main() {
    std::cout << "Everything seems to be working!" << std::endl;
    return 0;
}
```

### Issues

1. `add_executable` builds a **Windows EXE**. `et_bridge32.c` must be compiled as a **32-bit DLL** (`add_library(et_bridge32 SHARED et_bridge32.c)` with 32-bit toolchain).
2. `main.cpp` is a placeholder with no relevance to the bridge. It cannot be compiled with `et_bridge32.c` in the same target — `et_bridge32.c` defines `DllMain`, not `main()`, and has `__declspec(dllexport)` symbols.
3. `cmake_minimum_required(VERSION 4.2)` requires CMake 4.2 — an extremely recent version (as of early 2026, CMake stable is 3.x). This will fail on most development machines.
4. The CMake target entirely ignores all required compiler flags: `-m32` (32-bit), `-shared`, `-Wl,--subsystem,windows`, `et_bridge32.def` for exports.

### Consequence

Running CMake to build `et_bridge32.dll` produces a non-functional EXE that prints "Everything seems to be working!" and exits. The correct build path is `build.bat` (Phase 1), but anyone using the IDE's CMake integration gets the wrong artefact entirely.

---

## ISSUE-10 — MEDIUM: `ET32_GetNativeSystemInfo` Discards Broker Response

**File:** `et_bridge32.c`
**Lines:** 3257–3265
**Category:** Incomplete Implementation
**Audit Point:** 2

### Evidence

```c
VOID WINAPI ET32_GetNativeSystemInfo(LPSYSTEM_INFO lpSystemInfo)
{
    uint8_t pbuf[ET_IPC_BUFFER_SIZE / 8]; et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, 0x08, pbuf, 0, &err);
    if (err && lpSystemInfo)
        GetNativeSystemInfo(lpSystemInfo); /* fallback */
}
```

### Trace

1. Broker receives cmd=0x08 (NATIVE_SYS_INFO), calls `_handle_native_sys_info()`.
2. `_handle_native_sys_info()` (et_host64.py) calls 64-bit `GetNativeSystemInfo`, packs the result into a response packet.
3. The broker sends back SYSTEM_INFO fields (processor architecture, page size, memory limits, core count).
4. `et_call()` returns the first uint32 arg (dwOemId/processor arch).
5. **The response args are never read** into `lpSystemInfo`. The struct is only populated via `GetNativeSystemInfo(lpSystemInfo)` on ERROR — the 32-bit (WOW64-filtered) version.

### Consequence

`ET32_GetNativeSystemInfo` never delivers the true 64-bit system info to the 32-bit caller. The caller always receives the WOW64-filtered view (e.g. `wProcessorArchitecture = PROCESSOR_ARCHITECTURE_INTEL = 0` instead of `PROCESSOR_ARCHITECTURE_AMD64 = 9`). Any 32-bit application checking processor architecture via this hook gets the wrong answer.

### Fix Required

After `et_call()` succeeds, use `et_recv_argreader` to read all SYSTEM_INFO fields from `g_recv_buf` and populate `lpSystemInfo`, exactly as done in `ET32_GlobalMemoryStatusEx` (et_bridge32.c:3231–3253).

---

## ISSUE-11 — MEDIUM: `ET32_accept()` Zeros Peer `sockaddr` — Callers Cannot Retrieve Peer Address

**File:** `et_bridge32.c`
**Lines:** 3516–3537
**Category:** Incomplete Implementation
**Audit Point:** 2, 4

### Evidence

```c
/* Initialize the OUT address structure for the caller.
 * The broker performed the actual accept(); the peer address is broker-side.
 * We zero-fill the caller's addr buffer to indicate the connection is bridged.
 * Callers that need the true peer address should query the broker directly. */
if (addr && addrlen && *addrlen >= (int)sizeof(struct sockaddr_in)) {
    memset(addr, 0, (size_t)*addrlen);          // ← zeroes the peer address
    addr->sa_family = AF_INET;
    *addrlen = (int)sizeof(struct sockaddr_in);
}
```

### Consequence

Any 32-bit server application that calls `accept(&sockaddr, &addrlen)` and then uses the peer address (for logging, access control, IP-based routing) receives `0.0.0.0:0` as the peer. This is behaviorally wrong. The comment acknowledges callers "should query the broker directly" but provides no API to do so.

### ET Derivation

The broker has the real peer address (it performed the 64-bit `accept()`). The Descriptor Gap is: the peer address must be returned from the broker response and written into `addr`. The `_handle_net_accept` handler in et_host64.py returns the peer address args (packed via `pack_args`), but they are never read by the C DLL.

---

## ISSUE-12 — MEDIUM: `ETAPIGateway` Missing Handlers for Most Command Families

**File:** `et_api.py`
**Lines:** 442–850
**Category:** Incomplete Implementation
**Audit Point:** 2, 4, 6

### Trace

`ETAPIGateway` is the high-level Python-side call surface used by `ET32Helper` (et32_bridge_helper.py) and intended for any 32-bit Python caller. It wraps `ETIPCClient.call()` with semantic method names.

### Implemented (partial)
- MEMORY_BASIC: `virtual_alloc`, `virtual_free`, `virtual_protect`, `read_memory`, `write_memory`
- MEMORY_MAP: `create_file_mapping`, `map_view_of_file`, `unmap_view_of_file`
- DLL_OPS: `load_library`, `free_library`, `get_proc_address`, `call_function`
- PROCESS_OPS: partial
- REGISTRY_OPS: `reg_open`, `reg_query`, `reg_set`
- PYTHON_OPS: `python_init`, `python_exec`, `python_eval`

### Missing (no gateway methods)
- **THREAD_OPS (d=3):** No `set_thread_context`, `get_exit_code_thread`
- **REGISTRY_OPS (d=6):** No `reg_enum64`, `reg_create64`, `reg_delete_key64`, `reg_delete_val64`, `reg_close64`
- **GRAPHICS_OPS (d=7):** No gateway for any GPU operation (alloc VRAM, map VRAM, submit, query, enum adapters, create device, heaven call)
- **FILE_OPS (d=8):** No gateway for large-file operations (seek, read, write, getsize, seteof, flush, gettime/settime, find)
- **SYNC_OPS (d=9):** No gateway for any synchronisation object (create event, signal, wait, mutex, semaphore, wait multiple, reset, close)
- **NET_OPS (d=10):** No gateway for any network operation (socket, bind, send, recv, connect, listen, accept, close, select, sockopt)
- **COMPOUND_OPS (d=12):** No gateway for batch/atomic/rollback (also blocked by ISSUE-05)

### Consequence

A 32-bit Python process using `ETAPIGateway` has access to ~25% of the bridge's command surface. Everything in GRAPHICS, FILE (large), SYNC, NET, and COMPOUND is unreachable except by constructing raw `ETPacket` objects manually. The 32-bit helper process (`ET32Helper`) can only dispatch the 4 inline args per slot (arg0–arg3) via the `CallSlot` structure, giving it no way to call the unimplemented families with the required arguments either.

---

## ISSUE-13 — MEDIUM: `create_pipe_for_pid()` TOCTOU Race Condition

**File:** `et_ipc.py`
**Lines:** 404–488
**Category:** Race Condition
**Audit Point:** 1

### Full Trace

```python
def create_pipe_for_pid(self, pid: int) -> bool:
    ...
    with self._pipe_lock:
        if pid in self._pipe_handles:     # CHECK — under lock
            return True
    # ← LOCK RELEASED HERE

    h_pipe = getattr(kernel32, 'CreateNamedPipeW')(...)  # CREATE — outside lock
    ...
    with self._conn_lock:                # REGISTER conn — different lock
        self._connections[h_pipe] = conn
    with self._pipe_lock:                # REGISTER handle — original lock re-acquired
        self._pipe_handles[pid] = h_pipe
```

### Race Scenario

1. Thread A: acquires `_pipe_lock`, `pid not in _pipe_handles` → True, releases lock.
2. Thread B: acquires `_pipe_lock`, `pid not in _pipe_handles` → True (A hasn't registered yet), releases lock.
3. Thread A: creates `h_pipe_A`, registers it.
4. Thread B: creates `h_pipe_B`, registers it → **overwrites** `_pipe_handles[pid] = h_pipe_B`.
5. `h_pipe_A` is now leaked: in `_connections` but no longer reachable via `_pipe_handles`.
6. `h_pipe_A` is never closed. IOCP receives completions for both handles, creating duplicate dispatch.

### When This Occurs

`create_pipe_for_pid` is called by `ETHookManager.engage()` (et_api.py:279) on the process-found callback path. If the monitor fires two callbacks for the same PID (e.g. one from the `_scan()` loop, one from `force_scan()` at startup), the race is possible.

---

## ISSUE-14 — MEDIUM: `ETHookManager.engage()` TOCTOU Race Condition

**File:** `et_api.py`
**Lines:** 259–302
**Category:** Race Condition
**Audit Point:** 1

### Full Trace

```python
def engage(self, pid: int, config: TargetConfig) -> bool:
    with self._lock:
        if pid in self._states:            # CHECK — under lock
            return True
        if len(self._states) >= QUEUE_DEPTH:
            return False
    # ← LOCK RELEASED HERE

    if not self._ipc.create_pipe_for_pid(pid):   # INJECT — outside lock
        return False

    state = ETHookManager.HookState(pid, config)
    injector = ETInjector(PIPE_NAME_TEMPLATE)
    inject_ok = injector.inject(pid, config)     # INJECT — outside lock

    ...
    with self._lock:
        self._states[pid] = state           # REGISTER — last writer wins
    return inject_ok
```

### Race Scenario

Two threads calling `engage(pid)` simultaneously both pass the `pid not in _states` check, both create pipes for `pid` (ISSUE-13), and both inject into the same process. The second injection overwrites `_states[pid]` — the first injector is leaked. The target process has double the expected IAT hooks.

### When This Occurs

`ETBridgeAPI.on_process_found()` is the callback handler. If the monitor fires two callbacks for the same PID within the injection window (common during startup `force_scan()`), both reach `engage()` concurrently.

---

## ISSUE-15 — MEDIUM: Pipe Mode Mismatch — BYTE vs MESSAGE

**File:** `et_bridge32.c:1040`, `et_ipc.py:74–76, 426–435`
**Category:** Protocol Bug
**Audit Point:** 2, 5

### Broker Side (et_ipc.py)

```python
PIPE_TYPE_BYTE      = 0x00000000
PIPE_READMODE_BYTE  = 0x00000000

h_pipe = getattr(kernel32, 'CreateNamedPipeW')(
    pipe_name,
    PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_WRITE_THROUGH,
    PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,   # ← BYTE mode
    ...
)
```

### C DLL Side (et_bridge32.c:1038–1041)

```c
if (h != INVALID_HANDLE_VALUE) {
    /* Set message-mode reading to match broker's write mode */
    DWORD mode = PIPE_READMODE_MESSAGE;        // 0x00000002
    SetNamedPipeHandleState(h, &mode, NULL, NULL);
```

### Verification

Windows documentation: `SetNamedPipeHandleState` with `PIPE_READMODE_MESSAGE` returns `ERROR_INVALID_PARAMETER` (87) if the server pipe was created as `PIPE_TYPE_BYTE`. You cannot switch a byte pipe to message read mode.

The return value of `SetNamedPipeHandleState` is not checked. The error is silently discarded. Both sides operate in byte mode — functionally the pipe works, but:
1. The comment in the C code ("Set message-mode reading to match broker's write mode") is wrong — the broker uses byte mode.
2. If future code relies on message-boundary semantics (e.g. partial-read detection), it will fail silently.
3. The explicit error is unchecked — the DLL has no way to know the mode change failed.

---

## ISSUE-16 — LOW: `HandleTable.project_address()` O(n) Linear Scan Ignores O(1) Reverse Dict

**File:** `et_handle.py`
**Lines:** 264–279
**Category:** Performance Bug / Logic Error
**Audit Point:** 2

### Evidence

```python
def project_address(self, addr64: int) -> int:
    """Project a 64-bit address to a 32-bit handle."""
    if addr64 < ADDR64_BASE:
        return addr64 & 0xFFFFFFFF

    with self._lock:
        # Find existing
        for entry in self._entries.values():   # ← O(n) scan
            if entry.addr64 == addr64:
                return entry.handle
    # Allocate new
    return self.alloc(addr64, 0, 0, CmdFamily.MEMORY_BASIC)
```

The class maintains `self._addr64_to_handle: Dict[int, int]` specifically for O(1) reverse lookup (added and documented at lines 130–137). `alloc()` and `dealloc()` maintain this dict correctly. `project_address()` bypasses it entirely.

### Consequence

As the handle table grows (up to 144 initial slots, expandable to millions), `project_address()` performance degrades linearly. `ETMarshal.result_to_addr32()` (et_api.py:121–148) calls `project_address()` on every broker response that contains a 64-bit address — meaning every VIRT_ALLOC, FILE_MAP_VIEW, and THREAD_CREATE response is O(n) instead of O(1).

### Fix

```python
with self._lock:
    existing = self._addr64_to_handle.get(addr64)
    if existing is not None:
        entry = self._entries.get(existing)
        if entry is not None:
            return existing
        del self._addr64_to_handle[addr64]   # stale entry cleanup
```

---

## ISSUE-17 — LOW: `g_awe_windows` CLion/Clangd Warning Unresolved — ReSharper Comment Ineffective

**File:** `et_bridge32.c`
**Lines:** 2293 (declaration), 3083–3085 (suppression comment)
**Category:** Unresolved Static Analysis Warning
**Audit Point:** Previous session unresolved item

### Evidence

Line 2293: `static et_awe_window g_awe_windows[AWE_MAX_WINDOWS];`

Line 3083–3085:
```c
// g_awe_windows is for a static array, and it is not unused.
// ReSharper disable once CppDeclaratorNeverUsed
DWORD base = (DWORD)(UINT_PTR)g_awe_windows[i].va_base;
```

### Problem

`// ReSharper disable once CppDeclaratorNeverUsed` is a **ReSharper-specific** inline comment suppression directive. CLion's Clangd-based static analyzer ignores this comment entirely — it uses a different diagnostic engine (`-Wunused-variable` or clangd's own checks).

The array IS used at: lines 2376–2380 (`ET32_Shutdown`), 2601–2653 (`et_awe_update_window`), 2674–2710 (`et_awe_unmap_window`, `et_awe_release_window`), 3085 (`et_veh_handler`), 3122 (`et_veh_handler`).

### Why the Warning Persists

Clangd does static control-flow analysis. Within the specific code path of the VEH handler at line 3083, the analyser may not trace through the complex condition at line 3082 (`for (int i = 0; i < g_n_awe_windows; i++)`) and may flag the local variable `base` at line 3085 as "computed but unused" rather than the global array.

### Fix

For Clangd/GCC: `(void)(g_awe_windows[0].va_base);` at file scope or a `__attribute__((used))` annotation on the declaration.
For ReSharper: the existing comment is correct for that analyzer.

The declaration should also carry a clarifying compile-time assertion:
```c
/* g_awe_windows is used in: ET32_Shutdown, et_awe_update_window,
   et_awe_unmap_window, et_awe_release_window, et_veh_handler. */
static et_awe_window g_awe_windows[AWE_MAX_WINDOWS];
```

---

## ISSUE-18 — MEDIUM: Error Logging Gap — Attached Process Errors Not Fully Captured (Audit Point 5)

**Files:** `et_bridge32.c`, `et_logger.py`, `et_errors.py`, `et32_bridge_main.py`
**Category:** Incomplete (error logging / audit point 5)
**Audit Point:** 5

### What Is Implemented

The bridge has strong error capture mechanisms:

- **C DLL `et_report_error()`** (et_bridge32.c:160–225): captures Win32 errors with location, function, OS code, and sends them as `CMD_CTRL_ERR` packets to the broker.
- **Broker `_handle_compound_ops` CTRL_ERR handler** (et_host64.py:2420–2435): receives C-side error reports and creates `ETWindowsAPIError` entries in the registry.
- **`ETErrorRegistry`** (et_errors.py:724): centralised, thread-safe error log.
- **`ETCrashLogger`** (et_logger.py:129): intercepts Python unhandled exceptions, C-level crashes (faulthandler), SIGTERM, Windows SEH.

### What Is NOT Captured

1. **Attached process stdout/stderr**: No `CreatePipe` + `SetStdHandle` injection to redirect the target's stdout/stderr through the broker. Any `printf` or `fprintf(stderr, ...)` in the 32-bit target is invisible to the broker.

2. **`OutputDebugString` from the attached process**: OutputDebugString writes to the kernel debug buffer. Capturing it requires either a `DebugActiveProcess` call (which would conflict with other debugger attachments and alters process behaviour) or polling `NtQuerySystemInformation(SystemKernelDebuggerInformation)`. Neither is implemented.

3. **Structured Exception Handling (SEH) codes from the target beyond access violations**: The VEH handler in et_bridge32.c only intercepts `EXCEPTION_ACCESS_VIOLATION` (line 3068). Other exception codes (`EXCEPTION_STACK_OVERFLOW`, `EXCEPTION_ILLEGAL_INSTRUCTION`, `EXCEPTION_INT_DIVIDE_BY_ZERO`, etc.) are not forwarded to the broker.

4. **Process exit code**: When a bridged process exits, `on_process_exit()` (et_api.py:902) records only the exe name and PID. The exit code is not retrieved via `GetExitCodeProcess` and not logged.

5. **Child process errors**: The monitor tracks child processes (et_monitor.py:418) and bridges them, but child stdout/stderr and exit codes are also not captured.

### Impact on Audit Point 5

The specification requires "all errors in all attached programs (including all children)" are captured. The broker captures Win32 API errors that the DLL explicitly reports via `et_report_error()`. Any error not going through that path — including CRT errors, application-level exceptions, and all output to stdout/stderr — is invisible to the broker.

---

## ISSUE-19 — LOW: `-lntdll` in DLL Source Comment but Absent from `build.bat` MinGW Command

**Files:** `et_bridge32.c:26–28`, `build.bat:170–177`
**Category:** Documentation/Build Inconsistency
**Audit Point:** 2

### Evidence

Source comment (et_bridge32.c:26–28):
```
 * Build (MinGW 32-bit):
 *   gcc -m32 -O2 -shared -o et_bridge32.dll et_bridge32.c \
 *       -lkernel32 -ladvapi32 -lntdll \
 *       -Wl,--subsystem,windows
```

Actual build.bat MinGW command (lines 170–177):
```
gcc -m32 -O2 -shared ^
    -o et_bridge32.dll ^
    et_bridge32.c ^
    et_bridge32.def ^
    -lkernel32 -ladvapi32 ^
    -Wl,--subsystem,windows ^
    -Wl,--enable-stdcall-fixup ^
    -Wall -Wextra -Wno-unused-parameter
```

`-lntdll` is absent from `build.bat`. In practice no ntdll symbols are imported directly in the C code (all ntdll interactions go through the broker), so the omission does not cause a link error. However the discrepancy between the documented build command and the actual build command is misleading and should be resolved by removing `-lntdll` from the source comment.

---

## Summary Table

| ID | Severity | File | Lines | Category | Verified |
|----|----------|------|-------|----------|---------|
| ISSUE-01 | **CRITICAL** | `et_injector.py` | 441–521 | Placeholder — no pipe comm | ✓ |
| ISSUE-01B | **CRITICAL** | `et_injector.py` | 411–414 | `FF 15` call conv bug — crash | ✓ |
| ISSUE-02 | **CRITICAL** | `et_injector.py`, `et_api.py` | (all) | DLL never injected | ✓ |
| ISSUE-03 | **CRITICAL** | `et_wow64.py` | 1009–1013 | Fail-safe stub only | ✓ |
| ISSUE-04 | **CRITICAL** | `et_bridge32.c` | 2909, 2970, 3008 | NULL trampoline crash | ✓ |
| ISSUE-05 | **CRITICAL** | `et_host64.py` | 2438, 2521 | COMPOUND_BATCH dead code | ✓ |
| ISSUE-06 | **HIGH** | `et32_bridge_main.py` | (all) | No CLI / console | ✓ |
| ISSUE-07 | **HIGH** | `et_host64.py` | 2473–2475 | HG fallback broken | ✓ |
| ISSUE-08 | **HIGH** | `et_injector.py` | 938 | False `True` on partial success | ✓ |
| ISSUE-09 | **HIGH** | `CMakeLists.txt`, `main.cpp` | (all) | Wrong build target | ✓ |
| ISSUE-10 | **MEDIUM** | `et_bridge32.c` | 3257–3265 | GetNativeSystemInfo discards response | ✓ |
| ISSUE-11 | **MEDIUM** | `et_bridge32.c` | 3527–3535 | accept() zeroes peer addr | ✓ |
| ISSUE-12 | **MEDIUM** | `et_api.py` | 444–850 | ETAPIGateway incomplete | ✓ |
| ISSUE-13 | **MEDIUM** | `et_ipc.py` | 404–488 | create_pipe_for_pid TOCTOU | ✓ |
| ISSUE-14 | **MEDIUM** | `et_api.py` | 259–302 | engage() TOCTOU | ✓ |
| ISSUE-15 | **MEDIUM** | `et_bridge32.c`, `et_ipc.py` | 1040 / 74 | BYTE vs MESSAGE pipe | ✓ |
| ISSUE-16 | **LOW** | `et_handle.py` | 264–279 | O(n) scan ignores reverse dict | ✓ |
| ISSUE-17 | **LOW** | `et_bridge32.c` | 2293, 3083 | g_awe_windows CLion unresolved | ✓ |
| ISSUE-18 | **MEDIUM** | Multiple | — | Error capture incomplete | ✓ |
| ISSUE-19 | **LOW** | `build.bat`, `et_bridge32.c` | 27 / 170 | -lntdll inconsistency | ✓ |

**Total Issues Found: 20** (19 base + ISSUE-01B sub-issue)
**Critical: 6 · High: 4 · Medium: 6 · Low: 4**

---

## Priority Resolution Order

The issues are chained. The correct resolution order respects their dependency structure:

### Tier 1 — Bridge Is Currently Non-Functional (Must fix first)

ISSUE-02 blocks everything else. Resolving it also resolves ISSUE-03 and ISSUE-04 as consequences. The correct fix for ISSUE-02 is to implement DLL injection in `ETInjector._do_inject()`:

1. **ISSUE-02**: Implement DLL injection: `VirtualAllocEx` + `WriteProcessMemory` of the DLL path string + `CreateRemoteThread(LoadLibraryA)` to load `et_bridge32.dll` into the target, then call `ET32_Init(broker_pid)` and `ET32_SetKiFastTrampoline(trampoline_addr)`.
2. **ISSUE-01 + ISSUE-01B**: Replace `make_dispatcher_shellcode()` with a real pipe-communicating dispatcher. Fix `ETStubGenerator.make_stub()` to use `0xE8 <rel32>` (relative call) instead of `0xFF 0x15 <ptr>`.
3. **ISSUE-04**: Resolved once ISSUE-02 is fixed and `ET32_SetKiFastTrampoline` is called from the broker after trampoline allocation.
4. **ISSUE-03**: Resolved once ISSUE-02 is fixed and `ET32_Init` installs the real hook.

### Tier 2 — Bridge Partially Functional, Major Features Missing

5. **ISSUE-05**: Assign a new command code for `DYNAMIC_SYSCALL` (e.g. `0xB0` or a new family-12 code) distinct from `COMPOUND_BATCH = 0xB1`.
6. **ISSUE-06**: Implement interactive CLI with command parser, help menu, and step-by-step attachment feedback.
7. **ISSUE-09**: Replace `CMakeLists.txt` and `main.cpp` with a proper CMake DLL target for `et_bridge32.dll`.
8. **ISSUE-07**: Replace the `__syscall_XXXX` lookup with direct service table lookup via `self._wow64._svc_tbl.lookup(service_number)`.

### Tier 3 — Correctness and Completeness

9. **ISSUE-08**: `_inject_shellcode_fallback()` must return `False` — partial success is a failure.
10. **ISSUE-10**: Read broker response args into `lpSystemInfo` in `ET32_GetNativeSystemInfo`.
11. **ISSUE-11**: Return peer address from broker response in `ET32_accept`.
12. **ISSUE-12**: Implement remaining `ETAPIGateway` methods for all 12 families.
13. **ISSUE-18**: Implement exit-code capture in `on_process_exit()` and optionally `OutputDebugString` forwarding.

### Tier 4 — Race Conditions

14. **ISSUE-13**: Hold `_pipe_lock` across the entire `create_pipe_for_pid` sequence (check → create → IOCP associate → register).
15. **ISSUE-14**: Use a `_pending_pids: Set[int]` inside `_lock` to prevent concurrent engagement of the same PID.

### Tier 5 — Bugs and Polish

16. **ISSUE-15**: Change broker pipe creation to `PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE` to match the DLL's intent, or remove the `SetNamedPipeHandleState` call from the C DLL.
17. **ISSUE-16**: Replace the O(n) loop in `project_address()` with `self._addr64_to_handle.get(addr64)`.
18. **ISSUE-17**: Add `__attribute__((used))` (GCC) or pragma to the `g_awe_windows` declaration.
19. **ISSUE-19**: Remove `-lntdll` from the source comment to match `build.bat`.

---

*Audit complete. All 20 issues are individually traced, verified against source, and ready for resolution.*

*P ∘ D ∘ T = E — Michael James Muller / Aevum Defluo*

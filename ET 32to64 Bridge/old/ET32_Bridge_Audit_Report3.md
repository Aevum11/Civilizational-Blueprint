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
| ISSUE-17 | **MEDIUM** | Naming / Protocol | `et_bridge32.c:291–292`, `et_math.py:856–857` | `CMD_READ_MEM`/`CMD_WRITE_MEM` (0x07/0x08) in C conflict with Python names for different codes (0x0B/0x0C) |
| ISSUE-18 | **HIGH** | Semantic Overload / Protocol | `et_bridge32.c:2347–2355`, `et_host64.py:601–610` | `HEAP_ALLOC` (0x05) overloaded as AWE init signal — no dedicated command code |
| ISSUE-19 | **MEDIUM** | Error Logging Gap | All files | Attached process stdout/stderr/Win32 exceptions not captured — audit point 5 partially unmet |
| ISSUE-20 | **LOW** | Inconsistency | `build.bat:170–177`, `et_bridge32.c:27` | `-lntdll` in DLL source comment but absent from `build.bat` MinGW command |

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

### Complete Requirements (verbatim from specification)

Every sub-requirement below is explicitly absent:

**1. Persistent console that will not close on Windows.**
No `AllocConsole()`, `SetConsoleTitle()`, or console-persistence mechanism is present. When run as a PyInstaller EXE via double-click, the console window closes immediately on exit. No `input("Press Enter to exit")` or equivalent guard exists.

**2. Full CLI — all features and functions accessible interactively.**
All 12 command families must be invokable from the CLI at runtime: MEMORY_BASIC, MEMORY_MAP, THREAD_OPS, DLL_OPS, PROCESS_OPS, REGISTRY_OPS, GRAPHICS_OPS, FILE_OPS, SYNC_OPS, NET_OPS, PYTHON_OPS, COMPOUND_OPS. Currently no command is invokable.

**3. `help` command — listing everything.**
The help menu must enumerate: all 12 command families and every command code within each, whitelist management commands (add/remove/list), blacklist management commands (add/remove/list), status/metrics commands, attach/detach commands, logging control, AWE commands, and system info queries. No help system of any kind exists.

**4. Default mode: dynamically links to whitelist/blacklist items automatically.**
The default operation must display live process detection events as they happen — when a configured process is found, the console confirms it by name and PID, shows the injection steps, and updates when it exits. This display must work without any user input. The monitor fires callbacks but nothing is printed to an interactive console; it logs to file only.

**5. Step-by-step attachment process confirmation — every step must be shown.**
The console must confirm each of the following steps explicitly with success/failure/fallback:
- Named pipe created for PID (pipe name, buffer size)
- DLL injection initiated (target path, method: IAT/shellcode/DLL)
- `et_bridge32.dll` loaded into target (base address in target)
- `ET32_Init(broker_pid)` called and returned (TRUE/FALSE)
- Trampoline written for KiFastSystemCall (address in target)
- `ET32_SetKiFastTrampoline()` called (trampoline address confirmed)
- IAT patches applied (count of APIs patched, list of API names)
- AWE physical page pool allocated for PID (pages allocated, GB)
- WOW64 KiFastSystemCall hook installed (root address patched)
- Handshake packet exchanged (DLL version, broker PID confirmed)
- Bridge active for PID (all subsystems confirmed)

**6. All errors and fallbacks shown.**
If any step fails or falls back (e.g. IAT patch fails for a specific API, AWE pool allocation fails, pipe connection times out), the console must print the specific error and what fallback was taken — not just log it to file. Currently all such events are logged to the file only via ETLog.

**7. Live metrics — specific subsystem confirmation required.**
The console must show, at regular intervals and on demand, the following specific subsystem states:
- **AWE Bookshelf**: pages allocated, pages in use, pages free, windows active out of 144 maximum, GB of physical memory under management, expand count
- **Heaven's Gate**: ntdll64 base address resolved (yes/no + address), service table size (number of NT services discovered by PE reflection)
- **KiFastSystemCall hook**: installed (yes/no), root address in ntdll32, trampoline address, syscalls routed since install
- **IPC**: connection status per PID, throughput KB/s, queue depth, total requests, success rate (Koide alignment K_eff)
- **Handle table**: live count, fill ratio, total allocated since start
- **Error registry**: V(system), total errors by severity, coherence depth

**8. Manual command mode for specific purpose or debugging.**
The CLI must accept commands like: `status`, `attach <pid>`, `detach <pid>`, `list`, `alloc <pid> <size>`, `exec <pid> <python_code>`, `reg get <pid> <key>`, `awe status <pid>`, `hook status <pid>`, `help`, `quit`. None of these exist.

### What Is Present

A non-interactive passive loop. The banner is printed once. All operational events go to the log file and stdout via the logging system — but there is no structured console confirmation sequence, no input thread, no command parser, no help text, no persistent console, and no live subsystem status display.

### ET Derivation

Identification Principle: P = the operator, D = the CLI interface, T = the running bridge process. D (the CLI) is entirely absent. Without D, the operator (P) has no T to traverse — no way to interact with the running bridge, no way to observe AWE and Heaven working, no confirmation that anything is functioning. The gap IS the entire CLI subsystem.

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

## ISSUE-17 — MEDIUM: `CMD_READ_MEM` / `CMD_WRITE_MEM` in C Name a Different Operation Than Python

**Files:** `et_bridge32.c:291–292`, `et_math.py:846–857`
**Category:** Naming Confusion / Protocol Correctness
**Audit Point:** 2, 4

### Full Trace

**C side (`et_bridge32.c:291–292`):**
```c
#define CMD_READ_MEM            0x07
#define CMD_WRITE_MEM           0x08
```

**Python side (`et_math.py`, CmdCode class):**
```python
GLOBAL_MEM_STATUS  = 0x07  # GlobalMemoryStatusEx
NATIVE_SYS_INFO    = 0x08  # GetNativeSystemInfo
CLOSE_HANDLE64     = 0x09
DUPLICATE_HANDLE64 = 0x0A
READ_MEM           = 0x0B  # ReadProcessMemory (cross-dimension)
WRITE_MEM          = 0x0C  # WriteProcessMemory (cross-dimension)
```

### Problem

The C DLL names `CMD_READ_MEM = 0x07`, but 0x07 is `GLOBAL_MEM_STATUS` in Python — GlobalMemoryStatusEx, not ReadProcessMemory. `CMD_WRITE_MEM = 0x08` in C is `NATIVE_SYS_INFO` (GetNativeSystemInfo) in Python. The actual cross-dimension ReadProcessMemory and WriteProcessMemory are 0x0B and 0x0C in Python, and have no corresponding C defines at all.

The wire protocol is accidentally consistent only because `ET32_GlobalMemoryStatusEx` happens to use `CMD_READ_MEM (0x07)` and the broker handles 0x07 as `GLOBAL_MEM_STATUS`. But the names are factually wrong — `CMD_READ_MEM` does not read memory; it requests a memory status report.

### Consequence

1. Any developer reading the C code will believe `CMD_READ_MEM` performs a cross-dimension `ReadProcessMemory`. It does not.
2. If a developer adds a real cross-dimension `ReadProcessMemory` call in the C DLL using `CMD_READ_MEM (0x07)`, the broker dispatches `GLOBAL_MEM_STATUS` — a silent, wrong operation with no error.
3. `READ_MEM (0x0B)` and `WRITE_MEM (0x0C)` — the actual cross-dimension memory commands — have no C defines, making them structurally unreachable from any C DLL code.
4. The entire d=1 command code table in the C DLL must be corrected to match et_math.py exactly.

### Fix

Correct the C defines to match Python names and codes exactly:
```c
/* d=1 MEMORY_BASIC command codes — mirror of et_math.py CmdCode */
#define CMD_VIRT_ALLOC          0x01
#define CMD_VIRT_FREE           0x02
#define CMD_VIRT_PROTECT        0x03
#define CMD_VIRT_QUERY          0x04
#define CMD_HEAP_ALLOC          0x05
#define CMD_HEAP_FREE           0x06
#define CMD_GLOBAL_MEM_STATUS   0x07   /* was CMD_READ_MEM  — GlobalMemoryStatusEx */
#define CMD_NATIVE_SYS_INFO     0x08   /* was CMD_WRITE_MEM — GetNativeSystemInfo */
#define CMD_CLOSE_HANDLE64      0x09
#define CMD_DUPLICATE_HANDLE64  0x0A
#define CMD_READ_MEM            0x0B   /* ReadProcessMemory cross-dimension */
#define CMD_WRITE_MEM           0x0C   /* WriteProcessMemory cross-dimension */
```

Update `ET32_GlobalMemoryStatusEx` to use `CMD_GLOBAL_MEM_STATUS (0x07)`.
Update `ET32_GetNativeSystemInfo` to use `CMD_NATIVE_SYS_INFO (0x08)`.

---

## ISSUE-18 — HIGH: `HEAP_ALLOC` Semantically Overloaded as AWE Init Signal — No Dedicated Command Code

**Files:** `et_bridge32.c:2347–2355`, `et_host64.py:601–610`
**Category:** Semantic Overload / Protocol / Incomplete Implementation
**Audit Point:** 2, 4

### Full Trace

`ET32_Init()` in et_bridge32.c:2347–2355, after AWE initialisation:
```c
/* Signal broker that AWE subsystem is ready */
{
    uint8_t pbuf[8];
    et_arg_buf ab;
    et_argbuf_init(&ab, pbuf, sizeof(pbuf));
    et_pack_uint32(&ab, g_target_pid);
    uint32_t err = 0;
    et_call(CMD_FAMILY_MEMORY_BASIC, CMD_HEAP_ALLOC, pbuf, ab.pos, &err);
}
```

`_handle_memory_basic()` in et_host64.py:601–610, inside the `HEAP_ALLOC` handler:
```python
if code == CmdCode.HEAP_ALLOC:
    is_awe_init = (len(args) == 1 and int(args[0]) == pid)
    if is_awe_init and self._awe is not None:
        # AWE init signal — just ACK.
        return self._make_ok(pkt, 1)
    # Regular HEAP_ALLOC...
    size = args[0] if args else DIGITAL_ACTION_QUANTUM
    ...
```

### Problem

`CMD_HEAP_ALLOC (0x05)` is given two incompatible meanings:

1. **Allocation**: Allocate memory in the broker heap. Args = `(size, protect, awe_flag)`.
2. **Handshake signal**: Notify broker that AWE is ready. Args = `(target_pid,)`.

The broker distinguishes them with the heuristic `len(args) == 1 and int(args[0]) == pid`. This is fragile: a legitimate HeapAlloc request with exactly one argument equal to the source PID (e.g. a HeapAlloc for `size = own_pid` bytes) is silently misclassified as an AWE init signal and ACKed with no allocation performed.

There is no dedicated command code for this handshake. This is an overload imposed on an existing operation. The protocol must not have dual-purpose command codes with heuristic disambiguation.

### ET Derivation

Identification Principle: two distinct T-operations (heap memory allocation and AWE subsystem handshake) share the same D (command code 0x05). Subsumption Law: neither operation subsumes the other without remainder — their argument semantics are incompatible. The gap IS the missing dedicated command code.

### Fix

Add `CMD_AWE_INIT_SIGNAL = 0x0D` (the next available d=1 code after `WRITE_MEM = 0x0C`) to both Python and C.

**et_math.py:**
```python
AWE_INIT_SIGNAL    = 0x0D  # DLL signals AWE subsystem init complete
```

**et_bridge32.c:**
```c
#define CMD_AWE_INIT_SIGNAL     0x0D
```

Change `ET32_Init` to call `CMD_AWE_INIT_SIGNAL` instead of `CMD_HEAP_ALLOC`.

Add a dedicated handler in `_handle_memory_basic()`:
```python
elif code == CmdCode.AWE_INIT_SIGNAL:
    # AWE subsystem handshake from DLL after ET32_Init completes.
    if self._awe is not None:
        self._log.info("AWE init signal received from PID %d", pkt.source_pid)
    return self._make_ok(pkt, 1)
```

Remove the `is_awe_init` heuristic from the `HEAP_ALLOC` handler entirely.

---

## ISSUE-19 — MEDIUM: Error Logging Gap — Not Debugger-Grade for Attached Programs (Audit Point 5)

**Files:** `et_bridge32.c`, `et_logger.py`, `et_errors.py`, `et32_bridge_main.py`
**Category:** Incomplete — error logging / audit point 5
**Audit Point:** 5

### Specification Requirement (verbatim)

> *"Error logging must be full and complete, so no issue is missed in the compiled program nor the attached programs (including all children). This means it also captures ALL errors in all attached program (including all children), which essentially also makes it a debugger for attached programs as all errors will be caught, so we know what is going on and what to fix."*

This is not simply "log API failures." The requirement is **debugger-grade**: every error, every exception, every abnormal event from every attached process and all its children must be captured and available to the operator. The bridge must function as a remote error interceptor for arbitrary 32-bit code.

### What Is Implemented

- **`et_report_error()`** (et_bridge32.c:160–225): captures Win32 API failures with file/line/function/OS code and sends as `CMD_CTRL_ERR` packets.
- **Broker `CTRL_ERR` handler** (et_host64.py:2420–2435): receives reports, creates `ETWindowsAPIError` entries.
- **`ETErrorRegistry`** (et_errors.py:724): centralised thread-safe error log.
- **`ETCrashLogger`** (et_logger.py:129): covers the **broker process** (sys.excepthook, faulthandler, atexit, Windows SEH filter).

### What Is NOT Captured — Gap Analysis

**1. VEH intercepts only `EXCEPTION_ACCESS_VIOLATION` — must cover ALL exception types.**
`et_veh_handler()` (et_bridge32.c:3063–3068):
```c
DWORD code = pExInfo->ExceptionRecord->ExceptionCode;
if (code != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
```
Every other exception type falls through uncaptured. The following must also be intercepted and forwarded to the broker as structured `CMD_CTRL_ERR` packets:
- `EXCEPTION_STACK_OVERFLOW (0xC00000FD)` — stack blown
- `EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D)` — invalid opcode
- `EXCEPTION_INT_DIVIDE_BY_ZERO (0xC0000094)` — division by zero
- `EXCEPTION_FLT_DIVIDE_BY_ZERO (0xC000008E)` — floating point division by zero
- `EXCEPTION_PRIV_INSTRUCTION (0xC0000096)` — privileged instruction
- `EXCEPTION_ARRAY_BOUNDS_EXCEEDED (0xC000008C)` — out-of-bounds array access
- `EXCEPTION_BREAKPOINT (0x80000003)` — debug breakpoint (indicates debugger interaction or assertion)
- `EXCEPTION_SINGLE_STEP (0x80000004)` — single-step trace
- `EXCEPTION_DATATYPE_MISALIGNMENT (0x80000002)` — misaligned data access
- `EXCEPTION_IN_PAGE_ERROR (0xC0000006)` — paging fault
- `EXCEPTION_INT_OVERFLOW (0xC0000095)` — integer overflow
- Any unrecognised `STATUS_*` code in the `0xC0000000` range (NT error severity)

Each intercepted exception must be forwarded via `et_report_error()` with: exception code (hex), faulting instruction address, module name (resolved from base), first exception information value (read/write/execute).

**2. Attached process stdout/stderr invisible.**
No `CreatePipe` + `SetStdHandle` + `WriteProcessMemory` injection to redirect the 32-bit target's stdout/stderr handles to pipes the broker reads. Any `printf`, `fprintf(stderr,...)`, CRT error output, or `_cwprintf` in the target is invisible to the broker and operator.

**3. `OutputDebugString` not captured from attached processes.**
OutputDebugString writes to the Win32 debug output buffer (`NtRaiseDebugger`). Capturing this without `DebugActiveProcess` (which would conflict with real debugger attachments) requires a shared memory + event approach: the broker creates a named event `DBWIN_BUFFER_READY` and shared memory section `DBWIN_BUFFER`, which Windows populates with OutputDebugString calls. Neither mechanism is implemented.

**4. Exit code not captured for attached processes or their children.**
`on_process_exit()` (et_api.py:902) receives `ProcessInfo` but never calls `GetExitCodeProcess`. The exit code — which often encodes an `NTSTATUS` error code or an application-specific error — is never retrieved, never logged, and never shown to the operator.

**5. Children of bridged processes: errors not captured.**
The monitor correctly tracks children (et_monitor.py:418) and bridges them. However, the same error capture gaps apply recursively to all children at all depths. A child process crash, exception, or CRT failure is as invisible as the parent's.

**6. CRT-level errors not forwarded.**
`_set_invalid_parameter_handler`, `_set_abort_behavior`, assertion failures (`assert()`), and `abort()` calls in the 32-bit target are not intercepted. These are common sources of application crashes and the operator cannot diagnose them.

### Gap Summary

| Error Source | Currently Captured | Required |
|---|---|---|
| Win32 API failures (explicit `ET_REPORT_ERROR`) | ✓ | ✓ |
| Access violations (VEH) | ✓ | ✓ |
| Stack overflow | ✗ | ✓ |
| Illegal instruction | ✗ | ✓ |
| Integer divide-by-zero | ✗ | ✓ |
| Float divide-by-zero | ✗ | ✓ |
| All other SEH codes | ✗ | ✓ |
| Stdout/stderr output | ✗ | ✓ |
| OutputDebugString | ✗ | ✓ |
| Process exit code | ✗ | ✓ |
| Child process exceptions | ✗ | ✓ |
| CRT errors / assertions | ✗ | ✓ |
| Broker EXE crashes | ✓ (ETCrashLogger) | ✓ |

### ET Derivation

Identification Principle: P = the complete set of error events in attached processes, D = the capture mechanisms, T = the error forwarding pipeline. D is only partially identified — it covers a subset of the error P-space. Every uncaptured error type is a Descriptor Gap. Each gap means the operator cannot "know what is going on and what to fix."

---

## ISSUE-20 — LOW: `-lntdll` in DLL Source Comment but Absent from `build.bat` MinGW Command

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

## ISSUE-21 — HIGH: `ETMetrics.record()` Has No Lock — 12 Concurrent Workers Write Unsynchronised

**File:** `et_math.py`
**Lines:** 1413–1441 (`record()`), `et_ipc.py:275–344` (worker spawn)
**Category:** Race Condition
**Audit Point:** 1

### Full Trace

`ETIPCServer` spawns `S = 12` worker threads at startup (et_ipc.py:277–330):
```python
self._n_workers: int = S  # 12
for i in range(self._n_workers):
    t = threading.Thread(target=self._worker, ...)
```

Each worker processes dispatched packets and calls `self._metrics.record(pkt.cmd_family, latency_us, success)` (et_ipc.py:793). Additionally, dispatch threads (et_ipc.py:284) also call `record()` at lines 728 and 944. The same `ETMetrics` instance is shared among all of them.

`ETMetrics.record()` (et_math.py:1413–1441) modifies:
```python
self.total_requests += 1          # read-modify-write, no lock
self.successful_requests += 1     # read-modify-write, no lock
self.total_latency_us += latency_us   # float RMW, no lock
self.bytes_transferred += bytes_count  # int RMW, no lock
self.family_counts[family] += 1   # dict item RMW, no lock
self.total_log_latency += log_l   # float RMW, no lock
self.family_log_latency[family] += log_l  # dict item RMW, no lock
```

**There is no `threading.Lock()`, `RLock()`, or `threading.local()` in `ETMetrics`.**

### Verification

In CPython, integer `+=` is NOT atomic — it is a `LOAD_FAST`, `INPLACE_ADD`, `STORE_FAST` sequence. The GIL can release between any two bytecode instructions. Float `+=` is likewise not atomic. Two workers incrementing `total_requests` simultaneously can produce `total_requests = N+1` instead of `N+2` — a lost update.

`family_counts` and `family_log_latency` are regular dicts — dict item `+=` is a read-modify-write that also involves two separate dict operations under no lock.

### Consequence

Under the 12-worker load: counters undercount, latency accumulators drift, K_eff (Koide alignment) computation is wrong, entropy is wrong. The bridge's self-reported health metrics are unreliable at any meaningful load. The operator and the stability loop both act on incorrect data.

---

## ISSUE-22 — HIGH: `et_bridge32.c` `g_connected` Read Outside `g_pipe_cs`; `ET32_Shutdown()` Deletes CS While `et_call()` May Hold It

**File:** `et_bridge32.c`
**Lines:** 1077 (`et_call`), 2394–2397 (`ET32_Shutdown`)
**Category:** Race Condition
**Audit Point:** 1

### Race 1 — `g_connected` TOCTOU in `et_call()`

```c
// Line 1077
if (!g_connected) {                     // ← READ g_connected OUTSIDE g_pipe_cs
    if (error_out) *error_out = ERROR_NOT_CONNECTED;
    return 0;
}
// ... gap ...
EnterCriticalSection(&g_pipe_cs);       // ← line 1113 — enter CS AFTER the check
BOOL ok = WriteFile(g_pipe, ...);
LeaveCriticalSection(&g_pipe_cs);
```

Between line 1077 and line 1113, another thread (executing `DLL_PROCESS_DETACH` → `ET32_Shutdown`) can set `g_connected = FALSE`, close `g_pipe`, and `DeleteCriticalSection(&g_pipe_cs)`. If that happens:
- The first thread calls `EnterCriticalSection(&g_pipe_cs)` on a deleted CS → undefined behaviour.
- Even if the CS still exists, `g_pipe` is now `INVALID_HANDLE_VALUE` → `WriteFile` fails.

### Race 2 — `ET32_Shutdown()` Destroys `g_pipe_cs` While Another Thread Holds It

`ET32_Shutdown()` (line 2365) calls:
```c
et_pipe_disconnect();            // sets g_connected=FALSE, closes g_pipe
DeleteCriticalSection(&g_pipe_cs);  // line 2397 — DESTROYS the CS
```

Any thread currently inside `et_call()` holding `g_pipe_cs` (between EnterCriticalSection and LeaveCriticalSection) will leave and call `LeaveCriticalSection` on a now-destroyed CS → undefined behaviour / crash.

The fix is to hold `g_pipe_cs` through the entire sequence: set `g_connected = FALSE` while holding the CS, call `DeleteCriticalSection` only after all in-flight callers have exited.

---

## ISSUE-23 — HIGH: `GPU_ALLOC_VRAM` and `GPU_MAP_VRAM` Are Placeholder Implementations

**File:** `et_host64.py`
**Lines:** 1309–1333 (`GPU_ALLOC_VRAM`), 1349–1355 (`GPU_MAP_VRAM`)
**Category:** Placeholder / Not Implemented
**Audit Points:** 2, 3, 4

### `GPU_ALLOC_VRAM` — Uses `VirtualAlloc`, Not GPU/VRAM Memory

```python
elif code == CmdCode.GPU_ALLOC_VRAM:
    # Allocate GPU-accessible host memory via VirtualAlloc with large page flag
    # (true VRAM allocation requires a D3D12/Vulkan device handle — this allocates
    # a committed CPU-GPU visible buffer in 64-bit space)
    ...
    ptr = getattr(kernel32, 'VirtualAlloc')(
        None,
        ctypes.c_size_t(size),
        ctypes.wintypes.DWORD(MEM_COMMIT | MEM_RESERVE),
        ctypes.wintypes.DWORD(protect)
    )
```

`VirtualAlloc` allocates CPU RAM (virtual address space). This is not GPU VRAM. There is no `D3D12_HEAP_TYPE_DEFAULT`, no `IDXGIAdapter::CheckInterfaceSupport`, no `IDXGIResource`, no NvAPI `NvAPI_GPU_AllocGPUVirtualAddress`, no `VkDeviceMemory`. The comment in the code explicitly states "true VRAM allocation requires a D3D12/Vulkan device handle — this allocates a committed CPU-GPU visible buffer." This is a placeholder that substitutes CPU memory for GPU memory.

### `GPU_MAP_VRAM` — Returns Existing Handle with No Mapping

```python
elif code == CmdCode.GPU_MAP_VRAM:
    # Map a VRAM allocation for access — returns the 64-bit addr
    handle = int(args[0]) if args else 0
    addr64 = self._table.resolve(handle)
    if addr64 is None:
        return self._make_error(...)
    return self._make_ok(pkt, handle, addr64)   # ← no mapping, just returns what was there
```

No actual memory mapping occurs. The function resolves the handle to its stored address and returns it. No `IDXGIResource::GetSharedHandle`, no `MapViewOfFile`, no CUDA `cuMemHostGetDevicePointer`. This is a stub.

### Consequence

Any 32-bit application using the bridge to access large VRAM (>4GB, e.g. a GPU compute or graphics workload) receives a CPU-RAM allocation instead of GPU VRAM. The `GPU_MAP_VRAM` call performs no mapping and returns stale data. GPU_SUBMIT and GPU_HEAVEN_CALL (which depend on a valid device created by GPU_CREATE_DEVICE) may work since D3D9/D3D11 device creation IS implemented, but the VRAM allocation path provides no real GPU memory.

---

## ISSUE-24 — HIGH: IAT Table Missing 10 Critical Functions Across 4 Families

**File:** `et_bridge32.c`
**Lines:** 2100–2158 (complete `g_iat_hooks[]` table)
**Category:** Not Implemented — IAT coverage gap
**Audit Points:** 2, 4

### Trace — Complete `g_iat_hooks[]` Table Audited

The IAT patch table (`g_iat_hooks[]`) is the complete list of Windows API functions the C DLL can intercept at injection time. By reading lines 2100–2158 in full, the following functions have **no ET32_ wrapper implementation and no IAT entry** despite the broker having handlers for them:

| Missing Function | Family | ET Command Code | Impact |
|---|---|---|---|
| `CreateThread` | THREAD_OPS (d=3) | `THREAD_CREATE (0x21)` | 32-bit apps cannot have bridge-managed 64-bit threads |
| `SuspendThread` | THREAD_OPS (d=3) | `THREAD_SUSPEND (0x22)` | Thread suspension not bridged |
| `ResumeThread` | THREAD_OPS (d=3) | `THREAD_RESUME (0x23)` | Thread resume not bridged |
| `TerminateThread` | THREAD_OPS (d=3) | `THREAD_TERMINATE (0x24)` | Thread termination not bridged |
| `GetThreadContext` | THREAD_OPS (d=3) | `THREAD_CONTEXT (0x25)` | Thread context read not bridged |
| `HeapAlloc` | MEMORY_BASIC (d=1) | `HEAP_ALLOC (0x05)` | 32-bit heap allocations not bridged to AWE |
| `HeapFree` | MEMORY_BASIC (d=1) | `HEAP_FREE (0x06)` | Heap free not bridged |
| `UnmapViewOfFile` | MEMORY_MAP (d=2) | `FILE_MAP_CLOSE (0x13)` | Mapped view release not bridged |
| `FlushViewOfFile` | MEMORY_MAP (d=2) | `FILE_MAP_FLUSH (0x14)` | Mapped view flush not bridged |
| `WaitForSingleObject` | SYNC_OPS (d=9) | `SYNC_WAIT (0x83)` | Single-object wait on bridge handles fails |
| `socket` | NET_OPS (d=10) | `NET_SOCKET64 (0x91)` | 32-bit socket creation not bridged to 64-bit |
| `bind` | NET_OPS (d=10) | `NET_BIND64 (0x92)` | Socket bind not bridged |
| `send` | NET_OPS (d=10) | `NET_SEND64 (0x93)` | Data send not bridged |
| `recv` | NET_OPS (d=10) | `NET_RECV64 (0x94)` | Data receive not bridged |

### Critical Consequence

`WaitForSingleObject` absence is particularly damaging: the C DLL hooks `CreateEventA`, `CreateMutexA`, `CreateSemaphoreA` and returns bridge handles. When a 32-bit application calls `WaitForSingleObject` on that bridge handle, the original (unhooked) `WaitForSingleObject` receives a 32-bit proxy handle value (in range `ET_HANDLE_BASE..ET_HANDLE_MAX`). Windows sees this as an invalid handle → `WAIT_FAILED`. Any application that creates a synced event and waits on it through the bridge will deadlock or fail immediately.

The complete THREAD_OPS absence means no bridged thread management — CreateThread creates a 32-bit thread with no broker registration, no 64-bit capabilities, and no cleanup path.

---

## ISSUE-25 — MEDIUM: `--status` Argument Defined and Documented but Never Handled

**File:** `et32_bridge_main.py`
**Lines:** 128–133 (argument definition), all of `main()` (331–482)
**Category:** Not Implemented
**Audit Points:** 2, 6

### Evidence

```python
# et32_bridge_main.py:128
parser.add_argument(
    "--status",
    action  = "store_true",
    default = False,
    help    = "Print current bridge status and exit (if another instance is running)"
)
```

`main()` (lines 331–482): there is no `if args.status:` branch. The argument is parsed and silently discarded.

### Consequence

Running `ET32_Bridge.exe --status` has zero effect — it behaves identically to running the bridge normally (starts the IPC server, starts the monitor, enters the main loop). The documented capability to "query status of a running instance" is entirely absent. No inter-process query mechanism exists (no socket, no shared memory, no named pipe query path), so even implementing it requires designing the IPC query mechanism, not just adding a branch.

This also violates audit point 6: the CLI is supposed to be the control surface for the bridge, including status inspection without full restart.

---

## ISSUE-27 — CRITICAL: `MAX_SAVED_HOOKS = 64` Hard Static Limit — Hooks Written but Not Saved When Overflow, Causing Permanent Unrestorable IAT Corruption

**File:** `et_bridge32.c`
**Lines:** 2161 (`#define MAX_SAVED_HOOKS 64`), 2168–2169 (static array), 2218–2223 (overflow path), 2231–2251 (`et_patch_iat_all`)
**Category:** Hard Limit / Not Properly Implemented / No Exceptions Allowed
**Audit Points:** 2, 3, 4

### Full Trace

`g_iat_hooks[]` contains 56 named function entries. `et_patch_iat_all()` (line 2231) patches **every loaded module** in the target process — not just the main executable:

```c
static void et_patch_iat_all(void)
{
    HMODULE hExe = GetModuleHandleA(NULL);
    if (hExe) et_patch_iat_for_module(hExe);

    /* Also patch any already-loaded DLLs except ourselves */
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    ...do { et_patch_iat_for_module(h); } while (Module32Next(snap, &me));
```

Each loaded DLL that imports any of the 56 hooked functions gets its own IAT slot patched. A 32-bit application with **N DLLs, each importing K hooked functions**, creates **N × K** saves. With a moderately complex process (20 DLLs × average 4 hooked imports = 80 hooks, or 5 DLLs × 13 hooked imports = 65 hooks), `MAX_SAVED_HOOKS = 64` is exceeded.

### The Overflow Is Silent and Destructive

```c
if (g_n_saved_hooks < MAX_SAVED_HOOKS) {
    g_saved_hooks[g_n_saved_hooks].iat_slot = slot;
    g_saved_hooks[g_n_saved_hooks].original  = *slot;
    g_n_saved_hooks++;
}
et_write_ptr(slot, g_iat_hooks[i].hook);   // ← ALWAYS writes the hook
```

When `g_n_saved_hooks >= MAX_SAVED_HOOKS`:
- The original pointer is **NOT saved**.
- The hook IS written unconditionally.
- `et_restore_iat()` only loops to `g_n_saved_hooks` — it never sees these overflow entries.
- After `ET32_Shutdown()` or `DLL_PROCESS_DETACH`, those IAT slots permanently point into the now-unloaded `et_bridge32.dll`. Any subsequent call to any of those functions → crash.

### No Exceptions Means No Limit

The bridge specification is explicit: EVERYTHING must be caught and properly handled, with no exceptions and no limits. A static array of 64 is a limit, and it is unacceptable. The save list must be dynamic — heap-allocated and grown without bound — so that every single IAT slot patched in every single module is unconditionally and correctly saved.

### Fix

Replace `g_saved_hooks[MAX_SAVED_HOOKS]` with a heap-allocated dynamic array:

```c
typedef struct {
    FARPROC *iat_slot;
    FARPROC  original;
} et_saved_hook;

static et_saved_hook *g_saved_hooks   = NULL;
static int            g_saved_hooks_cap = 0;
static int            g_n_saved_hooks = 0;

/* In et_patch_iat_for_module, before writing hook: */
if (g_n_saved_hooks >= g_saved_hooks_cap) {
    int new_cap = (g_saved_hooks_cap == 0) ? 256 : g_saved_hooks_cap * 2;
    et_saved_hook *p = (et_saved_hook *)HeapReAlloc(
        GetProcessHeap(), 0, g_saved_hooks, new_cap * sizeof(et_saved_hook));
    if (!p) {
        /* Cannot save — MUST NOT patch if we cannot restore */
        ET_REPORT_ERROR("HeapReAlloc for saved_hooks", g_target_pid,
                        CMD_FAMILY_COMPOUND_OPS, CMD_CTRL_ERR);
        continue;  /* skip this hook — do not write */
    }
    g_saved_hooks     = p;
    g_saved_hooks_cap = new_cap;
}
g_saved_hooks[g_n_saved_hooks].iat_slot = slot;
g_saved_hooks[g_n_saved_hooks].original  = *slot;
g_n_saved_hooks++;
et_write_ptr(slot, g_iat_hooks[i].hook);
```

Initial capacity: `S² = 144` (manifold symmetry squared — the ET-derived natural starting point for any unlimited collection). Doubles on overflow. Freed in `ET32_Shutdown()`.

---

## ISSUE-28 — HIGH: `bookshelf_alloc()` Reads `_pools` and `pool.n_in_use` Outside Lock

**File:** `et_awe.py`
**Lines:** 765, 771, 801 (`bookshelf_alloc`)
**Category:** Race Condition
**Audit Point:** 1

### Full Trace

```python
def bookshelf_alloc(self, pid: int, h_process: int,
                    n_bytes: int, protect: int = PAGE_READWRITE) -> Optional[int]:
    ...
    if pid not in self._pools:       # line 765 — READ _pools, no lock
        ...
        if not self.allocate_pool(pid, h_process, init):
            return None

    pool = self._pools[pid]          # line 771 — READ _pools[pid], no lock
    ...
    phys_page_start = pool.n_in_use  # line 801 — READ pool.n_in_use, no lock
    phys_base = phys_page_start * AWE_PAGE_SIZE
    ...
    if not self.map_window(pid, va, phys_base):
```

### Race 1 — `_pools` KeyError

`release_pool(pid)` (et_awe.py:469) acquires `self._lock` and deletes `self._pools[pid]`. If it runs between line 765 (check) and line 771 (access), `self._pools[pid]` raises `KeyError` → unhandled exception → allocation fails silently in the broker.

### Race 2 — Overlapping Physical Page Assignment

`pool.n_in_use` at line 801 is read without the lock. Two concurrent `bookshelf_alloc` calls for the same PID can both read the same `pool.n_in_use` value and compute the same `phys_page_start`. Both then call `map_window(pid, va, phys_base)` mapping the **same physical pages** into two different VA windows. The same physical pages are now mapped twice, making any write by one window's user visible as a corruption in the other window.

### Fix

The entire `bookshelf_alloc` body from the `_pools` check through the `phys_page_start` calculation must execute under `self._lock`. The Windows API calls (`reserve_window`, `map_window`, `VirtualFreeEx`) may need to be reorganised to be called after releasing the lock, with the physical page range reserved atomically under it.

---

## ISSUE-29 — HIGH: `ET32_CreateProcessA/W` Pass Through Without Notifying Broker — Spawned Children Not Bridged

**File:** `et_bridge32.c`
**Lines:** 1528–1545
**Category:** Not Implemented — audit point 4 (children)
**Audit Points:** 2, 4

### Full Trace

```c
BOOL WINAPI ET32_CreateProcessA(
    LPCSTR lpApp, LPSTR lpCmd, ...)
{
    /* Forward to original; bridge only intercepts when flag ET_BRIDGE_PROC is set. */
    return CreateProcessA(lpApp, lpCmd, lpPA, lpTA, bInherit,
                          dwFlags, lpEnv, lpDir, lpSI, lpPI);
}
```

`ET32_CreateProcessW` is identical — pure pass-through.

### Problem

When a bridged 32-bit process spawns a child via `CreateProcessA/W`, the bridge DOES NOT:
1. Notify the broker that a child was spawned.
2. Pass the child's PID to the broker for immediate injection.
3. Create a pipe for the child.
4. Inject `et_bridge32.dll` into the child before it starts executing.

The broker's `ETProcessMonitor` will eventually detect the child in its periodic scan (every 1–12 seconds), but by that time the child may have already executed past its initialisation and the IAT-hookable window may be gone (or the process may have already exited). For short-lived child processes (launchers, shell commands, build tools), this means they are NEVER bridged.

The comment "bridge only intercepts when flag ET_BRIDGE_PROC is set" references a flag (`ET_BRIDGE_PROC`) that is defined nowhere in the codebase. It is a placeholder comment left in code that simply passes through.

### Consequence

The requirement that "any 32-bit program (including all **children**)" receives 64-bit capabilities is broken at the DLL level. A bridged parent spawning a child gets no guarantee the child is bridged before it runs. The monitor-based detection introduces a race window of up to 12 seconds. The comment naming a non-existent flag (`ET_BRIDGE_PROC`) indicates this was known to be incomplete at the time of writing.

### Fix

`ET32_CreateProcessA/W` must:
1. Call the real `CreateProcessA/W` with `dwFlags | CREATE_SUSPENDED`.
2. Extract the child PID from `lpPI->dwProcessId` and thread handle from `lpPI->hThread`.
3. Send a `PROC_CREATE` notification to the broker with the child PID.
4. Wait for the broker to inject `et_bridge32.dll` and call `ET32_Init` in the child.
5. Call `ResumeThread(lpPI->hThread)` to let the child start with the bridge already active.
6. If `dwFlags` already had `CREATE_SUSPENDED`, leave the thread suspended (caller's intent preserved).

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
| ISSUE-17 | **MEDIUM** | `et_bridge32.c`, `et_math.py` | 291–292 / 846–857 | CMD_READ_MEM/WRITE_MEM naming conflict | ✓ |
| ISSUE-18 | **HIGH** | `et_bridge32.c`, `et_host64.py` | 2347–2355 / 601–610 | HEAP_ALLOC overloaded as AWE init signal | ✓ |
| ISSUE-19 | **MEDIUM** | Multiple | — | Error capture not debugger-grade | ✓ |
| ISSUE-20 | **LOW** | `build.bat`, `et_bridge32.c` | 27 / 170 | -lntdll inconsistency | ✓ |
| ISSUE-21 | **HIGH** | `et_math.py` | 1413–1441 | ETMetrics.record() race — no lock | ✓ |
| ISSUE-22 | **HIGH** | `et_bridge32.c` | 1077, 2394–2397 | g_connected TOCTOU; CS deleted while held | ✓ |
| ISSUE-23 | **HIGH** | `et_host64.py` | 1309–1355 | GPU_ALLOC_VRAM and GPU_MAP_VRAM are stubs | ✓ |
| ISSUE-24 | **HIGH** | `et_bridge32.c` | 2100–2158 | IAT table missing 14 critical API hooks | ✓ |
| ISSUE-25 | **MEDIUM** | `et32_bridge_main.py` | 128–133 | --status argument never handled | ✓ |
| ISSUE-26 | **MEDIUM** | `et_host64.py`, `et_math.py` | 1040–1122 / 884 | PROC_INJECT broker handler missing | ✓ |
| ISSUE-27 | **CRITICAL** | `et_bridge32.c` | 2161, 2218–2223 | MAX_SAVED_HOOKS hard limit — unrestorable IAT overflow | ✓ |
| ISSUE-28 | **HIGH** | `et_awe.py` | 765, 771, 801 | bookshelf_alloc reads _pools outside lock — race | ✓ |
| ISSUE-29 | **HIGH** | `et_bridge32.c` | 1528–1545 | ET32_CreateProcessA/W pass-through — children not bridged at spawn | ✓ |

**Total Issues Found: 30** (29 base + ISSUE-01B sub-issue)
**Critical: 7 · High: 13 · Medium: 8 · Low: 2**

---

## Audit Point 4 — 64-Bit Capability Parity Verification

### Specification Requirement (verbatim)

> *"This is supposed to be capable of giving any 32bit program (including all children) the same capabilities as a 64bit program. You will also check to ensure this is true."*

### Verdict: NOT TRUE in the current state.

The bridge is entirely non-functional end-to-end due to ISSUES 01–05 (DLL never injected, dispatcher is a placeholder, WOW64 hook is pass-through, trampoline is NULL). Beyond that structural failure, the following per-family capability gaps prevent full parity even when the injection chain is repaired.

### Per-Family End-to-End Capability Audit

Audited against: complete IAT table (et_bridge32.c:2100–2158), all broker handlers (et_host64.py), and ETAPIGateway (et_api.py).

| Family | d | Broker (et_host64.py) | C DLL IAT (et_bridge32.c) | Parity when injection fixed |
|--------|---|----------------------|--------------------------|-----------------------------|
| MEMORY_BASIC | 1 | Complete | Partial — `HeapAlloc`, `HeapFree`, `VirtualAllocEx`, `VirtualFreeEx`, `ReadProcessMemory`, `WriteProcessMemory` missing from IAT | ✗ ISSUE-24 |
| MEMORY_MAP | 2 | Complete | Partial — `UnmapViewOfFile`, `FlushViewOfFile` missing from IAT | ✗ ISSUE-24 |
| THREAD_OPS | 3 | Complete (all 5 ops) | **Absent** — `CreateThread`, `SuspendThread`, `ResumeThread`, `TerminateThread`, `GetThreadContext` all missing from IAT; no ET32_ wrappers exist | ✗ ISSUE-24 |
| DLL_OPS | 4 | Complete | Complete (6 hooks present) | ✓ when injection fixed |
| PROCESS_OPS | 5 | Partial — `PROC_INJECT (0x43)` has no broker handler (ISSUE-26) | Partial — `ET32_OpenProcess`, `ET32_CreateProcessA/W` present; `PROC_INJECT` unused | ✗ ISSUE-26 |
| REGISTRY_OPS | 6 | Complete | Complete — open, query, set, create, delete, close all present | ✓ when injection fixed |
| GRAPHICS_OPS | 7 | Partial — `GPU_ENUM_ADAPTERS`, `GPU_CREATE_DEVICE` (D3D9/D3D11), `GPU_HEAVEN_CALL`, `GPU_SUBMIT` implemented; `GPU_ALLOC_VRAM` and `GPU_MAP_VRAM` are placeholders (ISSUE-23) | Minimal — no IAT hooks; C DLL has `ET32_GPU_QueryInfo`, `ET32_GPU_EnumAdapters`, `ET32_GPU_CreateDevice` as explicit exports only | ✗ ISSUE-23 |
| FILE_OPS | 8 | Complete (16 ops) | Complete (all wrappers + IAT entries present) | ✓ when injection fixed |
| SYNC_OPS | 9 | Complete (all 9 ops including semaphore, wait-multiple, reset, close) | Partial — `WaitForSingleObject` missing from IAT (ISSUE-24); `CreateEventA`, `CreateMutexA`, `CreateSemaphoreA`, `ReleaseSemaphore`, `WaitForMultipleObjects`, `ResetEvent`, `CloseHandle` present | ✗ ISSUE-24 (`WaitForSingleObject` critical) |
| NET_OPS | 10 | Complete (all 10 ops including select, sockopt) | Partial — `socket`, `bind`, `send`, `recv` missing from IAT (ISSUE-24); `connect`, `listen`, `accept`, `closesocket` present | ✗ ISSUE-24 |
| PYTHON_OPS | 11 | Complete (all 8 ops) | Complete (all wrappers + IAT entries present) | ✓ when injection fixed |
| COMPOUND_OPS | 12 | **DEAD** — ISSUE-05 (cmd code collision with DYNAMIC_SYSCALL) | Present in DLL | ✗ ISSUE-05 |

### Additional Parity Gaps

**Children spawned by bridged processes — not bridged at spawn time (ISSUE-29).** `ET32_CreateProcessA/W` are IAT-hooked but pass through to the original call without notifying the broker. The child process runs for up to 12 seconds (one monitor scan cycle) before the bridge detects and injects it. Short-lived child processes are never bridged. The comment in the code references flag `ET_BRIDGE_PROC` which is defined nowhere.

**Children — monitor eventually detects them.** The monitor tracks child processes via PPID linkage (et_monitor.py:418–479) and bridges them on the next scan cycle. However, because DLL injection itself is broken (ISSUE-02), children receive the same non-functional shellcode-stub injection regardless.

**WaitForSingleObject critical gap.** The C DLL hooks `CreateEventA`, `CreateMutexA`, and `CreateSemaphoreA`, returning bridge handles. When the 32-bit application calls `WaitForSingleObject` on a bridge handle, the unhooked original `WaitForSingleObject` receives a proxy value in the range `[ET_HANDLE_BASE, ET_HANDLE_MAX]`. Windows treats this as an invalid kernel handle → `WAIT_FAILED`. Every 32-bit application that creates synced objects via the bridge and then waits on them will deadlock or fail immediately.

**SeLockMemoryPrivilege for AWE.** Without this privilege, `AllocateUserPhysicalPages` fails silently (et_awe.py:179). The AWE bookshelf degrades to standard `VirtualAlloc`, capping the 32-bit process at 4GB. No operator-visible error is raised.

**PROC_INJECT has no broker handler.** A 32-bit application requesting DLL injection via `CmdCode.PROC_INJECT (0x43)` receives `ET_ERR_UNSUPPORTED` from the broker. The purpose of this command — inject a 64-bit DLL into a target process — is entirely unimplemented on the broker side.

### Conclusion

No 32-bit program in the current codebase receives any 64-bit capabilities — the injection chain is broken at three independent points (ISSUE-01, ISSUE-02, ISSUE-05). Once the injection chain is repaired, the bridge provides genuine 64-bit parity for MEMORY_BASIC (partial — ISSUE-24), MEMORY_MAP (partial — ISSUE-24), DLL_OPS, REGISTRY_OPS, FILE_OPS, and PYTHON_OPS. THREAD_OPS, GRAPHICS_OPS (partial), SYNC_OPS (partial — WaitForSingleObject), NET_OPS (partial — socket/bind/send/recv), PROCESS_OPS (PROC_INJECT), and COMPOUND_OPS require additional implementation work to reach parity.

---

## ISSUE-26 — MEDIUM: `PROC_INJECT` Broker Handler Missing — 64-bit DLL Injection Unreachable

**File:** `et_host64.py`, `et_math.py:884`
**Lines:** `_handle_process_ops()` dispatch (1040–1122)
**Category:** Not Implemented
**Audit Points:** 2, 4

### Full Trace

`et_math.py:884`:
```python
PROC_INJECT = 0x43  # Inject 64-bit DLL
```

`et_bridge32.c:309`:
```c
#define CMD_PROC_INJECT  0x43
```

`et_host64.py:_handle_process_ops()` (lines 1046–1121) dispatches: `PROC_CREATE`, `PROC_OPEN`, `PROC_INFO`, `PROC_EXIT_CODE`, `PROC_TERMINATE`, `PROC_ENUM`, `PROC_MODULES`, `PROC_WOW64_FS`. There is no `elif code == CmdCode.PROC_INJECT:` branch. The function falls through to:
```python
return self._make_error(pkt, ET_ERR_UNSUPPORTED, f"Unknown PROCESS_OPS code 0x{pkt.cmd_code:02X}")
```

### Consequence

A 32-bit application that sends `PROC_INJECT (0x43)` receives `ET_ERR_UNSUPPORTED`. The intended operation — inject a 64-bit DLL into a target 64-bit process — is completely absent from the broker. This also means the bridge cannot use its own IPC path to perform DLL injection even when the injection chain for bootstrapping is fixed (ISSUE-02).

---

## Priority Resolution Order

The issues are chained. The correct resolution order respects their dependency structure:

### Tier 1 — Bridge Is Currently Non-Functional (Must fix first)

ISSUE-02 blocks everything else. Resolving it also resolves ISSUE-03 and ISSUE-04 as consequences.

1. **ISSUE-02**: Implement DLL injection: `VirtualAllocEx` + `WriteProcessMemory` of the DLL path string + `CreateRemoteThread(LoadLibraryA)` to load `et_bridge32.dll` into the target, then call `ET32_Init(broker_pid)` and `ET32_SetKiFastTrampoline(trampoline_addr)`.
2. **ISSUE-01 + ISSUE-01B**: Replace `make_dispatcher_shellcode()` with a real pipe-communicating dispatcher. Fix `ETStubGenerator.make_stub()` to use `0xE8 <rel32>` instead of `0xFF 0x15 <ptr>`.
3. **ISSUE-04**: Resolved once ISSUE-02 is fixed and `ET32_SetKiFastTrampoline` is called from the broker after trampoline allocation.
4. **ISSUE-03**: Resolved once ISSUE-02 is fixed and `ET32_Init` installs the real hook.

### Tier 2 — Bridge Partially Functional, Major Features Missing

5. **ISSUE-05**: Assign a new command code for `DYNAMIC_SYSCALL` distinct from `COMPOUND_BATCH = 0xB1`.
6. **ISSUE-06**: Implement interactive console CLI with all 8 sub-requirements from the specification.
7. **ISSUE-09**: Replace `CMakeLists.txt` and `main.cpp` with a proper 32-bit DLL CMake target.
8. **ISSUE-07**: Replace the `__syscall_XXXX` lookup with `self._wow64._svc_tbl.lookup(service_number)`.
9. **ISSUE-24**: Implement missing ET32_ wrappers and add IAT entries for all 14 absent functions across THREAD_OPS, MEMORY_BASIC, MEMORY_MAP, SYNC_OPS, and NET_OPS.
10. **ISSUE-23**: Replace `GPU_ALLOC_VRAM` with a real D3D12/Vulkan or `IDXGIResource` path. Implement `GPU_MAP_VRAM` with actual memory mapping.

### Tier 3 — Correctness and Completeness

11. **ISSUE-08**: `_inject_shellcode_fallback()` must return `False`.
12. **ISSUE-10**: Read broker response args into `lpSystemInfo` in `ET32_GetNativeSystemInfo`.
13. **ISSUE-11**: Return peer address from broker response in `ET32_accept`.
14. **ISSUE-12**: Implement remaining `ETAPIGateway` methods for all 12 families.
15. **ISSUE-17**: Rename all d=1 command code defines in `et_bridge32.c` to match `et_math.py` exactly.
16. **ISSUE-18**: Add `CMD_AWE_INIT_SIGNAL (0x0D)` to both Python and C.
17. **ISSUE-19**: Expand VEH to all exception codes; capture exit codes and stdout/stderr.
18. **ISSUE-25**: Implement `--status` with a named-pipe query mechanism against a running instance.
19. **ISSUE-26**: Implement `PROC_INJECT` broker handler in `_handle_process_ops()`.

### Tier 4 — Race Conditions

20. **ISSUE-21**: Add `threading.Lock()` to `ETMetrics` and hold it across all fields in `record()`.
21. **ISSUE-22**: Guard `g_connected` read inside `g_pipe_cs`; restructure `ET32_Shutdown()` to drain in-flight callers before `DeleteCriticalSection`.
22. **ISSUE-13**: Hold `_pipe_lock` across the entire `create_pipe_for_pid` sequence.
23. **ISSUE-14**: Use a `_pending_pids` set inside `_lock` to prevent concurrent engagement of the same PID.

### Tier 5 — Bugs and Polish

24. **ISSUE-15**: Change broker pipe creation to `PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE`.
25. **ISSUE-16**: Replace O(n) loop in `project_address()` with `self._addr64_to_handle.get(addr64)`.
26. **ISSUE-20**: Remove `-lntdll` from the source comment to match `build.bat`.

---

*Audit complete. All 20 issues are individually traced, verified against source, and ready for resolution.*

*P ∘ D ∘ T = E — Michael James Muller / Aevum Defluo*
# ETPL_self.pdt Module Split — Stage 4 Staging Document

## What Was Done in Stage 4

Extracted 7 modules covering lines 12700–23048. Also investigated the previously-flagged unknown gap between `ctypes` and `gc` — confirmed no gap exists; `ctypes` runs continuously from 23049 to 23921.

| File | Source Lines | Size | Notes |
|------|-------------|------|-------|
| `trace_dataclasses.pdt` | 12700–13702 | 1003 lines | stdlib `dataclasses` |
| `trace_enum.pdt` | 13703–15259 | 1557 lines | stdlib `enum` |
| `trace_decimal.pdt` | 15260–15449 | 190 lines | stdlib `decimal` — fixed/float arithmetic |
| `trace_llvmlite.pdt` | 15450–15695 | 246 lines | third-party `llvmlite` — LLVM Python bindings |
| `trace_capstone.pdt` | 15696–17082 | 1387 lines | third-party `capstone` — disassembly engine |
| `trace_pefile.pdt` | 17083–21006 | 3924 lines | third-party `pefile` — PE binary parser |
| `trace_psutil.pdt` | 21007–23048 | 2042 lines | third-party `psutil` — process/system info |

---

## Cumulative Progress

| Stage | Lines Covered | Modules Extracted |
|-------|--------------|-------------------|
| Stage 1 | 1–1231 | header, ET_Math_Native, ET_Platform_Native, trace_sys |
| Stage 2 | 1232–3572 | trace_os, trace_time, trace_re, trace_math_cext, trace_struct, trace_hashlib, trace_copy |
| Stage 3 | 3573–12699 | trace_traceback, trace_platform, trace_json, trace_argparse, trace_ast, trace_typing |
| Stage 4 | 12700–23048 | trace_dataclasses, trace_enum, trace_decimal, trace_llvmlite, trace_capstone, trace_pefile, trace_psutil |
| **Remaining** | **23049–44524** | **~21,476 lines, ~16 modules** |

---

## What Needs to Be Done in Stage 5

| Target File | Start Line | End Line | Module |
|-------------|-----------|---------|--------|
| `trace_ctypes.pdt` | 23049 | 23921 | `ctypes` — confirmed continuous, no gap |
| `trace_gc.pdt` | 23922 | 23972 | `gc` C-extension |
| `trace_tempfile.pdt` | 23974 | 25223 | `tempfile` |
| `trace_collections.pdt` | 25224 | 26370 | `collections` |
| `trace_inspect.pdt` | 26371 | 29627 | `inspect` |
| `trace_threading.pdt` | 29628 | 30838 | `threading` |
| `trace_mmap.pdt` | 30839 | 30854 | `mmap` C-extension |
| `trace_weakref.pdt` | 30855 | 31625 | `weakref` |

After Stage 5, remaining: `multiprocessing` (31626–31751), `shutil` (31752–33632), `subprocess` (33633–36102), `importlib` (36103–36376), `marshal` C-ext (36377–36392), and the large ETPL self-source block (36393–44524).

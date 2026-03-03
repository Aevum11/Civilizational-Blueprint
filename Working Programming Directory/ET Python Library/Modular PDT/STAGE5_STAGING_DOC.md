# ETPL_self.pdt Module Split — Stage 5 Staging Document

## What Was Done in Stage 5

Extracted 8 modules covering lines 23049–31625.

| File | Source Lines | Size | Notes |
|------|-------------|------|-------|
| `trace_ctypes.pdt` | 23049–23921 | 873 lines | stdlib `ctypes` — C data types interface |
| `trace_gc.pdt` | 23922–23972 | 51 lines | C-extension `gc` — garbage collector stubs |
| `trace_tempfile.pdt` | 23974–25223 | 1250 lines | stdlib `tempfile` — temporary file/dir management |
| `trace_collections.pdt` | 25224–26370 | 1147 lines | stdlib `collections` — ChainMap, Counter, deque, etc. |
| `trace_inspect.pdt` | 26371–29627 | 3257 lines | stdlib `inspect` — live object introspection |
| `trace_threading.pdt` | 29628–30838 | 1211 lines | stdlib `threading` — thread model |
| `trace_mmap.pdt` | 30839–30854 | 16 lines | C-extension `mmap` — memory-mapped file stubs |
| `trace_weakref.pdt` | 30855–31625 | 771 lines | stdlib `weakref` — weak reference support |

**Note on line 23973:** Single blank line between `gc` end and `tempfile` start — absorbed as leading blank in `trace_tempfile.pdt` (harmless).

---

## Cumulative Progress

| Stage | Lines Covered | Modules Extracted |
|-------|--------------|-------------------|
| Stage 1 | 1–1231 | header, ET_Math_Native, ET_Platform_Native, trace_sys |
| Stage 2 | 1232–3572 | trace_os, trace_time, trace_re, trace_math_cext, trace_struct, trace_hashlib, trace_copy |
| Stage 3 | 3573–12699 | trace_traceback, trace_platform, trace_json, trace_argparse, trace_ast, trace_typing |
| Stage 4 | 12700–23048 | trace_dataclasses, trace_enum, trace_decimal, trace_llvmlite, trace_capstone, trace_pefile, trace_psutil |
| Stage 5 | 23049–31625 | trace_ctypes, trace_gc, trace_tempfile, trace_collections, trace_inspect, trace_threading, trace_mmap, trace_weakref |
| **Remaining** | **31626–44524** | **~12,899 lines, ~6 modules** |

---

## What Needs to Be Done in Stage 6 (FINAL)

This is the last stage — all remaining modules confirmed from boundary checks:

| Target File | Start Line | End Line | Module | Notes |
|-------------|-----------|---------|--------|-------|
| `trace_multiprocessing.pdt` | 31626 | 31751 | `multiprocessing` | stdlib |
| `trace_shutil.pdt` | 31752 | 33632 | `shutil` | stdlib |
| `trace_subprocess.pdt` | 33633 | 36102 | `subprocess` | stdlib |
| `trace_importlib.pdt` | 36103 | 36376 | `importlib` | stdlib |
| `trace_marshal.pdt` | 36377 | 36392 | `marshal` | C-extension — 4 stubs only (dump, dumps, load, loads) |
| `ETPL_source.pdt` | 36393 | 44524 | ETPL.py self-translation | Marked `@ETPL:entry-source ETPL.py` — this is the translated ETPL toolchain source itself; largest block at ~8132 lines |

**For `ETPL_source.pdt`:** After extraction, check whether it has internal sub-section markers worth a further split in a future pass. The entry point is `verify_etpl` per the file header.

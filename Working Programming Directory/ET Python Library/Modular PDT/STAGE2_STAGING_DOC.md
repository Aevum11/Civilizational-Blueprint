# ETPL_self.pdt Module Split — Stage 2 Staging Document

## What Was Done in Stage 2

Extracted 7 modules covering lines 1233–3572. All boundaries confirmed clean before cutting.

| File | Source Lines | Size | Notes |
|------|-------------|------|-------|
| `trace_os.pdt` | 1233–2501 | 1269 lines | stdlib `os` — full posix/nt OS interface trace |
| `trace_time.pdt` | 2502–2553 | 52 lines | C-extension `time` — clock, sleep, struct_time stubs |
| `trace_re.pdt` | 2555–2880 | 326 lines | stdlib `re` — regex engine trace |
| `trace_math_cext.pdt` | 2881–3001 | 121 lines | C-extension `math` — Python math module stubs (NOT ET_Math_Native; this is the trace of the original) |
| `trace_struct.pdt` | 3003–3028 | 26 lines | stdlib `struct` + embedded `_struct` C-extension module-start/end |
| `trace_hashlib.pdt` | 3029–3289 | 261 lines | stdlib `hashlib` — all hash constructors and algorithm sets |
| `trace_copy.pdt` | 3290–3572 | 283 lines | stdlib `copy` — shallow/deep copy dispatch |

**Naming note:** `trace_math_cext.pdt` is deliberately named to distinguish it from `ET_Math_Native.pdt`. It is the stub trace of CPython's `math` C-extension (what ETPL replaces), not the ET implementation.

**Gap note:** Line 2554 is a blank line between `@ETPL:trace-c-extension-end time` and `@ETPL:trace-stdlib __init__.py // module=re`. It is absorbed into `trace_re.pdt` as its first line (harmless blank).

---

## Cumulative Progress

| Stage | Lines Covered | Modules Extracted |
|-------|--------------|-------------------|
| Stage 1 | 1–1231 | header, ET_Math_Native, ET_Platform_Native, trace_sys |
| Stage 2 | 1232–3572 | trace_os, trace_time, trace_re, trace_math_cext, trace_struct, trace_hashlib, trace_copy |
| **Remaining** | **3573–44524** | **~40,952 lines, ~29 modules** |

---

## What Needs to Be Done in Stage 3

Next batch of stdlib traces. All start lines confirmed from Stage 1 map:

| Target File | Start Line | End Line | Module |
|-------------|-----------|---------|--------|
| `trace_traceback.pdt` | 3573 | 5044 | `traceback` |
| `trace_platform.pdt` | 5045 | 6606 | `platform` |
| `trace_json.pdt` | 6607 | 6743 | `json` |
| `trace_argparse.pdt` | 6744 | 8656 | `argparse` |
| `trace_ast.pdt` | 8657 | 10680 | `ast` (contains embedded `_ast` C-ext block at 8827–9078 — keep inside, do not sub-split) |
| `trace_typing.pdt` | 10681 | 12699 | `typing` |

**Before cutting each:** run `sed -n '<start>,<start+2>p'` and `sed -n '<end-1>,<end+1>p'` to confirm boundaries haven't shifted.

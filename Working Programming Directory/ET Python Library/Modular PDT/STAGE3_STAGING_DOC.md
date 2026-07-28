# ETPL_self.pdt Module Split — Stage 3 Staging Document

## What Was Done in Stage 3

Extracted 6 modules covering lines 3573–12699.

| File | Source Lines | Size | Notes |
|------|-------------|------|-------|
| `trace_traceback.pdt` | 3573–5044 | 1472 lines | stdlib `traceback` — stack trace formatting |
| `trace_platform.pdt` | 5045–6606 | 1562 lines | stdlib `platform` — OS/arch/version detection |
| `trace_json.pdt` | 6607–6743 | 137 lines | stdlib `json` — JSON encode/decode |
| `trace_argparse.pdt` | 6744–8656 | 1913 lines | stdlib `argparse` — CLI argument parsing |
| `trace_ast.pdt` | 8657–10680 | 2024 lines | stdlib `ast` — Python AST; contains embedded `_ast` C-ext block (lines 8827–9078 of source) kept inside |
| `trace_typing.pdt` | 10681–12699 | 2019 lines | stdlib `typing` — type annotation support |

---

## Cumulative Progress

| Stage | Lines Covered | Modules Extracted |
|-------|--------------|-------------------|
| Stage 1 | 1–1231 | header, ET_Math_Native, ET_Platform_Native, trace_sys |
| Stage 2 | 1232–3572 | trace_os, trace_time, trace_re, trace_math_cext, trace_struct, trace_hashlib, trace_copy |
| Stage 3 | 3573–12699 | trace_traceback, trace_platform, trace_json, trace_argparse, trace_ast, trace_typing |
| **Remaining** | **12700–44524** | **~31,825 lines, ~23 modules** |

---

## What Needs to Be Done in Stage 4

Next batch from the Stage 1 map:

| Target File | Start Line | End Line | Module |
|-------------|-----------|---------|--------|
| `trace_dataclasses.pdt` | 12700 | 13702 | `dataclasses` |
| `trace_enum.pdt` | 13703 | 15259 | `enum` |
| `trace_decimal.pdt` | 15260 | 15449 | `decimal` |
| `trace_llvmlite.pdt` | 15450 | 15695 | `llvmlite` |
| `trace_capstone.pdt` | 15696 | 17082 | `capstone` |
| `trace_pefile.pdt` | 17083 | 21006 | `pefile` |
| `trace_psutil.pdt` | 21007 | 23048 | `psutil` |

**Verify boundaries before cutting** — especially the gap between `psutil` end (23048) and `ctypes` start (23049). Also confirm what sits between `ctypes` end and `gc` start (the unknown block noted in Stage 1, around lines 23049–23921).

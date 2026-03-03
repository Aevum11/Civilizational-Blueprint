# ETPL_self.pdt Module Split — Stage 6 Staging Document (FINAL)

## What Was Done in Stage 6

Extracted the final 6 modules covering lines 31626–44524. The split is now **complete**.

| File | Source Lines | Size | Notes |
|------|-------------|------|-------|
| `trace_multiprocessing.pdt` | 31626–31751 | 126 lines | stdlib `multiprocessing` |
| `trace_shutil.pdt` | 31752–33632 | 1881 lines | stdlib `shutil` — file/dir copy & archive |
| `trace_subprocess.pdt` | 33633–36102 | 2470 lines | stdlib `subprocess` — process spawning |
| `trace_importlib.pdt` | 36103–36376 | 274 lines | stdlib `importlib` — pure-Python import |
| `trace_marshal.pdt` | 36377–36392 | 16 lines | C-extension `marshal` — 4 stubs only: dump, dumps, load, loads |
| `ETPL_source.pdt` | 36393–44524 | 8132 lines | `@ETPL:entry-source ETPL.py` — translated ETPL toolchain itself; entry point `verify_etpl` / `ETPLCLI` |

**Line count note:** Modules sum to 44,520 vs source 44,524. The 4-line difference is 4 inter-module blank separator lines that fall between module boundaries (at lines 1232, 2554, 3002, 23973) and are not assigned to any module. This is correct and expected.

---

## Complete Module Inventory (All 38 Files)

| # | File | Lines | Category |
|---|------|-------|----------|
| 1 | `ETPL_header.pdt` | 14 | Bootstrap header |
| 2 | `ET_Math_Native.pdt` | 481 | ET native library — math |
| 3 | `ET_Platform_Native.pdt` | 609 | ET native library — platform/sys/time/marshal |
| 4 | `trace_sys.pdt` | 127 | C-extension trace |
| 5 | `trace_os.pdt` | 1269 | stdlib trace |
| 6 | `trace_time.pdt` | 52 | C-extension trace |
| 7 | `trace_re.pdt` | 326 | stdlib trace |
| 8 | `trace_math_cext.pdt` | 121 | C-extension trace |
| 9 | `trace_struct.pdt` | 26 | stdlib trace |
| 10 | `trace_hashlib.pdt` | 261 | stdlib trace |
| 11 | `trace_copy.pdt` | 283 | stdlib trace |
| 12 | `trace_traceback.pdt` | 1472 | stdlib trace |
| 13 | `trace_platform.pdt` | 1562 | stdlib trace |
| 14 | `trace_json.pdt` | 137 | stdlib trace |
| 15 | `trace_argparse.pdt` | 1913 | stdlib trace |
| 16 | `trace_ast.pdt` | 2024 | stdlib trace (contains embedded `_ast` C-ext) |
| 17 | `trace_typing.pdt` | 2019 | stdlib trace |
| 18 | `trace_dataclasses.pdt` | 1003 | stdlib trace |
| 19 | `trace_enum.pdt` | 1557 | stdlib trace |
| 20 | `trace_decimal.pdt` | 190 | stdlib trace |
| 21 | `trace_llvmlite.pdt` | 246 | third-party trace |
| 22 | `trace_capstone.pdt` | 1387 | third-party trace |
| 23 | `trace_pefile.pdt` | 3924 | third-party trace |
| 24 | `trace_psutil.pdt` | 2042 | third-party trace |
| 25 | `trace_ctypes.pdt` | 873 | stdlib trace |
| 26 | `trace_gc.pdt` | 51 | C-extension trace |
| 27 | `trace_tempfile.pdt` | 1250 | stdlib trace |
| 28 | `trace_collections.pdt` | 1147 | stdlib trace |
| 29 | `trace_inspect.pdt` | 3257 | stdlib trace |
| 30 | `trace_threading.pdt` | 1211 | stdlib trace |
| 31 | `trace_mmap.pdt` | 16 | C-extension trace |
| 32 | `trace_weakref.pdt` | 771 | stdlib trace |
| 33 | `trace_multiprocessing.pdt` | 126 | stdlib trace |
| 34 | `trace_shutil.pdt` | 1881 | stdlib trace |
| 35 | `trace_subprocess.pdt` | 2470 | stdlib trace |
| 36 | `trace_importlib.pdt` | 274 | stdlib trace |
| 37 | `trace_marshal.pdt` | 16 | C-extension trace |
| 38 | `ETPL_source.pdt` | 8132 | ETPL.py self-translation — toolchain source |
| | **TOTAL** | **44,520** | (+4 inter-module blank lines = 44,524 source) |

---

## Suggested Next Steps

Now that the file is fully split, the issue-finding work can begin. Suggested areas to investigate per module:

- **`ET_Math_Native.pdt` / `ET_Platform_Native.pdt`** — verify ET series bounds, D-descriptor correctness, and that all compatibility binding names match CPython's API exactly
- **C-extension traces** (`trace_sys`, `trace_time`, `trace_math_cext`, `trace_gc`, `trace_mmap`, `trace_marshal`) — all callables are P stubs; check constants and flags for accuracy against CPython 3.13
- **Third-party traces** (`trace_llvmlite`, `trace_capstone`, `trace_pefile`, `trace_psutil`) — these are the most likely to have version-drift issues or incomplete stubs
- **`ETPL_source.pdt`** — the most complex module; may benefit from a further internal split pass along its own class/function boundaries before deep inspection

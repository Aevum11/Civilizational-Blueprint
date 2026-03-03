# ETPL Comprehensive Guide — Language-Level Additions Required
## Gap Analysis: Pure Language Features Missing or Incomplete in the Current Guide

**Source:** ETPL.py (tokenizer, parser, interpreter, compiler), ETPL_source.pdt, project docs  
**Scope:** Language features ONLY — syntax, semantics, operators, types, built-ins, constants  
**Purpose:** These are the gaps the full guide must close so every .pdt module can be audited against it  

---

## GAP 1: OPERATOR PRECEDENCE TABLE — COMPLETELY ABSENT

The guide lists operators in a reference table but **never defines their precedence order**.
Every .pdt module written without this will have ambiguous expressions. The full precedence
hierarchy from the parser is (lowest to highest):

| Level | Operator(s) | Associativity |
|-------|-------------|---------------|
| 1 (lowest) | `\|\|` logical OR | left |
| 2 | `&&` logical AND | left |
| 3 | `\|` bitwise OR | left |
| 4 | `^` bitwise XOR (in bitwise context) | left |
| 5 | `&` bitwise AND | left |
| 6 | `==` `!=` `<` `>` `<=` `>=` `≤` `≥` `≠` `≈` `~=` `=` | left, non-assoc |
| 7 | `<<` `>>` bit shift | left |
| 8 | `+` `-` additive | left |
| 9 | `*` `/` `%` `÷` multiplicative | left |
| 10 | `^` `**` exponent (in math context) | right |
| 11 (highest) | `∘` `::` `->` compose/apply | right |
| 12 | `[n]` `[a:b]` `D member` postfix | left |

**ET Derivation:** Precedence is ordered by the P∘D∘T hierarchy — substrate operations (arithmetic on P) are inner; descriptor application (∘, D) is outer.

---

## GAP 2: COMPLETE OPERATOR ALIAS TABLE — PARTIALLY ABSENT

The guide shows some operators but not their full alias set. Every operator has both a Unicode
canonical form and ASCII aliases. All are recognized by the tokenizer:

| Unicode (canonical) | ASCII alias | Keyword alias | Token |
|---------------------|-------------|---------------|-------|
| `→` | `->` | — | ARROW |
| `∘` | `::` | `compose` | COMPOSE |
| `∘∘` | (none) | — | DOUBLE_COMPOSE |
| `λ` | (none) | `lambda` | LAMBDA |
| `∞` | `inf` | `Infinity` | INFINITY |
| `Ω` | (none) | `Omega` | OMEGA |
| `ℵ` | (none) | `aleph` | ALEPH |
| `≤` | `<=` | — | LE |
| `≥` | `>=` | — | GE |
| `≠` | `!=` | — | NE |
| `≈` | `~=` | — | APPROX |
| `÷` | (none — `//` is a comment) | — | FLOOR_DIV |
| `∑` | (none) | `sum` | SIGMA |
| `∏` | (none) | `prod` | PI_PROD |
| `∫` | (none) | — | INTEGRAL |
| `∇` | (none) | `nabla`, `grad` | NABLA |
| `√` | (none) | `sqrt` | SQRT |
| `ψ` | (none) | `psi` | PSI |
| `&&` | — | `and` | LOGICAL_AND |
| `\|\|` | — | `or` | LOGICAL_OR |
| `!` | — | `not` | LOGICAL_NOT |
| `**` | — | — | DOUBLE_STAR (exponent alias) |

**Key rule documented nowhere:** The `=` single equals is also accepted as `==` in comparison
context. The parser includes `TokenType.EQUALS: '='` in its comparison operator dispatch.

---

## GAP 3: THE APPROXIMATE EQUALITY OPERATOR `≈` — COMPLETELY ABSENT

The `≈` operator (Unicode U+2248, ASCII alias `~=`) is a first-class ETPL comparison operator.
It is NOT in the guide's operator table or reference section.

**Semantics:**
- For numeric values: returns `1` if `|a - b| < 1e-9`, else `0`
- For non-numeric values: falls back to `==` comparison

```pdt
P a = 1.0000000001
P b = 1.0
P result = a ≈ b          // 1 (approximately equal within 1e-9)
P result2 = a ~= b        // same — ASCII alias

// In conditionals:
if computed_pi ≈ 3.14159265358979 → sovereign_print ∘ "correct to ET precision"
```

**ET Derivation:** `≈` is the ET Traverser's indeterminate boundary condition — the margin
within which two Exception states are considered the same grounding. It corresponds to
BASE_VARIANCE / MANIFOLD_SYMMETRY^(3/2) ≈ 1e-9.

---

## GAP 4: `∘∘` DOUBLE COMPOSE OPERATOR — COMPLETELY ABSENT

The `∘∘` (double compose) operator is tokenized as `DOUBLE_COMPOSE`. It is absent from
every part of the guide.

**Semantics:** Applies two levels of composition simultaneously — `f ∘∘ g` = `f(g(·))` as
a single composition step without intermediate binding.

```pdt
D f = λ x . x + 1
D g = λ x . x * 2

// Standard compose (two steps):
P r1 = f ∘ (g ∘ 3)       // g(3)=6, f(6)=7

// Double compose (one step):
D fog = f ∘∘ g            // creates f∘g as a single Descriptor
P r2 = fog ∘ 3             // 7
```

---

## GAP 5: COMPLETE TYPE SYSTEM AND TYPE COERCION RULES — ABSENT

The guide implies types but never formally documents them. The type system is:

### ETPL Value Types

| Type | ETPL representation | ET primitive | Numeric coercion |
|------|---------------------|--------------|-----------------|
| Integer | `42`, `-7` | D (finite constraint) | self |
| Float | `3.14`, `1e-9` | D (finite constraint) | self |
| String | `"hello"` | D (finite sequence of P) | raises if not numeric |
| Boolean (ET trinary: true) | `1` | E (grounded) | `1` |
| Boolean (ET trinary: false) | `0` | I (incoherent) | `0` |
| Indeterminate | `[0/0]` | T (agency) | `0` in classical mode |
| Null/unbound | `P` | P (substrate) | `0` |
| Manifold | `manifold [...]` | PDT bound | `len(manifold)` |
| Infinity | `Infinity`, `∞`, `inf` | P-unbounded | `float('inf')` |
| Omega | `Ω`, `Omega` | absolute ∞ | `float('inf')` |
| Aleph | `aleph`, `ℵ` | countable ∞ | `float('inf')` |
| NaN | `float("nan")` | indeterminate bound | `float('nan')` |

### Type Coercion Rules (the `_to_number` semantics)

When arithmetic is performed between values of different types:
- **Numeric + Numeric** → arithmetic result
- **String + anything** → string concatenation (`str(right)` appended)
- **`P` (null) in numeric context** → `0`
- **Manifold in numeric context** → `len(manifold)` (its cardinality `|m|`)
- **Bool (`1`/`0`) in numeric context** → `1` or `0`
- **Float string (`"3.14"`) in numeric context** → `float("3.14")`
- **Integer string (`"42"`) in numeric context** → `int("42")`

**Critical undocumented rule:** The `+` operator checks for string on EITHER side first:
```pdt
P a = "value: "
P b = 42
P result = a + b    // "value: 42" — string concat because left side is string
P result2 = b + a   // also string concat because right side is string
```

---

## GAP 6: ET MATH CONSTANTS — INCOMPLETE AND CONTAINS A TYPO

### Typo in current guide (line 1402):
`P ko = KOIDA_RATIO` — **WRONG.** The constant is `KOIDE_RATIO` (after Yoshio Koide).
This typo must be corrected throughout the guide.

### Missing constants (all available without any import):

| Constant | Value | ET Derivation |
|----------|-------|---------------|
| `MANIFOLD_SYMMETRY` | `12` | 3 primitives × 4 binding states |
| `BASE_VARIANCE` | `1/12` | uniform distribution variance of 12-state manifold |
| `KOIDE_RATIO` | `2/3` | mass/charge triadic coupling constant |
| `WHILE_LOOP_FINITE_BOUND` | `144` | MANIFOLD_SYMMETRY² = 12² |
| `pi` | 3.14159… | ET_PI via Machin formula (lowercase alias) |
| `e` | 2.71828… | ET_E via Taylor series (lowercase alias) |
| `tau` | 6.28318… | 2π (lowercase alias) |
| `phi` | 1.61803… | (1 + √5) / 2 — golden ratio, ET-derived |
| `inf` | ∞ | positive infinity (lowercase alias) |
| `nan` | NaN | not-a-number (lowercase alias) |
| `ET_MATH_PI` | 3.14159… | internal constant (uppercase form) |
| `ET_MATH_E` | 2.71828… | internal constant (uppercase form) |
| `ET_MATH_LN2` | 0.69314… | ln(2) via arctanh series |
| `ET_MATH_LN10` | 2.30258… | ln(10) via arctanh series |

**ET Physics note:** ET_PI is derived via Machin's formula (zero external dependencies):
`π/4 = 4·arctan(1/5) − arctan(1/239)` — the Traverser series is bounded by N=144=MANIFOLD_SYMMETRY².

---

## GAP 7: COMPLETE MATHEMATICAL BUILT-IN FUNCTION LIBRARY — INCOMPLETE

The guide lists a partial set. These are ALL available without import:

### Currently documented (partial list in guide):
`sin`, `cos`, `tan`, `log`, `log2`, `log10`, `sqrt`, `exp`, `abs`, `pow`, `atan`, `atan2`,
`floor`, `ceil`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `hypot`, `degrees`, `radians`,
`factorial`, `gcd`

### Missing from guide (all available):
| Function | Signature | Description |
|----------|-----------|-------------|
| `log1p(x)` | `log1p(x)` | ln(1+x), accurate for small x |
| `trunc(x)` | `trunc(x)` | truncate toward zero |
| `isnan(x)` | `isnan(x)` → 1/0 | returns 1 if x is NaN |
| `isinf(x)` | `isinf(x)` → 1/0 | returns 1 if x is infinite |
| `isfinite(x)` | `isfinite(x)` → 1/0 | returns 1 if x is finite |
| `isclose(a, b)` | `isclose(a, b)` → 1/0 | returns 1 if a ≈ b within default tolerance |
| `lcm(a, b)` | `lcm(a, b)` | least common multiple |
| `erf(x)` | `erf(x)` | error function |
| `erfc(x)` | `erfc(x)` | complementary error function |
| `gamma(x)` | `gamma(x)` | gamma function Γ(x) |
| `lgamma(x)` | `lgamma(x)` | log-gamma function |
| `fabs(x)` | `fabs(x)` | float absolute value |
| `fmod(x, y)` | `fmod(x, y)` | float modulo |
| `modf(x)` | `modf(x)` → manifold [frac, int] | fractional and integer parts |
| `frexp(x)` | `frexp(x)` → manifold [m, e] | mantissa and exponent |
| `ldexp(m, e)` | `ldexp(m, e)` | m × 2^e |
| `copysign(x, y)` | `copysign(x, y)` | magnitude of x with sign of y |
| `map(f, coll)` | `map(f, manifold)` | apply Descriptor to each element |
| `filter(f, coll)` | `filter(f, manifold)` | keep elements satisfying D |

**ET Derivation of all series:** Every transcendental function uses N=144=MANIFOLD_SYMMETRY²
as the iteration bound on its T-series. This is a language constant, not a library choice.

---

## GAP 8: `map` AND `filter` AS FIRST-CLASS BUILT-INS — ABSENT

`map` and `filter` are tokenized keywords (not just functions) and are first-class ETPL built-ins:

```pdt
D double = λ x . x * 2
D is_even = λ x . (x % 2) == 0

P numbers = manifold [1, 2, 3, 4, 5]

P doubled  = map(double, numbers)          // manifold [2, 4, 6, 8, 10]
P evens    = filter(is_even, numbers)      // manifold [2, 4]
```

**ET Derivation:** `map` is a T-traversal applying D to every P in a manifold. `filter` is
a T-traversal with a D-predicate gate: elements that fail the D-constraint are excluded.

---

## GAP 9: MANIFOLD INDEX ASSIGNMENT — ABSENT

The guide documents reading from manifold indices (`arr[0]`) but never documents WRITING
to them. Index assignment is a valid ETPL statement:

```pdt
P arr = manifold [10, 20, 30]
P arr[1] = 99              // arr is now manifold [10, 99, 30]
P arr[0] = arr[0] + 1      // arr is now manifold [11, 99, 30]
```

**ET Derivation:** Assigning to `arr[n]` is a T-traversal to index n, followed by a
D-rebinding of that position's P-substrate to the new value.

**This pattern is used extensively in ET_Runtime_Native.pdt and must be understood
to correctly audit any module that builds or modifies buffers.**

---

## GAP 10: EARLY RETURN WITH BARE `→` — INSUFFICIENTLY DOCUMENTED

The guide covers `→ E value` but not the bare `→ value` form (without `E`). These are
different and both are used throughout the codebase:

```pdt
D safe_fn = λ x . {
    → if x < 0 → 0              // bare → : conditional early return (no E needed here)
    P result = x * x
    → result                     // bare → : return result (ETPL's return statement)
}
```

**Rule:** Inside a Descriptor body:
- `→ E value` is the **exception-ground** path (the else branch of an if-path)
- `→ value` (bare) is a **direct traversal/return** — routes execution to the value immediately
- Both ground the Descriptor to the given value

**The bare `→` is the primary return mechanism in all ET-native module code.**

---

## GAP 11: FULL TRY/ATTEMPT EXCEPTION HANDLING PATTERNS — INCOMPLETE

The guide briefly mentions `T attempt = → P` but does not document the full pattern,
single-exception catch, multi-exception catch, or the exception binding variable:

### Pattern 1 — Basic attempt (try with no variable):
```pdt
T attempt = → P                    // try block begins
    P result = risky_operation()
→ E SomeException (_)              // catch SomeException; _ binds the exception object
    P result = default_value
```

### Pattern 2 — Named exception variable:
```pdt
T attempt = → P
    P x = dangerous_compute()
→ E ValueError (err)               // err binds the ValueError instance
    sovereign_print ∘ "Error: " + err
```

### Pattern 3 — Multi-exception catch (manifold as type list):
```pdt
T attempt = → P
    P val = parse_input()
→ E manifold [TypeError, ValueError] (_)    // catches either type
    P val = 0
```

### Pattern 4 — Multiple except clauses (chained):
```pdt
T attempt = → P
    P x = compute()
→ E TypeError (_)
    P x = 0
→ E ValueError (_)
    P x = -1
→ E Exception (_)
    P x = P                        // catch-all: ground to null
```

**ET Derivation:** The `T attempt = → P` pattern traverses to the P-substrate (the unbound
ground). An exception is an E-state that did not resolve through the intended D-path. The
`→ E ExcType (name)` path catches it and reroutes the traversal to an alternate grounding.

---

## GAP 12: LOOP INDEX SCOPING IN NESTED LOOPS — ABSENT

The guide mentions `_loop_index` but never documents how it behaves in nested loops.
This is critical for auditing any module with nested loop structures:

**Rule:** Each `∞` loop that is the direct body of a D-scope (Descriptor body or block `{ }`)
has its own `_loop_index`. Outer loop indices are NOT automatically accessible by name
from inner loops in separate D-scopes.

```pdt
// OUTER loop — _loop_index is the outer index
T outer = ∞ (
    {
        P i = _loop_index          // capture outer index before entering inner scope
        T inner = ∞ (
            {
                P j = _loop_index  // this is the INNER _loop_index (0-based)
                sovereign_print ∘ "i=" + i + " j=" + j
            }
        ) (D 3)
    }
) (D 3)
```

**Key:** To use the outer loop index inside an inner loop, capture it into a Point (`P i = _loop_index`) in the outer scope before the inner loop body.

---

## GAP 13: `float("inf")` AND `float("nan")` AS ETPL EXPRESSIONS — ABSENT

The guide documents `Infinity`, `∞`, `inf` as literals, but the `float("inf")` and
`float("nan")` call-form is used throughout ET source code and must be documented:

```pdt
P pos_inf = float("inf")           // equivalent to Infinity
P neg_inf = float("-inf")          // negative infinity
P not_a_number = float("nan")      // NaN value

// These are particularly relevant in ET math boundary conditions:
D safe_log = λ x . {
    if x <= 0 → float("-inf") → E log(x)
}
```

**These are valid ETPL expressions, not Python escape hatches.** The interpreter resolves
`float("inf")` and `float("nan")` to their numeric equivalents.

---

## GAP 14: `len()` AS AN ALTERNATIVE TO `|collection|` — ABSENT

The guide documents the `|collection|` cardinality notation but does not document that
`len(collection)` is also available and equivalent:

```pdt
P arr = manifold [1, 2, 3, 4]
P size1 = |arr|             // 4 — canonical ET form
P size2 = len(arr)          // 4 — also valid; both tokenized
```

`len` is a built-in available without import. The `|...|` form is preferred in ETPL
but `len()` appears in translated code and must be recognized.

---

## GAP 15: TYPE CONVERSION FUNCTIONS — ABSENT

The following type conversion functions are available as built-ins:

```pdt
P s = str(42)              // "42" — convert to string
P n = int("42")            // 42 — convert to integer
P f = float("3.14")        // 3.14 — convert to float
P b = bool(1)              // 1 (true)
P b2 = bool(0)             // 0 (false)
P b3 = bool(P)             // 0 (null → false)
```

**ET Derivation:** Type conversion is a D-reconfiguration on the P-substrate — the same
value, bound by a different descriptor constraint (the target type's D-rules).

---

## GAP 16: `isinstance()`, `type()`, `repr()` FUNCTIONS — ABSENT

These are available as built-ins in ETPL programs:

```pdt
P x = 42
P is_int = isinstance(x, int)         // 1 (true)
P is_str = isinstance(x, str)         // 0

// Multi-type check (manifold as type union):
P is_num = isinstance(x, manifold [int, float])    // 1

P t = type(x)                          // returns the type object
P r = repr(x)                          // "42" (string representation)
```

The `manifold [type1, type2]` pattern in `isinstance` is the canonical ETPL way to
check membership in a set of types, mirroring Python's `isinstance(x, (int, float))`.

---

## GAP 17: `set()` BUILT-IN — ABSENT

The `set()` function creates an empty or populated set, available without import:

```pdt
P empty_set = set()                    // empty set
P visited = set()                      // common pattern: track visited items
T add_item = → (visited D add)("item1")
```

Sets support `.add()`, `.discard()`, `.union()`, `.intersection()`, `.difference()`.

---

## GAP 18: KEYWORD ALIASES FOR LOGICAL OPERATORS — ABSENT

The tokenizer accepts `and`, `or`, `not` as aliases for `&&`, `||`, `!`:

```pdt
P a = 1
P b = 0
P c = a and b              // 0 (same as a && b)
P d = a or b               // 1 (same as a || b)
P e = not a                // 0 (same as !a)
```

These keyword forms appear in translated Python code and must be recognized as valid.

---

## GAP 19: THE ET SERIES BOUND N=144 AS A LANGUAGE CONSTANT — ABSENT

Every ET-native mathematical series (Taylor, Leibniz, Newton-Raphson, etc.) uses N=144
as its iteration bound. This is not arbitrary — it is **MANIFOLD_SYMMETRY² = 12² = 144**,
which is `WHILE_LOOP_FINITE_BOUND`. This is a fundamental language-level constant:

```pdt
// All ET-native series use this bound:
P series_bound = WHILE_LOOP_FINITE_BOUND    // 144

// Example: ET-native sine computation pattern
D et_sin_series = λ x . {
    P result = 0.0
    P term = x
    T series = ∞ (
        {
            P k = _loop_index
            P result = result + term
            P term = (0 - term) * x * x / ((2*k + 2) * (2*k + 3))
        }
    ) (D WHILE_LOOP_FINITE_BOUND)           // bounded by 144
    → result
}
```

**Language rule:** Any ET-native computation must use `WHILE_LOOP_FINITE_BOUND` (or an
explicit 144) as its series/iteration ceiling. Using an unbounded loop violates ET
Traverser Finiteness (Eq 219).

---

## GAP 20: ET DIVISION SEMANTICS — NEEDS DEEPER TREATMENT

The guide mentions 0/0=0 and 1/0=Infinity briefly in Section 13 but does not give
the complete specification. The full ET division semantics are:

```pdt
P r1 = 1 / 0       // Infinity    — P-substrate dominates over empty D (positive)
P r2 = -1 / 0      // -Infinity   — negative P-substrate dominates
P r3 = 0 / 0       // 0           — indeterminate form resolved to ground state
P r4 = 0 / 1       // 0           — zero over non-zero
P r5 = Infinity / Infinity  // 1  — ET-resolved: equal infinities = unity
```

**Exception behavior:** Division errors do NOT raise exceptions in ET — they evaluate
to infinity or zero according to ET boundary conditions. This is by design.

**ET Derivation (Eq 211):** When D-constraint (divisor) is empty (zero), the P-substrate
(numerator) is unconstrained — it expresses its full range, which is infinite. When both
are zero (0/0), the indeterminate form is resolved to the ET ground state (0), not infinity.

---

## GAP 21: COMPLETE `hardware_access` COMMAND STRING REFERENCE — ABSENT

The guide shows a partial list. The complete set of recognized `hardware_access` command strings:

```pdt
// Platform and system:
P platform = hardware_access ∘ "platform"        // "linux", "win32", "darwin"
P pid      = hardware_access ∘ "getpid"          // current process ID

// Console I/O:
P line  = hardware_access ∘ "readline"           // read line from stdin
T flush = hardware_access ∘ "flush"              // flush stdout

// File system:
P cwd      = hardware_access ∘ "getcwd"          // current working directory
P exists   = hardware_access ∘ "path_exists" ∘ path     // 1/0
P joined   = hardware_access ∘ "path_join" ∘ path       // join path components
P abs_path = hardware_access ∘ "path_abspath" ∘ path    // absolute path
P dir_name = hardware_access ∘ "path_dirname" ∘ path    // directory component
P listing  = hardware_access ∘ "listdir" ∘ path         // directory listing
P is_file  = hardware_access ∘ "path_isfile" ∘ path     // 1/0
P env_val  = hardware_access ∘ "environ_get" ∘ "VAR"    // env variable value

// Terminal:
T clear    = hardware_access ∘ "clear"           // clear terminal (ANSI escape)
```

**ET Derivation:** Every `hardware_access` string is a D-descriptor naming a specific
P-substrate of the platform. The hardware_access keyword is the bridge between ETPL's
abstract P∘D∘T and the concrete syscall ABI.

---

## GAP 22: THE `.etb` ET BINARY FORMAT — ABSENT

When the ETSovereign backend compiles a program (no LLVM available), the output is a
`.etb` (ET Binary) file, not a native executable. This format is undocumented in the guide.

**ETB format structure:**
```
Magic:        b'ETPL' (4 bytes)
Version:      4 bytes little-endian integer (currently 1)
Timestamp:    8 bytes little-endian int64 (Unix seconds)
Source size:  4 bytes little-endian (UTF-8 source byte count)
Payload:      4-byte length prefix + UTF-8 JSON payload
Checksum:     4 bytes little-endian Adler-32 of payload
```

**Running a .etb file:**
```bash
python ETPL.py run myprogram.etb     // interpret the ETB binary
```

**ET Derivation:** The ETB format is the P∘D∘T=E of the compilation process:
- P = source code (infinite potential)
- D = format constraint (magic + version + checksum)
- T = serialization traversal
- E = the grounded binary (the irreducible exception on disk)

---

## GAP 23: TRINARY LOGIC — NEEDS COMPLETE FORMAL TREATMENT

Section 2 and the reference table mention `1`/`0`/`[0/0]` but never fully document
the trinary logic rules. The guide is not a beginner guide if it leaves this informal.

### ET Trinary Logic (complete rules):

| Value | ET Name | ET Primitive | In conditionals | In arithmetic |
|-------|---------|--------------|-----------------|---------------|
| `1` | True / Exception | E (grounded) | branch taken | 1 |
| `0` | False / Incoherent | I (forbidden) | branch NOT taken | 0 |
| `[0/0]` | Indeterminate | T (agency) | undefined until measured | 0 (classical) |

### Trinary operations:
```pdt
// AND in trinary:
P r1 = 1 && 1         // 1
P r2 = 1 && 0         // 0
P r3 = 1 && [0/0]     // [0/0] — indeterminate (classical: first operand = 1, so result = second)

// OR in trinary:
P r4 = 0 || 1         // 1
P r5 = 0 || [0/0]     // [0/0] — indeterminate

// NOT in trinary:
P r6 = !1             // 0
P r7 = !0             // 1
P r8 = ![0/0]         // [0/0] — NOT of indeterminate is still indeterminate
```

### Conditional evaluation with trinary:
```pdt
P x = [0/0] 1 | 0    // indeterminate choice between 1 and 0
if x → sovereign_print ∘ "true path"    // taken if x resolved to 1
```

---

## GAP 24: `range()` FUNCTION AND ITS USE IN LOOP BOUNDS — ABSENT

The `range()` function is available as a built-in and is used in ET source for loop bounds:

```pdt
// Standard range usage:
T loop1 = ∞ (
    sovereign_print ∘ range(5)[_loop_index]     // 0, 1, 2, 3, 4
) (D |range(5)|)

// Range with start:
T loop2 = ∞ (
    P k = range(1, 11)[_loop_index]             // 1 through 10
    sovereign_print ∘ k
) (D |range(1, 11)|)

// Range with step:
T loop3 = ∞ (
    P k = range(0, 10, 2)[_loop_index]          // 0, 2, 4, 6, 8
    sovereign_print ∘ k
) (D |range(0, 10, 2)|)
```

**Critical pattern used throughout ET modules:** `(D |range(n)|)` as the loop bound with
`range(n)[_loop_index]` inside the body is the canonical ET pattern for indexed iteration.

---

## GAP 25: COMPLETE `[0/0]` INDETERMINATE OPERATOR SEMANTICS — INCOMPLETE

The guide covers basic `[0/0]` usage but misses the single-option form and the full
classical-vs-quantum behavioral split:

### Single-option form:
```pdt
P x = [0/0] 42           // valid: indeterminate with one resolution
                          // classical mode: resolves to 42
                          // quantum mode: superposition with single basis state
```

### Classical mode resolution rule:
```pdt
P x = [0/0] "A" | "B" | "C"
// Classical: x = "A" (first option always selected in deterministic execution)
```

### Quantum mode superposition rule:
When compiled with `--target quantum`, `[0/0] a | b` produces a Hadamard superposition:
- `[0/0] 0 | 1` → Hadamard gate `h q[0]` in QASM output
- Measurement collapses to 0 or 1 with equal probability
- `[0/0] 0 | 1 | 2` → requires multi-qubit encoding

### The `[0/0]` as the T-identity:
`[0/0]` is not merely a random choice operator — it IS the mathematical identity of T
(the Traverser's fundamental indeterminacy). Any expression using `[0/0]` is literally
invoking T's irreducible indeterminate nature.

---

## GAP 26: ET COSMOLOGICAL CONSTANTS AS LANGUAGE CONSTANTS — ABSENT

The ET physics mappings are mentioned briefly in Section 2 but their programmatic
representations are never documented:

```pdt
// These are language-accessible constants reflecting ET cosmology:
P dark_energy_fraction  = 0.683     // 68.3% of universe — P-dominant configurations
P dark_matter_fraction  = 0.268     // 26.8% — T-mediated (traverser configurations)
P ordinary_matter_frac  = 0.049     // 4.9% — D-bound (descriptor configurations)

// M-state fractions (ET Mediation states):
P m_vacuum_fraction     = 0.016     // 1.6% M-vacuum (dark energy-like)
P m_matter_fraction     = 0.014     // 1.4% M-matter (ordinary matter-like)
P m_total_fraction      = 0.030     // ~3% total M-states
```

These are not arbitrary — they derive from the P:D:T geometric ratios in the ET manifold.
The KOIDE_RATIO (2/3) and MANIFOLD_SYMMETRY (12) are the roots of these fractions.

---

## GAP 27: ET-DERIVED MATHEMATICAL FORMULAS AS LANGUAGE PATTERNS — INCOMPLETE

Section 8 of the examples shows ET-derived math but the core formulas are not formally
documented as language-level patterns available to all programs. These should be in the guide:

```pdt
// ET Variance (Eq from ET Math): describes the variance of a D-manifold of n states
D et_variance = λ n . (n * n - 1) / 12
// For n=12: et_variance(12) = 143/12 ≈ 11.916...

// ET Density (Eq 211): payload / container with ET zero-guard
D et_density = λ payload, container . {
    if container == 0 → Infinity → E payload / container
}

// ET Binding Depth (counting ∘ operations):
// BD(P) = 0, BD(P∘D) = 1, BD(P∘D∘T) = 2

// Fine structure constant (5-term ET formula):
// α⁻¹ = A₀ + A₁ - A₁·₅ - A₂ - A₃
// Where each Aₙ = BASE_VARIANCE × n × correction_term
// Agreement with CODATA: 0.19 ppb

// ET Koide extension (applies to any generation triplet):
D et_koide = λ m1, m2, m3 . (sqrt(m1) + sqrt(m2) + sqrt(m3))^2 / (3*(m1 + m2 + m3))
// Returns KOIDE_RATIO (2/3) for the three charged lepton masses
```

---

## GAP 28: BINDING PRECEDENCE / AXIOM ORDERING RULE — ABSENT

The mathematical axioms of ET (from Math_of_Exception_Theory.txt) have direct programming
consequences that are not documented:

**Binding Order Rule (Axiom A2, valid from A7):**
- `(P ∘ D) ∘ T` is valid
- `P ∘ (D ∘ T)` is valid (associative)
- `D ∘ P` is **ontologically prohibited** — a Descriptor cannot exist before a Point

**In ETPL code:** You cannot call a Descriptor that references a Point before the Point
is declared in the current scope. The parser enforces this:

```pdt
// INVALID — D references P before P exists:
D compute = λ . x * 2    // x not yet declared
P x = 5                   // too late

// VALID — P declared before D that references it:
P x = 5
D compute = λ . x * 2    // x available in D's closure
```

---

## GAP 29: NAMED TRAVERSER PATTERNS BEYOND BASICS — INCOMPLETE

The guide shows T for loops and basic path declaration but misses several patterns:

### T as a scoped conditional execution:
```pdt
T check = → if condition → result    // Returns result or P if condition false
```

### T as a resource scope:
```pdt
T file_scope = → hardware_access ∘ "path_exists" ∘ "data.txt"
P file_exists = file_scope
```

### T as a named computational step:
```pdt
D process = λ x . {
    T step1 = → x * 2
    T step2 = → step1 + 1
    T step3 = → if step2 > 10 → step2 → E 10
    → step3
}
```

### The `T _entry` pattern (program entry binding):
The canonical pattern to start a self-hosted program is:
```pdt
T _entry = → main()    // The first T-binding that drives the whole program
```

This is the T that completes the P∘D∘T=E equation at the top level: P (all .pdt modules)
∘ D (ETPL_main.pdt assembly) ∘ T (`_entry = → main()`) = E (the running binary).

---

## GAP 30: DESCRIPTOR BODY RETURN VALUE RULE — NEEDS EXPLICIT STATEMENT

The guide says "the last expression is the return value" in multi-statement bodies,
but never explicitly states the complete rule. The full rule is:

1. If a Descriptor body contains `→ value` or `→ E value` at any point, that value is immediately returned.
2. If no explicit `→` is reached, the **last evaluated expression** is the return value.
3. If the body is empty or all paths lead to `P` with no explicit return, the Descriptor returns `P` (null).
4. In a block `{ }`, the last expression before the `}` is implicitly returned if no `→` was hit.

```pdt
D example = λ x . {
    P doubled = x * 2
    doubled                // this is the return — last expression
}

D example2 = λ x . {
    → x * 2               // explicit return — returns immediately
    P unreachable = 99     // never reached
}

D example3 = λ x . {
    if x > 0 → → x        // early return if positive
    → 0 - x               // otherwise return negated
}
```

---

## SUMMARY TABLE: All Gaps and Their Criticality

| # | Gap | Category | Criticality |
|---|-----|----------|-------------|
| 1 | Operator precedence table | Syntax | CRITICAL |
| 2 | Operator alias table (ASCII ↔ Unicode ↔ keyword) | Syntax | CRITICAL |
| 3 | `≈` / `~=` approximate equality | Operators | HIGH |
| 4 | `∘∘` double compose | Operators | MEDIUM |
| 5 | Type system and coercion rules | Semantics | CRITICAL |
| 6 | KOIDE_RATIO typo fix + missing constants | Constants | CRITICAL |
| 7 | Complete math function library | Built-ins | HIGH |
| 8 | `map` and `filter` built-ins | Built-ins | HIGH |
| 9 | Manifold index assignment `arr[n] = v` | Syntax | CRITICAL |
| 10 | Bare `→ value` early return | Semantics | CRITICAL |
| 11 | Full try/attempt exception patterns | Control flow | CRITICAL |
| 12 | Nested loop `_loop_index` scoping | Semantics | CRITICAL |
| 13 | `float("inf")` / `float("nan")` expressions | Syntax | HIGH |
| 14 | `len()` as alternative to `|...|` | Built-ins | MEDIUM |
| 15 | Type conversion functions | Built-ins | HIGH |
| 16 | `isinstance()`, `type()`, `repr()` | Built-ins | HIGH |
| 17 | `set()` built-in | Built-ins | MEDIUM |
| 18 | Keyword aliases `and`/`or`/`not` | Syntax | HIGH |
| 19 | N=144 series bound as language constant | Constants | HIGH |
| 20 | Complete ET division semantics | Semantics | HIGH |
| 21 | Complete `hardware_access` command reference | Built-ins | HIGH |
| 22 | `.etb` ET Binary format | Compilation | MEDIUM |
| 23 | Trinary logic complete formal treatment | Semantics | CRITICAL |
| 24 | `range()` function and loop bound patterns | Built-ins | CRITICAL |
| 25 | Complete `[0/0]` semantics | Semantics | HIGH |
| 26 | ET cosmological constants | Theory | MEDIUM |
| 27 | ET-derived math formulas as language patterns | Math | HIGH |
| 28 | Binding precedence / axiom ordering rule | Theory | HIGH |
| 29 | Named T-traverser patterns + `T _entry` | Semantics | HIGH |
| 30 | Descriptor body return value rule | Semantics | CRITICAL |

---

*All gaps above are about HOW THE ETPL LANGUAGE WORKS — syntax, semantics, operators,
types, built-ins, and constants. Nothing in this document is about the modular
self-hosting architecture. This is the specification the modules will be audited against.*
